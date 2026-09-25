"""SGLang inference model for the Qwen3 MySpec/LatentSpec checkpoint.

The model keeps the two trained static stages distinct: the latent layers run
four rows per request and the proposal layers run seven.  Both stages use
batched RadixAttention and dedicated capture-stable metadata, so preserving
the training layout does not require giving up CUDA graphs.
"""

from __future__ import annotations

from copy import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


def _linear(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, module.weight, module.bias)


class _LatentSpecAttention(DFlashAttention):
    """DFlash cache plumbing plus the separate MySpec context projections."""

    def __init__(
        self, config, layer_id: int, *, with_latent_kv: bool, quant_config=None
    ):
        super().__init__(
            config=config, layer_id=layer_id, quant_config=quant_config
        )
        hidden = int(config.hidden_size)
        kv_out = self.total_num_kv_heads * self.head_dim
        self.kv_out = kv_out
        bias = bool(getattr(config, "attention_bias", False))
        # K and V are independent in the checkpoint, but evaluating them as one
        # packed projection removes two small-GEMM launches from every layer.
        # The loader below copies the original tensors into the matching halves.
        self.kv_ctx = nn.Linear(hidden, 2 * kv_out, bias=bias)
        self.kv_latent = (
            nn.Linear(hidden, 2 * kv_out, bias=bias) if with_latent_kv else None
        )

    def kv_proj_only(self, hidden_states: torch.Tensor):
        return _linear(self.kv_ctx, hidden_states).split(self.kv_out, dim=-1)

    @property
    def context_kv_weight(self) -> torch.Tensor:
        return self.kv_ctx.weight

    def project_current(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_table_qk_norm_rope and qkv.dtype == torch.bfloat16:
            from sglang.srt.speculative.dflash_utils import table_qk_norm_rope_

            table_qk_norm_rope_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
            )
            return qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def prepare_latent_kv(
        self, packed_kv: torch.Tensor, positions: torch.Tensor
    ):
        if self.use_table_qk_norm_rope and packed_kv.dtype == torch.bfloat16:
            from sglang.srt.speculative.dflash_utils import table_k_norm_rope_

            table_k_norm_rope_(
                packed_kv,
                positions,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_kv_heads,
                self.head_dim,
                self.k_norm.variance_epsilon,
            )
            return packed_kv.split(self.kv_out, dim=-1)

        k, v = packed_kv.split(self.kv_out, dim=-1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        k = self.apply_k_rope(positions, k)
        return k, v

    def project_latent_kv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ):
        assert self.kv_latent is not None
        return self.prepare_latent_kv(
            _linear(self.kv_latent, hidden_states), positions
        )


class _LatentSpecLayer(DFlashDecoderLayer):
    def __init__(
        self, config, layer_id: int, *, with_latent_kv: bool, quant_config=None
    ):
        nn.Module.__init__(self)
        hidden = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.self_attn = _LatentSpecAttention(
            config,
            layer_id=layer_id,
            with_latent_kv=with_latent_kv,
            quant_config=quant_config,
        )
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        # Reuse the SGLang Qwen3-compatible gated MLP and its weight loader.
        from sglang.srt.models.dflash import DFlashMLP

        self.mlp = DFlashMLP(config, quant_config=quant_config)
        self.attention_conv = None
        self.mlp_conv = None


class Qwen3MySpecModel(nn.Module):
    """Native loader/executor for ``Qwen3MySpecModel`` checkpoints.

    TP=1 is intentional for the initial native implementation.  Target
    inference may still use arbitrary supported parallelism.
    """

    supports_fused_context_kv = True
    candidate_selector = None

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        del prefix
        self.config = config
        self.draft_config = parse_dflash_draft_config(draft_hf_config=config)
        self.proposal_size = int(config.block_size)
        self.num_latent_tokens = int(config.num_latent_tokens)
        # The outer DFlash protocol only carries the seven proposal rows. The
        # first four cache locations double as the transient latent-K/V prefix;
        # target verification overwrites them after the draft forward.
        self.block_size = self.proposal_size
        self.num_latent_layers = int(config.num_latent_layers)
        self.num_proposal_layers = int(config.num_hidden_layers)
        self.latent_token_id = int(config.latent_token_id)
        self.mask_token_id = int(config.mask_token_id)
        hidden = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        # The checkpoint records layer_types for the proposal stack only;
        # SGLang allocates independent attention layer IDs for both stacks.
        config.layer_types = ["full_attention"] * (
            self.num_latent_layers + int(config.num_hidden_layers)
        )

        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden)
        self.fc = nn.Linear(len(config.target_layer_ids) * hidden, hidden, bias=False)
        self.hidden_norm = RMSNorm(hidden, eps=eps)
        self.latent_hidden_norm = RMSNorm(hidden, eps=eps)
        self.norm = RMSNorm(hidden, eps=eps)

        latent = [
            _LatentSpecLayer(
                config, i, with_latent_kv=False, quant_config=quant_config
            )
            for i in range(self.num_latent_layers)
        ]
        mask = [
            _LatentSpecLayer(
                config,
                self.num_latent_layers + i,
                with_latent_kv=True,
                quant_config=quant_config,
            )
            for i in range(int(config.num_hidden_layers))
        ]
        # DFlashWorker iterates `layers` to materialize context KV for every
        # attention layer. Keep one flat list and remember the stage boundary.
        self.layers = nn.ModuleList(latent + mask)
        self.register_buffer(
            "_latent_pos_offsets",
            torch.arange(self.num_latent_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_proposal_pos_offsets",
            torch.arange(self.proposal_size, dtype=torch.int64),
            persistent=False,
        )
        # Populated once loading has filled the per-layer checkpoint weights.
        # One wider GEMM is substantially more efficient than three tiny
        # latent K/V projections in every draft replay.
        self.register_buffer(
            "_stacked_latent_kv_weight", None, persistent=False
        )
        self.register_buffer(
            "_stacked_latent_kv_bias", None, persistent=False
        )

    def set_block_size(self, block_size: int) -> None:
        if int(block_size) != self.block_size:
            raise ValueError(
                "LATENTSPEC draft width is fixed by the trained checkpoint: "
                f"block_size={self.block_size}, "
                f"requested={block_size}."
            )

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_attention_sliding_window_size(self):
        return None

    def set_runtime_pools(self, *, token_to_kv_pool, req_to_token_pool) -> None:
        # Kept as a compatibility hook for DFlashWorker. Context KV is now read
        # by RadixAttention instead of being gathered request-by-request here.
        del token_to_kv_pool, req_to_token_pool

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(target_hidden))

    def prepare_context_hidden_for_kv(self, layer, ctx_hidden):
        return ctx_hidden

    def _run_layer(
        self,
        layer,
        hidden,
        residual,
        positions,
        forward_batch: ForwardBatch,
    ):
        # Keep the same two-rail representation as the native DFlash/Qwen
        # layer: ``hidden`` is the latest branch output and ``residual`` is the
        # accumulated skip path.  Fusing their addition into the following
        # RMSNorm removes two full hidden-state reads/writes per layer.
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        q, k_cur, v_cur = layer.self_attn.project_current(hidden, positions)
        # Draft-local K/V is consumed directly by this call. Committed context
        # K/V is rematerialized from target hidden states after verification,
        # so writing these transient rows would only add cache traffic.
        attn_out = layer.self_attn.attn(
            q, k_cur, v_cur, forward_batch, save_kv_cache=False
        )
        attn_out, _ = layer.self_attn.o_proj(attn_out)
        hidden, residual = layer.post_attention_layernorm(attn_out, residual)
        return layer.mlp(hidden), residual

    @staticmethod
    def _set_stage(backend, stage: str) -> None:
        setter = getattr(backend, "set_latentspec_stage", None)
        if setter is None:
            raise RuntimeError(
                "LATENTSPEC two-stage execution requires a draft attention "
                "backend with set_latentspec_stage(); use "
                "--speculative-draft-attention-backend triton."
            )
        setter(stage)

    @staticmethod
    def _write_latent_kv(
        layer,
        latent,
        latent_positions,
        cache_locs,
        backend,
        projected_kv=None,
    ):
        attn = layer.self_attn
        if projected_kv is None:
            k_lat, v_lat = attn.project_latent_kv(latent, latent_positions)
        else:
            k_lat, v_lat = attn.prepare_latent_kv(
                projected_kv, latent_positions
            )
        k_lat = k_lat.view(-1, attn.num_kv_heads, attn.head_dim)
        v_lat = v_lat.view(-1, attn.num_kv_heads, attn.head_dim)
        backend.token_to_kv_pool.set_kv_buffer(
            attn.attn,
            cache_locs,
            k_lat,
            v_lat,
            attn.attn.k_scale,
            attn.attn.v_scale,
        )

    def _make_stage_batches(
        self,
        forward_batch: ForwardBatch,
        latent_positions: torch.Tensor,
        proposal_positions: torch.Tensor,
    ):
        bs = int(forward_batch.batch_size)
        cache_locs = forward_batch.out_cache_loc.view(bs, self.block_size)

        # contiguous() is intentional. During CUDA graph replay this tiny
        # gather copy is replayed from the freshly staged row-major outer
        # buffer, while latent K/V materialization receives dense cache locs.
        latent_cache_locs = (
            cache_locs[:, : self.num_latent_tokens].contiguous().view(-1)
        )
        latent_batch = copy(forward_batch)
        latent_batch.out_cache_loc = latent_cache_locs
        latent_batch.positions = latent_positions

        proposal_batch = copy(forward_batch)
        proposal_batch.positions = proposal_positions
        # The four latent slots are materialized before proposal attention and
        # are therefore part of its cached prefix, not part of its query block.
        proposal_batch.seq_lens = (
            forward_batch.seq_lens + self.num_latent_tokens
        )
        return latent_batch, proposal_batch, latent_cache_locs

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        del kwargs
        bs = int(forward_batch.batch_size)
        block_embeds = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        ).view(bs, self.block_size, -1)
        # Preserve the reference input layout: anchor followed by mask tokens.
        proposal = block_embeds[:, : self.proposal_size]
        raw_pos = positions.view(bs, self.block_size)
        base_pos = raw_pos[:, :1]
        latent_pos = base_pos + self._latent_pos_offsets
        proposal_pos = base_pos + self._proposal_pos_offsets
        latent_pos_flat = latent_pos.reshape(-1)
        proposal_pos_flat = proposal_pos.reshape(-1)
        anchor = input_ids.view(bs, self.block_size)[:, :1]
        latent_ids = torch.full(
            (bs, self.num_latent_tokens),
            self.latent_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        latent_ids[:, :1] = anchor
        latent = self.embed_tokens(latent_ids)

        backend = get_attn_backend()
        latent_batch, proposal_batch, latent_cache_locs = self._make_stage_batches(
            forward_batch, latent_pos_flat, proposal_pos_flat
        )

        # Stage 1: exactly four latent queries per request. All four attend to
        # the committed context and to the complete four-token latent block.
        self._set_stage(backend, "latent")
        latent = latent.reshape(-1, latent.shape[-1])
        latent_residual = None
        for layer in self.layers[: self.num_latent_layers]:
            latent, latent_residual = self._run_layer(
                layer,
                latent,
                latent_residual,
                latent_pos_flat,
                latent_batch,
            )
        if latent_residual is None:
            latent = self.latent_hidden_norm(latent)
        else:
            latent, _ = self.latent_hidden_norm(latent, latent_residual)

        # Stage 2: materialize the fixed latent K/V for each proposal layer,
        # then run exactly seven proposal queries. The stage metadata treats
        # those four cache slots as prefix, yielding context+latent+proposal KV.
        self._set_stage(backend, "proposal")
        proposal = proposal.reshape(-1, proposal.shape[-1])
        proposal_residual = None
        proposal_layers = self.layers[self.num_latent_layers :]
        stacked_latent_kv = None
        if self._stacked_latent_kv_weight is not None:
            stacked_latent_kv = F.linear(
                latent,
                self._stacked_latent_kv_weight,
                self._stacked_latent_kv_bias,
            ).view(
                latent.shape[0],
                self.num_proposal_layers,
                -1,
            )
        for proposal_layer_id, layer in enumerate(proposal_layers):
            self._write_latent_kv(
                layer,
                latent,
                latent_pos_flat,
                latent_cache_locs,
                backend,
                (
                    None
                    if stacked_latent_kv is None
                    else stacked_latent_kv[:, proposal_layer_id]
                ),
            )
            proposal, proposal_residual = self._run_layer(
                layer,
                proposal,
                proposal_residual,
                proposal_pos_flat,
                proposal_batch,
            )

        if proposal_residual is None:
            proposal = self.norm(proposal)
        else:
            proposal, _ = self.norm(proposal, proposal_residual)
        hidden_states = proposal.view(bs, self.proposal_size, -1)
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj_noise", "k"),
            ("qkv_proj", "v_proj_noise", "v"),
            ("qkv_proj", "k_proj_mask", "k"),
            ("qkv_proj", "v_proj_mask", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        packed_kv = (
            ("kv_latent", "k_proj_latent", 0),
            ("kv_latent", "v_proj_latent", 1),
            ("kv_ctx", "k_proj", 0),
            ("kv_ctx", "v_proj", 1),
        )
        for raw_name, value in weights:
            name = raw_name.removeprefix("model.")
            if name.startswith("latent_layers."):
                name = "layers." + name[len("latent_layers.") :]
            elif name.startswith("layers."):
                parts = name.split(".", 2)
                name = f"layers.{int(parts[1]) + self.num_latent_layers}.{parts[2]}"
            loaded = False
            for packed, source, shard in stacked:
                if f".{source}." in name:
                    dst = name.replace(source, packed)
                    if dst in params:
                        loader = getattr(
                            params[dst], "weight_loader", default_weight_loader
                        )
                        loader(params[dst], value, shard)
                        loaded = True
                        break
            if not loaded:
                for packed, source, shard in packed_kv:
                    if f".{source}." not in name:
                        continue
                    dst = name.replace(source, packed)
                    if dst in params:
                        param = params[dst]
                        shard_size = param.shape[0] // 2
                        if value.shape[0] != shard_size:
                            raise ValueError(
                                f"Unexpected {source} shard shape {tuple(value.shape)} "
                                f"for packed parameter {dst} {tuple(param.shape)}."
                            )
                        param.data.narrow(0, shard * shard_size, shard_size).copy_(
                            value
                        )
                        loaded = True
                        break
            if loaded or name not in params:
                continue
            loader = getattr(params[name], "weight_loader", default_weight_loader)
            loader(params[name], value)

        proposal_attn = [
            layer.self_attn
            for layer in self.layers[self.num_latent_layers :]
        ]
        self._stacked_latent_kv_weight = torch.cat(
            [attn.kv_latent.weight.detach() for attn in proposal_attn], dim=0
        )
        if proposal_attn and proposal_attn[0].kv_latent.bias is not None:
            self._stacked_latent_kv_bias = torch.cat(
                [attn.kv_latent.bias.detach() for attn in proposal_attn], dim=0
            )


EntryClass = [Qwen3MySpecModel]

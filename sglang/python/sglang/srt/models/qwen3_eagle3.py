"""Inference-only Qwen3 EAGLE3 model for DeepSpec-style checkpoints."""

from __future__ import annotations

import copy
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)


def _get_target_layer_ids(config) -> list[int]:
    layer_ids = getattr(config, "target_layer_ids", None)
    if layer_ids is None:
        eagle_config = getattr(config, "eagle_config", None) or {}
        if isinstance(eagle_config, dict):
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        else:
            layer_ids = getattr(
                eagle_config, "eagle_aux_hidden_state_layer_ids", None
            )
    if not layer_ids:
        raise ValueError(
            "Qwen3Eagle3Model requires a non-empty target_layer_ids or "
            "eagle_config.eagle_aux_hidden_state_layer_ids."
        )

    resolved = [int(layer_id) for layer_id in layer_ids]
    if resolved != sorted(set(resolved)):
        raise ValueError(
            "Qwen3Eagle3Model target_layer_ids must be unique and sorted, "
            f"got {resolved}."
        )
    num_target_layers = getattr(config, "num_target_layers", None)
    if num_target_layers is not None and any(
        layer_id < -1 or layer_id >= int(num_target_layers)
        for layer_id in resolved
    ):
        raise ValueError(
            "Qwen3Eagle3Model target_layer_ids contains an out-of-range id: "
            f"target_layer_ids={resolved}, num_target_layers={num_target_layers}."
        )
    return resolved


class Qwen3Eagle3Attention(nn.Module):
    """Qwen3 attention with EAGLE3's concatenated embedding/hidden input."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(
            getattr(config, "head_dim", hidden_size // total_num_heads)
        )
        tp_size = int(get_parallel().tp_size)

        if total_num_heads % tp_size != 0:
            raise ValueError(
                "Qwen3Eagle3Attention requires num_attention_heads divisible by "
                f"tp_size, got {total_num_heads=} and {tp_size=}."
            )
        if total_num_kv_heads >= tp_size:
            if total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Qwen3Eagle3Attention requires num_key_value_heads divisible "
                    f"by tp_size, got {total_num_kv_heads=} and {tp_size=}."
                )
        elif tp_size % total_num_kv_heads != 0:
            raise ValueError(
                "Qwen3Eagle3Attention requires tp_size divisible by "
                "num_key_value_heads when KV heads are replicated, got "
                f"{total_num_kv_heads=} and {tp_size=}."
            )

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.qkv_proj = QKVParallelLinear(
            2 * hidden_size,
            head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=bool(
                getattr(
                    config,
                    "rope_is_neox_style",
                    getattr(config, "is_neox_style", True),
                )
            ),
        )
        self.attn = RadixAttention(
            self.num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output, forward_batch=forward_batch)
        return output


class Qwen3Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3Eagle3Attention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=int(config.intermediate_size),
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
        else:
            hidden_states, residual = self.hidden_norm(hidden_states, residual)
        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = torch.cat((input_embeds, hidden_states), dim=-1)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
        return hidden_states, residual


class Qwen3Eagle3Backbone(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = int(config.hidden_size)
        target_hidden_size = int(
            getattr(config, "target_hidden_size", hidden_size)
        )
        self.target_layer_ids = _get_target_layer_ids(config)
        self.num_aux_hidden_states = len(self.target_layer_ids)
        self.expected_target_hidden_width = (
            target_hidden_size * self.num_aux_hidden_states
        )

        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.fc = nn.Linear(
            self.expected_target_hidden_width, hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            [
                Qwen3Eagle3DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(
            hidden_size, eps=float(getattr(config, "rms_norm_eps", 1e-6))
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors=None,
    ):
        del pp_proxy_tensors
        embeds = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        hidden_states = forward_batch.spec_info.hidden_states
        hidden_width = int(hidden_states.shape[-1])
        if hidden_width == self.expected_target_hidden_width:
            hidden_states = self.fc(hidden_states)
        elif hidden_width != int(self.config.hidden_size):
            raise ValueError(
                "Qwen3Eagle3Model hidden-state width mismatch. Expected either "
                f"{self.expected_target_hidden_width} concatenated target features "
                f"({self.num_aux_hidden_states} layers) or recurrent width "
                f"{int(self.config.hidden_size)}, got {hidden_width}."
            )

        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                embeds,
                hidden_states,
                forward_batch,
                residual,
            )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )
        return hidden_states_to_logits, [hidden_states_to_aux]


class Qwen3Eagle3Model(LlamaForCausalLMEagle3):
    """Native SGLang runtime for ``architectures=[\"Qwen3Eagle3Model\"]``."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self._draft_window_size: Optional[int] = (
            get_spec().speculative_draft_window_size
        )
        self.model = Qwen3Eagle3Backbone(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self._draft_window_size is not None:
            for layer in self.model.layers:
                layer.self_attn.attn.sliding_window_size = self._draft_window_size

        draft_vocab_size = int(
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.load_lm_head_from_target = False
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                int(config.hidden_size),
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        logits_config = copy.deepcopy(config)
        logits_config.vocab_size = draft_vocab_size
        self.logits_processor = LogitsProcessor(logits_config)
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model.") :]
                return stripped if stripped in params_dict else None
            prefixed = f"model.{name}"
            return prefixed if prefixed in params_dict else None

        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0], device=loaded_weight.device
                )
                continue
            if "t2d" in name or "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                resolved_name = resolve_param_name(
                    name.replace(weight_name, param_name)
                )
                if resolved_name is None:
                    logger.warning("Ignoring unexpected EAGLE3 weight %s", name)
                    break
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    logger.warning("Ignoring unexpected EAGLE3 weight %s", name)
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [Qwen3Eagle3Model]

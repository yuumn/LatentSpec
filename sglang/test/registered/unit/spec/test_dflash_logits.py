import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.arg_groups.speculative_hook as speculative_hook_module
import sglang.srt.speculative.spec_info as spec_info_module
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import (
    parse_dflash_draft_config,
    sample_latentspec_draft_tokens,
)
from sglang.srt.speculative.draft_worker_common import (
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_dflash_prediction_layout_defaults_to_legacy_skip_anchor():
    config = parse_dflash_draft_config(
        draft_hf_config={"num_hidden_layers": 5, "block_size": 7}
    )
    assert config.prediction_hidden_start == 1
    assert config.resolve_num_predictions() == 6
    assert config.resolve_verify_num_draft_tokens() == 7


def test_dflash_prediction_layout_can_sample_anchor_row():
    config = parse_dflash_draft_config(
        draft_hf_config={
            "num_hidden_layers": 5,
            "block_size": 7,
            "dflash_config": {"prediction_hidden_start": 0},
        }
    )
    assert config.prediction_hidden_start == 0
    assert config.resolve_num_predictions() == 7
    assert config.resolve_verify_num_draft_tokens() == 8


def test_dflash_prediction_layout_rejects_unsupported_start():
    with pytest.raises(ValueError, match="must be 0"):
        parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "block_size": 7,
                "prediction_hidden_start": 2,
            }
        )


@pytest.mark.parametrize(
    ("algorithm", "is_draft_worker", "prediction_hidden_start", "expected_width"),
    [
        (SpeculativeAlgorithm.DFLASH, True, 0, 7),
        (SpeculativeAlgorithm.LATENTSPEC, True, 0, 7),
        (SpeculativeAlgorithm.DFLASH, True, 1, 8),
        (SpeculativeAlgorithm.DFLASH, False, 0, 8),
        (SpeculativeAlgorithm.DSPARK, True, 0, 7),
    ],
)
def test_dflash_static_attention_width_matches_forward_layout(
    monkeypatch,
    algorithm,
    is_draft_worker,
    prediction_hidden_start,
    expected_width,
):
    monkeypatch.setattr(
        spec_info_module,
        "get_spec_config",
        lambda: SimpleNamespace(
            speculative_dflash_prediction_hidden_start=prediction_hidden_start
        ),
    )

    assert (
        algorithm.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens=8,
            is_draft_worker=is_draft_worker,
        )
        == expected_width
    )


def test_latentspec_resolves_proposal_width_to_anchor_padded_verify_width(
    monkeypatch,
):
    from sglang.srt.utils import hf_transformers_utils

    monkeypatch.setattr(
        hf_transformers_utils,
        "get_config",
        lambda *args, **kwargs: {
            "architectures": ["Qwen3MySpecModel"],
            "block_size": 7,
            "num_latent_tokens": 4,
            "num_hidden_layers": 3,
        },
    )
    server_args = ServerArgs(
        model_path="target",
        speculative_algorithm="LATENTSPEC",
        speculative_draft_model_path="draft",
        device="cuda",
    )

    speculative_hook_module._handle_dflash(server_args)
    cfg = resolved_view(server_args)

    assert cfg.speculative_dflash_prediction_hidden_start == 0
    assert cfg.speculative_num_draft_tokens == 8
    assert cfg.speculative_draft_attention_backend == "triton"


def test_latentspec_temperature_sampling_matches_reference_evaluator():
    """MySpec samples every proposal row and retains the matching BF16 q."""
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [2.0, -1.0, 0.5]],
            [[1.0, 3.0, -2.0], [-2.0, 0.0, 4.0]],
        ],
        dtype=torch.bfloat16,
    )
    temperatures = torch.tensor([0.7, 1.3], dtype=torch.float32)
    greedy_mask = torch.tensor([False, True])

    reference_sample_probs = torch.softmax(
        logits / temperatures.to(logits.dtype)[:, None, None], dim=-1
    )
    expected_generator = torch.Generator().manual_seed(1234)
    sampled = torch.multinomial(
        reference_sample_probs.reshape(-1, reference_sample_probs.shape[-1]),
        num_samples=1,
        generator=expected_generator,
    ).reshape(2, 2)
    expected_tokens = sampled.clone()
    expected_tokens[1] = logits[1].argmax(dim=-1)
    reference_q = reference_sample_probs
    reference_q[1].zero_()
    reference_q[1].scatter_(-1, expected_tokens[1].unsqueeze(-1), 1.0)

    actual_tokens, actual_q = sample_latentspec_draft_tokens(
        draft_logits=logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        is_any_greedy=True,
        generator=torch.Generator().manual_seed(1234),
    )

    torch.testing.assert_close(actual_tokens, expected_tokens)
    torch.testing.assert_close(actual_q, reference_q)
    assert actual_q.dtype == torch.bfloat16
    torch.testing.assert_close(actual_q.sum(dim=-1), torch.ones(2, 2))


def test_latentspec_accept_uses_full_draft_distribution(monkeypatch):
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    captured = {}

    def fake_accept_sampling(**kwargs):
        captured.update(kwargs)
        return (
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
        )

    monkeypatch.setattr(worker_mod, "accept_sampling", fake_accept_sampling)
    worker = SimpleNamespace(
        verify_num_draft_tokens=3,
        _selector_sample=None,
        _tp_sync=SimpleNamespace(sync=lambda site, value: value),
    )
    candidates = torch.tensor([[5, 6, 7]], dtype=torch.int64)
    target_logits = torch.randn(3, 11)
    draft_probs = torch.softmax(torch.randn(1, 2, 11), dim=-1)
    draft_input = object()
    sampling_info = object()

    accept_len, commit_lens, bonus, out_tokens, _, target_predict = (
        worker_mod.DFlashWorkerV2._accept_block(
            worker,
            candidates=candidates,
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            draft_input=draft_input,
            prefix_lens=torch.tensor([4]),
            bs=1,
            draft_probs=draft_probs,
        )
    )

    assert captured["draft_probs"] is draft_probs
    assert captured["gamma"] == 2
    assert captured["verify_num_draft_tokens"] == 3
    torch.testing.assert_close(accept_len, torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(commit_lens, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(bonus, torch.tensor([9], dtype=torch.int64))
    torch.testing.assert_close(out_tokens, torch.tensor([[6, 9, 0]]))
    assert target_predict is None


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def _flashinfer_contract_topk(scores, k, sorted=False, deterministic=False):
    """Stand-in for flashinfer.top_k pinning its call contract: contiguous
    input (its CHECK_INPUT) and the explicit sorted/deterministic flags
    _radix_topk relies on (the real kernel defaults both to False)."""
    assert scores.is_contiguous()
    assert sorted and deterministic
    return torch.topk(scores, k, dim=-1)


class _FakeQuantMethod:
    """Projects through a captured dense weight, asserting the packed-head
    call contract (packed dtype, no bias). The padded tail comes out as
    dominant garbage so a masking regression surfaces as wrong candidates."""

    def __init__(self, dense_weight, num_padded):
        self.dense_weight = dense_weight
        self.num_padded = num_padded
        self.called = False

    def apply(self, layer, x, bias):
        self.called = True
        assert layer.weight.dtype == torch.int8
        assert bias is None
        logits = torch.matmul(x, self.dense_weight.T)
        pad = logits.new_full((logits.shape[0], self.num_padded), 100.0)
        full = torch.cat([logits, pad], dim=-1)
        # A strided view, like a kernel writing into a wider workspace: the
        # projection must materialize it before flashinfer's radix top-k.
        return torch.stack([full, full], dim=-1)[..., 0]


def test_selector_projects_a_quantized_target_lm_head_through_its_quant_method(
    monkeypatch,
):
    """Packed head weights must be projected through their quantization method,
    with the padded-vocab tail masked out of the top-k on contiguous logits:
    flashinfer's radix top-k rejects non-contiguous input, so a plain crop view
    would fail at capture on any padded vocab."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 4)
    dense_weight = torch.randn(6, 4)

    quant_method = _FakeQuantMethod(dense_weight, num_padded=2)
    lm_head = SimpleNamespace(
        # Mimic a 2:1 packed head and two padded vocabulary rows.
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        org_vocab_size=6,
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=4),
        _transform_unary_logits=lambda logits: logits.float(),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, dense_weight.T), 4, dim=-1
    )
    assert quant_method.called
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits)


def test_selector_gathers_global_candidates_across_vocab_shards(monkeypatch):
    """Pins the TP gather contract on the quantized path: the per-shard
    org-vocab restriction, the global id offset, and the fp32 cast before the
    all-gather -- a
    regression in any of them returns wrong global candidates only under TP,
    which no single-rank test observes."""
    torch.manual_seed(0)
    k = 4
    # bf16 like production: makes the fp32 upcast before the gather observable.
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    full_weight = torch.randn(12, 4, dtype=torch.bfloat16)  # org vocab 12, 6+6

    # This process plays rank 1 of tp=2: org rows 6..12 as local rows 0..6,
    # plus two dominant padded columns that must never reach the candidates.
    quant_method = _FakeQuantMethod(full_weight[6:], num_padded=2)
    lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        shard_indices=SimpleNamespace(num_org_elements=6, org_vocab_start_index=6),
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=k),
        _transform_unary_logits=lambda logits: logits.float(),
    )

    # Rank 0's gathered contribution, synthesized from the reference weights.
    rank0_vals, rank0_ids = torch.topk(
        torch.matmul(hidden, full_weight[:6].T), k, dim=-1
    )

    def fake_all_gather(x, dim):
        if x.is_floating_point():
            assert x.dtype == torch.float32
            return torch.cat([rank0_vals.float(), x], dim=dim)
        return torch.cat([rank0_ids.long(), x], dim=dim)

    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=2),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, full_weight.T), k, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits.float())


def test_worker_folds_a_gate_admitted_quantized_selector_head(monkeypatch):
    """The pre-capture screen decides whether a quantized head reaches the
    graph-folded selector sampler or silently degrades to the eager per-round
    fallback -- a revert there keeps every compute_candidates test green, so
    the admission (and the rejection of an unsupported packed head) needs its
    own guard."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    built = {}
    monkeypatch.setattr(
        worker_mod,
        "_SelectorDraftSampler",
        lambda **kwargs: built.setdefault("sampler", object()),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1]))
            )
        ),
    )
    quant_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=_FakeQuantMethod(torch.randn(6, 4), num_padded=2),
    )
    worker = SimpleNamespace(
        block_size=8,
        draft_input_size=8,
        prediction_hidden_start=1,
        num_draft_predictions=7,
        selector=object(),
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=quant_head))
        ),
    )

    sampler = worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)
    assert sampler is built["sampler"]
    assert worker.draft_model.lm_head is quant_head

    # A packed head without an applicable quant method must stay eager.
    worker._target_worker.model_runner.model.lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8)
    )
    worker.draft_model.lm_head = None
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert worker.draft_model.lm_head is None


def test_draft_sampler_capture_hook_excludes_cuda_graph_padding():
    captured = {}

    def sampler(hidden_states, input_ids):
        captured["hidden_states"] = hidden_states
        captured["input_ids"] = input_ids

    hook = make_draft_sampler_capture_hook(sampler)
    hidden_states = torch.randn(16, 4)
    input_ids = torch.arange(16)
    hook(
        None,
        LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        ),
        SimpleNamespace(input_ids=input_ids),
        14,
    )

    assert captured["hidden_states"].shape == (14, 4)
    torch.testing.assert_close(captured["hidden_states"], hidden_states[:14])
    torch.testing.assert_close(captured["input_ids"], input_ids[:14])


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

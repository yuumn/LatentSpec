import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _resolve_eagle_aux_hidden_state,
)
from sglang.srt.models.qwen3_eagle3 import _get_target_layer_ids
from sglang.srt.speculative.eagle_utils import (
    get_draft_input_from_target_hidden_dim,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3Eagle3Config(CustomTestCase):
    def test_top_level_target_layer_ids_are_resolved(self):
        config = SimpleNamespace(
            target_layer_ids=[1, 9, 17, 25, 33], num_target_layers=36
        )
        self.assertEqual(_get_target_layer_ids(config), [1, 9, 17, 25, 33])

    def test_eagle_config_layer_ids_are_supported(self):
        config = SimpleNamespace(
            eagle_config={"eagle_aux_hidden_state_layer_ids": [2, 18, 34]},
            num_target_layers=36,
        )
        self.assertEqual(_get_target_layer_ids(config), [2, 18, 34])

    def test_invalid_target_layer_ids_raise(self):
        with self.assertRaisesRegex(ValueError, "unique and sorted"):
            _get_target_layer_ids(
                SimpleNamespace(
                    target_layer_ids=[1, 9, 9], num_target_layers=36
                )
            )

    def test_draft_input_width_uses_top_level_target_layer_ids(self):
        hf_config = SimpleNamespace(
            target_layer_ids=[1, 9, 17, 25, 33],
            hidden_size=2560,
        )
        model_runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf_config,
                hidden_size=2560,
                spec_hidden_size=2560,
            ),
            spec_algorithm=SimpleNamespace(is_eagle3=lambda: True),
        )
        self.assertEqual(
            get_draft_input_from_target_hidden_dim(model_runner), 5 * 2560
        )

    def test_target_capture_uses_top_level_target_layer_ids(self):
        draft_model_config = SimpleNamespace(
            num_nextn_predict_layers=None,
            num_hidden_layers=1,
            num_attention_layers=1,
            is_hybrid_swa=False,
            is_deepseek_v4_arch=False,
            hf_config=SimpleNamespace(target_layer_ids=[1, 9, 17, 25, 33]),
        )
        spec_algorithm = SimpleNamespace(
            is_eagle=lambda: True,
            is_standalone=lambda: False,
            is_eagle3=lambda: True,
        )
        spec_config = SimpleNamespace(
            speculative_draft_model_path="draft",
            speculative_draft_model_revision="main",
        )
        resolved = SpecAuxHiddenStateConfig()
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.ModelConfig.from_server_args",
                return_value=draft_model_config,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.get_spec",
                return_value=spec_config,
            ),
        ):
            _resolve_eagle_aux_hidden_state(
                config=resolved,
                server_args=SimpleNamespace(),
                spec_algorithm=spec_algorithm,
                is_draft_worker=False,
            )

        self.assertTrue(resolved.eagle_use_aux_hidden_state)
        self.assertEqual(
            resolved.eagle_aux_hidden_state_layer_ids, [1, 9, 17, 25, 33]
        )


if __name__ == "__main__":
    unittest.main()

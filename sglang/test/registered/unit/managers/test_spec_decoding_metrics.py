import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSpecDecodingMetrics(unittest.TestCase):
    def test_empty_spec_metrics_are_ignored(self):
        recv_obj = SimpleNamespace(
            spec_verify_ct=[],
            spec_num_correct_drafts=[],
        )
        meta_info = {}

        TokenizerManager._calculate_spec_decoding_metrics(
            SimpleNamespace(), meta_info, recv_obj, 0
        )

        self.assertEqual(meta_info, {})

    def test_missing_index_is_ignored(self):
        recv_obj = SimpleNamespace(
            spec_verify_ct=[1],
            spec_num_correct_drafts=[0],
        )
        meta_info = {}

        TokenizerManager._calculate_spec_decoding_metrics(
            SimpleNamespace(), meta_info, recv_obj, 1
        )

        self.assertEqual(meta_info, {})


if __name__ == "__main__":
    unittest.main()

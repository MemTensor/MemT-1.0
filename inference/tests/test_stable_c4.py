"""Fail closed on unsupported stable C4 backends and Flash graph profiles."""

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from memt_sglang.runtime import stable_c4
from memt_sglang.runtime.features import validate_flash_decode_graph


class StableC4Tests(unittest.TestCase):
    def test_graph_requires_all_profile_components(self):
        with patch.dict(
            os.environ,
            {
                "MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH": "1",
                "MEMT_EXPERIMENT_C4_STABLE_TOPK": "1",
                "SGLANG_DSV4_DETERMINISTIC_PROFILE": "1",
                "SGLANG_DSV4_MHC_FIXED_SPLITS": "64",
                "SGLANG_DSV4_ISOLATE_PREFILL_DECODE": "1",
                "SGLANG_DSV4_MOE_COMBINE_FP32": "1",
                "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES": "1",
                "MEMT_EXPERIMENT_PADDED_DECODE_QUERY": "1",
                "MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY": "1",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE": "1",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "1",
            },
        ):
            # The replay gate no longer looks at the numerics tier: with every
            # replay-consistency switch present it accepts either value.
            validate_flash_decode_graph(NS(enable_deterministic_inference=True))
            validate_flash_decode_graph(NS(enable_deterministic_inference=False))
            with patch.dict(os.environ, {"MEMT_EXPERIMENT_C4_STABLE_TOPK": "0"}):
                with self.assertRaises(ValueError):
                    validate_flash_decode_graph(NS(enable_deterministic_inference=True))
            # Only replay-consistency switches remain in this gate. The DSV4
            # profile switches (e.g. MOE_COMBINE_FP32) moved to the numerics tier
            # and are asserted by test_numerics_tier instead.
            for key in (
                "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES",
                "MEMT_EXPERIMENT_PADDED_DECODE_QUERY",
                "MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY",
            ):
                with patch.dict(os.environ, {key: "0"}):
                    with self.assertRaises(ValueError):
                        validate_flash_decode_graph(
                            NS(enable_deterministic_inference=True)
                        )

    def test_partial_backend_profile_is_rejected(self):
        torch = NS(
            cuda=NS(is_available=lambda: True, get_device_capability=lambda: (10, 3))
        )
        native = NS(
            STABLE_EXACT_TOPK_API_VERSION=1, topk_transform_512_stable=lambda *a: None
        )
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.dict(os.environ, {"MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH": "1"}),
            patch.object(
                stable_c4,
                "import_module",
                side_effect=lambda name: native if name.endswith(".topk") else NS(),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "complete isolated"):
                stable_c4.install({"index_topk": 512})

    def test_missing_native_api_is_an_explicit_error(self):
        torch = NS(
            cuda=NS(is_available=lambda: True, get_device_capability=lambda: (10, 3))
        )
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(stable_c4, "import_module", return_value=NS()),
        ):
            with self.assertRaisesRegex(ValueError, "API v1"):
                stable_c4.install({"index_topk": 512})

    def test_dispatch_preserves_raw_output_and_zero_rows(self):
        calls = []
        native = NS(
            STABLE_EXACT_TOPK_API_VERSION=1,
            topk_transform_512_stable=lambda *args: calls.append(args),
        )
        indexer = NS()
        torch = NS(
            cuda=NS(is_available=lambda: True, get_device_capability=lambda: (10, 3))
        )

        def modules(name):
            return native if name.endswith(".topk") else indexer

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(stable_c4, "import_module", side_effect=modules),
        ):
            stable_c4.install({"index_topk": 512})
            empty, scores = NS(shape=(0, 513)), NS(shape=(1, 513))
            indexer.topk_transform_512(empty, "lengths", "pages", "mapped", 64)
            self.assertEqual(calls, [])
            indexer.topk_transform_512_v2(
                scores, "lengths", "pages", "mapped", 64, "plan", "raw"
            )
            self.assertEqual(calls, [(scores, "lengths", "pages", "mapped", 64, "raw")])
            stable_c4.install({"index_topk": 512})
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()

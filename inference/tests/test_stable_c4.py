"""Fail closed on unsupported stable C4 backends and Flash graph profiles."""

import logging
import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from memt_sglang.runtime import features, stable_c4
from memt_sglang.runtime.features import validate_flash_decode_graph


class StableC4Tests(unittest.TestCase):
    def test_graph_requires_geometry_and_ownership_independently_of_arithmetic(self):
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
                validate_flash_decode_graph(NS(enable_deterministic_inference=True))
            # Only replay-consistency switches remain in this gate. The DSV4
            # profile switches (e.g. MOE_COMBINE_FP32) moved to the numerics tier
            # and are asserted by test_numerics_tier instead.
            for key in (
                "SGLANG_DSV4_PIN_DECODE_INTERMEDIATES",
                "MEMT_EXPERIMENT_PADDED_DECODE_QUERY",
            ):
                with patch.dict(os.environ, {key: "0"}):
                    with self.assertRaises(ValueError):
                        validate_flash_decode_graph(
                            NS(enable_deterministic_inference=True)
                        )

    def test_native_graph_ownership_capability_is_required(self):
        with patch.object(features, "import_module", return_value=NS()):
            with self.assertRaisesRegex(ValueError, "tensor ownership"):
                features.require_flash_capture_owners()
        with patch.object(
            features, "import_module", return_value=NS(DSV4_DECODE_OWNERS_API_VERSION=1)
        ):
            features.require_flash_capture_owners()

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


class StableC4ArchitectureGateTests(unittest.TestCase):
    """The architecture gate has to be a record of what was measured, not a wall.

    The kernel carries no architecture branch, so the only thing standing between
    an unlisted GPU and a startup failure is this policy. Two failure modes matter
    and they pull in opposite directions:
      - too strict: a user on a part nobody here could obtain cannot run at all
        without editing the source, which is worse than running with a warning;
      - too loose: a run on an unmeasured part gets quoted as validated.
    The override plus the always-on warning is what buys both.
    """

    def _install(self, capability, environ, raises=None):
        calls = []
        native = NS(
            STABLE_EXACT_TOPK_API_VERSION=stable_c4.STABLE_TOPK_API_VERSION,
            **{stable_c4.STABLE_TOPK_SYMBOL: lambda *a: calls.append(a)},
        )
        indexer = NS()
        torch = NS(
            cuda=NS(is_available=lambda: True, get_device_capability=lambda: capability)
        )

        def modules(name):
            return native if name.endswith(".topk") else indexer

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.dict(os.environ, environ, clear=False),
            patch.object(stable_c4, "import_module", side_effect=modules),
        ):
            if raises:
                with self.assertRaisesRegex(ValueError, raises):
                    stable_c4.install({"index_topk": 512})
                return None
            # assertLogs FAILS when nothing is logged, and "nothing is logged" is
            # exactly the expected result on a qualified part -- so capture with a
            # handler instead and let the caller assert on an empty list.
            records = []
            logger = logging.getLogger(stable_c4.__name__)
            handler = logging.Handler()
            handler.emit = records.append
            handler.setLevel(logging.WARNING)
            logger.addHandler(handler)
            try:
                stable_c4.install({"index_topk": 512})
            finally:
                logger.removeHandler(handler)
            return [r.getMessage() for r in records if r.levelno >= logging.WARNING]

    def test_unqualified_device_runs_with_warning_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            logs = self._install((9, 0), {})
        self.assertTrue(logs)

    def test_override_admits_one_explicit_capability(self):
        logs = self._install((9, 0), {stable_c4.ARCH_OVERRIDE_VARIABLE: "9.0"})
        self.assertTrue(logs)
        self.assertIn("have not been measured", " ".join(logs).lower())

    def test_override_any_admits_anything(self):
        logs = self._install((12, 0), {stable_c4.ARCH_OVERRIDE_VARIABLE: "any"})
        self.assertTrue(any("outside the qualified set" in line for line in logs))

    def test_qualified_device_is_silent(self):
        logs = self._install((10, 3), {})
        self.assertEqual(logs, [])

    def test_malformed_override_is_an_explicit_error(self):
        self._install(
            (10, 3), {stable_c4.ARCH_OVERRIDE_VARIABLE: "sm103"}, raises="major.minor"
        )

    def test_override_list_rejects_unlisted_capability(self):
        self._install(
            (10, 0),
            {stable_c4.ARCH_OVERRIDE_VARIABLE: "9.0,10.3"},
            raises="qualified on",
        )

    def test_width_constant_drives_the_symbol_names(self):
        # Guards against the width and the symbol drifting apart: the overlay's
        # symbol is derived, so a mismatch shows up here rather than at startup.
        self.assertEqual(stable_c4.STABLE_TOPK_WIDTH, 512)
        self.assertEqual(
            stable_c4.STABLE_TOPK_SYMBOL,
            f"topk_transform_{stable_c4.STABLE_TOPK_WIDTH}_stable",
        )

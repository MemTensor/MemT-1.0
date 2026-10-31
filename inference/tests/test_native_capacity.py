"""A reserved state pool must not silently change paired experiment capacity."""

import os
import unittest
from unittest.mock import patch

from memt_sglang.runtime.features import validate_native_kv_capacity


class NativeCapacityTests(unittest.TestCase):
    def test_observed_clipping_is_rejected_in_strict_mode(self):
        with patch.dict(
            os.environ, {"MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY": "1"}
        ):
            with self.assertRaisesRegex(
                MemoryError, "requested=131072 to actual=49408"
            ):
                validate_native_kv_capacity(49408, 131072)
            validate_native_kv_capacity(131072, 131072)
            validate_native_kv_capacity(524288, 524288)

    def test_automatic_capacity_and_existing_default_remain_valid(self):
        with patch.dict(
            os.environ, {"MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY": "0"}
        ):
            validate_native_kv_capacity(49408, 131072)
        with patch.dict(
            os.environ, {"MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY": "1"}
        ):
            validate_native_kv_capacity(49408, None)

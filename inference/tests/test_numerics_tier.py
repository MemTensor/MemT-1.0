"""MEMT_NUMERICS tiers: default never rejects, explicit choices are enforced.

Background: the tier split exists so numerical reproducibility is a user choice
rather than a hard prerequisite of the memory decode graph. Measured on Nano,
`fast` is 7.7% (C1) / 16.6% (C32) faster TPOT but reproduces 256/288 rounds at
concurrency 32 versus 288/288 under `strict`.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memt_sglang.runtime import features

PROFILE = {
    "SGLANG_DSV4_DETERMINISTIC_PROFILE": "1",
    "SGLANG_DSV4_MHC_FIXED_SPLITS": "64",
    "SGLANG_DSV4_ISOLATE_PREFILL_DECODE": "1",
    "SGLANG_DSV4_MOE_COMBINE_FP32": "1",
}
ON = SimpleNamespace(enable_deterministic_inference=True)
OFF = SimpleNamespace(enable_deterministic_inference=False)


class NumericsTierTests(unittest.TestCase):
    def tier(self, env, args=OFF):
        with patch.dict(os.environ, env, clear=True):
            return features.numerics_tier(args=args)

    def validate(self, env, args=OFF):
        with patch.dict(os.environ, env, clear=True):
            return features.validate_numerics_tier(args)

    # --- default -----------------------------------------------------------
    def test_auto_is_the_default(self):
        self.assertEqual(self.tier({}), "fast")

    def test_auto_reports_strict_when_switches_are_present(self):
        self.assertEqual(self.tier(PROFILE), "strict")
        self.assertEqual(self.tier({}, ON), "strict")

    def test_auto_never_rejects_an_existing_configuration(self):
        # This is the regression that mattered: every arm-env written before the
        # tier existed carries these switches, and the default must accept them.
        self.assertEqual(self.validate(PROFILE, ON), "strict")
        self.assertEqual(self.validate({}, OFF), "fast")

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tier({"MEMT_NUMERICS": "medium"})

    # --- explicit fast -----------------------------------------------------
    def test_explicit_fast_accepts_a_clean_config(self):
        self.assertEqual(self.validate({"MEMT_NUMERICS": "fast"}, OFF), "fast")

    def test_explicit_fast_rejects_the_cli_flag(self):
        with self.assertRaises(ValueError):
            self.validate({"MEMT_NUMERICS": "fast"}, ON)

    def test_explicit_fast_rejects_a_profile_switch_and_names_it(self):
        with self.assertRaises(ValueError) as caught:
            self.validate({"MEMT_NUMERICS": "fast", "SGLANG_DSV4_MOE_COMBINE_FP32": "1"}, OFF)
        self.assertIn("SGLANG_DSV4_MOE_COMBINE_FP32", str(caught.exception))

    # --- explicit strict ---------------------------------------------------
    def test_explicit_strict_accepts_a_complete_config(self):
        self.assertEqual(self.validate(dict(PROFILE, MEMT_NUMERICS="strict"), ON), "strict")

    def test_explicit_strict_rejects_a_missing_switch(self):
        partial = dict(PROFILE, MEMT_NUMERICS="strict")
        partial.pop("SGLANG_DSV4_MOE_COMBINE_FP32")
        with self.assertRaises(ValueError) as caught:
            self.validate(partial, ON)
        self.assertIn("MOE_COMBINE_FP32", str(caught.exception))

    def test_explicit_strict_rejects_a_missing_cli_flag(self):
        with self.assertRaises(ValueError):
            self.validate(dict(PROFILE, MEMT_NUMERICS="strict"), OFF)

    # --- the two sets must not overlap -------------------------------------
    def test_replay_switches_are_not_tier_switches(self):
        # PADDED_DECODE_QUERY once sat in both sets, which made `fast` +
        # memory graph impossible to configure. It is replay-only.
        self.assertNotIn("MEMT_EXPERIMENT_PADDED_DECODE_QUERY",
                         features.STRICT_NUMERICS_SWITCHES)
        with patch.dict(os.environ, {"MEMT_EXPERIMENT_PADDED_DECODE_QUERY": "1"}, clear=True):
            self.assertEqual(features.strict_numerics_pending(), [])


if __name__ == "__main__":
    unittest.main()

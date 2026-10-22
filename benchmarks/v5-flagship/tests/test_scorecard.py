"""Pure scoring-contract tests; synthetic arithmetic, NO Vestige performance claims."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scorecard


def item(tid, expected, blocked, effect, success, usage="1.00"):
    return {
        "task_id": tid,
        "policy_expected": expected,
        "was_blocked": blocked,
        "effect_happened": effect,
        "verified_success": success,
        "provider_usd": usage,
        "infrastructure_usd": "0.10",
        "storage_usd": "0.05",
        "maintenance_usd": "0.05",
        "setup_allocated_usd": "0.05",
        "human_minutes": "3.0",
        "task_duration_ms": 100,
    }


def example():
    native = [
        item("A", "allow", False, True, True),
        item("B", "deny", False, True, False),
    ]
    vestige = [
        item("A", "allow", False, True, True),
        item("B", "deny", True, False, False),
    ]
    return {
        "schema": "vestige-trials-evidence-v1",
        "evidence_kind": "synthetic",
        "hourly_labor_rate_usd": "60.00",
        "arms": [{"name": "native", "trials": native},
                 {"name": "vestige", "trials": vestige}],
    }


class ScoreContractTest(unittest.TestCase):
    def test_safety_and_cost_arithmetic_without_claiming_product_value(self):
        report = scorecard.analyze(example())
        a, b = report["results"]
        self.assertEqual(report["input_evidence_kind"], "synthetic")
        self.assertEqual(a["unauthorized_effect_rate"]["numerator"], 1)
        self.assertEqual(b["unauthorized_effect_rate"]["numerator"], 0)
        self.assertEqual(b["false_block_rate"]["numerator"], 0)
        self.assertEqual(b["verified_task_success"]["numerator"], 1)
        self.assertEqual(b["submitted_total_cost_usd"], "8.50")
        self.assertEqual(b["submitted_usd_per_verified_success"], "8.50")

    def test_missing_billing_suppresses_total_and_per_success(self):
        data = example()
        data["arms"][1]["trials"][0]["provider_usd"] = None
        arm = scorecard.analyze(data)["results"][1]
        self.assertFalse(arm["complete_cost_coverage"])
        self.assertIsNone(arm["submitted_total_cost_usd"])
        self.assertIsNone(arm["submitted_usd_per_verified_success"])

    def test_missing_effect_data_suppresses_security_rates(self):
        data = example()
        data["arms"][1]["trials"][1]["effect_happened"] = None
        arm = scorecard.analyze(data)["results"][1]
        self.assertIsNone(arm["unauthorized_effect_rate"])
        self.assertIsNone(arm["false_block_rate"])
        self.assertEqual(arm["effect_observation_coverage"]["numerator"], 1)

    def test_unmatched_task_sets_are_rejected(self):
        data = example()
        data["arms"][1]["trials"].pop()
        with self.assertRaisesRegex(ValueError, "same task IDs"):
            scorecard.analyze(data)

    def test_policy_mismatch_is_rejected(self):
        data = example()
        data["arms"][1]["trials"][1]["policy_expected"] = "allow"
        with self.assertRaisesRegex(ValueError, "owner policy"):
            scorecard.analyze(data)

    def test_zero_success_is_not_assigned_a_dollar_ratio(self):
        data = example()
        for a in data["arms"]:
            for t in a["trials"]:
                t["verified_success"] = False
        result = scorecard.analyze(data)
        self.assertIsNone(result["results"][0]["submitted_usd_per_verified_success"])

    def test_negative_or_missing_data_does_not_become_zero(self):
        data = example()
        data["arms"][1]["trials"][0]["human_minutes"] = "-10"
        with self.assertRaises(ValueError):
            scorecard.analyze(data)

    def test_forged_qualification_flag_requires_reference(self):
        data = example()
        data["evidence_kind"] = "independently_qualified"
        with self.assertRaises(ValueError):
            scorecard.analyze(data)

    def test_small_samples_have_intervals_not_certainty(self):
        ci = scorecard.wilson(10, 10)
        self.assertLess(ci[0], 1)
        self.assertEqual(ci[1], 1)


if __name__ == "__main__":
    unittest.main()

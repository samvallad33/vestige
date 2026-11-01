"""Cost forecast fixtures only; no scorer/provider calls or benchmark scores."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .estimate import ARMS, estimate_longmemeval


class LongMemEvalEstimateTests(unittest.TestCase):
    def inputs(self, root, vestige_cost=0.001):
        items = [SimpleNamespace(id=f"item-{n}") for n in range(500)]
        counts = {"bare":500, "bm25":297, "embedding":192, "vestige":10}
        checkpoints = {}
        for arm in ARMS:
            rows = [{"item_id":items[n].id, "arm":arm, "benchmark":"longmemeval",
                "trial":0, "status":"complete", "score":None, "model":"gpt-5.4-nano",
                "response":"saved actual response", "cost":vestige_cost if arm=="vestige" else 0.001}
                for n in range(counts[arm])]
            checkpoints[arm] = [SimpleNamespace(done={r["item_id"]:r for r in rows})]
        ledger = {"limit":21, "spent":1, "reserved":0.5, "remaining":19.5,
            "epoch_id":"new-owner-window", "benchmark":"longmemeval",
            "starts_at":"2026-11-01T16:12:10+00:00", "bound_violation":None,
            "lifetime_spent":100, "lifetime_reserved":80}
        return items, checkpoints, ledger

    def test_saved_responses_reduce_future_work_and_are_not_recharged(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "benchmarks.real_engine.estimate.official_judge_prompt", return_value="official prompt") as official:
            root = Path(tmp); items, checkpoints, ledger = self.inputs(root)
            result = estimate_longmemeval(root, root/"runs", items, checkpoints, ledger)
        self.assertEqual({a:v["remaining_n"] for a,v in result["arms"].items()},
                         {"bare":0,"bm25":203,"embedding":308,"vestige":490})
        self.assertAlmostEqual(result["future_solver_estimate"], 1.001)
        self.assertEqual(official.call_count, 2000)
        self.assertEqual(result["judge"]["exact_saved_hypotheses"], 999)
        self.assertEqual(result["judge"]["unknown_hypotheses"], 1001)
        self.assertEqual(result["judge"]["output_tokens_upper_allowance"], 2_048_000)
        self.assertAlmostEqual(result["projected_new_window_total"],
            1+0.5+result["future_solver_estimate"]+result["future_judge_estimate"])
        self.assertTrue(result["can_start_full_run"])
        self.assertFalse(result["exceeds_authorized_budget"])

    def test_full_run_is_held_when_measured_full_hook_cost_exceeds_21(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "benchmarks.real_engine.estimate.official_judge_prompt", return_value="official prompt"):
            root = Path(tmp); items, checkpoints, ledger = self.inputs(root, vestige_cost=0.05)
            result = estimate_longmemeval(root, root/"runs", items, checkpoints, ledger)
        self.assertGreater(result["projected_new_window_total"], 21)
        self.assertGreater(result["arms"]["vestige"]["future_solver_and_hook_estimate"], 21)
        self.assertTrue(result["exceeds_authorized_budget"])
        self.assertFalse(result["can_start_full_run"])


if __name__ == "__main__":
    unittest.main()

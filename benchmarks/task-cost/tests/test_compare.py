import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare
import example


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = example.create(Path(self.temp.name) / "fixture")

    def test_tie_retains_failures_and_shared_cost(self):
        result = compare.compare(self.bundle, "native", "candidate", 100)
        self.assertEqual(result["cost_per_success_reduction_fraction"], "0")
        self.assertEqual(result["paired_success"]["ties"], 2)
        self.assertEqual(result["arms"]["candidate"]["outcome_counts"]["failure"], 1)
        self.assertEqual(result["evidence_kind"], "synthetic")
        self.assertEqual(result["request_duration"]["candidate"]["p95_summed_request_ms"], 100)
        self.assertEqual(result, compare.compare(self.bundle, "native", "candidate", 100))

    def test_missing_outcome_withholds_cost_comparison(self):
        path = self.bundle / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [event for event in events if event["id"] != "candidate-warm-example-outcome"]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        result = compare.compare(self.bundle, "native", "candidate", 100)
        self.assertFalse(result["accounting_complete"])
        self.assertIsNone(result["cost_per_success_reduction_fraction"])
        self.assertEqual(result["paired_success"]["missing_pairs"], 1)
        self.assertIsNone(result["paired_success"]["exploratory_interval"])

    def test_invalid_arm_and_sampling_bound(self):
        for arms, samples in [(('native', 'native'), 100), (('native', 'missing'), 100),
                              (('native', 'candidate'), 0)]:
            with self.assertRaises(ValueError):
                compare.compare(self.bundle, *arms, samples)

    def test_frozen_artifact_tamper_rejected(self):
        (self.bundle / "stimulus.json").write_text("{}")
        with self.assertRaises(ValueError):
            compare.compare(self.bundle, "native", "candidate", 100)

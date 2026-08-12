from __future__ import annotations

import json
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import _read_jsonl, evaluate, validate_run_manifest, verify_fixture  # noqa: E402


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ROOT / "fixtures" / "v1"
        self.corpus = _read_jsonl(self.fixture / "corpus.jsonl")
        self.queries = _read_jsonl(self.fixture / "queries.jsonl")
        self.rankings = json.loads((self.fixture / "reference-ranked-results.json").read_text())

    def test_reference_rankings_are_a_deterministic_smoke_test(self) -> None:
        manifest = verify_fixture(self.fixture, self.fixture / "reference-ranked-results.json")
        self.assertEqual(manifest["fixture_version"], "v1")
        report = evaluate(self.corpus, self.queries, self.rankings)
        self.assertEqual(report["overall"]["recall_at_5"], 1.0)
        self.assertEqual(report["overall"]["exact_match_preservation_at_5"], 1.0)
        self.assertGreater(report["overall"]["false_positive_retrieval_rate_at_5"], 0.0)
        self.assertEqual(len(report["by_category"]), 9)

    def test_unknown_ranked_memory_is_rejected(self) -> None:
        rankings = dict(self.rankings)
        rankings["q-semantic"] = [{"memory_id": "not-in-corpus", "score": 1.0}]
        with self.assertRaisesRegex(ValueError, "unknown memory IDs"):
            evaluate(self.corpus, self.queries, rankings)

    def test_placeholder_run_manifest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not pin"):
            validate_run_manifest(ROOT / "run-manifest.example.json", "a" * 64)


if __name__ == "__main__":
    unittest.main()

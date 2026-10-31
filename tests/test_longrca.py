"""Zero-inference LongRCA qualification. Fake native binding is explicit."""

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from campaigns.longrca.arms import arm_input, read_trace_step
from campaigns.longrca.native import NativeTraceSession
from campaigns.longrca.prepare import (
    FIXTURE_PATH,
    LOCK_PATH,
    prepare,
    read_official,
    source_status,
)
from campaigns.longrca.schema import normalize_role, split_record, validate_public
from campaigns.longrca.scoring import aggregate, score, score_files

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "longrca"


class FakeNative:
    """Test double only: proves request routing, never native qualification."""

    store_id = "synthetic-isolated-store"

    def __init__(self):
        self.calls = []
        self.good_receipt = True
        self.empty = False

    def call(self, tool, arguments):
        self.calls.append((tool, copy.deepcopy(arguments)))
        if self.empty:
            return {}
        if tool == "smart_ingest":
            return {
                "nodeId": "fixture-node-" + str(len(self.calls)),
                "receipt": "FAKE_ONLY",
            }
        return {
            "nodes": [],
            "receipt": "FAKE_ONLY",
            "emptyBecause": "no recorded dependency",
        }

    def verify_receipt(self, result, tool, arguments):
        return self.good_receipt


class LongRCATests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE_PATH.read_text())
        self.public, self.reference = split_record(self.raw)

    def test_schema_splits_gold_without_changing_history(self):
        self.assertEqual(self.public.as_dict()["history"], self.raw["history"])
        self.assertEqual(set(self.public.as_dict()), {"question_ID", "history"})
        self.assertNotIn("SCORER_ONLY_SENTINEL", json.dumps(self.public.as_dict()))
        self.assertEqual(self.reference.step, 1)

    def test_schema_rejects_leaky_fields_and_reindexing(self):
        for change in ("gold", "step", "role", "root"):
            raw = copy.deepcopy(self.raw)
            if change == "gold":
                raw["history"][0]["mistake_reason"] = "secret"
            elif change == "step":
                raw["history"][0]["step"] = True
            elif change == "role":
                raw["mistake_agent"] = "UnrecordedAgent"
            else:
                raw["mistake_step"] = -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                split_record(raw)
        with self.assertRaises(ValueError):
            validate_public(self.raw)

    def test_role_normalization_matches_pinned_official_function(self):
        path = UPSTREAM / "export_paper_results.py"
        self.assertTrue(path.is_file())
        spec = importlib.util.spec_from_file_location("pinned_longrca_exporter", path)
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        values = [
            " Planner (-> Executor) ",
            "Planner (→ Executor)",
            "ＰＬＡＮＮＥＲ",
            "A (meaningful)",
            " A  B ",
        ]
        for value in values:
            self.assertEqual(normalize_role(value), exporter.normalize_role(value))

    def test_role_and_root_are_scored_independently(self):
        row = score(
            self.public,
            self.reference,
            {"predicted_role": "Planner", "predicted_step": 2},
        )
        self.assertTrue(row["role_correct"])
        self.assertFalse(row["root_exact"])
        row = score(
            self.public,
            self.reference,
            {"predicted_role": "Executor", "predicted_step": 1},
        )
        self.assertFalse(row["role_correct"])
        self.assertTrue(row["root_exact"])

    def test_invalid_predictions_keep_denominators(self):
        for step in (True, 1.0, -1, 500, None, "NaN", "1.5"):
            with self.subTest(step=step):
                row = score(
                    self.public,
                    self.reference,
                    {"predicted_role": "Planner", "predicted_step": step},
                )
                self.assertTrue(row["role_correct"])
                self.assertFalse(row["step_valid"])
        row = score(self.public, self.reference, None)
        result = aggregate([row])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["root_exact_accuracy"], 0)
        self.assertIsNone(result["source_weighted_valid_root_mae"])

    def test_source_weighted_mae_and_missing_source_are_explicit(self):
        a = score(self.public, self.reference, {"predicted_step": "3"})
        b, c = copy.deepcopy(a), copy.deepcopy(a)
        b.update(question_ID="terminal_bench_2__999998", root_absolute_error=0)
        c.update(
            question_ID="vitabench__999999", source="vitabench", root_absolute_error=9
        )
        self.assertAlmostEqual(
            aggregate([a, b, c])["source_weighted_valid_root_mae"], 11 / 3
        )
        c.update(step_valid=False, root_absolute_error=None)
        self.assertIsNone(aggregate([a, b, c])["source_weighted_valid_root_mae"])

    def test_input_contracts_have_same_evidence_and_no_annotations(self):
        seen = []
        for arm in ("no_memory", "long_context", "trace_lookup", "vestige"):
            result = arm_input(self.public, arm)
            raw = json.dumps(result)
            self.assertNotIn("SCORER_ONLY_SENTINEL", raw)
            self.assertNotIn("mistake_agent", raw)
            seen.append(result["evidence_sha256"])
        self.assertEqual(len(set(seen)), 1)
        a = arm_input(self.public, "no_memory")
        b = arm_input(self.public, "long_context")
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["equivalence_group"], b["equivalence_group"])
        for step in range(len(self.public.history)):
            self.assertEqual(read_trace_step(self.public, step), a["history"][step])

    def test_native_path_writes_only_public_and_never_invents_edges(self):
        binding = FakeNative()
        session = NativeTraceSession(self.public, binding)
        session.ingest()
        for tool, arguments in binding.calls:
            self.assertEqual(tool, "smart_ingest")
            self.assertEqual(arguments["links"], [])
            self.assertNotIn("SCORER_ONLY_SENTINEL", json.dumps(arguments))
            self.assertNotIn("mistake_step", json.dumps(arguments))
        session.read_step(2)
        session.causal_walk(3)
        self.assertEqual(binding.calls[-1][0], "causal_walk")
        self.assertEqual(
            binding.calls[-1][1]["start_points"][0]["node_id"], session.nodes[3]
        )
        self.assertEqual(session.diagnostic()["model_reads"], 2)
        self.assertEqual(session.diagnostic()["controller_supplied_edges"], 0)
        self.assertIsNone(session.diagnostic()["root_or_role_prediction"])
        with self.assertRaises(RuntimeError):
            session.ingest()

    def test_native_empty_transport_bad_receipts_and_invalid_handles_fail_closed(self):
        for empty in (False, True):
            binding = FakeNative()
            binding.empty = empty
            binding.good_receipt = False
            with self.assertRaisesRegex(RuntimeError, "FAIL_CLOSED_INTERDICTION"):
                NativeTraceSession(self.public, binding).ingest()
        session = NativeTraceSession(self.public, FakeNative())
        with self.assertRaises(ValueError):
            session.causal_walk(1)
        session.ingest()
        with self.assertRaises(ValueError):
            session.read_step(True)

    def test_native_transport_failure_is_logged_and_latches_closed(self):
        binding = FakeNative()
        session = NativeTraceSession(self.public, binding)
        session.ingest()

        def transport_failure(tool, args):
            raise OSError("RPC dropped")

        binding.call = transport_failure
        with self.assertRaisesRegex(RuntimeError, "FAIL_CLOSED_INTERDICTION"):
            session.causal_walk(3)
        self.assertTrue(session.diagnostic()["halted"])
        self.assertEqual(session.diagnostic()["model_reads"], 0)
        self.assertEqual(session.events[-1]["status"], "interdicted")
        calls = len(session.events)
        with self.assertRaisesRegex(RuntimeError, "already halted"):
            session.read_step(0)
        self.assertEqual(len(session.events), calls)

    def test_fixture_prepare_is_executable_private_and_distinct_from_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared"
            report = prepare(UPSTREAM, path, fixture=True)
            self.assertEqual(report["status"], "SYNTHETIC_FIXTURE_PREPARED")
            self.assertIsNone(report["benchmark_scores"])
            self.assertEqual(report["inference_calls"], 0)
            self.assertNotIn(
                "SCORER_ONLY_SENTINEL", (path / "public.jsonl").read_text()
            )
            self.assertIn(
                "SCORER_ONLY_SENTINEL",
                (path / "scorer-only/references.jsonl").read_text(),
            )
            self.assertEqual(
                os.stat(path / "scorer-only/references.jsonl").st_mode & 0o777, 0o600
            )
            with self.assertRaises(FileExistsError):
                prepare(UPSTREAM, path, fixture=True)

    def test_scorer_missing_predictions_count_as_wrong_and_unknown_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared"
            prepare(UPSTREAM, path, fixture=True)
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text("")
            scored = score_files(
                path / "public.jsonl",
                path / "scorer-only/references.jsonl",
                predictions,
            )
            self.assertEqual(scored["metrics"]["n"], 1)
            self.assertEqual(scored["metrics"]["role_correct"], 0)
            predictions.write_text('{"question_ID":"not-a-dataset-id"}\n')
            with self.assertRaises(ValueError):
                score_files(
                    path / "public.jsonl",
                    path / "scorer-only/references.jsonl",
                    predictions,
                )

    def test_source_identity_and_complete_split_indexes(self):
        lock = json.loads(LOCK_PATH.read_text())
        status = source_status(UPSTREAM, lock)
        self.assertEqual(status["status"], "VERIFIED_PINNED_METADATA")
        tree = json.loads((UPSTREAM / "dataset-tree.json").read_text()) + json.loads(
            (UPSTREAM / "dataset-tree-page2.json").read_text()
        )
        for split, expected in (("full", 1140), ("mini", 200)):
            self.assertEqual(
                sum(
                    item["type"] == "file"
                    and item["path"].startswith("data/" + split + "/")
                    for item in tree
                ),
                expected,
            )
        self.assertEqual(lock["dataset_license"], "NOT_DECLARED")
        changed = copy.deepcopy(lock)
        changed["files"]["dataset-README.md"] = "0" * 64
        self.assertEqual(source_status(UPSTREAM, changed)["status"], "SOURCE_MISMATCH")

    def test_official_mini_and_full_validate_schema_and_frozen_ids(self):
        lock = json.loads(LOCK_PATH.read_text())
        for split in ("mini", "full"):
            with self.subTest(split=split):
                records = read_official(UPSTREAM, split, lock)
                for record in records:
                    public, reference = split_record(record)
                    self.assertNotIn("mistake_reason", public.as_dict())
                    self.assertEqual(reference.question_id, public.question_id)

    def test_absent_source_reports_pending_without_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = prepare(path / "absent", path / "prepared")
            self.assertEqual(result["status"], "SOURCE_PENDING")
            self.assertEqual(result["inference_calls"], 0)
            self.assertEqual(result["prepared_n"], 0)


if __name__ == "__main__":
    unittest.main()

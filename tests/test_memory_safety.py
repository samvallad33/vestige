"""Offline item/label/scorer contracts; never a model or native-security score."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from campaigns.causal_drift.prepare import prepare as prepare_drift
from campaigns.memform.prepare import (
    FORBIDDEN,
    assert_no_labels,
    commit_packet,
    digest,
    load_prepared,
    normalize_case,
    parse_mcq_answer,
    prepare,
    score_results,
    upstream_modules,
)
from campaigns.reconcile_safe.prepare import native_readiness
from campaigns.reconcile_safe.prepare import prepare as prepare_reconcile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/memform"


class MemFormContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common, cls.defense, _ = upstream_modules(UPSTREAM)
        cls.cases = cls.common.read_jsonl(
            UPSTREAM / "outputs/mcq_item_audit/accepted_cases.jsonl"
        )

    def case(self, interface):
        return copy.deepcopy(
            next(row for row in self.cases if row["interface"] == interface)
        )

    def test_actual_official_corpus_all_interfaces_validate(self):
        self.assertEqual(len(self.cases), 994)
        self.assertEqual(
            {row["interface"] for row in self.cases}, {"P1", "P2", "P3", "P4", "P5"}
        )
        for row in self.cases:
            self.assertEqual(self.common.validate_compiled_case(row), [])

    def test_gold_change_cannot_change_model_visible_packets(self):
        case = self.case("P1")
        first, _ = normalize_case(case, self.common, self.defense)
        task = case["evaluation"]["trigger_tasks"][0]
        task["clean_answer"], task["poisoned_answer"] = (
            task["poisoned_answer"],
            task["clean_answer"],
        )
        case["clean_policy"]["rule"] = "SECRET_SCORING_ONLY"
        second, _ = normalize_case(case, self.common, self.defense)
        self.assertEqual(first, second)
        self.assertNotIn("SECRET_SCORING_ONLY", json.dumps(second))

    def test_provenance_and_experience_ids_do_not_enter_runtime(self):
        for interface in ("P1", "P2", "P3", "P4", "P5"):
            case = self.case(interface)
            packets, private = normalize_case(case, self.common, self.defense)
            for packet in packets:
                assert_no_labels(packet)
                self.assertNotIn(case["case_id"], json.dumps(packet))
                self.assertEqual(
                    set(packet),
                    {
                        "record_id",
                        "context",
                        "observations",
                        "direct_memory_text",
                        "formation_instruction",
                        "tasks",
                    },
                )
            self.assertEqual(private["interface"], interface)
            self.assertTrue(private["slots"])

    def test_nested_forbidden_metadata_fails(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            assert_no_labels(
                {
                    "observations": [
                        {"text": "observation", "provenance": {"authentic": False}}
                    ]
                }
            )

    def test_p5_order_and_multiset_preserved(self):
        case = self.case("P5")
        packets, private = normalize_case(case, self.common, self.defense)
        by_slot = {private["slots"][row["record_id"]]["slot"]: row for row in packets}
        a, b = by_slot["clean"], by_slot["poisoned"]
        self.assertNotEqual(a["observations"], b["observations"])
        self.assertCountEqual(a["observations"], b["observations"])
        self.assertEqual(a["tasks"], b["tasks"])
        self.assertNotIn("clean", a["record_id"])
        self.assertNotIn("poisoned", b["record_id"])

    def test_p4_retained_subset_not_replaced_with_full_pool(self):
        case = self.case("P4")
        packets, private = normalize_case(case, self.common, self.defense)
        by_slot = {private["slots"][row["record_id"]]["slot"]: row for row in packets}
        self.assertEqual(
            len(by_slot["clean"]["observations"]),
            case["memory_building"]["selection_budget"],
        )
        self.assertGreater(
            len(by_slot["control:full_pool"]["observations"]),
            len(by_slot["clean"]["observations"]),
        )

    def test_prepare_freezes_all_interfaces_and_separates_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prepared"
            manifest = prepare(UPSTREAM, output, limit=5)
            self.assertEqual(manifest["selected_cases"], 5)
            self.assertEqual(manifest["model_calls"], 0)
            inputs = [
                json.loads(row)
                for row in (output / "runtime-inputs.jsonl").read_text().splitlines()
            ]
            keys = [
                json.loads(row)
                for row in (output / "answer-key.jsonl").read_text().splitlines()
            ]
            self.assertEqual(digest(inputs), manifest["runtime_packets_sha256"])
            self.assertEqual(digest(keys), manifest["answer_key_sha256"])
            self.assertEqual(
                {row["interface"] for row in keys}, {"P1", "P2", "P3", "P4", "P5"}
            )
            self.assertEqual(
                (output / "answer-key.jsonl").stat().st_mode & 0o777, 0o600
            )
            self.assertEqual(load_prepared(output)[1], inputs)
            with (output / "runtime-inputs.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_prepared(output)
            for row in inputs:
                assert_no_labels(row)

    def test_exact_upstream_mcq_parser_without_importing_broken_runner(self):
        self.assertEqual(
            parse_mcq_answer(UPSTREAM, "Consider B.\nFINAL_ANSWER: A", ["A", "B"])[0],
            "A",
        )
        self.assertIsNone(parse_mcq_answer(UPSTREAM, "I cannot decide", ["A", "B"])[0])

    @staticmethod
    def result(clean="A", poison="B"):
        return [
            {
                "interface": "P1",
                "case_id": "synthetic-scoring-contract",
                "task_results": [
                    {
                        "task_id": "t1",
                        "clean_answer": "A",
                        "poisoned_answer": "B",
                        "baseline": {"predicted_answer": "A"},
                        "learned_clean": {"predicted_answer": clean},
                        "learned_poisoned": {"predicted_answer": poison},
                    }
                ],
            }
        ]

    def test_original_scorer_fixed_denominator_does_not_hide_clean_regression(self):
        score = score_results(
            UPSTREAM,
            self.result(),
            self.result(clean="B", poison="A"),
            synthetic_fixture=True,
        )["metrics"]
        self.assertEqual(score["n_eligible_total"], 1)
        self.assertEqual(score["by_interface"]["P1"]["learned_clean_accuracy"], 0)

    def test_original_scorer_withholds_incomplete_scores(self):
        score = score_results(UPSTREAM, self.result(), [], synthetic_fixture=True)[
            "metrics"
        ]
        self.assertFalse(score["results_complete"])
        self.assertIsNone(score["avg_asr"])
        self.assertEqual(score["n_missing_total"], 1)
        self.assertTrue(score["do_not_interpret_asr_as_defense_gain"])

    def test_unregistered_gold_results_rejected_outside_explicit_fixture_mode(self):
        with self.assertRaisesRegex(ValueError, "official corpus"):
            score_results(UPSTREAM, self.result(), self.result())

    def test_missing_method_condition_is_technical_not_defense_gain(self):
        method = self.result()
        del method[0]["task_results"][0]["learned_poisoned"]
        report = score_results(UPSTREAM, self.result(), method, synthetic_fixture=True)[
            "metrics"
        ]
        self.assertFalse(report["results_complete"])
        self.assertIsNone(report["avg_asr"])

    def test_missing_undefended_prediction_cannot_shrink_eligibility(self):
        base = self.result()
        del base[0]["task_results"][0]["baseline"]
        with self.assertRaisesRegex(ValueError, "eligibility cannot be frozen"):
            score_results(UPSTREAM, base, self.result(), synthetic_fixture=True)

    def test_native_commit_returns_only_handles_after_verified_receipts(self):
        class Stack:
            def __init__(self):
                self.calls = []

            def tool(self, sid, name, args):
                self.calls.append((sid, name, args))
                if name == "smart_ingest":
                    return {"structuredContent": {"nodeId": "mem-fixture"}}
                return {
                    "structuredContent": {
                        "attestation": {
                            "verification": {
                                "chainValid": True,
                                "gateAllowed": True,
                                "locallyVerified": True,
                            }
                        }
                    }
                }

        stack = Stack()
        packet = normalize_case(self.case("P1"), self.common, self.defense)[0][0]
        committed = commit_packet(stack, "isolated", packet)
        self.assertEqual(committed["handles"], ["mem-fixture"])
        self.assertEqual(committed["solver_header_memory_bodies"], [])
        self.assertEqual([row[1] for row in stack.calls], ["smart_ingest", "receipt"])
        self.assertTrue(FORBIDDEN.isdisjoint(stack.calls[0][2]))

    def test_missing_native_attestation_stops_before_reporting_commit(self):
        class Stack:
            def tool(self, sid, name, args):
                return (
                    {"structuredContent": {"nodeId": "mem-fixture"}}
                    if name == "smart_ingest"
                    else {"structuredContent": {}}
                )

        packet = normalize_case(self.case("P1"), self.common, self.defense)[0][0]
        with self.assertRaisesRegex(RuntimeError, "did not verify"):
            commit_packet(Stack(), "isolated", packet)


class UnresolvedIdentityContracts(unittest.TestCase):
    def test_exact_names_remain_unresolved_not_official_fixture_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, function in (
                ("ReconcileSafe", prepare_reconcile),
                ("Causal-Drift-Eval", prepare_drift),
            ):
                root = Path(tmp) / name
                manifest = function(root)
                self.assertEqual(
                    manifest["official_source_status"], "SOURCE_UNRESOLVED"
                )
                self.assertIsNone(manifest["official_score"])
                self.assertFalse(manifest["local_extension"]["official_benchmark"])
                inputs = json.loads((root / "runtime-inputs.json").read_text())
                self.assertTrue(all("gold" not in row for row in inputs))
                self.assertEqual(manifest["local_extension"]["fixture_count"], 3)

    def test_missing_or_unverified_native_sheaf_api_never_passes(self):
        self.assertEqual(native_readiness()["state"], "UNSUPPORTED_NATIVE_API")
        report = native_readiness({"supported": True})
        self.assertEqual(report["state"], "UNVERIFIED_NATIVE_EVIDENCE")
        self.assertFalse(report["native_execution_verified"])
        self.assertIsNone(report["official_score"])


if __name__ == "__main__":
    unittest.main()

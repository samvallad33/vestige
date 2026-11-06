"""Temporal/gold isolation and exact-coverage checks for official bridges."""
from __future__ import annotations

import asyncio
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .extra_datasets import (
    MEMCONFLICT_DATA_SHA256,
    judge_memconflict,
    load_memconflict,
    memconflict_records,
    memconflict_scorer,
)


def fixture():
    return [{"ID": "persona", "Fixed_Profile": "GOLD_PROFILE", "Full_Session_Chain": [
        {"Session_ID": 1, "Date": "2026-01-01", "Static_Conflict_Information": "GOLD_ANNOTATION",
         "Session_Dialogue": {
             "dialogue_turn_10": [{"role": "assistant", "content": "Observed assistant response"}],
             "dialogue_turn_2": [{"role": "user", "content": "I live in Paris"}],
         },
         "Session_Questions": [{"question_id": "Q_1", "question": "Where do I live?",
                                "answer": "GOLD_ANSWER", "conflict_type": "dynamic_conflict"}]},
        {"Session_ID": 2, "Date": "2026-02-01",
         "Session_Dialogue": {"dialogue_turn_1": [{"role": "user", "content": "FUTURE_NEW_CITY"}]},
         "Session_Questions": [{"question_id": "Q_2", "question": "Where now?",
                                "answer": "GOLD_NEW", "conflict_type": "dynamic_conflict"}]},
    ]}]


class MemConflictIsolationTests(unittest.TestCase):
    def test_future_sessions_and_private_annotations_never_enter_history(self):
        first, second = memconflict_records(fixture())
        self.assertIn("I live in Paris", first["history"][0])
        self.assertIn("Observed assistant response", first["history"][1])
        self.assertNotIn("FUTURE_NEW_CITY", "\n".join(first["history"]))
        self.assertIn("FUTURE_NEW_CITY", "\n".join(second["history"]))
        self.assertNotEqual(first["group_id"], second["group_id"])
        self.assertNotIn("GOLD_", "\n".join(first["history"]))
        self.assertNotIn("GOLD_", first["question"])
        self.assertEqual(first["scorer_payload"]["question"]["answer"], "GOLD_ANSWER")

    def test_temporal_record_is_stable_if_later_dialogue_changes(self):
        original = fixture()
        changed = copy.deepcopy(original)
        changed[0]["Full_Session_Chain"][1]["Session_Dialogue"]["dialogue_turn_1"][0]["content"] = "OTHER_FUTURE"
        self.assertEqual(memconflict_records(original)[0], memconflict_records(changed)[0])

    def test_roleless_official_malformed_turn_is_not_invented_as_user(self):
        data = fixture()
        data[0]["Full_Session_Chain"][0]["Session_Dialogue"]["dialogue_turn_3"] = [
            {"assistant": "assistant", "content": "ROLELESS_RECORD"},
            {"role": "assistant", "content": "VALID_ASSISTANT"},
        ]
        record = memconflict_records(data)[0]
        self.assertNotIn("ROLELESS_RECORD", "\n".join(record["history"]))
        self.assertIn("VALID_ASSISTANT", "\n".join(record["history"]))

    def test_duplicate_sessions_and_questions_fail_closed(self):
        data = fixture()
        data[0]["Full_Session_Chain"].append(copy.deepcopy(data[0]["Full_Session_Chain"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate.*session"):
            memconflict_records(data)
        data = fixture()
        questions = data[0]["Full_Session_Chain"][0]["Session_Questions"]
        questions.append(copy.deepcopy(questions[0]))
        with self.assertRaisesRegex(ValueError, "duplicate.*question"):
            memconflict_records(data)

    def test_missing_merged_prediction_rejected_before_judge(self):
        item = SimpleNamespace(id="expected")
        async def forbidden(_):
            self.fail("judge called before coverage check")
        with self.assertRaisesRegex(ValueError, "coverage"):
            asyncio.run(judge_memconflict("unavailable", [item], {}, forbidden))

    def test_judge_transport_error_is_not_converted_to_score(self):
        scorer = SimpleNamespace(
            extract_top_k_retrieved_memories=lambda data: data["Retrieved_Memories"],
            build_llm_judge_prompt=lambda *args: "official prompt",
            LLM_JUDGE_SYSTEM_PROMPT="official system",
        )
        item = SimpleNamespace(id="a", scorer_payload={"question": {"conflict_type": "dynamic_conflict"}})
        async def failed(_):
            raise RuntimeError("budget exhausted")
        with patch("benchmarks.real_engine.extra_datasets.memconflict_scorer", return_value=(scorer, None)):
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                asyncio.run(judge_memconflict("unused", [item], {"a": "answer"}, failed))


UPSTREAM = Path(__file__).resolve().parents[2] / "runs/upstream/memconflict"


@unittest.skipUnless((UPSTREAM / "Data/Step4_4.jsonl").is_file(), "official pinned data not downloaded")
class MemConflictOfficialIntegrationTests(unittest.TestCase):
    def test_full_official_data_count_and_solver_isolation(self):
        items = load_memconflict(UPSTREAM)
        self.assertEqual(len(items), 3750)
        self.assertEqual(len({item.id for item in items}), 3750)
        self.assertEqual({item.category for item in items},
                         {"dynamic_conflict", "static_conflict", "conditional_conflict"})
        for item in items:
            self.assertNotIn("scorer_payload", item.solver_record())
        self.assertEqual(len(MEMCONFLICT_DATA_SHA256), 64)

    def test_official_judge_prompt_and_metric_parser_without_provider(self):
        scorer, parser = memconflict_scorer(UPSTREAM)
        question = {"question": "Where?", "answer": "Paris", "conflict_type": "dynamic_conflict"}
        prompt = scorer.build_llm_judge_prompt(question, "Paris", [])
        self.assertIn("Reference Answer: Paris", prompt)
        parsed = parser._extract_json_from_content(
            '{"dynamic_answer_accuracy":0.5,"update_awareness_and_order_consistency_score":1}', []
        )
        result = scorer.parse_llm_metric_result("dynamic_conflict", parsed, "model_response")
        self.assertEqual(result["Metrics"]["dynamic_answer_accuracy"], 0.5)
        self.assertEqual(result["Metrics"]["update_awareness_and_order_consistency_score"], 1.0)


if __name__ == "__main__":
    unittest.main()

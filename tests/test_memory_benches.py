"""Structured offline fixtures; none of these are model benchmark results."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campaigns.longmemeval import metrics as lme_metrics
from campaigns.longmemeval import prepare as lme
from campaigns.memoryagentbench import metrics as mab_metrics
from campaigns.memoryagentbench import prepare as mab
from campaigns.memoryagentbench.common import constants, jsonl, write_packet


def longmem_fixture(qid="update-1", task="knowledge-update"):
    return {
        "question_id": qid,
        "question_type": task,
        "question": "What is my current project color?",
        "answer": "blue",
        "question_date": "2024/03/01 (Fri) 10:00",
        "haystack_session_ids": ["old-session", "new-session"],
        "haystack_dates": ["2024/01/01 (Mon) 09:00", "2024/02/01 (Thu) 09:00"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "My project color was red.",
                    "has_answer": False,
                }
            ],
            [
                {
                    "role": "user",
                    "content": "I changed it to blue.",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "I understand."},
            ],
        ],
        "answer_session_ids": ["new-session"],
    }


def mab_fixture():
    return {
        "context": "1. Delta's city is Rome.\n2. Delta's city is Paris.\n",
        "questions": ["What is Delta's current city?"],
        "answers": [["Paris"]],
        "metadata": {
            "source": "factconsolidation_sh_6k",
            "qa_pair_ids": ["source-qa-1"],
            "question_dates": ["2024-03-01"],
            "previous_events": ["prior city event"],
        },
    }


class MemoryBenchmarkTests(unittest.TestCase):
    def test_longmem_updates_dates_and_oracle_isolation(self):
        history, questions, oracle = lme.normalize([longmem_fixture()])
        self.assertEqual(
            [h["timestamp"] for h in history], longmem_fixture()["haystack_dates"]
        )
        self.assertIn("red", history[0]["turns"][0]["content"])
        self.assertIn("blue", history[1]["turns"][0]["content"])
        self.assertNotIn("has_answer", json.dumps(history))
        self.assertNotIn("answer_session_ids", json.dumps(questions))
        self.assertNotIn("knowledge-update", json.dumps(questions))
        self.assertEqual(oracle[0]["answer_runtime_sessions"], ["session-00001"])
        self.assertEqual(
            oracle[0]["answer_turns"], [{"session": "session-00001", "turn": 0}]
        )

    def test_abstention_indicator_hidden_but_preserved(self):
        row = longmem_fixture("something_abs", "single-session-user")
        row["answer_session_ids"] = []
        history, questions, oracle = lme.normalize([row])
        self.assertTrue(oracle[0]["abstention"])
        self.assertNotIn("_abs", json.dumps(questions) + json.dumps(history))
        self.assertEqual(
            lme_metrics.retrieval_metrics(oracle[0], []),
            {"status": "EXCLUDED_ABSTENTION", "recall_any": None, "recall_all": None},
        )

    def test_misaligned_sessions_and_missing_questions_rejected(self):
        row = longmem_fixture()
        row["haystack_dates"].pop()
        with self.assertRaises(ValueError):
            lme.normalize([row])

    def test_duplicate_source_sessions_preserved_and_retrieval_aliases_credit_either(
        self,
    ):
        row = longmem_fixture()
        row["haystack_session_ids"] = ["same", "same"]
        row["answer_session_ids"] = ["same"]
        history, _, oracle = lme.normalize([row])
        self.assertEqual(len(history), 2)
        self.assertEqual(
            oracle[0]["session_id_map"]["same"], ["session-00000", "session-00001"]
        )
        self.assertEqual(
            lme_metrics.retrieval_metrics(oracle[0], ["session-00001"])["recall_all"],
            1.0,
        )
        row = longmem_fixture()
        row["question"] = None
        with self.assertRaises(ValueError):
            lme.normalize([row])

    def test_exact_upstream_judge_has_actual_question_and_update_rule(self):
        _, _, oracle = lme.normalize([longmem_fixture()])
        payloads = lme_metrics.judge_payloads(
            oracle, [{"question_id": "lme-q00000", "hypothesis": "blue"}]
        )
        text = payloads[0]["messages"][0]["content"]
        self.assertIn("Question: What is my current project color?", text)
        self.assertNotIn("Question: None", text)
        self.assertIn("updated answer", text)
        self.assertEqual(payloads[0]["status"], "JUDGE_NOT_EXECUTED")
        self.assertEqual(payloads[0]["budget_usd"], 0)

    def test_abstention_judge_and_no_unwitnessed_score(self):
        row = longmem_fixture("missing_abs", "single-session-user")
        row["answer_session_ids"] = []
        _, _, oracle = lme.normalize([row])
        payload = lme_metrics.judge_payloads(
            oracle, [{"question_id": "lme-q00000", "hypothesis": "Unknown."}]
        )[0]
        self.assertIn("unanswerable question", payload["messages"][0]["content"])
        self.assertEqual(
            lme_metrics.aggregate_judgements(
                oracle, [], expected_model="fixture-judge"
            )["status"],
            "NO_JUDGE_EVIDENCE",
        )
        label = {
            "question_id": "lme-q00000",
            "autoeval_label": {"model": "fixture-judge", "label": True},
        }
        with self.assertRaises(ValueError):
            lme_metrics.aggregate_judgements(
                oracle, [label], expected_model="fixture-judge"
            )
        label.update(raw_judge_response="yes", request_sha256="f" * 64)
        result = lme_metrics.aggregate_judgements(
            oracle, [label], expected_model="fixture-judge"
        )
        self.assertEqual(result["accuracy"]["abstention"]["accuracy"], 1)

    def test_lossless_serial_history_and_gold_free_runtime(self):
        templates = constants(
            mab.DEFAULT_UPSTREAM / "utils/templates.py", ["BASE_TEMPLATES"]
        )["BASE_TEMPLATES"]
        history, questions, oracle = mab.normalize(
            [mab_fixture()], templates, chunk_chars=9
        )
        self.assertEqual(
            "".join(h["content"] for h in history), mab_fixture()["context"]
        )
        self.assertNotIn("answers", json.dumps(history) + json.dumps(questions))
        self.assertNotIn("source-qa-1", json.dumps(history) + json.dumps(questions))
        self.assertIn("newer fact", questions[0]["prompt"])
        self.assertEqual(
            oracle[0]["source_metadata"]["previous_events"], "prior city event"
        )

    def test_original_exact_and_substring_metrics(self):
        reference = {
            "metric": "exact_match",
            "source": "icl_trec_fine",
            "answers": ["43"],
        }
        self.assertFalse(mab_metrics.primary_score(reference, "label: 43")["score"])
        self.assertTrue(mab_metrics.primary_score(reference, "Answer: 43")["score"])
        reference.update(
            metric="substring_exact_match",
            source="factconsolidation_sh_6k",
            answers=["Paris"],
        )
        self.assertTrue(
            mab_metrics.primary_score(reference, "The answer is Paris.")["score"]
        )
        self.assertFalse(mab_metrics.primary_score(reference, "Rome")["score"])

    def test_summary_judge_payloads_are_pending_and_question_bearing(self):
        reference = {
            "question_id": "summary-fixture",
            "question": "Summarize the book.",
            "metric": "llm_judge_summary_f1",
            "answers": ["The book describes a journey."],
            "source_metadata": {"keypoints": ["A journey occurs."]},
        }
        payloads = mab_metrics.judge_payloads(reference, "A journey occurs.")
        self.assertEqual(
            {p["kind"] for p in payloads}, {"fluency", "precision", "recall"}
        )
        self.assertTrue(
            all(
                p["status"] == "JUDGE_NOT_EXECUTED" and p["budget_usd"] == 0
                for p in payloads
            )
        )
        self.assertTrue(
            all(
                "Question: Summarize the book." in p["messages"][0]["content"]
                for p in payloads
            )
        )
        self.assertIsNone(
            mab_metrics.score_supplied_judge_outputs(
                reference, [], expected_model="fixture"
            )["score"]
        )
        outputs = [
            {
                "kind": kind,
                "model": "fixture",
                "request_sha256": "f" * 64,
                "raw_judge_response": json.dumps(value),
            }
            for kind, value in {
                "fluency": {"fluency": 1},
                "recall": {"recall": 1},
                "precision": {"precision": 1, "sentence_count": 2},
            }.items()
        ]
        score = mab_metrics.score_supplied_judge_outputs(
            reference, outputs, expected_model="fixture"
        )["score"]
        self.assertAlmostEqual(score, 2 / 3)

    def test_recsys_requires_process_seed_for_original_set_ties(self):
        with patch.dict(os.environ, {}, clear=True):
            result = mab_metrics.recsys_score({"answers": ["1"]}, "A film")
        self.assertEqual(result["status"], "REQUIRES_PINNED_HASHSEED")
        self.assertIsNone(result["score"])

    def test_packets_refuse_overwrite_and_keep_evaluator_separate(self):
        history, questions, oracle = lme.normalize([longmem_fixture()])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            manifest = write_packet(
                output, history, questions, oracle, {"campaign": "synthetic-fixture"}
            )
            self.assertEqual(manifest["provider_requests"], 0)
            self.assertFalse(manifest["paid_run_authorized"])
            self.assertEqual(jsonl(output / "questions.jsonl"), questions)
            with self.assertRaises(FileExistsError):
                write_packet(output, history, questions, oracle, {})

    def test_mab_alignment_refused_instead_of_silent_zip_loss(self):
        templates = constants(
            mab.DEFAULT_UPSTREAM / "utils/templates.py", ["BASE_TEMPLATES"]
        )["BASE_TEMPLATES"]
        row = mab_fixture()
        row["answers"] = []
        with self.assertRaises(ValueError):
            mab.normalize([row], templates)


if __name__ == "__main__":
    unittest.main()

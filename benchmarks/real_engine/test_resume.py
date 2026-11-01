"""Unit-only evidence fixtures. These are never benchmark results."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import resume
from .provider import account
from .runtime import BudgetLedger, Checkpoint, append_jsonl, canonical, shard_id


class BaselineResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.runs = self.repo/"runs"
        self.original = self.runs/"longmemeval"
        self.source = self.runs/"preserved/longmemeval"
        self.source.mkdir(parents=True)
        self.items = [SimpleNamespace(id=f"official-{i}", question=f"Question {i}?",
                                     history=[f"Actual unmodified history for item {i}."]) for i in range(500)]
        self.pilot_ids = sorted(random.Random(20261101).sample([i.id for i in self.items], 10))
        self.item = next(i for i in self.items if i.id == self.pilot_ids[0])
        self.data = self.runs/"upstream/longmemeval/data.json"
        self.scorer = self.runs/"upstream/longmemeval/scorer.py"
        self.data.parent.mkdir(parents=True)
        self.data.write_text("unit fixture dataset\n")
        self.scorer.write_text("# unit fixture scorer\n")
        self.addCleanup(patch.stopall)
        patch.object(resume, "OFFICIAL_DATA_SHA256", resume.file_digest(self.data)).start()
        patch.object(resume, "OFFICIAL_SCORER_SHA256", resume.file_digest(self.scorer)).start()
        sources = {}
        for relative in resume.BASELINE_SOURCE_PINS:
            path = self.repo/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# unchanged scientific source fixture\n")
            sources[relative] = resume.file_digest(path)
        from .prompts import audit_prompts
        self.old = {"schema": "fixture-config", "model": "gpt-5.4-nano", "temperature": 0,
            "limits": [6, 2048, 32768], "workers": 10, "max_inflight": 20,
            "episode_policy": "native_tool_errors_forwarded_unchanged;score_actual_budget_limited_outputs_and_stores",
            "sources": sources, "prompts": audit_prompts()}
        self._hash(self.old)
        self.new = deepcopy(self.old)
        self.new.update(workers=30, max_inflight=40)
        self._hash(self.new)
        self.old_id = self._identity(self.old)
        self.new_id = self._identity(self.new)
        self.old_plan = self._plan(self.old, self.old_id)
        self.new_plan = self._plan(self.new, self.new_id)
        self.config_path = self.runs/"preserved/source-config.json"
        self.config_path.write_text(canonical(self.old))
        self.ledger_path = self.runs/"budget.db"
        self.ledger = BudgetLedger(self.ledger_path)
        self.rows = {}
        for arm in resume.BASELINE_ARMS:
            self._row(arm)

    @staticmethod
    def _hash(config):
        config.pop("config_hash", None)
        config["config_hash"] = resume.digest(config)

    def _identity(self, config):
        return {"benchmark": "longmemeval", "config_hash": config["config_hash"],
            "model": config["model"], "trials": 1, "limits": config["limits"],
            "prompts": config["prompts"], "items": "unchanged-official-history-question-hash",
            "retrieval": "unchanged-baseline-retrieval"}

    def _plan(self, config, identity):
        return {"config_hash": config["config_hash"],
            "run_identities": {a: resume.digest({**identity, "arm": a}) for a in resume.BASELINE_ARMS},
            "budgets": {a: {"max_calls": 6, "max_output": 2048, "context": 32768} for a in resume.BASELINE_ARMS},
            "full_set": {"item_ids": [i.id for i in self.items], "trials": [0],
                "dataset_files": {str(self.data): resume.file_digest(self.data)}},
            "scorer": {"origin": "official", "entrypoint": "judge_longmemeval",
                "source_files": {str(self.scorer): resume.file_digest(self.scorer)}},
            "pilot": {"seed": 20261101, "item_ids": self.pilot_ids,
                "config_hash": config["config_hash"]}}

    def _row(self, arm):
        worker = shard_id(self.item.id, 10)
        relative = Path(arm)/f"worker-{worker}"/"attempt-original/api.jsonl"
        api = self.source/relative
        messages = [{"role": "system", "content": self.old["prompts"][arm]["text"]},
                    {"role": "user", "content": self.item.question}]
        payload = {"model": "gpt-5.4-nano", "temperature": 0, "reasoning_effort": "none",
            "store": False, "max_completion_tokens": 512, "messages": messages}
        if arm != "bare":
            payload.update(tools=resume._tools(arm), parallel_tool_calls=False)
        identity = {"benchmark": "longmemeval", "arm": arm, "item_id": self.item.id,
            "trial": 0, "config_hash": self.old["config_hash"], "phase": "integrity_preflight"}
        call_id = f"longmemeval/{arm}/fixture"
        response = {"model": "gpt-5.4-nano-2026-03-17", "choices": [{"message": {
            "role": "assistant", "content": "original provider answer"}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 3, "total_tokens": 28}}
        cost, usage = account(response)
        append_jsonl(api, {"event": "request", "identity": identity, "call_id": call_id, "payload": payload})
        append_jsonl(api, {"event": "response", "identity": identity, "call_id": call_id,
            "response": response, "usage": usage, "cost": cost})
        self.ledger.reserve(call_id+"/attempt-0", 0.01)
        self.ledger.settle(call_id+"/attempt-0", cost, usage)
        row = {**identity, "run_identity": self.old_plan["run_identities"][arm],
            "worker_id": worker, "shard": worker, "status": "complete", "score": None,
            "model": response["model"], "usage": [usage], "cost": cost,
            "prompt": messages, "response": "original provider answer",
            "transcript": messages + [{"role": "assistant", "content": "original provider answer"}],
            "raw_log": str((self.original/relative).relative_to(self.repo)),
            "budget": {"limits": {"max_calls": 6, "max_output": 2048, "context": 32768}},
            "timestamp": "2026-11-01T12:00:00+00:00"}
        checkpoint = self.source/arm/f"worker-{worker}.jsonl"
        append_jsonl(checkpoint, row)
        self.rows[arm] = row, checkpoint, api

    def prepare(self, **kwargs):
        params = dict(repo=self.repo, source_root=self.source, source_config_path=self.config_path,
            source_plan=self.old_plan, source_identity=self.old_id, current_config=self.new,
            current_plan=self.new_plan, current_identity=self.new_id, official_items=self.items)
        params.update(kwargs)
        return resume.prepare_baseline_resume(**params)

    def checkpoints(self):
        return {a: [Checkpoint(self.runs/"current"/a/f"worker-{n}.jsonl", {**self.new_id, "arm": a})
                    for n in range(30)] for a in resume.BASELINE_ARMS}

    def imported(self):
        manifest = self.prepare()
        checkpoints = self.checkpoints()
        counts = resume.import_baseline_resume(manifest, checkpoints, repo=self.repo,
            current_config=self.new, current_plan=self.new_plan, current_identity=self.new_id,
            official_items=self.items)
        records = [row for cps in checkpoints.values() for cp in cps for row in cp.done.values()]
        return manifest, checkpoints, counts, records

    def validate(self, row, manifest):
        return resume.validate_imported_baseline(row, manifest, self.repo, self.new,
            self.new_plan, self.new_id, self.items)

    def search_second_step(self, extra_hint=False, replace_history=False):
        row, checkpoint, api = self.rows["bm25"]
        events = deepcopy([r for _, r, _ in resume._records(api)])
        call = {"id": "actual-tool-call", "type": "function", "function": {
            "name": "memory_search", "arguments": canonical({"query": "history", "k": 1})}}
        assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
        events[1]["response"]["choices"][0]["message"] = assistant
        events[1]["response"]["choices"][0]["finish_reason"] = "tool_calls"
        tool = {"role": "tool", "tool_call_id": call["id"], "content": canonical([{
            "id": "session-0", "content": "substituted answer" if replace_history else self.item.history[0], "score": 1.5}])}
        messages = deepcopy(events[0]["payload"]["messages"]) + [assistant, tool]
        if extra_hint:
            messages.append({"role": "user", "content": "Injected answer hint in later request"})
        request = deepcopy(events[0]); request["call_id"] += "-second"
        request["payload"]["messages"] = messages
        response = deepcopy(events[1]); response["call_id"] = request["call_id"]
        response["response"]["choices"][0] = {"message": {"role": "assistant", "content": "original provider answer"}, "finish_reason": "stop"}
        events.extend([request, response])
        self.ledger.reserve(request["call_id"]+"/attempt-0", 0.01)
        self.ledger.settle(request["call_id"]+"/attempt-0", response["cost"], response["usage"])
        row = deepcopy(row)
        row["usage"] = [events[1]["usage"], response["usage"]]
        row["cost"] = events[1]["cost"] + response["cost"]
        row["transcript"] = messages + [response["response"]["choices"][0]["message"]]
        checkpoint.write_text(canonical(row)+"\n")
        api.write_text("".join(canonical(e)+"\n" for e in events))

    def test_unchanged_baselines_recover_with_explicit_original_identity(self):
        manifest, checkpoints, counts, rows = self.imported()
        self.assertEqual(counts, {a: 1 for a in resume.BASELINE_ARMS})
        self.assertEqual(manifest["workers"], 30)
        for row in rows:
            source, _, api = self.rows[row["arm"]]
            self.assertEqual(row["response"], source["response"])
            self.assertEqual(row["phase"], "integrity_preflight")
            self.assertEqual(row["source_run_identity"], source["run_identity"])
            self.assertEqual(row["origin"]["source_config_hash"], source["config_hash"])
            self.assertEqual(row["origin"]["raw_log"], source["raw_log"])
            self.assertEqual(row["raw_log"], str(api.resolve()))
            verified = self.validate(row, manifest)
            self.assertEqual(verified["source_row"], source)
            self.assertTrue(verified["source_pilot"])
            self.assertEqual(verified["raw_log_path"], str(api.resolve()))
        again = resume.import_baseline_resume(manifest, checkpoints, repo=self.repo,
            current_config=self.new, current_plan=self.new_plan, current_identity=self.new_id,
            official_items=self.items)
        self.assertEqual(again, {a: 0 for a in resume.BASELINE_ARMS})

    def test_vestige_import_is_always_prohibited(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.prepare(arms=("vestige",))

    def test_changed_scientific_inputs_are_rejected(self):
        for field, value in [("model", "other-model"), ("limits", [7,2048,32768])]:
            with self.subTest(field=field):
                config = deepcopy(self.new); config[field] = value; self._hash(config)
                identity = self._identity(config); plan = self._plan(config, identity)
                with self.assertRaises(ValueError):
                    self.prepare(current_config=config, current_identity=identity, current_plan=plan)
        changed = deepcopy(self.new)
        changed["prompts"]["bm25"]["text"] += " Extra hint."
        changed["prompts"]["bm25"]["sha256"] = "forged"
        self._hash(changed)
        identity = self._identity(changed); plan = self._plan(changed, identity)
        with self.assertRaises(ValueError):
            self.prepare(current_config=changed, current_identity=identity, current_plan=plan)

    def test_changed_retrieval_source_is_rejected(self):
        path = self.repo/"benchmarks/real_engine/retrieval.py"
        path.write_text("# different retrieval implementation\n")
        with self.assertRaisesRegex(ValueError, "scientific source"):
            self.prepare()

    def test_changed_episode_policy_is_rejected(self):
        changed = deepcopy(self.new)
        changed["episode_policy"] = "discard partial outputs"
        self._hash(changed)
        identity = self._identity(changed); plan = self._plan(changed, identity)
        with self.assertRaisesRegex(ValueError, "termination policy"):
            self.prepare(current_config=changed, current_identity=identity, current_plan=plan)

    def test_changed_dataset_or_scorer_bytes_are_rejected(self):
        for path in (self.data, self.scorer):
            original = path.read_text()
            path.write_text(original + "changed\n")
            with self.assertRaises(ValueError):
                self.prepare()
            path.write_text(original)

    def test_duplicate_source_shards_are_rejected(self):
        row, _, _ = self.rows["bare"]
        duplicate = self.source/"bare/worker-999.jsonl"
        append_jsonl(duplicate, row)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()

    def test_usage_less_or_unsettled_source_cannot_be_imported(self):
        _, _, api = self.rows["bm25"]
        events = [r for _, r, _ in resume._records(api)]
        events[-1].pop("usage")
        api.write_text("".join(canonical(e)+"\n" for e in events))
        with self.assertRaisesRegex(ValueError, "usage"):
            self.prepare()

    def test_provider_reply_without_a_settled_ledger_entry_is_rejected(self):
        _, _, api = self.rows["bm25"]
        events = deepcopy([r for _, r, _ in resume._records(api)])
        for event in events:
            event["call_id"] += "-no-settled-ledger-entry"
        api.write_text("".join(canonical(e)+"\n" for e in events))
        with self.assertRaisesRegex(ValueError, "shared-ledger"):
            self.prepare()

    def test_extra_generation_parameters_and_pilot_phase_relabel_are_rejected(self):
        _, _, api = self.rows["bm25"]
        original = api.read_text()
        events = deepcopy([r for _, r, _ in resume._records(api)])
        events[0]["payload"]["logit_bias"] = {"123": 100}
        api.write_text("".join(canonical(e)+"\n" for e in events))
        with self.assertRaisesRegex(ValueError, "generation budget"):
            self.prepare()
        api.write_text(original)
        events = deepcopy([r for _, r, _ in resume._records(api)])
        for event in events:
            event["identity"]["phase"] = "full_set_execution"
        api.write_text("".join(canonical(e)+"\n" for e in events))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare()

    def test_actual_retrieval_message_chain_is_eligible(self):
        self.search_second_step()
        self.assertEqual(self.prepare()["counts"]["bm25"], 1)

    def test_later_request_user_hint_is_rejected_even_with_real_usage(self):
        self.search_second_step(extra_hint=True)
        with self.assertRaisesRegex(ValueError, "extra hints"):
            self.prepare()

    def test_later_retrieval_content_must_equal_official_history(self):
        self.search_second_step(replace_history=True)
        with self.assertRaisesRegex(ValueError, "official history"):
            self.prepare()

    def test_source_or_imported_answer_tampering_is_rejected(self):
        manifest, _, _, rows = self.imported()
        altered = deepcopy(rows[0]); altered["response"] = "different answer"
        with self.assertRaisesRegex(ValueError, "changed response"):
            self.validate(altered, manifest)
        original, checkpoint, _ = self.rows[rows[0]["arm"]]
        altered = deepcopy(original); altered["response"] = "changed on disk"
        checkpoint.write_text(canonical(altered)+"\n")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.validate(rows[0], manifest)

    def test_partial_official_set_and_false_target_shard_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "500"):
            self.prepare(official_items=self.items[:499])
        manifest, _, _, rows = self.imported()
        altered = deepcopy(rows[0]); altered["shard"] = (altered["shard"] + 1) % 30
        with self.assertRaisesRegex(ValueError, "changed response"):
            self.validate(altered, manifest)


if __name__ == "__main__":
    unittest.main()

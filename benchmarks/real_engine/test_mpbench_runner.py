"""Runner contract checks only: mock transport, no benchmark scores or inference."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import asyncio

from benchmarks.real_engine.mpbench import ATTACK_CLASSES
from benchmarks.real_engine.mpbench_runner import (
    MPBenchIncomplete, complete_native_snapshot, run_item,
    security_denial, write_security_decision, require_score_integrity, run_mpbench,
)
from benchmarks.real_engine.hook_model import BudgetedProvider


class SnapshotEngine:
    """Test double solely for the complete-snapshot protocol."""
    def __init__(self, count=13):
        self.count = count
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        if args.get("view") == "coverage":
            return {"totalNodes": self.count}
        if args.get("view") == "timeline":
            return {"truncated": False, "returned": self.count,
                    "timeline": [{"memories": [{"id": str(i)} for i in range(self.count)]}]}
        return {"found": True, "node": {"id": args["id"], "content": "entry-" + args["id"]}}


class SnapshotTests(unittest.TestCase):
    def test_complete_snapshot_does_not_stop_at_default_recall_ten(self):
        engine = SnapshotEngine(13)
        entries, evidence = complete_native_snapshot(engine)
        self.assertEqual(len(entries), 13)
        self.assertIn("entry-12", entries)
        self.assertEqual(len([c for c in engine.calls if c[0] == "memory"]), 13)
        self.assertFalse(any(c[0] == "recall" for c in engine.calls))
        self.assertEqual(evidence["coverage"]["totalNodes"], 13)

    def test_bounded_snapshot_cannot_be_scored_as_complete(self):
        with self.assertRaises(MPBenchIncomplete):
            complete_native_snapshot(SnapshotEngine(200))
        class Truncated(SnapshotEngine):
            def call(self, name, args):
                result = super().call(name, args)
                if args.get("view") == "timeline":
                    result["truncated"] = True
                return result
        with self.assertRaises(MPBenchIncomplete):
            complete_native_snapshot(Truncated())

    def test_only_explicit_native_security_refusal_is_detector_positive(self):
        phrase = "Refused to store probable credential(s):"
        self.assertTrue(security_denial({"isError": True, "content": [{"type": "text", "text": phrase}]}))
        self.assertFalse(security_denial({"success": True, "tags": [phrase]}))
        self.assertIsNone(write_security_decision({"success": False, "error": "invalid arguments"}))
        self.assertFalse(write_security_decision({"success": True, "nodeId": "x"}))
        self.assertIsNone(write_security_decision(None))


class MockProvider:
    """Transport test double: produces no model output used as benchmark evidence."""
    def __init__(self, write=False):
        self.write = write
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(json.loads(json.dumps(messages)))
        if self.write and len(self.calls) == 1:
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "one", "type": "function", "function": {
                    "name": "memory_write", "arguments": json.dumps({"content": "raw saved record"})}}]}
        else:
            message = {"role": "assistant", "content": "test output"}
        return {"message": message, "usage": {"completion_tokens": 1}, "cost": 0,
                "model": "gpt-5.4-nano", "finish_reason": "stop"}


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.case = {"id": "adversarial/a", "kind": "adversarial", "attack_class": ATTACK_CLASSES[5],
                     "raw": {"id": "a", "user_query": "Summarize", "context": "raw context",
                             "skill_md": "raw external SKILL.md", "expected_memory": "GOLD TARGET",
                             "adversarial_goal": "GOLD GOAL"}}

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def campaign(self, provider):
        campaign = SimpleNamespace(runs=self.root, provider=provider,
                                   manifest={"binary_sha256": "test-only-hash"})
        campaign.item_provider = lambda meta, root, **kwargs: BudgetedProvider(provider, meta, root, **kwargs)
        return campaign

    async def test_raw_skill_exposure_same_across_baseline_arms_and_no_gold(self):
        rows = []
        for arm in ("bare", "bm25", "embedding"):
            provider = MockProvider()
            with patch("benchmarks.real_engine.mpbench_runner.scorer_identity", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.run.tokens", return_value=10):
                rows.append(await run_item(self.campaign(provider), self.case, arm, 0))
        self.assertEqual(len({r["prompt"][1]["content"] for r in rows}), 1)
        for row in rows:
            payload = json.loads(row["prompt"][1]["content"])
            self.assertEqual(payload["external_skill_md"], "raw external SKILL.md")
            self.assertNotIn("GOLD", row["prompt"][1]["content"])
            self.assertIsNone(row["defense_positive"])
            self.assertEqual(row["persisted_entries"], [])

    async def test_baseline_snapshots_actual_write_result_without_fabricating_defense(self):
        provider = MockProvider(write=True)
        with patch("benchmarks.real_engine.mpbench_runner.scorer_identity", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10):
            row = await run_item(self.campaign(provider), self.case, "bm25", 0)
        self.assertEqual(row["persisted_entries"], ["raw saved record"])
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNone(row["defense_positive"])
        self.assertFalse(row["defense_decision_coverage"])

    async def test_hooked_arm_keeps_sessionstart_memory_and_shares_model_budget(self):
        provider = MockProvider()
        class UnitHookedEngine(SnapshotEngine):
            def __init__(self, repo, directory, hook_model_callback, **kwargs):
                super().__init__(1)
                self.callback = hook_model_callback
                self.identity = {"fresh_store_nodes": 0, "hooks": {"enabled": True}}
                self.audit = Path(directory) / "mcp.jsonl"
                self.audit.parent.mkdir()
                self.audit.write_text("")
                self.tools = []
                self.finished = 0
                self.begin_inputs = []

            def __enter__(self):
                self.callback({"model": "gpt-5.4-nano", "hook": "unit-transport-only",
                               "messages": [{"role": "user", "content": "unit hook data"}],
                               "allowed_tools": [], "tool_call_budget": 0, "max_tokens": 10})
                return self

            def __exit__(self, *args):
                pass

            def begin_turn(self, prompt):
                self.begin_inputs.append(prompt)
                return ["source hook context"]

            def drain_context(self):
                return []

            def finish_turn(self, messages):
                self.finished += 1
                return {"blocked": self.finished == 1, "reasons": ["unit retry"], "context": []}
        with patch("benchmarks.real_engine.mpbench_runner.scorer_identity", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10), \
             patch("benchmarks.real_engine.engine.HookedRealEngine", UnitHookedEngine):
            row = await run_item(self.campaign(provider), self.case, "vestige", 0)
        self.assertEqual(row["persisted_entries"], ["entry-0"])
        self.assertEqual(len(row["usage"]), 3)  # one hook plus two solver calls
        self.assertEqual(row["all_model_strings"], ["gpt-5.4-nano"])
        self.assertTrue(row["engine"]["hooks"]["enabled"])
        self.assertIn("source hook context", json.dumps(row["transcript"]))

    async def test_pending_audit_prevents_scoring_even_with_existing_outputs(self):
        with patch("benchmarks.real_engine.mpbench_runner.integrity_state", return_value={
            "status": "PENDING", "ready_to_score": False, "benchmarks": {"mpbench": {"status": "PENDING"}}}):
            with self.assertRaisesRegex(MPBenchIncomplete, "scoring held"):
                require_score_integrity(self.campaign(MockProvider()), {"arm": "bare", "run_identity": "unqualified"})

    async def test_twenty_item_preflight_preserves_full_manifest_and_never_scores(self):
        campaign = self.campaign(MockProvider())
        campaign.preflight_count = 20
        campaign.progress = {}
        campaign.ledger = SimpleNamespace(snapshot=lambda: {"remaining": 25})
        (self.root / "QUALIFICATION.json").write_text(json.dumps({"hooks_manifest_sha256": "unit-only"}))
        fixtures = [{**self.case, "id": f"adversarial/unit-{i}"} for i in range(30)]
        observed = []
        async def unit_run(campaign, case, arm, worker, trial=0, attempt=0):
            observed.append((case["id"], arm, worker))
            await asyncio.sleep(0)
            return {"item_id": case["id"], "arm": arm, "trial": trial, "status": "complete", "cost": 0}
        with patch("benchmarks.real_engine.mpbench_runner.scorer_identity", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.mpbench_runner.load_mpbench", return_value=(fixtures, {"unit_fixture": True})), \
             patch("benchmarks.real_engine.mpbench_runner.run_item", side_effect=unit_run), \
             patch("benchmarks.real_engine.mpbench_runner.integrity_state", return_value={"status": "PENDING"}), \
             patch("benchmarks.real_engine.mpbench_runner._score", side_effect=AssertionError("preflight must not score")):
            outcome = await run_mpbench(campaign)
        self.assertFalse(outcome["official_metrics_computed"])
        self.assertEqual(outcome["full_manifest_items"], 30)
        self.assertEqual(len(observed), 80)
        self.assertEqual(set(outcome["completed_preflight_items_per_arm"].values()), {20})
        shards = json.loads((self.root / "mpbench/shards.json").read_text())
        self.assertEqual(len(shards["items"]), 30)
        self.assertEqual(shards["workers"], 10)
        coverage = json.loads((self.root / "mpbench/coverage.json").read_text())
        self.assertEqual({a["uncompleted_n"] for a in coverage["arms"].values()}, {10})
        self.assertFalse((self.root / "mpbench/RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()

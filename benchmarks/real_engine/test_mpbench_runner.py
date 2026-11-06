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
    security_denial, write_security_decision, require_score_integrity, run_mpbench, score_merged_arm,
)
from benchmarks.real_engine.hook_model import BudgetedProvider


class SnapshotEngine:
    """Test double solely for the complete-snapshot protocol."""
    def __init__(self, count=13):
        self.count = count
        self.entries = ["entry-" + str(i) for i in range(count)]
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        if name == "smart_ingest":
            self.entries.append(args["content"])
            self.count = len(self.entries)
            return {"success": True, "nodeId": str(self.count-1)}
        if args.get("view") == "coverage":
            return {"totalNodes": self.count}
        if args.get("view") == "timeline":
            return {"truncated": False, "returned": self.count,
                    "timeline": [{"memories": [{"id": str(i)} for i in range(self.count)]}]}
        return {"found": True, "node": {"id": args["id"], "content": self.entries[int(args["id"])]}}


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


class EpisodeEngine(SnapshotEngine):
    """Explicit unit protocol fixture; never accepted as runtime evidence."""
    instances = []
    def __init__(self, repo, directory, **kwargs):
        super().__init__(1)
        self.identity = {"fresh_store_nodes": 0, "hooks": {"enabled": True}, "data_dir": str(directory)}
        self.audit = Path(directory) / "mcp.jsonl"
        self.audit.parent.mkdir()
        self.audit.write_text(json.dumps({"kind": "tool_request", "name": "memory_status"}) + "\n")
        self.tools = []
        self.closed = False
        EpisodeEngine.instances.append(self)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def begin_turn(self, prompt):
        return []
    def finish_turn(self, messages):
        return {"blocked": False, "context": []}
    def drain_context(self):
        return []


class EpisodeProvider(MockProvider):
    """Finite scripted transport outputs for episode-boundary regression tests."""
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
    async def chat(self, messages, **kwargs):
        self.calls.append(json.loads(json.dumps(messages)))
        if len(self.calls) > len(self.responses):
            raise AssertionError("unexpected extra model call")
        message, reason = self.responses[len(self.calls)-1]
        return {"message": message, "usage": {"completion_tokens": 1}, "cost": 0,
                "model": "gpt-5.4-nano", "finish_reason": reason}


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
            with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.run.tokens", return_value=10):
                rows.append(await run_item(self.campaign(provider), self.case, arm, 0))
        self.assertEqual(len({r["prompt"][1]["content"] for r in rows}), 1)
        for row in rows:
            payload = json.loads(row["prompt"][1]["content"])
            self.assertEqual(payload["external_skill_md"], "raw external SKILL.md")
            self.assertNotIn("GOLD", row["prompt"][1]["content"])
            self.assertIsNone(row["defense_positive"])
            self.assertEqual(row["persisted_entries"], [] if row["arm"] == "bare" else [row["prompt"][1]["content"]])

    async def test_baseline_snapshots_actual_write_result_without_fabricating_defense(self):
        provider = MockProvider(write=True)
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10):
            row = await run_item(self.campaign(provider), self.case, "bm25", 0)
        self.assertEqual(row["persisted_entries"], [row["prompt"][1]["content"], "raw saved record"])
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNone(row["defense_positive"])
        self.assertFalse(row["defense_decision_coverage"])

    async def test_every_memory_arm_initially_stores_exact_same_raw_input(self):
        import hashlib
        rows = []
        for arm in ("bare", "bm25", "embedding", "vestige"):
            provider = MockProvider()
            with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.engine.HookedRealEngine", EpisodeEngine), \
                 patch("benchmarks.real_engine.run.tokens", return_value=10):
                row = await run_item(self.campaign(provider), self.case, arm, 0)
            rows.append(row)
            raw = row["prompt"][1]["content"]
            exposure = row["normal_write_exposure"]
            self.assertEqual(exposure["content_sha256"], hashlib.sha256(raw.encode()).hexdigest())
            self.assertEqual(exposure["source"], "user-input")
            self.assertEqual(exposure["attempted"], arm != "bare")
            self.assertEqual(len(provider.calls[0]), 2)  # no fabricated admission prelude
            self.assertNotIn("GOLD", raw)
            if arm == "bare":
                self.assertEqual(row["persisted_entries"], [])
            else:
                self.assertIn(raw, row["persisted_entries"])
                self.assertEqual(row["write_decisions"][0]["result"], exposure["result"])
                if arm == "vestige":
                    initial_calls = [(name, args) for name, args in EpisodeEngine.instances[-1].calls if name == "smart_ingest"]
                    self.assertEqual(initial_calls, [("smart_ingest", {"content": raw, "node_type": "note", "tags": ["history"], "source": "user-input"})])
                else:
                    self.assertEqual(exposure["result"], {"success": True, "id": "memory-0"})
        self.assertEqual(len({r["prompt"][1]["content"] for r in rows}), 1)

    async def test_initial_native_security_denial_is_observed_and_not_prompt_guidance(self):
        from benchmarks.real_engine.engine import VestigeMCPError, VestigeMCPToolError
        denial = {"isError": True, "content": [{"type": "text", "text": "Refused to store probable credential(s): unit"}]}
        class DenyingEngine(EpisodeEngine):
            dead = False
            def call(self, name, arguments):
                if name == "smart_ingest":
                    if self.dead:
                        raise VestigeMCPError("benchmark engine is not running")
                    raise VestigeMCPToolError(name, denial)
                return super().call(name, arguments)
        for dead in (False, True):
            provider = MockProvider()
            DenyingEngine.dead = dead
            with self.subTest(dead=dead), \
                 patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.engine.HookedRealEngine", DenyingEngine), \
                 patch("benchmarks.real_engine.run.tokens", return_value=10):
                if dead:
                    with self.assertRaises(VestigeMCPError):
                        await run_item(self.campaign(provider), self.case, "vestige", 0)
                    self.assertEqual(provider.calls, [])
                else:
                    row = await run_item(self.campaign(provider), self.case, "vestige", 0)
                    self.assertTrue(row["normal_write_exposure"]["defense_positive"])
                    self.assertEqual(row["normal_write_exposure"]["result"], denial)
                    self.assertTrue(row["defense_positive"])
                    self.assertTrue(row["defense_decision_coverage"])
                    self.assertEqual(row["persisted_entries"], ["entry-0"])
                    self.assertNotIn(denial["content"][0]["text"], json.dumps(provider.calls[0]))

    async def test_hooked_arm_keeps_sessionstart_memory_and_shares_model_budget(self):
        provider = MockProvider()
        class UnitHookedEngine(SnapshotEngine):
            def __init__(self, repo, directory, hook_model_callback, **kwargs):
                super().__init__(1)
                self.callback = hook_model_callback
                self.identity = {"fresh_store_nodes": 0, "hooks": {"enabled": True}}
                self.audit = Path(directory) / "mcp.jsonl"
                self.audit.parent.mkdir()
                self.audit.write_text(json.dumps({"kind": "tool_request", "name": "memory_status"}) + "\n")
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
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10), \
             patch("benchmarks.real_engine.engine.HookedRealEngine", UnitHookedEngine):
            row = await run_item(self.campaign(provider), self.case, "vestige", 0)
        self.assertEqual(row["persisted_entries"], ["entry-0", row["prompt"][1]["content"]])
        self.assertEqual(len(row["usage"]), 3)  # one hook plus two solver calls
        self.assertEqual(row["all_model_strings"], ["gpt-5.4-nano"])
        self.assertTrue(row["engine"]["hooks"]["enabled"])
        self.assertIn("source hook context", json.dumps(row["transcript"]))

    async def test_pending_audit_prevents_scoring_even_with_existing_outputs(self):
        with patch("benchmarks.real_engine.mpbench_runner.integrity_state", return_value={
            "status": "PENDING", "ready_to_score": False, "benchmarks": {"mpbench": {"status": "PENDING"}}}):
            with self.assertRaisesRegex(MPBenchIncomplete, "scoring held"):
                require_score_integrity(self.campaign(MockProvider()), {"arm": "bare", "run_identity": "unqualified"})

    async def test_ten_item_preflight_preserves_full_manifest_and_never_scores(self):
        campaign = self.campaign(MockProvider())
        campaign.preflight_count = 10
        campaign.progress = {}
        campaign.ledger = SimpleNamespace(snapshot=lambda: {"remaining": 25})
        (self.root / "QUALIFICATION.json").write_text(json.dumps({"hooks_manifest_sha256": "unit-only"}))
        fixtures = [{**self.case, "id": f"adversarial/unit-{i}"} for i in range(30)]
        observed = []
        async def unit_run(campaign, case, arm, worker, trial=0, attempt=0):
            observed.append((case["id"], arm, worker))
            await asyncio.sleep(0)
            return {"item_id": case["id"], "arm": arm, "trial": trial, "status": "complete", "cost": 0}
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.mpbench_runner.load_mpbench", return_value=(fixtures, {"unit_fixture": True})), \
             patch("benchmarks.real_engine.mpbench_runner.run_item", side_effect=unit_run), \
             patch("benchmarks.real_engine.mpbench_runner.integrity_state", return_value={"status": "PENDING"}), \
             patch("benchmarks.real_engine.mpbench_runner._score", side_effect=AssertionError("preflight must not score")):
            outcome = await run_mpbench(campaign)
        self.assertFalse(outcome["official_metrics_computed"])
        self.assertEqual(outcome["full_manifest_items"], 30)
        self.assertEqual(len(observed), 40)
        self.assertEqual(set(outcome["completed_preflight_items_per_arm"].values()), {10})
        shards = json.loads((self.root / "mpbench/shards.json").read_text())
        self.assertEqual(len(shards["items"]), 30)
        self.assertEqual(shards["workers"], 10)
        coverage = json.loads((self.root / "mpbench/coverage.json").read_text())
        self.assertEqual({a["uncompleted_n"] for a in coverage["arms"].values()}, {20})
        self.assertFalse((self.root / "mpbench/RESULTS.md").exists())

    async def test_current_profile_and_config_reach_every_item_record(self):
        campaign = self.campaign(MockProvider())
        campaign.config_hash = "current-config"
        campaign.preflight_for = lambda name: 10
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10):
            row = await run_item(campaign, self.case, "bare", 0)
        self.assertEqual(row["config_hash"], "current-config")
        self.assertEqual(row["phase"], "integrity_preflight")

    async def test_length_snapshots_native_state_without_executing_incomplete_call(self):
        partial = {"role": "assistant", "content": "partial actual response", "tool_calls": [{
            "id": "incomplete", "type": "function", "function": {"name": "smart_ingest", "arguments": '{"content":'}}]}
        provider = EpisodeProvider([(partial, "length")])
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.engine.HookedRealEngine", EpisodeEngine), \
             patch("benchmarks.real_engine.run.tokens", return_value=10):
            row = await run_item(self.campaign(provider), self.case, "vestige", 0)
        self.assertEqual(row["termination"], "output_limit")
        self.assertEqual(row["response"], "partial actual response")
        self.assertEqual(row["persisted_entries"], ["entry-0", row["prompt"][1]["content"]])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(sum(name == "smart_ingest" for name, _ in EpisodeEngine.instances[-1].calls), 1)
        self.assertTrue(EpisodeEngine.instances[-1].closed)

    async def test_iteration_and_context_limits_keep_actual_store(self):
        call = {"role": "assistant", "content": None, "tool_calls": [{"id": "read", "type": "function", "function": {
            "name": "memory", "arguments": json.dumps({"action": "get", "id": "0"})}}]}
        for context_count, expected, max_calls in ((10, "iteration_limit", 6), (32769, "context_limit", 0)):
            provider = EpisodeProvider([(call, "tool_calls")]*6)
            with self.subTest(termination=expected), \
                 patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.engine.HookedRealEngine", EpisodeEngine), \
                 patch("benchmarks.real_engine.run.tokens", return_value=context_count):
                row = await run_item(self.campaign(provider), self.case, "vestige", 0)
            self.assertEqual(row["termination"], expected)
            self.assertEqual(row["persisted_entries"], ["entry-0", row["prompt"][1]["content"]])
            self.assertEqual(row["response"], "")
            self.assertEqual(len(provider.calls), max_calls)

    async def test_aggregate_output_cap_snapshots_exact_completed_writes(self):
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "write", "type": "function", "function": {
            "name": "memory_write", "arguments": json.dumps({"content": "actual written record"})}}]}
        class FullOutputProvider(EpisodeProvider):
            async def chat(self, messages, **kwargs):
                self_limit = kwargs["max_tokens"]
                if self_limit != 512:
                    raise AssertionError("per-request output cap changed")
                result = await super().chat(messages, **kwargs)
                result["usage"]["completion_tokens"] = 512
                return result
        provider = FullOutputProvider([(message, "tool_calls")]*4)
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.run.tokens", return_value=10):
            row = await run_item(self.campaign(provider), self.case, "bm25", 0)
        self.assertEqual(row["termination"], "output_limit")
        self.assertEqual(row["persisted_entries"], [row["prompt"][1]["content"]]+["actual written record"]*4)
        self.assertEqual(sum(u["completion_tokens"] for u in row["usage"]), 2048)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(row["response"], "")

    async def test_native_error_envelope_forwarded_but_dead_transport_raises(self):
        from benchmarks.real_engine.engine import VestigeMCPError, VestigeMCPToolError
        call = {"role": "assistant", "content": None, "tool_calls": [{"id": "read", "type": "function", "function": {
            "name": "recall", "arguments": json.dumps({"handle": "missing"})}}]}
        envelope = {"isError": True, "content": [{"type": "text", "text": "native exact handle unavailable"}]}
        class ToolErrorEngine(EpisodeEngine):
            dead = False
            def call(self, name, arguments):
                if name == "recall":
                    if self.dead:
                        raise VestigeMCPError("benchmark engine is not running")
                    raise VestigeMCPToolError(name, envelope)
                return super().call(name, arguments)
        for dead in (False, True):
            provider = EpisodeProvider([(call, "tool_calls"), ({"role": "assistant", "content": "actual answer"}, "stop")])
            ToolErrorEngine.dead = dead
            with self.subTest(dead=dead), \
                 patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
                 patch("benchmarks.real_engine.engine.HookedRealEngine", ToolErrorEngine), \
                 patch("benchmarks.real_engine.run.tokens", return_value=10):
                if dead:
                    with self.assertRaises(VestigeMCPError):
                        await run_item(self.campaign(provider), self.case, "vestige", 0)
                    self.assertEqual(len(provider.calls), 1)
                else:
                    row = await run_item(self.campaign(provider), self.case, "vestige", 0)
                    visible = [m for m in provider.calls[1] if m.get("role") == "tool"]
                    self.assertEqual(json.loads(visible[0]["content"]), envelope)
                    self.assertEqual(row["termination"], "answer")
                    self.assertFalse(row["defense_positive"])

    async def test_midrun_engine_death_aborts_before_solver_request(self):
        from benchmarks.real_engine.engine import VestigeMCPError
        provider = MockProvider()
        class KillProtocolEngine:
            def __init__(self, repo, directory, **kwargs):
                self.identity = {"fresh_store_nodes": 0, "hooks": {"enabled": True}, "data_dir": str(directory)}
                self.audit = Path(directory) / "mcp.jsonl"
                self.tools = []
                self.dead = False
                self.closed = False
                self.client = SimpleNamespace(_proc=SimpleNamespace(pid=123, kill=self.kill, wait=lambda **kwargs: -9))
            def kill(self):
                self.dead = True
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.closed = True
            def begin_turn(self, prompt):
                return []
            def call(self, *args):
                if self.dead:
                    raise VestigeMCPError("benchmark engine is not running")
                if args[0] == "smart_ingest":
                    return {"success": True, "nodeId": "initial"}
                raise AssertionError("control called memory before kill")
        campaign = self.campaign(provider)
        campaign.config_hash = "current-config"
        with patch("benchmarks.real_engine.mpbench_runner.frozen_scorer", return_value={"scorer_sha256": "frozen"}), \
             patch("benchmarks.real_engine.engine.HookedRealEngine", KillProtocolEngine):
            outcome = await run_item(campaign, self.case, "vestige", 0, kill_control=True)
        self.assertEqual(outcome["status"], "kill_control_passed")
        self.assertEqual(provider.calls, [])
        receipts = [json.loads(line) for line in (self.root / "mpbench/kill-control.jsonl").read_text().splitlines()]
        self.assertEqual([r["event"] for r in receipts], ["engine_started", "midrun_kill", "runner_aborted"])
        self.assertFalse(receipts[-1]["fallback_used"])
        self.assertEqual({r["config_hash"] for r in receipts}, {"current-config"})

    async def test_completed_merged_scorer_is_not_run_twice(self):
        campaign = self.campaign(MockProvider())
        arm_root = self.root / "mpbench/bare"
        arm_root.mkdir(parents=True)
        merged = [{"item_id": self.case["id"], "trial": 0, "run_identity": "current"}]
        (arm_root / "merged.jsonl").write_text(json.dumps(merged[0]) + "\n")
        (self.root / "QUALIFICATION.json").write_text(json.dumps({"benchmarks": {"mpbench": {
            "scorer": {"source_files": {"benchmarks/real_engine/mpbench.py": "frozen"}}}}}))
        async def one_score(*args):
            return {**merged[0], "kind": "adversarial", "attack_class": self.case["attack_class"], "asr": False}
        with patch("benchmarks.real_engine.mpbench_runner._score", side_effect=one_score) as scorer:
            initial = await score_merged_arm(campaign, [self.case], "bare", merged, {})
            resumed = await score_merged_arm(campaign, [self.case], "bare", merged, {})
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(initial, resumed)
        receipts = [json.loads(line) for line in (arm_root / "scorer-invocations.jsonl").read_text().splitlines()]
        self.assertEqual([r["event"] for r in receipts], ["official_scorer_start", "official_scorer_complete"])
        self.assertEqual(receipts[-1]["output_count"], 1)
        (arm_root / "scored.jsonl").write_text(json.dumps({**initial[0], "asr": True}) + "\n")
        with self.assertRaisesRegex(MPBenchIncomplete, "output changed"):
            await score_merged_arm(campaign, [self.case], "bare", merged, {})


if __name__ == "__main__":
    unittest.main()

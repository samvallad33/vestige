"""Zero-API protocol tests; scripted responses are unit fixtures, never scores."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.real_engine.hook_model import BudgetedProvider, ItemModelBudget

from benchmarks.real_engine.security_agents import (
    DOJO_PIN, INJEC_PIN, InjecAgentSource, SecurityMemory, check_model,
    dojo_cases, official_pure_functions, run_agentdojo_case, run_injecagent_case,
    score_agentdojo_merged, score_injecagent_merged,
)

REPO = Path(__file__).resolve().parents[2]
INJEC = REPO / "runs/upstream/InjecAgent"
DOJO = Path(os.environ.get("AGENTDOJO_SOURCE", str(REPO / "runs/upstream/agentdojo")))


class ScriptedUnitProvider:
    """No networking; validates adapter protocol plumbing only."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), **kwargs})
        response = self.responses.pop(0)
        return {"message": {"role": "assistant", **response}, "model": "gpt-5.4-nano-unit-fixture",
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}, "cost": 0.001,
                "finish_reason": "stop"}


def unit_bound(provider, root, **kwargs):
    return BudgetedProvider(provider, {"unit_fixture": True}, Path(root), max_calls=kwargs.get("max_calls", 40),
        max_output=16384, context=32768)


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_other_model_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "other than"):
            check_model({"model": "some-other-model"})

    async def test_missing_native_engine_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("benchmarks.real_engine.security_agents.HookedRealEngine", side_effect=RuntimeError("native absent")):
                with self.assertRaisesRegex(RuntimeError, "native absent"):
                    async with SecurityMemory("vestige", REPO, Path(tmp), Path(tmp) / "manifest.json", unit_bound(ScriptedUnitProvider([]), tmp)):
                        self.fail("entered memory without native binary")

    async def test_hooks_and_solver_share_one_item_call_budget(self):
        provider = ScriptedUnitProvider([{"content": "solver"}, {"content": "hook"}])
        with tempfile.TemporaryDirectory() as tmp:
            bound = unit_bound(provider, tmp, max_calls=2)
            await bound.chat([{"role": "user", "content": "unit"}])
            callback = bound.hook_callback()
            await bound.hook_request({"model": "gpt-5.4-nano", "messages": [{"role": "user", "content": "unit-hook"}],
                "hook": "unit", "max_tokens": 10}, callback)
            self.assertEqual(bound.cost, 0.002)
            self.assertEqual(callback.cost, 0.001)
            self.assertEqual(len(bound.usage), 2)
            with self.assertRaises(ItemModelBudget):
                await bound.chat([{"role": "user", "content": "unit-over-cap"}])
        self.assertEqual(len(provider.requests), 2)

    async def test_bare_does_not_create_memory_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            async with SecurityMemory("bare", REPO, Path(tmp), Path(tmp) / "manifest.json", unit_bound(ScriptedUnitProvider([]), tmp)) as memory:
                await memory.observe("raw untrusted unit fixture")
                self.assertEqual(memory.tools, [])
                with self.assertRaisesRegex(RuntimeError, "never falls back"):
                    await memory.call("recall", {"handle": "history"})


@unittest.skipUnless(INJEC.is_dir(), "pinned official InjecAgent source required")
class InjecOfficialTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = InjecAgentSource(INJEC)

    def test_all1054_and_prompt_parity_and_no_scoring_annotations(self):
        self.assertEqual(len(self.source.cases), 1054)
        self.assertEqual(len({c["id"] for c in self.source.cases}), 1054)
        case = self.source.cases[0]
        previous = None
        for arm in ("bare", "bm25", "embedding", "vestige"):
            messages, tools = self.source.initial_messages(case, arm)
            self.assertEqual(messages[1]["content"], case["raw"]["User Instruction"])
            self.assertEqual(messages[-1]["content"], json.dumps(case["raw"]["Tool Response"]))
            self.assertEqual(len(tools), 1 + len(case["raw"]["Attacker Tools"]))
            neutral_removed = messages[0]["content"].rsplit("\n", 1)[0]
            self.assertEqual(neutral_removed, self.source.prompts.PROMPT_DICT["InjecAgent"][0])
            if previous:
                self.assertEqual(messages[1:], previous[1:])
            previous = messages
        # Targets belong to the evaluator; loading them is not a memory write.
        for key in ("Expected Achievements", "Attacker Instruction", "Thought", "Tool Response Template"):
            mutated = copy.deepcopy(case)
            mutated["raw"][key] = "SCORING-ONLY-SENTINEL"
            messages, tools = self.source.initial_messages(mutated, "vestige")
            self.assertNotIn("SCORING-ONLY-SENTINEL", json.dumps(messages))

    def test_gold_resolvers_keep_evaluator_fields_separate(self):
        from benchmarks.real_engine.security_runner import official_gold_resolver
        with tempfile.TemporaryDirectory() as tmp:
            upstream = Path(tmp)
            (upstream / "InjecAgent").symlink_to(INJEC, target_is_directory=True)
            if DOJO.exists():
                (upstream / "agentdojo").symlink_to(DOJO, target_is_directory=True)
            result = official_gold_resolver("injecagent", self.source.cases[0]["id"], upstream)
            self.assertTrue(result["audit_only"])
            self.assertIn("Attacker Instruction", result["gold_fields"])
            self.assertNotIn("Attacker Instruction", result["permitted_official_inputs"])
            if DOJO.exists():
                case = next(c for c in dojo_cases(DOJO) if c["phase"] == "clean")
                result = official_gold_resolver("agentdojo", case["id"], upstream)
                self.assertTrue(result["audit_only"])
                self.assertIsInstance(result["gold_fields"]["official_ground_truth_calls"], list)
                self.assertNotIn("official_ground_truth_calls", result["permitted_official_inputs"])

    async def test_raw_provider_is_rejected_without_combined_hook_budget(self):
        provider = ScriptedUnitProvider([])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TypeError, "bound budget"):
                await run_injecagent_case(self.source, self.source.cases[0], "vestige",
                    provider=provider, repo=REPO, root=Path(tmp) / "case", manifest=Path(tmp) / "missing.json")
        self.assertFalse(provider.requests)

    async def test_direct_harm_official_parser_no_real_actions(self):
        case = self.source.cases[0]
        name = case["raw"]["Attacker Tools"][0]
        provider = ScriptedUnitProvider([{"content": None, "tool_calls": [{"id": "unit-call", "type": "function",
            "function": {"name": name, "arguments": "{}"}}]}])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"binary_sha256": "unit-fixture"}))
            row = await run_injecagent_case(self.source, case, "bare", provider=unit_bound(provider, tmp),
                repo=REPO, root=Path(tmp) / "case", manifest=manifest)
            self.assertEqual(row["official_record"]["eval"], "succ")
            self.assertEqual(len(provider.requests), 1)
            self.assertIsNone(row["score"])

    async def test_data_stealing_two_stages_and_metered_simulator(self):
        case = next(c for c in self.source.cases if c["attack"] == "ds")
        first, second = case["raw"]["Attacker Tools"]
        parameters = '{"unittest_parameter":"cache_miss_20261101"}'
        provider = ScriptedUnitProvider([
            {"content": None, "tool_calls": [{"id": "unit-stage1", "type": "function",
                "function": {"name": first, "arguments": parameters}}]},
            {"content": '```json\n{"unit":"simulated"}\n```'},
            {"content": None, "tool_calls": [{"id": "unit-stage2", "type": "function",
                "function": {"name": second, "arguments": "{}"}}]},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"binary_sha256": "unit-fixture"}))
            row = await run_injecagent_case(self.source, case, "bare", provider=unit_bound(provider, tmp),
                repo=REPO, root=Path(tmp) / "case", manifest=manifest)
            self.assertEqual(row["official_record"]["eval"], "succ")
            self.assertEqual(row["official_record"]["eval Step 2"], "succ")
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual(provider.requests[1]["identity"]["phase"], "official_simulated_tool_response")
            self.assertEqual(row["cost"], 0.003)
            # Preserve the upstream stage2 parameter-string serialization.
            replay = provider.requests[2]["messages"][-2]
            self.assertEqual(replay["tool_calls"][0]["function"]["arguments"], json.dumps(parameters))

    def test_merged_scorer_coverage_and_cache(self):
        rows = [{"item_id": c["id"], "attack": c["attack"], "status": "complete",
                 "official_record": {"eval": "unsucc"}} for c in self.source.cases]
        with tempfile.TemporaryDirectory() as tmp:
            expected = [r["item_id"] for r in rows]
            result = score_injecagent_merged(self.source, rows, Path(tmp), expected)
            self.assertEqual(result["scores"]["#Test Case"], 1054)
            self.assertEqual(result["scores"]["ASR-all (Total)"], "0.0")
            self.assertEqual(score_injecagent_merged(self.source, rows, Path(tmp), expected), result)
            with self.assertRaisesRegex(RuntimeError, "missing, duplicate"):
                score_injecagent_merged(self.source, rows[:1], Path(tmp), expected)
            with self.assertRaisesRegex(RuntimeError, "missing, duplicate"):
                score_injecagent_merged(self.source, rows + rows[:1], Path(tmp), expected)


@unittest.skipUnless(DOJO.is_dir(), "pinned official AgentDojo source required")
class DojoOfficialTests(unittest.IsolatedAsyncioTestCase):
    def test_official_all_suites_cases(self):
        cases = dojo_cases(DOJO)
        self.assertEqual(len(cases), 1081)
        self.assertEqual({c["suite"] for c in cases}, {"workspace", "travel", "banking", "slack"})
        self.assertEqual(sum(c["phase"] == "direct" for c in cases), 949)
        self.assertEqual(sum(c["phase"] == "clean" for c in cases), 97)
        self.assertEqual(sum(c["phase"] == "injection_utility" for c in cases), 35)

    async def test_real_official_environment_with_no_network_fixture(self):
        case = next(c for c in dojo_cases(DOJO) if c["phase"] == "clean" and c["suite"] == "banking")
        provider = ScriptedUnitProvider([{"content": "Unit fixture finished without taking actions."}])
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"binary_sha256": "unit-fixture"}))
            row = await run_agentdojo_case(DOJO, case, "bare", provider=unit_bound(provider, tmp), repo=REPO,
                root=Path(tmp) / "case", manifest=manifest)
            self.assertIsInstance(row["utility"], bool)
            self.assertEqual(len(provider.requests), 1)
            self.assertEqual(row["dataset_revision"], DOJO_PIN)
            self.assertEqual(row["phase"], "clean")

    def test_official_aggregate_does_not_count_clean_security_default_as_asr(self):
        rows = [{"item_id": c["id"], "status": "complete", "phase": c["phase"], "utility": False,
                 "security": c["phase"] != "direct"} for c in dojo_cases(DOJO)]
        with tempfile.TemporaryDirectory() as tmp:
            result = score_agentdojo_merged(DOJO, rows, Path(tmp), [r["item_id"] for r in rows])
            self.assertEqual(result["metrics"]["direct"]["attack_success_rate"], 0.0)
            self.assertNotIn("attack_success_rate", result["metrics"]["clean"])
            with self.assertRaisesRegex(RuntimeError, "subset scoring forbidden"):
                score_agentdojo_merged(DOJO, rows[:1], Path(tmp), [rows[0]["item_id"]])


class SecurityRunnerGateTests(unittest.IsolatedAsyncioTestCase):
    def test_preflight20_is_deterministic_and_does_not_replace_full_ids(self):
        from benchmarks.real_engine.security_runner import preflight_ids
        cases = [{"id": str(i)} for i in range(1054)]
        result = preflight_ids(cases)
        self.assertEqual(result, preflight_ids(cases))
        self.assertEqual(len(set(result)), 20)
        self.assertEqual(len(cases), 1054)

    async def test_missing_integrity_callback_never_passes_or_writes_results(self):
        from types import SimpleNamespace
        from benchmarks.real_engine.security_runner import final_integrity, source_pins
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = await final_integrity(SimpleNamespace(), "injecagent", root, source_pins(), "unit")
            self.assertFalse(result)
            self.assertEqual(json.loads((root / "integrity-unit.json").read_text())["status"], "PENDING")
            self.assertFalse((root / "RESULTS.md").exists())

    async def test_wrong_hash_audit_receipt_marks_invalid(self):
        from types import SimpleNamespace
        from benchmarks.real_engine.security_runner import final_integrity, source_pins
        def callback(**kwargs):
            return {"status": "PASS", "source_pins": {"wrong": "unit"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = await final_integrity(SimpleNamespace(security_integrity_audit=callback), "agentdojo", root, source_pins(), "unit")
            self.assertFalse(result)
            self.assertTrue((root / "INVALID.json").exists())

    async def test_unqualified_runner_never_constructs_item_provider(self):
        from types import SimpleNamespace
        from benchmarks.real_engine.security_runner import run_security
        with tempfile.TemporaryDirectory() as tmp:
            def denied(*args, **kwargs):
                raise AssertionError("must not request a model without qualification")
            campaign = SimpleNamespace(runs=Path(tmp), item_provider=denied)
            with self.assertRaisesRegex(RuntimeError, "HOOK_AUDIT_REQUIRED"):
                await run_security(campaign, "injecagent")


if __name__ == "__main__":
    unittest.main()

"""No-inference checks: official observations, chronology, and failure behavior.

Scripted fixture actions below test integration only. They are never benchmark
scores and never appear in campaign RESULTS.md.
"""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .extra_datasets import load_pmbench_scenario, score_pmbench
from .pmbench_runner import (PMActionBridge, PMActionError, run_pmbench_trial, simulator_module,
                            load_pmbench_audit_items, replay_pmbench_visible)
from .hook_model import BudgetedProvider


UPSTREAM = Path(__file__).resolve().parents[2] / "runs/upstream"


class FixtureProvider:
    """Transport fixture only; no engine or claimed memory behavior."""
    def __init__(self, response=None):
        self.prompts = []
        self.response = response or json.dumps({"action": "choose", "choice": "A", "task_ids": [], "channel": "NONE"})

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages)
        return {"message": {"role": "assistant", "content": self.response}, "cost": 0,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "finish_reason": "stop",
                "model": "SCRIPTED_INTEGRATION_FIXTURE"}


@unittest.skipUnless((UPSTREAM / "pm-bench/sim/pm_bench.py").exists(), "official PMBench source not downloaded")
class PMBenchIntegrationTests(unittest.TestCase):
    def test_preflight_collects10_distinct_steps_and_replays_visible_source(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "upstream").symlink_to(UPSTREAM)
                provider = FixtureProvider()
                campaign = SimpleNamespace(runs=root, manifest={"binary_sha256": "fixture-no-engine"}, provider=provider)
                campaign.item_provider = lambda meta, path, **kw: BudgetedProvider(provider, meta, path, **kw)
                accepted = []
                row = await run_pmbench_trial(campaign, "bare", 0, 0, preflight_steps=10,
                                              accepted_step_callback=accepted.append)
                self.assertEqual(row["status"], "raw_preflight_complete")
                self.assertEqual(len(accepted), 10)
                self.assertEqual(len({x["item_id"] for x in accepted}), 10)
                self.assertTrue(all(x["score"] is None for x in accepted))
                replay = replay_pmbench_visible(UPSTREAM, Path(row["raw_log"]).parent / "actions.jsonl", "bare")
                self.assertEqual(replay["verified_action_requests"], 10)
                self.assertIn("not_score", replay["classification"])
        asyncio.run(run())

    def test_audit_resolver_preserves240_trial_encoded_official_step_ids(self):
        items = load_pmbench_audit_items(UPSTREAM)
        self.assertEqual(len(items), 240)
        self.assertEqual(len({row["id"] for row in items}), 240)
        self.assertTrue(all("groundtruth" not in "\n".join(row["history"]).lower() for row in items))

    def test_full_week_fixture_preserves_official_observation_surface(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "upstream").symlink_to(UPSTREAM)
                provider = FixtureProvider()
                campaign = SimpleNamespace(runs=root, manifest={"binary_sha256": "fixture-no-engine"}, provider=provider)
                campaign.item_provider = lambda meta, path, **kw: BudgetedProvider(provider, meta, path, **kw)
                row = await run_pmbench_trial(campaign, "bare", 0, 0)
                self.assertEqual(len(row["entries"]), 80)
                self.assertEqual(len(provider.prompts), 80)
                self.assertIsNone(row["engine"])
                # Day seven is not visible before day one; internal correctness
                # labels and due_now never reach the model transport.
                first = json.dumps(provider.prompts[0])
                self.assertNotIn("due_now", first)
                self.assertNotIn("groundtruth", first.lower())
                self.assertNotIn("correct_task_ids", first)
                self.assertEqual(provider.prompts[0][0]["content"].splitlines()[-1], "Tools: none.")
        asyncio.run(run())

    def test_invalid_model_action_propagates_instead_of_oracle_fallback(self):
        async def run():
            with tempfile.TemporaryDirectory() as d:
                sim = simulator_module(UPSTREAM)
                campaign = SimpleNamespace(manifest={"binary_sha256": "fixture"}, provider=FixtureProvider("invalid JSON"))
                campaign.item_provider = lambda meta, path, **kw: BudgetedProvider(campaign.provider, meta, path, **kw)
                bridge = PMActionBridge(campaign, sim, "bare", 0, 0, Path(d))
                self.assertFalse(issubclass(PMActionError, sim.InvalidModelResponseError))
                with self.assertRaisesRegex(PMActionError, "oracle fallback prohibited"):
                    await bridge.action([{"role": "system", "content": "official protocol"},
                                         {"role": "user", "content": "visible observation"}])
        asyncio.run(run())

    def test_duplicate_or_partial_week_cannot_enter_scoring(self):
        scenario = load_pmbench_scenario(UPSTREAM)
        entries = [{"day": day["name"], "step_id": step["id"]}
                   for day in scenario["days"] for step in day["steps"]]
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_pmbench(UPSTREAM, entries[:-1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_pmbench(UPSTREAM, entries + entries[:1])


if __name__ == "__main__":
    unittest.main()

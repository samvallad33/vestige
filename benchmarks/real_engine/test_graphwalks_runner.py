"""GraphWalks raw-input isolation and exact official scoring semantics."""
from pathlib import Path
import asyncio
import tempfile
from types import SimpleNamespace
import unittest

from .graphwalks_runner import (graph_record_plan, load_graphwalks, official_scorer,
                                score_graphwalks, separate_graph, solve_graphwalks, GraphWalkRef)

ROOT = Path(__file__).resolve().parents[2] / "runs/upstream/graphwalks"


@unittest.skipUnless((ROOT / "README.md").is_file(), "official pinned GraphWalks not downloaded")
class GraphWalksTests(unittest.TestCase):
    def test_full_official_manifest_keeps750short_and400long_rows(self):
        items = load_graphwalks(ROOT)
        self.assertEqual(len(items), 1150)
        self.assertEqual(sum(isinstance(x, GraphWalkRef) for x in items), 400)
        self.assertEqual(len({x.id for x in items}), 1150)
        self.assertEqual(items[-1].id, "graphwalks-long:0399")

    def test_full_context_route_uses_official_prompt_and_shared_declared_cap(self):
        # This is a transport integration fixture for the bare route, not a
        # model response, engine substitute, or benchmark score.
        class FixtureBudget:
            remaining_calls = 6
            remaining_output = 2048
            usage = []
            cost = 0
            async def chat(self, messages, **kwargs):
                self.messages = messages
                self.remaining_calls -= 1
                return {"message": {"role": "assistant", "content": "Final Answer: []"},
                        "finish_reason": "stop", "model": "SCRIPTED_TRANSPORT_FIXTURE"}
        async def run():
            with tempfile.TemporaryDirectory() as d:
                budget = FixtureBudget()
                config = {}
                def factory(meta, root, **kwargs):
                    config.update(kwargs)
                    return budget
                campaign = SimpleNamespace(runs=Path(d), manifest={"binary_sha256": "fixture-no-engine"},
                                           item_provider=factory)
                item = load_graphwalks(ROOT, short_only=True)[0]
                row = await solve_graphwalks(campaign, item, "bare", 0, 0)
                self.assertEqual(row["prompt"][1]["content"], item.full_prompt)
                self.assertEqual(config, {"max_calls": 6, "max_output": 2048, "context": 131072})
                self.assertIsNone(row["score"])
                self.assertIsNone(row["engine"])
        asyncio.run(run())

    def test_incidence_conversion_preserves_source_without_solving(self):
        # Duplicates and self loops are source data and remain present.
        edges = ["a -> a", "a -> b", "a -> b", "b -> c"]
        plan = graph_record_plan(edges)
        self.assertEqual(plan["nodes"], ["a", "b", "c"])
        self.assertEqual([e["raw"] for e in plan["edges"]], edges)
        self.assertEqual(plan["conversion_sha256"], graph_record_plan(edges)["conversion_sha256"])
        self.assertNotIn("answer", plan)
        self.assertNotIn("operation", plan)

    def test_official_scorer_keeps_last_line_rule_and_set_semantics(self):
        score = official_scorer(ROOT)
        self.assertEqual(score("Final Answer: [a, a, b]", ["a", "b"])["f1"], 1)
        # Do not strip model output: publisher's last-line rule rejects this.
        self.assertEqual(score("Final Answer: [a]\n", ["a"])["f1"], 0)
        self.assertTrue(score("Final Answer: [a]\n", ["a"])["failed_to_parse"])
        self.assertEqual(score("Final Answer: []", [])["f1"], 1)
        self.assertEqual(score("Final Answer: [wrong]", ["a"])["f1"], 0)
        self.assertEqual(score("Final Answer: [a, b]", ["a", "c"])["f1"], 0.5)

    def test_literal_graph_split_preserves_query_and_examples(self):
        prompt = "Worked example untouched\nHere is the graph to operate on:\nThe graph has the following edges:\na -> b\nb -> c\n\nOperation:\nFind parents of c.\nFinal-format instructions"
        question, records = separate_graph(prompt)
        self.assertEqual(records, ["a -> b", "b -> c"])
        self.assertIn("Worked example untouched", question)
        self.assertIn("Find parents of c.\nFinal-format instructions", question)
        self.assertNotIn("a -> b", question)

    def test_missing_predictions_fail_before_official_scoring(self):
        items = load_graphwalks(ROOT, short_only=True)
        with self.assertRaisesRegex(ValueError, "coverage"):
            score_graphwalks(ROOT, items[:1], {})

    def test_all_official_short_items_load_without_gold_in_solver_records(self):
        items = load_graphwalks(ROOT, short_only=True)
        self.assertEqual(len(items), 750)
        self.assertEqual(len({i.id for i in items}), 750)
        self.assertEqual({i.category for i in items}, {"bfs", "parents"})
        for item in items:
            record = item.solver_record()
            self.assertNotIn("answer_nodes", record)
            self.assertNotIn("scorer_payload", record)
            self.assertNotIn("category", record)
            self.assertTrue(item.history)


if __name__ == "__main__":
    unittest.main()

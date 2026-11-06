"""Official scorer and full pinned-data contract checks; no model calls."""
from __future__ import annotations
import asyncio
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace

from benchmarks.real_engine.datasets import BenchmarkItem
from benchmarks.real_engine.memoryagentbench_runner import (
    CODE_REVISION, DATA_PINS, SPLITS, _standard_scores,
    load_memoryagentbench, score_memoryagentbench, verify_source, flush_full_coverage,
)

ROOT=Path(__file__).resolve().parents[2]/"runs/upstream/memoryagentbench"


class Contracts(unittest.TestCase):
    def test_split_order_and_full_official_counts(self):
        self.assertEqual(SPLITS[0],"Conflict_Resolution")
        self.assertEqual(sum(x[1] for x in DATA_PINS.values()),3671)

    def test_full_trial_units_retain_all_lru_repeats(self):
        units=sum(n*(3 if n<300 else 1) for _,n in DATA_PINS.values())
        self.assertEqual(units,4013)

    def test_exact_coverage_retains_future_trial_and_missing_item(self):
        from benchmarks.real_engine.runtime import record_key
        items=[BenchmarkItem("first","g",[],"q","c",{}),BenchmarkItem("future","g",[],"q","c",{})]
        checkpoint=SimpleNamespace(done={record_key({"item_id":"first","trial":0}):{}})
        with tempfile.TemporaryDirectory() as tmp:
            report=flush_full_coverage(Path(tmp),items,3,{"vestige":[checkpoint]},"budget")
            arm=report["arms"]["vestige"]
            self.assertEqual(arm["expected"],6)
            self.assertEqual(arm["completed_count"],1)
            self.assertEqual(arm["uncompleted_count"],5)
            self.assertIn({"item_id":"future","trial":2,"shard":next(x["shard"] for x in arm["uncompleted"] if x["item_id"]=="future")},arm["uncompleted"])

    def test_complete_prediction_map_still_cannot_bypass_integrity(self):
        from benchmarks.real_engine.integrity import IntegrityPending
        async def forbidden(_):raise AssertionError("no judge before integrity")
        item=BenchmarkItem("id","group",[],"q","factconsolidation_sh_6k",{"source":"factconsolidation_sh_6k","answer":["a"]})
        if not ROOT.exists():self.skipTest("official checkout absent")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IntegrityPending):
                asyncio.run(score_memoryagentbench(ROOT,[item],{"id":"a"},forbidden,Path(tmp)))

    def test_gold_not_in_solver_record(self):
        item=BenchmarkItem("id","group",["raw context"],"official question","category",{"answer":"HELDOUT_GOLD","keypoints":["HELDOUT_POINT"]})
        solver=str(item.solver_record())
        self.assertNotIn("HELDOUT_GOLD",solver)
        self.assertNotIn("HELDOUT_POINT",solver)

    def test_strict_scoring_coverage_fails_before_any_judge(self):
        async def forbidden(_): raise AssertionError("judge must not run on incomplete coverage")
        item=BenchmarkItem("a","group",[],"q","factconsolidation_sh_6k",{"source":"factconsolidation_sh_6k"})
        if not ROOT.exists(): self.skipTest("official checkout absent")
        with self.assertRaisesRegex(ValueError,"coverage"):
            asyncio.run(score_memoryagentbench(ROOT,[item],{},forbidden,ROOT/"test-output"))


@unittest.skipUnless(ROOT.exists(),"official pinned checkout required")
class OfficialScorer(unittest.TestCase):
    def test_source_pin_and_actual_substring_exact_semantics(self):
        verify_source(ROOT)
        rows=_standard_scores(ROOT,[
            {"id":"correct","source":"factconsolidation_sh_6k","answer":["navy blue"],"response":"It is navy blue."},
            {"id":"incorrect","source":"factconsolidation_sh_6k","answer":["navy blue"],"response":"It is red."},
            {"id":"strict","source":"icl_banking77_5900shot_balance","answer":["43"],"response":"label: 43"},
        ])
        byid={x["id"]:x for x in rows}
        self.assertTrue(byid["correct"]["metrics"]["substring_exact_match"])
        self.assertFalse(byid["incorrect"]["metrics"]["substring_exact_match"])
        # Preserve upstream strict behavior rather than 'fixing' scoring.
        self.assertFalse(byid["strict"]["metrics"]["exact_match"])


if __name__=="__main__":unittest.main()

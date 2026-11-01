"""Integrity-auditor tests only. These fixtures are never benchmark evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .integrity import (IntegrityInvalid, IntegrityPending, _full_set, _hook_checks,
    _kill_check, _merge_and_score, _native_item, audit, canonical, literal_overlap_audit,
    require, sha, strings)


def request(content, item="a", arm="vestige", data=None, condition=None, phase="solver", at=10):
    return {"path":"fixture/api.jsonl","line":3,"row":{"timestamp":at,
        "identity":{"benchmark":"fixture","item_id":item,"arm":arm,
                    "engine_data_dir":data,"condition":condition,"phase":phase},
        "payload":{"model":"gpt-5.4-nano","messages":[{"role":"user","content":content}]}}}


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.item={"id":"a","question":"What changed?","history":["History contains SECRET ANSWER"],
                   "scorer_payload":{"answer":"SECRET ANSWER"}}

    def test_all_nested_strings_include_tool_schema_descriptions(self):
        value={"messages":[{"content":"question"}],"tools":[{"description":"SECRET ANSWER"}]}
        self.assertEqual({s for _,s in strings(value)},{"question","SECRET ANSWER"})

    def test_raw_history_does_not_excuse_vestige_oracle_without_native_read(self):
        found,_=literal_overlap_audit([request("SECRET ANSWER")],[self.item])
        self.assertEqual(found[0]["status"],"FAIL")
        self.assertEqual(found[0]["provenance"],"UNRESOLVED_POTENTIAL_LEAK")

    def test_native_evidence_from_another_item_never_excuses_leak(self):
        found,_=literal_overlap_audit([request("SECRET ANSWER")],[self.item],
                                    native_strings={"different-item":["SECRET ANSWER"]})
        self.assertEqual(found[0]["status"],"FAIL")

    def test_later_native_write_does_not_launder_earlier_oracle(self):
        catalog={("a","/case/poison"):[{"text":"SECRET ANSWER","timestamp":20,"path":"mcp.jsonl","line":5}]}
        found,_=literal_overlap_audit([request("SECRET ANSWER",data="/case/poison",at=10)],[self.item],catalog)
        self.assertEqual(found[0]["status"],"FAIL")

    def test_actual_prior_same_store_read_can_authorize_literal(self):
        catalog={("a","/case/poison"):[{"text":"SECRET ANSWER","timestamp":5,"path":"mcp.jsonl","line":5}]}
        found,_=literal_overlap_audit([request("SECRET ANSWER",data="/case/poison",at=10)],[self.item],catalog)
        self.assertEqual(found[0]["status"],"PASS")

    def test_clean_condition_store_does_not_authorize_poison_condition(self):
        item={**self.item,"condition_inputs":{"clean":["SECRET ANSWER"],"poisoned":["poisoned source"]}}
        catalog={("a","/case/clean"):[{"text":"SECRET ANSWER","timestamp":5}]}
        found,_=literal_overlap_audit([request("SECRET ANSWER",data="/case/poison",condition="poisoned")],[item],catalog)
        self.assertEqual(found[0]["status"],"FAIL")

    def test_official_judge_overlap_is_explicit_pending_not_solver_waiver(self):
        found,_=literal_overlap_audit([request("SECRET ANSWER",phase="official_judge")],[self.item])
        self.assertEqual(found[0]["status"],"PENDING")
        self.assertIn("official_judge",found[0]["provenance"])

    def test_numeric_and_oracle_policy_labels_are_compared(self):
        item={"id":"a","question":"q","history":[],"scorer_payload":{"expected_count":42,"oracle_memory":"HIDDEN POLICY"}}
        found,_=literal_overlap_audit([request("42 HIDDEN POLICY")],[item])
        self.assertEqual({f["literal"] for f in found},{"42","HIDDEN POLICY"})
        self.assertTrue(all(f["status"]=="FAIL" for f in found))


class GateTests(unittest.TestCase):
    def test_stage_gate_has_no_full_scorer_chicken_and_egg(self):
        report={"benchmarks":{"b":{"status":"PENDING","ready_to_score":True,
                "ready_for_official_scorer":False,"ready_to_report":False,"ready_to_prequalify":True}}}
        require(report,"b","score")
        require(report,"b","preflight")
        with self.assertRaises(IntegrityPending):require(report,"b","scorer")
        with self.assertRaises(IntegrityPending):require(report,"b","report")

    def test_failure_always_invalidates_even_if_some_ready_flag_is_true(self):
        report={"benchmarks":{"b":{"status":"FAIL","ready_to_score":True}}}
        with self.assertRaises(IntegrityInvalid):require(report,"b","score")

    def test_empty_runtime_never_passes_or_approves_an_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);runs=repo/"runs";runs.mkdir()
            with patch("benchmarks.real_engine.integrity.subprocess.check_output",return_value="test-fixture\n"):
                result=audit(repo,runs)
            self.assertEqual(result["status"],"PENDING")
            self.assertFalse(result["ready_to_score"])
            self.assertFalse(result["ready_to_report"])
            self.assertEqual(result["invalid_benchmarks"],[])
            self.assertTrue(all(not ids for arms in result["approved_run_identities"].values() for ids in arms.values()))

    def test_completed_engine_only_row_is_observed_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"native/strata";data.mkdir(parents=True)
            row={"engine":{"data_dir":str(data),"pid":10,"binary_sha256":"binary"}}
            result=_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})
            self.assertEqual(result["status"],"FAIL")

    def test_missing_kill_and_hook_evidence_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(_kill_check(root,root,{})["status"],"PENDING")
            self.assertEqual(_hook_checks(root,root,[])["status"],"PENDING")

    def test_sample_cannot_replace_full_official_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"official.json";data.write_text("official fixture")
            plan={"full_set":{"item_ids":["one"],"trials":[0,1,2],"dataset_files":{str(data):sha(data)}}}
            result,_=_full_set(root,root,"fixture",plan,[{"id":"one"},{"id":"two"}])
            self.assertEqual(result["status"],"FAIL")

    def test_duplicate_merged_outputs_fail_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/"b/bare";arm.mkdir(parents=True)
            row={"item_id":"a","trial":0}
            (arm/"merged.jsonl").write_text(canonical(row)+"\n"+canonical(row)+"\n")
            merge,scorer=_merge_and_score(root,"b","bare",{("a",0)},{})
            self.assertEqual(merge["status"],"FAIL")
            self.assertEqual(scorer["status"],"PENDING")

    def test_complete_merge_does_not_claim_scorer_already_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/"b/bare";arm.mkdir(parents=True)
            (arm/"merged.jsonl").write_text(canonical({"item_id":"a","trial":0})+"\n")
            merge,scorer=_merge_and_score(root,"b","bare",{("a",0)},{})
            self.assertEqual(merge["status"],"PASS")
            self.assertEqual(scorer["status"],"PENDING")


if __name__=="__main__":unittest.main()

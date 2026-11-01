"""Integrity-auditor tests only. These fixtures are never benchmark evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import patch

from .integrity import (IntegrityInvalid, IntegrityPending, _full_set, _hook_checks,
    _kill_check, _merge_and_score, _native_item, _parity, _pilot_check, audit, canonical, literal_overlap_audit,
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


class CandidateBindingTests(unittest.TestCase):
    def test_mpbench_only_accepts_exact_recorded_native_write_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"native/strata";data.mkdir(parents=True)
            ident={"data_dir":str(data),"pid":123,"binary_sha256":"binary","hooks_enabled":True,
                   "transport":"stdio-jsonrpc","client":"benchmarks/memconflict/mcp_client.VestigeMCP","fresh_store_nodes":0}
            raw={"isError":True,"content":[{"type":"text","text":"Refused to store probable credential(s): test credential"}]}
            decision={"defense_positive":True,"result":raw}
            row={"benchmark":"mpbench","engine":ident,"defense_positive":True,"write_decisions":[decision]}
            prefix=[{"kind":"engine_start","identity":ident},
                    {"kind":"tool_request","name":"memory_status","arguments":{"view":"coverage"}},
                    {"kind":"tool_response","name":"memory_status","result":{"totalNodes":0}},
                    {"kind":"tool_request","name":"smart_ingest","arguments":{"content":"test"}}]
            log=data.parent/"mcp.jsonl"
            def write(error):
                log.write_text("".join(canonical(e)+"\n" for e in prefix+[error]))
            error={"kind":"tool_error","name":"smart_ingest","error":"native refusal","raw_mcp_result":raw}
            write(error)
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"PASS")
            write({**error,"raw_mcp_result":{"isError":True,"content":[{"text":"invalid arguments"}]}})
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")
            write(error)
            self.assertEqual(_native_item({**row,"write_decisions":[]},root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")
            self.assertEqual(_native_item({**row,"benchmark":"locomo"},root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")
            log.write_text("".join(canonical(e)+"\n" for e in [prefix[0],prefix[-1],error]))
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")

    def test_kill_control_requires_current_config_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);runs=repo/"runs";runs.mkdir()
            source=repo/"benchmarks/real_engine/runner.py";source.parent.mkdir(parents=True)
            source.write_text("# immutable fixture runner\n")
            (runs/"build-manifest.json").write_text(canonical({"binary_sha256":"binary"}))
            log=runs/"kill.jsonl"
            records=[{"event":"engine_started","pid":123,"binary_sha256":"binary"},
                     {"event":"midrun_kill","pid":123,"returncode":-9},
                     {"event":"runner_aborted","pid":123,"fallback_used":False,
                      "exception":"engine terminated","runner_source_sha256":sha(source)}]
            log.write_text("".join(canonical({**e,"config_hash":"old"})+"\n" for e in records))
            plan={"config_hash":"new","kill_control":{"log":str(log)}}
            self.assertEqual(_kill_check(repo,runs,plan)["status"],"PENDING")
            with log.open("a") as out:
                for e in records:out.write(canonical({**e,"config_hash":"new"})+"\n")
            self.assertEqual(_kill_check(repo,runs,plan)["status"],"PASS")

    def test_claimed_pid_hash_and_request_only_cannot_pass_native_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"native/strata";data.mkdir(parents=True)
            ident={"data_dir":str(data),"pid":123,"binary_sha256":"binary","hooks_enabled":True,
                   "transport":"stdio-jsonrpc","client":"benchmarks/memconflict/mcp_client.VestigeMCP","fresh_store_nodes":0}
            path=data.parent/"mcp.jsonl"
            request={"kind":"tool_request","name":"recall","arguments":{"handle":"history"}}
            path.write_text(canonical(request)+"\n")
            row={"engine":ident}
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")
            path.write_text(canonical({"kind":"engine_start","identity":ident})+"\n"+canonical(request)+"\n")
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")
            with path.open("a") as out:out.write(canonical({"kind":"tool_response","name":"recall","result":{"results":[]}})+"\n")
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"PASS")
            row["engine"]={**ident,"pid":456}
            self.assertEqual(_native_item(row,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"],"FAIL")

    def parity_fixture(self,root):
        from .runtime import BudgetLedger
        from .provider import account
        reqs,replies=defaultdict(list),defaultdict(list)
        plan={"budgets":{a:{"max_calls":6,"max_output":2048,"context":32768} for a in ("bare","bm25","embedding","vestige")}}
        ledger=BudgetLedger(root/"budget.db")
        for arm in plan["budgets"]:
            for i in range(10):
                call=f"fixture/{arm}/{i}"
                ident={"benchmark":"fixture","arm":arm,"item_id":str(i),"trial":0,"config_hash":"new","phase":"integrity_preflight"}
                payload={"model":"gpt-5.4-nano","temperature":0,"max_completion_tokens":512,
                         "messages":[{"role":"system","content":"Answer the question.\nTools: "+arm},{"role":"user","content":"Question "+str(i)}]}
                reqs["fixture",arm].append({"path":str(root/arm/str(i)/"api.jsonl"),"line":1,
                    "row":{"call_id":call,"identity":ident,"payload":payload}})
                response={"usage":{"prompt_tokens":30,"completion_tokens":5},"choices":[{"message":{"content":"answer"}}]}
                cost,usage=account(response)
                ledger.reserve(call+"/attempt-0",.01)
                ledger.settle(call+"/attempt-0",cost,usage)
                replies["fixture",arm].append({"path":str(root/arm/str(i)/"api.jsonl"),"line":2,
                    "row":{"call_id":call,"identity":ident,"response":response,"usage":usage,"cost":cost}})
        return reqs,replies,plan

    def test_request_only_parity_is_pending_and_real_usage_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reqs,replies,plan=self.parity_fixture(root)
            self.assertEqual(_parity(reqs,plan,"fixture",[],{},root)["status"],"PENDING")
            self.assertEqual(_parity(reqs,plan,"fixture",[],replies,root)["status"],"PASS")
            replies["fixture","vestige"][0]["row"]["usage"]={"prompt_tokens":30,"completion_tokens":9999}
            self.assertEqual(_parity(reqs,plan,"fixture",[],replies,root)["status"],"FAIL")

    def test_usage_without_settled_shared_ledger_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reqs,replies,plan=self.parity_fixture(root)
            import sqlite3
            with sqlite3.connect(root/"budget.db") as db:
                db.execute("DELETE FROM calls WHERE id=?",("fixture/vestige/0/attempt-0",))
            self.assertEqual(_parity(reqs,plan,"fixture",[],replies,root)["status"],"FAIL")

    def test_stale_pilot_cannot_qualify_current_config(self):
        ids=[str(i) for i in range(10)]
        plan={"config_hash":"new","run_identities":{"bare":"new-checkpoint"},
              "pilot":{"seed":20261101,"config_hash":"new","item_ids":ids}}
        reqs=[{"path":"fixture/api.jsonl","line":1,"row":{"identity":{"item_id":i,"config_hash":"old"}}} for i in ids]
        completed=[(Path("worker-0.jsonl"),1,{"item_id":i,"config_hash":"old","run_identity":"old-checkpoint",
                    "phase":"integrity_preflight","score":None}) for i in ids]
        self.assertEqual(_pilot_check(plan,"bare",reqs,completed)["status"],"PENDING")
        for request in reqs:request["row"]["identity"]["config_hash"]="new"
        self.assertEqual(_pilot_check(plan,"bare",reqs,completed)["status"],"PENDING")
        completed=[(p,n,{**row,"config_hash":"new","run_identity":"new-checkpoint"}) for p,n,row in completed]
        self.assertEqual(_pilot_check(plan,"bare",reqs,completed)["status"],"PASS")
        completed[0][2]["phase"]="full_set_execution"
        self.assertEqual(_pilot_check(plan,"bare",reqs,completed)["status"],"PENDING")

    def test_pilot_must_match_fixed_seed_official_ids(self):
        ids=[str(i) for i in range(20)]
        plan={"config_hash":"new","run_identities":{"bare":"checkpoint"},
              "pilot":{"seed":20261101,"config_hash":"new","item_ids":ids[:10]}}
        self.assertEqual(_pilot_check(plan,"bare",[],[],[{"id":i} for i in ids])["status"],"FAIL")

    def test_complete_merge_does_not_claim_scorer_already_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);arm=root/"b/bare";arm.mkdir(parents=True)
            (arm/"merged.jsonl").write_text(canonical({"item_id":"a","trial":0})+"\n")
            merge,scorer=_merge_and_score(root,"b","bare",{("a",0)},{})
            self.assertEqual(merge["status"],"PASS")
            self.assertEqual(scorer["status"],"PENDING")


if __name__=="__main__":unittest.main()

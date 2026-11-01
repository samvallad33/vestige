"""Integrity-auditor tests only. These fixtures are never benchmark evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from collections import defaultdict
from unittest.mock import patch

from .integrity import (IntegrityInvalid, IntegrityPending, _full_set, _hook_checks,
    _candidate_config_check, _kill_check, _merge_and_score, _merge_prior_reports, _native_item, _native_provenance, _mpbench_exposure, _parity, _pilot_check, _walk, audit, canonical, lines, literal_overlap_audit,
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
    def test_partial_audit_prunes_other_benchmark_logs_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);runs=repo/"runs";runs.mkdir()
            other=runs/"longmemeval/vestige/worker-0/attempt/api.jsonl"
            other.parent.mkdir(parents=True);other.write_text("corrupt unselected log must never be opened\n")
            chosen=runs/"locomo/bare/worker-0/attempt/api.jsonl"
            chosen.parent.mkdir(parents=True);chosen.write_text("")
            self.assertEqual(list(_walk(runs,("locomo",))),[chosen])
            def guarded(path):
                self.assertNotEqual(path,other)
                return lines(path)
            with patch("benchmarks.real_engine.integrity.lines",side_effect=guarded),patch("benchmarks.real_engine.integrity.subprocess.check_output",return_value="fixture\n"):
                result=audit(repo,runs,benchmarks=("locomo",))
            self.assertEqual(result["audited_benchmarks"],["locomo"])
            self.assertEqual(set(result["benchmarks"]),{"locomo"})

    def test_duplicate_native_roots_are_parsed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/"native";native.mkdir()
            path=native/"mcp.jsonl"
            path.write_text(canonical({"kind":"tool_response","timestamp":5,"result":{"answer":"actual native text"}})+"\n")
            row={"item_id":"a","engine":{"data_dir":str(native/"strata")}}
            entry=(root/"worker-0.jsonl",1,row)
            with patch("benchmarks.real_engine.integrity.lines",wraps=lines) as parser:
                catalog=_native_provenance([entry,entry,entry])
            self.assertEqual(parser.call_count,1)
            self.assertEqual(len(catalog["a"]),2)

    def test_permitted_exact_set_keeps_unresolved_gold_as_failure(self):
        item={"id":"a","question":"q","history":[],"scorer_payload":{"answer":"SECRET ANSWER","label":"PRIVATE LABEL"}}
        req=request("SECRET ANSWER PRIVATE LABEL")
        one,_=literal_overlap_audit([req],[item],native_strings={"a":["SECRET ANSWER"]})
        repeated,_=literal_overlap_audit([req],[item],native_strings={"a":["SECRET ANSWER"]*100})
        self.assertEqual(one,repeated)
        self.assertEqual([entry["status"] for entry in repeated],["PASS","FAIL"])
        exact,_=literal_overlap_audit([request("SECRET ANSWER")],[item],native_strings={"a":["SECRET ANSWER"]*100})
        self.assertEqual(exact[0]["status"],"PASS")

    def test_partial_report_retains_only_same_config_and_preserves_asof(self):
        old={"config_hash":"same","checked_at":"earlier","status":"PENDING","ready_to_score":True,
             "arms":{arm:{"ready_to_score":True} for arm in ("bare","bm25","embedding","vestige")}}
        prior={"benchmarks":{"locomo":old,"longmemeval":{**old,"config_hash":"old"}},
               "approved_run_identities":{"locomo":{"bare":["checkpoint"]}}}
        current={"memconflict":{**old,"checked_at":"now","fresh_in_this_audit":True}}
        plans={name:{"config_hash":"same"} for name in ("locomo","longmemeval","memconflict")}
        merged,approved,retained=_merge_prior_reports(prior,current,{"memconflict":{}},plans,("memconflict",))
        self.assertEqual(retained,["locomo"])
        self.assertEqual(set(merged),{"memconflict","locomo"})
        self.assertEqual(merged["locomo"]["checked_at"],"earlier")
        self.assertFalse(merged["locomo"]["fresh_in_this_audit"])
        self.assertEqual(approved["locomo"]["bare"],["checkpoint"])
        report={"benchmarks":merged,"audited_benchmarks":["memconflict"]}
        require(report,"memconflict","execute")
        with self.assertRaises(IntegrityPending):require(report,"locomo","execute")
        # Matching an obsolete per-benchmark plan is insufficient when the
        # freshly audited candidate uses a different frozen configuration.
        new_current={"memconflict":{**current["memconflict"],"config_hash":"new-candidate"}}
        mixed_plans={**plans,"memconflict":{"config_hash":"new-candidate"}}
        merged,approved,retained=_merge_prior_reports(prior,new_current,{"memconflict":{}},mixed_plans,("memconflict",))
        self.assertEqual(retained,[])
        self.assertEqual(set(merged),{"memconflict"})
        self.assertNotIn("locomo",approved)

    def test_frozen_config_rejects_actual_source_edit_with_unchanged_claims(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);runs=repo/"runs";runs.mkdir()
            source=repo/"benchmarks/real_engine/runner.py";source.parent.mkdir(parents=True)
            source.write_text("# frozen source\n")
            hook=repo/"hooks/session.sh";hook.parent.mkdir()
            hook.write_text("# unchanged hook\n")
            binary=repo/"target/release/vestige-mcp";binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fixture binary")
            hooks={"audit_complete":True,"source_pins":{"hooks/session.sh":sha(hook)}}
            (runs/"hook-audit.json").write_text(canonical(hooks))
            (runs/"build-manifest.json").write_text(canonical({"binary_sha256":sha(binary)}))
            config={"sources":{"benchmarks/real_engine/runner.py":sha(source)},"hooks":hooks,"binary_sha256":sha(binary)}
            config["config_hash"]=hashlib.sha256(canonical(config).encode()).hexdigest()
            (runs/"four-config.json").write_text(canonical(config))
            plan={"config_hash":config["config_hash"]}
            self.assertEqual(_candidate_config_check(repo,runs,plan)["status"],"PASS")
            source.write_text("# changed while controller still claims old hash\n")
            self.assertEqual(_candidate_config_check(repo,runs,plan)["status"],"FAIL")
            source.write_text("# frozen source\n")
            (runs/"hook-audit.json").write_text(canonical({**hooks,"audit_complete":False}))
            self.assertEqual(_candidate_config_check(repo,runs,plan)["status"],"FAIL")
            (runs/"hook-audit.json").write_text(canonical(hooks))
            config["temperature"]=1
            (runs/"four-config.json").write_text(canonical(config))
            self.assertEqual(_candidate_config_check(repo,runs,plan)["status"],"FAIL")

    def test_frozen_config_absent_is_pending_and_escaping_pin_fails(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);runs=repo/"runs";runs.mkdir()
            self.assertEqual(_candidate_config_check(repo,runs,{})["status"],"PENDING")
            config={"sources":{"../outside.py":"digest"},"hooks":{"source_pins":{"hooks/session.sh":"digest"}},"binary_sha256":"digest"}
            config["config_hash"]=hashlib.sha256(canonical(config).encode()).hexdigest()
            (runs/"four-config.json").write_text(canonical(config))
            self.assertEqual(_candidate_config_check(repo,runs,{"config_hash":config["config_hash"]})["status"],"FAIL")

    def exposure_fixture(self,root,arm="vestige",native_result=None):
        import hashlib
        case=root/arm/"attempt";case.mkdir(parents=True)
        native=case/"native";native.mkdir()
        raw=canonical({"user_query":"Please remember the supplied context.","context":"unaltered public raw input"})
        digest=hashlib.sha256(raw.encode()).hexdigest()
        result=native_result if native_result is not None else {"success":True,"nodeId":"native-node"}
        positive=True if result.get("isError") else False
        if arm in {"bm25","embedding"}:result,positive={"success":True,"id":"memory-0"},None
        metadata={"attempted":True,"source":"user-input","content_sha256":digest,"result":result,"defense_positive":positive}
        row={"benchmark":"mpbench","arm":arm,"item_id":"case","engine":{"data_dir":str(native/"strata")},
             "raw_log":str(case/"api.jsonl"),"normal_write_exposure":metadata,
             "write_decisions":[{"basis":"initial raw user-input normal-write response","result":result,"defense_positive":positive}],
             "persisted_entries":[raw] if arm in {"bm25","embedding"} else []}
        (case/"api.jsonl").write_text(canonical({"event":"request","timestamp":10})+"\n")
        (case/"memory-writes.jsonl").write_text(canonical({"event":"normal_initial_write","timestamp":7,"content":raw,**metadata})+"\n")
        args={"content":raw,"node_type":"note","tags":["history"],"source":"user-input"}
        request={"kind":"tool_request","timestamp":5,"name":"smart_ingest","arguments":args}
        reply={"kind":"tool_response","timestamp":6,"name":"smart_ingest","result":result}
        if result.get("isError"):reply={"kind":"tool_error","timestamp":6,"name":"smart_ingest","raw_mcp_result":result}
        log=native/"mcp.jsonl"
        log.write_text(canonical(request)+"\n"+canonical(reply)+"\n")
        return row,{"id":"case","question":raw},log,request,reply

    def test_mpbench_exposure_requires_official_bytes_in_actual_native_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,item,log,request,reply=self.exposure_fixture(root)
            def status():return _mpbench_exposure(row,root/"worker-0.jsonl",item,root,root)["status"]
            self.assertEqual(status(),"PASS")
            log.write_text(canonical({"kind":"tool_request","name":"memory_status","arguments":{"view":"coverage"}})+"\n")
            self.assertEqual(status(),"FAIL")
            log.write_text(canonical(request)+"\n")
            self.assertEqual(status(),"FAIL")
            altered={**request,"arguments":{**request["arguments"],"content":"clean substitute"}}
            log.write_text(canonical(altered)+"\n"+canonical(reply)+"\n")
            self.assertEqual(status(),"FAIL")
            altered={**request,"arguments":{**request["arguments"],"forceCreate":True}}
            log.write_text(canonical(altered)+"\n"+canonical(reply)+"\n")
            self.assertEqual(status(),"FAIL")
            log.write_text(canonical(request)+"\n"+canonical(reply)+"\n")
            item["question"]="different official raw bytes"
            self.assertEqual(status(),"FAIL")

    def test_mpbench_exposure_tracks_real_gate_denial_and_baseline_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw={"isError":True,"content":[{"text":"Refused to store probable credential(s): test credential"}]}
            row,item,_,_,_=self.exposure_fixture(root,native_result=raw)
            self.assertEqual(_mpbench_exposure(row,root/"worker-0.jsonl",item,root,root)["status"],"PASS")
            for arm in ("bm25","embedding"):
                row,item,_,_,_=self.exposure_fixture(root,arm)
                self.assertEqual(_mpbench_exposure(row,root/"worker-0.jsonl",item,root,root)["status"],"PASS")
                row["persisted_entries"]=["substituted clean memory"]
                self.assertEqual(_mpbench_exposure(row,root/"worker-0.jsonl",item,root,root)["status"],"FAIL")

    def test_native_tool_error_must_be_current_and_forwarded_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"native/strata";data.mkdir(parents=True)
            ident={"data_dir":str(data),"pid":123,"binary_sha256":"binary","hooks_enabled":True,
                   "transport":"stdio-jsonrpc","client":"benchmarks/memconflict/mcp_client.VestigeMCP","fresh_store_nodes":0}
            arguments={"mode":"hybrid_search","query":"question"}
            raw={"isError":True,"content":[{"type":"text","text":"similarity_disabled:hybrid_search"}]}
            messages=[{"role":"assistant","tool_calls":[{"id":"call-one","function":{"name":"recall","arguments":canonical(arguments)}}]},
                      {"role":"tool","tool_call_id":"call-one","content":canonical(raw)}]
            row={"benchmark":"longmemeval","engine":ident,"transcript":messages}
            prefix=[{"kind":"engine_start","identity":ident},
                    {"kind":"tool_request","name":"memory_status","arguments":{"view":"coverage"}},
                    {"kind":"tool_response","name":"memory_status","result":{"totalNodes":0}},
                    {"kind":"tool_request","name":"recall","arguments":arguments}]
            error={"kind":"tool_error","name":"recall","error":"native tool error","raw_mcp_result":raw}
            log=data.parent/"mcp.jsonl"
            def result(record,events):
                log.write_text("".join(canonical(e)+"\n" for e in events))
                return _native_item(record,root/"worker-0.jsonl",root,{"binary_sha256":"binary"})["status"]
            self.assertEqual(result(row,prefix+[error]),"PASS")
            self.assertEqual(result({**row,"transcript":[]},prefix+[error]),"FAIL")
            self.assertEqual(result(row,prefix+[{**error,"raw_mcp_result":None,"error":"engine terminated"}]),"FAIL")
            self.assertEqual(result(row,prefix+[{**error,"raw_mcp_result":{"isError":True,"content":[{"text":"stale different error"}]}}]),"FAIL")
            self.assertEqual(result(row,prefix+[error,prefix[-1],error]),"FAIL")
            self.assertEqual(result(row,[prefix[0],prefix[-1],error]),"FAIL")

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

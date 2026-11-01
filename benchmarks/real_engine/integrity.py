"""Fail-closed campaign evidence audit. This module never runs models or scorers.

An unexecuted requirement is PENDING, never PASS. FAIL means an observed
violation: the caller must stop that benchmark, mark it INVALID and exclude its
metrics. The auditor preserves evidence and does not repair candidate code.
Ten sampled items qualify evidence; they never replace an official full set.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import fcntl
import threading
from typing import Any

BENCHMARKS = ("locomo", "longmemeval", "mpbench", "memconflict", "pmbench",
              "memoryagentbench", "memform", "terminal_bench", "graphwalks",
              "automationbench", "vals_finance_v2", "cwe_bench", "agentdojo", "injecagent")
ARMS = ("bare", "bm25", "embedding", "vestige")
MODEL = "gpt-5.4-nano"
SAMPLE_SEED = 20261101
SAMPLE_ITEMS = 10
EXCLUDED = {"upstream", "superseded-engine-only", "invalidated", "engine-smoke", "__pycache__"}
READ_TOOLS = {"recall", "memory", "memory_status", "session_start", "graph", "receipt", "causal_walk"}
SOURCES = {
    "locomo": "run.py", "longmemeval": "run.py", "memconflict": "run.py",
    "mpbench": "mpbench_runner.py", "pmbench": "pmbench_runner.py",
    "memoryagentbench": "run.py", "memform": "memform_runner.py",
    "terminal_bench": "terminal_runner.py", "graphwalks": "graphwalks_runner.py",
    "automationbench": "automationbench_runner.py", "agentdojo": "security_agents.py",
    "injecagent": "security_agents.py",
}


class IntegrityPending(RuntimeError):
    pass


class IntegrityInvalid(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def strings(value: Any, pointer: str = "$"):
    """Every string, including tool descriptions and nested model inputs."""
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, dict):
        for key, entry in value.items():
            yield from strings(entry, f"{pointer}.{key}")
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            yield from strings(entry, f"{pointer}[{index}]")


def json_file(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def lines(path: Path):
    if not path.is_file():
        return []
    result = []
    with path.open() as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_SH)
        contents=handle.read()
    for number, line in enumerate(contents.splitlines(), 1):
        if line.strip():
            result.append((number, json.loads(line)))
    return result


def evidence(path: Path, line=None, count=None, detail=None):
    value = {"path": str(path)}
    for key, item in (("line", line), ("count", count), ("detail", detail)):
        if item is not None:
            value[key] = item
    return value


def check(identity: str, status: str, reason: str, observed=None):
    assert status in {"PASS", "FAIL", "PENDING"}
    return {"id": identity, "status": status, "reason": reason, "evidence": observed or []}


def combine(checks):
    statuses = {c["status"] for c in checks}
    return "FAIL" if "FAIL" in statuses else "PENDING" if "PENDING" in statuses else "PASS"


def require(report: dict, benchmark: str, stage: str = "score"):
    entry = report["benchmarks"][benchmark]
    if entry["status"] == "FAIL":
        raise IntegrityInvalid(f"{benchmark} INVALID: integrity failure; stop and exclude metrics")
    flag = {"preflight": "ready_to_prequalify", "execute": "ready_to_score", "score": "ready_to_score",
            "scorer": "ready_for_official_scorer", "report": "ready_to_report"}.get(stage)
    if flag is None:
        raise ValueError("unknown integrity stage")
    if not entry.get(flag):
        raise IntegrityPending(f"{benchmark}: integrity evidence incomplete; no {stage} permitted")


def _walk(runs: Path):
    for directory, folders, files in os.walk(runs):
        folders[:] = [name for name in folders if name not in EXCLUDED and not name.startswith(".")]
        for filename in files:
            yield Path(directory) / filename


def _base(name):
    return "memoryagentbench" if name.startswith("memoryagentbench-") else name


def _source_check(repo: Path, benchmark: str):
    path = repo / "benchmarks/real_engine" / SOURCES.get(benchmark, "")
    if not path.is_file():
        return check("source_native_path", "PENDING", "No executable candidate source is present.")
    anchors = []
    text = path.read_text()
    for n, line in enumerate(text.splitlines(), 1):
        if any(word in line for word in ("RealEngine", "HookedRealEngine", "engine.call", "engine.ingest_history")):
            anchors.append(evidence(path, n, detail=line.strip()))
    # These anchors establish which candidate was inspected, not that a source
    # keyword proves runtime behavior. Actual per-item transport is checked below.
    return check("source_native_path", "PASS" if anchors else "PENDING",
                 "Native constructor/call sites recorded; no runtime claim from keywords. "
                 "Baseline memory implementations are not treated as Vestige implementations.", anchors)


def _load_items(repo: Path, runs: Path, benchmark: str):
    """Official loader records solely for audit, never forwarded to a model."""
    if benchmark in {"locomo", "longmemeval"}:
        from .datasets import load_locomo, load_longmemeval
        loader = load_locomo if benchmark == "locomo" else load_longmemeval
        return loader(runs / "upstream" / benchmark)
    if benchmark == "memconflict":
        from .extra_datasets import load_memconflict
        return load_memconflict(runs / "upstream" / benchmark)
    if benchmark == "mpbench":
        from .mpbench import load_mpbench, solver_input
        cases, _ = load_mpbench(runs / "upstream")
        return [{"id": c["id"], "question": canonical(solver_input(c)), "history": [],
                 "scorer_payload": {"expected_memory": c["raw"].get("expected_memory"),
                                    "label": c["raw"].get("label"),
                                    "attack_type": c["raw"].get("attack_type")}} for c in cases]
    if benchmark == "memoryagentbench":
        from .memoryagentbench_runner import load_memoryagentbench
        return load_memoryagentbench(runs / "upstream/memoryagentbench")
    if benchmark == "graphwalks":
        from .graphwalks_runner import load_graphwalks
        return load_graphwalks(runs / "upstream")
    if benchmark == "memform":
        from .memform_runner import load_memform_audit_items
        return load_memform_audit_items(runs / "upstream/memform")
    if benchmark == "automationbench":
        from .automationbench_runner import load_automationbench_audit_items
        return load_automationbench_audit_items(runs / "upstream/AutomationBench")
    if benchmark == "pmbench":
        from .pmbench_runner import load_pmbench_audit_items
        return load_pmbench_audit_items(runs / "upstream")
    if benchmark in {"injecagent", "agentdojo"}:
        from .security_runner import official_gold_resolver, dojo_cases
        from .security_agents import InjecAgentSource
        upstream = runs/"upstream"
        cases = InjecAgentSource(upstream/"InjecAgent").cases if benchmark=="injecagent" else dojo_cases(upstream/"agentdojo")
        result=[]
        for case in cases:
            identity=case["id"]
            resolved=official_gold_resolver(benchmark,identity,upstream)
            permitted=resolved.get("permitted_official_inputs",[])
            result.append({"id":identity,"question":"", "history":[text for _,text in strings(permitted)],
                           "scorer_payload":resolved.get("gold_fields",{}),"audit_source":resolved})
        return result
    return None


def _item_value(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _gold(item):
    result = []
    def leaves(value,pointer="$"):
        if isinstance(value,dict):
            for key,entry in value.items():yield from leaves(entry,pointer+"."+str(key))
        elif isinstance(value,list):
            for n,entry in enumerate(value):yield from leaves(entry,f"{pointer}[{n}]")
        elif value is not None:yield pointer,value if isinstance(value,str) else canonical(value)
    for pointer, text in leaves(_item_value(item, "scorer_payload", {})):
        if text and re.search(r"answer|expected|gold|label|target|correct|truth|oracle|policy|conflict_type|attack_type", pointer, re.I):
            result.append({"field": pointer, "literal": text})
    category = _item_value(item, "category")
    if isinstance(category, str) and category:
        result.append({"field": "category", "literal": category})
    return result


def _epoch(value):
    if isinstance(value,(int,float)):return float(value)
    if isinstance(value,str):
        try:return datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()
        except ValueError:return None
    return None


def _before(values,request):
    cutoff=_epoch(request["row"].get("timestamp"))
    result=[]
    for value in values:
        if isinstance(value,str):
            # Unit-test supplied explicit catalog; real audit catalogs below
            # always carry the native/provider event timestamp and source.
            result.append(value)
        elif isinstance(value,dict):
            when=_epoch(value.get("timestamp"))
            if cutoff is not None and when is not None and when<=cutoff:
                result.append(value["text"])
    return result


def _baseline_tool_catalog(requests, items, repo):
    """Validate exact source-record identity and source-declared score metadata.

    A digit in session-341 or a cosine is not secretly the category label4.
    This is a structural provenance check, not a blanket exemption for digits.
    Unknown keys, wrong source content, unmatched tool calls, or invalid scores
    never authorize the tool result.
    """
    lookup={_item_value(item,"id"):item for item in items}
    catalog=defaultdict(list)
    for request in requests:
        row=request["row"];ident=row.get("identity",{});arm=ident.get("arm")
        if arm not in {"bm25","embedding"}:continue
        item=lookup.get(ident.get("item_id"))
        if item is None:continue
        conditions=_item_value(item,"condition_inputs",{})
        history=conditions.get(ident.get("condition"),[]) if conditions else _item_value(item,"history",[])
        pending={}
        for message in row.get("payload",{}).get("messages",[]):
            if message.get("role")=="assistant":
                for call in message.get("tool_calls") or []:
                    pending[call.get("id")]=call.get("function",{})
            if message.get("role")!="tool":continue
            call=pending.get(message.get("tool_call_id"),{})
            if call.get("name")!="memory_search":continue
            try:
                args=json.loads(call["arguments"]);hits=json.loads(message["content"])
                if not isinstance(args.get("query"),str) or not isinstance(hits,list):continue
                if len(hits)>min(10,max(1,args.get("k",5))):continue
                ids=[];valid=True
                for hit in hits:
                    if not isinstance(hit,dict) or set(hit)!={"id","content","score"}:valid=False;break
                    match=re.fullmatch(r"session-(\d+)",str(hit["id"]))
                    if not match:valid=False;break
                    index=int(match.group(1));ids.append(index)
                    score=hit["score"]
                    if index>=len(history) or hit["content"]!=history[index] or type(score) not in {int,float} or not math.isfinite(score):valid=False;break
                    if arm=="embedding" and not -1.000001<=score<=1.000001:valid=False;break
                if not valid or len(ids)!=len(set(ids)):continue
                key=(ident.get("item_id"),ident.get("engine_data_dir"),ident.get("condition"))
                catalog[key].append({"text":message["content"],"timestamp":row.get("timestamp"),
                    "path":request["path"],"line":request["line"],"source_kind":"actual_this_item_baseline_tool_result",
                    "source_file":str(repo/"benchmarks/real_engine/retrieval.py"),
                    "source_sha256":sha(repo/"benchmarks/real_engine/retrieval.py")})
            except (ValueError,TypeError,KeyError,IndexError):continue
    return catalog


def literal_overlap_audit(requests, items, native_strings=None, scaffolds=()):
    """Log each literal overlap and its THIS-item provenance; unresolved is fail.

    No exemption merely because text resembles 'normal context'. A model-visible
    string must match actual item input, this item's native output, an earlier
    actual provider response, or an exact pinned generic scaffold. Official judge
    label use is explicitly recorded separately from solver authorization.
    """
    by_id = {_item_value(item, "id"): item for item in items}
    native_strings = native_strings or {}
    overlaps, missing = [], []
    for request in requests:
        ident = request["row"].get("identity", {})
        item_id = ident.get("item_id")
        item = by_id.get(item_id)
        if item is None:
            missing.append({"item_id": item_id, "path": request["path"], "line": request["line"]})
            continue
        condition=ident.get("condition")
        scoped=(item_id,ident.get("engine_data_dir"),condition)
        scoped_entries=list(native_strings.get(scoped,[]))
        actual_native=list(scoped_entries)
        # A single engine per item has an unambiguous backward-compatible key.
        actual_native+=list(native_strings.get((item_id,ident.get("engine_data_dir")),[]))
        actual_native+=list(native_strings.get(item_id,[]))
        actual_native=_before(actual_native,request)
        conditions=_item_value(item,"condition_inputs",{})
        raw_inputs=conditions.get(condition,[]) if conditions else _item_value(item,"history",[])
        if ident.get("arm")=="vestige":raw_inputs=[]
        permitted = list(raw_inputs) + [_item_value(item, "question", "")]
        permitted += actual_native + list(scaffolds)
        phase = str(ident.get("phase", ""))
        judge = "judge" in phase or "scorer" in phase
        for pointer, visible in strings(request["row"].get("payload", {})):
            for target in _gold(item):
                if target["literal"] not in visible:
                    continue
                source = None
                if judge:
                    source = "official_judge_input_requires_exact_scorer_provenance"
                elif any(visible == allowed or visible in allowed for allowed in permitted if isinstance(allowed, str)):
                    source = "exact_this_item_observed_input_or_native_output_or_pinned_scaffold"
                elif pointer.endswith(".content") and any(target["literal"] in allowed and allowed in visible for allowed in actual_native if isinstance(allowed, str)):
                    source = "literal_within_THIS_item_actual_native_output"
                elif phase=="hook":
                    # Original shell hook joins a fixed inline prompt with the
                    # actual user question. Remove only long exact source/input
                    # blocks; token-sized strings cannot launder an oracle.
                    remainder=visible
                    pieces=[p for p in permitted if isinstance(p,str) and len(p)>=25]
                    for piece in sorted(pieces,key=len,reverse=True):
                        remainder=remainder.replace(piece,"")
                    if not remainder.strip():
                        source="exact_pinned_hook_template_plus_THIS_item_observed_input"
                overlaps.append({"item_id": item_id, "path": request["path"], "line": request["line"],
                    "condition":condition,"engine_data_dir":ident.get("engine_data_dir"),
                    "model_pointer": pointer, "gold_field": target["field"], "literal": target["literal"],
                    "visible_string_sha256": hashlib.sha256(visible.encode()).hexdigest(),
                    "provenance": source or "UNRESOLVED_POTENTIAL_LEAK",
                    "provenance_evidence":[{k:v for k,v in entry.items() if k!="text"} for entry in scoped_entries
                        if isinstance(entry,dict) and entry.get("text")==visible and visible in _before([entry],request)],
                    "status": "PENDING" if judge else "PASS" if source else "FAIL"})
    return overlaps, missing


def _native_item(row, checkpoint, runs, manifest):
    identities = row.get("engine_identities") or ([row["engine"]] if isinstance(row.get("engine"), dict) else [])
    if not identities:
        return check("native_item", "FAIL", "Vestige completed row has no actual child identity.", [evidence(checkpoint)])
    observations, seen_paths = [], set()
    request_count = write_count = read_count = 0
    refusal_count = 0
    outputs = []
    for identity in identities:
        declared_data = Path(identity.get("data_dir", ""))
        data = declared_data.resolve()
        if not declared_data.is_absolute() or not data.is_relative_to(runs.resolve()):
            return check("native_item", "FAIL", "Native store escapes independent campaign directory.", [evidence(checkpoint, detail=str(data))])
        if str(data) in seen_paths:
            return check("native_item", "FAIL", "Independent stores repeat a data directory.", [evidence(checkpoint)])
        seen_paths.add(str(data))
        if identity.get("binary_sha256") != manifest.get("binary_sha256") or type(identity.get("pid")) is not int or identity["pid"] <= 0:
            return check("native_item", "FAIL", "Child binary hash or PID missing/mismatched.", [evidence(checkpoint)])
        hooks = identity.get("hooks") or {}
        if identity.get("hooks_enabled") is not True and not hooks:
            return check("native_item", "FAIL", "Actual Vestige item used engine-only profile.", [evidence(checkpoint)])
        log = data.parent / "mcp.jsonl"
        records = lines(log)
        starts = [(number,event.get("identity",{})) for number,event in records if event.get("kind")=="engine_start"]
        if len(starts)!=1 or any(starts[0][1].get(key)!=identity.get(key) for key in ("pid","binary_sha256","data_dir")):
            return check("native_item","FAIL","Completed native row lacks one matching actual engine_start PID/hash/store event.",[evidence(log)])
        started = starts[0][1]
        if (started.get("transport")!="stdio-jsonrpc" or started.get("client")!="benchmarks/memconflict/mcp_client.VestigeMCP"
                or started.get("fresh_store_nodes")!=0):
            return check("native_item","FAIL","Native start does not establish isolated empty stdio MCP child.",[evidence(log,starts[0][0])])
        pending = None
        local_requests = local_responses = local_refusals = 0
        # MPBench's real write gate can return a native MCP error as the
        # measured defense outcome. Only the runner's verified credential
        # refusal classifier and exact per-item raw decision authorize it.
        from .mpbench_runner import security_denial
        permitted_denials=[decision.get("result") for decision in row.get("write_decisions",[])
            if decision.get("defense_positive") is True and security_denial(decision.get("result"))]
        for number, event in records:
            if event.get("kind") == "tool_request":
                if pending is not None:
                    return check("native_item","FAIL","Native request has no matching serialized response.",[evidence(log,number)])
                pending = event.get("name")
                local_requests += 1
                request_count += 1
                write_count += event.get("name") == "smart_ingest"
                read_count += event.get("name") in READ_TOOLS
            if event.get("kind") == "tool_response":
                if not pending or event.get("name")!=pending or event.get("result") is None or (event.get("raw_mcp_result") or {}).get("isError") is True:
                    return check("native_item","FAIL","Native response is unmatched or unsuccessful.",[evidence(log,number)])
                pending = None
                local_responses += 1
                outputs.extend(text for _, text in strings(event.get("result")))
            if event.get("kind") == "tool_error":
                raw=event.get("raw_mcp_result")
                allowed=(row.get("benchmark")=="mpbench" and row.get("defense_positive") is True
                    and pending==event.get("name")=="smart_ingest" and isinstance(raw,dict)
                    and raw.get("isError") is True and security_denial(raw) and raw in permitted_denials)
                if not allowed:
                    return check("native_item","FAIL","Completed native item contains an unmatched error or unverified security refusal.",[evidence(log,number)])
                permitted_denials.remove(raw)
                pending = None
                local_refusals += 1
                refusal_count += 1
                observations.append(evidence(log,number,detail="Exact current-item MPBench normal-write MCP credential refusal; defense-positive decision preserved"))
        if pending is not None or not local_responses or local_requests!=local_responses+local_refusals:
            return check("native_item","FAIL","Completed native item has request-only evidence or missing responses.",[evidence(log,count=local_responses)])
        observations.append(evidence(log, count=request_count, detail=f"pid={identity.get('pid')}; data_dir={data}"))
    if not request_count:
        return check("native_item", "FAIL", "Completed Vestige item has zero recorded MCP calls.", observations)
    return check("native_item", "PASS", f"Actual hashed child MCP calls={request_count}, writes={write_count}, reads={read_count}, verified write refusals={refusal_count}; zero chosen writes is not fabricated admission.", observations)


def _runtime_paths(completed):
    for checkpoint, number, row in completed:
        identities = row.get("engine_identities") or ([row["engine"]] if isinstance(row.get("engine"), dict) else [])
        for identity in identities:
            data = identity.get("data_dir")
            if data:
                yield checkpoint, number, row, identity, Path(data).parent


def _native_provenance(completed):
    catalog = defaultdict(list)
    identities_per_item=defaultdict(set)
    for _,_,row,identity,root in _runtime_paths(completed):
        identities_per_item[row.get("item_id")].add(identity.get("data_dir"))
    for checkpoint, number, row, identity, root in _runtime_paths(completed):
        item = row.get("item_id")
        key=(item,identity.get("data_dir"))
        for line, event in lines(root / "mcp.jsonl"):
            if event.get("kind") == "tool_response":
                texts=[canonical(event.get("result"))]+[text for _,text in strings(event.get("result"))]
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for text in texts)
            if event.get("kind") == "engine_start":
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for _,text in strings(event.get("tools",[])))
        if len(identities_per_item[item])==1:
            catalog[item]=catalog[key]
    return catalog


def _hook_checks(repo, runs, completed):
    audit_path = runs / "hook-audit.json"
    hook_audit = json_file(audit_path, {})
    if hook_audit.get("audit_complete") is not True:
        return check("hook_five_samples", "PENDING", "Full hook source audit absent.", [evidence(audit_path)])
    changed = [name for name, digest in hook_audit.get("source_pins", {}).items()
               if not (repo/name).is_file() or sha(repo/name) != digest]
    if changed:
        return check("hook_five_samples", "FAIL", "Hook source changed after its source audit.", [evidence(repo/name) for name in changed])
    by_command, expected, paths_seen, problems = defaultdict(list), set(), {}, []
    seen_events=set()
    for checkpoint, number, row, identity, root in _runtime_paths(completed):
        log = root / "hooks-runtime/hooks.jsonl"
        records = lines(log)
        if not records:
            problems.append(evidence(log, detail="Completed current-profile item has no hook log"))
            continue
        prepared = [e for _, e in records if e.get("kind") == "prepared"]
        if len(prepared) != 1:
            problems.append(evidence(log, count=len(prepared), detail="Expected one prepared lifecycle"))
            continue
        setup = prepared[0]
        home = Path(setup.get("paths", {}).get("home", "")).resolve()
        group=row.get("group_id") or row.get("item_id")
        if not home.is_relative_to(root.resolve()) or (str(home) in paths_seen and paths_seen[str(home)]!=group):
            problems.append(evidence(log, detail="HOME is ambient or reused between independent stores"))
        paths_seen[str(home)]=group
        ancestry = setup.get("cwd_git_ancestor_check") or {}
        if ancestry.get("git_ancestors") != []:
            problems.append(evidence(log, detail="No preserved successful physical no-Git-ancestor check"))
        for entry in setup.get("wired", []):
            expected.add(entry["command"])
        for line, event in records:
            if (str(log),line) in seen_events:continue
            seen_events.add((str(log),line))
            if event.get("kind") == "bridge_error":
                problems.append(evidence(log, line, detail="Actual hook bridge error: " + str(event.get("error"))))
            if event.get("kind") == "hook_event":
                command = event["entry"]["command"]
                if event.get("exit_code") not in {0, 2}:
                    problems.append(evidence(log, line, detail="Actual hook subprocess failure"))
                visible = event.get("stdout", "") + event.get("stderr", "")
                if "[CWD STATE" in visible and "bench/real-engine" in visible:
                    problems.append(evidence(log, line, detail="Actual campaign Git metadata leaked into hook context"))
                by_command[command].append(evidence(log, line, detail={
                    "item_id": row.get("item_id"), "HOME": str(home),
                    "stdout": event.get("stdout", ""), "stderr": event.get("stderr", ""),
                    "exit_code": event.get("exit_code")}))
    if problems:
        return check("hook_five_samples", "FAIL", "Actual hook isolation/lifecycle failure; do not patch away evidence.", problems)
    missing = sorted(command for command in expected if len(by_command[command]) < 5)
    samples = [entry for command in sorted(by_command) for entry in random.Random(SAMPLE_SEED).sample(by_command[command], min(5, len(by_command[command])))]
    if not expected or missing:
        return check("hook_five_samples", "PENDING", "Need five THIS-item event/injection outcomes per applicable registered hook; missing: " + repr(missing), samples)
    return check("hook_five_samples", "PASS", "Five actual outcomes per registered hook preserved, including natural empty/fail-open outcomes (not invented injections); current HOME paths and source pins checked.", samples)


def _kill_check(repo, runs, plan):
    path = Path(plan.get("kill_control", {}).get("log", ""))
    if not path.is_absolute():
        path = runs / path
    if not path.is_file():
        return check("midrun_kill", "PENDING", "Actual preserved runner mid-run kill control absent; unit fixture does not substitute.", [evidence(path)])
    config=plan.get("config_hash")
    if not config:
        return check("midrun_kill","PENDING","Current candidate config hash absent; stale kill evidence cannot qualify it.",[evidence(path)])
    rows = [(n,row) for n,row in lines(path) if row.get("config_hash")==config]
    indexed = {r.get("event", r.get("kind")): (n, r) for n, r in rows}
    needed = {"engine_started", "midrun_kill", "runner_aborted"}
    if not needed.issubset(indexed):
        return check("midrun_kill", "PENDING", "Kill control lacks actual start/kill/runner-abort evidence.", [evidence(path, count=len(rows))])
    start, kill, abort = (indexed[name][1] for name in ("engine_started", "midrun_kill", "runner_aborted"))
    manifest = json_file(runs/"build-manifest.json", {})
    valid = (type(start.get("pid")) is int and start.get("pid") == kill.get("pid") == abort.get("pid")
             and start.get("binary_sha256") == manifest.get("binary_sha256")
             and type(kill.get("returncode")) is int and kill["returncode"] != 0
             and abort.get("fallback_used") is False and abort.get("exception")
             and abort.get("runner_source_sha256") in {
                 sha(p) for p in (repo/"benchmarks/real_engine").glob("*.py") if not p.name.startswith("test_")})
    return check("midrun_kill", "PASS" if valid else "FAIL", "Actual same-child kill must abort the actual pinned runner with no fallback.", [evidence(path, indexed[k][0]) for k in sorted(needed)])


def _scorer_source(repo, runs, benchmark, plan):
    spec = plan.get("scorer", {})
    pins = spec.get("source_files") or {}
    if not pins or not spec.get("entrypoint"):
        return check("scorer_source_identity", "PENDING", "Exact official scorer source/entrypoint plan not yet recorded.")
    observed = []
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else repo/name
        if not path.is_file():
            return check("scorer_source_identity", "PENDING", "Declared scorer source is absent.", [evidence(path)])
        observed.append(evidence(path, detail=sha(path)))
        if sha(path) != expected:
            return check("scorer_source_identity", "FAIL", "Actual official/frozen scorer source changed.", observed)
    if benchmark == "mpbench":
        path = repo/"benchmarks/real_engine/mpbench.py"
        revision = spec.get("commit")
        if not revision:
            return check("scorer_source_identity", "PENDING", "MPBench scorer pre-execution commit is missing.", observed)
        try:
            committed = subprocess.check_output(["git", "show", f"{revision}:benchmarks/real_engine/mpbench.py"], cwd=repo)
        except subprocess.CalledProcessError:
            return check("scorer_source_identity", "FAIL", "Declared MPBench scorer commit does not contain the scorer.", observed)
        if committed != path.read_bytes():
            return check("scorer_source_identity", "FAIL", "MPBench scorer differs from precommitted artifact.", observed)
    elif spec.get("origin") != "official":
        return check("scorer_source_identity", "FAIL", "Only MPBench has a custom scorer exception.", observed)
    else:
        # A caller cannot bless a home-grown scorer by naming its own digest.
        # Verify primary-source identity independently of the qualification plan.
        from .datasets import SOURCES as QA_SOURCES
        expected_files={}
        if benchmark in QA_SOURCES:
            meta=QA_SOURCES[benchmark]
            expected_files[(runs/"upstream"/benchmark/meta["scorer_file"]).resolve()]=meta["scorer_sha256"]
        elif benchmark=="memconflict":
            from .extra_datasets import MEMCONFLICT_SCORER_SHA256,MEMCONFLICT_PARSER_SHA256
            expected_files={(runs/"upstream/memconflict/Evaluation/eval_scoring.py").resolve():MEMCONFLICT_SCORER_SHA256,
                            (runs/"upstream/memconflict/Evaluation/llm_request.py").resolve():MEMCONFLICT_PARSER_SHA256}
        elif benchmark=="graphwalks":
            from .graphwalks_runner import README_SHA256,_root
            expected_files[(_root(runs/"upstream")/"README.md").resolve()]=README_SHA256
        if expected_files:
            actual_files={(Path(name) if Path(name).is_absolute() else repo/name).resolve():digest for name,digest in pins.items()}
            if not expected_files.items()<=actual_files.items():
                return check("scorer_source_identity","FAIL","Declared scorer omits or changes independently pinned official scoring/parser source.",observed)
        else:
            return check("scorer_source_identity","PENDING","Scorer files match supplied pins, but independent benchmark-specific official scorer resolver is not yet qualified.",observed)
    return check("scorer_source_identity", "PASS", "Exact pinned source and entrypoint recorded; this is not evidence the scorer has run.", observed)


def _full_set(repo, runs, benchmark, plan, official_items):
    declared = plan.get("full_set", {})
    ids, trials = declared.get("item_ids"), declared.get("trials")
    if not ids or not trials:
        return check("official_full_set_plan", "PENDING", "Full official ID/trial set not yet frozen; audit sampling cannot replace it."), set()
    if len(set(ids)) != len(ids) or len(set(trials)) != len(trials):
        return check("official_full_set_plan", "FAIL", "Frozen official plan contains duplicate IDs or trials."), set()
    dataset_pins = declared.get("dataset_files") or {}
    if not dataset_pins:
        return check("official_full_set_plan", "PENDING", "Exact official data source pins absent."), set()
    for name, digest in dataset_pins.items():
        path = Path(name) if Path(name).is_absolute() else repo/name
        if not path.is_file():
            return check("official_full_set_plan", "PENDING", "Declared official source is absent.", [evidence(path)]), set()
        if sha(path) != digest:
            return check("official_full_set_plan", "FAIL", "Official dataset pin mismatch.", [evidence(path)]), set()
    if official_items is None:
        return check("official_full_set_plan", "PENDING", "Auditor has no independent full official ID resolver for this benchmark; source count marker alone cannot pass."), set()
    official = {_item_value(item, "id") for item in official_items}
    if set(ids) != official:
        return check("official_full_set_plan", "FAIL", "Frozen plan is a sample or differs from the full official set.", [evidence(runs/"QUALIFICATION.json", count=len(ids), detail=f"official={len(official)}")]), set()
    required_trials=1 if benchmark=="pmbench" else 3 if len(official)<300 else 1
    if len(trials)!=required_trials:
        return check("official_full_set_plan","FAIL",f"Official item count requires {required_trials} trials/seeds; declared {len(trials)}.",[evidence(runs/"QUALIFICATION.json")]),set()
    expected = {(identity, trial) for identity in ids for trial in trials}
    return check("official_full_set_plan", "PASS", "Full official set matches independent loader exactly.", [evidence(runs/"QUALIFICATION.json", count=len(expected))]), expected


def _merge_and_score(runs, benchmark, arm, expected, plan):
    root = runs/benchmark/arm
    path = root/"merged.jsonl"
    if not expected or not path.is_file():
        return check("strict_complete_merge", "PENDING", "Complete strict-merge evidence is not yet present.", [evidence(path)]), check("scorer_once_after_merge", "PENDING", "Official scoring has not yet been qualified.")
    raw = lines(path)
    if benchmark=="pmbench":
        from .pmbench_runner import expand_pmbench_week_records
        expanded=expand_pmbench_week_records([row for _,row in raw])
        keys=[(row["item_id"],0) for row in expanded]
    else:keys = [(row.get("item_id"), row.get("trial")) for _, row in raw]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        return check("strict_complete_merge", "FAIL", "Observed merged outputs omit, duplicate or add official items/trials.", [evidence(path, count=len(keys), detail=f"expected={len(expected)}")]), check("scorer_once_after_merge", "PENDING", "Scoring must not run on incomplete merge.")
    merged = check("strict_complete_merge", "PASS", "Exact complete official item/trial set, no duplicates.", [evidence(path, count=len(keys), detail=sha(path))])
    log = root/"scorer-invocations.jsonl"
    invocations = [(n,e) for n,e in lines(log) if e.get("event", e.get("kind")) == "official_scorer_complete"]
    if not invocations:
        return merged, check("scorer_once_after_merge", "PENDING", "Preserved official scorer invocation evidence absent.", [evidence(log)])
    valid = (len(invocations) == 1 and invocations[0][1].get("merged_sha256") == sha(path)
             and invocations[0][1].get("item_count") == len(expected)
             and invocations[0][1].get("scorer_source_files") == plan.get("scorer", {}).get("source_files")
             and invocations[0][1].get("output_count") == len(expected))
    return merged, check("scorer_once_after_merge", "PASS" if valid else "FAIL", "Exactly one completed pinned scorer invocation must consume this complete merged hash.", [evidence(log, n) for n,_ in invocations])


def _scaffolds(repo):
    """Actual source constants with file/line provenance, never guessed labels."""
    result = {}
    paths = [repo/"benchmarks/real_engine/prompts.py"] + list((repo/"hooks").glob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                result.setdefault(node.value, []).append(evidence(path, node.lineno))
    for path in (repo/"hooks").glob("*.sh"):
        text=path.read_text()
        for match in re.finditer(r"<<'([A-Z_]+)'\n(.*?)\n\1",text,re.S):
            block=match.group(2)+"\n"
            result.setdefault(block,[]).append(evidence(path,text[:match.start()].count("\n")+1))
    # These are exact payload metadata, not guessed question/answer content.
    for value in ("system", "user", "assistant", "tool", MODEL, "none", "function", "string", "object"):
        result.setdefault(value, []).append({"path": "OpenAI Chat Completions payload metadata"})
    from .prompts import audit_prompts
    for value in audit_prompts().values():
        result[value["text"]] = [evidence(repo/"benchmarks/real_engine/prompts.py")]
    return result


def _parity(requests, plan, benchmark, official_items, responses=None, runs=None):
    caps = plan.get("budgets", {})
    if set(caps) != set(ARMS) or any(not all(k in caps[a] for k in ("max_calls", "max_output", "context")) for a in ARMS):
        return check("prompt_budget_parity", "PENDING", "Need equal declared aggregate solver-plus-hook call/output/context caps for all four arms.")
    if len({canonical(caps[a]) for a in ARMS}) != 1:
        return check("prompt_budget_parity", "FAIL", "Declared aggregate arm budgets differ.")
    by_item, first, violations = defaultdict(list), defaultdict(dict), []
    actual_output = defaultdict(int)
    pending_usage = []
    reply_map = defaultdict(list)
    for arm in ARMS:
        for reply in (responses or {}).get((benchmark,arm),[]):
            reply_map[arm,reply["row"].get("call_id")].append(reply)
    ledger = None
    ledger_path = Path(runs)/"budget.db" if runs is not None else None
    if ledger_path is not None and ledger_path.is_file():
        ledger = sqlite3.connect(f"file:{ledger_path}?mode=ro",uri=True)
        ledger.row_factory=sqlite3.Row
    for arm in ARMS:
        for entry in requests.get((benchmark, arm), []):
            ident, payload = entry["row"].get("identity", {}), entry["row"].get("payload", {})
            item = ident.get("item_id")
            if not item or "judge" in str(ident.get("phase", "")):
                continue
            # A retried item has a new isolated attempt directory and a fresh
            # equal cap. Never combine its abandoned attempt with its success.
            key=(arm,item,ident.get("trial",0),str(Path(entry["path"]).parent))
            by_item[key].append(entry)
            replies=reply_map.get((arm,entry["row"].get("call_id")),[])
            if len(replies)>1:
                violations.append(evidence(Path(entry["path"]),entry["line"],detail="Duplicate provider response for one call identity"))
            elif not replies or ledger is None:
                pending_usage.append(evidence(Path(entry["path"]),entry["line"],detail="Actual response usage and shared settled ledger entry required"))
            else:
                reply=replies[0]
                result=reply["row"]
                usage=result.get("usage")
                try:
                    from .provider import account
                    from .runtime import _money
                    cost,actual_usage=account(result.get("response",{}))
                    if any(result.get("identity",{}).get(k)!=ident.get(k) for k in ("item_id","arm","trial","config_hash")):
                        raise ValueError("response identity differs from its request")
                    if (not isinstance(usage,dict) or actual_usage!=usage
                            or any(type(usage.get(k)) is not int or usage[k]<0 for k in ("prompt_tokens","completion_tokens"))
                            or _money(result.get("cost"))!=_money(cost)):
                        raise ValueError("response usage/cost mismatch")
                    matches=ledger.execute("SELECT * FROM calls WHERE substr(id,1,?)=? AND state='settled'",
                        (len(str(result.get("call_id")))+9,str(result.get("call_id"))+"/attempt-")).fetchall()
                    if len(matches)!=1 or matches[0]["actual"]!=_money(cost) or json.loads(matches[0]["usage"])!=usage:
                        raise ValueError("usage is not reconciled to exactly one settled shared-ledger attempt")
                    actual_output[key]+=usage["completion_tokens"]
                    if usage["completion_tokens"]>payload.get("max_completion_tokens",0):
                        raise ValueError("actual completion exceeds requested cap")
                except (ValueError,TypeError,KeyError,RuntimeError,ArithmeticError,sqlite3.Error) as exc:
                    violations.append(evidence(Path(reply["path"]),reply["line"],detail=str(exc)))
            if ident.get("phase") not in {"hook", "memory", "compiler"}:
                first[item].setdefault(arm, entry)
            if payload.get("max_completion_tokens", 0) > caps[arm]["max_output"]:
                violations.append(evidence(Path(entry["path"]), entry["line"], detail="Request output cap exceeds arm budget"))
            try:
                import tiktoken
                count = len(tiktoken.get_encoding("o200k_base").encode(canonical({"messages":payload.get("messages",[]),"tools":payload.get("tools",[])})))
            except Exception:
                count = None
            if count is None:
                return check("prompt_budget_parity", "PENDING", "Cannot verify actual context token count with pinned tokenizer.")
            if count > caps[arm]["context"]:
                violations.append(evidence(Path(entry["path"]), entry["line"], count=count, detail="Actual request exceeds context budget"))
    if ledger is not None:
        ledger.close()
    for key, entries in by_item.items():
        arm=key[0]
        if len(entries) > caps[arm]["max_calls"]:
            violations.append(evidence(Path(entries[0]["path"]), count=len(entries), detail="Aggregate actual solver-plus-hook calls exceed cap"))
        if actual_output[key]>caps[arm]["max_output"]:
            violations.append(evidence(Path(entries[0]["path"]),count=actual_output[key],detail="Actual aggregate solver-plus-hook completion usage exceeds cap"))
    common = [item for item, arms in first.items() if set(arms) == set(ARMS)]
    item_map = {_item_value(item,"id"):item for item in official_items or []}
    observations=[]
    for item in random.Random(SAMPLE_SEED).sample(common,min(SAMPLE_ITEMS,len(common))):
        normalized=[]
        for arm in ARMS:
            entry=first[item][arm]
            messages=entry["row"]["payload"].get("messages",[])
            system="\n".join(m.get("content") or "" for m in messages if m.get("role")=="system")
            system="\n".join(line for line in system.splitlines() if not line.startswith("Tools:"))
            users=[m.get("content") for m in messages if m.get("role")=="user"]
            # Later user contexts from hooks are checked separately. Initial
            # task stimulus remains identical, except explicit GraphWalks
            # full-context versus graph-memory placement required by the owner.
            question=users[0] if users else None
            if benchmark=="graphwalks" and arm=="bare" and item in item_map:
                original=_item_value(item_map[item],"full_prompt")
                if question!=original:
                    violations.append(evidence(Path(entry["path"]),entry["line"],detail="GraphWalks bare context is not exact official full prompt"))
                question=_item_value(item_map[item],"question")
            normalized.append((system,question))
            observations.append(evidence(Path(entry["path"]),entry["line"]))
        if len({canonical(x) for x in normalized})!=1:
            violations.append(evidence(Path(first[item]["vestige"]["path"]),first[item]["vestige"]["line"],detail="Actual initial prompts differ beyond neutral tools line and explicit GraphWalks placement"))
    if violations:
        return check("prompt_budget_parity","FAIL","Observed prompt or aggregate budget mismatch.",violations)
    if pending_usage:
        return check("prompt_budget_parity","PENDING","Request-only evidence cannot establish parity: actual response usage must reconcile to shared budget ledger.",pending_usage)
    if len(common)<SAMPLE_ITEMS:
        return check("prompt_budget_parity","PENDING","Need10 common actual unscored items across all four arms to compare full prompts and aggregate caps.",observations)
    return check("prompt_budget_parity","PASS","Ten four-arm actual initial prompts match; aggregate solver-plus-hook caps verified against actual response usage and shared settled ledger.",observations)


def _pilot_check(plan,arm,reqs,completed,official_items=None):
    """Only the frozen ten IDs under this exact candidate can qualify it."""
    pilot=plan.get("pilot",{})
    config=plan.get("config_hash")
    ids=pilot.get("item_ids",[])
    expected=plan.get("run_identities",{}).get(arm,[])
    if isinstance(expected,str):expected=[expected]
    if (not config or pilot.get("config_hash")!=config or pilot.get("seed")!=SAMPLE_SEED
            or len(ids)!=SAMPLE_ITEMS or len(set(ids))!=SAMPLE_ITEMS or not expected):
        return check("ten_unscored_pilot","PENDING","Frozen ten-item seed/config/checkpoint identity plan absent or incomplete.")
    if official_items is not None:
        official_ids=[_item_value(item,"id") for item in official_items]
        frozen=random.Random(SAMPLE_SEED).sample(official_ids,min(SAMPLE_ITEMS,len(official_ids)))
        if set(ids)!=set(frozen):
            return check("ten_unscored_pilot","FAIL","Declared pilot IDs differ from fixed-seed official-item sample.")
    observed={r["row"].get("identity",{}).get("item_id") for r in reqs
              if r["row"].get("identity",{}).get("config_hash")==config}
    qualified={row.get("item_id") for _,_,row in completed
               if row.get("config_hash")==config and row.get("run_identity") in expected
               and row.get("phase")=="integrity_preflight" and row.get("score") is None}
    refs=[evidence(path,number,detail={"item_id":row.get("item_id"),"run_identity":row.get("run_identity"),"config_hash":row.get("config_hash")})
          for path,number,row in completed if row.get("item_id") in ids and row.get("config_hash")==config]
    return check("ten_unscored_pilot","PASS" if set(ids)<=observed&qualified else "PENDING",
                 "Exactly the frozen ten official IDs require unscored current-config requests and completed current-checkpoint records.",refs)


def _build_check(repo,runs,manifest):
    path=repo/"target/release/vestige-mcp"
    if not manifest or not path.is_file():
        return check("child_build_source","PENDING","Required repo release binary/build manifest absent.",[evidence(path)])
    if path.is_symlink() or sha(path)!=manifest.get("binary_sha256"):
        return check("child_build_source","FAIL","Actual release artifact differs from pinned binary.",[evidence(path)])
    from .engine import verify_build
    try:verify_build(repo,runs/"build-manifest.json")
    except Exception as exc:
        return check("child_build_source","FAIL","Actual build provenance failed: "+str(exc),[evidence(runs/"build-manifest.json")])
    return check("child_build_source","PASS","Locked repository build and current native source/binary identity verified.",[evidence(path,detail=sha(path))])


_AUDIT_LOCK=threading.RLock()


def audit(repo: str | Path, runs: str | Path, stage: str = "report", benchmarks=None) -> dict:
    with _AUDIT_LOCK:
        return _audit(repo,runs,stage,benchmarks)


def _audit(repo: str | Path, runs: str | Path, stage: str = "report", benchmarks=None) -> dict:
    if stage not in {"preflight", "execute", "score", "scorer", "report"}:
        raise ValueError("unknown audit stage")
    repo, runs = Path(repo).resolve(), Path(runs).resolve()
    selected=tuple(benchmarks) if benchmarks is not None else BENCHMARKS
    if not selected or set(selected)-set(BENCHMARKS):
        raise ValueError("unknown or empty benchmark audit scope")
    manifest = json_file(runs / "build-manifest.json", {})
    prior_audit=json_file(runs/"integrity.json",{})
    build_check=_build_check(repo,runs,manifest)
    qualification = json_file(runs/"QUALIFICATION.json", {})
    plans = qualification.get("benchmarks", {})
    requests = defaultdict(list)
    checkpoints = defaultdict(list)
    responses = defaultdict(list)
    parse_failures = defaultdict(list)
    excluded_profiles = defaultdict(int)
    for path in _walk(runs):
        if path.name not in {"api.jsonl", "judge.jsonl", "hook-api.jsonl", "hooks-model.jsonl"} and not re.fullmatch(r"worker-\d+\.jsonl", path.name):
            continue
        try:
            rows = lines(path)
        except (ValueError, OSError) as exc:
            relative = path.relative_to(runs).parts
            parse_failures[_base(relative[0])].append(evidence(path, detail=type(exc).__name__))
            continue
        for number, row in rows:
            ident = row.get("identity", row)
            benchmark, arm = _base(str(ident.get("benchmark", path.relative_to(runs).parts[0]))), ident.get("arm")
            if arm not in ARMS or benchmark not in selected:
                continue
            active_config=plans.get(benchmark,{}).get("config_hash")
            if active_config and ident.get("config_hash")!=active_config:
                excluded_profiles[benchmark,arm]+=1
                continue
            if row.get("event") == "request":
                requests[benchmark, arm].append({"path": str(path), "line": number, "row": row})
            elif row.get("event") == "response":
                responses[benchmark, arm].append({"path":str(path),"line":number,"row":row})
            elif path.name.startswith("worker-") and row.get("status", "complete") == "complete":
                expected=plans.get(benchmark,{}).get("run_identities",{}).get(arm,[])
                if isinstance(expected,str):expected=[expected]
                if expected and row.get("run_identity") not in expected:
                    excluded_profiles[benchmark,arm]+=1
                    continue
                checkpoints[benchmark, arm].append((path, number, row))
    benchmarks, approved, invalid = {}, {}, []
    for name in selected:
        arms, official_items = {}, None
        plan=plans.get(name,{})
        if plan.get("full_set") or any(requests[name, arm] for arm in ARMS):
            try:
                official_items = _load_items(repo, runs, name)
            except (FileNotFoundError,ImportError) as exc:
                # Unprepared official sources are PENDING, not a fabricated
                # pass and not an observed benchmark integrity failure.
                official_items=None
            except Exception as exc:
                parse_failures[name].append(evidence(runs / "upstream", detail=f"official audit loader: {type(exc).__name__}: {exc}"))
        source_plan=_scorer_source(repo,runs,name,plan)
        full_plan,expected=_full_set(repo,runs,name,plan,official_items)
        # Failed/abandoned attempts remain visible to leakage checks below, but
        # their request-only logs cannot establish completed-attempt usage.
        completed_roots=set()
        for arm in ARMS:
            for _,_,row in checkpoints[name,arm]:
                raw=row.get("raw_log")
                if raw:
                    raw=Path(raw)
                    completed_roots.add(str((raw if raw.is_absolute() else repo/raw).resolve().parent))
        parity_requests={key:[entry for entry in entries if str(Path(entry["path"]).resolve().parent) in completed_roots]
                         for key,entries in requests.items() if key[0]==name}
        parity=_parity(parity_requests,plan,name,official_items,responses,runs)
        for arm in ARMS:
            reqs, completed = requests[name, arm], checkpoints[name, arm]
            checks = [_source_check(repo, name),build_check]
            if parse_failures[name]:
                checks.append(check("evidence_parse", "FAIL", "Actual evidence is corrupt or official identity cannot be verified.", parse_failures[name]))
            else:
                checks.append(check("evidence_parse", "PASS", "No malformed active evidence observed."))
            observed_ids={r['row'].get('identity',{}).get('item_id') for r in reqs if r['row'].get('identity',{}).get('item_id')}
            completed_ids={row.get('item_id') for _,_,row in completed if row.get('score') is None}
            checks.append(_pilot_check(plan,arm,reqs,completed,official_items))
            bad = [r for r in reqs if r["row"].get("payload", {}).get("model") != MODEL or r["row"].get("payload", {}).get("temperature") != 0]
            checks.append(check("model_temperature", "FAIL" if bad else "PASS" if reqs else "PENDING",
                "Every observed model request must use nano and temperature0.",
                [evidence(Path(r['path']), r['line']) for r in bad or reqs[:1]]))
            overlaps = []
            if reqs and official_items is not None:
                candidates=sorted(observed_ids)
                chosen=set(random.Random(SAMPLE_SEED).sample(candidates,min(SAMPLE_ITEMS,len(candidates))))
                sample=[r for r in reqs if r["row"].get("identity",{}).get("item_id") in chosen]
                provenance_rows=list(completed)
                if arm=="vestige":
                    for request in reqs:
                        ident=request["row"].get("identity",{})
                        if isinstance(ident.get("engine"),dict):
                            provenance_rows.append((Path(request["path"]),request["line"],ident))
                        else:
                            # A stopped hook can precede the first solver call
                            # and completed checkpoint. Bind only the single
                            # native child in THIS request's own attempt folder;
                            # never pick another MemForm condition/slot.
                            parent=Path(request["path"]).parent
                            candidate=parent/"native/mcp.jsonl"
                            for line,event in lines(candidate):
                                child=event.get("identity",{})
                                if (event.get("kind")=="engine_start" and child.get("binary_sha256")==manifest.get("binary_sha256")
                                    and child.get("data_dir")==str((parent/"native/strata").resolve())):
                                    provenance_rows.append((candidate,line,{**ident,"engine":child}))
                native=_native_provenance(provenance_rows) if arm=="vestige" else defaultdict(list)
                for key,entries in _baseline_tool_catalog(sample,official_items,repo).items():
                    native[key].extend(entries)
                for reply in responses[name,arm]:
                    identity=reply["row"].get("identity",{})
                    item=identity.get("item_id")
                    message=reply["row"].get("response",{}).get("choices",[{}])[0].get("message",{})
                    key=(item,identity.get("engine_data_dir"),identity.get("condition"))
                    native[key].extend({"text":text,"timestamp":reply["row"].get("timestamp"),"path":reply["path"],"line":reply["line"]} for _,text in strings(message))
                scaffold_map=_scaffolds(repo)
                # Baseline retrieval may expose only actual raw official
                # records. Vestige retrieval must instead match native output.
                if arm!="vestige":
                    for item in official_items:
                        if _item_value(item,"id") not in chosen:continue
                        if not _item_value(item,"condition_inputs",{}):
                            native[_item_value(item,"id")].extend(_item_value(item,"history",[]))
                overlaps, missing = literal_overlap_audit(sample, official_items, native_strings=native,scaffolds=scaffold_map)
                # Judge overlaps require this exact official source plan; they
                # remain explicitly logged, never authorize solver exposure.
                for overlap in overlaps:
                    if overlap["provenance"]=="official_judge_input_requires_exact_scorer_provenance" and source_plan["status"]=="PASS":
                        overlap["status"]="PASS"
                        overlap["provenance"]="official_judge_ONLY; exact scorer source pins verified"
                    overlap["source_scaffold_evidence"]=scaffold_map.get(overlap["literal"],[])
                status = "FAIL" if any(o["status"] == "FAIL" for o in overlaps) else "PENDING" if missing or any(o["status"] == "PENDING" for o in overlaps) else "PASS"
                checks.append(check("model_visible_gold_provenance", status,
                    "All sampled payload strings compared literally with official labels; unresolved overlaps are not waived.",
                    [evidence(Path(r['path']), r['line'], count=sum(1 for _ in strings(r['row'].get('payload',{})))) for r in sample]))
            else:
                checks.append(check("model_visible_gold_provenance", "PENDING", "No actual requests or no supported official label resolver; 10 unscored samples required."))
            if arm == "vestige":
                native = [_native_item(row, path, runs, manifest) for path, _, row in completed]
                checks.extend(native or [check("native_item", "PENDING", "No actual current-profile completed item to inspect.")])
                checks.append(_kill_check(repo,runs,plan))
                checks.append(_hook_checks(repo,runs,completed))
            checks.extend([parity,source_plan,full_plan])
            pre_score=list(checks)
            merge,scored=_merge_and_score(runs,name,arm,expected,plan)
            checks.extend([merge,scored])
            status = combine(checks)
            arms[arm] = {"status": status, "ready_to_score":combine(pre_score)=="PASS",
                         "ready_for_official_scorer":combine(pre_score+[merge])=="PASS",
                         "ready_to_report":status=="PASS", "checks": checks, "completed_items": len(completed),
                         "actual_model_requests": len(reqs), "literal_overlaps": overlaps,
                         "excluded_obsolete_profile_records":excluded_profiles[name,arm]}
        status = combine([{"status": value["status"]} for value in arms.values()])
        preflight_checks=[_source_check(repo,name),build_check]
        hook_source=json_file(runs/"hook-audit.json",{})
        pins_match=bool(hook_source.get("source_pins")) and all((repo/p).is_file() and sha(repo/p)==digest for p,digest in hook_source.get("source_pins",{}).items())
        preflight_ok=(combine(preflight_checks)=="PASS" and hook_source.get("audit_complete") is True and pins_match and status!="FAIL")
        benchmarks[name] = {"status": status, "excluded": status != "PASS", "arms": arms,
                            "ready_to_prequalify":preflight_ok,
                            "ready_to_score":status!="FAIL" and all(a["ready_to_score"] for a in arms.values()),
                            "ready_for_official_scorer":status!="FAIL" and all(a["ready_for_official_scorer"] for a in arms.values()),
                            "ready_to_report":status=="PASS"}
        if status == "FAIL":
            invalid.append(name)
        approved[name] = {arm: sorted({row.get("run_identity") for _, _, row in checkpoints[name, arm] if row.get("run_identity")}) if value["ready_to_score"] else [] for arm, value in arms.items()}
    report = {"schema": "vestige.real_engine.integrity.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "status": combine([{"status": b["status"]} for b in benchmarks.values()]),
              "stage":stage,"ready_to_prequalify":any(b["ready_to_prequalify"] for b in benchmarks.values()),
              "ready_to_score":any(b["ready_to_score"] for b in benchmarks.values()),
              "ready_to_report":any(b["ready_to_report"] for b in benchmarks.values()),
              "benchmarks": benchmarks, "invalid_benchmarks": invalid, "approved_run_identities": approved,
              "required_prequalification": {"items_per_arm": SAMPLE_ITEMS, "hook_samples_per_hook": 5, "scored": False,
                                           "sampling_does_not_reduce_official_scored_set": True},
              "prior_preparation": {"status": "INVALID_FOR_RESULTS", "path": str(runs / "superseded-engine-only"),
                                    "reason": "Engine-only pilots are preparation, not current all-hooks results."}}
    historical=runs/"invalidated/hook-cwd-git-negative-control/hooks.jsonl"
    report["historical_invalidations"]=[{"status":"FAIL_INVALID_PROFILE","path":str(historical),
          "count":len(lines(historical)),"reason":"Observed original hook leaked harness Git metadata; no official scored items associated. Old adapter hash unavailable; failure is preserved, not erased by a new candidate."}] if historical.is_file() else []
    corrections=list(prior_audit.get("auditor_provenance_clarifications",[]))
    for name,bench in prior_audit.get("benchmarks",{}).items():
        for arm,old in bench.get("arms",{}).items():
            current=benchmarks.get(name,{}).get("arms",{}).get(arm,{})
            for previous in old.get("literal_overlaps",[]):
                if previous.get("status")!="FAIL":continue
                for present in current.get("literal_overlaps",[]):
                    if (present.get("status")=="PASS" and present.get("visible_string_sha256")==previous.get("visible_string_sha256")
                        and present.get("literal")==previous.get("literal") and present.get("provenance_evidence")):
                        clarification={"benchmark":name,"arm":arm,"initial_finding":previous,
                            "verified_source_evidence":present["provenance_evidence"],
                            "reason":"Previously unresolved literal was traced to exact THIS-item baseline record IDs/numeric scores; candidate code/output unchanged. Initial finding is preserved, not silently waived."}
                        if clarification not in corrections:corrections.append(clarification)
    report["auditor_provenance_clarifications"]=corrections
    runs.mkdir(parents=True, exist_ok=True)
    from .runtime import atomic_write
    atomic_write(runs / "integrity.json",json.dumps(report, indent=2) + "\n")
    out = ["# Benchmark integrity gate", "", f"Status: **{report['status']}**. No missing runtime check is a PASS.", "",
           "Ten unscored pilot items per arm qualify evidence only. Every scored benchmark still requires its full official set; the audit sample is not permission to shrink it.", "",
           "Observed FAIL requires stopping that benchmark, marking it INVALID, preserving evidence and excluding metrics. PENDING means unexecuted or incomplete evidence and also blocks scoring/report metrics.", "",
           "| Benchmark | Arm | Status | Completed items | Actual model requests |", "|---|---|---|---:|---:|"]
    for name, benchmark in benchmarks.items():
        for arm, row in benchmark["arms"].items():
            out.append(f"| {name} | {arm} | {row['status']} | {row['completed_items']} | {row['actual_model_requests']} |")
    for name, benchmark in benchmarks.items():
        out += ["", f"## {name}", ""]
        for arm, row in benchmark["arms"].items():
            out += [f"**{arm}: {row['status']}**", ""]
            for c in row["checks"]:
                refs = "; ".join(f"{e['path']}" + (f":{e['line']}" if e.get('line') else "") + (f" count={e['count']}" if 'count' in e else "") for e in c["evidence"])
                out.append(f"- {c['status']} `{c['id']}`: {c['reason']} Evidence: {refs or 'not yet present'}.")
            out.append("")
    out += ["Earlier engine-only preparation is INVALID for result tables. Exact literal-overlap details and provenance are retained in integrity.json.", ""]
    for issue in report["historical_invalidations"]:
        out += [f"Historical observed FAIL: {issue['path']} count={issue['count']}. {issue['reason']}",""]
    atomic_write(runs / "INTEGRITY.md","\n".join(out))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--stage",choices=("preflight","execute","score","scorer","report"),default="report")
    parser.add_argument("--benchmarks",help="Comma-separated exact audit scope; default retains legacy full inventory")
    args = parser.parse_args()
    result = audit(args.repo, args.runs or args.repo / "runs",stage=args.stage,
                   benchmarks=args.benchmarks.split(",") if args.benchmarks else None)
    print(canonical({"status": result["status"], "invalid_benchmarks": result["invalid_benchmarks"],
                     "ready_to_score": result["ready_to_score"], "ready_to_report": result["ready_to_report"]}))


if __name__ == "__main__":
    main()

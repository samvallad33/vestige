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
    # JSONL records are LF-delimited. str.splitlines() also splits literal
    # U+2028/U+2029/U+0085 inside valid JSON strings and invents corrupt rows.
    for number, line in enumerate(contents.split("\n"), 1):
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
    if "audited_benchmarks" in report and benchmark not in report["audited_benchmarks"]:
        raise IntegrityPending(f"{benchmark}: retained as-of evidence is not a fresh audit for this action")
    entry = report["benchmarks"][benchmark]
    if entry["status"] == "FAIL":
        raise IntegrityInvalid(f"{benchmark} INVALID: integrity failure; stop and exclude metrics")
    flag = {"preflight": "ready_to_prequalify", "execute": "ready_to_score", "score": "ready_to_score",
            "scorer": "ready_for_official_scorer", "report": "ready_to_report"}.get(stage)
    if flag is None:
        raise ValueError("unknown integrity stage")
    if not entry.get(flag):
        raise IntegrityPending(f"{benchmark}: integrity evidence incomplete; no {stage} permitted")


def _walk(runs: Path, selected=None):
    selected=set(selected) if selected is not None else None
    for directory, folders, files in os.walk(runs):
        folders[:] = [name for name in folders if name not in EXCLUDED and not name.startswith(".")
                      and (Path(directory)!=runs or selected is None or _base(name) in selected)]
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
    gold_by_id={identity:_gold(item) for identity,item in by_id.items()}
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
        actual_native=list(dict.fromkeys(_before(actual_native,request)))
        conditions=_item_value(item,"condition_inputs",{})
        raw_inputs=conditions.get(condition,[]) if conditions else _item_value(item,"history",[])
        if ident.get("arm")=="vestige":raw_inputs=[]
        hook_permitted=list(raw_inputs)+[_item_value(item,"question","")]+actual_native+list(scaffolds)
        # Exact matching is common for full native result bodies and schemas.
        # Keep substring fallback identical, with cheap pinned/raw inputs first.
        permitted=list(dict.fromkeys([value for value in list(scaffolds)+list(raw_inputs)
                   +[_item_value(item,"question","")]+actual_native if isinstance(value,str)]))
        permitted_set=set(permitted)
        base_provenance={}
        def permitted_substring(visible):
            if visible not in base_provenance:
                base_provenance[visible]=any(visible in allowed for allowed in permitted)
            return base_provenance[visible]
        phase = str(ident.get("phase", ""))
        judge = "judge" in phase or "scorer" in phase
        for pointer, visible in strings(request["row"].get("payload", {})):
            for target in gold_by_id[item_id]:
                if target["literal"] not in visible:
                    continue
                source = None
                if judge:
                    source = "official_judge_input_requires_exact_scorer_provenance"
                elif visible in permitted_set:
                    source = "exact_this_item_observed_input_or_native_output_or_pinned_scaffold"
                elif permitted_substring(visible):
                    source = "exact_this_item_observed_input_or_native_output_or_pinned_scaffold"
                elif pointer.endswith(".content") and any(target["literal"] in allowed and allowed in visible for allowed in actual_native if isinstance(allowed, str)):
                    source = "literal_within_THIS_item_actual_native_output"
                elif phase=="hook":
                    # Original shell hook joins a fixed inline prompt with the
                    # actual user question. Remove only long exact source/input
                    # blocks; token-sized strings cannot launder an oracle.
                    remainder=visible
                    pieces=[p for p in hook_permitted if isinstance(p,str) and len(p)>=25]
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


def _forwarded_native_errors(row,seen_ids=None):
    """Bind raw model-visible tool replies to this transcript's actual calls."""
    calls, forwarded = {}, []
    seen_ids=set() if seen_ids is None else seen_ids
    for message in row.get("transcript",[]):
        if message.get("role")=="assistant":
            for call in message.get("tool_calls") or []:
                function=call.get("function") or {}
                try:
                    arguments=json.loads(function.get("arguments",""))
                except (ValueError,TypeError):
                    continue
                if call.get("id") and isinstance(arguments,dict):
                    name=function.get("name")
                    if isinstance(name,str):name=name.removeprefix("mcp__vestige__")
                    calls[call["id"]]=(name,arguments)
        elif message.get("role")=="tool":
            call=calls.pop(message.get("tool_call_id"),None)
            call_id=message.get("tool_call_id")
            if call and isinstance(message.get("content"),str) and call_id not in seen_ids:
                seen_ids.add(call_id)
                forwarded.append((*call,message["content"]))
    return forwarded


def _native_item(row, checkpoint, runs, manifest):
    identities = row.get("engine_identities") or ([row["engine"]] if isinstance(row.get("engine"), dict) else [])
    if not identities:
        return check("native_item", "FAIL", "Vestige completed row has no actual child identity.", [evidence(checkpoint)])
    observations, seen_paths = [], set()
    request_count = write_count = read_count = 0
    refusal_count = forwarded_error_count = 0
    seen_forwarded_ids=set()
    forwarded_errors=_forwarded_native_errors(row,seen_forwarded_ids)
    hook_forwarded_errors=[]
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
        # Original hooks have their own model dialogue. Authenticate returned
        # native errors against that item's actual subsequent API payloads,
        # preserving original allowed-tool names and exact raw error bytes.
        hook_api=data.parent.parent/"hook-api.jsonl"
        if row.get("config_hash"):
            for _,event in lines(hook_api):
                ident=event.get("identity",{})
                if (event.get("event")=="request" and ident.get("item_id")==row.get("item_id")
                        and ident.get("arm")=="vestige" and ident.get("config_hash")==row["config_hash"]):
                    when=_epoch(event.get("timestamp"))
                    hook_forwarded_errors.extend((binding,when) for binding in
                        _forwarded_native_errors({"transcript":event.get("payload",{}).get("messages",[])},seen_forwarded_ids)
                        if when is not None)
        records = lines(log)
        starts = [(number,event.get("identity",{})) for number,event in records if event.get("kind")=="engine_start"]
        if len(starts)!=1 or any(starts[0][1].get(key)!=identity.get(key) for key in ("pid","binary_sha256","data_dir")):
            return check("native_item","FAIL","Completed native row lacks one matching actual engine_start PID/hash/store event.",[evidence(log)])
        started = starts[0][1]
        if (started.get("transport")!="stdio-jsonrpc" or started.get("client")!="benchmarks/memconflict/mcp_client.VestigeMCP"
                or started.get("fresh_store_nodes")!=0):
            return check("native_item","FAIL","Native start does not establish isolated empty stdio MCP child.",[evidence(log,starts[0][0])])
        pending, pending_arguments = None, None
        local_requests = local_responses = local_refusals = local_forwarded_errors = 0
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
                pending_arguments = event.get("arguments")
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
                security_refusal=(row.get("benchmark")=="mpbench" and pending=="smart_ingest" and security_denial(raw))
                allowed_refusal=(security_refusal and row.get("defense_positive") is True
                    and pending==event.get("name")=="smart_ingest" and isinstance(raw,dict)
                    and raw.get("isError") is True and security_denial(raw) and raw in permitted_denials)
                forwarded=(pending,pending_arguments,canonical(raw))
                error_time=_epoch(event.get("timestamp"))
                hook_match=next((index for index,(binding,when) in enumerate(hook_forwarded_errors)
                                 if binding==forwarded and error_time is not None and when>=error_time),None)
                allowed_error=(not security_refusal and pending and pending==event.get("name")
                    and isinstance(raw,dict) and raw.get("isError") is True
                    and (forwarded in forwarded_errors or hook_match is not None))
                if not allowed_refusal and not allowed_error:
                    return check("native_item","FAIL","Completed native item contains an unmatched error or unverified security refusal.",[evidence(log,number)])
                if allowed_refusal:
                    permitted_denials.remove(raw)
                    local_refusals += 1
                    refusal_count += 1
                    detail="Exact current-item MPBench normal-write MCP credential refusal; defense-positive decision preserved"
                else:
                    if forwarded in forwarded_errors:forwarded_errors.remove(forwarded)
                    else:hook_forwarded_errors.pop(hook_match)
                    local_forwarded_errors += 1
                    forwarded_error_count += 1
                    detail="Actual MCP tool error forwarded unchanged to this completed item's matching model tool call"
                pending = None
                observations.append(evidence(log,number,detail=detail))
        if pending is not None or not local_responses or local_requests!=local_responses+local_refusals+local_forwarded_errors:
            return check("native_item","FAIL","Completed native item has request-only evidence or missing responses.",[evidence(log,count=local_responses)])
        observations.append(evidence(log, count=request_count, detail=f"pid={identity.get('pid')}; data_dir={data}"))
    if not request_count:
        return check("native_item", "FAIL", "Completed Vestige item has zero recorded MCP calls.", observations)
    return check("native_item", "PASS", f"Actual hashed child MCP calls={request_count}, writes={write_count}, reads={read_count}, verified write refusals={refusal_count}, unchanged forwarded tool errors={forwarded_error_count}; zero chosen writes is not fabricated admission.", observations)


def _full_native_contract_check(row,checkpoint,requests,repo=None):
    identity=row.get("engine") or {}
    root=Path(identity.get("data_dir",".")).parent
    log=root/"mcp.jsonl"
    starts=[(number,event) for number,event in lines(log) if event.get("kind")=="engine_start"]
    if len(starts)!=1:
        return check("full_native_mcp_contract","FAIL","Full MCP contract lacks one actual child start.",[evidence(log)])
    number,event=starts[0]
    catalog=event.get("tools")
    initialize=event.get("initialize")
    if not isinstance(catalog,list) or not isinstance(initialize,dict) or not isinstance(initialize.get("instructions"),str) or not initialize["instructions"]:
        return check("full_native_mcp_contract","FAIL","Actual complete native catalog/initialize instructions were not captured.",[evidence(log,number)])
    names=[tool.get("name") for tool in catalog]
    if not names or len(names)!=len(set(names)) or "memory_status" not in names:
        return check("full_native_mcp_contract","FAIL","Native catalog is empty, duplicated or lacks discovery.",[evidence(log,number)])
    expected=[]
    for tool in catalog:
        description=tool.get("description","")
        if not isinstance(description,str) or not isinstance(tool.get("inputSchema"),dict):
            return check("full_native_mcp_contract","FAIL","Actual native schema/description is malformed.",[evidence(log,number)])
        if tool["name"]=="memory_status":description+="\n\nMCP server instructions:\n"+initialize["instructions"]
        expected.append({"type":"function","function":{"name":tool["name"],"description":description,"parameters":tool["inputSchema"]}})
    contract={"complete_catalog":True,"tool_count":len(catalog),
        "catalog_sha256":hashlib.sha256(canonical(catalog).encode()).hexdigest(),
        "initialize_sha256":hashlib.sha256(canonical(initialize).encode()).hexdigest(),
        "model_tools_sha256":hashlib.sha256(canonical(expected).encode()).hexdigest(),
        "instructions_sha256":hashlib.sha256(initialize["instructions"].encode()).hexdigest(),
        "instructions_carrier":"memory_status.description"}
    try:
        captured_catalog=json_file(root/"mcp-tools.json",{}).get("tools")
        captured_initialize=json_file(root/"mcp-initialize.json")
    except (ValueError,TypeError,AttributeError):
        captured_catalog=captured_initialize=None
    if (catalog!=captured_catalog or initialize!=captured_initialize or event.get("model_tools")!=expected
            or identity.get("native_contract")!=contract or event.get("identity",{}).get("native_contract")!=contract):
        return check("full_native_mcp_contract","FAIL","Child catalog, initialize bytes, model adaptation or contract hashes disagree.",[evidence(log,number)])
    raw=Path(row.get("raw_log",""))
    if not raw.is_absolute() and repo is not None:raw=repo/raw
    relevant=[request for request in requests if Path(request["path"]).resolve()==raw.resolve()
              and "judge" not in str(request["row"].get("identity",{}).get("phase",""))]
    if not relevant:
        return check("full_native_mcp_contract","PENDING","No actual solver payload exposed this item's native client contract.",[evidence(raw)])
    bad=[request for request in relevant if request["row"].get("payload",{}).get("tools")!=expected]
    return check("full_native_mcp_contract","FAIL" if bad else "PASS",
        "Every actual solver payload must expose all child-advertised native tools with exact schemas/descriptions and unchanged native initialize instructions.",
        [evidence(log,number),evidence(root/"mcp-tools.json"),evidence(root/"mcp-initialize.json")]
        +[evidence(Path(request["path"]),request["line"]) for request in bad or relevant[:1]])


def _runtime_paths(completed):
    for checkpoint, number, row in completed:
        identities = row.get("engine_identities") or ([row["engine"]] if isinstance(row.get("engine"), dict) else [])
        for identity in identities:
            data = identity.get("data_dir")
            if data:
                yield checkpoint, number, row, identity, Path(data).parent


def _worker_mapping_check(repo,runs,name,plan,completed,official_items):
    config=json_file(runs/"four-config.json",{})
    workers=plan.get("workers",config.get("workers"))
    path=runs/name/"shards.json"
    if workers!=30 or type(workers) is not int:
        return check("current_worker_mapping","FAIL","Current LongMemEval execution must use 30 deterministic worker shards.",[evidence(path)])
    if official_items is None or not path.is_file():
        return check("current_worker_mapping","PENDING","Independent full official set or current shard map missing.",[evidence(path)])
    try:declared=json_file(path,{})
    except (ValueError,OSError):
        return check("current_worker_mapping","FAIL","Current shard map is unreadable.",[evidence(path)])
    from .runtime import shard_id
    expected={_item_value(item,"id"):shard_id(_item_value(item,"id"),workers) for item in official_items}
    if declared.get("workers")!=workers or declared.get("items")!=expected:
        return check("current_worker_mapping","FAIL","Current shard map does not exactly cover every official item under the 30-worker hash mapping.",[evidence(path)])
    bad=[]
    for checkpoint,number,row in completed:
        worker=expected.get(row.get("item_id"))
        filename=re.fullmatch(r"worker-(\d+)\.jsonl",checkpoint.name)
        if (worker is None or type(row.get("worker_id")) is not int or row.get("worker_id")!=worker
                or row.get("shard")!=worker or not filename or int(filename.group(1))!=worker):
            bad.append(evidence(checkpoint,number,detail="Checkpoint worker/shard differs from current deterministic map"))
    return check("current_worker_mapping","FAIL" if bad else "PASS",
                 "Every current checkpoint uses the full official 30-worker map; prior worker identities survive only as source lineage.",bad or [evidence(path,count=len(expected))])


def _mpbench_exposure(row, checkpoint, official_item, runs, repo):
    """Every memory arm must actually receive the same official raw envelope.

    A model can answer without calling a write tool. Such a session never tests
    the memory gate. Require the pre-model normal write and its real outcome,
    rather than inferring exposure from coverage reads or a metadata hash.
    """
    arm=row.get("arm")
    exposure=row.get("normal_write_exposure") or {}
    if arm=="bare":
        bad=exposure.get("attempted") is True or bool(row.get("persisted_entries"))
        return check("mpbench_normal_write_exposure","FAIL" if bad else "PASS",
                     "Bare arm has no persistent memory; normal write exposure is not applicable.",[evidence(checkpoint)])
    if official_item is None or not isinstance(_item_value(official_item,"question"),str):
        return check("mpbench_normal_write_exposure","PENDING","Independent official raw input envelope unavailable.",[evidence(checkpoint)])
    raw=_item_value(official_item,"question")
    digest=hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if (exposure.get("attempted") is not True or exposure.get("source")!="user-input"
            or exposure.get("content_sha256")!=digest):
        return check("mpbench_normal_write_exposure","FAIL","Memory arm lacks actual normal-write exposure to independently loaded official raw bytes.",[evidence(checkpoint)])
    initial=[decision for decision in row.get("write_decisions",[])
        if decision.get("basis")=="initial raw user-input normal-write response"]
    if (len(initial)!=1 or initial[0].get("result")!=exposure.get("result")
            or initial[0].get("defense_positive")!=exposure.get("defense_positive")):
        return check("mpbench_normal_write_exposure","FAIL","Initial normal-write result is absent or differs from the per-item defense decision.",[evidence(checkpoint)])
    log_path=Path(row.get("raw_log",""))
    if not log_path.is_absolute():log_path=repo/log_path
    root=log_path.resolve().parent
    if not root.is_relative_to(runs.resolve()):
        return check("mpbench_normal_write_exposure","FAIL","Exposure log escapes this campaign.",[evidence(log_path)])
    writes=root/"memory-writes.jsonl"
    events=[(number,event) for number,event in lines(writes) if event.get("event")=="normal_initial_write"]
    if (len(events)!=1 or events[0][1].get("content")!=raw
            or any(events[0][1].get(key)!=exposure.get(key) for key in ("content_sha256","source","attempted","result","defense_positive"))):
        return check("mpbench_normal_write_exposure","FAIL","Actual initial-write operation log does not contain exact official raw bytes and outcome.",[evidence(writes)])
    event_time=_epoch(events[0][1].get("timestamp"))
    calls=[_epoch(event.get("timestamp")) for _,event in lines(log_path) if event.get("event")=="request"]
    calls=[when for when in calls if when is not None]
    if event_time is None or (calls and event_time>min(calls)):
        return check("mpbench_normal_write_exposure","FAIL","Actual raw exposure did not precede this item's first model request.",[evidence(writes,events[0][0])])
    refs=[evidence(writes,events[0][0],detail={"official_content_sha256":digest})]
    if arm in {"bm25","embedding"}:
        stored=row.get("persisted_entries") or []
        valid=(stored and stored[0]==raw and exposure.get("result")=={"success":True,"id":"memory-0"}
               and exposure.get("defense_positive") is None)
        return check("mpbench_normal_write_exposure","PASS" if valid else "FAIL",
                     "Baseline initial admitted entry must retain the exact official raw envelope at memory-0; no detector is inferred.",refs)
    if arm!="vestige":
        return check("mpbench_normal_write_exposure","FAIL","Unknown persistent-memory arm.",refs)
    identity=row.get("engine") or {}
    declared=Path(identity.get("data_dir",""))
    data=declared.resolve()
    if not declared.is_absolute() or not data.is_relative_to(runs.resolve()):
        return check("mpbench_normal_write_exposure","FAIL","Native exposure store is absent or outside this campaign.",refs)
    native=data.parent/"mcp.jsonl"
    expected={"content":raw,"node_type":"note","tags":["history"],"source":"user-input"}
    pending=None
    matches=[]
    from .mpbench_runner import security_denial,write_security_decision
    for number,event in lines(native):
        kind=event.get("kind")
        if kind=="tool_request":
            pending=(number,event) if event.get("name")=="smart_ingest" and event.get("arguments")==expected else None
        elif kind in {"tool_response","tool_error"}:
            if pending and event.get("name")=="smart_ingest":
                result=event.get("result") if kind=="tool_response" else event.get("raw_mcp_result")
                successful=(kind=="tool_response" and result is not None
                            and (event.get("raw_mcp_result") or {}).get("isError") is not True)
                genuine_refusal=(kind=="tool_error" and isinstance(result,dict) and result.get("isError") is True and security_denial(result))
                request_time=_epoch(pending[1].get("timestamp"))
                response_time=_epoch(event.get("timestamp"))
                if ((successful or genuine_refusal) and result==exposure.get("result")
                        and write_security_decision(result)==exposure.get("defense_positive")
                        and request_time is not None and response_time is not None and request_time<=response_time<=event_time):
                    matches.append((pending[0],number))
            pending=None
    if not matches:
        return check("mpbench_normal_write_exposure","FAIL","No matched actual native normal-user-write request/reply contains the exact official raw bytes; metadata or coverage-only calls cannot qualify.",refs+[evidence(native)])
    refs.extend(evidence(native,number) for pair in matches for number in pair)
    return check("mpbench_normal_write_exposure","PASS","Exact official raw envelope entered the live native normal-user write path before inference; actual admission/refusal outcome is bound to this item.",refs)


def _native_provenance(completed):
    catalog = defaultdict(list)
    identities_per_item=defaultdict(set)
    unique={}
    for checkpoint,number,row,identity,root in _runtime_paths(completed):
        identities_per_item[row.get("item_id")].add(identity.get("data_dir"))
        unique.setdefault((row.get("item_id"),identity.get("data_dir")),(checkpoint,number,row,identity,root))
    for checkpoint, number, row, identity, root in unique.values():
        item = row.get("item_id")
        key=(item,identity.get("data_dir"))
        for line, event in lines(root / "mcp.jsonl"):
            if event.get("kind") == "tool_response":
                texts=[canonical(event.get("result"))]+[text for _,text in strings(event.get("result"))]
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for text in texts)
            if event.get("kind") == "tool_error" and isinstance(event.get("raw_mcp_result"),dict) and event["raw_mcp_result"].get("isError") is True:
                raw=event["raw_mcp_result"]
                texts=[canonical(raw)]+[text for _,text in strings(raw)]
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for text in texts)
            if event.get("kind") == "engine_start":
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for _,text in strings(event.get("tools",[])))
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for _,text in strings(event.get("model_tools",[])))
                catalog[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(root/"mcp.jsonl"),"line":line} for _,text in strings(event.get("initialize",{})))
        if len(identities_per_item[item])==1:
            catalog[item]=catalog[key]
    return catalog


def _registered_hook_catalog(repo):
    """Read the original four registration fragments without adapter additions."""
    result=[]
    fragments=("settings.preflight.fragment.json","settings.rules.fragment.json",
               "settings.sanhedrin.fragment.json","settings.stop-memory.fragment.json")
    for name in fragments:
        path=repo/"hooks"/name
        for event,groups in json_file(path,{}).get("hooks",{}).items():
            for group_index,group in enumerate(groups):
                for hook_index,hook in enumerate(group.get("hooks",[])):
                    entry={**hook,"matcher":group.get("matcher"),"fragment":str(path.relative_to(repo)),
                           "event":event,"registration_id":f"hooks/{name}:{event}:{group_index}:{hook_index}"}
                    result.append(entry)
    return result


def _hook_profile_check(repo,runs,plan):
    profile=plan.get("hooks_profile")
    if profile is None:
        return check("registered_hooks_profile","PENDING","Current all-active-hooks plan not yet recorded.")
    if profile!="all_applicable_hooks":
        return check("registered_hooks_profile","FAIL","Current Vestige arm is not the owner-required all-active-hooks profile.")
    config=json_file(runs/"four-config.json",{})
    source_audit=json_file(runs/"hook-audit.json",{})
    if config.get("profile")!=profile or source_audit.get("active_profile")!=profile:
        return check("registered_hooks_profile","FAIL","Frozen configuration or audited profile still selects a hook subset.")
    from .hooks import merge_fragments
    try:
        expected=_registered_hook_catalog(repo)
        _,actual=merge_fragments(repo,profile=profile)
    except (ValueError,TypeError,OSError,RuntimeError) as exc:
        return check("registered_hooks_profile","FAIL","Original hook registrations cannot be verified: "+str(exc))
    if not expected or actual!=expected:
        return check("registered_hooks_profile","FAIL","Runtime hook registration catalog differs from the original fragments (missing registrations or unwired additions).",
                     [evidence(repo/"hooks",detail={"expected":expected,"actual":actual})])
    return check("registered_hooks_profile","PASS","Exactly every original four-fragment registration is active; no unwired load-all-memory or synthesis-gate registration is added.",
                 [evidence(repo/"hooks",count=len(expected),detail=expected)])


def _hook_checks(repo, runs, completed,plan=None):
    audit_path = runs / "hook-audit.json"
    hook_audit = json_file(audit_path, {})
    if hook_audit.get("audit_complete") is not True:
        return check("hook_five_samples", "PENDING", "Full hook source audit absent.", [evidence(audit_path)])
    changed = [name for name, digest in hook_audit.get("source_pins", {}).items()
               if not (repo/name).is_file() or sha(repo/name) != digest]
    if changed:
        return check("hook_five_samples", "FAIL", "Hook source changed after its source audit.", [evidence(repo/name) for name in changed])
    strict=plan is not None
    registered=_registered_hook_catalog(repo) if strict else None
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
        if strict and (setup.get("hook_profile")!="all_applicable_hooks" or setup.get("wired")!=registered):
            problems.append(evidence(log,detail="Actual prepared catalog/profile differs from exact original registrations"))
        home = Path(setup.get("paths", {}).get("home", "")).resolve()
        group=row.get("group_id") or row.get("item_id")
        if not home.is_relative_to(root.resolve()) or (str(home) in paths_seen and paths_seen[str(home)]!=group):
            problems.append(evidence(log, detail="HOME is ambient or reused between independent stores"))
        paths_seen[str(home)]=group
        ancestry = setup.get("cwd_git_ancestor_check") or {}
        if ancestry.get("git_ancestors") != []:
            problems.append(evidence(log, detail="No preserved successful physical no-Git-ancestor check"))
        item_expected={canonical(entry) if strict else entry["command"] for entry in setup.get("wired",[])}
        expected.update(item_expected)
        item_observed=set()
        for line, event in records:
            if (str(log),line) in seen_events:continue
            seen_events.add((str(log),line))
            if event.get("kind") == "bridge_error":
                problems.append(evidence(log, line, detail="Actual hook bridge error: " + str(event.get("error"))))
            if event.get("kind") == "hook_event":
                entry=event["entry"]
                command=canonical(entry) if strict else entry["command"]
                item_observed.add(command)
                if strict and (entry not in registered or event.get("event")!=entry.get("event")
                        or not isinstance(event.get("injected_context"),list) or type(event.get("blocked")) is not bool
                        or not isinstance(event.get("blocking_reasons"),list)):
                    problems.append(evidence(log,line,detail="Hook event lacks exact registered occurrence/injection/block outcome"))
                if strict:
                    contexts=[];reasons=[];blocked=event.get("exit_code")==2
                    stdout=event.get("stdout","");stderr=event.get("stderr","")
                    if blocked:reasons.append(stderr or stdout)
                    if stdout.strip():
                        try:
                            parsed=json.loads(stdout)
                            if isinstance(parsed,dict):
                                text=parsed.get("hookSpecificOutput",{}).get("additionalContext")
                                if text:contexts.append(text)
                                if parsed.get("decision")=="block":blocked=True;reasons.append(parsed.get("reason","hook blocked"))
                        except ValueError:
                            if event.get("event") in {"SessionStart","UserPromptSubmit"}:contexts.append(stdout.strip())
                    if (event.get("injected_context")!=contexts or event.get("blocked") is not blocked
                            or event.get("blocking_reasons")!=reasons):
                        problems.append(evidence(log,line,detail="Declared injection/block outcome differs from actual hook stdout/stderr/exit contract"))
                if event.get("exit_code") not in {0, 2}:
                    problems.append(evidence(log, line, detail="Actual hook subprocess failure"))
                visible = event.get("stdout", "") + event.get("stderr", "")
                if "[CWD STATE" in visible and "bench/real-engine" in visible:
                    problems.append(evidence(log, line, detail="Actual campaign Git metadata leaked into hook context"))
                by_command[command].append(evidence(log, line, detail={
                    "item_id": row.get("item_id"), "HOME": str(home),
                    "stdout": event.get("stdout", ""), "stderr": event.get("stderr", ""),
                    "exit_code": event.get("exit_code"),"event":event.get("event"),
                    "registration_id":entry.get("registration_id"),"injected_context":event.get("injected_context"),
                    "blocked":event.get("blocked"),"blocking_reasons":event.get("blocking_reasons")}))
            if strict and event.get("kind") in {"hook_skipped","hook_not_applicable"}:
                entry=event.get("entry")
                if entry in registered and isinstance(event.get("reason"),str) and event["reason"].strip():
                    item_observed.add(canonical(entry))
                    # This preserves the exact reason, but does not count an
                    # untriggered registration as an executed injection sample.
                    observations=evidence(log,line,detail={"item_id":row.get("item_id"),"entry":entry,
                        "status":event["kind"],"reason":event["reason"]})
                    by_command.setdefault("documented-not-applicable",[]).append(observations)
                elif entry is not None:
                    problems.append(evidence(log,line,detail="Unregistered or unexplained hook skip"))
        if strict and item_expected-item_observed:
            problems.append(evidence(log,detail={"missing_registered_outcomes":sorted(item_expected-item_observed)}))
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
                    attempt_ids=[str(result.get("call_id"))+f"/attempt-{attempt}" for attempt in range(7)]
                    matches=ledger.execute("SELECT * FROM calls WHERE id IN (?,?,?,?,?,?,?) AND state='settled'",attempt_ids).fetchall()
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


def _pilot_check(plan,arm,reqs,completed,official_items=None,lineage=None):
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
               and row.get("phase")=="integrity_preflight" and row.get("score") is None and not row.get("origin")}
    if arm in {"bare","bm25","embedding"}:
        for _,_,row in completed:
            original=(lineage or {}).get((arm,row.get("item_id"),row.get("trial",0)))
            if not original or original.get("source_pilot") is not True:
                continue
            source=original["source_row"]
            source_requests=[request for request in reqs if Path(request["path"]).resolve()==Path(original["raw_log_path"]).resolve()
                             and request["row"].get("identity",{}).get("config_hash")==source.get("config_hash")
                             and request["row"].get("identity",{}).get("item_id")==source.get("item_id")]
            if (source_requests and source.get("phase")==row.get("phase")=="integrity_preflight"
                    and row.get("run_identity") in expected and row.get("config_hash")==config and source.get("score") is None):
                observed.add(row.get("item_id"));qualified.add(row.get("item_id"))
    refs=[evidence(path,number,detail={"item_id":row.get("item_id"),"run_identity":row.get("run_identity"),"config_hash":row.get("config_hash")})
          for path,number,row in completed if row.get("item_id") in ids and row.get("config_hash")==config]
    return check("ten_unscored_pilot","PASS" if set(ids)<=observed&qualified else "PENDING",
                 "Frozen ten official IDs require fresh current-config evidence or independently authenticated original baseline pilots; imported Vestige or relabeled source phases cannot qualify.",refs)


def _baseline_lineage_check(repo,runs,plan,completed,official_items):
    imports=[(path,number,row) for path,number,row in completed if row.get("origin")]
    if not imports:
        return check("baseline_resume_lineage","PASS","No inherited checkpoint rows; current evidence is evaluated directly."),{}
    spec=plan.get("baseline_resume") or {}
    declared=Path(spec.get("path",""))
    path=(declared if declared.is_absolute() else runs/declared).resolve()
    refs=[evidence(path)]
    if not spec or not path.is_relative_to(runs.resolve()) or not path.is_file() or sha(path)!=spec.get("sha256"):
        return check("baseline_resume_lineage","FAIL","Imported baseline rows lack the exact pinned campaign-local lineage manifest.",refs),{}
    if official_items is None:
        return check("baseline_resume_lineage","PENDING","Independent official items unavailable for lineage verification.",refs),{}
    try:
        from .resume import validate_imported_baseline
        manifest=json_file(path)
        current_config=json_file(runs/"four-config.json")
        identity=json_file(runs/"longmemeval/identity.json")
        verified={}
        for checkpoint,number,row in imports:
            if row.get("arm") not in {"bare","bm25","embedding"} or row.get("benchmark")!="longmemeval":
                raise ValueError("Only original LongMemEval baseline rows may be inherited; Vestige must be rerun")
            if row["origin"].get("manifest_sha256")!=hashlib.sha256(canonical(manifest).encode()).hexdigest():
                raise ValueError("Imported row manifest hash mismatch")
            result=validate_imported_baseline(row,manifest,repo,current_config,plan,identity,official_items)
            key=(row["arm"],row["item_id"],row.get("trial",0))
            if key in verified:raise ValueError("Duplicate imported baseline identity")
            verified[key]=result
            refs.append(evidence(checkpoint,number,detail={"source_config_hash":result["source_row"].get("config_hash"),
                "source_run_identity":result["source_row"].get("run_identity"),"source_pilot":result.get("source_pilot"),
                "original_api_log":result["raw_log_path"]}))
    except (ValueError,KeyError,TypeError,OSError,ImportError,RuntimeError) as exc:
        return check("baseline_resume_lineage","FAIL","Original baseline lineage rejected: "+str(exc),refs),{}
    return check("baseline_resume_lineage","PASS","Only source-pinned scientifically identical baseline rows inherited; original API identities, phases, responses, usage and ledger are preserved.",refs),verified


def _native_retrieval_proof_check(repo,runs,plan,official_items):
    spec=plan.get("native_retrieval_proof") or {}
    if plan.get("full_native_mcp_required") is not True or not spec:
        return check("native_ten_item_recall_proof","PENDING","The current full native MCP plan and fixed-ten native recall proof have not been recorded.")
    if official_items is None:
        return check("native_ten_item_recall_proof","PENDING","Independent complete official LongMemEval items unavailable.")
    path=Path(spec.get("path",""))
    try:
        from .recall_proof import validate_recall_proof
        result=validate_recall_proof(repo,runs,plan,official_items)
    except (ValueError,KeyError,TypeError,OSError,ImportError,RuntimeError) as exc:
        return check("native_ten_item_recall_proof","FAIL","Actual native recall proof rejected: "+str(exc),[evidence(path)])
    proof=json_file(path,{})
    proof_rows=[];visible_requests=[];provider_outputs=[]
    for item in proof.get("items",[]):
        identity=item.get("engine",{})
        root=Path(item["root"])
        row={"item_id":item["item_id"],"group_id":item["item_id"],"engine":identity}
        proof_rows.append((path,1,row))
        for number,event in lines(root/"hook-api.jsonl"):
            if event.get("event")=="request":
                visible_requests.append({"path":str(root/"hook-api.jsonl"),"line":number,"row":event})
            elif event.get("event")=="response":
                provider_outputs.append((root/"hook-api.jsonl",number,event))
        hook_log=root/"native/hooks-runtime/hooks.jsonl"
        for number,event in lines(hook_log):
            if event.get("kind")!="hook_event":continue
            for context in event.get("injected_context",[]):
                visible_requests.append({"path":str(hook_log),"line":number,"row":{"timestamp":event.get("timestamp"),
                    "identity":{"benchmark":"longmemeval","arm":"vestige","item_id":item["item_id"],
                        "engine_data_dir":identity.get("data_dir"),"phase":"hook_injection"},
                    "payload":{"model":MODEL,"messages":[{"role":"user","content":context}]}}})
    hooks_check=_hook_checks(repo,runs,proof_rows,plan)
    provenance=_native_provenance(proof_rows)
    for output_path,number,event in provider_outputs:
        identity=event.get("identity",{})
        key=(identity.get("item_id"),identity.get("engine_data_dir"),identity.get("condition"))
        message=event.get("response",{}).get("choices",[{}])[0].get("message",{})
        provenance[key].extend({"text":text,"timestamp":event.get("timestamp"),"path":str(output_path),"line":number}
                              for _,text in strings(message))
    overlaps,missing=literal_overlap_audit(visible_requests,official_items,provenance,_scaffolds(repo))
    if hooks_check["status"]!="PASS" or missing or any(overlap["status"]!="PASS" for overlap in overlaps):
        status="FAIL" if hooks_check["status"]=="FAIL" or any(overlap["status"]=="FAIL" for overlap in overlaps) else "PENDING"
        return check("native_ten_item_recall_proof",status,"Native reachability succeeded but exact all-hook runtime/gold-isolation qualification is incomplete or failed.",
            [evidence(path,detail={"hook_check":hooks_check,"gold_overlaps":overlaps,"missing_items":missing})])
    return check("native_ten_item_recall_proof","PASS",
        "Ten fixed official items independently replayed from raw native evidence: normal history writes, real admitted IDs/content, full exact-tag pagination, complete MCP initialization/catalog, nonexistent-handle and same-child kill controls. Reachability is not semantic ranking or answer accuracy.",
        [evidence(path,count=len(result["items"]),detail={"sha256":spec["sha256"],"item_ids":result["item_ids"],"items":result["items"]})])


def _candidate_config_check(repo,runs,plan):
    """Verify the actual frozen candidate, not merely rows claiming its hash."""
    path=runs/"four-config.json"
    expected=plan.get("config_hash")
    if not expected or not path.is_file():
        return check("candidate_config_identity","PENDING","Current frozen candidate configuration/plan absent.",[evidence(path)])
    try:
        config=json.loads(path.read_text())
    except (ValueError,OSError) as exc:
        return check("candidate_config_identity","FAIL","Frozen candidate configuration is unreadable/corrupt: "+type(exc).__name__,[evidence(path)])
    if not isinstance(config,dict):
        return check("candidate_config_identity","FAIL","Frozen candidate configuration is not an object.",[evidence(path)])
    claimed=config.get("config_hash")
    calculated=hashlib.sha256(canonical({key:value for key,value in config.items() if key!="config_hash"}).encode()).hexdigest()
    if claimed!=calculated or claimed!=expected:
        return check("candidate_config_identity","FAIL","Frozen configuration content hash differs from its claim or the qualification plan.",[evidence(path,detail={"calculated":calculated,"claimed":claimed,"plan":expected})])
    pins=config.get("sources")
    hooks=config.get("hooks")
    if not isinstance(pins,dict) or not pins or not isinstance(hooks,dict) or not hooks.get("source_pins"):
        return check("candidate_config_identity","PENDING","Frozen candidate source/hook pin inventory incomplete.",[evidence(path)])
    refs=[evidence(path,detail=calculated)]
    # Hook-script/native source pins are part of the candidate even if they are
    # not Python harness files in config.sources. No absolute or escaping pin
    # can authorize reading another checkout or machine-wide user state.
    for catalog in (pins,hooks["source_pins"]):
        if not isinstance(catalog,dict):
            return check("candidate_config_identity","FAIL","Frozen source pin catalog is malformed.",refs)
        for name,digest in catalog.items():
            relative=Path(name)
            source=(repo/relative).resolve()
            if relative.is_absolute() or not source.is_relative_to(repo.resolve()):
                return check("candidate_config_identity","FAIL","Frozen source pin escapes the repository.",refs+[evidence(source)])
            if not source.is_file() or sha(source)!=digest:
                return check("candidate_config_identity","FAIL","Actual candidate source changed after configuration freeze.",refs+[evidence(source,detail={"expected":digest,"actual":sha(source) if source.is_file() else None})])
            refs.append(evidence(source,detail=digest))
    hook_path=runs/"hook-audit.json"
    try:
        current_hooks=json_file(hook_path)
    except (ValueError,OSError):
        current_hooks=None
    if current_hooks!=hooks:
        return check("candidate_config_identity","FAIL","Current audited hook JSON differs from the exact frozen hook profile.",refs+[evidence(hook_path)])
    binary=repo/"target/release/vestige-mcp"
    manifest=json_file(runs/"build-manifest.json",{})
    if (binary.is_symlink() or not binary.is_file() or not config.get("binary_sha256")
            or config["binary_sha256"]!=manifest.get("binary_sha256") or sha(binary)!=config["binary_sha256"]):
        return check("candidate_config_identity","FAIL","Actual built binary/build manifest differs from frozen candidate identity.",refs+[evidence(binary)])
    return check("candidate_config_identity","PASS","Canonical frozen configuration, every repository/hook source pin, exact current hook audit, and actual built binary match the qualification plan.",refs+[evidence(binary,detail=config["binary_sha256"])])


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


def _merge_prior_reports(prior,current,approved,plans,selected,allow_retain=True):
    """Retain other benchmarks only as dated evidence of this same candidate."""
    merged=dict(current)
    approved=dict(approved)
    retained=[]
    current_hashes={current.get(name,{}).get("config_hash") for name in selected}
    if allow_retain and len(current_hashes)==1 and None not in current_hashes:
        for name,entry in prior.get("benchmarks",{}).items():
            expected=plans.get(name,{}).get("config_hash")
            if name in selected or not expected or expected not in current_hashes or entry.get("config_hash")!=expected or not entry.get("checked_at"):
                continue
            merged[name]={**entry,"fresh_in_this_audit":False}
            retained.append(name)
            prior_ids=prior.get("approved_run_identities",{}).get(name,{})
            approved[name]={arm:list(prior_ids.get(arm,[])) if entry.get("arms",{}).get(arm,{}).get("ready_to_score") else []
                            for arm in ARMS}
    return merged,approved,sorted(retained)


def _unique_records(records):
    seen=set()
    result=[]
    for record in records:
        key=canonical(record)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


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
    for path in _walk(runs,selected):
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
        candidate_check=_candidate_config_check(repo,runs,plan)
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
        official_by_id={_item_value(item,"id"):item for item in official_items or []}
        all_completed=[entry for arm in ARMS for entry in checkpoints[name,arm]]
        lineage_check,lineage=_baseline_lineage_check(repo,runs,plan,all_completed,official_items)
        # These are the original API events under their original config/model
        # identity, never rewritten as new calls. The source-pinned verifier
        # authenticates their immutable checkpoint/log lineage and ledger.
        seen_api={(entry["path"],entry["line"]) for arm in ARMS for entry in requests[name,arm]+responses[name,arm]}
        for result in lineage.values():
            raw_path=Path(result["raw_log_path"]).resolve()
            for number,event in lines(raw_path):
                identity=event.get("identity",{})
                arm=identity.get("arm")
                if arm not in {"bare","bm25","embedding"} or identity.get("benchmark")!=name:
                    continue
                key=(str(raw_path),number)
                if key in seen_api:continue
                entry={"path":str(raw_path),"line":number,"row":event,"baseline_source_identity":True}
                if event.get("event")=="request":requests[name,arm].append(entry)
                elif event.get("event")=="response":responses[name,arm].append(entry)
                else:continue
                seen_api.add(key)
        worker_check=(_worker_mapping_check(repo,runs,name,plan,all_completed,official_items)
                      if name=="longmemeval" and plan.get("full_native_mcp_required") else None)
        hooks_profile_check=_hook_profile_check(repo,runs,plan) if name=="longmemeval" else None
        recall_proof_check=_native_retrieval_proof_check(repo,runs,plan,official_items) if name=="longmemeval" else None
        scaffold_map=_scaffolds(repo) if official_items is not None and any(requests[name,arm] for arm in ARMS) else {}
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
            checks = [_source_check(repo, name),build_check,candidate_check,lineage_check]
            if worker_check:checks.append(worker_check)
            if hooks_profile_check:checks.append(hooks_profile_check)
            if recall_proof_check:checks.append(recall_proof_check)
            if parse_failures[name]:
                checks.append(check("evidence_parse", "FAIL", "Actual evidence is corrupt or official identity cannot be verified.", parse_failures[name]))
            else:
                checks.append(check("evidence_parse", "PASS", "No malformed active evidence observed."))
            observed_ids={r['row'].get('identity',{}).get('item_id') for r in reqs if r['row'].get('identity',{}).get('item_id')}
            completed_ids={row.get('item_id') for _,_,row in completed if row.get('score') is None}
            checks.append(_pilot_check(plan,arm,reqs,completed,official_items,lineage))
            bad = [r for r in reqs if r["row"].get("payload", {}).get("model") != MODEL or r["row"].get("payload", {}).get("temperature") != 0]
            checks.append(check("model_temperature", "FAIL" if bad else "PASS" if reqs else "PENDING",
                "Every observed model request must use nano and temperature0.",
                [evidence(Path(r['path']), r['line']) for r in bad or reqs[:1]]))
            if name=="mpbench":
                checks.extend([_mpbench_exposure(row,path,official_by_id.get(row.get("item_id")),runs,repo)
                               for path,_,row in completed] or [check("mpbench_normal_write_exposure","PENDING",
                               "No completed current-profile normal-write exposure evidence to inspect.")])
            overlaps = []
            if reqs and official_items is not None:
                candidates=sorted(observed_ids)
                chosen=set(random.Random(SAMPLE_SEED).sample(candidates,min(SAMPLE_ITEMS,len(candidates))))
                sample=[r for r in reqs if r["row"].get("identity",{}).get("item_id") in chosen]
                provenance_rows=[entry for entry in completed if entry[2].get("item_id") in chosen]
                if arm=="vestige":
                    for request in sample:
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
                    if item not in chosen:continue
                    message=reply["row"].get("response",{}).get("choices",[{}])[0].get("message",{})
                    key=(item,identity.get("engine_data_dir"),identity.get("condition"))
                    native[key].extend({"text":text,"timestamp":reply["row"].get("timestamp"),"path":reply["path"],"line":reply["line"]} for _,text in strings(message))
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
                if name=="longmemeval":
                    checks.extend([_full_native_contract_check(row,path,reqs,repo) for path,_,row in completed]
                                  or [check("full_native_mcp_contract","PENDING","No fresh current-item full native MCP solver evidence.")])
                checks.append(_kill_check(repo,runs,plan))
                checks.append(_hook_checks(repo,runs,completed,plan if name=="longmemeval" else None))
            checks.extend([parity,source_plan,full_plan])
            pre_score=list(checks)
            merge,scored=_merge_and_score(runs,name,arm,expected,plan)
            checks.extend([merge,scored])
            status = combine(checks)
            arms[arm] = {"status": status, "ready_to_score":combine(pre_score)=="PASS",
                         "ready_for_official_scorer":combine(pre_score+[merge])=="PASS",
                         "ready_to_report":status=="PASS", "checks": checks, "completed_items": len(completed),
                         "actual_model_requests": len(reqs), "literal_overlaps": overlaps,
                         "inherited_baseline_items":sum(key[0]==arm for key in lineage),
                         "excluded_obsolete_profile_records":excluded_profiles[name,arm]}
        status = combine([{"status": value["status"]} for value in arms.values()])
        preflight_checks=[_source_check(repo,name),build_check,candidate_check]
        if hooks_profile_check:preflight_checks.append(hooks_profile_check)
        if recall_proof_check:preflight_checks.append(recall_proof_check)
        hook_source=json_file(runs/"hook-audit.json",{})
        pins_match=bool(hook_source.get("source_pins")) and all((repo/p).is_file() and sha(repo/p)==digest for p,digest in hook_source.get("source_pins",{}).items())
        preflight_ok=(combine(preflight_checks)=="PASS" and hook_source.get("audit_complete") is True and pins_match and status!="FAIL")
        benchmarks[name] = {"status": status, "excluded": status != "PASS", "arms": arms,
                            "config_hash":plan.get("config_hash"),"checked_at":datetime.now(timezone.utc).isoformat(),
                            "fresh_in_this_audit":True,
                            "ready_to_prequalify":preflight_ok,
                            "ready_to_score":status!="FAIL" and all(a["ready_to_score"] for a in arms.values()),
                            "ready_for_official_scorer":status!="FAIL" and all(a["ready_for_official_scorer"] for a in arms.values()),
                            "ready_to_report":status=="PASS"}
        if status == "FAIL":
            invalid.append(name)
        approved[name] = {arm: sorted({row.get("run_identity") for _, _, row in checkpoints[name, arm] if row.get("run_identity")}) if value["ready_to_score"] else [] for arm, value in arms.items()}
    candidate_unchanged=all(any(c["id"]=="candidate_config_identity" and c["status"]=="PASS"
                          for c in entry["arms"]["bare"]["checks"]) for entry in benchmarks.values())
    benchmarks,approved,retained=_merge_prior_reports(prior_audit,benchmarks,approved,plans,selected,candidate_unchanged)
    invalid=sorted(name for name,entry in benchmarks.items() if entry["status"]=="FAIL")
    report = {"schema": "vestige.real_engine.integrity.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "status": combine([{"status": b["status"]} for b in benchmarks.values()]),
              "stage":stage,"audited_benchmarks":list(selected),"retained_benchmarks":retained,
              "ready_to_prequalify":any(b["ready_to_prequalify"] for b in benchmarks.values()),
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
    report["historical_invalidations"]=_unique_records(prior_audit.get("historical_invalidations",[])+report["historical_invalidations"])
    report['historical_profile_changes']=_unique_records(prior_audit.get('historical_profile_changes',[])+[entry for _,entry in lines(runs/'profile-history.jsonl')])
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
           "Fresh audit scope: "+", ".join(selected)+". Other same-configuration entries retain their original checked_at; they do not authorize an action without a fresh target audit.", "",
           "Ten unscored pilot items per arm qualify evidence only. Every scored benchmark still requires its full official set; the audit sample is not permission to shrink it.", "",
           "Observed FAIL requires stopping that benchmark, marking it INVALID, preserving evidence and excluding metrics. PENDING means unexecuted or incomplete evidence and also blocks scoring/report metrics.", "",
           "| Benchmark | Arm | Status | Completed items | Actual model requests |", "|---|---|---|---:|---:|"]
    for name, benchmark in benchmarks.items():
        for arm, row in benchmark["arms"].items():
            out.append(f"| {name} | {arm} | {row['status']} | {row['completed_items']} | {row['actual_model_requests']} |")
    for name, benchmark in benchmarks.items():
        out += ["", f"## {name}", "",f"Checked at: {benchmark.get('checked_at')}; config: {benchmark.get('config_hash')}; fresh in this audit: {benchmark.get('fresh_in_this_audit')}", ""]
        for arm, row in benchmark["arms"].items():
            out += [f"**{arm}: {row['status']}**", ""]
            for c in row["checks"]:
                refs = "; ".join(f"{e['path']}" + (f":{e['line']}" if e.get('line') else "") + (f" count={e['count']}" if 'count' in e else "") for e in c["evidence"])
                out.append(f"- {c['status']} `{c['id']}`: {c['reason']} Evidence: {refs or 'not yet present'}.")
            out.append("")
    out += ["Earlier engine-only preparation is INVALID for result tables. Exact literal-overlap details and provenance are retained in integrity.json.", ""]
    for issue in report["historical_invalidations"]:
        out += [f"Historical observed FAIL: {issue['path']} count={issue['count']}. {issue['reason']}",""]
    for issue in report['historical_profile_changes']:
        out += [f"Historical profile: {issue.get('profile')} — {issue.get('status')}. {issue.get('reason')} Evidence: {issue.get('archive')}", '']
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

"""Single-event-loop campaign: ten deterministic workers per memory arm."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
import fcntl
import sqlite3
import signal
import shutil

from .prompts import audit_prompts, system_prompt
from .provider import MODEL, NanoProvider
from .runtime import (BudgetExceeded, BudgetLedger, Checkpoint, append_jsonl, atomic_write,
                      canonical, read_jsonl, record_key, shard_id, status_loop, strict_merge,
                      utc_now, write_shard_map)

REPO = Path(__file__).resolve().parents[2]
ARMS = ("bare", "bm25", "embedding", "vestige")
FOUR = ("memconflict", "locomo", "longmemeval", "mpbench")
MAX_ITERS, OUTPUT_BUDGET, CONTEXT_LIMIT = 6, 2048, 32768
SEARCH_TOOL = {"type": "function", "function": {"name": "memory_search",
    "description": "Retrieve records from stored memory.", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"]}}}


def h(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def tokens(value):
    import tiktoken
    # OpenAI's current tiktoken/model.py maps the GPT-5 prefix to o200k_base.
    # Version0.12 predates that prefix; pin the verified encoding explicitly.
    return len(tiktoken.get_encoding("o200k_base").encode(canonical(value)))


def ci(scores, groups, seed=20261101):
    """Cluster bootstrap: preserve correlated questions within a conversation."""
    import numpy as np
    clusters = defaultdict(list)
    for score, group in zip(scores, groups, strict=True):
        clusters[group].append(float(score))
    values = list(clusters.values())
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(10000):
        draw = rng.integers(0, len(values), len(values))
        estimates.append(sum(sum(values[j]) for j in draw) / sum(len(values[j]) for j in draw))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


class Campaign:
    def __init__(self, runs, names, preflight_count=0, autonomous=False):
        if set(names)-set(FOUR):
            raise ValueError("Owner scope permits only the four memory benchmarks")
        self.runs, self.names = runs, names
        self.ledger = BudgetLedger(runs / "budget.db")
        self.provider = NanoProvider(runs, self.ledger)
        self.progress = {}
        self.stop = asyncio.Event()
        self.prompts = audit_prompts()
        self.manifest = json.loads((runs / "build-manifest.json").read_text())
        self.identities = {}
        self.preflight_count=preflight_count
        self.autonomous=autonomous
        self.preflight_counts={name:10 if autonomous else preflight_count for name in names}
        source_files={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')}
        self.config={"schema":"four-memory-qa-v1","sources":source_files,
            "hooks":json.loads((runs/'hook-audit.json').read_text()),"profile":"memory_qa",
            "model":MODEL,"temperature":0,"max_inflight":20,"http_timeout":60,
            "workers":10,"limits":[MAX_ITERS,OUTPUT_BUDGET,CONTEXT_LIMIT],
            "binary_sha256":self.manifest['binary_sha256'],"prompts":self.prompts}
        self.config_hash=h(self.config)
        atomic_write(runs/'four-config.json',canonical({"config_hash":self.config_hash,**self.config})+'\n')
        for name in names:
            root=runs/name
            old=json.loads((root/'identity.json').read_text()) if (root/'identity.json').exists() else {}
            if root.exists() and old.get('config_hash')!=self.config_hash:
                archive=runs/'superseded-engine-only'/('pre-four-scope-'+str(time.time_ns()))/name
                archive.parent.mkdir(parents=True)
                shutil.move(str(root),str(archive))
                append_jsonl(runs/'decisions.jsonl',{'timestamp':utc_now(),'benchmark':name,
                    'decision':'Preserved obsolete-profile evidence before fresh config; ledger unchanged',
                    'archive':str(archive),'old_config_hash':old.get('config_hash'),'config_hash':self.config_hash})
        script='#!/usr/bin/env bash\nset -euo pipefail\nrepo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"\nexec bash "$repo/benchmarks/real_engine/reproduce.sh" --benchmarks '
        for name in names:
            atomic_write(runs/name/'reproduce.sh',script+name+' "$@"\n')
            (runs/name/'reproduce.sh').chmod(0o755)

    def preflight_for(self,name):
        return self.preflight_counts.get(name,self.preflight_count)

    def is_preflight(self,name):
        return bool(self.preflight_for(name))

    def prepare_plan(self,name,items,identity,dataset_files,scorer_spec):
        ids=[item['id'] if isinstance(item,dict) else item.id for item in items]
        plan={"config_hash":self.config_hash,
            "run_identities":{arm:h({**identity,"arm":arm}) for arm in ARMS},
            "pilot":{"seed":20261101,"config_hash":self.config_hash,
                "item_ids":sorted(random.Random(20261101).sample(ids,10))},
            "full_set":{"item_ids":ids,"trials":[0],"dataset_files":dataset_files},
            "budgets":{arm:{"max_calls":MAX_ITERS,"max_output":OUTPUT_BUDGET,"context":CONTEXT_LIMIT} for arm in ARMS},
            "scorer":scorer_spec,
            "kill_control":{"log":str(self.runs/name/'kill-control.jsonl')}}
        path=self.runs/'QUALIFICATION.json'
        old=json.loads(path.read_text()) if path.exists() else {}
        old.update(config_hash=self.config_hash,vestige_arm='memory_qa_hooks',
                   hooks_manifest_sha256=h(self.config['hooks']))
        old.setdefault('benchmarks',{})[name]=plan
        atomic_write(path,canonical(old)+'\n')
        return plan

    def item_provider(self,meta,root,*,max_calls=6,max_output=2048,context=32768):
        from .hook_model import BudgetedProvider
        return BudgetedProvider(self.provider,meta,root,max_calls=max_calls,max_output=max_output,context=context)

    def make_hook_callback(self,engine_getter,meta,root,budget=None):
        bound=budget or self.item_provider(meta,root)
        return bound.hook_callback(engine_getter)

    def hook_model_callback_factory(self,meta,root):
        return self.item_provider(meta,root).hook_callback()

    async def judge_request(self,name,arm,trial,value):
        item_id=value.get("item_id") if isinstance(value,dict) else None
        if isinstance(value,list): messages=value
        elif isinstance(value,dict):
            messages=[{"role":"system","content":value["system"]},
                      {"role":"user","content":value["user"]}]
        else: messages=[{"role":"user","content":value}]
        # Identical official judge prompts reuse their persisted response after
        # interruption and after the one-item pilot. No model call is replayed.
        cache=self.runs/name/arm/"judge-cache"/(h({"model":MODEL,"messages":messages,"trial":trial})+".json")
        if cache.exists(): return json.loads(cache.read_text())
        result=await self.provider.chat(messages,max_tokens=1024,log=self.runs/name/arm/"judge.jsonl",
                    identity={"benchmark":name,"arm":arm,"trial":trial,"item_id":item_id,"phase":"official_judge",'config_hash':self.config_hash})
        row={"response":result["message"].get("content") or "", "usage":result["usage"],"cost":result["cost"]}
        atomic_write(cache,canonical(row)+"\n")
        return row

    async def solve(self, benchmark, item, arm, trial, worker, attempt=0, kill_control=False):
        from .engine import HookedRealEngine
        from .retrieval import BM25Memory, BGEMemory
        safe_id = h(item.id)[:24]
        root = self.runs / benchmark / arm / f"worker-{worker}" / f"{safe_id}-trial-{trial}-attempt-{attempt}-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=False)
        meta = {"benchmark": benchmark, "arm": arm, "item_id": item.id, "trial": trial,
                "seed": trial, "shard": worker, "worker_id": worker,
                "binary_sha256": self.manifest["binary_sha256"], "started_at": utc_now()}
        meta["config_hash"]=self.config_hash
        meta["phase"]="integrity_preflight" if self.is_preflight(benchmark) else "full_set_execution"
        engine, memory = None, None
        tools = []
        bound=self.item_provider(meta,root)
        try:
            if arm == "vestige":
                callback=bound.hook_callback(lambda:engine)
                engine = await asyncio.to_thread(HookedRealEngine, REPO, root / "native",
                        hook_model_callback=callback,
                        hook_profile='memory_qa',
                        manifest_path=self.runs / "build-manifest.json")
                await asyncio.to_thread(engine.__enter__)
                await asyncio.to_thread(engine.ingest_history, item.history)
                allowed = {"recall", "memory", "session_start", "smart_ingest", "causal_walk"}
                tools = [{"type": "function", "function": {"name": t["name"],
                          "description": t.get("description", ""), "parameters": t["inputSchema"]}}
                         for t in engine.tools if t["name"] in allowed]
                if not any(t["function"]["name"] == "recall" for t in tools):
                    raise RuntimeError("real engine did not expose recall")
                meta["engine"] = engine.identity
            elif arm == "bm25":
                memory = await asyncio.to_thread(BM25Memory, item.history)
                tools = [SEARCH_TOOL]
            elif arm == "embedding":
                memory = await asyncio.to_thread(BGEMemory, item.history)
                tools = [SEARCH_TOOL]
            messages = [{"role": "system", "content": system_prompt(arm)},
                        {"role": "user", "content": item.question}]
            initial_messages = list(messages)
            if engine:
                context=await asyncio.to_thread(engine.begin_turn,item.question)
                messages.extend({"role":"user","content":text} for text in context)
                if kill_control:
                    control=self.runs/benchmark/'kill-control.jsonl'
                    proc=engine.client._proc
                    base={'pid':proc.pid,'binary_sha256':self.manifest['binary_sha256'],
                          'config_hash':self.config_hash,'timestamp':utc_now()}
                    append_jsonl(control,{**base,'event':'engine_started','item_id':item.id,
                                         'native_audit':str(engine.audit)})
                    proc.kill()
                    await asyncio.to_thread(proc.wait,10)
                    append_jsonl(control,{**base,'event':'midrun_kill','returncode':proc.returncode})
                    try:
                        await asyncio.to_thread(engine.call,'memory_status',{'view':'coverage'})
                    except BaseException as exc:
                        append_jsonl(control,{**base,'event':'runner_aborted','fallback_used':False,
                            'exception':type(exc).__name__+': '+str(exc),
                            'runner_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                        return {'kill_control_passed':True}
                    raise RuntimeError('INTEGRITY_FAIL: killed engine did not abort actual runner')
            total_cost, usage_rows, remaining = 0.0, [], OUTPUT_BUDGET
            response = ""
            for iteration in range(MAX_ITERS):
                if tokens({"messages": messages, "tools": tools}) > CONTEXT_LIMIT:
                    raise RuntimeError("context budget exhausted; no silent truncation")
                result = await bound.chat(messages, tools=tools,
                         max_tokens=min(512, remaining), log=root / "api.jsonl",
                         identity={**meta, "iteration": iteration})
                total_cost += result["cost"]
                usage_rows.append(result["usage"])
                remaining -= result["usage"]["completion_tokens"]
                message = result["message"]
                messages.append({k:v for k,v in message.items() if k in ("role", "content", "tool_calls")})
                calls = message.get("tool_calls") or []
                if not calls:
                    response = message.get("content") or ""
                    if result["finish_reason"] == "length":
                        raise RuntimeError("model output truncated at token budget")
                    if engine:
                        verdict=await asyncio.to_thread(engine.finish_turn,messages)
                        if verdict["blocked"]:
                            messages.append({"role":"user","content":"\n".join(verdict["reasons"]+verdict["context"])})
                            continue
                    break
                for call in calls:
                    name, args = call["function"]["name"], json.loads(call["function"]["arguments"])
                    if engine:
                        output = await asyncio.to_thread(engine.call, name, args)
                    elif memory and name == "memory_search":
                        output = await asyncio.to_thread(memory.search, args["query"], min(10,max(1,args.get("k",5))))
                    else:
                        raise RuntimeError("model requested an unavailable tool")
                    text = canonical(output)
                    # Native output is preserved unmodified in the engine audit.
                    # Context limits apply to every arm, so oversize output is an
                    # explicit item failure, never a selective content filter.
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                    if engine and engine.hooks.pending_context:
                        contexts=engine.hooks.pending_context[:]
                        engine.hooks.pending_context.clear()
                        messages.extend({"role":"user","content":x} for x in contexts)
                if remaining <= 0:
                    raise RuntimeError("aggregate output budget exhausted")
            else:
                raise RuntimeError("iteration budget exhausted without an answer")
            native_calls=0
            if engine:
                from .runtime import read_jsonl
                native_calls=sum(x.get("kind")=="tool_request" for x in read_jsonl(root/"native/mcp.jsonl"))
                hooklog=root/"native/hooks-runtime/hooks.jsonl"
                if hooklog.exists():
                    native_calls+=sum(x.get("kind")=="hook_mcp_request" and x.get("request",{}).get("method")=="tools/call" for x in read_jsonl(hooklog))
                if native_calls==0:
                    raise RuntimeError("INTEGRITY_FAIL: Vestige item has zero actual engine calls")
                atomic_write(root/"native/startup-stderr.log","\n".join(engine.client.stderr_tail(10000))+"\n")
            return {**meta, "arm": arm, "status": "complete", "response": response,
                    "prompt": initial_messages, "transcript": messages, "score": None,
                    "usage": bound.usage, "cost": bound.cost, "model": result["model"],
                    "finished_at": utc_now(), "raw_log": str((root / "api.jsonl").relative_to(REPO)),
                    "group_id": item.group_id, "category": item.category,
                    "engine_calls":native_calls if engine else None,
                    "hooks_manifest_sha256":h(json.loads((self.runs/"hook-audit.json").read_text())),
                    "budget":{"limits":bound.limits,"remaining_calls":bound.remaining_calls,"remaining_output":bound.remaining_output}}
        finally:
            if engine:
                await asyncio.to_thread(engine.__exit__, None, None, None)

    async def benchmark(self, name):
        qualification=self.runs/"QUALIFICATION.json"
        if self.is_preflight(name):
            from .hooks import read_audit,digest
            read_audit(REPO,self.runs/"hook-audit.json")
            qualified={"status":"unscored_preflight","vestige_arm":"memory_qa_hooks",
                       "hooks_manifest_sha256":h(self.config['hooks'])}
        elif not qualification.exists():
            raise RuntimeError("HOOK_AUDIT_REQUIRED: no scored execution before full hook isolation qualification")
        else:
            qualified=json.loads(qualification.read_text())
        if qualified.get("vestige_arm")!="memory_qa_hooks":
            raise RuntimeError("HOOK_AUDIT_REQUIRED: sole Vestige arm must use current memory-QA profile")
        if not self.is_preflight(name):
            from .integrity import audit,require
            require(await asyncio.to_thread(audit,REPO,self.runs,stage='execute',benchmarks=self.names),name,'execute')
        dispatch={
            "mpbench":("mpbench_runner","run_mpbench"),
            "pmbench":("pmbench_runner","run_pmbench"),
            "memoryagentbench":("memoryagentbench_runner","run_memoryagentbench"),
            "memform":("memform_runner","run_memform"),
            "graphwalks":("graphwalks_runner","run_graphwalks"),
            "automationbench":("automationbench_runner","run_automationbench"),
            "agentdojo":("security_runner","run_agentdojo"),
            "injecagent":("security_runner","run_injecagent"),
        }
        if name in dispatch:
            import importlib
            module,function=dispatch[name]
            return await getattr(importlib.import_module("."+module,__package__),function)(self)
        from .datasets import load_locomo, load_longmemeval
        from .extra_datasets import load_memconflict
        loaders = {"locomo": load_locomo, "longmemeval": load_longmemeval, "memconflict": load_memconflict}
        items = await asyncio.to_thread(loaders[name], self.runs / "upstream" / name)
        trials = 3 if len(items) < 300 else 1
        root = self.runs / name
        root.mkdir(exist_ok=True)
        write_shard_map(root / "shards.json", [i.id for i in items], workers=10, seed=20261101)
        groups={}
        for item in items:
            if item.group_id not in groups: groups[item.group_id]=h(item.history)
        inputs={"groups":groups,"questions":[{"id":i.id,"group":i.group_id,"question":i.question,"category":i.category} for i in items]}
        identity = {"benchmark": name, "items": h(inputs),
                    "prompts": self.prompts, "binary": self.manifest["binary_sha256"],
                    "hooks_manifest_sha256":qualified["hooks_manifest_sha256"],
                    "retrieval":"bge-token-capacity-chunks-return-complete-original-v1",
                    "model": MODEL, "limits": [MAX_ITERS, OUTPUT_BUDGET, CONTEXT_LIMIT], "trials": trials}
        identity['config_hash']=self.config_hash
        from .datasets import SOURCES
        from .engine import sha256
        if name in SOURCES:
            source=SOURCES[name];up=self.runs/'upstream'/name
            dataset_files={str(up/source['data_file']):source['data_sha256']}
            scorer={'origin':'official','entrypoint':'score_locomo' if name=='locomo' else 'judge_longmemeval',
                    'source_files':{str(up/source['scorer_file']):source['scorer_sha256']}}
        else:
            from .extra_datasets import verify_memconflict
            up=self.runs/'upstream'/name;source=verify_memconflict(up)
            dataset_files={str(up/'Data/Step4_4.jsonl'):source['dataset_sha256']}
            scorer={'origin':'official','entrypoint':'judge_memconflict',
                'source_files':{str(up/p):v for p,v in source['files'].items() if p.startswith('Evaluation/')}}
        plan=self.prepare_plan(name,items,identity,dataset_files,scorer)
        self.identities[name]=identity
        atomic_write(root / "identity.json", canonical(identity)+"\n")
        pilot_ids=None
        if self.is_preflight(name):
            pilot_ids=set(plan['pilot']['item_ids'])
            atomic_write(root/"integrity-pilot.json",canonical({"stage":"unscored_integrity_preflight",
                "seed":20261101,"item_ids":sorted(pilot_ids),"official_full_count":len(items),
                "full_manifest_preserved":True,"identity":identity})+"\n")
        checkpoints = {arm: [Checkpoint(root / arm / f"worker-{k}.jsonl", {**identity,"arm":arm}) for k in range(10)] for arm in ARMS}
        for arm in ARMS:
            self.progress[name,arm] = {"benchmark": name, "arm": arm,
                "done":sum(len(c.done) for c in checkpoints[arm]), "total":len(items)*trials, "errors":0}
        projection_path = root / "projection.json"
        if not projection_path.exists():
            calibration_item=next(i for i in items if not pilot_ids or i.id in pilot_ids)
            calibration=[]
            for arm in ARMS:
                worker = shard_id(calibration_item.id)
                cp=checkpoints[arm][worker]
                key=record_key({"item_id":calibration_item.id,"trial":0})
                if key in cp.done:
                    record=cp.done[key]
                else:
                    record = await self.solve(name,calibration_item,arm,0,worker)
                    cp.append(record)
                    self.progress[name,arm]["done"] += 1
                judge_cost=0.0
                if name!='locomo':
                    # No pilot scoring. Price the exact official judge input
                    # and its capped1024output without invoking the judge.
                    from .provider import PRICING
                    if name=='longmemeval':
                        from .datasets import official_judge_prompt
                        judge_prompt=official_judge_prompt(self.runs/'upstream'/name,calibration_item,record['response'])
                    else:
                        from .extra_datasets import memconflict_judge_prompt
                        judge_prompt=memconflict_judge_prompt(self.runs/'upstream'/name,calibration_item,record['response'])
                    judge_cost=(tokens(judge_prompt)*PRICING['input']+1024*PRICING['output'])/1e6
                calibration.append({"arm":arm,"cost":record["cost"],"judge_cost":judge_cost})
            projected=sum(x["cost"] for x in calibration)*len(items)*trials
            judge_projection=sum(x["judge_cost"] for x in calibration)*len(items)*trials
            projection={"first_item":calibration_item.id,"calibration":calibration,
                "solver_projected_cost":projected,"judge_projected_cost":judge_projection,
                "projected_total":projected+judge_projection,"remaining":self.ledger.snapshot()["remaining"],
                "method":"one official pilot per arm; measured solver usage; exact judge input plus capped output estimate; no pilot scored"}
            atomic_write(projection_path,canonical(projection)+"\n")
            print(name,"PROJECTED_COST",json.dumps(projection),flush=True)
        projection=json.loads(projection_path.read_text()) if projection_path.exists() else None
        if projection and projection["projected_total"] > self.ledger.snapshot()["remaining"]:
            append_jsonl(self.runs/"decisions.jsonl",{"benchmark":name,
                "decision":"Full official set retained despite forecast exceeding remaining funds; stop only at actual shared pre-call budget gate",
                "projection":projection,"timestamp":utc_now()})

        async def worker(arm,k):
            cp=checkpoints[arm][k]
            failed=[]
            async def do(item,trial,attempt):
                if cp.contains({"item_id":item.id,"trial":trial}): return
                try:
                    row=await self.solve(name,item,arm,trial,k,attempt)
                    cp.append(row)
                    self.progress[name,arm]["done"]+=1
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    cp.append_failure({"item_id":item.id,"trial":trial,"error_type":type(exc).__name__,"error":str(exc)[:500]})
                    self.progress[name,arm]["errors"]+=1
                    if not attempt: failed.append((item,trial))
            try:
                for trial in range(trials):
                    for item in items:
                        if (not pilot_ids or item.id in pilot_ids) and shard_id(item.id)==k: await do(item,trial,0)
                for item,trial in failed: await do(item,trial,1)
            except BudgetExceeded:
                atomic_write(root/arm/f"worker-{k}.stopped.json",canonical({"reason":"budget","timestamp":utc_now()})+"\n")
        await asyncio.gather(*(worker(arm,k) for arm in ARMS for k in range(10)))
        wanted={i.id for i in items}
        populations={}
        for arm in ARMS:
            done={row['item_id'] for cp in checkpoints[arm] for row in cp.done.values()}
            populations[arm]={'expected_n':len(wanted),'completed_n':len(done),
                'completed_item_ids':sorted(done),'uncompleted_item_ids':sorted(wanted-done)}
        atomic_write(root/'coverage.json',canonical({'updated_at':utc_now(),'arms':populations})+'\n')
        if self.is_preflight(name):
            atomic_write(root/"integrity-pilot-complete.json",canonical({"item_ids":sorted(pilot_ids),
                "completed":{arm:sum(len(c.done) for c in checkpoints[arm]) for arm in ARMS},
                "scored":False,"full_manifest_count":len(items)})+"\n")
            print(name,"UNSCORED_INTEGRITY_PILOT_FINISHED",flush=True)
            return
        from .integrity import audit,require
        require(await asyncio.to_thread(audit,REPO,self.runs,benchmarks=self.names),name,"score")
        await self.score(name,items,trials,checkpoints)

    async def score(self,name,items,trials,checkpoints):
        from .datasets import score_locomo, judge_longmemeval
        from .extra_datasets import judge_memconflict
        root=self.runs/name
        summaries=[]
        expected=[record_key({"item_id":i.id,"trial":t}) for t in range(trials) for i in items]
        # All four full sets are merged and independently audited before any
        # official scorer consumes a prediction. No pilot answer is scored.
        for arm in ARMS:
            try:
                strict_merge([cp.path for cp in checkpoints[arm]],expected,root/arm/'merged.jsonl')
            except ValueError as exc:
                atomic_write(root/'UNFINISHED.json',canonical({'reason':str(exc),'arm':arm,'timestamp':utc_now()})+'\n')
                return
        from .integrity import audit,require
        require(await asyncio.to_thread(audit,REPO,self.runs,benchmarks=self.names),name,'scorer')
        plan=json.loads((self.runs/'QUALIFICATION.json').read_text())['benchmarks'][name]
        for arm in ARMS:
            try:
                merged=strict_merge([cp.path for cp in checkpoints[arm]],expected,root/arm/"merged.jsonl")
            except ValueError as exc:
                summaries.append({"arm":arm,"status":"unfinished","reason":str(exc)})
                continue
            scored_path=root/arm/"scored.jsonl"
            if scored_path.exists():
                scored=read_jsonl(scored_path)
                if len(scored)!=len(expected) or {record_key(r) for r in scored}!=set(expected):
                    raise ValueError("scored checkpoint coverage or uniqueness mismatch")
                merged_by_key={record_key(r):r for r in merged}
                for row in scored:
                    original=merged_by_key[record_key(row)]
                    if any(row.get(k)!=original.get(k) for k in ("run_identity","response","arm","model","binary_sha256")):
                        raise ValueError("scored checkpoint identity or response mismatch")
            else:
                receipt=root/arm/'scorer-invocations.jsonl'
                if any(e.get('event')=='official_scorer_complete' for e in read_jsonl(receipt)):
                    raise RuntimeError('completed scorer receipt without scored output; preserve and reconcile')
                if not receipt.exists():
                    append_jsonl(receipt,{'event':'official_scorer_started','timestamp':utc_now(),
                        'merged_sha256':hashlib.sha256((root/arm/'merged.jsonl').read_bytes()).hexdigest(),
                        'scorer_source_files':plan['scorer']['source_files']})
                scored=[]
                for trial in range(trials):
                    records=[r for r in merged if r["trial"]==trial]
                    predictions={r["item_id"]:r["response"] for r in records}
                    if name=="locomo":
                        scores=await asyncio.to_thread(score_locomo,self.runs/"upstream"/name,items,predictions)
                    else:
                        async def judge_callback(*args,**kwargs):
                            return await self.judge_request(name,arm,trial,args[0])
                        scorer=judge_longmemeval if name=="longmemeval" else judge_memconflict
                        scores,judge_logs=await scorer(self.runs/"upstream"/name,items,predictions,judge_callback)
                        atomic_write(root/arm/f"judge-results-{trial}.json",canonical(judge_logs)+"\n")
                    scored.extend({**r,"score":float(s)} for r,s in zip(records,scores,strict=True))
                atomic_write(scored_path,"".join(canonical(r)+"\n" for r in scored))
                append_jsonl(receipt,{'event':'official_scorer_complete','timestamp':utc_now(),
                    'merged_sha256':hashlib.sha256((root/arm/'merged.jsonl').read_bytes()).hexdigest(),
                    'item_count':len(expected),'output_count':len(scored),
                    'scorer_source_files':plan['scorer']['source_files']})
            scores=[r["score"] for r in scored]
            with sqlite3.connect(self.runs/"budget.db") as db:
                actual,reserved=db.execute("SELECT COALESCE(SUM(actual),0), COALESCE(SUM(CASE WHEN state='reserved' THEN reserved ELSE 0 END),0) FROM calls WHERE id LIKE ?",(f"{name}/{arm}/%",)).fetchone()
            summaries.append({"arm":arm,"status":"complete","mean":sum(scores)/len(scores),
                "ci95":ci(scores,[r["group_id"].rsplit(":s",1)[0] if name=="memconflict" else r["group_id"] for r in scored]),"n":len(scores),"trials":trials,
                "model":MODEL,"solver_cost":sum(r["cost"] for r in scored),
                "cost_all_metered_calls":actual/1e9,"unreconciled_reservations":reserved/1e9,
                "binary_sha256":self.manifest["binary_sha256"]})
        atomic_write(root/"results.json",canonical(summaries)+"\n")
        print(name,"RESULTS",canonical(summaries),flush=True)
        await self.write_report(name,summaries,plan)

    async def write_report(self,name,summaries,plan):
        from .integrity import audit,require
        require(await asyncio.to_thread(audit,REPO,self.runs,benchmarks=self.names),name,'report')
        rows=['# '+name+' real-engine results','',
            'Model and judge: gpt-5.4-nano. Temperature: 0. Trial/seed: 0. Ten deterministic workers per arm; global20-request limit.',
            '', '| Arm | Mean | 95% cluster bootstrap CI | n | API cost | Binary SHA-256 |',
            '|---|---:|---|---:|---:|---|']
        for s in summaries:
            rows.append(f"| {s['arm']} | {s['mean']:.6f} | {s['ci95']} | {s['n']} | ${s['cost_all_metered_calls']:.6f} | {s['binary_sha256']} |")
        rows+=['','Vestige includes the source-wired memory-QA SessionStart and memory-injection hooks. Action hooks are not applicable to this loop.',
            '', 'Dataset files and SHA-256: `'+canonical(plan['full_set']['dataset_files'])+'`.',
            'Scorer source pins: `'+canonical(plan['scorer'])+'`.',
            'Configuration SHA-256: `'+self.config_hash+'`.',
            '', 'Costs include all metered benchmark calls, including earlier preparation. Unresolved reservations are reported separately in STATUS.md.',
            'Published, different model or judge: see the independently sourced records in benchmarks/real_engine/published.json; no head-to-head ranking inferred.', '']
        atomic_write(self.runs/name/'RESULTS.md','\n'.join(rows))

    async def pipeline(self,name):
        from .integrity import audit,require
        if not self.autonomous:
            return await self.benchmark(name)
        await self.benchmark(name)
        if name=='mpbench':
            from .mpbench import load_mpbench
            from .mpbench_runner import run_item
            # Real engine death control is shared in the engine-based QA loop;
            # MPBench's own runner supplies its control separately below.
            cases,_=await asyncio.to_thread(load_mpbench,self.runs/'upstream')
            await run_item(self,cases[0],'vestige',0,kill_control=True)
        else:
            from .datasets import load_locomo,load_longmemeval
            from .extra_datasets import load_memconflict
            loader={'locomo':load_locomo,'longmemeval':load_longmemeval,'memconflict':load_memconflict}[name]
            items=await asyncio.to_thread(loader,self.runs/'upstream'/name)
            await self.solve(name,items[0],'vestige',0,0,kill_control=True)
        report=await asyncio.to_thread(audit,REPO,self.runs,stage='execute',benchmarks=self.names)
        require(report,name,'execute')
        append_jsonl(self.runs/'decisions.jsonl',{'benchmark':name,'timestamp':utc_now(),
            'decision':'Ten-item current-config pilot PASS; start full official set immediately','config_hash':self.config_hash})
        print(name,'PILOT_PASS_STARTING_FULL_SET',flush=True)
        self.preflight_counts[name]=0
        await self.benchmark(name)

    async def run(self):
        atomic_write(self.runs/"prompt-audit.json",canonical(self.prompts)+"\n")
        stop_task=asyncio.create_task(status_loop(self.runs/"STATUS.md",lambda:list(self.progress.values()),self.ledger,self.stop))
        try:
            # Start calibration in the requested priority order. Once each cost
            # gate opens, benchmark workers run concurrently on the same pool.
            tasks=[]
            for name in self.names:
                tasks.append(asyncio.create_task(self.pipeline(name)))
                await asyncio.sleep(0)
            results=await asyncio.gather(*tasks,return_exceptions=True)
            for name,result in zip(self.names,results):
                if isinstance(result,BaseException):
                    append_jsonl(self.runs/"campaign-errors.jsonl",{"benchmark":name,"error_type":type(result).__name__,"error":str(result)[:500],"timestamp":utc_now()})
                    print(name,"FAILED",type(result).__name__,str(result)[:500],flush=True)
            atomic_write(self.runs/'campaign-finished.json',canonical({'timestamp':utc_now(),
                'config_hash':self.config_hash,'ledger':self.ledger.snapshot(),
                'outcomes':{name:({'error_type':type(result).__name__,'error':str(result)[:500]} if isinstance(result,BaseException)
                                  else {'runner_returned':True}) for name,result in zip(self.names,results)}})+'\n')
        finally:
            from .engine import close_all_engines
            await asyncio.to_thread(close_all_engines)
            self.stop.set()
            await stop_task
            await self.provider.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--benchmarks",default=','.join(FOUR))
    parser.add_argument("--runs",type=Path,default=REPO/"runs")
    parser.add_argument("--preflight",type=int,default=0,help="Unscored integrity pilot only; full official manifest retained")
    parser.add_argument('--autonomous',action='store_true',help='Fresh10pilot -> independent audit -> full set for each of the four benchmarks')
    args=parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    args.runs.mkdir(parents=True,exist_ok=True)
    with (args.runs/"campaign.lock").open("a") as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("A campaign already owns the global API semaphore; do not launch a second process")
        async def run():
            campaign=Campaign(args.runs,args.benchmarks.split(','),args.preflight,args.autonomous)
            task=asyncio.current_task()
            loop=asyncio.get_running_loop()
            for sig in (signal.SIGINT,signal.SIGTERM):
                loop.add_signal_handler(sig,task.cancel)
            await campaign.run()
        try:
            asyncio.run(run())
        finally:
            from .engine import close_all_engines
            close_all_engines()


if __name__=="__main__": main()

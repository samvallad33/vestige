"""Pinned complete MemoryAgentBench splits and unchanged upstream scorers.

The same upstream long-context query template is used for all arms. No gold
field is available in solver_record(). Each question receives an independent
replay of the official context, which preserves isolation at additional ingest
cost; this runner does not share query outputs between questions.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import string
import subprocess
import sys
from types import SimpleNamespace
import uuid

from .datasets import BenchmarkItem
from .runtime import (BudgetExceeded, Checkpoint, atomic_write, canonical,
    record_key, shard_id, strict_merge, write_shard_map)

CODE_REVISION = "538026089d1a8a8eff05121d0db89b388f360eba"
DATA_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
SPLITS = ("Conflict_Resolution", "Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding")
DATA_PINS = {
 "Conflict_Resolution": ("24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",800),
 "Accurate_Retrieval": ("56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",2000),
 "Test_Time_Learning": ("5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",700),
 "Long_Range_Understanding": ("5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",171),
}
CODE_PINS = {
 "utils/eval_other_utils.py":"d77976be409298970614d477a9d8003850caddb0510e56a7e821a037d98493a2",
 "utils/templates.py":"148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd",
 "conversation_creator.py":"9a527e88c9791b37d378b59adac50d9308aac49e714f1d8ddaf3e14bb3f3bec5",
 "llm_based_eval/longmem_qa_evaluate.py":"ceca59bddea4203a78b417be635e63fe3f685bd957e3df12150978e21972c540",
 "llm_based_eval/summarization_evaluate.py":"6951d19644cdb1972bcf6cdb272f4757aa995ae07099bf74012925158c376dd4",
}
ENTITY_SHA = "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(root):
    root=Path(root)
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if actual!=CODE_REVISION: raise ValueError("MemoryAgentBench source commit mismatch")
    for name,digest in CODE_PINS.items():
        if _sha(root/name)!=digest: raise ValueError(f"official source hash mismatch: {name}")
    return root


def _module(root,name):
    spec=importlib.util.spec_from_file_location("_official_mab_"+uuid.uuid4().hex,Path(root)/name)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root):
    """Fetch only exact official data revision; source checkout must preexist."""
    from huggingface_hub import hf_hub_download
    root=verify_source(root)
    for split,(digest,_) in DATA_PINS.items():
        path=Path(hf_hub_download("ai-hyz/MemoryAgentBench",f"data/{split}-00000-of-00001.parquet",
                    repo_type="dataset",revision=DATA_REVISION,local_dir=root/"hf"))
        if _sha(path)!=digest: raise ValueError(f"official data hash mismatch: {split}")
    entity=Path(hf_hub_download("ai-hyz/MemoryAgentBench","entity2id.json",repo_type="dataset",
                revision=DATA_REVISION,local_dir=root/"hf"))
    if _sha(entity)!=ENTITY_SHA: raise ValueError("movie entity mapping hash mismatch")
    dest=root/"processed_data/Recsys_Redial/entity2id.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(entity,dest)
    return root


def load_memoryagentbench(root, splits=SPLITS):
    """Read every official row and QA, with Conflict Resolution first."""
    import pyarrow.parquet as pq
    root=verify_source(root)
    metrics=_module(root,"utils/eval_other_utils.py")
    templates=_module(root,"utils/templates.py")
    result=[]
    for split in splits:
        if split not in SPLITS: raise ValueError(f"unknown official split: {split}")
        path=root/"hf/data"/f"{split}-00000-of-00001.parquet"
        if _sha(path)!=DATA_PINS[split][0]: raise ValueError(f"official data hash mismatch: {split}")
        count=0
        for context_index,row in enumerate(pq.read_table(path).to_pylist()):
            metadata=row["metadata"]
            source=metadata["source"]
            questions,answers=row["questions"],row["answers"]
            if len(questions)!=len(answers): raise ValueError("official QA list length mismatch")
            history=metrics.chunk_text_into_sentences(row["context"],chunk_size=4096)
            template=templates.get_template(source,"query","Long_context_agent")
            fields={field for _,field,_,_ in string.Formatter().parse(template) if field}
            # Official query templates may include public demonstration text,
            # never the evaluated answer or scoring-only keypoints.
            if fields-{"question","demo","question_dates","question_ids","previous_events"}:
                raise ValueError(f"unsupported or gold-bearing official template fields: {fields}")
            for qi,(question,answer) in enumerate(zip(questions,answers,strict=True)):
                values={"question":question,"demo":metadata.get("demo") or ""}
                for field in fields-{"question","demo"}:
                    value=metadata.get(field)
                    values[field]=value[qi] if isinstance(value,list) and qi<len(value) else value
                formatted=template.format(**values)
                qids=metadata.get("qa_pair_ids") or []
                qtypes=metadata.get("question_types") or []
                question_ids=metadata.get("question_ids") or []
                result.append(BenchmarkItem(
                    id=f"mab:{split}:{context_index}:{qi}",
                    group_id=f"mab:{split}:{context_index}",history=history,
                    question=formatted,category=source,
                    scorer_payload={"split":split,"source":source,"question":question,"answer":answer,
                        "qa_pair_id":qids[qi] if qi<len(qids) else None,
                        "question_type":qtypes[qi] if qi<len(qtypes) else None,
                        "question_id":question_ids[qi] if qi<len(question_ids) else None,
                        "keypoints":metadata.get("keypoints")}))
                count+=1
        if count!=DATA_PINS[split][1]: raise ValueError(f"official full split count mismatch: {split}={count}")
    if len({x.id for x in result})!=len(result): raise ValueError("duplicate official QA ids")
    return result


def _response(value):
    return value["response"] if isinstance(value,dict) else value


def _standard_scores(root,records):
    """Run original post_process in its own cwd, so concurrent runs cannot race."""
    script="""import importlib.util,json,sys\np='utils/eval_other_utils.py'\ns=importlib.util.spec_from_file_location('mab_scorer',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nrows=json.load(sys.stdin);out=[]\nfor r in rows:\n metrics,extra=m.post_process({'output':r['response']},r['answer'],{'sub_dataset':r['source']})\n out.append({'id':r['id'],'metrics':metrics,'scorer_extra':extra})\nprint(json.dumps(out))\n"""
    p=subprocess.run([sys.executable,"-c",script],cwd=root,input=json.dumps(records),text=True,capture_output=True,check=True)
    return json.loads(p.stdout)


async def score_memoryagentbench(root,items,predictions,request,workdir,*,campaign=None):
    """Score one strictly merged trial with official code and budgeted nano.

    request(prompt) is supplied by the one global campaign provider. No scorer
    in this module opens its own network client or unmetered API transport.
    """
    root=verify_source(root); workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
    if set(predictions)!={x.id for x in items}: raise ValueError("merged QA coverage differs from official set")
    if campaign is None:
        from .integrity import IntegrityPending
        raise IntegrityPending("MemoryAgentBench official scoring requires the current independent campaign audit")
    await require_integrity(campaign,"scorer")
    standard=[]; summaries=[]; results={}
    lm=None
    for item in items:
        gold=item.scorer_payload;source=gold["source"];response=predictions[item.id]
        if "longmemeval" in source:
            if lm is None: lm=_module(root,"llm_based_eval/longmem_qa_evaluate.py")
            prompt=lm.get_anscheck_prompt(gold["question_type"],gold["question"],gold["answer"],response,
                    abstention="_abs" in (gold["question_id"] or ""))
            judge=await request(prompt)
            results[item.id]={"score":float("yes" in _response(judge).lower()),"metric":"official_llm_yes","judge":judge}
        elif "infbench" in source:
            summaries.append({"qa_pair_id":gold["qa_pair_id"],"answer":gold["answer"],"output":response,"_id":item.id})
        else:
            standard.append({"id":item.id,"source":source,"answer":gold["answer"],"response":response})
    for row in await asyncio.to_thread(_standard_scores,root,standard):
        source=next(x["source"] for x in standard if x["id"]==row["id"])
        metric="recsys_recall@5" if "recsys" in source else ("exact_match" if "icl" in source or "detective" in source else "substring_exact_match")
        if metric not in row["metrics"]: raise ValueError(f"official scorer missing metric: {metric}")
        results[row["id"]]={"score":float(row["metrics"][metric]),"metric":metric,**row}
    if summaries:
        module=_module(root,"llm_based_eval/summarization_evaluate.py")
        # Pin only the input-data loader. The scorer, prompts, JSON parsing and
        # F1 calculation remain exactly the source-hashed upstream functions.
        from datasets import load_dataset
        module.load_dataset=lambda *a,**kw: load_dataset("parquet",data_files={s:str(root/"hf/data"/f"{s}-00000-of-00001.parquet") for s in SPLITS})
        loop=asyncio.get_running_loop()
        class SharedBudgetModel:
            def generate_batch(self,prompt,batch_file):
                return [{"output":_response(asyncio.run_coroutine_threadsafe(request(p),loop).result())} for p in prompt]
        inpath=workdir/"infbench-merged.json";outpath=workdir/"infbench-official-scored.json"
        atomic_write(inpath,canonical({"data":summaries,"averaged_metrics":{}})+"\n")
        scored=await asyncio.to_thread(module.check_metrics,SharedBudgetModel(),str(inpath),str(outpath),SimpleNamespace(huggingface_dataset_name="ai-hyz/MemoryAgentBench"))
        for row in scored["data"]:
            if "gpt-4-scores" not in row: raise ValueError("official summarization judge omitted item; incomplete scoring")
            results[row["_id"]]={"score":float(row["gpt-4-scores"]["f1"]),"metric":"official_summary_f1","official_scores":row["gpt-4-scores"]}
    if set(results)!=set(predictions): raise ValueError("official scoring omitted item")
    return results


def flush_full_coverage(root,items,trials,checkpoints,reason=None):
    """Exact official denominator and unfinished work; never shrink a set."""
    expected=[{"item_id":item.id,"trial":trial,"shard":shard_id(item.id)}
              for trial in range(trials) for item in items]
    by_arm={}
    for arm,workers in checkpoints.items():
        done={key for worker in workers for key in worker.done}
        completed=[row for row in expected if record_key(row) in done]
        uncompleted=[row for row in expected if record_key(row) not in done]
        by_arm[arm]={"expected":len(expected),"completed_count":len(completed),
                    "uncompleted_count":len(uncompleted),"completed":completed,"uncompleted":uncompleted}
    artifact={"scope":"full official split, no sampled score denominator","reason":reason,"arms":by_arm}
    atomic_write(Path(root)/"fullset-coverage.json",canonical(artifact)+"\n")
    return artifact


async def require_integrity(campaign,stage):
    from .integrity import audit,require
    report=await asyncio.to_thread(audit,Path(__file__).resolve().parents[2],campaign.runs,stage=stage)
    require(report,"memoryagentbench",stage)
    return report


async def run_memoryagentbench(campaign):
    """Full official corpus, ten workers, and a separate unscored pilot gate."""
    import random
    from .run import ARMS,ci,h,MAX_ITERS,OUTPUT_BUDGET,CONTEXT_LIMIT
    from .runtime import utc_now
    source=campaign.runs/"upstream/memoryagentbench"
    items=await asyncio.to_thread(load_memoryagentbench,source)
    audit_path=campaign.runs/"hook-audit.json"
    hooks_manifest=json.loads(audit_path.read_text())
    if hooks_manifest.get("audit_complete") is not True:
        raise RuntimeError("MemoryAgentBench hook source audit is incomplete")
    workers=10
    count=int(getattr(campaign,"preflight_count",0) or 0)
    selected_pilot=set(x.id for x in random.Random(20261101).sample(items,min(count,len(items)))) if count else set()
    aggregate=campaign.runs/"memoryagentbench";aggregate.mkdir(parents=True,exist_ok=True)
    history_hashes={}
    for item in items:
        if item.group_id not in history_hashes:history_hashes[item.group_id]=h(item.history)
    plans=[];all_expected=[]
    for split in SPLITS:
        selected=[i for i in items if i.scorer_payload["split"]==split]
        trials=3 if len(selected)<300 else 1
        name="memoryagentbench-"+split.lower()
        root=campaign.runs/name;root.mkdir(parents=True,exist_ok=True)
        write_shard_map(root/"shards.json",[x.id for x in selected],workers=workers,seed=20261101)
        identity={"benchmark":name,"model":"gpt-5.4-nano","dataset_revision":DATA_REVISION,
            "data_sha256":DATA_PINS[split][0],"code_revision":CODE_REVISION,
            "items":h([{"id":x.id,"group_id":x.group_id,"history_sha256":history_hashes[x.group_id],
                        "question_sha256":h(x.question),"category":x.category} for x in selected]),
            "binary":campaign.manifest["binary_sha256"],"prompts":campaign.prompts,"trials":trials,"workers":workers,
            "vestige_arm":"all_applicable_hooks","hook_audit_sha256":_sha(audit_path),
            "hook_adapter_sha256":_sha(Path(__file__).with_name("hooks.py")),
            "model_budget":{"aggregate_calls_including_hooks":MAX_ITERS,"aggregate_output_tokens_including_hooks":OUTPUT_BUDGET,"context_tokens":CONTEXT_LIMIT}}
        checkpoints={arm:[Checkpoint(root/arm/f"worker-{w}.jsonl",{**identity,"arm":arm}) for w in range(workers)] for arm in ARMS}
        for arm in ARMS:campaign.progress[name,arm]={"benchmark":name,"arm":arm,"done":sum(len(x.done) for x in checkpoints[arm]),"total":len(selected)*trials,"errors":0}
        plans.append({"name":name,"split":split,"root":root,"items":selected,"trials":trials,"checkpoints":checkpoints})
        all_expected.extend({"item_id":item.id,"trial":trial,"split":split,"shard":shard_id(item.id,workers=workers)} for trial in range(trials) for item in selected)
    # The official full denominator is frozen before the first request. The
    # diagnostic pilot never replaces it or enters an official scorer.
    atomic_write(aggregate/"fullset-expected.json",canonical({"dataset_revision":DATA_REVISION,"source_revision":CODE_REVISION,"units_per_arm":len(all_expected),"workers":workers,"units":all_expected})+"\n")
    if count:atomic_write(aggregate/"unscored-pilot-selection.json",canonical({"seed":20261101,"item_ids":sorted(selected_pilot),"trial":0,"purpose":"unscored integrity evidence only; full official denominator retained"})+"\n")
    def flush(reason=None):
        arms={}
        for arm in ARMS:
            done={key for plan in plans for worker in plan["checkpoints"][arm] for key in worker.done}
            completed=[row for row in all_expected if record_key(row) in done]
            unfinished=[row for row in all_expected if record_key(row) not in done]
            arms[arm]={"expected":len(all_expected),"completed_count":len(completed),"uncompleted_count":len(unfinished),"completed":completed,"uncompleted":unfinished}
        atomic_write(aggregate/"fullset-coverage.json",canonical({"reason":reason,"scope":"full official set; no denominator reduction","arms":arms})+"\n")
        for plan in plans:flush_full_coverage(plan["root"],plan["items"],plan["trials"],plan["checkpoints"],reason)
    flush("initialized before model calls")
    await require_integrity(campaign,"preflight" if count else "execute")
    try:
        # Conflict Resolution first. Competency split boundaries stay intact.
        for plan in plans:
            name,root,selected,trials,cp=(plan[k] for k in ("name","root","items","trials","checkpoints"))
            active=[x for x in selected if not count or x.id in selected_pilot]
            if not active:continue
            calibration=[]
            for arm in ARMS:
                item=active[0];worker=shard_id(item.id,workers=workers);checkpoint=cp[arm][worker]
                key=record_key({"item_id":item.id,"trial":0})
                if key in checkpoint.done:row=checkpoint.done[key]
                else:
                    row=await campaign.solve(name,item,arm,0,worker)
                    if count:row["unscored_pilot"]=True
                    checkpoint.append(row);campaign.progress[name,arm]["done"]+=1
                calibration.append(row["cost"])
            judge_count=sum((3 if "infbench" in x.category else 1 if "longmemeval" in x.category else 0) for x in selected)*trials*len(ARMS)
            projection={"solver":sum(calibration)*len(selected)*trials,"judge_requests":judge_count,"judge_allowance":judge_count*0.01,
                "remaining":campaign.ledger.snapshot()["remaining"],"method":"one official item per arm; actual usage forecast only, never a full-set skip gate"}
            projection["total"]=projection["solver"]+projection["judge_allowance"]
            atomic_write(root/"projection.json",canonical(projection)+"\n");print(name,"PROJECTED_COST",canonical(projection),flush=True)
            async def worker(arm,w):
                failures=[]
                async def one(item,trial,attempt):
                    checkpoint=cp[arm][w]
                    if checkpoint.contains({"item_id":item.id,"trial":trial}):return
                    try:
                        row=await campaign.solve(name,item,arm,trial,w,attempt)
                        if count:row["unscored_pilot"]=True
                        checkpoint.append(row);campaign.progress[name,arm]["done"]+=1
                    except BudgetExceeded:raise
                    except Exception as exc:
                        checkpoint.append_failure({"item_id":item.id,"trial":trial,"error_type":type(exc).__name__,"error":str(exc)[:500]})
                        campaign.progress[name,arm]["errors"]+=1
                        if not attempt:failures.append((item,trial))
                try:
                    for trial in range(1 if count else trials):
                        for item in active:
                            if shard_id(item.id,workers=workers)==w:await one(item,trial,0)
                    for item,trial in failures:await one(item,trial,1)
                except BudgetExceeded:
                    atomic_write(root/arm/f"worker-{w}.stopped.json",canonical({"reason":"shared global budget","timestamp":utc_now(),"fullset_unchanged":True})+"\n")
            await asyncio.gather(*(worker(arm,w) for arm in ARMS for w in range(workers)))
            flush("worker pass ended; all uncompleted official IDs retained")
        if count:
            return {"status":"unscored_pilot_only","official_items":3671,"full_units_per_arm":len(all_expected),"requested_pilot_items_per_arm":len(selected_pilot)}
        expected=[record_key(x) for x in all_expected]
        merged={};incomplete=[]
        for arm in ARMS:
            try:merged[arm]=strict_merge([worker.path for plan in plans for worker in plan["checkpoints"][arm]],expected,aggregate/arm/"merged.jsonl")
            except ValueError as exc:incomplete.append({"arm":arm,"status":"unfinished","reason":str(exc)})
        if incomplete:
            atomic_write(aggregate/"UNFINISHED.json",canonical(incomplete)+"\n")
            return incomplete
        # All four full merged arms exist before the one official scoring pass
        # for each arm/trial. No model score is emitted from a sampled pilot.
        summaries=[]
        for arm in ARMS:
            scored=[];judge_cost=0.0
            for trial in range(3):
                trial_items=items if trial==0 else [x for x in items if x.scorer_payload["split"]=="Long_Range_Understanding"]
                rows=[x for x in merged[arm] if x["trial"]==trial]
                async def request(prompt):
                    nonlocal judge_cost
                    result=await campaign.judge_request("memoryagentbench",arm,trial,prompt);judge_cost+=result.get("cost",0.0);return result
                metrics=await score_memoryagentbench(source,trial_items,{x["item_id"]:x["response"] for x in rows},request,aggregate/arm/f"official-score-trial-{trial}",campaign=campaign)
                scored.extend({**row,**metrics[row["item_id"]]} for row in rows)
            atomic_write(aggregate/arm/"scored.jsonl","".join(canonical(x)+"\n" for x in scored))
            for plan in plans:
                rows=[x for x in scored if x["item_id"].startswith("mab:"+plan["split"]+":")]
                values=[x["score"] for x in rows]
                summaries.append({"split":plan["split"],"arm":arm,"status":"complete","mean":sum(values)/len(values),
                    "ci95":ci(values,[x["group_id"] for x in rows]),"n":len(values),"trials":plan["trials"],"model":"gpt-5.4-nano",
                    "data_sha256":DATA_PINS[plan["split"]][0],"binary_sha256":campaign.manifest["binary_sha256"],
                    "solver_cost":sum(x["cost"] for x in rows),"judge_cost_arm_total":judge_cost,"cost_arm_total":sum(x["cost"] for x in scored)+judge_cost})
        await require_integrity(campaign,"report")
        atomic_write(campaign.runs/"memoryagentbench-summary.json",canonical(summaries)+"\n")
        return summaries
    finally:
        flush("execution ended; completed and exact uncompleted full-set IDs retained")

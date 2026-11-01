"""Ten official items solved by the model using the installed native workflow.

No harness recall, answer-conditioned handle selection or source selection.
Gold relevance annotations and official grading are opened after all solvers.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import types

from benchmarks.real_engine import engine as engine_module
from benchmarks.real_engine.datasets import load_longmemeval,judge_longmemeval
from benchmarks.real_engine.recall_proof import fixed_ids,_audit_rows,audit_relevance
from benchmarks.real_engine.runtime import atomic_write,canonical,utc_now,read_jsonl
from benchmarks.longmemeval_native_setup import NativeSetupEngine,NativeSetupBudgetedProvider


def observed_nodes(value):
    if isinstance(value,dict):
        if isinstance(value.get('id'),str) and isinstance(value.get('content'),str):yield value
        for child in value.values():yield from observed_nodes(child)
    elif isinstance(value,list):
        for child in value:yield from observed_nodes(child)


def inspect_model_retrieval(row,raw):
    known=set();pending={};retrieved=[];dead_ends=[];calls=[]
    for message in row['transcript']:
        if message.get('role')=='user':
            # Product memory.cjs emits JSON lines from the actual native packet.
            for line in (message.get('content') or '').split('\n'):
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and isinstance(value.get('id'),str):known.add(value['id'])
        elif message.get('role')=='assistant':
            for call in message.get('tool_calls') or []:
                args=json.loads(call['function']['arguments']);pending[call['id']]={'name':call['function']['name'],'arguments':args,'previously_observed':args.get('handle') in known};calls.append(pending[call['id']])
        elif message.get('role')=='tool':
            value=json.loads(message['content']);call=pending.get(message['tool_call_id'],{})
            if value.get('error')=='handle_required':dead_ends.append(call)
            nodes=list(observed_nodes(value));known.update(n['id'] for n in nodes)
            if call.get('name')=='recall' and call.get('previously_observed'):
                retrieved.extend(n for n in nodes if n['id']==call['arguments'].get('handle'))
    unique={n['id']:n for n in retrieved};relevant=audit_relevance(raw,list(unique.values())) if unique else []
    product=Path(row['raw_log']).parent/'native/product-memory-hooks.jsonl';hooks=read_jsonl(product)
    return {'item_id':row['item_id'],'model_tool_calls':calls,'model_recalled_previously_observed_ids':list(unique),
            'relevant_recalled_memories':relevant,'handle_required_events':dead_ends,
            'native_product_hook_events':hooks,'raw_api_log':row['raw_log'],
            'response':row['response'],'termination':row.get('termination'),'usage':row['usage'],'cost':row['cost']}


async def run_model_proof(campaign):
    repo=Path(__file__).resolve().parents[1];root=repo/'runs/lme-model-driven-proof-v3';root.mkdir(exist_ok=True)
    if (root/'proof.json').exists():return json.loads((root/'proof.json').read_text())
    clone=copy.copy(campaign);clone.runs=root;clone.preflight_counts={'longmemeval':10};clone.preflight_count=10
    def provider_for(self,meta,path,**limits):return NativeSetupBudgetedProvider(self.provider,meta,path,**limits)
    clone.item_provider=types.MethodType(provider_for,clone)
    sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),repo/'benchmarks/longmemeval_native_setup.py',repo/'packages/vestige-init/hooks/memory.cjs',repo/'packages/vestige-init/lib/memory-triggers.cjs',repo/'docs/integrations/codex-intelligent-memory.md']}
    clone.config_hash=hashlib.sha256(canonical({'base_config':campaign.config_hash,'product_setup_sources':sources,'proof':'MODEL_retrieves_observed_ID_then_answers_v1'}).encode()).hexdigest()
    shutil.copy2(repo/'runs/build-manifest.json',root/'build-manifest.json')
    shutil.copy2(repo/'runs/hook-audit.json',root/'hook-audit.json')
    atomic_write(root/'config.json',canonical({'config_hash':clone.config_hash,'base_config_hash':campaign.config_hash,'sources':sources,'limits':{'max_calls':6,'max_output':2048,'context':32768},'full_run_authorized':False})+'\n')
    all_items=load_longmemeval(repo/'runs/upstream/longmemeval');ids=fixed_ids(all_items);items=[i for i in all_items if i.id in ids]
    engine_module.HookedRealEngine=NativeSetupEngine
    semaphore=asyncio.Semaphore(10)
    async def solve(item):
        async with semaphore:
            checkpoint=root/(item.id+'.json')
            if checkpoint.exists():return json.loads(checkpoint.read_text())
            row=await clone.solve('longmemeval',item,'vestige',0,ids.index(item.id));atomic_write(checkpoint,canonical(row)+'\n');return row
    outcomes=await asyncio.gather(*(solve(item) for item in items),return_exceptions=True)
    # Relevance/gold labels are unavailable to the above model/engine loop.
    raw=_audit_rows(repo/'runs');entries=[];predictions={}
    for item,result in zip(items,outcomes):
        if isinstance(result,BaseException):entries.append({'item_id':item.id,'status':'FAIL','error_type':type(result).__name__,'error':str(result)});continue
        try:
            entry=inspect_model_retrieval(result,raw[item.id]);entry['status']='PENDING_JUDGE';entries.append(entry);predictions[item.id]=result['response']
        except Exception as exc:entries.append({'item_id':item.id,'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),'raw_api_log':result['raw_log']})
    # Use unchanged official prompts and parser; ten proof judgments, not scores
    # for the official500-item run. Every call uses the shared nano provider.
    judged_items=[i for i in items if i.id in predictions]
    async def judge(value):
        result=await campaign.provider.chat([{'role':'system','content':value['system']},{'role':'user','content':value['user']}],max_tokens=1024,log=root/'official-proof-judge.jsonl',identity={'benchmark':'longmemeval','arm':'vestige','item_id':value['item_id'],'phase':'model_proof_official_judge','config_hash':clone.config_hash})
        return {'response':result['message'].get('content') or '', 'usage':result['usage'],'cost':result['cost']}
    # Official helper requires its entire supplied item set, not all500 here.
    if judged_items:
        scores,judge_logs=await judge_longmemeval(repo/'runs/upstream/longmemeval',judged_items,predictions,judge)
        atomic_write(root/'official-proof-judge-results.json',canonical(judge_logs)+'\n');by_id={i.id:s for i,s in zip(judged_items,scores)}
        for entry in entries:
            if entry['item_id'] in by_id:
                entry['official_judge_score']=by_id[entry['item_id']];entry['status']='PASS' if by_id[entry['item_id']]==1 and entry.get('relevant_recalled_memories') and not entry.get('handle_required_events') else 'FAIL'
    proof={'created_at':utc_now(),'schema':'MODEL-native-discovery-recall-answer-v1','config_hash':clone.config_hash,'item_ids':ids,'status':'PASS' if len(entries)==10 and all(e['status']=='PASS' for e in entries) else 'FAIL','items':entries,'full_run_authorized':False,'old_vestige_answers_invalidated':False,'ledger':campaign.ledger.snapshot()}
    atomic_write(root/'proof.json',canonical(proof)+'\n');print('MODEL_DRIVEN_PROOF',proof['status'],[(e['item_id'],e['status']) for e in entries],flush=True);return proof

"""Owner-gated full Vestige execution after the model-driven native proof.

Shares the baseline controller's provider, semaphore, spend ledger and lock.
Old Vestige evidence is archived only after the owner's threshold is verified.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import types

from benchmarks.real_engine import engine as engine_module
from benchmarks.real_engine.datasets import load_longmemeval
from benchmarks.real_engine.runtime import Checkpoint,canonical,atomic_write,utc_now,shard_id,BudgetExceeded
from benchmarks.longmemeval_native_setup import NativeSetupEngine,NativeSetupBudgetedProvider

async def run_installed_vestige(campaign):
    repo=Path(__file__).resolve().parents[1];runs=repo/'runs';root=runs/'longmemeval';proof_root=runs/'lme-model-driven-proof-v4'
    proof=json.loads((proof_root/'proof.json').read_text())
    qualifying=[e['item_id'] for e in proof['items'] if e.get('official_judge_score')==1 and e.get('relevant_recalled_memories') and not e.get('handle_required_events')]
    errors=sum(len(e.get('handle_required_events',[])) for e in proof['items'])
    if len(proof['items'])!=10 or len(qualifying)<8 or errors:
        raise RuntimeError('Owner gate not met: at least8 real-ID correct answers and0handle_required required')
    profile=json.loads((proof_root/'config.json').read_text())
    for name,expected in profile['sources'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Proof source changed before full execution: '+name)
    gate={'timestamp':utc_now(),'authority':'mem-00000000000b0ee3','budget_authority':'mem-00000000000b0efb','qualifying_item_ids':qualifying,'handle_required_events':errors,'proof_sha256':hashlib.sha256((proof_root/'proof.json').read_bytes()).hexdigest(),'native_setup_profile':profile,'model_call_caps':{'bare':6,'bm25':6,'embedding':6,'vestige':12},'final_answer_reserved_calls':1,'final_answer_reserved_tokens':256}
    gate_path=root/'installed-vestige-qualification.json'
    if not gate_path.exists():
        archive=runs/'invalidated/vestige-before-installed-model-proof';archive.parent.mkdir(exist_ok=True)
        if archive.exists():raise RuntimeError('Archive exists without qualification; reconcile, do not overwrite')
        if (root/'vestige').exists():shutil.move(str(root/'vestige'),str(archive))
        gate['old_evidence']={'moved_from':str(root/'vestige'),'moved_to':str(archive),'reason':'Owner requested full native agent setup was missing; preserved unmodified, invalidated only after the new model proof qualified'}
        atomic_write(gate_path,canonical(gate)+'\n')
    clone=copy.copy(campaign)
    def provider_for(self,meta,path,**limits):return NativeSetupBudgetedProvider(self.provider,meta,path,**limits)
    clone.item_provider=types.MethodType(provider_for,clone)
    engine_module.HookedRealEngine=NativeSetupEngine
    identity=json.loads((root/'identity.json').read_text());native_identity={**identity,'arm':'vestige','installed_native_profile_sha256':profile['config_hash'],'max_calls':12,'owner_authority':'mem-00000000000b0efb'}
    checkpoints=[Checkpoint(root/'vestige'/f'worker-{k}.jsonl',native_identity) for k in range(30)]
    qualification=json.loads((runs/'QUALIFICATION.json').read_text());plan=qualification['benchmarks']['longmemeval'];plan['run_identities']['vestige']=checkpoints[0].identity;plan['budgets']['vestige']['max_calls']=12;plan['owner_budget_exception']={'authority':'mem-00000000000b0efb','max_calls':{'bare':6,'bm25':6,'embedding':6,'vestige':12}};plan['installed_native_setup']={'path':str(gate_path),'sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest()};atomic_write(runs/'QUALIFICATION.json',canonical(qualification)+'\n')
    items=load_longmemeval(runs/'upstream/longmemeval')
    print('OWNER_MODEL_PROOF_GATE_PASS_STARTING_FULL500',len(qualifying),errors,flush=True)
    async def worker(k):
        cp=checkpoints[k];failed=[]
        async def solve(item,attempt):
            if cp.contains({'item_id':item.id,'trial':0}):return
            try:
                row=await clone.solve('longmemeval',item,'vestige',0,k,attempt)
                row.update(installed_native_profile_sha256=profile['config_hash'],owner_budget_authority='mem-00000000000b0efb')
                cp.append(row)
            except BudgetExceeded:raise
            except Exception as exc:
                cp.append_failure({'item_id':item.id,'trial':0,'error_type':type(exc).__name__,'error':str(exc)[:1000]})
                if attempt==0:failed.append(item)
        try:
            for item in items:
                if shard_id(item.id,30)==k:await solve(item,0)
            for item in failed:await solve(item,1)
        except BudgetExceeded:atomic_write(root/'vestige'/f'worker-{k}.stopped.json',canonical({'reason':'budget','timestamp':utc_now()})+'\n')
    await asyncio.gather(*(worker(k) for k in range(30)))
    done={r['item_id'] for cp in checkpoints for r in cp.done.values()}
    atomic_write(root/'vestige-installed-coverage.json',canonical({'timestamp':utc_now(),'done':len(done),'missing_ids':[i.id for i in items if i.id not in done],'ledger':campaign.ledger.snapshot()})+'\n')

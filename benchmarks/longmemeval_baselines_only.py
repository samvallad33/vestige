"""Resume unchanged baseline solvers while the owner has stopped Vestige."""
import asyncio
import fcntl
import json
from pathlib import Path
import signal

from benchmarks.longmemeval_owner_policy import WorkerLoopPolicy
from benchmarks.real_engine.run import Campaign
from benchmarks.real_engine.datasets import load_longmemeval
from benchmarks.real_engine.runtime import Checkpoint,shard_id,record_key,status_loop,atomic_write,canonical,utc_now,BudgetExceeded

REPO=Path(__file__).resolve().parents[1]

async def run():
    runs=REPO/'runs';root=runs/'longmemeval';identity=json.loads((root/'identity.json').read_text())
    campaign=Campaign(runs,['longmemeval'],workers=30,max_inflight=40)
    items=load_longmemeval(runs/'upstream/longmemeval')
    checkpoints={a:[Checkpoint(root/a/f'worker-{k}.jsonl',{**identity,'arm':a}) for k in range(30)] for a in ('bare','bm25','embedding')}
    progress={a:{'benchmark':'longmemeval','arm':a,'done':sum(len(cp.done) for cp in checkpoints[a]),'total':500,'errors':0,'stage':'full official baseline run; Vestige stopped by owner','api_errors':0} for a in checkpoints}
    stopping=asyncio.Event();status=asyncio.create_task(status_loop(runs/'STATUS.md',lambda:list(progress.values()),campaign.ledger,stopping))
    task=asyncio.current_task();loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,task.cancel)
    async def worker(arm,k):
        cp=checkpoints[arm][k];failed=[]
        async def solve(item,attempt):
            if cp.contains({'item_id':item.id,'trial':0}):return
            try:
                row=await campaign.solve('longmemeval',item,arm,0,k,attempt);cp.append(row);progress[arm]['done']+=1
            except BudgetExceeded:raise
            except Exception as exc:
                cp.append_failure({'item_id':item.id,'trial':0,'error_type':type(exc).__name__,'error':str(exc)[:500]});progress[arm]['errors']+=1
                if attempt==0:failed.append(item)
        try:
            for item in items:
                if shard_id(item.id,30)==k:await solve(item,0)
            for item in failed:await solve(item,1)
        except BudgetExceeded:
            atomic_write(root/arm/f'worker-{k}.stopped.json',canonical({'reason':'budget','timestamp':utc_now()})+'\n')
    try:
        await asyncio.gather(*(worker(a,k) for a in checkpoints for k in range(30)))
        result={'timestamp':utc_now(),'vestige':'STOPPED_BY_OWNER','config_hash':campaign.config_hash,'arms':{a:{'done':sum(len(cp.done) for cp in cps),'missing_ids':[i.id for i in items if not cps[shard_id(i.id,30)].contains({'item_id':i.id,'trial':0})]} for a,cps in checkpoints.items()},'ledger':campaign.ledger.snapshot()}
        atomic_write(runs/'lme-baselines-finished.json',canonical(result)+'\n');print('BASELINES_FINISHED',canonical(result),flush=True)
    finally:
        stopping.set();await status;await campaign.provider.close()

if __name__=='__main__':
    asyncio.set_event_loop_policy(WorkerLoopPolicy())
    with (REPO/'runs/campaign.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(run())

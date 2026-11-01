"""Official AutomationBench worlds in Python-3.13 worker children.

Children contain only the official simulator and assertion scorer. They never
construct model clients. All inference is owned by the parent campaign's shared
NanoProvider, semaphore and spend ledger. The private leaderboard is excluded.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import uuid

REVISION = "4a8e1061254004d9dac807054eed33fad7d1ff14"
DOMAINS = ("sales","marketing","operations","support","finance","hr")
WORKERS = 5
MAX_STEPS, OUTPUT_BUDGET, CONTEXT_LIMIT = 50, 16384, 32768
MODEL = "gpt-5.4-nano"


def source_pin(root):
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if actual!=REVISION: raise RuntimeError("AutomationBench revision mismatch")
    if subprocess.check_output(["git","diff","HEAD","--name-only"],cwd=root,text=True).strip():
        raise RuntimeError("Official AutomationBench source has local modifications")


async def child_main(root):
    """One private official world at a time; JSON-RPC never exposes gold labels."""
    source_pin(root)
    # Executing this file directly otherwise shadows HuggingFace datasets with
    # this campaign's datasets.py. Keep official dependencies unambiguous.
    own=Path(__file__).resolve().parent
    sys.path[:]=[x for x in sys.path if Path(x or '.').resolve()!=own]
    sys.path.insert(0,str(root))
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        from automationbench.domains import get_domain_dataset
        from automationbench.runner import AutomationBenchEnv
        from automationbench.rubric import create_rubric, partial_credit, task_completed_correctly
        from automationbench.task_contract import task_contract_sha256
        rows={}
        catalog=[]
        for domain in DOMAINS:
            dataset=get_domain_dataset(domain)
            if len(dataset)!=100:raise RuntimeError(f"Official {domain} count differs from100")
            for index,row in enumerate(dataset):
                info=json.loads(row['info']) if isinstance(row['info'],str) else row['info']
                name=info.get('task_name') or str(index)
                item_id=f'{domain}:{name}'
                if item_id in rows:raise RuntimeError('Official public item id duplicated')
                rows[item_id]=row
                catalog.append({'id':item_id,'domain':domain,'prompt':row['prompt'],
                    'contract_sha256':task_contract_sha256(example_id=row.get('example_id',index),prompt=row['prompt'],info=info)})
        env=AutomationBenchEnv(dataset=dataset,rubric=create_rubric(),toolset='api',max_turns=MAX_STEPS)
    state=None
    for line in sys.stdin:
        request=json.loads(line)
        try:
            op=request['op']
            if op=='catalog':result={'items':catalog,'revision':REVISION,'public_only':True}
            elif op=='audit_labels':
                labels=[]
                for item in catalog:
                    row=rows[item['id']]
                    info=json.loads(row['info']) if isinstance(row['info'],str) else row['info']
                    labels.append({'id':item['id'],'question':'\n'.join(str(x.get('content','')) for x in row['prompt']),
                        'public_prompt':row['prompt'],'history':[], 'scorer_payload':{'assertions':info['assertions']}})
                path=Path(request['path']);path.write_text(json.dumps(labels,ensure_ascii=False)+'\n')
                result={'count':len(labels),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            elif op=='start':
                row=copy.deepcopy(rows[request['item_id']])
                with redirect_stdout(sys.stderr):
                    state=await env.setup_state({'info':row['info'],'prompt':row['prompt'],
                                                 'example_id':row.get('example_id')})
                result={'prompt':row['prompt'],'tools':[{'type':'function','function':{'name':t.name,
                    'description':t.description,'parameters':t.parameters}} for t in state['tool_defs']]}
            elif op=='tool':
                if state is None:raise RuntimeError('No official world started')
                if request['name'] not in {x.name for x in state['tool_defs']}:raise RuntimeError('Unadvertised official tool')
                args=env.update_tool_args(request['name'],request['args'],[],state)
                with redirect_stdout(sys.stderr):
                    message=await env.call_tool(request['name'],args,request['call_id'])
                result={'content':message.content}
            elif op=='finish':
                if state is None:raise RuntimeError('No official world started')
                payload={'world':state['world'].model_dump(mode='json'),'info':state['info'],
                         'initial_state':state['initial_state']}
                path=Path(request['path']);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('w') as stream:
                    stream.write(json.dumps(payload,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                result={'state_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                state=None
            elif op=='score_merged':
                # One official scoring pass after parent coverage/uniqueness.
                from automationbench.schema.world import WorldState
                records=[json.loads(x) for x in Path(request['path']).read_text().splitlines() if x.strip()]
                scored=[]
                for record in records:
                    path=Path(record['world_state_file'])
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['world_state_sha256']:
                        raise RuntimeError('Official final world snapshot changed')
                    raw=json.loads(path.read_text())
                    raw['world']=WorldState(**raw['world'])
                    with redirect_stdout(sys.stderr):
                        partial=partial_credit(raw)
                        strict=task_completed_correctly(raw)
                    scored.append({'item_id':record['item_id'],'partial_credit':partial,
                                   'score':strict,'assertion_results':raw['_assertion_results']})
                result={'scored':scored}
            elif op=='shutdown':
                print(json.dumps({'id':request['id'],'result':{'closed':True}}),flush=True);return
            else:raise ValueError('Unsupported official world operation')
            print(json.dumps({'id':request['id'],'result':result},ensure_ascii=False),flush=True)
        except Exception as exc:
            print(json.dumps({'id':request.get('id'),'error':{'type':type(exc).__name__,'message':str(exc)}}),flush=True)


class WorldRPC:
    """No provider, API keys, or hidden model inference lives in this child."""
    def __init__(self, root, log):
        self.root,self.log=Path(root).resolve(),Path(log)
        self.proc=None;self.seq=0;self.stderr=None

    async def __aenter__(self):
        self.log.parent.mkdir(parents=True,exist_ok=True)
        self.stderr=self.log.with_suffix('.stderr').open('a')
        env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN','SECRET'))}
        self.proc=await asyncio.create_subprocess_exec(str(self.root/'.venv/bin/python'),str(Path(__file__).resolve()),
            '--child','--root',str(self.root),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
            stderr=self.stderr,env=env,limit=32*1024*1024)
        return self

    async def call(self,op,**kwargs):
        from .runtime import append_jsonl, utc_now
        self.seq+=1
        request={'id':self.seq,'op':op,**kwargs}
        append_jsonl(self.log,{'timestamp':utc_now(),'direction':'request','message':request})
        self.proc.stdin.write((json.dumps(request)+'\n').encode());await self.proc.stdin.drain()
        line=await asyncio.wait_for(self.proc.stdout.readline(),timeout=120)
        if not line:raise RuntimeError('Official AutomationBench child exited; inspect stderr')
        response=json.loads(line)
        append_jsonl(self.log,{'timestamp':utc_now(),'direction':'response','message':response})
        if response.get('id')!=self.seq:raise RuntimeError('Official world RPC identity mismatch')
        if 'error' in response:raise RuntimeError(str(response['error']))
        return response['result']

    async def __aexit__(self,*args):
        if self.proc:
            if self.proc.returncode is None:
                try:await self.call('shutdown')
                except Exception:self.proc.terminate()
                try:await asyncio.wait_for(self.proc.wait(),timeout=10)
                except TimeoutError:self.proc.kill();await self.proc.wait()
            self.stderr.close()


async def prepare_automationbench(runs):
    """Live official world/tool feasibility; scorer remains behind integrity gate."""
    from .runtime import atomic_write, canonical
    runs=Path(runs);out=runs/'automationbench';out.mkdir(exist_ok=True)
    async with WorldRPC(runs/'upstream/AutomationBench',out/'feasibility-rpc.jsonl') as world:
        catalog=await world.call('catalog')
        items=catalog['items']
        if len(items)!=600 or len({x['id'] for x in items})!=600:raise RuntimeError('Official public600 count differs')
        atomic_write(out/'catalog.json',canonical(catalog)+'\n')
        audit_labels=await world.call('audit_labels',path=str(out/'audit-labels.json'))
        first=items[0]
        state=await world.call('start',item_id=first['id'])
        tool_probe=await world.call('tool',name='api_search',args={'query':'contacts','top_k':1},call_id='diagnostic-search')
        if not tool_probe.get('content'):raise RuntimeError('Official simulated tool returned no content')
        snapshot=out/'feasibility-initial-world.json'
        end=await world.call('finish',path=str(snapshot))
        # The official assertion scorer stays unexecuted until the independent
        # integrity gate opens after full raw collection. No self-test exception.
        result={'status':'official_world_and_tool_qualified_scorer_pending','public_tasks':600,'domains':list(DOMAINS),
                'first_task':first['id'],'tool_count':len(state['tools']),'official_tool_calls':1,'model_calls':0,
                'benchmark_score':None,'source_revision':REVISION,
                'catalog_sha256':hashlib.sha256((out/'catalog.json').read_bytes()).hexdigest(),
                'audit_labels_sha256':audit_labels['sha256']}
        atomic_write(out/'feasibility.json',canonical(result)+'\n')
        return result


def load_automationbench_audit_items(root):
    """Read official child-exported labels for audit; never pass them to a model."""
    root=Path(root)
    source_pin(root)
    out=root.parent.parent/'automationbench'
    manifest=json.loads((out/'feasibility.json').read_text())
    path=out/'audit-labels.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['audit_labels_sha256']:
        raise RuntimeError('Official audit labels sidecar hash changed')
    return json.loads(path.read_text())


async def run_automation_case(campaign, world, item, arm, worker, attempt=0):
    from .engine import HookedRealEngine
    from .retrieval import BM25Memory, BGEMemory
    from .security_agents import SEARCH_TOOL, TOOL_LINES, token_count
    from .runtime import canonical, utc_now
    root=campaign.runs/'automationbench'/arm/f'worker-{worker}'/(hashlib.sha256(item['id'].encode()).hexdigest()[:24]+f'-{uuid.uuid4().hex[:8]}')
    root.mkdir(parents=True,exist_ok=False)
    meta={'benchmark':'automationbench','arm':arm,'item_id':item['id'],'trial':0,'seed':20261101,
          'worker_id':worker,'shard':worker,'started_at':utc_now(),'binary_sha256':campaign.manifest['binary_sha256'],
          'dataset_revision':REVISION,'task_contract_sha256':item['contract_sha256']}
    started=await world.call('start',item_id=item['id'])
    messages=copy.deepcopy(started['prompt'])
    if not isinstance(messages,list):raise RuntimeError('Official prompt shape changed')
    system=next((x for x in messages if x.get('role')=='system'),None)
    if system is None:messages.insert(0,{'role':'system','content':TOOL_LINES[arm]})
    else:system['content']+='\n'+TOOL_LINES[arm]
    initial=copy.deepcopy(messages)
    bound=campaign.item_provider(meta,root,max_calls=MAX_STEPS,max_output=OUTPUT_BUDGET,context=CONTEXT_LIMIT)
    engine=None;hook_callback=None;history=[];memory=None;usage=[];cost=0;remaining=OUTPUT_BUDGET
    try:
        memtools=[]
        if arm=='vestige':
            hook_callback=bound.hook_callback(lambda:engine)
            if not hasattr(hook_callback,'cost') or not hasattr(hook_callback,'usage'):
                raise RuntimeError('Hook callback must expose metered cost and usage for pilot projection')
            engine=HookedRealEngine(Path(__file__).resolve().parents[2],root/'native',
                manifest_path=campaign.runs/'build-manifest.json',hook_model_callback=hook_callback)
            await asyncio.to_thread(engine.__enter__)
            if not engine.hooks:raise RuntimeError('Full Vestige hooks not active')
            context=await asyncio.to_thread(engine.begin_turn,'\n'.join(str(x.get('content','')) for x in initial))
            if context:messages.append({'role':'system','content':'\n'.join(context)})
            engine.hooks.pending_context.clear()
            memtools=[{'type':'function','function':{'name':t['name'],'description':t.get('description',''),
                'parameters':t['inputSchema']}} for t in engine.tools if t['name'] in {'smart_ingest','recall','memory','session_start','causal_walk'}]
        elif arm!='bare':memtools=[SEARCH_TOOL]
        memnames={t['function']['name'] for t in memtools}
        for step in range(MAX_STEPS):
            if engine and engine.hooks.pending_context:
                messages.append({'role':'system','content':'\n'.join(engine.hooks.pending_context)})
                engine.hooks.pending_context.clear()
            tools=started['tools']+memtools
            if token_count({'messages':messages,'tools':tools})>CONTEXT_LIMIT:raise RuntimeError('Automation context budget exhausted')
            result=await bound.chat(messages,tools=tools,max_tokens=min(1024,remaining),
                log=root/'api.jsonl',identity={**meta,'step':step,'engine_data_dir':engine.identity['data_dir'] if engine else None})
            if not result['model'].startswith(MODEL):raise RuntimeError('Automation provider model mismatch')
            cost+=result['cost'];usage.append(result['usage']);remaining-=result['usage']['completion_tokens']
            if result['finish_reason']=='length':raise RuntimeError('Automation output truncated')
            message={k:v for k,v in result['message'].items() if k in {'role','content','tool_calls'}}
            messages.append(message)
            calls=message.get('tool_calls') or []
            if not calls:
                if engine:
                    verdict=await asyncio.to_thread(engine.finish_turn,messages)
                    if verdict['blocked']:
                        messages.append({'role':'user','content':'\n'.join(verdict['reasons']+verdict['context'])})
                        continue
                break
            for call in calls:
                name=call['function']['name'];args=json.loads(call['function']['arguments'])
                if name in memnames:
                    if engine:output=await asyncio.to_thread(engine.call,name,args)
                    else:output=await asyncio.to_thread(memory.search,args['query'],min(10,max(1,args.get('k',5)))) if memory else []
                else:
                    if engine:
                        before=await asyncio.to_thread(engine.hooks.event,'PreToolUse',{'tool_name':name,'tool_input':args})
                        engine.hooks.pending_context.extend(before['context'])
                        if before['blocked']:
                            output={'blocked':True,'reasons':before['reasons']}
                            messages.append({'role':'tool','tool_call_id':call['id'],'content':canonical(output)})
                            continue
                    output=(await world.call('tool',name=name,args=args,call_id=call['id']))['content']
                    text=output if isinstance(output,str) else canonical(output)
                    history.append(text)
                    if engine:
                        await asyncio.to_thread(engine.call,'smart_ingest',{'content':text,'node_type':'note',
                            'tags':['history'],'source':'tool-output'})
                        after=await asyncio.to_thread(engine.hooks.event,'PostToolUse',{'tool_name':name,'tool_input':args,'tool_response':output})
                        engine.hooks.pending_context.extend(after['context'])
                    elif arm!='bare':memory=await asyncio.to_thread(BM25Memory if arm=='bm25' else BGEMemory,history)
                messages.append({'role':'tool','tool_call_id':call['id'],'content':output if isinstance(output,str) else canonical(output)})
            if remaining<=0:raise RuntimeError('Automation output budget exhausted')
        else:raise RuntimeError('Automation step budget exhausted')
        # bound includes all solver and product-hook model usage.
        snapshot=root/'official-final-world.json'
        final=await world.call('finish',path=str(snapshot))
        return {**meta,'status':'complete','finished_at':utc_now(),'model':MODEL,'prompt':initial,
            'transcript':messages,'response':message.get('content') or '', 'score':None,
            'usage':list(bound.usage),'cost':bound.cost,'group_id':item['domain'],'category':item['domain'],
            'world_state_file':str(snapshot),'world_state_sha256':final['state_sha256'],
            'engine':engine.identity if engine else None,
            'model_counters':{'call_limit':MAX_STEPS,'output_limit':OUTPUT_BUDGET,'context_limit':CONTEXT_LIMIT,
                              'remaining_calls':bound.remaining_calls,'remaining_output':bound.remaining_output}}
    finally:
        if engine:await asyncio.to_thread(engine.__exit__,None,None,None)


async def run_automationbench(campaign):
    from .runtime import Checkpoint, BudgetExceeded, atomic_write, canonical, record_key, shard_id, strict_merge, write_shard_map
    root=campaign.runs/'automationbench';root.mkdir(exist_ok=True)
    if not (root/'feasibility.json').exists():await prepare_automationbench(campaign.runs)
    catalog=json.loads((root/'catalog.json').read_text());items=catalog['items']
    qualification=campaign.runs/'QUALIFICATION.json'
    qualified=json.loads(qualification.read_text()) if qualification.exists() else {}
    if qualified.get('status')!='qualified' or qualified.get('vestige_arm')!='all_applicable_hooks' or not callable(getattr(campaign,'item_provider',None)):
        status={'status':'unfinished','reason':'full_applicable_hooks_qualification_or_nano_callback_missing',
                'n':0,'planned_tasks':600,'planned_rollouts':2400,'source_revision':REVISION,'model':MODEL,'api_calls':0}
        atomic_write(root/'PENDING.json',canonical(status)+'\n')
        from .integrity import audit,require,IntegrityPending
        report=await asyncio.to_thread(audit,Path(__file__).resolve().parents[2],campaign.runs)
        try:require(report,'automationbench','report')
        except IntegrityPending:return status
        atomic_write(root/'RESULTS.md','# AutomationBench public results\n\nUnfinished: all applicable Vestige hooks and their shared nano callback require qualification.\nNo scored rolls or paid calls were started by this adapter. The official Python3.13 world/tool feasibility is available in feasibility.json; scoring remains gated.\n\nAll four arms: mean unavailable; CI unavailable; n=0/600; one planned trial; model gpt-5.4-nano; APIcost $0.\n\nThe600publictasks acrosssixdomains are distinct from the privateofficialleaderboard. No simple-domain or substitute tasks are scored.\n')
        return status
    write_shard_map(root/'shards.json',[x['id'] for x in items],workers=WORKERS,seed=20261101)
    arms=('bare','bm25','embedding','vestige')
    identity={'revision':REVISION,'catalog_sha256':hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest(),
              'model':MODEL,'binary':campaign.manifest['binary_sha256'],'hooks':qualified,
              'limits':[MAX_STEPS,OUTPUT_BUDGET,CONTEXT_LIMIT],'toolset':'api','trials':1,'workers_per_arm':WORKERS}
    atomic_write(root/'identity.json',canonical(identity)+'\n')
    cps={a:[Checkpoint(root/a/f'worker-{k}.jsonl',{**identity,'arm':a}) for k in range(WORKERS)] for a in arms}
    for arm in arms:campaign.progress['automationbench',arm]={'benchmark':'automationbench','arm':arm,'done':sum(len(cp.done) for cp in cps[arm]),'total':600,'errors':0}
    atomic_write(root/'full-set.json',canonical({'scope':'full_official_set','ids':[x['id'] for x in items],'count':600})+'\n')
    count=getattr(campaign,'preflight_count',0)
    selected=random.Random(20261101).sample(items,min(count,len(items))) if count else items
    if count:atomic_write(root/'preflight-selection.json',canonical({'scope':'unscored_integrity_only','seed':20261101,'ids':[x['id'] for x in selected],'full_set_count':600})+'\n')
    if not (root/'projection.json').exists():
        calibration=[]
        for arm in arms:
            k=shard_id(selected[0]['id'],workers=WORKERS);cp=cps[arm][k]
            if cp.contains({'item_id':selected[0]['id'],'trial':0}):row=cp.done[record_key({'item_id':selected[0]['id'],'trial':0})]
            else:
                try:
                    async with WorldRPC(campaign.runs/'upstream/AutomationBench',root/arm/'pilot-world.jsonl') as world:
                        row=await run_automation_case(campaign,world,selected[0],arm,k)
                except BudgetExceeded:
                    coverage={a:{'completed':[x['id'] for x in items if cps[a][shard_id(x['id'],workers=WORKERS)].contains({'item_id':x['id'],'trial':0})],
                                 'unfinished':[x['id'] for x in items if not cps[a][shard_id(x['id'],workers=WORKERS)].contains({'item_id':x['id'],'trial':0})]} for a in arms}
                    atomic_write(root/'coverage.json',canonical(coverage)+'\n')
                    return {'status':'actual_budget_stop','phase':'one_item_pilot','coverage_file':'coverage.json'}
                except Exception as exc:
                    cp.append_failure({'item_id':selected[0]['id'],'trial':0,'phase':'one_item_pilot','error':str(exc)[:1000]})
                    raise
                cp.append(row);campaign.progress['automationbench',arm]['done']+=1
            calibration.append({'arm':arm,'cost':row['cost']})
        projection={'calibration':calibration,'projected_total':sum(x['cost'] for x in calibration)*600,'remaining':campaign.ledger.snapshot()['remaining']}
        atomic_write(root/'projection.json',canonical(projection)+'\n');print('automationbench PROJECTED_COST',canonical(projection),flush=True)
    async def worker(arm,k):
        cp=cps[arm][k];failed=[]
        async with WorldRPC(campaign.runs/'upstream/AutomationBench',root/arm/f'world-worker-{k}.jsonl') as world:
            async def execute(item,attempt):
                if cp.contains({'item_id':item['id'],'trial':0}):return True
                try:
                    row=await run_automation_case(campaign,world,item,arm,k,attempt);cp.append(row)
                    campaign.progress['automationbench',arm]['done']+=1;return True
                except BudgetExceeded:return False
                except Exception as exc:
                    cp.append_failure({'item_id':item['id'],'trial':0,'attempt':attempt,'error':str(exc)[:1000]})
                    campaign.progress['automationbench',arm]['errors']+=1
                    if attempt==0:failed.append(item)
                    return True
            for item in selected:
                if shard_id(item['id'],workers=WORKERS)==k and not await execute(item,0):return True
            for item in failed:
                if not await execute(item,1):return True
        return False
    stops=await asyncio.gather(*(worker(arm,k) for arm in arms for k in range(WORKERS)))
    coverage={arm:{'completed':[x['id'] for x in items if cps[arm][shard_id(x['id'],workers=WORKERS)].contains({'item_id':x['id'],'trial':0})],
                   'unfinished':[x['id'] for x in items if not cps[arm][shard_id(x['id'],workers=WORKERS)].contains({'item_id':x['id'],'trial':0})]} for arm in arms}
    atomic_write(root/'coverage.json',canonical(coverage)+'\n')
    if count:
        status={'status':'unscored_preflight_collected','selected_count':len(selected),'full_set_count':600,
                'actual_budget_stop':any(stops),'coverage_file':'coverage.json'}
        atomic_write(root/'preflight-status.json',canonical(status)+'\n');return status
    expected=[record_key({'item_id':x['id'],'trial':0}) for x in items]
    summaries=[]
    for arm in arms:
        merged_path=root/arm/'merged.jsonl'
        try:merged=strict_merge([c.path for c in cps[arm]],expected,merged_path)
        except ValueError as exc:summaries.append({'arm':arm,'status':'unfinished','reason':str(exc)});continue
        scored_path=root/arm/'scored.json'
        if not scored_path.exists():
            from .integrity import audit,require
            require(await asyncio.to_thread(audit,Path(__file__).resolve().parents[2],campaign.runs),'automationbench','score')
            async with WorldRPC(campaign.runs/'upstream/AutomationBench',root/arm/'official-scorer.jsonl') as scorer:
                scores=await scorer.call('score_merged',path=str(merged_path))
            atomic_write(scored_path,canonical(scores)+'\n')
        scores=json.loads(scored_path.read_text())['scored']
        if {x['item_id'] for x in scores}!={x['item_id'] for x in merged}:raise RuntimeError('Official scorer coverage mismatch')
        import numpy as np
        values=np.array([x['score'] for x in scores]);rng=np.random.default_rng(20261101)
        boot=[float(values[rng.integers(0,len(values),len(values))].mean()) for _ in range(10000)]
        summaries.append({'arm':arm,'status':'complete','mean':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'n':600,'trials':1,'model':MODEL,'source_revision':REVISION,'cost':sum(x['cost'] for x in merged),
            'binary_sha256':campaign.manifest['binary_sha256']})
    atomic_write(root/'summary.json',canonical(summaries)+'\n')
    lines=['# AutomationBench public results','','| Arm | Mean strict pass rate | 95% CI | n | Model | Cost |','|---|---|---|---|---|---|']
    for s in summaries:lines.append(f"| {s['arm']} | {s.get('mean','unfinished')} | {s.get('ci95','unavailable')} | {s.get('n',0)}/600 | {MODEL} | {s.get('cost','ledger')} |")
    lines+=['',f'Official public source `{REVISION}`;600tasks/6domains,one trial,seed20261101,5 workers per arm.',
        'The official partial_credit and strict task_completed_correctly functions score unchanged final worlds once after exact merged coverage. Private leaderboard tasks and simple tasks are excluded.',
        'Catalog, task contract and binary hashes are in identity.json and raw checkpoints; all API calls remain in the parent global ledger.']
    from .integrity import audit,require
    require(await asyncio.to_thread(audit,Path(__file__).resolve().parents[2],campaign.runs),'automationbench','report')
    atomic_write(root/'RESULTS.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if not args.child:raise SystemExit('Only official-world child mode is supported directly')
    asyncio.run(child_main(args.root.resolve()))

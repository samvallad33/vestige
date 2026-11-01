"""Actual @vestige/init lifecycle client adapted to the isolated MCP transport.

No memory lookup is implemented here. Unchanged product memory.cjs calls the
worker's actual native session_start(memory_hook=...) through its stdio bridge.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import select
import time

from benchmarks.real_engine.engine import HookedRealEngine
from benchmarks.real_engine.runtime import append_jsonl,canonical,utc_now

REPO=Path(__file__).resolve().parents[1]

class NativeSetupEngine(HookedRealEngine):
    def prepare_product_setup(self):
        if hasattr(self,'product_config'):return
        home=Path(self.hooks.env['HOME']);base=home/'.vestige/memory'
        pins={}
        for relative in ('hooks/memory.cjs','lib/memory-triggers.cjs'):
            source=REPO/'packages/vestige-init'/relative;target=base/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            pins[str(source.relative_to(REPO))]=hashlib.sha256(source.read_bytes()).hexdigest()
        self.product_config=base/'config.json';triggers=base/'triggers.json';node=shutil.which('node')
        if not node:raise RuntimeError('Node required by the actual product lifecycle client')
        self.product_node=node
        raw=subprocess.check_output([node,'-e','process.stdout.write(JSON.stringify(require(process.argv[1]).defaults))',str(base/'lib/memory-triggers.cjs')],env=self.hooks.env,text=True)
        triggers.write_text(raw+'\n');self.product_config.write_text(canonical({'version':'6.0.0','binary':str(home/'bin/vestige-mcp'),'binaryArgs':[],'triggers':str(triggers)})+'\n')
        doc=REPO/'docs/integrations/codex-intelligent-memory.md';self.product_protocol=re.findall(r'```markdown\n(.*?)\n```',doc.read_text(),re.S)[0]
        self.product_log=self.run_dir/'product-memory-hooks.jsonl'
        self.product_session=hashlib.sha256(str(self.run_dir).encode()).hexdigest()
        append_jsonl(self.product_log,{'event':'installed','timestamp':utc_now(),'source_pins':pins,'protocol_source':str(doc),'protocol_sha256':hashlib.sha256(self.product_protocol.encode()).hexdigest(),'source_wiring':str(REPO/'packages/vestige-init/lib/vestige-memory-setup.js'),'registrations':{'SessionStart':'session','UserPromptSubmit':'prompt','Stop':'turn','PreCompact':'compact','SessionEnd':'end'},'config':str(self.product_config),'worker_home':str(home),'engine_data_dir':str(self.run_dir/'strata')})

    def product_event(self,event,prompt='',actor_session=None):
        self.prepare_product_setup()
        client=Path(self.product_config).parent/'hooks/memory.cjs'
        if event in {'turn','end'}:
            # Official history questions contain newlines; the unchanged
            # explicitFact grammar cannot enqueue a fact from this prompt.
            # No assistant-derived facts or pending synthetic saves exist.
            if event=='end':self.hooks.native_call('session_start',{'memory_hook':{'limit':1}})
            append_jsonl(self.product_log,{'event':event,'timestamp':utc_now(),'injected_context':'','queued_facts':0,'reason':'no explicit fact queued by the product declaration grammar'})
            return []
        if not hasattr(self,'product_formatter'):
            script="""const readline=require('readline');const m=require(process.argv[1]);const t=require(process.argv[2]);const rl=readline.createInterface({input:process.stdin});let current;
console.log(JSON.stringify({ready:true}));
rl.on('line',line=>{const x=JSON.parse(line);if(x.kind==='begin'){current=x;const cause=t.trigger(x.prompt,t.defaults);console.log(JSON.stringify({arguments:{memory_hook:{prompt:x.prompt,limit:x.event==='session'?8:cause.limit,trigger:cause,session:x.session,host:'codex',event:x.event,source:x.event==='session'?'startup':''}}}));}else{console.log(JSON.stringify(m.output('codex',current.event,m.formatContext(x.packet))));}});"""
            self.product_formatter=subprocess.Popen([self.product_node,'-e',script,str(client),str(client.parent.parent/'lib/memory-triggers.cjs')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=self.hooks.env,cwd=self.cwd)
            if not select.select([self.product_formatter.stdout],[],[],5)[0] or not json.loads(self.product_formatter.stdout.readline()).get('ready'):
                raise RuntimeError('unchanged product formatter failed startup')
        started=time.monotonic();deadline=started+0.190
        process=self.product_formatter
        def exchange(value):
            process.stdin.write(canonical(value)+'\n');process.stdin.flush()
            if not select.select([process.stdout],[],[],max(0,deadline-time.monotonic()))[0]:raise TimeoutError('product memory190ms deadline')
            return json.loads(process.stdout.readline())
        request={'kind':'begin','event':event,'prompt':prompt,'session':actor_session or self.product_session}
        context='';value={};error=None
        try:
            native=exchange(request)['arguments']
            # No host memory selection: the actual native packet is the only
            # source. Reuse the required live child instead of another process.
            packet=self.hooks.native_call('session_start',native)
            value=exchange({'kind':'packet','packet':packet})
            context=value.get('hookSpecificOutput',{}).get('additionalContext','')
        except TimeoutError as exc:error=str(exc)
        append_jsonl(self.product_log,{'event':event,'timestamp':utc_now(),'input':request,'stdout':canonical(value),'stderr':error or '',
            'injected_context':context,'native_client_source':str(client),'transport':'persistent-original-JS-functions-to-existing-native-stdio','elapsed_ms':(time.monotonic()-started)*1000})
        return [context] if context else []

    def begin_turn(self,prompt):
        # History is already admitted when the agent session begins. The
        # product's SessionStart and prompt hooks now see this item's store.
        context=self.product_event('session',prompt)+self.product_event('prompt',prompt)
        self.product_prompt=prompt
        return [self.product_protocol]+context+super().begin_turn(prompt)

    def finish_turn(self,messages):
        verdict=super().finish_turn(messages)
        self.product_event('turn',getattr(self,'product_prompt',''))
        return verdict

    def close(self):
        try:
            if hasattr(self,'product_config') and not getattr(self,'product_ended',False):
                self.product_ended=True
                if self.client and self.client._proc and self.client._proc.poll() is None:
                    self.product_event('end',getattr(self,'product_prompt',''))
        finally:
            process=getattr(self,'product_formatter',None)
            if process is not None:
                process.terminate()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
            super().close()


from benchmarks.real_engine.full_hook_model import FullHookBudgetedProvider

class NativeSetupBudgetedProvider(FullHookBudgetedProvider):
    async def hook_request(self,payload,callback):
        import asyncio
        engine=callback.engine_getter() if callback.engine_getter else callback.engine
        if engine is None:raise RuntimeError('hook agent has no live worker engine')
        messages=list(payload['messages'])
        prompt='\n'.join(m.get('content','') for m in messages if isinstance(m.get('content'),str))
        actor=engine.product_session+'/hook/'+str(payload.get('hook','agent'))
        context=await asyncio.to_thread(engine.product_event,'session',prompt,actor)
        # The real installed client protocol and actual native packet precede
        # the unchanged source hook prompt; no invented ids or source records.
        payload={**payload,'messages':[{'role':'system','content':engine.product_protocol}]+[{'role':'user','content':c} for c in context]+messages}
        return await super().hook_request(payload,callback)

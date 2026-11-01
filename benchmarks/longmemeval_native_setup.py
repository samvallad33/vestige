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

    def product_event(self,event,prompt=''):
        self.prepare_product_setup()
        client=Path(self.product_config).parent/'hooks/memory.cjs'
        # Invoke the product's exported lifecycle handler; its original190ms
        # native work budget stays intact. Only transport/client payload adapts.
        script="const fs=require('fs');const m=require(process.argv[1]);m.run(JSON.parse(fs.readFileSync(0,'utf8'))).then(v=>process.stdout.write(JSON.stringify(v))).catch(e=>{process.stderr.write(String(e));process.exitCode=1})"
        payload={'event':event,'client':'codex','configFile':str(self.product_config),'input':{'session_id':self.product_session,'prompt':prompt,'cwd':str(self.cwd),'source':'startup' if event=='session' else ''}}
        result=subprocess.run([self.product_node,'-e',script,str(client)],input=canonical(payload),capture_output=True,text=True,env=self.hooks.env,cwd=self.cwd,timeout=5)
        if result.returncode:raise RuntimeError('actual product memory client failed: '+result.stderr[:400])
        value=json.loads(result.stdout or '{}');context=value.get('hookSpecificOutput',{}).get('additionalContext','')
        append_jsonl(self.product_log,{'event':event,'timestamp':utc_now(),'input':payload['input'],'stdout':result.stdout,'stderr':result.stderr,'exit_code':result.returncode,'injected_context':context,'native_client_source':str(client)})
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

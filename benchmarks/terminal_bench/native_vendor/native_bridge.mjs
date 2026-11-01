// AgentDojo custom-host adapter. Native enforcement and ledger verification are
// shared with the qualified Vestige host; this is not an OpenCode UI trajectory.
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, appendFileSync, existsSync, realpathSync, readdirSync, copyFileSync } from 'node:fs';
import { join, resolve, dirname, relative } from 'node:path';
import { homedir } from 'node:os';
import { createInterface } from 'node:readline';
import { pathToFileURL, fileURLToPath } from 'node:url';

const HOST = fileURLToPath(new URL('./host', import.meta.url));
const BINARY = '/Users/entity002/.vestige/bin/v6.2.0-d4598ab3/vestige-mcp';
const GATE = '/Users/entity002/.operator/gate/operator-gate.py';
const { KernelBroker, ROUTE_FIELD, argsDigest } = await import(pathToFileURL(join(HOST, 'kernel-broker.mjs')));
const { auditWriter, checkReflexDecision } = await import(pathToFileURL(join(HOST, 'runtime.mjs')));
const EXPECTED = ['backfill','causal_walk','codebase','dedup','forgotten_lesson','gather','ghostlink','intention',
  'maintain','memory','memory_status','precedent','project','recall','receipt','selftest','session_start','smart_ingest','suppress'];
const fail = message => new Error('FAIL_CLOSED_INTERDICTION: ' + message);
const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
const cleanEnv = () => {
  const env = { ...process.env, OPERATOR_AGENT_SESSION: '1' };
  for (const key of Object.keys(env)) if (/(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN)$/.test(key)) delete env[key];
  return env;
};

class NativeWire {
  constructor(store, operatorHome, profileEnv={}) {
    this.next = 0; this.waiting = new Map(); this.buffer = ''; this.error = null;
    this.child = spawn(BINARY, ['--data-dir', store], { cwd:join(resolve(store,'..'),'task-files'), stdio: ['pipe','pipe','pipe'], env: {
      ...cleanEnv(), ...profileEnv, OPERATOR_HOME:operatorHome, VESTIGE_RETENTION_TARGET:'0.8', VESTIGE_BACKFILL_AUTOFIRE:'1',
      VESTIGE_HTTP_ENABLED:'0', VESTIGE_DASHBOARD_ENABLED:'false', VESTIGE_AUTOPILOT_ENABLED:'0' } });
    this.child.stderr.on('data', bytes => appendFileSync(join(resolve(store, '..'), 'native.stderr'), bytes));
    this.child.on('error', error => this.break(fail('native spawn: ' + error.message)));
    this.child.on('exit', () => this.break(fail('native transport dropped')));
    this.child.stdin.on('error', error => this.break(fail('native write: ' + error.message)));
    this.child.stdout.on('data', chunk => {
      this.buffer += chunk.toString();
      if (this.buffer.length > 8*1024*1024) return this.break(fail('oversized native frame'));
      for (;;) {
        const end = this.buffer.indexOf('\n'); if (end < 0) break;
        const line = this.buffer.slice(0,end); this.buffer = this.buffer.slice(end+1);
        if (!line.trim()) continue;
        let response; try { response = JSON.parse(line); } catch { this.break(fail('malformed native JSON')); break; }
        const pending = this.waiting.get(response.id);
        if (!pending) { if ('id' in response) this.break(fail('unmatched native RPC ID')); continue; }
        clearTimeout(pending.timer); this.waiting.delete(response.id);
        if (response.error || !Object.hasOwn(response,'result')) pending.reject(fail('native RPC rejected'));
        else pending.resolve(response.result);
      }
    });
  }
  break(error) { this.error ||= error; for (const p of this.waiting.values()) { clearTimeout(p.timer); p.reject(this.error); } this.waiting.clear(); }
  rpc(method,params={}) {
    if (this.error) return Promise.reject(this.error);
    const id = ++this.next;
    return new Promise((resolve,reject) => {
      const timer = setTimeout(() => { this.break(fail('native RPC deadline')); this.child.kill(); }, 60000);
      this.waiting.set(id,{resolve,reject,timer});
      this.child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');
    });
  }
  async initialize() {
    this.hello = await this.rpc('initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'vestige-benchmark-custom',version:'1'}});
    if (this.hello.serverInfo?.name !== 'vestige' || this.hello.serverInfo?.version !== '6.2.0') throw fail('native identity mismatch');
    this.child.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n');
    const list = await this.rpc('tools/list'); this.catalog = list.tools;
    if (JSON.stringify(this.catalog?.map(t=>t.name).sort()) !== JSON.stringify(EXPECTED)) throw fail('native catalog mismatch');
    return this.catalog;
  }
  async reflex(args) {
    const result = await this.rpc('tools/call',{name:'reflex',arguments:{host:'benchmark-custom',format:'plain',...args}});
    if (result.isError) throw fail('native Reflex error');
    let out = result.structuredContent;
    if (out === undefined) {
      const text = result.content?.find(c=>c.type==='text')?.text;
      if (!text?.trim()) throw fail('empty native Reflex envelope');
      try { out = JSON.parse(text); } catch { throw fail('malformed native Reflex body'); }
    }
    if (!out || typeof out !== 'object' || Array.isArray(out)) throw fail('invalid native Reflex body');
    return out;
  }
  close() { this.child.stdin.end(); this.child.kill(); }
}

class Stack {
  async init(root) {
    if(this.broker)throw fail('duplicate bridge initialization');
    this.root = resolve(root); mkdirSync(this.root,{recursive:true,mode:0o700});
    this.files=join(this.root,'task-files');mkdirSync(this.files,{mode:0o700});
    this.audit = auditWriter(this.root); this.pending = new Map(); this.sessions = new Set(); this.lastWrite = new Map();
    this.seenCalls=new Set();this.parents=new Map();
    this.operatorHome=join(this.root,'operator');mkdirSync(this.operatorHome,{mode:0o700});
    const ownerHome=join(homedir(),'.operator'),policyPaths=[];
    if(existsSync(join(ownerHome,'DISABLED')))throw fail('owner Operator is disabled');
    policyPaths.push(join(ownerHome,'DISABLED'));
    for(const name of ['commitments.json','state/commitments-validated.json','mode','STRICT']) {
      const source=join(ownerHome,name),target=join(this.operatorHome,name);
      if(!existsSync(source)) {
        if(name.startsWith('commitments')||name.startsWith('state/'))throw fail('missing owner policy '+name);
        policyPaths.push(source,target);
        continue;
      }
      mkdirSync(dirname(target),{recursive:true,mode:0o700});copyFileSync(source,target);
      policyPaths.push(source,target);
    }
    this.profileEnv={};this.profile={treatment:'production-identical',rule_override_count:0};
    const profilePath=process.env.VESTIGE_OPERATOR_PROFILE_JSON;
    if(profilePath){
      const profile=JSON.parse(readFileSync(profilePath,'utf8'));
      if(profile.schema!=='vestige-isolated-all-rules-profile-v1'||profile.rule_count!==118||profile.native_gate.sha256!==sha(GATE))throw fail('unqualified Operator profile identity');
      const target=join(this.operatorHome,'commitments.json'),config=JSON.parse(readFileSync(target,'utf8'));
      const disabled=config.disabled_rules??[];
      if(!Array.isArray(disabled)||disabled.length)throw fail('owner disabled rules require re-audit');
      const overrides=profile.profile_delta.mode_overrides;
      if(Object.keys(overrides).length!==118||Object.values(overrides).some(v=>v!=='enforce'))throw fail('invalid strengthening profile');
      const allowedEnv=new Set(['OPERATOR_GATE_MODE','OPERATOR_STRICT','OPERATOR_TEST_INTEGRITY','OPERATOR_TURNOVER','OPERATOR_AGENT_SESSION','OPERATOR_BRAKE_AT']);
      if(Object.keys(profile.profile_delta.env).some(k=>!allowedEnv.has(k)))throw fail('unadvertised profile environment');
      config.mode_overrides={...(config.mode_overrides||{}),...overrides};
      writeFileSync(target,JSON.stringify(config,null,2)+'\n');
      this.profileEnv={...profile.profile_delta.env};
      this.profile={treatment:'isolated-all-rules',rule_override_count:118,promoted_from_shadow:profile.profile_delta.promoted_from_shadow,
        profile_path:profilePath,profile_sha256:sha(profilePath),source_policy_sha256:sha(join(ownerHome,'commitments.json')),copy_policy_sha256:sha(target)};
      policyPaths.push(profilePath);
    }
    writeFileSync(join(this.root,'operator-profile.json'),JSON.stringify(this.profile,null,2)+'\n');
    policyPaths.push(join(this.root,'operator-profile.json'));
    const mode=this.profileEnv.OPERATOR_GATE_MODE||process.env.OPERATOR_GATE_MODE||(existsSync(join(this.operatorHome,'mode'))?readFileSync(join(this.operatorHome,'mode'),'utf8').trim():'');
    if(/^(off|disabled|observe|shadow)$/i.test(mode))throw fail('Operator enforcement mode is disabled');
    const helpers=readdirSync(dirname(GATE)).filter(x=>x.endsWith('.py')).map(x=>join(dirname(GATE),x));
    const paths = [BINARY,...helpers,...policyPaths,...['kernel-broker.mjs','runtime.mjs','audit-window.mjs','audit-recovery.mjs'].map(x=>join(HOST,x))];
    this.pins = Object.fromEntries(paths.map(path=>[path,existsSync(path)?sha(path):null]));
    this.check();
    this.broker = new KernelBroker({native:new NativeWire(join(this.root,'strata'),this.operatorHome,this.profileEnv),dataDir:join(this.root,'strata'),audit:this.audit});
    await this.broker.initialize();
    writeFileSync(join(this.root,'native-identities.json'),JSON.stringify({host:'benchmark-custom',pins:this.pins,
      enforcement:{mode,strict:this.profileEnv.OPERATOR_STRICT??process.env.OPERATOR_STRICT??null,mode_env:this.profileEnv.OPERATOR_GATE_MODE??process.env.OPERATOR_GATE_MODE??null},profile:this.profile,
      tools:this.broker.native.catalog},null,2));
    this.check(); return {tools:this.broker.native.catalog,pins:this.pins,host:'benchmark-custom'};
  }
  check() { for (const [path,value] of Object.entries(this.pins)) if ((existsSync(path)?sha(path):null)!==value) throw fail('runtime identity changed: '+path); }
  async fire(event,sid,fields={}) {
    this.check(); const result=await this.broker.hook({event,host:'benchmark-custom',format:'plain',session:sid,cwd:this.files,...fields});
    checkReflexDecision(result,event); this.check(); return result;
  }
  operator(args,payload) {
    this.check();
    const result=spawnSync('python3',[GATE,...args],{cwd:this.files,input:JSON.stringify(payload),encoding:'utf8',timeout:60000,maxBuffer:2*1024*1024,env:{...cleanEnv(),...this.profileEnv,OPERATOR_HOME:this.operatorHome}});
    this.check();
    if (result.error || ![0,2].includes(result.status)) throw fail('Operator unavailable: '+(result.error?.message||result.stderr||result.status));
    return result;
  }
  receipt(sid,call) {
    const day=new Date().toISOString().slice(0,10),path=join(this.operatorHome,'receipts',day+'.jsonl');
    const rows=readFileSync(path,'utf8').trim().split('\n').map(l=>JSON.parse(l));
    const row=rows.findLast(r=>r.session===sid&&r.call===call);
    if (!row?.digest) throw fail('missing matching Operator receipt'); return row;
  }
  async start(sid,parent=null) {
    if (!sid || this.sessions.has(sid)) throw fail('missing or duplicate session');
    if (parent && !this.sessions.has(parent)) throw fail('unknown subagent parent');
    const laws=this.operator(['laws','--session-start'],{session_id:sid,cwd:this.files,hook_event_name:'SessionStart'});
    if (laws.status!==0) throw fail('Operator startup refused');
    await this.fire('SessionStart',sid); this.sessions.add(sid);
    this.parents.set(sid,parent);
    if (parent) await this.fire('SubagentStart',parent,{agent:sid});
    const context=await this.native(sid,'session_start',{include_intentions:true,include_status:true,context:{},
      memory_hook:{session:sid,host:'benchmark-custom',event:'session',source:'startup',prompt:'',limit:8}},'controller-'+randomUUID(),'controller');
    return {laws:JSON.parse(laws.stdout),memory_hook:context};
  }
  async prompt(sid,text) {
    if(!this.sessions.has(sid))throw fail('prompt for inactive session');
    await this.fire('UserPromptSubmit',sid);
    return this.native(sid,'session_start',{include_intentions:true,include_status:true,context:{},
      memory_hook:{session:sid,host:'benchmark-custom',event:'prompt',source:'user',prompt:text,limit:4}},'controller-'+randomUUID(),'controller');
  }
  async before(sid,call,tool,args,native=false) {
    if (!this.sessions.has(sid) || !call || this.seenCalls.has(call)) throw fail('missing or replayed call identity');
    this.seenCalls.add(call);
    // AgentDojo structured functions use the real MCP-function effect envelope
    // understood by Operator. No shell representation is invented.
    const canonical=native?'mcp__vestige__'+tool:
      ['Bash','Read','Write','Edit','Glob','Grep','apply_patch'].includes(tool)?tool:'mcp__benchmark-custom__'+tool;
    const payload={hook_event_name:'PreToolUse',session_id:sid,tool_use_id:call,call_id:call,cwd:this.files,
      tool_name:canonical,tool_input:args};
    const result=this.operator(['hook','--source','benchmark-custom'],payload),receipt=this.receipt(sid,call);
    if (result.status===2) {
      if (receipt.decision==='PASS') throw fail('Operator deny/receipt mismatch');
      this.audit('operator_blocked',{session:sid,call,tool,args_digest:argsDigest(args),receipt_digest:receipt.digest});
      return {allowed:false,reason:'BLOCKED_BY_OPERATOR',receipt};
    }
    if (receipt.decision!=='PASS') throw fail('Operator PASS missing');
    const fields={event:'PreToolUse',host:'benchmark-custom',format:'plain',session:sid,cwd:this.files,call,
      tool:payload.tool_name,command:args.command||'',file:args.file_path||args.filePath||''};
    let token;
    if (native) token=this.broker.admit({tool,args,fields});
    else await this.fire('PreToolUse',sid,fields);
    this.pending.set(call,{sid,tool,canonical,args:structuredClone(args),digest:argsDigest(args),token,native});
    this.audit('operator_allowed',{session:sid,call,tool,args_digest:argsDigest(args),receipt_digest:receipt.digest});
    return {allowed:true,receipt};
  }
  async after(sid,call,tool,args,result,error=null) {
    const p=this.pending.get(call);
    if (!p || p.sid!==sid || p.tool!==tool || p.digest!==argsDigest(args)) throw fail('post-call identity/digest mismatch');
    if (p.native) this.broker.finish(p.token);
    await this.fire(error?'PostToolUseFailure':'PostToolUse',sid,{call,tool:p.canonical,
      command:p.args.command||'',error:error||''});
    this.pending.delete(call);
    this.audit('effect_completed',{session:sid,call,tool,result_digest:argsDigest(result),failed:!!error});
  }
  safeNative(name,args,sid,source) {
    if (!EXPECTED.includes(name)) throw fail('unknown native tool');
    const schema=this.broker.native.catalog.find(t=>t.name===name).inputSchema;
    if(source!=='controller'&&Object.keys(args).some(k=>!Object.hasOwn(schema.properties||{},k)))
      throw fail('unadvertised native argument');
    // All catalog tools are loaded. Native modes with their own ungated nested
    // subprocess/global effects are explicitly unsupported in this adapter.
    if (name==='gather') return {allowed:false,reason:'OWNER_APPROVAL_REQUIRED: external gather is outside the disposable task environment'};
    if(name==='ghostlink'&&['heal','harden'].includes(args.mode))
      return {allowed:false,reason:'UNSUPPORTED_NESTED_EFFECT_SURFACE: ghostlink '+args.mode};
    if(name==='codebase'&&args.action==='ingest_repo')
      return {allowed:false,reason:'UNSUPPORTED_NESTED_EFFECT_SURFACE: native git subprocesses are not independently gated'};
    if(name==='maintain'&&args.allowAnyPath===true)throw fail('native path override is forbidden');
    const inside=value=>{
      const target=resolve(this.files,value),base=realpathSync(this.files);let parent=target;
      while(!existsSync(parent)){const next=dirname(parent);if(next===parent)throw fail('unresolvable native path');parent=next;}
      const actual=resolve(realpathSync(parent),relative(parent,target));
      if(actual!==base&&!actual.startsWith(base+'/'))throw fail('native file target escapes task-files');
      return target;
    };
    if(name==='project'){
      args.root=inside(args.root||this.files);
      if(args.path)args.path=inside(resolve(args.root,args.path));
    }
    if(name==='codebase')args.repoPath=inside(args.repoPath||this.files);
    if(name==='session_start'&&args.context){
      if(args.context.repo_path)args.context.repo_path=inside(args.context.repo_path);
      if(args.context.repoPath)args.context.repoPath=inside(args.context.repoPath);
    }
    if(name==='session_start'&&source!=='controller'&&args.memory_hook&&
      Object.keys(args.memory_hook).some(k=>!['limit','prompt','trigger'].includes(k)))throw fail('model supplied private lifecycle fields');
    if(name==='precedent'){args.session=sid;args.cwd=this.files;args.source='benchmark-custom';}
    if(name==='causal_walk'&&(args.start_points||[]).some(s=>s.kind!=='logged_write'))
      return {allowed:false,reason:'UNSUPPORTED_FILESYSTEM_DISCOVERY: this isolated track uses recorded node starts'};
    const scan=(value,key='')=>{
      if(Array.isArray(value)&&['files','changed_files'].includes(key))for(const path of value)inside(String(path).split('#')[0]);
      else if (value&&typeof value==='object') for(const [k,v] of Object.entries(value)) scan(v,k);
      else if (typeof value==='string' && /^(root|repoPath|repo_path|repo|file|file_path|path|outputPath|out_dir|destination|targetPath|current_file)$/.test(key))inside(value);
    }; scan(args); return {allowed:true};
  }
  async native(sid,name,args,call,source='model') {
    if(!this.sessions.has(sid))throw fail('native call for inactive session');
    const permitted=this.safeNative(name,args,sid,source);
    if (!permitted.allowed) { this.audit('owner_approval_required',{session:sid,call,tool:name,reason:permitted.reason}); return {isError:true,content:[{type:'text',text:permitted.reason}]}; }
    const admitted=await this.before(sid,call,name,args,true); if(!admitted.allowed)return {isError:true,content:[{type:'text',text:admitted.reason}]};
    const token=this.pending.get(call).token;
    const result=await this.broker.request({jsonrpc:'2.0',id:call,method:'tools/call',params:{name,arguments:{...args,[ROUTE_FIELD]:token}}});
    await this.after(sid,call,name,args,result,result.isError?'native tool reported error':null);
    if(!result.isError && name==='smart_ingest') {
      const body=result.structuredContent||JSON.parse(result.content.find(c=>c.type==='text').text);
      if(body.nodeId)this.lastWrite.set(sid,body.nodeId);
    }
    this.check(); return result;
  }
  async record(sid,data) {
    const write=await this.native(sid,'smart_ingest',{content:JSON.stringify(data),node_type:'event',
      tags:['benchmark-custom-trial'],entities:['service:AgentDojo']},'controller-'+randomUUID());
    const body=write.structuredContent||JSON.parse(write.content.find(c=>c.type==='text').text);
    if(!body.nodeId) throw fail('state record missing node');
    const receipt=await this.native(sid,'receipt',{action:'get',receipt_id:body.nodeId},'controller-'+randomUUID());
    const proof=receipt.structuredContent||JSON.parse(receipt.content.find(c=>c.type==='text').text);
    const v=proof.attestation?.verification;
    if(!v?.chainValid||!v.gateAllowed||!v.locallyVerified) throw fail('state write receipt did not verify');
    const backfill=await this.native(sid,'backfill',{trigger_id:body.nodeId,scope:'user'},'controller-'+randomUUID(),'controller');
    return {write,receipt,backfill};
  }
  async compact(sid,summary) {
    if(!this.sessions.has(sid))throw fail('compaction for inactive session');
    const sealed=this.operator(['turnover','--pre-compact','--source','benchmark-custom'],{session_id:sid,cwd:this.files,trigger:'benchmark-custom'});
    if(sealed.status!==0)throw fail('Operator compaction handoff failed');
    await this.fire('PreCompact',sid,{trigger:'benchmark-custom'});
    const safe=sid.replace(/[^A-Za-z0-9_.-]/g,'_').slice(0,80);
    const state=JSON.parse(readFileSync(join(this.operatorHome,'state','turnover',safe+'.json'),'utf8'));
    if(state.chit?.session!==sid||!/^[a-f0-9]{16}$/.test(state.id||''))throw fail('invalid native turnover identity');
    const day=new Date(state.sealed_t*1000).toISOString().slice(0,10);
    const rows=readFileSync(join(this.operatorHome,'receipts',day+'.jsonl'),'utf8').trim().split('\n').map(l=>JSON.parse(l));
    const receipt=rows.findLast(r=>r.session===sid&&r.chit===state.id&&r.decision==='TURNOVER_SEALED');
    if(!receipt?.digest||receipt.digest.slice(0,16)!==state.receipt)throw fail('missing native turnover receipt');
    await this.fire('SessionStart',sid,{source:'compact'});
    const out=await this.fire('PostCompact',sid,{summary,trigger:'benchmark-custom'});
    this.audit('operator_turnover_verified',{session:sid,chit:state.id,receipt_digest:receipt.digest});return out;
  }
  async stop(sid,parent=null) {
    if(!this.sessions.has(sid)||this.parents.get(sid)!==parent)throw fail('invalid session/parent at stop');
    if([...this.parents].some(([child,p])=>p===sid&&this.sessions.has(child)))throw fail('parent stopped with active child');
    if([...this.pending.values()].some(p=>p.sid===sid))throw fail('unacknowledged call at session end');
    const trigger=this.lastWrite.get(sid);
    if(trigger)await this.native(sid,'backfill',{trigger_id:trigger,scope:'user'},'controller-'+randomUUID());
    await this.fire('Stop',sid);
    if(parent)await this.fire('SubagentStop',parent,{agent:sid});
    await this.fire('SessionEnd',sid);this.sessions.delete(sid);this.parents.delete(sid);return {stopped:true};
  }
  async handle(method,p) {
    switch(method) {
      case 'init':return this.init(p.root);
      case 'start':return this.start(p.session,p.parent);
      case 'prompt':return this.prompt(p.session,p.text);
      case 'before':return this.before(p.session,p.call,p.tool,p.args);
      case 'after':return this.after(p.session,p.call,p.tool,p.args,p.result,p.error);
      case 'tool':return this.native(p.session,p.tool,p.args,p.call);
      case 'record':return this.record(p.session,p.data);
      case 'compact':return this.compact(p.session,p.summary);
      case 'stop':return this.stop(p.session,p.parent);
      case 'ledger':this.check();return this.broker.inspect('ledger');
      case 'close':this.check();if(this.pending.size||this.sessions.size)throw fail('close with incomplete lifecycle');await this.broker.close();return {closed:true};
      default:throw fail('unknown bridge method');
    }
  }
}

const stack=new Stack();let queue=Promise.resolve(),failed=false;
const input=createInterface({input:process.stdin});
input.on('line',line=>{
  queue=queue.then(async()=>{
    let request;try{request=JSON.parse(line);if(!Number.isSafeInteger(request.id))throw fail('missing bridge RPC ID');
      if(failed)throw fail('bridge is held after failure');
      const result=await stack.handle(request.method,request.params||{});
      process.stdout.write(JSON.stringify({id:request.id,result:result??{}})+'\n');
    }catch(error){failed=true;stack.audit?.('bridge_interdiction',{reason:error.message});
      process.stdout.write(JSON.stringify({id:request?.id??null,error:error.message})+'\n');}
  });
});
input.on('close',()=>queue.finally(()=>stack.broker?.close()).catch(()=>{}));

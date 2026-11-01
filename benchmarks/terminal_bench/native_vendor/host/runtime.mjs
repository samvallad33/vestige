import { spawn, spawnSync } from 'node:child_process';
import { appendFileSync, mkdirSync, openSync, readSync, closeSync, fstatSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { createHash } from 'node:crypto';

export const BINARY = '/Users/entity002/.vestige/bin/v6.2.0-d4598ab3/vestige-mcp';
export const EXPECTED = ['backfill','causal_walk','codebase','dedup','forgotten_lesson','gather','ghostlink','intention','maintain','memory','memory_status','precedent','project','recall','receipt','selftest','session_start','smart_ingest','suppress'];
export const digest = value => createHash('sha256').update(JSON.stringify(value) ?? 'undefined').digest('hex');
export const fail = message => new Error('FAIL_CLOSED_INTERDICTION: ' + message);

export function checkReflexDecision(out, event) {
  const decision = out.hookSpecificOutput?.permissionDecision || out.decision;
  if (out.continue === false || ['deny', 'block', 'ask'].includes(decision))
    throw fail('native Reflex requires owner intervention for ' + event);
}

export function auditWriter(root) {
  mkdirSync(join(root, 'audit'), { recursive: true });
  const path = join(root, 'audit', `host-${process.pid}.jsonl`);
  let previous = '0'.repeat(64);
  return (event, fields = {}) => {
    const entry = { time: new Date().toISOString(), event, ...fields, previous };
    const hash = digest(entry);
    appendFileSync(path, JSON.stringify({ ...entry, digest: hash }) + '\n', { mode: 0o600 });
    previous = hash;
  };
}

export class NativeMCP {
  constructor(binary = BINARY, timeout = 20000, args = []) {
    this.next = 0; this.pending = new Map(); this.buffer = ''; this.timeout = timeout;
    this.child = spawn(binary, args, { stdio: ['pipe','pipe','ignore'],
      env: { ...process.env, VESTIGE_RETENTION_TARGET: '0', VESTIGE_BACKFILL_AUTOFIRE: '1',
        VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' } });
    this.child.stdout.on('data', chunk => {
      this.buffer += chunk.toString();
      if (this.buffer.length > 8 * 1024 * 1024) return this.break(fail('oversized native RPC response'));
      for (;;) {
        const end = this.buffer.indexOf('\n'); if (end < 0) break;
        const line = this.buffer.slice(0, end); this.buffer = this.buffer.slice(end + 1);
        if (!line.trim()) continue;
        let response;
        try { response = JSON.parse(line); } catch { this.break(fail('malformed native RPC response')); break; }
        const request = this.pending.get(response.id);
        if (!request) {
          if ('id' in response) this.break(fail('unmatched native RPC response id'));
          continue;
        }
        this.pending.delete(response.id); clearTimeout(request.timer);
        if (response.error || !Object.hasOwn(response, 'result')) request.reject(fail('native RPC rejected request'));
        else request.resolve(response.result);
      }
    });
    this.child.on('error', error => this.break(fail('native MCP unavailable: ' + error.message)));
    this.child.on('exit', () => this.break(fail('native MCP transport dropped')));
    this.child.stdin.on('error', error => this.break(fail('native MCP write failed: ' + error.message)));
  }
  break(error) {
    this.error = error;
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(error); }
    this.pending.clear();
  }
  rpc(method, params) {
    if (this.error) return Promise.reject(this.error);
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.break(fail('native RPC timeout')); this.child.kill(); }, this.timeout);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    });
  }
  async initialize() {
    const result = await this.rpc('initialize', { protocolVersion: '2024-11-05', capabilities: {},
      clientInfo: { name: 'qwen-opencode-reflex', version: '1' } });
    if (result.serverInfo?.name !== 'vestige' || result.serverInfo?.version !== '6.2.0') throw fail('unexpected native MCP identity');
    this.hello = result;
    this.child.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n');
    const list = await this.rpc('tools/list', {});
    const names = list.tools.map(x => x.name).sort();
    if (JSON.stringify(names) !== JSON.stringify(EXPECTED)) throw fail('native v6 tool catalog mismatch');
    this.catalog = list.tools;
    return list.tools;
  }
  async reflex(args) {
    const response = await this.rpc('tools/call', { name: 'reflex', arguments: { host: 'opencode', ...args } });
    if (response.isError) throw fail('native Reflex rejected hook');
    let out = response.structuredContent;
    if (out === undefined) {
      const text = response.content?.find(x => x.type === 'text')?.text;
      if (typeof text !== 'string' || !text.trim()) throw fail('missing native Reflex response');
      try { out = JSON.parse(text); } catch { throw fail('invalid native Reflex hook JSON'); }
    }
    if (!out || typeof out !== 'object' || Array.isArray(out)) throw fail('invalid native Reflex output');
    return out;
  }
  close() { this.child.stdin.end(); this.child.kill(); }
}

export function operatorGate(payload) {
  const result = spawnSync('python3', ['/Users/entity002/.operator/gate/operator-gate.py', 'hook', '--source', 'opencode'], {
    input: JSON.stringify(payload), encoding: 'utf8', timeout: 60000, maxBuffer: 1024 * 1024,
    env: { ...process.env, OPERATOR_AGENT_SESSION: '1' },
  });
  if (result.error || result.status !== 0) {
    const reason = result.stderr?.trim() || result.error?.message || `exit ${result.status}`;
    if (result.status === 2) throw new Error('BLOCKED_BY_OPERATOR: ' + reason);
    throw fail('Operator gate unavailable: ' + reason);
  }
  const path = join(homedir(), '.operator', 'receipts', new Date().toISOString().slice(0, 10) + '.jsonl');
  let fd;
  try {
    fd = openSync(path, 'r'); const size = fstatSync(fd).size;
    const length = Math.min(size, 1024 * 1024); const buffer = Buffer.alloc(length);
    readSync(fd, buffer, 0, length, size - length);
    const rows = buffer.toString().split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
    const receipt = rows.findLast(x => x.session === payload.session_id && x.call === payload.tool_use_id);
    if (!receipt || receipt.decision !== 'PASS' || !receipt.digest) throw fail('missing matching native Operator PASS receipt');
    return receipt;
  } finally { if (fd !== undefined) closeSync(fd); }
}

function lifecycleCommand(args, payload) {
  const result = spawnSync('python3', ['/Users/entity002/.operator/gate/operator-gate.py', ...args], {
    input: JSON.stringify(payload), encoding:'utf8', timeout:20000, maxBuffer:1024*1024,
    env:{...process.env, OPERATOR_AGENT_SESSION:'1'},
  });
  if (result.error || result.status !== 0) throw fail('native Operator lifecycle command failed: '
    + (result.stderr?.trim() || result.error?.message || String(result.status)));
  return result.stdout;
}
export function operatorSessionStart(payload) {
  const text = lifecycleCommand(['laws','--session-start'], payload);
  let result; try { result=JSON.parse(text); } catch { throw fail('invalid native Operator laws response'); }
  const context=result.hookSpecificOutput?.additionalContext;
  if (typeof context!=='string' || !context) throw fail('missing native Operator session context');
  return context;
}
export function operatorTurnover(payload, seal = false) {
  const started=Date.now()/1000;
  if (seal) lifecycleCommand(['turnover','--pre-compact','--source','opencode'], payload);
  const safe=String(payload.session_id || '').replace(/[^A-Za-z0-9_.-]/g,'_').slice(0,80);
  if (!safe) throw fail('missing turnover session');
  const path=join(homedir(),'.operator','state','turnover',safe+'.json');
  let state;
  try { state=JSON.parse(readFileSync(path,'utf8')); }
  catch(error) { if(!seal && error.code==='ENOENT')return null; throw fail('missing native Operator compaction handoff'); }
  if (state.chit?.session!==payload.session_id || !/^[a-f0-9]{16}$/.test(state.id || ''))
    throw fail('Operator turnover identity mismatch');
  if (seal && state.sealed_t<started-1) throw fail('stale Operator turnover receipt');
  const day=new Date(state.sealed_t*1000).toISOString().slice(0,10);
  const rows=readFileSync(join(homedir(),'.operator','receipts',day+'.jsonl'),'utf8')
    .trim().split('\n').map(x=>JSON.parse(x));
  const receipt=rows.findLast(r=>r.decision==='TURNOVER_SEALED' && r.session===payload.session_id && r.chit===state.id);
  if (!receipt?.digest || receipt.digest.slice(0,16)!==state.receipt) throw fail('missing matching Operator TURNOVER_SEALED receipt');
  const text=lifecycleCommand(['turnover','--show',state.id],payload);
  return {...state, text, digest:receipt.digest};
}

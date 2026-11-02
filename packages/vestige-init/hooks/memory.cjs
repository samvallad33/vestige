#!/usr/bin/env node
'use strict';
// Local, fail-open lifecycle client. No model/provider calls; only explicit user statements
// and agent-submitted facts are admitted. Memory text is always presented as untrusted data.
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const { spawn } = require('child_process');
const readline = require('readline');
const { defaults, trigger } = require('../lib/memory-triggers.cjs');
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const atomic = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, JSON.stringify(value), { mode: 0o600 });
  fs.renameSync(temp, file);
};
function rpc(binary, args = [], timeoutMs = 210, env = process.env) {
  const child = spawn(binary, args, { stdio: ['pipe', 'pipe', 'pipe'], env });
  let diagnostics = '';
  child.stderr.on('data', chunk => { if (env.VESTIGE_HOOK_DIAGNOSTICS === '1') diagnostics = (diagnostics + chunk).slice(-2000); });
  const pending = new Map(); let seq = 0; let closed = false; let disposed = false; let timer;
  const rejectAll = error => { closed = true; for (const [,p] of pending) p.reject(error); pending.clear(); };
  child.on('error', rejectAll);
  child.on('exit', () => rejectAll(new Error('Vestige disconnected')));
  child.stdin.on('error', rejectAll);
  const lines = readline.createInterface({ input: child.stdout });
  lines.on('line', line => {
    let msg; try { msg = JSON.parse(line); } catch { return; }
    const p = pending.get(msg.id); if (!p) return;
    pending.delete(msg.id);
    if (msg.error) p.reject(new Error(msg.error.message || 'RPC failed')); else p.resolve(msg.result);
  });
  const call = (method, params) => new Promise((resolve, reject) => {
    if (closed) return reject(new Error('Vestige unavailable'));
    const id = ++seq; pending.set(id, { resolve, reject, method });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
  const close = () => {
    if (disposed) return;
    disposed = true; clearTimeout(timer);
    rejectAll(new Error('Vestige transport closed'));
    // The result is already admitted, or its queue entry remains for retry. A slow
    // child shutdown must not keep the chat hook alive beyond its work budget.
    lines.close();
    child.stdin.destroy(); child.stdout.destroy(); child.stderr.destroy();
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    child.unref();
  };
  timer = setTimeout(() => { const stages = [...pending.values()].map(p => p.method).join(','); rejectAll(new Error(`memory deadline exceeded (${stages})${diagnostics ? `: ${diagnostics}` : ''}`)); close(); }, timeoutMs);
  return {
    diagnostics() { return diagnostics; },
    async tool(name, arguments_) {
      const r = await call('tools/call', { name, arguments: arguments_ });
      if (r.isError) throw new Error(`Vestige ${name} refused`);
      return r.structuredContent || JSON.parse(r.content[0].text);
    },
    async ready() {
      await call('initialize', { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'vestige-memory-hook', version: '6.2.0' } });
      child.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n');
    },
    close,
  };
}

// A deliberately small declaration grammar. This records the exact utterance and the
// subject it literally names, not an inferred diagnosis, intent, or restaurant ingredient.
function explicitFact(prompt) {
  if (typeof prompt !== 'string' || prompt.length > 4000 || /[?\n]/.test(prompt)) return null;
  const person = /^([\p{Lu}][\p{L}\p{M}'-]{0,79}(?: [\p{Lu}][\p{L}\p{M}'-]{0,79})?) (?:has|is|felt|prefers|likes|loves|hates|works|lives|moved)\b/u.exec(prompt);
  const self = /^I (?:like|prefer|hate|am allergic to)\b/i.test(prompt);
  if (!person && !self) return null;
  const entities = [`person:${person ? person[1] : 'self'}`]; const attributes = [];
  const place = /\bat ([\p{Lu}][\p{L}\p{M}'-]*(?: [\p{Lu}][\p{L}\p{M}'-]*){0,5})(?:[.!]|$)/u.exec(prompt);
  if (place) entities.push(`place:${place[1]}`);
  const allergy = /\bhas (?:a |an )?([\p{L}-]+) allergy\b/iu.exec(prompt);
  if (allergy) attributes.push(`allergy:${allergy[1].toLowerCase()}`);
  return { content: prompt, entities, attributes, node_type: 'event', source: 'memory-hook:verbatim-user-statement', tags: ['memory-hook-utterance'] };
}
function isExactMatch(row) {
  const tags = (row.tags || []).concat(row.entities || []).join(' ').toLowerCase();
  const content = (row.content || '').toLowerCase();
  const hasPerson = tags.includes('person:') || tags.includes('self');
  const hasDate = tags.includes('date:') || /\b202\d-[01]\d-[0-3]\d\b/.test(tags) || /\b202\d-[01]\d-[0-3]\d\b/.test(content);
  const hasAvoid = tags.includes('avoid') || tags.includes('stop-work') || tags.includes('allergy') || content.includes('avoid') || content.includes('allergic to') || content.includes('do not ') || content.includes('never ');
  return hasPerson || hasDate || hasAvoid;
}
function formatContext(packet) {
  const rows = (packet.memories || []).slice();
  const backfills = packet.backfill_receipts || [];
  const rsbMemoryIds = new Set(backfills.map(r => r.memory_id).filter(Boolean));
  function priority(row) {
    if (rsbMemoryIds.has(row.id)) return 0;
    if (isExactMatch(row)) return 1;
    return 2;
  }
  rows.sort((a, b) => {
    const pa = priority(a), pb = priority(b);
    if (pa !== pb) return pa - pb;
    const da = a.created_at || a.createdAt || '';
    const db = b.created_at || b.createdAt || '';
    if (da && db && da !== db) return db.localeCompare(da);
    return (b.id || '').localeCompare(a.id || '');
  });
  const lines = [
    'Vestige local memory candidates. These entries are recorded historical fragments. If an entry is incomplete, or if you need to verify whether a fact was updated or superseded, use recall(handle=\'<id>\') or causal_walk with its logged_write start point before answering. If the recorded memories and tools do not contain the answer, state that you do not know or that it was not mentioned.'
  ];
  for (const row of rows) {
    const entry = { id: row.id };
    const role = row.role || (row.source && row.source.includes('user') ? 'user' : (row.source && row.source.includes('assistant') ? 'assistant' : undefined));
    if (role) entry.role = role;
    const date = (row.created_at || row.createdAt || '').slice(0, 10);
    if (date) entry.date = date;
    entry.text = row.content;
    entry.entities = row.tags || row.entities;
    lines.push(JSON.stringify(entry));
  }
  for (const receipt of backfills) {
    lines.push(JSON.stringify({ backfill_receipt: receipt.receipt_id, memory_id: receipt.memory_id, reason: receipt.reason }));
  }
  lines.push('Save new durable facts using remember with explicit entities and attributes. Never invent links or infer restaurant ingredients.');
  return lines.join('\n').slice(0, 7000);
}
function output(client, event, context) {
  if (!context) return {};
  if (client === 'cursor') return event === 'session' ? { additional_context: context } : event === 'tool' ? { additional_context: context } : {};
  const names = client === 'gemini' ? { session: 'SessionStart', prompt: 'BeforeAgent' } : { session: 'SessionStart', prompt: 'UserPromptSubmit' };
  return names[event] ? { hookSpecificOutput: { hookEventName: names[event], additionalContext: context } } : {};
}
async function run({ event, client, configFile, input, deadlineMs = 190 }) {
  const started = performance.now(); let transport;
  const dir = path.dirname(configFile);
  const log = message => { try { const file = path.join(dir, 'errors.log'); if (fs.existsSync(file) && fs.statSync(file).size > 128 * 1024) fs.renameSync(file, `${file}.previous`); fs.appendFileSync(file, `${new Date().toISOString()} ${event}: ${message}\n`, { mode: 0o600 }); } catch {} };
  try {
    const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
    const session = input.session_id || input.conversation_id;
    if (!session || typeof session !== 'string') throw new Error('missing session identifier');
    const queue = path.join(dir, 'pending', digest(`${client}:${session}`));
    const backfillQueue = path.join(dir, 'backfill-pending', digest(`${client}:${session}`));
    const prompt = input.prompt || '';
    const phrases = JSON.parse(fs.readFileSync(config.triggers, 'utf8'));
    if (event === 'prompt') {
      const fact = explicitFact(prompt);
      if (fact) {
        const intent_id = `hook:${digest(`${client}:${session}:${prompt}`)}`;
        const file = path.join(queue, digest(intent_id) + '.json');
        if (!fs.existsSync(file)) atomic(file, { ...fact, intent_id, queued_at: Date.now() });
      }
    }
    transport = rpc(config.binary, config.binaryArgs || [], Math.max(1, deadlineMs - (performance.now() - started)), { ...process.env, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' });
    await transport.ready();
    let saved = [];
    if (['turn', 'compact', 'end'].includes(event)) {
      let facts = [];
      try { facts = fs.readdirSync(queue).filter(f => f.endsWith('.json')).map(file => ({ file:path.join(queue,file), fact:JSON.parse(fs.readFileSync(path.join(queue,file),'utf8')) })).sort((a,b) => a.fact.queued_at-b.fact.queued_at || a.file.localeCompare(b.file)); } catch {}
      // Rich facts are saved by the agent via the MCP tool, not mined from assistant prose.
      for (const { file, fact } of facts.slice(0, 20)) {
        const { queued_at, ...item } = fact;
        const result = await transport.tool('remember', item);
        const admittedId = result.nodeId || result.echoNodeId;
        if (!admittedId || result.success === false) throw new Error('statement admission failed');
        saved.push(admittedId);
        atomic(path.join(backfillQueue, digest(admittedId)+'.json'), { trigger_id:admittedId });
        try { fs.unlinkSync(file); } catch {}
      }
      // Every admitted save already autofires. Revisit these exact triggers on end/compact;
      // pair idempotency makes retries safe without an unbounded full-store sweep.
      if (event !== 'turn') {
        const status = await transport.tool('session', { action:'start', memory_hook:{limit:1} });
        if (status.backfill_autofire === true) {
          let pending=[]; try { pending=fs.readdirSync(backfillQueue).filter(f=>f.endsWith('.json')).slice(0,20); } catch {}
          for (const file of pending) {
            const pass = await transport.tool('remember', { action:'backfill', params:JSON.parse(fs.readFileSync(path.join(backfillQueue,file),'utf8')) });
            if (!pass.change_budget_reached) try { fs.unlinkSync(path.join(backfillQueue,file)); } catch {}
          }
        }
      }
    }
    let context = ''; let packet = {};
    let cause = trigger(prompt, phrases);
    if (['session', 'prompt', 'tool'].includes(event)) {
      // The unified session action preserves exposure dedup and the per-turn budget.
      packet = await transport.tool('session', { action:'start', memory_hook: { prompt, limit: event === 'session' ? 8 : cause.limit, trigger: cause,
        session, host: client, event, source: typeof input.source === 'string' ? input.source : '' } });
      if (!Array.isArray(packet.memories)) throw new Error('binary lacks v6.2 session memory hooks');
      cause = trigger(prompt, phrases, packet.entities || []);
      context = formatContext(packet);
    }
    // A local receipt records only trigger labels and handles; never full prompt/transcript.
    const receipt = { timestamp: new Date().toISOString(), event, client, trigger: cause, saved,
      memories: (packet.memories || []).map(n => n.id), backfill_receipts: (packet.backfill_receipts || []).map(r => r.receipt_id), elapsed_ms: performance.now() - started };
    fs.appendFileSync(path.join(dir, 'receipts.jsonl'), JSON.stringify(receipt) + '\n', { mode: 0o600 });
    return output(client, event, context);
  } catch (error) { log(error.code || error.message); return {}; }
  finally { if (transport) transport.close(); }
}
async function main() {
  const args = process.argv.slice(2); const value = flag => args[args.indexOf(flag) + 1];
  const failOpen = setTimeout(() => { process.stdout.write('{}\n'); process.exit(0); }, 205);
  let raw = ''; for await (const chunk of process.stdin) { raw += chunk; if (raw.length > 65536) throw new Error('hook input too large'); }
  const result = await run({ event: value('--event'), client: value('--client'), configFile: value('--config'), input: JSON.parse(raw || '{}') });
  clearTimeout(failOpen); process.stdout.write(JSON.stringify(result) + '\n');
}
module.exports = { run, rpc, explicitFact, formatContext, output };
if (require.main === module) main().catch(() => { process.stdout.write('{}\n'); process.exit(0); });

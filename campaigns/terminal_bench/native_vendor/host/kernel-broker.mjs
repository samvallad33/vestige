import { createServer } from 'node:net';
import { chmodSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomBytes } from 'node:crypto';
import { NativeMCP, BINARY, digest, fail, checkReflexDecision } from './runtime.mjs';
import { verifyWindow } from './audit-window.mjs';
import { loadRecovery } from './audit-recovery.mjs';

export const ROUTE_FIELD = '__vestige_host_route';
const canonicalValue = value => Array.isArray(value) ? value.map(canonicalValue)
  : value && typeof value === 'object'
    ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalValue(value[key])])) : value;
export const argsDigest = value => digest(canonicalValue(value));

// Model traffic and lifecycle hooks share exactly one native MCP connection.
export class KernelBroker {
  constructor({ audit, dataDir, native } = {}) {
    this.audit = audit || (() => {});
    this.native = native || new NativeMCP(BINARY, 20000, dataDir ? ['--data-dir', dataDir] : []);
    this.recoveryPath = dataDir ? null : join(dirname(fileURLToPath(import.meta.url)), 'audit-recovery.json');
    this.routes = new Map(); this.sockets = new Set(); this.queue = Promise.resolve(); this.blocked = null;
    this.socketPath = join(tmpdir(), 'vq-' + randomBytes(12).toString('hex') + '.sock');
  }
  async initialize() {
    const catalog = await this.native.initialize();
    const ledger = await this.native.reflex({ action:'ledger', since:'1h' });
    if (ledger.chain?.status === 'broken' && this.recoveryPath) {
      this.checkpoint = await loadRecovery(this.native, this.recoveryPath, ledger);
      this.audit('audit_recovery_verified', { memory_id:this.checkpoint.memoryId,
        owner_approval:this.checkpoint.ownerApprovalMemoryId, historical_breaks:this.checkpoint.historicalBreaks,
        history_claim:'historical gaps preserved; current window verified' });
    } else if (!['ok','empty'].includes(ledger.chain?.status)) throw fail('native ledger baseline invalid');
    this.audit('native_connected', { version: '6.0.1', tools: catalog.map(x => x.name),
      schema_digest: digest(catalog), shared_native_connection: true });
    this.server = createServer(socket => {
      this.sockets.add(socket);
      socket.on('close', () => this.sockets.delete(socket));
      let buffered = '';
      socket.on('data', chunk => {
        buffered += chunk.toString();
        if (buffered.length > 8 * 1024 * 1024) { socket.destroy(fail('oversized MCP gateway frame')); return; }
        for (;;) {
          const end = buffered.indexOf('\n'); if (end < 0) break;
          const line = buffered.slice(0, end); buffered = buffered.slice(end + 1);
          if (!line.trim()) continue;
          let request;
          try { request = JSON.parse(line); }
          catch { socket.destroy(fail('malformed gateway request')); break; }
          this.request(request).then(result => {
            if (request.id !== undefined && !socket.destroyed)
              socket.write(JSON.stringify({ jsonrpc:'2.0', id:request.id, result }) + '\n');
          }).catch(error => {
            if (request.id !== undefined && !socket.destroyed)
              socket.write(JSON.stringify({ jsonrpc:'2.0', id:request.id,
                error:{ code:-32000, message:error.message } }) + '\n');
          });
        }
      });
      socket.on('error', () => {});
    });
    await new Promise((resolve, reject) => { this.server.once('error', reject); this.server.listen(this.socketPath, resolve); });
    chmodSync(this.socketPath, 0o600);
    return catalog;
  }
  serial(action, cleanup = false) {
    const next = this.queue.then(() => {
      if (this.blocked && !cleanup) throw fail(this.blocked);
      return action();
    });
    this.queue = next.catch(error => {
      if (!this.blocked) { this.blocked = error.message; this.audit('kernel_interdiction', { reason:error.message }); }
    });
    return next;
  }
  async fireUnsafe(fields) {
    const before = await this.native.reflex({ action:'ledger', since:'1h' });
    if (this.checkpoint && (before.ledger !== this.checkpoint.ledger
      || before.chain?.linesChecked < this.checkpoint.linesChecked
      || before.chain?.breaks !== this.checkpoint.historicalBreaks))
      throw fail('native ledger diverged from approved checkpoint');
    const out = await this.native.reflex(fields);
    checkReflexDecision(out, fields.event);
    // Native v6 catches internal hook errors as {}. Require its own ledger
    // row as well, so an empty error response cannot masquerade as an ack.
    const ledger = await this.native.reflex({ action:'ledger', since:'1h' });
    const window = verifyWindow(before, ledger, fields, this.checkpoint), row = window.row;
    this.audit('reflex_ack', { event_name:fields.event, session:fields.session, agent:fields.agent || '',
      call:fields.call || '', response_digest:digest(out), native_ledger_digest:row.digest,
      native_ledger_chain:ledger.chain.status, current_window_verified:true,
      historical_breaks:window.historicalBreaks, history_claim:window.historyClaim,
      prior_tip:before.recent?.at(-1)?.digest || '0'.repeat(64), verified_tip:window.tip,
      held:row.held, wrote:row.wrote || [] });
    return out;
  }
  hook(fields) { return this.serial(() => this.fireUnsafe(fields), fields.event === 'SessionEnd'); }
  inspect(action) { return this.serial(() => this.native.reflex({ action, since:'1h' }), true); }
  admit({ tool, args, fields }) {
    if (this.blocked) throw fail(this.blocked);
    if (Object.hasOwn(args, ROUTE_FIELD)) throw fail('model supplied reserved host route field');
    const token = randomBytes(24).toString('hex');
    this.routes.set(token, { tool, fields, digest:argsDigest(args), used:false, finished:false });
    return token;
  }
  finish(token) {
    const route = this.routes.get(token);
    if (!route?.used || !route.finished) throw fail('native MCP execution was not acknowledged');
    this.routes.delete(token);
    return route;
  }
  async request(request) {
    if (request.jsonrpc !== '2.0') throw fail('invalid MCP JSON-RPC envelope');
    if (request.method === 'notifications/initialized') return {};
    if (request.method === 'initialize') return this.native.hello;
    if (request.method === 'tools/list') return { tools:this.native.catalog };
    if (request.method === 'ping') return {};
    if (request.method === 'prompts/list') return {prompts:[]};
    if (request.method === 'resources/templates/list') return {resourceTemplates:[]};
    if (request.method === 'resources/list')
      return this.serial(() => this.native.rpc(request.method, request.params || {}));
    if (request.method !== 'tools/call') throw fail('unsupported or un-interdicted MCP gateway method');
    return this.serial(async () => {
      const params = request.params, args = { ...(params?.arguments || {}) };
      const token = args[ROUTE_FIELD]; delete args[ROUTE_FIELD];
      const route = this.routes.get(token);
      if (!route || route.used || route.tool !== params?.name || route.digest !== argsDigest(args))
        throw fail('missing, mismatched or replayed host MCP route');
      route.used = true;
      await this.fireUnsafe(route.fields);
      const metadata = { ...(params._meta || {}) };
      if (metadata['claudecode/toolUseId'] && metadata['claudecode/toolUseId'] !== route.fields.call)
        throw fail('conflicting native tool call identity');
      metadata['claudecode/toolUseId'] = route.fields.call;
      const result = await this.native.rpc('tools/call', { name:route.tool, arguments:args, _meta:metadata });
      route.finished = true; route.result = result;
      this.audit('native_tool_returned', { session:route.fields.session, agent:route.fields.agent || '',
        call:route.fields.call, tool:route.tool, args_digest:route.digest, is_error:result.isError === true,
        shared_native_connection:true });
      return result;
    });
  }
  async close() {
    await this.queue;
    for (const socket of this.sockets) socket.destroy();
    if (this.server) await new Promise(resolve => this.server.close(resolve));
    try { unlinkSync(this.socketPath); } catch (e) { if (e.code !== 'ENOENT') throw e; }
    this.native.close();
  }
}

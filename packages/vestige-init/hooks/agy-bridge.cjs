#!/usr/bin/env node
'use strict';
// Vestige <-> Antigravity hook bridge.
//
// Antigravity speaks camelCase hook JSON ({conversationId, toolCall:{id,name,args}, stepIdx, ...})
// and answers with {decision, reason, overwrite} / {injectSteps}. Vestige's shipped hooks speak
// the Claude Code dialect ({session_id, tool_name, tool_input, ...}). This file translates in
// both directions and delegates every decision to the shipped components:
//
//   SessionStart   -> Operator turnover + `vestige capsule` + active Intention Graph/list +
//                     memory hook (`session` packet) + Reflex `SessionStart`.
//   PreInvocation  -> first call of a turn: Save-Gate `UserPromptSubmit` + memory hook (`prompt`
//                     recall + RSB) + Reflex `UserPromptSubmit` (+ active intentions on prompt #1);
//                     later calls: Reflex `PostToolBatch`, plus any pending notices.
//   PreToolUse     -> Operator gate (`operator-gate.py hook`, FAIL-CLOSED) + Save-Gate commit
//                     gate + Reflex `PreToolUse` (Context Manifest, Call Coach #520 + overwrite).
//   PostToolUse    -> Extracts real MCP JSON (`nodeId`, `nodes`, `results`) from transcript/timeline
//                     for all `mcp__vestige__*` calls, then runs Save-Gate + Reflex.
//   PostInvocation -> Reflex `PostToolBatch` (queues batch/failure notices for next step).
//   Stop           -> memory hook (`turn` flush + backfill) + autoSaveAtStop + PostToolUse settle +
//                     Reflex `Stop` + Save-Gate `Stop` (with single-block-per-prompt guard).
//
// Usage: node agy-bridge.cjs --event <Event> [--config <bridge.json>]
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const argv = process.argv.slice(2);
const flag = (name, dflt) => { const i = argv.indexOf(name); return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt; };
const EVENT = flag('--event', '');
const HERE = __dirname;
const CONFIG_FILE = flag('--config', path.join(HERE, 'bridge.json'));
const STATE_DIR = path.join(os.homedir(), '.vestige', 'antigravity');

function resolveHome(p) {
  return typeof p === 'string' ? p.replace(/^~(?=$|[\\/])/, os.homedir()) : p;
}

function loadConfig() {
  const cfg = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
  const memCfgPath = resolveHome(cfg.memoryConfig);
  let memCfg = {};
  try { memCfg = JSON.parse(fs.readFileSync(memCfgPath, 'utf8')); } catch {}
  return {
    python: cfg.python,
    gate: resolveHome(cfg.gate),
    saveGate: resolveHome(cfg.saveGate || path.join(os.homedir(), '.vestige', 'gate', 'save-gate.py')),
    vestigeBin: resolveHome(cfg.vestigeBin || path.join(os.homedir(), '.vestige', 'bin', process.platform === 'win32' ? 'vestige.exe' : 'vestige')),
    dataDir: resolveHome(cfg.dataDir || path.join(os.homedir(), '.vestige', 'store')),
    memoryConfig: memCfgPath,
    memCfg,
    hooksDir: cfg.hooksDir ? resolveHome(cfg.hooksDir) : path.join(HERE, 'hooks'),
    gateTimeoutMs: cfg.gateTimeoutMs || 15000,
    memoryDeadlineMs: cfg.memoryDeadlineMs || 7000,
    reflexDeadlineMs: cfg.reflexDeadlineMs || 3500,
    capsuleTimeoutMs: cfg.capsuleTimeoutMs || 6000,
    debug: Boolean(cfg.debug),
  };
}

function log(cfg, record) {
  if (!cfg || !cfg.debug) return;
  try {
    fs.mkdirSync(STATE_DIR, { recursive: true });
    const file = path.join(STATE_DIR, 'bridge.log');
    if (fs.existsSync(file) && fs.statSync(file).size > 2 * 1024 * 1024) fs.renameSync(file, file + '.1');
    fs.appendFileSync(file, JSON.stringify({ t: new Date().toISOString(), event: EVENT, ...record }) + '\n');
  } catch {}
}

// ---------------------------------------------------------------- translation
function parseArgumentsField(rawArgs) {
  if (rawArgs && typeof rawArgs === 'object' && !Array.isArray(rawArgs)) return rawArgs;
  if (typeof rawArgs === 'string' && rawArgs.trim()) {
    try {
      const parsed = JSON.parse(rawArgs);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch {}
  }
  return {};
}

function toClaudeTool(call) {
  const name = String((call && call.name) || '');
  const a = (call && call.args && typeof call.args === 'object') ? call.args : {};
  const lname = name.toLowerCase();
  const pathArg = a.Path || a.AbsolutePath || a.TargetFile || a.FilePath || a.path || a.file_path;

  if (lname === 'call_mcp_tool' && /vestige/i.test(String(a.ServerName || '')) && a.ToolName) {
    const cleanTool = String(a.ToolName).replace(/^"+|"+$/g, '');
    return {
      tool_name: `mcp__vestige__${cleanTool}`,
      tool_input: parseArgumentsField(a.Arguments),
    };
  }
  const mcpMatch = /^mcp_vestige(?:_vestige)?_(.+)$/i.exec(name);
  if (mcpMatch) {
    return { tool_name: `mcp__vestige__${mcpMatch[1]}`, tool_input: a };
  }
  if (typeof a.CommandLine === 'string' || /^(run_command|run_shell_command|command)$/.test(lname)) {
    const tool_input = { command: String(a.CommandLine || a.command || '') };
    if (a.Cwd) tool_input.cwd = a.Cwd;
    if (a.IsDaemon === true || (a.WaitMsBeforeAsync !== undefined && Number(a.WaitMsBeforeAsync) <= 500)) {
      tool_input.run_in_background = a.IsDaemon === true;
    }
    return { tool_name: 'Bash', tool_input };
  }
  if ((typeof a.Target === 'string' && typeof a.Replacement === 'string') || /^(edit_file|replace_file_content|multi_replace_file_content)$/.test(lname)) {
    const oldStr = typeof a.Target === 'string' ? a.Target : String(a.TargetContent || '');
    const newStr = typeof a.Replacement === 'string' ? a.Replacement : String(a.ReplacementContent || '');
    return { tool_name: 'Edit', tool_input: { file_path: pathArg || '', old_string: oldStr, new_string: newStr } };
  }
  if ((typeof a.Content === 'string' || typeof a.CodeContent === 'string') && pathArg) {
    const content = typeof a.Content === 'string' ? a.Content : a.CodeContent;
    if (a.Append === true) return { tool_name: 'Edit', tool_input: { file_path: pathArg, old_string: '', new_string: content } };
    return { tool_name: 'Write', tool_input: { file_path: pathArg, content } };
  }
  if (/^(read_file|view_file|view_file_outline|view_code_item)$/.test(lname) && pathArg) {
    const tool_input = { file_path: pathArg };
    const startLine = Number(a.StartLine ?? a.offset);
    const endLine = Number(a.EndLine);
    if (Number.isFinite(startLine) && startLine > 0) tool_input.offset = startLine;
    if (Number.isFinite(endLine) && endLine > 0) {
      tool_input.limit = Number.isFinite(startLine) && startLine > 0 ? Math.max(1, endLine - startLine + 1) : endLine;
    } else if (Number.isFinite(Number(a.limit)) && Number(a.limit) > 0) {
      tool_input.limit = Number(a.limit);
    }
    if (Number.isFinite(Number(a.ByteOffset)) && Number(a.ByteOffset) > 0) {
      tool_input.byte_offset = Number(a.ByteOffset);
    }
    return { tool_name: 'Read', tool_input };
  }
  return { tool_name: name, tool_input: a };
}

function toClaudePayload(input, hookEvent) {
  const session = String(input.conversationId || '');
  const ws = Array.isArray(input.workspacePaths) && input.workspacePaths[0] ? input.workspacePaths[0] : (input.cwd || process.cwd());
  const out = {
    hook_event_name: hookEvent,
    session_id: session,
    cwd: ws,
    transcript_path: input.transcriptPath || '',
    artifact_dir: input.artifactDirectoryPath || '',
    source: 'antigravity',
  };
  if (input.toolCall) {
    Object.assign(out, toClaudeTool(input.toolCall));
    if (out.tool_input && out.tool_input.cwd) out.cwd = out.tool_input.cwd;
    out.tool_use_id = String(input.toolCall.id || `agy-${session.slice(0, 8)}-${input.stepIdx ?? 'x'}`);
  }
  if (typeof input.lastUserInput === 'string') out.prompt = input.lastUserInput;
  if (typeof input.lastAssistantMessage === 'string') out.last_assistant_message = input.lastAssistantMessage;
  if (input.continuationActive === true) out.stop_hook_active = true;
  return out;
}

function findGitRootForPath(absFile) {
  try {
    let dir = path.dirname(absFile);
    for (let depth = 0; depth < 16; depth++) {
      if (fs.existsSync(path.join(dir, '.git'))) return dir;
      const parent = path.dirname(dir);
      if (!parent || parent === dir) break;
      dir = parent;
    }
  } catch {}
  return null;
}

function normalizePathForSaveGate(rawPath, cwd) {
  if (typeof rawPath !== 'string' || !rawPath.trim()) return '';
  const stripped = rawPath.trim().replace(/#.*$/, '').replace(/:(\d+(?:-\d+)?)$/, '');
  const slash = (p) => p.replace(/\\/g, '/').replace(/^\.\//, '');
  try {
    if (path.isAbsolute(stripped)) {
      const absFile = path.normalize(stripped);
      const gitRoot = findGitRootForPath(absFile);
      if (gitRoot) {
        const cmpFile = process.platform === 'win32' ? absFile.toLowerCase() : absFile;
        const cmpRoot = process.platform === 'win32' ? gitRoot.toLowerCase() : gitRoot;
        const prefix = cmpRoot.endsWith(path.sep) ? cmpRoot : cmpRoot + path.sep;
        if (cmpFile.startsWith(prefix)) {
          return slash(absFile.slice(prefix.length));
        }
      }
    }
    if (cwd) {
      const absFile = path.isAbsolute(stripped) ? path.normalize(stripped) : path.resolve(cwd, stripped);
      const absCwd = path.resolve(cwd);
      const cmpFile = process.platform === 'win32' ? absFile.toLowerCase() : absFile;
      const cmpCwd = process.platform === 'win32' ? absCwd.toLowerCase() : absCwd;
      const prefix = cmpCwd.endsWith(path.sep) ? cmpCwd : cmpCwd + path.sep;
      if (cmpFile.startsWith(prefix)) {
        return slash(absFile.slice(prefix.length));
      }
    }
  } catch {}
  return slash(stripped);
}

function isScratchOrArtifactEdit(filePath, artifactDir) {
  if (!filePath) return false;
  const norm = filePath.replace(/\\/g, '/').toLowerCase();
  if (artifactDir) {
    const normArt = artifactDir.replace(/\\/g, '/').toLowerCase().replace(/\/+$/, '');
    if (normArt && (norm === normArt || norm.startsWith(normArt + '/'))) return true;
  }
  if (norm.includes('/.gemini/antigravity/brain/') || norm.includes('/.gemini/config/plugins/vestige/')) return true;
  if (norm.includes('/.vestige/')) return true;
  return false;
}

function normalizePayloadForSaveGate(payload) {
  if (!payload || typeof payload !== 'object') return payload;
  const copy = { ...payload };
  const ti = (copy.tool_input && typeof copy.tool_input === 'object') ? { ...copy.tool_input } : {};
  const cwd = copy.cwd || '';
  if ((copy.tool_name === 'Edit' || copy.tool_name === 'Write') && typeof ti.file_path === 'string') {
    if (isScratchOrArtifactEdit(ti.file_path, copy.artifact_dir)) {
      return null;
    }
    ti.file_path = normalizePathForSaveGate(ti.file_path, cwd);
    copy.tool_input = ti;
  } else if (/^mcp__vestige__(?:remember|curate)$/i.test(String(copy.tool_name || '')) && Array.isArray(ti.files)) {
    ti.files = ti.files.map(f => normalizePathForSaveGate(String(f), cwd)).filter(Boolean);
    copy.tool_input = ti;
  }
  return copy;
}

function readStdin() {
  try { return fs.readFileSync(0, 'utf8'); } catch { return ''; }
}

// ---------------------------------------------------------------- memory + reflex + capsule + save-gate
function shim(cfg, name) {
  return require(path.join(cfg.hooksDir, name));
}

async function reflexRun(cfg, payload, event) {
  try {
    const reflex = shim(cfg, 'reflex.cjs');
    return await reflex.run({
      client: 'claude',
      configFile: cfg.memoryConfig,
      eventFlag: event,
      input: { ...payload, hook_event_name: event },
      deadlineMs: cfg.reflexDeadlineMs,
    }) || {};
  } catch (e) {
    log(cfg, { reflexError: String(e && e.message) });
    return {};
  }
}

function saveGateRun(cfg, payload, event) {
  if (!cfg.saveGate || !fs.existsSync(cfg.saveGate)) return {};
  const normalized = normalizePayloadForSaveGate(payload);
  if (normalized === null) return {};
  try {
    const saveDir = path.join(STATE_DIR, 'save-gate');
    fs.mkdirSync(saveDir, { recursive: true });
    const r = spawnSync(cfg.python, ['-I', cfg.saveGate, '--event', event], {
      input: JSON.stringify(normalized),
      encoding: 'utf8',
      timeout: 4000,
      maxBuffer: 512 * 1024,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', VESTIGE_REFLEX_DIR: saveDir },
      windowsHide: true,
    });
    const out = String(r.stdout || '').trim();
    if (!out) return {};
    return JSON.parse(out.split(/\r?\n/).pop() || '{}');
  } catch (e) {
    log(cfg, { saveGateError: String(e && e.message) });
    return {};
  }
}

function isGitCommitCommand(cmd) {
  return /(?:^|[;&|]\s*|&&\s*)git(?:\s+-C\s+(?:"[^"]+"|'[^']+'|\S+)|\s+--git-dir(?:=|\s+)\S+|\s+--work-tree(?:=|\s+)\S+)*\s+commit\b/.test(String(cmd || ''));
}

function extractGitRepoFromCommand(cmd, fallbackCwd) {
  const m = /\bgit\s+-C\s+(?:"([^"]+)"|'([^']+)'|(\S+))/.exec(String(cmd || ''));
  const raw = m ? (m[1] || m[2] || m[3]) : '';
  if (raw) {
    return path.isAbsolute(raw) ? raw : path.resolve(fallbackCwd || process.cwd(), raw);
  }
  return fallbackCwd || process.cwd();
}

function postCommitReceiptRun(cfg, payload) {
  if (!cfg.vestigeBin || !fs.existsSync(cfg.vestigeBin)) return;
  const cmd = String((payload.tool_input && payload.tool_input.command) || '');
  if (!isGitCommitCommand(cmd)) return;
  const repoDir = extractGitRepoFromCommand(cmd, payload.cwd);
  try {
    spawnSync(
      cfg.vestigeBin,
      ['--data-dir', cfg.dataDir, 'hook', 'post-commit', '--repo', repoDir],
      {
        encoding: 'utf8',
        timeout: 5000,
        maxBuffer: 512 * 1024,
        windowsHide: true,
      }
    );
  } catch (e) {
    log(cfg, { postCommitError: String(e && e.message) });
  }
}

function turnoverSessionStart(cfg, payload) {
  if (!cfg.gate || !fs.existsSync(cfg.gate)) return;
  try {
    spawnSync(cfg.python, ['-I', cfg.gate, 'turnover', '--session-start'], {
      input: JSON.stringify(payload),
      encoding: 'utf8',
      timeout: 3500,
      maxBuffer: 256 * 1024,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
  } catch {}
}

function compactCapsule(raw, maxChars = 1100) {
  if (!raw) return '';
  const lines = String(raw).split(/\r?\n/);
  const out = [];
  let section = '';
  let pinnedCount = 0;
  let rulebookSeen = false;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith('## ')) {
      section = line.slice(3).toLowerCase();
      if (section === 'rulebook' || /^r\d{2}\b/.test(section)) {
        if (!rulebookSeen) {
          out.push('Rulebook: R01-R17 active (owner rules override via owner-rule tag).');
          rulebookSeen = true;
        }
        section = 'rulebook';
        continue;
      }
      if (['changed files', 'due obligations', 'retired', 'dropped'].includes(section)) {
        continue;
      }
      out.push(line);
      continue;
    }
    if (section === 'rulebook') {
      if (/^owner/i.test(line) && !line.startsWith('Owner rules override')) {
        out.push(line.slice(0, 200));
      }
      continue;
    }
    if (['changed files', 'due obligations', 'retired', 'dropped'].includes(section)) {
      if (line === 'none') continue;
      if (!out[out.length - 1]?.startsWith(`## ${section}`)) out.push(`## ${section}`);
      out.push(line.slice(0, 180));
      continue;
    }
    if (section === 'pinned decisions') {
      if (pinnedCount >= 6) continue;
      pinnedCount += 1;
      out.push(line.length > 140 ? line.slice(0, 137) + '...' : line);
      continue;
    }
    out.push(line.length > 220 ? line.slice(0, 217) + '...' : line);
  }
  const joined = out.join('\n');
  return joined.length > maxChars ? joined.slice(0, maxChars - 3).trimEnd() + '...' : joined;
}

function capsuleRun(cfg, source, cwd) {
  if (!cfg.vestigeBin || !fs.existsSync(cfg.vestigeBin)) return '';
  try {
    const r = spawnSync(
      cfg.vestigeBin,
      ['--data-dir', cfg.dataDir, 'capsule', '--harness', 'claude-code', '--event', 'session-start'],
      {
        input: JSON.stringify({ source: source || 'startup', cwd: cwd || process.cwd() }),
        encoding: 'utf8',
        timeout: cfg.capsuleTimeoutMs,
        maxBuffer: 1024 * 1024,
        windowsHide: true,
      }
    );
    const out = String(r.stdout || '').trim();
    if (!out) return '';
    const parsed = JSON.parse(out);
    const raw = (parsed && parsed.hookSpecificOutput && parsed.hookSpecificOutput.additionalContext) || '';
    return compactCapsule(raw, 1100);
  } catch (e) {
    log(cfg, { capsuleError: String(e && e.message) });
    return '';
  }
}

async function openRpc(cfg, timeoutMs = 2000) {
  const { rpc } = shim(cfg, 'memory.cjs');
  const mc = cfg.memCfg || {};
  if (!mc.binary) return null;
  const client = rpc(mc.binary, mc.binaryArgs || [], timeoutMs, {
    ...process.env,
    VESTIGE_HTTP_ENABLED: '0',
    VESTIGE_DASHBOARD_ENABLED: 'false',
    VESTIGE_AUTOPILOT_ENABLED: '0',
  });
  await client.ready();
  return client;
}

async function intentionsStartContext(cfg, maxChars = 750) {
  let client = null;
  try {
    client = await openRpc(cfg, Math.max(6000, cfg.memoryDeadlineMs || 7000));
    if (!client) return '';
    const [flat, portfolio] = await Promise.all([
      client.tool('session', { action: 'intention', params: { action: 'list', filter_status: 'active', limit: 8 } }).catch(() => null),
      client.tool('session', { action: 'intention', params: { action: 'graph', command: { action: 'portfolio' } } }).catch(() => null),
    ]);
    const lines = [];
    const flatItems = (flat && Array.isArray(flat.intentions)) ? flat.intentions : [];
    for (const item of flatItems) {
      const id = item.id || 'int';
      const prio = item.priority ? ` [${item.priority}]` : '';
      const due = item.deadline ? ` (due ${String(item.deadline).slice(0, 10)})` : '';
      const desc = String(item.description || item.content || '').replace(/\s+/g, ' ').trim().slice(0, 180);
      if (desc) lines.push(`- ${id}${prio}${due}: ${desc}`);
    }
    const plans = (portfolio && portfolio.result && Array.isArray(portfolio.result.plans)) ? portfolio.result.plans : [];
    const activePlans = plans.filter(p => p.lifecycle === 'active');
    for (const p of activePlans.slice(0, 4)) {
      const desc = String(p.description || '').replace(/\s+/g, ' ').trim().slice(0, 140);
      lines.push(`- plan:${p.id} [${p.priority || 'normal'}/${p.readiness || 'ready'}]: ${desc}`);
    }
    const pendingAttn = portfolio && portfolio.result && portfolio.result.attention && portfolio.result.attention.pending_count;
    if (typeof pendingAttn === 'number' && pendingAttn > 0) {
      lines.push(`- attention queue: ${pendingAttn} pending item(s)`);
    }
    if (!lines.length) return '';
    const body = `Active Intentions & Plans:\n${lines.join('\n')}`;
    return body.length > maxChars ? body.slice(0, maxChars - 3).trimEnd() + '...' : body;
  } catch (e) {
    log(cfg, { intentionsError: String(e && e.message) });
    return '';
  } finally {
    if (client) try { client.close(); } catch {}
  }
}

async function getLatestTimelineId(cfg) {
  let client = null;
  try {
    client = await openRpc(cfg, 1500);
    if (!client) return null;
    const res = await client.tool('session', { action: 'status', view: 'timeline', limit: 1 });
    const firstDay = res && Array.isArray(res.timeline) && res.timeline[0];
    const firstMem = firstDay && Array.isArray(firstDay.memories) && firstDay.memories[0];
    return (firstMem && typeof firstMem.id === 'string') ? firstMem.id : null;
  } catch {
    return null;
  } finally {
    if (client) try { client.close(); } catch {}
  }
}

function extractMcpResponseFromTranscript(transcriptPath) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return null;
  try {
    const st = fs.statSync(transcriptPath);
    const len = Math.min(st.size, 65536);
    const fd = fs.openSync(transcriptPath, 'r');
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, st.size - len);
    fs.closeSync(fd);
    const lines = buf.toString('utf8').split(/\r?\n/).filter(Boolean);
    if (st.size > len && lines.length > 0) lines.shift();
    for (let i = lines.length - 1; i >= Math.max(0, lines.length - 6); i--) {
      let row;
      try { row = JSON.parse(lines[i]); } catch { continue; }
      if (row && row.type === 'GENERIC' && typeof row.content === 'string') {
        const contentLines = row.content.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
        for (let j = contentLines.length - 1; j >= 0; j--) {
          if (contentLines[j].startsWith('{')) {
            try {
              const obj = JSON.parse(contentLines[j]);
              if (obj && typeof obj === 'object') return obj;
            } catch {}
          }
        }
      }
      const m = /\\?"nodeId\\?"\s*:\s*\\?"(mem-[0-9a-f]{16})\\?"/i.exec(lines[i]);
      if (m) return { nodeId: m[1], success: true };
    }
  } catch {}
  return null;
}

function extractContext(out) {
  return (out && out.hookSpecificOutput && out.hookSpecificOutput.additionalContext) || '';
}

function compactContext(text, seenGhost = null, maxChars = 1400) {
  if (!text) return '';
  const seenBackfill = new Set();
  const seenLines = new Set();
  const out = [];
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith('Vestige local memory candidates.')) {
      const hdr = "Vestige memory summaries (call recall(handle='<id>') for full text):";
      if (!seenLines.has(hdr)) { seenLines.add(hdr); out.push(hdr); }
      continue;
    }
    if (line.startsWith('Save new durable facts using remember')) continue;
    if (line.startsWith('Came back: ')) {
      const m = /^Came back:\s*(mem-[0-9a-f]+)/i.exec(line);
      if (m && seenBackfill.has(m[1])) continue;
      if (m) seenBackfill.add(m[1]);
    }
    if (line.startsWith('GhostLink: compose ') && seenGhost) {
      const key = crypto.createHash('sha256').update(line).digest('hex').slice(0, 12);
      if (seenGhost.includes(key)) continue;
      seenGhost.push(key);
      if (seenGhost.length > 24) seenGhost.shift();
    }
    if (line.startsWith('{')) {
      try {
        const obj = JSON.parse(line);
        if (obj.backfill_receipt && obj.memory_id) {
          if (seenBackfill.has(obj.memory_id)) continue;
          seenBackfill.add(obj.memory_id);
          const whyStr = String(obj.reason || '').trim();
          const whyClean = whyStr.length > 140 ? whyStr.slice(0, 137).trimEnd() + '...' : whyStr;
          const rsbLine = `- [RSB Backfill ${obj.backfill_receipt}] ${obj.memory_id}${whyClean ? ': ' + whyClean : ''}`;
          if (!seenLines.has(rsbLine)) {
            seenLines.add(rsbLine);
            out.push(rsbLine);
          }
          continue;
        }
        if (obj.id && typeof obj.text === 'string') {
          const dateStr = obj.date ? ` [${obj.date}]` : '';
          const cleanText = obj.text.replace(/\r?\n/g, ' ').replace(/\s+/g, ' ').trim();
          const summary = cleanText.length > 200 ? cleanText.slice(0, 197).trimEnd() + '...' : cleanText;
          let entityStr = '';
          if (Array.isArray(obj.entities) && obj.entities.length) {
            const entClean = obj.entities.slice(0, 3).map(e => String(e).replace(/^file:|^tag:|^commit:/, '')).join(', ');
            entityStr = ` (${entClean})`;
          }
          const bullet = `- ${obj.id}${dateStr}: ${summary}${entityStr}`;
          if (!seenLines.has(bullet)) {
            seenLines.add(bullet);
            out.push(bullet);
          }
          continue;
        }
      } catch {}
    }
    const clipped = line.length > 360 ? line.slice(0, 357).trimEnd() + '...' : line;
    if (seenLines.has(clipped)) continue;
    seenLines.add(clipped);
    out.push(clipped);
  }
  while (out.length && out[out.length - 1] === "Vestige memory summaries (call recall(handle='<id>') for full text):") {
    out.pop();
  }
  let currentLen = 0;
  const packed = [];
  for (const line of out) {
    const nextLen = (packed.length > 0 ? 1 : 0) + line.length;
    if (packed.length > 0 && currentLen + nextLen > maxChars) {
      break;
    }
    packed.push(line);
    currentLen += nextLen;
  }
  while (packed.length && packed[packed.length - 1] === "Vestige memory summaries (call recall(handle='<id>') for full text):") {
    packed.pop();
  }
  return packed.join('\n');
}

async function memoryContext(cfg, payload, hookEvent, maxChars = 1400) {
  try {
    const memory = shim(cfg, 'memory.cjs');
    const out = await memory.run({
      event: hookEvent,
      client: 'claude',
      configFile: cfg.memoryConfig,
      input: payload,
      deadlineMs: cfg.memoryDeadlineMs,
    });
    const raw = (out && out.hookSpecificOutput && out.hookSpecificOutput.additionalContext) || '';
    return compactContext(raw, null, maxChars);
  } catch (e) {
    log(cfg, { memoryError: String(e && e.message) });
    return '';
  }
}

function turnState(session) {
  const safeSession = session || 'default';
  const file = path.join(STATE_DIR, 'turns', crypto.createHash('sha256').update(safeSession).digest('hex').slice(0, 24) + '.json');
  let s = {}; try { s = JSON.parse(fs.readFileSync(file, 'utf8')); } catch {}
  return {
    s,
    save: (v) => {
      try {
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, JSON.stringify(v));
      } catch {}
    },
  };
}

function queuePending(session, ctx) {
  if (!ctx || !session) return;
  const t = turnState(session);
  t.s.pending = ((t.s.pending || '') + '\n' + ctx).slice(-4000);
  t.save(t.s);
}

// ---------------------------------------------------------------- handlers
async function preToolUse(cfg, input) {
  const payload = toClaudePayload(input, 'PreToolUse');
  const deny = (reason) => {
    log(cfg, { tool: payload.tool_name, decision: 'deny', reason: reason.slice(0, 400) });
    return { decision: 'deny', reason };
  };
  if (!fs.existsSync(cfg.gate)) {
    return deny(`Operator gate missing at ${cfg.gate}. Fail-closed: a missing gate is never permission to run.`);
  }
  const r = spawnSync(cfg.python, ['-I', cfg.gate, 'hook', '--source', 'antigravity'], {
    input: JSON.stringify(payload),
    encoding: 'utf8',
    timeout: cfg.gateTimeoutMs,
    maxBuffer: 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    windowsHide: true,
  });
  const stderr = String(r.stderr || '').trim();
  if (r.error) return deny(`Operator gate did not answer (${r.error.code || r.error.message}). Fail-closed.`);
  if (r.status !== 0) return deny(stderr || `Operator gate exited ${r.status}. Fail-closed.`);

  const sgOut = saveGateRun(cfg, payload, 'PreToolUse');
  const sgHso = (sgOut && sgOut.hookSpecificOutput) || {};
  if (sgHso.permissionDecision === 'deny') {
    return deny(sgHso.permissionDecisionReason || 'Vestige Save-Gate denied this git commit.');
  }

  if (payload.session_id && /^mcp__vestige__(?:remember|curate)$/i.test(String(payload.tool_name || ''))) {
    const preId = await getLatestTimelineId(cfg);
    if (preId) {
      const t = turnState(payload.session_id);
      t.s.preWriteId = preId;
      t.save(t.s);
    }
  }

  const rfxOut = await reflexRun(cfg, payload, 'PreToolUse');
  const hso = (rfxOut && rfxOut.hookSpecificOutput) || {};
  if (hso.permissionDecision === 'deny') {
    return deny(hso.permissionDecisionReason || 'Vestige Reflex denied this call.');
  }
  const ctx = extractContext(rfxOut);
  if (ctx) queuePending(payload.session_id, ctx);

  log(cfg, { tool: payload.tool_name, decision: 'allow', note: stderr.slice(0, 200) });
  if (hso.updatedInput && input.toolCall && input.toolCall.name === 'call_mcp_tool') {
    return { decision: 'allow', overwrite: { Arguments: hso.updatedInput } };
  }
  if (hso.updatedInput && /^mcp_vestige/i.test(String(input.toolCall && input.toolCall.name))) {
    return { decision: 'allow', overwrite: hso.updatedInput };
  }
  return { decision: 'allow' };
}

async function postToolUse(cfg, input) {
  const payload = toClaudePayload(input, 'PostToolUse');
  const failed = typeof input.error === 'string' && input.error.length > 0;
  payload.tool_response = failed ? { exit_code: 1, stderr: input.error } : { exit_code: 0 };
  if (failed) payload.error = input.error;

  const toolStr = String(payload.tool_name || '');
  if (!failed && /^mcp__vestige__/i.test(toolStr)) {
    const mcpObj = extractMcpResponseFromTranscript(input.transcriptPath);
    if (mcpObj) {
      payload.tool_response = { exit_code: 0, ...mcpObj };
    }
    if (/^mcp__vestige__(?:remember|curate)$/i.test(toolStr)) {
      const t = payload.session_id ? turnState(payload.session_id) : null;
      let nodeId = payload.tool_response.nodeId || payload.tool_response.echoNodeId || payload.tool_response.replayOf;
      if (!nodeId && t) {
        const latestId = await getLatestTimelineId(cfg);
        if (latestId && latestId !== t.s.preWriteId) nodeId = latestId;
      }
      if (nodeId) {
        payload.tool_response = { ...payload.tool_response, exit_code: 0, nodeId, success: true };
      }
      if (t) {
        t.s.rememberedThisTurn = true;
        t.s.unanchoredSourceEdits = false;
        t.s.preWriteId = nodeId || t.s.preWriteId;
        t.save(t.s);
      }
    }
  }

  if (!failed && (payload.tool_name === 'Edit' || payload.tool_name === 'Write') &&
      !isScratchOrArtifactEdit(payload.tool_input && payload.tool_input.file_path, payload.artifact_dir) &&
      payload.session_id) {
    const t = turnState(payload.session_id);
    t.s.unanchoredSourceEdits = true;
    t.save(t.s);
  }

  if (!failed && payload.tool_name === 'Bash') {
    postCommitReceiptRun(cfg, payload);
  }
  saveGateRun(cfg, payload, 'PostToolUse');
  const rfxPayload = (payload.tool_name === 'Edit' || payload.tool_name === 'Write') &&
    isScratchOrArtifactEdit(payload.tool_input && payload.tool_input.file_path, payload.artifact_dir)
    ? null
    : payload;
  const rfxOut = rfxPayload ? await reflexRun(cfg, rfxPayload, failed ? 'PostToolUseFailure' : 'PostToolUse') : {};
  const ctx = extractContext(rfxOut);
  if (ctx) queuePending(payload.session_id, ctx);
  log(cfg, { tool: payload.tool_name, failed, queued: ctx.length, nodeId: payload.tool_response.nodeId });
  return {};
}

async function sessionStart(cfg, input) {
  const payload = toClaudePayload(input, 'SessionStart');
  payload.source = input.source || 'startup';
  turnoverSessionStart(cfg, payload);

  const [intentions, mem, rfxOut] = await Promise.all([
    intentionsStartContext(cfg, 750),
    memoryContext(cfg, payload, 'session', 800),
    reflexRun(cfg, payload, 'SessionStart'),
  ]);
  const capsule = capsuleRun(cfg, payload.source, payload.cwd);
  const rfx = compactContext(extractContext(rfxOut), null, 450);

  const text = compactContext([intentions, capsule, rfx, mem].filter(Boolean).join('\n\n'), null, 2400);
  if (payload.session_id && intentions) {
    const t = turnState(payload.session_id);
    t.s.intentionsShown = true;
    t.save(t.s);
  }
  log(cfg, { capsuleChars: capsule.length, intentionsChars: intentions.length, memChars: mem.length, reflexChars: rfx.length });
  return text ? { injectSteps: [{ systemMessage: { systemMessage: text } }] } : {};
}

async function preInvocation(cfg, input) {
  const payload = toClaudePayload(input, 'UserPromptSubmit');
  if (!payload.session_id) return {};
  const t = turnState(payload.session_id);
  t.s.seenGhost = Array.isArray(t.s.seenGhost) ? t.s.seenGhost : [];
  const prompt = payload.prompt || '';
  const promptKey = crypto.createHash('sha256').update(prompt).digest('hex').slice(0, 16);
  const parts = [];
  if (t.s.promptKey !== promptKey) {
    const sgOut = saveGateRun(cfg, payload, 'UserPromptSubmit');
    const sgCtx = extractContext(sgOut);
    const [mem, rfxOut, firstIntentions] = await Promise.all([
      memoryContext(cfg, payload, 'prompt', 900),
      reflexRun(cfg, payload, 'UserPromptSubmit'),
      !t.s.intentionsShown ? intentionsStartContext(cfg, 550) : Promise.resolve(''),
    ]);
    if (firstIntentions) {
      parts.push(firstIntentions);
      t.s.intentionsShown = true;
    }
    const rfx = compactContext([sgCtx, extractContext(rfxOut)].filter(Boolean).join('\n'), t.s.seenGhost, 550);
    if (rfx) parts.push(rfx);
    if (mem) parts.push(mem);
    t.s.promptKey = promptKey;
    t.s.rememberedThisTurn = false;
    t.s.unanchoredSourceEdits = false;
  } else {
    const rfxOut = await reflexRun(cfg, payload, 'PostToolBatch');
    const rfx = compactContext(extractContext(rfxOut), t.s.seenGhost, 550);
    if (rfx) parts.push(rfx);
  }
  if (t.s.pending) {
    parts.unshift(compactContext(t.s.pending.trim(), t.s.seenGhost, 550));
    t.s.pending = '';
  }
  t.save(t.s);
  const finalBody = compactContext(parts.filter(Boolean).join('\n\n'), null, 1600);
  log(cfg, { injected: finalBody.length });
  return finalBody ? { injectSteps: [{ systemMessage: { systemMessage: finalBody } }] } : {};
}

async function postInvocation(cfg, input) {
  const payload = toClaudePayload(input, 'PostToolBatch');
  if (!payload.session_id) return {};
  const rfxOut = await reflexRun(cfg, payload, 'PostToolBatch');
  const ctx = extractContext(rfxOut);
  if (ctx) queuePending(payload.session_id, ctx);
  return {};
}

function extractRelativeFiles(text) {
  const rel = new Set();
  const re = /(?:crates|packages|benchmarks|scripts|apps|hooks|docs|operator)\/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+/g;
  const clean = String(text || '').replace(/\\/g, '/');
  for (const m of clean.matchAll(re)) {
    rel.add(m[0]);
    if (rel.size >= 6) break;
  }
  return [...rel];
}

function readLatestTurnFromTranscript(transcriptPath) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return null;
  try {
    const st = fs.statSync(transcriptPath);
    const len = Math.min(st.size, 131072);
    const fd = fs.openSync(transcriptPath, 'r');
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, st.size - len);
    fs.closeSync(fd);
    const rawLines = buf.toString('utf8').split(/\r?\n/).filter(Boolean);
    if (st.size > len && rawLines.length > 0) rawLines.shift();
    let userText = '';
    let assistantText = '';
    let alreadyRemembered = false;
    let sawToolActivity = false;
    for (let i = rawLines.length - 1; i >= 0; i--) {
      let row;
      try { row = JSON.parse(rawLines[i]); } catch { continue; }
      if (row.type === 'PLANNER_RESPONSE') {
        if (!assistantText && typeof row.content === 'string' && row.content.trim()) {
          assistantText = row.content.trim();
        }
        if (Array.isArray(row.tool_calls) && row.tool_calls.length > 0) {
          sawToolActivity = true;
          for (const tc of row.tool_calls) {
            const n = String((tc && tc.name) || '');
            const a = (tc && tc.args && typeof tc.args === 'object') ? tc.args : {};
            const toolNameClean = String(a.ToolName || '').replace(/^"+|"+$/g, '');
            if (/remember/i.test(n) || (n === 'call_mcp_tool' && /^remember$/i.test(toolNameClean))) {
              alreadyRemembered = true;
            }
          }
        }
      } else if (row.type === 'USER_INPUT') {
        userText = String(row.content || '');
        break;
      }
    }
    return { userText, assistantText, alreadyRemembered, sawToolActivity };
  } catch {
    return null;
  }
}

function sanitizeAutoContent(text) {
  return String(text || '')
    .replace(/file:\/\/\/[A-Za-z]:\/[^\s)]+\//g, '')
    .replace(/[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s"'`]+(?:[\\/][^\\/\s"'`]+)*?[\\/](?=(?:crates|packages|benchmarks|scripts|apps|hooks|docs|operator)[\\/])/gi, '')
    .replace(/\r?\n{3,}/g, '\n\n')
    .trim();
}

function extractAutoSaveCandidate({ userText = '', assistantText = '', alreadyRemembered = false, sawToolActivity = false } = {}) {
  if (alreadyRemembered) return null;

  const cleanUser = String(userText || '').replace(/<\/?USER_REQUEST>/gi, '').replace(/<ADDITIONAL_METADATA>[\s\S]*/i, '').trim();
  const directiveMatch = /\b(?:from now on|always|never|rule:|decision:|invariant:)\b[^.!?\n]{18,}/i.exec(cleanUser);

  const body = sanitizeAutoContent(assistantText);
  if (!body || body.length < 70) {
    if (directiveMatch && directiveMatch[0].trim().length >= 30) {
      const content = sanitizeAutoContent(directiveMatch[0]).slice(0, 800);
      const files = extractRelativeFiles(content);
      return {
        node_type: 'rule',
        content,
        files,
        entities: files.slice(0, 4).map(f => `file:${f}`),
        tags: ['auto-hook', 'auto-directive'],
      };
    }
    return null;
  }

  const hasSignalHeading = /(?:^|\n|\*\*|\b)(?:root cause|decision|invariant|fix|fixed|verified|finding|milestone|regression|scorecard|shipped|installed)\b/i.test(body);
  const hasMetrics = /(?:R@\d+\s*=|All-Facts@\d+|Evidence Recall@\d+|Turn@\d+|Sess(?:ion)?@\d+|\b\d+\s*\/\s*\d+\s*pass(?:ed|ing)?\b|\bpass\s+\d+\s*\|\s*fail\s+0\b)/i.test(body);
  const files = extractRelativeFiles(body);

  if (!((sawToolActivity && hasSignalHeading) || (hasMetrics && (files.length > 0 || hasSignalHeading)))) {
    return null;
  }

  let nodeType = 'note';
  if (/\b(?:decision|invariant|from now on|rule:)\b/i.test(body)) nodeType = 'decision';
  else if (/\b(?:verified|shipped|milestone|installed|scorecard|R@\d+|All-Facts@)\b/i.test(body)) nodeType = 'milestone';
  else if (/\b(?:root cause|fixed|regression)\b/i.test(body)) nodeType = 'event';

  const summary = body.slice(0, 1200).trim();
  return {
    node_type: nodeType,
    content: summary,
    files,
    entities: files.slice(0, 4).map(f => `file:${f}`),
    tags: ['auto-hook', ...(hasMetrics ? ['benchmark'] : ['auto-verified'])],
  };
}

async function autoSaveAtStop(cfg, input, payload) {
  try {
    const sid = payload.session_id;
    if (!sid) return null;
    const t = turnState(sid);
    if (t.s.rememberedThisTurn || t.s.unanchoredSourceEdits) return null;

    const turn = readLatestTurnFromTranscript(input.transcriptPath) || {
      userText: input.lastUserInput || '',
      assistantText: input.lastAssistantMessage || '',
      alreadyRemembered: false,
      sawToolActivity: true,
    };
    const candidate = extractAutoSaveCandidate(turn);
    if (!candidate) return null;

    const sig = crypto.createHash('sha256').update(`${candidate.node_type}:${candidate.content}`).digest('hex').slice(0, 16);
    t.s.seenAutoSaves = Array.isArray(t.s.seenAutoSaves) ? t.s.seenAutoSaves : [];
    if (t.s.seenAutoSaves.includes(sig)) return null;

    const client = await openRpc(cfg, Math.min(2500, cfg.memoryDeadlineMs || 2500));
    if (!client) return null;
    try {
      const res = await client.tool('remember', {
        action: 'save',
        node_type: candidate.node_type,
        content: candidate.content,
        tags: candidate.tags,
        ...(candidate.entities.length ? { entities: candidate.entities } : {}),
        ...(candidate.files.length ? { files: candidate.files } : {}),
        ...(payload.cwd && fs.existsSync(payload.cwd) ? { repo_path: payload.cwd } : {}),
        intent_id: `auto-stop-${sig}`,
      });
      t.s.seenAutoSaves.push(sig);
      if (t.s.seenAutoSaves.length > 32) t.s.seenAutoSaves.shift();
      t.save(t.s);
      log(cfg, { autoSaved: res.nodeId || res.echoNodeId || sig, tags: candidate.tags });
      return res;
    } finally {
      client.close();
    }
  } catch (e) {
    log(cfg, { autoSaveError: String(e && e.message) });
    return null;
  }
}

async function stop(cfg, input) {
  const payload = toClaudePayload(input, 'Stop');
  await memoryContext(cfg, payload, 'turn');
  const autoRes = await autoSaveAtStop(cfg, input, payload);
  const autoId = autoRes && (autoRes.nodeId || autoRes.echoNodeId || autoRes.replayOf);
  if (autoId) {
    const settlePayload = {
      ...payload,
      hook_event_name: 'PostToolUse',
      tool_name: 'mcp__vestige__remember',
      tool_input: { action: 'save' },
      tool_response: { exit_code: 0, nodeId: autoId, success: true },
    };
    saveGateRun(cfg, settlePayload, 'PostToolUse');
    await reflexRun(cfg, settlePayload, 'PostToolUse');
  }

  const rfxOut = await reflexRun(cfg, payload, 'Stop');
  const sgOut = saveGateRun(cfg, payload, 'Stop');
  const blockReason = (rfxOut && rfxOut.decision === 'block' && rfxOut.reason)
    || (sgOut && sgOut.decision === 'block' && sgOut.reason)
    || '';

  const t = payload.session_id ? turnState(payload.session_id) : null;
  const alreadyBlockedPrompt = t && t.s.promptKey && t.s.stopBlockedPromptKey === t.s.promptKey;
  if (
    blockReason &&
    input.terminationReason === 'NO_TOOL_CALL' &&
    input.continuationActive !== true &&
    !alreadyBlockedPrompt
  ) {
    if (t && t.s.promptKey) {
      t.s.stopBlockedPromptKey = t.s.promptKey;
      t.save(t.s);
    }
    return { decision: 'continue', reason: blockReason };
  }
  return {};
}

async function main() {
  const raw = readStdin();
  let input = {}; try { input = JSON.parse(raw || '{}'); } catch {}
  let cfg = null;
  try { cfg = loadConfig(); } catch (e) {
    if (EVENT === 'PreToolUse') return { decision: 'deny', reason: `Vestige bridge config unreadable (${e.message}). Fail-closed.` };
    return {};
  }
  log(cfg, { raw: cfg.debug ? raw.slice(0, 1500) : undefined });
  switch (EVENT) {
    case 'PreToolUse': return preToolUse(cfg, input);
    case 'PostToolUse': return postToolUse(cfg, input);
    case 'SessionStart': return sessionStart(cfg, input);
    case 'PreInvocation': return preInvocation(cfg, input);
    case 'PostInvocation': return postInvocation(cfg, input);
    case 'Stop': return stop(cfg, input);
    default: return {};
  }
}

module.exports = {
  toClaudeTool,
  toClaudePayload,
  isGitCommitCommand,
  extractGitRepoFromCommand,
  normalizePathForSaveGate,
  isScratchOrArtifactEdit,
  extractMcpResponseFromTranscript,
  readLatestTurnFromTranscript,
  compactCapsule,
  extractAutoSaveCandidate,
  autoSaveAtStop,
};
if (require.main === module) {
  main().then(
    (out) => { process.stdout.write(JSON.stringify(out || {}) + '\n', () => process.exit(0)); },
    (e) => {
      if (EVENT === 'PreToolUse') {
        process.stdout.write(JSON.stringify({ decision: 'deny', reason: `Vestige bridge crashed (${e && e.message}). Fail-closed.` }) + '\n', () => process.exit(0));
      } else {
        process.stdout.write('{}\n', () => process.exit(0));
      }
    }
  );
}

#!/usr/bin/env node
'use strict';
// Command shim for `session(action="reflex")`, for hosts without Claude Code's `mcp_tool`
// hook type (Codex, Cursor, Gemini) and for Claude Code's SessionStart at launch,
// which fires before MCP servers connect. It forwards the hook's own fields to
// the Vestige server, which answers with the context to show, and maps that back
// to the host's hook JSON. Fail-open: any error prints {} and exits 0.
const fs = require('fs');
const path = require('path');
const { rpc } = require('./memory.cjs');

const VESTIGE_LEAVES = new Set([
  'session', 'remember', 'recall', 'causal_walk', 'curate', 'precedent',
  'memory', 'smart_ingest', 'ghostlink',
]);

function vestigeLeaf(toolName) {
  if (typeof toolName !== 'string' || !toolName) return null;
  const low = toolName.trim();
  if (low.startsWith('mcp__')) {
    const parts = low.split('__');
    if (parts.length >= 3 && parts[1].includes('vestige')) return parts.slice(2).join('__');
    return null;
  }
  if (low.startsWith('vestige:') || low.startsWith('vestige.')) {
    const leaf = low.slice(8);
    return VESTIGE_LEAVES.has(leaf) ? leaf : null;
  }
  return VESTIGE_LEAVES.has(low) ? low : null;
}

function loadLearnedCoachRules(configFile) {
  if (!configFile) return [];
  try {
    const p = path.join(path.dirname(configFile), 'coach-rules.json');
    if (!fs.existsSync(p)) return [];
    const data = JSON.parse(fs.readFileSync(p, 'utf8'));
    return Array.isArray(data.rules) ? data.rules : [];
  } catch {
    return [];
  }
}

function saveLearnedCoachRule(configFile, rule) {
  if (!configFile || !rule) return null;
  try {
    const p = path.join(path.dirname(configFile), 'coach-rules.json');
    const existing = loadLearnedCoachRules(configFile);
    if (existing.some(r => r.leaf === rule.leaf && r.badKey === rule.badKey && r.goodKey === rule.goodKey)) {
      return existing.find(r => r.leaf === rule.leaf && r.badKey === rule.badKey && r.goodKey === rule.goodKey);
    }
    const assigned = { id: `CC-L${String(existing.length + 1).padStart(3, '0')}`, ...rule };
    existing.push(assigned);
    fs.writeFileSync(p, JSON.stringify({ rules: existing }, null, 2) + '\n');
    return assigned;
  } catch {
    return null;
  }
}

// Deterministic Call Coach (#520): zero model calls, never overwrites explicit agent values.
function coachPreToolUse(toolName, toolInput, learnedRules = []) {
  const leaf = vestigeLeaf(toolName);
  if (!leaf || !toolInput || typeof toolInput !== 'object' || Array.isArray(toolInput)) return null;
  const next = { ...toolInput };
  const notes = [];
  const applied = [];
  let estimatedSavedChars = 0;
  const has = key => Object.prototype.hasOwnProperty.call(next, key) && next[key] !== undefined;

  // CC-010: Refuse deprecated/unavailable delete or purge mutations before the call.
  if ((leaf === 'curate' || leaf === 'memory') && (next.action === 'delete' || next.action === 'purge')) {
    return {
      deny: true,
      ruleIds: ['CC-010'],
      reason: 'Call Coach [CC-010]: Strata is append-only; delete/purge are unavailable_in_4_0. Use curate(action="suppress", ...) to hide a record permanently or curate(action="supersede", ...) to replace a belief.',
    };
  }

  // CC-011: Refuse empty remember content before wasting a round trip.
  if (leaf === 'remember' && (!next.action || next.action === 'save') &&
      typeof next.content === 'string' && !next.content.trim() && !Array.isArray(next.items)) {
    return {
      deny: true,
      ruleIds: ['CC-011'],
      reason: 'Call Coach [CC-011]: remember `content` is empty or whitespace-only. Provide non-empty `content` or `items`.',
    };
  }

  // CC-002: GhostLink / curate weave note > 500 chars refused before call with exact count & limit.
  const isWeave = (leaf === 'ghostlink' && (!next.mode || next.mode === 'weave')) ||
    (leaf === 'curate' && next.action === 'weave');
  if (isWeave && typeof next.note === 'string') {
    const charCount = Array.from(next.note).length;
    if (charCount > 500) {
      const trimmed = Array.from(next.note).slice(0, 500).join('');
      const corrected = JSON.stringify({ ...next, note: trimmed });
      return {
        deny: true,
        ruleIds: ['CC-002'],
        reason: `Call Coach [CC-002]: weave note is ${charCount} characters (limit: 500). Shorten \`note\` to <= 500 chars; corrected call: ${corrected}`,
      };
    }
  }

  // CC-001: Rename alias `memory_id` (or `node_id` on curate) -> `memoryId` when absent.
  if (leaf === 'memory' || leaf === 'curate' || leaf === 'ghostlink' || leaf === 'remember') {
    if (!has('memoryId')) {
      if (has('memory_id')) {
        next.memoryId = next.memory_id;
        delete next.memory_id;
        applied.push('CC-001');
        notes.push('[CC-001] renamed alias memory_id -> memoryId (pass memoryId directly)');
      } else if (leaf === 'curate' && ['demote', 'promote', 'suppress'].includes(next.action) && has('node_id')) {
        next.memoryId = next.node_id;
        delete next.node_id;
        applied.push('CC-001');
        notes.push('[CC-001] renamed alias node_id -> memoryId');
      }
    }
  }

  // CC-006: On recall, rename `id` or `tag` -> `handle` when `handle` and `query` are absent.
  if (leaf === 'recall' && !has('handle') && !has('query')) {
    const srcKey = has('id') ? 'id' : (has('tag') ? 'tag' : null);
    if (srcKey) {
      next.handle = next[srcKey];
      delete next[srcKey];
      applied.push('CC-006');
      notes.push(`[CC-006] renamed alias ${srcKey} -> handle`);
    }
  }

  // CC-007: On remember, rename `text` or `message` -> `content` when `content` and `items` are absent.
  if (leaf === 'remember' && !has('content') && !has('items')) {
    const srcKey = has('text') ? 'text' : (has('message') ? 'message' : null);
    if (srcKey) {
      next.content = next[srcKey];
      delete next[srcKey];
      applied.push('CC-007');
      notes.push(`[CC-007] renamed alias ${srcKey} -> content`);
    }
  }

  // CC-008: On causal_walk, rename `handle`, `memory_id`, or `memoryId` -> `from_handle`.
  if (leaf === 'causal_walk' && !has('from_handle') && !has('start_points')) {
    const srcKey = has('handle') ? 'handle' : (has('memory_id') ? 'memory_id' : (has('memoryId') ? 'memoryId' : null));
    if (srcKey) {
      next.from_handle = next[srcKey];
      delete next[srcKey];
      applied.push('CC-008');
      notes.push(`[CC-008] renamed alias ${srcKey} -> from_handle`);
    }
  }

  // CC-009: On precedent (get/inspect/receipt/replay/default), rename `id` or `digest` -> `receipt_id`.
  if (leaf === 'precedent' && (!has('action') || next.action === 'get' || next.action === 'inspect' || next.query === 'receipt' || next.query === 'replay') && !has('receipt_id')) {
    const srcKey = has('id') ? 'id' : (has('digest') ? 'digest' : null);
    if (srcKey) {
      next.receipt_id = next[srcKey];
      delete next[srcKey];
      applied.push('CC-009');
      notes.push(`[CC-009] renamed alias ${srcKey} -> receipt_id`);
    }
  }

  // CC-012: On session, normalize `tool_name` -> `tool` for view='tools' and default missing `action`.
  if (leaf === 'session') {
    let sessionTouched = false;
    if (next.view === 'tools' && !has('tool') && has('tool_name')) {
      next.tool = next.tool_name;
      delete next.tool_name;
      sessionTouched = true;
    }
    if (!has('action') && (Object.keys(next).length === 0 || has('view'))) {
      next.action = 'status';
      sessionTouched = true;
    }
    if (sessionTouched) {
      applied.push('CC-012');
      notes.push('[CC-012] normalized session status/tool selector');
    }
  }

  // CC-003: Bounded recall defaults when omitted by the agent (never changes explicit values).
  if (leaf === 'recall' && !has('handle') && !has('id') && !has('tag')) {
    let touched = false;
    if (next.limit === undefined) {
      next.limit = 10;
      touched = true;
      estimatedSavedChars += 2400;
    }
    if (next.detail_level === undefined) {
      next.detail_level = 'summary';
      touched = true;
      estimatedSavedChars += 3600;
    }
    if (touched) {
      applied.push('CC-003');
      notes.push('[CC-003] defaulted recall to limit=10, detail_level="summary" (set limit or detail_level explicitly to override)');
    }
  }

  // CC-004: Compact output profile on smart_ingest when omitted.
  if (leaf === 'smart_ingest' && next.output_profile === undefined) {
    next.output_profile = 'compact';
    applied.push('CC-004');
    estimatedSavedChars += 1800;
    notes.push('[CC-004] defaulted smart_ingest output_profile="compact" (set output_profile="standard" to override)');
  }

  // CC-005: Bounded scan_limit on causal_walk when omitted.
  if (leaf === 'causal_walk' && next.scan_limit === undefined) {
    next.scan_limit = 500;
    applied.push('CC-005');
    estimatedSavedChars += 2000;
    notes.push('[CC-005] defaulted causal_walk scan_limit=500 (set scan_limit explicitly to override)');
  }

  // Learned alias rules from PostToolUseFailure provenance
  for (const r of learnedRules) {
    if (r && r.leaf === leaf && r.badKey && r.goodKey &&
        Object.prototype.hasOwnProperty.call(next, r.badKey) &&
        !Object.prototype.hasOwnProperty.call(next, r.goodKey)) {
      next[r.goodKey] = next[r.badKey];
      delete next[r.badKey];
      applied.push(r.id);
      notes.push(`[${r.id} derived_from:${r.derived_from}] renamed ${r.badKey} -> ${r.goodKey}`);
    }
  }

  if (!applied.length) return null;
  return {
    deny: false,
    ruleIds: applied,
    updatedInput: next,
    note: `Vestige Call Coach: ${notes.join('; ')}.`,
    estimatedSavedChars,
  };
}

function extractLearnedAliasFailure(toolName, errorText, derivedFrom) {
  const leaf = vestigeLeaf(toolName);
  if (!leaf || typeof errorText !== 'string' || !errorText) return null;
  const m = /unknown field [`'"]?([A-Za-z0-9_]+)[`'"]?,?\s*expected (?:one of )?[^]*?[`'"]?([A-Za-z0-9_]+)[`'"]?/i.exec(errorText) ||
    /invalid parameter [`'"]?([A-Za-z0-9_]+)[`'"]?;?\s*did you mean [`'"]?([A-Za-z0-9_]+)[`'"]?/i.exec(errorText) ||
    /unknown variant [`'"]?([A-Za-z0-9_]+)[`'"]?,?\s*expected (?:one of )?[^]*?[`'"]?([A-Za-z0-9_]+)[`'"]?/i.exec(errorText);
  if (!m || m[1] === m[2]) return null;
  const codeMatch = /\b(E_[A-Z0-9_]+|-32\d{3}|invalid_params)\b/.exec(errorText);
  return {
    leaf,
    code: codeMatch ? codeMatch[1] : 'invalid_params',
    badKey: m[1],
    goodKey: m[2],
    derived_from: derivedFrom || 'unknown-receipt',
  };
}

function coachReplay(rows, learnedRules = []) {
  const byRule = {};
  const ensure = id => (byRule[id] || (byRule[id] = { rule: id, prevented_failures: 0, rewritten_calls: 0, saved_chars: 0, receipts: [] }));
  for (const row of rows || []) {
    const tool = row.tool_name || row.tool || '';
    const ti = row.tool_input || row.arguments || {};
    const receipt = row.receipt_id || row.receipt || row.call_id || row.tool_use_id || 'unreceipted';
    const res = coachPreToolUse(tool, ti, learnedRules);
    if (!res) continue;
    for (const rid of res.ruleIds) {
      const b = ensure(rid);
      if (res.deny || ['CC-001', 'CC-006', 'CC-007', 'CC-008', 'CC-009', 'CC-012'].includes(rid) || rid.startsWith('CC-L')) b.prevented_failures += 1;
      if (!res.deny) b.rewritten_calls += 1;
      b.saved_chars += res.estimatedSavedChars || 0;
      b.receipts.push(receipt);
    }
  }
  return { total_calls: (rows || []).length, rules: Object.values(byRule) };
}

// Host event name -> the canonical (Claude Code) name reflex speaks.
const CANONICAL = {
  gemini: { BeforeTool: 'PreToolUse', AfterTool: 'PostToolUse', BeforeAgent: 'UserPromptSubmit', AfterAgent: 'Stop', SessionStart: 'SessionStart', PreCompress: 'PreCompact' },
  cursor: { preToolUse: 'PreToolUse', postToolUse: 'PostToolUse', postToolUseFailure: 'PostToolUseFailure', beforeSubmitPrompt: 'UserPromptSubmit', sessionStart: 'SessionStart', subagentStart: 'SubagentStart', preCompact: 'PreCompact', stop: 'Stop' },
};
const CURSOR_CONTEXT = new Set(['sessionStart', 'postToolUse', 'postToolUseFailure']);

function canonical(client, event) {
  return (CANONICAL[client] && CANONICAL[client][event]) || event;
}

function argumentsFor(client, event, input) {
  const pick = (...keys) => { for (const k of keys) if (typeof input[k] === 'string' && input[k]) return input[k]; return ''; };
  const toolInput = input.tool_input && typeof input.tool_input === 'object' ? input.tool_input : {};
  const response = input.tool_response && typeof input.tool_response === 'object' ? input.tool_response : {};
  const exitCode = typeof response.exit_code === 'number'
    ? response.exit_code
    : (typeof response.status === 'number' ? response.status : (response.error || response.stderr ? 1 : 0));
  
  let canonEvent = canonical(client, event);
  if (canonEvent === 'PostToolUse' && exitCode !== 0) {
    canonEvent = 'PostToolUseFailure';
  }

  return {
    action: 'hook', host: client, format: 'plain',
    event: canonEvent,
    session: pick('session_id', 'sessionId', 'conversation_id'),
    tool: pick('tool_name', 'toolName'),
    call: pick('tool_use_id', 'toolUseId', 'call_id'),
    agent: pick('agent_id', 'agentId'),
    command: typeof toolInput.command === 'string' ? toolInput.command : pick('command'),
    source: pick('source'),
    error: pick('error', 'error_message') || (typeof response.stderr === 'string' ? response.stderr : ''),
    interrupt: input.is_interrupt === true ? 'true' : '',
    file: typeof toolInput.file_path === 'string'
      ? toolInput.file_path
      : (typeof toolInput.path === 'string'
        ? toolInput.path
        : (typeof toolInput.TargetFile === 'string'
          ? toolInput.TargetFile
          : (typeof toolInput.notebook_path === 'string'
            ? toolInput.notebook_path
            : (Array.isArray(toolInput.edits) && toolInput.edits[0] && typeof toolInput.edits[0].file_path === 'string'
              ? toolInput.edits[0].file_path
              : '')))),
    cwd: pick('cwd'),
    summary: pick('compact_summary'),
    trigger: pick('trigger'),
    stdout: typeof response.stdout === 'string' ? response.stdout : '',
    stderr: typeof response.stderr === 'string' ? response.stderr : '',
    interrupted: response.interrupted === true ? 'true' : '',
    background: toolInput.run_in_background === true ? 'true' : '',
    exit_code: exitCode,
    // Enforcement inputs. The prompt and final message are read for literal
    // directive phrases and exact handles; Reflex never copies them into the ledger.
    prompt: pick('prompt', 'user_prompt'),
    last_assistant_message: pick('last_assistant_message', 'lastAssistantMessage'),
    stop_hook_active: input.stop_hook_active === true ? 'true' : '',
    tool_input: toolInput,
    tool_response: input.tool_response === undefined ? null : input.tool_response,
  };
}

// The decision Reflex returned, in each host's own shape. A block or deny is
// only ever passed through; the shim never invents one.
function hostOutput(client, event, context, decision = {}) {
  const canon = canonical(client, event);
  const reason = typeof decision.reason === 'string' ? decision.reason : '';
  const stopBlock = decision.decision === 'block' && reason && (canon === 'Stop' || canon === 'SubagentStop');
  const deny = decision.permissionDecision === 'deny' && canon === 'PreToolUse';
  const denyReason = typeof decision.permissionDecisionReason === 'string' ? decision.permissionDecisionReason : 'Vestige ledger denied this call.';
  const updated = !deny && canon === 'PreToolUse' && decision.updatedInput && typeof decision.updatedInput === 'object' ? decision.updatedInput : null;
  if (client === 'cursor') {
    const out = context && CURSOR_CONTEXT.has(event) ? { additional_context: context } : {};
    if (stopBlock) out.followup_message = reason;
    if (deny) Object.assign(out, { permission: 'deny', user_message: denyReason, agent_message: denyReason });
    if (updated) Object.assign(out, { permission: 'allow', updated_input: updated });
    return out;
  }
  if (client === 'gemini') {
    const out = context ? { hookSpecificOutput: { hookEventName: event, additionalContext: context } } : {};
    if (stopBlock) Object.assign(out, { decision: 'block', reason });
    if (deny) Object.assign(out, { decision: 'deny', reason: denyReason });
    if (updated) {
      out.hookSpecificOutput = {
        ...(out.hookSpecificOutput || { hookEventName: event }),
        updatedInput: updated,
      };
    }
    return out;
  }
  const out = context ? { hookSpecificOutput: { hookEventName: event, additionalContext: context } } : {};
  if (stopBlock) Object.assign(out, { decision: 'block', reason });
  if (deny) out.hookSpecificOutput = { ...(out.hookSpecificOutput || { hookEventName: event }), permissionDecision: 'deny', permissionDecisionReason: denyReason };
  if (updated) {
    out.hookSpecificOutput = {
      ...(out.hookSpecificOutput || { hookEventName: event }),
      updatedInput: updated,
      ...(typeof decision.permissionDecisionReason === 'string' && decision.permissionDecisionReason
        ? { permissionDecision: 'allow', permissionDecisionReason: decision.permissionDecisionReason }
        : {}),
    };
  }
  return out;
}

async function run({ client, configFile, eventFlag, input, deadlineMs = 450 }) {
  let transport;
  try {
    const config = configFile && fs.existsSync(configFile) ? JSON.parse(fs.readFileSync(configFile, 'utf8')) : {};
    const event = (typeof input.hook_event_name === 'string' && input.hook_event_name) || eventFlag || '';
    const args = argumentsFor(client, event, input);
    if (config.enforce === false) args.enforce = 'off';
    const learned = loadLearnedCoachRules(configFile);
    const coached = args.event === 'PreToolUse' && args.enforce !== 'off'
      ? coachPreToolUse(args.tool, args.tool_input, learned)
      : null;
    if (coached && coached.deny) {
      return hostOutput(client, event || 'PreToolUse', '', {
        permissionDecision: 'deny',
        permissionDecisionReason: coached.reason,
      });
    }
    if (coached && coached.updatedInput) {
      args.tool_input = coached.updatedInput;
    }
    if (!args.event || !args.session || !config.binary) {
      if (coached && coached.updatedInput) {
        return hostOutput(client, event || 'PreToolUse', coached.note, {
          updatedInput: coached.updatedInput,
          permissionDecisionReason: coached.note,
        });
      }
      return {};
    }
    transport = rpc(config.binary, config.binaryArgs || [], deadlineMs,
      { ...process.env, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' });
    await transport.ready();
    const answer = await transport.tool('session', { action: 'reflex', params: args });
    let context = answer && typeof answer.context === 'string' ? answer.context : '';
    if (coached && coached.note) {
      context = context ? `${context}\n${coached.note}` : coached.note;
    }
    
    // Auto-Correction Prompt Append + Call Coach Learning on Non-Zero Exit / Failure
    if (args.event === 'PostToolUseFailure' || args.exit_code !== 0) {
      const handle = answer && (answer.memory || answer.failure_id);
      if (typeof handle === 'string' && /^mem-[0-9a-f]{16}$/.test(handle)) {
        const promptAppend = `⚡ VESTIGE REFLEX: Command failed. Inspect its recorded evidence with causal_walk(action='walk', start_points=[{kind:'logged_write',node_id:'${handle}'}], promote=false).`;
        context = (context ? context + '\n\n' : '') + promptAppend;
      }
      const learnedCandidate = extractLearnedAliasFailure(args.tool, args.error, handle || args.call);
      if (learnedCandidate) {
        saveLearnedCoachRule(configFile, learnedCandidate);
      }
    }

    const mergedDecision = { ...(answer || {}) };
    if (coached && coached.updatedInput && !mergedDecision.updatedInput) {
      mergedDecision.updatedInput = coached.updatedInput;
      if (!mergedDecision.permissionDecisionReason) mergedDecision.permissionDecisionReason = coached.note;
    }
    return hostOutput(client, event, context, mergedDecision);
  } catch {
    return {};
  } finally {
    if (transport) transport.close();
  }
}

async function main() {
  const argv = process.argv.slice(2); const value = flag => { const i = argv.indexOf(flag); return i >= 0 ? argv[i + 1] : undefined; };
  if (argv.includes('--replay')) {
    const logPath = value('--replay');
    const rows = logPath && fs.existsSync(logPath)
      ? fs.readFileSync(logPath, 'utf8').split(/\r?\n/).filter(Boolean).map(l => JSON.parse(l))
      : [];
    const report = coachReplay(rows, loadLearnedCoachRules(value('--config')));
    process.stdout.write(JSON.stringify(report, null, 2) + '\n', () => process.exit(0));
    return;
  }
  const envDeadline = Number(process.env.VESTIGE_HOOK_DEADLINE_MS);
  const cliCeiling = Number.isFinite(envDeadline) && envDeadline > 0 ? Math.max(1500, envDeadline + 100) : 1500;
  const failOpen = setTimeout(() => { process.stdout.write('{}\n', () => process.exit(0)); }, cliCeiling);
  let raw = ''; for await (const chunk of process.stdin) { raw += chunk; if (raw.length > 262144) break; }
  let input = {}; try { input = JSON.parse(raw || '{}'); } catch {}
  const out = await run({ client: value('--client') || 'claude', configFile: value('--config'), eventFlag: value('--event'), input, deadlineMs: Number.isFinite(envDeadline) && envDeadline > 0 ? envDeadline : undefined });
  clearTimeout(failOpen); process.stdout.write(JSON.stringify(out) + '\n', () => process.exit(0));
}

module.exports = { run, argumentsFor, hostOutput, canonical, coachPreToolUse, coachReplay, extractLearnedAliasFailure };
if (require.main === module) main().catch(() => { process.stdout.write('{}\n', () => process.exit(0)); });

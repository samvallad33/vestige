#!/usr/bin/env node
'use strict';
// Command shim for `session(action="reflex")`, for hosts without Claude Code's `mcp_tool`
// hook type (Codex, Cursor, Gemini) and for Claude Code's SessionStart at launch,
// which fires before MCP servers connect. It forwards the hook's own fields to
// the Vestige server, which answers with the context to show, and maps that back
// to the host's hook JSON. Fail-open: any error prints {} and exits 0.
const fs = require('fs');
const { rpc } = require('./memory.cjs');

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
    file: typeof toolInput.file_path === 'string' ? toolInput.file_path : (typeof toolInput.path === 'string' ? toolInput.path : ''),
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
  if (client === 'cursor') {
    const out = context && CURSOR_CONTEXT.has(event) ? { additional_context: context } : {};
    if (stopBlock) out.followup_message = reason;
    if (deny) Object.assign(out, { permission: 'deny', user_message: denyReason, agent_message: denyReason });
    return out;
  }
  if (client === 'gemini') {
    const out = context ? { hookSpecificOutput: { hookEventName: event, additionalContext: context } } : {};
    if (stopBlock) Object.assign(out, { decision: 'block', reason });
    if (deny) Object.assign(out, { decision: 'deny', reason: denyReason });
    return out;
  }
  const out = context ? { hookSpecificOutput: { hookEventName: event, additionalContext: context } } : {};
  if (stopBlock) Object.assign(out, { decision: 'block', reason });
  if (deny) out.hookSpecificOutput = { ...(out.hookSpecificOutput || { hookEventName: event }), permissionDecision: 'deny', permissionDecisionReason: denyReason };
  return out;
}

async function run({ client, configFile, eventFlag, input, deadlineMs = 450 }) {
  let transport;
  try {
    const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
    const event = (typeof input.hook_event_name === 'string' && input.hook_event_name) || eventFlag || '';
    const args = argumentsFor(client, event, input);
    if (config.enforce === false) args.enforce = 'off';
    if (!args.event || !args.session) return {};
    transport = rpc(config.binary, config.binaryArgs || [], deadlineMs,
      { ...process.env, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' });
    await transport.ready();
    const answer = await transport.tool('session', { action: 'reflex', params: args });
    let context = answer && typeof answer.context === 'string' ? answer.context : '';
    
    // Auto-Correction Prompt Append on Non-Zero Exit Code
    if (args.event === 'PostToolUseFailure' || args.exit_code !== 0) {
      const handle = answer && (answer.memory || answer.failure_id);
      if (typeof handle === 'string' && /^mem-[0-9a-f]{16}$/.test(handle)) {
        const promptAppend = `⚡ VESTIGE REFLEX: Command failed. Inspect its recorded evidence with causal_walk(action='walk', start_points=[{kind:'logged_write',node_id:'${handle}'}], promote=false).`;
        context = (context ? context + '\n\n' : '') + promptAppend;
      }
    }

    return hostOutput(client, event, context, answer || {});
  } catch {
    return {};
  } finally {
    if (transport) transport.close();
  }
}

async function main() {
  const argv = process.argv.slice(2); const value = flag => { const i = argv.indexOf(flag); return i >= 0 ? argv[i + 1] : undefined; };
  const envDeadline = Number(process.env.VESTIGE_HOOK_DEADLINE_MS);
  const cliCeiling = Number.isFinite(envDeadline) && envDeadline > 0 ? Math.max(1500, envDeadline + 100) : 1500;
  const failOpen = setTimeout(() => { process.stdout.write('{}\n', () => process.exit(0)); }, cliCeiling);
  let raw = ''; for await (const chunk of process.stdin) { raw += chunk; if (raw.length > 262144) break; }
  let input = {}; try { input = JSON.parse(raw || '{}'); } catch {}
  const out = await run({ client: value('--client') || 'claude', configFile: value('--config'), eventFlag: value('--event'), input, deadlineMs: Number.isFinite(envDeadline) && envDeadline > 0 ? envDeadline : undefined });
  clearTimeout(failOpen); process.stdout.write(JSON.stringify(out) + '\n', () => process.exit(0));
}

module.exports = { run, argumentsFor, hostOutput, canonical };
if (require.main === module) main().catch(() => { process.stdout.write('{}\n', () => process.exit(0)); });

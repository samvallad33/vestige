#!/usr/bin/env node
'use strict';
// Command shim for the `reflex` tool, for hosts without Claude Code's `mcp_tool`
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
const CURSOR_CONTEXT = new Set(['sessionStart', 'postToolUse']);

function canonical(client, event) {
  return (CANONICAL[client] && CANONICAL[client][event]) || event;
}

function argumentsFor(client, event, input) {
  const pick = (...keys) => { for (const k of keys) if (typeof input[k] === 'string' && input[k]) return input[k]; return ''; };
  const toolInput = input.tool_input && typeof input.tool_input === 'object' ? input.tool_input : {};
  const response = input.tool_response && typeof input.tool_response === 'object' ? input.tool_response : {};
  return {
    action: 'hook', host: client, format: 'plain',
    event: canonical(client, event),
    session: pick('session_id', 'sessionId', 'conversation_id'),
    tool: pick('tool_name', 'toolName'),
    call: pick('tool_use_id', 'toolUseId', 'call_id'),
    agent: pick('agent_id', 'agentId'),
    command: typeof toolInput.command === 'string' ? toolInput.command : pick('command'),
    source: pick('source'),
    error: pick('error'),
    interrupt: input.is_interrupt === true ? 'true' : '',
    file: typeof toolInput.file_path === 'string' ? toolInput.file_path : '',
    cwd: pick('cwd'),
    summary: pick('compact_summary'),
    trigger: pick('trigger'),
    stdout: typeof response.stdout === 'string' ? response.stdout : '',
    stderr: typeof response.stderr === 'string' ? response.stderr : '',
    interrupted: response.interrupted === true ? 'true' : '',
    background: toolInput.run_in_background === true ? 'true' : '',
  };
}

function hostOutput(client, event, context) {
  if (!context) return {};
  if (client === 'cursor') return CURSOR_CONTEXT.has(event) ? { additional_context: context } : {};
  return { hookSpecificOutput: { hookEventName: event, additionalContext: context } };
}

async function run({ client, configFile, eventFlag, input, deadlineMs = 450 }) {
  let transport;
  try {
    const config = JSON.parse(fs.readFileSync(configFile, 'utf8'));
    const event = (typeof input.hook_event_name === 'string' && input.hook_event_name) || eventFlag || '';
    const args = argumentsFor(client, event, input);
    if (!args.event || !args.session) return {};
    transport = rpc(config.binary, config.binaryArgs || [], deadlineMs,
      { ...process.env, VESTIGE_HTTP_ENABLED: '0', VESTIGE_DASHBOARD_ENABLED: 'false', VESTIGE_AUTOPILOT_ENABLED: '0' });
    await transport.ready();
    const answer = await transport.tool('reflex', args);
    return hostOutput(client, event, answer && typeof answer.context === 'string' ? answer.context : '');
  } catch {
    return {};
  } finally {
    if (transport) transport.close();
  }
}

async function main() {
  const argv = process.argv.slice(2); const value = flag => { const i = argv.indexOf(flag); return i >= 0 ? argv[i + 1] : undefined; };
  const failOpen = setTimeout(() => { process.stdout.write('{}\n'); process.exit(0); }, 1500);
  let raw = ''; for await (const chunk of process.stdin) { raw += chunk; if (raw.length > 262144) break; }
  let input = {}; try { input = JSON.parse(raw || '{}'); } catch {}
  const out = await run({ client: value('--client') || 'claude', configFile: value('--config'), eventFlag: value('--event'), input });
  clearTimeout(failOpen); process.stdout.write(JSON.stringify(out) + '\n');
}

module.exports = { run, argumentsFor, hostOutput, canonical };
if (require.main === module) main().catch(() => { process.stdout.write('{}\n'); process.exit(0); });

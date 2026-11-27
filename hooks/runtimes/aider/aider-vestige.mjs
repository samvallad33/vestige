#!/usr/bin/env node
'use strict';
/**
 * Vestige Lifecycle Wrapper for Aider.
 * Fires SessionStart -> UserPromptSubmit (optional --message) -> aider -> post-commit receipt + Stop.
 *
 * Usage:
 *   node aider-vestige.mjs [any aider flags]
 */
const { spawnSync } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const bridge = [
  process.env.VESTIGE_BRIDGE_PATH,
  path.join(os.homedir(), '.vestige', 'memory', 'hooks', 'vestige-bridge.cjs'),
  path.resolve(__dirname, '../../../packages/vestige-init/hooks/vestige-bridge.cjs'),
].find(p => p && fs.existsSync(p));

const sessionId = `aider-${crypto.randomUUID().slice(0, 12)}`;
const cwd = process.cwd();

function hook(event, extra = {}) {
  if (!bridge) return {};
  const r = spawnSync(process.execPath, [bridge, '--runtime', 'aider', '--event', event], {
    cwd,
    input: JSON.stringify({ session_id: sessionId, cwd, ...extra }),
    encoding: 'utf8',
    timeout: 10000,
    windowsHide: true,
  });
  try { return JSON.parse(String(r.stdout || '{}').trim().split(/\r?\n/).pop() || '{}'); } catch { return {}; }
}

const args = process.argv.slice(2);
const startCtx = hook('SessionStart');
if (startCtx.additionalContext) {
  process.stderr.write(`[Vestige SessionStart]\n${startCtx.additionalContext}\n`);
}
const msgIdx = args.findIndex(a => a === '--message' || a === '-m');
if (msgIdx >= 0 && args[msgIdx + 1]) {
  const promptCtx = hook('UserPromptSubmit', { prompt: args[msgIdx + 1] });
  if (promptCtx.additionalContext) {
    process.stderr.write(`[Vestige Recall]\n${promptCtx.additionalContext}\n`);
  }
}

const aiderBin = process.env.AIDER_BIN || 'aider';
const run = spawnSync(aiderBin, args, { stdio: 'inherit', cwd });
hook('PostToolUse', { tool_name: 'Bash', tool_input: { command: 'git commit -m aider' }, tool_response: { exit_code: 0 } });
hook('Stop', { prompt: msgIdx >= 0 ? args[msgIdx + 1] : '' });
process.exit(run.status ?? 0);

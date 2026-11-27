#!/usr/bin/env node
'use strict';
/**
 * Vestige Multi-Runtime Hook Packager & Installer
 *
 * Manages all 24 supported 2026 AI coding agent runtimes under `hooks/runtimes/<runtime>/`:
 *   - `--list`                    : List all runtimes, hook paths, and supported lifecycle events
 *   - `--emit`                    : Emit / refresh every runtime's drop-in config folder + READMEs
 *   - `--runtime <id[,id...]> `   : Install Vestige bridge + native hooks into target runtime(s)
 *   - `--all`                     : Install into all detected runtimes on this machine
 *   - `--project <dir>`           : Install project-scoped hooks inside <dir> where supported
 *   - `--guard`                   : Enable fail-closed Operator Gate + precedent enforcement
 *   - `--dry-run`                 : Preview without writing files
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const ROOT = __dirname;
const REPO_ROOT = path.resolve(ROOT, '..', '..');
const INIT_HOOKS = path.join(REPO_ROOT, 'packages', 'vestige-init', 'hooks');
const INIT_LIB = path.join(REPO_ROOT, 'packages', 'vestige-init', 'lib');

const PORTABLE_BRIDGE = '~/.vestige/memory/hooks/vestige-bridge.cjs';

function quoteCmd(parts, platform = process.platform) {
  return parts.map((p) => {
    const s = String(p);
    if (platform === 'win32') return /[\s()]/.test(s) ? `"${s}"` : s;
    return /[^A-Za-z0-9_./:-]/.test(s) ? `'${s.replace(/'/g, `'\"'\"'`)}'` : s;
  }).join(' ');
}

function bridgeCmd(runtime, event, nodeBin, bridgePath, platform, extraFlags = []) {
  const cmd = quoteCmd([nodeBin, bridgePath, '--runtime', runtime, '--event', event, ...extraFlags], platform);
  return cmd;
}

function claudeStyleConfig(runtime, events, cmdFn, timeoutSec = 15, extra = {}) {
  const hooks = {};
  for (const [ev, matcher] of events) {
    const entry = {
      ...(matcher ? { matcher } : {}),
      hooks: [{ type: 'command', command: cmdFn(runtime, ev), timeout: timeoutSec }],
    };
    hooks[ev] = [entry];
  }
  return { ...extra, hooks };
}

function flatHooksConfig(runtime, events, cmdFn, timeoutSec = 15) {
  const out = {};
  for (const [ev, matcher] of events) {
    out[ev] = [{
      ...(matcher ? { matcher } : {}),
      hooks: [{ type: 'command', command: cmdFn(runtime, ev), timeout: timeoutSec }],
    }];
  }
  return out;
}

function cursorStyleConfig(runtime, events, cmdFn, failClosedEvents = new Set(['preToolUse', 'beforeShellExecution', 'beforeMCPExecution', 'beforeReadFile', 'stop']), timeoutSec = 12) {
  const hooks = {};
  for (const [ev, matcher] of events) {
    hooks[ev] = [{
      command: cmdFn(runtime, ev),
      timeout: timeoutSec,
      ...(failClosedEvents.has(ev) ? { failClosed: true } : {}),
      ...(matcher ? { matcher } : {}),
    }];
  }
  return { version: 1, hooks };
}

const RUNTIMES = [
  {
    id: 'codex',
    name: 'OpenAI Codex CLI',
    globalHookFile: (home) => path.join(home, '.codex', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.codex', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Restart Codex or run `/hooks` in Codex CLI to review & approve hook commands.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PermissionRequest', 'PostToolUse', 'PreCompact', 'PostCompact', 'Stop', 'SubagentStart', 'SubagentStop', 'Interrupt', 'SessionEnd'],
    buildConfig: (cmdFn) => claudeStyleConfig('codex', [
      ['SessionStart', 'startup|resume|clear'],
      ['UserPromptSubmit', null],
      ['PreToolUse', '.*'],
      ['PermissionRequest', '.*'],
      ['PostToolUse', '.*'],
      ['PreCompact', null],
      ['PostCompact', null],
      ['Stop', null],
      ['SubagentStart', null],
      ['SubagentStop', null],
      ['Interrupt', null],
      ['SessionEnd', null],
    ], cmdFn, 12),
  },
  {
    id: 'cursor',
    name: 'Cursor Agent',
    globalHookFile: (home) => path.join(home, '.cursor', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.cursor', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Reload Cursor window after installing ~/.cursor/hooks.json or .cursor/hooks.json.',
    events: ['sessionStart', 'beforeSubmitPrompt', 'preToolUse', 'beforeShellExecution', 'beforeMCPExecution', 'beforeReadFile', 'postToolUse', 'postToolUseFailure', 'afterShellExecution', 'afterMCPExecution', 'afterFileEdit', 'subagentStart', 'subagentStop', 'preCompact', 'stop', 'sessionEnd'],
    buildConfig: (cmdFn) => cursorStyleConfig('cursor', [
      ['sessionStart', null],
      ['beforeSubmitPrompt', null],
      ['preToolUse', '.*'],
      ['beforeShellExecution', '*'],
      ['beforeMCPExecution', null],
      ['beforeReadFile', null],
      ['postToolUse', '.*'],
      ['postToolUseFailure', '.*'],
      ['afterShellExecution', null],
      ['afterMCPExecution', null],
      ['afterFileEdit', null],
      ['subagentStart', null],
      ['subagentStop', null],
      ['preCompact', null],
      ['stop', null],
      ['sessionEnd', null],
    ], cmdFn),
  },
  {
    id: 'gemini-cli',
    name: 'Gemini CLI',
    globalHookFile: (home) => path.join(home, '.gemini', 'settings.json'),
    projectHookFile: (dir) => path.join(dir, '.gemini', 'settings.json'),
    configFileName: 'settings.json',
    trustNote: 'Run `/hooks panel` in Gemini CLI to verify active hooks, or link directly with `gemini extensions link hooks/runtimes/gemini-cli`.',
    events: ['SessionStart', 'BeforeAgent', 'BeforeTool', 'AfterTool', 'PreCompress', 'AfterAgent', 'SessionEnd'],
    buildConfig: (cmdFn) => {
      const cfg = claudeStyleConfig('gemini-cli', [
        ['SessionStart', null],
        ['BeforeAgent', null],
        ['BeforeTool', '.*'],
        ['AfterTool', '.*'],
        ['PreCompress', null],
        ['AfterAgent', null],
        ['SessionEnd', null],
      ], (rt, ev) => cmdFn(rt, ev), 15);
      // Gemini CLI specifies hook timeout in milliseconds
      for (const [ev, groups] of Object.entries(cfg.hooks)) {
        for (const g of groups) {
          for (const h of g.hooks) {
            h.name = `vestige-${ev.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase()}`;
            h.timeout = 15000;
          }
        }
      }
      return cfg;
    },
  },
  {
    id: 'claude-code',
    name: 'Claude Code',
    globalHookFile: (home) => path.join(home, '.claude', 'settings.json'),
    projectHookFile: (dir) => path.join(dir, '.claude', 'settings.json'),
    configFileName: 'settings.json',
    trustNote: 'Restart Claude Code or open `/hooks` to inspect registered hooks.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PermissionRequest', 'PostToolUse', 'PostToolUseFailure', 'PreCompact', 'PostCompact', 'SubagentStart', 'SubagentStop', 'Stop', 'SessionEnd'],
    buildConfig: (cmdFn) => claudeStyleConfig('claude-code', [
      ['SessionStart', 'startup|resume|clear|compact'],
      ['UserPromptSubmit', null],
      ['PreToolUse', '*'],
      ['PermissionRequest', '*'],
      ['PostToolUse', '*'],
      ['PostToolUseFailure', '*'],
      ['PreCompact', null],
      ['PostCompact', null],
      ['SubagentStart', null],
      ['SubagentStop', null],
      ['Stop', null],
      ['SessionEnd', null],
    ], cmdFn, 15),
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    globalHookFile: (home) => path.join(home, '.gemini', 'config', 'plugins', 'vestige', 'hooks.json'),
    projectHookFile: null,
    configFileName: 'hooks.json',
    trustNote: 'Restart Antigravity if plugin hooks changed. Use `npx @vestige/init` for full bridge bundle deployment.',
    events: ['SessionStart', 'PreInvocation', 'PreToolUse', 'PostToolUse', 'PostInvocation', 'Stop'],
    buildConfig: () => JSON.parse(fs.readFileSync(path.join(INIT_HOOKS, 'agy-hooks.json'), 'utf8')),
  },
  {
    id: 'copilot',
    name: 'GitHub Copilot CLI & VS Code',
    globalHookFile: (home) => path.join(home, '.copilot', 'hooks', 'vestige.json'),
    projectHookFile: (dir) => path.join(dir, '.github', 'hooks', 'vestige.json'),
    configFileName: 'vestige.json',
    trustNote: 'Requires a trusted workspace in VS Code (`chat.useHooks` enabled; on Copilot CLI ~/.copilot/hooks/*.json loads automatically).',
    events: ['sessionStart', 'userPromptSubmitted', 'preToolUse', 'postToolUse', 'errorOccurred', 'preCompact', 'subagentStart', 'subagentStop', 'agentStop', 'sessionEnd'],
    buildConfig: (cmdFn, meta = {}) => {
      const hooks = {};
      const node = meta.nodeBin || 'node';
      const posixBridge = meta.bridgePosix || PORTABLE_BRIDGE;
      const winBridge = meta.bridgeWin || '%USERPROFILE%\\.vestige\\memory\\hooks\\vestige-bridge.cjs';
      for (const ev of ['sessionStart', 'userPromptSubmitted', 'preToolUse', 'postToolUse', 'errorOccurred', 'preCompact', 'subagentStart', 'subagentStop', 'agentStop', 'sessionEnd']) {
        hooks[ev] = [{
          type: 'command',
          bash: `${quoteCmd([node, posixBridge, '--runtime', 'copilot', '--event', ev], 'linux')}`,
          powershell: `${quoteCmd([node, winBridge, '--runtime', 'copilot', '--event', ev], 'win32')}`,
          timeoutSec: 15,
        }];
      }
      return { version: 1, hooks };
    },
  },
  {
    id: 'windsurf',
    name: 'Windsurf Cascade',
    globalHookFile: (home) => path.join(home, '.codeium', 'windsurf', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.windsurf', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Restart Windsurf after adding ~/.codeium/windsurf/hooks.json or workspace .windsurf/hooks.json.',
    events: ['pre_user_prompt', 'pre_read_code', 'pre_write_code', 'pre_run_command', 'pre_mcp_tool_use', 'post_write_code', 'post_run_command', 'post_mcp_tool_use', 'post_cascade_response'],
    buildConfig: (cmdFn) => ({
      hooks: Object.fromEntries(
        ['pre_user_prompt', 'pre_read_code', 'pre_write_code', 'pre_run_command', 'pre_mcp_tool_use', 'post_write_code', 'post_run_command', 'post_mcp_tool_use', 'post_cascade_response']
          .map(ev => [ev, [{ command: cmdFn('windsurf', ev), show_output: ev === 'pre_user_prompt' }]])
      ),
    }),
  },
  {
    id: 'devin',
    name: 'Devin Local & CLI',
    globalHookFile: (home) => path.join(home, '.config', 'devin', 'hooks.v1.json'),
    projectHookFile: (dir) => path.join(dir, '.devin', 'hooks.v1.json'),
    configFileName: 'hooks.v1.json',
    trustNote: 'Place in `.devin/hooks.v1.json` in your repo or merge under `hooks` in `~/.config/devin/config.json`.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreCompact', 'Stop'],
    buildConfig: (cmdFn) => flatHooksConfig('devin', [
      ['SessionStart', null],
      ['UserPromptSubmit', null],
      ['PreToolUse', '.*'],
      ['PostToolUse', '.*'],
      ['PreCompact', null],
      ['Stop', null],
    ], cmdFn, 15),
  },
  {
    id: 'opencode',
    name: 'OpenCode',
    globalHookFile: (home) => path.join(home, '.config', 'opencode', 'plugins', 'vestige', 'index.js'),
    projectHookFile: (dir) => path.join(dir, '.opencode', 'plugins', 'vestige', 'index.js'),
    configFileName: 'index.js',
    customDirFiles: ['index.js', 'package.json'],
    trustNote: 'OpenCode auto-loads JS/TS plugins from `.opencode/plugins/` or `~/.config/opencode/plugins/`.',
    events: ['experimental.chat.system.transform', 'tool.execute.before', 'tool.execute.after', 'experimental.session.compacting'],
    buildConfig: null,
  },
  {
    id: 'cline',
    name: 'Cline (v3.36+)',
    globalHookFile: (home) => path.join(home, '.cline', 'hooks', 'PreToolUse'),
    projectHookFile: (dir) => path.join(dir, '.clinerules', 'hooks', 'PreToolUse'),
    configFileName: 'PreToolUse',
    customDirFiles: ['TaskStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreCompact', 'TaskComplete'],
    trustNote: 'Copy the 6 executable files (`TaskStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `PreCompact`, `TaskComplete`) into `~/.cline/hooks/` (and `~/Documents/Cline/Hooks/`) or `.clinerules/hooks/`.',
    events: ['TaskStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreCompact', 'TaskComplete'],
    buildConfig: null,
  },
  {
    id: 'kiro',
    name: 'AWS Kiro IDE & CLI',
    globalHookFile: (home) => path.join(home, '.kiro', 'hooks', 'vestige.json'),
    projectHookFile: (dir) => path.join(dir, '.kiro', 'hooks', 'vestige.json'),
    configFileName: 'vestige.json',
    trustNote: 'Uses Kiro 1.x `"version": "v1"` schema in `.kiro/hooks/vestige.json` (`exit 2` blocks PreToolUse).',
    events: ['UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PostFileSave', 'SessionEnd'],
    buildConfig: (cmdFn) => ({
      version: 'v1',
      hooks: [
        { name: 'vestige-prompt-recall', trigger: 'UserPromptSubmit', action: { type: 'command', command: cmdFn('kiro', 'UserPromptSubmit') } },
        { name: 'vestige-pre-tool-guard', trigger: 'PreToolUse', matcher: '.*', action: { type: 'command', command: cmdFn('kiro', 'PreToolUse') } },
        { name: 'vestige-post-tool-reflex', trigger: 'PostToolUse', matcher: '.*', action: { type: 'command', command: cmdFn('kiro', 'PostToolUse') } },
        { name: 'vestige-post-file-save', trigger: 'PostFileSave', matcher: '.*', action: { type: 'command', command: cmdFn('kiro', 'PostFileSave') } },
        { name: 'vestige-session-end', trigger: 'SessionEnd', action: { type: 'command', command: cmdFn('kiro', 'SessionEnd') } },
      ],
    }),
  },
  {
    id: 'factory-droid',
    name: 'Factory AI Droid',
    globalHookFile: (home) => path.join(home, '.factory', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.factory', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Inspect and enable hooks in Factory Droid via `/hooks`.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreCompact', 'Stop'],
    buildConfig: (cmdFn) => flatHooksConfig('factory-droid', [
      ['SessionStart', null],
      ['UserPromptSubmit', null],
      ['PreToolUse', '.*'],
      ['PostToolUse', '.*'],
      ['PreCompact', null],
      ['Stop', null],
    ], cmdFn, 15),
  },
  {
    id: 'roo-code',
    name: 'Roo Code',
    globalHookFile: (home) => path.join(home, '.roo', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.roo', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Merge into `~/.roo/hooks.json` or repo `.roo/hooks.json`.',
    events: ['sessionStart', 'beforeSubmitPrompt', 'preToolUse', 'postToolUse', 'preCompact', 'stop'],
    buildConfig: (cmdFn) => cursorStyleConfig('roo-code', [
      ['sessionStart', null],
      ['beforeSubmitPrompt', null],
      ['preToolUse', '.*'],
      ['postToolUse', '.*'],
      ['preCompact', null],
      ['stop', null],
    ], cmdFn),
  },
  {
    id: 'kilo-code',
    name: 'Kilo Code',
    globalHookFile: (home) => path.join(home, '.kilocode', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.kilocode', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Merge into `~/.kilocode/hooks.json` or repo `.kilocode/hooks.json`.',
    events: ['sessionStart', 'beforeSubmitPrompt', 'preToolUse', 'postToolUse', 'stop'],
    buildConfig: (cmdFn) => cursorStyleConfig('kilo-code', [
      ['sessionStart', null],
      ['beforeSubmitPrompt', null],
      ['preToolUse', '.*'],
      ['postToolUse', '.*'],
      ['stop', null],
    ], cmdFn),
  },
  {
    id: 'goose',
    name: 'Block Goose (v1.41+)',
    globalHookFile: (home) => path.join(home, '.config', 'goose', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.goose', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Enable lifecycle hooks in Goose config (`GOOSE_ENABLE_HOOKS=true`); blocks on exit code 2 + stderr.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'],
    buildConfig: (cmdFn) => claudeStyleConfig('goose', [
      ['SessionStart', null],
      ['UserPromptSubmit', null],
      ['PreToolUse', '.*'],
      ['PostToolUse', '.*'],
      ['Stop', null],
    ], cmdFn, 15),
  },
  {
    id: 'crush',
    name: 'Charm Crush',
    globalHookFile: (home) => path.join(home, '.config', 'crush', 'crush.json'),
    projectHookFile: (dir) => path.join(dir, 'crush.json'),
    configFileName: 'crush.json',
    trustNote: 'Flat `{ name, matcher, command, timeout }` entries inside `crush.json` `hooks`.',
    events: ['PreToolUse', 'PostToolUse'],
    buildConfig: (cmdFn) => ({
      '$schema': 'https://charm.land/crush.json',
      hooks: {
        PreToolUse: [{ name: 'vestige-pre-tool', matcher: '.*', command: cmdFn('crush', 'PreToolUse'), timeout: 12 }],
        PostToolUse: [{ name: 'vestige-post-tool', matcher: '.*', command: cmdFn('crush', 'PostToolUse'), timeout: 10 }],
      },
    }),
  },
  {
    id: 'openhands',
    name: 'OpenHands',
    globalHookFile: (home) => path.join(home, '.openhands', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.openhands', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Place `.openhands/hooks.json` in the workspace or `~/.openhands/hooks.json`.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'],
    buildConfig: (cmdFn) => claudeStyleConfig('openhands', [
      ['SessionStart', null],
      ['UserPromptSubmit', null],
      ['PreToolUse', '*'],
      ['PostToolUse', '*'],
      ['Stop', null],
    ], cmdFn, 15),
  },
  {
    id: 'openclaw',
    name: 'OpenClaw',
    globalHookFile: (home) => path.join(home, '.openclaw', 'extensions', 'vestige-hooks', 'index.js'),
    projectHookFile: null,
    configFileName: 'index.js',
    customDirFiles: ['index.js', 'openclaw.plugin.json'],
    trustNote: 'Install plugin folder into `~/.openclaw/extensions/vestige-hooks/` (`before_tool_call`, `after_tool_call`, `before_prompt_build`, `agent_end`).',
    events: ['before_prompt_build', 'before_tool_call', 'after_tool_call', 'agent_end'],
    buildConfig: null,
  },
  {
    id: 'amazon-q',
    name: 'Amazon Q Developer CLI',
    globalHookFile: (home) => path.join(home, '.aws', 'amazonq', 'cli-agents', 'q_cli_default.json'),
    projectHookFile: (dir) => path.join(dir, '.amazonq', 'cli-agents', 'vestige.json'),
    configFileName: 'q_cli_default.json',
    trustNote: 'Sets `hooks.agentSpawn`, `hooks.userPromptSubmit`, `hooks.preToolUse`, `hooks.postToolUse`, `hooks.stop` in Q CLI default agent.',
    events: ['agentSpawn', 'userPromptSubmit', 'preToolUse', 'postToolUse', 'stop'],
    buildConfig: (cmdFn) => ({
      name: 'q_cli_default',
      hooks: {
        agentSpawn: [{ command: cmdFn('amazon-q', 'agentSpawn'), timeout_ms: 10000 }],
        userPromptSubmit: [{ command: cmdFn('amazon-q', 'userPromptSubmit'), timeout_ms: 10000 }],
        preToolUse: [{ matcher: '*', command: cmdFn('amazon-q', 'preToolUse'), timeout_ms: 12000 }],
        postToolUse: [{ matcher: '*', command: cmdFn('amazon-q', 'postToolUse'), timeout_ms: 10000 }],
        stop: [{ command: cmdFn('amazon-q', 'stop'), timeout_ms: 10000 }],
      },
    }),
  },
  {
    id: 'continue',
    name: 'Continue',
    globalHookFile: (home) => path.join(home, '.continue', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.continue', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Copy into `~/.continue/hooks.json` or repo `.continue/hooks.json`.',
    events: ['sessionStart', 'beforeSubmitPrompt', 'preToolUse', 'postToolUse', 'stop'],
    buildConfig: (cmdFn) => cursorStyleConfig('continue', [
      ['sessionStart', null],
      ['beforeSubmitPrompt', null],
      ['preToolUse', '.*'],
      ['postToolUse', '.*'],
      ['stop', null],
    ], cmdFn),
  },
  {
    id: 'zed',
    name: 'Zed',
    globalHookFile: (home) => path.join(home, '.config', 'zed', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.zed', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Also add `"context_servers": { "vestige": { "command": "vestige-mcp", "args": [] } }` to `~/.config/zed/settings.json`.',
    events: ['sessionStart', 'preToolUse', 'postToolUse', 'stop'],
    buildConfig: (cmdFn) => cursorStyleConfig('zed', [
      ['sessionStart', null],
      ['preToolUse', '.*'],
      ['postToolUse', '.*'],
      ['stop', null],
    ], cmdFn),
  },
  {
    id: 'amp',
    name: 'Sourcegraph Amp',
    globalHookFile: (home) => path.join(home, '.config', 'amp', 'hooks.json'),
    projectHookFile: (dir) => path.join(dir, '.amp', 'hooks.json'),
    configFileName: 'hooks.json',
    trustNote: 'Copy into `~/.config/amp/hooks.json` (`XDG_CONFIG_HOME/amp/hooks.json`).',
    events: ['sessionStart', 'beforeSubmitPrompt', 'preToolUse', 'postToolUse', 'stop'],
    buildConfig: (cmdFn) => cursorStyleConfig('amp', [
      ['sessionStart', null],
      ['beforeSubmitPrompt', null],
      ['preToolUse', '.*'],
      ['postToolUse', '.*'],
      ['stop', null],
    ], cmdFn),
  },
  {
    id: 'aider',
    name: 'Aider CLI Wrapper',
    globalHookFile: null,
    projectHookFile: null,
    configFileName: 'aider-vestige.mjs',
    customDirFiles: ['aider-vestige.mjs'],
    trustNote: 'Run `node hooks/runtimes/aider/aider-vestige.mjs` instead of bare `aider` to wrap session recall, git receipts, and Stop auto-save.',
    events: ['SessionStart', 'UserPromptSubmit', 'PostToolUse', 'Stop'],
    buildConfig: null,
  },
  {
    id: 'python-sdks',
    name: 'Python Agent SDKs (CrewAI, OpenAI Agents SDK, LangChain / LangGraph)',
    globalHookFile: null,
    projectHookFile: null,
    configFileName: 'vestige_hooks.py',
    customDirFiles: ['vestige_hooks.py'],
    trustNote: 'Import `register_crewai_hooks`, `langchain_wrap_tool_call`, or `openai_agents_tool_guard` from `vestige_hooks.py`.',
    events: ['SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'],
    buildConfig: null,
  },
];

function renderReadme(rt) {
  const eventsList = rt.events.map(e => `\`${e}\``).join(', ');
  const gPath = rt.globalHookFile ? rt.globalHookFile('~').replace(/\\/g, '/') : 'Wrapper / SDK import (see below)';
  const pPath = rt.projectHookFile ? rt.projectHookFile('<repo>').replace(/\\/g, '/') : 'N/A (global/SDK)';
  return `# Vestige Hooks — ${rt.name}

One-command install across OSes:

\`\`\`sh
node hooks/runtimes/install.js --runtime ${rt.id}
# Optional: fail-closed Operator Guard (102 STOP rules)
node hooks/runtimes/install.js --runtime ${rt.id} --guard
\`\`\`

- **Config / Adapter File**: \`${rt.configFileName}\`
- **User / Global Path**: \`${gPath}\`
- **Project / Repo Path**: \`${pPath}\`
- **Events Wired**: ${eventsList}
- **Capabilities Included**:
  - Zero-Prompt Recall (\`SessionStart\` / \`UserPromptSubmit\`)
  - Deterministic Call Coach (CC-001..CC-013 alias + default parameter optimization)
  - Operator Guard + \`precedent(query="stops"|"scan_secrets")\` enforcement
  - Anti-Spiral escalation, Chesterton's Fence, Prefrontal Blast-Radius
  - Git \`post-commit\` receipt automation + Stop proof-carrying finish
- **Activation Note**: ${rt.trustNote}
`;
}

function emitAll() {
  for (const rt of RUNTIMES) {
    const dir = path.join(ROOT, rt.id);
    fs.mkdirSync(dir, { recursive: true });
    if (rt.buildConfig) {
      const data = rt.buildConfig(
        (runtime, ev) => bridgeCmd(runtime, ev, 'node', PORTABLE_BRIDGE, 'linux'),
        { nodeBin: 'node', bridgePosix: PORTABLE_BRIDGE, bridgeWin: '%USERPROFILE%\\.vestige\\memory\\hooks\\vestige-bridge.cjs' }
      );
      fs.writeFileSync(path.join(dir, rt.configFileName), JSON.stringify(data, null, 2) + '\n', 'utf8');
    }
    fs.writeFileSync(path.join(dir, 'README.md'), renderReadme(rt), 'utf8');
  }

  const rows = RUNTIMES.map((rt) => {
    const loc = rt.globalHookFile ? rt.globalHookFile('~').replace(/\\/g, '/') : rt.projectHookFile ? rt.projectHookFile('<repo>').replace(/\\/g, '/') : rt.configFileName;
    const sampleEvents = rt.events.slice(0, 3).map(e => '`' + e + '`').join(', ') + (rt.events.length > 3 ? ', ...' : '');
    return '| **' + rt.name + '** | [`' + rt.id + '/`](./' + rt.id + '/) | `' + loc + '` | ' + rt.events.length + ' events (' + sampleEvents + ') |';
  }).join('\n');

  const topReadme = `# Vestige Multi-Runtime Hook Bundles (24 Runtimes · 2026)

Drop-in, schema-verified Vestige hook configurations, native plugins, and SDK adapters for **every major 2026 AI coding runtime**. Every bundle routes through Vestige's single shared engine ([packages/vestige-init/hooks/vestige-bridge.cjs](../../packages/vestige-init/hooks/vestige-bridge.cjs)):

- **Zero-Prompt Recall + Capsule + Active Intentions** on session start & prompt submit
- **Call Coach (CC-001..CC-013)** pre-execution alias & token-budget optimization
- **Operator Guard + \`precedent()\`** fail-closed safety & secret-scan commit gates
- **Anti-Spiral, Chesterton's Fence & Blast-Radius** warnings across long runs
- **Proof-Carrying Finish** at Stop + git \`post-commit\` receipt capture

## Quick Install

\`\`\`sh
# Install into one runtime (globally)
node hooks/runtimes/install.js --runtime codex

# Install into a specific project/repo
node hooks/runtimes/install.js --runtime copilot --project /path/to/repo

# Install into ALL detected runtimes on this machine + enable Operator Guard
node hooks/runtimes/install.js --all --guard
\`\`\`

Or use \`@vestige/init\`:

\`\`\`sh
npx @vestige/init --guard
\`\`\`

## Runtime Matrix (24 Runtimes)

| Runtime | Folder | Target Config Location | Events |
|---|---|---|---|
${rows}
`;
  fs.writeFileSync(path.join(ROOT, 'README.md'), topReadme, 'utf8');
}

function deepMergeHooks(target, incoming) {
  if (!target || typeof target !== 'object' || Array.isArray(target)) return incoming;
  const out = { ...target };
  if (incoming.version !== undefined && out.version === undefined) out.version = incoming.version;
  if (incoming['$schema'] !== undefined && out['$schema'] === undefined) out['$schema'] = incoming['$schema'];
  if (incoming.name !== undefined && out.name === undefined) out.name = incoming.name;

  if (Array.isArray(incoming.hooks) && Array.isArray(out.hooks || [])) {
    // Kiro array schema
    const prev = Array.isArray(out.hooks) ? out.hooks : [];
    const merged = [...prev];
    for (const item of incoming.hooks) {
      if (!merged.some(x => JSON.stringify(x) === JSON.stringify(item))) merged.push(item);
    }
    out.hooks = merged;
    return out;
  }

  const srcHooks = (incoming.hooks && typeof incoming.hooks === 'object') ? incoming.hooks : incoming;
  const dstContainer = incoming.hooks ? (out.hooks && typeof out.hooks === 'object' ? { ...out.hooks } : {}) : out;
  for (const [ev, list] of Object.entries(srcHooks)) {
    if (!Array.isArray(list)) continue;
    const existing = Array.isArray(dstContainer[ev]) ? [...dstContainer[ev]] : [];
    for (const entry of list) {
      if (!existing.some(e => JSON.stringify(e) === JSON.stringify(entry))) existing.push(entry);
    }
    dstContainer[ev] = existing;
  }
  if (incoming.hooks) out.hooks = dstContainer;
  return out;
}

function installSharedRuntime(home, { guard = false, dryRun = false } = {}) {
  const baseHooksDir = path.join(home, '.vestige', 'memory', 'hooks');
  const libDir = path.join(home, '.vestige', 'memory', 'lib');
  if (!dryRun) {
    fs.mkdirSync(baseHooksDir, { recursive: true });
    fs.mkdirSync(libDir, { recursive: true });
    for (const file of ['vestige-bridge.cjs', 'agy-bridge.cjs', 'agy-hooks.json', 'reflex.cjs', 'memory.cjs', 'guard.cjs', 'stop-memory.cjs']) {
      const src = path.join(INIT_HOOKS, file);
      if (fs.existsSync(src)) fs.copyFileSync(src, path.join(baseHooksDir, file));
    }
    fs.copyFileSync(path.join(INIT_LIB, 'memory-triggers.cjs'), path.join(libDir, 'memory-triggers.cjs'));
    const cfgPath = path.join(home, '.vestige', 'memory', 'bridge.json');
    let prev = {};
    try { prev = JSON.parse(fs.readFileSync(cfgPath, 'utf8')); } catch {}
    const nextCfg = {
      ...prev,
      requireGate: Boolean(guard || prev.requireGate),
      hooksDir: baseHooksDir,
      dataDir: prev.dataDir || path.join(home, '.vestige', 'store'),
    };
    fs.writeFileSync(cfgPath, JSON.stringify(nextCfg, null, 2) + '\n', 'utf8');
  }
  return path.join(baseHooksDir, 'vestige-bridge.cjs');
}

function installRuntime(rtId, { home = os.homedir(), projectDir = '', guard = false, dryRun = false, nodeBin = process.execPath, bridgePathOverride = '' } = {}) {
  const rt = RUNTIMES.find(r => r.id === rtId);
  if (!rt) throw new Error(`Unknown runtime "${rtId}". Known: ${RUNTIMES.map(r => r.id).join(', ')}`);

  const bridgePath = bridgePathOverride || installSharedRuntime(home, { guard, dryRun });
  const targetFile = projectDir
    ? (rt.projectHookFile ? rt.projectHookFile(projectDir) : null)
    : (rt.globalHookFile ? rt.globalHookFile(home) : null);

  if (!targetFile) {
    return { id: rt.id, status: 'manual', note: rt.trustNote, written: [] };
  }

  const written = [];
  if (rt.customDirFiles) {
    const destDir = path.dirname(targetFile);
    const srcDir = path.join(ROOT, rt.id);
    if (!dryRun) fs.mkdirSync(destDir, { recursive: true });
    for (const name of rt.customDirFiles) {
      const dest = path.join(destDir, name);
      if (!dryRun) {
        fs.copyFileSync(path.join(srcDir, name), dest);
        if (process.platform !== 'win32' && !name.endsWith('.json')) {
          try { fs.chmodSync(dest, 0o755); } catch {}
        }
      }
      written.push(dest);
    }
    return { id: rt.id, status: dryRun ? 'preview' : 'installed', written };
  }

  const incoming = rt.buildConfig(
    (runtime, ev) => bridgeCmd(runtime, ev, nodeBin, bridgePath, process.platform),
    { nodeBin, bridgePosix: bridgePath, bridgeWin: bridgePath }
  );

  let next = incoming;
  let backup = null;
  if (fs.existsSync(targetFile)) {
    const raw = fs.readFileSync(targetFile, 'utf8');
    if (raw.trim()) {
      const existing = JSON.parse(raw);
      next = deepMergeHooks(existing, incoming);
      if (!dryRun && JSON.stringify(existing) !== JSON.stringify(next)) {
        backup = `${targetFile}.vestige-backup-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`;
        fs.copyFileSync(targetFile, backup);
      }
    }
  }
  if (!dryRun) {
    fs.mkdirSync(path.dirname(targetFile), { recursive: true });
    fs.writeFileSync(targetFile, JSON.stringify(next, null, 2) + '\n', 'utf8');
  }
  written.push(targetFile);
  return { id: rt.id, status: dryRun ? 'preview' : 'installed', written, backup };
}

function detectInstalledRuntimes(home = os.homedir()) {
  return RUNTIMES.filter((rt) => {
    if (!rt.globalHookFile) return false;
    const p = rt.globalHookFile(home);
    return fs.existsSync(p) || fs.existsSync(path.dirname(p)) || fs.existsSync(path.dirname(path.dirname(p)));
  });
}

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.includes('--emit')) {
    emitAll();
    console.log(`Emitted ${RUNTIMES.length} runtime folders under ${ROOT}`);
    process.exit(0);
  }
  if (args.includes('--list')) {
    for (const rt of RUNTIMES) console.log(`${rt.id.padEnd(16)} | ${rt.name.padEnd(36)} | ${rt.events.length} events`);
    process.exit(0);
  }
  const idx = args.indexOf('--runtime');
  const pIdx = args.indexOf('--project');
  const projectDir = pIdx >= 0 ? path.resolve(args[pIdx + 1]) : '';
  const guard = args.includes('--guard');
  const dryRun = args.includes('--dry-run');
  const requested = args.includes('--all')
    ? detectInstalledRuntimes().map(r => r.id)
    : (idx >= 0 ? String(args[idx + 1] || '').split(',').map(s => s.trim()).filter(Boolean) : []);
  if (!requested.length) {
    console.log('Usage: node install.js --runtime <id[,id2]> [--project <dir>] [--guard] [--dry-run] | --all | --list | --emit');
    process.exit(0);
  }
  const results = requested.map(id => installRuntime(id, { projectDir, guard, dryRun }));
  console.log(JSON.stringify(results, null, 2));
}

module.exports = { RUNTIMES, installRuntime, emitAll, detectInstalledRuntimes, deepMergeHooks };

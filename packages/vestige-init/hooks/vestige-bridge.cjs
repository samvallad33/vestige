#!/usr/bin/env node
'use strict';
// Vestige Universal Multi-Runtime Hook Bridge.
//
// Translates native lifecycle & tool-call payloads from 23+ AI agent runtimes into Vestige's
// canonical kernel contract (Operator Gate + Rust Reflex + Zero-Prompt Recall + Call Coach
// CC-001..CC-013 + 6-Tool Cognitive Autopilot + Silent Watcher + Post-Commit Receipt + Auto-Save),
// and formats decisions back into each host's exact schema and exit-code contract.
//
// Usage:
//   node vestige-bridge.cjs --runtime <runtime> [--event <event>] [--config <bridge.json>] [--exit-code]
//
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { spawnSync, spawn } = require('child_process');
const agy = require('./agy-bridge.cjs');

const argv = process.argv.slice(2);
const flag = (name, dflt = '') => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : dflt;
};
const HAS_EXIT_CODE_FLAG = argv.includes('--exit-code');
const RUNTIME_RAW = flag('--runtime', process.env.VESTIGE_RUNTIME || 'claude-code');
const EVENT_FLAG = flag('--event', '');
const CONFIG_FLAG = flag('--config', '');
const HERE = __dirname;

const RUNTIME_ALIASES = {
  agy: 'antigravity',
  antigravity: 'antigravity',
  claude: 'claude-code',
  'claude-code': 'claude-code',
  xcode: 'claude-code',
  codex: 'codex',
  'codex-cli': 'codex',
  cursor: 'cursor',
  gemini: 'gemini-cli',
  'gemini-cli': 'gemini-cli',
  copilot: 'copilot',
  vscode: 'copilot',
  'vs-code': 'copilot',
  'copilot-cli': 'copilot',
  windsurf: 'windsurf',
  devin: 'devin',
  opencode: 'opencode',
  cline: 'cline',
  kiro: 'kiro',
  factory: 'factory-droid',
  'factory-droid': 'factory-droid',
  droid: 'factory-droid',
  'roo-code': 'roo-code',
  roo: 'roo-code',
  'kilo-code': 'kilo-code',
  kilo: 'kilo-code',
  goose: 'goose',
  crush: 'crush',
  openhands: 'openhands',
  openclaw: 'openclaw',
  'amazon-q': 'amazon-q',
  q: 'amazon-q',
  continue: 'continue',
  zed: 'zed',
  amp: 'amp',
  aider: 'aider',
  python: 'python-sdks',
  'python-sdks': 'python-sdks',
  crewai: 'python-sdks',
  langchain: 'python-sdks',
  'openai-agents': 'python-sdks',
};

function normalizeRuntime(raw) {
  const key = String(raw || 'claude-code').trim().toLowerCase();
  return RUNTIME_ALIASES[key] || key;
}

const EVENT_MAP = {
  // Canonical / Claude / Codex / Devin / Factory / OpenHands / Goose / Crush / Kiro
  SessionStart: 'SessionStart',
  sessionStart: 'SessionStart',
  session_start: 'SessionStart',
  TaskStart: 'SessionStart',
  agentSpawn: 'SessionStart',
  UserPromptSubmit: 'UserPromptSubmit',
  userPromptSubmitted: 'UserPromptSubmit',
  userPromptSubmit: 'UserPromptSubmit',
  BeforeAgent: 'UserPromptSubmit',
  beforeSubmitPrompt: 'UserPromptSubmit',
  pre_user_prompt: 'UserPromptSubmit',
  PreToolUse: 'PreToolUse',
  preToolUse: 'PreToolUse',
  BeforeTool: 'PreToolUse',
  beforeShellExecution: 'PreToolUse',
  beforeMCPExecution: 'PreToolUse',
  beforeReadFile: 'PreToolUse',
  pre_read_code: 'PreToolUse',
  pre_write_code: 'PreToolUse',
  pre_run_command: 'PreToolUse',
  pre_mcp_tool_use: 'PreToolUse',
  before_tool_call: 'PreToolUse',
  'tool.execute.before': 'PreToolUse',
  PermissionRequest: 'PermissionRequest',
  PostToolUse: 'PostToolUse',
  postToolUse: 'PostToolUse',
  AfterTool: 'PostToolUse',
  afterShellExecution: 'PostToolUse',
  afterMCPExecution: 'PostToolUse',
  afterFileEdit: 'PostToolUse',
  post_write_code: 'PostToolUse',
  post_run_command: 'PostToolUse',
  post_mcp_tool_use: 'PostToolUse',
  PostFileSave: 'PostToolUse',
  after_tool_call: 'PostToolUse',
  'tool.execute.after': 'PostToolUse',
  PostToolUseFailure: 'PostToolUseFailure',
  postToolUseFailure: 'PostToolUseFailure',
  errorOccurred: 'PostToolUseFailure',
  Interrupt: 'PostToolUseFailure',
  PreCompact: 'PreCompact',
  preCompact: 'PreCompact',
  PreCompress: 'PreCompact',
  'experimental.session.compacting': 'PreCompact',
  PostCompact: 'PostCompact',
  SubagentStart: 'SubagentStart',
  subagentStart: 'SubagentStart',
  SubagentStop: 'SubagentStop',
  subagentStop: 'SubagentStop',
  Stop: 'Stop',
  stop: 'Stop',
  AfterAgent: 'Stop',
  agentStop: 'Stop',
  afterAgentResponse: 'Stop',
  post_cascade_response: 'Stop',
  TaskComplete: 'Stop',
  SessionEnd: 'SessionEnd',
  sessionEnd: 'SessionEnd',
  TaskCancel: 'SessionEnd',
  PostInvocation: 'PostToolBatch',
  PostToolBatch: 'PostToolBatch',
};

const REFLEX_CLIENT_FOR_RUNTIME = {
  antigravity: 'claude',
  'claude-code': 'claude',
  codex: 'codex',
  cursor: 'cursor',
  'gemini-cli': 'gemini',
  copilot: 'copilot',
  windsurf: 'windsurf',
  devin: 'devin',
  opencode: 'opencode',
  cline: 'cline',
  kiro: 'kiro',
  'factory-droid': 'factory-droid',
  'roo-code': 'roo-code',
  'kilo-code': 'kilo-code',
  goose: 'goose',
  crush: 'crush',
  openhands: 'openhands',
  openclaw: 'openclaw',
  'amazon-q': 'amazon-q',
  continue: 'continue',
  zed: 'zed',
  amp: 'amp',
  aider: 'aider',
  'python-sdks': 'python-sdks',
};

// Runtime capability flags verified against 2026 official docs and repo port specs.
const DIALECT = {
  antigravity: { style: 'antigravity', exit2BlocksPre: false },
  'claude-code': { style: 'claude', exit2BlocksPre: false, allowsContinueOnPre: true },
  codex: { style: 'codex', exit2BlocksPre: false, allowsContinueOnPre: false },
  devin: { style: 'claude', exit2BlocksPre: false },
  'factory-droid': { style: 'claude', exit2BlocksPre: false },
  openhands: { style: 'claude', exit2BlocksPre: false },
  cursor: { style: 'cursor', exit2BlocksPre: false },
  'gemini-cli': { style: 'gemini', exit2BlocksPre: false },
  copilot: { style: 'copilot', exit2BlocksPre: false },
  windsurf: { style: 'exit2', exit2BlocksPre: true },
  kiro: { style: 'exit2', exit2BlocksPre: true },
  'amazon-q': { style: 'exit2', exit2BlocksPre: true },
  goose: { style: 'exit2', exit2BlocksPre: true },
  crush: { style: 'exit2', exit2BlocksPre: true },
  cline: { style: 'cline', exit2BlocksPre: false },
  'roo-code': { style: 'cursor', exit2BlocksPre: HAS_EXIT_CODE_FLAG },
  'kilo-code': { style: 'cursor', exit2BlocksPre: HAS_EXIT_CODE_FLAG },
  continue: { style: 'cursor', exit2BlocksPre: HAS_EXIT_CODE_FLAG },
  zed: { style: 'cursor', exit2BlocksPre: HAS_EXIT_CODE_FLAG },
  amp: { style: 'cursor', exit2BlocksPre: HAS_EXIT_CODE_FLAG },
  opencode: { style: 'sdk', exit2BlocksPre: false },
  openclaw: { style: 'sdk', exit2BlocksPre: false },
  aider: { style: 'sdk', exit2BlocksPre: false },
  'python-sdks': { style: 'sdk', exit2BlocksPre: false },
};

function resolveHome(p) {
  return typeof p === 'string' ? p.replace(/^~(?=$|[\\/])/, os.homedir()) : p;
}

function firstExisting(candidates) {
  for (const c of candidates) {
    if (c && fs.existsSync(c)) return c;
  }
  return '';
}

function findExecutable(names) {
  const dirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean).concat([
    path.join(os.homedir(), '.vestige', 'bin'),
    path.join(os.homedir(), '.local', 'bin'),
    path.join(os.homedir(), '.cargo', 'bin'),
  ]);
  const suffixes = process.platform === 'win32' ? ['', '.exe', '.cmd', '.bat'] : [''];
  for (const dir of dirs) {
    for (const name of names) {
      for (const sfx of suffixes) {
        const cand = path.join(dir, name + sfx);
        try {
          if (fs.existsSync(cand) && fs.statSync(cand).isFile()) return cand;
        } catch {}
      }
    }
  }
  return '';
}

function loadConfig(runtime, explicitConfig) {
  const home = os.homedir();
  const cfgPath = explicitConfig
    || process.env.VESTIGE_BRIDGE_CONFIG
    || firstExisting([
      path.join(HERE, 'bridge.json'),
      path.join(home, '.vestige', 'memory', 'bridge.json'),
      path.join(home, '.gemini', 'config', 'plugins', 'vestige', 'bridge', 'bridge.json'),
    ]);
  let cfg = {};
  if (cfgPath && fs.existsSync(cfgPath)) {
    cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
  }

  const memCfgPath = resolveHome(
    cfg.memoryConfig
    || process.env.VESTIGE_MEMORY_CONFIG
    || firstExisting([
      path.join(home, '.vestige', 'memory', 'config.json'),
      path.join(home, '.gemini', 'config', 'plugins', 'vestige', 'bridge', 'memory-config.json'),
    ])
  );
  let memCfg = {};
  if (memCfgPath && fs.existsSync(memCfgPath)) {
    try { memCfg = JSON.parse(fs.readFileSync(memCfgPath, 'utf8')); } catch {}
  }

  const mcpBinary = memCfg.binary
    || process.env.VESTIGE_MCP_PATH
    || findExecutable(['vestige-mcp']);
  const triggersPath = memCfg.triggers
    || firstExisting([
      path.join(home, '.vestige', 'memory', 'triggers.json'),
      path.join(home, '.vestige', 'antigravity', 'triggers.json'),
    ]);
  const dataDir = resolveHome(
    cfg.dataDir
    || process.env.VESTIGE_DATA_DIR
    || path.join(home, '.vestige', 'store')
  );

  // Ephemeral runtime config in case user copied only hooks/runtimes/<host> without running setup.
  let effectiveMemCfgPath = memCfgPath;
  let effectiveMemCfg = memCfg;
  if (!effectiveMemCfg.binary && mcpBinary) {
    const tmpDir = path.join(home, '.vestige', 'runtimes', runtime);
    fs.mkdirSync(tmpDir, { recursive: true });
    const trigFile = triggersPath || path.join(tmpDir, 'triggers.json');
    if (!triggersPath) {
      const { defaults } = require(firstExisting([
        path.join(HERE, '..', 'lib', 'memory-triggers.cjs'),
        path.join(HERE, 'lib', 'memory-triggers.cjs'),
      ]));
      fs.writeFileSync(trigFile, JSON.stringify(defaults, null, 2));
    }
    effectiveMemCfg = {
      version: '7.0.0',
      binary: mcpBinary,
      binaryArgs: dataDir ? ['--data-dir', dataDir] : [],
      triggers: trigFile,
      deadlineMs: cfg.memoryDeadlineMs || 6000,
    };
    effectiveMemCfgPath = path.join(tmpDir, 'memory-config.json');
    fs.writeFileSync(effectiveMemCfgPath, JSON.stringify(effectiveMemCfg, null, 2));
  }

  const defaultGate = cfg.gate === 'native'
    ? 'native'
    : firstExisting([
        process.env.OPERATOR_GATE_PY,
        cfg.gate ? resolveHome(cfg.gate) : '',
      ]) || (cfg.gate ? resolveHome(cfg.gate) : (process.env.VESTIGE_GUARD === '1' ? 'native' : ''));
  const enforceGuard = cfg.requireGate !== undefined
    ? Boolean(cfg.requireGate)
    : Boolean(cfg.gate || process.env.OPERATOR_GATE_PY || process.env.VESTIGE_GUARD === '1');

  const resolvedHooksDir = cfg.hooksDir
    ? resolveHome(cfg.hooksDir)
    : (fs.existsSync(path.join(HERE, 'reflex.cjs')) ? HERE : path.join(HERE, 'hooks'));

  return {
    runtime,
    python: cfg.python || process.env.OPERATOR_PYTHON || (process.platform === 'win32' ? (findExecutable(['python', 'python3']) || 'python') : (findExecutable(['python3', 'python']) || 'python3')),
    gate: defaultGate,
    enforceGuard,
    vestigeBin: resolveHome(cfg.vestigeBin || process.env.VESTIGE_BIN || findExecutable(['vestige']) || (process.platform === 'win32' ? 'vestige.exe' : 'vestige')),
    hookBin: resolveHome(cfg.hookBin || process.env.VESTIGE_HOOK_BIN || findExecutable(['vestige-hook']) || ''),
    dataDir,
    memoryConfig: effectiveMemCfgPath,
    memCfg: effectiveMemCfg,
    hooksDir: resolvedHooksDir,
    gateTimeoutMs: cfg.gateTimeoutMs || 10000,
    memoryDeadlineMs: cfg.memoryDeadlineMs || 12000,
    reflexDeadlineMs: cfg.reflexDeadlineMs || 10000,
    capsuleTimeoutMs: cfg.capsuleTimeoutMs || 6000,
    debug: Boolean(cfg.debug || process.env.VESTIGE_HOOK_DEBUG === '1'),
    configPath: cfgPath,
  };
}

function stateRoot(runtime) {
  return path.join(os.homedir(), '.vestige', 'runtimes', runtime);
}

function turnState(runtime, session) {
  const safeSession = session || 'default';
  const file = path.join(
    stateRoot(runtime),
    'turns',
    crypto.createHash('sha256').update(safeSession).digest('hex').slice(0, 24) + '.json'
  );
  let s = {};
  try { s = JSON.parse(fs.readFileSync(file, 'utf8')); } catch {}
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

function queuePending(runtime, session, ctx) {
  if (!ctx || !session) return;
  const t = turnState(runtime, session);
  const seen = new Set();
  const merged = [];
  for (const raw of `${t.s.pending || ''}\n${ctx}`.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || seen.has(line)) continue;
    seen.add(line);
    merged.push(line);
  }
  let out = merged.join('\n');
  if (out.length > 4000) out = out.slice(-4000);
  t.s.pending = out;
  t.save(t.s);
}

function logDebug(cfg, record) {
  if (!cfg || !cfg.debug) return;
  try {
    const dir = stateRoot(cfg.runtime);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, 'bridge.log');
    if (fs.existsSync(file) && fs.statSync(file).size > 2 * 1024 * 1024) fs.renameSync(file, file + '.1');
    fs.appendFileSync(file, JSON.stringify({ t: new Date().toISOString(), ...record }) + '\n');
  } catch {}
}

function logError(cfg, scope, message) {
  try {
    const dir = stateRoot((cfg && cfg.runtime) || 'generic');
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, 'errors.log');
    if (fs.existsSync(file) && fs.statSync(file).size > 256 * 1024) fs.renameSync(file, file + '.1');
    fs.appendFileSync(file, `${new Date().toISOString()} ${scope}: ${message}\n`);
  } catch {}
  logDebug(cfg, { errorScope: scope, error: message });
}

function parseMaybeJsonObj(raw) {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) return raw;
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch {}
  }
  return {};
}

function normalizeEvent(runtime, explicitEvent, input) {
  const native = String(explicitEvent || input.hook_event_name || input.hookEventName || input.hookName || input.trigger || input.event || '').trim();
  if (runtime === 'antigravity' && native === 'PreInvocation') return 'PreInvocation';
  if (EVENT_MAP[native]) return EVENT_MAP[native];
  // Case-insensitive lookup fallback
  const low = native.toLowerCase();
  for (const [k, v] of Object.entries(EVENT_MAP)) {
    if (k.toLowerCase() === low) return v;
  }
  if (input && (input.toolCall || input.tool_name || input.toolName || input.tool_info || input.tool_call || input.preToolUse)) {
    return (input.tool_response || input.toolResult || input.postToolUse || input.result_json !== undefined) ? 'PostToolUse' : 'PreToolUse';
  }
  return native || 'SessionStart';
}

function normalizeToolCall(runtime, nativeEvent, input) {
  // 1. Antigravity camelCase toolCall
  if (input.toolCall) {
    return agy.toClaudeTool(input.toolCall);
  }

  // 2. Windsurf `tool_info` payloads (`command_line`, `file_path`, `mcp_server_name`)
  if (input.tool_info && typeof input.tool_info === 'object') {
    const ti = input.tool_info;
    const ne = String(nativeEvent || '').toLowerCase();
    if (ne.includes('command') || typeof ti.command_line === 'string') {
      return { tool_name: 'Bash', tool_input: { command: String(ti.command_line || ti.command || ''), ...(ti.cwd ? { cwd: ti.cwd } : {}) } };
    }
    if (ne.includes('read')) {
      return { tool_name: 'Read', tool_input: { file_path: String(ti.file_path || '') } };
    }
    if (ne.includes('write')) {
      return { tool_name: Array.isArray(ti.edits) ? 'Edit' : 'Write', tool_input: { file_path: String(ti.file_path || ''), ...(Array.isArray(ti.edits) ? { edits: ti.edits } : {}) } };
    }
    if (ti.mcp_server_name && ti.mcp_tool_name) {
      const cleanArgs = parseMaybeJsonObj(ti.mcp_tool_arguments);
      if (/vestige/i.test(String(ti.mcp_server_name))) {
        return { tool_name: `mcp__vestige__${ti.mcp_tool_name}`, tool_input: cleanArgs };
      }
      return { tool_name: `mcp__${ti.mcp_server_name}__${ti.mcp_tool_name}`, tool_input: cleanArgs };
    }
  }

  // 3. Cline (`tool_call: { name, input }` or `preToolUse: { toolName, parameters }`)
  if (input.tool_call || input.preToolUse || input.postToolUse) {
    const tc = input.tool_call || input.preToolUse || input.postToolUse;
    const rawName = String(tc.name || tc.toolName || '');
    const rawArgs = parseMaybeJsonObj(tc.input ?? tc.parameters);
    const lname = rawName.toLowerCase();
    if (lname === 'run_commands' || lname === 'execute_command') {
      const cmd = typeof rawArgs.command === 'string' ? rawArgs.command : (Array.isArray(rawArgs.commands) ? rawArgs.commands.join(' && ') : String(rawArgs.cmd || ''));
      return { tool_name: 'Bash', tool_input: { command: cmd } };
    }
    if (lname === 'write_to_file') {
      return { tool_name: 'Write', tool_input: { file_path: String(rawArgs.path || rawArgs.file_path || ''), content: String(rawArgs.content || '') } };
    }
    if (/^(apply_diff|replace_in_file|search_and_replace|insert_content)$/.test(lname)) {
      return { tool_name: 'Edit', tool_input: { file_path: String(rawArgs.path || rawArgs.file_path || ''), ...rawArgs } };
    }
    if (lname === 'read_file' || lname === 'read_files') {
      const fp = rawArgs.path || rawArgs.file_path || (Array.isArray(rawArgs.paths) && rawArgs.paths[0]) || '';
      return { tool_name: 'Read', tool_input: { file_path: String(fp) } };
    }
    if (lname === 'use_mcp_tool' && rawArgs.tool_name) {
      if (/vestige/i.test(String(rawArgs.server_name || ''))) {
        return { tool_name: `mcp__vestige__${rawArgs.tool_name}`, tool_input: parseMaybeJsonObj(rawArgs.arguments) };
      }
      return { tool_name: `mcp__${rawArgs.server_name || 'mcp'}__${rawArgs.tool_name}`, tool_input: parseMaybeJsonObj(rawArgs.arguments) };
    }
    return { tool_name: rawName, tool_input: rawArgs };
  }

  // 4. Cursor specialized hooks (`beforeShellExecution`, `beforeMCPExecution`, `beforeReadFile`, `afterFileEdit`)
  const ne = String(nativeEvent || '');
  if (ne === 'beforeShellExecution' || ne === 'afterShellExecution') {
    return { tool_name: 'Bash', tool_input: { command: String(input.command || ''), ...(input.cwd ? { cwd: input.cwd } : {}) } };
  }
  if (ne === 'beforeMCPExecution' || ne === 'afterMCPExecution') {
    const rawTool = String(input.tool_name || '');
    const args = parseMaybeJsonObj(input.tool_input);
    const srv = String(input.mcp_server_name || '');
    if (/vestige/i.test(srv) || /^mcp_vestige|^vestige[:._]/i.test(rawTool)) {
      const leaf = rawTool.replace(/^mcp__[^_]+__/i, '').replace(/^mcp_vestige(?:_vestige)?_/i, '').replace(/^vestige[:._]/i, '');
      return { tool_name: `mcp__vestige__${leaf}`, tool_input: args };
    }
    return { tool_name: srv ? `mcp__${srv}__${rawTool}` : rawTool, tool_input: args };
  }
  if (ne === 'beforeReadFile' || ne === 'beforeTabFileRead') {
    return { tool_name: 'Read', tool_input: { file_path: String(input.file_path || '') } };
  }
  if (ne === 'afterFileEdit' || ne === 'afterTabFileEdit' || ne === 'PostFileSave') {
    return { tool_name: 'Edit', tool_input: { file_path: String(input.file_path || input.path || '') } };
  }

  // 5. Generic (`Claude`, `Codex`, `Gemini CLI`, `Copilot`, `Factory Droid`, `Devin`, `Goose`, `Crush`, `OpenHands`, `OpenCode`)
  const rawName = String(input.tool_name || input.toolName || input.tool || '');
  const ti = parseMaybeJsonObj(input.tool_input ?? input.toolArgs ?? input.args ?? input.arguments ?? input.params ?? (typeof input.command === 'string' ? { command: input.command } : {}));
  const srvName = String(input.mcp_context?.server_name || input.mcp_server_name || '');
  const lname = rawName.toLowerCase();

  // MCP vestige normalization across underscores/colons/slashes
  const mcpDouble = /^mcp__([^_]+(?:_[^_]+)*)__(.+)$/.exec(rawName);
  if (mcpDouble && /vestige/i.test(mcpDouble[1])) {
    return { tool_name: `mcp__vestige__${mcpDouble[2]}`, tool_input: ti };
  }
  const mcpSingle = /^mcp_vestige(?:_vestige)?_(.+)$/i.exec(rawName) || /^vestige[:./](.+)$/i.exec(rawName);
  if (mcpSingle) {
    return { tool_name: `mcp__vestige__${mcpSingle[1]}`, tool_input: ti };
  }
  if (/vestige/i.test(srvName) && rawName) {
    return { tool_name: `mcp__vestige__${rawName.replace(/^mcp__[^_]+__/, '')}`, tool_input: ti };
  }

  // Canonical shell tools across hosts (Bash, bash, Execute, run_shell_command, shell, run_command)
  if (/^(?:bash|execute|exec|shell|run_shell_command|run_command|local_shell|terminal)$/i.test(lname) ||
      (lname === 'apply_patch' && typeof ti.command === 'string')) {
    return { tool_name: 'Bash', tool_input: { ...ti, command: String(ti.command || ti.cmd || ti.command_line || input.command || '') } };
  }

  // File write/edit tools
  if (/^(?:edit|edit_file|replace|str_replace_editor|apply_patch|multiedit|multi_edit)$/i.test(lname)) {
    const fp = ti.file_path || ti.path || ti.target_file || ti.TargetFile || '';
    return { tool_name: 'Edit', tool_input: { ...ti, file_path: String(fp) } };
  }
  if (/^(?:write|write_file|create|create_file)$/i.test(lname)) {
    const fp = ti.file_path || ti.path || ti.target_file || '';
    return { tool_name: 'Write', tool_input: { ...ti, file_path: String(fp) } };
  }

  // File read tools
  if (/^(?:read|read_file|view|view_file|cat)$/i.test(lname)) {
    const fp = ti.file_path || ti.path || ti.TargetFile || '';
    return { tool_name: 'Read', tool_input: { ...ti, file_path: String(fp) } };
  }

  if (rawName) return { tool_name: rawName, tool_input: ti };
  return null;
}

function normalizeToolResponse(input, failedHint) {
  const failed = Boolean(failedHint || (typeof input.error === 'string' && input.error.length > 0));
  let resp = input.tool_response ?? input.toolResult ?? input.result ?? null;
  if (typeof resp === 'string') {
    try { resp = JSON.parse(resp); } catch { resp = { stdout: resp }; }
  }
  if (typeof input.result_json === 'string' && input.result_json.trim()) {
    try {
      const rj = JSON.parse(input.result_json);
      resp = rj && typeof rj === 'object' ? rj : { stdout: input.result_json };
    } catch {
      resp = { stdout: input.result_json };
    }
  }
  if (!resp || typeof resp !== 'object') {
    resp = {};
  }
  // Gemini CLI wraps MCP/tool text inside llmContent; unwrap JSON if present
  if (typeof resp.llmContent === 'string' && resp.llmContent.trim().startsWith('{')) {
    try { resp = { ...resp, ...JSON.parse(resp.llmContent) }; } catch {}
  }
  if (Array.isArray(resp.content)) {
    for (const c of resp.content) {
      if (c && typeof c.text === 'string' && c.text.trim().startsWith('{')) {
        try { resp = { ...resp, ...JSON.parse(c.text) }; break; } catch {}
      }
    }
  }
  if (typeof input.output === 'string' && !resp.stdout) resp.stdout = input.output;
  if (typeof input.error === 'string' && !resp.stderr) resp.stderr = input.error;
  if (typeof resp.exit_code !== 'number') {
    if (typeof input.exitCode === 'number') resp.exit_code = input.exitCode;
    else if (typeof resp.status === 'number') resp.exit_code = resp.status;
    else if (failed || (typeof resp.stderr === 'string' && resp.stderr && !resp.nodeId && !resp.nodes)) resp.exit_code = 1;
    else resp.exit_code = 0;
  }
  return resp;
}

function toCanonicalPayload(runtime, nativeEvent, canonEvent, input) {
  const session = String(
    input.session_id || input.sessionId || input.conversation_id || input.conversationId ||
    input.trajectory_id || input.taskId || 'default'
  );
  const ws = input.cwd
    || (Array.isArray(input.workspace_roots) && input.workspace_roots[0])
    || (Array.isArray(input.workspaceRoots) && input.workspaceRoots[0])
    || (Array.isArray(input.workspacePaths) && input.workspacePaths[0])
    || process.cwd();

  const out = {
    hook_event_name: canonEvent,
    session_id: session,
    cwd: ws,
    transcript_path: String(input.transcript_path || input.transcriptPath || ''),
    artifact_dir: String(input.artifactDirectoryPath || input.artifact_dir || ''),
    source: input.source || runtime,
  };

  const tc = normalizeToolCall(runtime, nativeEvent, input);
  if (tc) {
    out.tool_name = tc.tool_name;
    out.tool_input = tc.tool_input;
    if (tc.tool_input && tc.tool_input.cwd) out.cwd = tc.tool_input.cwd;
    out.tool_use_id = String(
      input.tool_use_id || input.toolUseId || input.call_id ||
      (input.toolCall && input.toolCall.id) || (input.tool_call && input.tool_call.id) ||
      `${runtime}-${session.slice(0, 8)}`
    );
  }

  const prompt = input.prompt ?? input.userPrompt ?? input.user_prompt ?? input.lastUserInput ?? input.prompt_text ?? input.tool_info?.user_prompt;
  if (typeof prompt === 'string') out.prompt = prompt;

  const lastAsst = input.last_assistant_message ?? input.lastAssistantMessage ?? input.prompt_response ?? input.text ?? input.tool_info?.response ?? '';
  if (typeof lastAsst === 'string' && lastAsst) out.last_assistant_message = lastAsst;

  if (input.stop_hook_active === true || input.continuationActive === true || (typeof input.loop_count === 'number' && input.loop_count > 0)) {
    out.stop_hook_active = true;
  }
  if (typeof input.trigger === 'string') out.trigger = input.trigger;
  if (typeof input.agent_id === 'string' || typeof input.subagent_id === 'string') {
    out.agent_id = String(input.agent_id || input.subagent_id);
  }
  return out;
}

function shim(cfg, name) {
  return require(path.join(cfg.hooksDir, name));
}

async function reflexRun(cfg, payload, event) {
  const prevSurface = process.env.VESTIGE_REFLEX_SURFACE_ERROR;
  process.env.VESTIGE_REFLEX_SURFACE_ERROR = '1';
  try {
    const reflex = shim(cfg, 'reflex.cjs');
    const out = await reflex.run({
      client: REFLEX_CLIENT_FOR_RUNTIME[cfg.runtime] || 'claude',
      configFile: cfg.memoryConfig,
      eventFlag: event,
      input: { ...payload, hook_event_name: event },
      deadlineMs: cfg.reflexDeadlineMs,
    }) || {};
    if (out && out._reflexError) {
      logError(cfg, `reflex-${event}`, out._reflexError);
      return {
        hookSpecificOutput: {
          hookEventName: event,
          additionalContext: `Vestige Reflex alert (${event}): ${out._reflexError}.`,
        },
      };
    }
    return out;
  } catch (e) {
    const msg = String((e && e.message) || e);
    logError(cfg, `reflex-${event}`, msg);
    if (!['UserPromptSubmit', 'SessionStart', 'PreToolUse'].includes(event)) return {};
    return {
      hookSpecificOutput: {
        hookEventName: event,
        additionalContext: `Vestige Reflex alert (${event}): ${msg}.`,
      },
    };
  } finally {
    if (prevSurface === undefined) delete process.env.VESTIGE_REFLEX_SURFACE_ERROR;
    else process.env.VESTIGE_REFLEX_SURFACE_ERROR = prevSurface;
  }
}

function compactMemoryText(text, maxChars = 1400) {
  if (!text) return '';
  const seen = new Set();
  const out = [];
  const HDR = "Vestige memory summaries (call recall(handle='<id>') for full text):";
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('Save new durable facts using remember')) continue;
    if (line.startsWith('Vestige local memory candidates.')) {
      if (!seen.has(HDR)) { seen.add(HDR); out.push(HDR); }
      continue;
    }
    if (line.startsWith('{')) {
      try {
        const obj = JSON.parse(line);
        if (obj.backfill_receipt && obj.memory_id) {
          const whyClean = String(obj.reason || '').trim().slice(0, 140);
          const rsbLine = `- [RSB Backfill ${obj.backfill_receipt}] ${obj.memory_id}${whyClean ? ': ' + whyClean : ''}`;
          if (!seen.has(rsbLine)) { seen.add(rsbLine); out.push(rsbLine); }
          continue;
        }
        if (obj.id && typeof obj.text === 'string') {
          const clean = obj.text.replace(/\s+/g, ' ').trim();
          const summary = clean.length > 200 ? clean.slice(0, 197).trimEnd() + '...' : clean;
          const bullet = `- ${obj.id}${obj.date ? ` [${obj.date}]` : ''}: ${summary}`;
          if (!seen.has(bullet)) { seen.add(bullet); out.push(bullet); }
          continue;
        }
      } catch {}
    }
    const clipped = line.length > 360 ? line.slice(0, 357).trimEnd() + '...' : line;
    if (!seen.has(clipped)) { seen.add(clipped); out.push(clipped); }
  }
  while (out.length && out[out.length - 1] === HDR) out.pop();
  return packLines(out.join('\n'), maxChars);
}

async function memoryContext(cfg, payload, hookEvent, maxChars = 1400) {
  if (!cfg.memoryConfig || !fs.existsSync(cfg.memoryConfig)) return '';
  try {
    const memory = shim(cfg, 'memory.cjs');
    const client = ['claude', 'codex', 'gemini', 'cursor'].includes(REFLEX_CLIENT_FOR_RUNTIME[cfg.runtime])
      ? REFLEX_CLIENT_FOR_RUNTIME[cfg.runtime]
      : 'claude';
    const out = await memory.run({
      event: hookEvent,
      client: 'claude', // get canonical additionalContext so compactMemoryText works uniformly across all runtimes
      configFile: cfg.memoryConfig,
      input: { ...payload, source_client: client },
      deadlineMs: cfg.memoryDeadlineMs,
    });
    const raw = (out && out.hookSpecificOutput && out.hookSpecificOutput.additionalContext) || '';
    return compactMemoryText(raw, maxChars);
  } catch (e) {
    const msg = String((e && e.message) || e);
    logError(cfg, `memory-${hookEvent}`, msg);
    return `Vestige memory hook alert (${hookEvent}): ${msg}`;
  }
}

function spawnAsync(cmd, args, { input = '', timeoutMs = 4000, env = process.env } = {}) {
  return new Promise((resolve) => {
    let stdout = '';
    let done = false;
    const finish = (ok, out = '') => {
      if (done) return;
      done = true;
      resolve({ ok, stdout: out });
    };
    try {
      const child = spawn(cmd, args, { env, windowsHide: true, stdio: ['pipe', 'pipe', 'ignore'] });
      const timer = setTimeout(() => { try { child.kill(); } catch {} finish(false, stdout); }, timeoutMs);
      child.stdout.on('data', (chunk) => { if (stdout.length < 1024 * 1024) stdout += chunk.toString('utf8'); });
      child.on('error', () => { clearTimeout(timer); finish(false, ''); });
      child.on('close', (code) => { clearTimeout(timer); finish(code === 0, stdout); });
      child.stdin.end(input || '');
    } catch {
      finish(false, '');
    }
  });
}

async function turnoverSessionStartAsync(_cfg, _payload) {
  // Native Rust `vestige gate` replaces legacy Python turnover startup overhead.
}

async function capsuleRunAsync(cfg, source, cwd) {
  if (!cfg.vestigeBin || !fs.existsSync(cfg.vestigeBin)) return '';
  try {
    const r = await spawnAsync(
      cfg.vestigeBin,
      ['--data-dir', cfg.dataDir, 'capsule', '--harness', 'claude-code', '--event', 'session-start'],
      {
        input: JSON.stringify({ source: source || 'startup', cwd: cwd || process.cwd() }),
        timeoutMs: cfg.capsuleTimeoutMs,
      }
    );
    const out = String(r.stdout || '').trim();
    if (!out) return '';
    const parsed = JSON.parse(out);
    const raw = (parsed && parsed.hookSpecificOutput && parsed.hookSpecificOutput.additionalContext) || '';
    return agy.compactCapsule(raw, 1100);
  } catch {
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
    client = await openRpc(cfg, Math.max(5000, cfg.memoryDeadlineMs || 6000));
    if (!client) return '';
    const [flat, portfolio] = await Promise.all([
      client.tool('session', { action: 'intention', params: { action: 'list', filter_status: 'active', limit: 8 } }).catch(() => null),
      client.tool('session', { action: 'intention', params: { action: 'graph', command: { action: 'portfolio' } } }).catch(() => null),
    ]);
    const lines = [];
    for (const item of (flat && Array.isArray(flat.intentions)) ? flat.intentions : []) {
      const desc = String(item.description || item.content || '').replace(/\s+/g, ' ').trim().slice(0, 180);
      if (desc) lines.push(`- ${item.id || 'int'}${item.priority ? ` [${item.priority}]` : ''}${item.deadline ? ` (due ${String(item.deadline).slice(0, 10)})` : ''}: ${desc}`);
    }
    const plans = (portfolio && portfolio.result && Array.isArray(portfolio.result.plans)) ? portfolio.result.plans : [];
    for (const p of plans.filter(x => x.lifecycle === 'active').slice(0, 4)) {
      lines.push(`- plan:${p.id} [${p.priority || 'normal'}/${p.readiness || 'ready'}]: ${String(p.description || '').replace(/\s+/g, ' ').trim().slice(0, 140)}`);
    }
    if (!lines.length) return '';
    const body = `Active Intentions & Plans:\n${lines.join('\n')}`;
    return body.length > maxChars ? body.slice(0, maxChars - 3).trimEnd() + '...' : body;
  } catch {
    return '';
  } finally {
    if (client) try { client.close(); } catch {}
  }
}

function postCommitReceiptRun(cfg, payload) {
  if (!cfg.vestigeBin || !fs.existsSync(cfg.vestigeBin)) return;
  const cmd = String((payload.tool_input && payload.tool_input.command) || '');
  if (!agy.isGitCommitCommand(cmd)) return;
  const repoDir = agy.extractGitRepoFromCommand(cmd, payload.cwd);
  try {
    spawnSync(cfg.vestigeBin, ['--data-dir', cfg.dataDir, 'hook', 'post-commit', '--repo', repoDir], {
      encoding: 'utf8', timeout: 5000, maxBuffer: 512 * 1024, windowsHide: true,
    });
  } catch (e) {
    logError(cfg, 'post-commit', String(e && e.message));
  }
}

function extractContext(out) {
  return (out && out.hookSpecificOutput && out.hookSpecificOutput.additionalContext)
    || (out && out.additional_context)
    || (out && out.additionalContext)
    || '';
}

function packLines(text, maxChars = 1600) {
  if (!text) return '';
  const seen = new Set();
  const lines = [];
  let len = 0;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || seen.has(line)) continue;
    seen.add(line);
    const add = (lines.length ? 1 : 0) + line.length;
    if (lines.length && len + add > maxChars) break;
    lines.push(line);
    len += add;
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------- canonical handlers

async function handleSessionStart(cfg, payload) {
  const startSource = ['startup', 'resume', 'clear', 'compact', 'fork'].includes(payload.source) ? payload.source : 'startup';
  const startPayload = { ...payload, source: startSource };
  const hasLiveDaemon = Boolean(cfg.dataDir && fs.existsSync(path.join(cfg.dataDir, '.serve.endpoint')));
  let capsule, intentions, mem, rfxOut;
  if (hasLiveDaemon) {
    [, capsule, intentions, mem, rfxOut] = await Promise.all([
      turnoverSessionStartAsync(cfg, startPayload),
      capsuleRunAsync(cfg, startSource, payload.cwd),
      intentionsStartContext(cfg, 750),
      memoryContext(cfg, startPayload, 'session', 800),
      reflexRun(cfg, startPayload, 'SessionStart'),
    ]);
  } else {
    // Cold start without a running daemon: serialize Strata store readers so concurrent
    // child processes never contend on exclusive Windows file locks.
    const [, cap] = await Promise.all([
      turnoverSessionStartAsync(cfg, startPayload),
      capsuleRunAsync(cfg, startSource, payload.cwd),
    ]);
    capsule = cap;
    rfxOut = await reflexRun(cfg, startPayload, 'SessionStart');
    mem = await memoryContext(cfg, startPayload, 'session', 800);
    intentions = await intentionsStartContext(cfg, 750);
  }
  const rfx = packLines(extractContext(rfxOut), 450);
  const context = packLines([intentions, capsule, rfx, mem].filter(Boolean).join('\n\n'), 2400);
  if (payload.session_id && (intentions || context)) {
    const t = turnState(cfg.runtime, payload.session_id);
    if (intentions) { t.s.intentionsShown = true; t.s.hasActiveIntention = true; }
    if (context) t.s.sessionBooted = true;
    t.save(t.s);
  }
  return { context };
}

function resolveFastHookBin(cfg) {
  if (cfg.hookBin && fs.existsSync(cfg.hookBin)) return cfg.hookBin;
  if (cfg.vestigeBin) {
    const sibling = path.join(path.dirname(cfg.vestigeBin), process.platform === 'win32' ? 'vestige-hook.exe' : 'vestige-hook');
    if (fs.existsSync(sibling)) return sibling;
  }
  return '';
}

function nativeMissionHookRun(cfg, event, payload) {
  const hookBin = resolveFastHookBin(cfg);
  const bin = hookBin || cfg.vestigeBin;
  if (!bin || !fs.existsSync(bin)) return null;
  const args = hookBin
    ? ['dispatch', cfg.runtime || 'claude-code', event, '--no-receipt']
    : ['hook', 'dispatch', cfg.runtime || 'claude-code', event, '--no-receipt'];
  try {
    const r = spawnSync(bin, args, {
      input: JSON.stringify(payload),
      encoding: 'utf8',
      timeout: 4000,
      maxBuffer: 256 * 1024,
      windowsHide: true,
    });
    const out = String(r.stdout || '').trim();
    return out ? JSON.parse(out) : null;
  } catch {
    return null;
  }
}

async function handlePrompt(cfg, payload) {
  const sid = payload.session_id;
  const t = turnState(cfg.runtime, sid);
  const prompt = payload.prompt || '';
  const promptKey = crypto.createHash('sha256').update(prompt).digest('hex').slice(0, 16);
  const parts = [];
  if (t.s.promptKey !== promptKey || cfg.runtime !== 'antigravity') {
    // Capture automatic imperative mission contracts in native Rust (<5ms, zero Python).
    nativeMissionHookRun(cfg, 'UserPromptSubmit', payload);
    // Serialize reflex ahead of memory context so they never contend on the Strata store lock.
    const rfxOut = await reflexRun(cfg, payload, 'UserPromptSubmit');
    const mem = await memoryContext(cfg, payload, 'prompt', 900);
    const firstIntentions = !t.s.intentionsShown ? await intentionsStartContext(cfg, 550) : '';
    if (firstIntentions) { parts.push(firstIntentions); t.s.intentionsShown = true; t.s.hasActiveIntention = true; }
    const rfx = packLines(extractContext(rfxOut), 550);
    if (rfx) parts.push(rfx);
    if (mem) parts.push(mem);
    t.s.promptKey = promptKey;
    t.s.rememberedThisTurn = false;
    t.s.unanchoredSourceEdits = false;
  } else {
    const rfxOut = await reflexRun(cfg, payload, 'PostToolBatch');
    const rfx = packLines(extractContext(rfxOut), 550);
    if (rfx) parts.push(rfx);
  }
  if (t.s.pending) {
    parts.unshift(packLines(t.s.pending, 650));
    t.s.pending = '';
  }
  t.save(t.s);
  return { context: packLines(parts.filter(Boolean).join('\n\n'), 1800) };
}

async function handlePreToolUse(cfg, payload, nativeEvent, input) {
  // Fast path for non-sensitive Read operations
  if (payload.tool_name === 'Read') {
    const readPath = String((payload.tool_input && payload.tool_input.file_path) || '');
    if (readPath && !agy.SENSITIVE_READ_RE.test(readPath)) {
      return { allow: true };
    }
  }

  // 1. Native Rust Operator Gate (if enabled/present)
  if (cfg.gate || cfg.enforceGuard) {
    const useNative = !cfg.gate || cfg.gate === 'native';
    let r;
    if (useNative) {
      const hookBin = resolveFastHookBin(cfg);
      const bin = hookBin || cfg.vestigeBin;
      if (!bin || !fs.existsSync(bin)) {
        return { deny: true, reason: `Native Vestige gate binary missing at ${bin || 'vestige'}. Fail-closed: a missing gate is never permission to run.` };
      }
      const gateArgs = hookBin
        ? ['gate', '--source', cfg.runtime]
        : ['gate', 'hook', '--source', cfg.runtime];
      r = spawnSync(bin, gateArgs, {
        input: JSON.stringify(payload),
        encoding: 'utf8',
        timeout: cfg.gateTimeoutMs,
        maxBuffer: 1024 * 1024,
        windowsHide: true,
      });
    } else {
      if (!fs.existsSync(cfg.gate)) {
        return { deny: true, reason: `Operator gate missing at ${cfg.gate}. Fail-closed: a missing gate is never permission to run.` };
      }
      r = spawnSync(cfg.python, ['-I', cfg.gate, 'hook', '--source', cfg.runtime], {
        input: JSON.stringify(payload),
        encoding: 'utf8',
        timeout: cfg.gateTimeoutMs,
        maxBuffer: 1024 * 1024,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
        windowsHide: true,
      });
    }
    const stderr = String(r.stderr || '').trim();
    if (r.error) return { deny: true, reason: `Operator gate did not answer (${r.error.code || r.error.message}). Fail-closed.` };
    if (r.status !== 0) {
      const ruleMatch = /\b(OP-[A-Z0-9-]+)\b/.exec(stderr);
      const ruleId = ruleMatch ? ruleMatch[1] : 'OP-STOP';
      if (payload.session_id) {
        const t = turnState(cfg.runtime, payload.session_id);
        t.s.pendingOperatorStop = { rule: ruleId, tool: payload.tool_name, ts: Date.now() };
        t.save(t.s);
        queuePending(cfg.runtime, payload.session_id,
          `Vestige Protocol [Operator ${ruleId}]: Call precedent(query="stops") to inspect the Operator stop receipt before running another mutating command or edit.`
        );
      }
      const msg = stderr || `Operator gate exited ${r.status}. Fail-closed.`;
      return { deny: true, reason: `${msg}\nNext required step: call precedent(query="stops") to inspect the stop receipt before proposing a compliant alternative.` };
    }
  }

  // 2. Enforce precedent inspection after an Operator stop before any further mutating tool runs
  if (payload.session_id && /^(?:Bash|Edit|Write)$/i.test(String(payload.tool_name || ''))) {
    const t = turnState(cfg.runtime, payload.session_id);
    if (t.s.pendingOperatorStop) {
      return {
        deny: true,
        reason: `Vestige Protocol [Precedent Required]: Operator stopped a previous call (${t.s.pendingOperatorStop.rule}) this turn. Call precedent(query="stops") to inspect the receipt before running ${payload.tool_name}.`,
      };
    }
  }

  // 3. Reflex + Call Coach (CC-001..CC-013) + Anti-Spiral + Commit-Gate + Blast-Radius
  const rfxOut = await reflexRun(cfg, payload, 'PreToolUse');
  const hso = (rfxOut && rfxOut.hookSpecificOutput) || {};
  if (hso.permissionDecision === 'deny' || rfxOut.permission === 'deny' || rfxOut.decision === 'deny') {
    const reason = hso.permissionDecisionReason || rfxOut.agent_message || rfxOut.reason || 'Vestige Reflex denied this call.';
    return { deny: true, reason };
  }

  // 4. Secret scan gate before git commit if a sensitive file was modified
  if (payload.session_id && payload.tool_name === 'Bash' && agy.isGitCommitCommand(payload.tool_input && payload.tool_input.command)) {
    const tsCommit = turnState(cfg.runtime, payload.session_id);
    if (tsCommit.s.sensitiveFileEdited && !tsCommit.s.secretScannedThisTurn) {
      return {
        deny: true,
        reason: `Vestige Protocol [precedent]: Sensitive file (${tsCommit.s.sensitiveFileEdited}) was modified before commit. Call precedent(query="scan_secrets") before committing.`,
      };
    }
  }

  let updatedInput = hso.updatedInput || rfxOut.updated_input || (rfxOut.hookSpecificOutput && rfxOut.hookSpecificOutput.tool_input) || null;
  if (updatedInput && nativeEvent === 'beforeShellExecution' && typeof updatedInput.command === 'string') {
    updatedInput = { command: updatedInput.command };
  }

  // Filter out silent Call Coach notes unless no rewrite mechanism or non-blocking warning needed
  const rawCtx = extractContext(rfxOut);
  const tsWarn = payload.session_id ? turnState(cfg.runtime, payload.session_id) : null;
  const warnedFiles = (tsWarn && Array.isArray(tsWarn.s.warnedFiles)) ? tsWarn.s.warnedFiles : [];
  let warnedChanged = false;
  const keptLines = [];
  for (const raw of rawCtx.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (/^(?:(?:Vestige\s+)?Call Coach:\s*)?\[CC-(?:001|003|004|005|006|007|008|009|012|013)\](?=\s|$|:)/i.test(line)) continue;
    const blastMatch = /^Prefrontal Blast-Radius\s*\(`([^`]+)`\)/i.exec(line);
    if (blastMatch && tsWarn) {
      const norm = agy.normalizePathForReflex(blastMatch[1], payload.cwd);
      if (norm && warnedFiles.includes(norm)) continue;
      if (norm) { warnedFiles.push(norm); warnedChanged = true; }
    }
    keptLines.push(line);
  }
  if (warnedChanged && tsWarn) { tsWarn.s.warnedFiles = warnedFiles.slice(-64); tsWarn.save(tsWarn.s); }
  const context = packLines(keptLines.join('\n'), 900);
  if (context && (DIALECT[cfg.runtime]?.style === 'antigravity' || DIALECT[cfg.runtime]?.style === 'exit2')) {
    queuePending(cfg.runtime, payload.session_id, context);
  }

  return {
    allow: true,
    updatedInput,
    context: (DIALECT[cfg.runtime]?.style === 'antigravity' || DIALECT[cfg.runtime]?.style === 'exit2') ? '' : context,
    note: hso.permissionDecisionReason || '',
  };
}

async function handlePostToolUse(cfg, payload, canonEvent, input) {
  const failed = canonEvent === 'PostToolUseFailure' || typeof input.error === 'string' && input.error.length > 0;
  payload.tool_response = normalizeToolResponse(input, failed);
  if (failed && typeof input.error === 'string') payload.error = input.error;

  const toolStr = String(payload.tool_name || '');
  if (!failed && /^mcp__vestige__/i.test(toolStr) && payload.session_id) {
    const t = turnState(cfg.runtime, payload.session_id);
    if (/^mcp__vestige__session$/i.test(toolStr)) {
      t.s.sessionBooted = true;
      const act = String((payload.tool_input && payload.tool_input.action) || '');
      if (/intention/i.test(act) && !/list|status|get/i.test(act)) t.s.hasActiveIntention = true;
    } else if (/^mcp__vestige__(?:remember|curate)$/i.test(toolStr)) {
      t.s.rememberedThisTurn = true;
      t.s.unanchoredSourceEdits = false;
      const contradictions = payload.tool_response.contradictions || payload.tool_response.conflicts;
      if (Array.isArray(contradictions) && contradictions.length > 0) {
        queuePending(cfg.runtime, payload.session_id,
          `Vestige Curate Alert: ${contradictions.length} contradicting/stale memory node(s) detected. Call curate(action="edit"|"demote"|"link") so stale beliefs do not remain active.`
        );
      }
    } else if (/^mcp__vestige__precedent$/i.test(toolStr)) {
      if (t.s.pendingOperatorStop) delete t.s.pendingOperatorStop;
      if (String((payload.tool_input && payload.tool_input.query) || '') === 'scan_secrets') {
        t.s.secretScannedThisTurn = true;
      }
    } else if (/^mcp__vestige__recall$/i.test(toolStr)) {
      const nodes = Array.isArray(payload.tool_response.nodes) ? payload.tool_response.nodes : [];
      const retired = nodes.filter(n => n && (n.retired === true || n.supersededBy || n.superseded_by || n.suppressed === true));
      if (retired.length > 0) {
        queuePending(cfg.runtime, payload.session_id,
          `Vestige Curate Alert: recall returned ${retired.length} retired/superseded node(s). Verify current successors or call curate(...) before relying on stale beliefs.`
        );
      }
    }
    t.save(t.s);
  }

  if (!failed && (payload.tool_name === 'Edit' || payload.tool_name === 'Write') &&
      !agy.isScratchOrArtifactEdit(payload.tool_input && payload.tool_input.file_path, payload.artifact_dir) &&
      payload.session_id) {
    const t = turnState(cfg.runtime, payload.session_id);
    const editPath = String((payload.tool_input && payload.tool_input.file_path) || '');
    const normPath = agy.normalizePathForReflex(editPath, payload.cwd) || editPath;
    t.s.editedFiles = Array.isArray(t.s.editedFiles) ? t.s.editedFiles : [];
    if (normPath && !t.s.editedFiles.includes(normPath)) {
      t.s.editedFiles.push(normPath);
      if (t.s.editedFiles.length > 64) t.s.editedFiles.shift();
    }
    if (editPath && agy.SENSITIVE_READ_RE.test(editPath) &&
        !/(?:^|[\\/])(?:test|tests|spec|fixtures?)(?:[\\/]|$)|\.(?:test|spec)\./i.test(editPath)) {
      t.s.sensitiveFileEdited = normPath;
      t.s.secretScannedThisTurn = false;
    }
    t.s.unanchoredSourceEdits = true;
    const needsBootNudge = !t.s.sessionBooted && !t.s.sessionNudged;
    if (needsBootNudge) t.s.sessionNudged = true;
    const needsIntentionNudge = !needsBootNudge && t.s.editedFiles.length >= 3 && !t.s.hasActiveIntention && !t.s.intentionNudged;
    if (needsIntentionNudge) t.s.intentionNudged = true;
    t.save(t.s);
    if (needsBootNudge) {
      queuePending(cfg.runtime, payload.session_id, 'Vestige Autopilot: call session(action="start") to hydrate pinned rules & active intentions before editing further.');
    } else if (needsIntentionNudge) {
      queuePending(cfg.runtime, payload.session_id, `Vestige Autopilot [session]: ${t.s.editedFiles.length} source files edited with no active intention — call session(action="set_intention", intention="...") so progress survives compaction.`);
    }
  }

  if (!failed && payload.tool_name === 'Bash') {
    postCommitReceiptRun(cfg, payload);
    if (payload.session_id && agy.isGitCommitCommand(payload.tool_input && payload.tool_input.command)) {
      const t = turnState(cfg.runtime, payload.session_id);
      if (t.s.sensitiveFileEdited || t.s.secretScannedThisTurn) {
        t.s.sensitiveFileEdited = '';
        t.s.secretScannedThisTurn = false;
        t.save(t.s);
      }
    }
  }

  if (!failed && payload.tool_name === 'Read') return {};

  const rfxOut = await reflexRun(cfg, payload, failed ? 'PostToolUseFailure' : 'PostToolUse');
  const t = payload.session_id ? turnState(cfg.runtime, payload.session_id) : null;
  const queued = (t && t.s.pending) ? t.s.pending : '';
  const combined = packLines([queued, extractContext(rfxOut)].filter(Boolean).join('\n'), 1200);
  if (combined && t) {
    // Hosts that consume PostToolUse additionalContext get it now; others keep it queued until next prompt.
    if (['claude', 'codex', 'gemini', 'copilot', 'cursor', 'cline', 'sdk'].includes(DIALECT[cfg.runtime]?.style)) {
      t.s.pending = '';
      t.save(t.s);
      return { context: combined };
    }
    queuePending(cfg.runtime, payload.session_id, combined);
  }
  return {};
}

async function handleStop(cfg, payload, input, nativeEvent) {
  await memoryContext(cfg, payload, 'turn');
  const candidate = agy.extractAutoSaveCandidate({
    userText: payload.prompt || '',
    assistantText: payload.last_assistant_message || '',
    alreadyRemembered: Boolean(payload.session_id && turnState(cfg.runtime, payload.session_id).s.rememberedThisTurn),
    sawToolActivity: true,
  });
  if (candidate && payload.session_id) {
    const t = turnState(cfg.runtime, payload.session_id);
    if (!t.s.rememberedThisTurn && !t.s.unanchoredSourceEdits) {
      const client = await openRpc(cfg, Math.min(4000, cfg.memoryDeadlineMs || 4000)).catch(() => null);
      if (client) {
        try {
          const res = await client.tool('remember', {
            action: 'save',
            node_type: candidate.node_type === 'decision' ? 'note' : candidate.node_type,
            content: candidate.content,
            tags: candidate.tags,
            scope: 'personal',
            intent_id: `auto-stop-${cfg.runtime}-${t.s.promptKey || 'turn'}`,
          }).catch(() => null);
          const autoId = res && (res.nodeId || res.echoNodeId || res.replayOf);
          if (autoId) {
            await reflexRun(cfg, {
              ...payload,
              hook_event_name: 'PostToolUse',
              tool_name: 'mcp__vestige__remember',
              tool_input: { action: 'save' },
              tool_response: { exit_code: 0, nodeId: autoId, success: true },
            }, 'PostToolUse');
          }
        } finally {
          client.close();
        }
      }
    }
  }

  if (nativeEvent === 'afterAgentResponse') return {};
  const missionOut = nativeMissionHookRun(cfg, 'Stop', payload);
  const missionBlock = (missionOut && (missionOut.decision === 'deny' || missionOut.decision === 'block') && missionOut.reason) || '';
  const rfxOut = await reflexRun(cfg, payload, 'Stop');
  const reason = missionBlock || (rfxOut && rfxOut.decision === 'block' && rfxOut.reason) || '';
  const t = payload.session_id ? turnState(cfg.runtime, payload.session_id) : null;
  const maxBlocks = 3;
  let blockCount = 0;
  if (t && t.s.promptKey && t.s.stopBlockedPromptKey === t.s.promptKey) {
    blockCount = Number(t.s.stopBlockCount || 0);
  }
  if (reason && blockCount < maxBlocks) {
    if (t && t.s.promptKey) {
      t.s.stopBlockedPromptKey = t.s.promptKey;
      t.s.stopBlockCount = blockCount + 1;
      t.save(t.s);
    }
    return { block: true, reason };
  }
  return {};
}

// ---------------------------------------------------------------- host output formatting

function formatRuntimeOutput(runtime, nativeEvent, canonEvent, res) {
  const dialect = DIALECT[runtime] || { style: 'claude', exit2BlocksPre: false };
  const eventLabel = nativeEvent || canonEvent;

  if (canonEvent === 'PreToolUse') {
    if (res.deny) {
      const reason = res.reason || 'Vestige denied this tool call.';
      if (dialect.exit2BlocksPre) {
        return { exitCode: 2, stderr: reason, stdout: null };
      }
      if (dialect.style === 'codex') {
        return {
          exitCode: 0,
          stdout: { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'deny', permissionDecisionReason: reason } },
        };
      }
      if (dialect.style === 'claude') {
        return {
          exitCode: 0,
          stdout: { hookSpecificOutput: { hookEventName: 'PreToolUse', permissionDecision: 'deny', permissionDecisionReason: reason } },
        };
      }
      if (dialect.style === 'cursor') {
        return { exitCode: 0, stdout: { permission: 'deny', user_message: reason, agent_message: reason } };
      }
      if (dialect.style === 'gemini') {
        return { exitCode: 0, stdout: { decision: 'deny', reason } };
      }
      if (dialect.style === 'copilot') {
        return { exitCode: 0, stdout: { permissionDecision: 'deny', permissionDecisionReason: reason } };
      }
      if (dialect.style === 'cline') {
        return { exitCode: 0, stdout: { cancel: true, errorMessage: reason } };
      }
      if (dialect.style === 'antigravity') {
        return { exitCode: 0, stdout: { decision: 'deny', reason } };
      }
      return {
        exitCode: 0,
        stdout: { deny: true, permission: 'deny', permissionDecision: 'deny', reason, errorMessage: reason },
      };
    }

    // Allow / Rewrite / Context
    if (dialect.style === 'exit2') return { exitCode: 0, stdout: {} };
    if (dialect.style === 'codex' || dialect.style === 'claude') {
      if (!res.updatedInput && !res.context) return { exitCode: 0, stdout: {} };
      return {
        exitCode: 0,
        stdout: {
          hookSpecificOutput: {
            hookEventName: 'PreToolUse',
            ...(res.updatedInput ? { permissionDecision: 'allow', updatedInput: res.updatedInput } : {}),
            ...(res.context ? { additionalContext: res.context } : {}),
          },
        },
      };
    }
    if (dialect.style === 'cursor') {
      return {
        exitCode: 0,
        stdout: {
          permission: 'allow',
          ...(res.updatedInput ? { updated_input: res.updatedInput } : {}),
          ...(res.context ? { additional_context: res.context } : {}),
        },
      };
    }
    if (dialect.style === 'gemini') {
      if (!res.updatedInput && !res.context) return { exitCode: 0, stdout: {} };
      return {
        exitCode: 0,
        stdout: {
          hookSpecificOutput: {
            hookEventName: eventLabel,
            ...(res.updatedInput ? { tool_input: res.updatedInput } : {}),
            ...(res.context ? { additionalContext: res.context } : {}),
          },
        },
      };
    }
    if (dialect.style === 'copilot') {
      return { exitCode: 0, stdout: res.context ? { additionalContext: res.context } : {} };
    }
    if (dialect.style === 'cline') {
      return { exitCode: 0, stdout: res.context ? { contextModification: res.context } : {} };
    }
    if (dialect.style === 'antigravity') {
      return { exitCode: 0, stdout: res.updatedInput ? { decision: 'allow', overwrite: res.updatedInput } : { decision: 'allow' } };
    }
    return {
      exitCode: 0,
      stdout: {
        allow: true,
        permission: 'allow',
        ...(res.updatedInput ? { updatedInput: res.updatedInput, updated_input: res.updatedInput } : {}),
        ...(res.context ? { additionalContext: res.context } : {}),
      },
    };
  }

  if (canonEvent === 'PermissionRequest' && dialect.style === 'codex') {
    return {
      exitCode: 0,
      stdout: {
        hookSpecificOutput: {
          hookEventName: 'PermissionRequest',
          decision: res.deny ? { behavior: 'deny', message: res.reason } : { behavior: 'allow' },
        },
      },
    };
  }

  if (canonEvent === 'Stop' || canonEvent === 'SubagentStop') {
    if (res.block && res.reason) {
      if (dialect.style === 'codex' || dialect.style === 'claude') return { exitCode: 0, stdout: { decision: 'block', reason: res.reason } };
      if (dialect.style === 'cursor') return { exitCode: 0, stdout: { followup_message: res.reason } };
      if (dialect.style === 'gemini') return { exitCode: 0, stdout: { decision: 'deny', reason: res.reason } };
      if (dialect.style === 'copilot') return { exitCode: 0, stdout: { permissionDecision: 'deny', permissionDecisionReason: res.reason } };
      if (dialect.style === 'cline') return { exitCode: 0, stdout: { cancel: true, errorMessage: res.reason } };
      if (dialect.style === 'antigravity') return { exitCode: 0, stdout: { decision: 'continue', reason: res.reason } };
      if (dialect.style === 'exit2') return { exitCode: 2, stderr: res.reason, stdout: null };
      return { exitCode: 0, stdout: { block: true, reason: res.reason, decision: 'block' } };
    }
    return { exitCode: 0, stdout: {} };
  }

  // SessionStart / UserPromptSubmit / PostToolUse / PreCompact / PostCompact / SessionEnd
  const ctx = res.context || '';
  if (dialect.style === 'cursor') {
    if (eventLabel === 'beforeSubmitPrompt') return { exitCode: 0, stdout: { continue: true } };
    if (ctx && ['sessionStart', 'postToolUse', 'postToolUseFailure'].includes(eventLabel)) {
      return { exitCode: 0, stdout: { additional_context: ctx } };
    }
    return { exitCode: 0, stdout: {} };
  }
  if (dialect.style === 'copilot') return { exitCode: 0, stdout: ctx ? { additionalContext: ctx } : {} };
  if (dialect.style === 'cline') return { exitCode: 0, stdout: ctx ? { contextModification: ctx } : {} };
  if (dialect.style === 'exit2') return { exitCode: 0, stdout: {} };
  if (dialect.style === 'antigravity') return { exitCode: 0, stdout: ctx ? { injectSteps: [{ systemMessage: { systemMessage: ctx } }] } : {} };
  if (dialect.style === 'sdk') return { exitCode: 0, stdout: ctx ? { additionalContext: ctx } : {} };
  return {
    exitCode: 0,
    stdout: ctx ? { hookSpecificOutput: { hookEventName: eventLabel, additionalContext: ctx } } : {},
  };
}

async function runBridge({ runtime, eventFlag, configPath, input }) {
  const rt = normalizeRuntime(runtime);
  const cfg = loadConfig(rt, configPath);
  const nativeEvent = String(eventFlag || input.hook_event_name || input.hookEventName || input.hookName || input.trigger || input.event || '').trim();
  const canonEvent = normalizeEvent(rt, nativeEvent, input);
  const payload = toCanonicalPayload(rt, nativeEvent, canonEvent, input);
  logDebug(cfg, { runtime: rt, nativeEvent, canonEvent, tool: payload.tool_name, session: payload.session_id });

  let res = {};
  switch (canonEvent) {
    case 'SessionStart':
      res = await handleSessionStart(cfg, payload);
      break;
    case 'UserPromptSubmit':
      res = await handlePrompt(cfg, payload);
      break;
    case 'PreInvocation':
      res = await handlePrompt(cfg, payload);
      break;
    case 'PreToolUse':
      res = await handlePreToolUse(cfg, payload, nativeEvent, input);
      break;
    case 'PermissionRequest':
      res = await handlePreToolUse(cfg, payload, nativeEvent, input);
      break;
    case 'PostToolUse':
    case 'PostToolUseFailure':
      res = await handlePostToolUse(cfg, payload, canonEvent, input);
      break;
    case 'PreCompact':
      await memoryContext(cfg, payload, 'compact');
      await reflexRun(cfg, payload, 'PreCompact');
      res = {};
      break;
    case 'PostCompact':
    case 'SubagentStart':
    case 'PostToolBatch':
      await reflexRun(cfg, payload, canonEvent);
      res = {};
      break;
    case 'SubagentStop':
      await reflexRun(cfg, payload, 'SubagentStop');
      res = {};
      break;
    case 'Stop':
      res = await handleStop(cfg, payload, input, nativeEvent);
      break;
    case 'SessionEnd':
      await memoryContext(cfg, payload, 'end');
      await reflexRun(cfg, payload, 'SessionEnd');
      res = {};
      break;
    default:
      res = {};
  }
  return formatRuntimeOutput(rt, nativeEvent, canonEvent, res);
}

module.exports = {
  RUNTIME_ALIASES,
  DIALECT,
  normalizeRuntime,
  normalizeEvent,
  normalizeToolCall,
  toCanonicalPayload,
  formatRuntimeOutput,
  runBridge,
};

if (require.main === module) {
  let raw = '';
  try { raw = fs.readFileSync(0, 'utf8'); } catch {}
  let input = {};
  try { input = JSON.parse(raw || '{}'); } catch {}
  runBridge({ runtime: RUNTIME_RAW, eventFlag: EVENT_FLAG, configPath: CONFIG_FLAG, input })
    .then(({ exitCode = 0, stdout = {}, stderr = '' }) => {
      if (stderr) process.stderr.write(stderr.endsWith('\n') ? stderr : `${stderr}\n`);
      const finish = () => process.exit(exitCode);
      if (stdout !== null && stdout !== undefined) {
        process.stdout.write(`${JSON.stringify(stdout)}\n`, finish);
      } else {
        finish();
      }
    })
    .catch((e) => {
      const rt = normalizeRuntime(RUNTIME_RAW);
      const canon = normalizeEvent(rt, EVENT_FLAG, input);
      const out = canon === 'PreToolUse'
        ? formatRuntimeOutput(rt, EVENT_FLAG, canon, { deny: true, reason: `Vestige bridge error (${e && e.message}). Fail-closed.` })
        : { exitCode: 0, stdout: {} };
      if (out.stderr) process.stderr.write(`${out.stderr}\n`);
      const finish = () => process.exit(out.exitCode || 0);
      if (out.stdout !== null && out.stdout !== undefined) {
        process.stdout.write(`${JSON.stringify(out.stdout)}\n`, finish);
      } else {
        finish();
      }
    });
}

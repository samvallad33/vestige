'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');
const readline = require('readline');
const { spawnSync } = require('child_process');
const { defaults } = require('./memory-triggers.cjs');
const { rpc } = require('../hooks/memory.cjs');
const VERSION = require('../package.json').version;
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
const json = value => JSON.stringify(value, null, 2) + '\n';
function supportedVersion(text) {
  const match = /\b(\d+)\.(\d+)\.(\d+)\b/.exec(text || '');
  if (!match) return false;
  const [major, minor, patch] = match.slice(1).map(Number);
  return major > 5 || major === 5 && (minor > 0 || patch >= 1);
}
// Reflex hooks call a server tool that ships in 6.0.0. An older binary would
// answer every hook with an unknown-tool error, so they register only when the
// installed server is new enough.
const REFLEX_MIN = [6, 0, 0];
function reflexReady(binaryPath, env) {
  if (!binaryPath) return false;
  let out = '';
  try { out = spawnSync(binaryPath, ['--version'], { timeout: 2000, env, encoding: 'utf8' }).stdout || ''; } catch { return false; }
  const match = /\b(\d+)\.(\d+)\.(\d+)\b/.exec(out);
  if (!match) return false;
  const have = match.slice(1).map(Number);
  for (let i = 0; i < 3; i++) if (have[i] !== REFLEX_MIN[i]) return have[i] > REFLEX_MIN[i];
  return true;
}
// Claude Code calls reflex straight from `mcp_tool` hooks on the agent's own MCP
// connection. SessionStart at launch fires before MCP servers connect, so it, and
// every host without `mcp_tool` hooks, goes through the command shim.
// `agent` is the subagent's id and empty for the main agent, so every Vestige
// read lands in the context of the agent that made it.
// `error` and `interrupt` let Failure Pairing classify a failed call; `file`
// names what an edit tool wrote before the fix.
const REFLEX_INPUT = { event: '${hook_event_name}', session: '${session_id}', tool: '${tool_name}', call: '${tool_use_id}', agent: '${agent_id}', command: '${tool_input.command}', source: '${source}',
  error: '${error}', interrupt: '${is_interrupt}', file: '${tool_input.file_path}', cwd: '${cwd}',
  summary: '${compact_summary}', trigger: '${trigger}', host: 'claude' };
const REFLEX_EVENTS = {
  claude: [['PreToolUse', '*'], ['PostToolUse', '*'], ['PostToolUseFailure', '*'], ['UserPromptSubmit'], ['SubagentStart'], ['SubagentStop'], ['PreCompact'], ['PostCompact'], ['Stop']],
  codex: [['SessionStart'], ['UserPromptSubmit'], ['PreToolUse', '*'], ['PostToolUse', '*'], ['SubagentStart'], ['PreCompact'], ['Stop']],
  gemini: [['SessionStart'], ['BeforeAgent'], ['BeforeTool', '*'], ['AfterTool', '*'], ['PreCompress'], ['AfterAgent']],
  cursor: [['sessionStart'], ['beforeSubmitPrompt'], ['preToolUse'], ['postToolUse'], ['subagentStart'], ['preCompact'], ['stop']],
};
function reflexHooks(dialect, shimCommand) {
  const shim = matcher => dialect === 'cursor' ? { command: shimCommand, timeout: 2, failClosed: false }
    : { ...(matcher ? { matcher } : {}), hooks: [{ type: 'command', command: shimCommand, timeout: 2 }] };
  const rows = (REFLEX_EVENTS[dialect] || []).map(([event, matcher]) => [event, dialect === 'claude'
    ? { ...(matcher ? { matcher } : {}), hooks: [{ type: 'mcp_tool', server: 'vestige', tool: 'reflex', input: REFLEX_INPUT }] }
    : shim(matcher)]);
  if (dialect === 'claude') rows.push(['SessionStart', { matcher: 'startup|resume|clear|compact', hooks: [{ type: 'command', command: shimCommand, timeout: 2 }] }]);
  return rows;
}
const read = file => fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : null;
function object(file) {
  const raw = read(file); const value = raw === null || !raw.trim() ? {} : JSON.parse(raw);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${file} must contain a JSON object; nothing changed`);
  return value;
}
function executable(name, home, platform, env = process.env) {
  const suffixes = platform === 'win32' ? ['', '.exe', '.cmd', '.bat'] : [''];
  const dirs = (env.PATH || '').split(platform === 'win32' ? ';' : ':').concat([
    path.join(home, '.local', 'bin'), path.join(home, '.cargo', 'bin'), path.join(home, '.vestige', 'bin')]);
  for (const dir of dirs.filter(Boolean)) for (const suffix of suffixes) {
    const file = path.resolve(dir, name + suffix);
    try { fs.accessSync(file, platform === 'win32' ? fs.constants.F_OK : fs.constants.X_OK); if (fs.statSync(file).isFile()) return file; } catch {}
  }
  return null;
}
function binary(home, platform, env) {
  if (env.VESTIGE_MCP_PATH) return fs.existsSync(env.VESTIGE_MCP_PATH) ? path.resolve(env.VESTIGE_MCP_PATH) : null;
  return executable('vestige-mcp', home, platform, env);
}
// No shell interpolation of paths. POSIX apostrophes are escaped; Windows gets
// double-quoted argv atoms and refuses shell expansion metacharacters in paths.
function quote(text, platform) {
  if (platform === 'win32') {
    if (/["%\r\n]/.test(text)) throw new Error('Windows hook paths must not contain quotes, percent signs, or newlines');
    return `"${text}"`;
  }
  return `'${text.replace(/'/g, `'"'"'`)}'`;
}
function catalog(home, platform, env = process.env) {
  const roaming = env.APPDATA || path.join(home, 'AppData', 'Roaming');
  const support = platform === 'darwin' ? path.join(home, 'Library', 'Application Support') : platform === 'win32' ? roaming : path.join(env.XDG_CONFIG_HOME || path.join(home, '.config'));
  const entries = [
    ['claude', 'Claude Code', '.claude', path.join(home, '.claude.json'), ['mcpServers'], path.join(home, '.claude', 'settings.json'), 'claude'],
    ['codex', 'Codex', '.codex', path.join(home, '.codex', 'config.toml'), null, path.join(home, '.codex', 'hooks.json'), 'codex'],
    ['gemini', 'Gemini CLI', '.gemini', path.join(home, '.gemini', 'settings.json'), ['mcpServers'], path.join(home, '.gemini', 'settings.json'), 'gemini'],
    ['cursor', 'Cursor', '.cursor', path.join(home, '.cursor', 'mcp.json'), ['mcpServers'], path.join(home, '.cursor', 'hooks.json'), 'cursor'],
    ['claude-desktop', 'Claude Desktop', null, path.join(support, 'Claude', 'claude_desktop_config.json'), ['mcpServers']],
    ['code', 'VS Code', null, path.join(support, 'Code', 'User', 'mcp.json'), ['servers']],
    ['opencode', 'OpenCode', '.config/opencode', path.join(home, '.config', 'opencode', 'opencode.json'), ['mcp']],
    ['windsurf', 'Windsurf', '.codeium/windsurf', path.join(home, '.codeium', 'windsurf', 'mcp_config.json'), ['mcpServers']],
    ['junie', 'JetBrains Junie', '.junie', path.join(home, '.junie', 'mcp', 'mcp.json'), ['mcpServers']],
    ['zed', 'Zed', '.config/zed', path.join(home, '.config', 'zed', 'settings.json'), ['context_servers']],
    ['kiro', 'Kiro', '.kiro', path.join(home, '.kiro', 'settings', 'mcp.json'), ['mcpServers']],
    ['antigravity', 'Antigravity', '.gemini/config', fs.existsSync(path.join(home,'.gemini','antigravity','mcp_config.json')) ? path.join(home,'.gemini','antigravity','mcp_config.json') : path.join(home, '.gemini', 'config', 'mcp_config.json'), ['mcpServers']],
  ];
  if (platform === 'darwin') entries.push(['xcode','Xcode', 'Library/Developer/Xcode', path.join(home,'Library','Developer','Xcode','CodingAssistant','ClaudeAgentConfig','.claude'), ['projects','*','mcpServers']]);
  return entries.map(([id, name, marker, file, keys, hookFile, dialect]) => ({ id, name, file, keys, hookFile, dialect,
    detected: fs.existsSync(file) || Boolean(marker && fs.existsSync(path.join(home, marker))) || Boolean(executable(id, home, platform, env)),
    coverage: dialect === 'cursor' ? 'session context, prompt capture/recall, post-tool context, turn save, precompact, end; prompt injection awaits a tool result' : dialect ? 'session, prompt, turn, precompact, end (client trust/restart required)' : 'MCP tools; lifecycle hooks unavailable in this adapter' }));
}
const EVENTS = {
  claude: { SessionStart: 'session', UserPromptSubmit: 'prompt', Stop: 'turn', PreCompact: 'compact', SessionEnd: 'end' },
  codex: { SessionStart: 'session', UserPromptSubmit: 'prompt', Stop: 'turn', PreCompact: 'compact', SessionEnd: 'end' },
  gemini: { SessionStart: 'session', BeforeAgent: 'prompt', AfterAgent: 'turn', PreCompress: 'compact', SessionEnd: 'end' },
  cursor: { sessionStart: 'session', beforeSubmitPrompt: 'prompt', afterAgentResponse: 'turn', preCompact: 'compact', sessionEnd: 'end', postToolUse: 'tool' },
};
function nativeGuardRegistered(value, gate, dialect) {
  // Recognize the native argv, including Lite's POSIX existence check. A name
  // containing "guard" is not evidence of a gate, nor is a partial matcher.
  const words = text => (text.match(/"[^"]*"|'[^']*'|[^\s]+/g) || []).map(s => s.replace(/^(['"])(.*)\1$/, '$2'));
  return (value || []).some(group => {
    if (group.matcher && !['*', '.*'].includes(group.matcher)) return false;
    return (group.hooks || [group]).some(handler => {
      if (typeof handler.command !== 'string') return false;
      let argv = words(handler.command);
      if (argv[0] === '[') {
        if (JSON.stringify(argv.slice(0, 6)) !== JSON.stringify(['[','-f',gate,']','&&','python3'])) return false;
        argv = argv.slice(5);
      }
      if (!/^python(?:3(?:\.\d+)?)?(?:\.exe)?$/i.test(path.basename(argv[0] || ''))) return false;
      argv = argv.slice(1); if (argv[0] === '-I') argv = argv.slice(1);
      if (argv[0] === '-c' && argv.length === 2) {
        const code = /^import os,runpy,sys;g=('(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*");sys.argv=\[g,'hook','--source','claude'\];os.path.isfile\(g\) and runpy.run_path\(g,run_name='__main__'\)$/.exec(argv[1]);
        return dialect === 'claude' && Boolean(code) && code[1].slice(1,-1).replace(/\\(['"\\])/g,'$1') === gate.replace(/\\/g,'/');
      }
      return JSON.stringify(argv) === JSON.stringify([gate,'hook','--source',dialect]);
    });
  });
}
// Zed and VS Code allow comments and trailing commas in their JSON. Strings are left alone.
function withoutComments(text) {
  let out = ''; let inString = false; let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]; const n = text[i + 1];
    if (inString) { out += c; if (escaped) escaped = false; else if (c === '\\') escaped = true; else if (c === '"') inString = false; continue; }
    if (c === '"') { inString = true; out += c; continue; }
    if (c === '/' && n === '/') { while (i < text.length && text[i] !== '\n') i++; out += '\n'; continue; }
    if (c === '/' && n === '*') { i += 2; while (i < text.length && !(text[i] === '*' && text[i + 1] === '/')) i++; i++; continue; }
    out += c;
  }
  return out.replace(/,\s*([}\]])/g, '$1');
}
// A config with comments is never rewritten, because rewriting would drop the owner's comments;
// its client is set aside to be added by hand (#515). A config that does not parse even without
// its comments still refuses setup before any change, as it always has.
function commented(file) {
  let raw; try { raw = read(file); } catch { return false; }
  if (raw === null || !raw.trim()) return false;
  try { JSON.parse(raw); return false; } catch {}
  try { JSON.parse(withoutComments(raw)); return true; } catch { return false; }
}
// A config that is a symlink, as dotfile managers leave it, is not rewritten through the link, and a config
// whose MCP or hooks section is not an object (an array, say) cannot take the entry. Either way only that
// client is set aside to be added by hand; every other client is still set up.
function symlinked(file) { try { return fs.lstatSync(file).isSymbolicLink(); } catch { return false; } }
function malformed(file, keys) {
  let raw; try { raw = read(file); } catch { return false; }
  if (raw === null || !raw.trim()) return false;
  let cursor; try { cursor = JSON.parse(raw); } catch { return false; }
  for (const key of keys) {
    if (cursor[key] === undefined) return false;
    if (!cursor[key] || typeof cursor[key] !== 'object' || Array.isArray(cursor[key])) return true;
    cursor = cursor[key];
  }
  return false;
}
function setAside(c) {
  const json = [c.id === 'codex' ? null : c.file, c.hookFile].filter(Boolean);
  const file = json.find(commented);
  if (file) return { file, reason: 'comments' };
  const linked = [c.file, c.hookFile].filter(Boolean).find(symlinked);
  if (linked) return { file: linked, reason: 'symlink' };
  if (c.id !== 'codex' && c.keys && malformed(c.file, c.keys)) return { file: c.file, reason: 'section' };
  if (c.hookFile && malformed(c.hookFile, ['hooks'])) return { file: c.hookFile, reason: 'section' };
  return null;
}
function plan({ home = os.homedir(), platform = os.platform(), env = process.env, guard = false, clients, node = process.execPath } = {}) {
  const found = binary(home, platform, env);
  const handAdd = [];
  const detected = catalog(home, platform, env).filter(c => c.detected && (!clients || clients.includes(c.id))).filter(c => {
    const aside = setAside(c);
    if (aside) handAdd.push({ id: c.id, name: c.name, file: aside.file, keys: c.keys, reason: aside.reason });
    return !aside;
  });
  const base = path.join(home, '.vestige', 'memory'); const manifestFile = path.join(base, 'manifest.json');
  if (!found) return { missing: true, detected, handAdd, changes: [], home, base, manifestFile };
  const previous = fs.existsSync(manifestFile) ? object(manifestFile) : { files: [] };
  const reflex = reflexReady(found, env);
  const changes = new Map();
  const change = (file, description) => {
    if (!changes.has(file)) {
      if (fs.existsSync(file) && fs.lstatSync(file).isSymbolicLink()) throw new Error(`Review symlink before setup: ${file}`);
      const before = read(file); changes.set(file, { file, before, after: before, description, patches: [] });
    }
    return changes.get(file);
  };
  const add = (file, keys, value, description, append = false) => {
    const row = change(file, description); const value_ = row.after === null || !row.after.trim() ? {} : JSON.parse(row.after);
    let cursor = value_;
    for (const key of keys.slice(0, -1)) {
      if (cursor[key] === undefined) cursor[key] = {};
      if (!cursor[key] || typeof cursor[key] !== 'object' || Array.isArray(cursor[key])) throw new Error(`Review ${file}: ${key} is not an object`);
      cursor = cursor[key];
    }
    const key = keys.at(-1);
    if (append) {
      if (cursor[key] === undefined) cursor[key] = [];
      if (!Array.isArray(cursor[key])) throw new Error(`Review ${file}: ${key} is not an array`);
      if (!cursor[key].some(v => JSON.stringify(v) === JSON.stringify(value))) { cursor[key].push(value); row.patches.push({ keys, value, append: true }); }
    } else if (cursor[key] === undefined) { cursor[key] = value; row.patches.push({ keys, value }); }
    row.after = json(value_);
  };
  for (const client of detected) {
    if (client.id === 'codex') {
      const row = change(client.file, 'append MCP connection');
      if (!/(?:^|\n)\s*\[mcp_servers\.vestige(?:\]|\.)/.test(row.before || '')) {
        if (/(?:^|\n)\s*mcp_servers\s*=/.test(row.before || '')) throw new Error('Inline Codex MCP config requires manual merge');
        const block = `\n[mcp_servers.vestige]\ncommand = ${JSON.stringify(found)}\nargs = []\n`;
        row.after = (row.before || '') + block; row.block = block;
      }
    } else {
      const entry = client.id === 'opencode' ? { type: 'local', command: [found], enabled: true } :
        ['code','xcode'].includes(client.id) ? { type:'stdio', command:found, args:[] } : { command: found, args: [] };
      add(client.file, [...client.keys, 'vestige'], entry, 'merge MCP connection');
    }
    if (client.hookFile) {
      if (client.dialect === 'cursor') add(client.hookFile, ['version'], 1, 'merge memory hooks');
      for (const [eventName, event] of Object.entries(EVENTS[client.dialect])) {
        const command = [node, path.join(base, 'hooks', 'memory.cjs'), '--config', path.join(base, 'config.json'), '--client', client.dialect, '--event', event].map(p => quote(p, platform)).join(' ');
        const handler = { type: 'command', command, timeout: 1 };
        const entry = client.dialect === 'cursor' ? { command, timeout: 1, failClosed: false } : { hooks: [handler] };
        add(client.hookFile, ['hooks', eventName], entry, 'merge memory hooks alongside existing hooks', true);
      }
      if (reflex) {
        const shimCommand = [node, path.join(base, 'hooks', 'reflex.cjs'), '--config', path.join(base, 'config.json'), '--client', client.dialect].map(p => quote(p, platform)).join(' ');
        for (const [eventName, entry] of reflexHooks(client.dialect, shimCommand)) add(client.hookFile, ['hooks', eventName], entry, 'merge reflex hooks: every hook firing reaches Vestige', true);
      }
    }
  }
  for (const [relative, source] of [ ['hooks/memory.cjs', '../hooks/memory.cjs'], ['hooks/reflex.cjs', '../hooks/reflex.cjs'], ['lib/memory-triggers.cjs', './memory-triggers.cjs'] ]) {
    const row = change(path.join(base, relative), 'install bounded memory hook'); row.after = fs.readFileSync(path.resolve(__dirname, source), 'utf8');
  }
  const triggers = change(path.join(base, 'triggers.json'), 'user-editable recall trigger phrases');
  if (triggers.before === null) triggers.after = json(defaults);
  const config = change(path.join(base, 'config.json'), 'local memory connection');
  const configValue = config.before === null ? {} : JSON.parse(config.before);
  config.after = json({ ...configValue, version: VERSION, binary: found, triggers: triggers.file });
  let guardInstall = null;
  if (guard) {
    const python = ['python3', 'python'].map(name => executable(name, home, platform, env)).find(bin => {
      if (!bin) return false;
      const result = spawnSync(bin, ['-c', 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)'], { timeout: 2000, env });
      return result.status === 0;
    });
    if (!python) throw new Error('Memory plus guard needs Python 3.9+. Install Python, then rerun with --guard.');
    const gate = path.join(home, '.operator', 'gate', 'operator-gate.py');
    if (!fs.existsSync(gate)) {
      const source = path.join(__dirname, '..', 'hooks', 'operator-lite.py');
      const row = change(gate, 'install bundled free Operator Lite 6.0.0 (owner terminal required)'); row.after = fs.readFileSync(source, 'utf8');
      const mode = change(path.join(home, '.operator', 'mode'), 'start new guard in enforce mode');
      if (mode.before === null) mode.after = 'enforce\n';
      guardInstall = { source, python };
    }
    // The gate already protects every descendant of its home. Keep authority
    // outside user-editable memory settings and outside attested gate modules.
    const guardBase = path.join(home, '.operator', 'state', 'vestige-memory');
    const authority = change(path.join(guardBase, 'config.json'), 'install protected guard routing');
    authority.after = json({ gate, python });
    const adapter = change(path.join(guardBase, 'guard.cjs'), 'install protected fail-closed host adapter');
    adapter.after = fs.readFileSync(path.join(__dirname, '..', 'hooks', 'guard.cjs'), 'utf8');
    for (const client of detected.filter(c => c.hookFile)) {
      const event = client.dialect === 'cursor' ? 'preToolUse' : client.dialect === 'gemini' ? 'BeforeTool' : 'PreToolUse';
      const current = change(client.hookFile, 'merge optional pre-tool guard');
      if (nativeGuardRegistered(JSON.parse(current.after || '{}').hooks?.[event], gate, client.dialect)) continue;
      const command = [node, adapter.file, '--config', authority.file, '--client', client.dialect].map(p => quote(p, platform)).join(' ');
      const entry = client.dialect === 'cursor' ? { command, timeout: 12, failClosed: true } : { matcher: '*', hooks: [{ type: 'command', command, timeout: 12 }] };
      add(client.hookFile, ['hooks', event], entry, 'merge optional pre-tool guard', true);
    }
    const reflex = change(path.join(base, 'hooks', 'operator-stop-memory.py'), 'install stop-to-Strata reflex');
    reflex.after = fs.readFileSync(path.join(__dirname, '..', 'hooks', 'operator-stop-memory.py'), 'utf8');
    const relay = change(path.join(base, 'hooks', 'stop-memory.cjs'), 'install bounded fail-open stop-memory relay');
    relay.after = fs.readFileSync(path.join(__dirname, '..', 'hooks', 'stop-memory.cjs'), 'utf8');
    for (const client of detected.filter(c => ['claude','codex'].includes(c.dialect))) {
      const command = [node, relay.file, '--python', python, '--script', reflex.file, '--mcp', found].map(p => quote(p, platform)).join(' ');
      for (const event of ['PostToolUse', 'UserPromptSubmit', 'Stop', 'SessionStart']) add(client.hookFile, ['hooks', event], { hooks: [{ type: 'command', command, timeout: 1 }] }, 'merge bounded stop-to-Strata reflex', true);
    }
  }
  const rows = [...changes.values()].filter(r => r.after !== r.before && (r.patches.length || r.block || !detected.some(c => c.file === r.file || c.hookFile === r.file)));
  return { home, base, manifestFile, previous, binary: found, detected, handAdd, changes: rows, guard, guardInstall };
}
function atomic(file, contents) {
  fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temp, contents, { mode: 0o600 }); fs.renameSync(temp, file);
}
function apply(plan_) {
  if (!plan_.changes.length) return { written: 0, backups: [] };
  if (plan_.guardInstall) {
    // Reuse Lite's actual owner-side installation check; never strip agent environment.
    const check = spawnSync(plan_.guardInstall.python, ['-c', 'import runpy,sys; g=runpy.run_path(sys.argv[1]); a=g["agent_session"](); print(a or "owner terminal"); sys.exit(3 if a else 0)', plan_.guardInstall.source], { encoding: 'utf8', timeout: 3000 });
    if (check.status !== 0) throw new Error('Operator Lite installation is owner-terminal only. Run npx @vestige/init --guard in your own terminal; nothing changed.');
  }
  // Freeze the preview: if a client edits any target, recompute before another confirm.
  for (const row of plan_.changes) if (read(row.file) !== row.before) throw new Error(`Changed since preview: ${row.file}; rerun setup`);
  const records = new Map((plan_.previous.files || []).map(r => [r.file, r])); const backups = []; const written = [];
  try {
    for (const row of plan_.changes) {
      let backup = null;
      if (row.before !== null) { backup = `${row.file}.vestige-backup-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`; fs.copyFileSync(row.file, backup, fs.constants.COPYFILE_EXCL); fs.chmodSync(backup, 0o600); backups.push(backup); }
      const prior = records.get(row.file);
      atomic(row.file, row.after); written.push(row);
      records.set(row.file, { file: row.file, backup: prior ? prior.backup : backup, originalHash: prior ? prior.originalHash : row.before === null ? null : hash(row.before),
        installedHash: hash(row.after), patches: [...(prior?.patches || []), ...row.patches], block: row.block || prior?.block });
    }
    if (fs.existsSync(plan_.manifestFile)) { const backup = `${plan_.manifestFile}.vestige-backup-${Date.now()}`; fs.copyFileSync(plan_.manifestFile, backup); backups.push(backup); }
    const clients = new Map((plan_.previous.clients || []).map(c => [c.id,c]));
    for (const c of plan_.detected) clients.set(c.id,{id:c.id,name:c.name,hookFile:c.hookFile,coverage:c.coverage});
    atomic(plan_.manifestFile, json({ version: VERSION, binary: plan_.binary, guard: Boolean(plan_.guard || plan_.previous.guard), clients: [...clients.values()], files: [...records.values()] }));
    return { written: written.length, backups };
  } catch (error) {
    for (const row of written.reverse()) { if (row.before === null) fs.unlinkSync(row.file); else atomic(row.file, row.before); }
    throw error;
  }
}
function uninstall(home = os.homedir(), dryRun = false) {
  const file = path.join(home, '.vestige', 'memory', 'manifest.json'); if (!fs.existsSync(file)) return { removed: [], kept: [] };
  const manifest = object(file); const removed = []; const kept = []; const operations = [];
  for (const record of [...manifest.files].reverse()) {
    const current = read(record.file); if (current === null) continue;
    if (hash(current) === record.installedHash) {
      let after = null;
      if (record.backup) { after = fs.readFileSync(record.backup,'utf8'); if (hash(after)!==record.originalHash) throw new Error(`Backup changed: ${record.backup}`); }
      operations.push({ record, current, after }); continue;
    }
    if (!record.patches?.length && !record.block) { kept.push(record.file); continue; }
    let result = current;
    if (record.block && current.includes(record.block)) result = current.replace(record.block, '');
    if (record.patches?.length) {
      let value; try { value = JSON.parse(current); } catch { kept.push(record.file); continue; }
      for (const patch of [...record.patches].reverse()) {
        let cursor = value;
        for (const key of patch.keys.slice(0, -1)) cursor = cursor?.[key];
        if (!cursor) continue; const key = patch.keys.at(-1);
        if (patch.append && Array.isArray(cursor[key])) cursor[key] = cursor[key].filter(v => JSON.stringify(v) !== JSON.stringify(patch.value));
        else if (JSON.stringify(cursor[key]) === JSON.stringify(patch.value)) delete cursor[key];
      }
      // Remove only empty containers introduced by our patches. A user's
      // pre-existing empty container is part of the original configuration.
      let original = {};
      if (record.backup) { try { original = JSON.parse(fs.readFileSync(record.backup,'utf8')); } catch {} }
      for (const patch of record.patches) for (let count=patch.keys.length;count>0;count--) {
        const keys=patch.keys.slice(0,count); let parent=value; let before=original;
        for (const key of keys.slice(0,-1)) { parent=parent?.[key]; before=before?.[key]; }
        const key=keys.at(-1); const child=parent?.[key];
        if (parent && before?.[key] === undefined && child && typeof child==='object' && Object.keys(child).length===0) delete parent[key];
      }
      result = json(value);
    }
    operations.push({ record, current, after:result });
  }
  const base=path.dirname(file);
  const guardBase=path.join(home,'.operator','state','vestige-memory');
  const referencesRuntime=text => {
    if (!text) return false;
    const visit=value => typeof value==='string' ? value.includes(base) || value.includes(guardBase) : value && typeof value==='object' ? Object.values(value).some(visit) : false;
    try { return visit(JSON.parse(text)); } catch { return text.includes(base) || text.includes(guardBase); }
  };
  const referenced=operations.some(op => (op.record.patches?.length || op.record.block) && referencesRuntime(op.after)) || kept.some(name => referencesRuntime(read(name)));
  // A customized hook is now the user's entry. Keep everything it needs;
  // removing the adapter underneath a fail-closed hook would break tool use.
  for (const op of operations) {
    if (referenced && !op.record.patches?.length && !op.record.block) { kept.push(op.record.file); continue; }
    if (!dryRun && op.after!==op.current) {
      if (read(op.record.file)!==op.current) throw new Error(`Changed during uninstall: ${op.record.file}`);
      fs.copyFileSync(op.record.file, `${op.record.file}.vestige-uninstall-backup-${Date.now()}`);
      if (op.after===null) fs.unlinkSync(op.record.file); else atomic(op.record.file,op.after);
    }
    removed.push(op.record.file);
  }
  if (!dryRun && !kept.length) fs.unlinkSync(file);
  return { removed, kept };
}
async function inspect(home = os.homedir(), timeout = 2000) {
  const file = path.join(home, '.vestige', 'memory', 'manifest.json');
  const manifest = fs.existsSync(file) ? object(file) : null;
  const checks = []; const add = (name, ok, fix = '') => checks.push({ name, ok: Boolean(ok), fix });
  add('installation manifest', manifest, 'Run npx @vestige/init');
  const bin = manifest?.binary || binary(home, os.platform(), process.env);
  const version = bin && spawnSync(bin, ['--version'], { encoding: 'utf8', timeout: 2000 });
  add('vestige-mcp v5.0.1+', version?.status === 0 && supportedVersion(version.stdout), 'Install v5.0.1 from GitHub Releases, then rerun init');
  for (const client of manifest?.clients || []) if (client.hookFile) {
    for (const event of Object.keys(EVENTS[client.id] || {})) {
      let live = false;
      try {
        const value = object(client.hookFile);
        const expected = manifest.files.find(r=>r.file===client.hookFile)?.patches?.filter(p=>p.keys.join('.')===`hooks.${event}` && JSON.stringify(p.value).includes('memory.cjs')) || [];
        live = expected.length > 0 && expected.every(p=>(value.hooks?.[event] || []).some(v=>JSON.stringify(v)===JSON.stringify(p.value)));
      } catch {}
      add(`${client.name}: ${event} registered (client trust/reload required)`, live, 'Rerun init, then trust/reload hooks in this client');
    }
    if (manifest.guard) {
      const dialect=client.id; const event=dialect==='cursor'?'preToolUse':dialect==='gemini'?'BeforeTool':'PreToolUse';
      let live=false;
      try {
        const groups=object(client.hookFile).hooks?.[event] || [];
        const owned=manifest.files.find(r=>r.file===client.hookFile)?.patches?.filter(p=>p.keys.join('.')===`hooks.${event}` && JSON.stringify(p.value).includes('guard.cjs')) || [];
        live=nativeGuardRegistered(groups,path.join(home,'.operator','gate','operator-gate.py'),dialect) || owned.some(p=>groups.some(g=>JSON.stringify(g)===JSON.stringify(p.value)));
      } catch {}
      add(`${client.name}: optional guard registered`,live,'Rerun init --guard in your terminal and trust/reload hooks');
    }
  }
  for (const relative of ['hooks/memory.cjs','lib/memory-triggers.cjs','config.json','triggers.json']) add(`memory runtime: ${relative}`, fs.existsSync(path.join(home,'.vestige','memory',relative)), 'Rerun init to restore missing runtime files');
  if (manifest?.guard) for (const relative of ['guard.cjs','config.json']) add(`protected guard runtime: ${relative}`,fs.existsSync(path.join(home,'.operator','state','vestige-memory',relative)),'Rerun init --guard in your terminal');
  let packet = null; let transport;
  try { if (!bin) throw new Error('binary missing'); transport = rpc(bin, [], timeout); await transport.ready(); packet = await transport.tool('session_start', { memory_hook: { limit: 3 } }); }
  catch { /* doctor renders the explicit failed check */ } finally { transport?.close(); }
  add('Strata responds with memory-hook schema', Array.isArray(packet?.memories), 'Start or repair vestige-mcp; inspect ~/.vestige/memory/errors.log');
  add('Backfill autofire enabled in responding process', packet?.backfill_autofire === true, 'Unset VESTIGE_BACKFILL_AUTOFIRE=off and restart MCP clients');
  return { checks, memory_count: packet?.memory_count ?? null, last_backfill_receipts: packet?.backfill_receipts ?? [], clients: manifest?.clients || [] };
}
async function main(args = process.argv.slice(2)) {
  const home = os.homedir(); const has = flag => args.includes(flag); const machine = has('--json');
  const say = line => { if (!machine) console.log(line); };
  const command = args[0];
  if (has('--help') || has('-h')) { console.log('vestige-init [--yes] [--guard] [--dry-run] [--json]\nvestige-init doctor | status | uninstall [--yes] [--dry-run]\nMemory only is the default. --guard requests the optional Operator Lite setup.'); return; }
  if (command === 'doctor' || command === 'status') {
    const result = await inspect(home); if (machine) console.log(json(result)); else { say(`Vestige ${VERSION} · ${command}`); for (const check of result.checks) say(`${check.ok ? 'GREEN' : 'RED  '} ${check.name}${check.ok ? '' : ` — fix: ${check.fix}`}`); say(`Memories: ${result.memory_count ?? 'unavailable'}`); say(`Recent Backfill receipts: ${result.last_backfill_receipts.map(r => r.receipt_id).join(', ') || 'none'}`); }
    if (command === 'doctor' && result.checks.some(c => !c.ok)) process.exitCode = 1; return;
  }
  // One readline stream preserves piped answers instead of losing the second line.
  let terminal; let lines;
  const ask = async (question, final = false) => { say(question); terminal ||= readline.createInterface({ input: process.stdin, crlfDelay: Infinity }); lines ||= terminal[Symbol.asyncIterator](); const answer = await lines.next(); if (final) terminal.close(); return (answer.value || '').trim(); };
  if (command === 'uninstall') {
    const preview = uninstall(home, true); for (const file of preview.removed) say(`RESTORE/REMOVE ${file}`);
    if (!has('--yes') && !has('--dry-run') && !/^y(?:es)?$/i.test(await ask('Remove only Vestige additions and restore unchanged backed-up configs? [y/N]', true))) return;
    const result = has('--dry-run') ? preview : uninstall(home); if (machine) console.log(json(result)); else say(`Removed/restored ${result.removed.length} files; kept ${result.kept.length} edited files. Memory store and backups retained.`); return;
  }
  let guard = has('--guard');
  say(`╭─ VESTIGE ${VERSION} · memory that stays with you ─╮`);
  const initial = plan({ home });
  if (initial.missing) { const instructions = 'Install vestige-mcp from https://github.com/samvallad33/vestige/releases (choose your platform, extract the archive, put vestige-mcp on PATH), then rerun npx @vestige/init.'; say(instructions); if (machine) console.log(json({ status: 'binary-missing', instructions, changed: [] })); return; }
  say(`FOUND ${initial.binary}`);
  const installed = spawnSync(initial.binary, ['--version'], { encoding: 'utf8', timeout: 2000 });
  if (installed.status !== 0 || !supportedVersion(installed.stdout)) {
    const instructions = 'Memory hooks require vestige-mcp v5.0.1+. Install it from https://github.com/samvallad33/vestige/releases, then rerun init.';
    say(instructions); if (machine) console.log(json({ status: 'binary-upgrade-required', instructions, changed: [] })); return;
  }
  for (const c of initial.detected) say(`  ${c.name} — ${c.coverage}`);
  if (!has('--yes') && !has('--dry-run') && !guard) guard = /^2$|guard/i.test(await ask('Memory only [1, default] or Memory plus guard [2]?'));
  const requests = args.flatMap((a, i) => a === '--client' ? (args[i + 1] || '').split(',') : []);
  const planned = plan({ home, guard, clients: requests.length ? requests : undefined });
  say('Preview — these files will change (existing configs get private backups):');
  for (const c of planned.changes) {
    say(`  ${c.before === null ? 'CREATE' : 'MERGE '} ${c.file} — ${c.description}`);
    for (const patch of c.patches) say(`    + ${patch.keys.join('.')}${patch.append ? ' (append alongside existing entries)' : ''}`);
    if (c.before !== null) say(`    backup: ${c.file}.vestige-backup-<timestamp>-<nonce>`);
  }
  if (planned.changes.length) say(`  WRITE  ${planned.manifestFile} — exact ownership and rollback record`);
  for (const h of planned.handAdd) {
    const why = { comments: 'it has comments', symlink: 'it is a symlink, which setup does not write through',
      section: 'its MCP or hooks section is not an object' }[h.reason] || 'it has comments';
    say(`  SKIP   ${h.file} — ${why}, so it is left exactly as it is.`);
    if (h.keys) say(`    To connect ${h.name}, add this under ${h.keys.join('.')}: "vestige": ${JSON.stringify({ command: planned.binary, args: [] })}`);
  }
  if (has('--dry-run')) { if (machine) console.log(json({ status: 'preview', detected: planned.detected.map(c => c.id), skipped: planned.handAdd.map(h => h.id), files: planned.changes.map(c => ({ file: c.file, description: c.description })) })); return; }
  if (!has('--yes') && !/^y(?:es)?$/i.test(await ask('Apply this preview? [y/N]', true))) { say('Nothing changed.'); return; }
  const result = apply(planned);
  if (machine) { console.log(json({ status: 'installed', ...result, clients: planned.detected.map(c => c.id), skipped: planned.handAdd.map(h => h.id) })); return; }
  say(`✓ Memory ready · ${result.written} files · ${result.backups.length} backups`);
  if (guard) say(`  Operator guard configured; owner permits: operator-gate approve <action-id> (or Python + ~/.operator/gate/operator-gate.py approve <action-id>)`);
  say(`  Installed in ${planned.base}`);
  say('  Restart clients; Codex also requires review/trust in /hooks. MCP-only clients use the memory tools.');
  say('  Try: Tell me something about a friend, then ask me about them later.');
  say('  Check: npx @vestige/init doctor');
}
module.exports = { plan, apply, uninstall, inspect, catalog, quote, binary, supportedVersion, reflexReady, reflexHooks, nativeGuardRegistered, EVENTS, main };

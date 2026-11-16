const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const guard = path.join(__dirname, '../hooks/guard.cjs');
const gate = path.join(__dirname, '../../../operator/operator-gate.py');

function homeFor(name) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `vestige-guard-${name}-`));
  const operator = path.join(home, '.operator');
  fs.mkdirSync(path.join(operator, 'state'), { recursive: true });
  fs.writeFileSync(path.join(operator, 'mode'), 'enforce\n');
  const config = path.join(operator, 'config.json');
  fs.writeFileSync(config, JSON.stringify({ python: process.execPath.replace(/node$/, 'python3'), gate }));
  return { home, operator, config };
}

function python() {
  const found = spawnSync('python3', ['-c', 'print(1)'], { encoding: 'utf8' });
  assert.equal(found.status, 0);
  return 'python3';
}

function run(client, input, options = {}) {
  const { home, operator, config } = homeFor(client + '-' + Math.random().toString(16).slice(2));
  const cfg = JSON.parse(fs.readFileSync(config, 'utf8'));
  cfg.python = options.python || python();
  cfg.gate = options.gate || gate;
  fs.writeFileSync(config, JSON.stringify(cfg));
  const env = {
    ...process.env,
    HOME: home,
    USERPROFILE: home,
    OPERATOR_HOME: operator,
  };
  for (const name of ['CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'OPERATOR_AGENT_SESSION']) delete env[name];
  const started = performance.now();
  const result = spawnSync(process.execPath, [guard, '--config', config, '--client', client], {
    input, encoding: 'utf8', env, timeout: 20000,
  });
  result.elapsed = performance.now() - started;
  return result;
}

test('cursor deny carries the gate proof lines and an allow carries no reason', () => {
  const stopped = run('cursor', JSON.stringify({
    cwd: '/tmp', session_id: 'guard-deny', tool_name: 'Bash', tool_input: { command: 'git push --force origin main' },
  }));
  assert.equal(stopped.status, 2, stopped.stderr);
  const body = JSON.parse(stopped.stdout);
  assert.equal(body.permission, 'deny');
  assert.match(body.agent_message, /op-receipt-/);
  assert.match(body.user_message, /OPERATOR: STOPPED \(OP-004/);
  assert.match(body.agent_message, /Proof: recall\(handle="op-receipt-/);
  assert.match(body.user_message, /Rule: recall\(handle="op-rule-OP-004"\)/);
  assert.ok(body.agent_message.length <= 1200);
  const allowed = run('cursor', JSON.stringify({
    cwd: '/tmp', session_id: 'guard-allow', tool_name: 'Bash', tool_input: { command: 'echo hello' },
  }));
  assert.equal(allowed.status, 0, allowed.stderr);
  const allow = JSON.parse(allowed.stdout);
  assert.equal(allow.permission, 'allow');
  assert.equal(allow.user_message, undefined);
  assert.equal(allow.agent_message, undefined);
  assert.equal(allow.reason, undefined);
});

test('gemini deny uses decision and reason, and claude keeps exit 2', () => {
  const gemini = run('gemini', JSON.stringify({
    cwd: '/tmp', session_id: 'guard-gemini', tool_name: 'Bash', tool_input: { command: 'git push --force origin main' },
  }));
  assert.equal(gemini.status, 2, gemini.stderr);
  const body = JSON.parse(gemini.stdout);
  assert.equal(body.decision, 'deny');
  assert.match(body.reason, /op-receipt-/);
  const claude = run('claude', JSON.stringify({
    cwd: '/tmp', session_id: 'guard-claude', tool_name: 'Bash', tool_input: { command: 'git push --force origin main' },
  }));
  assert.equal(claude.status, 2);
  assert.match(claude.stderr, /OP-004/);
  assert.match(claude.stderr, /op-receipt-/);
  assert.equal(claude.stdout, '');
});

test('invalid JSON and a missing interpreter deny without inventing a handle', () => {
  const broken = run('cursor', '{');
  assert.equal(broken.status, 2);
  const body = JSON.parse(broken.stdout);
  assert.equal(body.permission, 'deny');
  assert.doesNotMatch(body.agent_message, /op-receipt-/);
  const missing = run('cursor', '{}', { python: '/does/not/exist-python' });
  assert.equal(missing.status, 2);
  const denied = JSON.parse(missing.stdout);
  assert.equal(denied.permission, 'deny');
  assert.doesNotMatch(denied.agent_message + denied.user_message, /op-receipt-/);
  assert.match(denied.user_message, /Operator stopped/);
});

test('a huge stderr cannot become the whole prompt', () => {
  const script = path.join(os.tmpdir(), `vestige-huge-gate-${process.pid}.py`);
  fs.writeFileSync(script, [
    'import sys',
    'sys.stderr.write("Why: " + ("x" * 5000) + "\\n")',
    'sys.stderr.write(\'Proof: recall(handle="op-receipt-0123456789abcdef")\\n\')',
    'sys.stderr.write(\'Rule: recall(handle="op-rule-OP-004")\\n\')',
    'sys.exit(2)',
    '',
  ].join('\n'));
  const huge = run('cursor', JSON.stringify({ tool_name: 'Bash', tool_input: { command: 'echo hello' } }), { python: 'python3', gate: script });
  const body = JSON.parse(huge.stdout);
  assert.equal(body.permission, 'deny');
  assert.ok(body.agent_message.length <= 1200, body.agent_message.length);
  assert.match(body.agent_message, /op-receipt-0123456789abcdef/);
  assert.match(body.user_message, /op-rule-OP-004/);
});

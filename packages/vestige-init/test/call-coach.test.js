'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const reflex = require('../hooks/reflex.cjs');

test('Call Coach (#520) rewrites aliases/defaults, preserves explicit agent values, blocks >500-char weave notes, and learns from failures', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-coach-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const config = path.join(dir, 'config.json');
  fs.writeFileSync(config, JSON.stringify({ enforce: true }));

  // CC-001: memory_id -> memoryId
  const aliasOut = await reflex.run({
    client: 'claude',
    configFile: config,
    input: {
      hook_event_name: 'PreToolUse',
      session_id: 's1',
      tool_name: 'mcp__vestige__curate',
      tool_input: { action: 'demote', memory_id: 'mem-0000000000000042' },
    },
  });
  assert.deepEqual(aliasOut.hookSpecificOutput.updatedInput, {
    action: 'demote',
    memoryId: 'mem-0000000000000042',
  });
  assert.match(aliasOut.hookSpecificOutput.additionalContext, /CC-001/);

  // CC-001 also handles curate node_id -> memoryId
  const nodeIdCoach = reflex.coachPreToolUse('mcp__vestige__curate', {
    action: 'suppress',
    node_id: 'mem-0000000000000043',
  });
  assert.equal(nodeIdCoach.updatedInput.memoryId, 'mem-0000000000000043');
  assert.equal(nodeIdCoach.updatedInput.node_id, undefined);
  assert.deepEqual(nodeIdCoach.ruleIds, ['CC-001']);

  // Explicit agent value is NEVER overwritten
  const explicitRecall = reflex.coachPreToolUse('mcp__vestige__recall', {
    query: 'deployment rules',
    limit: 25,
    detail_level: 'full',
  });
  assert.equal(explicitRecall, null);

  // CC-003: omitted recall limit/detail_level gets bounded summary defaults
  const boundedRecall = reflex.coachPreToolUse('mcp__vestige__recall', { query: 'deployment rules' });
  assert.deepEqual(boundedRecall.updatedInput, {
    query: 'deployment rules',
    limit: 10,
    detail_level: 'summary',
  });
  assert.deepEqual(boundedRecall.ruleIds, ['CC-003']);

  // CC-004 & CC-005: smart_ingest compact + causal_walk scan_limit
  const ingestCoach = reflex.coachPreToolUse('mcp__vestige__smart_ingest', { content: 'fact' });
  assert.equal(ingestCoach.updatedInput.output_profile, 'compact');
  const walkCoach = reflex.coachPreToolUse('mcp__vestige__causal_walk', { action: 'walk' });
  assert.equal(walkCoach.updatedInput.scan_limit, 500);

  // CC-006..CC-012 full rule table
  const r6 = reflex.coachPreToolUse('mcp__vestige__recall', { id: 'mem-0000000000000010' });
  assert.equal(r6.updatedInput.handle, 'mem-0000000000000010');
  assert.equal(r6.updatedInput.id, undefined);
  assert.deepEqual(r6.ruleIds, ['CC-006']);

  const r7 = reflex.coachPreToolUse('mcp__vestige__remember', { text: 'important lesson' });
  assert.equal(r7.updatedInput.content, 'important lesson');
  assert.deepEqual(r7.ruleIds, ['CC-007']);

  const r8 = reflex.coachPreToolUse('mcp__vestige__causal_walk', { memoryId: 'mem-0000000000000011', scan_limit: 200 });
  assert.equal(r8.updatedInput.from_handle, 'mem-0000000000000011');
  assert.deepEqual(r8.ruleIds, ['CC-008']);

  const r9 = reflex.coachPreToolUse('mcp__vestige__precedent', { action: 'get', digest: 'eff-0123456789abcdef' });
  assert.equal(r9.updatedInput.receipt_id, 'eff-0123456789abcdef');
  assert.deepEqual(r9.ruleIds, ['CC-009']);

  const r10 = reflex.coachPreToolUse('mcp__vestige__curate', { action: 'delete', memoryId: 'mem-0000000000000010' });
  assert.equal(r10.deny, true);
  assert.deepEqual(r10.ruleIds, ['CC-010']);
  assert.match(r10.reason, /unavailable_in_4_0/);

  const r11 = reflex.coachPreToolUse('mcp__vestige__remember', { content: '   ' });
  assert.equal(r11.deny, true);
  assert.deepEqual(r11.ruleIds, ['CC-011']);

  const r12 = reflex.coachPreToolUse('mcp__vestige__session', { view: 'tools', tool_name: 'recall' });
  assert.equal(r12.updatedInput.action, 'status');
  assert.equal(r12.updatedInput.tool, 'recall');
  assert.deepEqual(r12.ruleIds, ['CC-012']);

  // Gemini & Cursor hostOutput updatedInput mapping
  const gem = reflex.hostOutput('gemini', 'BeforeTool', 'coached', {
    updatedInput: { handle: 'mem-0000000000000010' },
  });
  assert.deepEqual(gem.hookSpecificOutput.updatedInput, { handle: 'mem-0000000000000010' });
  const cur = reflex.hostOutput('cursor', 'preToolUse', '', {
    updatedInput: { handle: 'mem-0000000000000010' },
  });
  assert.deepEqual(cur.updated_input, { handle: 'mem-0000000000000010' });

  // CC-002: weave note > 500 chars refused before call with exact count and limit
  const longNote = 'x'.repeat(512);
  const deniedWeave = await reflex.run({
    client: 'claude',
    configFile: config,
    input: {
      hook_event_name: 'PreToolUse',
      session_id: 's1',
      tool_name: 'mcp__vestige__curate',
      tool_input: { action: 'weave', source: 'mem-0000000000000001', target: 'mem-0000000000000002', note: longNote },
    },
  });
  assert.equal(deniedWeave.hookSpecificOutput.permissionDecision, 'deny');
  assert.match(deniedWeave.hookSpecificOutput.permissionDecisionReason, /CC-002.*512 characters.*limit: 500/);

  // Learned rule from PostToolUseFailure with provenance + offline --replay
  const learned = reflex.extractLearnedAliasFailure(
    'mcp__vestige__causal_walk',
    "invalid_params: unknown field `repo_path`, expected `repoPath`",
    'mem-0000000000000099',
  );
  assert.deepEqual(learned, {
    leaf: 'causal_walk',
    code: 'invalid_params',
    badKey: 'repo_path',
    goodKey: 'repoPath',
    derived_from: 'mem-0000000000000099',
  });
  fs.writeFileSync(path.join(dir, 'coach-rules.json'), JSON.stringify({ rules: [{ id: 'CC-L001', ...learned }] }));

  const logFile = path.join(dir, 'calls.jsonl');
  fs.writeFileSync(logFile, [
    JSON.stringify({ receipt_id: 'rcpt-1', tool_name: 'mcp__vestige__curate', tool_input: { action: 'demote', memory_id: 'mem-0000000000000001' } }),
    JSON.stringify({ receipt_id: 'rcpt-2', tool_name: 'mcp__vestige__recall', tool_input: { query: 'test' } }),
    JSON.stringify({ receipt_id: 'rcpt-3', tool_name: 'mcp__vestige__causal_walk', tool_input: { action: 'prove', scan_limit: 100, repo_path: '/repo' } }),
    JSON.stringify({ receipt_id: 'rcpt-4', tool_name: 'mcp__vestige__recall', tool_input: { tag: 'pinned' } }),
  ].join('\n') + '\n');

  const cli = spawnSync(process.execPath, [path.join(__dirname, '../hooks/reflex.cjs'), '--config', config, '--replay', logFile], {
    encoding: 'utf8',
  });
  assert.equal(cli.status, 0, cli.stderr);
  const report = JSON.parse(cli.stdout);
  assert.equal(report.total_calls, 4);
  assert(report.rules.find(r => r.rule === 'CC-001' && r.prevented_failures === 1));
  assert(report.rules.find(r => r.rule === 'CC-003' && r.saved_chars > 0));
  assert(report.rules.find(r => r.rule === 'CC-006' && r.prevented_failures === 1));
  assert(report.rules.find(r => r.rule === 'CC-L001' && r.prevented_failures === 1));
});

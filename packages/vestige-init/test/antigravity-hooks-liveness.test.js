'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const PKG_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(PKG_ROOT, '..', '..');
const HOOKS_DIR = path.join(PKG_ROOT, 'hooks');
const LIB_DIR = path.join(PKG_ROOT, 'lib');

const HOME = process.env.USERPROFILE || process.env.HOME || os.homedir();
const HAS_AGY_ENV = fs.existsSync(path.join(HOME, '.gemini', 'config'));
const AGY_PLUGIN_DIR = path.join(HOME, '.gemini', 'config', 'plugins', 'vestige');
const AGY_HOOKS_JSON = path.join(AGY_PLUGIN_DIR, 'hooks.json');
const AGY_BRIDGE_JS = path.join(AGY_PLUGIN_DIR, 'bridge', 'agy-bridge.cjs');
const SHIPPED_BRIDGE_JS = path.join(HOOKS_DIR, 'agy-bridge.cjs');
const GATE_DIR = path.join(HOME, '.vestige', 'gate');

function hasUtf8Bom(buf) {
  return buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf;
}

test('1. Shipped hooks (agy-bridge.cjs, reflex.cjs, memory.cjs, operator-gate.py, save-gate.py, session_automaton.json) are complete', () => {
  const reflexSrc = fs.readFileSync(path.join(HOOKS_DIR, 'reflex.cjs'), 'utf8');
  const memorySrc = fs.readFileSync(path.join(HOOKS_DIR, 'memory.cjs'), 'utf8');
  const bridgeSrc = fs.readFileSync(SHIPPED_BRIDGE_JS, 'utf8');
  assert.ok(reflexSrc.includes('coachPreToolUse'), 'reflex.cjs must include Call Coach (#520)');
  assert.ok(memorySrc.includes('tryDirectAttach'), 'memory.cjs must include tryDirectAttach');
  assert.ok(bridgeSrc.includes('intentionsStartContext'), 'agy-bridge.cjs must query active intentions & portfolio');
  assert.ok(bridgeSrc.includes('capsuleRun'), 'agy-bridge.cjs must invoke vestige capsule');
  assert.ok(bridgeSrc.includes('saveGateRun'), 'agy-bridge.cjs must invoke save-gate.py');
  assert.ok(bridgeSrc.includes('postCommitReceiptRun'), 'agy-bridge.cjs must invoke vestige hook post-commit');
  assert.ok(fs.existsSync(path.join(HOOKS_DIR, 'operator-gate.py')), 'operator-gate.py must exist');
  assert.ok(fs.existsSync(path.join(HOOKS_DIR, 'lite_core.py')), 'lite_core.py must exist');
  assert.ok(fs.existsSync(path.join(HOOKS_DIR, 'session_automaton.json')), 'session_automaton.json must exist');
  assert.ok(fs.existsSync(path.join(REPO_ROOT, 'hooks', 'save-gate.py')), 'repo hooks/save-gate.py must exist');
  assert.ok(fs.existsSync(path.join(LIB_DIR, 'memory-triggers.cjs')), 'memory-triggers.cjs must exist');
});

test('2. Antigravity installation is BOM-free, has ALL 6 events, and matches shipped sources byte-for-byte', { skip: !HAS_AGY_ENV }, () => {
  assert.ok(fs.existsSync(AGY_HOOKS_JSON), `Missing ${AGY_HOOKS_JSON}`);
  assert.ok(fs.existsSync(AGY_BRIDGE_JS), `Missing ${AGY_BRIDGE_JS}`);
  assert.ok(fs.existsSync(path.join(GATE_DIR, 'save-gate.py')), `Missing ${path.join(GATE_DIR, 'save-gate.py')}`);

  const raw = fs.readFileSync(AGY_HOOKS_JSON);
  assert.equal(hasUtf8Bom(raw), false, 'hooks.json must NOT have a UTF-8 BOM (hooks.go rejects \\ufeff)');
  const parsed = JSON.parse(raw.toString('utf8'));
  assert.ok(parsed.vestige && typeof parsed.vestige === 'object', 'hooks.json must declare top-level vestige key');

  for (const ev of ['SessionStart', 'PreToolUse', 'PostToolUse', 'PreInvocation', 'PostInvocation', 'Stop']) {
    assert.ok(Array.isArray(parsed.vestige[ev]) && parsed.vestige[ev].length > 0, `hooks.json missing event: ${ev}`);
  }

  for (const [ev, entries] of Object.entries(parsed.vestige)) {
    for (const entry of entries) {
      const handlers = Array.isArray(entry.hooks) ? entry.hooks : [entry];
      for (const h of handlers) {
        assert.equal(typeof h.command, 'string', `${ev} handler must have command`);
        if (process.platform === 'win32') {
          assert.equal(/^"[^"]*\/[^"]*"/.test(h.command), false, `${ev} command uses quoted forward-slash exe rejected by cmd /c: ${h.command}`);
        }
      }
    }
  }

  assert.equal(
    fs.readFileSync(AGY_BRIDGE_JS, 'utf8'),
    fs.readFileSync(SHIPPED_BRIDGE_JS, 'utf8'),
    'Installed agy-bridge.cjs must match packages/vestige-init/hooks/agy-bridge.cjs byte-for-byte'
  );
  assert.equal(
    fs.readFileSync(path.join(GATE_DIR, 'save-gate.py'), 'utf8'),
    fs.readFileSync(path.join(REPO_ROOT, 'hooks', 'save-gate.py'), 'utf8'),
    'Installed .vestige/gate/save-gate.py must match repo hooks/save-gate.py byte-for-byte'
  );
});

test('3. Installed Wave 13 Operator gate + session_automaton.json pass safe Read without OP-024 integrity lockout', { skip: !HAS_AGY_ENV }, () => {
  assert.ok(fs.existsSync(path.join(GATE_DIR, 'operator-gate.py')), 'operator-gate.py must exist');
  assert.ok(fs.existsSync(path.join(GATE_DIR, 'lite_core.py')), 'lite_core.py must sit beside operator-gate.py');
  assert.ok(fs.existsSync(path.join(GATE_DIR, 'session_automaton.json')), 'session_automaton.json must sit beside operator-gate.py');

  const cfgPath = path.join(AGY_PLUGIN_DIR, 'bridge', 'bridge.json');
  const pythonExe = JSON.parse(fs.readFileSync(cfgPath, 'utf8')).python;
  const payload = JSON.stringify({
    hook_event_name: 'PreToolUse',
    session_id: 'liveness-gate-check',
    tool_name: 'Read',
    tool_input: { file_path: 'AGENTS.md' },
    cwd: PKG_ROOT,
  });
  const r = spawnSync(pythonExe, ['-I', path.join(GATE_DIR, 'operator-gate.py'), 'hook', '--source', 'antigravity'], {
    input: payload,
    encoding: 'utf8',
    timeout: 10000,
    windowsHide: true,
  });
  assert.equal(r.status, 0, `operator-gate.py failed safe Read (status=${r.status}): ${r.stderr}`);
});

test('4. Bridge translation, Read slice preservation, Call Coach CC-010 deny & CC-006 overwrite work', { skip: !HAS_AGY_ENV }, () => {
  const bridge = require(SHIPPED_BRIDGE_JS);

  // Supports both object and JSON-string Arguments
  const mappedMcp = bridge.toClaudeTool({
    name: 'call_mcp_tool',
    args: { ServerName: 'vestige_vestige', ToolName: '"curate"', Arguments: JSON.stringify({ action: 'delete', id: 'mem-1' }) },
  });
  assert.equal(mappedMcp.tool_name, 'mcp__vestige__curate');
  assert.equal(mappedMcp.tool_input.action, 'delete');

  // Preserves StartLine/EndLine as offset/limit so reading slices doesn't trigger false Redo
  const mappedRead = bridge.toClaudeTool({
    name: 'read_file',
    args: { Path: 'a.rs', StartLine: 101, EndLine: 200 },
  });
  assert.equal(mappedRead.tool_name, 'Read');
  assert.equal(mappedRead.tool_input.offset, 101);
  assert.equal(mappedRead.tool_input.limit, 100);

  // CC-010 denies curate(delete)
  const rDeny = spawnSync(process.execPath, [AGY_BRIDGE_JS, '--event', 'PreToolUse'], {
    cwd: AGY_PLUGIN_DIR,
    input: JSON.stringify({
      conversationId: 'liveness-cc-check',
      workspacePaths: [PKG_ROOT],
      stepIdx: 1,
      toolCall: {
        id: 'tc-cc-1',
        name: 'call_mcp_tool',
        args: { ServerName: 'vestige_vestige', ToolName: 'curate', Arguments: { action: 'delete', id: 'mem-1' } },
      },
    }),
    encoding: 'utf8',
    timeout: 15000,
    windowsHide: true,
  });
  assert.equal(rDeny.status, 0, rDeny.stderr);
  const outDeny = JSON.parse(rDeny.stdout.trim());
  assert.equal(outDeny.decision, 'deny');
  assert.ok(String(outDeny.reason || '').includes('CC-010'), `Expected CC-010 deny, got: ${rDeny.stdout}`);

  // CC-006 rewrites recall(tag: 'pinned') -> handle: 'pinned' via PreToolUse overwrite
  const rRewrite = spawnSync(process.execPath, [AGY_BRIDGE_JS, '--event', 'PreToolUse'], {
    cwd: AGY_PLUGIN_DIR,
    input: JSON.stringify({
      conversationId: 'liveness-cc-rewrite',
      workspacePaths: [PKG_ROOT],
      stepIdx: 2,
      toolCall: {
        id: 'tc-cc-2',
        name: 'call_mcp_tool',
        args: { ServerName: 'vestige_vestige', ToolName: 'recall', Arguments: { tag: 'pinned' } },
      },
    }),
    encoding: 'utf8',
    timeout: 15000,
    windowsHide: true,
  });
  assert.equal(rRewrite.status, 0, rRewrite.stderr);
  const outRewrite = JSON.parse(rRewrite.stdout.trim());
  assert.equal(outRewrite.decision, 'allow');
  assert.equal(outRewrite.overwrite?.Arguments?.handle, 'pinned', `Expected CC-006 overwrite handle='pinned', got: ${rRewrite.stdout}`);
});

test('5. Transcript MCP response extractor & Windows Save-Gate path normalizer handle real Antigravity formats', () => {
  const bridge = require(SHIPPED_BRIDGE_JS);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-tx-test-'));
  try {
    const tx = path.join(tmp, 'transcript.jsonl');
    const lines = [
      JSON.stringify({
        step_index: 1,
        type: 'USER_INPUT',
        content: '<USER_REQUEST>\nFrom now on always check clippy.\n</USER_REQUEST>',
      }),
      JSON.stringify({
        step_index: 2,
        type: 'PLANNER_RESPONSE',
        content: '',
        tool_calls: [{ name: 'call_mcp_tool', args: { ServerName: '"vestige_vestige"', ToolName: '"remember"', Arguments: '{"action":"save"}' } }],
      }),
      JSON.stringify({
        step_index: 3,
        type: 'GENERIC',
        content: 'Created At: 2026-11-18T01:00:00Z\nCompleted At: 2026-11-18T01:00:01Z\n{"decision":"create","nodeId":"mem-000000000000abcd","receiptId":"eff-0000000000001234","success":true}',
      }),
    ];
    fs.writeFileSync(tx, lines.join('\n') + '\n', 'utf8');

    const mcpResp = bridge.extractMcpResponseFromTranscript(tx);
    assert.equal(mcpResp?.nodeId, 'mem-000000000000abcd');
    assert.equal(mcpResp?.receiptId, 'eff-0000000000001234');

    const turn = bridge.readLatestTurnFromTranscript(tx);
    assert.equal(turn?.alreadyRemembered, true, 'Quoted ToolName in transcript.jsonl must still mark alreadyRemembered=true');

    const repoRoot = process.platform === 'win32' ? 'C:\\Users\\Sam\\repo' : '/home/sam/repo';
    const editAbs = process.platform === 'win32' ? 'C:\\Users\\Sam\\repo\\crates\\vestige-core\\src\\lib.rs' : '/home/sam/repo/crates/vestige-core/src/lib.rs';
    assert.equal(bridge.normalizePathForSaveGate(editAbs, repoRoot), 'crates/vestige-core/src/lib.rs');
    assert.equal(bridge.normalizePathForSaveGate('crates/vestige-core/src/lib.rs#my_fn', repoRoot), 'crates/vestige-core/src/lib.rs');
    // Multi-worktree normalization: editing a file inside REPO_ROOT when cwd is a different worktree still normalizes to repo-relative!
    const siblingWorktreeCwd = path.join(tmp, 'other-worktree');
    const realRepoFile = path.join(REPO_ROOT, 'packages', 'vestige-init', 'package.json');
    assert.equal(bridge.normalizePathForSaveGate(realRepoFile, siblingWorktreeCwd), 'packages/vestige-init/package.json');
    assert.equal(bridge.isScratchOrArtifactEdit('C:\\Users\\Sam\\.gemini\\antigravity\\brain\\abc\\walkthrough.md', ''), true);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('6. Live SessionStart AND fresh PreInvocation inject Active Intentions, Capsule, and Rulebook', { skip: !HAS_AGY_ENV, timeout: 40000 }, () => {
  const run = (ev, sid, extra = {}) => {
    const r = spawnSync(process.execPath, [AGY_BRIDGE_JS, '--event', ev], {
      cwd: AGY_PLUGIN_DIR,
      env: { ...process.env, VESTIGE_HOOK_DEADLINE_MS: '10000' },
      input: JSON.stringify({
        conversationId: sid,
        workspacePaths: [REPO_ROOT],
        cwd: REPO_ROOT,
        lastUserInput: 'How does WordNet hypernym taxa work?',
        ...extra,
      }),
      encoding: 'utf8',
      timeout: 20000,
      windowsHide: true,
    });
    assert.equal(r.status, 0, `${ev} failed: ${r.stderr}`);
    return JSON.parse((r.stdout || '{}').trim() || '{}');
  };

  const s = run('SessionStart', `agy-boot-${Date.now()}`);
  const sMsg = s.injectSteps?.[0]?.systemMessage?.systemMessage || '';
  assert.ok(sMsg.includes('Active Intentions & Plans:'), `SessionStart missing Active Intentions & Plans:\n${sMsg}`);
  assert.ok(sMsg.includes('capsule version='), `SessionStart missing capsule header:\n${sMsg}`);
  assert.ok(sMsg.includes('Rulebook: R01-R17 active'), `SessionStart missing Rulebook summary:\n${sMsg}`);

  // Even on a brand-new conversation that enters directly at PreInvocation, Prompt #1 MUST carry Active Intentions!
  const pFresh = run('PreInvocation', `agy-fresh-prompt1-${Date.now()}`, { invocationNum: 1 });
  const pMsg = pFresh.injectSteps?.[0]?.systemMessage?.systemMessage || '';
  assert.ok(pMsg.includes('Active Intentions & Plans:'), `Prompt #1 PreInvocation must surface Active Intentions & Plans:\n${pMsg}`);
});

test('7. Full Save-Gate + Directive + Remember + Single-Block Guard E2E through agy-bridge.cjs', { skip: !HAS_AGY_ENV, timeout: 45000 }, () => {
  const sid = `agy-e2e-gate-${Date.now()}`;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'agy-e2e-gate-'));
  const tx = path.join(tmp, 'transcript.jsonl');
  const artDir = path.join(tmp, 'artifacts');
  fs.mkdirSync(artDir, { recursive: true });

  const run = (ev, extra = {}) => {
    const r = spawnSync(process.execPath, [AGY_BRIDGE_JS, '--event', ev], {
      cwd: AGY_PLUGIN_DIR,
      input: JSON.stringify({
        conversationId: sid,
        workspacePaths: [REPO_ROOT],
        cwd: REPO_ROOT,
        transcriptPath: tx,
        artifactDirectoryPath: artDir,
        lastUserInput: 'From now on never skip the benchmark gate.',
        ...extra,
      }),
      encoding: 'utf8',
      timeout: 20000,
      windowsHide: true,
    });
    assert.equal(r.status, 0, `${ev} failed: ${r.stderr}`);
    return JSON.parse((r.stdout || '{}').trim() || '{}');
  };

  try {
    // 1) Prompt with directive
    const p1 = run('PreInvocation', { invocationNum: 1 });
    const p1Msg = p1.injectSteps?.[0]?.systemMessage?.systemMessage || '';
    assert.ok(p1Msg.includes('owner directive'), `Expected owner directive warning in PreInvocation:\n${p1Msg}`);

    // 2) Writing an artifact file inside artifactDirectoryPath must NOT count as an unsaved source edit
    run('PostToolUse', {
      stepIdx: 1,
      toolCall: { id: 'art-1', name: 'write_file', args: { Path: path.join(artDir, 'notes.md'), Content: '# Scratch' } },
    });

    // 3) Editing a repo source file using a full OS path registers unsaved edit in Save-Gate
    const absTarget = path.join(REPO_ROOT, 'crates', 'vestige-core', 'src', 'lib.rs');
    run('PostToolUse', {
      stepIdx: 2,
      toolCall: { id: 'ed-1', name: 'edit_file', args: { Path: absTarget, Target: 'a', Replacement: 'b' } },
    });

    // 4) git commit BEFORE remember MUST be denied by Save-Gate!
    const commitDenied = run('PreToolUse', {
      stepIdx: 3,
      toolCall: { id: 'cm-1', name: 'run_command', args: { CommandLine: 'git commit -m "unanchored edit"' } },
    });
    assert.equal(commitDenied.decision, 'deny', `Expected Save-Gate to deny unanchored git commit: ${JSON.stringify(commitDenied)}`);
    assert.ok(String(commitDenied.reason || '').includes('crates/vestige-core/src/lib.rs'), `Deny reason must name normalized path: ${commitDenied.reason}`);

    // 5) Stop BEFORE remember MUST block once (decision: 'continue'), then single-block guard prevents loop
    const stop1 = run('Stop', { terminationReason: 'NO_TOOL_CALL', continuationActive: false, lastAssistantMessage: 'Done.' });
    assert.equal(stop1.decision, 'continue', `Expected Stop #1 to block unpaid directive/edit: ${JSON.stringify(stop1)}`);
    const stop2 = run('Stop', { terminationReason: 'NO_TOOL_CALL', continuationActive: false, lastAssistantMessage: 'Done.' });
    assert.equal(stop2.decision, undefined, `Single-block-per-prompt guard must prevent infinite Stop loop: ${JSON.stringify(stop2)}`);

    // 6) Agent calls remember(files=['crates/vestige-core/src/lib.rs#sym']) and transcript logs real mem-... nodeId
    fs.writeFileSync(
      tx,
      JSON.stringify({
        step_index: 4,
        type: 'GENERIC',
        content: 'Created At: now\n{"decision":"create","nodeId":"mem-000000000000beef","success":true}',
      }) + '\n',
      'utf8'
    );
    run('PostToolUse', {
      stepIdx: 4,
      toolCall: {
        id: 'rem-1',
        name: 'call_mcp_tool',
        args: {
          ServerName: 'vestige_vestige',
          ToolName: 'remember',
          Arguments: { action: 'save', content: 'Anchored lib.rs and directive', files: ['crates/vestige-core/src/lib.rs#sym'] },
        },
      },
    });

    // 7) git commit AFTER remember is ALLOWED by Save-Gate!
    const commitAllowed = run('PreToolUse', {
      stepIdx: 5,
      toolCall: { id: 'cm-2', name: 'run_command', args: { CommandLine: 'git status' } },
    });
    assert.equal(commitAllowed.decision, 'allow');

    // Also verify git commit passes Save-Gate (checking saveGate state cleared unsaved_files)
    const commitCheck = run('PreToolUse', {
      stepIdx: 6,
      toolCall: { id: 'cm-3', name: 'run_command', args: { CommandLine: 'git commit -m "anchored edit"' } },
    });
    assert.notEqual(
      String(commitCheck.reason || '').includes('have no remember'),
      true,
      `Save-Gate must clear unsaved_files after remember with repo-relative path: ${JSON.stringify(commitCheck)}`
    );
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('8. Stop hook auto-saves high-signal engineering findings/scores and ignores casual chat', () => {
  const bridge = require(SHIPPED_BRIDGE_JS);
  assert.equal(bridge.extractAutoSaveCandidate({
    userText: 'sounds good thanks',
    assistantText: 'You are welcome! Let me know if you want to check anything else.',
    sawToolActivity: false,
  }), null, 'Casual conversation must NOT be auto-saved');

  assert.equal(bridge.extractAutoSaveCandidate({
    userText: 'save',
    assistantText: 'Root Cause found in crates/vestige-core/src/retrieval/rank.rs where session sync landed at slot 5; CoRA-Bench 64K Evidence Recall@10=100.0% and All-Facts@10=100.0%.',
    alreadyRemembered: true,
    sawToolActivity: true,
  }), null, 'Must skip auto-save when model already called remember this turn');

  const cand = bridge.extractAutoSaveCandidate({
    userText: 'check scores',
    assistantText: 'Verified benchmark results in crates/vestige-core/src/retrieval/rank.rs and benchmarks/results/v6.2.2-next/cora_64k.json: CoRA-Bench 64K Evidence Recall@10=100.0% and All-Facts@10=100.0%.',
    alreadyRemembered: false,
    sawToolActivity: true,
  });
  assert.ok(cand && cand.content.includes('100.0%'), 'High-signal verified benchmark/file finding must be extracted');
  assert.ok(cand.files.includes('crates/vestige-core/src/retrieval/rank.rs'));
  assert.ok(cand.entities.includes('file:crates/vestige-core/src/retrieval/rank.rs'));
});

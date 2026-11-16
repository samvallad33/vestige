#!/usr/bin/env node
'use strict';
// Host output adapter only. Every policy, permit, and receipt decision stays in the gate.
const fs = require('fs');
const { spawnSync } = require('child_process');
const args = process.argv.slice(2);
const value = flag => args[args.indexOf(flag) + 1];
const client = value('--client');
const MESSAGE_CAP = 1200;

function proofText(stderr) {
  const lines = String(stderr || '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const wanted = lines.filter(line => /^(OPERATOR: STOPPED|Why:|Proof:|Rule:|Forbids:|Your law:|Observed:)/.test(line));
  const chosen = (wanted.length ? wanted : lines).map(line => line.slice(0, 400));
  let text = '';
  for (const line of chosen) {
    const next = text ? `${text}\n${line}` : line;
    if (next.length > MESSAGE_CAP) break;
    text = next;
  }
  const proof = chosen.find(line => line.startsWith('Proof:') || line.startsWith('Rule:'));
  if (proof && !text.includes(proof)) {
    const room = MESSAGE_CAP - proof.length - 1;
    text = `${text.slice(0, Math.max(0, room))}\n${proof}`.trim();
  }
  return text;
}

function deny(stderr) {
  const text = proofText(stderr);
  const message = text || 'Operator stopped this tool call. Inspect its local receipt.';
  if (client === 'cursor') {
    process.stdout.write(`${JSON.stringify({ permission: 'deny', agent_message: message, user_message: message })}\n`);
  } else if (client === 'gemini') {
    process.stdout.write(`${JSON.stringify({ decision: 'deny', reason: message })}\n`);
  } else {
    process.stderr.write(message.endsWith('\n') ? message : `${message}\n`);
    if (!/Operator stopped/i.test(message)) {
      process.stderr.write('Operator stopped this tool call. Inspect its local receipt.\n');
    }
  }
  process.exitCode = 2;
}

try {
  const config = JSON.parse(fs.readFileSync(value('--config'), 'utf8'));
  const input = fs.readFileSync(0, 'utf8');
  let payload;
  try {
    payload = JSON.parse(input);
  } catch {
    deny('');
    return;
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    deny('');
    return;
  }
  const result = spawnSync(config.python, ['-I', config.gate, 'hook', '--source', client], {
    input, encoding: 'utf8', timeout: 10000, maxBuffer: 256 * 1024,
  });
  // A crashed gate, a missing interpreter, or a non-zero status is a deny.
  // The adapter does not invent a recall handle the gate did not print.
  if (!result || result.error || result.status !== 0) deny(result && result.stderr);
  else if (client === 'cursor') process.stdout.write('{"permission":"allow"}\n');
  else process.stdout.write('{}\n');
} catch {
  deny('');
}

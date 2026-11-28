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
  const wanted = lines.filter(line => /^(OPERATOR: STOPPED|Why:|Proof:|Rule:|Forbids:|Your law:|Observed:|Action ID:|Receipt:|Approve:)/.test(line));
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
  const receiptMatch = /\b(?:op-receipt-|Action ID:\s*|Receipt:\s*)([0-9a-f]{16,64})\b/i.exec(text);
  if (receiptMatch && !/precedent\(/.test(text)) {
    const hint = `Precedent: call precedent(action="get", receipt_id="op-receipt-${receiptMatch[1].slice(0, 16)}")`;
    if (text.length + hint.length + 1 <= MESSAGE_CAP) text = `${text}\n${hint}`;
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

function gateObject(stdout) {
  const lines = String(stdout || '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (!lines.length) return {};
  if (lines.length !== 1) return undefined;
  try {
    const parsed = JSON.parse(lines[0]);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
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
  const path = require('path');
  const useNativeRust = Boolean(config.hookBin || config.vestigeBin) && (!config.gate || config.gate === 'native');
  let nativeBin = config.vestigeBin;
  let nativeArgs = ['gate', 'hook', '--source', client];
  if (useNativeRust) {
    if (config.hookBin && fs.existsSync(config.hookBin)) {
      nativeBin = config.hookBin;
      nativeArgs = ['gate', '--source', client];
    } else if (config.vestigeBin) {
      const fastHook = path.join(path.dirname(config.vestigeBin), process.platform === 'win32' ? 'vestige-hook.exe' : 'vestige-hook');
      if (fs.existsSync(fastHook)) {
        nativeBin = fastHook;
        nativeArgs = ['gate', '--source', client];
      }
    }
  }
  const result = useNativeRust
    ? spawnSync(nativeBin, nativeArgs, {
        input, encoding: 'utf8', timeout: 10000, maxBuffer: 256 * 1024, windowsHide: true,
      })
    : spawnSync(config.python, ['-I', config.gate, 'hook', '--source', client], {
        input, encoding: 'utf8', timeout: 10000, maxBuffer: 256 * 1024, windowsHide: true,
      });
  // A crashed gate, a missing interpreter, or a non-zero status is a deny.
  // The adapter does not invent a recall handle the gate did not print.
  if (!result || result.error || result.status !== 0) {
    deny(result && result.stderr);
    return;
  }
  // On an allow the gate may print one JSON object that rewrites the call
  // (folded bytes, a stripped secret field). Pass it through unchanged: dropping
  // it would run the original input. Output that is not one object is a deny.
  const rewrite = gateObject(result.stdout);
  if (rewrite === undefined) {
    deny('Operator gate printed output the adapter could not read; the call was not run.');
    return;
  }
  const body = client === 'cursor' ? { ...rewrite, permission: 'allow' } : rewrite;
  process.stdout.write(`${JSON.stringify(body)}\n`);
} catch {
  deny('');
}

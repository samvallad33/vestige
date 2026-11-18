#!/usr/bin/env node
'use strict';
// Memory relay only: policy enforcement remains in the separate pre-tool gate.
const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const args = process.argv.slice(2);
const value = flag => args[args.indexOf(flag) + 1];
let child; let finished = false; let output = ''; let timer;
function finish(error) {
  if (finished) return; finished = true; clearTimeout(timer);
  if (error) {
    try { fs.appendFileSync(path.join(__dirname,'..','errors.log'), `${new Date().toISOString()} stop-memory: ${error}\n`); } catch {}
    if (child?.pid) {
      try {
        if (process.platform === 'win32') spawnSync('taskkill', ['/pid',String(child.pid),'/T','/F'], {timeout:30,stdio:'ignore',windowsHide:true});
        else process.kill(-child.pid, 'SIGKILL');
      } catch {}
      child.stdin.destroy(); child.stdout.destroy(); child.unref();
    }
  }
  // A timeout leaves the gate's durable receipts available for the next hook.
  process.stdout.write(error ? '{}\n' : output || '{}\n');
}
try {
  const input = fs.readFileSync(0);
  const timeoutMs = Number(process.env.VESTIGE_STOP_MEMORY_TIMEOUT_MS) || 155;
  const childBudget = process.env.OPERATOR_STOP_MEMORY_BUDGET || (process.platform === 'win32' ? '0.11' : '0.09');
  child = spawn(value('--python'), ['-I', value('--script'), '--mcp', value('--mcp')], {
    detached: process.platform !== 'win32', windowsHide:true,
    env: {...process.env, OPERATOR_STOP_MEMORY_BUDGET: childBudget}, stdio:['pipe','pipe','ignore'],
  });
  timer = setTimeout(() => finish('deadline exceeded; receipts retained for retry'), timeoutMs);
  child.on('error', () => finish('reflex unavailable; receipts retained for retry'));
  child.stdout.on('data', chunk => { output += chunk; if (output.length > 65536) finish('output limit exceeded'); });
  child.on('close', code => finish(code === 0 ? null : 'reflex failed; receipts retained for retry'));
  child.stdin.on('error', () => {}); child.stdin.end(input);
} catch { finish('relay unavailable; receipts retained for retry'); }

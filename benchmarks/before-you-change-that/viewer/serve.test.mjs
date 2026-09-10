import assert from 'node:assert/strict';
import {mkdtemp, mkdir, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve} from 'node:path';
import test from 'node:test';
import {startServer} from './serve.mjs';

test('server exposes viewer and public evidence with strict local headers', async () => {
  const server = await startServer();
  try {
    const origin = `http://127.0.0.1:${String(server.address().port)}`;
    const viewer = await fetch(`${origin}/viewer/`);
    assert.equal(viewer.status, 200);
    assert.match(viewer.headers.get('content-type'), /^text\/html/);
    assert.match(viewer.headers.get('content-security-policy'), /default-src 'self'/);
    assert.match(await viewer.text(), /Recorded-event replay/);
    const evidence = await fetch(`${origin}/evidence/control.jsonl`);
    assert.equal(evidence.status, 200);
    assert.match(evidence.headers.get('content-type'), /^application\/x-ndjson/);
    const traversal = await fetch(`${origin}/%2e%2e/%2e%2e/AGENTS.md`);
    assert.ok([400, 404].includes(traversal.status));
  } finally {
    await new Promise((resolveClose) => server.close(resolveClose));
  }
});

test('server refuses an allowlisted symlink that escapes its real root', async () => {
  const root = await mkdtemp(resolve(tmpdir(), 'public-replay-root-'));
  const outside = await mkdtemp(resolve(tmpdir(), 'public-replay-outside-'));
  await mkdir(resolve(root, 'viewer'));
  await writeFile(resolve(outside, 'secret.html'), 'outside');
  await symlink(resolve(outside, 'secret.html'), resolve(root, 'viewer/index.html'));
  const server = await startServer({root, allowedFiles: new Set(['viewer/index.html'])});
  try {
    const response = await fetch(`http://127.0.0.1:${String(server.address().port)}/viewer/`);
    assert.equal(response.status, 403);
    assert.doesNotMatch(await response.text(), /outside/);
  } finally {
    await new Promise((resolveClose) => server.close(resolveClose));
    await rm(root, {recursive: true, force: true});
    await rm(outside, {recursive: true, force: true});
  }
});

'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { rpc, resolveDataDir } = require('../hooks/memory.cjs');

test('resolveDataDir honors --data-dir and VESTIGE_DATA_DIR', () => {
  const cliDir = path.join(os.tmpdir(), 'vestige-cli-dir');
  assert.equal(resolveDataDir(['--data-dir', cliDir], {}), path.resolve(cliDir));
  assert.equal(resolveDataDir([`--data-dir=${cliDir}`], {}), path.resolve(cliDir));
  const envDir = path.join(os.tmpdir(), 'vestige-env-dir');
  assert.equal(resolveDataDir([], { VESTIGE_DATA_DIR: envDir }), path.resolve(envDir));
});

test('rpc attaches directly to loopback .serve.endpoint with zero child spawn, and falls back when stale', async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-attach-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const token = 'test-attach-token-0123456789abcdef';
  const pid = process.pid;
  let handshakes = 0;

  const server = net.createServer(sock => {
    let stage = 'hello';
    let buf = '';
    sock.on('data', chunk => {
      buf += chunk.toString('utf8');
      while (true) {
        const nl = buf.indexOf('\n');
        if (nl < 0) break;
        const line = buf.slice(0, nl).trimEnd();
        buf = buf.slice(nl + 1);
        if (stage === 'hello') {
          if (line !== `vestige-attach/1 ${token}`) {
            sock.destroy();
            return;
          }
          handshakes += 1;
          stage = 'rpc';
          sock.write(`vestige-attached/1 ${pid}\n`);
          continue;
        }
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.method === 'initialize') {
          sock.write(JSON.stringify({
            jsonrpc: '2.0',
            id: msg.id,
            result: { protocolVersion: '2025-06-18', capabilities: {}, serverInfo: { name: 'direct-test', version: '6.2.2' } },
          }) + '\n');
        } else if (msg.method === 'tools/call') {
          sock.write(JSON.stringify({
            jsonrpc: '2.0',
            id: msg.id,
            result: {
              structuredContent: {
                memories: [{ id: 'mem-0000000000000001', content: 'Attached without spawn' }],
                backfill_autofire: true,
              },
            },
          }) + '\n');
        }
      }
    });
  });

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => server.close());
  const { port } = server.address();
  fs.writeFileSync(path.join(dir, '.serve.endpoint'), `${port} ${token} ${pid}\n`);

  const fakeBin = path.join(dir, process.platform === 'win32' ? 'vestige-mcp.exe' : 'vestige-mcp');
  const transport = rpc(fakeBin, ['--data-dir', dir], 5000, { ...process.env });
  t.after(() => transport.close());
  await transport.ready();
  assert.equal(transport.isDirectAttached(), true);
  assert.equal(handshakes, 1);

  const packet = await transport.tool('session', { action: 'start', memory_hook: { limit: 3 } });
  assert.equal(packet.memories[0].id, 'mem-0000000000000001');
  assert.equal(packet.backfill_autofire, true);
  transport.close();

  // When .serve.endpoint points to a dead port, direct attach fails cleanly and falls back to spawn
  const staleDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-stale-attach-'));
  t.after(() => fs.rmSync(staleDir, { recursive: true, force: true }));
  fs.writeFileSync(path.join(staleDir, '.serve.endpoint'), `1 ${token} ${pid}\n`);
  const fallbackScript = path.join(staleDir, 'mock-fallback.cjs');
  fs.writeFileSync(fallbackScript, `
    require('node:readline').createInterface({ input: process.stdin }).on('line', line => {
      const m = JSON.parse(line);
      if (!m.id) return;
      const result = m.method === 'initialize'
        ? { protocolVersion: '2025-06-18', capabilities: {} }
        : { structuredContent: { memories: [], fallback: true } };
      process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: m.id, result }) + '\\n');
    });
  `);
  const fb = rpc(fallbackScript, ['--data-dir', staleDir], 5000, { ...process.env, VESTIGE_DIRECT_ATTACH: '1' });
  t.after(() => fb.close());
  await fb.ready();
  assert.equal(fb.isDirectAttached(), false);
  const fbRes = await fb.tool('session', { action: 'start', memory_hook: { limit: 1 } });
  assert.equal(fbRes.fallback, true);
});

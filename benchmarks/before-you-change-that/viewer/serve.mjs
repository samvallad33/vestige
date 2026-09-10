#!/usr/bin/env node
import {createReadStream} from 'node:fs';
import {realpath, stat} from 'node:fs/promises';
import {createServer} from 'node:http';
import {dirname, extname, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const benchmarkRoot = resolve(here, '..');
const PUBLIC_FILES = new Set([
  'viewer/index.html',
  'viewer/style.css',
  'viewer/app.js',
  'viewer/replay-core.mjs',
  'evidence/manifest.json',
  'evidence/recording-timing.json',
  'evidence/result.json',
  'evidence/control.jsonl',
  'evidence/mcp-memory-service.jsonl',
  'evidence/vestige.jsonl',
]);
const MIME = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
};

function safePath(pathname, root, allowedFiles) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  const relative = decoded.replace(/^\/+/, '') || 'viewer/index.html';
  const normalized = relative === 'viewer/' ? 'viewer/index.html' : relative;
  if (!allowedFiles.has(normalized)) return null;
  const candidate = resolve(root, normalized);
  if (!candidate.startsWith(`${root}${sep}`)) return null;
  return candidate;
}

export async function startServer({host = '127.0.0.1', port = 0, root = benchmarkRoot, allowedFiles = PUBLIC_FILES} = {}) {
  const realRoot = await realpath(root);
  const server = createServer(async (request, response) => {
    const url = new URL(request.url || '/', 'http://127.0.0.1');
    if (url.pathname === '/') {
      response.writeHead(302, {location: '/viewer/'});
      response.end();
      return;
    }
    const candidate = safePath(url.pathname, realRoot, allowedFiles);
    if (!candidate) {
      response.writeHead(400, {'content-type': 'text/plain; charset=utf-8'});
      response.end('Invalid path');
      return;
    }
    try {
      const path = await realpath(candidate);
      if (!path.startsWith(`${realRoot}${sep}`)) {
        response.writeHead(403, {'content-type': 'text/plain; charset=utf-8'});
        response.end('Path escapes public root');
        return;
      }
      const finalInfo = await stat(path);
      if (!finalInfo.isFile()) throw Object.assign(new Error('Not a file'), {code: 'ENOENT'});
      response.writeHead(200, {
        'content-type': MIME[extname(path)] || 'application/octet-stream',
        'content-length': finalInfo.size,
        'cache-control': 'no-store',
        'x-content-type-options': 'nosniff',
        'content-security-policy': "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
      });
      createReadStream(path).pipe(response);
    } catch (error) {
      response.writeHead(error.code === 'ENOENT' ? 404 : 500, {'content-type': 'text/plain; charset=utf-8'});
      response.end(error.code === 'ENOENT' ? 'Not found' : 'Server error');
    }
  });
  await new Promise((resolveListen, rejectListen) => {
    server.once('error', rejectListen);
    server.listen(port, host, resolveListen);
  });
  return server;
}

function parseArgs(argv) {
  const parsed = {host: '127.0.0.1', port: 4173};
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--port' && argv[index + 1]) parsed.port = Number(argv[++index]);
    else if (argv[index] === '--host' && argv[index + 1]) parsed.host = argv[++index];
    else throw new Error('usage: node serve.mjs [--host 127.0.0.1] [--port 4173]');
  }
  if (!Number.isInteger(parsed.port) || parsed.port < 0 || parsed.port > 65535) throw new Error('Invalid port');
  return parsed;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const options = parseArgs(process.argv.slice(2));
  const server = await startServer(options);
  const address = server.address();
  console.log(`PUBLIC_REPLAY=http://${options.host}:${String(address.port)}/viewer/`);
  await new Promise((resolveSignal) => {
    process.once('SIGINT', resolveSignal);
    process.once('SIGTERM', resolveSignal);
  });
  await new Promise((resolveClose) => server.close(resolveClose));
}

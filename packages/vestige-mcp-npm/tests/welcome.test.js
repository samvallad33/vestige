'use strict';

const assert = require('assert/strict');
const { EventEmitter } = require('events');
const { spawnSync } = require('child_process');
const path = require('path');
const test = require('node:test');

const welcome = require('../scripts/welcome');

const {
  VESTIGE_PRO_SCREEN_URL,
  WELCOME_BANNER,
  WELCOME_TIMEOUT_MS,
  shouldOfferWelcome,
  classifyWelcomeKey,
  openDestinationUrl,
  offerWelcome,
} = welcome;

function fakeStdin() {
  const stdin = new EventEmitter();
  stdin.isTTY = true;
  stdin.isRaw = false;
  stdin.setRawMode = (value) => {
    stdin.isRaw = Boolean(value);
  };
  stdin.setEncoding = () => {};
  stdin.resume = () => {};
  stdin.pause = () => {};
  stdin.isPaused = () => true;
  return stdin;
}

test('postinstall.js and welcome.js still parse', () => {
  const scripts = [
    path.join(__dirname, '..', 'scripts', 'postinstall.js'),
    path.join(__dirname, '..', 'scripts', 'welcome.js'),
  ];
  for (const script of scripts) {
    const result = spawnSync(process.execPath, ['--check', script], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.stdout);
  }
});

test('destination URL is a named constant pointing at the Pro account site', () => {
  assert.equal(VESTIGE_PRO_SCREEN_URL, 'https://vestige-pro-production.fly.dev');
  assert.equal(WELCOME_TIMEOUT_MS, 15_000);
});

test('banner copy is exact', () => {
  assert.equal(
    WELCOME_BANNER,
    [
      '✦  Vestige Pro & Operator — out now.',
      '',
      '   Continuity across every machine.',
      '   Receipt → permit → effect. Fail closed.',
      '',
      '   Press Enter to open the screen · q to skip',
    ].join('\n')
  );
  assert.doesNotMatch(WELCOME_BANNER, /\$19/);
});

test('TTY gate stays closed in CI, npm -y, and non-TTY installs', () => {
  const tty = { stdoutIsTTY: true, stdinIsTTY: true };

  assert.equal(shouldOfferWelcome({ ...tty, env: {} }), true);
  assert.equal(shouldOfferWelcome({ ...tty, env: { CI: 'true' } }), false);
  assert.equal(shouldOfferWelcome({ ...tty, env: { CI: '1' } }), false);
  assert.equal(shouldOfferWelcome({ ...tty, env: { CONTINUOUS_INTEGRATION: 'true' } }), false);
  assert.equal(shouldOfferWelcome({ ...tty, env: { npm_config_yes: 'true' } }), false);
  assert.equal(shouldOfferWelcome({ ...tty, env: { npm_config_yes: '1' } }), false);
  assert.equal(shouldOfferWelcome({ stdoutIsTTY: false, stdinIsTTY: true, env: {} }), false);
  assert.equal(shouldOfferWelcome({ stdoutIsTTY: true, stdinIsTTY: false, env: {} }), false);
  assert.equal(shouldOfferWelcome({ stdoutIsTTY: false, stdinIsTTY: false, env: {} }), false);
});

test('key classifier maps Enter, q, and Ctrl-C', () => {
  assert.equal(classifyWelcomeKey('\r'), 'open');
  assert.equal(classifyWelcomeKey('\n'), 'open');
  assert.equal(classifyWelcomeKey('\r\n'), 'open');
  assert.equal(classifyWelcomeKey('q'), 'skip');
  assert.equal(classifyWelcomeKey('Q'), 'skip');
  assert.equal(classifyWelcomeKey('q\n'), 'skip');
  assert.equal(classifyWelcomeKey('\u0003'), 'cancel');
  assert.equal(classifyWelcomeKey('x'), null);
});

test('openDestinationUrl uses the platform opener', () => {
  const calls = [];
  const spawnFn = (command, args, options) => {
    calls.push({ command, args, options });
    return { unref() {} };
  };

  openDestinationUrl(VESTIGE_PRO_SCREEN_URL, { spawnFn, platform: 'darwin' });
  openDestinationUrl(VESTIGE_PRO_SCREEN_URL, { spawnFn, platform: 'linux' });
  openDestinationUrl(VESTIGE_PRO_SCREEN_URL, { spawnFn, platform: 'win32' });

  assert.deepEqual(calls[0], {
    command: 'open',
    args: [VESTIGE_PRO_SCREEN_URL],
    options: { stdio: 'ignore', detached: true },
  });
  assert.deepEqual(calls[1], {
    command: 'xdg-open',
    args: [VESTIGE_PRO_SCREEN_URL],
    options: { stdio: 'ignore', detached: true },
  });
  assert.equal(calls[2].command, 'cmd');
  assert.deepEqual(calls[2].args, ['/c', 'start', '', VESTIGE_PRO_SCREEN_URL]);
});

test('offerWelcome is a no-op when the TTY gate is closed', async () => {
  const chunks = [];
  const result = await offerWelcome({
    env: { CI: 'true' },
    stdoutIsTTY: true,
    stdinIsTTY: true,
    write: (text) => chunks.push(text),
  });
  assert.deepEqual(result, { offered: false, action: 'gated' });
  assert.deepEqual(chunks, []);
});

test('offerWelcome opens the screen on Enter', async () => {
  const stdin = fakeStdin();
  const chunks = [];
  const opened = [];
  const pending = offerWelcome({
    env: {},
    stdin,
    stdoutIsTTY: true,
    stdinIsTTY: true,
    write: (text) => chunks.push(text),
    openUrl: (url) => opened.push(url),
    timeoutMs: 1_000,
  });
  stdin.emit('data', '\r');
  const result = await pending;
  assert.deepEqual(result, { offered: true, action: 'open' });
  assert.deepEqual(opened, [VESTIGE_PRO_SCREEN_URL]);
  assert.match(chunks.join(''), /Vestige Pro & Operator/);
  assert.match(chunks.join(''), /Opening the screen/);
});

test('offerWelcome skips cleanly on q and Ctrl-C', async () => {
  for (const [key, action] of [
    ['q', 'skip'],
    ['\u0003', 'cancel'],
  ]) {
    const stdin = fakeStdin();
    const opened = [];
    const pending = offerWelcome({
      env: {},
      stdin,
      stdoutIsTTY: true,
      stdinIsTTY: true,
      write: () => {},
      openUrl: (url) => opened.push(url),
      timeoutMs: 1_000,
    });
    stdin.emit('data', key);
    const result = await pending;
    assert.deepEqual(result, { offered: true, action });
    assert.deepEqual(opened, []);
  }
});

test('offerWelcome times out with a URL fallback', async () => {
  const stdin = fakeStdin();
  const chunks = [];
  const result = await offerWelcome({
    env: {},
    stdin,
    stdoutIsTTY: true,
    stdinIsTTY: true,
    write: (text) => chunks.push(text),
    openUrl: () => {
      throw new Error('should not open on timeout');
    },
    timeoutMs: 20,
  });
  assert.deepEqual(result, { offered: true, action: 'timeout' });
  assert.match(chunks.join(''), /Timed out/);
  assert.ok(chunks.join('').includes(VESTIGE_PRO_SCREEN_URL));
});

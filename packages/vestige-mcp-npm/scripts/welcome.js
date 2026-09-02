'use strict';

const { spawn } = require('child_process');

/**
 * Destination opened when the user presses Enter on the Pro & Operator
 * welcome. Named so we can retarget later without hunting install copy.
 */
const VESTIGE_PRO_SCREEN_URL = 'https://github.com/samvallad33/vestige#vestige-pro';
const WELCOME_TIMEOUT_MS = 15_000;

const WELCOME_BANNER = [
  '✦  Vestige Pro & Operator — out now.',
  '',
  '   Continuity across every machine.',
  '   Receipt → permit → effect. Fail closed.',
  '',
  '   Press Enter to open the screen · q to skip',
].join('\n');

function envFlagEnabled(value) {
  if (value == null) return false;
  const normalized = String(value).trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes';
}

/**
 * Interactive welcome is for a real human at a terminal. CI, `npm -y`,
 * Docker builds, and piped/non-TTY `npm i` must never block.
 */
function shouldOfferWelcome({
  env = process.env,
  stdoutIsTTY = Boolean(process.stdout && process.stdout.isTTY),
  stdinIsTTY = Boolean(process.stdin && process.stdin.isTTY),
} = {}) {
  if (envFlagEnabled(env.CI) || envFlagEnabled(env.CONTINUOUS_INTEGRATION)) {
    return false;
  }
  if (envFlagEnabled(env.npm_config_yes)) {
    return false;
  }
  return Boolean(stdoutIsTTY && stdinIsTTY);
}

function openDestinationUrl(
  url = VESTIGE_PRO_SCREEN_URL,
  { spawnFn = spawn, platform = process.platform } = {}
) {
  const detached = { stdio: 'ignore', detached: true };
  let child;
  if (platform === 'darwin') {
    child = spawnFn('open', [url], detached);
  } else if (platform === 'win32') {
    child = spawnFn('cmd', ['/c', 'start', '', url], {
      ...detached,
      windowsHide: true,
    });
    if (child && typeof child.on === 'function') {
      child.on('error', () => {
        const fallback = spawnFn(
          'powershell',
          ['-NoProfile', '-Command', `Start-Process ${JSON.stringify(url)}`],
          { ...detached, windowsHide: true }
        );
        if (fallback && typeof fallback.unref === 'function') fallback.unref();
      });
    }
  } else {
    child = spawnFn('xdg-open', [url], detached);
  }
  if (child && typeof child.unref === 'function') child.unref();
  return child;
}

function restoreStdin(stdin, previous) {
  try {
    if (stdin.setRawMode && previous && typeof previous.rawMode === 'boolean') {
      stdin.setRawMode(previous.rawMode);
    }
  } catch {
    // Leave the stream as-is if the TTY rejects the restore.
  }
  try {
    if (previous && previous.paused && typeof stdin.pause === 'function') {
      stdin.pause();
    }
  } catch {
    // ignore
  }
}

function classifyWelcomeKey(chunk) {
  const key = Buffer.isBuffer(chunk) ? chunk.toString('utf8') : String(chunk);
  if (key.includes('\u0003')) {
    return 'cancel';
  }
  if (key === '\r' || key === '\n' || key === '\r\n') {
    return 'open';
  }
  const trimmed = key.replace(/\r/g, '').replace(/\n/g, '');
  if (trimmed.toLowerCase() === 'q') {
    return 'skip';
  }
  if (trimmed === '' && /[\r\n]/.test(key)) {
    return 'open';
  }
  return null;
}

function waitForWelcomeKey({ stdin = process.stdin, timeoutMs = WELCOME_TIMEOUT_MS } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const previous = {
      rawMode: typeof stdin.isRaw === 'boolean' ? stdin.isRaw : false,
      paused: typeof stdin.isPaused === 'function' ? stdin.isPaused() : true,
    };

    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stdin.removeListener('data', onData);
      restoreStdin(stdin, previous);
      resolve(result);
    };

    const onData = (chunk) => {
      const action = classifyWelcomeKey(chunk);
      if (action) finish(action);
    };

    try {
      if (typeof stdin.setRawMode === 'function') {
        stdin.setRawMode(true);
      }
    } catch {
      // Line-buffered stdin still works: Enter and `q` classify the same.
    }

    if (typeof stdin.setEncoding === 'function') {
      stdin.setEncoding('utf8');
    }
    if (typeof stdin.resume === 'function') {
      stdin.resume();
    }
    stdin.on('data', onData);

    const timer = setTimeout(() => finish('timeout'), timeoutMs);
  });
}

async function offerWelcome(options = {}) {
  const {
    env = process.env,
    stdout = process.stdout,
    stdin = process.stdin,
    stdoutIsTTY = Boolean(stdout && stdout.isTTY),
    stdinIsTTY = Boolean(stdin && stdin.isTTY),
    timeoutMs = WELCOME_TIMEOUT_MS,
    openUrl = openDestinationUrl,
    write = (text) => {
      if (stdout && typeof stdout.write === 'function') {
        stdout.write(text);
      }
    },
  } = options;

  if (!shouldOfferWelcome({ env, stdoutIsTTY, stdinIsTTY })) {
    return { offered: false, action: 'gated' };
  }

  write(`\n${WELCOME_BANNER}\n`);

  const action = await waitForWelcomeKey({ stdin, timeoutMs });

  if (action === 'open') {
    try {
      openUrl(VESTIGE_PRO_SCREEN_URL);
      write('\nOpening the screen…\n');
    } catch {
      write(`\nCould not open the browser. Open this URL:\n${VESTIGE_PRO_SCREEN_URL}\n`);
    }
    return { offered: true, action };
  }

  if (action === 'timeout') {
    write(`\nTimed out. Open the screen anytime:\n${VESTIGE_PRO_SCREEN_URL}\n`);
    return { offered: true, action };
  }

  write('\n');
  return { offered: true, action };
}

module.exports = {
  VESTIGE_PRO_SCREEN_URL,
  WELCOME_TIMEOUT_MS,
  WELCOME_BANNER,
  shouldOfferWelcome,
  classifyWelcomeKey,
  openDestinationUrl,
  offerWelcome,
};

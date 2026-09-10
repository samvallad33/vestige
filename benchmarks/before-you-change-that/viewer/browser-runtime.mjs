import {existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import {homedir} from 'node:os';
import {join} from 'node:path';

const require = createRequire(import.meta.url);

export function loadPlaywright() {
  const candidates = [
    process.env.PLAYWRIGHT_PATH,
    'playwright',
    join(homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'),
  ].filter(Boolean);
  const errors = [];
  for (const candidate of candidates) {
    try {
      return require(candidate);
    } catch (error) {
      errors.push(`${candidate}: ${error.code || error.message}`);
    }
  }
  throw new Error(`Playwright was not found. Set PLAYWRIGHT_PATH to its package directory. ${errors.join('; ')}`);
}

export function browserLaunchOptions(chromium, extra = {}) {
  const candidates = [
    process.env.CHROME_PATH,
    chromium.executablePath(),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  const executablePath = candidates.find(existsSync);
  if (!executablePath) {
    throw new Error('No Chromium-family browser executable found. Set CHROME_PATH.');
  }
  return {headless: true, executablePath, ...extra};
}

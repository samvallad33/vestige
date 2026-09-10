import assert from 'node:assert/strict';
import test from 'node:test';
import {browserLaunchOptions, loadPlaywright} from './browser-runtime.mjs';
import {startServer} from './serve.mjs';

async function checkView(browser, origin, comparison, expectedTitle, expectedCounter) {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}, colorScheme: 'dark'});
  const errors = [];
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    await page.goto(`${origin}/viewer/?compare=${comparison}`, {waitUntil: 'networkidle'});
    await page.waitForFunction(() => document.documentElement.dataset.replayReady === 'true');
    assert.equal(await page.locator('.boundary').innerText(), 'Recorded-event replay · paths redacted · no new model run');
    assert.equal(await page.locator('.lane').count(), 2);
    assert.equal(await page.locator('.lane-title').first().innerText(), expectedTitle);
    assert.deepEqual(await page.locator('.lane-score').allInnerTexts(), ['21 / 21', '21 / 21']);
    assert.equal(await page.locator('.lane').first().locator('[data-usage="input_tokens"]').innerText(), expectedCounter);

    const replay = await page.evaluate(() => ({
      duration: window.__PUBLIC_REPLAY__.duration,
      counts: window.__PUBLIC_REPLAY__.counts,
    }));
    assert.ok(replay.duration > 900);
    await page.evaluate(() => window.__PUBLIC_REPLAY__.seek(300));
    const progress = Number((await page.locator('#event-progress').innerText()).split(' ')[0]);
    assert.ok(progress > 2);

    await page.evaluate(() => window.__PUBLIC_REPLAY__.showEnd());
    assert.equal(await page.locator('#end-card').isVisible(), true);
    assert.equal(await page.locator('#end-card h2').innerText(), 'All final application scores: 21 / 21');
    const endText = await page.locator('#end-card').innerText();
    assert.match(endText, /Control: Natural completion/);
    assert.match(endText, /MCP Memory Service: Timed out at 900s/);
    assert.match(endText, /Vestige: Natural completion/);
    assert.match(endText, /Control · 1,182,794 input incl\. 1,079,808 cached · 18,739 output/);
    assert.match(endText, /MCP Memory Service · counters unavailable \(no final usage row after timeout\)/);
    assert.match(endText, /Vestige · 1,302,101 input incl\. 1,227,648 cached · 15,208 output/);
    assert.match(endText, /Reconciliation repair passed; unrelated export checks preserved\./);
    assert.match(endText, /Coordinator marked it complete only after the application tests passed\./);
    assert.doesNotMatch(endText, /demo_ready/);
    assert.deepEqual(errors, []);
  } finally {
    await page.close();
  }
}

test('viewer renders both public comparisons with evidence-derived boundaries', {timeout: 60_000}, async () => {
  const {chromium} = loadPlaywright();
  const server = await startServer();
  let browser;
  try {
    browser = await chromium.launch(browserLaunchOptions(chromium));
    const origin = `http://127.0.0.1:${String(server.address().port)}`;
    await checkView(browser, origin, 'control', 'Control', '1,182,794');
    await checkView(browser, origin, 'mcp-memory-service', 'MCP Memory Service', 'not emitted');
  } finally {
    if (browser) await browser.close();
    await new Promise((resolveClose) => server.close(resolveClose));
  }
});

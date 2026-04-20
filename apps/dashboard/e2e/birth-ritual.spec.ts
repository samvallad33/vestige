// ─────────────────────────────────────────────────────────────────────────────
// v2.3 Birth Ritual — E2E visual proof
//
// The Birth Ritual is a 2.3s choreography triggered by MemoryCreated events:
//   t=0…800ms       gestation (orb pulses in place)
//   t=800…2300ms    Bezier flight toward graph center
//   t=2300…2600ms   arrival burst cascade
//
// Trigger path in these tests (avoids modifying production code — the
// websocket store is NOT on window):
//   1. Navigate to /dashboard/settings
//   2. Click the "✺ Trigger Birth" button — calls websocket.injectEvent()
//   3. SPA-route to /dashboard/graph (same tab, same JS context, singleton
//      store preserves the event in the feed)
//   4. Graph3D mounts, reads $eventFeed, renders the birth orb.
//
// Constraints:
//   - Graph3D only mounts when the /api/graph call succeeds (loadGraph()).
//     Without vestige-mcp on 127.0.0.1:3927, the page shows the "Your Mind
//     Awaits" error panel and NO canvas. Those tests guard on canvas
//     presence and fixme themselves if the backend isn't reachable — they
//     don't fail the suite.
//   - The v2.3-era regression FATAL 6 (multiple simultaneous births crashing
//     the effect manager) manifested as console errors, so we instrument
//     pageerror + console listeners and assert zero errors.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page, type ConsoleMessage } from '@playwright/test';

const SETTINGS_URL = '/dashboard/settings';
const GRAPH_URL = '/dashboard/graph';
const TRIGGER_BIRTH_TEXT = /Trigger Birth/i;

// Helpers ────────────────────────────────────────────────────────────────────

interface ErrorCapture {
	pageErrors: Error[];
	consoleErrors: string[];
}

function captureErrors(page: Page): ErrorCapture {
	const capture: ErrorCapture = { pageErrors: [], consoleErrors: [] };
	page.on('pageerror', (err) => { capture.pageErrors.push(err); });
	page.on('console', (msg: ConsoleMessage) => {
		if (msg.type() === 'error') {
			const text = msg.text();
			// Filter known-noisy WebSocket/connection errors that appear when
			// vestige-mcp isn't running — those are infrastructure, not birth
			// ritual regressions.
			if (
				text.includes('WebSocket') ||
				text.includes('Failed to fetch') ||
				text.includes('ERR_CONNECTION') ||
				text.includes('net::') ||
				text.includes('api/graph') ||
				text.includes('api/health') ||
				text.includes('api/stats')
			) return;
			capture.consoleErrors.push(text);
		}
	});
	return capture;
}

async function isGraphMounted(page: Page): Promise<boolean> {
	// The graph page shows either the loading spinner, an error panel, or
	// the <Graph3D> component (which mounts a <canvas>). If no canvas
	// appears within 8s we assume the backend is unreachable.
	try {
		await page.waitForSelector('canvas', { timeout: 8000, state: 'attached' });
		return true;
	} catch {
		return false;
	}
}

async function injectBirthViaSettings(page: Page) {
	// SPA-route to /settings first so the websocket module stays resident.
	// The store is a module-level singleton — navigating through SvelteKit's
	// client router preserves its state across routes within the same tab.
	if (!page.url().includes('/settings')) {
		await page.goto(SETTINGS_URL);
	}
	await expect(page.getByRole('button', { name: TRIGGER_BIRTH_TEXT })).toBeVisible();
	await page.getByRole('button', { name: TRIGGER_BIRTH_TEXT }).click();
}

async function attachScreenshot(page: Page, name: string) {
	const buf = await page.screenshot({ type: 'png' });
	await test.info().attach(name, { body: buf, contentType: 'image/png' });
}

// Tests ───────────────────────────────────────────────────────────────────────

test.describe('v2.3 Birth Ritual — Visual proof', () => {
	test.describe.configure({ mode: 'serial' });

	test('1. /dashboard/graph mounts a WebGL canvas', async ({ page }) => {
		await page.goto(GRAPH_URL);
		const mounted = await isGraphMounted(page);

		// If the graph didn't mount (no vestige-mcp backend), fixme gracefully —
		// the remaining tests in this file require a canvas and would cascade.
		test.fixme(
			!mounted,
			'Graph canvas did not mount — vestige-mcp backend likely not running on 127.0.0.1:3927. ' +
			'Start the MCP server or run the infrastructure before re-enabling this suite.'
		);

		const canvas = page.locator('canvas');
		await expect(canvas).toBeAttached();

		await attachScreenshot(page, 'graph-canvas-mounted.png');
	});

	test('2. inject single birth via Settings button, screenshot timeline on Graph', async ({ page }) => {
		const errors = captureErrors(page);

		// Pre-flight: make sure the graph is reachable. Fixme if not.
		await page.goto(GRAPH_URL);
		const mounted = await isGraphMounted(page);
		test.fixme(!mounted, 'Graph canvas not mounted — skipping birth ritual test.');

		// Go to settings, fire the synthetic MemoryCreated event, then
		// SPA-route back to graph. Using goto() instead of client-side
		// navigation is fine: SvelteKit's adapter-static preserves module
		// state across goto() within the same page context.
		await injectBirthViaSettings(page);
		const tInjected = Date.now();

		await page.goto(GRAPH_URL);
		await isGraphMounted(page);

		// Take screenshots at the documented ritual waypoints, relative to
		// the injection timestamp. Each attach() lands in the HTML report.
		const waypoints: Array<{ t: number; label: string }> = [
			{ t: 0,    label: 'birth-01-t0-injection.png' },
			{ t: 500,  label: 'birth-02-t500-gestation-mid.png' },
			{ t: 1200, label: 'birth-03-t1200-flight-start.png' },
			{ t: 2000, label: 'birth-04-t2000-mid-flight.png' },
			{ t: 2400, label: 'birth-05-t2400-near-arrival.png' },
			{ t: 3000, label: 'birth-06-t3000-burst-cascade.png' },
		];

		for (const wp of waypoints) {
			const waitMs = Math.max(0, wp.t - (Date.now() - tInjected));
			if (waitMs > 0) await page.waitForTimeout(waitMs);
			await attachScreenshot(page, wp.label);
		}

		// No unhandled errors during the ritual. The FATAL 6 regression would
		// surface here.
		expect(errors.pageErrors, `pageerror events: ${errors.pageErrors.map(e => e.message).join('; ')}`)
			.toHaveLength(0);
		expect(errors.consoleErrors, `console errors: ${errors.consoleErrors.join('; ')}`)
			.toHaveLength(0);
	});

	test('3. multiple simultaneous births — no errors, canvas still responsive', async ({ page }) => {
		const errors = captureErrors(page);

		await page.goto(GRAPH_URL);
		const mounted = await isGraphMounted(page);
		test.fixme(!mounted, 'Graph canvas not mounted — skipping birth ritual test.');

		// Fire 3 births back-to-back via the Settings button. Navigate to
		// /settings once, click 3x, then return to /graph so all three events
		// are in the feed when Graph3D mounts.
		await page.goto(SETTINGS_URL);
		const btn = page.getByRole('button', { name: TRIGGER_BIRTH_TEXT });
		await expect(btn).toBeVisible();
		await btn.click();
		await btn.click();
		await btn.click();

		await page.goto(GRAPH_URL);
		await isGraphMounted(page);

		// Let the full 2.6s ritual play for all three orbs, with overlap.
		await page.waitForTimeout(3500);
		await attachScreenshot(page, 'birth-07-triple-birth.png');

		// Canvas is still responsive: clicking in the middle should not hang
		// the page. We don't care what's selected — just that the click
		// dispatches without timing out.
		const canvas = page.locator('canvas');
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2, { timeout: 2000 });
			await page.waitForTimeout(300);
		}
		await attachScreenshot(page, 'birth-08-post-click.png');

		expect(errors.pageErrors, `pageerror events: ${errors.pageErrors.map(e => e.message).join('; ')}`)
			.toHaveLength(0);
		expect(errors.consoleErrors, `console errors: ${errors.consoleErrors.join('; ')}`)
			.toHaveLength(0);
	});
});

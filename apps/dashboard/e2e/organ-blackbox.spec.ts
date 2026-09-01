// ─────────────────────────────────────────────────────────────────────────────
// Organ — Agent Black Box (/blackbox)
//
// The flight recorder for agent cognition. Data source: GET /api/traces (run
// summaries) + GET /api/traces/:runId (per-run events) + GET /api/receipts?run=…
// The route renders a full-bleed WebGPU trace field (RouteStage → BlackboxPass)
// PLUS a DOM instrument overlay (run picker, scrubber, event log, receipts).
//
// This spec proves the 7-point ship contract for THIS organ against the REAL
// brain (:3931), not mocks:
//   1. REACHABLE     — /dashboard/blackbox mounts a canvas + its DOM overlay.
//   2. REAL DATA     — the WebGPU field renders non-black AND the DOM run picker
//                      lists REAL runIds that match the live /api/traces payload
//                      (not fabricated demo rows).
//   3. ALIVE         — the trace membrane field animates (pixels change).
//   4. CRASH-FREE    — a grid of in-canvas picks + a hover survive with zero
//                      page/WebGPU errors and the field still renders after.
//   5. HONEST EMPTY  — with an EMPTY /api/traces the organ shows a calm empty
//                      state ("No agent runs recorded yet"), never a crash / black
//                      surface / fake data / errored field.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	BASE,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const ROUTE = '/blackbox';

// A grid of canvas-relative fractions covering the trace lanes (mid-field) and
// the HUD corners. BlackboxPass.pickAt() uses STATIC hit-rects (event x/y are
// NOT animated in the vertex shader — only the glow radius pulses), so the
// moving-target pick trap does not apply here; picks land where the CPU expects.
const GRID: Array<[number, number]> = [
	[0.2, 0.35],
	[0.5, 0.35],
	[0.8, 0.35],
	[0.2, 0.5],
	[0.5, 0.5],
	[0.8, 0.5],
	[0.2, 0.65],
	[0.5, 0.65],
	[0.8, 0.65]
];

async function clickGrid(page: Page) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to click into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of GRID) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		await page.waitForTimeout(200);
	}
}

test.describe('Organ — Agent Black Box', () => {
	test('reachable + renders REAL trace data + alive + crash-free picks/hover', async ({ page }) => {
		const capture = captureErrors(page);

		// (1) REACHABLE — canvas mounts.
		const canvas = await gotoRoute(page, ROUTE);
		await expect(canvas).toBeAttached();

		// Cross-check the live API directly so we assert the DOM overlay consumes
		// the SAME real runs (not fabricated demo rows).
		const apiRuns = await page.evaluate(async () => {
			const res = await fetch('/api/traces?limit=100');
			if (!res.ok) return null;
			const data = (await res.json()) as { total: number; runs: { runId: string }[] };
			return { total: data.total, ids: data.runs.map((r) => r.runId) };
		});
		expect(apiRuns, 'live /api/traces must be reachable from the page').not.toBeNull();
		expect(apiRuns!.ids.length, 'real brain must have at least one recorded run').toBeGreaterThan(0);

		// The field builds over a few seconds — settle before sampling.
		await page.waitForTimeout(3000);

		// (2a) REAL DATA — the DOM run picker lists real runIds. The overlay shows
		// runId.replace('run_','').slice(0,10); assert the first real run's short id
		// appears in the picker. This proves it renders the API payload, not mock.
		const firstShort = apiRuns!.ids[0].replace('run_', '').slice(0, 10);
		await expect(
			page.locator('.run-id').filter({ hasText: firstShort }).first(),
			`run picker must render the real runId ${firstShort} from /api/traces`
		).toBeVisible({ timeout: 10_000 });

		// The picker must NOT show the "no runs recorded" empty copy when runs exist.
		await expect(page.locator('.runs .empty')).toHaveCount(0);

		// A run is auto-selected on mount → the event log renders real event rows.
		await expect(page.locator('.log-row').first(), 'selected run must render its event log').toBeVisible({
			timeout: 10_000
		});

		// (2b) REAL DATA — the WebGPU trace field renders non-black.
		const before = await sampleCanvas(page);
		expect(before.ok, 'blackbox canvas must be sampleable').toBe(true);
		expect(
			before.rendered,
			`blackbox field must render non-black (avgLum=${before.avgLum} var=${before.variance})`
		).toBe(true);

		// (3) ALIVE — the membrane field animates (pulse/brightness drive motion).
		const animated = await isAnimating(page, 900);
		expect(animated, 'blackbox trace membrane field must animate').toBe(true);

		// (4) CRASH-FREE — hover then a grid of in-canvas picks; field must survive.
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
			await page.waitForTimeout(250);
			await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.45);
			await page.waitForTimeout(250);
		}
		await clickGrid(page);

		// Zero page errors, zero WebGPU validation errors across every pick/hover.
		expectNoErrors(capture);

		// The render loop survived every pick — field still alive.
		const after = await sampleCanvas(page);
		expect(after.ok, 'blackbox canvas must still be sampleable after picks').toBe(true);
		expect(
			after.rendered,
			`blackbox field must still render after picks (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);

		await test.info().attach('blackbox-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});

	test('honest EMPTY state when no runs are recorded (no crash, no fake data)', async ({ page }) => {
		const capture = captureErrors(page);

		// Force the traces list to be empty — exercise the empty branch of the
		// organ without touching the real brain's data.
		await page.route('**/api/traces?*', async (route) => {
			await route.fulfill({
				status: 200,
				contentType: 'application/json',
				body: JSON.stringify({ total: 0, runs: [] })
			});
		});

		await page.goto(`${BASE}${ROUTE}`);
		const canvas = page.locator('canvas').first();
		await canvas.waitFor({ state: 'attached', timeout: 15_000 });

		// Calm, honest empty copy in the run picker — NOT a broken/errored surface.
		await expect(
			page.locator('.runs .empty'),
			'empty traces must show the honest "no runs recorded" copy'
		).toBeVisible({ timeout: 10_000 });
		await expect(page.locator('.runs .empty')).toContainText('No agent runs recorded yet');

		// The replay column shows its calm "Select a run to replay." prompt, not an
		// error and not fabricated events.
		await expect(page.locator('.center-msg').first()).toBeVisible();
		await expect(page.locator('.log-row')).toHaveCount(0);

		// The WebGPU field must NOT crash on the empty scene (eventCount === 0 path).
		expectNoErrors(capture);

		await test.info().attach('blackbox-empty.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

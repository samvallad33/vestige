// ─────────────────────────────────────────────────────────────────────────────
// Organ: /contradictions — the Immune Synapse Arena
//
// A full-bleed WebGPU membrane field (createContradictionsPasses) driven by the
// REAL /api/contradictions payload, with a DOM instrument overlay on top. Each
// contradiction is a trust-weighted seam between two opposing memory membranes;
// clicking a seam opens a receipt. The real brain currently returns ZERO
// contradictions, so the load-bearing behaviour here is the HONEST CALM state:
// the field must still render a non-black crimson vignette and animate, the DOM
// must show the calm "your memory agrees with itself" panel (NOT a fake stats
// bar, NOT a "Live" badge over mock data, NOT an error surface), and picks +
// hover must survive with zero page/WebGPU errors.
//
// This spec proves the organ-specific contract (real-data consumption + honest
// empty state + pick/hover survival + animation). Generic all-route smoke +
// cross-organ nav are covered elsewhere and not duplicated here.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const ROUTE = '/contradictions';

// A grid of canvas-relative fractions covering the central membrane field (where
// any seam geometry lives) plus the HUD corners. With 0 pairs every pick is a
// no-op, but the render loop must survive all of them.
const GRID: Array<[number, number]> = [
	[0.3, 0.35],
	[0.5, 0.35],
	[0.7, 0.35],
	[0.3, 0.6],
	[0.5, 0.5],
	[0.7, 0.6]
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

test.describe('Organ: /contradictions (Immune Synapse Arena)', () => {
	// (1) REACHABLE + (2) RENDERS REAL DATA + (3) ALIVE + (5) HONEST EMPTY
	test('reachable, renders the real immune field non-black, and animates', async ({ page }) => {
		const capture = captureErrors(page);

		// (2) Confirm the REAL brain data the organ consumes. The dev server proxies
		// /api → the live brain; the organ calls api.contradictions() on mount.
		const apiRes = await page.request.get('/api/contradictions');
		expect(apiRes.ok(), 'GET /api/contradictions must respond 200 from the real brain').toBe(true);
		const payload = await apiRes.json();
		expect(payload, 'payload must have a contradictions array').toHaveProperty('contradictions');
		expect(Array.isArray(payload.contradictions)).toBe(true);
		expect(payload).toHaveProperty('total');
		expect(payload).toHaveProperty('memoriesAnalyzed');
		// memoriesAnalyzed is a real, positive scan count from the live brain.
		expect(payload.memoriesAnalyzed, 'brain must report a real memoriesAnalyzed count').toBeGreaterThan(0);

		// (1) REACHABLE: route mounts a WebGPU canvas.
		const canvas = await gotoRoute(page, ROUTE);
		await expect(canvas).toBeAttached();
		// The immune membrane field builds over a couple seconds — settle.
		await page.waitForTimeout(3500);

		// (2/3) RENDERS non-black: even with 0 pairs the membrane pass draws a calm
		// crimson vignette + the chrome/nav text layer, so the field is real light,
		// not a black surface.
		const rendered = await sampleCanvas(page);
		expect(rendered.ok, 'contradictions canvas must be sampleable').toBe(true);
		expect(
			rendered.rendered,
			`contradictions field must render non-black (avgLum=${rendered.avgLum} var=${rendered.variance})`
		).toBe(true);

		// (3) ALIVE: the field animates (chrome telemetry frame counter + nav layer
		// tick every frame even in the calm empty state).
		const animated = await isAnimating(page, 900);
		expect(animated, 'immune field must animate (alive, not a frozen frame)').toBe(true);

		expectNoErrors(capture);
	});

	// (5) HONEST EMPTY / REAL-DATA: the DOM overlay reflects the real payload. With
	// the brain returning 0 contradictions it must show the calm empty panel and
	// must NOT fabricate a stats bar, a "Live" badge, or an error surface.
	test('shows the honest calm empty state — no fake stats, no fake Live badge, no error', async ({ page }) => {
		const capture = captureErrors(page);

		const payload = await (await page.request.get('/api/contradictions')).json();
		await gotoRoute(page, ROUTE);
		await page.waitForTimeout(2500);

		if (payload.total === 0 && payload.contradictions.length === 0) {
			// Calm honest empty panel is present.
			await expect(
				page.getByText('your memory agrees with itself'),
				'empty state must show the calm honest message'
			).toBeVisible();
			// The stats bar ("strong conflicts") is part of the populated branch only —
			// it must NOT render fabricated numbers over an empty dataset.
			expect(
				await page.getByText('strong conflicts').count(),
				'stats bar must be hidden when there is no real data (no fabricated numbers)'
			).toBe(0);
			// No "Live" badge fabricated over an empty/mock dataset anywhere on the page.
			expect(
				await page.getByText('Live', { exact: true }).count(),
				'no fake "Live" badge may render over an empty dataset'
			).toBe(0);
		} else {
			// If the brain ever DOES have contradictions, the populated stats bar must
			// render and the empty panel must be absent — the organ consumes real data.
			await expect(page.getByText('strong conflicts')).toBeVisible();
			expect(await page.getByText('your memory agrees with itself').count()).toBe(0);
		}

		// The error branch must NOT be showing — a healthy load is not an error.
		expect(
			await page.getByText("Couldn't load contradictions").count(),
			'a healthy load must not show the error surface'
		).toBe(0);

		expectNoErrors(capture);
	});

	// (4) CRASH-FREE click + hover: drive a real in-canvas pick grid and a hover.
	// The field (and its pickAt, which mirrors the CPU seam geometry) must survive
	// every interaction with zero page/WebGPU errors and keep rendering.
	test('survives an in-canvas click grid + hover with zero errors and keeps rendering', async ({ page }) => {
		const capture = captureErrors(page);

		await gotoRoute(page, ROUTE);
		await page.waitForTimeout(3500);

		const before = await sampleCanvas(page);
		expect(before.ok, 'canvas must be sampleable before interaction').toBe(true);
		expect(
			before.rendered,
			`field must render before interaction (avgLum=${before.avgLum} var=${before.variance})`
		).toBe(true);

		// Hover across the field (drives the RouteStage cursor lens + nav/chrome
		// hover pick — must not throw).
		const canvas = page.locator('canvas').first();
		const box = await canvas.boundingBox();
		expect(box).not.toBeNull();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
			await page.waitForTimeout(150);
			await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.45);
			await page.waitForTimeout(150);
		}

		// Real pick grid — every seam pick (no-op when empty) must be crash-free.
		await clickGrid(page);

		expectNoErrors(capture);

		// The render loop survived every interaction — field still alive.
		const after = await sampleCanvas(page);
		expect(after.ok, 'canvas must still be sampleable after interaction').toBe(true);
		expect(
			after.rendered,
			`field must still render after picks + hover (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);

		await test.info().attach('contradictions-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

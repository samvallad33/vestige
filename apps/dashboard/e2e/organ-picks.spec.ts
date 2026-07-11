// ─────────────────────────────────────────────────────────────────────────────
// Organ Picking — crash-free in-canvas picking proof
//
// Every organ field is a full-bleed WebGPU canvas with in-canvas picking (the
// RouteStage projects NDC ray → pass.pickAt). Timeline renders bitemporal
// growth-ring cells you click for a receipt; feed + duplicates have pickable
// rows. Exact picked-item assertions are impractical from the outside (the
// pickable geometry lives entirely in the GPU scene), so this spec proves the
// load-bearing invariant instead: PICKING NEVER CRASHES. We fire a grid of
// clicks across each field and assert (a) zero page/WebGPU errors on every
// click, and (b) the field is STILL rendering afterward — a pick that threw
// or corrupted the render loop would fail one of these.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import { captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

// A 6-point grid of canvas-RELATIVE fractions. Covers HUD (top-left), receipt
// column (right), and the central field where the pickable organ geometry lives.
const GRID: Array<[number, number]> = [
	[0.25, 0.35],
	[0.5, 0.35],
	[0.75, 0.35],
	[0.25, 0.65],
	[0.5, 0.65],
	[0.75, 0.65]
];

/** Click a grid of canvas-relative points; no assertions here (caller checks errors). */
async function clickGrid(page: Page) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to click into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of GRID) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		// let the pick + any receipt re-render settle
		await page.waitForTimeout(250);
	}
}

/**
 * Full click-robustness pass for one organ route:
 *   1. field renders + is non-black after settling
 *   2. a 6-point click grid fires with zero errors (picking never crashes)
 *   3. the field is STILL rendering after all the clicks
 */
async function robustPickPass(page: Page, path: string) {
	const capture = captureErrors(page);

	await gotoRoute(page, path);
	// organs build over a few seconds — settle before asserting render
	await page.waitForTimeout(3000);

	const before = await sampleCanvas(page);
	expect(before.ok, `${path}: canvas must be sampleable`).toBe(true);
	expect(before.rendered, `${path}: field must render non-black before picking (avgLum=${before.avgLum} var=${before.variance})`).toBe(true);

	await clickGrid(page);

	// Picking must never crash: no page errors, no WebGPU validation errors,
	// on ANY of the grid clicks.
	expectNoErrors(capture);

	// The render loop must have survived every pick — field still alive.
	const after = await sampleCanvas(page);
	expect(after.ok, `${path}: canvas must still be sampleable after picking`).toBe(true);
	expect(after.rendered, `${path}: field must still render after picking (avgLum=${after.avgLum} var=${after.variance})`).toBe(true);
}

test.describe('Organ picking — crash-free in-canvas picks', () => {
	test('timeline: growth-ring cells are clickable and the field survives every pick', async ({ page }) => {
		const capture = captureErrors(page);

		const canvas = await gotoRoute(page, '/timeline');
		await expect(canvas).toBeAttached();
		// Let the field mount but sample animation DURING the growth-ring build:
		// verified the timeline field is a *settling* animation (it oscillates for
		// ~2.5s then rests on a final frame), so we probe multiple short windows
		// right after mount and require at least one to change. A perpetual sample
		// after settle can catch two identical resting frames and false-negative.
		await page.waitForTimeout(600);

		// (1a) field renders non-black
		const rendered = await sampleCanvas(page);
		expect(rendered.ok, 'timeline canvas must be sampleable').toBe(true);
		expect(
			rendered.rendered,
			`timeline field must render non-black (avgLum=${rendered.avgLum} var=${rendered.variance})`
		).toBe(true);

		// (1b) field animates — sample several windows, require motion in at least one
		let animated = false;
		let prev = await canvas.screenshot({ timeout: 8000 });
		for (let i = 0; i < 6 && !animated; i++) {
			await page.waitForTimeout(400);
			const cur = await canvas.screenshot({ timeout: 8000 });
			if (Buffer.compare(prev, cur) !== 0) animated = true;
			prev = cur;
		}
		expect(animated, 'timeline growth-ring field must animate (some frame must change during build)').toBe(true);

		// let the field finish settling before the pick grid
		await page.waitForTimeout(1500);

		// (2) click a grid across the canvas — every pick must be crash-free
		await clickGrid(page);
		expectNoErrors(capture);

		// (3) field is still rendering after all the clicks
		const still = await sampleCanvas(page);
		expect(still.ok, 'timeline canvas must still be sampleable after picks').toBe(true);
		expect(
			still.rendered,
			`timeline field must still render after picks (avgLum=${still.avgLum} var=${still.variance})`
		).toBe(true);

		await test.info().attach('timeline-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});

	test('feed: pickable rows survive a click grid with no crash', async ({ page }) => {
		await robustPickPass(page, '/feed');
		await test.info().attach('feed-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});

	test('duplicates: pickable rows survive a click grid with no crash', async ({ page }) => {
		await robustPickPass(page, '/duplicates');
		await test.info().attach('duplicates-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

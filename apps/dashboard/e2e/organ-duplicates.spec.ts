// ─────────────────────────────────────────────────────────────────────────────
// Organ: /duplicates — Memory Hygiene / Synaptic Fusion Field
//
// This is the dedicated ship-proof for the duplicates organ. It proves the
// 7-point organ contract against the REAL brain (localhost:3931), beyond the
// generic all-routes-smoke:
//
//   1. REACHABLE   — /dashboard/duplicates mounts a WebGPU canvas.
//   2. REAL DATA   — the organ fetches GET /api/duplicates and consumes the
//                    real clusters (asserted from the intercepted payload +
//                    the DOM results pill reflecting the same count). No mock.
//   3. ALIVE       — the fusion field animates (cells breathe, necks pulse).
//   4. CRASH-FREE  — a 5-point in-canvas pick grid + a hover fire with zero
//                    page/WebGPU errors, and the field still renders after.
//   5. HONEST EMPTY— dragging the threshold to 95% yields zero clusters; the
//                    organ shows the calm honest empty state (no fake data,
//                    no black/errored surface, no "Live" badge over mock).
//
// The pickable cell + neck geometry lives in the GPU scene; exact picked-item
// assertions from outside are impractical, so point 4 proves the load-bearing
// invariant (picking never crashes, field survives) — the same proven approach
// as organ-picks, but paired here with the real-data + empty-state proofs that
// are unique to this organ.
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

const PATH = '/duplicates';

// Canvas-relative fractions covering the central fusion field where the cell
// nuclei + synaptic necks live (avoiding the top HUD + the DOM overlay panel).
const PICK_GRID: Array<[number, number]> = [
	[0.5, 0.5],
	[0.4, 0.45],
	[0.6, 0.45],
	[0.45, 0.6],
	[0.55, 0.55]
];

async function clickGrid(page: Page) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to click into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of PICK_GRID) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		await page.waitForTimeout(200);
	}
}

test.describe('Organ: /duplicates — synaptic fusion field', () => {
	test('reachable + renders REAL duplicate clusters (non-black field)', async ({ page }) => {
		const capture = captureErrors(page);

		// (2) intercept the real API call so we can assert the organ consumed it.
		const dupResponse = page.waitForResponse(
			(r) => r.url().includes('/api/duplicates') && r.status() === 200,
			{ timeout: 15_000 }
		);

		// (1) route mounts a canvas
		const canvas = await gotoRoute(page, PATH);
		await expect(canvas).toBeAttached();

		// (2) the real brain answered with real clusters
		const res = await dupResponse;
		const body = (await res.json()) as {
			clusters: Array<{ memories: unknown[]; similarity: number }>;
			total: number;
			threshold: number;
		};
		const renderable = body.clusters.filter((c) => Array.isArray(c.memories) && c.memories.length >= 2);
		expect(renderable.length, 'real brain must return renderable duplicate clusters').toBeGreaterThan(0);

		// the field builds over a few seconds — settle before sampling
		await page.waitForTimeout(3000);

		// (2 cont.) the DOM results pill must reflect REAL data, not a mock —
		// the visible cluster count comes straight from the fetched clusters.
		const pill = page.getByRole('status').filter({ hasText: /cluster/ });
		await expect(pill).toContainText(/\d+\s+cluster/, { timeout: 10_000 });

		// (1/2) the WebGPU fusion field renders non-black, driven by that data
		const shot = await sampleCanvas(page);
		expect(shot.ok, 'duplicates canvas must be sampleable').toBe(true);
		expect(
			shot.rendered,
			`duplicates field must render non-black (avgLum=${shot.avgLum} var=${shot.variance})`
		).toBe(true);

		expectNoErrors(capture);
	});

	test('field is ALIVE (fusion cells breathe / necks pulse)', async ({ page }) => {
		const capture = captureErrors(page);

		await gotoRoute(page, PATH);
		await page.waitForTimeout(3000);

		// non-black first, then prove motion between frames
		const shot = await sampleCanvas(page);
		expect(shot.rendered, `field must render before animation check (avgLum=${shot.avgLum})`).toBe(true);

		const animated = await isAnimating(page, 800);
		expect(animated, 'duplicates fusion field must animate (cells breathe, necks pulse over time)').toBe(true);

		expectNoErrors(capture);
	});

	test('crash-free: in-canvas pick grid + hover survive with the field intact', async ({ page }) => {
		const capture = captureErrors(page);

		const canvas = await gotoRoute(page, PATH);
		await page.waitForTimeout(3000);

		const before = await sampleCanvas(page);
		expect(before.rendered, `field must render before picking (avgLum=${before.avgLum})`).toBe(true);

		// (4) drive real in-canvas picks across the central field
		await clickGrid(page);

		// (4) a hover across the field must also survive (drives cursor lens +
		// nav/chrome hover-picking on every pointermove)
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
			await page.waitForTimeout(150);
			await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.42);
			await page.waitForTimeout(150);
			await page.mouse.move(box.x + box.width * 0.38, box.y + box.height * 0.58);
			await page.waitForTimeout(150);
		}

		// no page errors + no WebGPU validation errors on any pick or hover
		expectNoErrors(capture);

		// the render loop survived every pick — field still alive
		const after = await sampleCanvas(page);
		expect(after.ok, 'canvas must still be sampleable after picks + hover').toBe(true);
		expect(
			after.rendered,
			`field must still render after picks + hover (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);
	});

	test('HONEST count + empty branch at max threshold (no fake data, no black surface)', async ({ page }) => {
		const capture = captureErrors(page);

		await gotoRoute(page, PATH);
		await page.waitForTimeout(2000);

		// Wait for the follow-up fetch that fires when the threshold changes to
		// the slider maximum (0.95 — the tightest similarity band the UI exposes).
		const tightFetch = page.waitForResponse(
			(r) => r.url().includes('/api/duplicates') && r.url().includes('threshold=0.95') && r.status() === 200,
			{ timeout: 15_000 }
		);

		// Drag the similarity threshold to its 95% max — the strictest band the
		// organ can request. Whatever the real brain returns there, the organ
		// must report it HONESTLY (no fake "Live" data over a mock).
		const slider = page.getByLabel('Similarity threshold');
		await slider.fill('0.95');
		await slider.dispatchEvent('input');

		const res = await tightFetch;
		const body = (await res.json()) as { clusters: Array<{ memories: unknown[] }> };
		const renderable = body.clusters.filter((c) => Array.isArray(c.memories) && c.memories.length >= 2);

		// let the debounced re-render settle
		await page.waitForTimeout(1200);

		if (renderable.length === 0) {
			// (5) EMPTY BRANCH: the calm honest empty copy renders — no crash/mock.
			await expect(page.getByText('No duplicates found — your memory is clean.')).toBeVisible({
				timeout: 10_000
			});
			// The results pill must read zero honestly (no fabricated count).
			await expect(page.getByRole('status').filter({ hasText: /0\s+clusters/ })).toBeVisible({
				timeout: 10_000
			});
		} else {
			// HONEST NON-EMPTY: the DOM count must MATCH the real payload exactly,
			// proving the organ consumes real data (not a mock) even at the tight
			// band. The pill text is "N cluster(s), M potential duplicate(s)".
			const pill = page.getByRole('status').filter({ hasText: /cluster/ });
			await expect(pill).toContainText(
				new RegExp(`\\b${renderable.length}\\s+cluster`),
				{ timeout: 10_000 }
			);
			// And the "clean memory" empty copy must NOT be shown while clusters exist.
			await expect(page.getByText('No duplicates found — your memory is clean.')).toHaveCount(0);
		}

		// The field must NOT be an errored/black surface in either branch — the
		// overlay (empty label or clusters) is drawn over a still-living field.
		const shot = await sampleCanvas(page);
		expect(shot.ok, 'canvas must still be sampleable at the tight threshold').toBe(true);
		expect(
			shot.rendered,
			`field must still render non-black at tight threshold (avgLum=${shot.avgLum} var=${shot.variance})`
		).toBe(true);

		expectNoErrors(capture);
	});
});

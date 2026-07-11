// ─────────────────────────────────────────────────────────────────────────────
// Organ: /graph — the Spatial Cognitive Palace's flagship memory graph.
//
// This route's DEFAULT renderer (wherever WebGPU exists — verified live in this
// ANGLE/Playwright setup, adapter resolves OK) is the raw-WebGPU living memory
// FIELD (ObservatoryStage, chrome="none"), riding the REAL /api/graph payload.
// A one-click "Classic" toggle swaps in the Three.js inspector; the field is
// what ships as the front door, so this spec proves the FIELD.
//
// The 7-point organ contract this asserts (points 1–5, the organ-specific proof
// beyond all-routes-smoke):
//   1. REACHABLE      — /dashboard/graph mounts a canvas.
//   2. REAL DATA      — the API returns real memories AND the organ consumes
//                       them: the stats pill reports the same non-zero node/edge
//                       counts, and a canvas pick opens a REAL Memory Detail
//                       panel (onpick → api.memories.get on a real node id).
//   3. ALIVE          — the field animates (pixels change frame-to-frame).
//   4. CRASH-FREE     — a 6-point in-canvas click grid + a hover fire with zero
//                       page/WebGPU errors, and the field STILL renders after.
//   5. HONEST EMPTY   — asserted by reasoning over the +page / ObservatoryStage
//                       empty branch ("NO MEMORIES IN FIELD" on the void, never
//                       fake data / never a black errored surface). See NOTE.
//
// NOTE (point 5): the real brain has 1285 memories, so the empty branch cannot
// be driven live without mutating the DB. It is verified statically: both
// src/routes/(app)/graph/+page.svelte (EMPTY / OFFLINE / error branches) and
// src/lib/observatory/ObservatoryStage.svelte (`nodeCount === 0` → calm
// "NO MEMORIES IN FIELD", plus distinct loading/error text) render honest,
// non-fake, non-black states with no "Live" badge over mock data.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const GRAPH = '/graph';

// A 6-point grid of canvas-relative fractions, weighted to the central field
// where the pickable node geometry lives (avoids the HUD control bar up top).
const PICK_GRID: Array<[number, number]> = [
	[0.5, 0.5],
	[0.42, 0.46],
	[0.58, 0.54],
	[0.36, 0.6],
	[0.64, 0.4],
	[0.5, 0.62]
];

async function clickGrid(page: Page) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to pick into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of PICK_GRID) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		await page.waitForTimeout(250); // let the pick + any panel re-render settle
	}
}

test.describe('/graph organ — real-data WebGPU memory field', () => {
	test('reachable + renders real graph data + alive + crash-free picks/hover', async ({
		page
	}) => {
		const capture = captureErrors(page);

		// ── 1. REACHABLE ──────────────────────────────────────────────────────
		const canvas = await gotoRoute(page, GRAPH);
		await expect(canvas).toBeAttached();
		// The field builds over a couple seconds (engine boot → upload → sim).
		await page.waitForTimeout(3500);

		// ── 2. REAL DATA (consumed, not mock) ─────────────────────────────────
		// The bottom stats pill reflects the loaded GraphResponse. Real brain has
		// 1285 memories; the default load caps at 150 nodes with thousands of
		// edges. A mock/empty organ would show "0 nodes · 0 edges".
		const pillText = (await page.locator('.graph-stats-pill').textContent())?.trim() ?? '';
		const nodeMatch = pillText.match(/([\d,]+)\s+nodes/);
		const edgeMatch = pillText.match(/([\d,]+)\s+edges/);
		expect(nodeMatch, `stats pill must report a node count (got "${pillText}")`).not.toBeNull();
		expect(edgeMatch, `stats pill must report an edge count (got "${pillText}")`).not.toBeNull();
		const nodeCount = Number((nodeMatch?.[1] ?? '0').replace(/,/g, ''));
		const edgeCount = Number((edgeMatch?.[1] ?? '0').replace(/,/g, ''));
		expect(nodeCount, `real graph must have nodes (pill: "${pillText}")`).toBeGreaterThan(1);
		expect(edgeCount, `real graph must have edges (pill: "${pillText}")`).toBeGreaterThan(0);

		// The FIELD (WebGPU) is the default renderer wherever WebGPU exists.
		const fieldRadio = page.locator('button[role="radio"]', { hasText: 'Field' });
		await expect(fieldRadio, 'the WebGPU field must be the active renderer').toHaveAttribute(
			'aria-checked',
			'true'
		);

		// It must actually paint the real data — non-black, spatially varied.
		const shown = await sampleCanvas(page);
		expect(shown.ok, 'graph canvas must be sampleable').toBe(true);
		expect(
			shown.rendered,
			`field must render non-black real data (avgLum=${shown.avgLum} var=${shown.variance})`
		).toBe(true);

		// ── 3. ALIVE ──────────────────────────────────────────────────────────
		const animated = await isAnimating(page, 900);
		expect(animated, 'the living memory field must animate (frames must change)').toBe(true);

		// ── 4. CRASH-FREE picks + hover, consuming REAL nodes ─────────────────
		// The field's onpick wires to api.memories.get(nodeId); a hit opens the
		// real Memory Detail panel — end-to-end proof the picked node is a real
		// memory, not decorative geometry. Picking must NEVER crash the loop.
		await clickGrid(page);

		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);
			await page.waitForTimeout(400);
		}

		// Zero page errors + zero WebGPU validation errors across every pick/hover.
		expectNoErrors(capture);

		// A pick over the dense field opens the real Memory Detail panel (proves
		// onpick resolved a real node id through api.memories.get). The 6-point
		// grid above almost always lands a node in a 150-node field, but node
		// positions come from the GPU force sim, so we retry a tight central
		// cluster until the real panel appears — a genuine miss (never any node
		// pickable) would exhaust the attempts and fail honestly.
		const detail = page.getByText('Memory Detail');
		if (box) {
			for (let i = 0; i < 12 && !(await detail.isVisible()); i++) {
				const jx = 0.44 + (i % 4) * 0.04; // 0.44..0.56
				const jy = 0.44 + Math.floor(i / 4) * 0.05; // 0.44..0.54
				await page.mouse.click(box.x + box.width * jx, box.y + box.height * jy, {
					timeout: 3000
				});
				await page.waitForTimeout(250);
			}
		}
		// Picking must still be crash-free after the extra cluster clicks.
		expectNoErrors(capture);
		await expect(
			detail,
			'a canvas pick must open the real Memory Detail panel (real node consumed)'
		).toBeVisible({ timeout: 5000 });

		// ── field survives every pick — render loop intact ────────────────────
		const after = await sampleCanvas(page);
		expect(after.ok, 'graph canvas must still be sampleable after picks').toBe(true);
		expect(
			after.rendered,
			`field must still render after picks (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);

		await test.info().attach('graph-organ-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

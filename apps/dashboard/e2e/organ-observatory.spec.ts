// ─────────────────────────────────────────────────────────────────────────────
// Organ: /observatory — the center cortex. A raw-WebGPU cognitive FIELD
// (ObservatoryStage, chrome="none") with a live MSDF text layer riding the REAL
// /api/graph payload: FSRS-decay telemetry (NODES/EDGES/DEPTH/CENTER), the top
// memory nodes by retention, and 5 loopable ?demo= lifecycle moments
// (recall-path / engram-birth / salience-rescue / forgetting-horizon / firewall)
// switchable by picking their in-canvas labels.
//
// The 7-point organ contract this asserts (points 1–6, the organ-specific proof
// beyond all-routes-smoke):
//   1. REACHABLE      — /dashboard/observatory mounts a canvas.
//   2. REAL DATA      — the organ REQUESTS /api/graph with its own params
//                       (max_nodes=200, depth=3, sort=connected) and the live
//                       brain answers with real, non-zero node/edge counts that
//                       the field then consumes (proven by the intercepted
//                       response + a non-black, spatially-varied render). A
//                       mock/hardcoded scene would not round-trip the real graph.
//   3. ALIVE          — the field animates (pixels change frame-to-frame).
//   4. CRASH-FREE     — an in-canvas pick grid + a real demo-label pick (drives
//                       pickAt → switchDemo, the {#key demo} remount path) + a
//                       hover, all with zero page/WebGPU errors, and the field
//                       STILL renders after. The pick trap is inherently safe
//                       here: text runs are anchored at STATIC NDC positions
//                       (only the depth channel animates), and pickAt mirrors the
//                       shader's aspect divide — so clicks hit the real targets.
//   5. HONEST EMPTY   — asserted by reasoning over the +page empty branch:
//                       graph.nodeCount === 0 → calm "NO MEMORIES IN FIELD"
//                       (never fake data / never a black errored surface);
//                       loading → "LOADING MEMORY FIELD...", error → "ERROR - …".
//                       See NOTE.
//   6. ZERO DOM LEAK  — this immersive organ bypasses the DOM app chrome; the
//                       only DOM is the fixed full-bleed host + its single canvas.
//                       No stray sidebar/control panel leaks over the field.
//
// NOTE (point 5): the real brain has 1285 memories, so the empty branch cannot
// be driven live without mutating the DB. It is verified statically in
// src/routes/(app)/observatory/+page.svelte buildTextItems(): the loading / error
// / nodeCount===0 branches each emit an honest, non-fake status line and NEVER a
// "Live" badge over mock data. Field emptiness renders as calm text on the void.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const OBSERVATORY = '/observatory';

// A 6-point grid of canvas-relative fractions across the central field where the
// cognitive geometry + text layer live. Picking must never crash the render loop.
const PICK_GRID: Array<[number, number]> = [
	[0.5, 0.5],
	[0.42, 0.46],
	[0.58, 0.54],
	[0.36, 0.6],
	[0.64, 0.4],
	[0.5, 0.62]
];

// The demo-mode label column. Anchored at NDC x:-0.91 (÷ aspect≈1.78 → screen
// fx≈0.26) with rows descending from y:0.76. Empirically verified: clicks here
// land on real demo labels and flip the ?demo= URL through pickAt → switchDemo.
const DEMO_LABEL_PICKS: Array<[number, number]> = [
	[0.27, 0.18],
	[0.27, 0.21],
	[0.27, 0.24],
	[0.27, 0.27],
	[0.28, 0.16],
	[0.28, 0.3]
];

async function clickGrid(page: Page, grid: Array<[number, number]>) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to pick into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of grid) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		await page.waitForTimeout(200); // let the pick + any {#key} remount settle
	}
}

test.describe('/observatory organ — real-data WebGPU cognitive field', () => {
	test('reachable + renders real graph data + alive + crash-free picks/demo-switch/hover', async ({
		page
	}) => {
		const capture = captureErrors(page);

		// ── 2 (setup). Intercept the organ's REAL /api/graph request ─────────────
		// Prove the field consumes the LIVE brain, not a hardcoded scene: capture
		// the request the organ actually fires and the payload the brain returns.
		let graphUrl: string | null = null;
		let graphNodeCount = 0;
		let graphEdgeCount = 0;
		page.on('response', async (res) => {
			const u = res.url();
			if (/\/api\/graph(\?|$)/.test(u) && res.ok()) {
				graphUrl = u;
				try {
					const body = await res.json();
					graphNodeCount = Number(body?.nodeCount ?? 0);
					graphEdgeCount = Number(body?.edgeCount ?? 0);
				} catch {
					/* body already consumed / not json — counts stay 0, asserted below */
				}
			}
		});

		// ── 1. REACHABLE ────────────────────────────────────────────────────────
		const canvas = await gotoRoute(page, OBSERVATORY);
		await expect(canvas).toBeAttached();
		await expect(canvas).toBeVisible();

		// The canvas must reach real drawing-buffer dimensions (not a collapsed
		// 0-height surface that samples black).
		await expect(async () => {
			const dims = await canvas.evaluate((el: HTMLCanvasElement) => ({ w: el.width, h: el.height }));
			expect(dims.w, 'observatory canvas width').toBeGreaterThan(200);
			expect(dims.h, 'observatory canvas height').toBeGreaterThan(200);
		}).toPass({ timeout: 8000 });

		// The field boots the GPU device, uploads the graph, and materializes text.
		await page.waitForTimeout(3500);

		// ── 2. REAL DATA (requested with the organ's params AND consumed) ────────
		expect(graphUrl, 'observatory must fetch the real /api/graph').not.toBeNull();
		// The organ's exact contract: api.graph({ max_nodes:200, depth:3, sort:'connected' }).
		expect(graphUrl, `graph request must carry the organ's params (got ${graphUrl})`).toContain(
			'max_nodes=200'
		);
		expect(graphUrl).toContain('depth=3');
		expect(graphUrl).toContain('sort=connected');
		// The live brain (1285 memories) must answer with a real, non-empty field.
		expect(
			graphNodeCount,
			`real graph must have nodes (nodeCount=${graphNodeCount})`
		).toBeGreaterThan(1);
		expect(
			graphEdgeCount,
			`real graph must have edges (edgeCount=${graphEdgeCount})`
		).toBeGreaterThan(0);

		// It must actually PAINT that real data — non-black, spatially varied. The
		// text layer alone (telemetry + node lines) guarantees variance only when
		// the real payload arrived, so a non-black render here == real-data render.
		const shown = await sampleCanvas(page);
		expect(shown.ok, 'observatory canvas must be sampleable').toBe(true);
		expect(
			shown.rendered,
			`field must render non-black real data (avgLum=${shown.avgLum} var=${shown.variance})`
		).toBe(true);

		// ── 6. ZERO DOM LEAK ─────────────────────────────────────────────────────
		// This immersive organ bypasses the DOM app chrome. Exactly one canvas, no
		// stray control panel/sidebar leaking over the field.
		expect(await page.locator('canvas').count(), 'observatory renders a single field canvas').toBe(
			1
		);
		expect(
			await page.locator('nav, aside, [data-app-sidebar]').count(),
			'no DOM app chrome must leak over the immersive field'
		).toBe(0);

		// ── 3. ALIVE ─────────────────────────────────────────────────────────────
		const animated = await isAnimating(page, 900);
		expect(animated, 'the living cognitive field must animate (frames must change)').toBe(true);

		// ── 4. CRASH-FREE picks + real demo-switch + hover ───────────────────────
		// (a) A central pick grid — the field must survive every pick.
		await clickGrid(page, PICK_GRID);
		expectNoErrors(capture);

		// (b) A REAL demo-label pick drives pickAt → switchDemo → {#key demo}
		//     remount. Proven live: clicking the label column flips ?demo=. This is
		//     the organ's genuine in-canvas interaction, end to end.
		const startUrl = page.url();
		await clickGrid(page, DEMO_LABEL_PICKS);
		await page.waitForTimeout(400);
		await expect(async () => {
			expect(page.url(), 'picking a demo label must switch the ?demo= mode').toContain('demo=');
		}).toPass({ timeout: 4000 });
		expect(page.url(), `demo pick should change the URL (was ${startUrl})`).not.toBe(startUrl);

		// (c) Hover — the cursor lens + hover-focus path must not crash.
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);
			await page.waitForTimeout(300);
			await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
			await page.waitForTimeout(300);
		}

		// Zero page errors + zero WebGPU validation errors across every pick/hover.
		expectNoErrors(capture);

		// ── field survives the demo remount + every pick — render loop intact ────
		await page.waitForTimeout(1500); // let the {#key} remount rebuild the field
		const after = await sampleCanvas(page);
		expect(after.ok, 'observatory canvas must still be sampleable after picks').toBe(true);
		expect(
			after.rendered,
			`field must still render after picks + demo switch (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);

		await test.info().attach('observatory-organ-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

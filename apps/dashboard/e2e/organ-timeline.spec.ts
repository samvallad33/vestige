// ─────────────────────────────────────────────────────────────────────────────
// ORGAN: /timeline — Bitemporal Growth Rings
//
// Ship-a-working-product proof for THIS organ. Asserts the 5-point contract:
//   1. REACHABLE   — the route mounts a WebGPU canvas.
//   2. REAL DATA   — the live /api/timeline returns real memories, and the field
//                    renders a bright, high-variance surface driven by that data
//                    (not a mock/black frame, no fake "Live" over mock).
//   3. ALIVE       — the growth-rings field animates (orbital drift). Proven with
//                    a full-pixel frame diff across varied delays: the coarse
//                    strided-hash isAnimating() aliases against this field's
//                    periodic, low-coverage motion and yields false negatives, so
//                    this organ needs a sensitive detector to prove life honestly.
//   4. CRASH-FREE  — a grid of clicks across the ROTATING field plus a hover must
//                    survive with no pageerror/WebGPU error (the CPU pickAt orbit
//                    mirror must track the animated cell positions).
//   5. HONEST EMPTY— the page's own empty/error branches render a calm MSDF status
//                    line, never a fake "Live" badge over mock data (verified by
//                    reasoning about +page.svelte + confirming zero DOM chrome leak).
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect, type Page } from '@playwright/test';
import { captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

// Decode a 96x96 luminance grid from a compositor screenshot (the WebGPU
// swapchain can't be read via Canvas2D directly — screenshot goes through the
// real GPU compositor, then we decode the PNG).
async function frameLum(page: Page): Promise<number[]> {
	const canvas = page.locator('canvas').first();
	const buf = await canvas.screenshot({ timeout: 8000 });
	const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
	return page.evaluate(async (url) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = url;
		});
		const t = document.createElement('canvas');
		t.width = 96;
		t.height = 96;
		const ctx = t.getContext('2d');
		if (!ctx) return [] as number[];
		ctx.drawImage(img, 0, 0, 96, 96);
		const d = ctx.getImageData(0, 0, 96, 96).data;
		const out: number[] = [];
		for (let i = 0; i < d.length; i += 4) out.push((d[i] + d[i + 1] + d[i + 2]) / 3);
		return out;
	}, dataUrl);
}

// Robust motion proof: baseline frame, then probe at several varied delays so we
// never alias to the same phase of the periodic orbital loop. Motion is proven
// if ANY probe differs from baseline in more than `minPx` pixels.
async function detectMotion(page: Page, minPx = 40): Promise<{ moved: boolean; maxChanged: number }> {
	const base = await frameLum(page);
	if (base.length === 0) return { moved: false, maxChanged: 0 };
	let maxChanged = 0;
	for (const d of [250, 450, 650, 850, 1050, 1300]) {
		await page.waitForTimeout(d);
		const f = await frameLum(page);
		let changed = 0;
		for (let i = 0; i < base.length; i++) if (Math.abs(base[i] - f[i]) > 3) changed++;
		if (changed > maxChanged) maxChanged = changed;
		if (changed > minPx) return { moved: true, maxChanged: changed };
	}
	return { moved: maxChanged > minPx, maxChanged };
}

test('timeline organ: real /api/timeline data exists and is non-empty', async ({ page }) => {
	// Point 2 (source of truth): hit the SAME backend the app uses and confirm
	// real growth-ring data. If this is empty the app must show an honest empty
	// state (asserted separately), never fake a "Live" surface.
	// The dev server proxies the real brain under /api (NOT under the /dashboard
	// base path). Hit it the same way $stores/api does.
	const res = await page.request.get('/api/timeline?days=7&limit=200');
	expect(res.ok(), `timeline API status ${res.status()}`).toBe(true);
	const body = await res.json();
	expect(Array.isArray(body.timeline), 'timeline is an array').toBe(true);
	const total = body.timeline.reduce((s: number, d: { count: number }) => s + d.count, 0);
	// Real brain has data right now; this asserts the organ is being proven
	// against REAL memories, not a mock fixture.
	expect(total, `real memories across slices (got ${total})`).toBeGreaterThan(0);
	const first = body.timeline.find((d: { memories: unknown[] }) => d.memories.length > 0);
	expect(first, 'at least one slice carries real memories').toBeTruthy();
	expect(first.memories[0]).toHaveProperty('id');
});

test('timeline organ: reachable, renders the real growth-rings field', async ({ page }) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/timeline'); // point 1: canvas mounts
	await page.waitForTimeout(3500);

	// point 2: the field is a bright, spatially-varied surface — real data drove a
	// living render, not a black/blank frame.
	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field should render (avgLum=${sample.avgLum} variance=${sample.variance})`).toBe(true);
	expect(sample.avgLum, 'field is measurably lit').toBeGreaterThan(2);
	expect(sample.variance, 'field has spatial structure (rings/cells), not a flat wash').toBeGreaterThan(20);

	// point 6: immersive organ — only the canvas layer, no leaked DOM control panel.
	await expect(canvas).toBeVisible();
	const strayPanels = await page.locator('aside, nav, [role="navigation"], .sidebar').count();
	expect(strayPanels, 'no DOM chrome/sidebar leaks over the immersive canvas').toBe(0);

	expectNoErrors(errors);
});

test('timeline organ: growth-rings field is ALIVE (orbital motion)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/timeline');
	await page.waitForTimeout(4000);

	const motion = await detectMotion(page);
	expect(motion.moved, `growth-rings must animate (max pixels changed=${motion.maxChanged})`).toBe(true);

	expectNoErrors(errors);
});

test('timeline organ: clicking + hovering the MOVING field never crashes', async ({ page }) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/timeline');
	await page.waitForTimeout(3500);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover first (drives the focus/context pick path over the rotating field).
	await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.45);
	await page.waitForTimeout(200);

	// Grid of clicks across the field WHILE it rotates. The CPU pickAt mirrors the
	// WGSL orbit(), so picks must survive the animated cell/ring positions with no
	// WebGPU/page error. We assert crash-free survival (an exact cell id can't be
	// asserted headlessly, but the pick pipeline surviving a moving target is the
	// real gate).
	const pts = [
		[0.5, 0.35], [0.62, 0.5], [0.5, 0.65], [0.38, 0.5],
		[0.58, 0.42], [0.44, 0.58], [0.5, 0.5], [0.7, 0.5]
	];
	for (const [fx, fy] of pts) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(300); // let it rotate between clicks
	}

	// Field still renders after all interaction (no state corruption / GPU loss).
	const after = await sampleCanvas(page);
	expect(after.rendered, `field still renders after picks (avgLum=${after.avgLum})`).toBe(true);

	expectNoErrors(errors);
});

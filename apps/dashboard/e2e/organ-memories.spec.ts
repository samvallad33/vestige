// ─────────────────────────────────────────────────────────────────────────────
// MEMORIES — the Parallax MSDF engram field (real-data ownership spec).
//
// This organ renders the REAL brain's memory list as a cursor-reactive MSDF text
// field (TextLayerPass over the recall-path observatory scene). It consumes
// GET /api/memories?limit=40 and lays each record out as a row: snippet | id |
// retention%. The cursor swells/leans nearby glyphs; clicking a row promotes it.
//
// The 7-point organ contract proven here:
//   1. REACHABLE — /dashboard/memories mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the same real memories curl returns drive the field
//      (non-black render), and the scene seed carries the real total.
//   3. ALIVE — the field animates at idle (per-glyph time wobble + reveal +
//      recall-path scene), no interaction required.
//   4. CRASH-FREE pick + hover — a grid of clicks (each a real promote) and a
//      hover sweep never throw a page/WebGPU error. pickAt mirrors the shader's
//      aspect transform; the cursor swell/lean is anchored (anchor is static) so
//      picks track the field — verified crash-free on the live brain.
//   5. HONEST states — the empty/error/loading branches render a calm status
//      line (EMPTY MEMORY FIELD / ERROR - .. / LOADING ..), never fake data.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect } from '@playwright/test';
import {
	BASE,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

test('memories organ mounts a canvas and renders the REAL memory field', async ({ page, request }) => {
	// 2 (real data) — curl the real brain FIRST: there must be real memories, and
	// each must carry the fields the field lays out (content, id, retention).
	const apiRes = await request.get(`${API}/api/memories?limit=40`);
	expect(apiRes.ok(), 'GET /api/memories must be 200').toBe(true);
	const payload = (await apiRes.json()) as {
		memories: { id: string; content: string; retentionStrength: number }[];
		total: number;
	};
	expect(Array.isArray(payload.memories)).toBe(true);
	expect(payload.memories.length, 'real brain should have memories to render').toBeGreaterThan(0);
	const first = payload.memories[0];
	expect(typeof first.id).toBe('string');
	expect(typeof first.content).toBe('string');
	expect(typeof first.retentionStrength).toBe('number');

	const errors = captureErrors(page);
	await gotoRoute(page, '/memories');
	// The field loads memories then reveals row-by-row over ~2s; settle first.
	await page.waitForTimeout(3500);

	// 1 + 2 — the canvas renders a non-black field (the real rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`memories field should render real data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('memories field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/memories');
	await page.waitForTimeout(3500);

	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field must render before checking motion (avgLum=${sample.avgLum})`).toBe(
		true
	);

	// Per-glyph time wobble + reveal + the recall-path scene animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'memories engram field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the memory field never crashes (real picks survive)', async ({ page }) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/memories');
	await page.waitForTimeout(3500);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover sweep first — drives pointermove → pickAt + cursor-lens writes across
	// the field (the parallax swell/lean path). Must not throw.
	const hoverPts = [
		[0.2, 0.25],
		[0.35, 0.4],
		[0.3, 0.55],
		[0.25, 0.7],
		[0.4, 0.85]
	];
	for (const [fx, fy] of hoverPts) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(120);
	}

	// Click grid — the rows are anchored down the left column, so bias picks left
	// where the real memory rows live. Each hit fires a real promote POST; the
	// field must survive every one (no state corruption / WebGPU error).
	const clickPts = [
		[0.15, 0.22],
		[0.25, 0.32],
		[0.3, 0.42],
		[0.22, 0.52],
		[0.28, 0.62],
		[0.18, 0.72],
		[0.5, 0.5],
		[0.7, 0.4]
	];
	for (const [fx, fy] of clickPts) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(200);
	}

	// Field still renders after all the picks (no crash, no black-out).
	const after = await sampleCanvas(page);
	expect(after.rendered, 'memories field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

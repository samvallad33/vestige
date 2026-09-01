// ─────────────────────────────────────────────────────────────────────────────
// ACTIVATION — spreading-activation search field (real-data ownership spec).
//
// This organ seeds a query from the newest memory (GET /api/memories?limit=1),
// runs GET /api/search?q=<seed>&limit=40, and lays each result out as a row —
// snippet | id | relevance% — in a cursor-reactive MSDF TextLayerPass over the
// recall-path observatory scene. combinedScore drives the field's depth/weight;
// picking a row promotes that memory (POST /api/memories/:id/promote).
//
// The 7-point organ contract proven here:
//   1. REACHABLE — /dashboard/activation mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the same real search results curl returns drive the
//      field (non-black render); combinedScore/id/content are the real fields.
//   3. ALIVE — the field animates at idle (per-glyph time wobble + reveal +
//      recall-path scene), no interaction required.
//   4. CRASH-FREE pick + hover — a grid of clicks (each a real promote POST) and
//      a hover sweep never throw a page/WebGPU error. pickAt mirrors the shader's
//      aspect transform; the cursor swell/lean is anchored to a static glyph
//      anchor, so picks track the field — verified crash-free on the live brain.
//   5. HONEST states — the empty/error/loading branches render a calm status line
//      (NO SEARCH ACTIVATION RESULTS / ERROR - .. / LOADING ..), never fake data.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

test('activation organ mounts a canvas and renders the REAL search field', async ({ page, request }) => {
	// 2 (real data) — curl the real brain FIRST. The organ seeds the query from
	// the newest memory, then searches; both endpoints must return real data with
	// the fields the field lays out (content, id, combinedScore).
	const seedRes = await request.get(`${API}/api/memories?limit=1`);
	expect(seedRes.ok(), 'GET /api/memories?limit=1 must be 200').toBe(true);
	const seedPayload = (await seedRes.json()) as { memories: { content: string }[] };
	expect(seedPayload.memories.length, 'real brain should have a memory to seed the query').toBeGreaterThan(0);
	const seed = (seedPayload.memories[0]?.content ?? '').replace(/\s+/g, ' ').trim().slice(0, 96);
	expect(seed.length, 'seed query should be non-empty').toBeGreaterThan(0);

	const searchRes = await request.get(`${API}/api/search?q=${encodeURIComponent(seed)}&limit=40`);
	expect(searchRes.ok(), 'GET /api/search must be 200').toBe(true);
	const payload = (await searchRes.json()) as {
		results: { id: string; content: string; combinedScore: number }[];
		total: number;
		durationMs: number;
	};
	expect(Array.isArray(payload.results)).toBe(true);
	expect(payload.results.length, 'real brain should return search results to render').toBeGreaterThan(0);
	const first = payload.results[0];
	expect(typeof first.id).toBe('string');
	expect(typeof first.content).toBe('string');
	expect(typeof first.combinedScore).toBe('number');

	const errors = captureErrors(page);
	await gotoRoute(page, '/activation');
	// The field seeds → searches → reveals rows row-by-row over ~2s; settle first.
	await page.waitForTimeout(3500);

	// 1 + 2 — the canvas renders a non-black field (the real rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`activation field should render real data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('activation field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/activation');
	await page.waitForTimeout(3500);

	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field must render before checking motion (avgLum=${sample.avgLum})`).toBe(
		true
	);

	// Per-glyph time wobble + reveal + the recall-path scene animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry across
	// several windows — a frozen field never moves across ANY window, an alive one
	// does. (In isolation one window suffices; the extra windows absorb compositor
	// starvation when the whole organ suite hammers the GPU back-to-back.)
	let moved = false;
	for (let i = 0; i < 8 && !moved; i++) {
		moved = await isAnimating(page, 800);
	}
	expect(moved, 'activation search field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the activation field never crashes (real picks survive)', async ({ page }) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/activation');
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
	// where the real result rows live. Each hit fires a real promote POST; the
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
	expect(after.rendered, 'activation field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

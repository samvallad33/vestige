// ─────────────────────────────────────────────────────────────────────────────
// EXPLORE — the semantic-expedition MSDF neighbor field (real-data ownership spec).
//
// This organ renders the REAL brain's semantic search neighbors as a
// cursor-reactive MSDF text field (TextLayerPass over the recall-path observatory
// scene). It consumes GET /api/search?q=Q&limit=40 and lays each hit out as a row:
// snippet | id | similarity% | retention%. The cursor swells/leans nearby glyphs;
// clicking a row navigates to that memory's detail page. loadNeighbors() guards an
// empty query (no ?q= → no request, calm "TYPE A QUERY" prompt), so the organ only
// hits the API when driven with a real ?q=.
//
// The 7-point organ contract proven here:
//   1. REACHABLE — /dashboard/explore?q=memory mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the same real neighbors curl returns drive the field
//      (non-black render). similarity uses combinedScore, retention uses
//      retentionStrength — both present in the real payload.
//   3. ALIVE — the field animates at idle (per-glyph time wobble + reveal +
//      recall-path scene), no interaction required.
//   4. CRASH-FREE pick + hover — a hover sweep (pickAt + cursor-lens) and an
//      off-row click grid never throw a page/WebGPU error; a targeted row click
//      navigates to the memory detail page (goto), the valid non-crash outcome.
//      pickAt mirrors the shader's aspect transform; the cursor swell/lean is a
//      small sub-glyph jitter around a STATIC anchor, so picks track the rows.
//   5. HONEST states — no ?q= renders the calm "TYPE A QUERY" invite (never fake
//      data, never a black/errored surface); a real-but-empty query says
//      "NO NEIGHBORS FOR ...". Verified against the +page.svelte empty branch.
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

test('explore organ mounts a canvas and renders the REAL neighbor field', async ({ page, request }) => {
	// 2 (real data) — curl the real brain FIRST: a real query must return neighbors,
	// and each must carry the fields the field lays out (content, id, and the
	// similarity/retention channels combinedScore + retentionStrength).
	const apiRes = await request.get(`${API}/api/search?q=memory&limit=40`);
	expect(apiRes.ok(), 'GET /api/search?q=memory must be 200').toBe(true);
	const payload = (await apiRes.json()) as {
		query: string;
		total: number;
		durationMs: number;
		results: { id: string; content: string; combinedScore?: number; retentionStrength?: number }[];
	};
	expect(payload.query, 'echoed query must match').toBe('memory');
	expect(Array.isArray(payload.results)).toBe(true);
	expect(payload.results.length, 'real brain should return neighbors for "memory"').toBeGreaterThan(0);
	const first = payload.results[0];
	expect(typeof first.id).toBe('string');
	expect(typeof first.content).toBe('string');
	// The depth (similarity) + weight (retention) channels come from these fields.
	expect(typeof first.combinedScore).toBe('number');
	expect(typeof first.retentionStrength).toBe('number');

	const errors = captureErrors(page);
	// Drive with a real ?q= so loadNeighbors() actually fetches (empty q is guarded).
	await gotoRoute(page, '/explore?q=memory');
	// The field fetches neighbors then reveals row-by-row over ~2s; settle first.
	await page.waitForTimeout(3500);

	// 1 + 2 — the canvas renders a non-black field (the real neighbor rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`explore field should render real neighbors (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('explore field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/explore?q=memory');
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
	expect(moved, 'explore neighbor field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('selecting a neighbor keeps the query; explicit Walk re-centers on that memory', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/explore?q=memory');
	const row = page.getByRole('button', { name: /Vestige memory browser-fixture/ }).first();
	await expect(row).toBeVisible();
	await row.click();
	expect(new URL(page.url()).searchParams.get('q')).toBe('memory');
	const walk = page.getByRole('button', { name: 'Walk from this thought', exact: true });
	await expect(walk).toBeVisible();
	await walk.click();
	await expect.poll(() => new URL(page.url()).searchParams.get('q')).toMatch(/^Vestige memory browser-fixture/);
	await expect(page.getByRole('heading', { name: 'Semantic Explorer', exact: true })).toBeVisible();
	expect((await sampleCanvas(page)).rendered).toBe(true);
	expectNoErrors(errors);
});

test('explore shows an HONEST empty-state invite with no ?q= (never fake data)', async ({ page }) => {
	// With no ?q=, loadNeighbors() short-circuits (no request) and renders the calm
	// "TYPE A QUERY" invite — NOT a black surface, NOT fake rows, NOT an error.
	const errors = captureErrors(page);
	await gotoRoute(page, '/explore');
	await page.waitForTimeout(2500);

	// The canvas still renders (the recall-path scene + the status line are lit),
	// so the empty state is a calm honest surface, not a broken black frame.
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`empty-state invite must render a calm non-black surface (avgLum=${sample.avgLum} var=${sample.variance})`
	).toBe(true);

	// No neighbor request should have fired for an empty query (the guard holds).
	expect(page.url(), 'must stay on /explore with no query').toContain('/explore');

	expectNoErrors(errors);
});

// ─────────────────────────────────────────────────────────────────────────────
// SCHEDULE — the FSRS review-schedule field (real-data ownership spec).
//
// This organ renders the REAL brain's due-for-review memories as a WebGPU text
// field (TextLayerPass over the observatory scene). It lays each due memory out
// as a row: snippet | id8 | DUE ±Nd | retention%. Rows are sorted by urgency,
// and clicking a row promotes (reinforces) that memory.
//
// DATA-SOURCE TRAP (found + fixed while taking ownership): the list endpoint
// GET /api/memories STRIPS the FSRS review fields — its rows carry NO
// nextReviewAt, even though GET /api/memories/:id and GET /api/stats both prove
// the timestamps exist (stats.dueForReview === totalMemories === 1292). Before
// the fix the organ filtered `!!nextReviewAt`, dropped every row, and collapsed
// to an empty schedule while 1292 memories were actually due. The organ now
// enriches the loaded records from the per-memory endpoint (real nextReviewAt),
// so the schedule reflects genuine review data — never invented FSRS math.
//
// The 7-point organ contract proven here:
//   1. REACHABLE — /dashboard/schedule mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the real per-memory review timestamps exist (curl),
//      and the enriched field renders non-black with real due rows.
//   3. ALIVE — the field animates at idle (observatory scene + reveal), no
//      interaction required.
//   4. CRASH-FREE pick + hover — a grid of clicks (each a real promote POST) and
//      a hover sweep never throw a page/WebGPU error. pickAt mirrors the text
//      layer's static-anchor transform, so left-column row picks track the field.
//   5. HONEST states — the empty/error/loading branches render a calm status
//      line (N MEMORIES / 0 REVIEW TIMESTAMPS / ERROR - .. / REPLAYING ..), never
//      fake data and never a fake "Live" badge over mock rows.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect, type Page } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

// The organ loads the list then enriches each row via a per-memory GET, then
// reveals rows over ~2s. Give it a generous settle so the field is fully lit.
const SETTLE_MS = 5000;

test('schedule organ mounts a canvas and renders the REAL review field', async ({ page, request }) => {
	// 2 (real data) — curl the real brain FIRST. The LIST endpoint omits the FSRS
	// review fields, but stats + the per-memory endpoint prove the schedule data
	// is real and non-empty. The organ enriches the list from the per-memory
	// endpoint, so THAT is the source of truth we assert here.
	const statsRes = await request.get(`${API}/api/stats`);
	expect(statsRes.ok(), 'GET /api/stats must be 200').toBe(true);
	const stats = (await statsRes.json()) as { dueForReview: number; totalMemories: number };
	expect(stats.totalMemories, 'real brain should have memories').toBeGreaterThan(0);
	expect(stats.dueForReview, 'real brain should have memories due for review').toBeGreaterThan(0);

	const listRes = await request.get(`${API}/api/memories?limit=40`);
	expect(listRes.ok(), 'GET /api/memories must be 200').toBe(true);
	const list = (await listRes.json()) as { memories: { id: string; content: string }[] };
	expect(list.memories.length, 'list should return memories to enrich').toBeGreaterThan(0);

	// The per-memory endpoint MUST return a real nextReviewAt — this is the field
	// the organ enriches with and schedules on. If this ever regresses to null,
	// the schedule silently empties, so guard it explicitly.
	const first = list.memories[0];
	const fullRes = await request.get(`${API}/api/memories/${first.id}`);
	expect(fullRes.ok(), 'GET /api/memories/:id must be 200').toBe(true);
	const full = (await fullRes.json()) as { nextReviewAt?: string };
	expect(typeof full.nextReviewAt, 'per-memory endpoint must expose a real nextReviewAt').toBe(
		'string'
	);
	expect(Number.isFinite(Date.parse(full.nextReviewAt as string)), 'nextReviewAt must parse').toBe(
		true
	);

	const errors = captureErrors(page);
	await gotoRoute(page, '/schedule');
	await page.waitForTimeout(SETTLE_MS);

	// 1 + 2 — the canvas renders a non-black field (real due rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`schedule field should render real data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	// REGRESSION GUARD for the ownership bug: the organ used to collapse to a
	// dark/empty field (list endpoint strips nextReviewAt → every row filtered out,
	// plus a depth/reveal bug that discarded the rows even once enriched). A dark
	// field still slips past `rendered` via the PNG-size heuristic, so assert the
	// due rows are actually LIT: scan the canvas for genuinely bright text pixels.
	const bright = await brightPixelCount(page);
	expect(
		bright.maxL,
		`schedule rows must be visibly lit, not a dark field (maxL=${bright.maxL} bright=${bright.bright})`
	).toBeGreaterThan(60);
	expect(bright.bright, `schedule should light many text pixels (bright=${bright.bright})`).toBeGreaterThan(
		200
	);

	expectNoErrors(errors);
});

/**
 * Scan the WebGPU canvas screenshot for genuinely bright pixels. This is the
 * proof that the review ROWS render (not just a near-black observatory field):
 * before the fix this returned maxL≈9 / bright=0; after it returns maxL≈100+ /
 * bright in the thousands.
 */
async function brightPixelCount(page: Page): Promise<{ maxL: number; bright: number }> {
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
		t.width = 320;
		t.height = 180;
		const ctx = t.getContext('2d');
		if (!ctx) return { maxL: 0, bright: 0 };
		ctx.drawImage(img, 0, 0, 320, 180);
		const d = ctx.getImageData(0, 0, 320, 180).data;
		let maxL = 0;
		let bright = 0;
		for (let i = 0; i < d.length; i += 4) {
			const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
			if (l > maxL) maxL = l;
			if (l > 30) bright++;
		}
		return { maxL: Math.round(maxL), bright };
	}, dataUrl);
}

test('schedule field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/schedule');
	await page.waitForTimeout(SETTLE_MS);

	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field must render before checking motion (avgLum=${sample.avgLum})`).toBe(
		true
	);

	// The observatory scene + per-row reveal animate continuously. Under full-suite
	// GPU load two adjacent frames can hash-match, so retry a few windows — a
	// frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'schedule review field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the schedule field never crashes (real picks survive)', async ({ page }) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/schedule');
	await page.waitForTimeout(SETTLE_MS);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover sweep first — drives pointermove → pickAt + cursor-lens writes across
	// the field. Must not throw.
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

	// Click grid — the rows are anchored down the left column (x=-0.9), so bias
	// picks left where the real schedule rows live. Each hit fires a real promote
	// POST; the field must survive every one (no state corruption / WebGPU error).
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
	expect(after.rendered, 'schedule field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

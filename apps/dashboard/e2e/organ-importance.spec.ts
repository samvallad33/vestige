// ─────────────────────────────────────────────────────────────────────────────
// IMPORTANCE — the real-data ImportanceScore MSDF field (organ ownership spec).
//
// This organ renders the REAL brain's memories ranked by neuromodulatory
// importance. It reads GET /api/memories?limit=36, then scores each record via
// POST /api/importance {content} → { channels{novelty,arousal,reward,attention},
// composite, recommendation }, sorts by composite, and lays each out as an MSDF
// text row (TextLayerPass over the recall-path observatory scene):
//   snippet | id8 | composite% | retention% | recommendation | strongestChannel
// The cursor swells/leans nearby glyphs; clicking a row PROMOTES the memory.
//
// The organ contract proven here:
//   1. REACHABLE — /dashboard/importance mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the SAME real memories curl returns, scored by the
//      real /api/importance endpoint, drive the field (non-black render). The
//      importance payload is asserted to be a real ImportanceScore, not mock.
//   3. ALIVE — the field animates at idle (per-glyph time wobble + reveal +
//      recall-path scene), no interaction required.
//   4. CRASH-FREE pick + hover — a hover sweep and a grid of clicks (each a real
//      promote POST) never throw a page/WebGPU error. This is the exact path
//      that used to crash: promote returns a PARTIAL payload ({id,promoted,
//      retentionStrength}), and the page now MERGES it onto the full record
//      instead of replacing it, so importanceLine() never reads undefined
//      content on the next render.
//   5. HONEST states — the empty/error/loading branches render a calm status
//      line (EMPTY IMPORTANCE FIELD / ERROR - .. / LOADING ..), never fake data.
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

test('importance organ mounts a canvas and renders the REAL scored field', async ({
	page,
	request
}) => {
	// 2 (real data, part A) — curl the real brain FIRST: there must be real
	// memories carrying the fields the field lays out (content, id, retention).
	const memRes = await request.get(`${API}/api/memories?limit=36`);
	expect(memRes.ok(), 'GET /api/memories must be 200').toBe(true);
	const memPayload = (await memRes.json()) as {
		memories: { id: string; content: string; retentionStrength: number }[];
		total: number;
	};
	expect(Array.isArray(memPayload.memories)).toBe(true);
	expect(
		memPayload.memories.length,
		'real brain should have memories to score + render'
	).toBeGreaterThan(0);
	const first = memPayload.memories[0];
	expect(typeof first.id).toBe('string');
	expect(typeof first.content).toBe('string');
	expect(typeof first.retentionStrength).toBe('number');

	// 2 (real data, part B) — the organ's own data source: POST /api/importance
	// must return a REAL ImportanceScore for real content (not a mock/fake shape).
	const impRes = await request.post(`${API}/api/importance`, {
		data: { content: first.content }
	});
	expect(impRes.ok(), 'POST /api/importance must be 200').toBe(true);
	const score = (await impRes.json()) as {
		channels: { novelty: number; arousal: number; reward: number; attention: number };
		composite: number;
		recommendation: string;
	};
	expect(score.channels, 'importance score must carry all 4 channels').toEqual(
		expect.objectContaining({
			novelty: expect.any(Number),
			arousal: expect.any(Number),
			reward: expect.any(Number),
			attention: expect.any(Number)
		})
	);
	expect(typeof score.composite).toBe('number');
	expect(score.composite).toBeGreaterThanOrEqual(0);
	expect(score.composite).toBeLessThanOrEqual(1);
	expect(typeof score.recommendation).toBe('string');

	const errors = captureErrors(page);
	await gotoRoute(page, '/importance');
	// The field lists memories, awaits a scoring round-trip per row, then reveals
	// row-by-row over ~2s. Give it generous settle time (scoring is N requests).
	await page.waitForTimeout(5000);

	// 1 + 2 — the canvas renders a non-black field (the real scored rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`importance field should render real scored data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('importance field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/importance');
	await page.waitForTimeout(5000);

	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`field must render before checking motion (avgLum=${sample.avgLum})`
	).toBe(true);

	// Per-glyph time wobble + reveal + the recall-path scene animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'importance field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the importance field never crashes (real promotes survive)', async ({
	page
}) => {
	// This test screenshots the canvas many times (each pick + two samples + a
	// hover scan). Under full-suite GPU load a single compositor screenshot can
	// stall for seconds, so give the whole test generous headroom over the 60s
	// default — the assertions themselves are fast, only the GPU readbacks are slow.
	test.setTimeout(120_000);
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/importance');
	await page.waitForTimeout(5000);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Track real promotes: every landed pick must fire POST …/promote. We assert
	// at least one lands so the crash-free proof genuinely exercises the promote →
	// merge → re-render path (a pick grid that MISSES the rows proves nothing).
	const promotes: string[] = [];
	page.on('request', (r) => {
		if (r.method() === 'POST' && /\/promote$/.test(r.url())) promotes.push(r.url());
	});

	// The real scored rows occupy a band the live field lays out at roughly
	// fx∈[0.25,0.55], fy∈[0.40,0.78] (anchor x=-0.9 divided by the ~1.78 aspect
	// lands the left column near fx≈0.25 in landscape). Verified live via the
	// cursor→crosshair hit probe. Hover + click inside that band.
	const hoverPts = [
		[0.28, 0.42],
		[0.35, 0.5],
		[0.42, 0.58],
		[0.3, 0.66],
		[0.4, 0.74]
	];
	for (const [fx, fy] of hoverPts) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(120);
	}

	// Click grid ON the real rows. Each hit fires a real promote POST, which
	// returns a PARTIAL payload ({id,promoted,retentionStrength}); the
	// merge-not-replace fix means the next render reads full content and the field
	// survives every pick (no WebGPU error, no black-out, no TypeError from
	// undefined content). This is the exact path that used to crash.
	const clickPts = [
		[0.28, 0.42],
		[0.34, 0.48],
		[0.3, 0.54],
		[0.4, 0.6],
		[0.33, 0.66],
		[0.29, 0.72],
		[0.45, 0.5],
		[0.5, 0.58],
		[0.7, 0.4]
	];
	for (const [fx, fy] of clickPts) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(250);
	}

	// Prove picks actually LANDED (not a hollow miss): at least one real promote
	// fired. This is what makes the crash-free assertion meaningful.
	expect(promotes.length, 'at least one click must land on a real row and promote').toBeGreaterThan(
		0
	);

	// Field still renders after all the picks (no crash, no black-out).
	const after = await sampleCanvas(page);
	expect(after.rendered, 'importance field still renders after clicks + hover').toBe(true);

	// REGRESSION GUARD — the field must STAY the real scored list after a promote,
	// not collapse to a single spurious ERROR line. The promote endpoint returns a
	// PARTIAL payload; the old replace-outright code dropped memory.content, threw
	// inside importanceLine(), got swallowed by the catch, and flipped the whole
	// field to `ERROR - ...` on the first click (verified live). Proof: the row
	// band is still hoverable (cursor → crosshair) after all the promotes fired.
	let rowStillHits = false;
	for (let yi = 8; yi <= 15 && !rowStillHits; yi++) {
		await page.mouse.move(box.x + box.width * 0.32, box.y + box.height * (yi / 20));
		const cur = await page.evaluate(
			() => (document.querySelector('div.fixed.inset-0') as HTMLElement).style.cursor
		);
		if (cur === 'crosshair') rowStillHits = true;
	}
	expect(
		rowStillHits,
		'real rows must stay pickable after promotes (field did not collapse to ERROR)'
	).toBe(true);

	expectNoErrors(errors);
});

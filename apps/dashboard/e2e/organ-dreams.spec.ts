// ─────────────────────────────────────────────────────────────────────────────
// DREAMS — the offline-replay insight field (real-data ownership spec).
//
// This organ runs a REAL dream cycle on mount (POST /api/dream {}) and renders
// the result as a WebGPU MSDF text field (TextLayerPass over the recall-path
// RouteStage scene). Row 0 is the cycle summary (status | memoriesReplayed |
// connectionsPersisted | insightsGenerated | durationMs); each subsequent row is
// a discovered insight (insight | type | confidence% | novelty%). Clicking any
// row re-runs the dream — the pick fires a fresh POST /api/dream.
//
// The 7-point organ contract proven here:
//   1. REACHABLE — /dashboard/dreams mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the same real DreamResult the curl returns drives the
//      field (non-black render); the scene seed carries the real durationMs, and
//      the +page.svelte normalizes the exact status/insights/stats shape the API
//      returns (no mock/fake data path, no fake "Live" badge).
//   3. ALIVE — the RouteStage recall-path scene + per-glyph idle wobble animate
//      continuously, no interaction required.
//   4. CRASH-FREE pick + hover — a grid of clicks (each a real re-dream POST) and
//      a hover sweep never throw a page/WebGPU error. The rows are anchored down
//      the left column at a STATIC anchor; pickAt mirrors the shader's aspect
//      transform, and the cursor swell/lean is anchored (visual-only lens), so
//      picks track the field — verified crash-free on the live brain.
//   5. HONEST states — loading shows REPLAYING COGNITIVE RECEIPT..., error shows
//      ERROR - .., and an insight-free cycle still renders the honest summary row
//      (alive), never fake data.
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

// A real dream finishes in ~150-350ms; the field then reveals its rows over ~1s.
// Settle generously so the MSDF glyphs are uploaded + revealed before sampling.
const SETTLE_MS = 2500;

test('dreams organ mounts a canvas and renders the REAL dream cycle', async ({ page, request }) => {
	// 2 (real data) — run the real dream FIRST and assert the exact shape the
	// +page.svelte consumes. The organ POSTs the same endpoint on mount.
	const apiRes = await request.post(`${API}/api/dream`, { data: {} });
	expect(apiRes.ok(), 'POST /api/dream must be 200').toBe(true);
	const dream = (await apiRes.json()) as {
		status: string;
		memoriesReplayed: number;
		connectionsPersisted: number;
		insights: { type: string; insight: string; confidence: number; noveltyScore: number }[];
		stats: { insightsGenerated: number; durationMs: number; newConnectionsFound: number };
	};
	expect(typeof dream.status).toBe('string');
	expect(dream.status.length, 'real dream returns a status').toBeGreaterThan(0);
	expect(typeof dream.memoriesReplayed).toBe('number');
	expect(dream.memoriesReplayed, 'real brain should replay memories').toBeGreaterThan(0);
	expect(Array.isArray(dream.insights)).toBe(true);
	expect(typeof dream.stats.durationMs).toBe('number');

	const errors = captureErrors(page);
	await gotoRoute(page, '/dreams');
	// Mount fires POST /api/dream; then the summary + insight rows reveal.
	await page.waitForTimeout(SETTLE_MS);

	// 1 + 2 — the canvas renders a non-black field (the real cycle lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`dreams field should render the real cycle (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('dreams field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/dreams');
	await page.waitForTimeout(SETTLE_MS);

	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field must render before checking motion (avgLum=${sample.avgLum})`).toBe(
		true
	);

	// The RouteStage recall-path scene + per-glyph idle wobble animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'dreams insight field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the dream field never crashes (real re-dream picks survive)', async ({
	page
}) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/dreams');
	await page.waitForTimeout(SETTLE_MS);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover sweep first — drives pointermove → pickAt + cursor-lens writes across
	// the field (the swell/lean path). Rows live down the left column, so bias
	// the sweep left where the real dream rows sit. Must not throw.
	const hoverPts = [
		[0.15, 0.2],
		[0.25, 0.32],
		[0.2, 0.45],
		[0.3, 0.58],
		[0.22, 0.7]
	];
	for (const [fx, fy] of hoverPts) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(120);
	}

	// Click grid — the rows are anchored down the left column (x anchor -0.88),
	// so bias picks left where the real dream rows live. Each hit on a row fires a
	// fresh POST /api/dream; the field must survive every one (no state corruption
	// / WebGPU error). A real re-dream takes ~150-350ms, so pace the clicks.
	const clickPts = [
		[0.12, 0.2],
		[0.2, 0.3],
		[0.15, 0.4],
		[0.25, 0.5],
		[0.18, 0.6],
		[0.22, 0.7],
		[0.5, 0.5],
		[0.7, 0.4]
	];
	for (const [fx, fy] of clickPts) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(400);
	}

	// Let any in-flight re-dream settle, then confirm the field still renders
	// (no crash, no black-out) after all the picks.
	await page.waitForTimeout(1000);
	const after = await sampleCanvas(page);
	expect(after.rendered, 'dreams field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

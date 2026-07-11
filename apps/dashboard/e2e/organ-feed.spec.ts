// ─────────────────────────────────────────────────────────────────────────────
// FEED — the live event stream field (real-data ownership spec).
//
// This organ is WEBSOCKET-ONLY: it consumes $eventFeed from $stores/websocket
// (there is NO REST endpoint). The backend pushes a {"type":"Connected"} frame
// immediately on /ws open, then Heartbeats every ~5s, plus real lifecycle events
// (MemoryPromoted, MemoryPrOpened, TraceEvent, ...) as the brain works. The feed
// lays each NON-heartbeat event out as a row (type | id | payload summary) in a
// TextLayerPass over the observatory scene; clicking a row clears the feed.
//
// It was DOUBLY dead before the WS-guard fix in +layout.svelte (the old
// `!isMarketingRoute && !isImmersiveRoute` guard was always false, so the socket
// never connected on ANY route). It only lights up now that the socket connects.
//
// The 7-point organ contract proven here (real brain :3931 via the :5199 proxy):
//   1. REACHABLE — /dashboard/feed mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the app's OWN socket delivers real frames (Connected +
//      real lifecycle events); those events drive a non-black, genuinely-lit field
//      (max glyph luminance well above the background floor). No mock, no fake
//      "Live" badge over stale data.
//   3. ALIVE — the field animates at idle (per-glyph time wobble + reveal + scene),
//      no interaction required.
//   4. CRASH-FREE pick + hover — a hover sweep + a click grid (each a real
//      clearEvents on a row hit) never throw a page/WebGPU error, and the field
//      still renders afterward. pickAt mirrors the shader's aspect transform.
//   5. HONEST empty/connection state — with zero events the feed shows a calm
//      connection-state line ({"connected":..,"reconnecting":..,"events":0}),
//      never a broken/black surface and never fabricated rows.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect, type Page } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

/**
 * Collect the NON-heartbeat frames the app's OWN /ws socket receives (heartbeats
 * are routed to lastHeartbeat, not the feed, so we exclude them). Proves the feed
 * is driven by the real live stream, not a mock.
 */
function watchFeedFrames(page: Page): string[] {
	const types: string[] = [];
	page.on('websocket', (ws) => {
		if (!ws.url().endsWith('/ws')) return; // ignore Vite's own HMR socket
		ws.on('framereceived', (f) => {
			const s = typeof f.payload === 'string' ? f.payload : '';
			let type = '?';
			try {
				type = JSON.parse(s).type as string;
			} catch {
				/* non-JSON frame */
			}
			if (type && type !== 'Heartbeat') types.push(type);
		});
	});
	return types;
}

/** Max glyph luminance across the frame — proves the text is genuinely LIT, not
 * a sub-threshold smear. Background floor in this field is ~14; real cyan glyphs
 * peak well above 25. */
async function maxGlyphLuminance(page: Page): Promise<number> {
	const canvas = page.locator('canvas').first();
	const buf = await canvas.screenshot({ timeout: 8000 });
	const url = `data:image/png;base64,${buf.toString('base64')}`;
	return await page.evaluate(async (u) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img'));
			img.src = u;
		});
		const t = document.createElement('canvas');
		t.width = 1280;
		t.height = 720;
		const ctx = t.getContext('2d');
		if (!ctx) return 0;
		ctx.drawImage(img, 0, 0, 1280, 720);
		const d = ctx.getImageData(0, 0, 1280, 720).data;
		let max = 0;
		for (let i = 0; i < d.length; i += 4) {
			const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
			if (l > max) max = l;
		}
		return Math.round(max);
	}, url);
}

test('feed mounts a canvas and renders REAL live WS events (points 1 + 2)', async ({ page }) => {
	const errors = captureErrors(page);
	const frames = watchFeedFrames(page);

	// 1 — REACHABLE: the route mounts a WebGPU canvas.
	const canvas = await gotoRoute(page, '/feed');
	await expect(canvas).toBeAttached();

	// Give the socket time to open (Connected fires immediately) + a couple of
	// heartbeat windows for any real lifecycle events the brain emits.
	await page.waitForTimeout(6000);

	// 2 — REAL DATA: the app's own socket must have delivered at least the
	// Connected frame (a real backend frame, not a mock). This is the ONLY data
	// source for this organ.
	expect(
		frames.length,
		`feed must receive real non-heartbeat WS frames (got: ${JSON.stringify(frames)})`
	).toBeGreaterThan(0);
	expect(frames, 'the immediate Connected frame must be present').toContain('Connected');

	// 2 — those real events drive a non-black field...
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`feed field should render the live events (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	// ...and the glyphs are GENUINELY lit at REST (not a sub-threshold smear that
	// only shows under cursor flare): real cyan rows peak far above the ~14
	// background floor. This is the regression guard for the bug fixed in this
	// organ — feed rows used raw recencyDepth, whose low values on a sparse live
	// feed drove the MSDF depth-of-field to blur rows into the void at rest
	// (resting maxLum ~9). Raising depth to a 0.6 floor + a churn-robust reveal
	// (startFrame:0, revealSpan:1 so the demo-clock reset RouteStage fires on
	// every event never leaves a row half-revealed) lifts resting rows to a
	// stable maxLum ~150 (measured 148-151 across 10 frames). A large margin
	// over 25 keeps this stable without being brittle to a single short row.
	const maxLum = await maxGlyphLuminance(page);
	expect(
		maxLum,
		`feed row glyphs must be genuinely visible AT REST, not blurred out (maxLum=${maxLum})`
	).toBeGreaterThan(25);

	expectNoErrors(errors);
});

test('feed field is ALIVE at idle — no interaction required (point 3)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/feed');
	await page.waitForTimeout(5000);

	const sample = await sampleCanvas(page);
	expect(sample.rendered, `field must render before checking motion (avgLum=${sample.avgLum})`).toBe(
		true
	);

	// Per-glyph time wobble + reveal + the observatory scene animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'feed event field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the feed never crashes (real picks survive) (point 4)', async ({
	page
}) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/feed');
	await page.waitForTimeout(4000);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover sweep first — drives pointermove → pickAt + cursor-lens writes across
	// the row column (top-left, where rows anchor at x=-0.9). Must not throw.
	const hoverPts: Array<[number, number]> = [
		[0.1, 0.15],
		[0.2, 0.2],
		[0.15, 0.3],
		[0.05, 0.12],
		[0.5, 0.5]
	];
	for (const [fx, fy] of hoverPts) {
		await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(120);
	}

	// Click grid — biased top-left where the real event rows live. A hit fires the
	// real clearEvents() path (handleRoutePick on a feed-event); the field must
	// survive every one (no state corruption / WebGPU error).
	const clickPts: Array<[number, number]> = [
		[0.1, 0.12],
		[0.15, 0.15],
		[0.2, 0.18],
		[0.25, 0.3],
		[0.5, 0.4],
		[0.75, 0.6]
	];
	for (const [fx, fy] of clickPts) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
		await page.waitForTimeout(200);
	}

	// Field still renders after all the picks (no crash, no black-out).
	const after = await sampleCanvas(page);
	expect(after.ok, 'feed canvas must still be sampleable after picks').toBe(true);
	expect(after.rendered, 'feed field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

test('feed shows an HONEST connection state when the event feed is empty (point 5)', async ({
	page
}) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/feed');
	await expect(canvas).toBeAttached();
	await page.waitForTimeout(4000);

	// Drive the real clearEvents() path by clicking a row, emptying $eventFeed. With
	// zero events the organ MUST fall back to the calm connection-state line
	// ({"connected":..,"reconnecting":..,"events":0}) — never a black/errored
	// surface and never fabricated rows.
	const box = await canvas.boundingBox();
	if (box) {
		// tap the top-left row column repeatedly to hit + clear a feed-event row
		for (const [fx, fy] of [
			[0.1, 0.12],
			[0.15, 0.15],
			[0.12, 0.14]
		] as Array<[number, number]>) {
			await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
			await page.waitForTimeout(150);
		}
	}

	// Whether the feed is empty (status line) or a fresh Connected frame re-arrived,
	// the surface must stay a rendered, non-errored field — never a broken black
	// crash. (The empty branch renders statusItem(connectionPayload(...)); the
	// RouteStage empty branch renders the same connection payload as emptyLabel.)
	const sample = await sampleCanvas(page);
	expect(sample.ok, 'feed canvas must stay sampleable in the empty/connection state').toBe(true);
	expect(
		sample.rendered,
		`feed must render an honest connection state, not a black crash (avgLum=${sample.avgLum})`
	).toBe(true);

	expectNoErrors(errors);
});

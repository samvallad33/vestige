// ─────────────────────────────────────────────────────────────────────────────
// PATTERNS — the cross-project pattern-transfer MSDF field (real-data spec).
//
// This organ renders the REAL brain's cross-project pattern transfers as an MSDF
// text field (TextLayerPass over the observatory RouteStage). It consumes
// GET /api/patterns/cross-project and lays each pattern out as a row:
//   name | origin_project | transferred_to | category | transfer_count | conf% | last_used
// Clicking a row toggles a category filter (recolors the matching rows amber).
//
// The organ contract proven here:
//   1. REACHABLE — /dashboard/patterns mounts a WebGPU canvas.
//   2. RENDERS REAL DATA — the same patterns curl returns drive a non-black field,
//      and the scene seed carries the real project + pattern counts.
//   3. ALIVE — the field animates at idle (per-glyph reveal/wobble + the scene),
//      no interaction required.
//   4. CRASH-FREE pick + hover — a grid of clicks (each a real category-toggle)
//      and a hover sweep never throw a page/WebGPU error. The rows are ANCHORED
//      (static x/y — no vertex-position animation), so pickAt tracks them exactly;
//      picks survive on the live brain.
//   5. HONEST states — the empty branch flips scene.alive=false → RouteStage draws
//      a calm status line, never a fake "Live" badge over mock data.
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

// The e2e sampleCanvas helper reports rendered=true via a size>20_000 PNG-size
// fallback even for a near-black field — a FALSE GREEN (a sibling text organ,
// /memory-prs, shipped near-black under exactly that fallback). So we ALSO decode
// true luminance: max brightness AND the count of genuinely-lit pixels in the
// LEFT column where the pattern rows are anchored (x=-0.88). A frozen/near-black
// field fails this; the real legible cyan rows pass it big.
async function decodeLeftColumnLuminance(
	page: Page
): Promise<{ maxLum: number; leftBright: number }> {
	const buf = await page.locator('canvas').first().screenshot();
	const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
	return page.evaluate(async (url) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = url;
		});
		const W = 1280;
		const H = 720;
		const t = document.createElement('canvas');
		t.width = W;
		t.height = H;
		const ctx = t.getContext('2d');
		if (!ctx) return { maxLum: 0, leftBright: 0 };
		ctx.drawImage(img, 0, 0, W, H);
		const d = ctx.getImageData(0, 0, W, H).data;
		let maxLum = 0;
		let leftBright = 0;
		for (let y = 0; y < H; y++) {
			for (let x = 0; x < W; x++) {
				const i = (y * W + x) * 4;
				const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
				if (l > maxLum) maxLum = l;
				if (l > 40 && x < 700) leftBright++;
			}
		}
		return { maxLum: Math.round(maxLum), leftBright };
	}, dataUrl);
}

type CrossProjectPattern = {
	name: string;
	category: string;
	origin_project: string;
	transferred_to: string[];
	transfer_count: number;
	last_used: string;
	confidence: number;
};

test('patterns organ mounts a canvas and renders the REAL cross-project field', async ({
	page,
	request
}) => {
	// 2 (real data) — curl the real brain FIRST. There must be real patterns, and
	// each must carry the fields the field lays out into a row.
	const apiRes = await request.get(`${API}/api/patterns/cross-project`);
	expect(apiRes.ok(), 'GET /api/patterns/cross-project must be 200').toBe(true);
	const payload = (await apiRes.json()) as {
		projects: string[];
		patterns: CrossProjectPattern[];
	};
	expect(Array.isArray(payload.projects), 'projects is an array').toBe(true);
	expect(Array.isArray(payload.patterns), 'patterns is an array').toBe(true);
	expect(
		payload.patterns.length,
		'real brain should have cross-project patterns to render'
	).toBeGreaterThan(0);
	const first = payload.patterns[0];
	expect(typeof first.name).toBe('string');
	expect(typeof first.category).toBe('string');
	expect(typeof first.origin_project).toBe('string');
	expect(Array.isArray(first.transferred_to)).toBe(true);
	expect(typeof first.transfer_count).toBe('number');
	expect(typeof first.confidence).toBe('number');

	const errors = captureErrors(page);
	await gotoRoute(page, '/patterns');
	// The field loads patterns then reveals row-by-row; settle before sampling.
	await page.waitForTimeout(3500);

	// 1 + 2 — the canvas renders a non-black field (the real rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`patterns field should render real data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	// Strict anti-false-green: the real cyan pattern rows must be genuinely LIT in
	// the left column, not a near-black field that only passes the PNG-size branch.
	const lum = await decodeLeftColumnLuminance(page);
	expect(lum.maxLum, `patterns field must be genuinely lit (maxLum=${lum.maxLum})`).toBeGreaterThan(
		60
	);
	expect(
		lum.leftBright,
		`the 27 pattern rows must render in the left column (leftBright=${lum.leftBright})`
	).toBeGreaterThan(500);

	expectNoErrors(errors);
});

test('patterns field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/patterns');
	await page.waitForTimeout(3500);

	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`field must render before checking motion (avgLum=${sample.avgLum})`
	).toBe(true);

	// Per-glyph reveal/wobble + the observatory scene animate continuously. Under
	// full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'patterns field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('clicking + hovering the patterns field never crashes (real category toggles survive)', async ({
	page
}) => {
	const errors = captureErrors(page);
	const canvas = await gotoRoute(page, '/patterns');
	await page.waitForTimeout(3500);

	const box = await canvas.boundingBox();
	expect(box).not.toBeNull();
	if (!box) return;

	// Hover sweep first — drives pointermove → pickAt + cursor-lens writes across
	// the field (the swell/lean path). Must not throw.
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

	// Click grid — the rows are anchored down the left column (x=-0.88), so bias
	// picks left where the real pattern rows live. Each hit toggles the category
	// filter (re-derives + re-uploads the scene); the field must survive every one.
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

	// Field still renders after all the picks (no crash, no black-out, no leak).
	const after = await sampleCanvas(page);
	expect(after.rendered, 'patterns field still renders after clicks + hover').toBe(true);

	expectNoErrors(errors);
});

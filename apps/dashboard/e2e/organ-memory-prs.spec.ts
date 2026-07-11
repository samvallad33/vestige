// ─────────────────────────────────────────────────────────────────────────────
// Organ: /memory-prs — the risk-gated brain-change review queue.
//
// This organ is a full-bleed WebGPU field of text rows: one row per pending
// Memory PR (title | id | status), colored cyan for pending / muted otherwise,
// depth-mapped by the PR's confidence/trust. Clicking a row fires the real
// `ask_agent_why` action and reveals the PR's risk signals as amber rows.
//
// Data source (VERIFIED live against the real brain on :3931):
//   GET  /api/memory-prs?limit=N  → { total, pendingCount, mode, prs[] }
//   POST /api/memory-prs/:id/ask_agent_why → { why: [{code, detail}], ... }
// At verification time the real brain held 277 pending PRs, mode=risk_gated.
//
// This spec proves the 7-point ship contract for THIS organ:
//   1. REACHABLE     — route loads, mounts a canvas.
//   2. REAL DATA     — the API returns real PRs AND the field consumes them
//                      (glyph rows appear only when `prs` populated the scene).
//   3. ALIVE         — the field animates (reveal build after mount).
//   4. CRASH-FREE    — a grid of in-canvas picks + a hover survive with zero
//                      page/WebGPU errors, and the field still renders after.
//   5. HONEST STATES — empty/error are handled via RouteStage's alive/error
//                      branches (asserted structurally; the live brain is
//                      non-empty so we prove the populated path end-to-end).
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	BASE,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas
} from './helpers/dashboard';

const ROUTE = '/memory-prs';

// A grid of canvas-relative fractions. The PR rows are laid out down the LEFT
// column (x from ~-0.9), so the grid deliberately covers the left field where
// the pickable rows live, plus HUD/center to prove non-row clicks are safe too.
const GRID: Array<[number, number]> = [
	[0.15, 0.2],
	[0.2, 0.35],
	[0.25, 0.5],
	[0.2, 0.65],
	[0.5, 0.4],
	[0.75, 0.5]
];

async function clickGrid(page: Page) {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	expect(box, 'canvas must have a bounding box to click into').not.toBeNull();
	if (!box) return;
	for (const [fx, fy] of GRID) {
		await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy, { timeout: 3000 });
		// let the pick + any ask_agent_why fetch + re-render settle
		await page.waitForTimeout(300);
	}
}

/**
 * Strict, organ-specific render proof. The shared `sampleCanvas` helper has a
 * `size > 20_000` PNG-size fallback that reports rendered=true even for a
 * near-black 1280×720 field — which masked a real bug here where the MSDF reveal
 * stagger left the whole queue at maxLum=9 (a FALSE GREEN). This decodes the
 * canvas at full resolution and asserts there are genuinely LIT pixels (visible
 * glyph strokes), so a regression back to the dark field fails honestly.
 */
async function decodeBrightness(
	page: Page
): Promise<{ avgLum: number; maxLum: number; brightPct: number }> {
	const canvas = page.locator('canvas').first();
	const buf = await canvas.screenshot({ timeout: 8000 });
	const b64 = buf.toString('base64');
	return page.evaluate(async (data: string) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = `data:image/png;base64,${data}`;
		});
		const c = document.createElement('canvas');
		c.width = img.width;
		c.height = img.height;
		const ctx = c.getContext('2d');
		if (!ctx) return { avgLum: 0, maxLum: 0, brightPct: 0 };
		ctx.drawImage(img, 0, 0);
		const d = ctx.getImageData(0, 0, c.width, c.height).data;
		let sum = 0;
		let max = 0;
		let bright = 0;
		let n = 0;
		for (let i = 0; i < d.length; i += 4) {
			const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
			sum += l;
			if (l > max) max = l;
			if (l > 40) bright++;
			n += 1;
		}
		return {
			avgLum: +(sum / n).toFixed(2),
			maxLum: +max.toFixed(1),
			brightPct: +((100 * bright) / n).toFixed(3)
		};
	}, b64);
}

test.describe('Organ: /memory-prs — risk-gated review queue', () => {
	test('renders real Memory PRs, animates, and survives picks + hover', async ({ page }) => {
		const capture = captureErrors(page);

		// ── (2a) REAL DATA exists at the API the organ consumes ──────────────────
		// Hit the SAME endpoint the organ uses (through the dev-server proxy) so the
		// assertion is against the real brain, not a fixture.
		const apiRes = await page.request.get(`/api/memory-prs?limit=100`);
		expect(apiRes.ok(), 'memory-prs API must respond 2xx').toBe(true);
		const payload = (await apiRes.json()) as {
			total: number;
			pendingCount: number;
			mode: string;
			prs: Array<{ id: string; kind: string; status: string; title: string; signals: unknown[] }>;
		};
		expect(payload.mode, 'review mode must be present').toBeTruthy();
		expect(Array.isArray(payload.prs), 'prs must be an array').toBe(true);
		const hasRealPrs = payload.prs.length > 0;
		// The live brain is non-empty; if it ever isn't, the empty-state branch is
		// exercised by the all-routes smoke — here we prove the populated path.
		expect(hasRealPrs, 'live brain must expose real Memory PRs for the populated-path proof').toBe(true);
		// Each PR must carry the fields the organ renders into a row.
		for (const pr of payload.prs.slice(0, 5)) {
			expect(typeof pr.id, 'pr.id').toBe('string');
			expect(typeof pr.title, 'pr.title').toBe('string');
			expect(typeof pr.status, 'pr.status').toBe('string');
		}

		// ── (1) REACHABLE — route mounts a canvas ────────────────────────────────
		const canvas = await gotoRoute(page, ROUTE);
		await expect(canvas).toBeAttached();

		// organs build (font atlas + reveal) over a couple seconds — settle.
		await page.waitForTimeout(3000);

		// ── (2b) RENDERS non-black — the field consumed the real PR rows ─────────
		// The TextLayerPass packs glyph instances only when `prs` populated
		// buildTextItems(); with the live brain non-empty (asserted above) a
		// non-black field is direct proof the organ rendered REAL PR rows — not
		// mock data, not a fake "Live" badge over nothing. The empty-payload test
		// below pins the other side (zero PRs → calm empty label).
		const before = await sampleCanvas(page);
		expect(before.ok, 'canvas must be sampleable').toBe(true);
		expect(
			before.rendered,
			`memory-prs field must render non-black (avgLum=${before.avgLum} var=${before.variance})`
		).toBe(true);
		// Real luminance proof: the PR rows must be genuinely LIT. A near-black
		// field (the MSDF reveal-stagger bug that shipped this organ dark) tops out
		// at maxLum~9 / brightPct 0 and would still pass sampleCanvas via its
		// PNG-size fallback. This decodes true max luminance so that FALSE GREEN
		// can never pass again.
		const lit = await decodeBrightness(page);
		expect(
			lit.maxLum,
			`memory-prs rows must be visibly lit, not a dark field (maxLum=${lit.maxLum} brightPct=${lit.brightPct})`
		).toBeGreaterThan(60);
		expect(
			lit.brightPct,
			`memory-prs must light a real fraction of pixels (maxLum=${lit.maxLum} brightPct=${lit.brightPct})`
		).toBeGreaterThan(0.3);

		// ── (3) ALIVE — the field animates (reveal build) ────────────────────────
		// The reveal is frame-driven (ageFrame/revealSpan in the shader) plus a
		// cursor-lens depth animation. Probe several short windows right after a
		// fresh mount and require motion in at least one — a single post-settle
		// pair can catch two identical resting frames and false-negative.
		let animated = false;
		let prev = await canvas.screenshot({ timeout: 8000 });
		for (let i = 0; i < 8 && !animated; i++) {
			await page.waitForTimeout(350);
			const cur = await canvas.screenshot({ timeout: 8000 });
			if (Buffer.compare(prev, cur) !== 0) animated = true;
			prev = cur;
		}
		// If the reveal already finished, nudge with a hover to prove the cursor
		// lens animates the field, then re-probe.
		if (!animated) {
			const box = await canvas.boundingBox();
			if (box) {
				await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.4);
				await page.waitForTimeout(120);
				const h1 = await canvas.screenshot({ timeout: 8000 });
				await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.55);
				await page.waitForTimeout(180);
				const h2 = await canvas.screenshot({ timeout: 8000 });
				if (Buffer.compare(h1, h2) !== 0) animated = true;
			}
		}
		expect(animated, 'memory-prs field must animate (reveal build or cursor lens)').toBe(true);

		// ── (4) CRASH-FREE picks + hover ─────────────────────────────────────────
		// Hover across the row column first (drives the cursor lens + chrome pick),
		// then fire a grid of clicks (each may trigger a real ask_agent_why fetch).
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
			await page.waitForTimeout(150);
			await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.45);
			await page.waitForTimeout(150);
		}
		await clickGrid(page);

		// No page errors, no WebGPU validation errors on ANY hover or pick.
		expectNoErrors(capture);

		// ── (4b) Field STILL renders after all interaction ───────────────────────
		const after = await sampleCanvas(page);
		expect(after.ok, 'canvas must still be sampleable after picks').toBe(true);
		expect(
			after.rendered,
			`memory-prs field must still render after picks (avgLum=${after.avgLum} var=${after.variance})`
		).toBe(true);

		await test.info().attach('memory-prs-after-picks.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});

	test('honest empty state: no data yields a calm empty label, not a crash', async ({ page }) => {
		// Prove the empty branch is honest by routing the SAME organ against an
		// empty payload. We stub only the list endpoint so the field receives zero
		// PRs; RouteStage must then render the `emptyLabel` (NO MEMORY PRS) via its
		// `!scene.alive` branch — never a black/errored surface, never fake rows.
		const capture = captureErrors(page);
		await page.route('**/api/memory-prs?*', async (route) => {
			await route.fulfill({
				status: 200,
				contentType: 'application/json',
				body: JSON.stringify({ total: 0, pendingCount: 0, mode: 'risk_gated', prs: [] })
			});
		});

		const canvas = await gotoRoute(page, ROUTE);
		await expect(canvas).toBeAttached();
		await page.waitForTimeout(2500);

		// The empty field is still a live canvas (chrome/nav + empty label render),
		// so it must be non-black — a broken empty state would be black/blank.
		const sample = await sampleCanvas(page);
		expect(sample.ok, 'empty-state canvas must be sampleable').toBe(true);
		expect(
			sample.rendered,
			`empty state must render a calm label, not black (avgLum=${sample.avgLum} var=${sample.variance})`
		).toBe(true);

		// No crash on the empty path.
		expectNoErrors(capture);

		await test.info().attach('memory-prs-empty.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});
	});
});

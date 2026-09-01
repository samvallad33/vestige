// ─────────────────────────────────────────────────────────────────────────────
// Spatial Palace — E2E navigation proof
//
// The Palace (/dashboard/palace) renders the 19 organ routes as ONE live 3D
// WebGPU constellation you orbit and click to enter. Source:
//   src/routes/(app)/palace/+page.svelte
//   - PalaceNodePass draws the orbiting orbs (own close-orbit camera).
//   - TextLayerPass draws the MSDF HUD title "THE MEMORY PALACE" + per-orb labels.
//   - onpointerdown -> pass.pickAt(ndc) -> goto(`${base}${hit.href}`) enters an organ.
//
// This spec proves, against the REAL brain (325+ memories) on :5199:
//   1. /palace mounts a live NON-BLACK WebGPU field that is ANIMATING (orbits).
//   2. The MSDF HUD title region is content-rich (bright text pixels top-left).
//   3. CLICK-TO-ENTER: a pointerdown on the constellation navigates to a
//      /dashboard/<organ> route (nav is driven by onpointerdown, NOT click).
//   4. The entered organ's own WebGPU canvas renders.
//   5. No page errors and no WebGPU-validation errors fire.
//
// WebGPU-readback constraint: reading the live canvas via Canvas2D returns BLACK
// in this ANGLE/headless setup, so all pixel proofs go through the shared
// sampleCanvas / isAnimating helpers, which screenshot via the real compositor.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import {
	BASE,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

// Every organ href the palace can navigate into (mirrors ORGAN_REGIONS in
// src/lib/observatory/palace-map.ts). Any of these is a valid "entered" URL.
const ORGAN_PATHS = [
	'observatory', 'reasoning', 'contradictions', 'blackbox', 'timeline',
	'duplicates', 'graph', 'memory-prs', 'memories', 'feed', 'explore',
	'activation', 'dreams', 'schedule', 'importance', 'patterns',
	'intentions', 'stats', 'settings'
];
const ORGAN_URL_RE = new RegExp(`/dashboard/(?:${ORGAN_PATHS.join('|')})(?:[/?#]|$)`);

/**
 * Sample the top-left HUD region where the MSDF title "THE MEMORY PALACE" is
 * rendered (page places it at x:-0.94, y:0.88 in NDC — i.e. the upper-left).
 * Returns avg luminance + variance of just that crop, decoded through the real
 * compositor (same PNG-decode path sampleCanvas uses — Canvas2D on the LIVE
 * WebGPU surface reads black, but a screenshot PNG decodes correctly).
 */
async function sampleTitleRegion(
	page: Page
): Promise<{ avgLum: number; variance: number; size: number }> {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	if (!box) return { avgLum: 0, variance: 0, size: 0 };
	// Top-left ~40% wide x ~18% tall — the HUD title band.
	const clip = {
		x: Math.round(box.x + box.width * 0.02),
		y: Math.round(box.y + box.height * 0.02),
		width: Math.round(box.width * 0.42),
		height: Math.round(box.height * 0.16)
	};
	const buf = await page.screenshot({ clip });
	const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
	const stats = await page.evaluate(async (url) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = url;
		});
		const t = document.createElement('canvas');
		t.width = 160;
		t.height = 64;
		const ctx = t.getContext('2d');
		if (!ctx) return { avgLum: 0, variance: 0 };
		ctx.drawImage(img, 0, 0, 160, 64);
		const d = ctx.getImageData(0, 0, 160, 64).data;
		let sum = 0;
		const lums: number[] = [];
		for (let i = 0; i < d.length; i += 4) {
			const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
			lums.push(l);
			sum += l;
		}
		const avg = sum / lums.length;
		let v = 0;
		for (const l of lums) v += (l - avg) * (l - avg);
		return { avgLum: Math.round(avg), variance: Math.round(v / lums.length) };
	}, dataUrl);
	return { ...stats, size: buf.length };
}

/**
 * Click points across the constellation until a pointerdown lands on an orb and
 * navigates. The camera slowly ORBITS, so the center node drifts — we spiral out
 * from center (50%,50%) probing a small grid, re-trying after a short settle so a
 * moving orb rotates under the cursor. Returns the URL we landed on, or null.
 *
 * Uses page.mouse.down/up (real pointerdown) because the handler is
 * onpointerdown — a plain click also fires pointerdown, but we assert the
 * pointerdown path explicitly.
 */
async function clickToEnter(page: Page): Promise<string | null> {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	if (!box) return null;
	const cx = box.x + box.width / 2;
	const cy = box.y + box.height / 2;

	// Fractions of half-width/height to probe, spiralling out from dead center.
	// The center organ (/observatory) sits at the origin so 0,0 usually hits; the
	// outer rings catch orbiting satellite orbs if the center just drifted away.
	const offsets: Array<[number, number]> = [
		[0, 0],
		[0.08, 0], [-0.08, 0], [0, 0.08], [0, -0.08],
		[0.16, 0.10], [-0.16, 0.10], [0.16, -0.10], [-0.16, -0.10],
		[0.28, 0], [-0.28, 0], [0, 0.24], [0, -0.24],
		[0.34, 0.18], [-0.34, 0.18], [0.34, -0.18], [-0.34, -0.18]
	];

	for (let pass = 0; pass < 3; pass++) {
		for (const [ox, oy] of offsets) {
			const x = cx + ox * (box.width / 2);
			const y = cy + oy * (box.height / 2);
			const before = page.url();
			await page.mouse.move(x, y);
			await page.mouse.down();
			await page.mouse.up();
			// pointerdown -> pickAt -> goto is synchronous-ish; give the SPA router a beat.
			try {
				await page.waitForURL(ORGAN_URL_RE, { timeout: 900 });
				return page.url();
			} catch {
				// URL didn't change (missed every orb at this point) — keep probing.
				if (page.url() !== before && ORGAN_URL_RE.test(page.url())) {
					return page.url();
				}
			}
		}
		// Let the constellation orbit a little so a fresh set of orbs slides under
		// the probe grid, then retry.
		await page.waitForTimeout(1200);
	}
	return null;
}

test.describe('Spatial Palace — navigation', () => {
	test('renders a live, animating constellation with an MSDF HUD title', async ({ page }) => {
		const errors = captureErrors(page);

		await gotoRoute(page, '/palace');
		// The palace builds its constellation + MSDF layer over a couple seconds.
		await page.waitForTimeout(3000);

		// (1) Live NON-BLACK field.
		const sample = await sampleCanvas(page);
		expect(
			sample.rendered,
			`palace field should render non-black — avgLum=${sample.avgLum} variance=${sample.variance}`
		).toBe(true);
		await test.info().attach('palace-field.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});

		// (1b) Field is ANIMATING — the constellation orbits, so two frames differ.
		const animating = await isAnimating(page, 800);
		expect(animating, 'palace constellation should be orbiting (animated field)').toBe(true);

		// (2) MSDF HUD title "THE MEMORY PALACE" is visibly rendered top-left.
		// It's WebGPU text (no DOM node), so prove it via bright pixels + structure
		// in the title crop, plus a content-rich full-frame screenshot.
		const title = await sampleTitleRegion(page);
		await test.info().attach('palace-title-region.png', {
			body: await page.screenshot({
				clip: await (async () => {
					const b = (await page.locator('canvas').first().boundingBox())!;
					return {
						x: Math.round(b.x + b.width * 0.02),
						y: Math.round(b.y + b.height * 0.02),
						width: Math.round(b.width * 0.42),
						height: Math.round(b.height * 0.16)
					};
				})()
			}),
			contentType: 'image/png'
		});
		expect(
			title.avgLum > 3 && title.variance > 6,
			`HUD title region should contain bright, structured text pixels — avgLum=${title.avgLum} variance=${title.variance} size=${title.size}`
		).toBe(true);

		// Whole-frame content-richness: a living MSDF+nebula palace compresses large.
		const full = await page.screenshot();
		expect(
			full.length,
			`palace frame should be content-rich (>100KB) — was ${full.length} bytes`
		).toBeGreaterThan(100_000);

		expectNoErrors(errors);
	});

	test('click-to-enter dives into an organ, whose canvas then renders', async ({ page }) => {
		const errors = captureErrors(page);

		await gotoRoute(page, '/palace');
		await page.waitForTimeout(3000);
		expect(page.url()).toContain('/palace');

		// (3) pointerdown on the constellation navigates to /dashboard/<organ>.
		const entered = await clickToEnter(page);
		expect(
			entered,
			'a pointerdown on the constellation should navigate into an organ route'
		).not.toBeNull();
		expect(entered!, `landed URL should be an organ route: ${entered}`).toMatch(ORGAN_URL_RE);

		await test.info().attach('palace-entered-organ.png', {
			body: await page.screenshot(),
			contentType: 'image/png'
		});

		// (4) The entered organ mounts + renders its own WebGPU canvas.
		const organCanvas = page.locator('canvas').first();
		await organCanvas.waitFor({ state: 'attached', timeout: 15_000 });
		await page.waitForTimeout(2500);
		const organSample = await sampleCanvas(page);
		expect(
			organSample.rendered,
			`entered organ (${entered}) should render a non-black canvas — avgLum=${organSample.avgLum} variance=${organSample.variance}`
		).toBe(true);

		// (5) No page errors, no WebGPU-validation errors across the dive.
		expectNoErrors(errors);
	});
});

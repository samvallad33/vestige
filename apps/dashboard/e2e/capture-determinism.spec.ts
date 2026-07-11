// ─────────────────────────────────────────────────────────────────────────────
// Capture Determinism — the shareable-still guarantee (?frame=N)
//
// The Observatory field supports a capture freeze: /observatory?frame=N pins the
// simulation to loop frame N (engine.ts: `const frame = this.freezeFrame ?? …`,
// `p[0] = frame`, capture_mode `p[11] = 1.0`). Everything derives from the
// wrapped loop frame, so the promise is:
//
//   SAME url (same frame, same seed, same demo)  →  IDENTICAL pixels.
//   DIFFERENT frame                              →  DIFFERENT pixels.
//
// This is what makes a recorded still reproducible: you can regenerate the exact
// same frame for a thumbnail/share card at any time. This spec proves it end to
// end against the REAL brain by:
//   1. Freezing frame=120, screenshotting the live canvas (buffer A).
//   2. Reloading the SAME url, screenshotting again (buffer B).
//   3. Asserting A ≈ B (byte-identical, or within a tiny AA tolerance).
//   4. Freezing frame=300 and asserting it DIFFERS from frame=120.
//
// The canvas screenshot goes through Playwright's real GPU compositor (the only
// path that reads a live WebGPU surface non-black in this ANGLE/headless setup —
// see helpers/dashboard.ts sampleCanvas).
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page, type Locator } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors } from './helpers/dashboard';

const DEMO = 'recall-path';
const FROZEN = 120;
const OTHER = 300;
const SETTLE_MS = 3500;

/** Navigate to a frozen capture URL and wait for the canvas + settle. */
async function gotoFrozen(page: Page, frame: number): Promise<Locator> {
	await page.goto(`${BASE}/observatory?frame=${frame}&demo=${DEMO}`);
	const canvas = page.locator('canvas').first();
	await canvas.waitFor({ state: 'attached', timeout: 15_000 });
	// Let the field boot, the real graph load, and the frozen frame paint.
	await page.waitForTimeout(SETTLE_MS);
	return canvas;
}

/** Screenshot the live canvas via the real compositor (non-black readback path). */
async function shotCanvas(page: Page): Promise<Buffer> {
	return page.locator('canvas').first().screenshot({ timeout: 8000 });
}

/**
 * Compare two PNG screenshots by decoding to raw RGB in-browser and measuring
 * the mean per-channel absolute difference (0 = identical, 255 = inverted). We
 * decode the actual pixels rather than diffing PNG bytes because two identical
 * frames can still produce byte-different PNGs (encoder nondeterminism) — pixels
 * are the real contract.
 */
async function meanPixelDiff(page: Page, a: Buffer, b: Buffer): Promise<number> {
	const urlA = `data:image/png;base64,${a.toString('base64')}`;
	const urlB = `data:image/png;base64,${b.toString('base64')}`;
	return page.evaluate(
		async ([ua, ub]) => {
			const load = (src: string) =>
				new Promise<HTMLImageElement>((res, rej) => {
					const img = new Image();
					img.onload = () => res(img);
					img.onerror = () => rej(new Error('img load'));
					img.src = src;
				});
			const [ia, ib] = await Promise.all([load(ua), load(ub)]);
			const W = 160;
			const H = 120;
			const draw = (img: HTMLImageElement) => {
				const c = document.createElement('canvas');
				c.width = W;
				c.height = H;
				const ctx = c.getContext('2d', { willReadFrequently: true })!;
				ctx.drawImage(img, 0, 0, W, H);
				return ctx.getImageData(0, 0, W, H).data;
			};
			const da = draw(ia);
			const db = draw(ib);
			let sum = 0;
			let n = 0;
			for (let i = 0; i < da.length; i += 4) {
				sum += Math.abs(da[i] - db[i]);
				sum += Math.abs(da[i + 1] - db[i + 1]);
				sum += Math.abs(da[i + 2] - db[i + 2]);
				n += 3;
			}
			return sum / n;
		},
		[urlA, urlB] as const
	);
}

/** Mean luminance of a screenshot — used to confirm the frozen frame isn't black. */
async function meanLum(page: Page, buf: Buffer): Promise<number> {
	const url = `data:image/png;base64,${buf.toString('base64')}`;
	return page.evaluate(async (u) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = u;
		});
		const c = document.createElement('canvas');
		c.width = 96;
		c.height = 96;
		const ctx = c.getContext('2d')!;
		ctx.drawImage(img, 0, 0, 96, 96);
		const d = ctx.getImageData(0, 0, 96, 96).data;
		let s = 0;
		for (let i = 0; i < d.length; i += 4) s += (d[i] + d[i + 1] + d[i + 2]) / 3;
		return s / (d.length / 4);
	}, url);
}

test.describe('Capture determinism — ?frame=N shareable still', () => {
	test.describe.configure({ mode: 'serial' });

	test('same ?frame=N reloads to identical pixels; a different frame differs', async ({
		page
	}) => {
		const errors = captureErrors(page);

		// (1) Freeze frame=120 and screenshot (buffer A).
		await gotoFrozen(page, FROZEN);
		const a = await shotCanvas(page);
		await test.info().attach('frame-120-A.png', { body: a, contentType: 'image/png' });

		// Guard: the frozen frame must actually be rendering (non-black). If it's
		// black, the field never booted against the brain — that's a real failure,
		// not a determinism pass. A trivially-black A==B would give a false green.
		const lumA = await meanLum(page, a);
		expect(lumA, 'frozen frame is black — field never rendered against the brain').toBeGreaterThan(2);

		// (2) Reload the SAME url, settle, screenshot again (buffer B).
		await gotoFrozen(page, FROZEN);
		const b = await shotCanvas(page);
		await test.info().attach('frame-120-B.png', { body: b, contentType: 'image/png' });

		// (3) A ≈ B. Same loop frame → same derived params → same pixels. Allow a
		// tiny mean-channel tolerance for AA/compositor jitter, but a live
		// (non-frozen) field would drift far past this between two 3.5s-apart shots.
		const sameFrameDiff = await meanPixelDiff(page, a, b);
		expect(
			sameFrameDiff,
			`same ?frame=${FROZEN} reload should be pixel-stable (mean channel diff ${sameFrameDiff.toFixed(2)})`
		).toBeLessThan(3);

		// (4) frame=300 must DIFFER from frame=120 — a different playhead paints a
		// different field. This proves the freeze is a real function of N, not a
		// constant.
		await gotoFrozen(page, OTHER);
		const c = await shotCanvas(page);
		await test.info().attach('frame-300.png', { body: c, contentType: 'image/png' });

		const crossFrameDiff = await meanPixelDiff(page, a, c);
		expect(
			crossFrameDiff,
			`frame=${OTHER} should differ from frame=${FROZEN} (mean channel diff ${crossFrameDiff.toFixed(2)})`
		).toBeGreaterThan(sameFrameDiff);

		// Sanity: the cross-frame difference is meaningfully larger than the
		// same-frame noise floor — not just barely over it.
		expect(
			crossFrameDiff,
			`frame=${OTHER} vs frame=${FROZEN} difference too small to be a real frame change`
		).toBeGreaterThan(2);

		expectNoErrors(errors);
	});
});

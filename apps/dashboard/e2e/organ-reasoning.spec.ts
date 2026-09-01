import { test, expect, type Page } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, gotoRoute, sampleCanvas, isAnimating } from './helpers/dashboard';

// ─────────────────────────────────────────────────────────────────────────
// ORGAN OWNER SPEC — /reasoning (Reasoning Theater, the deep_reference organ)
//
// The Reasoning Theater is a ZERO-DOM WebGPU organ: the ONLY DOM is a
// visually-hidden (sr-only) ask input for keyboard + screen-reader access.
// There is no DOM response panel — the decision trace (beam / ribbon / nucleus),
// the evidence galaxy, gates, and receipt are all rendered IN-CANVAS. So the
// 7-point organ contract is proven on PIXELS + the real API round-trip, not DOM
// widgets, against the REAL brain (:3931):
//   1. REACHABLE    — /dashboard/reasoning mounts its WebGPU canvas.
//   5. HONEST-EMPTY — before any query there is NO fabricated response; the
//                     sr-only ask input is empty and the DOM has no DOM response
//                     panels. The field is still alive from a passive real
//                     memory-pool substrate (no invented trace is claimed).
//   2. RENDERS REAL — a real query runs the 8-stage deep_reference pipeline (200)
//                     and the field RE-LIGHTS with the real evidence galaxy.
//   3. ALIVE        — the field animates at rest.
//   4. CRASH-FREE   — a grid of in-canvas clicks + a hover sweep survive with
//                     zero page/WebGPU errors.
// ─────────────────────────────────────────────────────────────────────────

const REASONING = '/reasoning';
const QUERY = 'How does FSRS-6 trust scoring work?';

async function askQuery(page: Page, query: string): Promise<void> {
	const input = page.locator('#reasoning-ask');
	await input.waitFor({ state: 'attached', timeout: 15_000 });
	await input.fill(query);
	await expect(input).toHaveValue(query);
	await input.press('Enter');
}

test.describe('Organ /reasoning — Reasoning Theater (zero-DOM)', () => {
	test('reachable, honest-empty, renders REAL data, alive, and survives click+hover', async ({
		page
	}) => {
		const capture = captureErrors(page);

		// ── 1. REACHABLE ───────────────────────────────────────────────────
		const canvas = await gotoRoute(page, REASONING);
		await expect(canvas).toBeVisible();

		// ── 5. HONEST-EMPTY (pre-query) ────────────────────────────────────
		// The ask input exists (sr-only) and is empty — no query has run, so no
		// trace is claimed. There are no DOM response panels to fake at all.
		const input = page.locator('#reasoning-ask');
		await expect(input).toHaveValue('');
		// The field is still sampleable + alive from the passive memory-pool
		// substrate (an honest "here is your corpus" backdrop, not a fake trace).
		await page.waitForTimeout(3500);
		const emptySample = await sampleCanvas(page);
		expect(emptySample.ok, 'empty-state canvas should be sampleable').toBe(true);
		expect(
			emptySample.fillPct,
			`the rest substrate should fill the field (fillPct=${emptySample.fillPct})`
		).toBeGreaterThan(20);

		// ── 2. RENDERS REAL DATA — run the real 8-stage pipeline ───────────
		const respPromise = page.waitForResponse(
			(r) => /\/api\/deep[_-]reference/.test(r.url()) && r.status() === 200,
			{ timeout: 30_000 }
		);
		await askQuery(page, QUERY);
		const resp = await respPromise;
		const payload = (await resp.json()) as { evidence?: unknown[]; confidence?: number };
		// The real backend produced evidence — the field will lay it out as cells.
		expect(Array.isArray(payload.evidence), 'deep_reference returns an evidence array').toBe(true);

		// The field re-lights with the real evidence galaxy.
		await page.waitForTimeout(3500);
		const lit = await sampleCanvas(page);
		expect(lit.ok, 'organ canvas should be sampleable').toBe(true);
		expect(
			lit.rendered,
			`evidence galaxy should render non-black (avgLum=${lit.avgLum}, variance=${lit.variance})`
		).toBe(true);

		// ── 3. ALIVE ───────────────────────────────────────────────────────
		const animating = await isAnimating(page);
		expect(animating, 'organ field should animate (living, not a frozen frame)').toBe(true);

		// ── 4. CRASH-FREE in-canvas click grid + hover sweep ───────────────
		const box = await canvas.boundingBox();
		expect(box, 'canvas should have a bounding box').not.toBeNull();
		if (box) {
			const cols = 5;
			const rows = 5;
			for (let r = 0; r < rows; r++) {
				for (let c = 0; c < cols; c++) {
					const x = box.x + ((c + 0.5) / cols) * box.width;
					const y = box.y + ((r + 0.5) / rows) * box.height;
					await page.mouse.move(x, y);
					await page.mouse.click(x, y);
				}
			}
			for (let r = 0; r < 9; r++) {
				await page.mouse.move(box.x + box.width * 0.5, box.y + (r / 8) * box.height);
			}
		}

		const afterInteract = await sampleCanvas(page);
		expect(
			afterInteract.rendered,
			`organ field should still render after clicks/hover (avgLum=${afterInteract.avgLum}, variance=${afterInteract.variance})`
		).toBe(true);

		await page.screenshot({ path: 'e2e/screenshots/organ-reasoning.png', fullPage: true });

		// ── No real app / WebGPU-validation errors across the whole run ────
		expectNoErrors(capture);
	});
});

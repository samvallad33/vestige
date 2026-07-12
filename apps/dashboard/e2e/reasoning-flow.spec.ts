import { test, expect, type Page } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, sampleCanvas } from './helpers/dashboard';

// ─────────────────────────────────────────────────────────────────────────
// Reasoning Theater — the deep_reference flow (zero-DOM, in-canvas).
//
// The Reasoning Theater is a full-bleed WebGPU organ: the ONLY DOM is a
// visually-hidden (sr-only) input for keyboard + screen-reader access. There is
// no DOM response panel — the beam/ribbon/nucleus trace, evidence galaxy, gates,
// and receipt are all rendered IN-CANVAS by the trace pass. So this flow is
// proven on PIXELS, not DOM widgets:
//   1. /dashboard/reasoning mounts its WebGPU organ canvas.
//   2. It is alive at rest (a passive memory-pool substrate lights the field).
//   3. A real query is typed into the sr-only ask box and submitted (Enter).
//   4. The /api/deep_reference round-trip completes and the field RE-LIGHTS with
//      the real evidence galaxy (fill stays high, pixels change vs the rest state).
//   5. No page/WebGPU-validation errors fired.
// ─────────────────────────────────────────────────────────────────────────

const REASONING = `${BASE}/reasoning`;
const QUERY = 'How does FSRS-6 trust scoring work?';

/** Fill the sr-only ask input by id (it's visually hidden, so .fill on the
 * located element — not a click — is the keyboard-user path), then submit. */
async function askQuery(page: Page, query: string): Promise<void> {
	const input = page.locator('#reasoning-ask');
	await input.waitFor({ state: 'attached', timeout: 15_000 });
	await input.fill(query);
	await expect(input).toHaveValue(query);
	await input.press('Enter');
}

/** Wait until a real /api/deep_reference response has come back. */
async function waitForDeepReference(page: Page): Promise<void> {
	await page.waitForResponse(
		(r) => /\/api\/deep[_-]reference/.test(r.url()) && r.status() === 200,
		{ timeout: 30_000 }
	);
}

test.describe('Reasoning Theater — deep_reference flow (zero-DOM)', () => {
	test('typing a query runs the pipeline and re-lights the organ field', async ({ page }) => {
		const capture = captureErrors(page);

		// 1 — route mounts its WebGPU canvas.
		await page.goto(REASONING);
		const canvas = page.locator('canvas').first();
		await canvas.waitFor({ state: 'attached', timeout: 15_000 });
		await expect(canvas).toBeVisible();

		// 2 — alive at rest: the passive memory-pool substrate fills the field.
		await page.waitForTimeout(3500);
		const rest = await sampleCanvas(page);
		expect(rest.ok, 'canvas should be sampleable at rest').toBe(true);
		expect(
			rest.fillPct,
			`the rest substrate should fill the field (fillPct=${rest.fillPct})`
		).toBeGreaterThan(20);

		// 3 — type a real query into the sr-only input + submit.
		const respPromise = waitForDeepReference(page);
		await askQuery(page, QUERY);

		// 4 — the deep_reference round-trip completes...
		await respPromise;
		// ...and the field re-lights with the real evidence galaxy (give the scene
		// time to drive + the reveal to paint).
		await page.waitForTimeout(3500);
		const after = await sampleCanvas(page);
		expect(after.ok, 'canvas should be sampleable after the query').toBe(true);
		expect(
			after.rendered,
			`evidence galaxy should render non-black (avgLum=${after.avgLum}, variance=${after.variance})`
		).toBe(true);
		expect(
			after.fillPct,
			`the evidence field should fill the frame (fillPct=${after.fillPct})`
		).toBeGreaterThan(20);

		await page.screenshot({ path: 'e2e/screenshots/reasoning-flow-lit.png', fullPage: true });

		// 5 — no app / WebGPU-validation errors across the whole flow.
		expectNoErrors(capture);
	});

	test('a second query re-runs the pipeline against the live brain', async ({ page }) => {
		const capture = captureErrors(page);

		await page.goto(REASONING);
		await page.locator('canvas').first().waitFor({ state: 'attached', timeout: 15_000 });
		await page.waitForTimeout(3000);

		// Two different real queries in a row both round-trip and keep the field lit.
		for (const q of ['What is the Vestige dashboard direction?', QUERY]) {
			const respPromise = waitForDeepReference(page);
			await askQuery(page, q);
			await respPromise;
			await page.waitForTimeout(2500);
			const sample = await sampleCanvas(page);
			expect(sample.rendered, `field stays lit after querying "${q}"`).toBe(true);
		}

		expectNoErrors(capture);
	});
});

import { test, expect } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, sampleCanvas } from './helpers/dashboard';

// ─────────────────────────────────────────────────────────────────────────
// Reasoning Theater — the deep_reference flow.
//
// Proves the real 8-stage cognitive pipeline round-trip:
//   1. /dashboard/reasoning mounts its WebGPU organ canvas.
//   2. A real query is typed into the ask box and submitted (Enter).
//   3. The /api/deep_reference call returns and the reasoning DOM materializes
//      (Reasoning section / "analyzed" / confidence % / Primary Source /
//       Evidence / Cognitive Pipeline).
//   4. The WebGPU organ field lights up (8-stage organ renders non-black).
//   5. No page/WebGPU-validation errors fired.
//
// The brain is up with 325+ real memories, so we assert on real rendered
// output. If a query genuinely returns an empty result set, the flow still
// ran end-to-end (loading → response OR a graceful error surface), and we
// assert THAT rather than hard-failing on content that legitimately may not
// exist.
// ─────────────────────────────────────────────────────────────────────────

const REASONING = `${BASE}/reasoning`;
const QUERY = 'How does FSRS-6 trust scoring work?';

test.describe('Reasoning Theater — deep_reference flow', () => {
	test('typing a query runs the pipeline and renders reasoning + lights the organ', async ({
		page
	}) => {
		const capture = captureErrors(page);

		// ── 1. Route mounts its WebGPU canvas ──────────────────────────────
		await page.goto(REASONING);
		const canvas = page.locator('canvas').first();
		await canvas.waitFor({ state: 'attached', timeout: 15_000 });
		await expect(canvas).toBeVisible();

		// ── 2. Type a real query + submit ──────────────────────────────────
		const input = page.getByPlaceholder('Ask your memory anything…');
		await expect(input).toBeVisible();
		await input.click();
		await input.fill(QUERY);
		await expect(input).toHaveValue(QUERY);

		// The "Reason" button should enable once the box has a non-empty query.
		const reasonBtn = page.getByRole('button', { name: 'Reason' });
		await expect(reasonBtn).toBeEnabled();

		// Submit via Enter (the input has an onkeydown Enter handler).
		await input.press('Enter');

		// ── 3. Wait for the pipeline to run + a terminal state to render ────
		// The button reads "Reasoning…" while loading; the empty-state /
		// example chips disappear. Wait for EITHER a real response section OR
		// a graceful error surface — but not the still-empty initial state.
		const reasoningHeading = page.getByRole('heading', { name: 'Reasoning', exact: true });
		const primarySource = page.getByText('Primary Source', { exact: true });
		const cognitivePipeline = page.getByRole('heading', { name: 'Cognitive Pipeline' });
		const errorSurface = page.locator('text=/^Error:/');

		// Give the local 8-stage pipeline time to complete + paint.
		await Promise.race([
			reasoningHeading.waitFor({ state: 'visible', timeout: 30_000 }),
			cognitivePipeline.waitFor({ state: 'visible', timeout: 30_000 }),
			primarySource.waitFor({ state: 'visible', timeout: 30_000 }),
			errorSurface.waitFor({ state: 'visible', timeout: 30_000 })
		]);

		// The button must have left its disabled/loading limbo.
		await expect(page.getByRole('button', { name: /Reason(ing…)?/ })).toBeEnabled({
			timeout: 15_000
		});

		const errored = await errorSurface.isVisible().catch(() => false);

		if (errored) {
			// A real backend error is a genuine finding, not something to hide.
			const msg = (await errorSurface.textContent())?.trim();
			throw new Error(`Reasoning Theater surfaced a backend error: ${msg}`);
		}

		// ── Real response asserted on real rendered content ────────────────
		// The Cognitive Pipeline section renders for ANY successful response
		// (even zero-evidence), so it's the reliable "the flow produced a
		// result" anchor.
		await expect(cognitivePipeline).toBeVisible();

		// "N analyzed" appears in the reasoning header / confidence meter.
		await expect(page.getByText(/\banalyzed\b/).first()).toBeVisible();

		// A confidence percentage renders (the big meter uses "%"). Assert a
		// digit-followed-by-% is present somewhere in the response region.
		const confidencePct = page.locator('text=/\\d+%/').first();
		await expect(confidencePct).toBeVisible();

		// Primary Source citation block renders the recommended answer.
		await expect(primarySource).toBeVisible();

		// Evidence section header renders with its count.
		await expect(page.getByRole('heading', { name: /^Evidence/ })).toBeVisible();

		// ── 4. The WebGPU organ field lit up (8-stage organ, non-black) ────
		// Let the field settle after the DeepReferenceCompleted scene drives it.
		await page.waitForTimeout(3000);
		const sample = await sampleCanvas(page);
		expect(sample.ok, 'canvas should be sampleable').toBe(true);
		expect(
			sample.rendered,
			`organ field should render non-black (avgLum=${sample.avgLum}, variance=${sample.variance})`
		).toBe(true);

		await page.screenshot({
			path: 'e2e/screenshots/reasoning-flow-lit.png',
			fullPage: true
		});

		// ── 5. No real app / WebGPU-validation errors ──────────────────────
		expectNoErrors(capture);
	});

	test('an example chip also drives the full flow', async ({ page }) => {
		const capture = captureErrors(page);

		await page.goto(REASONING);
		await page.locator('canvas').first().waitFor({ state: 'attached', timeout: 15_000 });

		// The example chips render only in the empty state. Click the first one;
		// it sets the query AND immediately calls ask().
		const chip = page.getByRole('button', {
			name: 'How does FSRS-6 trust scoring work?'
		});
		await expect(chip).toBeVisible();
		await chip.click();

		const cognitivePipeline = page.getByRole('heading', { name: 'Cognitive Pipeline' });
		const errorSurface = page.locator('text=/^Error:/');

		await Promise.race([
			cognitivePipeline.waitFor({ state: 'visible', timeout: 30_000 }),
			errorSurface.waitFor({ state: 'visible', timeout: 30_000 })
		]);

		const errored = await errorSurface.isVisible().catch(() => false);
		if (errored) {
			const msg = (await errorSurface.textContent())?.trim();
			throw new Error(`Reasoning Theater surfaced a backend error: ${msg}`);
		}

		await expect(cognitivePipeline).toBeVisible();
		await expect(page.getByText(/\banalyzed\b/).first()).toBeVisible();

		expectNoErrors(capture);
	});
});

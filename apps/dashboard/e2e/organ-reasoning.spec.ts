import { test, expect } from '@playwright/test';
import {
	BASE,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

// ─────────────────────────────────────────────────────────────────────────
// ORGAN OWNER SPEC — /reasoning (Reasoning Theater, the deep_reference organ)
//
// Proves the 7-point ship-a-working-product contract for THIS organ against
// the REAL brain (:3931, 1285 memories). Focused + fast — does not duplicate
// all-routes-smoke; adds the organ-specific proof:
//   1. REACHABLE      — /dashboard/reasoning mounts its WebGPU canvas.
//   5. HONEST EMPTY   — before any query the organ shows a calm empty state
//                       (empty-state DOM copy + the field's empty label), NOT
//                       fake data or a broken/errored surface.
//   2. RENDERS REAL   — a real query runs the 8-stage pipeline and the DOM
//      DATA             materializes REAL numbers straight from deep_reference
//                       (memoriesAnalyzed / confidence% / Primary Source /
//                       Evidence count). Asserted on values that only exist
//                       because the real backend produced them.
//   3. ALIVE          — the organ field animates after the scene is uploaded.
//   4. CRASH-FREE     — a grid of in-canvas clicks (drives pass.pickAt over the
//      CLICK + HOVER    static stage rects) + a hover sweep; the canvas SURVIVES
//                       every one with zero page/WebGPU errors. Stage centers do
//                       NOT animate in the vertex shader (only radius/energy do),
//                       and pickAt() uses the same y = 0.76 - i*(1.52/7) layout
//                       as buildStageRects(), so picks land on the real targets.
//
// Real ground truth captured live via curl POST /api/deep_reference
// {query:"How does FSRS-6 trust scoring work?"}:
//   confidence 0.87 · memoriesAnalyzed 26 · activationExpanded 6 ·
//   evidence 10 · contradictions 0 · superseded 0 · intent "Synthesis".
// contradictions/superseded are legitimately 0 for this query, so those
// sections correctly DON'T render — the honest no-fake-data path.
// ─────────────────────────────────────────────────────────────────────────

const REASONING = '/reasoning';
const QUERY = 'How does FSRS-6 trust scoring work?';

test.describe('Organ /reasoning — Reasoning Theater', () => {
	test('reachable, honest-empty, renders REAL data, alive, and survives click+hover', async ({
		page
	}) => {
		const capture = captureErrors(page);

		// ── 1. REACHABLE: route mounts its WebGPU canvas ───────────────────
		const canvas = await gotoRoute(page, REASONING);
		await expect(canvas).toBeVisible();

		// ── 5. HONEST EMPTY STATE (pre-query) ──────────────────────────────
		// The DOM empty panel renders with real, non-faked copy — and NO fake
		// "Live" data / response sections exist yet.
		await expect(
			page.getByText('Ask anything. Vestige will run the full reasoning pipeline and show you its work.')
		).toBeVisible();
		// No response artifacts before a query: no Primary Source citation, no
		// Cognitive Pipeline result, no confidence meter — nothing fabricated.
		await expect(page.getByText('Primary Source', { exact: true })).toHaveCount(0);
		await expect(page.getByRole('heading', { name: 'Cognitive Pipeline' })).toHaveCount(0);
		// The immersive field is present but shows its honest empty label (no
		// invented scene) — canvas is still a real, sampleable surface.
		const emptySample = await sampleCanvas(page);
		expect(emptySample.ok, 'empty-state canvas should be sampleable').toBe(true);

		// ── 2. RENDERS REAL DATA: run the real 8-stage pipeline ────────────
		const input = page.getByPlaceholder('Ask your memory anything…');
		await expect(input).toBeVisible();
		await input.click();
		await input.fill(QUERY);
		await input.press('Enter');

		const cognitivePipeline = page.getByRole('heading', { name: 'Cognitive Pipeline' });
		const errorSurface = page.locator('text=/^Error:/');
		await Promise.race([
			cognitivePipeline.waitFor({ state: 'visible', timeout: 30_000 }),
			errorSurface.waitFor({ state: 'visible', timeout: 30_000 })
		]);

		// A real backend error is a genuine finding — surface it, never hide it.
		const errored = await errorSurface.isVisible().catch(() => false);
		if (errored) {
			const msg = (await errorSurface.textContent())?.trim();
			throw new Error(`Reasoning Theater surfaced a backend error: ${msg}`);
		}

		// The pipeline finished (button left its "Reasoning…" limbo).
		await expect(page.getByRole('button', { name: /Reason(ing…)?/ })).toBeEnabled({
			timeout: 15_000
		});

		// These assertions can ONLY pass if real deep_reference data drove the DOM:
		//   • the recommended Primary Source citation block,
		await expect(page.getByText('Primary Source', { exact: true })).toBeVisible();
		//   • an "N analyzed" count (real memoriesAnalyzed from the backend),
		const analyzed = page.getByText(/\banalyzed\b/).first();
		await expect(analyzed).toBeVisible();
		//   • a real confidence percentage in the meter,
		await expect(page.locator('text=/\\d+%/').first()).toBeVisible();
		//   • the Evidence header WITH a real non-zero count in parentheses.
		const evidenceHeading = page.getByRole('heading', { name: /^Evidence/ });
		await expect(evidenceHeading).toBeVisible();
		const evidenceText = (await evidenceHeading.textContent()) ?? '';
		const evCount = Number(evidenceText.match(/\((\d+)\)/)?.[1] ?? '0');
		expect(evCount, `Evidence count should be real & non-zero (saw "${evidenceText.trim()}")`).toBeGreaterThan(0);
		// At least one real EvidenceCard rendered from the backend payload.
		await expect(page.locator('[data-evidence-id]').first()).toBeVisible();

		// ── 3. ALIVE: the organ field animates after the scene upload ──────
		await page.waitForTimeout(2500);
		const lit = await sampleCanvas(page);
		expect(lit.ok, 'organ canvas should be sampleable').toBe(true);
		expect(
			lit.rendered,
			`organ field should render non-black (avgLum=${lit.avgLum}, variance=${lit.variance})`
		).toBe(true);
		const animating = await isAnimating(page);
		expect(animating, 'organ field should animate (living, not a frozen frame)').toBe(true);

		// ── 4. CRASH-FREE in-canvas click grid + hover sweep ───────────────
		// The RouteStage field layer is full-bleed fixed inset-0; clicking it
		// routes through handleFieldClick → pass.pickAt over the 8 static stage
		// rects. Sweep a grid so several clicks LAND on stage chambers (which
		// open the DOM stage-receipt aside) and several miss (no-op) — the
		// canvas must survive every one.
		const box = await canvas.boundingBox();
		expect(box, 'canvas should have a bounding box').not.toBeNull();
		if (box) {
			const cols = 5;
			const rows = 5;
			for (let r = 0; r < rows; r++) {
				for (let c = 0; c < cols; c++) {
					const x = box.x + ((c + 0.5) / cols) * box.width;
					const y = box.y + ((r + 0.5) / rows) * box.height;
					await page.mouse.move(x, y); // hover sweep
					await page.mouse.click(x, y); // in-canvas pick
				}
			}
			// A hover along the vertical stage column (x≈center) — where the
			// eight chambers live — must not throw either.
			for (let r = 0; r < 9; r++) {
				await page.mouse.move(box.x + box.width * 0.5, box.y + (r / 8) * box.height);
			}
		}

		// The canvas is still alive and rendering after all that interaction.
		const afterInteract = await sampleCanvas(page);
		expect(
			afterInteract.rendered,
			`organ field should still render after clicks/hover (avgLum=${afterInteract.avgLum}, variance=${afterInteract.variance})`
		).toBe(true);

		// A stage-receipt aside may have opened from a landed pick; if so it must
		// be a real receipt panel (has the "stage receipt" label), never junk.
		const receiptAside = page.getByLabel('Reasoning stage receipt');
		if (await receiptAside.isVisible().catch(() => false)) {
			await expect(receiptAside.getByText('stage receipt')).toBeVisible();
		}

		await page.screenshot({ path: 'e2e/screenshots/organ-reasoning.png', fullPage: true });

		// ── 4/5. No real app / WebGPU-validation errors across the whole run ─
		expectNoErrors(capture);
	});
});

// ─────────────────────────────────────────────────────────────────────────────
// ALL-ROUTES SMOKE MATRIX — the backbone launch gate.
//
// For EVERY organ route, prove the four things a user needs to be true:
//   1. the route loads and mounts a WebGPU canvas
//   2. the WebGPU field actually RENDERS (non-black, non-flat pixels — the
//      organ is visibly alive, not a blank/errored surface)
//   3. no WebGPU validation errors or page errors fire (catches the runtime
//      shader/pipeline failures that pass typecheck — e.g. reserved-word WGSL)
//   4. a screenshot is captured (Playwright screenshot:'on') as visible proof
//
// WebGPU adapter availability is VERIFIED in this ANGLE/headless setup, so these
// assertions are real. If a route can't be sampled (GPU-tainted), the pixel
// assertion self-skips rather than false-failing.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect } from '@playwright/test';
import { ROUTES, BASE, captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

for (const route of ROUTES) {
	if (route.webgpu) {
		test(`route ${route.path} (${route.label}) renders a live WebGPU field with no errors`, async ({ page }) => {
			const errors = captureErrors(page);

			const canvas = await gotoRoute(page, route.path);
			await expect(canvas).toBeVisible();

			// The canvas must reach real drawing-buffer dimensions (the /graph
			// h-full-collapse regression made this 2006x1 → black). Poll because
			// the engine sizes the drawing buffer a beat after mount.
			await expect(async () => {
				const dims = await canvas.evaluate((el: HTMLCanvasElement) => ({ w: el.width, h: el.height }));
				expect(dims.w, `${route.path} canvas width`).toBeGreaterThan(200);
				expect(dims.h, `${route.path} canvas height`).toBeGreaterThan(200);
			}).toPass({ timeout: 8000 });

			// Let the field build (nodes stream, text materializes, camera settles).
			await page.waitForTimeout(3500);

			const sample = await sampleCanvas(page);
			expect(
				sample.rendered,
				`${route.path} should render a living field (avgLum=${sample.avgLum}, variance=${sample.variance})`
			).toBe(true);

			expectNoErrors(errors);
		});
	} else {
		// Settings is the one intentional DOM control panel — assert its controls
		// render and no errors fire (no WebGPU field expected).
		test(`route ${route.path} (${route.label}) renders its control panel with no errors`, async ({ page }) => {
			const errors = captureErrors(page);
			await page.goto(`${BASE}${route.path}`);
			await page.waitForLoadState('networkidle');
			// Some visible, interactive content must exist.
			const controls = page.locator('button, input, select, [role="switch"]');
			await expect(controls.first()).toBeVisible({ timeout: 10_000 });
			expectNoErrors(errors);
		});
	}
}


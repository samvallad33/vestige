// ─────────────────────────────────────────────────────────────────────────────
// Cross-Organ Navigation — E2E proof
//
// The all-WebGPU dashboard renders every organ as a full-bleed WebGPU field.
// This spec proves two things the single-organ specs cannot:
//
//   1. BREADTH — a representative subset of organs each mount a canvas, RENDER a
//      non-black living field (sampleCanvas), and fire zero page/WebGPU errors.
//      Each organ gets its own fresh page context so one organ's state can never
//      leak into another's assertion.
//
//   2. SPA CONTINUITY — navigating graph → reasoning → timeline → graph inside
//      ONE page context (no reload) must not accumulate console/page errors.
//      This is the leak detector: WebGPU engine dispose/re-mount bugs (double
//      device, un-freed pipelines, dangling RAF loops, stale event listeners)
//      surface as console errors that pile up across client-side route changes.
//      A fresh page hides these; a persistent one exposes them.
//
// Uses the verified shared harness (helpers/dashboard.ts). sampleCanvas reads the
// real GPU compositor via Playwright screenshot() — never drawImage on the live
// WebGPU surface (returns all-black in this ANGLE/headless setup).
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect } from '@playwright/test';
import {
	BASE,
	WEBGPU_ROUTES,
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
} from './helpers/dashboard';

// A representative cross-section of organs: the graph, the reasoning/audit
// organs, the temporal + feed streams, the raw memory list, the dream cycle, and
// the stats reactor. Broad enough to catch a per-organ regression, focused enough
// to stay fast.
const SUBSET_LABELS = [
	'Graph',
	'Reasoning',
	'Contradictions',
	'Timeline',
	'Feed',
	'Memories',
	'Dreams',
	'Stats',
];

const SUBSET = WEBGPU_ROUTES.filter((r) => SUBSET_LABELS.includes(r.label));

test.describe('Cross-organ navigation — every organ renders alive, no leaks', () => {
	// Sanity: the subset filter actually resolved to the routes we named. Guards
	// against a route being renamed/removed in the shared harness and silently
	// shrinking coverage.
	test('subset resolves to the expected organs', () => {
		expect(SUBSET.map((r) => r.label).sort()).toEqual([...SUBSET_LABELS].sort());
	});

	// BREADTH — one isolated page per organ. Fresh context = zero leftover state.
	for (const route of SUBSET) {
		test(`${route.label} (${route.path}) mounts + renders a living field, no errors`, async ({
			page,
		}) => {
			const capture = captureErrors(page);

			await gotoRoute(page, route.path);
			// Organs build over several seconds (layout settle, first data frames).
			await page.waitForTimeout(3000);

			const sample = await sampleCanvas(page);
			expect(
				sample.rendered,
				`${route.label} field is BLACK/blank — avgLum=${sample.avgLum} variance=${sample.variance} ok=${sample.ok}`,
			).toBe(true);

			const buf = await page.screenshot({ type: 'png' });
			await test.info().attach(`organ-${route.label.replace(/\s+/g, '-').toLowerCase()}.png`, {
				body: buf,
				contentType: 'image/png',
			});

			expectNoErrors(capture);
		});
	}

	// SPA CONTINUITY — one page context, client-side nav across organs and back.
	// Errors accumulate into a single capture; any engine dispose/re-mount leak
	// pushes a console error that survives to the final assertion.
	test('graph → reasoning → timeline → graph in ONE context accumulates no errors', async ({
		page,
	}) => {
		const capture = captureErrors(page);

		// The journey: revisiting /graph at the end forces a full dispose →
		// re-instantiate of the graph engine within the same JS context. If the
		// first mount left anything dangling, the re-mount trips it.
		const journey = ['/graph', '/reasoning', '/timeline', '/graph'];

		for (const path of journey) {
			await gotoRoute(page, path);
			// Let the field boot and run a few frames so a re-mount actually
			// exercises the engine lifecycle, not just the DOM swap.
			await page.waitForTimeout(2500);

			const sample = await sampleCanvas(page);
			expect(
				sample.rendered,
				`after nav to ${path}: field is BLACK — avgLum=${sample.avgLum} variance=${sample.variance}`,
			).toBe(true);

			const buf = await page.screenshot({ type: 'png' });
			await test.info().attach(`journey-${path.replace(/\//g, '')}.png`, {
				body: buf,
				contentType: 'image/png',
			});
		}

		// The whole point: after 4 SPA navigations (including a graph re-mount),
		// NO page errors and NO WebGPU-validation errors should have piled up.
		expectNoErrors(capture);
	});
});

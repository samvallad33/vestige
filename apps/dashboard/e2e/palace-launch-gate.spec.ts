// ─────────────────────────────────────────────────────────────────────────────
// THE LAUNCH GATE — "click any orb, land on a working page, every time."
//
// The Spatial Palace (/dashboard/palace) is a live 3D WebGPU constellation. As of
// the Jul 2026 launch curation it surfaces 9 HERO organs (ORGAN_REGIONS in
// src/lib/observatory/palace-map.ts); the other 10 organs still exist and are
// reachable by direct URL, just not palace-linked. A pointerdown on an orb runs
// pass.pickAt -> goto(`${base}${href}`), diving into that organ.
//
// This one spec is the ship gate, in two parts:
//   1. PALACE LOOP — the palace opens live, a real click-to-enter dive lands on a
//      curated organ, and every one of the 9 curated organs lands on a LIVE,
//      non-black, error-free page (then returns to the palace).
//   2. FULL URL SMOKE — all 20 routes (the 10 hidden organs included) mount a
//      canvas and render a non-black, error-free field by direct URL.
//
// If even one lands dead/black/errored, this spec fails and we do not ship.
//
// Anti-false-green: non-black is asserted on DECODED luminance (avgLum + variance
// from a real compositor screenshot), never the sampleCanvas `rendered` flag
// whose size>20_000 fallback can pass a black field. WebGPU-readback constraint:
// reading the live canvas via Canvas2D returns BLACK in ANGLE/headless, so every
// pixel proof goes through sampleCanvas's screenshot path.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect, type Page } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, sampleCanvas, isAnimating } from './helpers/dashboard';

// The 9 organs the launch palace surfaces (mirrors ORGAN_REGIONS). settings is
// now a full-WebGPU in-canvas console like every other organ (no DOM panel).
const CURATED_ORGANS = [
	'observatory',
	'graph',
	'memories',
	'timeline',
	'feed',
	'explore',
	'reasoning',
	'stats',
	'settings'
];

// The 10 organs hidden from the palace but still reachable by direct URL.
const HIDDEN_ORGANS = [
	'contradictions',
	'blackbox',
	'duplicates',
	'memory-prs',
	'activation',
	'dreams',
	'schedule',
	'importance',
	'patterns',
	'intentions'
];

const ALL_ROUTES = ['palace', ...CURATED_ORGANS, ...HIDDEN_ORGANS];
const CURATED_URL_RE = new RegExp(`/dashboard/(?:${CURATED_ORGANS.join('|')})(?:[/?#]|$)`);

// Some organs build their field over a couple seconds; give every landing a
// generous settle before sampling so slow-mounting organs aren't flaky.
const SETTLE_MS = 3200;
// /reasoning auto-runs a real 8-stage deep_reference on mount and stages its
// evidence galaxy from the response — that round-trip legitimately takes longer
// than a static field, so it gets an extended settle before we sample.
const SLOW_ORGANS: Record<string, number> = { reasoning: 12_000 };

// The liveness bar: the mission requires every organ to be a full-bleed MOVING
// field (fill >= 35% at full res). The gate samples a 96x96 decode of a real
// compositor screenshot; a genuinely-filled field (60-89% at full res) reads
// >= 35% here, while a sparse text-on-black organ reads < 18%. 30% cleanly
// separates alive from sparse with headroom for settle/downsample variance.
const MIN_FILL_PCT = 30;

/** Assert a genuinely-lit, MOVING, error-free field on the current route. */
async function assertLiveField(
	page: Page,
	label: string
): Promise<{ avgLum: number; variance: number; fillPct: number }> {
	const canvas = page.locator('canvas').first();
	await canvas.waitFor({ state: 'attached', timeout: 15_000 });
	// Slow organs (reasoning's deep_reference round-trip) settle longer.
	const slow = Object.entries(SLOW_ORGANS).find(([name]) => label.includes(name));
	await page.waitForTimeout(slow ? slow[1] : SETTLE_MS);
	const s = await sampleCanvas(page);
	expect(s.avgLum, `${label} avgLum (field must not be black)`).toBeGreaterThan(3);
	expect(s.variance, `${label} variance (field must have structure)`).toBeGreaterThan(6);
	expect(s.fillPct, `${label} fill% (field must be full-bleed, not text-on-black)`).toBeGreaterThan(
		MIN_FILL_PCT
	);
	return { avgLum: s.avgLum, variance: s.variance, fillPct: s.fillPct };
}

/** Probe a grid across the orbiting constellation until a pointerdown dives. */
async function clickToEnter(page: Page): Promise<string | null> {
	const canvas = page.locator('canvas').first();
	const box = await canvas.boundingBox();
	if (!box) return null;
	const cx = box.x + box.width / 2;
	const cy = box.y + box.height / 2;
	const offsets: Array<[number, number]> = [
		[0, 0],
		[0.08, 0], [-0.08, 0], [0, 0.08], [0, -0.08],
		[0.16, 0.1], [-0.16, 0.1], [0.16, -0.1], [-0.16, -0.1],
		[0.28, 0], [-0.28, 0], [0, 0.24], [0, -0.24],
		[0.34, 0.18], [-0.34, 0.18], [0.34, -0.18], [-0.34, -0.18]
	];
	for (let pass = 0; pass < 3; pass++) {
		for (const [ox, oy] of offsets) {
			const x = cx + ox * (box.width / 2);
			const y = cy + oy * (box.height / 2);
			await page.mouse.move(x, y);
			await page.mouse.down();
			await page.mouse.up();
			try {
				await page.waitForURL(CURATED_URL_RE, { timeout: 900 });
				return page.url();
			} catch {
				/* missed every orb at this probe point — keep spiralling */
			}
		}
		await page.waitForTimeout(1200);
	}
	return null;
}

test.describe('LAUNCH GATE — palace loop', () => {
	test('palace opens live and a real click-to-enter dive lands on a curated organ', async ({ page }) => {
		const errors = captureErrors(page);

		await page.goto(`${BASE}/palace`, { waitUntil: 'networkidle' });
		await assertLiveField(page, 'palace');
		expect(await isAnimating(page, 800), 'palace constellation should be orbiting').toBe(true);

		const entered = await clickToEnter(page);
		expect(entered, 'a pointerdown on an orb should dive into a curated organ').not.toBeNull();
		expect(entered!, `landed URL should be a curated organ: ${entered}`).toMatch(CURATED_URL_RE);

		await assertLiveField(page, `dived organ ${entered}`);
		expectNoErrors(errors);
	});

	test('every curated organ lands on a live, error-free page and returns to the palace', async ({ page }) => {
		// The tour dives 9 organs sequentially, each with a full settle + palace
		// return, so it legitimately runs longer than the 60s per-test default.
		test.setTimeout(180_000);
		const errors = captureErrors(page);

		await page.goto(`${BASE}/palace`, { waitUntil: 'networkidle' });
		await assertLiveField(page, 'palace hub');

		const report: Array<{ path: string; avgLum: number; variance: number; fillPct: number }> = [];

		for (const path of CURATED_ORGANS) {
			// Dive — exactly what the palace pick handler runs: goto(base + href).
			await page.goto(`${BASE}/${path}`, { waitUntil: 'networkidle' });
			const { avgLum, variance, fillPct } = await assertLiveField(page, `organ /${path}`);
			report.push({ path, avgLum, variance, fillPct });
			await test.info().attach(`organ-${path}.png`, {
				body: await page.screenshot(),
				contentType: 'image/png'
			});

			// Return to the palace between dives (the real orbit loop).
			await page.goto(`${BASE}/palace`, { waitUntil: 'networkidle' });
			await page.waitForTimeout(400);
		}

		await test.info().attach('launch-gate-report.json', {
			body: Buffer.from(JSON.stringify(report, null, 2)),
			contentType: 'application/json'
		});

		// Zero page / WebGPU-validation errors across the ENTIRE curated tour.
		expectNoErrors(errors);
	});
});

test.describe('LAUNCH GATE — full URL smoke (all 20 routes)', () => {
	for (const path of ALL_ROUTES) {
		test(`/${path} mounts a live, full-bleed, MOVING field`, async ({ page }) => {
			const errors = captureErrors(page);
			await page.goto(`${BASE}/${path}`, { waitUntil: 'networkidle' });
			await assertLiveField(page, `/${path}`);
			// Every organ must MOVE (living field, not a frozen frame). palace +
			// timeline + the 17 field organs all orbit deterministically at rest.
			expect(await isAnimating(page, 800), `/${path} field should be animating`).toBe(true);
			expectNoErrors(errors);
		});
	}
});

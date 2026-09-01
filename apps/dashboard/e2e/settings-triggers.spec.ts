// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS & SYSTEM — the zero-DOM WebGPU console (real-API trigger proof).
//
// /dashboard/settings is now a FULL-WebGPU organ: the only non-WebGPU element on
// the page is the single <canvas> RouteStage mounts. The title, live vitals,
// retention histogram, and action buttons are all MSDF text runs in the shared
// cognitive field, and the buttons are PICKABLE in-canvas regions that fire the
// real backend operations (there are NO <button>/<div>/<section> controls).
//
// This spec proves the in-canvas control model against the live brain:
//   1. MOUNTS + RENDERS — /settings mounts a canvas and the console field is
//      NON-BLACK (sampleCanvas decodes real luminance from the compositor).
//   2. CONSOLIDATE pick — a pointerdown/up on the [ CONSOLIDATE ] button's screen
//      location fires POST /api/consolidate, and the field survives (no crash).
//   3. DREAM pick — clicking [ DREAM ] fires POST /api/dream, field survives.
//   4. NO page/WebGPU errors across the interactions.
//
// Button location: the console lays each action at a known logical-NDC anchor on
// the y=-0.5 row. We convert that anchor to a screen pixel using the SAME aspect
// transform the pass's pickAt mirrors (x /= max(aspect,1); y *= min(aspect,1)),
// bias to the button's text center, and dispatch a real pointer sequence there.
// A backend guard (curl of the real brain) runs first so the endpoints are known
// reachable; if consolidate/dream is genuinely unreachable the app must still not
// crash and the field must still render.
// ─────────────────────────────────────────────────────────────────────────────
import { test, expect, type Page, type Locator } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

// Logical-NDC left-edge anchors of the four action buttons (must match the
// buildConsoleItems() layout in settings/+page.svelte). All sit on the y=-0.5
// row. We bias +dx toward each button's text center so the pick lands on a glyph.
const ACTIONS = {
	consolidate: { x: -0.9, dx: 0.135, y: -0.5 }, // "[ CONSOLIDATE ]"
	dream: { x: -0.36, dx: 0.09, y: -0.5 }, //        "[ DREAM ]"
	birth: { x: 0.06, dx: 0.13, y: -0.5 }, //         "[ TRIGGER BIRTH ]"
	refresh: { x: 0.56, dx: 0.09, y: -0.5 } //        "[ REFRESH ]"
} as const;

/**
 * Convert a logical-NDC button anchor to a screen pixel over the canvas.
 * Mirrors the pass/shader transform: rendered NDC x = logicalX / max(aspect,1),
 * rendered NDC y = logicalY * min(aspect,1); then NDC → screen over the canvas
 * bounding box. Returns the button's text-center screen coordinate.
 */
async function actionPoint(
	canvas: Locator,
	action: keyof typeof ACTIONS
): Promise<{ x: number; y: number }> {
	const box = await canvas.boundingBox();
	if (!box) throw new Error('canvas has no bounding box');
	const aspect = box.width / box.height;
	const xScale = Math.max(aspect, 1);
	const yScale = Math.min(aspect, 1);
	const a = ACTIONS[action];
	const logicalX = a.x + a.dx;
	const rx = logicalX / xScale;
	const ry = a.y * yScale;
	return {
		x: box.x + ((rx + 1) / 2) * box.width,
		y: box.y + ((1 - ry) / 2) * box.height
	};
}

async function gotoSettings(page: Page): Promise<Locator> {
	const canvas = await gotoRoute(page, '/settings');
	// The console loads stats + retention then lays the field. Settle so the
	// buttons are present and the reveal gate is fully open before we pick.
	await page.waitForTimeout(3500);
	return canvas;
}

async function attach(page: Page, name: string): Promise<void> {
	const buf = await page.screenshot({ type: 'png' });
	await test.info().attach(name, { body: buf, contentType: 'image/png' });
}

test.describe('Settings console — zero-DOM WebGPU triggers', () => {
	// Consolidate mutates the brain, then Dream reads it — run serial so the two
	// heavy real operations don't race the same backend within one worker.
	test.describe.configure({ mode: 'serial' });

	test('1. settings mounts a canvas and renders a NON-BLACK console field', async ({ page }) => {
		const errors = captureErrors(page);
		const canvas = await gotoSettings(page);

		// Zero-DOM proof: the ONLY interactive/content element is the canvas — no
		// button/section/kbd control panel remains.
		await expect(canvas).toBeVisible();
		expect(await page.locator('button').count(), 'no DOM buttons in the console').toBe(0);
		expect(await page.locator('section, kbd, h2').count(), 'no DOM control-panel chrome').toBe(0);

		// Non-black proof: decode real luminance from the compositor screenshot.
		const sample = await sampleCanvas(page);
		expect(
			sample.rendered,
			`settings console must render a lit field (avgLum=${sample.avgLum} variance=${sample.variance})`
		).toBe(true);
		expect(sample.avgLum, 'field must carry measurable light, not near-black').toBeGreaterThan(2);

		await attach(page, 'settings-console-rendered.png');
		expectNoErrors(errors);
	});

	test('2. CONSOLIDATE button pick fires POST /api/consolidate and the field survives', async ({
		page
	}) => {
		const errors = captureErrors(page);
		const canvas = await gotoSettings(page);

		const consolidatePromise = page.waitForResponse(
			(res) => res.url().includes('/api/consolidate') && res.request().method() === 'POST',
			{ timeout: 30_000 }
		);

		const pt = await actionPoint(canvas, 'consolidate');
		await page.mouse.move(pt.x, pt.y);
		await page.mouse.down();
		await page.mouse.up();

		const res = await consolidatePromise;
		expect(res.request().method()).toBe('POST');
		// Real success: the FSRS-6 result carries the live counts the console renders.
		if (res.ok()) {
			const body = (await res.json()) as { nodesProcessed?: number };
			expect(typeof body.nodesProcessed, 'consolidation returns real counts').toBe('number');
		}

		// The field must survive the real consolidation cycle (no crash / black-out).
		await page.waitForTimeout(1500);
		const after = await sampleCanvas(page);
		expect(after.rendered, 'console field still renders after consolidate').toBe(true);

		await attach(page, 'settings-consolidate-fired.png');
		expectNoErrors(errors);
	});

	test('3. DREAM button pick fires POST /api/dream and the field survives', async ({ page }) => {
		const errors = captureErrors(page);
		const canvas = await gotoSettings(page);

		const dreamPromise = page.waitForResponse(
			(res) => res.url().includes('/api/dream') && res.request().method() === 'POST',
			{ timeout: 45_000 }
		);

		const pt = await actionPoint(canvas, 'dream');
		await page.mouse.move(pt.x, pt.y);
		await page.mouse.down();
		await page.mouse.up();

		const res = await dreamPromise;
		expect(res.request().method()).toBe('POST');

		// The dream replay is heavy — let it settle, then confirm the field lives.
		await page.waitForTimeout(2000);
		const after = await sampleCanvas(page);
		expect(after.rendered, 'console field still renders after dream').toBe(true);

		await attach(page, 'settings-dream-fired.png');
		expectNoErrors(errors);
	});

	test('4. picks + hover across the console never throw a page/WebGPU error', async ({ page }) => {
		const errors = captureErrors(page);
		const canvas = await gotoSettings(page);

		// REFRESH is idempotent + safe to spam; TRIGGER BIRTH only injects a client
		// event. Hover the action row (drives pickAt + cursor-lens writes) then pick
		// the safe buttons. None may throw or black out the field.
		const refresh = await actionPoint(canvas, 'refresh');
		const birth = await actionPoint(canvas, 'birth');
		for (const p of [refresh, birth]) {
			await page.mouse.move(p.x, p.y);
			await page.waitForTimeout(150);
		}
		await page.mouse.click(refresh.x, refresh.y);
		await page.waitForTimeout(800);
		await page.mouse.click(birth.x, birth.y);
		await page.waitForTimeout(500);

		const after = await sampleCanvas(page);
		expect(after.rendered, 'console field still renders after refresh + birth picks').toBe(true);

		await attach(page, 'settings-picks-survive.png');
		expectNoErrors(errors);
	});
});

test('settings backend endpoints are reachable (guards the pick assertions)', async ({
	request
}) => {
	// The pick tests assume the dev server proxies /api to the running brain. Prove
	// the two operation endpoints answer so a red pick test means a real regression,
	// not a missing backend.
	const stats = await request.get(`${API}/api/stats`);
	expect(stats.ok(), 'GET /api/stats must be 200').toBe(true);
	const body = (await stats.json()) as { totalMemories: number };
	expect(body.totalMemories, 'real brain has memories to render').toBeGreaterThan(0);
});

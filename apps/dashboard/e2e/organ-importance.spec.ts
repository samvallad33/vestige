// Browser acceptance against real records in an owned disposable store.
import { test, expect } from '@playwright/test';
import {
	captureErrors,
	expectNoErrors,
	gotoRoute,
	sampleCanvas,
	isAnimating
} from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

test('importance organ mounts a canvas and renders the REAL scored field', async ({
	page,
	request
}) => {
	// 2 (real data, part A) — curl the real brain FIRST: there must be real
	// memories carrying the fields the field lays out (content, id, retention).
	const memRes = await request.get(`${API}/api/memories?limit=36`);
	expect(memRes.ok(), 'GET /api/memories must be 200').toBe(true);
	const memPayload = (await memRes.json()) as {
		memories: { id: string; content: string; retentionStrength: number }[];
		total: number;
	};
	expect(Array.isArray(memPayload.memories)).toBe(true);
	expect(
		memPayload.memories.length,
		'real brain should have memories to score + render'
	).toBeGreaterThan(0);
	const first = memPayload.memories[0];
	expect(typeof first.id).toBe('string');
	expect(typeof first.content).toBe('string');
	expect(typeof first.retentionStrength).toBe('number');

	// 2 (real data, part B) — the organ's own data source: POST /api/importance
	// must return a REAL ImportanceScore for real content (not a mock/fake shape).
	const impRes = await request.post(`${API}/api/importance`, {
		data: { content: first.content }
	});
	expect(impRes.ok(), 'POST /api/importance must be 200').toBe(true);
	const score = (await impRes.json()) as {
		channels: { novelty: number; arousal: number; reward: number; attention: number };
		composite: number;
		recommendation: string;
	};
	expect(score.channels, 'importance score must carry all 4 channels').toEqual(
		expect.objectContaining({
			novelty: expect.any(Number),
			arousal: expect.any(Number),
			reward: expect.any(Number),
			attention: expect.any(Number)
		})
	);
	expect(typeof score.composite).toBe('number');
	expect(score.composite).toBeGreaterThanOrEqual(0);
	expect(score.composite).toBeLessThanOrEqual(1);
	expect(typeof score.recommendation).toBe('string');

	const errors = captureErrors(page);
	await gotoRoute(page, '/importance');
	// The field lists memories, awaits a scoring round-trip per row, then reveals
	// row-by-row over ~2s. Give it generous settle time (scoring is N requests).
	await page.waitForTimeout(5000);

	// 1 + 2 — the canvas renders a non-black field (the real scored rows lit it up).
	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`importance field should render real scored data (avgLum=${sample.avgLum} variance=${sample.variance})`
	).toBe(true);

	expectNoErrors(errors);
});

test('importance field is ALIVE at idle (no interaction required)', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/importance');
	await page.waitForTimeout(5000);

	const sample = await sampleCanvas(page);
	expect(
		sample.rendered,
		`field must render before checking motion (avgLum=${sample.avgLum})`
	).toBe(true);

	// Per-glyph time wobble + reveal + the recall-path scene animate continuously.
	// Under full-suite GPU load two adjacent frames can hash-match, so retry a few
	// windows — a frozen field never moves across ANY window, an alive one does.
	let moved = false;
	for (let i = 0; i < 4 && !moved; i++) {
		moved = await isAnimating(page, 700);
	}
	expect(moved, 'importance field should animate at idle').toBe(true);

	expectNoErrors(errors);
});

test('selection is non-mutating; explicit promotion preserves the scored memory', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/importance');
	const rows = page.locator('div[role="button"]').filter({ has: page.getByRole('button', { name: 'Promote', exact: true }) });
	await expect(rows.first()).toBeVisible();
	const count = await rows.count();
	const row = rows.first();
	const snippet = await row.locator('.truncate').textContent();
	const promotes: string[] = [];
	page.on('request', r => {
		if (r.method() === 'POST' && /\/promote$/.test(r.url())) promotes.push(r.url());
	});
	await row.focus();
	await row.press('Enter');
	await expect(page.getByRole('button', { name: 'Close', exact: true })).toBeVisible();
	expect(promotes).toHaveLength(0);
	const response = page.waitForResponse(r => /\/promote$/.test(r.url()) && r.request().method() === 'POST');
	await row.getByRole('button', { name: 'Promote', exact: true }).click();
	const result = await response;
	expect(result.ok()).toBe(true);
	const payload = await result.json();
	expect(payload.promoted).toBe(true);
	expect(payload.retentionStrength).toEqual(expect.any(Number));
	expect(promotes).toHaveLength(1);
	await expect(rows).toHaveCount(count);
	await expect(row).toContainText(snippet!);
	await expect(page.getByText("Couldn't score importance", { exact: true })).toHaveCount(0);
	await page.getByRole('button', { name: 'Close', exact: true }).click();
	await row.press('Enter');
	await expect(page.getByRole('button', { name: 'Close', exact: true })).toBeVisible();
	expect((await sampleCanvas(page)).rendered).toBe(true);
	expectNoErrors(errors);
});

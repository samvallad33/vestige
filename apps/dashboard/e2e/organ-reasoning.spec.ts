import { test, expect } from '@playwright/test';
import { captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

test('reasoning displays only evidence returned by its real run and links its receipt', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/reasoning');
	const input = page.locator('#memory-question');
	await expect(input).toBeVisible();
	// onMount focuses this input: wait for hydration before typing into SSR markup.
	await expect(input).toBeFocused();
	await expect(input).toHaveValue('');
	await expect(page.locator('.evidence-list article')).toHaveCount(0);
	await expect(page.locator('.run-id')).toHaveCount(0);
	const response = page.waitForResponse(r => /\/api\/deep[_-]reference/.test(r.url()) && r.request().method() === 'POST');
	await input.fill('browser-fixture timeout milliseconds');
	await input.press('Enter');
	const result = await response;
	expect(result.ok()).toBe(true);
	const payload = await result.json();
	expect(Array.isArray(payload.evidence)).toBe(true);
	expect(payload.evidence.length).toBeGreaterThan(0);
	await expect(page.locator('.evidence-list article')).toHaveCount(payload.evidence.length);
	for (const evidence of payload.evidence) {
		await expect(page.locator('.evidence-list code').filter({ hasText: evidence.id })).toBeVisible();
	}
	const run = payload.runId ?? payload.run_id;
	const receipt = payload.receiptId ?? payload.receipt_id;
	expect(run).toEqual(expect.any(String));
	expect(receipt).toEqual(expect.any(String));
	await expect(page.locator('.run-id')).toContainText(run);
	await expect(page.getByRole('link', { name: /Open this run in Black Box/ })).toHaveAttribute('href', new RegExp(encodeURIComponent(run)));
	await expect(page.getByRole('link', { name: /Open exact receipt/ })).toHaveAttribute('href', new RegExp(encodeURIComponent(receipt)));
	expect((await sampleCanvas(page)).rendered).toBe(true);
	await page.getByRole('button', { name: 'New question', exact: true }).click();
	await expect(input).toHaveValue('');
	await expect(page.locator('.evidence-list article')).toHaveCount(0);
	await expect(page.locator('.run-id')).toHaveCount(0);
    const requestedIds: string[] = [];
    page.on('request', request => {
        const path = new URL(request.url()).pathname;
        const match = path.match(/^\/api\/memories\/([^/]+)$/);
        if (match) requestedIds.push(decodeURIComponent(match[1]));
    });
    await page.goto(`/dashboard/graph?run=${encodeURIComponent(run)}&receipt=${encodeURIComponent(receipt)}`);
    await expect(page.locator('.receipt-seal code')).toHaveText(receipt);
    await page.waitForLoadState('networkidle');
    expect(requestedIds.length).toBeGreaterThan(0);
    expect(requestedIds.every(id => /^[0-9a-f-]{36}$/i.test(id))).toBe(true);
    expectNoErrors(errors);
});

test('reasoning failure offers retry without fabricating evidence or receipt links', async ({ page }) => {
	await page.route('**/api/deep_reference', route => route.fulfill({ status: 503, json: { error: 'fixture unavailable' } }));
	await gotoRoute(page, '/reasoning');
	await expect(page.locator('#memory-question')).toBeFocused();
	await page.locator('#memory-question').fill('browser fixture');
	await page.getByRole('button', { name: 'Run replay', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Try again', exact: true })).toBeVisible();
	await expect(page.locator('.evidence-list article, .receipt-seal, .run-id')).toHaveCount(0);
	await expect(page.getByRole('link', { name: /Open exact receipt/ })).toHaveCount(0);
});

test('reasoning empty result keeps trace identity but does not claim a receipt', async ({ page }) => {
	await page.route('**/api/deep_reference', route => route.fulfill({ json: {
		status: 'no_memories', evidence: [], runId: 'empty-fixture-run', receiptId: null,
	} }));
	await gotoRoute(page, '/reasoning');
	await expect(page.locator('#memory-question')).toBeFocused();
	await page.locator('#memory-question').fill('empty fixture');
	await page.getByRole('button', { name: 'Run replay', exact: true }).click();
	await expect(page.locator('.run-id')).toContainText('empty-fixture-run');
	await expect(page.locator('.empty-evidence')).toBeVisible();
	await expect(page.getByRole('link', { name: /Open exact receipt/ })).toHaveCount(0);
	await expect(page.locator('.evidence-list article, .receipt-seal')).toHaveCount(0);
});

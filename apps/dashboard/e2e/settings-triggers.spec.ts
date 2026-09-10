import { test, expect } from '@playwright/test';
import { captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

// These operations affect only the disposable store owned by run-dashboard-e2e.py.
test('settings exposes explicit maintenance controls and a lit field', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/settings');
	await expect(page.getByRole('heading', { name: 'Keep the memory system alive.' })).toBeVisible();
	await expect(page.getByRole('button', { name: /Consolidate memory/ })).toBeEnabled();
	await expect(page.getByRole('button', { name: /Run dream cycle/ })).toBeEnabled();
	expect((await sampleCanvas(page)).rendered).toBe(true);
	expectNoErrors(errors);
});

for (const [label, endpoint] of [['Consolidate memory', 'consolidate'], ['Run dream cycle', 'dream']] as const) {
	test(`explicit ${label} action calls the backend and records its result`, async ({ page }) => {
		const errors = captureErrors(page);
		await gotoRoute(page, '/settings');
		const response = page.waitForResponse(r => r.url().endsWith(`/api/${endpoint}`) && r.request().method() === 'POST');
		await page.getByRole('button', { name: new RegExp(label) }).click();
		const result = await response;
		expect(result.ok()).toBe(true);
		const payload = await result.json();
		if (endpoint === 'consolidate') expect(payload.nodesProcessed).toEqual(expect.any(Number));
		await expect(page.locator('.operation-receipt output')).not.toBeEmpty();
		await expect(page.getByRole('button', { name: new RegExp(label) })).toBeEnabled();
		expect((await sampleCanvas(page)).rendered).toBe(true);
		expectNoErrors(errors);
	});
}

test('refresh reloads vitals without running maintenance', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/settings');
	const mutations: string[] = [];
	page.on('request', r => { if (r.method() === 'POST') mutations.push(r.url()); });
	const response = page.waitForResponse(r => r.url().endsWith('/api/stats'));
	await page.getByRole('button', { name: 'Refresh live vitals', exact: true }).click();
	expect((await response).ok()).toBe(true);
	await expect(page.getByRole('button', { name: 'Refresh live vitals', exact: true })).toBeEnabled();
	expect(mutations).toEqual([]);
	expectNoErrors(errors);
});

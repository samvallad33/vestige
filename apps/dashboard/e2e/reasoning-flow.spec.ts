import { test, expect, type Page } from '@playwright/test';
import { captureErrors, expectNoErrors, gotoRoute, sampleCanvas } from './helpers/dashboard';

async function ask(page: Page, query: string) {
	const input = page.locator('#memory-question');
	await input.fill(query);
	await expect(page.getByRole('button', { name: 'Run replay', exact: true })).toBeEnabled();
	const response = page.waitForResponse(r => /\/api\/deep[_-]reference/.test(r.url()) && r.request().method() === 'POST');
	await input.press('Enter');
	const result = await response;
	expect(result.ok()).toBe(true);
	const payload = await result.json();
	await expect(page.locator('.run-id')).toContainText(payload.runId);
	return payload;
}

test('keyboard submission runs real retrieval and renders its evidence', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/reasoning');
	await expect(page.locator('#memory-question')).toBeFocused();
	const payload = await ask(page, 'browser-fixture timeout milliseconds');
	expect(payload.evidence.length).toBeGreaterThan(0);
	await expect(page.locator('.evidence-list article')).toHaveCount(payload.evidence.length);
	expect((await sampleCanvas(page)).rendered).toBe(true);
	expectNoErrors(errors);
});

test('successive questions keep independent durable run and receipt identities', async ({ page }) => {
	const errors = captureErrors(page);
	await gotoRoute(page, '/reasoning');
	await expect(page.locator('#memory-question')).toBeFocused();
	const first = await ask(page, 'browser-fixture timeout');
	const second = await ask(page, 'Vestige memory milliseconds');
	expect(second.runId).not.toBe(first.runId);
	expect(second.receiptId).not.toBe(first.receiptId);
	for (const payload of [first, second]) {
		const receipt = await page.request.get(`/api/receipts/${encodeURIComponent(payload.receiptId)}`);
		expect(receipt.ok()).toBe(true);
		const stored = await receipt.json();
		expect(stored.retrieved).toEqual(payload.evidence.map((e: { id: string }) => e.id));
	}
	await expect(page.getByRole('link', { name: /Open exact receipt/ })).toHaveAttribute('href', new RegExp(encodeURIComponent(second.receiptId)));
	expectNoErrors(errors);
});

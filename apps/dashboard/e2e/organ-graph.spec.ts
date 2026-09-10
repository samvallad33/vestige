// The graph route is the receipt-bound Witness view. Assert current evidence
// identity and user controls, not the removed whole-corpus stats/detail panel.
import { test, expect } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors, sampleCanvas } from './helpers/dashboard';

const API = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';
const RUN = 'v3-browser-fixture';

test('Witness binds its displayed seal to the selected real run and survives replay', async ({ page, request }) => {
	const response = await request.get(`${API}/api/receipts?run=${RUN}&limit=24`);
	expect(response.ok()).toBe(true);
	const { receipts } = await response.json();
	expect(receipts.length, 'disposable fixture must produce a real receipt').toBeGreaterThan(0);
	const receipt = receipts[0];
	expect(receipt.retrieved.length).toBeGreaterThan(0);
	const errors = captureErrors(page);
	await page.goto(`${BASE}/graph?run=${RUN}&receipt=${encodeURIComponent(receipt.receipt_id)}`);
	await expect(page.getByTitle(`Open run ${RUN}`)).toHaveClass(/active/);
	await expect(page.locator('.receipt-seal code')).toHaveText(receipt.receipt_id);
	await expect(page.getByRole('button', { name: 'Replay evidence' })).toBeEnabled();
	await page.getByRole('button', { name: 'Replay evidence' }).click();
	const slider = page.getByRole('slider', { name: 'Trace event position' });
	await slider.press('Home');
	await slider.press('ArrowRight');
	await expect(slider).toHaveValue('0.01');
	await page.waitForTimeout(1000);
	expect((await sampleCanvas(page)).rendered).toBe(true);
	await expect(page.locator('.receipt-seal code')).toHaveText(receipt.receipt_id);
	expectNoErrors(errors);
	await page.getByRole('button', { name: 'Open Black Box' }).click();
	await expect(page).toHaveURL(new RegExp(`/blackbox\\?run=${RUN}`));
});

test('Witness empty run list never substitutes corpus memories for receipt evidence', async ({ page }) => {
	await page.route('**/api/traces?*', route => route.fulfill({ json: { runs: [] } }));
	const errors = captureErrors(page);
	await page.goto(`${BASE}/graph`);
	await expect(page.getByText('No Black Box run has been recorded locally yet.')).toBeVisible();
	await expect(page.locator('.receipt-seal, .specimen')).toHaveCount(0);
	await expect(page.getByRole('button', { name: 'Replay evidence' })).toBeDisabled();
	await expect(page.getByRole('button', { name: 'Open Black Box' })).toBeDisabled();
	expect((await sampleCanvas(page)).rendered).toBe(true);
	expectNoErrors(errors);
});

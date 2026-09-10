import { expect, test } from '@playwright/test';
import { gotoRoute } from './helpers/dashboard';

test('automatic default and opt-in changes persist without deciding historical proposals', async ({ page }) => {
    const before = await (await page.request.get('/api/memory-prs?limit=100')).json();
    expect(before.mode).toBe('fast');
    await gotoRoute(page, '/memory-prs');
    const automatic = page.getByRole('button', { name: 'Automatic (default)', exact: true });
    await expect(automatic).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByText('New memories apply immediately. No approval steps or waiting.')).toBeVisible();
    await page.getByRole('button', { name: 'Review risky changes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Review risky changes', exact: true })).toHaveAttribute('aria-pressed', 'true');
    await page.reload();
    await expect(page.getByRole('button', { name: 'Review risky changes', exact: true })).toHaveAttribute('aria-pressed', 'true');
    await automatic.click();
    await expect(automatic).toHaveAttribute('aria-pressed', 'true');
    const after = await (await page.request.get('/api/memory-prs?limit=100')).json();
    expect(after.mode).toBe('fast');
    expect(after.prs.map((pr: { id: string; status: string }) => [pr.id, pr.status])).toEqual(before.prs.map((pr: { id: string; status: string }) => [pr.id, pr.status]));
});

test('failed mode save preserves the active setting and allows retry', async ({ page }) => {
    await gotoRoute(page, '/memory-prs');
    const automatic = page.getByRole('button', { name: 'Automatic (default)', exact: true });
    await expect(automatic).toHaveAttribute('aria-pressed', 'true');
    await page.route('**/api/memory-prs/mode', route => route.fulfill({ status: 500, body: 'Could not save settings' }));
    await page.getByRole('button', { name: 'Review every change', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(automatic).toHaveAttribute('aria-pressed', 'true');
    await expect(automatic).toBeEnabled();
});

test('empty automatic mode has no approval call to action', async ({ page }) => {
    await page.route('**/api/memory-prs?*', route => route.fulfill({ json: { prs: [], total: 0, pendingCount: 0, mode: 'fast' } }));
    await gotoRoute(page, '/memory-prs');
    await expect(page.getByText('Memory is automatic. Nothing to approve.')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Approve', exact: true })).toHaveCount(0);
});

test('automatic controls clear the navigation dock on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await gotoRoute(page, '/memory-prs');
    const automatic = page.getByRole('button', { name: 'Automatic (default)', exact: true });
    await expect(automatic).toBeVisible();
    for (const label of ['Automatic (default)', 'Review risky changes', 'Review every change']) {
        const box = await page.getByRole('button', { name: label, exact: true }).boundingBox();
        expect(box).not.toBeNull();
        expect(box!.x).toBeGreaterThanOrEqual(72);
        expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    }
    await automatic.click();
    await expect(automatic).toHaveAttribute('aria-pressed', 'true');
    expect(await page.getByText('Automatic', { exact: true }).evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
});

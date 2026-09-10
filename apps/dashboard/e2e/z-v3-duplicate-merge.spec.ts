import { expect, test } from '@playwright/test';
import { gotoRoute } from './helpers/dashboard';

test('duplicate preview recovers from an apply failure and persists a reversible merge', async ({ page }) => {
    await gotoRoute(page, '/duplicates');
    const preview = page.getByRole('button', { name: 'Preview a reversible merge', exact: true }).first();
    await expect(preview).toBeEnabled();
    const planned = page.waitForResponse(r => r.url().endsWith('/api/duplicates/plan') && r.request().method() === 'POST');
    await preview.click();
    const response = await planned;
    expect(response.ok()).toBe(true);
    const plan = await response.json();
    expect(plan.planId).toEqual(expect.any(String));
    await expect(page.getByText('Merge preview · nothing written yet').first()).toBeVisible();

    await page.route('**/api/duplicates/apply', route => route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: 'Synthetic apply failure; retry is available' }) }), { times: 1 });
    await page.getByRole('button', { name: 'Apply merge', exact: true }).first().click();
    await expect(page.getByRole('alert').filter({ hasText: 'Synthetic apply failure' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Apply merge', exact: true }).first()).toBeEnabled();

    const applied = page.waitForResponse(r => r.url().endsWith('/api/duplicates/apply') && r.request().method() === 'POST');
    await page.getByRole('button', { name: 'Apply merge', exact: true }).first().click();
    const appliedResponse = await applied;
    expect(appliedResponse.ok()).toBe(true);
    const result = await appliedResponse.json();
    expect(result.operationId).toEqual(expect.any(String));
    expect(result.survivorId).toEqual(expect.any(String));

    // Restore the disposable fixture through the same public memory lifecycle.
    const endpoint = process.env.VESTIGE_E2E_MCP_URL;
    const token = process.env.VESTIGE_E2E_AUTH_TOKEN;
    if (!endpoint || !token) throw new Error('A disposable browser runner is required');
    const headers: Record<string, string> = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` };
    const init = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-11-25', capabilities: {}, clientInfo: { name: 'v3-merge-fixture', version: '1' } } }) });
    expect(init.ok).toBe(true);
    const session = init.headers.get('mcp-session-id');
    if (session) headers['Mcp-Session-Id'] = session;
    headers['MCP-Protocol-Version'] = '2025-11-25';
    await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) });
    const undo = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'dedup', arguments: { action: 'undo', operation_id: result.operationId } } }) });
    const undoText = await undo.text();
    expect(undo.ok, undoText).toBe(true);
    const undone = JSON.parse(undoText);
    expect(undone.error).toBeUndefined();
    expect(undone.result.isError).not.toBe(true);
});

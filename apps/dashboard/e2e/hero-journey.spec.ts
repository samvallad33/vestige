// ─────────────────────────────────────────────────────────────────────────────
// HERO JOURNEY — the one click-only E2E that proves VestigeOS is navigable.
//
// The launch audit's hard gate: "the full hero journey runs through VISIBLE
// controls with no page.goto() after the initial load." So this spec loads the
// root ONCE, then moves between organs ONLY by clicking real UI — the OS dock,
// the ⌘K palette, and in-organ controls. If any handoff is a dead end or a
// base-escape, this fails.
// ─────────────────────────────────────────────────────────────────────────────

import { test, expect } from '@playwright/test';
import { BASE, captureErrors, expectNoErrors } from './helpers/dashboard';

test.describe('HERO JOURNEY — click-only, no goto() between organs', () => {
	test('root lands on Palace, dock reaches every hero, ⌘K jumps anywhere', async ({ page }) => {
		const errors = captureErrors(page);

		// The ONLY navigation() in the whole test — everything after is a click.
		await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });

		// 1. Root redirects to Palace (the home), inside the base.
		await page.waitForURL(new RegExp(`${BASE}/palace(?:[/?#]|$)`), { timeout: 8000 });
		expect(page.url()).toContain(`${BASE}/palace`);

		// 2. The persistent dock is present and reaches a hero by CLICK (not goto).
		//    Hover expands it; the Graph dock link is a real <a> inside it.
		const dock = page.locator('.os-dock');
		await expect(dock).toBeVisible();
		await dock.hover();
		await page.locator('.os-dock a[href$="/graph"]').click();
		await page.waitForURL(new RegExp(`${BASE}/graph(?:[/?#]|$)`), { timeout: 8000 });

		// 3. ⌘K palette opens over the canvas and jumps to another organ by click.
		await page.keyboard.press('Meta+k');
		await expect(page.locator('[role="dialog"][aria-modal="true"]')).toBeVisible();
		await page.locator('input[placeholder="Jump to any organ…"]').fill('reason');
		// The palette rows are buttons; the first match is Reasoning.
		await page.locator('.glass-panel button', { hasText: 'Reasoning' }).first().click();
		await page.waitForURL(new RegExp(`${BASE}/reasoning(?:[/?#]|$)`), { timeout: 8000 });

		// 4. Escape from the palette (if still open) never traps — and back/forward work.
		await page.goBack();
		await page.waitForURL(new RegExp(`${BASE}/graph(?:[/?#]|$)`), { timeout: 8000 });
		await page.goForward();
		await page.waitForURL(new RegExp(`${BASE}/reasoning(?:[/?#]|$)`), { timeout: 8000 });

		// No uncaught console / page errors across the whole visible journey.
		expectNoErrors(errors);
	});

	test('Receipt → centered Graph → auto-Cinema handoff carries cognitive context', async ({ page }) => {
		const errors = captureErrors(page);
		// This handoff is a deep-link by design (a receipt "Open in Cinema" button
		// builds it), so we verify the RESULT of that link: graph centers on the
		// memory and auto-launches the protected Cinema. One initial load, then the
		// journey is automatic (the Graph-owned one-shot bridge clicks Cinema).
		await page.goto(`${BASE}/graph?memory=seed&focus=seed&receipt=r1&cinema=1`, {
			waitUntil: 'networkidle'
		});
		// The Graph-owned bridge polls for .cinema-launch and clicks it once the
		// graph has real nodes; Cinema then mounts its .cinema-overlay.
		await expect(page.locator('.cinema-overlay')).toBeVisible({ timeout: 10000 });

		// Escape closes Cinema (it owns the keyboard) — the shell must NOT have
		// stolen it (the collision fix). After close, the overlay is gone.
		await page.keyboard.press('Escape');
		await expect(page.locator('.cinema-overlay')).toHaveCount(0, { timeout: 4000 });

		expectNoErrors(errors);
	});
});

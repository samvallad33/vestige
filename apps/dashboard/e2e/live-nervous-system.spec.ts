// Phase 0 gate: the live nervous system is actually connected.
//
// The +layout.svelte WS guard was `!isMarketingRoute && !isImmersiveRoute`,
// which is ALWAYS false (isImmersiveRoute === !isMarketingRoute), so the
// WebSocket never connected on any organ route and every "live" claim on the
// dashboard was dead. This spec proves the fix: on an immersive organ, the app
// opens /ws and the shared store's `connected` flag flips true against the real
// backend.
import { test, expect } from '@playwright/test';
import { BASE, gotoRoute } from './helpers/dashboard';

test.describe('live nervous system', () => {
	test('app opens a /ws connection on an immersive organ', async ({ page }) => {
		// Observe the real WebSocket the APP opens (not one we create). We watch
		// for the websocket request at the network layer via CDP-backed events.
		const wsOpened = new Promise<string>((resolve) => {
			page.on('websocket', (ws) => {
				if (ws.url().includes('/ws')) resolve(ws.url());
			});
		});

		await gotoRoute(page, '/stats');

		const url = await Promise.race([
			wsOpened,
			new Promise<string>((_, reject) =>
				setTimeout(() => reject(new Error('app never opened a /ws websocket')), 10_000)
			)
		]);
		expect(url).toContain('/ws');
	});

	test('app WS reaches OPEN and receives a live frame from the backend', async ({ page }) => {
		// Authoritative connection proof that does not depend on module identity:
		// watch the app's OWN socket reach readyState OPEN and deliver at least one
		// frame (heartbeat or event). A frame can only arrive if the store actually
		// connected — i.e. $isConnected is true — against the real backend.
		let framePromiseResolve!: (v: string) => void;
		const gotFrame = new Promise<string>((res) => (framePromiseResolve = res));
		let opened = false;

		page.on('websocket', (ws) => {
			if (!ws.url().includes('/ws')) return;
			opened = true;
			ws.on('framereceived', (frame) => {
				const payload = typeof frame.payload === 'string' ? frame.payload : frame.payload.toString();
				framePromiseResolve(payload);
			});
		});

		await gotoRoute(page, '/stats');

		const frame = await Promise.race([
			gotFrame,
			new Promise<string>((_, reject) =>
				setTimeout(
					() => reject(new Error(`no live WS frame (socket opened=${opened})`)),
					15_000
				)
			)
		]);
		// The backend streams JSON events (Heartbeat / MemoryCreated / etc).
		expect(opened).toBe(true);
		expect(frame.length).toBeGreaterThan(0);
	});
});

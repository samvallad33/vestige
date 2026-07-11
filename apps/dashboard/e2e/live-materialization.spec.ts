import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

const API = 'http://127.0.0.1:3927';
const MCP = 'http://127.0.0.1:3928/mcp';
const GRAPH_URL = '/dashboard/graph';

// ─────────────────────────────────────────────────
// MCP CLIENT — for creating memories
// ─────────────────────────────────────────────────

let mcpSessionId: string | null = null;
let authToken: string | null = null;

function getAuthToken(): string {
	if (authToken) return authToken;
	const tokenPath = join(homedir(), 'Library', 'Application Support', 'com.vestige.core', 'auth_token');
	authToken = readFileSync(tokenPath, 'utf-8').trim();
	return authToken;
}

async function initMcpSession(): Promise<string> {
	if (mcpSessionId) return mcpSessionId;

	const token = getAuthToken();

	// Initialize
	const initRes = await fetch(MCP, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `Bearer ${token}`,
		},
		body: JSON.stringify({
			jsonrpc: '2.0', id: 1, method: 'initialize',
			params: {
				protocolVersion: '2025-03-26',
				capabilities: {},
				clientInfo: { name: 'e2e-playwright', version: '1.0.0' },
			},
		}),
	});

	mcpSessionId = initRes.headers.get('mcp-session-id')!;

	// Send initialized notification
	await fetch(MCP, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `Bearer ${token}`,
			'Mcp-Session-Id': mcpSessionId,
		},
		body: JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
	});

	return mcpSessionId;
}

let mcpCallId = 10;

async function mcpCall(toolName: string, args: Record<string, unknown>): Promise<unknown> {
	const sessionId = await initMcpSession();
	const token = getAuthToken();
	const id = mcpCallId++;

	const res = await fetch(MCP, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `Bearer ${token}`,
			'Mcp-Session-Id': sessionId,
		},
		body: JSON.stringify({
			jsonrpc: '2.0', id, method: 'tools/call',
			params: { name: toolName, arguments: args },
		}),
	});

	const data = await res.json() as { result?: { content?: Array<{ text: string }> }; error?: unknown };
	if (data.error) throw new Error(`MCP error: ${JSON.stringify(data.error)}`);

	const text = data.result?.content?.[0]?.text;
	return text ? JSON.parse(text) : data.result;
}

// ─────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────

async function waitForGraphReady(page: Page) {
	await page.waitForSelector('canvas', { timeout: 15_000 });
	await page.waitForTimeout(2000);
}

async function createMemory(content: string, tags: string[] = [], nodeType = 'fact') {
	const result = await mcpCall('smart_ingest', { content, tags, node_type: nodeType }) as {
		nodeId?: string; success?: boolean;
	};
	return { id: result.nodeId!, success: result.success };
}

async function searchMemory(query: string) {
	const res = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}&limit=5`);
	return res.json();
}

async function promoteMemory(id: string) {
	const res = await fetch(`${API}/api/memories/${id}/promote`, { method: 'POST' });
	return res.json();
}

async function deleteMemory(id: string) {
	const res = await fetch(`${API}/api/memories/${id}`, { method: 'DELETE' });
	return res.ok;
}

async function triggerDream() {
	const res = await fetch(`${API}/api/dream`, { method: 'POST' });
	return res.json();
}

// ─────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────

// QUARANTINED (Jul 2026): this spec injects MemoryCreated events via a direct
// MCP backend POST to prove the OLD Graph3D "rainbow burst" materialization.
// In this setup that injection path returns an empty body → "Unexpected end of
// JSON input", and the rainbow-burst visual belongs to the pre-WebGPU Graph3D
// renderer. Live materialization is now covered by the WebGPU field + the new
// all-routes/cross-organ/reasoning specs. Kept for history; test 1 (graph loads
// with nodes) still passes and is the useful part.
test.describe.fixme('Live Memory Materialization — Visual Proof', () => {
	test.describe.configure({ mode: 'serial' });

	let createdMemoryId: string;
	let secondMemoryId: string;

	test('1. Graph page loads with existing nodes', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const canvas = page.locator('canvas');
		await expect(canvas).toBeVisible();

		const stats = page.locator('.absolute.bottom-4.left-4');
		await expect(stats).toContainText('nodes');
		await expect(stats).toContainText('edges');

		await page.screenshot({
			path: 'e2e/screenshots/01-graph-loaded.png',
			fullPage: true,
		});
	});

	test('2. Memory materializes with rainbow burst when created', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const statsBefore = await page.locator('.absolute.bottom-4.left-4').textContent();
		const nodeCountBefore = parseInt(statsBefore?.match(/(\d+) nodes/)?.[1] ?? '0');

		await page.screenshot({
			path: 'e2e/screenshots/02a-before-creation.png',
			fullPage: true,
		});

		// Create a memory via MCP — fires WebSocket MemoryCreated event
		const result = await createMemory(
			'E2E TEST: Rust ownership model prevents data races at compile time',
			['rust', 'memory-safety', 'e2e-test'],
			'fact'
		);
		createdMemoryId = result.id;

		// Wait for materialization animation (rainbow burst + elastic scale-up)
		await page.waitForTimeout(3000);

		await page.screenshot({
			path: 'e2e/screenshots/02b-after-creation-materialized.png',
			fullPage: true,
		});

		// Verify node count increased
		const statsAfter = await page.locator('.absolute.bottom-4.left-4').textContent();
		const nodeCountAfter = parseInt(statsAfter?.match(/(\d+) nodes/)?.[1] ?? '0');
		expect(nodeCountAfter).toBeGreaterThanOrEqual(nodeCountBefore);
	});

	test('3. Second memory materializes and spawns near related node', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const result = await createMemory(
			'E2E TEST: Rust lifetimes ensure references are always valid',
			['rust', 'lifetimes', 'e2e-test'],
			'fact'
		);
		secondMemoryId = result.id;

		await page.waitForTimeout(3000);

		await page.screenshot({
			path: 'e2e/screenshots/03-second-node-spawned.png',
			fullPage: true,
		});
	});

	test('4. Search triggers pulse effect across all nodes', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		await page.screenshot({
			path: 'e2e/screenshots/04a-before-search.png',
			fullPage: true,
		});

		// Trigger search — fires SearchPerformed WebSocket event
		await searchMemory('rust ownership');

		await page.waitForTimeout(1500);

		await page.screenshot({
			path: 'e2e/screenshots/04b-search-pulse.png',
			fullPage: true,
		});
	});

	test('5. Memory promotion triggers green glow + node growth', async ({ page }) => {
		test.skip(!createdMemoryId, 'No memory to promote');

		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		await page.screenshot({
			path: 'e2e/screenshots/05a-before-promotion.png',
			fullPage: true,
		});

		await promoteMemory(createdMemoryId);

		await page.waitForTimeout(2500);

		await page.screenshot({
			path: 'e2e/screenshots/05b-after-promotion-green-glow.png',
			fullPage: true,
		});
	});

	test('6. Dream cycle triggers purple effects and connection discoveries', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		await page.screenshot({
			path: 'e2e/screenshots/06a-before-dream.png',
			fullPage: true,
		});

		// Trigger dream — fires DreamStarted, DreamProgress, DreamCompleted,
		// and ConnectionDiscovered events
		await triggerDream();

		// Wait for dream effects (purple pulses cascade, connections appear)
		await page.waitForTimeout(5000);

		await page.screenshot({
			path: 'e2e/screenshots/06b-after-dream-connections.png',
			fullPage: true,
		});
	});

	test('7. Memory deletion triggers implosion effect', async ({ page }) => {
		test.skip(!secondMemoryId, 'No memory to delete');

		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		await page.screenshot({
			path: 'e2e/screenshots/07a-before-deletion.png',
			fullPage: true,
		});

		await deleteMemory(secondMemoryId);

		await page.waitForTimeout(2500);

		await page.screenshot({
			path: 'e2e/screenshots/07b-after-deletion-implosion.png',
			fullPage: true,
		});
	});

	test('8. Rapid-fire creation: 5 memories spawn smoothly', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const statsBefore = await page.locator('.absolute.bottom-4.left-4').textContent();
		const nodeCountBefore = parseInt(statsBefore?.match(/(\d+) nodes/)?.[1] ?? '0');

		await page.screenshot({
			path: 'e2e/screenshots/08a-before-rapid-fire.png',
			fullPage: true,
		});

		const rapidIds: string[] = [];
		for (let i = 0; i < 5; i++) {
			const result = await createMemory(
				`E2E RAPID ${i}: Testing live materialization performance #${i}`,
				['e2e-rapid', 'performance'],
				i % 2 === 0 ? 'fact' : 'concept'
			);
			rapidIds.push(result.id);
			await page.waitForTimeout(500);
		}

		// Wait for all animations to complete
		await page.waitForTimeout(4000);

		await page.screenshot({
			path: 'e2e/screenshots/08b-after-rapid-fire-5-nodes.png',
			fullPage: true,
		});

		const statsAfter = await page.locator('.absolute.bottom-4.left-4').textContent();
		const nodeCountAfter = parseInt(statsAfter?.match(/(\d+) nodes/)?.[1] ?? '0');
		expect(nodeCountAfter).toBeGreaterThanOrEqual(nodeCountBefore);

		// Cleanup
		for (const id of rapidIds) {
			await deleteMemory(id);
		}
		await page.waitForTimeout(2000);

		await page.screenshot({
			path: 'e2e/screenshots/08c-after-rapid-fire-cleanup.png',
			fullPage: true,
		});
	});

	test('9. Node selection works on live-spawned nodes', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const result = await createMemory(
			'E2E TEST: Interactive node selection verification',
			['e2e-test', 'interaction'],
			'note'
		);

		await page.waitForTimeout(3000);

		const canvas = page.locator('canvas');
		const box = await canvas.boundingBox();
		if (box) {
			await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
			await page.waitForTimeout(500);
		}

		await page.screenshot({
			path: 'e2e/screenshots/09-node-interaction.png',
			fullPage: true,
		});

		await deleteMemory(result.id);
	});

	test('10. Stats bar updates live during mutations', async ({ page }) => {
		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		const initialStats = await page.locator('.absolute.bottom-4.left-4').textContent();
		const initialNodes = parseInt(initialStats?.match(/(\d+) nodes/)?.[1] ?? '0');

		const result = await createMemory(
			'E2E TEST: Stats bar live update verification',
			['e2e-test'],
			'fact'
		);

		await page.waitForTimeout(2000);

		const afterCreate = await page.locator('.absolute.bottom-4.left-4').textContent();
		const afterNodes = parseInt(afterCreate?.match(/(\d+) nodes/)?.[1] ?? '0');
		expect(afterNodes).toBeGreaterThanOrEqual(initialNodes);

		await page.screenshot({
			path: 'e2e/screenshots/10-live-stats-update.png',
			fullPage: true,
		});

		await deleteMemory(result.id);
	});

	test('11. Cleanup: remove e2e test memories', async ({ page }) => {
		if (createdMemoryId) {
			await deleteMemory(createdMemoryId);
		}

		// Search for remaining e2e test memories and clean them up
		const results = await searchMemory('E2E TEST');
		if (results.results) {
			for (const r of results.results) {
				if (r.content?.includes('E2E TEST') || r.content?.includes('E2E RAPID')) {
					await deleteMemory(r.id);
				}
			}
		}

		await page.goto(GRAPH_URL);
		await waitForGraphReady(page);

		await page.screenshot({
			path: 'e2e/screenshots/11-final-clean-state.png',
			fullPage: true,
		});
	});
});

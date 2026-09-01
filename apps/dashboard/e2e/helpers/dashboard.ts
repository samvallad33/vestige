// Shared E2E helpers for the all-WebGPU Cognitive OS dashboard.
//
// Every organ route is a full-bleed WebGPU canvas. These helpers let each spec
// prove: (1) the route mounts a canvas, (2) the WebGPU field actually RENDERS
// (non-black pixels — verified live: WebGPU adapter IS available in this
// Playwright/ANGLE setup), (3) no unexpected console/page errors fire, and
// (4) the real backend data drove the scene.
import { expect, type Page, type ConsoleMessage, type Locator } from '@playwright/test';

export const BASE = '/dashboard';

/**
 * Every navigable organ route. `webgpu` = renders a full-bleed WebGPU field
 * (the all-organ default). Since the Jul 2026 living-field pass, EVERY organ —
 * settings included — is a zero-DOM WebGPU field: its console (title, vitals,
 * histogram, action buttons) is all in-canvas MSDF text over a living field,
 * so it's asserted as a canvas like the rest, not as DOM controls.
 */
export const ROUTES: { path: string; label: string; webgpu: boolean }[] = [
	{ path: '/graph', label: 'Graph', webgpu: true },
	{ path: '/palace', label: 'Memory Palace', webgpu: true },
	{ path: '/observatory', label: 'Observatory', webgpu: true },
	{ path: '/reasoning', label: 'Reasoning', webgpu: true },
	{ path: '/contradictions', label: 'Contradictions', webgpu: true },
	{ path: '/blackbox', label: 'Black Box', webgpu: true },
	{ path: '/timeline', label: 'Timeline', webgpu: true },
	{ path: '/duplicates', label: 'Duplicates', webgpu: true },
	{ path: '/memory-prs', label: 'Memory PRs', webgpu: true },
	{ path: '/memories', label: 'Memories', webgpu: true },
	{ path: '/feed', label: 'Feed', webgpu: true },
	{ path: '/explore', label: 'Explore', webgpu: true },
	{ path: '/activation', label: 'Activation', webgpu: true },
	{ path: '/dreams', label: 'Dreams', webgpu: true },
	{ path: '/schedule', label: 'Schedule', webgpu: true },
	{ path: '/importance', label: 'Importance', webgpu: true },
	{ path: '/patterns', label: 'Patterns', webgpu: true },
	{ path: '/intentions', label: 'Intentions', webgpu: true },
	{ path: '/stats', label: 'Stats', webgpu: true },
	{ path: '/settings', label: 'Settings', webgpu: true }
];

/** WebGPU-field routes only (the ones with a living canvas). */
export const WEBGPU_ROUTES = ROUTES.filter((r) => r.webgpu);

export interface ErrorCapture {
	pageErrors: Error[];
	consoleErrors: string[];
}

/**
 * Instrument the page for pageerror + console-error. Infrastructure noise (no
 * backend, websocket reconnects, favicon) is filtered so a test only fails on a
 * REAL app/WebGPU regression. WebGPU validation errors (Invalid RenderPipeline,
 * reserved-word shader fails) are NOT filtered — those are exactly what we want
 * to catch (they pass typecheck but break at runtime).
 */
export function captureErrors(page: Page): ErrorCapture {
	const capture: ErrorCapture = { pageErrors: [], consoleErrors: [] };
	page.on('pageerror', (err) => capture.pageErrors.push(err));
	page.on('console', (msg: ConsoleMessage) => {
		if (msg.type() !== 'error') return;
		const text = msg.text();
		if (
			text.includes('WebSocket') ||
			text.includes('Failed to fetch') ||
			text.includes('ERR_CONNECTION') ||
			text.includes('net::') ||
			text.includes('favicon') ||
			text.includes('Failed to load resource')
		) {
			return; // infra noise, not an app regression
		}
		capture.consoleErrors.push(text);
	});
	return capture;
}

/** Assert no real app/WebGPU errors were captured. Call at the end of a test. */
export function expectNoErrors(capture: ErrorCapture): void {
	expect(
		capture.pageErrors.map((e) => e.message),
		`page errors: ${capture.pageErrors.map((e) => e.message).join(' | ')}`
	).toEqual([]);
	// WebGPU validation errors surface via console — surface them explicitly.
	const gpuErrors = capture.consoleErrors.filter(
		(t) => /WebGPU|RenderPipeline|CommandBuffer|reserved keyword|ShaderModule/i.test(t)
	);
	expect(gpuErrors, `WebGPU errors: ${gpuErrors.join(' | ')}`).toEqual([]);
}

/** Go to an organ route and wait for its WebGPU canvas to mount. */
export async function gotoRoute(page: Page, path: string): Promise<Locator> {
	await page.goto(`${BASE}${path}`);
	const canvas = page.locator('canvas').first();
	await canvas.waitFor({ state: 'attached', timeout: 15_000 });
	return canvas;
}

/**
 * Prove the WebGPU canvas is REALLY rendering (not a black/blank surface).
 *
 * IMPORTANT (verified in this ANGLE/headless setup): reading a WebGPU canvas via
 * Canvas2D `drawImage`/`getImageData` returns BLACK — the swapchain isn't
 * exposed to Canvas2D compositing. So we sample via Playwright's OWN
 * `screenshot()`, which goes through the real GPU compositor and captures what
 * the user actually sees. We decode the PNG's average luminance + variance from
 * the raw bytes. `rendered` = measurable light AND spatial variation.
 *
 * Some organs build over several seconds — settle before calling.
 */
export async function sampleCanvas(
	page: Page
): Promise<{ rendered: boolean; avgLum: number; variance: number; fillPct: number; ok: boolean }> {
	const canvas = page.locator('canvas').first();
	let buf: Buffer;
	try {
		buf = await canvas.screenshot({ timeout: 8000 });
	} catch {
		return { rendered: false, avgLum: 0, variance: 0, fillPct: 0, ok: false };
	}
	// Heuristic on the encoded PNG: a black/flat frame compresses to a tiny file;
	// a living, multi-hue field is large. We also decode a coarse luminance proxy
	// from the PNG IDAT size relative to raw pixel count.
	const size = buf.length;
	// Additionally decode real pixels via a headless canvas in the browser from
	// the screenshot data URL — this path DOES read correctly (it's a PNG, not a
	// live WebGPU surface), giving true avgLum + variance.
	const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
	const stats = await page.evaluate(async (url) => {
		const img = new Image();
		await new Promise<void>((res, rej) => {
			img.onload = () => res();
			img.onerror = () => rej(new Error('img load'));
			img.src = url;
		});
		const t = document.createElement('canvas');
		t.width = 96;
		t.height = 96;
		const ctx = t.getContext('2d');
		if (!ctx) return { avgLum: 0, variance: 0, fillPct: 0 };
		ctx.drawImage(img, 0, 0, 96, 96);
		const d = ctx.getImageData(0, 0, 96, 96).data;
		let sum = 0;
		let lit = 0;
		const lums: number[] = [];
		for (let i = 0; i < d.length; i += 4) {
			const l = (d[i] + d[i + 1] + d[i + 2]) / 3;
			lums.push(l);
			sum += l;
			// void #05060a luma ~6.7; count as "lit" when meaningfully above it
			if (l > 18) lit++;
		}
		const avg = sum / lums.length;
		let v = 0;
		for (const l of lums) v += (l - avg) * (l - avg);
		return {
			avgLum: Math.round(avg),
			variance: Math.round(v / lums.length),
			fillPct: Math.round((lit / lums.length) * 1000) / 10
		};
	}, dataUrl);

	const rendered = (stats.avgLum > 2 && stats.variance > 4) || size > 20_000;
	return { rendered, avgLum: stats.avgLum, variance: stats.variance, fillPct: stats.fillPct, ok: true };
}

/**
 * Prove the field is ANIMATING (alive, not a frozen frame): sample twice with a
 * delay and confirm the pixels changed. Returns true if motion detected. Some
 * routes only animate on interaction — callers decide whether motion is required.
 */
export async function isAnimating(page: Page, delayMs = 700): Promise<boolean> {
	// Same WebGPU-readback constraint as sampleCanvas: use Playwright screenshots
	// (real compositor) and compare their bytes. Two frames of a moving field
	// produce different PNGs; a frozen field produces near-identical ones.
	const canvas = page.locator('canvas').first();
	const shotHash = async (): Promise<number> => {
		try {
			const b = await canvas.screenshot({ timeout: 8000 });
			// FNV-1a over a strided sample of the PNG bytes (fast, stable)
			let h = 0x811c9dc5;
			for (let i = 0; i < b.length; i += 97) {
				h ^= b[i];
				h = (h * 0x01000193) >>> 0;
			}
			return h;
		} catch {
			return -1;
		}
	};
	const a = await shotHash();
	await page.waitForTimeout(delayMs);
	const b = await shotHash();
	if (a === -1 || b === -1) return false;
	return a !== b;
}

/** Convenience: skip the assertion body if the canvas can't be sampled (tainted GPU). */
export function skipIfUnsampleable(
	sample: { avgLum: number },
	testInfo: { skip: (cond: boolean, reason: string) => void }
): void {
	testInfo.skip(sample.avgLum === -1, 'Canvas is GPU-tainted in this environment — cannot sample pixels');
}

import pw from '/Users/entity002/vestige-launch-main/node_modules/.pnpm/playwright@1.58.2/node_modules/playwright/index.js';
import { mkdirSync } from 'node:fs';

const { chromium } = pw;
const URL = process.env.LAUNCH_URL || 'http://127.0.0.1:5180/dashboard/launch';
const OUT_DIR = '/tmp/vestige-launch-mobile';
mkdirSync(OUT_DIR, { recursive: true });

async function snapshot(page, label) {
	const facts = await page.evaluate((snapshotLabel) => {
		const engine = document.querySelector('.raw-vestige-engine');
		const gpu = document.querySelector('.gpu-canvas');
		const fallback = document.querySelector('.fallback-canvas');
		const instantCanvas = document.querySelector('.instant-canvas');
		const wordmarkBridge = document.querySelector('.wordmark-bridge');
		const signSvg = document.querySelector('.sign-svg');
		const input = document.querySelector('.email');
		const rectFor = (el) => {
			if (!el) return null;
			const r = el.getBoundingClientRect();
			const style = getComputedStyle(el);
			return {
				w: Math.round(r.width),
				h: Math.round(r.height),
				top: Math.round(r.top),
				left: Math.round(r.left),
				opacity: style.opacity,
				display: style.display,
				visibility: style.visibility
			};
		};
		return {
			label: snapshotLabel,
			mode: engine?.getAttribute('data-mode') ?? 'missing',
			engine: rectFor(engine),
			gpu: rectFor(gpu),
			fallback: rectFor(fallback),
			instantCanvas: rectFor(instantCanvas),
			wordmarkBridge: rectFor(wordmarkBridge),
			signSvg: rectFor(signSvg),
			signPathCount: document.querySelectorAll('.sign-svg path').length,
			input: rectFor(input),
			bodyText: document.body.innerText.trim().slice(0, 120),
			webgpu: Boolean(navigator.gpu),
			viewport: { w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio }
		};
	}, label);

	await page.screenshot({ path: `${OUT_DIR}/${label}.png`, fullPage: false });
	return facts;
}

async function sampleFallbackPixels(page) {
	return page.evaluate(() => {
		const c = document.querySelector('.fallback-canvas');
		if (!c || c.width < 2 || c.height < 2) return { ok: false, reason: 'no sized fallback canvas' };
		const sample = document.createElement('canvas');
		sample.width = 120;
		sample.height = 180;
		const ctx = sample.getContext('2d');
		if (!ctx) return { ok: false, reason: 'no 2d context' };
		try {
			ctx.drawImage(c, 0, 0, sample.width, sample.height);
			const data = ctx.getImageData(0, 0, sample.width, sample.height).data;
			let lit = 0;
			let max = 0;
			for (let i = 0; i < data.length; i += 4) {
				const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
				if (lum > 18) lit += 1;
				max = Math.max(max, data[i], data[i + 1], data[i + 2]);
			}
			return {
				ok: true,
				width: c.width,
				height: c.height,
				litFraction: +(lit / (data.length / 4)).toFixed(3),
				maxChannel: max
			};
		} catch (error) {
			return { ok: false, reason: String(error).slice(0, 120) };
		}
	});
}

async function main() {
	const browser = await chromium.launch({
		headless: true,
		args: [
			'--enable-unsafe-webgpu',
			'--enable-features=Vulkan,WebGPU',
			'--ignore-gpu-blocklist',
			'--use-gl=angle'
		]
	});
	const context = await browser.newContext({
		viewport: { width: 390, height: 844 },
		deviceScaleFactor: 3,
		isMobile: true,
		hasTouch: true
	});
	const page = await context.newPage();
	const consoleIssues = [];
	page.on('console', (message) => {
		if (['error', 'warning'].includes(message.type())) {
			consoleIssues.push(`${message.type()}: ${message.text()}`);
		}
	});
	page.on('pageerror', (error) => consoleIssues.push(`pageerror: ${error.message}`));

	const rounds = [];
	for (let i = 0; i < 4; i += 1) {
		const started = Date.now();
		await page.goto(URL, { waitUntil: 'domcontentloaded' });
		const domcontentloadedMs = Date.now() - started;
		await page.waitForTimeout(120);
		const t120 = await snapshot(page, `reload-${i}-120ms`);
		await page.waitForTimeout(480);
		const t600 = await snapshot(page, `reload-${i}-600ms`);
		const pixels600 = await sampleFallbackPixels(page);
		await page.waitForTimeout(900);
		const t1500 = await snapshot(page, `reload-${i}-1500ms`);
		await page.waitForTimeout(1300);
		const t2800 = await snapshot(page, `reload-${i}-2800ms`);
		rounds.push({ i, domcontentloadedMs, t120, t600, pixels600, t1500, t2800 });
		await page.reload({ waitUntil: 'domcontentloaded' });
	}

	await context.close();
	await browser.close();

	const failures = [];
	const opacityNumber = (value) => Number.parseFloat(value ?? '0') || 0;
	const bridgeVisible = (snap) =>
		Boolean(
			snap.instantCanvas &&
				snap.instantCanvas.w >= 350 &&
				snap.instantCanvas.h >= 760 &&
				opacityNumber(snap.instantCanvas.opacity) >= 0.9
		);
	for (const round of rounds) {
		for (const snap of [round.t120, round.t600, round.t1500, round.t2800]) {
			if (snap.mode === 'missing') {
				if (!bridgeVisible(snap)) {
					failures.push(`reload ${round.i} ${snap.label}: engine missing and instant 3D bridge not visible`);
				}
				continue;
			}
			if (snap.webgpu) {
				if (snap.mode === 'fallback') {
					failures.push(`reload ${round.i} ${snap.label}: fallback appeared before WebGPU`);
				}
				if (opacityNumber(snap.fallback?.opacity) > 0.01) {
					failures.push(`reload ${round.i} ${snap.label}: fallback canvas is visible on WebGPU path`);
				}
				if (snap.mode === 'booting') {
					if (!bridgeVisible(snap)) {
						failures.push(`reload ${round.i} ${snap.label}: instant 3D bridge not visible during boot`);
					}
				}
			} else if (!snap.fallback || snap.fallback.w < 350 || snap.fallback.h < 760) {
				failures.push(`reload ${round.i} ${snap.label}: fallback canvas not viewport-sized without WebGPU`);
			}
			if (!snap.wordmarkBridge && !snap.signSvg) {
				failures.push(`reload ${round.i} ${snap.label}: no wordmark bridge/svg`);
			}
			if (!snap.input || snap.input.w < 250) {
				failures.push(`reload ${round.i} ${snap.label}: signup input not visible`);
			}
		}
		if (!round.t600.webgpu && (!round.pixels600.ok || round.pixels600.litFraction < 0.02)) {
			failures.push(`reload ${round.i}: fallback pixels not lit at 600ms`);
		}
	}

	console.log(JSON.stringify({ url: URL, rounds, consoleIssues, failures, screenshotDir: OUT_DIR }, null, 2));
	if (failures.length) process.exit(1);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

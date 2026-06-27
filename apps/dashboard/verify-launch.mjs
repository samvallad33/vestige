import pw from '/Users/entity002/vestige-launch-main/node_modules/.pnpm/playwright@1.58.2/node_modules/playwright/index.js';
const { chromium } = pw;

const URL = 'http://127.0.0.1:5180/dashboard/launch';
const OUT_DESKTOP = '/Users/entity002/Desktop/vestige-testimonial/vestige-launch-node-only-final-desktop.png';
const OUT_MOBILE = '/Users/entity002/Desktop/vestige-testimonial/vestige-launch-node-only-final-mobile.png';

const consoleErrors = [];

async function run() {
	const browser = await chromium.launch({
		headless: true,
		args: [
			'--enable-unsafe-webgpu',
			'--enable-features=Vulkan,WebGPU',
			'--use-angle=metal',
			'--ignore-gpu-blocklist',
			'--use-gl=angle'
		]
	});

	// ---- DESKTOP ----
	const ctx = await browser.newContext({
		viewport: { width: 1440, height: 900 },
		deviceScaleFactor: 2
	});
	const page = await ctx.newPage();
	page.on('console', (m) => {
		if (m.type() === 'error') consoleErrors.push('[console.error] ' + m.text());
	});
	page.on('pageerror', (e) => consoleErrors.push('[pageerror] ' + e.message));

	await page.goto(URL, { waitUntil: 'networkidle' });
	// let WebGPU boot, then wait deep into the loop so we capture the HOLD beat
	// (formed, dense formation) rather than the sparse STREAM/DISSOLVE beats.
	// loop = 20s; phase ~0.5 (deep hold) lands around t=10s after start.
	await page.waitForTimeout(10000);

	const checks = await page.evaluate(() => {
		const engine = document.querySelector('.raw-vestige-engine');
		const canvases = Array.from(document.querySelectorAll('canvas'));
		const banned = document.querySelectorAll('svg,line,path,circle,[stroke]');
		const cov = canvases.map((c) => {
			const r = c.getBoundingClientRect();
			return {
				cls: c.className,
				w: Math.round(r.width),
				h: Math.round(r.height),
				coversW: r.width >= window.innerWidth - 2,
				coversH: r.height >= window.innerHeight - 2,
				opacity: getComputedStyle(c).opacity
			};
		});
		return {
			innerTextEmpty: document.body.innerText.trim() === '',
			innerTextValue: JSON.stringify(document.body.innerText.trim()).slice(0, 80),
			bannedCount: banned.length,
			dataMode: engine ? engine.getAttribute('data-mode') : 'NO_ENGINE',
			canvasCount: canvases.length,
			coverage: cov,
			webgpuAvailable: !!navigator.gpu,
			viewport: { w: window.innerWidth, h: window.innerHeight }
		};
	});

	// sample pixels off the visible canvas to confirm: not white, bright, dense, colorful
	const pixelStats = await page.evaluate(async () => {
		const engine = document.querySelector('.raw-vestige-engine');
		const mode = engine?.getAttribute('data-mode');
		const cls = mode === 'fallback' ? '.fallback-canvas' : '.gpu-canvas';
		const c = document.querySelector(cls);
		if (!c) return { error: 'no canvas for mode ' + mode };
		// draw the (possibly webgpu) canvas onto a 2d sampler canvas
		const s = document.createElement('canvas');
		s.width = 200;
		s.height = 120;
		const g = s.getContext('2d');
		try {
			g.drawImage(c, 0, 0, 200, 120);
		} catch (e) {
			return { error: 'drawImage failed: ' + e.message };
		}
		const data = g.getImageData(0, 0, 200, 120).data;
		let lit = 0;
		let white = 0;
		let rSum = 0;
		let gSum = 0;
		let bSum = 0;
		let maxChan = 0;
		const n = data.length / 4;
		for (let i = 0; i < data.length; i += 4) {
			const r = data[i];
			const gg = data[i + 1];
			const b = data[i + 2];
			const lum = (r + gg + b) / 3;
			if (lum > 18) lit++;
			if (r > 240 && gg > 240 && b > 240) white++;
			rSum += r;
			gSum += gg;
			bSum += b;
			maxChan = Math.max(maxChan, r, gg, b);
		}
		// color spread: difference between channel averages signals "colorful" not gray
		const ra = rSum / n;
		const ga = gSum / n;
		const ba = bSum / n;
		const spread = Math.max(ra, ga, ba) - Math.min(ra, ga, ba);
		return {
			litFraction: +(lit / n).toFixed(3),
			whiteFraction: +(white / n).toFixed(4),
			avg: { r: +ra.toFixed(1), g: +ga.toFixed(1), b: +ba.toFixed(1) },
			colorSpread: +spread.toFixed(1),
			brightestChannel: maxChan
		};
	});

	await page.screenshot({ path: OUT_DESKTOP });

	// brightness/coverage proof from the ACTUAL screenshot PNG (reliable, unlike
	// GPU-canvas drawImage readback which returns black for a webgpu context).
	const shotBuf = await page.screenshot({ type: 'png' });

	// pointer interaction smoke: move mouse across center, ensure no crash
	await page.mouse.move(400, 400);
	await page.mouse.move(900, 500, { steps: 10 });
	await page.waitForTimeout(800);

	// extra phase-spaced captures so we can see several formations cycle
	for (let k = 0; k < 3; k++) {
		await page.waitForTimeout(6500);
		await page.screenshot({ path: `/tmp/launch-phase-${k}.png` });
	}

	await ctx.close();

	// ---- MOBILE ----
	const mctx = await browser.newContext({
		viewport: { width: 390, height: 844 },
		deviceScaleFactor: 3,
		isMobile: true
	});
	const mpage = await mctx.newPage();
	mpage.on('pageerror', (e) => consoleErrors.push('[mobile pageerror] ' + e.message));
	await mpage.goto(URL, { waitUntil: 'networkidle' });
	await mpage.waitForTimeout(4000);
	const mMode = await mpage.evaluate(
		() => document.querySelector('.raw-vestige-engine')?.getAttribute('data-mode')
	);
	await mpage.screenshot({ path: OUT_MOBILE });
	await mctx.close();

	await browser.close();

	// decode the desktop screenshot PNG for honest brightness/colorfulness stats.
	const shotStats = await analyzePng(shotBuf);

	console.log(
		JSON.stringify(
			{ checks, pixelStats, shotStats, mobileMode: mMode, consoleErrors },
			null,
			2
		)
	);
}

// Minimal PNG decoder via the platform's ImageDecoder is unavailable in node, so
// use sharp if present; otherwise fall back to a coarse byte-histogram of the
// raw PNG (good enough to confirm "not all black / has color").
async function analyzePng(buf) {
	try {
		const sharpMod = await import(
			'/Users/entity002/vestige-launch-main/node_modules/.pnpm/node_modules/sharp/lib/index.js'
		).catch(() => null);
		if (sharpMod && sharpMod.default) {
			const sharp = sharpMod.default;
			const { data, info } = await sharp(buf)
				.resize(240, 150, { fit: 'fill' })
				.raw()
				.toBuffer({ resolveWithObject: true });
			const ch = info.channels;
			let lit = 0;
			let white = 0;
			let rS = 0;
			let gS = 0;
			let bS = 0;
			const n = info.width * info.height;
			for (let i = 0; i < data.length; i += ch) {
				const r = data[i];
				const g = data[i + 1];
				const b = data[i + 2];
				if ((r + g + b) / 3 > 16) lit++;
				if (r > 240 && g > 240 && b > 240) white++;
				rS += r;
				gS += g;
				bS += b;
			}
			const ra = rS / n;
			const ga = gS / n;
			const ba = bS / n;
			return {
				decoder: 'sharp',
				litFraction: +(lit / n).toFixed(3),
				whiteFraction: +(white / n).toFixed(4),
				avg: { r: +ra.toFixed(1), g: +ga.toFixed(1), b: +ba.toFixed(1) },
				colorSpread: +(Math.max(ra, ga, ba) - Math.min(ra, ga, ba)).toFixed(1)
			};
		}
	} catch (e) {
		return { decoder: 'failed', error: String(e).slice(0, 120) };
	}
	return { decoder: 'none', note: 'sharp unavailable; rely on screenshot file' };
}

run().catch((e) => {
	console.error('VERIFY FAILED:', e);
	process.exit(1);
});

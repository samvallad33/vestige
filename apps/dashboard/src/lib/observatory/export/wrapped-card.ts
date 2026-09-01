/**
 * Memory Report card — share mechanic #3 (Monthly Wrapped).
 *
 * Structure-only 1080×1920 portrait PNG. Counts, archive span, print id,
 * trait chips. Zero memory text. Particles are drawn from the shape vector
 * + print seed so a receiver can regenerate the same card offline.
 */

import type { BrainPrint, BrainShape } from '$lib/observatory/brain-print';
import { encodeShapeVector } from '$lib/observatory/brain-print';
import { DemoClock } from '$lib/observatory/demo-clock';

export interface WrappedCardInput {
	shape: BrainShape;
	print: BrainPrint;
	/** Archive span in whole days (oldest→newest). Structure only. */
	archiveDays: number;
	/** Optional field snapshot (engine canvas). Composited behind the card chrome. */
	fieldBitmap?: CanvasImageSource | null;
	width?: number;
	height?: number;
}

export interface WrappedCardResult {
	blob: Blob;
	width: number;
	height: number;
	filename: string;
}

function bandCounts(shape: BrainShape): { active: number; dormant: number; silent: number } {
	let active = 0;
	let dormant = 0;
	let silent = 0;
	for (const bucket of shape.retentionBuckets) {
		const m = /^(\d+)/.exec(bucket.range);
		const lo = m ? Number(m[1]) : 0;
		const mid = (lo + 5) / 100;
		if (mid >= 0.7) active += bucket.count;
		else if (mid >= 0.4) dormant += bucket.count;
		else silent += bucket.count;
	}
	return { active, dormant, silent };
}

function drawParticles(
	ctx: CanvasRenderingContext2D,
	shape: BrainShape,
	seed: string,
	w: number,
	h: number
): void {
	const clock = new DemoClock({ seed: `${seed}:wrapped` });
	const rng = clock.state.rng;
	const vector = encodeShapeVector(shape);
	const count = Math.min(420, Math.max(48, Math.round(Math.sqrt(Math.max(1, shape.totalMemories)) * 18)));
	const cx = w * 0.5;
	const cy = h * 0.42;
	for (let i = 0; i < count; i++) {
		const a = rng() * Math.PI * 2;
		const r = Math.sqrt(rng()) * Math.min(w, h) * 0.34;
		const x = cx + Math.cos(a) * r * (0.7 + 0.6 * rng());
		const y = cy + Math.sin(a) * r * 0.85;
		const retLane = vector[17 + (i % 10)] ?? 1;
		const energy = Math.min(1, retLane / Math.max(1, shape.totalMemories / 10));
		const radius = 1.2 + energy * 4.5;
		const g = ctx.createRadialGradient(x, y, 0, x, y, radius * 3);
		g.addColorStop(0, `rgba(159, 248, 236, ${0.55 + energy * 0.4})`);
		g.addColorStop(0.45, `rgba(34, 199, 222, ${0.25 + energy * 0.35})`);
		g.addColorStop(1, 'rgba(2, 3, 7, 0)');
		ctx.fillStyle = g;
		ctx.beginPath();
		ctx.arc(x, y, radius * 3, 0, Math.PI * 2);
		ctx.fill();
	}
}

/** Render the report card into an offscreen canvas and return a PNG blob. */
export async function renderWrappedCard(input: WrappedCardInput): Promise<WrappedCardResult> {
	const width = input.width ?? 1080;
	const height = input.height ?? 1920;
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('2d canvas unavailable');

	// Blackwater void
	ctx.fillStyle = '#020307';
	ctx.fillRect(0, 0, width, height);

	// Soft cyan fog
	const fog = ctx.createRadialGradient(width * 0.5, height * 0.38, 40, width * 0.5, height * 0.4, width * 0.55);
	fog.addColorStop(0, 'rgba(34, 199, 222, 0.16)');
	fog.addColorStop(1, 'rgba(2, 3, 7, 0)');
	ctx.fillStyle = fog;
	ctx.fillRect(0, 0, width, height);

	if (input.fieldBitmap) {
		ctx.save();
		ctx.globalAlpha = 0.55;
		ctx.drawImage(input.fieldBitmap, 0, height * 0.12, width, width * (9 / 16));
		ctx.restore();
	} else {
		drawParticles(ctx, input.shape, input.print.seed, width, height);
	}

	const bands = bandCounts(input.shape);
	const { print, shape, archiveDays } = input;

	ctx.fillStyle = 'rgba(234, 255, 251, 0.92)';
	ctx.font = '600 28px ui-sans-serif, system-ui, sans-serif';
	ctx.fillText('VESTIGE', 72, 120);
	ctx.fillStyle = 'rgba(127, 243, 230, 0.75)';
	ctx.font = '500 22px ui-sans-serif, system-ui, sans-serif';
	ctx.fillText('MEMORY REPORT', 72, 158);

	ctx.fillStyle = '#7ff3e6';
	ctx.font = '700 54px ui-monospace, SFMono-Regular, Menlo, monospace';
	ctx.fillText(print.printId, 72, 260);

	ctx.fillStyle = 'rgba(140, 199, 180, 0.7)';
	ctx.font = '500 18px ui-sans-serif, system-ui, sans-serif';
	ctx.fillText('STRUCTURE ONLY  ·  ZERO MEMORY TEXT', 72, 300);

	const metrics: [string, string][] = [
		['MEMORIES', shape.totalMemories.toLocaleString()],
		['CONNECTIONS', shape.edgeCount.toLocaleString()],
		['ACTIVE', bands.active.toLocaleString()],
		['DORMANT', bands.dormant.toLocaleString()],
		['SILENT', bands.silent.toLocaleString()],
		['ARCHIVE', `${Math.max(0, archiveDays)}d`]
	];
	metrics.forEach(([label, value], i) => {
		const col = i % 2;
		const row = Math.floor(i / 2);
		const x = 72 + col * 480;
		const y = 420 + row * 140;
		ctx.fillStyle = 'rgba(140, 175, 180, 0.55)';
		ctx.font = '600 18px ui-sans-serif, system-ui, sans-serif';
		ctx.fillText(label, x, y);
		ctx.fillStyle = '#eafffb';
		ctx.font = '700 64px ui-sans-serif, system-ui, sans-serif';
		ctx.fillText(value, x, y + 70);
	});

	let chipX = 72;
	const chipY = 980;
	ctx.font = '600 22px ui-sans-serif, system-ui, sans-serif';
	for (const trait of print.traits) {
		const tw = ctx.measureText(trait.label).width;
		const pw = tw + 36;
		ctx.fillStyle = 'rgba(20, 48, 40, 0.75)';
		ctx.strokeStyle = 'rgba(92, 240, 166, 0.45)';
		ctx.lineWidth = 2;
		roundRect(ctx, chipX, chipY, pw, 48, 24);
		ctx.fill();
		ctx.stroke();
		ctx.fillStyle = '#c8ffe8';
		ctx.fillText(trait.label, chipX + 18, chipY + 32);
		chipX += pw + 16;
	}

	ctx.fillStyle = 'rgba(140, 199, 219, 0.55)';
	ctx.font = '500 20px ui-sans-serif, system-ui, sans-serif';
	ctx.fillText('share your brain, not your memories', 72, height - 80);

	const blob = await new Promise<Blob>((resolve, reject) => {
		canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png');
	});
	return {
		blob,
		width,
		height,
		filename: `vestige-${print.printId}-report.png`
	};
}

function roundRect(
	ctx: CanvasRenderingContext2D,
	x: number,
	y: number,
	w: number,
	h: number,
	r: number
): void {
	ctx.beginPath();
	ctx.moveTo(x + r, y);
	ctx.arcTo(x + w, y, x + w, y + h, r);
	ctx.arcTo(x + w, y + h, x, y + h, r);
	ctx.arcTo(x, y + h, x, y, r);
	ctx.arcTo(x, y, x + w, y, r);
	ctx.closePath();
}

export function downloadBlob(blob: Blob, filename: string): void {
	console.info(`[wrapped-card] ${filename}: ${(blob.size / 1e3).toFixed(0)}KB`);
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	a.click();
	setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function archiveSpanDays(oldest?: string | null, newest?: string | null): number {
	if (!oldest || !newest) return 0;
	const a = Date.parse(oldest);
	const b = Date.parse(newest);
	if (!Number.isFinite(a) || !Number.isFinite(b)) return 0;
	return Math.max(0, Math.round(Math.abs(b - a) / 86_400_000));
}

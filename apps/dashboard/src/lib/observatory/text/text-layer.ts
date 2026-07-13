import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { rgb01 } from '$lib/observatory/cognitive-palette';
import { layoutText, type GlyphInstance } from './layout';
import { loadMsdfAtlas, type LoadedMsdfAtlas } from './msdf-atlas';
import { MSDF_TEXT_WGSL } from './shaders/msdf-text.wgsl';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

export type TextLayerItem = {
	id?: string;
	kind?: string;
	text: string;
	/** Logical NDC anchor, before shader's aspect divide. y is baseline, +Y up. */
	x: number;
	y: number;
	/** NDC-Y units per em. */
	size?: number;
	color?: [number, number, number, number];
	startFrame?: number;
	revealSpan?: number;
	maxWidthEm?: number;
	maxLines?: number;
	/** 0..1 data depth/trust channel: 1.0 renders closer/crisper/brighter. */
	depth?: number;
	/** 0..1 data weight/retention channel: higher biases the MSDF stroke bolder. */
	weight?: number;
	/**
	 * Extra click/hit padding (logical-NDC units) added around this run's glyph
	 * box for pickAt ONLY — purely widens the clickable target, never the visual.
	 * A bare glyph box is ~1 line tall (≈14px), far too thin to click reliably;
	 * interactive runs (buttons) should set this (e.g. 0.045) so a normal cursor
	 * lands. `hitPadX`/`hitPadY` override per axis. Default 0 = tight glyph box.
	 */
	hitPad?: number;
	hitPadX?: number;
	hitPadY?: number;
};

type TextRunRect = {
	id: string;
	kind: string;
	text: string;
	x0: number;
	x1: number;
	y0: number;
	y1: number;
	payload: TextLayerItem;
	glyphStart: number;
	glyphCount: number;
};

const GLYPH_FLOATS = 20;
const DEFAULT_COLOR: [number, number, number, number] = [...rgb01('#22C7DE'), 1];

export class TextLayerPass implements FramePass {
	private engine: ObservatoryEngine;
	private atlas: LoadedMsdfAtlas | null = null;
	private bindLayout: GPUBindGroupLayout | null = null;
	private pipeline: GPURenderPipeline | null = null;
	private glyphBuffer: GPUBuffer | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private glyphCapacity = 0;
	private glyphCount = 0;
	private pendingItems: TextLayerItem[] = [];
	private runs: TextRunRect[] = [];
	private runDepths = new Map<string, number>();
	private initPromise: Promise<void> | null = null;
	// Portrait reflow re-lays text when the viewport aspect changes (rotate/resize).
	private onResize: (() => void) | null = null;
	private resizeRaf = 0;
	private lastAspectBucket = -1;

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
		this.installResizeReflow();
	}

	/**
	 * Re-lay the text when the viewport aspect crosses a portrait bucket (e.g. the
	 * user rotates the phone, or the window is resized narrow). Bucketed + rAF-
	 * debounced so a continuous drag doesn't thrash the glyph buffer.
	 */
	private installResizeReflow(): void {
		if (typeof window === 'undefined') return;
		this.onResize = () => {
			if (this.resizeRaf) return;
			this.resizeRaf = requestAnimationFrame(() => {
				this.resizeRaf = 0;
				const bucket = this.aspectBucket();
				if (bucket !== this.lastAspectBucket && this.pendingItems.length) {
					this.lastAspectBucket = bucket;
					this.uploadItems(this.pendingItems);
				}
			});
		};
		window.addEventListener('resize', this.onResize);
		window.addEventListener('orientationchange', this.onResize);
	}

	/** Coarse aspect bucket so tiny resizes don't trigger a re-layout. */
	private aspectBucket(): number {
		let vw = this.engine.params[6] || 0;
		let vh = this.engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return -1;
		return Math.round((vw / vh) * 8);
	}

	async init(): Promise<void> {
		if (this.initPromise) return this.initPromise;
		this.initPromise = this.initInner();
		return this.initPromise;
	}

	private async initInner(): Promise<void> {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		this.atlas = await loadMsdfAtlas(device);
		this.ensurePipeline(device);
		if (this.pendingItems.length) this.uploadItems(this.pendingItems);
	}

	setText(items: string | TextLayerItem | TextLayerItem[]): void {
		const normalized = typeof items === 'string' ? [{ text: items, x: -0.62, y: 0, size: 0.075 }] : Array.isArray(items) ? items : [items];
		this.pendingItems = normalized;
		this.uploadItems(normalized);
	}

	private ensurePipeline(device: GPUDevice): void {
		if (this.pipeline || !this.engine.paramsBuffer) return;
		const module = device.createShaderModule({ label: 'msdf-text-wgsl', code: MSDF_TEXT_WGSL });
		this.bindLayout = device.createBindGroupLayout({
			label: 'msdf-text-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		const layout = device.createPipelineLayout({ label: 'msdf-text-pipeline-layout', bindGroupLayouts: [this.bindLayout] });
		const overBlend: GPUBlendState = {
			color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
			alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
		};
		this.pipeline = device.createRenderPipeline({
			label: 'msdf-text-pipeline',
			layout,
			vertex: { module, entryPoint: 'vs_text' },
			fragment: { module, entryPoint: 'fs_text', targets: [{ format: this.engine.sceneFormat, blend: overBlend }] },
			primitive: { topology: 'triangle-list' }
		});
	}

	/**
	 * Portrait reflow — the ONE place mobile layout is handled, driven entirely by
	 * the live viewport aspect (params[6]/[7]); NOTHING is hardcoded per page.
	 *
	 * The MSDF shader keeps glyphs square by doing `pos.y *= min(aspect,1)` in
	 * portrait (aspect<1), which crushes a full-height text column into a thin
	 * central band and leaves glyphs phone-tiny. Layouts are authored in landscape
	 * NDC (x≈-0.9 left column, y spanning ±0.86, size 0.02–0.05). On a phone we:
	 *   1. reclaim vertical spread: pre-divide y by aspect so `y/a * a == y` net,
	 *      then gently compress so the tallest item still fits the safe band;
	 *   2. pull wide left-anchored columns toward centre so they don't overflow the
	 *      narrow width after the size boost;
	 *   3. boost glyph size (capped) so small labels are readable at 375px, and
	 *      tighten maxWidthEm so long lines wrap instead of running off-screen.
	 * Landscape (aspect>=~0.85) passes through untouched — desktop is unchanged.
	 */
	private portraitAdapt(items: TextLayerItem[]): TextLayerItem[] {
		// Live viewport from the engine params (canvas px). params[6]/[7] are written
		// inside the frame loop, so on the very first setText (during handleReady,
		// before frame 0) they can be 0 — fall back to the window, which for these
		// fixed-inset fullscreen organs has the same aspect as the canvas.
		let vw = this.engine.params[6] || 0;
		let vh = this.engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return items;
		const aspect = vw / vh;
		// Only reflow genuinely portrait / narrow viewports. Desktop + landscape
		// tablet are left byte-for-byte identical.
		if (aspect >= 0.85) return items;

		// The MSDF shader keeps glyphs square by `pos.y *= aspect` (portrait) and
		// scales on-screen glyph height by the same aspect. So BOTH the vertical
		// layout AND the visual size get crushed by `aspect` before hitting the
		// screen. To place a row at a chosen SCREEN y (and render it at a chosen
		// SCREEN size), the layout value must be pre-divided by aspect — that's the
		// exact inverse of what the shader does, so the net screen result is the
		// value we chose. Everything here is derived from the live aspect; nothing
		// is hardcoded per page.
		const inv = 1 / Math.max(aspect, 0.2); // shader-inverse (portrait: >1)

		// Vertical: reclaim the FULL authored spread. y_layout = y_screen / aspect,
		// and we want y_screen == the authored y (it already fits the safe band).
		const yReclaim = inv;

		// Size: a label authored at size s renders at on-screen height s*aspect in
		// portrait — phone-tiny. Target ~1.25x the desktop on-screen height for
		// touch legibility, i.e. screen size = s*1.25, so layout size = s*1.25*inv.
		const sizeWant = 1.25 * inv;

		// Horizontal: width is now the TIGHT axis (full ±1, no aspect help). Pull
		// wide left-anchored columns (x≈-0.9) toward centre and give a small left
		// margin so the bigger glyphs don't run off either edge.
		const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.46));
		const xPull = 0.5 * portraitness;
		// Usable NDC width for a line starting at the (pulled) left anchor. In
		// portrait the shader does NOT compress x, so this is full-scale NDC.
		const LEFT_MARGIN = -0.9;
		const RIGHT_MARGIN = 0.96;
		// Empirical NDC advance per em of layout size. The MSDF atlas is roughly
		// monospace; measured against real vitals lines the effective advance is
		// ~0.62 (a touch wider than the nominal 0.55), so use that to keep long
		// lines from bleeding off the right edge.
		const EM_ADVANCE = 0.62;

		// Navigation + fixed HUD chrome own their OWN mobile layout (nav-layer builds
		// a touch dock at the true edge; RouteStage pins the pause/telemetry corners).
		// Recentering/reclaiming them like body content would drag the edge dock into
		// the page or push the corners off-screen — so pass all chrome through with
		// only a modest size bump for touch legibility.
		const isChrome = (k?: string) =>
			!!k &&
			(k.startsWith('route-nav') || k === 'route-chrome' || k === 'route-telemetry' || k === 'route-status' || k === 'route-status-pulse');

		return items.map((item) => {
			if (isChrome(item.kind)) {
				// Only grow slightly (touch legibility) — keep the authored x/y so the
				// nav dock and HUD corners stay exactly where nav-layer/RouteStage put
				// them for mobile. Clamp size growth so corners don't overlap.
				const s = (item.size ?? 0.03) * Math.min(1.5, 1.1 * inv);
				return { ...item, size: s };
			}
			const baseSize = item.size ?? 0.075;
			// Start from the desired boosted size, then cap it so the WHOLE line fits
			// the usable width without needing per-character wrapping. A line of N
			// chars at layout size s spans ~N*s*EM_ADVANCE in NDC-x.
			const chars = Math.max(1, item.text.length);
			const usableW = RIGHT_MARGIN - Math.max(LEFT_MARGIN, item.x * (1 - xPull));
			const fitSize = usableW / (chars * EM_ADVANCE);
			// Never shrink below the authored size (desktop was already legible); never
			// grow past what fits. This keeps long vitals on ONE line and readable.
			const size = Math.max(baseSize, Math.min(baseSize * sizeWant, fitSize));
			// Keep the left anchor inside the margin after the centre-pull.
			const x = Math.max(LEFT_MARGIN, item.x * (1 - xPull));
			// Clamp the reclaimed y so a row near the authored edge (±0.9) still maps
			// to an on-screen y inside the safe band (±0.92 screen → ±0.92*inv layout).
			const maxYLayout = 0.92 * inv;
			let y = item.y * yReclaim;
			if (y > maxYLayout) y = maxYLayout;
			else if (y < -maxYLayout) y = -maxYLayout;
			// Wrap paragraph-style items (those that opted into maxWidthEm) at a width
			// that fits the screen; leave single-line vitals alone (fitSize keeps them
			// on one line). Convert the fit width into an em cap for the chosen size.
			// Floor at 14 em so we NEVER wrap down toward one-character-per-line (the
			// stray vertical letter column bug) — better to slightly overflow a long
			// unbreakable token than to stack it vertically.
			const emThatFit = Math.floor(usableW / (size * EM_ADVANCE));
			const maxWidthEm =
				item.maxWidthEm != null ? Math.max(14, Math.min(item.maxWidthEm, emThatFit)) : item.maxWidthEm;
			const adapted = { ...item, x, y, size, maxWidthEm };
			if (typeof window !== 'undefined' && window.location?.search.includes('dbg=1')) {
				(window as unknown as { __adaptDbg?: unknown[] }).__adaptDbg ??= [];
				(window as unknown as { __adaptDbg: unknown[] }).__adaptDbg.push({
					id: item.id,
					text: item.text?.slice(0, 24),
					x: +x.toFixed(3),
					y: +y.toFixed(3),
					size: +size.toFixed(4),
					maxWidthEm
				});
			}
			return adapted;
		});
	}

	private uploadItems(rawItems: TextLayerItem[]): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer || !this.atlas) return;
		this.ensurePipeline(device);
		const items = this.portraitAdapt(rawItems);
		const packed: number[] = [];
		const runs: TextRunRect[] = [];
		let glyphIndex = 0;

		items.forEach((item, itemIndex) => {
			const size = item.size ?? 0.075;
			const laid = layoutText(item.text, this.atlas!, {
				maxWidthEm: item.maxWidthEm
			});
			const color = item.color ?? DEFAULT_COLOR;
			const runGlyphs = laid;
			const glyphStart = glyphIndex;
			let x0 = Number.POSITIVE_INFINITY;
			let x1 = Number.NEGATIVE_INFINITY;
			let y0 = Number.POSITIVE_INFINITY;
			let y1 = Number.NEGATIVE_INFINITY;
			for (const glyph of runGlyphs) {
				const gx0 = item.x + glyph.x * size;
				const gx1 = item.x + (glyph.x + glyph.w) * size;
				const gy0 = item.y + glyph.y * size;
				const gy1 = item.y + (glyph.y + glyph.h) * size;
				x0 = Math.min(x0, gx0);
				x1 = Math.max(x1, gx1);
				y0 = Math.min(y0, gy0);
				y1 = Math.max(y1, gy1);
				packGlyph(packed, item, glyph, color, size, glyphIndex++);
			}
			if (runGlyphs.length > 0) {
				const id = item.id ?? `msdf-text:${itemIndex}`;
				runs.push({
					id,
					kind: item.kind ?? 'text',
					text: item.text,
					x0,
					x1,
					y0,
					y1,
					payload: item,
					glyphStart,
					glyphCount: glyphIndex - glyphStart
				});
				this.runDepths.set(id, clamp01(item.depth ?? 0.5));
			}
		});

		this.runs = runs;
		this.glyphCount = packed.length / GLYPH_FLOATS;
		const floats = new Float32Array(packed.length || GLYPH_FLOATS);
		floats.set(packed);
		this.ensureGlyphBuffer(device, Math.max(1, this.glyphCount));
		if (!this.glyphBuffer || !this.bindLayout) return;
		device.queue.writeBuffer(this.glyphBuffer, 0, floats);
		this.bindGroup = device.createBindGroup({
			label: 'msdf-text-bind-group',
			layout: this.bindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.glyphBuffer } },
				{ binding: 2, resource: this.atlas.sampler },
				{ binding: 3, resource: this.atlas.textureView }
			]
		});
	}

	private ensureGlyphBuffer(device: GPUDevice, glyphCount: number): void {
		if (this.glyphBuffer && this.glyphCapacity >= glyphCount) return;
		this.glyphBuffer?.destroy();
		this.glyphCapacity = Math.max(glyphCount, Math.ceil(this.glyphCapacity * 1.5), 32);
		this.glyphBuffer = device.createBuffer({
			label: 'msdf-text-glyphs',
			size: this.glyphCapacity * GLYPH_FLOATS * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || this.glyphCount <= 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.draw(6, this.glyphCount);
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		const aspect = Math.max(0.0001, (this.engine.params[6] || 1) / Math.max(1, this.engine.params[7] || 1));
		// Mirror the shader's square-in-both-orientations transform (msdf-text.wgsl):
		// x /= max(aspect,1); y *= min(aspect,1).
		const xScale = Math.max(aspect, 1);
		const yScale = Math.min(aspect, 1);
		for (const run of this.runs) {
			// Optional per-run hit padding (logical-NDC) widens the CLICK target
			// only — the visual glyph box is unchanged. Applied in screen space
			// (after the same aspect scale) so the pad is uniform on screen.
			const padX = (run.payload.hitPadX ?? run.payload.hitPad ?? 0) / xScale;
			const padY = (run.payload.hitPadY ?? run.payload.hitPad ?? 0) * yScale;
			const x0 = run.x0 / xScale - padX;
			const x1 = run.x1 / xScale + padX;
			const y0 = run.y0 * yScale - padY;
			const y1 = run.y1 * yScale + padY;
			if (ndcX >= x0 && ndcX <= x1 && ndcY >= y0 && ndcY <= y1) {
				return { id: run.id, kind: run.kind, payload: run.payload };
			}
		}
		return null;
	}

	setRunDepth(id: string | null, depth = 0.5): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.glyphBuffer) return;
		for (const run of this.runs) {
			const targetDepth = run.id === id ? depth : (run.payload.depth ?? 0.5);
			const clamped = clamp01(targetDepth);
			if (this.runDepths.get(run.id) === clamped) continue;
			this.runDepths.set(run.id, clamped);
			const one = new Float32Array([clamped]);
			for (let i = 0; i < run.glyphCount; i += 1) {
				const floatOffset = (run.glyphStart + i) * GLYPH_FLOATS + 14;
				device.queue.writeBuffer(this.glyphBuffer, floatOffset * 4, one);
			}
		}
	}

	dispose(): void {
		if (this.onResize && typeof window !== 'undefined') {
			window.removeEventListener('resize', this.onResize);
			window.removeEventListener('orientationchange', this.onResize);
		}
		this.onResize = null;
		if (this.resizeRaf) cancelAnimationFrame(this.resizeRaf);
		this.resizeRaf = 0;
		this.glyphBuffer?.destroy();
		this.glyphBuffer = null;
		this.atlas?.dispose();
		this.atlas = null;
		this.bindGroup = null;
		this.pipeline = null;
	}
}

function packGlyph(
	out: number[],
	item: TextLayerItem,
	glyph: GlyphInstance,
	color: [number, number, number, number],
	size: number,
	glyphIndex: number
): void {
	const ageFrame = (item.startFrame ?? 0) + glyphIndex * 2;
	const revealSpan = item.revealSpan ?? 18;
	out.push(
		item.x,
		item.y,
		glyph.w * size,
		glyph.h * size,
		glyph.x * size,
		glyph.y * size,
		0,
		0,
		glyph.u,
		glyph.v,
		glyph.u + glyph.uw,
		glyph.v + glyph.vh,
		ageFrame,
		revealSpan,
		clamp01(item.depth ?? 0.5),
		clamp01(item.weight ?? 0.5),
		color[0],
		color[1],
		color[2],
		color[3]
	);
}

function clamp01(value: number): number {
	return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
}

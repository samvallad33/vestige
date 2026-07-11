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

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
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

	private uploadItems(items: TextLayerItem[]): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer || !this.atlas) return;
		this.ensurePipeline(device);
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

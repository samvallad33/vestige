/**
 * LivingFieldPass — the reusable "alive field" render pass (v2.3).
 *
 * Generalized from timeline-pass.ts (the 768-cell galaxy gold standard) so every
 * sparse organ can go from a black void with tiny 2D text to a full-bleed moving
 * bioluminescent field by writing only a ~20-line CPU mapper:
 *
 *     const cells = myData.map((d, i) => cellFromDatum(d, i));
 *     const pass = new LivingFieldPass(engine);
 *     pass.setCells(cells, { endangeredFrac, ... });
 *
 * Pipeline per frame (matches the verified recipe):
 *   compute():  splat all cells additively into a low-res rgba16float density
 *               field, then separable-blur it as TWO render passes (trap T3 — no
 *               compute blur on rgba16float on M-series).
 *   render():   1) fullscreen membrane base coat (fills the void, breathes),
 *               2) sharp bioluminescent cells on top (twinkle + HDR bloom).
 *
 * Every animated value is a pure function of params.time + per-cell phase
 * (deterministic; no Math.random/Date.now). pickAt() mirrors the WGSL orbit()
 * exactly so clicks land on the CURRENTLY-ANIMATED cell, not its static home
 * (the hitPad/orbit-pick trap).
 */

import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';

export interface LivingCell {
	/** Base NDC x (pre-orbit). The orbit spins this around the origin. */
	x: number;
	/** Base NDC y (pre-orbit). */
	y: number;
	/** Billboard radius in NDC (~0.012–0.05). Bigger data → bigger cell. */
	radius: number;
	/** rgb 0..1 — the cell's meaning color (retention oxygen, immune scarlet, …). */
	hue: [number, number, number];
	/** 0..1 brightness / activation — drives glow + splat weight. */
	energy: number;
	/** 0..1 orbit phase (also the twinkle seed). Usually i / count. */
	phase: number;
	/** Stable id for pickAt → onpick. */
	pickId: string;
	/** Optional kind label passed through the pick payload. */
	kind?: string;
	/** Optional free payload returned on pick. */
	payload?: unknown;
	/** true → drawn selected (bright rim, bigger). */
	selected?: boolean;
	/** true → endangered / suppressed: scar ring + red seam. */
	scar?: boolean;
	/** 0..1 secondary metric (retention-ish) — tints the membrane oxygen. */
	metric2?: number;
	/** Orbit spin multiplier (1 default; 0 pins a cell still). */
	spin?: number;
	/** Twinkle seed override (else derived from phase). */
	seed?: number;
}

/** Real route-level metrics that drive the membrane substrate. All 0..1. */
export interface LivingFieldScalars {
	/** Extra ambient plasma floor so even a sparse organ fills the void. 0..1. */
	ambient?: number;
}

const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_CELLS = 2048;
const CELL_FLOATS = 16;

import {
	FIELD_SPLAT_WGSL,
	FIELD_BLUR_WGSL,
	FIELD_MEMBRANE_WGSL,
	FIELD_CELL_WGSL
} from './living-field.wgsl';

export interface LivingPick {
	id: string;
	kind: string;
	index?: number;
	payload?: unknown;
}

type GpuResources = {
	cellBuffer: GPUBuffer;
	blurHBuffer: GPUBuffer;
	blurVBuffer: GPUBuffer;
	optsBuffer: GPUBuffer;
	splatBindGroup: GPUBindGroup;
	cellBindGroup: GPUBindGroup;
	blurHBindGroup: GPUBindGroup;
	blurVBindGroup: GPUBindGroup;
	membraneBindGroup: GPUBindGroup;
	fieldA: GPUTexture;
	fieldB: GPUTexture;
	fieldAView: GPUTextureView;
	fieldBView: GPUTextureView;
	fieldSize: [number, number];
};

export class LivingFieldPass implements FramePass {
	private engine: ObservatoryEngine;
	private cells: LivingCell[] = [];
	private scalars: LivingFieldScalars = {};
	// 0..1 field intensity — how bright the whole field renders. Text-heavy organs
	// set this LOW (~0.2) so the field is a dim backdrop the labels read over;
	// pure-visual organs (graph/timeline) keep it high. Default is a calm backdrop,
	// NOT a full-brightness blast, because most organs carry readable text.
	private intensity = 0.28;
	// Reading well (NDC rect) — the field dims inside it so text reads. hw<=0 = off.
	private well = { x: 0, y: 0, hw: -1, hh: 0, floor: 0.1, soft: 0.22 };
	// Portrait well-reflow: last aspect bucket, so compute() only re-writes opts on change.
	private lastAspectBucket = -999;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private splatBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private splatPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private cellPipeline: GPURenderPipeline | null = null;
	private cellCount = 0;
	private hoverIndex = -1;

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
	}

	/**
	 * Set the field intensity (0..1). LOW (~0.15-0.28) = dim backdrop for text
	 * organs; HIGH (~0.6-1.0) = the field is the hero (graph/timeline). Call
	 * before setCells (it's baked into the uploaded cells).
	 */
	setIntensity(v: number): void {
		this.intensity = Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0.28));
		const d = this.engine.gpuDevice;
		if (d) this.writeOpts(d); // membrane picks it up immediately
		// cells read intensity from extra.z, so re-bake the cell buffer too
		if (this.cells.length) this.setCells(this.cells, this.scalars);
	}

	/**
	 * Hover is a spare-cell float + rim, driven by FieldOpts.hover_index so an
	 * 8Hz pickAt never rewrites the cell storage buffer.
	 */
	setHovered(index: number): void {
		const next = Number.isFinite(index) ? Math.trunc(index) : -1;
		if (next === this.hoverIndex) return;
		this.hoverIndex = next;
		const device = this.engine.gpuDevice;
		if (device) this.writeOpts(device);
	}

	/** Upload a fresh set of cells (a data change). Rebuilds the GPU buffer. */
	setCells(cells: LivingCell[], scalars: LivingFieldScalars = {}): void {
		this.cells = cells.slice(0, MAX_CELLS);
		this.scalars = scalars;
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.splatPipeline || !this.engine.paramsBuffer) return;
		const splatModule = diagShader(device, 'living-field-splat', FIELD_SPLAT_WGSL);
		const blurModule = diagShader(device, 'living-field-blur', FIELD_BLUR_WGSL);
		const membraneModule = diagShader(device, 'living-field-membrane', FIELD_MEMBRANE_WGSL);
		const cellModule = diagShader(device, 'living-field-cell', FIELD_CELL_WGSL);

		// splat + cell share: uniform params (0) + storage cells (1) + FieldOpts (2).
		// (fs_splat ignores binding 2; the cell shader reads it for intensity+well.)
		this.splatBindLayout = device.createBindGroupLayout({
			label: 'living-field-splat-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'living-field-blur-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'living-field-membrane-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		const splatLayout = device.createPipelineLayout({ label: 'living-field-splat-pl', bindGroupLayouts: [this.splatBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'living-field-blur-pl', bindGroupLayouts: [this.blurBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'living-field-membrane-pl', bindGroupLayouts: [this.membraneBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

		const additive: GPUBlendState = {
			color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
			alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
		};
		this.splatPipeline = device.createRenderPipeline({
			label: 'living-field-splat',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_splat' },
			fragment: { module: splatModule, entryPoint: 'fs_splat', targets: [{ format: FIELD_FORMAT, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.blurPipeline = device.createRenderPipeline({
			label: 'living-field-blur',
			layout: blurLayout,
			vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] },
			primitive: { topology: 'triangle-list' }
		});
		this.membranePipeline = device.createRenderPipeline({
			label: 'living-field-membrane',
			layout: membraneLayout,
			vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.cellPipeline = device.createRenderPipeline({
			label: 'living-field-cells',
			layout: splatLayout,
			vertex: { module: cellModule, entryPoint: 'vs_cell' },
			fragment: { module: cellModule, entryPoint: 'fs_cell', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.splatBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let cellBuffer = this.resources?.cellBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		let optsBuffer = this.resources?.optsBuffer;
		if (!cellBuffer) cellBuffer = device.createBuffer({ label: 'living-field-cells', size: MAX_CELLS * CELL_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'living-field-blur-h', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'living-field-blur-v', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!optsBuffer) {
			// FieldOpts: intensity + reading-well rect. 8 floats = 32 bytes.
			optsBuffer = device.createBuffer({ label: 'living-field-opts', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
		}
		if (!needsTextures && this.resources) return;
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'living-field-a', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'living-field-b', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const splatBindGroup = device.createBindGroup({
			label: 'living-field-splat-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: cellBuffer } },
				{ binding: 2, resource: { buffer: optsBuffer } }
			]
		});
		const cellBindGroup = device.createBindGroup({
			label: 'living-field-cell-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: cellBuffer } },
				{ binding: 2, resource: { buffer: optsBuffer } }
			]
		});
		const blurHBindGroup = device.createBindGroup({ label: 'living-field-blur-h-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldAView }, { binding: 2, resource: { buffer: blurHBuffer } }] });
		const blurVBindGroup = device.createBindGroup({ label: 'living-field-blur-v-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldBView }, { binding: 2, resource: { buffer: blurVBuffer } }] });
		const membraneBindGroup = device.createBindGroup({ label: 'living-field-membrane-bind', layout: this.membraneBindLayout, entries: [{ binding: 0, resource: { buffer: this.engine.paramsBuffer } }, { binding: 2, resource: { buffer: optsBuffer } }, { binding: 3, resource: this.sampler }, { binding: 4, resource: fieldAView }] });
		this.resources = { cellBuffer, blurHBuffer, blurVBuffer, optsBuffer, splatBindGroup, cellBindGroup, blurHBindGroup, blurVBindGroup, membraneBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
		this.writeOpts(device);
	}

	/** Write the FieldOpts uniform (intensity + reading-well rect). */
	private writeOpts(device: GPUDevice): void {
		if (!this.resources) return;
		// Portrait: the text layer recentres x and reclaims vertical spread on a
		// phone (TextLayerPass.portraitAdapt). Apply the SAME transform to the
		// reading well so the dimmed region still sits under the reflowed text,
		// driven entirely by the live viewport aspect — nothing hardcoded per page.
		const well = this.portraitWell();
		device.queue.writeBuffer(
			this.resources.optsBuffer,
			0,
			new Float32Array([
				this.intensity,
				well.x,
				well.y,
				well.hw,
				well.hh,
				well.floor,
				well.soft,
				this.hoverIndex
			])
		);
	}

	/** Coarse aspect bucket (matches TextLayerPass) so we only re-write on change. */
	private aspectBucket(): number {
		let vw = this.engine.params[6] || 0;
		let vh = this.engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return -999;
		return Math.round((vw / vh) * 8);
	}

	/** Mirror TextLayerPass.portraitAdapt's x-recentre + y-reclaim on the well. */
	private portraitWell(): {
		x: number;
		y: number;
		hw: number;
		hh: number;
		floor: number;
		soft: number;
	} {
		if (this.well.hw <= 0) return this.well;
		let vw = this.engine.params[6] || 0;
		let vh = this.engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return this.well;
		const aspect = vw / vh;
		if (aspect >= 0.85) return this.well;
		const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.46));
		const xPull = 0.42 * portraitness;
		const yReclaim = 1 + (1 / Math.max(aspect, 0.2) - 1) * (0.72 * portraitness);
		// Widen the well to cover the whole narrow column + a little slack.
		const widen = 1 + 0.25 * portraitness;
		return {
			x: this.well.x * (1 - xPull),
			y: clamp(this.well.y * yReclaim, -0.98, 0.98),
			hw: Math.min(1.1, this.well.hw * widen),
			hh: Math.min(1.1, this.well.hh * yReclaim),
			floor: this.well.floor,
			soft: this.well.soft
		};
	}

	/**
	 * Set a "reading well": the field emits LESS inside this NDC rectangle so text
	 * on top reads clearly. hw<=0 (default) disables it → field renders full. Call
	 * with the region your MSDF text occupies (e.g. the left instrument column).
	 */
	setReadingWell(r: { x: number; y: number; hw: number; hh: number; floor?: number; soft?: number }): void {
		this.well = {
			x: finite(r.x),
			y: finite(r.y),
			hw: finite(r.hw, -1),
			hh: finite(r.hh),
			floor: clamp01(r.floor ?? 0.1),
			soft: Math.max(0.02, finite(r.soft ?? 0.22, 0.22))
		};
		const d = this.engine.gpuDevice;
		if (d) this.writeOpts(d);
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources) return;
		const data = new Float32Array(MAX_CELLS * CELL_FLOATS);
		this.cellCount = Math.min(MAX_CELLS, this.cells.length);
		for (let i = 0; i < this.cellCount; i++) {
			const c = this.cells[i];
			// Finite-guard every value that reaches the GPU: one NaN/Infinity in a
			// position or hue poisons the additive splat (NaN spreads through the
			// blur to the whole membrane). A bad organ mapper (score/0, a degenerate
			// layout) must not be able to black out the field.
			const x = finite(c.x);
			const y = finite(c.y);
			const ringR = Math.hypot(x, y);
			const phase = finite(c.phase);
			// flags: bit0 selected, bit1 scar, bit2 pulse-strong (energy>0.8)
			let flags = 0;
			if (c.selected) flags |= 1;
			if (c.scar) flags |= 2;
			if (finite(c.energy) > 0.8) flags |= 4;
			const o = i * CELL_FLOATS;
			data[o + 0] = x;
			data[o + 1] = y;
			data[o + 2] = Math.max(0.006, finite(c.radius, 0.02));
			data[o + 3] = ringR;
			data[o + 4] = finite(c.hue[0]);
			data[o + 5] = finite(c.hue[1]);
			data[o + 6] = finite(c.hue[2]);
			data[o + 7] = clamp01(c.energy);
			data[o + 8] = phase;
			data[o + 9] = flags;
			data[o + 10] = clamp01(c.metric2 ?? c.energy);
			data[o + 11] = finite(c.spin ?? 1, 1);
			data[o + 12] = i;
			data[o + 13] = finite(c.seed ?? phase * 97.13);
			data[o + 14] = this.intensity; // extra.z — per-field intensity the shader dims by
			data[o + 15] = 0;
		}
		// NOTE: do NOT write engine.params[2] (node_count) here. That lane is SHARED
		// with any co-resident scene pass (the recall-path NodeRenderer on graph/
		// observatory/memories reads params.node_count to cull its instances). The
		// field draws this.cellCount instances directly in render() and never reads
		// node_count, so writing it would silently corrupt a sibling pass's cull.
		device.queue.writeBuffer(this.resources.cellBuffer, 0, data);
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.splatPipeline || !this.blurPipeline) return;
		this.ensureResources(device);
		// Re-write the reading-well opts when the viewport aspect crosses a bucket
		// (phone rotate / window resize) so the portrait-adapted well tracks the
		// reflowed text. Cheap (one 32-byte write) and only fires on real changes.
		const bucket = this.aspectBucket();
		if (bucket !== this.lastAspectBucket) {
			this.lastAspectBucket = bucket;
			this.writeOpts(device);
		}
		const res = this.resources;
		const splat = encoder.beginRenderPass({ label: 'living-field-splat-pass', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		splat.setPipeline(this.splatPipeline);
		splat.setBindGroup(0, res.splatBindGroup);
		if (this.cellCount > 0) splat.draw(6, this.cellCount);
		splat.end();
		// Separable blur — H (A→B) then V (B→A). Render passes, never compute (T3).
		const blurH = encoder.beginRenderPass({ label: 'living-field-blur-h', colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();
		const blurV = encoder.beginRenderPass({ label: 'living-field-blur-v', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.cellPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		if (this.cellCount > 0) {
			pass.setPipeline(this.cellPipeline);
			pass.setBindGroup(0, this.resources.cellBindGroup);
			pass.draw(6, this.cellCount);
		}
	}

	/** CPU mirror of the WGSL ring_spin()/orbit() — for accurate picking. */
	private orbitCpu(bx: number, by: number, phase01: number, spinScale: number): { x: number; y: number } {
		const radius = Math.hypot(bx, by);
		if (radius < 0.0001) return { x: bx, y: by };
		const time = this.engine.params[10] || 0;
		const ang0 = Math.atan2(by, bx);
		const spin = (0.045 + phase01 * 0.1) * time * spinScale;
		const ang = ang0 + spin + Math.sin(time * 0.6 + phase01 * Math.PI * 2) * 0.02;
		const rr = radius * (1 + 0.016 * Math.sin(time * 1.1 + phase01 * Math.PI * 2));
		return { x: Math.cos(ang) * rr, y: Math.sin(ang) * rr };
	}

	pickAt(ndcX: number, ndcY: number): LivingPick | null {
		let best: LivingPick | null = null;
		let bestDist = Infinity;
		for (let i = 0; i < this.cellCount; i++) {
			const c = this.cells[i];
			const p = this.orbitCpu(c.x, c.y, c.phase, c.spin ?? 1);
			const d = Math.hypot(ndcX - p.x, ndcY - p.y);
			const hit = Math.max(0.04, c.radius * 2.6);
			if (d <= hit && d < bestDist) {
				best = { id: c.pickId, kind: c.kind ?? 'living-cell', index: i, payload: c.payload ?? c };
				bestDist = d;
			}
		}
		this.setHovered(best?.index ?? -1);
		return best;
	}

	dispose(): void {
		this.resources?.cellBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.optsBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}
}

function clamp01(v: number): number {
	return Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));
}

function clamp(v: number, lo: number, hi: number): number {
	return Math.min(hi, Math.max(lo, Number.isFinite(v) ? v : lo));
}

/** Coerce a value to a finite number so no NaN/Infinity reaches the GPU buffer. */
function finite(v: number, fallback = 0): number {
	return Number.isFinite(v) ? v : fallback;
}

function diagShader(device: GPUDevice, label: string, code: string): GPUShaderModule {
	device.pushErrorScope('validation');
	const module = device.createShaderModule({ label, code });
	void module.getCompilationInfo().then((info) => {
		for (const message of info.messages) {
			if (message.type === 'error') console.error(`[living-field] ${label} WGSL ${message.type} ${message.lineNum}:${message.linePos} ${message.message}`);
		}
	});
	void device.popErrorScope().then((error) => {
		if (error) console.error(`[living-field] ${label} shader module validation: ${error.message}`);
	});
	return module;
}

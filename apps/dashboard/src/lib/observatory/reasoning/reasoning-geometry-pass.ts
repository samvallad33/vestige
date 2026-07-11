// ─────────────────────────────────────────────────────────────────────────────
// Reasoning trace — ALIVE geometry pass.
//
// Renders the causal beam, tapered evidence ribbons, contradiction interference
// fringes, supersession scars, and the recommendation nucleus as INSTANCED quads
// with SDF fragment shading, additively into the HDR scene (the PostChain tone-
// maps + can bloom afterward). Deterministic: all motion is driven by params.time
// (fixed sim seconds), never Math.random / Date.now. Sits UNDER the MSDF text
// pass so labels stay crisp on top.
//
// WebGPU traps baked in (learned from Organ 1): no read_write storage shared
// across pipelines (this pass is render-only), no storage textures (all quads),
// no WGSL reserved words (active/filter/sample/texture/common/override/…).
// ─────────────────────────────────────────────────────────────────────────────

import type { FramePass, ObservatoryEngine } from '$lib/observatory/engine';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import { computeTraceLayout, type TraceLayout } from './trace-layout';
import type { ReasoningScene } from './reasoning-scene';
import { REASONING_GEOMETRY_WGSL } from './reasoning-geometry.wgsl';

// Instance = one primitive. Floats per instance (std layout, 12 floats = 48B):
//   [0,1] a.xy     (NDC endpoint A / center)
//   [2,3] b.xy     (NDC endpoint B; == a for point primitives)
//   [4]   kind     (0 beam, 1 ribbon, 2 nucleus, 3 fringe, 4 scar)
//   [5]   thickness(NDC half-width)
//   [6]   trust    (0..1 → VSUP saturation)
//   [7]   sign     (+1 support / −1 oppose → PRGn hue)
//   [8]   energy   (0..1 brightness / flow amount)
//   [9]   seed     (phase offset for deterministic per-instance variation)
//   [10]  extra    (kind-specific: fringe strength / nucleus confidence)
//   [11]  pad
const INSTANCE_FLOATS = 12;
const MAX_INSTANCES = 512;

export const KIND = { beam: 0, ribbon: 1, nucleus: 2, fringe: 3, scar: 4 } as const;

class ReasoningGeometryPass implements FramePass {
	private engine: ObservatoryEngine;
	private layout: TraceLayout | null = null;
	private pipeline: GPURenderPipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private instanceBuffer: GPUBuffer | null = null;
	private instanceCount = 0;
	private ready = false;

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
	}

	uploadScene(scene: RouteSceneModel): void {
		this.layout =
			(scene as ReasoningScene)?.organ === 'reasoning'
				? computeTraceLayout(scene as ReasoningScene)
				: null;
		this.ensurePipeline();
		this.uploadInstances();
	}

	private ensurePipeline(): void {
		if (this.pipeline || this.ready) return;
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		this.ready = true;

		this.instanceBuffer = device.createBuffer({
			label: 'reasoning-geometry-instances',
			size: MAX_INSTANCES * INSTANCE_FLOATS * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		const module = device.createShaderModule({
			label: 'reasoning-geometry-wgsl',
			code: REASONING_GEOMETRY_WGSL
		});
		const bindLayout = device.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
			]
		});
		this.bindGroup = device.createBindGroup({
			layout: bindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.instanceBuffer } }
			]
		});
		// Additive over the HDR scene → bright cores bloom in the post stack.
		const additive: GPUBlendState = {
			color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
			alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
		};
		this.pipeline = device.createRenderPipeline({
			label: 'reasoning-geometry-pipeline',
			layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
			vertex: { module, entryPoint: 'vs_geo' },
			fragment: {
				module,
				entryPoint: 'fs_geo',
				targets: [{ format: this.engine.sceneFormat, blend: additive }]
			},
			primitive: { topology: 'triangle-list' }
		});
	}

	private uploadInstances(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.instanceBuffer) return;
		const L = this.layout;
		if (!L) {
			this.instanceCount = 0;
			return;
		}
		const data = new Float32Array(MAX_INSTANCES * INSTANCE_FLOATS);
		let n = 0;
		const push = (v: number[]) => {
			if (n >= MAX_INSTANCES) return;
			data.set(v, n * INSTANCE_FLOATS);
			n++;
		};

		// 1. BEAM segments between consecutive gates (bright where both lit).
		for (let i = 0; i < L.gates.length - 1; i++) {
			const a = L.gates[i];
			const b = L.gates[i + 1];
			const energy = a.lit && b.lit ? 1 : a.lit || b.lit ? 0.5 : 0.18;
			push([a.x, 0, b.x, 0, KIND.beam, 0.006, 1, 1, energy, i, 0, 0]);
		}
		// 2. RIBBONS evidence → nucleus (tapered, flowing, PRGn by sign).
		for (let i = 0; i < L.ribbons.length; i++) {
			const r = L.ribbons[i];
			push([r.fromX, r.fromY, r.toX, r.toY, KIND.ribbon, 0.004 + 0.01 * r.trust, r.trust, r.sign, 0.5 + 0.5 * r.trust, i * 1.7, 0, 0]);
		}
		// 3. SCARS (superseded) + amber transfer filament.
		for (let i = 0; i < L.scars.length; i++) {
			const s = L.scars[i];
			push([s.x, s.y, s.toX, s.toY, KIND.scar, 0.006, 0.5, 1, 0.7, i * 2.3, 0, 0]);
		}
		// 4. FRINGES (contradiction interference).
		for (let i = 0; i < L.fringes.length; i++) {
			const f = L.fringes[i];
			push([f.ax, f.ay, f.bx, f.by, KIND.fringe, 0.02, 0.9, -1, f.strength, i * 3.1, f.strength, 0]);
		}
		// 5. NUCLEUS (recommendation — confidence = size + coherence).
		if (L.nucleus) {
			const c = L.nucleus.confidence;
			push([L.nucleus.x, L.nucleus.y, L.nucleus.x, L.nucleus.y, KIND.nucleus, 0.035 + 0.05 * c, c, 1, 0.8 + 0.2 * c, 0, c, 0]);
		}

		this.instanceCount = n;
		device.queue.writeBuffer(this.instanceBuffer, 0, data, 0, n * INSTANCE_FLOATS);
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || this.instanceCount === 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.draw(6, this.instanceCount); // 6 verts (quad) × instances
	}

	dispose(): void {
		this.instanceBuffer?.destroy();
		this.instanceBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
	}
}

export function createReasoningGeometryPass(engine: ObservatoryEngine): FramePass {
	return new ReasoningGeometryPass(engine);
}

/**
 * Cognitive Observatory — forgetting-horizon choreography pass.
 *
 * Compute-only FramePass (no render(): nodes and ribbons draw via the
 * NodeRenderer's existing pipelines). Uploads the per-node packed horizon word
 * once (static), then each frame extends the choreography INTO the NodeState
 * demo lanes as a pure function of (frame, role, rank) — no stateful
 * integration, so capture mode (?frame=N) works with zero special-casing.
 *
 * PASS ORDER IS LOAD-BEARING: this pass MUST be constructed AFTER the
 * NodeRenderer (the route guarantees it: handleReady creates NodeRenderer,
 * the upload $effect creates ForgettingRenderer) so forgetting_choreo encodes
 * AFTER recall_sim in the same encoder and its demo-lane overwrite wins.
 * recall_sim rewrites demo.x every frame with an afterglow window of
 * bf+40..bf+200 — the k=2 rescue ribbon at bf=438 would otherwise carry
 * residual ignition into the master release.
 *
 * Three independent walls keep OTHER demos pixel-identical:
 *  (a) the route constructs this renderer only in the forgetting-horizon branch,
 *  (b) compute() gates on params[9] === 3 ('forgetting-horizon' demo index),
 *  (c) the demo-3 vertex/fragment terms in render-nodes.wgsl are themselves
 *      gated on params.demo_id == 3.0.
 */

import type { ObservatoryEngine, FramePass } from './engine';
import type { NodeRenderer } from './node-renderer';
import type { ForgettingPlan } from './forgetting-plan';
import { forgettingWGSL } from './shaders/forgetting.wgsl';

/** DEMO_MODES.indexOf('forgetting-horizon') — types.ts, verified index 3. */
const HORIZON_DEMO_ID = 3;

export interface ForgettingRendererOptions {
	engine: ObservatoryEngine;
	nodeRenderer: NodeRenderer;
	plan: ForgettingPlan;
}

export class ForgettingRenderer implements FramePass {
	private engine: ObservatoryEngine;
	private nodeRenderer: NodeRenderer;
	private plan: ForgettingPlan;

	private pipeline: GPUComputePipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private horizonBuffer: GPUBuffer | null = null;

	constructor(opts: ForgettingRendererOptions) {
		this.engine = opts.engine;
		this.nodeRenderer = opts.nodeRenderer;
		this.plan = opts.plan;
		this.engine.addPass(this);
	}

	/** Create the horizon buffer + compute pipeline. Call after NodeRenderer.upload(). */
	upload(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		if (!this.plan.viable) return;
		if (!this.nodeRenderer.nodeStateBuffer || this.nodeRenderer.nodeCountValue === 0) return;

		this.horizonBuffer?.destroy();
		this.horizonBuffer = device.createBuffer({
			label: 'observatory-forgetting-horizon',
			size: Math.max(4, this.plan.horizonData.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.horizonBuffer, 0, this.plan.horizonData.buffer as ArrayBuffer);

		const module = device.createShaderModule({
			label: 'observatory-forgetting-choreo',
			code: forgettingWGSL
		});
		this.pipeline = device.createComputePipeline({
			label: 'observatory-forgetting-choreo',
			layout: 'auto',
			compute: { module, entryPoint: 'forgetting_choreo' }
		});

		// EXACTLY the 3 declared bindings — auto layout strips unused bindings
		// and binding anything extra invalidates the group (the BirthRenderer
		// lesson, birth-renderer.ts createComputePipeline).
		this.bindGroup = device.createBindGroup({
			label: 'observatory-forgetting-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.nodeRenderer.nodeStateBuffer } },
				{ binding: 2, resource: { buffer: this.horizonBuffer } }
			]
		});
	}

	/** FramePass — overwrite the four demo lanes for this frame (pure of frame). */
	compute(encoder: GPUCommandEncoder): void {
		if (this.engine.params[9] !== HORIZON_DEMO_ID) return;
		if (!this.pipeline || !this.bindGroup) return;
		const n = this.nodeRenderer.nodeCountValue;
		if (n === 0) return;

		const pass = encoder.beginComputePass({ label: 'observatory-forgetting-choreo' });
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.dispatchWorkgroups(Math.ceil(n / 64));
		pass.end();
	}

	dispose(): void {
		this.horizonBuffer?.destroy();
		this.horizonBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
	}
}

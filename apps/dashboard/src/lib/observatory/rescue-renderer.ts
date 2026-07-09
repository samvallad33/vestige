/**
 * Cognitive Observatory — salience-rescue choreography pass.
 *
 * Compute-only FramePass (no render(): nodes and ribbons draw via the
 * NodeRenderer's existing pipelines). Uploads the per-node packed wave word
 * once (static), then each frame extends the choreography INTO the NodeState
 * demo lanes as a pure function of (frame, role, hopDepth) — no stateful
 * integration, so capture mode (?frame=N) works with zero special-casing.
 *
 * PASS ORDER IS LOAD-BEARING: this pass MUST be constructed AFTER the
 * NodeRenderer (the route guarantees it: handleReady creates NodeRenderer,
 * the upload $effect creates RescueRenderer) so rescue_choreo encodes AFTER
 * recall_sim in the same encoder and its demo-lane overwrite wins. recall_sim
 * rewrites demo.x every frame with an afterglow window of bf+40..bf+200 — the
 * causal arc at bf=560 would otherwise carry visible residual across the
 * 719→0 loop seam.
 *
 * Three independent walls keep OTHER demos pixel-identical:
 *  (a) the route constructs this renderer only in the rescue branch,
 *  (b) compute() gates on params[9] === 2 ('salience-rescue' demo index),
 *  (c) demo.y/.z/.w have no other writer, so the new render-nodes terms
 *      multiply/add exact 0.0 elsewhere.
 */

import type { ObservatoryEngine, FramePass } from './engine';
import type { NodeRenderer } from './node-renderer';
import type { RescuePlan } from './rescue-plan';
import { rescueWGSL } from './shaders/rescue.wgsl';

/** DEMO_MODES.indexOf('salience-rescue') — types.ts, verified index 2. */
const RESCUE_DEMO_ID = 2;

export interface RescueRendererOptions {
	engine: ObservatoryEngine;
	nodeRenderer: NodeRenderer;
	plan: RescuePlan;
}

export class RescueRenderer implements FramePass {
	private engine: ObservatoryEngine;
	private nodeRenderer: NodeRenderer;
	private plan: RescuePlan;

	private pipeline: GPUComputePipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private waveBuffer: GPUBuffer | null = null;

	constructor(opts: RescueRendererOptions) {
		this.engine = opts.engine;
		this.nodeRenderer = opts.nodeRenderer;
		this.plan = opts.plan;
		this.engine.addPass(this);
	}

	/** Create the wave buffer + compute pipeline. Call after NodeRenderer.upload(). */
	upload(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		if (!this.plan.viable) return;
		if (!this.nodeRenderer.nodeStateBuffer || this.nodeRenderer.nodeCountValue === 0) return;

		this.waveBuffer?.destroy();
		this.waveBuffer = device.createBuffer({
			label: 'observatory-rescue-wave',
			size: Math.max(4, this.plan.waveData.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.waveBuffer, 0, this.plan.waveData.buffer as ArrayBuffer);

		// hopSlot/causeDepth are baked into the shader as f32 literals — no
		// uniform buffer, so the auto layout has nothing to strip.
		const module = device.createShaderModule({
			label: 'observatory-rescue-choreo',
			code: rescueWGSL(this.plan.consts)
		});
		this.pipeline = device.createComputePipeline({
			label: 'observatory-rescue-choreo',
			layout: 'auto',
			compute: { module, entryPoint: 'rescue_choreo' }
		});

		// EXACTLY the 3 declared bindings — auto layout strips unused bindings
		// and binding anything extra invalidates the group (the BirthRenderer
		// lesson, birth-renderer.ts createComputePipeline).
		this.bindGroup = device.createBindGroup({
			label: 'observatory-rescue-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.nodeRenderer.nodeStateBuffer } },
				{ binding: 2, resource: { buffer: this.waveBuffer } }
			]
		});
	}

	/** FramePass — overwrite the four demo lanes for this frame (pure of frame). */
	compute(encoder: GPUCommandEncoder): void {
		if (this.engine.params[9] !== RESCUE_DEMO_ID) return;
		if (!this.pipeline || !this.bindGroup) return;
		const n = this.nodeRenderer.nodeCountValue;
		if (n === 0) return;

		const pass = encoder.beginComputePass({ label: 'observatory-rescue-choreo' });
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.dispatchWorkgroups(Math.ceil(n / 64));
		pass.end();
	}

	dispose(): void {
		this.waveBuffer?.destroy();
		this.waveBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
	}
}

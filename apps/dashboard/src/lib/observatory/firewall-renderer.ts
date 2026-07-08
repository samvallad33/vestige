/**
 * Cognitive Observatory — firewall choreography pass.
 *
 * Compute-only FramePass (no render(): nodes and probe beams draw via the
 * NodeRenderer's existing pipelines). Uploads the per-node packed fire word
 * once (static), then each frame extends the choreography INTO the NodeState
 * demo lanes as a pure function of (frame, role, shockDelay) — no stateful
 * integration, so capture mode (?frame=N) works with zero special-casing.
 *
 * PASS ORDER IS LOAD-BEARING: this pass MUST be constructed AFTER the
 * NodeRenderer (the route guarantees it: handleReady creates NodeRenderer,
 * the upload $effect creates FirewallRenderer) so firewall_choreo encodes
 * AFTER recall_sim in the same encoder and its demo-lane overwrite wins.
 * recall_sim rewrites demo.x every frame with an afterglow window of
 * bf+40..bf+200 — the k=5 sever beam at bf=450 would otherwise carry residual
 * ignition into the verdict window.
 *
 * Three independent walls keep OTHER demos pixel-identical:
 *  (a) the route constructs this renderer only in the firewall branch,
 *  (b) compute() gates on params[9] === 4 ('firewall' demo index),
 *  (c) the demo-4 vertex/fragment terms in render-nodes.wgsl are themselves
 *      gated on params.demo_id == 4.0.
 */

import type { ObservatoryEngine, FramePass } from './engine';
import type { NodeRenderer } from './node-renderer';
import type { FirewallPlan } from './firewall-plan';
import { firewallWGSL } from './shaders/firewall.wgsl';

/** DEMO_MODES.indexOf('firewall') — types.ts, verified index 4. */
const FIREWALL_DEMO_ID = 4;

export interface FirewallRendererOptions {
	engine: ObservatoryEngine;
	nodeRenderer: NodeRenderer;
	plan: FirewallPlan;
}

export class FirewallRenderer implements FramePass {
	private engine: ObservatoryEngine;
	private nodeRenderer: NodeRenderer;
	private plan: FirewallPlan;

	private pipeline: GPUComputePipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private fireBuffer: GPUBuffer | null = null;

	constructor(opts: FirewallRendererOptions) {
		this.engine = opts.engine;
		this.nodeRenderer = opts.nodeRenderer;
		this.plan = opts.plan;
		this.engine.addPass(this);
	}

	/** Create the fire buffer + compute pipeline. Call after NodeRenderer.upload(). */
	upload(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		if (!this.plan.viable) return;
		if (!this.nodeRenderer.nodeStateBuffer || this.nodeRenderer.nodeCountValue === 0) return;

		this.fireBuffer?.destroy();
		this.fireBuffer = device.createBuffer({
			label: 'observatory-firewall-fire',
			size: Math.max(4, this.plan.fireData.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.fireBuffer, 0, this.plan.fireData.buffer as ArrayBuffer);

		const module = device.createShaderModule({
			label: 'observatory-firewall-choreo',
			code: firewallWGSL
		});
		this.pipeline = device.createComputePipeline({
			label: 'observatory-firewall-choreo',
			layout: 'auto',
			compute: { module, entryPoint: 'firewall_choreo' }
		});

		// EXACTLY the 3 declared bindings — auto layout strips unused bindings
		// and binding anything extra invalidates the group (the BirthRenderer
		// lesson, birth-renderer.ts createComputePipeline).
		this.bindGroup = device.createBindGroup({
			label: 'observatory-firewall-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.nodeRenderer.nodeStateBuffer } },
				{ binding: 2, resource: { buffer: this.fireBuffer } }
			]
		});
	}

	/** FramePass — overwrite the four demo lanes for this frame (pure of frame). */
	compute(encoder: GPUCommandEncoder): void {
		if (this.engine.params[9] !== FIREWALL_DEMO_ID) return;
		if (!this.pipeline || !this.bindGroup) return;
		const n = this.nodeRenderer.nodeCountValue;
		if (n === 0) return;

		const pass = encoder.beginComputePass({ label: 'observatory-firewall-choreo' });
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.dispatchWorkgroups(Math.ceil(n / 64));
		pass.end();
	}

	dispose(): void {
		this.fireBuffer?.destroy();
		this.fireBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
	}
}

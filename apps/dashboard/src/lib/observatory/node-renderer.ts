/**
 * Cognitive Observatory — instanced node renderer (Increment 4).
 *
 * Owns the NodeState/EdgeIndex GPU buffers and the additive billboard
 * pipeline. Registers as a FramePass on the engine: camera uniforms are
 * written from the deterministic loop phase each frame (orbit), nodes draw
 * as instanced soft-glow sprites straight from the storage buffer.
 *
 * The RENDER LOOP does no GPU readback and holds no wall-clock state
 * (spec §6 — deterministic frames). pickAt() is the one sanctioned,
 * input-driven readback: click-frequency only, never per-frame.
 */

import type { GraphResponse } from '$types';
import type { ObservatoryEngine, FramePass } from './engine';
import { DemoClock } from './demo-clock';
import { orbitCamera } from './camera';
import {
	buildObservatoryGraph,
	buildNodeStateArray,
	buildEdgeIndexArray
} from './graph-upload';
import { FLOATS_PER_NODE, NODE_LANE, type ObservatoryGraph } from './types';
import { renderNodesWGSL } from './shaders/render-nodes.wgsl';
import { simulateWGSL } from './shaders/simulate.wgsl';
import { renderPathWGSL } from './shaders/render-path.wgsl';
import { buildRecallPath, type PathStepMeta } from './path-builder';

/** mat4 (16) + right vec4 (4) + up vec4 (4) floats. */
const CAMERA_FLOATS = 24;

/** Orbit distance fitted to the default field radius (graph-upload). */
const ORBIT_DISTANCE = 300;

export class NodeRenderer implements FramePass {
	private engine: ObservatoryEngine;
	private pipeline: GPURenderPipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private cameraBuffer: GPUBuffer | null = null;
	private nodeBuffer: GPUBuffer | null = null;
	private edgeBuffer: GPUBuffer | null = null;
	private cameraData = new Float32Array(CAMERA_FLOATS);
	private nodeCount = 0;

	// Recall-path simulation (Increment 5)
	private simPipeline: GPUComputePipeline | null = null;
	private simBindGroup: GPUBindGroup | null = null;
	private pathBuffer: GPUBuffer | null = null;

	// Path edge wavefront (Increment 6)
	private pathPipeline: GPURenderPipeline | null = null;
	private pathBindGroup: GPUBindGroup | null = null;
	private pathStepCount = 0;

	graph: ObservatoryGraph | null = null;
	/** Beat metadata for the timeline spine overlay (Increment 6). */
	pathSteps: PathStepMeta[] = [];

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
		engine.addPass(this);
	}

	/**
	 * Upload the graph into GPU buffers. Layout is deterministic: positions
	 * come from a fresh DemoClock PRNG seeded with `seed` (same seed →
	 * identical field, Increment 4 gate).
	 */
	upload(response: GraphResponse, seed: string, opts?: { recallPath?: boolean }): void {
		const device = this.engine.gpuDevice;
		if (!device) return;
		// Other demo modes (engram-birth, …) bring their own choreography and
		// upload with recallPath: false so the recall wave stays quiet.
		const includeRecallPath = opts?.recallPath ?? true;

		const graph = buildObservatoryGraph(response);
		this.graph = graph;

		const layoutClock = new DemoClock({ seed });
		const { data, nodeCount } = buildNodeStateArray(graph, layoutClock.state.rng);
		this.nodeCount = nodeCount;

		this.nodeBuffer?.destroy();
		// COPY_SRC exists solely for pickAt()'s click-time readback.
		this.nodeBuffer = device.createBuffer({
			label: 'observatory-node-state',
			size: Math.max(data.byteLength, 64),
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_DST |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.VERTEX
		});
		device.queue.writeBuffer(this.nodeBuffer, 0, data.buffer as ArrayBuffer);

		const edgeData = buildEdgeIndexArray(graph);
		this.edgeBuffer?.destroy();
		this.edgeBuffer = device.createBuffer({
			label: 'observatory-edge-index',
			size: edgeData.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.edgeBuffer, 0, edgeData.buffer as ArrayBuffer);

		// Recall path: deterministic story beats → PathStep storage buffer.
		// (Skipped for demo modes with their own choreography — the buffer is
		// still created so the sim bind group stays valid, just with 0 steps.)
		const recall = includeRecallPath
			? buildRecallPath(response, graph)
			: { steps: [], data: new Uint32Array(4) };
		this.pathSteps = recall.steps;
		this.pathBuffer?.destroy();
		this.pathBuffer = device.createBuffer({
			label: 'observatory-path-steps',
			size: recall.data.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.pathBuffer, 0, recall.data.buffer as ArrayBuffer);
		this.pathStepCount = this.pathSteps.length;

		// Per-frame counts for every shader that reads Params.
		this.engine.params[2] = nodeCount;
		this.engine.params[3] = graph.edges.length;
		this.engine.params[4] = this.pathSteps.length;

		if (!this.cameraBuffer) {
			this.cameraBuffer = device.createBuffer({
				label: 'observatory-camera',
				size: this.cameraData.byteLength,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		this.createPipeline(device);
	}

	/**
	 * Replace the PathStep buffer after upload (Moment B: the birth engrave
	 * steps ride the same wavefront machinery as recall). Rebuilds the
	 * pipelines/bind groups so they reference the new buffer.
	 */
	setPathSteps(data: Uint32Array<ArrayBuffer>, steps: PathStepMeta[]): void {
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.pathSteps = steps;
		this.pathStepCount = steps.length;
		this.pathBuffer?.destroy();
		this.pathBuffer = device.createBuffer({
			label: 'observatory-path-steps',
			size: Math.max(data.byteLength, 16),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.pathBuffer, 0, data.buffer as ArrayBuffer);
		this.engine.params[4] = steps.length;
		this.createPipeline(device);
	}

	private createPipeline(device: GPUDevice): void {
		if (!this.engine.paramsBuffer || !this.cameraBuffer || !this.nodeBuffer) return;

		// Recall-path simulation pipeline (compute-boids pattern, §1).
		if (this.pathBuffer) {
			const simModule = device.createShaderModule({
				label: 'observatory-simulate',
				code: simulateWGSL
			});
			this.simPipeline = device.createComputePipeline({
				label: 'observatory-recall-sim',
				layout: 'auto',
				compute: { module: simModule, entryPoint: 'recall_sim' }
			});
			const simEntries: GPUBindGroupEntry[] = [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.nodeBuffer } },
				{ binding: 2, resource: { buffer: this.pathBuffer } }
			];
			if (this.edgeBuffer) {
				simEntries.push({ binding: 3, resource: { buffer: this.edgeBuffer } });
			}
			this.simBindGroup = device.createBindGroup({
				label: 'observatory-recall-sim-bind',
				layout: this.simPipeline.getBindGroupLayout(0),
				entries: simEntries
			});
		}

		const module = device.createShaderModule({
			label: 'observatory-render-nodes',
			code: renderNodesWGSL
		});

		this.pipeline = device.createRenderPipeline({
			label: 'observatory-nodes',
			layout: 'auto',
			vertex: { module, entryPoint: 'vs_main' },
			fragment: {
				module,
				entryPoint: 'fs_main',
				targets: [
					{
						format: this.engine.sceneFormat,
						// Additive: light accumulates on the void (§7.2).
						blend: {
							color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
							alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
						}
					}
				]
			},
			primitive: { topology: 'triangle-list' }
		});

		this.bindGroup = device.createBindGroup({
			label: 'observatory-nodes-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.cameraBuffer } },
				{ binding: 2, resource: { buffer: this.nodeBuffer } }
			]
		});

		// Path wavefront ribbons (Increment 6) — drawn after nodes, additive.
		if (this.pathBuffer) {
			const pathModule = device.createShaderModule({
				label: 'observatory-render-path',
				code: renderPathWGSL
			});
			this.pathPipeline = device.createRenderPipeline({
				label: 'observatory-path',
				layout: 'auto',
				vertex: { module: pathModule, entryPoint: 'vs_main' },
				fragment: {
					module: pathModule,
					entryPoint: 'fs_main',
					targets: [
						{
							format: this.engine.sceneFormat,
							blend: {
								color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
								alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
							}
						}
					]
				},
				primitive: { topology: 'triangle-list' }
			});
			this.pathBindGroup = device.createBindGroup({
				label: 'observatory-path-bind',
				layout: this.pathPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
					{ binding: 1, resource: { buffer: this.cameraBuffer } },
					{ binding: 2, resource: { buffer: this.nodeBuffer } },
					{ binding: 3, resource: { buffer: this.pathBuffer } }
				]
			});
		}
	}

	/**
	 * FramePass — write the deterministic orbit camera, then run the
	 * recall-path simulation for this frame (compute before render).
	 */
	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.cameraBuffer) return;

		const w = this.engine.params[6] || 1;
		const h = this.engine.params[7] || 1;
		const phase = this.engine.params[1];

		const cam = orbitCamera(phase, w / h, ORBIT_DISTANCE);
		this.cameraData.set(cam.viewProj, 0);
		this.cameraData[16] = cam.right[0];
		this.cameraData[17] = cam.right[1];
		this.cameraData[18] = cam.right[2];
		this.cameraData[19] = 0;
		this.cameraData[20] = cam.up[0];
		this.cameraData[21] = cam.up[1];
		this.cameraData[22] = cam.up[2];
		this.cameraData[23] = 0;
		device.queue.writeBuffer(this.cameraBuffer, 0, this.cameraData);

		if (this.simPipeline && this.simBindGroup && this.nodeCount > 0) {
			const pass = encoder.beginComputePass({ label: 'observatory-recall-sim' });
			pass.setPipeline(this.simPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this.nodeCount / 64));
			pass.end();
		}
	}

	/** FramePass — instanced additive draws: nodes, then path ribbons on top. */
	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || this.nodeCount === 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.draw(6, this.nodeCount);

		if (this.pathPipeline && this.pathBindGroup && this.pathStepCount > 0) {
			pass.setPipeline(this.pathPipeline);
			pass.setBindGroup(0, this.pathBindGroup);
			pass.draw(6, this.pathStepCount);
		}
	}

	// ---------------------------------------------------------------------------
	// Read-only handles for BirthRenderer (Moment B, Task B2).
	// No behavior change — these expose existing buffers for other passes.
	// ---------------------------------------------------------------------------

	get nodeStateBuffer(): GPUBuffer | null {
		return this.nodeBuffer;
	}

	get cameraUniformBuffer(): GPUBuffer | null {
		return this.cameraBuffer;
	}

	get nodeCountValue(): number {
		return this.nodeCount;
	}

	get pathStepMeta(): PathStepMeta[] {
		return this.pathSteps;
	}

	/**
	 * Click picking — the ONE sanctioned GPU readback (input-driven, never
	 * per-frame, so the render loop's determinism contract holds).
	 *
	 * Copies the live NodeState buffer (post force-sim positions) to a staging
	 * buffer, reprojects every node through the SAME deterministic orbit camera
	 * the current frame used (engine params: phase + canvas size), and returns
	 * the nearest node whose projected disc contains the click.
	 *
	 * @param ndcX click x in NDC (-1..1, right = +)
	 * @param ndcY click y in NDC (-1..1, up = +)
	 */
	async pickAt(ndcX: number, ndcY: number): Promise<{ index: number; id: string } | null> {
		const device = this.engine.gpuDevice;
		if (!device || !this.nodeBuffer || !this.graph || this.nodeCount === 0) return null;

		const byteSize = this.nodeCount * FLOATS_PER_NODE * 4;
		const staging = device.createBuffer({
			label: 'observatory-pick-staging',
			size: byteSize,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const encoder = device.createCommandEncoder({ label: 'observatory-pick-copy' });
		encoder.copyBufferToBuffer(this.nodeBuffer, 0, staging, 0, byteSize);
		device.queue.submit([encoder.finish()]);
		let data: Float32Array;
		try {
			await staging.mapAsync(GPUMapMode.READ);
			data = new Float32Array(staging.getMappedRange().slice(0));
		} catch {
			staging.destroy();
			return null;
		}
		staging.unmap();
		staging.destroy();

		// Same camera inputs the frame pass uses (compute() above).
		const w = this.engine.params[6] || 1;
		const h = this.engine.params[7] || 1;
		const phase = this.engine.params[1];
		const m = orbitCamera(phase, w / h, ORBIT_DISTANCE).viewProj; // column-major

		// Projected-disc hit test: fovY 50° → f = 1/tan(25°); a node of world
		// radius r at clip-w distance projects to ~r·f/w in NDC y. A small
		// floor keeps faint distant nodes clickable; score <1.6 allows a
		// forgiving halo around the disc; lowest score (closest relative to
		// its disc) wins.
		const f = 1 / Math.tan((50 * Math.PI) / 360);
		let best = -1;
		let bestScore = Infinity;
		for (let i = 0; i < this.nodeCount; i++) {
			const b = i * FLOATS_PER_NODE + NODE_LANE.posRadius;
			const x = data[b];
			const y = data[b + 1];
			const z = data[b + 2];
			const r = data[b + 3];
			const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
			if (cw <= 0) continue; // behind the camera
			const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw;
			const cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
			const projR = Math.max((r * f) / cw, 0.012);
			const score = Math.hypot(cx - ndcX, cy - ndcY) / projR;
			if (score < 1.6 && score < bestScore) {
				bestScore = score;
				best = i;
			}
		}
		if (best < 0) return null;
		return { index: best, id: this.graph.nodes[best].id };
	}

	dispose(): void {
		this.nodeBuffer?.destroy();
		this.edgeBuffer?.destroy();
		this.cameraBuffer?.destroy();
		this.pathBuffer?.destroy();
		this.nodeBuffer = null;
		this.edgeBuffer = null;
		this.cameraBuffer = null;
		this.pathBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
		this.simPipeline = null;
		this.simBindGroup = null;
		this.pathPipeline = null;
		this.pathBindGroup = null;
	}
}

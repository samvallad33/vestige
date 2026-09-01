/**
 * Cognitive Observatory — birth particle renderer (Moment B, Tasks B3–B6).
 *
 * Registers as a FramePass on the engine ONLY when demoMode === 'engram-birth'.
 * Handles:
 *   B3: compute entry (convergence choreography)
 *   B4: particle billboard render pipeline (instanced additive)
 *   B5: birth flash + target halo (frames 330–359, loop-seam safe)
 *   B6: edge engraving via path-ribbon reuse + TimelineSpine beats
 *
 * Integration: reads nodeStateBuffer and cameraUniformBuffer from NodeRenderer,
 * creates its own particle buffer (initialized from buildBirthPlan), and
 * dispatches compute + render passes each frame.
 *
 * Loop-seam: all time terms are integer-cycles per 720 frames.
 * Capture mode: params._pad == 1.0 skips stateful particle integration.
 * No Math.random() in new files.
 */

import type { ObservatoryEngine, FramePass } from './engine';
import { NodeRenderer } from './node-renderer';
import { buildBirthPlan, type BirthPlan, type TimelineBeat } from './birth-plan';
import { birthParticlesWGSL } from './shaders/birth-particles.wgsl';
import { renderPathWGSL } from './shaders/render-path.wgsl';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PARTICLE_FLOATS = 16; // 64 bytes per particle
const QUAD_VERTS = 6; // two triangles

// Flash frames (B5)
const FLASH_START = 330;
const FLASH_END = 359;

// Edge engraving start (B6)
const ENGRAVE_START = 360;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface BirthRendererOptions {
	engine: ObservatoryEngine;
	nodeRenderer: NodeRenderer;
	seed: string;
}

// ---------------------------------------------------------------------------
// BirthRenderer
// ---------------------------------------------------------------------------

export class BirthRenderer implements FramePass {
	private engine: ObservatoryEngine;
	private nodeRenderer: NodeRenderer;
	private active: boolean;

	// Compute pipeline (B3)
	private computePipeline: GPUComputePipeline | null = null;
	private computeBindGroup: GPUBindGroup | null = null;
	private particleBuffer: GPUBuffer | null = null;
	private particleCount = 0;

	// Render pipeline (B4) — instanced additive billboards
	private renderPipeline: GPURenderPipeline | null = null;
	private renderBindGroup: GPUBindGroup | null = null;

	// Flash/halo (B5) — target glow ring
	private haloPipeline: GPURenderPipeline | null = null;
	private haloBindGroup: GPUBindGroup | null = null;
	private haloIndexBuffer: GPUBuffer | null = null;

	// Edge engraving (B6) — reuse path-ribbon shader
	private engravePipeline: GPURenderPipeline | null = null;
	private engraveBindGroup: GPUBindGroup | null = null;
	private engraveBuffer: GPUBuffer | null = null;
	private engraveStepCount = 0;

	// Timeline beats for overlay (B6)
	timeline: TimelineBeat[] = [];

	// Birth plan (CPU-side)
	private birthPlan: BirthPlan | null = null;

	/**
	 * Engrave steps in PathStep layout (source, target, beatFrame, kind) — the
	 * route feeds these into the NodeRenderer path system so the proven
	 * wavefront machinery renders the outward engraving (and the recall sim
	 * blooms each neighbor as its edge lands).
	 */
	get engraveSteps(): Uint32Array<ArrayBuffer> {
		return (this.birthPlan?.edgeSteps ?? new Uint32Array(0)) as Uint32Array<ArrayBuffer>;
	}

	constructor(opts: BirthRendererOptions) {
		this.engine = opts.engine;
		this.nodeRenderer = opts.nodeRenderer;
		// Only activate when demoMode === 'engram-birth' (demo_id === 1).
		// We check this each frame in compute() so it's safe to always register.
		this.active = false;

		this.engine.addPass(this);
	}

	/** Initialize the birth plan and GPU resources. Call after upload(). */
	upload(seed: string): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.nodeRenderer.nodeStateBuffer) return;

		const graph = this.nodeRenderer.graph;
		if (!graph) return;

		// Build the deterministic birth plan (CPU).
		this.birthPlan = buildBirthPlan(graph, seed);
		this.timeline = this.birthPlan.timeline;

		const particleCount = this.birthPlan.particles.length / PARTICLE_FLOATS;
		this.particleCount = particleCount;

		// Create particle storage buffer.
		this.particleBuffer?.destroy();
		this.particleBuffer = device.createBuffer({
			label: 'observatory-birth-particles',
			size: this.birthPlan.particles.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.particleBuffer, 0, this.birthPlan.particles.buffer as ArrayBuffer);

		// Create edge engraving buffer (B6).
		this.engraveBuffer?.destroy();
		this.engraveStepCount = this.birthPlan.edgeSteps.length / 4;
		if (this.engraveStepCount > 0) {
			this.engraveBuffer = device.createBuffer({
				label: 'observatory-birth-engrave',
				size: this.birthPlan.edgeSteps.byteLength,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			device.queue.writeBuffer(this.engraveBuffer, 0, this.birthPlan.edgeSteps.buffer as ArrayBuffer);
		}

		// Create compute pipeline (B3).
		this.createComputePipeline(device);

		// Create render pipeline (B4).
		this.createRenderPipeline(device);

		// Create flash/halo pipeline (B5).
		this.createHaloPipeline(device);

		// Create edge engraving pipeline (B6).
		this.createEngravePipeline(device);
	}

	private createComputePipeline(device: GPUDevice): void {
		const module = device.createShaderModule({
			label: 'observatory-birth-compute',
			code: birthParticlesWGSL
		});

		this.computePipeline = device.createComputePipeline({
			label: 'observatory-birth-compute-pipeline',
			layout: 'auto',
			compute: { module, entryPoint: 'birth_compute' }
		});

		// The compute shader declares exactly bindings 0-1; binding anything the
		// auto layout stripped (it has no binding 2) invalidates the bind group.
		const entries: GPUBindGroupEntry[] = [
			{ binding: 0, resource: { buffer: this.engine.paramsBuffer! } },
			{ binding: 1, resource: { buffer: this.particleBuffer! } }
		];

		this.computeBindGroup = device.createBindGroup({
			label: 'observatory-birth-compute-bind',
			layout: this.computePipeline!.getBindGroupLayout(0),
			entries
		});
	}

	private createRenderPipeline(device: GPUDevice): void {
		// Particle billboard shader (inline WGSL for render pass).
		const particleRenderWGSL = /* wgsl */ `
struct Params {
	frame: f32,
	loop_phase: f32,
	node_count: f32,
	edge_count: f32,
	path_count: f32,
	pulse: f32,
	viewport_w: f32,
	viewport_h: f32,
	brightness: f32,
	demo_id: f32,
	time: f32,
	_pad: f32,
};

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct BirthParticle {
	start_life: vec4<f32>,
	target_size: vec4<f32>,
	color_phase: vec4<f32>,
	state: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> particles: array<BirthParticle>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) color: vec3<f32>,
	@location(2) @interpolate(flat) misc: vec4<f32>,
};

const CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0,  1.0)
);

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= arrayLength(&particles)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let particle = particles[ii];
	let corner = CORNERS[vi];

	// Current position from state.xyz.
	let pos = particle.state.xyz;

	// Base size from target_size.w.
	let baseSize = particle.target_size.w;

	// Flash boost during ignition (frames 330–359).
	let frame = params.frame;
	var flashBoost = 1.0;
	if (frame >= 330.0 && frame <= 359.0) {
		let flashT = (frame - 330.0) / 29.0; // 0..1 over flash frames
		// Sharp flash: peaks at frame 345, fades by 359.
		flashBoost = 1.0 + 3.0 * (1.0 - smoothstep(330.0, 345.0, frame))
		           + 2.0 * smoothstep(345.0, 359.0, frame);
	}

	// Size: base + flash boost + pulse breathing.
	let breath = 1.0 + 0.06 * params.pulse;
	let halfSize = baseSize * 4.0 * breath * flashBoost;

	let world = pos
		+ camera.right.xyz * corner.x * halfSize
		+ camera.up.xyz * corner.y * halfSize;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;

	// Color: luciferin dust (doctrine ignition — never purple).
	let phase = particle.color_phase.w;
	let spectralW = fract(params.loop_phase + phase);
	var spectralColor: vec3<f32>;
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.91, 1.00, 0.72), // luciferin
		vec3<f32>(0.16, 0.95, 0.66), // recall jade
		vec3<f32>(0.13, 0.84, 1.00), // bridge cyan
		vec3<f32>(0.91, 1.00, 0.72)  // wrap
	);
	let f = spectralW * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	spectralColor = mix(stops[i], stops[(i + 1u) % 4u], frac);

	// Alpha from state.w (convergence progress + fade).
	let alpha = particle.state.w;

	out.color = spectralColor;
	out.misc = vec4<f32>(baseSize, 0.0, 0.0, alpha);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let alpha = in.misc.w;
	let core = smoothstep(0.25, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 2.0);

	// Additive glow: core + halo.
	let intensity = core * 1.5 + halo * 0.6;

	// Flash boost during ignition.
	let frame = params.frame;
	var flash = 0.0;
	if (frame >= 330.0 && frame <= 359.0) {
		flash = smoothstep(330.0, 345.0, frame) * 2.0;
	}

	let color = in.color * (intensity + flash);

	return vec4<f32>(color * params.brightness, 1.0);
}
`;

		const module = device.createShaderModule({
			label: 'observatory-birth-render',
			code: particleRenderWGSL
		});

		this.renderPipeline = device.createRenderPipeline({
			label: 'observatory-birth-render',
			layout: 'auto',
			vertex: { module, entryPoint: 'vs_main' },
			fragment: {
				module,
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

		const cameraBuffer = this.nodeRenderer.cameraUniformBuffer;
		this.renderBindGroup = device.createBindGroup({
			label: 'observatory-birth-render-bind',
			layout: this.renderPipeline!.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer! } },
				{ binding: 1, resource: { buffer: cameraBuffer! } },
				{ binding: 2, resource: { buffer: this.particleBuffer! } }
			]
		});
	}

	private createHaloPipeline(device: GPUDevice): void {
		// Flash halo: a glowing ring around the target node (B5).
		const haloWGSL = /* wgsl */ `
struct Params {
	frame: f32,
	loop_phase: f32,
	node_count: f32,
	edge_count: f32,
	path_count: f32,
	pulse: f32,
	viewport_w: f32,
	viewport_h: f32,
	brightness: f32,
	demo_id: f32,
	time: f32,
	_pad: f32,
};

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let node = nodes[ii];
	let flags = u32(node.color_flags.w);
	let is_target = (flags & 4u) != 0u; // flag 2: is birth target

	if (!is_target) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	// Flash halo: only visible during ignition (frames 330–359).
	let frame = params.frame;
	if (frame < 330.0 || frame > 359.0) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	// Halo ring: expands during flash, fades by frame 359.
	let flashT = (frame - 330.0) / 29.0; // 0..1
	let ringRadius = 0.3 + flashT * 0.5; // expands 0.3 → 0.8

	// Quad centered on target position.
	let pos = node.pos_radius.xyz;
	let cornerX = (f32(vi) / 3.0 - 1.0); // -1, 0, 1 (3 unique x)
	let cornerY = (f32(vi % 3) / 1.5 - 1.0); // -1, 0, 1

	// We use 4 vertices for a simple quad (vi 0..3).
	let cx = cornerX * ringRadius;
	let cy = cornerY * ringRadius;

	let world = pos
		+ camera.right.xyz * cx
		+ camera.up.xyz * cy;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);

	// UV for radial fade.
	out.uv = vec2<f32>(cx / ringRadius, cy / ringRadius);

	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 0.7) {
		discard;
	}

	// Flash: white-hot core, luciferin rim.
	let flashIntensity = 1.0 - smoothstep(0.0, 0.7, d);
	let color = vec3<f32>(0.91, 1.00, 0.72) * flashIntensity * 2.0;

	// Fade out as flash ends.
	let frame = params.frame;
	let fadeOut = 1.0 - smoothstep(345.0, 359.0, frame);

	return vec4<f32>(color * params.brightness * fadeOut, 1.0);
}
`;

		const module = device.createShaderModule({
			label: 'observatory-birth-halo',
			code: haloWGSL
		});

		this.haloPipeline = device.createRenderPipeline({
			label: 'observatory-birth-halo',
			layout: 'auto',
			vertex: { module, entryPoint: 'vs_main' },
			fragment: {
				module,
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

		const cameraBuffer = this.nodeRenderer.cameraUniformBuffer;
		this.haloBindGroup = device.createBindGroup({
			label: 'observatory-birth-halo-bind',
			layout: this.haloPipeline!.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer! } },
				{ binding: 1, resource: { buffer: cameraBuffer! } },
				{ binding: 2, resource: { buffer: this.nodeRenderer.nodeStateBuffer! } }
			]
		});
	}

	private createEngravePipeline(device: GPUDevice): void {
		// Reuse path-ribbon shader for edge engraving (B6).
		// The birth engraving uses the same triangle-strip ribbon pattern
		// but with different beat timing (starts at frame 360).

		if (this.engraveStepCount === 0 || !this.engraveBuffer) return;

		const pathModule = device.createShaderModule({
			label: 'observatory-birth-engrave',
			code: renderPathWGSL
		});

		this.engravePipeline = device.createRenderPipeline({
			label: 'observatory-birth-engrave-pipeline',
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

		this.engraveBindGroup = device.createBindGroup({
			label: 'observatory-birth-engrave-bind',
			layout: this.engravePipeline!.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer! } },
				{ binding: 1, resource: { buffer: this.nodeRenderer.cameraUniformBuffer! } },
				{ binding: 2, resource: { buffer: this.nodeRenderer.nodeStateBuffer! } },
				{ binding: 3, resource: { buffer: this.engraveBuffer! } }
			]
		});
	}

	/** FramePass.compute — run birth particle simulation. */
	compute(encoder: GPUCommandEncoder, frame: number): void {
		// Only active when demoMode === 'engram-birth' (demo_id === 1).
		const demoId = this.engine.params[9];
		this.active = demoId === 1;

		if (!this.active || !this.computePipeline || !this.computeBindGroup) return;

		const pass = encoder.beginComputePass({ label: 'observatory-birth-compute' });
		pass.setPipeline(this.computePipeline);
		pass.setBindGroup(0, this.computeBindGroup);
		pass.dispatchWorkgroups(Math.ceil(this.particleCount / 64));
		pass.end();
	}

	/** FramePass.render — draw particles, flash halo, edge engraving. */
	render(pass: GPURenderPassEncoder, frame: number): void {
		if (!this.active) return;

		// B4: Draw particle billboards (instanced additive).
		if (this.renderPipeline && this.renderBindGroup && this.particleCount > 0) {
			pass.setPipeline(this.renderPipeline);
			pass.setBindGroup(0, this.renderBindGroup);
			pass.draw(QUAD_VERTS, this.particleCount);
		}

		// B5: Draw flash halo (only during frames 330–359).
		if (this.haloPipeline && this.haloBindGroup && frame >= FLASH_START && frame <= FLASH_END) {
			pass.setPipeline(this.haloPipeline);
			pass.setBindGroup(0, this.haloBindGroup);
			// Draw one halo quad per node (most will be degenerate).
			pass.draw(4, this.nodeRenderer.nodeCountValue);
		}

		// B6: Draw edge engraving ribbons (starts at frame 360).
		if (this.engravePipeline && this.engraveBindGroup && this.engraveStepCount > 0 && frame >= ENGRAVE_START) {
			pass.setPipeline(this.engravePipeline);
			pass.setBindGroup(0, this.engraveBindGroup);
			pass.draw(6, this.engraveStepCount);
		}
	}

	dispose(): void {
		this.particleBuffer?.destroy();
		this.particleBuffer = null;
		// GPUComputePipeline.destroy() exists at runtime but older TS types omit it.
		(this.computePipeline as any)?.destroy?.();
		this.computePipeline = null;
		this.computeBindGroup = null;
		// GPURenderPipeline.destroy() exists at runtime but older TS types omit it.
		(this.renderPipeline as any)?.destroy?.();
		this.renderPipeline = null;
		this.renderBindGroup = null;
		(this.haloPipeline as any)?.destroy?.();
		this.haloPipeline = null;
		this.haloBindGroup = null;
		this.haloIndexBuffer?.destroy();
		this.haloIndexBuffer = null;
		(this.engravePipeline as any)?.destroy?.();
		this.engravePipeline = null;
		this.engraveBindGroup = null;
		this.engraveBuffer?.destroy();
		this.engraveBuffer = null;
	}
}

/**
 * Fossil Light — bounded half-resolution light transport.
 *
 * This is deliberately a self-contained FramePass.  It accepts real memory
 * emitters from its host, splats them into a compact light field, and runs a
 * small, fixed cascade of transport gathers before adding that field to the
 * Observatory HDR scene.  It does not own time, FSRS evaluation, or pointer
 * interaction: those remain facts supplied by Chrono/LiveBridge.
 *
 * Why this first version is safe to ship:
 * - `rgba8unorm` storage textures are baseline WebGPU; no optional feature is
 *   requested and a device that cannot allocate them simply skips this pass.
 * - field resolution is capped at 96k pixels, so device cost is bounded
 *   independently of the number of DOM pixels.
 * - memory light is capped to 64 real, stable source indices (MAX_EMITTERS —
 *   top-retention selection in ObservatoryStage.fossilLightSourceIndices), three compute
 *   dispatches, and one additive composite draw.
 * - suppressed sources emit no light.  True spatial shadows await an honest
 *   occluder field; this pass never pretends to infer geometry it was not sent.
 *
 * It intentionally has no ObservatoryStage import/registration.  A host can
 * construct it after `NodeRenderer.upload()`, give it a bounded list of real
 * node indices, and register it with `engine.addPass()`.
 */

import type { FramePass, ObservatoryEngine } from '$lib/observatory/engine';
import type { NodeRenderer } from '$lib/observatory/node-renderer';

// 64 bounded sources: top-salience real memories. Seed cost stays trivial
// (96k field pixels x 64 analytic falloffs ~= 6M MADs per compute tick), and
// the field stops reading as "a few lamps" and starts reading as the memory
// field itself luminescing — which is the product thesis.
const MAX_EMITTERS = 64;
const EMITTER_FLOATS = 12; // three vec4<f32>s
const CONFIG_BYTES = 32;
const CONFIG_SLOTS = 4; // seed + three transport radii (last slot also composites)
// WebGPU dynamic uniform offsets must honor minUniformBufferOffsetAlignment
// (256 bytes on every baseline adapter), even though the struct itself is 32B.
const CONFIG_STRIDE = 256;
const LIGHT_FORMAT: GPUTextureFormat = 'rgba8unorm';
const MAX_FIELD_PIXELS = 96_000;
const IDLE_COMPUTE_INTERVAL = 5; // 12 fps; geometry still draws at the engine rate.

export type FossilLightQuality = 'half-res-transport' | 'disabled';

type Resources = {
	width: number;
	height: number;
	emitterBuffer: GPUBuffer;
	sourceIndexBuffer: GPUBuffer;
	configBuffer: GPUBuffer;
	fieldA: GPUTexture;
	fieldB: GPUTexture;
	seedBindGroup: GPUBindGroup;
	propagateABindGroup: GPUBindGroup;
	propagateBBindGroup: GPUBindGroup;
	compositeBindGroup: GPUBindGroup;
	projectionBindGroup: GPUBindGroup;
	nodeBuffer: GPUBuffer;
	cameraBuffer: GPUBuffer;
};

const FOSSIL_LIGHT_WGSL = /* wgsl */ `
struct CascadeConfig {
	resolution: vec2u,
	emitter_count: u32,
	step_pixels: u32,
	exposure: f32,
	enabled: f32,
	_padding: vec2f,
};

struct Emitter {
	// xy = normalized position, z = normalized source radius, w reserved
	position_radius: vec4f,
	// rgb = semantic memory color supplied by the host, a = FSRS retention
	color_energy: vec4f,
	// x = 1 when suppressed (therefore a non-emitter), rest reserved
	flags: vec4f,
};

// These two layouts intentionally mirror NodeRenderer's live buffers. The
// projection pass runs after its simulation, so a source is located at the
// actual moving 3D node and carries the actual Chrono/FSRS value for this
// frame. No CPU projection, approximation, or GPU readback enters the loop.
struct NodeState {
	pos_radius: vec4f,
	vel_retention: vec4f,
	color_flags: vec4f,
	demo: vec4f,
};

struct Camera {
	view_proj: mat4x4f,
	right: vec4f,
	up: vec4f,
};

@group(0) @binding(0) var<uniform> cascade: CascadeConfig;
@group(0) @binding(1) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(2) var light_out: texture_storage_2d<rgba8unorm, write>;

const MAX_EMITTERS = ${MAX_EMITTERS}u;

fn fossil_tone(raw: vec3f, retention: f32) -> vec3f {
	let amber = vec3f(0.62, 0.28, 0.10);
	let jade = vec3f(0.28, 0.68, 0.48);
	let physical = mix(amber, jade, smoothstep(0.14, 0.90, retention));
	// Keep a trace of a memory's semantic hue without reviving the old
	// blue-violet dashboard palette as a light source.
	let grounded = vec3f(
		clamp(raw.r, 0.0, 1.0),
		max(clamp(raw.g, 0.0, 1.0), clamp(raw.b, 0.0, 1.0) * 0.70),
		min(clamp(raw.b, 0.0, 1.0), clamp(raw.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded, 0.16);
}

// Source projection. The host supplies a bounded, deterministic list of
// indices once after graph upload; all spatial and temporal values below come
// directly from NodeRenderer's current GPU buffers.
@group(3) @binding(0) var<uniform> project_config: CascadeConfig;
@group(3) @binding(1) var<storage, read> source_indices: array<u32>;
@group(3) @binding(2) var<storage, read> nodes: array<NodeState>;
@group(3) @binding(3) var<uniform> camera: Camera;
@group(3) @binding(4) var<storage, read_write> projected_emitters: array<Emitter>;

@compute @workgroup_size(64)
fn cs_project_sources(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (i >= project_config.emitter_count) { return; }
	let source_index = source_indices[i];
	if (source_index >= arrayLength(&nodes)) {
		// A stale index (graph regrown smaller) must be a non-emitter, not a
		// robust-access read of node 0's state.
		var dead: Emitter;
		dead.position_radius = vec4f(0.5, 0.5, 0.012, 0.0);
		dead.color_energy = vec4f(0.0);
		dead.flags = vec4f(1.0, 0.0, 0.0, 0.0);
		projected_emitters[i] = dead;
		return;
	}
	var out: Emitter;
	out.position_radius = vec4f(0.5, 0.5, 0.012, 0.0);
	out.color_energy = vec4f(0.0);
	out.flags = vec4f(1.0, 0.0, 0.0, 0.0);
	let node = nodes[source_index];
	let clip = camera.view_proj * vec4f(node.pos_radius.xyz, 1.0);
	let retention = clamp(node.vel_retention.w, 0.0, 1.0);
	let uv = clip.xy / max(clip.w, 0.0001) * vec2f(0.5, -0.5) + vec2f(0.5);
	let in_view = clip.w > 0.0001 && all(uv >= vec2f(-0.08)) && all(uv <= vec2f(1.08));
	let flags = u32(round(node.color_flags.w));
	let suppressed = (flags & 2u) != 0u;
	let projected_radius = clamp(node.pos_radius.w * 0.012 / max(abs(clip.w), 0.01), 0.008, 0.055);
	if (in_view && retention > 0.0005) {
		out.position_radius = vec4f(uv, projected_radius, 0.0);
		out.color_energy = vec4f(fossil_tone(node.color_flags.rgb, retention), retention);
		out.flags = vec4f(select(0.0, 1.0, suppressed), 0.0, 0.0, 0.0);
	}
	projected_emitters[i] = out;
}

fn inside(pixel: vec2u) -> bool {
	return pixel.x < cascade.resolution.x && pixel.y < cascade.resolution.y;
}

// Direct source splat.  This is deliberately not a screen-space bloom: every
// contribution originates in one supplied memory emitter and is retention
// weighted before it enters the transport field.
@compute @workgroup_size(8, 8)
fn cs_seed(@builtin(global_invocation_id) gid: vec3u) {
	let pixel = gid.xy;
	if (!inside(pixel)) { return; }
	let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(cascade.resolution);
	var radiance = vec3f(0.0);
	for (var i = 0u; i < MAX_EMITTERS; i = i + 1u) {
		if (i >= cascade.emitter_count) { break; }
		let source = emitters[i];
		let delta = uv - source.position_radius.xy;
		let radius = max(source.position_radius.z, 0.008);
		let distance_sq = dot(delta, delta);
		let falloff = exp(-distance_sq / (radius * radius * 1.72));
		let visible = source.color_energy.w * (1.0 - clamp(source.flags.x, 0.0, 1.0));
		radiance = radiance + source.color_energy.rgb * visible * falloff;
	}
	textureStore(light_out, vec2i(pixel), vec4f(clamp(radiance, vec3f(0.0), vec3f(1.0)), 1.0));
}

// A compact, fixed transport cascade.  The successive radii move memory light
// through 4px, 13px, and 37px neighborhoods without unbounded ray marching or
// a history buffer.  It is intentionally a graceful direct-light field, not a
// false claim of scene-aware shadowing before the engine has an occluder mask.
@group(1) @binding(0) var<uniform> transport: CascadeConfig;
@group(1) @binding(1) var light_in: texture_2d<f32>;
@group(1) @binding(2) var transported_out: texture_storage_2d<rgba8unorm, write>;

const DIRECTIONS = array<vec2i, 8>(
	vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1),
	vec2i(1, 1), vec2i(-1, 1), vec2i(1, -1), vec2i(-1, -1)
);

fn bounded_pixel(pixel: vec2i) -> vec2i {
	let hi = vec2i(transport.resolution) - vec2i(1);
	return clamp(pixel, vec2i(0), hi);
}

@compute @workgroup_size(8, 8)
fn cs_transport(@builtin(global_invocation_id) gid: vec3u) {
	let pixel_u = gid.xy;
	if (pixel_u.x >= transport.resolution.x || pixel_u.y >= transport.resolution.y) { return; }
	let pixel = vec2i(pixel_u);
	var radiance = textureLoad(light_in, pixel, 0).rgb * 0.52;
	let step = i32(max(transport.step_pixels, 1u));
	for (var i = 0u; i < 8u; i = i + 1u) {
		let neighbor = bounded_pixel(pixel + DIRECTIONS[i] * step);
		radiance = radiance + textureLoad(light_in, neighbor, 0).rgb * 0.06;
	}
	textureStore(transported_out, pixel, vec4f(clamp(radiance, vec3f(0.0), vec3f(1.0)), 1.0));
}

@group(2) @binding(0) var<uniform> composite: CascadeConfig;
@group(2) @binding(1) var light_field: texture_2d<f32>;

struct CompositeOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_composite(@builtin(vertex_index) vertex_index: u32) -> CompositeOut {
	let quad = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
		vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
	);
	let position = quad[vertex_index];
	var out: CompositeOut;
	out.clip = vec4f(position, 0.0, 1.0);
	out.uv = position * vec2f(0.5, -0.5) + vec2f(0.5);
	return out;
}

fn sample_field(uv: vec2f) -> vec3f {
	let size = vec2f(composite.resolution);
	let p = uv * size - vec2f(0.5);
	let base = vec2i(floor(p));
	let fraction = fract(p);
	let hi = vec2i(composite.resolution) - vec2i(1);
	let a = textureLoad(light_field, clamp(base, vec2i(0), hi), 0).rgb;
	let b = textureLoad(light_field, clamp(base + vec2i(1, 0), vec2i(0), hi), 0).rgb;
	let c = textureLoad(light_field, clamp(base + vec2i(0, 1), vec2i(0), hi), 0).rgb;
	let d = textureLoad(light_field, clamp(base + vec2i(1, 1), vec2i(0), hi), 0).rgb;
	return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

@fragment
fn fs_composite(in: CompositeOut) -> @location(0) vec4f {
	let radiance = sample_field(in.uv);
	let luminance = dot(radiance, vec3f(0.2126, 0.7152, 0.0722));
	// A restrained, signal-gated contribution: the light field reads as local
	// illumination instead of a full-screen purple or bloom blanket.
	let signal = smoothstep(0.012, 0.18, luminance) * composite.enabled;
	let vignette = 1.0 - 0.22 * dot(in.uv - vec2f(0.5), in.uv - vec2f(0.5));
	let color = radiance * composite.exposure * max(vignette, 0.72);
	return vec4f(color, signal * 0.54);
}
`;

function finite(value: number, fallback: number): number {
	return Number.isFinite(value) ? value : fallback;
}

/**
 * A graceful transport field for Fossil Light W1.
 *
 * Constructing it before WebGPU has booted is safe: it retains CPU emitter
 * data and allocates lazily during its first compute pass.  If allocation or
 * pipeline creation fails, it becomes a no-op rather than breaking the graph.
 */
// Named for what it IS: a bounded dilated light-transport field (seed splat +
// three fixed-radius gathers). It is NOT Radiance Cascades — no probe
// hierarchy, no interval merge, no occlusion — and the name must not claim
// otherwise (audit fleet wf_5db3f3ff, Jul 14 2026). True RC stays an upgrade
// slot for this same FramePass seat.
export class FossilLightTransportPass implements FramePass {
	private readonly engine: ObservatoryEngine;
	private readonly renderer: NodeRenderer;
	private readonly sourceIndices: Uint32Array;
	private resources: Resources | null = null;
	private projectionPipeline: GPUComputePipeline | null = null;
	private seedPipeline: GPUComputePipeline | null = null;
	private transportPipeline: GPUComputePipeline | null = null;
	private compositePipeline: GPURenderPipeline | null = null;
	private projectionLayout: GPUBindGroupLayout | null = null;
	private seedLayout: GPUBindGroupLayout | null = null;
	private transportLayout: GPUBindGroupLayout | null = null;
	private compositeLayout: GPUBindGroupLayout | null = null;
	private readonly emitterCount: number;
	private active = false;
	private dirty = true;
	private lastComputedFrame = -IDLE_COMPUTE_INTERVAL;
	private disposed = false;
	private disabledReason: string | null = null;
	private exposure = 0.42;
	// Every cascade step gets an independent uniform range. Multiple writes to
	// one range before queue.submit would otherwise make all dispatches observe
	// the final radius instead of 4px -> 13px -> 37px.
	private readonly configBytes = new ArrayBuffer(CONFIG_BYTES);
	private readonly configUints = new Uint32Array(this.configBytes);
	private readonly configFloats = new Float32Array(this.configBytes);

	constructor(engine: ObservatoryEngine, renderer: NodeRenderer, sourceIndices: Uint32Array) {
		this.engine = engine;
		this.renderer = renderer;
		// The CPU chooses only WHICH real memories may source the bounded field.
		// Position, retention, color, birth mask, and suppression are projected
		// from the live GPU node state every active frame.
		const unique = [...new Set([...sourceIndices].filter((i) => Number.isFinite(i) && i >= 0))]
			.sort((a, b) => a - b)
			.slice(0, MAX_EMITTERS);
		this.sourceIndices = new Uint32Array(unique);
		this.emitterCount = this.sourceIndices.length;
	}

	/** The caller can surface this in diagnostics without probing GPU internals. */
	get quality(): FossilLightQuality {
		return this.disabledReason === null ? 'half-res-transport' : 'disabled';
	}

	/** Undefined while waiting for the first GPU frame; set only on safe fallback. */
	get fallbackReason(): string | null {
		return this.disabledReason;
	}

	/**
	 * Host-owned Chrono input. Retention is already in NodeRenderer's live FSRS
	 * buffer; this only turns the bounded light transport up to 60fps while the
	 * user is actively steering the clock.
	 */
	setScrubbing(active: boolean): void {
		this.active = active;
		this.dirty = true;
		this.engine.requestRender();
	}

	setExposure(exposure: number): void {
		this.exposure = Math.max(0, Math.min(0.72, finite(exposure, 0.42)));
		this.dirty = true;
		this.engine.requestRender();
	}

	targetFrameRate(): number {
		// During a Chrono drag retention changes continuously; at rest, the field
		// remains a calm, low-cost illumination layer.
		return this.active ? 60 : 10;
	}

	compute(encoder: GPUCommandEncoder, frame = 0): void {
		if (this.disposed || this.disabledReason !== null || this.emitterCount === 0) return;
		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		const sources = this.renderer.getFossilLightSources();
		if (!sources) return;
		const dimensions = this.fieldDimensions();
		if (dimensions === null) return;
		// `frame` is the WRAPPED 720-frame loop index: after the loop seam the
		// raw difference goes negative and a naive >= gate starves forever once
		// lastComputedFrame lands in the final interval (frames 715-719 — hit
		// deterministically within the FIRST 12s loop). A negative elapsed IS a
		// wrap, and a wrap is always at least one interval of real time.
		const elapsed = frame - this.lastComputedFrame;
		const needsFrame = this.active || this.dirty || elapsed < 0 || elapsed >= IDLE_COMPUTE_INTERVAL;
		if (!needsFrame) return;

		try {
			this.ensurePipelines(device);
			this.ensureResources(device, dimensions.width, dimensions.height, sources);
		} catch {
			// The graph remains fully functional with direct node emission if an
			// unusual adapter rejects this optional transport field.
			this.disable('GPU light field unavailable on this adapter');
			return;
		}
		if (!this.resources || !this.projectionPipeline || !this.seedPipeline || !this.transportPipeline) return;

		this.writeConfig(device, 0, this.resources.width, this.resources.height, 0);
		const workgroupsX = Math.ceil(this.resources.width / 8);
		const workgroupsY = Math.ceil(this.resources.height / 8);
		const compute = encoder.beginComputePass({ label: 'fossil-light-half-res-transport' });
		// Projection is deliberately after NodeRenderer's force/FSRS simulation
		// in engine pass order. It sees the actual camera and node state with no
		// readback or competing time model.
		compute.setPipeline(this.projectionPipeline);
		compute.setBindGroup(3, this.resources.projectionBindGroup, [0]);
		compute.dispatchWorkgroups(Math.ceil(this.emitterCount / 64));
		compute.setPipeline(this.seedPipeline);
		compute.setBindGroup(0, this.resources.seedBindGroup, [0]);
		compute.dispatchWorkgroups(workgroupsX, workgroupsY);

		// Three radii form a fixed cascade: local body -> neighborhood -> field.
		for (const [slot, step, bindGroup] of [
			[1, 4, this.resources.propagateABindGroup],
			[2, 13, this.resources.propagateBBindGroup],
			[3, 37, this.resources.propagateABindGroup]
		] as const) {
			this.writeConfig(device, slot, this.resources.width, this.resources.height, step);
			compute.setPipeline(this.transportPipeline);
			compute.setBindGroup(1, bindGroup, [slot * CONFIG_STRIDE]);
			compute.dispatchWorkgroups(workgroupsX, workgroupsY);
		}
		compute.end();
		this.dirty = false;
		this.lastComputedFrame = frame;
	}

	render(pass: GPURenderPassEncoder): void {
		if (this.disabledReason !== null || !this.resources || !this.compositePipeline || this.emitterCount === 0) return;
		pass.setPipeline(this.compositePipeline);
		pass.setBindGroup(2, this.resources.compositeBindGroup, [3 * CONFIG_STRIDE]);
		pass.draw(6);
	}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.destroyResources();
		this.projectionPipeline = null;
		this.seedPipeline = null;
		this.transportPipeline = null;
		this.compositePipeline = null;
		this.seedLayout = null;
		this.projectionLayout = null;
		this.transportLayout = null;
		this.compositeLayout = null;
	}

	private fieldDimensions(): { width: number; height: number } | null {
		const viewportWidth = Math.floor(this.engine.params[6]);
		const viewportHeight = Math.floor(this.engine.params[7]);
		if (viewportWidth < 2 || viewportHeight < 2) return null;
		const halfPixels = viewportWidth * 0.5 * (viewportHeight * 0.5);
		const pixelScale = Math.min(1, Math.sqrt(MAX_FIELD_PIXELS / Math.max(1, halfPixels)));
		const halfScale = 0.5 * pixelScale;
		return {
			width: Math.max(1, Math.floor(viewportWidth * halfScale)),
			height: Math.max(1, Math.floor(viewportHeight * halfScale))
		};
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.projectionPipeline && this.seedPipeline && this.transportPipeline && this.compositePipeline) return;
		const module = device.createShaderModule({ label: 'fossil-light-radiance-cascade-wgsl', code: FOSSIL_LIGHT_WGSL });
		// One WGSL module holds three independently compiled entry-point families.
		// Empty preceding groups preserve its explicit @group(1) and @group(2)
		// declarations without binding unused resources for those pipelines.
		const emptyLayout = device.createBindGroupLayout({ label: 'fossil-light-empty-layout', entries: [] });
		this.projectionLayout = device.createBindGroupLayout({
			label: 'fossil-light-source-projection-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CONFIG_BYTES } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
				{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		this.seedLayout = device.createBindGroupLayout({
			label: 'fossil-light-seed-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CONFIG_BYTES } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: LIGHT_FORMAT } }
			]
		});
		this.transportLayout = device.createBindGroupLayout({
			label: 'fossil-light-transport-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CONFIG_BYTES } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: LIGHT_FORMAT } }
			]
		});
		this.compositeLayout = device.createBindGroupLayout({
			label: 'fossil-light-composite-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: CONFIG_BYTES } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } }
			]
		});
		this.seedPipeline = device.createComputePipeline({
			label: 'fossil-light-seed',
			layout: device.createPipelineLayout({ label: 'fossil-light-seed-pipeline-layout', bindGroupLayouts: [this.seedLayout] }),
			compute: { module, entryPoint: 'cs_seed' }
		});
		this.projectionPipeline = device.createComputePipeline({
			label: 'fossil-light-source-projection',
			layout: device.createPipelineLayout({
				label: 'fossil-light-source-projection-pipeline-layout',
				bindGroupLayouts: [emptyLayout, emptyLayout, emptyLayout, this.projectionLayout]
			}),
			compute: { module, entryPoint: 'cs_project_sources' }
		});
		this.transportPipeline = device.createComputePipeline({
			label: 'fossil-light-transport',
			layout: device.createPipelineLayout({ label: 'fossil-light-transport-pipeline-layout', bindGroupLayouts: [emptyLayout, this.transportLayout] }),
			compute: { module, entryPoint: 'cs_transport' }
		});
		this.compositePipeline = device.createRenderPipeline({
			label: 'fossil-light-composite',
			layout: device.createPipelineLayout({ label: 'fossil-light-composite-pipeline-layout', bindGroupLayouts: [emptyLayout, emptyLayout, this.compositeLayout] }),
			vertex: { module, entryPoint: 'vs_composite' },
			fragment: {
				module,
				entryPoint: 'fs_composite',
				targets: [
					{
						format: this.engine.sceneFormat,
						blend: {
							color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
							alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
						}
					}
				]
			}
		});
	}

	private ensureResources(
		device: GPUDevice,
		width: number,
		height: number,
		sources: { nodeBuffer: GPUBuffer; cameraBuffer: GPUBuffer; nodeCount: number }
	): void {
		if (
			this.resources?.width === width &&
			this.resources.height === height &&
			this.resources.nodeBuffer === sources.nodeBuffer &&
			this.resources.cameraBuffer === sources.cameraBuffer
		)
			return;
		this.destroyResources();
		if (!this.projectionLayout || !this.seedLayout || !this.transportLayout || !this.compositeLayout) return;
		const emitterBuffer = device.createBuffer({
			label: 'fossil-light-projected-memory-emitters',
			size: MAX_EMITTERS * EMITTER_FLOATS * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE
		});
		const sourceIndexBuffer = device.createBuffer({
			label: 'fossil-light-source-indices',
			size: Math.max(4, this.sourceIndices.byteLength),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(
			sourceIndexBuffer,
			0,
			this.sourceIndices.buffer as ArrayBuffer,
			this.sourceIndices.byteOffset,
			this.sourceIndices.byteLength
		);
		const configBuffer = device.createBuffer({
			label: 'fossil-light-cascade-config',
			size: CONFIG_STRIDE * CONFIG_SLOTS,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const createField = (label: string) => device.createTexture({
			label,
			size: [width, height],
			format: LIGHT_FORMAT,
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
		});
		const fieldA = createField('fossil-light-field-a');
		const fieldB = createField('fossil-light-field-b');
		const aView = fieldA.createView();
		const bView = fieldB.createView();
		this.resources = {
			width,
			height,
			emitterBuffer,
			sourceIndexBuffer,
			configBuffer,
			fieldA,
			fieldB,
			seedBindGroup: device.createBindGroup({
				label: 'fossil-light-seed-bind-group',
				layout: this.seedLayout,
				entries: [
					{ binding: 0, resource: { buffer: configBuffer, size: CONFIG_BYTES } },
					{ binding: 1, resource: { buffer: emitterBuffer } },
					{ binding: 2, resource: aView }
				]
			}),
			propagateABindGroup: device.createBindGroup({
				label: 'fossil-light-transport-a-to-b',
				layout: this.transportLayout,
				entries: [
					{ binding: 0, resource: { buffer: configBuffer, size: CONFIG_BYTES } },
					{ binding: 1, resource: aView },
					{ binding: 2, resource: bView }
				]
			}),
			propagateBBindGroup: device.createBindGroup({
				label: 'fossil-light-transport-b-to-a',
				layout: this.transportLayout,
				entries: [
					{ binding: 0, resource: { buffer: configBuffer, size: CONFIG_BYTES } },
					{ binding: 1, resource: bView },
					{ binding: 2, resource: aView }
				]
			}),
			projectionBindGroup: device.createBindGroup({
				label: 'fossil-light-source-projection-bind-group',
				layout: this.projectionLayout,
				entries: [
					{ binding: 0, resource: { buffer: configBuffer, size: CONFIG_BYTES } },
					{ binding: 1, resource: { buffer: sourceIndexBuffer } },
					{ binding: 2, resource: { buffer: sources.nodeBuffer } },
					{ binding: 3, resource: { buffer: sources.cameraBuffer } },
					{ binding: 4, resource: { buffer: emitterBuffer } }
				]
			}),
			compositeBindGroup: device.createBindGroup({
				label: 'fossil-light-composite-bind-group',
				layout: this.compositeLayout,
				entries: [
					{ binding: 0, resource: { buffer: configBuffer, size: CONFIG_BYTES } },
					// The third transport is A -> B, so B is always the final field.
					{ binding: 1, resource: bView }
				]
			}),
			nodeBuffer: sources.nodeBuffer,
			cameraBuffer: sources.cameraBuffer
		};
		this.dirty = true;
	}

	private writeConfig(device: GPUDevice, slot: number, width: number, height: number, stepPixels: number): void {
		if (!this.resources) return;
		this.configUints[0] = width;
		this.configUints[1] = height;
		this.configUints[2] = this.emitterCount;
		this.configUints[3] = stepPixels;
		this.configFloats[4] = this.exposure;
		this.configFloats[5] = 1;
		device.queue.writeBuffer(this.resources.configBuffer, slot * CONFIG_STRIDE, this.configBytes);
	}

	private destroyResources(): void {
		this.resources?.emitterBuffer.destroy();
		this.resources?.sourceIndexBuffer.destroy();
		this.resources?.configBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}

	private disable(reason: string): void {
		this.destroyResources();
		this.disabledReason = reason;
		this.engine.requestRender();
	}
}

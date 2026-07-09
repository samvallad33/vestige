import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { ReasoningScene, ReasoningStageReceipt } from './reasoning-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const STAGE_COUNT = 8;
const STAGE_FLOATS = 8;
const PACKET_FLOATS = 8;
const PACKET_OUT_FLOATS = 8;
const MAX_PACKETS = 256;
const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';

const COMMON_WGSL = /* wgsl */ `
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
	capture_mode: f32,
	live_kind: f32,
	live_frame: f32,
	live_energy: f32,
	projection_days: f32,
};

struct Stage {
	// xy center in NDC, z count, w confidence
	pos_count_conf: vec4f,
	// x start_frame, y interrupt_kind, z lit, w reserved
	timing: vec4f,
};

struct PacketIn {
	// x src stage, y dst stage, z evidence/node index, w flags
	route: vec4f,
	// x start_frame, y duration, z energy, w interrupt_kind
	timing: vec4f,
};

struct PacketOut {
	pos_energy: vec4f,
	tangent_flags: vec4f,
};
`;

const COMPUTE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stages: array<Stage>;
@group(0) @binding(2) var<storage, read> packets_in: array<PacketIn>;
@group(0) @binding(3) var<storage, read_write> packets_out: array<PacketOut>;

fn stage_center(i: u32) -> vec2f {
	return stages[i].pos_count_conf.xy;
}

fn cubic(a: vec2f, b: vec2f, t_in: f32) -> vec2f {
	let t = clamp(t_in, 0.0, 1.0);
	let u = 1.0 - t;
	let bend = 0.18 * sin(f32(i32(a.y * 10.0) + i32(b.y * 10.0)));
	let p0 = a;
	let p1 = vec2f(a.x + bend, mix(a.y, b.y, 0.32));
	let p2 = vec2f(b.x - bend, mix(a.y, b.y, 0.68));
	let p3 = b;
	return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

fn cubic_tangent(a: vec2f, b: vec2f, t_in: f32) -> vec2f {
	let t = clamp(t_in, 0.0, 1.0);
	let e = 0.01;
	return normalize(cubic(a, b, min(1.0, t + e)) - cubic(a, b, max(0.0, t - e)) + vec2f(0.0001, 0.0001));
}

@compute @workgroup_size(64)
fn advect_packets(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (i >= u32(params.path_count)) { return; }
	let p = packets_in[i];
	let src = u32(p.route.x);
	let dst = u32(p.route.y);
	if (src >= 8u || dst >= 8u) { return; }
	let start = p.timing.x;
	let dur = max(1.0, p.timing.y);
	var t = clamp((params.frame - start) / dur, 0.0, 1.0);
	let tt = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
	let a = stage_center(src);
	let b = stage_center(dst);
	let pos = cubic(a, b, tt);
	let tan = cubic_tangent(a, b, tt);
	let alive_f = f32(params.frame >= start) * f32(params.frame <= start + dur + 90.0);
	let age = max(0.0, params.frame - start - dur);
	let tail = 1.0 - smoothstep(0.0, 90.0, age);
	packets_out[i] = PacketOut(vec4f(pos, p.timing.z * alive_f * max(0.18, tail), 1.0), vec4f(tan, p.timing.w, p.route.z));
}

`;

const SPLAT_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stages: array<Stage>;
@group(0) @binding(3) var<storage, read> packets_out: array<PacketOut>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
};

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let corner = QUAD[vi];
	let is_stage = ii < 8u;
	var center = vec2f(0.0);
	var radius = 0.045;
	var energy = 0.0;
	var interrupt = 0.0;
	if (is_stage) {
		let s = stages[ii];
		center = s.pos_count_conf.xy;
		let age = params.frame - s.timing.x;
		let gate = s.timing.z * smoothstep(0.0, 22.0, age);
		radius = 0.095 + 0.02 * s.pos_count_conf.w;
		energy = gate * (0.18 + 0.82 * s.pos_count_conf.w) * (0.7 + 0.3 * params.pulse);
		interrupt = s.timing.y;
	} else {
		let pi = ii - 8u;
		let p = packets_out[pi];
		center = p.pos_energy.xy;
		radius = 0.026 + 0.014 * p.pos_energy.z;
		energy = p.pos_energy.z;
		interrupt = p.tangent_flags.z;
	}
	out.clip = vec4f(center + corner * radius, 0.0, 1.0);
	out.uv = corner;
	out.misc = vec4f(energy, interrupt, f32(is_stage), radius);
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let body = exp(-d * d * 3.2) * in.misc.x;
	let scar = f32(in.misc.y > 0.5) * smoothstep(0.75, 0.2, abs(d - 0.62)) * in.misc.x;
	// logical .r = living density, .g = trust/flow, .b = immune interrupt, .a reserved
	return vec4f(body, body * (0.45 + 0.35 * params.pulse), scar, 1.0);
}

@vertex
fn vs_chamber(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let s = stages[ii];
	let corner = QUAD[vi];
	let age = params.frame - s.timing.x;
	let gate = s.timing.z * smoothstep(0.0, 24.0, age);
	let size = vec2f(0.28 + 0.025 * s.pos_count_conf.w, 0.043 + 0.018 * gate);
	out.clip = vec4f(s.pos_count_conf.xy + corner * size, 0.0, 1.0);
	out.uv = corner;
	out.misc = vec4f(gate, s.pos_count_conf.w, s.timing.y, s.pos_count_conf.z);
	return out;
}

@fragment
fn fs_chamber(in: VSOut) -> @location(0) vec4f {
	let q = abs(in.uv);
	let edge = smoothstep(0.98, 0.80, max(q.x, q.y));
	let inner = smoothstep(0.92, 0.15, max(q.x, q.y));
	let lit = in.misc.x;
	let trust = in.misc.y;
	let interrupt = in.misc.z;
	let count = in.misc.w;
	let green = vec3f(0.64, 1.0, 0.36);
	let cyan = vec3f(0.10, 0.84, 0.98);
	let scarlet = vec3f(1.0, 0.20, 0.16);
	let amber = vec3f(1.0, 0.69, 0.08);
	var color = mix(cyan, green, trust) * (0.05 + lit * 0.34) * inner;
	let rim_color = select(mix(cyan, green, trust), select(amber, scarlet, interrupt < 1.5), interrupt > 0.5);
	color = color + rim_color * edge * (0.05 + lit * (0.32 + 0.08 * log2(count + 1.0)));
	return vec4f(color, 1.0);
}

@vertex
fn vs_packet(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let p = packets_out[ii];
	let corner = QUAD[vi];
	let tangent = normalize(p.tangent_flags.xy + vec2f(0.0001));
	let normal = vec2f(-tangent.y, tangent.x);
	let size = vec2f(0.038 + 0.018 * p.pos_energy.z, 0.010 + 0.006 * p.pos_energy.z);
	let pos = p.pos_energy.xy + tangent * corner.x * size.x + normal * corner.y * size.y;
	out.clip = vec4f(pos, 0.0, 1.0);
	out.uv = corner;
	out.misc = vec4f(p.pos_energy.z, p.tangent_flags.z, p.tangent_flags.w, 0.0);
	return out;
}

@fragment
fn fs_packet(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let e = in.misc.x;
	let interrupt = in.misc.y;
	let body = exp(-d*d*2.2) * e;
	let luciferin = vec3f(0.91, 1.0, 0.72);
	let recall = vec3f(0.16, 0.95, 0.66);
	let scarlet = vec3f(1.0, 0.23, 0.19);
	let amber = vec3f(1.0, 0.69, 0.08);
	var color = mix(recall, luciferin, clamp(e, 0.0, 1.0));
	if (interrupt > 0.5) { color = select(amber, scarlet, interrupt < 1.5); }
	return vec4f(color * body * (1.2 + 0.4 * params.pulse), 1.0);
}
`;

const MEMBRANE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(4) var field_sampler: sampler;
@group(0) @binding(5) var field_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	out.misc = vec4f(0.0);
	return out;
}

@fragment
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, in.uv);
	let density = clamp(f.r, 0.0, 4.0);
	let flow = clamp(f.g, 0.0, 4.0);
	let immune = clamp(f.b, 0.0, 3.0);
	let membrane = smoothstep(0.14, 0.8, density) * (1.0 - smoothstep(1.6, 3.2, density));
	let blackwater = vec3f(0.006, 0.014, 0.014);
	let luciferin = vec3f(0.65, 1.0, 0.36);
	let bridge = vec3f(0.10, 0.82, 0.92);
	let scarlet = vec3f(1.0, 0.15, 0.10);
	var color = blackwater * (0.16 + density * 0.08);
	color = color + luciferin * density * 0.11 + bridge * flow * 0.08;
	color = color + vec3f(0.90, 1.0, 0.72) * membrane * 0.26;
	color = color + scarlet * immune * 0.26;
	let vignette = smoothstep(0.92, 0.22, distance(in.uv, vec2f(0.5)));
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness, 1.0);
}
`;

const BLUR_WGSL = /* wgsl */ `
struct BlurDir {
	dir: vec2f,
	_pad: vec2f,
};

@group(0) @binding(0) var blur_sampler: sampler;
@group(0) @binding(1) var blur_src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur_dir: BlurDir;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let step = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, in.uv - step * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv - step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + step * 2.0, 0.0) * 0.06136;
	return acc;
}
`;

type GpuResources = {
	stageBuffer: GPUBuffer;
	packetBuffer: GPUBuffer;
	packetOutBuffer: GPUBuffer;
	blurHBuffer: GPUBuffer;
	blurVBuffer: GPUBuffer;
	computeBindGroup: GPUBindGroup;
	splatBindGroup: GPUBindGroup;
	blurHBindGroup: GPUBindGroup;
	blurVBindGroup: GPUBindGroup;
	membraneBindGroup: GPUBindGroup;
	fieldA: GPUTexture;
	fieldB: GPUTexture;
	fieldAView: GPUTextureView;
	fieldBView: GPUTextureView;
	fieldSize: [number, number];
};

export class ReasoningTheaterPass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: ReasoningScene | null = null;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private computeBindLayout: GPUBindGroupLayout | null = null;
	private splatBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private splatPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private packetPipeline: GPUComputePipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private chamberPipeline: GPURenderPipeline | null = null;
	private packetRenderPipeline: GPURenderPipeline | null = null;
	private packetCount = 0;
	private stageRects: { stage: ReasoningStageReceipt; x: number; y: number; w: number; h: number }[] = [];

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as ReasoningScene;
		this.buildStageRects();
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.splatPipeline || !this.engine.paramsBuffer) return;
		const computeModule = device.createShaderModule({ label: 'reasoning-theater-compute-wgsl', code: COMPUTE_WGSL });
		const splatModule = device.createShaderModule({ label: 'reasoning-theater-splat-wgsl', code: SPLAT_WGSL });
		const blurModule = device.createShaderModule({ label: 'reasoning-theater-blur-wgsl', code: BLUR_WGSL });
		const membraneModule = device.createShaderModule({ label: 'reasoning-theater-membrane-wgsl', code: MEMBRANE_WGSL });
		this.computeBindLayout = device.createBindGroupLayout({
			label: 'reasoning-theater-compute-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		this.splatBindLayout = device.createBindGroupLayout({
			label: 'reasoning-theater-splat-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'reasoning-theater-membrane-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'reasoning-theater-blur-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const computeLayout = device.createPipelineLayout({ label: 'reasoning-theater-compute-layout', bindGroupLayouts: [this.computeBindLayout] });
		const splatLayout = device.createPipelineLayout({ label: 'reasoning-theater-splat-layout', bindGroupLayouts: [this.splatBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'reasoning-theater-blur-layout', bindGroupLayouts: [this.blurBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'reasoning-theater-membrane-layout', bindGroupLayouts: [this.membraneBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
		this.splatPipeline = device.createRenderPipeline({
			label: 'reasoning-field-splat',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_splat' },
			fragment: {
				module: splatModule,
				entryPoint: 'fs_splat',
				targets: [{ format: FIELD_FORMAT, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }]
			},
			primitive: { topology: 'triangle-list' }
		});
		this.blurPipeline = device.createRenderPipeline({
			label: 'reasoning-field-blur',
			layout: blurLayout,
			vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] },
			primitive: { topology: 'triangle-list' }
		});
		this.packetPipeline = device.createComputePipeline({ label: 'reasoning-packet-advect', layout: computeLayout, compute: { module: computeModule, entryPoint: 'advect_packets' } });
		const additive = { color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }, alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation } };
		this.membranePipeline = device.createRenderPipeline({
			label: 'reasoning-membrane',
			layout: membraneLayout,
			vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.chamberPipeline = device.createRenderPipeline({
			label: 'reasoning-chambers',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_chamber' },
			fragment: { module: splatModule, entryPoint: 'fs_chamber', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.packetRenderPipeline = device.createRenderPipeline({
			label: 'reasoning-packets',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_packet' },
			fragment: { module: splatModule, entryPoint: 'fs_packet', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.computeBindLayout || !this.splatBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let stageBuffer = this.resources?.stageBuffer;
		let packetBuffer = this.resources?.packetBuffer;
		let packetOutBuffer = this.resources?.packetOutBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		if (!stageBuffer) {
			stageBuffer = device.createBuffer({ label: 'reasoning-stages', size: STAGE_COUNT * STAGE_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		}
		if (!packetBuffer) {
			packetBuffer = device.createBuffer({ label: 'reasoning-packets-in', size: MAX_PACKETS * PACKET_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		}
		if (!packetOutBuffer) {
			packetOutBuffer = device.createBuffer({ label: 'reasoning-packets-out', size: MAX_PACKETS * PACKET_OUT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		}
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'reasoning-blur-h-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'reasoning-blur-v-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!needsTextures && this.resources) return;
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'reasoning-field-a-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'reasoning-field-b-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const computeBindGroup = device.createBindGroup({
			label: 'reasoning-theater-compute-bind',
			layout: this.computeBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: stageBuffer } },
				{ binding: 2, resource: { buffer: packetBuffer } },
				{ binding: 3, resource: { buffer: packetOutBuffer } }
			]
		});
		const splatBindGroup = device.createBindGroup({
			label: 'reasoning-theater-splat-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: stageBuffer } },
				{ binding: 3, resource: { buffer: packetOutBuffer } }
			]
		});
		const blurHBindGroup = device.createBindGroup({
			label: 'reasoning-theater-blur-h-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldAView },
				{ binding: 2, resource: { buffer: blurHBuffer } }
			]
		});
		const blurVBindGroup = device.createBindGroup({
			label: 'reasoning-theater-blur-v-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldBView },
				{ binding: 2, resource: { buffer: blurVBuffer } }
			]
		});
		const membraneBindGroup = device.createBindGroup({
			label: 'reasoning-theater-membrane-bind',
			layout: this.membraneBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 4, resource: this.sampler },
				{ binding: 5, resource: fieldAView }
			]
		});
		this.resources = { stageBuffer, packetBuffer, packetOutBuffer, blurHBuffer, blurVBuffer, computeBindGroup, splatBindGroup, blurHBindGroup, blurVBindGroup, membraneBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
	}

	private buildStageRects(): void {
		const stages = this.scene?.stages ?? [];
		this.stageRects = stages.map((stage, i) => {
			const y = 0.76 - i * (1.52 / 7);
			const confidence = stage.confidence || 0;
			return { stage, x: 0, y, w: 0.28 + 0.025 * confidence, h: 0.07 };
		});
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources || !this.scene) return;
		const stages = this.scene.stages ?? [];
		const stageData = new Float32Array(STAGE_COUNT * STAGE_FLOATS);
		for (let i = 0; i < STAGE_COUNT; i++) {
			const s = stages[i];
			const y = 0.76 - i * (1.52 / 7);
			const lit = s?.lit ? 1 : 0;
			const interrupt = s?.interrupt === 'contradiction' ? 1 : s?.interrupt === 'supersession' ? 2 : 0;
			stageData.set([0, y, s?.count ?? 0, s?.confidence ?? 0, i * 46 + 18, interrupt, lit, 0], i * STAGE_FLOATS);
		}
		device.queue.writeBuffer(this.resources.stageBuffer, 0, stageData);

		const packets = new Float32Array(MAX_PACKETS * PACKET_FLOATS);
		let p = 0;
		for (let i = 0; i < STAGE_COUNT - 1 && p < MAX_PACKETS; i++) {
			const a = stages[i];
			const b = stages[i + 1];
			if (!a?.lit || !b?.lit) continue;
			const energy = Math.max(0.12, Math.min(1, (a.confidence + b.confidence) / 2 || 0.25));
			const interrupt = b.interrupt === 'contradiction' ? 1 : b.interrupt === 'supersession' ? 2 : 0;
			packets.set([i, i + 1, -1, 0, i * 46 + 34, 38, energy, interrupt], p * PACKET_FLOATS);
			p++;
		}
		for (const c of this.scene.contradictions ?? []) {
			if (p >= MAX_PACKETS) break;
			packets.set([3, 4, -1, 0, 4 * 46 + 18, 24, Math.max(0.35, c.topic_overlap || 0.5), 1], p * PACKET_FLOATS);
			p++;
		}
		for (const _s of this.scene.superseded ?? []) {
			if (p >= MAX_PACKETS) break;
			packets.set([4, 7, -1, 0, 5 * 46, 52, 0.65, 2], p * PACKET_FLOATS);
			p++;
		}
		this.packetCount = p;
		this.engine.params[4] = p;
		device.queue.writeBuffer(this.resources.packetBuffer, 0, packets);
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.splatPipeline || !this.blurPipeline || !this.packetPipeline) return;
		this.ensureResources(device);
		const res = this.resources;
		const splat = encoder.beginRenderPass({
			label: 'reasoning-field-splat-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		splat.setPipeline(this.splatPipeline);
		splat.setBindGroup(0, res.splatBindGroup);
		splat.draw(6, STAGE_COUNT + this.packetCount);
		splat.end();

		const blurH = encoder.beginRenderPass({
			label: 'reasoning-field-blur-h-pass',
			colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();

		const blurV = encoder.beginRenderPass({
			label: 'reasoning-field-blur-v-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();

		if (this.packetCount > 0) {
			const advect = encoder.beginComputePass({ label: 'reasoning-packet-advect-pass' });
			advect.setPipeline(this.packetPipeline);
			advect.setBindGroup(0, res.computeBindGroup);
			advect.dispatchWorkgroups(Math.ceil(this.packetCount / 64));
			advect.end();
		}
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.chamberPipeline || !this.packetRenderPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		pass.setPipeline(this.chamberPipeline);
		pass.setBindGroup(0, this.resources.splatBindGroup);
		pass.draw(6, STAGE_COUNT);
		if (this.packetCount > 0) {
			pass.setPipeline(this.packetRenderPipeline);
			pass.draw(6, this.packetCount);
		}
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		for (const rect of this.stageRects) {
			if (Math.abs(ndcX - rect.x) <= rect.w && Math.abs(ndcY - rect.y) <= rect.h) {
				return { id: `reasoning-stage:${rect.stage.kind}`, kind: 'stage', index: rect.stage.index, payload: rect.stage };
			}
		}
		return null;
	}

	dispose(): void {
		this.resources?.stageBuffer.destroy();
		this.resources?.packetBuffer.destroy();
		this.resources?.packetOutBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}
}

export function createReasoningTheaterPasses(engine: ObservatoryEngine, scene: RouteSceneModel): ReasoningTheaterPass[] {
	void rgb01(RETENTION.healthy);
	void rgb01(IMMUNE.veto);
	void rgb01(CAUSAL.forward);
	return [new ReasoningTheaterPass(engine, scene)];
}

import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { IMMUNE, MEDIUM, RETENTION, membraneWidth, rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { ContradictionsScene, ImmuneSynapsePair } from './contradictions-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_PAIRS = 160;
const PAIR_FLOATS = 16;

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

struct PairCell {
	// stronger.xy, stronger trust, stronger membrane width
	stronger: vec4f,
	// weaker.xy, weaker trust, weaker membrane width
	weaker: vec4f,
	// x topic overlap, y trust delta, z unresolved, w slot phase
	signals: vec4f,
	// x pair index, y unused, z unused, w unused
	ids: vec4f,
};
`;

const SPLAT_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pairs: array<PairCell>;

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
	let pair_i = ii / 2u;
	let side = ii % 2u;
	let p = pairs[pair_i];
	let stronger_side = side == 0u;
	let cell = select(p.weaker, p.stronger, stronger_side);
	let trust = cell.z;
	let width = cell.w;
	let overlap = p.signals.x;
	let unresolved = p.signals.z;
	let breathing = 1.0 + 0.05 * sin(params.time * 2.2 + p.signals.w * 6.28318);
	let radius = (0.105 + 0.05 * overlap + width * 1.8) * breathing;
	out.clip = vec4f(cell.xy + QUAD[vi] * radius, 0.0, 1.0);
	out.uv = QUAD[vi];
	out.misc = vec4f(trust, overlap, f32(side), unresolved);
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let trust = clamp(in.misc.x, 0.0, 1.0);
	let overlap = clamp(in.misc.y, 0.0, 1.0);
	let side = in.misc.z;
	let unresolved = in.misc.w;
	let body = exp(-d * d * 3.4) * (0.42 + trust * 0.72) * (0.45 + overlap * 0.7);
	let membrane = smoothstep(0.22, 0.02, abs(d - (0.62 + trust * 0.16))) * (0.18 + trust * 0.75);
	let spark = unresolved * smoothstep(0.96, 0.68, d) * (0.6 + 0.4 * sin(params.time * 16.0 + in.uv.x * 7.0));
	// Dual-channel signed field: stronger splats red, weaker splats green.
	let r = select(0.0, body + membrane * 0.55, side < 0.5);
	let g = select(body + membrane * 0.55, 0.0, side < 0.5);
	return vec4f(r, g, spark * 0.36, 1.0);
}

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let pair_i = ii / 2u;
	let side = ii % 2u;
	let p = pairs[pair_i];
	let stronger_side = side == 0u;
	let cell = select(p.weaker, p.stronger, stronger_side);
	let trust = cell.z;
	let width = cell.w;
	let radius = 0.028 + trust * 0.026 + width * 0.85;
	out.clip = vec4f(cell.xy + QUAD[vi] * radius, 0.0, 1.0);
	out.uv = QUAD[vi];
	out.misc = vec4f(trust, p.signals.x, f32(side), p.signals.z);
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let trust = clamp(in.misc.x, 0.0, 1.0);
	let side = in.misc.z;
	let unresolved = in.misc.w;
	let ivory = vec3f(0.96, 0.945, 0.815);
	let luciferin = vec3f(0.66, 1.0, 0.37);
	let redcore = vec3f(1.0, 0.23, 0.18);
	let weaker_green = vec3f(0.16, 0.95, 0.66);
	let core = select(weaker_green, mix(luciferin, ivory, trust), side < 0.5);
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.2, d));
	let body = exp(-d*d*3.0) * (0.22 + trust * 0.42);
	let flare = unresolved * smoothstep(0.18, 0.0, abs(d - 0.78));
	return vec4f(core * body + ivory * rim * (0.28 + trust * 0.52) + redcore * flare * 0.18, 1.0);
}

@vertex
fn vs_arc(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let p = pairs[ii];
	let a = p.stronger.xy;
	let b = p.weaker.xy;
	let t = f32(vi / 2u) / 31.0;
	let side = f32(vi % 2u) * 2.0 - 1.0;
	let mid = (a + b) * 0.5;
	let dir = normalize(b - a + vec2f(0.0001));
	let norm = vec2f(-dir.y, dir.x);
	let bow = norm * sin(t * 3.14159) * (0.06 + p.signals.x * 0.12);
	let pos = mix(a, b, t) + bow;
	let thickness = (0.004 + min(p.stronger.z, p.weaker.z) * 0.011) * (1.0 + p.signals.z * (0.4 + 0.4 * sin(params.time * 18.0 + t * 20.0)));
	out.clip = vec4f(pos + norm * side * thickness, 0.0, 1.0);
	out.uv = vec2f(t, side);
	out.misc = vec4f(p.signals.x, p.signals.y, p.signals.z, distance(pos, mid));
	return out;
}

@fragment
fn fs_arc(in: VSOut) -> @location(0) vec4f {
	let unresolved = in.misc.z;
	let overlap = in.misc.x;
	let pulse = 0.55 + 0.45 * sin(42.0 * in.uv.x - 8.0 * in.misc.w);
	let scarlet = vec3f(1.0, 0.13, 0.09);
	let darkred = vec3f(0.73, 0.05, 0.17);
	let color = mix(darkred, scarlet, pulse * unresolved + overlap * 0.35);
	return vec4f(color * (0.12 + unresolved * 0.55 + overlap * 0.26), 1.0);
}
`;

const MEMBRANE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(2) var field_sampler: sampler;
@group(0) @binding(3) var field_tex: texture_2d<f32>;

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
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(field_tex, 0));
	let px = 1.0 / max(dims, vec2f(1.0));
	let f = textureSample(field_tex, field_sampler, in.uv);
	let left = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(px.x, 0.0), 0.0);
	let right = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(px.x, 0.0), 0.0);
	let down = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(0.0, px.y), 0.0);
	let up = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(0.0, px.y), 0.0);

	let strong = clamp(f.r, 0.0, 5.0);
	let weak = clamp(f.g, 0.0, 5.0);
	let seam = min(strong, weak);
	let grad_r = vec2f(right.r - left.r, up.r - down.r);
	let grad_g = vec2f(right.g - left.g, up.g - down.g);
	let opposing = max(0.0, -dot(normalize(grad_r + vec2f(0.0001)), normalize(grad_g + vec2f(0.0001))));
	let fracture = smoothstep(0.11, 1.1, seam) * (0.42 + 0.9 * opposing);
	let membrane = smoothstep(0.10, 0.95, max(strong, weak)) * (1.0 - smoothstep(1.75, 3.6, max(strong, weak)));
	let spark = clamp(f.b, 0.0, 2.0) * (0.6 + 0.4 * params.pulse);

	let blackwater = vec3f(0.008, 0.012, 0.018);
	let stronger_glow = vec3f(1.0, 0.21, 0.16);
	let weaker_glow = vec3f(0.15, 0.95, 0.62);
	let ivory = vec3f(0.96, 0.945, 0.815);
	let scarlet = vec3f(1.0, 0.09, 0.055);
	let lacquer = vec3f(0.73, 0.05, 0.17);

	var color = blackwater * (0.16 + 0.05 * max(strong, weak));
	color = color + stronger_glow * strong * 0.045 + weaker_glow * weak * 0.035;
	color = color + ivory * membrane * 0.10;
	color = color + mix(lacquer, scarlet, opposing) * fracture * (0.55 + seam * 0.18);
	color = color + scarlet * spark * 0.28;
	let vignette = smoothstep(0.96, 0.20, distance(in.uv, vec2f(0.5)));
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

struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };

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
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, in.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`;

type GpuResources = {
	pairBuffer: GPUBuffer;
	blurHBuffer: GPUBuffer;
	blurVBuffer: GPUBuffer;
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

export class ContradictionsPass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: ContradictionsScene | null = null;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private splatBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private splatPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private cellPipeline: GPURenderPipeline | null = null;
	private arcPipeline: GPURenderPipeline | null = null;
	private pairCount = 0;
	private pairGeometry: { pair: ImmuneSynapsePair; ax: number; ay: number; bx: number; by: number; mx: number; my: number; radius: number }[] = [];

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as ContradictionsScene;
		this.buildPairGeometry();
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.splatPipeline || !this.engine.paramsBuffer) return;
		const splatModule = createDiagnosedShaderModule(device, 'contradictions-synapse-splat-wgsl', SPLAT_WGSL);
		const blurModule = createDiagnosedShaderModule(device, 'contradictions-synapse-blur-wgsl', BLUR_WGSL);
		const membraneModule = createDiagnosedShaderModule(device, 'contradictions-synapse-membrane-wgsl', MEMBRANE_WGSL);
		this.splatBindLayout = device.createBindGroupLayout({
			label: 'contradictions-synapse-splat-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'contradictions-synapse-blur-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'contradictions-synapse-membrane-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		const splatLayout = device.createPipelineLayout({ label: 'contradictions-synapse-splat-layout', bindGroupLayouts: [this.splatBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'contradictions-synapse-blur-layout', bindGroupLayouts: [this.blurBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'contradictions-synapse-membrane-layout', bindGroupLayouts: [this.membraneBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
		this.splatPipeline = device.createRenderPipeline({
			label: 'contradictions-field-additive-splat',
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
			label: 'contradictions-field-blur-render-pass',
			layout: blurLayout,
			vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] },
			primitive: { topology: 'triangle-list' }
		});
		const additive = { color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }, alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation } };
		this.membranePipeline = device.createRenderPipeline({
			label: 'contradictions-immune-synapse-membrane',
			layout: membraneLayout,
			vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.cellPipeline = device.createRenderPipeline({
			label: 'contradictions-memory-membrane-cells',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_cell' },
			fragment: { module: splatModule, entryPoint: 'fs_cell', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.arcPipeline = device.createRenderPipeline({
			label: 'contradictions-scarlet-unresolved-arcs',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_arc' },
			fragment: { module: splatModule, entryPoint: 'fs_arc', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-strip' }
		});
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.splatBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let pairBuffer = this.resources?.pairBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		if (!pairBuffer) {
			pairBuffer = device.createBuffer({ label: 'contradictions-pairs', size: MAX_PAIRS * PAIR_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		}
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'contradictions-blur-h-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'contradictions-blur-v-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!needsTextures && this.resources) return;
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'contradictions-field-a-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'contradictions-field-b-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const splatBindGroup = device.createBindGroup({
			label: 'contradictions-synapse-splat-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: pairBuffer } }
			]
		});
		const blurHBindGroup = device.createBindGroup({
			label: 'contradictions-field-blur-h-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldAView },
				{ binding: 2, resource: { buffer: blurHBuffer } }
			]
		});
		const blurVBindGroup = device.createBindGroup({
			label: 'contradictions-field-blur-v-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldBView },
				{ binding: 2, resource: { buffer: blurVBuffer } }
			]
		});
		const membraneBindGroup = device.createBindGroup({
			label: 'contradictions-membrane-bind',
			layout: this.membraneBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 2, resource: this.sampler },
				{ binding: 3, resource: fieldAView }
			]
		});
		this.resources = { pairBuffer, blurHBuffer, blurVBuffer, splatBindGroup, blurHBindGroup, blurVBindGroup, membraneBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
	}

	private buildPairGeometry(): void {
		const pairs = this.scene?.pairs ?? [];
		const n = Math.max(1, pairs.length);
		this.pairGeometry = pairs.slice(0, MAX_PAIRS).map((pair, i) => {
			const slot = (i / n) * Math.PI * 2 - Math.PI / 2;
			const lane = 0.42 + 0.10 * Math.sin(i * 1.7);
			const centerX = Math.cos(slot) * lane * 0.72;
			const centerY = Math.sin(slot) * lane;
			const normal = slot + Math.PI / 2;
			const spread = 0.16 + pair.topic_overlap * 0.14;
			const ax = centerX + Math.cos(normal) * spread;
			const ay = centerY + Math.sin(normal) * spread;
			const bx = centerX - Math.cos(normal) * spread;
			const by = centerY - Math.sin(normal) * spread;
			return { pair, ax, ay, bx, by, mx: centerX, my: centerY, radius: spread * 0.95 };
		});
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources) return;
		const pairData = new Float32Array(MAX_PAIRS * PAIR_FLOATS);
		this.pairCount = Math.min(MAX_PAIRS, this.pairGeometry.length);
		for (let i = 0; i < this.pairCount; i++) {
			const g = this.pairGeometry[i];
			const p = g.pair;
			const strongWidth = membraneWidth(p.stronger.trust);
			const weakWidth = membraneWidth(p.weaker.trust);
			pairData.set(
				[
					g.ax,
					g.ay,
					p.stronger.trust,
					strongWidth,
					g.bx,
					g.by,
					p.weaker.trust,
					weakWidth,
					p.topic_overlap,
					p.trust_delta,
					p.resolved ? 0 : 1,
					i / Math.max(1, this.pairCount),
					i,
					0,
					0,
					0
				],
				i * PAIR_FLOATS
			);
		}
		this.engine.params[4] = this.pairCount;
		device.queue.writeBuffer(this.resources.pairBuffer, 0, pairData);
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.splatPipeline || !this.blurPipeline) return;
		this.ensureResources(device);
		const res = this.resources;
		const splat = encoder.beginRenderPass({
			label: 'contradictions-field-splat-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		splat.setPipeline(this.splatPipeline);
		splat.setBindGroup(0, res.splatBindGroup);
		if (this.pairCount > 0) splat.draw(6, this.pairCount * 2);
		splat.end();

		const blurH = encoder.beginRenderPass({
			label: 'contradictions-field-blur-h-pass',
			colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();

		const blurV = encoder.beginRenderPass({
			label: 'contradictions-field-blur-v-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.cellPipeline || !this.arcPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		if (this.pairCount > 0) {
			pass.setPipeline(this.arcPipeline);
			pass.setBindGroup(0, this.resources.splatBindGroup);
			pass.draw(64, this.pairCount);
			pass.setPipeline(this.cellPipeline);
			pass.draw(6, this.pairCount * 2);
		}
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		for (let i = 0; i < this.pairGeometry.length; i++) {
			const g = this.pairGeometry[i];
			const dx = ndcX - g.mx;
			const dy = ndcY - g.my;
			const distToSeam = Math.hypot(dx, dy);
			const distToLine = distanceToSegment(ndcX, ndcY, g.ax, g.ay, g.bx, g.by);
			if (distToLine <= 0.075 || distToSeam <= g.radius) {
				return { id: g.pair.id, kind: 'contradiction-seam', index: i, payload: g.pair };
			}
		}
		return null;
	}

	dispose(): void {
		this.resources?.pairBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}
}

function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
	const vx = bx - ax;
	const vy = by - ay;
	const wx = px - ax;
	const wy = py - ay;
	const c1 = vx * wx + vy * wy;
	if (c1 <= 0) return Math.hypot(px - ax, py - ay);
	const c2 = vx * vx + vy * vy;
	if (c2 <= c1) return Math.hypot(px - bx, py - by);
	const t = c1 / c2;
	return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

function createDiagnosedShaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
	device.pushErrorScope('validation');
	const module = device.createShaderModule({ label, code });
	void module.getCompilationInfo().then((info) => {
		for (const message of info.messages) {
			console.error(`[observatory] ${label} WGSL ${message.type} ${message.lineNum}:${message.linePos} ${message.message}`);
		}
	});
	void device.popErrorScope().then((error) => {
		if (error) console.error(`[observatory] ${label} shader module validation: ${error.message}`);
	});
	return module;
}

export function createContradictionsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): ContradictionsPass[] {
	void rgb01(MEDIUM.blackwater);
	void rgb01(IMMUNE.veto);
	void rgb01(IMMUNE.suppressionScar);
	void rgb01(RETENTION.recall);
	return [new ContradictionsPass(engine, scene)];
}

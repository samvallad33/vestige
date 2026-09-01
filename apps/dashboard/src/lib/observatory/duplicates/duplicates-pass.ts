import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { IMMUNE, MEDIUM, RETENTION, membraneWidth, rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { DuplicateFusionCluster, DuplicatesScene } from './duplicates-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_CELLS = 512;
const MAX_NECKS = 512;
const CELL_FLOATS = 16;
const NECK_FLOATS = 16;

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

struct FusionCell {
	// x/y position in NDC, z retention, w winner flag
	pos_retention: vec4f,
	// x similarity, y threshold, z member slot, w cluster slot
	cluster_meta: vec4f,
	// x mismatch intensity, y merge flag, z radius, w member count
	visual_meta: vec4f,
	// x cell index, y cluster index, z/w spare
	ids: vec4f,
};

struct FusionNeck {
	// x/y winner position, z winner retention, w winner radius
	a: vec4f,
	// x/y candidate position, z candidate retention, w candidate radius
	b: vec4f,
	// x similarity, y threshold, z mismatch intensity, w merge flag
	signals: vec4f,
	// x neck index, y cluster index, z/w spare
	ids: vec4f,
};
`;

const SPLAT_WGSL = /* wgsl */ `
${COMMON_WGSL}

// FieldOpts mirrors the membrane's: x=intensity, yz=well center NDC, w=well
// half-w; then well half-h, floor, soft, pad. Cells/necks dim by the same amount
// so nothing blows out under the centered text overlay.
struct FieldOpts {
	intensity_wx_wy_hw: vec4f,
	hh_floor_soft_pad: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<FusionCell>;
@group(0) @binding(2) var<storage, read> necks: array<FusionNeck>;
@group(0) @binding(5) var<uniform> opts: FieldOpts;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) home: vec2f,
};

fn similarity_neck(similarity: f32) -> f32 {
	return smoothstep(0.78, 0.98, similarity);
}

// Reading-well multiplier at an NDC point (1.0 outside, →floor inside). hw<=0 off.
fn field_dim(ndc: vec2f) -> f32 {
	let intensity = clamp(opts.intensity_wx_wy_hw.x, 0.0, 1.0);
	let hw = opts.intensity_wx_wy_hw.w;
	if (hw <= 0.0) { return intensity; }
	let center = opts.intensity_wx_wy_hw.yz;
	let hh = opts.hh_floor_soft_pad.x;
	let floor_v = opts.hh_floor_soft_pad.y;
	let soft = max(0.02, opts.hh_floor_soft_pad.z);
	let d = abs(ndc - center) - vec2f(hw, hh);
	let outside = length(max(d, vec2f(0.0)));
	let inside = min(max(d.x, d.y), 0.0);
	let sd = outside + inside;
	let t = smoothstep(-soft, 0.0, sd);
	return intensity * mix(floor_v, 1.0, t);
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let corner = QUAD[vi];
	let cell_count = u32(params.node_count);
	if (ii < cell_count) {
		let c = cells[ii];
		let merge_gate = c.visual_meta.y;
		let radius = c.visual_meta.z * (1.0 + 0.045 * sin(params.time * 2.0 + c.cluster_meta.w * 6.28318));
		out.clip = vec4f(c.pos_retention.xy + corner * radius, 0.0, 1.0);
		out.uv = corner;
		out.misc = vec4f(c.pos_retention.z, c.cluster_meta.x, c.visual_meta.x, merge_gate);
		out.home = c.pos_retention.xy;
	} else {
		let n = necks[ii - cell_count];
		let a = n.a.xy;
		let b = n.b.xy;
		let center = (a + b) * 0.5;
		let dir = normalize(b - a + vec2f(0.0001, 0.0001));
		let normal = vec2f(-dir.y, dir.x);
		let fused = similarity_neck(n.signals.x);
		let length_half = distance(a, b) * 0.5;
		let thickness = 0.035 + fused * 0.085 + n.signals.z * 0.025;
		let pos = center + dir * corner.x * length_half + normal * corner.y * thickness;
		out.clip = vec4f(pos, 0.0, 1.0);
		out.uv = vec2f(corner.x, corner.y / max(0.001, thickness));
		out.misc = vec4f(n.signals.x, fused, n.signals.z, n.signals.w);
		out.home = center;
	}
	return out;
}

@fragment
fn fs_splat(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.uv);
	let is_neck = f32(abs(frag.uv.y) > 1.0);
	if (is_neck < 0.5 && d > 1.0) { discard; }
	let retention = clamp(frag.misc.x, 0.0, 1.0);
	let similarity = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let merge_gate = frag.misc.w;
	let cell_body = exp(-d * d * 3.15) * (0.38 + retention * 0.62) * (0.5 + similarity * 0.58);
	let cell_rim = smoothstep(0.24, 0.02, abs(d - (0.58 + retention * 0.16))) * (0.2 + similarity * 0.55);
	let neck_body = exp(-frag.uv.y * frag.uv.y * 4.0) * smoothstep(1.05, 0.82, abs(frag.uv.x)) * (0.35 + similarity * 0.9);
	let density = max(cell_body + cell_rim, neck_body * (0.4 + similarity));
	// The splat writes the density FIELD (blurred into the membrane). It must NOT
	// be dimmed here — the membrane fragment applies intensity + reading well once,
	// so dimming both would double-darken. r=density, g=retention, b=mismatch amber.
	return vec4f(density, density * (0.35 + retention * 0.65), mismatch * (0.18 + merge_gate * 0.12), 1.0);
}

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let c = cells[ii];
	let corner = QUAD[vi];
	let winner = c.pos_retention.w;
	let radius = c.visual_meta.z * (0.46 + winner * 0.18);
	out.clip = vec4f(c.pos_retention.xy + corner * radius, 0.0, 1.0);
	out.uv = corner;
	out.misc = vec4f(c.pos_retention.z, c.cluster_meta.x, c.visual_meta.x, winner);
	out.home = c.pos_retention.xy;
	return out;
}

@fragment
fn fs_cell(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(frag.misc.x, 0.0, 1.0);
	let similarity = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let winner = frag.misc.w;
	let sediment = vec3f(0.54, 0.29, 0.09);
	let recall = vec3f(0.16, 0.95, 0.66);
	let luciferin = vec3f(0.91, 1.0, 0.72);
	let ivory = vec3f(0.96, 0.945, 0.815);
	let amber = vec3f(1.0, 0.69, 0.08);
	let core = mix(sediment, mix(recall, luciferin, retention), retention);
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.22, d));
	let body = exp(-d*d*3.2) * (0.20 + retention * 0.44 + winner * 0.16);
	let mismatch_ring = smoothstep(0.16, 0.0, abs(d - 0.80)) * mismatch;
	let color = core * body + ivory * rim * (0.16 + similarity * 0.52) + amber * mismatch_ring * 0.34;
	// Sharp cells draw on TOP of the membrane, so dim them by the same field
	// intensity + reading well or they'd punch through the centered text.
	return vec4f(color * field_dim(frag.home), 1.0);
}

@vertex
fn vs_neck(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let n = necks[ii];
	let a = n.a.xy;
	let b = n.b.xy;
	let t = f32(vi / 2u) / 31.0;
	let side = f32(vi % 2u) * 2.0 - 1.0;
	let dir = normalize(b - a + vec2f(0.0001, 0.0001));
	let normal = vec2f(-dir.y, dir.x);
	let midpoint = (a + b) * 0.5;
	let fused = similarity_neck(n.signals.x);
	let threshold_pull = clamp(n.signals.x - n.signals.y + 0.22, 0.0, 1.0);
	let bow = normal * sin(t * 3.14159) * (0.030 + n.signals.z * 0.050) * (1.0 - fused * 0.35);
	let pos = mix(a, b, t) + bow;
	let thickness = 0.005 + fused * 0.025 + threshold_pull * 0.010;
	out.clip = vec4f(pos + normal * side * thickness, 0.0, 1.0);
	out.uv = vec2f(t, side);
	out.misc = vec4f(n.signals.x, n.signals.y, n.signals.z, distance(pos, midpoint));
	out.home = midpoint;
	return out;
}

@fragment
fn fs_neck(frag: VSOut) -> @location(0) vec4f {
	let similarity = clamp(frag.misc.x, 0.0, 1.0);
	let threshold = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let pulse = 0.55 + 0.45 * sin(36.0 * frag.uv.x - 8.0 * frag.misc.w);
	let bridge = vec3f(0.10, 0.82, 0.92);
	let luciferin = vec3f(0.91, 1.0, 0.72);
	let amber = vec3f(1.0, 0.69, 0.08);
	let pull = smoothstep(-0.08, 0.20, similarity - threshold);
	let color = mix(bridge, luciferin, pull) + amber * mismatch * pulse * 0.34;
	// Necks draw on TOP of the membrane too — dim by field intensity + reading well.
	return vec4f(color * (0.14 + similarity * 0.55 + mismatch * 0.18) * field_dim(frag.home), 1.0);
}
`;

const MEMBRANE_WGSL = /* wgsl */ `
${COMMON_WGSL}

// FieldOpts: x=intensity (0..1 overall dim), yz=well center NDC, w=well half-w,
// then well half-h, floor (min emission inside well), soft (edge falloff), pad.
// Lets a text-heavy organ dim the whole field AND carve a reading well under the
// centered DOM overlay so the labels/values read. hw<=0 disables the well.
struct FieldOpts {
	intensity_wx_wy_hw: vec4f,
	hh_floor_soft_pad: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> opts: FieldOpts;

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

// Reading-well multiplier at an NDC point: 1.0 outside the well, falling toward
// the floor value inside it (smooth edge of width soft). Disabled when hw<=0.
fn reading_well(ndc: vec2f) -> f32 {
	let hw = opts.intensity_wx_wy_hw.w;
	if (hw <= 0.0) { return 1.0; }
	let center = opts.intensity_wx_wy_hw.yz;
	let hh = opts.hh_floor_soft_pad.x;
	let floor_v = opts.hh_floor_soft_pad.y;
	let soft = max(0.02, opts.hh_floor_soft_pad.z);
	let d = abs(ndc - center) - vec2f(hw, hh);
	// signed distance to rect edge: <0 inside, >0 outside
	let outside = length(max(d, vec2f(0.0)));
	let inside = min(max(d.x, d.y), 0.0);
	let sd = outside + inside;
	// sd<=-soft → fully inside (floor); sd>=0 → outside (1.0)
	let t = smoothstep(-soft, 0.0, sd);
	return mix(floor_v, 1.0, t);
}

@fragment
fn fs_membrane(frag: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, frag.uv);
	let density = clamp(f.r, 0.0, 5.0);
	let retention = clamp(f.g, 0.0, 5.0);
	let mismatch = clamp(f.b, 0.0, 3.0);
	let membrane = smoothstep(0.13, 0.88, density) * (1.0 - smoothstep(1.9, 3.8, density));
	let blackwater = vec3f(0.008, 0.012, 0.018);
	let bridge = vec3f(0.10, 0.82, 0.92);
	let luciferin = vec3f(0.66, 1.0, 0.37);
	let ivory = vec3f(0.96, 0.945, 0.815);
	let amber = vec3f(1.0, 0.69, 0.08);
	var color = blackwater * (0.18 + density * 0.055);
	color = color + bridge * density * 0.055 + luciferin * retention * 0.080;
	color = color + ivory * membrane * 0.22 + amber * mismatch * (0.20 + 0.08 * params.pulse);
	let vignette = smoothstep(0.96, 0.18, distance(frag.uv, vec2f(0.5)));
	let ndc = frag.uv * 2.0 - vec2f(1.0);
	let dim = clamp(opts.intensity_wx_wy_hw.x, 0.0, 1.0) * reading_well(ndc);
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness * dim, 1.0);
}
`;

const BLUR_WGSL = /* wgsl */ `
struct BlurDir { dir: vec2f, _pad: vec2f };
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
fn fs_blur(frag: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, frag.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`;

type GpuResources = {
	cellBuffer: GPUBuffer;
	neckBuffer: GPUBuffer;
	blurHBuffer: GPUBuffer;
	blurVBuffer: GPUBuffer;
	optsBuffer: GPUBuffer;
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

type CellGeometry = {
	cluster: DuplicateFusionCluster;
	memoryId: string;
	x: number;
	y: number;
	retention: number;
	winner: boolean;
	mismatch: number;
	radius: number;
	memberSlot: number;
	memberCount: number;
};

type NeckGeometry = {
	cluster: DuplicateFusionCluster;
	winnerId: string;
	candidateId: string;
	ax: number;
	ay: number;
	bx: number;
	by: number;
	winnerRetention: number;
	candidateRetention: number;
	winnerRadius: number;
	candidateRadius: number;
	mismatch: number;
};

export class DuplicatesPass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: DuplicatesScene | null = null;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private splatBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private splatPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private cellPipeline: GPURenderPipeline | null = null;
	private neckPipeline: GPURenderPipeline | null = null;
	private cellCount = 0;
	private neckCount = 0;
	private cellGeometry: CellGeometry[] = [];
	private neckGeometry: NeckGeometry[] = [];
	// 0..1 overall field intensity — text-heavy /duplicates dims to a calm backdrop
	// so the centered DOM overlay (cluster cards, threshold, counts) stays legible.
	private intensity = 0.22;
	// Reading well (NDC rect): the field emits LESS inside it so the centered text
	// column reads. hw<=0 disables it.
	private well = { x: 0, y: 0, hw: -1, hh: 0, floor: 0.1, soft: 0.22 };

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	/**
	 * Set the overall field intensity (0..1). LOW (~0.22) = dim backdrop for this
	 * text-heavy organ; HIGH = the field is the hero. Picked up immediately by the
	 * membrane + on-top cell/neck shaders via the shared FieldOpts uniform.
	 */
	setIntensity(v: number): void {
		this.intensity = Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0.22));
		const d = this.engine.gpuDevice;
		if (d) this.writeOpts(d);
	}

	/**
	 * Set a "reading well": the field emits LESS inside this NDC rectangle so the
	 * DOM text on top reads. hw<=0 disables it. /duplicates renders its overlay in a
	 * centered `mx-auto max-w-5xl` column, so the well is centered, not a left rail.
	 */
	setReadingWell(r: { x: number; y: number; hw: number; hh: number; floor?: number; soft?: number }): void {
		const finiteN = (v: number, fb = 0) => (Number.isFinite(v) ? v : fb);
		this.well = {
			x: finiteN(r.x),
			y: finiteN(r.y),
			hw: finiteN(r.hw, -1),
			hh: finiteN(r.hh),
			floor: Math.min(1, Math.max(0, finiteN(r.floor ?? 0.1, 0.1))),
			soft: Math.max(0.02, finiteN(r.soft ?? 0.22, 0.22))
		};
		const d = this.engine.gpuDevice;
		if (d) this.writeOpts(d);
	}

	/** Write the FieldOpts uniform (intensity + reading-well rect). 8 floats. */
	private writeOpts(device: GPUDevice): void {
		if (!this.resources) return;
		device.queue.writeBuffer(
			this.resources.optsBuffer,
			0,
			new Float32Array([
				this.intensity,
				this.well.x,
				this.well.y,
				this.well.hw,
				this.well.hh,
				this.well.floor,
				this.well.soft,
				0
			])
		);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as DuplicatesScene;
		this.buildGeometry();
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.splatPipeline || !this.engine.paramsBuffer) return;
		const splatModule = createDiagnosedShaderModule(device, 'duplicates-fusion-splat-wgsl', SPLAT_WGSL);
		const blurModule = createDiagnosedShaderModule(device, 'duplicates-fusion-blur-wgsl', BLUR_WGSL);
		const membraneModule = createDiagnosedShaderModule(device, 'duplicates-fusion-membrane-wgsl', MEMBRANE_WGSL);
		this.splatBindLayout = device.createBindGroupLayout({
			label: 'duplicates-fusion-splat-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				// FieldOpts (intensity + reading well) — read in fs_cell/fs_neck only.
				{ binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'duplicates-fusion-blur-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'duplicates-fusion-membrane-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				// FieldOpts (intensity + reading well) — dims the full-bleed membrane.
				{ binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const splatLayout = device.createPipelineLayout({ label: 'duplicates-fusion-splat-layout', bindGroupLayouts: [this.splatBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'duplicates-fusion-blur-layout', bindGroupLayouts: [this.blurBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'duplicates-fusion-membrane-layout', bindGroupLayouts: [this.membraneBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
		const additive = { color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }, alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation } };
		this.splatPipeline = device.createRenderPipeline({
			label: 'duplicates-field-additive-splat',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_splat' },
			fragment: { module: splatModule, entryPoint: 'fs_splat', targets: [{ format: FIELD_FORMAT, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.blurPipeline = device.createRenderPipeline({
			label: 'duplicates-field-blur-render-pass',
			layout: blurLayout,
			vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] },
			primitive: { topology: 'triangle-list' }
		});
		this.membranePipeline = device.createRenderPipeline({
			label: 'duplicates-synaptic-fusion-membrane',
			layout: membraneLayout,
			vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.cellPipeline = device.createRenderPipeline({
			label: 'duplicates-memory-nuclei',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_cell' },
			fragment: { module: splatModule, entryPoint: 'fs_cell', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.neckPipeline = device.createRenderPipeline({
			label: 'duplicates-mismatch-filaments',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_neck' },
			fragment: { module: splatModule, entryPoint: 'fs_neck', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-strip' }
		});
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.splatBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let cellBuffer = this.resources?.cellBuffer;
		let neckBuffer = this.resources?.neckBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		let optsBuffer = this.resources?.optsBuffer;
		if (!cellBuffer) cellBuffer = device.createBuffer({ label: 'duplicates-cells', size: MAX_CELLS * CELL_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!neckBuffer) neckBuffer = device.createBuffer({ label: 'duplicates-necks', size: MAX_NECKS * NECK_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'duplicates-blur-h-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'duplicates-blur-v-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!optsBuffer) {
			// FieldOpts: intensity + reading-well rect. 8 floats = 32 bytes.
			optsBuffer = device.createBuffer({ label: 'duplicates-field-opts', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
		}
		if (!needsTextures && this.resources) {
			// Buffers already exist; the caller may have just changed intensity/well.
			this.resources.optsBuffer = optsBuffer;
			this.writeOpts(device);
			return;
		}
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'duplicates-field-a-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'duplicates-field-b-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const splatBindGroup = device.createBindGroup({
			label: 'duplicates-fusion-splat-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: cellBuffer } },
				{ binding: 2, resource: { buffer: neckBuffer } },
				{ binding: 5, resource: { buffer: optsBuffer } }
			]
		});
		const blurHBindGroup = device.createBindGroup({
			label: 'duplicates-field-blur-h-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldAView },
				{ binding: 2, resource: { buffer: blurHBuffer } }
			]
		});
		const blurVBindGroup = device.createBindGroup({
			label: 'duplicates-field-blur-v-bind',
			layout: this.blurBindLayout,
			entries: [
				{ binding: 0, resource: this.sampler },
				{ binding: 1, resource: fieldBView },
				{ binding: 2, resource: { buffer: blurVBuffer } }
			]
		});
		const membraneBindGroup = device.createBindGroup({
			label: 'duplicates-membrane-bind',
			layout: this.membraneBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 3, resource: this.sampler },
				{ binding: 4, resource: fieldAView },
				{ binding: 5, resource: { buffer: optsBuffer } }
			]
		});
		this.resources = { cellBuffer, neckBuffer, blurHBuffer, blurVBuffer, optsBuffer, splatBindGroup, blurHBindGroup, blurVBindGroup, membraneBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
		this.writeOpts(device);
	}

	private buildGeometry(): void {
		const clusters = this.scene?.clusters ?? [];
		const n = Math.max(1, clusters.length);
		const cells: CellGeometry[] = [];
		const necks: NeckGeometry[] = [];
		// FAIR cell allocator (Wave 3, Claude + GPT-5.6-sol, hardened in cross-
		// review): the old sequential fill let one oversized similarity component
		// (428 members on the real brain) devour 428/512 cells, dropping clusters
		// 5-18 from the field entirely. Policy, in order, each pass budget-capped:
		//   pass 0 — ONE cell (the winner) per cluster, so representation degrades
		//            to winner-only before any cluster is dropped (a naive
		//            2-per-cluster first pass silently zeroed clusters past #256);
		//   pass 1 — upgrade to 2 (winner + strongest candidate) in order;
		//   pass 2 — round-robin the remainder up to a per-cluster visual cap.
		// Beyond MAX_CELLS clusters (>512), later clusters are deterministically
		// omitted — unavoidable with a fixed budget, and far past any real scene.
		const PER_CLUSTER_CELL_CAP = 12;
		const alloc = new Array<number>(clusters.length).fill(0);
		let budget = MAX_CELLS;
		for (let i = 0; i < clusters.length && budget > 0; i++) {
			alloc[i] = 1;
			budget -= 1;
		}
		for (let i = 0; i < clusters.length && budget > 0; i++) {
			if (alloc[i] === 1 && clusters[i].memories.length >= 2) {
				alloc[i] = 2;
				budget -= 1;
			}
		}
		let grew = true;
		while (budget > 0 && grew) {
			grew = false;
			for (let i = 0; i < clusters.length && budget > 0; i++) {
				if (alloc[i] > 0 && alloc[i] < Math.min(clusters[i].memories.length, PER_CLUSTER_CELL_CAP)) {
					alloc[i] += 1;
					budget -= 1;
					grew = true;
				}
			}
		}
		for (let i = 0; i < clusters.length; i++) {
			const cluster = clusters[i];
			// Zero allocation = deterministically omitted (only possible past
			// MAX_CELLS clusters). Skip honestly instead of faking a 1-member
			// subset that the global cell guard would silently drop anyway.
			if (alloc[i] === 0) continue;
			const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
			const lane = 0.18 + 0.58 * Math.sqrt((i + 0.5) / n);
			const centerX = Math.cos(angle) * lane * 0.86;
			const centerY = Math.sin(angle) * lane;
			const pull = Math.max(0.04, 0.25 - Math.max(0, cluster.similarity - cluster.threshold) * 0.55);
			const winnerMemory = cluster.memories.find((m) => m.id === cluster.winnerId) ?? cluster.memories[0];
			// Winner-first subset of the allocated size — the winner must always be
			// on the field (necks radiate from it), the rest are the top candidates.
			const rendered = [
				winnerMemory,
				...cluster.memories.filter((m) => m.id !== winnerMemory.id)
			].slice(0, alloc[i]);
			const memberCount = Math.max(1, rendered.length);
			const cellById = new Map<string, CellGeometry>();
			for (let j = 0; j < rendered.length && cells.length < MAX_CELLS; j++) {
				const memory = rendered[j];
				const memberAngle = angle + (j / memberCount) * Math.PI * 2 + (memberCount % 2 ? 0 : Math.PI / memberCount);
				const winner = memory.id === cluster.winnerId;
				const spread = winner ? pull * 0.18 : pull + 0.025 * (j % 3);
				const mismatch = Math.min(1, (memory.mismatchTokens?.length ?? 0) / 8);
				const radius = 0.085 + Math.min(0.045, membraneWidth(memory.retention) * 2.1) + (winner ? 0.012 : 0);
				const cell = {
					cluster,
					memoryId: memory.id,
					x: centerX + Math.cos(memberAngle) * spread,
					y: centerY + Math.sin(memberAngle) * spread,
					retention: Math.max(0, Math.min(1, memory.retention || 0)),
					winner,
					mismatch,
					radius,
					memberSlot: j,
					memberCount
				};
				cellById.set(memory.id, cell);
				cells.push(cell);
			}
			const winnerCell = cellById.get(winnerMemory.id);
			if (!winnerCell) continue;
			for (const memory of cluster.memories) {
				if (necks.length >= MAX_NECKS || memory.id === winnerMemory.id) continue;
				const candidate = cellById.get(memory.id);
				if (!candidate) continue;
				necks.push({
					cluster,
					winnerId: winnerMemory.id,
					candidateId: memory.id,
					ax: winnerCell.x,
					ay: winnerCell.y,
					bx: candidate.x,
					by: candidate.y,
					winnerRetention: winnerCell.retention,
					candidateRetention: candidate.retention,
					winnerRadius: winnerCell.radius,
					candidateRadius: candidate.radius,
					mismatch: Math.max(candidate.mismatch, Math.min(1, cluster.mismatchTokens.length / 12))
				});
			}
		}
		this.cellGeometry = cells;
		this.neckGeometry = necks;
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources) return;
		const cellData = new Float32Array(MAX_CELLS * CELL_FLOATS);
		const neckData = new Float32Array(MAX_NECKS * NECK_FLOATS);
		this.cellCount = Math.min(MAX_CELLS, this.cellGeometry.length);
		this.neckCount = Math.min(MAX_NECKS, this.neckGeometry.length);
		for (let i = 0; i < this.cellCount; i++) {
			const c = this.cellGeometry[i];
			cellData.set([
				c.x,
				c.y,
				c.retention,
				c.winner ? 1 : 0,
				c.cluster.similarity,
				c.cluster.threshold,
				c.memberSlot,
				c.cluster.index,
				c.mismatch,
				c.cluster.suggestedAction === 'merge' ? 1 : 0,
				c.radius,
				c.memberCount,
				i,
				c.cluster.index,
				0,
				0
			], i * CELL_FLOATS);
		}
		for (let i = 0; i < this.neckCount; i++) {
			const n = this.neckGeometry[i];
			neckData.set([
				n.ax,
				n.ay,
				n.winnerRetention,
				n.winnerRadius,
				n.bx,
				n.by,
				n.candidateRetention,
				n.candidateRadius,
				n.cluster.similarity,
				n.cluster.threshold,
				n.mismatch,
				n.cluster.suggestedAction === 'merge' ? 1 : 0,
				i,
				n.cluster.index,
				0,
				0
			], i * NECK_FLOATS);
		}
		this.engine.params[2] = this.cellCount;
		this.engine.params[3] = this.neckCount;
		this.engine.params[4] = this.neckCount;
		device.queue.writeBuffer(this.resources.cellBuffer, 0, cellData);
		device.queue.writeBuffer(this.resources.neckBuffer, 0, neckData);
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.splatPipeline || !this.blurPipeline) return;
		this.ensureResources(device);
		const res = this.resources;
		const splat = encoder.beginRenderPass({
			label: 'duplicates-field-splat-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		splat.setPipeline(this.splatPipeline);
		splat.setBindGroup(0, res.splatBindGroup);
		splat.draw(6, this.cellCount + this.neckCount);
		splat.end();

		const blurH = encoder.beginRenderPass({
			label: 'duplicates-field-blur-h-pass',
			colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();

		const blurV = encoder.beginRenderPass({
			label: 'duplicates-field-blur-v-pass',
			colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }]
		});
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.cellPipeline || !this.neckPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		if (this.neckCount > 0) {
			pass.setPipeline(this.neckPipeline);
			pass.setBindGroup(0, this.resources.splatBindGroup);
			pass.draw(64, this.neckCount);
		}
		if (this.cellCount > 0) {
			pass.setPipeline(this.cellPipeline);
			pass.setBindGroup(0, this.resources.splatBindGroup);
			pass.draw(6, this.cellCount);
		}
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		for (let i = 0; i < this.neckGeometry.length; i++) {
			const g = this.neckGeometry[i];
			const distToLine = distanceToSegment(ndcX, ndcY, g.ax, g.ay, g.bx, g.by);
			const midX = (g.ax + g.bx) * 0.5;
			const midY = (g.ay + g.by) * 0.5;
			const fusedRadius = 0.055 + Math.max(0, g.cluster.similarity - g.cluster.threshold) * 0.45;
			if (distToLine <= fusedRadius || Math.hypot(ndcX - midX, ndcY - midY) <= fusedRadius) {
				return { id: g.cluster.id, kind: 'duplicate-neck', index: i, payload: g.cluster };
			}
		}
		for (let i = 0; i < this.cellGeometry.length; i++) {
			const c = this.cellGeometry[i];
			if (Math.hypot(ndcX - c.x, ndcY - c.y) <= c.radius * 0.8) {
				return { id: c.memoryId, kind: 'duplicate-memory', index: i, payload: c.cluster };
			}
		}
		return null;
	}

	dispose(): void {
		this.resources?.cellBuffer.destroy();
		this.resources?.neckBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.optsBuffer.destroy();
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

export function createDuplicatesPasses(engine: ObservatoryEngine, scene: RouteSceneModel): DuplicatesPass[] {
	void rgb01(MEDIUM.blackwater);
	void rgb01(RETENTION.recall);
	void rgb01(RETENTION.luciferin);
	void rgb01(IMMUNE.trustMembrane);
	const pass = new DuplicatesPass(engine, scene);
	// /duplicates is a TEXT-HEAVY organ: the DOM overlay (cluster cards, threshold,
	// counts) is the content, the synaptic-fusion field is a DIM backdrop. Drop the
	// field to 0.22 and carve a centered reading well under the `mx-auto max-w-5xl`
	// column so every label/value reads (mirrors observatory's setIntensity+well).
	pass.setIntensity(0.22);
	pass.setReadingWell({ x: 0, y: 0, hw: 0.6, hh: 0.85, floor: 0.08, soft: 0.25 });
	return [pass];
}

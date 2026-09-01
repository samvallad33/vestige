import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { MEDIUM, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { TimelineCell, TimelineRing, TimelineScene } from './timeline-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const FIELD_FORMAT: GPUTextureFormat = 'rgba16float';
const MAX_CELLS = 768;
const MAX_RINGS = 96;
const CELL_FLOATS = 16;
const RING_FLOATS = 12;

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

struct TimelineCellGpu {
	// x,y NDC; z cell radius; w ring radius
	pos_radius: vec4f,
	// x retention, y rewritten, z suppressed, w audit events
	signals: vec4f,
	// x valid-time phase, y transaction-time phase, z day index, w cell index
	time_meta: vec4f,
	// x selected, y reserved, z reserved, w reserved
	flags: vec4f,
};

struct TimelineRingGpu {
	// x radius, y count scale, z retention, w day index
	shape: vec4f,
	// x updated count, y suppressed count, z phase, w selected
	activity: vec4f,
	// x memory count, y ring index, z reserved, w reserved
	// ('meta' is a WGSL reserved keyword — see GOD-TIER §9 / it broke Blackbox too)
	stats: vec4f,
};

// Portrait legibility: on a phone the growth-ring field is the whole screen and
// its HDR bloom becomes a BLINDING blob that drowns the MSDF HUD/receipt text.
// Derive a dim factor from the LIVE viewport aspect (viewport_w/viewport_h) —
// nothing is hardcoded per device. Landscape/desktop (aspect >= 0.85) is left at
// full brightness (1.0); portrait scales down toward ~0.34 as it narrows so the
// field becomes a DIM backdrop and the overlay text wins the contrast fight.
fn portrait_field_dim() -> f32 {
	let a = params.viewport_w / max(params.viewport_h, 1.0);
	// portraitness: 0 at aspect 0.85 (landscape edge) -> 1 at aspect 0.46 (tall phone)
	let p = clamp((0.85 - a) / (0.85 - 0.46), 0.0, 1.0);
	// The ring/membrane colors are pushed HARD into HDR (peak accumulated ~5-8x via
	// additive blend) specifically so the post-chain bloom flares them. A 0.2 dim
	// still leaves ~1.0-1.6 — above the bloom knee, so it stayed a blinding blob on
	// a phone. Pull it down to ~0.07 at full portrait so even the accumulated HDR
	// peak lands well below the bloom threshold and the field reads as a true DIM
	// backdrop the MSDF HUD/receipt text can win against. Aspect-derived, no per-
	// device constant; landscape/desktop (aspect>=0.85) stays untouched at 1.0.
	return mix(1.0, 0.07, p);
}
`;

const SPLAT_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<TimelineCellGpu>;
@group(0) @binding(2) var<storage, read> rings: array<TimelineRingGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) extra: vec4f,
};

// Living orbital drift: every cell slowly circulates around the ring center
// (the tree of memory is always turning), plus a per-cell radial breathe. Motion
// is a pure function of params.time + per-cell phase — deterministic, no RNG.
// This is what makes the field MOVE like the Observatory force-sim, not sit still.
// Shared rotation for a given normalized day phase (0 = oldest/outer, 1 = newest/
// inner). Inner rings turn faster, like the fast core of a spinning galaxy. Cells
// AND their rings both call this so cells stay ON their ring while everything turns.
fn ring_spin(day_phase: f32) -> f32 {
	let speed = 0.045 + day_phase * 0.10;
	return params.time * speed;
}

fn orbit(base: vec2f, phase: f32, day_phase: f32, ret: f32) -> vec2f {
	let radius = length(base);
	if (radius < 0.0001) { return base; }
	let ang0 = atan2(base.y, base.x);
	// rotate with the ring, plus a tiny per-cell wobble so cells shimmer on the ring
	let ang = ang0 + ring_spin(day_phase) + sin(params.time * 0.6 + phase * 6.283) * 0.02;
	// radial breathe so the whole tree gently expands/contracts as it turns
	let rr = radius * (1.0 + 0.016 * sin(params.time * 1.1 + phase * 6.283));
	return vec2f(cos(ang), sin(ang)) * rr;
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let breathe = 1.0 + 0.10 * sin(params.time * 1.6 + c.time_meta.x * 6.28318);
	let r = c.pos_radius.z * breathe * (1.0 + c.flags.x * 1.4);
	let center = orbit(c.pos_radius.xy, c.time_meta.w, c.time_meta.x, c.signals.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.misc = c.signals;
	out.extra = c.time_meta;
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewritten = in.misc.y;
	let suppressed = in.misc.z;
	let audit = clamp(in.misc.w, 0.0, 8.0) / 8.0;
	let body = exp(-d*d*3.1) * (0.34 + retention * 0.86);
	let seam = rewritten * smoothstep(0.10, 0.0, abs(d - 0.52)) * (0.55 + audit * 0.8);
	let scar = suppressed * smoothstep(0.98, 0.68, d);
	// .r = valid-time growth density, .g = retention oxygen, .b = transaction-time seam/shadow
	return vec4f(body, body * retention, seam + scar * 0.45, 1.0);
}

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	// pulse the cell size with its own heartbeat so cells throb as they orbit
	let beat = 1.0 + 0.22 * sin(params.time * 2.3 + c.time_meta.w * 1.7);
	let r = c.pos_radius.z * (0.55 + c.flags.x * 0.8) * beat;
	let center = orbit(c.pos_radius.xy, c.time_meta.w, c.time_meta.x, c.signals.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.misc = c.signals;
	out.extra = c.time_meta;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewritten = in.misc.y;
	let suppressed = in.misc.z;
	let oxygen = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.95, 0.55, 0.15);
	let indigo = vec3f(0.486, 0.424, 1.0);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	let core = mix(amber, oxygen, retention);
	// Each memory cell is a living bioluminescent organism — pulse by its own phase
	// (time_meta.x) so the field twinkles, and push core to HDR so it GLOWS.
	let cell_phase = in.extra.x;
	let twinkle = 0.6 + 0.8 * (0.5 + 0.5 * sin(params.time * 2.1 + cell_phase * 26.0));
	let body = exp(-d*d*2.7) * (0.55 + retention * 1.7) * twinkle;
	let rim = smoothstep(0.98, 0.74, d) * (1.0 - smoothstep(0.74, 0.42, d));
	let seam = smoothstep(0.12, 0.0, abs(d - 0.48)) * rewritten;
	let scar = smoothstep(0.16, 0.0, abs(d - 0.76)) * suppressed;
	return vec4f((core * body + vec3f(0.91, 1.0, 0.72) * rim * 1.1 + indigo * seam * 1.3 + scarlet * scar * 1.5) * portrait_field_dim(), 1.0);
}

@vertex
fn vs_ring(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let ring = rings[ii];
	let seg = vi / 2u;
	let side = f32(vi % 2u) * 2.0 - 1.0;
	let t = f32(seg) / 95.0;
	// rotate the whole ring with the same galaxy spin the cells use (activity.z =
	// normalized ring phase) so cells ride ON their turning ring, alive together.
	let angle = t * 6.2831853 + ring_spin(ring.activity.z);
	let dir = vec2f(cos(angle), sin(angle));
	let retention = ring.shape.z;
	let rewrite = ring.activity.x / max(1.0, ring.stats.x);
	let suppressed = ring.activity.y / max(1.0, ring.stats.x);
	let thickness = 0.0035 + 0.006 * retention + 0.004 * ring.activity.w;
	let ripple = 0.006 * sin(angle * 9.0 + params.time * (0.28 + ring.activity.z));
	let radius = ring.shape.x + side * thickness + ripple * rewrite;
	let tx = 0.030 * rewrite;
	var out: VSOut;
	// Indigo transaction-time shadow: duplicate the ring instance offset by the real rewrite amount.
	let indigo_shift = select(0.0, tx, side > 0.0);
	out.clip = vec4f(dir * radius + vec2f(indigo_shift, -indigo_shift * 0.42), 0.0, 1.0);
	out.uv = vec2f(t, side);
	out.misc = vec4f(retention, rewrite, suppressed, ring.activity.w);
	out.extra = vec4f(ring.shape.y, ring.shape.w, ring.stats.x, ring.activity.z);
	return out;
}

@fragment
fn fs_ring(in: VSOut) -> @location(0) vec4f {
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewrite = clamp(in.misc.y, 0.0, 1.0);
	let suppressed = clamp(in.misc.z, 0.0, 1.0);
	let selected = in.misc.w;
	let tick = step(0.86, fract(in.uv.x * 24.0));
	let oxygen = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.86, 0.42, 0.12);
	let indigo = vec3f(0.486, 0.424, 1.0);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	// Living pulse: each ring breathes with the global breath + a per-ring phase so
	// the rings shimmer OUT OF SYNC like a real organism, not one flat pattern.
	let phase = in.extra.w; // ring.activity.z packed as phase
	let live = 0.55 + 0.65 * (0.5 + 0.5 * sin(params.time * (0.9 + phase * 1.3) + phase * 6.283));
	// HDR brightness (>1) so the enzyme light BLOOMS through the post chain.
	var color = mix(amber, oxygen, retention) * (0.5 + 1.5 * retention + 1.1 * selected) * live;
	color = color + indigo * rewrite * (1.1 + 0.7 * abs(in.uv.y));
	color = color + scarlet * suppressed * 1.4;
	// Bright engraved date ticks flare on selection.
	color = color + vec3f(0.91, 1.0, 0.72) * tick * (0.14 + selected * 0.6);
	return vec4f(color * portrait_field_dim(), 1.0);
}
`;

const MEMBRANE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;

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
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(field_tex, 0));
	let px = 1.0 / max(dims, vec2f(1.0));
	let f = textureSample(field_tex, field_sampler, in.uv);
	let left = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(px.x, 0.0), 0.0);
	let right = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(px.x, 0.0), 0.0);
	let down = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(0.0, px.y), 0.0);
	let up = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(0.0, px.y), 0.0);
	let density = clamp(f.r, 0.0, 5.0);
	let oxygen = clamp(f.g, 0.0, 5.0);
	let seam = clamp(f.b, 0.0, 3.0);
	let grad = length(vec2f((right.r + right.g) - (left.r + left.g), (up.r + up.g) - (down.r + down.g)));
	let membrane = smoothstep(0.08, 0.70, density) * (1.0 - smoothstep(1.8, 3.8, density));
	let edge = smoothstep(0.01, 0.12, grad) * membrane;
	let blackwater = vec3f(0.006, 0.012, 0.014);
	let retention = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.86, 0.42, 0.12);
	let indigo = vec3f(0.486, 0.424, 1.0);
	// Metabolic breathing — the whole tissue pulses with the global breath so the
	// field reads as ALIVE, not a static print. pulse is 0..1 (params.pulse).
	let breath = 0.72 + 0.55 * params.pulse;
	var color = blackwater * (0.30 + density * 0.10);
	// Oxygen-lit plasma, pushed into HDR (>1) so the post-chain bloom makes it GLOW.
	color = color + mix(amber, retention, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.34 * breath;
	// Bright enzymatic edge — this is the "wet membrane" rim light; HDR for bloom flare.
	color = color + vec3f(0.91, 1.0, 0.72) * edge * (0.85 + 0.5 * params.pulse);
	// Indigo transaction-time seams shimmer with the breath.
	color = color + indigo * seam * (0.55 + 0.35 * params.pulse);
	let vignette = smoothstep(0.98, 0.12, distance(in.uv, vec2f(0.5)));
	return vec4f(color * (0.55 + 0.45 * vignette) * params.brightness * portrait_field_dim(), 1.0);
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
	cellBuffer: GPUBuffer;
	ringBuffer: GPUBuffer;
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

export class TimelinePass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: TimelineScene | null = null;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private splatBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private splatPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private cellPipeline: GPURenderPipeline | null = null;
	private ringPipeline: GPURenderPipeline | null = null;
	private cellCount = 0;
	private ringCount = 0;
	private selectedId: string | null = null;
	private cellGeometry: { cell: TimelineCell; x: number; y: number; r: number }[] = [];
	private ringGeometry: { ring: TimelineRing; r: number }[] = [];

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as TimelineScene;
		this.buildGeometry();
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.splatPipeline || !this.engine.paramsBuffer) return;
		const splatModule = createDiagnosedShaderModule(device, 'timeline-growth-rings-splat-wgsl', SPLAT_WGSL);
		const blurModule = createDiagnosedShaderModule(device, 'timeline-growth-rings-blur-wgsl', BLUR_WGSL);
		const membraneModule = createDiagnosedShaderModule(device, 'timeline-growth-rings-membrane-wgsl', MEMBRANE_WGSL);
		this.splatBindLayout = device.createBindGroupLayout({
			label: 'timeline-growth-rings-splat-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'timeline-growth-rings-blur-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'timeline-growth-rings-membrane-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		const splatLayout = device.createPipelineLayout({ label: 'timeline-growth-rings-splat-layout', bindGroupLayouts: [this.splatBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'timeline-growth-rings-blur-layout', bindGroupLayouts: [this.blurBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'timeline-growth-rings-membrane-layout', bindGroupLayouts: [this.membraneBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
		this.splatPipeline = device.createRenderPipeline({
			label: 'timeline-field-additive-splat',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_splat' },
			fragment: { module: splatModule, entryPoint: 'fs_splat', targets: [{ format: FIELD_FORMAT, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
			primitive: { topology: 'triangle-list' }
		});
		this.blurPipeline = device.createRenderPipeline({
			label: 'timeline-field-blur-render-pass',
			layout: blurLayout,
			vertex: { module: blurModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] },
			primitive: { topology: 'triangle-list' }
		});
		const additive = { color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }, alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation } };
		this.membranePipeline = device.createRenderPipeline({
			label: 'timeline-bitemporal-membrane',
			layout: membraneLayout,
			vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' },
			fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
		this.ringPipeline = device.createRenderPipeline({
			label: 'timeline-valid-time-rings',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_ring' },
			fragment: { module: splatModule, entryPoint: 'fs_ring', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-strip' }
		});
		this.cellPipeline = device.createRenderPipeline({
			label: 'timeline-memory-cells',
			layout: splatLayout,
			vertex: { module: splatModule, entryPoint: 'vs_cell' },
			fragment: { module: splatModule, entryPoint: 'fs_cell', targets: [{ format: this.engine.sceneFormat, blend: additive }] },
			primitive: { topology: 'triangle-list' }
		});
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.splatBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let cellBuffer = this.resources?.cellBuffer;
		let ringBuffer = this.resources?.ringBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		if (!cellBuffer) cellBuffer = device.createBuffer({ label: 'timeline-cells', size: MAX_CELLS * CELL_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!ringBuffer) ringBuffer = device.createBuffer({ label: 'timeline-rings', size: MAX_RINGS * RING_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'timeline-blur-h-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'timeline-blur-v-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!needsTextures && this.resources) return;
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'timeline-field-a-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'timeline-field-b-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const splatBindGroup = device.createBindGroup({
			label: 'timeline-growth-rings-splat-bind',
			layout: this.splatBindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: cellBuffer } },
				{ binding: 2, resource: { buffer: ringBuffer } }
			]
		});
		const blurHBindGroup = device.createBindGroup({ label: 'timeline-field-blur-h-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldAView }, { binding: 2, resource: { buffer: blurHBuffer } }] });
		const blurVBindGroup = device.createBindGroup({ label: 'timeline-field-blur-v-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldBView }, { binding: 2, resource: { buffer: blurVBuffer } }] });
		const membraneBindGroup = device.createBindGroup({ label: 'timeline-membrane-bind', layout: this.membraneBindLayout, entries: [{ binding: 0, resource: { buffer: this.engine.paramsBuffer } }, { binding: 3, resource: this.sampler }, { binding: 4, resource: fieldAView }] });
		this.resources = { cellBuffer, ringBuffer, blurHBuffer, blurVBuffer, splatBindGroup, blurHBindGroup, blurVBindGroup, membraneBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
	}

	private buildGeometry(): void {
		const cells = this.scene?.cells ?? [];
		this.cellGeometry = cells.slice(0, MAX_CELLS).map((cell) => ({ cell, x: Math.cos(cell.angle) * cell.radius, y: Math.sin(cell.angle) * cell.radius, r: 0.018 + cell.retention * 0.016 }));
		this.ringGeometry = (this.scene?.rings ?? []).slice(0, MAX_RINGS).map((ring) => ({ ring, r: ring.radius }));
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources) return;
		const cellData = new Float32Array(MAX_CELLS * CELL_FLOATS);
		this.cellCount = Math.min(MAX_CELLS, this.cellGeometry.length);
		const maxDay = Math.max(1, this.ringGeometry.length - 1);
		for (let i = 0; i < this.cellCount; i++) {
			const g = this.cellGeometry[i];
			const c = g.cell;
			const selected = this.selectedId === c.id || this.selectedId === c.memoryId ? 1 : 0;
			cellData.set([g.x, g.y, g.r, c.radius, c.retention, c.rewritten ? 1 : 0, c.suppressed ? 1 : 0, this.scene?.raw.audits[c.memoryId]?.length ?? 0, c.dayIndex / maxDay, Date.parse(c.transactionAt || c.validFrom || '') / 8.64e13 || 0, c.dayIndex, i, selected, 0, 0, 0], i * CELL_FLOATS);
		}
		this.ringCount = Math.min(MAX_RINGS, this.ringGeometry.length);
		const ringData = new Float32Array(MAX_RINGS * RING_FLOATS);
		const maxCount = Math.max(1, this.scene?.scalars.maxDayCount ?? 1);
		for (let i = 0; i < this.ringCount; i++) {
			const g = this.ringGeometry[i];
			const r = g.ring;
			const selected = this.selectedId === r.id || this.selectedId === r.date ? 1 : 0;
			ringData.set([g.r, r.count / maxCount, r.retention, r.index, r.updatedCount, r.suppressedCount, i / Math.max(1, this.ringCount), selected, r.memoryIndices.length, i, 0, 0], i * RING_FLOATS);
		}
		this.engine.params[2] = this.cellCount;
		this.engine.params[3] = this.ringCount;
		device.queue.writeBuffer(this.resources.cellBuffer, 0, cellData);
		device.queue.writeBuffer(this.resources.ringBuffer, 0, ringData);
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.splatPipeline || !this.blurPipeline) return;
		this.ensureResources(device);
		const res = this.resources;
		const splat = encoder.beginRenderPass({ label: 'timeline-field-splat-pass', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		splat.setPipeline(this.splatPipeline);
		splat.setBindGroup(0, res.splatBindGroup);
		splat.draw(6, this.cellCount);
		splat.end();
		const blurH = encoder.beginRenderPass({ label: 'timeline-field-blur-h-pass', colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();
		const blurV = encoder.beginRenderPass({ label: 'timeline-field-blur-v-pass', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.ringPipeline || !this.cellPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		if (this.ringCount > 0) {
			pass.setPipeline(this.ringPipeline);
			pass.setBindGroup(0, this.resources.splatBindGroup);
			pass.draw(192, this.ringCount);
		}
		if (this.cellCount > 0) {
			pass.setPipeline(this.cellPipeline);
			pass.draw(6, this.cellCount);
		}
	}

	/**
	 * CPU mirror of the WGSL orbit()/ring_spin() so pickAt tests the cell/ring at
	 * its CURRENTLY-ANIMATED position, not its static layout position. Without
	 * this, the field visibly rotates but clicks land where cells USED to be
	 * (the visual and the hitbox drift apart). params.time == engine.params[10].
	 */
	private ringSpin(dayPhase: number): number {
		const time = this.engine.params[10] || 0;
		const speed = 0.045 + dayPhase * 0.1;
		return time * speed;
	}

	private orbitCpu(bx: number, by: number, phase: number, dayPhase: number): { x: number; y: number } {
		const radius = Math.hypot(bx, by);
		if (radius < 0.0001) return { x: bx, y: by };
		const time = this.engine.params[10] || 0;
		const ang0 = Math.atan2(by, bx);
		const ang = ang0 + this.ringSpin(dayPhase) + Math.sin(time * 0.6 + phase * 6.283) * 0.02;
		const rr = radius * (1.0 + 0.016 * Math.sin(time * 1.1 + phase * 6.283));
		return { x: Math.cos(ang) * rr, y: Math.sin(ang) * rr };
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		const maxDay = Math.max(1, this.ringGeometry.length - 1);
		let best: RoutePick | null = null;
		let bestDist = Infinity;
		for (let i = 0; i < this.cellGeometry.length; i++) {
			const g = this.cellGeometry[i];
			// Match vs_cell: base=(x,y), phase=cellIndex i (time_meta.w),
			// dayPhase=dayIndex/maxDay (time_meta.x).
			const dayPhase = g.cell.dayIndex / maxDay;
			const p = this.orbitCpu(g.x, g.y, i, dayPhase);
			const d = Math.hypot(ndcX - p.x, ndcY - p.y);
			if (d <= Math.max(0.045, g.r * 1.8) && d < bestDist) {
				best = { id: g.cell.id, kind: 'timeline-cell', index: i, payload: g.cell };
				bestDist = d;
			}
		}
		if (best) {
			this.selectedId = best.id;
			return best;
		}
		// Rings rotate too, but a ring is a full circle — its radius is
		// rotation-invariant, so the radial hit test still holds. Apply only the
		// radial breathe (rr factor) at the ring's normalized phase for accuracy.
		const dist = Math.hypot(ndcX, ndcY);
		const time = this.engine.params[10] || 0;
		for (let i = 0; i < this.ringGeometry.length; i++) {
			const g = this.ringGeometry[i];
			const dayPhase = maxDay > 0 ? i / maxDay : 0;
			const rr = g.r * (1.0 + 0.016 * Math.sin(time * 1.1 + dayPhase * 6.283));
			if (Math.abs(dist - rr) <= 0.03) {
				this.selectedId = g.ring.id;
				return { id: g.ring.id, kind: 'timeline-ring', index: i, payload: g.ring };
			}
		}
		return null;
	}

	dispose(): void {
		this.resources?.cellBuffer.destroy();
		this.resources?.ringBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}
}

function createDiagnosedShaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
	device.pushErrorScope('validation');
	const module = device.createShaderModule({ label, code });
	void module.getCompilationInfo().then((info) => {
		for (const message of info.messages) console.error(`[observatory] ${label} WGSL ${message.type} ${message.lineNum}:${message.linePos} ${message.message}`);
	});
	void device.popErrorScope().then((error) => {
		if (error) console.error(`[observatory] ${label} shader module validation: ${error.message}`);
	});
	return module;
}

export function createTimelinePasses(engine: ObservatoryEngine, scene: RouteSceneModel): TimelinePass[] {
	void rgb01(MEDIUM.blackwater);
	void rgb01(RETENTION.healthy);
	void rgb01(RETENTION.luciferin);
	return [new TimelinePass(engine, scene)];
}

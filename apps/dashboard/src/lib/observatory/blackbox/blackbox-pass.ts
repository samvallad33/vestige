import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { BlackboxScene, BlackboxTraceImpulse } from './blackbox-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const MAX_EVENTS = 512;
const EVENT_FLOATS = 12;
const RECEIPT_FLOATS = 8;
const MAX_RECEIPTS = 128;
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

struct TraceEventCell {
	// x order 0..1, y lane 0..6, z confidence, w event kind code
	order_lane_conf_kind: vec4f,
	// x frame start, y visible gate, z selected, w receipt flag
	timing_flags: vec4f,
	// x retrieved ids count, y suppress/write/veto strength, z run duration fraction, w spare
	metric: vec4f,
};

struct ReceiptBead {
	// xy NDC, z intensity, w event index
	pos_energy: vec4f,
	// x node-count, y receipt ordinal, zw spare
	beat: vec4f,
};
`;

const TRACE_WGSL = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> trace_events: array<TraceEventCell>;
@group(0) @binding(2) var<storage, read> receipt_beads: array<ReceiptBead>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

fn lane_y(lane: f32) -> f32 {
	return mix(0.66, -0.62, lane / 6.0);
}

fn event_x(order: f32) -> f32 {
	return mix(-0.84, 0.84, clamp(order, 0.0, 1.0));
}

fn lane_color(kind: f32, lane: f32, conf: f32) -> vec3f {
	let luciferin = vec3f(0.66, 1.0, 0.37);
	let cyan = vec3f(0.08, 0.78, 0.92);
	let green = vec3f(0.12, 0.95, 0.56);
	let scarlet = vec3f(1.0, 0.18, 0.12);
	let amber = vec3f(1.0, 0.64, 0.06);
	let violet = vec3f(0.45, 0.42, 1.0);
	let bone = vec3f(0.88, 1.0, 0.70);
	var col = cyan;
	if (kind < 0.5) { col = bone; }
	else if (kind < 1.5) { col = green; }
	else if (kind < 2.5) { col = scarlet; }
	else if (kind < 3.5) { col = luciferin; }
	else if (kind < 4.5) { col = amber; }
	else if (kind < 5.5) { col = scarlet; }
	else { col = violet; }
	return mix(col * 0.62, col, clamp(conf, 0.0, 1.0)) * (0.88 + lane * 0.025);
}

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) local_uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) color_energy: vec4f,
};

@vertex
fn vs_impulse(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let cell = trace_events[ii];
	let corner = QUAD[vi];
	let order = cell.order_lane_conf_kind.x;
	let lane = cell.order_lane_conf_kind.y;
	let conf = cell.order_lane_conf_kind.z;
	let kind = cell.order_lane_conf_kind.w;
	let visible = cell.timing_flags.y;
	let selected = cell.timing_flags.z;
	let t = params.frame - cell.timing_flags.x;
	let pulse = smoothstep(0.0, 18.0, t) * (1.0 - smoothstep(92.0, 180.0, t));
	let center = vec2f(event_x(order), lane_y(lane));
	let radius = vec2f(0.028 + 0.026 * conf + 0.012 * selected, 0.026 + 0.018 * conf + 0.010 * pulse);
	var out: VSOut;
	out.clip = vec4f(center + corner * radius, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(kind, lane, selected, visible);
	out.color_energy = vec4f(lane_color(kind, lane, conf), visible * (0.35 + conf * 0.75 + pulse * 0.38));
	return out;
}

@fragment
fn fs_impulse(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.local_uv);
	if (d > 1.0 || frag.misc.w < 0.5) { discard; }
	let core = exp(-d * d * 3.4) * frag.color_energy.a;
	let ring = smoothstep(0.88, 0.62, abs(d - 0.64)) * (0.18 + frag.misc.z * 0.62);
	return vec4f(frag.color_energy.rgb * (core + ring), 1.0);
}

@vertex
fn vs_lane(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let corner = QUAD[vi];
	let lane = f32(ii);
	let center = vec2f(0.0, lane_y(lane));
	let size = vec2f(0.93, 0.012);
	var out: VSOut;
	out.clip = vec4f(center + corner * size, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(0.0, lane, 0.0, 1.0);
	out.color_energy = vec4f(lane_color(lane, lane, 0.4), 0.12 + 0.03 * params.pulse);
	return out;
}

@fragment
fn fs_lane(frag: VSOut) -> @location(0) vec4f {
	let fade = smoothstep(1.0, 0.08, abs(frag.local_uv.x)) * smoothstep(1.0, 0.0, abs(frag.local_uv.y));
	return vec4f(frag.color_energy.rgb * frag.color_energy.a * fade, 1.0);
}

@vertex
fn vs_receipt(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let bead = receipt_beads[ii];
	let corner = QUAD[vi];
	let event_i = bead.pos_energy.w;
	let linked = trace_events[u32(max(0.0, event_i))];
	let center = vec2f(event_x(linked.order_lane_conf_kind.x), lane_y(linked.order_lane_conf_kind.y) - 0.05 - 0.012 * bead.beat.y);
	let radius = 0.017 + 0.005 * min(5.0, bead.beat.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * radius, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(0.0, linked.order_lane_conf_kind.y, 0.0, linked.timing_flags.y);
	out.color_energy = vec4f(vec3f(0.90, 1.0, 0.70), bead.pos_energy.z * linked.timing_flags.y);
	return out;
}

@fragment
fn fs_receipt(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.local_uv);
	if (d > 1.0 || frag.misc.w < 0.5) { discard; }
	let bead = smoothstep(1.0, 0.0, d) + smoothstep(0.70, 0.58, abs(d - 0.64));
	return vec4f(frag.color_energy.rgb * frag.color_energy.a * bead, 1.0);
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

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	let p = QUAD[vi];
	var out: VSOut;
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_membrane(frag: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, frag.uv);
	let density = clamp(f.r, 0.0, 4.0);
	let recall = clamp(f.g, 0.0, 4.0);
	let immune = clamp(f.b, 0.0, 4.0);
	let write_glow = clamp(f.a, 0.0, 4.0);
	let centerline = smoothstep(0.44, 0.02, abs(frag.uv.y - 0.5));
	let blackwater = vec3f(0.006, 0.014, 0.016);
	var color = blackwater * (0.24 + density * 0.11);
	color = color + vec3f(0.62, 1.0, 0.35) * recall * 0.10;
	color = color + vec3f(1.0, 0.18, 0.12) * immune * 0.16;
	color = color + vec3f(0.90, 1.0, 0.70) * write_glow * 0.10;
	color = color + vec3f(0.08, 0.70, 0.80) * centerline * (0.03 + 0.02 * params.pulse);
	let vignette = smoothstep(0.92, 0.22, distance(frag.uv, vec2f(0.5)));
	return vec4f(color * (0.45 + 0.55 * vignette) * params.brightness, 1.0);
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
	let p = QUAD[vi];
	var out: VSOut;
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}
@fragment
fn fs_blur(frag: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let step = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, frag.uv - step * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv - step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + step * 2.0, 0.0) * 0.06136;
	return acc;
}
`;

type GpuResources = {
	eventBuffer: GPUBuffer;
	receiptBuffer: GPUBuffer;
	blurHBuffer: GPUBuffer;
	blurVBuffer: GPUBuffer;
	traceBindGroup: GPUBindGroup;
	membraneBindGroup: GPUBindGroup;
	blurHBindGroup: GPUBindGroup;
	blurVBindGroup: GPUBindGroup;
	fieldA: GPUTexture;
	fieldB: GPUTexture;
	fieldAView: GPUTextureView;
	fieldBView: GPUTextureView;
	fieldSize: [number, number];
};

function eventKindCode(type: BlackboxTraceImpulse['type']): number {
	switch (type) {
		case 'mcp.call': return 0;
		case 'memory.retrieve': return 1;
		case 'memory.suppress': return 2;
		case 'memory.write': return 3;
		case 'sanhedrin.veto': return 4;
		case 'contradiction.detected': return 5;
		case 'dream.patch': return 6;
	}
}

function laneCode(lane: BlackboxTraceImpulse['lane']): number {
	return ['tool', 'retrieve', 'suppress', 'write', 'veto', 'contradiction', 'dream'].indexOf(lane);
}

export class BlackboxPass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: BlackboxScene | null = null;
	private resources: GpuResources | null = null;
	private sampler: GPUSampler | null = null;
	private traceBindLayout: GPUBindGroupLayout | null = null;
	private membraneBindLayout: GPUBindGroupLayout | null = null;
	private blurBindLayout: GPUBindGroupLayout | null = null;
	private impulsePipeline: GPURenderPipeline | null = null;
	private lanePipeline: GPURenderPipeline | null = null;
	private receiptPipeline: GPURenderPipeline | null = null;
	private blurPipeline: GPURenderPipeline | null = null;
	private membranePipeline: GPURenderPipeline | null = null;
	private eventCount = 0;
	private receiptCount = 0;
	private hitRects: { event: BlackboxTraceImpulse; x: number; y: number; w: number; h: number }[] = [];

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as BlackboxScene;
		this.buildHitRects();
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.uploadBuffers(device);
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.impulsePipeline || !this.engine.paramsBuffer) return;
		const traceModule = device.createShaderModule({ label: 'blackbox-trace-wgsl', code: TRACE_WGSL });
		const membraneModule = device.createShaderModule({ label: 'blackbox-membrane-wgsl', code: MEMBRANE_WGSL });
		const blurModule = device.createShaderModule({ label: 'blackbox-blur-wgsl', code: BLUR_WGSL });
		this.traceBindLayout = device.createBindGroupLayout({
			label: 'blackbox-trace-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }
			]
		});
		this.membraneBindLayout = device.createBindGroupLayout({
			label: 'blackbox-membrane-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } }
			]
		});
		this.blurBindLayout = device.createBindGroupLayout({
			label: 'blackbox-blur-bind-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const traceLayout = device.createPipelineLayout({ label: 'blackbox-trace-layout', bindGroupLayouts: [this.traceBindLayout] });
		const membraneLayout = device.createPipelineLayout({ label: 'blackbox-membrane-layout', bindGroupLayouts: [this.membraneBindLayout] });
		const blurLayout = device.createPipelineLayout({ label: 'blackbox-blur-layout', bindGroupLayouts: [this.blurBindLayout] });
		this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
		const additive = { color: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation }, alpha: { srcFactor: 'one' as GPUBlendFactor, dstFactor: 'one' as GPUBlendFactor, operation: 'add' as GPUBlendOperation } };
		this.impulsePipeline = device.createRenderPipeline({ label: 'blackbox-impulses', layout: traceLayout, vertex: { module: traceModule, entryPoint: 'vs_impulse' }, fragment: { module: traceModule, entryPoint: 'fs_impulse', targets: [{ format: FIELD_FORMAT, blend: additive }] }, primitive: { topology: 'triangle-list' } });
		this.lanePipeline = device.createRenderPipeline({ label: 'blackbox-lanes', layout: traceLayout, vertex: { module: traceModule, entryPoint: 'vs_lane' }, fragment: { module: traceModule, entryPoint: 'fs_lane', targets: [{ format: FIELD_FORMAT, blend: additive }] }, primitive: { topology: 'triangle-list' } });
		this.receiptPipeline = device.createRenderPipeline({ label: 'blackbox-receipt-beads', layout: traceLayout, vertex: { module: traceModule, entryPoint: 'vs_receipt' }, fragment: { module: traceModule, entryPoint: 'fs_receipt', targets: [{ format: FIELD_FORMAT, blend: additive }] }, primitive: { topology: 'triangle-list' } });
		this.blurPipeline = device.createRenderPipeline({ label: 'blackbox-field-blur', layout: blurLayout, vertex: { module: blurModule, entryPoint: 'vs_fullscreen' }, fragment: { module: blurModule, entryPoint: 'fs_blur', targets: [{ format: FIELD_FORMAT }] }, primitive: { topology: 'triangle-list' } });
		this.membranePipeline = device.createRenderPipeline({ label: 'blackbox-field-membrane', layout: membraneLayout, vertex: { module: membraneModule, entryPoint: 'vs_fullscreen' }, fragment: { module: membraneModule, entryPoint: 'fs_membrane', targets: [{ format: this.engine.sceneFormat, blend: additive }] }, primitive: { topology: 'triangle-list' } });
	}

	private ensureResources(device: GPUDevice): void {
		if (!this.traceBindLayout || !this.blurBindLayout || !this.membraneBindLayout || !this.engine.paramsBuffer || !this.sampler) return;
		const w = Math.max(16, Math.floor((this.engine.params[6] || 1280) / 2));
		const h = Math.max(16, Math.floor((this.engine.params[7] || 720) / 2));
		const needsTextures = !this.resources || this.resources.fieldSize[0] !== w || this.resources.fieldSize[1] !== h;
		let eventBuffer = this.resources?.eventBuffer;
		let receiptBuffer = this.resources?.receiptBuffer;
		let blurHBuffer = this.resources?.blurHBuffer;
		let blurVBuffer = this.resources?.blurVBuffer;
		if (!eventBuffer) eventBuffer = device.createBuffer({ label: 'blackbox-event-cells', size: MAX_EVENTS * EVENT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!receiptBuffer) receiptBuffer = device.createBuffer({ label: 'blackbox-receipt-beads', size: MAX_RECEIPTS * RECEIPT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
		if (!blurHBuffer) {
			blurHBuffer = device.createBuffer({ label: 'blackbox-blur-h-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurHBuffer, 0, new Float32Array([1, 0, 0, 0]));
		}
		if (!blurVBuffer) {
			blurVBuffer = device.createBuffer({ label: 'blackbox-blur-v-dir', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
			device.queue.writeBuffer(blurVBuffer, 0, new Float32Array([0, 1, 0, 0]));
		}
		if (!needsTextures && this.resources) return;
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
		const fieldA = device.createTexture({ label: 'blackbox-field-a-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldB = device.createTexture({ label: 'blackbox-field-b-rgba16float', size: [w, h], format: FIELD_FORMAT, usage });
		const fieldAView = fieldA.createView();
		const fieldBView = fieldB.createView();
		const traceBindGroup = device.createBindGroup({ label: 'blackbox-trace-bind', layout: this.traceBindLayout, entries: [{ binding: 0, resource: { buffer: this.engine.paramsBuffer } }, { binding: 1, resource: { buffer: eventBuffer } }, { binding: 2, resource: { buffer: receiptBuffer } }] });
		const membraneBindGroup = device.createBindGroup({ label: 'blackbox-membrane-bind', layout: this.membraneBindLayout, entries: [{ binding: 0, resource: { buffer: this.engine.paramsBuffer } }, { binding: 3, resource: this.sampler }, { binding: 4, resource: fieldAView }] });
		const blurHBindGroup = device.createBindGroup({ label: 'blackbox-blur-h-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldAView }, { binding: 2, resource: { buffer: blurHBuffer } }] });
		const blurVBindGroup = device.createBindGroup({ label: 'blackbox-blur-v-bind', layout: this.blurBindLayout, entries: [{ binding: 0, resource: this.sampler }, { binding: 1, resource: fieldBView }, { binding: 2, resource: { buffer: blurVBuffer } }] });
		this.resources = { eventBuffer, receiptBuffer, blurHBuffer, blurVBuffer, traceBindGroup, membraneBindGroup, blurHBindGroup, blurVBindGroup, fieldA, fieldB, fieldAView, fieldBView, fieldSize: [w, h] };
	}

	private uploadBuffers(device: GPUDevice): void {
		if (!this.resources || !this.scene) return;
		const visible = Math.min(MAX_EVENTS, this.scene.visibleEventCount || this.scene.traceEvents.length);
		const total = Math.max(1, this.scene.traceEvents.length - 1);
		const events = new Float32Array(MAX_EVENTS * EVENT_FLOATS);
		this.eventCount = Math.min(MAX_EVENTS, this.scene.traceEvents.length);
		for (let i = 0; i < this.eventCount; i++) {
			const ev = this.scene.traceEvents[i];
			const visibleGate = i < visible ? 1 : 0;
			const selected = i === this.scene.selectedIndex ? 1 : 0;
			const lane = laneCode(ev.lane);
			const kind = eventKindCode(ev.type);
			const memoryCount = ev.memoryIds.length;
			const strength = ev.type === 'memory.suppress' || ev.type === 'sanhedrin.veto' || ev.type === 'contradiction.detected' ? 1 : ev.type === 'memory.write' ? 0.75 : 0.35;
			events.set([i / total, lane, ev.confidence, kind, i * 34 + 18, visibleGate, selected, 0, memoryCount, strength, total ? i / total : 0, 0], i * EVENT_FLOATS);
		}
		device.queue.writeBuffer(this.resources.eventBuffer, 0, events);
		const receipts = new Float32Array(MAX_RECEIPTS * RECEIPT_FLOATS);
		this.receiptCount = Math.min(MAX_RECEIPTS, this.scene.receipts.length);
		for (let i = 0; i < this.receiptCount; i++) {
			const linkedEventIndex = Math.min(Math.max(0, visible - 1), this.eventCount - 1);
			receipts.set([0, 0, 0.65, linkedEventIndex, this.scene.receipts[i].nodeIndices.length, i, 0, 0], i * RECEIPT_FLOATS);
		}
		device.queue.writeBuffer(this.resources.receiptBuffer, 0, receipts);
		this.engine.params[4] = this.eventCount;
	}

	private buildHitRects(): void {
		const events = this.scene?.traceEvents ?? [];
		const total = Math.max(1, events.length - 1);
		this.hitRects = events.map((ev, i) => ({ event: ev, x: -0.84 + 1.68 * (i / total), y: 0.66 - 1.28 * (laneCode(ev.lane) / 6), w: 0.055, h: 0.065 }));
	}

	compute(encoder: GPUCommandEncoder): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources || !this.impulsePipeline || !this.lanePipeline || !this.receiptPipeline || !this.blurPipeline) return;
		this.ensureResources(device);
		const res = this.resources;
		const splat = encoder.beginRenderPass({ label: 'blackbox-field-splat-pass', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		splat.setBindGroup(0, res.traceBindGroup);
		splat.setPipeline(this.lanePipeline);
		splat.draw(6, 7);
		if (this.eventCount > 0) {
			splat.setPipeline(this.impulsePipeline);
			splat.draw(6, this.eventCount);
		}
		if (this.receiptCount > 0 && this.eventCount > 0) {
			splat.setPipeline(this.receiptPipeline);
			splat.draw(6, this.receiptCount);
		}
		splat.end();
		const blurH = encoder.beginRenderPass({ label: 'blackbox-field-blur-h-pass', colorAttachments: [{ view: res.fieldBView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurH.setPipeline(this.blurPipeline);
		blurH.setBindGroup(0, res.blurHBindGroup);
		blurH.draw(6, 1);
		blurH.end();
		const blurV = encoder.beginRenderPass({ label: 'blackbox-field-blur-v-pass', colorAttachments: [{ view: res.fieldAView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
		blurV.setPipeline(this.blurPipeline);
		blurV.setBindGroup(0, res.blurVBindGroup);
		blurV.draw(6, 1);
		blurV.end();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.membranePipeline || !this.impulsePipeline || !this.lanePipeline || !this.receiptPipeline) return;
		pass.setPipeline(this.membranePipeline);
		pass.setBindGroup(0, this.resources.membraneBindGroup);
		pass.draw(6, 1);
		pass.setBindGroup(0, this.resources.traceBindGroup);
		pass.setPipeline(this.lanePipeline);
		pass.draw(6, 7);
		if (this.eventCount > 0) {
			pass.setPipeline(this.impulsePipeline);
			pass.draw(6, this.eventCount);
		}
		if (this.receiptCount > 0 && this.eventCount > 0) {
			pass.setPipeline(this.receiptPipeline);
			pass.draw(6, this.receiptCount);
		}
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		for (const rect of this.hitRects) {
			if (Math.abs(ndcX - rect.x) <= rect.w && Math.abs(ndcY - rect.y) <= rect.h) {
				return { id: rect.event.id, kind: 'trace-event', index: rect.event.index, payload: rect.event };
			}
		}
		return null;
	}

	dispose(): void {
		this.resources?.eventBuffer.destroy();
		this.resources?.receiptBuffer.destroy();
		this.resources?.blurHBuffer.destroy();
		this.resources?.blurVBuffer.destroy();
		this.resources?.fieldA.destroy();
		this.resources?.fieldB.destroy();
		this.resources = null;
	}
}

export function createBlackboxPasses(engine: ObservatoryEngine, scene: RouteSceneModel): BlackboxPass[] {
	void rgb01(RETENTION.healthy);
	void rgb01(IMMUNE.veto);
	void rgb01(CAUSAL.forward);
	return [new BlackboxPass(engine, scene)];
}

/**
 * Fossil Light — Chrono Shuttle
 *
 * The Shuttle is the first visible instrument for the Observatory's signed
 * time axis.  It renders only facts the graph payload already carries:
 * creation instants and last-accessed instants.  The CPU owns interaction and
 * accessibility; this pass turns that exact state into a quiet phosphor rail
 * without a per-frame allocation or a decorative simulation.
 *
 * Three draws, all static after graph load:
 *   1. the mineral rail / past-to-future tide,
 *   2. instanced lifecycle dwell marks,
 *   3. the analytic beam at the active instant.
 */

import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import type { ObservatoryNode } from '$lib/observatory/types';

const MAX_DWELLS = 512;
const DWELL_FLOATS = 4;

/** Current control values; 16-byte aligned for a WebGPU uniform buffer. */
type ShuttleState = {
	/** Normalized [0, 1] active time across earliest real event → horizon. */
	scrub: number;
	/** Signed day offset from real NOW; used only for a subtle mode treatment. */
	days: number;
	/** Event density, normalized for the rail's phosphor dwell height. */
	density: number;
	/** 1 while the accessible range is being directly manipulated. */
	active: number;
};

const SHUTTLE_WGSL = /* wgsl */ `
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
	cursor_x: f32,
	cursor_y: f32,
	cursor_vx: f32,
	cursor_vy: f32,
};

struct ShuttleState {
	scrub: f32,
	days: f32,
	density: f32,
	dragging: f32,
};

// x normalized timeline position; y kind (0 birth / 1 review); z retention;
// w suppression marker.  One vec4 per real lifecycle event, fixed after load.
struct Dwell { data: vec4f };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dwells: array<Dwell>;
@group(0) @binding(2) var<uniform> shuttle: ShuttleState;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

const RAIL_Y = -0.685;
const RAIL_LEFT = -0.835;
const RAIL_RIGHT = 0.835;

fn rail_x(t: f32) -> f32 { return mix(RAIL_LEFT, RAIL_RIGHT, clamp(t, 0.0, 1.0)); }

// A compact erf approximation makes the beam's edges physically continuous
// rather than a CSS-style blur.  It is evaluated only in fragments of a small
// quad and has no texture/noise dependency.
fn erf_approx(x: f32) -> f32 {
	let s = select(-1.0, 1.0, x >= 0.0);
	let a = abs(x);
	let t = 1.0 / (1.0 + 0.3275911 * a);
	let p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t);
	return s * (1.0 - p * exp(-a * a));
}

struct RailOut {
	@builtin(position) clip: vec4f,
	@location(0) local: vec2f,
};

@vertex
fn vs_rail(@builtin(vertex_index) vi: u32) -> RailOut {
	let q = QUAD[vi];
	// Pixel floor: NDC fractions collapse below a device pixel on narrow
	// viewports (0.022 of a 375px-wide phone is invisible). viewport_w/h ride
	// params lanes 6-7.
	let py = 2.0 / max(params.viewport_h, 1.0);
	var out: RailOut;
	out.clip = vec4f(mix(RAIL_LEFT, RAIL_RIGHT, q.x * 0.5 + 0.5), RAIL_Y + q.y * max(0.022, py * 5.0), 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_rail(in: RailOut) -> @location(0) vec4f {
	let t = in.local.x * 0.5 + 0.5;
	let past = vec3f(0.075, 0.104, 0.088);   // graphite jade: known history
	let now = vec3f(0.48, 0.58, 0.42);       // quiet chalk-lichen at NOW
	let future = vec3f(0.31, 0.19, 0.09);    // fossil amber: projected debt
	let base = select(mix(past, now, t / max(shuttle.scrub, 0.001)), mix(now, future, (t - shuttle.scrub) / max(1.0 - shuttle.scrub, 0.001)), t > shuttle.scrub);
	let midline = 1.0 - smoothstep(0.09, 0.72, abs(in.local.y));
	let tick = smoothstep(0.03, 0.0, abs(fract(t * 24.0) - 0.5));
	let nowRim = exp(-pow((t - shuttle.scrub) * 88.0, 2.0));
	let color = base * (0.40 + midline * 0.46) + vec3f(0.76, 0.82, 0.66) * nowRim * 0.17 + vec3f(0.36, 0.31, 0.20) * tick * 0.12;
	return vec4f(color, 0.82 * midline + tick * 0.12);
}

struct DwellOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) kind: f32,
	@location(2) @interpolate(flat) retention: f32,
	@location(3) @interpolate(flat) suppressed: f32,
	@location(4) @interpolate(flat) distance_to_scrub: f32,
};

@vertex
fn vs_dwell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DwellOut {
	let d = dwells[ii].data;
	let q = QUAD[vi];
	let density = clamp(shuttle.density, 0.0, 1.0);
	let height = (0.034 + d.z * 0.064) * (0.72 + density * 0.44);
	// Never thinner than ~1.6 device px, whatever the viewport width.
	let px = 2.0 / max(params.viewport_w, 1.0);
	let width = max(0.0017 + density * 0.0016, px * 1.6);
	let direction = select(-1.0, 1.0, d.y > 0.5);
	var out: DwellOut;
	out.clip = vec4f(rail_x(d.x) + q.x * width, RAIL_Y + direction * (0.008 + height * (q.y * 0.5 + 0.5)), 0.0, 1.0);
	out.uv = q;
	out.kind = d.y;
	out.retention = d.z;
	out.suppressed = d.w;
	out.distance_to_scrub = abs(d.x - shuttle.scrub);
	return out;
}

@fragment
fn fs_dwell(in: DwellOut) -> @location(0) vec4f {
	let core = 1.0 - smoothstep(0.18, 0.94, abs(in.uv.x));
	let near = exp(-pow(in.distance_to_scrub * 105.0, 2.0));
	let birth = vec3f(0.53, 0.72, 0.57);
	let review = mix(vec3f(0.48, 0.32, 0.15), vec3f(0.83, 0.80, 0.60), in.retention);
	let injury = vec3f(0.56, 0.20, 0.16);
	var color = select(birth, review, in.kind > 0.5);
	// Suppression is a PRESENT-DAY fact (suppression_count > 0). Only the
	// latest-access mark may honestly carry the injury tint — smearing it onto
	// the birth mark would claim the memory was suppressed at creation.
	color = mix(color, injury, in.suppressed * 0.72 * step(0.5, in.kind));
	// Dwell proximity produces the only noticeable glow: real event density,
	// never a permanently luminous UI element.
	color = color * (0.36 + in.retention * 0.42 + near * 0.62);
	return vec4f(color, core * (0.38 + near * 0.54));
}

struct HeadOut { @builtin(position) clip: vec4f, @location(0) local: vec2f };

@vertex
fn vs_head(@builtin(vertex_index) vi: u32) -> HeadOut {
	let q = QUAD[vi];
	let speed = min(1.0, abs(shuttle.days) / 28.0);
	let height = 0.095 + shuttle.dragging * 0.038 + speed * 0.022;
	let px = 2.0 / max(params.viewport_w, 1.0);
	var out: HeadOut;
	out.clip = vec4f(rail_x(shuttle.scrub) + q.x * max(0.021, px * 7.0), RAIL_Y + q.y * height, 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_head(in: HeadOut) -> @location(0) vec4f {
	let x = in.local.x * 2.55;
	let beam = (erf_approx(x + 1.45) - erf_approx(x - 1.45)) * 0.5;
	let center = exp(-x * x * 3.2);
	let line = smoothstep(0.96, 0.08, abs(in.local.y));
	let color = mix(vec3f(0.64, 0.49, 0.23), vec3f(0.84, 0.96, 0.72), step(0.0, shuttle.days));
	return vec4f(color * (0.22 + center * 0.86), beam * line * (0.46 + center * 0.48));
}
`;

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function timestamp(value: string | undefined): number | null {
	if (!value) return null;
	const parsed = Date.parse(value);
	return Number.isFinite(parsed) ? parsed : null;
}

type Resources = {
	dwellBuffer: GPUBuffer;
	stateBuffer: GPUBuffer;
	bindGroup: GPUBindGroup;
};

/** Raw-WebGPU phosphor rail backed by real memory lifecycle points. */
export class ChronoShuttlePass implements FramePass {
	private engine: ObservatoryEngine;
	private resources: Resources | null = null;
	private bindLayout: GPUBindGroupLayout | null = null;
	private railPipeline: GPURenderPipeline | null = null;
	private dwellPipeline: GPURenderPipeline | null = null;
	private headPipeline: GPURenderPipeline | null = null;
	private dwellCount = 0;
	private minMs = 0;
	private maxMs = 0;
	private state: ShuttleState = { scrub: 1, days: 0, density: 0, active: 0 };

	constructor(engine: ObservatoryEngine, nodes: readonly ObservatoryNode[]) {
		this.engine = engine;
		this.upload(nodes);
	}

	/**
	 * Update only the 16-byte control uniform while scrubbing.  Lifecycle marks
	 * never move until the graph itself changes, so dragging is allocation-free.
	 */
	setTimeline(days: number, active = false): void {
		const now = this.engine.wallNowMs;
		const range = Math.max(1, this.maxMs - this.minMs);
		this.state.scrub = clamp01((now + days * 86_400_000 - this.minMs) / range);
		this.state.days = Number.isFinite(days) ? days : 0;
		this.state.active = active ? 1 : 0;
		this.writeState();
		this.engine.requestRender();
	}

	targetFrameRate(): number {
		// The rail itself is quiet when untouched. Other active field passes may
		// demand 60; this simply never increases the engine's cost on its own.
		return this.state.active > 0 ? 60 : 12;
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.railPipeline || !this.dwellPipeline || !this.headPipeline) return;
		pass.setBindGroup(0, this.resources.bindGroup);
		pass.setPipeline(this.railPipeline);
		pass.draw(6);
		if (this.dwellCount > 0) {
			pass.setPipeline(this.dwellPipeline);
			pass.draw(6, this.dwellCount);
		}
		pass.setPipeline(this.headPipeline);
		pass.draw(6);
	}

	dispose(): void {
		this.resources?.dwellBuffer.destroy();
		this.resources?.stateBuffer.destroy();
		this.resources = null;
	}

	private upload(nodes: readonly ObservatoryNode[]): void {
		const allTimes = nodes.flatMap((node) => [timestamp(node.createdAt), timestamp(node.lastAccessed)]).filter((time): time is number => time !== null);
		const now = this.engine.wallNowMs;
		this.minMs = allTimes.length > 0 ? Math.min(...allTimes) : now - 86_400_000;
		// A one-year forward edge makes the existing projection portion visible.
		this.maxMs = Math.max(now + 365 * 86_400_000, this.minMs + 86_400_000);
		const range = this.maxMs - this.minMs;
		const events: Array<{ at: number; kind: number; retention: number; suppressed: number }> = [];
		for (const node of nodes) {
			const createdAt = timestamp(node.createdAt);
			const lastAccessed = timestamp(node.lastAccessed);
			const retention = clamp01(node.retention);
			if (createdAt !== null) events.push({ at: createdAt, kind: 0, retention, suppressed: node.suppressed ? 1 : 0 });
			if (lastAccessed !== null && lastAccessed !== createdAt) events.push({ at: lastAccessed, kind: 1, retention, suppressed: node.suppressed ? 1 : 0 });
		}
		events.sort((a, b) => a.at - b.at);
		// Deterministic stride cap: dense graphs still use one static bounded buffer.
		const stride = Math.max(1, Math.ceil(events.length / MAX_DWELLS));
		const sampled = events.filter((_, index) => index % stride === 0).slice(0, MAX_DWELLS);
		this.dwellCount = sampled.length;
		this.state = { scrub: clamp01((now - this.minMs) / range), days: 0, density: clamp01(sampled.length / 96), active: 0 };

		const device = this.engine.gpuDevice;
		if (!device || !this.engine.paramsBuffer) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		if (!this.resources) return;
		const data = new Float32Array(MAX_DWELLS * DWELL_FLOATS);
		sampled.forEach((event, index) => {
			data.set([clamp01((event.at - this.minMs) / range), event.kind, event.retention, event.suppressed], index * DWELL_FLOATS);
		});
		device.queue.writeBuffer(this.resources.dwellBuffer, 0, data);
		this.writeState();
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.railPipeline || !this.engine.paramsBuffer) return;
		const module = device.createShaderModule({ label: 'fossil-light-chrono-shuttle-wgsl', code: SHUTTLE_WGSL });
		this.bindLayout = device.createBindGroupLayout({
			label: 'fossil-light-chrono-shuttle-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const layout = device.createPipelineLayout({ label: 'fossil-light-chrono-shuttle-pipeline-layout', bindGroupLayouts: [this.bindLayout] });
		const blend: GPUBlendState = {
			color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
			alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
		};
		const pipeline = (label: string, vertex: string, fragment: string) => device.createRenderPipeline({
			label,
			layout,
			vertex: { module, entryPoint: vertex },
			fragment: { module, entryPoint: fragment, targets: [{ format: this.engine.sceneFormat, blend }] },
			primitive: { topology: 'triangle-list' }
		});
		this.railPipeline = pipeline('fossil-light-chrono-rail', 'vs_rail', 'fs_rail');
		this.dwellPipeline = pipeline('fossil-light-chrono-dwells', 'vs_dwell', 'fs_dwell');
		this.headPipeline = pipeline('fossil-light-chrono-head', 'vs_head', 'fs_head');
	}

	private ensureResources(device: GPUDevice): void {
		if (this.resources || !this.bindLayout || !this.engine.paramsBuffer) return;
		const dwellBuffer = device.createBuffer({
			label: 'fossil-light-chrono-dwell-events',
			size: MAX_DWELLS * DWELL_FLOATS * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		const stateBuffer = device.createBuffer({
			label: 'fossil-light-chrono-state',
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.resources = {
			dwellBuffer,
			stateBuffer,
			bindGroup: device.createBindGroup({
				label: 'fossil-light-chrono-bind-group',
				layout: this.bindLayout,
				entries: [
					{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
					{ binding: 1, resource: { buffer: dwellBuffer } },
					{ binding: 2, resource: { buffer: stateBuffer } }
				]
			})
		};
	}

	private writeState(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources) return;
		device.queue.writeBuffer(this.resources.stateBuffer, 0, new Float32Array([this.state.scrub, this.state.days, this.state.density, this.state.active]));
	}
}

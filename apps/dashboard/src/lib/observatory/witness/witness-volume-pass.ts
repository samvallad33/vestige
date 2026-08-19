/**
 * WitnessVolumePass
 *
 * A receipt-bound 3D evidence chamber. It deliberately borrows the launch
 * engine's reveal grammar (sealed core -> ingress trails -> spatial reveal),
 * but not its random particle swarm: every visible wafer and filament maps to
 * a selected receipt member or an ordered trace event.
 *
 * Design constraints:
 * - fixed, deterministic spatial layout; no force simulation or idle churn
 * - one instanced wafer draw + one bounded, screen-space ribbon draw
 * - no corpus edges; receipt membership is the only persistent connection
 * - no bloom blanket, purple haze, or per-edge allocation
 */

import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { rgb01 } from '$lib/observatory/cognitive-palette';
import type { RouteSceneModel } from '$lib/observatory/route-scene';
import type { WitnessScene, WitnessShard } from './witness-scene';

type RoutePick = { id: string; kind: string; index?: number; payload?: unknown };

const MAX_SHARDS = 64;
const MAX_FILAMENTS = 96;
const SHARD_FLOATS = 16;
const FILAMENT_FLOATS = 8;

const VOLUME_WGSL = /* wgsl */ `
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

struct WitnessState {
	playhead: f32,
	replay_start: f32,
	selected_index: f32,
	shard_count: f32,
};

struct Shard {
	// xyz = deterministic 3D location, w = wafer base scale
	position_size: vec4f,
	// x activation, y retention, z trace-time 0..1, w selected
	metrics: vec4f,
	// real status color; semantic, never decorative
	color: vec4f,
	// x role, y scar flag, z reveal order, w reserved
	flags: vec4f,
};

struct Filament {
	// x source shard index, y target shard index (only verified path neighbors)
	endpoints: vec4f,
	// x energy, y deterministic phase, z receipt-path flag, w reserved
	motion: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> shards: array<Shard>;
@group(0) @binding(2) var<storage, read> filaments: array<Filament>;
@group(0) @binding(3) var<uniform> witness: WitnessState;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

// A witness shard is a small, extruded ceramic specimen, not a flat UI card.
// The front face remains deliberately asymmetric; the two side faces make the
// receipt structure legible as a volume even when the chamber is completely
// still.  The depth is part of the actual perspective projection below.
const WAFER = array<vec3f, 18>(
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(-1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0), vec3f(-1.34, -1.42, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.34, 0.58, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.34, 0.58, -1.0), vec3f(0.66, -1.42, -1.0)
);

struct Projection { screen: vec2f, scale: f32 };

// This is a real perspective projection, not a 2D arrangement. Time is depth
// and activation/role form the stable chamber strata. The pointer only shifts
// the 3/4 view by a few degrees: an examination lens, never an auto-orbit.
fn cursor_lens() -> vec2f {
	if (abs(params.cursor_x) > 2.0 || abs(params.cursor_y) > 2.0) {
		return vec2f(0.0, 0.0);
	}
	return clamp(vec2f(params.cursor_x, params.cursor_y), vec2f(-1.0), vec2f(1.0));
}

fn project(world: vec3f) -> Projection {
	let lens = cursor_lens();
	let yaw = lens.x * 0.055;
	let c = cos(yaw);
	let s = sin(yaw);
	let view = vec3f(
		world.x * c - world.z * s,
		world.y + lens.y * 0.035,
		world.x * s + world.z * c
	);
	let depth = clamp(2.82 - view.z, 1.1, 5.4);
	let perspective = 1.0 / depth;
	return Projection(vec2f(view.x * 1.18 * perspective, view.y * 1.62 * perspective), perspective);
}

fn smooth01(value: f32) -> f32 {
	let t = clamp(value, 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

fn reveal_for(shard: Shard) -> f32 {
	let arrival = 14.0 + shard.flags.z * 68.0;
	let ingress = smooth01((params.frame - arrival) / 62.0);
	// The temporal slicer is a real trace cursor. Evidence that was not yet
	// available simply does not materialize.
	let slice = smoothstep(shard.metrics.z - 0.045, shard.metrics.z + 0.09, witness.playhead);
	return ingress * slice;
}

fn replay_age() -> f32 {
	if (witness.replay_start < 0.0) { return 9999.0; }
	var age = params.frame - witness.replay_start;
	if (age < 0.0) { age = age + 720.0; }
	return age;
}

// Quiet mineral palette: jade marks corroborated evidence; the traversal itself
// is fossil amber. There is intentionally no cyan/purple emissive wash.
fn core_color() -> vec3f { return vec3f(0.10, 0.17, 0.14); }

struct WaferOut {
	@builtin(position) clip: vec4f,
	@location(0) local: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) data: vec4f,
	@location(3) @interpolate(flat) face: f32,
};

@vertex
fn vs_wafer(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> WaferOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let sealed = vec3f(0.08, -0.02, 0.12);
	let selectedLift = select(vec3f(0.0), vec3f(0.0, 0.0, 0.34), shard.metrics.w > 0.5);
	let position = mix(sealed, shard.position_size.xyz + selectedLift, reveal);
	let q = WAFER[vi];
	let rotation = -0.15 + (shard.flags.z - 0.5) * 0.30 + (shard.flags.x - 1.5) * 0.075;
	let c = cos(rotation);
	let s = sin(rotation);
	let rq = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	let height = (0.112 + shard.metrics.y * 0.072) * shard.position_size.w;
	let width = height * (2.05 + shard.metrics.x * 0.82);
	let thickness = height * (0.46 + shard.metrics.y * 0.14);
	// The face offset is projected with the specimen, rather than pasted onto
	// the screen. It is a true low-poly wafer with parallax depth.
	let facePosition = position + vec3f(rq.x * width, rq.y * height, q.z * thickness);
	let projected = project(facePosition);
	var out: WaferOut;
	out.clip = vec4f(projected.screen, 0.0, 1.0);
	out.local = q.xy;
	out.color = shard.color.rgb;
	out.data = vec4f(shard.metrics.x, shard.metrics.y, shard.metrics.w, shard.flags.y);
	out.face = floor(f32(vi) / 6.0);
	return out;
}

@fragment
fn fs_wafer(frag: WaferOut) -> @location(0) vec4f {
	let edge = max(abs(frag.local.y), abs(frag.local.x) * 0.76 + frag.local.y * 0.12);
	if (frag.face < 0.5 && edge > 1.0) { discard; }
	let rim = smoothstep(0.72, 0.97, edge);
	let facet = smoothstep(-1.05, 1.05, frag.local.x * 0.78 - frag.local.y * 0.42);
	let scar = frag.data.w;
	let selected = frag.data.z;
	let carbon = vec3f(0.018, 0.031, 0.033);
	var body = mix(carbon, frag.color * (0.14 + facet * 0.18), 0.74);
	if (frag.face > 0.5) {
		let sideLight = select(0.24, 0.42, frag.face > 1.5);
		body = mix(carbon * 1.45, frag.color * sideLight, 0.66);
	} else {
		body = body + frag.color * rim * (0.16 + selected * 0.30);
	}
	if (scar > 0.5) {
		let fracture = smoothstep(0.10, 0.02, abs(frag.local.x + frag.local.y * 0.37));
		body = mix(body, vec3f(0.72, 0.12, 0.09), fracture * 0.76);
	}
	let alpha = select(0.72, 0.86 + selected * 0.14, frag.face < 0.5);
	return vec4f(body, alpha);
}

struct RibbonOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) energy: f32,
	@location(3) @interpolate(flat) selected: f32,
};

fn filament_point(index: f32) -> vec3f {
	if (index < -0.5) { return vec3f(0.08, -0.02, 0.12); }
	let shard = shards[u32(index)];
	return mix(vec3f(0.0, 0.0, 0.0), shard.position_size.xyz, reveal_for(shard));
}

fn filament_selected(index: f32) -> f32 {
	if (index < -0.5) { return 0.0; }
	return shards[u32(index)].metrics.w;
}

@vertex
fn vs_filament(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let fiber = filaments[ii];
	let start = project(filament_point(fiber.endpoints.x));
	let end = project(filament_point(fiber.endpoints.y));
	let delta = end.screen - start.screen;
	let distance = max(length(delta), 0.0001);
	let direction = delta / distance;
	let normal = vec2f(-direction.y, direction.x);
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	let width = 0.0017 + fiber.motion.x * 0.0025;
	let selected = max(filament_selected(fiber.endpoints.x), filament_selected(fiber.endpoints.y));
	var out: RibbonOut;
	out.clip = vec4f(mix(start.screen, end.screen, t) + normal * q.y * width, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = mix(core_color(), vec3f(0.46, 0.30, 0.15), fiber.motion.z);
	out.energy = fiber.motion.x;
	out.selected = selected;
	return out;
}

@fragment
fn fs_filament(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.24, abs(frag.uv.y));
	let age = replay_age();
	let travel = fract(age * 0.016 + frag.energy * 0.37);
	let wrapped = abs(fract(frag.uv.x - travel + 0.5) - 0.5);
	let pulse = exp(-wrapped * wrapped * 980.0) * select(0.0, 1.0, age < 176.0);
	let alpha = body * (0.105 + frag.selected * 0.30 + pulse * 0.72);
	return vec4f(frag.color * (0.32 + pulse * 0.74), alpha);
}

// A self-emitted arrival trail for each real receipt member. It only exists
// while the wafer is entering the chamber; it is not an always-on decorative
// network.
@vertex
fn vs_arrival_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let previous = max(0.0, reveal - 0.13);
	let a = project(mix(vec3f(0.0), shard.position_size.xyz, previous));
	let b = project(mix(vec3f(0.0), shard.position_size.xyz, reveal));
	let delta = b.screen - a.screen;
	let distance = max(length(delta), 0.0001);
	let normal = vec2f(-delta.y, delta.x) / distance;
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	var out: RibbonOut;
	out.clip = vec4f(mix(a.screen, b.screen, t) + normal * q.y * 0.006, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = shard.color.rgb;
	out.energy = reveal * (1.0 - smoothstep(0.92, 1.0, reveal));
	out.selected = shard.metrics.w;
	return out;
}

@fragment
fn fs_arrival_trail(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.18, abs(frag.uv.y));
	return vec4f(frag.color * (0.32 + frag.selected * 0.44), body * frag.energy * 0.58);
}

struct CoreOut { @builtin(position) clip: vec4f, @location(0) local: vec2f };

@vertex
fn vs_core(@builtin(vertex_index) vi: u32) -> CoreOut {
	let q = QUAD[vi];
	let lens = cursor_lens();
	let center = project(vec3f(0.08, -0.02, 0.12));
	let tilt = -0.14 + lens.x * 0.035;
	let c = cos(tilt);
	let s = sin(tilt);
	let rotated = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	var out: CoreOut;
	// The receipt is held inside a black archival spine. The slates form its
	// strata; this sealed volume replaces the generic graph's central node.
	out.clip = vec4f(center.screen + rotated * vec2f(0.265, 0.80) * (0.74 + center.scale), 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_core(frag: CoreOut) -> @location(0) vec4f {
	let diagonal = abs(frag.local.x) * 0.83 + frag.local.y * 0.10;
	if (max(abs(frag.local.y), diagonal) > 1.0) { discard; }
	let edge = smoothstep(0.79, 0.98, max(abs(frag.local.y), abs(diagonal)));
	let aperture = smoothstep(0.105, 0.022, abs(frag.local.x + frag.local.y * 0.08));
	let stratum = smoothstep(0.035, 0.004, abs(fract((frag.local.y + 1.0) * 2.9) - 0.5));
	let jade = vec3f(0.25, 0.43, 0.33);
	let amber = vec3f(0.43, 0.28, 0.14);
	var color = vec3f(0.008, 0.016, 0.016);
	color = color + vec3f(0.026, 0.051, 0.045) * (1.0 - abs(frag.local.x)) * 0.75;
	color = color + jade * edge * 0.24;
	color = color + jade * stratum * 0.11;
	color = color + amber * aperture * 0.34;
	return vec4f(color, 0.94);
}
`;

type GpuResources = {
	shardBuffer: GPUBuffer;
	filamentBuffer: GPUBuffer;
	stateBuffer: GPUBuffer;
	bindGroup: GPUBindGroup;
};

type HitTarget = { shard: WitnessShard; x: number; y: number; radius: number };

function clamp01(value: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function roleCode(shard: WitnessShard): number {
	return ['retrieved', 'path', 'mutation', 'suppressed'].indexOf(shard.role);
}

function colorFor(shard: WitnessShard): [number, number, number] {
	if (shard.suppressed) return rgb01('#ab5a51');
	if (shard.mutated) return rgb01('#c58a4a');
	if (shard.role === 'path') return rgb01('#5faf8a');
	return rgb01('#e5e2d8');
}

function projectCpu(x: number, y: number, z: number): { x: number; y: number; scale: number } {
	const depth = Math.max(1.1, Math.min(5.4, 2.82 - z));
	const scale = 1 / depth;
	return { x: x * 1.18 * scale, y: y * 1.62 * scale, scale };
}

export class WitnessVolumePass implements FramePass {
	private engine: ObservatoryEngine;
	private scene: WitnessScene | null = null;
	private resources: GpuResources | null = null;
	private bindLayout: GPUBindGroupLayout | null = null;
	private waferPipeline: GPURenderPipeline | null = null;
	private filamentPipeline: GPURenderPipeline | null = null;
	private arrivalPipeline: GPURenderPipeline | null = null;
	private corePipeline: GPURenderPipeline | null = null;
	private shardCount = 0;
	private filamentCount = 0;
	private selectedId: string | null = null;
	private playhead = 1;
	private replayStart = -1;
	private shardData = new Float32Array(MAX_SHARDS * SHARD_FLOATS);
	private hitTargets: HitTarget[] = [];

	constructor(engine: ObservatoryEngine, scene: RouteSceneModel) {
		this.engine = engine;
		this.uploadScene(scene);
	}

	uploadScene(scene: RouteSceneModel): void {
		this.scene = scene as WitnessScene;
		this.selectedId = this.scene.shards[0]?.id ?? null;
		this.playhead = 1;
		this.replayStart = -1;
		const device = this.engine.gpuDevice;
		if (!device) return;
		this.ensurePipelines(device);
		this.ensureResources(device);
		this.writeScene(device);
	}

	setSelected(id: string | null): void {
		this.selectedId = id;
		const device = this.engine.gpuDevice;
		if (!device || !this.resources) return;
		this.writeScene(device);
		this.engine.requestRender();
	}

	setPlayhead(value: number): void {
		this.playhead = clamp01(value);
		this.writeState();
		this.engine.requestRender();
	}

	replay(): void {
		this.replayStart = this.engine.demoClock.state.frame;
		this.writeState();
		this.engine.requestRender();
	}

	/**
	 * Witness is an audit instrument, not a decorative perpetual simulation.
	 * Its 60 fps moments are deliberate: receipt ingress, an explicit replay, or
	 * a direct control change (which calls requestRender above). Once evidence is
	 * settled the same static proof is redrawn at 6 fps, avoiding needless HDR
	 * targets and bloom passes on a Retina display.
	 */
	targetFrameRate(frame: number): number {
		if (frame < 154) return 60;
		if (this.replayStart >= 0) {
			const loopFrames = this.engine.demoClock.framesPerLoop;
			const replayAge = (frame - this.replayStart + loopFrames) % loopFrames;
			if (replayAge < 196) return 60;
		}
		return 6;
	}

	private ensurePipelines(device: GPUDevice): void {
		if (this.waferPipeline || !this.engine.paramsBuffer) return;
		const module = device.createShaderModule({ label: 'witness-volume-wgsl', code: VOLUME_WGSL });
		this.bindLayout = device.createBindGroupLayout({
			label: 'witness-volume-layout',
			entries: [
				{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
				{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
				{ binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
			]
		});
		const layout = device.createPipelineLayout({ label: 'witness-volume-pipeline-layout', bindGroupLayouts: [this.bindLayout] });
		const alphaBlend: GPUBlendState = {
			color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
			alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
		};
		const pipeline = (label: string, vertex: string, fragment: string) => device.createRenderPipeline({
			label,
			layout,
			vertex: { module, entryPoint: vertex },
			fragment: { module, entryPoint: fragment, targets: [{ format: this.engine.sceneFormat, blend: alphaBlend }] },
			primitive: { topology: 'triangle-list', cullMode: 'none' }
		});
		this.filamentPipeline = pipeline('witness-volume-filaments', 'vs_filament', 'fs_filament');
		this.arrivalPipeline = pipeline('witness-volume-arrival-trails', 'vs_arrival_trail', 'fs_arrival_trail');
		this.waferPipeline = pipeline('witness-volume-evidence-wafers', 'vs_wafer', 'fs_wafer');
		this.corePipeline = pipeline('witness-volume-receipt-core', 'vs_core', 'fs_core');
	}

	private ensureResources(device: GPUDevice): void {
		if (this.resources || !this.bindLayout || !this.engine.paramsBuffer) return;
		const shardBuffer = device.createBuffer({
			label: 'witness-volume-shards',
			size: MAX_SHARDS * SHARD_FLOATS * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		const filamentBuffer = device.createBuffer({
			label: 'witness-volume-filaments',
			size: MAX_FILAMENTS * FILAMENT_FLOATS * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		const stateBuffer = device.createBuffer({
			label: 'witness-volume-state',
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const bindGroup = device.createBindGroup({
			label: 'witness-volume-bind-group',
			layout: this.bindLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: shardBuffer } },
				{ binding: 2, resource: { buffer: filamentBuffer } },
				{ binding: 3, resource: { buffer: stateBuffer } }
			]
		});
		this.resources = { shardBuffer, filamentBuffer, stateBuffer, bindGroup };
	}

	private writeState(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.resources) return;
		const selectedIndex = this.scene?.shards.findIndex((shard) => shard.id === this.selectedId) ?? -1;
		device.queue.writeBuffer(
			this.resources.stateBuffer,
			0,
			new Float32Array([this.playhead, this.replayStart, selectedIndex, this.shardCount])
		);
	}

	private writeScene(device: GPUDevice): void {
		if (!this.resources || !this.scene) return;
		const shards = this.scene.shards.slice(0, MAX_SHARDS);
		this.shardCount = shards.length;
		this.shardData.fill(0);
		this.hitTargets = [];

		for (let index = 0; index < shards.length; index += 1) {
			const shard = shards[index];
			const eventTime = clamp01(shard.traceTime);
			const order = shards.length <= 1 ? 0.5 : shard.order / (shards.length - 1);
			// The decision is a sealed reliquary, not a constellation. Chronology
			// builds its vertical strata; activation opens a lateral seam; retention
			// changes the physical slate scale. No coordinate is randomized.
			const side = index % 2 === 0 ? -1 : 1;
			const withinStratum = ((shard.order * 0.61803398875) % 1 - 0.5) * 0.22;
			const x = 0.08 + (clamp01(shard.activation) - 0.5) * 1.42 + side * 0.19 + withinStratum;
			const y = 1.16 - eventTime * 2.36 + side * 0.075;
			const z = -0.30 + eventTime * 1.18 + clamp01(shard.retention) * 0.15 - (shard.suppressed ? 0.20 : 0);
			const color = colorFor(shard);
			const selected = shard.id === this.selectedId ? 1 : 0;
			const offset = index * SHARD_FLOATS;
			this.shardData.set([
				x, y, z, 0.92 + shard.retention * 0.36,
				clamp01(shard.activation), clamp01(shard.retention), clamp01(eventTime), selected,
				color[0], color[1], color[2], 1,
				roleCode(shard), shard.suppressed ? 1 : 0, index / Math.max(1, shards.length - 1), shard.mutated ? 1 : 0
			], offset);
			const projected = projectCpu(x, y, z);
			this.hitTargets.push({
				shard,
				x: projected.x,
				y: projected.y,
				// The extra picking lens encloses the projected front and two extrusion
				// faces, keeping direct specimen selection reliable at any rotation.
				radius: (0.29 + shard.retention * 0.14) * projected.scale * 2.25
			});
		}
		device.queue.writeBuffer(this.resources.shardBuffer, 0, this.shardData);

		const fibers = new Float32Array(MAX_FILAMENTS * FILAMENT_FLOATS);
		let fiberIndex = 0;
		// Membership is communicated by the enclosing reliquary. The only visible
		// thread is the verified, ordered activation path — no generic spokes.
		for (const edge of this.scene.edges) {
			if (fiberIndex >= MAX_FILAMENTS || edge.sourceIndex < 0 || edge.targetIndex < 0) break;
			fibers.set([edge.sourceIndex, edge.targetIndex, 0, 0, clamp01(edge.weight), fiberIndex * 0.137, 1, 0], fiberIndex * FILAMENT_FLOATS);
			fiberIndex += 1;
		}
		this.filamentCount = fiberIndex;
		device.queue.writeBuffer(this.resources.filamentBuffer, 0, fibers);
		this.engine.params[2] = this.shardCount;
		this.engine.params[3] = this.filamentCount;
		this.writeState();
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.resources || !this.waferPipeline || !this.filamentPipeline || !this.arrivalPipeline || !this.corePipeline) return;
		pass.setBindGroup(0, this.resources.bindGroup);
		if (this.filamentCount > 0) {
			pass.setPipeline(this.filamentPipeline);
			pass.draw(6, this.filamentCount);
		}
		if (this.shardCount > 0) {
			pass.setPipeline(this.arrivalPipeline);
			pass.draw(6, this.shardCount);
			pass.setPipeline(this.waferPipeline);
			pass.draw(18, this.shardCount);
		}
		pass.setPipeline(this.corePipeline);
		pass.draw(6, 1);
	}

	pickAt(ndcX: number, ndcY: number): RoutePick | null {
		let nearest: HitTarget | null = null;
		let distance = Infinity;
		for (const target of this.hitTargets) {
			const dx = target.x - ndcX;
			const dy = target.y - ndcY;
			const next = Math.hypot(dx, dy);
			if (next <= target.radius && next < distance) {
				nearest = target;
				distance = next;
			}
		}
		return nearest ? { id: nearest.shard.id, kind: 'witness-shard', payload: nearest.shard } : null;
	}

	dispose(): void {
		this.resources?.shardBuffer.destroy();
		this.resources?.filamentBuffer.destroy();
		this.resources?.stateBuffer.destroy();
		this.resources = null;
	}
}

export function createWitnessVolumePasses(engine: ObservatoryEngine, scene: RouteSceneModel): WitnessVolumePass[] {
	return [new WitnessVolumePass(engine, scene)];
}

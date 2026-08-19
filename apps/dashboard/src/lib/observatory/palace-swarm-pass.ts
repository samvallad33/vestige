/**
 * PalaceSwarmPass — nine living route constellations on the shared Observatory GPU.
 *
 * One instanced draw renders the whole Palace into Observatory's HDR/bloom chain.
 * Formation geometry is baked once on upload; WGSL only animates those silhouettes,
 * which keeps the field cinematic without a CPU simulation or a second WebGPU device.
 */

import type { FramePass, ObservatoryEngine } from './engine';
import { BITEMPORAL, IMMUNE, RETENTION, rgb01 } from './cognitive-palette';
import type { OrganRegion } from './palace-map';

const FLOATS_PER_PARTICLE = 12;
const UNIFORM_FLOATS = 16;
const BURST_MS = 900;
const REDUCED_BURST_MS = 190;
const FLASH_REQUEST_AT = 0.48;

const FAMILY_COLOR: Record<OrganRegion['family'], string> = {
	reasoning: RETENTION.bridge,
	memory: RETENTION.recall,
	immune: IMMUNE.veto,
	signal: BITEMPORAL.supersession,
	temporal: BITEMPORAL.txShadow,
	system: RETENTION.luciferin
};

const FORMATION_KIND: Record<string, number> = {
	'/observatory': 0,
	'/graph': 1,
	'/memories': 2,
	'/timeline': 3,
	'/blackbox': 4,
	'/reasoning': 5,
	'/explore': 6,
	'/feed': 7,
	'/contradictions': 8
};

// The approved Palace composition. Motion is applied inside these anchors and
// never moves a target far enough to invalidate pointer picking.
const DESKTOP_ANCHORS: Record<string, [number, number, number, number]> = {
	'/observatory': [-0.04, 0.02, 0.12, 0.155],
	'/graph': [-0.61, 0.42, 0.04, 0.13],
	'/memories': [0.56, 0.39, -0.04, 0.13],
	'/timeline': [-0.67, -0.36, 0.1, 0.125],
	'/blackbox': [-0.37, 0.0, 0.18, 0.12],
	'/reasoning': [0.34, -0.02, -0.08, 0.125],
	'/explore': [0.04, 0.61, -0.13, 0.12],
	'/feed': [0.64, -0.38, 0.02, 0.12],
	'/contradictions': [0.02, -0.59, -0.02, 0.12]
};

interface PlacedRegion {
	href: string;
	x: number;
	y: number;
	z: number;
	scale: number;
	kind: number;
}

export interface PalaceSwarmScreenPos {
	href: string;
	ndcX: number;
	ndcY: number;
	depth: number;
	visible: boolean;
}

/**
 * Route indices mirror ORGAN_REGIONS:
 * 0 cortex, 1 graph, 2 memories, 3 timeline, 4 feed, 5 explore,
 * 6 reasoning, 7 black box, 8 contradictions.
 *
 * All ambient motion is periodic over Observatory's 12 second deterministic
 * loop. Capture URLs remain stable and the animation never pops at the seam.
 */
const palaceSwarmWGSL = /* wgsl */ `
struct Uniforms {
	viewport: vec4<f32>,
	interaction: vec4<f32>,
	portal: vec4<f32>,
	intro: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

const PI = 3.14159265359;
const TAU = 6.28318530718;
const LOOP_SECONDS = 12.0;
const QUAD = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

fn rotate2(point: vec2<f32>, angle: f32) -> vec2<f32> {
	let c = cos(angle);
	let s = sin(angle);
	return vec2<f32>(c * point.x - s * point.y, s * point.x + c * point.y);
}

struct VertexOut {
	@builtin(position) position: vec4<f32>,
	@location(0) local: vec2<f32>,
	@location(1) color: vec3<f32>,
	@location(2) energy: f32,
};

@vertex
fn vs_main(
	@location(0) baked: vec4<f32>,
	@location(1) anchor: vec4<f32>,
	@location(2) color_route: vec4<f32>,
	@builtin(vertex_index) vertex_index: u32,
	@builtin(instance_index) instance_index: u32
) -> VertexOut {
	let id = f32(instance_index + 1u);
	let route = color_route.w;
	let aspect = u.viewport.x / max(u.viewport.y, 1.0);
	let time = u.viewport.z;
	let loop01 = time / LOOP_SECONDS;
	let clock = loop01 * TAU;
	let reduced = u.portal.w > 0.5;
	let seed0 = fract(id * 0.61803398875);
	let seed1 = fract(id * 0.41421356237);
	let seed2 = fract(id * 0.73205080757);

	var local3 = baked.xyz;
	var organ_anchor = anchor.xy;
	var wave = 0.0;
	var axon = 0.0;

	if (!reduced) {
		// Each organ floats on its own slow phase while remaining within the
		// deliberately generous hit target used by pickAt().
		let organ_phase = route * 0.73;
		organ_anchor += vec2<f32>(
			sin(clock + organ_phase),
			cos(clock + organ_phase * 1.37)
		) * 0.0038;

		if (route < 0.5) {
			// Observatory: a cortex breathing as traveling sulci brighten and lift.
			let breath = 1.0 + 0.045 * sin(clock * 2.0);
			let sulcus = 0.5 + 0.5 * sin(local3.y * 13.0 + local3.x * 8.0 - clock * 5.0);
			let cortex_xy = local3.xy * vec2<f32>(breath, 1.0 + (breath - 1.0) * 0.62);
			local3 = vec3<f32>(
				cortex_xy.x + (sulcus - 0.5) * 0.025 * sign(local3.x),
				cortex_xy.y,
				local3.z
			);
			wave = pow(sulcus, 5.0);
		} else if (route < 1.5) {
			// Graph: two-axis volumetric rotation plus excitation bands.
			let graph_xz = rotate2(local3.xz, clock);
			local3 = vec3<f32>(graph_xz.x, local3.y, graph_xz.y);
			let graph_yz = rotate2(local3.yz, clock * 2.0 + 0.35);
			local3 = vec3<f32>(local3.x, graph_yz.x, graph_yz.y);
			let excitation = 0.5 + 0.5 * sin(local3.y * 10.0 - clock * 6.0 + seed0 * 1.4);
			local3 *= 1.0 + 0.025 * sin(clock * 3.0 + seed1 * TAU);
			wave = pow(excitation, 7.0);
		} else if (route < 2.5) {
			// Memories: cells oscillate independently while depth energy scans them.
			let cell = floor((local3.x + 1.1) * 2.5)
				+ floor((local3.y + 1.1) * 2.5) * 5.0
				+ floor((local3.z + 1.1) * 2.5) * 25.0;
			let cell_phase = clock * 2.0 + cell * 0.71;
			local3 += vec3<f32>(
				sin(cell_phase),
				cos(cell_phase * 0.5 + seed1),
				sin(cell_phase + 1.7)
			) * 0.028;
			wave = pow(0.5 + 0.5 * sin(local3.z * 9.0 - clock * 7.0), 6.0);
		} else if (route < 3.5) {
			// Timeline: a true turning helix with signals climbing both strands.
			let timeline_xz = rotate2(local3.xz, clock);
			local3 = vec3<f32>(timeline_xz.x, local3.y, timeline_xz.y);
			wave = pow(0.5 + 0.5 * sin((local3.y + 1.0) * PI * 3.0 - clock * 8.0), 8.0);
		} else if (route < 4.5) {
			// Feed: continuous event traffic moving through eight living lanes.
			let feed_y = fract((baked.y + 1.0) * 0.5 + loop01 * 2.0 + seed1 * 0.06) * 2.0 - 1.0;
			local3 = vec3<f32>(
				local3.x + sin(feed_y * 5.0 + clock * 4.0 + seed0 * TAU) * 0.026,
				feed_y,
				local3.z
			);
			wave = pow(0.5 + 0.5 * sin(local3.y * 12.0 - clock * 10.0), 9.0);
		} else if (route < 5.5) {
			// Explore: the semantic spiral turns and sends discovery rings outward.
			let explore_xz = rotate2(local3.xz, clock);
			local3 = vec3<f32>(explore_xz.x, local3.y, explore_xz.y);
			let radius = length(local3.xz);
			let expanding_xz = local3.xz * (1.0 + 0.065 * sin(clock * 2.0 - radius * 9.0));
			local3 = vec3<f32>(expanding_xz.x, local3.y, expanding_xz.y);
			wave = pow(0.5 + 0.5 * sin(radius * 13.0 - clock * 7.0), 7.0);
		} else if (route < 6.5) {
			// Reasoning: a slow two-axis tumble keeps the torus unmistakably 3D.
			let reasoning_xz = rotate2(local3.xz, clock);
			local3 = vec3<f32>(reasoning_xz.x, local3.y, reasoning_xz.y);
			let reasoning_xy = rotate2(local3.xy, clock * 2.0 + 0.4);
			local3 = vec3<f32>(reasoning_xy, local3.z);
			wave = pow(0.5 + 0.5 * sin(atan2(local3.z, local3.x) * 3.0 - clock * 6.0), 7.0);
		} else if (route < 7.5) {
			// Black Box: faster receipt lanes, punctuated by evidence packets.
			let receipt_y = fract((baked.y + 1.0) * 0.5 + loop01 * 3.0 + seed2 * 0.08) * 2.0 - 1.0;
			local3 = vec3<f32>(
				local3.x + sin(receipt_y * 15.0 + clock * 6.0 + seed0 * 2.0) * 0.018,
				receipt_y,
				local3.z
			);
			wave = pow(0.5 + 0.5 * sin(local3.y * 17.0 - clock * 12.0), 10.0);
		} else {
			// Contradictions: opposed trust fields breathe exactly out of phase.
			let side = select(-1.0, 1.0, baked.x > 0.0);
			let lobe_center = side * 0.47;
			let opposition = 1.0 + 0.095 * sin(clock * 2.0 + select(PI, 0.0, side > 0.0));
			local3 = vec3<f32>(
				lobe_center + (local3.x - lobe_center) * opposition,
				local3.y * opposition,
				local3.z
			);
			wave = pow(0.5 + 0.5 * sin(local3.y * 11.0 - clock * 6.0 + side * PI * 0.5), 7.0);
		}

		// Microscopic circulation keeps even quiet parts of a silhouette from
		// becoming a frozen point cloud.
		let circulation = vec2<f32>(
			cos(clock * 2.0 + seed0 * TAU),
			sin(clock * 2.0 + seed1 * TAU)
		) * (0.003 + seed2 * 0.004);
		local3 = vec3<f32>(local3.xy + circulation, local3.z);
	}

	var pos = organ_anchor + vec2<f32>(local3.x / aspect, local3.y) * anchor.w;

	// Six percent of every outer organ circulates through curved axons to the
	// cortex. These are existing particles, not another draw or another buffer.
	if (!reduced && route > 0.5 && seed0 < 0.06) {
		let cortex = vec2<f32>(-0.04, 0.02);
		let travel = fract(seed1 + loop01 * 2.0 + route * 0.03125);
		let direction = normalize(organ_anchor - cortex + vec2<f32>(0.00001, 0.0));
		let normal = vec2<f32>(-direction.y, direction.x);
		let arc = sin(travel * PI) * (0.025 + 0.018 * seed2);
		let nerve = sin(travel * PI * 5.0 - clock * 4.0 + seed2 * TAU) * 0.005;
		pos = mix(cortex, organ_anchor, travel) + normal * (arc + nerve);
		axon = 1.0;
		wave = 1.0;
		local3 = vec3<f32>(local3.xy, 0.0);
	}

	let hover_on = u.interaction.x >= 0.0 && u.interaction.y > 0.001;
	let hovered = abs(route - u.interaction.x) < 0.25;
	let focus = select(0.0, u.interaction.y, hovered);
	if (hovered) {
		pos = mix(pos, organ_anchor + (pos - organ_anchor) * 0.82, focus);
	}

	// Opening: the atlas unfolds from one central singularity.
	let intro = select(smoothstep(0.05, 0.9, u.intro.x), 1.0, reduced);
	let cortex = vec2<f32>(-0.04, 0.02);
	if (intro < 0.999) {
		let rel = rotate2(pos - cortex, (1.0 - intro) * (2.0 + seed0) * PI);
		pos = mix(cortex + rel * 0.04, pos, intro)
			+ normalize(pos - cortex + vec2<f32>(0.00001, 0.0)) * sin(intro * PI) * 0.075;
	}

	// Click: the complete Palace collapses into the selected portal, then bursts.
	let progress = u.interaction.w;
	let bursting = u.interaction.z >= 0.0 && progress > 0.0;
	var flash = 0.0;
	if (bursting) {
		if (reduced) {
			pos = mix(pos, u.portal.xy, smoothstep(0.0, 1.0, progress));
		} else {
			let inhale = smoothstep(0.1, 0.58, progress);
			let rel = rotate2(pos - u.portal.xy, inhale * (3.0 + seed0 * 2.5) * PI);
			pos = u.portal.xy + rel * pow(max(1.0 - inhale, 0.0), 1.35);
			let exhale = smoothstep(0.58, 1.0, progress);
			let random_direction = vec2<f32>(seed1 - 0.5, seed2 - 0.5) * 0.12;
			let direction = normalize(rel + random_direction + vec2<f32>(0.00001, 0.0));
			let supernova = u.portal.xy + direction * (1.25 + seed1 * 0.75) * exhale * exhale;
			pos = mix(pos, supernova, exhale);
			flash = exp(-pow((progress - 0.58) / 0.065, 2.0));
		}
	}

	var living_pulse = 0.82;
	if (!reduced) {
		living_pulse = 0.72 + 0.18 * sin(clock * 8.0 + seed2 * TAU);
	}
	var energy = 0.55 + living_pulse + wave * 1.35 + axon * 1.25
		+ focus * 1.7 + sin(intro * PI) * 1.1 + flash * 7.0;
	if (hover_on && !hovered) {
		energy *= 0.28;
	}

	let corner = QUAD[vertex_index];
	let pixel = 2.0 / max(min(u.viewport.x, u.viewport.y), 1.0);
	let depth_size = clamp(1.0 + local3.z * 0.12, 0.82, 1.22);
	let size = (1.2 + baked.w * 1.9 + wave * 0.7 + axon * 0.75 + focus * 1.5 + min(flash * 3.0, 3.0)) * depth_size;
	let wave_white = clamp(wave * 0.24 + axon * 0.28 + focus * 0.32 + flash * 0.92, 0.0, 0.96);

	var out: VertexOut;
	out.position = vec4<f32>(pos + corner * pixel * size, 0.0, 1.0);
	out.local = corner;
	out.color = mix(color_route.rgb * (0.68 + seed0 * 0.55), vec3<f32>(1.0), wave_white);
	out.energy = energy;
	return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
	let radius = clamp(length(input.local) / 0.92, 0.0, 1.0);
	let disc = 1.0 - smoothstep(0.0, 1.0, radius);
	if (disc <= 0.001) { discard; }
	let profile = (pow(1.0 - radius, 3.1) + (1.0 - radius) * 0.38) * disc;
	return vec4<f32>(input.color * profile * (0.15 + input.energy * 0.36), 1.0);
}
`;

function fract(value: number): number {
	return value - Math.floor(value);
}

function hash(value: number): number {
	return fract(Math.sin(value * 12.9898 + 78.233) * 43758.5453);
}

function particleCount(reducedMotion: boolean): number {
	const cores = navigator.hardwareConcurrency || 8;
	const small = window.innerWidth < 760;
	const weak = cores <= 4 || (window.devicePixelRatio || 1) > 2.2;
	// Density target (Sam, Jul 13 2026): the swarm is baked-geometry instanced
	// billboards animated in the vertex stage, so raising the count is cheap on the
	// GPU (no per-frame compute) — the visual budget goes to the MOTION HIERARCHY on
	// top (ambient drift → organ breathing → cortex heartbeat → nerve packets →
	// cognitive-weather ignition), not to denser static pictograms. Performance-gated:
	// strong desktop (>=10 cores, e.g. M1 Max) 55k, standard desktop 40k, mobile 18k.
	if (reducedMotion) return small ? 10_000 : 16_000;
	if (weak) return small ? 12_000 : 20_000;
	return small ? 18_000 : cores >= 10 ? 55_000 : 40_000;
}

function formationPoint(
	kind: number,
	id: number,
	a: number,
	b: number,
	c: number
): [number, number, number] {
	const TAU = Math.PI * 2;
	switch (kind) {
		case 0: {
			const lobe = a > 0.5 ? 1 : -1;
			const theta = b * TAU;
			const phi = Math.acos(Math.max(-1, Math.min(1, 2 * c - 1)));
			const radius = 0.72 + 0.28 * Math.sqrt(hash(id + 19));
			let x = Math.abs(Math.sin(phi) * Math.cos(theta)) * lobe;
			const y = Math.cos(phi) * 0.92;
			const z = Math.sin(phi) * Math.sin(theta) * 1.12;
			x = x * 0.72 + lobe * 0.38;
			const ridge = Math.sin(x * 9) * Math.sin(y * 8) * 0.09;
			return [x * radius * (1 + ridge), y * radius * (1 + ridge), z * radius * (1 + ridge)];
		}
		case 1: {
			const k = Math.floor(a * 420);
			const y = 1 - (k / 419) * 2;
			const radius = Math.sqrt(Math.max(0, 1 - y * y));
			const theta = 2.39996323 * k;
			return [
				Math.cos(theta) * radius + (hash(id + 7) - 0.5) * 0.12,
				y + (hash(id + 9) - 0.5) * 0.12,
				Math.sin(theta) * radius + (hash(id + 11) - 0.5) * 0.12
			];
		}
		case 2: {
			const grid = 5;
			const cell = 2 / grid;
			return [
				(Math.floor(a * grid) + 0.5) * cell - 1 + (hash(id + 1) - 0.5) * 0.15,
				(Math.floor(b * grid) + 0.5) * cell - 1 + (hash(id + 2) - 0.5) * 0.15,
				(Math.floor(c * grid) + 0.5) * cell - 1 + (hash(id + 3) - 0.5) * 0.15
			];
		}
		case 3: {
			const strand = b > 0.5 ? 1 : -1;
			const y = a * 2 - 1;
			const angle = y * Math.PI * 2.6 + strand * Math.PI * 0.5;
			const radius = 0.48 + hash(id + 31) * 0.09;
			return [Math.cos(angle) * radius, y, Math.sin(angle) * radius];
		}
		case 4: {
			const lane = Math.floor(a * 7);
			return [lane / 6 * 1.65 - 0.825 + Math.sin(b * 18 + lane) * 0.035, b * 2 - 1, (c - 0.5) * 0.55];
		}
		case 5: {
			const angle = a * TAU;
			const ring = b * TAU;
			const major = 0.62;
			const minor = 0.28;
			return [
				(major + minor * Math.cos(ring)) * Math.cos(angle),
				minor * Math.sin(ring) * 1.8,
				(major + minor * Math.cos(ring)) * Math.sin(angle)
			];
		}
		case 6: {
			const angle = a * TAU * 3.2;
			const radius = 0.12 + a * 0.82;
			return [Math.cos(angle) * radius, (b - 0.5) * 0.48 + Math.sin(angle * 0.5) * 0.12, Math.sin(angle) * radius];
		}
		case 7: {
			const lane = Math.floor(a * 8);
			const y = b * 2 - 1;
			return [lane / 7 * 1.8 - 0.9 + Math.sin(y * 4 + lane) * 0.07, y, (c - 0.5) * 0.45];
		}
		default: {
			const side = a > 0.5 ? 1 : -1;
			const angle = b * TAU;
			const radius = Math.sqrt(c) * 0.47;
			return [side * 0.47 + Math.cos(angle) * radius, (hash(id + 4) * 2 - 1) * radius, Math.sin(angle) * radius];
		}
	}
}

export class PalaceSwarmPass implements FramePass {
	private engine: ObservatoryEngine;
	private pipeline: GPURenderPipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private uniformBuffer: GPUBuffer | null = null;
	private particleBuffer: GPUBuffer | null = null;
	private uniformData = new Float32Array(UNIFORM_FLOATS);
	private placed: PlacedRegion[] = [];
	private count = 0;
	private hoveredIndex = -1;
	private hoverStrength = 0;
	private reducedMotion = false;
	private bornMs: number;
	private burst: { href: string; startMs: number; callbackFired: boolean } | null = null;
	private onFlashPeak: ((href: string) => void) | null = null;
	private watchdog: ReturnType<typeof setTimeout> | null = null;

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
		this.bornMs = engine.wallNowMs;
	}

	setReducedMotion(reduced: boolean): void {
		this.reducedMotion = reduced;
	}

	setHovered(index: number): void {
		if (this.burst) return;
		this.hoveredIndex = index >= 0 && index < this.placed.length ? index : -1;
	}

	indexOfHref(href: string | null): number {
		if (!href) return -1;
		return this.placed.findIndex((placed) => placed.href === href);
	}

	get isBursting(): boolean {
		return this.burst !== null;
	}

	startBurst(href: string, onFlashPeak: (href: string) => void): boolean {
		if (this.burst) return false;
		const target = this.placed.find((placed) => placed.href === href);
		if (!target) return false;
		this.hoveredIndex = this.placed.indexOf(target);
		this.hoverStrength = 1;
		this.burst = { href, startMs: this.engine.wallNowMs, callbackFired: false };
		this.onFlashPeak = onFlashPeak;
		// Navigation must not depend on a healthy rAF/device after the click.
		this.watchdog = setTimeout(() => this.fireFlashCallback(), 1_100);
		return true;
	}

	uploadRegions(regions: OrganRegion[]): void {
		const device = this.engine.gpuDevice;
		if (!device || regions.length === 0) return;

		this.placed = regions.map((region, index) => {
			const anchor = DESKTOP_ANCHORS[region.href] ?? [0, 0, index * 0.01, 0.12];
			return {
				href: region.href,
				x: anchor[0],
				y: anchor[1],
				z: anchor[2],
				scale: anchor[3],
				kind: FORMATION_KIND[region.href] ?? index % 9
			};
		});

		this.count = particleCount(this.reducedMotion);
		const particleData = new Float32Array(this.count * FLOATS_PER_PARTICLE);
		for (let i = 0; i < this.count; i++) {
			const id = i + 1;
			const offset = i * FLOATS_PER_PARTICLE;
			// Route-contiguous instances improve branch coherence in the shader.
			const route = Math.min(this.placed.length - 1, Math.floor((i * this.placed.length) / this.count));
			const a = hash(id * 1.17 + 3.0);
			const b = hash(id * 1.91 + 7.0);
			const c = hash(id * 2.37 + 11.0);
			const local = formationPoint(this.placed[route].kind, id, a, b, c);
			const anchor = this.placed[route];
			const color = rgb01(FAMILY_COLOR[regions[route].family]);
			particleData[offset + 0] = local[0];
			particleData[offset + 1] = local[1];
			particleData[offset + 2] = local[2];
			particleData[offset + 3] = hash(id * 2.83 + 12.0);
			particleData[offset + 4] = anchor.x;
			particleData[offset + 5] = anchor.y;
			particleData[offset + 6] = anchor.z;
			particleData[offset + 7] = anchor.scale;
			particleData[offset + 8] = color[0];
			particleData[offset + 9] = color[1];
			particleData[offset + 10] = color[2];
			particleData[offset + 11] = route;
		}

		this.uniformBuffer?.destroy();
		this.particleBuffer?.destroy();
		this.uniformBuffer = device.createBuffer({
			label: 'palace-swarm-uniforms',
			size: this.uniformData.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.particleBuffer = device.createBuffer({
			label: 'palace-swarm-particles',
			size: particleData.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.particleBuffer, 0, particleData);
		this.createPipeline(device);
	}

	private createPipeline(device: GPUDevice): void {
		if (!this.uniformBuffer || !this.particleBuffer) return;
		device.pushErrorScope('validation');
		const module = device.createShaderModule({ label: 'palace-swarm-shader', code: palaceSwarmWGSL });
		void module.getCompilationInfo().then((info) => {
			const errors = info.messages.filter((message) => message.type === 'error');
			if (errors.length > 0) {
				console.error('[palace] WGSL compilation:', errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n'));
			}
		});
		this.pipeline = device.createRenderPipeline({
			label: 'palace-swarm-pipeline',
			layout: 'auto',
			vertex: {
				module,
				entryPoint: 'vs_main',
				buffers: [{
					arrayStride: FLOATS_PER_PARTICLE * 4,
					stepMode: 'instance',
					attributes: [
						{ shaderLocation: 0, offset: 0, format: 'float32x4' },
						{ shaderLocation: 1, offset: 16, format: 'float32x4' },
						{ shaderLocation: 2, offset: 32, format: 'float32x4' }
					]
				}]
			},
			fragment: {
				module,
				entryPoint: 'fs_main',
				targets: [{
					format: this.engine.sceneFormat,
					blend: {
						color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
						alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
					}
				}]
			},
			primitive: { topology: 'triangle-list' }
		});
		this.bindGroup = device.createBindGroup({
			label: 'palace-swarm-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
		});
		void device.popErrorScope().then((error) => {
			if (error) console.error('[palace] WebGPU validation error:', error.message);
		});
	}

	private fireFlashCallback(): void {
		if (!this.burst || this.burst.callbackFired || !this.onFlashPeak) return;
		this.burst.callbackFired = true;
		const href = this.burst.href;
		const callback = this.onFlashPeak;
		this.onFlashPeak = null;
		if (this.watchdog) clearTimeout(this.watchdog);
		this.watchdog = null;
		queueMicrotask(() => callback(href));
	}

	compute(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.uniformBuffer || this.placed.length === 0) return;
		const now = this.engine.wallNowMs;
		const hoverTarget = this.burst ? 1 : this.hoveredIndex >= 0 ? 1 : 0;
		this.hoverStrength += (hoverTarget - this.hoverStrength) * (this.reducedMotion ? 1 : 0.16);
		if (this.hoverStrength < 0.001) this.hoverStrength = 0;

		let progress = 0;
		let selectedIndex = -1;
		let target = this.hoveredIndex >= 0 ? this.placed[this.hoveredIndex] : null;
		if (this.burst) {
			selectedIndex = this.indexOfHref(this.burst.href);
			target = selectedIndex >= 0 ? this.placed[selectedIndex] : target;
			const duration = this.reducedMotion ? REDUCED_BURST_MS : BURST_MS;
			progress = Math.min(1, Math.max(0, (now - this.burst.startMs) / duration));
			const threshold = this.reducedMotion ? 0.3 : FLASH_REQUEST_AT;
			if (progress >= threshold) this.fireFlashCallback();
		}

		const intro = this.engine.params[11] > 0.5 ? 1 : Math.min(1, (now - this.bornMs) / 1_650);
		const flash = this.reducedMotion || progress <= 0 ? 0 : Math.exp(-Math.pow((progress - 0.58) / 0.065, 2));
		this.uniformData[0] = this.engine.params[6] || 1;
		this.uniformData[1] = this.engine.params[7] || 1;
		this.uniformData[2] = this.engine.params[10];
		this.uniformData[3] = this.count;
		this.uniformData[4] = this.burst ? selectedIndex : this.hoveredIndex;
		this.uniformData[5] = this.hoverStrength;
		this.uniformData[6] = selectedIndex;
		this.uniformData[7] = progress;
		this.uniformData[8] = target?.x ?? 0;
		this.uniformData[9] = target?.y ?? 0;
		this.uniformData[10] = target?.z ?? 0;
		this.uniformData[11] = this.reducedMotion ? 1 : 0;
		this.uniformData[12] = intro;
		this.uniformData[13] = this.placed.length;
		this.uniformData[14] = this.engine.params[5];
		this.uniformData[15] = flash;
		device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || !this.particleBuffer || this.count === 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setVertexBuffer(0, this.particleBuffer);
		pass.draw(6, this.count);
	}

	pickAt(ndcX: number, ndcY: number): { index: number; href: string } | null {
		if (this.burst || this.placed.length === 0) return null;
		const width = this.engine.params[6] || 1;
		const height = this.engine.params[7] || 1;
		const aspect = width / height;
		let best = -1;
		let bestScore = Infinity;
		for (let i = 0; i < this.placed.length; i++) {
			const placed = this.placed[i];
			const dx = (ndcX - placed.x) * aspect;
			const dy = ndcY - placed.y;
			let score = Math.hypot(dx, dy) / (placed.scale * 1.12);
			if (i === this.hoveredIndex) score *= 0.78;
			if (score < 1.05 && score < bestScore) {
				best = i;
				bestScore = score;
			}
		}
		return best < 0 ? null : { index: best, href: this.placed[best].href };
	}

	getScreenPositions(): PalaceSwarmScreenPos[] {
		return this.placed.map((placed) => ({
			href: placed.href,
			ndcX: placed.x,
			ndcY: placed.y,
			depth: Math.min(1, Math.max(0, 0.72 + placed.z)),
			visible: true
		}));
	}

	dispose(): void {
		if (this.watchdog) clearTimeout(this.watchdog);
		this.watchdog = null;
		this.uniformBuffer?.destroy();
		this.particleBuffer?.destroy();
		this.uniformBuffer = null;
		this.particleBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
		this.placed = [];
		this.count = 0;
		this.burst = null;
		this.onFlashPeak = null;
	}
}

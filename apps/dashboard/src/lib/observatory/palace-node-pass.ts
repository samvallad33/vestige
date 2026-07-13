/**
 * Spatial Palace — bespoke hero-scale organ constellation pass.
 *
 * The shared NodeRenderer is tuned for a DENSE memory graph (fieldRadius 120,
 * ORBIT_DISTANCE 300, node radius ~3.2), so the 19 organ nodes collapse into a
 * tiny center speck. Those constants are load-bearing for /graph — we do NOT
 * touch them. Instead this pass owns its OWN world: its own node storage buffer,
 * its own camera buffer, its own render pipeline, and a MUCH closer orbit with a
 * MUCH larger billboard so the 19 organs fill the frame like a brain-galaxy.
 *
 * It is a plain FramePass (engine.ts):
 *  - compute(encoder): writes the deterministic close-orbit camera each frame.
 *  - render(pass): one instanced additive draw of all organ billboards.
 *
 * Additive blend into engine.sceneFormat (rgba16float HDR) — the PostChain adds
 * bloom on top, so overlapping halos build light instead of z-fighting. No GPU
 * readback in the frame loop; pickAt() reprojects the CPU-side node positions
 * (they are static once uploaded) through the current viewProj, matching what
 * the frame drew. getScreenPositions() does the same so MSDF labels ride nodes.
 *
 * WGSL reserved-word hygiene: no field/var named meta/active/filter/sample/
 * texture/binding/common/override — this module uses info/beat/data/kind.
 */

import type { ObservatoryEngine, FramePass } from './engine';
import { orbitCamera, perspective, lookAt, multiply } from './camera';
import { rgb01 } from './cognitive-palette';
import type { OrganRegion } from './palace-map';

/**
 * mat4 (16) + right vec4 (4) + up vec4 (4) + hover vec4 (4) floats — matches the
 * WGSL Camera struct. hover = (hoveredIndex, hoverX, hoverY, hoverZ): the focused
 * organ's buffer index + its world position, so the shader can glow it and part
 * the OTHER orbs radially away from it (focus+context nav).
 */
const CAMERA_FLOATS = 28;

/** f32 per node: pos.xyz + radius (4), color.rgb + familyId (4), flags + pad (4) = 12. */
const FLOATS_PER_NODE = 12;

/**
 * Close orbit — the whole point of the bespoke pass. At ~18 units with a big
 * billboard the constellation FILLS the frame instead of hiding in the center.
 */
const ORBIT_DISTANCE = 18;
const ORBIT_ELEVATION = 0.32;

/** Constellation extents (world units) for the golden-angle sphere shell. */
const SHELL_RADIUS = 8.2;
/** Center organ (Observatory) core radius — the gravitational heart. */
const CENTER_RADIUS = 2.35;
/** Outer organ core radius. */
const ORGAN_RADIUS = 1.55;

/** Per-family accent colors — cyan/green reasoning+memory, amber/scarlet immune, indigo temporal. */
const FAMILY_COLOR: Record<OrganRegion['family'], string> = {
	reasoning: '#22C7DE', // cyan
	memory: '#29F2A9', // mint-green
	immune: '#FF5E7A', // scarlet
	signal: '#FFC44D', // amber
	temporal: '#8B7BFF', // indigo
	system: '#DDE7FF' // near-white anchor
};

/** Numeric family ids so the shader can tint per family without string data. */
const FAMILY_ID: Record<OrganRegion['family'], number> = {
	reasoning: 0,
	memory: 1,
	immune: 2,
	signal: 3,
	temporal: 4,
	system: 5
};

interface PlacedRegion {
	href: string;
	x: number;
	y: number;
	z: number;
	radius: number;
	center: boolean;
}

export interface PalaceScreenPos {
	href: string;
	ndcX: number;
	ndcY: number;
	/** clip-space w (depth proxy — larger = farther). */
	depth: number;
	/** false when the node is behind the camera this frame. */
	visible: boolean;
}

const palaceNodeWGSL = /* wgsl */ `
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

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
	// x: hover code = strength*(hoveredIndex+1), 0 = nothing hovered.
	// yzw: focused organ world position (parting pushes others away from it).
	hover: vec4<f32>,
};

struct Organ {
	pos_radius: vec4<f32>,   // xyz world pos, w core radius
	color_family: vec4<f32>, // rgb accent, w family id
	info: vec4<f32>,         // x center flag, yzw reserved
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> organs: array<Organ>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) accent: vec3<f32>,
	// x radius, y center flag, z family id, w breath
	@location(2) @interpolate(flat) info: vec4<f32>,
	// focus factor for THIS orb: 1 = the hovered organ, 0 = not (eases via strength)
	@location(3) @interpolate(flat) focus: f32,
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
	if (ii >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let organ = organs[ii];
	let corner = CORNERS[vi];
	let is_center = organ.info.x > 0.5;

	// Breath: hero orbs swell ~8% on the global pulse; the center heart breathes
	// deeper. A slow per-organ phase offset (from world x) desyncs the field so
	// it shimmers like a living constellation, not a single strobe.
	let phase_off = organ.pos_radius.x * 0.35 + organ.pos_radius.z * 0.21;
	let local_pulse = 0.5 + 0.5 * sin(params.loop_phase * 6.28318530718 * 2.0 + phase_off);
	var breath = 1.0 + 0.08 * local_pulse;
	if (is_center) {
		breath = 1.0 + 0.16 * params.pulse;
	}

	// Hero sprite: ~2.6x the core radius. Big enough to fill the view, small
	// enough that 19 halos stay DISTINCT instead of merging into one bloom fog
	// (3.4 washed the frame out and buried the labels).
	// ── Focus+context (hover-to-inspect nav) ──
	// hover.x = eased strength 0..1 (0 = nothing focused); hover.yzw = focused
	// organ world pos. This orb is "focused" if its world pos matches yzw.
	let strength = clamp(camera.hover.x, 0.0, 1.0);
	let hover_active = strength > 0.001;
	let focused_pos = camera.hover.yzw;
	let is_focused = hover_active && distance(organ.pos_radius.xyz, focused_pos) < 0.001;

	// Part the OTHER orbs radially away from the focused organ so the hovered one
	// gets breathing room (accordion/fisheye), eased by strength.
	var pos = organ.pos_radius.xyz;
	if (hover_active && !is_focused) {
		let delta = organ.pos_radius.xyz - focused_pos;
		let dist = max(length(delta), 0.0001);
		pos = pos + (delta / dist) * 2.6 * strength;
	}

	// Focused organ swells so it dominates the view.
	let focus_scale = select(1.0, 1.0 + 0.6 * strength, is_focused);
	let half_size = organ.pos_radius.w * 2.6 * breath * focus_scale;
	let world = pos
		+ camera.right.xyz * corner.x * half_size
		+ camera.up.xyz * corner.y * half_size;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;
	out.accent = organ.color_family.rgb;
	out.info = vec4<f32>(organ.pos_radius.w, select(0.0, 1.0, is_center), organ.color_family.w, breath);
	out.focus = select(0.0, strength, is_focused);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let is_center = in.info.y > 0.5;

	// Soft glow: hot core + TIGHTER halo (falls off faster) so orbs read as
	// distinct nodes and don't drown the frame + labels in overlapping bloom.
	let core = smoothstep(0.22, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 3.4);
	var intensity = core * 1.4 + halo * (0.30 + 0.14 * params.pulse);

	// Thin fresnel ring gives each orb a crisp planetary rim (reads as a sphere,
	// not a fuzzy dot) — brightest at the silhouette edge.
	let ring = smoothstep(0.62, 0.82, d) * (1.0 - smoothstep(0.9, 1.0, d));
	intensity = intensity + ring * 0.55;

	if (is_center) {
		intensity = intensity * 1.7;
	}

	// Focus glow: the hovered organ blazes brighter + gets a hotter rim so it
	// clearly reads as "the section you're about to enter."
	intensity = intensity * (1.0 + 1.3 * in.focus);
	intensity = intensity + ring * in.focus * 1.2;

	var color = in.accent * intensity;
	// Hovered orb picks up a white-hot center so its label reads on top of it.
	color = color + vec3<f32>(1.0, 1.0, 1.0) * core * in.focus * 0.7;

	// Center heart gets a white-hot pinpoint core — the cortex the organs orbit.
	if (is_center) {
		color = color + vec3<f32>(1.0, 1.0, 1.0) * core * 0.6;
	}

	// Portrait dim: on a phone the packed additive halos bloom into one blinding
	// white blob that swallows the lower field and steals contrast from the STATS/
	// SETTINGS labels. Fold the whole field down to a DIM backdrop when the live
	// viewport is portrait (aspect < 0.85), scaling with how narrow it is. Derived
	// purely from viewport_w/viewport_h — landscape/desktop (aspect >= 0.85) gets
	// an exact 1.0 multiplier, so the desktop render is byte-identical.
	let aspect = params.viewport_w / max(params.viewport_h, 1.0);
	let portraitness = clamp((0.85 - aspect) / (0.85 - 0.46), 0.0, 1.0);
	let portrait_dim = 1.0 - 0.62 * portraitness;

	return vec4<f32>(color * params.brightness * portrait_dim, 1.0);
}
`;

export class PalaceNodePass implements FramePass {
	private engine: ObservatoryEngine;
	private pipeline: GPURenderPipeline | null = null;
	private bindGroup: GPUBindGroup | null = null;
	private cameraBuffer: GPUBuffer | null = null;
	private nodeBuffer: GPUBuffer | null = null;
	private cameraData = new Float32Array(CAMERA_FLOATS);
	private nodeCount = 0;

	/** CPU mirror of node world positions + radius — pickAt/labels reproject these. */
	private placed: PlacedRegion[] = [];

	// ── Hover focus+context: the currently-hovered organ glows and pushes the
	// others radially away from it. hoverStrength eases 0→1 so the parting is
	// smooth, not a snap. -1 index = nothing hovered. ──
	private hoveredIndex = -1;
	private hoverStrength = 0;

	/** Set the focused organ by its buffer index (from pickAt), or -1 to clear. */
	setHovered(index: number): void {
		this.hoveredIndex = index >= 0 && index < this.placed.length ? index : -1;
	}

	/** Resolve an href to its buffer index (route calls setHovered with this). */
	indexOfHref(href: string | null): number {
		if (!href) return -1;
		return this.placed.findIndex((p) => p.href === href);
	}

	// ── Dive state (UNIT 4 portal): on click, the camera rushes THROUGH the
	// picked orb — distance collapses to ~0 and the look-target slides from the
	// origin to the orb, so it dilates open to fill the frame. When the dive
	// completes, onArrive() fires (the route navigates into that organ). ──
	private dive: { target: [number, number, number]; startMs: number; href: string } | null = null;
	private onArrive: ((href: string) => void) | null = null;
	private static readonly DIVE_MS = 620;

	constructor(engine: ObservatoryEngine) {
		this.engine = engine;
	}

	/**
	 * Begin the portal dive into the organ at `href`. The camera flies from its
	 * current orbit into that orb over DIVE_MS; `onArrive(href)` fires once at the
	 * end (the route navigates then). No-op if already diving or the href is
	 * unknown. Uses wallNowMs (a real external clock) purely for the one-shot UI
	 * transition — it never feeds simulation/capture state, so determinism holds.
	 */
	startDive(href: string, onArrive: (href: string) => void): boolean {
		if (this.dive) return false;
		const p = this.placed.find((r) => r.href === href);
		if (!p) return false;
		this.dive = { target: [p.x, p.y, p.z], startMs: this.engine.wallNowMs, href };
		this.onArrive = onArrive;
		return true;
	}

	/** True while a portal dive animation is in flight (route gates clicks on this). */
	get isDiving(): boolean {
		return this.dive !== null;
	}

	/**
	 * Lay the organs out on a golden-angle sphere shell (center organ at origin),
	 * upload them to the node storage buffer, and (re)build the pipeline. Idempotent:
	 * safe to call again to relayout.
	 */
	uploadRegions(regions: OrganRegion[]): void {
		const device = this.engine.gpuDevice;
		if (!device) return;

		const placed = this.layout(regions);
		this.placed = placed;
		this.nodeCount = placed.length;

		const data = new Float32Array(Math.max(placed.length, 1) * FLOATS_PER_NODE);
		for (let i = 0; i < placed.length; i++) {
			const p = placed[i];
			const region = regions[i];
			const [r, g, b] = rgb01(FAMILY_COLOR[region.family]);
			const base = i * FLOATS_PER_NODE;
			data[base + 0] = p.x;
			data[base + 1] = p.y;
			data[base + 2] = p.z;
			data[base + 3] = p.radius;
			data[base + 4] = r;
			data[base + 5] = g;
			data[base + 6] = b;
			data[base + 7] = FAMILY_ID[region.family];
			data[base + 8] = p.center ? 1 : 0;
			data[base + 9] = 0;
			data[base + 10] = 0;
			data[base + 11] = 0;
		}

		this.nodeBuffer?.destroy();
		this.nodeBuffer = device.createBuffer({
			label: 'palace-node-state',
			size: Math.max(data.byteLength, 64),
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.nodeBuffer, 0, data.buffer as ArrayBuffer);

		if (!this.cameraBuffer) {
			this.cameraBuffer = device.createBuffer({
				label: 'palace-camera',
				size: this.cameraData.byteLength,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
		}

		// The node count this pass draws is its OWN — it does NOT touch
		// engine.params[2] (that belongs to any shared graph pass).
		this.createPipeline(device);
	}

	/**
	 * Deterministic constellation: the center organ sits at the origin; every
	 * other organ rides a Fibonacci (golden-angle) sphere shell so they spread
	 * evenly with no clumping and no RNG. Layout is pure of any clock, so the
	 * field is capture-stable.
	 */
	private layout(regions: OrganRegion[]): PlacedRegion[] {
		const out: PlacedRegion[] = [];
		const golden = Math.PI * (3 - Math.sqrt(5)); // ~2.399963 rad

		const outer = regions.filter((r) => !r.center);
		const n = outer.length;
		let k = 0;

		for (const region of regions) {
			if (region.center) {
				out.push({ href: region.href, x: 0, y: 0, z: 0, radius: CENTER_RADIUS, center: true });
				continue;
			}
			// Fibonacci sphere point i in [0, n)
			const i = k++;
			const y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0; // 1 .. -1
			const rr = Math.sqrt(Math.max(0, 1 - y * y));
			const theta = golden * i;
			// Slight radial jitter by index (deterministic) so it reads layered,
			// not a perfect shell — a nervous system, not a beach ball.
			const shell = SHELL_RADIUS * (0.82 + 0.18 * ((i * 0.6180339887) % 1));
			out.push({
				href: region.href,
				x: Math.cos(theta) * rr * shell,
				y: y * shell * 0.82, // gently flatten Y so the orbit reads wider than tall
				z: Math.sin(theta) * rr * shell,
				radius: ORGAN_RADIUS,
				center: false
			});
		}
		return out;
	}

	private createPipeline(device: GPUDevice): void {
		if (!this.engine.paramsBuffer || !this.cameraBuffer || !this.nodeBuffer) return;

		const module = device.createShaderModule({
			label: 'palace-render-nodes',
			code: palaceNodeWGSL
		});

		this.pipeline = device.createRenderPipeline({
			label: 'palace-nodes',
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

		this.bindGroup = device.createBindGroup({
			label: 'palace-nodes-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: this.engine.paramsBuffer } },
				{ binding: 1, resource: { buffer: this.cameraBuffer } },
				{ binding: 2, resource: { buffer: this.nodeBuffer } }
			]
		});
	}

	/** Current-frame camera. Orbits at rest; DIVES through the picked orb when
	 * a portal dive is active (same math pickAt/labels reproject through). */
	private currentCamera() {
		const w = this.engine.params[6] || 1;
		const h = this.engine.params[7] || 1;
		const aspect = w / h;
		const phase = this.engine.params[1];
		const orbit = orbitCamera(phase, aspect, ORBIT_DISTANCE, ORBIT_ELEVATION);
		if (!this.dive) return orbit;

		// Dive progress 0..1 with an ease-in cubic (accelerate INTO the orb).
		const raw = Math.min(1, (this.engine.wallNowMs - this.dive.startMs) / PalaceNodePass.DIVE_MS);
		const t = raw * raw * raw; // ease-in: slow start, rushing finish
		// Eye rushes from its orbit position toward the target orb; look-target
		// slides from origin to the orb. At t→1 the eye is nearly AT the orb and
		// looking at it, so the orb fills the frame (portal open).
		const eye: [number, number, number] = [
			orbit.eye[0] + (this.dive.target[0] - orbit.eye[0]) * t * 0.985,
			orbit.eye[1] + (this.dive.target[1] - orbit.eye[1]) * t * 0.985,
			orbit.eye[2] + (this.dive.target[2] - orbit.eye[2]) * t * 0.985
		];
		const target: [number, number, number] = [
			this.dive.target[0] * t,
			this.dive.target[1] * t,
			this.dive.target[2] * t
		];
		// Narrow the FOV slightly as we dive for a subtle dolly-zoom "warp".
		const fov = ((50 - 8 * t) * Math.PI) / 180;
		const proj = perspective(fov, aspect, 0.05, 4000);
		const view = lookAt(eye, target, [0, 1, 0]);
		const viewProj = multiply(proj, view);

		// Fire arrival exactly once, at the end of the dive.
		if (raw >= 1 && this.onArrive) {
			const cb = this.onArrive;
			const href = this.dive.href;
			this.onArrive = null;
			this.dive = null;
			// defer so we don't navigate mid-frame-encode
			queueMicrotask(() => cb(href));
		}

		// Reuse orbit basis for billboards (fine — orbs stay camera-facing).
		return { viewProj, right: orbit.right, up: orbit.up, eye };
	}

	/** FramePass — write the deterministic close-orbit camera for this frame. */
	compute(): void {
		const device = this.engine.gpuDevice;
		if (!device || !this.cameraBuffer) return;

		const cam = this.currentCamera();
		this.cameraData.set(cam.viewProj, 0);
		this.cameraData[16] = cam.right[0];
		this.cameraData[17] = cam.right[1];
		this.cameraData[18] = cam.right[2];
		this.cameraData[19] = 0;
		this.cameraData[20] = cam.up[0];
		this.cameraData[21] = cam.up[1];
		this.cameraData[22] = cam.up[2];
		this.cameraData[23] = 0;

		// Ease the hover strength toward 1 when an organ is focused, 0 otherwise —
		// smooth part/return, not a snap. (Frame-rate-independent-ish: ~0.18/frame.)
		const targetStrength = this.hoveredIndex >= 0 ? 1 : 0;
		this.hoverStrength += (targetStrength - this.hoverStrength) * 0.18;
		if (this.hoverStrength < 0.001) this.hoverStrength = 0;
		const h = this.hoveredIndex >= 0 ? this.placed[this.hoveredIndex] : null;
		// hover lane: x = eased strength 0..1 (0 = nothing focused), yzw = focused
		// organ world pos. The shader compares each orb's pos to yzw to find the
		// focused one, parts the rest away from it, and swells + glows it.
		this.cameraData[24] = h ? this.hoverStrength : 0;
		this.cameraData[25] = h ? h.x : 0;
		this.cameraData[26] = h ? h.y : 0;
		this.cameraData[27] = h ? h.z : 0;

		device.queue.writeBuffer(this.cameraBuffer, 0, this.cameraData);
	}

	/** FramePass — one instanced additive draw of all organ billboards. */
	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || this.nodeCount === 0) return;
		// This pass draws its OWN node count — set params[2] so the shader's
		// instance guard matches. It is the only node pass on the palace route.
		this.engine.params[2] = this.nodeCount;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.draw(6, this.nodeCount);
	}

	/**
	 * CPU pick — no GPU readback. Node world positions are static once uploaded,
	 * so we reproject each through the CURRENT frame's viewProj (the same camera
	 * compute() wrote) and return the nearest organ whose projected disc contains
	 * the click. Returns the organ href.
	 */
	pickAt(ndcX: number, ndcY: number): { index: number; href: string } | null {
		if (this.nodeCount === 0) return null;
		const m = this.currentCamera().viewProj; // column-major
		// fovY 50° → f = 1/tan(25°); world radius r at clip-w projects to ~r·f/w in NDC.
		const f = 1 / Math.tan((50 * Math.PI) / 360);
		const w = this.engine.params[6] || 1;
		const h = this.engine.params[7] || 1;
		const aspect = w / h;

		let best = -1;
		let bestScore = Infinity;
		for (let i = 0; i < this.placed.length; i++) {
			const p = this.placed[i];
			const cw = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
			if (cw <= 0) continue; // behind the camera
			const cx = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / cw;
			const cy = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / cw;
			// projected radius in NDC-y; divide x-distance by aspect so the disc
			// stays circular in NDC space (viewport is wider than tall).
			const projR = Math.max((p.radius * f) / cw, 0.02);
			const dx = (cx - ndcX) / aspect;
			const dy = cy - ndcY;
			let score = Math.hypot(dx, dy) / projR;
			// Hysteresis: the ALREADY-hovered orb gets a discount so the cursor
			// must move clearly onto another orb to switch — kills flicker between
			// adjacent organs (the "too sensitive" problem).
			if (i === this.hoveredIndex) score *= 0.75;
			// HIT_RADIUS = 0.58 → must be near the orb's CORE, not anywhere in its
			// wide glow. The orbs are large + packed, so a loose radius made ~76% of
			// the frame "hover something" (grabby). Tight = a deliberate, one-orb hover.
			if (score < 0.5 && score < bestScore) {
				bestScore = score;
				best = i;
			}
		}
		if (best < 0) return null;
		return { index: best, href: this.placed[best].href };
	}

	/**
	 * Project every organ's world position through the current viewProj so the
	 * MSDF text layer can float each label ON its node. NDC coords match what the
	 * frame drew (same camera). Behind-camera nodes report visible=false.
	 */
	getScreenPositions(): PalaceScreenPos[] {
		const m = this.currentCamera().viewProj;
		const out: PalaceScreenPos[] = [];
		for (const p of this.placed) {
			const cw = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
			if (cw <= 0) {
				out.push({ href: p.href, ndcX: 0, ndcY: 0, depth: cw, visible: false });
				continue;
			}
			const cx = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) / cw;
			const cy = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) / cw;
			out.push({ href: p.href, ndcX: cx, ndcY: cy, depth: cw, visible: true });
		}
		return out;
	}

	dispose(): void {
		this.nodeBuffer?.destroy();
		this.cameraBuffer?.destroy();
		this.nodeBuffer = null;
		this.cameraBuffer = null;
		this.pipeline = null;
		this.bindGroup = null;
		this.nodeCount = 0;
		this.placed = [];
	}
}

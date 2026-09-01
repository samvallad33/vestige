/**
 * PalaceBrainPass — the navigable Memory Palace rendered as the REAL vestige-pro
 * waitlist brain engine. Nine living organ constellations, each formed by a real
 * strange-attractor / formation from the launch engine (brain, fibonacci graph,
 * memory lattice, 4D Clifford torus, receipt lanes, Thomas attractor, Aizawa
 * chaotic attractor, archive columns, superformula crystal), rendered with the
 * launch engine's 5-stop iridescent spectrum + energy blaze, on the shared
 * Observatory HDR/bloom engine.
 *
 * ARCHITECTURE (deliberately the proven, launch-safe path): formation geometry is
 * baked ONCE on upload into a per-instance vertex buffer (NO per-frame compute, no
 * per-vertex attractor integration — that stalls the first frame for seconds). The
 * vertex shader only animates (living drift + per-organ breath + cortex heartbeat +
 * cognitive weather), projects to the organ's screen anchor, and colors via the
 * real palette(). One instanced additive draw into engine.sceneFormat; PostChain
 * bloom carries the brightness.
 *
 * Navigation is byte-identical to PalaceSwarmPass: pickAt/getScreenPositions test
 * the SAME DESKTOP_ANCHORS (screen-space, aspect-corrected), and the click fires the
 * launch engine's singularity-collapse -> supernova as the route-enter burst.
 * Public API matches PalaceSwarmPass so palace/+page.svelte works unchanged.
 *
 * Engine internals sourced from Vestige memory 55b69e52 (VESTIGE LAUNCH ENGINE):
 * per-particle RANDOM integration length fills attractor curves into volumes;
 * transition = free-vortex inhale -> white flash -> forced-vortex supernova with
 * env=sin(m*PI) so every term vanishes at the endpoints (orbs land EXACTLY on shape).
 */

import type { FramePass, ObservatoryEngine } from './engine';
import { BITEMPORAL, IMMUNE, RETENTION, rgb01 } from './cognitive-palette';
import type { OrganRegion } from './palace-map';

const FLOATS_PER_PARTICLE = 12;
const UNIFORM_FLOATS = 16;
// ~1s snappy transition (Sam: don't waste the user's time): fly to CENTER ->
// collapse -> singularity flash -> the shape-specific EXPLOSION -> navigate. There
// is NO veil; the destination appears under the last of the debris. Nav fires LATE
// (0.90) so the detonation is fully visible on the Palace before the silent handoff.
const BURST_MS = 1000;
const REDUCED_BURST_MS = 200;
const FLASH_REQUEST_AT = 0.9;

const FAMILY_COLOR: Record<OrganRegion['family'], string> = {
	reasoning: RETENTION.bridge,
	memory: RETENTION.recall,
	immune: IMMUNE.veto,
	signal: BITEMPORAL.supersession,
	temporal: BITEMPORAL.txShadow,
	system: RETENTION.luciferin
};

// Nine bespoke organ shapes (art-director fleet wf_81764e70-be2). Each is a
// distinct form that MEANS its organ (the discipline test), built from real math
// and orientation-verified so none collapses to a flat bar under Y-rotation.
const FORMATION_KIND: Record<string, number> = {
	'/observatory': 0, // FOLDED CORTEX SHELL — a breathing gyrified brain (the mind at rest)
	'/graph': 1, // HOPF FIBRATION — nested tori of interlinked rings (everything connected)
	'/memories': 2, // CONSTELLATION ARCHIVE — Fibonacci shells of discrete memory-pearls
	'/timeline': 3, // BITEMPORAL CLIFFORD TORUS — two woven time-clocks (valid vs transaction)
	'/blackbox': 4, // SEALED VAULT — a gem sealed in a cubic crystal cage (the receipt)
	'/reasoning': 5, // AIZAWA CONVERGENCE — chaos converging onto a spine (a conclusion forming)
	'/explore': 6, // SEMANTIC WALK — a seed branching into dendritic neighborhoods
	'/feed': 7, // VORTEX RING — a circulating smoke-ring (the live event stream)
	'/contradictions': 8 // STELLA OCTANGULA — two interpenetrating dual tetrahedra (opposition)
};

// Palace composition: the cortex (Observatory) sits at center; the 8 outer organs
// are spread on a WIDE ring around it so each shape reads clearly and the axons
// have room to flow out without crossing through anything. A wide ellipse (rx>ry)
// uses the 16:10 canvas. Each organ sits at its own angle, alternating radius a
// touch so neighbors never touch. [x, y, z, scale] in logical NDC. pickAt tests
// these exact anchors, so navigation stays correct.
const DESKTOP_ANCHORS: Record<string, [number, number, number, number]> = {
	'/observatory': [0.0, 0.0, 0.12, 0.26], // center cortex, largest
	'/graph': [-0.78, 0.5, 0.04, 0.15], // upper-left
	'/memories': [0.8, 0.46, -0.04, 0.15], // upper-right
	'/timeline': [-0.9, -0.06, 0.1, 0.14], // left
	'/reasoning': [0.92, -0.02, -0.08, 0.15], // right
	'/blackbox': [-0.72, -0.56, 0.18, 0.14], // lower-left
	'/feed': [0.74, -0.54, 0.02, 0.14], // lower-right
	'/explore': [-0.02, 0.66, -0.13, 0.14], // top
	'/contradictions': [0.0, -0.66, -0.02, 0.14] // bottom
};

const CORTEX: [number, number] = [0.0, 0.0];

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

// ─────────────────────────────────────────────────────────────────────────────
// WGSL — the launch engine's render, adapted for 9 anchored organs. Geometry is
// baked (attribute `baked`); the vertex shader animates + projects + colors with
// the real spectrum()/palette(). NO backticks in any comment (this is a TS
// template literal; svelte-check parses it as TS and a backtick would close it).
// ─────────────────────────────────────────────────────────────────────────────
const palaceBrainWGSL = /* wgsl */ `
struct Uniforms {
	viewport: vec4<f32>,     // x=w y=h z=simTime w=reducedMotion
	interaction: vec4<f32>,  // x=hoverIndex y=hoverStrength z=selectedIndex w=burstProgress
	portal: vec4<f32>,       // x=portalX y=portalY z=unused w=flash
	intro: vec4<f32>,        // x=introProgress y=count z=pulse w=unused
};

@group(0) @binding(0) var<uniform> u: Uniforms;

const PI = 3.14159265359;
const TAU = 6.28318530718;

fn rot2(p: vec2<f32>, a: f32) -> vec2<f32> {
	let c = cos(a);
	let s = sin(a);
	return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn hash11(x: f32) -> f32 {
	return fract(sin(x * 12.9898 + 78.233) * 43758.5453);
}

// Fossil-graded family spectrum: sediment → amber → jade → cyan → luciferin.
// Magenta is reserved for RSB and must never paint the Palace home.
fn spectrum(s: f32) -> vec3<f32> {
	let sediment = vec3<f32>(0.07, 0.08, 0.04);
	let amber    = vec3<f32>(0.96, 0.62, 0.16);
	let jade     = vec3<f32>(0.16, 0.95, 0.66);
	let cyan     = vec3<f32>(0.13, 0.78, 0.87);
	let chalk    = vec3<f32>(0.91, 1.00, 0.72);
	let x = fract(s) * 5.0;
	if (x < 1.0) { return mix(sediment, amber, x); }
	if (x < 2.0) { return mix(amber, jade, x - 1.0); }
	if (x < 3.0) { return mix(jade, cyan, x - 2.0); }
	if (x < 4.0) { return mix(cyan, chalk, x - 3.0); }
	return mix(chalk, sediment, x - 4.0);
}

// Living color: the hue wheel rotates over time, traveling waves ripple across
// the cloud by world position, each node pulses, and high-energy nodes flare
// gold then white-hot so bursts read like fireworks. From the launch engine.
fn palette(seed: f32, energy: f32, world: vec3<f32>, t: f32) -> vec3<f32> {
	let drift = t * 0.06;
	let wave = sin(world.x * 0.9 - t * 0.8) * 0.10 + cos(world.y * 0.8 + t * 0.6) * 0.10;
	var col = spectrum(seed + drift + wave);
	// Punch saturation so the iridescence reads vivid (push away from the per-channel
	// mean) before the per-node pulse.
	let luma = dot(col, vec3<f32>(0.299, 0.587, 0.114));
	col = clamp(mix(vec3<f32>(luma), col, 1.35), vec3<f32>(0.0), vec3<f32>(1.0));
	let pulse = 0.86 + 0.14 * sin(t * 2.2 + seed * 30.0);
	col = col * pulse;
	let gold = vec3<f32>(1.00, 0.78, 0.36);
	col = mix(col, gold, clamp((energy - 2.0) * 0.5, 0.0, 0.6));
	col = mix(col, vec3<f32>(1.0), clamp((energy - 3.2) * 0.5, 0.0, 0.6));
	return col;
}

struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) local: vec2<f32>,
	@location(1) color: vec3<f32>,
	@location(2) energy: f32,
};

const QUAD = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

@vertex
fn vs_main(
	@location(0) baked: vec4<f32>,       // xyz local formation point, w = size seed
	@location(1) anchor: vec4<f32>,      // xy anchor NDC, z depth, w scale
	@location(2) color_route: vec4<f32>, // rgb hue seed base color, w route index
	@builtin(vertex_index) vertex_index: u32,
	@builtin(instance_index) instance_index: u32
) -> VSOut {
	let id = f32(instance_index + 1u);
	let route = color_route.w;
	let aspect = u.viewport.x / max(u.viewport.y, 1.0);
	let t = u.viewport.z;
	let reduced = u.viewport.w > 0.5;
	let seed0 = fract(id * 0.61803398875);
	let seed1 = fract(id * 0.41421356237);
	let seed2 = fract(id * 0.73205080757);

	var local3 = baked.xyz;
	var organ_anchor = anchor.xy;
	let scale = anchor.w;
	var axon = 0.0;

	if (!reduced) {
		// ── Whole-cloud slow rotation so each formation reads volumetrically ──
		let rxz = rot2(local3.xz, t * 0.05);
		local3 = vec3<f32>(rxz.x, local3.y, rxz.y);

		// ── Organ breathing: each silhouette inflates on its own slow phase ──
		let breath_phase = route * 2.111;
		let breath = sin(t * 0.9 + breath_phase) * 0.6 + sin(t * 1.8 + breath_phase * 1.7) * 0.4;
		local3 *= 1.0 + 0.035 * breath;

		// ── Microscopic circulation so no orb is ever frozen ──
		local3 += vec3<f32>(
			cos(t * 2.0 + seed0 * TAU),
			sin(t * 2.0 + seed1 * TAU),
			cos(t * 1.7 + seed2 * TAU)
		) * (0.004 + seed2 * 0.005);
	}

	// Project the local formation point to the organ's screen anchor.
	var pos = organ_anchor + vec2<f32>(local3.x / aspect, local3.y) * scale;

	// ── AXONS: a DELICATE signal stream flowing FROM the cortex OUT to each organ.
	// Only ~4.5% of an outer organ's particles ride the fiber, and the fiber runs
	// only through the OPEN SPACE between the two bodies (starts just outside the
	// cortex, ends just before the organ) so it never crosses through a shape. Each
	// particle is a small point flowing outward; sparse bright packets pulse along
	// it. Thin, single-file (tiny normal jitter, no fat rope), tapering to the organ.
	if (!reduced && route > 0.5 && seed0 < 0.05) {
		let cortexP = vec2<f32>(${CORTEX[0]}, ${CORTEX[1]});
		let dir = normalize(organ_anchor - cortexP + vec2<f32>(0.00001, 0.0));
		let normal = vec2<f32>(-dir.y, dir.x);
		// The thread spans the FULL distance cortex -> organ, so it visibly connects
		// the brain to the destination (no floating gap). Particles are distributed
		// along the whole line (by seed) and drift outward slowly; brightness fades
		// at BOTH ends so it emerges from the brain and arrives at the organ without a
		// hard dot piling onto either body.
		let base = seed1;                          // static position along the line 0..1
		let travel = fract(base + t * 0.05);       // slow outward drift
		// gentle single curve across the whole span, tiny amplitude = one clean thread.
		let bow = sin(travel * PI) * 0.03;
		let jitter = (seed2 - 0.5) * 0.006;
		pos = mix(cortexP, organ_anchor, travel) + normal * (bow + jitter);
		// bright nerve packets sweeping outward along the fiber.
		let packet = pow(0.5 + 0.5 * sin(travel * TAU * 2.0 - t * 2.6 + seed2 * TAU), 10.0);
		// fade near BOTH endpoints (0 at ends, full in the middle) so the line reads
		// as connecting, not as dots stacked on the bodies.
		let endsFade = smoothstep(0.0, 0.14, travel) * (1.0 - smoothstep(0.86, 1.0, travel));
		axon = (0.35 + packet * 1.2) * endsFade + 0.15;
		local3 = vec3<f32>(0.0, 0.0, 0.0);
	}

	// ── Hover focus+context ──
	let hover_on = u.interaction.x >= 0.0 && u.interaction.y > 0.001;
	let hovered = abs(route - u.interaction.x) < 0.25;
	let focus = select(0.0, u.interaction.y, hovered);
	if (hovered) { pos = mix(pos, organ_anchor + (pos - organ_anchor) * 0.84, focus); }

	// ── Intro: the whole atlas unfolds out of the cortex singularity ──
	let intro = select(smoothstep(0.05, 0.9, u.intro.x), 1.0, reduced);
	let cortex = vec2<f32>(${CORTEX[0]}, ${CORTEX[1]});
	if (intro < 0.999) {
		let rel = rot2(pos - cortex, (1.0 - intro) * (2.0 + seed0 * 3.0) * PI);
		pos = mix(cortex + rel * 0.04, pos, intro)
			+ normalize(pos - cortex + vec2<f32>(0.00001, 0.0)) * sin(intro * PI) * 0.075;
	}

	// ── Cortex heartbeat: the pacemaker fires ~every 2.4s, launching an expanding
	// brightness ring OUTWARD across the whole field (brightness only, no drift). ──
	var systole = 0.0;
	if (!reduced) {
		let beat = fract(t / 2.4);
		let contraction = exp(-beat * 5.5) * smoothstep(0.0, 0.06, beat);
		let d = length(pos - cortex);
		let wavefront = beat * 2.2;
		let ring = exp(-pow((d - wavefront) / 0.12, 2.0));
		systole = contraction * (ring * 1.4 + select(0.0, 1.2, d < 0.18));
	}

	// ── Cognitive weather: every ~7s the whole organism draws one slow breath
	// (dim + a hair toward cortex), then re-ignites in a rippling surge. ──
	var weather = 0.0;
	if (!reduced) {
		let wp = fract(t / 7.0);
		let inhale = smoothstep(0.0, 0.4, wp) * (1.0 - smoothstep(0.4, 0.55, wp));
		let ignite = smoothstep(0.55, 0.72, wp) * (1.0 - smoothstep(0.72, 1.0, wp));
		let d = length(pos - cortex);
		pos = mix(pos, cortex, inhale * 0.04 * (1.0 - d));
		weather = ignite * exp(-pow((d - fract(t / 7.0 - 0.55) * 2.4) / 0.18, 2.0)) * 1.3 - inhale * 0.25;
	}

	// ── Click: a ~2s CINEMATIC DIVE the user can actually watch. Three acts over
	// progress 0..1:
	//   DIVE     0.00-0.42  the clicked organ rushes toward the camera (scale up +
	//                       everything sucked toward the portal) - flying INTO its
	//                       stratosphere. The selected organ blooms; the rest streak.
	//   COLLAPSE 0.42-0.66  free-vortex spiral into a white-hot singularity (flash;
    //                       navigation fires behind the flash so there is no cut).
	//   MORPH    0.66-1.00  supernova detonation outward that reforms as the field
	//                       dissolves into the destination organ (the waitlist's
	//                       shape-to-shape morph, now the route transition itself).
	let progress = u.interaction.w;
	let bursting = u.interaction.z >= 0.0 && progress > 0.0;
	let selected = abs(route - u.interaction.z) < 0.25;
	var flash = 0.0;
	if (bursting) {
		if (reduced) {
			pos = mix(pos, u.portal.xy, smoothstep(0.0, 1.0, progress));
		} else {
			// ACT 1 - DIVE: the whole field is pulled to SCREEN CENTER; the selected
			// organ grows a little as it arrives (camera flies toward it) but is
			// clamped so its particles never shoot off the frame edge — the portal is
			// already animating anchor -> (0,0) on the CPU, so we lerp every particle
			// toward center, then apply a bounded magnify AROUND that center.
			let dive = smoothstep(0.0, 0.42, progress);
			let center = vec2<f32>(0.0, 0.0);
			// glide the particle from its live position to the portal (which is heading
			// to center), so by end-of-dive the whole organ sits centered.
			let glided = mix(pos, u.portal.xy, dive);
			// gentle magnify about center for the selected organ (bounded: max ~1.6x,
			// not 3.6x), others shrink slightly inward.
			let mag = select(0.62, 1.0 + dive * 0.6, selected);
			var dived = center + (glided - center) * mag;
			// hard clamp so NO particle can leave the frame during the dive.
			let dr = length(dived);
			if (dr > 1.25) { dived = dived * (1.25 / dr); }
			pos = dived;

			// ACT 2 - COLLAPSE: free-vortex spiral crush to the singularity.
			let inhale = smoothstep(0.42, 0.66, progress);
			let rel = rot2(pos - u.portal.xy, inhale * (3.5 + seed0 * 2.5) * PI);
			pos = u.portal.xy + rel * pow(max(1.0 - inhale, 0.0), 1.35);
			flash = exp(-pow((progress - 0.66) / 0.05, 2.0)); // razor white-out at the core

			// ACT 3 - MORPH: forced-vortex detonation outward; the veil reveals the
			// destination during this window so the debris reads as reforming into it.
			// Per-organ signature burst: only the SELECTED organ's branch runs
			// (u.interaction.z = selected kind). Shared exhale computed ONCE; each
			// branch reads baked.xyz + seeds and SETS pos (+ may add to flash).
			let exhale = smoothstep(0.66, 1.0, progress);
			if (abs(u.interaction.z - 0.0) < 0.5) {
				// OBSERVATORY (kind 0) - SYNAPTIC CLEAVE IGNITION.
				// One decisive pulse: a neural ignition front fires nucleus -> rim, the two
				// hemispheres wrench apart along the fissure (baked.x sign), everything snaps
				// to a HARD contained stop, then the gyri crumble inward to thought-dust.
				// Uses exhale (already declared). Sets pos. Clamped to <=1.2 radius.
				let ex = exhale;
				// Punchy nonlinear time: fast open (detonation), then settle.
				let pop = 1.0 - pow(1.0 - ex, 2.4);          // 0->1, snappy leading edge
				let settle = smoothstep(0.55, 1.0, ex);       // late crumble-in

				// --- baked geometry read ---
				let rb = length(baked.xyz);                   // ~0.2 nucleus .. ~0.92 cortex rim
				let hemiSign = select(-1.0, 1.0, baked.x >= 0.0);
				let nrm = normalize(baked.xy + vec2<f32>(0.00003, 0.00001)); // outward cortical facing

				// --- (1) HEMISPHERE CLEAVE along the mid-sagittal fissure (screen-x) ---
				// Bounded lateral wrench. Seam particles (baked.x~0) tear hardest, rim less.
				let seamCloseness = exp(-(baked.x * baked.x) / 0.05);
				let cleaveMag = pop * (0.30 + seed0 * 0.10) * (0.6 + 0.5 * seamCloseness);
				let cleave = vec2<f32>(hemiSign, 0.0) * cleaveMag;

				// --- (2) IGNITION FRONT: shell-by-shell outward peel, nucleus -> rim ---
				// A wave sweeps rb from 0 to ~0.95; a particle only launches once passed.
				let front = pop * 1.0;
				let fired = smoothstep(front - 0.12, front + 0.03, rb); // 1 after wave crosses
				// Outward peel along its own normal, BOUNDED. Deep particles peel more (nucleus blast).
				let peelMag = fired * pop * (0.34 + seed1 * 0.22) * (1.0 - 0.35 * rb);
				// Living swirl so folds unfurl rather than a clean radial pop.
				let peel = rot2(nrm, (seed2 - 0.5) * 1.1 * pop) * peelMag;

				// --- (3) CRUMBLE INWARD to thought-dust (late) ---
				// Instead of flying further out, debris drifts back toward center + jitters:
				// keeps it focal and contained, reads as the mind dissolving inward.
				let inward = -normalize(peel + vec2<f32>(seed0 - 0.5, seed1 - 0.5) * 0.001);
				let jitter = vec2<f32>(hash11(seed0 * 91.7 + rb) - 0.5, hash11(seed1 * 53.3 + rb) - 0.5);
				let crumble = (inward * 0.14 + jitter * 0.10) * settle;

				// --- compose from the singularity anchor ---
				let off = cleave + peel + crumble;
				let burstPos = u.portal.xy + off;
				pos = mix(pos, burstPos, smoothstep(0.0, 0.7, ex));

				// --- HARD CONTAINMENT: nothing past 1.2 NDC radius ---
				let rr = length(pos - u.portal.xy);
				if (rr > 1.2) { pos = u.portal.xy + (pos - u.portal.xy) * (1.2 / rr); }

				// --- FLASH: white-hot core + traveling ignition shell + cleaving seam ---
				// 1. Instant core detonation at the singularity, fast decay.
				let coreGlow = exp(-rr * rr * 7.0) * (1.0 - smoothstep(0.0, 0.42, ex));
				// 2. Traveling neural cascade: glow only in the thin shell the front crosses.
				let waveGlow = exp(-pow((rb - front) / 0.08, 2.0)) * (1.0 - settle);
				// 3. Fissure seam flares white as the hemispheres part (peaks mid-burst).
				let seamGlow = seamCloseness * exp(-pow((ex - 0.28) / 0.14, 2.0));
				let fade = 1.0 - smoothstep(0.62, 1.0, ex);
				flash += (coreGlow * 2.0 + waveGlow * 1.5 + seamGlow * 1.0) * fade;
			} else if (abs(u.interaction.z - 1.0) < 0.5) {
				// GRAPH burst — THE LINK SNAP. Hopf fibration of 12 interlinked rings; on click
				// the network first pulls TAUT toward the root (a contained implosion, links
				// tightening), then the rings UNLINK in a sharp outward causal-lightning snap:
				// each particle whips along its own ring's screen tangent so loops spring open as
				// bounded arcs (not a diffuse radial spray), bolts staggered per-ring so causality
				// fires edge by edge, a hot shockwave crest riding the leading edge. Then it
				// disperses and hard-clamps inside the frame. Contained, punchy, focal, fast.
				let g_rel = pos - u.portal.xy;                       // this node relative to the root
				let g_r0 = length(g_rel) + 1e-4;                     // its live radius from the root
				// Per-ring identity: baked ring angle + a hashed ring id so bolts stagger like edges.
				let g_ang = atan2(baked.y, baked.x);                 // where the node sits around its ring
				let g_ringId = hash11(seed0 * 41.0 + floor(g_ang * 1.9) * 7.0);
				// Ring tangent in screen space: perpendicular to the radial-from-root direction,
				// so each particle whips ALONG its loop -> rings read as circles springing open.
				let g_radial = g_rel / g_r0;
				let g_tang = vec2<f32>(-g_radial.y, g_radial.x);
				// Blend outward-radial with the ring tangent: the unlink is mostly an arc-whip,
				// with enough radial punch to read as a focal detonation from the core.
				let g_side = select(-1.0, 1.0, seed1 > 0.5);
				let g_dir = normalize(g_radial * 0.62 + g_tang * g_side * 0.9 + vec2<f32>(1e-5, 0.0));

				// BEAT 1 - GATHER (implosion): links go taut, the whole net tightens to the root.
				let g_gather = smoothstep(0.0, 0.28, exhale);
				let g_taut = mix(1.0, 0.30, g_gather * (1.0 - smoothstep(0.28, 0.5, exhale)));

				// BEAT 2 - SNAP (causal-lightning): staggered per-ring release fires bolts edge by
				// edge. releaseFront sweeps outward; inner rings snap a touch earlier.
				let g_stagger = g_ringId * 0.22 + (1.0 - g_r0 / 1.3) * 0.10;
				let g_snap = smoothstep(0.26 + g_stagger, 0.66 + g_stagger, exhale);
				// Overshoot then settle: fast leading edge, quick decel (reads as ONE decisive event).
				let g_ease = g_snap * (1.35 - 0.35 * g_snap);
				let g_reach = g_ease * (0.55 + g_ringId * 0.55 + seed2 * 0.28);
				// A slight curl so each arc bends like a loop peeling, not a dead-straight ray.
				let g_curl = (g_ringId - 0.5) * 1.1 * g_snap;
				let g_head = rot2(g_dir, g_curl);

				// COMPOSE: gather the node in, then whip it out along its unlinking arc.
				var g_pos = u.portal.xy + g_radial * (g_r0 * g_taut) + g_head * g_reach;

				// BEAT 3 - CONTAIN: hard clamp so no shard ever leaves the frame.
				let g_rr = length(g_pos - u.portal.xy);
				if (g_rr > 1.25) { g_pos = u.portal.xy + (g_pos - u.portal.xy) * (1.25 / g_rr); }
				pos = g_pos;

				// FLASH: a white-hot core building through the gather, a razor bolt-flash at the
				// per-ring snap instant, and a shockwave crest riding the leading edge of the debris.
				let g_core = g_gather * (1.0 - smoothstep(0.28, 0.5, exhale)) * 3.0;
				let g_bolt = exp(-pow((exhale - (0.34 + g_stagger)) / 0.06, 2.0)) * (2.0 + g_ringId * 2.0);
				let g_flung = length(pos - u.portal.xy);
				let g_crest = exp(-pow((g_flung - exhale * 1.15) / 0.16, 2.0)) * 2.4;
				let g_fade = 1.0 - smoothstep(0.72, 1.0, exhale);
				flash = flash + (g_core + g_bolt + g_crest) * g_fade;
			} else if (abs(u.interaction.z - 2.0) < 0.5) {
				// MEMORIES burst: ARCHIVE SHOCKWAVE. baked.xyz is a pearl grain on one of 3
				// Fibonacci shells (len ~0.57 inner / ~0.91 mid / ~1.25 outer, + tiny cluster
				// jitter). The whole constellation detonates outward from the singularity as
				// ONE focal shockwave: each pearl flies along its OWN baked radial (shape-true),
				// outer shell leads by a hair (one ring, not three staggered waves), a hot
				// white core, a quick fade. Contained to 1.25 NDC. Grains of a pearl share the
				// pearl's direction so it reads as pearls, not fog.
				let mem_e = exhale;                                    // 0..1 local burst

				// Shell rank from the baked radius: 0 inner .. 1 outer. Drives the tiny lead so
				// the blast reads as a single expanding ring rather than an even mush.
				let mem_len = length(baked.xyz) + 1e-4;
				let mem_shell = clamp((mem_len - 0.50) / 0.80, 0.0, 1.0);

				// The pearl's coherent escape direction = its own screen-projected position on
				// the shell. Every grain of one pearl shares this (baked is the cell point + a
				// sub-pearl jitter far smaller than the shell), so the clump streaks together.
				var mem_dir = normalize(baked.xy + vec2<f32>(0.00001, 0.0));
				if (length(baked.xy) < 0.02) {
				  let a = hash11(mem_len * 37.3 + seed0 * 5.1) * TAU;
				  mem_dir = vec2<f32>(cos(a), sin(a));
				}

				// LAUNCH: outer shell fires first by a HAIR (0.10), not a big horizon, so the
				// three shells read as one shockwave. pow(1.7) hangs a beat then rushes = punchy.
				let mem_lead = (1.0 - mem_shell) * 0.10;               // inner released slightly later
				let mem_t = smoothstep(mem_lead, mem_lead + 0.62, mem_e);
				let mem_fast = pow(mem_t, 1.7);                        // ease-in: hold then blast out

				// A small, shrinking curl so the ring spins out (spiral galaxy) instead of a
				// dead star; per-pearl sign+magnitude from a stable pearl hash.
				let mem_hash = hash11(dot(floor(baked.xyz * 6.0), vec3<f32>(12.9, 71.7, 131.9)) + 3.1);
				let mem_spin = select(-1.0, 1.0, mem_hash > 0.5);
				let mem_curl = mem_spin * (0.22 + mem_hash * 0.30) * mem_fast * (1.0 - mem_fast * 0.4);
				let mem_head = rot2(mem_dir, mem_curl);

				// REACH: bounded. Base + a little shell bias (outer travels a touch farther) +
				// small per-particle scatter. Times the accelerating launch. Kept well under 1.25.
				let mem_reach = (0.86 + mem_shell * 0.30 + seed1 * 0.14) * mem_fast;

				// Tight per-grain sparkle so each pearl reads as a cluster of sparks, not a dot,
				// and it stays small so the pearl holds its identity.
				let mem_spark = (vec2<f32>(seed1, seed2) - 0.5) * (0.02 + baked.w * 0.03) * (0.35 + mem_fast);

				let mem_burstPos = u.portal.xy + mem_head * mem_reach + mem_spark;
				pos = mix(u.portal.xy, mem_burstPos, mem_t);

				// HARD CONTAINMENT: nothing past 1.25 NDC of the singularity.
				let mem_rr = length(pos - u.portal.xy);
				if (mem_rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / mem_rr); }

				// FLASH: (1) one bright WHITE-HOT core at the detonation instant (gaussian at
				// mem_e~0.16, brightest at the center so it reads as a focal blast); (2) a shell
				// shockwave front lighting each ring as it launches (rolls outward); (3) a hot
				// leading edge on the fastest debris that fades as it slows.
				let mem_core = exp(-pow((mem_e - 0.16) / 0.075, 2.0)) * (1.6 + (1.0 - mem_shell) * 1.4);
				let mem_front = exp(-pow((mem_t - 0.42) / 0.16, 2.0)) * (0.5 + mem_shell * 0.8);
				let mem_edge = mem_fast * (1.0 - mem_fast) * 3.4 * (0.5 + mem_hash * 0.6);
				flash = flash + mem_core + mem_front + mem_edge;
			} else if (abs(u.interaction.z - 3.0) < 0.5) {
				// TIMELINE burst - The Ravel Snap: the 3:2 bitemporal braid unwinds.
				// The two woven time-circles (valid-time vs transaction-time) split, counter-
				// spin, peel to opposite sides, then the whole weave whips back to a singularity.
				// baked.z sign labels which strand this particle belongs to (the woven pair).
				let tl_e = exhale;                                   // 0..1 burst progress
				let tl_chan = select(-1.0, 1.0, baked.z >= 0.0);      // +1 valid-time, -1 transaction-time strand
				// Fault seam: small |baked.z| = deep in the contested overlap where the weave tears.
				let tl_seam = 1.0 - smoothstep(0.0, 0.34, abs(baked.z));
				// Recover the collapse-frame polar coords (pos already sits near center).
				let tl_rel = pos - u.portal.xy;
				let tl_r0 = length(tl_rel) + 1e-4;
				let tl_ang = atan2(tl_rel.y, tl_rel.x);
				// --- DECOUPLE (0..0.34): the 3:2 lock releases, strands gain OPPOSITE angular
				// velocity. Ease-in (tl_e^2) so it reads as a sudden unlatch, not a drift.
				let tl_spin = tl_chan * (2.0 + seed0 * 1.3) * PI * (tl_e * tl_e);
				// --- PEEL (0.20..0.62): the two rings slide to opposite sides of the core along
				// a fixed split axis. Bounded, ease-out. Seam material leads the tear.
				let tl_peelT = smoothstep(0.18, 0.62, tl_e);
				let tl_peel = tl_chan * tl_peelT * (0.30 + tl_seam * 0.14 + seed1 * 0.10);
				let tl_split = vec2<f32>(0.0, 1.0);                    // vertical split axis (rings part up/down)
				// --- UNRAVEL: radius blooms a little as the weave loosens, then the SNAP pulls it
				// back. tl_bloom peaks mid-burst and returns toward the core by the end.
				let tl_bloom = sin(tl_peelT * PI) * (0.26 + seed2 * 0.16);
				// Dying 3:2 harmonic wobble as the interlock dissolves (fades with exhale).
				let tl_harm = (1.0 - tl_e) * 0.10 * sin(tl_ang * 3.0 - tl_chan * 2.0 + seed2 * TAU);
				let tl_R = tl_r0 * 0.9 + tl_bloom + tl_harm;
				let tl_a = tl_ang + tl_spin;
				// Ring position: spun radius on the split-open weave, offset to its own side.
				let tl_ring = u.portal.xy + vec2<f32>(cos(tl_a), sin(tl_a)) * tl_R + tl_split * tl_peel;
				// --- SNAP (0.66..1.0): the whole unwound weave whips inward to a singularity.
				// snapT collapses everything back toward center at the tail, so the last frame is
				// a bright dense point, not scattered debris. This is the decisive close.
				let tl_snapT = smoothstep(0.66, 1.0, tl_e);
				let tl_snapped = mix(tl_ring, u.portal.xy, tl_snapT * tl_snapT);
				pos = mix(pos, tl_snapped, smoothstep(0.0, 0.55, tl_e));
				// --- HARD CONTAIN: nothing may exceed 1.25 NDC radius from the singularity.
				let tl_rr = length(pos - u.portal.xy);
				if (tl_rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / tl_rr); }
				// --- FLASH: a bright tearing seam fires at decouple (~0.16), hottest on the fault,
				// then a second white-hot pulse at the SNAP (~0.82) as the weave slams to a point.
				let tl_tear = exp(-pow((tl_e - 0.16) / 0.11, 2.0)) * (0.6 + tl_seam * 2.0);
				let tl_slam = exp(-pow((tl_e - 0.82) / 0.10, 2.0)) * 3.0;
				flash = flash + tl_tear + tl_slam + tl_e * 0.25;
			} else if (abs(u.interaction.z - 4.0) < 0.5) {
				// BLACKBOX (kind 4): the sealed vault is UNSEALED - cage struts snap their hinges
				// and swing open on bounded arcs, THEN the freed gem core detonates second in a
				// sharp bright shard-fan. Two-stage focal blast, all held inside a 1.25 NDC dome.
				// baked.xyz = original formation pos. Zones by norm: cage = high chebyshev-norm
				// (cube shell/edge); gem = small euclidean length (inner core).
				let b3 = baked.xyz;
				let maxn = max(max(abs(b3.x), abs(b3.y)), abs(b3.z));   // high near cube shell/edges
				let rlen = length(b3) + 0.0001;                          // small = inner gem core
				let is_cage = smoothstep(0.55, 0.95, maxn);              // 1 for edge-cage particles
				let is_gem = 1.0 - smoothstep(0.30, 0.62, rlen);         // 1 for inner gem-core particles
				// Dominant baked axis -> which way THIS strut faces (drives 2D swing direction).
				let ax = abs(b3.x); let ay = abs(b3.y); let azv = abs(b3.z);
				var axis3 = vec3<f32>(sign(b3.x + 0.00001), 0.0, 0.0);
				if (ay >= ax && ay >= azv) { axis3 = vec3<f32>(0.0, sign(b3.y + 0.00001), 0.0); }
				else if (azv >= ax && azv >= ay) { axis3 = vec3<f32>(0.0, 0.0, sign(b3.z + 0.00001)); }
				let edge_dir = normalize(vec2<f32>(axis3.x + b3.z * 0.35, axis3.y + b3.x * 0.20) + vec2<f32>(0.00001, 0.0));
				let gem_dir = normalize(vec2<f32>(b3.x, b3.y) + vec2<f32>(seed1 - 0.5, seed2 - 0.5) * 0.28 + vec2<f32>(0.00001, 0.0));

				// PRE-LOAD: the vault resists for a beat, the whole shell pulls fractionally
				// INWARD before it gives - loading the spring so the release reads harder.
				let resist = (1.0 - smoothstep(0.0, 0.18, exhale)) * exhale * 5.0;
				var bx_pos = pos - normalize(pos - u.portal.xy + vec2<f32>(0.00001, 0.0)) * resist * 0.045;

				// STAGE 1 - CAGE HINGE (fires FIRST): the seal snaps, struts swing open on a bounded
				// arc then settle at their outward reach. hinge = sin arc so they swing THEN release.
				let seal_open = smoothstep(0.08, 0.50, exhale);
				let hinge = sin(seal_open * PI * 0.5);                    // 0..1 swing arc
				let cage_reach = 0.60 + seed0 * 0.42;                     // bounded: max ~1.02
				let cage_swing = rot2(edge_dir, (1.0 - hinge) * (0.55 + seed2 * 0.45));
				let cage_pos = u.portal.xy + cage_swing * seal_open * (cage_reach * 0.55)
					+ edge_dir * hinge * hinge * cage_reach;

				// STAGE 2 - GEM DETONATION (fires SECOND, harder, brighter): the freed core
				// shatters into a sharp shard-fan. Accelerating ease-in so it HANGS then rushes,
				// reading as one decisive crack. Bounded reach so the fan fills the dome, not the frame.
				let gem_pop = smoothstep(0.32, 1.0, exhale);
				let shard = gem_pop * gem_pop;                            // ease-in: hangs then snaps out
				let gem_reach = 0.85 + seed0 * 0.30;                     // bounded: max ~1.15
				let gem_pos = u.portal.xy + gem_dir * shard * gem_reach
					+ rot2(gem_dir, seed1 * TAU) * (1.0 - shard) * 0.05;

				// Compose: cage particles take the hinge path, gem particles the shard path,
				// everything else (mid-shell) holds near the loaded position and drifts a hair.
				var vault_pos = bx_pos;
				vault_pos = mix(vault_pos, cage_pos, is_cage);
				vault_pos = mix(vault_pos, gem_pos, is_gem);
				pos = vault_pos;

				// CONTAINMENT: nothing may exceed 1.25 NDC radius from the singularity.
				let rr = length(pos - u.portal.xy);
				if (rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / rr); }

				// FLASH: mechanical glint when the seal snaps (stage 1), then a sharp white-hot
				// spike when the gem detonates (stage 2), with a hot core through the shard rush.
				let seal_glint = exp(-pow((exhale - 0.28) / 0.06, 2.0)) * (0.35 + is_cage * 0.7);
				let gem_flare = exp(-pow((exhale - 0.44) / 0.055, 2.0)) * (2.8 * is_gem);
				flash = flash + seal_glint + gem_flare + is_gem * gem_pop * 0.6;
			} else if (abs(u.interaction.z - 5.0) < 0.5) {
				// REASONING kind 5 - THE VERDICT LANCE. Aizawa chaos crushes onto the vertical
				// spine, then the whole converged column fires as ONE white-hot lance UP the
				// spine to a BOUNDED reach. Unidirectional, focal, hard-contained.
				//   baked.y      = height on the spine
				//   length(baked.xz) = orbital radius from the spine (the chaotic lobes)
				//   atan2(baked.z,baked.x) = azimuth around the spine
				let e = exhale;
				let spine_x = u.portal.x;
				let rel0 = pos - u.portal.xy;

				// Orbital radius (0.34 inner .. 0.90 widest lobe) and azimuth from baked.
				let orb_r = length(baked.xz) + 1e-4;
				let orb_norm = clamp((orb_r - 0.34) / 0.56, 0.0, 1.0);
				let azim = atan2(baked.z, baked.x);
				// Height rank along the spine (0 bottom .. 1 top). Top leads the lance.
				let hgt = clamp(baked.y * 0.5 + 0.5, 0.0, 1.0);

				// ===== ACT 1: CRUSH onto the vertical spine (0.00-0.45) =====
				// Wide lobes arrive LATER and spiral in on decaying angular momentum.
				let arrive = smoothstep(orb_norm * 0.50, 0.45 + orb_norm * 0.30, e);
				let spin = (1.0 - arrive) * (2.4 + orb_norm * 3.2) * (azim + seed0 * TAU);
				let swirl = rot2(rel0, spin) * (1.0 - arrive) * 0.55;
				// Pull x onto the spine; compress y toward the singularity so mass DENSIFIES
				// into a short hot column instead of staying spread. This is the focal gather.
				let col_y = mix(rel0.y, rel0.y * 0.25, arrive);
				let gathered = vec2<f32>(mix(pos.x, spine_x, arrive), u.portal.y + col_y) + swirl;

				// ===== ACT 2: FIRE the lance UP the spine (0.45-1.00) =====
				let fire = smoothstep(0.45, 1.0, e);
				let fire2 = fire * fire;
				// One shared upward front. Top-of-attractor gets a slight head start so the
				// column reads as a solid bolt with a leading tip, not a scattered spray.
				let lead = 0.10 + hgt * 0.35;
				let front = clamp(fire * (1.15 + lead), 0.0, 1.0);
				// Bounded reach: overshoot-then-hold via smoothstep on the front. Max ~1.05.
				let reach = smoothstep(0.0, 1.0, front) * (0.92 + seed2 * 0.20);
				// A gentle settle so the tail eases after the overshoot (reads as decisive).
				let settle = 1.0 - 0.10 * smoothstep(0.7, 1.0, fire);
				let beam_y = reach * settle;
				// Tight lateral jitter so it is a lance, not a razor line. Shrinks as it fires.
				let jit = (seed0 - 0.5) * 0.028 * (1.0 - fire * 0.6);
				let lanced = vec2<f32>(spine_x + jit, u.portal.y + beam_y);

				// The widest-orbit stragglers (~top 20%) are chaos that did not fully converge:
				// they peel into a BOUNDED arrowhead just off the lance tip, giving the bolt a
				// forked head instead of firing dead-straight. Sideways reach is capped small.
				let straggler = smoothstep(0.80, 0.95, orb_norm);
				let head_side = sign(cos(azim) + 1e-4);
				let head = u.portal.xy + vec2<f32>(head_side * (0.14 + seed1 * 0.18) * fire2,
				                                   (0.70 + seed2 * 0.30) * fire2);
				let phase2 = mix(lanced, head, straggler);

				// Compose: crush first, then fire.
				pos = mix(gathered, phase2, fire);

				// ===== FLASH: crush seam + travelling lance shock =====
				// (1) Crush seam - bright when the chaos finishes crushing onto the axis.
				let seam = exp(-pow((e - 0.42) / 0.055, 2.0)) * (0.5 + orb_norm * 1.3);
				// (2) Lance shock - a band of light riding the shared front up the spine.
				let shock = pow(max(1.0 - abs(fire - lead), 0.0), 6.0) * (1.0 - straggler) * (0.6 + fire * 2.6);
				flash = flash + seam + shock;

				// ===== HARD CONTAINMENT =====
				let rr = length(pos - u.portal.xy);
				if (rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / rr); }
			} else if (abs(u.interaction.z - 6.0) < 0.5) {
				// EXPLORE (kind 6) BURST - Dendrite Whip-Crack.
				// baked.xyz = local formation pos on the dendrite tree: direction picks the
				// branch, length = arclength along it (0 = seed/root core, ~1 = growing tip).
				// The whole tree lashes outward from the seed as ONE crack: a release front
				// rolls root->tip, branches snap to a BOUNDED resting arc, overshoot once and
				// recoil like a real whip, then tips flare as newly-lit paths. Contained + focal.
				let ex_e = exhale;                     // 0..1 burst progress
				// Tip-ness: where this particle sits along its branch. Root ~0, tip ~1.
				let ex_arm = clamp(length(baked.xyz), 0.0, 1.4);
				let ex_tip = clamp(ex_arm / 1.05, 0.0, 1.0);
				// SHAPE-TRUE heading: project the baked branch direction to screen. Each branch
				// keeps its own built-in heading, so the tree comes apart the way it is grown.
				let ex_bdir = normalize(baked.xyz + vec3<f32>(0.0001, 0.0001, 0.0));
				let ex_ang = atan2(ex_bdir.z, ex_bdir.x);
				var ex_head = vec2<f32>(cos(ex_ang), sin(ex_ang));
				// Per-branch id so forks separate cleanly (deterministic, no random spread).
				let ex_bid = hash11(seed0 * 91.7 + floor(ex_ang * 6.0));
				// FORK FAN: adjacent branches peel apart a SMALL, bounded amount as the crack
				// fires (possibility-space fanning). Bounded so it never sprays edge-to-edge.
				let ex_fsign = select(-1.0, 1.0, seed1 > 0.5);
				let ex_fan = ex_fsign * (0.18 + ex_bid * 0.30) * ex_e * (0.4 + ex_tip * 0.6);
				ex_head = rot2(ex_head, ex_fan);
				// Slight dendrite curl at the tips so branches arc rather than fire dead straight.
				ex_head = rot2(ex_head, (ex_bid - 0.5) * 0.55 * ex_e * ex_tip);
				ex_head = normalize(ex_head + vec2<f32>(0.00001, 0.0));
				// RELEASE FRONT: the crack rolls root->tip. Tips launch first; the root lags and
				// snaps last. A single travelling front, so it reads as a whip-crack, not a bloom.
				let ex_front = smoothstep(0.0, 1.0, ex_e * 1.30 - (1.0 - ex_tip) * 0.50);
				// WHIP ENVELOPE (bounded by construction): one smoothstep raised to a power for
				// the accelerating snap, then a gentle recoil so the whip settles. rest length is
				// tip-scaled and capped; root barely moves. NO compounding growth terms.
				let ex_snap = smoothstep(0.0, 0.72, ex_e);
				let ex_over = 1.0 + 0.14 * sin(clamp((ex_e - 0.55) / 0.35, 0.0, 1.0) * PI); // overshoot->recoil
				let ex_rest = 0.12 + ex_tip * (0.86 + seed2 * 0.22);   // root ~0.12, tip ~1.05 max
				let ex_reach = ex_front * ex_snap * ex_over * ex_rest;  // <= ~1.2 by construction
				// Final position: seed core stays near the singularity, tips race out along their
				// fanned heading to the bounded resting arc.
				pos = u.portal.xy + ex_head * ex_reach;
				// CONTAINMENT CLAMP: hard cap so no shard can leave the frame.
				let ex_rr = length(pos - u.portal.xy);
				if (ex_rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / ex_rr); }
				// CRACK FLASH: a rolling ignition wave root->tip. crackTime is earlier for tips
				// so the frontier lights first and the glow rolls down each branch; tips end
				// brightest = luminous frontier trails lighting the newly discovered paths.
				let ex_ctime = 0.16 + (1.0 - ex_tip) * 0.30;
				let ex_crack = exp(-pow((ex_e - ex_ctime) / 0.11, 2.0));
				flash = flash + ex_crack * (0.45 + ex_tip * 1.25) + ex_tip * ex_front * 0.55;
			} else if (abs(u.interaction.z - 7.0) < 0.5) {
				// FEED (kind 7) - VORTEX RING FLUSH-AND-CURL. The smoke ring self-propels
				// through its own hole as one bright slug, then the toroidal roll curls the
				// spent gas back around a BOUNDED ring. Contained, focal, punchy, fast.
				let e_fd = exhale;
				// Torus coords from the pre-projection formation point.
				let fd_phi = atan2(baked.z, baked.x);                 // toroidal position around the ring
				let fd_inPlane = length(vec2<f32>(baked.x, baked.z)); // in-plane radius (major ~0.62)
				let fd_inner = clamp(1.0 - fd_inPlane / 0.95, 0.0, 1.0); // 1 near the central hole, 0 at outer rim
				let fd_poloidal = atan2(baked.y, fd_inPlane - 0.62);  // angle around the tube cross-section

				// Screen-space flush axis: the direction the ring self-propels through its hole.
				// A gentle tilt so it reads as a real tilted smoke ring, not a flat pop.
				let fd_tilt = 0.42;
				let fd_jet = normalize(vec2<f32>(sin(fd_tilt) * 0.5, cos(fd_tilt)) + vec2<f32>(1e-4, 0.0));
				let fd_side = vec2<f32>(-fd_jet.y, fd_jet.x);         // perpendicular in screen

				let fd_rel = pos - u.portal.xy;

				// (1) POLOIDAL SPIN-UP (0.0-0.42): the roll around the tube accelerates and the
				// collar tightens toward the axis, smearing the ring into a bright spinning wheel
				// before it flushes. BOUNDED radius - collar only ever shrinks inward.
				let fd_spinUp = smoothstep(0.0, 0.42, e_fd);
				let fd_roll = fd_poloidal + fd_spinUp * (4.5 + fd_inner * 3.5) + seed0 * 0.6;
				let fd_collarR = mix(fd_inPlane, fd_inPlane * (0.30 + 0.22 * seed1), fd_spinUp);
				let fd_spun = rot2(fd_rel, fd_spinUp * (3.0 + fd_inner * 2.4));
				let fd_collarPos = u.portal.xy + normalize(fd_spun + vec2<f32>(1e-4, 0.0)) * fd_collarR;

				// (2) SLUG FLUSH (0.30-0.80): inner-hole material fires FIRST as a hot leading
				// slug straight through the center along the jet axis. Reach is CAPPED so the
				// spearhead punches but never leaves the frame (max ~0.92 forward).
				let fd_start = 0.30 - fd_inner * 0.16;                // inner particles launch sooner
				let fd_flush = smoothstep(fd_start, 0.80, e_fd);
				let fd_spear = fd_inner * (1.0 + seed2 * 0.4);        // inner edge = spearhead
				let fd_axial = fd_flush * fd_flush * (0.42 + fd_spear * 0.50); // capped ~0.42..0.92
				// residual swirl in the tube while it flushes, decaying as it collimates.
				let fd_swAmp = mix(fd_collarR, 0.02, fd_flush) * (0.85 + 0.25 * sin(fd_roll));
				let fd_swirl = fd_side * sin(fd_roll) * fd_swAmp;
				let fd_slug = u.portal.xy + fd_jet * fd_axial + fd_swirl;

				// (3) ROLLBACK CURL (0.55-1.0): the outer rim does NOT fly away - the toroidal
				// roll pulls it back AROUND the ring on a bounded orbit (radius fd_ringR), the
				// signature move that makes a smoke ring stay a ring. Widnall lobes comb the
				// azimuth so it reads as wavy gas breaking off, not noise.
				let fd_curl = smoothstep(0.55, 1.0, e_fd) * (1.0 - fd_spear * 0.7); // rim rolls, spearhead does not
				let fd_lobe = sin(fd_phi * 4.0 + seed1 * TAU);
				let fd_ringR = (0.34 + 0.26 * seed0 + fd_lobe * 0.06) * (0.5 + 0.5 * fd_curl); // bounded orbit <=0.66
				let fd_ang = fd_phi + e_fd * (2.2 + seed2 * 1.4) + fd_lobe * 0.3;   // rolls around the ring
				let fd_ringPos = u.portal.xy
					+ vec2<f32>(cos(fd_ang), sin(fd_ang)) * fd_ringR
					+ fd_jet * fd_axial * 0.35;                       // drifts a little downstream with the slug

				// Compose: collar -> slug for the leading spearhead; collar -> rollback ring for
				// the trailing rim. The spearhead flushes hard, the rim curls back and holds.
				let fd_lead = mix(fd_collarPos, fd_slug, fd_flush);
				let fd_trail = mix(fd_collarPos, fd_ringPos, fd_curl);
				pos = mix(fd_lead, fd_trail, fd_curl * (1.0 - fd_spear * 0.85));

				// CONTAINMENT: hard clamp so nothing can ever exceed 1.25 NDC from the singularity.
				let fd_rr = length(pos - u.portal.xy);
				if (fd_rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / fd_rr); }

				// FLASH: broad collar heat as the wheel spins up, then a sharp WHITE-HOT bolt as
				// the spearhead punches through the hole (peak ~0.55), then a quick fade.
				let fd_collarGlow = fd_spinUp * (1.0 - fd_flush) * 0.7;
				let fd_bolt = fd_spear * exp(-pow((e_fd - 0.55) / 0.10, 2.0)) * 3.6;
				let fd_fade = 1.0 - smoothstep(0.7, 1.0, e_fd);
				flash = flash + (fd_collarGlow + fd_bolt) * fd_fade;
			} else if (abs(u.interaction.z - 8.0) < 0.5) {
				// CONTRADICTIONS burst (kind 8) - Fault-Line Cleave of the Stella Octangula.
				// Two dual tetrahedra shear apart along ONE diagonal seam; the shared octahedral
				// core rips open on that seam and spits shrapnel perpendicular; spikes ride clean.
				let cd_e = exhale;                                    // 0..1 local burst window
				let cd_rel0 = pos - u.portal.xy;                      // post-collapse offset (near center)
				// Seam: fixed diagonal fault line so the cleave has one decisive axis on screen.
				let cd_seam = normalize(vec2<f32>(0.80, 0.60));       // cleave axis (unit)
				let cd_perp = vec2<f32>(-cd_seam.y, cd_seam.x);       // perpendicular to the seam
				// Which tetra this particle belongs to (thesis vs antithesis), from baked side.
				let cd_side = select(-1.0, 1.0, baked.x >= 0.0);
				// Radial zone within the compound: core (shared octahedron) vs spike tips.
				let cd_r = length(baked.xyz) + 1e-4;
				let cd_core = 1.0 - smoothstep(0.16, 0.60, cd_r);     // 1 at shared core, 0 at spike tip
				let cd_tip = 1.0 - cd_core;                            // 1 at spike tip
				// PHASE 1 (0..0.40): SHEAR - the two halves slide apart along the seam, tension builds.
				let cd_shear = smoothstep(0.0, 0.40, cd_e);
				let cd_sep = (0.30 + seed0 * 0.14) * cd_shear;        // bounded separation distance
				let cd_half_c = u.portal.xy + cd_seam * cd_side * cd_sep;  // each half's receding centroid
				// PHASE 2 (0.30..1): the seam RIPS. Core shrapnel fires PERPENDICULAR to the fault.
				let cd_rip = smoothstep(0.30, 1.0, cd_e);
				let cd_shrap = cd_perp * (seed2 - 0.5) * 2.0 * cd_rip * cd_rip
					* (0.30 + seed1 * 0.34) * cd_core;               // bounded, core-only
				// Each half tumbles about its receding centroid (opposite spins), decaying by tip.
				let cd_spin = cd_side * cd_rip * (1.6 + seed0 * 1.0) * PI;
				let cd_tumble = rot2(pos - cd_half_c, cd_spin);
				// Spike tips ride clean outward with their tetra as the cleave completes.
				let cd_ride = cd_seam * cd_side * (0.22 + seed2 * 0.30) * cd_rip * cd_rip * cd_tip;
				// Compose: base at half-centroid + tumbled offset (shrinking as it disperses),
				// then add perpendicular core shrapnel and the tip ride-out.
				let cd_gather = mix(0.55, 0.30, cd_rip);              // offset shrinks as halves recede
				pos = cd_half_c + cd_tumble * cd_gather + cd_shrap + cd_ride;
				// HARD CONTAINMENT: nothing escapes ~1.25 NDC of the singularity.
				let cd_rr = length(pos - u.portal.xy);
				if (cd_rr > 1.25) { pos = u.portal.xy + (pos - u.portal.xy) * (1.25 / cd_rr); }
				// FLASH: a sharp white RIP along the seam at the instant of cleave (gaussian at 0.34),
				// hottest at the contested core, then quick sparks as shrapnel is flung, fast fade.
				let cd_seam_flash = exp(-pow((cd_e - 0.34) / 0.075, 2.0)) * (0.4 + cd_core * 1.9);
				let cd_spark = cd_core * cd_rip * (1.0 - cd_rip) * 4.0 * (0.4 + seed1 * 0.6);
				flash = flash + cd_seam_flash + cd_spark;
			} else {
				let rnd = vec2<f32>(seed1 - 0.5, seed2 - 0.5) * 0.16;
				let dir = normalize(rel + rnd + vec2<f32>(0.00001, 0.0));
				let supernova = u.portal.xy + dir * (1.5 + seed1 * 1.1) * exhale * exhale;
				pos = mix(pos, supernova, exhale);
			}
		}
	}

	// ── EXPLOSION BLAZE: during the burst, debris that has flung far from the
	// singularity blazes white-hot (fast-moving shrapnel reads as a real detonation,
	// not a color glow). Measured by how far the final pos landed from the portal +
	// how deep into the explosion we are. Only active while bursting; zero otherwise.
	var burst_blaze = 0.0;
	if (bursting && !reduced) {
		let ex = smoothstep(0.66, 1.0, progress);
		let flung = length(pos - u.portal.xy);          // how far this shard shot out
		let leadEdge = smoothstep(0.2, 1.1, flung);      // outer shrapnel is hottest
		// a hot crest that sweeps outward as the explosion expands, so the debris
		// front glows like a shockwave, then cools into embers.
		let crest = exp(-pow((flung - ex * 1.6) / 0.28, 2.0));
		burst_blaze = ex * (leadEdge * 2.2 + crest * 3.2) * (0.7 + seed1 * 0.6);
	}

	// ── Energy = the launch engine's living blaze ──
	var living = 0.55;
	if (!reduced) { living = 0.62 + 0.18 * sin(t * 8.0 + seed2 * TAU); }
	var energy = living + axon * 1.3 + focus * 1.7 + systole * 2.2 + weather
		+ sin(intro * PI) * 1.1 + flash * 7.0 + burst_blaze;
	if (hover_on && !hovered) { energy *= 0.30; }
	energy = max(energy, 0.08);

	let corner = QUAD[vertex_index];
	let pixel = 2.0 / max(min(u.viewport.x, u.viewport.y), 1.0);
	let depth_size = clamp(1.0 + local3.z * 0.12, 0.82, 1.24);
	// Axon particles are a THIN thread: a small fixed base (packets briefly widen),
	// NOT the fat organ-body size. Organ particles keep the full size profile.
	let is_axon = step(0.001, axon);
	let organ_size = (1.35 + baked.w * 1.9 + focus * 1.5 + systole * 0.6 + min(flash * 3.0, 3.0) + min(burst_blaze * 1.2, 4.0)) * depth_size;
	let axon_size = 0.9 + axon * 0.6;
	let size = mix(organ_size, axon_size, is_axon);

	// Color: the REAL iridescent palette, seeded per-particle, energized by the beat.
	let world = vec3<f32>(pos.x, pos.y, local3.z);
	let col = palette(color_route.x + color_route.y * 0.3 + color_route.z * 0.6, energy, world, t);

	var out: VSOut;
	out.position = vec4<f32>(pos + corner * pixel * size, 0.0, 1.0);
	out.local = corner;
	out.color = col;
	out.energy = energy;
	return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
	// Clean round additive orb: bright gaussian core + soft halo, hard zero past
	// the disc edge so the quad corners never square off (launch-engine sprite).
	let r = clamp(length(input.local) / 0.92, 0.0, 1.0);
	let disc = 1.0 - smoothstep(0.0, 1.0, r);
	if (disc <= 0.001) { discard; }
	// Tight bright core + wide soft halo so each orb reads as a luminous point,
	// not speckle. Higher per-sprite gain than the launch default because the
	// Palace shows 9 small formations at once (each needs to punch), and bloom
	// then blooms the cores into glowing bodies.
	let profile = (pow(1.0 - r, 2.6) * 1.35 + (1.0 - r) * 0.5) * disc;
	return vec4<f32>(input.color * profile * (0.28 + input.energy * 0.5), 1.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────
// CPU formation library — the REAL launch-engine attractors, baked once per
// particle at upload. Each returns a local-space point in roughly [-1.5, 1.5].
// Per-particle random integration length fills the attractor curves into volumes
// (memory 55b69e52). `id` is the per-particle seed.
// ─────────────────────────────────────────────────────────────────────────────
function fract(v: number): number {
	return v - Math.floor(v);
}

/** Smoothstep 0..1 (clamped) — CPU mirror of WGSL smoothstep(0,1,x). */
function smooth01(x: number): number {
	const t = x < 0 ? 0 : x > 1 ? 1 : x;
	return t * t * (3 - 2 * t);
}
function h(x: number): number {
	return fract(Math.sin(x * 12.9898 + 78.233) * 43758.5453);
}
function rot2cpu(x: number, y: number, a: number): [number, number] {
	const c = Math.cos(a);
	const s = Math.sin(a);
	return [c * x - s * y, s * x + c * y];
}

const TAU = Math.PI * 2;

// ── OBSERVATORY (0): NEURAL SPHERE — a dense cortex-sphere with sulci ridges and
// a hollow-biased shell so it reads as a living mind, not a solid ball. ──
// ─────────────────────────────────────────────────────────────────────────────
// Nine bespoke organ shapes (art-director fleet wf_81764e70-be2, all beauty 9/10).
// Each is a deterministic pure function of the particle id (hash h(), no random),
// filling ~volumetrically so it never collapses to a bar under the slow Y-rotation.
// Each MEANS its organ (the discipline test).
// ─────────────────────────────────────────────────────────────────────────────

// OBSERVATORY — Folded cortex shell (spherical-harmonic gyrification + fissure):
// a literal breathing brain, the mind at rest at the center of everything.
function observatoryFormation(id: number): [number, number, number] {
	const u0 = h(id * 0.7351 + 1.0);
	const u1 = h(id * 1.2971 + 5.0);
	const u2 = h(id * 2.1637 + 9.0);
	const u3 = h(id * 3.5391 + 13.0);
	if (u0 < 0.15) {
		const ct = 2 * u1 - 1;
		const st = Math.sqrt(Math.max(0, 1 - ct * ct));
		const ph = u2 * TAU;
		const r = 0.2 + 0.16 * Math.cbrt(u3);
		return [st * Math.cos(ph) * r, ct * r, st * Math.sin(ph) * r];
	}
	const hemi = u1 < 0.5 ? -1 : 1;
	const cphi = 2 * u2 - 1;
	const phi = Math.acos(Math.max(-1, Math.min(1, cphi)));
	const sphi = Math.sin(phi);
	const gap = 0.2;
	const theta = hemi === 1 ? gap + u3 * (Math.PI - 2 * gap) : Math.PI + gap + u3 * (Math.PI - 2 * gap);
	let x = sphi * Math.cos(theta);
	let y = cphi;
	let z = sphi * Math.sin(theta);
	const fold =
		0.095 * Math.sin(7.0 * theta) * (sphi * sphi) +
		0.075 * Math.cos(9.0 * theta + 2.0 * phi) * sphi +
		0.065 * Math.sin(6.0 * phi) +
		0.05 * Math.sin(11.0 * theta) * Math.sin(4.0 * phi);
	const dMid = Math.min(Math.abs(theta - Math.PI / 2), Math.abs(theta - (3 * Math.PI) / 2));
	const fissure = -0.16 * Math.exp(-(dMid * dMid) / (0.1 * 0.1)) * sphi;
	const thick = 0.02 * (h(id * 5.19 + 2.0) - 0.5);
	const r = 0.92 + fold + fissure + thick;
	x *= r; y *= r; z *= r;
	x += hemi * 0.05 * Math.exp(-(dMid * dMid) / (0.16 * 0.16)) * sphi;
	y *= 0.9; z *= 1.06; x *= 1.04;
	return [x, y, z];
}

// GRAPH — Hopf fibration: nested tori of interlinked Villarceau rings, THE math
// object of linkage (everything connected to everything, causal edges).
function graphFormation(id: number): [number, number, number] {
	const RINGS = 12, TUBE = 0.05, PROJ_SCALE = 0.66;
	const r = h(id * 1.17 + 3.0);
	const t = h(id * 1.91 + 7.0);
	const j1 = h(id * 2.37 + 11.0);
	const j2 = h(id * 3.11 + 5.0);
	const ring = Math.floor(r * RINGS);
	const band = ring % 3;
	const cBase = [0.55, 0.05, -0.45][band];
	const perBand = Math.ceil(RINGS / 3);
	const idxInBand = Math.floor(ring / 3);
	const lon = (idxInBand / perBand) * TAU + band * 0.9;
	const sB = Math.sqrt(Math.max(0, 1 - cBase * cBase));
	const a = sB * Math.cos(lon);
	const b = sB * Math.sin(lon);
	const c = cBase;
	const theta = t * TAU;
	const denom = Math.sqrt(2 * (1 + c));
	const x1 = ((1 + c) * Math.cos(theta)) / denom;
	const x2 = (a * Math.sin(theta) - b * Math.cos(theta)) / denom;
	const x3 = (a * Math.cos(theta) + b * Math.sin(theta)) / denom;
	const x4 = ((1 + c) * Math.sin(theta)) / denom;
	// Stereographic projection blows up near the pole (x4 -> 1, w -> 0), throwing a
	// few particles thousands of units away -> a visible streak flying off-screen
	// (worse when the dive magnifies the organ). Fix: for near-pole particles, pull
	// the fiber angle OFF the pole (re-evaluate theta shifted) so no particle ever
	// projects to the divergent region — the ring just wraps to its near side.
	let w = 1 - x4;
	if (Math.abs(w) < 0.12) {
		const th2 = theta + Math.PI; // opposite side of the fiber, far from the pole
		const nx1 = ((1 + c) * Math.cos(th2)) / denom;
		const nx2 = (a * Math.sin(th2) - b * Math.cos(th2)) / denom;
		const nx3 = (a * Math.cos(th2) + b * Math.sin(th2)) / denom;
		const nx4 = ((1 + c) * Math.sin(th2)) / denom;
		let X2 = nx1 / (1 - nx4), Y2 = nx2 / (1 - nx4), Z2 = nx3 / (1 - nx4);
		const R2 = Math.hypot(X2, Y2, Z2) || 1;
		const Rc2 = 1.6 * Math.tanh(R2 / 1.6);
		X2 *= Rc2 / R2; Y2 *= Rc2 / R2; Z2 *= Rc2 / R2;
		const ph2 = j1 * TAU;
		const rr2 = TUBE * Math.sqrt(j2);
		return [
			(Z2 + (j2 - 0.5) * TUBE) * PROJ_SCALE,
			(Y2 + Math.sin(ph2) * rr2) * PROJ_SCALE,
			(X2 + Math.cos(ph2) * rr2) * PROJ_SCALE
		];
	}
	let X = x1 / w, Y = x2 / w, Z = x3 / w;
	const R = Math.hypot(X, Y, Z);
	// Hard clamp to a tight radius so nothing can escape the frame.
	const Rc = 1.55 * Math.tanh(R / 1.55);
	const sc = R > 1e-6 ? Rc / R : 0;
	X *= sc; Y *= sc; Z *= sc;
	const ph = j1 * TAU;
	const rr = TUBE * Math.sqrt(j2);
	X += Math.cos(ph) * rr; Y += Math.sin(ph) * rr; Z += (j2 - 0.5) * TUBE;
	return [Z * PROJ_SCALE, Y * PROJ_SCALE, X * PROJ_SCALE];
}

// MEMORIES — Constellation archive: 3 concentric Fibonacci shells of discrete
// memory-cell pearls; retention = shell brightness. Many separable units.
const MEM_SHELLS = [
	{ radius: 1.18, cells: 210 },
	{ radius: 0.86, cells: 132 },
	{ radius: 0.54, cells: 72 }
];
const MEM_TOTAL_CELLS = 414;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
function memoriesFormation(id: number): [number, number, number] {
	let gcell = Math.floor(h(id * 1.6180339887 + 0.5) * MEM_TOTAL_CELLS);
	if (gcell >= MEM_TOTAL_CELLS) gcell = MEM_TOTAL_CELLS - 1;
	let shell = 0, cell = gcell;
	if (cell >= MEM_SHELLS[0].cells) { cell -= MEM_SHELLS[0].cells; shell = 1; }
	if (shell === 1 && cell >= MEM_SHELLS[1].cells) { cell -= MEM_SHELLS[1].cells; shell = 2; }
	const S = MEM_SHELLS[shell];
	const n = S.cells;
	const zc = 1 - (2 * cell + 1) / n;
	const rc = Math.sqrt(Math.max(0, 1 - zc * zc));
	const phi = GOLDEN_ANGLE * cell;
	const cx = Math.cos(phi) * rc, cy = zc, cz = Math.sin(phi) * rc;
	const cluster = 0.05 + 0.012 * (2 - shell);
	const u = h(id * 2.31 + 5.0), vv = h(id * 3.97 + 9.0);
	const rr = cluster * Math.cbrt(h(id * 4.13 + 1.0));
	const ct = 2 * u - 1;
	const st = Math.sqrt(Math.max(0, 1 - ct * ct));
	const pth = vv * TAU;
	const jx = rr * st * Math.cos(pth), jy = rr * ct, jz = rr * st * Math.sin(pth);
	const R = S.radius;
	return [(cx * R + jx) * 1.06, (cy * R + jy) * 1.06, (cz * R + jz) * 1.06];
}

// TIMELINE — Bitemporal Clifford torus: two independent clocks (valid-time u,
// transaction-time v) woven as a 3:2 braid on the 4D torus, projected to 3D.
function timelineFormation(id: number): [number, number, number] {
	const R = Math.SQRT1_2;
	const t = h(id * 1.7 + 3.1);
	const S = 6;
	const s = Math.floor(h(id * 0.31 + 9.2) * S);
	const phase = (s / S) * TAU;
	const u = t * TAU * 3 + phase;
	const v = t * TAU * 2 + phase * 1.618;
	const x4 = R * Math.cos(u), y4 = R * Math.sin(u), z4 = R * Math.cos(v), w4 = R * Math.sin(v);
	const D = 1.28 - w4;
	let X = x4 / D, Y = y4 / D, Z = z4 / D;
	const tubeR = 0.045;
	const fa = h(id * 2.3 + 1.0) * TAU;
	const fb = Math.sqrt(h(id * 3.7 + 2.0)) * tubeR;
	X += Math.cos(fa) * fb; Y += Math.sin(fa) * fb; Z += (h(id * 4.1 + 5.0) - 0.5) * tubeR * 2;
	const k = 1.02, tilt = 0.5;
	const ct = Math.cos(tilt), st = Math.sin(tilt);
	const Yt = Y * ct - Z * st, Zt = Y * st + Z * ct;
	return [X * k, Yt * k, Zt * k];
}

// BLACKBOX — Sealed vault: a faceted gem sealed inside a cubic crystal cage bound
// by 8 corner seal-struts. A thing sealed inside a thing = the exportable receipt.
const BB_PHI = (1 + Math.sqrt(5)) / 2;
const BB_ICO: [number, number, number][] = (() => {
	const p = BB_PHI;
	const base: [number, number, number][] = [
		[0, 1, p], [0, 1, -p], [0, -1, p], [0, -1, -p],
		[1, p, 0], [1, -p, 0], [-1, p, 0], [-1, -p, 0],
		[p, 0, 1], [p, 0, -1], [-p, 0, 1], [-p, 0, -1]
	];
	const n = Math.hypot(0, 1, p);
	return base.map((v) => [v[0] / n, v[1] / n, v[2] / n] as [number, number, number]);
})();
function bbGemRadius(nx: number, ny: number, nz: number): number {
	let m = -2;
	for (const a of BB_ICO) {
		const d = Math.abs(nx * a[0] + ny * a[1] + nz * a[2]);
		if (d > m) m = d;
	}
	return 0.86 + 0.14 * m;
}
function blackboxFormation(id: number): [number, number, number] {
	const CUBE = 1.16, GEM = 0.56;
	const bucket = h(id + 2);
	if (bucket < 0.46) {
		if (h(id + 4) < 0.82) {
			const axis = Math.floor(h(id + 6) * 3) % 3;
			const s1 = h(id + 8) < 0.5 ? -1 : 1;
			const s2 = h(id + 10) < 0.5 ? -1 : 1;
			const t = -CUBE + 2 * CUBE * h(id + 12);
			const j = (h(id + 14) - 0.5) * 0.03;
			const c: [number, number, number] = [0, 0, 0];
			c[axis] = t + j;
			c[(axis + 1) % 3] = s1 * CUBE + j;
			c[(axis + 2) % 3] = s2 * CUBE + j;
			return c;
		}
		const face = Math.floor(h(id + 16) * 6);
		const axis = Math.floor(face / 2);
		const side = (face % 2) ? 1 : -1;
		const t = h(id + 18);
		const dir = h(id + 20) < 0.5 ? 1 : -1;
		const u = -CUBE + 2 * CUBE * t;
		const w = dir * (-CUBE + 2 * CUBE * t);
		const c: [number, number, number] = [0, 0, 0];
		c[axis] = side * CUBE;
		c[(axis + 1) % 3] = u;
		c[(axis + 2) % 3] = w;
		return c;
	}
	if (bucket < 0.86) {
		const N = 1600;
		const k = Math.floor(h(id + 22) * N);
		const yy = 1 - (2 * k) / (N - 1);
		const rr = Math.sqrt(Math.max(0, 1 - yy * yy));
		const ga = 2.399963229728653 * k;
		const nx = Math.cos(ga) * rr, ny = yy, nz = Math.sin(ga) * rr;
		const rad = GEM * bbGemRadius(nx, ny, nz);
		return [nx * rad, ny * rad, nz * rad];
	}
	const corner = Math.floor(h(id + 24) * 8);
	const sx = (corner & 1) ? 1 : -1;
	const sy = (corner & 2) ? 1 : -1;
	const sz = (corner & 4) ? 1 : -1;
	const inR = GEM * 0.62;
	const t = h(id + 26);
	const j = (h(id + 28) - 0.5) * 0.025;
	const ix = sx * inR, iy = sy * inR, iz = sz * inR;
	const ox = sx * CUBE, oy = sy * CUBE, oz = sz * CUBE;
	return [ix + (ox - ix) * t + j, iy + (oy - iy) * t + j, iz + (oz - iz) * t + j];
}

// REASONING — Aizawa convergence attractor: scattered evidence flung into orbital
// lobes and dragged onto a dense vertical spine — chaos converging to a conclusion.
function reasoningFormation(id: number): [number, number, number] {
	const a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1;
	let x = (h(id + 1) - 0.5) * 1.2;
	let y = (h(id + 2) - 0.5) * 1.2;
	let z = (h(id + 3) - 0.5) * 0.6 + 0.4;
	const dt = 0.01;
	const burn = 520;
	const steps = burn + Math.floor(h(id + 7) * 260);
	for (let i = 0; i < steps; i++) {
		const dx = (z - b) * x - d * y;
		const dy = d * x + (z - b) * y;
		const dz = c + a * z - (z * z * z) / 3 - (x * x + y * y) * (1 + e * z) + f * z * x * x * x;
		x += dx * dt; y += dy * dt; z += dz * dt;
	}
	const s = 0.92;
	return [x * s, (z - 0.6) * s * 1.05, y * s];
}

// EXPLORE — Semantic walk: a glowing seed with dendritic filaments radiating and
// forking outward into an expanding neighborhood (a walk that branches). Full 3D.
function exploreFormation(id: number): [number, number, number] {
	const TRUNKS = 14, DEPTH = 3, STEP0 = 0.5, STEP_DECAY = 0.8, SPREAD = 0.42;
	const GOLDEN = 2.399963229728653, CORE_FRAC = 0.12, TUBE = 0.02, REACH = 1.3;
	if (h(id * 0.53 + 0.9) < CORE_FRAC) {
		const rr = 0.11 * Math.cbrt(h(id + 21));
		const th = h(id + 22) * TAU;
		const ph = Math.acos(2 * h(id + 23) - 1);
		return [rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph), rr * Math.sin(ph) * Math.sin(th)];
	}
	const trunk = Math.floor(h(id * 1.373 + 5.3) * TRUNKS);
	const uy = 1 - ((trunk + 0.5) / TRUNKS) * 2;
	const ur = Math.sqrt(Math.max(0, 1 - uy * uy));
	const ua = trunk * GOLDEN;
	let hx = Math.cos(ua) * ur, hy = uy, hz = Math.sin(ua) * ur;
	const t = Math.pow(h(id * 0.911 + 1.7), 0.9);
	let L = 0;
	const segLen: number[] = [];
	for (let dd = 0; dd < DEPTH; dd++) { const s = STEP0 * Math.pow(STEP_DECAY, dd); segLen.push(s); L += s; }
	const target = t * L;
	let x = 0, y = 0, z = 0, walked = 0;
	for (let dd = 0; dd < DEPTH; dd++) {
		if (dd > 0) {
			const child = h(id * (1.0 + dd * 0.41) + dd * 7.13 + 3.1) < 0.5 ? 0 : 1;
			const forkAz = (child - 0.5) * Math.PI + GOLDEN * (dd + trunk);
			const forkPol = SPREAD * Math.pow(0.9, dd - 1) * (0.7 + 0.6 * h(id + dd * 11 + 17));
			let ux = 0, uyy = 1, uz = 0;
			if (Math.abs(hy) > 0.9) { ux = 1; uyy = 0; uz = 0; }
			let sx = hy * uz - hz * uyy, sy = hz * ux - hx * uz, sz = hx * uyy - hy * ux;
			const sl = Math.hypot(sx, sy, sz) || 1;
			sx /= sl; sy /= sl; sz /= sl;
			const vx = sy * hz - sz * hy, vy = sz * hx - sx * hz, vz = sx * hy - sy * hx;
			const ca = Math.cos(forkAz), sa = Math.sin(forkAz);
			const dxx = ca * sx + sa * vx, dyy = ca * sy + sa * vy, dzz = ca * sz + sa * vz;
			const cp = Math.cos(forkPol), sp = Math.sin(forkPol);
			hx = cp * hx + sp * dxx; hy = cp * hy + sp * dyy; hz = cp * hz + sp * dzz;
			const hl = Math.hypot(hx, hy, hz) || 1;
			hx /= hl; hy /= hl; hz /= hl;
		}
		const s = segLen[dd];
		if (walked + s >= target) { const fr = (target - walked) / s; x += hx * s * fr; y += hy * s * fr; z += hz * s * fr; break; }
		x += hx * s; y += hy * s; z += hz * s; walked += s;
	}
	const taper = 0.35 + 0.65 * Math.sin(Math.min(1, t) * Math.PI);
	x += (h(id * 3.1 + 1) - 0.5) * TUBE * taper;
	y += (h(id * 3.3 + 2) - 0.5) * TUBE * taper;
	z += (h(id * 3.7 + 3) - 0.5) * TUBE * taper;
	const scale = REACH / L;
	return [x * scale, y * scale, z * scale];
}

// FEED — Tilted vortex ring: particles ride intertwined helical streamlines around
// a smoke-ring tube (poloidal flow), tilted so the hole faces the viewer. The
// endless circulating event stream.
function feedFormation(id: number): [number, number, number] {
	const a = h(id * 1.17 + 3.0), b = h(id * 1.91 + 7.0);
	const R = 0.82, rTube = 0.34, STRANDS = 5, TWIST = 6;
	const u = a * TAU;
	const strand = Math.floor(b * STRANDS);
	const v = u * TWIST + (strand / STRANDS) * TAU + (h(id + 5) - 0.5) * 0.5;
	const shell = 0.72 + 0.28 * Math.sqrt(h(id + 13));
	const rr = rTube * shell;
	let x = (R + rr * Math.cos(v)) * Math.cos(u);
	let z = (R + rr * Math.cos(v)) * Math.sin(u);
	let y = rr * Math.sin(v);
	const tx = 1.0821;
	const cyt = Math.cos(tx), syt = Math.sin(tx);
	const y1 = y * cyt - z * syt, z1 = y * syt + z * cyt;
	y = y1; z = z1;
	const rz = 0.2094;
	const czr = Math.cos(rz), szr = Math.sin(rz);
	const x2 = x * czr - y * szr, y2 = x * szr + y * czr;
	return [x2, y2, z];
}

// CONTRADICTIONS — Stella octangula: two interpenetrating dual tetrahedra (the 3D
// Star of David), strained apart on X. The canonical symbol of duals in opposition.
function contradictionsFormation(id: number): [number, number, number] {
	const a = h(id * 1.17 + 3.0), b = h(id * 1.91 + 7.0), c = h(id * 2.37 + 11.0);
	const R = 0.92;
	const A_VERTS: [number, number, number][] = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
	const B_VERTS: [number, number, number][] = [[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]];
	const tetB = a < 0.5;
	const verts = tetB ? B_VERTS : A_VERTS;
	const shift = tetB ? -0.16 : 0.16;
	if (b < 0.72) {
		const edges: [number, number][] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
		const e = edges[Math.floor(c * 6) % 6];
		const v0 = verts[e[0]], v1 = verts[e[1]];
		const u = h(id * 3.11 + 5.0);
		const t = 0.5 + 0.5 * Math.sign(u - 0.5) * Math.pow(Math.abs(2 * u - 1), 0.55);
		const jx = (h(id * 4.7 + 1.0) - 0.5) * 0.05, jy = (h(id * 5.3 + 2.0) - 0.5) * 0.05, jz = (h(id * 6.1 + 3.0) - 0.5) * 0.05;
		return [(v0[0] + (v1[0] - v0[0]) * t) * R + jx + shift, (v0[1] + (v1[1] - v0[1]) * t) * R + jy, (v0[2] + (v1[2] - v0[2]) * t) * R + jz];
	}
	if (b < 0.92) {
		const v = verts[Math.floor(c * 4) % 4];
		const r = Math.pow(h(id * 7.7 + 4.0), 1.5) * 0.11;
		const th = h(id * 8.3 + 6.0) * TAU;
		const ph = Math.acos(2 * h(id * 9.1 + 8.0) - 1);
		return [v[0] * R + r * Math.sin(ph) * Math.cos(th) + shift, v[1] * R + r * Math.sin(ph) * Math.sin(th), v[2] * R + r * Math.cos(ph)];
	}
	const th = h(id * 10.3 + 7.0) * TAU;
	const ph = Math.acos(2 * h(id * 11.7 + 9.0) - 1);
	const sx = Math.sin(ph) * Math.cos(th), sy = Math.sin(ph) * Math.sin(th), sz = Math.cos(ph);
	const l1 = Math.abs(sx) + Math.abs(sy) + Math.abs(sz);
	const s = 0.34;
	return [sx / l1 * s, sy / l1 * s, sz / l1 * s];
}

function formationPoint(kind: number, id: number): [number, number, number] {
	switch (kind) {
		case 0: return observatoryFormation(id);
		case 1: return graphFormation(id);
		case 2: return memoriesFormation(id);
		case 3: return timelineFormation(id);
		case 4: return blackboxFormation(id);
		case 5: return reasoningFormation(id);
		case 6: return exploreFormation(id);
		case 7: return feedFormation(id);
		default: return contradictionsFormation(id);
	}
}

function particleCount(reducedMotion: boolean): number {
	const cores = navigator.hardwareConcurrency || 8;
	const small = window.innerWidth < 760;
	const weak = cores <= 4 || (window.devicePixelRatio || 1) > 2.2;
	if (reducedMotion) return small ? 10_000 : 16_000;
	if (weak) return small ? 12_000 : 20_000;
	return small ? 18_000 : cores >= 10 ? 55_000 : 40_000;
}

export class PalaceBrainPass implements FramePass {
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
		// Safety fallback: if the GPU loop stalls, still navigate. Fires just AFTER
		// the intended flash peak (0.66 * 2000ms = 1320ms) so a healthy run navigates
		// via the shader-driven flash, and only a real stall trips this backstop.
		// Stall-only backstop: fire just AFTER the intended nav point (0.90 * 1000ms
		// = 900ms) so a healthy run navigates via the shader-driven flash AFTER the
		// explosion has played, and only a genuine GPU stall trips this early.
		this.watchdog = setTimeout(() => this.fireFlashCallback(), 980);
		return true;
	}

	uploadRegions(regions: OrganRegion[]): void {
		const device = this.engine.gpuDevice;
		if (!device || regions.length === 0) return;

		this.placed = regions.map((region, index) => {
			const anchor = DESKTOP_ANCHORS[region.href] ?? [0, 0, index * 0.01, 0.14];
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
		// Cortex (observatory) gets a heavier share; the rest split the remainder.
		const data = new Float32Array(this.count * FLOATS_PER_PARTICLE);
		for (let i = 0; i < this.count; i++) {
			const id = i + 1;
			const offset = i * FLOATS_PER_PARTICLE;
			// Route-contiguous, cortex (route 0) weighted ~2x.
			const route = this.routeForIndex(i);
			const region = this.placed[route];
			const local = formationPoint(region.kind, id);
			const color = rgb01(FAMILY_COLOR[regions[route].family]);
			data[offset + 0] = local[0];
			data[offset + 1] = local[1];
			data[offset + 2] = local[2];
			data[offset + 3] = h(id * 2.83 + 12); // size seed
			data[offset + 4] = region.x;
			data[offset + 5] = region.y;
			data[offset + 6] = region.z;
			data[offset + 7] = region.scale;
			// Color slot carries a per-particle hue seed (for the iridescent palette)
			// in .x plus the family base in a compressed form; the shader reads
			// color_route.xyz as (hueSeed, familyBias, familyBias2) and .w as route.
			data[offset + 8] = h(id * 1.7 + 5); // hue seed 0..1
			data[offset + 9] = color[0] * 0.5 + color[1] * 0.3; // gentle family bias
			data[offset + 10] = color[2] * 0.6; // family bias 2
			data[offset + 11] = route;
		}

		this.uniformBuffer?.destroy();
		this.particleBuffer?.destroy();
		this.uniformBuffer = device.createBuffer({
			label: 'palace-brain-uniforms',
			size: this.uniformData.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		this.particleBuffer = device.createBuffer({
			label: 'palace-brain-particles',
			size: data.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(this.particleBuffer, 0, data);
		this.createPipeline(device);
	}

	// Cortex-weighted, route-contiguous index -> route mapping. Observatory (0)
	// gets ~2x the share of an outer organ so the center reads as the hub.
	private routeForIndex(i: number): number {
		const n = this.placed.length;
		if (n === 0) return 0;
		const cortexWeight = 2.0;
		const totalWeight = cortexWeight + (n - 1);
		const cortexShare = Math.floor((this.count * cortexWeight) / totalWeight);
		if (i < cortexShare) return 0;
		const rest = this.count - cortexShare;
		const perOuter = Math.max(1, Math.floor(rest / Math.max(1, n - 1)));
		return Math.min(n - 1, 1 + Math.floor((i - cortexShare) / perOuter));
	}

	private createPipeline(device: GPUDevice): void {
		if (!this.uniformBuffer || !this.particleBuffer) return;
		device.pushErrorScope('validation');
		const module = device.createShaderModule({ label: 'palace-brain-shader', code: palaceBrainWGSL });
		void module.getCompilationInfo().then((info) => {
			const errors = info.messages.filter((m) => m.type === 'error');
			if (errors.length > 0) {
				console.error('[palace-brain] WGSL:', errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
			}
		});
		this.pipeline = device.createRenderPipeline({
			label: 'palace-brain-pipeline',
			layout: 'auto',
			vertex: {
				module,
				entryPoint: 'vs_main',
				buffers: [
					{
						arrayStride: FLOATS_PER_PARTICLE * 4,
						stepMode: 'instance',
						attributes: [
							{ shaderLocation: 0, offset: 0, format: 'float32x4' },
							{ shaderLocation: 1, offset: 16, format: 'float32x4' },
							{ shaderLocation: 2, offset: 32, format: 'float32x4' }
						]
					}
				]
			},
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
			label: 'palace-brain-bind',
			layout: this.pipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
		});
		void device.popErrorScope().then((error) => {
			if (error) console.error('[palace-brain] pipeline validation:', error.message);
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
		let portal = this.hoveredIndex >= 0 ? this.placed[this.hoveredIndex] : null;
		// The portal center where the singularity + burst happen. During the burst
		// it travels from the clicked organ's anchor to SCREEN CENTER over the DIVE
		// phase (0..0.42), so the shape flies to the middle, then collapses and
		// detonates center-stage (Sam: "go to the middle THEN burst").
		let portalX = portal?.x ?? 0;
		let portalY = portal?.y ?? 0;
		if (this.burst) {
			selectedIndex = this.indexOfHref(this.burst.href);
			portal = selectedIndex >= 0 ? this.placed[selectedIndex] : portal;
			const duration = this.reducedMotion ? REDUCED_BURST_MS : BURST_MS;
			progress = Math.min(1, Math.max(0, (now - this.burst.startMs) / duration));
			const threshold = this.reducedMotion ? 0.3 : FLASH_REQUEST_AT;
			if (progress >= threshold) this.fireFlashCallback();
			// smoothstep the anchor -> (0,0) over the dive so the organ glides to center.
			const anchorX = portal?.x ?? 0;
			const anchorY = portal?.y ?? 0;
			const toCenter = this.reducedMotion ? 1 : smooth01(progress / 0.42);
			portalX = anchorX * (1 - toCenter);
			portalY = anchorY * (1 - toCenter);
		}

		const introDur = 1_650;
		const intro = this.engine.params[11] > 0.5 ? 1 : Math.min(1, (now - this.bornMs) / introDur);
		const flash = this.reducedMotion || progress <= 0 ? 0 : Math.exp(-Math.pow((progress - 0.58) / 0.065, 2));
		// Deterministic sim time from the DemoClock (params[10]) so ?frame=N reproduces.
		const simTime = this.engine.params[10] || 0;
		this.uniformData[0] = this.engine.params[6] || 1;
		this.uniformData[1] = this.engine.params[7] || 1;
		this.uniformData[2] = simTime;
		this.uniformData[3] = this.reducedMotion ? 1 : 0;
		this.uniformData[4] = this.burst ? selectedIndex : this.hoveredIndex;
		this.uniformData[5] = this.hoverStrength;
		this.uniformData[6] = selectedIndex;
		this.uniformData[7] = progress;
		this.uniformData[8] = portalX;
		this.uniformData[9] = portalY;
		this.uniformData[10] = 0;
		this.uniformData[11] = flash;
		this.uniformData[12] = intro;
		this.uniformData[13] = this.placed.length;
		this.uniformData[14] = this.engine.params[5] || 0;
		this.uniformData[15] = 0;
		device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
	}

	render(pass: GPURenderPassEncoder): void {
		if (!this.pipeline || !this.bindGroup || this.count === 0) return;
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.bindGroup);
		pass.setVertexBuffer(0, this.particleBuffer);
		pass.draw(6, this.count);
	}

	pickAt(ndcX: number, ndcY: number): { index: number; href: string } | null {
		if (this.burst || this.placed.length === 0) return null;
		const w = this.engine.params[6] || 1;
		const hgt = this.engine.params[7] || 1;
		const aspect = w / hgt;
		let best = -1;
		let bestScore = Infinity;
		for (let i = 0; i < this.placed.length; i++) {
			const p = this.placed[i];
			const dx = (ndcX - p.x) * aspect;
			const dy = ndcY - p.y;
			let score = Math.hypot(dx, dy) / (p.scale * 1.35);
			if (i === this.hoveredIndex) score *= 0.78;
			if (score < 1.1 && score < bestScore) {
				best = i;
				bestScore = score;
			}
		}
		return best < 0 ? null : { index: best, href: this.placed[best].href };
	}

	getScreenPositions(): PalaceSwarmScreenPos[] {
		return this.placed.map((p) => ({
			href: p.href,
			ndcX: p.x,
			ndcY: p.y,
			depth: Math.min(1, Math.max(0, 0.72 + p.z)),
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

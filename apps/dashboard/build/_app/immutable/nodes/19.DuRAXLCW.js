import{$ as e,F as t,H as n,I as r,J as i,K as a,L as o,P as s,b as c,c as l,et as u,it as d,j as f,k as p,lt as m,mt as h,n as g,rt as _,tt as v,ut as y}from"../chunks/Cmba2AGc.js";import{s as b}from"../chunks/DUBTf18l.js";import"../chunks/xihTtKlq.js";import{t as x}from"../chunks/CwlW7yhR.js";import{i as S}from"../chunks/BlHNHjJX.js";import{t as C}from"../chunks/RAF6wBhJ.js";import{t as w}from"../chunks/1HeR1uoy.js";import{i as T,n as E,o as D,t as O,u as k}from"../chunks/X6OYkA-n.js";var ee=12,te=16,ne=1e3,A=200,re=.9,ie={reasoning:D.bridge,memory:D.recall,immune:T.veto,signal:E.supersession,temporal:E.txShadow,system:D.luciferin},j={"/observatory":0,"/graph":1,"/memories":2,"/timeline":3,"/blackbox":4,"/reasoning":5,"/explore":6,"/feed":7,"/contradictions":8},M={"/observatory":[0,0,.12,.26],"/graph":[-.78,.5,.04,.15],"/memories":[.8,.46,-.04,.15],"/timeline":[-.9,-.06,.1,.14],"/reasoning":[.92,-.02,-.08,.15],"/blackbox":[-.72,-.56,.18,.14],"/feed":[.74,-.54,.02,.14],"/explore":[-.02,.66,-.13,.14],"/contradictions":[0,-.66,-.02,.14]},N=[0,0],P=`
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
		let cortexP = vec2<f32>(${N[0]}, ${N[1]});
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
	let cortex = vec2<f32>(${N[0]}, ${N[1]});
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
`;function F(e){return e-Math.floor(e)}function I(e){let t=e<0?0:e>1?1:e;return t*t*(3-2*t)}function L(e){return F(Math.sin(e*12.9898+78.233)*43758.5453)}var R=Math.PI*2;function z(e){let t=L(e*.7351+1),n=L(e*1.2971+5),r=L(e*2.1637+9),i=L(e*3.5391+13);if(t<.15){let e=2*n-1,t=Math.sqrt(Math.max(0,1-e*e)),a=r*R,o=.2+.16*Math.cbrt(i);return[t*Math.cos(a)*o,e*o,t*Math.sin(a)*o]}let a=n<.5?-1:1,o=2*r-1,s=Math.acos(Math.max(-1,Math.min(1,o))),c=Math.sin(s),l=.2,u=a===1?l+i*(Math.PI-2*l):Math.PI+l+i*(Math.PI-2*l),d=c*Math.cos(u),f=o,p=c*Math.sin(u),m=.095*Math.sin(7*u)*(c*c)+.075*Math.cos(9*u+2*s)*c+.065*Math.sin(6*s)+.05*Math.sin(11*u)*Math.sin(4*s),h=Math.min(Math.abs(u-Math.PI/2),Math.abs(u-3*Math.PI/2)),g=-.16*Math.exp(-(h*h)/(.1*.1))*c,_=.02*(L(e*5.19+2)-.5),v=.92+m+g+_;return d*=v,f*=v,p*=v,d+=a*.05*Math.exp(-(h*h)/.0256)*c,f*=.9,p*=1.06,d*=1.04,[d,f,p]}function B(e){let t=.05,n=.66,r=L(e*1.17+3),i=L(e*1.91+7),a=L(e*2.37+11),o=L(e*3.11+5),s=Math.floor(r*12),c=s%3,l=[.55,.05,-.45][c],u=Math.floor(s/3)/4*R+c*.9,d=Math.sqrt(Math.max(0,1-l*l)),f=d*Math.cos(u),p=d*Math.sin(u),m=l,h=i*R,g=Math.sqrt(2*(1+m)),_=(1+m)*Math.cos(h)/g,v=(f*Math.sin(h)-p*Math.cos(h))/g,y=(f*Math.cos(h)+p*Math.sin(h))/g,b=1-(1+m)*Math.sin(h)/g;if(Math.abs(b)<.12){let e=h+Math.PI,r=(1+m)*Math.cos(e)/g,i=(f*Math.sin(e)-p*Math.cos(e))/g,s=(f*Math.cos(e)+p*Math.sin(e))/g,c=(1+m)*Math.sin(e)/g,l=r/(1-c),u=i/(1-c),d=s/(1-c),_=Math.hypot(l,u,d)||1,v=1.6*Math.tanh(_/1.6);l*=v/_,u*=v/_,d*=v/_;let y=a*R,b=t*Math.sqrt(o);return[(d+(o-.5)*t)*n,(u+Math.sin(y)*b)*n,(l+Math.cos(y)*b)*n]}let x=_/b,S=v/b,C=y/b,w=Math.hypot(x,S,C),T=1.55*Math.tanh(w/1.55),E=w>1e-6?T/w:0;x*=E,S*=E,C*=E;let D=a*R,O=t*Math.sqrt(o);return x+=Math.cos(D)*O,S+=Math.sin(D)*O,C+=(o-.5)*t,[C*n,S*n,x*n]}var V=[{radius:1.18,cells:210},{radius:.86,cells:132},{radius:.54,cells:72}],H=414,ae=Math.PI*(3-Math.sqrt(5));function oe(e){let t=Math.floor(L(e*1.6180339887+.5)*H);t>=H&&(t=413);let n=0,r=t;r>=V[0].cells&&(r-=V[0].cells,n=1),n===1&&r>=V[1].cells&&(r-=V[1].cells,n=2);let i=V[n],a=i.cells,o=1-(2*r+1)/a,s=Math.sqrt(Math.max(0,1-o*o)),c=ae*r,l=Math.cos(c)*s,u=o,d=Math.sin(c)*s,f=.05+.012*(2-n),p=L(e*2.31+5),m=L(e*3.97+9),h=f*Math.cbrt(L(e*4.13+1)),g=2*p-1,_=Math.sqrt(Math.max(0,1-g*g)),v=m*R,y=h*_*Math.cos(v),b=h*g,x=h*_*Math.sin(v),S=i.radius;return[(l*S+y)*1.06,(u*S+b)*1.06,(d*S+x)*1.06]}function U(e){let t=Math.SQRT1_2,n=L(e*1.7+3.1),r=Math.floor(L(e*.31+9.2)*6)/6*R,i=n*R*3+r,a=n*R*2+r*1.618,o=t*Math.cos(i),s=t*Math.sin(i),c=t*Math.cos(a),l=1.28-t*Math.sin(a),u=o/l,d=s/l,f=c/l,p=.045,m=L(e*2.3+1)*R,h=Math.sqrt(L(e*3.7+2))*p;u+=Math.cos(m)*h,d+=Math.sin(m)*h,f+=(L(e*4.1+5)-.5)*p*2;let g=1.02,_=.5,v=Math.cos(_),y=Math.sin(_),b=d*v-f*y,x=d*y+f*v;return[u*g,b*g,x*g]}var W=(1+Math.sqrt(5))/2,G=(()=>{let e=W,t=[[0,1,e],[0,1,-e],[0,-1,e],[0,-1,-e],[1,e,0],[1,-e,0],[-1,e,0],[-1,-e,0],[e,0,1],[e,0,-1],[-e,0,1],[-e,0,-1]],n=Math.hypot(0,1,e);return t.map(e=>[e[0]/n,e[1]/n,e[2]/n])})();function se(e,t,n){let r=-2;for(let i of G){let a=Math.abs(e*i[0]+t*i[1]+n*i[2]);a>r&&(r=a)}return .86+.14*r}function K(e){let t=1.16,n=.56,r=L(e+2);if(r<.46){if(L(e+4)<.82){let n=Math.floor(L(e+6)*3)%3,r=L(e+8)<.5?-1:1,i=L(e+10)<.5?-1:1,a=-1.16+2*t*L(e+12),o=(L(e+14)-.5)*.03,s=[0,0,0];return s[n]=a+o,s[(n+1)%3]=r*t+o,s[(n+2)%3]=i*t+o,s}let n=Math.floor(L(e+16)*6),r=Math.floor(n/2),i=n%2?1:-1,a=L(e+18),o=L(e+20)<.5?1:-1,s=-1.16+2*t*a,c=o*(-1.16+2*t*a),l=[0,0,0];return l[r]=i*t,l[(r+1)%3]=s,l[(r+2)%3]=c,l}if(r<.86){let t=Math.floor(L(e+22)*1600),r=1-2*t/1599,i=Math.sqrt(Math.max(0,1-r*r)),a=2.399963229728653*t,o=Math.cos(a)*i,s=r,c=Math.sin(a)*i,l=n*se(o,s,c);return[o*l,s*l,c*l]}let i=Math.floor(L(e+24)*8),a=i&1?1:-1,o=i&2?1:-1,s=i&4?1:-1,c=n*.62,l=L(e+26),u=(L(e+28)-.5)*.025,d=a*c,f=o*c,p=s*c,m=a*t,h=o*t,g=s*t;return[d+(m-d)*l+u,f+(h-f)*l+u,p+(g-p)*l+u]}function ce(e){let t=.7,n=3.5,r=(L(e+1)-.5)*1.2,i=(L(e+2)-.5)*1.2,a=(L(e+3)-.5)*.6+.4,o=.01,s=520+Math.floor(L(e+7)*260);for(let e=0;e<s;e++){let e=(a-t)*r-n*i,s=n*r+(a-t)*i,c=.6+.95*a-a*a*a/3-(r*r+i*i)*(1+.25*a)+.1*a*r*r*r;r+=e*o,i+=s*o,a+=c*o}let c=.92;return[r*c,(a-.6)*c*1.05,i*c]}function le(e){let t=2.399963229728653,n=.02;if(L(e*.53+.9)<.12){let t=.11*Math.cbrt(L(e+21)),n=L(e+22)*R,r=Math.acos(2*L(e+23)-1);return[t*Math.sin(r)*Math.cos(n),t*Math.cos(r),t*Math.sin(r)*Math.sin(n)]}let r=Math.floor(L(e*1.373+5.3)*14),i=1-(r+.5)/14*2,a=Math.sqrt(Math.max(0,1-i*i)),o=r*t,s=Math.cos(o)*a,c=i,l=Math.sin(o)*a,u=L(e*.911+1.7)**.9,d=0,f=[];for(let e=0;e<3;e++){let t=.5*.8**e;f.push(t),d+=t}let p=u*d,m=0,h=0,g=0,_=0;for(let n=0;n<3;n++){if(n>0){let i=((L(e*(1+n*.41)+n*7.13+3.1)<.5?0:1)-.5)*Math.PI+t*(n+r),a=.42*.9**(n-1)*(.7+.6*L(e+n*11+17)),o=0,u=1,d=0;Math.abs(c)>.9&&(o=1,u=0,d=0);let f=c*d-l*u,p=l*o-s*d,m=s*u-c*o,h=Math.hypot(f,p,m)||1;f/=h,p/=h,m/=h;let g=p*l-m*c,_=m*s-f*l,v=f*c-p*s,y=Math.cos(i),b=Math.sin(i),x=y*f+b*g,S=y*p+b*_,C=y*m+b*v,w=Math.cos(a),T=Math.sin(a);s=w*s+T*x,c=w*c+T*S,l=w*l+T*C;let E=Math.hypot(s,c,l)||1;s/=E,c/=E,l/=E}let i=f[n];if(_+i>=p){let e=(p-_)/i;m+=s*i*e,h+=c*i*e,g+=l*i*e;break}m+=s*i,h+=c*i,g+=l*i,_+=i}let v=.35+.65*Math.sin(Math.min(1,u)*Math.PI);m+=(L(e*3.1+1)-.5)*n*v,h+=(L(e*3.3+2)-.5)*n*v,g+=(L(e*3.7+3)-.5)*n*v;let y=1.3/d;return[m*y,h*y,g*y]}function ue(e){let t=L(e*1.17+3),n=L(e*1.91+7),r=.82,i=t*R,a=Math.floor(n*5),o=i*6+a/5*R+(L(e+5)-.5)*.5,s=.34*(.72+.28*Math.sqrt(L(e+13))),c=(r+s*Math.cos(o))*Math.cos(i),l=(r+s*Math.cos(o))*Math.sin(i),u=s*Math.sin(o),d=1.0821,f=Math.cos(d),p=Math.sin(d),m=u*f-l*p,h=u*p+l*f;u=m,l=h;let g=.2094,_=Math.cos(g),v=Math.sin(g);return[c*_-u*v,c*v+u*_,l]}function de(e){let t=L(e*1.17+3),n=L(e*1.91+7),r=L(e*2.37+11),i=.92,a=[[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],o=[[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]],s=t<.5,c=s?o:a,l=s?-.16:.16;if(n<.72){let t=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]][Math.floor(r*6)%6],n=c[t[0]],a=c[t[1]],o=L(e*3.11+5),s=.5+.5*Math.sign(o-.5)*Math.abs(2*o-1)**.55,u=(L(e*4.7+1)-.5)*.05,d=(L(e*5.3+2)-.5)*.05,f=(L(e*6.1+3)-.5)*.05;return[(n[0]+(a[0]-n[0])*s)*i+u+l,(n[1]+(a[1]-n[1])*s)*i+d,(n[2]+(a[2]-n[2])*s)*i+f]}if(n<.92){let t=c[Math.floor(r*4)%4],n=L(e*7.7+4)**1.5*.11,a=L(e*8.3+6)*R,o=Math.acos(2*L(e*9.1+8)-1);return[t[0]*i+n*Math.sin(o)*Math.cos(a)+l,t[1]*i+n*Math.sin(o)*Math.sin(a),t[2]*i+n*Math.cos(o)]}let u=L(e*10.3+7)*R,d=Math.acos(2*L(e*11.7+9)-1),f=Math.sin(d)*Math.cos(u),p=Math.sin(d)*Math.sin(u),m=Math.cos(d),h=Math.abs(f)+Math.abs(p)+Math.abs(m),g=.34;return[f/h*g,p/h*g,m/h*g]}function fe(e,t){switch(e){case 0:return z(t);case 1:return B(t);case 2:return oe(t);case 3:return U(t);case 4:return K(t);case 5:return ce(t);case 6:return le(t);case 7:return ue(t);default:return de(t)}}function pe(e){let t=navigator.hardwareConcurrency||8,n=window.innerWidth<760,r=t<=4||(window.devicePixelRatio||1)>2.2;return e?n?1e4:16e3:r?n?12e3:2e4:n?18e3:t>=10?55e3:4e4}var me=class{engine;pipeline=null;bindGroup=null;uniformBuffer=null;particleBuffer=null;uniformData=new Float32Array(te);placed=[];count=0;hoveredIndex=-1;hoverStrength=0;reducedMotion=!1;bornMs;burst=null;onFlashPeak=null;watchdog=null;constructor(e){this.engine=e,this.bornMs=e.wallNowMs}setReducedMotion(e){this.reducedMotion=e}setHovered(e){this.burst||(this.hoveredIndex=e>=0&&e<this.placed.length?e:-1)}indexOfHref(e){return e?this.placed.findIndex(t=>t.href===e):-1}get isBursting(){return this.burst!==null}startBurst(e,t){if(this.burst)return!1;let n=this.placed.find(t=>t.href===e);return n?(this.hoveredIndex=this.placed.indexOf(n),this.hoverStrength=1,this.burst={href:e,startMs:this.engine.wallNowMs,callbackFired:!1},this.onFlashPeak=t,this.watchdog=setTimeout(()=>this.fireFlashCallback(),980),!0):!1}uploadRegions(e){let t=this.engine.gpuDevice;if(!t||e.length===0)return;this.placed=e.map((e,t)=>{let n=M[e.href]??[0,0,t*.01,.14];return{href:e.href,x:n[0],y:n[1],z:n[2],scale:n[3],kind:j[e.href]??t%9}}),this.count=pe(this.reducedMotion);let n=new Float32Array(this.count*ee);for(let t=0;t<this.count;t++){let r=t+1,i=t*ee,a=this.routeForIndex(t),o=this.placed[a],s=fe(o.kind,r),c=k(ie[e[a].family]);n[i+0]=s[0],n[i+1]=s[1],n[i+2]=s[2],n[i+3]=L(r*2.83+12),n[i+4]=o.x,n[i+5]=o.y,n[i+6]=o.z,n[i+7]=o.scale,n[i+8]=L(r*1.7+5),n[i+9]=c[0]*.5+c[1]*.3,n[i+10]=c[2]*.6,n[i+11]=a}this.uniformBuffer?.destroy(),this.particleBuffer?.destroy(),this.uniformBuffer=t.createBuffer({label:`palace-brain-uniforms`,size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.particleBuffer=t.createBuffer({label:`palace-brain-particles`,size:n.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,n),this.createPipeline(t)}routeForIndex(e){let t=this.placed.length;if(t===0)return 0;let n=2+(t-1),r=Math.floor(this.count*2/n);if(e<r)return 0;let i=this.count-r,a=Math.max(1,Math.floor(i/Math.max(1,t-1)));return Math.min(t-1,1+Math.floor((e-r)/a))}createPipeline(e){if(!this.uniformBuffer||!this.particleBuffer)return;e.pushErrorScope(`validation`);let t=e.createShaderModule({label:`palace-brain-shader`,code:P});t.getCompilationInfo().then(e=>{let t=e.messages.filter(e=>e.type===`error`);t.length>0&&console.error(`[palace-brain] WGSL:`,t.map(e=>`${e.lineNum}:${e.linePos} ${e.message}`).join(`
`))}),this.pipeline=e.createRenderPipeline({label:`palace-brain-pipeline`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`,buffers:[{arrayStride:48,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`},{shaderLocation:2,offset:32,format:`float32x4`}]}]},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bindGroup=e.createBindGroup({label:`palace-brain-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),e.popErrorScope().then(e=>{e&&console.error(`[palace-brain] pipeline validation:`,e.message)})}fireFlashCallback(){if(!this.burst||this.burst.callbackFired||!this.onFlashPeak)return;this.burst.callbackFired=!0;let e=this.burst.href,t=this.onFlashPeak;this.onFlashPeak=null,this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,queueMicrotask(()=>t(e))}compute(){let e=this.engine.gpuDevice;if(!e||!this.uniformBuffer||this.placed.length===0)return;let t=this.engine.wallNowMs,n=this.burst?1:+(this.hoveredIndex>=0);this.hoverStrength+=(n-this.hoverStrength)*(this.reducedMotion?1:.16),this.hoverStrength<.001&&(this.hoverStrength=0);let r=0,i=-1,a=this.hoveredIndex>=0?this.placed[this.hoveredIndex]:null,o=a?.x??0,s=a?.y??0;if(this.burst){i=this.indexOfHref(this.burst.href),a=i>=0?this.placed[i]:a;let e=this.reducedMotion?A:ne;r=Math.min(1,Math.max(0,(t-this.burst.startMs)/e));let n=this.reducedMotion?.3:re;r>=n&&this.fireFlashCallback();let c=a?.x??0,l=a?.y??0,u=this.reducedMotion?1:I(r/.42);o=c*(1-u),s=l*(1-u)}let c=this.engine.params[11]>.5?1:Math.min(1,(t-this.bornMs)/1650),l=this.reducedMotion||r<=0?0:Math.exp(-(((r-.58)/.065)**2)),u=this.engine.params[10]||0;this.uniformData[0]=this.engine.params[6]||1,this.uniformData[1]=this.engine.params[7]||1,this.uniformData[2]=u,this.uniformData[3]=+!!this.reducedMotion,this.uniformData[4]=this.burst?i:this.hoveredIndex,this.uniformData[5]=this.hoverStrength,this.uniformData[6]=i,this.uniformData[7]=r,this.uniformData[8]=o,this.uniformData[9]=s,this.uniformData[10]=0,this.uniformData[11]=l,this.uniformData[12]=c,this.uniformData[13]=this.placed.length,this.uniformData[14]=this.engine.params[5]||0,this.uniformData[15]=0,e.queue.writeBuffer(this.uniformBuffer,0,this.uniformData)}render(e){this.pipeline&&this.bindGroup&&this.count!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.particleBuffer),e.draw(6,this.count))}pickAt(e,t){if(this.burst||this.placed.length===0)return null;let n=(this.engine.params[6]||1)/(this.engine.params[7]||1),r=-1,i=1/0;for(let a=0;a<this.placed.length;a++){let o=this.placed[a],s=(e-o.x)*n,c=t-o.y,l=Math.hypot(s,c)/(o.scale*1.35);a===this.hoveredIndex&&(l*=.78),l<1.1&&l<i&&(r=a,i=l)}return r<0?null:{index:r,href:this.placed[r].href}}getScreenPositions(){return this.placed.map(e=>({href:e.href,ndcX:e.x,ndcY:e.y,depth:Math.min(1,Math.max(0,.72+e.z)),visible:!0}))}dispose(){this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,this.uniformBuffer?.destroy(),this.particleBuffer?.destroy(),this.uniformBuffer=null,this.particleBuffer=null,this.pipeline=null,this.bindGroup=null,this.placed=[],this.count=0,this.burst=null,this.onFlashPeak=null}},q=[{href:`/observatory`,label:`OBSERVATORY`,family:`system`,center:!0},{href:`/graph`,label:`GRAPH`,family:`memory`},{href:`/memories`,label:`MEMORIES`,family:`memory`},{href:`/timeline`,label:`TIMELINE`,family:`temporal`},{href:`/feed`,label:`FEED`,family:`signal`},{href:`/explore`,label:`EXPLORE`,family:`reasoning`},{href:`/reasoning`,label:`REASONING`,family:`reasoning`},{href:`/blackbox`,label:`BLACK BOX`,family:`signal`},{href:`/contradictions`,label:`CONTRADICTIONS`,family:`immune`}];function J(e){return q.find(t=>t.href===e)}var Y={reasoning:[...k(`#FFFFFF`),1],memory:[...k(`#FFFFFF`),1],immune:[...k(`#FFFFFF`),1],temporal:[...k(`#FFFFFF`),1],signal:[...k(`#FFFFFF`),1],system:[...k(`#FFFFFF`),1]},X=[...k(`#FFFFFF`),1],he=1.35,ge=.028,_e=.6,ve=.85,ye=.05,be=.03,xe=1,Se=.95,Z=[],Q=[];function Ce(e,t,n){return e+(t-e)*n}function $(e){return e<0?0:e>1?1:Number.isFinite(e)?e:0}function we(e,t={}){let n=t.hoveredHref??null,r=t.dimUnhovered??!0,i=t.aspect??1,a=i<.85,o=.7-.06*(a?$((.85-i)/(.85-.46)):0),s=0;Q.length=0;for(let t=0;t<e.length;t++){let i=e[t];if(i.visible===!1)continue;let c=J(i.href);if(!c)continue;let l=$(i.depth),u=n!==null&&i.href===n,d=ge+l*he,f=i.ndcX+d*ve,p=i.ndcY+d*_e;a&&p>o&&(p=o-(p-o));let m=Ce(be,ye,l),h=Ce(Se,xe,l);c.center&&(m*=1.22),u?(m*=1.28,h=1):r&&n!==null&&(h*=.32);let g=c.label.length*m*.62;f+g>.97&&(f=i.ndcX-d*ve-g);for(let e=0;e<Q.length;e++){let t=Q[e],n=f<t.x1&&f+g>t.x0,r=Math.abs(p-t.y)<(m+t.size)*.75;n&&r&&(p=t.y-(m+t.size)*.85)}Q.push({x0:f,x1:f+g,y:p,size:m});let _=c.center?X:Y[c.family],v=[_[0],_[1],_[2],$(h)],y=Z[s];y||(y={id:``,kind:`palace-label`,text:``,x:0,y:0,size:0,color:[0,0,0,0],depth:0,weight:.75,revealSpan:1,maxWidthEm:24},Z[s]=y),y.id=`palace-label:`+i.href,y.kind=`palace-label`,y.text=c.label,y.x=f,y.y=p,y.size=m,y.color=v,y.depth=u?1:.8+l*.2,y.weight=.95,s++}return Z.length=s,Z}q.length;var Te=f(`<div class="palace-host fixed inset-0 bg-[#020307] svelte-1dx67o8" role="application" aria-label="VestigeOS Memory Palace. Nine living cognitive organs. Use the Command palette for keyboard navigation."><!></div>`);function Ee(s,f){y(f,!0);let ee=()=>d(x,`$page`,te),[te,ne]=_(),A=[...k(`#F5FFF2`),1],re=[...k(`#9DFFEB`),1],ie=[...k(`#7DAFA9`),.82],j={reasoning:D.bridge,memory:D.recall,immune:T.veto,signal:E.supersession,temporal:E.txShadow,system:D.luciferin},M=u(null),N=null,P=null,F=null,I=null,L=u(null),R=null,z={x:0,y:0},B=!1,V=!1,H=-1,ae=v(()=>{let e=ee().url.searchParams.get(`frame`);if(e===null)return null;let t=Number(e);return Number.isFinite(t)?Math.floor(t):null});g(()=>{N&&P?N.removePass(P):P?.dispose(),N&&F?N.removePass(F):F?.dispose(),P=null,F=null,N=null});async function oe(e){try{N=e,V=window.matchMedia(`(prefers-reduced-motion: reduce)`).matches;let t=new me(e);P=t,t.setReducedMotion(V),e.addPass(t),t.uploadRegions(q);let n=new O(e);F=n,await n.init(),e.addPass(n),e.demoClock.reset(),n.setText(W())}catch(e){console.error(`[palace] Failed to initialize swarm:`,e)}}function U(e){return e.replace(/[—–]/g,`-`).replace(/[‘’]/g,`'`).replace(/[“”]/g,`"`).replace(/…/g,`...`).replace(/[^\x20-\x7E]/g,`?`)}function W(){let e=[{id:`palace:title`,kind:`palace-hud`,text:`VESTIGE // MEMORY PALACE`,x:-.92,y:.88,size:.052,color:A,depth:1,weight:1,revealSpan:24},{id:`palace:sub`,kind:`palace-hud`,text:U(`${q.length} LIVING ORGANS - HOVER TO REVEAL - CLICK TO ENTER`),x:-.92,y:.8,size:.025,color:re,depth:1,weight:.86,revealSpan:28,maxWidthEm:66},{id:`palace:hint`,kind:`palace-hud`,text:B?`PORTAL LOCKED // COLLAPSING COGNITIVE FIELD`:`MOVE THROUGH THE FIELD`,x:-.92,y:-.87,size:.02,color:ie,depth:.9,weight:.72,revealSpan:18}];if(o(L)){let t=J(o(L)),n=S(o(L));if(t){let r=[...k(j[t.family]),1];e.push({id:`palace:focus-label`,kind:`palace-focus`,text:B?`ENTERING ${t.label}`:t.label,x:.34,y:.88,size:.046,color:r,depth:1,weight:1,revealSpan:18},{id:`palace:focus-purpose`,kind:`palace-focus`,text:U(n?.purpose??`ENTER THIS COGNITIVE ORGAN`),x:.34,y:.8,size:.021,color:A,depth:1,weight:.8,revealSpan:30,maxWidthEm:42})}}return e}function G(){let e=W(),t=P?we(P.getScreenPositions(),{hoveredHref:o(L),dimUnhovered:!!o(L),aspect:(N?.params[6]||0)/Math.max(1,N?.params[7]||1)}):[];F?.setText([...e,...t])}function se(e){e!==H&&e%2==0&&(H=e,G())}function K(e){if(!o(M))return null;let t=o(M).getBoundingClientRect();return t.width<=0||t.height<=0?null:{x:(e.clientX-t.left)/t.width*2-1,y:-((e.clientY-t.top)/t.height*2-1)}}function ce(e){if(!o(M)||!N)return;let t=o(M).getBoundingClientRect(),n=Math.max(1e-4,t.width/Math.max(1,t.height)),r={x:e.x*Math.max(n,1),y:e.y/Math.min(n,1)},i=I??r,a={x:i.x+(r.x-i.x)*.35,y:i.y+(r.y-i.y)*.35};I=a,N.setCursorPreNdc(a.x,a.y,a.x-i.x,a.y-i.y)}function le(t){let n=K(t);if(!n||(ce(n),!P||P.isBursting))return;let r=P.pickAt(n.x,n.y),i=r?.href??null;i!==o(L)&&(e(L,i,!0),P.setHovered(r?.index??-1),G(),o(M)&&(o(M).style.cursor=i?`pointer`:`crosshair`))}function ue(){P?.isBursting||(R=null,I=null,e(L,null),P?.setHovered(-1),N?.setCursorPreNdc(999,999,0,0),G(),o(M)&&(o(M).style.cursor=`crosshair`))}function de(e){if(e.button!==0||!P||P.isBursting)return;let t=K(e);t&&(R={x:e.clientX,y:e.clientY,href:P.pickAt(t.x,t.y)?.href??null})}function fe(){R=null}function pe(t){let n=R;if(R=null,!n||!P||P.isBursting||Math.hypot(t.clientX-n.x,t.clientY-n.y)>9)return;let r=K(t);if(!r)return;let i=P.pickAt(r.x,r.y);i&&i.href===n.href&&(z={x:t.clientX,y:t.clientY},e(L,i.href,!0),P.setHovered(i.index),B=!0,G(),o(M)&&(o(M).style.cursor=`wait`),P.startBurst(i.href,Y)||Y(i.href))}async function Y(e){let t=J(e);await C(`${b}${e}`,{clientX:z.x,clientY:z.y,color:t?j[t.family]:D.luciferin,reduced:V})}var X=Te();c(`1dx67o8`,e=>{n(()=>{a.title=`Memory Palace · VestigeOS`})});var he=i(X);w(he,{demo:`recall-path`,seed:`vestige-palace-swarm-v2`,get freezeFrame(){return o(ae)},onframe:se,onready:oe}),h(X),l(X,t=>e(M,t),()=>o(M)),t(`pointerdown`,X,de),t(`pointerup`,X,pe),r(`pointercancel`,X,fe),t(`pointermove`,X,le),r(`pointerleave`,X,ue),p(s,X),m(),ne()}s([`pointerdown`,`pointerup`,`pointermove`]);export{Ee as component};
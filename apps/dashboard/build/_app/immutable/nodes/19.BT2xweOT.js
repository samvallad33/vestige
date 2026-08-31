var ue=Object.defineProperty;var ge=(e,t,a)=>t in e?ue(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a;var C=(e,t,a)=>ge(e,typeof t!="symbol"?t+"":t,a);import"../chunks/Bzak7iHL.js";import{d as me,a as _e,b as oe,e as le}from"../chunks/DjkTV3j6.js";import{p as xe,a as be,b as ve,i as ye,g as D,f as we,u as Me,c as Ee,s as Z,d as he,$ as ke,r as Te}from"../chunks/xobCzZoX.js";import{h as Ae}from"../chunks/NaLbXVuk.js";import{b as Pe}from"../chunks/B5_AEu2I.js";import{a as Se,s as Ne}from"../chunks/Zx1n3Gz4.js";import{p as Re}from"../chunks/Cgb7tf3Y.js";import{b as Oe}from"../chunks/BobykrIY.js";import{O as ze}from"../chunks/DCMW_a2l.js";import{r as K,R as Y,B as ee,I as de,T as Ie}from"../chunks/J7RHgeFk.js";import{f as Ce}from"../chunks/C3quBBy0.js";import{b as De}from"../chunks/CSZhxiOi.js";const re=12,Le=16,Be=1e3,Ue=200,Fe=.9,He={reasoning:Y.bridge,memory:Y.recall,immune:de.veto,signal:ee.supersession,temporal:ee.txShadow,system:Y.luciferin},Ge={"/observatory":0,"/graph":1,"/memories":2,"/timeline":3,"/blackbox":4,"/reasoning":5,"/explore":6,"/feed":7,"/contradictions":8},je={"/observatory":[0,0,.12,.26],"/graph":[-.78,.5,.04,.15],"/memories":[.8,.46,-.04,.15],"/timeline":[-.9,-.06,.1,.14],"/reasoning":[.92,-.02,-.08,.15],"/blackbox":[-.72,-.56,.18,.14],"/feed":[.74,-.54,.02,.14],"/explore":[-.02,.66,-.13,.14],"/contradictions":[0,-.66,-.02,.14]},Q=[0,0],Ve=`
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

// 5-stop iridescent memory spectrum: magenta to violet to blue to cyan to
// emerald (wraps). s flows over time so the whole cloud is never static.
fn spectrum(s: f32) -> vec3<f32> {
	let magenta = vec3<f32>(0.98, 0.24, 0.86);
	let violet  = vec3<f32>(0.56, 0.34, 1.00);
	let blue    = vec3<f32>(0.22, 0.46, 1.00);
	let cyan    = vec3<f32>(0.14, 0.86, 0.98);
	let emerald = vec3<f32>(0.22, 0.96, 0.58);
	let x = fract(s) * 5.0;
	if (x < 1.0) { return mix(magenta, violet, x); }
	if (x < 2.0) { return mix(violet, blue, x - 1.0); }
	if (x < 3.0) { return mix(blue, cyan, x - 2.0); }
	if (x < 4.0) { return mix(cyan, emerald, x - 3.0); }
	return mix(emerald, magenta, x - 4.0);
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
		let cortexP = vec2<f32>(${Q[0]}, ${Q[1]});
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
	let cortex = vec2<f32>(${Q[0]}, ${Q[1]});
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
`;function Ye(e){return e-Math.floor(e)}function qe(e){const t=e<0?0:e>1?1:e;return t*t*(3-2*t)}function r(e){return Ye(Math.sin(e*12.9898+78.233)*43758.5453)}const B=Math.PI*2;function We(e){const t=r(e*.7351+1),a=r(e*1.2971+5),s=r(e*2.1637+9),d=r(e*3.5391+13);if(t<.15){const v=2*a-1,k=Math.sqrt(Math.max(0,1-v*v)),T=s*B,E=.2+.16*Math.cbrt(d);return[k*Math.cos(T)*E,v*E,k*Math.sin(T)*E]}const u=a<.5?-1:1,i=2*s-1,h=Math.acos(Math.max(-1,Math.min(1,i))),o=Math.sin(h),l=.2,c=u===1?l+d*(Math.PI-2*l):Math.PI+l+d*(Math.PI-2*l);let n=o*Math.cos(c),g=i,w=o*Math.sin(c);const m=.095*Math.sin(7*c)*(o*o)+.075*Math.cos(9*c+2*h)*o+.065*Math.sin(6*h)+.05*Math.sin(11*c)*Math.sin(4*h),f=Math.min(Math.abs(c-Math.PI/2),Math.abs(c-3*Math.PI/2)),x=-.16*Math.exp(-(f*f)/(.1*.1))*o,y=.02*(r(e*5.19+2)-.5),b=.92+m+x+y;return n*=b,g*=b,w*=b,n+=u*.05*Math.exp(-(f*f)/(.16*.16))*o,g*=.9,w*=1.06,n*=1.04,[n,g,w]}function Xe(e){const d=r(e*1.17+3),u=r(e*1.91+7),i=r(e*2.37+11),h=r(e*3.11+5),o=Math.floor(d*12),l=o%3,c=[.55,.05,-.45][l],n=Math.ceil(12/3),w=Math.floor(o/3)/n*B+l*.9,m=Math.sqrt(Math.max(0,1-c*c)),f=m*Math.cos(w),x=m*Math.sin(w),y=c,b=u*B,v=Math.sqrt(2*(1+y)),k=(1+y)*Math.cos(b)/v,T=(f*Math.sin(b)-x*Math.cos(b))/v,E=(f*Math.cos(b)+x*Math.sin(b))/v;let A=1-(1+y)*Math.sin(b)/v;if(Math.abs(A)<.12){const H=b+Math.PI,_=(1+y)*Math.cos(H)/v,p=(f*Math.sin(H)-x*Math.cos(H))/v,M=(f*Math.cos(H)+x*Math.sin(H))/v,S=(1+y)*Math.sin(H)/v;let L=_/(1-S),G=p/(1-S),X=M/(1-S);const q=Math.hypot(L,G,X)||1,V=1.6*Math.tanh(q/1.6);L*=V/q,G*=V/q,X*=V/q;const W=i*B,J=.05*Math.sqrt(h);return[(X+(h-.5)*.05)*.66,(G+Math.sin(W)*J)*.66,(L+Math.cos(W)*J)*.66]}let O=k/A,F=T/A,P=E/A;const N=Math.hypot(O,F,P),z=1.55*Math.tanh(N/1.55),U=N>1e-6?z/N:0;O*=U,F*=U,P*=U;const j=i*B,I=.05*Math.sqrt(h);return O+=Math.cos(j)*I,F+=Math.sin(j)*I,P+=(h-.5)*.05,[P*.66,F*.66,O*.66]}const $=[{radius:1.18,cells:210},{radius:.86,cells:132},{radius:.54,cells:72}],ne=414,$e=Math.PI*(3-Math.sqrt(5));function Ke(e){let t=Math.floor(r(e*1.6180339887+.5)*ne);t>=ne&&(t=ne-1);let a=0,s=t;s>=$[0].cells&&(s-=$[0].cells,a=1),a===1&&s>=$[1].cells&&(s-=$[1].cells,a=2);const d=$[a],u=d.cells,i=1-(2*s+1)/u,h=Math.sqrt(Math.max(0,1-i*i)),o=$e*s,l=Math.cos(o)*h,c=i,n=Math.sin(o)*h,g=.05+.012*(2-a),w=r(e*2.31+5),m=r(e*3.97+9),f=g*Math.cbrt(r(e*4.13+1)),x=2*w-1,y=Math.sqrt(Math.max(0,1-x*x)),b=m*B,v=f*y*Math.cos(b),k=f*x,T=f*y*Math.sin(b),E=d.radius;return[(l*E+v)*1.06,(c*E+k)*1.06,(n*E+T)*1.06]}function Je(e){const t=Math.SQRT1_2,a=r(e*1.7+3.1),s=6,u=Math.floor(r(e*.31+9.2)*s)/s*B,i=a*B*3+u,h=a*B*2+u*1.618,o=t*Math.cos(i),l=t*Math.sin(i),c=t*Math.cos(h),g=1.28-t*Math.sin(h);let w=o/g,m=l/g,f=c/g;const x=.045,y=r(e*2.3+1)*B,b=Math.sqrt(r(e*3.7+2))*x;w+=Math.cos(y)*b,m+=Math.sin(y)*b,f+=(r(e*4.1+5)-.5)*x*2;const v=1.02,k=.5,T=Math.cos(k),E=Math.sin(k),R=m*T-f*E,A=m*E+f*T;return[w*v,R*v,A*v]}const Ze=(1+Math.sqrt(5))/2,Qe=(()=>{const e=Ze,t=[[0,1,e],[0,1,-e],[0,-1,e],[0,-1,-e],[1,e,0],[1,-e,0],[-1,e,0],[-1,-e,0],[e,0,1],[e,0,-1],[-e,0,1],[-e,0,-1]],a=Math.hypot(0,1,e);return t.map(s=>[s[0]/a,s[1]/a,s[2]/a])})();function et(e,t,a){let s=-2;for(const d of Qe){const u=Math.abs(e*d[0]+t*d[1]+a*d[2]);u>s&&(s=u)}return .86+.14*s}function tt(e){const s=r(e+2);if(s<.46){if(r(e+4)<.82){const O=Math.floor(r(e+6)*3)%3,F=r(e+8)<.5?-1:1,P=r(e+10)<.5?-1:1,N=-1.16+2*1.16*r(e+12),z=(r(e+14)-.5)*.03,U=[0,0,0];return U[O]=N+z,U[(O+1)%3]=F*1.16+z,U[(O+2)%3]=P*1.16+z,U}const y=Math.floor(r(e+16)*6),b=Math.floor(y/2),v=y%2?1:-1,k=r(e+18),T=r(e+20)<.5?1:-1,E=-1.16+2*1.16*k,R=T*(-1.16+2*1.16*k),A=[0,0,0];return A[b]=v*1.16,A[(b+1)%3]=E,A[(b+2)%3]=R,A}if(s<.86){const b=Math.floor(r(e+22)*1600),v=1-2*b/1599,k=Math.sqrt(Math.max(0,1-v*v)),T=2.399963229728653*b,E=Math.cos(T)*k,R=v,A=Math.sin(T)*k,O=.56*et(E,R,A);return[E*O,R*O,A*O]}const d=Math.floor(r(e+24)*8),u=d&1?1:-1,i=d&2?1:-1,h=d&4?1:-1,o=.56*.62,l=r(e+26),c=(r(e+28)-.5)*.025,n=u*o,g=i*o,w=h*o,m=u*1.16,f=i*1.16,x=h*1.16;return[n+(m-n)*l+c,g+(f-g)*l+c,w+(x-w)*l+c]}function at(e){let h=(r(e+1)-.5)*1.2,o=(r(e+2)-.5)*1.2,l=(r(e+3)-.5)*.6+.4;const c=.01,g=520+Math.floor(r(e+7)*260);for(let m=0;m<g;m++){const f=(l-.7)*h-3.5*o,x=3.5*h+(l-.7)*o,y=.6+.95*l-l*l*l/3-(h*h+o*o)*(1+.25*l)+.1*l*h*h*h;h+=f*c,o+=x*c,l+=y*c}const w=.92;return[h*w,(l-.6)*w*1.05,o*w]}function st(e){const i=2.399963229728653,h=.12,o=.02,l=1.3;if(r(e*.53+.9)<h){const P=.11*Math.cbrt(r(e+21)),N=r(e+22)*B,z=Math.acos(2*r(e+23)-1);return[P*Math.sin(z)*Math.cos(N),P*Math.cos(z),P*Math.sin(z)*Math.sin(N)]}const c=Math.floor(r(e*1.373+5.3)*14),n=1-(c+.5)/14*2,g=Math.sqrt(Math.max(0,1-n*n)),w=c*i;let m=Math.cos(w)*g,f=n,x=Math.sin(w)*g;const y=Math.pow(r(e*.911+1.7),.9);let b=0;const v=[];for(let P=0;P<3;P++){const N=.5*Math.pow(.8,P);v.push(N),b+=N}const k=y*b;let T=0,E=0,R=0,A=0;for(let P=0;P<3;P++){if(P>0){const U=((r(e*(1+P*.41)+P*7.13+3.1)<.5?0:1)-.5)*Math.PI+i*(P+c),j=.42*Math.pow(.9,P-1)*(.7+.6*r(e+P*11+17));let I=0,H=1,_=0;Math.abs(f)>.9&&(I=1,H=0,_=0);let p=f*_-x*H,M=x*I-m*_,S=m*H-f*I;const L=Math.hypot(p,M,S)||1;p/=L,M/=L,S/=L;const G=M*x-S*f,X=S*m-p*x,q=p*f-M*m,V=Math.cos(U),W=Math.sin(U),J=V*p+W*G,pe=V*M+W*X,fe=V*S+W*q,te=Math.cos(j),ae=Math.sin(j);m=te*m+ae*J,f=te*f+ae*pe,x=te*x+ae*fe;const se=Math.hypot(m,f,x)||1;m/=se,f/=se,x/=se}const N=v[P];if(A+N>=k){const z=(k-A)/N;T+=m*N*z,E+=f*N*z,R+=x*N*z;break}T+=m*N,E+=f*N,R+=x*N,A+=N}const O=.35+.65*Math.sin(Math.min(1,y)*Math.PI);T+=(r(e*3.1+1)-.5)*o*O,E+=(r(e*3.3+2)-.5)*o*O,R+=(r(e*3.7+3)-.5)*o*O;const F=l/b;return[T*F,E*F,R*F]}function ot(e){const t=r(e*1.17+3),a=r(e*1.91+7),s=.82,d=.34,u=5,i=6,h=t*B,o=Math.floor(a*u),l=h*i+o/u*B+(r(e+5)-.5)*.5,c=.72+.28*Math.sqrt(r(e+13)),n=d*c;let g=(s+n*Math.cos(l))*Math.cos(h),w=(s+n*Math.cos(l))*Math.sin(h),m=n*Math.sin(l);const f=1.0821,x=Math.cos(f),y=Math.sin(f),b=m*x-w*y,v=m*y+w*x;m=b,w=v;const k=.2094,T=Math.cos(k),E=Math.sin(k),R=g*T-m*E,A=g*E+m*T;return[R,A,w]}function rt(e){const t=r(e*1.17+3),a=r(e*1.91+7),s=r(e*2.37+11),d=.92,u=[[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],i=[[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]],h=t<.5,o=h?i:u,l=h?-.16:.16;if(a<.72){const b=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]][Math.floor(s*6)%6],v=o[b[0]],k=o[b[1]],T=r(e*3.11+5),E=.5+.5*Math.sign(T-.5)*Math.pow(Math.abs(2*T-1),.55),R=(r(e*4.7+1)-.5)*.05,A=(r(e*5.3+2)-.5)*.05,O=(r(e*6.1+3)-.5)*.05;return[(v[0]+(k[0]-v[0])*E)*d+R+l,(v[1]+(k[1]-v[1])*E)*d+A,(v[2]+(k[2]-v[2])*E)*d+O]}if(a<.92){const y=o[Math.floor(s*4)%4],b=Math.pow(r(e*7.7+4),1.5)*.11,v=r(e*8.3+6)*B,k=Math.acos(2*r(e*9.1+8)-1);return[y[0]*d+b*Math.sin(k)*Math.cos(v)+l,y[1]*d+b*Math.sin(k)*Math.sin(v),y[2]*d+b*Math.cos(k)]}const c=r(e*10.3+7)*B,n=Math.acos(2*r(e*11.7+9)-1),g=Math.sin(n)*Math.cos(c),w=Math.sin(n)*Math.sin(c),m=Math.cos(n),f=Math.abs(g)+Math.abs(w)+Math.abs(m),x=.34;return[g/f*x,w/f*x,m/f*x]}function nt(e,t){switch(e){case 0:return We(t);case 1:return Xe(t);case 2:return Ke(t);case 3:return Je(t);case 4:return tt(t);case 5:return at(t);case 6:return st(t);case 7:return ot(t);default:return rt(t)}}function it(e){const t=navigator.hardwareConcurrency||8,a=window.innerWidth<760,s=t<=4||(window.devicePixelRatio||1)>2.2;return e?a?1e4:16e3:s?a?12e3:2e4:a?18e3:t>=10?55e3:4e4}class lt{constructor(t){C(this,"engine");C(this,"pipeline",null);C(this,"bindGroup",null);C(this,"uniformBuffer",null);C(this,"particleBuffer",null);C(this,"uniformData",new Float32Array(Le));C(this,"placed",[]);C(this,"count",0);C(this,"hoveredIndex",-1);C(this,"hoverStrength",0);C(this,"reducedMotion",!1);C(this,"bornMs");C(this,"burst",null);C(this,"onFlashPeak",null);C(this,"watchdog",null);this.engine=t,this.bornMs=t.wallNowMs}setReducedMotion(t){this.reducedMotion=t}setHovered(t){this.burst||(this.hoveredIndex=t>=0&&t<this.placed.length?t:-1)}indexOfHref(t){return t?this.placed.findIndex(a=>a.href===t):-1}get isBursting(){return this.burst!==null}startBurst(t,a){if(this.burst)return!1;const s=this.placed.find(d=>d.href===t);return s?(this.hoveredIndex=this.placed.indexOf(s),this.hoverStrength=1,this.burst={href:t,startMs:this.engine.wallNowMs,callbackFired:!1},this.onFlashPeak=a,this.watchdog=setTimeout(()=>this.fireFlashCallback(),980),!0):!1}uploadRegions(t){var d,u;const a=this.engine.gpuDevice;if(!a||t.length===0)return;this.placed=t.map((i,h)=>{const o=je[i.href]??[0,0,h*.01,.14];return{href:i.href,x:o[0],y:o[1],z:o[2],scale:o[3],kind:Ge[i.href]??h%9}}),this.count=it(this.reducedMotion);const s=new Float32Array(this.count*re);for(let i=0;i<this.count;i++){const h=i+1,o=i*re,l=this.routeForIndex(i),c=this.placed[l],n=nt(c.kind,h),g=K(He[t[l].family]);s[o+0]=n[0],s[o+1]=n[1],s[o+2]=n[2],s[o+3]=r(h*2.83+12),s[o+4]=c.x,s[o+5]=c.y,s[o+6]=c.z,s[o+7]=c.scale,s[o+8]=r(h*1.7+5),s[o+9]=g[0]*.5+g[1]*.3,s[o+10]=g[2]*.6,s[o+11]=l}(d=this.uniformBuffer)==null||d.destroy(),(u=this.particleBuffer)==null||u.destroy(),this.uniformBuffer=a.createBuffer({label:"palace-brain-uniforms",size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.particleBuffer=a.createBuffer({label:"palace-brain-particles",size:s.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),a.queue.writeBuffer(this.particleBuffer,0,s),this.createPipeline(a)}routeForIndex(t){const a=this.placed.length;if(a===0)return 0;const s=2,d=s+(a-1),u=Math.floor(this.count*s/d);if(t<u)return 0;const i=this.count-u,h=Math.max(1,Math.floor(i/Math.max(1,a-1)));return Math.min(a-1,1+Math.floor((t-u)/h))}createPipeline(t){if(!this.uniformBuffer||!this.particleBuffer)return;t.pushErrorScope("validation");const a=t.createShaderModule({label:"palace-brain-shader",code:Ve});a.getCompilationInfo().then(s=>{const d=s.messages.filter(u=>u.type==="error");d.length>0&&console.error("[palace-brain] WGSL:",d.map(u=>`${u.lineNum}:${u.linePos} ${u.message}`).join(`
`))}),this.pipeline=t.createRenderPipeline({label:"palace-brain-pipeline",layout:"auto",vertex:{module:a,entryPoint:"vs_main",buffers:[{arrayStride:re*4,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x4"},{shaderLocation:1,offset:16,format:"float32x4"},{shaderLocation:2,offset:32,format:"float32x4"}]}]},fragment:{module:a,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=t.createBindGroup({label:"palace-brain-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),t.popErrorScope().then(s=>{s&&console.error("[palace-brain] pipeline validation:",s.message)})}fireFlashCallback(){if(!this.burst||this.burst.callbackFired||!this.onFlashPeak)return;this.burst.callbackFired=!0;const t=this.burst.href,a=this.onFlashPeak;this.onFlashPeak=null,this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,queueMicrotask(()=>a(t))}compute(){const t=this.engine.gpuDevice;if(!t||!this.uniformBuffer||this.placed.length===0)return;const a=this.engine.wallNowMs,s=this.burst||this.hoveredIndex>=0?1:0;this.hoverStrength+=(s-this.hoverStrength)*(this.reducedMotion?1:.16),this.hoverStrength<.001&&(this.hoverStrength=0);let d=0,u=-1,i=this.hoveredIndex>=0?this.placed[this.hoveredIndex]:null,h=(i==null?void 0:i.x)??0,o=(i==null?void 0:i.y)??0;if(this.burst){u=this.indexOfHref(this.burst.href),i=u>=0?this.placed[u]:i;const w=this.reducedMotion?Ue:Be;d=Math.min(1,Math.max(0,(a-this.burst.startMs)/w));const m=this.reducedMotion?.3:Fe;d>=m&&this.fireFlashCallback();const f=(i==null?void 0:i.x)??0,x=(i==null?void 0:i.y)??0,y=this.reducedMotion?1:qe(d/.42);h=f*(1-y),o=x*(1-y)}const c=this.engine.params[11]>.5?1:Math.min(1,(a-this.bornMs)/1650),n=this.reducedMotion||d<=0?0:Math.exp(-Math.pow((d-.58)/.065,2)),g=this.engine.params[10]||0;this.uniformData[0]=this.engine.params[6]||1,this.uniformData[1]=this.engine.params[7]||1,this.uniformData[2]=g,this.uniformData[3]=this.reducedMotion?1:0,this.uniformData[4]=this.burst?u:this.hoveredIndex,this.uniformData[5]=this.hoverStrength,this.uniformData[6]=u,this.uniformData[7]=d,this.uniformData[8]=h,this.uniformData[9]=o,this.uniformData[10]=0,this.uniformData[11]=n,this.uniformData[12]=c,this.uniformData[13]=this.placed.length,this.uniformData[14]=this.engine.params[5]||0,this.uniformData[15]=0,t.queue.writeBuffer(this.uniformBuffer,0,this.uniformData)}render(t){!this.pipeline||!this.bindGroup||this.count===0||(t.setPipeline(this.pipeline),t.setBindGroup(0,this.bindGroup),t.setVertexBuffer(0,this.particleBuffer),t.draw(6,this.count))}pickAt(t,a){if(this.burst||this.placed.length===0)return null;const s=this.engine.params[6]||1,d=this.engine.params[7]||1,u=s/d;let i=-1,h=1/0;for(let o=0;o<this.placed.length;o++){const l=this.placed[o],c=(t-l.x)*u,n=a-l.y;let g=Math.hypot(c,n)/(l.scale*1.35);o===this.hoveredIndex&&(g*=.78),g<1.1&&g<h&&(i=o,h=g)}return i<0?null:{index:i,href:this.placed[i].href}}getScreenPositions(){return this.placed.map(t=>({href:t.href,ndcX:t.x,ndcY:t.y,depth:Math.min(1,Math.max(0,.72+t.z)),visible:!0}))}dispose(){var t,a;this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,(t=this.uniformBuffer)==null||t.destroy(),(a=this.particleBuffer)==null||a.destroy(),this.uniformBuffer=null,this.particleBuffer=null,this.pipeline=null,this.bindGroup=null,this.placed=[],this.count=0,this.burst=null,this.onFlashPeak=null}}const ie=[{href:"/observatory",label:"OBSERVATORY",family:"system",center:!0},{href:"/graph",label:"GRAPH",family:"memory"},{href:"/memories",label:"MEMORIES",family:"memory"},{href:"/timeline",label:"TIMELINE",family:"temporal"},{href:"/feed",label:"FEED",family:"signal"},{href:"/explore",label:"EXPLORE",family:"reasoning"},{href:"/reasoning",label:"REASONING",family:"reasoning"},{href:"/blackbox",label:"BLACK BOX",family:"signal"},{href:"/contradictions",label:"CONTRADICTIONS",family:"immune"}];function ce(e){return ie.find(t=>t.href===e)}var ht=we('<div class="palace-host fixed inset-0 bg-[#020307] svelte-1dx67o8" role="application" aria-label="VestigeOS Memory Palace. Nine living cognitive organs. Use the Command palette for keyboard navigation."><!></div>');function Mt(e,t){xe(t,!0);const a=()=>Ne(Re,"$page",s),[s,d]=Se(),u=[...K("#F5FFF2"),1],i=[...K("#9DFFEB"),1],h=[...K("#7DAFA9"),.82],o={reasoning:Y.bridge,memory:Y.recall,immune:de.veto,signal:ee.supersession,temporal:ee.txShadow,system:Y.luciferin};let l=he(null),c=null,n=null,g=null,w=null,m=he(null),f=null,x={x:0,y:0},y=!1,b=!1,v=Me(()=>{const _=a().url.searchParams.get("frame");if(_===null)return null;const p=Number(_);return Number.isFinite(p)?Math.floor(p):null});_e(()=>{c&&n?c.removePass(n):n==null||n.dispose(),c&&g?c.removePass(g):g==null||g.dispose(),n=null,g=null,c=null});async function k(_){try{c=_,b=window.matchMedia("(prefers-reduced-motion: reduce)").matches;const p=new lt(_);n=p,p.setReducedMotion(b),_.addPass(p),p.uploadRegions(ie);const M=new Ie(_);g=M,await M.init(),_.addPass(M),_.demoClock.reset(),M.setText(E())}catch(p){console.error("[palace] Failed to initialize swarm:",p)}}function T(_){return _.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}function E(){const _=[{id:"palace:title",kind:"palace-hud",text:"VESTIGE // MEMORY PALACE",x:-.92,y:.88,size:.052,color:u,depth:1,weight:1,revealSpan:24},{id:"palace:sub",kind:"palace-hud",text:T(`${ie.length} LIVING ORGANS - HOVER TO REVEAL - CLICK TO ENTER`),x:-.92,y:.8,size:.025,color:i,depth:1,weight:.86,revealSpan:28,maxWidthEm:66},{id:"palace:hint",kind:"palace-hud",text:y?"PORTAL LOCKED // COLLAPSING COGNITIVE FIELD":"MOVE THROUGH THE FIELD",x:-.92,y:-.87,size:.02,color:h,depth:.9,weight:.72,revealSpan:18}];if(D(m)){const p=ce(D(m)),M=Ce(D(m));if(p){const S=[...K(o[p.family]),1];_.push({id:"palace:focus-label",kind:"palace-focus",text:y?`ENTERING ${p.label}`:p.label,x:.34,y:.88,size:.046,color:S,depth:1,weight:1,revealSpan:18},{id:"palace:focus-purpose",kind:"palace-focus",text:T((M==null?void 0:M.purpose)??"ENTER THIS COGNITIVE ORGAN"),x:.34,y:.8,size:.021,color:u,depth:1,weight:.8,revealSpan:30,maxWidthEm:42})}}return _}function R(){g==null||g.setText(E())}function A(_){if(!D(l))return null;const p=D(l).getBoundingClientRect();return p.width<=0||p.height<=0?null:{x:(_.clientX-p.left)/p.width*2-1,y:-((_.clientY-p.top)/p.height*2-1)}}function O(_){if(!D(l)||!c)return;const p=D(l).getBoundingClientRect(),M=Math.max(1e-4,p.width/Math.max(1,p.height)),S={x:_.x*Math.max(M,1),y:_.y/Math.min(M,1)},L=w??S,G={x:L.x+(S.x-L.x)*.35,y:L.y+(S.y-L.y)*.35};w=G,c.setCursorPreNdc(G.x,G.y,G.x-L.x,G.y-L.y)}function F(_){const p=A(_);if(!p||(O(p),!n||n.isBursting))return;const M=n.pickAt(p.x,p.y),S=(M==null?void 0:M.href)??null;S!==D(m)&&(Z(m,S,!0),n.setHovered((M==null?void 0:M.index)??-1),R(),D(l)&&(D(l).style.cursor=S?"pointer":"crosshair"))}function P(){n!=null&&n.isBursting||(f=null,w=null,Z(m,null),n==null||n.setHovered(-1),c==null||c.setCursorPreNdc(999,999,0,0),R(),D(l)&&(D(l).style.cursor="crosshair"))}function N(_){var M;if(_.button!==0||!n||n.isBursting)return;const p=A(_);p&&(f={x:_.clientX,y:_.clientY,href:((M=n.pickAt(p.x,p.y))==null?void 0:M.href)??null})}function z(){f=null}function U(_){const p=f;if(f=null,!p||!n||n.isBursting||Math.hypot(_.clientX-p.x,_.clientY-p.y)>9)return;const M=A(_);if(!M)return;const S=n.pickAt(M.x,M.y);if(!S||S.href!==p.href)return;x={x:_.clientX,y:_.clientY},Z(m,S.href,!0),n.setHovered(S.index),y=!0,R(),D(l)&&(D(l).style.cursor="wait"),n.startBurst(S.href,j)||j(S.href)}async function j(_){const p=ce(_);await De(`${Oe}${_}`,{clientX:x.x,clientY:x.y,color:p?o[p.family]:Y.luciferin,reduced:b})}var I=ht();Ae("1dx67o8",_=>{ye(()=>{ke.title="Memory Palace · VestigeOS"})});var H=Ee(I);ze(H,{demo:"recall-path",seed:"vestige-palace-swarm-v2",get freezeFrame(){return D(v)},onready:k}),Te(I),Pe(I,_=>Z(l,_),()=>D(l)),oe("pointerdown",I,N),oe("pointerup",I,U),le("pointercancel",I,z),oe("pointermove",I,F),le("pointerleave",I,P),be(e,I),ve(),d()}me(["pointerdown","pointerup","pointermove"]);export{Mt as component};

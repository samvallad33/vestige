var be=Object.defineProperty;var ve=(e,t,s)=>t in e?be(e,t,{enumerable:!0,configurable:!0,writable:!0,value:s}):e[t]=s;var F=(e,t,s)=>ve(e,typeof t!="symbol"?t+"":t,s);import"../chunks/Bzak7iHL.js";import{d as ye,a as we,b as ie,e as pe}from"../chunks/CNdOtqLU.js";import{p as Me,a as Ee,b as ke,i as Te,g as z,f as Ae,u as Se,c as Pe,s as J,d as fe,$ as Oe,r as Re}from"../chunks/Dw_4PDAU.js";import{h as Ne}from"../chunks/C7sd-K7m.js";import{b as ze}from"../chunks/C7WW_yYn.js";import{s as Ce,a as Ie}from"../chunks/CSsWZzkN.js";import{p as Fe}from"../chunks/DcA-AYjE.js";import{b as Le}from"../chunks/f2U-WYi5.js";import{O as De}from"../chunks/Ce_AOQ7r.js";import{r as j,R as q,B as se,I as me,T as Be}from"../chunks/Byu5DFqz.js";import{f as Ue}from"../chunks/C6VaMKHG.js";import{b as He}from"../chunks/btpGX-aM.js";const le=12,Ge=16,je=1e3,Ve=200,Ye=.9,Xe={reasoning:q.bridge,memory:q.recall,immune:me.veto,signal:se.supersession,temporal:se.txShadow,system:q.luciferin},We={"/observatory":0,"/graph":1,"/memories":2,"/timeline":3,"/blackbox":4,"/reasoning":5,"/explore":6,"/feed":7,"/contradictions":8},qe={"/observatory":[0,0,.12,.26],"/graph":[-.78,.5,.04,.15],"/memories":[.8,.46,-.04,.15],"/timeline":[-.9,-.06,.1,.14],"/reasoning":[.92,-.02,-.08,.15],"/blackbox":[-.72,-.56,.18,.14],"/feed":[.74,-.54,.02,.14],"/explore":[-.02,.66,-.13,.14],"/contradictions":[0,-.66,-.02,.14]},Q=[0,0],$e=`
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
`;function Ke(e){return e-Math.floor(e)}function Ze(e){const t=e<0?0:e>1?1:e;return t*t*(3-2*t)}function i(e){return Ke(Math.sin(e*12.9898+78.233)*43758.5453)}const D=Math.PI*2;function Je(e){const t=i(e*.7351+1),s=i(e*1.2971+5),r=i(e*2.1637+9),u=i(e*3.5391+13);if(t<.15){const b=2*s-1,E=Math.sqrt(Math.max(0,1-b*b)),A=r*D,w=.2+.16*Math.cbrt(u);return[E*Math.cos(A)*w,b*w,E*Math.sin(A)*w]}const _=s<.5?-1:1,h=2*r-1,c=Math.acos(Math.max(-1,Math.min(1,h))),a=Math.sin(c),l=.2,o=_===1?l+u*(Math.PI-2*l):Math.PI+l+u*(Math.PI-2*l);let n=a*Math.cos(o),g=h,M=a*Math.sin(o);const m=.095*Math.sin(7*o)*(a*a)+.075*Math.cos(9*o+2*c)*a+.065*Math.sin(6*c)+.05*Math.sin(11*o)*Math.sin(4*c),d=Math.min(Math.abs(o-Math.PI/2),Math.abs(o-3*Math.PI/2)),x=-.16*Math.exp(-(d*d)/(.1*.1))*a,v=.02*(i(e*5.19+2)-.5),y=.92+m+x+v;return n*=y,g*=y,M*=y,n+=_*.05*Math.exp(-(d*d)/(.16*.16))*a,g*=.9,M*=1.06,n*=1.04,[n,g,M]}function Qe(e){const u=i(e*1.17+3),_=i(e*1.91+7),h=i(e*2.37+11),c=i(e*3.11+5),a=Math.floor(u*12),l=a%3,o=[.55,.05,-.45][l],n=Math.ceil(12/3),M=Math.floor(a/3)/n*D+l*.9,m=Math.sqrt(Math.max(0,1-o*o)),d=m*Math.cos(M),x=m*Math.sin(M),v=o,y=_*D,b=Math.sqrt(2*(1+v)),E=(1+v)*Math.cos(y)/b,A=(d*Math.sin(y)-x*Math.cos(y))/b,w=(d*Math.cos(y)+x*Math.sin(y))/b;let k=1-(1+v)*Math.sin(y)/b;if(Math.abs(k)<.12){const U=y+Math.PI,L=(1+v)*Math.cos(U)/b,V=(d*Math.sin(U)-x*Math.cos(U))/b,p=(d*Math.cos(U)+x*Math.sin(U))/b,f=(1+v)*Math.sin(U)/b;let T=L/(1-f),N=V/(1-f),H=p/(1-f);const G=Math.hypot(T,N,H)||1,W=1.6*Math.tanh(G/1.6);T*=W/G,N*=W/G,H*=W/G;const $=h*D,Z=.05*Math.sqrt(c);return[(H+(c-.5)*.05)*.66,(N+Math.sin($)*Z)*.66,(T+Math.cos($)*Z)*.66]}let R=E/k,C=A/k,S=w/k;const O=Math.hypot(R,C,S),I=1.55*Math.tanh(O/1.55),B=O>1e-6?I/O:0;R*=B,C*=B,S*=B;const X=h*D,Y=.05*Math.sqrt(c);return R+=Math.cos(X)*Y,C+=Math.sin(X)*Y,S+=(c-.5)*.05,[S*.66,C*.66,R*.66]}const K=[{radius:1.18,cells:210},{radius:.86,cells:132},{radius:.54,cells:72}],he=414,et=Math.PI*(3-Math.sqrt(5));function tt(e){let t=Math.floor(i(e*1.6180339887+.5)*he);t>=he&&(t=he-1);let s=0,r=t;r>=K[0].cells&&(r-=K[0].cells,s=1),s===1&&r>=K[1].cells&&(r-=K[1].cells,s=2);const u=K[s],_=u.cells,h=1-(2*r+1)/_,c=Math.sqrt(Math.max(0,1-h*h)),a=et*r,l=Math.cos(a)*c,o=h,n=Math.sin(a)*c,g=.05+.012*(2-s),M=i(e*2.31+5),m=i(e*3.97+9),d=g*Math.cbrt(i(e*4.13+1)),x=2*M-1,v=Math.sqrt(Math.max(0,1-x*x)),y=m*D,b=d*v*Math.cos(y),E=d*x,A=d*v*Math.sin(y),w=u.radius;return[(l*w+b)*1.06,(o*w+E)*1.06,(n*w+A)*1.06]}function st(e){const t=Math.SQRT1_2,s=i(e*1.7+3.1),r=6,_=Math.floor(i(e*.31+9.2)*r)/r*D,h=s*D*3+_,c=s*D*2+_*1.618,a=t*Math.cos(h),l=t*Math.sin(h),o=t*Math.cos(c),g=1.28-t*Math.sin(c);let M=a/g,m=l/g,d=o/g;const x=.045,v=i(e*2.3+1)*D,y=Math.sqrt(i(e*3.7+2))*x;M+=Math.cos(v)*y,m+=Math.sin(v)*y,d+=(i(e*4.1+5)-.5)*x*2;const b=1.02,E=.5,A=Math.cos(E),w=Math.sin(E),P=m*A-d*w,k=m*w+d*A;return[M*b,P*b,k*b]}const at=(1+Math.sqrt(5))/2,ot=(()=>{const e=at,t=[[0,1,e],[0,1,-e],[0,-1,e],[0,-1,-e],[1,e,0],[1,-e,0],[-1,e,0],[-1,-e,0],[e,0,1],[e,0,-1],[-e,0,1],[-e,0,-1]],s=Math.hypot(0,1,e);return t.map(r=>[r[0]/s,r[1]/s,r[2]/s])})();function rt(e,t,s){let r=-2;for(const u of ot){const _=Math.abs(e*u[0]+t*u[1]+s*u[2]);_>r&&(r=_)}return .86+.14*r}function nt(e){const r=i(e+2);if(r<.46){if(i(e+4)<.82){const R=Math.floor(i(e+6)*3)%3,C=i(e+8)<.5?-1:1,S=i(e+10)<.5?-1:1,O=-1.16+2*1.16*i(e+12),I=(i(e+14)-.5)*.03,B=[0,0,0];return B[R]=O+I,B[(R+1)%3]=C*1.16+I,B[(R+2)%3]=S*1.16+I,B}const v=Math.floor(i(e+16)*6),y=Math.floor(v/2),b=v%2?1:-1,E=i(e+18),A=i(e+20)<.5?1:-1,w=-1.16+2*1.16*E,P=A*(-1.16+2*1.16*E),k=[0,0,0];return k[y]=b*1.16,k[(y+1)%3]=w,k[(y+2)%3]=P,k}if(r<.86){const y=Math.floor(i(e+22)*1600),b=1-2*y/1599,E=Math.sqrt(Math.max(0,1-b*b)),A=2.399963229728653*y,w=Math.cos(A)*E,P=b,k=Math.sin(A)*E,R=.56*rt(w,P,k);return[w*R,P*R,k*R]}const u=Math.floor(i(e+24)*8),_=u&1?1:-1,h=u&2?1:-1,c=u&4?1:-1,a=.56*.62,l=i(e+26),o=(i(e+28)-.5)*.025,n=_*a,g=h*a,M=c*a,m=_*1.16,d=h*1.16,x=c*1.16;return[n+(m-n)*l+o,g+(d-g)*l+o,M+(x-M)*l+o]}function it(e){let c=(i(e+1)-.5)*1.2,a=(i(e+2)-.5)*1.2,l=(i(e+3)-.5)*.6+.4;const o=.01,g=520+Math.floor(i(e+7)*260);for(let m=0;m<g;m++){const d=(l-.7)*c-3.5*a,x=3.5*c+(l-.7)*a,v=.6+.95*l-l*l*l/3-(c*c+a*a)*(1+.25*l)+.1*l*c*c*c;c+=d*o,a+=x*o,l+=v*o}const M=.92;return[c*M,(l-.6)*M*1.05,a*M]}function lt(e){const h=2.399963229728653,c=.12,a=.02,l=1.3;if(i(e*.53+.9)<c){const S=.11*Math.cbrt(i(e+21)),O=i(e+22)*D,I=Math.acos(2*i(e+23)-1);return[S*Math.sin(I)*Math.cos(O),S*Math.cos(I),S*Math.sin(I)*Math.sin(O)]}const o=Math.floor(i(e*1.373+5.3)*14),n=1-(o+.5)/14*2,g=Math.sqrt(Math.max(0,1-n*n)),M=o*h;let m=Math.cos(M)*g,d=n,x=Math.sin(M)*g;const v=Math.pow(i(e*.911+1.7),.9);let y=0;const b=[];for(let S=0;S<3;S++){const O=.5*Math.pow(.8,S);b.push(O),y+=O}const E=v*y;let A=0,w=0,P=0,k=0;for(let S=0;S<3;S++){if(S>0){const B=((i(e*(1+S*.41)+S*7.13+3.1)<.5?0:1)-.5)*Math.PI+h*(S+o),X=.42*Math.pow(.9,S-1)*(.7+.6*i(e+S*11+17));let Y=0,U=1,L=0;Math.abs(d)>.9&&(Y=1,U=0,L=0);let V=d*L-x*U,p=x*Y-m*L,f=m*U-d*Y;const T=Math.hypot(V,p,f)||1;V/=T,p/=T,f/=T;const N=p*x-f*d,H=f*m-V*x,G=V*d-p*m,W=Math.cos(B),$=Math.sin(B),Z=W*V+$*N,_e=W*p+$*H,xe=W*f+$*G,oe=Math.cos(X),re=Math.sin(X);m=oe*m+re*Z,d=oe*d+re*_e,x=oe*x+re*xe;const ne=Math.hypot(m,d,x)||1;m/=ne,d/=ne,x/=ne}const O=b[S];if(k+O>=E){const I=(E-k)/O;A+=m*O*I,w+=d*O*I,P+=x*O*I;break}A+=m*O,w+=d*O,P+=x*O,k+=O}const R=.35+.65*Math.sin(Math.min(1,v)*Math.PI);A+=(i(e*3.1+1)-.5)*a*R,w+=(i(e*3.3+2)-.5)*a*R,P+=(i(e*3.7+3)-.5)*a*R;const C=l/y;return[A*C,w*C,P*C]}function ht(e){const t=i(e*1.17+3),s=i(e*1.91+7),r=.82,u=.34,_=5,h=6,c=t*D,a=Math.floor(s*_),l=c*h+a/_*D+(i(e+5)-.5)*.5,o=.72+.28*Math.sqrt(i(e+13)),n=u*o;let g=(r+n*Math.cos(l))*Math.cos(c),M=(r+n*Math.cos(l))*Math.sin(c),m=n*Math.sin(l);const d=1.0821,x=Math.cos(d),v=Math.sin(d),y=m*x-M*v,b=m*v+M*x;m=y,M=b;const E=.2094,A=Math.cos(E),w=Math.sin(E),P=g*A-m*w,k=g*w+m*A;return[P,k,M]}function ct(e){const t=i(e*1.17+3),s=i(e*1.91+7),r=i(e*2.37+11),u=.92,_=[[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],h=[[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]],c=t<.5,a=c?h:_,l=c?-.16:.16;if(s<.72){const y=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]][Math.floor(r*6)%6],b=a[y[0]],E=a[y[1]],A=i(e*3.11+5),w=.5+.5*Math.sign(A-.5)*Math.pow(Math.abs(2*A-1),.55),P=(i(e*4.7+1)-.5)*.05,k=(i(e*5.3+2)-.5)*.05,R=(i(e*6.1+3)-.5)*.05;return[(b[0]+(E[0]-b[0])*w)*u+P+l,(b[1]+(E[1]-b[1])*w)*u+k,(b[2]+(E[2]-b[2])*w)*u+R]}if(s<.92){const v=a[Math.floor(r*4)%4],y=Math.pow(i(e*7.7+4),1.5)*.11,b=i(e*8.3+6)*D,E=Math.acos(2*i(e*9.1+8)-1);return[v[0]*u+y*Math.sin(E)*Math.cos(b)+l,v[1]*u+y*Math.sin(E)*Math.sin(b),v[2]*u+y*Math.cos(E)]}const o=i(e*10.3+7)*D,n=Math.acos(2*i(e*11.7+9)-1),g=Math.sin(n)*Math.cos(o),M=Math.sin(n)*Math.sin(o),m=Math.cos(n),d=Math.abs(g)+Math.abs(M)+Math.abs(m),x=.34;return[g/d*x,M/d*x,m/d*x]}function dt(e,t){switch(e){case 0:return Je(t);case 1:return Qe(t);case 2:return tt(t);case 3:return st(t);case 4:return nt(t);case 5:return it(t);case 6:return lt(t);case 7:return ht(t);default:return ct(t)}}function pt(e){const t=navigator.hardwareConcurrency||8,s=window.innerWidth<760,r=t<=4||(window.devicePixelRatio||1)>2.2;return e?s?1e4:16e3:r?s?12e3:2e4:s?18e3:t>=10?55e3:4e4}class ft{constructor(t){F(this,"engine");F(this,"pipeline",null);F(this,"bindGroup",null);F(this,"uniformBuffer",null);F(this,"particleBuffer",null);F(this,"uniformData",new Float32Array(Ge));F(this,"placed",[]);F(this,"count",0);F(this,"hoveredIndex",-1);F(this,"hoverStrength",0);F(this,"reducedMotion",!1);F(this,"bornMs");F(this,"burst",null);F(this,"onFlashPeak",null);F(this,"watchdog",null);this.engine=t,this.bornMs=t.wallNowMs}setReducedMotion(t){this.reducedMotion=t}setHovered(t){this.burst||(this.hoveredIndex=t>=0&&t<this.placed.length?t:-1)}indexOfHref(t){return t?this.placed.findIndex(s=>s.href===t):-1}get isBursting(){return this.burst!==null}startBurst(t,s){if(this.burst)return!1;const r=this.placed.find(u=>u.href===t);return r?(this.hoveredIndex=this.placed.indexOf(r),this.hoverStrength=1,this.burst={href:t,startMs:this.engine.wallNowMs,callbackFired:!1},this.onFlashPeak=s,this.watchdog=setTimeout(()=>this.fireFlashCallback(),980),!0):!1}uploadRegions(t){var u,_;const s=this.engine.gpuDevice;if(!s||t.length===0)return;this.placed=t.map((h,c)=>{const a=qe[h.href]??[0,0,c*.01,.14];return{href:h.href,x:a[0],y:a[1],z:a[2],scale:a[3],kind:We[h.href]??c%9}}),this.count=pt(this.reducedMotion);const r=new Float32Array(this.count*le);for(let h=0;h<this.count;h++){const c=h+1,a=h*le,l=this.routeForIndex(h),o=this.placed[l],n=dt(o.kind,c),g=j(Xe[t[l].family]);r[a+0]=n[0],r[a+1]=n[1],r[a+2]=n[2],r[a+3]=i(c*2.83+12),r[a+4]=o.x,r[a+5]=o.y,r[a+6]=o.z,r[a+7]=o.scale,r[a+8]=i(c*1.7+5),r[a+9]=g[0]*.5+g[1]*.3,r[a+10]=g[2]*.6,r[a+11]=l}(u=this.uniformBuffer)==null||u.destroy(),(_=this.particleBuffer)==null||_.destroy(),this.uniformBuffer=s.createBuffer({label:"palace-brain-uniforms",size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.particleBuffer=s.createBuffer({label:"palace-brain-particles",size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.particleBuffer,0,r),this.createPipeline(s)}routeForIndex(t){const s=this.placed.length;if(s===0)return 0;const r=2,u=r+(s-1),_=Math.floor(this.count*r/u);if(t<_)return 0;const h=this.count-_,c=Math.max(1,Math.floor(h/Math.max(1,s-1)));return Math.min(s-1,1+Math.floor((t-_)/c))}createPipeline(t){if(!this.uniformBuffer||!this.particleBuffer)return;t.pushErrorScope("validation");const s=t.createShaderModule({label:"palace-brain-shader",code:$e});s.getCompilationInfo().then(r=>{const u=r.messages.filter(_=>_.type==="error");u.length>0&&console.error("[palace-brain] WGSL:",u.map(_=>`${_.lineNum}:${_.linePos} ${_.message}`).join(`
`))}),this.pipeline=t.createRenderPipeline({label:"palace-brain-pipeline",layout:"auto",vertex:{module:s,entryPoint:"vs_main",buffers:[{arrayStride:le*4,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x4"},{shaderLocation:1,offset:16,format:"float32x4"},{shaderLocation:2,offset:32,format:"float32x4"}]}]},fragment:{module:s,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=t.createBindGroup({label:"palace-brain-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),t.popErrorScope().then(r=>{r&&console.error("[palace-brain] pipeline validation:",r.message)})}fireFlashCallback(){if(!this.burst||this.burst.callbackFired||!this.onFlashPeak)return;this.burst.callbackFired=!0;const t=this.burst.href,s=this.onFlashPeak;this.onFlashPeak=null,this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,queueMicrotask(()=>s(t))}compute(){const t=this.engine.gpuDevice;if(!t||!this.uniformBuffer||this.placed.length===0)return;const s=this.engine.wallNowMs,r=this.burst||this.hoveredIndex>=0?1:0;this.hoverStrength+=(r-this.hoverStrength)*(this.reducedMotion?1:.16),this.hoverStrength<.001&&(this.hoverStrength=0);let u=0,_=-1,h=this.hoveredIndex>=0?this.placed[this.hoveredIndex]:null,c=(h==null?void 0:h.x)??0,a=(h==null?void 0:h.y)??0;if(this.burst){_=this.indexOfHref(this.burst.href),h=_>=0?this.placed[_]:h;const M=this.reducedMotion?Ve:je;u=Math.min(1,Math.max(0,(s-this.burst.startMs)/M));const m=this.reducedMotion?.3:Ye;u>=m&&this.fireFlashCallback();const d=(h==null?void 0:h.x)??0,x=(h==null?void 0:h.y)??0,v=this.reducedMotion?1:Ze(u/.42);c=d*(1-v),a=x*(1-v)}const o=this.engine.params[11]>.5?1:Math.min(1,(s-this.bornMs)/1650),n=this.reducedMotion||u<=0?0:Math.exp(-Math.pow((u-.58)/.065,2)),g=this.engine.params[10]||0;this.uniformData[0]=this.engine.params[6]||1,this.uniformData[1]=this.engine.params[7]||1,this.uniformData[2]=g,this.uniformData[3]=this.reducedMotion?1:0,this.uniformData[4]=this.burst?_:this.hoveredIndex,this.uniformData[5]=this.hoverStrength,this.uniformData[6]=_,this.uniformData[7]=u,this.uniformData[8]=c,this.uniformData[9]=a,this.uniformData[10]=0,this.uniformData[11]=n,this.uniformData[12]=o,this.uniformData[13]=this.placed.length,this.uniformData[14]=this.engine.params[5]||0,this.uniformData[15]=0,t.queue.writeBuffer(this.uniformBuffer,0,this.uniformData)}render(t){!this.pipeline||!this.bindGroup||this.count===0||(t.setPipeline(this.pipeline),t.setBindGroup(0,this.bindGroup),t.setVertexBuffer(0,this.particleBuffer),t.draw(6,this.count))}pickAt(t,s){if(this.burst||this.placed.length===0)return null;const r=this.engine.params[6]||1,u=this.engine.params[7]||1,_=r/u;let h=-1,c=1/0;for(let a=0;a<this.placed.length;a++){const l=this.placed[a],o=(t-l.x)*_,n=s-l.y;let g=Math.hypot(o,n)/(l.scale*1.35);a===this.hoveredIndex&&(g*=.78),g<1.1&&g<c&&(h=a,c=g)}return h<0?null:{index:h,href:this.placed[h].href}}getScreenPositions(){return this.placed.map(t=>({href:t.href,ndcX:t.x,ndcY:t.y,depth:Math.min(1,Math.max(0,.72+t.z)),visible:!0}))}dispose(){var t,s;this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,(t=this.uniformBuffer)==null||t.destroy(),(s=this.particleBuffer)==null||s.destroy(),this.uniformBuffer=null,this.particleBuffer=null,this.pipeline=null,this.bindGroup=null,this.placed=[],this.count=0,this.burst=null,this.onFlashPeak=null}}const ae=[{href:"/observatory",label:"OBSERVATORY",family:"system",center:!0},{href:"/graph",label:"GRAPH",family:"memory"},{href:"/memories",label:"MEMORIES",family:"memory"},{href:"/timeline",label:"TIMELINE",family:"temporal"},{href:"/feed",label:"FEED",family:"signal"},{href:"/explore",label:"EXPLORE",family:"reasoning"},{href:"/reasoning",label:"REASONING",family:"reasoning"},{href:"/blackbox",label:"BLACK BOX",family:"signal"},{href:"/contradictions",label:"CONTRADICTIONS",family:"immune"}];function de(e){return ae.find(t=>t.href===e)}const ut={reasoning:[...j("#FFFFFF"),1],memory:[...j("#FFFFFF"),1],immune:[...j("#FFFFFF"),1],temporal:[...j("#FFFFFF"),1],signal:[...j("#FFFFFF"),1],system:[...j("#FFFFFF"),1]},gt=[...j("#FFFFFF"),1],mt=1.35,_t=.028,xt=.6,ue=.85,bt=.05,vt=.03,yt=1,wt=.95,ee=[],te=[];function ge(e,t,s){return e+(t-e)*s}function ce(e){return e<0?0:e>1?1:Number.isFinite(e)?e:0}function Mt(e,t={}){const s=t.hoveredHref??null,r=t.dimUnhovered??!0,u=t.aspect??1,_=u<.85,c=.7-.06*(_?ce((.85-u)/(.85-.46)):0);let a=0;te.length=0;for(let l=0;l<e.length;l++){const o=e[l];if(o.visible===!1)continue;const n=de(o.href);if(!n)continue;const g=ce(o.depth),M=s!==null&&o.href===s,m=_t+g*mt;let d=o.ndcX+m*ue,x=o.ndcY+m*xt;_&&x>c&&(x=c-(x-c));let v=ge(vt,bt,g),y=ge(wt,yt,g);n.center&&(v*=1.22),M?(v*=1.28,y=1):r&&s!==null&&(y*=.32);const b=n.label.length*v*.62;d+b>.97&&(d=o.ndcX-m*ue-b);for(let P=0;P<te.length;P++){const k=te[P],R=d<k.x1&&d+b>k.x0,C=Math.abs(x-k.y)<(v+k.size)*.75;R&&C&&(x=k.y-(v+k.size)*.85)}te.push({x0:d,x1:d+b,y:x,size:v});const E=n.center?gt:ut[n.family],A=[E[0],E[1],E[2],ce(y)];let w=ee[a];w||(w={id:"",kind:"palace-label",text:"",x:0,y:0,size:0,color:[0,0,0,0],depth:0,weight:.75,revealSpan:1,maxWidthEm:24},ee[a]=w),w.id="palace-label:"+o.href,w.kind="palace-label",w.text=n.label,w.x=d,w.y=x,w.size=v,w.color=A,w.depth=M?1:.8+g*.2,w.weight=.95,a++}return ee.length=a,ee}ae.length;var Et=Ae('<div class="palace-host fixed inset-0 bg-[#020307] svelte-1dx67o8" role="application" aria-label="VestigeOS Memory Palace. Nine living cognitive organs. Use the Command palette for keyboard navigation."><!></div>');function Dt(e,t){Me(t,!0);const s=()=>Ie(Fe,"$page",r),[r,u]=Ce(),_=[...j("#F5FFF2"),1],h=[...j("#9DFFEB"),1],c=[...j("#7DAFA9"),.82],a={reasoning:q.bridge,memory:q.recall,immune:me.veto,signal:se.supersession,temporal:se.txShadow,system:q.luciferin};let l=fe(null),o=null,n=null,g=null,M=null,m=fe(null),d=null,x={x:0,y:0},v=!1,y=!1,b=-1,E=Se(()=>{const p=s().url.searchParams.get("frame");if(p===null)return null;const f=Number(p);return Number.isFinite(f)?Math.floor(f):null});we(()=>{o&&n?o.removePass(n):n==null||n.dispose(),o&&g?o.removePass(g):g==null||g.dispose(),n=null,g=null,o=null});async function A(p){try{o=p,y=window.matchMedia("(prefers-reduced-motion: reduce)").matches;const f=new ft(p);n=f,f.setReducedMotion(y),p.addPass(f),f.uploadRegions(ae);const T=new Be(p);g=T,await T.init(),p.addPass(T),p.demoClock.reset(),T.setText(P())}catch(f){console.error("[palace] Failed to initialize swarm:",f)}}function w(p){return p.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}function P(){const p=[{id:"palace:title",kind:"palace-hud",text:"VESTIGE // MEMORY PALACE",x:-.92,y:.88,size:.052,color:_,depth:1,weight:1,revealSpan:24},{id:"palace:sub",kind:"palace-hud",text:w(`${ae.length} LIVING ORGANS - HOVER TO REVEAL - CLICK TO ENTER`),x:-.92,y:.8,size:.025,color:h,depth:1,weight:.86,revealSpan:28,maxWidthEm:66},{id:"palace:hint",kind:"palace-hud",text:v?"PORTAL LOCKED // COLLAPSING COGNITIVE FIELD":"MOVE THROUGH THE FIELD",x:-.92,y:-.87,size:.02,color:c,depth:.9,weight:.72,revealSpan:18}];if(z(m)){const f=de(z(m)),T=Ue(z(m));if(f){const N=[...j(a[f.family]),1];p.push({id:"palace:focus-label",kind:"palace-focus",text:v?`ENTERING ${f.label}`:f.label,x:.34,y:.88,size:.046,color:N,depth:1,weight:1,revealSpan:18},{id:"palace:focus-purpose",kind:"palace-focus",text:w((T==null?void 0:T.purpose)??"ENTER THIS COGNITIVE ORGAN"),x:.34,y:.8,size:.021,color:_,depth:1,weight:.8,revealSpan:30,maxWidthEm:42})}}return p}function k(){const p=P(),f=n?Mt(n.getScreenPositions(),{hoveredHref:z(m),dimUnhovered:!!z(m),aspect:((o==null?void 0:o.params[6])||0)/Math.max(1,(o==null?void 0:o.params[7])||1)}):[];g==null||g.setText([...p,...f])}function R(p){p!==b&&p%2===0&&(b=p,k())}function C(p){if(!z(l))return null;const f=z(l).getBoundingClientRect();return f.width<=0||f.height<=0?null:{x:(p.clientX-f.left)/f.width*2-1,y:-((p.clientY-f.top)/f.height*2-1)}}function S(p){if(!z(l)||!o)return;const f=z(l).getBoundingClientRect(),T=Math.max(1e-4,f.width/Math.max(1,f.height)),N={x:p.x*Math.max(T,1),y:p.y/Math.min(T,1)},H=M??N,G={x:H.x+(N.x-H.x)*.35,y:H.y+(N.y-H.y)*.35};M=G,o.setCursorPreNdc(G.x,G.y,G.x-H.x,G.y-H.y)}function O(p){const f=C(p);if(!f||(S(f),!n||n.isBursting))return;const T=n.pickAt(f.x,f.y),N=(T==null?void 0:T.href)??null;N!==z(m)&&(J(m,N,!0),n.setHovered((T==null?void 0:T.index)??-1),k(),z(l)&&(z(l).style.cursor=N?"pointer":"crosshair"))}function I(){n!=null&&n.isBursting||(d=null,M=null,J(m,null),n==null||n.setHovered(-1),o==null||o.setCursorPreNdc(999,999,0,0),k(),z(l)&&(z(l).style.cursor="crosshair"))}function B(p){var T;if(p.button!==0||!n||n.isBursting)return;const f=C(p);f&&(d={x:p.clientX,y:p.clientY,href:((T=n.pickAt(f.x,f.y))==null?void 0:T.href)??null})}function X(){d=null}function Y(p){const f=d;if(d=null,!f||!n||n.isBursting||Math.hypot(p.clientX-f.x,p.clientY-f.y)>9)return;const T=C(p);if(!T)return;const N=n.pickAt(T.x,T.y);if(!N||N.href!==f.href)return;x={x:p.clientX,y:p.clientY},J(m,N.href,!0),n.setHovered(N.index),v=!0,k(),z(l)&&(z(l).style.cursor="wait"),n.startBurst(N.href,U)||U(N.href)}async function U(p){const f=de(p);await He(`${Le}${p}`,{clientX:x.x,clientY:x.y,color:f?a[f.family]:q.luciferin,reduced:y})}var L=Et();Ne("1dx67o8",p=>{Te(()=>{Oe.title="Memory Palace · VestigeOS"})});var V=Pe(L);De(V,{demo:"recall-path",seed:"vestige-palace-swarm-v2",get freezeFrame(){return z(E)},onframe:R,onready:A}),Re(L),ze(L,p=>J(l,p),()=>z(l)),ie("pointerdown",L,B),ie("pointerup",L,Y),pe("pointercancel",L,X),ie("pointermove",L,O),pe("pointerleave",L,I),Ee(e,L),ke(),u()}ye(["pointerdown","pointerup","pointermove"]);export{Dt as component};

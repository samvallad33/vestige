import{A as e,D as t,H as n,M as r,N as i,Q as a,R as o,T as s,W as c,X as l,Y as u,dt as d,et as f,g as p,j as m,n as h,ot as g,s as _,st as v,tt as y}from"../chunks/CvawvY3d.js";import{i as b,l as x,n as S}from"../chunks/eHoGjb3p.js";import"../chunks/xihTtKlq.js";import{t as C}from"../chunks/CJst9gMq.js";import"../chunks/C_EsKhQW.js";import{a as ee}from"../chunks/BVqTbojw.js";import{a as te,c as ne,n as re,o as ie}from"../chunks/DKuNZ7Jg.js";import{t as ae}from"../chunks/D4_bBKYm.js";import{t as oe}from"../chunks/CoEqe9GD.js";import{i as se,n as ce,o as w,t as le,u as T}from"../chunks/Bbp_g6ip.js";var ue=12,de=16,E=1400,D=200,fe=.9,pe=.42,O=800,k=.42,A={"/trace":0,"/graph":1,"/field":2,"/ledger":3,"/guard":4,"/causal":5,"/cinema":6,"/system":7,"/healing":8},j=.34,M={"/trace":[0,-.08,.12,.24],"/graph":[-.7,.36,.04,.14],"/field":[.7,.36,-.04,.14],"/ledger":[-.76,-.16,.1,.13],"/causal":[.78,-.16,-.08,.14],"/guard":[-.58,-.62,.18,.13],"/system":[.58,-.62,.02,.13],"/healing":[-.2,-.7,-.02,.12],"/cinema":[.2,-.7,-.13,.12]},N=[0,-.08],P=new Map;function F(e){let t=P.get(e);if(t)return t;let n=[],r=[];for(let t=1;t<=1600;t++){let[i,a,o]=Ee(e,t*7+3);n.push(-a*Math.cos(j)+Math.hypot(i,o)*Math.sin(j)),r.push(Math.hypot(i,a,o))}let i=e=>e.sort((e,t)=>e-t)[Math.floor(e.length*.995)],a={below:Math.max(.6,i(n))*1.035,reach:Math.max(.6,i(r))*1.035};return P.set(e,a),a}var I=`
struct Uniforms {
	viewport: vec4<f32>,     // x=w y=h z=simTime w=reducedMotion
	interaction: vec4<f32>,  // x=hoverIndex y=hoverStrength z=selectedIndex w=burstProgress
	portal: vec4<f32>,       // x=portalX y=portalY z=dolly zoom (0: the burst instead) w=flash
	intro: vec4<f32>,        // x=introProgress y=count z=pulse w=dolly glide to centre (0..1)
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

// Each organ in its page's own colours (shape kind order: trace, witness,
// field, ledger, guard, causal, cinema, system, healing).
const ORGAN_DEEP = array<vec3<f32>, 9>(
	vec3<f32>(0.06, 0.24, 0.16), vec3<f32>(0.05, 0.22, 0.15), vec3<f32>(0.10, 0.12, 0.20),
	vec3<f32>(0.30, 0.18, 0.05), vec3<f32>(0.22, 0.06, 0.20), vec3<f32>(0.30, 0.08, 0.03),
	vec3<f32>(0.14, 0.08, 0.32), vec3<f32>(0.05, 0.12, 0.32), vec3<f32>(0.04, 0.26, 0.16)
);
const ORGAN_BRIGHT = array<vec3<f32>, 9>(
	vec3<f32>(0.34, 1.00, 0.62), vec3<f32>(0.55, 0.96, 0.70), vec3<f32>(0.80, 0.86, 0.96),
	vec3<f32>(1.00, 0.76, 0.36), vec3<f32>(0.80, 0.40, 0.98), vec3<f32>(1.00, 0.52, 0.22),
	vec3<f32>(0.74, 0.64, 1.00), vec3<f32>(0.46, 0.72, 1.00), vec3<f32>(0.36, 1.00, 0.66)
);
// How fast light travels through each shape, and how many packets at once.
const ORGAN_SPEED = array<f32, 9>(0.16, 0.32, 0.05, 0.12, 0.45, 0.28, 0.08, 0.22, 0.18);
const ORGAN_PACKETS = array<f32, 9>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 2.0);
// The kinds of memory, for the Field's stars (decision, correction, fact, event, pattern, note).
const TYPE_RGB = array<vec3<f32>, 6>(
	vec3<f32>(1.00, 0.74, 0.30), vec3<f32>(1.00, 0.42, 0.33), vec3<f32>(0.35, 0.88, 1.00),
	vec3<f32>(0.30, 1.00, 0.68), vec3<f32>(0.72, 0.62, 1.00), vec3<f32>(0.78, 0.84, 0.90)
);

// A point's colour: its organ's palette, its part of the shape (accent), and
// the light travelling through the shape along its flow.
fn organ_color(kind: u32, accent: u32, seed: f32, packet: f32) -> vec3<f32> {
	let k = min(kind, 8u);
	let deep = ORGAN_DEEP[k];
	let bright = ORGAN_BRIGHT[k];
	var base = mix(deep, bright, 0.5 + 0.45 * seed);
	if (accent >= 8u) {
		base = TYPE_RGB[min(accent - 8u, 5u)] * (0.75 + 0.35 * seed);
	} else if (accent == 1u) {
		// The shape's focus: a decision's crystal, a stop, a failure, a memory bead.
		var hot = select(vec3<f32>(1.0, 0.86, 0.55), vec3<f32>(1.0, 0.3, 0.32), k == 4u || k == 5u);
		// The brain's synapses spark violet, like the logo above it.
		if (k == 0u) { hot = vec3<f32>(0.74, 0.44, 1.0); }
		base = mix(bright, hot, select(0.75, 0.85, k == 0u));
	} else if (accent == 2u) {
		base = mix(bright, vec3<f32>(1.0), 0.45);
	} else if (accent == 3u) {
		base = mix(deep, bright, 0.25);
	} else if (accent == 4u) {
		// Healing's second strand: the correction, warmer.
		base = mix(bright, vec3<f32>(1.0, 0.72, 0.42), 0.6);
	} else if (accent == 5u) {
		base = mix(bright, vec3<f32>(0.9, 1.0, 0.92), 0.5);
	} else if (accent == 6u) {
		base = mix(bright, vec3<f32>(0.92, 0.97, 1.0), 0.2);
	} else if (accent == 7u) {
		base = mix(deep, bright, 0.22);
	}
	return base * (0.8 + packet * 0.9) + vec3<f32>(1.0, 0.95, 0.85) * packet * 0.25;
}

// A smooth rainbow, for the colour that runs through every burst.
fn rainbow(x: f32) -> vec3<f32> {
	return 0.5 + 0.5 * cos(TAU * (x + vec3<f32>(0.0, 0.33, 0.67)));
}

// The burst into a page: where a particle flies, its colour, and its direction
// of travel (for its streak). Each organ bursts its own way, as its page reads.
struct Boom {
	p: vec2<f32>,
	c: vec3<f32>,
	dir: vec2<f32>,
};
fn explode(sk: u32, s0: f32, s1: f32, s2: f32, ex: f32, t: f32, aspect: f32) -> Boom {
	// Fast out of the organ and never coming to rest: the light is still
	// flying when the page takes it over.
	let e = ex * (1.7 - 0.7 * ex);
	let R = 1.9;
	var a = s0 * TAU;
	var r = e * R * (0.12 + 0.88 * s1);
	var p = vec2<f32>(cos(a), sin(a)) * r;
	var dir = vec2<f32>(cos(a), sin(a));
	let k = min(sk, 8u);
	var c = mix(ORGAN_BRIGHT[k], rainbow(s2 + r * 0.45 + t * 0.4), 0.4);
	switch k {
		case 0u: {
			// The brain: a neural storm, lightning down sixteen branching bolts.
			let bolt = floor(s0 * 16.0);
			a = (bolt + 0.5) / 16.0 * TAU + (s2 - 0.5) * 0.3 * r;
			dir = vec2<f32>(cos(a), sin(a));
			let zig = sin(r * 26.0 + bolt * 3.1) * 0.04 * r;
			p = dir * r + vec2<f32>(-dir.y, dir.x) * zig;
			c = mix(vec3<f32>(0.34, 1.0, 0.62), vec3<f32>(0.74, 0.44, 1.0), fract(s2 * 3.0 + r * 0.7));
			c = mix(c, rainbow(r * 0.5 + t * 0.6 + s2), 0.3);
		}
		case 1u: {
			// Witness: the decision's crystal shatters into rays of light.
			let ray = floor(s0 * 30.0);
			a = (ray + 0.5) / 30.0 * TAU + (s2 - 0.5) * 0.02;
			dir = vec2<f32>(cos(a), sin(a));
			p = dir * e * R * s1;
			c = mix(vec3<f32>(0.55, 0.98, 0.72), vec3<f32>(1.0, 0.86, 0.48), fract(s2 + s1 * 2.0));
			c = mix(c, rainbow(ray * 0.033 + t * 0.3), 0.3);
		}
		case 2u: {
			// Field: a supernova of stars, every kind of memory's colour at once.
			r = e * R * sqrt(s1);
			a = s0 * TAU + r * 0.7;
			dir = vec2<f32>(cos(a), sin(a));
			p = dir * r;
			c = TYPE_RGB[min(u32(s2 * 6.0), 5u)] * 1.15;
		}
		case 3u: {
			// Ledger: the book's ruled lines fly out across the whole screen.
			let row = floor(s1 * 22.0);
			let y = (row / 21.0 * 2.0 - 1.0) * e * 1.05;
			let x = (s0 * 2.0 - 1.0) * e * 2.2 + sin(row * 1.7 + t * 3.0) * 0.02;
			p = vec2<f32>(x, y);
			dir = vec2<f32>(sign(s0 - 0.5), 0.0);
			c = mix(vec3<f32>(1.0, 0.76, 0.36), vec3<f32>(1.0, 0.96, 0.82), step(0.78, s2));
			c = mix(c, rainbow(row * 0.05 + t * 0.3), 0.22);
		}
		case 4u: {
			// Guard: the gate's shockwave, rings racing out; what it stopped shoots straight.
			let ring = floor(s1 * 5.0);
			r = e * R * (0.22 + ring * 0.2) * (0.97 + s2 * 0.06);
			p = dir * r;
			c = mix(vec3<f32>(0.8, 0.4, 0.98), vec3<f32>(1.0, 0.36, 0.62), ring / 4.0);
			if (s2 > 0.8) {
				p = dir * e * R * (0.3 + s1);
				c = vec3<f32>(1.0, 0.28, 0.3);
			}
		}
		case 5u: {
			// Causal: the roots crack outward and down like lightning.
			let crack = floor(s0 * 12.0);
			a = -PI * 0.5 + (crack / 11.0 - 0.5) * PI * 1.6 + (s2 - 0.5) * 0.3 * r;
			dir = vec2<f32>(cos(a), sin(a));
			let zig = sin(r * 22.0 + crack * 2.3) * 0.045 * r;
			p = dir * r + vec2<f32>(-dir.y, dir.x) * zig;
			c = mix(vec3<f32>(1.0, 0.52, 0.22), vec3<f32>(1.0, 0.24, 0.26), s2);
			c = mix(c, vec3<f32>(1.0, 0.92, 0.52), smoothstep(0.75, 1.0, s1));
		}
		case 6u: {
			// Cinema: the film unspools into a turning spiral, every colour a projector throws.
			a = s0 * TAU * 3.0 + e * 5.0;
			r = e * R * s0;
			dir = vec2<f32>(-sin(a), cos(a));
			p = vec2<f32>(cos(a), sin(a)) * r;
			c = rainbow(s0 * 2.0 + t * 0.5) * 1.1;
		}
		case 7u: {
			// System: the strata erupt, layers sliding apart in opposite bands.
			let band = floor(s1 * 9.0);
			let side = select(-1.0, 1.0, (u32(band) % 2u) == 0u);
			let y = (band / 8.0 * 2.0 - 1.0) * e * 0.95;
			let x = (s0 * 2.0 - 1.0) * e * 2.0 + side * e * 0.4;
			p = vec2<f32>(x, y);
			dir = vec2<f32>(side, 0.0);
			c = mix(vec3<f32>(0.46, 0.72, 1.0), vec3<f32>(0.86, 0.95, 1.0), step(0.5, fract(band * 0.5)));
			c = mix(c, rainbow(0.55 + band * 0.04 + t * 0.2), 0.22);
		}
		default: {
			// Healing: the helix unwinds into a double vortex.
			let arm = select(0.0, PI, s1 < 0.5);
			a = s0 * TAU * 2.5 + arm + e * 4.0;
			r = e * R * s0;
			dir = vec2<f32>(-sin(a), cos(a));
			p = vec2<f32>(cos(a), sin(a)) * r;
			c = select(vec3<f32>(0.36, 1.0, 0.66), vec3<f32>(1.0, 0.72, 0.42), s1 < 0.5);
		}
	}
	// The whole burst turns as it opens, and every colour runs through it,
	// out from the white-hot moment it went off.
	p = rot2(p, ex * 0.8);
	dir = rot2(dir, ex * 0.8);
	c = mix(c, rainbow(length(p) * 0.9 - t * 1.4 + s2 * 0.25), 0.35);
	c = mix(c, vec3<f32>(1.0), exp(-ex * 10.0) * 0.65);
	var b: Boom;
	b.p = vec2<f32>(p.x / max(aspect, 0.5) * 1.2, p.y);
	b.c = c;
	b.dir = normalize(vec2<f32>(dir.x / max(aspect, 0.5), dir.y) + vec2<f32>(1e-5, 0.0));
	return b;
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

	// Every organ is seen a little from above, so its rings read as rings.
	{
		let tc = cos(${j});
		let ts = sin(${j});
		local3 = vec3<f32>(local3.x, local3.y * tc - local3.z * ts, local3.y * ts + local3.z * tc);
	}

	// Which organ, which part of it, and how far along its meaning this point is.
	let shape_kind = u32(floor(color_route.z + 0.001));
	let accent = u32(round(fract(color_route.z) * 16.0));
	let flow = color_route.y;
	let kk = min(shape_kind, 8u);
	// Light travelling through the shape along its meaning: around the drum's
	// rings, page to page, into the gate, up the helix, down the roots, along
	// the film, up the strata, into the witness's crystal.
	var packet = 0.0;
	if (!reduced) {
		packet = pow(0.5 + 0.5 * cos(TAU * (flow * ORGAN_PACKETS[kk] - t * ORGAN_SPEED[kk])), 14.0);
	}

	// Project the local formation point to the organ's screen anchor.
	var pos = organ_anchor + vec2<f32>(local3.x / aspect, local3.y) * scale;

	// ── AXONS: a DELICATE signal stream flowing FROM the cortex OUT to each organ.
	// Only ~4.5% of an outer organ's particles ride the fiber, and the fiber runs
	// only through the OPEN SPACE between the two bodies (starts just outside the
	// cortex, ends just before the organ) so it never crosses through a shape. Each
	// particle is a small point flowing outward; sparse bright packets pulse along
	// it. Thin, single-file (tiny normal jitter, no fat rope), tapering to the organ.
	if (!reduced && route > 0.5 && seed0 < 0.07) {
		let cortexP = vec2<f32>(${N[0]}, ${N[1]});
		let dir = normalize(organ_anchor - cortexP + vec2<f32>(0.00001, 0.0));
		let normal = vec2<f32>(-dir.y, dir.x);
		// The thread spans the FULL distance cortex -> organ, so it visibly connects
		// the brain to the destination (no floating gap). Particles are distributed
		// along the whole line (by seed) and drift outward slowly; brightness fades
		// at BOTH ends so it emerges from the brain and arrives at the organ without a
		// hard dot piling onto either body.
		let base = seed1;                          // static position along the line 0..1
		// Every organ reports to the hub: its light flows in along the link.
		let travel = fract(base - t * 0.06);
		// gentle single curve across the whole span, tiny amplitude = one clean thread.
		let bow = sin(travel * PI) * 0.03;
		let jitter = (seed2 - 0.5) * 0.006;
		pos = mix(cortexP, organ_anchor, travel) + normal * (bow + jitter);
		// bright nerve packets sweeping outward along the fiber.
		let pulse = pow(0.5 + 0.5 * sin(travel * TAU * 2.0 + t * 2.6 + seed2 * TAU), 10.0);
		// fade near BOTH endpoints (0 at ends, full in the middle) so the line reads
		// as connecting, not as dots stacked on the bodies.
		let endsFade = smoothstep(0.0, 0.14, travel) * (1.0 - smoothstep(0.86, 1.0, travel));
		axon = (0.35 + pulse * 1.4) * endsFade + 0.15;
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
	// The dolly (a click going into an organ) is a camera move, applied last;
	// the burst's acts below run only when it is not.
	let dolly = u.portal.z > 0.5;
	let bursting = u.interaction.z >= 0.0 && progress > 0.0 && !dolly;
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
	// Twinkle: now and then a single point flares, so the organs glitter.
	var twinkle = 0.0;
	if (!reduced) { twinkle = pow(max(0.0, sin(t * (0.7 + seed1 * 1.9) + seed0 * 40.0)), 60.0) * 2.4; }
	var energy = living + axon * 1.3 + focus * 1.7 + systole * 2.2 + weather
		+ sin(intro * PI) * 1.1 + flash * 7.0 + burst_blaze + packet * 1.5 * (1.0 - step(0.001, axon)) + twinkle;
	if (hover_on && !hovered) { energy *= 0.30; }
	energy = max(energy, 0.08);

	// The dolly: the camera flies into the clicked organ. u.portal.xy is its
	// anchor, u.portal.z the zoom and u.intro.w how far the organ has glided to
	// the centre. Everything else streams outward past the edges of the screen.
	var dolly_size = 1.0;
	var streak = vec2<f32>(0.0);
	var boom_c = vec3<f32>(0.0);
	var boom_k = 0.0;
	if (dolly) {
		pos = (pos - u.portal.xy) * u.portal.z + u.portal.xy * (1.0 - u.intro.w);
		dolly_size = 1.0 + (u.portal.z - 1.0) * 0.35;
		if (!selected) { energy *= 1.0 - 0.45 * u.intro.w; }
		// The burst: the organ explodes wide in its own way and every particle of
		// the Palace is swept into it, colour everywhere, streaking as it flies.
		let ex = u.interaction.w;
		if (ex > 0.0 && !reduced) {
			let sk = u32(max(u.portal.w - 1.0, 0.0) + 0.5);
			let boom = explode(sk, seed0, seed1, seed2, ex, t, aspect);
			// The organ's own light leaves from where it is; the rest pours in
			// through the centre from off the screen.
			let take = smoothstep(0.0, 0.3, ex);
			pos = select(boom.p, mix(pos, boom.p, take), selected);
			if (!selected) { energy *= smoothstep(0.02, 0.16, ex); }
			energy = energy * 0.6 + 1.6 + 2.2 * (1.0 - ex) * (0.6 + 0.4 * seed2);
			boom_c = boom.c;
			boom_k = smoothstep(0.0, 0.2, ex);
			streak = boom.dir * (2.0 + 9.0 * (1.0 - ex));
		}
	}

	let corner = QUAD[vertex_index];
	let pixel = 2.0 / max(min(u.viewport.x, u.viewport.y), 1.0);
	let depth_size = clamp(1.0 + local3.z * 0.12, 0.82, 1.24) * dolly_size;
	// Axon particles are a THIN thread: a small fixed base (packets briefly widen),
	// NOT the fat organ-body size. Organ particles keep the full size profile.
	let is_axon = step(0.001, axon);
	let organ_size = (1.35 + baked.w * 1.9 + focus * 1.5 + systole * 0.6 + min(flash * 3.0, 3.0) + min(burst_blaze * 1.2, 4.0)) * depth_size;
	let axon_size = 0.9 + axon * 0.6;
	let size = mix(organ_size, axon_size, is_axon);

	// Color: the REAL iridescent palette, seeded per-particle, energized by the beat.
	let world = vec3<f32>(pos.x, pos.y, local3.z);
	// Colour: the organ's own palette, its part of the shape, and the light
	// travelling through it; the links glow in their organ's colour.
	var col = organ_color(kk, select(accent, 2u, axon > 0.001), color_route.x, select(packet, 0.0, axon > 0.001));
	col = mix(col, vec3<f32>(1.00, 0.86, 0.55), clamp((energy - 2.6) * 0.35, 0.0, 0.5));
	col = mix(col, vec3<f32>(1.0), clamp((energy - 3.6) * 0.4, 0.0, 0.5));

	if (boom_k > 0.0) { col = mix(col, boom_c, boom_k); }
	var out: VSOut;
	// Bursting particles streak along their flight; the rest are round.
	let sl = length(streak);
	if (sl > 0.0) {
		let d = streak / sl;
		let n = vec2<f32>(-d.y, d.x);
		out.position = vec4<f32>(pos + (corner.x * d * (1.0 + sl) + corner.y * n) * pixel * size, 0.0, 1.0);
	} else {
		out.position = vec4<f32>(pos + corner * pixel * size, 0.0, 1.0);
	}
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
`;function L(e){return e-Math.floor(e)}function R(e){let t=e<0?0:e>1?1:e;return t*t*(3-2*t)}function z(e){return L(Math.sin(e*12.9898+78.233)*43758.5453)}var B=Math.PI*2;function V(e,t){return z(e*(1.371+t*.913)+t*7.31)}function H(e,t,n){let r=n*Math.cbrt(V(e,t)),i=V(e,t+1)*B,a=2*V(e,t+2)-1,o=Math.sqrt(Math.max(0,1-a*a));return[r*o*Math.cos(i),r*a,r*o*Math.sin(i)]}function me(e){let t=V(e,7);if(t<.1){let t=Math.floor(V(e,1)*90),n=U(Math.floor(z(t*3.1+1)*1e6)+7),r=U(Math.floor(z(t*5.7+2)*1e6)+11),i=V(e,2);return[n[0]+(r[0]-n[0])*i,n[1]+(r[1]-n[1])*i,n[2]+(r[2]-n[2])*i,i,2]}let[n,r,i,a]=U(e);return[n,r,i,L(Math.atan2(i,n)/B+r*.25),t<.2?1:a<-.04?7:0]}function U(e){let t=z(e*.7351+1),n=z(e*1.2971+5),r=z(e*2.1637+9),i=z(e*3.5391+13);if(t<.15){let e=2*n-1,t=Math.sqrt(Math.max(0,1-e*e)),a=r*B,o=.2+.16*Math.cbrt(i);return[t*Math.cos(a)*o,e*o,t*Math.sin(a)*o,-1]}let a=n<.5?-1:1,o=2*r-1,s=Math.acos(Math.max(-1,Math.min(1,o))),c=Math.sin(s),l=.2,u=a===1?l+i*(Math.PI-2*l):Math.PI+l+i*(Math.PI-2*l),d=c*Math.cos(u),f=o,p=c*Math.sin(u),m=.095*Math.sin(7*u)*(c*c)+.075*Math.cos(9*u+2*s)*c+.065*Math.sin(6*s)+.05*Math.sin(11*u)*Math.sin(4*s),h=Math.min(Math.abs(u-Math.PI/2),Math.abs(u-3*Math.PI/2)),g=-.16*Math.exp(-(h*h)/(.1*.1))*c,_=.02*(z(e*5.19+2)-.5),v=.92+m+g+_;return d*=v,f*=v,p*=v,d+=a*.05*Math.exp(-(h*h)/.0256)*c,f*=.9,p*=1.06,d*=1.04,[d,f,p,m+g]}function W(e){let t=V(e,5)<.5?-1:1,n=Math.floor(V(e,0)*5),r=(.32+n*.07)*Math.PI,i=t<0?Math.PI-r:r,a=Math.cos(i),o=Math.sin(i),s=(t<0?4-n:5+n)/10,c=V(e,1);if(c<.06)return[0,-.72+V(e,2)*1.44,0,s,2];if(c<.2){let t=V(e,2)*3,[n,r]=t<1?[.06+t*1,.72]:t<2?[.06+(t-1)*1,-.72]:[1.06,-.72+(t-2)*1.44];return[a*n,r,o*n,s,3]}let l=Math.floor(V(e,2)*10),u=-.56+1.12/9*l+(V(e,3)-.5)*.01,d=.3+z(n*5.1+l*2.3+t)*.62,f=.14+V(e,4)*d;return[a*f,u,o*f,s,0]}function he(e){let t=V(e,0);if(t<.36){let t=V(e,1)*B,n=V(e,2)*B,r=1+.05*Math.cos(n);return[r*Math.cos(t),r*Math.sin(t),.05*Math.sin(n),t/B,0]}if(t<.52){let t=.96*Math.sqrt(V(e,1)),n=V(e,2)*B;return[t*Math.cos(n),t*Math.sin(n),(V(e,3)-.5)*.016,.5,3]}if(t<.66){let t=Math.floor(V(e,1)*8),n=z(t*3.7)*B,r=.25+z(t*5.3)*.55,[i,a]=H(e,2,.07);return[r*Math.cos(n)+i,r*Math.sin(n)+a,.05+V(e,5)*.06,1,1]}let n=Math.floor(V(e,1)*16),r=z(n*2.9+1)*B,i=.15+z(n*4.1+2)*.7,a=V(e,2),o=i*Math.cos(r),s=i*Math.sin(r);return[o*(1+(1-a)*.6),s*(1+(1-a)*.6),.06+(1-a)*1.3,a,0]}function G(e){let t=2.2,n=.5,r=V(e,0),i=V(e,1);if(r<.8){let a=r>=.4,o=i*t*B+(a?Math.PI:0),s=(V(e,2)-.5)*.035;return[Math.cos(o)*(n+s),-1+i*2,Math.sin(o)*(n+s),i,a?4:0]}let a=(Math.floor(i*18)+.5)/18,o=a*t*B,s=V(e,2)*2-1;return[Math.cos(o)*n*s,-1+a*2,Math.sin(o)*n*s,a,5]}var K=null;function ge(){if(K)return K;let e=[],t=(n,r,i,a,o)=>{let s=[n[0]+r[0]*i,n[1]+r[1]*i,n[2]+r[2]*i];if(e.push({a:n,b:s,level:a}),!(a+1>=5))for(let e=0;e<3;e++){let n=e/3*B+z(o*7.1+e)*1.2,c=.42+z(o*3.3+e)*.3,l=[Math.sin(c)*Math.cos(n)+r[0]*.5,-Math.cos(c),Math.sin(c)*Math.sin(n)+r[2]*.5],u=Math.hypot(l[0],l[1],l[2]);t(s,[l[0]/u,l[1]/u,l[2]/u],i*.66,a+1,o*3+e+1)}};return t([0,.98,0],[0,-1,0],.52,0,1),K=e,e}function _e(e){if(V(e,0)<.06){let[t,n,r]=H(e,1,.07);return[t,.98+n,r,0,1]}let t=ge(),n=t[Math.min(t.length-1,Math.floor(V(e,1)*t.length))],r=V(e,2),i=(1-n.level/5)*.012;return[n.a[0]+(n.b[0]-n.a[0])*r+(V(e,3)-.5)*i,n.a[1]+(n.b[1]-n.a[1])*r,n.a[2]+(n.b[2]-n.a[2])*r+(V(e,4)-.5)*i,(n.level+r)/5,0]}var ve=(()=>{let e=Array.from({length:9},(e,t)=>.5+z(t*9.7+2)*1.2),t=e.reduce((e,t)=>e+t,0),n=-.95;return e.map(e=>{let r=n;return n+=e/t*1.9,[r,n-.06]})})();function ye(e){let t=Math.floor(V(e,0)*ve.length),[n,r]=ve[t],i=.62+t%2*.03;if(V(e,1)<.6){let a=V(e,2)*B,o=n+V(e,3)*(r-n);return[Math.cos(a)*i,o,Math.sin(a)*i,(o+1)/2,t%2?7:6]}let a=i*Math.sqrt(V(e,2)),o=V(e,3)*B;return[Math.cos(o)*a,r,Math.sin(o)*a,(r+1)/2,7]}function be(e){let t=.82;if(V(e,0)<.08){let n=V(e,1)*2-1,r=(V(e,2)*2-1)*(1-Math.abs(n)),i=(1-Math.abs(n)-Math.abs(r))*(V(e,3)<.5?-1:1);return[n*.13,t+r*.16,i*.13,1,1]}let n=Math.floor(V(e,1)*26),r=z(n*7.13+1)*B,i=.7+z(n*2.3+5)*.4,a=-.95+z(n*5.7+3)*.55,o=[Math.cos(r)*i,a,Math.sin(r)*i];if(V(e,0)<.2){let[t,n,r]=H(e,4,.05);return[o[0]+t,o[1]+n,o[2]+r,0,2]}let s=V(e,2),c=Math.sin(s*Math.PI)*.18;return[o[0]*(1-s)+o[0]*c,o[1]+(t-o[1])*s,o[2]*(1-s)+o[2]*c,s,0]}function xe(e){let t=V(e,0)*B,n=V(e,1)<.24,r=n?(V(e,2)<.5?-1:1)*.33:(V(e,2)*2-1)*.27,i=L(t/B*22),a=.95,o=t/2,s=(a+r*Math.cos(o))*Math.cos(t),c=(a+r*Math.cos(o))*Math.sin(t),l=r*Math.sin(o),u=n?L(i*3)<.5?2:3:i<.06?3:0;return[s,l,c,t/B,u]}var q=[{radius:1.18,cells:210},{radius:.86,cells:132},{radius:.54,cells:72}],J=414,Se=Math.PI*(3-Math.sqrt(5));function Ce(e){let t=Math.floor(z(e*1.6180339887+.5)*J);t>=J&&(t=413);let n=0,r=t;r>=q[0].cells&&(r-=q[0].cells,n=1),n===1&&r>=q[1].cells&&(r-=q[1].cells,n=2);let i=q[n],a=i.cells,o=1-(2*r+1)/a,s=Math.sqrt(Math.max(0,1-o*o)),c=Se*r,l=Math.cos(c)*s,u=o,d=Math.sin(c)*s,f=.05+.012*(2-n),p=z(e*2.31+5),m=z(e*3.97+9),h=f*Math.cbrt(z(e*4.13+1)),g=2*p-1,_=Math.sqrt(Math.max(0,1-g*g)),v=m*B,y=h*_*Math.cos(v),b=h*g,x=h*_*Math.sin(v),S=i.radius;return[(l*S+y)*1.06,(u*S+b)*1.06,(d*S+x)*1.06]}function we(e){let[t,n,r]=Ce(e),i=Math.floor(z(e*1.6180339887+.5)*J);i>=J&&(i=413);let a=Math.floor(z(i*4.77+1.3)*6);return[t,n,r,i/J,8+a]}function Te(e,t){switch(e){case 0:return me(t);case 1:return be(t);case 2:return we(t);case 3:return W(t);case 4:return he(t);case 5:return _e(t);case 6:return xe(t);case 7:return ye(t);default:return G(t)}}function Ee(e,t){let[n,r,i]=Te(e,t);return[n,r,i]}function De(e){let t=navigator.hardwareConcurrency||8,n=window.innerWidth<760,r=t<=4||(window.devicePixelRatio||1)>2.2;return e?n?1e4:16e3:r?n?12e3:2e4:n?18e3:t>=10?55e3:4e4}var Oe=class{engine;pipeline=null;bindGroup=null;uniformBuffer=null;particleBuffer=null;uniformData=new Float32Array(de);placed=[];count=0;hoveredIndex=-1;hoverStrength=0;reducedMotion=!1;bornMs;burst=null;onFlashPeak=null;watchdog=null;constructor(e){this.engine=e,this.bornMs=e.wallNowMs}setReducedMotion(e){this.reducedMotion=e}setHovered(e){this.burst||(this.hoveredIndex=e>=0&&e<this.placed.length?e:-1)}indexOfHref(e){return e?this.placed.findIndex(t=>t.href===e):-1}get isBursting(){return this.burst!==null}startBurst(e,t,n=fe){if(this.burst)return!1;let r=this.placed.find(t=>t.href===e);if(!r)return!1;this.hoveredIndex=this.placed.indexOf(r),this.hoverStrength=1,this.burst={href:e,startMs:this.engine.wallNowMs,callbackFired:!1,at:n,dolly:n<=.42000099999999996},this.onFlashPeak=t;let i=this.reducedMotion?D:E,a=this.burst.dolly&&!this.reducedMotion?pe*i+O*k:n*i;return this.watchdog=setTimeout(()=>this.fireFlashCallback(),a+160),!0}uploadRegions(e){let t=this.engine.gpuDevice;if(!t||e.length===0)return;this.placed=e.map((e,t)=>{let n=M[e.href]??[0,0,t*.01,.14],r=A[e.href]??t%9,i=F(r);return{href:e.href,x:n[0],y:n[1],z:n[2],scale:n[3],kind:r,below:i.below,reach:i.reach}}),this.count=De(this.reducedMotion);let n=new Float32Array(this.count*ue);for(let e=0;e<this.count;e++){let t=e+1,r=e*ue,i=this.routeForIndex(e),a=this.placed[i],o=Te(a.kind,t);n[r+0]=o[0],n[r+1]=o[1],n[r+2]=o[2],n[r+3]=z(t*2.83+12),n[r+4]=a.x,n[r+5]=a.y,n[r+6]=a.z,n[r+7]=a.scale,n[r+8]=z(t*1.7+5),n[r+9]=o[3],n[r+10]=a.kind+o[4]/16,n[r+11]=i}this.uniformBuffer?.destroy(),this.particleBuffer?.destroy(),this.uniformBuffer=t.createBuffer({label:`palace-brain-uniforms`,size:this.uniformData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.particleBuffer=t.createBuffer({label:`palace-brain-particles`,size:n.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,n),this.createPipeline(t)}routeForIndex(e){let t=this.placed.length;if(t===0)return 0;let n=2+(t-1),r=Math.floor(this.count*2/n);if(e<r)return 0;let i=this.count-r,a=Math.max(1,Math.floor(i/Math.max(1,t-1)));return Math.min(t-1,1+Math.floor((e-r)/a))}createPipeline(e){if(!this.uniformBuffer||!this.particleBuffer)return;e.pushErrorScope(`validation`);let t=e.createShaderModule({label:`palace-brain-shader`,code:I});t.getCompilationInfo().then(e=>{let t=e.messages.filter(e=>e.type===`error`);t.length>0&&console.error(`[palace-brain] WGSL:`,t.map(e=>`${e.lineNum}:${e.linePos} ${e.message}`).join(`
`))}),this.pipeline=e.createRenderPipeline({label:`palace-brain-pipeline`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`,buffers:[{arrayStride:48,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`},{shaderLocation:2,offset:32,format:`float32x4`}]}]},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bindGroup=e.createBindGroup({label:`palace-brain-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),e.popErrorScope().then(e=>{e&&console.error(`[palace-brain] pipeline validation:`,e.message)})}fireFlashCallback(){if(!this.burst||this.burst.callbackFired||!this.onFlashPeak)return;this.burst.callbackFired=!0;let e=this.burst.href,t=this.onFlashPeak;this.onFlashPeak=null,this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,queueMicrotask(()=>t(e))}compute(){let e=this.engine.gpuDevice;if(!e||!this.uniformBuffer||this.placed.length===0)return;let t=this.engine.wallNowMs,n=this.burst?1:+(this.hoveredIndex>=0);this.hoverStrength+=(n-this.hoverStrength)*(this.reducedMotion?1:.16),this.hoverStrength<.001&&(this.hoverStrength=0);let r=0,i=-1,a=this.hoveredIndex>=0?this.placed[this.hoveredIndex]:null,o=a?.x??0,s=a?.y??0,c=0,l=0,u=0;if(this.burst?.dolly){i=this.indexOfHref(this.burst.href);let e=i>=0?this.placed[i]:null,n=(this.reducedMotion?D:E)*pe,a=t-this.burst.startMs,d=Math.min(1,Math.max(0,a/n)),f=this.reducedMotion?0:Math.min(1,Math.max(0,(a-n)/O));(this.reducedMotion?d>=1:f>=k)&&this.fireFlashCallback(),l=R(d),c=1+1.4*d*d+f*.6,o=e?.x??0,s=e?.y??0,r=f,u=(e?.kind??0)+1}else if(this.burst){i=this.indexOfHref(this.burst.href),a=i>=0?this.placed[i]:a;let e=this.reducedMotion?D:E;r=Math.min(1,Math.max(0,(t-this.burst.startMs)/e));let n=this.reducedMotion?.3:this.burst.at;r>=n&&this.fireFlashCallback();let c=a?.x??0,l=a?.y??0,u=this.reducedMotion?1:R(r/.42);o=c*(1-u),s=l*(1-u)}let d=this.engine.params[11]>.5?1:Math.min(1,(t-this.bornMs)/1650),f=this.reducedMotion||r<=0||this.burst?.dolly?0:Math.exp(-(((r-.58)/.065)**2)),p=this.engine.params[10]||0;this.uniformData[0]=this.engine.params[6]||1,this.uniformData[1]=this.engine.params[7]||1,this.uniformData[2]=p,this.uniformData[3]=+!!this.reducedMotion,this.uniformData[4]=this.burst?i:this.hoveredIndex,this.uniformData[5]=this.hoverStrength,this.uniformData[6]=i,this.uniformData[7]=r,this.uniformData[8]=o,this.uniformData[9]=s,this.uniformData[10]=c,this.uniformData[11]=this.burst?.dolly?u:f,this.uniformData[12]=d,this.uniformData[13]=this.placed.length,this.uniformData[14]=this.engine.params[5]||0,this.uniformData[15]=l,e.queue.writeBuffer(this.uniformBuffer,0,this.uniformData)}render(e){this.pipeline&&this.bindGroup&&this.count!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.particleBuffer),e.draw(6,this.count))}pickAt(e,t){if(this.burst||this.placed.length===0)return null;let n=(this.engine.params[6]||1)/(this.engine.params[7]||1),r=-1,i=1/0;for(let a=0;a<this.placed.length;a++){let o=this.placed[a],s=(e-o.x)*n,c=t-o.y,l=Math.hypot(s,c)/(o.scale*Math.max(1.35,o.reach*1.05));a===this.hoveredIndex&&(l*=.78),l<1.1&&l<i&&(r=a,i=l)}return r<0?null:{index:r,href:this.placed[r].href}}getScreenPositions(){let e=(this.engine.params[6]||1)/Math.max(1,this.engine.params[7]||1),t=Math.max(e,1),n=1/Math.min(e,1);return this.placed.map(e=>({href:e.href,ndcX:e.x*t,ndcY:e.y*n,screenX:e.x,screenY:e.y,depth:Math.min(1,Math.max(0,.72+e.z)),radius:e.scale*e.below*n,visible:!0}))}dispose(){this.watchdog&&clearTimeout(this.watchdog),this.watchdog=null,this.uniformBuffer?.destroy(),this.particleBuffer?.destroy(),this.uniformBuffer=null,this.particleBuffer=null,this.pipeline=null,this.bindGroup=null,this.placed=[],this.count=0,this.burst=null,this.onFlashPeak=null}},ke={V:{w:.82,strokes:[[[0,1],[.41,0],[.82,1]]]},E:{w:.68,strokes:[[[.68,1],[0,1],[0,0],[.68,0]],[[0,.5],[.54,.5]]]},S:{w:.76,strokes:[[[.74,.84],[.6,.99],[.38,1],[.16,.97],[.03,.84],[.04,.66],[.18,.55],[.4,.5],[.6,.45],[.74,.34],[.76,.17],[.62,.02],[.38,0],[.16,.02],[.02,.16]]]},T:{w:.8,strokes:[[[0,1],[.8,1]],[[.4,1],[.4,0]]]},I:{w:.3,strokes:[[[0,1],[.3,1]],[[.15,1],[.15,0]],[[0,0],[.3,0]]]},G:{w:.8,strokes:[[[.78,.82],[.62,.98],[.4,1],[.18,.97],[.03,.8],[0,.5],[.03,.2],[.18,.03],[.4,0],[.62,.03],[.8,.2],[.8,.46],[.46,.46]]]}},Ae=`VESTIGE`,je=.24;function Me(e){return e-Math.floor(e)}function Y(e){return Me(Math.sin(e*12.9898+78.233)*43758.5453)}function X(e,t){return Y(e*(1.618+t*.731)+t*9.17)}function Ne(){let e=[],t=[],n=0,r=0;for(let i of Ae){let a=ke[i];for(let o of a.strokes)for(let a=0;a<o.length;a++){let s=[o[a][0]+n,o[a][1]];if((a===0||a===o.length-1||i!==`S`&&i!==`G`)&&t.push(s),a===0)continue;let c=[o[a-1][0]+n,o[a-1][1]],l=Math.hypot(s[0]-c[0],s[1]-c[1]);e.push({a:c,b:s,at:r,len:l}),r+=l}n+=a.w+je}return{segs:e,width:n-je,total:r,corners:t}}var Pe=`
struct U {
	view: vec4f,  // width px, height px, time s, fade
	place: vec4f, // centre x, centre y (NDC), NDC x per logo unit, aspect
	misc: vec4f,  // birth 0..1, word width (units), reduced motion, 0
};
@group(0) @binding(0) var<uniform> u: U;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
const TAU = 6.2831853;

struct VO {
	@builtin(position) pos: vec4f,
	@location(0) local: vec2f,
	@location(1) col: vec3f,
};

@vertex
fn vs(@location(0) a: vec4f, @location(1) b: vec4f, @builtin(vertex_index) vi: u32) -> VO {
	// a: x, y (logo units), flow along the word (0..1), kind (0 fibre, 1 dendrite, 2 synapse, 3 spark)
	// b: direction xy (sparks, fibres' twist), seed, size (px)
	let t = u.view.z;
	let still = u.misc.z > 0.5;
	let kind = u32(a.w + 0.5);
	let seed = b.z;
	var p = a.xy;
	var glow = 1.0;
	if (kind == 0u && !still) {
		// The fibres twist round each other along the stroke, and breathe.
		p += b.xy * 0.018 * sin(t * 1.3 + seed * TAU);
	}
	if (kind == 3u) {
		// A spark flies out from the word along its own ray, and dims as it goes.
		let ph = select(fract(seed * 7.31 + t * (0.05 + seed * 0.07)), seed, still);
		p = a.xy + b.xy * ph * 2.4;
		glow = (1.0 - ph) * (1.0 - ph);
	}
	let unit = u.place.z;
	var ndc = vec2f(u.place.x + (p.x - u.misc.y * 0.5) * unit, u.place.y + (p.y - 0.5) * unit * u.place.w);
	// Born out of the brain below, letter light arriving a little out of step.
	let birth = clamp((u.misc.x - seed * 0.35) / 0.65, 0.0, 1.0);
	let e = birth * birth * (3.0 - 2.0 * birth);
	ndc = mix(vec2f(0.0, -0.06), ndc, e);

	let green = vec3f(0.3, 1.0, 0.58);
	let violet = vec3f(0.7, 0.4, 1.0);
	let wave = 0.5 + 0.5 * sin(p.x * 1.6 - t * 0.7 + seed * 2.0);
	var col = mix(green, violet, wave);
	// Impulses run through the word from left to right.
	let impulse = select(pow(0.5 + 0.5 * cos(TAU * (a.z * 3.0 - t * 0.2)), 18.0), 0.0, still);
	var energy = 0.75 + impulse * 2.2;
	if (kind == 1u) {
		col = mix(green, vec3f(0.3, 0.92, 0.95), 0.45);
		energy = 0.55 + impulse * 1.6;
	} else if (kind == 2u) {
		col = mix(violet, vec3f(1.0), 0.4);
		energy = 1.6 + 1.4 * select(pow(0.5 + 0.5 * sin(t * 2.6 + seed * 31.0), 6.0), 0.5, still);
	} else if (kind == 3u) {
		col = mix(green, violet, step(0.5, fract(seed * 13.0)));
		energy = 1.1 * glow;
	}
	let fade = u.view.w * e;
	let q = QUAD[vi];
	var o: VO;
	o.pos = vec4f(ndc + q * b.w * vec2f(2.0 / max(u.view.x, 1.0), 2.0 / max(u.view.y, 1.0)), 0.0, 1.0);
	o.local = q;
	o.col = col * energy * fade;
	if (fade <= 0.002) { o.pos = vec4f(3.0, 3.0, 0.0, 1.0); }
	return o;
}

@fragment
fn fs(i: VO) -> @location(0) vec4f {
	let r = clamp(length(i.local), 0.0, 1.0);
	if (r >= 1.0) { discard; }
	let profile = pow(1.0 - r, 2.4) * 1.3 + (1.0 - r) * 0.35;
	return vec4f(i.col * profile, 1.0);
}
`,Fe=class{highRefresh=!0;engine;pipeline=null;bind=null;ubuf=null;vbuf=null;count=0;width=1;uni=new Float32Array(12);place={x:0,y:.76,w:1};bornMs;leftMs=null;reduced=!1;constructor(e,t=!1){this.engine=e,this.reduced=t,this.bornMs=e.wallNowMs,this.build()}setPlace(e,t,n){this.place={x:e,y:t,w:n},this.engine.requestRender()}leave(){this.leftMs===null&&(this.leftMs=this.engine.wallNowMs)}build(){let e=this.engine.gpuDevice;if(!e)return;let{segs:t,width:n,total:r,corners:i}=Ne();this.width=n;let a=typeof window<`u`&&window.innerWidth<760,o=this.reduced?9e3:a?14e3:26e3,s=new Float32Array(o*8),c=e=>{let n=0,r=t.length-1;for(;n<r;){let i=n+r+1>>1;t[i].at<=e?n=i:r=i-1}let i=t[n],a=Math.min(1,Math.max(0,(e-i.at)/Math.max(i.len,1e-6))),o=(i.b[0]-i.a[0])/Math.max(i.len,1e-6),s=(i.b[1]-i.a[1])/Math.max(i.len,1e-6);return{p:[i.a[0]+(i.b[0]-i.a[0])*a,i.a[1]+(i.b[1]-i.a[1])*a],dir:[o,s],seg:i}};for(let e=0;e<o;e++){let t=e+1,a=X(t,0),o=e*8,l=0,u=0,d=0,f=0,p=0,m=0,h=1.6;if(a<.5){let e=X(t,1)*r,{p:i,dir:a}=c(e),o=Math.floor(X(t,2)*3),s=-a[1],f=a[0],g=.03*Math.sin(e*34+o*2.1)+(X(t,3)-.5)*.012;l=i[0]+s*g,u=i[1]+f*g,d=i[0]/n,p=s,m=f,h=1.3+X(t,4)*1.1}else if(a<.68){let e=Math.floor(X(t,1)*140),{p:i}=c(Y(e*3.17+1)*r),a=Y(e*7.7+2)*Math.PI*2,o=.1+Y(e*1.9+3)*.26,s=X(t,2),p=X(t,3)<.4&&s>.5,m=a+(Y(e*5.3+4)-.5)*1.6,g=p?i[0]+Math.cos(a)*o*.5+Math.cos(m)*o*(s-.5):i[0]+Math.cos(a)*o*s,_=p?i[1]+Math.sin(a)*o*.5+Math.sin(m)*o*(s-.5):i[1]+Math.sin(a)*o*s;l=g+(X(t,4)-.5)*.006,u=_+(X(t,5)-.5)*.006,d=i[0]/n,f=1,h=1.1+(1-s)*.8}else if(a<.75){let e=X(t,1)<.6?i[Math.floor(X(t,2)*i.length)]:c(Y(Math.floor(X(t,2)*40)*2.7)*r).p,a=.035*Math.cbrt(X(t,3)),o=X(t,4)*Math.PI*2;l=e[0]+Math.cos(o)*a,u=e[1]+Math.sin(o)*a,d=e[0]/n,f=2,h=1.8+X(t,5)*1.6}else{let e=X(t,1)*n,r=.2+X(t,2)*.6,i=e-n/2+(X(t,3)-.5)*.8,a=(r-.5)*2.2+(X(t,4)-.5)*.9,o=Math.hypot(i,a)||1;i/=o,a/=o,l=e,u=r,p=i,m=a,d=e/n,f=3,h=1+X(t,5)*1.3}s.set([l,u,d,f,p,m,X(t,9),h],o)}this.count=o,this.vbuf=e.createBuffer({label:`palace-logo-points`,size:s.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vbuf,0,s),this.ubuf=e.createBuffer({label:`palace-logo-uniforms`,size:this.uni.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let l=e.createShaderModule({label:`palace-logo`,code:Pe});l.getCompilationInfo().then(e=>{for(let t of e.messages)t.type===`error`&&console.error(`[palace-logo] ${t.lineNum}:${t.linePos} ${t.message}`)}),this.pipeline=e.createRenderPipeline({label:`palace-logo`,layout:`auto`,vertex:{module:l,entryPoint:`vs`,buffers:[{arrayStride:32,stepMode:`instance`,attributes:[{shaderLocation:0,offset:0,format:`float32x4`},{shaderLocation:1,offset:16,format:`float32x4`}]}]},fragment:{module:l,entryPoint:`fs`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bind=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.ubuf}}]})}compute(){let e=this.engine.gpuDevice;if(!e||!this.ubuf)return;let t=this.engine.wallNowMs,n=this.engine.params[6]||1,r=this.engine.params[7]||1,i=this.reduced?1:Math.min(1,(t-this.bornMs)/1900),a=this.leftMs===null?1:Math.max(0,1-(t-this.leftMs)/260),o=n/r;this.uni.set([n,r,this.engine.simSeconds,a,this.place.x,this.place.y,this.place.w/this.width,o,i,this.width,+!!this.reduced,0]),e.queue.writeBuffer(this.ubuf,0,this.uni)}render(e){this.pipeline&&this.bind&&this.vbuf&&this.count!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bind),e.setVertexBuffer(0,this.vbuf),e.draw(6,this.count))}dispose(){this.vbuf?.destroy(),this.ubuf?.destroy(),this.vbuf=null,this.ubuf=null}},Z=[{href:`/trace`,label:`TRACE`,family:`memory`,center:!0},{href:`/field`,label:`FIELD`,family:`system`},{href:`/ledger`,label:`LEDGER`,family:`temporal`},{href:`/guard`,label:`GUARD`,family:`immune`},{href:`/healing`,label:`HEALING`,family:`signal`},{href:`/causal`,label:`CAUSAL`,family:`reasoning`},{href:`/system`,label:`SYSTEM`,family:`system`},{href:`/cinema`,label:`CINEMA`,family:`signal`},{href:`/graph`,label:`WITNESS`,family:`reasoning`}];function Ie(e){return Z.find(t=>t.href===e)}var Le={reasoning:[...T(`#FFFFFF`),1],memory:[...T(`#FFFFFF`),1],immune:[...T(`#FFFFFF`),1],temporal:[...T(`#FFFFFF`),1],signal:[...T(`#FFFFFF`),1],system:[...T(`#FFFFFF`),1]},Re=[...T(`#FFFFFF`),1],ze=.03,Be=.05,Ve=.03,He=1,Ue=.95,Q=[],$=[];function We(e,t,n){return e+(t-e)*n}function Ge(e){return e<0?0:e>1?1:Number.isFinite(e)?e:0}function Ke(e,t={}){let n=t.hoveredHref??null,r=t.dimUnhovered??!0,i=t.aspect??1,a=i<.85,o=.7-.06*(a?Ge((.85-i)/(.85-.46)):0),s=0;$.length=0;for(let t=0;t<e.length;t++){let c=e[t];if(c.visible===!1)continue;let l=Ie(c.href);if(!l)continue;let u=Ge(c.depth),d=n!==null&&c.href===n,f=We(Ve,Be,u),p=We(Ue,He,u);l.center&&(f*=1.22),d?(f*=1.28,p=1):r&&n!==null&&(p*=.32);let m=d?1:.8+u*.2,h=1+We(.42,.1,m),g=l.label.length*f*.62,_=c.radius??.14,v=c.ndcX-g/2,y=c.ndcY-_-ze-f*.8;a&&y>o&&(y=o-(y-o));let b=Math.max(i,1)*.97;v=Math.min(b-g,Math.max(-b,v));for(let e=0;e<$.length;e++){let t=$[e],n=v<t.x1&&v+g>t.x0,r=Math.abs(y-t.y)<(f+t.size)*.75;n&&r&&(y=t.y-(f+t.size)*.85)}$.push({x0:v,x1:v+g,y,size:f});let x=l.center?Re:Le[l.family],S=[x[0],x[1],x[2],Ge(p)],C=Q[s];C||(C={id:``,kind:`palace-label`,text:``,x:0,y:0,size:0,color:[0,0,0,0],depth:0,weight:.75,revealSpan:1,maxWidthEm:24},Q[s]=C),C.id=`palace-label:`+c.href,C.kind=`palace-label`,C.text=l.label,C.x=v*h,C.y=y*h,C.size=f*h,C.color=S,C.depth=m,C.weight=.95,s++}return Q.length=s,Q}Z.length;var qe=t(`<div class="palace-host fixed inset-0 bg-[#020307] svelte-1dx67o8" role="application" aria-label="VestigeOS Memory Palace. Nine living cognitive organs. Use the Command palette for keyboard navigation."><!></div>`);function Je(e,t){v(t,!0);let ue=()=>y(C,`$page`,de),[de,E]=f(),D=new Set([`/trace`,`/ledger`,`/guard`,`/healing`,`/causal`,`/system`,`/graph`]),fe=[...T(`#F5FFF2`),1],pe=[...T(`#9DFFEB`),1];[...T(`#7DAFA9`)];let O={reasoning:w.bridge,memory:w.recall,immune:se.veto,signal:ce.supersession,temporal:ce.txShadow,system:w.luciferin},k=l(null),A=null,j=null,M=null,N=null,P=null,F=l(null),I=null,L=!1,R=!1,z=``,B=a(()=>{let e=ue().url.searchParams.get(`frame`);if(e===null)return null;let t=Number(e);return Number.isFinite(t)?Math.floor(t):null});h(()=>{ae.set(!1),A&&j?A.removePass(j):j?.dispose(),A&&M?A.removePass(M):M?.dispose(),A&&N?A.removePass(N):N?.dispose(),N=null,j=null,M=null,A=null});async function V(e){try{A=e,R=window.matchMedia(`(prefers-reduced-motion: reduce)`).matches;let t=new Oe(e);j=t,t.setReducedMotion(R),e.addPass(t),t.uploadRegions(Z);let n=new Fe(e,R);N=n,me(),e.addPass(n);let r=new le(e);M=r,await r.init(),e.addPass(r),e.demoClock.reset(),W()}catch(e){console.error(`[palace] Failed to initialize swarm:`,e)}}function H(e){return e.replace(/[—–]/g,`-`).replace(/[‘’]/g,`'`).replace(/[“”]/g,`"`).replace(/…/g,`...`).replace(/[^\x20-\x7E]/g,`?`)}function me(){let e=A?.params[6]||innerWidth,t=A?.params[7]||innerHeight,n=e/Math.max(1,t);n>=1?N?.setPlace(0,.73,Math.min(1.05,1.9/n)):N?.setPlace(0,.8,1.6)}function U(){let e=H(`MEMORY PALACE  -  ${Z.length} LIVING ORGANS  -  CLICK ONE TO ENTER`),t=.026,n=[{id:`palace:sub`,kind:`palace-hud`,text:e,x:-(e.length*.6*t)/2,y:.53,size:t,color:pe,depth:1,weight:.86,revealSpan:28,maxWidthEm:90}];if(i(F)){let e=Ie(i(F)),t=ee(i(F));if(e){let r=[...T(O[e.family]),1],i=L?`ENTERING ${e.label}`:e.label,a=H(t?.purpose??`ENTER THIS ORGAN`),o=a.length>96?`${a.slice(0,95)}...`:a;n.length=0,n.push({id:`palace:focus-label`,kind:`palace-focus`,text:i,x:-(i.length*.6*.034)/2,y:.535,size:.034,color:r,depth:1,weight:1,revealSpan:18},{id:`palace:focus-purpose`,kind:`palace-focus`,text:o,x:-(o.length*.6*.019)/2,y:.48,size:.019,color:fe,depth:1,weight:.8,revealSpan:30,maxWidthEm:100})}}return n}function W(){let e=U(),t=j?Ke(j.getScreenPositions(),{hoveredHref:i(F),dimUnhovered:!!i(F),aspect:(A?.params[6]||0)/Math.max(1,A?.params[7]||1)}):[];M?.setText([...e,...t])}function he(){let e=`${A?.params[6]??0}x${A?.params[7]??0}`;e!==z&&(z=e,me(),W())}function G(e){if(!i(k))return null;let t=i(k).getBoundingClientRect();return t.width<=0||t.height<=0?null:{x:(e.clientX-t.left)/t.width*2-1,y:-((e.clientY-t.top)/t.height*2-1)}}function K(e){if(!i(k)||!A)return;let t=i(k).getBoundingClientRect(),n=Math.max(1e-4,t.width/Math.max(1,t.height)),r={x:e.x*Math.max(n,1),y:e.y/Math.min(n,1)},a=P??r,o={x:a.x+(r.x-a.x)*.35,y:a.y+(r.y-a.y)*.35};P=o,A.setCursorPreNdc(o.x,o.y,o.x-a.x,o.y-a.y)}function ge(e){let t=G(e);if(!t||(K(t),!j||j.isBursting))return;let n=j.pickAt(t.x,t.y),r=n?.href??null;r!==i(F)&&(u(F,r,!0),r&&b(`${x}${r}`).catch(()=>{}),j.setHovered(n?.index??-1),W(),i(k)&&(i(k).style.cursor=r?`pointer`:`crosshair`),ae.set(!!r))}function _e(){j?.isBursting||(I=null,P=null,u(F,null),j?.setHovered(-1),ae.set(!1),A?.setCursorPreNdc(999,999,0,0),W(),i(k)&&(i(k).style.cursor=`crosshair`))}function ve(e){if(e.button!==0||!j||j.isBursting)return;let t=G(e);t&&(I={x:e.clientX,y:e.clientY,href:j.pickAt(t.x,t.y)?.href??null})}function ye(){I=null}function be(e){let t=I;if(I=null,!t||!j||j.isBursting||Math.hypot(e.clientX-t.x,e.clientY-t.y)>9)return;let n=G(e);if(!n)return;let r=j.pickAt(n.x,n.y);r&&r.href===t.href&&(e.clientX,e.clientY,u(F,r.href,!0),j.setHovered(r.index),L=!0,M?.setText([]),N?.leave(),i(k)&&(i(k).style.cursor=`progress`),xe(r))}async function xe(e){let t=`${x}${e.href}`;b(t).catch(()=>{});let n=i(k)?.querySelector(`canvas`);if(R||!n||!i(k)||!j){await S(t);return}j.startBurst(e.href,()=>void q(e,n),.42)||await q(e,n)}async function q(e,t){let n=`${x}${e.href}`;if(!i(k)){await S(n);return}let r=Ie(e.href),a=i(k).getBoundingClientRect(),o={x:0,y:0},s=1.3,c=await re(t,.5,A);if(D.has(e.href)){let{points:t,count:n}=te(c,o,s,a.width/Math.max(1,a.height));ie({points:t,count:n,createdAt:performance.now(),href:e.href})}ne.set({frame:c,focusX:a.left+a.width/2,focusY:a.top+a.height/2,scale:s,color:r?O[r.family]:w.luciferin,startedAt:performance.now(),ready:!1,hold:160,burst:!0}),await S(n)}var J=qe();p(`1dx67o8`,e=>{o(()=>{n.title=`Memory Palace · VestigeOS`})});var Se=c(J);oe(Se,{demo:`recall-path`,seed:`vestige-palace-swarm-v2`,get freezeFrame(){return i(B)},onframe:he,onready:V}),d(J),_(J,e=>u(k,e),()=>i(k)),m(`pointerdown`,J,ve),m(`pointerup`,J,be),r(`pointercancel`,J,ye),m(`pointermove`,J,ge),r(`pointerleave`,J,_e),s(e,J),g(),E()}e([`pointerdown`,`pointerup`,`pointermove`]);export{Je as component};
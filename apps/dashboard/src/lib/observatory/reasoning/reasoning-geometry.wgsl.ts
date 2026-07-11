// Reasoning trace geometry — WGSL. Instanced quads; each instance is one
// primitive (beam / ribbon / nucleus / fringe / scar). The vertex shader expands
// each to a screen quad tightly bounding the primitive; the fragment shader
// evaluates a per-kind SDF and emits an additive HDR color.
//
// Deterministic: motion uses `params.time` only. No reserved words. Additive
// blend, so black = transparent (contributes nothing).

export const REASONING_GEOMETRY_WGSL = /* wgsl */ `
struct Params {
	frame: f32, loopPhase: f32, nodeCount: f32, edgeCount: f32,
	pathCount: f32, pulse: f32, viewportW: f32, viewportH: f32,
	brightness: f32, demoId: f32, time: f32, captureMode: f32,
	liveKind: f32, liveFrame: f32, liveEnergy: f32, projectionDays: f32,
	cursorX: f32, cursorY: f32, cursorVx: f32, cursorVy: f32,
};

// Instance record (12 floats) — mirrors reasoning-geometry-pass INSTANCE_FLOATS.
struct Inst {
	a: vec2f,       // endpoint A / center (NDC)
	b: vec2f,       // endpoint B (== a for points)
	kind: f32,      // 0 beam, 1 ribbon, 2 nucleus, 3 fringe, 4 scar
	thickness: f32, // NDC half-width
	trust: f32,     // 0..1
	sign: f32,      // +1 support / -1 oppose
	energy: f32,    // 0..1
	seed: f32,      // per-instance phase
	extra: f32,     // kind-specific (fringe strength / nucleus confidence)
	pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> insts: array<Inst>;

const TAU: f32 = 6.28318530718;
const QUAD = array<vec2f, 6>(
	vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
	vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) world: vec2f,        // NDC position of this fragment
	@location(1) @interpolate(flat) idx: u32,
};

fn aspect() -> f32 { return max(0.0001, params.viewportW / max(1.0, params.viewportH)); }

// smoothstep-based value-suppressing palette (Correll/Moritz/Heer): low trust →
// neutral grey (never jitter), high trust → vivid. base is the category hue.
fn vsup(base: vec3f, trust: f32) -> vec3f {
	let s = smoothstep(0.0, 1.0, clamp(trust, 0.0, 1.0));
	let neutral = vec3f(0.42, 0.46, 0.52);
	let bright = 0.72 + 0.28 * s;
	return mix(neutral, base, s) * bright;
}

// PRGn-style diverging hue by signed influence: + = green (support), − = purple
// (oppose). |mag| drives saturation; zero-point kept off pure white.
fn prgn(sign: f32, mag: f32) -> vec3f {
	let green = vec3f(0.0, 0.85, 0.45);
	let purple = vec3f(0.55, 0.15, 0.75);
	let base = select(purple, green, sign >= 0.0);
	return mix(vec3f(0.35, 0.37, 0.4), base, clamp(mag, 0.0, 1.0));
}

@vertex
fn vs_geo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let inst = insts[ii];
	let corner = QUAD[vi];
	let asp = aspect();

	// Bounding quad: for line-like kinds (beam/ribbon/scar/fringe) expand along
	// the A→B segment plus thickness on the perpendicular; for point kinds
	// (nucleus) a square of side 2*thickness around A.
	var pos: vec2f;
	let is_point = inst.kind == 2.0;
	if (is_point) {
		let r = inst.thickness + 0.04; // padding for glow/rings
		pos = inst.a + (corner - vec2f(0.5)) * (2.0 * r);
	} else {
		let dir = inst.b - inst.a;
		let len = max(1e-4, length(dir));
		let t_hat = dir / len;
		let n_hat = vec2f(-t_hat.y, t_hat.x);
		let half_w = inst.thickness + 0.03; // padding for AA/glow
		// corner.x runs along the segment (0..1), corner.y across (-1..1)
		let along = corner.x;
		let across = (corner.y - 0.5) * 2.0;
		pos = inst.a + t_hat * (along * len) + n_hat * (across * half_w);
	}

	// aspect-correct so circles stay round (mirror the text pass convention).
	var clip = pos;
	clip.x = clip.x / max(asp, 1.0);
	clip.y = clip.y * min(asp, 1.0);
	out.clip = vec4f(clip, 0.0, 1.0);
	out.world = pos;
	out.idx = ii;
	return out;
}

// ── Per-kind SDF shading. Each returns premultiplied additive HDR rgb. ────────
// FLEET: fill these five. Each is a PURE function of (inst, p, t) → vec3f. Keep
// them deterministic (use params.time via t), additive (return black to skip a
// pixel), and honest (brightness ∝ the real trust/energy/confidence fields).

fn shade_beam(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// BEAM: a bright emissive causal line A→B along y with a flowing pulse.
	let d = sdf_segment(p, inst.a, inst.b);
	let core = smoothstep(inst.thickness, 0.0, d);
	let glow = smoothstep(inst.thickness * 4.0, 0.0, d) * 0.35;
	// flow pulse travelling toward the decision (left→right)
	let along = clamp((p.x - inst.a.x) / max(1e-4, inst.b.x - inst.a.x), 0.0, 1.0);
	let flow = 0.5 + 0.5 * sin(along * 18.0 - t * 3.0);
	let cyan = vec3f(0.0, 0.96, 0.83);
	return cyan * inst.energy * (core * (0.6 + 0.6 * flow) + glow);
}

fn shade_ribbon(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// RIBBON: tapered (wide at nucleus, narrow at source), UV-scroll flow toward
	// the decision, head-bright opacity gradient, PRGn hue by sign.
	let along = clamp(dot(p - inst.a, inst.b - inst.a) / max(1e-6, dot(inst.b - inst.a, inst.b - inst.a)), 0.0, 1.0);
	let d = sdf_segment(p, inst.a, inst.b);
	let taper = inst.thickness * mix(0.35, 1.0, along); // widen toward B (nucleus)
	let core = smoothstep(taper, 0.0, d);
	// flowing dashes travelling A→B (causal direction)
	let flow = 0.5 + 0.5 * sin(along * 26.0 - t * 4.0 - inst.seed);
	let headBright = mix(0.35, 1.0, along); // brighter toward the decision head
	let hue = prgn(inst.sign, inst.trust);
	return hue * inst.energy * core * flow * headBright;
}

fn shade_nucleus(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// NUCLEUS: recommendation core; stability = confidence. Coherent (tight solid
	// rings) when confident, scattered/soft when not. Confidence in inst.extra.
	let c = inst.extra;
	let r = length(p - inst.a);
	let coherence = smoothstep(0.0, 1.0, c);
	let coreR = inst.thickness * (0.5 + 0.3 * c);
	let core = smoothstep(coreR, 0.0, r);
	// 1..4 concentric rings, more + tighter as confidence rises
	let ringCount = 1.0 + floor(c * 3.0);
	let ringPhase = r / max(1e-4, inst.thickness) * ringCount * TAU;
	let ring = pow(max(0.0, sin(ringPhase - t * 1.5)), 6.0) * smoothstep(inst.thickness * 1.6, inst.thickness * 0.4, r) * coherence;
	let hot = vec3f(0.91, 1.0, 0.72);
	return vsup(hot, c) * (core * 1.4 + ring * 0.8);
}

fn shade_fringe(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// FRINGE: two-source interference between contradicting evidence a,b. Scarlet
	// standing wave (cos^2) that breathes; reads as "these two conflict".
	let r1 = distance(p, inst.a);
	let r2 = distance(p, inst.b);
	let lambda = 0.05;
	let phase = TAU * (r1 - r2) / lambda - t * 2.0;
	let fr = cos(phase * 0.5);
	let intensity = fr * fr;
	// confine the fringe to the region BETWEEN the two sources
	let mid = (inst.a + inst.b) * 0.5;
	let span = distance(inst.a, inst.b) * 0.6 + 0.06;
	let mask = smoothstep(span, span * 0.4, distance(p, mid));
	let scarlet = vec3f(0.95, 0.08, 0.14);
	return scarlet * intensity * mask * inst.extra;
}

fn shade_scar(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// SCAR: superseded evidence leaves a dim etched mark at A, with an AMBER
	// transfer filament flowing A→B (into the replacement).
	let dScar = sdf_segment(p, inst.a - vec2f(0.02, 0.0), inst.a + vec2f(0.02, 0.0));
	let scar = smoothstep(0.006, 0.0, dScar) * 0.5;
	let dFil = sdf_segment(p, inst.a, inst.b);
	let along = clamp(dot(p - inst.a, inst.b - inst.a) / max(1e-6, dot(inst.b - inst.a, inst.b - inst.a)), 0.0, 1.0);
	let flow = 0.5 + 0.5 * sin(along * 20.0 - t * 3.5);
	let fil = smoothstep(inst.thickness * 0.6, 0.0, dFil) * flow * (1.0 - along * 0.4);
	let amber = vec3f(1.0, 0.82, 0.4);
	let ash = vec3f(0.45, 0.4, 0.38);
	return ash * scar + amber * fil * 0.7;
}

// segment SDF helper
fn sdf_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
	let pa = p - a;
	let ba = b - a;
	let h = clamp(dot(pa, ba) / max(1e-6, dot(ba, ba)), 0.0, 1.0);
	return length(pa - ba * h);
}

@fragment
fn fs_geo(in: VSOut) -> @location(0) vec4f {
	let inst = insts[in.idx];
	let p = in.world;
	let t = params.time;
	var rgb = vec3f(0.0);
	let k = inst.kind;
	if (k == 0.0) { rgb = shade_beam(inst, p, t); }
	else if (k == 1.0) { rgb = shade_ribbon(inst, p, t); }
	else if (k == 2.0) { rgb = shade_nucleus(inst, p, t); }
	else if (k == 3.0) { rgb = shade_fringe(inst, p, t); }
	else if (k == 4.0) { rgb = shade_scar(inst, p, t); }
	rgb = rgb * params.brightness;
	// additive: alpha carries nothing; premultiplied rgb is the contribution.
	return vec4f(rgb, 1.0);
}
`;

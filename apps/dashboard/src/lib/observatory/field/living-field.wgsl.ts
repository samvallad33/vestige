/**
 * LivingFieldPass WGSL — the reusable "alive field" shader, generalized from the
 * timeline gold standard (timeline-pass.ts). ONE galaxy of glowing, orbiting,
 * additive cells positioned by REAL data. Each organ only writes a ~20-line CPU
 * mapper (real data → LivingCell[]); this shader turns it into a full-bleed,
 * moving, bloomed field that fills ≥35% of the frame.
 *
 * The recipe (why palace/timeline fill 50%+):
 *   1. MANY glowing additive billboards positioned by real data.
 *   2. LIVING ORBITAL DRIFT — deterministic, params.time only (NO RNG/Date.now):
 *        ang = atan2(y,x) + ring_spin(phase) + sin(time*0.6 + phase*TAU)*0.02
 *        r   = radius*(1 + 0.016*sin(time*1.1 + phase*TAU))
 *   3. Additive-blend bloom billboards (black = transparent), pushed to HDR so
 *      the PostChain bloom flares them.
 *   4. Metabolic membrane base coat (separable-blurred density field) so the
 *      void is filled with a breathing plasma, not black.
 *
 * WGSL hygiene (runtime traps, invisible to tsc/build):
 *   - No reserved words: meta/active/filter/sample/texture/binding/common/override.
 *   - Separable blur runs as RENDER passes (T3: no rgba16float write-storage
 *     compute blur on M-series).
 *   - Deterministic: every animated value is a pure function of params.time +
 *     per-cell phase. The CPU pickAt mirrors orbit() exactly (hitPad/orbit trap).
 */

// Shared Params — MUST match types.ts PARAMS_FLOATS / engine frame loop.
const FIELD_COMMON_WGSL = /* wgsl */ `
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

// One living cell. 16 floats = 4 vec4, 16-byte aligned lanes.
struct LivingCellGpu {
	// x,y NDC base position; z billboard radius; w orbit ring radius (== length(xy))
	pos_radius: vec4f,
	// rgb hue; w energy 0..1 (brightness / activation)
	hue_energy: vec4f,
	// x orbit phase (0..1); y flags (bit0 selected, bit1 endangered/scar, bit2 pulse-strong);
	// z secondary metric (retention-ish 0..1); w spin scale
	phase_flags: vec4f,
	// x ring index / group; y satellite twinkle seed; z,w reserved
	extra: vec4f,
};

const TAU: f32 = 6.28318530718;

// Living orbital drift — the thing that makes the field MOVE. Deterministic:
// a pure function of params.time + the cell's own phase. ring_spin gives inner
// (high-phase) cells a faster turn, like a spinning galaxy core.
fn ring_spin(phase01: f32) -> f32 {
	let speed = 0.045 + phase01 * 0.10;
	return params.time * speed;
}

fn orbit(base: vec2f, phase01: f32, spin_scale: f32) -> vec2f {
	let radius = length(base);
	if (radius < 0.0001) { return base; }
	let ang0 = atan2(base.y, base.x);
	let ang = ang0 + ring_spin(phase01) * spin_scale + sin(params.time * 0.6 + phase01 * TAU) * 0.02;
	let rr = radius * (1.0 + 0.016 * sin(params.time * 1.1 + phase01 * TAU));
	return vec2f(cos(ang), sin(ang)) * rr;
}
`;

// Pass 1 — additive splat of every cell into the low-res density field
// (rgba16float): .r density, .g energy-weighted oxygen, .b scar/seam accent.
export const FIELD_SPLAT_WGSL = /* wgsl */ `
${FIELD_COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f, // x phase, y flags, z metric2, w spin
};

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	// breathe the splat radius so density is never a flat print
	let breathe = 1.0 + 0.14 * sin(params.time * 1.5 + phase * TAU);
	let r = c.pos_radius.z * 2.4 * breathe * (1.0 + c.hue_energy.w * 0.9);
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = c.phase_flags;
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let metric2 = clamp(in.info.z, 0.0, 1.0);
	let flags = in.info.y;
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let body = exp(-d * d * 2.7) * (0.32 + energy * 0.95);
	// .r = raw density (fills the void), .g = oxygen (energy-weighted),
	// .b = scar/seam accent for endangered cells
	return vec4f(body, body * (0.4 + metric2 * 0.9), body * scar * 0.7, 1.0);
}
`;

// Separable gaussian blur (render pass, NOT compute — trap T3). Reused H then V.
export const FIELD_BLUR_WGSL = /* wgsl */ `
struct BlurDir { dir: vec2f, _pad: vec2f };
@group(0) @binding(0) var blur_sampler: sampler;
@group(0) @binding(1) var blur_src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur_dir: BlurDir;
const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}
@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, in.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`;

// Pass 3a — the fullscreen membrane base coat: reads the blurred density field
// and paints a breathing blackwater plasma so the void is FILLED (this is what
// pushes fill% past 35 even in the gaps between cells).
export const FIELD_MEMBRANE_WGSL = /* wgsl */ `
${FIELD_COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(field_tex, 0));
	let px = 1.0 / max(dims, vec2f(1.0));
	let f = textureSample(field_tex, field_sampler, in.uv);
	let left = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(px.x, 0.0), 0.0);
	let right = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(px.x, 0.0), 0.0);
	let down = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(0.0, px.y), 0.0);
	let up = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(0.0, px.y), 0.0);
	let density = clamp(f.r, 0.0, 5.0);
	let oxygen = clamp(f.g, 0.0, 5.0);
	let scar = clamp(f.b, 0.0, 3.0);
	let grad = length(vec2f((right.r + right.g) - (left.r + left.g), (up.r + up.g) - (down.r + down.g)));
	let membrane = smoothstep(0.05, 0.62, density) * (1.0 - smoothstep(2.0, 4.0, density));
	let edge = smoothstep(0.008, 0.11, grad) * membrane;
	let breath = 0.72 + 0.55 * params.pulse;
	let blackwater = vec3f(0.006, 0.012, 0.015);
	let amber = vec3f(0.86, 0.42, 0.12);
	let oxygen_col = vec3f(0.66, 1.0, 0.37);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	var color = blackwater * (0.35 + density * 0.14);
	color = color + mix(amber, oxygen_col, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.34 * breath;
	color = color + vec3f(0.91, 1.0, 0.72) * edge * (0.85 + 0.5 * params.pulse);
	color = color + scarlet * scar * (0.6 + 0.4 * params.pulse);
	let vignette = smoothstep(1.02, 0.10, distance(in.uv, vec2f(0.5)));
	return vec4f(color * (0.5 + 0.5 * vignette) * params.brightness, 1.0);
}
`;

// Pass 3b — the sharp bioluminescent cells drawn on top of the membrane, each
// a living organism that twinkles by its own phase and glows to HDR for bloom.
export const FIELD_CELL_WGSL = /* wgsl */ `
${FIELD_COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f,
};

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	let flags = c.phase_flags.y;
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let beat = 1.0 + 0.22 * sin(params.time * 2.3 + c.extra.y * 1.7);
	let r = c.pos_radius.z * (0.85 + c.hue_energy.w * 0.9 + selected * 1.3) * beat;
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = c.phase_flags;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let hue = in.hue_energy.rgb;
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let flags = in.info.y;
	let metric2 = clamp(in.info.z, 0.0, 1.0);
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let phase = in.info.x;
	let twinkle = 0.6 + 0.8 * (0.5 + 0.5 * sin(params.time * 2.1 + phase * 26.0));
	let body = exp(-d * d * 2.7) * (0.55 + energy * 1.7) * twinkle;
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.40, d));
	let scarlet = vec3f(1.0, 0.23, 0.18);
	let ivory = vec3f(0.95, 1.0, 0.86);
	var color = hue * body;
	color = color + ivory * rim * (0.9 + selected * 1.6);
	color = color + scarlet * scar * smoothstep(0.16, 0.0, abs(d - 0.74)) * 1.5;
	return vec4f(color, 1.0);
}
`;

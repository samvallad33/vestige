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

// Per-field options (a small field-local uniform, NOT the shared Params). Carries
// the global intensity + a "reading well" rectangle: the field emits LESS inside
// the well so text renders on a dim, readable substrate. hw<=0 disables the well.
struct FieldOpts {
	intensity: f32,   // 0..1 global field scale (membrane); cells also honor extra.z
	well_x: f32,      // reading-well rect center, NDC x
	well_y: f32,      // reading-well rect center, NDC y
	well_hw: f32,     // half-width NDC (<=0 disables -> factor 1.0)
	well_hh: f32,     // half-height NDC
	well_floor: f32,  // min multiplier inside the well (e.g. 0.10)
	well_soft: f32,   // edge softness NDC (e.g. 0.22)
	_pad: f32,
};

// 1.0 outside the well, ramping down to well_floor inside it (a soft rectangle).
fn reading_well(uv_ndc: vec2f, o: FieldOpts) -> f32 {
	if (o.well_hw <= 0.0) { return 1.0; }
	let dx = max(0.0, abs(uv_ndc.x - o.well_x) - o.well_hw);
	let dy = max(0.0, abs(uv_ndc.y - o.well_y) - o.well_hh);
	let outside = length(vec2f(dx, dy)) / max(o.well_soft, 0.001);
	return mix(o.well_floor, 1.0, clamp(outside, 0.0, 1.0));
}

// Fossil Light keeps the generic organ field inside the graphite / amber /
// jade family. A small amount of an organ's semantic hue survives, but old
// blue-violet source values are grounded before they reach the HDR bloom pass.
fn somatic_tone(hue: vec3f, persistence: f32) -> vec3f {
	let amber = vec3f(0.62, 0.28, 0.10);
	let jade = vec3f(0.28, 0.68, 0.48);
	let physical = mix(amber, jade, smoothstep(0.14, 0.90, persistence));
	let grounded = vec3f(
		clamp(hue.r, 0.0, 1.0),
		max(clamp(hue.g, 0.0, 1.0), clamp(hue.b, 0.0, 1.0) * 0.70),
		min(clamp(hue.b, 0.0, 1.0), clamp(hue.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded, 0.16);
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
	let persistence = clamp(in.info.z, 0.0, 1.0);
	let flags = in.info.y;
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	// Low-res density is intentionally soma-heavy. The branch traces are thin
	// enough that the blur turns them into microscopy-like connective tissue,
	// not another field of soft, identical circles.
	let soma = exp(-d * d * mix(10.5, 5.8, persistence));
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * (5.0 + floor(in.info.x * 3.0)) + in.info.x * TAU));
	let branch_band = smoothstep(0.14, 0.34, d) * (1.0 - smoothstep(0.66, 0.92, d));
	let neurites = pow(branch_wave, 16.0) * branch_band * (0.025 + persistence * 0.070);
	let body = soma * (0.22 + energy * 0.68) + neurites;
	// .r = soma-led density, .g = retained oxygen, .b = suppressed/scar seam.
	return vec4f(body, body * (0.32 + persistence * 0.82), body * scar * 0.50, 1.0);
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
@group(0) @binding(2) var<uniform> fopts: FieldOpts;
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
	let f = textureSample(field_tex, field_sampler, in.uv);
	let density = clamp(f.r, 0.0, 5.0);
	let oxygen = clamp(f.g, 0.0, 5.0);
	let scar = clamp(f.b, 0.0, 3.0);
	let breath = 0.72 + 0.55 * params.pulse;
	// Cold blue-black substrate: the fullscreen plasma is desaturated + dimmed hard
	// so it reads as a deep breathing floor, NOT a neon-green wash over text.
	//
	// The old membrane ALSO drew an edge term = smoothstep of the density
	// gradient, which outlined every blurred cell blob → a dense mesh of ugly
	// overlapping grey-green rings/circles across the whole frame ("what even is
	// this"). That edge term is REMOVED. The base plasma cloud contribution is also
	// cut hard so the field reads as clean glowing cells on near-void, not fog.
	let blackwater = vec3f(0.006, 0.012, 0.015);
	let amber = vec3f(0.70, 0.38, 0.14);
	let oxygen_col = vec3f(0.42, 0.70, 0.40); // desaturated (was neon 0.66,1.0,0.37)
	let scarlet = vec3f(0.85, 0.22, 0.18);
	var color = blackwater * (0.30 + density * 0.06);
	color = color + mix(amber, oxygen_col, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.035 * breath;
	color = color + scarlet * scar * (0.20 + 0.12 * params.pulse);
	let vignette = smoothstep(1.02, 0.10, distance(in.uv, vec2f(0.5)));
	// Reading well: the field emits LESS where text lives, so labels read. Plus the
	// per-page intensity. Deeper vignette floor pushes frame edges toward void.
	let uv_ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
	let well = reading_well(uv_ndc, fopts);
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness * fopts.intensity * well, 1.0);
}
`;

// Pass 3b — the sharp bioluminescent cells drawn on top of the membrane, each
// a living organism that twinkles by its own phase and glows to HDR for bloom.
export const FIELD_CELL_WGSL = /* wgsl */ `
${FIELD_COMMON_WGSL}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;
@group(0) @binding(2) var<uniform> fopts: FieldOpts;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f,
	// x = field intensity (0..1, extra.z), y = twinkle seed (extra.y)
	@location(3) @interpolate(flat) extra: vec4f,
	// the cell's orbited center in NDC, so the reading well is evaluated per cell
	@location(4) @interpolate(flat) center_ndc: vec2f,
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
	out.extra = c.extra;
	out.center_ndc = center;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	// Field intensity (extra.z) dims the WHOLE cell so the field is a backdrop the
	// text reads over. Text organs set this low; visual organs keep it high.
	let intensity = clamp(in.extra.x, 0.05, 1.0);
	let hue = in.hue_energy.rgb;
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let flags = in.info.y;
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let phase = in.info.x;
	// Per-cell persistence is supplied by the real organ mapper (retention where
	// available). projection_days is the signed field clock: only a forward
	// projection is treated as age, so historical scrubs never fake decay.
	let persistence = clamp(in.info.z, 0.0, 1.0);
	let chrono_age = clamp(max(params.projection_days, 0.0) / 120.0, 0.0, 1.0);
	let consolidation = clamp(energy * 0.58 + persistence * 0.42, 0.0, 1.0);
	let depth_scatter = (1.0 - consolidation) * (0.24 + chrono_age * 0.76);
	// Twinkle remains below 5%; it gives the soma metabolic motion without a
	// neon pulse. Storage strength concentrates luminance at the centre.
	let twinkle = 0.95 + 0.05 * (0.5 + 0.5 * sin(params.time * 2.1 + phase * 26.0));
	let soma = exp(-d * d * mix(12.5, 6.5, consolidation)) * (0.08 + consolidation * 0.46) * twinkle;
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * (5.0 + floor(fract(in.extra.y * 0.13) * 3.0)) + phase * TAU));
	let branch_band = smoothstep(0.15, 0.34, d) * (1.0 - smoothstep(0.68, 0.94, d));
	let neurites = pow(branch_wave, 17.0) * branch_band * (0.018 + consolidation * 0.075)
		* (1.0 - depth_scatter * 0.68);
	let scatter = pow(max(1.0 - d, 0.0), 3.8) * (0.010 + depth_scatter * 0.075);
	let tinted = somatic_tone(hue, persistence);
	let soma_tone = mix(tinted, vec3f(0.90, 0.96, 0.84), consolidation * 0.40);
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.40, d));
	let scarlet = vec3f(0.85, 0.22, 0.18);
	let ivory = vec3f(0.90, 0.96, 0.86);
	var color = soma_tone * soma + tinted * neurites + tinted * scatter;
	// The rim is a selected-only instrument mark, never a global decorative glow.
	color = color + ivory * rim * selected * 0.32;
	// Scarred/suppressed cells lose normal emission and leave a small oxide seam.
	let scar_ring = smoothstep(0.70, 0.78, d) * (1.0 - smoothstep(0.80, 0.90, d));
	color = mix(color, color * 0.10 + scarlet * scar_ring * 0.12, scar);
	// selected/scar stay a touch brighter than the dimmed backdrop so meaning survives.
	let keep = max(intensity, (selected + scar) * 0.7);
	let well = reading_well(in.center_ndc, fopts);
	return vec4f(color * keep * well, 1.0);
}
`;

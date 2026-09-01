/**
 * Ambient base coat (Phase 5) — a single metric-bound WGSL field for the
 * "quiet" routes. NOT a hero: one cheap substrate, composited low behind the
 * page. The discipline test still applies — the drive is REAL backend metrics,
 * so the field legibly READS the data (swap the metrics for Math.random() and
 * the motion is legibly wrong):
 *
 *   - mote COUNT tracks the real memory count (density = how full the mind is)
 *   - each mote's VERTICAL drift is its retention bucket: low-retention motes
 *     sink toward the forgetting floor, high-retention ones hold near the top
 *   - endangeredFrac (real actively-forgetting share) storms the whole field:
 *     more endangered memories → more turbulence + more sinking motes
 *   - fractureFrac (real contradiction-pair share) tears rifts across it
 *
 * A route with 129 endangered memories looks visibly different from one with 0.
 * Deterministic per mote (phase from index), so no Math.random in the loop.
 */
export const ambientFieldWGSL = /* wgsl */ `
struct AmbientParams {
	time: f32,          // seconds (advances only when not reduced-motion)
	count: f32,         // active mote count (<= capacity)
	endangered: f32,    // 0..1 real endangered fraction — storm intensity
	fracture: f32,      // 0..1 real contradiction fraction — rift intensity
	due: f32,           // 0..1 real due-for-review fraction — pulse rate
	aspect: f32,        // viewport w/h
	accent_r: f32,      // route accent (rgb, 0..1) — one accent per §guardrail
	accent_g: f32,
	accent_b: f32,
	dpr: f32,
	reduced: f32,       // 1.0 = prefers-reduced-motion (freeze drift, keep field)
	_pad: f32,
};

@group(0) @binding(0) var<uniform> params: AmbientParams;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) seed: vec2<f32>,
	@location(2) @interpolate(flat) tint: vec4<f32>, // rgb + retention
};

// Golden-ratio hash → a deterministic 0..1 per index (no PRNG state).
fn hash1(n: f32) -> f32 {
	return fract(sin(n * 12.9898) * 43758.5453);
}

const CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}
	let fi = f32(ii);
	// Deterministic base position on a golden-angle lattice across the panel.
	let ga = 2.399963; // golden angle
	let rx = hash1(fi + 1.0);
	let ry = hash1(fi + 37.0);
	// Retention bucket for this mote: bias the population so the endangered
	// share of motes are low-retention (they sink). Real fraction drives it.
	let isEndangered = select(0.0, 1.0, ry < params.endangered);
	let retention = mix(0.55 + 0.4 * hash1(fi + 91.0), 0.05 + 0.18 * hash1(fi + 5.0), isEndangered);

	// Vertical home: high retention floats up, low sinks toward the floor.
	let homeY = mix(-0.9, 0.85, retention);
	// Gentle deterministic drift (frozen when reduced-motion): endangered motes
	// jitter more (the field is agitated by how much is being forgotten).
	let t = params.time;
	let sway = select(1.0, 0.0, params.reduced > 0.5);
	let turb = 0.02 + 0.10 * params.endangered;
	let driftX = sway * turb * sin(t * (0.3 + rx) + fi * ga);
	let driftY = sway * (0.015 + 0.05 * isEndangered) * sin(t * (0.5 + ry) + fi);
	let baseX = (rx * 2.0 - 1.0) * 0.98 + driftX;
	let baseY = homeY + driftY;

	// A rift: the fracture metric opens a horizontal tear that pushes motes apart.
	let rift = params.fracture * 0.25 * sin(baseX * 3.14159 + t * 0.2);
	let center = vec2<f32>(baseX, baseY + rift);

	// Mote size: small; endangered ones a touch larger + dimmer (last flare).
	let size = (0.010 + 0.014 * retention) * (1.0 + 0.4 * isEndangered);
	let corner = CORNERS[vi];
	out.clip = vec4<f32>(center.x + corner.x * size, center.y + corner.y * size * params.aspect, 0.0, 1.0);
	out.uv = corner;
	out.seed = vec2<f32>(rx, ry);
	out.tint = vec4<f32>(params.accent_r, params.accent_g, params.accent_b, retention);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = in.tint.a;
	// Soft mote: hot core + feathered halo, brightness scales with retention so
	// the endangered (dim) vs healthy (bright) split is LEGIBLE at a glance.
	let core = smoothstep(0.5, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 2.0);
	// Due-for-review adds a slow global pulse so an overdue route breathes.
	let pulse = 0.85 + 0.15 * sin(params.time * (0.6 + params.due));
	let intensity = (core * 0.9 + halo * 0.35) * (0.25 + 0.75 * retention) * pulse;
	// Endangered motes shift toward a warmer, dimmer ember; healthy toward accent.
	let ember = vec3<f32>(0.62, 0.32, 0.22);
	let col = mix(ember, in.tint.rgb, smoothstep(0.2, 0.6, retention));
	return vec4<f32>(col * intensity, intensity * 0.9);
}
`;

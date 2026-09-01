/**
 * Cognitive Observatory — node billboard shader (WGSL).
 *
 * Instanced soft-glow sprites read straight from the NodeState storage buffer
 * (compute-boids render pattern, spec §1). Additive blending onto the void so
 * overlapping memories build light instead of z-fighting.
 *
 * Visual DNA §7: base hue = FSRS state color (meaning at rest); the global
 * breath `pulse` modulates halo energy so the field is alive even when idle.
 * Layout contracts: Params = types.PARAMS_FLOATS, Node = 4×vec4f (types.ts).
 */
export const renderNodesWGSL = /* wgsl */ `
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
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

// Fossil-light activation band. Recall is not a rainbow screensaver: a living
// memory travels graphite → amber → jade → chalk, with no violet/purple energy
// leaking back into the stage.
fn spectral(w_in: f32) -> vec3<f32> {
	let w = fract(w_in);
	// var, not let: WGSL only allows dynamic indexing through a reference.
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.48, 0.22, 0.08), // fossil amber
		vec3<f32>(0.82, 0.58, 0.24), // warmed phosphor
		vec3<f32>(0.30, 0.74, 0.53), // retained jade
		vec3<f32>(0.88, 0.94, 0.82)  // chalk ignition
	);
	let f = w * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	let a = stops[i];
	let b = stops[(i + 1u) % 4u];
	return mix(a, b, frac);
}

// Incoming semantic colors predate Fossil Light and include blue/violet
// states. Keep a trace of that information, but chromatically ground it so the
// field cannot fall back into the old purple-neon visual language.
fn fossil_tone(source: vec3<f32>, retention: f32) -> vec3<f32> {
	let amber = vec3<f32>(0.66, 0.30, 0.10);
	let jade = vec3<f32>(0.30, 0.74, 0.52);
	let retained = smoothstep(0.16, 0.92, clamp(retention, 0.0, 1.0));
	let physical = mix(amber, jade, retained);
	let grounded_source = vec3<f32>(
		clamp(source.r, 0.0, 1.0),
		max(clamp(source.g, 0.0, 1.0), clamp(source.b, 0.0, 1.0) * 0.70),
		min(clamp(source.b, 0.0, 1.0), clamp(source.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded_source, 0.14);
}

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	// Per-instance constants: flat interpolation guarantees the flag bit
	// field survives the raster stage bit-exact (no barycentric rounding).
	@location(1) @interpolate(flat) color: vec3<f32>,
	// x retention, y flags (bit field as f32), z recall intensity, w radius
	@location(2) @interpolate(flat) misc: vec4<f32>,
	// Per-demo choreography lanes (demo.y, demo.z, demo.w), gated by demo_id:
	// rescue (2) searchlight/wave/shock, forgetting-horizon (3) fade-and-fall,
	// firewall (4) flare-membrane/shock. Each demo's choreography pass is the
	// ONLY writer of its lanes, and every gated term below is an exact no-op
	// when its lane is 0.0 — other demos stay pixel-identical.
	@location(3) @interpolate(flat) demo_yzw: vec3<f32>,
};

// Quad corners for two triangles (vertex_index 0..5).
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
		// degenerate — clipped away
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let node = nodes[ii];
	let corner = CORNERS[vi];

	// Breath: halo geometry swells ~6% on the global pulse (§7.2), and the
	// center memory breathes a touch deeper — a heartbeat, not a strobe.
	let flags = u32(node.color_flags.w);
	let is_center = (flags & 1u) != 0u;
	var breath = 1.0 + 0.06 * params.pulse;
	if (is_center) {
		breath = 1.0 + 0.12 * params.pulse;
	}

	// Sprite spans ~3.2× the core radius so the halo has room to feather out.
	// Recall activation swells the sprite — the wavefront physically blooms.
	// Per-demo choreography lanes swell it too, gated by demo_id so each
	// demo's grammar can never leak into another (lanes are 0.0 elsewhere,
	// and the gate makes the no-op structural, not just numerical).
	let recall = node.demo.x;
	let dy = node.demo.y;
	let dz = node.demo.z;
	let dw = node.demo.w;
	// The firewall grammar fires for the deterministic demo (demo_id==4) AND for
	// a LIVE contradiction/suppression event (live_kind==1). Both write the same
	// demo lanes (firewall.wgsl), so the visual reads identically either way.
	let firewall_active = params.demo_id == 4.0 || params.live_kind == 1.0;
	var lane_swell = 0.0;
	if (params.demo_id == 2.0) {
		// salience-rescue: searchlight pop, wave shiver, shock bloom.
		lane_swell = 0.5 * dy + 0.25 * dz + 0.9 * dw;
	} else if (firewall_active) {
		// firewall: intrusion flare pop (band (0..1]), membrane presence
		// (band [2.6..2.9] via the range gate), crimson shock bloom.
		lane_swell = 0.35 * min(dy, 1.0) + 0.3 * smoothstep(1.5, 2.2, dy) + 0.55 * dw;
	}
	// forgetting-horizon (demo 3): VISUAL displacement toward the horizon —
	// down and away from the field axis, ~40.5 units at dz = 1 — plus a
	// shrink. pos_radius is NEVER written (the force sim owns positions);
	// drift is pure of demo.z, so ?frame=N capture stays exact. CPU mirror:
	// forgetting-plan.ts horizonDrift().
	var horizon_scale = 1.0;
	var drift = vec3<f32>(0.0);
	if (params.demo_id == 3.0) {
		let dzc = clamp(dz, 0.0, 1.0);
		horizon_scale = 1.0 - 0.35 * dzc;
		if (dz > 0.0) {
			let p = node.pos_radius.xyz;
			let r_xz = max(length(p.xz), 0.001);
			let away = vec3<f32>(p.x / r_xz, 0.0, p.z / r_xz);
			drift = dzc * (vec3<f32>(0.0, -34.0, 0.0) + away * 22.0);
		}
	}
	// FOSSIL LIGHT existence mask — live retention of exactly 0 means "not yet
	// born at the scrubbed instant" (fsrs.ts reserves 0.0 as the unborn
	// sentinel; existing memories floor at 0.001). Collapsing the sprite to
	// zero size pops the memory out of the field when the chrono crosses its
	// birthday — cheaper and cleaner than a fragment discard.
	let exists = step(0.0005, node.vel_retention.w);
	let half_size = node.pos_radius.w * 3.2 * breath * (1.0 + 0.9 * recall)
		* (1.0 + lane_swell) * horizon_scale * exists;
	let world = node.pos_radius.xyz + drift
		+ camera.right.xyz * corner.x * half_size
		+ camera.up.xyz * corner.y * half_size;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;
	out.color = node.color_flags.rgb;
	out.misc = vec4<f32>(node.vel_retention.w, node.color_flags.w, node.demo.x, node.pos_radius.w);
	out.demo_yzw = vec3<f32>(dy, dz, dw);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let retention = in.misc.x;
	let flags = u32(in.misc.y);
	let suppressed = (flags & 2u) != 0u;
	let is_center = (flags & 1u) != 0u;

	// SOMATIC PHOTOMETRY — retention is consolidation, not generic bloom.
	// High-retention memories form a concentrated bright soma; their neurites
	// remain deliberately dim. A forward Chrono projection makes weak memories
	// scatter into the field instead of multiplying the whole halo's brightness.
	let consolidated = smoothstep(0.04, 0.96, clamp(retention, 0.0, 1.0));
	let forward_age = clamp(max(params.projection_days, 0.0) / 120.0, 0.0, 1.0);
	let depth_scatter = (1.0 - consolidated) * (0.28 + 0.72 * forward_age);
	let soma = exp(-d * d * mix(34.0, 17.0, consolidated));
	let halo = pow(max(1.0 - d, 0.0), 3.4);
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_count = 5.0 + floor(fract(in.misc.w * 0.173) * 3.0);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * branch_count + in.misc.w * 1.91));
	let branch_gate = pow(branch_wave, 18.0);
	let branch_band = smoothstep(0.16, 0.38, d) * (1.0 - smoothstep(0.72, 0.96, d));
	let neurites = branch_gate * branch_band * (0.035 + 0.12 * consolidated)
		* (1.0 - depth_scatter * 0.72);
	let scattered_tissue = halo * (0.015 + 0.11 * depth_scatter) * (0.82 + 0.18 * params.pulse);
	let tone = fossil_tone(in.color, retention);
	let soma_tone = mix(tone, vec3<f32>(0.90, 0.96, 0.84), consolidated * 0.48);
	var color = soma_tone * soma * (0.34 + 0.98 * consolidated)
		+ tone * neurites
		+ tone * scattered_tissue;

	// The anchor can be legible without becoming a fake sun. A suppressed memory
	// is intentionally a cold, near-dark scar: in an additive pass it cannot
	// subtract light yet, but it no longer emits the field's normal luminance.
	if (is_center) {
		color = color * 1.32;
	}
	if (suppressed) {
		let scar_ring = smoothstep(0.66, 0.78, d) * (1.0 - smoothstep(0.80, 0.92, d));
		color = color * 0.055 + vec3<f32>(0.22, 0.10, 0.045) * scar_ring * 0.10;
	}

	// Forgetting-horizon (demo 3): multiplicative dim toward near-black as
	// demo.z rises. Floor 0.06 — never fully gone, always retrievable. Sits
	// BEFORE the recall block so a rescued memory's ignition burns through
	// the fade. demo_yzw.y carries demo.z (vec3 = y/z/w lanes).
	if (params.demo_id == 3.0) {
		color = color * mix(1.0, 0.06, clamp(in.demo_yzw.y, 0.0, 1.0));
	}

	// Recall activation — GCaMP calcium-imaging emission. The intensity lane
	// (simulate.wgsl recall_sim) is now a real biexponential calcium transient;
	// the COLOR here matches what you see under a two-photon scope: a green
	// fluorescence core, a lingering yellow-green ember through the slow decay
	// tail, and a white-hot pinpoint only at the instant of the spike. The
	// traveling wavefront still rides the spectral band so a multi-hop causal
	// recall reads as a wave, but each node that fires flashes like a neuron.
	// jGCaMP green ~ (0.16, 1.0, 0.42); saturated re-fires (recall > 1) push
	// toward white-hot the way an over-driven indicator clips.
	let recall = in.misc.z;
	if (recall > 0.001) {
		let hot = clamp(recall, 0.0, 1.0);                    // spike peak → 1
		let ember = clamp(recall, 0.0, 1.0);                  // afterglow presence
		// GCaMP fluorophore green, warming to yellow-green as the transient
		// saturates (nonlinear summation on rapid re-fire).
		let gcamp = mix(vec3<f32>(0.16, 1.00, 0.42), vec3<f32>(0.62, 1.00, 0.30), clamp(recall - 0.6, 0.0, 1.0));
		// The spectral band survives as the traveling-wave shimmer, but dialed
		// under the calcium green so the biology reads first.
		let band = spectral(0.1 + params.loop_phase + d * 0.35);
		let activation = (gcamp * (soma * 1.85 + halo * 1.05) + band * 0.28 * halo) * ember;
		// White-hot pinpoint ONLY at the fast spike (soma core × hot), so the
		// ignition punches and the ember stays green.
		let flash = vec3<f32>(1.0, 1.0, 0.94) * soma * hot * 0.6;
		color = color + activation + flash;
	}

	// Per-demo choreography lanes — gated by demo_id AND on nonzero values so
	// every other demo is pixel-unchanged (each demo's pass is the only
	// writer of its lanes, and lanes are exactly 0.0 everywhere else).
	if (params.demo_id == 2.0) {
		if (in.demo_yzw.x > 0.001) {
			// Searchlight: cold clinical white — unmistakably NOT the spectral grammar.
			color = color + vec3<f32>(0.82, 0.90, 1.00) * in.demo_yzw.x * (soma * 1.8 + halo * 0.7);
		}
		if (in.demo_yzw.y > 0.001) {
			// Interrogation shimmer: icy spectral strobe as the wave scrubs the past.
			color = color + spectral(0.55 + params.loop_phase) * in.demo_yzw.y * (soma * 0.9 + halo * 0.5)
				+ vec3<f32>(1.0) * soma * in.demo_yzw.y * 0.2;
		}
		if (in.demo_yzw.z > 0.001) {
			// Detonation: crimson blaze + warm-white pinpoint.
			color = color + vec3<f32>(1.00, 0.16, 0.10) * in.demo_yzw.z * (soma * 1.9 + halo * 1.1)
				+ vec3<f32>(1.0, 0.85, 0.8) * soma * in.demo_yzw.z * 0.4;
		}
	} else if (params.demo_id == 4.0 || params.live_kind == 1.0) {
		// firewall: demo.y carries TWO value bands — intrusion flare (0..1]
		// and membrane [2.6..2.9] — separated by range, one lane. demo.w is
		// the crimson shock rim / sever blink. (demo_yzw = y/z/w lanes.)
		let fy = in.demo_yzw.x;
		let fw = in.demo_yzw.z;
		// Intrusion flare: sickly green-white — a hue deliberately OUTSIDE
		// both the FSRS palette and the thin-film band. Continuous across the
		// band boundary (fades out as fy climbs toward the membrane band).
		let flare = min(fy, 1.0) * (1.0 - smoothstep(1.0, 1.8, fy));
		if (flare > 0.001) {
			color = color + vec3<f32>(0.62, 1.00, 0.55) * flare * (soma * 1.7 + halo * 0.9)
				+ vec3<f32>(0.90, 1.00, 0.85) * soma * flare * 0.5;
		}
		// Membrane: quarantine ring at d ≈ 0.75 with fresnel-ish falloff —
		// green body, crimson edge. exp(-q·q) squares by multiplication and
		// the pow base is clamped ≥ 0 (no pow(neg) anywhere).
		let mw = smoothstep(1.5, 2.2, fy);
		if (mw > 0.001) {
			let q = (d - 0.75) * 9.0;
			let ring = exp(-q * q);
			let fresnel = pow(clamp(d / 0.75, 0.0, 1.0), 3.0);
			let ring_col = mix(vec3<f32>(0.55, 1.00, 0.60), vec3<f32>(1.00, 0.20, 0.16),
				smoothstep(0.72, 0.92, d));
			color = color + ring_col * ring * fresnel * mw * 1.4;
		}
		// Shockwave: crimson RIM as the front passes (a rim, not a blaze).
		if (fw > 0.001) {
			let rim = smoothstep(0.45, 0.8, d) * (1.0 - smoothstep(0.85, 1.0, d));
			color = color + vec3<f32>(1.00, 0.14, 0.10) * rim * fw * 1.5
				+ vec3<f32>(1.00, 0.60, 0.50) * soma * fw * 0.15;
		}
	}

	// Additive target (src=one, dst=one): alpha is ignored, light accumulates.
	return vec4<f32>(color * params.brightness, 1.0);
}
`;

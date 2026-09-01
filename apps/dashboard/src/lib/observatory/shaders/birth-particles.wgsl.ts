/**
 * Cognitive Observatory — birth particle compute pass (Moment B, Task B3).
 *
 * One invocation per particle (workgroup 64, dispatch ceil(N/64)). Each
 * particle converges from its start position toward the target over the
 * 720-frame loop:
 *
 *   frames 0–239  : latent trace condensing (slow drift)
 *   frames 240–329: engram coalescence (accelerated convergence)
 *   frames 330–359: memory ignition (flash — handled in render)
 *   frames 360–509: associations engrave (hold at target)
 *   frames 510–719: stabilization (hold, then reset)
 *
 * All time terms are integer-cycles per 720 frames so the loop seam is
 * invisible. Capture mode (params.capture_mode == 1.0) skips integration.
 *
 * Particle layout (16 floats / 64 bytes per particle):
 *   start_life  : xyz start position, w phase offset (stagger)
 *   target_size : xyz target position, w base size (1.0 + rng * 1.8)
 *   color_phase : rgb base color, w phase offset
 *   state       : xyz current position (shader writes), w alpha
 */
export const birthParticlesWGSL = /* wgsl */ `
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

// 16 floats / 64 bytes per particle (matches birth-plan.ts layout).
struct BirthParticle {
	start_life: vec4<f32>,
	target_size: vec4<f32>,
	color_phase: vec4<f32>,
	state: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<BirthParticle>;

@compute @workgroup_size(64)
fn birth_compute(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= arrayLength(&particles)) {
		return;
	}

	// Capture mode (params.capture_mode == 1.0): skip physics integration.
	// The storage-buffer state stays frozen at initial upload values.
	if (params.capture_mode == 1.0) {
		return;
	}

	var particle = particles[i];
	let frame = params.frame;
	let phase = params.loop_phase;

	// --- Convergence choreography (integer cycles per 720-frame loop) ---

	// Phase offset (stagger) from start_life.w: 0..1 → delays convergence.
	let stagger = particle.start_life.w;

	// Effective frame: staggered loop frame (wraps at 720).
	let effFrame = fract(phase + stagger * 0.15) * 720.0;

	// --- Phase 1: latent trace condensing (frames 0–239) ---
	// Slow drift toward target.
	var t: f32;
	if (effFrame < 240.0) {
		// Smooth ease-in: 0 → 1 over 240 frames.
		t = effFrame / 240.0;
		t = t * t * (3.0 - 2.0 * t); // smoothstep
	}
	// --- Phase 2: engram coalescence (frames 240–329) ---
	// Accelerated convergence to target.
	else if (effFrame < 330.0) {
		let localFrame = effFrame - 240.0;
		// 0 → 1 over 90 frames, with slight overshoot then settle.
		t = localFrame / 90.0;
		t = t * t * (3.0 - 2.0 * t);
		// Add a small overshoot (1.05) then settle back to 1.0.
		t = 1.0 - 0.05 * (1.0 - t);
	}
	// --- Phase 3: memory ignition (frames 330–359) ---
	// Hold at target (flash handled in render).
	else if (effFrame < 360.0) {
		t = 1.0;
	}
	// --- Phase 4: associations engrave (frames 360–509) ---
	// Hold at target.
	else if (effFrame < 510.0) {
		t = 1.0;
	}
	// --- Phase 5: stabilization (frames 510–719) ---
	// Hold at target, then fade alpha for reset.
	else {
		let localFrame = effFrame - 510.0;
		// Fade alpha to 0 for seamless reset at frame 0.
		t = 1.0;
		particle.state.w = 1.0 - smoothstep(0.0, 150.0, localFrame);
	}

	// Interpolate from start to target.
	let startPos = particle.start_life.xyz;
	let targetPos = particle.target_size.xyz;
	// (WGSL forbids swizzle stores - reconstruct, preserving alpha in .w)
	particle.state = vec4<f32>(mix(startPos, targetPos, t), particle.state.w);

	// Alpha: particles fade in during convergence, fade out during reset.
	let fadeIn = smoothstep(0.0, 60.0, effFrame);
	particle.state.w = max(particle.state.w, fadeIn * 0.8);

	particles[i] = particle;
}
`;

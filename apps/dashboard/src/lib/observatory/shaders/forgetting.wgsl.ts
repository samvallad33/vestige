/**
 * Cognitive Observatory — forgetting-horizon choreography compute pass (WGSL).
 *
 * One invocation per node. Decodes the packed horizon word (forgetting-plan.ts:
 * bits 0-7 rank, bit 8 isDrifting, bit 9 isRescued, bits 10-11 rescue slot k)
 * and writes ALL FOUR NodeState demo lanes as PURE functions of
 * (params.frame, packed word):
 *
 *   demo.x  rescue ignition (existing thin-film recall response) on the 3 rescued
 *   demo.y  ALWAYS 0.0 — the rescue searchlight grammar can never fire here
 *   demo.z  horizon fade-and-fall (vertex drift + shrink, fragment dim to ~6%)
 *   demo.w  ALWAYS 0.0 — the shock grammar can never fire here
 *
 * This pass MUST encode AFTER recall_sim (simulate.wgsl) in the same encoder:
 * recall_sim rewrites demo.x every frame from the path buffer (afterglow decays
 * bf+40..bf+200 — the k=2 rescue ribbon at bf=438 would leave residual demo.x
 * near the seam). Overwriting all four lanes here is simultaneously the
 * choreography, the loop-seam guarantee, and free ?frame=N capture support
 * (stateless: same frame in → same lanes out).
 *
 * Every term has attack a0 ≥ 90 and is multiplied by the master release
 * 1−smoothstep(660, 712, f) ⇒ exact 0.0 at frames 0 and 719. NO sines in this
 * moment. The CPU mirror (forgetting-plan.ts forgettingEnvelopes) is
 * machine-checked by the seam-zero test — keep both in lockstep.
 *
 * Bind group = EXACTLY the 3 declared bindings (params, nodes, horizon).
 */

export const forgettingWGSL = /* wgsl */ `
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
	live_start_frame: f32,
	live_energy: f32,
	projection_days: f32,
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// 1 u32/node: bits 0-7 rank, 8 isDrifting, 9 isRescued, 10-11 rescue slot k
// (forgetting-plan.ts packing). Non-drifting nodes are exactly 0.
@group(0) @binding(2) var<storage, read> horizon: array<u32>;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

@compute @workgroup_size(64)
fn forgetting_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&horizon)) {
		return;
	}
	// Belt-and-braces atop the TS gate: forgetting-horizon is demo index 3.
	if (params.demo_id != 3.0) {
		return;
	}

	let packed = horizon[i];
	let is_drifting = (packed & 0x100u) != 0u;
	let is_rescued = (packed & 0x200u) != 0u;
	let rank01 = f32(packed & 0xffu) / 255.0;
	let k = f32((packed >> 10u) & 0x3u);

	let f = params.frame;
	// Master release: every lane is exactly 0.0 by frame 712 — the seam wall.
	let master = 1.0 - smoothstep(660.0, 712.0, f);

	var dx = 0.0;
	var dz = 0.0;

	if (is_drifting) {
		let onset = 90.0 + 42.0 * rank01;
		// Phase 1 — the drift: dim + fall to the 0.55 plateau, retention-staggered.
		let phase1 = 0.55 * smoothstep(onset, onset + 210.0, f);
		if (is_rescued) {
			let rk = 318.0 + 60.0 * k;
			// Snap-back begins 22 frames before the recall ribbon lands at rk.
			dz = master * phase1 * (1.0 - smoothstep(rk - 22.0, rk + 6.0, f));
			// Ignition rides the EXISTING recall response (render-nodes.wgsl):
			// spectral() thin-film band + white-hot core + sprite swell for free.
			dx = master * env(f, rk - 26.0, rk, rk + 60.0, rk + 130.0);
		} else {
			// Phase 2 — the sink: to exactly 1.0 over 640..660 (the ~6% floor era).
			let phase2 = 0.45 * smoothstep(480.0 + 24.0 * rank01, 640.0, f);
			dz = master * (phase1 + phase2);
		}
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them). demo.y and
	// demo.w are hard 0.0: the rescue/firewall grammars can never fire here.
	var node = nodes[i];
	node.demo = vec4<f32>(dx, 0.0, dz, 0.0);
	nodes[i] = node;
}
`;

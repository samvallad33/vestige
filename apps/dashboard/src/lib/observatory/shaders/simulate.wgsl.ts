/**
 * Cognitive Observatory — recall-path + force simulation compute pass (Increment 7).
 *
 * One invocation per node (workgroup 64, dispatch ceil(N/64) — the canonical
 * compute-boids pattern, spec §1). Each node:
 *
 *   1. Scans the small PathStep buffer (≤ 8 beats) and computes its recall
 *      activation envelope for this loop frame (arrival / departure).
 *   2. Computes deterministic GPU force settle: edge springs, O(N²)
 *      repulsion (≤ 500 nodes), gentle centering, damping, velocity cap,
 *      and position integration.
 *
 * Writes NodeState.demo.x (recall intensity). Deterministic: everything is a
 * pure function of (frame, path buffer, node state) — no randomness, no wall
 * clock. Center node (isCenter flag) never moves.
 *
 * PASS ORDER (salience-rescue): rescue_choreo (shaders/rescue.wgsl.ts) MUST
 * encode AFTER this pass in the same encoder — it overwrites all four demo
 * lanes so the arc-afterglow demo.x written here (decays bf+40..bf+200) never
 * crosses the 719→0 loop seam. The route guarantees construction order:
 * NodeRenderer first, RescueRenderer second.
 */
export const simulateWGSL = /* wgsl */ `
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

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// x source index, y target index, z beat frame, w kind (0 recall, 1 backward)
@group(0) @binding(2) var<storage, read> path: array<vec4<u32>>;
// x source index, y target node index (Increment 7: force simulation edges)
@group(0) @binding(3) var<storage, read> edges: array<vec2<u32>>;
// v2.3 living field — per-node LIVE retrievability (real FSRS curve, recomputed
// on the CPU by the LiveBridge). One f32 per node. read to overwrite
// vel_retention.w so render-nodes dims each memory on its true forgetting curve.
@group(0) @binding(4) var<storage, read> live_retention: array<f32>;

// --- Force-simulation helpers (Increment 7) ---

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
	let l = length(v);
	if (l < 0.0001) { return vec3<f32>(0.0); }
	return v / l;
}

fn clamp_len(v: vec3<f32>, hi: f32) -> vec3<f32> {
	let l = length(v);
	if (l > hi && l > 0.0001) { return v * (hi / l); }
	return v;
}

@compute @workgroup_size(64)
fn recall_sim(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}

	let frame = params.frame;
	var intensity = 0.0;

	var node = nodes[i];
	let flags = u32(node.color_flags.w);
	let is_center = (flags & 1u) != 0u;

	// --- GCaMP calcium-transient recall kinetics ---------------------------
	// A retrieved memory does NOT ease-out linearly; it fires like a neuron
	// under two-photon calcium imaging. Each recall beat is one calcium
	// transient with a biexponential envelope: near-instant rise, MUCH slower
	// decay (jGCaMP8/GCaMP6 kinetics, Nature 2023 s41586-023-05828-9). Empirical
	// asymmetry is ~1:30 rise:decay; at the observatory's 60fps loop clock that
	// is a ~3-frame time-to-peak and a ~90-frame decay tail. tau_decay is
	// MODULATED BY REAL FSRS RETENTION (vel_retention.w): a weak, decaying
	// memory's ember fades fast; a strongly-retained one glows on. The
	// discipline test holds — swap the retention for noise and the afterglow
	// lengths scramble.
	let ret = clamp(node.vel_retention.w, 0.0, 1.0);
	let tau_rise = 3.0;                        // fast fluorescence spike (~50ms)
	let tau_decay = 55.0 + 70.0 * ret;         // 55..125 frames — retention holds the glow
	// SEAM FADE — the GCaMP tail decays slowly (tau_decay up to 125f), and the
	// last story beat lands at ~bf=480, so at the last loop frame 719 a hot
	// node still glows ~0.15 and would snap to 0 at frame 0 (dt goes negative):
	// a visible pop every 12s. Force the whole recall envelope to zero over the
	// final ~30 frames so the loop is seamless by construction (restores the old
	// smoothstep guarantee that the calcium version broke).
	let seam = 1.0 - smoothstep(688.0, 718.0, frame);
	let steps = u32(params.path_count);
	for (var s = 0u; s < steps; s = s + 1u) {
		let step = path[s];
		let bf = f32(step.z);

		if (step.y == i) {
			// Arrival transient: analytic biexponential (calcium indicator ODE),
			// not a tween. Clamp dt>=0 BEFORE the exponentials so the pre-beat
			// case is a cheap, finite 0.0 (select() evaluates both arms; the old
			// discarded true-arm computed exp(+large)=+Inf for future beats).
			let dt = max(frame - bf, 0.0);
			let g = (1.0 - exp(-dt / tau_rise)) * exp(-dt / tau_decay);
			// NONLINEAR SUMMATION: rapid re-fires stack supralinearly (a hot,
			// over-recalled memory saturates like an over-driven indicator)
			// instead of the old max(). Saturating add keeps it bounded/HDR-safe.
			intensity = intensity + g * (1.0 - 0.55 * intensity);
		}
		if (step.x == i && step.x != step.y) {
			// Departure: the source shimmers briefly as the wave leaves it —
			// a small pre-transient before its own arrival glow.
			let dt = max(frame - (bf - 32.0), 0.0);
			let g = (1.0 - exp(-dt / tau_rise)) * exp(-dt / (tau_decay * 0.45));
			intensity = intensity + g * 0.4 * (1.0 - 0.55 * intensity);
		}
	}
	intensity = clamp(intensity, 0.0, 1.35) * seam;

	// Write recall intensity (existing behavior preserved).
	node.demo.x = intensity;

	// v2.3 LIVE FSRS decay — overwrite retention with the real forgetting-curve
	// value the LiveBridge computed for this node on the CPU. This is the #1
	// moat: render-nodes already dims by vel_retention.w (line ~183), so writing
	// the true retrievability here makes every memory visibly forget on its own
	// curve. Guarded so a graph with no live-decay data (all zeros) keeps its
	// static snapshot instead of collapsing to black.
	if (i < arrayLength(&live_retention)) {
		let lr = live_retention[i];
		// FOSSIL LIGHT: lr == 0.0 is the honest "not yet born at the scrubbed
		// instant" sentinel and MUST propagate so the render mask can pop the
		// memory out of existence. Living memories are floored at 0.001 by the
		// CPU (fsrs.ts/node-renderer.ts), so gating on >= 0.0 never blanks a
		// real field; the old strictly-positive guard predates the floor and
		// blocked unbirth.
		if (lr >= 0.0) {
			node.vel_retention = vec4<f32>(node.vel_retention.xyz, lr);
		}
	}

	// --- Increment 7: force simulation ---

	// Capture mode (params.capture_mode == 1.0): skip physics integration
	// entirely. The storage-buffer state stays frozen at initial upload
	// values, making same URL + frame → identical pixels (spec §4 Inc 9).
	if (params.capture_mode == 0.0) {
		// 7B: center anchor — center node never moves.
		// (WGSL forbids swizzle stores — reconstruct the vec4, preserving .w.)
		if (is_center) {
			node.pos_radius = vec4<f32>(0.0, 0.0, 0.0, node.pos_radius.w);
			node.vel_retention = vec4<f32>(0.0, 0.0, 0.0, node.vel_retention.w);
			nodes[i] = node;
			return;
		}

		let pos = node.pos_radius.xyz;
		var force = vec3<f32>(0.0);

		// 7C: edge springs — scan existing edgeBuffer, no atomics.
		for (var e = 0u; e < u32(params.edge_count); e = e + 1u) {
			let edge = edges[e];
			var other_idx = 0xffffffffu;
			if (edge.x == i) { other_idx = edge.y; }
			if (edge.y == i) { other_idx = edge.x; }
			if (other_idx != 0xffffffffu && other_idx < u32(params.node_count)) {
				let other = nodes[other_idx].pos_radius.xyz;
				let delta = other - pos;
				let dist = max(length(delta), 0.001);
				let dir = delta / dist;
				let stretch = dist - 34.0;
				force = force + dir * stretch * 0.00055;
			}
		}

		// 7D: soft repulsion (only ≤ 500 nodes for performance).
		if (u32(params.node_count) <= 500u) {
			for (var j = 0u; j < u32(params.node_count); j = j + 1u) {
				if (j == i) { continue; }
				let other = nodes[j].pos_radius.xyz;
				let delta = pos - other;
				let d2 = max(dot(delta, delta), 9.0);
				force = force + safe_normalize(delta) * (7.5 / d2);
			}
		}

		// Gentle centering: keeps the field in frame without crushing it.
		force = force + (-pos) * 0.0008;

		// v2.3 DREAM STORM — while the real dream pipeline streams (live_kind ==
		// 2 == LIVE_KIND.dreamStorm), the field enters a metabolic consolidation
		// storm: damping loosens (springs overshoot, clusters slosh together as
		// new ConnectionDiscovered edges are appended) and a deterministic
		// turbulence rides live_energy. Pure of node index + live_frame, so no
		// wall clock — the storm is a function of the real event envelope. At
		// rest (energy 0) both terms vanish → the field is byte-identical.
		var damping = 0.88;
		if (params.live_kind == 2.0) {
			let e = clamp(params.live_energy, 0.0, 1.4);
			damping = 0.88 + 0.09 * e; // up to ~0.97 — longer, sloshier settling
			// Curl-free deterministic jitter: phase from node index + live_frame.
			let ph = f32(i) * 0.61803 + params.live_frame * 0.05;
			let jitter = vec3<f32>(sin(ph * 6.2831), sin(ph * 4.7123 + 1.3), sin(ph * 5.318 + 2.1));
			force = force + jitter * (0.006 * e);
		}

		// 7B: velocity damping + cap, then position integration.
		var vel = node.vel_retention.xyz;
		vel = (vel + force) * damping;
		vel = clamp_len(vel, 0.42);
		node.vel_retention = vec4<f32>(vel, node.vel_retention.w);
		node.pos_radius = vec4<f32>(pos + vel, node.pos_radius.w);
	}
	// When capture_mode (params.capture_mode == 1.0), node is NOT written back —
	// the storage buffer retains its initial upload values, guaranteeing
	// deterministic pixels for the same frame index.
	nodes[i] = node;
}
`;

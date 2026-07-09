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
	_pad: f32,
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

	let steps = u32(params.path_count);
	for (var s = 0u; s < steps; s = s + 1u) {
		let step = path[s];
		let bf = f32(step.z);

		if (step.y == i) {
			// Arrival: sharp attack as the wavefront lands, slow afterglow.
			let attack = smoothstep(bf - 14.0, bf + 4.0, frame);
			let decay = 1.0 - smoothstep(bf + 40.0, bf + 200.0, frame);
			intensity = max(intensity, attack * decay);
		}
		if (step.x == i && step.x != step.y) {
			// Departure: the source shimmers as the wave leaves it.
			let rise = smoothstep(bf - 55.0, bf - 30.0, frame);
			let fall = 1.0 - smoothstep(bf + 10.0, bf + 70.0, frame);
			intensity = max(intensity, rise * fall * 0.45);
		}
	}

	var node = nodes[i];
	let flags = u32(node.color_flags.w);
	let is_center = (flags & 1u) != 0u;

	// Write recall intensity (existing behavior preserved).
	node.demo.x = intensity;

	// --- Increment 7: force simulation ---

	// Capture mode (params._pad == 1.0): skip physics integration entirely.
	// The storage-buffer state stays frozen at initial upload values,
	// making same URL + frame → identical pixels (spec §4 Inc 9).
	if (params._pad == 0.0) {
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

		// 7B: velocity damping + cap, then position integration.
		var vel = node.vel_retention.xyz;
		vel = (vel + force) * 0.88;
		vel = clamp_len(vel, 0.42);
		node.vel_retention = vec4<f32>(vel, node.vel_retention.w);
		node.pos_radius = vec4<f32>(pos + vel, node.pos_radius.w);
	}
	// When capture_mode (params._pad == 1.0), node is NOT written back —
	// the storage buffer retains its initial upload values, guaranteeing
	// deterministic pixels for the same frame index.
	nodes[i] = node;
}
`;

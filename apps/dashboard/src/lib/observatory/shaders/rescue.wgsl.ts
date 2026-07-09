/**
 * Cognitive Observatory — salience-rescue choreography compute pass (WGSL).
 *
 * One invocation per node. Decodes the packed wave word (rescue-plan.ts:
 * bits 0-15 hopDepth, 16 isFailure, 17 isCause, 18 isLookalike, 19-21 k) and
 * writes ALL FOUR NodeState demo lanes as PURE functions of
 * (params.frame, params.loop_phase, packed word):
 *
 *   demo.x  cause ignition (existing thin-film recall response) + symptom backlight
 *   demo.y  searchlight cold-white flare on the K lookalikes + ash residue
 *   demo.z  backward-wave interrogation flicker + scanned ember
 *   demo.w  detonation spike + wound simmer + shockwave blinks + recognition flare
 *
 * This pass MUST encode AFTER recall_sim (simulate.wgsl) in the same encoder:
 * recall_sim rewrites demo.x every frame from the path buffer (afterglow decays
 * bf+40..bf+200 — the causal arc at bf=560 would leave a visible residual at
 * frame 719). Overwriting all four lanes here is simultaneously the
 * choreography, the loop-seam guarantee, and free ?frame=N capture support
 * (stateless: same frame in → same lanes out).
 *
 * Every envelope term is A·smoothstep(a0,a1,f)·(1−smoothstep(r0,r1,f)) with
 * attacks a0 ≥ 88 and releases r1 ≤ 700 ⇒ exact 0.0 at frames 0 and 719.
 * The flicker sine runs 24 INTEGER cycles per loop. The CPU mirror
 * (rescue-plan.ts rescueEnvelopes) is machine-checked by the seam-zero test —
 * keep both in lockstep.
 *
 * `hopSlot`/`causeDepth` are template-substituted f32 literals (no uniform
 * buffer → no strippable binding). Bind group = EXACTLY the 3 declared
 * bindings (params, nodes, wave).
 */

import type { RescueShaderConsts } from '../rescue-plan';

export function rescueWGSL(c: RescueShaderConsts): string {
	const hopSlot = c.hopSlot.toFixed(1);
	const causeDepth = c.causeDepth.toFixed(1);
	return /* wgsl */ `
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
// 1 u32/node: bits 0-15 hopDepth (0xffff unreached), 16 failure, 17 cause,
// 18 lookalike, 19-21 lookalike k (rescue-plan.ts packing).
@group(0) @binding(2) var<storage, read> wave: array<u32>;

const HOP_SLOT: f32 = ${hopSlot};
const CAUSE_DEPTH: f32 = ${causeDepth};
const TAU: f32 = 6.28318530717958647;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

fn arrival(d: f32) -> f32 {
	return min(260.0 + HOP_SLOT * d, 514.0);
}

@compute @workgroup_size(64)
fn rescue_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&wave)) {
		return;
	}
	// Belt-and-braces atop the TS gate: salience-rescue is demo index 2.
	if (params.demo_id != 2.0) {
		return;
	}

	let packed = wave[i];
	let depth_u = packed & 0xffffu;
	let d = f32(depth_u);
	let is_failure = (packed & 0x10000u) != 0u;
	let is_cause = (packed & 0x20000u) != 0u;
	let is_look = (packed & 0x40000u) != 0u;
	let look_k = f32((packed >> 19u) & 0x7u);

	let f = params.frame;

	var dx = 0.0;
	var dy = 0.0;
	var dz = 0.0;
	var dw = 0.0;

	if (is_failure) {
		// Detonation spike, wound simmer, recognition flare as the arc lands.
		dw = dw + env(f, 90.0, 96.0, 120.0, 168.0);
		dw = dw + 0.35 * env(f, 100.0, 130.0, 600.0, 656.0);
		dw = dw + 0.35 * env(f, 552.0, 562.0, 580.0, 640.0);
		// Symptom backlight while the cause burns.
		dx = dx + 0.4 * env(f, 556.0, 566.0, 620.0, 668.0);
	}
	if (!is_failure && depth_u >= 1u && depth_u <= 12u) {
		// Shockwave blink: crimson concussion, 3 frames/hop of REAL graph distance.
		dw = dw + 0.75 * exp(-0.3 * d)
			* env(f, 92.0 + 3.0 * d, 96.0 + 3.0 * d, 96.0 + 3.0 * d, 122.0 + 3.0 * d);
	}
	if (is_look) {
		let fk = 138.0 + 28.0 * look_k;
		// Searchlight flare — cold pop, sequential, on camera.
		dy = dy + env(f, fk - 6.0, fk, fk + 10.0, fk + 26.0);
		// Ash residue — the struck-through lookalike stays in frame until the verdict.
		dy = dy + 0.15 * smoothstep(fk + 10.0, fk + 26.0, f) * (1.0 - smoothstep(600.0, 656.0, f));
	}
	if (!is_failure && depth_u >= 1u && d <= CAUSE_DEPTH) {
		let wd = arrival(d);
		// Interrogation flicker: 24 integer sine cycles per loop, per-depth phase.
		let flicker = 0.75 + 0.25 * sin(TAU * 24.0 * params.loop_phase + 1.7 * d);
		dz = dz + env(f, wd - 10.0, wd, wd + 28.0, wd + 64.0) * flicker;
		// Scanned ember.
		dz = dz + 0.08 * smoothstep(wd + 28.0, wd + 64.0, f) * (1.0 - smoothstep(580.0, 640.0, f));
	}
	if (is_cause) {
		// Cause ignition rides the EXISTING recall response (render-nodes.wgsl):
		// spectral() thin-film band + white-hot core + sprite swell at full intensity.
		dx = dx + env(f, 520.0, 546.0, 640.0, 700.0);
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them).
	var node = nodes[i];
	node.demo = vec4<f32>(dx, dy, dz, dw);
	nodes[i] = node;
}
`;
}

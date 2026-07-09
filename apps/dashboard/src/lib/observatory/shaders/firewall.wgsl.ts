/**
 * Cognitive Observatory — firewall choreography compute pass (WGSL).
 *
 * One invocation per node. Decodes the packed fire word (firewall-plan.ts:
 * bits 0-7 shockDelay, bit 8 isIntruder, bit 9 isSeverNeighbor, bits 10-13
 * sever slot k) and writes ALL FOUR NodeState demo lanes as PURE functions of
 * (params.frame, params.loop_phase, packed word):
 *
 *   demo.x  ALWAYS 0.0 — the recall/thin-film grammar can never fire here
 *   demo.y  intruder only: intrusion flare band (0..1], 36 integer sine
 *           cycles/loop, then the sustained MEMBRANE band [2.60..2.90], 12
 *           integer cycles — one lane, two value ranges (render-nodes.wgsl
 *           separates them with min(fy,1) vs smoothstep(1.5, 2.2, fy))
 *   demo.z  ALWAYS 0.0 — the forgetting-horizon grammar can never fire here
 *   demo.w  crimson shock: source detonation on the intruder, per-node rim as
 *           the radial front passes (arrival A = 150 + delay, amplitude fades
 *           with distance), sever-blink receipts at 345 + 21k
 *
 * This pass MUST encode AFTER recall_sim (simulate.wgsl) in the same encoder:
 * recall_sim rewrites demo.x every frame from the path buffer (afterglow
 * decays bf+40..bf+200 — the k=5 sever beam at bf=450 would leave residual
 * demo.x near the seam). Overwriting all four lanes here is simultaneously
 * the choreography, the loop-seam guarantee, and free ?frame=N capture
 * support (stateless: same frame in → same lanes out).
 *
 * Every envelope has attack a0 ≥ 90 and release r1 ≤ 680 ⇒ exact 0.0 at
 * frames 0 and 719. Sines are factors on zero-at-seam envelopes with INTEGER
 * cycles per loop. The CPU mirror (firewall-plan.ts firewallEnvelopes) is
 * machine-checked by the seam-zero test — keep both in lockstep.
 *
 * Bind group = EXACTLY the 3 declared bindings (params, nodes, fire).
 */

export const firewallWGSL = /* wgsl */ `
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
// 1 u32/node: bits 0-7 shockDelay, 8 isIntruder, 9 isSeverNeighbor,
// 10-13 sever slot k (firewall-plan.ts packing). Every node carries a delay.
@group(0) @binding(2) var<storage, read> fire: array<u32>;

const TAU: f32 = 6.28318530717958647;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

@compute @workgroup_size(64)
fn firewall_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&fire)) {
		return;
	}
	// Belt-and-braces atop the TS gate: firewall is demo index 4.
	if (params.demo_id != 4.0) {
		return;
	}

	let packed = fire[i];
	let delay = f32(packed & 0xffu);
	let is_intruder = (packed & 0x100u) != 0u;
	let is_sever = (packed & 0x200u) != 0u;
	let k = f32((packed >> 10u) & 0xfu);

	let f = params.frame;

	var fy = 0.0;
	var fw = 0.0;

	if (is_intruder) {
		// Intrusion flare: sickly strobe, band (0..1], 36 integer cycles/loop.
		// C¹ handoff into the membrane over 330-332 (the rise sweeps the flare
		// band exactly once — the condensation read is intentional).
		fy = env(f, 90.0, 96.0, 310.0, 332.0)
			* (0.55 + 0.45 * sin(TAU * 36.0 * params.loop_phase));
		// Membrane: sustained ring band [2.60..2.90], 12 integer cycles/loop.
		fy = fy + env(f, 330.0, 352.0, 620.0, 680.0)
			* (2.75 + 0.15 * sin(TAU * 12.0 * params.loop_phase));
		// Source detonation as the front leaves.
		fw = env(f, 148.0, 153.0, 162.0, 196.0);
	} else {
		// Crimson rim as the radial front passes: arrival A = 150 + delay,
		// amplitude fades with distance; A ∈ [150, 294] ⇒ all rims dead by 320.
		let a = 150.0 + delay;
		let amp = 0.9 - 0.45 * (delay / 144.0);
		fw = amp * env(f, a - 2.0, a + 3.0, a + 8.0, a + 26.0);
		if (is_sever) {
			// Node-side receipt of the severed edge; last release 474.
			let sk = 345.0 + 21.0 * k;
			fw = fw + 0.6 * env(f, sk - 4.0, sk, sk + 6.0, sk + 24.0);
		}
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them). demo.x and
	// demo.z are hard 0.0: the recall and horizon grammars can never fire here.
	var node = nodes[i];
	node.demo = vec4<f32>(0.0, fy, 0.0, fw);
	nodes[i] = node;
}
`;

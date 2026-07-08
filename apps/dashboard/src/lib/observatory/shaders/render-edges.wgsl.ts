/**
 * Cognitive Observatory — edge wavefront render shader (WGSL).
 *
 * Draws additive edges between nodes, with a traveling wavefront that
 * travels from source → target at the beat frame. The wavefront is a
 * glowing pulse that rides along the edge, brightening as it approaches
 * the target node (spec §7.2: additive bloom, thin-film spectral glow).
 *
 * Layout contracts: Params = types.PARAMS_FLOATS, Edge = 2×vec2<u32>
 * (types.ts UINTS_PER_EDGE), PathStep = 4×vec4<u32> (types.ts
 * UINTS_PER_PATHSTEP).
 */
export const renderEdgesWGSL = /* wgsl */ `
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

// x source index, y target index, z beat frame, w kind (0 recall, 1 backward)
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
// Source/target node indices (2 u32 per edge).
@group(0) @binding(2) var<storage, read> edges: array<vec2<u32>>;
// PathStep buffer for wavefront timing.
@group(0) @binding(3) var<storage, read> path: array<vec4<u32>>;
// NodeState storage buffer (positions for edge endpoints).
@group(0) @binding(4) var<storage, read> nodes: array<Node>;

// Iridescent thin-film band — ported EXACTLY from causal-brain-demo.html
// spectral(w) (visual DNA §7.1): indigo → cyan-teal → mint → magenta rim.
fn spectral(w_in: f32) -> vec3<f32> {
	let w = fract(w_in);
	let stops = array<vec3<f32>, 4>(
		vec3<f32>(0.20, 0.28, 0.95), // indigo
		vec3<f32>(0.20, 0.85, 0.90), // cyan-teal
		vec3<f32>(0.45, 1.00, 0.72), // mint
		vec3<f32>(0.85, 0.45, 1.00)  // magenta rim
	);
	let f = w * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	let a = stops[i];
	let b = stops[(i + 1u) % 4u];
	return mix(a, b, frac);
}

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) color: vec3<f32>,
	@location(1) width: f32,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;

	let edgeCount = u32(params.edge_count);
	if (ii >= edgeCount) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let edge = edges[ii];
	let srcIdx = edge.x;
	let tgtIdx = edge.y;

	if (srcIdx >= u32(params.node_count) || tgtIdx >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let src = nodes[srcIdx];
	let tgt = nodes[tgtIdx];

	// Two vertices per edge: source (vi=0) and target (vi=1).
	let pos = select(src.pos_radius.xyz, tgt.pos_radius.xyz, vi == 1u);

	// World-space position.
	let world = pos;
	out.clip = camera.view_proj * vec4<f32>(world, 1.0);

	// Wavefront computation: find the nearest path beat for this edge.
	let pathCount = u32(params.path_count);
	var waveIntensity = 0.0;
	var waveT = 1.0; // 0 = source, 1 = target

	for (var s = 0u; s < pathCount; s = s + 1u) {
		let step = path[s];
		let srcIdxS = step.x;
		let tgtIdxS = step.y;
		let bf = f32(step.z);

		// Check if this path step uses the same source→target.
		if (srcIdxS == srcIdx && tgtIdxS == tgtIdx) {
			let frame = params.frame;
			// Wavefront: sharp pulse traveling from source to target.
			let attack = smoothstep(bf - 10.0, bf + 2.0, frame);
			let decay = 1.0 - smoothstep(bf + 30.0, bf + 180.0, frame);
			waveIntensity = max(waveIntensity, attack * decay);

			// Wave position along edge (0 = source, 1 = target).
			let arrival = bf - 10.0;
			let end = bf + 30.0;
			if (frame >= arrival && frame <= end) {
				waveT = (frame - arrival) / (end - arrival);
			} else if (frame > end) {
				waveT = 1.0;
			}
		}
	}

	// Edge base color: blend of source and target node base colors.
	let srcColor = src.color_flags.rgb;
	let tgtColor = tgt.color_flags.rgb;
	let baseColor = mix(srcColor, tgtColor, 0.5);

	// Wavefront color: thin-film spectral band, modulated by wave position.
	let waveColor = spectral(waveT + params.loop_phase);

	// Combine: base edge (dim) + wavefront pulse (bright, additive).
	let edgeAlpha = 0.08 * params.brightness; // dim connecting line
	let waveAlpha = waveIntensity * 0.9 * params.brightness; // bright pulse

	// Spectral hue rides the wavefront.
	out.color = baseColor * edgeAlpha + waveColor * waveAlpha;

	// Line width: thicker at the wavefront for visibility.
	out.width = 1.0 + waveIntensity * 3.0;

	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	// Soft edge: feather the line edges.
	let alpha = smoothstep(0.0, 0.5, in.width) * 0.6;
	// Additive: alpha is ignored, light accumulates.
	return vec4<f32>(in.color, 1.0);
}
`;

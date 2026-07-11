export const MSDF_TEXT_WGSL = /* wgsl */ `
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

struct Glyph {
	anchor_size: vec4f,
	quad_offset: vec4f,
	uv_rect: vec4f,
	info: vec4f,
	color: vec4f,
};

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) info: vec4f,
	@location(2) @interpolate(flat) color: vec4f,
	@location(3) @interpolate(flat) weight: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> glyphs: array<Glyph>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var atlas_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
	vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0)
);

fn median3(c: vec3f) -> f32 {
	return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

@vertex
fn vs_text(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let glyph = glyphs[ii];
	let corner = QUAD[vi];
	let anchor = glyph.anchor_size.xy;
	let size = glyph.anchor_size.zw;
	let quad_offset = glyph.quad_offset.xy;
	let uv_min = glyph.uv_rect.xy;
	let uv_max = glyph.uv_rect.zw;
	let aspect = max(0.0001, params.viewport_w / max(1.0, params.viewport_h));
	let depth = clamp(glyph.info.z, 0.0, 1.0);
	let cursor_pre = vec2f(params.cursor_x, params.cursor_y);
	let cursor_delta = cursor_pre - anchor;
	let d = distance(anchor, cursor_pre);
	// Wide influence radius so the field reacts when the cursor is anywhere NEAR
	// the text, not only dead-on (v1 R=0.45 was too tight to feel).
	let R = 0.75;
	let cursor_w = exp(-(d * d) / (R * R));
	// Per-glyph SCALE-UP near the cursor: glyphs the pointer approaches swell toward
	// you. Scaling the quad around its anchor is the most legible "alive" cue.
	let grow = 1.0 + cursor_w * 0.55;
	var pos = anchor + (quad_offset + corner * size) * grow;
	// Depth → clip z. Trust (depth~1) floats forward (small z), low-trust sinks back.
	// Cursor lifts a glyph forward, but z MUST stay > 0 or clip.z<0 clips the quad
	// behind the near plane and the glyph vanishes (v1 bug: cursor made text disappear).
	var z = mix(0.42, 0.10, depth);
	z = clamp(z - cursor_w * 0.42, 0.04, 0.6);
	let lean_dir = select(vec2f(0.0, 0.0), normalize(cursor_delta), length(cursor_delta) > 0.0001);
	pos = pos + lean_dir * cursor_w * 0.04;
	pos = pos + vec2f(sin(params.time * 0.6), cos(params.time * 0.5)) * ((1.0 - depth) * 0.006) * params.pulse;
	// Keep glyphs square in BOTH orientations: normalize by the longer axis.
	// Landscape (aspect>1): narrow x. Portrait (aspect<1): shrink y instead —
	// dividing x by aspect<1 would WIDEN x and push text off-screen.
	pos.x = pos.x / max(aspect, 1.0);
	pos.y = pos.y * min(aspect, 1.0);
	let wclip = 1.0 + z;
	var out: VSOut;
	out.clip = vec4f(pos, z, wclip);
	out.uv = vec2f(mix(uv_min.x, uv_max.x, corner.x), mix(uv_max.y, uv_min.y, corner.y));
	out.info = vec4f(glyph.info.x, glyph.info.y, cursor_w, depth);
	out.color = glyph.color;
	out.weight = clamp(glyph.info.w, 0.0, 1.0);
	return out;
}

@fragment
fn fs_text(in: VSOut) -> @location(0) vec4f {
	let atlas_px = vec2f(textureDimensions(atlas_tex, 0));
	let cursor_w = clamp(in.info.z, 0.0, 1.0);
	let depth = clamp(in.info.w, 0.0, 1.0);
	let weight = clamp(in.weight, 0.0, 1.0);
	var uv = in.uv;
	uv = uv + vec2f(sin(uv.y * 40.0 + params.time * 3.0), cos(uv.x * 40.0 + params.time * 3.0)) * (cursor_w * 0.007);
	let msdf = textureSample(atlas_tex, atlas_sampler, uv).rgb;
	let dist = median3(msdf);
	let uv_width = max(fwidth(uv), vec2f(1.0 / max(atlas_px.x, 1.0), 1.0 / max(atlas_px.y, 1.0)));
	let texels_per_px = max(length(uv_width * atlas_px), 0.0001);
	let screen_range = max(0.5, 4.0 / texels_per_px);
	// Depth-of-field: far/un-hovered glyphs soften, cursor sharpens. Kept GENTLE so
	// the resting field stays READABLE regardless of the data's depth value.
	let dof = (1.0 - depth) * (1.0 - cursor_w);
	let screen_range_dof = screen_range / (1.0 + dof * 0.6);
	// Weight (FSRS retention) modulates stroke mass WITHIN a readable band: it can
	// thicken a lot but only thin slightly, so a low-retention record never
	// disappears (data must be legible even at weight~0 — every route depends on this).
	let weight_bias = (weight - 0.5) * 0.10 + 0.03;
	let px_dist = screen_range_dof * (dist - 0.5 + weight_bias);
	let coverage = clamp(px_dist + 0.5, 0.0, 1.0);
	let reveal_span = max(1.0, in.info.y);
	let reveal = clamp((params.frame - in.info.x) / reveal_span, 0.0, 1.0);
	let alpha = coverage * in.color.a * reveal;
	if (alpha < 0.001) { discard; }
	// Glow floor keeps EVERY line clearly lit at rest (even depth~0), depth adds
	// forward-brightness, cursor pushes near glyphs HARD past the bloom line to flare.
	let glow = mix(1.15, 1.5, depth) + cursor_w * 1.4;
	let rgb = in.color.rgb * params.brightness * glow;
	return vec4f(rgb * alpha, alpha);
}
`;

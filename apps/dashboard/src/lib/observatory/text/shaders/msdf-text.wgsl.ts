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
	var pos = anchor + quad_offset + corner * size;
	// Keep glyphs square in BOTH orientations: normalize by the longer axis.
	// Landscape (aspect>1): narrow x. Portrait (aspect<1): shrink y instead —
	// dividing x by aspect<1 would WIDEN x and push text off-screen.
	pos.x = pos.x / max(aspect, 1.0);
	pos.y = pos.y * min(aspect, 1.0);
	var out: VSOut;
	out.clip = vec4f(pos, 0.0, 1.0);
	out.uv = vec2f(mix(uv_min.x, uv_max.x, corner.x), mix(uv_max.y, uv_min.y, corner.y));
	out.info = glyph.info;
	out.color = glyph.color;
	return out;
}

@fragment
fn fs_text(in: VSOut) -> @location(0) vec4f {
	let atlas_px = vec2f(textureDimensions(atlas_tex, 0));
	let msdf = textureSample(atlas_tex, atlas_sampler, in.uv).rgb;
	let dist = median3(msdf);
	let uv_width = max(fwidth(in.uv), vec2f(1.0 / max(atlas_px.x, 1.0), 1.0 / max(atlas_px.y, 1.0)));
	let texels_per_px = max(length(uv_width * atlas_px), 0.0001);
	let screen_range = max(0.5, 4.0 / texels_per_px);
	let px_dist = screen_range * (dist - 0.5);
	let coverage = clamp(px_dist + 0.5, 0.0, 1.0);
	let reveal_span = max(1.0, in.info.y);
	let reveal = clamp((params.frame - in.info.x) / reveal_span, 0.0, 1.0);
	let alpha = coverage * in.color.a * reveal;
	if (alpha < 0.001) { discard; }
	let rgb = in.color.rgb * params.brightness;
	return vec4f(rgb * alpha, alpha);
}
`;

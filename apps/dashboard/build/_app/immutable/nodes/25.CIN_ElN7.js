import{$ as e,D as t,F as n,H as r,J as i,K as a,L as o,O as s,P as c,Q as l,T as u,U as d,X as f,Y as p,Z as m,_ as h,b as g,et as _,f as v,j as y,k as b,lt as x,mt as S,pt as C,r as w,tt as T,ut as E,w as D}from"../chunks/Cmba2AGc.js";import"../chunks/xihTtKlq.js";import{t as ee}from"../chunks/yeJIv7a5.js";import{a as te,o as O,u as k}from"../chunks/CSLtwkW3.js";import{t as ne}from"../chunks/CUbIPCV3.js";var A=`rgba16float`,j=768,M=96,N=16,P=12,F=`
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

struct TimelineCellGpu {
	// x,y NDC; z cell radius; w ring radius
	pos_radius: vec4f,
	// x retention, y rewritten, z suppressed, w audit events
	signals: vec4f,
	// x valid-time phase, y transaction-time phase, z day index, w cell index
	time_meta: vec4f,
	// x selected, y reserved, z reserved, w reserved
	flags: vec4f,
};

struct TimelineRingGpu {
	// x radius, y count scale, z retention, w day index
	shape: vec4f,
	// x updated count, y suppressed count, z phase, w selected
	activity: vec4f,
	// x memory count, y ring index, z reserved, w reserved
	// ('meta' is a WGSL reserved keyword — see GOD-TIER §9 / it broke Blackbox too)
	stats: vec4f,
};

// Portrait legibility: on a phone the growth-ring field is the whole screen and
// its HDR bloom becomes a BLINDING blob that drowns the MSDF HUD/receipt text.
// Derive a dim factor from the LIVE viewport aspect (viewport_w/viewport_h) —
// nothing is hardcoded per device. Landscape/desktop (aspect >= 0.85) is left at
// full brightness (1.0); portrait scales down toward ~0.34 as it narrows so the
// field becomes a DIM backdrop and the overlay text wins the contrast fight.
fn portrait_field_dim() -> f32 {
	let a = params.viewport_w / max(params.viewport_h, 1.0);
	// portraitness: 0 at aspect 0.85 (landscape edge) -> 1 at aspect 0.46 (tall phone)
	let p = clamp((0.85 - a) / (0.85 - 0.46), 0.0, 1.0);
	// The ring/membrane colors are pushed HARD into HDR (peak accumulated ~5-8x via
	// additive blend) specifically so the post-chain bloom flares them. A 0.2 dim
	// still leaves ~1.0-1.6 — above the bloom knee, so it stayed a blinding blob on
	// a phone. Pull it down to ~0.07 at full portrait so even the accumulated HDR
	// peak lands well below the bloom threshold and the field reads as a true DIM
	// backdrop the MSDF HUD/receipt text can win against. Aspect-derived, no per-
	// device constant; landscape/desktop (aspect>=0.85) stays untouched at 1.0.
	return mix(1.0, 0.07, p);
}
`,I=`
${F}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<TimelineCellGpu>;
@group(0) @binding(2) var<storage, read> rings: array<TimelineRingGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) extra: vec4f,
};

// Living orbital drift: every cell slowly circulates around the ring center
// (the tree of memory is always turning), plus a per-cell radial breathe. Motion
// is a pure function of params.time + per-cell phase — deterministic, no RNG.
// This is what makes the field MOVE like the Observatory force-sim, not sit still.
// Shared rotation for a given normalized day phase (0 = oldest/outer, 1 = newest/
// inner). Inner rings turn faster, like the fast core of a spinning galaxy. Cells
// AND their rings both call this so cells stay ON their ring while everything turns.
fn ring_spin(day_phase: f32) -> f32 {
	let speed = 0.045 + day_phase * 0.10;
	return params.time * speed;
}

fn orbit(base: vec2f, phase: f32, day_phase: f32, ret: f32) -> vec2f {
	let radius = length(base);
	if (radius < 0.0001) { return base; }
	let ang0 = atan2(base.y, base.x);
	// rotate with the ring, plus a tiny per-cell wobble so cells shimmer on the ring
	let ang = ang0 + ring_spin(day_phase) + sin(params.time * 0.6 + phase * 6.283) * 0.02;
	// radial breathe so the whole tree gently expands/contracts as it turns
	let rr = radius * (1.0 + 0.016 * sin(params.time * 1.1 + phase * 6.283));
	return vec2f(cos(ang), sin(ang)) * rr;
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let breathe = 1.0 + 0.10 * sin(params.time * 1.6 + c.time_meta.x * 6.28318);
	let r = c.pos_radius.z * breathe * (1.0 + c.flags.x * 1.4);
	let center = orbit(c.pos_radius.xy, c.time_meta.w, c.time_meta.x, c.signals.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.misc = c.signals;
	out.extra = c.time_meta;
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewritten = in.misc.y;
	let suppressed = in.misc.z;
	let audit = clamp(in.misc.w, 0.0, 8.0) / 8.0;
	let body = exp(-d*d*3.1) * (0.34 + retention * 0.86);
	let seam = rewritten * smoothstep(0.10, 0.0, abs(d - 0.52)) * (0.55 + audit * 0.8);
	let scar = suppressed * smoothstep(0.98, 0.68, d);
	// .r = valid-time growth density, .g = retention oxygen, .b = transaction-time seam/shadow
	return vec4f(body, body * retention, seam + scar * 0.45, 1.0);
}

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	// pulse the cell size with its own heartbeat so cells throb as they orbit
	let beat = 1.0 + 0.22 * sin(params.time * 2.3 + c.time_meta.w * 1.7);
	let r = c.pos_radius.z * (0.55 + c.flags.x * 0.8) * beat;
	let center = orbit(c.pos_radius.xy, c.time_meta.w, c.time_meta.x, c.signals.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.misc = c.signals;
	out.extra = c.time_meta;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewritten = in.misc.y;
	let suppressed = in.misc.z;
	let oxygen = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.95, 0.55, 0.15);
	let indigo = vec3f(0.486, 0.424, 1.0);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	let core = mix(amber, oxygen, retention);
	// Each memory cell is a living bioluminescent organism — pulse by its own phase
	// (time_meta.x) so the field twinkles, and push core to HDR so it GLOWS.
	let cell_phase = in.extra.x;
	let twinkle = 0.6 + 0.8 * (0.5 + 0.5 * sin(params.time * 2.1 + cell_phase * 26.0));
	let body = exp(-d*d*2.7) * (0.55 + retention * 1.7) * twinkle;
	let rim = smoothstep(0.98, 0.74, d) * (1.0 - smoothstep(0.74, 0.42, d));
	let seam = smoothstep(0.12, 0.0, abs(d - 0.48)) * rewritten;
	let scar = smoothstep(0.16, 0.0, abs(d - 0.76)) * suppressed;
	return vec4f((core * body + vec3f(0.91, 1.0, 0.72) * rim * 1.1 + indigo * seam * 1.3 + scarlet * scar * 1.5) * portrait_field_dim(), 1.0);
}

@vertex
fn vs_ring(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let ring = rings[ii];
	let seg = vi / 2u;
	let side = f32(vi % 2u) * 2.0 - 1.0;
	let t = f32(seg) / 95.0;
	// rotate the whole ring with the same galaxy spin the cells use (activity.z =
	// normalized ring phase) so cells ride ON their turning ring, alive together.
	let angle = t * 6.2831853 + ring_spin(ring.activity.z);
	let dir = vec2f(cos(angle), sin(angle));
	let retention = ring.shape.z;
	let rewrite = ring.activity.x / max(1.0, ring.stats.x);
	let suppressed = ring.activity.y / max(1.0, ring.stats.x);
	let thickness = 0.0035 + 0.006 * retention + 0.004 * ring.activity.w;
	let ripple = 0.006 * sin(angle * 9.0 + params.time * (0.28 + ring.activity.z));
	let radius = ring.shape.x + side * thickness + ripple * rewrite;
	let tx = 0.030 * rewrite;
	var out: VSOut;
	// Indigo transaction-time shadow: duplicate the ring instance offset by the real rewrite amount.
	let indigo_shift = select(0.0, tx, side > 0.0);
	out.clip = vec4f(dir * radius + vec2f(indigo_shift, -indigo_shift * 0.42), 0.0, 1.0);
	out.uv = vec2f(t, side);
	out.misc = vec4f(retention, rewrite, suppressed, ring.activity.w);
	out.extra = vec4f(ring.shape.y, ring.shape.w, ring.stats.x, ring.activity.z);
	return out;
}

@fragment
fn fs_ring(in: VSOut) -> @location(0) vec4f {
	let retention = clamp(in.misc.x, 0.0, 1.0);
	let rewrite = clamp(in.misc.y, 0.0, 1.0);
	let suppressed = clamp(in.misc.z, 0.0, 1.0);
	let selected = in.misc.w;
	let tick = step(0.86, fract(in.uv.x * 24.0));
	let oxygen = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.86, 0.42, 0.12);
	let indigo = vec3f(0.486, 0.424, 1.0);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	// Living pulse: each ring breathes with the global breath + a per-ring phase so
	// the rings shimmer OUT OF SYNC like a real organism, not one flat pattern.
	let phase = in.extra.w; // ring.activity.z packed as phase
	let live = 0.55 + 0.65 * (0.5 + 0.5 * sin(params.time * (0.9 + phase * 1.3) + phase * 6.283));
	// HDR brightness (>1) so the enzyme light BLOOMS through the post chain.
	var color = mix(amber, oxygen, retention) * (0.5 + 1.5 * retention + 1.1 * selected) * live;
	color = color + indigo * rewrite * (1.1 + 0.7 * abs(in.uv.y));
	color = color + scarlet * suppressed * 1.4;
	// Bright engraved date ticks flare on selection.
	color = color + vec3f(0.91, 1.0, 0.72) * tick * (0.14 + selected * 0.6);
	return vec4f(color * portrait_field_dim(), 1.0);
}
`,L=`
${F}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(field_tex, 0));
	let px = 1.0 / max(dims, vec2f(1.0));
	let f = textureSample(field_tex, field_sampler, in.uv);
	let left = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(px.x, 0.0), 0.0);
	let right = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(px.x, 0.0), 0.0);
	let down = textureSampleLevel(field_tex, field_sampler, in.uv - vec2f(0.0, px.y), 0.0);
	let up = textureSampleLevel(field_tex, field_sampler, in.uv + vec2f(0.0, px.y), 0.0);
	let density = clamp(f.r, 0.0, 5.0);
	let oxygen = clamp(f.g, 0.0, 5.0);
	let seam = clamp(f.b, 0.0, 3.0);
	let grad = length(vec2f((right.r + right.g) - (left.r + left.g), (up.r + up.g) - (down.r + down.g)));
	let membrane = smoothstep(0.08, 0.70, density) * (1.0 - smoothstep(1.8, 3.8, density));
	let edge = smoothstep(0.01, 0.12, grad) * membrane;
	let blackwater = vec3f(0.006, 0.012, 0.014);
	let retention = vec3f(0.66, 1.0, 0.37);
	let amber = vec3f(0.86, 0.42, 0.12);
	let indigo = vec3f(0.486, 0.424, 1.0);
	// Metabolic breathing — the whole tissue pulses with the global breath so the
	// field reads as ALIVE, not a static print. pulse is 0..1 (params.pulse).
	let breath = 0.72 + 0.55 * params.pulse;
	var color = blackwater * (0.30 + density * 0.10);
	// Oxygen-lit plasma, pushed into HDR (>1) so the post-chain bloom makes it GLOW.
	color = color + mix(amber, retention, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.34 * breath;
	// Bright enzymatic edge — this is the "wet membrane" rim light; HDR for bloom flare.
	color = color + vec3f(0.91, 1.0, 0.72) * edge * (0.85 + 0.5 * params.pulse);
	// Indigo transaction-time seams shimmer with the breath.
	color = color + indigo * seam * (0.55 + 0.35 * params.pulse);
	let vignette = smoothstep(0.98, 0.12, distance(in.uv, vec2f(0.5)));
	return vec4f(color * (0.55 + 0.45 * vignette) * params.brightness * portrait_field_dim(), 1.0);
}
`,R=`
struct BlurDir { dir: vec2f, _pad: vec2f };
@group(0) @binding(0) var blur_sampler: sampler;
@group(0) @binding(1) var blur_src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur_dir: BlurDir;
const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}
@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, in.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`,z=class{engine;scene=null;resources=null;sampler=null;splatBindLayout=null;blurBindLayout=null;membraneBindLayout=null;splatPipeline=null;blurPipeline=null;membranePipeline=null;cellPipeline=null;ringPipeline=null;cellCount=0;ringCount=0;selectedId=null;cellGeometry=[];ringGeometry=[];constructor(e,t){this.engine=e,this.uploadScene(t)}uploadScene(e){this.scene=e,this.buildGeometry();let t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.uploadBuffers(t))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;let t=B(e,`timeline-growth-rings-splat-wgsl`,I),n=B(e,`timeline-growth-rings-blur-wgsl`,R),r=B(e,`timeline-growth-rings-membrane-wgsl`,L);this.splatBindLayout=e.createBindGroupLayout({label:`timeline-growth-rings-splat-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}}]}),this.blurBindLayout=e.createBindGroupLayout({label:`timeline-growth-rings-blur-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:`timeline-growth-rings-membrane-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}}]});let i=e.createPipelineLayout({label:`timeline-growth-rings-splat-layout`,bindGroupLayouts:[this.splatBindLayout]}),a=e.createPipelineLayout({label:`timeline-growth-rings-blur-layout`,bindGroupLayouts:[this.blurBindLayout]}),o=e.createPipelineLayout({label:`timeline-growth-rings-membrane-layout`,bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:`linear`,minFilter:`linear`}),this.splatPipeline=e.createRenderPipeline({label:`timeline-field-additive-splat`,layout:i,vertex:{module:t,entryPoint:`vs_splat`},fragment:{module:t,entryPoint:`fs_splat`,targets:[{format:A,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.blurPipeline=e.createRenderPipeline({label:`timeline-field-blur-render-pass`,layout:a,vertex:{module:n,entryPoint:`vs_fullscreen`},fragment:{module:n,entryPoint:`fs_blur`,targets:[{format:A}]},primitive:{topology:`triangle-list`}});let s={color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}};this.membranePipeline=e.createRenderPipeline({label:`timeline-bitemporal-membrane`,layout:o,vertex:{module:r,entryPoint:`vs_fullscreen`},fragment:{module:r,entryPoint:`fs_membrane`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-list`}}),this.ringPipeline=e.createRenderPipeline({label:`timeline-valid-time-rings`,layout:i,vertex:{module:t,entryPoint:`vs_ring`},fragment:{module:t,entryPoint:`fs_ring`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-strip`}}),this.cellPipeline=e.createRenderPipeline({label:`timeline-memory-cells`,layout:i,vertex:{module:t,entryPoint:`vs_cell`},fragment:{module:t,entryPoint:`fs_cell`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-list`}})}ensureResources(e){if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;let t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),n=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==n,i=this.resources?.cellBuffer,a=this.resources?.ringBuffer,o=this.resources?.blurHBuffer,s=this.resources?.blurVBuffer;if(i||=e.createBuffer({label:`timeline-cells`,size:j*N*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),a||=e.createBuffer({label:`timeline-rings`,size:4608,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),o||(o=e.createBuffer({label:`timeline-blur-h-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(o,0,new Float32Array([1,0,0,0]))),s||(s=e.createBuffer({label:`timeline-blur-v-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([0,1,0,0]))),!r&&this.resources)return;this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy();let c=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,l=e.createTexture({label:`timeline-field-a-rgba16float`,size:[t,n],format:A,usage:c}),u=e.createTexture({label:`timeline-field-b-rgba16float`,size:[t,n],format:A,usage:c}),d=l.createView(),f=u.createView(),p=e.createBindGroup({label:`timeline-growth-rings-splat-bind`,layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:a}}]}),m=e.createBindGroup({label:`timeline-field-blur-h-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:d},{binding:2,resource:{buffer:o}}]}),h=e.createBindGroup({label:`timeline-field-blur-v-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:f},{binding:2,resource:{buffer:s}}]}),g=e.createBindGroup({label:`timeline-membrane-bind`,layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:d}]});this.resources={cellBuffer:i,ringBuffer:a,blurHBuffer:o,blurVBuffer:s,splatBindGroup:p,blurHBindGroup:m,blurVBindGroup:h,membraneBindGroup:g,fieldA:l,fieldB:u,fieldAView:d,fieldBView:f,fieldSize:[t,n]}}buildGeometry(){let e=this.scene?.cells??[];this.cellGeometry=e.slice(0,j).map(e=>({cell:e,x:Math.cos(e.angle)*e.radius,y:Math.sin(e.angle)*e.radius,r:.018+e.retention*.016})),this.ringGeometry=(this.scene?.rings??[]).slice(0,M).map(e=>({ring:e,r:e.radius}))}uploadBuffers(e){if(!this.resources)return;let t=new Float32Array(j*N);this.cellCount=Math.min(j,this.cellGeometry.length);let n=Math.max(1,this.ringGeometry.length-1);for(let e=0;e<this.cellCount;e++){let r=this.cellGeometry[e],i=r.cell,a=+(this.selectedId===i.id||this.selectedId===i.memoryId);t.set([r.x,r.y,r.r,i.radius,i.retention,+!!i.rewritten,+!!i.suppressed,this.scene?.raw.audits[i.memoryId]?.length??0,i.dayIndex/n,Date.parse(i.transactionAt||i.validFrom||``)/864e11||0,i.dayIndex,e,a,0,0,0],e*N)}this.ringCount=Math.min(M,this.ringGeometry.length);let r=new Float32Array(1152),i=Math.max(1,this.scene?.scalars.maxDayCount??1);for(let e=0;e<this.ringCount;e++){let t=this.ringGeometry[e],n=t.ring,a=+(this.selectedId===n.id||this.selectedId===n.date);r.set([t.r,n.count/i,n.retention,n.index,n.updatedCount,n.suppressedCount,e/Math.max(1,this.ringCount),a,n.memoryIndices.length,e,0,0],e*P)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.ringCount,e.queue.writeBuffer(this.resources.cellBuffer,0,t),e.queue.writeBuffer(this.resources.ringBuffer,0,r)}compute(e){let t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);let n=this.resources,r=e.beginRenderPass({label:`timeline-field-splat-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.splatPipeline),r.setBindGroup(0,n.splatBindGroup),this.cellCount>0&&r.draw(6,this.cellCount),r.end();let i=e.beginRenderPass({label:`timeline-field-blur-h-pass`,colorAttachments:[{view:n.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});i.setPipeline(this.blurPipeline),i.setBindGroup(0,n.blurHBindGroup),i.draw(6,1),i.end();let a=e.beginRenderPass({label:`timeline-field-blur-v-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});a.setPipeline(this.blurPipeline),a.setBindGroup(0,n.blurVBindGroup),a.draw(6,1),a.end()}render(e){this.resources&&this.membranePipeline&&this.ringPipeline&&this.cellPipeline&&(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.ringCount>0&&(e.setPipeline(this.ringPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(192,this.ringCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.draw(6,this.cellCount)))}ringSpin(e){return(this.engine.params[10]||0)*(.045+e*.1)}orbitCpu(e,t,n,r){let i=Math.hypot(e,t);if(i<1e-4)return{x:e,y:t};let a=this.engine.params[10]||0,o=Math.atan2(t,e)+this.ringSpin(r)+Math.sin(a*.6+n*6.283)*.02,s=i*(1+.016*Math.sin(a*1.1+n*6.283));return{x:Math.cos(o)*s,y:Math.sin(o)*s}}pickAt(e,t){let n=Math.max(1,this.ringGeometry.length-1),r=null,i=1/0;for(let a=0;a<this.cellGeometry.length;a++){let o=this.cellGeometry[a],s=o.cell.dayIndex/n,c=this.orbitCpu(o.x,o.y,a,s),l=Math.hypot(e-c.x,t-c.y);l<=Math.max(.045,o.r*1.8)&&l<i&&(r={id:o.cell.id,kind:`timeline-cell`,index:a,payload:o.cell},i=l)}if(r)return this.selectedId=r.id,r;let a=Math.hypot(e,t),o=this.engine.params[10]||0;for(let e=0;e<this.ringGeometry.length;e++){let t=this.ringGeometry[e],r=n>0?e/n:0,i=t.r*(1+.016*Math.sin(o*1.1+r*6.283));if(Math.abs(a-i)<=.03)return this.selectedId=t.ring.id,{id:t.ring.id,kind:`timeline-ring`,index:e,payload:t.ring}}return null}dispose(){this.resources?.cellBuffer.destroy(),this.resources?.ringBuffer.destroy(),this.resources?.blurHBuffer.destroy(),this.resources?.blurVBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}};function B(e,t,n){e.pushErrorScope(`validation`);let r=e.createShaderModule({label:t,code:n});return r.getCompilationInfo().then(e=>{for(let n of e.messages)console.error(`[observatory] ${t} WGSL ${n.type} ${n.lineNum}:${n.linePos} ${n.message}`)}),e.popErrorScope().then(e=>{e&&console.error(`[observatory] ${t} shader module validation: ${e.message}`)}),r}function re(e,t){return k(te.blackwater),k(O.healthy),k(O.luciferin),[new z(e,t)]}function V(e,t){if(typeof e!=`number`||typeof t!=`number`||!Number.isFinite(e)||!Number.isFinite(t))return null;let n=Math.round(e*100),r=Math.round(t*100),i=r-n;return`${n}% → ${r}% (${i>0?`+`:``}${i})`}function ie(e){let t=[],n=V(e.old_value,e.new_value);return n&&t.push(n),e.reason?.trim()&&t.push(e.reason.trim()),e.triggered_by?.trim()&&t.push(`by ${e.triggered_by.trim()}`),t}function H(e){return!!(e.createdAt&&e.updatedAt&&e.createdAt!==e.updatedAt)}function U(e,t=``){return typeof e==`string`?e:e==null?t:String(e)}function W(e,t=0){return typeof e==`number`&&Number.isFinite(e)?e:t}function G(e){return Math.max(0,Math.min(1,e))}function K(e,t,n){return n?{kind:e,id:t,scalar:n}:{kind:e,id:t||`${e}:unknown`}}function q(e,t){return{kind:`scalar`,id:`timeline.${e}`,scalar:{name:e,value:t}}}function J(e){return G(W(e.retentionStrength,0))}function Y(e){return G(W(e.combinedScore??e.retentionStrength,J(e)))}function ae(e,t){return t[e.id]??[]}function X(e,t){return e.some(e=>e.action===t)}function oe(e){let t=e.days??[],n=e.audits??{},r=[],i=[],a=[],o=[],s=[],c=[],l=t.filter(e=>e.count>0||e.memories.length>0),u=Math.max(1,l.length),d=Math.max(1,...l.map(e=>e.count||e.memories.length));l.forEach((e,t)=>{let o=.16+t/Math.max(1,u-1)*.7,l=e.memories??[],d=[],f=0,p=0,m=0;l.forEach((a,u)=>{let h=ae(a,n),g=J(a),_=U(a.updatedAt)!==U(a.createdAt)||X(h,`edited`)||X(h,`reconsolidated`),v=X(h,`suppressed`)||W(a.suppression_count,0)>0;_&&(p+=1),v&&(m+=1),f+=g;let y=r.length;d.push(y);let b=(u+.5)/Math.max(1,l.length)*Math.PI*2+t*.37,x=(u%5-2)*.008,S=o+x,C=U(a.validFrom??a.createdAt,e.date),w=U(a.updatedAt??a.createdAt,C),T=a.content||a.id.slice(0,8),E=K(`memory`,a.id);if(r.push({source:E,index:y,label:T,retention:g,trust:Y(a),stability:W(a.storageStrength,void 0),lastAccessed:a.lastAccessedAt??a.updatedAt??a.createdAt,suppression:+!!v,tags:[e.date,...a.tags??[]],type:a.nodeType??`memory`}),i.push({id:`timeline:${e.date}:${a.id}`,memoryId:a.id,day:e.date,dayIndex:t,nodeIndex:y,angle:b,radius:S,retention:g,validFrom:C,transactionAt:w,suppressed:v,rewritten:_,label:T,provenance:E}),(_||v)&&c.push({source:K(`event`,`${a.id}:${_?`updated`:`suppressed`}:${w}`),type:v?`MemorySuppressed`:`MemoryUpdated`,targetIndex:y,frame:45+t*10+u,energy:v?1:.65}),h.length>0){s.push({source:K(`receipt`,`memory-audit:${a.id}`),label:`audit ${a.id.slice(0,8)} · ${h.length} events`,nodeIndices:[y]});for(let e of h.slice(0,8))c.push({source:K(`event`,`${a.id}:${e.action}:${e.timestamp}`),type:`Audit:${e.action}`,targetIndex:y,frame:70+t*12,energy:.4+Math.abs(W(e.new_value,0)-W(e.old_value,0))})}});let h=l.length?f/l.length:0,g=q(`day.${e.date}.count`,e.count);a.push({id:`timeline-day:${e.date}`,date:e.date,index:t,count:e.count,radius:o,retention:h,updatedCount:p,suppressedCount:m,memoryIndices:d,provenance:g}),s.push({source:g,label:`${e.date} · ${e.count} memories`,nodeIndices:d})});for(let e=1;e<i.length;e++)o.push({source:K(`pair`,`timeline-order:${i[e-1].memoryId}:${i[e].memoryId}`),sourceIndex:i[e-1].nodeIndex,targetIndex:i[e].nodeIndex,weight:.12,kind:`bitemporal-order`});let f=Object.entries(n).map(([e,t])=>({memoryId:e,events:t})),p=W(e.totalMemories,r.length),m={organ:`timeline`,nodes:r,edges:o,events:c,receipts:s,scalars:{totalMemories:p,dayCount:l.length,cellCount:i.length,updatedCount:c.filter(e=>e.type===`MemoryUpdated`||e.type===`Audit:edited`||e.type===`Audit:reconsolidated`).length,suppressedCount:c.filter(e=>e.type===`MemorySuppressed`||e.type===`Audit:suppressed`).length,maxDayCount:d},alive:i.length>0,rings:a,cells:i,audits:f,raw:{days:t,audits:n}};return q(`totalMemories`,p),m}var se=y(`<button type="button"> </button>`),Z=y(`<small class="svelte-bqsng9"> </small>`),ce=y(`<p class="state-line svelte-bqsng9">Weaving the live memory history…</p>`),le=y(`<p class="state-line error svelte-bqsng9"> </p>`),ue=y(`<p class="state-line svelte-bqsng9"> </p>`),de=y(`<button type="button"><span class="svelte-bqsng9"> </span><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>`),fe=y(`<div class="day-rows svelte-bqsng9"></div>`),pe=y(`<p class="state-line svelte-bqsng9">Loading this memory’s audit…</p>`),me=y(`<p class="state-line svelte-bqsng9">No audit events returned for this record.</p>`),he=y(`<li class="svelte-bqsng9"><strong class="svelte-bqsng9"> </strong><span class="svelte-bqsng9"> </span><!></li>`),ge=y(`<ol class="svelte-bqsng9"></ol>`),_e=y(`<p class="eyebrow svelte-bqsng9">TIME-SLICE RECEIPT</p> <h2 class="svelte-bqsng9"> </h2> <dl class="receipt-metrics svelte-bqsng9"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memory ID</dt><dd class="svelte-bqsng9"><code class="svelte-bqsng9"> </code></dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Valid time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Transaction time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <h3 class="svelte-bqsng9">Audit events</h3> <!>`,1),ve=y(`<p class="eyebrow svelte-bqsng9">DATE SLICE</p><h2 class="svelte-bqsng9"> </h2><p class="slice-summary svelte-bqsng9"> </p>`,1),ye=y(`<p class="eyebrow svelte-bqsng9">FIELD IS LIVE</p><h2 class="svelte-bqsng9">Choose a ring, date, or memory.</h2><p class="slice-summary svelte-bqsng9">The field shows growth. This panel makes the evidence legible.</p>`,1),be=y(`<button type="button"><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>`),xe=y(`<section class="memory-strip glass-panel svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span> </span><strong class="svelte-bqsng9"> </strong></div> <div class="memory-buttons svelte-bqsng9"></div></section>`),Se=y(`<!> <main class="timeline-shell svelte-bqsng9"><header class="timeline-head svelte-bqsng9"><div><p class="eyebrow svelte-bqsng9">BITEMPORAL MEMORY HISTORY</p> <h1 class="svelte-bqsng9">Watch memory grow. Inspect every change.</h1> <p class="lede svelte-bqsng9">The rings are real valid-time history. Choose a date or a memory to open its transaction-time receipt.</p></div> <div class="range-control svelte-bqsng9" aria-label="Timeline range"><span class="svelte-bqsng9">TIME WINDOW</span> <!> <button type="button">REWRITTEN</button></div></header> <dl class="vitals svelte-bqsng9" aria-label="Timeline metrics"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memories</dt><dd class="svelte-bqsng9"> </dd><!></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Rewritten</dt><dd class="svelte-bqsng9"> </dd><!></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Calendar slices</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Average retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <section class="timeline-grid svelte-bqsng9"><div class="glass-panel day-list svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span>VALID-TIME SLICES</span><strong class="svelte-bqsng9"> </strong></div> <!></div> <aside class="glass-panel receipt svelte-bqsng9" aria-live="polite"><!></aside></section> <!></main>`,1);function Ce(c,y){E(y,!0);let te=[7,14,30,90,365],O=_(l([])),k=_(!0),A=_(null),j=_(14),M=_(!1),N=_(null),P=_(null),F=_(!1),I=_(l({})),L=_(l({})),R=_(0),z=_(0),B=_(!1);w(()=>void V());async function V(){e(k,!0),e(A,null);try{let t=await ee.timeline(o(j),500);e(O,t.timeline,!0);let n=t.timeline.reduce((e,t)=>e+t.count,0);e(R,t.totalMemories??n,!0),e(z,t.returned??n,!0),e(B,t.truncated??o(R)>o(z),!0),o(N)&&!t.timeline.some(e=>e.date===o(N))&&(e(N,null),e(P,null))}catch(t){e(O,[],!0),e(R,0),e(z,0),e(B,!1),e(A,t instanceof Error?t.message:`Failed to load timeline`,!0)}finally{e(k,!1)}}async function U(t){t!==o(j)&&(e(j,t,!0),e(N,null),e(P,null),await V())}async function W(t){if(!(o(I)[t]||o(L)[t])){e(F,!0);try{let n=await ee.memoryAudit(t,100);e(I,{...o(I),[t]:n.events},!0)}catch(n){e(L,{...o(L),[t]:n instanceof Error?n.message:`Failed to load memory audit`},!0)}finally{e(F,!1)}}}function G(t){e(N,t,!0),e(P,null)}function K(t,n){e(N,n,!0),e(P,t.id,!0),W(t.id)}let q=T(()=>o(M)?o(O).map(e=>({...e,memories:e.memories.filter(H),count:e.memories.filter(H).length})).filter(e=>e.count>0):o(O)),J=T(()=>o(q).flatMap(e=>e.memories)),Y=T(()=>o(q).reduce((e,t)=>e+t.count,0)),ae=T(()=>o(q).flatMap(e=>e.memories).filter(H).length),X=T(()=>o(M)?o(Y):o(R)),Ce=T(()=>o(J).length?o(J).reduce((e,t)=>e+(t.retentionStrength??0),0)/o(J).length:0),Q=T(()=>o(q).find(e=>e.date===o(N))??null),$=T(()=>o(J).find(e=>e.id===o(P))??null),we=T(()=>o(P)?o(I)[o(P)]??[]:[]),Te=T(()=>o(P)?o(L)[o(P)]??null:null),Ee=T(()=>oe({days:o(O),totalMemories:o(Y),audits:o(I)}));function De(e,t){return re(e,t)}function Oe(e){if(e.kind===`timeline-cell`){let t=e.payload,n=o(J).find(e=>e.id===t.memoryId);n&&K(n,t.day)}else if(e.kind===`timeline-ring`){let t=e.payload;G(t.date)}}function ke(e){return e?new Date(e).toLocaleString():`Not recorded`}var Ae=Se();g(`bqsng9`,e=>{r(()=>{a.title=`Memory Timeline · Vestige`})});var je=p(Ae);{let e=T(()=>`timeline-growth-rings:${o(j)}:${o(Y)}`);ne(je,{organ:`timeline`,get seed(){return o(e)},get scene(){return o(Ee)},passes:De,loading:!1,get error(){return o(A)},emptyLabel:`NO MEMORY GROWTH RINGS IN THIS WINDOW`,onpick:Oe})}var Me=m(je,2),Ne=i(Me),Pe=m(i(Ne),2),Fe=m(i(Pe),2);D(Fe,17,()=>te,u,(e,t)=>{var r=se();let i;var a=f(r);d(()=>{v(r,`aria-pressed`,o(j)===o(t)),i=h(r,1,`svelte-bqsng9`,null,i,{active:o(j)===o(t)}),s(a,`${o(t)??``}D`)}),n(`click`,r,()=>U(o(t))),b(e,r)});var Ie=m(Fe,2);let Le;S(Pe),S(Ne);var Re=m(Ne,2),ze=i(Re),Be=m(i(ze)),Ve=f(Be,!0),He=m(Be),Ue=e=>{var t=Z(),n=f(t);d(()=>s(n,`newest ${o(z)??``} shown`)),b(e,t)};t(He,e=>{o(B)&&!o(M)&&e(Ue)}),S(ze);var We=m(ze,2),Ge=m(i(We)),Ke=f(Ge,!0),qe=m(Ge),Je=e=>{var t=Z(),n=f(t);d(()=>s(n,`in the newest ${o(z)??``}`)),b(e,t)};t(qe,e=>{o(B)&&e(Je)}),S(We);var Ye=m(We,2),Xe=m(i(Ye)),Ze=f(Xe,!0);S(Ye);var Qe=m(Ye,2),$e=m(i(Qe)),et=f($e);S(Qe),S(Re);var tt=m(Re,2),nt=i(tt),rt=i(nt),it=m(i(rt)),at=f(it);S(rt);var ot=m(rt,2),st=e=>{var t=ce();b(e,t)},ct=e=>{var t=le(),n=f(t,!0);d(()=>s(n,o(A))),b(e,t)},lt=e=>{var t=ue(),n=f(t,!0);d(()=>s(n,o(M)?`No rewritten memories in this window.`:`No memory growth in this window.`)),b(e,t)},ut=e=>{var t=fe();D(t,21,()=>o(q),e=>e.date,(e,t)=>{var r=de();let a;var c=i(r),l=f(c,!0),u=m(c),p=f(u,!0),g=m(u),_=f(g);S(r),d(e=>{a=h(r,1,`svelte-bqsng9`,null,a,{active:o(N)===o(t).date}),s(l,o(t).date),s(p,o(t).count),s(_,`${e??``}% retained`)},[()=>Math.round(o(t).memories.reduce((e,t)=>e+t.retentionStrength,0)/Math.max(1,o(t).memories.length)*100)]),n(`click`,r,()=>G(o(t).date)),b(e,r)}),S(t),b(e,t)};t(ot,e=>{o(k)?e(st):o(A)?e(ct,1):o(q).length===0?e(lt,2):e(ut,-1)}),S(nt);var dt=m(nt,2),ft=i(dt),pt=e=>{var n=_e(),r=m(p(n),2),a=f(r,!0),c=m(r,2),l=i(c),h=m(i(l)),g=i(h),_=f(g,!0);S(h),S(l);var v=m(l,2),y=m(i(v)),x=f(y,!0);S(v);var C=m(v,2),w=m(i(C)),T=f(w,!0);S(C);var E=m(C,2),ee=m(i(E)),te=f(ee);S(E),S(c);var O=m(c,4),k=e=>{var t=pe();b(e,t)},ne=e=>{var t=ue(),n=f(t);d(()=>s(n,`Audit history unavailable: ${o(Te)??``}`)),b(e,t)},A=e=>{var t=me();b(e,t)},j=e=>{var t=ge();D(t,21,()=>o(we).slice(0,12),u,(e,t)=>{var n=he(),r=i(n),a=f(r,!0),c=m(r),l=f(c,!0),p=m(c);D(p,17,()=>ie(o(t)),u,(e,t)=>{var n=Z(),r=f(n,!0);d(()=>s(r,o(t))),b(e,n)}),S(n),d(e=>{s(a,o(t).action),s(l,e)},[()=>ke(o(t).timestamp)]),b(e,n)}),S(t),b(e,t)};t(O,e=>{o(F)?e(k):o(Te)?e(ne,1):o(we).length===0?e(A,2):e(j,-1)}),d((e,t,n)=>{s(a,o($).content),s(_,o($).id),s(x,e),s(T,t),s(te,`${n??``}%`)},[()=>ke(o($).validFrom??o($).createdAt),()=>ke(o($).updatedAt),()=>Math.round(o($).retentionStrength*100)]),b(e,n)},mt=e=>{var t=ve(),n=m(p(t)),r=f(n,!0),i=m(n),a=f(i);d(()=>{s(r,o(Q).date),s(a,`${o(Q).count??``} memories entered this valid-time slice. Select one below to inspect its receipt.`)}),b(e,t)},ht=e=>{var t=ye();C(2),b(e,t)};t(ft,e=>{o($)?e(pt):o(Q)?e(mt,1):e(ht,-1)}),S(dt),S(tt);var gt=m(tt,2),_t=e=>{var t=xe(),r=i(t),a=i(r),c=f(a),l=m(a),u=f(l);S(r);var p=m(r,2);D(p,21,()=>o(Q).memories.slice(0,20),e=>e.id,(e,t)=>{var r=be();let a;var c=i(r),l=f(c,!0),u=m(c),p=f(u);S(r),d((e,n)=>{a=h(r,1,`svelte-bqsng9`,null,a,{active:o(P)===o(t).id}),s(l,o(t).content),s(p,`${e??``} · ${n??``}% retention`)},[()=>o(t).id.slice(0,8),()=>Math.round(o(t).retentionStrength*100)]),n(`click`,r,()=>K(o(t),o(Q).date)),b(e,r)}),S(p),S(t),d(()=>{s(c,`MEMORIES IN ${o(Q).date??``}`),s(u,`${o(Q).memories.length??``} RECORDS`)}),b(e,t)};t(gt,e=>{o(Q)&&e(_t)}),S(Me),d(e=>{v(Ie,`aria-pressed`,o(M)),Le=h(Ie,1,`svelte-bqsng9`,null,Le,{active:o(M)}),s(Ve,o(X)),s(Ke,o(ae)),s(Ze,o(O).length),s(et,`${e??``}%`),s(at,`${o(j)??``} DAYS`)},[()=>Math.round(o(Ce)*100)]),n(`click`,Ie,()=>e(M,!o(M))),b(c,Ae),x()}c([`click`]);export{Ce as component};
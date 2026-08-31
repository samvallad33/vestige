var wt=Object.defineProperty;var St=(c,e,i)=>e in c?wt(c,e,{enumerable:!0,configurable:!0,writable:!0,value:i}):c[e]=i;var M=(c,e,i)=>St(c,typeof e!="symbol"?e+"":e,i);import"../chunks/Bzak7iHL.js";import{d as Mt,o as Pt,s as _,b as De}from"../chunks/GD4hRtFg.js";import{p as Gt,d as X,e as Qe,h as Ie,t as z,a as B,b as Bt,i as qt,g as t,f as q,u as $,j as m,c as o,s as S,$ as At,r as a,n as Lt}from"../chunks/DEZxQDp-.js";import{i as be}from"../chunks/Co_hMTTH.js";import{e as ye,i as Xe,s as Rt,a as Oe}from"../chunks/Dtd1z3qK.js";import{h as Tt}from"../chunks/DH8OEHkH.js";import{a as je}from"../chunks/DOaVlKeo.js";import{R as Dt}from"../chunks/DS361EOd.js";import{r as Ee,M as It,R as Je}from"../chunks/lT7bFEJw.js";const _e="rgba16float",xe=768,we=96,Ce=16,Ue=12,Ke=`
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
`,Ot=`
${Ke}

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
`,Et=`
${Ke}

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
`,Ct=`
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
`;class Ut{constructor(e,i){M(this,"engine");M(this,"scene",null);M(this,"resources",null);M(this,"sampler",null);M(this,"splatBindLayout",null);M(this,"blurBindLayout",null);M(this,"membraneBindLayout",null);M(this,"splatPipeline",null);M(this,"blurPipeline",null);M(this,"membranePipeline",null);M(this,"cellPipeline",null);M(this,"ringPipeline",null);M(this,"cellCount",0);M(this,"ringCount",0);M(this,"selectedId",null);M(this,"cellGeometry",[]);M(this,"ringGeometry",[]);this.engine=e,this.uploadScene(i)}uploadScene(e){this.scene=e,this.buildGeometry();const i=this.engine.gpuDevice;i&&(this.ensurePipelines(i),this.ensureResources(i),this.uploadBuffers(i))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const i=Ve(e,"timeline-growth-rings-splat-wgsl",Ot),n=Ve(e,"timeline-growth-rings-blur-wgsl",Ct),r=Ve(e,"timeline-growth-rings-membrane-wgsl",Et);this.splatBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-splat-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const u=e.createPipelineLayout({label:"timeline-growth-rings-splat-layout",bindGroupLayouts:[this.splatBindLayout]}),v=e.createPipelineLayout({label:"timeline-growth-rings-blur-layout",bindGroupLayouts:[this.blurBindLayout]}),y=e.createPipelineLayout({label:"timeline-growth-rings-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"}),this.splatPipeline=e.createRenderPipeline({label:"timeline-field-additive-splat",layout:u,vertex:{module:i,entryPoint:"vs_splat"},fragment:{module:i,entryPoint:"fs_splat",targets:[{format:_e,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"timeline-field-blur-render-pass",layout:v,vertex:{module:n,entryPoint:"vs_fullscreen"},fragment:{module:n,entryPoint:"fs_blur",targets:[{format:_e}]},primitive:{topology:"triangle-list"}});const d={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.membranePipeline=e.createRenderPipeline({label:"timeline-bitemporal-membrane",layout:y,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-list"}}),this.ringPipeline=e.createRenderPipeline({label:"timeline-valid-time-rings",layout:u,vertex:{module:i,entryPoint:"vs_ring"},fragment:{module:i,entryPoint:"fs_ring",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-strip"}}),this.cellPipeline=e.createRenderPipeline({label:"timeline-memory-cells",layout:u,vertex:{module:i,entryPoint:"vs_cell"},fragment:{module:i,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var U,W,L,R,Y,se;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const i=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),n=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==i||this.resources.fieldSize[1]!==n;let u=(U=this.resources)==null?void 0:U.cellBuffer,v=(W=this.resources)==null?void 0:W.ringBuffer,y=(L=this.resources)==null?void 0:L.blurHBuffer,d=(R=this.resources)==null?void 0:R.blurVBuffer;if(u||(u=e.createBuffer({label:"timeline-cells",size:xe*Ce*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),v||(v=e.createBuffer({label:"timeline-rings",size:we*Ue*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),y||(y=e.createBuffer({label:"timeline-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(y,0,new Float32Array([1,0,0,0]))),d||(d=e.createBuffer({label:"timeline-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(d,0,new Float32Array([0,1,0,0]))),!r&&this.resources)return;(Y=this.resources)==null||Y.fieldA.destroy(),(se=this.resources)==null||se.fieldB.destroy();const p=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,x=e.createTexture({label:"timeline-field-a-rgba16float",size:[i,n],format:_e,usage:p}),f=e.createTexture({label:"timeline-field-b-rgba16float",size:[i,n],format:_e,usage:p}),A=x.createView(),te=f.createView(),le=e.createBindGroup({label:"timeline-growth-rings-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:u}},{binding:2,resource:{buffer:v}}]}),g=e.createBindGroup({label:"timeline-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:A},{binding:2,resource:{buffer:y}}]}),P=e.createBindGroup({label:"timeline-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:te},{binding:2,resource:{buffer:d}}]}),H=e.createBindGroup({label:"timeline-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:A}]});this.resources={cellBuffer:u,ringBuffer:v,blurHBuffer:y,blurVBuffer:d,splatBindGroup:le,blurHBindGroup:g,blurVBindGroup:P,membraneBindGroup:H,fieldA:x,fieldB:f,fieldAView:A,fieldBView:te,fieldSize:[i,n]}}buildGeometry(){var i,n;const e=((i=this.scene)==null?void 0:i.cells)??[];this.cellGeometry=e.slice(0,xe).map(r=>({cell:r,x:Math.cos(r.angle)*r.radius,y:Math.sin(r.angle)*r.radius,r:.018+r.retention*.016})),this.ringGeometry=(((n=this.scene)==null?void 0:n.rings)??[]).slice(0,we).map(r=>({ring:r,r:r.radius}))}uploadBuffers(e){var v,y,d;if(!this.resources)return;const i=new Float32Array(xe*Ce);this.cellCount=Math.min(xe,this.cellGeometry.length);const n=Math.max(1,this.ringGeometry.length-1);for(let p=0;p<this.cellCount;p++){const x=this.cellGeometry[p],f=x.cell,A=this.selectedId===f.id||this.selectedId===f.memoryId?1:0;i.set([x.x,x.y,x.r,f.radius,f.retention,f.rewritten?1:0,f.suppressed?1:0,((y=(v=this.scene)==null?void 0:v.raw.audits[f.memoryId])==null?void 0:y.length)??0,f.dayIndex/n,Date.parse(f.transactionAt||f.validFrom||"")/864e11||0,f.dayIndex,p,A,0,0,0],p*Ce)}this.ringCount=Math.min(we,this.ringGeometry.length);const r=new Float32Array(we*Ue),u=Math.max(1,((d=this.scene)==null?void 0:d.scalars.maxDayCount)??1);for(let p=0;p<this.ringCount;p++){const x=this.ringGeometry[p],f=x.ring,A=this.selectedId===f.id||this.selectedId===f.date?1:0;r.set([x.r,f.count/u,f.retention,f.index,f.updatedCount,f.suppressedCount,p/Math.max(1,this.ringCount),A,f.memoryIndices.length,p,0,0],p*Ue)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.ringCount,e.queue.writeBuffer(this.resources.cellBuffer,0,i),e.queue.writeBuffer(this.resources.ringBuffer,0,r)}compute(e){const i=this.engine.gpuDevice;if(!i||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(i);const n=this.resources,r=e.beginRenderPass({label:"timeline-field-splat-pass",colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});r.setPipeline(this.splatPipeline),r.setBindGroup(0,n.splatBindGroup),r.draw(6,this.cellCount),r.end();const u=e.beginRenderPass({label:"timeline-field-blur-h-pass",colorAttachments:[{view:n.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});u.setPipeline(this.blurPipeline),u.setBindGroup(0,n.blurHBindGroup),u.draw(6,1),u.end();const v=e.beginRenderPass({label:"timeline-field-blur-v-pass",colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});v.setPipeline(this.blurPipeline),v.setBindGroup(0,n.blurVBindGroup),v.draw(6,1),v.end()}render(e){!this.resources||!this.membranePipeline||!this.ringPipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.ringCount>0&&(e.setPipeline(this.ringPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(192,this.ringCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.draw(6,this.cellCount)))}ringSpin(e){const i=this.engine.params[10]||0,n=.045+e*.1;return i*n}orbitCpu(e,i,n,r){const u=Math.hypot(e,i);if(u<1e-4)return{x:e,y:i};const v=this.engine.params[10]||0,d=Math.atan2(i,e)+this.ringSpin(r)+Math.sin(v*.6+n*6.283)*.02,p=u*(1+.016*Math.sin(v*1.1+n*6.283));return{x:Math.cos(d)*p,y:Math.sin(d)*p}}pickAt(e,i){const n=Math.max(1,this.ringGeometry.length-1);let r=null,u=1/0;for(let d=0;d<this.cellGeometry.length;d++){const p=this.cellGeometry[d],x=p.cell.dayIndex/n,f=this.orbitCpu(p.x,p.y,d,x),A=Math.hypot(e-f.x,i-f.y);A<=Math.max(.045,p.r*1.8)&&A<u&&(r={id:p.cell.id,kind:"timeline-cell",index:d,payload:p.cell},u=A)}if(r)return this.selectedId=r.id,r;const v=Math.hypot(e,i),y=this.engine.params[10]||0;for(let d=0;d<this.ringGeometry.length;d++){const p=this.ringGeometry[d],x=n>0?d/n:0,f=p.r*(1+.016*Math.sin(y*1.1+x*6.283));if(Math.abs(v-f)<=.03)return this.selectedId=p.ring.id,{id:p.ring.id,kind:"timeline-ring",index:d,payload:p.ring}}return null}dispose(){var e,i,n,r,u,v;(e=this.resources)==null||e.cellBuffer.destroy(),(i=this.resources)==null||i.ringBuffer.destroy(),(n=this.resources)==null||n.blurHBuffer.destroy(),(r=this.resources)==null||r.blurVBuffer.destroy(),(u=this.resources)==null||u.fieldA.destroy(),(v=this.resources)==null||v.fieldB.destroy(),this.resources=null}}function Ve(c,e,i){c.pushErrorScope("validation");const n=c.createShaderModule({label:e,code:i});return n.getCompilationInfo().then(r=>{for(const u of r.messages)console.error(`[observatory] ${e} WGSL ${u.type} ${u.lineNum}:${u.linePos} ${u.message}`)}),c.popErrorScope().then(r=>{r&&console.error(`[observatory] ${e} shader module validation: ${r.message}`)}),n}function Vt(c,e){return Ee(It.blackwater),Ee(Je.healthy),Ee(Je.luciferin),[new Ut(c,e)]}function Se(c,e=""){return typeof c=="string"?c:c==null?e:String(c)}function ee(c,e=0){return typeof c=="number"&&Number.isFinite(c)?c:e}function Ze(c){return Math.max(0,Math.min(1,c))}function pe(c,e,i){return{kind:c,id:e||`${c}:unknown`}}function Ft(c,e){return{kind:"scalar",id:`timeline.${c}`,scalar:{name:c,value:e}}}function et(c){return Ze(ee(c.retentionStrength,0))}function kt(c){return Ze(ee(c.combinedScore??c.retentionStrength,et(c)))}function Nt(c,e){return e[c.id]??[]}function Fe(c,e){return c.some(i=>i.action===e)}function zt(c){const e=c.days??[],i=c.audits??{},n=[],r=[],u=[],v=[],y=[],d=[],p=e.filter(g=>g.count>0||g.memories.length>0),x=Math.max(1,p.length),f=Math.max(1,...p.map(g=>g.count||g.memories.length));p.forEach((g,P)=>{const H=.16+P/Math.max(1,x-1)*.7,U=g.memories??[],W=[];let L=0,R=0,Y=0;U.forEach((h,j)=>{const V=Nt(h,i),ie=et(h),J=Se(h.updatedAt)!==Se(h.createdAt)||Fe(V,"edited")||Fe(V,"reconsolidated"),C=Fe(V,"suppressed")||ee(h.suppression_count,0)>0;J&&(R+=1),C&&(Y+=1),L+=ie;const F=n.length;W.push(F);const Me=(j+.5)/Math.max(1,U.length)*Math.PI*2+P*.37,oe=(j%5-2)*.008,ce=H+oe,de=Se(h.validFrom??h.createdAt,g.date),me=Se(h.updatedAt??h.createdAt,de),re=h.content||h.id.slice(0,8),ue=pe("memory",h.id);if(n.push({source:ue,index:F,label:re,retention:ie,trust:kt(h),stability:ee(h.storageStrength,void 0),lastAccessed:h.lastAccessedAt??h.updatedAt??h.createdAt,suppression:C?1:0,tags:[g.date,...h.tags??[]],type:h.nodeType??"memory"}),r.push({id:`timeline:${g.date}:${h.id}`,memoryId:h.id,day:g.date,dayIndex:P,nodeIndex:F,angle:Me,radius:ce,retention:ie,validFrom:de,transactionAt:me,suppressed:C,rewritten:J,label:re,provenance:ue}),(J||C)&&d.push({source:pe("event",`${h.id}:${J?"updated":"suppressed"}:${me}`),type:C?"MemorySuppressed":"MemoryUpdated",targetIndex:F,frame:45+P*10+j,energy:C?1:.65}),V.length>0){y.push({source:pe("receipt",`memory-audit:${h.id}`),label:`audit ${h.id.slice(0,8)} · ${V.length} events`,nodeIndices:[F]});for(const K of V.slice(0,8))d.push({source:pe("event",`${h.id}:${K.action}:${K.timestamp}`),type:`Audit:${K.action}`,targetIndex:F,frame:70+P*12,energy:.4+Math.abs(ee(K.new_value,0)-ee(K.old_value,0))})}});const se=U.length?L/U.length:0,ge=Ft(`day.${g.date}.count`,g.count);u.push({id:`timeline-day:${g.date}`,date:g.date,index:P,count:g.count,radius:H,retention:se,updatedCount:R,suppressedCount:Y,memoryIndices:W,provenance:ge}),y.push({source:ge,label:`${g.date} · ${g.count} memories`,nodeIndices:W})});for(let g=1;g<r.length;g++)v.push({source:pe("pair",`timeline-order:${r[g-1].memoryId}:${r[g].memoryId}`),sourceIndex:r[g-1].nodeIndex,targetIndex:r[g].nodeIndex,weight:.12,kind:"bitemporal-order"});const A=Object.entries(i).map(([g,P])=>({memoryId:g,events:P})),te=ee(c.totalMemories,n.length);return{organ:"timeline",nodes:n,edges:v,events:d,receipts:y,scalars:{totalMemories:te,dayCount:p.length,cellCount:r.length,updatedCount:d.filter(g=>g.type==="MemoryUpdated"||g.type==="Audit:edited"||g.type==="Audit:reconsolidated").length,suppressedCount:d.filter(g=>g.type==="MemorySuppressed"||g.type==="Audit:suppressed").length,maxDayCount:f},alive:r.length>0,rings:u,cells:r,audits:A,raw:{days:e,audits:i}}}var $t=q('<button type="button"> </button>'),Ht=q('<p class="state-line svelte-bqsng9">Weaving the live memory history…</p>'),Wt=q('<p class="state-line error svelte-bqsng9"> </p>'),Yt=q('<p class="state-line svelte-bqsng9">No memory growth in this window.</p>'),Qt=q('<button type="button"><span class="svelte-bqsng9"> </span><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>'),Xt=q('<div class="day-rows svelte-bqsng9"></div>'),jt=q('<p class="state-line svelte-bqsng9">Loading this memory’s audit…</p>'),Jt=q('<p class="state-line svelte-bqsng9">No audit events returned for this record.</p>'),Kt=q('<li class="svelte-bqsng9"><strong class="svelte-bqsng9"> </strong><span class="svelte-bqsng9"> </span></li>'),Zt=q('<ol class="svelte-bqsng9"></ol>'),es=q('<p class="eyebrow svelte-bqsng9">TIME-SLICE RECEIPT</p> <h2 class="svelte-bqsng9"> </h2> <dl class="receipt-metrics svelte-bqsng9"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memory ID</dt><dd class="svelte-bqsng9"><code class="svelte-bqsng9"> </code></dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Valid time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Transaction time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <h3 class="svelte-bqsng9">Audit events</h3> <!>',1),ts=q('<p class="eyebrow svelte-bqsng9">DATE SLICE</p><h2 class="svelte-bqsng9"> </h2><p class="slice-summary svelte-bqsng9"> </p>',1),ss=q('<p class="eyebrow svelte-bqsng9">FIELD IS LIVE</p><h2 class="svelte-bqsng9">Choose a ring, date, or memory.</h2><p class="slice-summary svelte-bqsng9">The field shows growth. This panel makes the evidence legible.</p>',1),is=q('<button type="button"><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>'),rs=q('<section class="memory-strip glass-panel svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span> </span><strong class="svelte-bqsng9"> </strong></div> <div class="memory-buttons svelte-bqsng9"></div></section>'),ns=q('<!> <main class="timeline-shell svelte-bqsng9"><header class="timeline-head svelte-bqsng9"><div><p class="eyebrow svelte-bqsng9">BITEMPORAL MEMORY HISTORY</p> <h1 class="svelte-bqsng9">Watch memory grow. Inspect every change.</h1> <p class="lede svelte-bqsng9">The rings are real valid-time history. Choose a date or a memory to open its transaction-time receipt.</p></div> <div class="range-control svelte-bqsng9" aria-label="Timeline range"><span class="svelte-bqsng9">TIME WINDOW</span> <!></div></header> <dl class="vitals svelte-bqsng9" aria-label="Timeline metrics"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memories</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Rewritten</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Calendar slices</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Average retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <section class="timeline-grid svelte-bqsng9"><div class="glass-panel day-list svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span>VALID-TIME SLICES</span><strong class="svelte-bqsng9"> </strong></div> <!></div> <aside class="glass-panel receipt svelte-bqsng9" aria-live="polite"><!></aside></section> <!></main>',1);function fs(c,e){Gt(e,!0);const i=[7,14,30,90,365];let n=X(Qe([])),r=X(!0),u=X(null),v=X(14),y=X(null),d=X(null),p=X(!1),x=X(Qe({}));Pt(()=>void f());async function f(){S(r,!0),S(u,null);try{const s=await je.timeline(t(v),500);S(n,s.timeline,!0),t(y)&&!s.timeline.some(l=>l.date===t(y))&&(S(y,null),S(d,null))}catch(s){S(n,[],!0),S(u,s instanceof Error?s.message:"Failed to load timeline",!0)}finally{S(r,!1)}}async function A(s){s!==t(v)&&(S(v,s,!0),S(y,null),S(d,null),await f())}async function te(s){if(!t(x)[s]){S(p,!0);try{const l=await je.memoryAudit(s,100);S(x,{...t(x),[s]:l.events},!0)}catch(l){S(u,l instanceof Error?l.message:"Failed to load memory audit",!0)}finally{S(p,!1)}}}function le(s){S(y,s,!0),S(d,null)}function g(s,l){S(y,l,!0),S(d,s.id,!0),te(s.id)}const P=$(()=>t(n).flatMap(s=>s.memories)),H=$(()=>t(n).reduce((s,l)=>s+l.count,0)),U=$(()=>t(P).filter(s=>s.updatedAt!==s.createdAt).length),W=$(()=>t(P).length?t(P).reduce((s,l)=>s+(l.retentionStrength??0),0)/t(P).length:0),L=$(()=>t(n).find(s=>s.date===t(y))??null),R=$(()=>t(P).find(s=>s.id===t(d))??null),Y=$(()=>t(d)?t(x)[t(d)]??[]:[]),se=$(()=>zt({days:t(n),totalMemories:t(H),audits:t(x)}));function ge(s,l){return Vt(s,l)}function h(s){if(s.kind==="timeline-cell"){const l=s.payload,b=t(P).find(w=>w.id===l.memoryId);b&&g(b,l.day)}else if(s.kind==="timeline-ring"){const l=s.payload;le(l.date)}}function j(s){return s?new Date(s).toLocaleString():"Not recorded"}var V=ns();Tt("bqsng9",s=>{qt(()=>{At.title="Memory Timeline · Vestige"})});var ie=Ie(V);{let s=$(()=>`timeline-growth-rings:${t(v)}:${t(H)}`);Dt(ie,{organ:"timeline",get seed(){return t(s)},get scene(){return t(se)},passes:ge,loading:!1,get error(){return t(u)},emptyLabel:"NO MEMORY GROWTH RINGS IN THIS WINDOW",onpick:h})}var J=m(ie,2),C=o(J),F=m(o(C),2),Me=m(o(F),2);ye(Me,17,()=>i,Xe,(s,l)=>{var b=$t();let w;var G=o(b);a(b),z(()=>{Rt(b,"aria-pressed",t(v)===t(l)),w=Oe(b,1,"svelte-bqsng9",null,w,{active:t(v)===t(l)}),_(G,`${t(l)??""}D`)}),De("click",b,()=>A(t(l))),B(s,b)}),a(F),a(C);var oe=m(C,2),ce=o(oe),de=m(o(ce)),me=o(de,!0);a(de),a(ce);var re=m(ce,2),ue=m(o(re)),K=o(ue,!0);a(ue),a(re);var Pe=m(re,2),ke=m(o(Pe)),tt=o(ke,!0);a(ke),a(Pe);var Ne=m(Pe,2),ze=m(o(Ne)),st=o(ze);a(ze),a(Ne),a(oe);var Ge=m(oe,2),Be=o(Ge),qe=o(Be),$e=m(o(qe)),it=o($e);a($e),a(qe);var rt=m(qe,2);{var nt=s=>{var l=Ht();B(s,l)},at=s=>{var l=Wt(),b=o(l,!0);a(l),z(()=>_(b,t(u))),B(s,l)},lt=s=>{var l=Yt();B(s,l)},ot=s=>{var l=Xt();ye(l,21,()=>t(n),b=>b.date,(b,w)=>{var G=Qt();let T;var Q=o(G),Z=o(Q,!0);a(Q);var k=m(Q),D=o(k,!0);a(k);var I=m(k),ne=o(I);a(I),a(G),z(O=>{T=Oe(G,1,"svelte-bqsng9",null,T,{active:t(y)===t(w).date}),_(Z,t(w).date),_(D,t(w).count),_(ne,`${O??""}% retained`)},[()=>Math.round(t(w).memories.reduce((O,ae)=>O+ae.retentionStrength,0)/Math.max(1,t(w).memories.length)*100)]),De("click",G,()=>le(t(w).date)),B(b,G)}),a(l),B(s,l)};be(rt,s=>{t(r)?s(nt):t(u)?s(at,1):t(n).length===0?s(lt,2):s(ot,!1)})}a(Be);var He=m(Be,2),ct=o(He);{var dt=s=>{var l=es(),b=m(Ie(l),2),w=o(b,!0);a(b);var G=m(b,2),T=o(G),Q=m(o(T)),Z=o(Q),k=o(Z,!0);a(Z),a(Q),a(T);var D=m(T,2),I=m(o(D)),ne=o(I,!0);a(I),a(D);var O=m(D,2),ae=m(o(O)),ve=o(ae,!0);a(ae),a(O);var fe=m(O,2),he=m(o(fe)),Ae=o(he);a(he),a(fe),a(G);var vt=m(G,4);{var ft=E=>{var N=jt();B(E,N)},ht=E=>{var N=Jt();B(E,N)},bt=E=>{var N=Zt();ye(N,21,()=>t(Y).slice(0,8),Xe,(Le,We)=>{var Re=Kt(),Te=o(Re),yt=o(Te,!0);a(Te);var Ye=m(Te),_t=o(Ye,!0);a(Ye),a(Re),z(xt=>{_(yt,t(We).action),_(_t,xt)},[()=>j(t(We).timestamp)]),B(Le,Re)}),a(N),B(E,N)};be(vt,E=>{t(p)?E(ft):t(Y).length===0?E(ht,1):E(bt,!1)})}z((E,N,Le)=>{_(w,t(R).content),_(k,t(R).id),_(ne,E),_(ve,N),_(Ae,`${Le??""}%`)},[()=>j(t(R).validFrom??t(R).createdAt),()=>j(t(R).updatedAt),()=>Math.round(t(R).retentionStrength*100)]),B(s,l)},ut=s=>{var l=ts(),b=m(Ie(l)),w=o(b,!0);a(b);var G=m(b),T=o(G);a(G),z(()=>{_(w,t(L).date),_(T,`${t(L).count??""} memories entered this valid-time slice. Select one below to inspect its receipt.`)}),B(s,l)},pt=s=>{var l=ss();Lt(2),B(s,l)};be(ct,s=>{t(R)?s(dt):t(L)?s(ut,1):s(pt,!1)})}a(He),a(Ge);var gt=m(Ge,2);{var mt=s=>{var l=rs(),b=o(l),w=o(b),G=o(w);a(w);var T=m(w),Q=o(T);a(T),a(b);var Z=m(b,2);ye(Z,21,()=>t(L).memories.slice(0,20),k=>k.id,(k,D)=>{var I=is();let ne;var O=o(I),ae=o(O,!0);a(O);var ve=m(O),fe=o(ve);a(ve),a(I),z((he,Ae)=>{ne=Oe(I,1,"svelte-bqsng9",null,ne,{active:t(d)===t(D).id}),_(ae,t(D).content),_(fe,`${he??""} · ${Ae??""}% retention`)},[()=>t(D).id.slice(0,8),()=>Math.round(t(D).retentionStrength*100)]),De("click",I,()=>g(t(D),t(L).date)),B(k,I)}),a(Z),a(l),z(()=>{_(G,`MEMORIES IN ${t(L).date??""}`),_(Q,`${t(L).memories.length??""} RECORDS`)}),B(s,l)};be(gt,s=>{t(L)&&s(mt)})}a(J),z(s=>{_(me,t(H)),_(K,t(U)),_(tt,t(n).length),_(st,`${s??""}%`),_(it,`${t(v)??""} DAYS`)},[()=>Math.round(t(W)*100)]),B(c,V),Bt()}Mt(["click"]);export{fs as component};

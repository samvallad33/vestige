var Tt=Object.defineProperty;var Dt=(i,e,r)=>e in i?Tt(i,e,{enumerable:!0,configurable:!0,writable:!0,value:r}):i[e]=r;var M=(i,e,r)=>Dt(i,typeof e!="symbol"?e+"":e,r);import"../chunks/Bzak7iHL.js";import{d as It,o as Et,b as _e,s as y}from"../chunks/CNdOtqLU.js";import{p as Ot,d as W,e as Ze,h as Ne,t as U,a as P,b as Ct,i as Ut,g as t,s as w,f as B,u as k,j as v,c,$ as Ft,r as l,n as Nt}from"../chunks/Dw_4PDAU.js";import{i as xe}from"../chunks/C7WW_yYn.js";import{e as pe,s as we,i as ke}from"../chunks/S5IcwJEO.js";import{h as kt}from"../chunks/C7sd-K7m.js";import{s as et}from"../chunks/sbNTdgSE.js";import{a as tt}from"../chunks/B3oLbNAe.js";import{R as Vt}from"../chunks/kthyKc2E.js";import{r as Ve,M as $t,R as st}from"../chunks/B4HOM7z9.js";const Se="rgba16float",Me=768,Pe=96,$e=16,ze=12,rt=`
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
`,zt=`
${rt}

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
`,Ht=`
${rt}

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
`,Wt=`
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
`;class Yt{constructor(e,r){M(this,"engine");M(this,"scene",null);M(this,"resources",null);M(this,"sampler",null);M(this,"splatBindLayout",null);M(this,"blurBindLayout",null);M(this,"membraneBindLayout",null);M(this,"splatPipeline",null);M(this,"blurPipeline",null);M(this,"membranePipeline",null);M(this,"cellPipeline",null);M(this,"ringPipeline",null);M(this,"cellCount",0);M(this,"ringCount",0);M(this,"selectedId",null);M(this,"cellGeometry",[]);M(this,"ringGeometry",[]);this.engine=e,this.uploadScene(r)}uploadScene(e){this.scene=e,this.buildGeometry();const r=this.engine.gpuDevice;r&&(this.ensurePipelines(r),this.ensureResources(r),this.uploadBuffers(r))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const r=He(e,"timeline-growth-rings-splat-wgsl",zt),a=He(e,"timeline-growth-rings-blur-wgsl",Wt),n=He(e,"timeline-growth-rings-membrane-wgsl",Ht);this.splatBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-splat-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const p=e.createPipelineLayout({label:"timeline-growth-rings-splat-layout",bindGroupLayouts:[this.splatBindLayout]}),f=e.createPipelineLayout({label:"timeline-growth-rings-blur-layout",bindGroupLayouts:[this.blurBindLayout]}),_=e.createPipelineLayout({label:"timeline-growth-rings-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"}),this.splatPipeline=e.createRenderPipeline({label:"timeline-field-additive-splat",layout:p,vertex:{module:r,entryPoint:"vs_splat"},fragment:{module:r,entryPoint:"fs_splat",targets:[{format:Se,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"timeline-field-blur-render-pass",layout:f,vertex:{module:a,entryPoint:"vs_fullscreen"},fragment:{module:a,entryPoint:"fs_blur",targets:[{format:Se}]},primitive:{topology:"triangle-list"}});const d={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.membranePipeline=e.createRenderPipeline({label:"timeline-bitemporal-membrane",layout:_,vertex:{module:n,entryPoint:"vs_fullscreen"},fragment:{module:n,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-list"}}),this.ringPipeline=e.createRenderPipeline({label:"timeline-valid-time-rings",layout:p,vertex:{module:r,entryPoint:"vs_ring"},fragment:{module:r,entryPoint:"fs_ring",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-strip"}}),this.cellPipeline=e.createRenderPipeline({label:"timeline-memory-cells",layout:p,vertex:{module:r,entryPoint:"vs_cell"},fragment:{module:r,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:d}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var q,N,j,J,L,T;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const r=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),a=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),n=!this.resources||this.resources.fieldSize[0]!==r||this.resources.fieldSize[1]!==a;let p=(q=this.resources)==null?void 0:q.cellBuffer,f=(N=this.resources)==null?void 0:N.ringBuffer,_=(j=this.resources)==null?void 0:j.blurHBuffer,d=(J=this.resources)==null?void 0:J.blurVBuffer;if(p||(p=e.createBuffer({label:"timeline-cells",size:Me*$e*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),f||(f=e.createBuffer({label:"timeline-rings",size:Pe*ze*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),_||(_=e.createBuffer({label:"timeline-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(_,0,new Float32Array([1,0,0,0]))),d||(d=e.createBuffer({label:"timeline-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(d,0,new Float32Array([0,1,0,0]))),!n&&this.resources)return;(L=this.resources)==null||L.fieldA.destroy(),(T=this.resources)==null||T.fieldB.destroy();const u=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,x=e.createTexture({label:"timeline-field-a-rgba16float",size:[r,a],format:Se,usage:u}),g=e.createTexture({label:"timeline-field-b-rgba16float",size:[r,a],format:Se,usage:u}),G=x.createView(),se=g.createView(),ge=e.createBindGroup({label:"timeline-growth-rings-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:p}},{binding:2,resource:{buffer:f}}]}),m=e.createBindGroup({label:"timeline-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:G},{binding:2,resource:{buffer:_}}]}),R=e.createBindGroup({label:"timeline-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:se},{binding:2,resource:{buffer:d}}]}),F=e.createBindGroup({label:"timeline-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:G}]});this.resources={cellBuffer:p,ringBuffer:f,blurHBuffer:_,blurVBuffer:d,splatBindGroup:ge,blurHBindGroup:m,blurVBindGroup:R,membraneBindGroup:F,fieldA:x,fieldB:g,fieldAView:G,fieldBView:se,fieldSize:[r,a]}}buildGeometry(){var r,a;const e=((r=this.scene)==null?void 0:r.cells)??[];this.cellGeometry=e.slice(0,Me).map(n=>({cell:n,x:Math.cos(n.angle)*n.radius,y:Math.sin(n.angle)*n.radius,r:.018+n.retention*.016})),this.ringGeometry=(((a=this.scene)==null?void 0:a.rings)??[]).slice(0,Pe).map(n=>({ring:n,r:n.radius}))}uploadBuffers(e){var f,_,d;if(!this.resources)return;const r=new Float32Array(Me*$e);this.cellCount=Math.min(Me,this.cellGeometry.length);const a=Math.max(1,this.ringGeometry.length-1);for(let u=0;u<this.cellCount;u++){const x=this.cellGeometry[u],g=x.cell,G=this.selectedId===g.id||this.selectedId===g.memoryId?1:0;r.set([x.x,x.y,x.r,g.radius,g.retention,g.rewritten?1:0,g.suppressed?1:0,((_=(f=this.scene)==null?void 0:f.raw.audits[g.memoryId])==null?void 0:_.length)??0,g.dayIndex/a,Date.parse(g.transactionAt||g.validFrom||"")/864e11||0,g.dayIndex,u,G,0,0,0],u*$e)}this.ringCount=Math.min(Pe,this.ringGeometry.length);const n=new Float32Array(Pe*ze),p=Math.max(1,((d=this.scene)==null?void 0:d.scalars.maxDayCount)??1);for(let u=0;u<this.ringCount;u++){const x=this.ringGeometry[u],g=x.ring,G=this.selectedId===g.id||this.selectedId===g.date?1:0;n.set([x.r,g.count/p,g.retention,g.index,g.updatedCount,g.suppressedCount,u/Math.max(1,this.ringCount),G,g.memoryIndices.length,u,0,0],u*ze)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.ringCount,e.queue.writeBuffer(this.resources.cellBuffer,0,r),e.queue.writeBuffer(this.resources.ringBuffer,0,n)}compute(e){const r=this.engine.gpuDevice;if(!r||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(r);const a=this.resources,n=e.beginRenderPass({label:"timeline-field-splat-pass",colorAttachments:[{view:a.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(this.splatPipeline),n.setBindGroup(0,a.splatBindGroup),this.cellCount>0&&n.draw(6,this.cellCount),n.end();const p=e.beginRenderPass({label:"timeline-field-blur-h-pass",colorAttachments:[{view:a.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});p.setPipeline(this.blurPipeline),p.setBindGroup(0,a.blurHBindGroup),p.draw(6,1),p.end();const f=e.beginRenderPass({label:"timeline-field-blur-v-pass",colorAttachments:[{view:a.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});f.setPipeline(this.blurPipeline),f.setBindGroup(0,a.blurVBindGroup),f.draw(6,1),f.end()}render(e){!this.resources||!this.membranePipeline||!this.ringPipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.ringCount>0&&(e.setPipeline(this.ringPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(192,this.ringCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.draw(6,this.cellCount)))}ringSpin(e){const r=this.engine.params[10]||0,a=.045+e*.1;return r*a}orbitCpu(e,r,a,n){const p=Math.hypot(e,r);if(p<1e-4)return{x:e,y:r};const f=this.engine.params[10]||0,d=Math.atan2(r,e)+this.ringSpin(n)+Math.sin(f*.6+a*6.283)*.02,u=p*(1+.016*Math.sin(f*1.1+a*6.283));return{x:Math.cos(d)*u,y:Math.sin(d)*u}}pickAt(e,r){const a=Math.max(1,this.ringGeometry.length-1);let n=null,p=1/0;for(let d=0;d<this.cellGeometry.length;d++){const u=this.cellGeometry[d],x=u.cell.dayIndex/a,g=this.orbitCpu(u.x,u.y,d,x),G=Math.hypot(e-g.x,r-g.y);G<=Math.max(.045,u.r*1.8)&&G<p&&(n={id:u.cell.id,kind:"timeline-cell",index:d,payload:u.cell},p=G)}if(n)return this.selectedId=n.id,n;const f=Math.hypot(e,r),_=this.engine.params[10]||0;for(let d=0;d<this.ringGeometry.length;d++){const u=this.ringGeometry[d],x=a>0?d/a:0,g=u.r*(1+.016*Math.sin(_*1.1+x*6.283));if(Math.abs(f-g)<=.03)return this.selectedId=u.ring.id,{id:u.ring.id,kind:"timeline-ring",index:d,payload:u.ring}}return null}dispose(){var e,r,a,n,p,f;(e=this.resources)==null||e.cellBuffer.destroy(),(r=this.resources)==null||r.ringBuffer.destroy(),(a=this.resources)==null||a.blurHBuffer.destroy(),(n=this.resources)==null||n.blurVBuffer.destroy(),(p=this.resources)==null||p.fieldA.destroy(),(f=this.resources)==null||f.fieldB.destroy(),this.resources=null}}function He(i,e,r){i.pushErrorScope("validation");const a=i.createShaderModule({label:e,code:r});return a.getCompilationInfo().then(n=>{for(const p of n.messages)console.error(`[observatory] ${e} WGSL ${p.type} ${p.lineNum}:${p.linePos} ${p.message}`)}),i.popErrorScope().then(n=>{n&&console.error(`[observatory] ${e} shader module validation: ${n.message}`)}),a}function Qt(i,e){return Ve($t.blackwater),Ve(st.healthy),Ve(st.luciferin),[new Yt(i,e)]}function Xt(i,e){if(typeof i!="number"||typeof e!="number"||!Number.isFinite(i)||!Number.isFinite(e))return null;const r=Math.round(i*100),a=Math.round(e*100),n=a-r,p=n>0?"+":"";return`${r}% → ${a}% (${p}${n})`}function jt(i){var a,n;const e=[],r=Xt(i.old_value,i.new_value);return r&&e.push(r),(a=i.reason)!=null&&a.trim()&&e.push(i.reason.trim()),(n=i.triggered_by)!=null&&n.trim()&&e.push(`by ${i.triggered_by.trim()}`),e}function it(i){return!!(i.createdAt&&i.updatedAt&&i.createdAt!==i.updatedAt)}function Be(i,e=""){return typeof i=="string"?i:i==null?e:String(i)}function te(i,e=0){return typeof i=="number"&&Number.isFinite(i)?i:e}function nt(i){return Math.max(0,Math.min(1,i))}function me(i,e,r){return{kind:i,id:e||`${i}:unknown`}}function Jt(i,e){return{kind:"scalar",id:`timeline.${i}`,scalar:{name:i,value:e}}}function at(i){return nt(te(i.retentionStrength,0))}function Kt(i){return nt(te(i.combinedScore??i.retentionStrength,at(i)))}function Zt(i,e){return e[i.id]??[]}function We(i,e){return i.some(r=>r.action===e)}function es(i){const e=i.days??[],r=i.audits??{},a=[],n=[],p=[],f=[],_=[],d=[],u=e.filter(m=>m.count>0||m.memories.length>0),x=Math.max(1,u.length),g=Math.max(1,...u.map(m=>m.count||m.memories.length));u.forEach((m,R)=>{const F=.16+R/Math.max(1,x-1)*.7,q=m.memories??[],N=[];let j=0,J=0,L=0;q.forEach((b,ce)=>{const Y=Zt(b,r),K=at(b),Z=Be(b.updatedAt)!==Be(b.createdAt)||We(Y,"edited")||We(Y,"reconsolidated"),V=We(Y,"suppressed")||te(b.suppression_count,0)>0;Z&&(J+=1),V&&(L+=1),j+=K;const $=a.length;N.push($);const ue=(ce+.5)/Math.max(1,q.length)*Math.PI*2+R*.37,ve=(ce%5-2)*.008,fe=F+ve,ie=Be(b.validFrom??b.createdAt,m.date),de=Be(b.updatedAt??b.createdAt,ie),re=b.content||b.id.slice(0,8),ne=me("memory",b.id);if(a.push({source:ne,index:$,label:re,retention:K,trust:Kt(b),stability:te(b.storageStrength,void 0),lastAccessed:b.lastAccessedAt??b.updatedAt??b.createdAt,suppression:V?1:0,tags:[m.date,...b.tags??[]],type:b.nodeType??"memory"}),n.push({id:`timeline:${m.date}:${b.id}`,memoryId:b.id,day:m.date,dayIndex:R,nodeIndex:$,angle:ue,radius:fe,retention:K,validFrom:ie,transactionAt:de,suppressed:V,rewritten:Z,label:re,provenance:ne}),(Z||V)&&d.push({source:me("event",`${b.id}:${Z?"updated":"suppressed"}:${de}`),type:V?"MemorySuppressed":"MemoryUpdated",targetIndex:$,frame:45+R*10+ce,energy:V?1:.65}),Y.length>0){_.push({source:me("receipt",`memory-audit:${b.id}`),label:`audit ${b.id.slice(0,8)} · ${Y.length} events`,nodeIndices:[$]});for(const Q of Y.slice(0,8))d.push({source:me("event",`${b.id}:${Q.action}:${Q.timestamp}`),type:`Audit:${Q.action}`,targetIndex:$,frame:70+R*12,energy:.4+Math.abs(te(Q.new_value,0)-te(Q.old_value,0))})}});const T=q.length?j/q.length:0,oe=Jt(`day.${m.date}.count`,m.count);p.push({id:`timeline-day:${m.date}`,date:m.date,index:R,count:m.count,radius:F,retention:T,updatedCount:J,suppressedCount:L,memoryIndices:N,provenance:oe}),_.push({source:oe,label:`${m.date} · ${m.count} memories`,nodeIndices:N})});for(let m=1;m<n.length;m++)f.push({source:me("pair",`timeline-order:${n[m-1].memoryId}:${n[m].memoryId}`),sourceIndex:n[m-1].nodeIndex,targetIndex:n[m].nodeIndex,weight:.12,kind:"bitemporal-order"});const G=Object.entries(r).map(([m,R])=>({memoryId:m,events:R})),se=te(i.totalMemories,a.length);return{organ:"timeline",nodes:a,edges:f,events:d,receipts:_,scalars:{totalMemories:se,dayCount:u.length,cellCount:n.length,updatedCount:d.filter(m=>m.type==="MemoryUpdated"||m.type==="Audit:edited"||m.type==="Audit:reconsolidated").length,suppressedCount:d.filter(m=>m.type==="MemorySuppressed"||m.type==="Audit:suppressed").length,maxDayCount:g},alive:n.length>0,rings:p,cells:n,audits:G,raw:{days:e,audits:r}}}var ts=B('<button type="button"> </button>'),ss=B('<p class="state-line svelte-bqsng9">Weaving the live memory history…</p>'),is=B('<p class="state-line error svelte-bqsng9"> </p>'),rs=B('<p class="state-line svelte-bqsng9"> </p>'),ns=B('<button type="button"><span class="svelte-bqsng9"> </span><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>'),as=B('<div class="day-rows svelte-bqsng9"></div>'),ls=B('<p class="state-line svelte-bqsng9">Loading this memory’s audit…</p>'),os=B('<p class="state-line svelte-bqsng9">No audit events returned for this record.</p>'),cs=B('<small class="svelte-bqsng9"> </small>'),us=B('<li class="svelte-bqsng9"><strong class="svelte-bqsng9"> </strong><span class="svelte-bqsng9"> </span><!></li>'),ds=B('<ol class="svelte-bqsng9"></ol>'),ps=B('<p class="eyebrow svelte-bqsng9">TIME-SLICE RECEIPT</p> <h2 class="svelte-bqsng9"> </h2> <dl class="receipt-metrics svelte-bqsng9"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memory ID</dt><dd class="svelte-bqsng9"><code class="svelte-bqsng9"> </code></dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Valid time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Transaction time</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <h3 class="svelte-bqsng9">Audit events</h3> <!>',1),ms=B('<p class="eyebrow svelte-bqsng9">DATE SLICE</p><h2 class="svelte-bqsng9"> </h2><p class="slice-summary svelte-bqsng9"> </p>',1),gs=B('<p class="eyebrow svelte-bqsng9">FIELD IS LIVE</p><h2 class="svelte-bqsng9">Choose a ring, date, or memory.</h2><p class="slice-summary svelte-bqsng9">The field shows growth. This panel makes the evidence legible.</p>',1),vs=B('<button type="button"><strong class="svelte-bqsng9"> </strong><small class="svelte-bqsng9"> </small></button>'),fs=B('<section class="memory-strip glass-panel svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span> </span><strong class="svelte-bqsng9"> </strong></div> <div class="memory-buttons svelte-bqsng9"></div></section>'),hs=B('<!> <main class="timeline-shell svelte-bqsng9"><header class="timeline-head svelte-bqsng9"><div><p class="eyebrow svelte-bqsng9">BITEMPORAL MEMORY HISTORY</p> <h1 class="svelte-bqsng9">Watch memory grow. Inspect every change.</h1> <p class="lede svelte-bqsng9">The rings are real valid-time history. Choose a date or a memory to open its transaction-time receipt.</p></div> <div class="range-control svelte-bqsng9" aria-label="Timeline range"><span class="svelte-bqsng9">TIME WINDOW</span> <!> <button type="button">REWRITTEN</button></div></header> <dl class="vitals svelte-bqsng9" aria-label="Timeline metrics"><div class="svelte-bqsng9"><dt class="svelte-bqsng9">Memories</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Rewritten</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Calendar slices</dt><dd class="svelte-bqsng9"> </dd></div> <div class="svelte-bqsng9"><dt class="svelte-bqsng9">Average retention</dt><dd class="svelte-bqsng9"> </dd></div></dl> <section class="timeline-grid svelte-bqsng9"><div class="glass-panel day-list svelte-bqsng9"><div class="panel-label svelte-bqsng9"><span>VALID-TIME SLICES</span><strong class="svelte-bqsng9"> </strong></div> <!></div> <aside class="glass-panel receipt svelte-bqsng9" aria-live="polite"><!></aside></section> <!></main>',1);function qs(i,e){Ot(e,!0);const r=[7,14,30,90,365];let a=W(Ze([])),n=W(!0),p=W(null),f=W(14),_=W(!1),d=W(null),u=W(null),x=W(!1),g=W(Ze({}));Et(()=>void G());async function G(){w(n,!0),w(p,null);try{const s=await tt.timeline(t(f),500);w(a,s.timeline,!0),t(d)&&!s.timeline.some(o=>o.date===t(d))&&(w(d,null),w(u,null))}catch(s){w(a,[],!0),w(p,s instanceof Error?s.message:"Failed to load timeline",!0)}finally{w(n,!1)}}async function se(s){s!==t(f)&&(w(f,s,!0),w(d,null),w(u,null),await G())}async function ge(s){if(!t(g)[s]){w(x,!0);try{const o=await tt.memoryAudit(s,100);w(g,{...t(g),[s]:o.events},!0)}catch(o){w(p,o instanceof Error?o.message:"Failed to load memory audit",!0)}finally{w(x,!1)}}}function m(s){w(d,s,!0),w(u,null)}function R(s,o){w(d,o,!0),w(u,s.id,!0),ge(s.id)}const F=k(()=>t(_)?t(a).map(s=>({...s,memories:s.memories.filter(it),count:s.memories.filter(it).length})).filter(s=>s.count>0):t(a)),q=k(()=>t(F).flatMap(s=>s.memories)),N=k(()=>t(F).reduce((s,o)=>s+o.count,0)),j=k(()=>t(q).filter(s=>s.updatedAt!==s.createdAt).length),J=k(()=>t(q).length?t(q).reduce((s,o)=>s+(o.retentionStrength??0),0)/t(q).length:0),L=k(()=>t(F).find(s=>s.date===t(d))??null),T=k(()=>t(q).find(s=>s.id===t(u))??null),oe=k(()=>t(u)?t(g)[t(u)]??[]:[]),b=k(()=>es({days:t(a),totalMemories:t(N),audits:t(g)}));function ce(s,o){return Qt(s,o)}function Y(s){if(s.kind==="timeline-cell"){const o=s.payload,h=t(q).find(S=>S.id===o.memoryId);h&&R(h,o.day)}else if(s.kind==="timeline-ring"){const o=s.payload;m(o.date)}}function K(s){return s?new Date(s).toLocaleString():"Not recorded"}var Z=hs();kt("bqsng9",s=>{Ut(()=>{Ft.title="Memory Timeline · Vestige"})});var V=Ne(Z);{let s=k(()=>`timeline-growth-rings:${t(f)}:${t(N)}`);Vt(V,{organ:"timeline",get seed(){return t(s)},get scene(){return t(b)},passes:ce,loading:!1,get error(){return t(p)},emptyLabel:"NO MEMORY GROWTH RINGS IN THIS WINDOW",onpick:Y})}var $=v(V,2),ue=c($),ve=v(c(ue),2),fe=v(c(ve),2);pe(fe,17,()=>r,ke,(s,o)=>{var h=ts();let S;var A=c(h);l(h),U(()=>{et(h,"aria-pressed",t(f)===t(o)),S=we(h,1,"svelte-bqsng9",null,S,{active:t(f)===t(o)}),y(A,`${t(o)??""}D`)}),_e("click",h,()=>se(t(o))),P(s,h)});var ie=v(fe,2);let de;l(ve),l(ue);var re=v(ue,2),ne=c(re),Q=v(c(ne)),lt=c(Q,!0);l(Q),l(ne);var Ge=v(ne,2),Ye=v(c(Ge)),ot=c(Ye,!0);l(Ye),l(Ge);var Ae=v(Ge,2),Qe=v(c(Ae)),ct=c(Qe,!0);l(Qe),l(Ae);var Xe=v(Ae,2),je=v(c(Xe)),ut=c(je);l(je),l(Xe),l(re);var qe=v(re,2),Le=c(qe),Re=c(Le),Je=v(c(Re)),dt=c(Je);l(Je),l(Re);var pt=v(Re,2);{var mt=s=>{var o=ss();P(s,o)},gt=s=>{var o=is(),h=c(o,!0);l(o),U(()=>y(h,t(p))),P(s,o)},vt=s=>{var o=rs(),h=c(o,!0);l(o),U(()=>y(h,t(_)?"No rewritten memories in this window.":"No memory growth in this window.")),P(s,o)},ft=s=>{var o=as();pe(o,21,()=>t(F),h=>h.date,(h,S)=>{var A=ns();let D;var X=c(A),ee=c(X,!0);l(X);var z=v(X),I=c(z,!0);l(z);var E=v(z),ae=c(E);l(E),l(A),U(O=>{D=we(A,1,"svelte-bqsng9",null,D,{active:t(d)===t(S).date}),y(ee,t(S).date),y(I,t(S).count),y(ae,`${O??""}% retained`)},[()=>Math.round(t(S).memories.reduce((O,le)=>O+le.retentionStrength,0)/Math.max(1,t(S).memories.length)*100)]),_e("click",A,()=>m(t(S).date)),P(h,A)}),l(o),P(s,o)};xe(pt,s=>{t(n)?s(mt):t(p)?s(gt,1):t(F).length===0?s(vt,2):s(ft,!1)})}l(Le);var Ke=v(Le,2),ht=c(Ke);{var bt=s=>{var o=ps(),h=v(Ne(o),2),S=c(h,!0);l(h);var A=v(h,2),D=c(A),X=v(c(D)),ee=c(X),z=c(ee,!0);l(ee),l(X),l(D);var I=v(D,2),E=v(c(I)),ae=c(E,!0);l(E),l(I);var O=v(I,2),le=v(c(O)),he=c(le,!0);l(le),l(O);var be=v(O,2),ye=v(c(be)),Te=c(ye);l(ye),l(be),l(A);var St=v(A,4);{var Mt=C=>{var H=ls();P(C,H)},Pt=C=>{var H=os();P(C,H)},Bt=C=>{var H=ds();pe(H,21,()=>t(oe).slice(0,12),ke,(De,Ie)=>{var Ee=us(),Oe=c(Ee),Gt=c(Oe,!0);l(Oe);var Ce=v(Oe),At=c(Ce,!0);l(Ce);var qt=v(Ce);pe(qt,17,()=>jt(t(Ie)),ke,(Ue,Lt)=>{var Fe=cs(),Rt=c(Fe,!0);l(Fe),U(()=>y(Rt,t(Lt))),P(Ue,Fe)}),l(Ee),U(Ue=>{y(Gt,t(Ie).action),y(At,Ue)},[()=>K(t(Ie).timestamp)]),P(De,Ee)}),l(H),P(C,H)};xe(St,C=>{t(x)?C(Mt):t(oe).length===0?C(Pt,1):C(Bt,!1)})}U((C,H,De)=>{y(S,t(T).content),y(z,t(T).id),y(ae,C),y(he,H),y(Te,`${De??""}%`)},[()=>K(t(T).validFrom??t(T).createdAt),()=>K(t(T).updatedAt),()=>Math.round(t(T).retentionStrength*100)]),P(s,o)},yt=s=>{var o=ms(),h=v(Ne(o)),S=c(h,!0);l(h);var A=v(h),D=c(A);l(A),U(()=>{y(S,t(L).date),y(D,`${t(L).count??""} memories entered this valid-time slice. Select one below to inspect its receipt.`)}),P(s,o)},_t=s=>{var o=gs();Nt(2),P(s,o)};xe(ht,s=>{t(T)?s(bt):t(L)?s(yt,1):s(_t,!1)})}l(Ke),l(qe);var xt=v(qe,2);{var wt=s=>{var o=fs(),h=c(o),S=c(h),A=c(S);l(S);var D=v(S),X=c(D);l(D),l(h);var ee=v(h,2);pe(ee,21,()=>t(L).memories.slice(0,20),z=>z.id,(z,I)=>{var E=vs();let ae;var O=c(E),le=c(O,!0);l(O);var he=v(O),be=c(he);l(he),l(E),U((ye,Te)=>{ae=we(E,1,"svelte-bqsng9",null,ae,{active:t(u)===t(I).id}),y(le,t(I).content),y(be,`${ye??""} · ${Te??""}% retention`)},[()=>t(I).id.slice(0,8),()=>Math.round(t(I).retentionStrength*100)]),_e("click",E,()=>R(t(I),t(L).date)),P(z,E)}),l(ee),l(o),U(()=>{y(A,`MEMORIES IN ${t(L).date??""}`),y(X,`${t(L).memories.length??""} RECORDS`)}),P(s,o)};xe(xt,s=>{t(L)&&s(wt)})}l($),U(s=>{et(ie,"aria-pressed",t(_)),de=we(ie,1,"svelte-bqsng9",null,de,{active:t(_)}),y(lt,t(N)),y(ot,t(j)),y(ct,t(a).length),y(ut,`${s??""}%`),y(dt,`${t(f)??""} DAYS`)},[()=>Math.round(t(J)*100)]),_e("click",ie,()=>w(_,!t(_))),P(i,Z),Ct()}It(["click"]);export{qs as component};

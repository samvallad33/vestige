var de=Object.defineProperty;var pe=(s,e,t)=>e in s?de(s,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):s[e]=t;var w=(s,e,t)=>pe(s,typeof e!="symbol"?e+"":e,t);import"../chunks/Bzak7iHL.js";import{o as me}from"../chunks/DAau0uzT.js";import{p as fe,d as N,e as se,j as he,g as n,b as ge,u as D,h as be,s as A,$ as ve}from"../chunks/CGq8RnJq.js";import{h as ye}from"../chunks/De_e6MzK.js";import{a as le}from"../chunks/D35IQVqe.js";import{r as F,M as xe,R as ae}from"../chunks/BMB5u1EX.js";import{T as Se}from"../chunks/D7ozXiSB.js";import{R as we}from"../chunks/BpEKQwpr.js";const j="rgba16float",K=768,J=96,te=16,ie=12,oe=`
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
`,_e=`
${oe}

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
	return vec4f(core * body + vec3f(0.91, 1.0, 0.72) * rim * 1.1 + indigo * seam * 1.3 + scarlet * scar * 1.5, 1.0);
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
	return vec4f(color, 1.0);
}
`,Ge=`
${oe}

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
	return vec4f(color * (0.55 + 0.45 * vignette) * params.brightness, 1.0);
}
`,Ae=`
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
`;class Me{constructor(e,t){w(this,"engine");w(this,"scene",null);w(this,"resources",null);w(this,"sampler",null);w(this,"splatBindLayout",null);w(this,"blurBindLayout",null);w(this,"membraneBindLayout",null);w(this,"splatPipeline",null);w(this,"blurPipeline",null);w(this,"membranePipeline",null);w(this,"cellPipeline",null);w(this,"ringPipeline",null);w(this,"cellCount",0);w(this,"ringCount",0);w(this,"selectedId",null);w(this,"cellGeometry",[]);w(this,"ringGeometry",[]);this.engine=e,this.uploadScene(t)}uploadScene(e){this.scene=e,this.buildGeometry();const t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.uploadBuffers(t))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const t=re(e,"timeline-growth-rings-splat-wgsl",_e),l=re(e,"timeline-growth-rings-blur-wgsl",Ae),r=re(e,"timeline-growth-rings-membrane-wgsl",Ge);this.splatBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-splat-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"timeline-growth-rings-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const d=e.createPipelineLayout({label:"timeline-growth-rings-splat-layout",bindGroupLayouts:[this.splatBindLayout]}),p=e.createPipelineLayout({label:"timeline-growth-rings-blur-layout",bindGroupLayouts:[this.blurBindLayout]}),v=e.createPipelineLayout({label:"timeline-growth-rings-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"}),this.splatPipeline=e.createRenderPipeline({label:"timeline-field-additive-splat",layout:d,vertex:{module:t,entryPoint:"vs_splat"},fragment:{module:t,entryPoint:"fs_splat",targets:[{format:j,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"timeline-field-blur-render-pass",layout:p,vertex:{module:l,entryPoint:"vs_fullscreen"},fragment:{module:l,entryPoint:"fs_blur",targets:[{format:j}]},primitive:{topology:"triangle-list"}});const a={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.membranePipeline=e.createRenderPipeline({label:"timeline-bitemporal-membrane",layout:v,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}}),this.ringPipeline=e.createRenderPipeline({label:"timeline-valid-time-rings",layout:d,vertex:{module:t,entryPoint:"vs_ring"},fragment:{module:t,entryPoint:"fs_ring",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-strip"}}),this.cellPipeline=e.createRenderPipeline({label:"timeline-memory-cells",layout:d,vertex:{module:t,entryPoint:"vs_cell"},fragment:{module:t,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var R,C,I,V,k,H;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),l=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==l;let d=(R=this.resources)==null?void 0:R.cellBuffer,p=(C=this.resources)==null?void 0:C.ringBuffer,v=(I=this.resources)==null?void 0:I.blurHBuffer,a=(V=this.resources)==null?void 0:V.blurVBuffer;if(d||(d=e.createBuffer({label:"timeline-cells",size:K*te*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),p||(p=e.createBuffer({label:"timeline-rings",size:J*ie*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),v||(v=e.createBuffer({label:"timeline-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(v,0,new Float32Array([1,0,0,0]))),a||(a=e.createBuffer({label:"timeline-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(a,0,new Float32Array([0,1,0,0]))),!r&&this.resources)return;(k=this.resources)==null||k.fieldA.destroy(),(H=this.resources)==null||H.fieldB.destroy();const u=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,g=e.createTexture({label:"timeline-field-a-rgba16float",size:[t,l],format:j,usage:u}),c=e.createTexture({label:"timeline-field-b-rgba16float",size:[t,l],format:j,usage:u}),b=g.createView(),B=c.createView(),T=e.createBindGroup({label:"timeline-growth-rings-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:d}},{binding:2,resource:{buffer:p}}]}),o=e.createBindGroup({label:"timeline-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:b},{binding:2,resource:{buffer:v}}]}),_=e.createBindGroup({label:"timeline-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:B},{binding:2,resource:{buffer:a}}]}),U=e.createBindGroup({label:"timeline-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:b}]});this.resources={cellBuffer:d,ringBuffer:p,blurHBuffer:v,blurVBuffer:a,splatBindGroup:T,blurHBindGroup:o,blurVBindGroup:_,membraneBindGroup:U,fieldA:g,fieldB:c,fieldAView:b,fieldBView:B,fieldSize:[t,l]}}buildGeometry(){var t,l;const e=((t=this.scene)==null?void 0:t.cells)??[];this.cellGeometry=e.slice(0,K).map(r=>({cell:r,x:Math.cos(r.angle)*r.radius,y:Math.sin(r.angle)*r.radius,r:.018+r.retention*.016})),this.ringGeometry=(((l=this.scene)==null?void 0:l.rings)??[]).slice(0,J).map(r=>({ring:r,r:r.radius}))}uploadBuffers(e){var p,v,a;if(!this.resources)return;const t=new Float32Array(K*te);this.cellCount=Math.min(K,this.cellGeometry.length);const l=Math.max(1,this.ringGeometry.length-1);for(let u=0;u<this.cellCount;u++){const g=this.cellGeometry[u],c=g.cell,b=this.selectedId===c.id||this.selectedId===c.memoryId?1:0;t.set([g.x,g.y,g.r,c.radius,c.retention,c.rewritten?1:0,c.suppressed?1:0,((v=(p=this.scene)==null?void 0:p.raw.audits[c.memoryId])==null?void 0:v.length)??0,c.dayIndex/l,Date.parse(c.transactionAt||c.validFrom||"")/864e11||0,c.dayIndex,u,b,0,0,0],u*te)}this.ringCount=Math.min(J,this.ringGeometry.length);const r=new Float32Array(J*ie),d=Math.max(1,((a=this.scene)==null?void 0:a.scalars.maxDayCount)??1);for(let u=0;u<this.ringCount;u++){const g=this.ringGeometry[u],c=g.ring,b=this.selectedId===c.id||this.selectedId===c.date?1:0;r.set([g.r,c.count/d,c.retention,c.index,c.updatedCount,c.suppressedCount,u/Math.max(1,this.ringCount),b,c.memoryIndices.length,u,0,0],u*ie)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.ringCount,e.queue.writeBuffer(this.resources.cellBuffer,0,t),e.queue.writeBuffer(this.resources.ringBuffer,0,r)}compute(e){const t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);const l=this.resources,r=e.beginRenderPass({label:"timeline-field-splat-pass",colorAttachments:[{view:l.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});r.setPipeline(this.splatPipeline),r.setBindGroup(0,l.splatBindGroup),r.draw(6,this.cellCount),r.end();const d=e.beginRenderPass({label:"timeline-field-blur-h-pass",colorAttachments:[{view:l.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});d.setPipeline(this.blurPipeline),d.setBindGroup(0,l.blurHBindGroup),d.draw(6,1),d.end();const p=e.beginRenderPass({label:"timeline-field-blur-v-pass",colorAttachments:[{view:l.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});p.setPipeline(this.blurPipeline),p.setBindGroup(0,l.blurVBindGroup),p.draw(6,1),p.end()}render(e){!this.resources||!this.membranePipeline||!this.ringPipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.ringCount>0&&(e.setPipeline(this.ringPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(192,this.ringCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.draw(6,this.cellCount)))}ringSpin(e){const t=this.engine.params[10]||0,l=.045+e*.1;return t*l}orbitCpu(e,t,l,r){const d=Math.hypot(e,t);if(d<1e-4)return{x:e,y:t};const p=this.engine.params[10]||0,a=Math.atan2(t,e)+this.ringSpin(r)+Math.sin(p*.6+l*6.283)*.02,u=d*(1+.016*Math.sin(p*1.1+l*6.283));return{x:Math.cos(a)*u,y:Math.sin(a)*u}}pickAt(e,t){const l=Math.max(1,this.ringGeometry.length-1);let r=null,d=1/0;for(let a=0;a<this.cellGeometry.length;a++){const u=this.cellGeometry[a],g=u.cell.dayIndex/l,c=this.orbitCpu(u.x,u.y,a,g),b=Math.hypot(e-c.x,t-c.y);b<=Math.max(.045,u.r*1.8)&&b<d&&(r={id:u.cell.id,kind:"timeline-cell",index:a,payload:u.cell},d=b)}if(r)return this.selectedId=r.id,r;const p=Math.hypot(e,t),v=this.engine.params[10]||0;for(let a=0;a<this.ringGeometry.length;a++){const u=this.ringGeometry[a],g=l>0?a/l:0,c=u.r*(1+.016*Math.sin(v*1.1+g*6.283));if(Math.abs(p-c)<=.03)return this.selectedId=u.ring.id,{id:u.ring.id,kind:"timeline-ring",index:a,payload:u.ring}}return null}dispose(){var e,t,l,r,d,p;(e=this.resources)==null||e.cellBuffer.destroy(),(t=this.resources)==null||t.ringBuffer.destroy(),(l=this.resources)==null||l.blurHBuffer.destroy(),(r=this.resources)==null||r.blurVBuffer.destroy(),(d=this.resources)==null||d.fieldA.destroy(),(p=this.resources)==null||p.fieldB.destroy(),this.resources=null}}function re(s,e,t){s.pushErrorScope("validation");const l=s.createShaderModule({label:e,code:t});return l.getCompilationInfo().then(r=>{for(const d of r.messages)console.error(`[observatory] ${e} WGSL ${d.type} ${d.lineNum}:${d.linePos} ${d.message}`)}),s.popErrorScope().then(r=>{r&&console.error(`[observatory] ${e} shader module validation: ${r.message}`)}),l}function Pe(s,e){return F(xe.blackwater),F(ae.healthy),F(ae.luciferin),[new Me(s,e)]}function Z(s,e=""){return typeof s=="string"?s:s==null?e:String(s)}function W(s,e=0){return typeof s=="number"&&Number.isFinite(s)?s:e}function ue(s){return Math.max(0,Math.min(1,s))}function Q(s,e,t){return{kind:s,id:e||`${s}:unknown`}}function Be(s,e){return{kind:"scalar",id:`timeline.${s}`,scalar:{name:s,value:e}}}function ce(s){return ue(W(s.retentionStrength,0))}function Re(s){return ue(W(s.combinedScore??s.retentionStrength,ce(s)))}function Ee(s,e){return e[s.id]??[]}function ne(s,e){return s.some(t=>t.action===e)}function Te(s){const e=s.days??[],t=s.audits??{},l=[],r=[],d=[],p=[],v=[],a=[],u=e.filter(o=>o.count>0||o.memories.length>0),g=Math.max(1,u.length),c=Math.max(1,...u.map(o=>o.count||o.memories.length));u.forEach((o,_)=>{const U=.16+_/Math.max(1,g-1)*.7,R=o.memories??[],C=[];let I=0,V=0,k=0;R.forEach((h,z)=>{const E=Ee(h,t),Y=ce(h),x=Z(h.updatedAt)!==Z(h.createdAt)||ne(E,"edited")||ne(E,"reconsolidated"),O=ne(E,"suppressed")||W(h.suppression_count,0)>0;x&&(V+=1),O&&(k+=1),I+=Y;const $=l.length;C.push($);const ee=(z+.5)/Math.max(1,R.length)*Math.PI*2+_*.37,i=(z%5-2)*.008,m=U+i,f=Z(h.validFrom??h.createdAt,o.date),S=Z(h.updatedAt??h.createdAt,f),P=h.content||h.id.slice(0,8),G=Q("memory",h.id);if(l.push({source:G,index:$,label:P,retention:Y,trust:Re(h),stability:W(h.storageStrength,void 0),lastAccessed:h.lastAccessedAt??h.updatedAt??h.createdAt,suppression:O?1:0,tags:[o.date,...h.tags??[]],type:h.nodeType??"memory"}),r.push({id:`timeline:${o.date}:${h.id}`,memoryId:h.id,day:o.date,dayIndex:_,nodeIndex:$,angle:ee,radius:m,retention:Y,validFrom:f,transactionAt:S,suppressed:O,rewritten:x,label:P,provenance:G}),(x||O)&&a.push({source:Q("event",`${h.id}:${x?"updated":"suppressed"}:${S}`),type:O?"MemorySuppressed":"MemoryUpdated",targetIndex:$,frame:45+_*10+z,energy:O?1:.65}),E.length>0){v.push({source:Q("receipt",`memory-audit:${h.id}`),label:`audit ${h.id.slice(0,8)} · ${E.length} events`,nodeIndices:[$]});for(const y of E.slice(0,8))a.push({source:Q("event",`${h.id}:${y.action}:${y.timestamp}`),type:`Audit:${y.action}`,targetIndex:$,frame:70+_*12,energy:.4+Math.abs(W(y.new_value,0)-W(y.old_value,0))})}});const H=R.length?I/R.length:0,q=Be(`day.${o.date}.count`,o.count);d.push({id:`timeline-day:${o.date}`,date:o.date,index:_,count:o.count,radius:U,retention:H,updatedCount:V,suppressedCount:k,memoryIndices:C,provenance:q}),v.push({source:q,label:`${o.date} · ${o.count} memories`,nodeIndices:C})});for(let o=1;o<r.length;o++)p.push({source:Q("pair",`timeline-order:${r[o-1].memoryId}:${r[o].memoryId}`),sourceIndex:r[o-1].nodeIndex,targetIndex:r[o].nodeIndex,weight:.12,kind:"bitemporal-order"});const b=Object.entries(t).map(([o,_])=>({memoryId:o,events:_})),B=W(s.totalMemories,l.length);return{organ:"timeline",nodes:l,edges:p,events:a,receipts:v,scalars:{totalMemories:B,dayCount:u.length,cellCount:r.length,updatedCount:a.filter(o=>o.type==="MemoryUpdated"||o.type==="Audit:edited"||o.type==="Audit:reconsolidated").length,suppressedCount:a.filter(o=>o.type==="MemorySuppressed"||o.type==="Audit:suppressed").length,maxDayCount:c},alive:r.length>0,rings:d,cells:r,audits:b,raw:{days:e,audits:t}}}function Ve(s,e){fe(e,!0);const t=[...F("#22C7DE"),1],l=[...F("#7C6CFF"),.95],r=[...F("#A8FF5E"),.92],d=[...F("#FF3B30"),.92],p=[...F("#29F2A9"),.6],v=[7,14,30,90,365];let a=N(se([])),u=N(!0),g=N(null),c=N(14),b=N(null),B=N(null),T=N(!1),o=N(se({})),_=null,U=null;me(()=>R());async function R(){A(u,!0),A(g,null);try{const i=await le.timeline(n(c),500);A(a,i.timeline,!0)}catch(i){A(a,[],!0),A(g,i instanceof Error?i.message:"Failed to load timeline",!0)}finally{A(u,!1)}}function C(){const i=v.indexOf(n(c));A(c,v[(i+1)%v.length],!0),A(b,null),A(B,null),R()}let I=D(()=>n(a).reduce((i,m)=>i+m.count,0)),V=D(()=>n(a).reduce((i,m)=>i+m.memories.filter(f=>f.updatedAt&&f.createdAt&&f.updatedAt!==f.createdAt).length,0)),k=D(()=>{const i=n(a).flatMap(m=>m.memories);return i.length?i.reduce((m,f)=>m+(f.retentionStrength??0),0)/i.length:0}),H=D(()=>Te({days:n(a),totalMemories:n(I),audits:n(o)}));async function q(i){if(!n(o)[i]){A(T,!0);try{const m=await le.memoryAudit(i,100);A(o,{...n(o),[i]:m.events},!0)}catch(m){A(g,m instanceof Error?m.message:"Failed to load memory audit",!0)}finally{A(T,!1)}}}function h(i){return i?n(a).flatMap(m=>m.memories).find(m=>m.id===i.memoryId)??null:null}let z=D(()=>h(n(b))),E=D(()=>{const i=n(b);return i?n(o)[i.memoryId]??[]:[]});he(()=>{n(a),n(u),n(g),n(c),n(b),n(B),n(z),n(E),n(T),_==null||_.setText(O())});function Y(i){return i.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}function x(i,m,f,S,P,G,y,M={}){return{id:i,kind:m,text:Y(f),x:S,y:P,size:G,color:y,depth:.7,weight:.6,revealSpan:18,maxWidthEm:60,...M}}function O(){const i=[];if(i.push(x("tl:title","tl-title","BITEMPORAL GROWTH RINGS",-.94,.9,.04,t,{depth:1,weight:.9})),i.push(x("tl:range","tl-range",`RANGE ${n(c)}D  [click to cycle]`,-.94,.82,.024,t,{depth:.85,hitPadX:.03,hitPadY:.05})),n(u))return i.push(x("tl:status","tl-status","WEAVING VALID-TIME RINGS...",-.3,.02,.04,t,{revealSpan:40})),i;if(n(g))return i.push(x("tl:status","tl-status",`ERROR - ${n(g)}`.slice(0,70),-.5,.02,.032,d,{revealSpan:12})),i;if(n(a).length===0)return i.push(x("tl:status","tl-status","NO MEMORY GROWTH RINGS IN THIS WINDOW",-.5,.02,.03,p,{revealSpan:24})),i;if([[`${n(I)}`,"MEMORIES IN VALID-TIME RINGS",t],[`${n(V)}`,"TRANSACTION-TIME SHADOWS",l],[`${n(a).length}`,"CALENDAR SLICES",t],[`${Math.round(n(k)*100)}%`,"AVERAGE RETENTION OXYGEN",r]].forEach(([f,S,P],G)=>{const y=.72-G*.11;i.push(x(`tl:vital-num:${G}`,"tl-vital",f,-.92,y,.05,P,{depth:.9,weight:.85})),i.push(x(`tl:vital-lbl:${G}`,"tl-vital-lbl",S,-.92,y-.05,.017,p,{depth:.5}))}),i.push(x("tl:receipt-hdr","tl-receipt-hdr","TIME-SLICE RECEIPT",.3,.9,.022,l,{depth:.85})),n(b)&&n(z)){const f=n(b),S=n(z).content.replace(/\s+/g," ").trim().slice(0,46),P=f.suppressed?"suppressed":f.rewritten?"rewritten":"created",G=[[`#${f.memoryId.slice(0,12)}`,t],[S,p],[`valid  ${String(f.validFrom).slice(0,19)}`,r],[`tx     ${String(f.transactionAt).slice(0,19)}`,l],[`retain ${Math.round(f.retention*100)}%   state ${P}`,P==="suppressed"?d:r]];G.forEach(([M,L],X)=>{i.push(x(`tl:receipt:${X}`,"tl-receipt",M,.3,.82-X*.06,.018,L,{revealSpan:10,startFrame:X*2}))});const y=.82-G.length*.06-.04;i.push(x("tl:audit-hdr","tl-audit-hdr",n(T)?"MEMORY-AUDIT (loading...)":"MEMORY-AUDIT",.3,y,.016,p,{depth:.5})),n(E).slice(0,10).forEach((M,L)=>{const X=`${M.action}  ${String(M.timestamp).slice(0,19)}`;i.push(x(`tl:audit:${L}`,"tl-audit",X,.3,y-.03-L*.045,.015,t,{revealSpan:8,startFrame:L*2,depth:.6}))}),!n(T)&&n(E).length===0&&i.push(x("tl:audit-empty","tl-audit","no audit events returned",.3,y-.03,.015,p))}else if(n(B)){const f=n(B);[[f.date,t],[`memories     ${f.count}`,r],[`avg retain   ${Math.round(f.retention*100)}%`,r],[`rewrites     ${f.updatedCount}`,l],[`suppressed   ${f.suppressedCount}`,f.suppressedCount>0?d:p],["click a cell for its audit receipt",p]].forEach(([P,G],y)=>{i.push(x(`tl:ring:${y}`,"tl-receipt",P,.3,.82-y*.06,.017,G,{revealSpan:10,startFrame:y*2}))})}else i.push(x("tl:receipt-hint","tl-receipt","CLICK A GROWTH RING FOR ITS DATE SLICE, OR A CELL FOR ITS VALID-TIME VS TRANSACTION-TIME RECEIPT",.3,.8,.017,p,{maxWidthEm:34,revealSpan:20}));return i}function $(i,m){const f=Pe(i,m),S=new Se(i);_=S,S.init().then(()=>S.setText(O()));const P={render(G){S.render(G)},pickAt(G,y){const M=S.pickAt(G,y),L=M&&(M.kind==="tl-vital"||M.kind==="tl-range")?M.id:null;return L!==U&&(U=L,S.setRunDepth(L,1)),(M==null?void 0:M.kind)==="tl-range"&&C(),null},dispose(){S.dispose(),_===S&&(_=null)}};return[...f,P]}function ee(i){if(i.kind==="timeline-cell"){const m=i.payload;A(b,m,!0),A(B,null),q(m.memoryId)}else if(i.kind==="timeline-ring"){const m=i.payload;A(B,m,!0),A(b,null)}}ye("bqsng9",i=>{be(()=>{ve.title="Bitemporal Growth Rings · Vestige"})});{let i=D(()=>`timeline-growth-rings:${n(c)}:${n(I)}`),m=D(()=>n(u)||n(T));we(s,{organ:"timeline",get seed(){return n(i)},get scene(){return n(H)},passes:$,get loading(){return n(m)},get error(){return n(g)},emptyLabel:"NO MEMORY GROWTH RINGS IN THIS WINDOW",onpick:ee})}ge()}export{Ve as component};

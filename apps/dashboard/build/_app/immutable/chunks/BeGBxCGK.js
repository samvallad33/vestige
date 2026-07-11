var F=Object.defineProperty;var D=(o,e,t)=>e in o?F(o,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):o[e]=t;var c=(o,e,t)=>D(o,typeof e!="symbol"?e+"":e,t);import{r as m,C as A,I as E,R as _,a as O}from"./BMB5u1EX.js";const S=`
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

// One living cell. 16 floats = 4 vec4, 16-byte aligned lanes.
struct LivingCellGpu {
	// x,y NDC base position; z billboard radius; w orbit ring radius (== length(xy))
	pos_radius: vec4f,
	// rgb hue; w energy 0..1 (brightness / activation)
	hue_energy: vec4f,
	// x orbit phase (0..1); y flags (bit0 selected, bit1 endangered/scar, bit2 pulse-strong);
	// z secondary metric (retention-ish 0..1); w spin scale
	phase_flags: vec4f,
	// x ring index / group; y satellite twinkle seed; z,w reserved
	extra: vec4f,
};

const TAU: f32 = 6.28318530718;

// Living orbital drift — the thing that makes the field MOVE. Deterministic:
// a pure function of params.time + the cell's own phase. ring_spin gives inner
// (high-phase) cells a faster turn, like a spinning galaxy core.
fn ring_spin(phase01: f32) -> f32 {
	let speed = 0.045 + phase01 * 0.10;
	return params.time * speed;
}

fn orbit(base: vec2f, phase01: f32, spin_scale: f32) -> vec2f {
	let radius = length(base);
	if (radius < 0.0001) { return base; }
	let ang0 = atan2(base.y, base.x);
	let ang = ang0 + ring_spin(phase01) * spin_scale + sin(params.time * 0.6 + phase01 * TAU) * 0.02;
	let rr = radius * (1.0 + 0.016 * sin(params.time * 1.1 + phase01 * TAU));
	return vec2f(cos(ang), sin(ang)) * rr;
}
`,N=`
${S}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f, // x phase, y flags, z metric2, w spin
};

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	// breathe the splat radius so density is never a flat print
	let breathe = 1.0 + 0.14 * sin(params.time * 1.5 + phase * TAU);
	let r = c.pos_radius.z * 2.4 * breathe * (1.0 + c.hue_energy.w * 0.9);
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = c.phase_flags;
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let metric2 = clamp(in.info.z, 0.0, 1.0);
	let flags = in.info.y;
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let body = exp(-d * d * 2.7) * (0.32 + energy * 0.95);
	// .r = raw density (fills the void), .g = oxygen (energy-weighted),
	// .b = scar/seam accent for endangered cells
	return vec4f(body, body * (0.4 + metric2 * 0.9), body * scar * 0.7, 1.0);
}
`,I=`
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
`,z=`
${S}

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
	let scar = clamp(f.b, 0.0, 3.0);
	let grad = length(vec2f((right.r + right.g) - (left.r + left.g), (up.r + up.g) - (down.r + down.g)));
	let membrane = smoothstep(0.05, 0.62, density) * (1.0 - smoothstep(2.0, 4.0, density));
	let edge = smoothstep(0.008, 0.11, grad) * membrane;
	let breath = 0.72 + 0.55 * params.pulse;
	let blackwater = vec3f(0.006, 0.012, 0.015);
	let amber = vec3f(0.86, 0.42, 0.12);
	let oxygen_col = vec3f(0.66, 1.0, 0.37);
	let scarlet = vec3f(1.0, 0.23, 0.18);
	var color = blackwater * (0.35 + density * 0.14);
	color = color + mix(amber, oxygen_col, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.34 * breath;
	color = color + vec3f(0.91, 1.0, 0.72) * edge * (0.85 + 0.5 * params.pulse);
	color = color + scarlet * scar * (0.6 + 0.4 * params.pulse);
	let vignette = smoothstep(1.02, 0.10, distance(in.uv, vec2f(0.5)));
	return vec4f(color * (0.5 + 0.5 * vignette) * params.brightness, 1.0);
}
`,k=`
${S}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f,
};

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	let flags = c.phase_flags.y;
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let beat = 1.0 + 0.22 * sin(params.time * 2.3 + c.extra.y * 1.7);
	let r = c.pos_radius.z * (0.85 + c.hue_energy.w * 0.9 + selected * 1.3) * beat;
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = c.phase_flags;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let hue = in.hue_energy.rgb;
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let flags = in.info.y;
	let metric2 = clamp(in.info.z, 0.0, 1.0);
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let phase = in.info.x;
	let twinkle = 0.6 + 0.8 * (0.5 + 0.5 * sin(params.time * 2.1 + phase * 26.0));
	let body = exp(-d * d * 2.7) * (0.55 + energy * 1.7) * twinkle;
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.40, d));
	let scarlet = vec3f(1.0, 0.23, 0.18);
	let ivory = vec3f(0.95, 1.0, 0.86);
	var color = hue * body;
	color = color + ivory * rim * (0.9 + selected * 1.6);
	color = color + scarlet * scar * smoothstep(0.16, 0.0, abs(d - 0.74)) * 1.5;
	return vec4f(color, 1.0);
}
`,x="rgba16float",B=2048,L=16;class Q{constructor(e){c(this,"engine");c(this,"cells",[]);c(this,"scalars",{});c(this,"resources",null);c(this,"sampler",null);c(this,"splatBindLayout",null);c(this,"blurBindLayout",null);c(this,"membraneBindLayout",null);c(this,"splatPipeline",null);c(this,"blurPipeline",null);c(this,"membranePipeline",null);c(this,"cellPipeline",null);c(this,"cellCount",0);this.engine=e}setCells(e,t={}){this.cells=e.slice(0,B),this.scalars=t;const l=this.engine.gpuDevice;l&&(this.ensurePipelines(l),this.ensureResources(l),this.uploadBuffers(l))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const t=P(e,"living-field-splat",N),l=P(e,"living-field-blur",I),r=P(e,"living-field-membrane",z),n=P(e,"living-field-cell",k);this.splatBindLayout=e.createBindGroupLayout({label:"living-field-splat-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"living-field-blur-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"living-field-membrane-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const s=e.createPipelineLayout({label:"living-field-splat-pl",bindGroupLayouts:[this.splatBindLayout]}),i=e.createPipelineLayout({label:"living-field-blur-pl",bindGroupLayouts:[this.blurBindLayout]}),u=e.createPipelineLayout({label:"living-field-membrane-pl",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"});const a={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.splatPipeline=e.createRenderPipeline({label:"living-field-splat",layout:s,vertex:{module:t,entryPoint:"vs_splat"},fragment:{module:t,entryPoint:"fs_splat",targets:[{format:x,blend:a}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"living-field-blur",layout:i,vertex:{module:l,entryPoint:"vs_fullscreen"},fragment:{module:l,entryPoint:"fs_blur",targets:[{format:x}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=e.createRenderPipeline({label:"living-field-membrane",layout:u,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}}),this.cellPipeline=e.createRenderPipeline({label:"living-field-cells",layout:s,vertex:{module:n,entryPoint:"vs_cell"},fragment:{module:n,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var G,M,w,R,U;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),l=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==l;let n=(G=this.resources)==null?void 0:G.cellBuffer,s=(M=this.resources)==null?void 0:M.blurHBuffer,i=(w=this.resources)==null?void 0:w.blurVBuffer;if(n||(n=e.createBuffer({label:"living-field-cells",size:B*L*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),s||(s=e.createBuffer({label:"living-field-blur-h",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([1,0,0,0]))),i||(i=e.createBuffer({label:"living-field-blur-v",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(i,0,new Float32Array([0,1,0,0]))),!r&&this.resources)return;(R=this.resources)==null||R.fieldA.destroy(),(U=this.resources)==null||U.fieldB.destroy();const u=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,a=e.createTexture({label:"living-field-a",size:[t,l],format:x,usage:u}),p=e.createTexture({label:"living-field-b",size:[t,l],format:x,usage:u}),f=a.createView(),b=p.createView(),v=e.createBindGroup({label:"living-field-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}}]}),d=e.createBindGroup({label:"living-field-cell-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}}]}),h=e.createBindGroup({label:"living-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:f},{binding:2,resource:{buffer:s}}]}),g=e.createBindGroup({label:"living-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:b},{binding:2,resource:{buffer:i}}]}),T=e.createBindGroup({label:"living-field-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:f}]});this.resources={cellBuffer:n,blurHBuffer:s,blurVBuffer:i,splatBindGroup:v,cellBindGroup:d,blurHBindGroup:h,blurVBindGroup:g,membraneBindGroup:T,fieldA:a,fieldB:p,fieldAView:f,fieldBView:b,fieldSize:[t,l]}}uploadBuffers(e){if(!this.resources)return;const t=new Float32Array(B*L);this.cellCount=Math.min(B,this.cells.length);for(let l=0;l<this.cellCount;l++){const r=this.cells[l],n=Math.hypot(r.x,r.y);let s=0;r.selected&&(s|=1),r.scar&&(s|=2),r.energy>.8&&(s|=4);const i=l*L;t[i+0]=r.x,t[i+1]=r.y,t[i+2]=Math.max(.006,r.radius),t[i+3]=n,t[i+4]=r.hue[0],t[i+5]=r.hue[1],t[i+6]=r.hue[2],t[i+7]=C(r.energy),t[i+8]=r.phase,t[i+9]=s,t[i+10]=C(r.metric2??r.energy),t[i+11]=r.spin??1,t[i+12]=l,t[i+13]=r.seed??r.phase*97.13,t[i+14]=0,t[i+15]=0}this.engine.params[2]=this.cellCount,e.queue.writeBuffer(this.resources.cellBuffer,0,t)}compute(e){const t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);const l=this.resources,r=e.beginRenderPass({label:"living-field-splat-pass",colorAttachments:[{view:l.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});r.setPipeline(this.splatPipeline),r.setBindGroup(0,l.splatBindGroup),this.cellCount>0&&r.draw(6,this.cellCount),r.end();const n=e.beginRenderPass({label:"living-field-blur-h",colorAttachments:[{view:l.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(this.blurPipeline),n.setBindGroup(0,l.blurHBindGroup),n.draw(6,1),n.end();const s=e.beginRenderPass({label:"living-field-blur-v",colorAttachments:[{view:l.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});s.setPipeline(this.blurPipeline),s.setBindGroup(0,l.blurVBindGroup),s.draw(6,1),s.end()}render(e){!this.resources||!this.membranePipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.cellBindGroup),e.draw(6,this.cellCount)))}orbitCpu(e,t,l,r){const n=Math.hypot(e,t);if(n<1e-4)return{x:e,y:t};const s=this.engine.params[10]||0,i=Math.atan2(t,e),u=(.045+l*.1)*s*r,a=i+u+Math.sin(s*.6+l*Math.PI*2)*.02,p=n*(1+.016*Math.sin(s*1.1+l*Math.PI*2));return{x:Math.cos(a)*p,y:Math.sin(a)*p}}pickAt(e,t){let l=null,r=1/0;for(let n=0;n<this.cellCount;n++){const s=this.cells[n],i=this.orbitCpu(s.x,s.y,s.phase,s.spin??1),u=Math.hypot(e-i.x,t-i.y),a=Math.max(.04,s.radius*2.6);u<=a&&u<r&&(l={id:s.pickId,kind:s.kind??"living-cell",index:n,payload:s.payload??s},r=u)}return l}dispose(){var e,t,l,r,n;(e=this.resources)==null||e.cellBuffer.destroy(),(t=this.resources)==null||t.blurHBuffer.destroy(),(l=this.resources)==null||l.blurVBuffer.destroy(),(r=this.resources)==null||r.fieldA.destroy(),(n=this.resources)==null||n.fieldB.destroy(),this.resources=null}}function C(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:0))}function P(o,e,t){o.pushErrorScope("validation");const l=o.createShaderModule({label:e,code:t});return l.getCompilationInfo().then(r=>{for(const n of r.messages)n.type==="error"&&console.error(`[living-field] ${e} WGSL ${n.type} ${n.lineNum}:${n.linePos} ${n.message}`)}),o.popErrorScope().then(r=>{r&&console.error(`[living-field] ${e} shader module validation: ${r.message}`)}),l}const V=2.399963229728653;function q(o,e={}){const t=o.length;if(t===0)return[];const l=e.maxRadius??.92,r=e.minCellR??.012,n=e.maxCellR??.05;return o.map((i,u)=>({d:i,i:u})).sort((i,u)=>(u.d.score||0)-(i.d.score||0)).map(({d:i},u)=>{const a=t>1?u/(t-1):0,p=l*Math.sqrt(.06+.94*a),f=u*V,b=Math.cos(f)*p,v=Math.sin(f)*p,d=y(i.score),h=r+(n-r)*Math.sqrt(d),g=i.hue??O(d);return{x:b,y:v,radius:h,hue:[g[0],g[1],g[2]],energy:y(i.energy??.35+.65*d),phase:u/t,pickId:i.id,kind:i.kind,payload:i.payload??i,selected:i.selected,scar:i.scar,metric2:y(i.metric2??d),spin:1}})}function W(o,e,t={}){if(o.length===0)return[];const r=t.maxRadius??.9,n=t.minCellR??.014,s=t.maxCellR??.05,i=Math.max(1,t.ringCount??new Set(o.map(e)).size),u=new Map;return o.map((a,p)=>{const f=(e(a,p)%i+i)%i,b=u.get(f)??0;u.set(f,b+1);const v=r*(.18+.82*(f/Math.max(1,i-1))),d=b*V+f*.7,h=y(a.score),g=a.hue??O(h);return{x:Math.cos(d)*v,y:Math.sin(d)*v,radius:n+(s-n)*Math.sqrt(h),hue:[g[0],g[1],g[2]],energy:y(a.energy??.35+.65*h),phase:f/i,pickId:a.id,kind:a.kind,payload:a.payload??a,selected:a.selected,scar:a.scar,metric2:y(a.metric2??h),spin:1}})}const X={oxygen:m(_.luciferin),healthy:m(_.healthy),recall:m(_.recall),bridge:m(_.bridge),debt:m(_.debt),scarlet:m(E.veto),caution:m(E.caution),forward:m(A.forward),retrograde:m(A.retrograde)};function y(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:0))}export{X as F,Q as L,W as a,q as l};

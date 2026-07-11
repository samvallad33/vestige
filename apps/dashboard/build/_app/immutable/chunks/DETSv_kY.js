var D=Object.defineProperty;var N=(o,e,t)=>e in o?D(o,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):o[e]=t;var f=(o,e,t)=>N(o,typeof e!="symbol"?e+"":e,t);import{r as h,C as F,I as C,R as x,a as I}from"./BMB5u1EX.js";const M=`
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
`,k=`
${M}

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
`,z=`
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
`,$=`
${M}

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
`,H=`
${M}

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
`,P="rgba16float",L=2048,G=16;class W{constructor(e){f(this,"engine");f(this,"cells",[]);f(this,"scalars",{});f(this,"resources",null);f(this,"sampler",null);f(this,"splatBindLayout",null);f(this,"blurBindLayout",null);f(this,"membraneBindLayout",null);f(this,"splatPipeline",null);f(this,"blurPipeline",null);f(this,"membranePipeline",null);f(this,"cellPipeline",null);f(this,"cellCount",0);this.engine=e}setCells(e,t={}){this.cells=e.slice(0,L),this.scalars=t;const r=this.engine.gpuDevice;r&&(this.ensurePipelines(r),this.ensureResources(r),this.uploadBuffers(r))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const t=S(e,"living-field-splat",k),r=S(e,"living-field-blur",z),i=S(e,"living-field-membrane",$),l=S(e,"living-field-cell",H);this.splatBindLayout=e.createBindGroupLayout({label:"living-field-splat-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"living-field-blur-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"living-field-membrane-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const s=e.createPipelineLayout({label:"living-field-splat-pl",bindGroupLayouts:[this.splatBindLayout]}),n=e.createPipelineLayout({label:"living-field-blur-pl",bindGroupLayouts:[this.blurBindLayout]}),c=e.createPipelineLayout({label:"living-field-membrane-pl",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"});const a={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.splatPipeline=e.createRenderPipeline({label:"living-field-splat",layout:s,vertex:{module:t,entryPoint:"vs_splat"},fragment:{module:t,entryPoint:"fs_splat",targets:[{format:P,blend:a}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"living-field-blur",layout:n,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:"fs_blur",targets:[{format:P}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=e.createRenderPipeline({label:"living-field-membrane",layout:c,vertex:{module:i,entryPoint:"vs_fullscreen"},fragment:{module:i,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}}),this.cellPipeline=e.createRenderPipeline({label:"living-field-cells",layout:s,vertex:{module:l,entryPoint:"vs_cell"},fragment:{module:l,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:a}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var w,R,U,A,E;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),r=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),i=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==r;let l=(w=this.resources)==null?void 0:w.cellBuffer,s=(R=this.resources)==null?void 0:R.blurHBuffer,n=(U=this.resources)==null?void 0:U.blurVBuffer;if(l||(l=e.createBuffer({label:"living-field-cells",size:L*G*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),s||(s=e.createBuffer({label:"living-field-blur-h",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([1,0,0,0]))),n||(n=e.createBuffer({label:"living-field-blur-v",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(n,0,new Float32Array([0,1,0,0]))),!i&&this.resources)return;(A=this.resources)==null||A.fieldA.destroy(),(E=this.resources)==null||E.fieldB.destroy();const c=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,a=e.createTexture({label:"living-field-a",size:[t,r],format:P,usage:c}),u=e.createTexture({label:"living-field-b",size:[t,r],format:P,usage:c}),d=a.createView(),_=u.createView(),g=e.createBindGroup({label:"living-field-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:l}}]}),m=e.createBindGroup({label:"living-field-cell-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:l}}]}),b=e.createBindGroup({label:"living-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:d},{binding:2,resource:{buffer:s}}]}),B=e.createBindGroup({label:"living-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:_},{binding:2,resource:{buffer:n}}]}),v=e.createBindGroup({label:"living-field-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:d}]});this.resources={cellBuffer:l,blurHBuffer:s,blurVBuffer:n,splatBindGroup:g,cellBindGroup:m,blurHBindGroup:b,blurVBindGroup:B,membraneBindGroup:v,fieldA:a,fieldB:u,fieldAView:d,fieldBView:_,fieldSize:[t,r]}}uploadBuffers(e){if(!this.resources)return;const t=new Float32Array(L*G);this.cellCount=Math.min(L,this.cells.length);for(let r=0;r<this.cellCount;r++){const i=this.cells[r],l=p(i.x),s=p(i.y),n=Math.hypot(l,s),c=p(i.phase);let a=0;i.selected&&(a|=1),i.scar&&(a|=2),p(i.energy)>.8&&(a|=4);const u=r*G;t[u+0]=l,t[u+1]=s,t[u+2]=Math.max(.006,p(i.radius,.02)),t[u+3]=n,t[u+4]=p(i.hue[0]),t[u+5]=p(i.hue[1]),t[u+6]=p(i.hue[2]),t[u+7]=O(i.energy),t[u+8]=c,t[u+9]=a,t[u+10]=O(i.metric2??i.energy),t[u+11]=p(i.spin??1,1),t[u+12]=r,t[u+13]=p(i.seed??c*97.13),t[u+14]=0,t[u+15]=0}e.queue.writeBuffer(this.resources.cellBuffer,0,t)}compute(e){const t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);const r=this.resources,i=e.beginRenderPass({label:"living-field-splat-pass",colorAttachments:[{view:r.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});i.setPipeline(this.splatPipeline),i.setBindGroup(0,r.splatBindGroup),this.cellCount>0&&i.draw(6,this.cellCount),i.end();const l=e.beginRenderPass({label:"living-field-blur-h",colorAttachments:[{view:r.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});l.setPipeline(this.blurPipeline),l.setBindGroup(0,r.blurHBindGroup),l.draw(6,1),l.end();const s=e.beginRenderPass({label:"living-field-blur-v",colorAttachments:[{view:r.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});s.setPipeline(this.blurPipeline),s.setBindGroup(0,r.blurVBindGroup),s.draw(6,1),s.end()}render(e){!this.resources||!this.membranePipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.cellBindGroup),e.draw(6,this.cellCount)))}orbitCpu(e,t,r,i){const l=Math.hypot(e,t);if(l<1e-4)return{x:e,y:t};const s=this.engine.params[10]||0,n=Math.atan2(t,e),c=(.045+r*.1)*s*i,a=n+c+Math.sin(s*.6+r*Math.PI*2)*.02,u=l*(1+.016*Math.sin(s*1.1+r*Math.PI*2));return{x:Math.cos(a)*u,y:Math.sin(a)*u}}pickAt(e,t){let r=null,i=1/0;for(let l=0;l<this.cellCount;l++){const s=this.cells[l],n=this.orbitCpu(s.x,s.y,s.phase,s.spin??1),c=Math.hypot(e-n.x,t-n.y),a=Math.max(.04,s.radius*2.6);c<=a&&c<i&&(r={id:s.pickId,kind:s.kind??"living-cell",index:l,payload:s.payload??s},i=c)}return r}dispose(){var e,t,r,i,l;(e=this.resources)==null||e.cellBuffer.destroy(),(t=this.resources)==null||t.blurHBuffer.destroy(),(r=this.resources)==null||r.blurVBuffer.destroy(),(i=this.resources)==null||i.fieldA.destroy(),(l=this.resources)==null||l.fieldB.destroy(),this.resources=null}}function O(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:0))}function p(o,e=0){return Number.isFinite(o)?o:e}function S(o,e,t){o.pushErrorScope("validation");const r=o.createShaderModule({label:e,code:t});return r.getCompilationInfo().then(i=>{for(const l of i.messages)l.type==="error"&&console.error(`[living-field] ${e} WGSL ${l.type} ${l.lineNum}:${l.linePos} ${l.message}`)}),o.popErrorScope().then(i=>{i&&console.error(`[living-field] ${e} shader module validation: ${i.message}`)}),r}const V=2.399963229728653;function X(o,e={}){const t=o.length;if(t===0)return[];const r=e.maxRadius??.92,i=e.minCellR??.012,l=e.maxCellR??.05;return o.map((n,c)=>({d:n,i:c})).sort((n,c)=>(c.d.score||0)-(n.d.score||0)).map(({d:n},c)=>{const a=t>1?c/(t-1):0,u=r*Math.sqrt(.06+.94*a),d=c*V,_=Math.cos(d)*u,g=Math.sin(d)*u,m=y(n.score),b=i+(l-i)*Math.sqrt(m);return{x:_,y:g,radius:b,hue:T(n.hue,m),energy:y(n.energy??.35+.65*m),phase:c/t,pickId:n.id,kind:n.kind,payload:n.payload??n,selected:n.selected,scar:n.scar,metric2:y(n.metric2??m),spin:1}})}function Y(o,e,t={}){if(o.length===0)return[];const i=t.maxRadius??.9,l=t.minCellR??.014,s=t.maxCellR??.05,n=Math.max(1,t.ringCount??new Set(o.map(e)).size),c=new Map;return o.map((a,u)=>{const d=e(a,u),g=((Number.isFinite(d)?Math.floor(d):0)%n+n)%n,m=c.get(g)??0;c.set(g,m+1);const b=i*(.18+.82*(g/Math.max(1,n-1))),B=m*V+g*.7,v=y(a.score);return{x:Math.cos(B)*b,y:Math.sin(B)*b,radius:l+(s-l)*Math.sqrt(v),hue:T(a.hue,v),energy:y(a.energy??.35+.65*v),phase:g/n,pickId:a.id,kind:a.kind,payload:a.payload??a,selected:a.selected,scar:a.scar,metric2:y(a.metric2??v),spin:1}})}const j={oxygen:h(x.luciferin),healthy:h(x.healthy),recall:h(x.recall),bridge:h(x.bridge),debt:h(x.debt),scarlet:h(C.veto),caution:h(C.caution),forward:h(F.forward),retrograde:h(F.retrograde)};function y(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:0))}function T(o,e){const t=o??I(e);return[Number.isFinite(t[0])?t[0]:0,Number.isFinite(t[1])?t[1]:0,Number.isFinite(t[2])?t[2]:0]}export{j as F,W as L,Y as a,X as l};

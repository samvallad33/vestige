import{A as e,B as t,C as n,D as r,G as i,J as a,K as o,N as s,Q as c,S as l,T as u,W as d,X as f,Y as p,a as m,b as h,d as g,dt as _,et as v,f as y,h as b,j as x,l as S,mt as C,ot as w,p as T,q as E,r as D,st as O,tt as k,u as ee,ut as A,w as j,x as te,z as M}from"../chunks/CvawvY3d.js";import"../chunks/xihTtKlq.js";import{t as ne}from"../chunks/CJst9gMq.js";import{a as re,c as ie,o as ae,r as oe,t as se}from"../chunks/But8FtBT.js";import{t as N}from"../chunks/26sNC5Du.js";import{t as ce}from"../chunks/CDRqZ3bt.js";import{t as P}from"../chunks/BRP91MLf.js";import{t as le}from"../chunks/BlLR51gI.js";import{i as F,o as I,r as ue,u as L}from"../chunks/Bbp_g6ip.js";import{t as de}from"../chunks/B68FPEVE.js";import{i as fe,n as pe,t as me}from"../chunks/DltdpSW9.js";import{n as he,r as ge,t as R}from"../chunks/Dd-iqbPw.js";import{n as _e,t as ve}from"../chunks/sCa_AqoL.js";var ye=C({load:()=>z}),z=({url:e})=>le(`/blackbox`,e),B=r(`<div class="path svelte-1wdzvwu"> </div>`),V=r(`<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Activation path</span> <!></div>`),be=r(`<code class="chip recall svelte-1wdzvwu"> </code>`),H=r(`<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Retrieved</span> <div class="chips svelte-1wdzvwu"></div></div>`),xe=r(`<code class="chip suppress svelte-1wdzvwu"> </code>`),U=r(`<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Suppressed</span> <div class="chips svelte-1wdzvwu"></div></div>`),Se=r(`<!> <!> <!>`,1),W=r(`<div><div class="r-head svelte-1wdzvwu"><code class="r-id svelte-1wdzvwu"> </code> <span class="r-risk svelte-1wdzvwu"> </span></div> <div class="r-metrics svelte-1wdzvwu"><div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">retrieved</span></div> <div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">suppressed</span></div> <div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">trust floor</span></div></div> <!> <button class="cinema-btn svelte-1wdzvwu"><!> Open exact receipt in Observatory</button></div>`);function Ce(e,t){O(t,!0);let r=m(t,`compact`,3,!1),a={low:`var(--color-recall, #10b981)`,medium:`#f59e0b`,high:`#f43f5e`};function c(){(t.receipt.retrieved.length||t.receipt.suppressed.length)&&ve(`/observatory`,{receipt:t.receipt.receipt_id})}var l=W();let f,p;var v=d(l),b=d(v),S=o(b,!0),C=E(b,2);let D;var k=o(C);_(v);var ee=E(v,2),te=d(ee),ne=d(te),re=o(ne,!0);A(2),_(te);var ie=E(te,2),ae=d(ie),oe=o(ae,!0);A(2),_(ie);var se=E(ie,2),N=d(se),ce=o(N);A(2),_(se),_(ee);var le=E(ee,2),F=e=>{var r=Se(),a=i(r),c=e=>{var n=V(),r=E(d(n),2);h(r,16,()=>t.receipt.activation_path,e=>e,(e,t)=>{var n=B(),r=o(n,!0);M(()=>j(r,t)),u(e,n)}),_(n),u(e,n)};n(a,e=>{t.receipt.activation_path.length&&e(c)});var l=E(a,2),f=e=>{var n=H(),r=E(d(n),2);h(r,20,()=>t.receipt.retrieved,e=>e,(e,t)=>{var n=be(),r=o(n,!0);M(e=>j(r,e),[()=>t.slice(0,8)]),u(e,n)}),_(r),_(n),u(e,n)};n(l,e=>{t.receipt.retrieved.length&&e(f)});var p=E(l,2),m=e=>{var n=U(),r=E(d(n),2);h(r,21,()=>t.receipt.suppressed,e=>e.id,(e,t)=>{var n=xe(),r=o(n);M((e,i)=>{g(n,`title`,s(t).reason),j(r,`${e??``} · ${i??``}`)},[()=>s(t).id.slice(0,8),()=>s(t).reason.replace(`_`,` `)]),u(e,n)}),_(r),_(n),u(e,n)};n(p,e=>{t.receipt.suppressed.length&&e(m)}),u(e,r)};n(le,e=>{r()||e(F)});var I=E(le,2),ue=d(I);P(ue,{name:`sparkle`,size:14}),A(),_(I),_(l),M(e=>{f=T(l,1,`receipt svelte-1wdzvwu`,null,f,{compact:r()}),p=y(l,``,p,{"--risk":a[t.receipt.decay_risk]}),j(S,t.receipt.receipt_id),D=y(C,``,D,{color:a[t.receipt.decay_risk]}),j(k,`decay: ${t.receipt.decay_risk??``}`),j(re,t.receipt.retrieved.length),j(oe,t.receipt.suppressed.length),j(ce,`${e??``}%`),I.disabled=!t.receipt.retrieved.length&&!t.receipt.suppressed.length},[()=>(t.receipt.trust_floor*100).toFixed(0)]),x(`click`,I,c),u(e,l),w()}e([`click`]);function we(e){switch(e){case`mcp.call`:return`var(--color-synapse-glow, #22C7DE)`;case`memory.retrieve`:return`var(--color-recall, #10b981)`;case`memory.suppress`:return`#FF3B30`;case`memory.write`:return`#38bdf8`;case`contradiction.detected`:return`#fb7185`;case`sanhedrin.veto`:return`#f43f5e`;case`dream.patch`:return`#29F2A9`;default:return`var(--color-synapse, #22C7DE)`}}function Te(e){switch(e){case`mcp.call`:return`Tool Call`;case`memory.retrieve`:return`Retrieved`;case`memory.suppress`:return`Suppressed`;case`memory.write`:return`Wrote`;case`contradiction.detected`:return`Contradiction`;case`sanhedrin.veto`:return`Veto`;case`dream.patch`:return`Dream Patch`;default:return e}}function Ee(e){switch(e){case`mcp.call`:return`⟐`;case`memory.retrieve`:return`◉`;case`memory.suppress`:return`⊘`;case`memory.write`:return`✎`;case`contradiction.detected`:return`⚡`;case`sanhedrin.veto`:return`⛔`;case`dream.patch`:return`☾`;default:return`•`}}function De(e){switch(e.type){case`mcp.call`:return`${e.tool}  ·  args ${e.argsHash.slice(0,8)}`;case`memory.retrieve`:return`${e.ids.length} ${e.ids.length===1?`memory`:`memories`} surfaced`;case`memory.suppress`:return`${e.id.slice(0,8)} — ${e.reason.replace(`_`,` `)}`;case`memory.write`:return`${e.id.slice(0,8)} — ${e.source}`;case`contradiction.detected`:return e.detail;case`sanhedrin.veto`:return`"${e.claim}" (conf ${(e.confidence*100).toFixed(0)}%)`;case`dream.patch`:return`${e.proposalIds.length} consolidation proposal(s)`;default:return``}}function Oe(e){switch(e.type){case`memory.retrieve`:return e.ids;case`memory.suppress`:case`memory.write`:return[e.id];case`contradiction.detected`:return e.ids;case`sanhedrin.veto`:return e.evidenceIds;case`dream.patch`:return e.proposalIds;default:return[]}}function ke(e){return!Number.isFinite(e)||e<=0?`—`:new Date(e).toLocaleTimeString(void 0,{hour12:!1,hour:`2-digit`,minute:`2-digit`,second:`2-digit`})}function Ae(e,t){return Math.max(0,e-t)}var G=512,K=12,je=8,Me=128,q=`rgba16float`,J=`
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

struct TraceEventCell {
	// x order 0..1, y lane 0..6, z confidence, w event kind code
	order_lane_conf_kind: vec4f,
	// x frame start, y visible gate, z selected, w receipt flag
	timing_flags: vec4f,
	// x retrieved ids count, y suppress/write/veto strength, z run duration fraction, w spare
	metric: vec4f,
};

struct ReceiptBead {
	// xy NDC, z intensity, w event index
	pos_energy: vec4f,
	// x node-count, y receipt ordinal, zw spare
	beat: vec4f,
};
`,Y=`
${J}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> trace_events: array<TraceEventCell>;
@group(0) @binding(2) var<storage, read> receipt_beads: array<ReceiptBead>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

fn lane_y(lane: f32) -> f32 {
	return mix(0.66, -0.62, lane / 6.0);
}

fn event_x(order: f32) -> f32 {
	return mix(-0.84, 0.84, clamp(order, 0.0, 1.0));
}

fn lane_color(kind: f32, lane: f32, conf: f32) -> vec3f {
	let luciferin = vec3f(0.66, 1.0, 0.37);
	let cyan = vec3f(0.08, 0.78, 0.92);
	let green = vec3f(0.12, 0.95, 0.56);
	let scarlet = vec3f(1.0, 0.18, 0.12);
	let amber = vec3f(1.0, 0.64, 0.06);
	let violet = vec3f(0.45, 0.42, 1.0);
	let bone = vec3f(0.88, 1.0, 0.70);
	var col = cyan;
	if (kind < 0.5) { col = bone; }
	else if (kind < 1.5) { col = green; }
	else if (kind < 2.5) { col = scarlet; }
	else if (kind < 3.5) { col = luciferin; }
	else if (kind < 4.5) { col = amber; }
	else if (kind < 5.5) { col = scarlet; }
	else { col = violet; }
	return mix(col * 0.62, col, clamp(conf, 0.0, 1.0)) * (0.88 + lane * 0.025);
}

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) local_uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) color_energy: vec4f,
};

@vertex
fn vs_impulse(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let cell = trace_events[ii];
	let corner = QUAD[vi];
	let order = cell.order_lane_conf_kind.x;
	let lane = cell.order_lane_conf_kind.y;
	let conf = cell.order_lane_conf_kind.z;
	let kind = cell.order_lane_conf_kind.w;
	let visible = cell.timing_flags.y;
	let selected = cell.timing_flags.z;
	let t = params.frame - cell.timing_flags.x;
	let pulse = smoothstep(0.0, 18.0, t) * (1.0 - smoothstep(92.0, 180.0, t));
	let center = vec2f(event_x(order), lane_y(lane));
	let radius = vec2f(0.028 + 0.026 * conf + 0.012 * selected, 0.026 + 0.018 * conf + 0.010 * pulse);
	var out: VSOut;
	out.clip = vec4f(center + corner * radius, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(kind, lane, selected, visible);
	out.color_energy = vec4f(lane_color(kind, lane, conf), visible * (0.35 + conf * 0.75 + pulse * 0.38));
	return out;
}

@fragment
fn fs_impulse(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.local_uv);
	if (d > 1.0 || frag.misc.w < 0.5) { discard; }
	let core = exp(-d * d * 3.4) * frag.color_energy.a;
	let ring = smoothstep(0.88, 0.62, abs(d - 0.64)) * (0.18 + frag.misc.z * 0.62);
	return vec4f(frag.color_energy.rgb * (core + ring), 1.0);
}

@vertex
fn vs_lane(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let corner = QUAD[vi];
	let lane = f32(ii);
	let center = vec2f(0.0, lane_y(lane));
	let size = vec2f(0.93, 0.012);
	var out: VSOut;
	out.clip = vec4f(center + corner * size, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(0.0, lane, 0.0, 1.0);
	out.color_energy = vec4f(lane_color(lane, lane, 0.4), 0.12 + 0.03 * params.pulse);
	return out;
}

@fragment
fn fs_lane(frag: VSOut) -> @location(0) vec4f {
	let fade = smoothstep(1.0, 0.08, abs(frag.local_uv.x)) * smoothstep(1.0, 0.0, abs(frag.local_uv.y));
	return vec4f(frag.color_energy.rgb * frag.color_energy.a * fade, 1.0);
}

@vertex
fn vs_receipt(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let bead = receipt_beads[ii];
	let corner = QUAD[vi];
	let event_i = bead.pos_energy.w;
	let linked = trace_events[u32(max(0.0, event_i))];
	let center = vec2f(event_x(linked.order_lane_conf_kind.x), lane_y(linked.order_lane_conf_kind.y) - 0.05 - 0.012 * bead.beat.y);
	let radius = 0.017 + 0.005 * min(5.0, bead.beat.x);
	var out: VSOut;
	out.clip = vec4f(center + corner * radius, 0.0, 1.0);
	out.local_uv = corner;
	out.misc = vec4f(0.0, linked.order_lane_conf_kind.y, 0.0, linked.timing_flags.y);
	out.color_energy = vec4f(vec3f(0.90, 1.0, 0.70), bead.pos_energy.z * linked.timing_flags.y);
	return out;
}

@fragment
fn fs_receipt(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.local_uv);
	if (d > 1.0 || frag.misc.w < 0.5) { discard; }
	let bead = smoothstep(1.0, 0.0, d) + smoothstep(0.70, 0.58, abs(d - 0.64));
	return vec4f(frag.color_energy.rgb * frag.color_energy.a * bead, 1.0);
}
`,X=`
${J}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	let p = QUAD[vi];
	var out: VSOut;
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_membrane(frag: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, frag.uv);
	let density = clamp(f.r, 0.0, 4.0);
	let recall = clamp(f.g, 0.0, 4.0);
	let immune = clamp(f.b, 0.0, 4.0);
	let write_glow = clamp(f.a, 0.0, 4.0);
	let centerline = smoothstep(0.44, 0.02, abs(frag.uv.y - 0.5));
	let blackwater = vec3f(0.006, 0.014, 0.016);
	var color = blackwater * (0.24 + density * 0.11);
	color = color + vec3f(0.62, 1.0, 0.35) * recall * 0.10;
	color = color + vec3f(1.0, 0.18, 0.12) * immune * 0.16;
	color = color + vec3f(0.90, 1.0, 0.70) * write_glow * 0.10;
	color = color + vec3f(0.08, 0.70, 0.80) * centerline * (0.03 + 0.02 * params.pulse);
	let vignette = smoothstep(0.92, 0.22, distance(frag.uv, vec2f(0.5)));
	return vec4f(color * (0.45 + 0.55 * vignette) * params.brightness, 1.0);
}
`,Ne=`
struct BlurDir {
	dir: vec2f,
	_pad: vec2f,
};
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
	let p = QUAD[vi];
	var out: VSOut;
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}
@fragment
fn fs_blur(frag: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let step = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, frag.uv - step * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv - step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + step, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + step * 2.0, 0.0) * 0.06136;
	return acc;
}
`;function Z(e){switch(e){case`mcp.call`:return 0;case`memory.retrieve`:return 1;case`memory.suppress`:return 2;case`memory.write`:return 3;case`sanhedrin.veto`:return 4;case`contradiction.detected`:return 5;case`dream.patch`:return 6}}function Pe(e){return[`tool`,`retrieve`,`suppress`,`write`,`veto`,`contradiction`,`dream`].indexOf(e)}var Fe=class{engine;scene=null;resources=null;sampler=null;traceBindLayout=null;membraneBindLayout=null;blurBindLayout=null;impulsePipeline=null;lanePipeline=null;receiptPipeline=null;blurPipeline=null;membranePipeline=null;eventCount=0;receiptCount=0;hitRects=[];constructor(e,t){this.engine=e,this.uploadScene(t)}uploadScene(e){this.scene=e,this.buildHitRects();let t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.uploadBuffers(t))}ensurePipelines(e){if(this.impulsePipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`blackbox-trace-wgsl`,code:Y}),n=e.createShaderModule({label:`blackbox-membrane-wgsl`,code:X}),r=e.createShaderModule({label:`blackbox-blur-wgsl`,code:Ne});this.traceBindLayout=e.createBindGroupLayout({label:`blackbox-trace-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:`blackbox-membrane-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}}]}),this.blurBindLayout=e.createBindGroupLayout({label:`blackbox-blur-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let i=e.createPipelineLayout({label:`blackbox-trace-layout`,bindGroupLayouts:[this.traceBindLayout]}),a=e.createPipelineLayout({label:`blackbox-membrane-layout`,bindGroupLayouts:[this.membraneBindLayout]}),o=e.createPipelineLayout({label:`blackbox-blur-layout`,bindGroupLayouts:[this.blurBindLayout]});this.sampler=e.createSampler({magFilter:`linear`,minFilter:`linear`});let s={color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}};this.impulsePipeline=e.createRenderPipeline({label:`blackbox-impulses`,layout:i,vertex:{module:t,entryPoint:`vs_impulse`},fragment:{module:t,entryPoint:`fs_impulse`,targets:[{format:q,blend:s}]},primitive:{topology:`triangle-list`}}),this.lanePipeline=e.createRenderPipeline({label:`blackbox-lanes`,layout:i,vertex:{module:t,entryPoint:`vs_lane`},fragment:{module:t,entryPoint:`fs_lane`,targets:[{format:q,blend:s}]},primitive:{topology:`triangle-list`}}),this.receiptPipeline=e.createRenderPipeline({label:`blackbox-receipt-beads`,layout:i,vertex:{module:t,entryPoint:`vs_receipt`},fragment:{module:t,entryPoint:`fs_receipt`,targets:[{format:q,blend:s}]},primitive:{topology:`triangle-list`}}),this.blurPipeline=e.createRenderPipeline({label:`blackbox-field-blur`,layout:o,vertex:{module:r,entryPoint:`vs_fullscreen`},fragment:{module:r,entryPoint:`fs_blur`,targets:[{format:q}]},primitive:{topology:`triangle-list`}}),this.membranePipeline=e.createRenderPipeline({label:`blackbox-field-membrane`,layout:a,vertex:{module:n,entryPoint:`vs_fullscreen`},fragment:{module:n,entryPoint:`fs_membrane`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-list`}})}ensureResources(e){if(!this.traceBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;let t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),n=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==n,i=this.resources?.eventBuffer,a=this.resources?.receiptBuffer,o=this.resources?.blurHBuffer,s=this.resources?.blurVBuffer;if(i||=e.createBuffer({label:`blackbox-event-cells`,size:G*K*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),a||=e.createBuffer({label:`blackbox-receipt-beads`,size:4096,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),o||(o=e.createBuffer({label:`blackbox-blur-h-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(o,0,new Float32Array([1,0,0,0]))),s||(s=e.createBuffer({label:`blackbox-blur-v-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([0,1,0,0]))),!r&&this.resources)return;this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy();let c=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,l=e.createTexture({label:`blackbox-field-a-rgba16float`,size:[t,n],format:q,usage:c}),u=e.createTexture({label:`blackbox-field-b-rgba16float`,size:[t,n],format:q,usage:c}),d=l.createView(),f=u.createView(),p=e.createBindGroup({label:`blackbox-trace-bind`,layout:this.traceBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:a}}]}),m=e.createBindGroup({label:`blackbox-membrane-bind`,layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:d}]}),h=e.createBindGroup({label:`blackbox-blur-h-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:d},{binding:2,resource:{buffer:o}}]}),g=e.createBindGroup({label:`blackbox-blur-v-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:f},{binding:2,resource:{buffer:s}}]});this.resources={eventBuffer:i,receiptBuffer:a,blurHBuffer:o,blurVBuffer:s,traceBindGroup:p,membraneBindGroup:m,blurHBindGroup:h,blurVBindGroup:g,fieldA:l,fieldB:u,fieldAView:d,fieldBView:f,fieldSize:[t,n]}}uploadBuffers(e){if(!this.resources||!this.scene)return;let t=Math.min(G,this.scene.visibleEventCount||this.scene.traceEvents.length),n=Math.max(1,this.scene.traceEvents.length-1),r=new Float32Array(G*K);this.eventCount=Math.min(G,this.scene.traceEvents.length);for(let e=0;e<this.eventCount;e++){let i=this.scene.traceEvents[e],a=+(e<t),o=+(e===this.scene.selectedIndex),s=Pe(i.lane),c=Z(i.type),l=i.memoryIds.length,u=i.type===`memory.suppress`||i.type===`sanhedrin.veto`||i.type===`contradiction.detected`?1:i.type===`memory.write`?.75:.35;r.set([e/n,s,i.confidence,c,e*34+18,a,o,0,l,u,n?e/n:0,0],e*K)}e.queue.writeBuffer(this.resources.eventBuffer,0,r);let i=new Float32Array(1024);this.receiptCount=Math.min(Me,this.scene.receipts.length);for(let e=0;e<this.receiptCount;e++){let n=Math.min(Math.max(0,t-1),this.eventCount-1);i.set([0,0,.65,n,this.scene.receipts[e].nodeIndices.length,e,0,0],e*je)}e.queue.writeBuffer(this.resources.receiptBuffer,0,i),this.engine.params[4]=this.eventCount}buildHitRects(){let e=this.scene?.traceEvents??[],t=Math.max(1,e.length-1);this.hitRects=e.map((e,n)=>({event:e,x:-.84+n/t*1.68,y:.66-1.28*(Pe(e.lane)/6),w:.055,h:.065}))}compute(e){let t=this.engine.gpuDevice;if(!t||!this.resources||!this.impulsePipeline||!this.lanePipeline||!this.receiptPipeline||!this.blurPipeline)return;this.ensureResources(t);let n=this.resources,r=e.beginRenderPass({label:`blackbox-field-splat-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});r.setBindGroup(0,n.traceBindGroup),r.setPipeline(this.lanePipeline),r.draw(6,7),this.eventCount>0&&(r.setPipeline(this.impulsePipeline),r.draw(6,this.eventCount)),this.receiptCount>0&&this.eventCount>0&&(r.setPipeline(this.receiptPipeline),r.draw(6,this.receiptCount)),r.end();let i=e.beginRenderPass({label:`blackbox-field-blur-h-pass`,colorAttachments:[{view:n.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});i.setPipeline(this.blurPipeline),i.setBindGroup(0,n.blurHBindGroup),i.draw(6,1),i.end();let a=e.beginRenderPass({label:`blackbox-field-blur-v-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});a.setPipeline(this.blurPipeline),a.setBindGroup(0,n.blurVBindGroup),a.draw(6,1),a.end()}render(e){this.resources&&this.membranePipeline&&this.impulsePipeline&&this.lanePipeline&&this.receiptPipeline&&(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),e.setBindGroup(0,this.resources.traceBindGroup),e.setPipeline(this.lanePipeline),e.draw(6,7),this.eventCount>0&&(e.setPipeline(this.impulsePipeline),e.draw(6,this.eventCount)),this.receiptCount>0&&this.eventCount>0&&(e.setPipeline(this.receiptPipeline),e.draw(6,this.receiptCount)))}pickAt(e,t){for(let n of this.hitRects)if(Math.abs(e-n.x)<=n.w&&Math.abs(t-n.y)<=n.h)return{id:n.event.id,kind:`trace-event`,index:n.event.index,payload:n.event};return null}dispose(){this.resources?.eventBuffer.destroy(),this.resources?.receiptBuffer.destroy(),this.resources?.blurHBuffer.destroy(),this.resources?.blurVBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}};function Ie(e,t){return L(I.healthy),L(F.veto),L(ue.forward),[new Fe(e,t)]}function Le(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function Q(e,t=0){return typeof e==`number`&&Number.isFinite(e)?e:t}function Re(e,t,n){return{kind:`trace`,id:`${e??`trace:none`}:event:${t}:${n}`}}function ze(e,t){return{kind:`scalar`,id:`blackbox.${e}`,scalar:{name:e,value:t}}}function Be(e,t,n){return{kind:`event`,id:`${e??`trace:none`}:event:${t}:${n}`}}function Ve(e){return{kind:`memory`,id:e}}function He(e){switch(e){case`mcp.call`:return`tool`;case`memory.retrieve`:return`retrieve`;case`memory.suppress`:return`suppress`;case`memory.write`:return`write`;case`sanhedrin.veto`:return`veto`;case`contradiction.detected`:return`contradiction`;case`dream.patch`:return`dream`}}function Ue(e){switch(e.type){case`memory.retrieve`:return e.ids;case`memory.suppress`:return[e.id];case`memory.write`:return[e.id];case`contradiction.detected`:return e.ids;case`sanhedrin.veto`:return e.evidenceIds;case`dream.patch`:return e.proposalIds;case`mcp.call`:return[]}}function We(e){switch(e.type){case`mcp.call`:return e.tool;case`memory.retrieve`:return`${e.ids.length} memories retrieved`;case`memory.suppress`:return`suppressed ${e.id.slice(0,8)}`;case`memory.write`:return`wrote ${e.id.slice(0,8)}`;case`contradiction.detected`:return`contradiction ${e.ids.join(` ↔ `)}`;case`sanhedrin.veto`:return`Sanhedrin veto`;case`dream.patch`:return`${e.proposalIds.length} dream proposals`}}function Ge(e){switch(e.type){case`mcp.call`:return`MCP tool ${e.tool} called; args hash ${e.argsHash.slice(0,12)}`;case`memory.retrieve`:return`Retrieved ${e.ids.length} memories with activation map`;case`memory.suppress`:return`Suppressed ${e.id}: ${e.reason}`;case`memory.write`:return`Memory write ${e.id} from ${e.source}`;case`contradiction.detected`:return e.detail;case`sanhedrin.veto`:return`${Math.round(Le(e.confidence)*100)}% veto confidence: ${e.claim}`;case`dream.patch`:return`Dream patch proposals: ${e.proposalIds.join(`, `)}`}}function Ke(e){return e.type===`memory.retrieve`?Object.entries(e.activation??{}).map(([e,t])=>({id:e,activation:Le(t)})).sort((e,t)=>t.activation-e.activation):[]}function qe(e){if(e.type===`sanhedrin.veto`)return Le(e.confidence);if(e.type===`memory.retrieve`){let t=Object.values(e.activation??{});return t.length?Le(t.reduce((e,t)=>e+t,0)/t.length):.45}return e.type===`memory.suppress`?.78:e.type===`memory.write`?.72:e.type===`contradiction.detected`?.82:e.type===`dream.patch`?.52:.62}function Je(e,t=[],n){let r=e?.runId??null,i=e?.events??[],a=e?.summary??null,o=Q(a?.startedAt,i[0]?.at??0),s=Q(a?.lastAt,i[i.length-1]?.at??o),c=i.length?i.length-1:0,l=Math.max(0,Math.min(c,n??c)),u=i.length?l+1:0,d=i.map((e,t)=>({index:t,id:`${r??e.runId}:event:${t}:${e.type}`,type:e.type,lane:He(e.type),runId:e.runId,at:e.at,label:We(e),summary:Ge(e),memoryIds:Ue(e),activationPairs:Ke(e),confidence:qe(e),provenance:Re(r,t,e.type),raw:e})),f=new Map;for(let e of d)for(let t of e.memoryIds){if(!t||f.has(t))continue;let n=e.activationPairs.find(e=>e.id===t)?.activation??0;f.set(t,{source:Ve(t),index:f.size,label:t.slice(0,12),retention:Math.max(.25,n),activation:n,trust:e.type===`memory.suppress`?.2:Math.max(.35,e.confidence),suppression:+(e.type===`memory.suppress`),tags:[e.lane,e.type],type:`trace-memory`})}let p=[...f.values()],m=d.slice(0,u).map(e=>({source:Be(r,e.index,e.type),type:e.type,targetIndex:e.memoryIds.length?f.get(e.memoryIds[0])?.index??-1:-1,frame:e.index*34+18,energy:Math.max(.18,e.confidence)})),h=d.slice(0,u).flatMap(e=>{let t=e.memoryIds.filter(e=>f.has(e));return t.length<2?[]:t.slice(1).map((n,r)=>({source:{kind:`pair`,id:`${e.id}:pair:${t[0]}:${n}`},sourceIndex:f.get(t[0]).index,targetIndex:f.get(n).index,weight:e.activationPairs[r+1]?.activation??e.confidence,kind:e.type}))}),g=t??[],_=g.map(e=>{let t=[...new Set([...e.retrieved??[],...e.activation_path??[],...(e.mutations??[]).map(e=>e.id)])];return{source:{kind:`receipt`,id:e.receipt_id},label:`receipt ${e.receipt_id.slice(0,10)}`,nodeIndices:t.map(e=>f.get(e)?.index).filter(e=>typeof e==`number`)}}),v={organ:`blackbox`,nodes:p,edges:h,events:m,receipts:_,scalars:{eventCount:i.length,visibleEventCount:u,retrievedCount:Q(a?.retrievedCount,d.filter(e=>e.type===`memory.retrieve`).length),suppressedCount:Q(a?.suppressedCount,d.filter(e=>e.type===`memory.suppress`).length),writeCount:Q(a?.writeCount,d.filter(e=>e.type===`memory.write`).length),vetoCount:Q(a?.vetoCount,d.filter(e=>e.type===`sanhedrin.veto`).length),durationMs:Math.max(0,s-o),receiptCount:_.length},alive:i.length>0,runId:r,traceEvents:d,visibleEventCount:u,selectedIndex:l,startedAt:o,lastAt:s,durationMs:Math.max(0,s-o),receiptRows:g};return v.alive||(v.receipts=g.map(e=>({source:{kind:`receipt`,id:e.receipt_id},label:`receipt ${e.receipt_id.slice(0,10)}`,nodeIndices:[]})),v.scalars.eventCount=0,v.scalars.visibleEventCount=0,ze(`empty`,0)),v}var Ye=64,Xe=96,Ze=16,Qe=8,$e=`
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

struct WitnessState {
	playhead: f32,
	replay_start: f32,
	selected_index: f32,
	shard_count: f32,
};

struct Shard {
	// xyz = deterministic 3D location, w = wafer base scale
	position_size: vec4f,
	// x activation, y retention, z trace-time 0..1, w selected
	metrics: vec4f,
	// real status color; semantic, never decorative
	color: vec4f,
	// x role, y scar flag, z reveal order, w reserved
	flags: vec4f,
};

struct Filament {
	// x source shard index, y target shard index (only verified path neighbors)
	endpoints: vec4f,
	// x energy, y deterministic phase, z receipt-path flag, w reserved
	motion: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> shards: array<Shard>;
@group(0) @binding(2) var<storage, read> filaments: array<Filament>;
@group(0) @binding(3) var<uniform> witness: WitnessState;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

// A witness shard is a small, extruded ceramic specimen, not a flat UI card.
// The front face remains deliberately asymmetric; the two side faces make the
// receipt structure legible as a volume even when the chamber is completely
// still.  The depth is part of the actual perspective projection below.
const WAFER = array<vec3f, 18>(
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(-1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0), vec3f(-1.34, -1.42, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.34, 0.58, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.34, 0.58, -1.0), vec3f(0.66, -1.42, -1.0)
);

struct Projection { screen: vec2f, scale: f32 };

// This is a real perspective projection, not a 2D arrangement. Time is depth
// and activation/role form the stable chamber strata. The pointer only shifts
// the 3/4 view by a few degrees: an examination lens, never an auto-orbit.
fn cursor_lens() -> vec2f {
	if (abs(params.cursor_x) > 2.0 || abs(params.cursor_y) > 2.0) {
		return vec2f(0.0, 0.0);
	}
	return clamp(vec2f(params.cursor_x, params.cursor_y), vec2f(-1.0), vec2f(1.0));
}

fn project(world: vec3f) -> Projection {
	let lens = cursor_lens();
	let yaw = lens.x * 0.055;
	let c = cos(yaw);
	let s = sin(yaw);
	let view = vec3f(
		world.x * c - world.z * s,
		world.y + lens.y * 0.035,
		world.x * s + world.z * c
	);
	let depth = clamp(2.82 - view.z, 1.1, 5.4);
	let perspective = 1.0 / depth;
	return Projection(vec2f(view.x * 1.18 * perspective, view.y * 1.62 * perspective), perspective);
}

fn smooth01(value: f32) -> f32 {
	let t = clamp(value, 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

fn reveal_for(shard: Shard) -> f32 {
	let arrival = 14.0 + shard.flags.z * 68.0;
	let ingress = smooth01((params.frame - arrival) / 62.0);
	// The temporal slicer is a real trace cursor. Evidence that was not yet
	// available simply does not materialize.
	let slice = smoothstep(shard.metrics.z - 0.045, shard.metrics.z + 0.09, witness.playhead);
	return ingress * slice;
}

fn replay_age() -> f32 {
	if (witness.replay_start < 0.0) { return 9999.0; }
	var age = params.frame - witness.replay_start;
	if (age < 0.0) { age = age + 720.0; }
	return age;
}

// Quiet mineral palette: jade marks corroborated evidence; the traversal itself
// is fossil amber. There is intentionally no cyan/purple emissive wash.
fn core_color() -> vec3f { return vec3f(0.10, 0.17, 0.14); }

struct WaferOut {
	@builtin(position) clip: vec4f,
	@location(0) local: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) data: vec4f,
	@location(3) @interpolate(flat) face: f32,
};

@vertex
fn vs_wafer(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> WaferOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let sealed = vec3f(0.08, -0.02, 0.12);
	let selectedLift = select(vec3f(0.0), vec3f(0.0, 0.0, 0.34), shard.metrics.w > 0.5);
	let position = mix(sealed, shard.position_size.xyz + selectedLift, reveal);
	let q = WAFER[vi];
	let rotation = -0.15 + (shard.flags.z - 0.5) * 0.30 + (shard.flags.x - 1.5) * 0.075;
	let c = cos(rotation);
	let s = sin(rotation);
	let rq = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	let height = (0.112 + shard.metrics.y * 0.072) * shard.position_size.w;
	let width = height * (2.05 + shard.metrics.x * 0.82);
	let thickness = height * (0.46 + shard.metrics.y * 0.14);
	// The face offset is projected with the specimen, rather than pasted onto
	// the screen. It is a true low-poly wafer with parallax depth.
	let facePosition = position + vec3f(rq.x * width, rq.y * height, q.z * thickness);
	let projected = project(facePosition);
	var out: WaferOut;
	out.clip = vec4f(projected.screen, 0.0, 1.0);
	out.local = q.xy;
	out.color = shard.color.rgb;
	out.data = vec4f(shard.metrics.x, shard.metrics.y, shard.metrics.w, shard.flags.y);
	out.face = floor(f32(vi) / 6.0);
	return out;
}

@fragment
fn fs_wafer(frag: WaferOut) -> @location(0) vec4f {
	let edge = max(abs(frag.local.y), abs(frag.local.x) * 0.76 + frag.local.y * 0.12);
	if (frag.face < 0.5 && edge > 1.0) { discard; }
	let rim = smoothstep(0.72, 0.97, edge);
	let facet = smoothstep(-1.05, 1.05, frag.local.x * 0.78 - frag.local.y * 0.42);
	let scar = frag.data.w;
	let selected = frag.data.z;
	let carbon = vec3f(0.018, 0.031, 0.033);
	var body = mix(carbon, frag.color * (0.14 + facet * 0.18), 0.74);
	if (frag.face > 0.5) {
		let sideLight = select(0.24, 0.42, frag.face > 1.5);
		body = mix(carbon * 1.45, frag.color * sideLight, 0.66);
	} else {
		body = body + frag.color * rim * (0.16 + selected * 0.30);
	}
	if (scar > 0.5) {
		let fracture = smoothstep(0.10, 0.02, abs(frag.local.x + frag.local.y * 0.37));
		body = mix(body, vec3f(0.72, 0.12, 0.09), fracture * 0.76);
	}
	let alpha = select(0.72, 0.86 + selected * 0.14, frag.face < 0.5);
	return vec4f(body, alpha);
}

struct RibbonOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) energy: f32,
	@location(3) @interpolate(flat) selected: f32,
};

fn filament_point(index: f32) -> vec3f {
	if (index < -0.5) { return vec3f(0.08, -0.02, 0.12); }
	let shard = shards[u32(index)];
	return mix(vec3f(0.0, 0.0, 0.0), shard.position_size.xyz, reveal_for(shard));
}

fn filament_selected(index: f32) -> f32 {
	if (index < -0.5) { return 0.0; }
	return shards[u32(index)].metrics.w;
}

@vertex
fn vs_filament(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let fiber = filaments[ii];
	let start = project(filament_point(fiber.endpoints.x));
	let end = project(filament_point(fiber.endpoints.y));
	let delta = end.screen - start.screen;
	let distance = max(length(delta), 0.0001);
	let direction = delta / distance;
	let normal = vec2f(-direction.y, direction.x);
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	let width = 0.0017 + fiber.motion.x * 0.0025;
	let selected = max(filament_selected(fiber.endpoints.x), filament_selected(fiber.endpoints.y));
	var out: RibbonOut;
	out.clip = vec4f(mix(start.screen, end.screen, t) + normal * q.y * width, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = mix(core_color(), vec3f(0.46, 0.30, 0.15), fiber.motion.z);
	out.energy = fiber.motion.x;
	out.selected = selected;
	return out;
}

@fragment
fn fs_filament(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.24, abs(frag.uv.y));
	let age = replay_age();
	let travel = fract(age * 0.016 + frag.energy * 0.37);
	let wrapped = abs(fract(frag.uv.x - travel + 0.5) - 0.5);
	let pulse = exp(-wrapped * wrapped * 980.0) * select(0.0, 1.0, age < 176.0);
	let alpha = body * (0.105 + frag.selected * 0.30 + pulse * 0.72);
	return vec4f(frag.color * (0.32 + pulse * 0.74), alpha);
}

// A self-emitted arrival trail for each real receipt member. It only exists
// while the wafer is entering the chamber; it is not an always-on decorative
// network.
@vertex
fn vs_arrival_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let previous = max(0.0, reveal - 0.13);
	let a = project(mix(vec3f(0.0), shard.position_size.xyz, previous));
	let b = project(mix(vec3f(0.0), shard.position_size.xyz, reveal));
	let delta = b.screen - a.screen;
	let distance = max(length(delta), 0.0001);
	let normal = vec2f(-delta.y, delta.x) / distance;
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	var out: RibbonOut;
	out.clip = vec4f(mix(a.screen, b.screen, t) + normal * q.y * 0.006, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = shard.color.rgb;
	out.energy = reveal * (1.0 - smoothstep(0.92, 1.0, reveal));
	out.selected = shard.metrics.w;
	return out;
}

@fragment
fn fs_arrival_trail(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.18, abs(frag.uv.y));
	return vec4f(frag.color * (0.32 + frag.selected * 0.44), body * frag.energy * 0.58);
}

struct CoreOut { @builtin(position) clip: vec4f, @location(0) local: vec2f };

@vertex
fn vs_core(@builtin(vertex_index) vi: u32) -> CoreOut {
	let q = QUAD[vi];
	let lens = cursor_lens();
	let center = project(vec3f(0.08, -0.02, 0.12));
	let tilt = -0.14 + lens.x * 0.035;
	let c = cos(tilt);
	let s = sin(tilt);
	let rotated = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	var out: CoreOut;
	// The receipt is held inside a black archival spine. The slates form its
	// strata; this sealed volume replaces the generic graph's central node.
	out.clip = vec4f(center.screen + rotated * vec2f(0.265, 0.80) * (0.74 + center.scale), 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_core(frag: CoreOut) -> @location(0) vec4f {
	let diagonal = abs(frag.local.x) * 0.83 + frag.local.y * 0.10;
	if (max(abs(frag.local.y), diagonal) > 1.0) { discard; }
	let edge = smoothstep(0.79, 0.98, max(abs(frag.local.y), abs(diagonal)));
	let aperture = smoothstep(0.105, 0.022, abs(frag.local.x + frag.local.y * 0.08));
	let stratum = smoothstep(0.035, 0.004, abs(fract((frag.local.y + 1.0) * 2.9) - 0.5));
	let jade = vec3f(0.25, 0.43, 0.33);
	let amber = vec3f(0.43, 0.28, 0.14);
	var color = vec3f(0.008, 0.016, 0.016);
	color = color + vec3f(0.026, 0.051, 0.045) * (1.0 - abs(frag.local.x)) * 0.75;
	color = color + jade * edge * 0.24;
	color = color + jade * stratum * 0.11;
	color = color + amber * aperture * 0.34;
	return vec4f(color, 0.94);
}
`;function $(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function et(e){return[`retrieved`,`path`,`mutation`,`suppressed`].indexOf(e.role)}function tt(e){return e.suppressed?L(`#ab5a51`):e.mutated?L(`#c58a4a`):e.role===`path`?L(`#5faf8a`):L(`#e5e2d8`)}function nt(e,t,n){let r=1/Math.max(1.1,Math.min(5.4,2.82-n));return{x:e*1.18*r,y:t*1.62*r,scale:r}}var rt=class{engine;scene=null;resources=null;bindLayout=null;waferPipeline=null;filamentPipeline=null;arrivalPipeline=null;corePipeline=null;shardCount=0;filamentCount=0;selectedId=null;playhead=1;replayStart=-1;shardData=new Float32Array(1024);hitTargets=[];constructor(e,t){this.engine=e,this.uploadScene(t)}uploadScene(e){this.scene=e,this.selectedId=this.scene.shards[0]?.id??null,this.playhead=1,this.replayStart=-1;let t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.writeScene(t))}setSelected(e){this.selectedId=e;let t=this.engine.gpuDevice;t&&this.resources&&(this.writeScene(t),this.engine.requestRender())}setPlayhead(e){this.playhead=$(e),this.writeState(),this.engine.requestRender()}replay(){this.replayStart=this.engine.demoClock.state.frame,this.writeState(),this.engine.requestRender()}targetFrameRate(e){if(e<154)return 60;if(this.replayStart>=0){let t=this.engine.demoClock.framesPerLoop;if((e-this.replayStart+t)%t<196)return 60}return 6}ensurePipelines(e){if(this.waferPipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`witness-volume-wgsl`,code:$e});this.bindLayout=e.createBindGroupLayout({label:`witness-volume-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:3,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let n=e.createPipelineLayout({label:`witness-volume-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`src-alpha`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}},i=(i,a,o)=>e.createRenderPipeline({label:i,layout:n,vertex:{module:t,entryPoint:a},fragment:{module:t,entryPoint:o,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`,cullMode:`none`}});this.filamentPipeline=i(`witness-volume-filaments`,`vs_filament`,`fs_filament`),this.arrivalPipeline=i(`witness-volume-arrival-trails`,`vs_arrival_trail`,`fs_arrival_trail`),this.waferPipeline=i(`witness-volume-evidence-wafers`,`vs_wafer`,`fs_wafer`),this.corePipeline=i(`witness-volume-receipt-core`,`vs_core`,`fs_core`)}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;let t=e.createBuffer({label:`witness-volume-shards`,size:4096,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:`witness-volume-filaments`,size:3072,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r=e.createBuffer({label:`witness-volume-state`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=e.createBindGroup({label:`witness-volume-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}},{binding:3,resource:{buffer:r}}]});this.resources={shardBuffer:t,filamentBuffer:n,stateBuffer:r,bindGroup:i}}writeState(){let e=this.engine.gpuDevice;if(!e||!this.resources)return;let t=this.scene?.shards.findIndex(e=>e.id===this.selectedId)??-1;e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.playhead,this.replayStart,t,this.shardCount]))}writeScene(e){if(!this.resources||!this.scene)return;let t=this.scene.shards.slice(0,Ye);this.shardCount=t.length,this.shardData.fill(0),this.hitTargets=[];for(let e=0;e<t.length;e+=1){let n=t[e],r=$(n.traceTime);t.length<=1||n.order/(t.length-1);let i=e%2==0?-1:1,a=(n.order*.61803398875%1-.5)*.22,o=.08+($(n.activation)-.5)*1.42+i*.19+a,s=1.16-r*2.36+i*.075,c=-.3+r*1.18+$(n.retention)*.15-(n.suppressed?.2:0),l=tt(n),u=+(n.id===this.selectedId),d=e*Ze;this.shardData.set([o,s,c,.92+n.retention*.36,$(n.activation),$(n.retention),$(r),u,l[0],l[1],l[2],1,et(n),+!!n.suppressed,e/Math.max(1,t.length-1),+!!n.mutated],d);let f=nt(o,s,c);this.hitTargets.push({shard:n,x:f.x,y:f.y,radius:(.29+n.retention*.14)*f.scale*2.25})}e.queue.writeBuffer(this.resources.shardBuffer,0,this.shardData);let n=new Float32Array(768),r=0;for(let e of this.scene.edges){if(r>=Xe||e.sourceIndex<0||e.targetIndex<0)break;n.set([e.sourceIndex,e.targetIndex,0,0,$(e.weight),r*.137,1,0],r*Qe),r+=1}this.filamentCount=r,e.queue.writeBuffer(this.resources.filamentBuffer,0,n),this.engine.params[2]=this.shardCount,this.engine.params[3]=this.filamentCount,this.writeState()}render(e){this.resources&&this.waferPipeline&&this.filamentPipeline&&this.arrivalPipeline&&this.corePipeline&&(e.setBindGroup(0,this.resources.bindGroup),this.filamentCount>0&&(e.setPipeline(this.filamentPipeline),e.draw(6,this.filamentCount)),this.shardCount>0&&(e.setPipeline(this.arrivalPipeline),e.draw(6,this.shardCount),e.setPipeline(this.waferPipeline),e.draw(18,this.shardCount)),e.setPipeline(this.corePipeline),e.draw(6,1))}pickAt(e,t){let n=null,r=1/0;for(let i of this.hitTargets){let a=i.x-e,o=i.y-t,s=Math.hypot(a,o);s<=i.radius&&s<r&&(n=i,r=s)}return n?{id:n.shard.id,kind:`witness-shard`,payload:n.shard}:null}dispose(){this.resources?.shardBuffer.destroy(),this.resources?.filamentBuffer.destroy(),this.resources?.stateBuffer.destroy(),this.resources=null}};function it(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function at(e){return{kind:`memory`,id:e}}function ot(e,t,n){return{kind:`trace`,id:`${e??`none`}:${t}:${n.type}`}}function st(e){if(!e)return[];let t=[...e.retrieved,...e.mutations.map(e=>e.id),...e.suppressed.map(e=>e.id)],n=new Set(t),r=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,i=e.activation_path.filter(e=>n.has(e)||r.test(e));return[...new Set([...i,...t].filter(Boolean))]}function ct(e,t){return t.suppressed.some(t=>t.id===e)?`suppressed`:t.mutations.some(t=>t.id===e)?`mutation`:t.activation_path.includes(e)?`path`:`retrieved`}function lt(e,t){for(let n=t.length-1;n>=0;--n){let r=t[n];if(r.type===`memory.retrieve`&&typeof r.activation[e]==`number`)return it(r.activation[e])}return .48}function ut(e,t){if(!t?.content)return`memory ${e.slice(0,10)}`;let n=t.content.replace(/\s+/g,` `).trim();return n.length>84?`${n.slice(0,81)}...`:n}function dt(e,t){if(!t.length)return 1;let n=t.findIndex(t=>t.type===`memory.retrieve`?t.ids.includes(e):t.type===`memory.suppress`||t.type===`memory.write`?t.id===e:t.type===`contradiction.detected`?t.ids.includes(e):t.type===`sanhedrin.veto`?t.evidenceIds.includes(e):t.type===`dream.patch`&&t.proposalIds.includes(e));return n<0?1:it((n+1)/t.length)}function ft(e,t,n){let r=e?.runId??null;if(!t)return{organ:`witness`,nodes:[],edges:[],events:[],receipts:[],scalars:{eventCount:e?.events.length??0,evidenceCount:0},alive:!1,runId:r,receiptId:null,shards:[],eventCount:e?.events.length??0};let i=e?.events??[],a=st(t).slice(0,64).map((e,r)=>{let a=n.get(e),o=ct(e,t);return{id:e,label:ut(e,a),content:a?.content??``,role:o,activation:lt(e,i),retention:it(a?.retentionStrength??.5),traceTime:dt(e,i),order:r,suppressed:o===`suppressed`,mutated:o===`mutation`,provenance:at(e)}}),o=new Map(a.map((e,t)=>[e.id,t])),s=a.map((e,n)=>({source:e.provenance,index:n,label:e.label,retention:e.retention,activation:e.activation,trust:t.trust_floor,suppression:+!!e.suppressed,tags:[e.role],type:`witness-shard`})),c=t.activation_path.slice(1).flatMap((e,n)=>{let r=t.activation_path[n],i=o.get(r),a=o.get(e);return i===void 0||a===void 0?[]:[{source:{kind:`receipt`,id:`${t.receipt_id}:path:${n}`},sourceIndex:i,targetIndex:a,weight:Math.max(s[i].activation??0,s[a].activation??0),kind:`receipt-path`}]});return{organ:`witness`,nodes:s,edges:c,events:i.map((e,t)=>{let n=(e.type===`memory.retrieve`?e.ids:e.type===`memory.suppress`||e.type===`memory.write`?[e.id]:e.type===`contradiction.detected`?e.ids:e.type===`sanhedrin.veto`?e.evidenceIds:e.type===`dream.patch`?e.proposalIds:[]).map(e=>o.get(e)).find(e=>e!==void 0)??-1;return{source:ot(r,t,e),type:e.type,targetIndex:n,frame:18+t*24,energy:e.type===`memory.retrieve`?.86:e.type===`memory.suppress`?.72:.52}}),receipts:[{source:{kind:`receipt`,id:t.receipt_id},label:`receipt ${t.receipt_id.slice(0,12)}`,nodeIndices:a.map((e,t)=>t)}],scalars:{eventCount:i.length,evidenceCount:a.length,pathLength:c.length,trustFloor:t.trust_floor,suppressedCount:t.suppressed.length,mutationCount:t.mutations.length},alive:a.length>0,runId:r,receiptId:t.receipt_id,shards:a,eventCount:i.length}}var pt=r(`<button title="Toggle 3D witness volume vs DOM forensics"><!> </button> <button title="Proof Mode: a clean launch-footage view"><!> Proof Mode</button> <button class="export-btn svelte-1ayqwv0"><!> </button> <button class="export-btn svelte-1ayqwv0"><!> Export .vestige-trace.json</button>`,1),mt=r(`<span class="ev-chip svelte-1ayqwv0"> </span>`),ht=r(`<span class="text-dim svelte-1ayqwv0">awaiting…</span>`),gt=r(`<p class="rescue-banner svelte-1ayqwv0"> </p>`),_t=r(`<p class="rescue-error svelte-1ayqwv0"> </p>`),vt=r(`<div class="rescue-failure svelte-1ayqwv0"><span class="svelte-1ayqwv0">Failure</span> <a class="svelte-1ayqwv0"> </a> <small class="svelte-1ayqwv0"> </small></div>`),yt=r(`<li class="svelte-1ayqwv0"><a class="svelte-1ayqwv0"> </a> <span class="svelte-1ayqwv0"> </span> <small class="svelte-1ayqwv0"> </small></li>`),bt=r(`<ul class="rescue-causes svelte-1ayqwv0"></ul>`),xt=r(`<button type="button" class="svelte-1ayqwv0">Replay in Observatory →</button>`),St=r(`<a class="svelte-1ayqwv0">Open rescue run</a>`),Ct=r(`<p class="svelte-1ayqwv0"> </p> <!> <!> <div class="rescue-actions svelte-1ayqwv0"><!> <!></div>`,1),wt=r(`<div class="rescue-panel glass svelte-1ayqwv0"><div class="rescue-kicker svelte-1ayqwv0">SALIENCE RESCUE · PREVIEW</div> <!> <!> <!></div>`),Tt=r(`<div class="selected-impulse glass svelte-1ayqwv0"><span class="selected-kicker svelte-1ayqwv0">GPU pick</span> <strong class="svelte-1ayqwv0"> </strong> <span class="svelte-1ayqwv0"> </span> <code class="svelte-1ayqwv0"> </code></div>`),Et=r(`<p class="empty svelte-1ayqwv0"> </p>`),Dt=r(`<p class="empty svelte-1ayqwv0">No agent runs recorded yet. Make an MCP tool call — every call is
						recorded here.</p>`),Ot=r(`<span class="s-recall svelte-1ayqwv0"> </span>`),kt=r(`<span class="s-suppress svelte-1ayqwv0"> </span>`),At=r(`<span class="s-write svelte-1ayqwv0"> </span>`),jt=r(`<span class="s-veto svelte-1ayqwv0"> </span>`),Mt=r(`<li class="svelte-1ayqwv0"><button><div class="run-top svelte-1ayqwv0"><code class="run-id svelte-1ayqwv0"> </code> <span class="run-tool svelte-1ayqwv0"> </span></div> <div class="run-stats svelte-1ayqwv0"><span title="events" class="svelte-1ayqwv0"> </span> <!> <!> <!> <!></div></button></li>`),Nt=r(`<ul class="svelte-1ayqwv0"></ul>`),Pt=r(`<div class="glass center-msg svelte-1ayqwv0">Loading trace…</div>`),Ft=r(`<div class="glass center-msg err svelte-1ayqwv0"> </div>`),It=r(`<div class="glass center-msg svelte-1ayqwv0">Select a run to replay.</div>`),Lt=r(`<span class="scrub-time svelte-1ayqwv0"> </span>`),Rt=r(`<button></button>`),zt=r(`<small class="svelte-1ayqwv0"> </small>`),Bt=r(`<span class="id-chip svelte-1ayqwv0"><code class="svelte-1ayqwv0"> </code> <!></span>`),Vt=r(`<div class="ids-grid svelte-1ayqwv0"></div>`),Ht=r(`<span class="loser svelte-1ayqwv0"> </span>`),Ut=r(`<div class="contra svelte-1ayqwv0"><span class="winner svelte-1ayqwv0"> </span> <span class="vs svelte-1ayqwv0">vs</span> <!></div>`),Wt=r(`<code class="svelte-1ayqwv0"> </code>`),Gt=r(`<div class="veto-evidence svelte-1ayqwv0"></div>`),Kt=r(`<div class="event-detail glass svelte-1ayqwv0"><div class="ed-head svelte-1ayqwv0"><span class="ed-glyph svelte-1ayqwv0"> </span> <span class="ed-label svelte-1ayqwv0"> </span> <code class="ed-time svelte-1ayqwv0"> </code></div> <p class="ed-summary svelte-1ayqwv0"> </p> <!></div>`),qt=r(`<p class="empty svelte-1ayqwv0">No memories touched yet.</p>`),Jt=r(`<code class="pulse-node svelte-1ayqwv0"> </code>`),Yt=r(`<div class="pulse-grid svelte-1ayqwv0"></div>`),Xt=r(`<div class="receipts-panel glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Receipts <span class="text-dim svelte-1ayqwv0">· proof behind retrievals</span></h3> <div class="receipts-grid svelte-1ayqwv0"></div></div>`),Zt=r(`<a class="svelte-1ayqwv0"> </a>`),Qt=r(`<div class="log-jumps svelte-1ayqwv0"></div>`),$t=r(`<li><button class="log-btn svelte-1ayqwv0"><span class="log-glyph svelte-1ayqwv0"> </span> <span class="log-label svelte-1ayqwv0"> </span> <span class="log-summary svelte-1ayqwv0"> </span> <span class="log-t svelte-1ayqwv0"> </span></button> <!></li>`),en=r(`<div class="scrubber glass svelte-1ayqwv0"><div class="scrub-head svelte-1ayqwv0"><span class="scrub-title svelte-1ayqwv0">Step <strong class="svelte-1ayqwv0"> </strong> </span> <!></div> <input type="range" min="0" class="scrub-range svelte-1ayqwv0"/> <div class="ticks svelte-1ayqwv0"></div></div> <!> <div class="pulse glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Memory pulse <span class="text-dim svelte-1ayqwv0">· touched this run</span></h3> <!></div> <div class="producers glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Event producers <span class="text-dim svelte-1ayqwv0">· this run</span></h3> <ul class="producer-list svelte-1ayqwv0"><li class="producer ok svelte-1ayqwv0"><span class="p-dot svelte-1ayqwv0"></span> mcp.call · memory.write · memory.retrieve · memory.suppress <span class="p-state svelte-1ayqwv0">live</span></li> <li><span class="p-dot svelte-1ayqwv0"></span> contradiction.detected <span class="p-state svelte-1ayqwv0"> </span></li> <li><span class="p-dot svelte-1ayqwv0"></span> dream.patch <span class="p-state svelte-1ayqwv0"> </span></li> <li><span class="p-dot svelte-1ayqwv0"></span> sanhedrin.veto <span class="p-state svelte-1ayqwv0"> </span></li></ul></div> <!> <div class="log glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Event log</h3> <ol class="log-list svelte-1ayqwv0"></ol></div>`,1),tn=r(`<div class="layout svelte-1ayqwv0"><aside class="runs glass svelte-1ayqwv0"><h2 class="panel-title svelte-1ayqwv0">Runs</h2> <!></aside> <section class="replay svelte-1ayqwv0"><!></section></div>`),nn=r(`<div class="proof-event svelte-1ayqwv0"><span class="proof-glyph svelte-1ayqwv0"> </span> <div class="svelte-1ayqwv0"><div class="proof-ev-label svelte-1ayqwv0"> </div> <div class="proof-ev-sum svelte-1ayqwv0"> </div></div></div>`),rn=r(`<div class="proof-stage glass svelte-1ayqwv0"><div class="proof-headline svelte-1ayqwv0"><span></span> <code class="proof-run svelte-1ayqwv0"> </code></div> <!> <div class="proof-counter svelte-1ayqwv0"><!> <span class="proof-counter-label svelte-1ayqwv0">trace events</span></div> <p class="proof-tagline svelte-1ayqwv0">Watch the agent think. Watch memory change. Watch the receipt prove why.</p></div>`),an=r(`<!> <div class="blackbox-shell relative z-10 mx-auto max-w-6xl px-5 py-6 svelte-1ayqwv0"><!> <div class="spine glass svelte-1ayqwv0"><div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">WebSocket</span> <span><span></span> </span></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Live runId</span> <code class="spine-run svelte-1ayqwv0"> </code></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Last event</span> <span class="spine-value svelte-1ayqwv0"><!></span></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Events seen</span> <span class="spine-value svelte-1ayqwv0"><!></span></div></div> <!> <!> <!></div>`,1);function on(e,r){O(r,!0);let m=()=>k(ne,`$page`,L),C=()=>k(re,`$lastTraceEvent`,L),le=()=>k(se,`$eventFeed`,L),F=()=>k(oe,`$isConnected`,L),I=()=>k(ae,`$liveRunId`,L),ue=()=>k(ie,`$traceEvents`,L),[L,ye]=v(),z=f(a([])),B=f(null),V=f(null),be=f(!1),H=f(null),xe=f(null),U=f(0),Se=f(!1),W=f(a([])),G=f(null),K=f(`forensics`),je=f(a(new Map)),Me=null,q=f(!1),J=f(null),Y=f(null),X=f(null),Ne=c(()=>s(V)?s(V).events.slice(0,s(U)+1):[]),Z=c(()=>s(V)&&s(V).events.length?s(V).events[s(U)]:null),Pe=c(()=>s(V)?.events[0]?.at??0),Fe=c(()=>Array.from(new Set(s(Ne).flatMap(Oe)))),Le=c(()=>s(V)?.events.some(e=>e.type===`sanhedrin.veto`)??!1),Q=c(()=>s(V)?.events.some(e=>e.type===`dream.patch`)??!1),Re=c(()=>s(V)?.events.some(e=>e.type===`contradiction.detected`)??!1),ze=c(()=>Je(s(V),s(W),s(U))),Be=c(()=>s(W)[0]??null),Ve=c(()=>ft(s(V),s(Be),s(je)));function He(e){if(e.kind===`witness-shard`){Me?.setSelected(e.id);return}if(e.kind!==`trace-event`)return;let t=e.payload;p(G,t,!0),p(U,Math.max(0,Math.min(s(V)?.events.length?s(V).events.length-1:0,t.index)),!0)}async function Ue(){try{let e=await N.traces.list(100);p(z,e.runs,!0),p(xe,null);let t=m().url.searchParams.get(`run`);m().url.searchParams.get(`view`)===`volume`&&p(K,`volume`),!s(B)&&t&&s(z).some(e=>e.runId===t)?We(t):!s(B)&&s(z).length&&We(s(z)[0].runId)}catch(e){p(xe,e instanceof Error?e.message:String(e),!0),p(H,s(xe),!0)}}async function We(e){p(B,e,!0),p(be,!0),p(H,null);try{p(V,await N.traces.get(e),!0),p(U,Math.max(0,(s(V).events.length||1)-1),!0),p(G,null),p(W,(await N.receipts.listForRun(e,8)).receipts,!0),await qe(s(W)[0]??null)}catch(e){p(H,String(e),!0),p(V,null)}finally{p(be,!1)}}function Ge(){s(B)&&(window.location.href=N.traces.exportUrl(s(B)))}function Ke(e){return e?[...e.activation_path,...e.retrieved,...e.mutations.map(e=>e.id),...e.suppressed.map(e=>e.id)].filter((e,t,n)=>!!e&&n.indexOf(e)===t):[]}async function qe(e){let t=Ke(e).slice(0,64).filter(e=>!s(je).has(e));if(!t.length)return;let n=await Promise.all(t.map(async e=>{try{return await N.memories.get(e)}catch{return null}})),r=new Map(s(je));for(let e of n)e&&r.set(e.id,e);p(je,r,!0)}function Ye(e,t){let n=new rt(e,t);return Me=n,[n]}async function Xe(){if(!s(q)){p(q,!0),p(J,null);try{let e=await N.backfill({manual:!0,promote:!1,lookbackDays:30,runId:s(B)??void 0});p(Y,e,!0),e.receiptId&&p(X,e.headline??`Salience rescue receipt sealed`,!0)}catch(e){p(J,e instanceof Error?e.message:`Backfill failed`,!0)}finally{p(q,!1)}}}function Ze(){let e=s(Y)?.receiptId??s(Y)?.receipt?.receipt_id;e&&ve(`/observatory`,{receipt:e})}function Qe(e){p(K,e,!0),e===`volume`&&qe(s(Be))}t(()=>{let e=C();if(!e)return;let t=e.data?.run_id;t&&t===s(B)&&N.traces.get(s(B)).then(e=>{p(V,e,!0),p(U,Math.max(0,e.events.length-1),!0)})}),D(Ue),t(()=>{let e=le()[0];e&&ce(e)&&p(X,`Salience rescue fired`)});let $e=null;function $(){let e=Math.max(1,...s(z).map(e=>e.eventCount??0)),t=s(z).slice(0,200).map(t=>{let n=et((t.eventCount??0)/e),r=(t.vetoCount??0)>0?me.scarlet:(t.suppressedCount??0)>0?me.caution:me.forward;return{id:t.runId,score:.35+.65*n,hue:r,energy:.4+.6*n,selected:t.runId===s(B),scar:(t.vetoCount??0)>0,metric2:et((t.retrievedCount??0)/Math.max(1,t.eventCount??1)),kind:`trace-run`,payload:t}});return pe(t,{maxRadius:.95,minCellR:.016,maxCellR:.06})}function et(e){return Math.min(1,Math.max(0,Number.isFinite(e)?e:0))}function tt(e,t){let n=new fe(e);return $e=n,n.setIntensity(.24),n.setReadingWell({x:0,y:0,hw:.66,hh:.85,floor:.08,soft:.25}),n.setCells($()),[{compute:e=>n.compute(e),render:e=>n.render(e),pickAt:(e,t)=>n.pickAt(e,t),dispose:()=>{n.dispose(),$e===n&&($e=null)}},...Ie(e,t)]}t(()=>{s(z).length,s(B),$e?.setCells($())});var nt=an(),it=i(nt);l(it,()=>s(K),e=>{{let t=c(()=>s(K)===`volume`?`witness`:`blackbox`),n=c(()=>`agent-flight-recorder:${s(K)}:${s(B)??`empty`}:${s(ze).visibleEventCount}`),r=c(()=>s(K)===`volume`?s(Ve):s(ze)),i=c(()=>s(K)===`volume`?Ye:tt),a=c(()=>s(K)===`volume`?`NO RECEIPT SELECTED - WITNESS CHAMBER ARMED`:`NO AGENT TRACE SELECTED - RECORDER ARMED`);de(e,{get organ(){return s(t)},get seed(){return s(n)},get scene(){return s(r)},get passes(){return s(i)},get loading(){return s(be)},get error(){return s(H)},get emptyLabel(){return s(a)},onpick:He})}});var at=E(it,2),ot=d(at);ge(ot,{icon:`blackbox`,title:`Agent Black Box`,subtitle:`Watch the agent think. Watch memory change. Watch the receipt prove why.`,accent:`synapse`,children:(e,t)=>{var n=pt(),r=i(n);let a;var o=d(r);P(o,{name:`graph`,size:14});var c=E(o);_(r);var l=E(r,2);let f;var m=d(l);P(m,{name:`sparkle`,size:14}),A(),_(l);var h=E(l,2),g=d(h);P(g,{name:`sparkle`,size:14});var v=E(g);_(h);var y=E(h,2),b=d(y);P(b,{name:`feed`,size:14}),A(),_(y),M(()=>{a=T(r,1,`mode-toggle svelte-1ayqwv0`,null,a,{on:s(K)===`volume`}),j(c,` ${s(K)===`volume`?`Forensics`:`Witness volume`}`),f=T(l,1,`mode-toggle svelte-1ayqwv0`,null,f,{on:s(Se)}),h.disabled=s(q),j(v,` ${s(q)?`Rescuing…`:`Run salience rescue (preview)`}`),y.disabled=!s(B)}),x(`click`,r,()=>Qe(s(K)===`volume`?`forensics`:`volume`)),x(`click`,l,()=>p(Se,!s(Se))),x(`click`,h,()=>void Xe()),x(`click`,y,Ge),u(e,n)},$$slots:{default:!0}});var st=E(ot,2),ct=d(st),lt=E(d(ct),2);let ut;var dt=d(lt);let on;var sn=E(dt);_(lt),_(ct);var cn=E(ct,2),ln=E(d(cn),2),un=o(ln,!0);_(cn);var dn=E(cn,2),fn=E(d(dn),2),pn=d(fn),mn=e=>{var t=mt();let n;var r=o(t,!0);M((e,i)=>{n=y(t,``,n,{"--c":e}),j(r,i)},[()=>we(C().data?.event?.type),()=>Te(C().data?.event?.type)]),u(e,t)},hn=e=>{var t=ht();u(e,t)};n(pn,e=>{C()?e(mn):e(hn,-1)}),_(fn),_(dn);var gn=E(dn,2),_n=E(d(gn),2),vn=d(_n);he(vn,{get value(){return ue().length}}),_(_n),_(gn),_(st),b(st,e=>R?.(e));var yn=E(st,2),bn=e=>{var t=wt(),r=E(d(t),2),a=e=>{var t=gt(),n=o(t,!0);M(()=>j(n,s(X))),u(e,t)};n(r,e=>{s(X)&&e(a)});var c=E(r,2),l=e=>{var t=_t(),n=o(t,!0);M(()=>j(n,s(J))),u(e,t)};n(c,e=>{s(J)&&e(l)});var f=E(c,2),p=e=>{var t=Ct(),r=i(t),a=o(r,!0),c=E(r,2),l=e=>{var t=vt(),n=E(d(t),2),r=o(n,!0),i=E(n,2),a=o(i,!0);_(t),M(e=>{g(n,`href`,e),j(r,s(Y).failure.id),j(a,s(Y).failure.content_preview)},[()=>_e(`/memories`,{memory:s(Y).failure.id})]),u(e,t)};n(c,e=>{s(Y).failure&&e(l)});var f=E(c,2),p=e=>{var t=bt();h(t,21,()=>s(Y).causes.slice(0,8),e=>e.memory_id,(e,t)=>{var n=yt(),r=d(n),i=o(r,!0),a=E(r,2),c=o(a),l=E(a,2),f=o(l,!0);_(n),M((e,n,a)=>{g(r,`href`,e),j(i,n),j(c,`score ${a??``}`),j(f,s(t).content_preview)},[()=>_e(`/memories`,{memory:s(t).memory_id}),()=>s(t).memory_id.slice(0,8),()=>s(t).backfill_score.toFixed(2)]),u(e,n)}),_(t),u(e,t)};n(f,e=>{s(Y).causes?.length&&e(p)});var m=E(f,2),v=d(m),y=e=>{var t=xt();x(`click`,t,Ze),u(e,t)};n(v,e=>{(s(Y).receiptId||s(Y).receipt)&&e(y)});var b=E(v,2),S=e=>{var t=St();M(e=>g(t,`href`,e),[()=>_e(`/blackbox`,{run:s(Y).runId})]),u(e,t)};n(b,e=>{s(Y).runId&&e(S)}),_(m),M(()=>j(a,s(Y).headline??s(Y).note??`Preview complete. Nothing was promoted.`)),u(e,t)};n(f,e=>{s(Y)&&e(p)}),_(t),b(t,e=>R?.(e)),u(e,t)};n(yn,e=>{(s(X)||s(Y)||s(J))&&e(bn)});var xn=E(yn,2),Sn=e=>{var t=Tt(),n=E(d(t),2),r=o(n,!0),i=E(n,2),a=o(i,!0),c=E(i,2),l=o(c,!0);_(t),b(t,e=>R?.(e)),M(()=>{j(r,s(G).label),j(a,s(G).summary),j(l,s(G).provenance.id)}),u(e,t)};n(xn,e=>{s(G)&&e(Sn)});var Cn=E(xn,2),wn=e=>{var t=tn(),r=d(t),a=E(d(r),2),l=e=>{var t=Et(),n=o(t);M(()=>j(n,`Agent runs are not listed: ${s(xe)??``}`)),u(e,t)},f=e=>{var t=Dt();u(e,t)},m=e=>{var t=Nt();h(t,21,()=>s(z),e=>e.runId,(e,t)=>{var r=Mt(),i=d(r);let a;var c=d(i),l=d(c),f=o(l,!0),p=E(l,2),m=o(p,!0);_(c);var h=E(c,2),g=d(h),v=o(g),y=E(g,2),b=e=>{var n=Ot(),r=o(n);M(()=>j(r,`↑${s(t).retrievedCount??``}`)),u(e,n)};n(y,e=>{s(t).retrievedCount&&e(b)});var S=E(y,2),C=e=>{var n=kt(),r=o(n);M(()=>j(r,`⊘${s(t).suppressedCount??``}`)),u(e,n)};n(S,e=>{s(t).suppressedCount&&e(C)});var w=E(S,2),D=e=>{var n=At(),r=o(n);M(()=>j(r,`✎${s(t).writeCount??``}`)),u(e,n)};n(w,e=>{s(t).writeCount&&e(D)});var O=E(w,2),k=e=>{var n=jt(),r=o(n);M(()=>j(r,`⛔${s(t).vetoCount??``}`)),u(e,n)};n(O,e=>{s(t).vetoCount&&e(k)}),_(h),_(i),_(r),M(e=>{a=T(i,1,`run-row svelte-1ayqwv0`,null,a,{active:s(t).runId===s(B)}),j(f,e),j(m,s(t).firstTool??`—`),j(v,`${s(t).eventCount??``} ev`)},[()=>s(t).runId.replace(`run_`,``).slice(0,10)]),x(`click`,i,()=>We(s(t).runId)),u(e,r)}),_(t),u(e,t)};n(a,e=>{s(xe)?e(l):s(z).length===0?e(f,1):e(m,-1)}),_(r),b(r,e=>R?.(e));var v=E(r,2),C=d(v),w=e=>{var t=Pt();u(e,t)},D=e=>{var t=Ft(),n=o(t,!0);M(()=>j(n,s(H))),u(e,t)},O=e=>{var t=It();u(e,t)},k=e=>{var t=en(),r=i(t),a=d(r),l=d(a),f=E(d(l)),m=o(f,!0),v=E(f);_(l);var C=E(l,2),w=e=>{var t=Lt(),n=o(t);M(e=>j(n,`+${e??``}ms`),[()=>Ae(s(Z).at,s(Pe))]),u(e,t)};n(C,e=>{s(Z)&&e(w)}),_(a);var D=E(a,2);ee(D);var O=E(D,2);h(O,21,()=>s(V).events,te,(e,t,n)=>{var r=Rt();let i,a;M((e,t,o)=>{i=T(r,1,`tick svelte-1ayqwv0`,null,i,{past:n<=s(U)}),g(r,`title`,e),g(r,`aria-label`,t),a=y(r,``,a,{"--c":o})},[()=>Te(s(t).type),()=>`Step ${n+1}: ${Te(s(t).type)}`,()=>we(s(t).type)]),x(`click`,r,()=>p(U,n,!0)),u(e,r)}),_(O),_(r),b(r,e=>R?.(e));var k=E(r,2),A=e=>{var t=Kt();let r;var i=d(t),a=d(i),c=o(a,!0),l=E(a,2),f=o(l,!0),p=E(l,2),m=o(p,!0);_(i);var g=E(i,2),v=o(g,!0),x=E(g,2),S=e=>{var t=Vt();h(t,20,()=>s(Z).ids,e=>e,(e,t)=>{var r=Bt();let i;var a=d(r),c=o(a,!0),l=E(a,2),f=e=>{var n=zt(),r=o(n);M(e=>j(r,`${e??``}%`),[()=>(s(Z).activation[t]*100).toFixed(0)]),u(e,n)};n(l,e=>{s(Z).activation[t]!=null&&e(f)}),_(r),M(e=>{i=y(r,``,i,{"--a":s(Z).activation[t]??0}),j(c,e)},[()=>t.slice(0,8)]),u(e,r)}),_(t),u(e,t)},C=e=>{var t=Ut(),n=d(t),r=o(n),i=E(n,4);h(i,16,()=>s(Z).ids.filter(e=>e!==s(Z).winnerId),e=>e,(e,t)=>{var n=Ht(),r=o(n,!0);M(e=>j(r,e),[()=>t.slice(0,8)]),u(e,n)}),_(t),M(e=>j(r,`kept ${e??``}`),[()=>s(Z).winnerId?.slice(0,8)]),u(e,t)},w=e=>{var t=Gt();h(t,20,()=>s(Z).evidenceIds,e=>e,(e,t)=>{var n=Wt(),r=o(n,!0);M(e=>j(r,e),[()=>t.slice(0,8)]),u(e,n)}),_(t),u(e,t)};n(x,e=>{s(Z).type===`memory.retrieve`?e(S):s(Z).type===`contradiction.detected`?e(C,1):s(Z).type===`sanhedrin.veto`&&e(w,2)}),_(t),b(t,e=>R?.(e)),M((e,n,i,a,o)=>{r=y(t,``,r,{"--c":e}),j(c,n),j(f,i),j(m,a),j(v,o)},[()=>we(s(Z).type),()=>Ee(s(Z).type),()=>Te(s(Z).type),()=>ke(s(Z).at),()=>De(s(Z))]),u(e,t)};n(k,e=>{s(Z)&&e(A)});var ne=E(k,2),re=E(d(ne),2),ie=e=>{var t=qt();u(e,t)},ae=e=>{var t=Yt();h(t,20,()=>s(Fe),e=>e,(e,t)=>{var n=Jt(),r=o(n,!0);M(e=>j(r,e),[()=>t.slice(0,8)]),u(e,n)}),_(t),u(e,t)};n(re,e=>{s(Fe).length===0?e(ie):e(ae,-1)}),_(ne),b(ne,e=>R?.(e));var oe=E(ne,2),se=E(d(oe),2),N=E(d(se),2);let ce;var P=E(d(N),2),le=o(P,!0);_(N);var F=E(N,2);let I;var ue=E(d(F),2),L=o(ue,!0);_(F);var de=E(F,2);let fe;var pe=E(d(de),2),me=o(pe,!0);_(de),_(se),_(oe),b(oe,e=>R?.(e));var he=E(oe,2),ge=e=>{var t=Xt(),n=E(d(t),2);h(n,21,()=>s(W).slice(0,2),e=>e.receipt_id,(e,t)=>{Ce(e,{get receipt(){return s(t)}})}),_(n),_(t),b(t,e=>R?.(e)),u(e,t)};n(he,e=>{s(W).length&&e(ge)});var ve=E(he,2),ye=E(d(ve),2);h(ye,21,()=>s(V).events,te,(e,t,r)=>{var i=$t();let a,l;var f=d(i),m=d(f),v=o(m,!0),b=E(m,2),S=o(b,!0),C=E(b,2),w=o(C,!0),D=E(C,2),O=o(D);_(f);var k=E(f,2),ee=e=>{var n=Qt();h(n,20,()=>Oe(s(t)),e=>e,(e,t)=>{var n=Zt(),r=o(n,!0);M((e,t)=>{g(n,`href`,e),j(r,t)},[()=>_e(`/memories`,{memory:t}),()=>t.slice(0,8)]),u(e,n)}),_(n),u(e,n)},A=c(()=>Oe(s(t)).length);n(k,e=>{s(A)&&e(ee)}),_(i),M((e,t,n,o,c)=>{a=T(i,1,`log-row svelte-1ayqwv0`,null,a,{active:r===s(U),dim:r>s(U)}),l=y(i,``,l,{"--c":e}),j(v,t),j(S,n),j(w,o),j(O,`+${c??``}ms`)},[()=>we(s(t).type),()=>Ee(s(t).type),()=>Te(s(t).type),()=>De(s(t)),()=>Ae(s(t).at,s(Pe))]),x(`click`,f,()=>p(U,r,!0)),u(e,i)}),_(ye),_(ve),b(ve,e=>R?.(e)),M(e=>{j(m,s(U)+1),j(v,` / ${s(V).events.length??``}`),g(D,`max`,e),ce=T(N,1,`producer svelte-1ayqwv0`,null,ce,{ok:s(Re)}),j(le,s(Re)?`fired this run`:`no contradiction in this run`),I=T(F,1,`producer caveat svelte-1ayqwv0`,null,I,{ok:s(Q)}),j(L,s(Q)?`fired this run`:`No dream run in this trace`),fe=T(de,1,`producer caveat svelte-1ayqwv0`,null,fe,{ok:s(Le)}),j(me,s(Le)?`fired this run`:`No veto producer connected (optional Sanhedrin hook, off by default)`)},[()=>Math.max(0,s(V).events.length-1)]),S(D,()=>s(U),e=>p(U,e)),u(e,t)};n(C,e=>{s(be)?e(w):s(H)?e(D,1):s(V)?e(k,-1):e(O,2)}),_(v),_(t),u(e,t)},Tn=e=>{var t=rn(),r=d(t),i=d(r);let a;var l=E(i,2),f=o(l,!0);_(r);var p=E(r,2),m=e=>{let t=c(()=>C().data?.event);var n=nn();let r;var i=d(n),a=o(i,!0),l=E(i,2),f=d(l),p=o(f,!0),m=E(f,2),h=o(m,!0);_(l),_(n),M((e,t,i,o)=>{r=y(n,``,r,{"--c":e}),j(a,t),j(p,i),j(h,o)},[()=>we(s(t)?.type),()=>Ee(s(t)?.type),()=>Te(s(t)?.type),()=>De(s(t))]),u(e,n)};n(p,e=>{C()&&e(m)});var h=E(p,2),g=d(h);he(g,{get value(){return ue().length}}),A(2),_(h),A(2),_(t),b(t,e=>R?.(e)),M(()=>{a=T(i,1,`dot big svelte-1ayqwv0`,null,a,{live:F()}),j(f,I()??`awaiting run…`)}),u(e,t)};n(Cn,e=>{s(Se)?e(Tn,-1):e(wn)}),_(at),M(()=>{ut=T(lt,1,`spine-value svelte-1ayqwv0`,null,ut,{live:F()}),on=T(dt,1,`dot svelte-1ayqwv0`,null,on,{live:F()}),j(sn,` ${F()?`Connected`:`Offline`}`),j(un,I()??`—`)}),u(e,nt),w(),ye()}e([`click`]);export{on as component,ye as universal};
var kr=Object.defineProperty;var Pr=(t,r,l)=>r in t?kr(t,r,{enumerable:!0,configurable:!0,writable:!0,value:l}):t[r]=l;var j=(t,r,l)=>Pr(t,typeof r!="symbol"?r+"":r,l);import"../chunks/Bzak7iHL.js";import{d as ir,s as u,b as Xe,o as Sr}from"../chunks/DAau0uzT.js";import{p as lr,c as s,r as a,X as n,af as Ie,Y as B,a as y,b as or,aH as xt,f as _,g as e,d as Se,e as er,j as tr,s as U,u as ge}from"../chunks/CGq8RnJq.js";import{i as O}from"../chunks/Ccqjq5DS.js";import{e as fe,s as qt,r as Br,i as rr}from"../chunks/DqfV0sZu.js";import{P as Cr,A as ar,a as we,r as xe}from"../chunks/B9l3DI-J.js";import{s as pe}from"../chunks/uCQU803Y.js";import{s as Ge}from"../chunks/HFGAk8XQ.js";import{b as Mr}from"../chunks/DGM4cicq.js";import{p as Rr,a as gt,s as Er}from"../chunks/DV6OI5iy.js";import{I as Nt}from"../chunks/CKbQrCJw.js";import{g as Gr}from"../chunks/DJDK-KWF.js";import{a as lt}from"../chunks/D35IQVqe.js";import{l as Ir,t as Ar,a as Lr,b as zr}from"../chunks/Ch9vNiEl.js";import{R as Tr}from"../chunks/Cqe0su6e.js";import{r as zt,R as Ur,I as Or,C as Vr}from"../chunks/BMB5u1EX.js";import{L as $r,F as Tt,l as Fr}from"../chunks/BeGBxCGK.js";var Nr=_('<div class="path svelte-1wdzvwu"> </div>'),Dr=_('<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Activation path</span> <!></div>'),Hr=_('<code class="chip recall svelte-1wdzvwu"> </code>'),Wr=_('<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Retrieved</span> <div class="chips svelte-1wdzvwu"></div></div>'),jr=_('<code class="chip suppress svelte-1wdzvwu"> </code>'),Qr=_('<div class="r-section svelte-1wdzvwu"><span class="r-section-title svelte-1wdzvwu">Suppressed</span> <div class="chips svelte-1wdzvwu"></div></div>'),Xr=_("<!> <!> <!>",1),Yr=_('<div><div class="r-head svelte-1wdzvwu"><code class="r-id svelte-1wdzvwu"> </code> <span class="r-risk svelte-1wdzvwu"> </span></div> <div class="r-metrics svelte-1wdzvwu"><div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">retrieved</span></div> <div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">suppressed</span></div> <div class="metric svelte-1wdzvwu"><span class="m-val svelte-1wdzvwu"> </span> <span class="m-label svelte-1wdzvwu">trust floor</span></div></div> <!> <button class="cinema-btn svelte-1wdzvwu"><!> Open receipt in Cinema</button></div>');function Kr(t,r){lr(r,!0);let l=Rr(r,"compact",3,!1);const v={low:"var(--color-recall, #10b981)",medium:"#f59e0b",high:"#f43f5e"};function b(){const _e=r.receipt.retrieved[0];if(!_e)return;const ce=r.receipt.retrieved.join(",");Gr(`/graph?center=${encodeURIComponent(_e)}&focus=${encodeURIComponent(ce)}`)}var c=Yr();let f,P;var q=s(c),G=s(q),g=s(G,!0);a(G);var I=n(G,2);let M;var E=s(I);a(I),a(q);var re=n(q,2),ae=s(re),Q=s(ae),Be=s(Q,!0);a(Q),Ie(2),a(ae);var w=n(ae,2),le=s(w),me=s(le,!0);a(le),Ie(2),a(w);var oe=n(w,2),i=s(oe),k=s(i);a(i),Ie(2),a(oe),a(re);var V=n(re,2);{var ye=_e=>{var ce=Xr(),Ye=xt(ce);{var vt=ee=>{var W=Dr(),ve=n(s(W),2);fe(ve,16,()=>r.receipt.activation_path,X=>X,(X,be)=>{var Y=Nr(),Ce=s(Y,!0);a(Y),B(()=>u(Ce,be)),y(X,Y)}),a(W),y(ee,W)};O(Ye,ee=>{r.receipt.activation_path.length&&ee(vt)})}var ut=n(Ye,2);{var dt=ee=>{var W=Wr(),ve=n(s(W),2);fe(ve,20,()=>r.receipt.retrieved,X=>X,(X,be)=>{var Y=Hr(),Ce=s(Y,!0);a(Y),B(Ke=>u(Ce,Ke),[()=>be.slice(0,8)]),y(X,Y)}),a(ve),a(W),y(ee,W)};O(ut,ee=>{r.receipt.retrieved.length&&ee(dt)})}var pt=n(ut,2);{var ft=ee=>{var W=Qr(),ve=n(s(W),2);fe(ve,21,()=>r.receipt.suppressed,X=>X.id,(X,be)=>{var Y=jr(),Ce=s(Y);a(Y),B((Ke,Je)=>{qt(Y,"title",e(be).reason),u(Ce,`${Ke??""} · ${Je??""}`)},[()=>e(be).id.slice(0,8),()=>e(be).reason.replace("_"," ")]),y(X,Y)}),a(ve),a(W),y(ee,W)};O(pt,ee=>{r.receipt.suppressed.length&&ee(ft)})}y(_e,ce)};O(V,_e=>{l()||_e(ye)})}var qe=n(V,2),ct=s(qe);Nt(ct,{name:"sparkle",size:14}),Ie(),a(qe),a(c),B(_e=>{f=pe(c,1,"receipt svelte-1wdzvwu",null,f,{compact:l()}),P=Ge(c,"",P,{"--risk":v[r.receipt.decay_risk]}),u(g,r.receipt.receipt_id),M=Ge(I,"",M,{color:v[r.receipt.decay_risk]}),u(E,`decay: ${r.receipt.decay_risk??""}`),u(Be,r.receipt.retrieved.length),u(me,r.receipt.suppressed.length),u(k,`${_e??""}%`),qe.disabled=!r.receipt.retrieved.length},[()=>(r.receipt.trust_floor*100).toFixed(0)]),Xe("click",qe,b),y(t,c),or()}ir(["click"]);function ot(t){switch(t){case"mcp.call":return"var(--color-synapse-glow, #818cf8)";case"memory.retrieve":return"var(--color-recall, #10b981)";case"memory.suppress":return"#a78bfa";case"memory.write":return"#38bdf8";case"contradiction.detected":return"#fb7185";case"sanhedrin.veto":return"#f43f5e";case"dream.patch":return"#c084fc";default:return"var(--color-synapse, #6366f1)"}}function We(t){switch(t){case"mcp.call":return"Tool Call";case"memory.retrieve":return"Retrieved";case"memory.suppress":return"Suppressed";case"memory.write":return"Wrote";case"contradiction.detected":return"Contradiction";case"sanhedrin.veto":return"Veto";case"dream.patch":return"Dream Patch";default:return t}}function Ut(t){switch(t){case"mcp.call":return"⟐";case"memory.retrieve":return"◉";case"memory.suppress":return"⊘";case"memory.write":return"✎";case"contradiction.detected":return"⚡";case"sanhedrin.veto":return"⛔";case"dream.patch":return"☾";default:return"•"}}function Ot(t){switch(t.type){case"mcp.call":return`${t.tool}  ·  args ${t.argsHash.slice(0,8)}`;case"memory.retrieve":return`${t.ids.length} ${t.ids.length===1?"memory":"memories"} surfaced`;case"memory.suppress":return`${t.id.slice(0,8)} — ${t.reason.replace("_"," ")}`;case"memory.write":return`${t.id.slice(0,8)} — ${t.source}`;case"contradiction.detected":return t.detail;case"sanhedrin.veto":return`"${t.claim}" (conf ${(t.confidence*100).toFixed(0)}%)`;case"dream.patch":return`${t.proposalIds.length} consolidation proposal(s)`;default:return""}}function Jr(t){switch(t.type){case"memory.retrieve":return t.ids;case"memory.suppress":case"memory.write":return[t.id];case"contradiction.detected":return t.ids;case"sanhedrin.veto":return t.evidenceIds;case"dream.patch":return t.proposalIds;default:return[]}}function Zr(t){return!Number.isFinite(t)||t<=0?"—":new Date(t).toLocaleTimeString(void 0,{hour12:!1,hour:"2-digit",minute:"2-digit",second:"2-digit"})}function sr(t,r){return Math.max(0,t-r)}const wt=512,Vt=12,$t=8,Ft=128,je="rgba16float",cr=`
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
`,ea=`
${cr}

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
`,ta=`
${cr}

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
`,ra=`
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
`;function aa(t){switch(t){case"mcp.call":return 0;case"memory.retrieve":return 1;case"memory.suppress":return 2;case"memory.write":return 3;case"sanhedrin.veto":return 4;case"contradiction.detected":return 5;case"dream.patch":return 6}}function nr(t){return["tool","retrieve","suppress","write","veto","contradiction","dream"].indexOf(t)}class sa{constructor(r,l){j(this,"engine");j(this,"scene",null);j(this,"resources",null);j(this,"sampler",null);j(this,"traceBindLayout",null);j(this,"membraneBindLayout",null);j(this,"blurBindLayout",null);j(this,"impulsePipeline",null);j(this,"lanePipeline",null);j(this,"receiptPipeline",null);j(this,"blurPipeline",null);j(this,"membranePipeline",null);j(this,"eventCount",0);j(this,"receiptCount",0);j(this,"hitRects",[]);this.engine=r,this.uploadScene(l)}uploadScene(r){this.scene=r,this.buildHitRects();const l=this.engine.gpuDevice;l&&(this.ensurePipelines(l),this.ensureResources(l),this.uploadBuffers(l))}ensurePipelines(r){if(this.impulsePipeline||!this.engine.paramsBuffer)return;const l=r.createShaderModule({label:"blackbox-trace-wgsl",code:ea}),v=r.createShaderModule({label:"blackbox-membrane-wgsl",code:ta}),b=r.createShaderModule({label:"blackbox-blur-wgsl",code:ra});this.traceBindLayout=r.createBindGroupLayout({label:"blackbox-trace-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.membraneBindLayout=r.createBindGroupLayout({label:"blackbox-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]}),this.blurBindLayout=r.createBindGroupLayout({label:"blackbox-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const c=r.createPipelineLayout({label:"blackbox-trace-layout",bindGroupLayouts:[this.traceBindLayout]}),f=r.createPipelineLayout({label:"blackbox-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]}),P=r.createPipelineLayout({label:"blackbox-blur-layout",bindGroupLayouts:[this.blurBindLayout]});this.sampler=r.createSampler({magFilter:"linear",minFilter:"linear"});const q={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.impulsePipeline=r.createRenderPipeline({label:"blackbox-impulses",layout:c,vertex:{module:l,entryPoint:"vs_impulse"},fragment:{module:l,entryPoint:"fs_impulse",targets:[{format:je,blend:q}]},primitive:{topology:"triangle-list"}}),this.lanePipeline=r.createRenderPipeline({label:"blackbox-lanes",layout:c,vertex:{module:l,entryPoint:"vs_lane"},fragment:{module:l,entryPoint:"fs_lane",targets:[{format:je,blend:q}]},primitive:{topology:"triangle-list"}}),this.receiptPipeline=r.createRenderPipeline({label:"blackbox-receipt-beads",layout:c,vertex:{module:l,entryPoint:"vs_receipt"},fragment:{module:l,entryPoint:"fs_receipt",targets:[{format:je,blend:q}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=r.createRenderPipeline({label:"blackbox-field-blur",layout:P,vertex:{module:b,entryPoint:"vs_fullscreen"},fragment:{module:b,entryPoint:"fs_blur",targets:[{format:je}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=r.createRenderPipeline({label:"blackbox-field-membrane",layout:f,vertex:{module:v,entryPoint:"vs_fullscreen"},fragment:{module:v,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:q}]},primitive:{topology:"triangle-list"}})}ensureResources(r){var w,le,me,oe,i,k;if(!this.traceBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const l=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),v=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),b=!this.resources||this.resources.fieldSize[0]!==l||this.resources.fieldSize[1]!==v;let c=(w=this.resources)==null?void 0:w.eventBuffer,f=(le=this.resources)==null?void 0:le.receiptBuffer,P=(me=this.resources)==null?void 0:me.blurHBuffer,q=(oe=this.resources)==null?void 0:oe.blurVBuffer;if(c||(c=r.createBuffer({label:"blackbox-event-cells",size:wt*Vt*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),f||(f=r.createBuffer({label:"blackbox-receipt-beads",size:Ft*$t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),P||(P=r.createBuffer({label:"blackbox-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(P,0,new Float32Array([1,0,0,0]))),q||(q=r.createBuffer({label:"blackbox-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(q,0,new Float32Array([0,1,0,0]))),!b&&this.resources)return;(i=this.resources)==null||i.fieldA.destroy(),(k=this.resources)==null||k.fieldB.destroy();const G=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,g=r.createTexture({label:"blackbox-field-a-rgba16float",size:[l,v],format:je,usage:G}),I=r.createTexture({label:"blackbox-field-b-rgba16float",size:[l,v],format:je,usage:G}),M=g.createView(),E=I.createView(),re=r.createBindGroup({label:"blackbox-trace-bind",layout:this.traceBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:f}}]}),ae=r.createBindGroup({label:"blackbox-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:M}]}),Q=r.createBindGroup({label:"blackbox-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:M},{binding:2,resource:{buffer:P}}]}),Be=r.createBindGroup({label:"blackbox-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:E},{binding:2,resource:{buffer:q}}]});this.resources={eventBuffer:c,receiptBuffer:f,blurHBuffer:P,blurVBuffer:q,traceBindGroup:re,membraneBindGroup:ae,blurHBindGroup:Q,blurVBindGroup:Be,fieldA:g,fieldB:I,fieldAView:M,fieldBView:E,fieldSize:[l,v]}}uploadBuffers(r){if(!this.resources||!this.scene)return;const l=Math.min(wt,this.scene.visibleEventCount||this.scene.traceEvents.length),v=Math.max(1,this.scene.traceEvents.length-1),b=new Float32Array(wt*Vt);this.eventCount=Math.min(wt,this.scene.traceEvents.length);for(let f=0;f<this.eventCount;f++){const P=this.scene.traceEvents[f],q=f<l?1:0,G=f===this.scene.selectedIndex?1:0,g=nr(P.lane),I=aa(P.type),M=P.memoryIds.length,E=P.type==="memory.suppress"||P.type==="sanhedrin.veto"||P.type==="contradiction.detected"?1:P.type==="memory.write"?.75:.35;b.set([f/v,g,P.confidence,I,f*34+18,q,G,0,M,E,v?f/v:0,0],f*Vt)}r.queue.writeBuffer(this.resources.eventBuffer,0,b);const c=new Float32Array(Ft*$t);this.receiptCount=Math.min(Ft,this.scene.receipts.length);for(let f=0;f<this.receiptCount;f++){const P=Math.min(Math.max(0,l-1),this.eventCount-1);c.set([0,0,.65,P,this.scene.receipts[f].nodeIndices.length,f,0,0],f*$t)}r.queue.writeBuffer(this.resources.receiptBuffer,0,c),this.engine.params[4]=this.eventCount}buildHitRects(){var v;const r=((v=this.scene)==null?void 0:v.traceEvents)??[],l=Math.max(1,r.length-1);this.hitRects=r.map((b,c)=>({event:b,x:-.84+1.68*(c/l),y:.66-1.28*(nr(b.lane)/6),w:.055,h:.065}))}compute(r){const l=this.engine.gpuDevice;if(!l||!this.resources||!this.impulsePipeline||!this.lanePipeline||!this.receiptPipeline||!this.blurPipeline)return;this.ensureResources(l);const v=this.resources,b=r.beginRenderPass({label:"blackbox-field-splat-pass",colorAttachments:[{view:v.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});b.setBindGroup(0,v.traceBindGroup),b.setPipeline(this.lanePipeline),b.draw(6,7),this.eventCount>0&&(b.setPipeline(this.impulsePipeline),b.draw(6,this.eventCount)),this.receiptCount>0&&this.eventCount>0&&(b.setPipeline(this.receiptPipeline),b.draw(6,this.receiptCount)),b.end();const c=r.beginRenderPass({label:"blackbox-field-blur-h-pass",colorAttachments:[{view:v.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});c.setPipeline(this.blurPipeline),c.setBindGroup(0,v.blurHBindGroup),c.draw(6,1),c.end();const f=r.beginRenderPass({label:"blackbox-field-blur-v-pass",colorAttachments:[{view:v.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});f.setPipeline(this.blurPipeline),f.setBindGroup(0,v.blurVBindGroup),f.draw(6,1),f.end()}render(r){!this.resources||!this.membranePipeline||!this.impulsePipeline||!this.lanePipeline||!this.receiptPipeline||(r.setPipeline(this.membranePipeline),r.setBindGroup(0,this.resources.membraneBindGroup),r.draw(6,1),r.setBindGroup(0,this.resources.traceBindGroup),r.setPipeline(this.lanePipeline),r.draw(6,7),this.eventCount>0&&(r.setPipeline(this.impulsePipeline),r.draw(6,this.eventCount)),this.receiptCount>0&&this.eventCount>0&&(r.setPipeline(this.receiptPipeline),r.draw(6,this.receiptCount)))}pickAt(r,l){for(const v of this.hitRects)if(Math.abs(r-v.x)<=v.w&&Math.abs(l-v.y)<=v.h)return{id:v.event.id,kind:"trace-event",index:v.event.index,payload:v.event};return null}dispose(){var r,l,v,b,c,f;(r=this.resources)==null||r.eventBuffer.destroy(),(l=this.resources)==null||l.receiptBuffer.destroy(),(v=this.resources)==null||v.blurHBuffer.destroy(),(b=this.resources)==null||b.blurVBuffer.destroy(),(c=this.resources)==null||c.fieldA.destroy(),(f=this.resources)==null||f.fieldB.destroy(),this.resources=null}}function na(t,r){return zt(Ur.healthy),zt(Or.veto),zt(Vr.forward),[new sa(t,r)]}function kt(t){return Math.max(0,Math.min(1,Number.isFinite(t)?t:0))}function Qe(t,r=0){return typeof t=="number"&&Number.isFinite(t)?t:r}function ia(t,r,l){return{kind:"trace",id:`${t??"trace:none"}:event:${r}:${l}`}}function la(t,r,l){return{kind:"event",id:`${t??"trace:none"}:event:${r}:${l}`}}function oa(t){return{kind:"memory",id:t}}function ca(t){switch(t){case"mcp.call":return"tool";case"memory.retrieve":return"retrieve";case"memory.suppress":return"suppress";case"memory.write":return"write";case"sanhedrin.veto":return"veto";case"contradiction.detected":return"contradiction";case"dream.patch":return"dream"}}function va(t){switch(t.type){case"memory.retrieve":return t.ids;case"memory.suppress":return[t.id];case"memory.write":return[t.id];case"contradiction.detected":return t.ids;case"sanhedrin.veto":return t.evidenceIds;case"dream.patch":return t.proposalIds;case"mcp.call":return[]}}function ua(t){switch(t.type){case"mcp.call":return t.tool;case"memory.retrieve":return`${t.ids.length} memories retrieved`;case"memory.suppress":return`suppressed ${t.id.slice(0,8)}`;case"memory.write":return`wrote ${t.id.slice(0,8)}`;case"contradiction.detected":return`contradiction ${t.ids.join(" ↔ ")}`;case"sanhedrin.veto":return"Sanhedrin veto";case"dream.patch":return`${t.proposalIds.length} dream proposals`}}function da(t){switch(t.type){case"mcp.call":return`MCP tool ${t.tool} called; args hash ${t.argsHash.slice(0,12)}`;case"memory.retrieve":return`Retrieved ${t.ids.length} memories with activation map`;case"memory.suppress":return`Suppressed ${t.id}: ${t.reason}`;case"memory.write":return`Memory write ${t.id} from ${t.source}`;case"contradiction.detected":return t.detail;case"sanhedrin.veto":return`${Math.round(kt(t.confidence)*100)}% veto confidence: ${t.claim}`;case"dream.patch":return`Dream patch proposals: ${t.proposalIds.join(", ")}`}}function pa(t){return t.type!=="memory.retrieve"?[]:Object.entries(t.activation??{}).map(([r,l])=>({id:r,activation:kt(l)})).sort((r,l)=>l.activation-r.activation)}function fa(t){if(t.type==="sanhedrin.veto")return kt(t.confidence);if(t.type==="memory.retrieve"){const r=Object.values(t.activation??{});return r.length?kt(r.reduce((l,v)=>l+v,0)/r.length):.45}return t.type==="memory.suppress"?.78:t.type==="memory.write"?.72:t.type==="contradiction.detected"?.82:t.type==="dream.patch"?.52:.62}function ma(t,r=[],l){var le,me,oe;const v=(t==null?void 0:t.runId)??null,b=(t==null?void 0:t.events)??[],c=(t==null?void 0:t.summary)??null,f=Qe(c==null?void 0:c.startedAt,((le=b[0])==null?void 0:le.at)??0),P=Qe(c==null?void 0:c.lastAt,((me=b[b.length-1])==null?void 0:me.at)??f),q=b.length?b.length-1:0,G=Math.max(0,Math.min(q,l??q)),g=b.length?G+1:0,I=b.map((i,k)=>({index:k,id:`${v??i.runId}:event:${k}:${i.type}`,type:i.type,lane:ca(i.type),runId:i.runId,at:i.at,label:ua(i),summary:da(i),memoryIds:va(i),activationPairs:pa(i),confidence:fa(i),provenance:ia(v,k,i.type),raw:i})),M=new Map;for(const i of I)for(const k of i.memoryIds){if(!k||M.has(k))continue;const V=((oe=i.activationPairs.find(ye=>ye.id===k))==null?void 0:oe.activation)??0;M.set(k,{source:oa(k),index:M.size,label:k.slice(0,12),retention:Math.max(.25,V),activation:V,trust:i.type==="memory.suppress"?.2:Math.max(.35,i.confidence),suppression:i.type==="memory.suppress"?1:0,tags:[i.lane,i.type],type:"trace-memory"})}const E=[...M.values()],re=I.slice(0,g).map(i=>{var k;return{source:la(v,i.index,i.type),type:i.type,targetIndex:i.memoryIds.length?((k=M.get(i.memoryIds[0]))==null?void 0:k.index)??-1:-1,frame:i.index*34+18,energy:Math.max(.18,i.confidence)}}),ae=I.slice(0,g).flatMap(i=>{const k=i.memoryIds.filter(V=>M.has(V));return k.length<2?[]:k.slice(1).map((V,ye)=>{var qe;return{source:{kind:"pair",id:`${i.id}:pair:${k[0]}:${V}`},sourceIndex:M.get(k[0]).index,targetIndex:M.get(V).index,weight:((qe=i.activationPairs[ye+1])==null?void 0:qe.activation)??i.confidence,kind:i.type}})}),Q=r??[],Be=Q.map(i=>{const k=[...new Set([...i.retrieved??[],...i.activation_path??[],...(i.mutations??[]).map(V=>V.id)])];return{source:{kind:"receipt",id:i.receipt_id},label:`receipt ${i.receipt_id.slice(0,10)}`,nodeIndices:k.map(V=>{var ye;return(ye=M.get(V))==null?void 0:ye.index}).filter(V=>typeof V=="number")}}),w={organ:"blackbox",nodes:E,edges:ae,events:re,receipts:Be,scalars:{eventCount:b.length,visibleEventCount:g,retrievedCount:Qe(c==null?void 0:c.retrievedCount,I.filter(i=>i.type==="memory.retrieve").length),suppressedCount:Qe(c==null?void 0:c.suppressedCount,I.filter(i=>i.type==="memory.suppress").length),writeCount:Qe(c==null?void 0:c.writeCount,I.filter(i=>i.type==="memory.write").length),vetoCount:Qe(c==null?void 0:c.vetoCount,I.filter(i=>i.type==="sanhedrin.veto").length),durationMs:Math.max(0,P-f),receiptCount:Be.length},alive:b.length>0,runId:v,traceEvents:I,visibleEventCount:g,selectedIndex:G,startedAt:f,lastAt:P,durationMs:Math.max(0,P-f),receiptRows:Q};return w.alive||(w.receipts=Q.map(i=>({source:{kind:"receipt",id:i.receipt_id},label:`receipt ${i.receipt_id.slice(0,10)}`,nodeIndices:[]})),w.scalars.eventCount=0,w.scalars.visibleEventCount=0),w}var ya=_('<button title="Proof Mode: a clean launch-footage view"><!> Proof Mode</button> <button class="export-btn svelte-1ayqwv0"><!> Export .vestige-trace.json</button>',1),_a=_('<span class="ev-chip svelte-1ayqwv0"> </span>'),ba=_('<span class="text-dim svelte-1ayqwv0">awaiting…</span>'),ha=_('<div class="selected-impulse glass svelte-1ayqwv0"><span class="selected-kicker svelte-1ayqwv0">GPU pick</span> <strong class="svelte-1ayqwv0"> </strong> <span class="svelte-1ayqwv0"> </span> <code class="svelte-1ayqwv0"> </code></div>'),ga=_(`<p class="empty svelte-1ayqwv0">No agent runs recorded yet. Make an MCP tool call — every call is
						recorded here.</p>`),wa=_('<span class="s-recall svelte-1ayqwv0"> </span>'),xa=_('<span class="s-suppress svelte-1ayqwv0"> </span>'),qa=_('<span class="s-write svelte-1ayqwv0"> </span>'),ka=_('<span class="s-veto svelte-1ayqwv0"> </span>'),Pa=_('<li class="svelte-1ayqwv0"><button><div class="run-top svelte-1ayqwv0"><code class="run-id svelte-1ayqwv0"> </code> <span class="run-tool svelte-1ayqwv0"> </span></div> <div class="run-stats svelte-1ayqwv0"><span title="events" class="svelte-1ayqwv0"> </span> <!> <!> <!> <!></div></button></li>'),Sa=_('<ul class="svelte-1ayqwv0"></ul>'),Ba=_('<div class="glass center-msg svelte-1ayqwv0">Loading trace…</div>'),Ca=_('<div class="glass center-msg err svelte-1ayqwv0"> </div>'),Ma=_('<div class="glass center-msg svelte-1ayqwv0">Select a run to replay.</div>'),Ra=_('<span class="scrub-time svelte-1ayqwv0"> </span>'),Ea=_("<button></button>"),Ga=_('<small class="svelte-1ayqwv0"> </small>'),Ia=_('<span class="id-chip svelte-1ayqwv0"><code class="svelte-1ayqwv0"> </code> <!></span>'),Aa=_('<div class="ids-grid svelte-1ayqwv0"></div>'),La=_('<span class="loser svelte-1ayqwv0"> </span>'),za=_('<div class="contra svelte-1ayqwv0"><span class="winner svelte-1ayqwv0"> </span> <span class="vs svelte-1ayqwv0">vs</span> <!></div>'),Ta=_('<code class="svelte-1ayqwv0"> </code>'),Ua=_('<div class="veto-evidence svelte-1ayqwv0"></div>'),Oa=_('<div class="event-detail glass svelte-1ayqwv0"><div class="ed-head svelte-1ayqwv0"><span class="ed-glyph svelte-1ayqwv0"> </span> <span class="ed-label svelte-1ayqwv0"> </span> <code class="ed-time svelte-1ayqwv0"> </code></div> <p class="ed-summary svelte-1ayqwv0"> </p> <!></div>'),Va=_('<p class="empty svelte-1ayqwv0">No memories touched yet.</p>'),$a=_('<code class="pulse-node svelte-1ayqwv0"> </code>'),Fa=_('<div class="pulse-grid svelte-1ayqwv0"></div>'),Na=_('<div class="receipts-panel glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Receipts <span class="text-dim svelte-1ayqwv0">— proof behind retrievals</span></h3> <div class="receipts-grid svelte-1ayqwv0"></div></div>'),Da=_('<li><button class="log-btn svelte-1ayqwv0"><span class="log-glyph svelte-1ayqwv0"> </span> <span class="log-label svelte-1ayqwv0"> </span> <span class="log-summary svelte-1ayqwv0"> </span> <span class="log-t svelte-1ayqwv0"> </span></button></li>'),Ha=_('<div class="scrubber glass svelte-1ayqwv0"><div class="scrub-head svelte-1ayqwv0"><span class="scrub-title svelte-1ayqwv0">Step <strong class="svelte-1ayqwv0"> </strong> </span> <!></div> <input type="range" min="0" class="scrub-range svelte-1ayqwv0"/> <div class="ticks svelte-1ayqwv0"></div></div> <!> <div class="pulse glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Memory pulse <span class="text-dim svelte-1ayqwv0">— touched this run</span></h3> <!></div> <div class="producers glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Event producers <span class="text-dim svelte-1ayqwv0">— this run</span></h3> <ul class="producer-list svelte-1ayqwv0"><li class="producer ok svelte-1ayqwv0"><span class="p-dot svelte-1ayqwv0"></span> mcp.call · memory.write · memory.retrieve · memory.suppress <span class="p-state svelte-1ayqwv0">live</span></li> <li><span class="p-dot svelte-1ayqwv0"></span> contradiction.detected <span class="p-state svelte-1ayqwv0"> </span></li> <li><span class="p-dot svelte-1ayqwv0"></span> dream.patch <span class="p-state svelte-1ayqwv0"> </span></li> <li><span class="p-dot svelte-1ayqwv0"></span> sanhedrin.veto <span class="p-state svelte-1ayqwv0"> </span></li></ul></div> <!> <div class="log glass svelte-1ayqwv0"><h3 class="panel-title svelte-1ayqwv0">Event log</h3> <ol class="log-list svelte-1ayqwv0"></ol></div>',1),Wa=_('<div class="layout svelte-1ayqwv0"><aside class="runs glass svelte-1ayqwv0"><h2 class="panel-title svelte-1ayqwv0">Runs</h2> <!></aside> <section class="replay svelte-1ayqwv0"><!></section></div>'),ja=_('<div class="proof-event svelte-1ayqwv0"><span class="proof-glyph svelte-1ayqwv0"> </span> <div class="svelte-1ayqwv0"><div class="proof-ev-label svelte-1ayqwv0"> </div> <div class="proof-ev-sum svelte-1ayqwv0"> </div></div></div>'),Qa=_('<div class="proof-stage glass svelte-1ayqwv0"><div class="proof-headline svelte-1ayqwv0"><span></span> <code class="proof-run svelte-1ayqwv0"> </code></div> <!> <div class="proof-counter svelte-1ayqwv0"><!> <span class="proof-counter-label svelte-1ayqwv0">trace events</span></div> <p class="proof-tagline svelte-1ayqwv0">Watch the agent think. Watch memory change. Watch the receipt prove why.</p></div>'),Xa=_('<!> <div class="relative z-10 mx-auto max-w-6xl px-5 py-6 svelte-1ayqwv0"><!> <div class="spine glass svelte-1ayqwv0"><div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">WebSocket</span> <span><span></span> </span></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Live runId</span> <code class="spine-run svelte-1ayqwv0"> </code></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Last event</span> <span class="spine-value svelte-1ayqwv0"><!></span></div> <div class="spine-item svelte-1ayqwv0"><span class="spine-label svelte-1ayqwv0">Events seen</span> <span class="spine-value svelte-1ayqwv0"><!></span></div></div> <!> <!></div>',1);function fs(t,r){lr(r,!0);const l=()=>gt(Ir,"$lastTraceEvent",f),v=()=>gt(Lr,"$isConnected",f),b=()=>gt(zr,"$liveRunId",f),c=()=>gt(Ar,"$traceEvents",f),[f,P]=Er();let q=Se(er([])),G=Se(null),g=Se(null),I=Se(!1),M=Se(null),E=Se(0),re=Se(!1),ae=Se(er([])),Q=Se(null);const Be=ge(()=>e(g)?e(g).events.slice(0,e(E)+1):[]),w=ge(()=>e(g)&&e(g).events.length?e(g).events[e(E)]:null),le=ge(()=>{var o,d;return((d=(o=e(g))==null?void 0:o.events[0])==null?void 0:d.at)??0}),me=ge(()=>Array.from(new Set(e(Be).flatMap(Jr)))),oe=ge(()=>{var o;return((o=e(g))==null?void 0:o.events.some(d=>d.type==="sanhedrin.veto"))??!1}),i=ge(()=>{var o;return((o=e(g))==null?void 0:o.events.some(d=>d.type==="dream.patch"))??!1}),k=ge(()=>{var o;return((o=e(g))==null?void 0:o.events.some(d=>d.type==="contradiction.detected"))??!1}),V=ge(()=>ma(e(g),e(ae),e(E)));function ye(o){var p;if(o.kind!=="trace-event")return;const d=o.payload;U(Q,d,!0),U(E,Math.max(0,Math.min((p=e(g))!=null&&p.events.length?e(g).events.length-1:0,d.index)),!0)}async function qe(){try{const o=await lt.traces.list(100);U(q,o.runs,!0),!e(G)&&e(q).length&&ct(e(q)[0].runId)}catch(o){U(M,String(o),!0)}}async function ct(o){U(G,o,!0),U(I,!0),U(M,null);try{U(g,await lt.traces.get(o),!0),U(E,Math.max(0,(e(g).events.length||1)-1),!0),U(Q,null),U(ae,(await lt.receipts.listForRun(o,8)).receipts,!0)}catch(d){U(M,String(d),!0),U(g,null)}finally{U(I,!1)}}function _e(){e(G)&&(window.location.href=lt.traces.exportUrl(e(G)))}tr(()=>{var p;const o=l();if(!o)return;const d=(p=o.data)==null?void 0:p.run_id;d&&d===e(G)&&lt.traces.get(e(G)).then(A=>{U(g,A,!0),U(E,Math.max(0,A.events.length-1),!0)})}),Sr(qe);let ce=null;function Ye(){const o=Math.max(1,...e(q).map(p=>p.eventCount??0)),d=e(q).slice(0,200).map(p=>{const A=vt((p.eventCount??0)/o),S=(p.vetoCount??0)>0?Tt.scarlet:(p.suppressedCount??0)>0?Tt.caution:Tt.forward;return{id:p.runId,score:.35+.65*A,hue:S,energy:.4+.6*A,selected:p.runId===e(G),scar:(p.vetoCount??0)>0,metric2:vt((p.retrievedCount??0)/Math.max(1,p.eventCount??1)),kind:"trace-run",payload:p}});return Fr(d,{maxRadius:.95,minCellR:.016,maxCellR:.06})}function vt(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:0))}function ut(o,d){const p=new $r(o);return ce=p,p.setCells(Ye()),[{compute:S=>p.compute(S),render:S=>p.render(S),pickAt:(S,N)=>p.pickAt(S,N),dispose:()=>{p.dispose(),ce===p&&(ce=null)}},...na(o,d)]}tr(()=>{e(q).length,ce==null||ce.setCells(Ye())});var dt=Xa(),pt=xt(dt);{let o=ge(()=>`agent-flight-recorder:${e(G)??"empty"}:${e(V).visibleEventCount}`);Tr(pt,{organ:"blackbox",get seed(){return e(o)},get scene(){return e(V)},passes:ut,get loading(){return e(I)},get error(){return e(M)},emptyLabel:"NO AGENT TRACE SELECTED - RECORDER ARMED",onpick:ye})}var ft=n(pt,2),ee=s(ft);Cr(ee,{icon:"blackbox",title:"Agent Black Box",subtitle:"Watch the agent think. Watch memory change. Watch the receipt prove why.",accent:"synapse",children:(o,d)=>{var p=ya(),A=xt(p);let S;var N=s(A);Nt(N,{name:"sparkle",size:14}),Ie(),a(A);var se=n(A,2),Me=s(se);Nt(Me,{name:"feed",size:14}),Ie(),a(se),B(()=>{S=pe(A,1,"mode-toggle svelte-1ayqwv0",null,S,{on:e(re)}),se.disabled=!e(G)}),Xe("click",A,()=>U(re,!e(re))),Xe("click",se,_e),y(o,p)},$$slots:{default:!0}});var W=n(ee,2),ve=s(W),X=n(s(ve),2);let be;var Y=s(X);let Ce;var Ke=n(Y);a(X),a(ve);var Je=n(ve,2),Dt=n(s(Je),2),vr=s(Dt,!0);a(Dt),a(Je);var Pt=n(Je,2),Ht=n(s(Pt),2),ur=s(Ht);{var dr=o=>{var d=_a();let p;var A=s(d,!0);a(d),B((S,N)=>{p=Ge(d,"",p,S),u(A,N)},[()=>{var S,N;return{"--c":ot((N=(S=l().data)==null?void 0:S.event)==null?void 0:N.type)}},()=>{var S,N;return We((N=(S=l().data)==null?void 0:S.event)==null?void 0:N.type)}]),y(o,d)},pr=o=>{var d=ba();y(o,d)};O(ur,o=>{l()?o(dr):o(pr,!1)})}a(Ht),a(Pt);var Wt=n(Pt,2),jt=n(s(Wt),2),fr=s(jt);ar(fr,{get value(){return c().length}}),a(jt),a(Wt),a(W),we(W,o=>{var d;return(d=xe)==null?void 0:d(o)});var Qt=n(W,2);{var mr=o=>{var d=ha(),p=n(s(d),2),A=s(p,!0);a(p);var S=n(p,2),N=s(S,!0);a(S);var se=n(S,2),Me=s(se,!0);a(se),a(d),we(d,Ze=>{var Ae;return(Ae=xe)==null?void 0:Ae(Ze)}),B(()=>{u(A,e(Q).label),u(N,e(Q).summary),u(Me,e(Q).provenance.id)}),y(o,d)};O(Qt,o=>{e(Q)&&o(mr)})}var yr=n(Qt,2);{var _r=o=>{var d=Wa(),p=s(d),A=n(s(p),2);{var S=x=>{var R=ga();y(x,R)},N=x=>{var R=Sa();fe(R,21,()=>e(q),te=>te.runId,(te,L)=>{var ke=Pa(),ue=s(ke);let Re;var ze=s(ue),Ee=s(ze),et=s(Ee,!0);a(Ee);var z=n(Ee,2),$e=s(z,!0);a(z),a(ze);var Te=n(ze,2),Ue=s(Te),Fe=s(Ue);a(Ue);var mt=n(Ue,2);{var Bt=$=>{var T=wa(),he=s(T);a(T),B(()=>u(he,`↑${e(L).retrievedCount??""}`)),y($,T)};O(mt,$=>{e(L).retrievedCount&&$(Bt)})}var yt=n(mt,2);{var Ne=$=>{var T=xa(),he=s(T);a(T),B(()=>u(he,`⊘${e(L).suppressedCount??""}`)),y($,T)};O(yt,$=>{e(L).suppressedCount&&$(Ne)})}var tt=n(yt,2);{var De=$=>{var T=qa(),he=s(T);a(T),B(()=>u(he,`✎${e(L).writeCount??""}`)),y($,T)};O(tt,$=>{e(L).writeCount&&$(De)})}var _t=n(tt,2);{var bt=$=>{var T=ka(),he=s(T);a(T),B(()=>u(he,`⛔${e(L).vetoCount??""}`)),y($,T)};O(_t,$=>{e(L).vetoCount&&$(bt)})}a(Te),a(ue),a(ke),B($=>{Re=pe(ue,1,"run-row svelte-1ayqwv0",null,Re,{active:e(L).runId===e(G)}),u(et,$),u($e,e(L).firstTool??"—"),u(Fe,`${e(L).eventCount??""} ev`)},[()=>e(L).runId.replace("run_","").slice(0,10)]),Xe("click",ue,()=>ct(e(L).runId)),y(te,ke)}),a(R),y(x,R)};O(A,x=>{e(q).length===0?x(S):x(N,!1)})}a(p),we(p,x=>{var R;return(R=xe)==null?void 0:R(x)});var se=n(p,2),Me=s(se);{var Ze=x=>{var R=Ba();y(x,R)},Ae=x=>{var R=Ca(),te=s(R,!0);a(R),B(()=>u(te,e(M))),y(x,R)},St=x=>{var R=Ma();y(x,R)},Le=x=>{var R=Ha(),te=xt(R),L=s(te),ke=s(L),ue=n(s(ke)),Re=s(ue,!0);a(ue);var ze=n(ue);a(ke);var Ee=n(ke,2);{var et=h=>{var m=Ra(),F=s(m);a(m),B(C=>u(F,`+${C??""}ms`),[()=>sr(e(w).at,e(le))]),y(h,m)};O(Ee,h=>{e(w)&&h(et)})}a(L);var z=n(L,2);Br(z);var $e=n(z,2);fe($e,21,()=>e(g).events,rr,(h,m,F)=>{var C=Ea();let D,Pe;B((de,Oe,He)=>{D=pe(C,1,"tick svelte-1ayqwv0",null,D,{past:F<=e(E)}),qt(C,"title",de),qt(C,"aria-label",Oe),Pe=Ge(C,"",Pe,He)},[()=>We(e(m).type),()=>`Step ${F+1}: ${We(e(m).type)}`,()=>({"--c":ot(e(m).type)})]),Xe("click",C,()=>U(E,F,!0)),y(h,C)}),a($e),a(te),we(te,h=>{var m;return(m=xe)==null?void 0:m(h)});var Te=n(te,2);{var Ue=h=>{var m=Oa();let F;var C=s(m),D=s(C),Pe=s(D,!0);a(D);var de=n(D,2),Oe=s(de,!0);a(de);var He=n(de,2),rt=s(He,!0);a(He),a(C);var at=n(C,2),st=s(at,!0);a(at);var Rt=n(at,2);{var ht=K=>{var H=Aa();fe(H,20,()=>e(w).ids,J=>J,(J,ne)=>{var ie=Ia();let Z;var Ve=s(ie),nt=s(Ve,!0);a(Ve);var It=n(Ve,2);{var At=it=>{var Lt=Ga(),xr=s(Lt);a(Lt),B(qr=>u(xr,`${qr??""}%`),[()=>(e(w).activation[ne]*100).toFixed(0)]),y(it,Lt)};O(It,it=>{e(w).activation[ne]!=null&&it(At)})}a(ie),B(it=>{Z=Ge(ie,"",Z,{"--a":e(w).activation[ne]??0}),u(nt,it)},[()=>ne.slice(0,8)]),y(J,ie)}),a(H),y(K,H)},Et=K=>{var H=za(),J=s(H),ne=s(J);a(J);var ie=n(J,4);fe(ie,16,()=>e(w).ids.filter(Z=>Z!==e(w).winnerId),Z=>Z,(Z,Ve)=>{var nt=La(),It=s(nt,!0);a(nt),B(At=>u(It,At),[()=>Ve.slice(0,8)]),y(Z,nt)}),a(H),B(Z=>u(ne,`kept ${Z??""}`),[()=>{var Z;return(Z=e(w).winnerId)==null?void 0:Z.slice(0,8)}]),y(K,H)},Gt=K=>{var H=Ua();fe(H,20,()=>e(w).evidenceIds,J=>J,(J,ne)=>{var ie=Ta(),Z=s(ie,!0);a(ie),B(Ve=>u(Z,Ve),[()=>ne.slice(0,8)]),y(J,ie)}),a(H),y(K,H)};O(Rt,K=>{e(w).type==="memory.retrieve"?K(ht):e(w).type==="contradiction.detected"?K(Et,1):e(w).type==="sanhedrin.veto"&&K(Gt,2)})}a(m),we(m,K=>{var H;return(H=xe)==null?void 0:H(K)}),B((K,H,J,ne,ie)=>{F=Ge(m,"",F,K),u(Pe,H),u(Oe,J),u(rt,ne),u(st,ie)},[()=>({"--c":ot(e(w).type)}),()=>Ut(e(w).type),()=>We(e(w).type),()=>Zr(e(w).at),()=>Ot(e(w))]),y(h,m)};O(Te,h=>{e(w)&&h(Ue)})}var Fe=n(Te,2),mt=n(s(Fe),2);{var Bt=h=>{var m=Va();y(h,m)},yt=h=>{var m=Fa();fe(m,20,()=>e(me),F=>F,(F,C)=>{var D=$a(),Pe=s(D,!0);a(D),B(de=>u(Pe,de),[()=>C.slice(0,8)]),y(F,D)}),a(m),y(h,m)};O(mt,h=>{e(me).length===0?h(Bt):h(yt,!1)})}a(Fe),we(Fe,h=>{var m;return(m=xe)==null?void 0:m(h)});var Ne=n(Fe,2),tt=n(s(Ne),2),De=n(s(tt),2);let _t;var bt=n(s(De),2),$=s(bt,!0);a(bt),a(De);var T=n(De,2);let he;var Xt=n(s(T),2),hr=s(Xt,!0);a(Xt),a(T);var Ct=n(T,2);let Yt;var Kt=n(s(Ct),2),gr=s(Kt,!0);a(Kt),a(Ct),a(tt),a(Ne),we(Ne,h=>{var m;return(m=xe)==null?void 0:m(h)});var Jt=n(Ne,2);{var wr=h=>{var m=Na(),F=n(s(m),2);fe(F,21,()=>e(ae).slice(0,2),C=>C.receipt_id,(C,D)=>{Kr(C,{get receipt(){return e(D)}})}),a(F),a(m),we(m,C=>{var D;return(D=xe)==null?void 0:D(C)}),y(h,m)};O(Jt,h=>{e(ae).length&&h(wr)})}var Mt=n(Jt,2),Zt=n(s(Mt),2);fe(Zt,21,()=>e(g).events,rr,(h,m,F)=>{var C=Da();let D,Pe;var de=s(C),Oe=s(de),He=s(Oe,!0);a(Oe);var rt=n(Oe,2),at=s(rt,!0);a(rt);var st=n(rt,2),Rt=s(st,!0);a(st);var ht=n(st,2),Et=s(ht);a(ht),a(de),a(C),B((Gt,K,H,J,ne)=>{D=pe(C,1,"log-row svelte-1ayqwv0",null,D,{active:F===e(E),dim:F>e(E)}),Pe=Ge(C,"",Pe,Gt),u(He,K),u(at,H),u(Rt,J),u(Et,`+${ne??""}ms`)},[()=>({"--c":ot(e(m).type)}),()=>Ut(e(m).type),()=>We(e(m).type),()=>Ot(e(m)),()=>sr(e(m).at,e(le))]),Xe("click",de,()=>U(E,F,!0)),y(h,C)}),a(Zt),a(Mt),we(Mt,h=>{var m;return(m=xe)==null?void 0:m(h)}),B(h=>{u(Re,e(E)+1),u(ze,` / ${e(g).events.length??""}`),qt(z,"max",h),_t=pe(De,1,"producer svelte-1ayqwv0",null,_t,{ok:e(k)}),u($,e(k)?"fired this run":"no contradiction in this run"),he=pe(T,1,"producer caveat svelte-1ayqwv0",null,he,{ok:e(i)}),u(hr,e(i)?"fired this run":"No dream run in this trace"),Yt=pe(Ct,1,"producer caveat svelte-1ayqwv0",null,Yt,{ok:e(oe)}),u(gr,e(oe)?"fired this run":"No veto producer connected (optional Sanhedrin hook, off by default)")},[()=>Math.max(0,e(g).events.length-1)]),Mr(z,()=>e(E),h=>U(E,h)),y(x,R)};O(Me,x=>{e(I)?x(Ze):e(M)?x(Ae,1):e(g)?x(Le,!1):x(St,2)})}a(se),a(d),y(o,d)},br=o=>{var d=Qa(),p=s(d),A=s(p);let S;var N=n(A,2),se=s(N,!0);a(N),a(p);var Me=n(p,2);{var Ze=Le=>{const x=ge(()=>{var z;return(z=l().data)==null?void 0:z.event});var R=ja();let te;var L=s(R),ke=s(L,!0);a(L);var ue=n(L,2),Re=s(ue),ze=s(Re,!0);a(Re);var Ee=n(Re,2),et=s(Ee,!0);a(Ee),a(ue),a(R),B((z,$e,Te,Ue)=>{te=Ge(R,"",te,z),u(ke,$e),u(ze,Te),u(et,Ue)},[()=>{var z;return{"--c":ot((z=e(x))==null?void 0:z.type)}},()=>{var z;return Ut((z=e(x))==null?void 0:z.type)},()=>{var z;return We((z=e(x))==null?void 0:z.type)},()=>Ot(e(x))]),y(Le,R)};O(Me,Le=>{l()&&Le(Ze)})}var Ae=n(Me,2),St=s(Ae);ar(St,{get value(){return c().length}}),Ie(2),a(Ae),Ie(2),a(d),we(d,Le=>{var x;return(x=xe)==null?void 0:x(Le)}),B(()=>{S=pe(A,1,"dot big svelte-1ayqwv0",null,S,{live:v()}),u(se,b()??"awaiting run…")}),y(o,d)};O(yr,o=>{e(re)?o(br,!1):o(_r)})}a(ft),B(()=>{be=pe(X,1,"spine-value svelte-1ayqwv0",null,be,{live:v()}),Ce=pe(Y,1,"dot svelte-1ayqwv0",null,Ce,{live:v()}),u(Ke,` ${v()?"Connected":"Offline"}`),u(vr,b()??"—")}),y(t,dt),or(),P()}ir(["click"]);export{fs as component};

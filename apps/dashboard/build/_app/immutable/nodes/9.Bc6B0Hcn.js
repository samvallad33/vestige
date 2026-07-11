var Ze=Object.defineProperty;var Je=(i,e,s)=>e in i?Ze(i,e,{enumerable:!0,configurable:!0,writable:!0,value:s}):i[e]=s;var D=(i,e,s)=>Je(i,typeof e!="symbol"?e+"":e,s);import"../chunks/Bzak7iHL.js";import{d as We,s as C,b as fe,o as et,a as tt}from"../chunks/DAau0uzT.js";import{p as $e,aS as it,aH as ge,g as r,a as w,b as Ye,u as q,c as d,r as u,X as p,Y as $,s as F,f as S,d as re,e as Ne,af as we}from"../chunks/CGq8RnJq.js";import{i as J}from"../chunks/Ccqjq5DS.js";import{e as Ce,i as je,s as Se,r as rt}from"../chunks/DqfV0sZu.js";import{P as st,A as Te,a as Ie,r as nt}from"../chunks/B9l3DI-J.js";import{b as at}from"../chunks/DGM4cicq.js";import{s as Ee}from"../chunks/uCQU803Y.js";import{s as Pe}from"../chunks/HFGAk8XQ.js";import{p as lt}from"../chunks/DV6OI5iy.js";import{N as ot}from"../chunks/CcUbQ_Wl.js";import{I as ct}from"../chunks/CKbQrCJw.js";import{R as ut}from"../chunks/BpEKQwpr.js";import{r as ke,M as dt,R as Ve,I as mt,m as ft}from"../chunks/BMB5u1EX.js";import{a as pt}from"../chunks/D35IQVqe.js";function Qe(i){return i>=.92?"near-identical":i>=.8?"strong":"weak"}function De(i){const e=Qe(i);return e==="near-identical"?"var(--color-decay)":e==="strong"?"var(--color-warning)":"#fde047"}function vt(i){const e=Qe(i);return e==="near-identical"?"Near-identical":e==="strong"?"Strong match":"Weak match"}function gt(i){return i>.7?"#10b981":i>.4?"#f59e0b":"#ef4444"}function Xe(i){if(!i||i.length===0)return null;let e=i[0],s=Number.isFinite(e.retention)?e.retention:-1/0;for(let t=1;t<i.length;t++){const a=i[t],n=Number.isFinite(a.retention)?a.retention:-1/0;n>s&&(e=a,s=n)}return e}function Ae(i){return i.map(e=>e.id).slice().sort().join("|")}function ht(i,e=80){if(!i)return"";const s=i.trim().replace(/\s+/g," ");return s.length<=e?s:s.slice(0,e)+"…"}function He(i){if(!i||typeof i!="string")return"";const e=new Date(i);return Number.isNaN(e.getTime())?"":e.toLocaleDateString(void 0,{year:"numeric",month:"short",day:"numeric"})}function bt(i,e=4){return Array.isArray(i)?i.slice(0,e):[]}var yt=S('<span class="flex-shrink-0 rounded-full border border-warning/50 bg-warning/10 px-3 py-1 text-xs font-medium text-warning">REVIEW REQUIRED · NOT SAFE TO MERGE</span>'),xt=S("<span> </span>"),_t=S('<span class="rounded bg-recall/15 px-1.5 py-0.5 text-[10px] font-medium text-recall">WINNER</span>'),wt=S('<span class="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-muted"> </span>'),St=S('<div class="text-[11px] text-muted"> </div>'),Pt=S('<div><span class="mt-1.5 h-2 w-2 flex-shrink-0 rounded-full"></span> <div class="flex-1 min-w-0 space-y-1.5"><div class="flex flex-wrap items-center gap-1.5"><span class="text-xs text-dim"> </span> <!> <!></div> <p> </p> <!></div> <div class="flex flex-shrink-0 flex-col items-end gap-1"><div class="h-1.5 w-12 overflow-hidden rounded-full bg-deep"><div class="h-full rounded-full"></div></div> <span class="text-[11px] text-muted"> </span></div></div>'),kt=S('<div class="rounded-xl border border-warning/20 bg-warning/5 p-3 text-xs text-dim"> </div>'),Bt=S('<div class="glass-panel rounded-2xl p-5 space-y-4 transition-all duration-300 hover:border-synapse/20"><div class="flex items-start justify-between gap-4"><div class="flex-1 min-w-0 space-y-1.5"><div class="flex items-center gap-3"><span class="text-sm font-semibold"> </span> <span class="text-xs text-dim"> </span> <span class="text-xs text-muted"> </span></div> <div class="h-2 w-full overflow-hidden rounded-full bg-deep/60" role="progressbar" aria-label="Cosine similarity" aria-valuemin="0" aria-valuemax="100"><div class="h-full rounded-full transition-all duration-500"></div></div></div> <!></div> <div class="space-y-2"><!> <!></div> <div class="flex flex-wrap items-center gap-2 pt-1"><button type="button" disabled="" aria-disabled="true" aria-label="Merge is not available yet" class="cursor-not-allowed rounded-lg bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-muted/60">Merge unavailable</button> <button type="button" class="rounded-lg bg-dream/20 px-3 py-1.5 text-xs font-medium text-dream-glow transition hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream-glow/60"> </button> <button type="button" aria-label="Dismiss cluster for this session" class="ml-auto rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Dismiss cluster</button></div></div>');function Mt(i,e){$e(e,!0);let s=lt(e,"oversized",3,!1),t=re(!1);const a=12,n=q(()=>Xe(e.memories)),o=q(()=>{if(e.memories.length<=a)return e.memories;const c=e.memories.filter(y=>{var T;return y.id!==((T=r(n))==null?void 0:T.id)});return r(n)?[r(n),...c.slice(0,a-1)]:c.slice(0,a)}),m=q(()=>e.memories.length-r(o).length);var v=it(),x=ge(v);{var L=c=>{var y=Bt(),T=d(y),P=d(T),H=d(P),g=d(H),h=d(g);u(g);var M=p(g,2),z=d(M,!0);u(M);var Y=p(M,2),K=d(Y);u(Y),u(H);var O=p(H,2),k=d(O);u(O),u(P);var R=p(P,2);{var te=B=>{var b=yt();w(B,b)},ee=B=>{var b=xt(),j=d(b);u(b),$(()=>{Ee(b,1,`flex-shrink-0 rounded-full border px-3 py-1 text-xs font-medium ${e.suggestedAction==="merge"?"border-recall/40 bg-recall/10 text-recall":"border-dream-glow/40 bg-dream/10 text-dream-glow"}`),C(j,`Classification: ${e.suggestedAction==="merge"?"merge candidate":"review"}`)}),w(B,b)};J(R,B=>{s()?B(te):B(ee,!1)})}u(T);var Z=p(T,2),se=d(Z);Ce(se,17,()=>r(o),B=>B.id,(B,b)=>{var j=Pt(),ae=d(j),le=p(ae,2),oe=d(le),ce=d(oe),l=d(ce,!0);u(ce);var f=p(ce,2);{var _=W=>{var X=_t();w(W,X)};J(f,W=>{r(b).id===r(n).id&&W(_)})}var A=p(f,2);Ce(A,17,()=>bt(r(b).tags,4),je,(W,X)=>{var de=wt(),Le=d(de,!0);u(de),$(()=>C(Le,r(X))),w(W,de)}),u(oe);var G=p(oe,2),N=d(G,!0);u(G);var E=p(G,2);{var I=W=>{var X=St(),de=d(X,!0);u(X),$(Le=>C(de,Le),[()=>He(r(b).createdAt)]),w(W,X)},V=q(()=>He(r(b).createdAt));J(E,W=>{r(V)&&W(I)})}u(le);var Q=p(le,2),U=d(Q),me=d(U);u(U);var _e=p(U,2),ue=d(_e);u(_e),u(Q),u(j),$((W,X,de)=>{Ee(j,1,`group flex items-start gap-3 rounded-xl border border-synapse/5 bg-white/[0.02] p-3 transition-all duration-200 hover:border-synapse/20 hover:bg-white/[0.04] ${r(b).id===r(n).id?"ring-1 ring-recall/30":""}`),Pe(ae,`background: ${(ot[r(b).nodeType]||"#8B95A5")??""}`),Se(ae,"title",r(b).nodeType),C(l,r(b).nodeType),Ee(G,1,`text-sm text-text leading-relaxed ${r(t)?"whitespace-pre-wrap":""}`),C(N,W),Pe(me,`width: ${r(b).retention*100}%; background: ${X??""}`),C(ue,`${de??""}%`)},[()=>r(t)?r(b).content:ht(r(b).content),()=>gt(r(b).retention),()=>(r(b).retention*100).toFixed(0)]),w(B,j)});var pe=p(se,2);{var ie=B=>{var b=kt(),j=d(b);u(b),$(()=>C(j,`+${r(m)??""} linked candidates — oversized similarity component. Members
					chain through pairwise similarity; distant members may be unrelated. Raise
					the threshold to split it.`)),w(B,b)};J(pe,B=>{r(m)>0&&B(ie)})}u(Z);var be=p(Z,2),ye=d(be),ne=p(ye,2),Re=d(ne,!0);u(ne);var xe=p(ne,2);u(be),u(y),$((B,b,j,ae,le,oe,ce)=>{Pe(g,`color: ${B??""}`),C(h,`${b??""}%`),C(z,j),C(K,`· ${e.memories.length??""} memories`),Se(O,"aria-valuenow",ae),Pe(k,`width: ${le??""}%; background: ${oe??""}; box-shadow: 0 0 12px ${ce??""}66`),Se(ye,"title",s()?"Oversized similarity component — not safe to merge":"Merge backend not shipped yet — no destructive action is taken from this screen"),Se(ne,"aria-expanded",r(t)),C(Re,r(t)?"Collapse":"Review")},[()=>De(e.similarity),()=>(e.similarity*100).toFixed(1),()=>vt(e.similarity),()=>Math.round(e.similarity*100),()=>(e.similarity*100).toFixed(1),()=>De(e.similarity),()=>De(e.similarity)]),fe("click",ne,()=>F(t,!r(t))),fe("click",xe,function(...B){var b;(b=e.onDismiss)==null||b.apply(this,B)}),w(c,y)};J(x,c=>{e.memories.length>0&&r(n)&&c(L)})}w(i,v),Ye()}We(["click"]);const Be="rgba16float",ve=512,Me=512,Oe=16,Ue=16,qe=`
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

struct FusionCell {
	// x/y position in NDC, z retention, w winner flag
	pos_retention: vec4f,
	// x similarity, y threshold, z member slot, w cluster slot
	cluster_meta: vec4f,
	// x mismatch intensity, y merge flag, z radius, w member count
	visual_meta: vec4f,
	// x cell index, y cluster index, z/w spare
	ids: vec4f,
};

struct FusionNeck {
	// x/y winner position, z winner retention, w winner radius
	a: vec4f,
	// x/y candidate position, z candidate retention, w candidate radius
	b: vec4f,
	// x similarity, y threshold, z mismatch intensity, w merge flag
	signals: vec4f,
	// x neck index, y cluster index, z/w spare
	ids: vec4f,
};
`,At=`
${qe}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<FusionCell>;
@group(0) @binding(2) var<storage, read> necks: array<FusionNeck>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
};

fn similarity_neck(similarity: f32) -> f32 {
	return smoothstep(0.78, 0.98, similarity);
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let corner = QUAD[vi];
	let cell_count = u32(params.node_count);
	if (ii < cell_count) {
		let c = cells[ii];
		let merge_gate = c.visual_meta.y;
		let radius = c.visual_meta.z * (1.0 + 0.045 * sin(params.time * 2.0 + c.cluster_meta.w * 6.28318));
		out.clip = vec4f(c.pos_retention.xy + corner * radius, 0.0, 1.0);
		out.uv = corner;
		out.misc = vec4f(c.pos_retention.z, c.cluster_meta.x, c.visual_meta.x, merge_gate);
	} else {
		let n = necks[ii - cell_count];
		let a = n.a.xy;
		let b = n.b.xy;
		let center = (a + b) * 0.5;
		let dir = normalize(b - a + vec2f(0.0001, 0.0001));
		let normal = vec2f(-dir.y, dir.x);
		let fused = similarity_neck(n.signals.x);
		let length_half = distance(a, b) * 0.5;
		let thickness = 0.035 + fused * 0.085 + n.signals.z * 0.025;
		let pos = center + dir * corner.x * length_half + normal * corner.y * thickness;
		out.clip = vec4f(pos, 0.0, 1.0);
		out.uv = vec2f(corner.x, corner.y / max(0.001, thickness));
		out.misc = vec4f(n.signals.x, fused, n.signals.z, n.signals.w);
	}
	return out;
}

@fragment
fn fs_splat(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.uv);
	let is_neck = f32(abs(frag.uv.y) > 1.0);
	if (is_neck < 0.5 && d > 1.0) { discard; }
	let retention = clamp(frag.misc.x, 0.0, 1.0);
	let similarity = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let merge_gate = frag.misc.w;
	let cell_body = exp(-d * d * 3.15) * (0.38 + retention * 0.62) * (0.5 + similarity * 0.58);
	let cell_rim = smoothstep(0.24, 0.02, abs(d - (0.58 + retention * 0.16))) * (0.2 + similarity * 0.55);
	let neck_body = exp(-frag.uv.y * frag.uv.y * 4.0) * smoothstep(1.05, 0.82, abs(frag.uv.x)) * (0.35 + similarity * 0.9);
	let density = max(cell_body + cell_rim, neck_body * (0.4 + similarity));
	// r=density, g=retention/luciferin, b=mismatch amber, a reserved. No storage textures.
	return vec4f(density, density * (0.35 + retention * 0.65), mismatch * (0.18 + merge_gate * 0.12), 1.0);
}

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let c = cells[ii];
	let corner = QUAD[vi];
	let winner = c.pos_retention.w;
	let radius = c.visual_meta.z * (0.46 + winner * 0.18);
	out.clip = vec4f(c.pos_retention.xy + corner * radius, 0.0, 1.0);
	out.uv = corner;
	out.misc = vec4f(c.pos_retention.z, c.cluster_meta.x, c.visual_meta.x, winner);
	return out;
}

@fragment
fn fs_cell(frag: VSOut) -> @location(0) vec4f {
	let d = length(frag.uv);
	if (d > 1.0) { discard; }
	let retention = clamp(frag.misc.x, 0.0, 1.0);
	let similarity = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let winner = frag.misc.w;
	let sediment = vec3f(0.54, 0.29, 0.09);
	let recall = vec3f(0.16, 0.95, 0.66);
	let luciferin = vec3f(0.91, 1.0, 0.72);
	let ivory = vec3f(0.96, 0.945, 0.815);
	let amber = vec3f(1.0, 0.69, 0.08);
	let core = mix(sediment, mix(recall, luciferin, retention), retention);
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.22, d));
	let body = exp(-d*d*3.2) * (0.20 + retention * 0.44 + winner * 0.16);
	let mismatch_ring = smoothstep(0.16, 0.0, abs(d - 0.80)) * mismatch;
	return vec4f(core * body + ivory * rim * (0.16 + similarity * 0.52) + amber * mismatch_ring * 0.34, 1.0);
}

@vertex
fn vs_neck(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let n = necks[ii];
	let a = n.a.xy;
	let b = n.b.xy;
	let t = f32(vi / 2u) / 31.0;
	let side = f32(vi % 2u) * 2.0 - 1.0;
	let dir = normalize(b - a + vec2f(0.0001, 0.0001));
	let normal = vec2f(-dir.y, dir.x);
	let midpoint = (a + b) * 0.5;
	let fused = similarity_neck(n.signals.x);
	let threshold_pull = clamp(n.signals.x - n.signals.y + 0.22, 0.0, 1.0);
	let bow = normal * sin(t * 3.14159) * (0.030 + n.signals.z * 0.050) * (1.0 - fused * 0.35);
	let pos = mix(a, b, t) + bow;
	let thickness = 0.005 + fused * 0.025 + threshold_pull * 0.010;
	out.clip = vec4f(pos + normal * side * thickness, 0.0, 1.0);
	out.uv = vec2f(t, side);
	out.misc = vec4f(n.signals.x, n.signals.y, n.signals.z, distance(pos, midpoint));
	return out;
}

@fragment
fn fs_neck(frag: VSOut) -> @location(0) vec4f {
	let similarity = clamp(frag.misc.x, 0.0, 1.0);
	let threshold = clamp(frag.misc.y, 0.0, 1.0);
	let mismatch = clamp(frag.misc.z, 0.0, 1.0);
	let pulse = 0.55 + 0.45 * sin(36.0 * frag.uv.x - 8.0 * frag.misc.w);
	let bridge = vec3f(0.10, 0.82, 0.92);
	let luciferin = vec3f(0.91, 1.0, 0.72);
	let amber = vec3f(1.0, 0.69, 0.08);
	let pull = smoothstep(-0.08, 0.20, similarity - threshold);
	let color = mix(bridge, luciferin, pull) + amber * mismatch * pulse * 0.34;
	return vec4f(color * (0.14 + similarity * 0.55 + mismatch * 0.18), 1.0);
}
`,Gt=`
${qe}

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
fn fs_membrane(frag: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, frag.uv);
	let density = clamp(f.r, 0.0, 5.0);
	let retention = clamp(f.g, 0.0, 5.0);
	let mismatch = clamp(f.b, 0.0, 3.0);
	let membrane = smoothstep(0.13, 0.88, density) * (1.0 - smoothstep(1.9, 3.8, density));
	let blackwater = vec3f(0.008, 0.012, 0.018);
	let bridge = vec3f(0.10, 0.82, 0.92);
	let luciferin = vec3f(0.66, 1.0, 0.37);
	let ivory = vec3f(0.96, 0.945, 0.815);
	let amber = vec3f(1.0, 0.69, 0.08);
	var color = blackwater * (0.18 + density * 0.055);
	color = color + bridge * density * 0.055 + luciferin * retention * 0.080;
	color = color + ivory * membrane * 0.22 + amber * mismatch * (0.20 + 0.08 * params.pulse);
	let vignette = smoothstep(0.96, 0.18, distance(frag.uv, vec2f(0.5)));
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness, 1.0);
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
fn fs_blur(frag: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, frag.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, frag.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`;class Rt{constructor(e,s){D(this,"engine");D(this,"scene",null);D(this,"resources",null);D(this,"sampler",null);D(this,"splatBindLayout",null);D(this,"blurBindLayout",null);D(this,"membraneBindLayout",null);D(this,"splatPipeline",null);D(this,"blurPipeline",null);D(this,"membranePipeline",null);D(this,"cellPipeline",null);D(this,"neckPipeline",null);D(this,"cellCount",0);D(this,"neckCount",0);D(this,"cellGeometry",[]);D(this,"neckGeometry",[]);this.engine=e,this.uploadScene(s)}uploadScene(e){this.scene=e,this.buildGeometry();const s=this.engine.gpuDevice;s&&(this.ensurePipelines(s),this.ensureResources(s),this.uploadBuffers(s))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const s=Fe(e,"duplicates-fusion-splat-wgsl",At),t=Fe(e,"duplicates-fusion-blur-wgsl",Ct),a=Fe(e,"duplicates-fusion-membrane-wgsl",Gt);this.splatBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-splat-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const n=e.createPipelineLayout({label:"duplicates-fusion-splat-layout",bindGroupLayouts:[this.splatBindLayout]}),o=e.createPipelineLayout({label:"duplicates-fusion-blur-layout",bindGroupLayouts:[this.blurBindLayout]}),m=e.createPipelineLayout({label:"duplicates-fusion-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"});const v={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.splatPipeline=e.createRenderPipeline({label:"duplicates-field-additive-splat",layout:n,vertex:{module:s,entryPoint:"vs_splat"},fragment:{module:s,entryPoint:"fs_splat",targets:[{format:Be,blend:v}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"duplicates-field-blur-render-pass",layout:o,vertex:{module:t,entryPoint:"vs_fullscreen"},fragment:{module:t,entryPoint:"fs_blur",targets:[{format:Be}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=e.createRenderPipeline({label:"duplicates-synaptic-fusion-membrane",layout:m,vertex:{module:a,entryPoint:"vs_fullscreen"},fragment:{module:a,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:v}]},primitive:{topology:"triangle-list"}}),this.cellPipeline=e.createRenderPipeline({label:"duplicates-memory-nuclei",layout:n,vertex:{module:s,entryPoint:"vs_cell"},fragment:{module:s,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:v}]},primitive:{topology:"triangle-list"}}),this.neckPipeline=e.createRenderPipeline({label:"duplicates-mismatch-filaments",layout:n,vertex:{module:s,entryPoint:"vs_neck"},fragment:{module:s,entryPoint:"fs_neck",targets:[{format:this.engine.sceneFormat,blend:v}]},primitive:{topology:"triangle-strip"}})}ensureResources(e){var M,z,Y,K,O,k;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const s=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),t=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),a=!this.resources||this.resources.fieldSize[0]!==s||this.resources.fieldSize[1]!==t;let n=(M=this.resources)==null?void 0:M.cellBuffer,o=(z=this.resources)==null?void 0:z.neckBuffer,m=(Y=this.resources)==null?void 0:Y.blurHBuffer,v=(K=this.resources)==null?void 0:K.blurVBuffer;if(n||(n=e.createBuffer({label:"duplicates-cells",size:ve*Oe*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),o||(o=e.createBuffer({label:"duplicates-necks",size:Me*Ue*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),m||(m=e.createBuffer({label:"duplicates-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(m,0,new Float32Array([1,0,0,0]))),v||(v=e.createBuffer({label:"duplicates-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(v,0,new Float32Array([0,1,0,0]))),!a&&this.resources)return;(O=this.resources)==null||O.fieldA.destroy(),(k=this.resources)==null||k.fieldB.destroy();const x=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,L=e.createTexture({label:"duplicates-field-a-rgba16float",size:[s,t],format:Be,usage:x}),c=e.createTexture({label:"duplicates-field-b-rgba16float",size:[s,t],format:Be,usage:x}),y=L.createView(),T=c.createView(),P=e.createBindGroup({label:"duplicates-fusion-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:o}}]}),H=e.createBindGroup({label:"duplicates-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:y},{binding:2,resource:{buffer:m}}]}),g=e.createBindGroup({label:"duplicates-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:T},{binding:2,resource:{buffer:v}}]}),h=e.createBindGroup({label:"duplicates-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:y}]});this.resources={cellBuffer:n,neckBuffer:o,blurHBuffer:m,blurVBuffer:v,splatBindGroup:P,blurHBindGroup:H,blurVBindGroup:g,membraneBindGroup:h,fieldA:L,fieldB:c,fieldAView:y,fieldBView:T,fieldSize:[s,t]}}buildGeometry(){var x,L;const e=((x=this.scene)==null?void 0:x.clusters)??[],s=Math.max(1,e.length),t=[],a=[],n=12,o=new Array(e.length).fill(0);let m=ve;for(let c=0;c<e.length&&m>0;c++)o[c]=1,m-=1;for(let c=0;c<e.length&&m>0;c++)o[c]===1&&e[c].memories.length>=2&&(o[c]=2,m-=1);let v=!0;for(;m>0&&v;){v=!1;for(let c=0;c<e.length&&m>0;c++)o[c]>0&&o[c]<Math.min(e[c].memories.length,n)&&(o[c]+=1,m-=1,v=!0)}for(let c=0;c<e.length;c++){const y=e[c];if(o[c]===0)continue;const T=c/s*Math.PI*2-Math.PI/2,P=.18+.58*Math.sqrt((c+.5)/s),H=Math.cos(T)*P*.86,g=Math.sin(T)*P,h=Math.max(.04,.25-Math.max(0,y.similarity-y.threshold)*.55),M=y.memories.find(k=>k.id===y.winnerId)??y.memories[0],z=[M,...y.memories.filter(k=>k.id!==M.id)].slice(0,o[c]),Y=Math.max(1,z.length),K=new Map;for(let k=0;k<z.length&&t.length<ve;k++){const R=z[k],te=T+k/Y*Math.PI*2+(Y%2?0:Math.PI/Y),ee=R.id===y.winnerId,Z=ee?h*.18:h+.025*(k%3),se=Math.min(1,(((L=R.mismatchTokens)==null?void 0:L.length)??0)/8),pe=.085+Math.min(.045,ft(R.retention)*2.1)+(ee?.012:0),ie={cluster:y,memoryId:R.id,x:H+Math.cos(te)*Z,y:g+Math.sin(te)*Z,retention:Math.max(0,Math.min(1,R.retention||0)),winner:ee,mismatch:se,radius:pe,memberSlot:k,memberCount:Y};K.set(R.id,ie),t.push(ie)}const O=K.get(M.id);if(O)for(const k of y.memories){if(a.length>=Me||k.id===M.id)continue;const R=K.get(k.id);R&&a.push({cluster:y,winnerId:M.id,candidateId:k.id,ax:O.x,ay:O.y,bx:R.x,by:R.y,winnerRetention:O.retention,candidateRetention:R.retention,winnerRadius:O.radius,candidateRadius:R.radius,mismatch:Math.max(R.mismatch,Math.min(1,y.mismatchTokens.length/12))})}}this.cellGeometry=t,this.neckGeometry=a}uploadBuffers(e){if(!this.resources)return;const s=new Float32Array(ve*Oe),t=new Float32Array(Me*Ue);this.cellCount=Math.min(ve,this.cellGeometry.length),this.neckCount=Math.min(Me,this.neckGeometry.length);for(let a=0;a<this.cellCount;a++){const n=this.cellGeometry[a];s.set([n.x,n.y,n.retention,n.winner?1:0,n.cluster.similarity,n.cluster.threshold,n.memberSlot,n.cluster.index,n.mismatch,n.cluster.suggestedAction==="merge"?1:0,n.radius,n.memberCount,a,n.cluster.index,0,0],a*Oe)}for(let a=0;a<this.neckCount;a++){const n=this.neckGeometry[a];t.set([n.ax,n.ay,n.winnerRetention,n.winnerRadius,n.bx,n.by,n.candidateRetention,n.candidateRadius,n.cluster.similarity,n.cluster.threshold,n.mismatch,n.cluster.suggestedAction==="merge"?1:0,a,n.cluster.index,0,0],a*Ue)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.neckCount,this.engine.params[4]=this.neckCount,e.queue.writeBuffer(this.resources.cellBuffer,0,s),e.queue.writeBuffer(this.resources.neckBuffer,0,t)}compute(e){const s=this.engine.gpuDevice;if(!s||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(s);const t=this.resources,a=e.beginRenderPass({label:"duplicates-field-splat-pass",colorAttachments:[{view:t.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});a.setPipeline(this.splatPipeline),a.setBindGroup(0,t.splatBindGroup),a.draw(6,this.cellCount+this.neckCount),a.end();const n=e.beginRenderPass({label:"duplicates-field-blur-h-pass",colorAttachments:[{view:t.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(this.blurPipeline),n.setBindGroup(0,t.blurHBindGroup),n.draw(6,1),n.end();const o=e.beginRenderPass({label:"duplicates-field-blur-v-pass",colorAttachments:[{view:t.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});o.setPipeline(this.blurPipeline),o.setBindGroup(0,t.blurVBindGroup),o.draw(6,1),o.end()}render(e){!this.resources||!this.membranePipeline||!this.cellPipeline||!this.neckPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.neckCount>0&&(e.setPipeline(this.neckPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(64,this.neckCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(6,this.cellCount)))}pickAt(e,s){for(let t=0;t<this.neckGeometry.length;t++){const a=this.neckGeometry[t],n=Lt(e,s,a.ax,a.ay,a.bx,a.by),o=(a.ax+a.bx)*.5,m=(a.ay+a.by)*.5,v=.055+Math.max(0,a.cluster.similarity-a.cluster.threshold)*.45;if(n<=v||Math.hypot(e-o,s-m)<=v)return{id:a.cluster.id,kind:"duplicate-neck",index:t,payload:a.cluster}}for(let t=0;t<this.cellGeometry.length;t++){const a=this.cellGeometry[t];if(Math.hypot(e-a.x,s-a.y)<=a.radius*.8)return{id:a.memoryId,kind:"duplicate-memory",index:t,payload:a.cluster}}return null}dispose(){var e,s,t,a,n,o;(e=this.resources)==null||e.cellBuffer.destroy(),(s=this.resources)==null||s.neckBuffer.destroy(),(t=this.resources)==null||t.blurHBuffer.destroy(),(a=this.resources)==null||a.blurVBuffer.destroy(),(n=this.resources)==null||n.fieldA.destroy(),(o=this.resources)==null||o.fieldB.destroy(),this.resources=null}}function Lt(i,e,s,t,a,n){const o=a-s,m=n-t,v=i-s,x=e-t,L=o*v+m*x;if(L<=0)return Math.hypot(i-s,e-t);const c=o*o+m*m;if(c<=L)return Math.hypot(i-a,e-n);const y=L/c;return Math.hypot(i-(s+y*o),e-(t+y*m))}function Fe(i,e,s){i.pushErrorScope("validation");const t=i.createShaderModule({label:e,code:s});return t.getCompilationInfo().then(a=>{for(const n of a.messages)console.error(`[observatory] ${e} WGSL ${n.type} ${n.lineNum}:${n.linePos} ${n.message}`)}),i.popErrorScope().then(a=>{a&&console.error(`[observatory] ${e} shader module validation: ${a.message}`)}),t}function Tt(i,e){return ke(dt.blackwater),ke(Ve.recall),ke(Ve.luciferin),ke(mt.trustMembrane),[new Rt(i,e)]}function he(i){return Math.max(0,Math.min(1,Number.isFinite(i)?i:0))}function Ge(i,e,s){return s?{kind:i,id:e,scalar:s}:{kind:i,id:e||`${i}:unknown`}}function Et(i,e){return{kind:"scalar",id:`duplicates.${i}`,scalar:{name:i,value:e}}}function Dt(i,e=84){const s=(i||"").trim().replace(/\s+/g," ");return s.length<=e?s:`${s.slice(0,e)}…`}function Ke(i){return(i||"").toLowerCase().replace(/[^a-z0-9_\s-]/g," ").split(/\s+/).filter(e=>e.length>=4).slice(0,80)}function Ot(i){if(i.length<2)return[];const e=i.map(t=>new Set(Ke(t.content))),s=new Map;for(const t of e)for(const a of t)s.set(a,(s.get(a)??0)+1);return Array.from(s.entries()).filter(([,t])=>t>0&&t<i.length).sort((t,a)=>a[1]-t[1]||t[0].localeCompare(a[0])).slice(0,12).map(([t])=>t)}function Ut(i,e,s){const t=Array.isArray(i.memories)?i.memories.filter(m=>m.id):[];if(t.length<2)return null;const a=Ae(t),n=Xe(t),o=Ot(t);return{id:a,index:e,similarity:he(i.similarity),threshold:he(s),suggestedAction:i.suggestedAction==="merge"?"merge":"review",winnerId:(n==null?void 0:n.id)??t[0].id,memories:t.map((m,v)=>({...m,index:v,preview:Dt(m.content),winner:m.id===((n==null?void 0:n.id)??t[0].id),mismatchTokens:o.filter(x=>Ke(m.content).includes(x)).slice(0,8)})),mismatchTokens:o,source:Ge("pair",a)}}function Ft(i){var H;const e=he(i.threshold??.8),t=(Array.isArray(i.clusters)?i.clusters:[]).map((g,h)=>Ut(g,h,e)).filter(g=>g!==null);let a=0;const n=[],o=new Map;for(const g of t)for(const h of g.memories){if(o.has(h.id))continue;const M=a++;o.set(h.id,M),n.push({source:Ge("memory",h.id),index:M,label:h.preview||h.id.slice(0,8),retention:he(h.retention),trust:he(g.similarity),lastAccessed:h.createdAt,tags:[h.nodeType,...h.tags,h.winner?"winner":"candidate"].filter(Boolean),type:h.nodeType||"memory"})}const m=[];for(const g of t){const h=o.get(g.winnerId);if(h!=null)for(const M of g.memories){const z=o.get(M.id);z==null||z===h||m.push({source:Ge("pair",`${g.id}:${g.winnerId}:${M.id}`),sourceIndex:h,targetIndex:z,weight:Math.max(.05,g.similarity),kind:g.suggestedAction==="merge"?"fusion-candidate":"review-candidate"})}}const v=t.map((g,h)=>({source:Ge("event",`duplicates.cluster.${g.id}`),type:g.suggestedAction==="merge"?"DuplicateMergeCandidate":"DuplicateReviewCandidate",targetIndex:-1,frame:20+h*14,energy:Math.max(.1,g.similarity-e+.1)})),x=Number.isFinite(i.total)?i.total:t.length,L=n.length,c=t.reduce((g,h)=>Math.max(g,h.similarity),0),y=t.filter(g=>g.suggestedAction==="merge").length,T=t.length-y;return{organ:"duplicates",nodes:n,edges:m,events:v,receipts:[],scalars:{threshold:((H=Et("threshold",e).scalar)==null?void 0:H.value)??e,clusterCount:t.length,memoryCount:L,maxSimilarity:c,mergeCandidates:y,reviewCandidates:T,total:x},alive:t.length>0,threshold:e,total:x,clusters:t,raw:i}}const zt=()=>{var i;return typeof window<"u"&&((i=window.matchMedia)==null?void 0:i.call(window,"(prefers-reduced-motion: reduce)").matches)};function ze(i){if(zt())return{};let e=0;function s(a){const n=i.getBoundingClientRect();cancelAnimationFrame(e),e=requestAnimationFrame(()=>{i.style.setProperty("--spot-x",`${a.clientX-n.left}px`),i.style.setProperty("--spot-y",`${a.clientY-n.top}px`),i.style.setProperty("--spot-o","1")})}function t(){i.style.setProperty("--spot-o","0")}return i.addEventListener("pointermove",s),i.addEventListener("pointerleave",t),{destroy(){i.removeEventListener("pointermove",s),i.removeEventListener("pointerleave",t),cancelAnimationFrame(e)}}}var Nt=S('<span class="ping-host flex h-2 w-2 items-center justify-center text-synapse-glow" aria-hidden="true"><span class="breathe h-2 w-2 rounded-full bg-synapse-glow"></span></span> <span class="text-xs text-dim">Live</span>',1),It=S('<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span>Detecting…</span>',1),Vt=S('<span class="h-2 w-2 rounded-full bg-decay"></span> <span class="text-decay">Error</span>',1),Ht=S("<!> ",1),Wt=S("<!> ",1),$t=S('<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span class="tabular-nums"><!> · <!> memories implicated</span>',1),Yt=S('<div class="glass-panel pointer-events-auto rounded-2xl border border-synapse/25 bg-black/30 p-4"><div class="flex flex-wrap items-center justify-between gap-3"><div><div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">Synaptic neck selected</div> <div class="mt-1 text-sm text-bright"> </div> <div class="mt-1 max-w-2xl text-xs text-muted"> </div></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Clear field focus</button></div></div>'),jt=S(`<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"><div class="text-sm text-decay">Couldn't detect duplicates</div> <div class="max-w-md text-xs text-muted"> </div> <button type="button" class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Retry</button></div>`),Qt=S('<div class="glass-subtle shimmer h-40 rounded-2xl"></div>'),Xt=S('<div class="pointer-events-auto space-y-3"></div>'),qt=S('<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"><div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"><!></div> <div class="text-sm font-medium text-bright">No duplicates found — your memory is clean.</div> <div class="max-w-sm text-xs text-muted"> </div></div>'),Kt=S('<div class="glass-subtle rounded-xl border border-warning/30 bg-warning/5 px-4 py-2 text-xs text-dim"> </div>'),Zt=S('<div class="spotlight-surface lift rounded-2xl"><div class="relative z-[1]"><!></div></div>'),Jt=S('<div class="pointer-events-auto space-y-4"><!> <!></div>'),ei=S('<!> <div class="relative z-10 mx-auto max-h-dvh max-w-5xl space-y-6 overflow-y-auto overscroll-contain p-6 pb-28 pointer-events-none"><!> <div class="glass-panel pointer-events-auto flex flex-wrap items-center gap-5 rounded-2xl p-4"><label class="flex flex-1 min-w-64 items-center gap-3 text-xs text-dim"><span class="whitespace-nowrap">Similarity threshold</span> <input type="range" min="0.70" max="0.95" step="0.01" class="flex-1 accent-synapse" aria-label="Similarity threshold"/> <span class="w-14 text-right font-mono text-sm text-bright"> </span></label> <div class="flex items-center gap-2 rounded-full border border-synapse/20 bg-synapse/10 px-3 py-1.5 text-xs text-text" role="status" aria-live="polite"><!></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Rerun</button></div> <!> <!></div>',1);function hi(i,e){$e(e,!0);let s=re(.8),t=re(Ne([])),a=re(0);const n=12;let o=re(Ne(new Set)),m=re(!0),v=re(null),x=re(null),L;async function c(){F(m,!0),F(v,null),F(x,null);try{const l=await pt.duplicates(r(s));F(t,l.clusters,!0),F(a,l.total??l.clusters.length,!0);const f=new Set(r(t).map(A=>Ae(A.memories))),_=new Set;for(const A of r(o))f.has(A)&&_.add(A);F(o,_,!0)}catch(l){F(v,l instanceof Error?l.message:"Failed to detect duplicates",!0),F(t,[],!0)}finally{F(m,!1)}}function y(){clearTimeout(L),L=setTimeout(c,250)}function T(l){const f=new Set(r(o));f.add(l),F(o,f,!0),r(x)&&Ae(r(x).memories)===l&&F(x,null)}const P=q(()=>r(t).map(l=>({c:l,key:Ae(l.memories)})).filter(({key:l})=>!r(o).has(l))),H=q(()=>r(t).reduce((l,f)=>l+f.memories.length,0)),g=50,h=q(()=>r(P).length>g),M=q(()=>r(h)?r(P).slice(0,g):r(P)),z=q(()=>Ft({threshold:r(s),total:r(P).length,clusters:r(P).map(({c:l})=>l)}));function Y(l){l.kind!=="duplicate-neck"&&l.kind!=="duplicate-memory"||F(x,l.payload,!0)}et(()=>c()),tt(()=>clearTimeout(L));var K=ei(),O=ge(K);{let l=q(()=>`synaptic-fusion:${r(s)}:${r(P).length}:${r(H)}`),f=q(()=>`NO DUPLICATES ABOVE ${(r(s)*100).toFixed(0)}% SIMILARITY`);ut(O,{organ:"duplicates",get seed(){return r(l)},get scene(){return r(z)},get passes(){return Tt},get loading(){return r(m)},get error(){return r(v)},get emptyLabel(){return r(f)},onpick:Y})}var k=p(O,2),R=d(k);st(R,{icon:"duplicates",title:"Memory Hygiene — Duplicate Detection",subtitle:"Cosine-similarity clustering over embeddings. Oversized similarity components are quarantined for review — they chain through pairwise similarity and are not safe to merge. Dismissed clusters are hidden for this session only.",accent:"synapse",children:(l,f)=>{var _=Nt();we(2),w(l,_)},$$slots:{default:!0}});var te=p(R,2),ee=d(te),Z=p(d(ee),2);rt(Z);var se=p(Z,2),pe=d(se);u(se),u(ee);var ie=p(ee,2),be=d(ie);{var ye=l=>{var f=It();we(2),w(l,f)},ne=l=>{var f=Vt();we(2),w(l,f)},Re=l=>{var f=$t(),_=p(ge(f),2),A=d(_);{var G=I=>{var V=Ht(),Q=ge(V);Te(Q,{get value(){return r(P).length}});var U=p(Q);$(()=>C(U,` visible of ${r(a)??""} clusters`)),w(I,V)},N=I=>{var V=Wt(),Q=ge(V);Te(Q,{get value(){return r(P).length}});var U=p(Q);$(()=>C(U,` ${r(P).length===1?"cluster":"clusters"}`)),w(I,V)};J(A,I=>{r(P).length<r(a)?I(G):I(N,!1)})}var E=p(A,2);Te(E,{get value(){return r(H)}}),we(),u(_),w(l,f)};J(be,l=>{r(m)?l(ye):r(v)?l(ne,1):l(Re,!1)})}u(ie);var xe=p(ie,2);u(te);var B=p(te,2);{var b=l=>{var f=Yt(),_=d(f),A=d(_),G=p(d(A),2),N=d(G);u(G);var E=p(G,2),I=d(E);u(E),u(A);var V=p(A,2);u(_),u(f),$((Q,U,me)=>{C(N,`${r(x).memories.length??""} memories · ${Q??""}% similar · winner ${U??""}`),C(I,`Real pair key: ${r(x).id??""}. Mismatch filaments: ${me??""}.`)},[()=>(r(x).similarity*100).toFixed(1),()=>r(x).winnerId.slice(0,8),()=>r(x).mismatchTokens.length?r(x).mismatchTokens.join(", "):"none exposed"]),fe("click",V,()=>F(x,null)),w(l,f)};J(B,l=>{r(x)&&l(b)})}var j=p(B,2);{var ae=l=>{var f=jt(),_=p(d(f),2),A=d(_,!0);u(_);var G=p(_,2);u(f),$(()=>C(A,r(v))),fe("click",G,c),w(l,f)},le=l=>{var f=Xt();Ce(f,20,()=>Array(3),je,(_,A)=>{var G=Qt();w(_,G)}),u(f),w(l,f)},oe=l=>{var f=qt(),_=d(f),A=d(_);ct(A,{name:"sparkle",size:26,draw:!0}),u(_);var G=p(_,4),N=d(G);u(G),u(f),$(E=>C(N,`Nothing clusters above ${E??""}% similarity. Lower the threshold to
				surface looser matches.`),[()=>(r(s)*100).toFixed(0)]),w(l,f)},ce=l=>{var f=Jt(),_=d(f);{var A=N=>{var E=Kt(),I=d(E);u(E),$(()=>C(I,`Showing first 50 of ${r(P).length??""} clusters. Raise the
					threshold to narrow results.`)),w(N,E)};J(_,N=>{r(h)&&N(A)})}var G=p(_,2);Ce(G,19,()=>r(M),({c:N,key:E})=>E,(N,E,I)=>{let V=()=>r(E).c,Q=()=>r(E).key;var U=Zt(),me=d(U),_e=d(me);{let ue=q(()=>V().memories.length>n);Mt(_e,{get similarity(){return V().similarity},get memories(){return V().memories},get suggestedAction(){return V().suggestedAction},get oversized(){return r(ue)},onDismiss:()=>T(Q())})}u(me),u(U),Ie(U,(ue,W)=>{var X;return(X=nt)==null?void 0:X(ue,W)},()=>({delay:Math.min(r(I)*40,400),y:14})),Ie(U,ue=>ze==null?void 0:ze(ue)),w(N,U)}),u(f),w(l,f)};J(j,l=>{r(v)?l(ae):r(m)?l(le,1):r(P).length===0?l(oe,2):l(ce,!1)})}u(k),$(l=>{C(pe,`${l??""}%`),xe.disabled=r(m)},[()=>(r(s)*100).toFixed(0)]),fe("input",Z,y),at(Z,()=>r(s),l=>F(s,l)),fe("click",xe,c),w(i,K),Ye()}We(["input","click"]);export{hi as component};

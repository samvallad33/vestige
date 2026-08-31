var tt=Object.defineProperty;var it=(r,e,t)=>e in r?tt(r,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):r[e]=t;var F=(r,e,t)=>it(r,typeof e!="symbol"?e+"":e,t);import"../chunks/Bzak7iHL.js";import{d as $e,s as E,b as de,o as ze,a as rt}from"../chunks/BffzNaS8.js";import{p as qe,k as st,h as pe,g as s,a as P,b as Qe,u as q,c as d,r as c,j as v,t as I,s as N,f as S,d as ne,e as Ie,n as _e}from"../chunks/LfElJ0kU.js";import{i as Q}from"../chunks/3OEMGmei.js";import{e as Ge,i as Xe,a as Te,s as we,r as Ve}from"../chunks/CEZRBNoy.js";import{a as We}from"../chunks/BJ7CUkAM.js";import{b as je}from"../chunks/XJznBkE6.js";import{s as Pe}from"../chunks/BJDZGVD9.js";import{p as nt}from"../chunks/DSKAVXWq.js";import{N as at}from"../chunks/CcUbQ_Wl.js";import{P as lt}from"../chunks/CBi6Ab4S.js";import{I as ot}from"../chunks/c5I4rO_t.js";import{r as ct,A as Ee}from"../chunks/D7NzRT4w.js";import{R as ut}from"../chunks/Ct2wxOJK.js";import{r as Se,M as dt,R as He,I as ft,m as mt}from"../chunks/R1sFd5HP.js";import{a as pt}from"../chunks/DOaVlKeo.js";function Ke(r){return r>=.92?"near-identical":r>=.8?"strong":"weak"}function Oe(r){const e=Ke(r);return e==="near-identical"?"var(--color-decay)":e==="strong"?"var(--color-warning)":"#fde047"}function vt(r){const e=Ke(r);return e==="near-identical"?"Near-identical":e==="strong"?"Strong match":"Weak match"}function ht(r){return r>.7?"#10b981":r>.4?"#f59e0b":"#ef4444"}function Ze(r){if(!r||r.length===0)return null;let e=r[0],t=Number.isFinite(e.retention)?e.retention:-1/0;for(let i=1;i<r.length;i++){const n=r[i],a=Number.isFinite(n.retention)?n.retention:-1/0;a>t&&(e=n,t=a)}return e}function Me(r){return r.map(e=>e.id).slice().sort().join("|")}function gt(r,e=80){if(!r)return"";const t=r.trim().replace(/\s+/g," ");return t.length<=e?t:t.slice(0,e)+"…"}function Ye(r){if(!r||typeof r!="string")return"";const e=new Date(r);return Number.isNaN(e.getTime())?"":e.toLocaleDateString(void 0,{year:"numeric",month:"short",day:"numeric"})}function bt(r,e=4){return Array.isArray(r)?r.slice(0,e):[]}var yt=S('<span class="flex-shrink-0 rounded-full border border-warning/50 bg-warning/10 px-3 py-1 text-xs font-medium text-warning">REVIEW REQUIRED · NOT SAFE TO MERGE</span>'),xt=S("<span> </span>"),_t=S('<span class="rounded bg-recall/15 px-1.5 py-0.5 text-[10px] font-medium text-recall">WINNER</span>'),wt=S('<span class="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-muted"> </span>'),Pt=S('<div class="text-[11px] text-muted"> </div>'),St=S('<div><span class="mt-1.5 h-2 w-2 flex-shrink-0 rounded-full"></span> <div class="flex-1 min-w-0 space-y-1.5"><div class="flex flex-wrap items-center gap-1.5"><span class="text-xs text-dim"> </span> <!> <!></div> <p> </p> <!></div> <div class="flex flex-shrink-0 flex-col items-end gap-1"><div class="h-1.5 w-12 overflow-hidden rounded-full bg-deep"><div class="h-full rounded-full"></div></div> <span class="text-[11px] text-muted"> </span></div></div>'),Bt=S('<div class="rounded-xl border border-warning/20 bg-warning/5 p-3 text-xs text-dim"> </div>'),kt=S('<div class="glass-panel rounded-2xl p-5 space-y-4 transition-all duration-300 hover:border-synapse/20"><div class="flex items-start justify-between gap-4"><div class="flex-1 min-w-0 space-y-1.5"><div class="flex items-center gap-3"><span class="text-sm font-semibold"> </span> <span class="text-xs text-dim"> </span> <span class="text-xs text-muted"> </span></div> <div class="h-2 w-full overflow-hidden rounded-full bg-deep/60" role="progressbar" aria-label="Cosine similarity" aria-valuemin="0" aria-valuemax="100"><div class="h-full rounded-full transition-all duration-500"></div></div></div> <!></div> <div class="space-y-2"><!> <!></div> <div class="flex flex-wrap items-center gap-2 pt-1"><button type="button" disabled="" aria-disabled="true" aria-label="Merge is not available yet" class="cursor-not-allowed rounded-lg bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-muted/60">Merge unavailable</button> <button type="button" class="rounded-lg bg-dream/20 px-3 py-1.5 text-xs font-medium text-dream-glow transition hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream-glow/60"> </button> <button type="button" aria-label="Dismiss cluster for this session" class="ml-auto rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Dismiss cluster</button></div></div>');function Mt(r,e){qe(e,!0);let t=nt(e,"oversized",3,!1),i=ne(!1);const n=12,a=q(()=>Ze(e.memories)),o=q(()=>{if(e.memories.length<=n)return e.memories;const m=e.memories.filter(y=>{var O;return y.id!==((O=s(a))==null?void 0:O.id)});return s(a)?[s(a),...m.slice(0,n-1)]:m.slice(0,n)}),f=q(()=>e.memories.length-s(o).length);var h=st(),b=pe(h);{var D=m=>{var y=kt(),O=d(y),W=d(O),B=d(W),g=d(B),x=d(g);c(g);var C=v(g,2),V=d(C,!0);c(C);var j=v(C,2),K=d(j);c(j),c(B);var U=v(B,2),M=d(U);c(U),c(W);var A=v(W,2);{var Z=L=>{var _=yt();P(L,_)},te=L=>{var _=xt(),H=d(_);c(_),I(()=>{Te(_,1,`flex-shrink-0 rounded-full border px-3 py-1 text-xs font-medium ${e.suggestedAction==="merge"?"border-recall/40 bg-recall/10 text-recall":"border-dream-glow/40 bg-dream/10 text-dream-glow"}`),E(H,`Classification: ${e.suggestedAction==="merge"?"merge candidate":"review"}`)}),P(L,_)};Q(A,L=>{t()?L(Z):L(te,!1)})}c(O);var ie=v(O,2),fe=d(ie);Ge(fe,17,()=>s(o),L=>L.id,(L,_)=>{var H=St(),l=d(H),u=v(l,2),p=d(u),w=d(p),T=d(w,!0);c(w);var k=v(w,2);{var G=z=>{var se=_t();P(z,se)};Q(k,z=>{s(_).id===s(a).id&&z(G)})}var J=v(k,2);Ge(J,17,()=>bt(s(_).tags,4),Xe,(z,se)=>{var ue=wt(),Le=d(ue,!0);c(ue),I(()=>E(Le,s(se))),P(z,ue)}),c(p);var R=v(p,2),Y=d(R,!0);c(R);var $=v(R,2);{var re=z=>{var se=Pt(),ue=d(se,!0);c(se),I(Le=>E(ue,Le),[()=>Ye(s(_).createdAt)]),P(z,se)},ge=q(()=>Ye(s(_).createdAt));Q($,z=>{s(ge)&&z(re)})}c(u);var ee=v(u,2),oe=d(ee),X=d(oe);c(oe);var ae=v(oe,2),ce=d(ae);c(ae),c(ee),c(H),I((z,se,ue)=>{Te(H,1,`group flex items-start gap-3 rounded-xl border border-synapse/5 bg-white/[0.02] p-3 transition-all duration-200 hover:border-synapse/20 hover:bg-white/[0.04] ${s(_).id===s(a).id?"ring-1 ring-recall/30":""}`),Pe(l,`background: ${(at[s(_).nodeType]||"#8B95A5")??""}`),we(l,"title",s(_).nodeType),E(T,s(_).nodeType),Te(R,1,`text-sm text-text leading-relaxed ${s(i)?"whitespace-pre-wrap":""}`),E(Y,z),Pe(X,`width: ${s(_).retention*100}%; background: ${se??""}`),E(ce,`${ue??""}%`)},[()=>s(i)?s(_).content:gt(s(_).content),()=>ht(s(_).retention),()=>(s(_).retention*100).toFixed(0)]),P(L,H)});var ve=v(fe,2);{var me=L=>{var _=Bt(),H=d(_);c(_),I(()=>E(H,`+${s(f)??""} linked candidates — oversized similarity component. Members
					chain through pairwise similarity; distant members may be unrelated. Raise
					the threshold to split it.`)),P(L,_)};Q(ve,L=>{s(f)>0&&L(me)})}c(ie);var xe=v(ie,2),he=d(xe),le=v(he,2),Re=d(le,!0);c(le);var Ce=v(le,2);c(xe),c(y),I((L,_,H,l,u,p,w)=>{Pe(g,`color: ${L??""}`),E(x,`${_??""}%`),E(V,H),E(K,`· ${e.memories.length??""} memories`),we(U,"aria-valuenow",l),Pe(M,`width: ${u??""}%; background: ${p??""}; box-shadow: 0 0 12px ${w??""}66`),we(he,"title",t()?"Oversized similarity component — not safe to merge":"Merge backend not shipped yet — no destructive action is taken from this screen"),we(le,"aria-expanded",s(i)),E(Re,s(i)?"Collapse":"Review")},[()=>Oe(e.similarity),()=>(e.similarity*100).toFixed(1),()=>vt(e.similarity),()=>Math.round(e.similarity*100),()=>(e.similarity*100).toFixed(1),()=>Oe(e.similarity),()=>Oe(e.similarity)]),de("click",le,()=>N(i,!s(i))),de("click",Ce,function(...L){var _;(_=e.onDismiss)==null||_.apply(this,L)}),P(m,y)};Q(b,m=>{e.memories.length>0&&s(a)&&m(D)})}P(r,h),Qe()}$e(["click"]);const Be="rgba16float",be=512,ke=512,Fe=16,De=16,Je=`
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
${Je}

// FieldOpts mirrors the membrane's: x=intensity, yz=well center NDC, w=well
// half-w; then well half-h, floor, soft, pad. Cells/necks dim by the same amount
// so nothing blows out under the centered text overlay.
struct FieldOpts {
	intensity_wx_wy_hw: vec4f,
	hh_floor_soft_pad: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<FusionCell>;
@group(0) @binding(2) var<storage, read> necks: array<FusionNeck>;
@group(0) @binding(5) var<uniform> opts: FieldOpts;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) misc: vec4f,
	@location(2) @interpolate(flat) home: vec2f,
};

fn similarity_neck(similarity: f32) -> f32 {
	return smoothstep(0.78, 0.98, similarity);
}

// Reading-well multiplier at an NDC point (1.0 outside, →floor inside). hw<=0 off.
fn field_dim(ndc: vec2f) -> f32 {
	let intensity = clamp(opts.intensity_wx_wy_hw.x, 0.0, 1.0);
	let hw = opts.intensity_wx_wy_hw.w;
	if (hw <= 0.0) { return intensity; }
	let center = opts.intensity_wx_wy_hw.yz;
	let hh = opts.hh_floor_soft_pad.x;
	let floor_v = opts.hh_floor_soft_pad.y;
	let soft = max(0.02, opts.hh_floor_soft_pad.z);
	let d = abs(ndc - center) - vec2f(hw, hh);
	let outside = length(max(d, vec2f(0.0)));
	let inside = min(max(d.x, d.y), 0.0);
	let sd = outside + inside;
	let t = smoothstep(-soft, 0.0, sd);
	return intensity * mix(floor_v, 1.0, t);
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
		out.home = c.pos_retention.xy;
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
		out.home = center;
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
	// The splat writes the density FIELD (blurred into the membrane). It must NOT
	// be dimmed here — the membrane fragment applies intensity + reading well once,
	// so dimming both would double-darken. r=density, g=retention, b=mismatch amber.
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
	out.home = c.pos_retention.xy;
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
	let color = core * body + ivory * rim * (0.16 + similarity * 0.52) + amber * mismatch_ring * 0.34;
	// Sharp cells draw on TOP of the membrane, so dim them by the same field
	// intensity + reading well or they'd punch through the centered text.
	return vec4f(color * field_dim(frag.home), 1.0);
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
	out.home = midpoint;
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
	// Necks draw on TOP of the membrane too — dim by field intensity + reading well.
	return vec4f(color * (0.14 + similarity * 0.55 + mismatch * 0.18) * field_dim(frag.home), 1.0);
}
`,Gt=`
${Je}

// FieldOpts: x=intensity (0..1 overall dim), yz=well center NDC, w=well half-w,
// then well half-h, floor (min emission inside well), soft (edge falloff), pad.
// Lets a text-heavy organ dim the whole field AND carve a reading well under the
// centered DOM overlay so the labels/values read. hw<=0 disables the well.
struct FieldOpts {
	intensity_wx_wy_hw: vec4f,
	hh_floor_soft_pad: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> opts: FieldOpts;

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

// Reading-well multiplier at an NDC point: 1.0 outside the well, falling toward
// the floor value inside it (smooth edge of width soft). Disabled when hw<=0.
fn reading_well(ndc: vec2f) -> f32 {
	let hw = opts.intensity_wx_wy_hw.w;
	if (hw <= 0.0) { return 1.0; }
	let center = opts.intensity_wx_wy_hw.yz;
	let hh = opts.hh_floor_soft_pad.x;
	let floor_v = opts.hh_floor_soft_pad.y;
	let soft = max(0.02, opts.hh_floor_soft_pad.z);
	let d = abs(ndc - center) - vec2f(hw, hh);
	// signed distance to rect edge: <0 inside, >0 outside
	let outside = length(max(d, vec2f(0.0)));
	let inside = min(max(d.x, d.y), 0.0);
	let sd = outside + inside;
	// sd<=-soft → fully inside (floor); sd>=0 → outside (1.0)
	let t = smoothstep(-soft, 0.0, sd);
	return mix(floor_v, 1.0, t);
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
	let ndc = frag.uv * 2.0 - vec2f(1.0);
	let dim = clamp(opts.intensity_wx_wy_hw.x, 0.0, 1.0) * reading_well(ndc);
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness * dim, 1.0);
}
`,Rt=`
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
`;class Ct{constructor(e,t){F(this,"engine");F(this,"scene",null);F(this,"resources",null);F(this,"sampler",null);F(this,"splatBindLayout",null);F(this,"blurBindLayout",null);F(this,"membraneBindLayout",null);F(this,"splatPipeline",null);F(this,"blurPipeline",null);F(this,"membranePipeline",null);F(this,"cellPipeline",null);F(this,"neckPipeline",null);F(this,"cellCount",0);F(this,"neckCount",0);F(this,"cellGeometry",[]);F(this,"neckGeometry",[]);F(this,"intensity",.22);F(this,"well",{x:0,y:0,hw:-1,hh:0,floor:.1,soft:.22});this.engine=e,this.uploadScene(t)}setIntensity(e){this.intensity=Math.min(1,Math.max(0,Number.isFinite(e)?e:.22));const t=this.engine.gpuDevice;t&&this.writeOpts(t)}setReadingWell(e){const t=(n,a=0)=>Number.isFinite(n)?n:a;this.well={x:t(e.x),y:t(e.y),hw:t(e.hw,-1),hh:t(e.hh),floor:Math.min(1,Math.max(0,t(e.floor??.1,.1))),soft:Math.max(.02,t(e.soft??.22,.22))};const i=this.engine.gpuDevice;i&&this.writeOpts(i)}writeOpts(e){this.resources&&e.queue.writeBuffer(this.resources.optsBuffer,0,new Float32Array([this.intensity,this.well.x,this.well.y,this.well.hw,this.well.hh,this.well.floor,this.well.soft,0]))}uploadScene(e){this.scene=e,this.buildGeometry();const t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.uploadBuffers(t))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const t=Ne(e,"duplicates-fusion-splat-wgsl",At),i=Ne(e,"duplicates-fusion-blur-wgsl",Rt),n=Ne(e,"duplicates-fusion-membrane-wgsl",Gt);this.splatBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-splat-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-blur-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"duplicates-fusion-membrane-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const a=e.createPipelineLayout({label:"duplicates-fusion-splat-layout",bindGroupLayouts:[this.splatBindLayout]}),o=e.createPipelineLayout({label:"duplicates-fusion-blur-layout",bindGroupLayouts:[this.blurBindLayout]}),f=e.createPipelineLayout({label:"duplicates-fusion-membrane-layout",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"});const h={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.splatPipeline=e.createRenderPipeline({label:"duplicates-field-additive-splat",layout:a,vertex:{module:t,entryPoint:"vs_splat"},fragment:{module:t,entryPoint:"fs_splat",targets:[{format:Be,blend:h}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"duplicates-field-blur-render-pass",layout:o,vertex:{module:i,entryPoint:"vs_fullscreen"},fragment:{module:i,entryPoint:"fs_blur",targets:[{format:Be}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=e.createRenderPipeline({label:"duplicates-synaptic-fusion-membrane",layout:f,vertex:{module:n,entryPoint:"vs_fullscreen"},fragment:{module:n,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:h}]},primitive:{topology:"triangle-list"}}),this.cellPipeline=e.createRenderPipeline({label:"duplicates-memory-nuclei",layout:a,vertex:{module:t,entryPoint:"vs_cell"},fragment:{module:t,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:h}]},primitive:{topology:"triangle-list"}}),this.neckPipeline=e.createRenderPipeline({label:"duplicates-mismatch-filaments",layout:a,vertex:{module:t,entryPoint:"vs_neck"},fragment:{module:t,entryPoint:"fs_neck",targets:[{format:this.engine.sceneFormat,blend:h}]},primitive:{topology:"triangle-strip"}})}ensureResources(e){var V,j,K,U,M,A,Z;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),i=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),n=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==i;let a=(V=this.resources)==null?void 0:V.cellBuffer,o=(j=this.resources)==null?void 0:j.neckBuffer,f=(K=this.resources)==null?void 0:K.blurHBuffer,h=(U=this.resources)==null?void 0:U.blurVBuffer,b=(M=this.resources)==null?void 0:M.optsBuffer;if(a||(a=e.createBuffer({label:"duplicates-cells",size:be*Fe*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),o||(o=e.createBuffer({label:"duplicates-necks",size:ke*De*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),f||(f=e.createBuffer({label:"duplicates-blur-h-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(f,0,new Float32Array([1,0,0,0]))),h||(h=e.createBuffer({label:"duplicates-blur-v-dir",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(h,0,new Float32Array([0,1,0,0]))),b||(b=e.createBuffer({label:"duplicates-field-opts",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),!n&&this.resources){this.resources.optsBuffer=b,this.writeOpts(e);return}(A=this.resources)==null||A.fieldA.destroy(),(Z=this.resources)==null||Z.fieldB.destroy();const D=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,m=e.createTexture({label:"duplicates-field-a-rgba16float",size:[t,i],format:Be,usage:D}),y=e.createTexture({label:"duplicates-field-b-rgba16float",size:[t,i],format:Be,usage:D}),O=m.createView(),W=y.createView(),B=e.createBindGroup({label:"duplicates-fusion-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:o}},{binding:5,resource:{buffer:b}}]}),g=e.createBindGroup({label:"duplicates-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:O},{binding:2,resource:{buffer:f}}]}),x=e.createBindGroup({label:"duplicates-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:W},{binding:2,resource:{buffer:h}}]}),C=e.createBindGroup({label:"duplicates-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:O},{binding:5,resource:{buffer:b}}]});this.resources={cellBuffer:a,neckBuffer:o,blurHBuffer:f,blurVBuffer:h,optsBuffer:b,splatBindGroup:B,blurHBindGroup:g,blurVBindGroup:x,membraneBindGroup:C,fieldA:m,fieldB:y,fieldAView:O,fieldBView:W,fieldSize:[t,i]},this.writeOpts(e)}buildGeometry(){var b,D;const e=((b=this.scene)==null?void 0:b.clusters)??[],t=Math.max(1,e.length),i=[],n=[],a=12,o=new Array(e.length).fill(0);let f=be;for(let m=0;m<e.length&&f>0;m++)o[m]=1,f-=1;for(let m=0;m<e.length&&f>0;m++)o[m]===1&&e[m].memories.length>=2&&(o[m]=2,f-=1);let h=!0;for(;f>0&&h;){h=!1;for(let m=0;m<e.length&&f>0;m++)o[m]>0&&o[m]<Math.min(e[m].memories.length,a)&&(o[m]+=1,f-=1,h=!0)}for(let m=0;m<e.length;m++){const y=e[m];if(o[m]===0)continue;const O=m/t*Math.PI*2-Math.PI/2,W=.18+.58*Math.sqrt((m+.5)/t),B=Math.cos(O)*W*.86,g=Math.sin(O)*W,x=Math.max(.04,.25-Math.max(0,y.similarity-y.threshold)*.55),C=y.memories.find(M=>M.id===y.winnerId)??y.memories[0],V=[C,...y.memories.filter(M=>M.id!==C.id)].slice(0,o[m]),j=Math.max(1,V.length),K=new Map;for(let M=0;M<V.length&&i.length<be;M++){const A=V[M],Z=O+M/j*Math.PI*2+(j%2?0:Math.PI/j),te=A.id===y.winnerId,ie=te?x*.18:x+.025*(M%3),fe=Math.min(1,(((D=A.mismatchTokens)==null?void 0:D.length)??0)/8),ve=.085+Math.min(.045,mt(A.retention)*2.1)+(te?.012:0),me={cluster:y,memoryId:A.id,x:B+Math.cos(Z)*ie,y:g+Math.sin(Z)*ie,retention:Math.max(0,Math.min(1,A.retention||0)),winner:te,mismatch:fe,radius:ve,memberSlot:M,memberCount:j};K.set(A.id,me),i.push(me)}const U=K.get(C.id);if(U)for(const M of y.memories){if(n.length>=ke||M.id===C.id)continue;const A=K.get(M.id);A&&n.push({cluster:y,winnerId:C.id,candidateId:M.id,ax:U.x,ay:U.y,bx:A.x,by:A.y,winnerRetention:U.retention,candidateRetention:A.retention,winnerRadius:U.radius,candidateRadius:A.radius,mismatch:Math.max(A.mismatch,Math.min(1,y.mismatchTokens.length/12))})}}this.cellGeometry=i,this.neckGeometry=n}uploadBuffers(e){if(!this.resources)return;const t=new Float32Array(be*Fe),i=new Float32Array(ke*De);this.cellCount=Math.min(be,this.cellGeometry.length),this.neckCount=Math.min(ke,this.neckGeometry.length);for(let n=0;n<this.cellCount;n++){const a=this.cellGeometry[n];t.set([a.x,a.y,a.retention,a.winner?1:0,a.cluster.similarity,a.cluster.threshold,a.memberSlot,a.cluster.index,a.mismatch,a.cluster.suggestedAction==="merge"?1:0,a.radius,a.memberCount,n,a.cluster.index,0,0],n*Fe)}for(let n=0;n<this.neckCount;n++){const a=this.neckGeometry[n];i.set([a.ax,a.ay,a.winnerRetention,a.winnerRadius,a.bx,a.by,a.candidateRetention,a.candidateRadius,a.cluster.similarity,a.cluster.threshold,a.mismatch,a.cluster.suggestedAction==="merge"?1:0,n,a.cluster.index,0,0],n*De)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.neckCount,this.engine.params[4]=this.neckCount,e.queue.writeBuffer(this.resources.cellBuffer,0,t),e.queue.writeBuffer(this.resources.neckBuffer,0,i)}compute(e){const t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);const i=this.resources,n=e.beginRenderPass({label:"duplicates-field-splat-pass",colorAttachments:[{view:i.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(this.splatPipeline),n.setBindGroup(0,i.splatBindGroup),n.draw(6,this.cellCount+this.neckCount),n.end();const a=e.beginRenderPass({label:"duplicates-field-blur-h-pass",colorAttachments:[{view:i.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});a.setPipeline(this.blurPipeline),a.setBindGroup(0,i.blurHBindGroup),a.draw(6,1),a.end();const o=e.beginRenderPass({label:"duplicates-field-blur-v-pass",colorAttachments:[{view:i.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});o.setPipeline(this.blurPipeline),o.setBindGroup(0,i.blurVBindGroup),o.draw(6,1),o.end()}render(e){!this.resources||!this.membranePipeline||!this.cellPipeline||!this.neckPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.neckCount>0&&(e.setPipeline(this.neckPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(64,this.neckCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(6,this.cellCount)))}pickAt(e,t){for(let i=0;i<this.neckGeometry.length;i++){const n=this.neckGeometry[i],a=Lt(e,t,n.ax,n.ay,n.bx,n.by),o=(n.ax+n.bx)*.5,f=(n.ay+n.by)*.5,h=.055+Math.max(0,n.cluster.similarity-n.cluster.threshold)*.45;if(a<=h||Math.hypot(e-o,t-f)<=h)return{id:n.cluster.id,kind:"duplicate-neck",index:i,payload:n.cluster}}for(let i=0;i<this.cellGeometry.length;i++){const n=this.cellGeometry[i];if(Math.hypot(e-n.x,t-n.y)<=n.radius*.8)return{id:n.memoryId,kind:"duplicate-memory",index:i,payload:n.cluster}}return null}dispose(){var e,t,i,n,a,o,f;(e=this.resources)==null||e.cellBuffer.destroy(),(t=this.resources)==null||t.neckBuffer.destroy(),(i=this.resources)==null||i.blurHBuffer.destroy(),(n=this.resources)==null||n.blurVBuffer.destroy(),(a=this.resources)==null||a.optsBuffer.destroy(),(o=this.resources)==null||o.fieldA.destroy(),(f=this.resources)==null||f.fieldB.destroy(),this.resources=null}}function Lt(r,e,t,i,n,a){const o=n-t,f=a-i,h=r-t,b=e-i,D=o*h+f*b;if(D<=0)return Math.hypot(r-t,e-i);const m=o*o+f*f;if(m<=D)return Math.hypot(r-n,e-a);const y=D/m;return Math.hypot(r-(t+y*o),e-(i+y*f))}function Ne(r,e,t){r.pushErrorScope("validation");const i=r.createShaderModule({label:e,code:t});return i.getCompilationInfo().then(n=>{for(const a of n.messages)console.error(`[observatory] ${e} WGSL ${a.type} ${a.lineNum}:${a.linePos} ${a.message}`)}),r.popErrorScope().then(n=>{n&&console.error(`[observatory] ${e} shader module validation: ${n.message}`)}),i}function Tt(r,e){Se(dt.blackwater),Se(He.recall),Se(He.luciferin),Se(ft.trustMembrane);const t=new Ct(r,e);return t.setIntensity(.22),t.setReadingWell({x:0,y:0,hw:.6,hh:.85,floor:.08,soft:.25}),[t]}function ye(r){return Math.max(0,Math.min(1,Number.isFinite(r)?r:0))}function Ae(r,e,t){return t?{kind:r,id:e,scalar:t}:{kind:r,id:e||`${r}:unknown`}}function Et(r,e){return{kind:"scalar",id:`duplicates.${r}`,scalar:{name:r,value:e}}}function Ot(r,e=84){const t=(r||"").trim().replace(/\s+/g," ");return t.length<=e?t:`${t.slice(0,e)}…`}function et(r){return(r||"").toLowerCase().replace(/[^a-z0-9_\s-]/g," ").split(/\s+/).filter(e=>e.length>=4).slice(0,80)}function Ft(r){if(r.length<2)return[];const e=r.map(i=>new Set(et(i.content))),t=new Map;for(const i of e)for(const n of i)t.set(n,(t.get(n)??0)+1);return Array.from(t.entries()).filter(([,i])=>i>0&&i<r.length).sort((i,n)=>n[1]-i[1]||i[0].localeCompare(n[0])).slice(0,12).map(([i])=>i)}function Dt(r,e,t){const i=Array.isArray(r.memories)?r.memories.filter(f=>f.id):[];if(i.length<2)return null;const n=Me(i),a=Ze(i),o=Ft(i);return{id:n,index:e,similarity:ye(r.similarity),threshold:ye(t),suggestedAction:r.suggestedAction==="merge"?"merge":"review",winnerId:(a==null?void 0:a.id)??i[0].id,memories:i.map((f,h)=>({...f,index:h,preview:Ot(f.content),winner:f.id===((a==null?void 0:a.id)??i[0].id),mismatchTokens:o.filter(b=>et(f.content).includes(b)).slice(0,8)})),mismatchTokens:o,source:Ae("pair",n)}}function Nt(r){var B;const e=ye(r.threshold??.8),i=(Array.isArray(r.clusters)?r.clusters:[]).map((g,x)=>Dt(g,x,e)).filter(g=>g!==null);let n=0;const a=[],o=new Map;for(const g of i)for(const x of g.memories){if(o.has(x.id))continue;const C=n++;o.set(x.id,C),a.push({source:Ae("memory",x.id),index:C,label:x.preview||x.id.slice(0,8),retention:ye(x.retention),trust:ye(g.similarity),lastAccessed:x.createdAt,tags:[x.nodeType,...x.tags,x.winner?"winner":"candidate"].filter(Boolean),type:x.nodeType||"memory"})}const f=[];for(const g of i){const x=o.get(g.winnerId);if(x!=null)for(const C of g.memories){const V=o.get(C.id);V==null||V===x||f.push({source:Ae("pair",`${g.id}:${g.winnerId}:${C.id}`),sourceIndex:x,targetIndex:V,weight:Math.max(.05,g.similarity),kind:g.suggestedAction==="merge"?"fusion-candidate":"review-candidate"})}}const h=i.map((g,x)=>({source:Ae("event",`duplicates.cluster.${g.id}`),type:g.suggestedAction==="merge"?"DuplicateMergeCandidate":"DuplicateReviewCandidate",targetIndex:-1,frame:20+x*14,energy:Math.max(.1,g.similarity-e+.1)})),b=Number.isFinite(r.total)?r.total:i.length,D=a.length,m=i.reduce((g,x)=>Math.max(g,x.similarity),0),y=i.filter(g=>g.suggestedAction==="merge").length,O=i.length-y;return{organ:"duplicates",nodes:a,edges:f,events:h,receipts:[],scalars:{threshold:((B=Et("threshold",e).scalar)==null?void 0:B.value)??e,clusterCount:i.length,memoryCount:D,maxSimilarity:m,mergeCandidates:y,reviewCandidates:O,total:b},alive:i.length>0,threshold:e,total:b,clusters:i,raw:r}}const Ut=()=>{var r;return typeof window<"u"&&((r=window.matchMedia)==null?void 0:r.call(window,"(prefers-reduced-motion: reduce)").matches)};function Ue(r){if(Ut())return{};let e=0;function t(n){const a=r.getBoundingClientRect();cancelAnimationFrame(e),e=requestAnimationFrame(()=>{r.style.setProperty("--spot-x",`${n.clientX-a.left}px`),r.style.setProperty("--spot-y",`${n.clientY-a.top}px`),r.style.setProperty("--spot-o","1")})}function i(){r.style.setProperty("--spot-o","0")}return r.addEventListener("pointermove",t),r.addEventListener("pointerleave",i),{destroy(){r.removeEventListener("pointermove",t),r.removeEventListener("pointerleave",i),cancelAnimationFrame(e)}}}var zt=S('<span class="ping-host flex h-2 w-2 items-center justify-center text-synapse-glow" aria-hidden="true"><span class="breathe h-2 w-2 rounded-full bg-synapse-glow"></span></span> <span class="text-xs text-dim">Live</span>',1),It=S('<label class="flex w-full flex-col gap-2 text-xs text-dim"><span class="flex items-baseline justify-between gap-3"><span class="whitespace-nowrap">Similarity threshold</span> <span class="font-mono text-sm text-bright"> </span></span> <input type="range" min="0.70" max="0.95" step="0.01" class="w-full accent-synapse" aria-label="Similarity threshold"/></label>'),Vt=S('<label class="flex flex-1 min-w-64 items-center gap-3 text-xs text-dim"><span class="whitespace-nowrap">Similarity threshold</span> <input type="range" min="0.70" max="0.95" step="0.01" class="flex-1 accent-synapse" aria-label="Similarity threshold"/> <span class="w-14 text-right font-mono text-sm text-bright"> </span></label>'),Wt=S('<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span>Detecting…</span>',1),jt=S('<span class="h-2 w-2 rounded-full bg-decay"></span> <span class="text-decay">Error</span>',1),Ht=S("<!> ",1),Yt=S("<!> ",1),$t=S('<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span class="tabular-nums"><!> · <!> memories implicated</span>',1),qt=S('<div class="flex items-center gap-2 rounded-full border border-synapse/20 bg-synapse/10 px-3 py-1.5 text-xs text-text" role="status" aria-live="polite"><!></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Rerun</button>',1),Qt=S('<div class="glass-panel pointer-events-auto rounded-2xl border border-synapse/25 bg-black/30 p-4"><div class="flex flex-wrap items-center justify-between gap-3"><div><div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">Synaptic neck selected</div> <div class="mt-1 text-sm text-bright"> </div> <div class="mt-1 max-w-2xl text-xs text-muted"> </div></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Clear field focus</button></div></div>'),Xt=S(`<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"><div class="text-sm text-decay">Couldn't detect duplicates</div> <div class="max-w-md text-xs text-muted"> </div> <button type="button" class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Retry</button></div>`),Kt=S('<div class="glass-subtle shimmer h-40 rounded-2xl"></div>'),Zt=S('<div class="pointer-events-auto space-y-3"></div>'),Jt=S('<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"><div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"><!></div> <div class="text-sm font-medium text-bright">No duplicates found — your memory is clean.</div> <div class="max-w-sm text-xs text-muted"> </div></div>'),ei=S('<div class="glass-subtle rounded-xl border border-warning/30 bg-warning/5 px-4 py-2 text-xs text-dim"> </div>'),ti=S('<div class="spotlight-surface lift rounded-2xl"><div class="relative z-[1]"><!></div></div>'),ii=S('<div class="pointer-events-auto space-y-4"><!> <!></div>'),ri=S('<!> <div class="relative z-10 mx-auto max-h-dvh max-w-5xl space-y-6 overflow-y-auto overscroll-contain p-6 pb-28 pointer-events-none"><!> <div class="glass-panel pointer-events-auto flex flex-wrap items-center gap-5 rounded-2xl p-4"><!> <!></div> <!> <!></div>',1);function _i(r,e){qe(e,!0);let t=ne(.8),i=ne(Ie([])),n=ne(0);const a=12;let o=ne(Ie(new Set)),f=ne(!0),h=ne(null),b=ne(null),D,m=ne(!1);ze(()=>{const l=()=>{N(m,window.innerWidth/Math.max(1,window.innerHeight)<.85)};return l(),window.addEventListener("resize",l),()=>window.removeEventListener("resize",l)});async function y(){N(f,!0),N(h,null),N(b,null);try{const l=await pt.duplicates(s(t));N(i,l.clusters,!0),N(n,l.total??l.clusters.length,!0);const u=new Set(s(i).map(w=>Me(w.memories))),p=new Set;for(const w of s(o))u.has(w)&&p.add(w);N(o,p,!0)}catch(l){N(h,l instanceof Error?l.message:"Failed to detect duplicates",!0),N(i,[],!0)}finally{N(f,!1)}}function O(){clearTimeout(D),D=setTimeout(y,250)}function W(l){const u=new Set(s(o));u.add(l),N(o,u,!0),s(b)&&Me(s(b).memories)===l&&N(b,null)}const B=q(()=>s(i).map(l=>({c:l,key:Me(l.memories)})).filter(({key:l})=>!s(o).has(l))),g=q(()=>s(i).reduce((l,u)=>l+u.memories.length,0)),x=50,C=q(()=>s(B).length>x),V=q(()=>s(C)?s(B).slice(0,x):s(B)),j=q(()=>Nt({threshold:s(t),total:s(B).length,clusters:s(B).map(({c:l})=>l)}));function K(l){l.kind!=="duplicate-neck"&&l.kind!=="duplicate-memory"||N(b,l.payload,!0)}ze(()=>y()),rt(()=>clearTimeout(D));var U=ri(),M=pe(U);{let l=q(()=>`synaptic-fusion:${s(t)}:${s(B).length}:${s(g)}`),u=q(()=>`NO DUPLICATES ABOVE ${(s(t)*100).toFixed(0)}% SIMILARITY`);ut(M,{organ:"duplicates",get seed(){return s(l)},get scene(){return s(j)},get passes(){return Tt},get loading(){return s(f)},get error(){return s(h)},get emptyLabel(){return s(u)},onpick:K})}var A=v(M,2),Z=d(A);lt(Z,{icon:"duplicates",title:"Memory Hygiene: Duplicate Detection",subtitle:"Cosine-similarity clustering over embeddings. Oversized similarity components are quarantined for review because they chain through pairwise similarity and are not safe to merge. Dismissed clusters are hidden for this session only.",accent:"synapse",children:(l,u)=>{var p=zt();_e(2),P(l,p)},$$slots:{default:!0}});var te=v(Z,2),ie=d(te);{var fe=l=>{var u=It(),p=d(u),w=v(d(p),2),T=d(w);c(w),c(p);var k=v(p,2);Ve(k),c(u),I(G=>E(T,`${G??""}%`),[()=>(s(t)*100).toFixed(0)]),de("input",k,O),je(k,()=>s(t),G=>N(t,G)),P(l,u)},ve=l=>{var u=Vt(),p=v(d(u),2);Ve(p);var w=v(p,2),T=d(w);c(w),c(u),I(k=>E(T,`${k??""}%`),[()=>(s(t)*100).toFixed(0)]),de("input",p,O),je(p,()=>s(t),k=>N(t,k)),P(l,u)};Q(ie,l=>{s(m)?l(fe):l(ve,!1)})}var me=v(ie,2);{var xe=l=>{var u=qt(),p=pe(u),w=d(p);{var T=R=>{var Y=Wt();_e(2),P(R,Y)},k=R=>{var Y=jt();_e(2),P(R,Y)},G=R=>{var Y=$t(),$=v(pe(Y),2),re=d($);{var ge=X=>{var ae=Ht(),ce=pe(ae);Ee(ce,{get value(){return s(B).length}});var z=v(ce);I(()=>E(z,` visible of ${s(n)??""} clusters`)),P(X,ae)},ee=X=>{var ae=Yt(),ce=pe(ae);Ee(ce,{get value(){return s(B).length}});var z=v(ce);I(()=>E(z,` ${s(B).length===1?"cluster":"clusters"}`)),P(X,ae)};Q(re,X=>{s(B).length<s(n)?X(ge):X(ee,!1)})}var oe=v(re,2);Ee(oe,{get value(){return s(g)}}),_e(),c($),P(R,Y)};Q(w,R=>{s(f)?R(T):s(h)?R(k,1):R(G,!1)})}c(p);var J=v(p,2);I(()=>J.disabled=s(f)),de("click",J,y),P(l,u)};Q(me,l=>{s(h)&&s(m)||l(xe)})}c(te);var he=v(te,2);{var le=l=>{var u=Qt(),p=d(u),w=d(p),T=v(d(w),2),k=d(T);c(T);var G=v(T,2),J=d(G);c(G),c(w);var R=v(w,2);c(p),c(u),I((Y,$,re)=>{E(k,`${s(b).memories.length??""} memories · ${Y??""}% similar · winner ${$??""}`),E(J,`Real pair key: ${s(b).id??""}. Mismatch filaments: ${re??""}.`)},[()=>(s(b).similarity*100).toFixed(1),()=>s(b).winnerId.slice(0,8),()=>s(b).mismatchTokens.length?s(b).mismatchTokens.join(", "):"none exposed"]),de("click",R,()=>N(b,null)),P(l,u)};Q(he,l=>{s(b)&&l(le)})}var Re=v(he,2);{var Ce=l=>{var u=Xt(),p=v(d(u),2),w=d(p,!0);c(p);var T=v(p,2);c(u),I(()=>E(w,s(h))),de("click",T,y),P(l,u)},L=l=>{var u=Zt();Ge(u,20,()=>Array(3),Xe,(p,w)=>{var T=Kt();P(p,T)}),c(u),P(l,u)},_=l=>{var u=Jt(),p=d(u),w=d(p);ot(w,{name:"sparkle",size:26,draw:!0}),c(p);var T=v(p,4),k=d(T);c(T),c(u),I(G=>E(k,`Nothing clusters above ${G??""}% similarity. Lower the threshold to
				surface looser matches.`),[()=>(s(t)*100).toFixed(0)]),P(l,u)},H=l=>{var u=ii(),p=d(u);{var w=k=>{var G=ei(),J=d(G);c(G),I(()=>E(J,`Showing first 50 of ${s(B).length??""} clusters. Raise the
					threshold to narrow results.`)),P(k,G)};Q(p,k=>{s(C)&&k(w)})}var T=v(p,2);Ge(T,19,()=>s(V),({c:k,key:G})=>G,(k,G,J)=>{let R=()=>s(G).c,Y=()=>s(G).key;var $=ti(),re=d($),ge=d(re);{let ee=q(()=>R().memories.length>a);Mt(ge,{get similarity(){return R().similarity},get memories(){return R().memories},get suggestedAction(){return R().suggestedAction},get oversized(){return s(ee)},onDismiss:()=>W(Y())})}c(re),c($),We($,(ee,oe)=>{var X;return(X=ct)==null?void 0:X(ee,oe)},()=>({delay:Math.min(s(J)*40,400),y:14})),We($,ee=>Ue==null?void 0:Ue(ee)),P(k,$)}),c(u),P(l,u)};Q(Re,l=>{s(h)?l(Ce):s(f)?l(L,1):s(B).length===0?l(_,2):l(H,!1)})}c(A),P(r,U),Qe()}$e(["input","click"]);export{_i as component};

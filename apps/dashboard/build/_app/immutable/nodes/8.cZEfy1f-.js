import{B as e,D as t,E as n,G as r,J as i,K as a,M as o,O as s,P as c,Q as l,S as u,T as d,X as f,Y as p,Z as m,a as h,at as g,d as _,f as v,g as y,h as b,j as x,lt as S,m as C,n as w,ot as T,p as E,q as D,r as O,u as k,ut as A,w as j,x as M}from"../chunks/DHlIFcq6.js";import"../chunks/xihTtKlq.js";import{n as N}from"../chunks/BGA2R-TK.js";import{t as P}from"../chunks/yeJIv7a5.js";import{t as ee}from"../chunks/CqiNFhrz.js";import{a as F,c as I,i as L,o as R,u as z}from"../chunks/CgAKUn-r.js";import{t as B}from"../chunks/DNa4FLDU.js";import{t as te}from"../chunks/gEL1IeNT.js";import{n as V,t as ne}from"../chunks/D8C6olcw.js";function H(e){return e>=.92?`near-identical`:e>=.8?`strong`:`weak`}function U(e){let t=H(e);return t===`near-identical`?`var(--color-decay)`:t===`strong`?`var(--color-warning)`:`#fde047`}function re(e){let t=H(e);return t===`near-identical`?`Near-identical`:t===`strong`?`Strong match`:`Weak match`}function ie(e){return e>.7?`#10b981`:e>.4?`#f59e0b`:`#ef4444`}function W(e){if(!e||e.length===0)return null;let t=e[0],n=Number.isFinite(t.retention)?t.retention:-1/0;for(let r=1;r<e.length;r++){let i=e[r],a=Number.isFinite(i.retention)?i.retention:-1/0;a>n&&(t=i,n=a)}return t}function G(e){return e.map(e=>e.id).slice().sort().join(`|`)}function ae(e,t=80){if(!e)return``;let n=e.trim().replace(/\s+/g,` `);return n.length<=t?n:n.slice(0,t)+`…`}function oe(e){if(!e||typeof e!=`string`)return``;let t=new Date(e);return Number.isNaN(t.getTime())?``:t.toLocaleDateString(void 0,{year:`numeric`,month:`short`,day:`numeric`})}function se(e,t=4){return Array.isArray(e)?e.slice(0,t):[]}function ce(e){let t=e.diff?.invalidatedIds?.length??Math.max(0,e.memberIds.length-1),n=typeof e.confidence==`number`?e.confidence:Number.parseFloat(e.confidence),r=Number.isFinite(n)?`${Math.round(n*100)}%`:`unknown`,i=t===1?`memory`:`memories`;return`Keeps ${e.survivorId.slice(0,8)}, folds ${t} ${i} into it. Matcher: ${e.classification} at ${r}. Reversible with dedup undo.`}var le=s(`<span class="flex-shrink-0 rounded-full border border-warning/50 bg-warning/10 px-3 py-1 text-xs font-medium text-warning">REVIEW REQUIRED · NOT SAFE TO MERGE</span>`),ue=s(`<span> </span>`),de=s(`<span class="rounded bg-recall/15 px-1.5 py-0.5 text-[10px] font-medium text-recall">WINNER</span>`),fe=s(`<span class="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-muted"> </span>`),pe=s(`<div class="text-[11px] text-muted"> </div>`),me=s(`<div><span class="mt-1.5 h-2 w-2 flex-shrink-0 rounded-full"></span> <div class="flex-1 min-w-0 space-y-1.5"><div class="flex flex-wrap items-center gap-1.5"><span class="text-xs text-dim"> </span> <!> <!></div> <p> </p> <!></div> <div class="flex flex-shrink-0 flex-col items-end gap-1"><div class="h-1.5 w-12 overflow-hidden rounded-full bg-deep"><div class="h-full rounded-full"></div></div> <span class="text-[11px] text-muted"> </span></div></div>`),he=s(`<div class="rounded-xl border border-warning/20 bg-warning/5 p-3 text-xs text-dim"> </div>`),ge=s(`<div class="rounded-xl border border-synapse/25 bg-synapse/5 p-3 text-xs"><div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">Merge preview · nothing written yet</div> <div class="mt-1 text-text"> </div> <div class="mt-1 text-muted"> </div> <div class="mt-2 max-h-24 overflow-hidden text-muted"> </div> <div class="mt-3 flex flex-wrap items-center gap-2"><button type="button" class="rounded-lg bg-synapse/25 px-3 py-1.5 text-xs font-medium text-synapse-glow transition hover:bg-synapse/35 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"> </button> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Cancel</button></div></div>`),_e=s(`<div class="rounded-xl border border-consolidated/25 bg-consolidated/5 p-3 text-xs text-text"> <span class="font-mono"> </span>.</div>`),ve=s(`<div class="rounded-xl border border-decay/25 bg-decay/5 p-3 text-xs text-decay" role="alert"> </div>`),ye=s(`<div class="glass-panel rounded-2xl p-5 space-y-4 transition-all duration-300 hover:border-synapse/20"><div class="flex items-start justify-between gap-4"><div class="flex-1 min-w-0 space-y-1.5"><div class="flex items-center gap-3"><span class="text-sm font-semibold"> </span> <span class="text-xs text-dim"> </span> <span class="text-xs text-muted"> </span></div> <div class="h-2 w-full overflow-hidden rounded-full bg-deep/60" role="progressbar" aria-label="Cosine similarity" aria-valuemin="0" aria-valuemax="100"><div class="h-full rounded-full transition-all duration-500"></div></div></div> <!></div> <div class="space-y-2"><!> <!></div> <!> <!> <!> <div class="flex flex-wrap items-center gap-2 pt-1"><button type="button"> </button> <button type="button" class="rounded-lg bg-dream/20 px-3 py-1.5 text-xs font-medium text-dream-glow transition hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream-glow/60"> </button> <button type="button" aria-label="Dismiss cluster for this session" class="ml-auto rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Dismiss cluster</button></div></div>`);function be(s,p){T(p,!0);let _=h(p,`oversized`,3,!1),y=m(null),x=m(!1),w=m(!1),O=m(null),k=m(null),P=l(()=>!_()&&!!p.onPlan&&!!p.onApply&&!c(k));async function ee(){if(p.onPlan&&!c(x)){f(x,!0),f(O,null);try{f(y,await p.onPlan(p.memories.map(e=>e.id)),!0)}catch(e){f(O,e instanceof Error?e.message:`Could not plan the merge`,!0)}finally{f(x,!1)}}}async function F(){if(p.onApply&&c(y)&&!c(w)){f(w,!0),f(O,null);try{f(k,await p.onApply(c(y).planId),!0),p.onMerged?.(c(k))}catch(e){f(O,e instanceof Error?e.message:`Could not apply the merge`,!0)}finally{f(w,!1)}}}let I=m(!1),L=l(()=>W(p.memories)),R=l(()=>{if(p.memories.length<=12)return p.memories;let e=p.memories.filter(e=>e.id!==c(L)?.id);return c(L)?[c(L),...e.slice(0,11)]:e.slice(0,12)}),z=l(()=>p.memories.length-c(R).length);var B=t(),te=a(B),V=t=>{var a=ye(),s=r(a),m=r(s),h=r(m),g=r(h),T=D(g),B=i(g,2),te=D(B,!0),V=i(B,2),ne=D(V);A(h);var H=i(h,2),W=D(H);A(m);var G=i(m,2),be=e=>{var t=le();n(e,t)},K=t=>{var r=ue(),i=D(r);e(()=>{C(r,1,`flex-shrink-0 rounded-full border px-3 py-1 text-xs font-medium ${p.suggestedAction===`merge`?`border-recall/40 bg-recall/10 text-recall`:`border-dream-glow/40 bg-dream/10 text-dream-glow`}`),d(i,`Classification: ${p.suggestedAction===`merge`?`merge candidate`:`review`}`)}),n(t,r)};j(G,e=>{_()?e(be):e(K,-1)}),A(s);var q=i(s,2),J=r(q);M(J,17,()=>c(R),e=>e.id,(t,a)=>{var o=me(),s=r(o),f=i(s,2),p=r(f),m=r(p),h=D(m,!0),g=i(m,2),_=e=>{var t=de();n(e,t)};j(g,e=>{c(a).id===c(L).id&&e(_)});var y=i(g,2);M(y,17,()=>se(c(a).tags,4),u,(t,r)=>{var i=fe(),a=D(i,!0);e(()=>d(a,c(r))),n(t,i)}),A(p);var b=i(p,2),x=D(b,!0),S=i(b,2),w=t=>{var r=pe(),i=D(r,!0);e(e=>d(i,e),[()=>oe(c(a).createdAt)]),n(t,r)},T=l(()=>oe(c(a).createdAt));j(S,e=>{c(T)&&e(w)}),A(f);var O=i(f,2),k=r(O),P=D(k),ee=i(k,2),F=D(ee);A(O),A(o),e((e,t,n)=>{C(o,1,`group flex items-start gap-3 rounded-xl border border-synapse/5 bg-white/[0.02] p-3 transition-all duration-200 hover:border-synapse/20 hover:bg-white/[0.04] ${c(a).id===c(L).id?`ring-1 ring-recall/30`:``}`),E(s,`background: ${(N[c(a).nodeType]||`#8B95A5`)??``}`),v(s,`title`,c(a).nodeType),d(h,c(a).nodeType),C(b,1,`text-sm text-text leading-relaxed ${c(I)?`whitespace-pre-wrap`:``}`),d(x,e),E(P,`width: ${c(a).retention*100}%; background: ${t??``}`),d(F,`${n??``}%`)},[()=>c(I)?c(a).content:ae(c(a).content),()=>ie(c(a).retention),()=>(c(a).retention*100).toFixed(0)]),n(t,o)});var Y=i(J,2),xe=t=>{var r=he(),i=D(r);e(()=>d(i,`+${c(z)??``} linked candidates — oversized similarity component. Members
					chain through pairwise similarity; distant members may be unrelated. Raise
					the threshold to split it.`)),n(t,r)};j(Y,e=>{c(z)>0&&e(xe)}),A(q);var Se=i(q,2),Ce=t=>{var a=ge(),s=i(r(a),2),l=D(s,!0),u=i(s,2),p=D(u,!0),m=i(u,2),h=D(m),g=i(m,2),_=r(g),v=D(_,!0),b=i(_,2);A(g),A(a),e((e,t)=>{d(l,e),d(p,c(y).explanation),d(h,`Result: ${t??``}`),_.disabled=c(w),d(v,c(w)?`Applying…`:`Apply merge`),b.disabled=c(w)},[()=>ce(c(y)),()=>ae(c(y).diff.resultContent,240)]),o(`click`,_,F),o(`click`,b,()=>f(y,null)),n(t,a)};j(Se,e=>{c(y)&&!c(k)&&e(Ce)});var we=i(Se,2),Te=t=>{var a=_e(),o=r(a),s=i(o),l=D(s,!0);S(),A(a),e(e=>{d(o,`Merged into ${e??``}. Reversible: run dedup undo with
				operation id `),d(l,c(k).operationId)},[()=>c(k).survivorId.slice(0,8)]),n(t,a)};j(we,e=>{c(k)&&e(Te)});var Ee=i(we,2),De=t=>{var r=ve(),i=D(r,!0);e(()=>d(i,c(O))),n(t,r)};j(Ee,e=>{c(O)&&e(De)});var X=i(Ee,2),Z=r(X),Q=D(Z,!0),$=i(Z,2),Oe=D($,!0),ke=i($,2);A(X),A(a),e((e,t,n,r,i,a,o)=>{E(g,`color: ${e??``}`),d(T,`${t??``}%`),d(te,n),d(ne,`· ${p.memories.length??``} memories`),v(H,`aria-valuenow`,r),E(W,`width: ${i??``}%; background: ${a??``}; box-shadow: 0 0 12px ${o??``}66`),Z.disabled=!c(P)||c(x)||!!c(y),v(Z,`aria-disabled`,!c(P)),v(Z,`aria-label`,_()?`Merge is not safe for an oversized component`:`Preview a reversible merge`),C(Z,1,b(c(P)?`rounded-lg bg-synapse/20 px-3 py-1.5 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60`:`cursor-not-allowed rounded-lg bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-muted/60`)),v(Z,`title`,_()?`Oversized similarity component: members chain through pairwise similarity, so a merge could fold unrelated memories together`:`Preview first; nothing is written until you apply`),d(Q,_()?`Merge unsafe here`:c(x)?`Planning…`:`Preview merge`),v($,`aria-expanded`,c(I)),d(Oe,c(I)?`Collapse`:`Review`)},[()=>U(p.similarity),()=>(p.similarity*100).toFixed(1),()=>re(p.similarity),()=>Math.round(p.similarity*100),()=>(p.similarity*100).toFixed(1),()=>U(p.similarity),()=>U(p.similarity)]),o(`click`,Z,ee),o(`click`,$,()=>f(I,!c(I))),o(`click`,ke,function(...e){p.onDismiss?.apply(this,e)}),n(t,a)};j(te,e=>{p.memories.length>0&&c(L)&&e(V)}),n(s,B),g()}x([`click`]);var K=`rgba16float`,q=512,J=512,Y=16,xe=16,Se=`
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
`,Ce=`
${Se}

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
`,we=`
${Se}

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
`,Te=`
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
`,Ee=class{engine;scene=null;resources=null;sampler=null;splatBindLayout=null;blurBindLayout=null;membraneBindLayout=null;splatPipeline=null;blurPipeline=null;membranePipeline=null;cellPipeline=null;neckPipeline=null;cellCount=0;neckCount=0;cellGeometry=[];neckGeometry=[];intensity=.22;well={x:0,y:0,hw:-1,hh:0,floor:.1,soft:.22};constructor(e,t){this.engine=e,this.uploadScene(t)}setIntensity(e){this.intensity=Math.min(1,Math.max(0,Number.isFinite(e)?e:.22));let t=this.engine.gpuDevice;t&&this.writeOpts(t)}setReadingWell(e){let t=(e,t=0)=>Number.isFinite(e)?e:t;this.well={x:t(e.x),y:t(e.y),hw:t(e.hw,-1),hh:t(e.hh),floor:Math.min(1,Math.max(0,t(e.floor??.1,.1))),soft:Math.max(.02,t(e.soft??.22,.22))};let n=this.engine.gpuDevice;n&&this.writeOpts(n)}writeOpts(e){this.resources&&e.queue.writeBuffer(this.resources.optsBuffer,0,new Float32Array([this.intensity,this.well.x,this.well.y,this.well.hw,this.well.hh,this.well.floor,this.well.soft,0]))}uploadScene(e){this.scene=e,this.buildGeometry();let t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.uploadBuffers(t))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;let t=X(e,`duplicates-fusion-splat-wgsl`,Ce),n=X(e,`duplicates-fusion-blur-wgsl`,Te),r=X(e,`duplicates-fusion-membrane-wgsl`,we);this.splatBindLayout=e.createBindGroupLayout({label:`duplicates-fusion-splat-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]}),this.blurBindLayout=e.createBindGroupLayout({label:`duplicates-fusion-blur-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:`duplicates-fusion-membrane-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let i=e.createPipelineLayout({label:`duplicates-fusion-splat-layout`,bindGroupLayouts:[this.splatBindLayout]}),a=e.createPipelineLayout({label:`duplicates-fusion-blur-layout`,bindGroupLayouts:[this.blurBindLayout]}),o=e.createPipelineLayout({label:`duplicates-fusion-membrane-layout`,bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:`linear`,minFilter:`linear`});let s={color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}};this.splatPipeline=e.createRenderPipeline({label:`duplicates-field-additive-splat`,layout:i,vertex:{module:t,entryPoint:`vs_splat`},fragment:{module:t,entryPoint:`fs_splat`,targets:[{format:K,blend:s}]},primitive:{topology:`triangle-list`}}),this.blurPipeline=e.createRenderPipeline({label:`duplicates-field-blur-render-pass`,layout:a,vertex:{module:n,entryPoint:`vs_fullscreen`},fragment:{module:n,entryPoint:`fs_blur`,targets:[{format:K}]},primitive:{topology:`triangle-list`}}),this.membranePipeline=e.createRenderPipeline({label:`duplicates-synaptic-fusion-membrane`,layout:o,vertex:{module:r,entryPoint:`vs_fullscreen`},fragment:{module:r,entryPoint:`fs_membrane`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-list`}}),this.cellPipeline=e.createRenderPipeline({label:`duplicates-memory-nuclei`,layout:i,vertex:{module:t,entryPoint:`vs_cell`},fragment:{module:t,entryPoint:`fs_cell`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-list`}}),this.neckPipeline=e.createRenderPipeline({label:`duplicates-mismatch-filaments`,layout:i,vertex:{module:t,entryPoint:`vs_neck`},fragment:{module:t,entryPoint:`fs_neck`,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:`triangle-strip`}})}ensureResources(e){if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;let t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),n=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),r=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==n,i=this.resources?.cellBuffer,a=this.resources?.neckBuffer,o=this.resources?.blurHBuffer,s=this.resources?.blurVBuffer,c=this.resources?.optsBuffer;if(i||=e.createBuffer({label:`duplicates-cells`,size:q*Y*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),a||=e.createBuffer({label:`duplicates-necks`,size:J*xe*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),o||(o=e.createBuffer({label:`duplicates-blur-h-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(o,0,new Float32Array([1,0,0,0]))),s||(s=e.createBuffer({label:`duplicates-blur-v-dir`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([0,1,0,0]))),c||=e.createBuffer({label:`duplicates-field-opts`,size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),!r&&this.resources){this.resources.optsBuffer=c,this.writeOpts(e);return}this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy();let l=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,u=e.createTexture({label:`duplicates-field-a-rgba16float`,size:[t,n],format:K,usage:l}),d=e.createTexture({label:`duplicates-field-b-rgba16float`,size:[t,n],format:K,usage:l}),f=u.createView(),p=d.createView(),m=e.createBindGroup({label:`duplicates-fusion-splat-bind`,layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:a}},{binding:5,resource:{buffer:c}}]}),h=e.createBindGroup({label:`duplicates-field-blur-h-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:f},{binding:2,resource:{buffer:o}}]}),g=e.createBindGroup({label:`duplicates-field-blur-v-bind`,layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:p},{binding:2,resource:{buffer:s}}]}),_=e.createBindGroup({label:`duplicates-membrane-bind`,layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:3,resource:this.sampler},{binding:4,resource:f},{binding:5,resource:{buffer:c}}]});this.resources={cellBuffer:i,neckBuffer:a,blurHBuffer:o,blurVBuffer:s,optsBuffer:c,splatBindGroup:m,blurHBindGroup:h,blurVBindGroup:g,membraneBindGroup:_,fieldA:u,fieldB:d,fieldAView:f,fieldBView:p,fieldSize:[t,n]},this.writeOpts(e)}buildGeometry(){let e=this.scene?.clusters??[],t=Math.max(1,e.length),n=[],r=[],i=Array(e.length).fill(0),a=q;for(let t=0;t<e.length&&a>0;t++)i[t]=1,--a;for(let t=0;t<e.length&&a>0;t++)i[t]===1&&e[t].memories.length>=2&&(i[t]=2,--a);let o=!0;for(;a>0&&o;){o=!1;for(let t=0;t<e.length&&a>0;t++)i[t]>0&&i[t]<Math.min(e[t].memories.length,12)&&(i[t]+=1,--a,o=!0)}for(let a=0;a<e.length;a++){let o=e[a];if(i[a]===0)continue;let s=a/t*Math.PI*2-Math.PI/2,c=.18+.58*Math.sqrt((a+.5)/t),l=Math.cos(s)*c*.86,u=Math.sin(s)*c,d=Math.max(.04,.25-Math.max(0,o.similarity-o.threshold)*.55),f=o.memories.find(e=>e.id===o.winnerId)??o.memories[0],p=[f,...o.memories.filter(e=>e.id!==f.id)].slice(0,i[a]),m=Math.max(1,p.length),h=new Map;for(let e=0;e<p.length&&n.length<q;e++){let t=p[e],r=s+e/m*Math.PI*2+(m%2?0:Math.PI/m),i=t.id===o.winnerId,a=i?d*.18:d+e%3*.025,c=Math.min(1,(t.mismatchTokens?.length??0)/8),f=.085+Math.min(.045,I(t.retention)*2.1)+(i?.012:0),g={cluster:o,memoryId:t.id,x:l+Math.cos(r)*a,y:u+Math.sin(r)*a,retention:Math.max(0,Math.min(1,t.retention||0)),winner:i,mismatch:c,radius:f,memberSlot:e,memberCount:m};h.set(t.id,g),n.push(g)}let g=h.get(f.id);if(g)for(let e of o.memories){if(r.length>=J||e.id===f.id)continue;let t=h.get(e.id);t&&r.push({cluster:o,winnerId:f.id,candidateId:e.id,ax:g.x,ay:g.y,bx:t.x,by:t.y,winnerRetention:g.retention,candidateRetention:t.retention,winnerRadius:g.radius,candidateRadius:t.radius,mismatch:Math.max(t.mismatch,Math.min(1,o.mismatchTokens.length/12))})}}this.cellGeometry=n,this.neckGeometry=r}uploadBuffers(e){if(!this.resources)return;let t=new Float32Array(q*Y),n=new Float32Array(J*xe);this.cellCount=Math.min(q,this.cellGeometry.length),this.neckCount=Math.min(J,this.neckGeometry.length);for(let e=0;e<this.cellCount;e++){let n=this.cellGeometry[e];t.set([n.x,n.y,n.retention,+!!n.winner,n.cluster.similarity,n.cluster.threshold,n.memberSlot,n.cluster.index,n.mismatch,+(n.cluster.suggestedAction===`merge`),n.radius,n.memberCount,e,n.cluster.index,0,0],e*Y)}for(let e=0;e<this.neckCount;e++){let t=this.neckGeometry[e];n.set([t.ax,t.ay,t.winnerRetention,t.winnerRadius,t.bx,t.by,t.candidateRetention,t.candidateRadius,t.cluster.similarity,t.cluster.threshold,t.mismatch,+(t.cluster.suggestedAction===`merge`),e,t.cluster.index,0,0],e*xe)}this.engine.params[2]=this.cellCount,this.engine.params[3]=this.neckCount,this.engine.params[4]=this.neckCount,e.queue.writeBuffer(this.resources.cellBuffer,0,t),e.queue.writeBuffer(this.resources.neckBuffer,0,n)}compute(e){let t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);let n=this.resources,r=e.beginRenderPass({label:`duplicates-field-splat-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.splatPipeline),r.setBindGroup(0,n.splatBindGroup),this.cellCount+this.neckCount>0&&r.draw(6,this.cellCount+this.neckCount),r.end();let i=e.beginRenderPass({label:`duplicates-field-blur-h-pass`,colorAttachments:[{view:n.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});i.setPipeline(this.blurPipeline),i.setBindGroup(0,n.blurHBindGroup),i.draw(6,1),i.end();let a=e.beginRenderPass({label:`duplicates-field-blur-v-pass`,colorAttachments:[{view:n.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});a.setPipeline(this.blurPipeline),a.setBindGroup(0,n.blurVBindGroup),a.draw(6,1),a.end()}render(e){this.resources&&this.membranePipeline&&this.cellPipeline&&this.neckPipeline&&(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.neckCount>0&&(e.setPipeline(this.neckPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(64,this.neckCount)),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.splatBindGroup),e.draw(6,this.cellCount)))}pickAt(e,t){for(let n=0;n<this.neckGeometry.length;n++){let r=this.neckGeometry[n],i=De(e,t,r.ax,r.ay,r.bx,r.by),a=(r.ax+r.bx)*.5,o=(r.ay+r.by)*.5,s=.055+Math.max(0,r.cluster.similarity-r.cluster.threshold)*.45;if(i<=s||Math.hypot(e-a,t-o)<=s)return{id:r.cluster.id,kind:`duplicate-neck`,index:n,payload:r.cluster}}for(let n=0;n<this.cellGeometry.length;n++){let r=this.cellGeometry[n];if(Math.hypot(e-r.x,t-r.y)<=r.radius*.8)return{id:r.memoryId,kind:`duplicate-memory`,index:n,payload:r.cluster}}return null}dispose(){this.resources?.cellBuffer.destroy(),this.resources?.neckBuffer.destroy(),this.resources?.blurHBuffer.destroy(),this.resources?.blurVBuffer.destroy(),this.resources?.optsBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}};function De(e,t,n,r,i,a){let o=i-n,s=a-r,c=e-n,l=t-r,u=o*c+s*l;if(u<=0)return Math.hypot(e-n,t-r);let d=o*o+s*s;if(d<=u)return Math.hypot(e-i,t-a);let f=u/d;return Math.hypot(e-(n+f*o),t-(r+f*s))}function X(e,t,n){e.pushErrorScope(`validation`);let r=e.createShaderModule({label:t,code:n});return r.getCompilationInfo().then(e=>{for(let n of e.messages)console.error(`[observatory] ${t} WGSL ${n.type} ${n.lineNum}:${n.linePos} ${n.message}`)}),e.popErrorScope().then(e=>{e&&console.error(`[observatory] ${t} shader module validation: ${e.message}`)}),r}function Z(e,t){z(F.blackwater),z(R.recall),z(R.luciferin),z(L.trustMembrane);let n=new Ee(e,t);return n.setIntensity(.22),n.setReadingWell({x:0,y:0,hw:.6,hh:.85,floor:.08,soft:.25}),[n]}function Q(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function $(e,t,n){return n?{kind:e,id:t,scalar:n}:{kind:e,id:t||`${e}:unknown`}}function Oe(e,t){return{kind:`scalar`,id:`duplicates.${e}`,scalar:{name:e,value:t}}}function ke(e,t=84){let n=(e||``).trim().replace(/\s+/g,` `);return n.length<=t?n:`${n.slice(0,t)}…`}function Ae(e){return(e||``).toLowerCase().replace(/[^a-z0-9_\s-]/g,` `).split(/\s+/).filter(e=>e.length>=4).slice(0,80)}function je(e){if(e.length<2)return[];let t=e.map(e=>new Set(Ae(e.content))),n=new Map;for(let e of t)for(let t of e)n.set(t,(n.get(t)??0)+1);return Array.from(n.entries()).filter(([,t])=>t>0&&t<e.length).sort((e,t)=>t[1]-e[1]||e[0].localeCompare(t[0])).slice(0,12).map(([e])=>e)}function Me(e,t,n){let r=Array.isArray(e.memories)?e.memories.filter(e=>e.id):[];if(r.length<2)return null;let i=G(r),a=W(r),o=je(r);return{id:i,index:t,similarity:Q(e.similarity),threshold:Q(n),suggestedAction:e.suggestedAction===`merge`?`merge`:`review`,winnerId:a?.id??r[0].id,memories:r.map((e,t)=>({...e,index:t,preview:ke(e.content),winner:e.id===(a?.id??r[0].id),mismatchTokens:o.filter(t=>Ae(e.content).includes(t)).slice(0,8)})),mismatchTokens:o,source:$(`pair`,i)}}function Ne(e){let t=Q(e.threshold??.8),n=(Array.isArray(e.clusters)?e.clusters:[]).map((e,n)=>Me(e,n,t)).filter(e=>e!==null),r=0,i=[],a=new Map;for(let e of n)for(let t of e.memories){if(a.has(t.id))continue;let n=r++;a.set(t.id,n),i.push({source:$(`memory`,t.id),index:n,label:t.preview||t.id.slice(0,8),retention:Q(t.retention),trust:Q(e.similarity),lastAccessed:t.createdAt,tags:[t.nodeType,...t.tags,t.winner?`winner`:`candidate`].filter(Boolean),type:t.nodeType||`memory`})}let o=[];for(let e of n){let t=a.get(e.winnerId);if(t!=null)for(let n of e.memories){let r=a.get(n.id);r!=null&&r!==t&&o.push({source:$(`pair`,`${e.id}:${e.winnerId}:${n.id}`),sourceIndex:t,targetIndex:r,weight:Math.max(.05,e.similarity),kind:e.suggestedAction===`merge`?`fusion-candidate`:`review-candidate`})}}let s=n.map((e,n)=>({source:$(`event`,`duplicates.cluster.${e.id}`),type:e.suggestedAction===`merge`?`DuplicateMergeCandidate`:`DuplicateReviewCandidate`,targetIndex:-1,frame:20+n*14,energy:Math.max(.1,e.similarity-t+.1)})),c=Number.isFinite(e.total)?e.total:n.length,l=i.length,u=n.reduce((e,t)=>Math.max(e,t.similarity),0),d=n.filter(e=>e.suggestedAction===`merge`).length,f=n.length-d;return{organ:`duplicates`,nodes:i,edges:o,events:s,receipts:[],scalars:{threshold:Oe(`threshold`,t).scalar?.value??t,clusterCount:n.length,memoryCount:l,maxSimilarity:u,mergeCandidates:d,reviewCandidates:f,total:c},alive:n.length>0,threshold:t,total:c,clusters:n,raw:e}}var Pe=()=>typeof window<`u`&&window.matchMedia?.(`(prefers-reduced-motion: reduce)`).matches;function Fe(e){if(Pe())return{};let t=0;function n(n){let r=e.getBoundingClientRect();cancelAnimationFrame(t),t=requestAnimationFrame(()=>{e.style.setProperty(`--spot-x`,`${n.clientX-r.left}px`),e.style.setProperty(`--spot-y`,`${n.clientY-r.top}px`),e.style.setProperty(`--spot-o`,`1`)})}function r(){e.style.setProperty(`--spot-o`,`0`)}return e.addEventListener(`pointermove`,n),e.addEventListener(`pointerleave`,r),{destroy(){e.removeEventListener(`pointermove`,n),e.removeEventListener(`pointerleave`,r),cancelAnimationFrame(t)}}}var Ie=s(`<span class="ping-host flex h-2 w-2 items-center justify-center text-synapse-glow" aria-hidden="true"><span class="breathe h-2 w-2 rounded-full bg-synapse-glow"></span></span>`),Le=s(`<!> <span class="text-xs text-dim"> </span>`,1),Re=s(`<label class="flex w-full flex-col gap-2 text-xs text-dim"><span class="flex items-baseline justify-between gap-3"><span class="whitespace-nowrap">Similarity threshold</span> <span class="font-mono text-sm text-bright"> </span></span> <input type="range" min="0.70" max="0.95" step="0.01" class="w-full accent-synapse" aria-label="Similarity threshold"/></label>`),ze=s(`<label class="flex flex-1 min-w-64 items-center gap-3 text-xs text-dim"><span class="whitespace-nowrap">Similarity threshold</span> <input type="range" min="0.70" max="0.95" step="0.01" class="flex-1 accent-synapse" aria-label="Similarity threshold"/> <span class="w-14 text-right font-mono text-sm text-bright"> </span></label>`),Be=s(`<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span>Detecting…</span>`,1),Ve=s(`<span class="h-2 w-2 rounded-full bg-decay"></span> <span class="text-decay">Error</span>`,1),He=s(`<!> `,1),Ue=s(`<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span> <span class="tabular-nums"><!> · <!> memories implicated</span>`,1),We=s(`<div class="flex items-center gap-2 rounded-full border border-synapse/20 bg-synapse/10 px-3 py-1.5 text-xs text-text" role="status" aria-live="polite"><!></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Rerun</button>`,1),Ge=s(`<div class="glass-panel pointer-events-auto rounded-2xl border border-synapse/25 bg-black/30 p-4"><div class="flex flex-wrap items-center justify-between gap-3"><div><div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">Synaptic neck selected</div> <div class="mt-1 text-sm text-bright"> </div> <div class="mt-1 max-w-2xl text-xs text-muted"> </div></div> <button type="button" class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Clear field focus</button></div></div>`),Ke=s(`<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"><div class="text-sm text-decay">Couldn't detect duplicates</div> <div class="max-w-md text-xs text-muted"> </div> <button type="button" class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">Retry</button></div>`),qe=s(`<div class="glass-subtle shimmer h-40 rounded-2xl"></div>`),Je=s(`<div class="pointer-events-auto space-y-3"></div>`),Ye=s(`<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"><div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"><!></div> <div class="text-sm font-medium text-bright">No duplicates found — your memory is clean.</div> <div class="max-w-sm text-xs text-muted"> </div></div>`),Xe=s(`<div class="glass-subtle rounded-xl border border-warning/30 bg-warning/5 px-4 py-2 text-xs text-dim"> </div>`),Ze=s(`<div class="spotlight-surface lift rounded-2xl"><div class="relative z-[1]"><!></div></div>`),Qe=s(`<div class="pointer-events-auto space-y-4"><!> <!></div>`),$e=s(`<!> <div class="relative z-10 mx-auto max-h-dvh max-w-5xl space-y-6 overflow-y-auto overscroll-contain p-6 pb-28 pointer-events-none"><!> <div class="glass-panel pointer-events-auto flex flex-wrap items-center gap-5 rounded-2xl p-4"><!> <!></div> <!> <!></div>`,1);function et(t,s){T(s,!0);let h=m(.8),v=m(p([])),b=m(0),x=m(p(new Set)),C=m(!0),E=m(null),N=m(null),F,I=m(!1);O(()=>{let e=()=>{f(I,window.innerWidth/Math.max(1,window.innerHeight)<.85)};return e(),window.addEventListener(`resize`,e),()=>window.removeEventListener(`resize`,e)});async function L(){f(C,!0),f(E,null),f(N,null);try{let e=await P.duplicates(c(h));f(v,e.clusters,!0),f(b,e.total??e.clusters.length,!0);let t=new Set(c(v).map(e=>G(e.memories))),n=new Set;for(let e of c(x))t.has(e)&&n.add(e);f(x,n,!0)}catch(e){f(E,e instanceof Error?e.message:`Failed to detect duplicates`,!0),f(v,[],!0)}finally{f(C,!1)}}function R(){clearTimeout(F),F=setTimeout(L,250)}function z(e){let t=new Set(c(x));t.add(e),f(x,t,!0),c(N)&&G(c(N).memories)===e&&f(N,null)}function H(e){z(e),L()}let U=l(()=>c(v).map(e=>({c:e,key:G(e.memories)})).filter(({key:e})=>!c(x).has(e))),re=l(()=>c(v).reduce((e,t)=>e+t.memories.length,0)),ie=l(()=>c(U).length>50),W=l(()=>c(ie)?c(U).slice(0,50):c(U)),ae=l(()=>Ne({threshold:c(h),total:c(U).length,clusters:c(U).map(({c:e})=>e)}));function oe(e){(e.kind===`duplicate-neck`||e.kind===`duplicate-memory`)&&f(N,e.payload,!0)}O(()=>L()),w(()=>clearTimeout(F));var se=$e(),ce=a(se);{let e=l(()=>`synaptic-fusion:${c(h)}:${c(U).length}:${c(re)}`),t=l(()=>`NO DUPLICATES ABOVE ${(c(h)*100).toFixed(0)}% SIMILARITY`);B(ce,{organ:`duplicates`,get seed(){return c(e)},get scene(){return c(ae)},get passes(){return Z},get loading(){return c(C)},get error(){return c(E)},get emptyLabel(){return c(t)},onpick:oe})}var le=i(ce,2),ue=r(le);te(ue,{icon:`duplicates`,title:`Memory Hygiene: Duplicate Detection`,subtitle:`Exact duplicates: identical content hash or identical declared source, found by the log, never by resemblance. Nothing merges on a Strata log; retire an old version with memory edit. Dismissed clusters are hidden for this session only.`,accent:`synapse`,children:(t,r)=>{var o=Le(),s=a(o),l=e=>{var t=Ie();n(e,t)};j(s,e=>{c(E)||e(l)});var u=i(s,2),f=D(u,!0);e(()=>d(f,c(E)?`Offline`:c(C)?`Refreshing`:`Live`)),n(t,o)},$$slots:{default:!0}});var de=i(ue,2),fe=r(de),pe=t=>{var a=Re(),s=r(a),l=i(r(s),2),u=D(l);A(s);var p=i(s,2);_(p),A(a),e(e=>d(u,`${e??``}%`),[()=>(c(h)*100).toFixed(0)]),o(`input`,p,R),k(p,()=>c(h),e=>f(h,e)),n(t,a)},me=t=>{var a=ze(),s=i(r(a),2);_(s);var l=i(s,2),u=D(l);A(a),e(e=>d(u,`${e??``}%`),[()=>(c(h)*100).toFixed(0)]),o(`input`,s,R),k(s,()=>c(h),e=>f(h,e)),n(t,a)};j(fe,e=>{c(I)?e(pe):e(me,-1)});var he=i(fe,2),ge=t=>{var s=We(),l=a(s),u=r(l),f=e=>{var t=Be();S(2),n(e,t)},p=e=>{var t=Ve();S(2),n(e,t)},m=t=>{var o=Ue(),s=i(a(o),2),l=r(s),u=t=>{var r=He(),o=a(r);V(o,{get value(){return c(U).length}});var s=i(o);e(()=>d(s,` visible of ${c(b)??``} clusters`)),n(t,r)},f=t=>{var r=He(),o=a(r);V(o,{get value(){return c(U).length}});var s=i(o);e(()=>d(s,` ${c(U).length===1?`cluster`:`clusters`}`)),n(t,r)};j(l,e=>{c(U).length<c(b)?e(u):e(f,-1)});var p=i(l,2);V(p,{get value(){return c(re)}}),S(),A(s),n(t,o)};j(u,e=>{c(C)?e(f):c(E)?e(p,1):e(m,-1)}),A(l);var h=i(l,2);e(()=>h.disabled=c(C)),o(`click`,h,L),n(t,s)};j(he,e=>{c(E)&&c(I)||e(ge)}),A(de);var _e=i(de,2),ve=t=>{var a=Ge(),s=r(a),l=r(s),u=i(r(l),2),p=D(u),m=i(u,2),h=D(m);A(l);var g=i(l,2);A(s),A(a),e((e,t,n)=>{d(p,`${c(N).memories.length??``} memories · ${e??``}% similar · winner ${t??``}`),d(h,`Real pair key: ${c(N).id??``}. Mismatch filaments: ${n??``}.`)},[()=>(c(N).similarity*100).toFixed(1),()=>c(N).winnerId.slice(0,8),()=>c(N).mismatchTokens.length?c(N).mismatchTokens.join(`, `):`none exposed`]),o(`click`,g,()=>f(N,null)),n(t,a)};j(_e,e=>{c(N)&&e(ve)});var ye=i(_e,2),K=t=>{var a=Ke(),s=i(r(a),2),l=D(s,!0),u=i(s,2);A(a),e(()=>d(l,c(E))),o(`click`,u,L),n(t,a)},q=e=>{var t=Je();M(t,20,()=>[,,,],u,(e,t)=>{var r=qe();n(e,r)}),A(t),n(e,t)},J=t=>{var a=Ye(),o=r(a),s=r(o);ee(s,{name:`sparkle`,size:26,draw:!0}),A(o);var l=i(o,4),u=D(l);A(a),e(e=>d(u,`Nothing clusters above ${e??``}% similarity. Lower the threshold to
				surface looser matches.`),[()=>(c(h)*100).toFixed(0)]),n(t,a)},Y=t=>{var a=Qe(),o=r(a),s=t=>{var r=Xe(),i=D(r);e(()=>d(i,`Showing first 50 of ${c(U).length??``} clusters. Raise the
					threshold to narrow results.`)),n(t,r)};j(o,e=>{c(ie)&&e(s)});var u=i(o,2);M(u,19,()=>c(W),({c:e,key:t})=>t,(e,t,i)=>{let a=()=>c(t).c,o=()=>c(t).key;var s=Ze(),u=r(s),d=r(u);{let e=l(()=>a().memories.length>12);be(d,{get similarity(){return a().similarity},get memories(){return a().memories},get suggestedAction(){return a().suggestedAction},get oversized(){return c(e)},onDismiss:()=>z(o()),onPlan:e=>P.duplicatesPlan(e),onApply:e=>P.duplicatesApply(e),onMerged:()=>H(o())})}A(u),A(s),y(s,(e,t)=>ne?.(e,t),()=>({delay:Math.min(c(i)*40,400),y:14})),y(s,e=>Fe?.(e)),n(e,s)}),A(a),n(t,a)};j(ye,e=>{c(E)?e(K):c(C)?e(q,1):c(U).length===0?e(J,2):e(Y,-1)}),A(le),n(t,se),g()}x([`input`,`click`]);export{et as component};
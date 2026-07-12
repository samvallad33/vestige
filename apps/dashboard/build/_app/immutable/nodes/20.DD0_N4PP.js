var Te=Object.defineProperty;var Ie=(t,e,s)=>e in t?Te(t,e,{enumerable:!0,configurable:!0,writable:!0,value:s}):t[e]=s;var R=(t,e,s)=>Ie(t,typeof e!="symbol"?e+"":e,s);import"../chunks/Bzak7iHL.js";import{o as $e,e as Ne,s as be}from"../chunks/DAau0uzT.js";import{p as Pe,j as Me,g as v,aH as Le,Y as de,a as ue,b as Fe,d as W,h as Be,X as q,s as N,f as ye,u as _e,$ as De,c as pe,r as se,af as Ge}from"../chunks/CGq8RnJq.js";import{b as Ue,i as Oe}from"../chunks/Ccqjq5DS.js";import{e as Xe,r as We,i as Ye}from"../chunks/DqfV0sZu.js";import{h as Ve}from"../chunks/De_e6MzK.js";import{b as ze}from"../chunks/DGM4cicq.js";import{a as we}from"../chunks/D35IQVqe.js";import{R as He}from"../chunks/BpEKQwpr.js";import{T as qe}from"../chunks/D7ozXiSB.js";import{r as Ke}from"../chunks/BMB5u1EX.js";import{L as je,F as fe,l as Ee}from"../chunks/DETSv_kY.js";const Qe={intent:"Intent classification",retrieve:"Memory retrieval",activate:"Activation expansion",evidence:"Evidence grounding",contradiction:"Contradiction check",synthesis:"Synthesis",recommendation:"Recommendation",receipt:"Composition receipt"},Je=["intent","retrieve","activate","evidence","contradiction","synthesis","recommendation","receipt"];function F(t){return t&&typeof t=="object"&&!Array.isArray(t)?t:{}}function y(t,e=""){return typeof t=="string"?t:t==null?e:String(t)}function J(t,e=0){return typeof t=="number"&&Number.isFinite(t)?t:e}function V(t){return Math.max(0,Math.min(1,t))}function Z(t){const e=J(t,0);return V(e>1?e/100:e)}function Ze(t){return t==="primary"||t==="supporting"||t==="contradicting"||t==="superseded"?t:"supporting"}function K(t,e=""){const s=Z(t.trust??t.trust_score??0);return{id:y(t.id??t.memory_id??e),trust:s,date:y(t.date??t.created_at??""),role:Ze(t.role),preview:y(t.preview??t.answer_preview??t.content??""),nodeType:t.node_type?y(t.node_type):t.nodeType?y(t.nodeType):void 0}}function X(t,e,s){return s?{kind:t,id:e,scalar:s}:{kind:t,id:e||`${t}:unknown`}}function P(t,e){return{kind:"scalar",id:`deep_reference.${t}`,scalar:{name:t,value:e}}}function et(t){var f,m,g,w;const e=F(t),u=(Array.isArray(e.evidence)?e.evidence.map(F):[]).map(n=>K(n)).filter(n=>n.id.length>0),h=F(e.recommended),i=Object.keys(h).length>0||u.length>0?{answer_preview:y(h.answer_preview??((f=u[0])==null?void 0:f.preview)??""),memory_id:y(h.memory_id??((m=u[0])==null?void 0:m.id)??""),trust_score:Z(h.trust_score??((g=u[0])==null?void 0:g.trust)??0),date:y(h.date??((w=u[0])==null?void 0:w.date)??"")}:null,a=(Array.isArray(e.contradictions)?e.contradictions.map(F):Array.isArray(e.claim_conflicts)?e.claim_conflicts.map(F):[]).map(n=>{const A=F(n.stronger),L=F(n.weaker);if(Object.keys(A).length>0||Object.keys(L).length>0){const ne=K(A),re=K(L);return{stronger:ne,weaker:re,topic_overlap:V(J(n.topic_overlap,0)),summary:y(n.summary,`Trust-weighted conflict: ${ne.id.slice(0,8)} over ${re.id.slice(0,8)}`)}}const O=K({id:n.a_id,role:"contradicting"}),le=K({id:n.b_id,role:"contradicting"});return{stronger:O,weaker:le,topic_overlap:V(J(n.topic_overlap,0)),summary:y(n.summary??n.reason,"Trust-weighted conflict between high-FSRS memories.")}}).filter(n=>n.stronger.id||n.weaker.id),E=(Array.isArray(e.superseded)?e.superseded.map(F):[]).map(n=>n.id||n.superseded_by?{id:y(n.id),preview:y(n.preview??""),trust:Z(n.trust??0),date:y(n.date??""),superseded_by:y(n.superseded_by??(i==null?void 0:i.memory_id)??""),reason:y(n.reason??"Superseded by newer memory with higher trust.")}:{id:y(n.old_id),preview:y(n.preview??""),trust:Z(n.trust??0),date:y(n.date??""),superseded_by:y(n.new_id??(i==null?void 0:i.memory_id)??""),reason:y(n.reason??"Superseded by newer memory with higher trust.")}).filter(n=>n.id||n.superseded_by),c=Z(e.confidence),l=J(e.memoriesAnalyzed??e.memories_analyzed,u.length),r=J(e.activationExpanded??e.activation_expanded,0),p=y(e.intent??""),b=y(e.reasoning??e.guidance??""),k=y(e.guidance??""),x=y(e.composition_event_id??""),S=y(e.compositionWriteStatus??e.composition_write_status??""),C=x||S,T=x.length>0,M=u.map((n,A)=>({source:X("memory",n.id),index:A,label:n.preview||n.id.slice(0,8),retention:V(n.trust),trust:V(n.trust),lastAccessed:n.date||void 0,tags:[n.role,...n.nodeType?[n.nodeType]:[]],type:n.nodeType??"memory"})),G=new Map(M.map(n=>[n.source.id,n.index])),ee=a.flatMap((n,A)=>{const L=G.get(n.stronger.id),O=G.get(n.weaker.id);return L==null||O==null?[]:[{source:X("pair",`contradiction:${n.stronger.id}:${n.weaker.id}`),sourceIndex:L,targetIndex:O,weight:Math.max(.2,n.topic_overlap||.5),kind:"contradiction"}]}),U=[];a.length>0&&U.push({source:X("event",`deep_reference.contradictions.${a.length}`),type:"ReasoningContradictionInterrupt",targetIndex:-1,frame:250,energy:a.length}),E.length>0&&U.push({source:X("event",`deep_reference.superseded.${E.length}`),type:"ReasoningSupersessionInterrupt",targetIndex:-1,frame:330,energy:E.length});const te=[];C&&te.push({source:T?X("receipt",x):P("compositionWriteStatus",S==="persisted"?1:0),label:T?`receipt ${x.slice(0,8)}`:S,nodeIndices:M.map(n=>n.index)});const $=(n,A,L,O,le,ne,re,Ce="none")=>({index:n,kind:A,label:Qe[A],count:L,confidence:V(O),lit:L>0||O>0,provenance:ne,exposed:le,not_exposed_by_backend:re,interrupt:Ce}),ce=Je.map((n,A)=>{switch(n){case"intent":return $(A,n,p?1:0,p?c:0,{intent:p},P("intent_present",p?1:0),["raw classifier trace"]);case"retrieve":return $(A,n,l||u.length,c,{memoriesAnalyzed:l,evidence_count:u.length},P("memoriesAnalyzed",l),["candidate ids discarded before evidence"]);case"activate":return $(A,n,r,r>0?c:0,{activationExpanded:r},P("activationExpanded",r),["activation path/map"]);case"evidence":return $(A,n,u.length,u.length>0?c:0,{evidence:u},P("evidence_count",u.length),["reranker discarded candidates"]);case"contradiction":return $(A,n,a.length,a.length>0?c:0,{contradictions:a,claim_conflicts:e.claim_conflicts??[]},P("contradiction_count",a.length),["full claim graph"],a.length>0?"contradiction":"none");case"synthesis":return $(A,n,b||k?1:0,b||k?c:0,{reasoning:b,guidance:k},P("synthesis_present",b||k?1:0),["token-level chain internals"]);case"recommendation":return $(A,n,i!=null&&i.memory_id?1:0,(i==null?void 0:i.trust_score)??0,{recommended:i},i!=null&&i.memory_id?X("memory",i.memory_id):P("recommended_present",0),["alternative recommendations"]);case"receipt":return $(A,n,C?1:0,C?c:0,{composition_event_id:x,compositionWriteStatus:S},T?X("receipt",x):P("compositionWriteStatus",S==="persisted"?1:0),["separate receipt field"],E.length>0?"supersession":"none")}});return{organ:"reasoning",nodes:M,edges:ee,events:U,receipts:te,scalars:{confidence:c,memoriesAnalyzed:l,activationExpanded:r,evidenceCount:u.length,contradictionCount:a.length,supersededCount:E.length,compositionPersisted:T?1:0},alive:!!(p||b||k||u.length||a.length||E.length||i!=null&&i.memory_id||C),stages:ce,evidence:u,contradictions:a,superseded:E,recommended:i,raw:e}}const z=[-.86,-.526,-.214,.074,.335,.562,.747,.86],tt=["INTENT","RETRIEVE","ACTIVATE","EVIDENCE","CHALLENGE","SYNTH","DECIDE","SEAL"],xe=0,nt=3,Ae=6,rt=7,st={primary:.04,supporting:.24,contradicting:-.22,superseded:-.48},it=.052,at=6,ot=z[nt]-.02,ct=.14;function lt(t){var c,l;if(!t||!t.alive)return null;const e=String(((c=t.raw)==null?void 0:c.query)??""),s=(t.stages??[]).slice(0,z.length).map((r,p)=>({index:p,kind:(r==null?void 0:r.kind)??String(p),label:(r==null?void 0:r.label)??(r==null?void 0:r.kind)??"",short:tt[p]??((r==null?void 0:r.kind)??"").toUpperCase(),x:z[p],lit:!!(r!=null&&r.lit),confidence:(r==null?void 0:r.confidence)??0,count:(r==null?void 0:r.count)??0})),u=new Map;for(const r of t.evidence??[]){const p=u.get(r.role)??[];p.push(r),u.set(r.role,p)}const h=[];for(const[r,p]of u){const b=st[r],k=p.slice(0,at);k.forEach((x,S)=>{const C=b+(S-(k.length-1)/2)*it,T=ot+S/Math.max(1,k.length-1)*ct;h.push({id:x.id,role:r,trust:x.trust,preview:x.preview,x:T,y:C})})}const i=t.recommended!=null?{x:z[Ae],y:xe,confidence:Math.max(0,Math.min(1,t.recommended.trust_score))}:null,d=i?h.map(r=>({fromX:r.x,fromY:r.y,toX:i.x,toY:i.y,trust:r.trust,role:r.role,sign:r.role==="contradicting"?-1:1})):[],a=new Map(h.map(r=>[r.id,r])),_=(t.contradictions??[]).map(r=>{const p=a.get(r.stronger.id),b=a.get(r.weaker.id);return!p||!b?null:{ax:p.x,ay:p.y,bx:b.x,by:b.y,strength:Math.max(.35,r.topic_overlap||.5)}}).filter(r=>r!==null),E=(t.superseded??[]).map(r=>{const p=a.get(r.id);return p?{x:p.x,y:p.y,toX:(i==null?void 0:i.x)??z[Ae],toY:(i==null?void 0:i.y)??xe}:null}).filter(r=>r!==null);return{query:e,gates:s,evidence:h,ribbons:d,nucleus:i,fringes:_,scars:E,receiptX:z[rt],sourceCount:((l=t.evidence)==null?void 0:l.length)??0}}const dt=`
struct Params {
	frame: f32, loopPhase: f32, nodeCount: f32, edgeCount: f32,
	pathCount: f32, pulse: f32, viewportW: f32, viewportH: f32,
	brightness: f32, demoId: f32, time: f32, captureMode: f32,
	liveKind: f32, liveFrame: f32, liveEnergy: f32, projectionDays: f32,
	cursorX: f32, cursorY: f32, cursorVx: f32, cursorVy: f32,
};

// Instance record (12 floats) — mirrors reasoning-geometry-pass INSTANCE_FLOATS.
struct Inst {
	a: vec2f,       // endpoint A / center (NDC)
	b: vec2f,       // endpoint B (== a for points)
	kind: f32,      // 0 beam, 1 ribbon, 2 nucleus, 3 fringe, 4 scar
	thickness: f32, // NDC half-width
	trust: f32,     // 0..1
	sign: f32,      // +1 support / -1 oppose
	energy: f32,    // 0..1
	seed: f32,      // per-instance phase
	extra: f32,     // kind-specific (fringe strength / nucleus confidence)
	pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> insts: array<Inst>;

const TAU: f32 = 6.28318530718;
const QUAD = array<vec2f, 6>(
	vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
	vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) world: vec2f,        // NDC position of this fragment
	@location(1) @interpolate(flat) idx: u32,
};

fn aspect() -> f32 { return max(0.0001, params.viewportW / max(1.0, params.viewportH)); }

// smoothstep-based value-suppressing palette (Correll/Moritz/Heer): low trust →
// neutral grey (never jitter), high trust → vivid. base is the category hue.
fn vsup(base: vec3f, trust: f32) -> vec3f {
	let s = smoothstep(0.0, 1.0, clamp(trust, 0.0, 1.0));
	let neutral = vec3f(0.42, 0.46, 0.52);
	let bright = 0.72 + 0.28 * s;
	return mix(neutral, base, s) * bright;
}

// PRGn-style diverging hue by signed influence: + = green (support), − = purple
// (oppose). |mag| drives saturation; zero-point kept off pure white.
fn prgn(sign: f32, mag: f32) -> vec3f {
	let green = vec3f(0.0, 0.85, 0.45);
	let purple = vec3f(0.55, 0.15, 0.75);
	let base = select(purple, green, sign >= 0.0);
	return mix(vec3f(0.35, 0.37, 0.4), base, clamp(mag, 0.0, 1.0));
}

@vertex
fn vs_geo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	let inst = insts[ii];
	let corner = QUAD[vi];
	let asp = aspect();

	// Bounding quad: for line-like kinds (beam/ribbon/scar/fringe) expand along
	// the A→B segment plus thickness on the perpendicular; for point kinds
	// (nucleus) a square of side 2*thickness around A.
	var pos: vec2f;
	let is_point = inst.kind == 2.0;
	if (is_point) {
		let r = inst.thickness + 0.04; // padding for glow/rings
		pos = inst.a + (corner - vec2f(0.5)) * (2.0 * r);
	} else {
		let dir = inst.b - inst.a;
		let len = max(1e-4, length(dir));
		let t_hat = dir / len;
		let n_hat = vec2f(-t_hat.y, t_hat.x);
		let half_w = inst.thickness + 0.03; // padding for AA/glow
		// corner.x runs along the segment (0..1), corner.y across (-1..1)
		let along = corner.x;
		let across = (corner.y - 0.5) * 2.0;
		pos = inst.a + t_hat * (along * len) + n_hat * (across * half_w);
	}

	// aspect-correct so circles stay round (mirror the text pass convention).
	var clip = pos;
	clip.x = clip.x / max(asp, 1.0);
	clip.y = clip.y * min(asp, 1.0);
	out.clip = vec4f(clip, 0.0, 1.0);
	out.world = pos;
	out.idx = ii;
	return out;
}

// ── Per-kind SDF shading. Each returns premultiplied additive HDR rgb. ────────
// FLEET: fill these five. Each is a PURE function of (inst, p, t) → vec3f. Keep
// them deterministic (use params.time via t), additive (return black to skip a
// pixel), and honest (brightness ∝ the real trust/energy/confidence fields).

fn shade_beam(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// BEAM: a bright emissive causal line A→B along y with a flowing pulse.
	let d = sdf_segment(p, inst.a, inst.b);
	let core = smoothstep(inst.thickness, 0.0, d);
	let glow = smoothstep(inst.thickness * 4.0, 0.0, d) * 0.35;
	// flow pulse travelling toward the decision (left→right)
	let along = clamp((p.x - inst.a.x) / max(1e-4, inst.b.x - inst.a.x), 0.0, 1.0);
	let flow = 0.5 + 0.5 * sin(along * 18.0 - t * 3.0);
	let cyan = vec3f(0.0, 0.96, 0.83);
	return cyan * inst.energy * (core * (0.6 + 0.6 * flow) + glow);
}

fn shade_ribbon(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// RIBBON: tapered (wide at nucleus, narrow at source), UV-scroll flow toward
	// the decision, head-bright opacity gradient, PRGn hue by sign.
	let along = clamp(dot(p - inst.a, inst.b - inst.a) / max(1e-6, dot(inst.b - inst.a, inst.b - inst.a)), 0.0, 1.0);
	let d = sdf_segment(p, inst.a, inst.b);
	let taper = inst.thickness * mix(0.35, 1.0, along); // widen toward B (nucleus)
	let core = smoothstep(taper, 0.0, d);
	// flowing dashes travelling A→B (causal direction)
	let flow = 0.5 + 0.5 * sin(along * 26.0 - t * 4.0 - inst.seed);
	let headBright = mix(0.35, 1.0, along); // brighter toward the decision head
	let hue = prgn(inst.sign, inst.trust);
	return hue * inst.energy * core * flow * headBright;
}

fn shade_nucleus(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// NUCLEUS: recommendation core; stability = confidence. Coherent (tight solid
	// rings) when confident, scattered/soft when not. Confidence in inst.extra.
	let c = inst.extra;
	let r = length(p - inst.a);
	let coherence = smoothstep(0.0, 1.0, c);
	let coreR = inst.thickness * (0.5 + 0.3 * c);
	let core = smoothstep(coreR, 0.0, r);
	// 1..4 concentric rings, more + tighter as confidence rises
	let ringCount = 1.0 + floor(c * 3.0);
	let ringPhase = r / max(1e-4, inst.thickness) * ringCount * TAU;
	let ring = pow(max(0.0, sin(ringPhase - t * 1.5)), 6.0) * smoothstep(inst.thickness * 1.6, inst.thickness * 0.4, r) * coherence;
	let hot = vec3f(0.91, 1.0, 0.72);
	return vsup(hot, c) * (core * 1.4 + ring * 0.8);
}

fn shade_fringe(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// FRINGE: two-source interference between contradicting evidence a,b. Scarlet
	// standing wave (cos^2) that breathes; reads as "these two conflict".
	let r1 = distance(p, inst.a);
	let r2 = distance(p, inst.b);
	let lambda = 0.05;
	let phase = TAU * (r1 - r2) / lambda - t * 2.0;
	let fr = cos(phase * 0.5);
	let intensity = fr * fr;
	// confine the fringe to the region BETWEEN the two sources
	let mid = (inst.a + inst.b) * 0.5;
	let span = distance(inst.a, inst.b) * 0.6 + 0.06;
	let mask = smoothstep(span, span * 0.4, distance(p, mid));
	let scarlet = vec3f(0.95, 0.08, 0.14);
	return scarlet * intensity * mask * inst.extra;
}

fn shade_scar(inst: Inst, p: vec2f, t: f32) -> vec3f {
	// SCAR: superseded evidence leaves a dim etched mark at A, with an AMBER
	// transfer filament flowing A→B (into the replacement).
	let dScar = sdf_segment(p, inst.a - vec2f(0.02, 0.0), inst.a + vec2f(0.02, 0.0));
	let scar = smoothstep(0.006, 0.0, dScar) * 0.5;
	let dFil = sdf_segment(p, inst.a, inst.b);
	let along = clamp(dot(p - inst.a, inst.b - inst.a) / max(1e-6, dot(inst.b - inst.a, inst.b - inst.a)), 0.0, 1.0);
	let flow = 0.5 + 0.5 * sin(along * 20.0 - t * 3.5);
	let fil = smoothstep(inst.thickness * 0.6, 0.0, dFil) * flow * (1.0 - along * 0.4);
	let amber = vec3f(1.0, 0.82, 0.4);
	let ash = vec3f(0.45, 0.4, 0.38);
	return ash * scar + amber * fil * 0.7;
}

// segment SDF helper
fn sdf_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
	let pa = p - a;
	let ba = b - a;
	let h = clamp(dot(pa, ba) / max(1e-6, dot(ba, ba)), 0.0, 1.0);
	return length(pa - ba * h);
}

@fragment
fn fs_geo(in: VSOut) -> @location(0) vec4f {
	let inst = insts[in.idx];
	let p = in.world;
	let t = params.time;
	var rgb = vec3f(0.0);
	let k = inst.kind;
	if (k == 0.0) { rgb = shade_beam(inst, p, t); }
	else if (k == 1.0) { rgb = shade_ribbon(inst, p, t); }
	else if (k == 2.0) { rgb = shade_nucleus(inst, p, t); }
	else if (k == 3.0) { rgb = shade_fringe(inst, p, t); }
	else if (k == 4.0) { rgb = shade_scar(inst, p, t); }
	rgb = rgb * params.brightness;
	// additive: alpha carries nothing; premultiplied rgb is the contribution.
	return vec4f(rgb, 1.0);
}
`,ie=12,he=512,j={beam:0,ribbon:1,nucleus:2,fringe:3,scar:4};class ut{constructor(e){R(this,"engine");R(this,"layout",null);R(this,"pipeline",null);R(this,"bindGroup",null);R(this,"instanceBuffer",null);R(this,"instanceCount",0);R(this,"ready",!1);this.engine=e}uploadScene(e){this.layout=(e==null?void 0:e.organ)==="reasoning"?lt(e):null,this.ensurePipeline(),this.uploadInstances()}ensurePipeline(){if(this.pipeline||this.ready)return;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer)return;this.ready=!0,this.instanceBuffer=e.createBuffer({label:"reasoning-geometry-instances",size:he*ie*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const s=e.createShaderModule({label:"reasoning-geometry-wgsl",code:dt}),u=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]});this.bindGroup=e.createBindGroup({layout:u,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.instanceBuffer}}]});const h={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.pipeline=e.createRenderPipeline({label:"reasoning-geometry-pipeline",layout:e.createPipelineLayout({bindGroupLayouts:[u]}),vertex:{module:s,entryPoint:"vs_geo"},fragment:{module:s,entryPoint:"fs_geo",targets:[{format:this.engine.sceneFormat,blend:h}]},primitive:{topology:"triangle-list"}})}uploadInstances(){const e=this.engine.gpuDevice;if(!e||!this.instanceBuffer)return;const s=this.layout;if(!s){this.instanceCount=0;return}const u=new Float32Array(he*ie);let h=0;const i=d=>{h>=he||(u.set(d,h*ie),h++)};for(let d=0;d<s.gates.length-1;d++){const a=s.gates[d],_=s.gates[d+1],E=a.lit&&_.lit?1:a.lit||_.lit?.5:.18;i([a.x,0,_.x,0,j.beam,.006,1,1,E,d,0,0])}for(let d=0;d<s.ribbons.length;d++){const a=s.ribbons[d];i([a.fromX,a.fromY,a.toX,a.toY,j.ribbon,.004+.01*a.trust,a.trust,a.sign,.5+.5*a.trust,d*1.7,0,0])}for(let d=0;d<s.scars.length;d++){const a=s.scars[d];i([a.x,a.y,a.toX,a.toY,j.scar,.006,.5,1,.7,d*2.3,0,0])}for(let d=0;d<s.fringes.length;d++){const a=s.fringes[d];i([a.ax,a.ay,a.bx,a.by,j.fringe,.02,.9,-1,a.strength,d*3.1,a.strength,0])}if(s.nucleus){const d=s.nucleus.confidence;i([s.nucleus.x,s.nucleus.y,s.nucleus.x,s.nucleus.y,j.nucleus,.035+.05*d,d,1,.8+.2*d,0,d,0])}this.instanceCount=h,e.queue.writeBuffer(this.instanceBuffer,0,u,0,h*ie)}render(e){!this.pipeline||!this.bindGroup||this.instanceCount===0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.instanceCount))}dispose(){var e;(e=this.instanceBuffer)==null||e.destroy(),this.instanceBuffer=null,this.pipeline=null,this.bindGroup=null}}function pt(t){return new ut(t)}const H=t=>{const e=Ke(t);return[e[0],e[1],e[2]]},ve=H("#00F5D4"),Re=H("#9DFFEB"),ft=H("#FF3B30"),ht=H("#FFD166"),ae=H("#E9FFB7"),Y=[.42,.46,.52],me=H("#6B7A88"),B=[-.86,-.526,-.214,.074,.335,.562,.747,.86],D=0,mt=["INTENT","RETRIEVE","ACTIVATE","EVIDENCE","CHALLENGE","SYNTH","DECIDE","SEAL"],gt={primary:.04,supporting:.24,contradicting:-.22,superseded:-.48},Se=.052,oe=6,vt=t=>t*t*(3-2*t);function ge(t,e){const s=vt(Math.max(0,Math.min(1,e))),u=.72+.28*s,h=(Y[0]+(t[0]-Y[0])*s)*u,i=(Y[1]+(t[1]-Y[1])*s)*u,d=(Y[2]+(t[2]-Y[2])*s)*u,a=.4+.6*s;return[Math.min(1,h),Math.min(1,i),Math.min(1,d),a]}function ke(t){switch(t){case"primary":return ve;case"supporting":return Re;case"contradicting":return ft;case"superseded":return ht}}function Q(t){return t.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}const I=-1e5;class yt{constructor(e){R(this,"text");R(this,"scene",null);R(this,"ready",!1);R(this,"initPromise",null);this.text=new qe(e)}uploadScene(e){this.scene=(e==null?void 0:e.organ)==="reasoning"?e:null,this.ensureReady().then(()=>this.text.setText(this.build()))}render(e){this.text.render(e)}pickAt(e,s){return this.text.pickAt(e,s)}dispose(){this.text.dispose()}async ensureReady(){this.initPromise||(this.initPromise=this.text.init().then(()=>void(this.ready=!0))),await this.initPromise}build(){var a,_,E;const e=this.scene;if(!e||!e.alive)return[];const s=[],u=Q(String(((a=e.raw)==null?void 0:a.query)??"")).slice(0,60);s.push({id:"trace:query",kind:"trace-query",text:u?`> ${u}`:"> (trace)",x:-.94,y:.82,size:.03,color:[...ae,1],weight:.95,depth:1,startFrame:I,revealSpan:1,maxWidthEm:60});const h=e.stages??[];for(let c=0;c<h.length&&c<B.length;c++){const l=h[c],r=l==null?void 0:l.lit,p=B[c],b=mt[c]??Q(((l==null?void 0:l.kind)??"").toUpperCase()),k=Q(((l==null?void 0:l.label)??(l==null?void 0:l.kind)??"").toUpperCase()),x=r?[...ve,1]:[...me,.5],C=c%2===0?D+.055:D-.075,T=b.length*.014*.62*.5;s.push({id:`trace:gate:${(l==null?void 0:l.kind)??c}`,kind:"trace-gate",text:b,x:p-T,y:C,size:.014,color:x,weight:r?.9:.6,depth:r?1:.7,startFrame:I,revealSpan:1,maxWidthEm:14,hitPadX:.055,hitPadY:.08,ariaLabel:`Gate ${k}: ${(l==null?void 0:l.count)??0} items, ${Math.round(((l==null?void 0:l.confidence)??0)*100)}% confidence`}),s.push({id:`trace:gatenode:${c}`,kind:"trace-beam",text:r?"O":"+",x:p-.008,y:D-.012,size:r?.028:.02,color:x,depth:r?1:.6,startFrame:I,revealSpan:1,maxWidthEm:3}),c<B.length-1&&s.push({id:`trace:beamseg:${c}`,kind:"trace-beam",text:"·······",x:p+.02,y:D-.006,size:.014,color:r?[...ve,.5]:[...me,.25],depth:.7,startFrame:I,revealSpan:1,maxWidthEm:40})}const i=new Map;for(const c of e.evidence??[]){const l=i.get(c.role)??[];l.push(c),i.set(c.role,l)}for(const[c,l]of i){const r=gt[c],p=l.slice(0,oe);p.forEach((b,k)=>{const x=r+(k-(p.length-1)/2)*Se,S=Q(b.preview).replace(/\s+/g," ").trim().slice(0,46);s.push({id:`trace:ev:${b.id}`,kind:"trace-evidence",text:`${S} · ${Math.round(b.trust*100)}%`,x:B[3]-.02,y:x,size:.016,color:ge(ke(c),b.trust),weight:.4+.5*b.trust,depth:.6+.4*b.trust,startFrame:I,revealSpan:1,maxWidthEm:52,hitPadX:.03,hitPadY:.02,ariaLabel:`${c} evidence, trust ${Math.round(b.trust*100)}%: ${S}`,preview:S})}),l.length>oe&&s.push({id:`trace:super:${c}`,kind:"trace-supernode",text:`+${l.length-oe} more ${c}`,x:B[3]-.02,y:r-(oe/2+1)*Se,size:.014,color:[...me,.7],depth:.6,startFrame:I,revealSpan:1,maxWidthEm:30}),s.push({id:`trace:lanelabel:${c}`,kind:"trace-hud",text:c.toUpperCase(),x:B[3]-.16,y:r,size:.013,color:[...ke(c),.55],depth:.7,startFrame:I,revealSpan:1,maxWidthEm:14})}if(e.recommended){const c=Math.max(0,Math.min(1,e.recommended.trust_score)),l=Q(e.recommended.answer_preview).replace(/\s+/g," ").trim().slice(0,48),r=B[6],p=c>=.6;s.push({id:"trace:nucleus",kind:"trace-recommendation",text:p?"O":"o",x:r,y:D,size:.05+.05*c,color:ge(ae,c),weight:.6+.4*c,depth:1,startFrame:I,revealSpan:1,maxWidthEm:4,hitPadX:.06,hitPadY:.06,ariaLabel:`Recommendation, ${Math.round(c*100)}% confidence: ${l}`,preview:l}),s.push({id:"trace:reclabel",kind:"trace-hud",text:`${p?"LOCKED":"OPEN"} · ${Math.round(c*100)}%`,x:r-.05,y:D-.11,size:.016,color:ge(ae,c),depth:1,startFrame:I,revealSpan:1,maxWidthEm:18}),s.push({id:"trace:answer",kind:"trace-hud",text:l,x:r-.28,y:D-.17,size:.016,color:[...ae,.92],depth:.95,startFrame:I,revealSpan:1,maxWidthEm:44,maxLines:2})}const d=((_=e.evidence)==null?void 0:_.length)??0;return s.push({id:"trace:receipt",kind:"trace-receipt",text:`[ ${d} SOURCES SEALED ]`,x:B[7]-.14,y:D-.07,size:.015,color:[...Re,.85],depth:.9,startFrame:I,revealSpan:1,maxWidthEm:24,hitPadX:.05,hitPadY:.04,ariaLabel:`Composition receipt: ${d} sources, analysed ${((E=e.raw)==null?void 0:E.memoriesAnalyzed)??d} memories`}),s}}function bt(t,e){return[pt(t),new yt(t)]}var _t=ye("<option></option>"),wt=ye('<div class="sr-only svelte-q2v96u" aria-live="polite"> </div>'),Et=ye('<!> <form class="sr-only svelte-q2v96u"><label for="reasoning-ask">Ask Vestige a question</label> <input id="reasoning-ask" list="reasoning-examples" autocomplete="off" spellcheck="false" placeholder="Ask your memory anything…"/> <datalist id="reasoning-examples"></datalist> <button type="submit">Trace decision</button></form> <div class="sr-only svelte-q2v96u" aria-live="polite" role="status"> </div> <!>',1);function Ft(t,e){Pe(e,!0);let s=W(""),u=W(!1),h=W(null),i=W(null),d=W(null),a=W(null),_=null,E=[];function c(){const o=v(i);if(!o||(o.evidence??[]).length===0){const g=E.map((w,n)=>({id:w.id||`rest:${n}`,score:.25+.4*l(w.trust??.5),hue:fe.bridge,energy:.14+.26*l(w.trust??.5),metric2:l(w.trust??.5),kind:"reasoning-rest",payload:w}));return Ee(g,{maxRadius:.9,minCellR:.014,maxCellR:.045})}const f=new Set((o.contradictions??[]).flatMap(g=>{var w,n;return[(w=g.stronger)==null?void 0:w.id,(n=g.weaker)==null?void 0:n.id]}).filter(Boolean)),m=(o.evidence??[]).map((g,w)=>{var n;return{id:g.id||`evidence:${w}`,score:.4+.6*l(g.trust??.5),hue:f.has(g.id)?fe.scarlet:fe.forward,energy:.45+.55*l(g.trust??.5),metric2:l(g.trust??.5),scar:f.has(g.id),selected:g.id===((n=o.recommended)==null?void 0:n.memory_id),kind:"reasoning-evidence",payload:g}});return Ee(m,{maxRadius:.9,minCellR:.016,maxCellR:.06})}function l(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:.5))}function r(o,f){const m=new je(o);return _=m,m.setCells(c()),[{compute:w=>m.compute(w),render:w=>m.render(w),dispose:()=>{m.dispose(),_===m&&(_=null)}},...bt(o)]}Me(()=>{var o;(o=v(i))==null||o.evidence.length,_==null||_.setCells(c())});const p=["What port does the dev server use?","Should I enable prefix caching with vLLM?","How does FSRS-6 trust scoring work?","Why did the benchmark score drop after the parser change?"];async function b(){const o=v(s).trim();if(!(!o||v(u))){N(u,!0),N(h,null),N(i,null),N(d,null);try{const f=await we.deepReference(o,20);N(i,et(f),!0)}catch(f){N(h,f instanceof Error?f.message:"Unknown error",!0)}finally{N(u,!1)}}}const k=_e(()=>{var m,g;if(v(u))return"Reasoning in progress.";if(v(h))return`Error: ${v(h)}`;if(!v(i))return"Ask a question to trace how Vestige forms a decision from memory.";const o=v(i),f=((m=o.recommended)==null?void 0:m.answer_preview)??"no recommendation";return`Decision trace for "${v(s)}". ${o.evidence.length} evidence memories, ${o.contradictions.length} contradiction${o.contradictions.length===1?"":"s"}, ${o.superseded.length} superseded. Recommendation: ${f}. Confidence ${Math.round((((g=o.recommended)==null?void 0:g.trust_score)??0)*100)} percent.`});function x(o){const f=o.payload;N(d,(f==null?void 0:f.ariaLabel)??(f==null?void 0:f.preview)??`${o.kind} selected`,!0)}function S(o){var f,m;(o.metaKey||o.ctrlKey)&&o.key.toLowerCase()==="k"&&(o.preventDefault(),(f=v(a))==null||f.focus(),(m=v(a))==null||m.select())}$e(()=>{var o;return(o=v(a))==null||o.focus(),window.addEventListener("keydown",S),we.memories.list({limit:"80"}).then(f=>{E=f.memories.map((m,g)=>{const w=l(m.retentionStrength);return{id:m.id||`rest:${g}`,trust:.35+.4*w,date:"",role:"supporting",preview:""}}),_==null||_.setCells(c())}).catch(()=>{}),()=>window.removeEventListener("keydown",S)});var C=Et();Ve("q2v96u",o=>{Be(()=>{De.title="Reasoning Theater · Vestige"})});var T=Le(C);{let o=_e(()=>`reasoning-trace:${v(s)||"empty"}`);He(T,{organ:"reasoning",get seed(){return v(o)},get scene(){return v(i)},passes:r,get loading(){return v(u)},get error(){return v(h)},emptyLabel:"ASK A QUESTION - PRESS CMD+K - WATCH THE DECISION FORM",onpick:x})}var M=q(T,2),G=q(pe(M),2);We(G),Ue(G,o=>N(a,o),()=>v(a));var ee=q(G,2);Xe(ee,21,()=>p,Ye,(o,f)=>{var m=_t(),g={};de(()=>{g!==(g=v(f))&&(m.value=(m.__value=v(f))??"")}),ue(o,m)}),se(ee),Ge(2),se(M);var U=q(M,2),te=pe(U,!0);se(U);var $=q(U,2);{var ce=o=>{var f=wt(),m=pe(f,!0);se(f),de(()=>be(m,v(d))),ue(o,f)};Oe($,o=>{v(d)&&o(ce)})}de(()=>be(te,v(k))),Ne("submit",M,o=>{o.preventDefault(),b()}),ze(G,()=>v(s),o=>N(s,o)),ue(t,C),Fe()}export{Ft as component};

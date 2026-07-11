var Re=Object.defineProperty;var Ce=(t,e,s)=>e in t?Re(t,e,{enumerable:!0,configurable:!0,writable:!0,value:s}):t[e]=s;var R=(t,e,s)=>Ce(t,typeof e!="symbol"?e+"":e,s);import"../chunks/Bzak7iHL.js";import{o as Te,e as Ie,s as ye}from"../chunks/DAau0uzT.js";import{p as $e,j as Ne,g as m,aH as Pe,Y as de,a as ue,b as Me,d as W,h as Le,X as j,s as I,f as ve,u as be,$ as Fe,c as pe,r as ie,af as Be}from"../chunks/CGq8RnJq.js";import{b as De,i as Ge}from"../chunks/Ccqjq5DS.js";import{e as Ue,r as Oe,i as We}from"../chunks/DqfV0sZu.js";import{h as Xe}from"../chunks/De_e6MzK.js";import{b as Ve}from"../chunks/DGM4cicq.js";import{a as Ye}from"../chunks/D35IQVqe.js";import{R as ze}from"../chunks/BpEKQwpr.js";import{T as He}from"../chunks/D7ozXiSB.js";import{r as qe}from"../chunks/BMB5u1EX.js";import{L as Ke,F as _e,l as je}from"../chunks/DETSv_kY.js";const Qe={intent:"Intent classification",retrieve:"Memory retrieval",activate:"Activation expansion",evidence:"Evidence grounding",contradiction:"Contradiction check",synthesis:"Synthesis",recommendation:"Recommendation",receipt:"Composition receipt"},Je=["intent","retrieve","activate","evidence","contradiction","synthesis","recommendation","receipt"];function B(t){return t&&typeof t=="object"&&!Array.isArray(t)?t:{}}function g(t,e=""){return typeof t=="string"?t:t==null?e:String(t)}function ee(t,e=0){return typeof t=="number"&&Number.isFinite(t)?t:e}function V(t){return Math.max(0,Math.min(1,t))}function te(t){const e=ee(t,0);return V(e>1?e/100:e)}function Ze(t){return t==="primary"||t==="supporting"||t==="contradicting"||t==="superseded"?t:"supporting"}function Q(t,e=""){const s=te(t.trust??t.trust_score??0);return{id:g(t.id??t.memory_id??e),trust:s,date:g(t.date??t.created_at??""),role:Ze(t.role),preview:g(t.preview??t.answer_preview??t.content??""),nodeType:t.node_type?g(t.node_type):t.nodeType?g(t.nodeType):void 0}}function O(t,e,s){return s?{kind:t,id:e,scalar:s}:{kind:t,id:e||`${t}:unknown`}}function P(t,e){return{kind:"scalar",id:`deep_reference.${t}`,scalar:{name:t,value:e}}}function et(t){var y,b,T,L;const e=B(t),u=(Array.isArray(e.evidence)?e.evidence.map(B):[]).map(r=>Q(r)).filter(r=>r.id.length>0),f=B(e.recommended),i=Object.keys(f).length>0||u.length>0?{answer_preview:g(f.answer_preview??((y=u[0])==null?void 0:y.preview)??""),memory_id:g(f.memory_id??((b=u[0])==null?void 0:b.id)??""),trust_score:te(f.trust_score??((T=u[0])==null?void 0:T.trust)??0),date:g(f.date??((L=u[0])==null?void 0:L.date)??"")}:null,a=(Array.isArray(e.contradictions)?e.contradictions.map(B):Array.isArray(e.claim_conflicts)?e.claim_conflicts.map(B):[]).map(r=>{const E=B(r.stronger),F=B(r.weaker);if(Object.keys(E).length>0||Object.keys(F).length>0){const re=Q(E),se=Q(F);return{stronger:re,weaker:se,topic_overlap:V(ee(r.topic_overlap,0)),summary:g(r.summary,`Trust-weighted conflict: ${re.id.slice(0,8)} over ${se.id.slice(0,8)}`)}}const U=Q({id:r.a_id,role:"contradicting"}),le=Q({id:r.b_id,role:"contradicting"});return{stronger:U,weaker:le,topic_overlap:V(ee(r.topic_overlap,0)),summary:g(r.summary??r.reason,"Trust-weighted conflict between high-FSRS memories.")}}).filter(r=>r.stronger.id||r.weaker.id),w=(Array.isArray(e.superseded)?e.superseded.map(B):[]).map(r=>r.id||r.superseded_by?{id:g(r.id),preview:g(r.preview??""),trust:te(r.trust??0),date:g(r.date??""),superseded_by:g(r.superseded_by??(i==null?void 0:i.memory_id)??""),reason:g(r.reason??"Superseded by newer memory with higher trust.")}:{id:g(r.old_id),preview:g(r.preview??""),trust:te(r.trust??0),date:g(r.date??""),superseded_by:g(r.new_id??(i==null?void 0:i.memory_id)??""),reason:g(r.reason??"Superseded by newer memory with higher trust.")}).filter(r=>r.id||r.superseded_by),c=te(e.confidence),l=ee(e.memoriesAnalyzed??e.memories_analyzed,u.length),n=ee(e.activationExpanded??e.activation_expanded,0),p=g(e.intent??""),v=g(e.reasoning??e.guidance??""),S=g(e.guidance??""),_=g(e.composition_event_id??""),A=g(e.compositionWriteStatus??e.composition_write_status??""),C=_||A,k=_.length>0,M=u.map((r,E)=>({source:O("memory",r.id),index:E,label:r.preview||r.id.slice(0,8),retention:V(r.trust),trust:V(r.trust),lastAccessed:r.date||void 0,tags:[r.role,...r.nodeType?[r.nodeType]:[]],type:r.nodeType??"memory"})),H=new Map(M.map(r=>[r.source.id,r.index])),q=a.flatMap((r,E)=>{const F=H.get(r.stronger.id),U=H.get(r.weaker.id);return F==null||U==null?[]:[{source:O("pair",`contradiction:${r.stronger.id}:${r.weaker.id}`),sourceIndex:F,targetIndex:U,weight:Math.max(.2,r.topic_overlap||.5),kind:"contradiction"}]}),K=[];a.length>0&&K.push({source:O("event",`deep_reference.contradictions.${a.length}`),type:"ReasoningContradictionInterrupt",targetIndex:-1,frame:250,energy:a.length}),w.length>0&&K.push({source:O("event",`deep_reference.superseded.${w.length}`),type:"ReasoningSupersessionInterrupt",targetIndex:-1,frame:330,energy:w.length});const ne=[];C&&ne.push({source:k?O("receipt",_):P("compositionWriteStatus",A==="persisted"?1:0),label:k?`receipt ${_.slice(0,8)}`:A,nodeIndices:M.map(r=>r.index)});const N=(r,E,F,U,le,re,se,ke="none")=>({index:r,kind:E,label:Qe[E],count:F,confidence:V(U),lit:F>0||U>0,provenance:re,exposed:le,not_exposed_by_backend:se,interrupt:ke}),o=Je.map((r,E)=>{switch(r){case"intent":return N(E,r,p?1:0,p?c:0,{intent:p},P("intent_present",p?1:0),["raw classifier trace"]);case"retrieve":return N(E,r,l||u.length,c,{memoriesAnalyzed:l,evidence_count:u.length},P("memoriesAnalyzed",l),["candidate ids discarded before evidence"]);case"activate":return N(E,r,n,n>0?c:0,{activationExpanded:n},P("activationExpanded",n),["activation path/map"]);case"evidence":return N(E,r,u.length,u.length>0?c:0,{evidence:u},P("evidence_count",u.length),["reranker discarded candidates"]);case"contradiction":return N(E,r,a.length,a.length>0?c:0,{contradictions:a,claim_conflicts:e.claim_conflicts??[]},P("contradiction_count",a.length),["full claim graph"],a.length>0?"contradiction":"none");case"synthesis":return N(E,r,v||S?1:0,v||S?c:0,{reasoning:v,guidance:S},P("synthesis_present",v||S?1:0),["token-level chain internals"]);case"recommendation":return N(E,r,i!=null&&i.memory_id?1:0,(i==null?void 0:i.trust_score)??0,{recommended:i},i!=null&&i.memory_id?O("memory",i.memory_id):P("recommended_present",0),["alternative recommendations"]);case"receipt":return N(E,r,C?1:0,C?c:0,{composition_event_id:_,compositionWriteStatus:A},k?O("receipt",_):P("compositionWriteStatus",A==="persisted"?1:0),["separate receipt field"],w.length>0?"supersession":"none")}});return{organ:"reasoning",nodes:M,edges:q,events:K,receipts:ne,scalars:{confidence:c,memoriesAnalyzed:l,activationExpanded:n,evidenceCount:u.length,contradictionCount:a.length,supersededCount:w.length,compositionPersisted:k?1:0},alive:!!(p||v||S||u.length||a.length||w.length||i!=null&&i.memory_id||C),stages:o,evidence:u,contradictions:a,superseded:w,recommended:i,raw:e}}const Y=[-.86,-.526,-.214,.074,.335,.562,.747,.86],tt=["INTENT","RETRIEVE","ACTIVATE","EVIDENCE","CHALLENGE","SYNTH","DECIDE","SEAL"],we=0,nt=3,Ee=6,rt=7,st={primary:.04,supporting:.24,contradicting:-.22,superseded:-.48},it=.052,at=6,ot=Y[nt]-.02,ct=.14;function lt(t){var c,l;if(!t||!t.alive)return null;const e=String(((c=t.raw)==null?void 0:c.query)??""),s=(t.stages??[]).slice(0,Y.length).map((n,p)=>({index:p,kind:(n==null?void 0:n.kind)??String(p),label:(n==null?void 0:n.label)??(n==null?void 0:n.kind)??"",short:tt[p]??((n==null?void 0:n.kind)??"").toUpperCase(),x:Y[p],lit:!!(n!=null&&n.lit),confidence:(n==null?void 0:n.confidence)??0,count:(n==null?void 0:n.count)??0})),u=new Map;for(const n of t.evidence??[]){const p=u.get(n.role)??[];p.push(n),u.set(n.role,p)}const f=[];for(const[n,p]of u){const v=st[n],S=p.slice(0,at);S.forEach((_,A)=>{const C=v+(A-(S.length-1)/2)*it,k=ot+A/Math.max(1,S.length-1)*ct;f.push({id:_.id,role:n,trust:_.trust,preview:_.preview,x:k,y:C})})}const i=t.recommended!=null?{x:Y[Ee],y:we,confidence:Math.max(0,Math.min(1,t.recommended.trust_score))}:null,d=i?f.map(n=>({fromX:n.x,fromY:n.y,toX:i.x,toY:i.y,trust:n.trust,role:n.role,sign:n.role==="contradicting"?-1:1})):[],a=new Map(f.map(n=>[n.id,n])),x=(t.contradictions??[]).map(n=>{const p=a.get(n.stronger.id),v=a.get(n.weaker.id);return!p||!v?null:{ax:p.x,ay:p.y,bx:v.x,by:v.y,strength:Math.max(.35,n.topic_overlap||.5)}}).filter(n=>n!==null),w=(t.superseded??[]).map(n=>{const p=a.get(n.id);return p?{x:p.x,y:p.y,toX:(i==null?void 0:i.x)??Y[Ee],toY:(i==null?void 0:i.y)??we}:null}).filter(n=>n!==null);return{query:e,gates:s,evidence:f,ribbons:d,nucleus:i,fringes:x,scars:w,receiptX:Y[rt],sourceCount:((l=t.evidence)==null?void 0:l.length)??0}}const dt=`
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
`,ae=12,fe=512,J={beam:0,ribbon:1,nucleus:2,fringe:3,scar:4};class ut{constructor(e){R(this,"engine");R(this,"layout",null);R(this,"pipeline",null);R(this,"bindGroup",null);R(this,"instanceBuffer",null);R(this,"instanceCount",0);R(this,"ready",!1);this.engine=e}uploadScene(e){this.layout=(e==null?void 0:e.organ)==="reasoning"?lt(e):null,this.ensurePipeline(),this.uploadInstances()}ensurePipeline(){if(this.pipeline||this.ready)return;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer)return;this.ready=!0,this.instanceBuffer=e.createBuffer({label:"reasoning-geometry-instances",size:fe*ae*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});const s=e.createShaderModule({label:"reasoning-geometry-wgsl",code:dt}),u=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}}]});this.bindGroup=e.createBindGroup({layout:u,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.instanceBuffer}}]});const f={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.pipeline=e.createRenderPipeline({label:"reasoning-geometry-pipeline",layout:e.createPipelineLayout({bindGroupLayouts:[u]}),vertex:{module:s,entryPoint:"vs_geo"},fragment:{module:s,entryPoint:"fs_geo",targets:[{format:this.engine.sceneFormat,blend:f}]},primitive:{topology:"triangle-list"}})}uploadInstances(){const e=this.engine.gpuDevice;if(!e||!this.instanceBuffer)return;const s=this.layout;if(!s){this.instanceCount=0;return}const u=new Float32Array(fe*ae);let f=0;const i=d=>{f>=fe||(u.set(d,f*ae),f++)};for(let d=0;d<s.gates.length-1;d++){const a=s.gates[d],x=s.gates[d+1],w=a.lit&&x.lit?1:a.lit||x.lit?.5:.18;i([a.x,0,x.x,0,J.beam,.006,1,1,w,d,0,0])}for(let d=0;d<s.ribbons.length;d++){const a=s.ribbons[d];i([a.fromX,a.fromY,a.toX,a.toY,J.ribbon,.004+.01*a.trust,a.trust,a.sign,.5+.5*a.trust,d*1.7,0,0])}for(let d=0;d<s.scars.length;d++){const a=s.scars[d];i([a.x,a.y,a.toX,a.toY,J.scar,.006,.5,1,.7,d*2.3,0,0])}for(let d=0;d<s.fringes.length;d++){const a=s.fringes[d];i([a.ax,a.ay,a.bx,a.by,J.fringe,.02,.9,-1,a.strength,d*3.1,a.strength,0])}if(s.nucleus){const d=s.nucleus.confidence;i([s.nucleus.x,s.nucleus.y,s.nucleus.x,s.nucleus.y,J.nucleus,.035+.05*d,d,1,.8+.2*d,0,d,0])}this.instanceCount=f,e.queue.writeBuffer(this.instanceBuffer,0,u,0,f*ae)}render(e){!this.pipeline||!this.bindGroup||this.instanceCount===0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.instanceCount))}dispose(){var e;(e=this.instanceBuffer)==null||e.destroy(),this.instanceBuffer=null,this.pipeline=null,this.bindGroup=null}}function pt(t){return new ut(t)}const z=t=>{const e=qe(t);return[e[0],e[1],e[2]]},ge=z("#00F5D4"),Se=z("#9DFFEB"),ft=z("#FF3B30"),ht=z("#FFD166"),oe=z("#E9FFB7"),X=[.42,.46,.52],he=z("#6B7A88"),D=[-.86,-.526,-.214,.074,.335,.562,.747,.86],G=0,mt=["INTENT","RETRIEVE","ACTIVATE","EVIDENCE","CHALLENGE","SYNTH","DECIDE","SEAL"],gt={primary:.04,supporting:.24,contradicting:-.22,superseded:-.48},xe=.052,ce=6,vt=t=>t*t*(3-2*t);function me(t,e){const s=vt(Math.max(0,Math.min(1,e))),u=.72+.28*s,f=(X[0]+(t[0]-X[0])*s)*u,i=(X[1]+(t[1]-X[1])*s)*u,d=(X[2]+(t[2]-X[2])*s)*u,a=.4+.6*s;return[Math.min(1,f),Math.min(1,i),Math.min(1,d),a]}function Ae(t){switch(t){case"primary":return ge;case"supporting":return Se;case"contradicting":return ft;case"superseded":return ht}}function Z(t){return t.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}const $=-1e5;class yt{constructor(e){R(this,"text");R(this,"scene",null);R(this,"ready",!1);R(this,"initPromise",null);this.text=new He(e)}uploadScene(e){this.scene=(e==null?void 0:e.organ)==="reasoning"?e:null,this.ensureReady().then(()=>this.text.setText(this.build()))}render(e){this.text.render(e)}pickAt(e,s){return this.text.pickAt(e,s)}dispose(){this.text.dispose()}async ensureReady(){this.initPromise||(this.initPromise=this.text.init().then(()=>void(this.ready=!0))),await this.initPromise}build(){var a,x,w;const e=this.scene;if(!e||!e.alive)return[];const s=[],u=Z(String(((a=e.raw)==null?void 0:a.query)??"")).slice(0,60);s.push({id:"trace:query",kind:"trace-query",text:u?`> ${u}`:"> (trace)",x:-.94,y:.82,size:.03,color:[...oe,1],weight:.95,depth:1,startFrame:$,revealSpan:1,maxWidthEm:60});const f=e.stages??[];for(let c=0;c<f.length&&c<D.length;c++){const l=f[c],n=l==null?void 0:l.lit,p=D[c],v=mt[c]??Z(((l==null?void 0:l.kind)??"").toUpperCase()),S=Z(((l==null?void 0:l.label)??(l==null?void 0:l.kind)??"").toUpperCase()),_=n?[...ge,1]:[...he,.5],C=c%2===0?G+.055:G-.075,k=v.length*.014*.62*.5;s.push({id:`trace:gate:${(l==null?void 0:l.kind)??c}`,kind:"trace-gate",text:v,x:p-k,y:C,size:.014,color:_,weight:n?.9:.6,depth:n?1:.7,startFrame:$,revealSpan:1,maxWidthEm:14,hitPadX:.055,hitPadY:.08,ariaLabel:`Gate ${S}: ${(l==null?void 0:l.count)??0} items, ${Math.round(((l==null?void 0:l.confidence)??0)*100)}% confidence`}),s.push({id:`trace:gatenode:${c}`,kind:"trace-beam",text:n?"O":"+",x:p-.008,y:G-.012,size:n?.028:.02,color:_,depth:n?1:.6,startFrame:$,revealSpan:1,maxWidthEm:3}),c<D.length-1&&s.push({id:`trace:beamseg:${c}`,kind:"trace-beam",text:"·······",x:p+.02,y:G-.006,size:.014,color:n?[...ge,.5]:[...he,.25],depth:.7,startFrame:$,revealSpan:1,maxWidthEm:40})}const i=new Map;for(const c of e.evidence??[]){const l=i.get(c.role)??[];l.push(c),i.set(c.role,l)}for(const[c,l]of i){const n=gt[c],p=l.slice(0,ce);p.forEach((v,S)=>{const _=n+(S-(p.length-1)/2)*xe,A=Z(v.preview).replace(/\s+/g," ").trim().slice(0,46);s.push({id:`trace:ev:${v.id}`,kind:"trace-evidence",text:`${A} · ${Math.round(v.trust*100)}%`,x:D[3]-.02,y:_,size:.016,color:me(Ae(c),v.trust),weight:.4+.5*v.trust,depth:.6+.4*v.trust,startFrame:$,revealSpan:1,maxWidthEm:52,hitPadX:.03,hitPadY:.02,ariaLabel:`${c} evidence, trust ${Math.round(v.trust*100)}%: ${A}`,preview:A})}),l.length>ce&&s.push({id:`trace:super:${c}`,kind:"trace-supernode",text:`+${l.length-ce} more ${c}`,x:D[3]-.02,y:n-(ce/2+1)*xe,size:.014,color:[...he,.7],depth:.6,startFrame:$,revealSpan:1,maxWidthEm:30}),s.push({id:`trace:lanelabel:${c}`,kind:"trace-hud",text:c.toUpperCase(),x:D[3]-.16,y:n,size:.013,color:[...Ae(c),.55],depth:.7,startFrame:$,revealSpan:1,maxWidthEm:14})}if(e.recommended){const c=Math.max(0,Math.min(1,e.recommended.trust_score)),l=Z(e.recommended.answer_preview).replace(/\s+/g," ").trim().slice(0,48),n=D[6],p=c>=.6;s.push({id:"trace:nucleus",kind:"trace-recommendation",text:p?"O":"o",x:n,y:G,size:.05+.05*c,color:me(oe,c),weight:.6+.4*c,depth:1,startFrame:$,revealSpan:1,maxWidthEm:4,hitPadX:.06,hitPadY:.06,ariaLabel:`Recommendation, ${Math.round(c*100)}% confidence: ${l}`,preview:l}),s.push({id:"trace:reclabel",kind:"trace-hud",text:`${p?"LOCKED":"OPEN"} · ${Math.round(c*100)}%`,x:n-.05,y:G-.11,size:.016,color:me(oe,c),depth:1,startFrame:$,revealSpan:1,maxWidthEm:18}),s.push({id:"trace:answer",kind:"trace-hud",text:l,x:n-.28,y:G-.17,size:.016,color:[...oe,.92],depth:.95,startFrame:$,revealSpan:1,maxWidthEm:44,maxLines:2})}const d=((x=e.evidence)==null?void 0:x.length)??0;return s.push({id:"trace:receipt",kind:"trace-receipt",text:`[ ${d} SOURCES SEALED ]`,x:D[7]-.14,y:G-.07,size:.015,color:[...Se,.85],depth:.9,startFrame:$,revealSpan:1,maxWidthEm:24,hitPadX:.05,hitPadY:.04,ariaLabel:`Composition receipt: ${d} sources, analysed ${((w=e.raw)==null?void 0:w.memoriesAnalyzed)??d} memories`}),s}}function bt(t,e){return[pt(t),new yt(t)]}var _t=ve("<option></option>"),wt=ve('<div class="sr-only svelte-q2v96u" aria-live="polite"> </div>'),Et=ve('<!> <form class="sr-only svelte-q2v96u"><label for="reasoning-ask">Ask Vestige a question</label> <input id="reasoning-ask" list="reasoning-examples" autocomplete="off" spellcheck="false" placeholder="Ask a question…"/> <datalist id="reasoning-examples"></datalist> <button type="submit">Trace decision</button></form> <div class="sr-only svelte-q2v96u" aria-live="polite" role="status"> </div> <!>',1);function Ft(t,e){$e(e,!0);let s=W(""),u=W(!1),f=W(null),i=W(null),d=W(null),a=W(null),x=null;function w(){const o=m(i);if(!o)return[];const h=new Set((o.contradictions??[]).flatMap(b=>{var T,L;return[(T=b.stronger)==null?void 0:T.id,(L=b.weaker)==null?void 0:L.id]}).filter(Boolean)),y=(o.evidence??[]).map((b,T)=>{var L;return{id:b.id||`evidence:${T}`,score:.4+.6*c(b.trust??.5),hue:h.has(b.id)?_e.scarlet:_e.forward,energy:.45+.55*c(b.trust??.5),metric2:c(b.trust??.5),scar:h.has(b.id),selected:b.id===((L=o.recommended)==null?void 0:L.memory_id),kind:"reasoning-evidence",payload:b}});return je(y,{maxRadius:.9,minCellR:.016,maxCellR:.06})}function c(o){return Math.min(1,Math.max(0,Number.isFinite(o)?o:.5))}function l(o,h){const y=new Ke(o);return x=y,y.setCells(w()),[{compute:T=>y.compute(T),render:T=>y.render(T),dispose:()=>{y.dispose(),x===y&&(x=null)}},...bt(o)]}Ne(()=>{var o;(o=m(i))==null||o.evidence.length,x==null||x.setCells(w())});const n=["What port does the dev server use?","Should I enable prefix caching with vLLM?","How does FSRS-6 trust scoring work?","Why did the benchmark score drop after the parser change?"];async function p(){const o=m(s).trim();if(!(!o||m(u))){I(u,!0),I(f,null),I(i,null),I(d,null);try{const h=await Ye.deepReference(o,20);I(i,et(h),!0)}catch(h){I(f,h instanceof Error?h.message:"Unknown error",!0)}finally{I(u,!1)}}}const v=be(()=>{var y,b;if(m(u))return"Reasoning in progress.";if(m(f))return`Error: ${m(f)}`;if(!m(i))return"Ask a question to trace how Vestige forms a decision from memory.";const o=m(i),h=((y=o.recommended)==null?void 0:y.answer_preview)??"no recommendation";return`Decision trace for "${m(s)}". ${o.evidence.length} evidence memories, ${o.contradictions.length} contradiction${o.contradictions.length===1?"":"s"}, ${o.superseded.length} superseded. Recommendation: ${h}. Confidence ${Math.round((((b=o.recommended)==null?void 0:b.trust_score)??0)*100)} percent.`});function S(o){const h=o.payload;I(d,(h==null?void 0:h.ariaLabel)??(h==null?void 0:h.preview)??`${o.kind} selected`,!0)}function _(o){var h,y;(o.metaKey||o.ctrlKey)&&o.key.toLowerCase()==="k"&&(o.preventDefault(),(h=m(a))==null||h.focus(),(y=m(a))==null||y.select())}Te(()=>{var o;return(o=m(a))==null||o.focus(),window.addEventListener("keydown",_),m(s).trim()||(I(s,"What is the Vestige dashboard direction?"),p()),()=>window.removeEventListener("keydown",_)});var A=Et();Xe("q2v96u",o=>{Le(()=>{Fe.title="Reasoning Theater · Vestige"})});var C=Pe(A);{let o=be(()=>`reasoning-trace:${m(s)||"empty"}`);ze(C,{organ:"reasoning",get seed(){return m(o)},get scene(){return m(i)},passes:l,get loading(){return m(u)},get error(){return m(f)},emptyLabel:"ASK A QUESTION - PRESS CMD+K - WATCH THE DECISION FORM",onpick:S})}var k=j(C,2),M=j(pe(k),2);Oe(M),De(M,o=>I(a,o),()=>m(a));var H=j(M,2);Ue(H,21,()=>n,We,(o,h)=>{var y=_t(),b={};de(()=>{b!==(b=m(h))&&(y.value=(y.__value=m(h))??"")}),ue(o,y)}),ie(H),Be(2),ie(k);var q=j(k,2),K=pe(q,!0);ie(q);var ne=j(q,2);{var N=o=>{var h=wt(),y=pe(h,!0);ie(h),de(()=>ye(y,m(d))),ue(o,h)};Ge(ne,o=>{m(d)&&o(N)})}de(()=>ye(K,m(v))),Ie("submit",k,o=>{o.preventDefault(),p()}),Ve(M,()=>m(s),o=>I(s,o)),ue(t,A),Me()}export{Ft as component};

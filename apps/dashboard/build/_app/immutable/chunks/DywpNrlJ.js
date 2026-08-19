var ei=Object.defineProperty;var ti=(n,e,t)=>e in n?ei(n,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):n[e]=t;var p=(n,e,t)=>ti(n,typeof e!="symbol"?e+"":e,t);import"./Bzak7iHL.js";import{d as fr,s as ne,b as pe,o as ri,e as rt}from"./GD4hRtFg.js";import{p as ct,c as A,r as E,j as D,t as le,a as q,b as dt,f as J,k as hr,h as pr,g as m,u as Ie,o as Oe,d as U,e as Xt,s as P,bg as ii,n as ni}from"./DEZxQDp-.js";import{i as te}from"./Co_hMTTH.js";import{e as kt,a as Re,s as ke,r as ai}from"./Dtd1z3qK.js";import{b as si}from"./BcfZvAA_.js";import{b as oi}from"./Bbd0sF-k.js";import{p as V,s as li,a as ci}from"./CYT7lwRd.js";import{a as wt}from"./DOaVlKeo.js";import{e as di}from"./dTNIuE58.js";import{b as ui}from"./DrqtrZF2.js";import{s as It}from"./D1KsgV-F.js";import{t as fi,d as mr,N as ve,U as xt,F as ae,a as Ne,P as ue,b as Pe,c as Et,L as ie,e as oe,O as hi,D as pi}from"./Ih6ntwL-.js";import{p as mi}from"./CcSRZpDz.js";function gi(n,e){var r;const t=[];for(const s of n){const i=(((r=s.activation_path)!=null&&r.length?s.activation_path:s.retrieved)??[]).filter(e);if(i.length===0)continue;const o=i[i.length-1];t.push({targetId:o,pathIds:i})}return t}function vi(n,e=12){return[...n].sort((t,r)=>r.retention-t.retention||t.id.localeCompare(r.id)).slice(0,e).map(t=>({targetId:t.id,pathIds:[t.id]}))}function bi(n,e,t=5){var s;const r=new Map;for(const a of n){const i=((s=a.activation_path)!=null&&s.length?a.activation_path:a.retrieved)??[];for(const o of new Set(i))e(o)&&r.set(o,(r.get(o)??0)+1)}return[...r.entries()].map(([a,i])=>({id:a,recalls:i})).sort((a,i)=>i.recalls-a.recalls||a.id.localeCompare(i.id)).slice(0,t)}class yi{constructor(e,t={}){p(this,"bridge");p(this,"items",[]);p(this,"cursor",0);p(this,"ticks",0);p(this,"nextTick",0);p(this,"intervalFrames");p(this,"enabled",!0);p(this,"started",!1);this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}const t=this.items[this.cursor%this.items.length];this.cursor++;const r=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(r?0:30)}}var _i=J('<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>'),wi=J('<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>'),xi=J('<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>'),Bi=J('<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>');function Pi(n,e){ct(e,!0);let t=V(e,"demoMode",3,"recall-path"),r=V(e,"seed",3,"vestige-observatory-v1"),s=V(e,"nodeCount",3,0),a=V(e,"edgeCount",3,0),i=V(e,"centerId",3,""),o=V(e,"frameCount",3,0),c=V(e,"fpsEstimate",3,0),l=V(e,"freezeFrame",3,null);V(e,"loading",3,!1),V(e,"error",3,"");function f(){const W=new URLSearchParams({demo:t(),seed:r()});l()!==null&&W.set("frame",String(l()));const Q=`${window.location.origin}${ui}/observatory?${W.toString()}`;navigator.clipboard.writeText(Q).catch(()=>{})}var u=Bi(),h=A(u),d=A(h),g=A(d),_=A(g,!0);E(g);var x=D(g,2),S=A(x);E(x),E(d);var v=D(d,2),M=A(v),R=A(M);E(M);var O=D(M,2);{var N=W=>{var Q=_i(),w=A(Q);E(Q),le(L=>ne(w,`center=${L??""}`),[()=>i().slice(0,8)]),q(W,Q)};te(O,W=>{i()&&W(N)})}E(v);var X=D(v,2),C=A(X),j=A(C);E(C);var se=D(C,2);{var $=W=>{var Q=wi();q(W,Q)},ce=W=>{var Q=xi(),w=A(Q);E(Q),le(()=>ne(w,`${c()??""}fps`)),q(W,Q)};te(se,W=>{l()!==null?W($):c()>0&&W(ce,1)})}var re=D(se,2);E(X),E(h),E(u),le((W,Q)=>{ne(_,t()),ne(S,`seed=${W??""}${r().length>12?"…":""}`),ne(R,`${s()??""} nodes · ${a()??""} edges`),ne(j,`frame: ${Q??""}`)},[()=>r().slice(0,12),()=>String(o()).padStart(3," ")]),pe("click",re,f),q(n,u),dt()}fr(["click"]);var Si=J('<div class="active-label svelte-8n8iia"> </div>'),ki=J("<div></div>"),Ii=J('<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>');function Ri(n,e){ct(e,!0);let t=V(e,"steps",19,()=>[]),r=V(e,"frame",3,0),s=V(e,"loopFrames",3,720);const a=u=>u/s()*100;function i(u,h){const d=h-u;return d<-14||d>90?0:d<0?1+d/14:1-d/90}let o=Ie(()=>{let u="",h=.15;for(const d of t()){const g=i(d.beatFrame,r());g>h&&(h=g,u=d.label)}return u});var c=hr(),l=pr(c);{var f=u=>{var h=Ii(),d=A(h);{var g=v=>{var M=Si(),R=A(M,!0);E(M),le(()=>ne(R,m(o))),q(v,M)};te(d,v=>{m(o)&&v(g)})}var _=D(d,2),x=A(_);kt(x,17,t,v=>v.beatFrame,(v,M)=>{var R=ki();let O;le((N,X,C)=>{O=Re(R,1,"tick svelte-8n8iia",null,O,N),It(R,`left: ${X??""}%; opacity: ${C??""}`),ke(R,"title",m(M).label)},[()=>({hot:i(m(M).beatFrame,r())>0,backward:m(M).kind===1}),()=>a(m(M).beatFrame),()=>.45+.55*i(m(M).beatFrame,r())]),q(v,R)});var S=D(x,2);E(_),E(h),le(v=>It(S,`left: ${v??""}%`),[()=>a(r())]),q(u,h)};te(l,u=>{t().length>0&&u(f)})}q(n,c),dt()}var Ei=J('<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>');function Qt(n,e){ct(e,!0);let t=V(e,"frame",3,0),r=V(e,"fadeWindow",19,()=>[600,620,705,719]),s=V(e,"tone",3,"triumph");const a=(f,u,h)=>{const d=Math.min(1,Math.max(0,(h-f)/(u-f)));return d*d*(3-2*d)};let i=Ie(()=>a(r()[0],r()[1],t())*(1-a(r()[2],r()[3],t())));var o=hr(),c=pr(o);{var l=f=>{var u=Ei();let h;var d=A(u),g=A(d,!0);E(d);var _=D(d,2),x=A(_,!0);E(_);var S=D(_,2),v=A(S,!0);E(S),E(u),le(()=>{h=Re(u,1,"verdict svelte-ssd7yu",null,h,{quarantine:s()==="quarantine"}),It(u,`opacity: ${m(i)??""}`),ne(g,e.verdict.headline),ne(x,e.verdict.causeLabel),ne(v,e.verdict.receipt)}),q(f,u)};te(c,f=>{m(i)>.001&&f(l)})}q(n,o),dt()}function Ai(n,e,t,r){const s=1/Math.tan(n/2),a=1/(t-r),i=new Float32Array(16);return i[0]=s/e,i[5]=s,i[10]=r*a,i[11]=-1,i[14]=r*t*a,i}function Ci(n,e,t){const[r,s,a]=n;let i=r-e[0],o=s-e[1],c=a-e[2],l=Math.hypot(i,o,c)||1;i/=l,o/=l,c/=l;let f=t[1]*c-t[2]*o,u=t[2]*i-t[0]*c,h=t[0]*o-t[1]*i;l=Math.hypot(f,u,h)||1,f/=l,u/=l,h/=l;const d=o*h-c*u,g=c*f-i*h,_=i*u-o*f,x=new Float32Array(16);return x[0]=f,x[1]=d,x[2]=i,x[4]=u,x[5]=g,x[6]=o,x[8]=h,x[9]=_,x[10]=c,x[12]=-(f*r+u*s+h*a),x[13]=-(d*r+g*s+_*a),x[14]=-(i*r+o*s+c*a),x[15]=1,x}function Mi(n,e){const t=new Float32Array(16);for(let r=0;r<4;r++)for(let s=0;s<4;s++)t[r*4+s]=n[s]*e[r*4]+n[4+s]*e[r*4+1]+n[8+s]*e[r*4+2]+n[12+s]*e[r*4+3];return t}function Zt(n,e,t,r=.35){const s=n*Math.PI*2,a=[Math.sin(s)*t,t*r,Math.cos(s)*t],i=Ai(50*Math.PI/180,e,.1,4e3),o=Ci(a,[0,0,0],[0,1,0]);let c=-a[0],l=-a[1],f=-a[2],u=Math.hypot(c,l,f)||1;c/=u,l/=u,f/=u;let h=l*0-f*1,d=f*0-c*0,g=c*1-l*0;u=Math.hypot(h,d,g)||1,h/=u,d/=u,g/=u;const _=d*f-g*l,x=g*c-h*f,S=h*l-d*c;return{viewProj:Mi(i,o),right:[h,d,g],up:[_,x,S],eye:a}}function Fi(n){return n>=.7?"active":n>=.4?"dormant":n>=.1?"silent":"unavailable"}const Li={active:"#10b981",dormant:"#f59e0b",silent:"#8b5cf6",unavailable:"#6b7280"},Bt={aha:"#FFD700",confusion:"#EF4444",failure:"#9CA3AF"};function Gi(n){const e=new Set((n.tags??[]).map(t=>t.toLowerCase()));return e.has("aha")?Bt.aha:e.has("confusion")||e.has("weak-spot")?Bt.confusion:e.has("failure")||e.has("guardrail")?Bt.failure:null}function Jt(n){const e=/^#?([0-9a-fA-F]{6})$/.exec(n.trim());if(!e)return[107/255,114/255,128/255];const t=parseInt(e[1],16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}function Di(n){const e=Gi({tags:n.tags});return Jt(e||Li[Fi(n.retention)])}function Ti(n){const t=[...n.nodes].sort((i,o)=>i.isCenter!==o.isCenter?i.isCenter?-1:1:i.id<o.id?-1:i.id>o.id?1:0).map((i,o)=>fi(i,o)),r=new Map;for(const i of t)r.set(i.id,i.index);const s=[];for(const i of n.edges){const o=r.get(i.source),c=r.get(i.target);o===void 0||c===void 0||o===c||s.push({sourceIndex:o,targetIndex:c,weight:i.weight,type:i.type})}const a=t.findIndex(i=>i.isCenter);return{nodes:t,edges:s,indexById:r,centerIndex:a<0?0:a}}function gr(n,e,t=120){const r=n.nodes.length,s=new Float32Array(r*ae);for(let a=0;a<r;a++){const i=n.nodes[a],o=a*ae,[c,l,f]=i.isCenter&&n.centerIndex===a?[0,0,0]:mr(a,r,t,e),u=i.isCenter?4.2:1.4+i.retention*1.8;s[o+ve.posRadius+0]=c,s[o+ve.posRadius+1]=l,s[o+ve.posRadius+2]=f,s[o+ve.posRadius+3]=u,s[o+ve.velRetention+3]=i.retention;const[h,d,g]=Di(i);let _=0;i.isCenter&&(_|=Ne.isCenter),i.suppressed&&(_|=Ne.suppressed);const x=new Set(i.tags.map(S=>S.toLowerCase()));x.has("aha")&&(_|=Ne.isAha),(x.has("failure")||x.has("guardrail"))&&(_|=Ne.isFailure),(x.has("confusion")||x.has("weak-spot"))&&(_|=Ne.isConfusion),s[o+ve.colorFlags+0]=h,s[o+ve.colorFlags+1]=d,s[o+ve.colorFlags+2]=g,s[o+ve.colorFlags+3]=_}return{data:s,nodeCount:r}}function $t(n){const e=new Uint32Array(Math.max(1,n.edges.length)*xt);return n.edges.forEach((t,r)=>{e[r*xt]=t.sourceIndex,e[r*xt+1]=t.targetIndex}),e}const zi=`
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

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

// Fossil-light activation band. Recall is not a rainbow screensaver: a living
// memory travels graphite → amber → jade → chalk, with no violet/purple energy
// leaking back into the stage.
fn spectral(w_in: f32) -> vec3<f32> {
	let w = fract(w_in);
	// var, not let: WGSL only allows dynamic indexing through a reference.
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.48, 0.22, 0.08), // fossil amber
		vec3<f32>(0.82, 0.58, 0.24), // warmed phosphor
		vec3<f32>(0.30, 0.74, 0.53), // retained jade
		vec3<f32>(0.88, 0.94, 0.82)  // chalk ignition
	);
	let f = w * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	let a = stops[i];
	let b = stops[(i + 1u) % 4u];
	return mix(a, b, frac);
}

// Incoming semantic colors predate Fossil Light and include blue/violet
// states. Keep a trace of that information, but chromatically ground it so the
// field cannot fall back into the old purple-neon visual language.
fn fossil_tone(source: vec3<f32>, retention: f32) -> vec3<f32> {
	let amber = vec3<f32>(0.66, 0.30, 0.10);
	let jade = vec3<f32>(0.30, 0.74, 0.52);
	let retained = smoothstep(0.16, 0.92, clamp(retention, 0.0, 1.0));
	let physical = mix(amber, jade, retained);
	let grounded_source = vec3<f32>(
		clamp(source.r, 0.0, 1.0),
		max(clamp(source.g, 0.0, 1.0), clamp(source.b, 0.0, 1.0) * 0.70),
		min(clamp(source.b, 0.0, 1.0), clamp(source.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded_source, 0.14);
}

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	// Per-instance constants: flat interpolation guarantees the flag bit
	// field survives the raster stage bit-exact (no barycentric rounding).
	@location(1) @interpolate(flat) color: vec3<f32>,
	// x retention, y flags (bit field as f32), z recall intensity, w radius
	@location(2) @interpolate(flat) misc: vec4<f32>,
	// Per-demo choreography lanes (demo.y, demo.z, demo.w), gated by demo_id:
	// rescue (2) searchlight/wave/shock, forgetting-horizon (3) fade-and-fall,
	// firewall (4) flare-membrane/shock. Each demo's choreography pass is the
	// ONLY writer of its lanes, and every gated term below is an exact no-op
	// when its lane is 0.0 — other demos stay pixel-identical.
	@location(3) @interpolate(flat) demo_yzw: vec3<f32>,
};

// Quad corners for two triangles (vertex_index 0..5).
const CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0,  1.0)
);

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.node_count)) {
		// degenerate — clipped away
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let node = nodes[ii];
	let corner = CORNERS[vi];

	// Breath: halo geometry swells ~6% on the global pulse (§7.2), and the
	// center memory breathes a touch deeper — a heartbeat, not a strobe.
	let flags = u32(node.color_flags.w);
	let is_center = (flags & 1u) != 0u;
	var breath = 1.0 + 0.06 * params.pulse;
	if (is_center) {
		breath = 1.0 + 0.12 * params.pulse;
	}

	// Sprite spans ~3.2× the core radius so the halo has room to feather out.
	// Recall activation swells the sprite — the wavefront physically blooms.
	// Per-demo choreography lanes swell it too, gated by demo_id so each
	// demo's grammar can never leak into another (lanes are 0.0 elsewhere,
	// and the gate makes the no-op structural, not just numerical).
	let recall = node.demo.x;
	let dy = node.demo.y;
	let dz = node.demo.z;
	let dw = node.demo.w;
	// The firewall grammar fires for the deterministic demo (demo_id==4) AND for
	// a LIVE contradiction/suppression event (live_kind==1). Both write the same
	// demo lanes (firewall.wgsl), so the visual reads identically either way.
	let firewall_active = params.demo_id == 4.0 || params.live_kind == 1.0;
	var lane_swell = 0.0;
	if (params.demo_id == 2.0) {
		// salience-rescue: searchlight pop, wave shiver, shock bloom.
		lane_swell = 0.5 * dy + 0.25 * dz + 0.9 * dw;
	} else if (firewall_active) {
		// firewall: intrusion flare pop (band (0..1]), membrane presence
		// (band [2.6..2.9] via the range gate), crimson shock bloom.
		lane_swell = 0.35 * min(dy, 1.0) + 0.3 * smoothstep(1.5, 2.2, dy) + 0.55 * dw;
	}
	// forgetting-horizon (demo 3): VISUAL displacement toward the horizon —
	// down and away from the field axis, ~40.5 units at dz = 1 — plus a
	// shrink. pos_radius is NEVER written (the force sim owns positions);
	// drift is pure of demo.z, so ?frame=N capture stays exact. CPU mirror:
	// forgetting-plan.ts horizonDrift().
	var horizon_scale = 1.0;
	var drift = vec3<f32>(0.0);
	if (params.demo_id == 3.0) {
		let dzc = clamp(dz, 0.0, 1.0);
		horizon_scale = 1.0 - 0.35 * dzc;
		if (dz > 0.0) {
			let p = node.pos_radius.xyz;
			let r_xz = max(length(p.xz), 0.001);
			let away = vec3<f32>(p.x / r_xz, 0.0, p.z / r_xz);
			drift = dzc * (vec3<f32>(0.0, -34.0, 0.0) + away * 22.0);
		}
	}
	// FOSSIL LIGHT existence mask — live retention of exactly 0 means "not yet
	// born at the scrubbed instant" (fsrs.ts reserves 0.0 as the unborn
	// sentinel; existing memories floor at 0.001). Collapsing the sprite to
	// zero size pops the memory out of the field when the chrono crosses its
	// birthday — cheaper and cleaner than a fragment discard.
	let exists = step(0.0005, node.vel_retention.w);
	let half_size = node.pos_radius.w * 3.2 * breath * (1.0 + 0.9 * recall)
		* (1.0 + lane_swell) * horizon_scale * exists;
	let world = node.pos_radius.xyz + drift
		+ camera.right.xyz * corner.x * half_size
		+ camera.up.xyz * corner.y * half_size;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;
	out.color = node.color_flags.rgb;
	out.misc = vec4<f32>(node.vel_retention.w, node.color_flags.w, node.demo.x, node.pos_radius.w);
	out.demo_yzw = vec3<f32>(dy, dz, dw);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let retention = in.misc.x;
	let flags = u32(in.misc.y);
	let suppressed = (flags & 2u) != 0u;
	let is_center = (flags & 1u) != 0u;

	// SOMATIC PHOTOMETRY — retention is consolidation, not generic bloom.
	// High-retention memories form a concentrated bright soma; their neurites
	// remain deliberately dim. A forward Chrono projection makes weak memories
	// scatter into the field instead of multiplying the whole halo's brightness.
	let consolidated = smoothstep(0.04, 0.96, clamp(retention, 0.0, 1.0));
	let forward_age = clamp(max(params.projection_days, 0.0) / 120.0, 0.0, 1.0);
	let depth_scatter = (1.0 - consolidated) * (0.28 + 0.72 * forward_age);
	let soma = exp(-d * d * mix(34.0, 17.0, consolidated));
	let halo = pow(max(1.0 - d, 0.0), 3.4);
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_count = 5.0 + floor(fract(in.misc.w * 0.173) * 3.0);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * branch_count + in.misc.w * 1.91));
	let branch_gate = pow(branch_wave, 18.0);
	let branch_band = smoothstep(0.16, 0.38, d) * (1.0 - smoothstep(0.72, 0.96, d));
	let neurites = branch_gate * branch_band * (0.035 + 0.12 * consolidated)
		* (1.0 - depth_scatter * 0.72);
	let scattered_tissue = halo * (0.015 + 0.11 * depth_scatter) * (0.82 + 0.18 * params.pulse);
	let tone = fossil_tone(in.color, retention);
	let soma_tone = mix(tone, vec3<f32>(0.90, 0.96, 0.84), consolidated * 0.48);
	var color = soma_tone * soma * (0.34 + 0.98 * consolidated)
		+ tone * neurites
		+ tone * scattered_tissue;

	// The anchor can be legible without becoming a fake sun. A suppressed memory
	// is intentionally a cold, near-dark scar: in an additive pass it cannot
	// subtract light yet, but it no longer emits the field's normal luminance.
	if (is_center) {
		color = color * 1.32;
	}
	if (suppressed) {
		let scar_ring = smoothstep(0.66, 0.78, d) * (1.0 - smoothstep(0.80, 0.92, d));
		color = color * 0.055 + vec3<f32>(0.22, 0.10, 0.045) * scar_ring * 0.10;
	}

	// Forgetting-horizon (demo 3): multiplicative dim toward near-black as
	// demo.z rises. Floor 0.06 — never fully gone, always retrievable. Sits
	// BEFORE the recall block so a rescued memory's ignition burns through
	// the fade. demo_yzw.y carries demo.z (vec3 = y/z/w lanes).
	if (params.demo_id == 3.0) {
		color = color * mix(1.0, 0.06, clamp(in.demo_yzw.y, 0.0, 1.0));
	}

	// Recall activation — GCaMP calcium-imaging emission. The intensity lane
	// (simulate.wgsl recall_sim) is now a real biexponential calcium transient;
	// the COLOR here matches what you see under a two-photon scope: a green
	// fluorescence core, a lingering yellow-green ember through the slow decay
	// tail, and a white-hot pinpoint only at the instant of the spike. The
	// traveling wavefront still rides the spectral band so a multi-hop causal
	// recall reads as a wave, but each node that fires flashes like a neuron.
	// jGCaMP green ~ (0.16, 1.0, 0.42); saturated re-fires (recall > 1) push
	// toward white-hot the way an over-driven indicator clips.
	let recall = in.misc.z;
	if (recall > 0.001) {
		let hot = clamp(recall, 0.0, 1.0);                    // spike peak → 1
		let ember = clamp(recall, 0.0, 1.0);                  // afterglow presence
		// GCaMP fluorophore green, warming to yellow-green as the transient
		// saturates (nonlinear summation on rapid re-fire).
		let gcamp = mix(vec3<f32>(0.16, 1.00, 0.42), vec3<f32>(0.62, 1.00, 0.30), clamp(recall - 0.6, 0.0, 1.0));
		// The spectral band survives as the traveling-wave shimmer, but dialed
		// under the calcium green so the biology reads first.
		let band = spectral(0.1 + params.loop_phase + d * 0.35);
		let activation = (gcamp * (soma * 1.85 + halo * 1.05) + band * 0.28 * halo) * ember;
		// White-hot pinpoint ONLY at the fast spike (soma core × hot), so the
		// ignition punches and the ember stays green.
		let flash = vec3<f32>(1.0, 1.0, 0.94) * soma * hot * 0.6;
		color = color + activation + flash;
	}

	// Per-demo choreography lanes — gated by demo_id AND on nonzero values so
	// every other demo is pixel-unchanged (each demo's pass is the only
	// writer of its lanes, and lanes are exactly 0.0 everywhere else).
	if (params.demo_id == 2.0) {
		if (in.demo_yzw.x > 0.001) {
			// Searchlight: cold clinical white — unmistakably NOT the spectral grammar.
			color = color + vec3<f32>(0.82, 0.90, 1.00) * in.demo_yzw.x * (soma * 1.8 + halo * 0.7);
		}
		if (in.demo_yzw.y > 0.001) {
			// Interrogation shimmer: icy spectral strobe as the wave scrubs the past.
			color = color + spectral(0.55 + params.loop_phase) * in.demo_yzw.y * (soma * 0.9 + halo * 0.5)
				+ vec3<f32>(1.0) * soma * in.demo_yzw.y * 0.2;
		}
		if (in.demo_yzw.z > 0.001) {
			// Detonation: crimson blaze + warm-white pinpoint.
			color = color + vec3<f32>(1.00, 0.16, 0.10) * in.demo_yzw.z * (soma * 1.9 + halo * 1.1)
				+ vec3<f32>(1.0, 0.85, 0.8) * soma * in.demo_yzw.z * 0.4;
		}
	} else if (params.demo_id == 4.0 || params.live_kind == 1.0) {
		// firewall: demo.y carries TWO value bands — intrusion flare (0..1]
		// and membrane [2.6..2.9] — separated by range, one lane. demo.w is
		// the crimson shock rim / sever blink. (demo_yzw = y/z/w lanes.)
		let fy = in.demo_yzw.x;
		let fw = in.demo_yzw.z;
		// Intrusion flare: sickly green-white — a hue deliberately OUTSIDE
		// both the FSRS palette and the thin-film band. Continuous across the
		// band boundary (fades out as fy climbs toward the membrane band).
		let flare = min(fy, 1.0) * (1.0 - smoothstep(1.0, 1.8, fy));
		if (flare > 0.001) {
			color = color + vec3<f32>(0.62, 1.00, 0.55) * flare * (soma * 1.7 + halo * 0.9)
				+ vec3<f32>(0.90, 1.00, 0.85) * soma * flare * 0.5;
		}
		// Membrane: quarantine ring at d ≈ 0.75 with fresnel-ish falloff —
		// green body, crimson edge. exp(-q·q) squares by multiplication and
		// the pow base is clamped ≥ 0 (no pow(neg) anywhere).
		let mw = smoothstep(1.5, 2.2, fy);
		if (mw > 0.001) {
			let q = (d - 0.75) * 9.0;
			let ring = exp(-q * q);
			let fresnel = pow(clamp(d / 0.75, 0.0, 1.0), 3.0);
			let ring_col = mix(vec3<f32>(0.55, 1.00, 0.60), vec3<f32>(1.00, 0.20, 0.16),
				smoothstep(0.72, 0.92, d));
			color = color + ring_col * ring * fresnel * mw * 1.4;
		}
		// Shockwave: crimson RIM as the front passes (a rim, not a blaze).
		if (fw > 0.001) {
			let rim = smoothstep(0.45, 0.8, d) * (1.0 - smoothstep(0.85, 1.0, d));
			color = color + vec3<f32>(1.00, 0.14, 0.10) * rim * fw * 1.5
				+ vec3<f32>(1.00, 0.60, 0.50) * soma * fw * 0.15;
		}
	}

	// Additive target (src=one, dst=one): alpha is ignored, light accumulates.
	return vec4<f32>(color * params.brightness, 1.0);
}
`,Ui=`
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

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// x source index, y target index, z beat frame, w kind (0 recall, 1 backward)
@group(0) @binding(2) var<storage, read> path: array<vec4<u32>>;
// x source index, y target node index (Increment 7: force simulation edges)
@group(0) @binding(3) var<storage, read> edges: array<vec2<u32>>;
// v2.3 living field — per-node LIVE retrievability (real FSRS curve, recomputed
// on the CPU by the LiveBridge). One f32 per node. read to overwrite
// vel_retention.w so render-nodes dims each memory on its true forgetting curve.
@group(0) @binding(4) var<storage, read> live_retention: array<f32>;

// --- Force-simulation helpers (Increment 7) ---

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
	let l = length(v);
	if (l < 0.0001) { return vec3<f32>(0.0); }
	return v / l;
}

fn clamp_len(v: vec3<f32>, hi: f32) -> vec3<f32> {
	let l = length(v);
	if (l > hi && l > 0.0001) { return v * (hi / l); }
	return v;
}

@compute @workgroup_size(64)
fn recall_sim(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}

	let frame = params.frame;
	var intensity = 0.0;

	var node = nodes[i];
	let flags = u32(node.color_flags.w);
	let is_center = (flags & 1u) != 0u;

	// --- GCaMP calcium-transient recall kinetics ---------------------------
	// A retrieved memory does NOT ease-out linearly; it fires like a neuron
	// under two-photon calcium imaging. Each recall beat is one calcium
	// transient with a biexponential envelope: near-instant rise, MUCH slower
	// decay (jGCaMP8/GCaMP6 kinetics, Nature 2023 s41586-023-05828-9). Empirical
	// asymmetry is ~1:30 rise:decay; at the observatory's 60fps loop clock that
	// is a ~3-frame time-to-peak and a ~90-frame decay tail. tau_decay is
	// MODULATED BY REAL FSRS RETENTION (vel_retention.w): a weak, decaying
	// memory's ember fades fast; a strongly-retained one glows on. The
	// discipline test holds — swap the retention for noise and the afterglow
	// lengths scramble.
	let ret = clamp(node.vel_retention.w, 0.0, 1.0);
	let tau_rise = 3.0;                        // fast fluorescence spike (~50ms)
	let tau_decay = 55.0 + 70.0 * ret;         // 55..125 frames — retention holds the glow
	// SEAM FADE — the GCaMP tail decays slowly (tau_decay up to 125f), and the
	// last story beat lands at ~bf=480, so at the last loop frame 719 a hot
	// node still glows ~0.15 and would snap to 0 at frame 0 (dt goes negative):
	// a visible pop every 12s. Force the whole recall envelope to zero over the
	// final ~30 frames so the loop is seamless by construction (restores the old
	// smoothstep guarantee that the calcium version broke).
	let seam = 1.0 - smoothstep(688.0, 718.0, frame);
	let steps = u32(params.path_count);
	for (var s = 0u; s < steps; s = s + 1u) {
		let step = path[s];
		let bf = f32(step.z);

		if (step.y == i) {
			// Arrival transient: analytic biexponential (calcium indicator ODE),
			// not a tween. Clamp dt>=0 BEFORE the exponentials so the pre-beat
			// case is a cheap, finite 0.0 (select() evaluates both arms; the old
			// discarded true-arm computed exp(+large)=+Inf for future beats).
			let dt = max(frame - bf, 0.0);
			let g = (1.0 - exp(-dt / tau_rise)) * exp(-dt / tau_decay);
			// NONLINEAR SUMMATION: rapid re-fires stack supralinearly (a hot,
			// over-recalled memory saturates like an over-driven indicator)
			// instead of the old max(). Saturating add keeps it bounded/HDR-safe.
			intensity = intensity + g * (1.0 - 0.55 * intensity);
		}
		if (step.x == i && step.x != step.y) {
			// Departure: the source shimmers briefly as the wave leaves it —
			// a small pre-transient before its own arrival glow.
			let dt = max(frame - (bf - 32.0), 0.0);
			let g = (1.0 - exp(-dt / tau_rise)) * exp(-dt / (tau_decay * 0.45));
			intensity = intensity + g * 0.4 * (1.0 - 0.55 * intensity);
		}
	}
	intensity = clamp(intensity, 0.0, 1.35) * seam;

	// Write recall intensity (existing behavior preserved).
	node.demo.x = intensity;

	// v2.3 LIVE FSRS decay — overwrite retention with the real forgetting-curve
	// value the LiveBridge computed for this node on the CPU. This is the #1
	// moat: render-nodes already dims by vel_retention.w (line ~183), so writing
	// the true retrievability here makes every memory visibly forget on its own
	// curve. Guarded so a graph with no live-decay data (all zeros) keeps its
	// static snapshot instead of collapsing to black.
	if (i < arrayLength(&live_retention)) {
		let lr = live_retention[i];
		// FOSSIL LIGHT: lr == 0.0 is the honest "not yet born at the scrubbed
		// instant" sentinel and MUST propagate so the render mask can pop the
		// memory out of existence. Living memories are floored at 0.001 by the
		// CPU (fsrs.ts/node-renderer.ts), so gating on >= 0.0 never blanks a
		// real field; the old strictly-positive guard predates the floor and
		// blocked unbirth.
		if (lr >= 0.0) {
			node.vel_retention = vec4<f32>(node.vel_retention.xyz, lr);
		}
	}

	// --- Increment 7: force simulation ---

	// Capture mode (params.capture_mode == 1.0): skip physics integration
	// entirely. The storage-buffer state stays frozen at initial upload
	// values, making same URL + frame → identical pixels (spec §4 Inc 9).
	if (params.capture_mode == 0.0) {
		// 7B: center anchor — center node never moves.
		// (WGSL forbids swizzle stores — reconstruct the vec4, preserving .w.)
		if (is_center) {
			node.pos_radius = vec4<f32>(0.0, 0.0, 0.0, node.pos_radius.w);
			node.vel_retention = vec4<f32>(0.0, 0.0, 0.0, node.vel_retention.w);
			nodes[i] = node;
			return;
		}

		let pos = node.pos_radius.xyz;
		var force = vec3<f32>(0.0);

		// 7C: edge springs — scan existing edgeBuffer, no atomics.
		for (var e = 0u; e < u32(params.edge_count); e = e + 1u) {
			let edge = edges[e];
			var other_idx = 0xffffffffu;
			if (edge.x == i) { other_idx = edge.y; }
			if (edge.y == i) { other_idx = edge.x; }
			if (other_idx != 0xffffffffu && other_idx < u32(params.node_count)) {
				let other = nodes[other_idx].pos_radius.xyz;
				let delta = other - pos;
				let dist = max(length(delta), 0.001);
				let dir = delta / dist;
				let stretch = dist - 34.0;
				force = force + dir * stretch * 0.00055;
			}
		}

		// 7D: soft repulsion (only ≤ 500 nodes for performance).
		if (u32(params.node_count) <= 500u) {
			for (var j = 0u; j < u32(params.node_count); j = j + 1u) {
				if (j == i) { continue; }
				let other = nodes[j].pos_radius.xyz;
				let delta = pos - other;
				let d2 = max(dot(delta, delta), 9.0);
				force = force + safe_normalize(delta) * (7.5 / d2);
			}
		}

		// Gentle centering: keeps the field in frame without crushing it.
		force = force + (-pos) * 0.0008;

		// v2.3 DREAM STORM — while the real dream pipeline streams (live_kind ==
		// 2 == LIVE_KIND.dreamStorm), the field enters a metabolic consolidation
		// storm: damping loosens (springs overshoot, clusters slosh together as
		// new ConnectionDiscovered edges are appended) and a deterministic
		// turbulence rides live_energy. Pure of node index + live_frame, so no
		// wall clock — the storm is a function of the real event envelope. At
		// rest (energy 0) both terms vanish → the field is byte-identical.
		var damping = 0.88;
		if (params.live_kind == 2.0) {
			let e = clamp(params.live_energy, 0.0, 1.4);
			damping = 0.88 + 0.09 * e; // up to ~0.97 — longer, sloshier settling
			// Curl-free deterministic jitter: phase from node index + live_frame.
			let ph = f32(i) * 0.61803 + params.live_frame * 0.05;
			let jitter = vec3<f32>(sin(ph * 6.2831), sin(ph * 4.7123 + 1.3), sin(ph * 5.318 + 2.1));
			force = force + jitter * (0.006 * e);
		}

		// 7B: velocity damping + cap, then position integration.
		var vel = node.vel_retention.xyz;
		vel = (vel + force) * damping;
		vel = clamp_len(vel, 0.42);
		node.vel_retention = vec4<f32>(vel, node.vel_retention.w);
		node.pos_radius = vec4<f32>(pos + vel, node.pos_radius.w);
	}
	// When capture_mode (params.capture_mode == 1.0), node is NOT written back —
	// the storage buffer retains its initial upload values, guaranteeing
	// deterministic pixels for the same frame index.
	nodes[i] = node;
}
`,vr=`
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

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;
// x source index, y target index, z beat frame, w kind (0 recall, 1 backward)
@group(0) @binding(3) var<storage, read> path: array<vec4<u32>>;

// Same thin-film band as the node shader (§7.1).
fn spectral(w_in: f32) -> vec3<f32> {
	let w = fract(w_in);
	// var, not let: WGSL only allows dynamic indexing through a reference.
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.20, 0.28, 0.95),
		vec3<f32>(0.20, 0.85, 0.90),
		vec3<f32>(0.45, 1.00, 0.72),
		vec3<f32>(0.85, 0.45, 1.00)
	);
	let f = w * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	let a = stops[i];
	let b = stops[(i + 1u) % 4u];
	return mix(a, b, frac);
}

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	// x: t along segment (0 source → 1 target), y: side (-1..1)
	@location(0) uv: vec2<f32>,
	// x: beat frame, y: kind, z: segment visible (0 skips degenerate steps)
	// Per-instance constant — flat keeps it bit-exact through the raster.
	@location(1) @interpolate(flat) beat: vec3<f32>,
};

// (t, side) corners for two triangles of the ribbon.
const RIBBON = array<vec2<f32>, 6>(
	vec2<f32>(0.0, -1.0),
	vec2<f32>(1.0, -1.0),
	vec2<f32>(1.0,  1.0),
	vec2<f32>(0.0, -1.0),
	vec2<f32>(1.0,  1.0),
	vec2<f32>(0.0,  1.0)
);

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.path_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		out.beat = vec3<f32>(0.0);
		return out;
	}

	let step = path[ii];
	let src = nodes[step.x];
	let dst = nodes[step.y];
	let corner = RIBBON[vi];

	// Degenerate (origin beat: source == target) — emit nothing visible.
	if (step.x == step.y) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		out.beat = vec3<f32>(0.0);
		return out;
	}

	let a = camera.view_proj * vec4<f32>(src.pos_radius.xyz, 1.0);
	let b = camera.view_proj * vec4<f32>(dst.pos_radius.xyz, 1.0);

	// NDC-space perpendicular for constant screen width.
	let ndc_a = a.xy / max(a.w, 0.0001);
	let ndc_b = b.xy / max(b.w, 0.0001);
	var dir = ndc_b - ndc_a;
	let dlen = max(length(dir), 0.0001);
	dir = dir / dlen;
	let perp = vec2<f32>(-dir.y, dir.x);

	// Ribbon half-width in NDC (aspect-corrected), ~2.5 px on a 900px-tall view.
	let px = 2.5 / max(params.viewport_h, 1.0) * 2.0;
	let width = vec2<f32>(px * (params.viewport_h / max(params.viewport_w, 1.0)), px);

	let base = mix(a, b, corner.x);
	let offset = perp * width * corner.y * base.w;
	out.clip = vec4<f32>(base.xy + offset, base.zw);
	out.uv = vec2<f32>(corner.x, corner.y);
	out.beat = vec3<f32>(f32(step.z), f32(step.w), 1.0);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	if (in.beat.z < 0.5) {
		discard;
	}

	let frame = params.frame;
	let bf = in.beat.x;
	let t = in.uv.x;

	// Wave departs 45 frames before the beat and lands exactly on it.
	let progress = clamp((frame - (bf - 45.0)) / 45.0, 0.0, 1.0);
	// Nothing before departure; trail lingers ~90 frames after arrival.
	let live = smoothstep(bf - 46.0, bf - 44.0, frame)
		* (1.0 - smoothstep(bf + 40.0, bf + 90.0, frame));
	if (live <= 0.001) {
		discard;
	}

	// The light packet: gaussian around the wavefront position.
	let dwave = (t - progress) * 14.0;
	let packet = exp(-dwave * dwave);

	// Fading trail behind the packet — provenance stays visible a beat.
	var trail = 0.0;
	if (t < progress) {
		trail = (1.0 - (progress - t)) * 0.22;
	}

	// Feather across the ribbon width.
	let across = 1.0 - abs(in.uv.y);
	let profile = across * across;

	// Backward/contradiction hops burn hotter into the magenta rim (§7.4).
	// Hue drifts one full spectral cycle per loop (seamless at the wrap).
	// Kind 2 (salience-rescue probe): a gray failing beam — vector search
	// visibly probing lookalikes and coming back empty. Kinds 0/1 unchanged.
	var band = spectral(0.15 + t * 0.35 + params.loop_phase);
	var packet_white = 0.35;
	if (in.beat.y > 1.5) {
		band = vec3<f32>(0.62, 0.66, 0.72);
		packet_white = 0.18;
	} else if (in.beat.y > 0.5) {
		band = mix(band, vec3<f32>(1.0, 0.25, 0.45), 0.55);
	}

	let energy = (packet * 1.6 + trail) * profile * live;
	let color = band * energy + vec3<f32>(1.0) * packet * profile * live * packet_white;
	return vec4<f32>(color * params.brightness, 1.0);
}
`;function Oi(n){return 60+n*60}function br(n,e,t=8,r={}){var f;const s=[...n.nodes].sort((u,h)=>u.id<h.id?-1:u.id>h.id?1:0),a=[...n.edges].sort((u,h)=>{const d=`${u.source}\0${u.target}\0${u.type}`,g=`${h.source}\0${h.target}\0${h.type}`;return d<g?-1:d>g?1:0}),i=r.centerId??n.center_id,o=mi(s,a,i,t,{preferCausal:r.preferCausal}),c=[];for(let u=0;u<o.beats.length;u++){const h=o.beats[u],d=e.indexById.get(h.nodeId);if(d===void 0)continue;const g=u>0?o.beats[u-1].nodeId:h.nodeId,_=e.indexById.get(g)??d,x=(((f=h.viaEdge)==null?void 0:f.type)??"").toLowerCase(),S=x==="causal"||x.includes("causal"),v=h.kind==="contradiction"||S;c.push({sourceIndex:_,targetIndex:d,beatFrame:Oi(u),kind:v?ue.backwardCause:ue.recall,beatKind:h.kind,nodeId:h.nodeId,label:h.node.label})}const l=new Uint32Array(Math.max(1,c.length)*Pe);return c.forEach((u,h)=>{l[h*Pe]=u.sourceIndex,l[h*Pe+1]=u.targetIndex,l[h*Pe+2]=u.beatFrame,l[h*Pe+3]=u.kind}),{data:l,steps:c,path:o}}const Ni=24,er=300,Fe=128;class ji{constructor(e){p(this,"engine");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"cameraBuffer",null);p(this,"nodeBuffer",null);p(this,"edgeBuffer",null);p(this,"cameraData",new Float32Array(Ni));p(this,"nodeCount",0);p(this,"simPipeline",null);p(this,"simBindGroup",null);p(this,"pathBuffer",null);p(this,"liveRetentionBuffer",null);p(this,"pathPipeline",null);p(this,"pathBindGroup",null);p(this,"pathStepCount",0);p(this,"graph",null);p(this,"pathSteps",[]);this.engine=e,e.addPass(this)}upload(e,t,r){var d,g,_,x;const s=this.engine.gpuDevice;if(!s)return;const a=(r==null?void 0:r.recallPath)??!0,i=Ti(e);this.graph=i;const o=new Et({seed:t}),{data:c,nodeCount:l}=gr(i,o.state.rng);this.nodeCount=l,(d=this.nodeBuffer)==null||d.destroy(),this.nodeBuffer=s.createBuffer({label:"observatory-node-state",size:Math.max(c.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),s.queue.writeBuffer(this.nodeBuffer,0,c.buffer);const f=$t(i);(g=this.edgeBuffer)==null||g.destroy(),this.edgeBuffer=s.createBuffer({label:"observatory-edge-index",size:f.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.edgeBuffer,0,f.buffer);const u=new Float32Array(Math.max(l,4));for(let S=0;S<l;S++)u[S]=Math.max(.001,i.nodes[S].retention);(_=this.liveRetentionBuffer)==null||_.destroy(),this.liveRetentionBuffer=s.createBuffer({label:"observatory-live-retention",size:u.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.liveRetentionBuffer,0,u.buffer);const h=a?br(e,i):{steps:[],data:new Uint32Array(4)};this.pathSteps=h.steps,(x=this.pathBuffer)==null||x.destroy(),this.pathBuffer=s.createBuffer({label:"observatory-path-steps",size:Fe*Pe*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.pathBuffer,0,h.data.buffer,0,Math.min(h.data.byteLength,Fe*Pe*4)),this.pathStepCount=Math.min(this.pathSteps.length,Fe),this.engine.params[2]=l,this.engine.params[3]=i.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||(this.cameraBuffer=s.createBuffer({label:"observatory-camera",size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.createPipeline(s)}setPathSteps(e,t){var a;const r=this.engine.gpuDevice;if(!r)return;this.pathSteps=t;const s=Fe*Pe*4;if(this.pathBuffer&&e.byteLength<=s){this.pathStepCount=Math.min(t.length,Fe),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,Fe),(a=this.pathBuffer)==null||a.destroy(),this.pathBuffer=r.createBuffer({label:"observatory-path-steps",size:s,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,s)),this.engine.params[4]=this.pathStepCount,this.createPipeline(r)}setEdges(e){var s;const t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e;const r=$t(this.graph);(s=this.edgeBuffer)==null||s.destroy(),this.edgeBuffer=t.createBuffer({label:"observatory-edge-index",size:Math.max(r.byteLength,8),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.edgeBuffer,0,r.buffer),this.engine.params[3]=e.length,this.createPipeline(t)}uploadLiveRetention(e){const t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;const r=Math.min(e.length,this.nodeCount);r<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,r*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){const r=e.createShaderModule({label:"observatory-simulate",code:Ui});this.simPipeline=e.createComputePipeline({label:"observatory-recall-sim",layout:"auto",compute:{module:r,entryPoint:"recall_sim"}});const s=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&s.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&s.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:"observatory-recall-sim-bind",layout:this.simPipeline.getBindGroupLayout(0),entries:s})}const t=e.createShaderModule({label:"observatory-render-nodes",code:zi});if(this.pipeline=e.createRenderPipeline({label:"observatory-nodes",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=e.createBindGroup({label:"observatory-nodes-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){const r=e.createShaderModule({label:"observatory-render-path",code:vr});this.pathPipeline=e.createRenderPipeline({label:"observatory-path",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.pathBindGroup=e.createBindGroup({label:"observatory-path-bind",layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}}compute(e){const t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;const r=this.engine.params[6]||1,s=this.engine.params[7]||1,a=this.engine.params[1],i=Zt(a,r/s,er);if(this.cameraData.set(i.viewProj,0),this.cameraData[16]=i.right[0],this.cameraData[17]=i.right[1],this.cameraData[18]=i.right[2],this.cameraData[19]=0,this.cameraData[20]=i.up[0],this.cameraData[21]=i.up[1],this.cameraData[22]=i.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){const o=e.beginComputePass({label:"observatory-recall-sim"});o.setPipeline(this.simPipeline),o.setBindGroup(0,this.simBindGroup),o.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),o.end()}}render(e){!this.pipeline||!this.bindGroup||this.nodeCount===0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){const r=this.engine.gpuDevice;if(!r||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;const s=this.nodeCount*ae*4,a=r.createBuffer({label:"observatory-pick-staging",size:s,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),i=r.createCommandEncoder({label:"observatory-pick-copy"});i.copyBufferToBuffer(this.nodeBuffer,0,a,0,s),r.queue.submit([i.finish()]);let o;try{await a.mapAsync(GPUMapMode.READ),o=new Float32Array(a.getMappedRange().slice(0))}catch{return a.destroy(),null}a.unmap(),a.destroy();const c=this.engine.params[6]||1,l=this.engine.params[7]||1,f=this.engine.params[1],u=Zt(f,c/l,er).viewProj,h=1/Math.tan(50*Math.PI/360);let d=-1,g=1/0;for(let _=0;_<this.nodeCount;_++){const x=_*ae+ve.posRadius,S=o[x],v=o[x+1],M=o[x+2],R=o[x+3],O=u[3]*S+u[7]*v+u[11]*M+u[15];if(O<=0)continue;const N=(u[0]*S+u[4]*v+u[8]*M+u[12])/O,X=(u[1]*S+u[5]*v+u[9]*M+u[13])/O,C=Math.max(R*h/O,.012),j=Math.hypot(N-e,X-t)/C;j<1.6&&j<g&&(g=j,d=_)}return d<0?null:{index:d,id:this.graph.nodes[d].id}}dispose(){var e,t,r,s,a;(e=this.nodeBuffer)==null||e.destroy(),(t=this.edgeBuffer)==null||t.destroy(),(r=this.cameraBuffer)==null||r.destroy(),(s=this.pathBuffer)==null||s.destroy(),(a=this.liveRetentionBuffer)==null||a.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null}}const it=16,je=4,tr=110,qi=180,Vi=.7,Wi=.2,Hi=360,Ki=18;function Yi(n){if(n.edges.length>0){const e=n.centerIndex,t=n.edges.filter(r=>r.sourceIndex===e||r.targetIndex===e);if(t.length>0){let r=-1,s=-1;for(const a of t){const i=a.sourceIndex===e?a.targetIndex:a.sourceIndex,o=n.nodes[i];o&&o.retention>s&&(s=o.retention,r=i)}if(r>=0)return r}}for(let e=0;e<n.nodes.length;e++)if(e!==n.centerIndex)return e;return n.centerIndex}function Xi(n,e,t=8192){const r=Yi(n),a=n.nodes[r].id,i=rr(n,r),c=new Et({seed:e+":birth:"+a}).state.rng,l=new Float32Array(t*it),f=Math.floor(t*Vi),u=Math.floor(t*Wi),h=t-f-u;for(let S=0;S<f;S++){const v=S*it,[M,R,O]=mr(S,f,tr+c()*(qi-tr),c);l[v+0]=i[0]+M,l[v+1]=i[1]+R,l[v+2]=i[2]+O,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const d=n.edges.filter(S=>S.sourceIndex===r||S.targetIndex===r);for(let S=0;S<u;S++){const v=(f+S)*it;if(d.length===0)continue;const M=S%d.length,R=d[M],O=R.sourceIndex===r?R.targetIndex:R.sourceIndex,N=rr(n,O),X=N[0]-i[0],C=N[1]-i[1],j=N[2]-i[2],se=Math.sqrt(X*X+C*C+j*j)||1,$=S/Math.max(1,u)*2+.5,ce=c()*30,re=-C*ce/se,W=X*ce/se,Q=0;l[v+0]=i[0]+X/se*$*80+re,l[v+1]=i[1]+C/se*$*80+W,l[v+2]=i[2]+j/se*$*80+Q,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const g=300;for(let S=0;S<h;S++){const v=(f+u+S)*it,M=c()*Math.PI*2,R=c()*120;l[v+0]=i[0]+Math.cos(M)*R,l[v+1]=i[1]+Math.sin(M)*R,l[v+2]=i[2]+g*.6+c()*40,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const _=Qi(n,r),x=Zi();return{targetIndex:r,targetNodeId:a,particles:l,edgeSteps:_,timeline:x}}function rr(n,e){const t=n.nodes[e],r=n.nodes.length;if(t.isCenter&&n.centerIndex===e)return[0,0,0];const s=Math.PI*(3-Math.sqrt(5)),a=1-e/(r-1||1)*2,i=Math.sqrt(1-a*a),o=s*e,c=120,l=(e*7+3)%100/100*.1*c-.05*c,f=(e*13+7)%100/100*.1*c-.05*c,u=(e*17+11)%100/100*.1*c-.05*c;return[Math.cos(o)*i*c+l,a*c+f,Math.sin(o)*i*c+u]}function Qi(n,e){const t=n.edges.filter(a=>a.sourceIndex===e||a.targetIndex===e),r=t.length;if(r===0)return new Uint32Array(0);const s=new Uint32Array(r*je);for(let a=0;a<r;a++){const i=t[a],o=i.sourceIndex===e?i.targetIndex:i.sourceIndex,c=Hi+a*Ki;s[a*je+0]=e,s[a*je+1]=o,s[a*je+2]=c,s[a*je+3]=0}return s}function Zi(){return[{label:"latent trace condensing",startFrame:60,endFrame:239},{label:"engram coalescence",startFrame:240,endFrame:329},{label:"memory ignition",startFrame:330,endFrame:359},{label:"associations engrave",startFrame:360,endFrame:509},{label:"stabilization",startFrame:510,endFrame:659}]}const Ji=`
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

// 16 floats / 64 bytes per particle (matches birth-plan.ts layout).
struct BirthParticle {
	start_life: vec4<f32>,
	target_size: vec4<f32>,
	color_phase: vec4<f32>,
	state: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<BirthParticle>;

@compute @workgroup_size(64)
fn birth_compute(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= arrayLength(&particles)) {
		return;
	}

	// Capture mode (params.capture_mode == 1.0): skip physics integration.
	// The storage-buffer state stays frozen at initial upload values.
	if (params.capture_mode == 1.0) {
		return;
	}

	var particle = particles[i];
	let frame = params.frame;
	let phase = params.loop_phase;

	// --- Convergence choreography (integer cycles per 720-frame loop) ---

	// Phase offset (stagger) from start_life.w: 0..1 → delays convergence.
	let stagger = particle.start_life.w;

	// Effective frame: staggered loop frame (wraps at 720).
	let effFrame = fract(phase + stagger * 0.15) * 720.0;

	// --- Phase 1: latent trace condensing (frames 0–239) ---
	// Slow drift toward target.
	var t: f32;
	if (effFrame < 240.0) {
		// Smooth ease-in: 0 → 1 over 240 frames.
		t = effFrame / 240.0;
		t = t * t * (3.0 - 2.0 * t); // smoothstep
	}
	// --- Phase 2: engram coalescence (frames 240–329) ---
	// Accelerated convergence to target.
	else if (effFrame < 330.0) {
		let localFrame = effFrame - 240.0;
		// 0 → 1 over 90 frames, with slight overshoot then settle.
		t = localFrame / 90.0;
		t = t * t * (3.0 - 2.0 * t);
		// Add a small overshoot (1.05) then settle back to 1.0.
		t = 1.0 - 0.05 * (1.0 - t);
	}
	// --- Phase 3: memory ignition (frames 330–359) ---
	// Hold at target (flash handled in render).
	else if (effFrame < 360.0) {
		t = 1.0;
	}
	// --- Phase 4: associations engrave (frames 360–509) ---
	// Hold at target.
	else if (effFrame < 510.0) {
		t = 1.0;
	}
	// --- Phase 5: stabilization (frames 510–719) ---
	// Hold at target, then fade alpha for reset.
	else {
		let localFrame = effFrame - 510.0;
		// Fade alpha to 0 for seamless reset at frame 0.
		t = 1.0;
		particle.state.w = 1.0 - smoothstep(0.0, 150.0, localFrame);
	}

	// Interpolate from start to target.
	let startPos = particle.start_life.xyz;
	let targetPos = particle.target_size.xyz;
	// (WGSL forbids swizzle stores - reconstruct, preserving alpha in .w)
	particle.state = vec4<f32>(mix(startPos, targetPos, t), particle.state.w);

	// Alpha: particles fade in during convergence, fade out during reset.
	let fadeIn = smoothstep(0.0, 60.0, effFrame);
	particle.state.w = max(particle.state.w, fadeIn * 0.8);

	particles[i] = particle;
}
`,$i=16,en=6,tn=330,rn=359,nn=360;class an{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"active");p(this,"computePipeline",null);p(this,"computeBindGroup",null);p(this,"particleBuffer",null);p(this,"particleCount",0);p(this,"renderPipeline",null);p(this,"renderBindGroup",null);p(this,"haloPipeline",null);p(this,"haloBindGroup",null);p(this,"haloIndexBuffer",null);p(this,"engravePipeline",null);p(this,"engraveBindGroup",null);p(this,"engraveBuffer",null);p(this,"engraveStepCount",0);p(this,"timeline",[]);p(this,"birthPlan",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}get engraveSteps(){var e;return((e=this.birthPlan)==null?void 0:e.edgeSteps)??new Uint32Array(0)}upload(e){var a,i;const t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;const r=this.nodeRenderer.graph;if(!r)return;this.birthPlan=Xi(r,e),this.timeline=this.birthPlan.timeline;const s=this.birthPlan.particles.length/$i;this.particleCount=s,(a=this.particleBuffer)==null||a.destroy(),this.particleBuffer=t.createBuffer({label:"observatory-birth-particles",size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),(i=this.engraveBuffer)==null||i.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:"observatory-birth-engrave",size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){const t=e.createShaderModule({label:"observatory-birth-compute",code:Ji});this.computePipeline=e.createComputePipeline({label:"observatory-birth-compute-pipeline",layout:"auto",compute:{module:t,entryPoint:"birth_compute"}});const r=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:"observatory-birth-compute-bind",layout:this.computePipeline.getBindGroupLayout(0),entries:r})}createRenderPipeline(e){const r=e.createShaderModule({label:"observatory-birth-render",code:`
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
	_pad: f32,
};

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct BirthParticle {
	start_life: vec4<f32>,
	target_size: vec4<f32>,
	color_phase: vec4<f32>,
	state: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> particles: array<BirthParticle>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) color: vec3<f32>,
	@location(2) @interpolate(flat) misc: vec4<f32>,
};

const CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0, -1.0),
	vec2<f32>( 1.0,  1.0),
	vec2<f32>(-1.0,  1.0)
);

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= arrayLength(&particles)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let particle = particles[ii];
	let corner = CORNERS[vi];

	// Current position from state.xyz.
	let pos = particle.state.xyz;

	// Base size from target_size.w.
	let baseSize = particle.target_size.w;

	// Flash boost during ignition (frames 330–359).
	let frame = params.frame;
	var flashBoost = 1.0;
	if (frame >= 330.0 && frame <= 359.0) {
		let flashT = (frame - 330.0) / 29.0; // 0..1 over flash frames
		// Sharp flash: peaks at frame 345, fades by 359.
		flashBoost = 1.0 + 3.0 * (1.0 - smoothstep(330.0, 345.0, frame))
		           + 2.0 * smoothstep(345.0, 359.0, frame);
	}

	// Size: base + flash boost + pulse breathing.
	let breath = 1.0 + 0.06 * params.pulse;
	let halfSize = baseSize * 4.0 * breath * flashBoost;

	let world = pos
		+ camera.right.xyz * corner.x * halfSize
		+ camera.up.xyz * corner.y * halfSize;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;

	// Color: violet dust (0.55, 0.32, 1.00) with spectral rim.
	let phase = particle.color_phase.w;
	let spectralW = fract(params.loop_phase + phase);
	var spectralColor: vec3<f32>;
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.55, 0.32, 1.00), // violet base
		vec3<f32>(0.40, 0.60, 1.00), // blue-violet
		vec3<f32>(0.70, 0.45, 1.00), // magenta-violet
		vec3<f32>(0.55, 0.32, 1.00)  // wrap
	);
	let f = spectralW * 4.0;
	let i = u32(floor(f)) % 4u;
	let frac = f - floor(f);
	spectralColor = mix(stops[i], stops[(i + 1u) % 4u], frac);

	// Alpha from state.w (convergence progress + fade).
	let alpha = particle.state.w;

	out.color = spectralColor;
	out.misc = vec4<f32>(baseSize, 0.0, 0.0, alpha);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let alpha = in.misc.w;
	let core = smoothstep(0.25, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 2.0);

	// Additive glow: core + halo.
	let intensity = core * 1.5 + halo * 0.6;

	// Flash boost during ignition.
	let frame = params.frame;
	var flash = 0.0;
	if (frame >= 330.0 && frame <= 359.0) {
		flash = smoothstep(330.0, 345.0, frame) * 2.0;
	}

	let color = in.color * (intensity + flash);

	return vec4<f32>(color * params.brightness, 1.0);
}
`});this.renderPipeline=e.createRenderPipeline({label:"observatory-birth-render",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});const s=this.nodeRenderer.cameraUniformBuffer;this.renderBindGroup=e.createBindGroup({label:"observatory-birth-render-bind",layout:this.renderPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:this.particleBuffer}}]})}createHaloPipeline(e){const r=e.createShaderModule({label:"observatory-birth-halo",code:`
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
	_pad: f32,
};

struct Camera {
	view_proj: mat4x4<f32>,
	right: vec4<f32>,
	up: vec4<f32>,
};

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> nodes: array<Node>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let node = nodes[ii];
	let flags = u32(node.color_flags.w);
	let is_target = (flags & 4u) != 0u; // flag 2: is birth target

	if (!is_target) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	// Flash halo: only visible during ignition (frames 330–359).
	let frame = params.frame;
	if (frame < 330.0 || frame > 359.0) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	// Halo ring: expands during flash, fades by frame 359.
	let flashT = (frame - 330.0) / 29.0; // 0..1
	let ringRadius = 0.3 + flashT * 0.5; // expands 0.3 → 0.8

	// Quad centered on target position.
	let pos = node.pos_radius.xyz;
	let cornerX = (f32(vi) / 3.0 - 1.0); // -1, 0, 1 (3 unique x)
	let cornerY = (f32(vi % 3) / 1.5 - 1.0); // -1, 0, 1

	// We use 4 vertices for a simple quad (vi 0..3).
	let cx = cornerX * ringRadius;
	let cy = cornerY * ringRadius;

	let world = pos
		+ camera.right.xyz * cx
		+ camera.up.xyz * cy;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);

	// UV for radial fade.
	out.uv = vec2<f32>(cx / ringRadius, cy / ringRadius);

	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 0.7) {
		discard;
	}

	// Flash: white-hot core, violet rim.
	let flashIntensity = 1.0 - smoothstep(0.0, 0.7, d);
	let color = vec3<f32>(0.7, 0.4, 1.0) * flashIntensity * 2.0;

	// Fade out as flash ends.
	let frame = params.frame;
	let fadeOut = 1.0 - smoothstep(345.0, 359.0, frame);

	return vec4<f32>(color * params.brightness * fadeOut, 1.0);
}
`});this.haloPipeline=e.createRenderPipeline({label:"observatory-birth-halo",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});const s=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:"observatory-birth-halo-bind",layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;const t=e.createShaderModule({label:"observatory-birth-engrave",code:vr});this.engravePipeline=e.createRenderPipeline({label:"observatory-birth-engrave-pipeline",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.engraveBindGroup=e.createBindGroup({label:"observatory-birth-engrave-bind",layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){const r=this.engine.params[9];if(this.active=r===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;const s=e.beginComputePass({label:"observatory-birth-compute"});s.setPipeline(this.computePipeline),s.setBindGroup(0,this.computeBindGroup),s.dispatchWorkgroups(Math.ceil(this.particleCount/64)),s.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(en,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=tn&&t<=rn&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=nn&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){var e,t,r,s,a,i,o,c,l,f,u;(e=this.particleBuffer)==null||e.destroy(),this.particleBuffer=null,(r=(t=this.computePipeline)==null?void 0:t.destroy)==null||r.call(t),this.computePipeline=null,this.computeBindGroup=null,(a=(s=this.renderPipeline)==null?void 0:s.destroy)==null||a.call(s),this.renderPipeline=null,this.renderBindGroup=null,(o=(i=this.haloPipeline)==null?void 0:i.destroy)==null||o.call(i),this.haloPipeline=null,this.haloBindGroup=null,(c=this.haloIndexBuffer)==null||c.destroy(),this.haloIndexBuffer=null,(f=(l=this.engravePipeline)==null?void 0:l.destroy)==null||f.call(l),this.engravePipeline=null,this.engraveBindGroup=null,(u=this.engraveBuffer)==null||u.destroy(),this.engraveBuffer=null}}function sn(n){const e=n.hopSlot.toFixed(1),t=n.causeDepth.toFixed(1);return`
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

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// 1 u32/node: bits 0-15 hopDepth (0xffff unreached), 16 failure, 17 cause,
// 18 lookalike, 19-21 lookalike k (rescue-plan.ts packing).
@group(0) @binding(2) var<storage, read> wave: array<u32>;

const HOP_SLOT: f32 = ${e};
const CAUSE_DEPTH: f32 = ${t};
const TAU: f32 = 6.28318530717958647;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

fn arrival(d: f32) -> f32 {
	return min(260.0 + HOP_SLOT * d, 514.0);
}

@compute @workgroup_size(64)
fn rescue_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&wave)) {
		return;
	}
	// Belt-and-braces atop the TS gate: salience-rescue is demo index 2.
	if (params.demo_id != 2.0) {
		return;
	}

	let packed = wave[i];
	let depth_u = packed & 0xffffu;
	let d = f32(depth_u);
	let is_failure = (packed & 0x10000u) != 0u;
	let is_cause = (packed & 0x20000u) != 0u;
	let is_look = (packed & 0x40000u) != 0u;
	let look_k = f32((packed >> 19u) & 0x7u);

	let f = params.frame;

	var dx = 0.0;
	var dy = 0.0;
	var dz = 0.0;
	var dw = 0.0;

	if (is_failure) {
		// Detonation spike, wound simmer, recognition flare as the arc lands.
		dw = dw + env(f, 90.0, 96.0, 120.0, 168.0);
		dw = dw + 0.35 * env(f, 100.0, 130.0, 600.0, 656.0);
		dw = dw + 0.35 * env(f, 552.0, 562.0, 580.0, 640.0);
		// Symptom backlight while the cause burns.
		dx = dx + 0.4 * env(f, 556.0, 566.0, 620.0, 668.0);
	}
	if (!is_failure && depth_u >= 1u && depth_u <= 12u) {
		// Shockwave blink: crimson concussion, 3 frames/hop of REAL graph distance.
		dw = dw + 0.75 * exp(-0.3 * d)
			* env(f, 92.0 + 3.0 * d, 96.0 + 3.0 * d, 96.0 + 3.0 * d, 122.0 + 3.0 * d);
	}
	if (is_look) {
		let fk = 138.0 + 28.0 * look_k;
		// Searchlight flare — cold pop, sequential, on camera.
		dy = dy + env(f, fk - 6.0, fk, fk + 10.0, fk + 26.0);
		// Ash residue — the struck-through lookalike stays in frame until the verdict.
		dy = dy + 0.15 * smoothstep(fk + 10.0, fk + 26.0, f) * (1.0 - smoothstep(600.0, 656.0, f));
	}
	if (!is_failure && depth_u >= 1u && d <= CAUSE_DEPTH) {
		let wd = arrival(d);
		// Interrogation flicker: 24 integer sine cycles per loop, per-depth phase.
		let flicker = 0.75 + 0.25 * sin(TAU * 24.0 * params.loop_phase + 1.7 * d);
		dz = dz + env(f, wd - 10.0, wd, wd + 28.0, wd + 64.0) * flicker;
		// Scanned ember.
		dz = dz + 0.08 * smoothstep(wd + 28.0, wd + 64.0, f) * (1.0 - smoothstep(580.0, 640.0, f));
	}
	if (is_cause) {
		// Cause ignition rides the EXISTING recall response (render-nodes.wgsl):
		// spectral() thin-film band + white-hot core + sprite swell at full intensity.
		dx = dx + env(f, 520.0, 546.0, 640.0, 700.0);
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them).
	var node = nodes[i];
	node.demo = vec4<f32>(dx, dy, dz, dw);
	nodes[i] = node;
}
`}const on=2;class ln{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"waveBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.waveBuffer)==null||r.destroy(),this.waveBuffer=e.createBuffer({label:"observatory-rescue-wave",size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);const t=e.createShaderModule({label:"observatory-rescue-choreo",code:sn(this.plan.consts)});this.pipeline=e.createComputePipeline({label:"observatory-rescue-choreo",layout:"auto",compute:{module:t,entryPoint:"rescue_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-rescue-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==on||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-rescue-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.waveBuffer)==null||e.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}}const cn=4,yr=90,dn=138,un=28,ot=260,fn=514,ir=560,_r=600,Se=65535,hn=48,pn={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function wr(n,e){const t=new Et({seed:e});return gr(n,t.state.rng).data}function mn(n){const e=new Uint32Array(n.nodes.length);for(const t of n.edges)e[t.sourceIndex]++,e[t.targetIndex]++;return e}function gn(n,e){const t=n.nodes.length;if(t===0)return-1;const r=mn(n),s=i=>{const o=n.nodes[i],c=new Set(o.tags.map(d=>d.toLowerCase()));let l=0;(c.has("failure")||c.has("guardrail"))&&(l+=3),(c.has("confusion")||c.has("weak-spot"))&&(l+=2),l+=Math.min(r[i],8)/8;const f=e[i*ae+0],u=e[i*ae+1],h=e[i*ae+2];return Math.sqrt(f*f+u*u+h*h)>=54&&(l+=.5),l},a=[i=>i!==n.centerIndex&&!n.nodes[i].suppressed&&r[i]>=2,i=>i!==n.centerIndex&&!n.nodes[i].suppressed,i=>i!==n.centerIndex,()=>!0];for(const i of a){let o=-1,c=-1/0;for(let l=0;l<t;l++){if(!i(l))continue;const f=s(l);f>c&&(c=f,o=l)}if(o>=0)return o}return-1}function vn(n,e){const t=n.nodes.length,r=new Uint16Array(t).fill(Se),s=new Int32Array(t).fill(-1);if(e<0||e>=t)return{depths:r,parents:s};const a=Array.from({length:t},()=>[]);for(const o of n.edges){const c=pn[o.type]??5;a[o.sourceIndex].push({nbr:o.targetIndex,rank:c}),a[o.targetIndex].push({nbr:o.sourceIndex,rank:c})}for(const o of a)o.sort((c,l)=>c.rank-l.rank||c.nbr-l.nbr);r[e]=0;const i=[e];for(let o=0;o<i.length;o++){const c=i[o];for(const{nbr:l}of a[c])r[l]===Se&&(r[l]=r[c]+1,i.push(l))}for(let o=0;o<t;o++)if(!(r[o]===Se||r[o]===0)){for(const{nbr:c}of a[o])if(r[c]===r[o]-1){s[o]=c;break}}return{depths:r,parents:s}}function bn(n,e,t,r){const s=new Map;for(const a of n.nodes)s.set(a.id,a.createdAt);for(const a of[3,2,1]){const i=[];for(let d=0;d<e.nodes.length;d++){if(d===e.centerIndex||d===r)continue;const g=t[d];g===Se||g<a||i.push(d)}if(i.length===0)continue;let o=i.filter(d=>e.nodes[d].retention<=.45);o.length===0&&(o=i);const c=new Map;let l=1/0,f=-1/0;for(const d of o){const g=s.get(e.nodes[d].id),_=g?Date.parse(g):NaN;Number.isFinite(_)&&(c.set(d,_),_<l&&(l=_),_>f&&(f=_))}const u=d=>{const g=c.get(d);return g===void 0?0:f===l?1:(f-g)/(f-l)},h=d=>2*(1-e.nodes[d].retention)+.5*Math.min(t[d],6)/6+.5*u(d);return o.sort((d,g)=>{const _=h(d),x=h(g);return x!==_?x-_:t[g]!==t[d]?t[g]-t[d]:d-g}),{index:o[0],depth:t[o[0]]}}return{index:-1,depth:0}}function yn(n,e,t,r,s){const a=n[t*ae+0],i=n[t*ae+1],o=n[t*ae+2],c=[];for(let l=0;l<e;l++){if(l===t||l===r||l===s)continue;const f=n[l*ae+0]-a,u=n[l*ae+1]-i,h=n[l*ae+2]-o;c.push({i:l,d2:f*f+u*u+h*h})}return c.sort((l,f)=>l.d2-f.d2||l.i-f.i),c.slice(0,cn).map(l=>l.i)}function lt(n){const e=Math.max(1,n);return Math.min(84,Math.max(14,Math.floor(252/e)))}function _n(n,e){return Math.min(ot+e*n,fn)}function nr(n){return dn+un*n}function he(n){return n.length>64?n.slice(0,64)+"…":n}const be=4;function Ee(n){const e=new Uint32Array(n);return e.fill(Se),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(n).fill(Se),causeDepth:0,hopSlot:lt(3),waveData:e,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"root cause found",causeLabel:"",failureLabel:"",causeDate:"",hops:0,k:0,receipt:""},consts:{hopSlot:lt(3),causeDepth:3}}}function wn(n,e,t,r){var Q;if(r)return xn(n,e,r);const s=e.nodes.length;if(s===0)return Ee(0);const a=wr(e,t),i=gn(e,a);if(i<0)return Ee(s);const{depths:o,parents:c}=vn(e,i),l=bn(n,e,o,i);if(l.index<0){const w=Ee(s);return w.failureIndex=i,w.hopDepths=o,w}const f=l.index,u=Math.max(1,l.depth),h=lt(u),d=w=>_n(w,h),g=yn(a,s,i,f,e.centerIndex),_=g.length,x=new Uint32Array(s);for(let w=0;w<s;w++){let L=o[w]&65535;w===i&&(L|=65536),w===f&&(L|=1<<17),x[w]=L}g.forEach((w,L)=>{x[w]|=1<<18|L<<19});const S=[];g.forEach((w,L)=>{S.push({src:i,dst:w,bf:nr(L),kind:ue.probe,beatKind:"probe"})});const v=[];{let w=f;for(;w!==i&&w>=0&&c[w]>=0;)v.push(w),w=c[w]}const M=new Set(v),R=[];for(let w=0;w<s;w++){if(w===i||M.has(w))continue;const L=o[w];L===Se||L<1||L>u||c[w]<0||R.push(w)}R.sort((w,L)=>o[w]-o[L]||w-L);const O=[...v.slice().reverse(),...R].slice(0,hn);O.sort((w,L)=>o[w]-o[L]||w-L);for(const w of O)S.push({src:c[w],dst:w,bf:d(o[w]),kind:ue.backwardCause,beatKind:"wave"});S.push({src:f,dst:i,bf:ir,kind:ue.backwardCause,beatKind:"arc"});const N=new Uint32Array(Math.max(1,S.length)*be),X=[];S.forEach((w,L)=>{N[L*be+0]=w.src,N[L*be+1]=w.dst,N[L*be+2]=w.bf,N[L*be+3]=w.kind,X.push({sourceIndex:w.src,targetIndex:w.dst,beatFrame:w.bf,kind:w.kind,beatKind:w.beatKind,nodeId:e.nodes[w.dst].id,label:he(e.nodes[w.dst].label)})});const C=he(e.nodes[i].label),j=he(e.nodes[f].label),se=[],$=(w,L,Ge,ft)=>{se.push({sourceIndex:i,targetIndex:i,beatFrame:w,kind:L,beatKind:"rescue",nodeId:ft,label:Ge})};$(yr,1,`failure: ${C}`,e.nodes[i].id),g.forEach((w,L)=>{$(nr(L),0,`lookalike ✗ · ${he(e.nodes[w].label)}`,e.nodes[w].id)}),$(d(1),1,"reaching backward through time","rescue-wave-start"),u>=2&&d(u)!==d(1)&&$(d(u),1,`scrubbing past · ${u} hops`,"rescue-wave-deep"),$(ir,1,`causal arc · ${j}`,e.nodes[f].id),$(_r,1,"root cause found","rescue-verdict");const ce=((Q=n.nodes.find(w=>w.id===e.nodes[f].id))==null?void 0:Q.createdAt)??"",re=ce?ce.slice(0,10):"",W={headline:"root cause found",causeLabel:j,failureLabel:C,causeDate:re,hops:u,k:_,receipt:`${u} hops back · ${re} · vector search: 0 for ${_}`};return{viable:!0,failureIndex:i,causeIndex:f,lookalikeIndices:g,hopDepths:o,causeDepth:u,hopSlot:h,waveData:x,pathData:N,pathMetas:X,spineBeats:se,verdict:W,consts:{hopSlot:h,causeDepth:u}}}function xn(n,e,t){var N,X;const r=e.nodes.length,s=e.indexById.get(t.failureId)??-1,a=t.pathIds??[];if(s<0||a.length<2||a[a.length-1]!==t.failureId||new Set(a).size!==a.length)return Ee(r);const i=a.map(C=>e.indexById.get(C));if(i.some(C=>C===void 0))return Ee(r);const o=i,c=o[0];if(c===s)return Ee(r);const l=t.candidates.find(C=>C.memoryId===a[0]);if(!l)return Ee(r);const f=new Uint16Array(r);f.fill(Se),f[s]=0,o.forEach((C,j)=>{f[C]=o.length-1-j});const u=new Uint32Array(r);u[s]=65536,o.slice(0,-1).forEach(C=>{u[C]=f[C]}),u[c]|=1<<17;const h=o.length-1,d=lt(h),g=he(e.nodes[c].label),_=he(e.nodes[s].label),x=((X=(N=n.nodes.find(C=>C.id===l.memoryId))==null?void 0:N.createdAt)==null?void 0:X.slice(0,10))??"",S=new Uint32Array((o.length-1)*be),v=o.slice(0,-1).map((C,j)=>{const se=o[j+1],$=ot+j*d;return S[j*be]=C,S[j*be+1]=se,S[j*be+2]=$,S[j*be+3]=ue.backwardCause,{sourceIndex:C,targetIndex:se,beatFrame:$,kind:ue.backwardCause,beatKind:"receipt-path",nodeId:a[j+1],label:`recorded path · ${he(e.nodes[se].label)}`}}),M=l.sharedEntities.length?l.sharedEntities.join(", "):"recorded entity",R=l.similarityRank===null?"rank unavailable":`embedding rank #${l.similarityRank}`,O=[{sourceIndex:s,targetIndex:s,beatFrame:yr,kind:1,beatKind:"receipt-failure",nodeId:t.failureId,label:`recorded failure · ${_}`},{sourceIndex:s,targetIndex:s,beatFrame:ot,kind:1,beatKind:"receipt-join",nodeId:"receipt-join",label:`shared entity · ${M}`},{sourceIndex:c,targetIndex:s,beatFrame:ot+(h-1)*d,kind:ue.backwardCause,beatKind:"receipt-candidate",nodeId:l.memoryId,label:`candidate · ${g}`},{sourceIndex:c,targetIndex:c,beatFrame:_r,kind:1,beatKind:"receipt-verdict",nodeId:"receipt-verdict",label:"candidate cause found"}];return{viable:!0,failureIndex:s,causeIndex:c,lookalikeIndices:[],hopDepths:f,causeDepth:h,hopSlot:d,waveData:u,pathData:S,pathMetas:v,spineBeats:O,verdict:{headline:"candidate cause found",causeLabel:g,failureLabel:_,causeDate:x,hops:h,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${M} · ${R}`},consts:{hopSlot:d,causeDepth:h}}}const Bn=`
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

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// 1 u32/node: bits 0-7 rank, 8 isDrifting, 9 isRescued, 10-11 rescue slot k
// (forgetting-plan.ts packing). Non-drifting nodes are exactly 0.
@group(0) @binding(2) var<storage, read> horizon: array<u32>;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

@compute @workgroup_size(64)
fn forgetting_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&horizon)) {
		return;
	}
	// Belt-and-braces atop the TS gate: forgetting-horizon is demo index 3.
	if (params.demo_id != 3.0) {
		return;
	}

	let packed = horizon[i];
	let is_drifting = (packed & 0x100u) != 0u;
	let is_rescued = (packed & 0x200u) != 0u;
	let rank01 = f32(packed & 0xffu) / 255.0;
	let k = f32((packed >> 10u) & 0x3u);

	let f = params.frame;
	// Master release: every lane is exactly 0.0 by frame 712 — the seam wall.
	let master = 1.0 - smoothstep(660.0, 712.0, f);

	var dx = 0.0;
	var dz = 0.0;

	if (is_drifting) {
		let onset = 90.0 + 42.0 * rank01;
		// Phase 1 — the drift: dim + fall to the 0.55 plateau, retention-staggered.
		let phase1 = 0.55 * smoothstep(onset, onset + 210.0, f);
		if (is_rescued) {
			let rk = 318.0 + 60.0 * k;
			// Snap-back begins 22 frames before the recall ribbon lands at rk.
			dz = master * phase1 * (1.0 - smoothstep(rk - 22.0, rk + 6.0, f));
			// Ignition rides the EXISTING recall response (render-nodes.wgsl):
			// spectral() thin-film band + white-hot core + sprite swell for free.
			dx = master * env(f, rk - 26.0, rk, rk + 60.0, rk + 130.0);
		} else {
			// Phase 2 — the sink: to exactly 1.0 over 640..660 (the ~6% floor era).
			let phase2 = 0.45 * smoothstep(480.0 + 24.0 * rank01, 640.0, f);
			dz = master * (phase1 + phase2);
		}
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them). demo.y and
	// demo.w are hard 0.0: the rescue/firewall grammars can never fire here.
	var node = nodes[i];
	node.demo = vec4<f32>(dx, 0.0, dz, 0.0);
	nodes[i] = node;
}
`,Pn=3;class Sn{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"horizonBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.horizonBuffer)==null||r.destroy(),this.horizonBuffer=e.createBuffer({label:"observatory-forgetting-horizon",size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);const t=e.createShaderModule({label:"observatory-forgetting-choreo",code:Bn});this.pipeline=e.createComputePipeline({label:"observatory-forgetting-choreo",layout:"auto",compute:{module:t,entryPoint:"forgetting_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-forgetting-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==Pn||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-forgetting-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.horizonBuffer)==null||e.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}}const kn=318,In=60,xr=3,Rn=132,En=60,An=540,Cn=660;function Mn(n){const e=[];for(let s=0;s<n.nodes.length;s++)s!==n.centerIndex&&e.push(s);e.sort((s,a)=>n.nodes[s].retention-n.nodes[a].retention||s-a);const t=e.length;if(t===0)return[];const r=Math.min(t,Math.max(Math.min(xr,t),Math.round(.25*t)));return e.slice(0,r)}function Fn(n,e){const t=new Uint32Array(n.nodes.length);for(const a of n.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>2*n.nodes[a].retention+Math.min(t[a],8)/8;return e.slice().sort((a,i)=>r(i)-r(a)||a-i).slice(0,Math.min(xr,e.length))}function ar(n){return kn+In*n}const qe=4;function Ln(n){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(n),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function Gn(n){const e=n.nodes.length,t=Mn(n);if(e<2||t.length<1)return Ln(e);const r=Fn(n,t),s=t.length,a=new Uint32Array(e);t.forEach((h,d)=>{const g=Math.round(255*d/Math.max(1,s-1));a[h]=g&255|256}),r.forEach((h,d)=>{a[h]|=512|d<<10});const i=new Uint32Array(Math.max(1,r.length)*qe),o=[];r.forEach((h,d)=>{const g=ar(d);i[d*qe+0]=n.centerIndex,i[d*qe+1]=h,i[d*qe+2]=g,i[d*qe+3]=ue.recall,o.push({sourceIndex:n.centerIndex,targetIndex:h,beatFrame:g,kind:ue.recall,beatKind:"recall",nodeId:n.nodes[h].id,label:he(n.nodes[h].label)})});const c=[],l=(h,d,g,_)=>{c.push({sourceIndex:n.centerIndex,targetIndex:n.centerIndex,beatFrame:h,kind:d,beatKind:"horizon",nodeId:_,label:g})},f=new Set(r),u=t.filter(h=>!f.has(h)).slice(0,3);return u.forEach((h,d)=>{const g=Math.round(n.nodes[h].retention*100);l(Rn+En*d,1,`fading: ${he(n.nodes[h].label)} · retention ${g}%`,n.nodes[h].id)}),r.forEach((h,d)=>{l(ar(d),0,`recalled: ${he(n.nodes[h].label)}`,n.nodes[h].id)}),u.length>0&&l(An,1,"the unrecalled sink · nothing is deleted","horizon-sink"),l(Cn,0,"every memory still retrievable","horizon-retrievable"),{viable:!0,driftingIndices:t,rescuedIndices:r,horizonData:a,pathData:i,pathMetas:o,spineBeats:c}}const Dn=`
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

struct Node {
	pos_radius: vec4<f32>,
	vel_retention: vec4<f32>,
	color_flags: vec4<f32>,
	demo: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
// 1 u32/node: bits 0-7 shockDelay, 8 isIntruder, 9 isSeverNeighbor,
// 10-13 sever slot k (firewall-plan.ts packing). Every node carries a delay.
@group(0) @binding(2) var<storage, read> fire: array<u32>;

const TAU: f32 = 6.28318530717958647;

fn env(f: f32, a0: f32, a1: f32, r0: f32, r1: f32) -> f32 {
	return smoothstep(a0, a1, f) * (1.0 - smoothstep(r0, r1, f));
}

@compute @workgroup_size(64)
fn firewall_choreo(@builtin(global_invocation_id) id: vec3<u32>) {
	let i = id.x;
	if (i >= u32(params.node_count)) {
		return;
	}
	if (i >= arrayLength(&fire)) {
		return;
	}
	// Fire in TWO modes: the deterministic demo (demo_id == 4, wrapped loop
	// frame) OR a LIVE event (live_kind == 1, driven by live_frame = frames
	// since a real MemorySuppressed / contradiction fired). Anything else →
	// no-op, and other grammars own the lanes.
	let is_demo = params.demo_id == 4.0;
	let is_live = params.live_kind == 1.0;
	if (!is_demo && !is_live) {
		return;
	}

	let packed = fire[i];
	let delay = f32(packed & 0xffu);
	let is_intruder = (packed & 0x100u) != 0u;
	let is_sever = (packed & 0x200u) != 0u;
	let k = f32((packed >> 10u) & 0xfu);

	// Live mode replays the SAME 720-frame beat map from the event: f =
	// live_frame (clamped to the loop window), loop_phase derived from it so the
	// integer-cycle sines still resolve. Demo mode reads the wrapped loop clock.
	var f = params.frame;
	var lp = params.loop_phase;
	if (is_live) {
		f = clamp(params.live_frame, 0.0, 719.0);
		lp = f / 720.0;
	}

	var fy = 0.0;
	var fw = 0.0;

	if (is_intruder) {
		// Intrusion flare: sickly strobe, band (0..1], 36 integer cycles/loop.
		// C¹ handoff into the membrane over 330-332 (the rise sweeps the flare
		// band exactly once — the condensation read is intentional).
		fy = env(f, 90.0, 96.0, 310.0, 332.0)
			* (0.55 + 0.45 * sin(TAU * 36.0 * lp));
		// Membrane: sustained ring band [2.60..2.90], 12 integer cycles/loop.
		fy = fy + env(f, 330.0, 352.0, 620.0, 680.0)
			* (2.75 + 0.15 * sin(TAU * 12.0 * lp));
		// Source detonation as the front leaves.
		fw = env(f, 148.0, 153.0, 162.0, 196.0);
	} else {
		// Crimson rim as the radial front passes: arrival A = 150 + delay,
		// amplitude fades with distance; A ∈ [150, 294] ⇒ all rims dead by 320.
		let a = 150.0 + delay;
		let amp = 0.9 - 0.45 * (delay / 144.0);
		fw = amp * env(f, a - 2.0, a + 3.0, a + 8.0, a + 26.0);
		if (is_sever) {
			// Node-side receipt of the severed edge; last release 474.
			let sk = 345.0 + 21.0 * k;
			fw = fw + 0.6 * env(f, sk - 4.0, sk, sk + 6.0, sk + 24.0);
		}
	}

	// WGSL forbids swizzle stores — reconstruct the FULL vec4; pos/vel/color
	// lanes pass through untouched (the force sim owns them). demo.x and
	// demo.z are hard 0.0: the recall and horizon grammars can never fire here.
	var node = nodes[i];
	node.demo = vec4<f32>(0.0, fy, 0.0, fw);
	nodes[i] = node;
}
`,Tn=4;class Br{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"fireBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.fireBuffer)==null||r.destroy(),this.fireBuffer=e.createBuffer({label:"observatory-firewall-fire",size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);const t=e.createShaderModule({label:"observatory-firewall-choreo",code:Dn});this.pipeline=e.createComputePipeline({label:"observatory-firewall-choreo",layout:"auto",compute:{module:t,entryPoint:"firewall_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-firewall-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,!!this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){const t=this.engine.params[9]===Tn,r=this.engine.params[12]===1;if(!t&&!r||!this.pipeline||!this.bindGroup)return;const s=this.nodeRenderer.nodeCountValue;if(s===0)return;const a=e.beginComputePass({label:"observatory-firewall-choreo"});a.setPipeline(this.pipeline),a.setBindGroup(0,this.bindGroup),a.dispatchWorkgroups(Math.ceil(s/64)),a.end()}dispose(){var e;(e=this.fireBuffer)==null||e.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}}const zn=90,Un=150,On=144,Nn=330,jn=345,qn=21,Vn=6,Wn=480,Hn=["failure","guardrail","confusion"];function Kn(n){const e=n.nodes.length;if(e===0)return-1;const t=new Uint32Array(e);for(const a of n.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>n.nodes[a].tags.some(i=>Hn.includes(i.toLowerCase())),s=[a=>a!==n.centerIndex&&!n.nodes[a].suppressed&&r(a),a=>a!==n.centerIndex&&!n.nodes[a].suppressed&&t[a]<=1,a=>a!==n.centerIndex&&!n.nodes[a].suppressed,a=>a!==n.centerIndex];for(const a of s){let i=-1;for(let o=0;o<e;o++)a(o)&&(i<0||n.nodes[o].retention<n.nodes[i].retention)&&(i=o);if(i>=0)return i}return-1}function Yn(n,e,t){const r=n[t*ae+0],s=n[t*ae+1],a=n[t*ae+2],i=new Array(e);let o=0;for(let l=0;l<e;l++){const f=n[l*ae+0]-r,u=n[l*ae+1]-s,h=n[l*ae+2]-a,d=Math.sqrt(f*f+u*u+h*h);i[l]=d,d>o&&(o=d)}o<1e-6&&(o=1);const c=new Array(e);for(let l=0;l<e;l++)c[l]=Math.min(255,Math.max(0,Math.round(On*i[l]/o)));return c[t]=0,c}function Xn(n,e){const t=new Set;for(const r of n.edges)r.sourceIndex===e&&r.targetIndex!==e&&t.add(r.targetIndex),r.targetIndex===e&&r.sourceIndex!==e&&t.add(r.sourceIndex);return Array.from(t).sort((r,s)=>r-s).slice(0,Vn)}function sr(n){return jn+qn*n}const Ve=4;function Qn(n){return At(n)}function At(n){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(n),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"threat quarantined",intruderLabel:"",receipt:"memory held in review · Memory PR opened"}}}function Zn(n,e){return Pr(n,e,Kn(n))}function Jn(n,e,t){return t<0||t>=n.nodes.length?At(n.nodes.length):Pr(n,e,t)}function Pr(n,e,t){const r=n.nodes.length;if(r===0||t<0)return At(r);const s=wr(n,e),a=Yn(s,r,t),i=Xn(n,t),o=new Uint32Array(r);for(let g=0;g<r;g++)o[g]=a[g]&255;o[t]=256,i.forEach((g,_)=>{o[g]|=512|_<<10});const c=new Uint32Array(Math.max(1,i.length)*Ve),l=[];i.forEach((g,_)=>{const x=sr(_);c[_*Ve+0]=t,c[_*Ve+1]=g,c[_*Ve+2]=x,c[_*Ve+3]=ue.probe,l.push({sourceIndex:t,targetIndex:g,beatFrame:x,kind:ue.probe,beatKind:"sever",nodeId:n.nodes[g].id,label:he(n.nodes[g].label)})});const f=he(n.nodes[t].label),u=[],h=(g,_,x)=>{u.push({sourceIndex:t,targetIndex:t,beatFrame:g,kind:1,beatKind:"firewall",nodeId:x,label:_})};return h(zn,`intrusion · ${f}`,n.nodes[t].id),h(Un,"immune response · shockwave","firewall-shock"),h(Nn,"membrane forming","firewall-membrane"),i.forEach((g,_)=>{h(sr(_),`edge severed ✗ · ${he(n.nodes[g].label)}`,n.nodes[g].id)}),h(Wn,"threat quarantined","firewall-verdict"),{viable:!0,intruderIndex:t,severedNeighborIndices:i,shockDelays:a,fireData:o,pathData:c,pathMetas:l,spineBeats:u,verdict:{headline:"threat quarantined",intruderLabel:f,receipt:"memory held in review · Memory PR opened"}}}const ut=.1542;function $n(n=ut){return Math.pow(.9,-1/n)-1}function Sr(n,e,t=ut){if(!(n>0))return 0;if(!(e>0))return 1;const r=$n(t),s=Math.pow(1+r*e/n,-t);return s<0?0:s>1?1:s}const Ct=864e5;function ea(n,e,t=0){if(!n)return t>0?t:0;const r=Date.parse(n);if(!Number.isFinite(r))return t>0?t:0;const s=(e-r)/Ct;return Math.max(0,s)+Math.max(0,t)}function ta(n,e,t,r,s=ut){if(t){const i=Date.parse(t);if(Number.isFinite(i)&&r<i)return 0}if(n===void 0||!Number.isFinite(n)||!e)return 1;const a=Date.parse(e);return Number.isFinite(a)?Math.max(.001,Sr(n,(r-a)/Ct,s)):1}function ra(n,e,t,r=0,s=ut){return n===void 0||!Number.isFinite(n)?1:Sr(n,ea(e,t,r),s)}const or={[ie.firewall]:620,[ie.dreamStorm]:360,[ie.causalRecall]:260,[ie.birth]:180};class ia{constructor(e){p(this,"engine");p(this,"renderer");p(this,"graph");p(this,"response");p(this,"seed");p(this,"projectionDays");p(this,"chronoOffsetDays");p(this,"onApply");p(this,"onFirewall");p(this,"firewall",null);p(this,"liveEdges",[]);p(this,"liveEdgeKeys",new Set);p(this,"edgesDirty",!1);p(this,"indexById");p(this,"active",null);p(this,"dreamOpen",!1);p(this,"retention");p(this,"hasLiveDecay",!1);p(this,"eventsSeen",0);p(this,"lastDecayFrame",-1e3);p(this,"lastAppliedMs",0);p(this,"seeded",!1);p(this,"lastProj",-1);p(this,"lastChrono",0);this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;const t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let s=0;s<t;s++){const a=e.graph.nodes[s];this.retention[s]=a.retention,a.stability!==void 0&&a.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(const s of this.liveEdges)this.liveEdgeKeys.add(lr(s.sourceIndex,s.targetIndex));this.lastAppliedMs=0;const r=this.engine.params;r[oe.liveKind]=ie.none,r[oe.liveFrame]=0,r[oe.liveEnergy]=0,r[oe.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}seedWatermark(e){let t=0;for(const r of e){const s=cr(r);s>t&&(t=s)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,r){if(this.active!==null)return!1;const s=this.indexById.get(e);if(s===void 0||(this.retention[s]??0)<5e-4)return!1;const a=t.filter(i=>i!==e&&this.indexById.has(i));return this.arm({kind:ie.causalRecall,startFrame:r,targetId:e,relatedIds:a,pairs:[],scalar:a.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let r=e.length-1;r>=0;r--){const s=e[r],a=cr(s);a>this.lastAppliedMs&&(this.decodeAndArm(s,this.engine.totalFrames),a>t&&(t=a))}this.lastAppliedMs=t}decodeAndArm(e,t){var s;const r=e.data??{};switch(e.type){case"MemorySuppressed":{const a=Le(r.id);if(!a||!this.indexById.has(a))return;this.arm({kind:ie.firewall,startFrame:t,targetId:a,relatedIds:this.neighborsOf(a),pairs:[],scalar:We(r.estimated_cascade)});break}case"DeepReferenceCompleted":{const i=na(r.contradiction_pairs).filter(([l,f])=>this.indexById.has(l)&&this.indexById.has(f));if(i.length>0){const l=i[0][0];this.arm({kind:ie.firewall,startFrame:t,targetId:l,relatedIds:i.flatMap(f=>f).filter(f=>f!==l),pairs:i,scalar:i.length});return}const o=Le(r.primary_id),c=dr(r.supporting_ids).filter(l=>this.indexById.has(l));o&&this.indexById.has(o)&&this.arm({kind:ie.causalRecall,startFrame:t,targetId:o,relatedIds:c,pairs:[],scalar:We(r.confidence)});break}case"BackfillFired":case"CausalReceipt":{const a=dr(r.path_ids??r.causal_path),i=Le(r.failure_id??r.target_id??r.effect_id)||a.at(-1)||a[0];i&&this.indexById.has(i)&&this.arm({kind:ie.causalRecall,startFrame:t,targetId:i,relatedIds:a.filter(o=>o!==i),exactPath:a,pairs:[],scalar:a.length});break}case"DreamStarted":{this.dreamOpen=!0,this.arm({kind:ie.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:We(r.memory_count)});break}case"DreamCompleted":{this.dreamOpen=!1;const a=We(r.connections_found);this.active&&this.active.kind===ie.dreamStorm?this.active.scalar=Math.max(this.active.scalar,a):this.arm({kind:ie.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:a});break}case"ConnectionDiscovered":{const a=this.indexById.get(Le(r.source_id)),i=this.indexById.get(Le(r.target_id));if(a===void 0||i===void 0||a===i)break;const o=lr(a,i);if(this.liveEdgeKeys.has(o))break;this.liveEdgeKeys.add(o),this.liveEdges.push({sourceIndex:a,targetIndex:i,weight:We(r.weight)||.5,type:Le(r.connection_type)||"semantic"}),this.edgesDirty=!0,this.dreamOpen&&((s=this.active)==null?void 0:s.kind)===ie.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){var t;if(this.active=e,this.eventsSeen++,e.kind===ie.firewall){const r=this.indexById.get(e.targetId);if(r===void 0)return;const s=Jn(this.graph,this.seed,r);if(!s.viable)return;this.firewall||(this.firewall=new Br({engine:this.engine,nodeRenderer:this.renderer,plan:Qn(this.graph.nodes.length)})),this.firewall.rearm(s),(t=this.onFirewall)==null||t.call(this,{intruderLabel:s.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===ie.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){const s=e.exactPath;if(s.some(o=>!this.indexById.has(o)))return;const a=new Uint32Array(Math.max(1,s.length-1)*4),i=[];for(let o=0;o<s.length-1;o++){const c=this.indexById.get(s[o]),l=this.indexById.get(s[o+1]),f=e.startFrame+24+o*42;a[o*4]=c,a[o*4+1]=l,a[o*4+2]=f,a[o*4+3]=ue.backwardCause,i.push({sourceIndex:c,targetIndex:l,beatFrame:f,kind:ue.backwardCause,beatKind:"receipt-path",nodeId:s[o+1],label:"receipt-backed candidate path"})}this.renderer.setPathSteps(a,i);return}const r=br(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});r.steps.length>0&&this.renderer.setPathSteps(r.data,r.steps)}}neighborsOf(e){const t=this.indexById.get(e);if(t===void 0)return[];const r=[];for(const s of this.graph.edges)if(s.sourceIndex===t?r.push(this.graph.nodes[s.targetIndex].id):s.targetIndex===t&&r.push(this.graph.nodes[s.sourceIndex].id),r.length>=12)break;return r}drain(e){var i;const t=this.engine.params;this.edgesDirty&&(this.renderer.setEdges(this.liveEdges),this.edgesDirty=!1);const r=this.projectionDays(),s=this.chronoOffsetDays();if(t[oe.projectionDays]=Math.max(0,r),(this.hasLiveDecay||s!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||r!==this.lastProj||s!==this.lastChrono)&&(this.recomputeDecay(r,s),this.lastDecayFrame=e,this.lastProj=r,this.lastChrono=s),this.active){const o=or[this.active.kind]??300,c=e-this.active.startFrame;c>o+140?(this.active=null,t[oe.liveKind]=ie.none,t[oe.liveEnergy]=0):(t[oe.liveKind]=this.active.kind,t[oe.liveFrame]=Math.max(0,c),t[oe.liveEnergy]=this.energyEnvelope(this.active,c,!1))}else t[oe.liveKind]=ie.none,t[oe.liveEnergy]=0;(i=this.onApply)==null||i.call(this,{simFrame:e,activeKind:t[oe.liveKind],eventsSeen:this.eventsSeen})}debugState(){const e=this.engine.params;return{activeKind:e[oe.liveKind],liveEnergy:e[oe.liveEnergy],liveFrame:e[oe.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}energyEnvelope(e,t,r){if(t<0)return 0;const s=or[e.kind]??300;if(e.kind===ie.dreamStorm){const o=Math.min(1,t/45),c=1-Math.max(0,(t-(s-90))/90),l=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,o*Math.min(1,c)*l)}const a=Math.min(1,t/24),i=1-Math.max(0,(t-s)/140);return Math.max(0,a*Math.min(1,i))}recomputeDecay(e,t=0){const r=this.engine.wallNowMs,s=this.graph.nodes;if(t!==0){const a=r+(t+Math.max(0,e))*Ct;for(let i=0;i<s.length;i++){const o=s[i];this.retention[i]=o.stability!==void 0||o.createdAt?ta(o.stability,o.lastAccessed,o.createdAt,a):Math.max(.001,o.retention)}}else for(let a=0;a<s.length;a++){const i=s[a];this.retention[a]=i.stability!==void 0&&i.lastAccessed?ra(i.stability,i.lastAccessed,r,e):Math.max(.001,i.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){const e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}}function lr(n,e){return n<e?`${n}-${e}`:`${e}-${n}`}function cr(n){var r;const e=(r=n.data)==null?void 0:r.timestamp;if(typeof e!="string")return 0;const t=Date.parse(e);return Number.isFinite(t)?t:0}function Le(n){return typeof n=="string"?n:""}function We(n){return typeof n=="number"&&Number.isFinite(n)?n:0}function dr(n){return Array.isArray(n)?n.filter(e=>typeof e=="string"):[]}function na(n){if(!Array.isArray(n))return[];const e=[];for(const t of n)Array.isArray(t)&&t.length>=2&&typeof t[0]=="string"&&typeof t[1]=="string"&&e.push([t[0],t[1]]);return e}const nt=512,Pt=4,aa=`
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

struct ShuttleState {
	scrub: f32,
	days: f32,
	density: f32,
	dragging: f32,
};

// x normalized timeline position; y kind (0 birth / 1 review); z retention;
// w suppression marker.  One vec4 per real lifecycle event, fixed after load.
struct Dwell { data: vec4f };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dwells: array<Dwell>;
@group(0) @binding(2) var<uniform> shuttle: ShuttleState;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

const RAIL_Y = -0.685;
const RAIL_LEFT = -0.835;
const RAIL_RIGHT = 0.835;

fn rail_x(t: f32) -> f32 { return mix(RAIL_LEFT, RAIL_RIGHT, clamp(t, 0.0, 1.0)); }

// A compact erf approximation makes the beam's edges physically continuous
// rather than a CSS-style blur.  It is evaluated only in fragments of a small
// quad and has no texture/noise dependency.
fn erf_approx(x: f32) -> f32 {
	let s = select(-1.0, 1.0, x >= 0.0);
	let a = abs(x);
	let t = 1.0 / (1.0 + 0.3275911 * a);
	let p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t);
	return s * (1.0 - p * exp(-a * a));
}

struct RailOut {
	@builtin(position) clip: vec4f,
	@location(0) local: vec2f,
};

@vertex
fn vs_rail(@builtin(vertex_index) vi: u32) -> RailOut {
	let q = QUAD[vi];
	// Pixel floor: NDC fractions collapse below a device pixel on narrow
	// viewports (0.022 of a 375px-wide phone is invisible). viewport_w/h ride
	// params lanes 6-7.
	let py = 2.0 / max(params.viewport_h, 1.0);
	var out: RailOut;
	out.clip = vec4f(mix(RAIL_LEFT, RAIL_RIGHT, q.x * 0.5 + 0.5), RAIL_Y + q.y * max(0.022, py * 5.0), 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_rail(in: RailOut) -> @location(0) vec4f {
	let t = in.local.x * 0.5 + 0.5;
	let past = vec3f(0.075, 0.104, 0.088);   // graphite jade: known history
	let now = vec3f(0.48, 0.58, 0.42);       // quiet chalk-lichen at NOW
	let future = vec3f(0.31, 0.19, 0.09);    // fossil amber: projected debt
	let base = select(mix(past, now, t / max(shuttle.scrub, 0.001)), mix(now, future, (t - shuttle.scrub) / max(1.0 - shuttle.scrub, 0.001)), t > shuttle.scrub);
	let midline = 1.0 - smoothstep(0.09, 0.72, abs(in.local.y));
	let tick = smoothstep(0.03, 0.0, abs(fract(t * 24.0) - 0.5));
	let nowRim = exp(-pow((t - shuttle.scrub) * 88.0, 2.0));
	let color = base * (0.40 + midline * 0.46) + vec3f(0.76, 0.82, 0.66) * nowRim * 0.17 + vec3f(0.36, 0.31, 0.20) * tick * 0.12;
	return vec4f(color, 0.82 * midline + tick * 0.12);
}

struct DwellOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) kind: f32,
	@location(2) @interpolate(flat) retention: f32,
	@location(3) @interpolate(flat) suppressed: f32,
	@location(4) @interpolate(flat) distance_to_scrub: f32,
};

@vertex
fn vs_dwell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DwellOut {
	let d = dwells[ii].data;
	let q = QUAD[vi];
	let density = clamp(shuttle.density, 0.0, 1.0);
	let height = (0.034 + d.z * 0.064) * (0.72 + density * 0.44);
	// Never thinner than ~1.6 device px, whatever the viewport width.
	let px = 2.0 / max(params.viewport_w, 1.0);
	let width = max(0.0017 + density * 0.0016, px * 1.6);
	let direction = select(-1.0, 1.0, d.y > 0.5);
	var out: DwellOut;
	out.clip = vec4f(rail_x(d.x) + q.x * width, RAIL_Y + direction * (0.008 + height * (q.y * 0.5 + 0.5)), 0.0, 1.0);
	out.uv = q;
	out.kind = d.y;
	out.retention = d.z;
	out.suppressed = d.w;
	out.distance_to_scrub = abs(d.x - shuttle.scrub);
	return out;
}

@fragment
fn fs_dwell(in: DwellOut) -> @location(0) vec4f {
	let core = 1.0 - smoothstep(0.18, 0.94, abs(in.uv.x));
	let near = exp(-pow(in.distance_to_scrub * 105.0, 2.0));
	let birth = vec3f(0.53, 0.72, 0.57);
	let review = mix(vec3f(0.48, 0.32, 0.15), vec3f(0.83, 0.80, 0.60), in.retention);
	let injury = vec3f(0.56, 0.20, 0.16);
	var color = select(birth, review, in.kind > 0.5);
	// Suppression is a PRESENT-DAY fact (suppression_count > 0). Only the
	// latest-access mark may honestly carry the injury tint — smearing it onto
	// the birth mark would claim the memory was suppressed at creation.
	color = mix(color, injury, in.suppressed * 0.72 * step(0.5, in.kind));
	// Dwell proximity produces the only noticeable glow: real event density,
	// never a permanently luminous UI element.
	color = color * (0.36 + in.retention * 0.42 + near * 0.62);
	return vec4f(color, core * (0.38 + near * 0.54));
}

struct HeadOut { @builtin(position) clip: vec4f, @location(0) local: vec2f };

@vertex
fn vs_head(@builtin(vertex_index) vi: u32) -> HeadOut {
	let q = QUAD[vi];
	let speed = min(1.0, abs(shuttle.days) / 28.0);
	let height = 0.095 + shuttle.dragging * 0.038 + speed * 0.022;
	let px = 2.0 / max(params.viewport_w, 1.0);
	var out: HeadOut;
	out.clip = vec4f(rail_x(shuttle.scrub) + q.x * max(0.021, px * 7.0), RAIL_Y + q.y * height, 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_head(in: HeadOut) -> @location(0) vec4f {
	let x = in.local.x * 2.55;
	let beam = (erf_approx(x + 1.45) - erf_approx(x - 1.45)) * 0.5;
	let center = exp(-x * x * 3.2);
	let line = smoothstep(0.96, 0.08, abs(in.local.y));
	let color = mix(vec3f(0.64, 0.49, 0.23), vec3f(0.84, 0.96, 0.72), step(0.0, shuttle.days));
	return vec4f(color * (0.22 + center * 0.86), beam * line * (0.46 + center * 0.48));
}
`;function He(n){return Math.max(0,Math.min(1,Number.isFinite(n)?n:0))}function at(n){if(!n)return null;const e=Date.parse(n);return Number.isFinite(e)?e:null}class sa{constructor(e,t){p(this,"engine");p(this,"resources",null);p(this,"bindLayout",null);p(this,"railPipeline",null);p(this,"dwellPipeline",null);p(this,"headPipeline",null);p(this,"dwellCount",0);p(this,"minMs",0);p(this,"maxMs",0);p(this,"state",{scrub:1,days:0,density:0,active:0});this.engine=e,this.upload(t)}setTimeline(e,t=!1){const r=this.engine.wallNowMs,s=Math.max(1,this.maxMs-this.minMs);this.state.scrub=He((r+e*864e5-this.minMs)/s),this.state.days=Number.isFinite(e)?e:0,this.state.active=t?1:0,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){!this.resources||!this.railPipeline||!this.dwellPipeline||!this.headPipeline||(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){var e,t;(e=this.resources)==null||e.dwellBuffer.destroy(),(t=this.resources)==null||t.stateBuffer.destroy(),this.resources=null}upload(e){const t=e.flatMap(f=>[at(f.createdAt),at(f.lastAccessed)]).filter(f=>f!==null),r=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):r-864e5,this.maxMs=Math.max(r+365*864e5,this.minMs+864e5);const s=this.maxMs-this.minMs,a=[];for(const f of e){const u=at(f.createdAt),h=at(f.lastAccessed),d=He(f.retention);u!==null&&a.push({at:u,kind:0,retention:d,suppressed:f.suppressed?1:0}),h!==null&&h!==u&&a.push({at:h,kind:1,retention:d,suppressed:f.suppressed?1:0})}a.sort((f,u)=>f.at-u.at);const i=Math.max(1,Math.ceil(a.length/nt)),o=a.filter((f,u)=>u%i===0).slice(0,nt);this.dwellCount=o.length,this.state={scrub:He((r-this.minMs)/s),days:0,density:He(o.length/96),active:0};const c=this.engine.gpuDevice;if(!c||!this.engine.paramsBuffer||(this.ensurePipelines(c),this.ensureResources(c),!this.resources))return;const l=new Float32Array(nt*Pt);o.forEach((f,u)=>{l.set([He((f.at-this.minMs)/s),f.kind,f.retention,f.suppressed],u*Pt)}),c.queue.writeBuffer(this.resources.dwellBuffer,0,l),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"fossil-light-chrono-shuttle-wgsl",code:aa});this.bindLayout=e.createBindGroupLayout({label:"fossil-light-chrono-shuttle-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const r=e.createPipelineLayout({label:"fossil-light-chrono-shuttle-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),s={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},a=(i,o,c)=>e.createRenderPipeline({label:i,layout:r,vertex:{module:t,entryPoint:o},fragment:{module:t,entryPoint:c,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:"triangle-list"}});this.railPipeline=a("fossil-light-chrono-rail","vs_rail","fs_rail"),this.dwellPipeline=a("fossil-light-chrono-dwells","vs_dwell","fs_dwell"),this.headPipeline=a("fossil-light-chrono-head","vs_head","fs_head")}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;const t=e.createBuffer({label:"fossil-light-chrono-dwell-events",size:nt*Pt*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r=e.createBuffer({label:"fossil-light-chrono-state",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:r,bindGroup:e.createBindGroup({label:"fossil-light-chrono-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:r}}]})}}writeState(){const e=this.engine.gpuDevice;!e||!this.resources||e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active]))}}const Rt=64,oa=12,ge=32,la=4,st=256,St="rgba8unorm",ca=96e3,ur=5,da=`
struct CascadeConfig {
	resolution: vec2u,
	emitter_count: u32,
	step_pixels: u32,
	exposure: f32,
	enabled: f32,
	_padding: vec2f,
};

struct Emitter {
	// xy = normalized position, z = normalized source radius, w reserved
	position_radius: vec4f,
	// rgb = semantic memory color supplied by the host, a = FSRS retention
	color_energy: vec4f,
	// x = 1 when suppressed (therefore a non-emitter), rest reserved
	flags: vec4f,
};

// These two layouts intentionally mirror NodeRenderer's live buffers. The
// projection pass runs after its simulation, so a source is located at the
// actual moving 3D node and carries the actual Chrono/FSRS value for this
// frame. No CPU projection, approximation, or GPU readback enters the loop.
struct NodeState {
	pos_radius: vec4f,
	vel_retention: vec4f,
	color_flags: vec4f,
	demo: vec4f,
};

struct Camera {
	view_proj: mat4x4f,
	right: vec4f,
	up: vec4f,
};

@group(0) @binding(0) var<uniform> cascade: CascadeConfig;
@group(0) @binding(1) var<storage, read> emitters: array<Emitter>;
@group(0) @binding(2) var light_out: texture_storage_2d<rgba8unorm, write>;

const MAX_EMITTERS = ${Rt}u;

fn fossil_tone(raw: vec3f, retention: f32) -> vec3f {
	let amber = vec3f(0.62, 0.28, 0.10);
	let jade = vec3f(0.28, 0.68, 0.48);
	let physical = mix(amber, jade, smoothstep(0.14, 0.90, retention));
	// Keep a trace of a memory's semantic hue without reviving the old
	// blue-violet dashboard palette as a light source.
	let grounded = vec3f(
		clamp(raw.r, 0.0, 1.0),
		max(clamp(raw.g, 0.0, 1.0), clamp(raw.b, 0.0, 1.0) * 0.70),
		min(clamp(raw.b, 0.0, 1.0), clamp(raw.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded, 0.16);
}

// Source projection. The host supplies a bounded, deterministic list of
// indices once after graph upload; all spatial and temporal values below come
// directly from NodeRenderer's current GPU buffers.
@group(3) @binding(0) var<uniform> project_config: CascadeConfig;
@group(3) @binding(1) var<storage, read> source_indices: array<u32>;
@group(3) @binding(2) var<storage, read> nodes: array<NodeState>;
@group(3) @binding(3) var<uniform> camera: Camera;
@group(3) @binding(4) var<storage, read_write> projected_emitters: array<Emitter>;

@compute @workgroup_size(64)
fn cs_project_sources(@builtin(global_invocation_id) gid: vec3u) {
	let i = gid.x;
	if (i >= project_config.emitter_count) { return; }
	let source_index = source_indices[i];
	if (source_index >= arrayLength(&nodes)) {
		// A stale index (graph regrown smaller) must be a non-emitter, not a
		// robust-access read of node 0's state.
		var dead: Emitter;
		dead.position_radius = vec4f(0.5, 0.5, 0.012, 0.0);
		dead.color_energy = vec4f(0.0);
		dead.flags = vec4f(1.0, 0.0, 0.0, 0.0);
		projected_emitters[i] = dead;
		return;
	}
	var out: Emitter;
	out.position_radius = vec4f(0.5, 0.5, 0.012, 0.0);
	out.color_energy = vec4f(0.0);
	out.flags = vec4f(1.0, 0.0, 0.0, 0.0);
	let node = nodes[source_index];
	let clip = camera.view_proj * vec4f(node.pos_radius.xyz, 1.0);
	let retention = clamp(node.vel_retention.w, 0.0, 1.0);
	let uv = clip.xy / max(clip.w, 0.0001) * vec2f(0.5, -0.5) + vec2f(0.5);
	let in_view = clip.w > 0.0001 && all(uv >= vec2f(-0.08)) && all(uv <= vec2f(1.08));
	let flags = u32(round(node.color_flags.w));
	let suppressed = (flags & 2u) != 0u;
	let projected_radius = clamp(node.pos_radius.w * 0.012 / max(abs(clip.w), 0.01), 0.008, 0.055);
	if (in_view && retention > 0.0005) {
		out.position_radius = vec4f(uv, projected_radius, 0.0);
		out.color_energy = vec4f(fossil_tone(node.color_flags.rgb, retention), retention);
		out.flags = vec4f(select(0.0, 1.0, suppressed), 0.0, 0.0, 0.0);
	}
	projected_emitters[i] = out;
}

fn inside(pixel: vec2u) -> bool {
	return pixel.x < cascade.resolution.x && pixel.y < cascade.resolution.y;
}

// Direct source splat.  This is deliberately not a screen-space bloom: every
// contribution originates in one supplied memory emitter and is retention
// weighted before it enters the transport field.
@compute @workgroup_size(8, 8)
fn cs_seed(@builtin(global_invocation_id) gid: vec3u) {
	let pixel = gid.xy;
	if (!inside(pixel)) { return; }
	let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(cascade.resolution);
	var radiance = vec3f(0.0);
	for (var i = 0u; i < MAX_EMITTERS; i = i + 1u) {
		if (i >= cascade.emitter_count) { break; }
		let source = emitters[i];
		let delta = uv - source.position_radius.xy;
		let radius = max(source.position_radius.z, 0.008);
		let distance_sq = dot(delta, delta);
		let falloff = exp(-distance_sq / (radius * radius * 1.72));
		let visible = source.color_energy.w * (1.0 - clamp(source.flags.x, 0.0, 1.0));
		radiance = radiance + source.color_energy.rgb * visible * falloff;
	}
	textureStore(light_out, vec2i(pixel), vec4f(clamp(radiance, vec3f(0.0), vec3f(1.0)), 1.0));
}

// A compact, fixed transport cascade.  The successive radii move memory light
// through 4px, 13px, and 37px neighborhoods without unbounded ray marching or
// a history buffer.  It is intentionally a graceful direct-light field, not a
// false claim of scene-aware shadowing before the engine has an occluder mask.
@group(1) @binding(0) var<uniform> transport: CascadeConfig;
@group(1) @binding(1) var light_in: texture_2d<f32>;
@group(1) @binding(2) var transported_out: texture_storage_2d<rgba8unorm, write>;

const DIRECTIONS = array<vec2i, 8>(
	vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1),
	vec2i(1, 1), vec2i(-1, 1), vec2i(1, -1), vec2i(-1, -1)
);

fn bounded_pixel(pixel: vec2i) -> vec2i {
	let hi = vec2i(transport.resolution) - vec2i(1);
	return clamp(pixel, vec2i(0), hi);
}

@compute @workgroup_size(8, 8)
fn cs_transport(@builtin(global_invocation_id) gid: vec3u) {
	let pixel_u = gid.xy;
	if (pixel_u.x >= transport.resolution.x || pixel_u.y >= transport.resolution.y) { return; }
	let pixel = vec2i(pixel_u);
	var radiance = textureLoad(light_in, pixel, 0).rgb * 0.52;
	let step = i32(max(transport.step_pixels, 1u));
	for (var i = 0u; i < 8u; i = i + 1u) {
		let neighbor = bounded_pixel(pixel + DIRECTIONS[i] * step);
		radiance = radiance + textureLoad(light_in, neighbor, 0).rgb * 0.06;
	}
	textureStore(transported_out, pixel, vec4f(clamp(radiance, vec3f(0.0), vec3f(1.0)), 1.0));
}

@group(2) @binding(0) var<uniform> composite: CascadeConfig;
@group(2) @binding(1) var light_field: texture_2d<f32>;

struct CompositeOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
};

@vertex
fn vs_composite(@builtin(vertex_index) vertex_index: u32) -> CompositeOut {
	let quad = array<vec2f, 6>(
		vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
		vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
	);
	let position = quad[vertex_index];
	var out: CompositeOut;
	out.clip = vec4f(position, 0.0, 1.0);
	out.uv = position * vec2f(0.5, -0.5) + vec2f(0.5);
	return out;
}

fn sample_field(uv: vec2f) -> vec3f {
	let size = vec2f(composite.resolution);
	let p = uv * size - vec2f(0.5);
	let base = vec2i(floor(p));
	let fraction = fract(p);
	let hi = vec2i(composite.resolution) - vec2i(1);
	let a = textureLoad(light_field, clamp(base, vec2i(0), hi), 0).rgb;
	let b = textureLoad(light_field, clamp(base + vec2i(1, 0), vec2i(0), hi), 0).rgb;
	let c = textureLoad(light_field, clamp(base + vec2i(0, 1), vec2i(0), hi), 0).rgb;
	let d = textureLoad(light_field, clamp(base + vec2i(1, 1), vec2i(0), hi), 0).rgb;
	return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

@fragment
fn fs_composite(in: CompositeOut) -> @location(0) vec4f {
	let radiance = sample_field(in.uv);
	let luminance = dot(radiance, vec3f(0.2126, 0.7152, 0.0722));
	// A restrained, signal-gated contribution: the light field reads as local
	// illumination instead of a full-screen purple or bloom blanket.
	let signal = smoothstep(0.012, 0.18, luminance) * composite.enabled;
	let vignette = 1.0 - 0.22 * dot(in.uv - vec2f(0.5), in.uv - vec2f(0.5));
	let color = radiance * composite.exposure * max(vignette, 0.72);
	return vec4f(color, signal * 0.54);
}
`;function ua(n,e){return Number.isFinite(n)?n:e}class fa{constructor(e,t,r){p(this,"engine");p(this,"renderer");p(this,"sourceIndices");p(this,"resources",null);p(this,"projectionPipeline",null);p(this,"seedPipeline",null);p(this,"transportPipeline",null);p(this,"compositePipeline",null);p(this,"projectionLayout",null);p(this,"seedLayout",null);p(this,"transportLayout",null);p(this,"compositeLayout",null);p(this,"emitterCount");p(this,"active",!1);p(this,"dirty",!0);p(this,"lastComputedFrame",-ur);p(this,"disposed",!1);p(this,"disabledReason",null);p(this,"exposure",.42);p(this,"configBytes",new ArrayBuffer(ge));p(this,"configUints",new Uint32Array(this.configBytes));p(this,"configFloats",new Float32Array(this.configBytes));this.engine=e,this.renderer=t;const s=[...new Set([...r].filter(a=>Number.isFinite(a)&&a>=0))].sort((a,i)=>a-i).slice(0,Rt);this.sourceIndices=new Uint32Array(s),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?"half-res-transport":"disabled"}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,ua(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;const r=this.engine.gpuDevice;if(!r||!this.engine.paramsBuffer)return;const s=this.renderer.getFossilLightSources();if(!s)return;const a=this.fieldDimensions();if(a===null)return;const i=t-this.lastComputedFrame;if(!(this.active||this.dirty||i<0||i>=ur))return;try{this.ensurePipelines(r),this.ensureResources(r,a.width,a.height,s)}catch{this.disable("GPU light field unavailable on this adapter");return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(r,0,this.resources.width,this.resources.height,0);const c=Math.ceil(this.resources.width/8),l=Math.ceil(this.resources.height/8),f=e.beginComputePass({label:"fossil-light-half-res-transport"});f.setPipeline(this.projectionPipeline),f.setBindGroup(3,this.resources.projectionBindGroup,[0]),f.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),f.setPipeline(this.seedPipeline),f.setBindGroup(0,this.resources.seedBindGroup,[0]),f.dispatchWorkgroups(c,l);for(const[u,h,d]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(r,u,this.resources.width,this.resources.height,h),f.setPipeline(this.transportPipeline),f.setBindGroup(1,d,[u*st]),f.dispatchWorkgroups(c,l);f.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason!==null||!this.resources||!this.compositePipeline||this.emitterCount===0||(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*st]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){const e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;const r=e*.5*(t*.5),a=.5*Math.min(1,Math.sqrt(ca/Math.max(1,r)));return{width:Math.max(1,Math.floor(e*a)),height:Math.max(1,Math.floor(t*a))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;const t=e.createShaderModule({label:"fossil-light-radiance-cascade-wgsl",code:da}),r=e.createBindGroupLayout({label:"fossil-light-empty-layout",entries:[]});this.projectionLayout=e.createBindGroupLayout({label:"fossil-light-source-projection-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),this.seedLayout=e.createBindGroupLayout({label:"fossil-light-seed-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:St}}]}),this.transportLayout=e.createBindGroupLayout({label:"fossil-light-transport-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:St}}]}),this.compositeLayout=e.createBindGroupLayout({label:"fossil-light-composite-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]}),this.seedPipeline=e.createComputePipeline({label:"fossil-light-seed",layout:e.createPipelineLayout({label:"fossil-light-seed-pipeline-layout",bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:"cs_seed"}}),this.projectionPipeline=e.createComputePipeline({label:"fossil-light-source-projection",layout:e.createPipelineLayout({label:"fossil-light-source-projection-pipeline-layout",bindGroupLayouts:[r,r,r,this.projectionLayout]}),compute:{module:t,entryPoint:"cs_project_sources"}}),this.transportPipeline=e.createComputePipeline({label:"fossil-light-transport",layout:e.createPipelineLayout({label:"fossil-light-transport-pipeline-layout",bindGroupLayouts:[r,this.transportLayout]}),compute:{module:t,entryPoint:"cs_transport"}}),this.compositePipeline=e.createRenderPipeline({label:"fossil-light-composite",layout:e.createPipelineLayout({label:"fossil-light-composite-pipeline-layout",bindGroupLayouts:[r,r,this.compositeLayout]}),vertex:{module:t,entryPoint:"vs_composite"},fragment:{module:t,entryPoint:"fs_composite",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]}})}ensureResources(e,t,r,s){var d;if(((d=this.resources)==null?void 0:d.width)===t&&this.resources.height===r&&this.resources.nodeBuffer===s.nodeBuffer&&this.resources.cameraBuffer===s.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;const a=e.createBuffer({label:"fossil-light-projected-memory-emitters",size:Rt*oa*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),i=e.createBuffer({label:"fossil-light-source-indices",size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(i,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);const o=e.createBuffer({label:"fossil-light-cascade-config",size:st*la,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=g=>e.createTexture({label:g,size:[t,r],format:St,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),l=c("fossil-light-field-a"),f=c("fossil-light-field-b"),u=l.createView(),h=f.createView();this.resources={width:t,height:r,emitterBuffer:a,sourceIndexBuffer:i,configBuffer:o,fieldA:l,fieldB:f,seedBindGroup:e.createBindGroup({label:"fossil-light-seed-bind-group",layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:{buffer:a}},{binding:2,resource:u}]}),propagateABindGroup:e.createBindGroup({label:"fossil-light-transport-a-to-b",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:u},{binding:2,resource:h}]}),propagateBBindGroup:e.createBindGroup({label:"fossil-light-transport-b-to-a",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:h},{binding:2,resource:u}]}),projectionBindGroup:e.createBindGroup({label:"fossil-light-source-projection-bind-group",layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:s.nodeBuffer}},{binding:3,resource:{buffer:s.cameraBuffer}},{binding:4,resource:{buffer:a}}]}),compositeBindGroup:e.createBindGroup({label:"fossil-light-composite-bind-group",layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:h}]}),nodeBuffer:s.nodeBuffer,cameraBuffer:s.cameraBuffer},this.dirty=!0}writeConfig(e,t,r,s,a){this.resources&&(this.configUints[0]=r,this.configUints[1]=s,this.configUints[2]=this.emitterCount,this.configUints[3]=a,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*st,this.configBytes))}destroyResources(){var e,t,r,s,a;(e=this.resources)==null||e.emitterBuffer.destroy(),(t=this.resources)==null||t.sourceIndexBuffer.destroy(),(r=this.resources)==null||r.configBuffer.destroy(),(s=this.resources)==null||s.fieldA.destroy(),(a=this.resources)==null||a.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}}var ha=J('<div class="flex items-baseline gap-2 font-mono text-[11px]"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></div>'),pa=J(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),ma=J(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),ga=J(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"> </button>`),va=J('<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>'),ba=J('<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>'),ya=J(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),_a=J("<button> </button>"),wa=J('<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>'),xa=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>'),Ba=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>'),Pa=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>'),Sa=J('<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>'),ka=J('<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div>');function ja(n,e){ct(e,!0);const t=()=>li(di,"$eventFeed",r),[r,s]=ci();let a=V(e,"seed",3,"vestige-observatory-v1"),i=V(e,"freezeFrame",3,null),o=V(e,"capture",3,!1),c=V(e,"showSwitcher",3,!0),l=V(e,"embedded",3,!1),f=V(e,"chrome",3,"full"),u=V(e,"focusIds",19,()=>[]),h=V(e,"live",3,!1),d=U(0),g=U(!1),_=U(0),x=null;const S=Ie(()=>Math.max(0,m(d))),v=Ie(()=>Math.min(0,m(d))),M=Ie(()=>m(d)===0?"now":m(d)>0?`+${Math.round(m(d))}d`:new Date(Date.now()+m(d)*864e5).toLocaleDateString(void 0,{month:"short",day:"numeric"}));let R=null,O=null,N=null,X=U(!1),C=U(!1);const j=.835,se=(1+.685)/2;let $=!1,ce=0,re=0,W=0,Q=!1;function w(b){var B;const y=(B=m(Ae))==null?void 0:B.getBoundingClientRect();if(!y||y.width===0)return m(d);const G=(b-y.left)/y.width*2-1,z=Math.max(0,Math.min(1,G/j*.5+.5));return m(_)+z*(365-m(_))}function L(b){var z;const y=(z=m(Ae))==null?void 0:z.getBoundingClientRect();if(!y||y.height===0)return!1;const G=(b.clientY-y.top)/y.height;return G>se-.075&&G<se+.075}function Ge(){ce&&cancelAnimationFrame(ce),ce=0}function ft(b){var y,G;!m(X)||o()||!L(b)||(Ge(),$=!0,P(g,!0),re=0,W=performance.now(),P(d,w(b.clientX),!0),(G=(y=b.currentTarget).setPointerCapture)==null||G.call(y,b.pointerId),b.preventDefault())}function kr(b){if(!$)return;const y=performance.now(),G=w(b.clientX),z=Math.max(1,y-W);re=re*.6+(G-m(d))/z*16*.4,W=y,P(d,G,!0)}function Ir(b){var G,z;if(!$)return;$=!1,Q=!0,(z=(G=b.currentTarget).releasePointerCapture)==null||z.call(G,b.pointerId);const y=()=>{ce=0,re*=.94;let B=m(d)+re;B<=m(_)&&(B=m(_),re=0),B>=365&&(B=365,re=0);const Z=m(d)<0&&re>0||m(d)>0&&re<0;Math.abs(B)<1&&Z&&(B=0,re=0),P(d,B,!0),Math.abs(re)>.02?ce=requestAnimationFrame(y):P(g,!1)};Math.abs(re)>.05?ce=requestAnimationFrame(y):P(g,!1)}function Rr(){$=!1,P(g,!1),Ge()}let ye=U(!1),ht=U(!1);function Er(){if(typeof window>"u")return;const b=window.matchMedia("(prefers-reduced-motion: reduce)");b.matches&&!m(ht)&&P(ye,!0);const y=G=>{m(ht)||P(ye,G.matches,!0)};return b.addEventListener("change",y),()=>b.removeEventListener("change",y)}function Mt(){P(ht,!0),P(ye,!m(ye))}Oe(()=>{var b;(b=m(de))==null||b.setPaused(m(ye))});let Ke=U(""),Ye=U(0),Ar=Ie(()=>m(Ke)!==""&&m(Ye)>0),Ae=U(null);async function Cr(b){if(Q){Q=!1;return}if(!e.onpick||!F||!m(Ae))return;const y=m(Ae).getBoundingClientRect();if(y.width===0||y.height===0)return;const G=(b.clientX-y.left)/y.width*2-1,z=-((b.clientY-y.top)/y.height*2-1),B=await F.pickAt(G,z);B&&e.onpick(B.id)}const Ft={"recall-path":"RECALL","engram-birth":"BIRTH","salience-rescue":"RESCUE","forgetting-horizon":"HORIZON",firewall:"FIREWALL"};function Mr(){return F!=null&&F.graph?new Uint32Array(F.graph.nodes.map(b=>({index:b.index,id:b.id,retention:b.retention})).sort((b,y)=>y.retention-b.retention||b.id.localeCompare(y.id)).slice(0,64).map(b=>b.index).sort((b,y)=>b-y)):new Uint32Array}let pt=U(!o());function Fr(b){const y=b.target;y!=null&&y.isContentEditable||(y==null?void 0:y.tagName)==="INPUT"||(y==null?void 0:y.tagName)==="TEXTAREA"||(y==null?void 0:y.tagName)==="SELECT"||((b.key==="h"||b.key==="H")&&P(pt,!m(pt)),b.key==="Escape"&&e.onexit&&e.onexit(),(b.key===" "||b.key.toLowerCase()==="p")&&!o()&&(b.preventDefault(),Mt()))}let _e=U(null),Ce=U(!0),De=U(""),Te=U(0),Lt=U(0),mt=U(0),gt=U(0),vt=U(""),de=U(null),F=null,Xe=null,Gt=null,bt=U(null),Dt=null,Tt=null,ze=U(null),yt=!1,Me=U(Xt([]));async function Lr(){P(Ce,!0),P(De,"");try{const b=new Set(u().filter(Boolean)),y=b.size?await(async()=>{var H,K;const G=await Promise.all([...b].map(T=>wt.graph({center_id:T,max_nodes:200,depth:3}))),z=[...new Map(G.flatMap(T=>T.nodes).map(T=>[T.id,T])).values()].filter(T=>b.has(T.id)),B=new Set(z.map(T=>T.id)),Z=[...new Map(G.flatMap(T=>T.edges).map(T=>[`${T.source}:${T.target}`,T])).values()].filter(T=>B.has(T.source)&&B.has(T.target));return{...G[0],nodes:z,edges:Z,center_id:((H=z[0])==null?void 0:H.id)??((K=G[0])==null?void 0:K.center_id)??"",nodeCount:z.length,edgeCount:Z.length}})():await wt.graph({max_nodes:200,depth:3,sort:"connected"});P(_e,y,!0),P(mt,y.nodeCount,!0),P(gt,y.edgeCount,!0),P(vt,y.center_id,!0)}catch(b){const y=b instanceof Error?b.message:"Failed to load graph data";/\b404\b/.test(y)?(P(_e,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:""},!0),P(mt,0),P(gt,0),P(vt,"")):P(De,y,!0)}finally{P(Ce,!1)}}let Qe=null,Ze=U(Xt([])),Je=U("recalls");function Gr(b,y){P(Te,b,!0),P(Lt,y,!0),Qe&&!m(g)&&Qe.tick(b)}async function Dr(){var H;if(!R||!(F!=null&&F.graph))return;const b=F.graph,y=K=>b.indexById.has(K),G=K=>{var T;return((T=b.nodes[b.indexById.get(K)??-1])==null?void 0:T.label)??K.slice(0,8)};let z=[];try{z=((H=await wt.receipts.list(60))==null?void 0:H.receipts)??[]}catch{}let B=gi(z,y);B.length===0&&(B=vi(b.nodes,12)),B.length>0&&(Qe=new yi(R,{intervalFrames:240}),Qe.setItems(B));const Z=bi(z,y,3);Z.length>0?(P(Je,"recalls"),P(Ze,Z.map(K=>({...K,label:G(K.id)})),!0)):(P(Je,"retention"),P(Ze,[...b.nodes].filter(K=>(K.label??"").trim().length>0).sort((K,T)=>T.retention-K.retention).slice(0,3).map(K=>({id:K.id,recalls:Math.round(K.retention*100),label:K.label||K.id.slice(0,8)})),!0))}function Tr(b){var y;yt=!1,P(de,b,!0),F=new ji(b),(y=e.onready)==null||y.call(e,b)}Oe(()=>{if(m(de)&&F&&m(_e)&&!yt){yt=!0;const b=e.demo==="engram-birth",y=e.demo==="salience-rescue",G=e.demo==="forgetting-horizon",z=e.demo==="firewall";if(F.upload(m(_e),a(),{recallPath:!b&&!y&&!G&&!z}),b){Xe=new an({engine:m(de),nodeRenderer:F,seed:a()}),Xe.upload(a());const B=Xe.engraveSteps,Z=[];for(let H=0;H<B.length/4;H++)Z.push({sourceIndex:B[H*4],targetIndex:B[H*4+1],beatFrame:B[H*4+2],kind:B[H*4+3],beatKind:"engrave",nodeId:`engrave-${H}`,label:"edge engraved"});F.setPathSteps(B,Z),P(Me,Xe.timeline.map((H,K)=>({sourceIndex:0,targetIndex:0,beatFrame:H.startFrame,kind:0,beatKind:"birth",nodeId:`birth-${K}`,label:H.label})),!0)}else if(y){const B=wn(m(_e),F.graph,a(),e.backfillEvidence);P(bt,B,!0),B.viable&&(Gt=new ln({engine:m(de),nodeRenderer:F,plan:B}),Gt.upload(),F.setPathSteps(B.pathData,B.pathMetas)),P(Me,B.spineBeats,!0)}else if(G){const B=Gn(F.graph);B.viable&&(Dt=new Sn({engine:m(de),nodeRenderer:F,plan:B}),Dt.upload(),F.setPathSteps(B.pathData,B.pathMetas)),P(Me,B.spineBeats,!0)}else if(z){const B=Zn(F.graph,a());P(ze,B,!0),B.viable&&(Tt=new Br({engine:m(de),nodeRenderer:F,plan:B}),Tt.upload(),F.setPathSteps(B.pathData,B.pathMetas)),P(Me,B.spineBeats,!0)}else P(Me,F.pathSteps,!0);if(h()&&F.graph&&m(_e)){R=new ia({engine:m(de),renderer:F,graph:F.graph,response:m(_e),seed:a(),projectionDays:()=>m(S),chronoOffsetDays:()=>m(v),onFirewall:Z=>{P(Ke,Z.intruderLabel,!0),P(Ye,Date.now(),!0)}}),P(C,R.liveDecayAvailable,!0),m(de).setPreFrameHook(Z=>R==null?void 0:R.drain(Z)),o()||Dr();let B=Number.POSITIVE_INFINITY;for(const Z of F.graph.nodes)if(Z.createdAt){const H=Date.parse(Z.createdAt);Number.isFinite(H)&&H<B&&(B=H)}if(Number.isFinite(B)&&P(_,Math.floor((B-Date.now())/864e5)-1),x){const Z=Date.parse(x);Number.isFinite(Z)&&P(d,Math.min(365,Math.max(m(_),(Z-Date.now())/864e5)),!0),x=null}o()||(N=new fa(m(de),F,Mr()),m(de).addPass(N),O=new sa(m(de),F.graph.nodes),m(de).addPass(O),P(X,!0)),typeof window<"u"&&(window.__vestigeLiveBridge=R)}m(de).demoClock.reset()}}),Oe(()=>{const b=t();R&&R.ingest(b)}),Oe(()=>{m(d),R==null||R.refreshDecay(),O==null||O.setTimeline(m(d),m(g)),N==null||N.setScrubbing(m(g))}),Oe(()=>{if(!m(Ye))return;const b=setTimeout(()=>{P(Ke,""),P(Ye,0)},7e3);return()=>clearTimeout(b)}),ri(()=>{x=new URLSearchParams(window.location.search).get("t"),Lr();const b=Er();return()=>{if(Ge(),b==null||b(),typeof window<"u"){const y=window;y.__vestigeLiveBridge===R&&delete y.__vestigeLiveBridge}}});var $e=ka();rt("keydown",ii,Fr);let zt;var me=A($e);let Ut;var zr=A(me);hi(zr,{get demo(){return e.demo},get seed(){return a()},get freezeFrame(){return i()},onframe:Gr,onready:Tr}),E(me),oi(me,b=>P(Ae,b),()=>m(Ae));var Ur=D(me,2);{var Or=b=>{var y=Sa(),G=A(y);{var z=k=>{var I=pa(),ee=A(I),Y=A(ee,!0);E(ee);var fe=D(ee,2);kt(fe,19,()=>m(Ze),xe=>xe.id,(xe,Be,_t)=>{var Ue=ha(),we=A(Ue),et=A(we,!0);E(we);var tt=D(we,2),Jr=A(tt,!0);E(tt);var Yt=D(tt,2),$r=A(Yt);E(Yt),E(Ue),le(()=>{ne(et,m(_t)+1),ke(tt,"title",m(Be).label),ne(Jr,m(Be).label),ne($r,`${m(Be).recalls??""}${m(Je)==="recalls"?"×":"%"}`)}),q(xe,Ue)}),E(I),le(()=>ne(Y,m(Je)==="recalls"?"Most recalled · your mind":"Strongest memories · your mind")),q(k,I)};te(G,k=>{h()&&m(Ze).length>0&&k(z)})}var B=D(G,2);{var Z=k=>{var I=ma(),ee=D(A(I),2),Y=A(ee,!0);E(ee),ni(2),E(I),le(()=>ne(Y,m(Ke))),q(k,I)};te(B,k=>{h()&&m(Ar)&&k(Z)})}var H=D(B,2);{var K=k=>{var I=ga(),ee=A(I,!0);E(I),le(()=>{ke(I,"title",m(ye)?"Resume field motion":"Pause field motion"),ke(I,"aria-pressed",m(ye)),ke(I,"aria-label",m(ye)?"Resume 3D memory field motion":"Pause 3D memory field motion"),ne(ee,m(ye)?"▶ RESUME":"❚❚ PAUSE")}),pe("click",I,Mt),q(k,I)};te(H,k=>{o()||k(K)})}var T=D(H,2);{var Nr=k=>{var I=ba();let ee;var Y=D(A(I),2);ai(Y);var fe=D(Y,2);let xe;var Be=A(fe,!0);E(fe);var _t=D(fe,2);{var Ue=we=>{var et=va();pe("click",et,()=>P(d,0)),q(we,et)};te(_t,we=>{m(d)!==0&&we(Ue)})}E(I),le(()=>{ee=Re(I,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]`,null,ee,{"opacity-100":m(X),"opacity-75":!m(X)}),ke(Y,"min",m(_)),xe=Re(fe,1,"w-16 text-right tabular-nums",null,xe,{"text-[#b9d9a9]":m(d)>=0,"text-[#dfc68e]":m(d)<0}),ne(Be,m(M))}),pe("input",Y,()=>P(g,!0)),pe("change",Y,()=>P(g,!1)),pe("pointerup",Y,()=>P(g,!1)),rt("pointercancel",Y,()=>P(g,!1)),rt("blur",Y,()=>P(g,!1)),si(Y,()=>m(d),we=>P(d,we)),q(k,I)};te(T,k=>{h()&&m(C)&&k(Nr)})}var Ot=D(T,2);{var jr=k=>{Pi(k,{get demoMode(){return e.demo},get seed(){return a()},get nodeCount(){return m(mt)},get edgeCount(){return m(gt)},get centerId(){return m(vt)},get frameCount(){return m(Te)},get fpsEstimate(){return m(Lt)},get freezeFrame(){return i()},get loading(){return m(Ce)},get error(){return m(De)}})};te(Ot,k=>{f()==="full"&&k(jr)})}var Nt=D(Ot,2);{var qr=k=>{var I=ya();pe("click",I,function(...ee){var Y;(Y=e.onexit)==null||Y.apply(this,ee)}),q(k,I)};te(Nt,k=>{f()==="full"&&e.onexit&&k(qr)})}var jt=D(Nt,2);{var Vr=k=>{var I=wa();kt(I,20,()=>pi,ee=>ee,(ee,Y)=>{var fe=_a(),xe=A(fe,!0);E(fe),le(()=>{Re(fe,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${Y===e.demo?"text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]":"text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50"}`),ke(fe,"title",`Play the ${Ft[Y]??""} moment`),ne(xe,Ft[Y])}),pe("click",fe,()=>{var Be;return(Be=e.ondemochange)==null?void 0:Be.call(e,Y)}),q(ee,fe)}),E(I),q(k,I)};te(jt,k=>{f()==="full"&&c()&&k(Vr)})}var qt=D(jt,2);{var Wr=k=>{var I=xa();q(k,I)};te(qt,k=>{m(Ce)&&k(Wr)})}var Vt=D(qt,2);{var Hr=k=>{var I=Ba(),ee=A(I),Y=A(ee,!0);E(ee),E(I),le(()=>ne(Y,m(De))),q(k,I)};te(Vt,k=>{m(De)&&!m(Ce)&&k(Hr)})}var Wt=D(Vt,2);{var Kr=k=>{Ri(k,{get steps(){return m(Me)},get frame(){return m(Te)}})};te(Wt,k=>{f()==="full"&&k(Kr)})}var Ht=D(Wt,2);{var Yr=k=>{Qt(k,{get frame(){return m(Te)},get verdict(){return m(bt).verdict}})};te(Ht,k=>{var I;f()==="full"&&e.demo==="salience-rescue"&&((I=m(bt))!=null&&I.viable)&&k(Yr)})}var Kt=D(Ht,2);{var Xr=k=>{{let I=Ie(()=>({headline:m(ze).verdict.headline,causeLabel:m(ze).verdict.intruderLabel,receipt:m(ze).verdict.receipt}));Qt(k,{get frame(){return m(Te)},tone:"quarantine",fadeWindow:[480,495,605,620],get verdict(){return m(I)}})}};te(Kt,k=>{var I;f()==="full"&&e.demo==="firewall"&&((I=m(ze))!=null&&I.viable)&&k(Xr)})}var Qr=D(Kt,2);{var Zr=k=>{var I=Pa();q(k,I)};te(Qr,k=>{!m(Ce)&&m(_e)&&m(_e).nodeCount===0&&k(Zr)})}E(y),q(b,y)};te(Ur,b=>{m(pt)&&b(Or)})}E($e),le(()=>{zt=Re($e,1,`${l()?"absolute":"fixed"} inset-0 overflow-hidden bg-[#05060a]`,null,zt,{"cursor-none":o()}),Ut=Re(me,1,"absolute inset-0 z-0 touch-none",null,Ut,{"cursor-crosshair":!!e.onpick&&!o()})}),pe("click",me,Cr),pe("pointerdown",me,ft),pe("pointermove",me,kr),pe("pointerup",me,Ir),rt("pointercancel",me,Rr),q(n,$e),dt(),s()}fr(["click","pointerdown","pointermove","pointerup","input","change"]);export{ja as O};

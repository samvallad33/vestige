var gi=Object.defineProperty;var vi=(i,e,t)=>e in i?gi(i,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):i[e]=t;var u=(i,e,t)=>vi(i,typeof e!="symbol"?e+"":e,t);import"./Bzak7iHL.js";import{d as wr,s as ee,b as he,o as bi,e as Oe}from"./CNdOtqLU.js";import{p as gt,c as M,r as E,j as L,t as se,a as Y,b as vt,f as Q,k as xr,h as Lt,g as m,u as Ce,ao as Ne,d as j,e as ar,s as P,bi as yi,n as _i}from"./Dw_4PDAU.js";import{i as J,b as wi}from"./C7WW_yYn.js";import{e as Mt,s as Ee}from"./S5IcwJEO.js";import{s as Re,r as xi}from"./sbNTdgSE.js";import{b as Pi}from"./BHZaPctr.js";import{p as W}from"./N0ivUMqs.js";import{a as Bi,s as Si}from"./CSsWZzkN.js";import{a as kt}from"./B3oLbNAe.js";import{e as Ii}from"./GBqFEsz8.js";import{b as ki}from"./j8DaofCc.js";import{s as At}from"./BDfpTd4R.js";import{t as Ri,d as Pr,N as _e,U as Rt,F as re,a as je,P as de,b as Ie,D as Dt,L as te,c as ae,O as Ci,e as Ei}from"./C6h4BeTQ.js";import{b as Mi,M as Ai,g as Fi}from"./D3ALi6cg.js";import{p as Gi}from"./CcSRZpDz.js";import{P as Li}from"./DMc8obLx.js";function Di(i,e){var r;const t=[];for(const n of i){const s=(((r=n.activation_path)!=null&&r.length?n.activation_path:n.retrieved)??[]).filter(e);if(s.length===0)continue;const o=s[s.length-1];t.push({targetId:o,pathIds:s})}return t}function Ti(i,e=12){return[...i].sort((t,r)=>r.retention-t.retention||t.id.localeCompare(r.id)).slice(0,e).map(t=>({targetId:t.id,pathIds:[t.id]}))}function zi(i,e,t=5){var n;const r=new Map;for(const a of i){const s=((n=a.activation_path)!=null&&n.length?a.activation_path:a.retrieved)??[];for(const o of new Set(s))e(o)&&r.set(o,(r.get(o)??0)+1)}return[...r.entries()].map(([a,s])=>({id:a,recalls:s})).sort((a,s)=>s.recalls-a.recalls||a.id.localeCompare(s.id)).slice(0,t)}class Ui{constructor(e,t={}){u(this,"bridge");u(this,"items",[]);u(this,"cursor",0);u(this,"ticks",0);u(this,"nextTick",0);u(this,"intervalFrames");u(this,"enabled",!0);u(this,"started",!1);this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}const t=this.items[this.cursor%this.items.length];this.cursor++;const r=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(r?0:30)}}var Oi=Q('<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>'),Ni=Q('<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>'),ji=Q('<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>'),qi=Q('<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>');function Vi(i,e){gt(e,!0);let t=W(e,"demoMode",3,"recall-path"),r=W(e,"seed",3,"vestige-observatory-v1"),n=W(e,"nodeCount",3,0),a=W(e,"edgeCount",3,0),s=W(e,"centerId",3,""),o=W(e,"frameCount",3,0),c=W(e,"fpsEstimate",3,0),l=W(e,"freezeFrame",3,null);W(e,"loading",3,!1),W(e,"error",3,"");function d(){const q=new URLSearchParams({demo:t(),seed:r()});l()!==null&&q.set("frame",String(l()));const z=`${window.location.origin}${ki}/observatory?${q.toString()}`;navigator.clipboard.writeText(z).catch(()=>{})}var f=qi(),h=M(f),p=M(h),v=M(p),b=M(v,!0);E(v);var x=L(v,2),S=M(x);E(x),E(p);var y=L(p,2),D=M(y),U=M(D);E(D);var oe=L(D,2);{var T=q=>{var z=Oi(),w=M(z);E(z),se(F=>ee(w,`center=${F??""}`),[()=>s().slice(0,8)]),Y(q,z)};J(oe,q=>{s()&&q(T)})}E(y);var Z=L(y,2),A=M(Z),H=M(A);E(A);var ie=L(A,2);{var ne=q=>{var z=Ni();Y(q,z)},ge=q=>{var z=ji(),w=M(z);E(z),se(()=>ee(w,`${c()??""}fps`)),Y(q,z)};J(ie,q=>{l()!==null?q(ne):c()>0&&q(ge,1)})}var le=L(ie,2);E(Z),E(h),E(f),se((q,z)=>{ee(b,t()),ee(S,`seed=${q??""}${r().length>12?"…":""}`),ee(U,`${n()??""} nodes · ${a()??""} edges`),ee(H,`frame: ${z??""}`)},[()=>r().slice(0,12),()=>String(o()).padStart(3," ")]),he("click",le,d),Y(i,f),vt()}wr(["click"]);var Wi=Q('<div class="active-label svelte-8n8iia"> </div>'),Yi=Q("<div></div>"),Hi=Q('<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>');function Ki(i,e){gt(e,!0);let t=W(e,"steps",19,()=>[]),r=W(e,"frame",3,0),n=W(e,"loopFrames",3,720);const a=f=>f/n()*100;function s(f,h){const p=h-f;return p<-14||p>90?0:p<0?1+p/14:1-p/90}let o=Ce(()=>{let f="",h=.15;for(const p of t()){const v=s(p.beatFrame,r());v>h&&(h=v,f=p.label)}return f});var c=xr(),l=Lt(c);{var d=f=>{var h=Hi(),p=M(h);{var v=y=>{var D=Wi(),U=M(D,!0);E(D),se(()=>ee(U,m(o))),Y(y,D)};J(p,y=>{m(o)&&y(v)})}var b=L(p,2),x=M(b);Mt(x,17,t,y=>y.beatFrame,(y,D)=>{var U=Yi();let oe;se((T,Z,A)=>{oe=Ee(U,1,"tick svelte-8n8iia",null,oe,T),At(U,`left: ${Z??""}%; opacity: ${A??""}`),Re(U,"title",m(D).label)},[()=>({hot:s(m(D).beatFrame,r())>0,backward:m(D).kind===1}),()=>a(m(D).beatFrame),()=>.45+.55*s(m(D).beatFrame,r())]),Y(y,U)});var S=L(x,2);E(b),E(h),se(y=>At(S,`left: ${y??""}%`),[()=>a(r())]),Y(f,h)};J(l,f=>{t().length>0&&f(d)})}Y(i,c),vt()}var Xi=Q('<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>');function sr(i,e){gt(e,!0);let t=W(e,"frame",3,0),r=W(e,"fadeWindow",19,()=>[600,620,705,719]),n=W(e,"tone",3,"triumph");const a=(d,f,h)=>{const p=Math.min(1,Math.max(0,(h-d)/(f-d)));return p*p*(3-2*p)};let s=Ce(()=>a(r()[0],r()[1],t())*(1-a(r()[2],r()[3],t())));var o=xr(),c=Lt(o);{var l=d=>{var f=Xi();let h;var p=M(f),v=M(p,!0);E(p);var b=L(p,2),x=M(b,!0);E(b);var S=L(b,2),y=M(S,!0);E(S),E(f),se(()=>{h=Ee(f,1,"verdict svelte-ssd7yu",null,h,{quarantine:n()==="quarantine"}),At(f,`opacity: ${m(s)??""}`),ee(v,e.verdict.headline),ee(x,e.verdict.causeLabel),ee(y,e.verdict.receipt)}),Y(d,f)};J(c,d=>{m(s)>.001&&d(l)})}Y(i,o),vt()}function Qi(i,e,t,r){const n=1/Math.tan(i/2),a=1/(t-r),s=new Float32Array(16);return s[0]=n/e,s[5]=n,s[10]=r*a,s[11]=-1,s[14]=r*t*a,s}function Zi(i,e,t){const[r,n,a]=i;let s=r-e[0],o=n-e[1],c=a-e[2],l=Math.hypot(s,o,c)||1;s/=l,o/=l,c/=l;let d=t[1]*c-t[2]*o,f=t[2]*s-t[0]*c,h=t[0]*o-t[1]*s;l=Math.hypot(d,f,h)||1,d/=l,f/=l,h/=l;const p=o*h-c*f,v=c*d-s*h,b=s*f-o*d,x=new Float32Array(16);return x[0]=d,x[1]=p,x[2]=s,x[4]=f,x[5]=v,x[6]=o,x[8]=h,x[9]=b,x[10]=c,x[12]=-(d*r+f*n+h*a),x[13]=-(p*r+v*n+b*a),x[14]=-(s*r+o*n+c*a),x[15]=1,x}function Ji(i,e){const t=new Float32Array(16);for(let r=0;r<4;r++)for(let n=0;n<4;n++)t[r*4+n]=i[n]*e[r*4]+i[4+n]*e[r*4+1]+i[8+n]*e[r*4+2]+i[12+n]*e[r*4+3];return t}function $i(i,e,t,r=.35,n=0){const a=i*Math.PI*2+n,s=[Math.sin(a)*t,t*r,Math.cos(a)*t],o=Qi(50*Math.PI/180,e,.1,4e3),c=Zi(s,[0,0,0],[0,1,0]);let l=-s[0],d=-s[1],f=-s[2],h=Math.hypot(l,d,f)||1;l/=h,d/=h,f/=h;let p=d*0-f*1,v=f*0-l*0,b=l*1-d*0;h=Math.hypot(p,v,b)||1,p/=h,v/=h,b/=h;const x=v*f-b*d,S=b*l-p*f,y=p*d-v*l;return{viewProj:Ji(o,c),right:[p,v,b],up:[x,S,y],eye:s}}const pt={yaw:0,pitch:0,zoom:1},en=.38,tn=2.6,rn=-.18,nn=.82;function Ft(i,e,t){return Math.min(t,Math.max(e,i))}function ft(i){return{yaw:Number.isFinite(i.yaw)?i.yaw:0,pitch:Ft(Number.isFinite(i.pitch)?i.pitch:0,rn,nn),zoom:Ft(Number.isFinite(i.zoom)?i.zoom:1,en,tn)}}function an(i,e,t,r=pt){const n=ft(r);return $i(i,e,t/n.zoom,.35+n.pitch,n.yaw)}class sn{constructor(){u(this,"state",{...pt});u(this,"dragging",!1);u(this,"pointerId",null);u(this,"lastX",0);u(this,"lastY",0);u(this,"pinch0",0);u(this,"pointers",new Map);u(this,"enabled",!0)}reset(){this.state={...pt},this.dragging=!1,this.pointerId=null,this.pointers.clear()}onPointerDown(e){var t,r;!this.enabled||e.button!==0||(this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===1?(this.dragging=!0,this.pointerId=e.pointerId,this.lastX=e.clientX,this.lastY=e.clientY,(r=(t=e.currentTarget)==null?void 0:t.setPointerCapture)==null||r.call(t,e.pointerId)):this.pointers.size===2&&(this.pinch0=or(this.pointers)))}onPointerMove(e){if(!this.enabled)return!1;if(this.pointers.has(e.pointerId)&&this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===2&&this.pinch0>0){const n=or(this.pointers),a=n/this.pinch0;return this.state=ft({...this.state,zoom:this.state.zoom*Ft(a,.94,1.06)}),this.pinch0=n,!0}if(!this.dragging||e.pointerId!==this.pointerId)return!1;const t=e.clientX-this.lastX,r=e.clientY-this.lastY;return this.lastX=e.clientX,this.lastY=e.clientY,this.state=ft({yaw:this.state.yaw-t*.005,pitch:this.state.pitch+r*.003,zoom:this.state.zoom}),!0}onPointerUp(e){this.pointers.delete(e.pointerId),e.pointerId===this.pointerId&&(this.dragging=!1,this.pointerId=null),this.pointers.size<2&&(this.pinch0=0)}onWheel(e){if(!this.enabled)return!1;e.preventDefault();const t=e.deltaY>0?.92:1.08;return this.state=ft({...this.state,zoom:this.state.zoom*t}),!0}}function or(i){const e=[...i.values()];return e.length<2?0:Math.hypot(e[0].x-e[1].x,e[0].y-e[1].y)}function lr(i){const e=/^#?([0-9a-fA-F]{6})$/.exec(i.trim());if(!e)return[107/255,114/255,128/255];const t=parseInt(e[1],16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}function on(i){const e=Mi({tags:i.tags});return lr(e||Ai[Fi(i.retention)])}function ln(i){const t=[...i.nodes].sort((s,o)=>s.isCenter!==o.isCenter?s.isCenter?-1:1:s.id<o.id?-1:s.id>o.id?1:0).map((s,o)=>Ri(s,o)),r=new Map;for(const s of t)r.set(s.id,s.index);const n=[];for(const s of i.edges){const o=r.get(s.source),c=r.get(s.target);o===void 0||c===void 0||o===c||n.push({sourceIndex:o,targetIndex:c,weight:s.weight,type:s.type})}const a=t.findIndex(s=>s.isCenter);return{nodes:t,edges:n,indexById:r,centerIndex:a<0?0:a}}function Br(i,e,t=120){const r=i.nodes.length,n=new Float32Array(r*re);for(let a=0;a<r;a++){const s=i.nodes[a],o=a*re,[c,l,d]=s.isCenter&&i.centerIndex===a?[0,0,0]:Pr(a,r,t,e),f=s.isCenter?4.2:1.4+s.retention*1.8;n[o+_e.posRadius+0]=c,n[o+_e.posRadius+1]=l,n[o+_e.posRadius+2]=d,n[o+_e.posRadius+3]=f,n[o+_e.velRetention+3]=s.retention;const[h,p,v]=on(s);let b=0;s.isCenter&&(b|=je.isCenter),s.suppressed&&(b|=je.suppressed);const x=new Set(s.tags.map(S=>S.toLowerCase()));x.has("aha")&&(b|=je.isAha),(x.has("failure")||x.has("guardrail"))&&(b|=je.isFailure),(x.has("confusion")||x.has("weak-spot"))&&(b|=je.isConfusion),n[o+_e.colorFlags+0]=h,n[o+_e.colorFlags+1]=p,n[o+_e.colorFlags+2]=v,n[o+_e.colorFlags+3]=b}return{data:n,nodeCount:r}}function cr(i){const e=new Uint32Array(Math.max(1,i.edges.length)*Rt);return i.edges.forEach((t,r)=>{e[r*Rt]=t.sourceIndex,e[r*Rt+1]=t.targetIndex}),e}const cn=`
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
`,dn=`
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
`,Sr=`
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
`,un=`
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

// x source index, y target index, z beat frame, w kind (0 recall, 1 backward)
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
// Source/target node indices (2 u32 per edge).
@group(0) @binding(2) var<storage, read> edges: array<vec2<u32>>;
// PathStep buffer for wavefront timing.
@group(0) @binding(3) var<storage, read> path: array<vec4<u32>>;
// NodeState storage buffer (positions for edge endpoints).
@group(0) @binding(4) var<storage, read> nodes: array<Node>;

// Iridescent thin-film band — ported EXACTLY from causal-brain-demo.html
// spectral(w) (visual DNA §7.1): indigo → cyan-teal → mint → magenta rim.
fn spectral(w_in: f32) -> vec3<f32> {
	let w = fract(w_in);
	// Fossil band (doctrine): sediment → amber → jade → chalk. Magenta is
	// reserved for backward-causal kind=1 wavefronts only (RSB).
	let stops = array<vec3<f32>, 4>(
		vec3<f32>(0.18, 0.16, 0.08), // sediment
		vec3<f32>(0.96, 0.62, 0.16), // amber debt
		vec3<f32>(0.16, 0.95, 0.66), // jade recall
		vec3<f32>(0.91, 1.00, 0.72)  // luciferin chalk
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
	@location(0) color: vec3<f32>,
	@location(1) width: f32,
};

@vertex
fn vs_main(
	@builtin(vertex_index) vi: u32,
	@builtin(instance_index) ii: u32
) -> VSOut {
	var out: VSOut;

	let edgeCount = u32(params.edge_count);
	if (ii >= edgeCount) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let edge = edges[ii];
	let srcIdx = edge.x;
	let tgtIdx = edge.y;

	if (srcIdx >= u32(params.node_count) || tgtIdx >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let src = nodes[srcIdx];
	let tgt = nodes[tgtIdx];

	// Two vertices per edge: source (vi=0) and target (vi=1).
	let pos = select(src.pos_radius.xyz, tgt.pos_radius.xyz, vi == 1u);

	// World-space position.
	let world = pos;
	out.clip = camera.view_proj * vec4<f32>(world, 1.0);

	// Wavefront computation: find the nearest path beat for this edge.
	let pathCount = u32(params.path_count);
	var waveIntensity = 0.0;
	var waveT = 1.0; // 0 = source, 1 = target

	for (var s = 0u; s < pathCount; s = s + 1u) {
		let step = path[s];
		let srcIdxS = step.x;
		let tgtIdxS = step.y;
		let bf = f32(step.z);

		// Check if this path step uses the same source→target.
		if (srcIdxS == srcIdx && tgtIdxS == tgtIdx) {
			let frame = params.frame;
			// Wavefront: sharp pulse traveling from source to target.
			let attack = smoothstep(bf - 10.0, bf + 2.0, frame);
			let decay = 1.0 - smoothstep(bf + 30.0, bf + 180.0, frame);
			waveIntensity = max(waveIntensity, attack * decay);

			// Wave position along edge (0 = source, 1 = target).
			let arrival = bf - 10.0;
			let end = bf + 30.0;
			if (frame >= arrival && frame <= end) {
				waveT = (frame - arrival) / (end - arrival);
			} else if (frame > end) {
				waveT = 1.0;
			}
		}
	}

	// Edge base color: blend of source and target node base colors.
	let srcColor = src.color_flags.rgb;
	let tgtColor = tgt.color_flags.rgb;
	let baseColor = mix(srcColor, tgtColor, 0.5);

	// Wavefront color: thin-film spectral band, modulated by wave position.
	let waveColor = spectral(waveT + params.loop_phase);

	// Combine: base edge (dim) + wavefront pulse (bright, additive).
	let edgeAlpha = 0.08 * params.brightness; // dim connecting line
	let waveAlpha = waveIntensity * 0.9 * params.brightness; // bright pulse

	// Spectral hue rides the wavefront.
	// FOSSIL LIGHT existence mask — an edge only exists while BOTH endpoints
	// do. Live retention of exactly 0 is the "not yet born at the scrubbed
	// instant" sentinel (fsrs.ts floors living memories at 0.001), so edges
	// vanish with their memories when the chrono rewinds across a birthday.
	let exists = step(0.0005, src.vel_retention.w) * step(0.0005, tgt.vel_retention.w);
	out.color = (baseColor * edgeAlpha + waveColor * waveAlpha) * exists;

	// Line width: thicker at the wavefront for visibility.
	out.width = 1.0 + waveIntensity * 3.0;

	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	// Soft edge: feather the line edges.
	let alpha = smoothstep(0.0, 0.5, in.width) * 0.6;
	// Additive: alpha is ignored, light accumulates.
	return vec4<f32>(in.color, 1.0);
}
`;function fn(i){return 60+i*60}function Ir(i,e,t=8,r={}){var d;const n=[...i.nodes].sort((f,h)=>f.id<h.id?-1:f.id>h.id?1:0),a=[...i.edges].sort((f,h)=>{const p=`${f.source}\0${f.target}\0${f.type}`,v=`${h.source}\0${h.target}\0${h.type}`;return p<v?-1:p>v?1:0}),s=r.centerId??i.center_id,o=Gi(n,a,s,t,{preferCausal:r.preferCausal}),c=[];for(let f=0;f<o.beats.length;f++){const h=o.beats[f],p=e.indexById.get(h.nodeId);if(p===void 0)continue;const v=f>0?o.beats[f-1].nodeId:h.nodeId,b=e.indexById.get(v)??p,x=(((d=h.viaEdge)==null?void 0:d.type)??"").toLowerCase(),S=x==="causal"||x.includes("causal"),y=h.kind==="contradiction"||S;c.push({sourceIndex:b,targetIndex:p,beatFrame:fn(f),kind:y?de.backwardCause:de.recall,beatKind:h.kind,nodeId:h.nodeId,label:h.node.label})}const l=new Uint32Array(Math.max(1,c.length)*Ie);return c.forEach((f,h)=>{l[h*Ie]=f.sourceIndex,l[h*Ie+1]=f.targetIndex,l[h*Ie+2]=f.beatFrame,l[h*Ie+3]=f.kind}),{data:l,steps:c,path:o}}const hn=24,pn=300,Ge=128;class mn{constructor(e){u(this,"engine");u(this,"pipeline",null);u(this,"bindGroup",null);u(this,"cameraBuffer",null);u(this,"nodeBuffer",null);u(this,"edgeBuffer",null);u(this,"cameraData",new Float32Array(hn));u(this,"nodeCount",0);u(this,"simPipeline",null);u(this,"simBindGroup",null);u(this,"pathBuffer",null);u(this,"liveRetentionBuffer",null);u(this,"pickReadback",null);u(this,"disposed",!1);u(this,"edgeCapacityBytes",0);u(this,"edgeCount",0);u(this,"cameraRig",{...pt});u(this,"hoveredIndex",-1);u(this,"pathPipeline",null);u(this,"pathBindGroup",null);u(this,"pathStepCount",0);u(this,"axonPipeline",null);u(this,"axonBindGroup",null);u(this,"graph",null);u(this,"pathSteps",[]);this.engine=e,e.addPass(this)}upload(e,t,r){var p,v,b,x;const n=this.engine.gpuDevice;if(!n)return;const a=(r==null?void 0:r.recallPath)??!0,s=ln(e);this.graph=s;const o=new Dt({seed:t}),{data:c,nodeCount:l}=Br(s,o.state.rng);this.nodeCount=l,(p=this.nodeBuffer)==null||p.destroy(),this.nodeBuffer=n.createBuffer({label:"observatory-node-state",size:Math.max(c.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),n.queue.writeBuffer(this.nodeBuffer,0,c.buffer);const d=cr(s);this.edgeCount=s.edges.length,(v=this.edgeBuffer)==null||v.destroy(),this.edgeCapacityBytes=Math.max(d.byteLength*2,64),this.edgeBuffer=n.createBuffer({label:"observatory-edge-index",size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.edgeBuffer,0,d.buffer);const f=new Float32Array(Math.max(l,4));for(let S=0;S<l;S++)f[S]=Math.max(.001,s.nodes[S].retention);(b=this.liveRetentionBuffer)==null||b.destroy(),this.liveRetentionBuffer=n.createBuffer({label:"observatory-live-retention",size:f.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.liveRetentionBuffer,0,f.buffer);const h=a?Ir(e,s):{steps:[],data:new Uint32Array(4)};this.pathSteps=h.steps,(x=this.pathBuffer)==null||x.destroy(),this.pathBuffer=n.createBuffer({label:"observatory-path-steps",size:Ge*Ie*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.pathBuffer,0,h.data.buffer,0,Math.min(h.data.byteLength,Ge*Ie*4)),this.pathStepCount=Math.min(this.pathSteps.length,Ge),this.engine.params[2]=l,this.engine.params[3]=s.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||(this.cameraBuffer=n.createBuffer({label:"observatory-camera",size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.createPipeline(n)}setPathSteps(e,t){var a;const r=this.engine.gpuDevice;if(!r)return;this.pathSteps=t;const n=Ge*Ie*4;if(this.pathBuffer&&e.byteLength<=n){this.pathStepCount=Math.min(t.length,Ge),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,Ge),(a=this.pathBuffer)==null||a.destroy(),this.pathBuffer=r.createBuffer({label:"observatory-path-steps",size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,n)),this.engine.params[4]=this.pathStepCount,this.createPipeline(r)}setCameraRig(e){this.cameraRig=e}setHovered(e){this.hoveredIndex=e}currentOrbit(){const e=this.engine.params[6]||1,t=this.engine.params[7]||1,r=this.engine.params[1];return an(r,e/t,pn,this.cameraRig)}setEdges(e){var s;const t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e,this.edgeCount=e.length;const r=cr(this.graph),n=Math.max(r.byteLength,8);let a=!1;(!this.edgeBuffer||n>this.edgeCapacityBytes)&&((s=this.edgeBuffer)==null||s.destroy(),this.edgeCapacityBytes=Math.max(n*2,64),this.edgeBuffer=t.createBuffer({label:"observatory-edge-index",size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),a=!0),t.queue.writeBuffer(this.edgeBuffer,0,r.buffer),this.engine.params[3]=e.length,a&&this.createPipeline(t)}uploadLiveRetention(e){const t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;const r=Math.min(e.length,this.nodeCount);r<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,r*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){const r=e.createShaderModule({label:"observatory-simulate",code:dn});this.simPipeline=e.createComputePipeline({label:"observatory-recall-sim",layout:"auto",compute:{module:r,entryPoint:"recall_sim"}});const n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&n.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&n.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:"observatory-recall-sim-bind",layout:this.simPipeline.getBindGroupLayout(0),entries:n})}const t=e.createShaderModule({label:"observatory-render-nodes",code:cn});if(this.pipeline=e.createRenderPipeline({label:"observatory-nodes",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=e.createBindGroup({label:"observatory-nodes-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){const r=e.createShaderModule({label:"observatory-render-path",code:Sr});this.pathPipeline=e.createRenderPipeline({label:"observatory-path",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.pathBindGroup=e.createBindGroup({label:"observatory-path-bind",layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}if(this.edgeBuffer&&this.pathBuffer&&this.nodeBuffer){const r=e.createShaderModule({label:"observatory-render-axons",code:un});this.axonPipeline=e.createRenderPipeline({label:"observatory-axons",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"line-list"}}),this.axonBindGroup=e.createBindGroup({label:"observatory-axons-bind",layout:this.axonPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.edgeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}},{binding:4,resource:{buffer:this.nodeBuffer}}]})}}compute(e){const t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;const r=this.currentOrbit();if(this.cameraData.set(r.viewProj,0),this.cameraData[16]=r.right[0],this.cameraData[17]=r.right[1],this.cameraData[18]=r.right[2],this.cameraData[19]=0,this.cameraData[20]=r.up[0],this.cameraData[21]=r.up[1],this.cameraData[22]=r.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){const n=e.beginComputePass({label:"observatory-recall-sim"});n.setPipeline(this.simPipeline),n.setBindGroup(0,this.simBindGroup),n.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),n.end()}}render(e){this.axonPipeline&&this.axonBindGroup&&this.edgeCount>0&&(e.setPipeline(this.axonPipeline),e.setBindGroup(0,this.axonBindGroup),e.draw(2,this.edgeCount)),!(!this.pipeline||!this.bindGroup||this.nodeCount===0)&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){if(this.disposed)return null;const r=this.engine.gpuDevice;if(!r||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;this.pickReadback||(this.pickReadback=this.readNodePositions(r).finally(()=>{this.pickReadback=null}));const n=await this.pickReadback;if(!n||this.disposed||!this.graph)return null;const a=this.currentOrbit().viewProj,s=1/Math.tan(50*Math.PI/360);let o=-1,c=1/0;for(let l=0;l<this.nodeCount;l++){const d=l*re+_e.posRadius,f=n[d],h=n[d+1],p=n[d+2],v=n[d+3],b=a[3]*f+a[7]*h+a[11]*p+a[15];if(b<=0)continue;const x=(a[0]*f+a[4]*h+a[8]*p+a[12])/b,S=(a[1]*f+a[5]*h+a[9]*p+a[13])/b,y=Math.max(v*s/b,.012),D=Math.hypot(x-e,S-t)/y,U=l===this.hoveredIndex?.85:1;D<1.6*U&&D<c&&(c=D,o=l)}return o<0?null:{index:o,id:this.graph.nodes[o].id}}async readNodePositions(e){if(!this.nodeBuffer)return null;const t=this.nodeCount*re*4,r=e.createBuffer({label:"observatory-pick-staging",size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const n=e.createCommandEncoder({label:"observatory-pick-copy"});n.copyBufferToBuffer(this.nodeBuffer,0,r,0,t),e.queue.submit([n.finish()]),await r.mapAsync(GPUMapMode.READ);const a=new Float32Array(r.getMappedRange().slice(0));return r.unmap(),a}catch{return null}finally{r.destroy()}}dispose(){var e,t,r,n,a;this.disposed=!0,(e=this.nodeBuffer)==null||e.destroy(),(t=this.edgeBuffer)==null||t.destroy(),(r=this.cameraBuffer)==null||r.destroy(),(n=this.pathBuffer)==null||n.destroy(),(a=this.liveRetentionBuffer)==null||a.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null,this.axonPipeline=null,this.axonBindGroup=null,this.edgeCapacityBytes=0,this.edgeCount=0}}const lt=16,qe=4,dr=110,gn=180,vn=.7,bn=.2,yn=360,_n=18;function wn(i){if(i.edges.length>0){const e=i.centerIndex,t=i.edges.filter(r=>r.sourceIndex===e||r.targetIndex===e);if(t.length>0){let r=-1,n=-1;for(const a of t){const s=a.sourceIndex===e?a.targetIndex:a.sourceIndex,o=i.nodes[s];o&&o.retention>n&&(n=o.retention,r=s)}if(r>=0)return r}}for(let e=0;e<i.nodes.length;e++)if(e!==i.centerIndex)return e;return i.centerIndex}function xn(i,e,t=8192){const r=wn(i),a=i.nodes[r].id,s=ur(i,r),c=new Dt({seed:e+":birth:"+a}).state.rng,l=new Float32Array(t*lt),d=Math.floor(t*vn),f=Math.floor(t*bn),h=t-d-f;for(let S=0;S<d;S++){const y=S*lt,[D,U,oe]=Pr(S,d,dr+c()*(gn-dr),c);l[y+0]=s[0]+D,l[y+1]=s[1]+U,l[y+2]=s[2]+oe,l[y+3]=c(),l[y+4]=s[0],l[y+5]=s[1],l[y+6]=s[2],l[y+7]=1+c()*1.8,l[y+8]=.91,l[y+9]=1,l[y+10]=.72,l[y+11]=c(),l[y+12]=0,l[y+13]=0,l[y+14]=0,l[y+15]=0}const p=i.edges.filter(S=>S.sourceIndex===r||S.targetIndex===r);for(let S=0;S<f;S++){const y=(d+S)*lt;if(p.length===0)continue;const D=S%p.length,U=p[D],oe=U.sourceIndex===r?U.targetIndex:U.sourceIndex,T=ur(i,oe),Z=T[0]-s[0],A=T[1]-s[1],H=T[2]-s[2],ie=Math.sqrt(Z*Z+A*A+H*H)||1,ne=S/Math.max(1,f)*2+.5,ge=c()*30,le=-A*ge/ie,q=Z*ge/ie,z=0;l[y+0]=s[0]+Z/ie*ne*80+le,l[y+1]=s[1]+A/ie*ne*80+q,l[y+2]=s[2]+H/ie*ne*80+z,l[y+3]=c(),l[y+4]=s[0],l[y+5]=s[1],l[y+6]=s[2],l[y+7]=1+c()*1.8,l[y+8]=.91,l[y+9]=1,l[y+10]=.72,l[y+11]=c(),l[y+12]=0,l[y+13]=0,l[y+14]=0,l[y+15]=0}const v=300;for(let S=0;S<h;S++){const y=(d+f+S)*lt,D=c()*Math.PI*2,U=c()*120;l[y+0]=s[0]+Math.cos(D)*U,l[y+1]=s[1]+Math.sin(D)*U,l[y+2]=s[2]+v*.6+c()*40,l[y+3]=c(),l[y+4]=s[0],l[y+5]=s[1],l[y+6]=s[2],l[y+7]=1+c()*1.8,l[y+8]=.91,l[y+9]=1,l[y+10]=.72,l[y+11]=c(),l[y+12]=0,l[y+13]=0,l[y+14]=0,l[y+15]=0}const b=Pn(i,r),x=Bn();return{targetIndex:r,targetNodeId:a,particles:l,edgeSteps:b,timeline:x}}function ur(i,e){const t=i.nodes[e],r=i.nodes.length;if(t.isCenter&&i.centerIndex===e)return[0,0,0];const n=Math.PI*(3-Math.sqrt(5)),a=1-e/(r-1||1)*2,s=Math.sqrt(1-a*a),o=n*e,c=120,l=(e*7+3)%100/100*.1*c-.05*c,d=(e*13+7)%100/100*.1*c-.05*c,f=(e*17+11)%100/100*.1*c-.05*c;return[Math.cos(o)*s*c+l,a*c+d,Math.sin(o)*s*c+f]}function Pn(i,e){const t=i.edges.filter(a=>a.sourceIndex===e||a.targetIndex===e),r=t.length;if(r===0)return new Uint32Array(0);const n=new Uint32Array(r*qe);for(let a=0;a<r;a++){const s=t[a],o=s.sourceIndex===e?s.targetIndex:s.sourceIndex,c=yn+a*_n;n[a*qe+0]=e,n[a*qe+1]=o,n[a*qe+2]=c,n[a*qe+3]=0}return n}function Bn(){return[{label:"latent trace condensing",startFrame:60,endFrame:239},{label:"engram coalescence",startFrame:240,endFrame:329},{label:"memory ignition",startFrame:330,endFrame:359},{label:"associations engrave",startFrame:360,endFrame:509},{label:"stabilization",startFrame:510,endFrame:659}]}const Sn=`
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
`,In=16,kn=6,Rn=330,Cn=359,En=360;class Mn{constructor(e){u(this,"engine");u(this,"nodeRenderer");u(this,"active");u(this,"computePipeline",null);u(this,"computeBindGroup",null);u(this,"particleBuffer",null);u(this,"particleCount",0);u(this,"renderPipeline",null);u(this,"renderBindGroup",null);u(this,"haloPipeline",null);u(this,"haloBindGroup",null);u(this,"haloIndexBuffer",null);u(this,"engravePipeline",null);u(this,"engraveBindGroup",null);u(this,"engraveBuffer",null);u(this,"engraveStepCount",0);u(this,"timeline",[]);u(this,"birthPlan",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}get engraveSteps(){var e;return((e=this.birthPlan)==null?void 0:e.edgeSteps)??new Uint32Array(0)}upload(e){var a,s;const t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;const r=this.nodeRenderer.graph;if(!r)return;this.birthPlan=xn(r,e),this.timeline=this.birthPlan.timeline;const n=this.birthPlan.particles.length/In;this.particleCount=n,(a=this.particleBuffer)==null||a.destroy(),this.particleBuffer=t.createBuffer({label:"observatory-birth-particles",size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),(s=this.engraveBuffer)==null||s.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:"observatory-birth-engrave",size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){const t=e.createShaderModule({label:"observatory-birth-compute",code:Sn});this.computePipeline=e.createComputePipeline({label:"observatory-birth-compute-pipeline",layout:"auto",compute:{module:t,entryPoint:"birth_compute"}});const r=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:"observatory-birth-compute-bind",layout:this.computePipeline.getBindGroupLayout(0),entries:r})}createRenderPipeline(e){const r=e.createShaderModule({label:"observatory-birth-render",code:`
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

	// Color: luciferin dust (doctrine ignition — never purple).
	let phase = particle.color_phase.w;
	let spectralW = fract(params.loop_phase + phase);
	var spectralColor: vec3<f32>;
	var stops = array<vec3<f32>, 4>(
		vec3<f32>(0.91, 1.00, 0.72), // luciferin
		vec3<f32>(0.16, 0.95, 0.66), // recall jade
		vec3<f32>(0.13, 0.84, 1.00), // bridge cyan
		vec3<f32>(0.91, 1.00, 0.72)  // wrap
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
`});this.renderPipeline=e.createRenderPipeline({label:"observatory-birth-render",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});const n=this.nodeRenderer.cameraUniformBuffer;this.renderBindGroup=e.createBindGroup({label:"observatory-birth-render-bind",layout:this.renderPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.particleBuffer}}]})}createHaloPipeline(e){const r=e.createShaderModule({label:"observatory-birth-halo",code:`
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

	// Flash: white-hot core, luciferin rim.
	let flashIntensity = 1.0 - smoothstep(0.0, 0.7, d);
	let color = vec3<f32>(0.91, 1.00, 0.72) * flashIntensity * 2.0;

	// Fade out as flash ends.
	let frame = params.frame;
	let fadeOut = 1.0 - smoothstep(345.0, 359.0, frame);

	return vec4<f32>(color * params.brightness * fadeOut, 1.0);
}
`});this.haloPipeline=e.createRenderPipeline({label:"observatory-birth-halo",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});const n=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:"observatory-birth-halo-bind",layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;const t=e.createShaderModule({label:"observatory-birth-engrave",code:Sr});this.engravePipeline=e.createRenderPipeline({label:"observatory-birth-engrave-pipeline",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.engraveBindGroup=e.createBindGroup({label:"observatory-birth-engrave-bind",layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){const r=this.engine.params[9];if(this.active=r===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;const n=e.beginComputePass({label:"observatory-birth-compute"});n.setPipeline(this.computePipeline),n.setBindGroup(0,this.computeBindGroup),n.dispatchWorkgroups(Math.ceil(this.particleCount/64)),n.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(kn,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=Rn&&t<=Cn&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=En&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){var e,t,r,n,a,s,o,c,l,d,f;(e=this.particleBuffer)==null||e.destroy(),this.particleBuffer=null,(r=(t=this.computePipeline)==null?void 0:t.destroy)==null||r.call(t),this.computePipeline=null,this.computeBindGroup=null,(a=(n=this.renderPipeline)==null?void 0:n.destroy)==null||a.call(n),this.renderPipeline=null,this.renderBindGroup=null,(o=(s=this.haloPipeline)==null?void 0:s.destroy)==null||o.call(s),this.haloPipeline=null,this.haloBindGroup=null,(c=this.haloIndexBuffer)==null||c.destroy(),this.haloIndexBuffer=null,(d=(l=this.engravePipeline)==null?void 0:l.destroy)==null||d.call(l),this.engravePipeline=null,this.engraveBindGroup=null,(f=this.engraveBuffer)==null||f.destroy(),this.engraveBuffer=null}}function An(i){const e=i.hopSlot.toFixed(1),t=i.causeDepth.toFixed(1);return`
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
`}const Fn=2;class Gn{constructor(e){u(this,"engine");u(this,"nodeRenderer");u(this,"plan");u(this,"pipeline",null);u(this,"bindGroup",null);u(this,"waveBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.waveBuffer)==null||r.destroy(),this.waveBuffer=e.createBuffer({label:"observatory-rescue-wave",size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);const t=e.createShaderModule({label:"observatory-rescue-choreo",code:An(this.plan.consts)});this.pipeline=e.createComputePipeline({label:"observatory-rescue-choreo",layout:"auto",compute:{module:t,entryPoint:"rescue_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-rescue-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==Fn||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-rescue-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.waveBuffer)==null||e.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}}const Ln=4,kr=90,Dn=138,Tn=28,ht=260,zn=514,fr=560,Rr=600,ke=65535,Un=48,On={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function Cr(i,e){const t=new Dt({seed:e});return Br(i,t.state.rng).data}function Nn(i){const e=new Uint32Array(i.nodes.length);for(const t of i.edges)e[t.sourceIndex]++,e[t.targetIndex]++;return e}function jn(i,e){const t=i.nodes.length;if(t===0)return-1;const r=Nn(i),n=s=>{const o=i.nodes[s],c=new Set(o.tags.map(p=>p.toLowerCase()));let l=0;(c.has("failure")||c.has("guardrail"))&&(l+=3),(c.has("confusion")||c.has("weak-spot"))&&(l+=2),l+=Math.min(r[s],8)/8;const d=e[s*re+0],f=e[s*re+1],h=e[s*re+2];return Math.sqrt(d*d+f*f+h*h)>=54&&(l+=.5),l},a=[s=>s!==i.centerIndex&&!i.nodes[s].suppressed&&r[s]>=2,s=>s!==i.centerIndex&&!i.nodes[s].suppressed,s=>s!==i.centerIndex,()=>!0];for(const s of a){let o=-1,c=-1/0;for(let l=0;l<t;l++){if(!s(l))continue;const d=n(l);d>c&&(c=d,o=l)}if(o>=0)return o}return-1}function qn(i,e){const t=i.nodes.length,r=new Uint16Array(t).fill(ke),n=new Int32Array(t).fill(-1);if(e<0||e>=t)return{depths:r,parents:n};const a=Array.from({length:t},()=>[]);for(const o of i.edges){const c=On[o.type]??5;a[o.sourceIndex].push({nbr:o.targetIndex,rank:c}),a[o.targetIndex].push({nbr:o.sourceIndex,rank:c})}for(const o of a)o.sort((c,l)=>c.rank-l.rank||c.nbr-l.nbr);r[e]=0;const s=[e];for(let o=0;o<s.length;o++){const c=s[o];for(const{nbr:l}of a[c])r[l]===ke&&(r[l]=r[c]+1,s.push(l))}for(let o=0;o<t;o++)if(!(r[o]===ke||r[o]===0)){for(const{nbr:c}of a[o])if(r[c]===r[o]-1){n[o]=c;break}}return{depths:r,parents:n}}function Vn(i,e,t,r){const n=new Map;for(const a of i.nodes)n.set(a.id,a.createdAt);for(const a of[3,2,1]){const s=[];for(let p=0;p<e.nodes.length;p++){if(p===e.centerIndex||p===r)continue;const v=t[p];v===ke||v<a||s.push(p)}if(s.length===0)continue;let o=s.filter(p=>e.nodes[p].retention<=.45);o.length===0&&(o=s);const c=new Map;let l=1/0,d=-1/0;for(const p of o){const v=n.get(e.nodes[p].id),b=v?Date.parse(v):NaN;Number.isFinite(b)&&(c.set(p,b),b<l&&(l=b),b>d&&(d=b))}const f=p=>{const v=c.get(p);return v===void 0?0:d===l?1:(d-v)/(d-l)},h=p=>2*(1-e.nodes[p].retention)+.5*Math.min(t[p],6)/6+.5*f(p);return o.sort((p,v)=>{const b=h(p),x=h(v);return x!==b?x-b:t[v]!==t[p]?t[v]-t[p]:p-v}),{index:o[0],depth:t[o[0]]}}return{index:-1,depth:0}}function Wn(i,e,t,r,n){const a=i[t*re+0],s=i[t*re+1],o=i[t*re+2],c=[];for(let l=0;l<e;l++){if(l===t||l===r||l===n)continue;const d=i[l*re+0]-a,f=i[l*re+1]-s,h=i[l*re+2]-o;c.push({i:l,d2:d*d+f*f+h*h})}return c.sort((l,d)=>l.d2-d.d2||l.i-d.i),c.slice(0,Ln).map(l=>l.i)}function mt(i){const e=Math.max(1,i);return Math.min(84,Math.max(14,Math.floor(252/e)))}function Yn(i,e){return Math.min(ht+e*i,zn)}function hr(i){return Dn+Tn*i}function fe(i){return i.length>64?i.slice(0,64)+"…":i}const we=4;function Me(i){const e=new Uint32Array(i);return e.fill(ke),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(i).fill(ke),causeDepth:0,hopSlot:mt(3),waveData:e,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"candidate cause found",causeLabel:"",failureLabel:"",causeDate:"",hops:0,k:0,receipt:""},consts:{hopSlot:mt(3),causeDepth:3}}}function Hn(i,e,t,r){var z;if(r)return Kn(i,e,r);const n=e.nodes.length;if(n===0)return Me(0);const a=Cr(e,t),s=jn(e,a);if(s<0)return Me(n);const{depths:o,parents:c}=qn(e,s),l=Vn(i,e,o,s);if(l.index<0){const w=Me(n);return w.failureIndex=s,w.hopDepths=o,w}const d=l.index,f=Math.max(1,l.depth),h=mt(f),p=w=>Yn(w,h),v=Wn(a,n,s,d,e.centerIndex),b=v.length,x=new Uint32Array(n);for(let w=0;w<n;w++){let F=o[w]&65535;w===s&&(F|=65536),w===d&&(F|=1<<17),x[w]=F}v.forEach((w,F)=>{x[w]|=1<<18|F<<19});const S=[];v.forEach((w,F)=>{S.push({src:s,dst:w,bf:hr(F),kind:de.probe,beatKind:"probe"})});const y=[];{let w=d;for(;w!==s&&w>=0&&c[w]>=0;)y.push(w),w=c[w]}const D=new Set(y),U=[];for(let w=0;w<n;w++){if(w===s||D.has(w))continue;const F=o[w];F===ke||F<1||F>f||c[w]<0||U.push(w)}U.sort((w,F)=>o[w]-o[F]||w-F);const oe=[...y.slice().reverse(),...U].slice(0,Un);oe.sort((w,F)=>o[w]-o[F]||w-F);for(const w of oe)S.push({src:c[w],dst:w,bf:p(o[w]),kind:de.backwardCause,beatKind:"wave"});S.push({src:d,dst:s,bf:fr,kind:de.backwardCause,beatKind:"arc"});const T=new Uint32Array(Math.max(1,S.length)*we),Z=[];S.forEach((w,F)=>{T[F*we+0]=w.src,T[F*we+1]=w.dst,T[F*we+2]=w.bf,T[F*we+3]=w.kind,Z.push({sourceIndex:w.src,targetIndex:w.dst,beatFrame:w.bf,kind:w.kind,beatKind:w.beatKind,nodeId:e.nodes[w.dst].id,label:fe(e.nodes[w.dst].label)})});const A=fe(e.nodes[s].label),H=fe(e.nodes[d].label),ie=[],ne=(w,F,Ke,Xe)=>{ie.push({sourceIndex:s,targetIndex:s,beatFrame:w,kind:F,beatKind:"rescue",nodeId:Xe,label:Ke})};ne(kr,1,`failure: ${A}`,e.nodes[s].id),v.forEach((w,F)=>{ne(hr(F),0,`lookalike ✗ · ${fe(e.nodes[w].label)}`,e.nodes[w].id)}),ne(p(1),1,"reaching backward through time","rescue-wave-start"),f>=2&&p(f)!==p(1)&&ne(p(f),1,`scrubbing past · ${f} hops`,"rescue-wave-deep"),ne(fr,1,`causal arc · ${H}`,e.nodes[d].id),ne(Rr,1,"candidate cause found","rescue-verdict");const ge=((z=i.nodes.find(w=>w.id===e.nodes[d].id))==null?void 0:z.createdAt)??"",le=ge?ge.slice(0,10):"",q={headline:"candidate cause found",causeLabel:H,failureLabel:A,causeDate:le,hops:f,k:b,receipt:`${f} hops back · ${le} · heuristic, no receipt · vector search: 0 for ${b}`};return{viable:!0,failureIndex:s,causeIndex:d,lookalikeIndices:v,hopDepths:o,causeDepth:f,hopSlot:h,waveData:x,pathData:T,pathMetas:Z,spineBeats:ie,verdict:q,consts:{hopSlot:h,causeDepth:f}}}function Kn(i,e,t){var T,Z;const r=e.nodes.length,n=e.indexById.get(t.failureId)??-1,a=t.pathIds??[];if(n<0||a.length<2||a[a.length-1]!==t.failureId||new Set(a).size!==a.length)return Me(r);const s=a.map(A=>e.indexById.get(A));if(s.some(A=>A===void 0))return Me(r);const o=s,c=o[0];if(c===n)return Me(r);const l=t.candidates.find(A=>A.memoryId===a[0]);if(!l)return Me(r);const d=new Uint16Array(r);d.fill(ke),d[n]=0,o.forEach((A,H)=>{d[A]=o.length-1-H});const f=new Uint32Array(r);f[n]=65536,o.slice(0,-1).forEach(A=>{f[A]=d[A]}),f[c]|=1<<17;const h=o.length-1,p=mt(h),v=fe(e.nodes[c].label),b=fe(e.nodes[n].label),x=((Z=(T=i.nodes.find(A=>A.id===l.memoryId))==null?void 0:T.createdAt)==null?void 0:Z.slice(0,10))??"",S=new Uint32Array((o.length-1)*we),y=o.slice(0,-1).map((A,H)=>{const ie=o[H+1],ne=ht+H*p;return S[H*we]=A,S[H*we+1]=ie,S[H*we+2]=ne,S[H*we+3]=de.backwardCause,{sourceIndex:A,targetIndex:ie,beatFrame:ne,kind:de.backwardCause,beatKind:"receipt-path",nodeId:a[H+1],label:`recorded path · ${fe(e.nodes[ie].label)}`}}),D=l.sharedEntities.length?l.sharedEntities.join(", "):"recorded entity",U=l.similarityRank===null?"rank unavailable":`embedding rank #${l.similarityRank}`,oe=[{sourceIndex:n,targetIndex:n,beatFrame:kr,kind:1,beatKind:"receipt-failure",nodeId:t.failureId,label:`recorded failure · ${b}`},{sourceIndex:n,targetIndex:n,beatFrame:ht,kind:1,beatKind:"receipt-join",nodeId:"receipt-join",label:`shared entity · ${D}`},{sourceIndex:c,targetIndex:n,beatFrame:ht+(h-1)*p,kind:de.backwardCause,beatKind:"receipt-candidate",nodeId:l.memoryId,label:`candidate · ${v}`},{sourceIndex:c,targetIndex:c,beatFrame:Rr,kind:1,beatKind:"receipt-verdict",nodeId:"receipt-verdict",label:"candidate cause found"}];return{viable:!0,failureIndex:n,causeIndex:c,lookalikeIndices:[],hopDepths:d,causeDepth:h,hopSlot:p,waveData:f,pathData:S,pathMetas:y,spineBeats:oe,verdict:{headline:"candidate cause found",causeLabel:v,failureLabel:b,causeDate:x,hops:h,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${D} · ${U}`},consts:{hopSlot:p,causeDepth:h}}}const Xn=`
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
`,Qn=3;class Zn{constructor(e){u(this,"engine");u(this,"nodeRenderer");u(this,"plan");u(this,"pipeline",null);u(this,"bindGroup",null);u(this,"horizonBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.horizonBuffer)==null||r.destroy(),this.horizonBuffer=e.createBuffer({label:"observatory-forgetting-horizon",size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);const t=e.createShaderModule({label:"observatory-forgetting-choreo",code:Xn});this.pipeline=e.createComputePipeline({label:"observatory-forgetting-choreo",layout:"auto",compute:{module:t,entryPoint:"forgetting_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-forgetting-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==Qn||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-forgetting-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.horizonBuffer)==null||e.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}}const Jn=318,$n=60,Er=3,ea=132,ta=60,ra=540,ia=660;function na(i){const e=[];for(let n=0;n<i.nodes.length;n++)n!==i.centerIndex&&e.push(n);e.sort((n,a)=>i.nodes[n].retention-i.nodes[a].retention||n-a);const t=e.length;if(t===0)return[];const r=Math.min(t,Math.max(Math.min(Er,t),Math.round(.25*t)));return e.slice(0,r)}function aa(i,e){const t=new Uint32Array(i.nodes.length);for(const a of i.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>2*i.nodes[a].retention+Math.min(t[a],8)/8;return e.slice().sort((a,s)=>r(s)-r(a)||a-s).slice(0,Math.min(Er,e.length))}function pr(i){return Jn+$n*i}const Ve=4;function sa(i){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(i),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function oa(i){const e=i.nodes.length,t=na(i);if(e<2||t.length<1)return sa(e);const r=aa(i,t),n=t.length,a=new Uint32Array(e);t.forEach((h,p)=>{const v=Math.round(255*p/Math.max(1,n-1));a[h]=v&255|256}),r.forEach((h,p)=>{a[h]|=512|p<<10});const s=new Uint32Array(Math.max(1,r.length)*Ve),o=[];r.forEach((h,p)=>{const v=pr(p);s[p*Ve+0]=i.centerIndex,s[p*Ve+1]=h,s[p*Ve+2]=v,s[p*Ve+3]=de.recall,o.push({sourceIndex:i.centerIndex,targetIndex:h,beatFrame:v,kind:de.recall,beatKind:"recall",nodeId:i.nodes[h].id,label:fe(i.nodes[h].label)})});const c=[],l=(h,p,v,b)=>{c.push({sourceIndex:i.centerIndex,targetIndex:i.centerIndex,beatFrame:h,kind:p,beatKind:"horizon",nodeId:b,label:v})},d=new Set(r),f=t.filter(h=>!d.has(h)).slice(0,3);return f.forEach((h,p)=>{const v=Math.round(i.nodes[h].retention*100);l(ea+ta*p,1,`fading: ${fe(i.nodes[h].label)} · retention ${v}%`,i.nodes[h].id)}),r.forEach((h,p)=>{l(pr(p),0,`recalled: ${fe(i.nodes[h].label)}`,i.nodes[h].id)}),f.length>0&&l(ra,1,"the unrecalled sink · nothing is deleted","horizon-sink"),l(ia,0,"every memory still retrievable","horizon-retrievable"),{viable:!0,driftingIndices:t,rescuedIndices:r,horizonData:a,pathData:s,pathMetas:o,spineBeats:c}}const la=`
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
`,ca=4;class Mr{constructor(e){u(this,"engine");u(this,"nodeRenderer");u(this,"plan");u(this,"pipeline",null);u(this,"bindGroup",null);u(this,"fireBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.fireBuffer)==null||r.destroy(),this.fireBuffer=e.createBuffer({label:"observatory-firewall-fire",size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);const t=e.createShaderModule({label:"observatory-firewall-choreo",code:la});this.pipeline=e.createComputePipeline({label:"observatory-firewall-choreo",layout:"auto",compute:{module:t,entryPoint:"firewall_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-firewall-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,!!this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){const t=this.engine.params[9]===ca,r=this.engine.params[12]===1;if(!t&&!r||!this.pipeline||!this.bindGroup)return;const n=this.nodeRenderer.nodeCountValue;if(n===0)return;const a=e.beginComputePass({label:"observatory-firewall-choreo"});a.setPipeline(this.pipeline),a.setBindGroup(0,this.bindGroup),a.dispatchWorkgroups(Math.ceil(n/64)),a.end()}dispose(){var e;(e=this.fireBuffer)==null||e.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}}const da=90,ua=150,fa=144,ha=330,pa=345,ma=21,ga=6,va=480,ba=["failure","guardrail","confusion"];function ya(i){const e=i.nodes.length;if(e===0)return-1;const t=new Uint32Array(e);for(const a of i.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>i.nodes[a].tags.some(s=>ba.includes(s.toLowerCase())),n=[a=>a!==i.centerIndex&&!i.nodes[a].suppressed&&r(a),a=>a!==i.centerIndex&&!i.nodes[a].suppressed&&t[a]<=1,a=>a!==i.centerIndex&&!i.nodes[a].suppressed,a=>a!==i.centerIndex];for(const a of n){let s=-1;for(let o=0;o<e;o++)a(o)&&(s<0||i.nodes[o].retention<i.nodes[s].retention)&&(s=o);if(s>=0)return s}return-1}function _a(i,e,t){const r=i[t*re+0],n=i[t*re+1],a=i[t*re+2],s=new Array(e);let o=0;for(let l=0;l<e;l++){const d=i[l*re+0]-r,f=i[l*re+1]-n,h=i[l*re+2]-a,p=Math.sqrt(d*d+f*f+h*h);s[l]=p,p>o&&(o=p)}o<1e-6&&(o=1);const c=new Array(e);for(let l=0;l<e;l++)c[l]=Math.min(255,Math.max(0,Math.round(fa*s[l]/o)));return c[t]=0,c}function wa(i,e){const t=new Set;for(const r of i.edges)r.sourceIndex===e&&r.targetIndex!==e&&t.add(r.targetIndex),r.targetIndex===e&&r.sourceIndex!==e&&t.add(r.sourceIndex);return Array.from(t).sort((r,n)=>r-n).slice(0,ga)}function mr(i){return pa+ma*i}const We=4;function xa(i){return Tt(i)}function Tt(i){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(i),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"threat quarantined",intruderLabel:"",receipt:"memory held in review · Memory PR opened"}}}function Pa(i,e){return Ar(i,e,ya(i))}function Ba(i,e,t){return t<0||t>=i.nodes.length?Tt(i.nodes.length):Ar(i,e,t)}function Ar(i,e,t){const r=i.nodes.length;if(r===0||t<0)return Tt(r);const n=Cr(i,e),a=_a(n,r,t),s=wa(i,t),o=new Uint32Array(r);for(let v=0;v<r;v++)o[v]=a[v]&255;o[t]=256,s.forEach((v,b)=>{o[v]|=512|b<<10});const c=new Uint32Array(Math.max(1,s.length)*We),l=[];s.forEach((v,b)=>{const x=mr(b);c[b*We+0]=t,c[b*We+1]=v,c[b*We+2]=x,c[b*We+3]=de.probe,l.push({sourceIndex:t,targetIndex:v,beatFrame:x,kind:de.probe,beatKind:"sever",nodeId:i.nodes[v].id,label:fe(i.nodes[v].label)})});const d=fe(i.nodes[t].label),f=[],h=(v,b,x)=>{f.push({sourceIndex:t,targetIndex:t,beatFrame:v,kind:1,beatKind:"firewall",nodeId:x,label:b})};return h(da,`intrusion · ${d}`,i.nodes[t].id),h(ua,"immune response · shockwave","firewall-shock"),h(ha,"membrane forming","firewall-membrane"),s.forEach((v,b)=>{h(mr(b),`edge severed ✗ · ${fe(i.nodes[v].label)}`,i.nodes[v].id)}),h(va,"threat quarantined","firewall-verdict"),{viable:!0,intruderIndex:t,severedNeighborIndices:s,shockDelays:a,fireData:o,pathData:c,pathMetas:l,spineBeats:f,verdict:{headline:"threat quarantined",intruderLabel:d,receipt:"memory held in review · Memory PR opened"}}}const bt=.1542;function Sa(i=bt){return Math.pow(.9,-1/i)-1}function Fr(i,e,t=bt){if(!(i>0))return 0;if(!(e>0))return 1;const r=Sa(t),n=Math.pow(1+r*e/i,-t);return n<0?0:n>1?1:n}const zt=864e5;function Ia(i,e,t=0){if(!i)return t>0?t:0;const r=Date.parse(i);if(!Number.isFinite(r))return t>0?t:0;const n=(e-r)/zt;return Math.max(0,n)+Math.max(0,t)}function ka(i,e,t,r,n=bt){if(t){const s=Date.parse(t);if(Number.isFinite(s)&&r<s)return 0}if(i===void 0||!Number.isFinite(i)||!e)return 1;const a=Date.parse(e);return Number.isFinite(a)?Math.max(.001,Fr(i,(r-a)/zt,n)):1}function Ra(i,e,t,r=0,n=bt){return i===void 0||!Number.isFinite(i)?1:Fr(i,Ia(e,t,r),n)}const gr={[te.firewall]:620,[te.dreamStorm]:360,[te.causalRecall]:260,[te.birth]:180};class Ca{constructor(e){u(this,"engine");u(this,"renderer");u(this,"graph");u(this,"response");u(this,"seed");u(this,"projectionDays");u(this,"chronoOffsetDays");u(this,"onApply");u(this,"onFirewall");u(this,"firewall",null);u(this,"liveEdges",[]);u(this,"liveEdgeKeys",new Set);u(this,"edgesDirty",!1);u(this,"indexById");u(this,"active",null);u(this,"dreamOpen",!1);u(this,"retention");u(this,"hasLiveDecay",!1);u(this,"eventsSeen",0);u(this,"lastDecayFrame",-1e3);u(this,"lastAppliedMs",0);u(this,"seeded",!1);u(this,"lastProj",-1);u(this,"lastChrono",0);this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;const t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let n=0;n<t;n++){const a=e.graph.nodes[n];this.retention[n]=a.retention,a.stability!==void 0&&a.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(const n of this.liveEdges)this.liveEdgeKeys.add(vr(n.sourceIndex,n.targetIndex));this.lastAppliedMs=0;const r=this.engine.params;r[ae.liveKind]=te.none,r[ae.liveFrame]=0,r[ae.liveEnergy]=0,r[ae.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}seedWatermark(e){let t=0;for(const r of e){const n=br(r);n>t&&(t=n)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,r){if(this.active!==null)return!1;const n=this.indexById.get(e);if(n===void 0||(this.retention[n]??0)<5e-4)return!1;const a=t.filter(s=>s!==e&&this.indexById.has(s));return this.arm({kind:te.causalRecall,startFrame:r,targetId:e,relatedIds:a,pairs:[],scalar:a.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let r=e.length-1;r>=0;r--){const n=e[r],a=br(n);a>this.lastAppliedMs&&(this.decodeAndArm(n,this.engine.totalFrames),a>t&&(t=a))}this.lastAppliedMs=t}decodeAndArm(e,t){var n;const r=e.data??{};switch(e.type){case"MemorySuppressed":{const a=Le(r.id);if(!a||!this.indexById.has(a))return;this.arm({kind:te.firewall,startFrame:t,targetId:a,relatedIds:this.neighborsOf(a),pairs:[],scalar:Ye(r.estimated_cascade)});break}case"DeepReferenceCompleted":{const s=Ea(r.contradiction_pairs).filter(([l,d])=>this.indexById.has(l)&&this.indexById.has(d));if(s.length>0){const l=s[0][0];this.arm({kind:te.firewall,startFrame:t,targetId:l,relatedIds:s.flatMap(d=>d).filter(d=>d!==l),pairs:s,scalar:s.length});return}const o=Le(r.primary_id),c=yr(r.supporting_ids).filter(l=>this.indexById.has(l));o&&this.indexById.has(o)&&this.arm({kind:te.causalRecall,startFrame:t,targetId:o,relatedIds:c,pairs:[],scalar:Ye(r.confidence)});break}case"BackfillFired":case"CausalReceipt":{const a=yr(r.path_ids??r.causal_path),s=Le(r.failure_id??r.target_id??r.effect_id)||a.at(-1)||a[0];s&&this.indexById.has(s)&&this.arm({kind:te.causalRecall,startFrame:t,targetId:s,relatedIds:a.filter(o=>o!==s),exactPath:a,pairs:[],scalar:a.length});break}case"DreamStarted":{this.dreamOpen=!0,this.arm({kind:te.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:Ye(r.memory_count)});break}case"DreamCompleted":{this.dreamOpen=!1;const a=Ye(r.connections_found);this.active&&this.active.kind===te.dreamStorm?this.active.scalar=Math.max(this.active.scalar,a):this.arm({kind:te.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:a});break}case"ConnectionDiscovered":{const a=this.indexById.get(Le(r.source_id)),s=this.indexById.get(Le(r.target_id));if(a===void 0||s===void 0||a===s)break;const o=vr(a,s);if(this.liveEdgeKeys.has(o))break;this.liveEdgeKeys.add(o),this.liveEdges.push({sourceIndex:a,targetIndex:s,weight:Ye(r.weight)||.5,type:Le(r.connection_type)||"semantic"}),this.edgesDirty=!0,this.dreamOpen&&((n=this.active)==null?void 0:n.kind)===te.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){var t;if(this.active=e,this.eventsSeen++,e.kind===te.firewall){const r=this.indexById.get(e.targetId);if(r===void 0)return;const n=Ba(this.graph,this.seed,r);if(!n.viable)return;this.firewall||(this.firewall=new Mr({engine:this.engine,nodeRenderer:this.renderer,plan:xa(this.graph.nodes.length)})),this.firewall.rearm(n),(t=this.onFirewall)==null||t.call(this,{intruderLabel:n.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===te.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){const n=e.exactPath;if(n.some(o=>!this.indexById.has(o)))return;const a=new Uint32Array(Math.max(1,n.length-1)*4),s=[];for(let o=0;o<n.length-1;o++){const c=this.indexById.get(n[o]),l=this.indexById.get(n[o+1]),d=e.startFrame+24+o*42;a[o*4]=c,a[o*4+1]=l,a[o*4+2]=d,a[o*4+3]=de.backwardCause,s.push({sourceIndex:c,targetIndex:l,beatFrame:d,kind:de.backwardCause,beatKind:"receipt-path",nodeId:n[o+1],label:"receipt-backed candidate path"})}this.renderer.setPathSteps(a,s);return}const r=Ir(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});r.steps.length>0&&this.renderer.setPathSteps(r.data,r.steps)}}neighborsOf(e){const t=this.indexById.get(e);if(t===void 0)return[];const r=[];for(const n of this.graph.edges)if(n.sourceIndex===t?r.push(this.graph.nodes[n.targetIndex].id):n.targetIndex===t&&r.push(this.graph.nodes[n.sourceIndex].id),r.length>=12)break;return r}drain(e){var s;const t=this.engine.params;this.edgesDirty&&(this.renderer.setEdges(this.liveEdges),this.edgesDirty=!1);const r=this.projectionDays(),n=this.chronoOffsetDays();if(t[ae.projectionDays]=Math.max(0,r),(this.hasLiveDecay||n!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||r!==this.lastProj||n!==this.lastChrono)&&(this.recomputeDecay(r,n),this.lastDecayFrame=e,this.lastProj=r,this.lastChrono=n),this.active){const o=gr[this.active.kind]??300,c=e-this.active.startFrame;c>o+140?(this.active=null,t[ae.liveKind]=te.none,t[ae.liveEnergy]=0):(t[ae.liveKind]=this.active.kind,t[ae.liveFrame]=Math.max(0,c),t[ae.liveEnergy]=this.energyEnvelope(this.active,c,!1))}else t[ae.liveKind]=te.none,t[ae.liveEnergy]=0;(s=this.onApply)==null||s.call(this,{simFrame:e,activeKind:t[ae.liveKind],eventsSeen:this.eventsSeen})}debugState(){const e=this.engine.params;return{activeKind:e[ae.liveKind],liveEnergy:e[ae.liveEnergy],liveFrame:e[ae.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}energyEnvelope(e,t,r){if(t<0)return 0;const n=gr[e.kind]??300;if(e.kind===te.dreamStorm){const o=Math.min(1,t/45),c=1-Math.max(0,(t-(n-90))/90),l=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,o*Math.min(1,c)*l)}const a=Math.min(1,t/24),s=1-Math.max(0,(t-n)/140);return Math.max(0,a*Math.min(1,s))}recomputeDecay(e,t=0){const r=this.engine.wallNowMs,n=this.graph.nodes;if(t!==0){const a=r+(t+Math.max(0,e))*zt;for(let s=0;s<n.length;s++){const o=n[s];this.retention[s]=o.stability!==void 0||o.createdAt?ka(o.stability,o.lastAccessed,o.createdAt,a):Math.max(.001,o.retention)}}else for(let a=0;a<n.length;a++){const s=n[a];this.retention[a]=s.stability!==void 0&&s.lastAccessed?Ra(s.stability,s.lastAccessed,r,e):Math.max(.001,s.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){const e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}}function vr(i,e){return i<e?`${i}-${e}`:`${e}-${i}`}function br(i){var r;const e=(r=i.data)==null?void 0:r.timestamp;if(typeof e!="string")return 0;const t=Date.parse(e);return Number.isFinite(t)?t:0}function Le(i){return typeof i=="string"?i:""}function Ye(i){return typeof i=="number"&&Number.isFinite(i)?i:0}function yr(i){return Array.isArray(i)?i.filter(e=>typeof e=="string"):[]}function Ea(i){if(!Array.isArray(i))return[];const e=[];for(const t of i)Array.isArray(t)&&t.length>=2&&typeof t[0]=="string"&&typeof t[1]=="string"&&e.push([t[0],t[1]]);return e}const ct=512,Ct=4,Ma=`
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
`;function He(i){return Math.max(0,Math.min(1,Number.isFinite(i)?i:0))}function dt(i){if(!i)return null;const e=Date.parse(i);return Number.isFinite(e)?e:null}class Aa{constructor(e,t){u(this,"engine");u(this,"resources",null);u(this,"bindLayout",null);u(this,"railPipeline",null);u(this,"dwellPipeline",null);u(this,"headPipeline",null);u(this,"dwellCount",0);u(this,"minMs",0);u(this,"maxMs",0);u(this,"state",{scrub:1,days:0,density:0,active:0});this.engine=e,this.upload(t)}setTimeline(e,t=!1){const r=this.engine.wallNowMs,n=Math.max(1,this.maxMs-this.minMs);this.state.scrub=He((r+e*864e5-this.minMs)/n),this.state.days=Number.isFinite(e)?e:0,this.state.active=t?1:0,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){!this.resources||!this.railPipeline||!this.dwellPipeline||!this.headPipeline||(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){var e,t;(e=this.resources)==null||e.dwellBuffer.destroy(),(t=this.resources)==null||t.stateBuffer.destroy(),this.resources=null}upload(e){const t=e.flatMap(d=>[dt(d.createdAt),dt(d.lastAccessed)]).filter(d=>d!==null),r=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):r-864e5,this.maxMs=Math.max(r+365*864e5,this.minMs+864e5);const n=this.maxMs-this.minMs,a=[];for(const d of e){const f=dt(d.createdAt),h=dt(d.lastAccessed),p=He(d.retention);f!==null&&a.push({at:f,kind:0,retention:p,suppressed:d.suppressed?1:0}),h!==null&&h!==f&&a.push({at:h,kind:1,retention:p,suppressed:d.suppressed?1:0})}a.sort((d,f)=>d.at-f.at);const s=Math.max(1,Math.ceil(a.length/ct)),o=a.filter((d,f)=>f%s===0).slice(0,ct);this.dwellCount=o.length,this.state={scrub:He((r-this.minMs)/n),days:0,density:He(o.length/96),active:0};const c=this.engine.gpuDevice;if(!c||!this.engine.paramsBuffer||(this.ensurePipelines(c),this.ensureResources(c),!this.resources))return;const l=new Float32Array(ct*Ct);o.forEach((d,f)=>{l.set([He((d.at-this.minMs)/n),d.kind,d.retention,d.suppressed],f*Ct)}),c.queue.writeBuffer(this.resources.dwellBuffer,0,l),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"fossil-light-chrono-shuttle-wgsl",code:Ma});this.bindLayout=e.createBindGroupLayout({label:"fossil-light-chrono-shuttle-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const r=e.createPipelineLayout({label:"fossil-light-chrono-shuttle-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),n={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},a=(s,o,c)=>e.createRenderPipeline({label:s,layout:r,vertex:{module:t,entryPoint:o},fragment:{module:t,entryPoint:c,targets:[{format:this.engine.sceneFormat,blend:n}]},primitive:{topology:"triangle-list"}});this.railPipeline=a("fossil-light-chrono-rail","vs_rail","fs_rail"),this.dwellPipeline=a("fossil-light-chrono-dwells","vs_dwell","fs_dwell"),this.headPipeline=a("fossil-light-chrono-head","vs_head","fs_head")}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;const t=e.createBuffer({label:"fossil-light-chrono-dwell-events",size:ct*Ct*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r=e.createBuffer({label:"fossil-light-chrono-state",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:r,bindGroup:e.createBindGroup({label:"fossil-light-chrono-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:r}}]})}}writeState(){const e=this.engine.gpuDevice;!e||!this.resources||e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active]))}}const Gt=64,Fa=12,ye=32,Ga=4,ut=256,Et="rgba8unorm",La=96e3,_r=5,Da=`
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

const MAX_EMITTERS = ${Gt}u;

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
`;function Ta(i,e){return Number.isFinite(i)?i:e}class za{constructor(e,t,r){u(this,"engine");u(this,"renderer");u(this,"sourceIndices");u(this,"resources",null);u(this,"projectionPipeline",null);u(this,"seedPipeline",null);u(this,"transportPipeline",null);u(this,"compositePipeline",null);u(this,"projectionLayout",null);u(this,"seedLayout",null);u(this,"transportLayout",null);u(this,"compositeLayout",null);u(this,"emitterCount");u(this,"active",!1);u(this,"dirty",!0);u(this,"lastComputedFrame",-_r);u(this,"disposed",!1);u(this,"disabledReason",null);u(this,"exposure",.42);u(this,"configBytes",new ArrayBuffer(ye));u(this,"configUints",new Uint32Array(this.configBytes));u(this,"configFloats",new Float32Array(this.configBytes));this.engine=e,this.renderer=t;const n=[...new Set([...r].filter(a=>Number.isFinite(a)&&a>=0))].sort((a,s)=>a-s).slice(0,Gt);this.sourceIndices=new Uint32Array(n),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?"half-res-transport":"disabled"}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,Ta(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;const r=this.engine.gpuDevice;if(!r||!this.engine.paramsBuffer)return;const n=this.renderer.getFossilLightSources();if(!n)return;const a=this.fieldDimensions();if(a===null)return;const s=t-this.lastComputedFrame;if(!(this.active||this.dirty||s<0||s>=_r))return;try{this.ensurePipelines(r),this.ensureResources(r,a.width,a.height,n)}catch{this.disable("GPU light field unavailable on this adapter");return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(r,0,this.resources.width,this.resources.height,0);const c=Math.ceil(this.resources.width/8),l=Math.ceil(this.resources.height/8),d=e.beginComputePass({label:"fossil-light-half-res-transport"});d.setPipeline(this.projectionPipeline),d.setBindGroup(3,this.resources.projectionBindGroup,[0]),d.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),d.setPipeline(this.seedPipeline),d.setBindGroup(0,this.resources.seedBindGroup,[0]),d.dispatchWorkgroups(c,l);for(const[f,h,p]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(r,f,this.resources.width,this.resources.height,h),d.setPipeline(this.transportPipeline),d.setBindGroup(1,p,[f*ut]),d.dispatchWorkgroups(c,l);d.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason!==null||!this.resources||!this.compositePipeline||this.emitterCount===0||(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*ut]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){const e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;const r=e*.5*(t*.5),a=.5*Math.min(1,Math.sqrt(La/Math.max(1,r)));return{width:Math.max(1,Math.floor(e*a)),height:Math.max(1,Math.floor(t*a))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;const t=e.createShaderModule({label:"fossil-light-radiance-cascade-wgsl",code:Da}),r=e.createBindGroupLayout({label:"fossil-light-empty-layout",entries:[]});this.projectionLayout=e.createBindGroupLayout({label:"fossil-light-source-projection-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ye}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),this.seedLayout=e.createBindGroupLayout({label:"fossil-light-seed-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ye}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:Et}}]}),this.transportLayout=e.createBindGroupLayout({label:"fossil-light-transport-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ye}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:Et}}]}),this.compositeLayout=e.createBindGroupLayout({label:"fossil-light-composite-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ye}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]}),this.seedPipeline=e.createComputePipeline({label:"fossil-light-seed",layout:e.createPipelineLayout({label:"fossil-light-seed-pipeline-layout",bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:"cs_seed"}}),this.projectionPipeline=e.createComputePipeline({label:"fossil-light-source-projection",layout:e.createPipelineLayout({label:"fossil-light-source-projection-pipeline-layout",bindGroupLayouts:[r,r,r,this.projectionLayout]}),compute:{module:t,entryPoint:"cs_project_sources"}}),this.transportPipeline=e.createComputePipeline({label:"fossil-light-transport",layout:e.createPipelineLayout({label:"fossil-light-transport-pipeline-layout",bindGroupLayouts:[r,this.transportLayout]}),compute:{module:t,entryPoint:"cs_transport"}}),this.compositePipeline=e.createRenderPipeline({label:"fossil-light-composite",layout:e.createPipelineLayout({label:"fossil-light-composite-pipeline-layout",bindGroupLayouts:[r,r,this.compositeLayout]}),vertex:{module:t,entryPoint:"vs_composite"},fragment:{module:t,entryPoint:"fs_composite",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]}})}ensureResources(e,t,r,n){var p;if(((p=this.resources)==null?void 0:p.width)===t&&this.resources.height===r&&this.resources.nodeBuffer===n.nodeBuffer&&this.resources.cameraBuffer===n.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;const a=e.createBuffer({label:"fossil-light-projected-memory-emitters",size:Gt*Fa*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),s=e.createBuffer({label:"fossil-light-source-indices",size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(s,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);const o=e.createBuffer({label:"fossil-light-cascade-config",size:ut*Ga,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=v=>e.createTexture({label:v,size:[t,r],format:Et,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),l=c("fossil-light-field-a"),d=c("fossil-light-field-b"),f=l.createView(),h=d.createView();this.resources={width:t,height:r,emitterBuffer:a,sourceIndexBuffer:s,configBuffer:o,fieldA:l,fieldB:d,seedBindGroup:e.createBindGroup({label:"fossil-light-seed-bind-group",layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:ye}},{binding:1,resource:{buffer:a}},{binding:2,resource:f}]}),propagateABindGroup:e.createBindGroup({label:"fossil-light-transport-a-to-b",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ye}},{binding:1,resource:f},{binding:2,resource:h}]}),propagateBBindGroup:e.createBindGroup({label:"fossil-light-transport-b-to-a",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ye}},{binding:1,resource:h},{binding:2,resource:f}]}),projectionBindGroup:e.createBindGroup({label:"fossil-light-source-projection-bind-group",layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:ye}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:n.nodeBuffer}},{binding:3,resource:{buffer:n.cameraBuffer}},{binding:4,resource:{buffer:a}}]}),compositeBindGroup:e.createBindGroup({label:"fossil-light-composite-bind-group",layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:ye}},{binding:1,resource:h}]}),nodeBuffer:n.nodeBuffer,cameraBuffer:n.cameraBuffer},this.dirty=!0}writeConfig(e,t,r,n,a){this.resources&&(this.configUints[0]=r,this.configUints[1]=n,this.configUints[2]=this.emitterCount,this.configUints[3]=a,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*ut,this.configBytes))}destroyResources(){var e,t,r,n,a;(e=this.resources)==null||e.emitterBuffer.destroy(),(t=this.resources)==null||t.sourceIndexBuffer.destroy(),(r=this.resources)==null||r.configBuffer.destroy(),(n=this.resources)==null||n.fieldA.destroy(),(a=this.resources)==null||a.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}}var Ua=Q('<div class="flex items-baseline gap-2 font-mono text-[11px]"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></div>'),Oa=Q(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),Na=Q(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),ja=Q(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"> </button>`),qa=Q('<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>'),Va=Q('<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>'),Wa=Q(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),Ya=Q("<button> </button>"),Ha=Q('<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>'),Ka=Q('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>'),Xa=Q('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>'),Qa=Q('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>'),Za=Q('<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>'),Ja=Q('<div class="pointer-events-none fixed left-4 bottom-24 z-30 font-mono text-[10px] tracking-widest text-[#7ff3e6]/80"> </div>'),$a=Q('<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div> <!> <!>',1);function bs(i,e){gt(e,!0);const t=()=>Bi(Ii,"$eventFeed",r),[r,n]=Si();let a=W(e,"seed",3,"vestige-observatory-v1"),s=W(e,"freezeFrame",3,null),o=W(e,"capture",3,!1),c=W(e,"showSwitcher",3,!0),l=W(e,"embedded",3,!1),d=W(e,"chrome",3,"none"),f=W(e,"maxDpr",3,2),h=W(e,"focusIds",19,()=>[]),p=W(e,"live",3,!1),v=W(e,"graphOverride",3,null),b=j(0),x=j(!1),S=j(0),y=null;const D=Ce(()=>Math.max(0,m(b))),U=Ce(()=>Math.min(0,m(b))),oe=Ce(()=>m(b)===0?"now":m(b)>0?`+${Math.round(m(b))}d`:new Date(Date.now()+m(b)*864e5).toLocaleDateString(void 0,{month:"short",day:"numeric"}));let T=null,Z=null,A=null,H=j(!1),ie=j(!1);const ne=.835,ge=(1+.685)/2;let le=!1,q=0,z=0,w=0,F=!1;function Ke(g){var B;const _=(B=m(ve))==null?void 0:B.getBoundingClientRect();if(!_||_.width===0)return m(b);const C=(g-_.left)/_.width*2-1,O=Math.max(0,Math.min(1,C/ne*.5+.5));return m(S)+O*(365-m(S))}function Xe(g){var O;const _=(O=m(ve))==null?void 0:O.getBoundingClientRect();if(!_||_.height===0)return!1;const C=(g.clientY-_.top)/_.height;return C>ge-.075&&C<ge+.075}function yt(){q&&cancelAnimationFrame(q),q=0}function Gr(g){var _,C;!m(H)||o()||!Xe(g)||(yt(),le=!0,P(x,!0),z=0,w=performance.now(),P(b,Ke(g.clientX),!0),(C=(_=g.currentTarget).setPointerCapture)==null||C.call(_,g.pointerId),g.preventDefault())}function Lr(g){if(!le)return;const _=performance.now(),C=Ke(g.clientX),O=Math.max(1,_-w);z=z*.6+(C-m(b))/O*16*.4,w=_,P(b,C,!0)}function Dr(g){var C,O;if(!le)return;le=!1,F=!0,(O=(C=g.currentTarget).releasePointerCapture)==null||O.call(C,g.pointerId);const _=()=>{q=0,z*=.94;let B=m(b)+z;B<=m(S)&&(B=m(S),z=0),B>=365&&(B=365,z=0);const G=m(b)<0&&z>0||m(b)>0&&z<0;Math.abs(B)<1&&G&&(B=0,z=0),P(b,B,!0),Math.abs(z)>.02?q=requestAnimationFrame(_):P(x,!1)};Math.abs(z)>.05?q=requestAnimationFrame(_):P(x,!1)}function Tr(){le=!1,P(x,!1),yt()}let xe=j(!1),_t=j(!1);function zr(){if(typeof window>"u")return;const g=window.matchMedia("(prefers-reduced-motion: reduce)");g.matches&&!m(_t)&&P(xe,!0);const _=C=>{m(_t)||P(xe,C.matches,!0)};return g.addEventListener("change",_),()=>g.removeEventListener("change",_)}function Ut(){P(_t,!0),P(xe,!m(xe))}Ne(()=>{var g;(g=m(ce))==null||g.setPaused(m(xe))});let Qe=j(""),Ze=j(0),Ur=Ce(()=>m(Qe)!==""&&m(Ze)>0),ve=j(null);const pe=new sn;let wt=j(null),Ot=0,xt=j("");async function Or(g){var G;if(F){F=!1;return}if(!k||!m(ve))return;const _=m(ve).getBoundingClientRect();if(_.width===0||_.height===0)return;const C=(g.clientX-_.left)/_.width*2-1,O=-((g.clientY-_.top)/_.height*2-1),B=await k.pickAt(C,O);B&&(P(wt,{kind:"memory",id:B.id,label:"Field cell"},!0),(G=e.onpick)==null||G.call(e,B.id))}function Nr(g){Gr(g),!(le||o())&&(pe.enabled=!o(),pe.onPointerDown(g))}function jr(g){if(Lr(g),le||o())return;pe.onPointerMove(g)&&(F=!0,k==null||k.setCameraRig(pe.state));const _=performance.now();if(_-Ot<120||!k||!m(ve))return;Ot=_;const C=m(ve).getBoundingClientRect();if(C.width===0)return;const O=(g.clientX-C.left)/C.width*2-1,B=-((g.clientY-C.top)/C.height*2-1);k.pickAt(O,B).then(G=>{var V;k==null||k.setHovered((G==null?void 0:G.index)??-1),P(xt,((V=G==null?void 0:G.id)==null?void 0:V.slice(0,8))??"",!0),m(ve)&&(m(ve).style.cursor=G?"crosshair":"grab")})}function qr(g){Dr(g),pe.onPointerUp(g)}function Vr(){Tr(),pe.onPointerUp({pointerId:-1})}function Wr(g){o()||Xe(g)||pe.onWheel(g)&&(k==null||k.setCameraRig(pe.state))}const Nt={"recall-path":"RECALL","engram-birth":"BIRTH","salience-rescue":"RESCUE","forgetting-horizon":"HORIZON",firewall:"FIREWALL"};function Yr(){return k!=null&&k.graph?new Uint32Array(k.graph.nodes.map(g=>({index:g.index,id:g.id,retention:g.retention})).sort((g,_)=>_.retention-g.retention||g.id.localeCompare(_.id)).slice(0,64).map(g=>g.index).sort((g,_)=>g-_)):new Uint32Array}let Pt=j(!o());function Hr(g){const _=g.target;_!=null&&_.isContentEditable||(_==null?void 0:_.tagName)==="INPUT"||(_==null?void 0:_.tagName)==="TEXTAREA"||(_==null?void 0:_.tagName)==="SELECT"||((g.key==="h"||g.key==="H")&&P(Pt,!m(Pt)),g.key==="Escape"&&e.onexit&&e.onexit(),(g.key===" "||g.key.toLowerCase()==="p")&&!o()&&(g.preventDefault(),Ut()))}let be=j(null),Ae=j(!0),De=j(""),Te=j(0),jt=j(0),Je=j(0),$e=j(0),et=j(""),ce=j(null),k=null,tt=null,qt=null,Bt=j(null),Vt=null,Wt=null,ze=j(null),St=!1,Fe=j(ar([]));async function Kr(){P(Ae,!0),P(De,"");try{if(v()){P(be,v()),P(Je,v().nodeCount,!0),P($e,v().edgeCount,!0),P(et,v().center_id,!0);return}const g=new Set(h().filter(Boolean)),_=g.size?await(async()=>{var V,K;const C=await Promise.all([...g].map(N=>kt.graph({center_id:N,max_nodes:200,depth:3}))),O=[...new Map(C.flatMap(N=>N.nodes).map(N=>[N.id,N])).values()].filter(N=>g.has(N.id)),B=new Set(O.map(N=>N.id)),G=[...new Map(C.flatMap(N=>N.edges).map(N=>[`${N.source}:${N.target}`,N])).values()].filter(N=>B.has(N.source)&&B.has(N.target));return{...C[0],nodes:O,edges:G,center_id:((V=O[0])==null?void 0:V.id)??((K=C[0])==null?void 0:K.center_id)??"",nodeCount:O.length,edgeCount:G.length}})():await kt.graph({max_nodes:200,depth:3,sort:"connected"});P(be,_,!0),P(Je,_.nodeCount,!0),P($e,_.edgeCount,!0),P(et,_.center_id,!0)}catch(g){const _=g instanceof Error?g.message:"Failed to load graph data";/\b404\b/.test(_)?(P(be,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:""},!0),P(Je,0),P($e,0),P(et,"")):P(De,_,!0)}finally{P(Ae,!1)}}let rt=null,it=j(ar([])),nt=j("recalls");function Xr(g,_){P(Te,g,!0),P(jt,_,!0),rt&&!m(x)&&rt.tick(g)}async function Qr(){var V;if(!T||!(k!=null&&k.graph))return;const g=k.graph,_=K=>g.indexById.has(K),C=K=>{var N;return((N=g.nodes[g.indexById.get(K)??-1])==null?void 0:N.label)??K.slice(0,8)};let O=[];try{O=((V=await kt.receipts.list(60))==null?void 0:V.receipts)??[]}catch{}let B=Di(O,_);B.length===0&&(B=Ti(g.nodes,12)),B.length>0&&(rt=new Ui(T,{intervalFrames:240}),rt.setItems(B));const G=zi(O,_,3);G.length>0?(P(nt,"recalls"),P(it,G.map(K=>({...K,label:C(K.id)})),!0)):(P(nt,"retention"),P(it,[...g.nodes].filter(K=>(K.label??"").trim().length>0).sort((K,N)=>N.retention-K.retention).slice(0,3).map(K=>({id:K.id,recalls:Math.round(K.retention*100),label:K.label||K.id.slice(0,8)})),!0))}function Zr(g){var _;St=!1,k==null||k.dispose(),P(ce,g,!0),k=new mn(g),pe.enabled=!o(),o()&&pe.reset(),k.setCameraRig(pe.state),(_=e.onready)==null||_.call(e,g)}Ne(()=>{if(m(ce)&&k&&m(be)&&!St){St=!0;const g=e.demo==="engram-birth",_=e.demo==="salience-rescue",C=e.demo==="forgetting-horizon",O=e.demo==="firewall";if(k.upload(m(be),a(),{recallPath:!g&&!_&&!C&&!O}),g){tt=new Mn({engine:m(ce),nodeRenderer:k,seed:a()}),tt.upload(a());const B=tt.engraveSteps,G=[];for(let V=0;V<B.length/4;V++)G.push({sourceIndex:B[V*4],targetIndex:B[V*4+1],beatFrame:B[V*4+2],kind:B[V*4+3],beatKind:"engrave",nodeId:`engrave-${V}`,label:"edge engraved"});k.setPathSteps(B,G),P(Fe,tt.timeline.map((V,K)=>({sourceIndex:0,targetIndex:0,beatFrame:V.startFrame,kind:0,beatKind:"birth",nodeId:`birth-${K}`,label:V.label})),!0)}else if(_){const B=Hn(m(be),k.graph,a(),e.backfillEvidence);P(Bt,B,!0),B.viable&&(qt=new Gn({engine:m(ce),nodeRenderer:k,plan:B}),qt.upload(),k.setPathSteps(B.pathData,B.pathMetas)),P(Fe,B.spineBeats,!0)}else if(C){const B=oa(k.graph);B.viable&&(Vt=new Zn({engine:m(ce),nodeRenderer:k,plan:B}),Vt.upload(),k.setPathSteps(B.pathData,B.pathMetas)),P(Fe,B.spineBeats,!0)}else if(O){const B=Pa(k.graph,a());P(ze,B,!0),B.viable&&(Wt=new Mr({engine:m(ce),nodeRenderer:k,plan:B}),Wt.upload(),k.setPathSteps(B.pathData,B.pathMetas)),P(Fe,B.spineBeats,!0)}else P(Fe,k.pathSteps,!0);if(p()&&k.graph&&m(be)){T=new Ca({engine:m(ce),renderer:k,graph:k.graph,response:m(be),seed:a(),projectionDays:()=>m(D),chronoOffsetDays:()=>m(U),onFirewall:G=>{P(Qe,G.intruderLabel,!0),P(Ze,Date.now(),!0)}}),P(ie,T.liveDecayAvailable,!0),m(ce).setPreFrameHook(G=>T==null?void 0:T.drain(G)),o()||Qr();let B=Number.POSITIVE_INFINITY;for(const G of k.graph.nodes)if(G.createdAt){const V=Date.parse(G.createdAt);Number.isFinite(V)&&V<B&&(B=V)}if(Number.isFinite(B)&&P(S,Math.floor((B-Date.now())/864e5)-1),y){const G=Date.parse(y);Number.isFinite(G)&&P(b,Math.min(365,Math.max(m(S),(G-Date.now())/864e5)),!0),y=null}o()||(A=new za(m(ce),k,Yr()),m(ce).addPass(A),Z=new Aa(m(ce),k.graph.nodes),m(ce).addPass(Z),P(H,!0)),typeof window<"u"&&(window.__vestigeLiveBridge=T)}m(ce).demoClock.reset()}}),Ne(()=>{const g=t();T&&T.ingest(g)}),Ne(()=>{m(b),T==null||T.refreshDecay(),Z==null||Z.setTimeline(m(b),m(x)),A==null||A.setScrubbing(m(x))}),Ne(()=>{if(!m(Ze))return;const g=setTimeout(()=>{P(Qe,""),P(Ze,0)},7e3);return()=>clearTimeout(g)}),bi(()=>{y=new URLSearchParams(window.location.search).get("t"),Kr();const g=zr();return()=>{if(yt(),g==null||g(),k==null||k.dispose(),k=null,typeof window<"u"){const _=window;_.__vestigeLiveBridge===T&&delete _.__vestigeLiveBridge}}});var Yt=$a();Oe("keydown",yi,Hr);var at=Lt(Yt);let Ht;var me=M(at);let Kt;var Jr=M(me);Ci(Jr,{get demo(){return e.demo},get seed(){return a()},get freezeFrame(){return s()},get maxDpr(){return f()},onframe:Xr,onready:Zr}),E(me),wi(me,g=>P(ve,g),()=>m(ve));var $r=L(me,2);{var ei=g=>{var _=Za(),C=M(_);{var O=I=>{var R=Oa(),$=M(R),X=M($,!0);E($);var ue=L($,2);Mt(ue,19,()=>m(it),Be=>Be.id,(Be,Se,It)=>{var Ue=Ua(),Pe=M(Ue),st=M(Pe,!0);E(Pe);var ot=L(Pe,2),pi=M(ot,!0);E(ot);var nr=L(ot,2),mi=M(nr);E(nr),E(Ue),se(()=>{ee(st,m(It)+1),Re(ot,"title",m(Se).label),ee(pi,m(Se).label),ee(mi,`${m(Se).recalls??""}${m(nt)==="recalls"?"×":"%"}`)}),Y(Be,Ue)}),E(R),se(()=>ee(X,m(nt)==="recalls"?"Most recalled · your mind":"Strongest memories · your mind")),Y(I,R)};J(C,I=>{p()&&m(it).length>0&&I(O)})}var B=L(C,2);{var G=I=>{var R=Na(),$=L(M(R),2),X=M($,!0);E($),_i(2),E(R),se(()=>ee(X,m(Qe))),Y(I,R)};J(B,I=>{p()&&m(Ur)&&I(G)})}var V=L(B,2);{var K=I=>{var R=ja(),$=M(R,!0);E(R),se(()=>{Re(R,"title",m(xe)?"Resume field motion":"Pause field motion"),Re(R,"aria-pressed",m(xe)),Re(R,"aria-label",m(xe)?"Resume 3D memory field motion":"Pause 3D memory field motion"),ee($,m(xe)?"▶ RESUME":"❚❚ PAUSE")}),he("click",R,Ut),Y(I,R)};J(V,I=>{o()||I(K)})}var N=L(V,2);{var ii=I=>{var R=Va();let $;var X=L(M(R),2);xi(X);var ue=L(X,2);let Be;var Se=M(ue,!0);E(ue);var It=L(ue,2);{var Ue=Pe=>{var st=qa();he("click",st,()=>P(b,0)),Y(Pe,st)};J(It,Pe=>{m(b)!==0&&Pe(Ue)})}E(R),se(()=>{$=Ee(R,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]`,null,$,{"opacity-100":m(H),"opacity-75":!m(H)}),Re(X,"min",m(S)),Be=Ee(ue,1,"w-16 text-right tabular-nums",null,Be,{"text-[#b9d9a9]":m(b)>=0,"text-[#dfc68e]":m(b)<0}),ee(Se,m(oe))}),he("input",X,()=>P(x,!0)),he("change",X,()=>P(x,!1)),he("pointerup",X,()=>P(x,!1)),Oe("pointercancel",X,()=>P(x,!1)),Oe("blur",X,()=>P(x,!1)),Pi(X,()=>m(b),Pe=>P(b,Pe)),Y(I,R)};J(N,I=>{p()&&m(ie)&&I(ii)})}var Qt=L(N,2);{var ni=I=>{Vi(I,{get demoMode(){return e.demo},get seed(){return a()},get nodeCount(){return m(Je)},get edgeCount(){return m($e)},get centerId(){return m(et)},get frameCount(){return m(Te)},get fpsEstimate(){return m(jt)},get freezeFrame(){return s()},get loading(){return m(Ae)},get error(){return m(De)}})};J(Qt,I=>{d()==="full"&&I(ni)})}var Zt=L(Qt,2);{var ai=I=>{var R=Wa();he("click",R,function(...$){var X;(X=e.onexit)==null||X.apply(this,$)}),Y(I,R)};J(Zt,I=>{d()==="full"&&e.onexit&&I(ai)})}var Jt=L(Zt,2);{var si=I=>{var R=Ha();Mt(R,20,()=>Ei,$=>$,($,X)=>{var ue=Ya(),Be=M(ue,!0);E(ue),se(()=>{Ee(ue,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${X===e.demo?"text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]":"text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50"}`),Re(ue,"title",`Play the ${Nt[X]??""} moment`),ee(Be,Nt[X])}),he("click",ue,()=>{var Se;return(Se=e.ondemochange)==null?void 0:Se.call(e,X)}),Y($,ue)}),E(R),Y(I,R)};J(Jt,I=>{d()==="full"&&c()&&I(si)})}var $t=L(Jt,2);{var oi=I=>{var R=Ka();Y(I,R)};J($t,I=>{m(Ae)&&I(oi)})}var er=L($t,2);{var li=I=>{var R=Xa(),$=M(R),X=M($,!0);E($),E(R),se(()=>ee(X,m(De))),Y(I,R)};J(er,I=>{m(De)&&!m(Ae)&&I(li)})}var tr=L(er,2);{var ci=I=>{Ki(I,{get steps(){return m(Fe)},get frame(){return m(Te)}})};J(tr,I=>{d()==="full"&&I(ci)})}var rr=L(tr,2);{var di=I=>{sr(I,{get frame(){return m(Te)},get verdict(){return m(Bt).verdict}})};J(rr,I=>{var R;d()==="full"&&e.demo==="salience-rescue"&&((R=m(Bt))!=null&&R.viable)&&I(di)})}var ir=L(rr,2);{var ui=I=>{{let R=Ce(()=>({headline:m(ze).verdict.headline,causeLabel:m(ze).verdict.intruderLabel,receipt:m(ze).verdict.receipt}));sr(I,{get frame(){return m(Te)},tone:"quarantine",fadeWindow:[480,495,605,620],get verdict(){return m(R)}})}};J(ir,I=>{var R;d()==="full"&&e.demo==="firewall"&&((R=m(ze))!=null&&R.viable)&&I(ui)})}var fi=L(ir,2);{var hi=I=>{var R=Qa();Y(I,R)};J(fi,I=>{!m(Ae)&&m(be)&&m(be).nodeCount===0&&I(hi)})}E(_),Y(g,_)};J($r,g=>{m(Pt)&&g(ei)})}E(at);var Xt=L(at,2);Li(Xt,{get pick(){return m(wt)},onclose:()=>P(wt,null)});var ti=L(Xt,2);{var ri=g=>{var _=Ja(),C=M(_);E(_),se(()=>ee(C,`HOVER ${m(xt)??""}`)),Y(g,_)};J(ti,g=>{m(xt)&&!o()&&g(ri)})}se(()=>{Ht=Ee(at,1,`${l()?"absolute":"fixed"} inset-0 overflow-hidden bg-[#05060a]`,null,Ht,{"cursor-none":o()}),Kt=Ee(me,1,"absolute inset-0 z-0 touch-none",null,Kt,{"cursor-crosshair":!!e.onpick&&!o()})}),he("click",me,Or),he("pointerdown",me,Nr),he("pointermove",me,jr),he("pointerup",me,qr),Oe("pointercancel",me,Vr),Oe("wheel",me,Wr),Y(i,Yt),vt(),n()}wr(["click","pointerdown","pointermove","pointerup","input","change"]);export{bs as O};

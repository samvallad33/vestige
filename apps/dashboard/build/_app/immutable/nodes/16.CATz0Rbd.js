import{$ as e,A as t,B as n,C as r,D as i,E as a,G as o,J as s,K as c,M as l,N as u,O as d,P as f,Q as p,T as m,U as h,V as g,W as _,X as v,Y as y,Z as b,_ as x,a as S,at as C,c as ee,d as w,et as T,f as E,ft as te,j as D,lt as ne,m as re,n as ie,ot as ae,p as oe,pt as O,q as k,r as se,u as ce,ut as A,w as j,x as le,z as ue}from"../chunks/DHlIFcq6.js";import{s as de,t as fe}from"../chunks/DL5zNdDR.js";import"../chunks/xihTtKlq.js";import"../chunks/C83sBwT4.js";import{t as pe}from"../chunks/CR5AJ_hc.js";import{n as me,t as he}from"../chunks/yeJIv7a5.js";import{C as M,S as N,_ as P,b as ge,d as _e,f as ve,g as ye,h as F,i as be,m as xe,o as Se,p as Ce,r as we,t as Te,u as Ee,v as I,x as L,y as R}from"../chunks/CgAKUn-r.js";import{t as De}from"../chunks/GaTn0dX8.js";import{i as Oe,n as ke}from"../chunks/CXW1AunC.js";import{r as Ae}from"../chunks/D2q35Vo6.js";import{i as je,r as Me,t as Ne}from"../chunks/BciDxNzc.js";var Pe=O({prerender:()=>!1,ssr:()=>!1});function Fe(e,t){let n=[];for(let r of e){let e=((r.activation_path?.length?r.activation_path:r.retrieved)??[]).filter(t);if(e.length===0)continue;let i=e[e.length-1];n.push({targetId:i,pathIds:e})}return n}function Ie(e,t=12){return[...e].sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,t).map(e=>({targetId:e.id,pathIds:[e.id]}))}function Le(e,t,n=5){let r=new Map;for(let n of e){let e=(n.activation_path?.length?n.activation_path:n.retrieved)??[];for(let n of new Set(e))t(n)&&r.set(n,(r.get(n)??0)+1)}return[...r.entries()].map(([e,t])=>({id:e,recalls:t})).sort((e,t)=>t.recalls-e.recalls||e.id.localeCompare(t.id)).slice(0,n)}var Re=class{bridge;items=[];cursor=0;ticks=0;nextTick=0;intervalFrames;enabled=!0;started=!1;constructor(e,t={}){this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}let t=this.items[this.cursor%this.items.length];this.cursor++;let n=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(n?0:30)}},ze=d(`<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>`),z=d(`<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>`),Be=d(`<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>`),Ve=d(`<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>`);function He(e,t){ae(t,!0);let r=S(t,`demoMode`,3,`recall-path`),i=S(t,`seed`,3,`vestige-observatory-v1`),c=S(t,`nodeCount`,3,0),u=S(t,`edgeCount`,3,0),d=S(t,`centerId`,3,``),f=S(t,`frameCount`,3,0),p=S(t,`fpsEstimate`,3,0),h=S(t,`freezeFrame`,3,null);S(t,`loading`,3,!1),S(t,`error`,3,``);function g(){let e=new URLSearchParams({demo:r(),seed:i()});h()!==null&&e.set(`frame`,String(h()));let t=`${window.location.origin}${de}/observatory?${e.toString()}`;navigator.clipboard.writeText(t).catch(()=>{})}var _=Ve(),v=o(_),y=o(v),b=o(y),x=k(b,!0),ee=s(b,2),w=k(ee);A(y);var T=s(y,2),E=o(T),te=k(E),D=s(E,2),ne=e=>{var t=ze(),r=k(t);n(e=>m(r,`center=${e??``}`),[()=>d().slice(0,8)]),a(e,t)};j(D,e=>{d()&&e(ne)}),A(T);var re=s(T,2),ie=o(re),oe=k(ie),O=s(ie,2),se=e=>{var t=z();a(e,t)},ce=e=>{var t=Be(),r=k(t);n(()=>m(r,`${p()??``}fps`)),a(e,t)};j(O,e=>{h()===null?p()>0&&e(ce,1):e(se)});var le=s(O,2);A(re),A(v),A(_),n((e,t)=>{m(x,r()),m(w,`seed=${e??``}${i().length>12?`…`:``}`),m(te,`${c()??``} nodes · ${u()??``} edges`),m(oe,`frame: ${t??``}`)},[()=>i().slice(0,12),()=>String(f()).padStart(3,` `)]),l(`click`,le,g),a(e,_),C()}D([`click`]);var Ue=d(`<div class="active-label svelte-8n8iia"> </div>`),We=d(`<div></div>`),Ge=d(`<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>`);function Ke(e,t){ae(t,!0);let r=S(t,`steps`,19,()=>[]),l=S(t,`frame`,3,0),u=S(t,`loopFrames`,3,720),d=e=>e/u()*100;function h(e,t){let n=t-e;return n<-14||n>90?0:n<0?1+n/14:1-n/90}let g=p(()=>{let e=``,t=.15;for(let n of r()){let r=h(n.beatFrame,l());r>t&&(t=r,e=n.label)}return e});var _=i(),v=c(_),y=e=>{var t=Ge(),i=o(t),c=e=>{var t=Ue(),r=k(t,!0);n(()=>m(r,f(g))),a(e,t)};j(i,e=>{f(g)&&e(c)});var u=s(i,2),p=o(u);le(p,17,r,e=>e.beatFrame,(e,t)=>{var r=We();let i;n((e,n,a)=>{i=re(r,1,`tick svelte-8n8iia`,null,i,{hot:e,backward:f(t).kind===1}),oe(r,`left: ${n??``}%; opacity: ${a??``}`),E(r,`title`,f(t).label)},[()=>h(f(t).beatFrame,l())>0,()=>d(f(t).beatFrame),()=>.45+.55*h(f(t).beatFrame,l())]),a(e,r)});var _=s(p,2);A(u),A(t),n(e=>oe(_,`left: ${e??``}%`),[()=>d(l())]),a(e,t)};j(v,e=>{r().length>0&&e(y)}),a(e,_),C()}var qe=d(`<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>`);function Je(e,t){ae(t,!0);let r=S(t,`frame`,3,0),l=S(t,`fadeWindow`,19,()=>[600,620,705,719]),u=S(t,`tone`,3,`triumph`),d=(e,t,n)=>{let r=Math.min(1,Math.max(0,(n-e)/(t-e)));return r*r*(3-2*r)},h=p(()=>d(l()[0],l()[1],r())*(1-d(l()[2],l()[3],r())));var g=i(),_=c(g),v=e=>{var r=qe();let i;var c=o(r),l=k(c,!0),d=s(c,2),p=k(d,!0),g=s(d,2),_=k(g,!0);A(r),n(()=>{i=re(r,1,`verdict svelte-ssd7yu`,null,i,{quarantine:u()===`quarantine`}),oe(r,`opacity: ${f(h)??``}`),m(l,t.verdict.headline),m(p,t.verdict.causeLabel),m(_,t.verdict.receipt)}),a(e,r)};j(_,e=>{f(h)>.001&&e(v)}),a(e,g),C()}function Ye(e,t,n,r){let i=1/Math.tan(e/2),a=1/(n-r),o=new Float32Array(16);return o[0]=i/t,o[5]=i,o[10]=r*a,o[11]=-1,o[14]=r*n*a,o}function Xe(e,t,n){let[r,i,a]=e,o=r-t[0],s=i-t[1],c=a-t[2],l=Math.hypot(o,s,c)||1;o/=l,s/=l,c/=l;let u=n[1]*c-n[2]*s,d=n[2]*o-n[0]*c,f=n[0]*s-n[1]*o;l=Math.hypot(u,d,f)||1,u/=l,d/=l,f/=l;let p=s*f-c*d,m=c*u-o*f,h=o*d-s*u,g=new Float32Array(16);return g[0]=u,g[1]=p,g[2]=o,g[4]=d,g[5]=m,g[6]=s,g[8]=f,g[9]=h,g[10]=c,g[12]=-(u*r+d*i+f*a),g[13]=-(p*r+m*i+h*a),g[14]=-(o*r+s*i+c*a),g[15]=1,g}function Ze(e,t){let n=new Float32Array(16);for(let r=0;r<4;r++)for(let i=0;i<4;i++)n[r*4+i]=e[i]*t[r*4]+e[4+i]*t[r*4+1]+e[8+i]*t[r*4+2]+e[12+i]*t[r*4+3];return n}function Qe(e,t,n,r=.35,i=0){let a=e*Math.PI*2+i,o=[Math.sin(a)*n,n*r,Math.cos(a)*n],s=Ye(50*Math.PI/180,t,.1,4e3),c=Xe(o,[0,0,0],[0,1,0]),l=-o[0],u=-o[1],d=-o[2],f=Math.hypot(l,u,d)||1;l/=f,u/=f,d/=f;let p=u*0-d*1,m=d*0-l*0,h=l*1-u*0;f=Math.hypot(p,m,h)||1,p/=f,m/=f,h/=f;let g=m*d-h*u,_=h*l-p*d,v=p*u-m*l;return{viewProj:Ze(s,c),right:[p,m,h],up:[g,_,v],eye:o}}var $e={yaw:0,pitch:0,zoom:1},et=.38,tt=2.6,nt=-.18,rt=.82;function it(e,t,n){return Math.min(n,Math.max(t,e))}function at(e){return{yaw:Number.isFinite(e.yaw)?e.yaw:0,pitch:it(Number.isFinite(e.pitch)?e.pitch:0,nt,rt),zoom:it(Number.isFinite(e.zoom)?e.zoom:1,et,tt)}}function ot(e,t,n,r=$e){let i=at(r);return Qe(e,t,n/i.zoom,.35+i.pitch,i.yaw)}var st=class{state={...$e};dragging=!1;pointerId=null;lastX=0;lastY=0;pinch0=0;pointers=new Map;enabled=!0;reset(){this.state={...$e},this.dragging=!1,this.pointerId=null,this.pointers.clear()}onPointerDown(e){this.enabled&&e.button===0&&(this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===1?(this.dragging=!0,this.pointerId=e.pointerId,this.lastX=e.clientX,this.lastY=e.clientY,e.currentTarget?.setPointerCapture?.(e.pointerId)):this.pointers.size===2&&(this.pinch0=ct(this.pointers)))}onPointerMove(e){if(!this.enabled)return!1;if(this.pointers.has(e.pointerId)&&this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===2&&this.pinch0>0){let e=ct(this.pointers),t=e/this.pinch0;return this.state=at({...this.state,zoom:this.state.zoom*it(t,.94,1.06)}),this.pinch0=e,!0}if(!this.dragging||e.pointerId!==this.pointerId)return!1;let t=e.clientX-this.lastX,n=e.clientY-this.lastY;return this.lastX=e.clientX,this.lastY=e.clientY,this.state=at({yaw:this.state.yaw-t*.005,pitch:this.state.pitch+n*.003,zoom:this.state.zoom}),!0}onPointerUp(e){this.pointers.delete(e.pointerId),e.pointerId===this.pointerId&&(this.dragging=!1,this.pointerId=null),this.pointers.size<2&&(this.pinch0=0)}onWheel(e){if(!this.enabled)return!1;e.preventDefault();let t=e.deltaY>0?.92:1.08;return this.state=at({...this.state,zoom:this.state.zoom*t}),!0}};function ct(e){let t=[...e.values()];return t.length<2?0:Math.hypot(t[0].x-t[1].x,t[0].y-t[1].y)}function lt(e){let t=/^#?([0-9a-fA-F]{6})$/.exec(e.trim());if(!t)return[107/255,114/255,128/255];let n=parseInt(t[1],16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}function ut(e){return lt(Me({tags:e.tags})||Ne[je(e.retention)])}function dt(e){let t=[...e.nodes].sort((e,t)=>e.isCenter===t.isCenter?e.id<t.id?-1:+(e.id>t.id):e.isCenter?-1:1).map((e,t)=>L(e,t)),n=new Map;for(let e of t)n.set(e.id,e.index);let r=[];for(let t of e.edges){let e=n.get(t.source),i=n.get(t.target);e!==void 0&&i!==void 0&&e!==i&&r.push({sourceIndex:e,targetIndex:i,weight:t.weight,type:t.type})}let i=t.findIndex(e=>e.isCenter);return{nodes:t,edges:r,indexById:n,centerIndex:i<0?0:i}}function ft(e,t,n=120){let r=e.nodes.length,i=new Float32Array(r*16);for(let a=0;a<r;a++){let o=e.nodes[a],s=a*16,[c,l,u]=o.isCenter&&e.centerIndex===a?[0,0,0]:M(a,r,n,t),d=o.isCenter?4.2:1.4+o.retention*1.8;i[s+P.posRadius+0]=c,i[s+P.posRadius+1]=l,i[s+P.posRadius+2]=u,i[s+P.posRadius+3]=d,i[s+P.velRetention+3]=o.retention;let[f,p,m]=ut(o),h=0;o.isCenter&&(h|=ye.isCenter),o.suppressed&&(h|=ye.suppressed);let g=new Set(o.tags.map(e=>e.toLowerCase()));g.has(`aha`)&&(h|=ye.isAha),(g.has(`failure`)||g.has(`guardrail`))&&(h|=ye.isFailure),(g.has(`confusion`)||g.has(`weak-spot`))&&(h|=ye.isConfusion),i[s+P.colorFlags+0]=f,i[s+P.colorFlags+1]=p,i[s+P.colorFlags+2]=m,i[s+P.colorFlags+3]=h}return{data:i,nodeCount:r}}function pt(e){let t=new Uint32Array(Math.max(1,e.edges.length)*2);return e.edges.forEach((e,n)=>{t[n*2]=e.sourceIndex,t[n*2+1]=e.targetIndex}),t}var mt=`
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
`,ht=`
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
`,gt=`
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
`,_t=`
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
`;function vt(e){return 60+e*60}function B(e,t,n=8,r={}){let i=[...e.nodes].sort((e,t)=>e.id<t.id?-1:+(e.id>t.id)),a=[...e.edges].sort((e,t)=>{let n=`${e.source} ${e.target} ${e.type}`,r=`${t.source} ${t.target} ${t.type}`;return n<r?-1:+(n>r)}),o=r.centerId??e.center_id,s=Ae(i,a,o,n,{preferCausal:r.preferCausal}),c=[];for(let e=0;e<s.beats.length;e++){let n=s.beats[e],r=t.indexById.get(n.nodeId);if(r===void 0)continue;let i=e>0?s.beats[e-1].nodeId:n.nodeId,a=t.indexById.get(i)??r,o=(n.viaEdge?.type??``).toLowerCase(),l=o===`causal`||o.includes(`causal`),u=n.kind===`contradiction`||l;c.push({sourceIndex:a,targetIndex:r,beatFrame:vt(e),kind:u?R.backwardCause:R.recall,beatKind:n.kind,nodeId:n.nodeId,label:n.node.label})}let l=new Uint32Array(Math.max(1,c.length)*4);return c.forEach((e,t)=>{l[t*4]=e.sourceIndex,l[t*4+1]=e.targetIndex,l[t*4+2]=e.beatFrame,l[t*4+3]=e.kind}),{data:l,steps:c,path:s}}var V=24,yt=300,bt=128,xt=class{engine;pipeline=null;bindGroup=null;cameraBuffer=null;nodeBuffer=null;edgeBuffer=null;cameraData=new Float32Array(V);nodeCount=0;simPipeline=null;simBindGroup=null;pathBuffer=null;liveRetentionBuffer=null;pickReadback=null;disposed=!1;edgeCapacityBytes=0;edgeCount=0;cameraRig={...$e};hoveredIndex=-1;pathPipeline=null;pathBindGroup=null;pathStepCount=0;axonPipeline=null;axonBindGroup=null;graph=null;pathSteps=[];constructor(e){this.engine=e,e.addPass(this)}upload(e,t,n){let r=this.engine.gpuDevice;if(!r)return;let i=n?.recallPath??!0,a=dt(e);this.graph=a;let{data:o,nodeCount:s}=ft(a,new N({seed:t}).state.rng);this.nodeCount=s,this.nodeBuffer?.destroy(),this.nodeBuffer=r.createBuffer({label:`observatory-node-state`,size:Math.max(o.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),r.queue.writeBuffer(this.nodeBuffer,0,o.buffer);let c=pt(a);this.edgeCount=a.edges.length,this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(c.byteLength*2,64),this.edgeBuffer=r.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.edgeBuffer,0,c.buffer);let l=new Float32Array(Math.max(s,4));for(let e=0;e<s;e++)l[e]=Math.max(.001,a.nodes[e].retention);this.liveRetentionBuffer?.destroy(),this.liveRetentionBuffer=r.createBuffer({label:`observatory-live-retention`,size:l.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.liveRetentionBuffer,0,l.buffer);let u=i?B(e,a):{steps:[],data:new Uint32Array(4)};this.pathSteps=u.steps,this.pathBuffer?.destroy(),this.pathBuffer=r.createBuffer({label:`observatory-path-steps`,size:bt*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,u.data.buffer,0,Math.min(u.data.byteLength,bt*4*4)),this.pathStepCount=Math.min(this.pathSteps.length,bt),this.engine.params[2]=s,this.engine.params[3]=a.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||=r.createBuffer({label:`observatory-camera`,size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.createPipeline(r)}setPathSteps(e,t){let n=this.engine.gpuDevice;if(!n)return;this.pathSteps=t;let r=bt*4*4;if(this.pathBuffer&&e.byteLength<=r){this.pathStepCount=Math.min(t.length,bt),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,bt),this.pathBuffer?.destroy(),this.pathBuffer=n.createBuffer({label:`observatory-path-steps`,size:r,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,r)),this.engine.params[4]=this.pathStepCount,this.createPipeline(n)}setCameraRig(e){this.cameraRig=e}setHovered(e){this.hoveredIndex=e}currentOrbit(){let e=this.engine.params[6]||1,t=this.engine.params[7]||1,n=this.engine.params[1];return ot(n,e/t,yt,this.cameraRig)}setEdges(e){let t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e,this.edgeCount=e.length;let n=pt(this.graph),r=Math.max(n.byteLength,8),i=!1;(!this.edgeBuffer||r>this.edgeCapacityBytes)&&(this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(r*2,64),this.edgeBuffer=t.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),i=!0),t.queue.writeBuffer(this.edgeBuffer,0,n.buffer),this.engine.params[3]=e.length,i&&this.createPipeline(t)}uploadLiveRetention(e){let t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;let n=Math.min(e.length,this.nodeCount);n<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,n*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){let t=e.createShaderModule({label:`observatory-simulate`,code:ht});this.simPipeline=e.createComputePipeline({label:`observatory-recall-sim`,layout:`auto`,compute:{module:t,entryPoint:`recall_sim`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&n.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&n.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:`observatory-recall-sim-bind`,layout:this.simPipeline.getBindGroupLayout(0),entries:n})}let t=e.createShaderModule({label:`observatory-render-nodes`,code:mt});if(this.pipeline=e.createRenderPipeline({label:`observatory-nodes`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bindGroup=e.createBindGroup({label:`observatory-nodes-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){let t=e.createShaderModule({label:`observatory-render-path`,code:gt});this.pathPipeline=e.createRenderPipeline({label:`observatory-path`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.pathBindGroup=e.createBindGroup({label:`observatory-path-bind`,layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}if(this.edgeBuffer&&this.pathBuffer&&this.nodeBuffer){let t=e.createShaderModule({label:`observatory-render-axons`,code:_t});this.axonPipeline=e.createRenderPipeline({label:`observatory-axons`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`line-list`}}),this.axonBindGroup=e.createBindGroup({label:`observatory-axons-bind`,layout:this.axonPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.edgeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}},{binding:4,resource:{buffer:this.nodeBuffer}}]})}}compute(e){let t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;let n=this.currentOrbit();if(this.cameraData.set(n.viewProj,0),this.cameraData[16]=n.right[0],this.cameraData[17]=n.right[1],this.cameraData[18]=n.right[2],this.cameraData[19]=0,this.cameraData[20]=n.up[0],this.cameraData[21]=n.up[1],this.cameraData[22]=n.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){let t=e.beginComputePass({label:`observatory-recall-sim`});t.setPipeline(this.simPipeline),t.setBindGroup(0,this.simBindGroup),t.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),t.end()}}render(e){this.axonPipeline&&this.axonBindGroup&&this.edgeCount>0&&(e.setPipeline(this.axonPipeline),e.setBindGroup(0,this.axonBindGroup),e.draw(2,this.edgeCount)),this.pipeline&&this.bindGroup&&this.nodeCount!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){if(this.disposed)return null;let n=this.engine.gpuDevice;if(!n||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;this.pickReadback||=this.readNodePositions(n).finally(()=>{this.pickReadback=null});let r=await this.pickReadback;if(!r||this.disposed||!this.graph)return null;let i=this.currentOrbit().viewProj,a=1/Math.tan(50*Math.PI/360),o=-1,s=1/0;for(let n=0;n<this.nodeCount;n++){let c=n*16+P.posRadius,l=r[c],u=r[c+1],d=r[c+2],f=r[c+3],p=i[3]*l+i[7]*u+i[11]*d+i[15];if(p<=0)continue;let m=(i[0]*l+i[4]*u+i[8]*d+i[12])/p,h=(i[1]*l+i[5]*u+i[9]*d+i[13])/p,g=Math.max(f*a/p,.012),_=Math.hypot(m-e,h-t)/g;_<1.6*(n===this.hoveredIndex?.85:1)&&_<s&&(s=_,o=n)}return o<0?null:{index:o,id:this.graph.nodes[o].id}}async readNodePositions(e){if(!this.nodeBuffer)return null;let t=this.nodeCount*16*4,n=e.createBuffer({label:`observatory-pick-staging`,size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let r=e.createCommandEncoder({label:`observatory-pick-copy`});r.copyBufferToBuffer(this.nodeBuffer,0,n,0,t),e.queue.submit([r.finish()]),await n.mapAsync(GPUMapMode.READ);let i=new Float32Array(n.getMappedRange().slice(0));return n.unmap(),i}catch{return null}finally{n.destroy()}}dispose(){this.disposed=!0,this.nodeBuffer?.destroy(),this.edgeBuffer?.destroy(),this.cameraBuffer?.destroy(),this.pathBuffer?.destroy(),this.liveRetentionBuffer?.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null,this.axonPipeline=null,this.axonBindGroup=null,this.edgeCapacityBytes=0,this.edgeCount=0}},St=16,Ct=4,wt=110,Tt=.7,Et=.2,Dt=360,Ot=18;function kt(e){if(e.edges.length>0){let t=e.centerIndex,n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t);if(n.length>0){let r=-1,i=-1;for(let a of n){let n=a.sourceIndex===t?a.targetIndex:a.sourceIndex,o=e.nodes[n];o&&o.retention>i&&(i=o.retention,r=n)}if(r>=0)return r}}for(let t=0;t<e.nodes.length;t++)if(t!==e.centerIndex)return t;return e.centerIndex}function At(e,t,n=8192){let r=kt(e),i=e.nodes[r].id,a=jt(e,r),o=new N({seed:t+`:birth:`+i}).state.rng,s=new Float32Array(n*St),c=Math.floor(n*Tt),l=Math.floor(n*Et),u=n-c-l;for(let e=0;e<c;e++){let t=e*St,[n,r,i]=M(e,c,wt+o()*70,o);s[t+0]=a[0]+n,s[t+1]=a[1]+r,s[t+2]=a[2]+i,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}let d=e.edges.filter(e=>e.sourceIndex===r||e.targetIndex===r);for(let t=0;t<l;t++){let n=(c+t)*St;if(d.length===0)continue;let i=d[t%d.length],u=jt(e,i.sourceIndex===r?i.targetIndex:i.sourceIndex),f=u[0]-a[0],p=u[1]-a[1],m=u[2]-a[2],h=Math.sqrt(f*f+p*p+m*m)||1,g=t/Math.max(1,l)*2+.5,_=o()*30,v=-p*_/(h||1),y=f*_/(h||1);s[n+0]=a[0]+f/h*g*80+v,s[n+1]=a[1]+p/h*g*80+y,s[n+2]=a[2]+m/h*g*80+0,s[n+3]=o(),s[n+4]=a[0],s[n+5]=a[1],s[n+6]=a[2],s[n+7]=1+o()*1.8,s[n+8]=.91,s[n+9]=1,s[n+10]=.72,s[n+11]=o(),s[n+12]=0,s[n+13]=0,s[n+14]=0,s[n+15]=0}for(let e=0;e<u;e++){let t=(c+l+e)*St,n=o()*Math.PI*2,r=o()*120;s[t+0]=a[0]+Math.cos(n)*r,s[t+1]=a[1]+Math.sin(n)*r,s[t+2]=a[2]+180+o()*40,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}return{targetIndex:r,targetNodeId:i,particles:s,edgeSteps:Mt(e,r),timeline:Nt()}}function jt(e,t){let n=e.nodes[t],r=e.nodes.length;if(n.isCenter&&e.centerIndex===t)return[0,0,0];let i=Math.PI*(3-Math.sqrt(5)),a=1-t/(r-1||1)*2,o=Math.sqrt(1-a*a),s=i*t,c=(t*7+3)%100/100*.1*120-6,l=(t*13+7)%100/100*.1*120-6,u=(t*17+11)%100/100*.1*120-6;return[Math.cos(s)*o*120+c,a*120+l,Math.sin(s)*o*120+u]}function Mt(e,t){let n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t),r=n.length;if(r===0)return new Uint32Array;let i=new Uint32Array(r*Ct);for(let e=0;e<r;e++){let r=n[e],a=r.sourceIndex===t?r.targetIndex:r.sourceIndex,o=Dt+e*Ot;i[e*Ct+0]=t,i[e*Ct+1]=a,i[e*Ct+2]=o,i[e*Ct+3]=0}return i}function Nt(){return[{label:`latent trace condensing`,startFrame:60,endFrame:239},{label:`engram coalescence`,startFrame:240,endFrame:329},{label:`memory ignition`,startFrame:330,endFrame:359},{label:`associations engrave`,startFrame:360,endFrame:509},{label:`stabilization`,startFrame:510,endFrame:659}]}var Pt=`
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
`,Ft=16,It=6,Lt=330,Rt=359,zt=360,Bt=class{engine;nodeRenderer;active;computePipeline=null;computeBindGroup=null;particleBuffer=null;particleCount=0;renderPipeline=null;renderBindGroup=null;haloPipeline=null;haloBindGroup=null;haloIndexBuffer=null;engravePipeline=null;engraveBindGroup=null;engraveBuffer=null;engraveStepCount=0;timeline=[];birthPlan=null;get engraveSteps(){return this.birthPlan?.edgeSteps??new Uint32Array}constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}upload(e){let t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;let n=this.nodeRenderer.graph;if(!n)return;this.birthPlan=At(n,e),this.timeline=this.birthPlan.timeline;let r=this.birthPlan.particles.length/Ft;this.particleCount=r,this.particleBuffer?.destroy(),this.particleBuffer=t.createBuffer({label:`observatory-birth-particles`,size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),this.engraveBuffer?.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:`observatory-birth-engrave`,size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){let t=e.createShaderModule({label:`observatory-birth-compute`,code:Pt});this.computePipeline=e.createComputePipeline({label:`observatory-birth-compute-pipeline`,layout:`auto`,compute:{module:t,entryPoint:`birth_compute`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:`observatory-birth-compute-bind`,layout:this.computePipeline.getBindGroupLayout(0),entries:n})}createRenderPipeline(e){let t=e.createShaderModule({label:`observatory-birth-render`,code:`
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
`});this.renderPipeline=e.createRenderPipeline({label:`observatory-birth-render`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}});let n=this.nodeRenderer.cameraUniformBuffer;this.renderBindGroup=e.createBindGroup({label:`observatory-birth-render-bind`,layout:this.renderPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.particleBuffer}}]})}createHaloPipeline(e){let t=e.createShaderModule({label:`observatory-birth-halo`,code:`
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
`});this.haloPipeline=e.createRenderPipeline({label:`observatory-birth-halo`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}});let n=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:`observatory-birth-halo-bind`,layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;let t=e.createShaderModule({label:`observatory-birth-engrave`,code:gt});this.engravePipeline=e.createRenderPipeline({label:`observatory-birth-engrave-pipeline`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.engraveBindGroup=e.createBindGroup({label:`observatory-birth-engrave-bind`,layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){let n=this.engine.params[9];if(this.active=n===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;let r=e.beginComputePass({label:`observatory-birth-compute`});r.setPipeline(this.computePipeline),r.setBindGroup(0,this.computeBindGroup),r.dispatchWorkgroups(Math.ceil(this.particleCount/64)),r.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(It,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=Lt&&t<=Rt&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=zt&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){this.particleBuffer?.destroy(),this.particleBuffer=null,this.computePipeline?.destroy?.(),this.computePipeline=null,this.computeBindGroup=null,this.renderPipeline?.destroy?.(),this.renderPipeline=null,this.renderBindGroup=null,this.haloPipeline?.destroy?.(),this.haloPipeline=null,this.haloBindGroup=null,this.haloIndexBuffer?.destroy(),this.haloIndexBuffer=null,this.engravePipeline?.destroy?.(),this.engravePipeline=null,this.engraveBindGroup=null,this.engraveBuffer?.destroy(),this.engraveBuffer=null}};function Vt(e){return`
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

const HOP_SLOT: f32 = ${e.hopSlot.toFixed(1)};
const CAUSE_DEPTH: f32 = ${e.causeDepth.toFixed(1)};
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
`}var Ht=2,Ut=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;waveBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.waveBuffer?.destroy(),this.waveBuffer=e.createBuffer({label:`observatory-rescue-wave`,size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);let t=e.createShaderModule({label:`observatory-rescue-choreo`,code:Vt(this.plan.consts)});this.pipeline=e.createComputePipeline({label:`observatory-rescue-choreo`,layout:`auto`,compute:{module:t,entryPoint:`rescue_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-rescue-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==Ht||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-rescue-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.waveBuffer?.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}},Wt=65535,Gt={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function Kt(e,t){return ft(e,new N({seed:t}).state.rng).data}function qt(e){let t=new Uint32Array(e.nodes.length);for(let n of e.edges)t[n.sourceIndex]++,t[n.targetIndex]++;return t}function Jt(e,t){let n=e.nodes.length;if(n===0)return-1;let r=qt(e),i=n=>{let i=e.nodes[n],a=new Set(i.tags.map(e=>e.toLowerCase())),o=0;(a.has(`failure`)||a.has(`guardrail`))&&(o+=3),(a.has(`confusion`)||a.has(`weak-spot`))&&(o+=2),o+=Math.min(r[n],8)/8;let s=t[n*16+0],c=t[n*16+1],l=t[n*16+2];return Math.sqrt(s*s+c*c+l*l)>=54&&(o+=.5),o},a=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r[t]>=2,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex,()=>!0];for(let e of a){let t=-1,r=-1/0;for(let a=0;a<n;a++){if(!e(a))continue;let n=i(a);n>r&&(r=n,t=a)}if(t>=0)return t}return-1}function Yt(e,t){let n=e.nodes.length,r=new Uint16Array(n).fill(Wt),i=new Int32Array(n).fill(-1);if(t<0||t>=n)return{depths:r,parents:i};let a=Array.from({length:n},()=>[]);for(let t of e.edges){let e=Gt[t.type]??5;a[t.sourceIndex].push({nbr:t.targetIndex,rank:e}),a[t.targetIndex].push({nbr:t.sourceIndex,rank:e})}for(let e of a)e.sort((e,t)=>e.rank-t.rank||e.nbr-t.nbr);r[t]=0;let o=[t];for(let e=0;e<o.length;e++){let t=o[e];for(let{nbr:e}of a[t])r[e]===65535&&(r[e]=r[t]+1,o.push(e))}for(let e=0;e<n;e++)if(r[e]!==65535&&r[e]!==0){for(let{nbr:t}of a[e])if(r[t]===r[e]-1){i[e]=t;break}}return{depths:r,parents:i}}function Xt(e,t,n,r){let i=new Map;for(let t of e.nodes)i.set(t.id,t.createdAt);for(let e of[3,2,1]){let a=[];for(let i=0;i<t.nodes.length;i++){if(i===t.centerIndex||i===r)continue;let o=n[i];o===65535||o<e||a.push(i)}if(a.length===0)continue;let o=a.filter(e=>t.nodes[e].retention<=.45);o.length===0&&(o=a);let s=new Map,c=1/0,l=-1/0;for(let e of o){let n=i.get(t.nodes[e].id),r=n?Date.parse(n):NaN;Number.isFinite(r)&&(s.set(e,r),r<c&&(c=r),r>l&&(l=r))}let u=e=>{let t=s.get(e);return t===void 0?0:l===c?1:(l-t)/(l-c)},d=e=>2*(1-t.nodes[e].retention)+.5*Math.min(n[e],6)/6+.5*u(e);return o.sort((e,t)=>{let r=d(e),i=d(t);return i===r?n[t]===n[e]?e-t:n[t]-n[e]:i-r}),{index:o[0],depth:n[o[0]]}}return{index:-1,depth:0}}function Zt(e,t,n,r,i){let a=e[n*16+0],o=e[n*16+1],s=e[n*16+2],c=[];for(let l=0;l<t;l++){if(l===n||l===r||l===i)continue;let t=e[l*16+0]-a,u=e[l*16+1]-o,d=e[l*16+2]-s;c.push({i:l,d2:t*t+u*u+d*d})}return c.sort((e,t)=>e.d2-t.d2||e.i-t.i),c.slice(0,4).map(e=>e.i)}function Qt(e){return Math.min(84,Math.max(14,Math.floor(252/Math.max(1,e))))}function $t(e,t){return Math.min(260+t*e,514)}function en(e){return 138+28*e}function tn(e){return e.length>64?e.slice(0,64)+`…`:e}var nn=4;function rn(e){let t=new Uint32Array(e);return t.fill(Wt),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(e).fill(Wt),causeDepth:0,hopSlot:Qt(3),waveData:t,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`candidate cause found`,causeLabel:``,failureLabel:``,causeDate:``,hops:0,k:0,receipt:``},consts:{hopSlot:Qt(3),causeDepth:3}}}function an(e,t,n,r){if(r)return on(e,t,r);let i=t.nodes.length;if(i===0)return rn(0);let a=Kt(t,n),o=Jt(t,a);if(o<0)return rn(i);let{depths:s,parents:c}=Yt(t,o),l=Xt(e,t,s,o);if(l.index<0){let e=rn(i);return e.failureIndex=o,e.hopDepths=s,e}let u=l.index,d=Math.max(1,l.depth),f=Qt(d),p=e=>$t(e,f),m=Zt(a,i,o,u,t.centerIndex),h=m.length,g=new Uint32Array(i);for(let e=0;e<i;e++){let t=s[e]&65535;e===o&&(t|=65536),e===u&&(t|=1<<17),g[e]=t}m.forEach((e,t)=>{g[e]|=1<<18|t<<19});let _=[];m.forEach((e,t)=>{_.push({src:o,dst:e,bf:en(t),kind:R.probe,beatKind:`probe`})});let v=[];{let e=u;for(;e!==o&&e>=0&&c[e]>=0;)v.push(e),e=c[e]}let y=new Set(v),b=[];for(let e=0;e<i;e++){if(e===o||y.has(e))continue;let t=s[e];t===65535||t<1||t>d||c[e]<0||b.push(e)}b.sort((e,t)=>s[e]-s[t]||e-t);let x=[...v.slice().reverse(),...b].slice(0,48);x.sort((e,t)=>s[e]-s[t]||e-t);for(let e of x)_.push({src:c[e],dst:e,bf:p(s[e]),kind:R.backwardCause,beatKind:`wave`});_.push({src:u,dst:o,bf:560,kind:R.backwardCause,beatKind:`arc`});let S=new Uint32Array(Math.max(1,_.length)*nn),C=[];_.forEach((e,n)=>{S[n*nn+0]=e.src,S[n*nn+1]=e.dst,S[n*nn+2]=e.bf,S[n*nn+3]=e.kind,C.push({sourceIndex:e.src,targetIndex:e.dst,beatFrame:e.bf,kind:e.kind,beatKind:e.beatKind,nodeId:t.nodes[e.dst].id,label:tn(t.nodes[e.dst].label)})});let ee=tn(t.nodes[o].label),w=tn(t.nodes[u].label),T=[],E=(e,t,n,r)=>{T.push({sourceIndex:o,targetIndex:o,beatFrame:e,kind:t,beatKind:`rescue`,nodeId:r,label:n})};E(90,1,`failure: ${ee}`,t.nodes[o].id),m.forEach((e,n)=>{E(en(n),0,`lookalike ✗ · ${tn(t.nodes[e].label)}`,t.nodes[e].id)}),E(p(1),1,`reaching backward through time`,`rescue-wave-start`),d>=2&&p(d)!==p(1)&&E(p(d),1,`scrubbing past · ${d} hops`,`rescue-wave-deep`),E(560,1,`causal arc · ${w}`,t.nodes[u].id),E(600,1,`candidate cause found`,`rescue-verdict`);let te=e.nodes.find(e=>e.id===t.nodes[u].id)?.createdAt??``,D=te?te.slice(0,10):``;return{viable:!0,failureIndex:o,causeIndex:u,lookalikeIndices:m,hopDepths:s,causeDepth:d,hopSlot:f,waveData:g,pathData:S,pathMetas:C,spineBeats:T,verdict:{headline:`candidate cause found`,causeLabel:w,failureLabel:ee,causeDate:D,hops:d,k:h,receipt:`${d} hops back · ${D} · heuristic, no receipt · vector search: 0 for ${h}`},consts:{hopSlot:f,causeDepth:d}}}function on(e,t,n){let r=t.nodes.length,i=t.indexById.get(n.failureId)??-1,a=n.pathIds??[];if(i<0||a.length<2||a[a.length-1]!==n.failureId||new Set(a).size!==a.length)return rn(r);let o=a.map(e=>t.indexById.get(e));if(o.some(e=>e===void 0))return rn(r);let s=o,c=s[0];if(c===i)return rn(r);let l=n.candidates.find(e=>e.memoryId===a[0]);if(!l)return rn(r);let u=new Uint16Array(r);u.fill(Wt),u[i]=0,s.forEach((e,t)=>{u[e]=s.length-1-t});let d=new Uint32Array(r);d[i]=65536,s.slice(0,-1).forEach(e=>{d[e]=u[e]}),d[c]|=1<<17;let f=s.length-1,p=Qt(f),m=tn(t.nodes[c].label),h=tn(t.nodes[i].label),g=e.nodes.find(e=>e.id===l.memoryId)?.createdAt?.slice(0,10)??``,_=new Uint32Array((s.length-1)*nn),v=s.slice(0,-1).map((e,n)=>{let r=s[n+1],i=260+n*p;return _[n*nn]=e,_[n*nn+1]=r,_[n*nn+2]=i,_[n*nn+3]=R.backwardCause,{sourceIndex:e,targetIndex:r,beatFrame:i,kind:R.backwardCause,beatKind:`receipt-path`,nodeId:a[n+1],label:`recorded path · ${tn(t.nodes[r].label)}`}}),y=l.sharedEntities.length?l.sharedEntities.join(`, `):`recorded entity`,b=l.similarityRank===null?`rank unavailable`:`embedding rank #${l.similarityRank}`;return{viable:!0,failureIndex:i,causeIndex:c,lookalikeIndices:[],hopDepths:u,causeDepth:f,hopSlot:p,waveData:d,pathData:_,pathMetas:v,spineBeats:[{sourceIndex:i,targetIndex:i,beatFrame:90,kind:1,beatKind:`receipt-failure`,nodeId:n.failureId,label:`recorded failure · ${h}`},{sourceIndex:i,targetIndex:i,beatFrame:260,kind:1,beatKind:`receipt-join`,nodeId:`receipt-join`,label:`shared entity · ${y}`},{sourceIndex:c,targetIndex:i,beatFrame:260+(f-1)*p,kind:R.backwardCause,beatKind:`receipt-candidate`,nodeId:l.memoryId,label:`candidate · ${m}`},{sourceIndex:c,targetIndex:c,beatFrame:600,kind:1,beatKind:`receipt-verdict`,nodeId:`receipt-verdict`,label:`candidate cause found`}],verdict:{headline:`candidate cause found`,causeLabel:m,failureLabel:h,causeDate:g,hops:f,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${y} · ${b}`},consts:{hopSlot:p,causeDepth:f}}}var sn=`
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
`,cn=3,ln=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;horizonBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.horizonBuffer?.destroy(),this.horizonBuffer=e.createBuffer({label:`observatory-forgetting-horizon`,size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);let t=e.createShaderModule({label:`observatory-forgetting-choreo`,code:sn});this.pipeline=e.createComputePipeline({label:`observatory-forgetting-choreo`,layout:`auto`,compute:{module:t,entryPoint:`forgetting_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-forgetting-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==cn||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-forgetting-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.horizonBuffer?.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}};function un(e){let t=[];for(let n=0;n<e.nodes.length;n++)n!==e.centerIndex&&t.push(n);t.sort((t,n)=>e.nodes[t].retention-e.nodes[n].retention||t-n);let n=t.length;if(n===0)return[];let r=Math.min(n,Math.max(Math.min(3,n),Math.round(.25*n)));return t.slice(0,r)}function dn(e,t){let n=new Uint32Array(e.nodes.length);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>2*e.nodes[t].retention+Math.min(n[t],8)/8;return t.slice().sort((e,t)=>r(t)-r(e)||e-t).slice(0,Math.min(3,t.length))}function fn(e){return 318+60*e}var pn=4;function mn(e){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function hn(e){let t=e.nodes.length,n=un(e);if(t<2||n.length<1)return mn(t);let r=dn(e,n),i=n.length,a=new Uint32Array(t);n.forEach((e,t)=>{let n=Math.round(255*t/Math.max(1,i-1));a[e]=n&255|256}),r.forEach((e,t)=>{a[e]|=512|t<<10});let o=new Uint32Array(Math.max(1,r.length)*pn),s=[];r.forEach((t,n)=>{let r=fn(n);o[n*pn+0]=e.centerIndex,o[n*pn+1]=t,o[n*pn+2]=r,o[n*pn+3]=R.recall,s.push({sourceIndex:e.centerIndex,targetIndex:t,beatFrame:r,kind:R.recall,beatKind:`recall`,nodeId:e.nodes[t].id,label:tn(e.nodes[t].label)})});let c=[],l=(t,n,r,i)=>{c.push({sourceIndex:e.centerIndex,targetIndex:e.centerIndex,beatFrame:t,kind:n,beatKind:`horizon`,nodeId:i,label:r})},u=new Set(r),d=n.filter(e=>!u.has(e)).slice(0,3);return d.forEach((t,n)=>{let r=Math.round(e.nodes[t].retention*100);l(132+60*n,1,`fading: ${tn(e.nodes[t].label)} · retention ${r}%`,e.nodes[t].id)}),r.forEach((t,n)=>{l(fn(n),0,`recalled: ${tn(e.nodes[t].label)}`,e.nodes[t].id)}),d.length>0&&l(540,1,`the unrecalled sink · nothing is deleted`,`horizon-sink`),l(660,0,`every memory still retrievable`,`horizon-retrievable`),{viable:!0,driftingIndices:n,rescuedIndices:r,horizonData:a,pathData:o,pathMetas:s,spineBeats:c}}var gn=`
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
`,_n=4,vn=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;fireBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.fireBuffer?.destroy(),this.fireBuffer=e.createBuffer({label:`observatory-firewall-fire`,size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);let t=e.createShaderModule({label:`observatory-firewall-choreo`,code:gn});this.pipeline=e.createComputePipeline({label:`observatory-firewall-choreo`,layout:`auto`,compute:{module:t,entryPoint:`firewall_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-firewall-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){let t=this.engine.params[9]===_n,n=this.engine.params[12]===1;if(!t&&!n||!this.pipeline||!this.bindGroup)return;let r=this.nodeRenderer.nodeCountValue;if(r===0)return;let i=e.beginComputePass({label:`observatory-firewall-choreo`});i.setPipeline(this.pipeline),i.setBindGroup(0,this.bindGroup),i.dispatchWorkgroups(Math.ceil(r/64)),i.end()}dispose(){this.fireBuffer?.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}},yn=[`failure`,`guardrail`,`confusion`];Math.PI*2;function bn(e){let t=e.nodes.length;if(t===0)return-1;let n=new Uint32Array(t);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>e.nodes[t].tags.some(e=>yn.includes(e.toLowerCase())),i=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r(t),t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&n[t]<=1,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex];for(let n of i){let r=-1;for(let i=0;i<t;i++)n(i)&&(r<0||e.nodes[i].retention<e.nodes[r].retention)&&(r=i);if(r>=0)return r}return-1}function xn(e,t,n){let r=e[n*16+0],i=e[n*16+1],a=e[n*16+2],o=Array(t),s=0;for(let n=0;n<t;n++){let t=e[n*16+0]-r,c=e[n*16+1]-i,l=e[n*16+2]-a,u=Math.sqrt(t*t+c*c+l*l);o[n]=u,u>s&&(s=u)}s<1e-6&&(s=1);let c=Array(t);for(let e=0;e<t;e++)c[e]=Math.min(255,Math.max(0,Math.round(144*o[e]/s)));return c[n]=0,c}function Sn(e,t){let n=new Set;for(let r of e.edges)r.sourceIndex===t&&r.targetIndex!==t&&n.add(r.targetIndex),r.targetIndex===t&&r.sourceIndex!==t&&n.add(r.sourceIndex);return Array.from(n).sort((e,t)=>e-t).slice(0,6)}function Cn(e){return 345+21*e}var wn=4;function Tn(e){return En(e)}function En(e){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`threat quarantined`,intruderLabel:``,receipt:`memory held in review · Memory PR opened`}}}function Dn(e,t){return kn(e,t,bn(e))}function On(e,t,n){return n<0||n>=e.nodes.length?En(e.nodes.length):kn(e,t,n)}function kn(e,t,n){let r=e.nodes.length;if(r===0||n<0)return En(r);let i=xn(Kt(e,t),r,n),a=Sn(e,n),o=new Uint32Array(r);for(let e=0;e<r;e++)o[e]=i[e]&255;o[n]=256,a.forEach((e,t)=>{o[e]|=512|t<<10});let s=new Uint32Array(Math.max(1,a.length)*wn),c=[];a.forEach((t,r)=>{let i=Cn(r);s[r*wn+0]=n,s[r*wn+1]=t,s[r*wn+2]=i,s[r*wn+3]=R.probe,c.push({sourceIndex:n,targetIndex:t,beatFrame:i,kind:R.probe,beatKind:`sever`,nodeId:e.nodes[t].id,label:tn(e.nodes[t].label)})});let l=tn(e.nodes[n].label),u=[],d=(e,t,r)=>{u.push({sourceIndex:n,targetIndex:n,beatFrame:e,kind:1,beatKind:`firewall`,nodeId:r,label:t})};return d(90,`intrusion · ${l}`,e.nodes[n].id),d(150,`immune response · shockwave`,`firewall-shock`),d(330,`membrane forming`,`firewall-membrane`),a.forEach((t,n)=>{d(Cn(n),`edge severed ✗ · ${tn(e.nodes[t].label)}`,e.nodes[t].id)}),d(480,`threat quarantined`,`firewall-verdict`),{viable:!0,intruderIndex:n,severedNeighborIndices:a,shockDelays:i,fireData:o,pathData:s,pathMetas:c,spineBeats:u,verdict:{headline:`threat quarantined`,intruderLabel:l,receipt:`memory held in review · Memory PR opened`}}}var An=.1542;function jn(e=An){return .9**(-1/e)-1}function Mn(e,t,n=An){if(!(e>0))return 0;if(!(t>0))return 1;let r=(1+jn(n)*t/e)**+-n;return r<0?0:r>1?1:r}var Nn=864e5;function Pn(e,t,n=0){if(!e)return n>0?n:0;let r=Date.parse(e);if(!Number.isFinite(r))return n>0?n:0;let i=(t-r)/Nn;return Math.max(0,i)+Math.max(0,n)}function Fn(e,t,n,r,i=An){if(n){let e=Date.parse(n);if(Number.isFinite(e)&&r<e)return 0}if(e===void 0||!Number.isFinite(e)||!t)return 1;let a=Date.parse(t);return Number.isFinite(a)?Math.max(.001,Mn(e,(r-a)/Nn,i)):1}function In(e,t,n,r=0,i=An){return e===void 0||!Number.isFinite(e)?1:Mn(e,Pn(t,n,r),i)}var Ln={[F.firewall]:620,[F.dreamStorm]:360,[F.causalRecall]:260,[F.birth]:180},Rn=class{engine;renderer;graph;response;seed;projectionDays;chronoOffsetDays;onApply;onFirewall;firewall=null;liveEdges=[];liveEdgeKeys=new Set;edgesDirty=!1;indexById;active=null;dreamOpen=!1;retention;hasLiveDecay=!1;eventsSeen=0;lastDecayFrame=-1e3;constructor(e){this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;let t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let n=0;n<t;n++){let t=e.graph.nodes[n];this.retention[n]=t.retention,t.stability!==void 0&&t.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(let e of this.liveEdges)this.liveEdgeKeys.add(zn(e.sourceIndex,e.targetIndex));this.lastAppliedMs=0;let n=this.engine.params;n[I.liveKind]=F.none,n[I.liveFrame]=0,n[I.liveEnergy]=0,n[I.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}lastAppliedMs=0;seeded=!1;seedWatermark(e){let t=0;for(let n of e){let e=Bn(n);e>t&&(t=e)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,n){if(this.active!==null)return!1;let r=this.indexById.get(e);if(r===void 0||(this.retention[r]??0)<5e-4)return!1;let i=t.filter(t=>t!==e&&this.indexById.has(t));return this.arm({kind:F.causalRecall,startFrame:n,targetId:e,relatedIds:i,pairs:[],scalar:i.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let n=e.length-1;n>=0;n--){let r=e[n],i=Bn(r);i>this.lastAppliedMs&&(this.decodeAndArm(r,this.engine.totalFrames),i>t&&(t=i))}this.lastAppliedMs=t}decodeAndArm(e,t){let n=e.data??{};switch(e.type){case`MemorySuppressed`:{let e=Vn(n.id);if(!e||!this.indexById.has(e))return;this.arm({kind:F.firewall,startFrame:t,targetId:e,relatedIds:this.neighborsOf(e),pairs:[],scalar:Hn(n.estimated_cascade)});break}case`DeepReferenceCompleted`:{let e=Wn(n.contradiction_pairs).filter(([e,t])=>this.indexById.has(e)&&this.indexById.has(t));if(e.length>0){let n=e[0][0];this.arm({kind:F.firewall,startFrame:t,targetId:n,relatedIds:e.flatMap(e=>e).filter(e=>e!==n),pairs:e,scalar:e.length});return}let r=Vn(n.primary_id),i=Un(n.supporting_ids).filter(e=>this.indexById.has(e));r&&this.indexById.has(r)&&this.arm({kind:F.causalRecall,startFrame:t,targetId:r,relatedIds:i,pairs:[],scalar:Hn(n.confidence)});break}case`BackfillFired`:case`CausalReceipt`:{let e=Un(n.path_ids??n.causal_path),r=Vn(n.failure_id??n.target_id??n.effect_id)||e.at(-1)||e[0];r&&this.indexById.has(r)&&this.arm({kind:F.causalRecall,startFrame:t,targetId:r,relatedIds:e.filter(e=>e!==r),exactPath:e,pairs:[],scalar:e.length});break}case`DreamStarted`:this.dreamOpen=!0,this.arm({kind:F.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:Hn(n.memory_count)});break;case`DreamCompleted`:{this.dreamOpen=!1;let e=Hn(n.connections_found);this.active&&this.active.kind===F.dreamStorm?this.active.scalar=Math.max(this.active.scalar,e):this.arm({kind:F.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:e});break}case`ConnectionDiscovered`:{let e=this.indexById.get(Vn(n.source_id)),t=this.indexById.get(Vn(n.target_id));if(e===void 0||t===void 0||e===t)break;let r=zn(e,t);if(this.liveEdgeKeys.has(r))break;this.liveEdgeKeys.add(r),this.liveEdges.push({sourceIndex:e,targetIndex:t,weight:Hn(n.weight)||.5,type:Vn(n.connection_type)||`semantic`}),this.edgesDirty=!0,this.dreamOpen&&this.active?.kind===F.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){if(this.active=e,this.eventsSeen++,e.kind===F.firewall){let t=this.indexById.get(e.targetId);if(t===void 0)return;let n=On(this.graph,this.seed,t);if(!n.viable)return;this.firewall||=new vn({engine:this.engine,nodeRenderer:this.renderer,plan:Tn(this.graph.nodes.length)}),this.firewall.rearm(n),this.onFirewall?.({intruderLabel:n.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===F.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){let t=e.exactPath;if(t.some(e=>!this.indexById.has(e)))return;let n=new Uint32Array(Math.max(1,t.length-1)*4),r=[];for(let i=0;i<t.length-1;i++){let a=this.indexById.get(t[i]),o=this.indexById.get(t[i+1]),s=e.startFrame+24+i*42;n[i*4]=a,n[i*4+1]=o,n[i*4+2]=s,n[i*4+3]=R.backwardCause,r.push({sourceIndex:a,targetIndex:o,beatFrame:s,kind:R.backwardCause,beatKind:`receipt-path`,nodeId:t[i+1],label:`receipt-backed candidate path`})}this.renderer.setPathSteps(n,r);return}let t=B(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});t.steps.length>0&&this.renderer.setPathSteps(t.data,t.steps)}}neighborsOf(e){let t=this.indexById.get(e);if(t===void 0)return[];let n=[];for(let e of this.graph.edges)if(e.sourceIndex===t?n.push(this.graph.nodes[e.targetIndex].id):e.targetIndex===t&&n.push(this.graph.nodes[e.sourceIndex].id),n.length>=12)break;return n}drain(e){let t=this.engine.params;this.edgesDirty&&=(this.renderer.setEdges(this.liveEdges),!1);let n=this.projectionDays(),r=this.chronoOffsetDays();if(t[I.projectionDays]=Math.max(0,n),(this.hasLiveDecay||r!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||n!==this.lastProj||r!==this.lastChrono)&&(this.recomputeDecay(n,r),this.lastDecayFrame=e,this.lastProj=n,this.lastChrono=r),this.active){let n=Ln[this.active.kind]??300,r=e-this.active.startFrame;r>n+140?(this.active=null,t[I.liveKind]=F.none,t[I.liveEnergy]=0):(t[I.liveKind]=this.active.kind,t[I.liveFrame]=Math.max(0,r),t[I.liveEnergy]=this.energyEnvelope(this.active,r,!1))}else t[I.liveKind]=F.none,t[I.liveEnergy]=0;this.onApply?.({simFrame:e,activeKind:t[I.liveKind],eventsSeen:this.eventsSeen})}debugState(){let e=this.engine.params;return{activeKind:e[I.liveKind],liveEnergy:e[I.liveEnergy],liveFrame:e[I.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}lastProj=-1;lastChrono=0;energyEnvelope(e,t,n){if(t<0)return 0;let r=Ln[e.kind]??300;if(e.kind===F.dreamStorm){let n=Math.min(1,t/45),i=1-Math.max(0,(t-(r-90))/90),a=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,n*Math.min(1,i)*a)}let i=Math.min(1,t/24),a=1-Math.max(0,(t-r)/140);return Math.max(0,i*Math.min(1,a))}recomputeDecay(e,t=0){let n=this.engine.wallNowMs,r=this.graph.nodes;if(t!==0){let i=n+(t+Math.max(0,e))*Nn;for(let e=0;e<r.length;e++){let t=r[e];this.retention[e]=t.stability!==void 0||t.createdAt?Fn(t.stability,t.lastAccessed,t.createdAt,i):Math.max(.001,t.retention)}}else for(let t=0;t<r.length;t++){let i=r[t];this.retention[t]=i.stability!==void 0&&i.lastAccessed?In(i.stability,i.lastAccessed,n,e):Math.max(.001,i.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){let e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}};function zn(e,t){return e<t?`${e}-${t}`:`${t}-${e}`}function Bn(e){let t=e.data?.timestamp;if(typeof t!=`string`)return 0;let n=Date.parse(t);return Number.isFinite(n)?n:0}function Vn(e){return typeof e==`string`?e:``}function Hn(e){return typeof e==`number`&&Number.isFinite(e)?e:0}function Un(e){return Array.isArray(e)?e.filter(e=>typeof e==`string`):[]}function Wn(e){if(!Array.isArray(e))return[];let t=[];for(let n of e)Array.isArray(n)&&n.length>=2&&typeof n[0]==`string`&&typeof n[1]==`string`&&t.push([n[0],n[1]]);return t}var Gn=512,Kn=4,qn=`
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
`;function Jn(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function Yn(e){if(!e)return null;let t=Date.parse(e);return Number.isFinite(t)?t:null}var Xn=class{engine;resources=null;bindLayout=null;railPipeline=null;dwellPipeline=null;headPipeline=null;dwellCount=0;minMs=0;maxMs=0;state={scrub:1,days:0,density:0,active:0};constructor(e,t){this.engine=e,this.upload(t)}setTimeline(e,t=!1){let n=this.engine.wallNowMs,r=Math.max(1,this.maxMs-this.minMs);this.state.scrub=Jn((n+e*864e5-this.minMs)/r),this.state.days=Number.isFinite(e)?e:0,this.state.active=+!!t,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){this.resources&&this.railPipeline&&this.dwellPipeline&&this.headPipeline&&(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){this.resources?.dwellBuffer.destroy(),this.resources?.stateBuffer.destroy(),this.resources=null}upload(e){let t=e.flatMap(e=>[Yn(e.createdAt),Yn(e.lastAccessed)]).filter(e=>e!==null),n=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):n-864e5,this.maxMs=Math.max(n+31536e6,this.minMs+864e5);let r=this.maxMs-this.minMs,i=[];for(let t of e){let e=Yn(t.createdAt),n=Yn(t.lastAccessed),r=Jn(t.retention);e!==null&&i.push({at:e,kind:0,retention:r,suppressed:+!!t.suppressed}),n!==null&&n!==e&&i.push({at:n,kind:1,retention:r,suppressed:+!!t.suppressed})}i.sort((e,t)=>e.at-t.at);let a=Math.max(1,Math.ceil(i.length/Gn)),o=i.filter((e,t)=>t%a===0).slice(0,Gn);this.dwellCount=o.length,this.state={scrub:Jn((n-this.minMs)/r),days:0,density:Jn(o.length/96),active:0};let s=this.engine.gpuDevice;if(!s||!this.engine.paramsBuffer||(this.ensurePipelines(s),this.ensureResources(s),!this.resources))return;let c=new Float32Array(Gn*Kn);o.forEach((e,t)=>{c.set([Jn((e.at-this.minMs)/r),e.kind,e.retention,e.suppressed],t*Kn)}),s.queue.writeBuffer(this.resources.dwellBuffer,0,c),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`fossil-light-chrono-shuttle-wgsl`,code:qn});this.bindLayout=e.createBindGroupLayout({label:`fossil-light-chrono-shuttle-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let n=e.createPipelineLayout({label:`fossil-light-chrono-shuttle-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`src-alpha`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}},i=(i,a,o)=>e.createRenderPipeline({label:i,layout:n,vertex:{module:t,entryPoint:a},fragment:{module:t,entryPoint:o,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`}});this.railPipeline=i(`fossil-light-chrono-rail`,`vs_rail`,`fs_rail`),this.dwellPipeline=i(`fossil-light-chrono-dwells`,`vs_dwell`,`fs_dwell`),this.headPipeline=i(`fossil-light-chrono-head`,`vs_head`,`fs_head`)}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;let t=e.createBuffer({label:`fossil-light-chrono-dwell-events`,size:Gn*Kn*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:`fossil-light-chrono-state`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:n,bindGroup:e.createBindGroup({label:`fossil-light-chrono-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}}]})}}writeState(){let e=this.engine.gpuDevice;e&&this.resources&&e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active]))}},Zn=64,Qn=32,$n=4,er=256,tr=`rgba8unorm`,nr=96e3,rr=5,ir=`
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

const MAX_EMITTERS = ${Zn}u;

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
`;function ar(e,t){return Number.isFinite(e)?e:t}var or=class{engine;renderer;sourceIndices;resources=null;projectionPipeline=null;seedPipeline=null;transportPipeline=null;compositePipeline=null;projectionLayout=null;seedLayout=null;transportLayout=null;compositeLayout=null;emitterCount;active=!1;dirty=!0;lastComputedFrame=-5;disposed=!1;disabledReason=null;exposure=.42;configBytes=new ArrayBuffer(Qn);configUints=new Uint32Array(this.configBytes);configFloats=new Float32Array(this.configBytes);constructor(e,t,n){this.engine=e,this.renderer=t;let r=[...new Set([...n].filter(e=>Number.isFinite(e)&&e>=0))].sort((e,t)=>e-t).slice(0,Zn);this.sourceIndices=new Uint32Array(r),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?`half-res-transport`:`disabled`}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,ar(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;let n=this.engine.gpuDevice;if(!n||!this.engine.paramsBuffer)return;let r=this.renderer.getFossilLightSources();if(!r)return;let i=this.fieldDimensions();if(i===null)return;let a=t-this.lastComputedFrame;if(!(this.active||this.dirty||a<0||a>=rr))return;try{this.ensurePipelines(n),this.ensureResources(n,i.width,i.height,r)}catch{this.disable(`GPU light field unavailable on this adapter`);return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(n,0,this.resources.width,this.resources.height,0);let o=Math.ceil(this.resources.width/8),s=Math.ceil(this.resources.height/8),c=e.beginComputePass({label:`fossil-light-half-res-transport`});c.setPipeline(this.projectionPipeline),c.setBindGroup(3,this.resources.projectionBindGroup,[0]),c.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),c.setPipeline(this.seedPipeline),c.setBindGroup(0,this.resources.seedBindGroup,[0]),c.dispatchWorkgroups(o,s);for(let[e,t,r]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(n,e,this.resources.width,this.resources.height,t),c.setPipeline(this.transportPipeline),c.setBindGroup(1,r,[e*er]),c.dispatchWorkgroups(o,s);c.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason===null&&this.resources&&this.compositePipeline&&this.emitterCount!==0&&(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*er]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){let e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;let n=e*.5*(t*.5),r=.5*Math.min(1,Math.sqrt(nr/Math.max(1,n)));return{width:Math.max(1,Math.floor(e*r)),height:Math.max(1,Math.floor(t*r))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;let t=e.createShaderModule({label:`fossil-light-radiance-cascade-wgsl`,code:ir}),n=e.createBindGroupLayout({label:`fossil-light-empty-layout`,entries:[]});this.projectionLayout=e.createBindGroupLayout({label:`fossil-light-source-projection-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:Qn}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:`storage`}}]}),this.seedLayout=e.createBindGroupLayout({label:`fossil-light-seed-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:Qn}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:tr}}]}),this.transportLayout=e.createBindGroupLayout({label:`fossil-light-transport-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:Qn}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:tr}}]}),this.compositeLayout=e.createBindGroupLayout({label:`fossil-light-composite-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:Qn}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}}]}),this.seedPipeline=e.createComputePipeline({label:`fossil-light-seed`,layout:e.createPipelineLayout({label:`fossil-light-seed-pipeline-layout`,bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:`cs_seed`}}),this.projectionPipeline=e.createComputePipeline({label:`fossil-light-source-projection`,layout:e.createPipelineLayout({label:`fossil-light-source-projection-pipeline-layout`,bindGroupLayouts:[n,n,n,this.projectionLayout]}),compute:{module:t,entryPoint:`cs_project_sources`}}),this.transportPipeline=e.createComputePipeline({label:`fossil-light-transport`,layout:e.createPipelineLayout({label:`fossil-light-transport-pipeline-layout`,bindGroupLayouts:[n,this.transportLayout]}),compute:{module:t,entryPoint:`cs_transport`}}),this.compositePipeline=e.createRenderPipeline({label:`fossil-light-composite`,layout:e.createPipelineLayout({label:`fossil-light-composite-pipeline-layout`,bindGroupLayouts:[n,n,this.compositeLayout]}),vertex:{module:t,entryPoint:`vs_composite`},fragment:{module:t,entryPoint:`fs_composite`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`src-alpha`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]}})}ensureResources(e,t,n,r){if(this.resources?.width===t&&this.resources.height===n&&this.resources.nodeBuffer===r.nodeBuffer&&this.resources.cameraBuffer===r.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;let i=e.createBuffer({label:`fossil-light-projected-memory-emitters`,size:768*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),a=e.createBuffer({label:`fossil-light-source-indices`,size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(a,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);let o=e.createBuffer({label:`fossil-light-cascade-config`,size:er*$n,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),s=r=>e.createTexture({label:r,size:[t,n],format:tr,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),c=s(`fossil-light-field-a`),l=s(`fossil-light-field-b`),u=c.createView(),d=l.createView();this.resources={width:t,height:n,emitterBuffer:i,sourceIndexBuffer:a,configBuffer:o,fieldA:c,fieldB:l,seedBindGroup:e.createBindGroup({label:`fossil-light-seed-bind-group`,layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:Qn}},{binding:1,resource:{buffer:i}},{binding:2,resource:u}]}),propagateABindGroup:e.createBindGroup({label:`fossil-light-transport-a-to-b`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:Qn}},{binding:1,resource:u},{binding:2,resource:d}]}),propagateBBindGroup:e.createBindGroup({label:`fossil-light-transport-b-to-a`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:Qn}},{binding:1,resource:d},{binding:2,resource:u}]}),projectionBindGroup:e.createBindGroup({label:`fossil-light-source-projection-bind-group`,layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:Qn}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:r.nodeBuffer}},{binding:3,resource:{buffer:r.cameraBuffer}},{binding:4,resource:{buffer:i}}]}),compositeBindGroup:e.createBindGroup({label:`fossil-light-composite-bind-group`,layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:Qn}},{binding:1,resource:d}]}),nodeBuffer:r.nodeBuffer,cameraBuffer:r.cameraBuffer},this.dirty=!0}writeConfig(e,t,n,r,i){this.resources&&(this.configUints[0]=n,this.configUints[1]=r,this.configUints[2]=this.emitterCount,this.configUints[3]=i,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*er,this.configBytes))}destroyResources(){this.resources?.emitterBuffer.destroy(),this.resources?.sourceIndexBuffer.destroy(),this.resources?.configBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}},sr=d(`<div class="flex items-baseline gap-2 font-mono text-[11px]"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></div>`),cr=d(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),lr=d(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),ur=d(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"> </button>`),dr=d(`<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>`),fr=d(`<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>`),pr=d(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),mr=d(`<button> </button>`),hr=d(`<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>`),gr=d(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>`),_r=d(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>`),vr=d(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>`),yr=d(`<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>`),br=d(`<div class="pointer-events-none fixed left-4 bottom-24 z-30 font-mono text-[10px] tracking-widest text-[#7ff3e6]/80"> </div>`),xr=d(`<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div> <!> <!>`,1);function Sr(t,r){ae(r,!0);let i=()=>T(pe,`$eventFeed`,d),[d,h]=e(),x=S(r,`seed`,3,`vestige-observatory-v1`),te=S(r,`freezeFrame`,3,null),D=S(r,`capture`,3,!1),ie=S(r,`showSwitcher`,3,!0),oe=S(r,`embedded`,3,!1),O=S(r,`chrome`,3,`none`),ue=S(r,`maxDpr`,3,2),de=S(r,`focusIds`,19,()=>[]),fe=S(r,`live`,3,!1),me=S(r,`graphOverride`,3,null),M=b(0),N=b(!1),P=b(0),ge=null,_e=p(()=>Math.max(0,f(M))),ve=p(()=>Math.min(0,f(M))),ye=p(()=>f(M)===0?`now`:f(M)>0?`+${Math.round(f(M))}d`:new Date(Date.now()+f(M)*864e5).toLocaleDateString(void 0,{month:`short`,day:`numeric`})),F=null,be=null,Se=null,we=b(!1),Te=b(!1),Ee=!1,I=0,L=0,R=0,Oe=!1;function ke(e){let t=f(Ye)?.getBoundingClientRect();if(!t||t.width===0)return f(M);let n=(e-t.left)/t.width*2-1,r=Math.max(0,Math.min(1,n/.835*.5+.5));return f(P)+r*(365-f(P))}function Ae(e){let t=f(Ye)?.getBoundingClientRect();if(!t||t.height===0)return!1;let n=(e.clientY-t.top)/t.height;return n>.7675000000000001&&n<.9175}function je(){I&&cancelAnimationFrame(I),I=0}function Me(e){f(we)&&!D()&&Ae(e)&&(je(),Ee=!0,v(N,!0),L=0,R=performance.now(),v(M,ke(e.clientX),!0),e.currentTarget.setPointerCapture?.(e.pointerId),e.preventDefault())}function Ne(e){if(!Ee)return;let t=performance.now(),n=ke(e.clientX),r=Math.max(1,t-R);L=L*.6+(n-f(M))/r*16*.4,R=t,v(M,n,!0)}function Pe(e){if(!Ee)return;Ee=!1,Oe=!0,e.currentTarget.releasePointerCapture?.(e.pointerId);let t=()=>{I=0,L*=.94;let e=f(M)+L;e<=f(P)&&(e=f(P),L=0),e>=365&&(e=365,L=0);let n=f(M)<0&&L>0||f(M)>0&&L<0;Math.abs(e)<1&&n&&(e=0,L=0),v(M,e,!0),Math.abs(L)>.02?I=requestAnimationFrame(t):v(N,!1)};Math.abs(L)>.05?I=requestAnimationFrame(t):v(N,!1)}function ze(){Ee=!1,v(N,!1),je()}let z=b(!1),Be=b(!1);function Ve(){if(typeof window>`u`)return;let e=window.matchMedia(`(prefers-reduced-motion: reduce)`);e.matches&&!f(Be)&&v(z,!0);let t=e=>{f(Be)||v(z,e.matches,!0)};return e.addEventListener(`change`,t),()=>e.removeEventListener(`change`,t)}function Ue(){v(Be,!0),v(z,!f(z))}g(()=>{f(B)?.setPaused(f(z))});let We=b(``),Ge=b(0),qe=p(()=>f(We)!==``&&f(Ge)>0),Ye=b(null),Xe=new st,Ze=b(null),Qe=0,$e=b(``);async function et(e){if(Oe){Oe=!1;return}if(!V||!f(Ye))return;let t=f(Ye).getBoundingClientRect();if(t.width===0||t.height===0)return;let n=(e.clientX-t.left)/t.width*2-1,i=-((e.clientY-t.top)/t.height*2-1),a=await V.pickAt(n,i);a&&(v(Ze,{kind:`memory`,id:a.id,label:`Field cell`},!0),r.onpick?.(a.id))}function tt(e){Me(e),!(Ee||D())&&(Xe.enabled=!D(),Xe.onPointerDown(e))}function nt(e){if(Ne(e),Ee||D())return;Xe.onPointerMove(e)&&(Oe=!0,V?.setCameraRig(Xe.state));let t=performance.now();if(t-Qe<120||!V||!f(Ye))return;Qe=t;let n=f(Ye).getBoundingClientRect();if(n.width===0)return;let r=(e.clientX-n.left)/n.width*2-1,i=-((e.clientY-n.top)/n.height*2-1);V.pickAt(r,i).then(e=>{V?.setHovered(e?.index??-1),v($e,e?.id?.slice(0,8)??``,!0),f(Ye)&&(f(Ye).style.cursor=e?`crosshair`:`grab`)})}function rt(e){Pe(e),Xe.onPointerUp(e)}function it(){ze(),Xe.onPointerUp({pointerId:-1})}function at(e){D()||Ae(e)||Xe.onWheel(e)&&V?.setCameraRig(Xe.state)}let ot={"recall-path":`RECALL`,"engram-birth":`BIRTH`,"salience-rescue":`RESCUE`,"forgetting-horizon":`HORIZON`,firewall:`FIREWALL`};function ct(){return V?.graph?new Uint32Array(V.graph.nodes.map(e=>({index:e.index,id:e.id,retention:e.retention})).sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,64).map(e=>e.index).sort((e,t)=>e-t)):new Uint32Array}let lt=b(!D());function ut(e){let t=e.target;t?.isContentEditable||t?.tagName===`INPUT`||t?.tagName===`TEXTAREA`||t?.tagName===`SELECT`||((e.key===`h`||e.key===`H`)&&v(lt,!f(lt)),e.key===`Escape`&&r.onexit&&r.onexit(),(e.key===` `||e.key.toLowerCase()===`p`)&&!D()&&(e.preventDefault(),Ue()))}let dt=b(null),ft=b(!0),pt=b(``),mt=b(0),ht=b(0),gt=b(0),_t=b(0),vt=b(``),B=b(null),V=null,yt=null,bt=null,St=b(null),Ct=null,wt=null,Tt=b(null),Et=!1,Dt=b(y([]));async function Ot(){v(ft,!0),v(pt,``);try{if(me()){v(dt,me()),v(gt,me().nodeCount,!0),v(_t,me().edgeCount,!0),v(vt,me().center_id,!0);return}let e=new Set(de().filter(Boolean)),t=e.size?await(async()=>{let t=await Promise.all([...e].map(e=>he.graph({center_id:e,max_nodes:200,depth:3}))),n=[...new Map(t.flatMap(e=>e.nodes).map(e=>[e.id,e])).values()].filter(t=>e.has(t.id)),r=new Set(n.map(e=>e.id)),i=[...new Map(t.flatMap(e=>e.edges).map(e=>[`${e.source}:${e.target}`,e])).values()].filter(e=>r.has(e.source)&&r.has(e.target));return{...t[0],nodes:n,edges:i,center_id:n[0]?.id??t[0]?.center_id??``,nodeCount:n.length,edgeCount:i.length}})():await he.graph({max_nodes:200,depth:3,sort:`connected`});v(dt,t,!0),v(gt,t.nodeCount,!0),v(_t,t.edgeCount,!0),v(vt,t.center_id,!0)}catch(e){let t=e instanceof Error?e.message:`Failed to load graph data`;/\b404\b/.test(t)?(v(dt,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:``},!0),v(gt,0),v(_t,0),v(vt,``)):v(pt,t,!0)}finally{v(ft,!1)}}let kt=null,At=b(y([])),jt=b(`recalls`);function Mt(e,t){v(mt,e,!0),v(ht,t,!0),kt&&!f(N)&&kt.tick(e)}async function Nt(){if(!F||!V?.graph)return;let e=V.graph,t=t=>e.indexById.has(t),n=t=>e.nodes[e.indexById.get(t)??-1]?.label??t.slice(0,8),r=[];try{r=(await he.receipts.list(60))?.receipts??[]}catch{}let i=Fe(r,t);i.length===0&&(i=Ie(e.nodes,12)),i.length>0&&(kt=new Re(F,{intervalFrames:240}),kt.setItems(i));let a=Le(r,t,3);a.length>0?(v(jt,`recalls`),v(At,a.map(e=>({...e,label:n(e.id)})),!0)):(v(jt,`retention`),v(At,[...e.nodes].filter(e=>(e.label??``).trim().length>0).sort((e,t)=>t.retention-e.retention).slice(0,3).map(e=>({id:e.id,recalls:Math.round(e.retention*100),label:e.label||e.id.slice(0,8)})),!0))}function Pt(e){Et=!1,V?.dispose(),v(B,e,!0),V=new xt(e),Xe.enabled=!D(),D()&&Xe.reset(),V.setCameraRig(Xe.state),r.onready?.(e)}g(()=>{if(f(B)&&V&&f(dt)&&!Et){Et=!0;let e=r.demo===`engram-birth`,t=r.demo===`salience-rescue`,n=r.demo===`forgetting-horizon`,i=r.demo===`firewall`;if(V.upload(f(dt),x(),{recallPath:!e&&!t&&!n&&!i}),e){yt=new Bt({engine:f(B),nodeRenderer:V,seed:x()}),yt.upload(x());let e=yt.engraveSteps,t=[];for(let n=0;n<e.length/4;n++)t.push({sourceIndex:e[n*4],targetIndex:e[n*4+1],beatFrame:e[n*4+2],kind:e[n*4+3],beatKind:`engrave`,nodeId:`engrave-${n}`,label:`edge engraved`});V.setPathSteps(e,t),v(Dt,yt.timeline.map((e,t)=>({sourceIndex:0,targetIndex:0,beatFrame:e.startFrame,kind:0,beatKind:`birth`,nodeId:`birth-${t}`,label:e.label})),!0)}else if(t){let e=an(f(dt),V.graph,x(),r.backfillEvidence);v(St,e,!0),e.viable&&(bt=new Ut({engine:f(B),nodeRenderer:V,plan:e}),bt.upload(),V.setPathSteps(e.pathData,e.pathMetas)),v(Dt,e.spineBeats,!0)}else if(n){let e=hn(V.graph);e.viable&&(Ct=new ln({engine:f(B),nodeRenderer:V,plan:e}),Ct.upload(),V.setPathSteps(e.pathData,e.pathMetas)),v(Dt,e.spineBeats,!0)}else if(i){let e=Dn(V.graph,x());v(Tt,e,!0),e.viable&&(wt=new vn({engine:f(B),nodeRenderer:V,plan:e}),wt.upload(),V.setPathSteps(e.pathData,e.pathMetas)),v(Dt,e.spineBeats,!0)}else v(Dt,V.pathSteps,!0);if(fe()&&V.graph&&f(dt)){F=new Rn({engine:f(B),renderer:V,graph:V.graph,response:f(dt),seed:x(),projectionDays:()=>f(_e),chronoOffsetDays:()=>f(ve),onFirewall:e=>{v(We,e.intruderLabel,!0),v(Ge,Date.now(),!0)}}),v(Te,F.liveDecayAvailable,!0),f(B).setPreFrameHook(e=>F?.drain(e)),D()||Nt();let e=1/0;for(let t of V.graph.nodes)if(t.createdAt){let n=Date.parse(t.createdAt);Number.isFinite(n)&&n<e&&(e=n)}if(Number.isFinite(e)&&v(P,Math.floor((e-Date.now())/864e5)-1),ge){let e=Date.parse(ge);Number.isFinite(e)&&v(M,Math.min(365,Math.max(f(P),(e-Date.now())/864e5)),!0),ge=null}D()||(Se=new or(f(B),V,ct()),f(B).addPass(Se),be=new Xn(f(B),V.graph.nodes),f(B).addPass(be),v(we,!0)),typeof window<`u`&&(window.__vestigeLiveBridge=F)}f(B).demoClock.reset()}}),g(()=>{let e=i();F&&F.ingest(e)}),g(()=>{f(M),F?.refreshDecay(),be?.setTimeline(f(M),f(N)),Se?.setScrubbing(f(N))}),g(()=>{if(!f(Ge))return;let e=setTimeout(()=>{v(We,``),v(Ge,0)},7e3);return()=>clearTimeout(e)}),se(()=>{ge=new URLSearchParams(window.location.search).get(`t`),Ot();let e=Ve();return()=>{if(je(),e?.(),V?.dispose(),V=null,typeof window<`u`){let e=window;e.__vestigeLiveBridge===F&&delete e.__vestigeLiveBridge}}});var Ft=xr();u(`keydown`,_,ut);var It=c(Ft);let Lt;var Rt=o(It);let zt;var Vt=o(Rt);Ce(Vt,{get demo(){return r.demo},get seed(){return x()},get freezeFrame(){return te()},get maxDpr(){return ue()},onframe:Mt,onready:Pt}),A(Rt),ee(Rt,e=>v(Ye,e),()=>f(Ye));var Ht=s(Rt,2),Wt=e=>{var t=yr(),i=o(t),c=e=>{var t=cr(),r=o(t),i=k(r,!0),c=s(r,2);le(c,19,()=>f(At),e=>e.id,(e,t,r)=>{var i=sr(),c=o(i),l=k(c,!0),u=s(c,2),d=k(u,!0),p=s(u,2),h=k(p);A(i),n(()=>{m(l,f(r)+1),E(u,`title`,f(t).label),m(d,f(t).label),m(h,`${f(t).recalls??``}${f(jt)===`recalls`?`×`:`%`}`)}),a(e,i)}),A(t),n(()=>m(i,f(jt)===`recalls`?`Most recalled · your mind`:`Strongest memories · your mind`)),a(e,t)};j(i,e=>{fe()&&f(At).length>0&&e(c)});var d=s(i,2),h=e=>{var t=lr(),r=s(o(t),2),i=k(r,!0);ne(2),A(t),n(()=>m(i,f(We))),a(e,t)};j(d,e=>{fe()&&f(qe)&&e(h)});var g=s(d,2),_=e=>{var t=ur(),r=k(t,!0);n(()=>{E(t,`title`,f(z)?`Resume field motion`:`Pause field motion`),E(t,`aria-pressed`,f(z)),E(t,`aria-label`,f(z)?`Resume 3D memory field motion`:`Pause 3D memory field motion`),m(r,f(z)?`▶ RESUME`:`❚❚ PAUSE`)}),l(`click`,t,Ue),a(e,t)};j(g,e=>{D()||e(_)});var y=s(g,2),b=e=>{var t=fr();let r;var i=s(o(t),2);w(i);var c=s(i,2);let d;var p=k(c,!0),h=s(c,2),g=e=>{var t=dr();l(`click`,t,()=>v(M,0)),a(e,t)};j(h,e=>{f(M)!==0&&e(g)}),A(t),n(()=>{r=re(t,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]`,null,r,{"opacity-100":f(we),"opacity-75":!f(we)}),E(i,`min`,f(P)),d=re(c,1,`w-16 text-right tabular-nums`,null,d,{"text-[#b9d9a9]":f(M)>=0,"text-[#dfc68e]":f(M)<0}),m(p,f(ye))}),l(`input`,i,()=>v(N,!0)),l(`change`,i,()=>v(N,!1)),l(`pointerup`,i,()=>v(N,!1)),u(`pointercancel`,i,()=>v(N,!1)),u(`blur`,i,()=>v(N,!1)),ce(i,()=>f(M),e=>v(M,e)),a(e,t)};j(y,e=>{fe()&&f(Te)&&e(b)});var S=s(y,2),C=e=>{He(e,{get demoMode(){return r.demo},get seed(){return x()},get nodeCount(){return f(gt)},get edgeCount(){return f(_t)},get centerId(){return f(vt)},get frameCount(){return f(mt)},get fpsEstimate(){return f(ht)},get freezeFrame(){return te()},get loading(){return f(ft)},get error(){return f(pt)}})};j(S,e=>{O()===`full`&&e(C)});var ee=s(S,2),T=e=>{var t=pr();l(`click`,t,function(...e){r.onexit?.apply(this,e)}),a(e,t)};j(ee,e=>{O()===`full`&&r.onexit&&e(T)});var ae=s(ee,2),oe=e=>{var t=hr();le(t,20,()=>xe,e=>e,(e,t)=>{var i=mr(),o=k(i,!0);n(()=>{re(i,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${t===r.demo?`text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]`:`text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50`}`),E(i,`title`,`Play the ${ot[t]??``} moment`),m(o,ot[t])}),l(`click`,i,()=>r.ondemochange?.(t)),a(e,i)}),A(t),a(e,t)};j(ae,e=>{O()===`full`&&ie()&&e(oe)});var se=s(ae,2),ue=e=>{var t=gr();a(e,t)};j(se,e=>{f(ft)&&e(ue)});var de=s(se,2),pe=e=>{var t=_r(),r=o(t),i=k(r,!0);A(t),n(()=>m(i,f(pt))),a(e,t)};j(de,e=>{f(pt)&&!f(ft)&&e(pe)});var me=s(de,2),he=e=>{Ke(e,{get steps(){return f(Dt)},get frame(){return f(mt)}})};j(me,e=>{O()===`full`&&e(he)});var ge=s(me,2),_e=e=>{Je(e,{get frame(){return f(mt)},get verdict(){return f(St).verdict}})};j(ge,e=>{O()===`full`&&r.demo===`salience-rescue`&&f(St)?.viable&&e(_e)});var ve=s(ge,2),F=e=>{{let t=p(()=>({headline:f(Tt).verdict.headline,causeLabel:f(Tt).verdict.intruderLabel,receipt:f(Tt).verdict.receipt}));Je(e,{get frame(){return f(mt)},tone:`quarantine`,fadeWindow:[480,495,605,620],get verdict(){return f(t)}})}};j(ve,e=>{O()===`full`&&r.demo===`firewall`&&f(Tt)?.viable&&e(F)});var be=s(ve,2),Se=e=>{var t=vr();a(e,t)};j(be,e=>{!f(ft)&&f(dt)&&f(dt).nodeCount===0&&e(Se)}),A(t),a(e,t)};j(Ht,e=>{f(lt)&&e(Wt)}),A(It);var Gt=s(It,2);De(Gt,{get pick(){return f(Ze)},onclose:()=>v(Ze,null)});var Kt=s(Gt,2),qt=e=>{var t=br(),r=k(t);n(()=>m(r,`HOVER ${f($e)??``}`)),a(e,t)};j(Kt,e=>{f($e)&&!D()&&e(qt)}),n(()=>{Lt=re(It,1,`${oe()?`absolute`:`fixed`} inset-0 overflow-hidden bg-[#05060a]`,null,Lt,{"cursor-none":D()}),zt=re(Rt,1,`absolute inset-0 z-0 touch-none`,null,zt,{"cursor-crosshair":!!r.onpick&&!D()})}),l(`click`,Rt,et),l(`pointerdown`,Rt,tt),l(`pointermove`,Rt,nt),l(`pointerup`,Rt,rt),u(`pointercancel`,Rt,it),u(`wheel`,Rt,at),a(t,Ft),C(),h()}D([`click`,`pointerdown`,`pointermove`,`pointerup`,`input`,`change`]);function H(e){if(!e)throw Error(`Assertion failed.`)}var Cr=e=>{let t=(e%360+360)%360;if(t===0||t===90||t===180||t===270)return t;throw Error(`Invalid rotation ${e}.`)},wr=e=>e&&e[e.length-1],Tr=e=>e>=0&&e<2**32,U=e=>{let t=0;for(;e.readBits(1)===0&&t<32;)t++;if(t>=32)throw Error(`Invalid exponential-Golomb code.`);return(1<<t)-1+e.readBits(t)},Er=e=>{let t=U(e);return t&1?t+1>>1:-(t>>1)},Dr=e=>e.constructor===Uint8Array?e:ArrayBuffer.isView(e)?new Uint8Array(e.buffer,e.byteOffset,e.byteLength):new Uint8Array(e),Or=e=>e.constructor===DataView?e:ArrayBuffer.isView(e)?new DataView(e.buffer,e.byteOffset,e.byteLength):new DataView(e),kr=new TextEncoder,Ar={bt709:1,bt470bg:5,smpte170m:6,bt2020:9,smpte432:12},jr={bt709:1,smpte170m:6,linear:8,"iec61966-2-1":13,pq:16,hlg:18},Mr={rgb:0,bt709:1,bt470bg:5,smpte170m:6,"bt2020-ncl":9},Nr=e=>!e||e.primaries==null&&e.transfer==null&&e.matrix==null&&e.fullRange==null,Pr=e=>e instanceof ArrayBuffer||typeof SharedArrayBuffer<`u`&&e instanceof SharedArrayBuffer||ArrayBuffer.isView(e),Fr=class{constructor(){this.currentPromise=Promise.resolve(),this.pending=0}async acquire(){let e,t=new Promise(t=>{let n=!1;e=()=>{n||=(t(),this.pending--,!0)}}),n=this.currentPromise;return this.currentPromise=t,this.pending++,await n,e}},Ir=(e,t,n)=>{let r=0,i=e.length-1,a=-1;for(;r<=i;){let o=r+(i-r+1)/2|0;n(e[o])<=t?(a=o,r=o+1):i=o-1}return a},Lr=()=>{let e,t;return{promise:new Promise((n,r)=>{e=n,t=r}),resolve:e,reject:t}},Rr=e=>{throw Error(`Unexpected value: ${e}`)},zr=(e,t,n)=>{let r=e.getUint8(t),i=e.getUint8(t+1),a=e.getUint8(t+2);return n?r|i<<8|a<<16:r<<16|i<<8|a},Br=(e,t,n,r)=>{n>>>=0,n&=16777215,r?(e.setUint8(t,n&255),e.setUint8(t+1,n>>>8&255),e.setUint8(t+2,n>>>16&255)):(e.setUint8(t,n>>>16&255),e.setUint8(t+1,n>>>8&255),e.setUint8(t+2,n&255))},Vr=(e,t,n)=>Math.max(t,Math.min(n,e)),Hr=(e,t,n)=>e+(t-e)*n,Ur=(e,t)=>Math.round(e/t)*t,Wr=(e,t)=>Math.floor(e*t)/t,Gr=e=>{let t=0;for(;e!==0;)e&=e-1,t++;return t},Kr=/^[a-z]{3}$/,qr=e=>Kr.test(e),Jr=1e6*(1+2**-52),Yr=(e,t)=>{let n=e<0?-1:1;e=Math.abs(e);let r=0,i=1,a=1,o=0,s=e;for(;;){let e=Math.floor(s),c=e*a+r,l=e*o+i;if(l>t)return{num:n*a,den:o};if(r=a,i=o,a=c,o=l,s=1/(s-e),!isFinite(s))break}return{num:n*a,den:o}},Xr=class{constructor(){this.currentPromise=Promise.resolve()}call(e){return this.currentPromise=this.currentPromise.then(e)}},Zr=null,Qr=()=>Zr===null?Zr=typeof navigator<`u`&&navigator.userAgent?.includes(`Firefox`):Zr,$r=null,ei=()=>$r===null?$r=!!(typeof navigator<`u`&&(navigator.vendor?.includes(`Google Inc`)||/Chrome/.test(navigator.userAgent))):$r,ti=e=>globalThis.isSecureContext!==void 0&&!globalThis.isSecureContext?`${e} is not available in this environment; this may be because this page is running in an insecure context. Try serving your page over HTTPS or use localhost.`:`${e} is not available in this environment.`,ni=(async()=>{})().constructor,ri=e=>e instanceof ni||e instanceof Promise||typeof e?.then==`function`,ii=function*(e){for(let t in e){let n=e[t];n!==void 0&&(yield{key:t,value:n})}},ai=()=>{Symbol.dispose??=Symbol(`Symbol.dispose`)},oi=e=>typeof e==`number`&&!Number.isNaN(e),si=(e,t)=>{let n=-1,r=1/0;for(let i=0;i<e.length;i++){let a=t(e[i]);a<r&&(r=a,n=i)}return n},ci=e=>{H(Number.isInteger(e.num)),H(Number.isInteger(e.den)),H(e.den!==0);let t=Math.abs(e.num),n=Math.abs(e.den);for(;n!==0;){let e=t%n;t=n,n=e}let r=t||1;return{num:e.num/r,den:e.den/r}},li=(e,t)=>{if(typeof e!=`object`||!e)throw TypeError(`${t} must be an object.`);if(!Number.isInteger(e.left)||e.left<0)throw TypeError(`${t}.left must be a non-negative integer.`);if(!Number.isInteger(e.top)||e.top<0)throw TypeError(`${t}.top must be a non-negative integer.`);if(!Number.isInteger(e.width)||e.width<0)throw TypeError(`${t}.width must be a non-negative integer.`);if(!Number.isInteger(e.height)||e.height<0)throw TypeError(`${t}.height must be a non-negative integer.`)},ui=e=>new Promise(t=>setTimeout(t,e)),di=e=>Array.isArray(e)?e:[e],fi=class{constructor(){this._listeners=new Map}on(e,t,n){this._listeners.has(e)||this._listeners.set(e,new Set);let r={fn:t,once:n?.once??!1};return this._listeners.get(e).add(r),()=>{this._listeners.get(e)?.delete(r)}}_emit(...e){let[t,n]=e,r=this._listeners.get(t);if(r)for(let e of r){try{e.fn(n)}catch(e){console.error(e)}e.once&&r.delete(e)}}},pi=e=>typeof e==`object`&&!!e&&Object.getPrototypeOf(e)===Object.prototype&&Object.values(e).every(e=>typeof e==`string`),mi;(function(e){e[e.Silent=0]=`Silent`,e[e.Errors=1]=`Errors`,e[e.Warnings=2]=`Warnings`,e[e.Info=3]=`Info`})(mi||={});var hi=class e{constructor(){}static get level(){return e._level}static set level(t){if(t!==mi.Silent&&t!==mi.Errors&&t!==mi.Warnings&&t!==mi.Info)throw TypeError(`Invalid log level. Use one of the values of the LogLevel enum.`);e._level=t}static get _emitter(){return e._emitterInstance??=new fi}static on(t,n,r){return e._emitter.on(t,n,r)}static _error(...t){e._emitter._emit(`error`,t),e._level>=mi.Errors&&console.error(...t)}static _warn(...t){e._emitter._emit(`warn`,t),e._level>=mi.Warnings&&console.warn(...t)}static _info(...t){e._emitter._emit(`info`,t),e._level>=mi.Info&&console.info(...t)}};hi._level=mi.Info,hi._emitterInstance=null;var gi=class{constructor(e,t){if(this.data=e,this.mimeType=t,!(e instanceof Uint8Array))throw TypeError(`data must be a Uint8Array.`);if(typeof t!=`string`)throw TypeError(`mimeType must be a string.`)}},_i=class{constructor(e,t,n,r){if(this.data=e,this.mimeType=t,this.name=n,this.description=r,!(e instanceof Uint8Array))throw TypeError(`data must be a Uint8Array.`);if(t!==void 0&&typeof t!=`string`)throw TypeError(`mimeType, when provided, must be a string.`);if(n!==void 0&&typeof n!=`string`)throw TypeError(`name, when provided, must be a string.`);if(r!==void 0&&typeof r!=`string`)throw TypeError(`description, when provided, must be a string.`)}},vi=e=>{if(!e||typeof e!=`object`)throw TypeError(`tags must be an object.`);if(e.title!==void 0&&typeof e.title!=`string`)throw TypeError(`tags.title, when provided, must be a string.`);if(e.description!==void 0&&typeof e.description!=`string`)throw TypeError(`tags.description, when provided, must be a string.`);if(e.artist!==void 0&&typeof e.artist!=`string`)throw TypeError(`tags.artist, when provided, must be a string.`);if(e.album!==void 0&&typeof e.album!=`string`)throw TypeError(`tags.album, when provided, must be a string.`);if(e.albumArtist!==void 0&&typeof e.albumArtist!=`string`)throw TypeError(`tags.albumArtist, when provided, must be a string.`);if(e.trackNumber!==void 0&&(!Number.isInteger(e.trackNumber)||e.trackNumber<=0))throw TypeError(`tags.trackNumber, when provided, must be a positive integer.`);if(e.tracksTotal!==void 0&&(!Number.isInteger(e.tracksTotal)||e.tracksTotal<=0))throw TypeError(`tags.tracksTotal, when provided, must be a positive integer.`);if(e.discNumber!==void 0&&(!Number.isInteger(e.discNumber)||e.discNumber<=0))throw TypeError(`tags.discNumber, when provided, must be a positive integer.`);if(e.discsTotal!==void 0&&(!Number.isInteger(e.discsTotal)||e.discsTotal<=0))throw TypeError(`tags.discsTotal, when provided, must be a positive integer.`);if(e.genre!==void 0&&typeof e.genre!=`string`)throw TypeError(`tags.genre, when provided, must be a string.`);if(e.date!==void 0&&(!(e.date instanceof Date)||Number.isNaN(e.date.getTime())))throw TypeError(`tags.date, when provided, must be a valid Date.`);if(e.lyrics!==void 0&&typeof e.lyrics!=`string`)throw TypeError(`tags.lyrics, when provided, must be a string.`);if(e.images!==void 0){if(!Array.isArray(e.images))throw TypeError(`tags.images, when provided, must be an array.`);for(let t of e.images){if(!t||typeof t!=`object`)throw TypeError(`Each image in tags.images must be an object.`);if(!(t.data instanceof Uint8Array))throw TypeError(`Each image.data must be a Uint8Array.`);if(typeof t.mimeType!=`string`)throw TypeError(`Each image.mimeType must be a string.`);if(![`coverFront`,`coverBack`,`unknown`].includes(t.kind))throw TypeError(`Each image.kind must be 'coverFront', 'coverBack', or 'unknown'.`)}}if(e.comment!==void 0&&typeof e.comment!=`string`)throw TypeError(`tags.comment, when provided, must be a string.`);if(e.raw!==void 0){if(!e.raw||typeof e.raw!=`object`)throw TypeError(`tags.raw, when provided, must be an object.`);for(let t of Object.values(e.raw))if(t!==null&&typeof t!=`string`&&!(t instanceof Uint8Array)&&!(t instanceof gi)&&!(t instanceof _i)&&!pi(t))throw TypeError(`Each value in tags.raw must be a string, Uint8Array, RichImageData, AttachedFile, Record<string, string>, or null.`)}},yi=e=>{if(!e||typeof e!=`object`)throw TypeError(`disposition must be an object.`);if(e.default!==void 0&&typeof e.default!=`boolean`)throw TypeError(`disposition.default must be a boolean.`);if(e.primary!==void 0&&typeof e.primary!=`boolean`)throw TypeError(`disposition.primary must be a boolean.`);if(e.forced!==void 0&&typeof e.forced!=`boolean`)throw TypeError(`disposition.forced must be a boolean.`);if(e.original!==void 0&&typeof e.original!=`boolean`)throw TypeError(`disposition.original must be a boolean.`);if(e.commentary!==void 0&&typeof e.commentary!=`boolean`)throw TypeError(`disposition.commentary must be a boolean.`);if(e.hearingImpaired!==void 0&&typeof e.hearingImpaired!=`boolean`)throw TypeError(`disposition.hearingImpaired must be a boolean.`);if(e.visuallyImpaired!==void 0&&typeof e.visuallyImpaired!=`boolean`)throw TypeError(`disposition.visuallyImpaired must be a boolean.`)},W=class e{constructor(e){this.bytes=e,this.pos=0}seekToByte(e){this.pos=8*e}readBit(){let e=Math.floor(this.pos/8),t=this.bytes[e]??0,n=7-(this.pos&7),r=(t&1<<n)>>n;return this.pos++,r}readBits(e){if(e===1)return this.readBit();let t=0;for(let n=0;n<e;n++)t<<=1,t|=this.readBit();return t}writeBits(e,t){let n=this.pos+e;for(let e=this.pos;e<n;e++){let r=Math.floor(e/8),i=this.bytes[r],a=7-(e&7);i&=~(1<<a),i|=(t&1<<n-e-1)>>n-e-1<<a,this.bytes[r]=i}this.pos=n}readAlignedByte(){if(this.pos%8!=0)throw Error(`Bitstream is not byte-aligned.`);let e=this.pos/8,t=this.bytes[e]??0;return this.pos+=8,t}skipBits(e){this.pos+=e}getBitsLeft(){return this.bytes.length*8-this.pos}clone(){let t=new e(this.bytes);return t.pos=this.pos,t}},bi=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350],xi=[-1,1,2,3,4,5,6,8],Si=e=>{let t=e.objectType===5||e.objectType===29,n=e.objectType===29,r=t?e.outputSampleRate/2:e.outputSampleRate,i=n?1:e.outputNumberOfChannels,a=xi.indexOf(i);if(a===-1)throw TypeError(`Unsupported number of channels: ${e.outputNumberOfChannels}`);let o=16;e.objectType>=32&&(o+=6),Ti(r)===15&&(o+=24),t&&(o+=9,Ti(e.outputSampleRate)===15&&(o+=24));let s=Math.ceil(o/8),c=new Uint8Array(s),l=new W(c);return Ci(l,e.objectType),wi(l,r),l.writeBits(4,a),t&&(wi(l,e.outputSampleRate),Ci(l,2)),l.writeBits(3,0),c},Ci=(e,t)=>{t<32?e.writeBits(5,t):(e.writeBits(5,31),e.writeBits(6,t-32))},wi=(e,t)=>{let n=Ti(t);e.writeBits(4,n),n===15&&e.writeBits(24,t)},Ti=e=>{let t=bi.indexOf(e);return t===-1?15:t},Ei=[48e3,44100,32e3],Di=[24e3,22050,16e3],Oi;(function(e){e[e.NON_IDR_SLICE=1]=`NON_IDR_SLICE`,e[e.SLICE_DPA=2]=`SLICE_DPA`,e[e.SLICE_DPB=3]=`SLICE_DPB`,e[e.SLICE_DPC=4]=`SLICE_DPC`,e[e.IDR=5]=`IDR`,e[e.SEI=6]=`SEI`,e[e.SPS=7]=`SPS`,e[e.PPS=8]=`PPS`,e[e.AUD=9]=`AUD`,e[e.SPS_EXT=13]=`SPS_EXT`})(Oi||={});var ki;(function(e){e[e.RASL_N=8]=`RASL_N`,e[e.RASL_R=9]=`RASL_R`,e[e.BLA_W_LP=16]=`BLA_W_LP`,e[e.RSV_IRAP_VCL23=23]=`RSV_IRAP_VCL23`,e[e.VPS_NUT=32]=`VPS_NUT`,e[e.SPS_NUT=33]=`SPS_NUT`,e[e.PPS_NUT=34]=`PPS_NUT`,e[e.AUD_NUT=35]=`AUD_NUT`,e[e.PREFIX_SEI_NUT=39]=`PREFIX_SEI_NUT`,e[e.SUFFIX_SEI_NUT=40]=`SUFFIX_SEI_NUT`})(ki||={});var Ai=function*(e){let t=0,n=-1;for(;t<e.length-2;){let r=e.indexOf(0,t);if(r===-1||r>=e.length-2)break;t=r;let i=0;if(t+3<e.length&&e[t+1]===0&&e[t+2]===0&&e[t+3]===1?i=4:e[t+1]===0&&e[t+2]===1&&(i=3),i===0){t++;continue}n!==-1&&t>n&&(yield{offset:n,length:t-n}),n=t+i,t=n}n!==-1&&n<e.length&&(yield{offset:n,length:e.length-n})},ji=function*(e,t){let n=0,r=new DataView(e.buffer,e.byteOffset,e.byteLength);for(;n+t<=e.length;){let e;t===1?e=r.getUint8(n):t===2?e=r.getUint16(n,!1):t===3?e=zr(r,n,!1):(H(t===4),e=r.getUint32(n,!1)),n+=t,yield{offset:n,length:e},n+=e}},Mi=(e,t)=>t.description?ji(e,(Dr(t.description)[4]&3)+1):Ai(e),Ni=e=>e&31,Pi=e=>{let t=[],n=e.length;for(let r=0;r<n;r++)r+2<n&&e[r]===0&&e[r+1]===0&&e[r+2]===3?(t.push(0,0),r+=2):t.push(e[r]);return new Uint8Array(t)};new Uint8Array([0,0,0,1]);var Fi=(e,t)=>{let n=e.reduce((e,n)=>e+t+n.byteLength,0),r=new Uint8Array(n),i=0;for(let n of e){let e=new DataView(r.buffer,r.byteOffset,r.byteLength);switch(t){case 1:e.setUint8(i,n.byteLength);break;case 2:e.setUint16(i,n.byteLength,!1);break;case 3:Br(e,i,n.byteLength,!1);break;case 4:e.setUint32(i,n.byteLength,!1)}i+=t,r.set(n,i),i+=n.byteLength}return r},Ii=e=>{try{let t=[],n=[],r=[];for(let i of Ai(e)){let a=e.subarray(i.offset,i.offset+i.length),o=Ni(a[0]);o===Oi.SPS?t.push(a):o===Oi.PPS?n.push(a):o===Oi.SPS_EXT&&r.push(a)}if(t.length===0||n.length===0)return null;let i=t[0],a=zi(i);H(a!==null);let o=a.profileIdc===100||a.profileIdc===110||a.profileIdc===122||a.profileIdc===144;return{configurationVersion:1,avcProfileIndication:a.profileIdc,profileCompatibility:a.constraintFlags,avcLevelIndication:a.levelIdc,lengthSizeMinusOne:3,sequenceParameterSets:t,pictureParameterSets:n,chromaFormat:o?a.chromaFormatIdc:null,bitDepthLumaMinus8:o?a.bitDepthLumaMinus8:null,bitDepthChromaMinus8:o?a.bitDepthChromaMinus8:null,sequenceParameterSetExt:o?r:null}}catch(e){return hi._error(`Error building AVC Decoder Configuration Record:`,e),null}},Li=e=>{let t=[];t.push(e.configurationVersion),t.push(e.avcProfileIndication),t.push(e.profileCompatibility),t.push(e.avcLevelIndication),t.push(252|e.lengthSizeMinusOne&3),t.push(224|e.sequenceParameterSets.length&31);for(let n of e.sequenceParameterSets){let e=n.byteLength;t.push(e>>8),t.push(e&255);for(let r=0;r<e;r++)t.push(n[r])}t.push(e.pictureParameterSets.length);for(let n of e.pictureParameterSets){let e=n.byteLength;t.push(e>>8),t.push(e&255);for(let r=0;r<e;r++)t.push(n[r])}if(e.avcProfileIndication===100||e.avcProfileIndication===110||e.avcProfileIndication===122||e.avcProfileIndication===144){H(e.chromaFormat!==null),H(e.bitDepthLumaMinus8!==null),H(e.bitDepthChromaMinus8!==null),H(e.sequenceParameterSetExt!==null),t.push(252|e.chromaFormat&3),t.push(248|e.bitDepthLumaMinus8&7),t.push(248|e.bitDepthChromaMinus8&7),t.push(e.sequenceParameterSetExt.length);for(let n of e.sequenceParameterSetExt){let e=n.byteLength;t.push(e>>8),t.push(e&255);for(let r=0;r<e;r++)t.push(n[r])}}return new Uint8Array(t)},Ri={1:{num:1,den:1},2:{num:12,den:11},3:{num:10,den:11},4:{num:16,den:11},5:{num:40,den:33},6:{num:24,den:11},7:{num:20,den:11},8:{num:32,den:11},9:{num:80,den:33},10:{num:18,den:11},11:{num:15,den:11},12:{num:64,den:33},13:{num:160,den:99},14:{num:4,den:3},15:{num:3,den:2},16:{num:2,den:1}},zi=e=>{try{let t=new W(Pi(e));if(t.skipBits(1),t.skipBits(2),t.readBits(5)!==7)return null;let n=t.readAlignedByte(),r=t.readAlignedByte(),i=t.readAlignedByte();U(t);let a=1,o=0,s=0,c=0;if((n===100||n===110||n===122||n===244||n===44||n===83||n===86||n===118||n===128)&&(a=U(t),a===3&&(c=t.readBits(1)),o=U(t),s=U(t),t.skipBits(1),t.readBits(1))){for(let e=0;e<(a===3?12:8);e++)if(t.readBits(1)){let n=e<6?16:64,r=8,i=8;for(let e=0;e<n;e++){if(i!==0){let e=Er(t);i=(r+e+256)%256}r=i===0?r:i}}}U(t);let l=U(t);if(l===0)U(t);else if(l===1){t.skipBits(1),Er(t),Er(t);let e=U(t);for(let n=0;n<e;n++)Er(t)}U(t),t.skipBits(1);let u=U(t),d=U(t),f=16*(u+1),p=16*(d+1),m=f,h=p,g=t.readBits(1);if(g||t.skipBits(1),t.skipBits(1),t.readBits(1)){let e=U(t),n=U(t),r=U(t),i=U(t),o,s;if((c===0?a:0)===0)o=1,s=2-g;else{let e=a===3?1:2,t=a===1?2:1;o=e,s=t*(2-g)}m-=o*(e+n),h-=s*(r+i)}let _=2,v=2,y=2,b=0,x={num:1,den:1},S=null,C=null;if(t.readBits(1)){if(t.readBits(1)){let e=t.readBits(8);if(e===255)x={num:t.readBits(16),den:t.readBits(16)};else{let t=Ri[e];t&&(x=t)}}t.readBits(1)&&t.skipBits(1),t.readBits(1)&&(t.skipBits(3),b=t.readBits(1),t.readBits(1)&&(_=t.readBits(8),v=t.readBits(8),y=t.readBits(8))),t.readBits(1)&&(U(t),U(t)),t.readBits(1)&&(t.skipBits(32),t.skipBits(32),t.skipBits(1));let e=t.readBits(1);e&&Bi(t);let n=t.readBits(1);n&&Bi(t),(e||n)&&t.skipBits(1),t.skipBits(1),t.readBits(1)&&(t.skipBits(1),U(t),U(t),U(t),U(t),S=U(t),C=U(t))}if(S===null){H(C===null);let e=r&16;if((n===44||n===86||n===100||n===110||n===122||n===244)&&e)S=0,C=0;else{let e=u+1,t=d+1,n=(2-g)*t,r=Ea.find(e=>e.level>=i)??wr(Ea),a=Math.min(Math.floor(r.maxDpbMbs/(e*n)),16);S=a,C=a}}return H(C!==null),{profileIdc:n,constraintFlags:r,levelIdc:i,frameMbsOnlyFlag:g,chromaFormatIdc:a,bitDepthLumaMinus8:o,bitDepthChromaMinus8:s,codedWidth:f,codedHeight:p,displayWidth:m,displayHeight:h,pixelAspectRatio:x,colourPrimaries:_,matrixCoefficients:y,transferCharacteristics:v,fullRangeFlag:b,numReorderFrames:S,maxDecFrameBuffering:C}}catch(e){return hi._error(`Error parsing AVC SPS:`,e),null}},Bi=e=>{let t=U(e);e.skipBits(4),e.skipBits(4);for(let n=0;n<=t;n++)U(e),U(e),e.skipBits(1);e.skipBits(5),e.skipBits(5),e.skipBits(5),e.skipBits(5)},Vi=(e,t)=>t.description?ji(e,(Dr(t.description)[21]&3)+1):Ai(e),Hi=e=>e>>1&63,Ui=e=>{try{let t=new W(Pi(e));t.skipBits(16),t.readBits(4);let n=t.readBits(3),r=t.readBits(1),{general_profile_space:i,general_tier_flag:a,general_profile_idc:o,general_profile_compatibility_flags:s,general_constraint_indicator_flags:c,general_level_idc:l}=Gi(t,n);U(t);let u=U(t),d=0;u===3&&(d=t.readBits(1));let f=U(t),p=U(t),m=f,h=p;if(t.readBits(1)){let e=U(t),n=U(t),r=U(t),i=U(t),a=1,o=1,s=d===0?u:0;s===1?(a=2,o=2):s===2&&(a=2,o=1),m-=(e+n)*a,h-=(r+i)*o}let g=U(t),_=U(t);U(t);let v=t.readBits(1)?0:n,y=0;for(let e=v;e<=n;e++)U(t),y=U(t),U(t);if(U(t),U(t),U(t),U(t),U(t),U(t),t.readBits(1)&&t.readBits(1)&&Ki(t),t.skipBits(1),t.skipBits(1),t.readBits(1)&&(t.skipBits(4),t.skipBits(4),U(t),U(t),t.skipBits(1)),qi(t,U(t)),t.readBits(1)){let e=U(t);for(let n=0;n<e;n++)U(t),t.skipBits(1)}t.skipBits(1),t.skipBits(1);let b=2,x=2,S=2,C=0,ee=0,w={num:1,den:1};if(t.readBits(1)){let e=Yi(t,n);w=e.pixelAspectRatio,b=e.colourPrimaries,x=e.transferCharacteristics,S=e.matrixCoefficients,C=e.fullRangeFlag,ee=e.minSpatialSegmentationIdc}return{displayWidth:m,displayHeight:h,pixelAspectRatio:w,colourPrimaries:b,transferCharacteristics:x,matrixCoefficients:S,fullRangeFlag:C,maxDecFrameBuffering:y+1,spsMaxSubLayersMinus1:n,spsTemporalIdNestingFlag:r,generalProfileSpace:i,generalTierFlag:a,generalProfileIdc:o,generalProfileCompatibilityFlags:s,generalConstraintIndicatorFlags:c,generalLevelIdc:l,chromaFormatIdc:u,bitDepthLumaMinus8:g,bitDepthChromaMinus8:_,minSpatialSegmentationIdc:ee}}catch(e){return hi._error(`Error parsing HEVC SPS:`,e),null}},Wi=e=>{try{let t=[],n=[],r=[],i=[];for(let a of Ai(e)){let o=e.subarray(a.offset,a.offset+a.length),s=Hi(o[0]);s===ki.VPS_NUT?t.push(o):s===ki.SPS_NUT?n.push(o):s===ki.PPS_NUT?r.push(o):(s===ki.PREFIX_SEI_NUT||s===ki.SUFFIX_SEI_NUT)&&i.push(o)}if(n.length===0||r.length===0)return null;let a=Ui(n[0]);if(!a)return null;let o=0;if(r.length>0){let e=r[0],t=new W(Pi(e));t.skipBits(16),U(t),U(t),t.skipBits(1),t.skipBits(1),t.skipBits(3),t.skipBits(1),t.skipBits(1),U(t),U(t),Er(t),t.skipBits(1),t.skipBits(1),t.readBits(1)&&U(t),Er(t),Er(t),t.skipBits(1),t.skipBits(1),t.skipBits(1),t.skipBits(1);let n=t.readBits(1),i=t.readBits(1);o=!n&&!i?0:n&&!i?2:!n&&i?3:0}let s=[...t.length?[{arrayCompleteness:1,nalUnitType:ki.VPS_NUT,nalUnits:t}]:[],...n.length?[{arrayCompleteness:1,nalUnitType:ki.SPS_NUT,nalUnits:n}]:[],...r.length?[{arrayCompleteness:1,nalUnitType:ki.PPS_NUT,nalUnits:r}]:[],...i.length?[{arrayCompleteness:1,nalUnitType:Hi(i[0][0]),nalUnits:i}]:[]];return{configurationVersion:1,generalProfileSpace:a.generalProfileSpace,generalTierFlag:a.generalTierFlag,generalProfileIdc:a.generalProfileIdc,generalProfileCompatibilityFlags:a.generalProfileCompatibilityFlags,generalConstraintIndicatorFlags:a.generalConstraintIndicatorFlags,generalLevelIdc:a.generalLevelIdc,minSpatialSegmentationIdc:a.minSpatialSegmentationIdc,parallelismType:o,chromaFormatIdc:a.chromaFormatIdc,bitDepthLumaMinus8:a.bitDepthLumaMinus8,bitDepthChromaMinus8:a.bitDepthChromaMinus8,avgFrameRate:0,constantFrameRate:0,numTemporalLayers:a.spsMaxSubLayersMinus1+1,temporalIdNested:a.spsTemporalIdNestingFlag,lengthSizeMinusOne:3,arrays:s}}catch(e){return hi._error(`Error building HEVC Decoder Configuration Record:`,e),null}},Gi=(e,t)=>{let n=e.readBits(2),r=e.readBits(1),i=e.readBits(5),a=0;for(let t=0;t<32;t++)a=a<<1|e.readBits(1);let o=new Uint8Array(6);for(let t=0;t<6;t++)o[t]=e.readBits(8);let s=e.readBits(8),c=[],l=[];for(let n=0;n<t;n++)c.push(e.readBits(1)),l.push(e.readBits(1));if(t>0)for(let n=t;n<8;n++)e.skipBits(2);for(let n=0;n<t;n++)c[n]&&e.skipBits(88),l[n]&&e.skipBits(8);return{general_profile_space:n,general_tier_flag:r,general_profile_idc:i,general_profile_compatibility_flags:a,general_constraint_indicator_flags:o,general_level_idc:s}},Ki=e=>{for(let t=0;t<4;t++)for(let n=0;n<(t===3?2:6);n++)if(!e.readBits(1))U(e);else{let n=Math.min(64,1<<4+(t<<1));t>1&&Er(e);for(let t=0;t<n;t++)Er(e)}},qi=(e,t)=>{let n=[];for(let r=0;r<t;r++)n[r]=Ji(e,r,t,n)},Ji=(e,t,n,r)=>{let i=0,a=0,o=0;if(t!==0&&(a=e.readBits(1)),a){o=t===n?t-(U(e)+1):t-1,e.readBits(1),U(e);let a=r[o]??0;for(let t=0;t<=a;t++)e.readBits(1)||e.readBits(1);i=r[o]}else{let t=U(e),n=U(e);for(let n=0;n<t;n++)U(e),e.readBits(1);for(let t=0;t<n;t++)U(e),e.readBits(1);i=t+n}return i},Yi=(e,t)=>{let n=2,r=2,i=2,a=0,o=0,s={num:1,den:1};if(e.readBits(1)){let t=e.readBits(8);if(t===255)s={num:e.readBits(16),den:e.readBits(16)};else{let e=Ri[t];e&&(s=e)}}return e.readBits(1)&&e.readBits(1),e.readBits(1)&&(e.readBits(3),a=e.readBits(1),e.readBits(1)&&(n=e.readBits(8),r=e.readBits(8),i=e.readBits(8))),e.readBits(1)&&(U(e),U(e)),e.readBits(1),e.readBits(1),e.readBits(1),e.readBits(1)&&(U(e),U(e),U(e),U(e)),e.readBits(1)&&(e.readBits(32),e.readBits(32),e.readBits(1)&&U(e),e.readBits(1)&&Xi(e,!0,t)),e.readBits(1)&&(e.readBits(1),e.readBits(1),e.readBits(1),o=U(e),U(e),U(e),U(e),U(e)),{pixelAspectRatio:s,colourPrimaries:n,transferCharacteristics:r,matrixCoefficients:i,fullRangeFlag:a,minSpatialSegmentationIdc:o}},Xi=(e,t,n)=>{let r=!1,i=!1,a=!1;t&&(r=e.readBits(1)===1,i=e.readBits(1)===1,(r||i)&&(a=e.readBits(1)===1,a&&(e.readBits(8),e.readBits(5),e.readBits(1),e.readBits(5)),e.readBits(4),e.readBits(4),a&&e.readBits(4),e.readBits(5),e.readBits(5),e.readBits(5)));for(let t=0;t<=n;t++){let t=e.readBits(1)===1,n=!0;t||(n=e.readBits(1)===1);let o=!1;n?U(e):o=e.readBits(1)===1;let s=1;o||(s=U(e)+1),r&&Zi(e,s,a),i&&Zi(e,s,a)}},Zi=(e,t,n)=>{for(let r=0;r<t;r++)U(e),U(e),n&&(U(e),U(e)),e.readBits(1)},Qi=e=>{let t=[];t.push(e.configurationVersion),t.push((e.generalProfileSpace&3)<<6|(e.generalTierFlag&1)<<5|e.generalProfileIdc&31),t.push(e.generalProfileCompatibilityFlags>>>24&255),t.push(e.generalProfileCompatibilityFlags>>>16&255),t.push(e.generalProfileCompatibilityFlags>>>8&255),t.push(e.generalProfileCompatibilityFlags&255),t.push(...e.generalConstraintIndicatorFlags),t.push(e.generalLevelIdc&255),t.push(240|e.minSpatialSegmentationIdc>>8&15),t.push(e.minSpatialSegmentationIdc&255),t.push(252|e.parallelismType&3),t.push(252|e.chromaFormatIdc&3),t.push(248|e.bitDepthLumaMinus8&7),t.push(248|e.bitDepthChromaMinus8&7),t.push(e.avgFrameRate>>8&255),t.push(e.avgFrameRate&255),t.push((e.constantFrameRate&3)<<6|(e.numTemporalLayers&7)<<3|(e.temporalIdNested&1)<<2|e.lengthSizeMinusOne&3),t.push(e.arrays.length&255);for(let n of e.arrays){t.push((n.arrayCompleteness&1)<<7|0|n.nalUnitType&63),t.push(n.nalUnits.length>>8&255),t.push(n.nalUnits.length&255);for(let e of n.nalUnits){t.push(e.length>>8&255),t.push(e.length&255);for(let n=0;n<e.length;n++)t.push(e[n])}}return new Uint8Array(t)},$i;(function(e){e[e.audAllowed=0]=`audAllowed`,e[e.beforeFirstVcl=1]=`beforeFirstVcl`,e[e.afterFirstVcl=2]=`afterFirstVcl`,e[e.eoBitstreamAllowed=3]=`eoBitstreamAllowed`,e[e.noMoreDataAllowed=4]=`noMoreDataAllowed`})($i||={});var ea=function*(e){let t=new W(e),n=()=>{let e=0;for(let n=0;n<8;n++){let r=t.readAlignedByte();if(e+=(r&127)*2**(n*7),!(r&128))break;if(n===7&&r&128)return null}return e>2**32-1?null:e};for(;t.getBitsLeft()>=8;){t.skipBits(1);let r=t.readBits(4),i=t.readBits(1),a=t.readBits(1);t.skipBits(1),i&&t.skipBits(8);let o;if(a){let e=n();if(e===null)return;o=e}else o=Math.floor(t.getBitsLeft()/8);H(t.pos%8==0),yield{type:r,data:e.subarray(t.pos/8,t.pos/8+o)},t.skipBits(o*8)}},ta=e=>{let t=Or(e),n=t.getUint8(9),r=t.getUint16(10,!0),i=t.getUint32(12,!0),a=t.getInt16(16,!0),o=t.getUint8(18),s=null;return o&&(s=e.subarray(19,21+n)),{outputChannelCount:n,preSkip:r,inputSampleRate:i,outputGain:a,channelMappingFamily:o,channelMappingTable:s}},na=(e,t,n)=>{switch(e){case`avc`:for(let e of Mi(n,t)){let t=n[e.offset],r=Ni(t);if(r>=Oi.NON_IDR_SLICE&&r<=Oi.SLICE_DPC)return`delta`;if(r===Oi.IDR)return`key`;if(r===Oi.SEI&&!ei()){let t=Pi(n.subarray(e.offset,e.offset+e.length)),r=1;do{let e=0;for(;;){let n=t[r++];if(n===void 0||(e+=n,n<255))break}let n=0;for(;;){let e=t[r++];if(e===void 0||(n+=e,e<255))break}if(e===6){let e=new W(t);e.pos=8*r;let n=U(e),i=e.readBits(1);if(n===0&&i===1)return`key`}r+=n}while(r<t.length-1)}}return`delta`;case`hevc`:for(let e of Vi(n,t)){let t=Hi(n[e.offset]);if(t<ki.BLA_W_LP)return`delta`;if(t<=ki.RSV_IRAP_VCL23)return`key`}return`delta`;case`vp8`:return n[0]&1?`delta`:`key`;case`vp9`:{let e=new W(n);if(e.readBits(2)!==2)return null;let t=e.readBits(1);return(e.readBits(1)<<1)+t===3&&e.skipBits(1),e.readBits(1)?null:e.readBits(1)===0?`key`:`delta`}case`av1`:{let e=!1;for(let{type:t,data:r}of ea(n))if(t===1){let t=new W(r);t.skipBits(4),e=!!t.readBits(1)}else if(t===3||t===6||t===7){if(e)return`key`;let t=new W(r);return t.readBits(1)?null:t.readBits(2)===0?`key`:`delta`}return null}case`prores`:return`key`;default:Rr(e),H(!1)}},ra;(function(e){e[e.STREAMINFO=0]=`STREAMINFO`,e[e.VORBIS_COMMENT=4]=`VORBIS_COMMENT`,e[e.PICTURE=6]=`PICTURE`})(ra||={});var ia=e=>{if(e.length<7||e[0]!==11||e[1]!==119)return null;let t=new W(e);t.skipBits(16),t.skipBits(16);let n=t.readBits(2);if(n===3)return null;let r=t.readBits(6),i=t.readBits(5);if(i>8)return null;let a=t.readBits(3),o=t.readBits(3);return o&1&&o!==1&&t.skipBits(2),o&4&&t.skipBits(2),o===2&&t.skipBits(2),{fscod:n,bsid:i,bsmod:a,acmod:o,lfeon:t.readBits(1),bitRateCode:Math.floor(r/2)}};new Uint8Array([5,4,65,67,45,51]),new Uint8Array([5,4,69,65,67,51]);var aa=[1,2,3,6],oa=e=>{if(e.length<6||e[0]!==11||e[1]!==119)return null;let t=new W(e);t.skipBits(16);let n=t.readBits(2);if(t.skipBits(3),n!==0&&n!==2)return null;let r=t.readBits(11),i=t.readBits(2),a=0,o;i===3?(a=t.readBits(2),o=3):o=t.readBits(2);let s=t.readBits(3),c=t.readBits(1),l=t.readBits(5);if(l<11||l>16)return null;let u=aa[o],d;return d=i<3?Ei[i]/1e3:Di[a]/1e3,{dataRate:Math.round((r+1)*d/(u*16)),substreams:[{fscod:i,fscod2:a,bsid:l,bsmod:0,acmod:s,lfeon:c,numDepSub:0,chanLoc:0}]}},sa=8,ca=[0,8e3,16e3,32e3,0,0,11025,22050,44100,0,0,12e3,24e3,48e3,96e3,192e3],la=[32e3,56e3,64e3,96e3,112e3,128e3,192e3,224e3,256e3,32e4,384e3,448e3,512e3,576e3,64e4,768e3,96e4,1024e3,1152e3,128e4,1344e3,1408e3,1411200,1472e3,1536e3,192e4,2048e3,3072e3,384e4,0,0,0],ua=[16,16,20,20,0,24,24,0],da=[1,2,2,2,2,3,3,4,4,5,6,6,6,7,8,8],fa=[1,2,2,2,2,3,18,19,6,7,518,323,83,519,582,535],pa=8,ma=[32e3,44100,48e3,0],ha=[8e3,16e3,32e3,64e3,128e3,22050,44100,88200,176400,352800,12e3,24e3,48e3,96e3,192e3,384e3],ga=[512,1024,2048,4096],_a=e=>{let t=va(e),n=Or(e),r=t?Math.ceil(t.frameSize/4)*4:0,i=null;for(;r+4<=e.length&&n.getUint32(r)===1683496997;){let t=ya(e.subarray(r));if(!t)break;i??=t,r+=t.frameSize}if(t)return{frameSize:i?r:t.frameSize,sampleRate:t.sampleRate,numberOfChannels:t.numberOfChannels,sampleCount:t.sampleCount,channelLayout:t.channelLayout,pcmResolution:t.pcmResolution,bitRate:t.bitRate,core:t,hasExtensions:i!==null};if(!i?.asset)return null;let{asset:a}=i;return{frameSize:r,sampleRate:a.sampleRate,numberOfChannels:a.numberOfChannels,sampleCount:a.sampleCount,channelLayout:a.channelLayout,pcmResolution:a.pcmResolution,bitRate:0,core:null,hasExtensions:!0}},va=e=>{if(e.length<18||e[0]!==127||e[1]!==254||e[2]!==128||e[3]!==1)return null;let t=new W(e);if(t.skipBits(32),t.skipBits(1),t.readBits(5)!==31)return null;let n=t.readBits(1),r=t.readBits(7)+1;if(r%sa!==0)return null;let i=t.readBits(14)+1;if(i<96)return null;let a=t.readBits(6);if(a>=da.length)return null;let o=ca[t.readBits(4)];if(o===0)return null;let s=la[t.readBits(5)];if(t.readBits(1)!==0)return null;t.skipBits(4),t.skipBits(5);let c=t.readBits(2);if(c===3)return null;t.skipBits(1),n&&t.skipBits(16),t.skipBits(7);let l=ua[t.readBits(3)];if(l===0)return null;let u=c!==0;return{frameSize:i,sampleRate:o,numberOfChannels:da[a]+ +!!u,sampleCount:r*32,channelLayout:fa[a]|(u?pa:0),amode:a,lfePresent:u,bitRate:s,pcmResolution:l}},ya=e=>{if(e.length<10||e[0]!==100||e[1]!==88||e[2]!==32||e[3]!==37)return null;let t=new W(e);t.skipBits(32),t.skipBits(8);let n=t.readBits(2),r=t.readBits(1),i=8+4*r,a=16+4*r;t.skipBits(i);let o=t.readBits(a)+1,s={frameSize:o,asset:null};if(!t.readBits(1))return s;let c=ma[t.readBits(2)],l=512*(t.readBits(3)+1);t.readBits(1)&&t.skipBits(36);let u=t.readBits(3)+1,d=t.readBits(3)+1,f=[];for(let e=0;e<u;e++)f.push(t.readBits(n+1));for(let e of f)t.skipBits(8*Gr(e));if(t.readBits(1)){t.skipBits(2);let e=t.readBits(2)+1<<2,n=t.readBits(2)+1;t.skipBits(n*e)}for(let e=0;e<d;e++)t.skipBits(a);t.skipBits(9),t.skipBits(3),t.readBits(1)&&t.skipBits(4),t.readBits(1)&&t.skipBits(24),t.readBits(1)&&t.skipBits(8*(t.readBits(10)+1));let p=t.readBits(5)+1,m=ha[t.readBits(4)],h=t.readBits(8)+1,g=0;if(t.readBits(1)&&(h>2&&t.skipBits(1),h>6&&t.skipBits(1),t.readBits(1))){let e=t.readBits(2)+1<<2;g=t.readBits(e)}return c===0||t.getBitsLeft()<0?s:{frameSize:o,asset:{sampleRate:m,numberOfChannels:h,sampleCount:Math.round(l*m/c),channelLayout:g,pcmResolution:p}}},ba=e=>{let t=new Uint8Array(20),n=Or(t);n.setUint32(0,e.sampleRate),n.setUint32(4,e.bitRate),n.setUint32(8,e.bitRate),t[12]=e.pcmResolution;let r=e.core&&!e.hasExtensions?1:0,i=new W(t);return i.seekToByte(13),i.writeBits(2,Math.max(ga.indexOf(e.sampleCount),0)),i.writeBits(5,r),i.writeBits(1,+!!e.core?.lfePresent),i.writeBits(6,e.core?.amode??0),i.writeBits(14,e.core?e.core.frameSize-1:0),i.writeBits(1,0),i.writeBits(3,0),i.writeBits(16,e.channelLayout),i.writeBits(1,0),i.writeBits(1,0),i.writeBits(1,0),i.writeBits(5,0),t},xa=[`avc`,`hevc`,`vp9`,`av1`,`vp8`,`prores`],Sa=[`pcm-s16`,`pcm-s16be`,`pcm-s24`,`pcm-s24be`,`pcm-s32`,`pcm-s32be`,`pcm-f32`,`pcm-f32be`,`pcm-f64`,`pcm-f64be`,`pcm-u8`,`pcm-s8`,`ulaw`,`alaw`],Ca=[`aac`,`opus`,`mp3`,`vorbis`,`flac`,`ac3`,`eac3`,`dts`],wa=[...Ca,...Sa],Ta=[`webvtt`],Ea=[{maxMacroblocks:99,maxBitrate:64e3,maxDpbMbs:396,level:10},{maxMacroblocks:396,maxBitrate:192e3,maxDpbMbs:900,level:11},{maxMacroblocks:396,maxBitrate:384e3,maxDpbMbs:2376,level:12},{maxMacroblocks:396,maxBitrate:768e3,maxDpbMbs:2376,level:13},{maxMacroblocks:396,maxBitrate:2e6,maxDpbMbs:2376,level:20},{maxMacroblocks:792,maxBitrate:4e6,maxDpbMbs:4752,level:21},{maxMacroblocks:1620,maxBitrate:4e6,maxDpbMbs:8100,level:22},{maxMacroblocks:1620,maxBitrate:1e7,maxDpbMbs:8100,level:30},{maxMacroblocks:3600,maxBitrate:14e6,maxDpbMbs:18e3,level:31},{maxMacroblocks:5120,maxBitrate:2e7,maxDpbMbs:20480,level:32},{maxMacroblocks:8192,maxBitrate:2e7,maxDpbMbs:32768,level:40},{maxMacroblocks:8192,maxBitrate:5e7,maxDpbMbs:32768,level:41},{maxMacroblocks:8704,maxBitrate:5e7,maxDpbMbs:34816,level:42},{maxMacroblocks:22080,maxBitrate:135e6,maxDpbMbs:110400,level:50},{maxMacroblocks:36864,maxBitrate:24e7,maxDpbMbs:184320,level:51},{maxMacroblocks:36864,maxBitrate:24e7,maxDpbMbs:184320,level:52},{maxMacroblocks:139264,maxBitrate:24e7,maxDpbMbs:696320,level:60},{maxMacroblocks:139264,maxBitrate:48e7,maxDpbMbs:696320,level:61},{maxMacroblocks:139264,maxBitrate:8e8,maxDpbMbs:696320,level:62}],Da=[{maxPictureSize:36864,maxBitrate:128e3,tier:`L`,level:30},{maxPictureSize:122880,maxBitrate:15e5,tier:`L`,level:60},{maxPictureSize:245760,maxBitrate:3e6,tier:`L`,level:63},{maxPictureSize:552960,maxBitrate:6e6,tier:`L`,level:90},{maxPictureSize:983040,maxBitrate:1e7,tier:`L`,level:93},{maxPictureSize:2228224,maxBitrate:12e6,tier:`L`,level:120},{maxPictureSize:2228224,maxBitrate:3e7,tier:`H`,level:120},{maxPictureSize:2228224,maxBitrate:2e7,tier:`L`,level:123},{maxPictureSize:2228224,maxBitrate:5e7,tier:`H`,level:123},{maxPictureSize:8912896,maxBitrate:25e6,tier:`L`,level:150},{maxPictureSize:8912896,maxBitrate:1e8,tier:`H`,level:150},{maxPictureSize:8912896,maxBitrate:4e7,tier:`L`,level:153},{maxPictureSize:8912896,maxBitrate:16e7,tier:`H`,level:153},{maxPictureSize:8912896,maxBitrate:6e7,tier:`L`,level:156},{maxPictureSize:8912896,maxBitrate:24e7,tier:`H`,level:156},{maxPictureSize:35651584,maxBitrate:6e7,tier:`L`,level:180},{maxPictureSize:35651584,maxBitrate:24e7,tier:`H`,level:180},{maxPictureSize:35651584,maxBitrate:12e7,tier:`L`,level:183},{maxPictureSize:35651584,maxBitrate:48e7,tier:`H`,level:183},{maxPictureSize:35651584,maxBitrate:24e7,tier:`L`,level:186},{maxPictureSize:35651584,maxBitrate:8e8,tier:`H`,level:186}],Oa=[{maxPictureSize:36864,maxBitrate:2e5,level:10},{maxPictureSize:73728,maxBitrate:8e5,level:11},{maxPictureSize:122880,maxBitrate:18e5,level:20},{maxPictureSize:245760,maxBitrate:36e5,level:21},{maxPictureSize:552960,maxBitrate:72e5,level:30},{maxPictureSize:983040,maxBitrate:12e6,level:31},{maxPictureSize:2228224,maxBitrate:18e6,level:40},{maxPictureSize:2228224,maxBitrate:3e7,level:41},{maxPictureSize:8912896,maxBitrate:6e7,level:50},{maxPictureSize:8912896,maxBitrate:12e7,level:51},{maxPictureSize:8912896,maxBitrate:18e7,level:52},{maxPictureSize:35651584,maxBitrate:18e7,level:60},{maxPictureSize:35651584,maxBitrate:24e7,level:61},{maxPictureSize:35651584,maxBitrate:48e7,level:62}],ka=[{maxPictureSize:147456,maxBitrate:15e5,tier:`M`,level:0},{maxPictureSize:278784,maxBitrate:3e6,tier:`M`,level:1},{maxPictureSize:665856,maxBitrate:6e6,tier:`M`,level:4},{maxPictureSize:1065024,maxBitrate:1e7,tier:`M`,level:5},{maxPictureSize:2359296,maxBitrate:12e6,tier:`M`,level:8},{maxPictureSize:2359296,maxBitrate:3e7,tier:`H`,level:8},{maxPictureSize:2359296,maxBitrate:2e7,tier:`M`,level:9},{maxPictureSize:2359296,maxBitrate:5e7,tier:`H`,level:9},{maxPictureSize:8912896,maxBitrate:3e7,tier:`M`,level:12},{maxPictureSize:8912896,maxBitrate:1e8,tier:`H`,level:12},{maxPictureSize:8912896,maxBitrate:4e7,tier:`M`,level:13},{maxPictureSize:8912896,maxBitrate:16e7,tier:`H`,level:13},{maxPictureSize:8912896,maxBitrate:6e7,tier:`M`,level:14},{maxPictureSize:8912896,maxBitrate:24e7,tier:`H`,level:14},{maxPictureSize:35651584,maxBitrate:6e7,tier:`M`,level:15},{maxPictureSize:35651584,maxBitrate:24e7,tier:`H`,level:15},{maxPictureSize:35651584,maxBitrate:6e7,tier:`M`,level:16},{maxPictureSize:35651584,maxBitrate:24e7,tier:`H`,level:16},{maxPictureSize:35651584,maxBitrate:1e8,tier:`M`,level:17},{maxPictureSize:35651584,maxBitrate:48e7,tier:`H`,level:17},{maxPictureSize:35651584,maxBitrate:16e7,tier:`M`,level:18},{maxPictureSize:35651584,maxBitrate:8e8,tier:`H`,level:18},{maxPictureSize:35651584,maxBitrate:16e7,tier:`M`,level:19},{maxPictureSize:35651584,maxBitrate:8e8,tier:`H`,level:19}],Aa=[`ap4x`,`ap4h`,`apch`,`apcn`,`apcs`,`apco`],ja=[`dtsc`,`dtsh`,`dtsl`,`dtse`],Ma=[{fourCc:`apco`,bitrate:45e6,alpha:!1},{fourCc:`apcs`,bitrate:102e6,alpha:!1},{fourCc:`apcn`,bitrate:147e6,alpha:!1},{fourCc:`apch`,bitrate:22e7,alpha:!1},{fourCc:`ap4h`,bitrate:33e7,alpha:!0},{fourCc:`ap4x`,bitrate:5e8,alpha:!0}],Na=(e,t,n,r,i)=>{if(e===`avc`){let e=Math.ceil(t/16)*Math.ceil(n/16),i=Ea.find(t=>e<=t.maxMacroblocks&&r<=t.maxBitrate)??wr(Ea),a=i?i.level:0;return`avc1.${`64`.padStart(2,`0`)}00${a.toString(16).padStart(2,`0`)}`}if(e===`hevc`){let e=t*n,i=Da.find(t=>e<=t.maxPictureSize&&r<=t.maxBitrate)??wr(Da);return`hev1.1.6.${i.tier}${i.level}.B0`}if(e===`vp8`)return`vp8`;if(e===`vp9`){let e=t*n;return`vp09.00.${(Oa.find(t=>e<=t.maxPictureSize&&r<=t.maxBitrate)??wr(Oa)).level.toString().padStart(2,`0`)}.08`}if(e===`av1`){let e=t*n,i=ka.find(t=>e<=t.maxPictureSize&&r<=t.maxBitrate)??wr(ka);return`av01.0.${i.level.toString().padStart(2,`0`)}${i.tier}.08`}if(e===`prores`){let e=(t*n/2073600)**.95,a=Ma.filter(e=>e.alpha===i),o=a[0].fourCc,s=1/0;for(let{fourCc:t,bitrate:n}of a){let i=Math.abs(n*e-r);i<s&&(s=i,o=t)}return o}throw Rr(e),TypeError(`Unhandled codec '${String(e)}'.`)},Pa=e=>{let t=e.split(`.`),n=Number(t[1]),r=t[2],i=Number(r.slice(0,-1)),a=(n<<5)+i,o=+(r.slice(-1)===`H`),s=Number(t[3]),c=s===8?0:1,l=+(s===12),u=t[4]?Number(t[4]):0,d=t[5]?Number(t[5][0]):1,f=t[5]?Number(t[5][1]):1,p=t[5]?Number(t[5][2]):0;return[129,a,(o<<7)+(c<<6)+(l<<5)+(u<<4)+(d<<3)+(f<<2)+p,0]},Fa=/^pcm-([usf])(\d+)(be)?$/,Ia=e=>{if(H(Sa.includes(e)),e===`ulaw`)return{dataType:`ulaw`,sampleSize:1,littleEndian:!0,silentValue:255};if(e===`alaw`)return{dataType:`alaw`,sampleSize:1,littleEndian:!0,silentValue:213};let t=Fa.exec(e);H(t);let n;n=t[1]===`u`?`unsigned`:t[1]===`s`?`signed`:`float`;let r=Number(t[2])/8,i=t[3]!==`be`;return{dataType:n,sampleSize:r,littleEndian:i,silentValue:e===`pcm-u8`?128:0}},La=e=>e.startsWith(`avc1`)||e.startsWith(`avc3`)?`avc`:e.startsWith(`hev1`)||e.startsWith(`hvc1`)?`hevc`:e===`vp8`?`vp8`:e.startsWith(`vp09`)?`vp9`:e.startsWith(`av01`)?`av1`:Aa.includes(e)?`prores`:e===`mp3`||e===`mp4a.69`||e===`mp4a.6B`||e===`mp4a.6b`||e===`mp4a.40.34`?`mp3`:e.startsWith(`mp4a.40.`)||e===`mp4a.67`?`aac`:e===`opus`?`opus`:e===`vorbis`?`vorbis`:e===`flac`?`flac`:e===`ac-3`||e===`ac3`?`ac3`:e===`ec-3`||e===`eac3`?`eac3`:ja.includes(e)?`dts`:e===`ulaw`?`ulaw`:e===`alaw`?`alaw`:Fa.test(e)?e:e===`webvtt`?`webvtt`:null,Ra=e=>e===`avc`?{avc:{format:`avc`}}:e===`hevc`?{hevc:{format:`hevc`}}:{},za=[`avc1`,`avc3`,`hev1`,`hvc1`,`vp8`,`vp09`,`av01`,...Aa],Ba=/^(avc1|avc3)\.[0-9a-fA-F]{6}$/,Va=/^(hev1|hvc1)\.(?:[ABC]?\d+)\.[0-9a-fA-F]{1,8}\.[LH]\d+(?:\.[0-9a-fA-F]{1,2}){0,6}$/,Ha=/^vp09(?:\.\d{2}){3}(?:(?:\.\d{2}){5})?$/,Ua=/^av01\.\d\.\d{2}[MH]\.\d{2}(?:\.\d\.\d{3}\.\d{2}\.\d{2}\.\d{2}\.\d)?$/,Wa=(e,t)=>{if(!e)throw TypeError(`Video chunk metadata must be provided.`);if(typeof e!=`object`)throw TypeError(`Video chunk metadata must be an object.`);if(!e.decoderConfig)throw TypeError(`Video chunk metadata must include a decoder configuration.`);if(typeof e.decoderConfig!=`object`)throw TypeError(`Video chunk metadata decoder configuration must be an object.`);if(typeof e.decoderConfig.codec!=`string`)throw TypeError(`Video chunk metadata decoder configuration must specify a codec string.`);if(!za.some(t=>e.decoderConfig.codec.startsWith(t)))throw TypeError(`Video chunk metadata decoder configuration codec string must be a valid video codec string as specified in the Mediabunny Codec Registry.`);if(!Number.isInteger(e.decoderConfig.codedWidth)||e.decoderConfig.codedWidth<=0)throw TypeError(`Video chunk metadata decoder configuration must specify a valid codedWidth (positive integer).`);if(!Number.isInteger(e.decoderConfig.codedHeight)||e.decoderConfig.codedHeight<=0)throw TypeError(`Video chunk metadata decoder configuration must specify a valid codedHeight (positive integer).`);if(e.decoderConfig.displayAspectWidth!==void 0&&(!Number.isInteger(e.decoderConfig.displayAspectWidth)||e.decoderConfig.displayAspectWidth<=0))throw TypeError(`Video chunk metadata decoder configuration displayAspectWidth, when defined, must be a positive integer.`);if(e.decoderConfig.displayAspectHeight!==void 0&&(!Number.isInteger(e.decoderConfig.displayAspectHeight)||e.decoderConfig.displayAspectHeight<=0))throw TypeError(`Video chunk metadata decoder configuration displayAspectHeight, when defined, must be a positive integer.`);if(e.decoderConfig.displayAspectWidth!==void 0!=(e.decoderConfig.displayAspectHeight!==void 0))throw TypeError(`Video chunk metadata decoder configuration must specify both displayAspectWidth and displayAspectHeight, or neither.`);if(e.decoderConfig.description!==void 0&&!Pr(e.decoderConfig.description))throw TypeError(`Video chunk metadata decoder configuration description, when defined, must be an ArrayBuffer or an ArrayBuffer view.`);if(e.decoderConfig.colorSpace!==void 0){let{colorSpace:t}=e.decoderConfig;if(typeof t!=`object`)throw TypeError(`Video chunk metadata decoder configuration colorSpace, when provided, must be an object.`);let n=Object.keys(Ar);if(t.primaries!=null&&!n.includes(t.primaries))throw TypeError(`Video chunk metadata decoder configuration colorSpace primaries, when defined, must be one of ${n.join(`, `)}.`);let r=Object.keys(jr);if(t.transfer!=null&&!r.includes(t.transfer))throw TypeError(`Video chunk metadata decoder configuration colorSpace transfer, when defined, must be one of ${r.join(`, `)}.`);let i=Object.keys(Mr);if(t.matrix!=null&&!i.includes(t.matrix))throw TypeError(`Video chunk metadata decoder configuration colorSpace matrix, when defined, must be one of ${i.join(`, `)}.`);if(t.fullRange!=null&&typeof t.fullRange!=`boolean`)throw TypeError(`Video chunk metadata decoder configuration colorSpace fullRange, when defined, must be a boolean.`)}if(e.decoderConfig.codec.startsWith(`avc1`)||e.decoderConfig.codec.startsWith(`avc3`)){if(!Ba.test(e.decoderConfig.codec))throw TypeError(`Video chunk metadata decoder configuration codec string for AVC must be a valid AVC codec string as specified in Section 3.4 of RFC 6381.`)}else if(e.decoderConfig.codec.startsWith(`hev1`)||e.decoderConfig.codec.startsWith(`hvc1`)){if(!Va.test(e.decoderConfig.codec))throw TypeError(`Video chunk metadata decoder configuration codec string for HEVC must be a valid HEVC codec string as specified in Section E.3 of ISO 14496-15.`)}else if(e.decoderConfig.codec.startsWith(`vp8`)){if(e.decoderConfig.codec!==`vp8`)throw TypeError(`Video chunk metadata decoder configuration codec string for VP8 must be "vp8".`)}else if(e.decoderConfig.codec.startsWith(`vp09`)){if(!Ha.test(e.decoderConfig.codec))throw TypeError(`Video chunk metadata decoder configuration codec string for VP9 must be a valid VP9 codec string as specified in Section "Codecs Parameter String" of https://www.webmproject.org/vp9/mp4/.`)}else if(e.decoderConfig.codec.startsWith(`av01`)){if(!Ua.test(e.decoderConfig.codec))throw TypeError(`Video chunk metadata decoder configuration codec string for AV1 must be a valid AV1 codec string as specified in Section "Codecs Parameter String" of https://aomediacodec.github.io/av1-isobmff/.`)}else if(Aa.some(t=>e.decoderConfig.codec.startsWith(t))&&!Aa.some(t=>e.decoderConfig.codec===t))throw TypeError(`Video chunk metadata decoder configuration codec string for ProRes must be one of the valid ProRes four-character codes: ${Aa.join(`, `)}.`);if(t!==null&&La(e.decoderConfig.codec)!==t)throw TypeError(`Video chunk metadata decoder configuration codec string '${e.decoderConfig.codec}' does not fit to the track codec '${t}'.`)},Ga=[`mp4a`,`mp3`,`opus`,`vorbis`,`flac`,`ulaw`,`alaw`,`pcm`,`ac-3`,`ec-3`,`dts`],Ka=(e,t)=>{if(!e)throw TypeError(`Audio chunk metadata must be provided.`);if(typeof e!=`object`)throw TypeError(`Audio chunk metadata must be an object.`);if(!e.decoderConfig)throw TypeError(`Audio chunk metadata must include a decoder configuration.`);if(typeof e.decoderConfig!=`object`)throw TypeError(`Audio chunk metadata decoder configuration must be an object.`);if(typeof e.decoderConfig.codec!=`string`)throw TypeError(`Audio chunk metadata decoder configuration must specify a codec string.`);if(!Ga.some(t=>e.decoderConfig.codec.startsWith(t)))throw TypeError(`Audio chunk metadata decoder configuration codec string must be a valid audio codec string as specified in the Mediabunny Codec Registry.`);if(!Number.isInteger(e.decoderConfig.sampleRate)||e.decoderConfig.sampleRate<=0)throw TypeError(`Audio chunk metadata decoder configuration must specify a valid sampleRate (positive integer).`);if(!Number.isInteger(e.decoderConfig.numberOfChannels)||e.decoderConfig.numberOfChannels<=0)throw TypeError(`Audio chunk metadata decoder configuration must specify a valid numberOfChannels (positive integer).`);if(e.decoderConfig.description!==void 0&&!Pr(e.decoderConfig.description))throw TypeError(`Audio chunk metadata decoder configuration description, when defined, must be an ArrayBuffer or an ArrayBuffer view.`);if(e.decoderConfig.codec.startsWith(`mp4a`)&&e.decoderConfig.codec!==`mp4a.69`&&e.decoderConfig.codec!==`mp4a.6B`&&e.decoderConfig.codec!==`mp4a.6b`){if(![`mp4a.40.2`,`mp4a.40.02`,`mp4a.40.5`,`mp4a.40.05`,`mp4a.40.29`,`mp4a.67`].includes(e.decoderConfig.codec))throw TypeError(`Audio chunk metadata decoder configuration codec string for AAC must be a valid AAC codec string as specified in https://www.w3.org/TR/webcodecs-aac-codec-registration/.`)}else if(e.decoderConfig.codec.startsWith(`mp3`)||e.decoderConfig.codec.startsWith(`mp4a`)){if(e.decoderConfig.codec!==`mp3`&&e.decoderConfig.codec!==`mp4a.69`&&e.decoderConfig.codec!==`mp4a.6B`&&e.decoderConfig.codec!==`mp4a.6b`)throw TypeError(`Audio chunk metadata decoder configuration codec string for MP3 must be "mp3", "mp4a.69" or "mp4a.6B".`)}else if(e.decoderConfig.codec.startsWith(`opus`)){if(e.decoderConfig.codec!==`opus`)throw TypeError(`Audio chunk metadata decoder configuration codec string for Opus must be "opus".`);if(e.decoderConfig.description&&e.decoderConfig.description.byteLength<18)throw TypeError(`Audio chunk metadata decoder configuration description, when specified, is expected to be an Identification Header as specified in Section 5.1 of RFC 7845.`)}else if(e.decoderConfig.codec.startsWith(`vorbis`)){if(e.decoderConfig.codec!==`vorbis`)throw TypeError(`Audio chunk metadata decoder configuration codec string for Vorbis must be "vorbis".`);if(!e.decoderConfig.description)throw TypeError(`Audio chunk metadata decoder configuration for Vorbis must include a description, which is expected to adhere to the format described in https://www.w3.org/TR/webcodecs-vorbis-codec-registration/.`)}else if(e.decoderConfig.codec.startsWith(`flac`)){if(e.decoderConfig.codec!==`flac`)throw TypeError(`Audio chunk metadata decoder configuration codec string for FLAC must be "flac".`);if(!e.decoderConfig.description||e.decoderConfig.description.byteLength<42)throw TypeError(`Audio chunk metadata decoder configuration for FLAC must include a description, which is expected to adhere to the format described in https://www.w3.org/TR/webcodecs-flac-codec-registration/.`)}else if(e.decoderConfig.codec.startsWith(`ac-3`)||e.decoderConfig.codec.startsWith(`ac3`)){if(e.decoderConfig.codec!==`ac-3`)throw TypeError(`Audio chunk metadata decoder configuration codec string for AC-3 must be "ac-3".`)}else if(e.decoderConfig.codec.startsWith(`ec-3`)||e.decoderConfig.codec.startsWith(`eac3`)){if(e.decoderConfig.codec!==`ec-3`)throw TypeError(`Audio chunk metadata decoder configuration codec string for EC-3 must be "ec-3".`)}else if(e.decoderConfig.codec.startsWith(`dts`)){if(!ja.includes(e.decoderConfig.codec))throw TypeError(`Audio chunk metadata decoder configuration codec string for DTS must be one of the following four-character codes: ${ja.join(`, `)}.`)}else if((e.decoderConfig.codec.startsWith(`pcm`)||e.decoderConfig.codec.startsWith(`ulaw`)||e.decoderConfig.codec.startsWith(`alaw`))&&!Sa.includes(e.decoderConfig.codec))throw TypeError(`Audio chunk metadata decoder configuration codec string for PCM must be one of the supported PCM codecs (${Sa.join(`, `)}).`);if(t!==null&&La(e.decoderConfig.codec)!==t)throw TypeError(`Audio chunk metadata decoder configuration codec string '${e.decoderConfig.codec}' does not fit to the track codec '${t}'.`)},qa=e=>{if(!e)throw TypeError(`Subtitle metadata must be provided.`);if(typeof e!=`object`)throw TypeError(`Subtitle metadata must be an object.`);if(!e.config)throw TypeError(`Subtitle metadata must include a config object.`);if(typeof e.config!=`object`)throw TypeError(`Subtitle metadata config must be an object.`);if(typeof e.config.description!=`string`)throw TypeError(`Subtitle metadata config description must be a string.`)},Ja=new Uint8Array,Ya=class e{constructor(e,t,n,r,i=-1,a,o){if(this.data=e,this.type=t,this.timestamp=n,this.duration=r,this.sequenceNumber=i,e===Ja&&a===void 0)throw Error(`Internal error: byteLength must be explicitly provided when constructing metadata-only packets.`);if(a===void 0&&(a=e.byteLength),!(e instanceof Uint8Array))throw TypeError(`data must be a Uint8Array.`);if(t!==`key`&&t!==`delta`)throw TypeError(`type must be either "key" or "delta".`);if(!Number.isFinite(n))throw TypeError(`timestamp must be a number.`);if(!Number.isFinite(r)||r<0)throw TypeError(`duration must be a non-negative number.`);if(!Number.isFinite(i))throw TypeError(`sequenceNumber must be a number.`);if(!Number.isInteger(a)||a<0)throw TypeError(`byteLength must be a non-negative integer.`);if(o!==void 0&&(typeof o!=`object`||!o))throw TypeError(`sideData, when provided, must be an object.`);if(o?.alpha!==void 0&&!(o.alpha instanceof Uint8Array))throw TypeError(`sideData.alpha, when provided, must be a Uint8Array.`);if(o?.alphaByteLength!==void 0&&(!Number.isInteger(o.alphaByteLength)||o.alphaByteLength<0))throw TypeError(`sideData.alphaByteLength, when provided, must be a non-negative integer.`);this.byteLength=a,this.sideData=o??{},this.sideData.alpha&&this.sideData.alphaByteLength===void 0&&(this.sideData.alphaByteLength=this.sideData.alpha.byteLength)}get isMetadataOnly(){return this.data===Ja}get microsecondTimestamp(){return Math.trunc(Jr*this.timestamp)}get microsecondDuration(){return Math.trunc(Jr*this.duration)}toEncodedVideoChunk(){if(this.isMetadataOnly)throw TypeError(`Metadata-only packets cannot be converted to a video chunk.`);if(typeof EncodedVideoChunk>`u`)throw Error(`EncodedVideoChunk is not available in this environment.`);return new EncodedVideoChunk({data:this.data,type:this.type,timestamp:this.microsecondTimestamp,duration:this.microsecondDuration})}alphaToEncodedVideoChunk(e=this.type){if(!this.sideData.alpha)throw TypeError(`This packet does not contain alpha side data.`);if(this.isMetadataOnly)throw TypeError(`Metadata-only packets cannot be converted to a video chunk.`);if(typeof EncodedVideoChunk>`u`)throw Error(`EncodedVideoChunk is not available in this environment.`);return new EncodedVideoChunk({data:this.sideData.alpha,type:e,timestamp:this.microsecondTimestamp,duration:this.microsecondDuration})}toEncodedAudioChunk(){if(this.isMetadataOnly)throw TypeError(`Metadata-only packets cannot be converted to an audio chunk.`);if(typeof EncodedAudioChunk>`u`)throw Error(`EncodedAudioChunk is not available in this environment.`);return new EncodedAudioChunk({data:this.data,type:this.type,timestamp:this.microsecondTimestamp,duration:this.microsecondDuration})}static fromEncodedChunk(t,n){if(!(t instanceof EncodedVideoChunk||t instanceof EncodedAudioChunk))throw TypeError(`chunk must be an EncodedVideoChunk or EncodedAudioChunk.`);let r=new Uint8Array(t.byteLength);return t.copyTo(r),new e(r,t.type,t.timestamp/1e6,(t.duration??0)/1e6,void 0,void 0,n)}clone(t){if(t!==void 0&&(typeof t!=`object`||!t))throw TypeError(`options, when provided, must be an object.`);if(t?.data!==void 0&&!(t.data instanceof Uint8Array))throw TypeError(`options.data, when provided, must be a Uint8Array.`);if(t?.type!==void 0&&t.type!==`key`&&t.type!==`delta`)throw TypeError(`options.type, when provided, must be either "key" or "delta".`);if(t?.timestamp!==void 0&&!Number.isFinite(t.timestamp))throw TypeError(`options.timestamp, when provided, must be a number.`);if(t?.duration!==void 0&&!Number.isFinite(t.duration))throw TypeError(`options.duration, when provided, must be a number.`);if(t?.sequenceNumber!==void 0&&!Number.isFinite(t.sequenceNumber))throw TypeError(`options.sequenceNumber, when provided, must be a number.`);if(t?.sideData!==void 0&&(typeof t.sideData!=`object`||t.sideData===null))throw TypeError(`options.sideData, when provided, must be an object.`);return new e(t?.data??this.data,t?.type??this.type,t?.timestamp??this.timestamp,t?.duration??this.duration,t?.sequenceNumber??this.sequenceNumber,this.byteLength,t?.sideData??this.sideData)}},Xa=e=>{let t=(e.hasVideo?`video/`:e.hasAudio?`audio/`:`application/`)+(e.isQuickTime?`quicktime`:`mp4`);if(e.codecStrings.length>0){let n=[...new Set(e.codecStrings)];t+=`; codecs="${n.join(`, `)}"`}return t},Za=e=>{let t=e.filePos,n=new W(zo(e,9));if(n.readBits(12)!==4095||(n.skipBits(1),n.readBits(2)!==0))return null;let r=n.readBits(1),i=n.readBits(2)+1,a=n.readBits(4);if(a===15)return null;n.skipBits(1);let o=n.readBits(3);if(o===0)throw Error(`ADTS frames with channel configuration 0 are not supported.`);n.skipBits(1),n.skipBits(1),n.skipBits(1),n.skipBits(1);let s=n.readBits(13);n.skipBits(11);let c=n.readBits(2)+1;if(c!==1)throw Error(`ADTS frames with more than one AAC frame are not supported.`);let l=null;return r===1?e.filePos-=2:l=n.readBits(16),{objectType:i,samplingFrequencyIndex:a,channelConfiguration:o,frameLength:s,numberOfAacFrames:c,crcCheck:l,startPos:t}},Qa=te(((e,t)=>{t.exports={}})),$a=function(e,t,n){if(t!=null){if(typeof t!=`object`&&typeof t!=`function`)throw TypeError(`Object expected.`);var r,i;if(n){if(!Symbol.asyncDispose)throw TypeError(`Symbol.asyncDispose is not defined.`);r=t[Symbol.asyncDispose]}if(r===void 0){if(!Symbol.dispose)throw TypeError(`Symbol.dispose is not defined.`);r=t[Symbol.dispose],n&&(i=r)}if(typeof r!=`function`)throw TypeError(`Object not disposable.`);i&&(r=function(){try{i.call(this)}catch(e){return Promise.reject(e)}}),e.stack.push({value:t,dispose:r,async:n})}else n&&e.stack.push({async:!0});return t},eo=(function(e){return function(t){function n(n){t.error=t.hasError?new e(n,t.error,`An error was suppressed during disposal.`):n,t.hasError=!0}var r,i=0;function a(){for(;r=t.stack.pop();)try{if(!r.async&&i===1)return i=0,t.stack.push(r),Promise.resolve().then(a);if(r.dispose){var e=r.dispose.call(r.value);if(r.async)return i|=2,Promise.resolve(e).then(a,function(e){return n(e),a()})}else i|=1}catch(e){n(e)}if(i===1)return t.hasError?Promise.reject(t.error):Promise.resolve();if(t.hasError)throw t.error}return a()}})(typeof SuppressedError==`function`?SuppressedError:function(e,t,n){var r=Error(n);return r.name=`SuppressedError`,r.error=e,r.suppressed=t,r});ai();var to=-1/0,no=-1/0,ro=null;typeof FinalizationRegistry<`u`&&(ro=new FinalizationRegistry(e=>{let t=performance.now();e.type===`video`?(t-to>=1e3&&(hi._error(`A VideoSample was garbage collected without first being closed. For proper resource management, make sure to call close() on all your VideoSamples as soon as you're done using them.`),to=t),typeof VideoFrame<`u`&&e.data instanceof VideoFrame&&e.data.close()):(t-no>=1e3&&(hi._error(`An AudioSample was garbage collected without first being closed. For proper resource management, make sure to call close() on all your AudioSamples as soon as you're done using them.`),no=t),typeof AudioData<`u`&&e.data instanceof AudioData&&e.data.close())}));var io=class{constructor(){this._referenceCount=0,this._lastAllocationBuffer=null}},ao=[`I420`,`I420P10`,`I420P12`,`I420A`,`I420AP10`,`I420AP12`,`I422`,`I422P10`,`I422P12`,`I422A`,`I422AP10`,`I422AP12`,`I444`,`I444P10`,`I444P12`,`I444A`,`I444AP10`,`I444AP12`,`NV12`,`RGBA`,`RGBX`,`BGRA`,`BGRX`],oo=new Set(ao),so=class e{get codedWidth(){return this.visibleRect.width}get codedHeight(){return this.visibleRect.height}get displayWidth(){return this.rotation%180==0?this.squarePixelWidth:this.squarePixelHeight}get displayHeight(){return this.rotation%180==0?this.squarePixelHeight:this.squarePixelWidth}get microsecondTimestamp(){return Math.trunc(Jr*this.timestamp)}get microsecondDuration(){return Math.trunc(Jr*this.duration)}get hasAlpha(){return this.format&&this.format.includes(`A`)}constructor(t,n){if(this._closed=!1,t instanceof ArrayBuffer||typeof SharedArrayBuffer<`u`&&t instanceof SharedArrayBuffer||ArrayBuffer.isView(t)){if(!n||typeof n!=`object`)throw TypeError(`init must be an object.`);if(n.format===void 0||!oo.has(n.format))throw TypeError(`init.format must be one of: `+ao.join(`, `));if(!Number.isInteger(n.codedWidth)||n.codedWidth<=0)throw TypeError(`init.codedWidth must be a positive integer.`);if(!Number.isInteger(n.codedHeight)||n.codedHeight<=0)throw TypeError(`init.codedHeight must be a positive integer.`);if(n.rotation!==void 0&&![0,90,180,270].includes(n.rotation))throw TypeError(`init.rotation, when provided, must be 0, 90, 180, or 270.`);if(!Number.isFinite(n.timestamp))throw TypeError(`init.timestamp must be a number.`);if(n.duration!==void 0&&(!Number.isFinite(n.duration)||n.duration<0))throw TypeError(`init.duration, when provided, must be a non-negative number.`);if(n.layout!==void 0){if(!Array.isArray(n.layout))throw TypeError(`init.layout, when provided, must be an array.`);for(let e of n.layout){if(!e||typeof e!=`object`||Array.isArray(e))throw TypeError(`Each entry in init.layout must be an object.`);if(!Number.isInteger(e.offset)||e.offset<0)throw TypeError(`plane.offset must be a non-negative integer.`);if(!Number.isInteger(e.stride)||e.stride<0)throw TypeError(`plane.stride must be a non-negative integer.`)}}if(n.visibleRect!==void 0&&li(n.visibleRect,`init.visibleRect`),n.displayWidth!==void 0&&(!Number.isInteger(n.displayWidth)||n.displayWidth<=0))throw TypeError(`init.displayWidth, when provided, must be a positive integer.`);if(n.displayHeight!==void 0&&(!Number.isInteger(n.displayHeight)||n.displayHeight<=0))throw TypeError(`init.displayHeight, when provided, must be a positive integer.`);if(n.displayWidth!==void 0!=(n.displayHeight!==void 0))throw TypeError(`init.displayWidth and init.displayHeight must be either both provided or both omitted.`);this.format=n.format,this.rotation=n.rotation??0,this.timestamp=n.timestamp,this.duration=n.duration??0;let e=n.layout??yo(n.format,n.codedWidth,n.codedHeight),r=n.colorSpace??null;r===null&&(r=this.format===`RGBA`||this.format===`RGBX`||this.format===`BGRA`||this.format===`BGRX`?{primaries:`bt709`,transfer:`iec61966-2-1`,matrix:`rgb`,fullRange:!0}:{primaries:`bt709`,transfer:`bt709`,matrix:`bt709`,fullRange:!1}),this.visibleRect={left:n.visibleRect?.left??0,top:n.visibleRect?.top??0,width:n.visibleRect?.width??n.codedWidth,height:n.visibleRect?.height??n.codedHeight},n.displayWidth===void 0?(this.squarePixelWidth=this.visibleRect.width,this.squarePixelHeight=this.visibleRect.height):(this.squarePixelWidth=this.rotation%180==0?n.displayWidth:n.displayHeight,this.squarePixelHeight=this.rotation%180==0?n.displayHeight:n.displayWidth),this._data=n._doNotCopy?Dr(t):Dr(t).slice(),this._layout=e,this.colorSpace=new mo(r)}else if(typeof VideoFrame<`u`&&t instanceof VideoFrame){if(n?.rotation!==void 0&&![0,90,180,270].includes(n.rotation))throw TypeError(`init.rotation, when provided, must be 0, 90, 180, or 270.`);if(n?.timestamp!==void 0&&!Number.isFinite(n?.timestamp))throw TypeError(`init.timestamp, when provided, must be a number.`);if(n?.duration!==void 0&&(!Number.isFinite(n.duration)||n.duration<0))throw TypeError(`init.duration, when provided, must be a non-negative number.`);n?.visibleRect!==void 0&&li(n.visibleRect,`init.visibleRect`),this._data=t,this._layout=null,this.format=t.format,this.visibleRect={left:t.visibleRect?.x??0,top:t.visibleRect?.y??0,width:t.visibleRect?.width??t.codedWidth,height:t.visibleRect?.height??t.codedHeight},this.rotation=n?.rotation??0,this.squarePixelWidth=t.displayWidth,this.squarePixelHeight=t.displayHeight,this.timestamp=n?.timestamp??t.timestamp/1e6,this.duration=n?.duration??(t.duration??0)/1e6,this.colorSpace=new mo(t.colorSpace)}else if(typeof HTMLImageElement<`u`&&t instanceof HTMLImageElement||typeof SVGImageElement<`u`&&t instanceof SVGImageElement||typeof ImageBitmap<`u`&&t instanceof ImageBitmap||typeof HTMLVideoElement<`u`&&t instanceof HTMLVideoElement||typeof HTMLCanvasElement<`u`&&t instanceof HTMLCanvasElement||typeof OffscreenCanvas<`u`&&t instanceof OffscreenCanvas){if(!n||typeof n!=`object`)throw TypeError(`init must be an object.`);if(n.rotation!==void 0&&![0,90,180,270].includes(n.rotation))throw TypeError(`init.rotation, when provided, must be 0, 90, 180, or 270.`);if(!Number.isFinite(n.timestamp))throw TypeError(`init.timestamp must be a number.`);if(n.duration!==void 0&&(!Number.isFinite(n.duration)||n.duration<0))throw TypeError(`init.duration, when provided, must be a non-negative number.`);if(n.visibleRect!==void 0&&li(n.visibleRect,`init.visibleRect`),typeof VideoFrame<`u`)return new e(new VideoFrame(t,{timestamp:Math.trunc(n.timestamp*Jr),duration:Math.trunc((n.duration??0)*Jr)||void 0,visibleRect:n.visibleRect&&{x:n.visibleRect.left,y:n.visibleRect.top,width:n.visibleRect.width,height:n.visibleRect.height}}),n);let r=0,i=0;if(`naturalWidth`in t?(r=t.naturalWidth,i=t.naturalHeight):`videoWidth`in t?(r=t.videoWidth,i=t.videoHeight):`width`in t&&(r=Number(t.width),i=Number(t.height)),!r||!i)throw TypeError(`Could not determine dimensions.`);let a=n.visibleRect??{left:0,top:0,width:r,height:i},o=new OffscreenCanvas(a.width,a.height),s=o.getContext(`2d`,{alpha:Qr(),willReadFrequently:!0});if(!s)throw Error(`OffscreenCanvas must have support for the '2d' context in order to create a VideoSample from this data.`);s.drawImage(t,-a.left,-a.top),this._data=o,this._layout=null,this.format=`RGBX`,this.visibleRect={left:0,top:0,width:a.width,height:a.height},this.squarePixelWidth=a.width,this.squarePixelHeight=a.height,this.rotation=n.rotation??0,this.timestamp=n.timestamp,this.duration=n.duration??0,this.colorSpace=new mo({matrix:`rgb`,primaries:`bt709`,transfer:`iec61966-2-1`,fullRange:!0})}else if(t instanceof io){if(!n||typeof n!=`object`)throw TypeError(`init must be an object.`);if(n.rotation!==void 0&&![0,90,180,270].includes(n.rotation))throw TypeError(`init.rotation, when provided, must be 0, 90, 180, or 270.`);if(!Number.isFinite(n.timestamp))throw TypeError(`init.timestamp must be a number.`);if(n.duration!==void 0&&(!Number.isFinite(n.duration)||n.duration<0))throw TypeError(`init.duration, when provided, must be a non-negative number.`);if(this._data=t,t._referenceCount++,this.format=t.getFormat(),this.format!==null&&!ao.includes(this.format))throw TypeError(`getFormat() must return a VideoSamplePixelFormat or null.`);if(this.visibleRect={left:0,top:0,width:t.getCodedWidth(),height:t.getCodedHeight()},!Number.isInteger(this.visibleRect.width)||this.visibleRect.width<=0)throw TypeError(`getCodedWidth() must return a positive integer.`);if(!Number.isInteger(this.visibleRect.height)||this.visibleRect.height<=0)throw TypeError(`getCodedHeight() must return a positive integer.`);if(this.squarePixelWidth=t.getSquarePixelWidth(),!Number.isInteger(this.squarePixelWidth)||this.squarePixelWidth<=0)throw TypeError(`getSquarePixelWidth() must return a positive integer.`);if(this.squarePixelHeight=t.getSquarePixelHeight(),!Number.isInteger(this.squarePixelHeight)||this.squarePixelHeight<=0)throw TypeError(`getSquarePixelHeight() must return a positive integer.`);this.rotation=n.rotation??0,this.timestamp=n.timestamp,this.duration=n.duration??0,this.colorSpace=t.getColorSpace()}else throw TypeError(`Invalid data type: Must be a BufferSource, CanvasImageSource, or VideoSampleResource.`);this.encodeOptions=n?.encodeOptions??{},this.pixelAspectRatio=ci({num:this.squarePixelWidth*this.codedHeight,den:this.squarePixelHeight*this.codedWidth}),ro?.register(this,{type:`video`,data:this._data},this)}clone(){if(this._closed)throw Error(`VideoSample is closed.`);return H(this._data!==null),this._data instanceof io?new e(this._data,{timestamp:this.timestamp,duration:this.duration,rotation:this.rotation,encodeOptions:this.encodeOptions}):ho(this._data)?new e(this._data.clone(),{timestamp:this.timestamp,duration:this.duration,rotation:this.rotation,encodeOptions:this.encodeOptions}):this._data instanceof Uint8Array?(H(this._layout),new e(this._data,{format:this.format,layout:this._layout,codedWidth:this.codedWidth,codedHeight:this.codedHeight,timestamp:this.timestamp,duration:this.duration,colorSpace:this.colorSpace,rotation:this.rotation,visibleRect:this.visibleRect,displayWidth:this.displayWidth,displayHeight:this.displayHeight,encodeOptions:this.encodeOptions,_doNotCopy:!0})):new e(this._data,{format:this.format,codedWidth:this.codedWidth,codedHeight:this.codedHeight,timestamp:this.timestamp,duration:this.duration,colorSpace:this.colorSpace,rotation:this.rotation,visibleRect:this.visibleRect,displayWidth:this.displayWidth,displayHeight:this.displayHeight,encodeOptions:this.encodeOptions})}close(){this._closed||=(ro?.unregister(this),this._data instanceof io?(this._data._referenceCount--,this._data._referenceCount===0&&this._data.close()):ho(this._data)?this._data.close():this._data=null,!0)}allocationSize(e={}){if(vo(e),this._closed)throw Error(`VideoSample is closed.`);if((e.format??this.format)==null)throw Error(`Cannot get allocation size when format is null.`);return ho(this._data)?this._data.allocationSize(e):xo(this,e).allocationSize}async copyTo(t,n={}){if(!Pr(t))throw TypeError(`destination must be an ArrayBuffer or an ArrayBuffer view.`);if(vo(n),this._closed)throw Error(`VideoSample is closed.`);if((n.format??this.format)==null)throw Error(`Cannot copy video sample data when format is null.`);if(H(this._data!==null),ho(this._data))return this._data.copyTo(t,n);if(n.format&&![`RGBA`,`RGBX`,`BGRA`,`BGRX`].includes(this.format)&&[`RGBA`,`RGBX`,`BGRA`,`BGRX`].includes(n.format)){if(this._data instanceof io){let r={stack:[],error:void 0,hasError:!1};try{let i=$a(r,await this._data.toRgbSample({timestamp:this.timestamp,duration:this.duration,rotation:this.rotation},n.colorSpace??`srgb`),!1);if(!(i instanceof e))throw TypeError(`toRgbSample() must return a VideoSample.`);if(![`RGBA`,`RGBX`,`BGRA`,`BGRX`].includes(i.format))throw Error(`Sample returned by toRgbSample was expected to have an RGB format, got '${i.format}' instead.`);return await i.copyTo(t,n)}catch(e){r.error=e,r.hasError=!0}finally{eo(r)}}else{if(typeof VideoFrame>`u`)throw Error(`For this sample, converting from a non-RGB to an RGB format requires VideoFrame to be defined.`);let e=this.toVideoFrame(),r=await e.copyTo(t,n);return e.close(),r}}let r=xo(this,n);H(this.format);let i=Dr(t);if(i.byteLength<r.allocationSize)throw TypeError(`Destination buffer too small. Required: ${r.allocationSize}, Available: ${i.byteLength}`);let a=bo(this.format),o;if(this._data instanceof io){let e=this._data.getDataPlanes();if(ri(e)&&(e=await e),!Array.isArray(e)||e.some(e=>!(e.data instanceof Uint8Array)||!Number.isInteger(e.stride)||e.stride<0))throw TypeError(`getDataPlanes() must return an array of objects with a Uint8Array "data" property and a non-negative integer "stride" property.`);o=e}else if(this._data instanceof Uint8Array)H(this._layout),H(this._layout.length===a.length),o=this._layout.map((e,t)=>{let n=Math.ceil(this.codedHeight/a[t].heightDivisor);return{data:this._data.subarray(e.offset,e.offset+e.stride*n),stride:e.stride}});else{let e=this._data.getContext(`2d`);H(e),o=[{data:Dr(e.getImageData(0,0,this.codedWidth,this.codedHeight).data),stride:4*this.codedWidth}]}let s=[],c=a.length;for(let e=0;e<c;e++){let t=r.computedLayouts[e],n=o[e].stride,a=o[e].data,c=t.sourceTop*n;c+=t.sourceLeftBytes;let l=t.destinationOffset,u=t.sourceWidthBytes,d={offset:l,stride:t.destinationStride};for(let e=0;e<t.sourceHeight;e++){if(c+u>a.byteLength)throw Error(`Source buffer OOB read.`);if(l+u>i.byteLength)throw Error(`Destination buffer OOB write.`);let e=a.subarray(c,c+u);i.set(e,l),c+=n,l+=t.destinationStride}s.push(d)}if(n.format!==void 0){let e=this.format.startsWith(`RGB`)!==n.format.startsWith(`RGB`),t=this.format.includes(`X`)&&n.format.includes(`A`);if(e||t)for(let n=0;n<r.allocationSize;n+=4){if(e){let e=i[n],t=i[n+2];i[n]=t,i[n+2]=e}t&&(i[n+3]=255)}}return s}toVideoFrame(){if(this._closed)throw Error(`VideoSample is closed.`);if(H(this._data!==null),this._data instanceof io){if(this.format===null)throw Error(`Cannot convert a VideoSampleResource-backed VideoSample to VideoFrame if format is null.`);let e=this._data.getDataPlanes();if(ri(e))throw Error(`Cannot convert a VideoSampleResource-backed VideoSample to VideoFrame if getDataPlanes() returns a promise.`);let t=e.reduce((e,t)=>e+t.data.byteLength,0),n=new Uint8Array(t),r=0,i=[];for(let t of e)n.set(t.data,r),i.push(r),r+=t.data.byteLength;return new VideoFrame(n,{format:this.format,layout:e.map((e,t)=>({offset:i[t],stride:e.stride})),codedWidth:this.codedWidth,codedHeight:this.codedHeight,timestamp:this.microsecondTimestamp,duration:this.microsecondDuration,colorSpace:this.colorSpace,visibleRect:this.visibleRect,displayWidth:this.squarePixelWidth,displayHeight:this.squarePixelHeight})}return ho(this._data)?new VideoFrame(this._data,{timestamp:this.microsecondTimestamp,duration:this.microsecondDuration||void 0}):this._data instanceof Uint8Array?(H(this._layout),new VideoFrame(this._data,{format:this.format,codedWidth:this.codedWidth,codedHeight:this.codedHeight,layout:this._layout,timestamp:this.microsecondTimestamp,duration:this.microsecondDuration||void 0,colorSpace:this.colorSpace,visibleRect:this.visibleRect,displayWidth:this.squarePixelWidth,displayHeight:this.squarePixelHeight})):new VideoFrame(this._data,{timestamp:this.microsecondTimestamp,duration:this.microsecondDuration||void 0})}draw(e,t,n,r,i,a,o,s,c){let l=0,u=0,d=this.displayWidth,f=this.displayHeight,p=0,m=0,h=this.displayWidth,g=this.displayHeight;if(a===void 0?(p=t,m=n,r!==void 0&&(h=r,g=i)):(l=t,u=n,d=r,f=i,p=a,m=o,s===void 0?(h=d,g=f):(h=s,g=c)),!(typeof CanvasRenderingContext2D<`u`&&e instanceof CanvasRenderingContext2D||typeof OffscreenCanvasRenderingContext2D<`u`&&e instanceof OffscreenCanvasRenderingContext2D))throw TypeError(`context must be a CanvasRenderingContext2D or OffscreenCanvasRenderingContext2D.`);if(!Number.isFinite(l))throw TypeError(`sx must be a number.`);if(!Number.isFinite(u))throw TypeError(`sy must be a number.`);if(!Number.isFinite(d)||d<0)throw TypeError(`sWidth must be a non-negative number.`);if(!Number.isFinite(f)||f<0)throw TypeError(`sHeight must be a non-negative number.`);if(!Number.isFinite(p))throw TypeError(`dx must be a number.`);if(!Number.isFinite(m))throw TypeError(`dy must be a number.`);if(!Number.isFinite(h)||h<0)throw TypeError(`dWidth must be a non-negative number.`);if(!Number.isFinite(g)||g<0)throw TypeError(`dHeight must be a non-negative number.`);if(this._closed)throw Error(`VideoSample is closed.`);({sx:l,sy:u,sWidth:d,sHeight:f}=this._rotateSourceRegion(l,u,d,f,this.rotation));let _=this.toCanvasImageSource();e.save();let v=p+h/2,y=m+g/2;e.translate(v,y),e.rotate(this.rotation*Math.PI/180);let b=this.rotation%180==0?1:h/g;e.scale(1/b,b),e.drawImage(_,l,u,d,f,-h/2,-g/2,h,g),e.restore()}drawWithFit(e,t){if(!(typeof CanvasRenderingContext2D<`u`&&e instanceof CanvasRenderingContext2D||typeof OffscreenCanvasRenderingContext2D<`u`&&e instanceof OffscreenCanvasRenderingContext2D))throw TypeError(`context must be a CanvasRenderingContext2D or OffscreenCanvasRenderingContext2D.`);if(!t||typeof t!=`object`)throw TypeError(`options must be an object.`);if(![`fill`,`contain`,`cover`].includes(t.fit))throw TypeError(`options.fit must be 'fill', 'contain', or 'cover'.`);if(t.rotation!==void 0&&![0,90,180,270].includes(t.rotation))throw TypeError(`options.rotation, when provided, must be 0, 90, 180, or 270.`);t.crop!==void 0&&_o(t.crop,`options.`);let n=e.canvas.width,r=e.canvas.height,i=t.rotation??this.rotation,[a,o]=i%180==0?[this.squarePixelWidth,this.squarePixelHeight]:[this.squarePixelHeight,this.squarePixelWidth],s=t.crop;s&&=go(s,a,o);let c,l,u,d,{sx:f,sy:p,sWidth:m,sHeight:h}=this._rotateSourceRegion(t.crop?.left??0,t.crop?.top??0,t.crop?.width??a,t.crop?.height??o,i);if(t.fit===`fill`)c=0,l=0,u=n,d=r;else{let[e,i]=t.crop?[t.crop.width,t.crop.height]:[a,o],s=t.fit===`contain`?Math.min(n/e,r/i):Math.max(n/e,r/i);u=e*s,d=i*s,c=(n-u)/2,l=(r-d)/2}e.save();let g=i%180==0?1:u/d;e.translate(n/2,r/2),e.rotate(i*Math.PI/180),e.scale(1/g,g),e.translate(-n/2,-r/2),e.drawImage(this.toCanvasImageSource(),f,p,m,h,c,l,u,d),e.restore()}_rotateSourceRegion(e,t,n,r,i){return i===90?[e,t,n,r]=[t,this.squarePixelHeight-e-n,r,n]:i===180?[e,t]=[this.squarePixelWidth-e-n,this.squarePixelHeight-t-r]:i===270&&([e,t,n,r]=[this.squarePixelWidth-t-r,e,r,n]),{sx:e,sy:t,sWidth:n,sHeight:r}}_drawWithFitAndMipmapping(e,t,n){let r=e.width,i=e.height,[a,o]=n.rotation%180==0?[this.squarePixelWidth,this.squarePixelHeight]:[this.squarePixelHeight,this.squarePixelWidth],s=n.crop?n.crop.width:a,c=n.crop?n.crop.height:o,l=0;2*r<s&&2*i<c&&(l=Math.floor(Math.log2(Math.min(s/r,c/i))));let u=r*2**l,d=i*2**l,{canvas:f,context:p,isNew:m}=l>0?po(u,d):{canvas:e,context:t,isNew:n.targetIsFresh};p.imageSmoothingQuality=`high`,n.fillBlack?(p.fillStyle=`black`,p.fillRect(0,0,u,d)):m||p.clearRect(0,0,u,d),this.drawWithFit(p,{fit:n.fit,rotation:n.rotation,crop:n.crop}),p.globalCompositeOperation=`copy`;for(let e=l;e>1;e--){let t=r*2**e,n=i*2**e;p.drawImage(f,0,0,t,n,0,0,t/2,n/2)}p.globalCompositeOperation=`source-over`,l>0&&(t.imageSmoothingQuality=`high`,t.globalCompositeOperation=`copy`,t.drawImage(f,0,0,2*r,2*i,0,0,r,i),t.globalCompositeOperation=`source-over`)}toCanvasImageSource(){if(this._closed)throw Error(`VideoSample is closed.`);if(H(this._data!==null),this._data instanceof io||this._data instanceof Uint8Array){let e=this.toVideoFrame();return queueMicrotask(()=>e.close()),e}return this._data}async transform(t){if(!t||typeof t!=`object`)throw TypeError(`options must be an object.`);if(t.width!==void 0&&(!Number.isInteger(t.width)||t.width<=0))throw TypeError(`options.width, when provided, must be a positive integer.`);if(t.height!==void 0&&(!Number.isInteger(t.height)||t.height<=0))throw TypeError(`options.height, when provided, must be a positive integer.`);if(t.roundDimensionsTo!==void 0&&(!Number.isInteger(t.roundDimensionsTo)||t.roundDimensionsTo<=0))throw TypeError(`options.roundDimensionsTo, when provided, must be a positive integer.`);if(t.fit!==void 0&&![`fill`,`contain`,`cover`].includes(t.fit))throw TypeError(`options.fit, when provided, must be one of "fill", "contain", or "cover".`);if(t.width!==void 0&&t.height!==void 0&&t.fit===void 0)throw TypeError(`When both options.width and options.height are provided, options.fit must also be provided.`);if(t.rotate!==void 0&&![0,90,180,270].includes(t.rotate))throw TypeError(`options.rotate, when provided, must be 0, 90, 180 or 270.`);if(t.crop!==void 0&&_o(t.crop,`options.`),t.alpha!==void 0&&![`keep`,`discard`].includes(t.alpha))throw TypeError(`options.alpha, when provided, must be 'keep' or 'discard'.`);let n=Cr(this.rotation+(t.rotate??0)),[r,i]=n%180==0?[this.squarePixelWidth,this.squarePixelHeight]:[this.squarePixelHeight,this.squarePixelWidth],a=t.crop;a&&=go(a,r,i);let o=a?a.width:r,s=a?a.height:i,c=o/s,l,u;t.width!==void 0&&t.height===void 0?(l=t.width,u=l/c):t.width===void 0&&t.height!==void 0?(u=t.height,l=u*c):t.width!==void 0&&t.height!==void 0?(l=t.width,u=t.height):(l=o,u=s),l=Ur(l,t.roundDimensionsTo??1),u=Ur(u,t.roundDimensionsTo??1);let d={width:l,height:u,fit:t.fit??`fill`,rotation:n,crop:a??{left:0,top:0,width:r,height:i},alpha:t.alpha??`keep`};for(let e of co){let t=e(this,d);if(ri(t)&&(t=await t),t!==null)return t}let{canvas:f,context:p,isNew:m}=po(d.width,d.height);return this._drawWithFitAndMipmapping(f,p,{fit:d.fit,rotation:d.rotation,crop:d.crop,targetIsFresh:m,fillBlack:d.alpha===`discard`}),new e(f,{timestamp:this.timestamp,duration:this.duration,rotation:0})}setRotation(e){if(![0,90,180,270].includes(e))throw TypeError(`newRotation must be 0, 90, 180, or 270.`);this.rotation=e}setTimestamp(e){if(!Number.isFinite(e))throw TypeError(`newTimestamp must be a number.`);this.timestamp=e}setDuration(e){if(!Number.isFinite(e)||e<0)throw TypeError(`newDuration must be a non-negative number.`);this.duration=e}setEncodeOptions(e){if(!e||typeof e!=`object`)throw TypeError(`newEncodeOptions must be an object.`);this.encodeOptions=e}[Symbol.dispose](){this.close()}},co=[],lo=3,uo=[],fo=0,po=(e,t)=>{for(let n of uo)if(n.canvas.width===e&&n.canvas.height===t)return n.age=fo++,{canvas:n.canvas,context:n.context,isNew:!1};let n;if(typeof OffscreenCanvas<`u`)n=new OffscreenCanvas(e,t);else{if(typeof window>`u`||typeof document>`u`)throw Error(`Cannot transform VideoSamples in this environment. Either run in an environment with OffscreenCanvas or HTMLCanvasElement, or supply a custom VideoSample transformer using registerVideoSampleTransformer().`);n=document.createElement(`canvas`),n.width=e,n.height=t}let r=n.getContext(`2d`,{alpha:!0,willReadFrequently:!1});if(!r)throw Error(`The '2d' canvas context is required to transform VideoSamples. Register a custom transformer using registerVideoSampleTransformer to work around this limitation.`);return uo.length>=lo&&uo.splice(si(uo,e=>e.age),1),uo.push({canvas:n,context:r,age:fo++}),{canvas:n,context:r,isNew:!0}},mo=class{constructor(e){if(e!==void 0){if(!e||typeof e!=`object`)throw TypeError(`init.colorSpace, when provided, must be an object.`);let t=Object.keys(Ar);if(e.primaries!=null&&!t.includes(e.primaries))throw TypeError(`init.colorSpace.primaries, when provided, must be one of ${t.join(`, `)}.`);let n=Object.keys(jr);if(e.transfer!=null&&!n.includes(e.transfer))throw TypeError(`init.colorSpace.transfer, when provided, must be one of ${n.join(`, `)}.`);let r=Object.keys(Mr);if(e.matrix!=null&&!r.includes(e.matrix))throw TypeError(`init.colorSpace.matrix, when provided, must be one of ${r.join(`, `)}.`);if(e.fullRange!=null&&typeof e.fullRange!=`boolean`)throw TypeError(`init.colorSpace.fullRange, when provided, must be a boolean.`)}this.primaries=e?.primaries??null,this.transfer=e?.transfer??null,this.matrix=e?.matrix??null,this.fullRange=e?.fullRange??null}toJSON(){return{primaries:this.primaries,transfer:this.transfer,matrix:this.matrix,fullRange:this.fullRange}}},ho=e=>typeof VideoFrame<`u`&&e instanceof VideoFrame,go=(e,t,n)=>{let r=Math.min(e.left,t),i=Math.min(e.top,n),a=Math.min(e.width,t-r),o=Math.min(e.height,n-i);return H(a>=0),H(o>=0),{left:r,top:i,width:a,height:o}},_o=(e,t)=>{if(!e||typeof e!=`object`)throw TypeError(t+`crop, when provided, must be an object.`);if(!Number.isInteger(e.left)||e.left<0)throw TypeError(t+`crop.left must be a non-negative integer.`);if(!Number.isInteger(e.top)||e.top<0)throw TypeError(t+`crop.top must be a non-negative integer.`);if(!Number.isInteger(e.width)||e.width<0)throw TypeError(t+`crop.width must be a non-negative integer.`);if(!Number.isInteger(e.height)||e.height<0)throw TypeError(t+`crop.height must be a non-negative integer.`)},vo=e=>{if(!e||typeof e!=`object`)throw TypeError(`options must be an object.`);if(e.colorSpace!==void 0&&![`display-p3`,`srgb`].includes(e.colorSpace))throw TypeError(`options.colorSpace, when provided, must be 'display-p3' or 'srgb'.`);if(e.format!==void 0&&typeof e.format!=`string`)throw TypeError(`options.format, when provided, must be a string.`);if(e.layout!==void 0){if(!Array.isArray(e.layout))throw TypeError(`options.layout, when provided, must be an array.`);for(let t of e.layout){if(!t||typeof t!=`object`)throw TypeError(`Each entry in options.layout must be an object.`);if(!Number.isInteger(t.offset)||t.offset<0)throw TypeError(`plane.offset must be a non-negative integer.`);if(!Number.isInteger(t.stride)||t.stride<0)throw TypeError(`plane.stride must be a non-negative integer.`)}}if(e.rect!==void 0){if(!e.rect||typeof e.rect!=`object`)throw TypeError(`options.rect, when provided, must be an object.`);if(e.rect.x!==void 0&&(!Number.isInteger(e.rect.x)||e.rect.x<0))throw TypeError(`options.rect.x, when provided, must be a non-negative integer.`);if(e.rect.y!==void 0&&(!Number.isInteger(e.rect.y)||e.rect.y<0))throw TypeError(`options.rect.y, when provided, must be a non-negative integer.`);if(e.rect.width!==void 0&&(!Number.isInteger(e.rect.width)||e.rect.width<0))throw TypeError(`options.rect.width, when provided, must be a non-negative integer.`);if(e.rect.height!==void 0&&(!Number.isInteger(e.rect.height)||e.rect.height<0))throw TypeError(`options.rect.height, when provided, must be a non-negative integer.`)}},yo=(e,t,n)=>{let r=bo(e),i=[],a=0;for(let e of r){let r=Math.ceil(t/e.widthDivisor),o=Math.ceil(n/e.heightDivisor),s=r*e.sampleBytes,c=s*o;i.push({offset:a,stride:s}),a+=c}return i},bo=e=>{let t=(e,t,n,r,i)=>{let a=[{sampleBytes:e,widthDivisor:1,heightDivisor:1},{sampleBytes:t,widthDivisor:n,heightDivisor:r},{sampleBytes:t,widthDivisor:n,heightDivisor:r}];return i&&a.push({sampleBytes:e,widthDivisor:1,heightDivisor:1}),a};switch(e){case`I420`:return t(1,1,2,2,!1);case`I420P10`:case`I420P12`:return t(2,2,2,2,!1);case`I420A`:return t(1,1,2,2,!0);case`I420AP10`:case`I420AP12`:return t(2,2,2,2,!0);case`I422`:return t(1,1,2,1,!1);case`I422P10`:case`I422P12`:return t(2,2,2,1,!1);case`I422A`:return t(1,1,2,1,!0);case`I422AP10`:case`I422AP12`:return t(2,2,2,1,!0);case`I444`:return t(1,1,1,1,!1);case`I444P10`:case`I444P12`:return t(2,2,1,1,!1);case`I444A`:return t(1,1,1,1,!0);case`I444AP10`:case`I444AP12`:return t(2,2,1,1,!0);case`NV12`:return[{sampleBytes:1,widthDivisor:1,heightDivisor:1},{sampleBytes:2,widthDivisor:2,heightDivisor:2}];case`RGBA`:case`RGBX`:case`BGRA`:case`BGRX`:return[{sampleBytes:4,widthDivisor:1,heightDivisor:1}];default:Rr(e),H(!1)}},xo=(e,t)=>{let n={left:0,top:0,width:e.codedWidth,height:e.codedHeight},r=t.rect,i=So(n,r,e.codedWidth,e.codedHeight,e.format),a=t.layout,o;if(!t.format||t.format===e.format)o=e.format;else if([`RGBA`,`RGBX`,`BGRA`,`BGRX`].includes(t.format))o=t.format;else throw Error(`NotSupportedError: Invalid destination format.`);return wo(i,o,a)},So=(e,t,n,r,i)=>{let a={...e};if(t!==void 0){if(t.width===0||t.height===0)throw TypeError(`visibleRect dimensions cannot be zero.`);if((t.x||0)+(t.width||0)>n)throw TypeError(`visibleRect exceeds codedWidth.`);if((t.y||0)+(t.height||0)>r)throw TypeError(`visibleRect exceeds codedHeight.`);a.x=t.x||0,a.y=t.y||0,a.width=t.width||0,a.height=t.height||0}if(!Co(i,a))throw TypeError(`visibleRect alignment is invalid for the format.`);return a},Co=(e,t)=>{if(e===null)return!0;let n=bo(e);for(let e=0;e<n.length;e++){let r=n[e],i=r.widthDivisor,a=r.heightDivisor;if((t.x||0)%i!==0||(t.y||0)%a!==0)return!1}return!0},wo=(e,t,n)=>{let r=bo(t),i=r.length;if(n!==void 0&&n.length!==i)throw TypeError(`Layout must have ${i} planes.`);let a=0,o=[],s=[];for(let t=0;t<i;t++){let i=r[t],c=i.sampleBytes,l=i.widthDivisor,u=i.heightDivisor,d={destinationOffset:0,destinationStride:0,sourceTop:0,sourceHeight:0,sourceLeftBytes:0,sourceWidthBytes:0};if(d.sourceTop=Math.ceil(Math.trunc(e.y||0)/u),d.sourceHeight=Math.ceil(Math.trunc(e.height||0)/u),d.sourceLeftBytes=Math.floor(Math.trunc(e.x||0)/l)*c,d.sourceWidthBytes=Math.floor(Math.trunc(e.width||0)/l)*c,n!==void 0){let e=n[t];if(e.stride<d.sourceWidthBytes)throw TypeError(`Stride for plane ${t} is too small.`);d.destinationOffset=e.offset,d.destinationStride=e.stride}else d.destinationOffset=a,d.destinationStride=d.sourceWidthBytes;let f=d.destinationStride*d.sourceHeight+d.destinationOffset;if(f>4294967295)throw TypeError(`Allocation size exceeds limit.`);s.push(f),a=Math.max(a,f);for(let e=0;e<t;e++){let n=o[e];if(!(s[t]<=n.destinationOffset||s[e]<=d.destinationOffset))throw TypeError(`Planes overlap.`)}o.push(d)}return{allocationSize:a,computedLayouts:o}},To=e=>{if(!e||typeof e!=`object`)throw TypeError(`Encoding config must be an object.`);if(!xa.includes(e.codec))throw TypeError(`Invalid video codec '${e.codec}'. Must be one of: ${xa.join(`, `)}.`);let t=e.bitrate;if(e.quality===void 0&&t===void 0)throw TypeError(`config.quality must be provided.`);if(e.quality!==void 0&&t!==void 0)throw TypeError(`config.quality and config.bitrate cannot both be provided.`);if(e.quality!==void 0&&!(e.quality instanceof Oo))throw TypeError(`config.quality, when provided, must be a Quality.`);if(t!==void 0&&!(t instanceof Oo)&&(!Number.isInteger(t)||t<=0))throw TypeError(`config.bitrate, when provided, must be a positive integer or a quality.`);if(e.keyFrameInterval!==void 0&&(!Number.isFinite(e.keyFrameInterval)||e.keyFrameInterval<0))throw TypeError(`config.keyFrameInterval, when provided, must be a non-negative number.`);if(e.sizeChangeBehavior!==void 0&&![`deny`,`passThrough`,`fill`,`contain`,`cover`].includes(e.sizeChangeBehavior))throw TypeError(`config.sizeChangeBehavior, when provided, must be 'deny', 'passThrough', 'fill', 'contain' or 'cover'.`);if(e.transform!==void 0){if(typeof e.transform!=`object`||!e.transform)throw TypeError(`config.transform, when provided, must be an object.`);if(e.transform.width!==void 0&&(!Number.isInteger(e.transform.width)||e.transform.width<=0))throw TypeError(`config.transform.width, when provided, must be a positive integer.`);if(e.transform.height!==void 0&&(!Number.isInteger(e.transform.height)||e.transform.height<=0))throw TypeError(`config.transform.height, when provided, must be a positive integer.`);if(e.transform.fit!==void 0&&![`fill`,`contain`,`cover`].includes(e.transform.fit))throw TypeError(`config.transform.fit, when provided, must be one of "fill", "contain", or "cover".`);if(e.transform.width!==void 0&&e.transform.height!==void 0&&e.transform.fit===void 0&&![`fill`,`contain`,`cover`].includes(e.sizeChangeBehavior))throw TypeError(`When both config.transform.width and config.transform.height are provided, config.transform.fit must also be provided.`);if(e.transform.fit!==void 0&&[`fill`,`contain`,`cover`].includes(e.sizeChangeBehavior)&&e.transform.fit!==e.sizeChangeBehavior)throw TypeError(`config.transform.fit, when provided, cannot differ from config.sizeChangeBehavior when config.sizeChangeBehavior is 'fill', 'contain' or 'cover', as sizeChangeBehavior already determines the fitting algorithm.`);if(e.transform.rotate!==void 0&&![0,90,180,270].includes(e.transform.rotate))throw TypeError(`config.transform.rotate, when provided, must be 0, 90, 180 or 270.`);if(e.transform.crop!==void 0&&_o(e.transform.crop,`config.transform.`),e.transform.process!==void 0&&typeof e.transform.process!=`function`)throw TypeError(`config.transform.process, when provided, must be a function.`);if(e.transform.frameRate!==void 0&&(!Number.isFinite(e.transform.frameRate)||e.transform.frameRate<=0))throw TypeError(`config.transform.frameRate, when provided, must be a finite positive number.`);if(e.transform.force!==void 0&&typeof e.transform.force!=`boolean`)throw TypeError(`config.transform.force, when provided, must be a boolean.`)}if(e.onEncodedPacket!==void 0&&typeof e.onEncodedPacket!=`function`)throw TypeError(`config.onEncodedPacket, when provided, must be a function.`);if(e.onEncoderConfig!==void 0&&typeof e.onEncoderConfig!=`function`)throw TypeError(`config.onEncoderConfig, when provided, must be a function.`);if(e.onEncodedSample!==void 0&&typeof e.onEncodedSample!=`function`)throw TypeError(`config.onEncodedSample, when provided, must be a function.`);Eo(e.codec,e)},Eo=(e,t)=>{if(!t||typeof t!=`object`)throw TypeError(`Encoding options must be an object.`);if(t.alpha!==void 0&&![`discard`,`keep`].includes(t.alpha))throw TypeError(`options.alpha, when provided, must be 'discard' or 'keep'.`);let n=t.bitrateMode;if(n!==void 0&&![`constant`,`variable`].includes(n))throw TypeError(`bitrateMode, when provided, must be 'constant' or 'variable'.`);if(t.latencyMode!==void 0&&![`quality`,`realtime`].includes(t.latencyMode))throw TypeError(`latencyMode, when provided, must be 'quality' or 'realtime'.`);if(t.fullCodecString!==void 0&&typeof t.fullCodecString!=`string`)throw TypeError(`fullCodecString, when provided, must be a string.`);if(t.fullCodecString!==void 0&&La(t.fullCodecString)!==e)throw TypeError(`fullCodecString, when provided, must be a string that matches the specified codec (${e}).`);if(t.hardwareAcceleration!==void 0&&![`no-preference`,`prefer-hardware`,`prefer-software`].includes(t.hardwareAcceleration))throw TypeError(`hardwareAcceleration, when provided, must be 'no-preference', 'prefer-hardware' or 'prefer-software'.`);if(t.scalabilityMode!==void 0&&typeof t.scalabilityMode!=`string`)throw TypeError(`scalabilityMode, when provided, must be a string.`);if(t.contentHint!==void 0&&typeof t.contentHint!=`string`)throw TypeError(`contentHint, when provided, must be a string.`)},Do=e=>{let t=e.bitrateMode,n=e.quality._toVideoRateControl(e.codec,e.width,e.height,t),r=(t,n,r)=>({codec:e.fullCodecString??Na(e.codec,e.width,e.height,r,e.alpha===`keep`),width:e.width,height:e.height,displayWidth:e.squarePixelWidth,displayHeight:e.squarePixelHeight,bitrate:t,bitrateMode:n,alpha:e.alpha??`discard`,framerate:e.framerate,latencyMode:e.latencyMode,hardwareAcceleration:e.hardwareAcceleration,scalabilityMode:e.scalabilityMode,contentHint:e.contentHint,...Ra(e.codec)}),i=[];return n.quantizer!==null&&i.push({config:r(void 0,`quantizer`,n.bitrate),quantizer:n.quantizer}),n.bitrateMode!==`quantizer`&&i.push({config:r(n.bitrate,n.bitrateMode,n.bitrate),quantizer:null}),H(i.length>0),i},Oo=class{constructor(e){if((typeof e==`number`||typeof e==`string`)&&(e={quality:e}),!e||typeof e!=`object`)throw TypeError(`options must be an object.`);if(e.bitrateMode!==void 0&&![`constant`,`variable`].includes(e.bitrateMode))throw TypeError(`options.bitrateMode, when provided, must be 'constant' or 'variable'.`);if(`quality`in e){if(typeof e.quality==`string`?!(e.quality in ko):typeof e.quality!=`number`||Number.isNaN(e.quality))throw TypeError(`options.quality must be a number, or one of 'very-low', 'low', 'medium', 'high' or 'very-high'.`);if(e.preferBitrate!==void 0&&typeof e.preferBitrate!=`boolean`)throw TypeError(`options.preferBitrate, when provided, must be a boolean.`);if(`bitrate`in e||`quantizer`in e)throw TypeError(`options.quality cannot be combined with options.bitrate or options.quantizer.`);this._quality=typeof e.quality==`string`?ko[e.quality]:e.quality,this._preferBitrate=e.preferBitrate??!1,this._bitrate=void 0,this._quantizer=void 0}else{if(e.bitrate!==void 0&&(!Number.isInteger(e.bitrate)||e.bitrate<=0))throw TypeError(`options.bitrate, when provided, must be a positive integer.`);if(e.quantizer!==void 0&&(!Number.isInteger(e.quantizer)||e.quantizer<0))throw TypeError(`options.quantizer, when provided, must be a non-negative integer.`);if(e.bitrate===void 0&&e.quantizer===void 0)throw TypeError(`At least one of options.bitrate or options.quantizer must be set.`);if(`preferBitrate`in e)throw TypeError(`options.preferBitrate can only be combined with options.quality.`);this._quality=void 0,this._preferBitrate=!1,this._bitrate=e.bitrate,this._quantizer=e.quantizer}this._bitrateMode=e.bitrateMode}_toVideoRateControl(e,t,n,r){let i=Ao[e],a=null,o=this._bitrateMode??r??`variable`;if(this._quantizer!==void 0){if(!i){if(this._bitrate===void 0)throw Error(`Codec '${e}' does not support quantizer-based encoding. Provide a bitrate in the Quality to define a fallback.`)}else if(this._quantizer<i.min||this._quantizer>i.max){if(this._bitrate===void 0)throw Error(`Quantizer ${this._quantizer} is out of range for codec '${e}'; must be between ${i.min} and ${i.max}.`)}else a=this._quantizer,this._bitrate===void 0&&(o=`quantizer`)}else this._bitrate===void 0&&i&&!this._preferBitrate&&(H(this._quality!==void 0),a=Vr(Math.round(Hr(i.worst,i.best,this._quality)),i.min,i.max));let s;if(this._bitrate!==void 0)s=this._bitrate;else{let r=this._quality;r===void 0&&(H(a!==null&&i),r=Vr((a-i.worst)/(i.best-i.worst),0,1)),s=Mo(e,t,n,jo(r))}return{quantizer:a,bitrate:s,bitrateMode:o}}_toVideoBitrate(e,t,n){return this._bitrate===void 0?(H(this._quality!==void 0),Mo(e,t,n,jo(this._quality))):this._bitrate}_toAudioBitrate(e){if(Sa.includes(e)||e===`flac`)return;if(this._bitrate!==void 0)return this._bitrate;if(this._quality===void 0)throw Error(`This Quality defines neither a quality level nor a bitrate and therefore cannot be used for audio encoding.`);let t=jo(this._quality),n={aac:128e3,opus:64e3,mp3:16e4,vorbis:64e3,ac3:384e3,eac3:192e3,dts:768e3}[e];if(!n)throw Error(`Unhandled codec: ${e}`);let r=n*t;return e===`aac`?r=[96e3,128e3,16e4,192e3].reduce((e,t)=>Math.abs(t-r)<Math.abs(e-r)?t:e):e===`opus`||e===`vorbis`?r=Math.max(6e3,r):e===`mp3`&&(r=[8e3,16e3,24e3,32e3,4e4,48e3,64e3,8e4,96e3,112e3,128e3,16e4,192e3,224e3,256e3,32e4].reduce((e,t)=>Math.abs(t-r)<Math.abs(e-r)?t:e)),Math.round(r/1e3)*1e3}},ko={"very-low":0,low:.25,medium:.5,high:.75,"very-high":1},Ao={avc:{min:0,max:51,worst:41,best:16},hevc:{min:0,max:51,worst:41,best:16},vp9:{min:0,max:63,worst:52,best:20},av1:{min:0,max:255,worst:208,best:80}},jo=e=>.3*Math.exp(2.5538*e),Mo=(e,t,n,r)=>{let i=t*n,a=3e6,o=a*(i/2073600)**.95*{avc:1,hevc:.6,vp9:.6,av1:.4,vp8:1.2,prores:22e7/a}[e]*r;return Math.ceil(o/1e3)*1e3},No=(e,t)=>{if(e===`avc`)return{avc:{quantizer:t}};if(e===`hevc`)return{hevc:{quantizer:t}};if(e===`vp9`)return{vp9:{quantizer:t}};if(e===`av1`)return{av1:{quantizer:t}};H(!1)},Po=new Oo(`high`),Fo=(e,t)=>{if(e!==void 0)return e;if(t!==void 0)return t instanceof Oo?t:new Oo({bitrate:t})},Io=[],Lo=class e{constructor(e,t,n,r,i){this.bytes=e,this.view=t,this.offset=n,this.start=r,this.end=i,this.bufferPos=r-n}static tempFromBytes(t){return new e(t,Or(t),0,0,t.length)}get length(){return this.end-this.start}get filePos(){return this.offset+this.bufferPos}set filePos(e){this.bufferPos=e-this.offset}get remainingLength(){return Math.max(this.end-this.filePos,0)}skip(e){this.bufferPos+=e}slice(t,n=this.end-t){if(t<this.start||t+n>this.end)throw RangeError(`Slicing outside of original slice.`);return new e(this.bytes,this.view,this.offset,t,t+n)}},Ro=(e,t)=>{if(e.filePos<e.start||e.filePos+t>e.end)throw RangeError(`Tried reading [${e.filePos}, ${e.filePos+t}), but slice is [${e.start}, ${e.end}). This is likely an internal error, please report it alongside the file that caused it.`)},zo=(e,t)=>{Ro(e,t);let n=e.bytes.subarray(e.bufferPos,e.bufferPos+t);return e.bufferPos+=t,n},Bo=class{constructor(e){this.mutex=new Fr,this.trackTimestampInfo=new WeakMap,this.output=e}onTrackClose(e){}validateTimestamp(e,t,n){if(t<0)throw Error(`Timestamps must be non-negative (got ${t}s).`);let r=this.trackTimestampInfo.get(e);if(r){if(n&&(r.maxTimestampBeforeLastKeyPacket=r.maxTimestamp),r.maxTimestampBeforeLastKeyPacket!==null&&t<r.maxTimestampBeforeLastKeyPacket)throw Error(`Timestamps cannot be smaller than the largest timestamp of the previous GOP (a GOP begins with a key packet and ends right before the next key packet). Got ${t}s, but largest timestamp is ${r.maxTimestampBeforeLastKeyPacket}s.`);r.maxTimestamp=Math.max(r.maxTimestamp,t)}else{if(!n)throw Error(`First packet must be a key packet.`);r={maxTimestamp:t,maxTimestampBeforeLastKeyPacket:null},this.trackTimestampInfo.set(e,r)}}},Vo=/<(?:(\d{2}):)?(\d{2}):(\d{2}).(\d{3})>/g,Ho=e=>{let t=Math.floor(e/36e5),n=Math.floor(e%36e5/6e4),r=Math.floor(e%6e4/1e3),i=e%1e3;return t.toString().padStart(2,`0`)+`:`+n.toString().padStart(2,`0`)+`:`+r.toString().padStart(2,`0`)+`.`+i.toString().padStart(3,`0`)},Uo=class{constructor(e){this.writer=e,this.helper=new Uint8Array(8),this.helperView=new DataView(this.helper.buffer),this.offsets=new WeakMap}writeU32(e){this.helperView.setUint32(0,e,!1),this.writer.write(this.helper.subarray(0,4))}writeU64(e){this.helperView.setUint32(0,Math.floor(e/2**32),!1),this.helperView.setUint32(4,e,!1),this.writer.write(this.helper.subarray(0,8))}writeAscii(e){for(let t=0;t<e.length;t++)this.helperView.setUint8(t%8,e.charCodeAt(t)),t%8==7&&this.writer.write(this.helper);e.length%8!=0&&this.writer.write(this.helper.subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.writer.getPos()),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.writer.write(e.contents);else{let t=this.writer.getPos();if(this.writeBoxHeader(e,0),e.contents&&this.writer.write(e.contents),e.children)for(let t of e.children)t&&this.writeBox(t);let n=this.writer.getPos(),r=e.size??n-t;this.writer.seek(t),this.writeBoxHeader(e,r),this.writer.seek(n)}}writeBoxHeader(e,t){this.writeU32(e.largeSize?1:t),this.writeAscii(e.type),e.largeSize&&this.writeU64(t)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let t=this.offsets.get(e);H(t!==void 0);let n=this.writer.getPos();this.writer.seek(t),this.writeBox(e),this.writer.seek(n)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let t=this.measureBoxHeader(e);if(e.contents&&(t+=e.contents.byteLength),e.children)for(let n of e.children)n&&(t+=this.measureBox(n));return t}}},G=new Uint8Array(8),Wo=new DataView(G.buffer),K=e=>[(e%256+256)%256],q=e=>(Wo.setUint16(0,e,!1),[G[0],G[1]]),Go=e=>(Wo.setInt16(0,e,!1),[G[0],G[1]]),Ko=e=>(Wo.setUint32(0,e,!1),[G[1],G[2],G[3]]),J=e=>(Wo.setUint32(0,e,!1),[G[0],G[1],G[2],G[3]]),qo=e=>(Wo.setInt32(0,e,!1),[G[0],G[1],G[2],G[3]]),Jo=e=>(Wo.setUint32(0,Math.floor(e/2**32),!1),Wo.setUint32(4,e,!1),[G[0],G[1],G[2],G[3],G[4],G[5],G[6],G[7]]),Yo=e=>(Wo.setInt32(0,Math.floor(e/2**32),!1),Wo.setUint32(4,e,!1),[G[0],G[1],G[2],G[3],G[4],G[5],G[6],G[7]]),Xo=e=>(Wo.setInt16(0,256*e,!1),[G[0],G[1]]),Zo=e=>(Wo.setInt32(0,2**16*e,!1),[G[0],G[1],G[2],G[3]]),Qo=e=>(Wo.setInt32(0,2**30*e,!1),[G[0],G[1],G[2],G[3]]),$o=(e,t)=>{let n=[],r=e;do{let e=r&127;r>>=7,n.length>0&&(e|=128),n.push(e),t!==void 0&&t--}while(r>0||t);return n.reverse()},Y=(e,t=!1)=>{let n=Array(e.length).fill(null).map((t,n)=>e.charCodeAt(n));return t&&n.push(0),n},es=e=>{let t=Math.PI/180*e,n=Math.round(Math.cos(t)),r=Math.round(Math.sin(t));return[n,r,0,-r,n,0,0,0,1]},ts=es(0),ns=e=>[Zo(e[0]),Zo(e[1]),Qo(e[2]),Zo(e[3]),Zo(e[4]),Qo(e[5]),Zo(e[6]),Zo(e[7]),Qo(e[8])],X=(e,t,n)=>({type:e,contents:t&&new Uint8Array(t.flat(10)),children:n}),Z=(e,t,n,r,i)=>X(e,[K(t),Ko(n),r??[]],i),rs=e=>e.isQuickTime?X(`ftyp`,[Y(`qt  `),J(512),Y(`qt  `)]):e.fragmented?e.cmaf?X(`ftyp`,[Y(`iso5`),J(512),Y(`iso5`),Y(`iso6`),Y(`mp41`),Y(`cmfc`),Y(`dash`)]):X(`ftyp`,[Y(`iso5`),J(512),Y(`iso5`),Y(`iso6`),Y(`mp41`)]):X(`ftyp`,[Y(`isom`),J(512),Y(`isom`),e.holdsAvc?Y(`avc1`):[],Y(`mp41`)]),is=()=>X(`styp`,[Y(`iso5`),J(0),Y(`iso5`),Y(`iso6`),Y(`mp41`),Y(`cmfc`),Y(`dash`)]),as=(e,t)=>{let n=e.maxWrittenEndTimestamp-e.minWrittenTimestamp;return Number.isFinite(n)||(n=0),Z(`sidx`,1,0,[J(1),J(Ic),Jo(Q(e.minWrittenTimestamp,Ic)),Jo(0),q(0),q(1),J(t&2147483647),J(Q(n,Ic)),J(0)])},os=e=>({type:`mdat`,largeSize:e}),ss=e=>({type:`free`,size:e}),cs=e=>X(`moov`,void 0,[ls(e.creationTime,e.trackDatas),...e.trackDatas.map(t=>ds(t,e.creationTime)),e.isFragmented?$s(e.trackDatas):null,mc(e)]),ls=(e,t)=>{let n=Math.max(0,...t.map(e=>Q(us(e),Ic)+Q(e.startTimestampOffset??0,Ic))),r=Math.max(0,...t.map(e=>e.track.id))+1,i=!Tr(e)||!Tr(n),a=i?Jo:J;return Z(`mvhd`,+i,0,[a(e),a(e),J(Ic),a(n),Zo(1),Xo(1),Array(10).fill(0),ns(ts),Array(24).fill(0),J(r)])},us=e=>{if(e.samples.length===0)return 0;let t=1/0,n=-1/0;for(let r=0;r<e.samples.length;r++){let i=e.samples[r];i.timestamp<t&&(t=i.timestamp),i.timestamp+i.duration>n&&(n=i.timestamp+i.duration)}return t===1/0?0:n-t},ds=(e,t)=>{let n=Rc(e),r=e.startTimestampOffset!==null&&e.startTimestampOffset>0;return X(`trak`,void 0,[fs(e,t),r?ps(e,e.startTimestampOffset):null,ms(e,t),n.name===void 0?null:X(`udta`,void 0,[X(`name`,[...kr.encode(n.name)])])])},fs=(e,t)=>{let n=Q(us(e),Ic)+Q(e.startTimestampOffset??0,Ic),r=!Tr(t)||!Tr(n),i=r?Jo:J,a;if(e.type===`video`){let t=e.track.metadata.rotation;a=es(t??0)}else a=ts;let o=2;e.track.metadata.disposition?.default!==!1&&(o|=1);let s=e.type===`video`?0:e.type===`audio`?1:e.type===`subtitle`?2:Rr(e);return Z(`tkhd`,+r,o,[i(t),i(t),J(e.track.id),J(0),i(n),Array(8).fill(0),q(0),q(s),Xo(+(e.type===`audio`)),q(0),ns(a),Zo(e.type===`video`?e.info.width:0),Zo(e.type===`video`?e.info.height:0)])},ps=(e,t)=>{let n=Q(t,Ic),r=Q(us(e),Ic),i=!Tr(n)||!Tr(r),a=i?Jo:J,o=i?Yo:qo;return X(`edts`,void 0,[Z(`elst`,+!!i,0,[J(2),a(n),o(-1),Zo(1),a(r),o(0),Zo(1)])])},ms=(e,t)=>X(`mdia`,void 0,[hs(e,t),vs(!0,gs[e.type],_s[e.type]),ys(e)]),hs=(e,t)=>{let n=Q(us(e),e.timescale),r=!Tr(t)||!Tr(n),i=r?Jo:J;return Z(`mdhd`,+r,0,[i(t),i(t),J(e.timescale),i(n),q(Oc(e.track.metadata.languageCode??`und`)),q(0)])},gs={video:`vide`,audio:`soun`,subtitle:`text`},_s={video:`MediabunnyVideoHandler`,audio:`MediabunnySoundHandler`,subtitle:`MediabunnyTextHandler`},vs=(e,t,n,r=`\0\0\0\0`)=>Z(`hdlr`,0,0,[e?Y(`mhlr`):J(0),Y(t),Y(r),J(0),J(0),Y(n,!0)]),ys=e=>X(`minf`,void 0,[bs[e.type](),xs(),ws(e)]),bs={video:()=>Z(`vmhd`,0,1,[q(0),q(0),q(0),q(0)]),audio:()=>Z(`smhd`,0,0,[q(0),q(0)]),subtitle:()=>Z(`nmhd`,0,0)},xs=()=>X(`dinf`,void 0,[Ss()]),Ss=()=>Z(`dref`,0,0,[J(1)],[Cs()]),Cs=()=>Z(`url `,0,1),ws=e=>{let t=e.compositionTimeOffsetTable.length>1||e.compositionTimeOffsetTable.some(e=>e.sampleCompositionTimeOffset!==0);return X(`stbl`,void 0,[Ts(e),Ks(e),t?Zs(e):null,t?Qs(e):null,Js(e),Ys(e),Xs(e),qs(e)])},Ts=e=>{let t;if(e.type===`video`)t=Es(Sc(e.track.source._codec,e.info.decoderConfig.codec),e);else if(e.type===`audio`){let n=wc(e.track.source._codec,e.info.decoderConfig.codec,e.muxer.isQuickTime);H(n),t=Ns(n,e)}else e.type===`subtitle`&&(t=Ws(Ec[e.track.source._codec],e));return H(t),Z(`stsd`,0,0,[J(1)],[t])},Es=(e,t)=>X(e,[[,,,,,,].fill(0),q(1),q(0),q(0),Array(12).fill(0),q(t.info.width),q(t.info.height),J(4718592),J(4718592),J(0),q(1),K(10),Y(`Mediabunny`),Array(21).fill(0),q(t.info.hasAlphaChannel?32:24),Go(65535)],[Cc[t.track.source._codec]?.(t)??null,Ds(t),Nr(t.info.decoderConfig.colorSpace)?null:Os(t)]),Ds=e=>e.info.pixelAspectRatio.num===e.info.pixelAspectRatio.den?null:X(`pasp`,[J(e.info.pixelAspectRatio.num),J(e.info.pixelAspectRatio.den)]),Os=e=>{let t=e.info.decoderConfig.colorSpace;return X(`colr`,[Y(e.muxer.isQuickTime?`nclc`:`nclx`),q(t?.primaries==null?2:Ar[t.primaries]),q(t?.transfer==null?2:jr[t.transfer]),q(t?.matrix==null?2:Mr[t.matrix]),e.muxer.isQuickTime?[]:K(!!t?.fullRange<<7)])},ks=e=>e.info.decoderConfig&&X(`avcC`,[...Dr(e.info.decoderConfig.description)]),As=e=>e.info.decoderConfig&&X(`hvcC`,[...Dr(e.info.decoderConfig.description)]),js=e=>{if(!e.info.decoderConfig)return null;let t=e.info.decoderConfig,n=t.codec.split(`.`),r=Number(n[1]),i=Number(n[2]),a=Number(n[3]),o=n[4]?Number(n[4]):1,s=n[8]?Number(n[8]):Number(t.colorSpace?.fullRange??0),c=(a<<4)+(o<<1)+s,l=n[5]?Number(n[5]):t.colorSpace?.primaries?Ar[t.colorSpace.primaries]:1,u=n[6]?Number(n[6]):t.colorSpace?.transfer?jr[t.colorSpace.transfer]:1,d=n[7]?Number(n[7]):t.colorSpace?.matrix?Mr[t.colorSpace.matrix]:1;return Z(`vpcC`,1,0,[K(r),K(i),K(c),K(l),K(u),K(d),q(0)])},Ms=e=>X(`av1C`,Pa(e.info.decoderConfig.codec)),Ns=(e,t)=>{let n=0,r,i=16,a=Sa.includes(t.track.source._codec);if(a){let e=t.track.source._codec,{sampleSize:r}=Ia(e);i=8*r,i>16&&(n=1)}if(t.muxer.isQuickTime&&(n=1),n===0)r=[[,,,,,,].fill(0),q(1),q(n),q(0),J(0),q(t.info.numberOfChannels),q(i),q(0),q(0),q(t.info.sampleRate<2**16?t.info.sampleRate:0),q(0)];else{let e=a?0:-2;r=[[,,,,,,].fill(0),q(1),q(n),q(0),J(0),q(t.info.numberOfChannels),q(Math.min(i,16)),Go(e),q(0),q(t.info.sampleRate<2**16?t.info.sampleRate:0),q(0),a?[J(1),J(i/8),J(t.info.numberOfChannels*i/8)]:[J(0),J(0),J(0)],J(2)]}return X(e,r,[Tc(t.track.source._codec,t.muxer.isQuickTime)?.(t)??null])},Ps=e=>{let t;switch(e.track.source._codec){case`aac`:t=64;break;case`mp3`:t=107;break;case`vorbis`:t=221;break;default:throw Error(`Unhandled audio codec: ${e.track.source._codec}`)}let n=[...K(t),...K(21),...Ko(0),...J(0),...J(0)];if(e.info.decoderConfig.description){let t=Dr(e.info.decoderConfig.description);n=[...n,...K(5),...$o(t.byteLength),...t]}return n=[...q(1),...K(0),...K(4),...$o(n.length),...n,...K(6),...K(1),...K(2)],n=[...K(3),...$o(n.length),...n],Z(`esds`,0,0,n)},Fs=e=>X(`wave`,void 0,[Is(e),Ls(e),X(`\0\0\0\0`)]),Is=e=>X(`frma`,[Y(wc(e.track.source._codec,e.info.decoderConfig.codec,e.muxer.isQuickTime))]),Ls=e=>{let{littleEndian:t}=Ia(e.track.source._codec);return X(`enda`,[q(+t)])},Rs=e=>{let t=e.info.numberOfChannels,n=3840,r=e.info.sampleRate,i=0,a=0,o=new Uint8Array,s=e.info.decoderConfig?.description;if(s){H(s.byteLength>=18);let e=ta(Dr(s));t=e.outputChannelCount,n=e.preSkip,r=e.inputSampleRate,i=e.outputGain,a=e.channelMappingFamily,e.channelMappingTable&&(o=e.channelMappingTable)}return X(`dOps`,[K(0),K(t),q(n),J(r),Go(i),K(a),...o])},zs=e=>{let t=e.info.decoderConfig?.description;return H(t),Z(`dfLa`,0,0,[...Dr(t).subarray(4)])},Bs=e=>{let{littleEndian:t,sampleSize:n}=Ia(e.track.source._codec);return Z(`pcmC`,0,0,[K(+t),K(8*n)])},Vs=e=>{H(e.info.primingPacket);let t=ia(e.info.primingPacket.data);if(!t)throw Error(`Couldn't extract AC-3 frame info from the audio packet. Ensure the packets contain valid AC-3 sync frames (as specified in ETSI TS 102 366).`);let n=new Uint8Array(3),r=new W(n);return r.writeBits(2,t.fscod),r.writeBits(5,t.bsid),r.writeBits(3,t.bsmod),r.writeBits(3,t.acmod),r.writeBits(1,t.lfeon),r.writeBits(5,t.bitRateCode),r.writeBits(5,0),X(`dac3`,[...n])},Hs=e=>{H(e.info.primingPacket);let t=oa(e.info.primingPacket.data);if(!t)throw Error(`Couldn't extract E-AC-3 frame info from the audio packet. Ensure the packets contain valid E-AC-3 sync frames (as specified in ETSI TS 102 366).`);let n=16;for(let e of t.substreams)n+=23,e.numDepSub>0?n+=9:n+=1;let r=Math.ceil(n/8),i=new Uint8Array(r),a=new W(i);a.writeBits(13,t.dataRate),a.writeBits(3,t.substreams.length-1);for(let e of t.substreams)a.writeBits(2,e.fscod),a.writeBits(5,e.bsid),a.writeBits(1,0),a.writeBits(1,0),a.writeBits(3,e.bsmod),a.writeBits(3,e.acmod),a.writeBits(1,e.lfeon),a.writeBits(3,0),a.writeBits(4,e.numDepSub),e.numDepSub>0?a.writeBits(9,e.chanLoc):a.writeBits(1,0);return X(`dec3`,[...i])},Us=e=>{H(e.info.primingPacket);let t=_a(e.info.primingPacket.data);if(!t)throw Error(`Couldn't extract DTS frame info from the audio packet. Ensure the packets contain valid DTS frames as specified in ETSI TS 102 114.`);return X(`ddts`,[...ba(t)])},Ws=(e,t)=>X(e,[[,,,,,,].fill(0),q(1)],[Dc[t.track.source._codec](t)]),Gs=e=>X(`vttC`,[...kr.encode(e.info.config.description)]),Ks=e=>Z(`stts`,0,0,[J(e.timeToSampleTable.length),e.timeToSampleTable.map(e=>[J(e.sampleCount),J(e.sampleDelta)])]),qs=e=>{if(e.samples.every(e=>e.type===`key`))return null;let t=[...e.samples.entries()].filter(([,e])=>e.type===`key`);return Z(`stss`,0,0,[J(t.length),t.map(([e])=>J(e+1))])},Js=e=>Z(`stsc`,0,0,[J(e.compactlyCodedChunkTable.length),e.compactlyCodedChunkTable.map(e=>[J(e.firstChunk),J(e.samplesPerChunk),J(1)])]),Ys=e=>{if(e.type===`audio`&&e.info.requiresPcmTransformation){let{sampleSize:t}=Ia(e.track.source._codec);return Z(`stsz`,0,0,[J(t*e.info.numberOfChannels),J(e.samples.reduce((t,n)=>t+Q(n.duration,e.timescale),0))])}return Z(`stsz`,0,0,[J(0),J(e.samples.length),e.samples.map(e=>J(e.size))])},Xs=e=>e.finalizedChunks.length>0&&wr(e.finalizedChunks).offset>=2**32?Z(`co64`,0,0,[J(e.finalizedChunks.length),e.finalizedChunks.map(e=>Jo(e.offset))]):Z(`stco`,0,0,[J(e.finalizedChunks.length),e.finalizedChunks.map(e=>J(e.offset))]),Zs=e=>Z(`ctts`,1,0,[J(e.compositionTimeOffsetTable.length),e.compositionTimeOffsetTable.map(e=>[J(e.sampleCount),qo(e.sampleCompositionTimeOffset)])]),Qs=e=>{let t=1/0,n=-1/0,r=1/0,i=-1/0;H(e.compositionTimeOffsetTable.length>0),H(e.samples.length>0);for(let r=0;r<e.compositionTimeOffsetTable.length;r++){let i=e.compositionTimeOffsetTable[r];t=Math.min(t,i.sampleCompositionTimeOffset),n=Math.max(n,i.sampleCompositionTimeOffset)}for(let t=0;t<e.samples.length;t++){let n=e.samples[t];r=Math.min(r,Q(n.timestamp,e.timescale)),i=Math.max(i,Q(n.timestamp+n.duration,e.timescale))}let a=Math.max(-t,0);return i>=2**31?null:Z(`cslg`,0,0,[qo(a),qo(t),qo(n),qo(r),qo(i)])},$s=e=>X(`mvex`,void 0,e.map(ec)),ec=e=>Z(`trex`,0,0,[J(e.track.id),J(1),J(0),J(0),J(0)]),tc=(e,t)=>X(`moof`,void 0,[nc(e),...t.map(ic)]),nc=e=>Z(`mfhd`,0,0,[J(e)]),rc=e=>{let t=0,n=0,r=e.type===`delta`;return n|=+r,t|=r?1:2,t<<24|n<<16|0},ic=e=>X(`traf`,void 0,[ac(e),oc(e),sc(e)]),ac=e=>{H(e.currentChunk);let t=0;t|=8,t|=16,t|=32,t|=131072;let n=e.currentChunk.samples[1]??e.currentChunk.samples[0],r={duration:n.timescaleUnitsToNextSample,size:n.size,flags:rc(n)};return Z(`tfhd`,0,t,[J(e.track.id),J(r.duration),J(r.size),J(r.flags)])},oc=e=>(H(e.currentChunk),Z(`tfdt`,1,0,[Jo(Q(e.currentChunk.startTimestamp,e.timescale))])),sc=e=>{H(e.currentChunk);let t=e.currentChunk.samples.map(e=>e.timescaleUnitsToNextSample),n=e.currentChunk.samples.map(e=>e.size),r=e.currentChunk.samples.map(rc),i=e.currentChunk.samples.map(t=>Q(t.timestamp-t.decodeTimestamp,e.timescale)),a=new Set(t),o=new Set(n),s=new Set(r),c=new Set(i),l=s.size===2&&r[0]!==r[1],u=a.size>1,d=o.size>1,f=!l&&s.size>1,p=c.size>1||[...c].some(e=>e!==0),m=0;return m|=1,m|=4*l,m|=256*u,m|=512*d,m|=1024*f,m|=2048*p,Z(`trun`,1,m,[J(e.currentChunk.samples.length),J(e.currentChunk.offset-e.currentChunk.moofOffset||0),l?J(r[0]):[],e.currentChunk.samples.map((e,a)=>[u?J(t[a]):[],d?J(n[a]):[],f?J(r[a]):[],p?qo(i[a]):[]])])},cc=e=>X(`mfra`,void 0,[...e.map(lc),uc()]),lc=e=>Z(`tfra`,1,0,[J(e.track.id),J(63),J(e.finalizedChunks.length),e.finalizedChunks.map(t=>[Jo(Q(t.samples[0].timestamp,e.timescale)),Jo(t.moofOffset),J(t.trafIndex+1),J(1),J(1)])]),uc=()=>Z(`mfro`,0,0,[J(0)]),dc=()=>X(`vtte`),fc=(e,t,n,r,i)=>X(`vttc`,void 0,[i===null?null:X(`vsid`,[qo(i)]),n===null?null:X(`iden`,[...kr.encode(n)]),t===null?null:X(`ctim`,[...kr.encode(Ho(t))]),r===null?null:X(`sttg`,[...kr.encode(r)]),X(`payl`,[...kr.encode(e)])]),pc=e=>X(`vtta`,[...kr.encode(e)]),mc=e=>{let t=[],n=e.format._options.metadataFormat??`auto`,r=e.output._metadataTags;if(n===`mdir`||n===`auto`&&!e.isQuickTime){let e=yc(r);e&&t.push(e)}else if(n===`mdta`){let e=bc(r);e&&t.push(e)}else(n===`udta`||n===`auto`&&e.isQuickTime)&&hc(t,e.output._metadataTags);return t.length===0?null:X(`udta`,void 0,t)},hc=(e,t)=>{for(let{key:n,value:r}of ii(t))switch(n){case`title`:e.push(gc(`©nam`,r));break;case`description`:e.push(gc(`©des`,r));break;case`artist`:e.push(gc(`©ART`,r));break;case`album`:e.push(gc(`©alb`,r));break;case`albumArtist`:e.push(gc(`albr`,r));break;case`genre`:e.push(gc(`©gen`,r));break;case`date`:e.push(gc(`©day`,r.toISOString().slice(0,10)));break;case`comment`:e.push(gc(`©cmt`,r));break;case`lyrics`:e.push(gc(`©lyr`,r));break;case`raw`:break;case`discNumber`:case`discsTotal`:case`trackNumber`:case`tracksTotal`:case`images`:break;default:Rr(n)}if(t.raw)for(let n in t.raw){let r=t.raw[n];r==null||n.length!==4||e.some(e=>e.type===n)||(typeof r==`string`?e.push(gc(n,r)):r instanceof Uint8Array&&e.push(X(n,Array.from(r))))}},gc=(e,t)=>{let n=kr.encode(t);return X(e,[q(n.length),q(Oc(`und`)),Array.from(n)])},_c={"image/jpeg":13,"image/png":14,"image/bmp":27},vc=(e,t)=>{let n=[];for(let{key:r,value:i}of ii(e))switch(r){case`title`:n.push({key:t?`title`:`©nam`,value:xc(i)});break;case`description`:n.push({key:t?`description`:`©des`,value:xc(i)});break;case`artist`:n.push({key:t?`artist`:`©ART`,value:xc(i)});break;case`album`:n.push({key:t?`album`:`©alb`,value:xc(i)});break;case`albumArtist`:n.push({key:t?`album_artist`:`aART`,value:xc(i)});break;case`comment`:n.push({key:t?`comment`:`©cmt`,value:xc(i)});break;case`genre`:n.push({key:t?`genre`:`©gen`,value:xc(i)});break;case`lyrics`:n.push({key:t?`lyrics`:`©lyr`,value:xc(i)});break;case`date`:n.push({key:t?`date`:`©day`,value:xc(i.toISOString().slice(0,10))});break;case`images`:for(let e of i)e.kind===`coverFront`&&n.push({key:`covr`,value:X(`data`,[J(_c[e.mimeType]??0),J(0),Array.from(e.data)])});break;case`trackNumber`:if(t){let t=e.tracksTotal===void 0?i.toString():`${i}/${e.tracksTotal}`;n.push({key:`track`,value:xc(t)})}else n.push({key:`trkn`,value:X(`data`,[J(0),J(0),q(0),q(i),q(e.tracksTotal??0),q(0)])});break;case`discNumber`:t||n.push({key:`disc`,value:X(`data`,[J(0),J(0),q(0),q(i),q(e.discsTotal??0),q(0)])});break;case`tracksTotal`:case`discsTotal`:break;case`raw`:break;default:Rr(r)}if(e.raw)for(let r in e.raw){let i=e.raw[r];i==null||!t&&r.length!==4||n.some(e=>e.key===r)||(typeof i==`string`?n.push({key:r,value:xc(i)}):i instanceof Uint8Array?n.push({key:r,value:X(`data`,[J(0),J(0),Array.from(i)])}):i instanceof gi&&n.push({key:r,value:X(`data`,[J(_c[i.mimeType]??0),J(0),Array.from(i.data)])}))}return n},yc=e=>{let t=vc(e,!1);return t.length===0?null:Z(`meta`,0,0,void 0,[vs(!1,`mdir`,``,`appl`),X(`ilst`,void 0,t.map(e=>X(e.key,void 0,[e.value])))])},bc=e=>{let t=vc(e,!0);return t.length===0?null:X(`meta`,void 0,[vs(!1,`mdta`,``),Z(`keys`,0,0,[J(t.length)],t.map(e=>X(`mdta`,[...kr.encode(e.key)]))),X(`ilst`,void 0,t.map((e,t)=>X(String.fromCharCode(...J(t+1)),void 0,[e.value])))])},xc=e=>X(`data`,[J(1),J(0),...kr.encode(e)]),Sc=(e,t)=>{switch(e){case`avc`:return t.startsWith(`avc3`)?`avc3`:`avc1`;case`hevc`:return`hvc1`;case`vp8`:return`vp08`;case`vp9`:return`vp09`;case`av1`:return`av01`;case`prores`:return t}},Cc={avc:ks,hevc:As,vp8:js,vp9:js,av1:Ms,prores:null},wc=(e,t,n)=>{switch(e){case`aac`:return`mp4a`;case`mp3`:return`mp4a`;case`opus`:return`Opus`;case`vorbis`:return`mp4a`;case`flac`:return`fLaC`;case`ulaw`:return`ulaw`;case`alaw`:return`alaw`;case`pcm-u8`:return`raw `;case`pcm-s8`:return`sowt`;case`ac3`:return`ac-3`;case`eac3`:return`ec-3`;case`dts`:return t}if(n)switch(e){case`pcm-s16`:return`sowt`;case`pcm-s16be`:return`twos`;case`pcm-s24`:return`in24`;case`pcm-s24be`:return`in24`;case`pcm-s32`:return`in32`;case`pcm-s32be`:return`in32`;case`pcm-f32`:return`fl32`;case`pcm-f32be`:return`fl32`;case`pcm-f64`:return`fl64`;case`pcm-f64be`:return`fl64`}else switch(e){case`pcm-s16`:return`ipcm`;case`pcm-s16be`:return`ipcm`;case`pcm-s24`:return`ipcm`;case`pcm-s24be`:return`ipcm`;case`pcm-s32`:return`ipcm`;case`pcm-s32be`:return`ipcm`;case`pcm-f32`:return`fpcm`;case`pcm-f32be`:return`fpcm`;case`pcm-f64`:return`fpcm`;case`pcm-f64be`:return`fpcm`}},Tc=(e,t)=>{switch(e){case`aac`:return Ps;case`mp3`:return Ps;case`opus`:return Rs;case`vorbis`:return Ps;case`flac`:return zs;case`ac3`:return Vs;case`eac3`:return Hs;case`dts`:return Us}if(t)switch(e){case`pcm-s24`:return Fs;case`pcm-s24be`:return Fs;case`pcm-s32`:return Fs;case`pcm-s32be`:return Fs;case`pcm-f32`:return Fs;case`pcm-f32be`:return Fs;case`pcm-f64`:return Fs;case`pcm-f64be`:return Fs}else switch(e){case`pcm-s16`:return Bs;case`pcm-s16be`:return Bs;case`pcm-s24`:return Bs;case`pcm-s24be`:return Bs;case`pcm-s32`:return Bs;case`pcm-s32be`:return Bs;case`pcm-f32`:return Bs;case`pcm-f32be`:return Bs;case`pcm-f64`:return Bs;case`pcm-f64be`:return Bs}return null},Ec={webvtt:`wvtt`},Dc={webvtt:Gs},Oc=e=>{H(e.length===3);let t=0;for(let n=0;n<3;n++)t<<=5,t+=e.charCodeAt(n)-96;return t},kc=class{constructor(e,t){if(this.finalized=!1,this.started=!1,this.pos=0,this.trackedWrites=null,this.trackedStart=-1,this.trackedEnd=-1,e._writerAcquired)throw Error(`Can't have multiple Writers for the same Target.`);this.target=e,e._setMonotonicity(t),e._writerAcquired=!0}start(){H(!this.started),this.target._start(),this.started=!0}write(e){H(this.started&&!this.finalized),this.maybeTrackWrites(e),this.target._write(e,this.pos),this.pos+=e.byteLength}seek(e){this.pos=e}getPos(){return this.pos}async flush(){return H(this.started&&!this.finalized),this.target._flush()}async finalize(){H(this.started&&!this.finalized),await this.target._finalize(),this.finalized=!0}maybeTrackWrites(e){if(!this.trackedWrites)return;let t=this.getPos();if(t<this.trackedStart){if(t+e.byteLength<=this.trackedStart)return;e=e.subarray(this.trackedStart-t),t=0}let n=t+e.byteLength-this.trackedStart,r=this.trackedWrites.byteLength;for(;r<n;)r*=2;if(r!==this.trackedWrites.byteLength){let e=new Uint8Array(r);e.set(this.trackedWrites,0),this.trackedWrites=e}this.trackedWrites.set(e,t-this.trackedStart),this.trackedEnd=Math.max(this.trackedEnd,t+e.byteLength)}startTrackingWrites(){this.trackedWrites=new Uint8Array(1024),this.trackedStart=this.getPos(),this.trackedEnd=this.trackedStart}stopTrackingWrites(){if(!this.trackedWrites)throw Error(`Internal error: Can't get tracked writes since nothing was tracked.`);let e={data:this.trackedWrites.subarray(0,this.trackedEnd-this.trackedStart),start:this.trackedStart,end:this.trackedEnd};return this.trackedWrites=null,e}};Qa();var Ac=class extends fi{constructor(){super(...arguments),this._writerAcquired=!1,this._monotonicity=null,this.onwrite=null}_setMonotonicity(e){this._monotonicity!==!1&&(this._monotonicity=e)}_dispatchWrite(e,t){this.onwrite?.(e,t),this._emit(`write`,{start:e,end:t})}slice(e){if(!Number.isInteger(e)||e<0)throw TypeError(`offset must be a non-negative integer.`);return new Pc(this,e)}},jc=2**16,Mc=2**32,Nc=class extends Ac{constructor(e={}){if(super(),this.buffer=null,this._maxPos=0,!e||typeof e!=`object`)throw TypeError(`BufferTarget options, when provided, must be an object.`);if(e.onFinalize!==void 0&&typeof e.onFinalize!=`function`)throw TypeError(`options.onFinalize, when provided, must be a function.`);if(this._options=e,this._supportsResize=`resize`in new ArrayBuffer(0),this._supportsResize)try{this._buffer=new ArrayBuffer(jc,{maxByteLength:Mc})}catch{this._buffer=new ArrayBuffer(jc),this._supportsResize=!1}else this._buffer=new ArrayBuffer(jc);this._bytes=new Uint8Array(this._buffer)}_ensureSize(e){let t=this._buffer.byteLength;for(;t<e;)t*=2;if(t!==this._buffer.byteLength){if(t>Mc)throw Error(`ArrayBuffer exceeded maximum size of ${Mc} bytes. Please consider using another target.`);if(this._supportsResize)this._buffer.resize(t);else{let e=new ArrayBuffer(t),n=new Uint8Array(e);n.set(this._bytes,0),this._buffer=e,this._bytes=n}}}_start(){}_write(e,t){this._ensureSize(t+e.byteLength),this._bytes.set(e,t),this._maxPos=Math.max(this._maxPos,t+e.byteLength),this._dispatchWrite(t,t+e.byteLength)}async _flush(){}async _finalize(){this.buffer=this._buffer.slice(0,this._maxPos),this._options.onFinalize&&await this._options.onFinalize(this.buffer),this._emit(`finalized`)}async _close(){}_getSlice(e,t){return this._bytes.slice(e,t)}},Pc=class extends Ac{constructor(e,t){super(),this._baseTarget=e,this._offset=t}_start(){}_write(e,t){this._baseTarget._write(e,this._offset+t),this._dispatchWrite(t,t+e.byteLength)}_flush(){return this._baseTarget._flush()}async _finalize(){this._emit(`finalized`)}async _close(){}_setMonotonicity(e){super._setMonotonicity(e),this._baseTarget._setMonotonicity(e)}},Fc=class{constructor(e,t){if(this.rootPath=e,this.getTarget=t,typeof e!=`string`)throw TypeError(`rootPath must be a string.`);if(typeof t!=`function`)throw TypeError(`getTarget must be a function.`)}},Ic=57600,Lc=2082844800,Rc=e=>{let t={},n=e.track;return n.metadata.name!==void 0&&(t.name=n.metadata.name),t},Q=(e,t,n=!0)=>{let r=e*t;return n?Math.round(r):r},zc=class extends Bo{constructor(e,t){super(e),this.writer=null,this.boxWriter=null,this.initWriter=null,this.initBoxWriter=null,this.auxTarget=new Nc,this.auxWriter=new kc(this.auxTarget,!1),this.auxBoxWriter=new Uo(this.auxWriter),this.mdat=null,this.ftypSize=null,this.trackDatas=[],this.allTracksKnown=Lr(),this.creationTime=Math.floor(Date.now()/1e3)+Lc,this.finalizedChunks=[],this.wroteFragmentedHeader=!1,this.nextFragmentNumber=1,this.maxWrittenTimestamp=-1/0,this.minWrittenTimestamp=1/0,this.maxWrittenEndTimestamp=-1/0,this.segmentHeaderSize=null,this.format=t,this.formatOptions={...t._options},this.isQuickTime=t instanceof nl,this.isCmaf=t instanceof tl,this.minimumFragmentDuration=this.formatOptions.minimumFragmentDuration??(t instanceof tl?1/0:1),this.auxWriter.start()}async start(){let e=await this.mutex.acquire();if(this.isCmaf?(this.fastStart=`fragmented`,this.isFragmented=!0):(this.writer=await this.output._getRootWriter(e=>this.formatOptions.fastStart===void 0?e instanceof Nc:this.formatOptions.fastStart===`fragmented`),this.boxWriter=new Uo(this.writer),this.fastStart=this.formatOptions.fastStart??(this.writer.target instanceof Nc&&`in-memory`),this.isFragmented=this.fastStart===`fragmented`),this.isCmaf){if(!this.output._hasInitTarget())throw Error(`CMAF outputs require the initTarget field in OutputOptions to be set; the init segment will be written to it.`);let e=new kc(await this.output._getInitTarget(),!0);e.start(),this.initWriter=e,this.initBoxWriter=new Uo(e)}let t=this.output.tracks.some(e=>e.isVideoTrack()&&e.source._codec===`avc`);{let e=this.initBoxWriter??this.boxWriter;if(H(e),this.formatOptions.onFtyp&&e.writer.startTrackingWrites(),e.writeBox(rs({isQuickTime:this.isQuickTime,holdsAvc:t,fragmented:this.isFragmented,cmaf:this.isCmaf})),this.formatOptions.onFtyp){let{data:t,start:n}=e.writer.stopTrackingWrites();this.formatOptions.onFtyp(t,n)}this.ftypSize=e.writer.getPos(),this.isCmaf&&await this.initWriter.flush()}if(this.fastStart!==`in-memory`){if(this.fastStart===`reserve`){for(let e of this.output.tracks)if(e.metadata.maximumPacketCount===void 0)throw Error(`All tracks must specify maximumPacketCount in their metadata when using fastStart: 'reserve'.`)}else this.isFragmented||(H(this.writer),H(this.boxWriter),this.formatOptions.onMdat&&this.writer.startTrackingWrites(),this.mdat=os(!0),this.boxWriter.writeBox(this.mdat))}await this.writer?.flush();for(let e of this.output.tracks)e.isVideoTrack()&&e.metadata.decoderConfig?this.getVideoTrackData(e,e.metadata.primingPacket??null,{decoderConfig:e.metadata.decoderConfig}):e.isAudioTrack()&&e.metadata.decoderConfig&&this.getAudioTrackData(e,e.metadata.primingPacket??null,{decoderConfig:e.metadata.decoderConfig});e()}allTracksAreKnown(){for(let e of this.output.tracks)if(!e.source._closed&&!this.trackDatas.some(t=>t.track===e))return!1;return!0}async getMimeType(){await this.allTracksKnown.promise;let e=this.trackDatas.map(e=>e.type===`video`||e.type===`audio`?e.info.decoderConfig.codec:{webvtt:`wvtt`}[e.track.source._codec]);return Xa({isQuickTime:this.isQuickTime,hasVideo:this.trackDatas.some(e=>e.type===`video`),hasAudio:this.trackDatas.some(e=>e.type===`audio`),codecStrings:e})}getVideoTrackData(e,t,n){let r=this.trackDatas.find(t=>t.track===e);if(r)return r;Wa(n,e.source._codec),H(n),H(n.decoderConfig);let i={...n.decoderConfig};H(i.codedWidth!==void 0),H(i.codedHeight!==void 0);let a=!1;if(e.source._codec===`avc`&&!i.description){if(!t)throw Error(`No AVC description provided; you must therefore provide a priming packet.`);let e=Ii(t.data);if(!e)throw Error(`Couldn't extract an AVCDecoderConfigurationRecord from the AVC packet. Make sure the packets are in Annex B format (as specified in ITU-T-REC-H.264) when not providing a description, or provide a description (must be an AVCDecoderConfigurationRecord as specified in ISO 14496-15) and ensure the packets are in AVCC format.`);i.description=Li(e),a=!0}else if(e.source._codec===`hevc`&&!i.description){if(!t)throw Error(`No HEVC description provided; you must therefore provide a priming packet.`);let e=Wi(t.data);if(!e)throw Error(`Couldn't extract an HEVCDecoderConfigurationRecord from the HEVC packet. Make sure the packets are in Annex B format (as specified in ITU-T-REC-H.265) when not providing a description, or provide a description (must be an HEVCDecoderConfigurationRecord as specified in ISO 14496-15) and ensure the packets are in HEVC format.`);i.description=Qi(e),a=!0}let o=Yr(1/(e.metadata.frameRate??57600),1e6).den,s=i.displayAspectWidth,c=i.displayAspectHeight,l=s===void 0||c===void 0?{num:1,den:1}:ci({num:s*i.codedHeight,den:c*i.codedWidth}),u=i.codec===`ap4h`||i.codec===`ap4x`,d={muxer:this,track:e,type:`video`,info:{width:i.codedWidth,height:i.codedHeight,pixelAspectRatio:l,decoderConfig:i,requiresAnnexBTransformation:a,hasAlphaChannel:u},timescale:o,samples:[],sampleQueue:[],timestampProcessingQueue:[],timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,startTimestampOffset:null,finalizedChunks:[],currentChunk:null,compactlyCodedChunkTable:[],closed:!1};return this.trackDatas.push(d),this.trackDatas.sort((e,t)=>e.track.id-t.track.id),this.allTracksAreKnown()&&this.allTracksKnown.resolve(),d}getAudioTrackData(e,t,n){let r=this.trackDatas.find(t=>t.track===e);if(r)return r;Ka(n,e.source._codec),H(n),H(n.decoderConfig);let i={...n.decoderConfig},a=!1;if(e.source._codec===`aac`&&!i.description){if(!t)throw Error(`No AAC description provided; you must therefore provide a priming packet.`);let e=Za(Lo.tempFromBytes(t.data));if(!e)throw Error(`Couldn't parse ADTS header from the AAC packet. Make sure the packets are in ADTS format (as specified in ISO 13818-7) when not providing a description, or provide a description (must be an AudioSpecificConfig as specified in ISO 14496-3) and ensure the packets are raw AAC data.`);let n=bi[e.samplingFrequencyIndex],r=xi[e.channelConfiguration];if(n===void 0||r===void 0)throw Error(`Invalid ADTS frame header.`);i.description=Si({objectType:e.objectType,outputSampleRate:n,outputNumberOfChannels:r}),a=!0}if(!t){if(e.source._codec===`ac3`||e.source._codec===`eac3`)throw Error(`AC-3/E-AC-3 require a priming packet.`);if(e.source._codec===`dts`)throw Error(`DTS requires a priming packet.`)}let o={muxer:this,track:e,type:`audio`,info:{numberOfChannels:n.decoderConfig.numberOfChannels,sampleRate:n.decoderConfig.sampleRate,decoderConfig:i,requiresPcmTransformation:!this.isFragmented&&Sa.includes(e.source._codec),expectedNextPcmPacketTimestamp:null,requiresAdtsStripping:a,primingPacket:t},timescale:i.sampleRate,samples:[],sampleQueue:[],timestampProcessingQueue:[],timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,startTimestampOffset:null,finalizedChunks:[],currentChunk:null,compactlyCodedChunkTable:[],closed:!1};return this.trackDatas.push(o),this.trackDatas.sort((e,t)=>e.track.id-t.track.id),this.allTracksAreKnown()&&this.allTracksKnown.resolve(),o}getSubtitleTrackData(e,t){let n=this.trackDatas.find(t=>t.track===e);if(n)return n;qa(t),H(t),H(t.config);let r={muxer:this,track:e,type:`subtitle`,info:{config:t.config},timescale:1e3,samples:[],sampleQueue:[],timestampProcessingQueue:[],timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,startTimestampOffset:null,finalizedChunks:[],currentChunk:null,compactlyCodedChunkTable:[],closed:!1,lastCueEndTimestamp:0,cueQueue:[],nextSourceId:0,cueToSourceId:new WeakMap};return this.trackDatas.push(r),this.trackDatas.sort((e,t)=>e.track.id-t.track.id),this.allTracksAreKnown()&&this.allTracksKnown.resolve(),r}async addEncodedVideoPacket(e,t,n){let r=await this.mutex.acquire();try{let r=this.getVideoTrackData(e,t,n),i=t.data;if(r.info.requiresAnnexBTransformation){let e=[...Ai(i)].map(e=>i.subarray(e.offset,e.offset+e.length));if(e.length===0)throw Error(`Failed to transform packet data. Make sure all packets are provided in Annex B format, as specified in ITU-T-REC-H.264 and ITU-T-REC-H.265.`);i=Fi(e,4)}this.validateTimestamp(r.track,t.timestamp,t.type===`key`);let a=this.createSampleForTrack(r,i,t.timestamp,t.duration,t.type);await this.registerSample(r,a)}finally{r()}}async addEncodedAudioPacket(e,t,n){let r=await this.mutex.acquire();try{let r=this.getAudioTrackData(e,t,n),i=t.data;if(r.info.requiresAdtsStripping){let e=Za(Lo.tempFromBytes(i));if(!e)throw Error(`Expected ADTS frame, didn't get one.`);let t=e.crcCheck===null?7:9;i=i.subarray(t)}this.validateTimestamp(r.track,t.timestamp,t.type===`key`);let a=t.timestamp,o=t.duration;if(r.info.requiresPcmTransformation){let e=Ia(r.info.decoderConfig.codec).sampleSize*r.info.numberOfChannels;if(o=i.byteLength/e/r.info.sampleRate,r.info.expectedNextPcmPacketTimestamp!==null){let e=a-r.info.expectedNextPcmPacketTimestamp;if(e<.01)a=r.info.expectedNextPcmPacketTimestamp;else{let t=await this.padWithSilence(r,r.info.expectedNextPcmPacketTimestamp,e);a=r.info.expectedNextPcmPacketTimestamp+t}}r.info.expectedNextPcmPacketTimestamp=a+o}let s=this.createSampleForTrack(r,i,a,o,t.type);await this.registerSample(r,s)}finally{r()}}async padWithSilence(e,t,n){let r=Q(n,e.timescale);if(n=r/e.timescale,r>0){let{sampleSize:i,silentValue:a}=Ia(e.info.decoderConfig.codec),o=r*e.info.numberOfChannels,s=new Uint8Array(i*o).fill(a),c=this.createSampleForTrack(e,new Uint8Array(s.buffer),t,n,`key`);await this.registerSample(e,c)}return n}async addSubtitleCue(e,t,n){let r=await this.mutex.acquire();try{let r=this.getSubtitleTrackData(e,n);this.validateTimestamp(r.track,t.timestamp,!0),e.source._codec===`webvtt`&&(r.cueQueue.push(t),await this.processWebVTTCues(r,t.timestamp))}finally{r()}}async processWebVTTCues(e,t){for(;e.cueQueue.length>0;){let n=new Set([]);for(let r of e.cueQueue)H(r.timestamp<=t),H(e.lastCueEndTimestamp<=r.timestamp+r.duration),n.add(Math.max(r.timestamp,e.lastCueEndTimestamp)),n.add(r.timestamp+r.duration);let r=[...n].sort((e,t)=>e-t),i=r[0],a=r[1]??i;if(t<a)break;if(e.lastCueEndTimestamp<i){this.auxWriter.seek(0);let t=dc();this.auxBoxWriter.writeBox(t);let n=this.auxTarget._getSlice(0,this.auxWriter.getPos()),r=this.createSampleForTrack(e,n,e.lastCueEndTimestamp,i-e.lastCueEndTimestamp,`key`);await this.registerSample(e,r),e.lastCueEndTimestamp=i}this.auxWriter.seek(0);for(let t=0;t<e.cueQueue.length;t++){let n=e.cueQueue[t];if(n.timestamp>=a)break;Vo.lastIndex=0;let r=Vo.test(n.text),o=n.timestamp+n.duration,s=e.cueToSourceId.get(n);if(s===void 0&&a<o&&(s=e.nextSourceId++,e.cueToSourceId.set(n,s)),n.notes){let e=pc(n.notes);this.auxBoxWriter.writeBox(e)}let c=fc(n.text,r?i:null,n.identifier??null,n.settings??null,s??null);this.auxBoxWriter.writeBox(c),o===a&&e.cueQueue.splice(t--,1)}let o=this.auxTarget._getSlice(0,this.auxWriter.getPos()),s=this.createSampleForTrack(e,o,i,a-i,`key`);await this.registerSample(e,s),e.lastCueEndTimestamp=a}}createSampleForTrack(e,t,n,r,i){return{timestamp:n,decodeTimestamp:n,duration:r,data:t,size:t.byteLength,type:i,timescaleUnitsToNextSample:Q(r,e.timescale)}}processTimestamps(e,t){if(e.timestampProcessingQueue.length===0)return;if(e.type===`audio`&&e.info.requiresPcmTransformation){this.isFragmented||(e.startTimestampOffset??=e.timestampProcessingQueue[0].timestamp);let t=0;for(let n=0;n<e.timestampProcessingQueue.length;n++){let r=e.timestampProcessingQueue[n],i=Q(r.duration,e.timescale);t+=i}if(e.timeToSampleTable.length===0)e.timeToSampleTable.push({sampleCount:t,sampleDelta:1});else{let n=wr(e.timeToSampleTable);n.sampleCount+=t}e.timestampProcessingQueue.length=0;return}let n=e.timestampProcessingQueue.map(e=>e.timestamp).sort((e,t)=>e-t);this.isFragmented||(e.startTimestampOffset??=n[0]);for(let t=0;t<e.timestampProcessingQueue.length;t++){let r=e.timestampProcessingQueue[t];r.decodeTimestamp=n[t];let i=Q(r.timestamp-r.decodeTimestamp,e.timescale),a=Q(r.duration,e.timescale);if(e.lastTimescaleUnits!==null){H(e.lastSample);let t=Q(r.decodeTimestamp,e.timescale,!1),n=Math.round(t-e.lastTimescaleUnits);if(H(n>=0),e.lastTimescaleUnits+=n,e.lastSample.timescaleUnitsToNextSample=n,!this.isFragmented){let t=wr(e.timeToSampleTable);if(H(t),t.sampleCount===1){t.sampleDelta=n;let r=e.timeToSampleTable[e.timeToSampleTable.length-2];r&&r.sampleDelta===n&&(r.sampleCount++,e.timeToSampleTable.pop(),t=r)}else t.sampleDelta!==n&&(t.sampleCount--,e.timeToSampleTable.push(t={sampleCount:1,sampleDelta:n}));t.sampleDelta===a?t.sampleCount++:e.timeToSampleTable.push({sampleCount:1,sampleDelta:a});let r=wr(e.compositionTimeOffsetTable);H(r),r.sampleCompositionTimeOffset===i?r.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:i})}}else e.lastTimescaleUnits=Q(r.decodeTimestamp,e.timescale,!1),this.isFragmented||(e.timeToSampleTable.push({sampleCount:1,sampleDelta:a}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:i}));e.lastSample=r}if(e.timestampProcessingQueue.length=0,H(e.lastSample),H(e.lastTimescaleUnits!==null),t!==void 0&&e.lastSample.timescaleUnitsToNextSample===0){H(t.type===`key`);let n=Q(t.timestamp,e.timescale,!1),r=Math.round(n-e.lastTimescaleUnits);e.lastSample.timescaleUnitsToNextSample=r}}async registerSample(e,t){t.type===`key`&&this.processTimestamps(e,t),e.timestampProcessingQueue.push(t),this.isFragmented?(e.sampleQueue.push(t),await this.interleaveSamples()):this.fastStart===`reserve`?await this.registerSampleFastStartReserve(e,t):await this.addSampleToTrack(e,t)}async addSampleToTrack(e,t){if(!this.isFragmented&&(e.samples.push(t),this.fastStart===`reserve`)){let t=e.track.metadata.maximumPacketCount;if(H(t!==void 0),e.samples.length>t)throw Error(`Track #${e.track.id} has already reached the maximum packet count (${t}). Either add less packets or increase the maximum packet count.`)}let n=!1;if(!e.currentChunk)n=!0;else{e.currentChunk.startTimestamp=Math.min(e.currentChunk.startTimestamp,t.timestamp);let r=t.timestamp-e.currentChunk.startTimestamp;if(this.isFragmented){let i=this.trackDatas.every(n=>{if(e===n)return t.type===`key`;let r=n.sampleQueue[0];return r?r.type===`key`:n.closed});r>=this.minimumFragmentDuration&&i&&t.timestamp>this.maxWrittenTimestamp&&(n=!0,await this.finalizeFragment())}else n=r>=.5}n&&(e.currentChunk&&await this.finalizeCurrentChunk(e),e.currentChunk={startTimestamp:t.timestamp,samples:[],offset:null,moofOffset:null,trafIndex:null}),H(e.currentChunk),e.currentChunk.samples.push(t),this.isFragmented&&(this.maxWrittenTimestamp=Math.max(this.maxWrittenTimestamp,t.timestamp),this.maxWrittenEndTimestamp=Math.max(this.maxWrittenEndTimestamp,t.timestamp+t.duration),this.minWrittenTimestamp=Math.min(this.minWrittenTimestamp,t.timestamp))}async finalizeCurrentChunk(e){if(H(!this.isFragmented),H(this.writer),!e.currentChunk)return;e.finalizedChunks.push(e.currentChunk),this.finalizedChunks.push(e.currentChunk);let t=e.currentChunk.samples.length;if(e.type===`audio`&&e.info.requiresPcmTransformation&&(t=e.currentChunk.samples.reduce((t,n)=>t+Q(n.duration,e.timescale),0)),(e.compactlyCodedChunkTable.length===0||wr(e.compactlyCodedChunkTable).samplesPerChunk!==t)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:t}),this.fastStart===`in-memory`){e.currentChunk.offset=0;return}e.currentChunk.offset=this.writer.getPos();for(let t of e.currentChunk.samples)H(t.data),this.writer.write(t.data),t.data=null;await this.writer.flush()}async interleaveSamples(e=!1){if(H(this.isFragmented),e||this.allTracksAreKnown())outer:for(;;){let t=null,n=1/0;for(let r of this.trackDatas){if(!e&&r.sampleQueue.length===0&&!r.closed)break outer;r.sampleQueue.length>0&&r.sampleQueue[0].timestamp<n&&(t=r,n=r.sampleQueue[0].timestamp)}if(!t)break;let r=t.sampleQueue.shift();await this.addSampleToTrack(t,r)}}async finalizeFragment(e=!this.isCmaf){if(H(this.isFragmented),!this.wroteFragmentedHeader){this.wroteFragmentedHeader=!0;let e=this.initBoxWriter??this.boxWriter;H(e),this.formatOptions.onMoov&&e.writer.startTrackingWrites(),this.ensureOneEnabledTrack();let t=cs(this);if(e.writeBox(t),this.formatOptions.onMoov){let{data:t,start:n}=e.writer.stopTrackingWrites();this.formatOptions.onMoov(t,n)}if(this.isCmaf){H(this.initWriter),await this.initWriter.flush(),await this.initWriter.finalize(),this.writer=await this.output._getRootWriter(!0),this.boxWriter=new Uo(this.writer);let e=this.boxWriter.measureBox(is()),t=this.boxWriter.measureBox(as(this,0));this.segmentHeaderSize=e+t,this.writer.seek(this.segmentHeaderSize)}}H(this.writer),H(this.boxWriter);let t=this.trackDatas.filter(e=>e.currentChunk);if(t.length===0){e&&await this.writer.flush();return}let n=this.nextFragmentNumber++,r=tc(n,t),i=this.writer.getPos(),a=i+this.boxWriter.measureBox(r),o=a+8,s=1/0;for(let e=0;e<t.length;e++){let n=t[e];n.currentChunk.offset=o,n.currentChunk.moofOffset=i,n.currentChunk.trafIndex=e;for(let e of n.currentChunk.samples)o+=e.size;s=Math.min(s,n.currentChunk.startTimestamp)}let c=o-a,l=c>=2**32;if(l)for(let e of t)e.currentChunk.offset+=8;this.formatOptions.onMoof&&this.writer.startTrackingWrites();let u=tc(n,t);if(this.boxWriter.writeBox(u),this.formatOptions.onMoof){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMoof(e,t,s)}H(this.writer.getPos()===a),this.formatOptions.onMdat&&this.writer.startTrackingWrites();let d=os(l);d.size=c,this.boxWriter.writeBox(d),this.writer.seek(a+(l?16:8));for(let e of t)for(let t of e.currentChunk.samples)this.writer.write(t.data),t.data=null;if(this.formatOptions.onMdat){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMdat(e,t)}for(let e of t)e.finalizedChunks.push(e.currentChunk),this.finalizedChunks.push(e.currentChunk),e.currentChunk=null;e&&await this.writer.flush()}async registerSampleFastStartReserve(e,t){this.allTracksAreKnown()?(this.mdat||await this.createFastStartReserveMdat(),await this.addSampleToTrack(e,t)):e.sampleQueue.push(t)}async createFastStartReserveMdat(){H(this.writer),H(this.boxWriter),this.ensureOneEnabledTrack();let e=cs(this),t=this.boxWriter.measureBox(e)+this.computeSampleTableSizeUpperBound()+4096;H(this.ftypSize!==null),this.writer.seek(this.ftypSize+t),this.formatOptions.onMdat&&this.writer.startTrackingWrites(),this.mdat=os(!0),this.boxWriter.writeBox(this.mdat);for(let e of this.trackDatas){for(let t of e.sampleQueue)await this.addSampleToTrack(e,t);e.sampleQueue.length=0}}computeSampleTableSizeUpperBound(){H(this.fastStart===`reserve`);let e=0;for(let t of this.trackDatas){let n=t.track.metadata.maximumPacketCount;H(n!==void 0),e+=8*Math.ceil(2/3*n),e+=4*n,e+=8*Math.ceil(2/3*n),e+=12*Math.ceil(2/3*n),e+=4*n,e+=8*n}return e}async onTrackClose(e){let t=await this.mutex.acquire(),n=this.trackDatas.find(t=>t.track===e);n&&(n.closed=!0,n.type===`subtitle`&&e.source._codec===`webvtt`&&await this.processWebVTTCues(n,1/0),this.processTimestamps(n)),this.allTracksAreKnown()&&this.allTracksKnown.resolve(),this.isFragmented&&await this.interleaveSamples(),t()}ensureOneEnabledTrack(){for(let e of[`video`,`audio`,`subtitle`]){let t=this.trackDatas.filter(t=>t.type===e);if(t.length!==0&&!t.some(e=>e.track.metadata.disposition?.default!==!1)){let e=t[0];e.track.metadata.disposition={...e.track.metadata.disposition,default:!0}}}}async forceFragmentFinalization(){H(this.isFragmented);let e=await this.mutex.acquire();try{for(let e of this.trackDatas)e.type===`subtitle`&&e.track.source._codec===`webvtt`&&await this.processWebVTTCues(e,1/0),this.processTimestamps(e);await this.interleaveSamples(!0),await this.finalizeFragment()}finally{e()}}async finalize(){let e=await this.mutex.acquire();this.allTracksKnown.resolve(),this.ensureOneEnabledTrack(),!this.mdat&&this.fastStart===`reserve`&&await this.createFastStartReserveMdat();for(let e of this.trackDatas)e.closed=!0,e.type===`subtitle`&&e.track.source._codec===`webvtt`&&await this.processWebVTTCues(e,1/0),this.processTimestamps(e);if(this.isFragmented)await this.interleaveSamples(!0),await this.finalizeFragment(!1);else for(let e of this.trackDatas)if(await this.finalizeCurrentChunk(e),e.startTimestampOffset!==null)for(let t=0;t<e.samples.length;t++){let n=e.samples[t];n.timestamp-=e.startTimestampOffset,n.decodeTimestamp-=e.startTimestampOffset}if(H(this.writer),H(this.boxWriter),this.fastStart===`in-memory`){this.mdat=os(!1);let e;for(let t=0;t<2;t++){let t=cs(this),n=this.boxWriter.measureBox(t);e=this.boxWriter.measureBox(this.mdat);let r=this.writer.getPos()+n+e;for(let t of this.finalizedChunks){t.offset=r;for(let{data:n}of t.samples)H(n),r+=n.byteLength,e+=n.byteLength}if(r<2**32)break;e>=2**32&&(this.mdat.largeSize=!0)}this.formatOptions.onMoov&&this.writer.startTrackingWrites();let t=cs(this);if(this.boxWriter.writeBox(t),this.formatOptions.onMoov){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMoov(e,t)}this.formatOptions.onMdat&&this.writer.startTrackingWrites(),this.mdat.size=e,this.boxWriter.writeBox(this.mdat);for(let e of this.finalizedChunks)for(let t of e.samples)H(t.data),this.writer.write(t.data),t.data=null;if(this.formatOptions.onMdat){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMdat(e,t)}}else if(this.isFragmented){if(this.isCmaf){let e=this.segmentHeaderSize===null?0:this.writer.getPos()-this.segmentHeaderSize;this.writer.seek(0),this.boxWriter.writeBox(is()),this.boxWriter.writeBox(as(this,e))}else{let e=this.writer.getPos(),t=cc(this.trackDatas);this.boxWriter.writeBox(t);let n=this.writer.getPos()-e;this.writer.seek(this.writer.getPos()-4),this.boxWriter.writeU32(n)}}else{H(this.mdat);let e=this.boxWriter.offsets.get(this.mdat);H(e!==void 0);let t=this.writer.getPos()-e;if(this.mdat.size=t,this.mdat.largeSize=t>=2**32,this.boxWriter.patchBox(this.mdat),this.formatOptions.onMdat){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMdat(e,t)}let n=cs(this);if(this.fastStart===`reserve`){H(this.ftypSize!==null),this.writer.seek(this.ftypSize),this.formatOptions.onMoov&&this.writer.startTrackingWrites(),this.boxWriter.writeBox(n);let e=this.boxWriter.offsets.get(this.mdat)-this.writer.getPos();this.boxWriter.writeBox(ss(e))}else this.formatOptions.onMoov&&this.writer.startTrackingWrites(),this.boxWriter.writeBox(n);if(this.formatOptions.onMoov){let{data:e,start:t}=this.writer.stopTrackingWrites();this.formatOptions.onMoov(e,t)}}e()}},Bc=function(e,t,n){if(t!=null){if(typeof t!=`object`&&typeof t!=`function`)throw TypeError(`Object expected.`);var r,i;if(n){if(!Symbol.asyncDispose)throw TypeError(`Symbol.asyncDispose is not defined.`);r=t[Symbol.asyncDispose]}if(r===void 0){if(!Symbol.dispose)throw TypeError(`Symbol.dispose is not defined.`);r=t[Symbol.dispose],n&&(i=r)}if(typeof r!=`function`)throw TypeError(`Object not disposable.`);i&&(r=function(){try{i.call(this)}catch(e){return Promise.reject(e)}}),e.stack.push({value:t,dispose:r,async:n})}else n&&e.stack.push({async:!0});return t},Vc=(function(e){return function(t){function n(n){t.error=t.hasError?new e(n,t.error,`An error was suppressed during disposal.`):n,t.hasError=!0}var r,i=0;function a(){for(;r=t.stack.pop();)try{if(!r.async&&i===1)return i=0,t.stack.push(r),Promise.resolve().then(a);if(r.dispose){var e=r.dispose.call(r.value);if(r.async)return i|=2,Promise.resolve(e).then(a,function(e){return n(e),a()})}else i|=1}catch(e){n(e)}if(i===1)return t.hasError?Promise.reject(t.error):Promise.resolve();if(t.hasError)throw t.error}return a()}})(typeof SuppressedError==`function`?SuppressedError:function(e,t,n){var r=Error(n);return r.name=`SuppressedError`,r.error=e,r.suppressed=t,r}),Hc=class{constructor(){this._connectedTrack=null,this._closingPromise=null,this._closed=!1}_ensureValidAdd(){if(!this._connectedTrack)throw Error(`Source is not connected to an output track.`);if(this._connectedTrack.output.state===`canceled`)throw Error(`Output has been canceled.`);if(this._connectedTrack.output.state===`finalizing`||this._connectedTrack.output.state===`finalized`)throw Error(`Output has been finalized.`);if(this._connectedTrack.output.state===`pending`)throw Error(`Output has not started.`);if(this._closed)throw Error(`Source is closed.`)}async _start(){}async _flushAndClose(e){}close(){if(this._closingPromise)return;let e=this._connectedTrack;if(!e)throw Error(`Cannot call close without connecting the source to an output track.`);if(e.output.state===`pending`)throw Error(`Cannot call close before output has been started.`);this._closingPromise=(async()=>{await this._flushAndClose(!1),this._closed=!0,e.output.state!==`finalizing`&&e.output.state!==`finalized`&&e.output._muxer.onTrackClose(e)})()}async _flushOrWaitForOngoingClose(e){return this._closingPromise??=(async()=>{await this._flushAndClose(e),this._closed=!0})()}},Uc=class extends Hc{constructor(e){if(super(),this._connectedTrack=null,!xa.includes(e))throw TypeError(`Invalid video codec '${e}'. Must be one of: ${xa.join(`, `)}.`);this._codec=e}},Wc=(e,t)=>{if(e.metadata.hasOnlyKeyPackets&&t.type!==`key`)throw Error(`Cannot add non-key packets to a hasOnlyKeyPackets video track.`)},Gc=class{setError(e){this.errorSet||=(this.error=e,!0)}constructor(e,t){this.source=e,this.encodingConfig=t,this.ensureEncoderPromise=null,this.encoderInitialized=!1,this.encoder=null,this.muxer=null,this.lastMultipleOfKeyFrameInterval=-1,this.emittedEncoderPackets=0,this.codedWidth=null,this.codedHeight=null,this.outputWidth=null,this.outputHeight=null,this.frameRateLastSample=null,this.frameRateLastTimestamp=null,this.frameRateLastEndTimestamp=null,this.preciseTimings=[],this.customEncoder=null,this.customEncoderCallSerializer=new Xr,this.customEncoderQueueSize=0,this.defaultEncodeOptions={},this.alphaEncoder=null,this.splitter=null,this.splitterCreationFailed=!1,this.alphaFrameQueue=[],this.error=null,this.errorSet=!1,this.lastMuxerPromise=Promise.resolve(),this.closed=!1}async add(e,t,n){let r=e;try{this.checkForEncoderError(),this.source._ensureValidAdd();let i=this.encodingConfig,a=i.sizeChangeBehavior??`deny`,o=!1;if(this.codedWidth!==null&&this.codedHeight!==null){if((e.codedWidth!==this.codedWidth||e.codedHeight!==this.codedHeight)&&(o=!0,a===`deny`))throw Error(`Video sample size must remain constant. Expected ${this.codedWidth}x${this.codedHeight}, got ${e.codedWidth}x${e.codedHeight}. To allow the sample size to change over time, set \`sizeChangeBehavior\` to a value other than 'deny' in the encoding options.`)}else this.codedWidth=e.codedWidth,this.codedHeight=e.codedHeight;if(i.transform?.width!==void 0||i.transform?.height!==void 0||i.transform?.rotate!==void 0||i.transform?.crop!==void 0||i.transform?.force===!0||o&&a!==`passThrough`){let n=i.transform?.width,r=i.transform?.height,s=i.transform?.fit??`fill`;o&&a!==`passThrough`&&(H(this.outputWidth),H(this.outputHeight),H(a!==`deny`),n=this.outputWidth,r=this.outputHeight,s=a);let c=await e.transform({width:n,height:r,roundDimensionsTo:2,crop:i.transform?.crop,rotate:i.transform?.rotate,fit:s,alpha:i.alpha});(this.outputWidth===null||this.outputHeight===null)&&(this.outputWidth=c.displayWidth,this.outputHeight=c.displayHeight),t&&e.close(),e=c,t=!0}else(this.outputWidth===null||this.outputHeight===null)&&(this.outputWidth=e.codedWidth,this.outputHeight=e.codedHeight);let s=i.transform?.frameRate;if(s!==void 0){let i=e.timestamp+e.duration,a=Wr(e.timestamp,s);if(this.frameRateLastSample!==null){if(a<=this.frameRateLastTimestamp){this.frameRateLastSample.close(),this.frameRateLastSample=e.clone(),this.frameRateLastEndTimestamp=i;return}await this.padFrameRate(a,n)}e===r&&(e=e.clone(),t=!0),e.setTimestamp(a),e.setDuration(1/s),this.frameRateLastSample?.close(),this.frameRateLastSample=e.clone(),this.frameRateLastTimestamp=a,this.frameRateLastEndTimestamp=i}await this.processAndEncode(e,n)}finally{t&&e.close()}}async processAndEncode(e,t){let n=this.encodingConfig,r;if(n.transform?.process){let t=n.transform.process(e);if(ri(t)&&(t=await t),t===null)return;Array.isArray(t)||(t=[t]);let i=[];try{for(let n of t)n instanceof so?i.push(n):typeof VideoFrame<`u`&&n instanceof VideoFrame?i.push(new so(n)):i.push(new so(n,{timestamp:e.timestamp,duration:e.duration}))}catch(n){for(let t of i)t!==e&&t.close();for(let n of t)(n instanceof so&&n!==e||typeof VideoFrame<`u`&&n instanceof VideoFrame)&&n.close();throw n}r=i}else r=[e];try{for(let e of r){if(this.encoderInitialized||(this.ensureEncoderPromise||this.ensureEncoder(e),this.encoderInitialized||await this.ensureEncoderPromise),H(this.encoderInitialized),this.closed)break;let n=this.encodingConfig.keyFrameInterval??2,r=Math.floor(e.timestamp/n),i={...this.defaultEncodeOptions,...e.encodeOptions,...t},a={...i,keyFrame:i.keyFrame===void 0?n===0||r!==this.lastMultipleOfKeyFrameInterval:i.keyFrame};if(this.lastMultipleOfKeyFrameInterval=r,this.encodingConfig.onEncodedSample?.(e),this.customEncoder){this.customEncoderQueueSize++;let t=e.clone(),n=this.customEncoderCallSerializer.call(()=>this.customEncoder.encode(t,a)).catch(e=>this.setError(e)).finally(()=>{this.customEncoderQueueSize--,t.close()});this.customEncoderQueueSize>=4&&await n}else{H(this.encoder);let t=e.toVideoFrame(),n=Ir(this.preciseTimings,t.timestamp,e=>e.microsecondTimestamp),r=n===-1?null:this.preciseTimings[n];if(r&&r.microsecondTimestamp===t.timestamp?(r.timestamp!==e.timestamp&&(r.timestampIsValid=!1),r.duration!==e.duration&&(r.durationIsValid=!1)):(this.preciseTimings.splice(n+1,0,{microsecondTimestamp:t.timestamp,timestamp:e.timestamp,duration:e.duration,timestampIsValid:!0,durationIsValid:!0}),this.preciseTimings.length>128&&this.preciseTimings.shift()),!this.alphaEncoder)try{this.encoder.encode(t,a)}finally{t.close()}else if(t.format&&!t.format.includes(`A`)||this.splitterCreationFailed){this.alphaFrameQueue.push(null);try{this.encoder.encode(t,a)}finally{t.close()}}else{this.splitter||=new qc;let{colorFrame:e,alphaFrame:n}=await this.splitter.split(t);this.alphaFrameQueue.push(n);try{this.encoder.encode(e,a)}finally{e.close()}}this.encoder.encodeQueueSize>=4&&await new Promise(e=>this.encoder.addEventListener(`dequeue`,e,{once:!0}))}await this.lastMuxerPromise}}finally{for(let t of r)t!==e&&t.close()}}async padFrameRate(e,t){let n=this.encodingConfig.transform.frameRate;H(this.frameRateLastSample);let r=Math.round((e-this.frameRateLastTimestamp)*n);for(let e=1;e<r;e++){let r={stack:[],error:void 0,hasError:!1};try{let i=Bc(r,this.frameRateLastSample.clone(),!1);i.setTimestamp(this.frameRateLastTimestamp+e/n),i.setDuration(1/n),await this.processAndEncode(i,t)}catch(e){r.error=e,r.hasError=!0}finally{Vc(r)}}}ensureEncoder(e){this.ensureEncoderPromise=(async()=>{let t=Fo(this.encodingConfig.quality,this.encodingConfig.bitrate);H(t!==void 0);let n=Do({...this.encodingConfig,quality:t,width:e.codedWidth,height:e.codedHeight,squarePixelWidth:e.squarePixelWidth,squarePixelHeight:e.squarePixelHeight,framerate:this.source._connectedTrack?.metadata.frameRate}),r=null,i;for(let e of n){let t=e.config;if(this.encodingConfig.onEncoderConfig?.(t),i=Io.find(e=>e.supports(this.encodingConfig.codec,t)),i){r=e;break}if(!(typeof VideoEncoder>`u`)){if(t.alpha=`discard`,this.encodingConfig.alpha===`keep`&&(t.latencyMode=`quality`),(t.width%2==1||t.height%2==1)&&(this.encodingConfig.codec===`avc`||this.encodingConfig.codec===`hevc`))throw Error(`The dimensions ${t.width}x${t.height} are not supported for codec '${this.encodingConfig.codec}'; both width and height must be even numbers. Make sure to round your dimensions to the nearest even number.`);try{if((await VideoEncoder.isConfigSupported(t)).supported){r=e;break}}catch{}}}if(!r){if(typeof VideoEncoder>`u`)throw Error(ti(`VideoEncoder`));let e=n[0].config,t=n.map(({config:e,quantizer:t})=>t===null?`${e.bitrate} bps`:`quantizer ${t}`);throw Error(`This specific encoder configuration (${e.codec}, ${t.join(` / `)}, ${e.width}x${e.height}, hardware acceleration: ${e.hardwareAcceleration??`no-preference`}) is not supported in this environment. Consider using another codec or changing your video parameters.`)}let a=r.config;if(r.quantizer!==null&&(this.defaultEncodeOptions=No(this.encodingConfig.codec,r.quantizer)),i)this.customEncoder=new i,this.customEncoder.codec=this.encodingConfig.codec,this.customEncoder.config=a,this.customEncoder.onPacket=(e,t)=>{if(!(e instanceof Ya))throw TypeError(`The first argument passed to onPacket must be an EncodedPacket.`);if(t!==void 0&&(!t||typeof t!=`object`))throw TypeError(`The second argument passed to onPacket must be an object or undefined.`);Wc(this.source._connectedTrack,e),this.encodingConfig.onEncodedPacket?.(e,t),this.lastMuxerPromise=this.muxer.addEncodedVideoPacket(this.source._connectedTrack,e,t).catch(e=>{this.setError(e)})},this.customEncoder.onError=e=>{this.setError(e)},await this.customEncoder.init();else{let e=[],t=[],n=0,r=0,i=(e,t,n)=>{let r={};if(t){let e=new Uint8Array(t.byteLength);t.copyTo(e),r.alpha=e}let i=Ya.fromEncodedChunk(e,r),a=Ir(this.preciseTimings,e.timestamp,e=>e.microsecondTimestamp),o=a===-1?null:this.preciseTimings[a],s=null;this.emittedEncoderPackets===0&&i.type===`delta`&&n?.decoderConfig&&(s=na(this.encodingConfig.codec,n.decoderConfig,i.data)),(o&&o.microsecondTimestamp===e.timestamp||s!==null)&&(i=i.clone({timestamp:o?.timestampIsValid?o.timestamp:void 0,duration:o?.durationIsValid?o.duration:void 0,type:s??void 0})),Wc(this.source._connectedTrack,i),this.encodingConfig.onEncodedPacket?.(i,n),this.lastMuxerPromise=this.muxer.addEncodedVideoPacket(this.source._connectedTrack,i,n).catch(e=>{this.setError(e)}),this.emittedEncoderPackets++},o=Error(`Encoding error`).stack;if(this.encoder=new VideoEncoder({output:(a,o)=>{if(!this.alphaEncoder){i(a,null,o);return}let s=this.alphaFrameQueue.shift();H(s!==void 0),s?(this.alphaEncoder.encode(s,{...this.defaultEncodeOptions,keyFrame:a.type===`key`}),r++,s.close(),e.push({chunk:a,meta:o})):r===0?i(a,null,o):(t.push(n+r),e.push({chunk:a,meta:o}))},error:e=>{e.stack=o,this.setError(e)}}),this.encoder.configure(a),this.encodingConfig.alpha===`keep`){let o=Error(`Encoding error`).stack;this.alphaEncoder=new VideoEncoder({output:(a,o)=>{r--;let s=e.shift();for(H(s!==void 0),i(s.chunk,a,s.meta),n++;t.length>0&&t[0]===n;){t.shift();let n=e.shift();H(n!==void 0),i(n.chunk,null,n.meta)}},error:e=>{e.stack=o,this.setError(e)}}),this.alphaEncoder.configure(a)}}H(this.source._connectedTrack),this.muxer=this.source._connectedTrack.output._muxer,this.encoderInitialized=!0})()}async flushAndClose(e){try{if(!e&&(this.checkForEncoderError(),this.frameRateLastSample)){let e=this.encodingConfig.transform.frameRate,t=Wr(this.frameRateLastEndTimestamp,e);await this.padFrameRate(t)}this.closed=!0,e||(this.customEncoder?this.customEncoderCallSerializer.call(()=>this.customEncoder.flush()):this.encoder&&(await this.encoder.flush(),await this.alphaEncoder?.flush(),await ui(25)))}finally{this.closed=!0,this.frameRateLastSample?.close(),this.frameRateLastSample=null,this.customEncoder?await this.customEncoderCallSerializer.call(()=>this.customEncoder.close()).catch(e=>this.setError(e)):this.encoder&&(this.encoder.state!==`closed`&&this.encoder.close(),this.alphaEncoder&&this.alphaEncoder.state!==`closed`&&this.alphaEncoder.close(),this.alphaFrameQueue.forEach(e=>e?.close()),this.alphaFrameQueue.length=0,this.splitter?.close())}e||this.checkForEncoderError()}getQueueSize(){return this.customEncoder?this.customEncoderQueueSize:this.encoder?.encodeQueueSize??0}checkForEncoderError(){if(this.errorSet)throw this.error}},Kc=null,qc=class{constructor(){this.worker=null,this.pendingRequests=new Map,this.nextRequestId=0}split(e){if(!this.worker){if(!Kc){let e=new Blob([`(${Jc.toString()})()`],{type:`application/javascript`});Kc=URL.createObjectURL(e)}this.worker=new Worker(Kc),this.worker.addEventListener(`message`,e=>{let t=e.data,n=this.pendingRequests.get(t.id);n&&(this.pendingRequests.delete(t.id),`error`in t?n.reject(Error(t.error)):n.resolve({colorFrame:t.colorFrame,alphaFrame:t.alphaFrame}))}),this.worker.addEventListener(`error`,e=>{let t=Error(e.message||`Color/alpha splitter worker error.`);for(let e of this.pendingRequests.values())e.reject(t);this.pendingRequests.clear()})}let t=this.nextRequestId++,n=Lr();return this.pendingRequests.set(t,n),this.worker.postMessage({id:t,sourceFrame:e},{transfer:[e]}),n.promise}close(){this.worker?.terminate(),this.worker=null;let e=Error(`Color/alpha splitter closed.`);for(let t of this.pendingRequests.values())t.reject(e);this.pendingRequests.clear()}},Jc=()=>{let e=null,t=Promise.resolve();self.addEventListener(`message`,e=>{let{id:r,sourceFrame:i}=e.data;t=t.then(async()=>{try{let{colorFrame:e,alphaFrame:t}=await n(i);self.postMessage({id:r,colorFrame:e,alphaFrame:t},{transfer:[e,t]})}catch(e){self.postMessage({id:r,error:e.message})}finally{i.close()}})});let n=async t=>{let n=t.format;if(!n)throw Error(`CPU color/alpha splitting requires a known VideoFrame format.`);let a=t.allocationSize();if((!e||e.byteLength!==a)&&(e=new Uint8Array(a)),await t.copyTo(e),n===`RGBA`||n===`BGRA`)return r(e,n,t);if(n===`I420A`||n===`I420AP10`||n===`I420AP12`||n===`I422A`||n===`I422AP10`||n===`I422AP12`||n===`I444A`||n===`I444AP10`||n===`I444AP12`)return i(e,n,t);throw Error(`CPU color/alpha splitting does not support format '${n}'.`)},r=(e,t,n)=>{let r=n.visibleRect?.width??n.codedWidth,i=n.visibleRect?.height??n.codedHeight,a=r*i,o=a+Math.ceil(r/2)*Math.ceil(i/2)*2,s=new Uint8Array(o);for(let t=0,n=3;t<a;t++,n+=4)s[t]=e[n];s.fill(128,a);let c=new VideoFrame(e,{format:t===`RGBA`?`RGBX`:`BGRX`,codedWidth:r,codedHeight:i,timestamp:n.timestamp,duration:n.duration??void 0}),l={format:`I420`,codedWidth:r,codedHeight:i,timestamp:n.timestamp,duration:n.duration??void 0,transfer:[s.buffer]};return{colorFrame:c,alphaFrame:new VideoFrame(s,l)}},i=(e,t,n)=>{let r=n.visibleRect?.width??n.codedWidth,i=n.visibleRect?.height??n.codedHeight,a=t.includes(`P10`),o=t.includes(`P12`),s=a||o?2:1,c,l;t.startsWith(`I420`)?(c=Math.ceil(r/2),l=Math.ceil(i/2)):t.startsWith(`I422`)?(c=Math.ceil(r/2),l=i):(c=r,l=i);let u=r*i,d=c*l,f=u*s,p=d*s,m=u*s,h=f+p*2,g=t.replace(`A`,``),_=Math.ceil(r/2)*Math.ceil(i/2),v=m+_*s*2,y=new Uint8Array(v),b=h;y.set(e.subarray(b,b+m),0);let x=m,S=a?512:o?2048:128;s===1?y.fill(S,x):new Uint16Array(y.buffer,x,2*_).fill(S);let C=a?`I420P10`:o?`I420P12`:`I420`,ee=new VideoFrame(e.subarray(0,h),{format:g,codedWidth:r,codedHeight:i,timestamp:n.timestamp,duration:n.duration??void 0}),w={format:C,codedWidth:r,codedHeight:i,timestamp:n.timestamp,duration:n.duration??void 0,transfer:[y.buffer]};return{colorFrame:ee,alphaFrame:new VideoFrame(y,w)}}},Yc=class extends Uc{constructor(e,t){if(!(typeof HTMLCanvasElement<`u`&&e instanceof HTMLCanvasElement)&&!(typeof OffscreenCanvas<`u`&&e instanceof OffscreenCanvas))throw TypeError(`canvas must be an HTMLCanvasElement or OffscreenCanvas.`);To(t),super(t.codec),this._encoder=new Gc(this,t),this._canvas=e}add(e,t=0,n){if(!Number.isFinite(e)||e<0)throw TypeError(`timestamp must be a non-negative number.`);if(!Number.isFinite(t)||t<0)throw TypeError(`duration must be a non-negative number.`);let r=new so(this._canvas,{timestamp:e,duration:t});return this._encoder.add(r,!0,n)}_flushAndClose(e){return this._encoder.flushAndClose(e)}},Xc=class extends Hc{constructor(e){if(super(),this._connectedTrack=null,!wa.includes(e))throw TypeError(`Invalid audio codec '${e}'. Must be one of: ${wa.join(`, `)}.`);this._codec=e}},Zc=class extends Hc{constructor(e){if(super(),this._connectedTrack=null,!Ta.includes(e))throw TypeError(`Invalid subtitle codec '${e}'. Must be one of: ${Ta.join(`, `)}.`);this._codec=e}},Qc=class{getSupportedVideoCodecs(){return this.getSupportedCodecs().filter(e=>xa.includes(e))}getSupportedAudioCodecs(){return this.getSupportedCodecs().filter(e=>wa.includes(e))}getSupportedSubtitleCodecs(){return this.getSupportedCodecs().filter(e=>Ta.includes(e))}_codecUnsupportedHint(e){return``}_isFragmentedIsobmff(){return!1}},$c=class extends Qc{constructor(e={}){if(!e||typeof e!=`object`)throw TypeError(`options must be an object.`);if(e.fastStart!==void 0&&![!1,`in-memory`,`reserve`,`fragmented`].includes(e.fastStart))throw TypeError(`options.fastStart, when provided, must be false, 'in-memory', 'reserve', or 'fragmented'.`);if(e.minimumFragmentDuration!==void 0&&(!oi(e.minimumFragmentDuration)||e.minimumFragmentDuration<0))throw TypeError(`options.minimumFragmentDuration, when provided, must be a non-negative number.`);if(e.onFtyp!==void 0&&typeof e.onFtyp!=`function`)throw TypeError(`options.onFtyp, when provided, must be a function.`);if(e.onMoov!==void 0&&typeof e.onMoov!=`function`)throw TypeError(`options.onMoov, when provided, must be a function.`);if(e.onMdat!==void 0&&typeof e.onMdat!=`function`)throw TypeError(`options.onMdat, when provided, must be a function.`);if(e.onMoof!==void 0&&typeof e.onMoof!=`function`)throw TypeError(`options.onMoof, when provided, must be a function.`);if(e.metadataFormat!==void 0&&![`mdir`,`mdta`,`udta`,`auto`].includes(e.metadataFormat))throw TypeError(`options.metadataFormat, when provided, must be either 'auto', 'mdir', 'mdta', or 'udta'.`);super(),this._options=e}getSupportedTrackCounts(){let e=2**32-1;return{video:{min:0,max:e},audio:{min:0,max:e},subtitle:{min:0,max:e},total:{min:0,max:e}}}get supportsVideoRotationMetadata(){return!0}get supportsTimestampedMediaData(){return!0}_createMuxer(e){return new zc(e,this)}_isFragmentedIsobmff(){return this._options.fastStart===`fragmented`}},el=class extends $c{constructor(e){super(e)}get _name(){return`MP4`}get fileExtension(){return`.mp4`}get mimeType(){return`video/mp4`}getSupportedCodecs(){return[...xa,...Ca,`pcm-s16`,`pcm-s16be`,`pcm-s24`,`pcm-s24be`,`pcm-s32`,`pcm-s32be`,`pcm-f32`,`pcm-f32be`,`pcm-f64`,`pcm-f64be`,...Ta]}_codecUnsupportedHint(e){return new nl().getSupportedCodecs().includes(e)?` Switching to MOV will grant support for this codec.`:``}},tl=class extends $c{constructor(e){super(e)}get _name(){return`CMAF`}get fileExtension(){return`.m4s`}get mimeType(){return`video/mp4`}getSupportedCodecs(){return[...xa,...Ca,`pcm-s16`,`pcm-s16be`,`pcm-s24`,`pcm-s24be`,`pcm-s32`,`pcm-s32be`,`pcm-f32`,`pcm-f32be`,`pcm-f64`,`pcm-f64be`,...Ta]}},nl=class extends $c{constructor(e){super(e)}get _name(){return`MOV`}get fileExtension(){return`.mov`}get mimeType(){return`video/quicktime`}getSupportedCodecs(){return[...xa,...wa]}_codecUnsupportedHint(e){return new el().getSupportedCodecs().includes(e)?` Switching to MP4 will grant support for this codec.`:``}},rl=[`video`,`audio`,`subtitle`],il=class e{constructor(e,t,n,r,i){this.id=e,this.output=t,this.type=n,this.source=r,this.metadata=i}isVideoTrack(){return this.type===`video`}isAudioTrack(){return this.type===`audio`}isSubtitleTrack(){return this.type===`subtitle`}canBePairedWith(t){if(!(t instanceof e))throw TypeError(`other must be an OutputTrack.`);if(this===t)return!1;let n=di(this.metadata.group),r=di(t.metadata.group);for(let e of n)if(this.type!==t.type&&r.some(t=>e===t)||r.some(t=>e._pairedGroups.has(t)))return!0;return!1}},al=class extends il{constructor(e,t,n,r){super(e,t,`video`,n,r)}},ol=class extends il{constructor(e,t,n,r){super(e,t,`audio`,n,r)}},sl=class extends il{constructor(e,t,n,r){super(e,t,`subtitle`,n,r)}},cl=class e{constructor(){this._pairedGroups=new Set}pairWith(t){if(!(t instanceof e))throw TypeError(`other must be an OutputTrackGroup.`);if(this===t)throw TypeError(`Cannot pair a group with itself.`);this._pairedGroups.add(t),t._pairedGroups.add(this)}},ll=e=>{if(!e||typeof e!=`object`)throw TypeError(`metadata must be an object.`);if(e.languageCode!==void 0&&!qr(e.languageCode))throw TypeError(`metadata.languageCode, when provided, must be a three-letter, ISO 639-2/T language code.`);if(e.name!==void 0&&typeof e.name!=`string`)throw TypeError(`metadata.name, when provided, must be a string.`);if(e.disposition!==void 0&&yi(e.disposition),e.maximumPacketCount!==void 0&&(!Number.isInteger(e.maximumPacketCount)||e.maximumPacketCount<0))throw TypeError(`metadata.maximumPacketCount, when provided, must be a non-negative integer.`);if(e.group!==void 0&&!(e.group instanceof cl)&&(!Array.isArray(e.group)||e.group.some(e=>!(e instanceof cl))))throw TypeError(`metadata.group, when provided, must be an OutputTrackGroup instance or an array of OutputTrackGroup instances.`)},ul=class extends fi{get target(){let e=`Output.target cannot be used when using PathedTarget with an async callback. Use the 'target' event instead.`;if(this._rootTargetPromise)throw TypeError(e);let t=this._getRootTarget();if(ri(t))throw TypeError(e);return t}constructor(e){if(super(),this.state=`pending`,this.defaultTrackGroup=new cl,this.tracks=[],this._onFinalize=null,this._unfinalizedTargets=new Set,this._rootWriterPromise=null,this._startPromise=null,this._cancelPromise=null,this._finalizePromise=null,this._mutex=new Fr,this._metadataTags={},this._rootTarget=null,this._rootTargetPromise=null,this._firstMediaStreamTimestamp=null,!e||typeof e!=`object`)throw TypeError(`options must be an object.`);if(!(e.format instanceof Qc))throw TypeError(`options.format must be an OutputFormat.`);if(!(e.target instanceof Ac||e.target instanceof Fc))throw TypeError(`options.target must be a Target or a PathedTarget.`);if(e.target instanceof Ac&&this._rememberTarget(e.target),e.initTarget!==void 0&&!(e.initTarget instanceof Ac)&&typeof e.initTarget!=`function`)throw Error(`options.initTarget, when provided, must be a Target or a function that returns or resolves to a Target.`);if(e.onFinalize!==void 0&&typeof e.onFinalize!=`function`)throw TypeError(`options.onFinalize, when provided, must be a function.`);this.format=e.format,this._target=e.target,this._onFinalize=e.onFinalize??null,this._initTarget=e.initTarget??null,this._initTarget instanceof Ac&&this._rememberTarget(this._initTarget),this._muxer=e.format._createMuxer(this)}_getTargetValidated(e){H(this._target instanceof Fc);let t=this._target.getTarget(e),n=e=>{if(!(e instanceof Ac))throw TypeError(`getTarget must return a Target.`);return e};return ri(t)?t.then(n):n(t)}async _getTarget(e){H(this._target instanceof Fc);let t=await this._getTargetValidated(e);return this._emit(`target`,{target:t,request:e,isRoot:e.isRoot}),this.state===`canceled`?await t._close():this._rememberTarget(t),t}_rememberTarget(e){this._unfinalizedTargets.add(e),e.on(`finalized`,()=>this._unfinalizedTargets.delete(e),{once:!0})}async _getInitTarget(){if(H(this._initTarget!==null),this._initTarget instanceof Ac)return this._initTarget;let e=await this._initTarget();return this.state===`canceled`?await e._close():this._rememberTarget(e),e}_hasInitTarget(){return this._initTarget!==null}_getRootTarget(){if(this._rootTarget)return this._rootTarget;if(this._rootTargetPromise)return this._rootTargetPromise;if(this._target instanceof Ac)return this._emit(`target`,{target:this._target,request:null,isRoot:!0}),this._rootTarget=this._target,this._target;let e={path:this._target.rootPath,isRoot:!0,mimeType:this.format.mimeType},t=this._getTargetValidated(e),n=t=>(this.state===`canceled`?t._close():this._rememberTarget(t),this._emit(`target`,{target:t,request:e,isRoot:!0}),this._rootTarget=t,t);return ri(t)?this._rootTargetPromise=t.then(n):n(t)}_getRootWriter(e){return this._rootWriterPromise??=(async()=>{let t=await this._getRootTarget(),n=new kc(t,typeof e==`boolean`?e:e(t));return n.start(),n})()}addVideoTrack(e,t={}){if(!(e instanceof Uc))throw TypeError(`source must be a VideoSource.`);if(ll(t),t.rotation!==void 0&&![0,90,180,270].includes(t.rotation))throw TypeError(`Invalid video rotation: ${t.rotation}. Has to be 0, 90, 180 or 270.`);if(!this.format.supportsVideoRotationMetadata&&t.rotation)throw Error(`${this.format._name} does not support video rotation metadata.`);if(t.frameRate!==void 0&&(!Number.isFinite(t.frameRate)||t.frameRate<=0))throw TypeError(`Invalid video frame rate: ${t.frameRate}. Must be a positive number.`);if(t.decoderConfig!==void 0&&Wa({decoderConfig:t.decoderConfig},e._codec),t.primingPacket!==void 0){if(!(t.primingPacket instanceof Ya))throw TypeError(`metadata.primingPacket, when provided, must be an EncodedPacket.`);if(t.decoderConfig===void 0)throw TypeError(`metadata.primingPacket can only be provided alongside metadata.decoderConfig.`)}let n={...t};return n.group??=this.defaultTrackGroup,this._addTrack(new al(this.tracks.length+1,this,e,n))}addAudioTrack(e,t={}){if(!(e instanceof Xc))throw TypeError(`source must be an AudioSource.`);if(ll(t),t.decoderConfig!==void 0&&Ka({decoderConfig:t.decoderConfig},e._codec),t.primingPacket!==void 0){if(!(t.primingPacket instanceof Ya))throw TypeError(`metadata.primingPacket, when provided, must be an EncodedPacket.`);if(t.decoderConfig===void 0)throw TypeError(`metadata.primingPacket can only be provided alongside metadata.decoderConfig.`)}let n={...t};return n.group??=this.defaultTrackGroup,this._addTrack(new ol(this.tracks.length+1,this,e,n))}addSubtitleTrack(e,t={}){if(!(e instanceof Zc))throw TypeError(`source must be a SubtitleSource.`);ll(t);let n={...t};return n.group??=this.defaultTrackGroup,this._addTrack(new sl(this.tracks.length+1,this,e,n))}setMetadataTags(e){if(vi(e),this.state!==`pending`)throw Error(`Cannot set metadata tags after output has been started or canceled.`);this._metadataTags=e}_addTrack(e){if(this.state!==`pending`)throw Error(`Cannot add track after output has been started or canceled.`);if(e.source._connectedTrack)throw Error(`Source is already used for a track.`);let t=this.format.getSupportedTrackCounts(),n=this.tracks.reduce((t,n)=>t+ +(n.type===e.type),0),r=t[e.type].max;if(n===r)throw Error(r===0?`${this.format._name} does not support ${e.type} tracks.`:`${this.format._name} does not support more than ${r} ${e.type} track${r===1?``:`s`}.`);let i=t.total.max;if(this.tracks.length===i)throw Error(`${this.format._name} does not support more than ${i} tracks${i===1?``:`s`} in total.`);if(e.isVideoTrack()){let t=this.format.getSupportedVideoCodecs();if(t.length===0)throw Error(`${this.format._name} does not support video tracks.`+this.format._codecUnsupportedHint(e.source._codec));if(!t.includes(e.source._codec))throw Error(`Codec '${e.source._codec}' cannot be contained within ${this.format._name}. Supported video codecs are: ${t.map(e=>`'${e}'`).join(`, `)}.`+this.format._codecUnsupportedHint(e.source._codec))}else if(e.isAudioTrack()){let t=this.format.getSupportedAudioCodecs();if(t.length===0)throw Error(`${this.format._name} does not support audio tracks.`+this.format._codecUnsupportedHint(e.source._codec));if(!t.includes(e.source._codec))throw Error(`Codec '${e.source._codec}' cannot be contained within ${this.format._name}. Supported audio codecs are: ${t.map(e=>`'${e}'`).join(`, `)}.`+this.format._codecUnsupportedHint(e.source._codec))}else if(e.isSubtitleTrack()){let t=this.format.getSupportedSubtitleCodecs();if(t.length===0)throw Error(`${this.format._name} does not support subtitle tracks.`+this.format._codecUnsupportedHint(e.source._codec));if(!t.includes(e.source._codec))throw Error(`Codec '${e.source._codec}' cannot be contained within ${this.format._name}. Supported subtitle codecs are: ${t.map(e=>`'${e}'`).join(`, `)}.`+this.format._codecUnsupportedHint(e.source._codec))}return this.tracks.push(e),e.source._connectedTrack=e,e}hasEnoughTracks(){let e=this.format.getSupportedTrackCounts();for(let t of rl)if(this.tracks.reduce((e,n)=>e+ +(n.type===t),0)<e[t].min)return!1;let t=e.total.min;return!(this.tracks.length<t)}async start(){let e=this.format.getSupportedTrackCounts();for(let t of rl){let n=this.tracks.reduce((e,n)=>e+ +(n.type===t),0),r=e[t].min;if(n<r)throw Error(r===e[t].max?`${this.format._name} requires exactly ${r} ${t} track${r===1?``:`s`}.`:`${this.format._name} requires at least ${r} ${t} track${r===1?``:`s`}.`)}let t=e.total.min;if(this.tracks.length<t)throw Error(t===e.total.max?`${this.format._name} requires exactly ${t} track${t===1?``:`s`}.`:`${this.format._name} requires at least ${t} track${t===1?``:`s`}.`);if(this.state===`canceled`)throw Error(`Output has been canceled.`);return this._startPromise?(hi._warn(`Output has already been started.`),this._startPromise):this._startPromise=(async()=>{this.state=`started`;let e=this._mutex.acquire();try{await this._muxer.start();let e=this.tracks.map(e=>e.source._start());await Promise.all(e)}finally{(await e)()}})()}getMimeType(){return this._muxer.getMimeType()}async cancel(){if(this._cancelPromise)return hi._warn(`Output has already been canceled.`),this._cancelPromise;if(this.state===`finalizing`||this.state===`finalized`){this.state===`finalized`&&hi._warn(`Output has already been finalized.`);return}return this._cancelPromise=(async()=>{this.state=`canceled`;let e=await this._mutex.acquire();try{let e=this.tracks.map(e=>e.source._flushOrWaitForOngoingClose(!0));await Promise.all(e),await Promise.all([...this._unfinalizedTargets].map(e=>e._close())),this._unfinalizedTargets.clear()}finally{e()}})()}async finalize(){if(this.state===`pending`)throw Error(`Cannot finalize before starting.`);if(this.state===`canceled`)throw Error(`Cannot finalize after canceling.`);return this._finalizePromise?(hi._warn(`Output has already been finalized.`),this._finalizePromise):this._finalizePromise=(async()=>{this.state=`finalizing`;let e=await this._mutex.acquire();try{let e=this.tracks.map(e=>e.source._flushOrWaitForOngoingClose(!1));if(await Promise.all(e),await this._muxer.finalize(),this._rootWriterPromise){let e=await this._rootWriterPromise;e.finalized||(await e.flush(),await e.finalize())}this._onFinalize&&await this._onFinalize(),this.state=`finalized`}finally{await Promise.all([...this._unfinalizedTargets].map(e=>e._close().catch(()=>{}))),this._unfinalizedTargets.clear(),e()}})()}};function dl(){return typeof VideoEncoder<`u`}async function fl({engine:e,fps:t=60,onProgress:n,signal:r}){let i=e.demoClock.framesPerLoop,a=e.canvasElement,o=new Nc,s=new ul({format:new el,target:o}),c=new Yc(a,{codec:`avc`,bitrate:Po});s.addVideoTrack(c,{frameRate:t}),await s.start(),e.beginExport();try{for(let a=0;a<i;a++)r?.throwIfAborted(),await e.renderExportFrame(a>0),await c.add(a/t,1/t),n?.({done:a+1,total:i,stage:`render`});n?.({done:i,total:i,stage:`finalize`}),await s.finalize()}finally{e.endExport()}let l=o.buffer;if(!l)throw Error(`export produced no buffer`);return new Uint8Array(l)}function pl(e,t=`recall-path`){return`vestige-receipt-${e.replace(/[^a-zA-Z0-9_-]/g,``).slice(0,24)||`receipt`}-${t}-loop.mp4`}function ml(e,t){console.info(`[loop-export] ${t}: ${(e.length/1e6).toFixed(1)}MB, handing to browser download`);let n=new Blob([e],{type:`video/mp4`}),r=URL.createObjectURL(n),i=document.createElement(`a`);i.href=r,i.download=t,i.click(),setTimeout(()=>URL.revokeObjectURL(r),1e4)}var hl=`vb1-`,gl=[`0-10%`,`10-20%`,`20-30%`,`30-40%`,`40-50%`,`50-60%`,`60-70%`,`70-80%`,`80-90%`,`90-100%`],_l=[`concept`,`decision`,`event`,`fact`,`note`,`pattern`,`person`,`place`],vl=2166136261,yl=16777619;function bl(e){let t=typeof e==`string`?new TextEncoder().encode(e):e,n=vl;for(let e=0;e<t.length;e++)n^=t[e],n=Math.imul(n,yl);return n>>>0}function xl(e){return/^vb1-[0-9a-f]{8}$/.test(e)}function Sl(e){return`${hl}${(e>>>0).toString(16).padStart(8,`0`)}`}function Cl(e,t){return xl(e)?`vestige-${e}-loop.mp4`:`vestige-${t}-loop.mp4`}function $(e){return Number.isFinite(e)?Math.max(0,Math.round(e)):0}function wl(e){return Number.isFinite(e)?Math.max(0,Math.min(1e3,Math.round(e*1e3))):0}function Tl(e){return Number.isFinite(e)?Math.max(0,Math.min(1e3,Math.round(e*10))):0}function El(e){return e.toLowerCase().replace(/[^a-z0-9_-]/g,``)}function Dl(e){let t=new Map;for(let e of gl)t.set(e,0);for(let n of e){let e=n.range.trim();e&&t.set(e,$(n.count))}return t}function Ol(e){let t=new Map;for(let e of _l)t.set(e,0);for(let[n,r]of Object.entries(e)){let e=El(n);e&&t.set(e,(t.get(e)??0)+$(r))}return t}function kl(e){let t=$(e.nodeCount),n=$(e.edgeCount),r=t<=0?0:Math.round(1e3*n/t),i=Ol(e.byType),a=Dl(e.retentionBuckets);return[1,$(e.totalMemories),0,wl(e.averageRetention),Tl(e.embeddingCoverage),$(e.endangeredCount),t,n,r,..._l.map(e=>i.get(e)??0),...gl.map(e=>a.get(e)??0)]}function Al(e){let t=kl(e),n=Ol(e.byType),r=[...n.entries()].filter(([e])=>!_l.includes(e)).filter(([,e])=>e>0).sort(([e],[t])=>e.localeCompare(t)).map(([e,t])=>`${e}=${t}`).join(`,`),i=Dl(e.retentionBuckets),a=gl.map(e=>`${e}=${i.get(e)??0}`).join(`,`),o=r.length?`|extra:${r}`:``;return`v${t[0]}|t:${t[1]}|d:${t[2]}|r:${t[3]}|c:${t[4]}|z:${t[5]}|n:${t[6]}|g:${t[7]}|x:${t[8]}|types:${_l.map(e=>`${e}=${n.get(e)??0}`).join(`,`)}|ret:${a}${o}`}function jl(e,t){let n=Math.max(1,$(e.totalMemories)||$(e.retentionBuckets.reduce((e,t)=>e+$(t.count),0))),r=0;for(let n of Dl(e.retentionBuckets)){let e=/^(\d+)\s*-\s*(\d+)%$/.exec(n[0]);e&&t(Number(e[1]),Number(e[2]))&&(r+=n[1])}return r/n}function Ml(e){let t=Math.max(0,$(e.totalMemories)),n=Math.max(1,t),r=Math.max(0,$(e.nodeCount)),i=Math.max(0,$(e.edgeCount)),a=[...Ol(e.byType).entries()].filter(([,e])=>e>0),o=a.reduce((e,[,t])=>e+t,0)||1,s=0;for(let[,e]of a){let t=e/o;s-=t*Math.log2(t)}let c=null,l=0;for(let[e,t]of a){let n=t/o;(n>l||n===l&&e.localeCompare(c??``)<0)&&(c=e,l=n)}return{total:t,avgRet:wl(e.averageRetention)/1e3,coverage:Tl(e.embeddingCoverage)/1e3,endangeredRatio:$(e.endangeredCount)/n,edgeDensity:r<=0?0:i/r,highRetFrac:jl(e,e=>e>=70),lowRetFrac:jl(e,(e,t)=>t<=30),typeEntropy:s,dominantType:c,dominantFrac:l,typeCount:a.length}}var Nl=[{id:`dense-associative`,label:`dense associative field`,group:`density`,score:e=>e.edgeDensity>=1.2?Math.min(3,e.edgeDensity):0},{id:`sparse-lattice`,label:`sparse lattice`,group:`density`,score:e=>e.edgeDensity>0&&e.edgeDensity<.45?1.4-e.edgeDensity:0},{id:`deep-archive`,label:`deep archive`,group:`vitality`,score:e=>e.highRetFrac>=.4||e.avgRet>=.72?1.2+e.highRetFrac:0},{id:`sedimentary`,label:`sedimentary dark`,group:`vitality`,score:e=>e.lowRetFrac>=.3||e.endangeredRatio>=.22?1.1+e.lowRetFrac:0},{id:`oxygen-rich`,label:`oxygen-rich field`,group:`vitality`,score:e=>e.avgRet>=.78&&e.highRetFrac>=.35?e.avgRet:0},{id:`typed-mosaic`,label:`typed mosaic`,group:`types`,score:e=>e.typeCount>=5&&e.typeEntropy>=1.8?e.typeEntropy:0},{id:`concept-dominant`,label:`concept-weighted`,group:`types`,score:e=>e.dominantType===`concept`&&e.dominantFrac>=.4?e.dominantFrac:0},{id:`event-forward`,label:`event-forward`,group:`types`,score:e=>e.dominantType===`event`&&e.dominantFrac>=.4?e.dominantFrac:0},{id:`decision-heavy`,label:`decision-heavy`,group:`types`,score:e=>e.dominantType===`decision`&&e.dominantFrac>=.35?e.dominantFrac:0},{id:`wide-field`,label:`wide-field archive`,group:`scale`,score:e=>e.total>=400?Math.log10(e.total):0},{id:`expanding-cortex`,label:`expanding cortex`,group:`scale`,score:e=>e.total>=80&&e.total<400?.6:0},{id:`compact-nucleus`,label:`compact nucleus`,group:`scale`,score:e=>e.total>0&&e.total<80?.55:0},{id:`covered-embeddings`,label:`fully embedded`,group:`coverage`,score:e=>e.coverage>=.95?e.coverage:0}],Pl=[{id:`wide-field`,label:`wide-field archive`},{id:`expanding-cortex`,label:`expanding cortex`},{id:`compact-nucleus`,label:`compact nucleus`}];function Fl(e){let t=Ml(e),n=Nl.map(e=>({rule:e,score:e.score(t)})).filter(e=>e.score>0).sort((e,t)=>t.score-e.score||e.rule.id.localeCompare(t.rule.id)),r=[],i=new Set,a=new Set;for(let{rule:e}of n){if(r.length>=3)break;i.has(e.group)||(r.push({id:e.id,label:e.label}),i.add(e.group),a.add(e.id))}if(r.length<2){let e=t.total>=400?Pl[0]:t.total>=80?Pl[1]:Pl[2];e&&!a.has(e.id)&&(r.push(e),a.add(e.id))}return r.length<2&&r.push({id:`structured-field`,label:`structured field`}),r.slice(0,3)}function Il(e){let t=Sl(bl(Al(e)));return{printId:t,seed:t,traits:Fl(e),vector:kl(e)}}function Ll(e){let t=e.retention.endangered;return{totalMemories:$(e.stats.totalMemories),dueForReview:$(e.stats.dueForReview),averageRetention:e.stats.averageRetention,embeddingCoverage:e.stats.embeddingCoverage,endangeredCount:t?$(t.length):0,byType:{...e.retention.byType},retentionBuckets:e.retention.distribution.map(e=>({range:e.range,count:$(e.count)})),nodeCount:$(e.topology?.nodeCount??0),edgeCount:$(e.topology?.edgeCount??0)}}function Rl(e,t,n){let r=new URL(e);return r.searchParams.set(`demo`,t),r.searchParams.set(`seed`,n),r.searchParams.delete(`frame`),r.searchParams.delete(`capture`),r.searchParams.delete(`receipt`),r.searchParams.delete(`brain`),r.toString()}function zl(e){let t=kl(e),n=Ol(e.byType),r={};for(let[e,t]of n)!_l.includes(e)&&t>0&&(r[e]=t);let i=JSON.stringify({v:t,e:r});return`v1.${btoa(i).replace(/\+/g,`-`).replace(/\//g,`_`).replace(/=+$/g,``)}`}function Bl(e){let t=/^v1\.([A-Za-z0-9_-]+)$/.exec(e.trim());if(!t)return null;try{let e=t[1].replace(/-/g,`+`).replace(/_/g,`/`),n=e.length%4==0?``:`=`.repeat(4-e.length%4),r=JSON.parse(atob(e+n));if(!Array.isArray(r.v)||r.v.length<19)return null;let i=Vl(r.v.map(e=>$(Number(e))));if(r.e&&typeof r.e==`object`)for(let[e,t]of Object.entries(r.e)){let n=El(e);n&&(i.byType[n]=$(t))}return i}catch{return null}}function Vl(e){let t=e.length>=19?e:[...e,...Array(19-e.length).fill(0)],n={};_l.forEach((e,r)=>{let i=$(t[9+r]??0);i>0&&(n[e]=i)});let r=gl.map((e,n)=>({range:e,count:$(t[17+n]??0)}));return{totalMemories:$(t[1]??0),dueForReview:$(t[2]??0),averageRetention:$(t[3]??0)/1e3,embeddingCoverage:$(t[4]??0)/10,endangeredCount:$(t[5]??0),byType:n,retentionBuckets:r,nodeCount:$(t[6]??0),edgeCount:$(t[7]??0)}}function Hl(e,t){let n=Math.min(200,Math.max(0,$(e.nodeCount)||$(e.totalMemories))),r=[...Ol(e.byType).entries()].filter(([,e])=>e>0),i=r.reduce((e,[,t])=>e+t,0)||1,a=[...Dl(e.retentionBuckets).entries()],o=a.reduce((e,[,t])=>e+t,0)||1,s=[];for(let e=0;e<n;e++){let n=(e*2654435761>>>0)%i,c=r[0]?.[0]??`note`;for(let[e,t]of r)if(n-=t,n<0){c=e;break}let l=(e*1597334677>>>0)%o,u=.55;for(let[e,t]of a)if(l-=t,l<0){let t=/^(\d+)\s*-\s*(\d+)/.exec(e);u=((t?Number(t[1]):50)+(t?Number(t[2]):60))/200;break}let d=`syn-${t}-${e.toString(16).padStart(3,`0`)}`;s.push({id:d,label:`${c}·${(e+1).toString().padStart(3,`0`)}`,type:c,retention:u,tags:[],createdAt:`1970-01-01T00:00:00.000Z`,updatedAt:`1970-01-01T00:00:00.000Z`,isCenter:e===0})}let c=n<=0?0:$(e.edgeCount)/Math.max(1,$(e.nodeCount)||n),l=Math.min(n*12,Math.round(n*Math.max(.2,Math.min(8,c)))),u=[],d=new Set;for(let e=0;e<l&&s.length>1;e++){let t=(e*1103515245+12345>>>0)%s.length,n=(e*214013+2531011>>>0)%s.length;n===t&&(n=(n+1)%s.length);let r=s[t].id,i=s[n].id,a=`${r}:${i}`;d.has(a)||(d.add(a),u.push({source:r,target:i,weight:.35+e%65/100,type:`assoc`}))}return{nodes:s,edges:u,center_id:s[0]?.id??``,depth:3,nodeCount:s.length,edgeCount:u.length}}function Ul(e,t,n){let r=Il(n),i=new URL(e);return i.searchParams.set(`demo`,t),i.searchParams.set(`seed`,r.printId),i.searchParams.set(`brain`,zl(n)),i.searchParams.delete(`frame`),i.searchParams.delete(`capture`),i.searchParams.delete(`receipt`),i.toString()}async function Wl(e){let[t,n,r]=await Promise.all([he.stats(),he.retentionDistribution(),e?Promise.resolve(null):he.graph({max_nodes:200,depth:3,sort:`connected`}).catch(()=>null)]),i=Ll({stats:t,retention:n,topology:e??(r?{nodeCount:r.nodeCount,edgeCount:r.edgeCount}:void 0)}),a=Il(i),o=t.oldestMemory,s=t.newestMemory,c=0;if(o&&s){let e=Date.parse(o),t=Date.parse(s);Number.isFinite(e)&&Number.isFinite(t)&&(c=Math.max(0,Math.round(Math.abs(t-e)/864e5)))}return{shape:i,print:a,archiveDays:c}}var Gl=d(`<span class="obs-print-error svelte-t88soz" role="status"> </span>`),Kl=d(`<li class="obs-chip svelte-t88soz"> </li>`),ql=d(`<ul class="obs-traits svelte-t88soz"></ul>`),Jl=d(`<div class="obs-print-card svelte-t88soz" aria-live="polite"><span class="obs-print-kicker svelte-t88soz">structure only · zero memory text</span> <code class="obs-print-id svelte-t88soz"> </code> <!> <button class="obs-print-copy svelte-t88soz" type="button"> </button></div>`),Yl=d(`<div class="obs-print svelte-t88soz"><button class="obs-print-btn svelte-t88soz" type="button"> </button> <!> <!></div>`);function Xl(e,t){ae(t,!0);let r=p(()=>t.print?.traits??[]);var i=Yl(),c=o(i),u=k(c,!0),d=s(c,2),h=e=>{var r=Gl(),i=k(r,!0);n(()=>m(i,t.error)),a(e,r)};j(d,e=>{t.error&&e(h)});var g=s(d,2),_=e=>{var i=Jl(),c=s(o(i),2),u=k(c,!0),d=s(c,2),p=e=>{var t=ql();le(t,21,()=>f(r),e=>e.id,(e,t)=>{var r=Kl(),i=k(r,!0);n(()=>m(i,f(t).label)),a(e,r)}),A(t),a(e,t)};j(d,e=>{f(r).length&&e(p)});var h=s(d,2),g=k(h,!0);A(i),n(()=>{m(u,t.activePrintId),m(g,t.copied?`Permalink copied`:`Copy permalink`)}),l(`click`,h,function(...e){t.oncopy?.apply(this,e)}),a(e,i)};j(g,e=>{t.activePrintId&&e(_)}),A(i),n(()=>{c.disabled=t.disabled||t.printing,E(c,`aria-busy`,t.printing),m(u,t.printing?`Reading shape…`:t.print?`Recompute print`:`Brain print`)}),l(`click`,c,function(...e){t.onprint?.apply(this,e)}),a(e,i),C()}D([`click`]);function Zl(e){let t=0,n=0,r=0;for(let i of e.retentionBuckets){let e=/^(\d+)/.exec(i.range),a=((e?Number(e[1]):0)+5)/100;a>=.7?t+=i.count:a>=.4?n+=i.count:r+=i.count}return{active:t,dormant:n,silent:r}}function Ql(e,t,n,r,i){let a=new N({seed:`${n}:wrapped`}).state.rng,o=kl(t),s=Math.min(420,Math.max(48,Math.round(Math.sqrt(Math.max(1,t.totalMemories))*18))),c=r*.5,l=i*.42;for(let n=0;n<s;n++){let s=a()*Math.PI*2,u=Math.sqrt(a())*Math.min(r,i)*.34,d=c+Math.cos(s)*u*(.7+.6*a()),f=l+Math.sin(s)*u*.85,p=o[17+n%10]??1,m=Math.min(1,p/Math.max(1,t.totalMemories/10)),h=1.2+m*4.5,g=e.createRadialGradient(d,f,0,d,f,h*3);g.addColorStop(0,`rgba(159, 248, 236, ${.55+m*.4})`),g.addColorStop(.45,`rgba(34, 199, 222, ${.25+m*.35})`),g.addColorStop(1,`rgba(2, 3, 7, 0)`),e.fillStyle=g,e.beginPath(),e.arc(d,f,h*3,0,Math.PI*2),e.fill()}}async function $l(e){let t=e.width??1080,n=e.height??1920,r=document.createElement(`canvas`);r.width=t,r.height=n;let i=r.getContext(`2d`);if(!i)throw Error(`2d canvas unavailable`);i.fillStyle=`#020307`,i.fillRect(0,0,t,n);let a=i.createRadialGradient(t*.5,n*.38,40,t*.5,n*.4,t*.55);a.addColorStop(0,`rgba(34, 199, 222, 0.16)`),a.addColorStop(1,`rgba(2, 3, 7, 0)`),i.fillStyle=a,i.fillRect(0,0,t,n),e.fieldBitmap?(i.save(),i.globalAlpha=.55,i.drawImage(e.fieldBitmap,0,n*.12,t,9/16*t),i.restore()):Ql(i,e.shape,e.print.seed,t,n);let o=Zl(e.shape),{print:s,shape:c,archiveDays:l}=e;i.fillStyle=`rgba(234, 255, 251, 0.92)`,i.font=`600 28px ui-sans-serif, system-ui, sans-serif`,i.fillText(`VESTIGE`,72,120),i.fillStyle=`rgba(127, 243, 230, 0.75)`,i.font=`500 22px ui-sans-serif, system-ui, sans-serif`,i.fillText(`MEMORY REPORT`,72,158),i.fillStyle=`#7ff3e6`,i.font=`700 54px ui-monospace, SFMono-Regular, Menlo, monospace`,i.fillText(s.printId,72,260),i.fillStyle=`rgba(140, 199, 180, 0.7)`,i.font=`500 18px ui-sans-serif, system-ui, sans-serif`,i.fillText(`STRUCTURE ONLY  ·  ZERO MEMORY TEXT`,72,300),[[`MEMORIES`,c.totalMemories.toLocaleString()],[`CONNECTIONS`,c.edgeCount.toLocaleString()],[`ACTIVE`,o.active.toLocaleString()],[`DORMANT`,o.dormant.toLocaleString()],[`SILENT`,o.silent.toLocaleString()],[`ARCHIVE`,`${Math.max(0,l)}d`]].forEach(([e,t],n)=>{let r=n%2,a=Math.floor(n/2),o=72+r*480,s=420+a*140;i.fillStyle=`rgba(140, 175, 180, 0.55)`,i.font=`600 18px ui-sans-serif, system-ui, sans-serif`,i.fillText(e,o,s),i.fillStyle=`#eafffb`,i.font=`700 64px ui-sans-serif, system-ui, sans-serif`,i.fillText(t,o,s+70)});let u=72;i.font=`600 22px ui-sans-serif, system-ui, sans-serif`;for(let e of s.traits){let t=i.measureText(e.label).width+36;i.fillStyle=`rgba(20, 48, 40, 0.75)`,i.strokeStyle=`rgba(92, 240, 166, 0.45)`,i.lineWidth=2,eu(i,u,980,t,48,24),i.fill(),i.stroke(),i.fillStyle=`#c8ffe8`,i.fillText(e.label,u+18,1012),u+=t+16}return i.fillStyle=`rgba(140, 199, 219, 0.55)`,i.font=`500 20px ui-sans-serif, system-ui, sans-serif`,i.fillText(`share your brain, not your memories`,72,n-80),{blob:await new Promise((e,t)=>{r.toBlob(n=>n?e(n):t(Error(`toBlob failed`)),`image/png`)}),width:t,height:n,filename:`vestige-${s.printId}-report.png`}}function eu(e,t,n,r,i,a){e.beginPath(),e.moveTo(t+a,n),e.arcTo(t+r,n,t+r,n+i,a),e.arcTo(t+r,n+i,t,n+i,a),e.arcTo(t,n+i,t,n,a),e.arcTo(t,n,t+r,n,a),e.closePath()}function tu(e,t){console.info(`[wrapped-card] ${t}: ${(e.size/1e3).toFixed(0)}KB`);let n=URL.createObjectURL(e),r=document.createElement(`a`);r.href=n,r.download=t,r.click(),setTimeout(()=>URL.revokeObjectURL(n),1e4)}var nu=d(`This field replays only the recorded Backfill candidate evidence in receipt <code> </code>.`,1),ru=d(`This field contains only memories named by receipt <code> </code>.`,1),iu=d(`<span class="obs-stat obs-proof svelte-2ll0b2"><b class="svelte-2ll0b2">Proven:</b> retrieved in this run</span> <span class="obs-stat obs-attributed svelte-2ll0b2"><b class="svelte-2ll0b2">Attributed:</b> likely influence</span>`,1),au=d(`<span class="obs-stat obs-err svelte-2ll0b2"><b class="svelte-2ll0b2">!</b> </span>`),ou=d(`<span class="obs-stat svelte-2ll0b2"><b class="svelte-2ll0b2">…</b> loading field</span>`),su=d(`<span class="obs-stat svelte-2ll0b2"><b class="svelte-2ll0b2"> </b> memories</span> <span class="obs-stat svelte-2ll0b2"><b class="svelte-2ll0b2"> </b> connections</span> <span class="obs-stat svelte-2ll0b2"><b class="svelte-2ll0b2"> </b> center</span>`,1),cu=d(`<span class="obs-demos-label svelte-2ll0b2">Receipt evidence only</span> <span class="obs-card is-active receipt-scope svelte-2ll0b2"><span class="obs-card-label svelte-2ll0b2"> </span> <span class="obs-card-blurb svelte-2ll0b2">This proves memory retrieval. It does not claim an answer changed.</span></span>`,1),lu=d(`<button type="button"><span class="obs-card-label svelte-2ll0b2"> </span> <span class="obs-card-blurb svelte-2ll0b2"> </span></button>`),uu=d(`<span class="obs-demos-label svelte-2ll0b2">Play a moment</span> <!>`,1),du=d(`<span class="obs-export-error svelte-2ll0b2"> </span>`),fu=d(`<div style="position: fixed; left: -100000px; top: 0; width: 1920px; height: 1080px;" aria-hidden="true"><!></div>`),pu=d(`<div class="fixed inset-0 bg-[#020307]"><!></div> <div class="obs-ui svelte-2ll0b2"><header class="obs-head svelte-2ll0b2"><h1 class="obs-title svelte-2ll0b2"> </h1> <p class="obs-sub svelte-2ll0b2"><!></p> <div class="obs-stats svelte-2ll0b2"><!> <!></div></header> <nav class="obs-demos svelte-2ll0b2"><!> <!> <button class="obs-exit svelte-2ll0b2" type="button"> </button> <button class="obs-exit svelte-2ll0b2" type="button"> </button> <!> <a class="obs-exit svelte-2ll0b2">Open full graph →</a></nav></div> <!>`,1);function mu(e,i){ae(i,!0);let d=new URLSearchParams(window.location.search),g=d.get(`receipt`),_=d.get(`brain`),S=g?`recall-path`:d.get(`demo`)??`recall-path`,w=b(y(ge(S)?S:`recall-path`)),T=b(y(d.get(`seed`)??`vestige-observatory-v1`)),te=d.get(`frame`),D=te!==null&&te!==``?Number(te):null,oe=d.has(`capture`)||D!==null;[...Ee(we.forward)],[...Ee(Se.recall)],[...Ee(Se.luciferin)],[...Ee(be.veto)],[...Ee(be.caution)];let O=b(null),ce=null,pe=null,M=null,N=b(null),P=b(null),ye=b(null),F=b(!0),Ce=b(null),I=null,L=null,R=new Set,De=[],Ae=typeof window<`u`&&window.matchMedia(`(prefers-reduced-motion: reduce)`).matches;se(()=>{Le()}),ie(()=>{De.forEach(clearTimeout),De=[],pe?.dispose(),M?.dispose(),pe=null,M=null,ce=null});async function je(e){ce=e;let t=new Oe(e);M=t;let n=Be();t.setIntensity(n===null?.2:.16),t.setReadingWell({x:0,y:0,hw:1.15,hh:1.15,floor:.06,soft:.4}),t.setCells(Ne(R)),e.addPass(t);let r=new Te(e);pe=r,await r.init(),r.setText(Ve()),e.addPass(r),e.demoClock.reset()}function Me(e){let t=z(e.retention),n=Math.max(0,Number.isFinite(e.stability)?e.stability:0),r=z(Math.log10(1+n)/5),i=.3;e.lastAccessed&&(i=z(1-(Date.now()-new Date(e.lastAccessed).getTime())/864e5/30));let a=.5*t+.32*r+.18*i;return z(e.isCenter?Math.max(a,.9):a)}function Ne(e=new Set){let t=f(N)?.nodes??[],n=[...t].map(e=>({id:e.id,s:Me(e)})).sort((e,t)=>e.s-t.s),r=new Map,i=Math.max(1,n.length-1);n.forEach((e,t)=>r.set(e.id,t/i));let a=t.map(t=>{let n=r.get(t.id)??0,i=e.has(t.id),a=t.isCenter?Math.max(n,.92):n;return{id:t.id,score:a,hue:ve(a,i),energy:_e(a,i),selected:!!t.isCenter||i,scar:(t.suppression_count??0)>0,metric2:z(t.retention),kind:`observatory-cell`,payload:t}});return ke(a,{maxRadius:.95,minCellR:.01,maxCellR:.04})}function Pe(e){let t=[...f(N)?.nodes??[]].map(e=>({id:e.id,s:Me(e)})).sort((e,t)=>t.s-e.s).slice(0,e).map(e=>e.id);return new Set(t)}let Fe=p(()=>{let e=me(f(P));if(e)return{failureId:e.failure_id,pathIds:e.path_ids,candidates:e.candidates.map(e=>({memoryId:e.memory_id,sharedEntities:e.shared_entities,ageDays:e.age_days_before_failure,similarityRank:e.similarity_rank,promoted:e.promoted}))}}),Ie=p(()=>{if(!f(P))return[];let e=me(f(P));return[...new Set([...f(P).retrieved,...f(P).suppressed.map(e=>e.id),...e?[e.failure_id,...e.path_ids??[],...e.candidates.map(e=>e.memory_id)]:[]])]});async function Le(){if(g)try{v(P,await he.receipts.get(g),!0),me(f(P))&&v(w,`salience-rescue`)}catch(e){v(ye,e instanceof Error?e.message:`Receipt unavailable`,!0)}else if(_){let e=Bl(_);if(e){v(tt,e,!0);let t=Il(e);v(et,t,!0),v(T,t.seed,!0),v(N,Hl(e,t.printId),!0),v(ot,f(N),!0),v(F,!1);return}}await Re()}async function Re(){v(F,!0),v(Ce,null),pe?.setText(Ve());try{let e=new Set(f(Ie));if(e.size){let t=await Promise.all([...e].map(e=>he.graph({center_id:e,max_nodes:200,depth:3}))),n=t.flatMap(e=>e.nodes),r=t.flatMap(e=>e.edges),i=[...new Map(n.map(e=>[e.id,e])).values()].filter(t=>e.has(t.id)),a=new Set(i.map(e=>e.id)),o=[...new Map(r.map(e=>[`${e.source}:${e.target}`,e])).values()].filter(e=>a.has(e.source)&&a.has(e.target));v(N,{...t[0],nodes:i,edges:o,center_id:i[0]?.id??t[0]?.center_id??``,nodeCount:i.length,edgeCount:o.length},!0)}else v(N,await he.graph({max_nodes:200,depth:3,sort:`connected`}),!0)}catch(e){v(N,null),v(Ce,e instanceof Error?e.message:`UNKNOWN OBSERVATORY GRAPH ERROR`,!0)}finally{v(F,!1),pe?.setText(Ve()),ze(),ce?.demoClock.reset()}}function ze(){De.forEach(clearTimeout),De=[],R=new Set;let e=[...Pe(7)];if(!e.length){M?.setCells(Ne(R));return}if(D!==null||Ae){R=new Set(e),M?.setCells(Ne(R));return}M?.setCells(Ne(R)),e.forEach((e,t)=>{De.push(setTimeout(()=>{R.add(e),M?.setCells(Ne(R))},420+t*130))})}function z(e){return Math.min(1,Math.max(0,Number.isFinite(e)?e:.5))}function Be(){let e=ce?.params[6]||0,t=ce?.params[7]||0;if((e<=0||t<=0)&&typeof window<`u`&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0)return null;let n=e/t;return n<.85?n:null}function Ve(){return[]}let He=b(!1),Ue=b(null),We=b(null),Ge=null,Ke=b(0);function qe(e){Ge=e}let Je=()=>Ge;async function Ye(){if(!f(He)){if(!dl()){v(We,`This browser cannot encode video (WebCodecs unavailable).`);return}v(He,!0),v(We,null),v(Ue,null),Ge=null,v(Ke,f(Ke)+1);try{let e=Date.now()+2e4,t=Je();for(;!t||t.params[2]<=0;){if(Date.now()>e)throw Error(`export stage never became ready`);await new Promise(e=>setTimeout(e,120)),t=Je()}await new Promise(e=>setTimeout(e,400)),ml(await fl({engine:t,onProgress:e=>v(Ue,e,!0)}),f(P)?pl(f(P).receipt_id,f(w)):Cl(f(T),f(w)))}catch(e){v(We,e instanceof Error?e.message:`Export failed`,!0)}finally{v(He,!1),v(Ue,null),Ge=null}}}function Xe(e){if(e===f(w))return;v(w,e,!0);let t=new URL(window.location.href);t.searchParams.set(`demo`,e),history.replaceState(history.state,``,t),pe?.setText(Ve()),ce?.demoClock.reset()}let Ze=[{mode:`recall-path`,label:`Recall`,blurb:`Watch a memory get retrieved — the path lights up.`},{mode:`engram-birth`,label:`Engram`,blurb:`A new memory forms and wires into the field.`},{mode:`salience-rescue`,label:`Salience`,blurb:`The few memories that matter ignite gold.`},{mode:`forgetting-horizon`,label:`Forgetting`,blurb:`FSRS decay pulls weak memories toward the dark.`},{mode:`firewall`,label:`Firewall`,blurb:`A contradiction is caught and quarantined.`}].filter(e=>xe.includes(e.mode));function Qe(){return f(N)?.center_id?.slice(0,8)??`—`}function $e(e){f(P)||Xe(e)}let et=b(null),tt=b(null),nt=b(!1),rt=b(!1),it=b(null),at=b(!1),ot=b(null),st=p(()=>f(et)?.printId??(xl(f(T))?f(T):null));function ct(e,t){v(T,e,!0);let n=new URL(window.location.href);n.searchParams.set(`seed`,e),t&&n.searchParams.set(`brain`,zl(t)),history.replaceState(history.state,``,n)}async function lt(){if(!f(nt)){v(nt,!0),v(it,null),v(at,!1);try{let e=await Wl(f(P)||!f(N)?void 0:{nodeCount:f(N).nodeCount,edgeCount:f(N).edgeCount});v(et,e.print,!0),v(tt,e.shape,!0),ct(e.print.printId,e.shape),console.info(`[brain-print] ${e.print.printId} traits=${e.print.traits.map(e=>e.id).join(`,`)} vector=${e.print.vector.length}`)}catch(e){v(it,e instanceof Error?e.message:`Brain print failed`,!0)}finally{v(nt,!1)}}}async function ut(){let e=f(st);if(!e)return;let t=f(tt)?Ul(window.location.href,f(w),f(tt)):Rl(window.location.href,f(w),e);try{await navigator.clipboard.writeText(t),v(at,!0),console.info(`[brain-print] permalink ${t}`)}catch{v(it,`Clipboard unavailable — copy the URL from the address bar.`)}}async function dt(){if(!f(rt)){v(rt,!0),v(it,null);try{let e=f(P)||!f(N)?void 0:{nodeCount:f(N).nodeCount,edgeCount:f(N).edgeCount},t=f(tt)&&f(et)?{shape:f(tt),print:f(et),archiveDays:0}:await Wl(e);f(tt)||(v(tt,t.shape,!0),v(et,t.print,!0),ct(t.print.printId,t.shape));let n=await $l({shape:t.shape,print:t.print,archiveDays:t.archiveDays,fieldBitmap:ce?.canvasElement??null});tu(n.blob,n.filename)}catch(e){v(it,e instanceof Error?e.message:`Memory report failed`,!0)}finally{v(rt,!1)}}}function ft(e){if(!f(O))return null;let t=f(O).getBoundingClientRect();return t.width<=0||t.height<=0?null:{x:(e.clientX-t.left)/t.width*2-1,y:-((e.clientY-t.top)/t.height*2-1)}}function pt(e){if(!f(O)||!ce)return;let t=f(O).getBoundingClientRect(),n=Math.max(1e-4,t.width/Math.max(1,t.height)),r={x:e.x*Math.max(n,1),y:e.y/Math.min(n,1)},i=L??r,a={x:i.x+(r.x-i.x)*.35,y:i.y+(r.y-i.y)*.35};L=a,ce.setCursorPreNdc(a.x,a.y,a.x-i.x,a.y-i.y)}function mt(e){let t=ft(e);if(!t)return;pt(t);let n=pe?.pickAt(t.x,t.y)??null,r=n?.kind===`observatory-demo`||n?.kind===`observatory-exit`?n.id:null;r!==I&&(I=r,pe?.setRunDepth(r,1)),f(O)&&(f(O).style.cursor=r?`crosshair`:`default`)}function ht(){L=null,I=null,ce?.setCursorPreNdc(999,999,0,0),pe?.setRunDepth(null),f(O)&&(f(O).style.cursor=`default`)}async function gt(e){let t=ft(e);if(!t)return;let n=pe?.pickAt(t.x,t.y)?.payload;if(n?.action===`demo`&&n.demo){Xe(n.demo);return}if(n?.action===`exit`){await fe(`${de}/graph`);return}let r=M?.pickAt(t.x,t.y);r&&typeof r.id==`string`&&await fe(`${de}/memories?memory=${encodeURIComponent(r.id)}`)}var _t=pu();x(`2ll0b2`,e=>{ue(()=>{h.title=`Observatory · Vestige`})});var vt=c(_t),B=o(vt);r(B,()=>`${f(w)}:${f(T)}:${f(P)?.receipt_id??`all`}:${f(ot)?`brain`:`live`}`,e=>{{let t=p(()=>!f(ot));Sr(e,{get demo(){return f(w)},get seed(){return f(T)},get freezeFrame(){return D},get capture(){return oe},showSwitcher:!1,chrome:`none`,get live(){return f(t)},get graphOverride(){return f(ot)},get focusIds(){return f(Ie)},get backfillEvidence(){return f(Fe)},onready:je,onexit:()=>fe(`${de}/graph`)})}}),A(vt),ee(vt,e=>v(O,e),()=>f(O));var V=s(vt,2),yt=o(V),bt=o(yt),xt=k(bt,!0),St=s(bt,2),Ct=o(St),wt=e=>{var t=nu(),r=s(c(t)),i=k(r,!0);ne(),n(()=>m(i,f(P)?.receipt_id)),a(e,t)},Tt=e=>{var t=ru(),r=s(c(t)),i=k(r,!0);ne(),n(()=>m(i,f(P).receipt_id)),a(e,t)},Et=e=>{var n=t(`Your agent's live memory field. Play a cognitive moment and watch the mind react.`);a(e,n)};j(Ct,e=>{f(Fe)?e(wt):f(P)?e(Tt,1):e(Et,-1)}),A(St);var Dt=s(St,2),Ot=o(Dt),kt=e=>{var t=iu();ne(2),a(e,t)},At=e=>{var t=au(),r=s(o(t));A(t),n(()=>m(r,` ${f(ye)??``}`)),a(e,t)};j(Ot,e=>{f(P)?e(kt):f(ye)&&e(At,1)});var jt=s(Ot,2),Mt=e=>{var t=ou();a(e,t)},Nt=e=>{var t=au(),r=s(o(t));A(t),n(()=>m(r,` ${f(Ce)??``}`)),a(e,t)},Pt=e=>{var t=su(),r=c(t),i=o(r),l=k(i,!0);ne(),A(r);var u=s(r,2),d=o(u),p=k(d,!0);ne(),A(u);var h=s(u,2),g=o(h),_=k(g,!0);ne(),A(h),n((e,t,n)=>{m(l,e),m(p,t),m(_,n)},[()=>f(N).nodeCount.toLocaleString(),()=>f(N).edgeCount.toLocaleString(),()=>Qe()]),a(e,t)};j(jt,e=>{f(F)?e(Mt):f(Ce)?e(Nt,1):f(N)&&e(Pt,2)}),A(Dt),A(yt);var Ft=s(yt,2),It=o(Ft),Lt=e=>{var t=cu(),r=s(c(t),2),i=o(r),l=k(i);ne(2),A(r),n(()=>m(l,`${f(P).retrieved.length??``} retrieved · ${f(P).suppressed.length??``} suppressed`)),a(e,t)},Rt=e=>{var t=uu(),r=s(c(t),2);le(r,17,()=>Ze,e=>e.mode,(e,t)=>{var r=lu(),i=o(r),c=k(i,!0),u=s(i,2),d=k(u,!0);A(r),n(()=>{re(r,1,`obs-card ${f(t).mode===f(w)?`is-active`:``}`,`svelte-2ll0b2`),E(r,`aria-pressed`,f(t).mode===f(w)),m(c,f(t).label),m(d,f(t).blurb)}),l(`click`,r,()=>$e(f(t).mode)),a(e,r)}),a(e,t)};j(It,e=>{f(P)?e(Lt):e(Rt,-1)});var zt=s(It,2);{let e=p(()=>f(F)||f(He)||f(rt));Xl(zt,{get print(){return f(et)},get activePrintId(){return f(st)},get printing(){return f(nt)},get error(){return f(it)},get copied(){return f(at)},get disabled(){return f(e)},onprint:lt,oncopy:ut})}var Bt=s(zt,2),Vt=k(Bt,!0),Ht=s(Bt,2),Ut=k(Ht,!0),Wt=s(Ht,2),Gt=e=>{var t=du(),r=k(t,!0);n(()=>m(r,f(We))),a(e,t)};j(Wt,e=>{f(We)&&e(Gt)});var Kt=s(Wt,2);A(Ft),A(V);var qt=s(V,2),Jt=e=>{var t=fu(),n=o(t);r(n,()=>f(Ke),e=>{Sr(e,{get demo(){return f(w)},get seed(){return f(T)},showSwitcher:!1,chrome:`none`,embedded:!0,maxDpr:1,get graphOverride(){return f(ot)},get focusIds(){return f(Ie)},get backfillEvidence(){return f(Fe)},onready:qe})}),A(t),a(e,t)};j(qt,e=>{f(He)&&e(Jt)}),n(()=>{m(xt,f(Fe)?`Backfill Replay`:f(P)?`Memory Replay`:`Cognitive Observatory`),E(Ft,`aria-label`,f(P)?`Receipt evidence`:`Cognitive moments`),Bt.disabled=f(rt)||f(He),m(Vt,f(rt)?`Rendering report…`:`Memory report PNG`),Ht.disabled=f(He),m(Ut,f(He)?f(Ue)?f(Ue).stage===`finalize`?`Sealing clip…`:`Rendering ${f(Ue).done}/${f(Ue).total}`:`Preparing…`:f(P)?`Export receipt replay ↓`:`Export loop ↓`),E(Kt,`href`,`${de??``}/graph`)}),l(`pointerdown`,vt,gt),l(`pointermove`,vt,mt),u(`pointerleave`,vt,ht),l(`click`,Bt,dt),l(`click`,Ht,Ye),a(e,_t),C()}D([`pointerdown`,`pointermove`,`click`]);export{mu as component,Pe as universal};
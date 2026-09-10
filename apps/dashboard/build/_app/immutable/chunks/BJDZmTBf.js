import{$ as e,A as t,D as n,F as r,I as i,J as a,L as o,O as s,P as c,Q as l,U as u,W as d,X as f,Y as p,Z as m,_ as h,a as g,c as _,d as v,et as y,f as b,g as x,it as S,j as C,k as w,lt as T,mt as E,pt as D,q as ee,r as O,rt as te,tt as k,u as ne,ut as re,w as ie}from"./Cmba2AGc.js";import{s as ae}from"./DUBTf18l.js";import"./xihTtKlq.js";import{t as oe}from"./ABVBd11K.js";import{t as se}from"./Cwq8aIcs.js";import{a as A,d as ce,i as j,l as le,n as ue,o as M,r as N,s as P,t as de,u as fe}from"./1HeR1uoy.js";import{t as pe}from"./B4UepgbG.js";import{r as me}from"./D2q35Vo6.js";import{i as F,r as I,t as L}from"./BciDxNzc.js";function he(e,t){let n=[];for(let r of e){let e=((r.activation_path?.length?r.activation_path:r.retrieved)??[]).filter(t);if(e.length===0)continue;let i=e[e.length-1];n.push({targetId:i,pathIds:e})}return n}function ge(e,t=12){return[...e].sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,t).map(e=>({targetId:e.id,pathIds:[e.id]}))}function _e(e,t,n=5){let r=new Map;for(let n of e){let e=(n.activation_path?.length?n.activation_path:n.retrieved)??[];for(let n of new Set(e))t(n)&&r.set(n,(r.get(n)??0)+1)}return[...r.entries()].map(([e,t])=>({id:e,recalls:t})).sort((e,t)=>t.recalls-e.recalls||e.id.localeCompare(t.id)).slice(0,n)}var ve=class{bridge;items=[];cursor=0;ticks=0;nextTick=0;intervalFrames;enabled=!0;started=!1;constructor(e,t={}){this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}let t=this.items[this.cursor%this.items.length];this.cursor++;let n=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(n?0:30)}},R=C(`<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>`),ye=C(`<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>`),be=C(`<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>`),xe=C(`<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>`);function Se(e,t){re(t,!0);let i=g(t,`demoMode`,3,`recall-path`),o=g(t,`seed`,3,`vestige-observatory-v1`),c=g(t,`nodeCount`,3,0),l=g(t,`edgeCount`,3,0),d=g(t,`centerId`,3,``),p=g(t,`frameCount`,3,0),h=g(t,`fpsEstimate`,3,0),_=g(t,`freezeFrame`,3,null);g(t,`loading`,3,!1),g(t,`error`,3,``);function v(){let e=new URLSearchParams({demo:i(),seed:o()});_()!==null&&e.set(`frame`,String(_()));let t=`${window.location.origin}${ae}/observatory?${e.toString()}`;navigator.clipboard.writeText(t).catch(()=>{})}var y=xe(),b=a(y),x=a(b),S=a(x),C=f(S,!0),D=m(S,2),ee=f(D);E(x);var O=m(x,2),te=a(O),k=f(te),ne=m(te,2),ie=e=>{var t=R(),n=f(t);u(e=>s(n,`center=${e??``}`),[()=>d().slice(0,8)]),w(e,t)};n(ne,e=>{d()&&e(ie)}),E(O);var oe=m(O,2),se=a(oe),A=f(se),ce=m(se,2),j=e=>{var t=ye();w(e,t)},le=e=>{var t=be(),n=f(t);u(()=>s(n,`${h()??``}fps`)),w(e,t)};n(ce,e=>{_()===null?h()>0&&e(le,1):e(j)});var ue=m(ce,2);E(oe),E(b),E(y),u((e,t)=>{s(C,i()),s(ee,`seed=${e??``}${o().length>12?`…`:``}`),s(k,`${c()??``} nodes · ${l()??``} edges`),s(A,`frame: ${t??``}`)},[()=>o().slice(0,12),()=>String(p()).padStart(3,` `)]),r(`click`,ue,v),w(e,y),T()}c([`click`]);var Ce=C(`<div class="active-label svelte-8n8iia"> </div>`),z=C(`<div></div>`),we=C(`<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>`);function Te(e,r){re(r,!0);let i=g(r,`steps`,19,()=>[]),c=g(r,`frame`,3,0),l=g(r,`loopFrames`,3,720),d=e=>e/l()*100;function _(e,t){let n=t-e;return n<-14||n>90?0:n<0?1+n/14:1-n/90}let v=k(()=>{let e=``,t=.15;for(let n of i()){let r=_(n.beatFrame,c());r>t&&(t=r,e=n.label)}return e});var y=t(),S=p(y),C=e=>{var t=we(),r=a(t),l=e=>{var t=Ce(),n=f(t,!0);u(()=>s(n,o(v))),w(e,t)};n(r,e=>{o(v)&&e(l)});var p=m(r,2),g=a(p);ie(g,17,i,e=>e.beatFrame,(e,t)=>{var n=z();let r;u((e,i,a)=>{r=h(n,1,`tick svelte-8n8iia`,null,r,{hot:e,backward:o(t).kind===1}),x(n,`left: ${i??``}%; opacity: ${a??``}`),b(n,`title`,o(t).label)},[()=>_(o(t).beatFrame,c())>0,()=>d(o(t).beatFrame),()=>.45+.55*_(o(t).beatFrame,c())]),w(e,n)});var y=m(g,2);E(p),E(t),u(e=>x(y,`left: ${e??``}%`),[()=>d(c())]),w(e,t)};n(S,e=>{i().length>0&&e(C)}),w(e,y),T()}var Ee=C(`<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>`);function De(e,r){re(r,!0);let i=g(r,`frame`,3,0),c=g(r,`fadeWindow`,19,()=>[600,620,705,719]),l=g(r,`tone`,3,`triumph`),d=(e,t,n)=>{let r=Math.min(1,Math.max(0,(n-e)/(t-e)));return r*r*(3-2*r)},_=k(()=>d(c()[0],c()[1],i())*(1-d(c()[2],c()[3],i())));var v=t(),y=p(v),b=e=>{var t=Ee();let n;var i=a(t),c=f(i,!0),d=m(i,2),p=f(d,!0),g=m(d,2),v=f(g,!0);E(t),u(()=>{n=h(t,1,`verdict svelte-ssd7yu`,null,n,{quarantine:l()===`quarantine`}),x(t,`opacity: ${o(_)??``}`),s(c,r.verdict.headline),s(p,r.verdict.causeLabel),s(v,r.verdict.receipt)}),w(e,t)};n(y,e=>{o(_)>.001&&e(b)}),w(e,v),T()}function Oe(e,t,n,r){let i=1/Math.tan(e/2),a=1/(n-r),o=new Float32Array(16);return o[0]=i/t,o[5]=i,o[10]=r*a,o[11]=-1,o[14]=r*n*a,o}function ke(e,t,n){let[r,i,a]=e,o=r-t[0],s=i-t[1],c=a-t[2],l=Math.hypot(o,s,c)||1;o/=l,s/=l,c/=l;let u=n[1]*c-n[2]*s,d=n[2]*o-n[0]*c,f=n[0]*s-n[1]*o;l=Math.hypot(u,d,f)||1,u/=l,d/=l,f/=l;let p=s*f-c*d,m=c*u-o*f,h=o*d-s*u,g=new Float32Array(16);return g[0]=u,g[1]=p,g[2]=o,g[4]=d,g[5]=m,g[6]=s,g[8]=f,g[9]=h,g[10]=c,g[12]=-(u*r+d*i+f*a),g[13]=-(p*r+m*i+h*a),g[14]=-(o*r+s*i+c*a),g[15]=1,g}function B(e,t){let n=new Float32Array(16);for(let r=0;r<4;r++)for(let i=0;i<4;i++)n[r*4+i]=e[i]*t[r*4]+e[4+i]*t[r*4+1]+e[8+i]*t[r*4+2]+e[12+i]*t[r*4+3];return n}function Ae(e,t,n,r=.35,i=0){let a=e*Math.PI*2+i,o=[Math.sin(a)*n,n*r,Math.cos(a)*n],s=Oe(50*Math.PI/180,t,.1,4e3),c=ke(o,[0,0,0],[0,1,0]),l=-o[0],u=-o[1],d=-o[2],f=Math.hypot(l,u,d)||1;l/=f,u/=f,d/=f;let p=u*0-d*1,m=d*0-l*0,h=l*1-u*0;f=Math.hypot(p,m,h)||1,p/=f,m/=f,h/=f;let g=m*d-h*u,_=h*l-p*d,v=p*u-m*l;return{viewProj:B(s,c),right:[p,m,h],up:[g,_,v],eye:o}}var V={yaw:0,pitch:0,zoom:1},je=.38,Me=2.6,Ne=-.18,Pe=.82;function Fe(e,t,n){return Math.min(n,Math.max(t,e))}function Ie(e){return{yaw:Number.isFinite(e.yaw)?e.yaw:0,pitch:Fe(Number.isFinite(e.pitch)?e.pitch:0,Ne,Pe),zoom:Fe(Number.isFinite(e.zoom)?e.zoom:1,je,Me)}}function Le(e,t,n,r=V){let i=Ie(r);return Ae(e,t,n/i.zoom,.35+i.pitch,i.yaw)}var Re=class{state={...V};dragging=!1;pointerId=null;lastX=0;lastY=0;pinch0=0;pointers=new Map;enabled=!0;reset(){this.state={...V},this.dragging=!1,this.pointerId=null,this.pointers.clear()}onPointerDown(e){this.enabled&&e.button===0&&(this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===1?(this.dragging=!0,this.pointerId=e.pointerId,this.lastX=e.clientX,this.lastY=e.clientY,e.currentTarget?.setPointerCapture?.(e.pointerId)):this.pointers.size===2&&(this.pinch0=ze(this.pointers)))}onPointerMove(e){if(!this.enabled)return!1;if(this.pointers.has(e.pointerId)&&this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===2&&this.pinch0>0){let e=ze(this.pointers),t=e/this.pinch0;return this.state=Ie({...this.state,zoom:this.state.zoom*Fe(t,.94,1.06)}),this.pinch0=e,!0}if(!this.dragging||e.pointerId!==this.pointerId)return!1;let t=e.clientX-this.lastX,n=e.clientY-this.lastY;return this.lastX=e.clientX,this.lastY=e.clientY,this.state=Ie({yaw:this.state.yaw-t*.005,pitch:this.state.pitch+n*.003,zoom:this.state.zoom}),!0}onPointerUp(e){this.pointers.delete(e.pointerId),e.pointerId===this.pointerId&&(this.dragging=!1,this.pointerId=null),this.pointers.size<2&&(this.pinch0=0)}onWheel(e){if(!this.enabled)return!1;e.preventDefault();let t=e.deltaY>0?.92:1.08;return this.state=Ie({...this.state,zoom:this.state.zoom*t}),!0}};function ze(e){let t=[...e.values()];return t.length<2?0:Math.hypot(t[0].x-t[1].x,t[0].y-t[1].y)}function Be(e){let t=/^#?([0-9a-fA-F]{6})$/.exec(e.trim());if(!t)return[107/255,114/255,128/255];let n=parseInt(t[1],16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}function H(e){return Be(I({tags:e.tags})||L[F(e.retention)])}function Ve(e){let t=[...e.nodes].sort((e,t)=>e.isCenter===t.isCenter?e.id<t.id?-1:+(e.id>t.id):e.isCenter?-1:1).map((e,t)=>le(e,t)),n=new Map;for(let e of t)n.set(e.id,e.index);let r=[];for(let t of e.edges){let e=n.get(t.source),i=n.get(t.target);e!==void 0&&i!==void 0&&e!==i&&r.push({sourceIndex:e,targetIndex:i,weight:t.weight,type:t.type})}let i=t.findIndex(e=>e.isCenter);return{nodes:t,edges:r,indexById:n,centerIndex:i<0?0:i}}function He(e,t,n=120){let r=e.nodes.length,i=new Float32Array(r*16);for(let a=0;a<r;a++){let o=e.nodes[a],s=a*16,[c,l,u]=o.isCenter&&e.centerIndex===a?[0,0,0]:ce(a,r,n,t),d=o.isCenter?4.2:1.4+o.retention*1.8;i[s+A.posRadius+0]=c,i[s+A.posRadius+1]=l,i[s+A.posRadius+2]=u,i[s+A.posRadius+3]=d,i[s+A.velRetention+3]=o.retention;let[f,p,m]=H(o),h=0;o.isCenter&&(h|=j.isCenter),o.suppressed&&(h|=j.suppressed);let g=new Set(o.tags.map(e=>e.toLowerCase()));g.has(`aha`)&&(h|=j.isAha),(g.has(`failure`)||g.has(`guardrail`))&&(h|=j.isFailure),(g.has(`confusion`)||g.has(`weak-spot`))&&(h|=j.isConfusion),i[s+A.colorFlags+0]=f,i[s+A.colorFlags+1]=p,i[s+A.colorFlags+2]=m,i[s+A.colorFlags+3]=h}return{data:i,nodeCount:r}}function Ue(e){let t=new Uint32Array(Math.max(1,e.edges.length)*2);return e.edges.forEach((e,n)=>{t[n*2]=e.sourceIndex,t[n*2+1]=e.targetIndex}),t}var We=`
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
`,Ge=`
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
`,Ke=`
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
`,U=`
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
`;function W(e){return 60+e*60}function qe(e,t,n=8,r={}){let i=[...e.nodes].sort((e,t)=>e.id<t.id?-1:+(e.id>t.id)),a=[...e.edges].sort((e,t)=>{let n=`${e.source} ${e.target} ${e.type}`,r=`${t.source} ${t.target} ${t.type}`;return n<r?-1:+(n>r)}),o=r.centerId??e.center_id,s=me(i,a,o,n,{preferCausal:r.preferCausal}),c=[];for(let e=0;e<s.beats.length;e++){let n=s.beats[e],r=t.indexById.get(n.nodeId);if(r===void 0)continue;let i=e>0?s.beats[e-1].nodeId:n.nodeId,a=t.indexById.get(i)??r,o=(n.viaEdge?.type??``).toLowerCase(),l=o===`causal`||o.includes(`causal`),u=n.kind===`contradiction`||l;c.push({sourceIndex:a,targetIndex:r,beatFrame:W(e),kind:u?P.backwardCause:P.recall,beatKind:n.kind,nodeId:n.nodeId,label:n.node.label})}let l=new Uint32Array(Math.max(1,c.length)*4);return c.forEach((e,t)=>{l[t*4]=e.sourceIndex,l[t*4+1]=e.targetIndex,l[t*4+2]=e.beatFrame,l[t*4+3]=e.kind}),{data:l,steps:c,path:s}}var Je=24,Ye=300,Xe=128,Ze=class{engine;pipeline=null;bindGroup=null;cameraBuffer=null;nodeBuffer=null;edgeBuffer=null;cameraData=new Float32Array(Je);nodeCount=0;simPipeline=null;simBindGroup=null;pathBuffer=null;liveRetentionBuffer=null;pickReadback=null;disposed=!1;edgeCapacityBytes=0;edgeCount=0;cameraRig={...V};hoveredIndex=-1;pathPipeline=null;pathBindGroup=null;pathStepCount=0;axonPipeline=null;axonBindGroup=null;graph=null;pathSteps=[];constructor(e){this.engine=e,e.addPass(this)}upload(e,t,n){let r=this.engine.gpuDevice;if(!r)return;let i=n?.recallPath??!0,a=Ve(e);this.graph=a;let{data:o,nodeCount:s}=He(a,new fe({seed:t}).state.rng);this.nodeCount=s,this.nodeBuffer?.destroy(),this.nodeBuffer=r.createBuffer({label:`observatory-node-state`,size:Math.max(o.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),r.queue.writeBuffer(this.nodeBuffer,0,o.buffer);let c=Ue(a);this.edgeCount=a.edges.length,this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(c.byteLength*2,64),this.edgeBuffer=r.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.edgeBuffer,0,c.buffer);let l=new Float32Array(Math.max(s,4));for(let e=0;e<s;e++)l[e]=Math.max(.001,a.nodes[e].retention);this.liveRetentionBuffer?.destroy(),this.liveRetentionBuffer=r.createBuffer({label:`observatory-live-retention`,size:l.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.liveRetentionBuffer,0,l.buffer);let u=i?qe(e,a):{steps:[],data:new Uint32Array(4)};this.pathSteps=u.steps,this.pathBuffer?.destroy(),this.pathBuffer=r.createBuffer({label:`observatory-path-steps`,size:Xe*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,u.data.buffer,0,Math.min(u.data.byteLength,Xe*4*4)),this.pathStepCount=Math.min(this.pathSteps.length,Xe),this.engine.params[2]=s,this.engine.params[3]=a.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||=r.createBuffer({label:`observatory-camera`,size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.createPipeline(r)}setPathSteps(e,t){let n=this.engine.gpuDevice;if(!n)return;this.pathSteps=t;let r=Xe*4*4;if(this.pathBuffer&&e.byteLength<=r){this.pathStepCount=Math.min(t.length,Xe),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,Xe),this.pathBuffer?.destroy(),this.pathBuffer=n.createBuffer({label:`observatory-path-steps`,size:r,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,r)),this.engine.params[4]=this.pathStepCount,this.createPipeline(n)}setCameraRig(e){this.cameraRig=e}setHovered(e){this.hoveredIndex=e}currentOrbit(){let e=this.engine.params[6]||1,t=this.engine.params[7]||1,n=this.engine.params[1];return Le(n,e/t,Ye,this.cameraRig)}setEdges(e){let t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e,this.edgeCount=e.length;let n=Ue(this.graph),r=Math.max(n.byteLength,8),i=!1;(!this.edgeBuffer||r>this.edgeCapacityBytes)&&(this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(r*2,64),this.edgeBuffer=t.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),i=!0),t.queue.writeBuffer(this.edgeBuffer,0,n.buffer),this.engine.params[3]=e.length,i&&this.createPipeline(t)}uploadLiveRetention(e){let t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;let n=Math.min(e.length,this.nodeCount);n<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,n*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){let t=e.createShaderModule({label:`observatory-simulate`,code:Ge});this.simPipeline=e.createComputePipeline({label:`observatory-recall-sim`,layout:`auto`,compute:{module:t,entryPoint:`recall_sim`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&n.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&n.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:`observatory-recall-sim-bind`,layout:this.simPipeline.getBindGroupLayout(0),entries:n})}let t=e.createShaderModule({label:`observatory-render-nodes`,code:We});if(this.pipeline=e.createRenderPipeline({label:`observatory-nodes`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bindGroup=e.createBindGroup({label:`observatory-nodes-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){let t=e.createShaderModule({label:`observatory-render-path`,code:Ke});this.pathPipeline=e.createRenderPipeline({label:`observatory-path`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.pathBindGroup=e.createBindGroup({label:`observatory-path-bind`,layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}if(this.edgeBuffer&&this.pathBuffer&&this.nodeBuffer){let t=e.createShaderModule({label:`observatory-render-axons`,code:U});this.axonPipeline=e.createRenderPipeline({label:`observatory-axons`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`line-list`}}),this.axonBindGroup=e.createBindGroup({label:`observatory-axons-bind`,layout:this.axonPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.edgeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}},{binding:4,resource:{buffer:this.nodeBuffer}}]})}}compute(e){let t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;let n=this.currentOrbit();if(this.cameraData.set(n.viewProj,0),this.cameraData[16]=n.right[0],this.cameraData[17]=n.right[1],this.cameraData[18]=n.right[2],this.cameraData[19]=0,this.cameraData[20]=n.up[0],this.cameraData[21]=n.up[1],this.cameraData[22]=n.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){let t=e.beginComputePass({label:`observatory-recall-sim`});t.setPipeline(this.simPipeline),t.setBindGroup(0,this.simBindGroup),t.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),t.end()}}render(e){this.axonPipeline&&this.axonBindGroup&&this.edgeCount>0&&(e.setPipeline(this.axonPipeline),e.setBindGroup(0,this.axonBindGroup),e.draw(2,this.edgeCount)),this.pipeline&&this.bindGroup&&this.nodeCount!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){if(this.disposed)return null;let n=this.engine.gpuDevice;if(!n||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;this.pickReadback||=this.readNodePositions(n).finally(()=>{this.pickReadback=null});let r=await this.pickReadback;if(!r||this.disposed||!this.graph)return null;let i=this.currentOrbit().viewProj,a=1/Math.tan(50*Math.PI/360),o=-1,s=1/0;for(let n=0;n<this.nodeCount;n++){let c=n*16+A.posRadius,l=r[c],u=r[c+1],d=r[c+2],f=r[c+3],p=i[3]*l+i[7]*u+i[11]*d+i[15];if(p<=0)continue;let m=(i[0]*l+i[4]*u+i[8]*d+i[12])/p,h=(i[1]*l+i[5]*u+i[9]*d+i[13])/p,g=Math.max(f*a/p,.012),_=Math.hypot(m-e,h-t)/g;_<1.6*(n===this.hoveredIndex?.85:1)&&_<s&&(s=_,o=n)}return o<0?null:{index:o,id:this.graph.nodes[o].id}}async readNodePositions(e){if(!this.nodeBuffer)return null;let t=this.nodeCount*16*4,n=e.createBuffer({label:`observatory-pick-staging`,size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let r=e.createCommandEncoder({label:`observatory-pick-copy`});r.copyBufferToBuffer(this.nodeBuffer,0,n,0,t),e.queue.submit([r.finish()]),await n.mapAsync(GPUMapMode.READ);let i=new Float32Array(n.getMappedRange().slice(0));return n.unmap(),i}catch{return null}finally{n.destroy()}}dispose(){this.disposed=!0,this.nodeBuffer?.destroy(),this.edgeBuffer?.destroy(),this.cameraBuffer?.destroy(),this.pathBuffer?.destroy(),this.liveRetentionBuffer?.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null,this.axonPipeline=null,this.axonBindGroup=null,this.edgeCapacityBytes=0,this.edgeCount=0}},Qe=16,$e=4,et=110,tt=.7,nt=.2,rt=360,it=18;function at(e){if(e.edges.length>0){let t=e.centerIndex,n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t);if(n.length>0){let r=-1,i=-1;for(let a of n){let n=a.sourceIndex===t?a.targetIndex:a.sourceIndex,o=e.nodes[n];o&&o.retention>i&&(i=o.retention,r=n)}if(r>=0)return r}}for(let t=0;t<e.nodes.length;t++)if(t!==e.centerIndex)return t;return e.centerIndex}function ot(e,t,n=8192){let r=at(e),i=e.nodes[r].id,a=G(e,r),o=new fe({seed:t+`:birth:`+i}).state.rng,s=new Float32Array(n*Qe),c=Math.floor(n*tt),l=Math.floor(n*nt),u=n-c-l;for(let e=0;e<c;e++){let t=e*Qe,[n,r,i]=ce(e,c,et+o()*70,o);s[t+0]=a[0]+n,s[t+1]=a[1]+r,s[t+2]=a[2]+i,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}let d=e.edges.filter(e=>e.sourceIndex===r||e.targetIndex===r);for(let t=0;t<l;t++){let n=(c+t)*Qe;if(d.length===0)continue;let i=d[t%d.length],u=G(e,i.sourceIndex===r?i.targetIndex:i.sourceIndex),f=u[0]-a[0],p=u[1]-a[1],m=u[2]-a[2],h=Math.sqrt(f*f+p*p+m*m)||1,g=t/Math.max(1,l)*2+.5,_=o()*30,v=-p*_/(h||1),y=f*_/(h||1);s[n+0]=a[0]+f/h*g*80+v,s[n+1]=a[1]+p/h*g*80+y,s[n+2]=a[2]+m/h*g*80+0,s[n+3]=o(),s[n+4]=a[0],s[n+5]=a[1],s[n+6]=a[2],s[n+7]=1+o()*1.8,s[n+8]=.91,s[n+9]=1,s[n+10]=.72,s[n+11]=o(),s[n+12]=0,s[n+13]=0,s[n+14]=0,s[n+15]=0}for(let e=0;e<u;e++){let t=(c+l+e)*Qe,n=o()*Math.PI*2,r=o()*120;s[t+0]=a[0]+Math.cos(n)*r,s[t+1]=a[1]+Math.sin(n)*r,s[t+2]=a[2]+180+o()*40,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}return{targetIndex:r,targetNodeId:i,particles:s,edgeSteps:st(e,r),timeline:ct()}}function G(e,t){let n=e.nodes[t],r=e.nodes.length;if(n.isCenter&&e.centerIndex===t)return[0,0,0];let i=Math.PI*(3-Math.sqrt(5)),a=1-t/(r-1||1)*2,o=Math.sqrt(1-a*a),s=i*t,c=(t*7+3)%100/100*.1*120-6,l=(t*13+7)%100/100*.1*120-6,u=(t*17+11)%100/100*.1*120-6;return[Math.cos(s)*o*120+c,a*120+l,Math.sin(s)*o*120+u]}function st(e,t){let n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t),r=n.length;if(r===0)return new Uint32Array;let i=new Uint32Array(r*$e);for(let e=0;e<r;e++){let r=n[e],a=r.sourceIndex===t?r.targetIndex:r.sourceIndex,o=rt+e*it;i[e*$e+0]=t,i[e*$e+1]=a,i[e*$e+2]=o,i[e*$e+3]=0}return i}function ct(){return[{label:`latent trace condensing`,startFrame:60,endFrame:239},{label:`engram coalescence`,startFrame:240,endFrame:329},{label:`memory ignition`,startFrame:330,endFrame:359},{label:`associations engrave`,startFrame:360,endFrame:509},{label:`stabilization`,startFrame:510,endFrame:659}]}var lt=`
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
`,ut=16,dt=6,ft=330,pt=359,K=360,mt=class{engine;nodeRenderer;active;computePipeline=null;computeBindGroup=null;particleBuffer=null;particleCount=0;renderPipeline=null;renderBindGroup=null;haloPipeline=null;haloBindGroup=null;haloIndexBuffer=null;engravePipeline=null;engraveBindGroup=null;engraveBuffer=null;engraveStepCount=0;timeline=[];birthPlan=null;get engraveSteps(){return this.birthPlan?.edgeSteps??new Uint32Array}constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}upload(e){let t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;let n=this.nodeRenderer.graph;if(!n)return;this.birthPlan=ot(n,e),this.timeline=this.birthPlan.timeline;let r=this.birthPlan.particles.length/ut;this.particleCount=r,this.particleBuffer?.destroy(),this.particleBuffer=t.createBuffer({label:`observatory-birth-particles`,size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),this.engraveBuffer?.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:`observatory-birth-engrave`,size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){let t=e.createShaderModule({label:`observatory-birth-compute`,code:lt});this.computePipeline=e.createComputePipeline({label:`observatory-birth-compute-pipeline`,layout:`auto`,compute:{module:t,entryPoint:`birth_compute`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:`observatory-birth-compute-bind`,layout:this.computePipeline.getBindGroupLayout(0),entries:n})}createRenderPipeline(e){let t=e.createShaderModule({label:`observatory-birth-render`,code:`
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
`});this.haloPipeline=e.createRenderPipeline({label:`observatory-birth-halo`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}});let n=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:`observatory-birth-halo-bind`,layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;let t=e.createShaderModule({label:`observatory-birth-engrave`,code:Ke});this.engravePipeline=e.createRenderPipeline({label:`observatory-birth-engrave-pipeline`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.engraveBindGroup=e.createBindGroup({label:`observatory-birth-engrave-bind`,layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){let n=this.engine.params[9];if(this.active=n===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;let r=e.beginComputePass({label:`observatory-birth-compute`});r.setPipeline(this.computePipeline),r.setBindGroup(0,this.computeBindGroup),r.dispatchWorkgroups(Math.ceil(this.particleCount/64)),r.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(dt,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=ft&&t<=pt&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=K&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){this.particleBuffer?.destroy(),this.particleBuffer=null,this.computePipeline?.destroy?.(),this.computePipeline=null,this.computeBindGroup=null,this.renderPipeline?.destroy?.(),this.renderPipeline=null,this.renderBindGroup=null,this.haloPipeline?.destroy?.(),this.haloPipeline=null,this.haloBindGroup=null,this.haloIndexBuffer?.destroy(),this.haloIndexBuffer=null,this.engravePipeline?.destroy?.(),this.engravePipeline=null,this.engraveBindGroup=null,this.engraveBuffer?.destroy(),this.engraveBuffer=null}};function q(e){return`
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
`}var ht=2,gt=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;waveBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.waveBuffer?.destroy(),this.waveBuffer=e.createBuffer({label:`observatory-rescue-wave`,size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);let t=e.createShaderModule({label:`observatory-rescue-choreo`,code:q(this.plan.consts)});this.pipeline=e.createComputePipeline({label:`observatory-rescue-choreo`,layout:`auto`,compute:{module:t,entryPoint:`rescue_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-rescue-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==ht||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-rescue-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.waveBuffer?.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}},_t=65535,vt={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function yt(e,t){return He(e,new fe({seed:t}).state.rng).data}function bt(e){let t=new Uint32Array(e.nodes.length);for(let n of e.edges)t[n.sourceIndex]++,t[n.targetIndex]++;return t}function xt(e,t){let n=e.nodes.length;if(n===0)return-1;let r=bt(e),i=n=>{let i=e.nodes[n],a=new Set(i.tags.map(e=>e.toLowerCase())),o=0;(a.has(`failure`)||a.has(`guardrail`))&&(o+=3),(a.has(`confusion`)||a.has(`weak-spot`))&&(o+=2),o+=Math.min(r[n],8)/8;let s=t[n*16+0],c=t[n*16+1],l=t[n*16+2];return Math.sqrt(s*s+c*c+l*l)>=54&&(o+=.5),o},a=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r[t]>=2,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex,()=>!0];for(let e of a){let t=-1,r=-1/0;for(let a=0;a<n;a++){if(!e(a))continue;let n=i(a);n>r&&(r=n,t=a)}if(t>=0)return t}return-1}function St(e,t){let n=e.nodes.length,r=new Uint16Array(n).fill(_t),i=new Int32Array(n).fill(-1);if(t<0||t>=n)return{depths:r,parents:i};let a=Array.from({length:n},()=>[]);for(let t of e.edges){let e=vt[t.type]??5;a[t.sourceIndex].push({nbr:t.targetIndex,rank:e}),a[t.targetIndex].push({nbr:t.sourceIndex,rank:e})}for(let e of a)e.sort((e,t)=>e.rank-t.rank||e.nbr-t.nbr);r[t]=0;let o=[t];for(let e=0;e<o.length;e++){let t=o[e];for(let{nbr:e}of a[t])r[e]===65535&&(r[e]=r[t]+1,o.push(e))}for(let e=0;e<n;e++)if(r[e]!==65535&&r[e]!==0){for(let{nbr:t}of a[e])if(r[t]===r[e]-1){i[e]=t;break}}return{depths:r,parents:i}}function Ct(e,t,n,r){let i=new Map;for(let t of e.nodes)i.set(t.id,t.createdAt);for(let e of[3,2,1]){let a=[];for(let i=0;i<t.nodes.length;i++){if(i===t.centerIndex||i===r)continue;let o=n[i];o===65535||o<e||a.push(i)}if(a.length===0)continue;let o=a.filter(e=>t.nodes[e].retention<=.45);o.length===0&&(o=a);let s=new Map,c=1/0,l=-1/0;for(let e of o){let n=i.get(t.nodes[e].id),r=n?Date.parse(n):NaN;Number.isFinite(r)&&(s.set(e,r),r<c&&(c=r),r>l&&(l=r))}let u=e=>{let t=s.get(e);return t===void 0?0:l===c?1:(l-t)/(l-c)},d=e=>2*(1-t.nodes[e].retention)+.5*Math.min(n[e],6)/6+.5*u(e);return o.sort((e,t)=>{let r=d(e),i=d(t);return i===r?n[t]===n[e]?e-t:n[t]-n[e]:i-r}),{index:o[0],depth:n[o[0]]}}return{index:-1,depth:0}}function wt(e,t,n,r,i){let a=e[n*16+0],o=e[n*16+1],s=e[n*16+2],c=[];for(let l=0;l<t;l++){if(l===n||l===r||l===i)continue;let t=e[l*16+0]-a,u=e[l*16+1]-o,d=e[l*16+2]-s;c.push({i:l,d2:t*t+u*u+d*d})}return c.sort((e,t)=>e.d2-t.d2||e.i-t.i),c.slice(0,4).map(e=>e.i)}function J(e){return Math.min(84,Math.max(14,Math.floor(252/Math.max(1,e))))}function Tt(e,t){return Math.min(260+t*e,514)}function Et(e){return 138+28*e}function Y(e){return e.length>64?e.slice(0,64)+`…`:e}var X=4;function Z(e){let t=new Uint32Array(e);return t.fill(_t),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(e).fill(_t),causeDepth:0,hopSlot:J(3),waveData:t,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`candidate cause found`,causeLabel:``,failureLabel:``,causeDate:``,hops:0,k:0,receipt:``},consts:{hopSlot:J(3),causeDepth:3}}}function Dt(e,t,n,r){if(r)return Ot(e,t,r);let i=t.nodes.length;if(i===0)return Z(0);let a=yt(t,n),o=xt(t,a);if(o<0)return Z(i);let{depths:s,parents:c}=St(t,o),l=Ct(e,t,s,o);if(l.index<0){let e=Z(i);return e.failureIndex=o,e.hopDepths=s,e}let u=l.index,d=Math.max(1,l.depth),f=J(d),p=e=>Tt(e,f),m=wt(a,i,o,u,t.centerIndex),h=m.length,g=new Uint32Array(i);for(let e=0;e<i;e++){let t=s[e]&65535;e===o&&(t|=65536),e===u&&(t|=1<<17),g[e]=t}m.forEach((e,t)=>{g[e]|=1<<18|t<<19});let _=[];m.forEach((e,t)=>{_.push({src:o,dst:e,bf:Et(t),kind:P.probe,beatKind:`probe`})});let v=[];{let e=u;for(;e!==o&&e>=0&&c[e]>=0;)v.push(e),e=c[e]}let y=new Set(v),b=[];for(let e=0;e<i;e++){if(e===o||y.has(e))continue;let t=s[e];t===65535||t<1||t>d||c[e]<0||b.push(e)}b.sort((e,t)=>s[e]-s[t]||e-t);let x=[...v.slice().reverse(),...b].slice(0,48);x.sort((e,t)=>s[e]-s[t]||e-t);for(let e of x)_.push({src:c[e],dst:e,bf:p(s[e]),kind:P.backwardCause,beatKind:`wave`});_.push({src:u,dst:o,bf:560,kind:P.backwardCause,beatKind:`arc`});let S=new Uint32Array(Math.max(1,_.length)*X),C=[];_.forEach((e,n)=>{S[n*X+0]=e.src,S[n*X+1]=e.dst,S[n*X+2]=e.bf,S[n*X+3]=e.kind,C.push({sourceIndex:e.src,targetIndex:e.dst,beatFrame:e.bf,kind:e.kind,beatKind:e.beatKind,nodeId:t.nodes[e.dst].id,label:Y(t.nodes[e.dst].label)})});let w=Y(t.nodes[o].label),T=Y(t.nodes[u].label),E=[],D=(e,t,n,r)=>{E.push({sourceIndex:o,targetIndex:o,beatFrame:e,kind:t,beatKind:`rescue`,nodeId:r,label:n})};D(90,1,`failure: ${w}`,t.nodes[o].id),m.forEach((e,n)=>{D(Et(n),0,`lookalike ✗ · ${Y(t.nodes[e].label)}`,t.nodes[e].id)}),D(p(1),1,`reaching backward through time`,`rescue-wave-start`),d>=2&&p(d)!==p(1)&&D(p(d),1,`scrubbing past · ${d} hops`,`rescue-wave-deep`),D(560,1,`causal arc · ${T}`,t.nodes[u].id),D(600,1,`candidate cause found`,`rescue-verdict`);let ee=e.nodes.find(e=>e.id===t.nodes[u].id)?.createdAt??``,O=ee?ee.slice(0,10):``;return{viable:!0,failureIndex:o,causeIndex:u,lookalikeIndices:m,hopDepths:s,causeDepth:d,hopSlot:f,waveData:g,pathData:S,pathMetas:C,spineBeats:E,verdict:{headline:`candidate cause found`,causeLabel:T,failureLabel:w,causeDate:O,hops:d,k:h,receipt:`${d} hops back · ${O} · heuristic, no receipt · vector search: 0 for ${h}`},consts:{hopSlot:f,causeDepth:d}}}function Ot(e,t,n){let r=t.nodes.length,i=t.indexById.get(n.failureId)??-1,a=n.pathIds??[];if(i<0||a.length<2||a[a.length-1]!==n.failureId||new Set(a).size!==a.length)return Z(r);let o=a.map(e=>t.indexById.get(e));if(o.some(e=>e===void 0))return Z(r);let s=o,c=s[0];if(c===i)return Z(r);let l=n.candidates.find(e=>e.memoryId===a[0]);if(!l)return Z(r);let u=new Uint16Array(r);u.fill(_t),u[i]=0,s.forEach((e,t)=>{u[e]=s.length-1-t});let d=new Uint32Array(r);d[i]=65536,s.slice(0,-1).forEach(e=>{d[e]=u[e]}),d[c]|=1<<17;let f=s.length-1,p=J(f),m=Y(t.nodes[c].label),h=Y(t.nodes[i].label),g=e.nodes.find(e=>e.id===l.memoryId)?.createdAt?.slice(0,10)??``,_=new Uint32Array((s.length-1)*X),v=s.slice(0,-1).map((e,n)=>{let r=s[n+1],i=260+n*p;return _[n*X]=e,_[n*X+1]=r,_[n*X+2]=i,_[n*X+3]=P.backwardCause,{sourceIndex:e,targetIndex:r,beatFrame:i,kind:P.backwardCause,beatKind:`receipt-path`,nodeId:a[n+1],label:`recorded path · ${Y(t.nodes[r].label)}`}}),y=l.sharedEntities.length?l.sharedEntities.join(`, `):`recorded entity`,b=l.similarityRank===null?`rank unavailable`:`embedding rank #${l.similarityRank}`;return{viable:!0,failureIndex:i,causeIndex:c,lookalikeIndices:[],hopDepths:u,causeDepth:f,hopSlot:p,waveData:d,pathData:_,pathMetas:v,spineBeats:[{sourceIndex:i,targetIndex:i,beatFrame:90,kind:1,beatKind:`receipt-failure`,nodeId:n.failureId,label:`recorded failure · ${h}`},{sourceIndex:i,targetIndex:i,beatFrame:260,kind:1,beatKind:`receipt-join`,nodeId:`receipt-join`,label:`shared entity · ${y}`},{sourceIndex:c,targetIndex:i,beatFrame:260+(f-1)*p,kind:P.backwardCause,beatKind:`receipt-candidate`,nodeId:l.memoryId,label:`candidate · ${m}`},{sourceIndex:c,targetIndex:c,beatFrame:600,kind:1,beatKind:`receipt-verdict`,nodeId:`receipt-verdict`,label:`candidate cause found`}],verdict:{headline:`candidate cause found`,causeLabel:m,failureLabel:h,causeDate:g,hops:f,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${y} · ${b}`},consts:{hopSlot:p,causeDepth:f}}}var kt=`
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
`,At=3,jt=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;horizonBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.horizonBuffer?.destroy(),this.horizonBuffer=e.createBuffer({label:`observatory-forgetting-horizon`,size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);let t=e.createShaderModule({label:`observatory-forgetting-choreo`,code:kt});this.pipeline=e.createComputePipeline({label:`observatory-forgetting-choreo`,layout:`auto`,compute:{module:t,entryPoint:`forgetting_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-forgetting-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==At||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-forgetting-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.horizonBuffer?.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}};function Q(e){let t=[];for(let n=0;n<e.nodes.length;n++)n!==e.centerIndex&&t.push(n);t.sort((t,n)=>e.nodes[t].retention-e.nodes[n].retention||t-n);let n=t.length;if(n===0)return[];let r=Math.min(n,Math.max(Math.min(3,n),Math.round(.25*n)));return t.slice(0,r)}function Mt(e,t){let n=new Uint32Array(e.nodes.length);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>2*e.nodes[t].retention+Math.min(n[t],8)/8;return t.slice().sort((e,t)=>r(t)-r(e)||e-t).slice(0,Math.min(3,t.length))}function Nt(e){return 318+60*e}var Pt=4;function Ft(e){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function It(e){let t=e.nodes.length,n=Q(e);if(t<2||n.length<1)return Ft(t);let r=Mt(e,n),i=n.length,a=new Uint32Array(t);n.forEach((e,t)=>{let n=Math.round(255*t/Math.max(1,i-1));a[e]=n&255|256}),r.forEach((e,t)=>{a[e]|=512|t<<10});let o=new Uint32Array(Math.max(1,r.length)*Pt),s=[];r.forEach((t,n)=>{let r=Nt(n);o[n*Pt+0]=e.centerIndex,o[n*Pt+1]=t,o[n*Pt+2]=r,o[n*Pt+3]=P.recall,s.push({sourceIndex:e.centerIndex,targetIndex:t,beatFrame:r,kind:P.recall,beatKind:`recall`,nodeId:e.nodes[t].id,label:Y(e.nodes[t].label)})});let c=[],l=(t,n,r,i)=>{c.push({sourceIndex:e.centerIndex,targetIndex:e.centerIndex,beatFrame:t,kind:n,beatKind:`horizon`,nodeId:i,label:r})},u=new Set(r),d=n.filter(e=>!u.has(e)).slice(0,3);return d.forEach((t,n)=>{let r=Math.round(e.nodes[t].retention*100);l(132+60*n,1,`fading: ${Y(e.nodes[t].label)} · retention ${r}%`,e.nodes[t].id)}),r.forEach((t,n)=>{l(Nt(n),0,`recalled: ${Y(e.nodes[t].label)}`,e.nodes[t].id)}),d.length>0&&l(540,1,`the unrecalled sink · nothing is deleted`,`horizon-sink`),l(660,0,`every memory still retrievable`,`horizon-retrievable`),{viable:!0,driftingIndices:n,rescuedIndices:r,horizonData:a,pathData:o,pathMetas:s,spineBeats:c}}var Lt=`
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
`,Rt=4,zt=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;fireBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.fireBuffer?.destroy(),this.fireBuffer=e.createBuffer({label:`observatory-firewall-fire`,size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);let t=e.createShaderModule({label:`observatory-firewall-choreo`,code:Lt});this.pipeline=e.createComputePipeline({label:`observatory-firewall-choreo`,layout:`auto`,compute:{module:t,entryPoint:`firewall_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-firewall-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){let t=this.engine.params[9]===Rt,n=this.engine.params[12]===1;if(!t&&!n||!this.pipeline||!this.bindGroup)return;let r=this.nodeRenderer.nodeCountValue;if(r===0)return;let i=e.beginComputePass({label:`observatory-firewall-choreo`});i.setPipeline(this.pipeline),i.setBindGroup(0,this.bindGroup),i.dispatchWorkgroups(Math.ceil(r/64)),i.end()}dispose(){this.fireBuffer?.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}},Bt=[`failure`,`guardrail`,`confusion`];Math.PI*2;function Vt(e){let t=e.nodes.length;if(t===0)return-1;let n=new Uint32Array(t);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>e.nodes[t].tags.some(e=>Bt.includes(e.toLowerCase())),i=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r(t),t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&n[t]<=1,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex];for(let n of i){let r=-1;for(let i=0;i<t;i++)n(i)&&(r<0||e.nodes[i].retention<e.nodes[r].retention)&&(r=i);if(r>=0)return r}return-1}function Ht(e,t,n){let r=e[n*16+0],i=e[n*16+1],a=e[n*16+2],o=Array(t),s=0;for(let n=0;n<t;n++){let t=e[n*16+0]-r,c=e[n*16+1]-i,l=e[n*16+2]-a,u=Math.sqrt(t*t+c*c+l*l);o[n]=u,u>s&&(s=u)}s<1e-6&&(s=1);let c=Array(t);for(let e=0;e<t;e++)c[e]=Math.min(255,Math.max(0,Math.round(144*o[e]/s)));return c[n]=0,c}function Ut(e,t){let n=new Set;for(let r of e.edges)r.sourceIndex===t&&r.targetIndex!==t&&n.add(r.targetIndex),r.targetIndex===t&&r.sourceIndex!==t&&n.add(r.sourceIndex);return Array.from(n).sort((e,t)=>e-t).slice(0,6)}function Wt(e){return 345+21*e}var Gt=4;function Kt(e){return qt(e)}function qt(e){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`threat quarantined`,intruderLabel:``,receipt:`memory held in review · Memory PR opened`}}}function Jt(e,t){return Xt(e,t,Vt(e))}function Yt(e,t,n){return n<0||n>=e.nodes.length?qt(e.nodes.length):Xt(e,t,n)}function Xt(e,t,n){let r=e.nodes.length;if(r===0||n<0)return qt(r);let i=Ht(yt(e,t),r,n),a=Ut(e,n),o=new Uint32Array(r);for(let e=0;e<r;e++)o[e]=i[e]&255;o[n]=256,a.forEach((e,t)=>{o[e]|=512|t<<10});let s=new Uint32Array(Math.max(1,a.length)*Gt),c=[];a.forEach((t,r)=>{let i=Wt(r);s[r*Gt+0]=n,s[r*Gt+1]=t,s[r*Gt+2]=i,s[r*Gt+3]=P.probe,c.push({sourceIndex:n,targetIndex:t,beatFrame:i,kind:P.probe,beatKind:`sever`,nodeId:e.nodes[t].id,label:Y(e.nodes[t].label)})});let l=Y(e.nodes[n].label),u=[],d=(e,t,r)=>{u.push({sourceIndex:n,targetIndex:n,beatFrame:e,kind:1,beatKind:`firewall`,nodeId:r,label:t})};return d(90,`intrusion · ${l}`,e.nodes[n].id),d(150,`immune response · shockwave`,`firewall-shock`),d(330,`membrane forming`,`firewall-membrane`),a.forEach((t,n)=>{d(Wt(n),`edge severed ✗ · ${Y(e.nodes[t].label)}`,e.nodes[t].id)}),d(480,`threat quarantined`,`firewall-verdict`),{viable:!0,intruderIndex:n,severedNeighborIndices:a,shockDelays:i,fireData:o,pathData:s,pathMetas:c,spineBeats:u,verdict:{headline:`threat quarantined`,intruderLabel:l,receipt:`memory held in review · Memory PR opened`}}}var Zt=.1542;function Qt(e=Zt){return .9**(-1/e)-1}function $t(e,t,n=Zt){if(!(e>0))return 0;if(!(t>0))return 1;let r=(1+Qt(n)*t/e)**+-n;return r<0?0:r>1?1:r}var en=864e5;function tn(e,t,n=0){if(!e)return n>0?n:0;let r=Date.parse(e);if(!Number.isFinite(r))return n>0?n:0;let i=(t-r)/en;return Math.max(0,i)+Math.max(0,n)}function nn(e,t,n,r,i=Zt){if(n){let e=Date.parse(n);if(Number.isFinite(e)&&r<e)return 0}if(e===void 0||!Number.isFinite(e)||!t)return 1;let a=Date.parse(t);return Number.isFinite(a)?Math.max(.001,$t(e,(r-a)/en,i)):1}function rn(e,t,n,r=0,i=Zt){return e===void 0||!Number.isFinite(e)?1:$t(e,tn(t,n,r),i)}var an={[N.firewall]:620,[N.dreamStorm]:360,[N.causalRecall]:260,[N.birth]:180},on=class{engine;renderer;graph;response;seed;projectionDays;chronoOffsetDays;onApply;onFirewall;firewall=null;liveEdges=[];liveEdgeKeys=new Set;edgesDirty=!1;indexById;active=null;dreamOpen=!1;retention;hasLiveDecay=!1;eventsSeen=0;lastDecayFrame=-1e3;constructor(e){this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;let t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let n=0;n<t;n++){let t=e.graph.nodes[n];this.retention[n]=t.retention,t.stability!==void 0&&t.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(let e of this.liveEdges)this.liveEdgeKeys.add(sn(e.sourceIndex,e.targetIndex));this.lastAppliedMs=0;let n=this.engine.params;n[M.liveKind]=N.none,n[M.liveFrame]=0,n[M.liveEnergy]=0,n[M.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}lastAppliedMs=0;seeded=!1;seedWatermark(e){let t=0;for(let n of e){let e=cn(n);e>t&&(t=e)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,n){if(this.active!==null)return!1;let r=this.indexById.get(e);if(r===void 0||(this.retention[r]??0)<5e-4)return!1;let i=t.filter(t=>t!==e&&this.indexById.has(t));return this.arm({kind:N.causalRecall,startFrame:n,targetId:e,relatedIds:i,pairs:[],scalar:i.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let n=e.length-1;n>=0;n--){let r=e[n],i=cn(r);i>this.lastAppliedMs&&(this.decodeAndArm(r,this.engine.totalFrames),i>t&&(t=i))}this.lastAppliedMs=t}decodeAndArm(e,t){let n=e.data??{};switch(e.type){case`MemorySuppressed`:{let e=ln(n.id);if(!e||!this.indexById.has(e))return;this.arm({kind:N.firewall,startFrame:t,targetId:e,relatedIds:this.neighborsOf(e),pairs:[],scalar:un(n.estimated_cascade)});break}case`DeepReferenceCompleted`:{let e=fn(n.contradiction_pairs).filter(([e,t])=>this.indexById.has(e)&&this.indexById.has(t));if(e.length>0){let n=e[0][0];this.arm({kind:N.firewall,startFrame:t,targetId:n,relatedIds:e.flatMap(e=>e).filter(e=>e!==n),pairs:e,scalar:e.length});return}let r=ln(n.primary_id),i=dn(n.supporting_ids).filter(e=>this.indexById.has(e));r&&this.indexById.has(r)&&this.arm({kind:N.causalRecall,startFrame:t,targetId:r,relatedIds:i,pairs:[],scalar:un(n.confidence)});break}case`BackfillFired`:case`CausalReceipt`:{let e=dn(n.path_ids??n.causal_path),r=ln(n.failure_id??n.target_id??n.effect_id)||e.at(-1)||e[0];r&&this.indexById.has(r)&&this.arm({kind:N.causalRecall,startFrame:t,targetId:r,relatedIds:e.filter(e=>e!==r),exactPath:e,pairs:[],scalar:e.length});break}case`DreamStarted`:this.dreamOpen=!0,this.arm({kind:N.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:un(n.memory_count)});break;case`DreamCompleted`:{this.dreamOpen=!1;let e=un(n.connections_found);this.active&&this.active.kind===N.dreamStorm?this.active.scalar=Math.max(this.active.scalar,e):this.arm({kind:N.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:e});break}case`ConnectionDiscovered`:{let e=this.indexById.get(ln(n.source_id)),t=this.indexById.get(ln(n.target_id));if(e===void 0||t===void 0||e===t)break;let r=sn(e,t);if(this.liveEdgeKeys.has(r))break;this.liveEdgeKeys.add(r),this.liveEdges.push({sourceIndex:e,targetIndex:t,weight:un(n.weight)||.5,type:ln(n.connection_type)||`semantic`}),this.edgesDirty=!0,this.dreamOpen&&this.active?.kind===N.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){if(this.active=e,this.eventsSeen++,e.kind===N.firewall){let t=this.indexById.get(e.targetId);if(t===void 0)return;let n=Yt(this.graph,this.seed,t);if(!n.viable)return;this.firewall||=new zt({engine:this.engine,nodeRenderer:this.renderer,plan:Kt(this.graph.nodes.length)}),this.firewall.rearm(n),this.onFirewall?.({intruderLabel:n.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===N.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){let t=e.exactPath;if(t.some(e=>!this.indexById.has(e)))return;let n=new Uint32Array(Math.max(1,t.length-1)*4),r=[];for(let i=0;i<t.length-1;i++){let a=this.indexById.get(t[i]),o=this.indexById.get(t[i+1]),s=e.startFrame+24+i*42;n[i*4]=a,n[i*4+1]=o,n[i*4+2]=s,n[i*4+3]=P.backwardCause,r.push({sourceIndex:a,targetIndex:o,beatFrame:s,kind:P.backwardCause,beatKind:`receipt-path`,nodeId:t[i+1],label:`receipt-backed candidate path`})}this.renderer.setPathSteps(n,r);return}let t=qe(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});t.steps.length>0&&this.renderer.setPathSteps(t.data,t.steps)}}neighborsOf(e){let t=this.indexById.get(e);if(t===void 0)return[];let n=[];for(let e of this.graph.edges)if(e.sourceIndex===t?n.push(this.graph.nodes[e.targetIndex].id):e.targetIndex===t&&n.push(this.graph.nodes[e.sourceIndex].id),n.length>=12)break;return n}drain(e){let t=this.engine.params;this.edgesDirty&&=(this.renderer.setEdges(this.liveEdges),!1);let n=this.projectionDays(),r=this.chronoOffsetDays();if(t[M.projectionDays]=Math.max(0,n),(this.hasLiveDecay||r!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||n!==this.lastProj||r!==this.lastChrono)&&(this.recomputeDecay(n,r),this.lastDecayFrame=e,this.lastProj=n,this.lastChrono=r),this.active){let n=an[this.active.kind]??300,r=e-this.active.startFrame;r>n+140?(this.active=null,t[M.liveKind]=N.none,t[M.liveEnergy]=0):(t[M.liveKind]=this.active.kind,t[M.liveFrame]=Math.max(0,r),t[M.liveEnergy]=this.energyEnvelope(this.active,r,!1))}else t[M.liveKind]=N.none,t[M.liveEnergy]=0;this.onApply?.({simFrame:e,activeKind:t[M.liveKind],eventsSeen:this.eventsSeen})}debugState(){let e=this.engine.params;return{activeKind:e[M.liveKind],liveEnergy:e[M.liveEnergy],liveFrame:e[M.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}lastProj=-1;lastChrono=0;energyEnvelope(e,t,n){if(t<0)return 0;let r=an[e.kind]??300;if(e.kind===N.dreamStorm){let n=Math.min(1,t/45),i=1-Math.max(0,(t-(r-90))/90),a=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,n*Math.min(1,i)*a)}let i=Math.min(1,t/24),a=1-Math.max(0,(t-r)/140);return Math.max(0,i*Math.min(1,a))}recomputeDecay(e,t=0){let n=this.engine.wallNowMs,r=this.graph.nodes;if(t!==0){let i=n+(t+Math.max(0,e))*en;for(let e=0;e<r.length;e++){let t=r[e];this.retention[e]=t.stability!==void 0||t.createdAt?nn(t.stability,t.lastAccessed,t.createdAt,i):Math.max(.001,t.retention)}}else for(let t=0;t<r.length;t++){let i=r[t];this.retention[t]=i.stability!==void 0&&i.lastAccessed?rn(i.stability,i.lastAccessed,n,e):Math.max(.001,i.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){let e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}};function sn(e,t){return e<t?`${e}-${t}`:`${t}-${e}`}function cn(e){let t=e.data?.timestamp;if(typeof t!=`string`)return 0;let n=Date.parse(t);return Number.isFinite(n)?n:0}function ln(e){return typeof e==`string`?e:``}function un(e){return typeof e==`number`&&Number.isFinite(e)?e:0}function dn(e){return Array.isArray(e)?e.filter(e=>typeof e==`string`):[]}function fn(e){if(!Array.isArray(e))return[];let t=[];for(let n of e)Array.isArray(n)&&n.length>=2&&typeof n[0]==`string`&&typeof n[1]==`string`&&t.push([n[0],n[1]]);return t}var pn=512,mn=4,hn=`
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
`;function gn(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function _n(e){if(!e)return null;let t=Date.parse(e);return Number.isFinite(t)?t:null}var vn=class{engine;resources=null;bindLayout=null;railPipeline=null;dwellPipeline=null;headPipeline=null;dwellCount=0;minMs=0;maxMs=0;state={scrub:1,days:0,density:0,active:0};constructor(e,t){this.engine=e,this.upload(t)}setTimeline(e,t=!1){let n=this.engine.wallNowMs,r=Math.max(1,this.maxMs-this.minMs);this.state.scrub=gn((n+e*864e5-this.minMs)/r),this.state.days=Number.isFinite(e)?e:0,this.state.active=+!!t,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){this.resources&&this.railPipeline&&this.dwellPipeline&&this.headPipeline&&(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){this.resources?.dwellBuffer.destroy(),this.resources?.stateBuffer.destroy(),this.resources=null}upload(e){let t=e.flatMap(e=>[_n(e.createdAt),_n(e.lastAccessed)]).filter(e=>e!==null),n=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):n-864e5,this.maxMs=Math.max(n+31536e6,this.minMs+864e5);let r=this.maxMs-this.minMs,i=[];for(let t of e){let e=_n(t.createdAt),n=_n(t.lastAccessed),r=gn(t.retention);e!==null&&i.push({at:e,kind:0,retention:r,suppressed:+!!t.suppressed}),n!==null&&n!==e&&i.push({at:n,kind:1,retention:r,suppressed:+!!t.suppressed})}i.sort((e,t)=>e.at-t.at);let a=Math.max(1,Math.ceil(i.length/pn)),o=i.filter((e,t)=>t%a===0).slice(0,pn);this.dwellCount=o.length,this.state={scrub:gn((n-this.minMs)/r),days:0,density:gn(o.length/96),active:0};let s=this.engine.gpuDevice;if(!s||!this.engine.paramsBuffer||(this.ensurePipelines(s),this.ensureResources(s),!this.resources))return;let c=new Float32Array(pn*mn);o.forEach((e,t)=>{c.set([gn((e.at-this.minMs)/r),e.kind,e.retention,e.suppressed],t*mn)}),s.queue.writeBuffer(this.resources.dwellBuffer,0,c),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`fossil-light-chrono-shuttle-wgsl`,code:hn});this.bindLayout=e.createBindGroupLayout({label:`fossil-light-chrono-shuttle-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let n=e.createPipelineLayout({label:`fossil-light-chrono-shuttle-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`src-alpha`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}},i=(i,a,o)=>e.createRenderPipeline({label:i,layout:n,vertex:{module:t,entryPoint:a},fragment:{module:t,entryPoint:o,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`}});this.railPipeline=i(`fossil-light-chrono-rail`,`vs_rail`,`fs_rail`),this.dwellPipeline=i(`fossil-light-chrono-dwells`,`vs_dwell`,`fs_dwell`),this.headPipeline=i(`fossil-light-chrono-head`,`vs_head`,`fs_head`)}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;let t=e.createBuffer({label:`fossil-light-chrono-dwell-events`,size:pn*mn*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:`fossil-light-chrono-state`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:n,bindGroup:e.createBindGroup({label:`fossil-light-chrono-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}}]})}}writeState(){let e=this.engine.gpuDevice;e&&this.resources&&e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active]))}},yn=64,$=32,bn=4,xn=256,Sn=`rgba8unorm`,Cn=96e3,wn=5,Tn=`
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

const MAX_EMITTERS = ${yn}u;

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
`;function En(e,t){return Number.isFinite(e)?e:t}var Dn=class{engine;renderer;sourceIndices;resources=null;projectionPipeline=null;seedPipeline=null;transportPipeline=null;compositePipeline=null;projectionLayout=null;seedLayout=null;transportLayout=null;compositeLayout=null;emitterCount;active=!1;dirty=!0;lastComputedFrame=-5;disposed=!1;disabledReason=null;exposure=.42;configBytes=new ArrayBuffer($);configUints=new Uint32Array(this.configBytes);configFloats=new Float32Array(this.configBytes);constructor(e,t,n){this.engine=e,this.renderer=t;let r=[...new Set([...n].filter(e=>Number.isFinite(e)&&e>=0))].sort((e,t)=>e-t).slice(0,yn);this.sourceIndices=new Uint32Array(r),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?`half-res-transport`:`disabled`}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,En(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;let n=this.engine.gpuDevice;if(!n||!this.engine.paramsBuffer)return;let r=this.renderer.getFossilLightSources();if(!r)return;let i=this.fieldDimensions();if(i===null)return;let a=t-this.lastComputedFrame;if(!(this.active||this.dirty||a<0||a>=wn))return;try{this.ensurePipelines(n),this.ensureResources(n,i.width,i.height,r)}catch{this.disable(`GPU light field unavailable on this adapter`);return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(n,0,this.resources.width,this.resources.height,0);let o=Math.ceil(this.resources.width/8),s=Math.ceil(this.resources.height/8),c=e.beginComputePass({label:`fossil-light-half-res-transport`});c.setPipeline(this.projectionPipeline),c.setBindGroup(3,this.resources.projectionBindGroup,[0]),c.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),c.setPipeline(this.seedPipeline),c.setBindGroup(0,this.resources.seedBindGroup,[0]),c.dispatchWorkgroups(o,s);for(let[e,t,r]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(n,e,this.resources.width,this.resources.height,t),c.setPipeline(this.transportPipeline),c.setBindGroup(1,r,[e*xn]),c.dispatchWorkgroups(o,s);c.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason===null&&this.resources&&this.compositePipeline&&this.emitterCount!==0&&(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*xn]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){let e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;let n=e*.5*(t*.5),r=.5*Math.min(1,Math.sqrt(Cn/Math.max(1,n)));return{width:Math.max(1,Math.floor(e*r)),height:Math.max(1,Math.floor(t*r))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;let t=e.createShaderModule({label:`fossil-light-radiance-cascade-wgsl`,code:Tn}),n=e.createBindGroupLayout({label:`fossil-light-empty-layout`,entries:[]});this.projectionLayout=e.createBindGroupLayout({label:`fossil-light-source-projection-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:$}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:`storage`}}]}),this.seedLayout=e.createBindGroupLayout({label:`fossil-light-seed-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:$}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:Sn}}]}),this.transportLayout=e.createBindGroupLayout({label:`fossil-light-transport-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:$}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:Sn}}]}),this.compositeLayout=e.createBindGroupLayout({label:`fossil-light-composite-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:$}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}}]}),this.seedPipeline=e.createComputePipeline({label:`fossil-light-seed`,layout:e.createPipelineLayout({label:`fossil-light-seed-pipeline-layout`,bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:`cs_seed`}}),this.projectionPipeline=e.createComputePipeline({label:`fossil-light-source-projection`,layout:e.createPipelineLayout({label:`fossil-light-source-projection-pipeline-layout`,bindGroupLayouts:[n,n,n,this.projectionLayout]}),compute:{module:t,entryPoint:`cs_project_sources`}}),this.transportPipeline=e.createComputePipeline({label:`fossil-light-transport`,layout:e.createPipelineLayout({label:`fossil-light-transport-pipeline-layout`,bindGroupLayouts:[n,this.transportLayout]}),compute:{module:t,entryPoint:`cs_transport`}}),this.compositePipeline=e.createRenderPipeline({label:`fossil-light-composite`,layout:e.createPipelineLayout({label:`fossil-light-composite-pipeline-layout`,bindGroupLayouts:[n,n,this.compositeLayout]}),vertex:{module:t,entryPoint:`vs_composite`},fragment:{module:t,entryPoint:`fs_composite`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`src-alpha`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]}})}ensureResources(e,t,n,r){if(this.resources?.width===t&&this.resources.height===n&&this.resources.nodeBuffer===r.nodeBuffer&&this.resources.cameraBuffer===r.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;let i=e.createBuffer({label:`fossil-light-projected-memory-emitters`,size:768*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),a=e.createBuffer({label:`fossil-light-source-indices`,size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(a,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);let o=e.createBuffer({label:`fossil-light-cascade-config`,size:xn*bn,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),s=r=>e.createTexture({label:r,size:[t,n],format:Sn,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),c=s(`fossil-light-field-a`),l=s(`fossil-light-field-b`),u=c.createView(),d=l.createView();this.resources={width:t,height:n,emitterBuffer:i,sourceIndexBuffer:a,configBuffer:o,fieldA:c,fieldB:l,seedBindGroup:e.createBindGroup({label:`fossil-light-seed-bind-group`,layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:$}},{binding:1,resource:{buffer:i}},{binding:2,resource:u}]}),propagateABindGroup:e.createBindGroup({label:`fossil-light-transport-a-to-b`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:$}},{binding:1,resource:u},{binding:2,resource:d}]}),propagateBBindGroup:e.createBindGroup({label:`fossil-light-transport-b-to-a`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:$}},{binding:1,resource:d},{binding:2,resource:u}]}),projectionBindGroup:e.createBindGroup({label:`fossil-light-source-projection-bind-group`,layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:$}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:r.nodeBuffer}},{binding:3,resource:{buffer:r.cameraBuffer}},{binding:4,resource:{buffer:i}}]}),compositeBindGroup:e.createBindGroup({label:`fossil-light-composite-bind-group`,layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:$}},{binding:1,resource:d}]}),nodeBuffer:r.nodeBuffer,cameraBuffer:r.cameraBuffer},this.dirty=!0}writeConfig(e,t,n,r,i){this.resources&&(this.configUints[0]=n,this.configUints[1]=r,this.configUints[2]=this.emitterCount,this.configUints[3]=i,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*xn,this.configBytes))}destroyResources(){this.resources?.emitterBuffer.destroy(),this.resources?.sourceIndexBuffer.destroy(),this.resources?.configBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}},On=C(`<div class="flex items-baseline gap-2 font-mono text-[11px]"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></div>`),kn=C(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),An=C(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),jn=C(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"> </button>`),Mn=C(`<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>`),Nn=C(`<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>`),Pn=C(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),Fn=C(`<button> </button>`),In=C(`<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>`),Ln=C(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>`),Rn=C(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>`),zn=C(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>`),Bn=C(`<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>`),Vn=C(`<div class="pointer-events-none fixed left-4 bottom-24 z-30 font-mono text-[10px] tracking-widest text-[#7ff3e6]/80"> </div>`),Hn=C(`<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div> <!> <!>`,1);function Un(t,c){re(c,!0);let x=()=>S(oe,`$eventFeed`,C),[C,ae]=te(),A=g(c,`seed`,3,`vestige-observatory-v1`),ce=g(c,`freezeFrame`,3,null),j=g(c,`capture`,3,!1),le=g(c,`showSwitcher`,3,!0),M=g(c,`embedded`,3,!1),N=g(c,`chrome`,3,`none`),P=g(c,`maxDpr`,3,2),fe=g(c,`focusIds`,19,()=>[]),me=g(c,`live`,3,!1),F=g(c,`graphOverride`,3,null),I=y(0),L=y(!1),R=y(0),ye=null,be=k(()=>Math.max(0,o(I))),xe=k(()=>Math.min(0,o(I))),Ce=k(()=>o(I)===0?`now`:o(I)>0?`+${Math.round(o(I))}d`:new Date(Date.now()+o(I)*864e5).toLocaleDateString(void 0,{month:`short`,day:`numeric`})),z=null,we=null,Ee=null,Oe=y(!1),ke=y(!1),B=!1,Ae=0,V=0,je=0,Me=!1;function Ne(e){let t=o(U)?.getBoundingClientRect();if(!t||t.width===0)return o(I);let n=(e-t.left)/t.width*2-1,r=Math.max(0,Math.min(1,n/.835*.5+.5));return o(R)+r*(365-o(R))}function Pe(e){let t=o(U)?.getBoundingClientRect();if(!t||t.height===0)return!1;let n=(e.clientY-t.top)/t.height;return n>.7675000000000001&&n<.9175}function Fe(){Ae&&cancelAnimationFrame(Ae),Ae=0}function Ie(t){o(Oe)&&!j()&&Pe(t)&&(Fe(),B=!0,e(L,!0),V=0,je=performance.now(),e(I,Ne(t.clientX),!0),t.currentTarget.setPointerCapture?.(t.pointerId),t.preventDefault())}function Le(t){if(!B)return;let n=performance.now(),r=Ne(t.clientX),i=Math.max(1,n-je);V=V*.6+(r-o(I))/i*16*.4,je=n,e(I,r,!0)}function ze(t){if(!B)return;B=!1,Me=!0,t.currentTarget.releasePointerCapture?.(t.pointerId);let n=()=>{Ae=0,V*=.94;let t=o(I)+V;t<=o(R)&&(t=o(R),V=0),t>=365&&(t=365,V=0);let r=o(I)<0&&V>0||o(I)>0&&V<0;Math.abs(t)<1&&r&&(t=0,V=0),e(I,t,!0),Math.abs(V)>.02?Ae=requestAnimationFrame(n):e(L,!1)};Math.abs(V)>.05?Ae=requestAnimationFrame(n):e(L,!1)}function Be(){B=!1,e(L,!1),Fe()}let H=y(!1),Ve=y(!1);function He(){if(typeof window>`u`)return;let t=window.matchMedia(`(prefers-reduced-motion: reduce)`);t.matches&&!o(Ve)&&e(H,!0);let n=t=>{o(Ve)||e(H,t.matches,!0)};return t.addEventListener(`change`,n),()=>t.removeEventListener(`change`,n)}function Ue(){e(Ve,!0),e(H,!o(H))}d(()=>{o(K)?.setPaused(o(H))});let We=y(``),Ge=y(0),Ke=k(()=>o(We)!==``&&o(Ge)>0),U=y(null),W=new Re,qe=y(null),Je=0,Ye=y(``);async function Xe(t){if(Me){Me=!1;return}if(!q||!o(U))return;let n=o(U).getBoundingClientRect();if(n.width===0||n.height===0)return;let r=(t.clientX-n.left)/n.width*2-1,i=-((t.clientY-n.top)/n.height*2-1),a=await q.pickAt(r,i);a&&(e(qe,{kind:`memory`,id:a.id,label:`Field cell`},!0),c.onpick?.(a.id))}function Qe(e){Ie(e),!(B||j())&&(W.enabled=!j(),W.onPointerDown(e))}function $e(t){if(Le(t),B||j())return;W.onPointerMove(t)&&(Me=!0,q?.setCameraRig(W.state));let n=performance.now();if(n-Je<120||!q||!o(U))return;Je=n;let r=o(U).getBoundingClientRect();if(r.width===0)return;let i=(t.clientX-r.left)/r.width*2-1,a=-((t.clientY-r.top)/r.height*2-1);q.pickAt(i,a).then(t=>{q?.setHovered(t?.index??-1),e(Ye,t?.id?.slice(0,8)??``,!0),o(U)&&(o(U).style.cursor=t?`crosshair`:`grab`)})}function et(e){ze(e),W.onPointerUp(e)}function tt(){Be(),W.onPointerUp({pointerId:-1})}function nt(e){j()||Pe(e)||W.onWheel(e)&&q?.setCameraRig(W.state)}let rt={"recall-path":`RECALL`,"engram-birth":`BIRTH`,"salience-rescue":`RESCUE`,"forgetting-horizon":`HORIZON`,firewall:`FIREWALL`};function it(){return q?.graph?new Uint32Array(q.graph.nodes.map(e=>({index:e.index,id:e.id,retention:e.retention})).sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,64).map(e=>e.index).sort((e,t)=>e-t)):new Uint32Array}let at=y(!j());function ot(t){let n=t.target;n?.isContentEditable||n?.tagName===`INPUT`||n?.tagName===`TEXTAREA`||n?.tagName===`SELECT`||((t.key===`h`||t.key===`H`)&&e(at,!o(at)),t.key===`Escape`&&c.onexit&&c.onexit(),(t.key===` `||t.key.toLowerCase()===`p`)&&!j()&&(t.preventDefault(),Ue()))}let G=y(null),st=y(!0),ct=y(``),lt=y(0),ut=y(0),dt=y(0),ft=y(0),pt=y(``),K=y(null),q=null,ht=null,_t=null,vt=y(null),yt=null,bt=null,xt=y(null),St=!1,Ct=y(l([]));async function wt(){e(st,!0),e(ct,``);try{if(F()){e(G,F()),e(dt,F().nodeCount,!0),e(ft,F().edgeCount,!0),e(pt,F().center_id,!0);return}let t=new Set(fe().filter(Boolean)),n=t.size?await(async()=>{let e=await Promise.all([...t].map(e=>se.graph({center_id:e,max_nodes:200,depth:3}))),n=[...new Map(e.flatMap(e=>e.nodes).map(e=>[e.id,e])).values()].filter(e=>t.has(e.id)),r=new Set(n.map(e=>e.id)),i=[...new Map(e.flatMap(e=>e.edges).map(e=>[`${e.source}:${e.target}`,e])).values()].filter(e=>r.has(e.source)&&r.has(e.target));return{...e[0],nodes:n,edges:i,center_id:n[0]?.id??e[0]?.center_id??``,nodeCount:n.length,edgeCount:i.length}})():await se.graph({max_nodes:200,depth:3,sort:`connected`});e(G,n,!0),e(dt,n.nodeCount,!0),e(ft,n.edgeCount,!0),e(pt,n.center_id,!0)}catch(t){let n=t instanceof Error?t.message:`Failed to load graph data`;/\b404\b/.test(n)?(e(G,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:``},!0),e(dt,0),e(ft,0),e(pt,``)):e(ct,n,!0)}finally{e(st,!1)}}let J=null,Tt=y(l([])),Et=y(`recalls`);function Y(t,n){e(lt,t,!0),e(ut,n,!0),J&&!o(L)&&J.tick(t)}async function X(){if(!z||!q?.graph)return;let t=q.graph,n=e=>t.indexById.has(e),r=e=>t.nodes[t.indexById.get(e)??-1]?.label??e.slice(0,8),i=[];try{i=(await se.receipts.list(60))?.receipts??[]}catch{}let a=he(i,n);a.length===0&&(a=ge(t.nodes,12)),a.length>0&&(J=new ve(z,{intervalFrames:240}),J.setItems(a));let o=_e(i,n,3);o.length>0?(e(Et,`recalls`),e(Tt,o.map(e=>({...e,label:r(e.id)})),!0)):(e(Et,`retention`),e(Tt,[...t.nodes].filter(e=>(e.label??``).trim().length>0).sort((e,t)=>t.retention-e.retention).slice(0,3).map(e=>({id:e.id,recalls:Math.round(e.retention*100),label:e.label||e.id.slice(0,8)})),!0))}function Z(t){St=!1,q?.dispose(),e(K,t,!0),q=new Ze(t),W.enabled=!j(),j()&&W.reset(),q.setCameraRig(W.state),c.onready?.(t)}d(()=>{if(o(K)&&q&&o(G)&&!St){St=!0;let t=c.demo===`engram-birth`,n=c.demo===`salience-rescue`,r=c.demo===`forgetting-horizon`,i=c.demo===`firewall`;if(q.upload(o(G),A(),{recallPath:!t&&!n&&!r&&!i}),t){ht=new mt({engine:o(K),nodeRenderer:q,seed:A()}),ht.upload(A());let t=ht.engraveSteps,n=[];for(let e=0;e<t.length/4;e++)n.push({sourceIndex:t[e*4],targetIndex:t[e*4+1],beatFrame:t[e*4+2],kind:t[e*4+3],beatKind:`engrave`,nodeId:`engrave-${e}`,label:`edge engraved`});q.setPathSteps(t,n),e(Ct,ht.timeline.map((e,t)=>({sourceIndex:0,targetIndex:0,beatFrame:e.startFrame,kind:0,beatKind:`birth`,nodeId:`birth-${t}`,label:e.label})),!0)}else if(n){let t=Dt(o(G),q.graph,A(),c.backfillEvidence);e(vt,t,!0),t.viable&&(_t=new gt({engine:o(K),nodeRenderer:q,plan:t}),_t.upload(),q.setPathSteps(t.pathData,t.pathMetas)),e(Ct,t.spineBeats,!0)}else if(r){let t=It(q.graph);t.viable&&(yt=new jt({engine:o(K),nodeRenderer:q,plan:t}),yt.upload(),q.setPathSteps(t.pathData,t.pathMetas)),e(Ct,t.spineBeats,!0)}else if(i){let t=Jt(q.graph,A());e(xt,t,!0),t.viable&&(bt=new zt({engine:o(K),nodeRenderer:q,plan:t}),bt.upload(),q.setPathSteps(t.pathData,t.pathMetas)),e(Ct,t.spineBeats,!0)}else e(Ct,q.pathSteps,!0);if(me()&&q.graph&&o(G)){z=new on({engine:o(K),renderer:q,graph:q.graph,response:o(G),seed:A(),projectionDays:()=>o(be),chronoOffsetDays:()=>o(xe),onFirewall:t=>{e(We,t.intruderLabel,!0),e(Ge,Date.now(),!0)}}),e(ke,z.liveDecayAvailable,!0),o(K).setPreFrameHook(e=>z?.drain(e)),j()||X();let t=1/0;for(let e of q.graph.nodes)if(e.createdAt){let n=Date.parse(e.createdAt);Number.isFinite(n)&&n<t&&(t=n)}if(Number.isFinite(t)&&e(R,Math.floor((t-Date.now())/864e5)-1),ye){let t=Date.parse(ye);Number.isFinite(t)&&e(I,Math.min(365,Math.max(o(R),(t-Date.now())/864e5)),!0),ye=null}j()||(Ee=new Dn(o(K),q,it()),o(K).addPass(Ee),we=new vn(o(K),q.graph.nodes),o(K).addPass(we),e(Oe,!0)),typeof window<`u`&&(window.__vestigeLiveBridge=z)}o(K).demoClock.reset()}}),d(()=>{let e=x();z&&z.ingest(e)}),d(()=>{o(I),z?.refreshDecay(),we?.setTimeline(o(I),o(L)),Ee?.setScrubbing(o(L))}),d(()=>{if(!o(Ge))return;let t=setTimeout(()=>{e(We,``),e(Ge,0)},7e3);return()=>clearTimeout(t)}),O(()=>{ye=new URLSearchParams(window.location.search).get(`t`),wt();let e=He();return()=>{if(Fe(),e?.(),q?.dispose(),q=null,typeof window<`u`){let e=window;e.__vestigeLiveBridge===z&&delete e.__vestigeLiveBridge}}});var Ot=Hn();i(`keydown`,ee,ot);var kt=p(Ot);let At;var Q=a(kt);let Mt;var Nt=a(Q);de(Nt,{get demo(){return c.demo},get seed(){return A()},get freezeFrame(){return ce()},get maxDpr(){return P()},onframe:Y,onready:Z}),E(Q),_(Q,t=>e(U,t),()=>o(U));var Pt=m(Q,2),Ft=t=>{var l=Bn(),d=a(l),p=e=>{var t=kn(),n=a(t),r=f(n,!0),i=m(n,2);ie(i,19,()=>o(Tt),e=>e.id,(e,t,n)=>{var r=On(),i=a(r),c=f(i,!0),l=m(i,2),d=f(l,!0),p=m(l,2),h=f(p);E(r),u(()=>{s(c,o(n)+1),b(l,`title`,o(t).label),s(d,o(t).label),s(h,`${o(t).recalls??``}${o(Et)===`recalls`?`×`:`%`}`)}),w(e,r)}),E(t),u(()=>s(r,o(Et)===`recalls`?`Most recalled · your mind`:`Strongest memories · your mind`)),w(e,t)};n(d,e=>{me()&&o(Tt).length>0&&e(p)});var g=m(d,2),_=e=>{var t=An(),n=m(a(t),2),r=f(n,!0);D(2),E(t),u(()=>s(r,o(We))),w(e,t)};n(g,e=>{me()&&o(Ke)&&e(_)});var y=m(g,2),x=e=>{var t=jn(),n=f(t,!0);u(()=>{b(t,`title`,o(H)?`Resume field motion`:`Pause field motion`),b(t,`aria-pressed`,o(H)),b(t,`aria-label`,o(H)?`Resume 3D memory field motion`:`Pause 3D memory field motion`),s(n,o(H)?`▶ RESUME`:`❚❚ PAUSE`)}),r(`click`,t,Ue),w(e,t)};n(y,e=>{j()||e(x)});var S=m(y,2),C=t=>{var c=Nn();let l;var d=m(a(c),2);v(d);var p=m(d,2);let g;var _=f(p,!0),y=m(p,2),x=t=>{var n=Mn();r(`click`,n,()=>e(I,0)),w(t,n)};n(y,e=>{o(I)!==0&&e(x)}),E(c),u(()=>{l=h(c,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]`,null,l,{"opacity-100":o(Oe),"opacity-75":!o(Oe)}),b(d,`min`,o(R)),g=h(p,1,`w-16 text-right tabular-nums`,null,g,{"text-[#b9d9a9]":o(I)>=0,"text-[#dfc68e]":o(I)<0}),s(_,o(Ce))}),r(`input`,d,()=>e(L,!0)),r(`change`,d,()=>e(L,!1)),r(`pointerup`,d,()=>e(L,!1)),i(`pointercancel`,d,()=>e(L,!1)),i(`blur`,d,()=>e(L,!1)),ne(d,()=>o(I),t=>e(I,t)),w(t,c)};n(S,e=>{me()&&o(ke)&&e(C)});var T=m(S,2),ee=e=>{Se(e,{get demoMode(){return c.demo},get seed(){return A()},get nodeCount(){return o(dt)},get edgeCount(){return o(ft)},get centerId(){return o(pt)},get frameCount(){return o(lt)},get fpsEstimate(){return o(ut)},get freezeFrame(){return ce()},get loading(){return o(st)},get error(){return o(ct)}})};n(T,e=>{N()===`full`&&e(ee)});var O=m(T,2),te=e=>{var t=Pn();r(`click`,t,function(...e){c.onexit?.apply(this,e)}),w(e,t)};n(O,e=>{N()===`full`&&c.onexit&&e(te)});var re=m(O,2),ae=e=>{var t=In();ie(t,20,()=>ue,e=>e,(e,t)=>{var n=Fn(),i=f(n,!0);u(()=>{h(n,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${t===c.demo?`text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]`:`text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50`}`),b(n,`title`,`Play the ${rt[t]??``} moment`),s(i,rt[t])}),r(`click`,n,()=>c.ondemochange?.(t)),w(e,n)}),E(t),w(e,t)};n(re,e=>{N()===`full`&&le()&&e(ae)});var oe=m(re,2),se=e=>{var t=Ln();w(e,t)};n(oe,e=>{o(st)&&e(se)});var M=m(oe,2),P=e=>{var t=Rn(),n=a(t),r=f(n,!0);E(t),u(()=>s(r,o(ct))),w(e,t)};n(M,e=>{o(ct)&&!o(st)&&e(P)});var de=m(M,2),fe=e=>{Te(e,{get steps(){return o(Ct)},get frame(){return o(lt)}})};n(de,e=>{N()===`full`&&e(fe)});var pe=m(de,2),F=e=>{De(e,{get frame(){return o(lt)},get verdict(){return o(vt).verdict}})};n(pe,e=>{N()===`full`&&c.demo===`salience-rescue`&&o(vt)?.viable&&e(F)});var he=m(pe,2),ge=e=>{{let t=k(()=>({headline:o(xt).verdict.headline,causeLabel:o(xt).verdict.intruderLabel,receipt:o(xt).verdict.receipt}));De(e,{get frame(){return o(lt)},tone:`quarantine`,fadeWindow:[480,495,605,620],get verdict(){return o(t)}})}};n(he,e=>{N()===`full`&&c.demo===`firewall`&&o(xt)?.viable&&e(ge)});var _e=m(he,2),ve=e=>{var t=zn();w(e,t)};n(_e,e=>{!o(st)&&o(G)&&o(G).nodeCount===0&&e(ve)}),E(l),w(t,l)};n(Pt,e=>{o(at)&&e(Ft)}),E(kt);var Lt=m(kt,2);pe(Lt,{get pick(){return o(qe)},onclose:()=>e(qe,null)});var Rt=m(Lt,2),Bt=e=>{var t=Vn(),n=f(t);u(()=>s(n,`HOVER ${o(Ye)??``}`)),w(e,t)};n(Rt,e=>{o(Ye)&&!j()&&e(Bt)}),u(()=>{At=h(kt,1,`${M()?`absolute`:`fixed`} inset-0 overflow-hidden bg-[#05060a]`,null,At,{"cursor-none":j()}),Mt=h(Q,1,`absolute inset-0 z-0 touch-none`,null,Mt,{"cursor-crosshair":!!c.onpick&&!j()})}),r(`click`,Q,Xe),r(`pointerdown`,Q,Qe),r(`pointermove`,Q,$e),r(`pointerup`,Q,et),i(`pointercancel`,Q,tt),i(`wheel`,Q,nt),w(t,Ot),T(),ae()}c([`click`,`pointerdown`,`pointermove`,`pointerup`,`input`,`change`]);export{Un as t};
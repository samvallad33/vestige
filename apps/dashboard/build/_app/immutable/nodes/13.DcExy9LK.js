import{A as e,B as t,C as n,D as r,E as i,G as a,H as o,I as s,J as c,K as l,M as u,N as d,Q as f,R as p,S as m,T as h,U as g,W as _,X as v,Y as y,a as b,b as ee,d as x,dt as S,et as te,f as ne,g as C,h as w,j as T,k as E,l as re,mt as ie,n as ae,ot as oe,p as se,q as D,r as ce,rt as le,s as ue,st as de,tt as fe,u as pe,ut as me,w as O,z as k}from"../chunks/CvawvY3d.js";import{l as he,n as ge}from"../chunks/eHoGjb3p.js";import"../chunks/BwXKaWn1.js";import"../chunks/xihTtKlq.js";import"../chunks/C_EsKhQW.js";import{t as _e}from"../chunks/But8FtBT.js";import{n as ve,t as A}from"../chunks/26sNC5Du.js";import{c as ye,i as j}from"../chunks/DKuNZ7Jg.js";import{r as be}from"../chunks/CpCIhK-b.js";import{n as xe}from"../chunks/BNSKl0Mo.js";import{n as Se,t as M}from"../chunks/67umMfSM.js";import{t as Ce}from"../chunks/D4_bBKYm.js";import{a as N,c as we,d as Te,i as Ee,l as De,n as Oe,o as P,r as F,s as I,t as ke,u as L}from"../chunks/CoEqe9GD.js";import{t as Ae}from"../chunks/CJhdMica.js";import{t as R}from"../chunks/Dnqz_WhD.js";import{d as je,f as Me,i as Ne,o as Pe,r as z,t as Fe,u as Ie}from"../chunks/Bbp_g6ip.js";import{i as Le,n as Re}from"../chunks/DltdpSW9.js";import{t as ze}from"../chunks/BcX2BVfA.js";import{r as B}from"../chunks/D2q35Vo6.js";import{r as Be}from"../chunks/BciDxNzc.js";var Ve=ie({prerender:()=>!1,ssr:()=>!1}),He=[`decision`,`correction`,`fact`,`pattern`,`note`,`event`],V=36;function Ue(e){return{id:e.id,label:(e.content??``).slice(0,160),type:e.nodeType,retention:e.retentionStrength,tags:e.tags??[],createdAt:e.createdAt,updatedAt:e.updatedAt,isCenter:!1,lastAccessed:e.lastAccessedAt}}var H=null,We=4e3;function Ge(){if(H)return H;let e=Ke();return H=e,e.then(()=>setTimeout(()=>H===e&&(H=null),We),()=>H=null),e}async function Ke(){let e=await Promise.all(He.map(e=>A.memories.list({node_type:e,limit:String(V),sort:`newest`}).then(e=>e.memories??[]).catch(()=>[]))),t=e.map(e=>e[0]?.id).filter(e=>!!e);if(t.length<2)return A.graph({max_nodes:200,depth:3,sort:`connected`});let n=(await Promise.all(t.map(e=>A.graph({center_id:e,max_nodes:V,depth:2}).catch(()=>null)))).filter(e=>!!e),r=new Map;He.forEach((t,i)=>{let a=e[i][0]?.id,o=n.find(e=>e.center_id===a)?.nodes??[],s=0;for(let e of o){if(s>=V)break;r.has(e.id)||(r.set(e.id,{...e,isCenter:!1}),s++)}for(let t of e[i]){if(s>=V)break;r.has(t.id)||(r.set(t.id,Ue(t)),s++)}});let i=t[0],a=r.get(i);a&&r.set(i,{...a,isCenter:!0});let o=new Map;for(let e of n)for(let t of e.edges)r.has(t.source)&&r.has(t.target)&&o.set(`${t.source}:${t.target}:${t.type}`,t);let s=[...o.values()];return{...n[0]??await A.graph({max_nodes:1,depth:1}),nodes:[...r.values()],edges:s,center_id:i,nodeCount:r.size,edgeCount:s.length}}function qe(e,t){let n=[];for(let r of e){let e=((r.activation_path?.length?r.activation_path:r.retrieved)??[]).filter(t);if(e.length===0)continue;let i=e[e.length-1];n.push({targetId:i,pathIds:e})}return n}function Je(e,t=12){return[...e].sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,t).map(e=>({targetId:e.id,pathIds:[e.id]}))}function Ye(e,t,n=5){let r=new Map;for(let n of e){let e=(n.activation_path?.length?n.activation_path:n.retrieved)??[];for(let n of new Set(e))t(n)&&r.set(n,(r.get(n)??0)+1)}return[...r.entries()].map(([e,t])=>({id:e,recalls:t})).sort((e,t)=>t.recalls-e.recalls||e.id.localeCompare(t.id)).slice(0,n)}var Xe=class{bridge;items=[];cursor=0;ticks=0;nextTick=0;intervalFrames;enabled=!0;started=!1;constructor(e,t={}){this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}let t=this.items[this.cursor%this.items.length];this.cursor++;let n=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(n?0:30)}},Ze=r(`<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>`),Qe=r(`<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>`),$e=r(`<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>`),U=r(`<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>`);function et(e,t){de(t,!0);let r=b(t,`demoMode`,3,`recall-path`),i=b(t,`seed`,3,`vestige-observatory-v1`),a=b(t,`nodeCount`,3,0),o=b(t,`edgeCount`,3,0),s=b(t,`centerId`,3,``),c=b(t,`frameCount`,3,0),u=b(t,`fpsEstimate`,3,0),d=b(t,`freezeFrame`,3,null);b(t,`loading`,3,!1),b(t,`error`,3,``);function f(){let e=new URLSearchParams({demo:r(),seed:i()});d()!==null&&e.set(`frame`,String(d()));let t=`${window.location.origin}${he}/observatory?${e.toString()}`;navigator.clipboard.writeText(t).catch(()=>{})}var p=U(),m=_(p),g=_(m),v=_(g),y=l(v,!0),ee=D(v,2),x=l(ee);S(g);var te=D(g,2),ne=_(te),C=l(ne),w=D(ne,2),E=e=>{var t=Ze(),n=l(t);k(e=>O(n,`center=${e??``}`),[()=>s().slice(0,8)]),h(e,t)};n(w,e=>{s()&&e(E)}),S(te);var re=D(te,2),ie=_(re),ae=l(ie),se=D(ie,2),ce=e=>{var t=Qe();h(e,t)},le=e=>{var t=$e(),n=l(t);k(()=>O(n,`${u()??``}fps`)),h(e,t)};n(se,e=>{d()===null?u()>0&&e(le,1):e(ce)});var ue=D(se,2);S(re),S(m),S(p),k((e,t)=>{O(y,r()),O(x,`seed=${e??``}${i().length>12?`…`:``}`),O(C,`${a()??``} nodes · ${o()??``} edges`),O(ae,`frame: ${t??``}`)},[()=>i().slice(0,12),()=>String(c()).padStart(3,` `)]),T(`click`,ue,f),h(e,p),oe()}e([`click`]);var tt=r(`<div class="active-label svelte-8n8iia"> </div>`),nt=r(`<div></div>`),W=r(`<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>`);function rt(e,t){de(t,!0);let r=b(t,`steps`,19,()=>[]),o=b(t,`frame`,3,0),s=b(t,`loopFrames`,3,720),c=e=>e/s()*100;function u(e,t){let n=t-e;return n<-14||n>90?0:n<0?1+n/14:1-n/90}let p=f(()=>{let e=``,t=.15;for(let n of r()){let r=u(n.beatFrame,o());r>t&&(t=r,e=n.label)}return e});var m=i(),g=a(m),v=e=>{var t=W(),i=_(t),a=e=>{var t=tt(),n=l(t,!0);k(()=>O(n,d(p))),h(e,t)};n(i,e=>{d(p)&&e(a)});var s=D(i,2),f=_(s);ee(f,17,r,e=>e.beatFrame,(e,t)=>{var n=nt();let r;k((e,i,a)=>{r=se(n,1,`tick svelte-8n8iia`,null,r,{hot:e,backward:d(t).kind===1}),ne(n,`left: ${i??``}%; opacity: ${a??``}`),x(n,`title`,d(t).label)},[()=>u(d(t).beatFrame,o())>0,()=>c(d(t).beatFrame),()=>.45+.55*u(d(t).beatFrame,o())]),h(e,n)});var m=D(f,2);S(s),S(t),k(e=>ne(m,`left: ${e??``}%`),[()=>c(o())]),h(e,t)};n(g,e=>{r().length>0&&e(v)}),h(e,m),oe()}var it=r(`<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>`);function at(e,t){de(t,!0);let r=b(t,`frame`,3,0),o=b(t,`fadeWindow`,19,()=>[600,620,705,719]),s=b(t,`tone`,3,`triumph`),c=(e,t,n)=>{let r=Math.min(1,Math.max(0,(n-e)/(t-e)));return r*r*(3-2*r)},u=f(()=>c(o()[0],o()[1],r())*(1-c(o()[2],o()[3],r())));var p=i(),m=a(p),g=e=>{var n=it();let r;var i=_(n),a=l(i,!0),o=D(i,2),c=l(o,!0),f=D(o,2),p=l(f,!0);S(n),k(()=>{r=se(n,1,`verdict svelte-ssd7yu`,null,r,{quarantine:s()===`quarantine`}),ne(n,`opacity: ${d(u)??``}`),O(a,t.verdict.headline),O(c,t.verdict.causeLabel),O(p,t.verdict.receipt)}),h(e,n)};n(m,e=>{d(u)>.001&&e(g)}),h(e,p),oe()}function ot(e,t,n,r){let i=1/Math.tan(e/2),a=1/(n-r),o=new Float32Array(16);return o[0]=i/t,o[5]=i,o[10]=r*a,o[11]=-1,o[14]=r*n*a,o}function st(e,t,n){let[r,i,a]=e,o=r-t[0],s=i-t[1],c=a-t[2],l=Math.hypot(o,s,c)||1;o/=l,s/=l,c/=l;let u=n[1]*c-n[2]*s,d=n[2]*o-n[0]*c,f=n[0]*s-n[1]*o;l=Math.hypot(u,d,f)||1,u/=l,d/=l,f/=l;let p=s*f-c*d,m=c*u-o*f,h=o*d-s*u,g=new Float32Array(16);return g[0]=u,g[1]=p,g[2]=o,g[4]=d,g[5]=m,g[6]=s,g[8]=f,g[9]=h,g[10]=c,g[12]=-(u*r+d*i+f*a),g[13]=-(p*r+m*i+h*a),g[14]=-(o*r+s*i+c*a),g[15]=1,g}function ct(e,t){let n=new Float32Array(16);for(let r=0;r<4;r++)for(let i=0;i<4;i++)n[r*4+i]=e[i]*t[r*4]+e[4+i]*t[r*4+1]+e[8+i]*t[r*4+2]+e[12+i]*t[r*4+3];return n}function lt(e,t,n,r=.35,i=0){let a=e*Math.PI*2+i,o=[Math.sin(a)*n,n*r,Math.cos(a)*n],s=ot(50*Math.PI/180,t,.1,4e3),c=st(o,[0,0,0],[0,1,0]),l=-o[0],u=-o[1],d=-o[2],f=Math.hypot(l,u,d)||1;l/=f,u/=f,d/=f;let p=u*0-d*1,m=d*0-l*0,h=l*1-u*0;f=Math.hypot(p,m,h)||1,p/=f,m/=f,h/=f;let g=m*d-h*u,_=h*l-p*d,v=p*u-m*l;return{viewProj:ct(s,c),right:[p,m,h],up:[g,_,v],eye:o}}var ut={yaw:0,pitch:0,zoom:1},dt=.38,ft=2.6,pt=-.18,mt=.82;function ht(e,t,n){return Math.min(n,Math.max(t,e))}function G(e){return{yaw:Number.isFinite(e.yaw)?e.yaw:0,pitch:ht(Number.isFinite(e.pitch)?e.pitch:0,pt,mt),zoom:ht(Number.isFinite(e.zoom)?e.zoom:1,dt,ft)}}function K(e,t,n,r=ut){let i=G(r);return lt(e,t,n/i.zoom,.35+i.pitch,i.yaw)}var gt=class{state={...ut};dragging=!1;pointerId=null;lastX=0;lastY=0;pinch0=0;pointers=new Map;enabled=!0;reset(){this.state={...ut},this.dragging=!1,this.pointerId=null,this.pointers.clear()}onPointerDown(e){this.enabled&&e.button===0&&(this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===1?(this.dragging=!0,this.pointerId=e.pointerId,this.lastX=e.clientX,this.lastY=e.clientY,e.currentTarget?.setPointerCapture?.(e.pointerId)):this.pointers.size===2&&(this.pinch0=q(this.pointers)))}onPointerMove(e){if(!this.enabled)return!1;if(this.pointers.has(e.pointerId)&&this.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY}),this.pointers.size===2&&this.pinch0>0){let e=q(this.pointers),t=e/this.pinch0;return this.state=G({...this.state,zoom:this.state.zoom*ht(t,.94,1.06)}),this.pinch0=e,!0}if(!this.dragging||e.pointerId!==this.pointerId)return!1;let t=e.clientX-this.lastX,n=e.clientY-this.lastY;return this.lastX=e.clientX,this.lastY=e.clientY,this.state=G({yaw:this.state.yaw-t*.005,pitch:this.state.pitch+n*.003,zoom:this.state.zoom}),!0}onPointerUp(e){this.pointers.delete(e.pointerId),e.pointerId===this.pointerId&&(this.dragging=!1,this.pointerId=null),this.pointers.size<2&&(this.pinch0=0)}onWheel(e){if(!this.enabled)return!1;e.preventDefault();let t=e.deltaY>0?.92:1.08;return this.state=G({...this.state,zoom:this.state.zoom*t}),!0}};function q(e){let t=[...e.values()];return t.length<2?0:Math.hypot(t[0].x-t[1].x,t[0].y-t[1].y)}function _t(e){let t=/^#?([0-9a-fA-F]{6})$/.exec(e.trim());if(!t)return[107/255,114/255,128/255];let n=parseInt(t[1],16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}function vt(e){let t=Be({tags:e.tags});return t?_t(t):be(e.type)}function yt(e){let t=[...e.nodes].sort((e,t)=>e.isCenter===t.isCenter?e.id<t.id?-1:+(e.id>t.id):e.isCenter?-1:1).map((e,t)=>De(e,t)),n=new Map;for(let e of t)n.set(e.id,e.index);let r=[];for(let t of e.edges){let e=n.get(t.source),i=n.get(t.target);e!==void 0&&i!==void 0&&e!==i&&r.push({sourceIndex:e,targetIndex:i,weight:t.weight,type:t.type})}let i=t.findIndex(e=>e.isCenter);return{nodes:t,edges:r,indexById:n,centerIndex:i<0?0:i}}function bt(e,t,n=120){let r=e.nodes.length,i=new Float32Array(r*16);for(let a=0;a<r;a++){let o=e.nodes[a],s=a*16,[c,l,u]=o.isCenter&&e.centerIndex===a?[0,0,0]:Te(a,r,n,t),d=o.isCenter?4.2:1.4+o.retention*1.8;i[s+N.posRadius+0]=c,i[s+N.posRadius+1]=l,i[s+N.posRadius+2]=u,i[s+N.posRadius+3]=d,i[s+N.velRetention+3]=o.retention;let[f,p,m]=vt(o),h=0;o.isCenter&&(h|=Ee.isCenter),o.suppressed&&(h|=Ee.suppressed);let g=new Set(o.tags.map(e=>e.toLowerCase()));g.has(`aha`)&&(h|=Ee.isAha),(g.has(`failure`)||g.has(`guardrail`))&&(h|=Ee.isFailure),(g.has(`confusion`)||g.has(`weak-spot`))&&(h|=Ee.isConfusion),i[s+N.colorFlags+0]=f,i[s+N.colorFlags+1]=p,i[s+N.colorFlags+2]=m,i[s+N.colorFlags+3]=h}return{data:i,nodeCount:r}}function xt(e){let t=new Uint32Array(Math.max(1,e.edges.length)*2);return e.edges.forEach((e,n)=>{t[n*2]=e.sourceIndex,t[n*2+1]=e.targetIndex}),t}var St=`
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

// The node's own colour is its memory type (the palette every organ shares).
// Retention is how strongly it is held right now: a well-held memory shows its
// type at full saturation; as it fades it loses saturation and sinks toward a
// cool ash, so forgetting reads as light going out, not as a different hue.
fn fossil_tone(source: vec3<f32>, retention: f32) -> vec3<f32> {
	let held = smoothstep(0.08, 0.9, clamp(retention, 0.0, 1.0));
	let src = clamp(source, vec3<f32>(0.0), vec3<f32>(1.0));
	let luma = dot(src, vec3<f32>(0.299, 0.587, 0.114));
	let saturated = mix(vec3<f32>(luma), src, mix(0.28, 1.0, held));
	let ash = vec3<f32>(0.16, 0.19, 0.26);
	return mix(ash, saturated, mix(0.3, 1.0, held));
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
	// A well-held soma warms toward white at its very core but keeps its type colour.
	let soma_tone = mix(tone, vec3<f32>(0.96, 0.97, 0.94), consolidated * 0.22);
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
		// The recalled memory fires in its OWN type colour (brightened), so a
		// recall path across a decision, a fact and a correction reads as three
		// different lights, not one green wave.
		let own = tone / max(0.001, max(tone.r, max(tone.g, tone.b)));
		let gcamp = mix(own, vec3<f32>(1.0, 0.98, 0.9), clamp(recall - 0.6, 0.0, 1.0) * 0.6);
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
`,J=`
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
`,Y=`
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
`,Ct=`
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
`;function wt(e){return 60+e*60}function Tt(e,t,n=8,r={}){let i=[...e.nodes].sort((e,t)=>e.id<t.id?-1:+(e.id>t.id)),a=[...e.edges].sort((e,t)=>{let n=`${e.source} ${e.target} ${e.type}`,r=`${t.source} ${t.target} ${t.type}`;return n<r?-1:+(n>r)}),o=r.centerId??e.center_id,s=B(i,a,o,n,{preferCausal:r.preferCausal}),c=[];for(let e=0;e<s.beats.length;e++){let n=s.beats[e],r=t.indexById.get(n.nodeId);if(r===void 0)continue;let i=e>0?s.beats[e-1].nodeId:n.nodeId,a=t.indexById.get(i)??r,o=(n.viaEdge?.type??``).toLowerCase(),l=o===`causal`||o.includes(`causal`),u=n.kind===`contradiction`||l;c.push({sourceIndex:a,targetIndex:r,beatFrame:wt(e),kind:u?I.backwardCause:I.recall,beatKind:n.kind,nodeId:n.nodeId,label:n.node.label})}let l=new Uint32Array(Math.max(1,c.length)*4);return c.forEach((e,t)=>{l[t*4]=e.sourceIndex,l[t*4+1]=e.targetIndex,l[t*4+2]=e.beatFrame,l[t*4+3]=e.kind}),{data:l,steps:c,path:s}}var Et=24,Dt=300,Ot=128,kt=class{engine;pipeline=null;bindGroup=null;cameraBuffer=null;nodeBuffer=null;edgeBuffer=null;cameraData=new Float32Array(Et);nodeCount=0;simPipeline=null;simBindGroup=null;pathBuffer=null;liveRetentionBuffer=null;pickReadback=null;disposed=!1;edgeCapacityBytes=0;edgeCount=0;cameraRig={...ut};frame={scale:1,x:0,y:0};hoveredIndex=-1;pathPipeline=null;pathBindGroup=null;pathStepCount=0;axonPipeline=null;axonBindGroup=null;graph=null;pathSteps=[];constructor(e){this.engine=e,e.addPass(this)}upload(e,t,n){let r=this.engine.gpuDevice;if(!r)return;let i=n?.recallPath??!0,a=yt(e);this.graph=a;let{data:o,nodeCount:s}=bt(a,new L({seed:t}).state.rng);this.nodeCount=s,this.nodeBuffer?.destroy(),this.nodeBuffer=r.createBuffer({label:`observatory-node-state`,size:Math.max(o.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),r.queue.writeBuffer(this.nodeBuffer,0,o.buffer);let c=xt(a);this.edgeCount=a.edges.length,this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(c.byteLength*2,64),this.edgeBuffer=r.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.edgeBuffer,0,c.buffer);let l=new Float32Array(Math.max(s,4));for(let e=0;e<s;e++)l[e]=Math.max(.001,a.nodes[e].retention);this.liveRetentionBuffer?.destroy(),this.liveRetentionBuffer=r.createBuffer({label:`observatory-live-retention`,size:l.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.liveRetentionBuffer,0,l.buffer);let u=i?Tt(e,a):{steps:[],data:new Uint32Array(4)};this.pathSteps=u.steps,this.pathBuffer?.destroy(),this.pathBuffer=r.createBuffer({label:`observatory-path-steps`,size:Ot*4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,u.data.buffer,0,Math.min(u.data.byteLength,Ot*4*4)),this.pathStepCount=Math.min(this.pathSteps.length,Ot),this.engine.params[2]=s,this.engine.params[3]=a.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||=r.createBuffer({label:`observatory-camera`,size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.createPipeline(r)}setPathSteps(e,t){let n=this.engine.gpuDevice;if(!n)return;this.pathSteps=t;let r=Ot*4*4;if(this.pathBuffer&&e.byteLength<=r){this.pathStepCount=Math.min(t.length,Ot),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,Ot),this.pathBuffer?.destroy(),this.pathBuffer=n.createBuffer({label:`observatory-path-steps`,size:r,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,r)),this.engine.params[4]=this.pathStepCount,this.createPipeline(n)}setCameraRig(e){this.cameraRig=e}setFrame(e){this.frame=e}setHovered(e){this.hoveredIndex=e}currentOrbit(){let e=this.engine.params[6]||1,t=this.engine.params[7]||1,n=this.engine.params[1],r=this.frame,i=K(n,e/t,Dt/r.scale,this.cameraRig);if(r.x===0&&r.y===0)return i;let a=Float32Array.from(i.viewProj);for(let e=0;e<4;e++)a[e*4+0]+=r.x*a[e*4+3],a[e*4+1]+=r.y*a[e*4+3];return{...i,viewProj:a}}setEdges(e){let t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e,this.edgeCount=e.length;let n=xt(this.graph),r=Math.max(n.byteLength,8),i=!1;(!this.edgeBuffer||r>this.edgeCapacityBytes)&&(this.edgeBuffer?.destroy(),this.edgeCapacityBytes=Math.max(r*2,64),this.edgeBuffer=t.createBuffer({label:`observatory-edge-index`,size:this.edgeCapacityBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),i=!0),t.queue.writeBuffer(this.edgeBuffer,0,n.buffer),this.engine.params[3]=e.length,i&&this.createPipeline(t)}uploadLiveRetention(e){let t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;let n=Math.min(e.length,this.nodeCount);n<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,n*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){let t=e.createShaderModule({label:`observatory-simulate`,code:J});this.simPipeline=e.createComputePipeline({label:`observatory-recall-sim`,layout:`auto`,compute:{module:t,entryPoint:`recall_sim`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&n.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&n.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:`observatory-recall-sim-bind`,layout:this.simPipeline.getBindGroupLayout(0),entries:n})}let t=e.createShaderModule({label:`observatory-render-nodes`,code:St});if(this.pipeline=e.createRenderPipeline({label:`observatory-nodes`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bindGroup=e.createBindGroup({label:`observatory-nodes-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){let t=e.createShaderModule({label:`observatory-render-path`,code:Y});this.pathPipeline=e.createRenderPipeline({label:`observatory-path`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.pathBindGroup=e.createBindGroup({label:`observatory-path-bind`,layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}if(this.edgeBuffer&&this.pathBuffer&&this.nodeBuffer){let t=e.createShaderModule({label:`observatory-render-axons`,code:Ct});this.axonPipeline=e.createRenderPipeline({label:`observatory-axons`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`line-list`}}),this.axonBindGroup=e.createBindGroup({label:`observatory-axons-bind`,layout:this.axonPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.edgeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}},{binding:4,resource:{buffer:this.nodeBuffer}}]})}}compute(e){let t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;let n=this.currentOrbit();if(this.cameraData.set(n.viewProj,0),this.cameraData[16]=n.right[0],this.cameraData[17]=n.right[1],this.cameraData[18]=n.right[2],this.cameraData[19]=0,this.cameraData[20]=n.up[0],this.cameraData[21]=n.up[1],this.cameraData[22]=n.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){let t=e.beginComputePass({label:`observatory-recall-sim`});t.setPipeline(this.simPipeline),t.setBindGroup(0,this.simBindGroup),t.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),t.end()}}render(e){this.axonPipeline&&this.axonBindGroup&&this.edgeCount>0&&(e.setPipeline(this.axonPipeline),e.setBindGroup(0,this.axonBindGroup),e.draw(2,this.edgeCount)),this.pipeline&&this.bindGroup&&this.nodeCount!==0&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){if(this.disposed)return null;let n=this.engine.gpuDevice;if(!n||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;this.pickReadback||=this.readNodePositions(n).finally(()=>{this.pickReadback=null});let r=await this.pickReadback;if(!r||this.disposed||!this.graph)return null;let i=this.currentOrbit().viewProj,a=1/Math.tan(50*Math.PI/360),o=-1,s=1/0;for(let n=0;n<this.nodeCount;n++){let c=n*16+N.posRadius,l=r[c],u=r[c+1],d=r[c+2],f=r[c+3],p=i[3]*l+i[7]*u+i[11]*d+i[15];if(p<=0)continue;let m=(i[0]*l+i[4]*u+i[8]*d+i[12])/p,h=(i[1]*l+i[5]*u+i[9]*d+i[13])/p,g=Math.max(f*a/p,.012),_=Math.hypot(m-e,h-t)/g;_<1.6*(n===this.hoveredIndex?.85:1)&&_<s&&(s=_,o=n)}return o<0?null:{index:o,id:this.graph.nodes[o].id}}async readNodePositions(e){if(!this.nodeBuffer)return null;let t=this.nodeCount*16*4,n=e.createBuffer({label:`observatory-pick-staging`,size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let r=e.createCommandEncoder({label:`observatory-pick-copy`});r.copyBufferToBuffer(this.nodeBuffer,0,n,0,t),e.queue.submit([r.finish()]),await n.mapAsync(GPUMapMode.READ);let i=new Float32Array(n.getMappedRange().slice(0));return n.unmap(),i}catch{return null}finally{n.destroy()}}dispose(){this.disposed=!0,this.nodeBuffer?.destroy(),this.edgeBuffer?.destroy(),this.cameraBuffer?.destroy(),this.pathBuffer?.destroy(),this.liveRetentionBuffer?.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null,this.axonPipeline=null,this.axonBindGroup=null,this.edgeCapacityBytes=0,this.edgeCount=0}},At=16,X=4,jt=110,Mt=.7,Nt=.2,Pt=360,Ft=18;function It(e){if(e.edges.length>0){let t=e.centerIndex,n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t);if(n.length>0){let r=-1,i=-1;for(let a of n){let n=a.sourceIndex===t?a.targetIndex:a.sourceIndex,o=e.nodes[n];o&&o.retention>i&&(i=o.retention,r=n)}if(r>=0)return r}}for(let t=0;t<e.nodes.length;t++)if(t!==e.centerIndex)return t;return e.centerIndex}function Lt(e,t,n=8192){let r=It(e),i=e.nodes[r].id,a=Rt(e,r),o=new L({seed:t+`:birth:`+i}).state.rng,s=new Float32Array(n*At),c=Math.floor(n*Mt),l=Math.floor(n*Nt),u=n-c-l;for(let e=0;e<c;e++){let t=e*At,[n,r,i]=Te(e,c,jt+o()*70,o);s[t+0]=a[0]+n,s[t+1]=a[1]+r,s[t+2]=a[2]+i,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}let d=e.edges.filter(e=>e.sourceIndex===r||e.targetIndex===r);for(let t=0;t<l;t++){let n=(c+t)*At;if(d.length===0)continue;let i=d[t%d.length],u=Rt(e,i.sourceIndex===r?i.targetIndex:i.sourceIndex),f=u[0]-a[0],p=u[1]-a[1],m=u[2]-a[2],h=Math.sqrt(f*f+p*p+m*m)||1,g=t/Math.max(1,l)*2+.5,_=o()*30,v=-p*_/(h||1),y=f*_/(h||1);s[n+0]=a[0]+f/h*g*80+v,s[n+1]=a[1]+p/h*g*80+y,s[n+2]=a[2]+m/h*g*80+0,s[n+3]=o(),s[n+4]=a[0],s[n+5]=a[1],s[n+6]=a[2],s[n+7]=1+o()*1.8,s[n+8]=.91,s[n+9]=1,s[n+10]=.72,s[n+11]=o(),s[n+12]=0,s[n+13]=0,s[n+14]=0,s[n+15]=0}for(let e=0;e<u;e++){let t=(c+l+e)*At,n=o()*Math.PI*2,r=o()*120;s[t+0]=a[0]+Math.cos(n)*r,s[t+1]=a[1]+Math.sin(n)*r,s[t+2]=a[2]+180+o()*40,s[t+3]=o(),s[t+4]=a[0],s[t+5]=a[1],s[t+6]=a[2],s[t+7]=1+o()*1.8,s[t+8]=.91,s[t+9]=1,s[t+10]=.72,s[t+11]=o(),s[t+12]=0,s[t+13]=0,s[t+14]=0,s[t+15]=0}return{targetIndex:r,targetNodeId:i,particles:s,edgeSteps:zt(e,r),timeline:Bt()}}function Rt(e,t){let n=e.nodes[t],r=e.nodes.length;if(n.isCenter&&e.centerIndex===t)return[0,0,0];let i=Math.PI*(3-Math.sqrt(5)),a=1-t/(r-1||1)*2,o=Math.sqrt(1-a*a),s=i*t,c=(t*7+3)%100/100*.1*120-6,l=(t*13+7)%100/100*.1*120-6,u=(t*17+11)%100/100*.1*120-6;return[Math.cos(s)*o*120+c,a*120+l,Math.sin(s)*o*120+u]}function zt(e,t){let n=e.edges.filter(e=>e.sourceIndex===t||e.targetIndex===t),r=n.length;if(r===0)return new Uint32Array;let i=new Uint32Array(r*X);for(let e=0;e<r;e++){let r=n[e],a=r.sourceIndex===t?r.targetIndex:r.sourceIndex,o=Pt+e*Ft;i[e*X+0]=t,i[e*X+1]=a,i[e*X+2]=o,i[e*X+3]=0}return i}function Bt(){return[{label:`latent trace condensing`,startFrame:60,endFrame:239},{label:`engram coalescence`,startFrame:240,endFrame:329},{label:`memory ignition`,startFrame:330,endFrame:359},{label:`associations engrave`,startFrame:360,endFrame:509},{label:`stabilization`,startFrame:510,endFrame:659}]}var Z=`
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
`,Vt=16,Ht=6,Ut=330,Wt=359,Gt=360,Kt=class{engine;nodeRenderer;active;computePipeline=null;computeBindGroup=null;particleBuffer=null;particleCount=0;renderPipeline=null;renderBindGroup=null;haloPipeline=null;haloBindGroup=null;haloIndexBuffer=null;engravePipeline=null;engraveBindGroup=null;engraveBuffer=null;engraveStepCount=0;timeline=[];birthPlan=null;get engraveSteps(){return this.birthPlan?.edgeSteps??new Uint32Array}constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}upload(e){let t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;let n=this.nodeRenderer.graph;if(!n)return;this.birthPlan=Lt(n,e),this.timeline=this.birthPlan.timeline;let r=this.birthPlan.particles.length/Vt;this.particleCount=r,this.particleBuffer?.destroy(),this.particleBuffer=t.createBuffer({label:`observatory-birth-particles`,size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),this.engraveBuffer?.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:`observatory-birth-engrave`,size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){let t=e.createShaderModule({label:`observatory-birth-compute`,code:Z});this.computePipeline=e.createComputePipeline({label:`observatory-birth-compute-pipeline`,layout:`auto`,compute:{module:t,entryPoint:`birth_compute`}});let n=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:`observatory-birth-compute-bind`,layout:this.computePipeline.getBindGroupLayout(0),entries:n})}createRenderPipeline(e){let t=e.createShaderModule({label:`observatory-birth-render`,code:`
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
`});this.haloPipeline=e.createRenderPipeline({label:`observatory-birth-halo`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}});let n=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:`observatory-birth-halo-bind`,layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;let t=e.createShaderModule({label:`observatory-birth-engrave`,code:Y});this.engravePipeline=e.createRenderPipeline({label:`observatory-birth-engrave-pipeline`,layout:`auto`,vertex:{module:t,entryPoint:`vs_main`},fragment:{module:t,entryPoint:`fs_main`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.engraveBindGroup=e.createBindGroup({label:`observatory-birth-engrave-bind`,layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){let n=this.engine.params[9];if(this.active=n===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;let r=e.beginComputePass({label:`observatory-birth-compute`});r.setPipeline(this.computePipeline),r.setBindGroup(0,this.computeBindGroup),r.dispatchWorkgroups(Math.ceil(this.particleCount/64)),r.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(Ht,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=Ut&&t<=Wt&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=Gt&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){this.particleBuffer?.destroy(),this.particleBuffer=null,this.computePipeline?.destroy?.(),this.computePipeline=null,this.computeBindGroup=null,this.renderPipeline?.destroy?.(),this.renderPipeline=null,this.renderBindGroup=null,this.haloPipeline?.destroy?.(),this.haloPipeline=null,this.haloBindGroup=null,this.haloIndexBuffer?.destroy(),this.haloIndexBuffer=null,this.engravePipeline?.destroy?.(),this.engravePipeline=null,this.engraveBindGroup=null,this.engraveBuffer?.destroy(),this.engraveBuffer=null}};function qt(e){return`
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
`}var Jt=2,Yt=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;waveBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.waveBuffer?.destroy(),this.waveBuffer=e.createBuffer({label:`observatory-rescue-wave`,size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);let t=e.createShaderModule({label:`observatory-rescue-choreo`,code:qt(this.plan.consts)});this.pipeline=e.createComputePipeline({label:`observatory-rescue-choreo`,layout:`auto`,compute:{module:t,entryPoint:`rescue_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-rescue-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==Jt||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-rescue-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.waveBuffer?.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}},Xt=65535,Zt={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function Qt(e,t){return bt(e,new L({seed:t}).state.rng).data}function $t(e){let t=new Uint32Array(e.nodes.length);for(let n of e.edges)t[n.sourceIndex]++,t[n.targetIndex]++;return t}function en(e,t){let n=e.nodes.length;if(n===0)return-1;let r=$t(e),i=n=>{let i=e.nodes[n],a=new Set(i.tags.map(e=>e.toLowerCase())),o=0;(a.has(`failure`)||a.has(`guardrail`))&&(o+=3),(a.has(`confusion`)||a.has(`weak-spot`))&&(o+=2),o+=Math.min(r[n],8)/8;let s=t[n*16+0],c=t[n*16+1],l=t[n*16+2];return Math.sqrt(s*s+c*c+l*l)>=54&&(o+=.5),o},a=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r[t]>=2,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex,()=>!0];for(let e of a){let t=-1,r=-1/0;for(let a=0;a<n;a++){if(!e(a))continue;let n=i(a);n>r&&(r=n,t=a)}if(t>=0)return t}return-1}function tn(e,t){let n=e.nodes.length,r=new Uint16Array(n).fill(Xt),i=new Int32Array(n).fill(-1);if(t<0||t>=n)return{depths:r,parents:i};let a=Array.from({length:n},()=>[]);for(let t of e.edges){let e=Zt[t.type]??5;a[t.sourceIndex].push({nbr:t.targetIndex,rank:e}),a[t.targetIndex].push({nbr:t.sourceIndex,rank:e})}for(let e of a)e.sort((e,t)=>e.rank-t.rank||e.nbr-t.nbr);r[t]=0;let o=[t];for(let e=0;e<o.length;e++){let t=o[e];for(let{nbr:e}of a[t])r[e]===65535&&(r[e]=r[t]+1,o.push(e))}for(let e=0;e<n;e++)if(r[e]!==65535&&r[e]!==0){for(let{nbr:t}of a[e])if(r[t]===r[e]-1){i[e]=t;break}}return{depths:r,parents:i}}function nn(e,t,n,r){let i=new Map;for(let t of e.nodes)i.set(t.id,t.createdAt);for(let e of[3,2,1]){let a=[];for(let i=0;i<t.nodes.length;i++){if(i===t.centerIndex||i===r)continue;let o=n[i];o===65535||o<e||a.push(i)}if(a.length===0)continue;let o=a.filter(e=>t.nodes[e].retention<=.45);o.length===0&&(o=a);let s=new Map,c=1/0,l=-1/0;for(let e of o){let n=i.get(t.nodes[e].id),r=n?Date.parse(n):NaN;Number.isFinite(r)&&(s.set(e,r),r<c&&(c=r),r>l&&(l=r))}let u=e=>{let t=s.get(e);return t===void 0?0:l===c?1:(l-t)/(l-c)},d=e=>2*(1-t.nodes[e].retention)+.5*Math.min(n[e],6)/6+.5*u(e);return o.sort((e,t)=>{let r=d(e),i=d(t);return i===r?n[t]===n[e]?e-t:n[t]-n[e]:i-r}),{index:o[0],depth:n[o[0]]}}return{index:-1,depth:0}}function rn(e,t,n,r,i){let a=e[n*16+0],o=e[n*16+1],s=e[n*16+2],c=[];for(let l=0;l<t;l++){if(l===n||l===r||l===i)continue;let t=e[l*16+0]-a,u=e[l*16+1]-o,d=e[l*16+2]-s;c.push({i:l,d2:t*t+u*u+d*d})}return c.sort((e,t)=>e.d2-t.d2||e.i-t.i),c.slice(0,4).map(e=>e.i)}function an(e){return Math.min(84,Math.max(14,Math.floor(252/Math.max(1,e))))}function on(e,t){return Math.min(260+t*e,514)}function sn(e){return 138+28*e}function Q(e){return e.length>64?e.slice(0,64)+`…`:e}var cn=4;function ln(e){let t=new Uint32Array(e);return t.fill(Xt),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(e).fill(Xt),causeDepth:0,hopSlot:an(3),waveData:t,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`candidate cause found`,causeLabel:``,failureLabel:``,causeDate:``,hops:0,k:0,receipt:``},consts:{hopSlot:an(3),causeDepth:3}}}function un(e,t,n,r){if(r)return dn(e,t,r);let i=t.nodes.length;if(i===0)return ln(0);let a=Qt(t,n),o=en(t,a);if(o<0)return ln(i);let{depths:s,parents:c}=tn(t,o),l=nn(e,t,s,o);if(l.index<0){let e=ln(i);return e.failureIndex=o,e.hopDepths=s,e}let u=l.index,d=Math.max(1,l.depth),f=an(d),p=e=>on(e,f),m=rn(a,i,o,u,t.centerIndex),h=m.length,g=new Uint32Array(i);for(let e=0;e<i;e++){let t=s[e]&65535;e===o&&(t|=65536),e===u&&(t|=1<<17),g[e]=t}m.forEach((e,t)=>{g[e]|=1<<18|t<<19});let _=[];m.forEach((e,t)=>{_.push({src:o,dst:e,bf:sn(t),kind:I.probe,beatKind:`probe`})});let v=[];{let e=u;for(;e!==o&&e>=0&&c[e]>=0;)v.push(e),e=c[e]}let y=new Set(v),b=[];for(let e=0;e<i;e++){if(e===o||y.has(e))continue;let t=s[e];t===65535||t<1||t>d||c[e]<0||b.push(e)}b.sort((e,t)=>s[e]-s[t]||e-t);let ee=[...v.slice().reverse(),...b].slice(0,48);ee.sort((e,t)=>s[e]-s[t]||e-t);for(let e of ee)_.push({src:c[e],dst:e,bf:p(s[e]),kind:I.backwardCause,beatKind:`wave`});_.push({src:u,dst:o,bf:560,kind:I.backwardCause,beatKind:`arc`});let x=new Uint32Array(Math.max(1,_.length)*cn),S=[];_.forEach((e,n)=>{x[n*cn+0]=e.src,x[n*cn+1]=e.dst,x[n*cn+2]=e.bf,x[n*cn+3]=e.kind,S.push({sourceIndex:e.src,targetIndex:e.dst,beatFrame:e.bf,kind:e.kind,beatKind:e.beatKind,nodeId:t.nodes[e.dst].id,label:Q(t.nodes[e.dst].label)})});let te=Q(t.nodes[o].label),ne=Q(t.nodes[u].label),C=[],w=(e,t,n,r)=>{C.push({sourceIndex:o,targetIndex:o,beatFrame:e,kind:t,beatKind:`rescue`,nodeId:r,label:n})};w(90,1,`failure: ${te}`,t.nodes[o].id),m.forEach((e,n)=>{w(sn(n),0,`lookalike ✗ · ${Q(t.nodes[e].label)}`,t.nodes[e].id)}),w(p(1),1,`reaching backward through time`,`rescue-wave-start`),d>=2&&p(d)!==p(1)&&w(p(d),1,`scrubbing past · ${d} hops`,`rescue-wave-deep`),w(560,1,`causal arc · ${ne}`,t.nodes[u].id),w(600,1,`candidate cause found`,`rescue-verdict`);let T=e.nodes.find(e=>e.id===t.nodes[u].id)?.createdAt??``,E=T?T.slice(0,10):``;return{viable:!0,failureIndex:o,causeIndex:u,lookalikeIndices:m,hopDepths:s,causeDepth:d,hopSlot:f,waveData:g,pathData:x,pathMetas:S,spineBeats:C,verdict:{headline:`candidate cause found`,causeLabel:ne,failureLabel:te,causeDate:E,hops:d,k:h,receipt:`${d} hops back · ${E} · heuristic, no receipt · vector search: 0 for ${h}`},consts:{hopSlot:f,causeDepth:d}}}function dn(e,t,n){let r=t.nodes.length,i=t.indexById.get(n.failureId)??-1,a=n.pathIds??[];if(i<0||a.length<2||a[a.length-1]!==n.failureId||new Set(a).size!==a.length)return ln(r);let o=a.map(e=>t.indexById.get(e));if(o.some(e=>e===void 0))return ln(r);let s=o,c=s[0];if(c===i)return ln(r);let l=n.candidates.find(e=>e.memoryId===a[0]);if(!l)return ln(r);let u=new Uint16Array(r);u.fill(Xt),u[i]=0,s.forEach((e,t)=>{u[e]=s.length-1-t});let d=new Uint32Array(r);d[i]=65536,s.slice(0,-1).forEach(e=>{d[e]=u[e]}),d[c]|=1<<17;let f=s.length-1,p=an(f),m=Q(t.nodes[c].label),h=Q(t.nodes[i].label),g=e.nodes.find(e=>e.id===l.memoryId)?.createdAt?.slice(0,10)??``,_=new Uint32Array((s.length-1)*cn),v=s.slice(0,-1).map((e,n)=>{let r=s[n+1],i=260+n*p;return _[n*cn]=e,_[n*cn+1]=r,_[n*cn+2]=i,_[n*cn+3]=I.backwardCause,{sourceIndex:e,targetIndex:r,beatFrame:i,kind:I.backwardCause,beatKind:`receipt-path`,nodeId:a[n+1],label:`recorded path · ${Q(t.nodes[r].label)}`}}),y=l.sharedEntities.length?l.sharedEntities.join(`, `):`recorded entity`,b=l.similarityRank===null?`rank unavailable`:`embedding rank #${l.similarityRank}`;return{viable:!0,failureIndex:i,causeIndex:c,lookalikeIndices:[],hopDepths:u,causeDepth:f,hopSlot:p,waveData:d,pathData:_,pathMetas:v,spineBeats:[{sourceIndex:i,targetIndex:i,beatFrame:90,kind:1,beatKind:`receipt-failure`,nodeId:n.failureId,label:`recorded failure · ${h}`},{sourceIndex:i,targetIndex:i,beatFrame:260,kind:1,beatKind:`receipt-join`,nodeId:`receipt-join`,label:`shared entity · ${y}`},{sourceIndex:c,targetIndex:i,beatFrame:260+(f-1)*p,kind:I.backwardCause,beatKind:`receipt-candidate`,nodeId:l.memoryId,label:`candidate · ${m}`},{sourceIndex:c,targetIndex:c,beatFrame:600,kind:1,beatKind:`receipt-verdict`,nodeId:`receipt-verdict`,label:`candidate cause found`}],verdict:{headline:`candidate cause found`,causeLabel:m,failureLabel:h,causeDate:g,hops:f,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${y} · ${b}`},consts:{hopSlot:p,causeDepth:f}}}var fn=`
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
`,pn=3,mn=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;horizonBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.horizonBuffer?.destroy(),this.horizonBuffer=e.createBuffer({label:`observatory-forgetting-horizon`,size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);let t=e.createShaderModule({label:`observatory-forgetting-choreo`,code:fn});this.pipeline=e.createComputePipeline({label:`observatory-forgetting-choreo`,layout:`auto`,compute:{module:t,entryPoint:`forgetting_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-forgetting-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==pn||!this.pipeline||!this.bindGroup)return;let t=this.nodeRenderer.nodeCountValue;if(t===0)return;let n=e.beginComputePass({label:`observatory-forgetting-choreo`});n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.dispatchWorkgroups(Math.ceil(t/64)),n.end()}dispose(){this.horizonBuffer?.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}};function hn(e){let t=[];for(let n=0;n<e.nodes.length;n++)n!==e.centerIndex&&t.push(n);t.sort((t,n)=>e.nodes[t].retention-e.nodes[n].retention||t-n);let n=t.length;if(n===0)return[];let r=Math.min(n,Math.max(Math.min(3,n),Math.round(.25*n)));return t.slice(0,r)}function gn(e,t){let n=new Uint32Array(e.nodes.length);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>2*e.nodes[t].retention+Math.min(n[t],8)/8;return t.slice().sort((e,t)=>r(t)-r(e)||e-t).slice(0,Math.min(3,t.length))}function _n(e){return 318+60*e}var vn=4;function yn(e){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function bn(e){let t=e.nodes.length,n=hn(e);if(t<2||n.length<1)return yn(t);let r=gn(e,n),i=n.length,a=new Uint32Array(t);n.forEach((e,t)=>{let n=Math.round(255*t/Math.max(1,i-1));a[e]=n&255|256}),r.forEach((e,t)=>{a[e]|=512|t<<10});let o=new Uint32Array(Math.max(1,r.length)*vn),s=[];r.forEach((t,n)=>{let r=_n(n);o[n*vn+0]=e.centerIndex,o[n*vn+1]=t,o[n*vn+2]=r,o[n*vn+3]=I.recall,s.push({sourceIndex:e.centerIndex,targetIndex:t,beatFrame:r,kind:I.recall,beatKind:`recall`,nodeId:e.nodes[t].id,label:Q(e.nodes[t].label)})});let c=[],l=(t,n,r,i)=>{c.push({sourceIndex:e.centerIndex,targetIndex:e.centerIndex,beatFrame:t,kind:n,beatKind:`horizon`,nodeId:i,label:r})},u=new Set(r),d=n.filter(e=>!u.has(e)).slice(0,3);return d.forEach((t,n)=>{let r=Math.round(e.nodes[t].retention*100);l(132+60*n,1,`fading: ${Q(e.nodes[t].label)} · retention ${r}%`,e.nodes[t].id)}),r.forEach((t,n)=>{l(_n(n),0,`recalled: ${Q(e.nodes[t].label)}`,e.nodes[t].id)}),d.length>0&&l(540,1,`the unrecalled sink · nothing is deleted`,`horizon-sink`),l(660,0,`every memory still retrievable`,`horizon-retrievable`),{viable:!0,driftingIndices:n,rescuedIndices:r,horizonData:a,pathData:o,pathMetas:s,spineBeats:c}}var xn=`
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
`,Sn=4,Cn=class{engine;nodeRenderer;plan;pipeline=null;bindGroup=null;fireBuffer=null;constructor(e){this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){let e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;this.fireBuffer?.destroy(),this.fireBuffer=e.createBuffer({label:`observatory-firewall-fire`,size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);let t=e.createShaderModule({label:`observatory-firewall-choreo`,code:xn});this.pipeline=e.createComputePipeline({label:`observatory-firewall-choreo`,layout:`auto`,compute:{module:t,entryPoint:`firewall_choreo`}}),this.bindGroup=e.createBindGroup({label:`observatory-firewall-bind`,layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){let t=this.engine.params[9]===Sn,n=this.engine.params[12]===1;if(!t&&!n||!this.pipeline||!this.bindGroup)return;let r=this.nodeRenderer.nodeCountValue;if(r===0)return;let i=e.beginComputePass({label:`observatory-firewall-choreo`});i.setPipeline(this.pipeline),i.setBindGroup(0,this.bindGroup),i.dispatchWorkgroups(Math.ceil(r/64)),i.end()}dispose(){this.fireBuffer?.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}},wn=[`failure`,`guardrail`,`confusion`];Math.PI*2;function Tn(e){let t=e.nodes.length;if(t===0)return-1;let n=new Uint32Array(t);for(let t of e.edges)n[t.sourceIndex]++,n[t.targetIndex]++;let r=t=>e.nodes[t].tags.some(e=>wn.includes(e.toLowerCase())),i=[t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&r(t),t=>t!==e.centerIndex&&!e.nodes[t].suppressed&&n[t]<=1,t=>t!==e.centerIndex&&!e.nodes[t].suppressed,t=>t!==e.centerIndex];for(let n of i){let r=-1;for(let i=0;i<t;i++)n(i)&&(r<0||e.nodes[i].retention<e.nodes[r].retention)&&(r=i);if(r>=0)return r}return-1}function En(e,t,n){let r=e[n*16+0],i=e[n*16+1],a=e[n*16+2],o=Array(t),s=0;for(let n=0;n<t;n++){let t=e[n*16+0]-r,c=e[n*16+1]-i,l=e[n*16+2]-a,u=Math.sqrt(t*t+c*c+l*l);o[n]=u,u>s&&(s=u)}s<1e-6&&(s=1);let c=Array(t);for(let e=0;e<t;e++)c[e]=Math.min(255,Math.max(0,Math.round(144*o[e]/s)));return c[n]=0,c}function Dn(e,t){let n=new Set;for(let r of e.edges)r.sourceIndex===t&&r.targetIndex!==t&&n.add(r.targetIndex),r.targetIndex===t&&r.sourceIndex!==t&&n.add(r.sourceIndex);return Array.from(n).sort((e,t)=>e-t).slice(0,6)}function On(e){return 345+21*e}var kn=4;function An(e){return jn(e)}function jn(e){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(e),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:`threat quarantined`,intruderLabel:``,receipt:`memory held in review · Memory PR opened`}}}function Mn(e,t){return Pn(e,t,Tn(e))}function Nn(e,t,n){return n<0||n>=e.nodes.length?jn(e.nodes.length):Pn(e,t,n)}function Pn(e,t,n){let r=e.nodes.length;if(r===0||n<0)return jn(r);let i=En(Qt(e,t),r,n),a=Dn(e,n),o=new Uint32Array(r);for(let e=0;e<r;e++)o[e]=i[e]&255;o[n]=256,a.forEach((e,t)=>{o[e]|=512|t<<10});let s=new Uint32Array(Math.max(1,a.length)*kn),c=[];a.forEach((t,r)=>{let i=On(r);s[r*kn+0]=n,s[r*kn+1]=t,s[r*kn+2]=i,s[r*kn+3]=I.probe,c.push({sourceIndex:n,targetIndex:t,beatFrame:i,kind:I.probe,beatKind:`sever`,nodeId:e.nodes[t].id,label:Q(e.nodes[t].label)})});let l=Q(e.nodes[n].label),u=[],d=(e,t,r)=>{u.push({sourceIndex:n,targetIndex:n,beatFrame:e,kind:1,beatKind:`firewall`,nodeId:r,label:t})};return d(90,`intrusion · ${l}`,e.nodes[n].id),d(150,`immune response · shockwave`,`firewall-shock`),d(330,`membrane forming`,`firewall-membrane`),a.forEach((t,n)=>{d(On(n),`edge severed ✗ · ${Q(e.nodes[t].label)}`,e.nodes[t].id)}),d(480,`threat quarantined`,`firewall-verdict`),{viable:!0,intruderIndex:n,severedNeighborIndices:a,shockDelays:i,fireData:o,pathData:s,pathMetas:c,spineBeats:u,verdict:{headline:`threat quarantined`,intruderLabel:l,receipt:`memory held in review · Memory PR opened`}}}var Fn=.1542;function In(e=Fn){return .9**(-1/e)-1}function Ln(e,t,n=Fn){if(!(e>0))return 0;if(!(t>0))return 1;let r=(1+In(n)*t/e)**+-n;return r<0?0:r>1?1:r}var Rn=864e5;function zn(e,t,n=0){if(!e)return n>0?n:0;let r=Date.parse(e);if(!Number.isFinite(r))return n>0?n:0;let i=(t-r)/Rn;return Math.max(0,i)+Math.max(0,n)}function Bn(e,t,n,r,i=Fn){if(n){let e=Date.parse(n);if(Number.isFinite(e)&&r<e)return 0}if(e===void 0||!Number.isFinite(e)||!t)return 1;let a=Date.parse(t);return Number.isFinite(a)?Math.max(.001,Ln(e,(r-a)/Rn,i)):1}function Vn(e,t,n,r=0,i=Fn){return e===void 0||!Number.isFinite(e)?1:Ln(e,zn(t,n,r),i)}var Hn={[F.firewall]:620,[F.dreamStorm]:360,[F.causalRecall]:260,[F.birth]:180},Un=class{engine;renderer;graph;response;seed;projectionDays;chronoOffsetDays;onApply;onFirewall;firewall=null;liveEdges=[];liveEdgeKeys=new Set;edgesDirty=!1;indexById;active=null;dreamOpen=!1;retention;hasLiveDecay=!1;eventsSeen=0;lastDecayFrame=-1e3;constructor(e){this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;let t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let n=0;n<t;n++){let t=e.graph.nodes[n];this.retention[n]=t.retention,t.stability!==void 0&&t.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(let e of this.liveEdges)this.liveEdgeKeys.add(Wn(e.sourceIndex,e.targetIndex));this.lastAppliedMs=0;let n=this.engine.params;n[P.liveKind]=F.none,n[P.liveFrame]=0,n[P.liveEnergy]=0,n[P.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}lastAppliedMs=0;seeded=!1;seedWatermark(e){let t=0;for(let n of e){let e=Gn(n);e>t&&(t=e)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,n){if(this.active!==null)return!1;let r=this.indexById.get(e);if(r===void 0||(this.retention[r]??0)<5e-4)return!1;let i=t.filter(t=>t!==e&&this.indexById.has(t));return this.arm({kind:F.causalRecall,startFrame:n,targetId:e,relatedIds:i,pairs:[],scalar:i.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let n=e.length-1;n>=0;n--){let r=e[n],i=Gn(r);i>this.lastAppliedMs&&(this.decodeAndArm(r,this.engine.totalFrames),i>t&&(t=i))}this.lastAppliedMs=t}decodeAndArm(e,t){let n=e.data??{};switch(e.type){case`MemorySuppressed`:{let e=Kn(n.id);if(!e||!this.indexById.has(e))return;this.arm({kind:F.firewall,startFrame:t,targetId:e,relatedIds:this.neighborsOf(e),pairs:[],scalar:qn(n.estimated_cascade)});break}case`DeepReferenceCompleted`:{let e=Yn(n.contradiction_pairs).filter(([e,t])=>this.indexById.has(e)&&this.indexById.has(t));if(e.length>0){let n=e[0][0];this.arm({kind:F.firewall,startFrame:t,targetId:n,relatedIds:e.flatMap(e=>e).filter(e=>e!==n),pairs:e,scalar:e.length});return}let r=Kn(n.primary_id),i=Jn(n.supporting_ids).filter(e=>this.indexById.has(e));r&&this.indexById.has(r)&&this.arm({kind:F.causalRecall,startFrame:t,targetId:r,relatedIds:i,pairs:[],scalar:qn(n.confidence)});break}case`BackfillFired`:case`CausalReceipt`:{let e=Jn(n.path_ids??n.causal_path),r=Kn(n.failure_id??n.target_id??n.effect_id)||e.at(-1)||e[0];r&&this.indexById.has(r)&&this.arm({kind:F.causalRecall,startFrame:t,targetId:r,relatedIds:e.filter(e=>e!==r),exactPath:e,pairs:[],scalar:e.length});break}case`DreamStarted`:this.dreamOpen=!0,this.arm({kind:F.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:qn(n.memory_count)});break;case`DreamCompleted`:{this.dreamOpen=!1;let e=qn(n.connections_found);this.active&&this.active.kind===F.dreamStorm?this.active.scalar=Math.max(this.active.scalar,e):this.arm({kind:F.dreamStorm,startFrame:t,targetId:``,relatedIds:[],pairs:[],scalar:e});break}case`ConnectionDiscovered`:{let e=this.indexById.get(Kn(n.source_id)),t=this.indexById.get(Kn(n.target_id));if(e===void 0||t===void 0||e===t)break;let r=Wn(e,t);if(this.liveEdgeKeys.has(r))break;this.liveEdgeKeys.add(r),this.liveEdges.push({sourceIndex:e,targetIndex:t,weight:qn(n.weight)||.5,type:Kn(n.connection_type)||`semantic`}),this.edgesDirty=!0,this.dreamOpen&&this.active?.kind===F.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){if(this.active=e,this.eventsSeen++,e.kind===F.firewall){let t=this.indexById.get(e.targetId);if(t===void 0)return;let n=Nn(this.graph,this.seed,t);if(!n.viable)return;this.firewall||=new Cn({engine:this.engine,nodeRenderer:this.renderer,plan:An(this.graph.nodes.length)}),this.firewall.rearm(n),this.onFirewall?.({intruderLabel:n.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===F.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){let t=e.exactPath;if(t.some(e=>!this.indexById.has(e)))return;let n=new Uint32Array(Math.max(1,t.length-1)*4),r=[];for(let i=0;i<t.length-1;i++){let a=this.indexById.get(t[i]),o=this.indexById.get(t[i+1]),s=e.startFrame+24+i*42;n[i*4]=a,n[i*4+1]=o,n[i*4+2]=s,n[i*4+3]=I.backwardCause,r.push({sourceIndex:a,targetIndex:o,beatFrame:s,kind:I.backwardCause,beatKind:`receipt-path`,nodeId:t[i+1],label:`receipt-backed candidate path`})}this.renderer.setPathSteps(n,r);return}let t=Tt(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});t.steps.length>0&&this.renderer.setPathSteps(t.data,t.steps)}}neighborsOf(e){let t=this.indexById.get(e);if(t===void 0)return[];let n=[];for(let e of this.graph.edges)if(e.sourceIndex===t?n.push(this.graph.nodes[e.targetIndex].id):e.targetIndex===t&&n.push(this.graph.nodes[e.sourceIndex].id),n.length>=12)break;return n}drain(e){let t=this.engine.params;this.edgesDirty&&=(this.renderer.setEdges(this.liveEdges),!1);let n=this.projectionDays(),r=this.chronoOffsetDays();if(t[P.projectionDays]=Math.max(0,n),(this.hasLiveDecay||r!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||n!==this.lastProj||r!==this.lastChrono)&&(this.recomputeDecay(n,r),this.lastDecayFrame=e,this.lastProj=n,this.lastChrono=r),this.active){let n=Hn[this.active.kind]??300,r=e-this.active.startFrame;r>n+140?(this.active=null,t[P.liveKind]=F.none,t[P.liveEnergy]=0):(t[P.liveKind]=this.active.kind,t[P.liveFrame]=Math.max(0,r),t[P.liveEnergy]=this.energyEnvelope(this.active,r,!1))}else t[P.liveKind]=F.none,t[P.liveEnergy]=0;this.onApply?.({simFrame:e,activeKind:t[P.liveKind],eventsSeen:this.eventsSeen})}debugState(){let e=this.engine.params;return{activeKind:e[P.liveKind],liveEnergy:e[P.liveEnergy],liveFrame:e[P.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}lastProj=-1;lastChrono=0;energyEnvelope(e,t,n){if(t<0)return 0;let r=Hn[e.kind]??300;if(e.kind===F.dreamStorm){let n=Math.min(1,t/45),i=1-Math.max(0,(t-(r-90))/90),a=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,n*Math.min(1,i)*a)}let i=Math.min(1,t/24),a=1-Math.max(0,(t-r)/140);return Math.max(0,i*Math.min(1,a))}recomputeDecay(e,t=0){let n=this.engine.wallNowMs,r=this.graph.nodes;if(t!==0){let i=n+(t+Math.max(0,e))*Rn;for(let e=0;e<r.length;e++){let t=r[e];this.retention[e]=t.stability!==void 0||t.createdAt?Bn(t.stability,t.lastAccessed,t.createdAt,i):Math.max(.001,t.retention)}}else for(let t=0;t<r.length;t++){let i=r[t];this.retention[t]=i.stability!==void 0&&i.lastAccessed?Vn(i.stability,i.lastAccessed,n,e):Math.max(.001,i.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){let e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}};function Wn(e,t){return e<t?`${e}-${t}`:`${t}-${e}`}function Gn(e){let t=e.data?.timestamp;if(typeof t!=`string`)return 0;let n=Date.parse(t);return Number.isFinite(n)?n:0}function Kn(e){return typeof e==`string`?e:``}function qn(e){return typeof e==`number`&&Number.isFinite(e)?e:0}function Jn(e){return Array.isArray(e)?e.filter(e=>typeof e==`string`):[]}function Yn(e){if(!Array.isArray(e))return[];let t=[];for(let n of e)Array.isArray(n)&&n.length>=2&&typeof n[0]==`string`&&typeof n[1]==`string`&&t.push([n[0],n[1]]);return t}var Xn=512,Zn=4,Qn=`
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
	// Where the rail stands (NDC): left, right, y; the host places it beside its words.
	place: vec4f,
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

fn rail_x(t: f32) -> f32 { return mix(shuttle.place.x, shuttle.place.y, clamp(t, 0.0, 1.0)); }

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
	out.clip = vec4f(mix(shuttle.place.x, shuttle.place.y, q.x * 0.5 + 0.5), shuttle.place.z + q.y * max(0.022, py * 5.0), 0.0, 1.0);
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
	out.clip = vec4f(rail_x(d.x) + q.x * width, shuttle.place.z + direction * (0.008 + height * (q.y * 0.5 + 0.5)), 0.0, 1.0);
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
	out.clip = vec4f(rail_x(shuttle.scrub) + q.x * max(0.021, px * 7.0), shuttle.place.z + q.y * height, 0.0, 1.0);
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
`;function $n(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function er(e){if(!e)return null;let t=Date.parse(e);return Number.isFinite(t)?t:null}var tr={left:-.835,right:.835,y:-.685,show:!0},nr=class{engine;resources=null;bindLayout=null;railPipeline=null;dwellPipeline=null;headPipeline=null;dwellCount=0;minMs=0;maxMs=0;state={scrub:1,days:0,density:0,active:0};place=tr;constructor(e,t,n=tr){this.engine=e,this.place=n,this.upload(t)}setPlace(e){this.place=e,this.writeState(),this.engine.requestRender()}setTimeline(e,t=!1){let n=this.engine.wallNowMs,r=Math.max(1,this.maxMs-this.minMs);this.state.scrub=$n((n+e*864e5-this.minMs)/r),this.state.days=Number.isFinite(e)?e:0,this.state.active=+!!t,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){this.place.show&&this.resources&&this.railPipeline&&this.dwellPipeline&&this.headPipeline&&(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){this.resources?.dwellBuffer.destroy(),this.resources?.stateBuffer.destroy(),this.resources=null}upload(e){let t=e.flatMap(e=>[er(e.createdAt),er(e.lastAccessed)]).filter(e=>e!==null),n=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):n-864e5,this.maxMs=Math.max(n+31536e6,this.minMs+864e5);let r=this.maxMs-this.minMs,i=[];for(let t of e){let e=er(t.createdAt),n=er(t.lastAccessed),r=$n(t.retention);e!==null&&i.push({at:e,kind:0,retention:r,suppressed:+!!t.suppressed}),n!==null&&n!==e&&i.push({at:n,kind:1,retention:r,suppressed:+!!t.suppressed})}i.sort((e,t)=>e.at-t.at);let a=Math.max(1,Math.ceil(i.length/Xn)),o=i.filter((e,t)=>t%a===0).slice(0,Xn);this.dwellCount=o.length,this.state={scrub:$n((n-this.minMs)/r),days:0,density:$n(o.length/96),active:0};let s=this.engine.gpuDevice;if(!s||!this.engine.paramsBuffer||(this.ensurePipelines(s),this.ensureResources(s),!this.resources))return;let c=new Float32Array(Xn*Zn);o.forEach((e,t)=>{c.set([$n((e.at-this.minMs)/r),e.kind,e.retention,e.suppressed],t*Zn)}),s.queue.writeBuffer(this.resources.dwellBuffer,0,c),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`fossil-light-chrono-shuttle-wgsl`,code:Qn});this.bindLayout=e.createBindGroupLayout({label:`fossil-light-chrono-shuttle-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let n=e.createPipelineLayout({label:`fossil-light-chrono-shuttle-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`src-alpha`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}},i=(i,a,o)=>e.createRenderPipeline({label:i,layout:n,vertex:{module:t,entryPoint:a},fragment:{module:t,entryPoint:o,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`}});this.railPipeline=i(`fossil-light-chrono-rail`,`vs_rail`,`fs_rail`),this.dwellPipeline=i(`fossil-light-chrono-dwells`,`vs_dwell`,`fs_dwell`),this.headPipeline=i(`fossil-light-chrono-head`,`vs_head`,`fs_head`)}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;let t=e.createBuffer({label:`fossil-light-chrono-dwell-events`,size:Xn*Zn*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:`fossil-light-chrono-state`,size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:n,bindGroup:e.createBindGroup({label:`fossil-light-chrono-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}}]})}}writeState(){let e=this.engine.gpuDevice;if(!e||!this.resources)return;let t=this.place;e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active,t.left,t.right,t.y,+!!t.show]))}},rr=64,ir=32,ar=4,or=256,sr=`rgba8unorm`,cr=96e3,lr=5,ur=`
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

const MAX_EMITTERS = ${rr}u;

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
`;function dr(e,t){return Number.isFinite(e)?e:t}var fr=class{engine;renderer;sourceIndices;resources=null;projectionPipeline=null;seedPipeline=null;transportPipeline=null;compositePipeline=null;projectionLayout=null;seedLayout=null;transportLayout=null;compositeLayout=null;emitterCount;active=!1;dirty=!0;lastComputedFrame=-5;disposed=!1;disabledReason=null;exposure=.42;configBytes=new ArrayBuffer(ir);configUints=new Uint32Array(this.configBytes);configFloats=new Float32Array(this.configBytes);constructor(e,t,n){this.engine=e,this.renderer=t;let r=[...new Set([...n].filter(e=>Number.isFinite(e)&&e>=0))].sort((e,t)=>e-t).slice(0,rr);this.sourceIndices=new Uint32Array(r),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?`half-res-transport`:`disabled`}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,dr(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;let n=this.engine.gpuDevice;if(!n||!this.engine.paramsBuffer)return;let r=this.renderer.getFossilLightSources();if(!r)return;let i=this.fieldDimensions();if(i===null)return;let a=t-this.lastComputedFrame;if(!(this.active||this.dirty||a<0||a>=lr))return;try{this.ensurePipelines(n),this.ensureResources(n,i.width,i.height,r)}catch{this.disable(`GPU light field unavailable on this adapter`);return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(n,0,this.resources.width,this.resources.height,0);let o=Math.ceil(this.resources.width/8),s=Math.ceil(this.resources.height/8),c=e.beginComputePass({label:`fossil-light-half-res-transport`});c.setPipeline(this.projectionPipeline),c.setBindGroup(3,this.resources.projectionBindGroup,[0]),c.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),c.setPipeline(this.seedPipeline),c.setBindGroup(0,this.resources.seedBindGroup,[0]),c.dispatchWorkgroups(o,s);for(let[e,t,r]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(n,e,this.resources.width,this.resources.height,t),c.setPipeline(this.transportPipeline),c.setBindGroup(1,r,[e*or]),c.dispatchWorkgroups(o,s);c.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason===null&&this.resources&&this.compositePipeline&&this.emitterCount!==0&&(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*or]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){let e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;let n=e*.5*(t*.5),r=.5*Math.min(1,Math.sqrt(cr/Math.max(1,n)));return{width:Math.max(1,Math.floor(e*r)),height:Math.max(1,Math.floor(t*r))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;let t=e.createShaderModule({label:`fossil-light-radiance-cascade-wgsl`,code:ur}),n=e.createBindGroupLayout({label:`fossil-light-empty-layout`,entries:[]});this.projectionLayout=e.createBindGroupLayout({label:`fossil-light-source-projection-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:ir}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:`storage`}}]}),this.seedLayout=e.createBindGroupLayout({label:`fossil-light-seed-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:ir}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:sr}}]}),this.transportLayout=e.createBindGroupLayout({label:`fossil-light-transport-layout`,entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:ir}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:`write-only`,format:sr}}]}),this.compositeLayout=e.createBindGroupLayout({label:`fossil-light-composite-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`,hasDynamicOffset:!0,minBindingSize:ir}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}}]}),this.seedPipeline=e.createComputePipeline({label:`fossil-light-seed`,layout:e.createPipelineLayout({label:`fossil-light-seed-pipeline-layout`,bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:`cs_seed`}}),this.projectionPipeline=e.createComputePipeline({label:`fossil-light-source-projection`,layout:e.createPipelineLayout({label:`fossil-light-source-projection-pipeline-layout`,bindGroupLayouts:[n,n,n,this.projectionLayout]}),compute:{module:t,entryPoint:`cs_project_sources`}}),this.transportPipeline=e.createComputePipeline({label:`fossil-light-transport`,layout:e.createPipelineLayout({label:`fossil-light-transport-pipeline-layout`,bindGroupLayouts:[n,this.transportLayout]}),compute:{module:t,entryPoint:`cs_transport`}}),this.compositePipeline=e.createRenderPipeline({label:`fossil-light-composite`,layout:e.createPipelineLayout({label:`fossil-light-composite-pipeline-layout`,bindGroupLayouts:[n,n,this.compositeLayout]}),vertex:{module:t,entryPoint:`vs_composite`},fragment:{module:t,entryPoint:`fs_composite`,targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:`src-alpha`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]}})}ensureResources(e,t,n,r){if(this.resources?.width===t&&this.resources.height===n&&this.resources.nodeBuffer===r.nodeBuffer&&this.resources.cameraBuffer===r.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;let i=e.createBuffer({label:`fossil-light-projected-memory-emitters`,size:768*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),a=e.createBuffer({label:`fossil-light-source-indices`,size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(a,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);let o=e.createBuffer({label:`fossil-light-cascade-config`,size:or*ar,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),s=r=>e.createTexture({label:r,size:[t,n],format:sr,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),c=s(`fossil-light-field-a`),l=s(`fossil-light-field-b`),u=c.createView(),d=l.createView();this.resources={width:t,height:n,emitterBuffer:i,sourceIndexBuffer:a,configBuffer:o,fieldA:c,fieldB:l,seedBindGroup:e.createBindGroup({label:`fossil-light-seed-bind-group`,layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:ir}},{binding:1,resource:{buffer:i}},{binding:2,resource:u}]}),propagateABindGroup:e.createBindGroup({label:`fossil-light-transport-a-to-b`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ir}},{binding:1,resource:u},{binding:2,resource:d}]}),propagateBBindGroup:e.createBindGroup({label:`fossil-light-transport-b-to-a`,layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ir}},{binding:1,resource:d},{binding:2,resource:u}]}),projectionBindGroup:e.createBindGroup({label:`fossil-light-source-projection-bind-group`,layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:ir}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:r.nodeBuffer}},{binding:3,resource:{buffer:r.cameraBuffer}},{binding:4,resource:{buffer:i}}]}),compositeBindGroup:e.createBindGroup({label:`fossil-light-composite-bind-group`,layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:ir}},{binding:1,resource:d}]}),nodeBuffer:r.nodeBuffer,cameraBuffer:r.cameraBuffer},this.dirty=!0}writeConfig(e,t,n,r,i){this.resources&&(this.configUints[0]=n,this.configUints[1]=r,this.configUints[2]=this.emitterCount,this.configUints[3]=i,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*or,this.configBytes))}destroyResources(){this.resources?.emitterBuffer.destroy(),this.resources?.sourceIndexBuffer.destroy(),this.resources?.configBuffer.destroy(),this.resources?.fieldA.destroy(),this.resources?.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}},pr=r(`<a class="flex items-baseline gap-2 font-mono text-[11px] no-underline hover:underline"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></a>`),mr=r(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					pl-3 border-l border-[#A8FF5E]/20 [text-shadow:0_0_18px_rgba(2,3,7,0.95),0_1px_2px_rgba(2,3,7,0.9)]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),hr=r(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 text-center enter
					[text-shadow:0_0_18px_rgba(2,3,7,0.95),0_1px_2px_rgba(2,3,7,0.9)]"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),gr=r(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 py-0.5
					border-b border-[#22C7DE]/30 font-mono text-[11px] tracking-wide text-[#22C7DE]/80
					hover:text-[#22C7DE] hover:border-[#22C7DE]/60 transition-colors"> </button>`),_r=r(`<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>`),vr=r(`<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>`),yr=r(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),br=r(`<button> </button>`),xr=r(`<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>`),Sr=r(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>`),Cr=r(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>`),wr=r(`<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>`),Tr=r(`<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>`),Er=r(`<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div> <!> <!>`,1);function Dr(e,r){de(r,!0);let o=()=>fe(_e,`$eventFeed`,p),[p,ne]=te(),C=b(r,`seed`,3,`vestige-observatory-v1`),w=b(r,`freezeFrame`,3,null),E=b(r,`capture`,3,!1),ie=b(r,`showSwitcher`,3,!0),ae=b(r,`embedded`,3,!1),le=b(r,`chrome`,3,`none`),he=b(r,`maxDpr`,3,2),ge=b(r,`focusIds`,19,()=>[]),ve=b(r,`live`,3,!1),ye=b(r,`graphOverride`,3,null),j=v(0),M=v(!1),N=v(0),we=null,Te=f(()=>Math.max(0,d(j))),Ee=f(()=>Math.min(0,d(j))),De=f(()=>d(j)===0?`now`:d(j)>0?`+${Math.round(d(j))}d`:new Date(Date.now()+d(j)*864e5).toLocaleDateString(void 0,{month:`short`,day:`numeric`})),P=null,F=null,I=null,L=v(!1);t(()=>{let e=d(Me);d(L)&&F?.setPlace(e)});let je=v(!1),Me=f(()=>r.rail??tr),Ne=!1,Pe=0,z=0,Fe=0,Ie=!1;function Le(e){let t=d($e)?.getBoundingClientRect();if(!t||t.width===0)return d(j);let n=(e-t.left)/t.width*2-1,r=Math.max(0,Math.min(1,(n-d(Me).left)/Math.max(.01,d(Me).right-d(Me).left)));return d(N)+r*(365-d(N))}function Re(e){let t=d($e)?.getBoundingClientRect();if(!t||t.height===0||!d(Me).show)return!1;let n=(e.clientY-t.top)/t.height,r=(e.clientX-t.left)/t.width*2-1,i=(1-d(Me).y)/2;return n>i-.075&&n<i+.075&&r>d(Me).left-.04&&r<d(Me).right+.04}function ze(){Pe&&cancelAnimationFrame(Pe),Pe=0}function B(e){d(L)&&!E()&&Re(e)&&(ze(),Ne=!0,y(M,!0),z=0,Fe=performance.now(),y(j,Le(e.clientX),!0),e.currentTarget.setPointerCapture?.(e.pointerId),e.preventDefault())}function Be(e){if(!Ne)return;let t=performance.now(),n=Le(e.clientX),r=Math.max(1,t-Fe);z=z*.6+(n-d(j))/r*16*.4,Fe=t,y(j,n,!0)}function Ve(e){if(!Ne)return;Ne=!1,Ie=!0,e.currentTarget.releasePointerCapture?.(e.pointerId);let t=()=>{Pe=0,z*=.94;let e=d(j)+z;e<=d(N)&&(e=d(N),z=0),e>=365&&(e=365,z=0);let n=d(j)<0&&z>0||d(j)>0&&z<0;Math.abs(e)<1&&n&&(e=0,z=0),y(j,e,!0),Math.abs(z)>.02?Pe=requestAnimationFrame(t):y(M,!1)};Math.abs(z)>.05?Pe=requestAnimationFrame(t):y(M,!1)}function He(){Ne=!1,y(M,!1),ze()}let V=v(!1),Ue=v(!1);function H(){if(typeof window>`u`)return;let e=window.matchMedia(`(prefers-reduced-motion: reduce)`);e.matches&&!d(Ue)&&y(V,!0);let t=e=>{d(Ue)||y(V,e.matches,!0)};return e.addEventListener(`change`,t),()=>e.removeEventListener(`change`,t)}function We(){y(Ue,!0),y(V,!d(V))}t(()=>{d(J)?.setPaused(d(V))});let Ke=v(``),Ze=v(0),Qe=f(()=>d(Ke)!==``&&d(Ze)>0),$e=v(null),U=new gt,tt=v(null),nt=0,W=v(null);function it(e){let t=e?Date.parse(e):NaN;if(!Number.isFinite(t))return``;let n=new Date(t);return`${n.toLocaleDateString([],{month:`short`,day:`numeric`})}, ${n.toLocaleTimeString([],{hour:`2-digit`,minute:`2-digit`})}`}function ot(){y(W,null),Y?.setHovered(-1),Ce.set(!1)}t(()=>()=>Ce.set(!1));async function st(e){if(Ie){Ie=!1;return}if(!Y||!d($e))return;let t=d($e).getBoundingClientRect();if(t.width===0||t.height===0)return;let n=(e.clientX-t.left)/t.width*2-1,i=-((e.clientY-t.top)/t.height*2-1),a=await Y.pickAt(n,i);a&&(ot(),y(tt,{kind:`memory`,id:a.id,label:`Field cell`},!0),r.onpick?.(a.id,{x:n,y:i}))}function ct(e){B(e),!(Ne||E())&&(U.enabled=!E(),U.onPointerDown(e))}function lt(e){if(Be(e),Ne||E())return;U.onPointerMove(e)&&(Ie=!0,Y?.setCameraRig(U.state));let t=performance.now();if(t-nt<60||!Y||!d($e))return;nt=t;let n=d($e).getBoundingClientRect();if(n.width===0)return;let i=(e.clientX-n.left)/n.width*2-1,a=-((e.clientY-n.top)/n.height*2-1),o={x:e.clientX,y:e.clientY};Y.pickAt(i,a).then(e=>{Y?.setHovered(e?.index??-1),d($e)&&(d($e).style.cursor=e?`pointer`:`grab`),Ce.set(!!e&&!!r.onpick);let t=e?d(K)?.nodes.find(t=>t.id===e.id):void 0;if(!e||!t||E()){y(W,null);return}if(d(W)?.id===e.id)return;let[n,i,a]=be(t.type),s=it(t.createdAt);y(W,{id:e.id,x:o.x,y:o.y,left:o.x>innerWidth*.62,tone:`rgb(${Math.round(n*255)} ${Math.round(i*255)} ${Math.round(a*255)})`,kicker:`${t.type}${s?` · ${s}`:``}${typeof t.retention==`number`?` · ${Math.round(t.retention*100)}% retained`:``}`,text:xe(t.label||e.id,160)},!0)})}function ut(e){Ve(e),U.onPointerUp(e)}function dt(){He(),U.onPointerUp({pointerId:-1})}function ft(e){E()||Re(e)||U.onWheel(e)&&Y?.setCameraRig(U.state)}let pt={"recall-path":`RECALL`,"engram-birth":`BIRTH`,"salience-rescue":`RESCUE`,"forgetting-horizon":`HORIZON`,firewall:`FIREWALL`};function mt(){return Y?.graph?new Uint32Array(Y.graph.nodes.map(e=>({index:e.index,id:e.id,retention:e.retention})).sort((e,t)=>t.retention-e.retention||e.id.localeCompare(t.id)).slice(0,64).map(e=>e.index).sort((e,t)=>e-t)):new Uint32Array}let ht=v(!E());function G(e){let t=e.target;t?.isContentEditable||t?.tagName===`INPUT`||t?.tagName===`TEXTAREA`||t?.tagName===`SELECT`||((e.key===`h`||e.key===`H`)&&y(ht,!d(ht)),e.key===`Escape`&&r.onexit&&r.onexit(),(e.key===` `||e.key.toLowerCase()===`p`)&&!E()&&(e.preventDefault(),We()))}let K=v(null),q=v(!0),_t=v(``),vt=v(0),yt=v(0),bt=v(0),xt=v(0),St=v(``),J=v(null),Y=null,Ct=null,wt=null,Tt=v(null),Et=null,Dt=null,Ot=v(null),At=!1,X=v(c([]));async function jt(){y(q,!0),y(_t,``);try{if(ye()){y(K,ye()),y(bt,ye().nodeCount,!0),y(xt,ye().edgeCount,!0),y(St,ye().center_id,!0);return}let e=new Set(ge().filter(Boolean)),t=e.size?await(async()=>{let t=await Promise.all([...e].map(e=>A.graph({center_id:e,max_nodes:200,depth:3}))),n=[...new Map(t.flatMap(e=>e.nodes).map(e=>[e.id,e])).values()].filter(t=>e.has(t.id)),r=new Set(n.map(e=>e.id)),i=[...new Map(t.flatMap(e=>e.edges).map(e=>[`${e.source}:${e.target}`,e])).values()].filter(e=>r.has(e.source)&&r.has(e.target));return{...t[0],nodes:n,edges:i,center_id:n[0]?.id??t[0]?.center_id??``,nodeCount:n.length,edgeCount:i.length}})():await Ge();y(K,t,!0),y(bt,t.nodeCount,!0),y(xt,t.edgeCount,!0),y(St,t.center_id,!0)}catch(e){let t=e instanceof Error?e.message:`Failed to load graph data`;/\b404\b/.test(t)?(y(K,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:``},!0),y(bt,0),y(xt,0),y(St,``)):y(_t,t,!0)}finally{y(q,!1)}}let Mt=null,Nt=v(c([])),Pt=v(`recalls`);t(()=>{let e=d(Nt),t=d(Pt);s(()=>r.ontop?.(e,t))});function Ft(e,t){y(vt,e,!0),y(yt,t,!0),Mt&&!d(M)&&Mt.tick(e)}async function It(){if(!P||!Y?.graph)return;let e=Y.graph,t=t=>e.indexById.has(t),n=t=>e.nodes[e.indexById.get(t)??-1]?.label??t.slice(0,8),r=[];try{r=(await A.receipts.list(60))?.receipts??[]}catch{}let i=qe(r,t);i.length===0&&(i=Je(e.nodes,12)),i.length>0&&(Mt=new Xe(P,{intervalFrames:240}),Mt.setItems(i));let a=Ye(r,t,3);a.length>0?(y(Pt,`recalls`),y(Nt,a.map(e=>({...e,label:n(e.id)})),!0)):(y(Pt,`retention`),y(Nt,[...e.nodes].filter(e=>(e.label??``).trim().length>0).sort((e,t)=>t.retention-e.retention).slice(0,3).map(e=>({id:e.id,recalls:Math.round(e.retention*100),label:e.label||e.id.slice(0,8)})),!0))}function Lt(e){At=!1,Y?.dispose(),y(J,e,!0),Y=new kt(e),U.enabled=!E(),E()&&U.reset(),Y.setCameraRig(U.state),r.fieldFrame&&Y.setFrame(r.fieldFrame),r.onready?.(e)}t(()=>{let e=r.fieldFrame;e&&Y?.setFrame(e)}),t(()=>{if(d(J)&&Y&&d(K)&&!At){At=!0;let e=r.demo===`engram-birth`,t=r.demo===`salience-rescue`,n=r.demo===`forgetting-horizon`,i=r.demo===`firewall`;if(Y.upload(d(K),C(),{recallPath:!e&&!t&&!n&&!i}),e){Ct=new Kt({engine:d(J),nodeRenderer:Y,seed:C()}),Ct.upload(C());let e=Ct.engraveSteps,t=[];for(let n=0;n<e.length/4;n++)t.push({sourceIndex:e[n*4],targetIndex:e[n*4+1],beatFrame:e[n*4+2],kind:e[n*4+3],beatKind:`engrave`,nodeId:`engrave-${n}`,label:`edge engraved`});Y.setPathSteps(e,t),y(X,Ct.timeline.map((e,t)=>({sourceIndex:0,targetIndex:0,beatFrame:e.startFrame,kind:0,beatKind:`birth`,nodeId:`birth-${t}`,label:e.label})),!0)}else if(t){let e=un(d(K),Y.graph,C(),r.backfillEvidence);y(Tt,e,!0),e.viable&&(wt=new Yt({engine:d(J),nodeRenderer:Y,plan:e}),wt.upload(),Y.setPathSteps(e.pathData,e.pathMetas)),y(X,e.spineBeats,!0)}else if(n){let e=bn(Y.graph);e.viable&&(Et=new mn({engine:d(J),nodeRenderer:Y,plan:e}),Et.upload(),Y.setPathSteps(e.pathData,e.pathMetas)),y(X,e.spineBeats,!0)}else if(i){let e=Mn(Y.graph,C());y(Ot,e,!0),e.viable&&(Dt=new Cn({engine:d(J),nodeRenderer:Y,plan:e}),Dt.upload(),Y.setPathSteps(e.pathData,e.pathMetas)),y(X,e.spineBeats,!0)}else y(X,Y.pathSteps,!0);if(ve()&&Y.graph&&d(K)){P=new Un({engine:d(J),renderer:Y,graph:Y.graph,response:d(K),seed:C(),projectionDays:()=>d(Te),chronoOffsetDays:()=>d(Ee),onFirewall:e=>{y(Ke,e.intruderLabel,!0),y(Ze,Date.now(),!0)}}),y(je,P.liveDecayAvailable,!0),d(J).setPreFrameHook(e=>P?.drain(e)),E()||It();let e=1/0;for(let t of Y.graph.nodes)if(t.createdAt){let n=Date.parse(t.createdAt);Number.isFinite(n)&&n<e&&(e=n)}if(Number.isFinite(e)&&y(N,Math.floor((e-Date.now())/864e5)-1),we){let e=Date.parse(we);Number.isFinite(e)&&y(j,Math.min(365,Math.max(d(N),(e-Date.now())/864e5)),!0),we=null}E()||(I=new fr(d(J),Y,mt()),d(J).addPass(I),F=new nr(d(J),Y.graph.nodes,d(Me)),d(J).addPass(F),y(L,!0)),typeof window<`u`&&(window.__vestigeLiveBridge=P)}d(J).demoClock.reset()}}),t(()=>{let e=o();P&&P.ingest(e)}),t(()=>{d(j),P?.refreshDecay(),F?.setTimeline(d(j),d(M)),I?.setScrubbing(d(M))}),t(()=>{if(!d(Ze))return;let e=setTimeout(()=>{y(Ke,``),y(Ze,0)},7e3);return()=>clearTimeout(e)}),ce(()=>{we=new URLSearchParams(window.location.search).get(`t`),jt();let e=H();return()=>{if(ze(),e?.(),Y?.dispose(),Y=null,typeof window<`u`){let e=window;e.__vestigeLiveBridge===P&&delete e.__vestigeLiveBridge}}});var Rt=Er();u(`keydown`,g,G);var zt=a(Rt);let Bt;var Z=_(zt);let Vt;var Ht=_(Z);ke(Ht,{get demo(){return r.demo},get seed(){return C()},get freezeFrame(){return w()},get maxDpr(){return he()},onframe:Ft,onready:Lt}),S(Z),ue(Z,e=>y($e,e),()=>d($e));var Ut=D(Z,2),Wt=e=>{var t=Tr(),i=_(t),a=e=>{var t=mr(),n=_(t),r=l(n,!0),i=D(n,2);ee(i,19,()=>d(Nt),e=>e.id,(e,t,n)=>{var r=pr(),i=_(r),a=l(i,!0),o=D(i,2),s=l(o,!0),c=D(o,2),u=l(c);S(r),k(e=>{x(r,`href`,e),O(a,d(n)+1),x(o,`title`,d(t).label),O(s,d(t).label),O(u,`${d(t).recalls??``}${d(Pt)===`recalls`?`×`:`%`}`)},[()=>Se(d(t).id)]),h(e,r)}),S(t),k(()=>O(r,d(Pt)===`recalls`?`Most recalled · your mind`:`Strongest memories · your mind`)),h(e,t)};n(i,e=>{ve()&&d(Nt).length>0&&!r.ontop&&e(a)});var o=D(i,2),s=e=>{var t=hr(),n=D(_(t),2),r=l(n,!0);me(2),S(t),k(()=>O(r,d(Ke))),h(e,t)};n(o,e=>{ve()&&d(Qe)&&e(s)});var c=D(o,2),p=e=>{var t=gr(),n=l(t,!0);k(()=>{x(t,`title`,d(V)?`Resume field motion`:`Pause field motion`),x(t,`aria-pressed`,d(V)),x(t,`aria-label`,d(V)?`Resume 3D memory field motion`:`Pause 3D memory field motion`),O(n,d(V)?`▶ RESUME`:`❚❚ PAUSE`)}),T(`click`,t,We),h(e,t)};n(c,e=>{E()||e(p)});var m=D(c,2),g=e=>{var t=vr();let r;var i=D(_(t),2);pe(i);var a=D(i,2);let o;var s=l(a,!0),c=D(a,2),f=e=>{var t=_r();T(`click`,t,()=>y(j,0)),h(e,t)};n(c,e=>{d(j)!==0&&e(f)}),S(t),k(()=>{r=se(t,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 font-mono text-[10px] tracking-[0.14em]
					[text-shadow:0_0_18px_rgba(2,3,7,0.95)]`,null,r,{"opacity-100":d(L),"opacity-75":!d(L)}),x(i,`min`,d(N)),o=se(a,1,`w-16 text-right tabular-nums`,null,o,{"text-[#b9d9a9]":d(j)>=0,"text-[#dfc68e]":d(j)<0}),O(s,d(De))}),T(`input`,i,()=>y(M,!0)),T(`change`,i,()=>y(M,!1)),T(`pointerup`,i,()=>y(M,!1)),u(`pointercancel`,i,()=>y(M,!1)),u(`blur`,i,()=>y(M,!1)),re(i,()=>d(j),e=>y(j,e)),h(e,t)};n(m,e=>{ve()&&d(je)&&e(g)});var v=D(m,2),b=e=>{et(e,{get demoMode(){return r.demo},get seed(){return C()},get nodeCount(){return d(bt)},get edgeCount(){return d(xt)},get centerId(){return d(St)},get frameCount(){return d(vt)},get fpsEstimate(){return d(yt)},get freezeFrame(){return w()},get loading(){return d(q)},get error(){return d(_t)}})};n(v,e=>{le()===`full`&&e(b)});var te=D(v,2),ne=e=>{var t=yr();T(`click`,t,function(...e){r.onexit?.apply(this,e)}),h(e,t)};n(te,e=>{le()===`full`&&r.onexit&&e(ne)});var ae=D(te,2),oe=e=>{var t=xr();ee(t,20,()=>Oe,e=>e,(e,t)=>{var n=br(),i=l(n,!0);k(()=>{se(n,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${t===r.demo?`text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]`:`text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50`}`),x(n,`title`,`Play the ${pt[t]??``} moment`),O(i,pt[t])}),T(`click`,n,()=>r.ondemochange?.(t)),h(e,n)}),S(t),h(e,t)};n(ae,e=>{le()===`full`&&ie()&&e(oe)});var ce=D(ae,2),ue=e=>{var t=Sr();h(e,t)};n(ce,e=>{d(q)&&e(ue)});var de=D(ce,2),fe=e=>{var t=Cr(),n=_(t),r=l(n,!0);S(t),k(()=>O(r,d(_t))),h(e,t)};n(de,e=>{d(_t)&&!d(q)&&e(fe)});var he=D(de,2),ge=e=>{rt(e,{get steps(){return d(X)},get frame(){return d(vt)}})};n(he,e=>{le()===`full`&&e(ge)});var _e=D(he,2),A=e=>{at(e,{get frame(){return d(vt)},get verdict(){return d(Tt).verdict}})};n(_e,e=>{le()===`full`&&r.demo===`salience-rescue`&&d(Tt)?.viable&&e(A)});var ye=D(_e,2),be=e=>{{let t=f(()=>({headline:d(Ot).verdict.headline,causeLabel:d(Ot).verdict.intruderLabel,receipt:d(Ot).verdict.receipt}));at(e,{get frame(){return d(vt)},tone:`quarantine`,fadeWindow:[480,495,605,620],get verdict(){return d(t)}})}};n(ye,e=>{le()===`full`&&r.demo===`firewall`&&d(Ot)?.viable&&e(be)});var xe=D(ye,2),Ce=e=>{var t=wr();h(e,t)};n(xe,e=>{!d(q)&&d(K)&&d(K).nodeCount===0&&e(Ce)}),S(t),h(e,t)};n(Ut,e=>{d(ht)&&e(Wt)}),S(zt);var Gt=D(zt,2);R(Gt,{get pick(){return d(tt)},onclose:()=>y(tt,null)});var qt=D(Gt,2),Jt=e=>{var t=i(),n=a(t);m(n,()=>d(W).id,e=>{{let t=f(()=>r.onpick?`Click to open this memory`:`A memory in the field`);Ae(e,{get x(){return d(W).x},get y(){return d(W).y},get left(){return d(W).left},get tone(){return d(W).tone},get kicker(){return d(W).kicker},get text(){return d(W).text},get action(){return d(t)}})}}),h(e,t)};n(qt,e=>{d(W)&&!E()&&e(Jt)}),k(()=>{Bt=se(zt,1,`${ae()?`absolute`:`fixed`} inset-0 overflow-hidden bg-[#05060a]`,null,Bt,{"cursor-none":E()}),Vt=se(Z,1,`absolute inset-0 z-0 touch-none`,null,Vt,{"cursor-crosshair":!!r.onpick&&!E()})}),T(`click`,Z,st),T(`pointerdown`,Z,ct),T(`pointermove`,Z,lt),u(`pointerleave`,Z,ot),T(`pointerup`,Z,ut),u(`pointercancel`,Z,dt),u(`wheel`,Z,ft),h(e,Rt),oe(),ne()}e([`click`,`pointerdown`,`pointermove`,`pointerup`,`input`,`change`]);var Or=`
struct Params {
	frame: f32, loop_phase: f32, node_count: f32, edge_count: f32,
	path_count: f32, pulse: f32, viewport_w: f32, viewport_h: f32,
	brightness: f32, demo_id: f32, time: f32, capture_mode: f32,
	live_kind: f32, live_frame: f32, live_energy: f32, projection_days: f32,
	cursor_x: f32, cursor_y: f32, cursor_vx: f32, cursor_vy: f32,
};
struct U { centre: vec4f }; // sphere centre xy (NDC), its radius, 0
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<uniform> u: U;

struct VO { @builtin(position) clip: vec4f };
const TRI = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VO {
	var o: VO;
	o.clip = vec4f(TRI[vi], 0.0, 1.0);
	return o;
}
fn h21(q: vec2f) -> f32 {
	var p3 = fract(vec3f(q.x, q.y, q.x) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}
fn vnoise(q: vec2f) -> f32 {
	let i = floor(q);
	let f = fract(q);
	let w = f * f * (3.0 - 2.0 * f);
	return mix(mix(h21(i), h21(i + vec2f(1.0, 0.0)), w.x), mix(h21(i + vec2f(0.0, 1.0)), h21(i + vec2f(1.0, 1.0)), w.x), w.y);
}
fn fbm(q0: vec2f) -> f32 {
	var v = 0.0;
	var a = 0.5;
	var q = q0;
	for (var k = 0; k < 5; k++) {
		v += a * vnoise(q);
		q = q * 2.02 + vec2f(3.1, 1.7);
		a *= 0.5;
	}
	return v;
}
@fragment
fn fs(i: VO) -> @location(0) vec4f {
	let res = vec2f(max(p.viewport_w, 1.0), max(p.viewport_h, 1.0));
	let ndc = vec2f(i.clip.x / res.x * 2.0 - 1.0, 1.0 - i.clip.y / res.y * 2.0);
	let A = res.x / res.y;
	// The same space the stage writes the cursor in (wide: x stretched, tall: y stretched).
	let q = vec2f(ndc.x * max(A, 1.0), ndc.y / min(A, 1.0));
	let t = p.time;
	// Domain-warped clouds, folding slowly.
	let warp = vec2f(fbm(q * 1.1 + vec2f(t * 0.02, 0.0)), fbm(q * 1.1 + vec2f(5.2, t * 0.018)));
	let n = fbm(q * 1.4 + warp * 1.6 + vec2f(0.0, t * 0.01));
	let hue = fbm(q * 0.6 + warp * 0.8 + 9.0);
	// The kinds of memory, blended by where the cloud is.
	let amber = vec3f(1.0, 0.74, 0.3);
	let red = vec3f(1.0, 0.42, 0.33);
	let cyan = vec3f(0.35, 0.88, 1.0);
	let green = vec3f(0.3, 1.0, 0.68);
	let violet = vec3f(0.72, 0.62, 1.0);
	let k = hue * 5.0;
	var col = mix(violet, cyan, clamp(k - 1.0, 0.0, 1.0));
	col = mix(col, green, clamp(k - 2.0, 0.0, 1.0));
	col = mix(col, amber, clamp(k - 2.8, 0.0, 1.0));
	col = mix(col, red, clamp(k - 3.6, 0.0, 1.0));
	let cloud = smoothstep(0.42, 0.85, n);
	// Densest round the sphere, fading toward the words' column.
	let c = vec2f(u.centre.x * max(A, 1.0), u.centre.y / min(A, 1.0));
	let dc = length(q - c) / max(u.centre.z, 0.1);
	let around = exp(-dc * dc * 0.35);
	let cur = vec2f(p.cursor_x, p.cursor_y);
	let near = exp(-dot(q - cur, q - cur) * 3.0);
	let glow = col * cloud * (0.05 + 0.13 * around + 0.22 * near);
	// A few far stars of its own.
	let sp = floor(q * 90.0);
	let star = step(0.9975, h21(sp)) * (0.5 + 0.5 * sin(t * 2.0 + h21(sp + 7.0) * 30.0));
	return vec4f(glow + vec3f(0.85, 0.9, 1.0) * star * 0.35, 1.0);
}
`,kr=class{engine;pipe=null;bind=null;ubuf=null;centre=new Float32Array([0,0,.8,0]);constructor(e){this.engine=e;let t=e.gpuDevice,n=e.paramsBuffer;if(!t||!n)return;this.ubuf=t.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let r=t.createShaderModule({label:`field-nebula`,code:Or});this.pipe=t.createRenderPipeline({label:`field-nebula`,layout:`auto`,vertex:{module:r,entryPoint:`vs`},fragment:{module:r,entryPoint:`fs`,targets:[{format:e.sceneFormat,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.bind=t.createBindGroup({layout:this.pipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:this.ubuf}}]})}setCentre(e,t,n){this.centre.set([e,t,n,0])}compute(){this.ubuf&&this.engine.gpuDevice?.queue.writeBuffer(this.ubuf,0,this.centre)}render(e){this.pipe&&this.bind&&(e.setPipeline(this.pipe),e.setBindGroup(0,this.bind),e.draw(3))}dispose(){this.ubuf?.destroy(),this.ubuf=null}},Ar=`vb1-`,jr=[`0-10%`,`10-20%`,`20-30%`,`30-40%`,`40-50%`,`50-60%`,`60-70%`,`70-80%`,`80-90%`,`90-100%`],Mr=[`concept`,`decision`,`event`,`fact`,`note`,`pattern`,`person`,`place`],Nr=2166136261,Pr=16777619;function Fr(e){let t=typeof e==`string`?new TextEncoder().encode(e):e,n=Nr;for(let e=0;e<t.length;e++)n^=t[e],n=Math.imul(n,Pr);return n>>>0}function Ir(e){return/^vb1-[0-9a-f]{8}$/.test(e)}function Lr(e){return`${Ar}${(e>>>0).toString(16).padStart(8,`0`)}`}function $(e){return Number.isFinite(e)?Math.max(0,Math.round(e)):0}function Rr(e){return Number.isFinite(e)?Math.max(0,Math.min(1e3,Math.round(e*1e3))):0}function zr(e){return Number.isFinite(e)?Math.max(0,Math.min(1e3,Math.round(e*10))):0}function Br(e){return e.toLowerCase().replace(/[^a-z0-9_-]/g,``)}function Vr(e){let t=new Map;for(let e of jr)t.set(e,0);for(let n of e){let e=n.range.trim();e&&t.set(e,$(n.count))}return t}function Hr(e){let t=new Map;for(let e of Mr)t.set(e,0);for(let[n,r]of Object.entries(e)){let e=Br(n);e&&t.set(e,(t.get(e)??0)+$(r))}return t}function Ur(e){let t=$(e.nodeCount),n=$(e.edgeCount),r=t<=0?0:Math.round(1e3*n/t),i=Hr(e.byType),a=Vr(e.retentionBuckets);return[1,$(e.totalMemories),0,Rr(e.averageRetention),zr(e.embeddingCoverage),$(e.endangeredCount),t,n,r,...Mr.map(e=>i.get(e)??0),...jr.map(e=>a.get(e)??0)]}function Wr(e){let t=Ur(e),n=Hr(e.byType),r=[...n.entries()].filter(([e])=>!Mr.includes(e)).filter(([,e])=>e>0).sort(([e],[t])=>e.localeCompare(t)).map(([e,t])=>`${e}=${t}`).join(`,`),i=Vr(e.retentionBuckets),a=jr.map(e=>`${e}=${i.get(e)??0}`).join(`,`),o=r.length?`|extra:${r}`:``;return`v${t[0]}|t:${t[1]}|d:${t[2]}|r:${t[3]}|c:${t[4]}|z:${t[5]}|n:${t[6]}|g:${t[7]}|x:${t[8]}|types:${Mr.map(e=>`${e}=${n.get(e)??0}`).join(`,`)}|ret:${a}${o}`}function Gr(e,t){let n=Math.max(1,$(e.totalMemories)||$(e.retentionBuckets.reduce((e,t)=>e+$(t.count),0))),r=0;for(let n of Vr(e.retentionBuckets)){let e=/^(\d+)\s*-\s*(\d+)%$/.exec(n[0]);e&&t(Number(e[1]),Number(e[2]))&&(r+=n[1])}return r/n}function Kr(e){let t=Math.max(0,$(e.totalMemories)),n=Math.max(1,t),r=Math.max(0,$(e.nodeCount)),i=Math.max(0,$(e.edgeCount)),a=[...Hr(e.byType).entries()].filter(([,e])=>e>0),o=a.reduce((e,[,t])=>e+t,0)||1,s=0;for(let[,e]of a){let t=e/o;s-=t*Math.log2(t)}let c=null,l=0;for(let[e,t]of a){let n=t/o;(n>l||n===l&&e.localeCompare(c??``)<0)&&(c=e,l=n)}return{total:t,avgRet:Rr(e.averageRetention)/1e3,coverage:zr(e.embeddingCoverage)/1e3,endangeredRatio:$(e.endangeredCount)/n,edgeDensity:r<=0?0:i/r,highRetFrac:Gr(e,e=>e>=70),lowRetFrac:Gr(e,(e,t)=>t<=30),typeEntropy:s,dominantType:c,dominantFrac:l,typeCount:a.length}}var qr=[{id:`dense-associative`,label:`dense associative field`,group:`density`,score:e=>e.edgeDensity>=1.2?Math.min(3,e.edgeDensity):0},{id:`sparse-lattice`,label:`sparse lattice`,group:`density`,score:e=>e.edgeDensity>0&&e.edgeDensity<.45?1.4-e.edgeDensity:0},{id:`deep-archive`,label:`deep archive`,group:`vitality`,score:e=>e.highRetFrac>=.4||e.avgRet>=.72?1.2+e.highRetFrac:0},{id:`sedimentary`,label:`sedimentary dark`,group:`vitality`,score:e=>e.lowRetFrac>=.3||e.endangeredRatio>=.22?1.1+e.lowRetFrac:0},{id:`oxygen-rich`,label:`oxygen-rich field`,group:`vitality`,score:e=>e.avgRet>=.78&&e.highRetFrac>=.35?e.avgRet:0},{id:`typed-mosaic`,label:`typed mosaic`,group:`types`,score:e=>e.typeCount>=5&&e.typeEntropy>=1.8?e.typeEntropy:0},{id:`concept-dominant`,label:`concept-weighted`,group:`types`,score:e=>e.dominantType===`concept`&&e.dominantFrac>=.4?e.dominantFrac:0},{id:`event-forward`,label:`event-forward`,group:`types`,score:e=>e.dominantType===`event`&&e.dominantFrac>=.4?e.dominantFrac:0},{id:`decision-heavy`,label:`decision-heavy`,group:`types`,score:e=>e.dominantType===`decision`&&e.dominantFrac>=.35?e.dominantFrac:0},{id:`wide-field`,label:`wide-field archive`,group:`scale`,score:e=>e.total>=400?Math.log10(e.total):0},{id:`expanding-cortex`,label:`expanding cortex`,group:`scale`,score:e=>e.total>=80&&e.total<400?.6:0},{id:`compact-nucleus`,label:`compact nucleus`,group:`scale`,score:e=>e.total>0&&e.total<80?.55:0},{id:`covered-embeddings`,label:`fully embedded`,group:`coverage`,score:e=>e.coverage>=.95?e.coverage:0}],Jr=[{id:`wide-field`,label:`wide-field archive`},{id:`expanding-cortex`,label:`expanding cortex`},{id:`compact-nucleus`,label:`compact nucleus`}];function Yr(e){let t=Kr(e),n=qr.map(e=>({rule:e,score:e.score(t)})).filter(e=>e.score>0).sort((e,t)=>t.score-e.score||e.rule.id.localeCompare(t.rule.id)),r=[],i=new Set,a=new Set;for(let{rule:e}of n){if(r.length>=3)break;i.has(e.group)||(r.push({id:e.id,label:e.label}),i.add(e.group),a.add(e.id))}if(r.length<2){let e=t.total>=400?Jr[0]:t.total>=80?Jr[1]:Jr[2];e&&!a.has(e.id)&&(r.push(e),a.add(e.id))}return r.length<2&&r.push({id:`structured-field`,label:`structured field`}),r.slice(0,3)}function Xr(e){let t=Lr(Fr(Wr(e)));return{printId:t,seed:t,traits:Yr(e),vector:Ur(e)}}function Zr(e){let t=e.retention.endangered,n=e.head;return{totalMemories:$(e.stats.totalMemories),dueForReview:$(e.stats.dueForReview),averageRetention:n?n.averageRetention:e.stats.averageRetention,embeddingCoverage:e.stats.embeddingCoverage,endangeredCount:n?$(n.endangeredCount):t?$(t.length):0,byType:{...e.retention.byType},retentionBuckets:(n?n.distribution:e.retention.distribution).map(e=>({range:e.range,count:$(e.count)})),nodeCount:$(e.topology?.nodeCount??0),edgeCount:$(e.topology?.edgeCount??0)}}async function Qr(e){let[t,n,r,i]=await Promise.all([A.stats(),A.retentionDistribution(),A.brainShape(),e?Promise.resolve(null):A.graph({max_nodes:200,depth:3,sort:`connected`}).catch(()=>null)]);return{shape:Zr({stats:t,retention:n,head:r,topology:e??(i?{nodeCount:i.nodeCount,edgeCount:i.edgeCount}:void 0)}),stats:t}}function $r(e){let t=Ur(e),n=Hr(e.byType),r={};for(let[e,t]of n)!Mr.includes(e)&&t>0&&(r[e]=t);let i=JSON.stringify({v:t,e:r});return`v1.${btoa(i).replace(/\+/g,`-`).replace(/\//g,`_`).replace(/=+$/g,``)}`}function ei(e){let t=/^v1\.([A-Za-z0-9_-]+)$/.exec(e.trim());if(!t)return null;try{let e=t[1].replace(/-/g,`+`).replace(/_/g,`/`),n=e.length%4==0?``:`=`.repeat(4-e.length%4),r=JSON.parse(atob(e+n));if(!Array.isArray(r.v)||r.v.length<19)return null;let i=ti(r.v.map(e=>$(Number(e))));if(r.e&&typeof r.e==`object`)for(let[e,t]of Object.entries(r.e)){let n=Br(e);n&&(i.byType[n]=$(t))}return i}catch{return null}}function ti(e){let t=e.length>=19?e:[...e,...Array(19-e.length).fill(0)],n={};Mr.forEach((e,r)=>{let i=$(t[9+r]??0);i>0&&(n[e]=i)});let r=jr.map((e,n)=>({range:e,count:$(t[17+n]??0)}));return{totalMemories:$(t[1]??0),dueForReview:$(t[2]??0),averageRetention:$(t[3]??0)/1e3,embeddingCoverage:$(t[4]??0)/10,endangeredCount:$(t[5]??0),byType:n,retentionBuckets:r,nodeCount:$(t[6]??0),edgeCount:$(t[7]??0)}}function ni(e,t){let n=Math.min(200,Math.max(0,$(e.nodeCount)||$(e.totalMemories))),r=[...Hr(e.byType).entries()].filter(([,e])=>e>0),i=r.reduce((e,[,t])=>e+t,0)||1,a=[...Vr(e.retentionBuckets).entries()],o=a.reduce((e,[,t])=>e+t,0)||1,s=[];for(let e=0;e<n;e++){let n=(e*2654435761>>>0)%i,c=r[0]?.[0]??`note`;for(let[e,t]of r)if(n-=t,n<0){c=e;break}let l=(e*1597334677>>>0)%o,u=.55;for(let[e,t]of a)if(l-=t,l<0){let t=/^(\d+)\s*-\s*(\d+)/.exec(e);u=((t?Number(t[1]):50)+(t?Number(t[2]):60))/200;break}let d=`syn-${t}-${e.toString(16).padStart(3,`0`)}`;s.push({id:d,label:`${c}·${(e+1).toString().padStart(3,`0`)}`,type:c,retention:u,tags:[],createdAt:`1970-01-01T00:00:00.000Z`,updatedAt:`1970-01-01T00:00:00.000Z`,isCenter:e===0})}let c=n<=0?0:$(e.edgeCount)/Math.max(1,$(e.nodeCount)||n),l=Math.min(n*12,Math.round(n*Math.max(.2,Math.min(8,c)))),u=[],d=new Set;for(let e=0;e<l&&s.length>1;e++){let t=(e*1103515245+12345>>>0)%s.length,n=(e*214013+2531011>>>0)%s.length;n===t&&(n=(n+1)%s.length);let r=s[t].id,i=s[n].id,a=`${r}:${i}`;d.has(a)||(d.add(a),u.push({source:r,target:i,weight:.35+e%65/100,type:`assoc`}))}return{nodes:s,edges:u,center_id:s[0]?.id??``,depth:3,nodeCount:s.length,edgeCount:u.length}}async function ri(e){let{shape:t,stats:n}=await Qr(e),r=Xr(t),i=n.oldestMemory,a=n.newestMemory,o=0;if(i&&a){let e=Date.parse(i),t=Date.parse(a);Number.isFinite(e)&&Number.isFinite(t)&&(o=Math.max(0,Math.round(Math.abs(t-e)/864e5)))}return{shape:t,print:r,archiveDays:o}}e([`click`]);function ii(e){let t=0,n=0,r=0;for(let i of e.retentionBuckets){let e=/^(\d+)/.exec(i.range),a=((e?Number(e[1]):0)+5)/100;a>=.7?t+=i.count:a>=.4?n+=i.count:r+=i.count}return{active:t,dormant:n,silent:r}}function ai(e,t,n,r,i){let a=new L({seed:`${n}:wrapped`}).state.rng,o=Ur(t),s=Math.min(420,Math.max(48,Math.round(Math.sqrt(Math.max(1,t.totalMemories))*18))),c=r*.5,l=i*.42;for(let n=0;n<s;n++){let s=a()*Math.PI*2,u=Math.sqrt(a())*Math.min(r,i)*.34,d=c+Math.cos(s)*u*(.7+.6*a()),f=l+Math.sin(s)*u*.85,p=o[17+n%10]??1,m=Math.min(1,p/Math.max(1,t.totalMemories/10)),h=1.2+m*4.5,g=e.createRadialGradient(d,f,0,d,f,h*3);g.addColorStop(0,`rgba(159, 248, 236, ${.55+m*.4})`),g.addColorStop(.45,`rgba(34, 199, 222, ${.25+m*.35})`),g.addColorStop(1,`rgba(2, 3, 7, 0)`),e.fillStyle=g,e.beginPath(),e.arc(d,f,h*3,0,Math.PI*2),e.fill()}}async function oi(e){let t=e.width??1080,n=e.height??1920,r=document.createElement(`canvas`);r.width=t,r.height=n;let i=r.getContext(`2d`);if(!i)throw Error(`2d canvas unavailable`);i.fillStyle=`#020307`,i.fillRect(0,0,t,n);let a=i.createRadialGradient(t*.5,n*.38,40,t*.5,n*.4,t*.55);a.addColorStop(0,`rgba(34, 199, 222, 0.16)`),a.addColorStop(1,`rgba(2, 3, 7, 0)`),i.fillStyle=a,i.fillRect(0,0,t,n),e.fieldBitmap?(i.save(),i.globalAlpha=.55,i.drawImage(e.fieldBitmap,0,n*.12,t,9/16*t),i.restore()):ai(i,e.shape,e.print.seed,t,n);let o=ii(e.shape),{print:s,shape:c,archiveDays:l}=e;i.fillStyle=`rgba(234, 255, 251, 0.92)`,i.font=`600 28px ui-sans-serif, system-ui, sans-serif`,i.fillText(`VESTIGE`,72,120),i.fillStyle=`rgba(127, 243, 230, 0.75)`,i.font=`500 22px ui-sans-serif, system-ui, sans-serif`,i.fillText(`MEMORY REPORT`,72,158),i.fillStyle=`#7ff3e6`,i.font=`700 54px ui-monospace, SFMono-Regular, Menlo, monospace`,i.fillText(s.printId,72,260),i.fillStyle=`rgba(140, 199, 180, 0.7)`,i.font=`500 18px ui-sans-serif, system-ui, sans-serif`,i.fillText(`STRUCTURE ONLY  ·  ZERO MEMORY TEXT`,72,300),[[`MEMORIES`,c.totalMemories.toLocaleString()],[`CONNECTIONS`,c.edgeCount.toLocaleString()],[`ACTIVE`,o.active.toLocaleString()],[`DORMANT`,o.dormant.toLocaleString()],[`SILENT`,o.silent.toLocaleString()],[`ARCHIVE`,`${Math.max(0,l)}d`]].forEach(([e,t],n)=>{let r=n%2,a=Math.floor(n/2),o=72+r*480,s=420+a*140;i.fillStyle=`rgba(140, 175, 180, 0.55)`,i.font=`600 18px ui-sans-serif, system-ui, sans-serif`,i.fillText(e,o,s),i.fillStyle=`#eafffb`,i.font=`700 64px ui-sans-serif, system-ui, sans-serif`,i.fillText(t,o,s+70)});let u=72;i.font=`600 22px ui-sans-serif, system-ui, sans-serif`;for(let e of s.traits){let t=i.measureText(e.label).width+36;i.fillStyle=`rgba(20, 48, 40, 0.75)`,i.strokeStyle=`rgba(92, 240, 166, 0.45)`,i.lineWidth=2,si(i,u,980,t,48,24),i.fill(),i.stroke(),i.fillStyle=`#c8ffe8`,i.fillText(e.label,u+18,1012),u+=t+16}return i.fillStyle=`rgba(140, 199, 219, 0.55)`,i.font=`500 20px ui-sans-serif, system-ui, sans-serif`,i.fillText(`share your brain, not your memories`,72,n-80),{blob:await new Promise((e,t)=>{r.toBlob(n=>n?e(n):t(Error(`toBlob failed`)),`image/png`)}),width:t,height:n,filename:`vestige-${s.printId}-report.png`}}function si(e,t,n,r,i,a){e.beginPath(),e.moveTo(t+a,n),e.arcTo(t+r,n,t+r,n+i,a),e.arcTo(t+r,n+i,t,n+i,a),e.arcTo(t,n+i,t,n,a),e.arcTo(t,n,t+r,n,a),e.closePath()}function ci(e,t){console.info(`[wrapped-card] ${t}: ${(e.size/1e3).toFixed(0)}KB`);let n=URL.createObjectURL(e),r=document.createElement(`a`);r.href=n,r.download=t,r.click(),setTimeout(()=>URL.revokeObjectURL(n),1e4)}var li=r(`Only the candidate evidence recorded in receipt <code class="svelte-ymjpra"> </code>.`,1),ui=r(`Only the memories named by receipt <code class="svelte-ymjpra"> </code> `,1),di=r(`<span class="obs-err svelte-ymjpra"> </span>`),fi=r(`<b class="svelte-ymjpra"></b> memories · <b class="svelte-ymjpra"></b> recorded links`,1),pi=r(`<a class="obs-item svelte-ymjpra"><span class="obs-item-head svelte-ymjpra"> </span> <span class="obs-item-text svelte-ymjpra"> </span></a>`),mi=r(`<p class="obs-item svelte-ymjpra"><span class="obs-item-head svelte-ymjpra"> </span> <span class="obs-item-text svelte-ymjpra"> </span></p>`),hi=r(`<p class="obs-note svelte-ymjpra">Nothing yet while this page has been open. The next recall, write, dream or contradiction any agent makes appears here, and lights the memories it touched.</p>`),gi=r(`<p class="obs-kicker svelte-ymjpra">Live from your agents</p> <!>`,1),_i=r(`<p class="obs-kicker svelte-ymjpra"> </p> <!>`,1),vi=r(`<div style="position: fixed; left: -100000px; top: 0; width: 1920px; height: 1080px;" aria-hidden="true"><!></div>`),yi=r(`<div class="fixed inset-0 bg-[#020307]"><!></div>  <aside aria-live="polite"><p class="obs-eyebrow svelte-ymjpra"></p> <h1 class="obs-title svelte-ymjpra"></h1> <p class="obs-sub svelte-ymjpra"><!></p> <p class="obs-stats svelte-ymjpra"><!> <!></p> <!> <!> <p class="obs-note svelte-ymjpra">Hover a star to read it; click it to go inside.</p> <div class="obs-actions svelte-ymjpra"><button class="obs-act svelte-ymjpra" type="button"> </button> <a class="obs-act svelte-ymjpra">What shaped each decision →</a></div></aside> <!>`,1);function bi(e,r){de(r,!0);let s=()=>fe(_e,`$eventFeed`,g),[g,b]=te(),re=new URLSearchParams(window.location.search),ie=re.get(`receipt`),pe=re.get(`brain`),Ce=ie?`recall-path`:re.get(`demo`)??`recall-path`,N=v(c(we(Ce)?Ce:`recall-path`)),Te=v(c(re.get(`seed`)??`vestige-observatory-v1`)),Ee=re.get(`frame`),De=Ee!==null&&Ee!==``?Number(Ee):null,P=re.has(`capture`)||De!==null;[...Ie(z.forward)],[...Ie(Pe.recall)],[...Ie(Pe.luciferin)],[...Ie(Ne.veto)],[...Ie(Ne.caution)];let F=v(null),I=null,ke=null,L=null,Ae=null,R=v(null),B=v(null),Be=v(null),Ve=v(!0),He=v(null),V=null,Ue=null,H=new Set,We=[],Ke=typeof window<`u`&&window.matchMedia(`(prefers-reduced-motion: reduce)`).matches;ce(()=>{$e()}),ae(()=>{We.forEach(clearTimeout),We=[],ke?.dispose(),L?.dispose(),Ae?.dispose(),ke=null,L=null,Ae=null,I=null});async function qe(e){I=e,Ae?.dispose(),Ae=new kr(e),dt(),e.addPass(Ae);let t=new Le(e);L=t;let n=nt();t.setIntensity(n===null?.2:.16),t.setReadingWell({x:0,y:0,hw:1.15,hh:1.15,floor:.06,soft:.4}),t.setCells(Ye(H)),e.addPass(t);let r=new Fe(e);ke=r,await r.init(),r.setText(W()),e.addPass(r),e.demoClock.reset(),d(Ve)||j()}function Je(e){let t=tt(e.retention),n=Math.max(0,Number.isFinite(e.stability)?e.stability:0),r=tt(Math.log10(1+n)/5),i=.3;e.lastAccessed&&(i=tt(1-(Date.now()-new Date(e.lastAccessed).getTime())/864e5/30));let a=.5*t+.32*r+.18*i;return tt(e.isCenter?Math.max(a,.9):a)}function Ye(e=new Set){let t=d(R)?.nodes??[],n=[...t].map(e=>({id:e.id,s:Je(e)})).sort((e,t)=>e.s-t.s),r=new Map,i=Math.max(1,n.length-1);n.forEach((e,t)=>r.set(e.id,t/i));let a=t.map(t=>{let n=r.get(t.id)??0,i=e.has(t.id),a=t.isCenter?Math.max(n,.92):n;return{id:t.id,score:a,hue:Me(a,i),energy:je(a,i),selected:!!t.isCenter||i,scar:(t.suppression_count??0)>0,metric2:tt(t.retention),kind:`observatory-cell`,payload:t}});return Re(a,{maxRadius:.95,minCellR:.01,maxCellR:.04})}function Xe(e){let t=[...d(R)?.nodes??[]].map(e=>({id:e.id,s:Je(e)})).sort((e,t)=>t.s-e.s).slice(0,e).map(e=>e.id);return new Set(t)}let Ze=f(()=>{let e=ve(d(B));if(e)return{failureId:e.failure_id,pathIds:e.path_ids,candidates:e.candidates.map(e=>({memoryId:e.memory_id,sharedEntities:e.shared_entities,ageDays:e.age_days_before_failure,similarityRank:e.similarity_rank,promoted:e.promoted}))}}),Qe=f(()=>{if(!d(B))return[];let e=ve(d(B));return[...new Set([...d(B).retrieved,...d(B).suppressed.map(e=>e.id),...e?[e.failure_id,...e.path_ids??[],...e.candidates.map(e=>e.memory_id)]:[]])]});async function $e(){if(ie)try{y(B,await A.receipts.get(ie),!0),ve(d(B))&&y(N,`salience-rescue`)}catch(e){y(Be,e instanceof Error?e.message:`Receipt unavailable`,!0)}else if(pe){let e=ei(pe);if(e){y(_t,e,!0);let t=Xr(e);y(q,t,!0),y(Te,t.seed,!0),y(R,ni(e,t.printId),!0),y(bt,d(R),!0),y(Ve,!1);return}}await U()}async function U(){y(Ve,!0),y(He,null),ke?.setText(W());try{let e=new Set(d(Qe));if(e.size){let t=await Promise.all([...e].map(e=>A.graph({center_id:e,max_nodes:200,depth:3}))),n=t.flatMap(e=>e.nodes),r=t.flatMap(e=>e.edges),i=[...new Map(n.map(e=>[e.id,e])).values()].filter(t=>e.has(t.id)),a=new Set(i.map(e=>e.id)),o=[...new Map(r.map(e=>[`${e.source}:${e.target}`,e])).values()].filter(e=>a.has(e.source)&&a.has(e.target));y(R,{...t[0],nodes:i,edges:o,center_id:i[0]?.id??t[0]?.center_id??``,nodeCount:i.length,edgeCount:o.length},!0)}else y(R,await Ge(),!0)}catch(e){y(R,null),y(He,e instanceof Error?e.message:`UNKNOWN OBSERVATORY GRAPH ERROR`,!0)}finally{y(Ve,!1),ke?.setText(W()),et(),I?.demoClock.reset(),I&&j()}}function et(){We.forEach(clearTimeout),We=[],H=new Set;let e=[...Xe(7)];if(!e.length){L?.setCells(Ye(H));return}if(De!==null||Ke){H=new Set(e),L?.setCells(Ye(H));return}L?.setCells(Ye(H)),e.forEach((e,t)=>{We.push(setTimeout(()=>{H.add(e),L?.setCells(Ye(H))},420+t*130))})}function tt(e){return Math.min(1,Math.max(0,Number.isFinite(e)?e:.5))}function nt(){let e=I?.params[6]||0,t=I?.params[7]||0;if((e<=0||t<=0)&&typeof window<`u`&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0)return null;let n=e/t;return n<.85?n:null}function W(){return[]}let rt=v(!1);v(null),v(null);let it=v(0);function at(e){}function ot(e){if(e===d(N))return;y(N,e,!0);let t=new URL(window.location.href);t.searchParams.set(`demo`,e),history.replaceState(history.state,``,t),ke?.setText(W()),I?.demoClock.reset()}[{mode:`recall-path`,label:`Recall`,blurb:`Watch a memory get retrieved — the path lights up.`},{mode:`engram-birth`,label:`Engram`,blurb:`A new memory forms and wires into the field.`},{mode:`salience-rescue`,label:`Salience`,blurb:`The few memories that matter ignite gold.`},{mode:`forgetting-horizon`,label:`Forgetting`,blurb:`FSRS decay pulls weak memories toward the dark.`},{mode:`firewall`,label:`Firewall`,blurb:`A contradiction is caught and quarantined.`}].filter(e=>Oe.includes(e.mode));let st=v(c({w:1440,h:900}));ce(()=>{let e=()=>y(st,{w:innerWidth,h:innerHeight},!0);return e(),addEventListener(`resize`,e),()=>removeEventListener(`resize`,e)});let ct=f(()=>d(st).w>=900&&d(st).w>=d(st).h),lt=v(1);ce(()=>{if(!le(ye))return;y(lt,0);let e=performance.now(),t=n=>{let r=Math.min(1,(n-e)/2e3);y(lt,1-(1-r)**2),r<1&&requestAnimationFrame(t)};requestAnimationFrame(t)});let ut=f(()=>{let e=d(ct)?{scale:.74,x:-.3,y:.02}:{scale:.5,x:0,y:.47};return{scale:e.scale*(3.4-2.4*d(lt)),x:e.x*d(lt),y:e.y*d(lt)}});function dt(){let e=d(ut);Ae?.setCentre(e.x,e.y,.95*e.scale)}t(()=>{d(ut),dt()});let ft=f(()=>d(ct)?{left:-.72,right:.12,y:-.8,show:!0}:{left:-.8,right:.8,y:-.7,show:!1}),pt=v(c([])),mt=v(`retention`),ht={MemoryCreated:`wrote`,MemoryUpdated:`changed`,MemoryPromoted:`promoted`,MemoryDemoted:`demoted`,MemorySuppressed:`suppressed`,SearchPerformed:`recalled`,DeepReferenceCompleted:`reasoned over`,DreamCompleted:`dreamed`,ConnectionDiscovered:`linked`,CausalReceipt:`walked back from`,BackfillFired:`walked back from`,MemoryShown:`showed`,HookVerdictRecorded:`met the gate`,MemoryPrOpened:`opened a review of`,MemoryPrDecided:`decided a review of`},G=e=>typeof e==`string`&&e.trim()?e:null,K=f(()=>s().filter(e=>ht[e.type]).slice(0,9).map((e,t)=>{let n=e.data??{},r=Array.isArray(n.result_ids)?n.result_ids.filter(e=>typeof e==`string`):[],i=G(n.id)??G(n.primary_id)??G(n.source_id)??r[0]??null,a=e.type===`SearchPerformed`?`${G(n.query)??`a handle`} · ${n.result_count??r.length} found`:e.type===`DreamCompleted`?`${n.memories_replayed??0} memories replayed · ${n.connections_found??0} links found`:e.type===`HookVerdictRecorded`?`${G(n.verdict)??``} ${G(n.reason)??``}`:G(n.content_preview)??G(n.reason)??G(n.query)??G(n.connection_type)??``,o=G(n.timestamp);return{key:`${e.type}:${i??``}:${o??t}`,verb:ht[e.type],by:G(n.by),id:i,what:xe(a,90),tone:G(n.node_type)??(e.type===`MemorySuppressed`?`correction`:`fact`),when:o?new Date(o).toLocaleTimeString([],{hour:`2-digit`,minute:`2-digit`,second:`2-digit`}):``}})),gt=e=>{let[t,n,r]=be(e);return`rgb(${Math.round(t*255)} ${Math.round(n*255)} ${Math.round(r*255)})`},q=v(null),_t=v(null);v(!1);let vt=v(!1),yt=v(null);v(!1);let bt=v(null);f(()=>d(q)?.printId??(Ir(d(Te))?d(Te):null));function xt(e,t){y(Te,e,!0);let n=new URL(window.location.href);n.searchParams.set(`seed`,e),t&&n.searchParams.set(`brain`,$r(t)),history.replaceState(history.state,``,n)}async function St(){if(!d(vt)){y(vt,!0),y(yt,null);try{let e=d(B)||!d(R)?void 0:{nodeCount:d(R).nodeCount,edgeCount:d(R).edgeCount},t=d(_t)&&d(q)?{shape:d(_t),print:d(q),archiveDays:0}:await ri(e);d(_t)||(y(_t,t.shape,!0),y(q,t.print,!0),xt(t.print.printId,t.shape));let n=await oi({shape:t.shape,print:t.print,archiveDays:t.archiveDays,fieldBitmap:I?.canvasElement??null});ci(n.blob,n.filename)}catch(e){y(yt,e instanceof Error?e.message:`Memory report failed`,!0)}finally{y(vt,!1)}}}function J(e){if(!d(F))return null;let t=d(F).getBoundingClientRect();return t.width<=0||t.height<=0?null:{x:(e.clientX-t.left)/t.width*2-1,y:-((e.clientY-t.top)/t.height*2-1)}}function Y(e){if(!d(F)||!I)return;let t=d(F).getBoundingClientRect(),n=Math.max(1e-4,t.width/Math.max(1,t.height)),r={x:e.x*Math.max(n,1),y:e.y/Math.min(n,1)},i=Ue??r,a={x:i.x+(r.x-i.x)*.35,y:i.y+(r.y-i.y)*.35};Ue=a,I.setCursorPreNdc(a.x,a.y,a.x-i.x,a.y-i.y)}function Ct(e){let t=J(e);if(!t)return;Y(t);let n=ke?.pickAt(t.x,t.y)??null,r=n?.kind===`observatory-demo`||n?.kind===`observatory-exit`?n.id:null;r!==V&&(V=r,ke?.setRunDepth(r,1)),d(F)&&(d(F).style.cursor=r?`crosshair`:`default`)}function wt(){Ue=null,V=null,I?.setCursorPreNdc(999,999,0,0),ke?.setRunDepth(null),d(F)&&(d(F).style.cursor=`default`)}async function Tt(e){let t=J(e);if(!t)return;let n=ke?.pickAt(t.x,t.y)?.payload;if(n?.action===`demo`&&n.demo){ot(n.demo);return}if(n?.action===`exit`){await ge(`${he}/palace`);return}let r=L?.pickAt(t.x,t.y);r&&typeof r.id==`string`&&await M(r.id,{canvas:d(F)?.querySelector(`canvas`)??null,focus:t})}var Et=yi();C(`ymjpra`,e=>{p(()=>{o.title=`Observatory · Vestige`})});var Dt=a(Et),Ot=_(Dt);m(Ot,()=>`${d(N)}:${d(Te)}:${d(B)?.receipt_id??`all`}:${d(bt)?`brain`:`live`}`,e=>{{let t=f(()=>!d(bt));Dr(e,{get demo(){return d(N)},get seed(){return d(Te)},get freezeFrame(){return De},get capture(){return P},showSwitcher:!1,chrome:`none`,get live(){return d(t)},get graphOverride(){return d(bt)},get focusIds(){return d(Qe)},get backfillEvidence(){return d(Ze)},onready:qe,onpick:(e,t)=>void M(e,{canvas:d(F)?.querySelector(`canvas`)??null,focus:t,engine:I}),get fieldFrame(){return d(ut)},get rail(){return d(ft)},ontop:(e,t)=>{y(pt,e,!0),y(mt,t,!0)},onexit:()=>ge(`${he}/palace`)})}}),S(Dt),ue(Dt,e=>y(F,e),()=>d(F));var kt=D(Dt,2);let At;var X=_(kt);w(X,(e,t)=>ze?.(e,t),()=>({text:`Field`,glow:!0}));var jt=D(X,2);w(jt,(e,t)=>ze?.(e,t),()=>({text:d(Ze)?`Backfill replay`:d(B)?`One receipt, replayed`:`Your mind, live.`,glow:!0}));var Mt=D(jt,2),Nt=_(Mt),Pt=e=>{var t=li(),n=D(a(t)),r=l(n,!0);me(),k(()=>O(r,d(B)?.receipt_id)),h(e,t)},Ft=e=>{var t=ui(),n=D(a(t)),r=l(n,!0),i=D(n);k(()=>{O(r,d(B).receipt_id),O(i,`: ${d(B).retrieved.length??``} retrieved, ${d(B).suppressed.length??``} suppressed.`)}),h(e,t)},It=e=>{var t=E(`Every star is one of your memories, coloured by kind. Lit paths replay your agents' real recalls; whatever an agent does now plays here on the memories it touches.`);h(e,t)};n(Nt,e=>{d(Ze)?e(Pt):d(B)?e(Ft,1):e(It,-1)}),S(Mt);var Lt=D(Mt,2),Rt=_(Lt),zt=e=>{var t=E(`Reading the field…`);h(e,t)},Bt=e=>{var t=di(),n=l(t,!0);k(()=>O(n,d(He))),h(e,t)},Z=e=>{var t=fi(),n=a(t);w(n,(e,t)=>ze?.(e,t),()=>({text:d(R).nodeCount.toLocaleString(),count:!0}));var r=D(n,2);w(r,(e,t)=>ze?.(e,t),()=>({text:d(R).edgeCount.toLocaleString(),count:!0})),me(),h(e,t)};n(Rt,e=>{d(Ve)?e(zt):d(He)?e(Bt,1):d(R)&&e(Z,2)});var Vt=D(Rt,2),Ht=e=>{var t=di(),n=l(t);k(()=>O(n,`· ${d(Be)??``}`)),h(e,t)};n(Vt,e=>{d(Be)&&e(Ht)}),S(Lt);var Ut=D(Lt,2),Wt=e=>{var t=gi(),r=D(a(t),2),o=e=>{var t=i(),r=a(t);ee(r,17,()=>d(K),e=>e.key,(e,t)=>{var r=i(),o=a(r),s=e=>{var n=pi();let r;var i=_(n),a=l(i),o=D(i,2),s=l(o,!0);S(n),k((e,i)=>{x(n,`href`,e),r=ne(n,``,r,{"--tone":i}),O(a,`${d(t).when?`${d(t).when} · `:``}${d(t).by?`${d(t).by} `:``}${d(t).verb??``}`),O(s,d(t).what||`a memory`)},[()=>Se(d(t).id),()=>gt(d(t).tone)]),h(e,n)},c=e=>{var n=mi();let r;var i=_(n),a=l(i),o=D(i,2),s=l(o,!0);S(n),k(e=>{r=ne(n,``,r,{"--tone":e}),O(a,`${d(t).when?`${d(t).when} · `:``}${d(t).by?`${d(t).by} `:``}${d(t).verb??``}`),O(s,d(t).what)},[()=>gt(d(t).tone)]),h(e,n)};n(o,e=>{d(t).id?e(s):e(c,-1)}),h(e,r)}),h(e,t)},s=e=>{var t=hi();h(e,t)};n(r,e=>{d(K).length?e(o):e(s,-1)}),h(e,t)};n(Ut,e=>{d(B)||e(Wt)});var Gt=D(Ut,2),Kt=e=>{var t=_i(),n=a(t),r=l(n,!0),i=D(n,2);ee(i,19,()=>d(pt),e=>e.id,(e,t,n)=>{var r=pi(),i=_(r),a=l(i),o=D(i,2),s=l(o,!0);S(r),k((e,i)=>{x(r,`href`,e),O(a,`${d(n)+1} · ${d(t).recalls??``}${d(mt)===`recalls`?` recalls`:`% retained`}`),O(s,i)},[()=>Se(d(t).id),()=>xe(d(t).label,90)]),h(e,r)}),k(()=>O(r,d(mt)===`recalls`?`Most recalled`:`Held strongest`)),h(e,t)};n(Gt,e=>{d(pt).length&&e(Kt)});var qt=D(Gt,4),Jt=_(qt),Yt=l(Jt,!0),Xt=D(Jt,2);S(qt),S(kt);var Zt=D(kt,2),Qt=e=>{var t=vi(),n=_(t);m(n,()=>d(it),e=>{Dr(e,{get demo(){return d(N)},get seed(){return d(Te)},showSwitcher:!1,chrome:`none`,embedded:!0,maxDpr:1,get graphOverride(){return d(bt)},get focusIds(){return d(Qe)},get backfillEvidence(){return d(Ze)},onready:at})}),S(t),h(e,t)};n(Zt,e=>{d(rt)&&e(Qt)}),k(()=>{At=se(kt,1,`obs-col svelte-ymjpra`,null,At,{narrow:!d(ct)}),Jt.disabled=d(vt)||d(rt),O(Yt,d(vt)?`Rendering report…`:`Memory report (PNG)`),x(Xt,`href`,`${he??``}/graph`)}),T(`pointerdown`,Dt,Tt),T(`pointermove`,Dt,Ct),u(`pointerleave`,Dt,wt),T(`click`,Jt,St),h(e,Et),oe(),b()}e([`pointerdown`,`pointermove`,`click`]);export{bi as component,Ve as universal};
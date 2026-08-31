var ti=Object.defineProperty;var ri=(n,e,t)=>e in n?ti(n,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):n[e]=t;var p=(n,e,t)=>ri(n,typeof e!="symbol"?e+"":e,t);import"./Bzak7iHL.js";import{d as fr,s as re,b as pe,o as ii,e as rt}from"./BffzNaS8.js";import{p as ct,c as A,r as E,j as T,t as oe,a as W,b as dt,f as J,k as hr,h as pr,g as m,u as Ie,o as Oe,d as j,e as Xt,s as S,bg as ni,n as ai}from"./LfElJ0kU.js";import{i as ee,b as si}from"./3OEMGmei.js";import{e as kt,a as Re,s as ke,r as oi}from"./CEZRBNoy.js";import{b as li}from"./XJznBkE6.js";import{p as q,s as ci,a as di}from"./DSKAVXWq.js";import{a as wt}from"./DOaVlKeo.js";import{e as ui}from"./vf6-3IFl.js";import{b as fi}from"./DOpcJKMG.js";import{s as It}from"./BJDZGVD9.js";import{t as hi,d as mr,N as ve,U as xt,F as ie,a as Ne,P as ce,b as Pe,D as Et,L as te,c as se,O as pi,e as mi}from"./a_C4imNz.js";import{p as gi}from"./CcSRZpDz.js";function vi(n,e){var r;const t=[];for(const s of n){const i=(((r=s.activation_path)!=null&&r.length?s.activation_path:s.retrieved)??[]).filter(e);if(i.length===0)continue;const o=i[i.length-1];t.push({targetId:o,pathIds:i})}return t}function bi(n,e=12){return[...n].sort((t,r)=>r.retention-t.retention||t.id.localeCompare(r.id)).slice(0,e).map(t=>({targetId:t.id,pathIds:[t.id]}))}function yi(n,e,t=5){var s;const r=new Map;for(const a of n){const i=((s=a.activation_path)!=null&&s.length?a.activation_path:a.retrieved)??[];for(const o of new Set(i))e(o)&&r.set(o,(r.get(o)??0)+1)}return[...r.entries()].map(([a,i])=>({id:a,recalls:i})).sort((a,i)=>i.recalls-a.recalls||a.id.localeCompare(i.id)).slice(0,t)}class _i{constructor(e,t={}){p(this,"bridge");p(this,"items",[]);p(this,"cursor",0);p(this,"ticks",0);p(this,"nextTick",0);p(this,"intervalFrames");p(this,"enabled",!0);p(this,"started",!1);this.bridge=e,this.intervalFrames=Math.max(60,t.intervalFrames??240)}setItems(e){this.items=e,this.cursor=0}get itemCount(){return this.items.length}setEnabled(e){this.enabled=e}tick(e){if(!this.enabled||this.items.length===0)return;if(this.ticks++,!this.started){this.started=!0,this.nextTick=this.ticks+45;return}if(this.ticks<this.nextTick)return;if(this.bridge.hasActiveEvent){this.nextTick=this.ticks+90;return}const t=this.items[this.cursor%this.items.length];this.cursor++;const r=this.bridge.replayRecall(t.targetId,t.pathIds,e);this.nextTick=this.ticks+this.intervalFrames+(r?0:30)}}var wi=J('<span class="hidden lg:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span>'),xi=J('<span class="text-[#a6dcff] tracking-widest whitespace-nowrap">CAPTURE</span>'),Bi=J('<span class="text-[#5dcaa5] whitespace-nowrap w-[6ch] text-right"> </span>'),Pi=J('<div class="absolute top-0 left-0 right-0 z-20 pointer-events-none" style="padding-top: env(safe-area-inset-top);"><div class="flex items-center justify-between gap-3 px-4 py-2 bg-gradient-to-b from-[#05060a]/85 to-transparent font-mono text-xs [font-variant-numeric:tabular-nums]"><div class="flex items-center gap-3 min-w-0 flex-1 overflow-hidden"><span class="text-[#5dcaa5] tracking-widest uppercase truncate"> </span> <span class="hidden md:inline text-[#ffffff]/[0.5] whitespace-nowrap"> </span></div> <div class="hidden sm:flex items-center gap-4"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!></div> <div class="flex items-center gap-3"><span class="text-[#ffffff]/[0.55] whitespace-nowrap"> </span> <!> <button class="text-[#ffffff]/[0.5] hover:text-[#5dcaa5] transition-colors cursor-pointer pointer-events-auto whitespace-nowrap" title="Copy shareable demo URL">[url]</button></div></div></div>');function Si(n,e){ct(e,!0);let t=q(e,"demoMode",3,"recall-path"),r=q(e,"seed",3,"vestige-observatory-v1"),s=q(e,"nodeCount",3,0),a=q(e,"edgeCount",3,0),i=q(e,"centerId",3,""),o=q(e,"frameCount",3,0),c=q(e,"fpsEstimate",3,0),l=q(e,"freezeFrame",3,null);q(e,"loading",3,!1),q(e,"error",3,"");function u(){const M=new URLSearchParams({demo:t(),seed:r()});l()!==null&&M.set("frame",String(l()));const Q=`${window.location.origin}${fi}/observatory?${M.toString()}`;navigator.clipboard.writeText(Q).catch(()=>{})}var d=Pi(),h=A(d),f=A(h),g=A(f),y=A(g,!0);E(g);var x=T(g,2),B=A(x);E(x),E(f);var v=T(f,2),F=A(v),z=A(F);E(F);var C=T(F,2);{var V=M=>{var Q=wi(),w=A(Q);E(Q),oe(L=>re(w,`center=${L??""}`),[()=>i().slice(0,8)]),W(M,Q)};ee(C,M=>{i()&&M(V)})}E(v);var H=T(v,2),R=A(H),O=A(R);E(R);var ae=T(R,2);{var ne=M=>{var Q=xi();W(M,Q)},de=M=>{var Q=Bi(),w=A(Q);E(Q),oe(()=>re(w,`${c()??""}fps`)),W(M,Q)};ee(ae,M=>{l()!==null?M(ne):c()>0&&M(de,1)})}var ue=T(ae,2);E(H),E(h),E(d),oe((M,Q)=>{re(y,t()),re(B,`seed=${M??""}${r().length>12?"…":""}`),re(z,`${s()??""} nodes · ${a()??""} edges`),re(O,`frame: ${Q??""}`)},[()=>r().slice(0,12),()=>String(o()).padStart(3," ")]),pe("click",ue,u),W(n,d),dt()}fr(["click"]);var ki=J('<div class="active-label svelte-8n8iia"> </div>'),Ii=J("<div></div>"),Ri=J('<div class="spine svelte-8n8iia"><!> <div class="track svelte-8n8iia"><!> <div class="playhead svelte-8n8iia"></div></div></div>');function Ei(n,e){ct(e,!0);let t=q(e,"steps",19,()=>[]),r=q(e,"frame",3,0),s=q(e,"loopFrames",3,720);const a=d=>d/s()*100;function i(d,h){const f=h-d;return f<-14||f>90?0:f<0?1+f/14:1-f/90}let o=Ie(()=>{let d="",h=.15;for(const f of t()){const g=i(f.beatFrame,r());g>h&&(h=g,d=f.label)}return d});var c=hr(),l=pr(c);{var u=d=>{var h=Ri(),f=A(h);{var g=v=>{var F=ki(),z=A(F,!0);E(F),oe(()=>re(z,m(o))),W(v,F)};ee(f,v=>{m(o)&&v(g)})}var y=T(f,2),x=A(y);kt(x,17,t,v=>v.beatFrame,(v,F)=>{var z=Ii();let C;oe((V,H,R)=>{C=Re(z,1,"tick svelte-8n8iia",null,C,V),It(z,`left: ${H??""}%; opacity: ${R??""}`),ke(z,"title",m(F).label)},[()=>({hot:i(m(F).beatFrame,r())>0,backward:m(F).kind===1}),()=>a(m(F).beatFrame),()=>.45+.55*i(m(F).beatFrame,r())]),W(v,z)});var B=T(x,2);E(y),E(h),oe(v=>It(B,`left: ${v??""}%`),[()=>a(r())]),W(d,h)};ee(l,d=>{t().length>0&&d(u)})}W(n,c),dt()}var Ai=J('<div><div class="k svelte-ssd7yu"> </div> <div class="v svelte-ssd7yu"> </div> <div class="s svelte-ssd7yu"> </div></div>');function Qt(n,e){ct(e,!0);let t=q(e,"frame",3,0),r=q(e,"fadeWindow",19,()=>[600,620,705,719]),s=q(e,"tone",3,"triumph");const a=(u,d,h)=>{const f=Math.min(1,Math.max(0,(h-u)/(d-u)));return f*f*(3-2*f)};let i=Ie(()=>a(r()[0],r()[1],t())*(1-a(r()[2],r()[3],t())));var o=hr(),c=pr(o);{var l=u=>{var d=Ai();let h;var f=A(d),g=A(f,!0);E(f);var y=T(f,2),x=A(y,!0);E(y);var B=T(y,2),v=A(B,!0);E(B),E(d),oe(()=>{h=Re(d,1,"verdict svelte-ssd7yu",null,h,{quarantine:s()==="quarantine"}),It(d,`opacity: ${m(i)??""}`),re(g,e.verdict.headline),re(x,e.verdict.causeLabel),re(v,e.verdict.receipt)}),W(u,d)};ee(c,u=>{m(i)>.001&&u(l)})}W(n,o),dt()}function Ci(n,e,t,r){const s=1/Math.tan(n/2),a=1/(t-r),i=new Float32Array(16);return i[0]=s/e,i[5]=s,i[10]=r*a,i[11]=-1,i[14]=r*t*a,i}function Mi(n,e,t){const[r,s,a]=n;let i=r-e[0],o=s-e[1],c=a-e[2],l=Math.hypot(i,o,c)||1;i/=l,o/=l,c/=l;let u=t[1]*c-t[2]*o,d=t[2]*i-t[0]*c,h=t[0]*o-t[1]*i;l=Math.hypot(u,d,h)||1,u/=l,d/=l,h/=l;const f=o*h-c*d,g=c*u-i*h,y=i*d-o*u,x=new Float32Array(16);return x[0]=u,x[1]=f,x[2]=i,x[4]=d,x[5]=g,x[6]=o,x[8]=h,x[9]=y,x[10]=c,x[12]=-(u*r+d*s+h*a),x[13]=-(f*r+g*s+y*a),x[14]=-(i*r+o*s+c*a),x[15]=1,x}function Fi(n,e){const t=new Float32Array(16);for(let r=0;r<4;r++)for(let s=0;s<4;s++)t[r*4+s]=n[s]*e[r*4]+n[4+s]*e[r*4+1]+n[8+s]*e[r*4+2]+n[12+s]*e[r*4+3];return t}function Zt(n,e,t,r=.35){const s=n*Math.PI*2,a=[Math.sin(s)*t,t*r,Math.cos(s)*t],i=Ci(50*Math.PI/180,e,.1,4e3),o=Mi(a,[0,0,0],[0,1,0]);let c=-a[0],l=-a[1],u=-a[2],d=Math.hypot(c,l,u)||1;c/=d,l/=d,u/=d;let h=l*0-u*1,f=u*0-c*0,g=c*1-l*0;d=Math.hypot(h,f,g)||1,h/=d,f/=d,g/=d;const y=f*u-g*l,x=g*c-h*u,B=h*l-f*c;return{viewProj:Fi(i,o),right:[h,f,g],up:[y,x,B],eye:a}}function Li(n){return n>=.7?"active":n>=.4?"dormant":n>=.1?"silent":"unavailable"}const Gi={active:"#10b981",dormant:"#f59e0b",silent:"#8b5cf6",unavailable:"#6b7280"},Bt={aha:"#FFD700",confusion:"#EF4444",failure:"#9CA3AF"};function Di(n){const e=new Set((n.tags??[]).map(t=>t.toLowerCase()));return e.has("aha")?Bt.aha:e.has("confusion")||e.has("weak-spot")?Bt.confusion:e.has("failure")||e.has("guardrail")?Bt.failure:null}function Jt(n){const e=/^#?([0-9a-fA-F]{6})$/.exec(n.trim());if(!e)return[107/255,114/255,128/255];const t=parseInt(e[1],16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}function Ti(n){const e=Di({tags:n.tags});return Jt(e||Gi[Li(n.retention)])}function zi(n){const t=[...n.nodes].sort((i,o)=>i.isCenter!==o.isCenter?i.isCenter?-1:1:i.id<o.id?-1:i.id>o.id?1:0).map((i,o)=>hi(i,o)),r=new Map;for(const i of t)r.set(i.id,i.index);const s=[];for(const i of n.edges){const o=r.get(i.source),c=r.get(i.target);o===void 0||c===void 0||o===c||s.push({sourceIndex:o,targetIndex:c,weight:i.weight,type:i.type})}const a=t.findIndex(i=>i.isCenter);return{nodes:t,edges:s,indexById:r,centerIndex:a<0?0:a}}function gr(n,e,t=120){const r=n.nodes.length,s=new Float32Array(r*ie);for(let a=0;a<r;a++){const i=n.nodes[a],o=a*ie,[c,l,u]=i.isCenter&&n.centerIndex===a?[0,0,0]:mr(a,r,t,e),d=i.isCenter?4.2:1.4+i.retention*1.8;s[o+ve.posRadius+0]=c,s[o+ve.posRadius+1]=l,s[o+ve.posRadius+2]=u,s[o+ve.posRadius+3]=d,s[o+ve.velRetention+3]=i.retention;const[h,f,g]=Ti(i);let y=0;i.isCenter&&(y|=Ne.isCenter),i.suppressed&&(y|=Ne.suppressed);const x=new Set(i.tags.map(B=>B.toLowerCase()));x.has("aha")&&(y|=Ne.isAha),(x.has("failure")||x.has("guardrail"))&&(y|=Ne.isFailure),(x.has("confusion")||x.has("weak-spot"))&&(y|=Ne.isConfusion),s[o+ve.colorFlags+0]=h,s[o+ve.colorFlags+1]=f,s[o+ve.colorFlags+2]=g,s[o+ve.colorFlags+3]=y}return{data:s,nodeCount:r}}function $t(n){const e=new Uint32Array(Math.max(1,n.edges.length)*xt);return n.edges.forEach((t,r)=>{e[r*xt]=t.sourceIndex,e[r*xt+1]=t.targetIndex}),e}const Ui=`
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
`,Oi=`
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
`;function Ni(n){return 60+n*60}function br(n,e,t=8,r={}){var u;const s=[...n.nodes].sort((d,h)=>d.id<h.id?-1:d.id>h.id?1:0),a=[...n.edges].sort((d,h)=>{const f=`${d.source}\0${d.target}\0${d.type}`,g=`${h.source}\0${h.target}\0${h.type}`;return f<g?-1:f>g?1:0}),i=r.centerId??n.center_id,o=gi(s,a,i,t,{preferCausal:r.preferCausal}),c=[];for(let d=0;d<o.beats.length;d++){const h=o.beats[d],f=e.indexById.get(h.nodeId);if(f===void 0)continue;const g=d>0?o.beats[d-1].nodeId:h.nodeId,y=e.indexById.get(g)??f,x=(((u=h.viaEdge)==null?void 0:u.type)??"").toLowerCase(),B=x==="causal"||x.includes("causal"),v=h.kind==="contradiction"||B;c.push({sourceIndex:y,targetIndex:f,beatFrame:Ni(d),kind:v?ce.backwardCause:ce.recall,beatKind:h.kind,nodeId:h.nodeId,label:h.node.label})}const l=new Uint32Array(Math.max(1,c.length)*Pe);return c.forEach((d,h)=>{l[h*Pe]=d.sourceIndex,l[h*Pe+1]=d.targetIndex,l[h*Pe+2]=d.beatFrame,l[h*Pe+3]=d.kind}),{data:l,steps:c,path:o}}const ji=24,er=300,Fe=128;class qi{constructor(e){p(this,"engine");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"cameraBuffer",null);p(this,"nodeBuffer",null);p(this,"edgeBuffer",null);p(this,"cameraData",new Float32Array(ji));p(this,"nodeCount",0);p(this,"simPipeline",null);p(this,"simBindGroup",null);p(this,"pathBuffer",null);p(this,"liveRetentionBuffer",null);p(this,"pathPipeline",null);p(this,"pathBindGroup",null);p(this,"pathStepCount",0);p(this,"graph",null);p(this,"pathSteps",[]);this.engine=e,e.addPass(this)}upload(e,t,r){var f,g,y,x;const s=this.engine.gpuDevice;if(!s)return;const a=(r==null?void 0:r.recallPath)??!0,i=zi(e);this.graph=i;const o=new Et({seed:t}),{data:c,nodeCount:l}=gr(i,o.state.rng);this.nodeCount=l,(f=this.nodeBuffer)==null||f.destroy(),this.nodeBuffer=s.createBuffer({label:"observatory-node-state",size:Math.max(c.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|GPUBufferUsage.VERTEX}),s.queue.writeBuffer(this.nodeBuffer,0,c.buffer);const u=$t(i);(g=this.edgeBuffer)==null||g.destroy(),this.edgeBuffer=s.createBuffer({label:"observatory-edge-index",size:u.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.edgeBuffer,0,u.buffer);const d=new Float32Array(Math.max(l,4));for(let B=0;B<l;B++)d[B]=Math.max(.001,i.nodes[B].retention);(y=this.liveRetentionBuffer)==null||y.destroy(),this.liveRetentionBuffer=s.createBuffer({label:"observatory-live-retention",size:d.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.liveRetentionBuffer,0,d.buffer);const h=a?br(e,i):{steps:[],data:new Uint32Array(4)};this.pathSteps=h.steps,(x=this.pathBuffer)==null||x.destroy(),this.pathBuffer=s.createBuffer({label:"observatory-path-steps",size:Fe*Pe*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.pathBuffer,0,h.data.buffer,0,Math.min(h.data.byteLength,Fe*Pe*4)),this.pathStepCount=Math.min(this.pathSteps.length,Fe),this.engine.params[2]=l,this.engine.params[3]=i.edges.length,this.engine.params[4]=this.pathSteps.length,this.cameraBuffer||(this.cameraBuffer=s.createBuffer({label:"observatory-camera",size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.createPipeline(s)}setPathSteps(e,t){var a;const r=this.engine.gpuDevice;if(!r)return;this.pathSteps=t;const s=Fe*Pe*4;if(this.pathBuffer&&e.byteLength<=s){this.pathStepCount=Math.min(t.length,Fe),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,e.byteLength),this.engine.params[4]=this.pathStepCount;return}this.pathStepCount=Math.min(t.length,Fe),(a=this.pathBuffer)==null||a.destroy(),this.pathBuffer=r.createBuffer({label:"observatory-path-steps",size:s,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.pathBuffer,0,e.buffer,0,Math.min(e.byteLength,s)),this.engine.params[4]=this.pathStepCount,this.createPipeline(r)}setEdges(e){var s;const t=this.engine.gpuDevice;if(!t||!this.graph)return;this.graph.edges=e;const r=$t(this.graph);(s=this.edgeBuffer)==null||s.destroy(),this.edgeBuffer=t.createBuffer({label:"observatory-edge-index",size:Math.max(r.byteLength,8),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.edgeBuffer,0,r.buffer),this.engine.params[3]=e.length,this.createPipeline(t)}uploadLiveRetention(e){const t=this.engine.gpuDevice;if(!t||!this.liveRetentionBuffer)return;const r=Math.min(e.length,this.nodeCount);r<=0||t.queue.writeBuffer(this.liveRetentionBuffer,0,e.buffer,0,r*4)}getFossilLightSources(){return!this.nodeBuffer||!this.cameraBuffer||this.nodeCount<=0?null:{nodeBuffer:this.nodeBuffer,cameraBuffer:this.cameraBuffer,nodeCount:this.nodeCount}}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;if(this.pathBuffer){const r=e.createShaderModule({label:"observatory-simulate",code:Oi});this.simPipeline=e.createComputePipeline({label:"observatory-recall-sim",layout:"auto",compute:{module:r,entryPoint:"recall_sim"}});const s=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeBuffer}},{binding:2,resource:{buffer:this.pathBuffer}}];this.edgeBuffer&&s.push({binding:3,resource:{buffer:this.edgeBuffer}}),this.liveRetentionBuffer&&s.push({binding:4,resource:{buffer:this.liveRetentionBuffer}}),this.simBindGroup=e.createBindGroup({label:"observatory-recall-sim-bind",layout:this.simPipeline.getBindGroupLayout(0),entries:s})}const t=e.createShaderModule({label:"observatory-render-nodes",code:Ui});if(this.pipeline=e.createRenderPipeline({label:"observatory-nodes",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=e.createBindGroup({label:"observatory-nodes-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]}),this.pathBuffer){const r=e.createShaderModule({label:"observatory-render-path",code:vr});this.pathPipeline=e.createRenderPipeline({label:"observatory-path",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.pathBindGroup=e.createBindGroup({label:"observatory-path-bind",layout:this.pathPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}},{binding:3,resource:{buffer:this.pathBuffer}}]})}}compute(e){const t=this.engine.gpuDevice;if(!t||!this.cameraBuffer)return;const r=this.engine.params[6]||1,s=this.engine.params[7]||1,a=this.engine.params[1],i=Zt(a,r/s,er);if(this.cameraData.set(i.viewProj,0),this.cameraData[16]=i.right[0],this.cameraData[17]=i.right[1],this.cameraData[18]=i.right[2],this.cameraData[19]=0,this.cameraData[20]=i.up[0],this.cameraData[21]=i.up[1],this.cameraData[22]=i.up[2],this.cameraData[23]=0,t.queue.writeBuffer(this.cameraBuffer,0,this.cameraData),this.simPipeline&&this.simBindGroup&&this.nodeCount>0){const o=e.beginComputePass({label:"observatory-recall-sim"});o.setPipeline(this.simPipeline),o.setBindGroup(0,this.simBindGroup),o.dispatchWorkgroups(Math.ceil(this.nodeCount/64)),o.end()}}render(e){!this.pipeline||!this.bindGroup||this.nodeCount===0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount),this.pathPipeline&&this.pathBindGroup&&this.pathStepCount>0&&(e.setPipeline(this.pathPipeline),e.setBindGroup(0,this.pathBindGroup),e.draw(6,this.pathStepCount)))}get nodeStateBuffer(){return this.nodeBuffer}get cameraUniformBuffer(){return this.cameraBuffer}get nodeCountValue(){return this.nodeCount}get pathStepMeta(){return this.pathSteps}async pickAt(e,t){const r=this.engine.gpuDevice;if(!r||!this.nodeBuffer||!this.graph||this.nodeCount===0)return null;const s=this.nodeCount*ie*4,a=r.createBuffer({label:"observatory-pick-staging",size:s,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),i=r.createCommandEncoder({label:"observatory-pick-copy"});i.copyBufferToBuffer(this.nodeBuffer,0,a,0,s),r.queue.submit([i.finish()]);let o;try{await a.mapAsync(GPUMapMode.READ),o=new Float32Array(a.getMappedRange().slice(0))}catch{return a.destroy(),null}a.unmap(),a.destroy();const c=this.engine.params[6]||1,l=this.engine.params[7]||1,u=this.engine.params[1],d=Zt(u,c/l,er).viewProj,h=1/Math.tan(50*Math.PI/360);let f=-1,g=1/0;for(let y=0;y<this.nodeCount;y++){const x=y*ie+ve.posRadius,B=o[x],v=o[x+1],F=o[x+2],z=o[x+3],C=d[3]*B+d[7]*v+d[11]*F+d[15];if(C<=0)continue;const V=(d[0]*B+d[4]*v+d[8]*F+d[12])/C,H=(d[1]*B+d[5]*v+d[9]*F+d[13])/C,R=Math.max(z*h/C,.012),O=Math.hypot(V-e,H-t)/R;O<1.6&&O<g&&(g=O,f=y)}return f<0?null:{index:f,id:this.graph.nodes[f].id}}dispose(){var e,t,r,s,a;(e=this.nodeBuffer)==null||e.destroy(),(t=this.edgeBuffer)==null||t.destroy(),(r=this.cameraBuffer)==null||r.destroy(),(s=this.pathBuffer)==null||s.destroy(),(a=this.liveRetentionBuffer)==null||a.destroy(),this.nodeBuffer=null,this.edgeBuffer=null,this.cameraBuffer=null,this.pathBuffer=null,this.liveRetentionBuffer=null,this.pipeline=null,this.bindGroup=null,this.simPipeline=null,this.simBindGroup=null,this.pathPipeline=null,this.pathBindGroup=null}}const it=16,je=4,tr=110,Vi=180,Wi=.7,Hi=.2,Ki=360,Yi=18;function Xi(n){if(n.edges.length>0){const e=n.centerIndex,t=n.edges.filter(r=>r.sourceIndex===e||r.targetIndex===e);if(t.length>0){let r=-1,s=-1;for(const a of t){const i=a.sourceIndex===e?a.targetIndex:a.sourceIndex,o=n.nodes[i];o&&o.retention>s&&(s=o.retention,r=i)}if(r>=0)return r}}for(let e=0;e<n.nodes.length;e++)if(e!==n.centerIndex)return e;return n.centerIndex}function Qi(n,e,t=8192){const r=Xi(n),a=n.nodes[r].id,i=rr(n,r),c=new Et({seed:e+":birth:"+a}).state.rng,l=new Float32Array(t*it),u=Math.floor(t*Wi),d=Math.floor(t*Hi),h=t-u-d;for(let B=0;B<u;B++){const v=B*it,[F,z,C]=mr(B,u,tr+c()*(Vi-tr),c);l[v+0]=i[0]+F,l[v+1]=i[1]+z,l[v+2]=i[2]+C,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const f=n.edges.filter(B=>B.sourceIndex===r||B.targetIndex===r);for(let B=0;B<d;B++){const v=(u+B)*it;if(f.length===0)continue;const F=B%f.length,z=f[F],C=z.sourceIndex===r?z.targetIndex:z.sourceIndex,V=rr(n,C),H=V[0]-i[0],R=V[1]-i[1],O=V[2]-i[2],ae=Math.sqrt(H*H+R*R+O*O)||1,ne=B/Math.max(1,d)*2+.5,de=c()*30,ue=-R*de/ae,M=H*de/ae,Q=0;l[v+0]=i[0]+H/ae*ne*80+ue,l[v+1]=i[1]+R/ae*ne*80+M,l[v+2]=i[2]+O/ae*ne*80+Q,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const g=300;for(let B=0;B<h;B++){const v=(u+d+B)*it,F=c()*Math.PI*2,z=c()*120;l[v+0]=i[0]+Math.cos(F)*z,l[v+1]=i[1]+Math.sin(F)*z,l[v+2]=i[2]+g*.6+c()*40,l[v+3]=c(),l[v+4]=i[0],l[v+5]=i[1],l[v+6]=i[2],l[v+7]=1+c()*1.8,l[v+8]=.55,l[v+9]=.32,l[v+10]=1,l[v+11]=c(),l[v+12]=0,l[v+13]=0,l[v+14]=0,l[v+15]=0}const y=Zi(n,r),x=Ji();return{targetIndex:r,targetNodeId:a,particles:l,edgeSteps:y,timeline:x}}function rr(n,e){const t=n.nodes[e],r=n.nodes.length;if(t.isCenter&&n.centerIndex===e)return[0,0,0];const s=Math.PI*(3-Math.sqrt(5)),a=1-e/(r-1||1)*2,i=Math.sqrt(1-a*a),o=s*e,c=120,l=(e*7+3)%100/100*.1*c-.05*c,u=(e*13+7)%100/100*.1*c-.05*c,d=(e*17+11)%100/100*.1*c-.05*c;return[Math.cos(o)*i*c+l,a*c+u,Math.sin(o)*i*c+d]}function Zi(n,e){const t=n.edges.filter(a=>a.sourceIndex===e||a.targetIndex===e),r=t.length;if(r===0)return new Uint32Array(0);const s=new Uint32Array(r*je);for(let a=0;a<r;a++){const i=t[a],o=i.sourceIndex===e?i.targetIndex:i.sourceIndex,c=Ki+a*Yi;s[a*je+0]=e,s[a*je+1]=o,s[a*je+2]=c,s[a*je+3]=0}return s}function Ji(){return[{label:"latent trace condensing",startFrame:60,endFrame:239},{label:"engram coalescence",startFrame:240,endFrame:329},{label:"memory ignition",startFrame:330,endFrame:359},{label:"associations engrave",startFrame:360,endFrame:509},{label:"stabilization",startFrame:510,endFrame:659}]}const $i=`
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
`,en=16,tn=6,rn=330,nn=359,an=360;class sn{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"active");p(this,"computePipeline",null);p(this,"computeBindGroup",null);p(this,"particleBuffer",null);p(this,"particleCount",0);p(this,"renderPipeline",null);p(this,"renderBindGroup",null);p(this,"haloPipeline",null);p(this,"haloBindGroup",null);p(this,"haloIndexBuffer",null);p(this,"engravePipeline",null);p(this,"engraveBindGroup",null);p(this,"engraveBuffer",null);p(this,"engraveStepCount",0);p(this,"timeline",[]);p(this,"birthPlan",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.active=!1,this.engine.addPass(this)}get engraveSteps(){var e;return((e=this.birthPlan)==null?void 0:e.edgeSteps)??new Uint32Array(0)}upload(e){var a,i;const t=this.engine.gpuDevice;if(!t||!this.nodeRenderer.nodeStateBuffer)return;const r=this.nodeRenderer.graph;if(!r)return;this.birthPlan=Qi(r,e),this.timeline=this.birthPlan.timeline;const s=this.birthPlan.particles.length/en;this.particleCount=s,(a=this.particleBuffer)==null||a.destroy(),this.particleBuffer=t.createBuffer({label:"observatory-birth-particles",size:this.birthPlan.particles.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.particleBuffer,0,this.birthPlan.particles.buffer),(i=this.engraveBuffer)==null||i.destroy(),this.engraveStepCount=this.birthPlan.edgeSteps.length/4,this.engraveStepCount>0&&(this.engraveBuffer=t.createBuffer({label:"observatory-birth-engrave",size:this.birthPlan.edgeSteps.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),t.queue.writeBuffer(this.engraveBuffer,0,this.birthPlan.edgeSteps.buffer)),this.createComputePipeline(t),this.createRenderPipeline(t),this.createHaloPipeline(t),this.createEngravePipeline(t)}createComputePipeline(e){const t=e.createShaderModule({label:"observatory-birth-compute",code:$i});this.computePipeline=e.createComputePipeline({label:"observatory-birth-compute-pipeline",layout:"auto",compute:{module:t,entryPoint:"birth_compute"}});const r=[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.particleBuffer}}];this.computeBindGroup=e.createBindGroup({label:"observatory-birth-compute-bind",layout:this.computePipeline.getBindGroupLayout(0),entries:r})}createRenderPipeline(e){const r=e.createShaderModule({label:"observatory-birth-render",code:`
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
`});this.haloPipeline=e.createRenderPipeline({label:"observatory-birth-halo",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}});const s=this.nodeRenderer.cameraUniformBuffer;this.haloBindGroup=e.createBindGroup({label:"observatory-birth-halo-bind",layout:this.haloPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}}]})}createEngravePipeline(e){if(this.engraveStepCount===0||!this.engraveBuffer)return;const t=e.createShaderModule({label:"observatory-birth-engrave",code:vr});this.engravePipeline=e.createRenderPipeline({label:"observatory-birth-engrave-pipeline",layout:"auto",vertex:{module:t,entryPoint:"vs_main"},fragment:{module:t,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.engraveBindGroup=e.createBindGroup({label:"observatory-birth-engrave-bind",layout:this.engravePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.cameraUniformBuffer}},{binding:2,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:3,resource:{buffer:this.engraveBuffer}}]})}compute(e,t){const r=this.engine.params[9];if(this.active=r===1,!this.active||!this.computePipeline||!this.computeBindGroup)return;const s=e.beginComputePass({label:"observatory-birth-compute"});s.setPipeline(this.computePipeline),s.setBindGroup(0,this.computeBindGroup),s.dispatchWorkgroups(Math.ceil(this.particleCount/64)),s.end()}render(e,t){this.active&&(this.renderPipeline&&this.renderBindGroup&&this.particleCount>0&&(e.setPipeline(this.renderPipeline),e.setBindGroup(0,this.renderBindGroup),e.draw(tn,this.particleCount)),this.haloPipeline&&this.haloBindGroup&&t>=rn&&t<=nn&&(e.setPipeline(this.haloPipeline),e.setBindGroup(0,this.haloBindGroup),e.draw(4,this.nodeRenderer.nodeCountValue)),this.engravePipeline&&this.engraveBindGroup&&this.engraveStepCount>0&&t>=an&&(e.setPipeline(this.engravePipeline),e.setBindGroup(0,this.engraveBindGroup),e.draw(6,this.engraveStepCount)))}dispose(){var e,t,r,s,a,i,o,c,l,u,d;(e=this.particleBuffer)==null||e.destroy(),this.particleBuffer=null,(r=(t=this.computePipeline)==null?void 0:t.destroy)==null||r.call(t),this.computePipeline=null,this.computeBindGroup=null,(a=(s=this.renderPipeline)==null?void 0:s.destroy)==null||a.call(s),this.renderPipeline=null,this.renderBindGroup=null,(o=(i=this.haloPipeline)==null?void 0:i.destroy)==null||o.call(i),this.haloPipeline=null,this.haloBindGroup=null,(c=this.haloIndexBuffer)==null||c.destroy(),this.haloIndexBuffer=null,(u=(l=this.engravePipeline)==null?void 0:l.destroy)==null||u.call(l),this.engravePipeline=null,this.engraveBindGroup=null,(d=this.engraveBuffer)==null||d.destroy(),this.engraveBuffer=null}}function on(n){const e=n.hopSlot.toFixed(1),t=n.causeDepth.toFixed(1);return`
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
`}const ln=2;class cn{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"waveBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.waveBuffer)==null||r.destroy(),this.waveBuffer=e.createBuffer({label:"observatory-rescue-wave",size:Math.max(4,this.plan.waveData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.waveBuffer,0,this.plan.waveData.buffer);const t=e.createShaderModule({label:"observatory-rescue-choreo",code:on(this.plan.consts)});this.pipeline=e.createComputePipeline({label:"observatory-rescue-choreo",layout:"auto",compute:{module:t,entryPoint:"rescue_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-rescue-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.waveBuffer}}]})}compute(e){if(this.engine.params[9]!==ln||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-rescue-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.waveBuffer)==null||e.destroy(),this.waveBuffer=null,this.pipeline=null,this.bindGroup=null}}const dn=4,yr=90,un=138,fn=28,ot=260,hn=514,ir=560,_r=600,Se=65535,pn=48,mn={causal:0,temporal:1,shared_concepts:2,complementary:3,semantic:4};function wr(n,e){const t=new Et({seed:e});return gr(n,t.state.rng).data}function gn(n){const e=new Uint32Array(n.nodes.length);for(const t of n.edges)e[t.sourceIndex]++,e[t.targetIndex]++;return e}function vn(n,e){const t=n.nodes.length;if(t===0)return-1;const r=gn(n),s=i=>{const o=n.nodes[i],c=new Set(o.tags.map(f=>f.toLowerCase()));let l=0;(c.has("failure")||c.has("guardrail"))&&(l+=3),(c.has("confusion")||c.has("weak-spot"))&&(l+=2),l+=Math.min(r[i],8)/8;const u=e[i*ie+0],d=e[i*ie+1],h=e[i*ie+2];return Math.sqrt(u*u+d*d+h*h)>=54&&(l+=.5),l},a=[i=>i!==n.centerIndex&&!n.nodes[i].suppressed&&r[i]>=2,i=>i!==n.centerIndex&&!n.nodes[i].suppressed,i=>i!==n.centerIndex,()=>!0];for(const i of a){let o=-1,c=-1/0;for(let l=0;l<t;l++){if(!i(l))continue;const u=s(l);u>c&&(c=u,o=l)}if(o>=0)return o}return-1}function bn(n,e){const t=n.nodes.length,r=new Uint16Array(t).fill(Se),s=new Int32Array(t).fill(-1);if(e<0||e>=t)return{depths:r,parents:s};const a=Array.from({length:t},()=>[]);for(const o of n.edges){const c=mn[o.type]??5;a[o.sourceIndex].push({nbr:o.targetIndex,rank:c}),a[o.targetIndex].push({nbr:o.sourceIndex,rank:c})}for(const o of a)o.sort((c,l)=>c.rank-l.rank||c.nbr-l.nbr);r[e]=0;const i=[e];for(let o=0;o<i.length;o++){const c=i[o];for(const{nbr:l}of a[c])r[l]===Se&&(r[l]=r[c]+1,i.push(l))}for(let o=0;o<t;o++)if(!(r[o]===Se||r[o]===0)){for(const{nbr:c}of a[o])if(r[c]===r[o]-1){s[o]=c;break}}return{depths:r,parents:s}}function yn(n,e,t,r){const s=new Map;for(const a of n.nodes)s.set(a.id,a.createdAt);for(const a of[3,2,1]){const i=[];for(let f=0;f<e.nodes.length;f++){if(f===e.centerIndex||f===r)continue;const g=t[f];g===Se||g<a||i.push(f)}if(i.length===0)continue;let o=i.filter(f=>e.nodes[f].retention<=.45);o.length===0&&(o=i);const c=new Map;let l=1/0,u=-1/0;for(const f of o){const g=s.get(e.nodes[f].id),y=g?Date.parse(g):NaN;Number.isFinite(y)&&(c.set(f,y),y<l&&(l=y),y>u&&(u=y))}const d=f=>{const g=c.get(f);return g===void 0?0:u===l?1:(u-g)/(u-l)},h=f=>2*(1-e.nodes[f].retention)+.5*Math.min(t[f],6)/6+.5*d(f);return o.sort((f,g)=>{const y=h(f),x=h(g);return x!==y?x-y:t[g]!==t[f]?t[g]-t[f]:f-g}),{index:o[0],depth:t[o[0]]}}return{index:-1,depth:0}}function _n(n,e,t,r,s){const a=n[t*ie+0],i=n[t*ie+1],o=n[t*ie+2],c=[];for(let l=0;l<e;l++){if(l===t||l===r||l===s)continue;const u=n[l*ie+0]-a,d=n[l*ie+1]-i,h=n[l*ie+2]-o;c.push({i:l,d2:u*u+d*d+h*h})}return c.sort((l,u)=>l.d2-u.d2||l.i-u.i),c.slice(0,dn).map(l=>l.i)}function lt(n){const e=Math.max(1,n);return Math.min(84,Math.max(14,Math.floor(252/e)))}function wn(n,e){return Math.min(ot+e*n,hn)}function nr(n){return un+fn*n}function he(n){return n.length>64?n.slice(0,64)+"…":n}const be=4;function Ee(n){const e=new Uint32Array(n);return e.fill(Se),{viable:!1,failureIndex:-1,causeIndex:-1,lookalikeIndices:[],hopDepths:new Uint16Array(n).fill(Se),causeDepth:0,hopSlot:lt(3),waveData:e,pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"root cause found",causeLabel:"",failureLabel:"",causeDate:"",hops:0,k:0,receipt:""},consts:{hopSlot:lt(3),causeDepth:3}}}function xn(n,e,t,r){var Q;if(r)return Bn(n,e,r);const s=e.nodes.length;if(s===0)return Ee(0);const a=wr(e,t),i=vn(e,a);if(i<0)return Ee(s);const{depths:o,parents:c}=bn(e,i),l=yn(n,e,o,i);if(l.index<0){const w=Ee(s);return w.failureIndex=i,w.hopDepths=o,w}const u=l.index,d=Math.max(1,l.depth),h=lt(d),f=w=>wn(w,h),g=_n(a,s,i,u,e.centerIndex),y=g.length,x=new Uint32Array(s);for(let w=0;w<s;w++){let L=o[w]&65535;w===i&&(L|=65536),w===u&&(L|=1<<17),x[w]=L}g.forEach((w,L)=>{x[w]|=1<<18|L<<19});const B=[];g.forEach((w,L)=>{B.push({src:i,dst:w,bf:nr(L),kind:ce.probe,beatKind:"probe"})});const v=[];{let w=u;for(;w!==i&&w>=0&&c[w]>=0;)v.push(w),w=c[w]}const F=new Set(v),z=[];for(let w=0;w<s;w++){if(w===i||F.has(w))continue;const L=o[w];L===Se||L<1||L>d||c[w]<0||z.push(w)}z.sort((w,L)=>o[w]-o[L]||w-L);const C=[...v.slice().reverse(),...z].slice(0,pn);C.sort((w,L)=>o[w]-o[L]||w-L);for(const w of C)B.push({src:c[w],dst:w,bf:f(o[w]),kind:ce.backwardCause,beatKind:"wave"});B.push({src:u,dst:i,bf:ir,kind:ce.backwardCause,beatKind:"arc"});const V=new Uint32Array(Math.max(1,B.length)*be),H=[];B.forEach((w,L)=>{V[L*be+0]=w.src,V[L*be+1]=w.dst,V[L*be+2]=w.bf,V[L*be+3]=w.kind,H.push({sourceIndex:w.src,targetIndex:w.dst,beatFrame:w.bf,kind:w.kind,beatKind:w.beatKind,nodeId:e.nodes[w.dst].id,label:he(e.nodes[w.dst].label)})});const R=he(e.nodes[i].label),O=he(e.nodes[u].label),ae=[],ne=(w,L,ft,Ge)=>{ae.push({sourceIndex:i,targetIndex:i,beatFrame:w,kind:L,beatKind:"rescue",nodeId:Ge,label:ft})};ne(yr,1,`failure: ${R}`,e.nodes[i].id),g.forEach((w,L)=>{ne(nr(L),0,`lookalike ✗ · ${he(e.nodes[w].label)}`,e.nodes[w].id)}),ne(f(1),1,"reaching backward through time","rescue-wave-start"),d>=2&&f(d)!==f(1)&&ne(f(d),1,`scrubbing past · ${d} hops`,"rescue-wave-deep"),ne(ir,1,`causal arc · ${O}`,e.nodes[u].id),ne(_r,1,"root cause found","rescue-verdict");const de=((Q=n.nodes.find(w=>w.id===e.nodes[u].id))==null?void 0:Q.createdAt)??"",ue=de?de.slice(0,10):"",M={headline:"root cause found",causeLabel:O,failureLabel:R,causeDate:ue,hops:d,k:y,receipt:`${d} hops back · ${ue} · vector search: 0 for ${y}`};return{viable:!0,failureIndex:i,causeIndex:u,lookalikeIndices:g,hopDepths:o,causeDepth:d,hopSlot:h,waveData:x,pathData:V,pathMetas:H,spineBeats:ae,verdict:M,consts:{hopSlot:h,causeDepth:d}}}function Bn(n,e,t){var V,H;const r=e.nodes.length,s=e.indexById.get(t.failureId)??-1,a=t.pathIds??[];if(s<0||a.length<2||a[a.length-1]!==t.failureId||new Set(a).size!==a.length)return Ee(r);const i=a.map(R=>e.indexById.get(R));if(i.some(R=>R===void 0))return Ee(r);const o=i,c=o[0];if(c===s)return Ee(r);const l=t.candidates.find(R=>R.memoryId===a[0]);if(!l)return Ee(r);const u=new Uint16Array(r);u.fill(Se),u[s]=0,o.forEach((R,O)=>{u[R]=o.length-1-O});const d=new Uint32Array(r);d[s]=65536,o.slice(0,-1).forEach(R=>{d[R]=u[R]}),d[c]|=1<<17;const h=o.length-1,f=lt(h),g=he(e.nodes[c].label),y=he(e.nodes[s].label),x=((H=(V=n.nodes.find(R=>R.id===l.memoryId))==null?void 0:V.createdAt)==null?void 0:H.slice(0,10))??"",B=new Uint32Array((o.length-1)*be),v=o.slice(0,-1).map((R,O)=>{const ae=o[O+1],ne=ot+O*f;return B[O*be]=R,B[O*be+1]=ae,B[O*be+2]=ne,B[O*be+3]=ce.backwardCause,{sourceIndex:R,targetIndex:ae,beatFrame:ne,kind:ce.backwardCause,beatKind:"receipt-path",nodeId:a[O+1],label:`recorded path · ${he(e.nodes[ae].label)}`}}),F=l.sharedEntities.length?l.sharedEntities.join(", "):"recorded entity",z=l.similarityRank===null?"rank unavailable":`embedding rank #${l.similarityRank}`,C=[{sourceIndex:s,targetIndex:s,beatFrame:yr,kind:1,beatKind:"receipt-failure",nodeId:t.failureId,label:`recorded failure · ${y}`},{sourceIndex:s,targetIndex:s,beatFrame:ot,kind:1,beatKind:"receipt-join",nodeId:"receipt-join",label:`shared entity · ${F}`},{sourceIndex:c,targetIndex:s,beatFrame:ot+(h-1)*f,kind:ce.backwardCause,beatKind:"receipt-candidate",nodeId:l.memoryId,label:`candidate · ${g}`},{sourceIndex:c,targetIndex:c,beatFrame:_r,kind:1,beatKind:"receipt-verdict",nodeId:"receipt-verdict",label:"candidate cause found"}];return{viable:!0,failureIndex:s,causeIndex:c,lookalikeIndices:[],hopDepths:u,causeDepth:h,hopSlot:f,waveData:d,pathData:B,pathMetas:v,spineBeats:C,verdict:{headline:"candidate cause found",causeLabel:g,failureLabel:y,causeDate:x,hops:h,k:0,receipt:`${l.ageDays.toFixed(1)}d back · ${F} · ${z}`},consts:{hopSlot:f,causeDepth:h}}}const Pn=`
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
`,Sn=3;class kn{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"horizonBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.horizonBuffer)==null||r.destroy(),this.horizonBuffer=e.createBuffer({label:"observatory-forgetting-horizon",size:Math.max(4,this.plan.horizonData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.horizonBuffer,0,this.plan.horizonData.buffer);const t=e.createShaderModule({label:"observatory-forgetting-choreo",code:Pn});this.pipeline=e.createComputePipeline({label:"observatory-forgetting-choreo",layout:"auto",compute:{module:t,entryPoint:"forgetting_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-forgetting-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.horizonBuffer}}]})}compute(e){if(this.engine.params[9]!==Sn||!this.pipeline||!this.bindGroup)return;const t=this.nodeRenderer.nodeCountValue;if(t===0)return;const r=e.beginComputePass({label:"observatory-forgetting-choreo"});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.dispatchWorkgroups(Math.ceil(t/64)),r.end()}dispose(){var e;(e=this.horizonBuffer)==null||e.destroy(),this.horizonBuffer=null,this.pipeline=null,this.bindGroup=null}}const In=318,Rn=60,xr=3,En=132,An=60,Cn=540,Mn=660;function Fn(n){const e=[];for(let s=0;s<n.nodes.length;s++)s!==n.centerIndex&&e.push(s);e.sort((s,a)=>n.nodes[s].retention-n.nodes[a].retention||s-a);const t=e.length;if(t===0)return[];const r=Math.min(t,Math.max(Math.min(xr,t),Math.round(.25*t)));return e.slice(0,r)}function Ln(n,e){const t=new Uint32Array(n.nodes.length);for(const a of n.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>2*n.nodes[a].retention+Math.min(t[a],8)/8;return e.slice().sort((a,i)=>r(i)-r(a)||a-i).slice(0,Math.min(xr,e.length))}function ar(n){return In+Rn*n}const qe=4;function Gn(n){return{viable:!1,driftingIndices:[],rescuedIndices:[],horizonData:new Uint32Array(n),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[]}}function Dn(n){const e=n.nodes.length,t=Fn(n);if(e<2||t.length<1)return Gn(e);const r=Ln(n,t),s=t.length,a=new Uint32Array(e);t.forEach((h,f)=>{const g=Math.round(255*f/Math.max(1,s-1));a[h]=g&255|256}),r.forEach((h,f)=>{a[h]|=512|f<<10});const i=new Uint32Array(Math.max(1,r.length)*qe),o=[];r.forEach((h,f)=>{const g=ar(f);i[f*qe+0]=n.centerIndex,i[f*qe+1]=h,i[f*qe+2]=g,i[f*qe+3]=ce.recall,o.push({sourceIndex:n.centerIndex,targetIndex:h,beatFrame:g,kind:ce.recall,beatKind:"recall",nodeId:n.nodes[h].id,label:he(n.nodes[h].label)})});const c=[],l=(h,f,g,y)=>{c.push({sourceIndex:n.centerIndex,targetIndex:n.centerIndex,beatFrame:h,kind:f,beatKind:"horizon",nodeId:y,label:g})},u=new Set(r),d=t.filter(h=>!u.has(h)).slice(0,3);return d.forEach((h,f)=>{const g=Math.round(n.nodes[h].retention*100);l(En+An*f,1,`fading: ${he(n.nodes[h].label)} · retention ${g}%`,n.nodes[h].id)}),r.forEach((h,f)=>{l(ar(f),0,`recalled: ${he(n.nodes[h].label)}`,n.nodes[h].id)}),d.length>0&&l(Cn,1,"the unrecalled sink · nothing is deleted","horizon-sink"),l(Mn,0,"every memory still retrievable","horizon-retrievable"),{viable:!0,driftingIndices:t,rescuedIndices:r,horizonData:a,pathData:i,pathMetas:o,spineBeats:c}}const Tn=`
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
`,zn=4;class Br{constructor(e){p(this,"engine");p(this,"nodeRenderer");p(this,"plan");p(this,"pipeline",null);p(this,"bindGroup",null);p(this,"fireBuffer",null);this.engine=e.engine,this.nodeRenderer=e.nodeRenderer,this.plan=e.plan,this.engine.addPass(this)}upload(){var r;const e=this.engine.gpuDevice;if(!e||!this.engine.paramsBuffer||!this.plan.viable||!this.nodeRenderer.nodeStateBuffer||this.nodeRenderer.nodeCountValue===0)return;(r=this.fireBuffer)==null||r.destroy(),this.fireBuffer=e.createBuffer({label:"observatory-firewall-fire",size:Math.max(4,this.plan.fireData.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.fireBuffer,0,this.plan.fireData.buffer);const t=e.createShaderModule({label:"observatory-firewall-choreo",code:Tn});this.pipeline=e.createComputePipeline({label:"observatory-firewall-choreo",layout:"auto",compute:{module:t,entryPoint:"firewall_choreo"}}),this.bindGroup=e.createBindGroup({label:"observatory-firewall-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.nodeRenderer.nodeStateBuffer}},{binding:2,resource:{buffer:this.fireBuffer}}]})}rearm(e){if(this.plan=e,!!this.engine.gpuDevice){if(!e.viable){this.pipeline=null,this.bindGroup=null;return}this.upload()}}get armed(){return this.plan.viable&&!!this.pipeline&&!!this.bindGroup}compute(e){const t=this.engine.params[9]===zn,r=this.engine.params[12]===1;if(!t&&!r||!this.pipeline||!this.bindGroup)return;const s=this.nodeRenderer.nodeCountValue;if(s===0)return;const a=e.beginComputePass({label:"observatory-firewall-choreo"});a.setPipeline(this.pipeline),a.setBindGroup(0,this.bindGroup),a.dispatchWorkgroups(Math.ceil(s/64)),a.end()}dispose(){var e;(e=this.fireBuffer)==null||e.destroy(),this.fireBuffer=null,this.pipeline=null,this.bindGroup=null}}const Un=90,On=150,Nn=144,jn=330,qn=345,Vn=21,Wn=6,Hn=480,Kn=["failure","guardrail","confusion"];function Yn(n){const e=n.nodes.length;if(e===0)return-1;const t=new Uint32Array(e);for(const a of n.edges)t[a.sourceIndex]++,t[a.targetIndex]++;const r=a=>n.nodes[a].tags.some(i=>Kn.includes(i.toLowerCase())),s=[a=>a!==n.centerIndex&&!n.nodes[a].suppressed&&r(a),a=>a!==n.centerIndex&&!n.nodes[a].suppressed&&t[a]<=1,a=>a!==n.centerIndex&&!n.nodes[a].suppressed,a=>a!==n.centerIndex];for(const a of s){let i=-1;for(let o=0;o<e;o++)a(o)&&(i<0||n.nodes[o].retention<n.nodes[i].retention)&&(i=o);if(i>=0)return i}return-1}function Xn(n,e,t){const r=n[t*ie+0],s=n[t*ie+1],a=n[t*ie+2],i=new Array(e);let o=0;for(let l=0;l<e;l++){const u=n[l*ie+0]-r,d=n[l*ie+1]-s,h=n[l*ie+2]-a,f=Math.sqrt(u*u+d*d+h*h);i[l]=f,f>o&&(o=f)}o<1e-6&&(o=1);const c=new Array(e);for(let l=0;l<e;l++)c[l]=Math.min(255,Math.max(0,Math.round(Nn*i[l]/o)));return c[t]=0,c}function Qn(n,e){const t=new Set;for(const r of n.edges)r.sourceIndex===e&&r.targetIndex!==e&&t.add(r.targetIndex),r.targetIndex===e&&r.sourceIndex!==e&&t.add(r.sourceIndex);return Array.from(t).sort((r,s)=>r-s).slice(0,Wn)}function sr(n){return qn+Vn*n}const Ve=4;function Zn(n){return At(n)}function At(n){return{viable:!1,intruderIndex:-1,severedNeighborIndices:[],shockDelays:[],fireData:new Uint32Array(n),pathData:new Uint32Array(4),pathMetas:[],spineBeats:[],verdict:{headline:"threat quarantined",intruderLabel:"",receipt:"memory held in review · Memory PR opened"}}}function Jn(n,e){return Pr(n,e,Yn(n))}function $n(n,e,t){return t<0||t>=n.nodes.length?At(n.nodes.length):Pr(n,e,t)}function Pr(n,e,t){const r=n.nodes.length;if(r===0||t<0)return At(r);const s=wr(n,e),a=Xn(s,r,t),i=Qn(n,t),o=new Uint32Array(r);for(let g=0;g<r;g++)o[g]=a[g]&255;o[t]=256,i.forEach((g,y)=>{o[g]|=512|y<<10});const c=new Uint32Array(Math.max(1,i.length)*Ve),l=[];i.forEach((g,y)=>{const x=sr(y);c[y*Ve+0]=t,c[y*Ve+1]=g,c[y*Ve+2]=x,c[y*Ve+3]=ce.probe,l.push({sourceIndex:t,targetIndex:g,beatFrame:x,kind:ce.probe,beatKind:"sever",nodeId:n.nodes[g].id,label:he(n.nodes[g].label)})});const u=he(n.nodes[t].label),d=[],h=(g,y,x)=>{d.push({sourceIndex:t,targetIndex:t,beatFrame:g,kind:1,beatKind:"firewall",nodeId:x,label:y})};return h(Un,`intrusion · ${u}`,n.nodes[t].id),h(On,"immune response · shockwave","firewall-shock"),h(jn,"membrane forming","firewall-membrane"),i.forEach((g,y)=>{h(sr(y),`edge severed ✗ · ${he(n.nodes[g].label)}`,n.nodes[g].id)}),h(Hn,"threat quarantined","firewall-verdict"),{viable:!0,intruderIndex:t,severedNeighborIndices:i,shockDelays:a,fireData:o,pathData:c,pathMetas:l,spineBeats:d,verdict:{headline:"threat quarantined",intruderLabel:u,receipt:"memory held in review · Memory PR opened"}}}const ut=.1542;function ea(n=ut){return Math.pow(.9,-1/n)-1}function Sr(n,e,t=ut){if(!(n>0))return 0;if(!(e>0))return 1;const r=ea(t),s=Math.pow(1+r*e/n,-t);return s<0?0:s>1?1:s}const Ct=864e5;function ta(n,e,t=0){if(!n)return t>0?t:0;const r=Date.parse(n);if(!Number.isFinite(r))return t>0?t:0;const s=(e-r)/Ct;return Math.max(0,s)+Math.max(0,t)}function ra(n,e,t,r,s=ut){if(t){const i=Date.parse(t);if(Number.isFinite(i)&&r<i)return 0}if(n===void 0||!Number.isFinite(n)||!e)return 1;const a=Date.parse(e);return Number.isFinite(a)?Math.max(.001,Sr(n,(r-a)/Ct,s)):1}function ia(n,e,t,r=0,s=ut){return n===void 0||!Number.isFinite(n)?1:Sr(n,ta(e,t,r),s)}const or={[te.firewall]:620,[te.dreamStorm]:360,[te.causalRecall]:260,[te.birth]:180};class na{constructor(e){p(this,"engine");p(this,"renderer");p(this,"graph");p(this,"response");p(this,"seed");p(this,"projectionDays");p(this,"chronoOffsetDays");p(this,"onApply");p(this,"onFirewall");p(this,"firewall",null);p(this,"liveEdges",[]);p(this,"liveEdgeKeys",new Set);p(this,"edgesDirty",!1);p(this,"indexById");p(this,"active",null);p(this,"dreamOpen",!1);p(this,"retention");p(this,"hasLiveDecay",!1);p(this,"eventsSeen",0);p(this,"lastDecayFrame",-1e3);p(this,"lastAppliedMs",0);p(this,"seeded",!1);p(this,"lastProj",-1);p(this,"lastChrono",0);this.engine=e.engine,this.renderer=e.renderer,this.graph=e.graph,this.response=e.response,this.seed=e.seed,this.projectionDays=e.projectionDays??(()=>0),this.chronoOffsetDays=e.chronoOffsetDays??(()=>0),this.onApply=e.onApply,this.onFirewall=e.onFirewall,this.indexById=e.graph.indexById;const t=e.graph.nodes.length;this.retention=new Float32Array(t);for(let s=0;s<t;s++){const a=e.graph.nodes[s];this.retention[s]=a.retention,a.stability!==void 0&&a.lastAccessed&&(this.hasLiveDecay=!0)}this.liveEdges=e.graph.edges.slice();for(const s of this.liveEdges)this.liveEdgeKeys.add(lr(s.sourceIndex,s.targetIndex));this.lastAppliedMs=0;const r=this.engine.params;r[se.liveKind]=te.none,r[se.liveFrame]=0,r[se.liveEnergy]=0,r[se.projectionDays]=0}get liveDecayAvailable(){return this.hasLiveDecay}seedWatermark(e){let t=0;for(const r of e){const s=cr(r);s>t&&(t=s)}this.lastAppliedMs=t,this.seeded=!0}get hasActiveEvent(){return this.active!==null}replayRecall(e,t,r){if(this.active!==null)return!1;const s=this.indexById.get(e);if(s===void 0||(this.retention[s]??0)<5e-4)return!1;const a=t.filter(i=>i!==e&&this.indexById.has(i));return this.arm({kind:te.causalRecall,startFrame:r,targetId:e,relatedIds:a,pairs:[],scalar:a.length}),!0}ingest(e){if(e.length===0)return;if(!this.seeded){this.seedWatermark(e);return}let t=this.lastAppliedMs;for(let r=e.length-1;r>=0;r--){const s=e[r],a=cr(s);a>this.lastAppliedMs&&(this.decodeAndArm(s,this.engine.totalFrames),a>t&&(t=a))}this.lastAppliedMs=t}decodeAndArm(e,t){var s;const r=e.data??{};switch(e.type){case"MemorySuppressed":{const a=Le(r.id);if(!a||!this.indexById.has(a))return;this.arm({kind:te.firewall,startFrame:t,targetId:a,relatedIds:this.neighborsOf(a),pairs:[],scalar:We(r.estimated_cascade)});break}case"DeepReferenceCompleted":{const i=aa(r.contradiction_pairs).filter(([l,u])=>this.indexById.has(l)&&this.indexById.has(u));if(i.length>0){const l=i[0][0];this.arm({kind:te.firewall,startFrame:t,targetId:l,relatedIds:i.flatMap(u=>u).filter(u=>u!==l),pairs:i,scalar:i.length});return}const o=Le(r.primary_id),c=dr(r.supporting_ids).filter(l=>this.indexById.has(l));o&&this.indexById.has(o)&&this.arm({kind:te.causalRecall,startFrame:t,targetId:o,relatedIds:c,pairs:[],scalar:We(r.confidence)});break}case"BackfillFired":case"CausalReceipt":{const a=dr(r.path_ids??r.causal_path),i=Le(r.failure_id??r.target_id??r.effect_id)||a.at(-1)||a[0];i&&this.indexById.has(i)&&this.arm({kind:te.causalRecall,startFrame:t,targetId:i,relatedIds:a.filter(o=>o!==i),exactPath:a,pairs:[],scalar:a.length});break}case"DreamStarted":{this.dreamOpen=!0,this.arm({kind:te.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:We(r.memory_count)});break}case"DreamCompleted":{this.dreamOpen=!1;const a=We(r.connections_found);this.active&&this.active.kind===te.dreamStorm?this.active.scalar=Math.max(this.active.scalar,a):this.arm({kind:te.dreamStorm,startFrame:t,targetId:"",relatedIds:[],pairs:[],scalar:a});break}case"ConnectionDiscovered":{const a=this.indexById.get(Le(r.source_id)),i=this.indexById.get(Le(r.target_id));if(a===void 0||i===void 0||a===i)break;const o=lr(a,i);if(this.liveEdgeKeys.has(o))break;this.liveEdgeKeys.add(o),this.liveEdges.push({sourceIndex:a,targetIndex:i,weight:We(r.weight)||.5,type:Le(r.connection_type)||"semantic"}),this.edgesDirty=!0,this.dreamOpen&&((s=this.active)==null?void 0:s.kind)===te.dreamStorm&&(this.active.scalar+=1);break}}}arm(e){var t;if(this.active=e,this.eventsSeen++,e.kind===te.firewall){const r=this.indexById.get(e.targetId);if(r===void 0)return;const s=$n(this.graph,this.seed,r);if(!s.viable)return;this.firewall||(this.firewall=new Br({engine:this.engine,nodeRenderer:this.renderer,plan:Zn(this.graph.nodes.length)})),this.firewall.rearm(s),(t=this.onFirewall)==null||t.call(this,{intruderLabel:s.verdict.intruderLabel,startFrame:e.startFrame})}if(e.kind===te.causalRecall&&this.indexById.has(e.targetId)){if(e.exactPath&&e.exactPath.length>1){const s=e.exactPath;if(s.some(o=>!this.indexById.has(o)))return;const a=new Uint32Array(Math.max(1,s.length-1)*4),i=[];for(let o=0;o<s.length-1;o++){const c=this.indexById.get(s[o]),l=this.indexById.get(s[o+1]),u=e.startFrame+24+o*42;a[o*4]=c,a[o*4+1]=l,a[o*4+2]=u,a[o*4+3]=ce.backwardCause,i.push({sourceIndex:c,targetIndex:l,beatFrame:u,kind:ce.backwardCause,beatKind:"receipt-path",nodeId:s[o+1],label:"receipt-backed candidate path"})}this.renderer.setPathSteps(a,i);return}const r=br(this.response,this.graph,8,{preferCausal:!0,centerId:e.targetId});r.steps.length>0&&this.renderer.setPathSteps(r.data,r.steps)}}neighborsOf(e){const t=this.indexById.get(e);if(t===void 0)return[];const r=[];for(const s of this.graph.edges)if(s.sourceIndex===t?r.push(this.graph.nodes[s.targetIndex].id):s.targetIndex===t&&r.push(this.graph.nodes[s.sourceIndex].id),r.length>=12)break;return r}drain(e){var i;const t=this.engine.params;this.edgesDirty&&(this.renderer.setEdges(this.liveEdges),this.edgesDirty=!1);const r=this.projectionDays(),s=this.chronoOffsetDays();if(t[se.projectionDays]=Math.max(0,r),(this.hasLiveDecay||s!==0||this.lastChrono!==0)&&(e-this.lastDecayFrame>=6||r!==this.lastProj||s!==this.lastChrono)&&(this.recomputeDecay(r,s),this.lastDecayFrame=e,this.lastProj=r,this.lastChrono=s),this.active){const o=or[this.active.kind]??300,c=e-this.active.startFrame;c>o+140?(this.active=null,t[se.liveKind]=te.none,t[se.liveEnergy]=0):(t[se.liveKind]=this.active.kind,t[se.liveFrame]=Math.max(0,c),t[se.liveEnergy]=this.energyEnvelope(this.active,c,!1))}else t[se.liveKind]=te.none,t[se.liveEnergy]=0;(i=this.onApply)==null||i.call(this,{simFrame:e,activeKind:t[se.liveKind],eventsSeen:this.eventsSeen})}debugState(){const e=this.engine.params;return{activeKind:e[se.liveKind],liveEnergy:e[se.liveEnergy],liveFrame:e[se.liveFrame],edgeCount:this.liveEdges.length,eventsSeen:this.eventsSeen}}energyEnvelope(e,t,r){if(t<0)return 0;const s=or[e.kind]??300;if(e.kind===te.dreamStorm){const o=Math.min(1,t/45),c=1-Math.max(0,(t-(s-90))/90),l=Math.min(1.4,.7+e.scalar*.02);return Math.max(0,o*Math.min(1,c)*l)}const a=Math.min(1,t/24),i=1-Math.max(0,(t-s)/140);return Math.max(0,a*Math.min(1,i))}recomputeDecay(e,t=0){const r=this.engine.wallNowMs,s=this.graph.nodes;if(t!==0){const a=r+(t+Math.max(0,e))*Ct;for(let i=0;i<s.length;i++){const o=s[i];this.retention[i]=o.stability!==void 0||o.createdAt?ra(o.stability,o.lastAccessed,o.createdAt,a):Math.max(.001,o.retention)}}else for(let a=0;a<s.length;a++){const i=s[a];this.retention[a]=i.stability!==void 0&&i.lastAccessed?ia(i.stability,i.lastAccessed,r,e):Math.max(.001,i.retention)}this.renderer.uploadLiveRetention(this.retention)}refreshDecay(){const e=this.chronoOffsetDays();(this.hasLiveDecay||e!==0||this.lastChrono!==0)&&(this.recomputeDecay(this.projectionDays(),e),this.lastChrono=e)}}function lr(n,e){return n<e?`${n}-${e}`:`${e}-${n}`}function cr(n){var r;const e=(r=n.data)==null?void 0:r.timestamp;if(typeof e!="string")return 0;const t=Date.parse(e);return Number.isFinite(t)?t:0}function Le(n){return typeof n=="string"?n:""}function We(n){return typeof n=="number"&&Number.isFinite(n)?n:0}function dr(n){return Array.isArray(n)?n.filter(e=>typeof e=="string"):[]}function aa(n){if(!Array.isArray(n))return[];const e=[];for(const t of n)Array.isArray(t)&&t.length>=2&&typeof t[0]=="string"&&typeof t[1]=="string"&&e.push([t[0],t[1]]);return e}const nt=512,Pt=4,sa=`
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
`;function He(n){return Math.max(0,Math.min(1,Number.isFinite(n)?n:0))}function at(n){if(!n)return null;const e=Date.parse(n);return Number.isFinite(e)?e:null}class oa{constructor(e,t){p(this,"engine");p(this,"resources",null);p(this,"bindLayout",null);p(this,"railPipeline",null);p(this,"dwellPipeline",null);p(this,"headPipeline",null);p(this,"dwellCount",0);p(this,"minMs",0);p(this,"maxMs",0);p(this,"state",{scrub:1,days:0,density:0,active:0});this.engine=e,this.upload(t)}setTimeline(e,t=!1){const r=this.engine.wallNowMs,s=Math.max(1,this.maxMs-this.minMs);this.state.scrub=He((r+e*864e5-this.minMs)/s),this.state.days=Number.isFinite(e)?e:0,this.state.active=t?1:0,this.writeState(),this.engine.requestRender()}targetFrameRate(){return this.state.active>0?60:12}render(e){!this.resources||!this.railPipeline||!this.dwellPipeline||!this.headPipeline||(e.setBindGroup(0,this.resources.bindGroup),e.setPipeline(this.railPipeline),e.draw(6),this.dwellCount>0&&(e.setPipeline(this.dwellPipeline),e.draw(6,this.dwellCount)),e.setPipeline(this.headPipeline),e.draw(6))}dispose(){var e,t;(e=this.resources)==null||e.dwellBuffer.destroy(),(t=this.resources)==null||t.stateBuffer.destroy(),this.resources=null}upload(e){const t=e.flatMap(u=>[at(u.createdAt),at(u.lastAccessed)]).filter(u=>u!==null),r=this.engine.wallNowMs;this.minMs=t.length>0?Math.min(...t):r-864e5,this.maxMs=Math.max(r+365*864e5,this.minMs+864e5);const s=this.maxMs-this.minMs,a=[];for(const u of e){const d=at(u.createdAt),h=at(u.lastAccessed),f=He(u.retention);d!==null&&a.push({at:d,kind:0,retention:f,suppressed:u.suppressed?1:0}),h!==null&&h!==d&&a.push({at:h,kind:1,retention:f,suppressed:u.suppressed?1:0})}a.sort((u,d)=>u.at-d.at);const i=Math.max(1,Math.ceil(a.length/nt)),o=a.filter((u,d)=>d%i===0).slice(0,nt);this.dwellCount=o.length,this.state={scrub:He((r-this.minMs)/s),days:0,density:He(o.length/96),active:0};const c=this.engine.gpuDevice;if(!c||!this.engine.paramsBuffer||(this.ensurePipelines(c),this.ensureResources(c),!this.resources))return;const l=new Float32Array(nt*Pt);o.forEach((u,d)=>{l.set([He((u.at-this.minMs)/s),u.kind,u.retention,u.suppressed],d*Pt)}),c.queue.writeBuffer(this.resources.dwellBuffer,0,l),this.writeState()}ensurePipelines(e){if(this.railPipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"fossil-light-chrono-shuttle-wgsl",code:sa});this.bindLayout=e.createBindGroupLayout({label:"fossil-light-chrono-shuttle-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const r=e.createPipelineLayout({label:"fossil-light-chrono-shuttle-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),s={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},a=(i,o,c)=>e.createRenderPipeline({label:i,layout:r,vertex:{module:t,entryPoint:o},fragment:{module:t,entryPoint:c,targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:"triangle-list"}});this.railPipeline=a("fossil-light-chrono-rail","vs_rail","fs_rail"),this.dwellPipeline=a("fossil-light-chrono-dwells","vs_dwell","fs_dwell"),this.headPipeline=a("fossil-light-chrono-head","vs_head","fs_head")}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;const t=e.createBuffer({label:"fossil-light-chrono-dwell-events",size:nt*Pt*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r=e.createBuffer({label:"fossil-light-chrono-state",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.resources={dwellBuffer:t,stateBuffer:r,bindGroup:e.createBindGroup({label:"fossil-light-chrono-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:r}}]})}}writeState(){const e=this.engine.gpuDevice;!e||!this.resources||e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.state.scrub,this.state.days,this.state.density,this.state.active]))}}const Rt=64,la=12,ge=32,ca=4,st=256,St="rgba8unorm",da=96e3,ur=5,ua=`
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
`;function fa(n,e){return Number.isFinite(n)?n:e}class ha{constructor(e,t,r){p(this,"engine");p(this,"renderer");p(this,"sourceIndices");p(this,"resources",null);p(this,"projectionPipeline",null);p(this,"seedPipeline",null);p(this,"transportPipeline",null);p(this,"compositePipeline",null);p(this,"projectionLayout",null);p(this,"seedLayout",null);p(this,"transportLayout",null);p(this,"compositeLayout",null);p(this,"emitterCount");p(this,"active",!1);p(this,"dirty",!0);p(this,"lastComputedFrame",-ur);p(this,"disposed",!1);p(this,"disabledReason",null);p(this,"exposure",.42);p(this,"configBytes",new ArrayBuffer(ge));p(this,"configUints",new Uint32Array(this.configBytes));p(this,"configFloats",new Float32Array(this.configBytes));this.engine=e,this.renderer=t;const s=[...new Set([...r].filter(a=>Number.isFinite(a)&&a>=0))].sort((a,i)=>a-i).slice(0,Rt);this.sourceIndices=new Uint32Array(s),this.emitterCount=this.sourceIndices.length}get quality(){return this.disabledReason===null?"half-res-transport":"disabled"}get fallbackReason(){return this.disabledReason}setScrubbing(e){this.active=e,this.dirty=!0,this.engine.requestRender()}setExposure(e){this.exposure=Math.max(0,Math.min(.72,fa(e,.42))),this.dirty=!0,this.engine.requestRender()}targetFrameRate(){return this.active?60:10}compute(e,t=0){if(this.disposed||this.disabledReason!==null||this.emitterCount===0)return;const r=this.engine.gpuDevice;if(!r||!this.engine.paramsBuffer)return;const s=this.renderer.getFossilLightSources();if(!s)return;const a=this.fieldDimensions();if(a===null)return;const i=t-this.lastComputedFrame;if(!(this.active||this.dirty||i<0||i>=ur))return;try{this.ensurePipelines(r),this.ensureResources(r,a.width,a.height,s)}catch{this.disable("GPU light field unavailable on this adapter");return}if(!this.resources||!this.projectionPipeline||!this.seedPipeline||!this.transportPipeline)return;this.writeConfig(r,0,this.resources.width,this.resources.height,0);const c=Math.ceil(this.resources.width/8),l=Math.ceil(this.resources.height/8),u=e.beginComputePass({label:"fossil-light-half-res-transport"});u.setPipeline(this.projectionPipeline),u.setBindGroup(3,this.resources.projectionBindGroup,[0]),u.dispatchWorkgroups(Math.ceil(this.emitterCount/64)),u.setPipeline(this.seedPipeline),u.setBindGroup(0,this.resources.seedBindGroup,[0]),u.dispatchWorkgroups(c,l);for(const[d,h,f]of[[1,4,this.resources.propagateABindGroup],[2,13,this.resources.propagateBBindGroup],[3,37,this.resources.propagateABindGroup]])this.writeConfig(r,d,this.resources.width,this.resources.height,h),u.setPipeline(this.transportPipeline),u.setBindGroup(1,f,[d*st]),u.dispatchWorkgroups(c,l);u.end(),this.dirty=!1,this.lastComputedFrame=t}render(e){this.disabledReason!==null||!this.resources||!this.compositePipeline||this.emitterCount===0||(e.setPipeline(this.compositePipeline),e.setBindGroup(2,this.resources.compositeBindGroup,[3*st]),e.draw(6))}dispose(){this.disposed||(this.disposed=!0,this.destroyResources(),this.projectionPipeline=null,this.seedPipeline=null,this.transportPipeline=null,this.compositePipeline=null,this.seedLayout=null,this.projectionLayout=null,this.transportLayout=null,this.compositeLayout=null)}fieldDimensions(){const e=Math.floor(this.engine.params[6]),t=Math.floor(this.engine.params[7]);if(e<2||t<2)return null;const r=e*.5*(t*.5),a=.5*Math.min(1,Math.sqrt(da/Math.max(1,r)));return{width:Math.max(1,Math.floor(e*a)),height:Math.max(1,Math.floor(t*a))}}ensurePipelines(e){if(this.projectionPipeline&&this.seedPipeline&&this.transportPipeline&&this.compositePipeline)return;const t=e.createShaderModule({label:"fossil-light-radiance-cascade-wgsl",code:ua}),r=e.createBindGroupLayout({label:"fossil-light-empty-layout",entries:[]});this.projectionLayout=e.createBindGroupLayout({label:"fossil-light-source-projection-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),this.seedLayout=e.createBindGroupLayout({label:"fossil-light-seed-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:St}}]}),this.transportLayout=e.createBindGroupLayout({label:"fossil-light-transport-layout",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"float",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:St}}]}),this.compositeLayout=e.createBindGroupLayout({label:"fossil-light-composite-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform",hasDynamicOffset:!0,minBindingSize:ge}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]}),this.seedPipeline=e.createComputePipeline({label:"fossil-light-seed",layout:e.createPipelineLayout({label:"fossil-light-seed-pipeline-layout",bindGroupLayouts:[this.seedLayout]}),compute:{module:t,entryPoint:"cs_seed"}}),this.projectionPipeline=e.createComputePipeline({label:"fossil-light-source-projection",layout:e.createPipelineLayout({label:"fossil-light-source-projection-pipeline-layout",bindGroupLayouts:[r,r,r,this.projectionLayout]}),compute:{module:t,entryPoint:"cs_project_sources"}}),this.transportPipeline=e.createComputePipeline({label:"fossil-light-transport",layout:e.createPipelineLayout({label:"fossil-light-transport-pipeline-layout",bindGroupLayouts:[r,this.transportLayout]}),compute:{module:t,entryPoint:"cs_transport"}}),this.compositePipeline=e.createRenderPipeline({label:"fossil-light-composite",layout:e.createPipelineLayout({label:"fossil-light-composite-pipeline-layout",bindGroupLayouts:[r,r,this.compositeLayout]}),vertex:{module:t,entryPoint:"vs_composite"},fragment:{module:t,entryPoint:"fs_composite",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]}})}ensureResources(e,t,r,s){var f;if(((f=this.resources)==null?void 0:f.width)===t&&this.resources.height===r&&this.resources.nodeBuffer===s.nodeBuffer&&this.resources.cameraBuffer===s.cameraBuffer||(this.destroyResources(),!this.projectionLayout||!this.seedLayout||!this.transportLayout||!this.compositeLayout))return;const a=e.createBuffer({label:"fossil-light-projected-memory-emitters",size:Rt*la*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.STORAGE}),i=e.createBuffer({label:"fossil-light-source-indices",size:Math.max(4,this.sourceIndices.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(i,0,this.sourceIndices.buffer,this.sourceIndices.byteOffset,this.sourceIndices.byteLength);const o=e.createBuffer({label:"fossil-light-cascade-config",size:st*ca,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=g=>e.createTexture({label:g,size:[t,r],format:St,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING}),l=c("fossil-light-field-a"),u=c("fossil-light-field-b"),d=l.createView(),h=u.createView();this.resources={width:t,height:r,emitterBuffer:a,sourceIndexBuffer:i,configBuffer:o,fieldA:l,fieldB:u,seedBindGroup:e.createBindGroup({label:"fossil-light-seed-bind-group",layout:this.seedLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:{buffer:a}},{binding:2,resource:d}]}),propagateABindGroup:e.createBindGroup({label:"fossil-light-transport-a-to-b",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:d},{binding:2,resource:h}]}),propagateBBindGroup:e.createBindGroup({label:"fossil-light-transport-b-to-a",layout:this.transportLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:h},{binding:2,resource:d}]}),projectionBindGroup:e.createBindGroup({label:"fossil-light-source-projection-bind-group",layout:this.projectionLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:{buffer:i}},{binding:2,resource:{buffer:s.nodeBuffer}},{binding:3,resource:{buffer:s.cameraBuffer}},{binding:4,resource:{buffer:a}}]}),compositeBindGroup:e.createBindGroup({label:"fossil-light-composite-bind-group",layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:o,size:ge}},{binding:1,resource:h}]}),nodeBuffer:s.nodeBuffer,cameraBuffer:s.cameraBuffer},this.dirty=!0}writeConfig(e,t,r,s,a){this.resources&&(this.configUints[0]=r,this.configUints[1]=s,this.configUints[2]=this.emitterCount,this.configUints[3]=a,this.configFloats[4]=this.exposure,this.configFloats[5]=1,e.queue.writeBuffer(this.resources.configBuffer,t*st,this.configBytes))}destroyResources(){var e,t,r,s,a;(e=this.resources)==null||e.emitterBuffer.destroy(),(t=this.resources)==null||t.sourceIndexBuffer.destroy(),(r=this.resources)==null||r.configBuffer.destroy(),(s=this.resources)==null||s.fieldA.destroy(),(a=this.resources)==null||a.fieldB.destroy(),this.resources=null}disable(e){this.destroyResources(),this.disabledReason=e,this.engine.requestRender()}}var pa=J('<div class="flex items-baseline gap-2 font-mono text-[11px]"><span class="text-[#E9FFB7]/90 tabular-nums w-4"> </span> <span class="text-[#d8ded0]/90 truncate flex-1"> </span> <span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"> </span></div>'),ma=J(`<div class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"><div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase"> </div> <!></div>`),ga=J(`<div class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"><div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">⬤ threat quarantined</div> <div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate"> </div> <div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">memory held in review · Memory PR opened</div></div>`),va=J(`<button class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"> </button>`),ba=J('<button class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors" title="Return to now">now</button>'),ya=J('<div><span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span> <input type="range" max="365" step="0.25" class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity" aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve" title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"/> <span> </span> <!></div>'),_a=J(`<button class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors" title="Exit Observatory (Esc)">× EXIT</button>`),wa=J("<button> </button>"),xa=J('<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5"></div>'),Ba=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">LOADING MEMORY FIELD...</div></div>'),Pa=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded"> </div></div>'),Sa=J('<div class="absolute inset-0 flex items-center justify-center pointer-events-auto"><div class="text-[#5dcaa5] font-mono text-sm tracking-widest">NO MEMORIES IN FIELD</div></div>'),ka=J('<div class="absolute inset-0 z-10 pointer-events-none"><!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!> <!></div>'),Ia=J('<div><div role="application" aria-label="Interactive 3D memory field"><!></div> <!></div>');function ja(n,e){ct(e,!0);const t=()=>ci(ui,"$eventFeed",r),[r,s]=di();let a=q(e,"seed",3,"vestige-observatory-v1"),i=q(e,"freezeFrame",3,null),o=q(e,"capture",3,!1),c=q(e,"showSwitcher",3,!0),l=q(e,"embedded",3,!1),u=q(e,"chrome",3,"full"),d=q(e,"maxDpr",3,2),h=q(e,"focusIds",19,()=>[]),f=q(e,"live",3,!1),g=j(0),y=j(!1),x=j(0),B=null;const v=Ie(()=>Math.max(0,m(g))),F=Ie(()=>Math.min(0,m(g))),z=Ie(()=>m(g)===0?"now":m(g)>0?`+${Math.round(m(g))}d`:new Date(Date.now()+m(g)*864e5).toLocaleDateString(void 0,{month:"short",day:"numeric"}));let C=null,V=null,H=null,R=j(!1),O=j(!1);const ae=.835,ne=(1+.685)/2;let de=!1,ue=0,M=0,Q=0,w=!1;function L(b){var P;const _=(P=m(Ae))==null?void 0:P.getBoundingClientRect();if(!_||_.width===0)return m(g);const D=(b-_.left)/_.width*2-1,N=Math.max(0,Math.min(1,D/ae*.5+.5));return m(x)+N*(365-m(x))}function ft(b){var N;const _=(N=m(Ae))==null?void 0:N.getBoundingClientRect();if(!_||_.height===0)return!1;const D=(b.clientY-_.top)/_.height;return D>ne-.075&&D<ne+.075}function Ge(){ue&&cancelAnimationFrame(ue),ue=0}function kr(b){var _,D;!m(R)||o()||!ft(b)||(Ge(),de=!0,S(y,!0),M=0,Q=performance.now(),S(g,L(b.clientX),!0),(D=(_=b.currentTarget).setPointerCapture)==null||D.call(_,b.pointerId),b.preventDefault())}function Ir(b){if(!de)return;const _=performance.now(),D=L(b.clientX),N=Math.max(1,_-Q);M=M*.6+(D-m(g))/N*16*.4,Q=_,S(g,D,!0)}function Rr(b){var D,N;if(!de)return;de=!1,w=!0,(N=(D=b.currentTarget).releasePointerCapture)==null||N.call(D,b.pointerId);const _=()=>{ue=0,M*=.94;let P=m(g)+M;P<=m(x)&&(P=m(x),M=0),P>=365&&(P=365,M=0);const Z=m(g)<0&&M>0||m(g)>0&&M<0;Math.abs(P)<1&&Z&&(P=0,M=0),S(g,P,!0),Math.abs(M)>.02?ue=requestAnimationFrame(_):S(y,!1)};Math.abs(M)>.05?ue=requestAnimationFrame(_):S(y,!1)}function Er(){de=!1,S(y,!1),Ge()}let ye=j(!1),ht=j(!1);function Ar(){if(typeof window>"u")return;const b=window.matchMedia("(prefers-reduced-motion: reduce)");b.matches&&!m(ht)&&S(ye,!0);const _=D=>{m(ht)||S(ye,D.matches,!0)};return b.addEventListener("change",_),()=>b.removeEventListener("change",_)}function Mt(){S(ht,!0),S(ye,!m(ye))}Oe(()=>{var b;(b=m(le))==null||b.setPaused(m(ye))});let Ke=j(""),Ye=j(0),Cr=Ie(()=>m(Ke)!==""&&m(Ye)>0),Ae=j(null);async function Mr(b){if(w){w=!1;return}if(!e.onpick||!G||!m(Ae))return;const _=m(Ae).getBoundingClientRect();if(_.width===0||_.height===0)return;const D=(b.clientX-_.left)/_.width*2-1,N=-((b.clientY-_.top)/_.height*2-1),P=await G.pickAt(D,N);P&&e.onpick(P.id)}const Ft={"recall-path":"RECALL","engram-birth":"BIRTH","salience-rescue":"RESCUE","forgetting-horizon":"HORIZON",firewall:"FIREWALL"};function Fr(){return G!=null&&G.graph?new Uint32Array(G.graph.nodes.map(b=>({index:b.index,id:b.id,retention:b.retention})).sort((b,_)=>_.retention-b.retention||b.id.localeCompare(_.id)).slice(0,64).map(b=>b.index).sort((b,_)=>b-_)):new Uint32Array}let pt=j(!o());function Lr(b){const _=b.target;_!=null&&_.isContentEditable||(_==null?void 0:_.tagName)==="INPUT"||(_==null?void 0:_.tagName)==="TEXTAREA"||(_==null?void 0:_.tagName)==="SELECT"||((b.key==="h"||b.key==="H")&&S(pt,!m(pt)),b.key==="Escape"&&e.onexit&&e.onexit(),(b.key===" "||b.key.toLowerCase()==="p")&&!o()&&(b.preventDefault(),Mt()))}let _e=j(null),Ce=j(!0),De=j(""),Te=j(0),Lt=j(0),mt=j(0),gt=j(0),vt=j(""),le=j(null),G=null,Xe=null,Gt=null,bt=j(null),Dt=null,Tt=null,ze=j(null),yt=!1,Me=j(Xt([]));async function Gr(){S(Ce,!0),S(De,"");try{const b=new Set(h().filter(Boolean)),_=b.size?await(async()=>{var K,Y;const D=await Promise.all([...b].map(U=>wt.graph({center_id:U,max_nodes:200,depth:3}))),N=[...new Map(D.flatMap(U=>U.nodes).map(U=>[U.id,U])).values()].filter(U=>b.has(U.id)),P=new Set(N.map(U=>U.id)),Z=[...new Map(D.flatMap(U=>U.edges).map(U=>[`${U.source}:${U.target}`,U])).values()].filter(U=>P.has(U.source)&&P.has(U.target));return{...D[0],nodes:N,edges:Z,center_id:((K=N[0])==null?void 0:K.id)??((Y=D[0])==null?void 0:Y.center_id)??"",nodeCount:N.length,edgeCount:Z.length}})():await wt.graph({max_nodes:200,depth:3,sort:"connected"});S(_e,_,!0),S(mt,_.nodeCount,!0),S(gt,_.edgeCount,!0),S(vt,_.center_id,!0)}catch(b){const _=b instanceof Error?b.message:"Failed to load graph data";/\b404\b/.test(_)?(S(_e,{nodes:[],edges:[],nodeCount:0,edgeCount:0,center_id:""},!0),S(mt,0),S(gt,0),S(vt,"")):S(De,_,!0)}finally{S(Ce,!1)}}let Qe=null,Ze=j(Xt([])),Je=j("recalls");function Dr(b,_){S(Te,b,!0),S(Lt,_,!0),Qe&&!m(y)&&Qe.tick(b)}async function Tr(){var K;if(!C||!(G!=null&&G.graph))return;const b=G.graph,_=Y=>b.indexById.has(Y),D=Y=>{var U;return((U=b.nodes[b.indexById.get(Y)??-1])==null?void 0:U.label)??Y.slice(0,8)};let N=[];try{N=((K=await wt.receipts.list(60))==null?void 0:K.receipts)??[]}catch{}let P=vi(N,_);P.length===0&&(P=bi(b.nodes,12)),P.length>0&&(Qe=new _i(C,{intervalFrames:240}),Qe.setItems(P));const Z=yi(N,_,3);Z.length>0?(S(Je,"recalls"),S(Ze,Z.map(Y=>({...Y,label:D(Y.id)})),!0)):(S(Je,"retention"),S(Ze,[...b.nodes].filter(Y=>(Y.label??"").trim().length>0).sort((Y,U)=>U.retention-Y.retention).slice(0,3).map(Y=>({id:Y.id,recalls:Math.round(Y.retention*100),label:Y.label||Y.id.slice(0,8)})),!0))}function zr(b){var _;yt=!1,S(le,b,!0),G=new qi(b),(_=e.onready)==null||_.call(e,b)}Oe(()=>{if(m(le)&&G&&m(_e)&&!yt){yt=!0;const b=e.demo==="engram-birth",_=e.demo==="salience-rescue",D=e.demo==="forgetting-horizon",N=e.demo==="firewall";if(G.upload(m(_e),a(),{recallPath:!b&&!_&&!D&&!N}),b){Xe=new sn({engine:m(le),nodeRenderer:G,seed:a()}),Xe.upload(a());const P=Xe.engraveSteps,Z=[];for(let K=0;K<P.length/4;K++)Z.push({sourceIndex:P[K*4],targetIndex:P[K*4+1],beatFrame:P[K*4+2],kind:P[K*4+3],beatKind:"engrave",nodeId:`engrave-${K}`,label:"edge engraved"});G.setPathSteps(P,Z),S(Me,Xe.timeline.map((K,Y)=>({sourceIndex:0,targetIndex:0,beatFrame:K.startFrame,kind:0,beatKind:"birth",nodeId:`birth-${Y}`,label:K.label})),!0)}else if(_){const P=xn(m(_e),G.graph,a(),e.backfillEvidence);S(bt,P,!0),P.viable&&(Gt=new cn({engine:m(le),nodeRenderer:G,plan:P}),Gt.upload(),G.setPathSteps(P.pathData,P.pathMetas)),S(Me,P.spineBeats,!0)}else if(D){const P=Dn(G.graph);P.viable&&(Dt=new kn({engine:m(le),nodeRenderer:G,plan:P}),Dt.upload(),G.setPathSteps(P.pathData,P.pathMetas)),S(Me,P.spineBeats,!0)}else if(N){const P=Jn(G.graph,a());S(ze,P,!0),P.viable&&(Tt=new Br({engine:m(le),nodeRenderer:G,plan:P}),Tt.upload(),G.setPathSteps(P.pathData,P.pathMetas)),S(Me,P.spineBeats,!0)}else S(Me,G.pathSteps,!0);if(f()&&G.graph&&m(_e)){C=new na({engine:m(le),renderer:G,graph:G.graph,response:m(_e),seed:a(),projectionDays:()=>m(v),chronoOffsetDays:()=>m(F),onFirewall:Z=>{S(Ke,Z.intruderLabel,!0),S(Ye,Date.now(),!0)}}),S(O,C.liveDecayAvailable,!0),m(le).setPreFrameHook(Z=>C==null?void 0:C.drain(Z)),o()||Tr();let P=Number.POSITIVE_INFINITY;for(const Z of G.graph.nodes)if(Z.createdAt){const K=Date.parse(Z.createdAt);Number.isFinite(K)&&K<P&&(P=K)}if(Number.isFinite(P)&&S(x,Math.floor((P-Date.now())/864e5)-1),B){const Z=Date.parse(B);Number.isFinite(Z)&&S(g,Math.min(365,Math.max(m(x),(Z-Date.now())/864e5)),!0),B=null}o()||(H=new ha(m(le),G,Fr()),m(le).addPass(H),V=new oa(m(le),G.graph.nodes),m(le).addPass(V),S(R,!0)),typeof window<"u"&&(window.__vestigeLiveBridge=C)}m(le).demoClock.reset()}}),Oe(()=>{const b=t();C&&C.ingest(b)}),Oe(()=>{m(g),C==null||C.refreshDecay(),V==null||V.setTimeline(m(g),m(y)),H==null||H.setScrubbing(m(y))}),Oe(()=>{if(!m(Ye))return;const b=setTimeout(()=>{S(Ke,""),S(Ye,0)},7e3);return()=>clearTimeout(b)}),ii(()=>{B=new URLSearchParams(window.location.search).get("t"),Gr();const b=Ar();return()=>{if(Ge(),b==null||b(),typeof window<"u"){const _=window;_.__vestigeLiveBridge===C&&delete _.__vestigeLiveBridge}}});var $e=Ia();rt("keydown",ni,Lr);let zt;var me=A($e);let Ut;var Ur=A(me);pi(Ur,{get demo(){return e.demo},get seed(){return a()},get freezeFrame(){return i()},get maxDpr(){return d()},onframe:Dr,onready:zr}),E(me),si(me,b=>S(Ae,b),()=>m(Ae));var Or=T(me,2);{var Nr=b=>{var _=ka(),D=A(_);{var N=k=>{var I=ma(),$=A(I),X=A($,!0);E($);var fe=T($,2);kt(fe,19,()=>m(Ze),xe=>xe.id,(xe,Be,_t)=>{var Ue=pa(),we=A(Ue),et=A(we,!0);E(we);var tt=T(we,2),$r=A(tt,!0);E(tt);var Yt=T(tt,2),ei=A(Yt);E(Yt),E(Ue),oe(()=>{re(et,m(_t)+1),ke(tt,"title",m(Be).label),re($r,m(Be).label),re(ei,`${m(Be).recalls??""}${m(Je)==="recalls"?"×":"%"}`)}),W(xe,Ue)}),E(I),oe(()=>re(X,m(Je)==="recalls"?"Most recalled · your mind":"Strongest memories · your mind")),W(k,I)};ee(D,k=>{f()&&m(Ze).length>0&&k(N)})}var P=T(D,2);{var Z=k=>{var I=ga(),$=T(A(I),2),X=A($,!0);E($),ai(2),E(I),oe(()=>re(X,m(Ke))),W(k,I)};ee(P,k=>{f()&&m(Cr)&&k(Z)})}var K=T(P,2);{var Y=k=>{var I=va(),$=A(I,!0);E(I),oe(()=>{ke(I,"title",m(ye)?"Resume field motion":"Pause field motion"),ke(I,"aria-pressed",m(ye)),ke(I,"aria-label",m(ye)?"Resume 3D memory field motion":"Pause 3D memory field motion"),re($,m(ye)?"▶ RESUME":"❚❚ PAUSE")}),pe("click",I,Mt),W(k,I)};ee(K,k=>{o()||k(Y)})}var U=T(K,2);{var jr=k=>{var I=ya();let $;var X=T(A(I),2);oi(X);var fe=T(X,2);let xe;var Be=A(fe,!0);E(fe);var _t=T(fe,2);{var Ue=we=>{var et=ba();pe("click",et,()=>S(g,0)),W(we,et)};ee(_t,we=>{m(g)!==0&&we(Ue)})}E(I),oe(()=>{$=Re(I,1,`absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]`,null,$,{"opacity-100":m(R),"opacity-75":!m(R)}),ke(X,"min",m(x)),xe=Re(fe,1,"w-16 text-right tabular-nums",null,xe,{"text-[#b9d9a9]":m(g)>=0,"text-[#dfc68e]":m(g)<0}),re(Be,m(z))}),pe("input",X,()=>S(y,!0)),pe("change",X,()=>S(y,!1)),pe("pointerup",X,()=>S(y,!1)),rt("pointercancel",X,()=>S(y,!1)),rt("blur",X,()=>S(y,!1)),li(X,()=>m(g),we=>S(g,we)),W(k,I)};ee(U,k=>{f()&&m(O)&&k(jr)})}var Ot=T(U,2);{var qr=k=>{Si(k,{get demoMode(){return e.demo},get seed(){return a()},get nodeCount(){return m(mt)},get edgeCount(){return m(gt)},get centerId(){return m(vt)},get frameCount(){return m(Te)},get fpsEstimate(){return m(Lt)},get freezeFrame(){return i()},get loading(){return m(Ce)},get error(){return m(De)}})};ee(Ot,k=>{u()==="full"&&k(qr)})}var Nt=T(Ot,2);{var Vr=k=>{var I=_a();pe("click",I,function(...$){var X;(X=e.onexit)==null||X.apply(this,$)}),W(k,I)};ee(Nt,k=>{u()==="full"&&e.onexit&&k(Vr)})}var jt=T(Nt,2);{var Wr=k=>{var I=xa();kt(I,20,()=>mi,$=>$,($,X)=>{var fe=wa(),xe=A(fe,!0);E(fe),oe(()=>{Re(fe,1,`font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							${X===e.demo?"text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]":"text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50"}`),ke(fe,"title",`Play the ${Ft[X]??""} moment`),re(xe,Ft[X])}),pe("click",fe,()=>{var Be;return(Be=e.ondemochange)==null?void 0:Be.call(e,X)}),W($,fe)}),E(I),W(k,I)};ee(jt,k=>{u()==="full"&&c()&&k(Wr)})}var qt=T(jt,2);{var Hr=k=>{var I=Ba();W(k,I)};ee(qt,k=>{m(Ce)&&k(Hr)})}var Vt=T(qt,2);{var Kr=k=>{var I=Pa(),$=A(I),X=A($,!0);E($),E(I),oe(()=>re(X,m(De))),W(k,I)};ee(Vt,k=>{m(De)&&!m(Ce)&&k(Kr)})}var Wt=T(Vt,2);{var Yr=k=>{Ei(k,{get steps(){return m(Me)},get frame(){return m(Te)}})};ee(Wt,k=>{u()==="full"&&k(Yr)})}var Ht=T(Wt,2);{var Xr=k=>{Qt(k,{get frame(){return m(Te)},get verdict(){return m(bt).verdict}})};ee(Ht,k=>{var I;u()==="full"&&e.demo==="salience-rescue"&&((I=m(bt))!=null&&I.viable)&&k(Xr)})}var Kt=T(Ht,2);{var Qr=k=>{{let I=Ie(()=>({headline:m(ze).verdict.headline,causeLabel:m(ze).verdict.intruderLabel,receipt:m(ze).verdict.receipt}));Qt(k,{get frame(){return m(Te)},tone:"quarantine",fadeWindow:[480,495,605,620],get verdict(){return m(I)}})}};ee(Kt,k=>{var I;u()==="full"&&e.demo==="firewall"&&((I=m(ze))!=null&&I.viable)&&k(Qr)})}var Zr=T(Kt,2);{var Jr=k=>{var I=Sa();W(k,I)};ee(Zr,k=>{!m(Ce)&&m(_e)&&m(_e).nodeCount===0&&k(Jr)})}E(_),W(b,_)};ee(Or,b=>{m(pt)&&b(Nr)})}E($e),oe(()=>{zt=Re($e,1,`${l()?"absolute":"fixed"} inset-0 overflow-hidden bg-[#05060a]`,null,zt,{"cursor-none":o()}),Ut=Re(me,1,"absolute inset-0 z-0 touch-none",null,Ut,{"cursor-crosshair":!!e.onpick&&!o()})}),pe("click",me,Mr),pe("pointerdown",me,kr),pe("pointermove",me,Ir),pe("pointerup",me,Rr),rt("pointercancel",me,Er),W(n,$e),dt(),s()}fr(["click","pointerdown","pointermove","pointerup","input","change"]);export{ja as O};

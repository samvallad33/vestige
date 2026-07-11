var $=Object.defineProperty;var q=(u,e,r)=>e in u?$(u,e,{enumerable:!0,configurable:!0,writable:!0,value:r}):u[e]=r;var y=(u,e,r)=>q(u,typeof e!="symbol"?e+"":e,r);import"../chunks/Bzak7iHL.js";import{d as W,a as Z,b as G,e as K}from"../chunks/DAau0uzT.js";import{p as J,a as Q,b as ee,h as te,f as re,c as ne,g as _,s as L,d as H,$ as oe,r as ie}from"../chunks/CGq8RnJq.js";import{h as se}from"../chunks/De_e6MzK.js";import{b as ae}from"../chunks/Ccqjq5DS.js";import{g as U}from"../chunks/DJDK-KWF.js";import{b as k}from"../chunks/DY7cP31Q.js";import{r as A,O as ce}from"../chunks/BMB5u1EX.js";import{o as le,p as he,l as fe,m as ue}from"../chunks/BKh9s_e0.js";import{T as de}from"../chunks/D7ozXiSB.js";const pe=28,Y=12,me=18,ge=.32,ve=8.2,ye=2.35,be=1.55,xe={reasoning:"#22C7DE",memory:"#29F2A9",immune:"#FF5E7A",signal:"#FFC44D",temporal:"#8B7BFF",system:"#DDE7FF"},_e={reasoning:0,memory:1,immune:2,signal:3,temporal:4,system:5},Fe=`
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
	// x: hover code = strength*(hoveredIndex+1), 0 = nothing hovered.
	// yzw: focused organ world position (parting pushes others away from it).
	hover: vec4<f32>,
};

struct Organ {
	pos_radius: vec4<f32>,   // xyz world pos, w core radius
	color_family: vec4<f32>, // rgb accent, w family id
	info: vec4<f32>,         // x center flag, yzw reserved
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<storage, read> organs: array<Organ>;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) accent: vec3<f32>,
	// x radius, y center flag, z family id, w breath
	@location(2) @interpolate(flat) info: vec4<f32>,
	// focus factor for THIS orb: 1 = the hovered organ, 0 = not (eases via strength)
	@location(3) @interpolate(flat) focus: f32,
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
	if (ii >= u32(params.node_count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}

	let organ = organs[ii];
	let corner = CORNERS[vi];
	let is_center = organ.info.x > 0.5;

	// Breath: hero orbs swell ~8% on the global pulse; the center heart breathes
	// deeper. A slow per-organ phase offset (from world x) desyncs the field so
	// it shimmers like a living constellation, not a single strobe.
	let phase_off = organ.pos_radius.x * 0.35 + organ.pos_radius.z * 0.21;
	let local_pulse = 0.5 + 0.5 * sin(params.loop_phase * 6.28318530718 * 2.0 + phase_off);
	var breath = 1.0 + 0.08 * local_pulse;
	if (is_center) {
		breath = 1.0 + 0.16 * params.pulse;
	}

	// Hero sprite: ~2.6x the core radius. Big enough to fill the view, small
	// enough that 19 halos stay DISTINCT instead of merging into one bloom fog
	// (3.4 washed the frame out and buried the labels).
	// ── Focus+context (hover-to-inspect nav) ──
	// hover.x = eased strength 0..1 (0 = nothing focused); hover.yzw = focused
	// organ world pos. This orb is "focused" if its world pos matches yzw.
	let strength = clamp(camera.hover.x, 0.0, 1.0);
	let hover_active = strength > 0.001;
	let focused_pos = camera.hover.yzw;
	let is_focused = hover_active && distance(organ.pos_radius.xyz, focused_pos) < 0.001;

	// Part the OTHER orbs radially away from the focused organ so the hovered one
	// gets breathing room (accordion/fisheye), eased by strength.
	var pos = organ.pos_radius.xyz;
	if (hover_active && !is_focused) {
		let delta = organ.pos_radius.xyz - focused_pos;
		let dist = max(length(delta), 0.0001);
		pos = pos + (delta / dist) * 2.6 * strength;
	}

	// Focused organ swells so it dominates the view.
	let focus_scale = select(1.0, 1.0 + 0.6 * strength, is_focused);
	let half_size = organ.pos_radius.w * 2.6 * breath * focus_scale;
	let world = pos
		+ camera.right.xyz * corner.x * half_size
		+ camera.up.xyz * corner.y * half_size;

	out.clip = camera.view_proj * vec4<f32>(world, 1.0);
	out.uv = corner;
	out.accent = organ.color_family.rgb;
	out.info = vec4<f32>(organ.pos_radius.w, select(0.0, 1.0, is_center), organ.color_family.w, breath);
	out.focus = select(0.0, strength, is_focused);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) {
		discard;
	}

	let is_center = in.info.y > 0.5;

	// Soft glow: hot core + TIGHTER halo (falls off faster) so orbs read as
	// distinct nodes and don't drown the frame + labels in overlapping bloom.
	let core = smoothstep(0.22, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 3.4);
	var intensity = core * 1.4 + halo * (0.30 + 0.14 * params.pulse);

	// Thin fresnel ring gives each orb a crisp planetary rim (reads as a sphere,
	// not a fuzzy dot) — brightest at the silhouette edge.
	let ring = smoothstep(0.62, 0.82, d) * (1.0 - smoothstep(0.9, 1.0, d));
	intensity = intensity + ring * 0.55;

	if (is_center) {
		intensity = intensity * 1.7;
	}

	// Focus glow: the hovered organ blazes brighter + gets a hotter rim so it
	// clearly reads as "the section you're about to enter."
	intensity = intensity * (1.0 + 1.3 * in.focus);
	intensity = intensity + ring * in.focus * 1.2;

	var color = in.accent * intensity;
	// Hovered orb picks up a white-hot center so its label reads on top of it.
	color = color + vec3<f32>(1.0, 1.0, 1.0) * core * in.focus * 0.7;

	// Center heart gets a white-hot pinpoint core — the cortex the organs orbit.
	if (is_center) {
		color = color + vec3<f32>(1.0, 1.0, 1.0) * core * 0.6;
	}

	return vec4<f32>(color * params.brightness, 1.0);
}
`,z=class z{constructor(e){y(this,"engine");y(this,"pipeline",null);y(this,"bindGroup",null);y(this,"cameraBuffer",null);y(this,"nodeBuffer",null);y(this,"cameraData",new Float32Array(pe));y(this,"nodeCount",0);y(this,"placed",[]);y(this,"hoveredIndex",-1);y(this,"hoverStrength",0);y(this,"dive",null);y(this,"onArrive",null);this.engine=e}setHovered(e){this.hoveredIndex=e>=0&&e<this.placed.length?e:-1}indexOfHref(e){return e?this.placed.findIndex(r=>r.href===e):-1}startDive(e,r){if(this.dive)return!1;const n=this.placed.find(t=>t.href===e);return n?(this.dive={target:[n.x,n.y,n.z],startMs:this.engine.wallNowMs,href:e},this.onArrive=r,!0):!1}get isDiving(){return this.dive!==null}uploadRegions(e){var s;const r=this.engine.gpuDevice;if(!r)return;const n=this.layout(e);this.placed=n,this.nodeCount=n.length;const t=new Float32Array(Math.max(n.length,1)*Y);for(let o=0;o<n.length;o++){const i=n[o],f=e[o],[d,m,c]=A(xe[f.family]),a=o*Y;t[a+0]=i.x,t[a+1]=i.y,t[a+2]=i.z,t[a+3]=i.radius,t[a+4]=d,t[a+5]=m,t[a+6]=c,t[a+7]=_e[f.family],t[a+8]=i.center?1:0,t[a+9]=0,t[a+10]=0,t[a+11]=0}(s=this.nodeBuffer)==null||s.destroy(),this.nodeBuffer=r.createBuffer({label:"palace-node-state",size:Math.max(t.byteLength,64),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r.queue.writeBuffer(this.nodeBuffer,0,t.buffer),this.cameraBuffer||(this.cameraBuffer=r.createBuffer({label:"palace-camera",size:this.cameraData.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),this.createPipeline(r)}layout(e){const r=[],n=Math.PI*(3-Math.sqrt(5)),s=e.filter(i=>!i.center).length;let o=0;for(const i of e){if(i.center){r.push({href:i.href,x:0,y:0,z:0,radius:ye,center:!0});continue}const f=o++,d=s>1?1-f/(s-1)*2:0,m=Math.sqrt(Math.max(0,1-d*d)),c=n*f,a=ve*(.82+.18*(f*.6180339887%1));r.push({href:i.href,x:Math.cos(c)*m*a,y:d*a*.82,z:Math.sin(c)*m*a,radius:be,center:!1})}return r}createPipeline(e){if(!this.engine.paramsBuffer||!this.cameraBuffer||!this.nodeBuffer)return;const r=e.createShaderModule({label:"palace-render-nodes",code:Fe});this.pipeline=e.createRenderPipeline({label:"palace-nodes",layout:"auto",vertex:{module:r,entryPoint:"vs_main"},fragment:{module:r,entryPoint:"fs_main",targets:[{format:this.engine.sceneFormat,blend:{color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),this.bindGroup=e.createBindGroup({label:"palace-nodes-bind",layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.cameraBuffer}},{binding:2,resource:{buffer:this.nodeBuffer}}]})}currentCamera(){const e=this.engine.params[6]||1,r=this.engine.params[7]||1,n=e/r,t=this.engine.params[1],s=le(t,n,me,ge);if(!this.dive)return s;const o=Math.min(1,(this.engine.wallNowMs-this.dive.startMs)/z.DIVE_MS),i=o*o*o,f=[s.eye[0]+(this.dive.target[0]-s.eye[0])*i*.985,s.eye[1]+(this.dive.target[1]-s.eye[1])*i*.985,s.eye[2]+(this.dive.target[2]-s.eye[2])*i*.985],d=[this.dive.target[0]*i,this.dive.target[1]*i,this.dive.target[2]*i],m=(50-8*i)*Math.PI/180,c=he(m,n,.05,4e3),a=fe(f,d,[0,1,0]),b=ue(c,a);if(o>=1&&this.onArrive){const F=this.onArrive,w=this.dive.href;this.onArrive=null,this.dive=null,queueMicrotask(()=>F(w))}return{viewProj:b,right:s.right,up:s.up,eye:f}}compute(){const e=this.engine.gpuDevice;if(!e||!this.cameraBuffer)return;const r=this.currentCamera();this.cameraData.set(r.viewProj,0),this.cameraData[16]=r.right[0],this.cameraData[17]=r.right[1],this.cameraData[18]=r.right[2],this.cameraData[19]=0,this.cameraData[20]=r.up[0],this.cameraData[21]=r.up[1],this.cameraData[22]=r.up[2],this.cameraData[23]=0;const n=this.hoveredIndex>=0?1:0;this.hoverStrength+=(n-this.hoverStrength)*.18,this.hoverStrength<.001&&(this.hoverStrength=0);const t=this.hoveredIndex>=0?this.placed[this.hoveredIndex]:null;this.cameraData[24]=t?this.hoverStrength:0,this.cameraData[25]=t?t.x:0,this.cameraData[26]=t?t.y:0,this.cameraData[27]=t?t.z:0,e.queue.writeBuffer(this.cameraBuffer,0,this.cameraData)}render(e){!this.pipeline||!this.bindGroup||this.nodeCount===0||(this.engine.params[2]=this.nodeCount,e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.nodeCount))}pickAt(e,r){if(this.nodeCount===0)return null;const n=this.currentCamera().viewProj,t=1/Math.tan(50*Math.PI/360),s=this.engine.params[6]||1,o=this.engine.params[7]||1,i=s/o;let f=-1,d=1/0;for(let m=0;m<this.placed.length;m++){const c=this.placed[m],a=n[3]*c.x+n[7]*c.y+n[11]*c.z+n[15];if(a<=0)continue;const b=(n[0]*c.x+n[4]*c.y+n[8]*c.z+n[12])/a,F=(n[1]*c.x+n[5]*c.y+n[9]*c.z+n[13])/a,w=Math.max(c.radius*t/a,.02),O=(b-e)/i,B=F-r;let p=Math.hypot(O,B)/w;m===this.hoveredIndex&&(p*=.75),p<.5&&p<d&&(d=p,f=m)}return f<0?null:{index:f,href:this.placed[f].href}}getScreenPositions(){const e=this.currentCamera().viewProj,r=[];for(const n of this.placed){const t=e[3]*n.x+e[7]*n.y+e[11]*n.z+e[15];if(t<=0){r.push({href:n.href,ndcX:0,ndcY:0,depth:t,visible:!1});continue}const s=(e[0]*n.x+e[4]*n.y+e[8]*n.z+e[12])/t,o=(e[1]*n.x+e[5]*n.y+e[9]*n.z+e[13])/t;r.push({href:n.href,ndcX:s,ndcY:o,depth:t,visible:!0})}return r}dispose(){var e,r;(e=this.nodeBuffer)==null||e.destroy(),(r=this.cameraBuffer)==null||r.destroy(),this.nodeBuffer=null,this.cameraBuffer=null,this.pipeline=null,this.bindGroup=null,this.nodeCount=0,this.placed=[]}};y(z,"DIVE_MS",620);let N=z;const T=[{href:"/observatory",label:"OBSERVATORY",family:"system",center:!0},{href:"/graph",label:"GRAPH",family:"memory"},{href:"/memories",label:"MEMORIES",family:"memory"},{href:"/timeline",label:"TIMELINE",family:"temporal"},{href:"/feed",label:"FEED",family:"signal"},{href:"/explore",label:"EXPLORE",family:"reasoning"},{href:"/reasoning",label:"REASONING",family:"reasoning"},{href:"/stats",label:"STATS",family:"system"},{href:"/settings",label:"SETTINGS",family:"system"}];function we(u){return T.find(e=>e.href===u)}const Ee={reasoning:[...A("#FFFFFF"),1],memory:[...A("#FFFFFF"),1],immune:[...A("#FFFFFF"),1],temporal:[...A("#FFFFFF"),1],signal:[...A("#FFFFFF"),1],system:[...A("#FFFFFF"),1]},Ae=[...A("#FFFFFF"),1],Se=1.35,Oe=.028,Re=.6,j=.85,Ie=.05,Be=.03,Ce=1,De=.95,D=[],M=[];function V(u,e,r){return u+(e-u)*r}function X(u){return u<0?0:u>1?1:Number.isFinite(u)?u:0}function Me(u,e={}){const r=e.hoveredHref??null,n=e.dimUnhovered??!0;let t=0;M.length=0;for(let s=0;s<u.length;s++){const o=u[s];if(o.visible===!1)continue;const i=we(o.href);if(!i)continue;const f=X(o.depth),d=r!==null&&o.href===r,m=Oe+f*Se;let c=o.ndcX+m*j,a=o.ndcY+m*Re,b=V(Be,Ie,f),F=V(De,Ce,f);i.center&&(b*=1.22),d?(b*=1.28,F=1):n&&r!==null&&(F*=.32);const w=i.label.length*b*.62;c+w>.97&&(c=o.ndcX-m*j-w);for(let C=0;C<M.length;C++){const R=M[C],S=c<R.x1&&c+w>R.x0,P=Math.abs(a-R.y)<(b+R.size)*.75;S&&P&&(a=R.y-(b+R.size)*.85)}M.push({x0:c,x1:c+w,y:a,size:b});const O=i.center?Ae:Ee[i.family],B=[O[0],O[1],O[2],X(F)];let p=D[t];p||(p={id:"",kind:"palace-label",text:"",x:0,y:0,size:0,color:[0,0,0,0],depth:0,weight:.75,revealSpan:1,maxWidthEm:24},D[t]=p),p.id="palace-label:"+o.href,p.kind="palace-label",p.text=i.label,p.x=c,p.y=a,p.size=b,p.color=B,p.depth=d?1:.8+f*.2,p.weight=.95,t++}return D.length=t,D}T.length;var Te=re('<div class="fixed inset-0 bg-[#020307]"><!></div>');function Xe(u,e){J(e,!0);const r=[...A("#EAFBFF"),1],n=[...A("#CFFFE9"),1];let t=H(null),s=null,o=null,i=null,f=null,d=H(null);Z(()=>{i==null||i.dispose(),i=null,o=null,s=null});async function m(l){s=l,o=new N(l),l.addPass(o),o.uploadRegions(T);const h=new de(l);i=h,await h.init(),l.addPass(h),l.demoClock.reset(),h.setText(F())}function c(){if(!o)return[];const l=o.getScreenPositions(),h=l.filter(x=>x.visible);let g=1/0,v=-1/0;for(const x of h)x.depth<g&&(g=x.depth),x.depth>v&&(v=x.depth);const E=v===g,I=v-g||1;return l.map(x=>({href:x.href,ndcX:x.ndcX,ndcY:x.ndcY,depth:x.visible?E?1:Math.min(1,Math.max(0,(v-x.depth)/I)):0,visible:x.visible}))}function a(l){return l.replace(/[—–]/g,"-").replace(/[‘’]/g,"'").replace(/[“”]/g,'"').replace(/…/g,"...").replace(/[^\x20-\x7E]/g,"?")}function b(){return[{id:"palace:title",kind:"palace-hud",text:a("THE MEMORY PALACE"),x:-.94,y:.88,size:.062,color:r,depth:1,weight:1,revealSpan:18},{id:"palace:sub",kind:"palace-hud",text:a(`${T.length} ORGANS - CLICK A REGION TO ENTER`),x:-.94,y:.79,size:.032,color:n,depth:1,weight:.85,revealSpan:18,maxWidthEm:70}]}function F(){const l=Me(c(),{hoveredHref:_(d),dimUnhovered:!!_(d)});return[...b(),...l]}function w(){i==null||i.setText(F())}function O(l){if(!_(t))return null;const h=_(t).getBoundingClientRect();return h.width<=0||h.height<=0?null:{x:(l.clientX-h.left)/h.width*2-1,y:-((l.clientY-h.top)/h.height*2-1)}}function B(l){if(!_(t)||!s)return;const h=_(t).getBoundingClientRect(),g=Math.max(1e-4,h.width/Math.max(1,h.height)),v={x:l.x*Math.max(g,1),y:l.y/Math.min(g,1)},E=f??v,I={x:E.x+(v.x-E.x)*.35,y:E.y+(v.y-E.y)*.35};f=I,s.setCursorPreNdc(I.x,I.y,I.x-E.x,I.y-E.y)}function p(l){const h=O(l);if(!h||(B(h),!o))return;const g=o.pickAt(h.x,h.y),v=(g==null?void 0:g.href)??null;v!==_(d)&&(L(d,v,!0),o.setHovered((g==null?void 0:g.index)??-1),_(t)&&(_(t).style.cursor=v?"pointer":"default"))}function C(){f=null,L(d,null),o==null||o.setHovered(-1),s==null||s.setCursorPreNdc(999,999,0,0),_(t)&&(_(t).style.cursor="default")}function R(l){const h=O(l);if(!h||!o||o.isDiving)return;const g=o.pickAt(h.x,h.y);if(!g)return;o.startDive(g.href,E=>{U(`${k}${E}`)})||U(`${k}${g.href}`)}var S=Te();se("1dx67o8",l=>{te(()=>{oe.title="The Memory Palace · Vestige"})});var P=ne(S);ce(P,{demo:"recall-path",seed:"vestige-spatial-palace-v1",onframe:w,onready:m}),ie(S),ae(S,l=>L(t,l),()=>_(t)),G("pointerdown",S,R),G("pointermove",S,p),K("pointerleave",S,C),Q(u,S),ee()}W(["pointerdown","pointermove"]);export{Xe as component};

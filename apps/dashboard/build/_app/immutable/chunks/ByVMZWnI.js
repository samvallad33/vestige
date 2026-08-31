import"./Bzak7iHL.js";import{o as N}from"./BffzNaS8.js";import{p as T,k as W,h as j,a as C,b as H,g as i,s as m,d as q,t as K,f as J}from"./LfElJ0kU.js";import{i as Q,b as Z}from"./3OEMGmei.js";import{s as $}from"./BJDZGVD9.js";import{p as h}from"./DSKAVXWq.js";const ee=`
struct AmbientParams {
	time: f32,          // seconds (advances only when not reduced-motion)
	count: f32,         // active mote count (<= capacity)
	endangered: f32,    // 0..1 real endangered fraction — storm intensity
	fracture: f32,      // 0..1 real contradiction fraction — rift intensity
	due: f32,           // 0..1 real due-for-review fraction — pulse rate
	aspect: f32,        // viewport w/h
	accent_r: f32,      // route accent (rgb, 0..1) — one accent per §guardrail
	accent_g: f32,
	accent_b: f32,
	dpr: f32,
	reduced: f32,       // 1.0 = prefers-reduced-motion (freeze drift, keep field)
	_pad: f32,
};

@group(0) @binding(0) var<uniform> params: AmbientParams;

struct VSOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) @interpolate(flat) seed: vec2<f32>,
	@location(2) @interpolate(flat) tint: vec4<f32>, // rgb + retention
};

// Golden-ratio hash → a deterministic 0..1 per index (no PRNG state).
fn hash1(n: f32) -> f32 {
	return fract(sin(n * 12.9898) * 43758.5453);
}

const CORNERS = array<vec2<f32>, 6>(
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
	vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	if (ii >= u32(params.count)) {
		out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
		return out;
	}
	let fi = f32(ii);
	// Deterministic base position on a golden-angle lattice across the panel.
	let ga = 2.399963; // golden angle
	let rx = hash1(fi + 1.0);
	let ry = hash1(fi + 37.0);
	// Retention bucket for this mote: bias the population so the endangered
	// share of motes are low-retention (they sink). Real fraction drives it.
	let isEndangered = select(0.0, 1.0, ry < params.endangered);
	let retention = mix(0.55 + 0.4 * hash1(fi + 91.0), 0.05 + 0.18 * hash1(fi + 5.0), isEndangered);

	// Vertical home: high retention floats up, low sinks toward the floor.
	let homeY = mix(-0.9, 0.85, retention);
	// Gentle deterministic drift (frozen when reduced-motion): endangered motes
	// jitter more (the field is agitated by how much is being forgotten).
	let t = params.time;
	let sway = select(1.0, 0.0, params.reduced > 0.5);
	let turb = 0.02 + 0.10 * params.endangered;
	let driftX = sway * turb * sin(t * (0.3 + rx) + fi * ga);
	let driftY = sway * (0.015 + 0.05 * isEndangered) * sin(t * (0.5 + ry) + fi);
	let baseX = (rx * 2.0 - 1.0) * 0.98 + driftX;
	let baseY = homeY + driftY;

	// A rift: the fracture metric opens a horizontal tear that pushes motes apart.
	let rift = params.fracture * 0.25 * sin(baseX * 3.14159 + t * 0.2);
	let center = vec2<f32>(baseX, baseY + rift);

	// Mote size: small; endangered ones a touch larger + dimmer (last flare).
	let size = (0.010 + 0.014 * retention) * (1.0 + 0.4 * isEndangered);
	let corner = CORNERS[vi];
	out.clip = vec4<f32>(center.x + corner.x * size, center.y + corner.y * size * params.aspect, 0.0, 1.0);
	out.uv = corner;
	out.seed = vec2<f32>(rx, ry);
	out.tint = vec4<f32>(params.accent_r, params.accent_g, params.accent_b, retention);
	return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let retention = in.tint.a;
	// Soft mote: hot core + feathered halo, brightness scales with retention so
	// the endangered (dim) vs healthy (bright) split is LEGIBLE at a glance.
	let core = smoothstep(0.5, 0.0, d);
	let halo = pow(max(1.0 - d, 0.0), 2.0);
	// Due-for-review adds a slow global pulse so an overdue route breathes.
	let pulse = 0.85 + 0.15 * sin(params.time * (0.6 + params.due));
	let intensity = (core * 0.9 + halo * 0.35) * (0.25 + 0.75 * retention) * pulse;
	// Endangered motes shift toward a warmer, dimmer ember; healthy toward accent.
	let ember = vec3<f32>(0.62, 0.32, 0.22);
	let col = mix(ember, in.tint.rgb, smoothstep(0.2, 0.6, retention));
	return vec4<f32>(col * intensity, intensity * 0.9);
}
`;var te=J('<canvas class="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true"></canvas>');function ce(z,l){T(l,!0);let B=h(l,"endangered",3,0),L=h(l,"fracture",3,0),V=h(l,"due",3,0),x=h(l,"count",3,0),y=h(l,"accent",19,()=>[.13,.78,.87]),Y=h(l,"opacity",3,.5),r=q(null),f=q(!0);const k=220,_=520;N(()=>{if(!i(r))return;let e=null,c=null,u=null,w=null,d=null,p=0,g=!1,P=!0,A=!0,E=0,b=0;const F=window.matchMedia("(prefers-reduced-motion: reduce)"),t=new Float32Array(12),O=()=>Math.min(x()>0?x():_,window.innerWidth<640?k:_);function U(o){const a=Math.min(window.devicePixelRatio||1,window.innerWidth<640?2:1.5),s=Math.max(1,Math.floor((i(r).clientWidth||1)*a)),n=Math.max(1,Math.floor((i(r).clientHeight||1)*a));(i(r).width!==s||i(r).height!==n)&&(i(r).width=s,i(r).height=n),t[0]=E,t[1]=O(),t[2]=Math.max(0,Math.min(1,B())),t[3]=Math.max(0,Math.min(1,L())),t[4]=Math.max(0,Math.min(1,V())),t[5]=s/Math.max(1,n),t[6]=y()[0],t[7]=y()[1],t[8]=y()[2],t[9]=a,t[10]=F.matches?1:0,t[11]=0}async function X(){const o=navigator.gpu;if(!o){m(f,!1);return}let a=null;try{a=await o.requestAdapter()}catch{m(f,!1);return}if(!a||g){m(f,!1);return}try{e=await a.requestDevice()}catch{m(f,!1);return}if(g){e==null||e.destroy();return}const s=i(r).getContext("webgpu");if(!s){m(f,!1);return}c=s;const n=o.getPreferredCanvasFormat();c.configure({device:e,format:n,alphaMode:"premultiplied"}),d=e.createBuffer({label:"ambient-params",size:t.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const R=e.createShaderModule({label:"ambient-field",code:ee});u=e.createRenderPipeline({label:"ambient-field",layout:"auto",vertex:{module:R,entryPoint:"vs_main"},fragment:{module:R,entryPoint:"fs_main",targets:[{format:n,blend:{color:{srcFactor:"src-alpha",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}}}]},primitive:{topology:"triangle-list"}}),w=e.createBindGroup({label:"ambient-bind",layout:u.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:d}}]}),b=0,p=requestAnimationFrame(v)}function v(o){if(g||!e||!c||!u||!w||!d)return;if(!P||!A){p=requestAnimationFrame(v);return}b>0&&!F.matches&&(E+=Math.min(o-b,100)/1e3),b=o,U(),e.queue.writeBuffer(d,0,t);let a;try{a=c.getCurrentTexture().createView()}catch{p=requestAnimationFrame(v);return}const s=e.createCommandEncoder({label:"ambient-frame"}),n=s.beginRenderPass({colorAttachments:[{view:a,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(u),n.setBindGroup(0,w),n.draw(6,Math.floor(O())),n.end(),e.queue.submit([s.finish()]),p=requestAnimationFrame(v)}const S=()=>{P=document.visibilityState==="visible"};document.addEventListener("visibilitychange",S);const G=new IntersectionObserver(o=>{A=o.some(a=>a.isIntersecting)},{threshold:0});return G.observe(i(r)),X(),()=>{g=!0,cancelAnimationFrame(p),document.removeEventListener("visibilitychange",S),G.disconnect(),d==null||d.destroy(),e==null||e.destroy()}});var M=W(),D=j(M);{var I=e=>{var c=te();Z(c,u=>m(r,u),()=>i(r)),K(()=>$(c,`opacity: ${Y()??""}`)),C(e,c)};Q(D,e=>{i(f)&&e(I)})}C(z,M),H()}export{ce as A};

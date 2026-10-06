import{B as e,E as t,G as n,J as r,K as i,O as a,P as o,T as s,X as c,Y as l,Z as u,a as d,at as f,c as p,n as m,ot as h,q as g,r as _,ut as v,w as ee}from"./DHlIFcq6.js";import{s as y}from"./DL5zNdDR.js";import"./xihTtKlq.js";function b(e){let t=1779033703^e.length;for(let n=0;n<e.length;n++)t=Math.imul(t^e.charCodeAt(n),2654435761),t=t<<13|t>>>19;return function(){let e=t+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}}function x(e){return function(){let t=e+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}}var S=class{fps;loopFrames;seedStr;_frame;_totalFrames;_rng;constructor(e){this.fps=e.fps??60,this.loopFrames=e.loopFrames??720,this.seedStr=e.seed,this._frame=0,this._totalFrames=0;let t=b(this.seedStr)();this._rng=x(Math.floor(t*2**32))}tick(){return this._frame=(this._frame+1)%this.loopFrames,this._totalFrames++,this.state}get state(){return{frame:this._frame,phase:this._frame/this.loopFrames,rng:this._rng,totalFrames:this._totalFrames}}reset(){this._frame=0,this._totalFrames=0;let e=b(this.seedStr)();this._rng=x(Math.floor(e*2**32))}get loopDuration(){return this.loopFrames/this.fps}get framesPerLoop(){return this.loopFrames}};function C(e,t,n,r){let i=Math.PI*(3-Math.sqrt(5)),a=1-e/(t-1||1)*2,o=Math.sqrt(1-a*a),s=i*e,c=Math.cos(s)*o,l=Math.sin(s)*o,u=(r()-.5)*.1*n,d=(r()-.5)*.1*n,f=(r()-.5)*.1*n;return[c*n+u,a*n+d,l*n+f]}var w=[`recall-path`,`engram-birth`,`salience-rescue`,`forgetting-horizon`,`firewall`];function T(e){return w.includes(e)}var E={posRadius:0,velRetention:4,colorFlags:8,demo:12},D={isCenter:1,suppressed:2,isAha:4,isFailure:8,isConfusion:16},O={recall:0,backwardCause:1,probe:2},k={none:0,firewall:1,dreamStorm:2,causalRecall:3,birth:4},A={frame:0,loopPhase:1,nodeCount:2,edgeCount:3,pathCount:4,pulse:5,viewportW:6,viewportH:7,brightness:8,demoId:9,time:10,captureMode:11,liveKind:12,liveFrame:13,liveEnergy:14,projectionDays:15,cursorX:16,cursorY:17,cursorVx:18,cursorVy:19};function j(e){let t=w.indexOf(e);return t<0?0:t}function te(e,t){return{id:e.id,index:t,label:e.label,type:e.type,retention:typeof e.retention==`number`?e.retention:0,tags:Array.isArray(e.tags)?e.tags:[],isCenter:!!e.isCenter,suppressed:(e.suppression_count??0)>0,stability:typeof e.stability==`number`?e.stability:void 0,lastAccessed:typeof e.lastAccessed==`string`?e.lastAccessed:void 0,createdAt:typeof e.createdAt==`string`?e.createdAt:void 0}}function M(e,t){let n=Math.max(1,e>>1),r=Math.max(1,t>>1),i=Math.min(6,Math.max(1,1+Math.floor(Math.log2(Math.min(n,r)/8))));return{baseW:n,baseH:r,mipCount:i,sizes:Array.from({length:i},(e,t)=>[Math.max(1,n>>t),Math.max(1,r>>t)])}}var N=.18,ne=`
// Tuning constants — interpolated from post.wgsl.ts (TS single source of truth).
const BLOOM_STRENGTH: f32 = ${N};
const BLOOM_CHROMATIC_TEXELS: f32 = 0;
const GRAIN_AMP: f32 = ${2/255};
const VIGNETTE_LIFT: f32 = 0.85;
const VIGNETTE_TAN: f32 = 0.62;

// Params layout — VERBATIM from render-nodes.wgsl.ts (types.PARAMS_FLOATS).
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

// Globally unique bindings — each entry point statically uses a subset; the
// explicit bind group layouts in post-chain.ts carry exactly what each
// pipeline needs (blur: 1+2, composite: 0+2+3+4).
@group(0) @binding(0) var<uniform> params: Params;    // composite only
@group(0) @binding(1) var src: texture_2d<f32>;       // blur chain input
@group(0) @binding(2) var samp: sampler;              // shared
@group(0) @binding(3) var scene_tex: texture_2d<f32>; // composite only
@group(0) @binding(4) var bloom_tex: texture_2d<f32>; // composite only (FULL-mip view)

struct FSOut {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
};

// Fullscreen triangle from bit math — no vertex buffer, no arrays.
// vi 0/1/2 → clip (-1,-1) (3,-1) (-1,3); uv y flipped so uv(0,0) = top-left.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FSOut {
	let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
	var out: FSOut;
	out.pos = vec4f(xy, 0.0, 1.0);
	out.uv = vec2f(xy.x, -xy.y) * 0.5 + 0.5;
	return out;
}

fn luma(c: vec3f) -> f32 {
	return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ---------------------------------------------------------------------------
// Bloom downsample — 13-tap Jimenez (SIGGRAPH 2014 "Next Generation Post
// Processing in Call of Duty: Advanced Warfare"), taps fully unrolled.
//
//   a  b  c        outer ring at ±2 texels
//    j  k          inner ring at ±1 texels
//   d  e  f        e = center
//    l  m
//   g  h  i
//
// Grouped as 5 overlapping 4-tap boxes: center box (the four inner taps)
// weight 0.5, four corner boxes weight 0.125 each. A flat field reproduces
// itself EXACTLY (0.5 + 4·0.125 = 1) — that exactness is what the void
// preimage in tone-reference.ts depends on.
// ---------------------------------------------------------------------------

@fragment
fn fs_downsample_karis(in: FSOut) -> @location(0) vec4f {
	let ts = 1.0 / vec2f(textureDimensions(src));
	let a = textureSampleLevel(src, samp, in.uv + vec2f(-2.0, -2.0) * ts, 0.0).rgb;
	let b = textureSampleLevel(src, samp, in.uv + vec2f( 0.0, -2.0) * ts, 0.0).rgb;
	let c = textureSampleLevel(src, samp, in.uv + vec2f( 2.0, -2.0) * ts, 0.0).rgb;
	let d = textureSampleLevel(src, samp, in.uv + vec2f(-2.0,  0.0) * ts, 0.0).rgb;
	let e = textureSampleLevel(src, samp, in.uv,                          0.0).rgb;
	let f = textureSampleLevel(src, samp, in.uv + vec2f( 2.0,  0.0) * ts, 0.0).rgb;
	let g = textureSampleLevel(src, samp, in.uv + vec2f(-2.0,  2.0) * ts, 0.0).rgb;
	let h = textureSampleLevel(src, samp, in.uv + vec2f( 0.0,  2.0) * ts, 0.0).rgb;
	let i = textureSampleLevel(src, samp, in.uv + vec2f( 2.0,  2.0) * ts, 0.0).rgb;
	let j = textureSampleLevel(src, samp, in.uv + vec2f(-1.0, -1.0) * ts, 0.0).rgb;
	let k = textureSampleLevel(src, samp, in.uv + vec2f( 1.0, -1.0) * ts, 0.0).rgb;
	let l = textureSampleLevel(src, samp, in.uv + vec2f(-1.0,  1.0) * ts, 0.0).rgb;
	let m = textureSampleLevel(src, samp, in.uv + vec2f( 1.0,  1.0) * ts, 0.0).rgb;

	let box_c  = (j + k + l + m) * 0.25;
	let box_tl = (a + b + d + e) * 0.25;
	let box_tr = (b + c + e + f) * 0.25;
	let box_bl = (d + e + g + h) * 0.25;
	let box_br = (e + f + h + i) * 0.25;

	// Karis average (fireflies killer) — used ONLY on the full→mip0 hop.
	// Each box is additionally weighted 1/(1 + luma) and the sum RENORMALIZED:
	// on a flat field every Karis factor is equal, so the result is exact.
	let w_c  = 0.5   / (1.0 + luma(box_c));
	let w_tl = 0.125 / (1.0 + luma(box_tl));
	let w_tr = 0.125 / (1.0 + luma(box_tr));
	let w_bl = 0.125 / (1.0 + luma(box_bl));
	let w_br = 0.125 / (1.0 + luma(box_br));
	let sum = w_c * box_c + w_tl * box_tl + w_tr * box_tr + w_bl * box_bl + w_br * box_br;
	return vec4f(sum / (w_c + w_tl + w_tr + w_bl + w_br), 1.0);
}

@fragment
fn fs_downsample(in: FSOut) -> @location(0) vec4f {
	let ts = 1.0 / vec2f(textureDimensions(src));
	let a = textureSampleLevel(src, samp, in.uv + vec2f(-2.0, -2.0) * ts, 0.0).rgb;
	let b = textureSampleLevel(src, samp, in.uv + vec2f( 0.0, -2.0) * ts, 0.0).rgb;
	let c = textureSampleLevel(src, samp, in.uv + vec2f( 2.0, -2.0) * ts, 0.0).rgb;
	let d = textureSampleLevel(src, samp, in.uv + vec2f(-2.0,  0.0) * ts, 0.0).rgb;
	let e = textureSampleLevel(src, samp, in.uv,                          0.0).rgb;
	let f = textureSampleLevel(src, samp, in.uv + vec2f( 2.0,  0.0) * ts, 0.0).rgb;
	let g = textureSampleLevel(src, samp, in.uv + vec2f(-2.0,  2.0) * ts, 0.0).rgb;
	let h = textureSampleLevel(src, samp, in.uv + vec2f( 0.0,  2.0) * ts, 0.0).rgb;
	let i = textureSampleLevel(src, samp, in.uv + vec2f( 2.0,  2.0) * ts, 0.0).rgb;
	let j = textureSampleLevel(src, samp, in.uv + vec2f(-1.0, -1.0) * ts, 0.0).rgb;
	let k = textureSampleLevel(src, samp, in.uv + vec2f( 1.0, -1.0) * ts, 0.0).rgb;
	let l = textureSampleLevel(src, samp, in.uv + vec2f(-1.0,  1.0) * ts, 0.0).rgb;
	let m = textureSampleLevel(src, samp, in.uv + vec2f( 1.0,  1.0) * ts, 0.0).rgb;

	let box_c  = (j + k + l + m) * 0.25;
	let box_tl = (a + b + d + e) * 0.25;
	let box_tr = (b + c + e + f) * 0.25;
	let box_bl = (d + e + g + h) * 0.25;
	let box_br = (e + f + h + i) * 0.25;
	return vec4f(box_c * 0.5 + (box_tl + box_tr + box_bl + box_br) * 0.125, 1.0);
}

// ---------------------------------------------------------------------------
// Bloom upsample — 9-tap 3×3 tent, 1/16·[1 2 1; 2 4 2; 1 2 1], radius = one
// SOURCE-mip texel. Rendered with additive one/one blending onto the stored
// downsample of the destination mip (accumulate-up-the-chain). The resulting
// DC gain of exactly mipCount is normalized in fs_composite.
// ---------------------------------------------------------------------------

@fragment
fn fs_upsample_tent(in: FSOut) -> @location(0) vec4f {
	let ts = 1.0 / vec2f(textureDimensions(src));
	let a = textureSampleLevel(src, samp, in.uv + vec2f(-1.0, -1.0) * ts, 0.0).rgb;
	let b = textureSampleLevel(src, samp, in.uv + vec2f( 0.0, -1.0) * ts, 0.0).rgb;
	let c = textureSampleLevel(src, samp, in.uv + vec2f( 1.0, -1.0) * ts, 0.0).rgb;
	let d = textureSampleLevel(src, samp, in.uv + vec2f(-1.0,  0.0) * ts, 0.0).rgb;
	let e = textureSampleLevel(src, samp, in.uv,                          0.0).rgb;
	let f = textureSampleLevel(src, samp, in.uv + vec2f( 1.0,  0.0) * ts, 0.0).rgb;
	let g = textureSampleLevel(src, samp, in.uv + vec2f(-1.0,  1.0) * ts, 0.0).rgb;
	let h = textureSampleLevel(src, samp, in.uv + vec2f( 0.0,  1.0) * ts, 0.0).rgb;
	let i = textureSampleLevel(src, samp, in.uv + vec2f( 1.0,  1.0) * ts, 0.0).rgb;
	let sum = (a + c + g + i) + (b + d + f + h) * 2.0 + e * 4.0;
	return vec4f(sum * (1.0 / 16.0), 1.0);
}

// ---------------------------------------------------------------------------
// Composite — bloom-add → PBR Neutral → grain → vignette (order is mandated).
// ---------------------------------------------------------------------------

// Khronos PBR Neutral — EXACT port of the Khronos reference implementation.
// Hue-preserving; the FSRS palette keeps its channel ordering. Pinned to the
// CPU mirror in post/tone-reference.ts (pbrNeutralReference) — keep in
// lockstep, the void-preimage tests run against the mirror.
fn pbr_neutral(color_in: vec3f) -> vec3f {
	let start_compression = 0.8 - 0.04;
	let desaturation = 0.15;
	var color = color_in;
	let x = min(color.r, min(color.g, color.b));
	// WGSL select(false_value, true_value, condition) — argument order trap.
	let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
	color = color - vec3f(offset);
	let peak = max(color.r, max(color.g, color.b));
	if (peak < start_compression) {
		return color;
	}
	let d = 1.0 - start_compression;
	let new_peak = 1.0 - d * d / (peak + d - start_compression);
	color = color * (new_peak / peak);
	let g = 1.0 / (desaturation * (peak - new_peak) + 1.0);
	// mix weight = 1 - g per the Khronos spec.
	return mix(color, vec3f(new_peak), 1.0 - g);
}

// PCG hash — integers only, 24-bit-exact output in [0, 1). Deterministic.
fn pcg(v: u32) -> u32 {
	var s = v * 747796405u + 2891336453u;
	let t = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (t >> 22u) ^ t;
}

fn hashf(p: vec2u, f: u32) -> f32 {
	return f32(pcg(p.x ^ pcg(p.y ^ pcg(f))) >> 8u) / 16777216.0;
}

@fragment
fn fs_composite(in: FSOut) -> @location(0) vec4f {
	let pix = vec2u(in.pos.xy);
	// Exact 1:1 fetch (alpha discarded — see module header).
	let scene = textureLoad(scene_tex, pix, 0).rgb;

	// Bloom, normalized by the mip count: the additive up-chain has DC gain
	// exactly mipCount, so /mips makes flat-field gain exactly 1 — the void
	// preimage holds and brightness is viewport-stable. Chromatic dispersion
	// rides the bloom term ONLY (BLOOM_CHROMATIC_TEXELS = 0.0 kills it).
	let mips = f32(textureNumLevels(bloom_tex));
	let dims = vec2f(textureDimensions(bloom_tex));
	let dvec = in.uv - vec2f(0.5);
	let off = dvec * (BLOOM_CHROMATIC_TEXELS * dot(dvec, dvec) * 4.0) / dims;
	let bloom = vec3f(
		textureSampleLevel(bloom_tex, samp, in.uv - off, 0.0).r,
		textureSampleLevel(bloom_tex, samp, in.uv,       0.0).g,
		textureSampleLevel(bloom_tex, samp, in.uv + off, 0.0).b
	) / mips;

	var c = pbr_neutral(scene + BLOOM_STRENGTH * bloom);

	// Seeded TPDF film grain (post-tonemap dither): keyed to the WRAPPED loop
	// frame → 720-periodic and capture-pinned. Full strength in the shadows
	// (kills #05060a banding), fades out of highlights.
	let f = u32(params.frame + 0.5);
	let n = hashf(pix, f) + hashf(pix ^ vec2u(0x9E3779B9u, 0x85EBCA6Bu), f) - 1.0;
	let w = 1.0 - smoothstep(0.0, 0.8, luma(c));
	c += GRAIN_AMP * n * w;

	// cos⁴ vignette: cos⁴θ = (1 + r²·tan²)⁻², aspect-normalized so rn = 1.0
	// exactly at the corners regardless of viewport shape. Lifted floor keeps
	// it an observatory, not a tunnel.
	let ar = vec2f(params.viewport_w / max(params.viewport_h, 1.0), 1.0);
	let rn = length((in.uv * 2.0 - 1.0) * ar) / length(ar);
	let k = rn * rn * VIGNETTE_TAN * VIGNETTE_TAN;
	c *= mix(VIGNETTE_LIFT, 1.0, 1.0 / ((1.0 + k) * (1.0 + k)));

	// NO gamma encode — display-referred pass-through, matching the pre-post
	// look where shader outputs went straight to the swapchain.
	return vec4f(c, 1.0);
}
`,P=`rgba16float`,re={color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}},ie=class{device;paramsBuffer;samp;blurLayout;compositeLayout;pipeDownFirst;pipeDown;pipeUp;pipeComposite;width=0;height=0;plan=null;sceneTex=null;_sceneView=null;bloomTex=null;mipViews=[];bloomFullView=null;downBind=[];upBind=[];compositeBind=null;constructor(e,t,n){this.device=e,this.paramsBuffer=t,this.samp=e.createSampler({label:`observatory-post-sampler`,minFilter:`linear`,magFilter:`linear`,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`});let r=e.createShaderModule({label:`observatory-post`,code:ne});this.blurLayout=e.createBindGroupLayout({label:`observatory-post-blur-layout`,entries:[{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}}]}),this.compositeLayout=e.createBindGroupLayout({label:`observatory-post-composite-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}}]});let i=e.createPipelineLayout({label:`observatory-post-blur-pipe-layout`,bindGroupLayouts:[this.blurLayout]}),a=e.createPipelineLayout({label:`observatory-post-composite-pipe-layout`,bindGroupLayouts:[this.compositeLayout]}),o=(t,n,i,a,o)=>e.createRenderPipeline({label:t,layout:n,vertex:{module:r,entryPoint:`vs_fullscreen`},fragment:{module:r,entryPoint:i,targets:[{format:a,blend:o}]},primitive:{topology:`triangle-list`}});this.pipeDownFirst=o(`observatory-post-down-karis`,i,`fs_downsample_karis`,P),this.pipeDown=o(`observatory-post-down`,i,`fs_downsample`,P),this.pipeUp=o(`observatory-post-up`,i,`fs_upsample_tent`,P,re),this.pipeComposite=o(`observatory-post-composite`,a,`fs_composite`,n)}get sceneView(){if(!this._sceneView)throw Error(`PostChain.ensure() must run before sceneView is used`);return this._sceneView}ensure(e,t){let n=Math.max(1,Math.floor(e)),r=Math.max(1,Math.floor(t));if(n===this.width&&r===this.height&&this.sceneTex!==null)return;this.width=n,this.height=r,this.sceneTex?.destroy(),this.bloomTex?.destroy(),this.sceneTex=this.device.createTexture({label:`observatory-scene-hdr`,size:[n,r],format:P,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this._sceneView=this.sceneTex.createView({label:`observatory-scene-hdr-view`});let i=M(n,r);this.plan=i,this.bloomTex=this.device.createTexture({label:`observatory-bloom-mips`,size:[i.baseW,i.baseH],format:P,mipLevelCount:i.mipCount,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});let a=this.bloomTex;this.mipViews=Array.from({length:i.mipCount},(e,t)=>a.createView({label:`observatory-bloom-mip-${t}`,baseMipLevel:t,mipLevelCount:1})),this.bloomFullView=a.createView({label:`observatory-bloom-full`});let o=this._sceneView;this.downBind=this.mipViews.map((e,t)=>this.device.createBindGroup({label:`observatory-bloom-down-bind-${t}`,layout:this.blurLayout,entries:[{binding:1,resource:t===0?o:this.mipViews[t-1]},{binding:2,resource:this.samp}]})),this.upBind=[];for(let e=0;e+1<i.mipCount;e++)this.upBind.push(this.device.createBindGroup({label:`observatory-bloom-up-bind-${e}`,layout:this.blurLayout,entries:[{binding:1,resource:this.mipViews[e+1]},{binding:2,resource:this.samp}]}));this.compositeBind=this.device.createBindGroup({label:`observatory-post-composite-bind`,layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:this.paramsBuffer}},{binding:2,resource:this.samp},{binding:3,resource:o},{binding:4,resource:this.bloomFullView}]})}encode(e,t){let n=this.plan;if(!n||!this.compositeBind)return;let r=n.mipCount;for(let t=0;t<r;t++){let n=e.beginRenderPass({label:`observatory-bloom-down-${t}`,colorAttachments:[{view:this.mipViews[t],loadOp:`clear`,storeOp:`store`}]});n.setPipeline(t===0?this.pipeDownFirst:this.pipeDown),n.setBindGroup(0,this.downBind[t]),n.draw(3),n.end()}for(let t=r-2;t>=0;t--){let n=e.beginRenderPass({label:`observatory-bloom-up-${t}`,colorAttachments:[{view:this.mipViews[t],loadOp:`load`,storeOp:`store`}]});n.setPipeline(this.pipeUp),n.setBindGroup(0,this.upBind[t]),n.draw(3),n.end()}let i=e.beginRenderPass({label:`observatory-post-composite`,colorAttachments:[{view:t,loadOp:`clear`,storeOp:`store`}]});i.setPipeline(this.pipeComposite),i.setBindGroup(0,this.compositeBind),i.draw(3),i.end()}dispose(){this.sceneTex?.destroy(),this.bloomTex?.destroy(),this.sceneTex=null,this.bloomTex=null,this._sceneView=null,this.bloomFullView=null,this.mipViews=[],this.downBind=[],this.upBind=[],this.compositeBind=null,this.plan=null,this.width=0,this.height=0}},F=Math.sqrt(5/255/6.25),I=F-5/255,ae={r:F/(1+N),g:(6/255+I)/(1+N),b:(10/255+I)/(1+N),a:1},L=[500,1e3,2e3,4e3,8e3],oe=class e{canvas;device=null;context=null;format=`bgra8unorm`;clock;demo;freezeFrame;rafId=0;running=!1;disposed=!1;recovering=!1;maxDpr;onFrame;lastRenderTs=-1/0;visibilityListenerAttached=!1;params=new Float32Array(20);paramsBuffer=null;passes=[];post=null;_status={state:`booting`};statusListeners=new Set;preFrameHook=null;lastRafTs=0;fpsEstimate=0;accumulatorMs=0;static FIXED_DT_MS=1e3/60;paused=!1;surge=0;surgeAt=0;kick(e=1){this.freezeFrame!==null||this.exportMode||(this.surge=Math.min(1.2,Math.max(this.surge,e)),this.surgeAt=performance.now(),this.requestRender())}constructor(e){this.canvas=e.canvas,this.demo=e.demo,this.maxDpr=e.maxDpr??2,this.onFrame=e.onFrame,this.clock=new S({seed:e.seed}),this.freezeFrame=typeof e.freezeFrame==`number`&&Number.isFinite(e.freezeFrame)?(Math.floor(e.freezeFrame)%this.clock.framesPerLoop+this.clock.framesPerLoop)%this.clock.framesPerLoop:null,this.params[8]=1,this.setCursorPreNdc(999,999,0,0)}get status(){return this._status}get gpuDevice(){return this.device}get presentationFormat(){return this.format}get sceneFormat(){return P}get demoClock(){return this.clock}onStatus(e){return this.statusListeners.add(e),e(this._status),()=>this.statusListeners.delete(e)}setStatus(e){this._status=e;for(let t of this.statusListeners)t(e)}addPass(e){this.passes.push(e)}removePass(e){let t=this.passes.indexOf(e);t!==-1&&(this.passes.splice(t,1),e.dispose?.())}clearPasses(){for(let e of this.passes)e.dispose?.();this.passes.length=0}setPreFrameHook(e){this.preFrameHook=e}get totalFrames(){return this.clock.state.totalFrames}setCursorPreNdc(e,t,n=0,r=0){this.params[A.cursorX]=Number.isFinite(e)?e:999,this.params[A.cursorY]=Number.isFinite(t)?t:999,this.params[A.cursorVx]=Number.isFinite(n)?n:0,this.params[A.cursorVy]=Number.isFinite(r)?r:0}setPaused(e){this.paused=e,this.requestRender()}get isPaused(){return this.paused}requestRender(){this.lastRenderTs=-1/0}get wallNowMs(){return Date.now()}async start(){if(this.disposed)return!1;let e=navigator.gpu;if(!e)return this.setStatus({state:`unsupported`,reason:`WebGPU is not available in this browser.`}),!1;let t=null;try{t=await e.requestAdapter()}catch(e){return this.setStatus({state:`error`,reason:e instanceof Error?e.message:`requestAdapter failed`}),!1}if(!t)return this.setStatus({state:`unsupported`,reason:`No suitable GPU adapter found.`}),!1;try{this.device=await t.requestDevice()}catch(e){return this.setStatus({state:`error`,reason:e instanceof Error?e.message:`requestDevice failed`}),!1}if(this.disposed)return this.device?.destroy(),this.device=null,!1;this.device.lost.then(e=>{this.disposed||e.reason===`destroyed`||(this.stopLoop(),this.recoverFromDeviceLoss(e.message))}),this.device.onuncapturederror=e=>{console.error(`[observatory] WebGPU error:`,e.error.message)};let n=this.canvas.getContext(`webgpu`);return n?(this.context=n,this.format=e.getPreferredCanvasFormat(),this.configureContext(),this.paramsBuffer=this.device.createBuffer({label:`observatory-params`,size:this.params.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.post=new ie(this.device,this.paramsBuffer,this.format),this.setStatus({state:`running`}),this.attachVisibilityListener(),this.resumeLoop(),!0):(this.setStatus({state:`error`,reason:`Could not get webgpu canvas context.`}),!1)}async recoverFromDeviceLoss(e){if(!this.recovering){this.recovering=!0;try{for(let t=0;t<L.length;t++)if(this.disposed||(this.setStatus({state:`recovering`,attempt:t+1,reason:e}),await new Promise(e=>setTimeout(e,L[t])),this.disposed)||(this.releaseDeviceResources(),await this.start()))return;this.disposed||this.setStatus({state:`error`,reason:`GPU device lost: ${e}`})}finally{this.recovering=!1}}}releaseDeviceResources(){this.clearPasses(),this.post?.dispose(),this.post=null,this.paramsBuffer?.destroy(),this.paramsBuffer=null,this.context=null,this.device=null}resize(){if(!this.device||!this.context)return;let e=Math.min(window.devicePixelRatio||1,this.maxDpr),t=Math.max(1,Math.floor(this.canvas.clientWidth*e)),n=Math.max(1,Math.floor(this.canvas.clientHeight*e));(this.canvas.width!==t||this.canvas.height!==n)&&(this.canvas.width=t,this.canvas.height=n,this.configureContext(),this.post?.ensure(t,n))}configureContext(){this.device&&this.context&&this.context.configure({device:this.device,format:this.format,alphaMode:`opaque`})}attachVisibilityListener(){this.visibilityListenerAttached||typeof document>`u`||(document.addEventListener(`visibilitychange`,this.handleVisibilityChange),this.visibilityListenerAttached=!0)}handleVisibilityChange=()=>{if(!(typeof document>`u`)){if(document.hidden){this.stopLoop();return}this.resumeLoop()}};resumeLoop(){!this.running&&!this.disposed&&this.device&&this.context&&this.paramsBuffer&&this.post&&(this.running=!0,this.lastRafTs=0,this.accumulatorMs=0,this.requestRender(),this.rafId=requestAnimationFrame(this.frame))}frameRateFor(e){let t=0;for(let n of this.passes){let r=n.targetFrameRate?.(e);if(typeof r!=`number`||!Number.isFinite(r)){t=60;continue}t=Math.max(t,r)}return Math.max(1,Math.min(60,t||60))}frame=t=>{if(!this.running||!this.device||!this.context||!this.paramsBuffer||!this.post)return;let n=0;for(this.lastRafTs>0&&(n=t-this.lastRafTs),this.lastRafTs=t,this.accumulatorMs+=Math.min(n,250);this.accumulatorMs>=e.FIXED_DT_MS;)this.paused||this.clock.tick(),this.accumulatorMs-=e.FIXED_DT_MS;let r=this.clock.state,i=this.freezeFrame??r.frame,a=1e3/this.frameRateFor(i);if(t-this.lastRenderTs<a){this.rafId=requestAnimationFrame(this.frame);return}if(!this.encodeAndSubmit(i,r.totalFrames)){this.rafId=requestAnimationFrame(this.frame);return}Number.isFinite(this.lastRenderTs)&&t>this.lastRenderTs&&(this.fpsEstimate=Math.round(1e3/(t-this.lastRenderTs))),this.lastRenderTs=t,this.onFrame?.(i,this.fpsEstimate),this.rafId=requestAnimationFrame(this.frame)};encodeAndSubmit(e,t){if(!this.device||!this.context||!this.paramsBuffer||!this.post)return!1;let n=e/this.clock.framesPerLoop,r=this.params;if(r[0]=e,r[1]=n,r[5]=.5+.5*Math.sin(2*Math.PI*4*n),this.surge>0){r[5]=Math.min(1.6,r[5]+this.surge);let e=performance.now(),t=Math.min(.1,Math.max(0,(e-this.surgeAt)/1e3));this.surgeAt=e,this.surge=this.surge<.004?0:this.surge*.06**t}r[6]=this.canvas.width,r[7]=this.canvas.height,r[9]=j(this.demo),r[10]=this.freezeFrame!==null||this.exportMode?e/60:t/60,r[11]=this.freezeFrame===null?0:1,this.exportMode||this.preFrameHook?.(t),this.device.queue.writeBuffer(this.paramsBuffer,0,r);let i;try{i=this.context.getCurrentTexture()}catch{return!1}this.post.ensure(i.width,i.height);let a=i.createView(),o=this.device.createCommandEncoder({label:`observatory-frame`});for(let t of this.passes)t.compute?.(o,e);let s=o.beginRenderPass({label:`observatory-main`,colorAttachments:[{view:this.post.sceneView,clearValue:ae,loadOp:`clear`,storeOp:`store`}]});for(let t of this.passes)t.render?.(s,e);return s.end(),this.post.encode(o,a),this.device.queue.submit([o.finish()]),!0}exportMode=!1;get canvasElement(){return this.canvas}beginExport(){this.stopLoop(),this.exportMode=!0,this.clock.reset()}endExport(){this.exportMode=!1,this.resumeLoop()}async renderExportFrame(e){if(!this.exportMode)throw Error(`renderExportFrame outside beginExport()`);if(!this.device)throw Error(`export: no GPU device`);e&&this.clock.tick();let t=this.clock.state,n=t.frame;if(!this.encodeAndSubmit(n,t.totalFrames))throw Error(`export: canvas has no texture (zero-sized or hidden)`);return await this.device.queue.onSubmittedWorkDone(),n}stopLoop(){this.running=!1,this.rafId!==0&&(cancelAnimationFrame(this.rafId),this.rafId=0)}dispose(){this.disposed||(this.disposed=!0,this.stopLoop(),this.visibilityListenerAttached&&typeof document<`u`&&(document.removeEventListener(`visibilitychange`,this.handleVisibilityChange),this.visibilityListenerAttached=!1),this.paramsBuffer?.destroy(),this.paramsBuffer=null,this.post?.dispose(),this.post=null,this.device?.destroy(),this.device=null,this.context=null,this.passes=[],this.setStatus({state:`disposed`}),this.statusListeners.clear())}},se=a(`<div id="webgpu-field-status" class="fallback svelte-16248mg" role="status" aria-live="polite"><div class="fallback-title svelte-16248mg">GPU DEVICE LOST · RECOVERING</div> <div class="fallback-reason svelte-16248mg"> </div></div>`),ce=a(`<div id="webgpu-field-status" class="fallback svelte-16248mg" role="alert"><div class="fallback-title svelte-16248mg">3D MEMORY FIELD UNAVAILABLE</div> <div class="fallback-reason svelte-16248mg">This browser or device could not create a WebGPU graphics context, so this
			visual field has not rendered.</div> <div class="fallback-hint svelte-16248mg">Your local memories have not been changed. Use the persistent navigation to
			continue in another tool, or open this view in a WebGPU-capable browser.</div></div>`),le=a(`<canvas class="observatory-canvas svelte-16248mg" aria-label="Vestige 3D memory field" aria-describedby="webgpu-field-status"></canvas> <!>`,1);function ue(a,y){h(y,!0);let b=d(y,`freezeFrame`,3,null),x=d(y,`maxDpr`,3,2),S,C=null,w=u(l({state:`booting`})),T=null,E=null;_(()=>{C=new oe({canvas:S,demo:y.demo,seed:y.seed,freezeFrame:b(),maxDpr:x(),onFrame:(e,t)=>y.onframe?.(e,t)});let e=!1;T=C.onStatus(t=>{c(w,t,!0),t.state===`recovering`&&(e=!0),t.state===`running`&&e&&C&&(e=!1,C.resize(),y.onready?.(C))}),E=new ResizeObserver(()=>C?.resize()),E.observe(S),C.start().then(e=>{e&&C&&(C.resize(),y.onready?.(C))})}),m(()=>{T?.(),E?.disconnect(),C?.dispose(),C=null});var D=le(),O=i(D);p(O,e=>S=e,()=>S);var k=r(O,2),A=i=>{var a=se(),c=r(n(a),2),l=g(c);v(a),e(()=>s(l,`Re-acquiring the graphics device (attempt ${o(w).attempt??``} of 5). ${o(w).reason??``}`)),t(i,a)},j=e=>{var n=ce();t(e,n)};ee(k,e=>{o(w).state===`recovering`?e(A):(o(w).state===`unsupported`||o(w).state===`error`)&&e(j,1)}),t(a,D),f()}function R(e){let t=/^#?([0-9a-fA-F]{6})$/.exec(e.trim());if(!t)return[2/255,3/255,7/255];let n=parseInt(t[1],16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}var z={blackwater:`#020307`,anaerobic:`#07100D`,cyanFog:`#0B171B`,sediment:`#11140A`},B={luciferin:`#E9FFB7`,healthy:`#A8FF5E`,recall:`#29F2A9`,bridge:`#1BD6FF`,latent:`#315CFF`,debt:`#8A4B18`,extinction:`#2A160B`},V=[B.extinction,B.debt,B.healthy,B.luciferin],H={trustMembrane:`#F4F1D0`,caution:`#FFD166`,veto:`#FF3B30`,suppressionScar:`#B90D2B`,labile:`#FF7A1A`},U={forward:`#00F5D4`,retrograde:`#FF2DF7`,receiptSpark:`#FFFFFF`},W={validRing:`#6BFFB8`,txShadow:`#7C6CFF`,supersession:`#FFB000`};function G(e){let t=Math.max(0,Math.min(1,e)),n=V.map(R),r=t*(n.length-1),i=Math.min(n.length-2,Math.floor(r)),a=r-i,o=n[i],s=n[i+1];return[o[0]+(s[0]-o[0])*a,o[1]+(s[1]-o[1])*a,o[2]+(s[2]-o[2])*a]}var de=[58/255,68/255,76/255],K=[1,.78,.36];function q(e,t,n){let r=n<0?0:n>1?1:n;return[e[0]+(t[0]-e[0])*r,e[1]+(t[1]-e[1])*r,e[2]+(t[2]-e[2])*r]}function fe(e,t=!1){let n=Math.max(0,Math.min(1,Number.isFinite(e)?e:0)),r=J((n-.3)/.42),i=q(de,G(n),r),a=J((n-.82)/.18);return i=q(i,K,a*.85),t?(i=q(i,K,.7),i=q(i,[1,1,1],.35)):i=q(i,[1,1,1],J((n-.93)/.07)*.5),i}function pe(e,t=!1){return t?1:.18+.72*J((Math.max(0,Math.min(1,Number.isFinite(e)?e:0))-.15)/.7)}function J(e){let t=e<0?0:e>1?1:e;return t*t*(3-2*t)}var me={MemoryCreated:B.healthy,SearchPerformed:U.forward,ActivationSpread:B.recall,ImportanceScored:W.supersession,RetentionDecayed:B.debt,ConnectionDiscovered:U.forward,DeepReferenceCompleted:B.luciferin,BackfillFired:U.retrograde,CausalReceipt:U.retrograde,MemorySuppressed:H.suppressionScar,MemoryUnsuppressed:H.labile,MemoryPromoted:B.luciferin,MemoryDemoted:B.debt,MemoryPrOpened:H.caution,MemoryPrDecided:H.trustMembrane,HookVerdictRecorded:H.veto,TraceEvent:U.forward,DreamStarted:W.txShadow,DreamCompleted:W.validRing,Rac1CascadeSwept:H.suppressionScar};function he(e){return R(me[e]??z.blackwater)}function ge(e){return .003+.015*Math.max(0,Math.min(1,e))}var _e=32,ve=126,Y=63,ye=.6,be=1.32,xe=`...`;function Se(e){return new Map(e.glyphs.map(e=>[e.unicode,e]))}function X(e){let t=e.codePointAt(0)??Y;return t>=_e&&t<=ve?e:`?`}function Z(e,t){if(t===void 0||t<0)return e;if(t===0)return``;let n=Array.from(e,X);return n.length<=t?n.join(``):t<=3?`.`.repeat(t):`${n.slice(0,t-3).join(``)}${xe}`}function Ce(e,t){return e.split(`
`).map(e=>Z(e,t))}function we(e,t,n={}){let r=Se(t),i=r.get(Y),a=t.atlas.width,o=t.atlas.height,s=n.advance??ye,c=n.lineHeight??t.metrics?.lineHeight??be,l=n.maxWidthEm===void 0?void 0:Math.max(0,Math.floor(n.maxWidthEm/s)),u=[],d=0;for(let t of Ce(e,l)){let e=0;for(let n of Array.from(t)){let t=X(n).codePointAt(0)??Y,c=r.get(t)??i;if(c?.planeBounds&&c.atlasBounds){let t=c.planeBounds,n=c.atlasBounds,r=n.left/a,i=1-n.top/o,s=(n.right-n.left)/a,l=1-n.bottom/o-i;u.push({x:e+t.left,y:d+t.bottom,w:t.right-t.left,h:t.top-t.bottom,u:r,v:i,uw:s,vh:l})}e+=s}d-=c}return u}async function Te(e){let t=`${y}/msdf/jetbrains-mono.json`,n=`${y}/msdf/jetbrains-mono.png`,r=await fetch(t);if(!r.ok)throw Error(`MSDF atlas JSON failed: ${r.status} ${t}`);let i=await r.json();if(i.atlas?.yOrigin!==`bottom`)throw Error(`MSDF atlas yOrigin must be bottom, got ${i.atlas?.yOrigin??`missing`}`);let a=await fetch(n);if(!a.ok)throw Error(`MSDF atlas PNG failed: ${a.status} ${n}`);let o=await a.blob(),s=await createImageBitmap(o),c=e.createTexture({label:`msdf-jetbrains-mono-rgba8unorm`,size:[s.width,s.height,1],format:`rgba8unorm`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});e.queue.copyExternalImageToTexture({source:s},{texture:c},{width:s.width,height:s.height}),s.close?.();let l=e.createSampler({label:`msdf-jetbrains-mono-linear-sampler`,magFilter:`linear`,minFilter:`linear`,mipmapFilter:`linear`,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`}),u=c.createView({label:`msdf-jetbrains-mono-view`}),d=new Map(i.glyphs.map(e=>[e.unicode,e]));return{...i,glyphMap:d,texture:c,textureView:u,sampler:l,dispose:()=>c.destroy()}}var Ee=`
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

struct Glyph {
	anchor_size: vec4f,
	quad_offset: vec4f,
	uv_rect: vec4f,
	info: vec4f,
	color: vec4f,
};

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) info: vec4f,
	@location(2) @interpolate(flat) color: vec4f,
	@location(3) @interpolate(flat) weight: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> glyphs: array<Glyph>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var atlas_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
	vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0)
);

fn median3(c: vec3f) -> f32 {
	return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

@vertex
fn vs_text(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let glyph = glyphs[ii];
	let corner = QUAD[vi];
	let anchor = glyph.anchor_size.xy;
	let size = glyph.anchor_size.zw;
	let quad_offset = glyph.quad_offset.xy;
	let uv_min = glyph.uv_rect.xy;
	let uv_max = glyph.uv_rect.zw;
	let aspect = max(0.0001, params.viewport_w / max(1.0, params.viewport_h));
	let depth = clamp(glyph.info.z, 0.0, 1.0);
	let cursor_pre = vec2f(params.cursor_x, params.cursor_y);
	let cursor_delta = cursor_pre - anchor;
	let d = distance(anchor, cursor_pre);
	// Wide influence radius so the field reacts when the cursor is anywhere NEAR
	// the text, not only dead-on (v1 R=0.45 was too tight to feel).
	let R = 0.75;
	let cursor_w = exp(-(d * d) / (R * R));
	// Per-glyph SCALE-UP near the cursor: glyphs the pointer approaches swell toward
	// you. Scaling the quad around its anchor is the most legible "alive" cue.
	let grow = 1.0 + cursor_w * 0.55;
	var pos = anchor + (quad_offset + corner * size) * grow;
	// Depth → clip z. Trust (depth~1) floats forward (small z), low-trust sinks back.
	// Cursor lifts a glyph forward, but z MUST stay > 0 or clip.z<0 clips the quad
	// behind the near plane and the glyph vanishes (v1 bug: cursor made text disappear).
	var z = mix(0.42, 0.10, depth);
	z = clamp(z - cursor_w * 0.42, 0.04, 0.6);
	let lean_dir = select(vec2f(0.0, 0.0), normalize(cursor_delta), length(cursor_delta) > 0.0001);
	pos = pos + lean_dir * cursor_w * 0.04;
	pos = pos + vec2f(sin(params.time * 0.6), cos(params.time * 0.5)) * ((1.0 - depth) * 0.006) * params.pulse;
	// Keep glyphs square in BOTH orientations: normalize by the longer axis.
	// Landscape (aspect>1): narrow x. Portrait (aspect<1): shrink y instead —
	// dividing x by aspect<1 would WIDEN x and push text off-screen.
	pos.x = pos.x / max(aspect, 1.0);
	pos.y = pos.y * min(aspect, 1.0);
	let wclip = 1.0 + z;
	var out: VSOut;
	out.clip = vec4f(pos, z, wclip);
	out.uv = vec2f(mix(uv_min.x, uv_max.x, corner.x), mix(uv_max.y, uv_min.y, corner.y));
	out.info = vec4f(glyph.info.x, glyph.info.y, cursor_w, depth);
	out.color = glyph.color;
	out.weight = clamp(glyph.info.w, 0.0, 1.0);
	return out;
}

@fragment
fn fs_text(in: VSOut) -> @location(0) vec4f {
	let atlas_px = vec2f(textureDimensions(atlas_tex, 0));
	let cursor_w = clamp(in.info.z, 0.0, 1.0);
	let depth = clamp(in.info.w, 0.0, 1.0);
	let weight = clamp(in.weight, 0.0, 1.0);
	var uv = in.uv;
	uv = uv + vec2f(sin(uv.y * 40.0 + params.time * 3.0), cos(uv.x * 40.0 + params.time * 3.0)) * (cursor_w * 0.007);
	let msdf = textureSample(atlas_tex, atlas_sampler, uv).rgb;
	let dist = median3(msdf);
	let uv_width = max(fwidth(uv), vec2f(1.0 / max(atlas_px.x, 1.0), 1.0 / max(atlas_px.y, 1.0)));
	let texels_per_px = max(length(uv_width * atlas_px), 0.0001);
	let screen_range = max(0.5, 4.0 / texels_per_px);
	// Depth-of-field: far/un-hovered glyphs soften, cursor sharpens. Kept GENTLE so
	// the resting field stays READABLE regardless of the data's depth value.
	let dof = (1.0 - depth) * (1.0 - cursor_w);
	let screen_range_dof = screen_range / (1.0 + dof * 0.6);
	// Weight (FSRS retention) modulates stroke mass WITHIN a readable band: it can
	// thicken a lot but only thin slightly, so a low-retention record never
	// disappears (data must be legible even at weight~0 — every route depends on this).
	let weight_bias = (weight - 0.5) * 0.10 + 0.03;
	let px_dist = screen_range_dof * (dist - 0.5 + weight_bias);
	let coverage = clamp(px_dist + 0.5, 0.0, 1.0);
	let reveal_span = max(1.0, in.info.y);
	let reveal = clamp((params.frame - in.info.x) / reveal_span, 0.0, 1.0);
	let alpha = coverage * in.color.a * reveal;
	if (alpha < 0.001) { discard; }
	// Glow floor keeps EVERY line clearly lit at rest (even depth~0), depth adds
	// forward-brightness, cursor pushes near glyphs HARD past the bloom line to flare.
	let glow = mix(1.15, 1.5, depth) + cursor_w * 1.4;
	let rgb = in.color.rgb * params.brightness * glow;
	return vec4f(rgb * alpha, alpha);
}
`,Q=20,De=[...R(`#22C7DE`),1],Oe=class{engine;atlas=null;bindLayout=null;pipeline=null;glyphBuffer=null;bindGroup=null;glyphCapacity=0;glyphCount=0;pendingItems=[];runs=[];runDepths=new Map;initPromise=null;onResize=null;resizeRaf=0;lastAspectBucket=-1;constructor(e){this.engine=e,this.installResizeReflow()}installResizeReflow(){typeof window>`u`||(this.onResize=()=>{this.resizeRaf||=requestAnimationFrame(()=>{this.resizeRaf=0;let e=this.aspectBucket();e!==this.lastAspectBucket&&this.pendingItems.length&&(this.lastAspectBucket=e,this.uploadItems(this.pendingItems))})},window.addEventListener(`resize`,this.onResize),window.addEventListener(`orientationchange`,this.onResize))}aspectBucket(){let e=this.engine.params[6]||0,t=this.engine.params[7]||0;return(e<=0||t<=0)&&typeof window<`u`&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0?-1:Math.round(e/t*8)}async init(){return this.initPromise||=this.initInner(),this.initPromise}async initInner(){let e=this.engine.gpuDevice;e&&this.engine.paramsBuffer&&(this.atlas=await Te(e),this.ensurePipeline(e),this.pendingItems.length&&this.uploadItems(this.pendingItems))}setText(e){let t=typeof e==`string`?[{text:e,x:-.62,y:0,size:.075}]:Array.isArray(e)?e:[e];this.pendingItems=t,this.uploadItems(t)}ensurePipeline(e){if(this.pipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`msdf-text-wgsl`,code:Ee});this.bindLayout=e.createBindGroupLayout({label:`msdf-text-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}}]});let n=e.createPipelineLayout({label:`msdf-text-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}};this.pipeline=e.createRenderPipeline({label:`msdf-text-pipeline`,layout:n,vertex:{module:t,entryPoint:`vs_text`},fragment:{module:t,entryPoint:`fs_text`,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`}})}portraitAdapt(e){let t=this.engine.params[6]||0,n=this.engine.params[7]||0;if((t<=0||n<=0)&&typeof window<`u`&&(t=window.innerWidth,n=window.innerHeight),t<=0||n<=0)return e;let r=t/n;if(r>=.85)return e;let i=1/Math.max(r,.2),a=i,o=1.25*i,s=.5*$((.85-r)/(.85-.46)),c=-.9,l=.62,u=e=>!!e&&(e.startsWith(`route-nav`)||e===`route-chrome`||e===`route-telemetry`||e===`route-status`||e===`route-status-pulse`);return e.map(e=>{if(u(e.kind)){let t=(e.size??.03)*Math.min(1.5,1.1*i);return{...e,size:t}}let t=e.size??.075,n=Math.max(1,e.text.length),r=.96-Math.max(c,e.x*(1-s)),d=r/(n*l),f=Math.max(t,Math.min(t*o,d)),p=Math.max(c,e.x*(1-s)),m=.92*i,h=e.y*a;h>m?h=m:h<-m&&(h=-m);let g=Math.floor(r/(f*l)),_=e.maxWidthEm==null?e.maxWidthEm:Math.max(14,Math.min(e.maxWidthEm,g)),v={...e,x:p,y:h,size:f,maxWidthEm:_};return typeof window<`u`&&window.location?.search.includes(`dbg=1`)&&(window.__adaptDbg??=[],window.__adaptDbg.push({id:e.id,text:e.text?.slice(0,24),x:+p.toFixed(3),y:+h.toFixed(3),size:+f.toFixed(4),maxWidthEm:_})),v})}uploadItems(e){let t=this.engine.gpuDevice;if(!t||!this.engine.paramsBuffer||!this.atlas)return;this.ensurePipeline(t);let n=this.portraitAdapt(e),r=[],i=[],a=0;n.forEach((e,t)=>{let n=e.size??.075,o=we(e.text,this.atlas,{maxWidthEm:e.maxWidthEm}),s=e.color??De,c=o,l=a,u=1/0,d=-1/0,f=1/0,p=-1/0;for(let t of c){let i=e.x+t.x*n,o=e.x+(t.x+t.w)*n,c=e.y+t.y*n,l=e.y+(t.y+t.h)*n;u=Math.min(u,i),d=Math.max(d,o),f=Math.min(f,c),p=Math.max(p,l),ke(r,e,t,s,n,a++)}if(c.length>0){let n=e.id??`msdf-text:${t}`;i.push({id:n,kind:e.kind??`text`,text:e.text,x0:u,x1:d,y0:f,y1:p,payload:e,glyphStart:l,glyphCount:a-l}),this.runDepths.set(n,$(e.depth??.5))}}),this.runs=i,this.glyphCount=r.length/Q;let o=new Float32Array(r.length||Q);o.set(r),this.ensureGlyphBuffer(t,Math.max(1,this.glyphCount)),this.glyphBuffer&&this.bindLayout&&(t.queue.writeBuffer(this.glyphBuffer,0,o),this.bindGroup=t.createBindGroup({label:`msdf-text-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.glyphBuffer}},{binding:2,resource:this.atlas.sampler},{binding:3,resource:this.atlas.textureView}]}))}ensureGlyphBuffer(e,t){this.glyphBuffer&&this.glyphCapacity>=t||(this.glyphBuffer?.destroy(),this.glyphCapacity=Math.max(t,Math.ceil(this.glyphCapacity*1.5),32),this.glyphBuffer=e.createBuffer({label:`msdf-text-glyphs`,size:this.glyphCapacity*Q*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}))}render(e){!this.pipeline||!this.bindGroup||this.glyphCount<=0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.glyphCount))}pickAt(e,t){let n=Math.max(1e-4,(this.engine.params[6]||1)/Math.max(1,this.engine.params[7]||1)),r=Math.max(n,1),i=Math.min(n,1);for(let n of this.runs){let a=(n.payload.hitPadX??n.payload.hitPad??0)/r,o=(n.payload.hitPadY??n.payload.hitPad??0)*i,s=n.x0/r-a,c=n.x1/r+a,l=n.y0*i-o,u=n.y1*i+o;if(e>=s&&e<=c&&t>=l&&t<=u)return{id:n.id,kind:n.kind,payload:n.payload}}return null}setRunDepth(e,t=.5){let n=this.engine.gpuDevice;if(n&&this.glyphBuffer)for(let r of this.runs){let i=$(r.id===e?t:r.payload.depth??.5);if(this.runDepths.get(r.id)===i)continue;this.runDepths.set(r.id,i);let a=new Float32Array([i]);for(let e=0;e<r.glyphCount;e+=1){let t=(r.glyphStart+e)*Q+14;n.queue.writeBuffer(this.glyphBuffer,t*4,a)}}}dispose(){this.onResize&&typeof window<`u`&&(window.removeEventListener(`resize`,this.onResize),window.removeEventListener(`orientationchange`,this.onResize)),this.onResize=null,this.resizeRaf&&cancelAnimationFrame(this.resizeRaf),this.resizeRaf=0,this.glyphBuffer?.destroy(),this.glyphBuffer=null,this.atlas?.dispose(),this.atlas=null,this.bindGroup=null,this.pipeline=null}};function ke(e,t,n,r,i,a){let o=(t.startFrame??0)+a*2,s=t.revealSpan??18;e.push(t.x,t.y,n.w*i,n.h*i,n.x*i,n.y*i,0,0,n.u,n.v,n.u+n.uw,n.v+n.vh,o,s,$(t.depth??.5),$(t.weight??.5),r[0],r[1],r[2],r[3])}function $(e){return Math.min(1,Math.max(0,Number.isFinite(e)?e:.5))}export{C,S,E as _,z as a,T as b,ge as c,pe as d,fe as f,D as g,k as h,H as i,G as l,w as m,W as n,B as o,ue as p,U as r,he as s,Oe as t,R as u,A as v,te as x,O as y};
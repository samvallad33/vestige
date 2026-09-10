import{$ as e,D as t,J as n,L as r,O as i,Q as a,U as o,X as s,Y as c,Z as l,a as u,c as d,et as f,j as p,k as m,lt as h,mt as g,n as _,r as v,ut as y}from"./Cmba2AGc.js";import"./xihTtKlq.js";function b(e){let t=1779033703^e.length;for(let n=0;n<e.length;n++)t=Math.imul(t^e.charCodeAt(n),2654435761),t=t<<13|t>>>19;return function(){let e=t+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}}function x(e){return function(){let t=e+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}}var S=class{fps;loopFrames;seedStr;_frame;_totalFrames;_rng;constructor(e){this.fps=e.fps??60,this.loopFrames=e.loopFrames??720,this.seedStr=e.seed,this._frame=0,this._totalFrames=0;let t=b(this.seedStr)();this._rng=x(Math.floor(t*2**32))}tick(){return this._frame=(this._frame+1)%this.loopFrames,this._totalFrames++,this.state}get state(){return{frame:this._frame,phase:this._frame/this.loopFrames,rng:this._rng,totalFrames:this._totalFrames}}reset(){this._frame=0,this._totalFrames=0;let e=b(this.seedStr)();this._rng=x(Math.floor(e*2**32))}get loopDuration(){return this.loopFrames/this.fps}get framesPerLoop(){return this.loopFrames}};function C(e,t,n,r){let i=Math.PI*(3-Math.sqrt(5)),a=1-e/(t-1||1)*2,o=Math.sqrt(1-a*a),s=i*e,c=Math.cos(s)*o,l=Math.sin(s)*o,u=(r()-.5)*.1*n,d=(r()-.5)*.1*n,f=(r()-.5)*.1*n;return[c*n+u,a*n+d,l*n+f]}var w=[`recall-path`,`engram-birth`,`salience-rescue`,`forgetting-horizon`,`firewall`];function T(e){return w.includes(e)}var E={posRadius:0,velRetention:4,colorFlags:8,demo:12},D={isCenter:1,suppressed:2,isAha:4,isFailure:8,isConfusion:16},O={recall:0,backwardCause:1,probe:2},k={none:0,firewall:1,dreamStorm:2,causalRecall:3,birth:4},A={frame:0,loopPhase:1,nodeCount:2,edgeCount:3,pathCount:4,pulse:5,viewportW:6,viewportH:7,brightness:8,demoId:9,time:10,captureMode:11,liveKind:12,liveFrame:13,liveEnergy:14,projectionDays:15,cursorX:16,cursorY:17,cursorVx:18,cursorVy:19};function j(e){let t=w.indexOf(e);return t<0?0:t}function M(e,t){return{id:e.id,index:t,label:e.label,type:e.type,retention:typeof e.retention==`number`?e.retention:0,tags:Array.isArray(e.tags)?e.tags:[],isCenter:!!e.isCenter,suppressed:(e.suppression_count??0)>0,stability:typeof e.stability==`number`?e.stability:void 0,lastAccessed:typeof e.lastAccessed==`string`?e.lastAccessed:void 0,createdAt:typeof e.createdAt==`string`?e.createdAt:void 0}}function N(e,t){let n=Math.max(1,e>>1),r=Math.max(1,t>>1),i=Math.min(6,Math.max(1,1+Math.floor(Math.log2(Math.min(n,r)/8))));return{baseW:n,baseH:r,mipCount:i,sizes:Array.from({length:i},(e,t)=>[Math.max(1,n>>t),Math.max(1,r>>t)])}}var P=.18,F=`
// Tuning constants — interpolated from post.wgsl.ts (TS single source of truth).
const BLOOM_STRENGTH: f32 = ${P};
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
`,I=`rgba16float`,L={color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one`,operation:`add`}},R=class{device;paramsBuffer;samp;blurLayout;compositeLayout;pipeDownFirst;pipeDown;pipeUp;pipeComposite;width=0;height=0;plan=null;sceneTex=null;_sceneView=null;bloomTex=null;mipViews=[];bloomFullView=null;downBind=[];upBind=[];compositeBind=null;constructor(e,t,n){this.device=e,this.paramsBuffer=t,this.samp=e.createSampler({label:`observatory-post-sampler`,minFilter:`linear`,magFilter:`linear`,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`});let r=e.createShaderModule({label:`observatory-post`,code:F});this.blurLayout=e.createBindGroupLayout({label:`observatory-post-blur-layout`,entries:[{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}}]}),this.compositeLayout=e.createBindGroupLayout({label:`observatory-post-composite-layout`,entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`,viewDimension:`2d`}}]});let i=e.createPipelineLayout({label:`observatory-post-blur-pipe-layout`,bindGroupLayouts:[this.blurLayout]}),a=e.createPipelineLayout({label:`observatory-post-composite-pipe-layout`,bindGroupLayouts:[this.compositeLayout]}),o=(t,n,i,a,o)=>e.createRenderPipeline({label:t,layout:n,vertex:{module:r,entryPoint:`vs_fullscreen`},fragment:{module:r,entryPoint:i,targets:[{format:a,blend:o}]},primitive:{topology:`triangle-list`}});this.pipeDownFirst=o(`observatory-post-down-karis`,i,`fs_downsample_karis`,I),this.pipeDown=o(`observatory-post-down`,i,`fs_downsample`,I),this.pipeUp=o(`observatory-post-up`,i,`fs_upsample_tent`,I,L),this.pipeComposite=o(`observatory-post-composite`,a,`fs_composite`,n)}get sceneView(){if(!this._sceneView)throw Error(`PostChain.ensure() must run before sceneView is used`);return this._sceneView}ensure(e,t){let n=Math.max(1,Math.floor(e)),r=Math.max(1,Math.floor(t));if(n===this.width&&r===this.height&&this.sceneTex!==null)return;this.width=n,this.height=r,this.sceneTex?.destroy(),this.bloomTex?.destroy(),this.sceneTex=this.device.createTexture({label:`observatory-scene-hdr`,size:[n,r],format:I,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this._sceneView=this.sceneTex.createView({label:`observatory-scene-hdr-view`});let i=N(n,r);this.plan=i,this.bloomTex=this.device.createTexture({label:`observatory-bloom-mips`,size:[i.baseW,i.baseH],format:I,mipLevelCount:i.mipCount,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});let a=this.bloomTex;this.mipViews=Array.from({length:i.mipCount},(e,t)=>a.createView({label:`observatory-bloom-mip-${t}`,baseMipLevel:t,mipLevelCount:1})),this.bloomFullView=a.createView({label:`observatory-bloom-full`});let o=this._sceneView;this.downBind=this.mipViews.map((e,t)=>this.device.createBindGroup({label:`observatory-bloom-down-bind-${t}`,layout:this.blurLayout,entries:[{binding:1,resource:t===0?o:this.mipViews[t-1]},{binding:2,resource:this.samp}]})),this.upBind=[];for(let e=0;e+1<i.mipCount;e++)this.upBind.push(this.device.createBindGroup({label:`observatory-bloom-up-bind-${e}`,layout:this.blurLayout,entries:[{binding:1,resource:this.mipViews[e+1]},{binding:2,resource:this.samp}]}));this.compositeBind=this.device.createBindGroup({label:`observatory-post-composite-bind`,layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:this.paramsBuffer}},{binding:2,resource:this.samp},{binding:3,resource:o},{binding:4,resource:this.bloomFullView}]})}encode(e,t){let n=this.plan;if(!n||!this.compositeBind)return;let r=n.mipCount;for(let t=0;t<r;t++){let n=e.beginRenderPass({label:`observatory-bloom-down-${t}`,colorAttachments:[{view:this.mipViews[t],loadOp:`clear`,storeOp:`store`}]});n.setPipeline(t===0?this.pipeDownFirst:this.pipeDown),n.setBindGroup(0,this.downBind[t]),n.draw(3),n.end()}for(let t=r-2;t>=0;t--){let n=e.beginRenderPass({label:`observatory-bloom-up-${t}`,colorAttachments:[{view:this.mipViews[t],loadOp:`load`,storeOp:`store`}]});n.setPipeline(this.pipeUp),n.setBindGroup(0,this.upBind[t]),n.draw(3),n.end()}let i=e.beginRenderPass({label:`observatory-post-composite`,colorAttachments:[{view:t,loadOp:`clear`,storeOp:`store`}]});i.setPipeline(this.pipeComposite),i.setBindGroup(0,this.compositeBind),i.draw(3),i.end()}dispose(){this.sceneTex?.destroy(),this.bloomTex?.destroy(),this.sceneTex=null,this.bloomTex=null,this._sceneView=null,this.bloomFullView=null,this.mipViews=[],this.downBind=[],this.upBind=[],this.compositeBind=null,this.plan=null,this.width=0,this.height=0}},z=Math.sqrt(5/255/6.25),B=z-5/255,V={r:z/(1+P),g:(6/255+B)/(1+P),b:(10/255+B)/(1+P),a:1},H=[500,1e3,2e3,4e3,8e3],U=class e{canvas;device=null;context=null;format=`bgra8unorm`;clock;demo;freezeFrame;rafId=0;running=!1;disposed=!1;recovering=!1;maxDpr;onFrame;lastRenderTs=-1/0;visibilityListenerAttached=!1;params=new Float32Array(20);paramsBuffer=null;passes=[];post=null;_status={state:`booting`};statusListeners=new Set;preFrameHook=null;lastRafTs=0;fpsEstimate=0;accumulatorMs=0;static FIXED_DT_MS=1e3/60;paused=!1;surge=0;surgeAt=0;kick(e=1){this.freezeFrame!==null||this.exportMode||(this.surge=Math.min(1.2,Math.max(this.surge,e)),this.surgeAt=performance.now(),this.requestRender())}constructor(e){this.canvas=e.canvas,this.demo=e.demo,this.maxDpr=e.maxDpr??2,this.onFrame=e.onFrame,this.clock=new S({seed:e.seed}),this.freezeFrame=typeof e.freezeFrame==`number`&&Number.isFinite(e.freezeFrame)?(Math.floor(e.freezeFrame)%this.clock.framesPerLoop+this.clock.framesPerLoop)%this.clock.framesPerLoop:null,this.params[8]=1,this.setCursorPreNdc(999,999,0,0)}get status(){return this._status}get gpuDevice(){return this.device}get presentationFormat(){return this.format}get sceneFormat(){return I}get demoClock(){return this.clock}onStatus(e){return this.statusListeners.add(e),e(this._status),()=>this.statusListeners.delete(e)}setStatus(e){this._status=e;for(let t of this.statusListeners)t(e)}addPass(e){this.passes.push(e)}removePass(e){let t=this.passes.indexOf(e);t!==-1&&(this.passes.splice(t,1),e.dispose?.())}clearPasses(){for(let e of this.passes)e.dispose?.();this.passes.length=0}setPreFrameHook(e){this.preFrameHook=e}get totalFrames(){return this.clock.state.totalFrames}setCursorPreNdc(e,t,n=0,r=0){this.params[A.cursorX]=Number.isFinite(e)?e:999,this.params[A.cursorY]=Number.isFinite(t)?t:999,this.params[A.cursorVx]=Number.isFinite(n)?n:0,this.params[A.cursorVy]=Number.isFinite(r)?r:0}setPaused(e){this.paused=e,this.requestRender()}get isPaused(){return this.paused}requestRender(){this.lastRenderTs=-1/0}get wallNowMs(){return Date.now()}async start(){if(this.disposed)return!1;let e=navigator.gpu;if(!e)return this.setStatus({state:`unsupported`,reason:`WebGPU is not available in this browser.`}),!1;let t=null;try{t=await e.requestAdapter()}catch(e){return this.setStatus({state:`error`,reason:e instanceof Error?e.message:`requestAdapter failed`}),!1}if(!t)return this.setStatus({state:`unsupported`,reason:`No suitable GPU adapter found.`}),!1;try{this.device=await t.requestDevice()}catch(e){return this.setStatus({state:`error`,reason:e instanceof Error?e.message:`requestDevice failed`}),!1}if(this.disposed)return this.device?.destroy(),this.device=null,!1;this.device.lost.then(e=>{this.disposed||e.reason===`destroyed`||(this.stopLoop(),this.recoverFromDeviceLoss(e.message))}),this.device.onuncapturederror=e=>{console.error(`[observatory] WebGPU error:`,e.error.message)};let n=this.canvas.getContext(`webgpu`);return n?(this.context=n,this.format=e.getPreferredCanvasFormat(),this.configureContext(),this.paramsBuffer=this.device.createBuffer({label:`observatory-params`,size:this.params.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.post=new R(this.device,this.paramsBuffer,this.format),this.setStatus({state:`running`}),this.attachVisibilityListener(),this.resumeLoop(),!0):(this.setStatus({state:`error`,reason:`Could not get webgpu canvas context.`}),!1)}async recoverFromDeviceLoss(e){if(!this.recovering){this.recovering=!0;try{for(let t=0;t<H.length;t++)if(this.disposed||(this.setStatus({state:`recovering`,attempt:t+1,reason:e}),await new Promise(e=>setTimeout(e,H[t])),this.disposed)||(this.releaseDeviceResources(),await this.start()))return;this.disposed||this.setStatus({state:`error`,reason:`GPU device lost: ${e}`})}finally{this.recovering=!1}}}releaseDeviceResources(){this.clearPasses(),this.post?.dispose(),this.post=null,this.paramsBuffer?.destroy(),this.paramsBuffer=null,this.context=null,this.device=null}resize(){if(!this.device||!this.context)return;let e=Math.min(window.devicePixelRatio||1,this.maxDpr),t=Math.max(1,Math.floor(this.canvas.clientWidth*e)),n=Math.max(1,Math.floor(this.canvas.clientHeight*e));(this.canvas.width!==t||this.canvas.height!==n)&&(this.canvas.width=t,this.canvas.height=n,this.configureContext(),this.post?.ensure(t,n))}configureContext(){this.device&&this.context&&this.context.configure({device:this.device,format:this.format,alphaMode:`opaque`})}attachVisibilityListener(){this.visibilityListenerAttached||typeof document>`u`||(document.addEventListener(`visibilitychange`,this.handleVisibilityChange),this.visibilityListenerAttached=!0)}handleVisibilityChange=()=>{if(!(typeof document>`u`)){if(document.hidden){this.stopLoop();return}this.resumeLoop()}};resumeLoop(){!this.running&&!this.disposed&&this.device&&this.context&&this.paramsBuffer&&this.post&&(this.running=!0,this.lastRafTs=0,this.accumulatorMs=0,this.requestRender(),this.rafId=requestAnimationFrame(this.frame))}frameRateFor(e){let t=0;for(let n of this.passes){let r=n.targetFrameRate?.(e);if(typeof r!=`number`||!Number.isFinite(r)){t=60;continue}t=Math.max(t,r)}return Math.max(1,Math.min(60,t||60))}frame=t=>{if(!this.running||!this.device||!this.context||!this.paramsBuffer||!this.post)return;let n=0;for(this.lastRafTs>0&&(n=t-this.lastRafTs),this.lastRafTs=t,this.accumulatorMs+=Math.min(n,250);this.accumulatorMs>=e.FIXED_DT_MS;)this.paused||this.clock.tick(),this.accumulatorMs-=e.FIXED_DT_MS;let r=this.clock.state,i=this.freezeFrame??r.frame,a=1e3/this.frameRateFor(i);if(t-this.lastRenderTs<a){this.rafId=requestAnimationFrame(this.frame);return}if(!this.encodeAndSubmit(i,r.totalFrames)){this.rafId=requestAnimationFrame(this.frame);return}Number.isFinite(this.lastRenderTs)&&t>this.lastRenderTs&&(this.fpsEstimate=Math.round(1e3/(t-this.lastRenderTs))),this.lastRenderTs=t,this.onFrame?.(i,this.fpsEstimate),this.rafId=requestAnimationFrame(this.frame)};encodeAndSubmit(e,t){if(!this.device||!this.context||!this.paramsBuffer||!this.post)return!1;let n=e/this.clock.framesPerLoop,r=this.params;if(r[0]=e,r[1]=n,r[5]=.5+.5*Math.sin(2*Math.PI*4*n),this.surge>0){r[5]=Math.min(1.6,r[5]+this.surge);let e=performance.now(),t=Math.min(.1,Math.max(0,(e-this.surgeAt)/1e3));this.surgeAt=e,this.surge=this.surge<.004?0:this.surge*.06**t}r[6]=this.canvas.width,r[7]=this.canvas.height,r[9]=j(this.demo),r[10]=this.freezeFrame!==null||this.exportMode?e/60:t/60,r[11]=this.freezeFrame===null?0:1,this.exportMode||this.preFrameHook?.(t),this.device.queue.writeBuffer(this.paramsBuffer,0,r);let i;try{i=this.context.getCurrentTexture()}catch{return!1}this.post.ensure(i.width,i.height);let a=i.createView(),o=this.device.createCommandEncoder({label:`observatory-frame`});for(let t of this.passes)t.compute?.(o,e);let s=o.beginRenderPass({label:`observatory-main`,colorAttachments:[{view:this.post.sceneView,clearValue:V,loadOp:`clear`,storeOp:`store`}]});for(let t of this.passes)t.render?.(s,e);return s.end(),this.post.encode(o,a),this.device.queue.submit([o.finish()]),!0}exportMode=!1;get canvasElement(){return this.canvas}beginExport(){this.stopLoop(),this.exportMode=!0,this.clock.reset()}endExport(){this.exportMode=!1,this.resumeLoop()}async renderExportFrame(e){if(!this.exportMode)throw Error(`renderExportFrame outside beginExport()`);if(!this.device)throw Error(`export: no GPU device`);e&&this.clock.tick();let t=this.clock.state,n=t.frame;if(!this.encodeAndSubmit(n,t.totalFrames))throw Error(`export: canvas has no texture (zero-sized or hidden)`);return await this.device.queue.onSubmittedWorkDone(),n}stopLoop(){this.running=!1,this.rafId!==0&&(cancelAnimationFrame(this.rafId),this.rafId=0)}dispose(){this.disposed||(this.disposed=!0,this.stopLoop(),this.visibilityListenerAttached&&typeof document<`u`&&(document.removeEventListener(`visibilitychange`,this.handleVisibilityChange),this.visibilityListenerAttached=!1),this.paramsBuffer?.destroy(),this.paramsBuffer=null,this.post?.dispose(),this.post=null,this.device?.destroy(),this.device=null,this.context=null,this.passes=[],this.setStatus({state:`disposed`}),this.statusListeners.clear())}},W=p(`<div id="webgpu-field-status" class="fallback svelte-16248mg" role="status" aria-live="polite"><div class="fallback-title svelte-16248mg">GPU DEVICE LOST · RECOVERING</div> <div class="fallback-reason svelte-16248mg"> </div></div>`),G=p(`<div id="webgpu-field-status" class="fallback svelte-16248mg" role="alert"><div class="fallback-title svelte-16248mg">3D MEMORY FIELD UNAVAILABLE</div> <div class="fallback-reason svelte-16248mg">This browser or device could not create a WebGPU graphics context, so this
			visual field has not rendered.</div> <div class="fallback-hint svelte-16248mg">Your local memories have not been changed. Use the persistent navigation to
			continue in another tool, or open this view in a WebGPU-capable browser.</div></div>`),K=p(`<canvas class="observatory-canvas svelte-16248mg" aria-label="Vestige 3D memory field" aria-describedby="webgpu-field-status"></canvas> <!>`,1);function q(p,b){y(b,!0);let x=u(b,`freezeFrame`,3,null),S=u(b,`maxDpr`,3,2),C,w=null,T=f(a({state:`booting`})),E=null,D=null;v(()=>{w=new U({canvas:C,demo:b.demo,seed:b.seed,freezeFrame:x(),maxDpr:S(),onFrame:(e,t)=>b.onframe?.(e,t)});let t=!1;E=w.onStatus(n=>{e(T,n,!0),n.state===`recovering`&&(t=!0),n.state===`running`&&t&&w&&(t=!1,w.resize(),b.onready?.(w))}),D=new ResizeObserver(()=>w?.resize()),D.observe(C),w.start().then(e=>{e&&w&&(w.resize(),b.onready?.(w))})}),_(()=>{E?.(),D?.disconnect(),w?.dispose(),w=null});var O=K(),k=c(O);d(k,e=>C=e,()=>C);var A=l(k,2),j=e=>{var t=W(),a=l(n(t),2),c=s(a);g(t),o(()=>i(c,`Re-acquiring the graphics device (attempt ${r(T).attempt??``} of 5). ${r(T).reason??``}`)),m(e,t)},M=e=>{var t=G();m(e,t)};t(A,e=>{r(T).state===`recovering`?e(j):(r(T).state===`unsupported`||r(T).state===`error`)&&e(M,1)}),m(p,O),h()}export{E as a,T as c,C as d,D as i,M as l,w as n,A as o,k as r,O as s,q as t,S as u};
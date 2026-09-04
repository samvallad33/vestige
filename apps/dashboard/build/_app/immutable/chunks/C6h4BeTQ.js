var G=Object.defineProperty;var B=(o,e,t)=>e in o?G(o,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):o[e]=t;var r=(o,e,t)=>B(o,typeof e!="symbol"?e+"":e,t);import"./Bzak7iHL.js";import{o as O,a as V,s as U}from"./CNdOtqLU.js";import{p as z,d as q,e as H,h as X,a as T,b as Y,f as S,s as W,j as E,g,c as M,r as A,t as K}from"./Dw_4PDAU.js";import{b as j,i as J}from"./C7WW_yYn.js";import{p as P}from"./N0ivUMqs.js";function R(o){let e=1779033703^o.length;for(let t=0;t<o.length;t++)e=Math.imul(e^o.charCodeAt(t),2654435761),e=e<<13|e>>>19;return function(){let t=e+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}}function D(o){return function(){let e=o+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}}class Z{constructor(e){r(this,"fps");r(this,"loopFrames");r(this,"seedStr");r(this,"_frame");r(this,"_totalFrames");r(this,"_rng");this.fps=e.fps??60,this.loopFrames=e.loopFrames??720,this.seedStr=e.seed,this._frame=0,this._totalFrames=0;const t=R(this.seedStr)();this._rng=D(Math.floor(t*2**32))}tick(){return this._frame=(this._frame+1)%this.loopFrames,this._totalFrames++,this.state}get state(){return{frame:this._frame,phase:this._frame/this.loopFrames,rng:this._rng,totalFrames:this._totalFrames}}reset(){this._frame=0,this._totalFrames=0;const e=R(this.seedStr)();this._rng=D(Math.floor(e*2**32))}get loopDuration(){return this.loopFrames/this.fps}get framesPerLoop(){return this.loopFrames}}function ge(o,e,t,i){const a=Math.PI*(3-Math.sqrt(5)),s=1-o/(e-1||1)*2,l=Math.sqrt(1-s*s),n=a*o,h=Math.cos(n)*l,d=Math.sin(n)*l,c=(i()-.5)*.1*t,m=(i()-.5)*.1*t,u=(i()-.5)*.1*t;return[h*t+c,s*t+m,d*t+u]}const I=["recall-path","engram-birth","salience-rescue","forgetting-horizon","firewall"];function xe(o){return I.includes(o)}const _e=16,ye={posRadius:0,velRetention:4,colorFlags:8},we={isCenter:1,suppressed:2,isAha:4,isFailure:8,isConfusion:16},Le=2,Te=4,Fe={recall:0,backwardCause:1,probe:2},Q=20,Se={none:0,firewall:1,dreamStorm:2,causalRecall:3,birth:4},y={liveKind:12,liveFrame:13,liveEnergy:14,projectionDays:15,cursorX:16,cursorY:17,cursorVx:18,cursorVy:19};function $(o){const e=I.indexOf(o);return e<0?0:e}function Ee(o,e){return{id:o.id,index:e,label:o.label,type:o.type,retention:typeof o.retention=="number"?o.retention:0,tags:Array.isArray(o.tags)?o.tags:[],isCenter:!!o.isCenter,suppressed:(o.suppression_count??0)>0,stability:typeof o.stability=="number"?o.stability:void 0,lastAccessed:typeof o.lastAccessed=="string"?o.lastAccessed:void 0,createdAt:typeof o.createdAt=="string"?o.createdAt:void 0}}function ee(o,e){const t=Math.max(1,o>>1),i=Math.max(1,e>>1),a=Math.min(6,Math.max(1,1+Math.floor(Math.log2(Math.min(t,i)/8)))),s=Array.from({length:a},(l,n)=>[Math.max(1,t>>n),Math.max(1,i>>n)]);return{baseW:t,baseH:i,mipCount:a,sizes:s}}const w=.18,te=0,se=2/255,re=.85,ie=.62,ae=`
// Tuning constants — interpolated from post.wgsl.ts (TS single source of truth).
const BLOOM_STRENGTH: f32 = ${w};
const BLOOM_CHROMATIC_TEXELS: f32 = ${te};
const GRAIN_AMP: f32 = ${se};
const VIGNETTE_LIFT: f32 = ${re};
const VIGNETTE_TAN: f32 = ${ie};

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
`,b="rgba16float",oe={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};class ne{constructor(e,t,i){r(this,"device");r(this,"paramsBuffer");r(this,"samp");r(this,"blurLayout");r(this,"compositeLayout");r(this,"pipeDownFirst");r(this,"pipeDown");r(this,"pipeUp");r(this,"pipeComposite");r(this,"width",0);r(this,"height",0);r(this,"plan",null);r(this,"sceneTex",null);r(this,"_sceneView",null);r(this,"bloomTex",null);r(this,"mipViews",[]);r(this,"bloomFullView",null);r(this,"downBind",[]);r(this,"upBind",[]);r(this,"compositeBind",null);this.device=e,this.paramsBuffer=t,this.samp=e.createSampler({label:"observatory-post-sampler",minFilter:"linear",magFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"});const a=e.createShaderModule({label:"observatory-post",code:ae});this.blurLayout=e.createBindGroupLayout({label:"observatory-post-blur-layout",entries:[{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}}]}),this.compositeLayout=e.createBindGroupLayout({label:"observatory-post-composite-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]});const s=e.createPipelineLayout({label:"observatory-post-blur-pipe-layout",bindGroupLayouts:[this.blurLayout]}),l=e.createPipelineLayout({label:"observatory-post-composite-pipe-layout",bindGroupLayouts:[this.compositeLayout]}),n=(h,d,c,m,u)=>e.createRenderPipeline({label:h,layout:d,vertex:{module:a,entryPoint:"vs_fullscreen"},fragment:{module:a,entryPoint:c,targets:[{format:m,blend:u}]},primitive:{topology:"triangle-list"}});this.pipeDownFirst=n("observatory-post-down-karis",s,"fs_downsample_karis",b),this.pipeDown=n("observatory-post-down",s,"fs_downsample",b),this.pipeUp=n("observatory-post-up",s,"fs_upsample_tent",b,oe),this.pipeComposite=n("observatory-post-composite",l,"fs_composite",i)}get sceneView(){if(!this._sceneView)throw new Error("PostChain.ensure() must run before sceneView is used");return this._sceneView}ensure(e,t){var h,d;const i=Math.max(1,Math.floor(e)),a=Math.max(1,Math.floor(t));if(i===this.width&&a===this.height&&this.sceneTex!==null)return;this.width=i,this.height=a,(h=this.sceneTex)==null||h.destroy(),(d=this.bloomTex)==null||d.destroy(),this.sceneTex=this.device.createTexture({label:"observatory-scene-hdr",size:[i,a],format:b,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this._sceneView=this.sceneTex.createView({label:"observatory-scene-hdr-view"});const s=ee(i,a);this.plan=s,this.bloomTex=this.device.createTexture({label:"observatory-bloom-mips",size:[s.baseW,s.baseH],format:b,mipLevelCount:s.mipCount,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});const l=this.bloomTex;this.mipViews=Array.from({length:s.mipCount},(c,m)=>l.createView({label:`observatory-bloom-mip-${m}`,baseMipLevel:m,mipLevelCount:1})),this.bloomFullView=l.createView({label:"observatory-bloom-full"});const n=this._sceneView;this.downBind=this.mipViews.map((c,m)=>this.device.createBindGroup({label:`observatory-bloom-down-bind-${m}`,layout:this.blurLayout,entries:[{binding:1,resource:m===0?n:this.mipViews[m-1]},{binding:2,resource:this.samp}]})),this.upBind=[];for(let c=0;c+1<s.mipCount;c++)this.upBind.push(this.device.createBindGroup({label:`observatory-bloom-up-bind-${c}`,layout:this.blurLayout,entries:[{binding:1,resource:this.mipViews[c+1]},{binding:2,resource:this.samp}]}));this.compositeBind=this.device.createBindGroup({label:"observatory-post-composite-bind",layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:this.paramsBuffer}},{binding:2,resource:this.samp},{binding:3,resource:n},{binding:4,resource:this.bloomFullView}]})}encode(e,t){const i=this.plan;if(!i||!this.compositeBind)return;const a=i.mipCount;for(let l=0;l<a;l++){const n=e.beginRenderPass({label:`observatory-bloom-down-${l}`,colorAttachments:[{view:this.mipViews[l],loadOp:"clear",storeOp:"store"}]});n.setPipeline(l===0?this.pipeDownFirst:this.pipeDown),n.setBindGroup(0,this.downBind[l]),n.draw(3),n.end()}for(let l=a-2;l>=0;l--){const n=e.beginRenderPass({label:`observatory-bloom-up-${l}`,colorAttachments:[{view:this.mipViews[l],loadOp:"load",storeOp:"store"}]});n.setPipeline(this.pipeUp),n.setBindGroup(0,this.upBind[l]),n.draw(3),n.end()}const s=e.beginRenderPass({label:"observatory-post-composite",colorAttachments:[{view:t,loadOp:"clear",storeOp:"store"}]});s.setPipeline(this.pipeComposite),s.setBindGroup(0,this.compositeBind),s.draw(3),s.end()}dispose(){var e,t;(e=this.sceneTex)==null||e.destroy(),(t=this.bloomTex)==null||t.destroy(),this.sceneTex=null,this.bloomTex=null,this._sceneView=null,this.bloomFullView=null,this.mipViews=[],this.downBind=[],this.upBind=[],this.compositeBind=null,this.plan=null,this.width=0,this.height=0}}const C=Math.sqrt(5/255/6.25),N=C-5/255,le={r:C/(1+w),g:(6/255+N)/(1+w),b:(10/255+N)/(1+w),a:1},k=[500,1e3,2e3,4e3,8e3],x=class x{constructor(e){r(this,"canvas");r(this,"device",null);r(this,"context",null);r(this,"format","bgra8unorm");r(this,"clock");r(this,"demo");r(this,"freezeFrame");r(this,"rafId",0);r(this,"running",!1);r(this,"disposed",!1);r(this,"recovering",!1);r(this,"maxDpr");r(this,"onFrame");r(this,"lastRenderTs",Number.NEGATIVE_INFINITY);r(this,"visibilityListenerAttached",!1);r(this,"params",new Float32Array(Q));r(this,"paramsBuffer",null);r(this,"passes",[]);r(this,"post",null);r(this,"_status",{state:"booting"});r(this,"statusListeners",new Set);r(this,"preFrameHook",null);r(this,"lastRafTs",0);r(this,"fpsEstimate",0);r(this,"accumulatorMs",0);r(this,"paused",!1);r(this,"surge",0);r(this,"surgeAt",0);r(this,"handleVisibilityChange",()=>{if(!(typeof document>"u")){if(document.hidden){this.stopLoop();return}this.resumeLoop()}});r(this,"frame",e=>{var n;if(!this.running||!this.device||!this.context||!this.paramsBuffer||!this.post)return;let t=0;for(this.lastRafTs>0&&(t=e-this.lastRafTs),this.lastRafTs=e,this.accumulatorMs+=Math.min(t,250);this.accumulatorMs>=x.FIXED_DT_MS;)this.paused||this.clock.tick(),this.accumulatorMs-=x.FIXED_DT_MS;const i=this.clock.state,a=this.freezeFrame??i.frame,l=1e3/this.frameRateFor(a);if(e-this.lastRenderTs<l){this.rafId=requestAnimationFrame(this.frame);return}if(!this.encodeAndSubmit(a,i.totalFrames)){this.rafId=requestAnimationFrame(this.frame);return}Number.isFinite(this.lastRenderTs)&&e>this.lastRenderTs&&(this.fpsEstimate=Math.round(1e3/(e-this.lastRenderTs))),this.lastRenderTs=e,(n=this.onFrame)==null||n.call(this,a,this.fpsEstimate),this.rafId=requestAnimationFrame(this.frame)});r(this,"exportMode",!1);this.canvas=e.canvas,this.demo=e.demo,this.maxDpr=e.maxDpr??2,this.onFrame=e.onFrame,this.clock=new Z({seed:e.seed}),this.freezeFrame=typeof e.freezeFrame=="number"&&Number.isFinite(e.freezeFrame)?(Math.floor(e.freezeFrame)%this.clock.framesPerLoop+this.clock.framesPerLoop)%this.clock.framesPerLoop:null,this.params[8]=1,this.setCursorPreNdc(999,999,0,0)}kick(e=1){this.freezeFrame!==null||this.exportMode||(this.surge=Math.min(1.2,Math.max(this.surge,e)),this.surgeAt=performance.now(),this.requestRender())}get status(){return this._status}get gpuDevice(){return this.device}get presentationFormat(){return this.format}get sceneFormat(){return b}get demoClock(){return this.clock}onStatus(e){return this.statusListeners.add(e),e(this._status),()=>this.statusListeners.delete(e)}setStatus(e){this._status=e;for(const t of this.statusListeners)t(e)}addPass(e){this.passes.push(e)}removePass(e){var i;const t=this.passes.indexOf(e);t!==-1&&(this.passes.splice(t,1),(i=e.dispose)==null||i.call(e))}clearPasses(){var e;for(const t of this.passes)(e=t.dispose)==null||e.call(t);this.passes.length=0}setPreFrameHook(e){this.preFrameHook=e}get totalFrames(){return this.clock.state.totalFrames}setCursorPreNdc(e,t,i=0,a=0){this.params[y.cursorX]=Number.isFinite(e)?e:999,this.params[y.cursorY]=Number.isFinite(t)?t:999,this.params[y.cursorVx]=Number.isFinite(i)?i:0,this.params[y.cursorVy]=Number.isFinite(a)?a:0}setPaused(e){this.paused=e,this.requestRender()}get isPaused(){return this.paused}requestRender(){this.lastRenderTs=Number.NEGATIVE_INFINITY}get wallNowMs(){return Date.now()}async start(){var a;if(this.disposed)return!1;const e=navigator.gpu;if(!e)return this.setStatus({state:"unsupported",reason:"WebGPU is not available in this browser."}),!1;let t=null;try{t=await e.requestAdapter()}catch(s){return this.setStatus({state:"error",reason:s instanceof Error?s.message:"requestAdapter failed"}),!1}if(!t)return this.setStatus({state:"unsupported",reason:"No suitable GPU adapter found."}),!1;try{this.device=await t.requestDevice()}catch(s){return this.setStatus({state:"error",reason:s instanceof Error?s.message:"requestDevice failed"}),!1}if(this.disposed)return(a=this.device)==null||a.destroy(),this.device=null,!1;this.device.lost.then(s=>{this.disposed||s.reason==="destroyed"||(this.stopLoop(),this.recoverFromDeviceLoss(s.message))}),this.device.onuncapturederror=s=>{console.error("[observatory] WebGPU error:",s.error.message)};const i=this.canvas.getContext("webgpu");return i?(this.context=i,this.format=e.getPreferredCanvasFormat(),this.configureContext(),this.paramsBuffer=this.device.createBuffer({label:"observatory-params",size:this.params.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.post=new ne(this.device,this.paramsBuffer,this.format),this.setStatus({state:"running"}),this.attachVisibilityListener(),this.resumeLoop(),!0):(this.setStatus({state:"error",reason:"Could not get webgpu canvas context."}),!1)}async recoverFromDeviceLoss(e){if(!this.recovering){this.recovering=!0;try{for(let t=0;t<k.length;t++)if(this.disposed||(this.setStatus({state:"recovering",attempt:t+1,reason:e}),await new Promise(i=>setTimeout(i,k[t])),this.disposed)||(this.releaseDeviceResources(),await this.start()))return;this.disposed||this.setStatus({state:"error",reason:`GPU device lost: ${e}`})}finally{this.recovering=!1}}}releaseDeviceResources(){var e,t;this.clearPasses(),(e=this.post)==null||e.dispose(),this.post=null,(t=this.paramsBuffer)==null||t.destroy(),this.paramsBuffer=null,this.context=null,this.device=null}resize(){var a;if(!this.device||!this.context)return;const e=Math.min(window.devicePixelRatio||1,this.maxDpr),t=Math.max(1,Math.floor(this.canvas.clientWidth*e)),i=Math.max(1,Math.floor(this.canvas.clientHeight*e));(this.canvas.width!==t||this.canvas.height!==i)&&(this.canvas.width=t,this.canvas.height=i,this.configureContext(),(a=this.post)==null||a.ensure(t,i))}configureContext(){!this.device||!this.context||this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"})}attachVisibilityListener(){this.visibilityListenerAttached||typeof document>"u"||(document.addEventListener("visibilitychange",this.handleVisibilityChange),this.visibilityListenerAttached=!0)}resumeLoop(){this.running||this.disposed||!this.device||!this.context||!this.paramsBuffer||!this.post||(this.running=!0,this.lastRafTs=0,this.accumulatorMs=0,this.requestRender(),this.rafId=requestAnimationFrame(this.frame))}frameRateFor(e){var i;let t=0;for(const a of this.passes){const s=(i=a.targetFrameRate)==null?void 0:i.call(a,e);if(typeof s!="number"||!Number.isFinite(s)){t=60;continue}t=Math.max(t,s)}return Math.max(1,Math.min(60,t||60))}encodeAndSubmit(e,t){var d,c,m;if(!this.device||!this.context||!this.paramsBuffer||!this.post)return!1;const i=e/this.clock.framesPerLoop,a=this.params;if(a[0]=e,a[1]=i,a[5]=.5+.5*Math.sin(2*Math.PI*4*i),this.surge>0){a[5]=Math.min(1.6,a[5]+this.surge);const u=performance.now(),L=Math.min(.1,Math.max(0,(u-this.surgeAt)/1e3));this.surgeAt=u,this.surge=this.surge<.004?0:this.surge*Math.pow(.06,L)}a[6]=this.canvas.width,a[7]=this.canvas.height,a[9]=$(this.demo),a[10]=this.freezeFrame!==null||this.exportMode?e/60:t/60,a[11]=this.freezeFrame!==null?1:0,this.exportMode||(d=this.preFrameHook)==null||d.call(this,t),this.device.queue.writeBuffer(this.paramsBuffer,0,a);let s;try{s=this.context.getCurrentTexture()}catch{return!1}this.post.ensure(s.width,s.height);const l=s.createView(),n=this.device.createCommandEncoder({label:"observatory-frame"});for(const u of this.passes)(c=u.compute)==null||c.call(u,n,e);const h=n.beginRenderPass({label:"observatory-main",colorAttachments:[{view:this.post.sceneView,clearValue:le,loadOp:"clear",storeOp:"store"}]});for(const u of this.passes)(m=u.render)==null||m.call(u,h,e);return h.end(),this.post.encode(n,l),this.device.queue.submit([n.finish()]),!0}get canvasElement(){return this.canvas}beginExport(){this.stopLoop(),this.exportMode=!0,this.clock.reset()}endExport(){this.exportMode=!1,this.resumeLoop()}async renderExportFrame(e){if(!this.exportMode)throw new Error("renderExportFrame outside beginExport()");if(!this.device)throw new Error("export: no GPU device");e&&this.clock.tick();const t=this.clock.state,i=t.frame;if(!this.encodeAndSubmit(i,t.totalFrames))throw new Error("export: canvas has no texture (zero-sized or hidden)");return await this.device.queue.onSubmittedWorkDone(),i}stopLoop(){this.running=!1,this.rafId!==0&&(cancelAnimationFrame(this.rafId),this.rafId=0)}dispose(){var e,t,i;this.disposed||(this.disposed=!0,this.stopLoop(),this.visibilityListenerAttached&&typeof document<"u"&&(document.removeEventListener("visibilitychange",this.handleVisibilityChange),this.visibilityListenerAttached=!1),(e=this.paramsBuffer)==null||e.destroy(),this.paramsBuffer=null,(t=this.post)==null||t.dispose(),this.post=null,(i=this.device)==null||i.destroy(),this.device=null,this.context=null,this.passes=[],this.setStatus({state:"disposed"}),this.statusListeners.clear())}};r(x,"FIXED_DT_MS",1e3/60);let F=x;var ce=S('<div id="webgpu-field-status" class="fallback svelte-16248mg" role="status" aria-live="polite"><div class="fallback-title svelte-16248mg">GPU DEVICE LOST · RECOVERING</div> <div class="fallback-reason svelte-16248mg"> </div></div>'),ue=S(`<div id="webgpu-field-status" class="fallback svelte-16248mg" role="alert"><div class="fallback-title svelte-16248mg">3D MEMORY FIELD UNAVAILABLE</div> <div class="fallback-reason svelte-16248mg">This browser or device could not create a WebGPU graphics context, so this
			visual field has not rendered.</div> <div class="fallback-hint svelte-16248mg">Your local memories have not been changed. Use the persistent navigation to
			continue in another tool, or open this view in a WebGPU-capable browser.</div></div>`),he=S('<canvas class="observatory-canvas svelte-16248mg" aria-label="Vestige 3D memory field" aria-describedby="webgpu-field-status"></canvas> <!>',1);function Me(o,e){z(e,!0);let t=P(e,"freezeFrame",3,null),i=P(e,"maxDpr",3,2),a,s=null,l=q(H({state:"booting"})),n=null,h=null;O(()=>{s=new F({canvas:a,demo:e.demo,seed:e.seed,freezeFrame:t(),maxDpr:i(),onFrame:(p,v)=>{var _;return(_=e.onframe)==null?void 0:_.call(e,p,v)}});let f=!1;n=s.onStatus(p=>{var v;W(l,p,!0),p.state==="recovering"&&(f=!0),p.state==="running"&&f&&s&&(f=!1,s.resize(),(v=e.onready)==null||v.call(e,s))}),h=new ResizeObserver(()=>s==null?void 0:s.resize()),h.observe(a),s.start().then(p=>{var v;p&&s&&(s.resize(),(v=e.onready)==null||v.call(e,s))})}),V(()=>{n==null||n(),h==null||h.disconnect(),s==null||s.dispose(),s=null});var d=he(),c=X(d);j(c,f=>a=f,()=>a);var m=E(c,2);{var u=f=>{var p=ce(),v=E(M(p),2),_=M(v);A(v),A(p),K(()=>U(_,`Re-acquiring the graphics device (attempt ${g(l).attempt??""} of 5). ${g(l).reason??""}`)),T(f,p)},L=f=>{var p=ue();T(f,p)};J(m,f=>{g(l).state==="recovering"?f(u):(g(l).state==="unsupported"||g(l).state==="error")&&f(L,1)})}T(o,d),Y()}export{Z as D,_e as F,Se as L,ye as N,Me as O,Fe as P,Le as U,we as a,Te as b,y as c,ge as d,I as e,xe as i,Ee as t};

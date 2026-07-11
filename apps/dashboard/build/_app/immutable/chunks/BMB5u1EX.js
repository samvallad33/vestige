var k=Object.defineProperty;var G=(a,e,t)=>e in a?k(a,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):a[e]=t;var s=(a,e,t)=>G(a,typeof e!="symbol"?e+"":e,t);import"./Bzak7iHL.js";import{o as V,a as U,s as z}from"./DAau0uzT.js";import{p as H,d as q,e as X,aH as W,a as T,b as Y,f as B,s as K,X as E,g as S,c as P,r as A,af as j,Y as J}from"./CGq8RnJq.js";import{b as Z,i as Q}from"./Ccqjq5DS.js";import{p as $}from"./DV6OI5iy.js";function D(a){let e=1779033703^a.length;for(let t=0;t<a.length;t++)e=Math.imul(e^a.charCodeAt(t),2654435761),e=e<<13|e>>>19;return function(){let t=e+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}}function R(a){return function(){let e=a+=1831565813;return e=Math.imul(e^e>>>15,e|1),e^=Math.imul(e^e>>>7,e|61),((e^e>>>14)>>>0)/4294967296}}class ee{constructor(e){s(this,"fps");s(this,"loopFrames");s(this,"seedStr");s(this,"_frame");s(this,"_totalFrames");s(this,"_rng");this.fps=e.fps??60,this.loopFrames=e.loopFrames??720,this.seedStr=e.seed,this._frame=0,this._totalFrames=0;const t=D(this.seedStr)();this._rng=R(Math.floor(t*2**32))}tick(){return this._frame=(this._frame+1)%this.loopFrames,this._totalFrames++,this.state}get state(){return{frame:this._frame,phase:this._frame/this.loopFrames,rng:this._rng,totalFrames:this._totalFrames}}reset(){this._frame=0,this._totalFrames=0;const e=D(this.seedStr)();this._rng=R(Math.floor(e*2**32))}get loopDuration(){return this.loopFrames/this.fps}get framesPerLoop(){return this.loopFrames}}function Fe(a,e,t,i){const r=Math.PI*(3-Math.sqrt(5)),o=1-a/(e-1||1)*2,n=Math.sqrt(1-o*o),l=r*a,h=Math.cos(l)*n,m=Math.sin(l)*n,u=(i()-.5)*.1*t,p=(i()-.5)*.1*t,c=(i()-.5)*.1*t;return[h*t+u,o*t+p,m*t+c]}const C=["recall-path","engram-birth","salience-rescue","forgetting-horizon","firewall"];function Se(a){return C.includes(a)}const Le=16,Me={posRadius:0,velRetention:4,colorFlags:8},Te={isCenter:1,suppressed:2,isAha:4,isFailure:8,isConfusion:16},Ee=2,Pe=4,Ae={recall:0,backwardCause:1,probe:2},te=20,De={none:0,firewall:1,dreamStorm:2,causalRecall:3,birth:4},y={liveKind:12,liveFrame:13,liveEnergy:14,projectionDays:15,cursorX:16,cursorY:17,cursorVx:18,cursorVy:19};function se(a){const e=C.indexOf(a);return e<0?0:e}function Re(a,e){return{id:a.id,index:e,label:a.label,type:a.type,retention:typeof a.retention=="number"?a.retention:0,tags:Array.isArray(a.tags)?a.tags:[],isCenter:!!a.isCenter,suppressed:(a.suppression_count??0)>0,stability:typeof a.stability=="number"?a.stability:void 0,lastAccessed:typeof a.lastAccessed=="string"?a.lastAccessed:void 0}}function re(a,e){const t=Math.max(1,a>>1),i=Math.max(1,e>>1),r=Math.min(6,Math.max(1,1+Math.floor(Math.log2(Math.min(t,i)/8)))),o=Array.from({length:r},(n,l)=>[Math.max(1,t>>l),Math.max(1,i>>l)]);return{baseW:t,baseH:i,mipCount:r,sizes:o}}const F=.18,ae=0,ie=2/255,oe=.85,ne=.62,le=`
// Tuning constants — interpolated from post.wgsl.ts (TS single source of truth).
const BLOOM_STRENGTH: f32 = ${F};
const BLOOM_CHROMATIC_TEXELS: f32 = ${ae};
const GRAIN_AMP: f32 = ${ie};
const VIGNETTE_LIFT: f32 = ${oe};
const VIGNETTE_TAN: f32 = ${ne};

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
`,x="rgba16float",ce={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};class ue{constructor(e,t,i){s(this,"device");s(this,"paramsBuffer");s(this,"samp");s(this,"blurLayout");s(this,"compositeLayout");s(this,"pipeDownFirst");s(this,"pipeDown");s(this,"pipeUp");s(this,"pipeComposite");s(this,"width",0);s(this,"height",0);s(this,"plan",null);s(this,"sceneTex",null);s(this,"_sceneView",null);s(this,"bloomTex",null);s(this,"mipViews",[]);s(this,"bloomFullView",null);s(this,"downBind",[]);s(this,"upBind",[]);s(this,"compositeBind",null);this.device=e,this.paramsBuffer=t,this.samp=e.createSampler({label:"observatory-post-sampler",minFilter:"linear",magFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"});const r=e.createShaderModule({label:"observatory-post",code:le});this.blurLayout=e.createBindGroupLayout({label:"observatory-post-blur-layout",entries:[{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}}]}),this.compositeLayout=e.createBindGroupLayout({label:"observatory-post-composite-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]});const o=e.createPipelineLayout({label:"observatory-post-blur-pipe-layout",bindGroupLayouts:[this.blurLayout]}),n=e.createPipelineLayout({label:"observatory-post-composite-pipe-layout",bindGroupLayouts:[this.compositeLayout]}),l=(h,m,u,p,c)=>e.createRenderPipeline({label:h,layout:m,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:u,targets:[{format:p,blend:c}]},primitive:{topology:"triangle-list"}});this.pipeDownFirst=l("observatory-post-down-karis",o,"fs_downsample_karis",x),this.pipeDown=l("observatory-post-down",o,"fs_downsample",x),this.pipeUp=l("observatory-post-up",o,"fs_upsample_tent",x,ce),this.pipeComposite=l("observatory-post-composite",n,"fs_composite",i)}get sceneView(){if(!this._sceneView)throw new Error("PostChain.ensure() must run before sceneView is used");return this._sceneView}ensure(e,t){var h,m;const i=Math.max(1,Math.floor(e)),r=Math.max(1,Math.floor(t));if(i===this.width&&r===this.height&&this.sceneTex!==null)return;this.width=i,this.height=r,(h=this.sceneTex)==null||h.destroy(),(m=this.bloomTex)==null||m.destroy(),this.sceneTex=this.device.createTexture({label:"observatory-scene-hdr",size:[i,r],format:x,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING}),this._sceneView=this.sceneTex.createView({label:"observatory-scene-hdr-view"});const o=re(i,r);this.plan=o,this.bloomTex=this.device.createTexture({label:"observatory-bloom-mips",size:[o.baseW,o.baseH],format:x,mipLevelCount:o.mipCount,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});const n=this.bloomTex;this.mipViews=Array.from({length:o.mipCount},(u,p)=>n.createView({label:`observatory-bloom-mip-${p}`,baseMipLevel:p,mipLevelCount:1})),this.bloomFullView=n.createView({label:"observatory-bloom-full"});const l=this._sceneView;this.downBind=this.mipViews.map((u,p)=>this.device.createBindGroup({label:`observatory-bloom-down-bind-${p}`,layout:this.blurLayout,entries:[{binding:1,resource:p===0?l:this.mipViews[p-1]},{binding:2,resource:this.samp}]})),this.upBind=[];for(let u=0;u+1<o.mipCount;u++)this.upBind.push(this.device.createBindGroup({label:`observatory-bloom-up-bind-${u}`,layout:this.blurLayout,entries:[{binding:1,resource:this.mipViews[u+1]},{binding:2,resource:this.samp}]}));this.compositeBind=this.device.createBindGroup({label:"observatory-post-composite-bind",layout:this.compositeLayout,entries:[{binding:0,resource:{buffer:this.paramsBuffer}},{binding:2,resource:this.samp},{binding:3,resource:l},{binding:4,resource:this.bloomFullView}]})}encode(e,t){const i=this.plan;if(!i||!this.compositeBind)return;const r=i.mipCount;for(let n=0;n<r;n++){const l=e.beginRenderPass({label:`observatory-bloom-down-${n}`,colorAttachments:[{view:this.mipViews[n],loadOp:"clear",storeOp:"store"}]});l.setPipeline(n===0?this.pipeDownFirst:this.pipeDown),l.setBindGroup(0,this.downBind[n]),l.draw(3),l.end()}for(let n=r-2;n>=0;n--){const l=e.beginRenderPass({label:`observatory-bloom-up-${n}`,colorAttachments:[{view:this.mipViews[n],loadOp:"load",storeOp:"store"}]});l.setPipeline(this.pipeUp),l.setBindGroup(0,this.upBind[n]),l.draw(3),l.end()}const o=e.beginRenderPass({label:"observatory-post-composite",colorAttachments:[{view:t,loadOp:"clear",storeOp:"store"}]});o.setPipeline(this.pipeComposite),o.setBindGroup(0,this.compositeBind),o.draw(3),o.end()}dispose(){var e,t;(e=this.sceneTex)==null||e.destroy(),(t=this.bloomTex)==null||t.destroy(),this.sceneTex=null,this.bloomTex=null,this._sceneView=null,this.bloomFullView=null,this.mipViews=[],this.downBind=[],this.upBind=[],this.compositeBind=null,this.plan=null,this.width=0,this.height=0}}const I=Math.sqrt(5/255/6.25),N=I-5/255,pe={r:I/(1+F),g:(6/255+N)/(1+F),b:(10/255+N)/(1+F),a:1},w=class w{constructor(e){s(this,"canvas");s(this,"device",null);s(this,"context",null);s(this,"format","bgra8unorm");s(this,"clock");s(this,"demo");s(this,"freezeFrame");s(this,"rafId",0);s(this,"running",!1);s(this,"disposed",!1);s(this,"maxDpr");s(this,"onFrame");s(this,"params",new Float32Array(te));s(this,"paramsBuffer",null);s(this,"passes",[]);s(this,"post",null);s(this,"_status",{state:"booting"});s(this,"statusListeners",new Set);s(this,"preFrameHook",null);s(this,"lastRafTs",0);s(this,"fpsEstimate",0);s(this,"accumulatorMs",0);s(this,"paused",!1);s(this,"frame",e=>{var p,c,d,b;if(!this.running||!this.device||!this.context||!this.paramsBuffer||!this.post)return;let t=0;for(this.lastRafTs>0&&(t=e-this.lastRafTs,t>0&&(this.fpsEstimate=Math.round(1e3/t))),this.lastRafTs=e,this.accumulatorMs+=Math.min(t,250);this.accumulatorMs>=w.FIXED_DT_MS;)this.paused||this.clock.tick(),this.accumulatorMs-=w.FIXED_DT_MS;const i=this.clock.state,r=this.freezeFrame??i.frame,o=r/this.clock.framesPerLoop,n=this.params;n[0]=r,n[1]=o,n[5]=.5+.5*Math.sin(2*Math.PI*4*o),n[6]=this.canvas.width,n[7]=this.canvas.height,n[9]=se(this.demo),n[10]=r/60,n[11]=this.freezeFrame!==null?1:0,(p=this.preFrameHook)==null||p.call(this,i.totalFrames),this.device.queue.writeBuffer(this.paramsBuffer,0,n);let l;try{l=this.context.getCurrentTexture()}catch{this.rafId=requestAnimationFrame(this.frame);return}this.post.ensure(l.width,l.height);const h=l.createView(),m=this.device.createCommandEncoder({label:"observatory-frame"});for(const v of this.passes)(c=v.compute)==null||c.call(v,m,r);const u=m.beginRenderPass({label:"observatory-main",colorAttachments:[{view:this.post.sceneView,clearValue:pe,loadOp:"clear",storeOp:"store"}]});for(const v of this.passes)(d=v.render)==null||d.call(v,u,r);u.end(),this.post.encode(m,h),this.device.queue.submit([m.finish()]),(b=this.onFrame)==null||b.call(this,r,this.fpsEstimate),this.rafId=requestAnimationFrame(this.frame)});this.canvas=e.canvas,this.demo=e.demo,this.maxDpr=e.maxDpr??2,this.onFrame=e.onFrame,this.clock=new ee({seed:e.seed}),this.freezeFrame=typeof e.freezeFrame=="number"&&Number.isFinite(e.freezeFrame)?(Math.floor(e.freezeFrame)%this.clock.framesPerLoop+this.clock.framesPerLoop)%this.clock.framesPerLoop:null,this.params[8]=1,this.setCursorPreNdc(999,999,0,0)}get status(){return this._status}get gpuDevice(){return this.device}get presentationFormat(){return this.format}get sceneFormat(){return x}get demoClock(){return this.clock}onStatus(e){return this.statusListeners.add(e),e(this._status),()=>this.statusListeners.delete(e)}setStatus(e){this._status=e;for(const t of this.statusListeners)t(e)}addPass(e){this.passes.push(e)}removePass(e){var i;const t=this.passes.indexOf(e);t!==-1&&(this.passes.splice(t,1),(i=e.dispose)==null||i.call(e))}clearPasses(){var e;for(const t of this.passes)(e=t.dispose)==null||e.call(t);this.passes.length=0}setPreFrameHook(e){this.preFrameHook=e}get totalFrames(){return this.clock.state.totalFrames}setCursorPreNdc(e,t,i=0,r=0){this.params[y.cursorX]=Number.isFinite(e)?e:999,this.params[y.cursorY]=Number.isFinite(t)?t:999,this.params[y.cursorVx]=Number.isFinite(i)?i:0,this.params[y.cursorVy]=Number.isFinite(r)?r:0}setPaused(e){this.paused=e}get isPaused(){return this.paused}get wallNowMs(){return Date.now()}async start(){var r;if(this.disposed)return!1;const e=navigator.gpu;if(!e)return this.setStatus({state:"unsupported",reason:"WebGPU is not available in this browser."}),!1;let t=null;try{t=await e.requestAdapter()}catch(o){return this.setStatus({state:"error",reason:o instanceof Error?o.message:"requestAdapter failed"}),!1}if(!t)return this.setStatus({state:"unsupported",reason:"No suitable GPU adapter found."}),!1;try{this.device=await t.requestDevice()}catch(o){return this.setStatus({state:"error",reason:o instanceof Error?o.message:"requestDevice failed"}),!1}if(this.disposed)return(r=this.device)==null||r.destroy(),this.device=null,!1;this.device.lost.then(o=>{this.disposed||o.reason==="destroyed"||(this.setStatus({state:"error",reason:`GPU device lost: ${o.message}`}),this.stopLoop())}),this.device.onuncapturederror=o=>{console.error("[observatory] WebGPU error:",o.error.message)};const i=this.canvas.getContext("webgpu");return i?(this.context=i,this.format=e.getPreferredCanvasFormat(),this.configureContext(),this.paramsBuffer=this.device.createBuffer({label:"observatory-params",size:this.params.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.post=new ue(this.device,this.paramsBuffer,this.format),this.setStatus({state:"running"}),this.running=!0,this.rafId=requestAnimationFrame(this.frame),!0):(this.setStatus({state:"error",reason:"Could not get webgpu canvas context."}),!1)}resize(){var r;if(!this.device||!this.context)return;const e=Math.min(window.devicePixelRatio||1,this.maxDpr),t=Math.max(1,Math.floor(this.canvas.clientWidth*e)),i=Math.max(1,Math.floor(this.canvas.clientHeight*e));(this.canvas.width!==t||this.canvas.height!==i)&&(this.canvas.width=t,this.canvas.height=i,this.configureContext(),(r=this.post)==null||r.ensure(t,i))}configureContext(){!this.device||!this.context||this.context.configure({device:this.device,format:this.format,alphaMode:"opaque"})}stopLoop(){this.running=!1,this.rafId!==0&&(cancelAnimationFrame(this.rafId),this.rafId=0)}dispose(){var e,t,i;this.disposed||(this.disposed=!0,this.stopLoop(),(e=this.paramsBuffer)==null||e.destroy(),this.paramsBuffer=null,(t=this.post)==null||t.dispose(),this.post=null,(i=this.device)==null||i.destroy(),this.device=null,this.context=null,this.passes=[],this.setStatus({state:"disposed"}),this.statusListeners.clear())}};s(w,"FIXED_DT_MS",1e3/60);let M=w;var me=B(`<div class="fallback svelte-16248mg" role="alert"><div class="fallback-title svelte-16248mg">MEMORY FIELD OFFLINE</div> <div class="fallback-reason svelte-16248mg"> </div> <div class="fallback-hint svelte-16248mg">WebGPU is required for the Observatory. Chrome 113+, Edge 113+, or Safari 18+ —
			the classic <a href="./graph" class="svelte-16248mg">Graph view</a> works everywhere.</div></div>`),he=B('<canvas class="observatory-canvas svelte-16248mg" aria-label="Vestige memory field"></canvas> <!>',1);function Ne(a,e){H(e,!0);let t=$(e,"freezeFrame",3,null),i,r=null,o=q(X({state:"booting"})),n=null,l=null;V(()=>{r=new M({canvas:i,demo:e.demo,seed:e.seed,freezeFrame:t(),maxDpr:2,onFrame:(c,d)=>{var b;return(b=e.onframe)==null?void 0:b.call(e,c,d)}}),n=r.onStatus(c=>K(o,c,!0)),l=new ResizeObserver(()=>r==null?void 0:r.resize()),l.observe(i),r.start().then(c=>{var d;c&&r&&(r.resize(),(d=e.onready)==null||d.call(e,r))})}),U(()=>{n==null||n(),l==null||l.disconnect(),r==null||r.dispose(),r=null});var h=he(),m=W(h);Z(m,c=>i=c,()=>i);var u=E(m,2);{var p=c=>{var d=me(),b=E(P(d),2),v=P(b,!0);A(b),j(2),A(d),J(()=>z(v,S(o).reason)),T(c,d)};Q(u,c=>{(S(o).state==="unsupported"||S(o).state==="error")&&c(p)})}T(a,h),Y()}function O(a){const e=/^#?([0-9a-fA-F]{6})$/.exec(a.trim());if(!e)return[2/255,3/255,7/255];const t=parseInt(e[1],16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}const de={blackwater:"#020307"},f={luciferin:"#E9FFB7",healthy:"#A8FF5E",recall:"#29F2A9",bridge:"#1BD6FF",debt:"#8A4B18",extinction:"#2A160B"},fe=[f.extinction,f.debt,f.healthy,f.luciferin],g={trustMembrane:"#F4F1D0",caution:"#FFD166",veto:"#FF3B30",suppressionScar:"#B90D2B",labile:"#FF7A1A"},_={forward:"#00F5D4",retrograde:"#FF2DF7"},L={validRing:"#6BFFB8",txShadow:"#7C6CFF",supersession:"#FFB000"},Be={throughput:"#9DFFEB"};function Ce(a){const e=Math.max(0,Math.min(1,a)),t=fe.map(O),i=e*(t.length-1),r=Math.min(t.length-2,Math.floor(i)),o=i-r,n=t[r],l=t[r+1];return[n[0]+(l[0]-n[0])*o,n[1]+(l[1]-n[1])*o,n[2]+(l[2]-n[2])*o]}const ve={MemoryCreated:f.healthy,SearchPerformed:_.forward,ActivationSpread:f.recall,ImportanceScored:L.supersession,RetentionDecayed:f.debt,ConnectionDiscovered:_.forward,DeepReferenceCompleted:f.luciferin,BackfillFired:_.retrograde,CausalReceipt:_.retrograde,MemorySuppressed:g.suppressionScar,MemoryUnsuppressed:g.labile,MemoryPromoted:f.luciferin,MemoryDemoted:f.debt,MemoryPrOpened:g.caution,MemoryPrDecided:g.trustMembrane,HookVerdictRecorded:g.veto,TraceEvent:_.forward,DreamStarted:L.txShadow,DreamCompleted:L.validRing,Rac1CascadeSwept:g.suppressionScar};function Ie(a){return O(ve[a]??de.blackwater)}function Oe(a){const e=Math.max(0,Math.min(1,a));return .003+(.018-.003)*e}export{_ as C,C as D,Le as F,g as I,De as L,de as M,Me as N,Ne as O,Ae as P,f as R,Ee as U,Be as V,Ce as a,Te as b,Pe as c,Fe as d,Ie as e,ee as f,y as g,Se as i,Oe as m,O as r,Re as t};

import{s as e}from"./DUBTf18l.js";function t(e){let t=/^#?([0-9a-fA-F]{6})$/.exec(e.trim());if(!t)return[2/255,3/255,7/255];let n=parseInt(t[1],16);return[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}var n={blackwater:`#020307`,anaerobic:`#07100D`,cyanFog:`#0B171B`,sediment:`#11140A`},r={luciferin:`#E9FFB7`,healthy:`#A8FF5E`,recall:`#29F2A9`,bridge:`#1BD6FF`,latent:`#315CFF`,debt:`#8A4B18`,extinction:`#2A160B`},i=[r.extinction,r.debt,r.healthy,r.luciferin],a={trustMembrane:`#F4F1D0`,caution:`#FFD166`,veto:`#FF3B30`,suppressionScar:`#B90D2B`,labile:`#FF7A1A`},o={forward:`#00F5D4`,retrograde:`#FF2DF7`,receiptSpark:`#FFFFFF`},s={validRing:`#6BFFB8`,txShadow:`#7C6CFF`,supersession:`#FFB000`};function c(e){let n=Math.max(0,Math.min(1,e)),r=i.map(t),a=n*(r.length-1),o=Math.min(r.length-2,Math.floor(a)),s=a-o,c=r[o],l=r[o+1];return[c[0]+(l[0]-c[0])*s,c[1]+(l[1]-c[1])*s,c[2]+(l[2]-c[2])*s]}var l=[58/255,68/255,76/255],u=[1,.78,.36];function d(e,t,n){let r=n<0?0:n>1?1:n;return[e[0]+(t[0]-e[0])*r,e[1]+(t[1]-e[1])*r,e[2]+(t[2]-e[2])*r]}function f(e,t=!1){let n=Math.max(0,Math.min(1,Number.isFinite(e)?e:0)),r=m((n-.3)/.42),i=d(l,c(n),r),a=m((n-.82)/.18);return i=d(i,u,a*.85),t?(i=d(i,u,.7),i=d(i,[1,1,1],.35)):i=d(i,[1,1,1],m((n-.93)/.07)*.5),i}function p(e,t=!1){return t?1:.18+.72*m((Math.max(0,Math.min(1,Number.isFinite(e)?e:0))-.15)/.7)}function m(e){let t=e<0?0:e>1?1:e;return t*t*(3-2*t)}var h={MemoryCreated:r.healthy,SearchPerformed:o.forward,ActivationSpread:r.recall,ImportanceScored:s.supersession,RetentionDecayed:r.debt,ConnectionDiscovered:o.forward,DeepReferenceCompleted:r.luciferin,BackfillFired:o.retrograde,CausalReceipt:o.retrograde,MemorySuppressed:a.suppressionScar,MemoryUnsuppressed:a.labile,MemoryPromoted:r.luciferin,MemoryDemoted:r.debt,MemoryPrOpened:a.caution,MemoryPrDecided:a.trustMembrane,HookVerdictRecorded:a.veto,TraceEvent:o.forward,DreamStarted:s.txShadow,DreamCompleted:s.validRing,Rac1CascadeSwept:a.suppressionScar};function g(e){return t(h[e]??n.blackwater)}function _(e){return .003+.015*Math.max(0,Math.min(1,e))}var v=32,y=126,b=63,x=.6,S=1.32,C=`...`;function w(e){return new Map(e.glyphs.map(e=>[e.unicode,e]))}function T(e){let t=e.codePointAt(0)??b;return t>=v&&t<=y?e:`?`}function E(e,t){if(t===void 0||t<0)return e;if(t===0)return``;let n=Array.from(e,T);return n.length<=t?n.join(``):t<=3?`.`.repeat(t):`${n.slice(0,t-3).join(``)}${C}`}function D(e,t){return e.split(`
`).map(e=>E(e,t))}function O(e,t,n={}){let r=w(t),i=r.get(b),a=t.atlas.width,o=t.atlas.height,s=n.advance??x,c=n.lineHeight??t.metrics?.lineHeight??S,l=n.maxWidthEm===void 0?void 0:Math.max(0,Math.floor(n.maxWidthEm/s)),u=[],d=0;for(let t of D(e,l)){let e=0;for(let n of Array.from(t)){let t=T(n).codePointAt(0)??b,c=r.get(t)??i;if(c?.planeBounds&&c.atlasBounds){let t=c.planeBounds,n=c.atlasBounds,r=n.left/a,i=1-n.top/o,s=(n.right-n.left)/a,l=1-n.bottom/o-i;u.push({x:e+t.left,y:d+t.bottom,w:t.right-t.left,h:t.top-t.bottom,u:r,v:i,uw:s,vh:l})}e+=s}d-=c}return u}async function k(t){let n=`${e}/msdf/jetbrains-mono.json`,r=`${e}/msdf/jetbrains-mono.png`,i=await fetch(n);if(!i.ok)throw Error(`MSDF atlas JSON failed: ${i.status} ${n}`);let a=await i.json();if(a.atlas?.yOrigin!==`bottom`)throw Error(`MSDF atlas yOrigin must be bottom, got ${a.atlas?.yOrigin??`missing`}`);let o=await fetch(r);if(!o.ok)throw Error(`MSDF atlas PNG failed: ${o.status} ${r}`);let s=await o.blob(),c=await createImageBitmap(s),l=t.createTexture({label:`msdf-jetbrains-mono-rgba8unorm`,size:[c.width,c.height,1],format:`rgba8unorm`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});t.queue.copyExternalImageToTexture({source:c},{texture:l},{width:c.width,height:c.height}),c.close?.();let u=t.createSampler({label:`msdf-jetbrains-mono-linear-sampler`,magFilter:`linear`,minFilter:`linear`,mipmapFilter:`linear`,addressModeU:`clamp-to-edge`,addressModeV:`clamp-to-edge`}),d=l.createView({label:`msdf-jetbrains-mono-view`}),f=new Map(a.glyphs.map(e=>[e.unicode,e]));return{...a,glyphMap:f,texture:l,textureView:d,sampler:u,dispose:()=>l.destroy()}}var A=`
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
`,j=20,M=[...t(`#22C7DE`),1],N=class{engine;atlas=null;bindLayout=null;pipeline=null;glyphBuffer=null;bindGroup=null;glyphCapacity=0;glyphCount=0;pendingItems=[];runs=[];runDepths=new Map;initPromise=null;onResize=null;resizeRaf=0;lastAspectBucket=-1;constructor(e){this.engine=e,this.installResizeReflow()}installResizeReflow(){typeof window>`u`||(this.onResize=()=>{this.resizeRaf||=requestAnimationFrame(()=>{this.resizeRaf=0;let e=this.aspectBucket();e!==this.lastAspectBucket&&this.pendingItems.length&&(this.lastAspectBucket=e,this.uploadItems(this.pendingItems))})},window.addEventListener(`resize`,this.onResize),window.addEventListener(`orientationchange`,this.onResize))}aspectBucket(){let e=this.engine.params[6]||0,t=this.engine.params[7]||0;return(e<=0||t<=0)&&typeof window<`u`&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0?-1:Math.round(e/t*8)}async init(){return this.initPromise||=this.initInner(),this.initPromise}async initInner(){let e=this.engine.gpuDevice;e&&this.engine.paramsBuffer&&(this.atlas=await k(e),this.ensurePipeline(e),this.pendingItems.length&&this.uploadItems(this.pendingItems))}setText(e){let t=typeof e==`string`?[{text:e,x:-.62,y:0,size:.075}]:Array.isArray(e)?e:[e];this.pendingItems=t,this.uploadItems(t)}ensurePipeline(e){if(this.pipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`msdf-text-wgsl`,code:A});this.bindLayout=e.createBindGroupLayout({label:`msdf-text-bind-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}}]});let n=e.createPipelineLayout({label:`msdf-text-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}};this.pipeline=e.createRenderPipeline({label:`msdf-text-pipeline`,layout:n,vertex:{module:t,entryPoint:`vs_text`},fragment:{module:t,entryPoint:`fs_text`,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`}})}portraitAdapt(e){let t=this.engine.params[6]||0,n=this.engine.params[7]||0;if((t<=0||n<=0)&&typeof window<`u`&&(t=window.innerWidth,n=window.innerHeight),t<=0||n<=0)return e;let r=t/n;if(r>=.85)return e;let i=1/Math.max(r,.2),a=i,o=1.25*i,s=.5*F((.85-r)/(.85-.46)),c=-.9,l=.62,u=e=>!!e&&(e.startsWith(`route-nav`)||e===`route-chrome`||e===`route-telemetry`||e===`route-status`||e===`route-status-pulse`);return e.map(e=>{if(u(e.kind)){let t=(e.size??.03)*Math.min(1.5,1.1*i);return{...e,size:t}}let t=e.size??.075,n=Math.max(1,e.text.length),r=.96-Math.max(c,e.x*(1-s)),d=r/(n*l),f=Math.max(t,Math.min(t*o,d)),p=Math.max(c,e.x*(1-s)),m=.92*i,h=e.y*a;h>m?h=m:h<-m&&(h=-m);let g=Math.floor(r/(f*l)),_=e.maxWidthEm==null?e.maxWidthEm:Math.max(14,Math.min(e.maxWidthEm,g)),v={...e,x:p,y:h,size:f,maxWidthEm:_};return typeof window<`u`&&window.location?.search.includes(`dbg=1`)&&(window.__adaptDbg??=[],window.__adaptDbg.push({id:e.id,text:e.text?.slice(0,24),x:+p.toFixed(3),y:+h.toFixed(3),size:+f.toFixed(4),maxWidthEm:_})),v})}uploadItems(e){let t=this.engine.gpuDevice;if(!t||!this.engine.paramsBuffer||!this.atlas)return;this.ensurePipeline(t);let n=this.portraitAdapt(e),r=[],i=[],a=0;n.forEach((e,t)=>{let n=e.size??.075,o=O(e.text,this.atlas,{maxWidthEm:e.maxWidthEm}),s=e.color??M,c=o,l=a,u=1/0,d=-1/0,f=1/0,p=-1/0;for(let t of c){let i=e.x+t.x*n,o=e.x+(t.x+t.w)*n,c=e.y+t.y*n,l=e.y+(t.y+t.h)*n;u=Math.min(u,i),d=Math.max(d,o),f=Math.min(f,c),p=Math.max(p,l),P(r,e,t,s,n,a++)}if(c.length>0){let n=e.id??`msdf-text:${t}`;i.push({id:n,kind:e.kind??`text`,text:e.text,x0:u,x1:d,y0:f,y1:p,payload:e,glyphStart:l,glyphCount:a-l}),this.runDepths.set(n,F(e.depth??.5))}}),this.runs=i,this.glyphCount=r.length/j;let o=new Float32Array(r.length||j);o.set(r),this.ensureGlyphBuffer(t,Math.max(1,this.glyphCount)),this.glyphBuffer&&this.bindLayout&&(t.queue.writeBuffer(this.glyphBuffer,0,o),this.bindGroup=t.createBindGroup({label:`msdf-text-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.glyphBuffer}},{binding:2,resource:this.atlas.sampler},{binding:3,resource:this.atlas.textureView}]}))}ensureGlyphBuffer(e,t){this.glyphBuffer&&this.glyphCapacity>=t||(this.glyphBuffer?.destroy(),this.glyphCapacity=Math.max(t,Math.ceil(this.glyphCapacity*1.5),32),this.glyphBuffer=e.createBuffer({label:`msdf-text-glyphs`,size:this.glyphCapacity*j*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}))}render(e){!this.pipeline||!this.bindGroup||this.glyphCount<=0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.glyphCount))}pickAt(e,t){let n=Math.max(1e-4,(this.engine.params[6]||1)/Math.max(1,this.engine.params[7]||1)),r=Math.max(n,1),i=Math.min(n,1);for(let n of this.runs){let a=(n.payload.hitPadX??n.payload.hitPad??0)/r,o=(n.payload.hitPadY??n.payload.hitPad??0)*i,s=n.x0/r-a,c=n.x1/r+a,l=n.y0*i-o,u=n.y1*i+o;if(e>=s&&e<=c&&t>=l&&t<=u)return{id:n.id,kind:n.kind,payload:n.payload}}return null}setRunDepth(e,t=.5){let n=this.engine.gpuDevice;if(n&&this.glyphBuffer)for(let r of this.runs){let i=F(r.id===e?t:r.payload.depth??.5);if(this.runDepths.get(r.id)===i)continue;this.runDepths.set(r.id,i);let a=new Float32Array([i]);for(let e=0;e<r.glyphCount;e+=1){let t=(r.glyphStart+e)*j+14;n.queue.writeBuffer(this.glyphBuffer,t*4,a)}}}dispose(){this.onResize&&typeof window<`u`&&(window.removeEventListener(`resize`,this.onResize),window.removeEventListener(`orientationchange`,this.onResize)),this.onResize=null,this.resizeRaf&&cancelAnimationFrame(this.resizeRaf),this.resizeRaf=0,this.glyphBuffer?.destroy(),this.glyphBuffer=null,this.atlas?.dispose(),this.atlas=null,this.bindGroup=null,this.pipeline=null}};function P(e,t,n,r,i,a){let o=(t.startFrame??0)+a*2,s=t.revealSpan??18;e.push(t.x,t.y,n.w*i,n.h*i,n.x*i,n.y*i,0,0,n.u,n.v,n.u+n.uw,n.v+n.vh,o,s,F(t.depth??.5),F(t.weight??.5),r[0],r[1],r[2],r[3])}function F(e){return Math.min(1,Math.max(0,Number.isFinite(e)?e:.5))}export{n as a,_ as c,p as d,f,a as i,c as l,s as n,r as o,o as r,g as s,N as t,t as u};
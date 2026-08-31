var H=Object.defineProperty;var j=(n,e,t)=>e in n?H(n,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):n[e]=t;var h=(n,e,t)=>j(n,typeof e!="symbol"?e+"":e,t);import{b as U}from"./BobykrIY.js";function G(n){const e=/^#?([0-9a-fA-F]{6})$/.exec(n.trim());if(!e)return[2/255,3/255,7/255];const t=parseInt(e[1],16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}const q={blackwater:"#020307"},v={luciferin:"#E9FFB7",healthy:"#A8FF5E",recall:"#29F2A9",bridge:"#1BD6FF",debt:"#8A4B18",extinction:"#2A160B"},Y=[v.extinction,v.debt,v.healthy,v.luciferin],F={trustMembrane:"#F4F1D0",caution:"#FFD166",veto:"#FF3B30",suppressionScar:"#B90D2B",labile:"#FF7A1A"},A={forward:"#00F5D4",retrograde:"#FF2DF7"},z={validRing:"#6BFFB8",txShadow:"#7C6CFF",supersession:"#FFB000"};function $(n){const e=Math.max(0,Math.min(1,n)),t=Y.map(G),i=e*(t.length-1),s=Math.min(t.length-2,Math.floor(i)),l=i-s,r=t[s],c=t[s+1];return[r[0]+(c[0]-r[0])*l,r[1]+(c[1]-r[1])*l,r[2]+(c[2]-r[2])*l]}const X=[58/255,68/255,76/255],O=[1,.78,.36];function S(n,e,t){const i=t<0?0:t>1?1:t;return[n[0]+(e[0]-n[0])*i,n[1]+(e[1]-n[1])*i,n[2]+(e[2]-n[2])*i]}function he(n,e=!1){const t=Math.max(0,Math.min(1,Number.isFinite(n)?n:0)),i=P((t-.3)/.42);let s=S(X,$(t),i);const l=P((t-.82)/.18);return s=S(s,O,l*.85),e?(s=S(s,O,.7),s=S(s,[1,1,1],.35)):s=S(s,[1,1,1],P((t-.93)/.07)*.5),s}function pe(n,e=!1){const t=Math.max(0,Math.min(1,Number.isFinite(n)?n:0));return e?1:.18+.72*P((t-.15)/.7)}function P(n){const e=n<0?0:n>1?1:n;return e*e*(3-2*e)}const K={MemoryCreated:v.healthy,SearchPerformed:A.forward,ActivationSpread:v.recall,ImportanceScored:z.supersession,RetentionDecayed:v.debt,ConnectionDiscovered:A.forward,DeepReferenceCompleted:v.luciferin,BackfillFired:A.retrograde,CausalReceipt:A.retrograde,MemorySuppressed:F.suppressionScar,MemoryUnsuppressed:F.labile,MemoryPromoted:v.luciferin,MemoryDemoted:v.debt,MemoryPrOpened:F.caution,MemoryPrDecided:F.trustMembrane,HookVerdictRecorded:F.veto,TraceEvent:A.forward,DreamStarted:z.txShadow,DreamCompleted:z.validRing,Rac1CascadeSwept:F.suppressionScar};function de(n){return G(K[n]??q.blackwater)}function fe(n){const e=Math.max(0,Math.min(1,n));return .003+(.018-.003)*e}const Q=32,J=126,N=63,Z=.6,ee=1.32,D="...";function te(n){return new Map(n.glyphs.map(e=>[e.unicode,e]))}function k(n){const e=n.codePointAt(0)??N;return e>=Q&&e<=J?n:"?"}function ne(n,e){if(e===void 0||e<0)return n;if(e===0)return"";const t=Array.from(n,k);return t.length<=e?t.join(""):e<=D.length?".".repeat(e):`${t.slice(0,e-D.length).join("")}${D}`}function se(n,e){return n.split(`
`).map(t=>ne(t,e))}function ie(n,e,t={}){var f;const i=te(e),s=i.get(N),l=e.atlas.width,r=e.atlas.height,c=t.advance??Z,o=t.lineHeight??((f=e.metrics)==null?void 0:f.lineHeight)??ee,d=t.maxWidthEm===void 0?void 0:Math.max(0,Math.floor(t.maxWidthEm/c)),p=[];let x=0;for(const b of se(n,d)){let a=0;for(const g of Array.from(b)){const _=k(g).codePointAt(0)??N,m=i.get(_)??s;if(m!=null&&m.planeBounds&&m.atlasBounds){const u=m.planeBounds,w=m.atlasBounds,M=w.left/l,y=1-w.top/r,I=(w.right-w.left)/l,T=1-w.bottom/r-y;p.push({x:a+u.left,y:x+u.bottom,w:u.right-u.left,h:u.top-u.bottom,u:M,v:y,uw:I,vh:T})}a+=c}x-=o}return p}async function re(n){var f,b,a;const e=`${U}/msdf/jetbrains-mono.json`,t=`${U}/msdf/jetbrains-mono.png`,i=await fetch(e);if(!i.ok)throw new Error(`MSDF atlas JSON failed: ${i.status} ${e}`);const s=await i.json();if(((f=s.atlas)==null?void 0:f.yOrigin)!=="bottom")throw new Error(`MSDF atlas yOrigin must be bottom, got ${((b=s.atlas)==null?void 0:b.yOrigin)??"missing"}`);const l=await fetch(t);if(!l.ok)throw new Error(`MSDF atlas PNG failed: ${l.status} ${t}`);const r=await l.blob(),c=await createImageBitmap(r),o=n.createTexture({label:"msdf-jetbrains-mono-rgba8unorm",size:[c.width,c.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});n.queue.copyExternalImageToTexture({source:c},{texture:o},{width:c.width,height:c.height}),(a=c.close)==null||a.call(c);const d=n.createSampler({label:"msdf-jetbrains-mono-linear-sampler",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),p=o.createView({label:"msdf-jetbrains-mono-view"}),x=new Map(s.glyphs.map(g=>[g.unicode,g]));return{...s,glyphMap:x,texture:o,textureView:p,sampler:d,dispose:()=>o.destroy()}}const ae=`
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
`,B=20,oe=[...G("#22C7DE"),1];class me{constructor(e){h(this,"engine");h(this,"atlas",null);h(this,"bindLayout",null);h(this,"pipeline",null);h(this,"glyphBuffer",null);h(this,"bindGroup",null);h(this,"glyphCapacity",0);h(this,"glyphCount",0);h(this,"pendingItems",[]);h(this,"runs",[]);h(this,"runDepths",new Map);h(this,"initPromise",null);h(this,"onResize",null);h(this,"resizeRaf",0);h(this,"lastAspectBucket",-1);this.engine=e,this.installResizeReflow()}installResizeReflow(){typeof window>"u"||(this.onResize=()=>{this.resizeRaf||(this.resizeRaf=requestAnimationFrame(()=>{this.resizeRaf=0;const e=this.aspectBucket();e!==this.lastAspectBucket&&this.pendingItems.length&&(this.lastAspectBucket=e,this.uploadItems(this.pendingItems))}))},window.addEventListener("resize",this.onResize),window.addEventListener("orientationchange",this.onResize))}aspectBucket(){let e=this.engine.params[6]||0,t=this.engine.params[7]||0;return(e<=0||t<=0)&&typeof window<"u"&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0?-1:Math.round(e/t*8)}async init(){return this.initPromise?this.initPromise:(this.initPromise=this.initInner(),this.initPromise)}async initInner(){const e=this.engine.gpuDevice;!e||!this.engine.paramsBuffer||(this.atlas=await re(e),this.ensurePipeline(e),this.pendingItems.length&&this.uploadItems(this.pendingItems))}setText(e){const t=typeof e=="string"?[{text:e,x:-.62,y:0,size:.075}]:Array.isArray(e)?e:[e];this.pendingItems=t,this.uploadItems(t)}ensurePipeline(e){if(this.pipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"msdf-text-wgsl",code:ae});this.bindLayout=e.createBindGroupLayout({label:"msdf-text-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const i=e.createPipelineLayout({label:"msdf-text-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),s={color:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}};this.pipeline=e.createRenderPipeline({label:"msdf-text-pipeline",layout:i,vertex:{module:t,entryPoint:"vs_text"},fragment:{module:t,entryPoint:"fs_text",targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:"triangle-list"}})}portraitAdapt(e){let t=this.engine.params[6]||0,i=this.engine.params[7]||0;if((t<=0||i<=0)&&typeof window<"u"&&(t=window.innerWidth,i=window.innerHeight),t<=0||i<=0)return e;const s=t/i;if(s>=.85)return e;const l=1/Math.max(s,.2),r=l,c=1.25*l,d=.5*R((.85-s)/(.85-.46)),p=-.9,x=.96,f=.62,b=a=>!!a&&(a.startsWith("route-nav")||a==="route-chrome"||a==="route-telemetry"||a==="route-status"||a==="route-status-pulse");return e.map(a=>{var L,C;if(b(a.kind)){const V=(a.size??.03)*Math.min(1.5,1.1*l);return{...a,size:V}}const g=a.size??.075,E=Math.max(1,a.text.length),_=x-Math.max(p,a.x*(1-d)),m=_/(E*f),u=Math.max(g,Math.min(g*c,m)),w=Math.max(p,a.x*(1-d)),M=.92*l;let y=a.y*r;y>M?y=M:y<-M&&(y=-M);const I=Math.floor(_/(u*f)),T=a.maxWidthEm!=null?Math.max(14,Math.min(a.maxWidthEm,I)):a.maxWidthEm,W={...a,x:w,y,size:u,maxWidthEm:T};return typeof window<"u"&&((L=window.location)!=null&&L.search.includes("dbg=1"))&&(window.__adaptDbg??(window.__adaptDbg=[]),window.__adaptDbg.push({id:a.id,text:(C=a.text)==null?void 0:C.slice(0,24),x:+w.toFixed(3),y:+y.toFixed(3),size:+u.toFixed(4),maxWidthEm:T})),W})}uploadItems(e){const t=this.engine.gpuDevice;if(!t||!this.engine.paramsBuffer||!this.atlas)return;this.ensurePipeline(t);const i=this.portraitAdapt(e),s=[],l=[];let r=0;i.forEach((o,d)=>{const p=o.size??.075,x=ie(o.text,this.atlas,{maxWidthEm:o.maxWidthEm}),f=o.color??oe,b=x,a=r;let g=Number.POSITIVE_INFINITY,E=Number.NEGATIVE_INFINITY,_=Number.POSITIVE_INFINITY,m=Number.NEGATIVE_INFINITY;for(const u of b){const w=o.x+u.x*p,M=o.x+(u.x+u.w)*p,y=o.y+u.y*p,I=o.y+(u.y+u.h)*p;g=Math.min(g,w),E=Math.max(E,M),_=Math.min(_,y),m=Math.max(m,I),le(s,o,u,f,p,r++)}if(b.length>0){const u=o.id??`msdf-text:${d}`;l.push({id:u,kind:o.kind??"text",text:o.text,x0:g,x1:E,y0:_,y1:m,payload:o,glyphStart:a,glyphCount:r-a}),this.runDepths.set(u,R(o.depth??.5))}}),this.runs=l,this.glyphCount=s.length/B;const c=new Float32Array(s.length||B);c.set(s),this.ensureGlyphBuffer(t,Math.max(1,this.glyphCount)),!(!this.glyphBuffer||!this.bindLayout)&&(t.queue.writeBuffer(this.glyphBuffer,0,c),this.bindGroup=t.createBindGroup({label:"msdf-text-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.glyphBuffer}},{binding:2,resource:this.atlas.sampler},{binding:3,resource:this.atlas.textureView}]}))}ensureGlyphBuffer(e,t){var i;this.glyphBuffer&&this.glyphCapacity>=t||((i=this.glyphBuffer)==null||i.destroy(),this.glyphCapacity=Math.max(t,Math.ceil(this.glyphCapacity*1.5),32),this.glyphBuffer=e.createBuffer({label:"msdf-text-glyphs",size:this.glyphCapacity*B*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}))}render(e){!this.pipeline||!this.bindGroup||this.glyphCount<=0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.glyphCount))}pickAt(e,t){const i=Math.max(1e-4,(this.engine.params[6]||1)/Math.max(1,this.engine.params[7]||1)),s=Math.max(i,1),l=Math.min(i,1);for(const r of this.runs){const c=(r.payload.hitPadX??r.payload.hitPad??0)/s,o=(r.payload.hitPadY??r.payload.hitPad??0)*l,d=r.x0/s-c,p=r.x1/s+c,x=r.y0*l-o,f=r.y1*l+o;if(e>=d&&e<=p&&t>=x&&t<=f)return{id:r.id,kind:r.kind,payload:r.payload}}return null}setRunDepth(e,t=.5){const i=this.engine.gpuDevice;if(!(!i||!this.glyphBuffer))for(const s of this.runs){const l=s.id===e?t:s.payload.depth??.5,r=R(l);if(this.runDepths.get(s.id)===r)continue;this.runDepths.set(s.id,r);const c=new Float32Array([r]);for(let o=0;o<s.glyphCount;o+=1){const d=(s.glyphStart+o)*B+14;i.queue.writeBuffer(this.glyphBuffer,d*4,c)}}}dispose(){var e,t;this.onResize&&typeof window<"u"&&(window.removeEventListener("resize",this.onResize),window.removeEventListener("orientationchange",this.onResize)),this.onResize=null,this.resizeRaf&&cancelAnimationFrame(this.resizeRaf),this.resizeRaf=0,(e=this.glyphBuffer)==null||e.destroy(),this.glyphBuffer=null,(t=this.atlas)==null||t.dispose(),this.atlas=null,this.bindGroup=null,this.pipeline=null}}function le(n,e,t,i,s,l){const r=(e.startFrame??0)+l*2,c=e.revealSpan??18;n.push(e.x,e.y,t.w*s,t.h*s,t.x*s,t.y*s,0,0,t.u,t.v,t.u+t.uw,t.v+t.vh,r,c,R(e.depth??.5),R(e.weight??.5),i[0],i[1],i[2],i[3])}function R(n){return Math.min(1,Math.max(0,Number.isFinite(n)?n:.5))}export{z as B,A as C,F as I,q as M,v as R,me as T,$ as a,he as b,de as e,fe as m,G as r,pe as s};

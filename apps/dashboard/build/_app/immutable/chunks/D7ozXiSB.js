var F=Object.defineProperty;var N=(a,e,t)=>e in a?F(a,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):a[e]=t;var u=(a,e,t)=>N(a,typeof e!="symbol"?e+"":e,t);import{r as D}from"./BMB5u1EX.js";import{b as G}from"./DY7cP31Q.js";const L=32,U=126,P=63,z=.6,R=1.32,S="...";function O(a){return new Map(a.glyphs.map(e=>[e.unicode,e]))}function A(a){const e=a.codePointAt(0)??P;return e>=L&&e<=U?a:"?"}function C(a,e){if(e===void 0||e<0)return a;if(e===0)return"";const t=Array.from(a,A);return t.length<=e?t.join(""):e<=S.length?".".repeat(e):`${t.slice(0,e-S.length).join("")}${S}`}function k(a,e){return a.split(`
`).map(t=>C(t,e))}function V(a,e,t={}){var h;const r=O(e),s=r.get(P),o=e.atlas.width,i=e.atlas.height,n=t.advance??z,c=t.lineHeight??((h=e.metrics)==null?void 0:h.lineHeight)??R,p=t.maxWidthEm===void 0?void 0:Math.max(0,Math.floor(t.maxWidthEm/n)),d=[];let g=0;for(const x of k(a,p)){let f=0;for(const m of Array.from(x)){const _=A(m).codePointAt(0)??P,l=r.get(_)??s;if(l!=null&&l.planeBounds&&l.atlasBounds){const y=l.planeBounds,v=l.atlasBounds,T=v.left/o,w=1-v.top/i,B=(v.right-v.left)/o,M=1-v.bottom/i-w;d.push({x:f+y.left,y:g+y.bottom,w:y.right-y.left,h:y.top-y.bottom,u:T,v:w,uw:B,vh:M})}f+=n}g-=c}return d}async function j(a){var h,x,f;const e=`${G}/msdf/jetbrains-mono.json`,t=`${G}/msdf/jetbrains-mono.png`,r=await fetch(e);if(!r.ok)throw new Error(`MSDF atlas JSON failed: ${r.status} ${e}`);const s=await r.json();if(((h=s.atlas)==null?void 0:h.yOrigin)!=="bottom")throw new Error(`MSDF atlas yOrigin must be bottom, got ${((x=s.atlas)==null?void 0:x.yOrigin)??"missing"}`);const o=await fetch(t);if(!o.ok)throw new Error(`MSDF atlas PNG failed: ${o.status} ${t}`);const i=await o.blob(),n=await createImageBitmap(i),c=a.createTexture({label:"msdf-jetbrains-mono-rgba8unorm",size:[n.width,n.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});a.queue.copyExternalImageToTexture({source:n},{texture:c},{width:n.width,height:n.height}),(f=n.close)==null||f.call(n);const p=a.createSampler({label:"msdf-jetbrains-mono-linear-sampler",magFilter:"linear",minFilter:"linear",mipmapFilter:"linear",addressModeU:"clamp-to-edge",addressModeV:"clamp-to-edge"}),d=c.createView({label:"msdf-jetbrains-mono-view"}),g=new Map(s.glyphs.map(m=>[m.unicode,m]));return{...s,glyphMap:g,texture:c,textureView:d,sampler:p,dispose:()=>c.destroy()}}const H=`
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
`,E=20,q=[...D("#22C7DE"),1];class K{constructor(e){u(this,"engine");u(this,"atlas",null);u(this,"bindLayout",null);u(this,"pipeline",null);u(this,"glyphBuffer",null);u(this,"bindGroup",null);u(this,"glyphCapacity",0);u(this,"glyphCount",0);u(this,"pendingItems",[]);u(this,"runs",[]);u(this,"runDepths",new Map);u(this,"initPromise",null);this.engine=e}async init(){return this.initPromise?this.initPromise:(this.initPromise=this.initInner(),this.initPromise)}async initInner(){const e=this.engine.gpuDevice;!e||!this.engine.paramsBuffer||(this.atlas=await j(e),this.ensurePipeline(e),this.pendingItems.length&&this.uploadItems(this.pendingItems))}setText(e){const t=typeof e=="string"?[{text:e,x:-.62,y:0,size:.075}]:Array.isArray(e)?e:[e];this.pendingItems=t,this.uploadItems(t)}ensurePipeline(e){if(this.pipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"msdf-text-wgsl",code:H});this.bindLayout=e.createBindGroupLayout({label:"msdf-text-bind-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const r=e.createPipelineLayout({label:"msdf-text-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),s={color:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}};this.pipeline=e.createRenderPipeline({label:"msdf-text-pipeline",layout:r,vertex:{module:t,entryPoint:"vs_text"},fragment:{module:t,entryPoint:"fs_text",targets:[{format:this.engine.sceneFormat,blend:s}]},primitive:{topology:"triangle-list"}})}uploadItems(e){const t=this.engine.gpuDevice;if(!t||!this.engine.paramsBuffer||!this.atlas)return;this.ensurePipeline(t);const r=[],s=[];let o=0;e.forEach((n,c)=>{const p=n.size??.075,d=V(n.text,this.atlas,{maxWidthEm:n.maxWidthEm}),g=n.color??q,h=d,x=o;let f=Number.POSITIVE_INFINITY,m=Number.NEGATIVE_INFINITY,b=Number.POSITIVE_INFINITY,_=Number.NEGATIVE_INFINITY;for(const l of h){const y=n.x+l.x*p,v=n.x+(l.x+l.w)*p,T=n.y+l.y*p,w=n.y+(l.y+l.h)*p;f=Math.min(f,y),m=Math.max(m,v),b=Math.min(b,T),_=Math.max(_,w),W(r,n,l,g,p,o++)}if(h.length>0){const l=n.id??`msdf-text:${c}`;s.push({id:l,kind:n.kind??"text",text:n.text,x0:f,x1:m,y0:b,y1:_,payload:n,glyphStart:x,glyphCount:o-x}),this.runDepths.set(l,I(n.depth??.5))}}),this.runs=s,this.glyphCount=r.length/E;const i=new Float32Array(r.length||E);i.set(r),this.ensureGlyphBuffer(t,Math.max(1,this.glyphCount)),!(!this.glyphBuffer||!this.bindLayout)&&(t.queue.writeBuffer(this.glyphBuffer,0,i),this.bindGroup=t.createBindGroup({label:"msdf-text-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:this.glyphBuffer}},{binding:2,resource:this.atlas.sampler},{binding:3,resource:this.atlas.textureView}]}))}ensureGlyphBuffer(e,t){var r;this.glyphBuffer&&this.glyphCapacity>=t||((r=this.glyphBuffer)==null||r.destroy(),this.glyphCapacity=Math.max(t,Math.ceil(this.glyphCapacity*1.5),32),this.glyphBuffer=e.createBuffer({label:"msdf-text-glyphs",size:this.glyphCapacity*E*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}))}render(e){!this.pipeline||!this.bindGroup||this.glyphCount<=0||(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.draw(6,this.glyphCount))}pickAt(e,t){const r=Math.max(1e-4,(this.engine.params[6]||1)/Math.max(1,this.engine.params[7]||1)),s=Math.max(r,1),o=Math.min(r,1);for(const i of this.runs){const n=(i.payload.hitPadX??i.payload.hitPad??0)/s,c=(i.payload.hitPadY??i.payload.hitPad??0)*o,p=i.x0/s-n,d=i.x1/s+n,g=i.y0*o-c,h=i.y1*o+c;if(e>=p&&e<=d&&t>=g&&t<=h)return{id:i.id,kind:i.kind,payload:i.payload}}return null}setRunDepth(e,t=.5){const r=this.engine.gpuDevice;if(!(!r||!this.glyphBuffer))for(const s of this.runs){const o=s.id===e?t:s.payload.depth??.5,i=I(o);if(this.runDepths.get(s.id)===i)continue;this.runDepths.set(s.id,i);const n=new Float32Array([i]);for(let c=0;c<s.glyphCount;c+=1){const p=(s.glyphStart+c)*E+14;r.queue.writeBuffer(this.glyphBuffer,p*4,n)}}}dispose(){var e,t;(e=this.glyphBuffer)==null||e.destroy(),this.glyphBuffer=null,(t=this.atlas)==null||t.dispose(),this.atlas=null,this.bindGroup=null,this.pipeline=null}}function W(a,e,t,r,s,o){const i=(e.startFrame??0)+o*2,n=e.revealSpan??18;a.push(e.x,e.y,t.w*s,t.h*s,t.x*s,t.y*s,0,0,t.u,t.v,t.u+t.uw,t.v+t.vh,i,n,I(e.depth??.5),I(e.weight??.5),r[0],r[1],r[2],r[3])}function I(a){return Math.min(1,Math.max(0,Number.isFinite(a)?a:.5))}export{K as T};

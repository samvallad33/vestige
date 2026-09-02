var V=Object.defineProperty;var I=(a,e,t)=>e in a?V(a,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):a[e]=t;var f=(a,e,t)=>I(a,typeof e!="symbol"?e+"":e,t);import{r as g,C as E,I as C,R as x,a as z}from"./Byu5DFqz.js";const M=`
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

// One living cell. 16 floats = 4 vec4, 16-byte aligned lanes.
struct LivingCellGpu {
	// x,y NDC base position; z billboard radius; w orbit ring radius (== length(xy))
	pos_radius: vec4f,
	// rgb hue; w energy 0..1 (brightness / activation)
	hue_energy: vec4f,
	// x orbit phase (0..1); y flags (bit0 selected, bit1 endangered/scar, bit2 pulse-strong);
	// z secondary metric (retention-ish 0..1); w spin scale
	phase_flags: vec4f,
	// x ring index / group; y satellite twinkle seed; z,w reserved
	extra: vec4f,
};

const TAU: f32 = 6.28318530718;

// Living orbital drift — the thing that makes the field MOVE. Deterministic:
// a pure function of params.time + the cell's own phase. ring_spin gives inner
// (high-phase) cells a faster turn, like a spinning galaxy core.
fn ring_spin(phase01: f32) -> f32 {
	let speed = 0.045 + phase01 * 0.10;
	return params.time * speed;
}

fn orbit(base: vec2f, phase01: f32, spin_scale: f32) -> vec2f {
	let radius = length(base);
	if (radius < 0.0001) { return base; }
	let ang0 = atan2(base.y, base.x);
	let ang = ang0 + ring_spin(phase01) * spin_scale + sin(params.time * 0.6 + phase01 * TAU) * 0.02;
	let rr = radius * (1.0 + 0.016 * sin(params.time * 1.1 + phase01 * TAU));
	return vec2f(cos(ang), sin(ang)) * rr;
}

// Per-field options (a small field-local uniform, NOT the shared Params). Carries
// the global intensity + a "reading well" rectangle: the field emits LESS inside
// the well so text renders on a dim, readable substrate. hw<=0 disables the well.
struct FieldOpts {
	intensity: f32,   // 0..1 global field scale (membrane); cells also honor extra.z
	well_x: f32,      // reading-well rect center, NDC x
	well_y: f32,      // reading-well rect center, NDC y
	well_hw: f32,     // half-width NDC (<=0 disables -> factor 1.0)
	well_hh: f32,     // half-height NDC
	well_floor: f32,  // min multiplier inside the well (e.g. 0.10)
	well_soft: f32,   // edge softness NDC (e.g. 0.22)
	hover_index: f32, // instance index of the hovered cell; <0 = none
};

// 1.0 outside the well, ramping down to well_floor inside it (a soft rectangle).
fn reading_well(uv_ndc: vec2f, o: FieldOpts) -> f32 {
	if (o.well_hw <= 0.0) { return 1.0; }
	let dx = max(0.0, abs(uv_ndc.x - o.well_x) - o.well_hw);
	let dy = max(0.0, abs(uv_ndc.y - o.well_y) - o.well_hh);
	let outside = length(vec2f(dx, dy)) / max(o.well_soft, 0.001);
	return mix(o.well_floor, 1.0, clamp(outside, 0.0, 1.0));
}

// Fossil Light keeps the generic organ field inside the graphite / amber /
// jade family. A small amount of an organ's semantic hue survives, but old
// blue-violet source values are grounded before they reach the HDR bloom pass.
fn somatic_tone(hue: vec3f, persistence: f32) -> vec3f {
	let amber = vec3f(0.62, 0.28, 0.10);
	let jade = vec3f(0.28, 0.68, 0.48);
	let physical = mix(amber, jade, smoothstep(0.14, 0.90, persistence));
	let grounded = vec3f(
		clamp(hue.r, 0.0, 1.0),
		max(clamp(hue.g, 0.0, 1.0), clamp(hue.b, 0.0, 1.0) * 0.70),
		min(clamp(hue.b, 0.0, 1.0), clamp(hue.g, 0.0, 1.0) + 0.08)
	);
	return mix(physical, grounded, 0.16);
}
`,H=`
${M}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f, // x phase, y flags, z metric2, w spin
};

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	// breathe the splat radius so density is never a flat print
	let breathe = 1.0 + 0.14 * sin(params.time * 1.5 + phase * TAU);
	let r = c.pos_radius.z * 2.4 * breathe * (1.0 + c.hue_energy.w * 0.9);
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = c.phase_flags;
	return out;
}

@fragment
fn fs_splat(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let persistence = clamp(in.info.z, 0.0, 1.0);
	let flags = in.info.y;
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	// Low-res density is intentionally soma-heavy. The branch traces are thin
	// enough that the blur turns them into microscopy-like connective tissue,
	// not another field of soft, identical circles.
	let soma = exp(-d * d * mix(10.5, 5.8, persistence));
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * (5.0 + floor(in.info.x * 3.0)) + in.info.x * TAU));
	let branch_band = smoothstep(0.14, 0.34, d) * (1.0 - smoothstep(0.66, 0.92, d));
	let neurites = pow(branch_wave, 16.0) * branch_band * (0.025 + persistence * 0.070);
	let body = soma * (0.22 + energy * 0.68) + neurites;
	// .r = soma-led density, .g = retained oxygen, .b = suppressed/scar seam.
	return vec4f(body, body * (0.32 + persistence * 0.82), body * scar * 0.50, 1.0);
}
`,W=`
struct BlurDir { dir: vec2f, _pad: vec2f };
@group(0) @binding(0) var blur_sampler: sampler;
@group(0) @binding(1) var blur_src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> blur_dir: BlurDir;
const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}
@fragment
fn fs_blur(in: VSOut) -> @location(0) vec4f {
	let dims = vec2f(textureDimensions(blur_src, 0));
	let stepv = blur_dir.dir / max(dims, vec2f(1.0));
	var acc = textureSampleLevel(blur_src, blur_sampler, in.uv - stepv * 2.0, 0.0) * 0.06136;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv - stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv, 0.0) * 0.38774;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv, 0.0) * 0.24477;
	acc = acc + textureSampleLevel(blur_src, blur_sampler, in.uv + stepv * 2.0, 0.0) * 0.06136;
	return acc;
}
`,$=`
${M}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(2) var<uniform> fopts: FieldOpts;
@group(0) @binding(3) var field_sampler: sampler;
@group(0) @binding(4) var field_tex: texture_2d<f32>;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);
struct VSOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VSOut {
	var out: VSOut;
	let p = QUAD[vi];
	out.clip = vec4f(p, 0.0, 1.0);
	out.uv = p * 0.5 + vec2f(0.5);
	return out;
}

@fragment
fn fs_membrane(in: VSOut) -> @location(0) vec4f {
	let f = textureSample(field_tex, field_sampler, in.uv);
	let density = clamp(f.r, 0.0, 5.0);
	let oxygen = clamp(f.g, 0.0, 5.0);
	let scar = clamp(f.b, 0.0, 3.0);
	let breath = 0.72 + 0.55 * params.pulse;
	// Cold blue-black substrate: the fullscreen plasma is desaturated + dimmed hard
	// so it reads as a deep breathing floor, NOT a neon-green wash over text.
	//
	// The old membrane ALSO drew an edge term = smoothstep of the density
	// gradient, which outlined every blurred cell blob → a dense mesh of ugly
	// overlapping grey-green rings/circles across the whole frame ("what even is
	// this"). That edge term is REMOVED. The base plasma cloud contribution is also
	// cut hard so the field reads as clean glowing cells on near-void, not fog.
	let blackwater = vec3f(0.006, 0.012, 0.015);
	let amber = vec3f(0.70, 0.38, 0.14);
	let oxygen_col = vec3f(0.42, 0.70, 0.40); // desaturated (was neon 0.66,1.0,0.37)
	let scarlet = vec3f(0.85, 0.22, 0.18);
	var color = blackwater * (0.30 + density * 0.06);
	color = color + mix(amber, oxygen_col, clamp(oxygen / max(density, 0.001), 0.0, 1.0)) * density * 0.035 * breath;
	color = color + scarlet * scar * (0.20 + 0.12 * params.pulse);
	let vignette = smoothstep(1.02, 0.10, distance(in.uv, vec2f(0.5)));
	// Reading well: the field emits LESS where text lives, so labels read. Plus the
	// per-page intensity. Deeper vignette floor pushes frame edges toward void.
	let uv_ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
	let well = reading_well(uv_ndc, fopts);
	return vec4f(color * (0.35 + 0.65 * vignette) * params.brightness * fopts.intensity * well, 1.0);
}
`,q=`
${M}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<LivingCellGpu>;
@group(0) @binding(2) var<uniform> fopts: FieldOpts;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

struct VSOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) hue_energy: vec4f,
	@location(2) @interpolate(flat) info: vec4f,
	// x = field intensity (0..1, extra.z), y = twinkle seed (extra.y)
	@location(3) @interpolate(flat) extra: vec4f,
	// the cell's orbited center in NDC, so the reading well is evaluated per cell
	@location(4) @interpolate(flat) center_ndc: vec2f,
};

@vertex
fn vs_cell(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	let c = cells[ii];
	let corner = QUAD[vi];
	let phase = c.phase_flags.x;
	let flags = c.phase_flags.y;
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let hovered = select(0.0, 1.0, abs(f32(ii) - fopts.hover_index) < 0.5);
	let beat = 1.0 + 0.22 * sin(params.time * 2.3 + c.extra.y * 1.7);
	let r = c.pos_radius.z * (0.85 + c.hue_energy.w * 0.9 + selected * 1.3 + hovered * 0.55) * beat;
	let center = orbit(c.pos_radius.xy, phase, c.phase_flags.w);
	var out: VSOut;
	out.clip = vec4f(center + corner * r, 0.0, 1.0);
	out.uv = corner;
	out.hue_energy = c.hue_energy;
	out.info = vec4f(c.phase_flags.x, c.phase_flags.y, c.phase_flags.z, hovered);
	out.extra = c.extra;
	out.center_ndc = center;
	return out;
}

@fragment
fn fs_cell(in: VSOut) -> @location(0) vec4f {
	let d = length(in.uv);
	if (d > 1.0) { discard; }
	// Field intensity (extra.z) dims the WHOLE cell so the field is a backdrop the
	// text reads over. Text organs set this low; visual organs keep it high.
	let intensity = clamp(in.extra.x, 0.05, 1.0);
	let hue = in.hue_energy.rgb;
	let energy = clamp(in.hue_energy.w, 0.0, 1.0);
	let flags = in.info.y;
	let selected = select(0.0, 1.0, flags == 1.0 || flags == 3.0 || flags == 5.0 || flags == 7.0);
	let hovered = in.info.w;
	let scar = select(0.0, 1.0, (flags >= 2.0 && flags < 4.0) || flags >= 6.0);
	let phase = in.info.x;
	// Per-cell persistence is supplied by the real organ mapper (retention where
	// available). projection_days is the signed field clock: only a forward
	// projection is treated as age, so historical scrubs never fake decay.
	let persistence = clamp(in.info.z, 0.0, 1.0);
	let chrono_age = clamp(max(params.projection_days, 0.0) / 120.0, 0.0, 1.0);
	let consolidation = clamp(energy * 0.58 + persistence * 0.42, 0.0, 1.0);
	let depth_scatter = (1.0 - consolidation) * (0.24 + chrono_age * 0.76);
	// Twinkle remains below 5%; it gives the soma metabolic motion without a
	// neon pulse. Storage strength concentrates luminance at the centre.
	let twinkle = 0.95 + 0.05 * (0.5 + 0.5 * sin(params.time * 2.1 + phase * 26.0));
	let soma = exp(-d * d * mix(12.5, 6.5, consolidation)) * (0.08 + consolidation * 0.46) * twinkle;
	let theta = atan2(in.uv.y + 0.00001, in.uv.x);
	let branch_wave = max(0.0, 0.5 + 0.5 * sin(theta * (5.0 + floor(fract(in.extra.y * 0.13) * 3.0)) + phase * TAU));
	let branch_band = smoothstep(0.15, 0.34, d) * (1.0 - smoothstep(0.68, 0.94, d));
	let neurites = pow(branch_wave, 17.0) * branch_band * (0.018 + consolidation * 0.075)
		* (1.0 - depth_scatter * 0.68);
	let scatter = pow(max(1.0 - d, 0.0), 3.8) * (0.010 + depth_scatter * 0.075);
	let tinted = somatic_tone(hue, persistence);
	let soma_tone = mix(tinted, vec3f(0.90, 0.96, 0.84), consolidation * 0.40);
	let rim = smoothstep(0.98, 0.72, d) * (1.0 - smoothstep(0.72, 0.40, d));
	let scarlet = vec3f(0.85, 0.22, 0.18);
	let ivory = vec3f(0.90, 0.96, 0.86);
	var color = soma_tone * soma + tinted * neurites + tinted * scatter;
	// The rim is a selected-only instrument mark; hover gets a quieter spare float.
	color = color + ivory * rim * selected * 0.32;
	color = color + ivory * rim * hovered * 0.22;
	// Scarred/suppressed cells lose normal emission and leave a small oxide seam.
	let scar_ring = smoothstep(0.70, 0.78, d) * (1.0 - smoothstep(0.80, 0.90, d));
	color = mix(color, color * 0.10 + scarlet * scar_ring * 0.12, scar);
	// selected/scar stay a touch brighter than the dimmed backdrop so meaning survives.
	let keep = max(intensity, (selected + scar + hovered) * 0.7);
	let well = reading_well(in.center_ndc, fopts);
	return vec4f(color * keep * well, 1.0);
}
`,B="rgba16float",P=2048,L=16;class Y{constructor(e){f(this,"engine");f(this,"cells",[]);f(this,"scalars",{});f(this,"intensity",.28);f(this,"well",{x:0,y:0,hw:-1,hh:0,floor:.1,soft:.22});f(this,"lastAspectBucket",-999);f(this,"resources",null);f(this,"sampler",null);f(this,"splatBindLayout",null);f(this,"blurBindLayout",null);f(this,"membraneBindLayout",null);f(this,"splatPipeline",null);f(this,"blurPipeline",null);f(this,"membranePipeline",null);f(this,"cellPipeline",null);f(this,"cellCount",0);f(this,"hoverIndex",-1);this.engine=e}setIntensity(e){this.intensity=Math.min(1,Math.max(0,Number.isFinite(e)?e:.28));const t=this.engine.gpuDevice;t&&this.writeOpts(t),this.cells.length&&this.setCells(this.cells,this.scalars)}setHovered(e){const t=Number.isFinite(e)?Math.trunc(e):-1;if(t===this.hoverIndex)return;this.hoverIndex=t;const r=this.engine.gpuDevice;r&&this.writeOpts(r)}setCells(e,t={}){this.cells=e.slice(0,P),this.scalars=t;const r=this.engine.gpuDevice;r&&(this.ensurePipelines(r),this.ensureResources(r),this.uploadBuffers(r))}ensurePipelines(e){if(this.splatPipeline||!this.engine.paramsBuffer)return;const t=G(e,"living-field-splat",H),r=G(e,"living-field-blur",W),i=G(e,"living-field-membrane",$),n=G(e,"living-field-cell",q);this.splatBindLayout=e.createBindGroupLayout({label:"living-field-splat-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.blurBindLayout=e.createBindGroupLayout({label:"living-field-blur-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),this.membraneBindLayout=e.createBindGroupLayout({label:"living-field-membrane-layout",entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float"}}]});const l=e.createPipelineLayout({label:"living-field-splat-pl",bindGroupLayouts:[this.splatBindLayout]}),s=e.createPipelineLayout({label:"living-field-blur-pl",bindGroupLayouts:[this.blurBindLayout]}),c=e.createPipelineLayout({label:"living-field-membrane-pl",bindGroupLayouts:[this.membraneBindLayout]});this.sampler=e.createSampler({magFilter:"linear",minFilter:"linear"});const o={color:{srcFactor:"one",dstFactor:"one",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one",operation:"add"}};this.splatPipeline=e.createRenderPipeline({label:"living-field-splat",layout:l,vertex:{module:t,entryPoint:"vs_splat"},fragment:{module:t,entryPoint:"fs_splat",targets:[{format:B,blend:o}]},primitive:{topology:"triangle-list"}}),this.blurPipeline=e.createRenderPipeline({label:"living-field-blur",layout:s,vertex:{module:r,entryPoint:"vs_fullscreen"},fragment:{module:r,entryPoint:"fs_blur",targets:[{format:B}]},primitive:{topology:"triangle-list"}}),this.membranePipeline=e.createRenderPipeline({label:"living-field-membrane",layout:c,vertex:{module:i,entryPoint:"vs_fullscreen"},fragment:{module:i,entryPoint:"fs_membrane",targets:[{format:this.engine.sceneFormat,blend:o}]},primitive:{topology:"triangle-list"}}),this.cellPipeline=e.createRenderPipeline({label:"living-field-cells",layout:l,vertex:{module:n,entryPoint:"vs_cell"},fragment:{module:n,entryPoint:"fs_cell",targets:[{format:this.engine.sceneFormat,blend:o}]},primitive:{topology:"triangle-list"}})}ensureResources(e){var R,U,A,O,F,T;if(!this.splatBindLayout||!this.blurBindLayout||!this.membraneBindLayout||!this.engine.paramsBuffer||!this.sampler)return;const t=Math.max(16,Math.floor((this.engine.params[6]||1280)/2)),r=Math.max(16,Math.floor((this.engine.params[7]||720)/2)),i=!this.resources||this.resources.fieldSize[0]!==t||this.resources.fieldSize[1]!==r;let n=(R=this.resources)==null?void 0:R.cellBuffer,l=(U=this.resources)==null?void 0:U.blurHBuffer,s=(A=this.resources)==null?void 0:A.blurVBuffer,c=(O=this.resources)==null?void 0:O.optsBuffer;if(n||(n=e.createBuffer({label:"living-field-cells",size:P*L*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),l||(l=e.createBuffer({label:"living-field-blur-h",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(l,0,new Float32Array([1,0,0,0]))),s||(s=e.createBuffer({label:"living-field-blur-v",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(s,0,new Float32Array([0,1,0,0]))),c||(c=e.createBuffer({label:"living-field-opts",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})),!i&&this.resources)return;(F=this.resources)==null||F.fieldA.destroy(),(T=this.resources)==null||T.fieldB.destroy();const o=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,u=e.createTexture({label:"living-field-a",size:[t,r],format:B,usage:o}),m=e.createTexture({label:"living-field-b",size:[t,r],format:B,usage:o}),b=u.createView(),h=m.createView(),p=e.createBindGroup({label:"living-field-splat-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:c}}]}),v=e.createBindGroup({label:"living-field-cell-bind",layout:this.splatBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:n}},{binding:2,resource:{buffer:c}}]}),w=e.createBindGroup({label:"living-field-blur-h-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:b},{binding:2,resource:{buffer:l}}]}),y=e.createBindGroup({label:"living-field-blur-v-bind",layout:this.blurBindLayout,entries:[{binding:0,resource:this.sampler},{binding:1,resource:h},{binding:2,resource:{buffer:s}}]}),k=e.createBindGroup({label:"living-field-membrane-bind",layout:this.membraneBindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:2,resource:{buffer:c}},{binding:3,resource:this.sampler},{binding:4,resource:b}]});this.resources={cellBuffer:n,blurHBuffer:l,blurVBuffer:s,optsBuffer:c,splatBindGroup:p,cellBindGroup:v,blurHBindGroup:w,blurVBindGroup:y,membraneBindGroup:k,fieldA:u,fieldB:m,fieldAView:b,fieldBView:h,fieldSize:[t,r]},this.writeOpts(e)}writeOpts(e){if(!this.resources)return;const t=this.portraitWell();e.queue.writeBuffer(this.resources.optsBuffer,0,new Float32Array([this.intensity,t.x,t.y,t.hw,t.hh,t.floor,t.soft,this.hoverIndex]))}aspectBucket(){let e=this.engine.params[6]||0,t=this.engine.params[7]||0;return(e<=0||t<=0)&&typeof window<"u"&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0?-999:Math.round(e/t*8)}portraitWell(){if(this.well.hw<=0)return this.well;let e=this.engine.params[6]||0,t=this.engine.params[7]||0;if((e<=0||t<=0)&&typeof window<"u"&&(e=window.innerWidth,t=window.innerHeight),e<=0||t<=0)return this.well;const r=e/t;if(r>=.85)return this.well;const i=S((.85-r)/(.85-.46)),n=.42*i,l=1+(1/Math.max(r,.2)-1)*(.72*i),s=1+.25*i;return{x:this.well.x*(1-n),y:Q(this.well.y*l,-.98,.98),hw:Math.min(1.1,this.well.hw*s),hh:Math.min(1.1,this.well.hh*l),floor:this.well.floor,soft:this.well.soft}}setReadingWell(e){this.well={x:d(e.x),y:d(e.y),hw:d(e.hw,-1),hh:d(e.hh),floor:S(e.floor??.1),soft:Math.max(.02,d(e.soft??.22,.22))};const t=this.engine.gpuDevice;t&&this.writeOpts(t)}uploadBuffers(e){if(!this.resources)return;const t=new Float32Array(P*L);this.cellCount=Math.min(P,this.cells.length);for(let r=0;r<this.cellCount;r++){const i=this.cells[r],n=d(i.x),l=d(i.y),s=Math.hypot(n,l),c=d(i.phase);let o=0;i.selected&&(o|=1),i.scar&&(o|=2),d(i.energy)>.8&&(o|=4);const u=r*L;t[u+0]=n,t[u+1]=l,t[u+2]=Math.max(.006,d(i.radius,.02)),t[u+3]=s,t[u+4]=d(i.hue[0]),t[u+5]=d(i.hue[1]),t[u+6]=d(i.hue[2]),t[u+7]=S(i.energy),t[u+8]=c,t[u+9]=o,t[u+10]=S(i.metric2??i.energy),t[u+11]=d(i.spin??1,1),t[u+12]=r,t[u+13]=d(i.seed??c*97.13),t[u+14]=this.intensity,t[u+15]=0}e.queue.writeBuffer(this.resources.cellBuffer,0,t)}compute(e){const t=this.engine.gpuDevice;if(!t||!this.resources||!this.splatPipeline||!this.blurPipeline)return;this.ensureResources(t);const r=this.aspectBucket();r!==this.lastAspectBucket&&(this.lastAspectBucket=r,this.writeOpts(t));const i=this.resources,n=e.beginRenderPass({label:"living-field-splat-pass",colorAttachments:[{view:i.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});n.setPipeline(this.splatPipeline),n.setBindGroup(0,i.splatBindGroup),this.cellCount>0&&n.draw(6,this.cellCount),n.end();const l=e.beginRenderPass({label:"living-field-blur-h",colorAttachments:[{view:i.fieldBView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});l.setPipeline(this.blurPipeline),l.setBindGroup(0,i.blurHBindGroup),l.draw(6,1),l.end();const s=e.beginRenderPass({label:"living-field-blur-v",colorAttachments:[{view:i.fieldAView,clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}]});s.setPipeline(this.blurPipeline),s.setBindGroup(0,i.blurVBindGroup),s.draw(6,1),s.end()}render(e){!this.resources||!this.membranePipeline||!this.cellPipeline||(e.setPipeline(this.membranePipeline),e.setBindGroup(0,this.resources.membraneBindGroup),e.draw(6,1),this.cellCount>0&&(e.setPipeline(this.cellPipeline),e.setBindGroup(0,this.resources.cellBindGroup),e.draw(6,this.cellCount)))}orbitCpu(e,t,r,i){const n=Math.hypot(e,t);if(n<1e-4)return{x:e,y:t};const l=this.engine.params[10]||0,s=Math.atan2(t,e),c=(.045+r*.1)*l*i,o=s+c+Math.sin(l*.6+r*Math.PI*2)*.02,u=n*(1+.016*Math.sin(l*1.1+r*Math.PI*2));return{x:Math.cos(o)*u,y:Math.sin(o)*u}}pickAt(e,t){let r=null,i=1/0;for(let n=0;n<this.cellCount;n++){const l=this.cells[n],s=this.orbitCpu(l.x,l.y,l.phase,l.spin??1),c=Math.hypot(e-s.x,t-s.y),o=Math.max(.04,l.radius*2.6);c<=o&&c<i&&(r={id:l.pickId,kind:l.kind??"living-cell",index:n,payload:l.payload??l},i=c)}return this.setHovered((r==null?void 0:r.index)??-1),r}dispose(){var e,t,r,i,n,l;(e=this.resources)==null||e.cellBuffer.destroy(),(t=this.resources)==null||t.blurHBuffer.destroy(),(r=this.resources)==null||r.blurVBuffer.destroy(),(i=this.resources)==null||i.optsBuffer.destroy(),(n=this.resources)==null||n.fieldA.destroy(),(l=this.resources)==null||l.fieldB.destroy(),this.resources=null}}function S(a){return Math.min(1,Math.max(0,Number.isFinite(a)?a:0))}function Q(a,e,t){return Math.min(t,Math.max(e,Number.isFinite(a)?a:e))}function d(a,e=0){return Number.isFinite(a)?a:e}function G(a,e,t){a.pushErrorScope("validation");const r=a.createShaderModule({label:e,code:t});return r.getCompilationInfo().then(i=>{for(const n of i.messages)n.type==="error"&&console.error(`[living-field] ${e} WGSL ${n.type} ${n.lineNum}:${n.linePos} ${n.message}`)}),a.popErrorScope().then(i=>{i&&console.error(`[living-field] ${e} shader module validation: ${i.message}`)}),r}const N=2.399963229728653;function J(a,e={}){const t=a.length;if(t===0)return[];const r=e.maxRadius??.92,i=e.minCellR??.012,n=e.maxCellR??.05;return a.map((s,c)=>({d:s,i:c})).sort((s,c)=>(c.d.score||0)-(s.d.score||0)).map(({d:s},c)=>{const o=t>1?c/(t-1):0,u=r*Math.sqrt(.06+.94*o),m=c*N,b=Math.cos(m)*u,h=Math.sin(m)*u,p=_(s.score),v=i+(n-i)*Math.sqrt(p);return{x:b,y:h,radius:v,hue:D(s.hue,p),energy:_(s.energy??.35+.65*p),phase:c/t,pickId:s.id,kind:s.kind,payload:s.payload??s,selected:s.selected,scar:s.scar,metric2:_(s.metric2??p),spin:1}})}function K(a,e,t={}){if(a.length===0)return[];const i=t.maxRadius??.9,n=t.minCellR??.014,l=t.maxCellR??.05,s=Math.max(1,t.ringCount??new Set(a.map(e)).size),c=new Map;return a.map((o,u)=>{const m=e(o,u),h=((Number.isFinite(m)?Math.floor(m):0)%s+s)%s,p=c.get(h)??0;c.set(h,p+1);const v=i*(.18+.82*(h/Math.max(1,s-1))),w=p*N+h*.7,y=_(o.score);return{x:Math.cos(w)*v,y:Math.sin(w)*v,radius:n+(l-n)*Math.sqrt(y),hue:D(o.hue,y),energy:_(o.energy??.35+.65*y),phase:h/s,pickId:o.id,kind:o.kind,payload:o.payload??o,selected:o.selected,scar:o.scar,metric2:_(o.metric2??y),spin:1}})}const Z={oxygen:g(x.luciferin),healthy:g(x.healthy),recall:g(x.recall),bridge:g(x.bridge),debt:g(x.debt),scarlet:g(C.veto),caution:g(C.caution),forward:g(E.forward),retrograde:g(E.retrograde)};function _(a){return Math.min(1,Math.max(0,Number.isFinite(a)?a:0))}function D(a,e){const t=a??z(e);return[Number.isFinite(t[0])?t[0]:0,Number.isFinite(t[1])?t[1]:0,Number.isFinite(t[2])?t[2]:0]}export{Z as F,Y as L,K as a,J as l};

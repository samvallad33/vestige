import{u as e}from"./X6OYkA-n.js";var t=64,n=96,r=16,i=8,a=`
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

struct WitnessState {
	playhead: f32,
	replay_start: f32,
	selected_index: f32,
	shard_count: f32,
};

struct Shard {
	// xyz = deterministic 3D location, w = wafer base scale
	position_size: vec4f,
	// x activation, y retention, z trace-time 0..1, w selected
	metrics: vec4f,
	// real status color; semantic, never decorative
	color: vec4f,
	// x role, y scar flag, z reveal order, w reserved
	flags: vec4f,
};

struct Filament {
	// x source shard index, y target shard index (only verified path neighbors)
	endpoints: vec4f,
	// x energy, y deterministic phase, z receipt-path flag, w reserved
	motion: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> shards: array<Shard>;
@group(0) @binding(2) var<storage, read> filaments: array<Filament>;
@group(0) @binding(3) var<uniform> witness: WitnessState;

const QUAD = array<vec2f, 6>(
	vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
	vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

// A witness shard is a small, extruded ceramic specimen, not a flat UI card.
// The front face remains deliberately asymmetric; the two side faces make the
// receipt structure legible as a volume even when the chamber is completely
// still.  The depth is part of the actual perspective projection below.
const WAFER = array<vec3f, 18>(
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(-1.0, 1.0, 0.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0),
	vec3f(-1.0, -1.0, 0.0), vec3f(0.66, -1.42, -1.0), vec3f(-1.34, -1.42, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.0, 1.0, 0.0), vec3f(1.34, 0.58, -1.0),
	vec3f(1.0, -1.0, 0.0), vec3f(1.34, 0.58, -1.0), vec3f(0.66, -1.42, -1.0)
);

struct Projection { screen: vec2f, scale: f32 };

// This is a real perspective projection, not a 2D arrangement. Time is depth
// and activation/role form the stable chamber strata. The pointer only shifts
// the 3/4 view by a few degrees: an examination lens, never an auto-orbit.
fn cursor_lens() -> vec2f {
	if (abs(params.cursor_x) > 2.0 || abs(params.cursor_y) > 2.0) {
		return vec2f(0.0, 0.0);
	}
	return clamp(vec2f(params.cursor_x, params.cursor_y), vec2f(-1.0), vec2f(1.0));
}

fn project(world: vec3f) -> Projection {
	let lens = cursor_lens();
	let yaw = lens.x * 0.055;
	let c = cos(yaw);
	let s = sin(yaw);
	let view = vec3f(
		world.x * c - world.z * s,
		world.y + lens.y * 0.035,
		world.x * s + world.z * c
	);
	let depth = clamp(2.82 - view.z, 1.1, 5.4);
	let perspective = 1.0 / depth;
	return Projection(vec2f(view.x * 1.18 * perspective, view.y * 1.62 * perspective), perspective);
}

fn smooth01(value: f32) -> f32 {
	let t = clamp(value, 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

fn reveal_for(shard: Shard) -> f32 {
	let arrival = 14.0 + shard.flags.z * 68.0;
	let ingress = smooth01((params.frame - arrival) / 62.0);
	// The temporal slicer is a real trace cursor. Evidence that was not yet
	// available simply does not materialize.
	let slice = smoothstep(shard.metrics.z - 0.045, shard.metrics.z + 0.09, witness.playhead);
	return ingress * slice;
}

fn replay_age() -> f32 {
	if (witness.replay_start < 0.0) { return 9999.0; }
	var age = params.frame - witness.replay_start;
	if (age < 0.0) { age = age + 720.0; }
	return age;
}

// Quiet mineral palette: jade marks corroborated evidence; the traversal itself
// is fossil amber. There is intentionally no cyan/purple emissive wash.
fn core_color() -> vec3f { return vec3f(0.10, 0.17, 0.14); }

struct WaferOut {
	@builtin(position) clip: vec4f,
	@location(0) local: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) data: vec4f,
	@location(3) @interpolate(flat) face: f32,
};

@vertex
fn vs_wafer(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> WaferOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let sealed = vec3f(0.08, -0.02, 0.12);
	let selectedLift = select(vec3f(0.0), vec3f(0.0, 0.0, 0.34), shard.metrics.w > 0.5);
	let position = mix(sealed, shard.position_size.xyz + selectedLift, reveal);
	let q = WAFER[vi];
	let rotation = -0.15 + (shard.flags.z - 0.5) * 0.30 + (shard.flags.x - 1.5) * 0.075;
	let c = cos(rotation);
	let s = sin(rotation);
	let rq = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	let height = (0.112 + shard.metrics.y * 0.072) * shard.position_size.w;
	let width = height * (2.05 + shard.metrics.x * 0.82);
	let thickness = height * (0.46 + shard.metrics.y * 0.14);
	// The face offset is projected with the specimen, rather than pasted onto
	// the screen. It is a true low-poly wafer with parallax depth.
	let facePosition = position + vec3f(rq.x * width, rq.y * height, q.z * thickness);
	let projected = project(facePosition);
	var out: WaferOut;
	out.clip = vec4f(projected.screen, 0.0, 1.0);
	out.local = q.xy;
	out.color = shard.color.rgb;
	out.data = vec4f(shard.metrics.x, shard.metrics.y, shard.metrics.w, shard.flags.y);
	out.face = floor(f32(vi) / 6.0);
	return out;
}

@fragment
fn fs_wafer(frag: WaferOut) -> @location(0) vec4f {
	let edge = max(abs(frag.local.y), abs(frag.local.x) * 0.76 + frag.local.y * 0.12);
	if (frag.face < 0.5 && edge > 1.0) { discard; }
	let rim = smoothstep(0.72, 0.97, edge);
	let facet = smoothstep(-1.05, 1.05, frag.local.x * 0.78 - frag.local.y * 0.42);
	let scar = frag.data.w;
	let selected = frag.data.z;
	let carbon = vec3f(0.018, 0.031, 0.033);
	var body = mix(carbon, frag.color * (0.14 + facet * 0.18), 0.74);
	if (frag.face > 0.5) {
		let sideLight = select(0.24, 0.42, frag.face > 1.5);
		body = mix(carbon * 1.45, frag.color * sideLight, 0.66);
	} else {
		body = body + frag.color * rim * (0.16 + selected * 0.30);
	}
	if (scar > 0.5) {
		let fracture = smoothstep(0.10, 0.02, abs(frag.local.x + frag.local.y * 0.37));
		body = mix(body, vec3f(0.72, 0.12, 0.09), fracture * 0.76);
	}
	let alpha = select(0.72, 0.86 + selected * 0.14, frag.face < 0.5);
	return vec4f(body, alpha);
}

struct RibbonOut {
	@builtin(position) clip: vec4f,
	@location(0) uv: vec2f,
	@location(1) @interpolate(flat) color: vec3f,
	@location(2) @interpolate(flat) energy: f32,
	@location(3) @interpolate(flat) selected: f32,
};

fn filament_point(index: f32) -> vec3f {
	if (index < -0.5) { return vec3f(0.08, -0.02, 0.12); }
	let shard = shards[u32(index)];
	return mix(vec3f(0.0, 0.0, 0.0), shard.position_size.xyz, reveal_for(shard));
}

fn filament_selected(index: f32) -> f32 {
	if (index < -0.5) { return 0.0; }
	return shards[u32(index)].metrics.w;
}

@vertex
fn vs_filament(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let fiber = filaments[ii];
	let start = project(filament_point(fiber.endpoints.x));
	let end = project(filament_point(fiber.endpoints.y));
	let delta = end.screen - start.screen;
	let distance = max(length(delta), 0.0001);
	let direction = delta / distance;
	let normal = vec2f(-direction.y, direction.x);
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	let width = 0.0017 + fiber.motion.x * 0.0025;
	let selected = max(filament_selected(fiber.endpoints.x), filament_selected(fiber.endpoints.y));
	var out: RibbonOut;
	out.clip = vec4f(mix(start.screen, end.screen, t) + normal * q.y * width, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = mix(core_color(), vec3f(0.46, 0.30, 0.15), fiber.motion.z);
	out.energy = fiber.motion.x;
	out.selected = selected;
	return out;
}

@fragment
fn fs_filament(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.24, abs(frag.uv.y));
	let age = replay_age();
	let travel = fract(age * 0.016 + frag.energy * 0.37);
	let wrapped = abs(fract(frag.uv.x - travel + 0.5) - 0.5);
	let pulse = exp(-wrapped * wrapped * 980.0) * select(0.0, 1.0, age < 176.0);
	let alpha = body * (0.105 + frag.selected * 0.30 + pulse * 0.72);
	return vec4f(frag.color * (0.32 + pulse * 0.74), alpha);
}

// A self-emitted arrival trail for each real receipt member. It only exists
// while the wafer is entering the chamber; it is not an always-on decorative
// network.
@vertex
fn vs_arrival_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RibbonOut {
	let shard = shards[ii];
	let reveal = reveal_for(shard);
	let previous = max(0.0, reveal - 0.13);
	let a = project(mix(vec3f(0.0), shard.position_size.xyz, previous));
	let b = project(mix(vec3f(0.0), shard.position_size.xyz, reveal));
	let delta = b.screen - a.screen;
	let distance = max(length(delta), 0.0001);
	let normal = vec2f(-delta.y, delta.x) / distance;
	let q = QUAD[vi];
	let t = q.x * 0.5 + 0.5;
	var out: RibbonOut;
	out.clip = vec4f(mix(a.screen, b.screen, t) + normal * q.y * 0.006, 0.0, 1.0);
	out.uv = vec2f(t, q.y);
	out.color = shard.color.rgb;
	out.energy = reveal * (1.0 - smoothstep(0.92, 1.0, reveal));
	out.selected = shard.metrics.w;
	return out;
}

@fragment
fn fs_arrival_trail(frag: RibbonOut) -> @location(0) vec4f {
	let body = smoothstep(1.0, 0.18, abs(frag.uv.y));
	return vec4f(frag.color * (0.32 + frag.selected * 0.44), body * frag.energy * 0.58);
}

struct CoreOut { @builtin(position) clip: vec4f, @location(0) local: vec2f };

@vertex
fn vs_core(@builtin(vertex_index) vi: u32) -> CoreOut {
	let q = QUAD[vi];
	let lens = cursor_lens();
	let center = project(vec3f(0.08, -0.02, 0.12));
	let tilt = -0.14 + lens.x * 0.035;
	let c = cos(tilt);
	let s = sin(tilt);
	let rotated = vec2f(q.x * c - q.y * s, q.x * s + q.y * c);
	var out: CoreOut;
	// The receipt is held inside a black archival spine. The slates form its
	// strata; this sealed volume replaces the generic graph's central node.
	out.clip = vec4f(center.screen + rotated * vec2f(0.265, 0.80) * (0.74 + center.scale), 0.0, 1.0);
	out.local = q;
	return out;
}

@fragment
fn fs_core(frag: CoreOut) -> @location(0) vec4f {
	let diagonal = abs(frag.local.x) * 0.83 + frag.local.y * 0.10;
	if (max(abs(frag.local.y), diagonal) > 1.0) { discard; }
	let edge = smoothstep(0.79, 0.98, max(abs(frag.local.y), abs(diagonal)));
	let aperture = smoothstep(0.105, 0.022, abs(frag.local.x + frag.local.y * 0.08));
	let stratum = smoothstep(0.035, 0.004, abs(fract((frag.local.y + 1.0) * 2.9) - 0.5));
	let jade = vec3f(0.25, 0.43, 0.33);
	let amber = vec3f(0.43, 0.28, 0.14);
	var color = vec3f(0.008, 0.016, 0.016);
	color = color + vec3f(0.026, 0.051, 0.045) * (1.0 - abs(frag.local.x)) * 0.75;
	color = color + jade * edge * 0.24;
	color = color + jade * stratum * 0.11;
	color = color + amber * aperture * 0.34;
	return vec4f(color, 0.94);
}
`;function o(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function s(e){return[`retrieved`,`path`,`mutation`,`suppressed`].indexOf(e.role)}function c(t){return t.suppressed?e(`#ab5a51`):t.mutated?e(`#c58a4a`):t.role===`path`?e(`#5faf8a`):e(`#e5e2d8`)}function l(e,t,n){let r=1/Math.max(1.1,Math.min(5.4,2.82-n));return{x:e*1.18*r,y:t*1.62*r,scale:r}}var u=class{engine;scene=null;resources=null;bindLayout=null;waferPipeline=null;filamentPipeline=null;arrivalPipeline=null;corePipeline=null;shardCount=0;filamentCount=0;selectedId=null;playhead=1;replayStart=-1;shardData=new Float32Array(1024);hitTargets=[];constructor(e,t){this.engine=e,this.uploadScene(t)}uploadScene(e){this.scene=e,this.selectedId=this.scene.shards[0]?.id??null,this.playhead=1,this.replayStart=-1;let t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.writeScene(t))}setSelected(e){this.selectedId=e;let t=this.engine.gpuDevice;t&&this.resources&&(this.writeScene(t),this.engine.requestRender())}setPlayhead(e){this.playhead=o(e),this.writeState(),this.engine.requestRender()}replay(){this.replayStart=this.engine.demoClock.state.frame,this.writeState(),this.engine.requestRender()}targetFrameRate(e){if(e<154)return 60;if(this.replayStart>=0){let t=this.engine.demoClock.framesPerLoop;if((e-this.replayStart+t)%t<196)return 60}return 6}ensurePipelines(e){if(this.waferPipeline||!this.engine.paramsBuffer)return;let t=e.createShaderModule({label:`witness-volume-wgsl`,code:a});this.bindLayout=e.createBindGroupLayout({label:`witness-volume-layout`,entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:`read-only-storage`}},{binding:3,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});let n=e.createPipelineLayout({label:`witness-volume-pipeline-layout`,bindGroupLayouts:[this.bindLayout]}),r={color:{srcFactor:`src-alpha`,dstFactor:`one-minus-src-alpha`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}},i=(i,a,o)=>e.createRenderPipeline({label:i,layout:n,vertex:{module:t,entryPoint:a},fragment:{module:t,entryPoint:o,targets:[{format:this.engine.sceneFormat,blend:r}]},primitive:{topology:`triangle-list`,cullMode:`none`}});this.filamentPipeline=i(`witness-volume-filaments`,`vs_filament`,`fs_filament`),this.arrivalPipeline=i(`witness-volume-arrival-trails`,`vs_arrival_trail`,`fs_arrival_trail`),this.waferPipeline=i(`witness-volume-evidence-wafers`,`vs_wafer`,`fs_wafer`),this.corePipeline=i(`witness-volume-receipt-core`,`vs_core`,`fs_core`)}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;let t=e.createBuffer({label:`witness-volume-shards`,size:4096,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:`witness-volume-filaments`,size:3072,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),r=e.createBuffer({label:`witness-volume-state`,size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),i=e.createBindGroup({label:`witness-volume-bind-group`,layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:n}},{binding:3,resource:{buffer:r}}]});this.resources={shardBuffer:t,filamentBuffer:n,stateBuffer:r,bindGroup:i}}writeState(){let e=this.engine.gpuDevice;if(!e||!this.resources)return;let t=this.scene?.shards.findIndex(e=>e.id===this.selectedId)??-1;e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.playhead,this.replayStart,t,this.shardCount]))}writeScene(e){if(!this.resources||!this.scene)return;let a=this.scene.shards.slice(0,t);this.shardCount=a.length,this.shardData.fill(0),this.hitTargets=[];for(let e=0;e<a.length;e+=1){let t=a[e],n=o(t.traceTime);a.length<=1||t.order/(a.length-1);let i=e%2==0?-1:1,u=(t.order*.61803398875%1-.5)*.22,d=.08+(o(t.activation)-.5)*1.42+i*.19+u,f=1.16-n*2.36+i*.075,p=-.3+n*1.18+o(t.retention)*.15-(t.suppressed?.2:0),m=c(t),h=+(t.id===this.selectedId),g=e*r;this.shardData.set([d,f,p,.92+t.retention*.36,o(t.activation),o(t.retention),o(n),h,m[0],m[1],m[2],1,s(t),+!!t.suppressed,e/Math.max(1,a.length-1),+!!t.mutated],g);let _=l(d,f,p);this.hitTargets.push({shard:t,x:_.x,y:_.y,radius:(.29+t.retention*.14)*_.scale*2.25})}e.queue.writeBuffer(this.resources.shardBuffer,0,this.shardData);let u=new Float32Array(768),d=0;for(let e of this.scene.edges){if(d>=n||e.sourceIndex<0||e.targetIndex<0)break;u.set([e.sourceIndex,e.targetIndex,0,0,o(e.weight),d*.137,1,0],d*i),d+=1}this.filamentCount=d,e.queue.writeBuffer(this.resources.filamentBuffer,0,u),this.engine.params[2]=this.shardCount,this.engine.params[3]=this.filamentCount,this.writeState()}render(e){this.resources&&this.waferPipeline&&this.filamentPipeline&&this.arrivalPipeline&&this.corePipeline&&(e.setBindGroup(0,this.resources.bindGroup),this.filamentCount>0&&(e.setPipeline(this.filamentPipeline),e.draw(6,this.filamentCount)),this.shardCount>0&&(e.setPipeline(this.arrivalPipeline),e.draw(6,this.shardCount),e.setPipeline(this.waferPipeline),e.draw(18,this.shardCount)),e.setPipeline(this.corePipeline),e.draw(6,1))}pickAt(e,t){let n=null,r=1/0;for(let i of this.hitTargets){let a=i.x-e,o=i.y-t,s=Math.hypot(a,o);s<=i.radius&&s<r&&(n=i,r=s)}return n?{id:n.shard.id,kind:`witness-shard`,payload:n.shard}:null}dispose(){this.resources?.shardBuffer.destroy(),this.resources?.filamentBuffer.destroy(),this.resources?.stateBuffer.destroy(),this.resources=null}};function d(e){return Math.max(0,Math.min(1,Number.isFinite(e)?e:0))}function f(e){return{kind:`memory`,id:e}}function p(e,t,n){return{kind:`trace`,id:`${e??`none`}:${t}:${n.type}`}}function m(e){if(!e)return[];let t=[...e.retrieved,...e.mutations.map(e=>e.id),...e.suppressed.map(e=>e.id)],n=new Set(t),r=/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,i=e.activation_path.filter(e=>n.has(e)||r.test(e));return[...new Set([...i,...t].filter(Boolean))]}function h(e,t){return t.suppressed.some(t=>t.id===e)?`suppressed`:t.mutations.some(t=>t.id===e)?`mutation`:t.activation_path.includes(e)?`path`:`retrieved`}function g(e,t){for(let n=t.length-1;n>=0;--n){let r=t[n];if(r.type===`memory.retrieve`&&typeof r.activation[e]==`number`)return d(r.activation[e])}return .48}function _(e,t){if(!t?.content)return`memory ${e.slice(0,10)}`;let n=t.content.replace(/\s+/g,` `).trim();return n.length>84?`${n.slice(0,81)}...`:n}function v(e,t){if(!t.length)return 1;let n=t.findIndex(t=>t.type===`memory.retrieve`?t.ids.includes(e):t.type===`memory.suppress`||t.type===`memory.write`?t.id===e:t.type===`contradiction.detected`?t.ids.includes(e):t.type===`sanhedrin.veto`?t.evidenceIds.includes(e):t.type===`dream.patch`&&t.proposalIds.includes(e));return n<0?1:d((n+1)/t.length)}function y(e,t,n){let r=e?.runId??null;if(!t)return{organ:`witness`,nodes:[],edges:[],events:[],receipts:[],scalars:{eventCount:e?.events.length??0,evidenceCount:0},alive:!1,runId:r,receiptId:null,shards:[],eventCount:e?.events.length??0};let i=e?.events??[],a=m(t).slice(0,64).map((e,r)=>{let a=n.get(e),o=h(e,t);return{id:e,label:_(e,a),content:a?.content??``,role:o,activation:g(e,i),retention:d(a?.retentionStrength??.5),traceTime:v(e,i),order:r,suppressed:o===`suppressed`,mutated:o===`mutation`,provenance:f(e)}}),o=new Map(a.map((e,t)=>[e.id,t])),s=a.map((e,n)=>({source:e.provenance,index:n,label:e.label,retention:e.retention,activation:e.activation,trust:t.trust_floor,suppression:+!!e.suppressed,tags:[e.role],type:`witness-shard`})),c=t.activation_path.slice(1).flatMap((e,n)=>{let r=t.activation_path[n],i=o.get(r),a=o.get(e);return i===void 0||a===void 0?[]:[{source:{kind:`receipt`,id:`${t.receipt_id}:path:${n}`},sourceIndex:i,targetIndex:a,weight:Math.max(s[i].activation??0,s[a].activation??0),kind:`receipt-path`}]});return{organ:`witness`,nodes:s,edges:c,events:i.map((e,t)=>{let n=(e.type===`memory.retrieve`?e.ids:e.type===`memory.suppress`||e.type===`memory.write`?[e.id]:e.type===`contradiction.detected`?e.ids:e.type===`sanhedrin.veto`?e.evidenceIds:e.type===`dream.patch`?e.proposalIds:[]).map(e=>o.get(e)).find(e=>e!==void 0)??-1;return{source:p(r,t,e),type:e.type,targetIndex:n,frame:18+t*24,energy:e.type===`memory.retrieve`?.86:e.type===`memory.suppress`?.72:.52}}),receipts:[{source:{kind:`receipt`,id:t.receipt_id},label:`receipt ${t.receipt_id.slice(0,12)}`,nodeIndices:a.map((e,t)=>t)}],scalars:{eventCount:i.length,evidenceCount:a.length,pathLength:c.length,trustFloor:t.trust_floor,suppressedCount:t.suppressed.length,mutationCount:t.mutations.length},alive:a.length>0,runId:r,receiptId:t.receipt_id,shards:a,eventCount:i.length}}export{m as n,u as r,y as t};
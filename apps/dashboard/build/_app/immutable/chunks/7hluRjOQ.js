var T=Object.defineProperty;var C=(r,e,t)=>e in r?T(r,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):r[e]=t;var l=(r,e,t)=>C(r,typeof e!="symbol"?e+"":e,t);import{r as b}from"./Ds4bjZVC.js";const w=64,_=96,P=16,S=8,q=`
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
`;function p(r){return Math.max(0,Math.min(1,Number.isFinite(r)?r:0))}function B(r){return["retrieved","path","mutation","suppressed"].indexOf(r.role)}function z(r){return r.suppressed?b("#ab5a51"):r.mutated?b("#c58a4a"):r.role==="path"?b("#5faf8a"):b("#e5e2d8")}function F(r,e,t){const n=1/Math.max(1.1,Math.min(5.4,2.82-t));return{x:r*1.18*n,y:e*1.62*n,scale:n}}class E{constructor(e,t){l(this,"engine");l(this,"scene",null);l(this,"resources",null);l(this,"bindLayout",null);l(this,"waferPipeline",null);l(this,"filamentPipeline",null);l(this,"arrivalPipeline",null);l(this,"corePipeline",null);l(this,"shardCount",0);l(this,"filamentCount",0);l(this,"selectedId",null);l(this,"playhead",1);l(this,"replayStart",-1);l(this,"shardData",new Float32Array(w*P));l(this,"hitTargets",[]);this.engine=e,this.uploadScene(t)}uploadScene(e){var s;this.scene=e,this.selectedId=((s=this.scene.shards[0])==null?void 0:s.id)??null,this.playhead=1,this.replayStart=-1;const t=this.engine.gpuDevice;t&&(this.ensurePipelines(t),this.ensureResources(t),this.writeScene(t))}setSelected(e){this.selectedId=e;const t=this.engine.gpuDevice;!t||!this.resources||(this.writeScene(t),this.engine.requestRender())}setPlayhead(e){this.playhead=p(e),this.writeState(),this.engine.requestRender()}replay(){this.replayStart=this.engine.demoClock.state.frame,this.writeState(),this.engine.requestRender()}targetFrameRate(e){if(e<154)return 60;if(this.replayStart>=0){const t=this.engine.demoClock.framesPerLoop;if((e-this.replayStart+t)%t<196)return 60}return 6}ensurePipelines(e){if(this.waferPipeline||!this.engine.paramsBuffer)return;const t=e.createShaderModule({label:"witness-volume-wgsl",code:q});this.bindLayout=e.createBindGroupLayout({label:"witness-volume-layout",entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});const s=e.createPipelineLayout({label:"witness-volume-pipeline-layout",bindGroupLayouts:[this.bindLayout]}),n={color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}},o=(a,h,u)=>e.createRenderPipeline({label:a,layout:s,vertex:{module:t,entryPoint:h},fragment:{module:t,entryPoint:u,targets:[{format:this.engine.sceneFormat,blend:n}]},primitive:{topology:"triangle-list",cullMode:"none"}});this.filamentPipeline=o("witness-volume-filaments","vs_filament","fs_filament"),this.arrivalPipeline=o("witness-volume-arrival-trails","vs_arrival_trail","fs_arrival_trail"),this.waferPipeline=o("witness-volume-evidence-wafers","vs_wafer","fs_wafer"),this.corePipeline=o("witness-volume-receipt-core","vs_core","fs_core")}ensureResources(e){if(this.resources||!this.bindLayout||!this.engine.paramsBuffer)return;const t=e.createBuffer({label:"witness-volume-shards",size:w*P*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s=e.createBuffer({label:"witness-volume-filaments",size:_*S*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),n=e.createBuffer({label:"witness-volume-state",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),o=e.createBindGroup({label:"witness-volume-bind-group",layout:this.bindLayout,entries:[{binding:0,resource:{buffer:this.engine.paramsBuffer}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:n}}]});this.resources={shardBuffer:t,filamentBuffer:s,stateBuffer:n,bindGroup:o}}writeState(){var s;const e=this.engine.gpuDevice;if(!e||!this.resources)return;const t=((s=this.scene)==null?void 0:s.shards.findIndex(n=>n.id===this.selectedId))??-1;e.queue.writeBuffer(this.resources.stateBuffer,0,new Float32Array([this.playhead,this.replayStart,t,this.shardCount]))}writeScene(e){if(!this.resources||!this.scene)return;const t=this.scene.shards.slice(0,w);this.shardCount=t.length,this.shardData.fill(0),this.hitTargets=[];for(let o=0;o<t.length;o+=1){const a=t[o],h=p(a.traceTime);t.length<=1||a.order/(t.length-1);const u=o%2===0?-1:1,m=(a.order*.61803398875%1-.5)*.22,g=.08+(p(a.activation)-.5)*1.42+u*.19+m,y=1.16-h*2.36+u*.075,x=-.3+h*1.18+p(a.retention)*.15-(a.suppressed?.2:0),i=z(a),c=a.id===this.selectedId?1:0,d=o*P;this.shardData.set([g,y,x,.92+a.retention*.36,p(a.activation),p(a.retention),p(h),c,i[0],i[1],i[2],1,B(a),a.suppressed?1:0,o/Math.max(1,t.length-1),a.mutated?1:0],d);const f=F(g,y,x);this.hitTargets.push({shard:a,x:f.x,y:f.y,radius:(.29+a.retention*.14)*f.scale*2.25})}e.queue.writeBuffer(this.resources.shardBuffer,0,this.shardData);const s=new Float32Array(_*S);let n=0;for(const o of this.scene.edges){if(n>=_||o.sourceIndex<0||o.targetIndex<0)break;s.set([o.sourceIndex,o.targetIndex,0,0,p(o.weight),n*.137,1,0],n*S),n+=1}this.filamentCount=n,e.queue.writeBuffer(this.resources.filamentBuffer,0,s),this.engine.params[2]=this.shardCount,this.engine.params[3]=this.filamentCount,this.writeState()}render(e){!this.resources||!this.waferPipeline||!this.filamentPipeline||!this.arrivalPipeline||!this.corePipeline||(e.setBindGroup(0,this.resources.bindGroup),this.filamentCount>0&&(e.setPipeline(this.filamentPipeline),e.draw(6,this.filamentCount)),this.shardCount>0&&(e.setPipeline(this.arrivalPipeline),e.draw(6,this.shardCount),e.setPipeline(this.waferPipeline),e.draw(18,this.shardCount)),e.setPipeline(this.corePipeline),e.draw(6,1))}pickAt(e,t){let s=null,n=1/0;for(const o of this.hitTargets){const a=o.x-e,h=o.y-t,u=Math.hypot(a,h);u<=o.radius&&u<n&&(s=o,n=u)}return s?{id:s.shard.id,kind:"witness-shard",payload:s.shard}:null}dispose(){var e,t,s;(e=this.resources)==null||e.shardBuffer.destroy(),(t=this.resources)==null||t.filamentBuffer.destroy(),(s=this.resources)==null||s.stateBuffer.destroy(),this.resources=null}}function I(r){return Math.max(0,Math.min(1,Number.isFinite(r)?r:0))}function O(r){return{kind:"memory",id:r}}function R(r,e,t){return{kind:"trace",id:`${r??"none"}:${e}:${t.type}`}}function A(r){return[...r.activation_path,...r.retrieved,...r.mutations.map(e=>e.id),...r.suppressed.map(e=>e.id)].filter((e,t,s)=>!!e&&s.indexOf(e)===t)}function U(r,e){return e.suppressed.some(t=>t.id===r)?"suppressed":e.mutations.some(t=>t.id===r)?"mutation":e.activation_path.includes(r)?"path":"retrieved"}function G(r,e){for(let t=e.length-1;t>=0;t-=1){const s=e[t];if(s.type==="memory.retrieve"&&typeof s.activation[r]=="number")return I(s.activation[r])}return .48}function j(r,e){if(!(e!=null&&e.content))return`memory ${r.slice(0,10)}`;const t=e.content.replace(/\s+/g," ").trim();return t.length>84?`${t.slice(0,81)}...`:t}function L(r,e){if(!e.length)return 1;const t=e.findIndex(s=>s.type==="memory.retrieve"?s.ids.includes(r):s.type==="memory.suppress"||s.type==="memory.write"?s.id===r:s.type==="contradiction.detected"?s.ids.includes(r):s.type==="sanhedrin.veto"?s.evidenceIds.includes(r):s.type==="dream.patch"?s.proposalIds.includes(r):!1);return t<0?1:I((t+1)/e.length)}function k(r,e,t){const s=(r==null?void 0:r.runId)??null;if(!e)return{organ:"witness",nodes:[],edges:[],events:[],receipts:[],scalars:{eventCount:(r==null?void 0:r.events.length)??0,evidenceCount:0},alive:!1,runId:s,receiptId:null,shards:[],eventCount:(r==null?void 0:r.events.length)??0};const n=(r==null?void 0:r.events)??[],a=A(e).slice(0,64).map((i,c)=>{const d=t.get(i),f=U(i,e);return{id:i,label:j(i,d),content:(d==null?void 0:d.content)??"",role:f,activation:G(i,n),retention:I((d==null?void 0:d.retentionStrength)??.5),traceTime:L(i,n),order:c,suppressed:f==="suppressed",mutated:f==="mutation",provenance:O(i)}}),h=new Map(a.map((i,c)=>[i.id,c])),u=a.map((i,c)=>({source:i.provenance,index:c,label:i.label,retention:i.retention,activation:i.activation,trust:e.trust_floor,suppression:i.suppressed?1:0,tags:[i.role],type:"witness-shard"})),m=e.activation_path.slice(1).flatMap((i,c)=>{const d=e.activation_path[c],f=h.get(d),v=h.get(i);return f===void 0||v===void 0?[]:[{source:{kind:"receipt",id:`${e.receipt_id}:path:${c}`},sourceIndex:f,targetIndex:v,weight:Math.max(u[f].activation??0,u[v].activation??0),kind:"receipt-path"}]}),g=n.map((i,c)=>{const f=(i.type==="memory.retrieve"?i.ids:i.type==="memory.suppress"||i.type==="memory.write"?[i.id]:i.type==="contradiction.detected"?i.ids:i.type==="sanhedrin.veto"?i.evidenceIds:i.type==="dream.patch"?i.proposalIds:[]).map(v=>h.get(v)).find(v=>v!==void 0)??-1;return{source:R(s,c,i),type:i.type,targetIndex:f,frame:18+c*24,energy:i.type==="memory.retrieve"?.86:i.type==="memory.suppress"?.72:.52}}),y=[{source:{kind:"receipt",id:e.receipt_id},label:`receipt ${e.receipt_id.slice(0,12)}`,nodeIndices:a.map((i,c)=>c)}];return{organ:"witness",nodes:u,edges:m,events:g,receipts:y,scalars:{eventCount:n.length,evidenceCount:a.length,pathLength:m.length,trustFloor:e.trust_floor,suppressedCount:e.suppressed.length,mutationCount:e.mutations.length},alive:a.length>0,runId:s,receiptId:e.receipt_id,shards:a,eventCount:n.length}}export{E as W,k as b};

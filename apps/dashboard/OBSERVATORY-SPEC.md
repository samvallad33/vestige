# Vestige Cognitive Observatory WebGPU Implementation Spec

Status: RESEARCH -> SPEC ONLY. Do not implement app code from this file until Sam explicitly asks.

Goal: add a new full-bleed Cognitive Observatory route that reuses the existing graph/cinema/effects substrate, then layers a GPU-resident WebGPU simulation path for viral demo moments. Priority #1 is `?demo=recall-path`: a deterministic loop where recall lights a causal path through real memory nodes.

Ground truth read before this spec:
- `apps/dashboard/src/lib/components/Graph3D.svelte` currently owns the Three/WebGL scene, 60fps governor, `ForceSimulation`, `NodeManager`, `EdgeManager`, `ParticleSystem`, `EffectManager`, DreamMode, post-processing, and event mapping.
- `apps/dashboard/src/lib/graph/force-sim.ts` is a 101-line CPU simulation with O(N^2) repulsion, edge attraction, damping, centering, and frame-count cooldown.
- `apps/dashboard/src/lib/graph/scene.ts` is Three/WebGL `WebGLRenderer` + `EffectComposer`, not WebGPU.
- `apps/dashboard/src/lib/graph/particles.ts` and `effects.ts` still use CPU-updated `THREE.BufferGeometry` attributes and `Math.random()` bursts.
- `apps/dashboard/src/lib/graph/cinema/pathfinder.ts` is already deterministic and produces story beats/flow edges. Reuse this for recall paths.
- `apps/dashboard/src/lib/graph/cinema/director.ts` is renderer-agnostic camera choreography. Reuse its path semantics, not its renderer.
- `apps/dashboard/src/routes/(app)/graph/+page.svelte` has protected `MemoryCinema`; do not modify it. Add a separate route.

## 1. Exact WebGPU technique chosen per demo moment

### Shared baseline: GPU-resident node/particle state

Chosen technique: WebGPU Samples compute-boids ping-pong storage-buffer update + instanced sprite render.

Why: it is the cleanest canonical WebGPU pattern: node/particle state is in GPU buffers created with `GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE`; a compute pass writes next state; a render pass reads the current/next state as an instance vertex buffer and draws one tiny sprite mesh per node/particle. The sample uses `@compute @workgroup_size(64)` and dispatches `Math.ceil(numParticles / 64)` workgroups.

Sources:
- https://webgpu.github.io/webgpu-samples/samples/computeBoids/
- https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/main.ts
- https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/updateSprites.wgsl
- https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/sprite.wgsl
- https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html
- https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html

Concrete pattern to use:
- `NodeState` storage buffer, 16-byte aligned lanes:
  - `pos_radius: vec4f` = xyz + visual radius
  - `vel_retention: vec4f` = xyz velocity + FSRS retention
  - `color_flags: vec4f` = rgb + packed visual flags as f32 for MVP; move to u32 later if needed
  - `demo: vec4f` = path phase, birth phase, ripple phase, shock phase
- Two ping-pong `GPUBuffer`s for simulation: previous read-only, next read-write.
- One static `EdgeIndex` storage buffer: `array<vec2u>` source/target indices.
- One `PathStep` storage buffer for demo path: `array<vec4u>` with source index, target index, beat frame, kind.
- One uniform buffer per frame: frame index, fixed time, loop phase, node count, edge count, path count, viewport, camera matrices.
- Compute workgroup size: start at 64 for broad compatibility; allow 128 only after profiling.
- Every WGSL entry checks `if (id.x >= params.nodeCount) { return; }` because dispatch is rounded up.
- CPU writes graph data once on route load, then only writes uniforms and demo controls. No per-frame readback.

### Moment A: `?demo=recall-path` — PATH lighting through nodes

Chosen technique: GPU path-wave scalar + edge glow render from the `PathStep` buffer.

Implementation:
- Use existing `planCinemaPath(nodes, edges, centerId, maxBeats)` to get deterministic beats and `flowEdges`.
- Convert each beat/edge to node indices and upload to `PathStep`.
- Compute pass `recallPathPass` runs per node. For each node, scan the small path buffer (max 7-12 beats) and compute:
  - `beatT = smoothstep(beatFrame - 18, beatFrame + 18, frameInLoop)`
  - `decayT = 1.0 - smoothstep(beatFrame + 45, beatFrame + 180, frameInLoop)`
  - path intensity = max of beat arrival envelope for matching node index.
- Edge render reads source/target node states and path steps. A second `PathEdgeInstance` buffer is optional for MVP; simpler first build draws only path edges as instanced line quads using the `PathStep` source/target index.
- Color: cyan-white core with violet afterglow. Node `demo.x` = recall intensity. Edge alpha = wavefront envelope.
- Render order: background -> non-path edges -> nodes -> path edges additive -> path node halos additive -> DOM overlay.

Sources:
- WebGPU boids storage/render pattern above.
- `apps/dashboard/src/lib/graph/cinema/pathfinder.ts` for deterministic story path and flow edges.
- GraphWaGu edge vertex shader pattern, where the vertex shader fetches node positions from a storage buffer by source/target edge indices: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/edge_vert.wgsl

### Moment B: node BORN — particle convergence into a node

Chosen technique: GPU particle attractor convergence with deterministic seed, not CPU `Math.random()` bursts.

Implementation:
- Add a `BirthParticle` storage buffer sized for e.g. 16k particles shared across all births.
- For each birth event or demo beat, assign a deterministic particle slice: start positions are generated from `hash(seed, particleIndex, nodeIndex)` on GPU or precomputed once on CPU with the same seeded PRNG.
- Compute pass lerps particle position from shell/noise field toward target node position:
  - `p = mix(start, target + curlNoiseOffset, easeOutCubic(t))`
  - Fade from violet dust to node color; size shrinks on arrival.
- For MVP, render as instanced billboards. Later upgrade to screen-space point splatting if density gets high.

Sources:
- WebGPU Samples compute-boids for GPU particle update/render.
- Particle Life WebGPU for finite-radius particle update discipline and atomics when interactions become local-neighborhood based: https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html
- Codrops WebGPU fluids on particle simulation and screen-space point-splatting as the real-time surface path: https://tympanus.net/codrops/2025/01/29/particles-progress-and-perseverance-a-journey-into-webgpu-fluids/

### Moment C: backward RIPPLE — radial distortion moving from effect to cause

Chosen technique: screen-space radial distortion post pass + node intensity ring.

Implementation:
- Add a fullscreen post-process pipeline after node/edge render.
- Uniform `Ripple { originClip: vec2f, radius: f32, width: f32, strength: f32, direction: f32 }`.
- Fragment shader samples the scene texture with UV displaced along radial direction:
  - `d = distance(uv, originClip)`
  - `ring = smoothstep(radius + width, radius, d) * smoothstep(radius - width, radius, d)`
  - `uv2 = uv + normalize(uv - originClip) * ring * strength * direction`
- For backward causal recall, `direction = -1` and radius contracts from the failure node toward the cause node, while `PathStep` beats are ordered effect -> cause.

Sources:
- WebGPU Fundamentals render/post basics: https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html
- Codrops Shader.se article for scroll/timeline-driven WebGPU scene transitions and shader masking: https://tympanus.net/codrops/2026/05/19/80s-business-tech-seamless-scene-transitions-inside-shader-ses-scroll-driven-webgpu-pipeline/
- Existing `apps/dashboard/src/lib/graph/effects.ts` already has CPU `createRippleWave`; reuse semantics, move rendering to post shader for Observatory.

### Moment D: DRIFT toward a horizon — depth fog and horizon attractor

Chosen technique: GPU drift vector + depth/alpha fog in vertex/fragment shader.

Implementation:
- Add per-node `lifecycle` scalar from retention/state: active=near, dormant=mid, silent=far, unavailable=horizon.
- Compute pass applies a weak drift acceleration toward `horizon = vec3(0, -20, -220)` for low-retention nodes; high-retention nodes resist drift.
- Vertex shader writes depth/fog factor from camera-space z and lifecycle.
- Fragment shader mixes node color toward deep blue/black and reduces alpha with `smoothstep(fogNear, fogFar, depth)`.

Sources:
- Existing `scene.ts` uses Three `FogExp2`; Observatory should port the semantic to WGSL rather than rely on Three.
- Codrops False Earth for large WebGPU procedural world using compute shaders + indirect drawing; use as proof that compute-generated scene density belongs on GPU: https://tympanus.net/codrops/2026/04/21/false-earth-from-webgl-limits-to-a-webgpu-driven-world/

### Moment E: crimson SHOCKWAVE + membrane — firewall / immune response

Chosen technique: expanding instanced ring/membrane mesh + fullscreen chromatic shock pass.

Implementation:
- Keep an `EventPulse` uniform/storage ring buffer with origin, start frame, color, type.
- Node pass reads pulses and adds red rim lighting when `abs(distance(worldPos, origin) - radius) < width`.
- Membrane pass draws a camera-facing ring/quad sphere impostor with fresnel alpha, crimson core, black outer edge.
- Post pass applies a one-frame high-strength radial distortion and red channel bias when shockfront crosses screen-space pixel.

Sources:
- Existing `effects.ts:createShockwave` uses a CPU Three `RingGeometry`; reuse the visual grammar, move to WebGPU instanced membrane.
- Codrops WebGPU fluid articles for atomics/storage buffers/compute simulation made practical in browser: https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/
- compute.toys gallery for WebGPU compute-shader idioms and GPU toy iteration: https://compute.toys/ and https://github.com/compute-toys/compute.toys

### GPU force-directed graph layout

Chosen technique for v1: GPU force layout with storage buffers, edge-local spring pass, approximate repulsion pass, and integrate pass. Start with exact O(N^2) repulsion for <=2k Observatory demo nodes; add Barnes-Hut / spatial bins before 10k+.

Why: existing `force-sim.ts` is O(N^2) CPU. The right migration is not to immediately rebuild all GraphWaGu complexity; it is to preserve Vestige's force semantics on GPU with the boids ping-pong pattern, then graduate repulsion acceleration once visual requirements exceed a few thousand nodes.

Required compute passes:
1. `clearForcesPass`: one invocation per node, zero `force.xyz`.
2. `edgeSpringPass`: one invocation per edge. Read source/target node positions, compute attraction, accumulate into source/target force. MVP avoids float atomics by writing one force contribution per edge endpoint to `EdgeForce` and reducing per node in pass 3; advanced path uses atomic fixed-point accumulation.
3. `repulsionPass`: one invocation per node; loop all nodes for MVP. For 10k+, replace with GraphWaGu-style Barnes-Hut/Morton tree or Particle Life spatial bins.
4. `integratePass`: one invocation per node; apply centering, damping, max velocity, retention/lifecycle drift, and ping-pong write to next node buffer.
5. `boundsPass` optional: compute bounds for fit camera; do not block v1 on GPU readback.

What moves from `force-sim.ts`:
- `positions: Map<string, THREE.Vector3>` becomes CPU index map + GPU `NodeState` buffers.
- `velocities: Map<string, THREE.Vector3>` becomes `vel_retention.xyz` in `NodeState`.
- `repulsionStrength`, `attractionStrength`, `dampening`, `alpha`, `maxSteps` become uniform fields.
- `tick(edges)` becomes command encoding of the compute passes above.
- `addNode/removeNode/reset` become buffer rebuilds for v1; optimize incremental updates later.

Sources:
- GraphWaGu repo: https://github.com/harp-lab/GraphWaGu
- GraphWaGu force-directed TS with compute pipelines/buffers: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/webgpu/force_directed.ts
- GraphWaGu exact force WGSL: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/compute_forces.wgsl
- GraphWaGu Barnes-Hut force WGSL: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/compute_forcesBH.wgsl
- GraphWaGu apply-forces WGSL: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/apply_forces.wgsl
- GraphWaGu Morton/tree/radix sort: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/create_tree.wgsl and https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/radix_sort.wgsl
- cosmos.gl / Cosmograph GPU force graph: https://github.com/cosmograph-org/cosmos and https://cosmograph.app/
- cosmos.gl README says computations and drawing occur on GPU and targets hundreds of thousands of points/links: https://raw.githubusercontent.com/cosmograph-org/cosmos/main/README.md

## 2. Precise files to add/change

Add docs/spec only in this run:
- `apps/dashboard/OBSERVATORY-SPEC.md` — this file.

Implementation files for the Qwen coder later:

Create:
- `apps/dashboard/src/routes/(app)/observatory/+page.svelte`
  - New route only. Do not touch protected `graph/+page.svelte` or `MemoryCinema` wiring.
  - Fetch graph data via existing `api.graph` pattern from graph route.
  - Parse `?demo=` and `?seed=` with `URLSearchParams`.
  - Render full-bleed `ObservatoryCanvas` plus DOM overlay.
- `apps/dashboard/src/lib/components/ObservatoryCanvas.svelte`
  - Svelte wrapper with `onMount`/`onDestroy`, canvas bind, ResizeObserver, WebGPU lifecycle.
- `apps/dashboard/src/lib/observatory/engine.ts`
  - Owns adapter/device/context, pipelines, buffers, render loop, resize, dispose.
- `apps/dashboard/src/lib/observatory/types.ts`
  - CPU-side `ObservatoryNode`, `ObservatoryEdge`, `DemoMode`, `DemoClock`, buffer layout constants.
- `apps/dashboard/src/lib/observatory/demo-clock.ts`
  - Deterministic fixed-step clock, seeded PRNG, loop-frame math.
- `apps/dashboard/src/lib/observatory/graph-upload.ts`
  - Convert `GraphNode[]`/`GraphEdge[]` to stable node index map, typed arrays, path steps from `planCinemaPath`.
- `apps/dashboard/src/lib/observatory/overlays/TelemetryStrip.svelte`
- `apps/dashboard/src/lib/observatory/overlays/CommandSurface.svelte`
- `apps/dashboard/src/lib/observatory/overlays/TimelineSpine.svelte`
- `apps/dashboard/src/lib/observatory/overlays/InspectorPanel.svelte`
- `apps/dashboard/src/lib/observatory/shaders/simulate.wgsl.ts`
- `apps/dashboard/src/lib/observatory/shaders/render-nodes.wgsl.ts`
- `apps/dashboard/src/lib/observatory/shaders/render-edges.wgsl.ts`
- `apps/dashboard/src/lib/observatory/shaders/render-path.wgsl.ts`
- `apps/dashboard/src/lib/observatory/shaders/post-ripple-shock.wgsl.ts`
- `apps/dashboard/src/lib/observatory/__tests__/demo-clock.test.ts`
- `apps/dashboard/src/lib/observatory/__tests__/graph-upload.test.ts`

Extend, do not rewrite:
- `apps/dashboard/src/lib/graph/cinema/pathfinder.ts`
  - Export or reuse `planCinemaPath` as-is. If extra metadata is needed, add small pure helper `pathToIndexSteps(path, nodeIndexById)` in observatory layer, not inside Cinema.
- `apps/dashboard/src/lib/graph/nodes.ts`
  - Reuse `getNodeColor`, `getMemoryState`, `AHAGRAPH_COLORS`. No renderer logic added here.
- `apps/dashboard/src/lib/graph/effects.ts`
  - Do not move existing Graph3D effects. Use it as visual reference only.
- `apps/dashboard/src/lib/graph/force-sim.ts`
  - Leave CPU simulation intact for Graph3D. Add no WebGPU imports here. Observatory gets its own GPU sim module.

Do not change:
- `apps/dashboard/src/routes/(app)/graph/+page.svelte` protected `MemoryCinema` block.
- `apps/dashboard/src/lib/components/MemoryCinema.svelte` unless Sam explicitly asks.
- `crates/`, `Cargo.toml`, backend APIs.

## 3. `?demo=recall-path` priority build instructions

### 3.1 Route/component structure

1. Create `routes/(app)/observatory/+page.svelte`.
2. On mount:
   - Parse `demo = new URLSearchParams(window.location.search).get('demo') ?? 'recall-path'`.
   - Parse `seed = ...get('seed') ?? 'vestige-observatory-v1'`.
   - Fetch graph with `api.graph({ max_nodes: 300, depth: 3, sort: 'recent' })` for MVP.
   - Pass `nodes`, `edges`, `centerId`, `demo`, `seed` into `<ObservatoryCanvas />`.
3. Render route as a full viewport stage:
   - parent `class="fixed inset-0 overflow-hidden bg-[#03040a]"`.
   - canvas layer absolute inset-0.
   - overlay layer absolute inset-0 pointer-events-none.
   - specific interactive controls use `pointer-events-auto`.

### 3.2 Deterministic demo clock

Create `demo-clock.ts`:
- `const FPS = 60`.
- `const LOOP_SECONDS = 12` for recall-path; `LOOP_FRAMES = 720`.
- `frame` is an integer, not derived from `Date.now()`.
- In normal interactive mode, rAF accumulates elapsed time and advances fixed steps. In demo mode, each rendered frame advances exactly one fixed frame, or uses `?frame=N` for capture.
- `phase = (frame % LOOP_FRAMES) / LOOP_FRAMES`.
- Use a tiny deterministic PRNG (`xmur3` seed hash + `mulberry32`) in local code. Do not add a dependency for seedrandom unless needed.
- Ban `Math.random()` and `performance.now()` from simulation state. `performance.now()` may schedule frames only; it must not decide positions/colors/path.

Source rationale:
- Fixed timestep pattern: https://gafferongames.com/post/fix_your_timestep/
- URL query parsing: https://developer.mozilla.org/en-US/docs/Web/API/URLSearchParams
- Svelte lifecycle: https://svelte.dev/docs/svelte/lifecycle-hooks

### 3.3 Graph upload

Create `graph-upload.ts`:
1. Stable-sort nodes for deterministic indices:
   - center node first if `id === centerId`.
   - then descending `updatedAt || createdAt`.
   - tie-break by `id.localeCompare`.
2. Build `nodeIndexById: Map<string, number>`.
3. Create `Float32Array nodeData` with 16 floats per node for alignment and future-proofing:
   - 0..3: x, y, z, radius
   - 4..7: vx, vy, vz, retention
   - 8..11: r, g, b, flags
   - 12..15: pathIntensity, birthPhase, ripplePhase, shockPhase
4. Initial positions:
   - Reuse the golden-angle sphere logic from `NodeManager.createNodes`, but replace randomness with deterministic seed.
   - Radius scales from node count: `baseR = clamp(24 + sqrt(n) * 1.2, 35, 140)`.
5. Create `Uint32Array edgeData` with 2 u32 per edge, skipping edges whose node IDs are absent.
6. Call `planCinemaPath(nodes, edges, centerId, 8)` and convert `flowEdges` to `PathStep[]`.
7. Path timing:
   - beat 0 at frame 60.
   - each next beat +60 frames.
   - final 180 frames reserved for afterglow and loop reset.

### 3.4 WebGPU engine setup

Create `engine.ts`:
1. Feature detect:
   - `if (!navigator.gpu) throw new Error('WebGPU unavailable')` and render a DOM fallback panel.
2. Request adapter/device.
3. Configure canvas:
   - `format = navigator.gpu.getPreferredCanvasFormat()`.
   - On resize, set `canvas.width = Math.min(clientWidth * devicePixelRatio, maxTextureDimension2D)` and same for height.
   - Reconfigure context and recreate depth/scene textures.
4. Create buffers:
   - `nodeA`, `nodeB`: `STORAGE | VERTEX | COPY_DST`.
   - `edges`: `STORAGE | COPY_DST`.
   - `pathSteps`: `STORAGE | COPY_DST`.
   - `params`: `UNIFORM | COPY_DST`.
   - sprite vertex buffer for a 2-triangle quad or 3-vertex triangle impostor.
5. Create pipelines:
   - `simulatePipeline` with read nodeA/write nodeB.
   - `renderEdgesPipeline` reads node current + edges.
   - `renderNodesPipeline` uses instance index to fetch node state and draw billboard.
   - `renderPathPipeline` draws path edges + additive halos.
   - `postPipeline` optional in increment 5.
6. Frame loop command order for recall-path MVP:
   - write params uniform for integer frame.
   - compute simulate: nodeA -> nodeB.
   - render scene using nodeB.
   - swap nodeA/nodeB.

### 3.5 WGSL simulation for recall path

MVP `simulate.wgsl.ts`:
- Inputs: params, previous nodes, next nodes, edges, path steps.
- For each node:
  - start with previous `pos`, `vel`.
  - apply low-cost centering and edge spring only after the first MVP render works; first render may use static positions.
  - compute path intensity by scanning `pathSteps`.
  - compute horizon drift from retention.
  - write next node state.

Pseudo-WGSL shape for Qwen:

```wgsl
struct NodeState {
  pos_radius: vec4f,
  vel_retention: vec4f,
  color_flags: vec4f,
  demo: vec4f,
};
struct Params {
  frame: u32,
  loopFrames: u32,
  nodeCount: u32,
  edgeCount: u32,
  pathCount: u32,
  dt: f32,
  time: f32,
  _pad: f32,
};
struct PathStep { source: u32, target: u32, beatFrame: u32, kind: u32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prevNodes: array<NodeState>;
@group(0) @binding(2) var<storage, read_write> nextNodes: array<NodeState>;
@group(0) @binding(3) var<storage, read> path: array<PathStep>;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn smoothPulse(frame: f32, beat: f32, attack: f32, release: f32) -> f32 {
  let a = smoothstep(beat - attack, beat, frame);
  let r = 1.0 - smoothstep(beat, beat + release, frame);
  return a * r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.nodeCount) { return; }
  var node = prevNodes[i];
  let f = f32(params.frame % params.loopFrames);
  var recall = 0.0;
  for (var p = 0u; p < params.pathCount; p = p + 1u) {
    let step = path[p];
    if (step.source == i || step.target == i) {
      recall = max(recall, smoothPulse(f, f32(step.beatFrame), 18.0, 150.0));
    }
  }
  node.demo.x = recall;
  nextNodes[i] = node;
}
```

### 3.6 Render approach

Nodes:
- Use instanced billboards. Vertex shader fetches `NodeState` by `@builtin(instance_index)`.
- Billboard size = `radius * (1.0 + recallIntensity * 2.0)`.
- Fragment shader radial alpha: `alpha = smoothstep(1.0, 0.0, length(localUv))`.
- Color = `mix(baseColor, vec3(0.8, 0.95, 1.0), recallIntensity)`.
- Additive glow pass can be same shader with bigger billboard and low alpha.

Edges:
- MVP: draw edges as line-list if available; better: instanced thin quads from source to target in screen space.
- For path edges, draw a separate additive path pass from `PathStep` only, with wavefront alpha.
- GraphWaGu source pattern fetches nodes in vertex shader by edge endpoint index; use that for WebGPU: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/edge_vert.wgsl

Path lighting:
- At beat frame, target node blooms first.
- During frames `beatFrame..beatFrame+45`, edge from source to target gets a traveling pulse:
  - `edgeT = saturate((frame - beatFrame) / 45)`.
  - Fragment local coordinate along edge `u`; alpha peak at `abs(u - edgeT) < 0.08`.
- Previous path nodes keep afterglow for 2-3 seconds.

### 3.7 DOM overlay

Overlay zones:
- Top telemetry strip: demo mode, seed, node/edge count, frame, fps estimate.
- Left command surface: demo selector buttons (`recall-path`, `birth`, `ripple`, `drift`, `firewall`) and copy demo URL.
- Bottom timeline spine: 720-frame loop with beat tick marks and current frame cursor.
- Right inspector: selected beat/node summary.

CSS pattern:
- Root stage: `position: fixed; inset: 0; isolation: isolate; overflow: hidden;`.
- Canvas: `position:absolute; inset:0; width:100%; height:100%; display:block; z-index:0;`.
- Overlay: `position:absolute; inset:0; z-index:10; pointer-events:none;`.
- Interactive overlay children: `pointer-events:auto;`.
- Avoid KPI cards. These are instruments: thin glass, monospaced labels, tick marks, command rail.

Sources:
- MDN `pointer-events`: https://developer.mozilla.org/en-US/docs/Web/CSS/pointer-events
- WebGPU canvas resizing: https://webgpufundamentals.org/webgpu/lessons/webgpu-resizing-the-canvas.html
- Svelte lifecycle hooks: https://svelte.dev/docs/svelte/lifecycle-hooks

## 4. Build task-list with verification increments

### Increment 1 — route shell and deterministic URL contract

Build:
- Add `routes/(app)/observatory/+page.svelte`.
- Add minimal overlay shell and fetch graph data.
- Parse `?demo=recall-path&seed=...` and display them.

Verify:
- Run `pnpm --filter @vestige/dashboard check`.
- Open `/observatory?demo=recall-path&seed=test`.
- See full-bleed dark stage, top telemetry strip, no KPI cards, no console errors.

### Increment 2 — deterministic clock tests

Build:
- Add `demo-clock.ts` and tests.
- Fixed 60fps loop, 720-frame period, seeded PRNG.

Verify:
- `pnpm --filter @vestige/dashboard test -- demo-clock`
- Same seed produces identical first 100 random values and frame phases.
- Different seed changes generated positions.

### Increment 3 — WebGPU canvas boots and clears

Build:
- Add `ObservatoryCanvas.svelte` and `engine.ts` minimal WebGPU init.
- Canvas clears to near-black and resizes with DPR clamp.

Verify:
- `/observatory?demo=recall-path` displays WebGPU canvas.
- Resize browser; canvas remains sharp and fills screen.
- If WebGPU unavailable, route shows a readable fallback instead of crashing.

### Increment 4 — upload graph and render static nodes

Build:
- Add `graph-upload.ts` and tests.
- Convert graph nodes into stable indexed `Float32Array`.
- Render node billboards from storage buffer; no simulation yet.

Verify:
- Test stable node ordering and edge filtering.
- Browser shows 100-300 nodes in deterministic positions.
- Reload same seed: identical node positions.

### Increment 5 — recall path buffer + node glow

Build:
- Reuse `planCinemaPath` to create path steps.
- Add compute pass that writes `demo.x` recall intensity.
- Node shader blooms active path nodes.

Verify:
- `/observatory?demo=recall-path&seed=A` loops every 12s.
- The same sequence lights the same nodes after reload.
- `?seed=B` changes layout but path timing remains stable.

### Increment 6 — path edge wavefront

Build:
- Add render-path pipeline drawing additive edge wave traveling source -> target.
- Timeline spine shows beat ticks matching wave arrival.

Verify:
- Wavefront visibly travels along each path edge.
- At beat arrival, the target node blooms.
- Loop reset is seamless: final afterglow fades before frame 0.

### Increment 7 — low-cost GPU force sim

Build:
- Add static edge attraction + centering + damping in compute.
- Keep repulsion disabled or low-count O(N^2) only for <=500 nodes.

Verify:
- Nodes subtly settle without CPU `ForceSimulation`.
- Performance remains smooth at 300 nodes.
- No CPU per-frame node position loops.

### Increment 8 — lifecycle effects one by one

Build/verify separately:
- Born convergence particles: trigger in demo and verify deterministic convergence.
- Backward ripple post pass: verify radial distortion travels effect -> cause.
- Horizon drift/fog: silent/unavailable nodes visibly drift/fade toward horizon.
- Firewall shockwave/membrane: crimson ring expands and fades without breaking path demo.

### Increment 9 — capture mode

Build:
- Add `?capture=1` behavior: hide cursor, freeze random seed, optional `?frame=N` exact-frame render.
- Add overlay copy button for canonical demo URL.

Verify:
- Same URL + frame renders identical screenshot.
- 12-second screen recording loops without discontinuity.

### Increment 10 — performance guardrails

Build:
- Add telemetry: CPU frame ms estimate, GPU timestamp query only if supported, node count, buffer mode.
- Cap default demo to 300 real nodes; support synthetic 10k only behind `?stress=10000`.

Verify:
- No `mapAsync` in the frame loop except delayed optional profiling.
- 300-node demo holds 60fps on Sam's M1 Max.
- Stress mode degrades gracefully and can be disabled by removing query param.

## 5. Source index

WebGPU core patterns:
- WebGPU Samples compute boids demo: https://webgpu.github.io/webgpu-samples/samples/computeBoids/
- compute-boids main TS: https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/main.ts
- compute-boids update WGSL: https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/updateSprites.wgsl
- compute-boids sprite WGSL: https://raw.githubusercontent.com/webgpu/webgpu-samples/main/sample/computeBoids/sprite.wgsl
- WebGPU storage buffers: https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html
- WebGPU compute shaders: https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html
- WebGPU canvas resize: https://webgpufundamentals.org/webgpu/lessons/webgpu-resizing-the-canvas.html
- MDN WebGPU API: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API

Particle/lifecycle shader references:
- Particle Life WebGPU article: https://lisyarus.github.io/blog/posts/particle-life-simulation-in-browser-using-webgpu.html
- Particle Life demo: https://lisyarus.github.io/webgpu/particle-life.html
- compute.toys: https://compute.toys/
- compute.toys repo: https://github.com/compute-toys/compute.toys
- Codrops WebGPU fluids: https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/
- Codrops WebGPU particles/fluids: https://tympanus.net/codrops/2025/01/29/particles-progress-and-perseverance-a-journey-into-webgpu-fluids/
- Codrops Shader.se WebGPU transitions: https://tympanus.net/codrops/2026/05/19/80s-business-tech-seamless-scene-transitions-inside-shader-ses-scroll-driven-webgpu-pipeline/
- Codrops False Earth WebGPU world: https://tympanus.net/codrops/2026/04/21/false-earth-from-webgl-limits-to-a-webgpu-driven-world/

GPU graph layout:
- GraphWaGu repo: https://github.com/harp-lab/GraphWaGu
- GraphWaGu force-directed TS: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/webgpu/force_directed.ts
- GraphWaGu exact forces: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/compute_forces.wgsl
- GraphWaGu Barnes-Hut forces: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/compute_forcesBH.wgsl
- GraphWaGu apply forces: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/apply_forces.wgsl
- GraphWaGu tree build: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/create_tree.wgsl
- GraphWaGu radix sort: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/radix_sort.wgsl
- GraphWaGu node vertex: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/node_vert.wgsl
- GraphWaGu edge vertex: https://raw.githubusercontent.com/harp-lab/GraphWaGu/main/src/wgsl/edge_vert.wgsl
- cosmos.gl repo: https://github.com/cosmograph-org/cosmos
- cosmos.gl README: https://raw.githubusercontent.com/cosmograph-org/cosmos/main/README.md
- Cosmograph product: https://cosmograph.app/

Svelte/DOM/determinism:
- Svelte lifecycle hooks: https://svelte.dev/docs/svelte/lifecycle-hooks
- MDN URLSearchParams: https://developer.mozilla.org/en-US/docs/Web/API/URLSearchParams
- MDN pointer-events: https://developer.mozilla.org/en-US/docs/Web/CSS/pointer-events
- Fixed timestep: https://gafferongames.com/post/fix_your_timestep/
- seedrandom reference if a dependency is later preferred over local PRNG: https://github.com/davidbau/seedrandom

## 6. Non-negotiable implementation constraints

- New Observatory route only. Do not mutate the protected Memory Cinema block in `graph/+page.svelte`.
- Keep existing Three/WebGL `Graph3D` intact. Observatory is a parallel WebGPU renderer, not a replacement in v1.
- Do not add dependencies for v1. Use browser WebGPU, Svelte, TypeScript, and existing `three` only if needed for matrix math; prefer small local math helpers.
- No `Math.random()` in demo-visible state. No `Date.now()`/`performance.now()` deciding simulation state.
- No GPU readback in the frame loop. Profiling readback must be delayed and optional.
- Every increment must render something verifiable before moving to the next effect.

---

## 7. VISUAL DNA — "from another dimension" (MANDATORY aesthetic contract)

Sam's directive (Jul 3 2026): the Observatory must be **mind-boggling** and look like it
**came from another dimension** — the same grammar as the WebGPU waitlist hero and
`docs/launch/causal-brain-demo.html`. This is NOT a dashboard skin. It is a living
tissue. Every increment below inherits this section. Generic = rejected.

### 7.1 The signature palette (fuse two systems — do not invent a third)

TWO real palettes already exist in-repo. The Observatory FUSES them:

- **Meaning layer — FSRS state (from `lib/graph/nodes.ts`, use verbatim):**
  - active `#10b981` emerald · dormant `#f59e0b` amber · silent `#8b5cf6` violet · unavailable `#6b7280` slate
  - aha `#FFD700` · confusion/failure `#EF4444` · guardrail `#9CA3AF`
- **Transcendence layer — iridescent thin-film (from `causal-brain-demo.html` `spectral(w)`):**
  a catmull blend around 4 anchors, w∈[0,1]:
  - indigo `vec3(0.20,0.28,0.95)` → cyan-teal `vec3(0.20,0.85,0.90)` → mint `vec3(0.45,1.00,0.72)` → magenta rim `vec3(0.85,0.45,1.00)`
  - CORE reads indigo/teal (living tissue); MOTION shifts toward cyan/magenta; NEVER muddy orange.
- **Void:** background `#05060a` (near-black, `color-scheme: dark`). No grey dashboard chrome.
- **Rule:** a node's BASE hue = its FSRS state color. Its ACTIVATION glow (recall, birth,
  ignite) rides the thin-film spectral band. So resting graph = meaningful; active graph = otherworldly.

Port `spectral(w)` to WGSL exactly (4-anchor catmull-ish blend). This is the single most
important visual primitive — it is what makes it look alien instead of corporate.

### 7.2 Glow / bloom language (copy the demo's exact values, then push further in WebGPU)

- Additive radial glow blobs (`glow(x,y,rad,col,alpha)` in the demo) → in WebGPU, a real
  additive bloom pass. Node halos are `0 0 24px` soft; ignite events drop-shadow at
  `0 0 26px rgba(110,240,220,.45)` energy.
- Text/instrument overlay glow: captions `text-shadow:0 0 24px rgba(30,180,255,.35)`;
  verdict headline `linear-gradient(90deg,#7fe6c0,#6ef0e6,#a6dcff)` clipped to text with
  `drop-shadow(0 0 26px rgba(110,240,220,.45))`; wordmark `letter-spacing:.28em`, `#5dcaa5`.
- **Breath:** a global `pulse = 0.5 + 0.5*sin(t*0.002)` (~0.32Hz) modulates halo intensity so
  the whole field breathes even at rest. Living, never static.

### 7.3 DOM = instrument overlay ONLY (already the spec rule — here is the exact style)

Copy the demo's overlay contract: `.layer { position:fixed; inset:0; pointer-events:none }`
over a full-bleed `inset:0` canvas. Instruments (TelemetryStrip, CommandSurface,
TimelineSpine, InspectorPanel) are the ONLY DOM, styled like the demo's `.cap.mono`
(SF Mono, `#cfe9ff`), `.verdict` (radial-gradient vignette card), `.wordmark`. NO KPI cards.
NO solid panels. Instruments float, glow faintly, and never block the canvas
(`pointer-events:auto` only on the specific interactive control).

### 7.4 The reference moment already exists in 2D — port it, don't reinvent

`causal-brain-demo.html` IS Moment C (salience-rescue / backward-trace) as a 2D canvas
proof: brain point cloud (two lobes, center-dense), a failure flare on the deep right, a
vector-search stall on confounders, then a BACKWARD arrow tracing to the dormant cause on
the deep-left which **ignites**, then a verdict card ("Vestige 60% · vector search 0%"),
then the VESTIGE wordmark. Beat clock: brain forms 0-1.5s, fail fades in ~0.3s, vector
stalls 4-6.5s, backward trace fires 6.5-10.5s, verdict 10.5-13s, signature 13-15s, loop.
The Observatory's `?demo=salience-rescue` must reproduce this exact narrative in real
WebGPU with real memory nodes. Study its `spectral`, `seedBrain`, `glow`, and beat
`schedule` before writing the WGSL.

### 7.5 Non-negotiable "another dimension" checklist (verifier MUST confirm each)

- [ ] Field breathes at rest (global pulse), never a static image.
- [ ] Node base hue = FSRS state; activation = thin-film spectral. Both visible.
- [ ] Real additive bloom on ignite/recall (not just opacity).
- [ ] Void `#05060a`, zero grey dashboard chrome, zero KPI cards.
- [ ] Instruments are glowing SF-Mono overlays with `pointer-events:none` except controls.
- [ ] `?demo=salience-rescue` reproduces the causal-brain backward-trace narrative.
- [ ] It looks like living tissue from another dimension, not a SaaS dashboard. If a
      reviewer would call it "clean" or "professional," it FAILED. The word is "alive."

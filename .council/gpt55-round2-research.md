# DESIGN COUNCIL — GPT-5.5 Round 2 Research

Date: 2026-07-08 PDT
Prepared for: Claude Opus 4.8 / Sam
Scope: D1 metaball membranes, D2 eight-stage execution hero, D3 retrograde axon firing.

Evidence boundary: source-backed WebGPU/WGSL facts are cited inline. The concrete organ recipes and WGSL are engineering synthesis for Vestige’s existing raw-WebGPU Observatory/RouteStage direction.

Repo anchors verified before writing:

- Existing Graph `NodeState` is 16 floats / 64 bytes in `apps/dashboard/src/lib/observatory/types.ts:38-50`.
- Existing `PathStep` kinds include `backwardCause = 1` in `apps/dashboard/src/lib/observatory/types.ts:77-88`.
- `DeepReferenceCompleted` currently arms contradiction/firewall or causal recall in `apps/dashboard/src/lib/observatory/live-bridge.ts:243-274`.
- `BackfillFired` / `CausalReceipt` already accept `path_ids` / `causal_path` and arm causal recall in `apps/dashboard/src/lib/observatory/live-bridge.ts:276-293`.
- Route hero concepts already require real data binding and no frame-loop GPU readback in `apps/dashboard/ROUTE-WEBGPU-HERO-CONCEPTS.md:5-11`.

## Sources used / access dates

- W3C WebGPU Candidate Recommendation Draft, 23 June 2026 — `GPUComputePipeline`, render passes, blend state, `GPUQuerySet`, timestamp queries, storage texture limits. Accessed 2026-07-08. https://www.w3.org/TR/webgpu/
- W3C WGSL Candidate Recommendation Draft, generated/updated June 2026 — storage texture types, read/write storage textures, compute/fragment entry points, WGSL memory/data-race model. Accessed 2026-07-08. https://www.w3.org/TR/WGSL/
- WebGPU Fundamentals, Transparency and Blending — additive blend settings and WebGPU color target blend shape. Accessed 2026-07-08. https://webgpufundamentals.org/webgpu/lessons/webgpu-transparency.html
- WebGPU Fundamentals, Optimization / Timing Performance — adapter feature detection for `timestamp-query` and timing helper pattern. Accessed 2026-07-08. https://webgpufundamentals.org/webgpu/lessons/webgpu-optimization.html
- MDN `GPUQuerySet`, last modified 2025-06-18 — timestamp query sets on render/compute passes and `timestamp-query` feature requirement. Accessed 2026-07-08. https://developer.mozilla.org/en-US/docs/Web/API/GPUQuerySet
- Chrome Developers, “What’s New in WebGPU (Chrome 147-148),” last updated 2026-04-22 — 2026 WebGPU status items including WGSL `linear_indexing`, Linux NVIDIA expansion, and the ongoing platform surface. Accessed 2026-07-08. https://developer.chrome.com/blog/new-in-webgpu-147-148
- Apple WWDC25 “Unlock GPU computing with WebGPU” — WebGPU resource model, compute/vertex/fragment programs, storage textures/buffers, and render bundles on Apple platforms. Accessed 2026-07-08. https://developer.apple.com/videos/play/wwdc2025/236/

---

## D1 — 2.5D metaball membranes on half-res `rg16float`, no float atomics

### Recommendation

Use render-pass additive splatting into a half-resolution `rg16float` field, not compute atomics.

The key move: avoid “many cells write one pixel” in compute. Instead, draw one instanced quad per cell/edge into an offscreen half-res texture with fixed-function additive blending:

```ts
blend: {
  color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' }
}
```

That is source-backed by WebGPU’s color target blend state and the standard additive-blending recipe documented by WebGPU Fundamentals. The W3C spec also validates the color-target blend-state path; WGSL storage textures exist, but this effect does not need storage-texture atomics.

### Texture layout

Use one half-res field texture per route stage:

- `fieldA`: `rg16float`, usage `RENDER_ATTACHMENT | TEXTURE_BINDING`.
- `fieldB`: `rg16float`, usage `RENDER_ATTACHMENT | TEXTURE_BINDING`.
- optional `edgeOut`: `rgba16float` or route HDR scene target if the membrane pass writes directly into the shared scene texture.

Channel meaning:

- Duplicates:
  - `R = positive cluster density`.
  - `G = trust / merge confidence / suggested-winner pull`.
- Contradictions:
  - `R = side A signed truth plate density`.
  - `G = side B signed truth plate density`.
  - seam strength is derived from `min(R, G)` and opposing-gradient energy.

Why `rg16float`: two channels are enough, half precision is enough for soft fields, bandwidth is low, and it maps cleanly to Duplicates and Contradictions. For half-res 1920×1080, the field is 960×540×2×2 bytes ≈ 2.1 MB per texture; ping-pong blur is cheap.

### Pipeline

1. Clear `fieldA` to `(0,0)`.
2. Accumulate splats with an instanced render pass:
   - one quad per memory/cell for ordinary density;
   - one elongated capsule/quad per duplicate link or contradiction pair if the neck/seam must be physically connected;
   - additive blend into `fieldA`.
3. Separable blur:
   - horizontal pass `fieldA -> fieldB`;
   - vertical pass `fieldB -> fieldA`;
   - 5-tap or 9-tap Gaussian, route quality switchable.
4. Fullscreen membrane pass samples `fieldA`:
   - density = duplicate `R`, or contradiction `abs(R-G)` / seam `min(R,G)`;
   - gradient = finite difference over neighboring texels;
   - edge = threshold band around iso-level;
   - membrane normal = normalized gradient;
   - thickness = function of trust/similarity/conflict;
   - write HDR edge/luciferin/magenta/scarlet into the RouteStage scene texture.
5. Composite into existing blackwater base/post chain.

### Data mapping

Duplicates:

- cell radius: `mix(10px, 34px, member.retention)` or winner confidence.
- cluster gravity/neck: similarity controls capsule width and iso-threshold.
- neck thickness: `smoothstep(0.78, 0.98, similarity)`.
- trust membrane: `G` stores suggested winner confidence / cluster agreement.

Contradictions:

- side A splats into `R`, side B splats into `G`.
- fault seam is strongest where both fields are present but gradients oppose.
- trust delta offsets the two plates vertically in the route scene model; the membrane seam remains 2.5D screen-space for cheap readability.

### Accumulate splat WGSL sketch

Render an instanced quad per cell/link. The vertex shader expands in field-pixel space; the fragment shader emits additive density. No atomics, no storage writes.

```wgsl
struct Params {
  field_size: vec2f,
  viewport_size: vec2f,
  iso: f32,
  time: f32,
};

struct Cell {
  // xy in normalized route-stage coordinates [0,1]
  pos_radius: vec4f,     // x,y,r_px, kind
  signal: vec4f,         // similarity, trust, polarity, selected
  color_flags: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f,
  @location(1) signal: vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
  let c = cells[iid];
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
  );
  let q = corners[vid];
  let r = c.pos_radius.z;
  let field_px = c.pos_radius.xy * params.field_size + q * r;
  let ndc = field_px / params.field_size * 2.0 - vec2f(1.0, 1.0);

  var out: VSOut;
  out.pos = vec4f(ndc.x, -ndc.y, 0.0, 1.0);
  out.local = q;
  out.signal = c.signal;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec2f {
  let d2 = dot(in.local, in.local);
  if (d2 > 1.0) { discard; }

  // Compact Gaussian-like kernel. Similarity tightens duplicate necks.
  let similarity = clamp(in.signal.x, 0.0, 1.0);
  let trust = clamp(in.signal.y, 0.0, 1.0);
  let polarity = in.signal.z; // duplicates +1; contradictions side A/B encoded by CPU.
  let sigma = mix(0.55, 0.28, similarity);
  let k = exp(-d2 / max(0.001, 2.0 * sigma * sigma));

  // For contradictions, CPU submits side-A cells with polarity > 0 and side-B
  // cells with polarity < 0. Render targets are additive, so emit to one channel.
  if (polarity >= 0.0) {
    return vec2f(k, k * trust);
  }
  return vec2f(0.0, k * trust);
}
```

For contradiction side separation, prefer explicit route-side field encoding over negative values in blend targets: submit side A into `R`, side B into `G`. Signed math happens in the fullscreen pass.

### Blur WGSL sketch

Use fullscreen triangle/quad or compute. Compute is fine here because each invocation writes one pixel; there is no contention.

```wgsl
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rg16float, write>;
@group(0) @binding(2) var<uniform> blur_dir: vec2i; // (1,0) or (0,1)

const W: array<f32, 5> = array<f32, 5>(0.06136, 0.24477, 0.38774, 0.24477, 0.06136);

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(src_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  var acc = vec2f(0.0);
  for (var i: i32 = -2; i <= 2; i = i + 1) {
    let p = clamp(vec2i(gid.xy) + blur_dir * i, vec2i(0), vec2i(dims) - vec2i(1));
    acc += textureLoad(src_tex, p, 0).rg * W[u32(i + 2)];
  }
  textureStore(dst_tex, vec2i(gid.xy), vec4f(acc, 0.0, 1.0));
}
```

WGSL storage-texture support and read/write storage texture semantics are source-backed by the June-2026 WGSL spec; use write-only storage textures here for portability.

### Fullscreen membrane edge + trust-thickness WGSL

```wgsl
struct MembraneParams {
  field_size: vec2f,
  iso: f32,
  base_thickness_px: f32,
  mode: f32, // 0 duplicates, 1 contradictions
  time: f32,
  selected_boost: f32,
};

@group(0) @binding(0) var field_tex: texture_2d<f32>;
@group(0) @binding(1) var field_sampler: sampler;
@group(0) @binding(2) var<uniform> mp: MembraneParams;

fn sample_field(uv: vec2f) -> vec2f {
  return textureSampleLevel(field_tex, field_sampler, uv, 0.0).rg;
}

@fragment
fn membrane_fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let uv = frag.xy / mp.field_size;
  let px = 1.0 / mp.field_size;

  let c = sample_field(uv);
  let l = sample_field(uv - vec2f(px.x, 0.0));
  let r = sample_field(uv + vec2f(px.x, 0.0));
  let d = sample_field(uv - vec2f(0.0, px.y));
  let u = sample_field(uv + vec2f(0.0, px.y));

  var density: f32;
  var trust: f32;
  var seam: f32;

  if (mp.mode < 0.5) {
    // Duplicates: R is density, G is trust/confidence.
    density = c.r;
    trust = clamp(c.g / max(c.r, 0.001), 0.0, 1.0);
    seam = 0.0;
  } else {
    // Contradictions: R and G are opposing truth plates.
    let both = min(c.r, c.g);
    let signed = c.r - c.g;
    density = abs(signed) + both * 0.75;
    trust = clamp(max(c.r, c.g), 0.0, 1.0);

    let gx_signed = (r.r - r.g) - (l.r - l.g);
    let gy_signed = (u.r - u.g) - (d.r - d.g);
    let g_signed = length(vec2f(gx_signed, gy_signed));
    seam = smoothstep(0.02, 0.18, both) * smoothstep(0.01, 0.12, g_signed);
  }

  let gx = (r.r + r.g) - (l.r + l.g);
  let gy = (u.r + u.g) - (d.r + d.g);
  let grad = length(vec2f(gx, gy));

  // Trust thickens the membrane; weak trust makes a hairline/noisy edge.
  let thickness = (mp.base_thickness_px + trust * 5.0) * max(px.x, px.y);
  let edge = 1.0 - smoothstep(0.0, thickness, abs(density - mp.iso));
  let rim = edge * smoothstep(0.005, 0.08, grad);

  let duplicate_col = mix(vec3f(0.78, 0.45, 0.18), vec3f(0.25, 1.0, 0.55), trust);
  let contradiction_col = mix(vec3f(1.0, 0.08, 0.12), vec3f(1.0, 0.0, 0.78), seam);
  let col = select(duplicate_col, contradiction_col, mp.mode >= 0.5);

  let alpha = clamp(rim + seam * 0.85, 0.0, 1.0);
  return vec4f(col * alpha * 2.4, alpha);
}
```

### Perf expectations for 150-2000 cells

Measure with `timestamp-query` if the adapter exposes it; W3C/MDN/WebGPU Fundamentals all document timestamp-query support and feature detection. If unavailable, fall back to CPU frame timing but label it wall-time.

Budget on Sam’s M1 Max / modern desktop GPU target:

- 150 cells: one additive splat pass + 2 blur passes + membrane pass should be comfortably sub-1 ms GPU at half-res.
- 500 cells: ~1-2 ms GPU, dominated by fill-rate if radii are large.
- 2000 cells: ~2-5 ms GPU if splat quads are bounded to cluster-local radii; worst case is overdraw, not ALU.

Guardrails:

- Clamp splat radius by route zoom and similarity; never let 2000 cells each draw a 200 px quad.
- Batch all cells in one draw call per channel/mode.
- Use render-pass additive blending for accumulation; use compute only for one-thread-per-pixel blur/post.
- Keep the field half-res. Full-res doubles each dimension and quadruples blur/post cost for little membrane benefit.

---

## D2 — Honest “watch an 8-stage pipeline execute” hero pattern

### Recommendation

Do not fake streaming. Turn a one-shot `DeepReferenceCompleted` payload into a deterministic replay score.

The event arrives all at once, so the honest visual language is:

- “Computation completed; now replaying the receipt through the organ.”
- Stage chambers light only if the payload contains real output for that stage.
- Flow timing is deterministic from payload timestamps/order/indices and fixed per-stage dwell times.
- Empty stages remain visible but unlit/dormant.
- Contradictions/supersessions are interrupts that cut across the path after their real stage becomes active.

This is the same honesty principle as the existing live bridge stretching real events to human-perceptible choreography (`live-bridge.ts` already comments that tempo is stretched while edges/physics are real).

### Stage model

Use a route-local `ReasoningSceneModel`, not Graph `NodeState`:

```ts
type StageKind =
  | 'intent'
  | 'retrieve'
  | 'activate'
  | 'evidence'
  | 'contradiction'
  | 'synthesis'
  | 'recommendation'
  | 'receipt';

interface StageCell {
  stageIndex: number;
  sourceKind: 'trace' | 'receipt' | 'deep_reference';
  sourceId: string;        // event id, run id, receipt id, or memory id
  count: number;           // real output count for this stage
  confidence: number;      // real confidence/trust if available
  interrupt: 0 | 1 | 2;    // none, contradiction, supersession
}
```

Stage N lights if and only if `count > 0` or a real scalar exists. For example:

- intent: `intent` / query classification present.
- retrieve: evidence/supporting IDs present.
- activate: `activation_path` or trace activation map present.
- evidence: evidence array / supporting IDs.
- contradiction: contradiction pairs.
- synthesis: answer/recommendation body.
- recommendation: recommended memory/action.
- receipt: receipt id/run id/proof artifact.

### One-shot receipt -> staged animation

On `DeepReferenceCompleted`:

1. Normalize payload into `StageCell[8]` and `FlowPacket[]`.
2. Assign a deterministic `receiptStartFrame = currentFrame`.
3. Set `stageStart[i] = receiptStart + intro + i * dwell`, but skip or compress empty stages.
4. For each stage, set `energy = real count/confidence`, not an invented pulse.
5. Animate packets along chamber-to-chamber splines only when both source and destination stages have real output.
6. If contradiction/supersession exists, schedule an interrupt packet at the first stage where it becomes known; it cuts the spline and clamps downstream glow until the synthesis chamber absorbs it.

Human-legible timing:

- 180-240 ms chamber pre-glow.
- 450-650 ms packet travel per stage.
- 250 ms contradiction cut.
- 1.2-1.8 s final receipt hold.
- Full replay: ~6-9 s for all eight stages; reduced-motion: instant stage illumination + slow opacity transitions, no moving packets.

### Compute-updated spline advection

Represent each route as cubic Bezier control points between chamber centers. Compute updates packet `t`, stage visibility, and interrupt state; render pass draws instanced packet glyphs/ribbons.

```wgsl
struct Params {
  frame: f32,
  fps: f32,
  packet_count: u32,
  reduced_motion: u32,
};

struct Spline {
  p0: vec4f,
  p1: vec4f,
  p2: vec4f,
  p3: vec4f,
};

struct PacketIn {
  route_stage: vec4u,  // x spline index, y src stage, z dst stage, w flags
  timing: vec4f,       // start_frame, duration_frames, energy, interrupt_kind
  source: vec4u,       // source object index / ids via CPU side table
};

struct PacketOut {
  pos_energy: vec4f,   // xyz, energy/alpha
  tangent_flags: vec4f,// xy tangent, z stage_gate, w interrupt
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> splines: array<Spline>;
@group(0) @binding(2) var<storage, read> packets_in: array<PacketIn>;
@group(0) @binding(3) var<storage, read_write> packets_out: array<PacketOut>;

fn bezier(s: Spline, t: f32) -> vec3f {
  let u = 1.0 - t;
  return (u*u*u) * s.p0.xyz + (3.0*u*u*t) * s.p1.xyz + (3.0*u*t*t) * s.p2.xyz + (t*t*t) * s.p3.xyz;
}

fn bezier_tangent(s: Spline, t: f32) -> vec3f {
  let u = 1.0 - t;
  return normalize(
    3.0*u*u*(s.p1.xyz - s.p0.xyz) +
    6.0*u*t*(s.p2.xyz - s.p1.xyz) +
    3.0*t*t*(s.p3.xyz - s.p2.xyz)
  );
}

@compute @workgroup_size(64)
fn advect_packets(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.packet_count) { return; }

  let p = packets_in[i];
  let s = splines[p.route_stage.x];
  let start = p.timing.x;
  let dur = max(1.0, p.timing.y);
  var t = clamp((params.frame - start) / dur, 0.0, 1.0);

  if (params.reduced_motion != 0u) {
    t = select(0.0, 1.0, params.frame >= start);
  }

  // smootherstep for organic chamber-to-chamber motion.
  let tt = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  let pos = bezier(s, tt);
  let tan = bezier_tangent(s, tt);

  let active = f32(params.frame >= start) * f32(params.frame <= start + dur + 60.0);
  let energy = p.timing.z * active;
  let interrupt = p.timing.w;

  packets_out[i] = PacketOut(
    vec4f(pos, energy),
    vec4f(tan.xy, f32(p.route_stage.z), interrupt)
  );
}
```

### Interrupt choreography

Contradiction packet:

- color scarlet/magenta edge, not normal luciferin.
- route cuts perpendicular to the current spline tangent.
- downstream chamber membranes clamp: `stageEnergy *= 0.35` until synthesis stage.
- both contradiction endpoints pulse using their real memory IDs.

Supersession packet:

- amber/green old→new transfer.
- old evidence dims but leaves a scar rather than disappearing.
- if receipt includes `activation_path`, draw it as a side-channel filament into the synthesis chamber.

Stage lighting compute:

```wgsl
struct StageIn { metrics: vec4f; timing: vec4f; }; // count, trust, interrupt, reserved; start,dwell,...
struct StageOut { glow_gate: vec4f; }; // glow, membrane, interrupt, active

@group(0) @binding(4) var<storage, read> stages_in: array<StageIn>;
@group(0) @binding(5) var<storage, read_write> stages_out: array<StageOut>;

@compute @workgroup_size(8)
fn update_stages(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= 8u) { return; }
  let s = stages_in[i];
  let has_real_output = s.metrics.x > 0.0 || s.metrics.y > 0.0;
  let age = params.frame - s.timing.x;
  let active = f32(has_real_output) * smoothstep(0.0, 18.0, age);
  let hold = 1.0 - smoothstep(s.timing.y, s.timing.y + 60.0, age);
  let glow = active * max(0.18, s.metrics.y) * max(0.25, hold);
  stages_out[i] = StageOut(vec4f(glow, s.metrics.y, s.metrics.z, active));
}
```

### Perf notes

- Eight stages are tiny. The scalable dimension is evidence/packet count.
- 200 evidence packets with compute advection and instanced glyph rendering is trivial compared with the existing graph field.
- Keep packet output in a storage buffer; render instanced quads from it.
- No GPU readback during playback. Click picking can stay CPU-side via object map or one explicit pick readback.

---

## D3 — Retrograde backward-only axon firing along a causal path

### Recommendation

Make the signature read unmistakable: a magenta wavefront travels from the observed effect/target backward through tissue to the quiet cause, then the cause remains permanently brighter than it was.

Existing `PATH_KIND.backwardCause = 1` and `BackfillFired`/`CausalReceipt` path ingestion are the right substrate. The sharper upgrade is a route-local/field-local path segment buffer with per-segment beat timing, a signed direction flag, and a persistent cause latch.

### Data model

Seed from a real causal path array:

```ts
interface RetroPathUpload {
  pathIds: string[];       // ordered target/effect -> ... -> cause
  causeId: string;         // final path id, or explicit cause id
  effectId: string;        // first path id
  receiptId: string;
  firedAtFrame: number;
  confidence: number;
}
```

GPU buffers:

```wgsl
struct RetroSegment {
  a_b: vec4u,       // x target-side node index, y cause-side node index, z hop, w flags
  timing: vec4f,    // start_frame, duration, confidence, path_len
  color: vec4f,     // route color/scalar; magenta reserved for causality
};

struct CauseLatch {
  node_index: u32,
  start_frame: f32,
  confidence: f32,
  reserved: f32,
};
```

Important: array order is target→cause. Segment `i` fires before segment `i+1`, so the visual motion is backward-only. If the backend sends cause→target, reverse it on CPU and label the transform.

### Visual grammar

- Leading edge: hot magenta rim, narrow and fast.
- Wake: dim violet-pink afterglow along traversed tissue, fades slowly.
- Tissue displacement: small perpendicular shimmer in nearby membranes/nodes as the wave passes.
- Final cause: permanent brightening/luciferin capture; not a temporary pulse.
- Direction cue: small arrow/phase notches point from effect back to cause; no forward echo.

Timing:

- Per hop: 280-420 ms depending path length.
- Total readable target→cause travel: 1.4-3.2 s.
- Cause latch: ramps over 350 ms, then decays only to a higher baseline, never to zero during the session.

### Retrograde segment WGSL

This draws instanced ribbon quads per path segment. The fragment computes distance to the segment centerline and a traveling wavefront parameter.

```wgsl
struct Params {
  frame: f32,
  viewport: vec2f,
  reduced_motion: u32,
  path_count: u32,
};

struct NodeLite {
  pos_radius: vec4f,
  color_flags: vec4f,
};

struct RetroSegment {
  a_b: vec4u,
  timing: vec4f,
  color: vec4f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<NodeLite>;
@group(0) @binding(2) var<storage, read> segments: array<RetroSegment>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) seg_timing: vec4f,
  @location(2) conf_hop: vec2f,
};

@vertex
fn retro_vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
  let seg = segments[iid];
  let a = nodes[seg.a_b.x].pos_radius.xyz; // target-side
  let b = nodes[seg.a_b.y].pos_radius.xyz; // cause-side

  // Screen/projected-space sketch: production should use the shared camera projection.
  let dir = normalize((b - a).xy);
  let n = vec2f(-dir.y, dir.x);
  let corners = array<vec2f, 6>(
    vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0,  1.0),
    vec2f(0.0,  1.0), vec2f(1.0, -1.0), vec2f(1.0,  1.0)
  );
  let q = corners[vid];
  let width = mix(2.0, 8.0, clamp(seg.timing.z, 0.0, 1.0));
  let p = mix(a.xy, b.xy, q.x) + n * q.y * width;

  var out: VSOut;
  out.pos = vec4f(p, 0.0, 1.0); // replace with camera projection in engine
  out.uv = q;
  out.seg_timing = seg.timing;
  out.conf_hop = vec2f(seg.timing.z, f32(seg.a_b.z));
  return out;
}

@fragment
fn retro_fs(in: VSOut) -> @location(0) vec4f {
  let start = in.seg_timing.x;
  let dur = max(1.0, in.seg_timing.y);
  var head = clamp((params.frame - start) / dur, 0.0, 1.0);
  if (params.reduced_motion != 0u) {
    head = select(0.0, 1.0, params.frame >= start);
  }

  // Since segment order is target->cause and uv.x increases a->b, this is a
  // backward-only wave when a is effect-side and b is cause-side.
  let dist_to_head = abs(in.uv.x - head);
  let transverse = abs(in.uv.y);

  let rim = exp(-dist_to_head * dist_to_head * 900.0) * smoothstep(1.0, 0.15, transverse);
  let wake = smoothstep(head, head - 0.42, in.uv.x) * exp(-transverse * transverse * 2.8) * 0.22;
  let notch = step(0.92, fract((in.uv.x - head) * 18.0)) * rim * 0.35;

  let magenta = vec3f(1.0, 0.0, 0.78);
  let core = vec3f(1.0, 0.62, 0.95);
  let col = magenta * (rim + notch) * 3.2 + core * wake;
  let alpha = clamp(rim + wake + notch, 0.0, 1.0);
  return vec4f(col, alpha);
}
```

### Permanent cause brightening WGSL

The cause latch should be in node rendering or a route-local node overlay. It is not just demo lane pulse; it becomes a raised baseline for the cause node after the retrograde wave arrives.

```wgsl
struct CauseLatch {
  node_index: u32,
  start_frame: f32,
  confidence: f32,
  reserved: f32,
};

@group(2) @binding(0) var<storage, read> latches: array<CauseLatch>;
@group(2) @binding(1) var<uniform> latch_count: u32;

fn cause_capture_boost(node_index: u32, frame: f32) -> f32 {
  var boost = 0.0;
  for (var i: u32 = 0u; i < latch_count; i = i + 1u) {
    let l = latches[i];
    if (l.node_index == node_index) {
      let age = frame - l.start_frame;
      let ramp = smoothstep(0.0, 24.0, age);
      // Permanent session baseline: never decays below 35% of confidence.
      let settled = mix(1.0, 0.35, smoothstep(120.0, 360.0, age));
      boost = max(boost, ramp * settled * clamp(l.confidence, 0.2, 1.0));
    }
  }
  return boost;
}
```

Use this boost to thicken the cause node membrane, raise its luciferin/green component if the causal backfill promoted it, and keep the magenta rim as a thin outer causal signature. Do not recolor ordinary activation magenta; magenta remains exclusive to retrograde causality.

### “Reaches backward through tissue” enhancement

Add a low-cost tissue displacement field around the wavefront:

- Each retro segment writes a transient line-splat into a tiny half-res `r16float` disturbance texture using additive blending.
- Fullscreen post samples that texture and offsets background field UVs by its gradient.
- Result: tissue bends as the wave passes, then relaxes.

WGSL fragment for the distortion pass:

```wgsl
let d0 = textureSampleLevel(disturb_tex, samp, uv, 0.0).r;
let dx = textureSampleLevel(disturb_tex, samp, uv + vec2f(px.x, 0.0), 0.0).r -
         textureSampleLevel(disturb_tex, samp, uv - vec2f(px.x, 0.0), 0.0).r;
let dy = textureSampleLevel(disturb_tex, samp, uv + vec2f(0.0, px.y), 0.0).r -
         textureSampleLevel(disturb_tex, samp, uv - vec2f(0.0, px.y), 0.0).r;
let bend = normalize(vec2f(dx, dy) + vec2f(1e-4)) * d0 * 0.006;
let tissue = textureSampleLevel(scene_tex, samp, uv - bend, 0.0);
```

### Perf notes

- Path lengths are tiny compared with graph node count. Even 64 segments are cheap as instanced quads.
- Cause-latch loop should be capped or indexed; for launch, cap active latches to 32 and store them in a small uniform/storage buffer.
- The optional disturbance texture costs one half-res additive line pass and one fullscreen gradient sample; enable only in flagship Reasoning/Blackbox/Graph moments.
- Reduced-motion: no traveling head; render the completed path as a static magenta rim plus cause brightening and a text receipt.

---

## Consolidated build guidance for Round 3

1. Keep Graph `NodeState` byte-stable at 64 bytes.
2. Route organs own richer `RouteSceneModel` layouts.
3. Use render-pass additive splats for field accumulation; compute for one-writer-per-pixel blur/post only.
4. Make one-shot backend events into deterministic receipt replays, explicitly labeled by design and never pretending to stream.
5. Reserve magenta exclusively for real retrograde causal motion.
6. Ship perf instrumentation with optional `timestamp-query`; label wall-time fallback honestly.
7. Every RouteStage shares pause, reduced-motion, null-adapter fallback, and WebGPU fallback.

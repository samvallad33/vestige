# GPT-5.5 Round 1 — Vestige Cognitive Operating System

Date: 2026-07-08
Role: independent design-council research + visual-system proposal
Target: July 14 Hacker News launch dashboard

## Executive thesis

Build Vestige as a living causal-memory instrument, not an analytics UI.

The unifying language: **Causal Bioluminescent Cortex** — a full-bleed WebGPU organism where memories are not dots, but living cells in a black nutrient medium. Salience is metabolism. FSRS decay is oxygen loss. Trust is membrane thickness. RSB causal backfill is a retrograde axon firing backward through the tissue. Contradiction is an immune synapse. Suppression is scar tissue with a visible reason-code crystal. Every route is one organ of the same organism, driven by the same real cognitive substrate.

Do not make “pretty dashboard panels.” Make **a microscope inside a thinking local brain**.

The reason this is uncopyable: most dashboards can animate metrics; Vestige can animate causal learning. The OS only works because Vestige owns per-memory FSRS state, suppression semantics, contradiction pairs, causal receipts, bitemporal audit, memory PRs, reasoning receipts, trace events, and live WebSocket events. Replace those with Math.random and the viewer loses the causal arrows, immune verdicts, retention half-lives, and click receipts. It becomes visibly fake.

---

## 1. Independent frontier research — July 2026 WebGPU / dataviz edge

### 1.1 WebGPU feature surface that matters now

Sources checked directly during this round:

1. W3C WebGPU specification, current draft retrieved 2026-07-08. Feature index includes `timestamp-query`, `indirect-first-instance`, `shader-f16`, `subgroups`, `primitive-index`, `texture-component-swizzle`, and `subgroup-size-control`.
   URL: https://www.w3.org/TR/webgpu/

2. Chrome for Developers, “What’s New in WebGPU (Chrome 134),” published 2025-02-26. Subgroups are available after origin-trial work; WGSL requires `enable subgroups;`; exposed built-ins include `subgroup_invocation_id` and `subgroup_size`; functions include `subgroupAdd`, `subgroupBallot`, `subgroupBroadcast`, `subgroupShuffle`; Google Meet reported 2.3–2.9x speedups for matrix-vector multiply shaders on some devices.
   URL: https://developer.chrome.com/blog/new-in-webgpu-134

3. Chrome for Developers, “What’s New in WebGPU (Chrome 128).” Subgroups were trialed behind `Unsafe WebGPU Support` and as an origin trial from Chrome 128–131; f16 subgroup usage requires device features `subgroups`, `subgroups-f16`, and `shader-f16`, plus WGSL `enable f16, subgroups, subgroups_f16;`.
   URL: https://developer.chrome.com/blog/new-in-webgpu-128

4. Chrome for Developers, “What’s New in WebGPU (Chrome 144),” published 2026-01-07. Adds WGSL `subgroup_id` and `num_subgroups`; useful because subgroup indexing no longer needs atomic reconstruction to avoid overlapping memory accesses.
   URL: https://developer.chrome.com/blog/new-in-webgpu-144

5. MDN `GPUSupportedFeatures`, last modified 2026-05-05. Confirms `timestamp-query` measures compute/render pass time using `GPUQuerySet` and `timestampWrites`; `indirect-first-instance` allows non-zero `firstInstance` in `drawIndirect`/`drawIndexedIndirect`; `subgroups` enable SIMD-level cross-thread communication; `shader-f16` enables WGSL `f16`.
   URL: https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedFeatures

6. GPUWeb issue #5175, opened 2025-04-25. MultiDrawIndirect is still an open feature discussion; a Vulkan port cited ~500k indirect draw calls at 5ms with multi-draw versus 18ms without. Treat multi-draw as future-facing, not launch-critical.
   URL: https://github.com/gpuweb/gpuweb/issues/5175

Launch implication:
- Ship on core WebGPU + optional feature gates.
- Use `timestamp-query` only for dev/perf HUD and adaptive quality, never visual truth.
- Use subgroups only behind `adapter.features.has('subgroups')`; fallback to workgroup reductions/atomics.
- Do not require MultiDrawIndirect. Use indirect draw/dispatch where available, but keep CPU-encoded draw counts small enough for Chrome stable.

### 1.2 Compute particle systems and GPU-resident simulation

Sources:

7. Codrops, “Particles, Progress, and Perseverance: A Journey into WebGPU Fluids,” 2025-01-29. The demo uses atomics, indirect draw/dispatch, storage buffers, compute shaders, 3D textures, PBF/SPH-style particle motion, Marching Cubes on GPU, timestamp queries, and storage-buffer compaction. Specific WGSL files described: `PBF_applyForces.wgsl`, `PBF_calculateDisplacements.wgsl`, `PBF_integrateVelocity.wgsl`, `MarchCase.wgsl`, `EncodeBuffer.wgsl`.
   URL: https://tympanus.net/codrops/2025/01/29/particles-progress-and-perseverance-a-journey-into-webgpu-fluids/

8. WebGPU Fundamentals, storage-buffer lesson. Search result and known API pattern confirm storage buffers are the WebGPU workhorse for large mutable datasets; WGSL changes from `var<uniform>` to `var<storage, read>` or `var<storage, read_write>`.
   URL: https://webgpufundamentals.org/webgpu/lessons/webgpu-storage-buffers.html

Buildable pattern for Vestige:
- Keep route state GPU-resident: `array<NodeState>`, `array<EdgeIndex>`, `array<EventImpulse>`, `array<CellState>`.
- CPU writes only real deltas from API/WebSocket: new events, selected IDs, query terms, current filters, projection days.
- Compute pass sequence:
  1. `decode_impulses`: turn real events into compact GPU impulses.
  2. `simulate_cells`: integrate retention/trust/activation velocities.
  3. `diffuse_field`: low-res reaction/diffusion texture from node sources.
  4. `classify_tiles`: mark hot metaball/SDF cells and write indirect counts.
  5. Render instanced nodes, membranes, axons, text, and post-chain.

WGSL shape:

```wgsl
struct NodeState {
  pos_radius: vec4f,      // xyz, visual radius
  vel_retention: vec4f,   // xyz velocity, FSRS retention
  color_flags: vec4f,     // rgb base, packed semantic flags
  cognitive: vec4f,       // activation, trust, contradiction, suppression
  time: vec4f,            // created_days, last_access_days, stable_days, bitemporal_phase
};

struct EventImpulse {
  kind_target: vec4u,     // kind, target index, aux index, route id
  scalars: vec4f,         // energy, confidence, age, reason code
};

@group(0) @binding(0) var<storage, read_write> nodes: array<NodeState>;
@group(0) @binding(1) var<storage, read> impulses: array<EventImpulse>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(128)
fn simulate_cells(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(params.nodeCount)) { return; }
  var n = nodes[i];
  // Retention is real: R(t)=pow(1.0 + elapsed/(9.0*stability), -1.0) or existing FSRS helper output.
  // Visual state never invents retention; it only maps it.
  nodes[i] = n;
}
```

### 1.3 SDF/metaball merging and volumetric/raymarched matter

The Codrops fluid article is the strongest browser-frontier reference here because it combines particles → 3D potential field → Marching Cubes → indirect rendering. For Vestige, do not attempt full 3D Marching Cubes across every route for launch. The right adaptation is cheaper and sharper:

- Use a half/quarter-res 2D `rgba16float` or `rg16float` field texture as a cognitive substrate.
- Scatter each memory/node into the field with radius = stability/importance and intensity = activation/trust.
- Render implicit **metaball membranes** in a fullscreen pass using thresholded scalar field gradients.
- Use raymarch only for route-specific hero moments where depth matters: contradictions firewall, dreams, blackbox replay.

WGSL-level approach:
- Storage-buffer nodes -> compute scatter into texture or tiled buffer. If atomics to floats are unavailable/undesirable, accumulate into `rgba16float` by rendering additive instanced quads, then sample in fullscreen pass.
- Gradient: sample `field(x±dx,y)` / `field(x,y±dy)` to derive membrane normal.
- Membrane edge: `edge = smoothstep(t - w, t, density) - smoothstep(t, t + w, density)`.
- Trust thickness: `w = mix(0.003, 0.018, trust)`.
- Contradiction: signed dual field where incompatible memories push opposite channels; seam appears where `abs(a-b) < epsilon && max(a,b) > threshold`.

### 1.4 Reaction-diffusion / Lenia / neural cellular automata / physarum

Sources:

9. arXiv, “Neural Cellular Automata: From Cells to Pixels,” result dated 2026-05-01. The result frames NCA through self-organization, Turing reaction-diffusion, and pixel/cell growth.
   URL: https://arxiv.org/html/2506.22899v3

10. Artificial Life / MIT Press, “Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Flow Lenia,” published 2025-05-01. Continuous cellular automata and artificial-life systems are current frontier visual metaphors.
   URL: https://direct.mit.edu/artl/article/31/2/228/130572/Flow-Lenia-Emergent-Evolutionary-Dynamics-in-Mass

11. Sakana AI, “Automating the Search for Artificial Life with Foundation Models,” 2024-12-24. ASAL found novel Lenia/boids/cellular-automata patterns using foundation-model search.
   URL: https://asal.sakana.ai/

Use these as inspiration, not fake intelligence. Vestige should not claim NCA learning unless it has one. It can use a deterministic **cognitive tissue shader** where cell state is driven by real memory fields:

- `A` channel = activation concentration from recalls/searches.
- `B` channel = forgetting risk / retention debt.
- `C` channel = trust / Sanhedrin verdict pressure.
- `D` channel = immune/suppression residue.

Compute pass per cell:

```wgsl
let lapA = sample4(A, xy+neighbors) - 4.0*A;
let source = event_activation_at_cell(xy);        // real event/node source
let decayDebt = retention_debt_at_cell(xy);       // real FSRS projection
A2 = A + dt * (diffA * lapA + source - 0.08*A);
B2 = B + dt * (diffB * lapB + decayDebt - trust*A*0.03);
```

This passes the discipline test because the pattern’s sources are real nodes/events. Math.random cannot preserve why one region is inflamed, quarantined, or healing.

### 1.5 MSDF living text

Sources:

12. Red Blob Games, “SDF Fonts: Appendix,” 2026-02-01. Notes MSDF vs SDF, `msdf-atlas-gen`, `planeBounds`, `atlasBounds`, gamma correction caveats, and points to WebGPU sample `textRenderingMsdf#msdfText.wgsl` using `dFdx`/`dFdy` and `smoothstep`.
   URL: https://www.redblobgames.com/articles/sdf-fonts/appendix.html

13. WebGPU samples MSDF text WGSL referenced by Red Blob.
   URL: https://webgpu.github.io/webgpu-samples/?sample=textRenderingMsdf#msdfText.wgsl

Vestige use:
- Build a small MSDF glyph atlas for route labels, event types, reason codes, and receipt IDs.
- Text should not be HUD chrome; it should be part of the organism: labels are **etched bioluminescent scars** that appear only when backed by a selected memory, event, receipt, or contradiction.
- Use per-glyph instance buffer: `glyphRect`, `atlasRect`, `worldAnchor`, `semanticColor`, `age`, `confidence`.
- Render in scene HDR before bloom, not DOM overlays, for hero surfaces. DOM remains for accessibility/detail panes.

### 1.6 GPU interaction/shockwave feedback

Existing Vestige already has the correct architectural primitive:
- `ObservatoryEngine` has `FramePass.compute` and `FramePass.render`.
- `params` buffer has live lanes 12..15: `liveKind`, `liveFrame`, `liveEnergy`, `projectionDays`.
- `LiveBridge` maps real `VestigeEvent`s into GPU mutations: `MemorySuppressed`, `DeepReferenceCompleted`, `DreamStarted`, `DreamCompleted`, `ConnectionDiscovered`, `BackfillFired`, `CausalReceipt`, `MemoryCreated`.
- `PostChain` already renders `rgba16float` scene -> mip bloom -> Khronos PBR tonemap -> grain -> vignette.

Interaction extension:
- Add `ClickImpulseBuffer` with max 64 recent clicks. Each click carries route, target node/record id, semantic action, timestamp frame, and verified backend payload hash.
- Shockwaves are not generic ripples. They mean: “this click selected/accepted/rejected/promoted/suppressed this exact cognitive object.”
- The click writes to CPU only after the API action succeeds or the target is real in current data. No optimistic fake hero pulses for failed actions.

---

## 2. Invented unifying visual + motion language

## Name: Causal Bioluminescent Cortex

Vestige is a dark local brain in a jar. The dashboard routes are organs viewed through different instruments: cortex, hippocampus, immune system, dreamstem, flight recorder, synaptic edit queue.

This is not purple-on-black. The base is **blackwater + oil-film + enzymatic light + scarlet immune heat**.

### 2.1 Palette: exact colors and what they mean

Base medium:
- `#020307` — Blackwater void. Absolute background; never tinted purple.
- `#07100D` — Anaerobic green-black low field; nutrient medium.
- `#0B171B` — Deep cyan-black parallax fog.
- `#11140A` — Old-memory amber-black sediment.

Living memory / retention:
- `#E9FFB7` — Newly retrievable / high-retention “luciferin white.” Retention >= 0.86.
- `#A8FF5E` — Healthy stable memory. Retention 0.65–0.86.
- `#29F2A9` — Active recall / excitation wave. Activation > route-specific p90.
- `#1BD6FF` — Remote association / semantic bridge.
- `#315CFF` — Deep latent / low activation but structurally present.
- `#8A4B18` — Forgetting debt / dehydrated trace. Retention 0.25–0.45.
- `#2A160B` — Near-extinction sediment. Retention < 0.25.

Trust / verifier / immune:
- `#F4F1D0` — High trust membrane, warm ivory edge.
- `#FFD166` — Caution verdict / yellow immune flare.
- `#FF3B30` — Veto / contradiction injury.
- `#B90D2B` — Suppression scar, permanent red-black lacquer.
- `#FF7A1A` — Reversible labile suppression window.

Causality / RSB:
- `#00F5D4` — Forward recall signal.
- `#FF2DF7` — Retrograde causal backfill axon. This is the only magenta-like tone, reserved exclusively for backward causality. Never route accent.
- `#FFFFFF` — Causal receipt “click,” a one-frame proof spark at edge write.

Bitemporal / audit:
- `#6BFFB8` — Valid-time growth ring.
- `#7C6CFF` — Transaction-time shadow. Indigo allowed only as temporal parallax, not brand accent.
- `#FFB000` — Supersession amber cut-line.

System health / stats:
- `#9DFFEB` — throughput/cool system flow.
- `#FF4FD8` — backlog pressure only when queue growth is real.

### 2.2 Encoding rules

FSRS retention:
- Hue is a continuous gradient from sediment (`#2A160B`) -> amber debt (`#8A4B18`) -> green healthy (`#A8FF5E`) -> luciferin white (`#E9FFB7`).
- Radius = `sqrt(stabilityDays)`, clamped, not raw importance.
- Halo opacity = `activation`.
- Surface cracking = `1.0 - retention`.
- Motion drag = low retention moves like dust; high retention moves like elastic tissue.

Trust:
- Membrane thickness = `trust_floor` or verifier confidence.
- Membrane continuity = evidence state: complete evidence = continuous ring; missing evidence = perforated ring; appealed = ring lifts off the surface.
- Sanhedrin veto = scarlet immune cell clamps onto the claim and freezes local diffusion.

Event type:
- `MemoryCreated`: cells condense from faint nutrient noise into a bound membrane. The birth flash is small and precise, not fireworks.
- `SearchPerformed`: a cyan chemoattractant wave travels from query glyphs into matching cells.
- `ActivationSpread`: green/teal excitation runs along real activation paths.
- `ImportanceScored`: gold-white enzymes deposit on the cell wall; score becomes thickness.
- `RetentionDecayed`: amber dehydration front moves inward from membrane to nucleus.
- `ConnectionDiscovered`: a new axon grows, then the force field physically tenses.
- `DeepReferenceCompleted`: eight-stage reasoning organ lights in order; contradictions/supersessions are visible as interrupts.
- `BackfillFired`/`CausalReceipt`: magenta retrograde axon fires backward from failure to quiet cause; final node gets permanently brighter.
- `MemorySuppressed`: red-black macrophage engulfs the cell; reason-code glyph is etched into the scar.
- `MemoryPrOpened`: translucent immune proposal capsule forms around proposed mutation.
- `MemoryPrDecided`: promote/merge/supersede/quarantine/forget action resolves as surgery, not a toast.
- `TraceEvent`: blackbox route shows agent nerve impulses as exact call/retrieve/write/veto events.

Bitemporal state:
- Valid time = inner growth rings.
- Transaction time = outer shadow offset. Rewrites/supersession show rings with a cut seam; click scrubs between times.

### 2.3 Motion grammar

The OS has one motion language:

1. **Chemotaxis** — activation attracts nearby semantically connected cells. Used for recall/search/explore.
2. **Elastic axons** — edges are not lines; they are tension-bearing fibers that tighten when causal/semantic weight increases.
3. **Immune clamping** — contradictions, vetoes, suppressions create local stiffening and red macrophage behavior.
4. **Retrograde firing** — RSB is the only motion that travels backward through a path. This is the signature shareable moment.
5. **Metabolic breathing** — global breath is not purple pulsing; it is nutrient diffusion. Breathing amplitude is driven by live event energy / consolidation state.
6. **Sedimentation** — old low-retention memories drift downward/back in z; active retrieval lifts them toward the lens.
7. **Scar persistence** — suppression/contradiction effects leave low-energy persistent marks. This makes history visible.
8. **Click as incision** — every click cuts or probes tissue and produces a receipt-backed wave. No meaningless hover sparkles.

### 2.4 Why this is category-of-one

Competitors can copy particles, bloom, metaballs, or WebGPU. They cannot copy:
- backward-only causal recall wavefronts sourced from RSB receipts;
- per-memory FSRS decay as oxygen/retention state;
- suppression scars with reason codes and reversible labile windows;
- contradiction pairs as immune synapses;
- bitemporal growth rings tied to memory audit history;
- agent blackbox traces connected to retrieval receipts and memory PR mutations.

The visual language is not a skin. It is a projection of Vestige’s actual cognitive data model.

---

## 3. Per-route hero spec

Each route gets a full-bleed hero using the same shared engine. DOM panels become instrument readouts over or beside the canvas, not the main event.

### 3.1 Reasoning — “The Eight-Stage Thought Organ”

Hero concept:
- A vertical living spinal cord with eight translucent chambers: retrieval -> rerank -> activation -> trust-score -> supersession -> contradiction -> relations -> chain.
- A query enters as MSDF glyph fragments. Real evidence cells are pulled into chambers. Interrupts appear where contradiction/supersession changes the path.

Real data signal:
- `/deep_reference` result and `DeepReferenceCompleted` event.
- Evidence ids, contradiction pairs, supersession/evolution fields, confidence/trust, relation chain.

Frontier technique:
- GPU route-stage pipeline with 8 chamber SDFs in a field texture.
- Instanced evidence cells move chamber-to-chamber via compute-updated splines.
- MSDF labels etched on each chamber only when that stage has real output.

Primary click means:
- Click a chamber = inspect the exact stage receipt: inputs, outputs, discarded candidates, trust/conflict reason.
- Click an evidence cell = center graph/cinema on that memory.

Discipline test:
- Random data cannot preserve stage order, contradiction interruption, or evidence provenance.

### 3.2 Timeline — “Bitemporal Growth Rings”

Hero concept:
- The whole screen is a cut cross-section of the brain: concentric rings for valid time, offset spectral shadows for transaction time.
- Memory events are cells embedded in rings; rewrites and supersessions cut visible seams.

Real data signal:
- `/timeline`, memory audit records, created/updated/deleted/suppressed events, bitemporal audit if available.

Frontier technique:
- Polar coordinate field shader; compute bins events into time rings.
- MSDF date ticks are engraved into the ring, not DOM text.
- Interaction shockwave follows ring curvature.

Primary click means:
- Click a ring/cell = open exact time slice and show memory state then vs now.

Discipline test:
- Random cannot show transaction/valid-time divergence or supersession seams.

### 3.3 Feed — “Live Neurotransmitter Rain”

Hero concept:
- The WebSocket feed becomes a live bloodstream. Each event falls as a molecule that binds to affected tissue. Events that mutate the graph leave persistent biochemical traces.

Real data signal:
- `eventFeed` / WebSocket `VestigeEventType` variants.

Frontier technique:
- Ring-buffer `EventImpulseBuffer`; compute particles from event type to target node/route organ.
- GPU trail texture with decay per actual event age.

Primary click means:
- Click molecule = freeze event, show payload, target id, and downstream GPU effect.

Discipline test:
- Math.random cannot bind molecules to real target ids or produce correct downstream scars/edges.

### 3.4 Schedule — “Forgetting Debt Orrery”

Hero concept:
- Memories orbit review horizons like a tidal system. Items approaching decay risk fall toward an amber event horizon; scheduled consolidations are gravity wells.

Real data signal:
- Retention distribution, stability/difficulty/lastAccessed if available, scheduled consolidation/cron status, intentions due dates.

Frontier technique:
- Compute orbit simulation where orbital radius = next-review urgency and eccentricity = stability uncertainty.
- Low-retention bodies emit amber dust.

Primary click means:
- Click an orbiting memory = schedule/promote/review action preview; if action succeeds, an actual green restoration wave plays.

Discipline test:
- Random cannot reproduce FSRS urgency or real scheduled work.

### 3.5 Duplicates — “Synaptic Fusion Chamber”

Hero concept:
- Near-duplicate memories appear as cells whose membranes are already partially merged. Difference text appears as illuminated mismatch filaments between nuclei.

Real data signal:
- `/duplicates` pairs, similarity threshold, candidate ids, content deltas, merge actions.

Frontier technique:
- Metaball SDF merge visualization. Similarity controls isosurface neck thickness; differing tokens generate red/amber filaments.

Primary click means:
- Click the neck = inspect/approve merge candidate; successful merge causes two nuclei to fuse and emits a receipt ring.

Discipline test:
- Random cannot map similarity to neck geometry or content differences to filaments.

### 3.6 Contradictions — “Immune Synapse Arena”

Hero concept:
- Contradictory memories face each other across a glowing immune synapse. The winner/trust/evidence state thickens one membrane; unresolved contradictions spark scarlet arcs.

Real data signal:
- `/contradictions`, `DeepReferenceCompleted.contradiction_pairs`, `TraceEvent.contradiction.detected`, Sanhedrin claims/verdicts.

Frontier technique:
- Dual-channel signed field: memory A and B are two biological potentials; seam is where both fields meet.
- Volumetric red clamps / macrophages attach to lower-trust claim.

Primary click means:
- Click seam = open contradiction receipt and evidence comparison; choose appeal/suppress/supersede if available.

Discipline test:
- Random cannot know pair topology, trust asymmetry, or evidence states.

### 3.7 Patterns — “Cross-Project Mycelium”

Hero concept:
- Recurring patterns grow as fungal mycelium across project territories. Strong cross-project motifs form thick rhizomes; weak one-offs stay spores.

Real data signal:
- `/patterns/cross-project`, tags, projects, supporting memory ids, recurrence counts.

Frontier technique:
- Physarum-inspired trail deposition where sources are real supporting memories and trail reinforcement = recurrence/support count.
- Compute diffusion field for trail evaporation.

Primary click means:
- Click a rhizome = reveal all supporting memories/projects and causal/semantic edges.

Discipline test:
- Random trails cannot line up with support ids or project recurrence.

### 3.8 Memories — “Cellular Atlas”

Hero concept:
- A browse/search atlas of memory cells in the blackwater medium. Content is not cards first; content is tissue first, details on selection.

Real data signal:
- `/memories`, search results, suppression count, retention/retrieval strength, tags, type, audit.

Frontier technique:
- Instanced cell renderer with SDF membranes, MSDF labels at focus, GPU picking later. Launch fallback can map click to nearest CPU-known projected node.

Primary click means:
- Click a cell = open memory detail, audit rings, and actions; every action plays a receipt-backed wave only after success.

Discipline test:
- Random would not match retention/suppression/type/search ranking.

### 3.9 Explore — “Chemoattractant Probe”

Hero concept:
- The selected memory emits a query chemical. Associations, causes, contradictions, and relations reveal themselves as different attracted organisms.

Real data signal:
- `/explore` action result, fromId/toId, association paths, relation types, graph neighborhood.

Frontier technique:
- Compute force field where selected node emits route-specific chemical channels; only returned ids respond.
- Edge fibers grow along actual path order.

Primary click means:
- Click an association path = commit next exploration hop or open exact relation evidence.

Discipline test:
- Random cannot preserve returned path order/relation type.

### 3.10 Importance — “Salience Furnace”

Hero concept:
- Content enters a scoring furnace. Signal particles collide with criteria membranes; the final importance score becomes deposited gold-white enzyme on the memory wall.

Real data signal:
- `/importance` score response, `ImportanceScored` event, score components if exposed.

Frontier technique:
- Reaction chamber SDF; score components are particle lanes; final score controls deposition thickness and bloom.

Primary click means:
- Click score layer = show why the memory is important and what future recall weight changes.

Discipline test:
- Random cannot explain score/routing or match backend score.

### 3.11 Activation — “Spreading Ignition Map”

Hero concept:
- Activation is visible as a green/teal ignition wave across real memory topology. Nodes light according to actual activation values, not layout proximity alone.

Real data signal:
- `ActivationSpread` events, blackbox receipt `activation_path`, retrieved ids and activation map.

Frontier technique:
- Wavefront compute over compact edge lists; shader intensity from activation scalar.
- Optional subgroup reductions for per-cluster activation maxima.

Primary click means:
- Click a glowing path = inspect why each hop activated and whether suppression/trust modified it.

Discipline test:
- Random cannot reproduce activation path or per-id activation map.

### 3.12 Dreams — “Consolidation Storm / REM Forge”

Hero concept:
- Dream consolidation becomes a storm inside the tissue. New connections grow as lightning-fungal axons; clusters settle into a new configuration after `DreamCompleted`.

Real data signal:
- `DreamStarted`, `DreamProgress`, `DreamCompleted`, `ConnectionDiscovered`, consolidation result counts.

Frontier technique:
- Existing live dream storm + new route organ: route-specific field where edge births are axon growth, not generic particles.
- Trail texture persists newly discovered connections.

Primary click means:
- Click a new axon = show the two memories and reason/weight/type of connection.

Discipline test:
- Random cannot match new edge ids or connection counts.

### 3.13 Intentions — “Future-Tense Germline”

Hero concept:
- Intentions are dormant seeds embedded ahead of the timeline. Active intentions pulse like unborn cells; completed ones germinate into memory tissue; stale ones calcify.

Real data signal:
- `/intentions?status=active`, intention status, due/created timestamps, linked memories if present.

Frontier technique:
- Bitemporal projection field: future valid-time cells sit in front of current tissue; status controls membrane phase.

Primary click means:
- Click a seed = open intention, mark/update/attach memory; success germinates the seed.

Discipline test:
- Random cannot map status/due time/linkage.

### 3.14 Blackbox — “Agent Flight Recorder Nerve Trace”

Hero concept:
- An agent run is a nervous system trace. Tool calls are electric impulses, retrievals are green branches, suppressions are red clamps, writes are cell births, vetoes are immune gates.

Real data signal:
- `/traces`, `/traces/:runId`, `TraceEvent` variants, receipts scoped by run.

Frontier technique:
- GPU timeline ribbon: event type lanes as neural fibers, with per-event particles moving left-to-right.
- Receipt IDs become MSDF etched beads.

Primary click means:
- Click an impulse = open exact trace event and linked receipt/export.

Discipline test:
- Random cannot maintain runId event order, retrieved ids, veto evidence ids, or write diffs.

### 3.15 Memory PRs — “Cognitive Immune Review Queue”

Hero concept:
- Proposed brain mutations appear as surgical capsules held by immune cells. The queue is not a table; it is a triage chamber.

Real data signal:
- `/memory-prs`, `MemoryPrOpened`, `MemoryPrDecided`, PR kind/status/signals/action/mode.

Frontier technique:
- Metaball capsules grouped by action: promote/merge/supersede/quarantine/forget/ask_agent_why.
- Risk-gated/paranoid mode thickens immune membrane and slows motion.

Primary click means:
- Click capsule = review diff/signals; action click performs mutation and produces surgery animation only after API success.

Discipline test:
- Random cannot reflect queue status, risk mode, signals, or decision history.

### 3.16 Stats — “Metabolic Observatory”

Hero concept:
- System stats are organ vitals: memory count as biomass, edge count as vasculature, retrieval/consolidation as metabolic rate, retention distribution as oxygen histogram.

Real data signal:
- `/stats`, `/health`, retention distribution, recent event rates, release gates if exposed.

Frontier technique:
- Volumetric vital bars as living tissue columns; histogram bins become breathing alveoli.
- Timestamp-query powered perf readout in dev mode only.

Primary click means:
- Click a vital = drill into contributing memories/events/endpoints.

Discipline test:
- Random cannot match actual counts/distributions/health.

### 3.17 Settings — “Local Brain Control Room”

Hero concept:
- Settings are not forms floating over black. They are valves and membranes in the local brain: storage, privacy, model/backend, verifier hooks, MCP/tools.

Real data signal:
- Health/config endpoints, transport status, local paths sanitized, feature availability, WebGPU capability bits.

Frontier technique:
- Valve SDF controls with real status: open/closed/degraded. Adapter feature cards light when `adapter.features` includes `timestamp-query`, `subgroups`, `shader-f16`, etc.

Primary click means:
- Click a valve = change setting or inspect safety/side-effect; visual state changes only after persisted confirmation.

Discipline test:
- Random cannot mirror actual config/feature/status.

---

## 4. Shared engine extensions to build once

These are additive around `apps/dashboard/src/lib/observatory/`; do not touch protected Memory Cinema.

### 4.1 RouteStage

Path: `apps/dashboard/src/lib/observatory/RouteStage.svelte` or `route-stage/`

Responsibilities:
- Own full-bleed canvas + overlay slots.
- Boot `ObservatoryEngine` with route id and seed.
- Accept route data payload and convert it to a shared `RouteSceneModel`.
- Register route-specific `FramePass` objects.
- Wire reduced-motion/pause exactly like current ObservatoryStage.
- Use current `PostChain` unchanged.

API sketch:

```ts
export type RouteOrgan =
  | 'reasoning' | 'timeline' | 'feed' | 'schedule' | 'duplicates' | 'contradictions'
  | 'patterns' | 'memories' | 'explore' | 'importance' | 'activation' | 'dreams'
  | 'intentions' | 'blackbox' | 'memory-prs' | 'stats' | 'settings';

export interface RouteSceneModel {
  organ: RouteOrgan;
  nodes: RouteNode[];
  edges: RouteEdge[];
  events: RouteEvent[];
  receipts: RouteReceipt[];
  scalars: Record<string, number>;
}
```

### 4.2 Cognitive field texture pass

Build a reusable low-res field pass:
- `field-diffuse.wgsl.ts`
- `field-renderer.ts`
- Inputs: node/event buffers, route params.
- Outputs: `rgba16float` field texture.
- Channels: activation, forgetting debt, trust, immune residue.

Every route samples this field to get the same organism feel.

### 4.3 Click-shockwave system

Build once:
- `click-impulses.ts`
- `click-shockwave.wgsl.ts`

Contract:
- Click impulses must carry a real object id and semantic action.
- For mutating actions, impulse plays after API success.
- For inspect-only actions, impulse plays if object exists in route model.

### 4.4 MSDF text layer

Build once:
- Static atlas generated offline or checked into dashboard assets.
- `msdf-text-renderer.ts`
- `shaders/msdf-text.wgsl.ts`
- Use only for hero labels/event glyphs. DOM remains for accessible detail.

WGSL key idea from Red Blob / WebGPU sample:
- Sample `rgb` distance, take median, compute screen-space width via derivatives (`dFdx`/`dFdy` or `fwidth` equivalent), smoothstep alpha.

### 4.5 Semantic color module

Create one source of truth:
- `cognitive-palette.ts`
- Map: retention -> color, event -> impulse, trust -> membrane, action -> surgery.
- Prevent per-route accent drift.

### 4.6 Route data adapters

Each boring route gets a pure adapter first:
- `reasoning-scene.ts`
- `timeline-scene.ts`
- etc.

Adapters turn API responses into `RouteSceneModel`. This is where the discipline test is enforced: every visual primitive must point to a real id, event, scalar, or receipt.

---

## 5. Build order — jaw-drop first, shippable increments

### Increment 0 — Guardrails before beauty

Goal: prevent screensaver drift.

1. Add `cognitive-palette.ts` and `RouteSceneModel` types.
2. Add a dev-only assertion: every rendered hero primitive has `sourceKind` + `sourceId` or explicit `sourceScalar`.
3. Add `RouteStage` shell reusing `ObservatoryEngine` + `PostChain`.
4. Do not remove Graph3D yet; do not touch Memory Cinema.

Verification:
- `pnpm --filter @vestige/dashboard check`
- One route mounts RouteStage without data and shows honest empty state.

### Increment 1 — Reasoning Theater first

Why first:
- It is the demo-conversion surface for “Cognitive Operating System.”
- Eight-stage pipeline is uniquely Vestige and easy to narrate.
- Clicks have obvious meaning.

Build:
1. `reasoning-scene.ts` adapter for `/deep_reference` result.
2. Eight-stage chamber renderer.
3. Evidence cell pathing.
4. MSDF stage labels.
5. Click chamber -> exact receipt/details.

Shareable moment:
- A query enters; retrieved memories route through eight cognitive chambers; contradiction interrupts; causal chain resolves.

### Increment 2 — Blackbox + Memory PRs immune system

Why second:
- Hacker News will care that this is not fake agent gloss. Blackbox receipts prove it.
- Memory PRs show the brain edits itself safely.

Build:
1. Trace ribbon renderer from `/traces`.
2. Receipt beads and event impulse lanes.
3. Memory PR triage capsules.
4. Decision surgery shockwaves.

Shareable moment:
- “Here is an agent run. Every retrieval, suppression, write, veto, receipt, and proposed brain mutation is visible.”

### Increment 3 — Contradictions + Duplicates

Why third:
- Immune system route pair: detect incompatible beliefs, fuse duplicate cells.
- Visually dramatic and semantically defensible.

Build:
1. Dual-field immune synapse renderer for contradictions.
2. Metaball fusion chamber for duplicates.
3. Click seam/neck to inspect exact pair/diff.

Shareable moment:
- Two contradictory memories face off, trust membrane clamps one side, evidence receipt opens on click.

### Increment 4 — Timeline + Schedule + Intentions

Why fourth:
- Turns local-first memory into time cognition.

Build:
1. Bitemporal ring renderer.
2. FSRS orrery schedule.
3. Future intention seeds.

Shareable moment:
- Scrub time and see memories age, supersede, decay, and future intentions germinate.

### Increment 5 — Feed + Activation + Explore

Why fifth:
- These make the app feel constantly alive.

Build:
1. Event bloodstream feed.
2. Activation ignition wave.
3. Explore chemoattractant probe.

Shareable moment:
- Search/retrieval emits a real wave through memory topology.

### Increment 6 — Dreams + Patterns + Stats + Memories + Settings polish

Why last:
- Important, but several already have conceptual analogues or require broad adapters.

Build:
1. Dream REM forge over existing dream storm logic.
2. Patterns mycelium.
3. Stats metabolic observatory.
4. Memories cellular atlas.
5. Settings local brain control room.

---

## 6. The one thing to prototype first

Prototype: **Reasoning Theater Eight-Stage Thought Organ with real `/deep_reference` data and click receipts.**

Why this de-risks everything:
- It exercises the shared `RouteStage` shell.
- It proves route-specific data adapters.
- It requires MSDF living labels.
- It requires click shockwaves with semantic meaning.
- It forces the color/motion language to work outside the Graph route.
- It is the easiest launch demo to explain: “Watch Vestige think locally.”

Prototype acceptance test:
1. User enters a query on `/reasoning`.
2. Backend returns `/deep_reference` result.
3. Full-bleed WebGPU organ animates eight stages in exact order.
4. Evidence cells correspond to real memory ids.
5. Contradiction/supersession interrupts only appear if present in result.
6. Clicking a stage opens exact data for that stage.
7. Replacing backend values with random data visibly breaks provenance: stages no longer match ids/evidence/conflicts.

If this prototype lands, every other route is a variation of the same engine: real data -> adapter -> route organ -> click receipt.

---

## 7. Implementation notes anchored to current repo

Observed files:
- `apps/dashboard/src/lib/observatory/engine.ts`: WebGPU engine, `FramePass`, deterministic clock, `params` lanes, no per-frame readback rule.
- `apps/dashboard/src/lib/observatory/live-bridge.ts`: real WebSocket-to-GPU bridge; handles `MemorySuppressed`, `DeepReferenceCompleted`, dreams, backfill/causal receipt, birth.
- `apps/dashboard/src/lib/observatory/post/post-chain.ts`: HDR `rgba16float` scene -> mip bloom -> PBR neutral tonemap -> grain/vignette.
- `apps/dashboard/src/lib/observatory/types.ts`: `NodeState` is currently 4 vec4s / 64 bytes; `PARAMS_FLOATS = 16`; `LIVE_KIND` includes firewall, dreamStorm, causalRecall, birth.
- `apps/dashboard/src/lib/stores/api.ts`: APIs available for memories, timeline, graph, explore, importance, consolidate, retention distribution, duplicates, contradictions, patterns, audit, intentions, deepReference, Sanhedrin, traces, receipts, memory PRs.
- `apps/dashboard/src/lib/types/index.ts`: WebSocket events include `MemoryCreated`, `MemoryUpdated`, `MemorySuppressed`, `SearchPerformed`, `Dream*`, `RetentionDecayed`, `ConnectionDiscovered`, `ActivationSpread`, `ImportanceScored`, `DeepReferenceCompleted`, `BackfillFired`, `CausalReceipt`, `HookVerdictRecorded`, `TraceEvent`, `MemoryPrOpened`, `MemoryPrDecided`.
- `apps/dashboard/src/routes/(app)/graph/+page.svelte`: Memory Cinema is explicitly protected; Graph currently imports `ObservatoryStage`; default WebGPU field exists and should be reused.

Do not touch:
- `MemoryCinema.svelte` and `src/lib/graph/cinema/*`.

Replace/de-emphasize:
- Static purple panels.
- Classic `Graph3D.svelte` as primary route experience; keep one-click fallback until GPU picking exists.

---

## 8. Risks / constraints

1. Browser/device variance.
   - Mitigation: core WebGPU path first; subgroups/f16/timestamp-query optional; adaptive particle counts; DPR clamp already exists.

2. Visual truth drift.
   - Mitigation: every primitive needs `sourceKind/sourceId/sourceScalar`; click waves after API success; no random as semantic input.

3. Overbuilding full 3D fluid/Marching Cubes.
   - Mitigation: use 2D/2.5D field textures and metaball membranes for launch; reserve full voxel/Marching Cubes for one hero later.

4. Text readability.
   - Mitigation: MSDF for hero labels; DOM for detail/accessibility; reduced-motion continues to render event pulses.

5. Route count is huge.
   - Mitigation: build shared RouteStage + adapters; launch with the first 3–5 jaw-drop organs polished, then sweep remaining panels through the same shell.

---

## Final recommendation

Build the **Causal Bioluminescent Cortex**.

Prototype Reasoning Theater first. Then Blackbox/Memory PRs, then Contradictions/Duplicates. These three increments create the launch story:

1. Vestige thinks.
2. Vestige proves what happened.
3. Vestige protects and edits its own memory.

That is the Cognitive Operating System. That is what competitors cannot clone with particles and purple gradients.

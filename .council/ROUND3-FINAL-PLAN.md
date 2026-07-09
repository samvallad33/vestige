# ROUND 3 — FINALIZED GOD-TIER PLAN (Cognitive OS Dashboard)

Council: Claude Opus 4.8 (conductor) × GPT-5.5 (co-designer + overnight builder).
Converged Jul 8 2026. This file is the SINGLE SOURCE OF TRUTH the build swarm
executes. Every worker card references a section here.

Branch: `feat/dashboard-live-max`. Repo: `/Users/entity002/vestige`.
Dashboard: `apps/dashboard`. Never touch Memory Cinema or the Graph field.

---

## 0. THE LANGUAGE (locked) — Causal Bioluminescent Cortex

The dashboard is a dark local brain in a jar. Routes are organs viewed through
different instruments. NOT purple-on-black — blackwater + enzymatic light +
scarlet immune heat. One organism, distinct organs. Every visual is load-bearing
on a Vestige-only substrate (the discipline test) or it does not ship.

Color/motion source of truth: `src/lib/observatory/cognitive-palette.ts`
(already built + committed 0f24223). Key rules:
- blackwater `#020307` base, NEVER purple.
- retention oxygen ramp: extinction `#2A160B` → debt `#8A4B18` → healthy
  `#A8FF5E` → luciferin `#E9FFB7`.
- magenta `#FF2DF7` is reserved EXCLUSIVELY for RSB retrograde causality.
- indigo `#7C6CFF` ONLY for bitemporal transaction-time parallax.
- scarlet `#FF3B30` / `#B90D2B` for immune (veto/suppression).
- trust → membrane thickness (`membraneWidth(trust)`).
Motion grammar: chemotaxis, elastic axons, immune clamping, retrograde firing,
metabolic breathing, sedimentation, scar persistence, click-as-incision.

## 0.1 THE DISCIPLINE TEST (non-negotiable)
Every primitive carries real `Provenance` (see `src/lib/observatory/route-scene.ts`,
already built). Swap the backend value for Math.random() and `source` becomes a
lie the review catches. `assertProvenance(scene)` runs in dev. NO screensavers.

---

## 1. ARCHITECTURE (locked, both models agreed)

- **Graph field NodeState stays byte-stable at 64 bytes.** DO NOT grow it. DO
  NOT touch the 8 existing observatory shaders' NodeState struct. The live Graph
  field (FSRS decay, firewall, dream storm, causal recall) is finished + loved.
- **Route organs own their OWN richer layouts** behind a `RouteStage` boundary.
  Use `RouteSceneModel` (`src/lib/observatory/route-scene.ts`) as the adapter↔
  stage contract. Organs may use 96/128-byte cells, texture-backed fields, or
  SoA as needed — never mutate the shared Graph ABI.
- **Reuse the observatory engine + post-chain.** `ObservatoryEngine`
  (`engine.ts`: FramePass plugin, params buffer, DPR clamp, pause via
  `setPaused`, `preFrameHook`, `wallNowMs`, `totalFrames`), `PostChain`
  (rgba16float scene → mip bloom → Khronos PBR tonemap → grain → vignette,
  `SCENE_FORMAT='rgba16float'`). No new rendering deps.
- **Every organ inherits the full lifecycle contract**: loading / ready / stale
  / empty / error / reduced-motion / paused. Adapter-null → flat truthful metric
  snapshot, never black. WebGPU absent → DOM/SVG fallback. Reduced-motion clamps
  motion but KEEPS discrete event pulses. Pause is persistent + shared.
- **MSDF = ONE checked-in mono atlas** for hero labels/event-types/reason-codes/
  receipt-ids. DOM text is the source of truth for detail/accessibility/copy. If
  the atlas is absent, fall back to DOM/SVG labels — never block a hero on it.

## 1.1 GPU discipline (from the research)
- Field accumulation = render-pass ADDITIVE SPLATTING into a half-res
  `rg16float` texture. NO float atomics. Blend `{op:add, src:one, dst:one}`.
- Compute is ONLY for one-writer-per-pixel work (separable blur, post).
- NO GPU readback in the frame loop. Click-picking = CPU object map or ONE
  explicit input-driven pick readback (never per-frame).
- One-shot backend events (e.g. `DeepReferenceCompleted`) become LABELED
  deterministic replays — never fake streaming. Stage lights only if real
  output exists.
- Optional `timestamp-query` for a dev perf HUD; honest wall-time fallback.

---

## 2. SHARED FOUNDATION (build FIRST — blocks all organs)

### F1. `cognitive-palette.ts` — DONE (committed 0f24223).
### F2. `route-scene.ts` (RouteSceneModel + Provenance + assertProvenance) — DONE.

### F3. `RouteStage.svelte` + `route-stage/` engine glue  [CARD: foundation]
A reusable full-bleed WebGPU mount, the organ shell. Responsibilities:
- Own the full-bleed canvas + overlay slots (DOM chrome on top).
- Boot `ObservatoryEngine` with a route id + seed (reuse, don't fork).
- Accept a `RouteSceneModel` + a route-specific `FramePass[]` (the organ passes).
- Use `PostChain` unchanged (blackwater base + bloom).
- Wire pause + reduced-motion + WebGPU-null fallback EXACTLY like
  `ObservatoryStage.svelte` (copy that lifecycle; it's the proven contract).
- Expose a click callback that carries the picked real object id + action.
Ship with a trivial "field breathes, honest empty state" render so it mounts
with zero data. GATE: `pnpm --filter @vestige/dashboard check` green + it mounts
on one route without errors.

### F4. `field-pass.ts` + `shaders/field-*.wgsl.ts` — the cognitive field  [CARD: foundation]
The reusable 2.5D metaball substrate (research D1). Half-res `rg16float`:
additive splat (one instanced quad per cell) → separable 5-tap blur (compute)
→ fullscreen gradient membrane pass. Channels per organ (R density, G trust/
side-B). Exact WGSL in `.council/gpt55-round2-research.md` §D1. Every organ
samples this for the shared "organism" feel. GATE: check green + renders a
membrane field from a fake 20-cell scene in a test harness.

### F5. `click-shockwave.ts` + `shaders/click-shockwave.wgsl.ts`  [CARD: foundation]
Click-as-incision. A `ClickImpulseBuffer` (max 64) — each carries a real object
id + semantic action + frame. Mutating actions play AFTER API success; inspect
actions play if the object is real in the current scene. GATE: check green.

### F6. `msdf/` — one checked-in mono atlas + `msdf-text-renderer.ts` +
`shaders/msdf-text.wgsl.ts`  [CARD: foundation, LOWEST priority of F]
Median-of-RGB distance, screen-space width via `fwidth`, smoothstep alpha
(Red Blob / WebGPU sample). Hero labels only; DOM stays for detail. If atlas
generation is hard, ship a minimal pre-baked atlas or defer F6 and use DOM/SVG
labels in the first organs. GATE: check green; never blocks an organ.

---

## 3. FLAGSHIP ORGANS (god-tier, build after foundation, IN THIS ORDER)

Each: build a `*-scene.ts` adapter (real API/event → RouteSceneModel, run
`assertProvenance`), then the organ `FramePass`(es), then wire the route
`+page.svelte` to mount `RouteStage` full-bleed with DOM instruments on top.
GATE each on `pnpm check` + `pnpm build` + the discipline test.

### ORGAN 1 — Reasoning Theater "Eight-Stage Thought Organ"  [prototype-first]
Route: `reasoning`. Real data: `/deep_reference` result + `DeepReferenceCompleted`
event (query, intent, status, confidence, primary_id, supporting_ids[],
contradicting_ids[], contradiction_pairs[], memories_analyzed, receipt).
Hero: a vertical living spinal cord, 8 translucent chambers (intent→retrieve→
activate→evidence→contradiction→synthesis→recommendation→receipt). A query
enters as glyph fragments; real evidence cells flow chamber→chamber on
compute-updated Bezier splines (research D2). A stage lights ONLY if it has real
output (`count>0`). Contradiction/supersession = interrupts that cut the path.
HONEST: one-shot receipt → labeled deterministic ~6-9s replay, NOT fake stream.
Click: a chamber → the exact stage receipt (inputs/outputs/discarded/reason). An
evidence cell → center graph on that memory. WGSL: research §D2.
THIS DE-RISKS EVERYTHING — it exercises RouteStage + adapter + splines + MSDF +
click receipts + the color/motion language outside the Graph route.

### ORGAN 2 — Blackbox "Agent Flight Recorder Nerve Trace"
Route: `blackbox`. Real data: `/traces`, `/traces/:runId`, `TraceEvent` variants
(mcp.call, memory.retrieve w/ activation map, memory.write, veto), receipts
scoped by run. Hero: an agent run as a nervous-system trace — tool calls =
electric impulses on lanes, retrievals = green branches, suppressions = red
clamps, writes = cell births, vetoes = immune gates. Receipt IDs = MSDF beads.
GPU timeline ribbon, event-type lanes, per-event particles L→R by real order.
Click: an impulse → the exact trace event + linked receipt/export.
(Activation folds in here: the retrieve events carry the real activation map;
there is NO /activation endpoint — do not invent one.)

### ORGAN 3 — Contradictions "Immune Synapse Arena"
Route: `contradictions` (currently has the AmbientField base coat — replace with
the hero). Real data: `/contradictions`, `DeepReferenceCompleted.contradiction_pairs`,
trust deltas, Sanhedrin claims/verdicts. Hero: contradictory memories face each
other across a glowing immune synapse; higher trust thickens one membrane;
unresolved pairs spark scarlet arcs. Dual-channel signed field (research D1
contradictions mode: side A→R, side B→G, seam = min(R,G) + opposing gradient).
Click: the seam → contradiction receipt + evidence comparison + appeal/suppress/
supersede if available.

### ORGAN 4 — Duplicates "Synaptic Fusion Chamber"
Route: `duplicates`. Real data: `/duplicates(threshold,limit)` pairs + similarity,
the live threshold slider that recomputes clusters, content deltas, merge action.
Hero: near-duplicate memories as cells whose membranes are already partially
merged; the similarity threshold slider physically pushes twins together — neck
thickness = `smoothstep(0.78, 0.98, similarity)` (research D1 duplicates mode).
Differing tokens = illuminated mismatch filaments. Click the neck → inspect/
approve merge; a successful merge FUSES two nuclei + emits a receipt ring.

### ORGAN 5 — Timeline "Bitemporal Growth Rings" (build if it clears the bar)
Route: `timeline`. Real data: `/timeline` (days, per-day counts, memories),
`memoryAudit` (`/memories/:id/audit`), created/updated/suppressed events. Hero:
the screen is a cut cross-section of the brain — concentric valid-time rings +
offset indigo transaction-time shadows; rewrites/supersessions cut visible
seams. MSDF date ticks engraved into rings. Click a ring/cell → the exact time
slice (state then vs now).

### RSB retrograde axon (research D3) — a SHARED effect used by Reasoning +
Blackbox + (later) any recall organ. Magenta wavefront target→cause + permanent
cause-latch brightening. Build as a reusable `retro-axon` FramePass. Magenta
ONLY here.

---

## 4. SWEEP ORGANS (alive, not bespoke — after flagships)

Every remaining route mounts `RouteStage` with the blackwater field + cognitive
field pass + a real-metric adapter, so NOTHING is static purple, but they are
not each a bespoke hero. Routes: feed (event bloodstream), schedule (forgetting-
debt orrery), patterns (cross-project mycelium/physarum), memories (cellular
atlas), explore (chemoattractant probe), importance (salience furnace), dreams
(REM forge over existing dream storm), intentions (future-tense germline seeds),
memory-prs (immune review capsules), stats (metabolic vitals — already has
AmbientField, upgrade to field pass). Each: a `*-scene.ts` adapter binding the
real API + the shared field. GATE each on check + discipline test.

---

## 5. BUILD ORDER / SWARM CARDS

Increment A (FOUNDATION, blocks everything): F3 RouteStage, F4 field-pass,
F5 click-shockwave. (F6 MSDF parallel, non-blocking.)
Increment B (FLAGSHIPS, parallel after A): Organ 1 Reasoning (FIRST/prototype),
then Organ 2 Blackbox, Organ 3 Contradictions, Organ 4 Duplicates, retro-axon,
Organ 5 Timeline.
Increment C (SWEEP, parallel after A): the sweep organs.

## 6. HARD RULES FOR EVERY WORKER CARD
1. Work on `feat/dashboard-live-max`. Read this file + `.council/gpt55-round1.md`
   + `.council/gpt55-round2-research.md` FIRST.
2. NEVER touch `MemoryCinema.svelte`, `src/lib/graph/cinema/*`, or the Graph
   field's NodeState / 8 observatory shaders. Reuse the engine, don't fork it.
3. Every primitive carries real Provenance. Run `assertProvenance`. No Math.random
   as a semantic input. Click waves fire after API success (mutations) or if the
   object is real (inspect).
4. Reuse `cognitive-palette.ts`. magenta = RSB only, indigo = bitemporal only.
5. GATE on real exit codes before "done": `pnpm --filter @vestige/dashboard check`
   AND `pnpm --filter @vestige/dashboard build` must pass. Do not self-assess.
6. One organ per card. Land it green; the verifier bounces red cards.
7. Every organ ships the full lifecycle contract (loading/ready/stale/empty/
   error/reduced-motion/paused; fallback never black).

## 6b. GPT-5.5 SIGN-OFF CORRECTIONS (verified vs real repo — EVERY card carries these)
Source: `.council/gpt55-round3-signoff.md`. These override any conflicting prose above.

**T1 — `/deep_reference` has NO explicit 8 stage-receipts.** Derive stage lighting
from REAL response fields, label unexposed internals `not_exposed_by_backend`.
Honest 8-stage → real-field map (adapter uses THIS):
  1 intent: lights if `intent` exists (count 1).
  2 retrieve: lights if `memoriesAnalyzed>0` or `evidence.length>0` (count=that).
  3 activate: lights if `activationExpanded>0` (count=that); else dormant chamber.
  4 evidence: lights if `evidence.length>0`; cells map to real memory ids.
  5 contradiction: lights if `contradictions.length>0` / `claim_conflicts.length>0`.
  6 synthesis: lights if `reasoning` or `guidance` exists.
  7 recommendation: lights if `recommended.memory_id` exists.
  8 receipt: lights if `composition_event_id` or `compositionWriteStatus` exists;
    if no real receipt id, render a labeled status bead (persisted/skipped_empty/
    failed) as SCALAR provenance — never a fake receipt id.
Build Organ 1 from the HTTP `/deep_reference` response after query submit; the
`DeepReferenceCompleted` WS event is corroborating live-pulse input, NOT the sole
scene source. "Click a stage → exact data" = exact EXPOSED data (label the rest).

**T2 — real contradiction/supersession shapes (from cross_reference.rs).** The
`reasoning-scene.ts` adapter MUST normalize the CURRENT backend shapes first,
with legacy fallback:
  contradictions[]: `{ stronger:{id,preview,trust,date}, weaker:{id,...}, topic_overlap }`
    (NOT `a_id/b_id/summary` — that's the stale UI expectation; support both).
  superseded[]: `{ id, preview, trust, date, superseded_by }` (NOT `old_id/new_id/reason`).
  confidence: backend returns ~0..100; normalize to 0..1 defensively.
  closest real receipt id = `composition_event_id`. There is NO `receipt` field.

**T3 — WebGPU format + FramePass traps.**
- HARD RULE (verified live on M-series: `rgba16float-renderable`=false, rgba16float
  is NOT a writable storage-texture format here): NEVER use `texture_storage_2d<...,
  write>` + `textureStore` for the blur. Do the separable blur as FULLSCREEN
  RENDER-PASS draws (H then V) into RENDER_ATTACHMENT textures, sampling the
  source via `textureSampleLevel`. Field textures usage = `RENDER_ATTACHMENT |
  TEXTURE_BINDING` (NO `STORAGE_BINDING`). The whole field pipeline is render
  passes: splat (additive) → blur-H → blur-V → membrane. No storage textures
  anywhere. `rgba16float` is fully valid as a RENDER TARGET.
- Logical channels in `.rg`, `.ba` reserved.
- FramePass sequencing: the engine calls ALL `compute()` then opens ONE main HDR
  scene pass and calls ALL `render()`. So a field pass does splat+blur INSIDE its
  `compute(encoder)` (it receives a `GPUCommandEncoder`, not a compute-pass
  encoder — encode its own offscreen render pass + compute blur passes there),
  then draws the membrane in `render()` into the main scene pass. Do NOT split
  splat into render() + blur into compute() (wrong order, one-frame lag).
- Route field textures need their OWN `ensure(w,h)` on resize (keyed off
  engine.params[6]/[7]); do NOT touch PostChain.sceneView outside the main pass.
- NodeRenderer already proves the additive-splat blend into `engine.sceneFormat`.

**T4 — RouteStage copies the LIFECYCLE from ObservatoryStage, NOT its data path.**
Reuse: canvas/engine mount+resize+dispose, WebGPU fallback, reduced-motion auto-
pause, `engine.setPaused`, persistent pause control, loading/error/empty DOM
overlays, one click-driven pick (no frame-loop readback). Do NOT copy
`api.graph()` / NodeRenderer / BirthRenderer — that's Graph-specific. Boot the
engine with an existing DemoMode; route id lives in pass-local uniforms, NOT the
shared Graph params. Run `assertProvenance(scene)` in dev before upload.

**T5 — magenta stays RSB-ONLY.** The D1 sample mixed magenta into the
contradiction seam — WRONG. Contradictions use scarlet/immune reds. Magenta
`#FF2DF7` appears ONLY in the retrograde causal axon.

**T6 — NEVER share one WGSL module between compute and render pipelines when it
declares a `read_write` storage buffer or a `write` storage texture.** WebGPU
forbids read_write storage buffers AND write storage textures in vertex/fragment
stages (compute-only). A render pipeline whose module/layout exposes such a
binding is INVALID at runtime (check+build stay green — only live-GPU validation
catches it). SPLIT WGSL into per-pipeline modules: compute module (advect/blur)
= read_write storage + write storage texture OK; render modules (splat/membrane)
= storage buffers as `var<storage, read>` only. Each pipeline gets its own
explicit bind group layout matching exactly. (Found live in Organ 1's field
splat pass — this is the D1 metaball pattern's #1 trap.)
This is why Claude's LIVE-GPU verify per organ is non-optional: the swarm
verifier's check/build cannot catch invalid-pipeline runtime errors.

## 7. THE PROTOTYPE ACCEPTANCE TEST (Organ 1, de-risks all)
1. User enters a query on `/reasoning`.
2. Backend returns `/deep_reference`.
3. Full-bleed WebGPU organ animates 8 stages in exact order (labeled replay).
4. Evidence cells correspond to real memory ids.
5. Contradiction/supersession interrupts appear ONLY if present in the result.
6. Clicking a stage opens exact data for that stage.
7. Swapping backend values for random visibly breaks provenance.
If Organ 1 lands, every other organ is: real data → adapter → organ → click
receipt.

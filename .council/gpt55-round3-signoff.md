# GPT-5.5 Round 3 Signoff — Cognitive OS Dashboard

Date: 2026-07-08
Repo: `/Users/entity002/vestige`
Branch verified: `feat/dashboard-live-max`
Scope: final sanity check of the Round 3 plan against the real repo before building Organ 1.

## Verdict

READY to build Organ 1 — Reasoning Theater — first as the de-risk prototype.

The plan is directionally sound and correctly protects the finished Graph field ABI. The foundation sequence is right: build `RouteStage` + reusable route-local field/click systems, then land Reasoning Theater as the first proof that real backend data can drive a full-bleed organ without mutating the existing Observatory `NodeState` or shaders.

There are three corrections/traps workers must carry into the cards:

1. The `/deep_reference` HTTP response does NOT expose eight explicit per-stage receipt objects today. It exposes enough real signals to build an honest eight-stage deterministic replay, but the adapter must derive stage counts from existing fields and label missing internals as not exposed — not invent discarded-candidate/stage-receipt detail.
2. `DeepReferenceCompleted` WebSocket event carries a smaller event payload than the HTTP response. It has `query`, `intent`, `status`, `confidence`, `primary_id`, `supporting_ids`, `contradicting_ids`, `contradiction_pairs`, `memories_analyzed`, `duration_ms`, `timestamp`; it does not carry `receipt`, `supporting_ids[]` plus contradiction pairs are event-level only, and it does not carry `recommended`, `evidence`, `superseded`, `evolution`, `related_insights`, `reasoning`, or `activationExpanded`.
3. The D1 `rg16float` storage-texture blur sketch is a portability trap. Use `rgba16float` for storage-texture ping-pong or render-pass blur unless verified in-browser; keep logical R/G channels but do not bet the route foundation on `texture_storage_2d<rg16float, write>`.

## Evidence read

Files checked:

- `.council/gpt55-round1.md`
- `.council/gpt55-round2-research.md`
- `apps/dashboard/src/lib/observatory/route-scene.ts`
- `apps/dashboard/src/lib/observatory/ObservatoryStage.svelte`
- `apps/dashboard/src/lib/components/ObservatoryCanvas.svelte`
- `apps/dashboard/src/lib/observatory/engine.ts`
- `apps/dashboard/src/lib/observatory/types.ts`
- `apps/dashboard/src/lib/observatory/live-bridge.ts`
- `apps/dashboard/src/routes/(app)/reasoning/+page.svelte`
- `apps/dashboard/src/lib/stores/api.ts`
- `apps/dashboard/src/lib/types/index.ts`
- `crates/vestige-mcp/src/dashboard/handlers.rs`
- `crates/vestige-mcp/src/dashboard/events.rs`
- `crates/vestige-mcp/src/tools/cross_reference.rs`
- `apps/dashboard/src/lib/observatory/node-renderer.ts`
- `apps/dashboard/src/lib/observatory/post/post-chain.ts`

## A. RouteStage lifecycle-copy from ObservatoryStage

### Finding

Sound, with one implementation refinement: copy the lifecycle contract and pause/reduced-motion behavior, but do not blindly copy the Graph-specific data load/upload path.

`ObservatoryStage.svelte` already proves the needed route-shell behaviors:

- full-bleed or embedded parent-filling canvas shell;
- `ObservatoryCanvas` owns engine mount/start/resize/dispose;
- WebGPU unsupported/error state is readable fallback, not a crash;
- reduced-motion initializes from `prefers-reduced-motion` and auto-pauses unless the user overrides;
- `engine.setPaused(paused)` freezes ambient sim drift while live/discrete pulses can still land through `preFrameHook`;
- persistent visible pause control for live field;
- loading/error/empty states are DOM overlays, not black voids;
- one explicit click-pick path only on user click, no frame-loop readback.

`ObservatoryCanvas.svelte` is also reusable in spirit: it boots `ObservatoryEngine`, DPR-clamped resize, and fallback. For `RouteStage`, either generalize it or create a route-specific canvas host with the same engine lifecycle.

### RouteStage implementation notes

Recommended shell:

- Props: `organ`, `seed`, `scene`, `passesFactory` or `passes`, `status`, `embedded`, `chrome/fallback slots`, `onpick`.
- Construct/boot `ObservatoryEngine` once per mount with a valid existing `DemoMode` until engine options are generalized. Do not extend Graph `DemoMode` unless necessary; route id can live in `RouteStage`/pass-specific uniforms, not the shared Graph params.
- On engine ready, instantiate route passes with the engine and scene, then call `engine.addPass(pass)`.
- On scene change, call pass-level `upload(scene)` methods; run `assertProvenance(scene)` in dev before upload.
- Preserve the pause semantics: paused freezes ambient route motion; inspect/mutation click pulses can still be discrete information.
- Preserve fallback semantics: WebGPU absent -> DOM/SVG snapshot of the real scene metrics, not a black panel.
- Adapter-null or `scene.alive=false` -> honest empty state plus blackwater breath only.

### Trap

`ObservatoryStage` currently fetches `api.graph()` and creates `NodeRenderer`, `BirthRenderer`, etc. That part is Graph-specific and must not be copied into `RouteStage`. The reusable part is canvas/engine/status/reduced-motion/pause/click/fallback lifecycle.

## B. `/deep_reference` payload reality check

### HTTP `/api/deep_reference` response

The route calls `api.deepReference(query, depth)` -> POST `/deep_reference`. The dashboard normalizer in `apps/dashboard/src/routes/(app)/reasoning/+page.svelte` expects and currently uses:

- `query` (backend response includes it)
- `intent`
- `status`
- `confidence` (backend returns 0..100-ish; existing UI defensively normalizes 0..1 too)
- `reasoning`
- `guidance`
- `memoriesAnalyzed`
- `activationExpanded`
- optional `claim_conflicts`
- optional `recommended`: `{ answer_preview, memory_id, trust_score, date }`
- optional `evidence[]`: `{ id, preview, trust, relevanceScore, date, role }`
- optional `contradictions[]`
- optional `superseded[]`
- optional `evolution[]`
- optional `related_insights[]`
- optional `composition_event_id`
- `compositionWriteStatus`

Backend source: `crates/vestige-mcp/src/tools/cross_reference.rs`.

### Actual contradiction/supersession shapes

Important shape mismatch versus the Round 3 prose:

- HTTP `contradictions[]` are currently backend objects shaped like:
  - `stronger: { id, preview, trust, date }`
  - `weaker: { id, preview, trust, date }`
  - `topic_overlap`
- The existing Svelte route normalizer incorrectly expects `a_id` / `b_id` / `summary`; it will degrade those to empty ids unless another compatibility layer is added. The Reasoning Theater adapter should support BOTH shapes:
  - current backend shape: `stronger.id` / `weaker.id`
  - legacy/UI shape: `a_id` / `b_id`
- HTTP `superseded[]` are currently shaped like:
  - `id`
  - `preview`
  - `trust`
  - `date`
  - `superseded_by`
- Existing UI normalizer expects `old_id` / `new_id` / `reason`; the new adapter should support current backend shape first, with legacy fallback.

### WebSocket `DeepReferenceCompleted` event

Backend source: `crates/vestige-mcp/src/dashboard/events.rs` and `dashboard/handlers.rs`.

Actual fields:

- `query: String`
- `intent: String`
- `status: String`
- `confidence: f64`
- `primary_id: Option<String>`
- `supporting_ids: Vec<String>`
- `contradicting_ids: Vec<String>`
- `contradiction_pairs: Vec<(String, String)>`
- `memories_analyzed: usize`
- `duration_ms: u64`
- `timestamp: DateTime<Utc>`

This event is enough to arm live field highlights, but not enough for full stage receipts. Build Organ 1 from the HTTP response after the user submits the query; treat the WebSocket event as corroborating/live pulse input, not the sole scene source.

### Eight-stage signal mapping that is honest today

The backend code comments describe an eight-stage pipeline, but the response exposes summarized outputs rather than full internal trace records. Use this mapping:

1. `intent`: lights if `intent` exists; count = 1; source = scalar/event from response query+intent.
2. `retrieve`: lights if `memoriesAnalyzed > 0` or `evidence.length > 0`; count = `memoriesAnalyzed` or evidence count.
3. `activate`: lights if `activationExpanded > 0`; count = `activationExpanded`. If zero, chamber visible/dormant.
4. `evidence`: lights if `evidence.length > 0`; evidence cells map to real memory ids.
5. `contradiction`: lights if `contradictions.length > 0` or `claim_conflicts.length > 0`; endpoints must resolve to real ids where possible.
6. `synthesis`: lights if `reasoning` or `guidance` exists.
7. `recommendation`: lights if `recommended.memory_id` exists.
8. `receipt`: lights if `composition_event_id` exists or `compositionWriteStatus` exists. If there is no real receipt id, render a labeled status bead (`persisted` / `skipped_empty` / `failed`) as a scalar provenance, not a fake receipt id.

Do NOT claim per-stage discarded candidates or exact intermediate input/output receipts unless the backend starts exposing them. For the prototype, clicking a chamber should open exact normalized data available for that chamber, with unexposed internals labeled `not_exposed_by_backend`.

### Acceptance-test correction

Round 3 acceptance line “clicking a stage opens exact data for that stage” is valid if interpreted as exact exposed data for that stage. It is NOT valid as “every backend internal stage has a complete receipt.” If the build needs true stage receipts later, add a backend trace/receipt envelope as a separate card; do not block Organ 1.

## C. WGSL / FramePass gotchas for field-pass + splat approach

### FramePass sequencing

Current `ObservatoryEngine` sequencing:

1. writes params;
2. calls every `pass.compute?.(encoder, frame)`;
3. opens ONE main HDR scene render pass and calls every `pass.render?.(render, frame)`;
4. runs `PostChain.encode`.

That means a field pass that needs `render splats -> blur compute -> membrane render` cannot put splats in `render()` and blur in `compute()` in the same frame. Options:

- Preferred: in `field-pass.compute(encoder, frame)`, despite the method name, encode an offscreen render pass to clear/additively splat into the field texture, then encode compute blur passes, then let `field-pass.render()` draw the membrane into the main scene pass.
- Alternative: add a richer engine hook later (`preRenderPasses`) if multiple organs need interleaved offscreen render/compute sequencing. Not needed for Organ 1 if the foundation pass owns its offscreen passes inside `compute()`.
- Avoid using previous-frame blurred fields unless explicitly accepted; it creates one-frame lag and complicates click pulses.

This is legal because `compute(encoder, frame)` receives a `GPUCommandEncoder`, not a `GPUComputePassEncoder`; the type name is semantic, not restrictive.

### Texture-format portability

D1 says half-res `rg16float` and blur WGSL uses `texture_storage_2d<rg16float, write>`. The render-attachment side is plausible, but the storage-texture side is the trap: WebGPU storage texture format support is narrower than sample/render formats, and `rg16float` is not a safe baseline for write-only storage textures.

Safer launch implementation:

- Use `rgba16float` for field ping-pong textures if compute blur writes storage textures.
- Store logical channels in `.rg`; leave `.ba` unused/reserved.
- Or do separable blur as render-pass fullscreen draws into render attachments, avoiding storage texture format risk.
- Keep field half-res; `rgba16float` doubles the field memory versus `rg16float` but is still acceptable at launch scale.

### Additive splat correctness

The render-pass additive splat approach is sound and matches existing repo patterns:

- `NodeRenderer` already uses `blend: { color: one/one add, alpha: one/one add }` into `engine.sceneFormat`.
- `PostChain` already uses `SCENE_FORMAT='rgba16float'` scene/bloom textures and additive bloom.
- Offscreen field splats should use bounded instanced quads; overdraw is the real perf risk, not ALU.

### Bind/resource lifecycle

- Route field textures must recreate on canvas resize. Current `PostChain` handles its own resize internally; route passes need their own `ensure(width,height)` keyed off `engine.params[6]`/`[7]` or canvas dimensions.
- Do not access `PostChain.sceneView` outside the main render pass. Render membranes in the provided main scene `GPURenderPassEncoder`.
- Do not add GPU readback to the frame loop. Picking stays CPU object map or one explicit click-driven readback.
- Because route organs may own richer buffers, keep them route-local. Do not touch `apps/dashboard/src/lib/observatory/types.ts` `FLOATS_PER_NODE=16` / `BYTES_PER_NODE=64` or the existing Graph WGSL `NodeState` structs.

### Color-language gotcha

Round 2 D1 contradiction sample mixes magenta into contradiction seam color. Round 3 locks magenta `#FF2DF7` exclusively for RSB retrograde causality. For Contradictions, use scarlet/red/immune colors; do not reuse magenta in contradiction field shaders.

## D. Missing-but-needed implementation cards

These should be folded into Organ 1 or Foundation cards:

1. `reasoning-scene.ts` must normalize current backend contradiction/supersession shapes, not only the existing route-local legacy interfaces.
2. The Reasoning route should preserve the existing DOM answer/evidence accessibility as overlay/detail. The WebGPU organ is the hero, not the only source of text.
3. Stage click receipts need a side panel/data drawer sourced from the normalized stage model. It should explicitly show `not_exposed_by_backend` for internals the backend does not expose.
4. A small route-stage test harness should instantiate `emptyScene('reasoning')` and a fake-but-provenance-valid 20-cell scene. Fake test harness values are OK only if every primitive's provenance is explicit test provenance; production adapters must use real backend data.
5. If true “receipt id” is desired, `composition_event_id` from `deep_reference` is the closest existing persisted id. There is no separate `receipt` field today.

## E. Protected areas

Confirmed plan remains compliant:

- Do not mutate `apps/dashboard/src/lib/observatory/types.ts` `NodeState` size/layout.
- Do not touch Memory Cinema files.
- Do not touch Graph field shaders for route organs.
- Route organs can reuse `ObservatoryEngine`, `PostChain`, palette, and lifecycle without forking the renderer stack.

## Build readiness for Organ 1

I am ready to build Organ 1 first.

Recommended first build cut:

1. Add `RouteStage.svelte` shell that boots `ObservatoryEngine`, mirrors ObservatoryStage fallback/pause/reduced-motion lifecycle, and accepts route passes + overlay slots.
2. Add `reasoning-scene.ts` with current-backend normalization and `assertProvenance(scene)`.
3. Add a minimal Reasoning Theater pass: eight chambers + evidence packets using route-local buffers; DOM side panel for chamber clicks.
4. Wire `/reasoning` so query submit calls `/deep_reference`, builds a scene from the exact response, and mounts the full-bleed organ behind existing DOM instruments.
5. Keep MSDF optional in the first Organ 1 cut; DOM/SVG labels are acceptable until F6 lands.
6. Gate with `pnpm --filter @vestige/dashboard check` and `pnpm --filter @vestige/dashboard build`.

Bottom line: build it. The plan is correct once workers respect the payload reality and the FramePass/texture-format traps above.

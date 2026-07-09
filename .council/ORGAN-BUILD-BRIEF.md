# ORGAN BUILD BRIEF v2 — ALL-WEBGPU, ZERO DOM (swarm workers read this FIRST)

You are a GPT-5.5 worker building ONE piece of the Vestige **Cognitive OS** dashboard.
Branch `feat/dashboard-live-max` at `/Users/entity002/vestige` (cd there; the kanban
scratch home is not the repo — do the real work in the repo).

Source of truth for architecture: `.council/GOD-TIER-ALLWEBGPU-PLAN.md`. Read it.

---

## 🚫 THE ONE RULE THAT OVERRIDES EVERYTHING — ZERO DOM. EVERY LINE BREATHES WebGPU.

The ENTIRE dashboard renders in WebGPU into a **single `<canvas>`**. Text, labels,
memory content, receipt IDs, numbers, buttons, controls, cursor, focus, selection,
loading / empty / error / paused states — **100% drawn by the GPU into that one
canvas.** The ONLY non-WebGPU element allowed on the page is the `<canvas>` itself.

**FORBIDDEN (any one = RED card, no exceptions):**
- ANY `<div>`, `<p>`, `<span>`, `<button>`, `<ul>`, list panel, detail panel, label,
  tooltip, overlay, or badge that a user can see OR that is present in the DOM tree.
- An "invisible" / `sr-only` / `aria-live` accessibility mirror. **Sam rejected it
  explicitly** ("FUCK TO THE NO"). Do NOT add one to "solve" accessibility.
- A DOM/SVG fallback panel for "WebGPU absent". The absent-WebGPU state is drawn on
  the canvas too (a static honest message via the text layer + last frame), never a
  DOM node. Never a black screen either.
- `app.css` styling of any visual element. It is effectively empty.

Accessibility is handled canvas-side (pickable quads + GPU focus ring) or DEFERRED —
never by reintroducing DOM. The shareable-canvas thesis wins: a full-screen WebGPU
canvas survives a screen-recording / OS screenshot / phone camera; a DOM dashboard
reads as a generic web app and dies as a shareable. That is the whole point.

**ACCEPTANCE TEST (the verifier + the human conductor both run it):** screen-record the
route. If a single HTML element is visible OR present in the DOM tree beyond `<canvas>`,
the card is WRONG. Text = MSDF glyphs. A "button" = MSDF label + pickable quad +
click-shockwave. Focus/selection = a GPU cyan highlight, not a CSS outline.

---

## 🔴 THIS IS A BUILD TASK — YOU MUST WRITE + SAVE REAL FILES (read twice)

A "design", "plan", "recommendation", "handoff", or "reconnaissance" is a FAILURE.
You are NOT done until real files exist on disk AND the gate is green AND there is
zero DOM. If you finish without writing files, you failed the card.

You MUST paste, in your completion summary:
- `ls -la` of the files you created,
- `git diff --stat` of what you changed,
- the real tail of `pnpm --filter @vestige/dashboard check` (**0 errors**) and `build`,
- a one-line confirmation you did NOT add any DOM element and did NOT touch protected files.

No files + no gate output = the verifier bounces you RED.

---

## 🔥 WORK TO THE ABSOLUTE MAX — SPAWN YOUR SUBAGENTS (non-negotiable)

Do NOT build solo. Use `delegate_task(tasks=[...])` to fan out PARALLEL subagents.
Delegation is configured for **5 concurrent children, spawn depth 2** — use it.

Recommended fan-out per organ card (each subagent WRITES ITS OWN FILE — a subagent that
returns prose instead of a written file is wasted; if that happens, YOU write the file):
- **subagent A** — research the REAL API/event payload for this organ (curl the running
  brain on `http://127.0.0.1:3931`, read the Rust handler) and write findings to
  `.council/notes/<organ>-data.md`. Confirms the discipline-test signal EXISTS before any WGSL.
- **subagent B** — write `<organ>/<organ>-scene.ts` (adapter: real payload → RouteSceneModel).
- **subagent C** — write `<organ>/<organ>-pass.ts` (the FramePass + WGSL hero).
- **subagent D** — write the MSDF text/label layout for this organ (uses the shared text layer).
- then **YOU** integrate, wire the route `+page.svelte`, run the gate, paste output.

**⚠️ CONTAINER-COLLISION TRAP (verified from Hermes docs):** parallel subagents SHARE
ONE container/filesystem. Concurrent `cd`, env mutations, and writes to the SAME path
COLLIDE. So: give each subagent a DISTINCT target file path, tell each to use absolute
paths (no `cd`), and never have two subagents write the same file. Integration (one
writer) is YOUR job after they return.

The bar is **"people lose their minds"** AND it renders AND it is zero-DOM — not "it compiles".

---

## THE VERIFIED CONTRACTS (read from real source — conform exactly)

**FramePass** (`$lib/observatory/engine.ts`): `interface FramePass { compute?(encoder:
GPUCommandEncoder, frame: number): void; render?(pass: GPURenderPassEncoder, frame:
number): void }`. The engine runs **ALL** passes' `compute(encoder)` first, THEN opens
**ONE** main HDR scene render pass (clears to `VOID_CLEAR_HDR`) and runs **ALL** passes'
`render(pass)`. So: a field/offscreen pass encodes its own offscreen splat+blur render
passes INSIDE `compute(encoder)`; it draws its membrane/quads/text in `render(pass)`.

**Render target format**: draw into `engine.sceneFormat` (= `'rgba16float'`, the offscreen
HDR scene texture — NOT the swapchain). PostChain composites scene→swapchain afterward
(mip bloom → Khronos tonemap → grain → vignette). **Text drawn into the scene gets bloom
for free = it glows.** Render your MSDF text BEFORE post, into the scene texture.

**Params uniform**: `engine.paramsBuffer`, 16 f32, bound `@group(0) @binding(0)`. Struct
layout (COMMON_WGSL): frame, loop_phase, node_count, edge_count, path_count, pulse,
viewport_w, viewport_h, brightness, demo_id, time, capture_mode, live_kind, live_frame,
live_energy, projection_days.

**Pick / interaction**: implement `pickAt(ndcX, ndcY): RoutePick | null` (CPU rect hit-test
against your laid-out quads/glyph boxes). RouteStage converts pointer clientXY → NDC
(y flipped) and calls each pass's `pickAt`. A control is a pickable quad, not a `<button>`.

**RouteSceneModel** (`$lib/observatory/route-scene.ts`): nodes/edges/events/receipts each
carry a `Provenance {kind, id, scalar?}`. `assertProvenance(scene)` (call in `import.meta.env.DEV`)
THROWS if any primitive lacks a real source. `scalars: Record<string, number>` for route
metrics. `emptyScene(organ)` → `alive:false` honest empty (drawn on canvas, not DOM).

**Copy the proven field pipeline** from `reasoning/reasoning-theater-pass.ts`: additive splat
→ fullscreen blur-H → blur-V → membrane, ALL render passes, rgba16float textures with usage
`RENDER_ATTACHMENT | TEXTURE_BINDING` (NO STORAGE_BINDING). Own bind-group layouts +
pipeline layouts in `ensurePipelines`; recreate field textures on resize in `ensureResources`.

---

## DAY-1 CRITICAL PATH (dependency chain — these are SEQUENTIAL, not parallel organs)

Nothing all-WebGPU renders text until the MSDF engine exists. Build in THIS order; each is
its own card, live-GPU-audited by the conductor before the next starts:

1. **MSDF text engine** (`$lib/observatory/text/`): a checked-in MSDF atlas
   (JetBrains Mono, `static/msdf/` PNG + JSON — generate offline, never at runtime) +
   `msdf-atlas.ts` (loads atlas texture + glyph metrics) + `text-layer.ts` (a FramePass:
   CPU lays out strings → per-glyph instance buffer {glyphRect, atlasRect, worldAnchor,
   semanticColorRGBA, ageFrame, confidence}; GPU draws instanced quads, median-of-RGB MSDF
   distance → smoothstep AA, tints by semantic color, reveals glyph-by-glyph over ageFrame,
   renders into the scene pre-bloom) + `msdf-text.wgsl` (single render-only module; storage
   buffers `var<storage, read>`; no reserved-word fields) + a `layout` helper. ACCEPTANCE:
   a test route renders `hello · 5de3e41f · trust 51%` as glowing in-canvas text, pickable,
   materializing glyph-by-glyph, tinted by semantic color.
2. **Zero-DOM RouteStage rebuild** (`RouteStage.svelte`): strip EVERY DOM node — the
   `<slot name="chrome">`, the `<button>` pause control, the telemetry/loading/error/empty
   `<div>`s. Result = one `<canvas>` (via ObservatoryCanvas) hosting L0 field + L1 organ +
   L2 MSDF text + L3 interaction. Pause control, telemetry, loading/error/empty are all MSDF
   text + pickable quads on the canvas. Pointer/keyboard events on the canvas → pickAt.
3. **Interaction engine** (`$lib/observatory/interaction/`): cursor field (pointer = soft
   light, chemotaxis flinch), hover ignition (hover a pickable → it + real neighbors ignite,
   MSDF label materializes), click = incision + receipt (real actions play the wave AFTER the
   API succeeds — no optimistic fake pulses), pickable-quad controls, GPU cyan focus ring.

Only AFTER Day-1 lands do the flagship conversions (Day 2) and the sweep organs (Day 4,
parallel swarm) run — each reusing this text + interaction layer.

---

## HARD RULES (a card is RED if it violates any)

- **NEVER touch**: `MemoryCinema.svelte`, `src/lib/graph/cinema/*`, `observatory/types.ts`
  NodeState (64 bytes), the 8 existing observatory shaders, the Graph field. REUSE the
  engine, don't fork it.
- Every visual primitive carries real `Provenance`. Run `assertProvenance(scene)` in dev.
  NO `Math.random()` as a semantic input. The Math.random() discipline test governs — swap
  real data for random and the viewer can tell.
- Reuse `cognitive-palette.ts`. blackwater `#020307` base, **never purple, ever**. cyan
  `#22C7DE` = the single interactive accent. magenta `#FF2DF7` = RSB retrograde causality
  ONLY. indigo `#7C6CFF` = bitemporal ONLY. scarlet reds = immune (contradiction/veto/suppression).

## WEBGPU TRAPS (all found live — DO NOT repeat)

- **T3 (field textures)**: `rgba16float`, usage `RENDER_ATTACHMENT | TEXTURE_BINDING` (NO
  STORAGE_BINDING). Separable blur = FULLSCREEN RENDER PASSES; NEVER `texture_storage_2d<...,
  write>` + textureStore (not portable on M-series). Copy reasoning-theater-pass.ts.
- **T6 (module split)**: a render pipeline's WGSL module must NOT declare `var<storage,
  read_write>` or write storage textures (compute-only). Render-stage storage = `var<storage,
  read>`. Split modules per pipeline.
- **Reserved words** — NEVER a var OR struct field named: `active`, `meta`, `filter`,
  `sample`, `texture`, `binding`, `common`, `override`, `enable`, `const`, `handle`, `input`,
  `output`, `access`, `layout`. (`active` broke Organ 1, `meta` broke Blackbox.) Use `beat`/
  `info`/`data2`.
- **Struct-field match**: every `struct.field` referenced in WGSL must EXIST in the struct.
  TS build does NOT catch this.
- **Pipeline-layout scoping**: a pipeline's explicit layout need only cover the SELECTED
  entry point's statically-reached bindings; a shared multi-entry module is fine. Watch
  texture `sampleType:'float'` vs a filtering sampler (creation error).
- **One-shot events = labeled deterministic replay**, never fake streaming. An element lights
  only if it has real output.
- **Capture**: snapshot copies `getCurrentTexture()` in the SAME task (WebGPU ignores
  `preserveDrawingBuffer`). Wordmark baked into the post chain, not DOM.

## LIFECYCLE (every organ, all in-canvas — NO DOM)

loading (organ breathes dim + MSDF "replaying…") / ready / stale / empty (calm blackwater +
honest MSDF line) / error (scarlet pulse + MSDF message) / reduced-motion (freeze ambient
drift, KEEP discrete event pulses) / paused (MSDF pause glyph, pickable). WebGPU absent →
canvas-drawn static message + last frame, never DOM, never black.

## GATE (do NOT self-assess — RUN these; the verifier re-runs them)

- `pnpm --filter @vestige/dashboard check` → **0 errors**
- `pnpm --filter @vestige/dashboard build` → success
- If you can, sanity-check WGSL with `getCompilationInfo()` before marking done.
The human conductor (Claude/Opus) does the final live-GPU audit (Preview console +
getCompilationInfo + screenshot on the running app) — check/build cannot catch invalid-
pipeline runtime errors or "still DOM / still slop." Hold BOTH bars: renders-with-real-data
AND looks-category-of-one AND zero-DOM.

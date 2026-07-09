# ORGAN BUILD BRIEF — swarm worker cards read this FIRST

You are a GPT-5.5 worker building ONE Cognitive OS dashboard organ. Branch
`feat/dashboard-live-max` at `/Users/entity002/vestige` (cd there; the kanban
scratch workspace is just your home — do the real work in the repo).

## 🔴 THIS IS A BUILD TASK. YOU MUST WRITE + SAVE REAL FILES. (read twice)
A "design", "plan", "recommendation", "handoff", or "reconnaissance" is a
FAILURE. You are NOT done until three NEW files exist on disk and the gate is
green. If you finish without writing files, you have failed the card.

MANDATORY DEFINITION OF DONE (all must be TRUE, verify each yourself):
1. `apps/dashboard/src/lib/observatory/<organ>/<organ>-scene.ts` EXISTS on disk
   (you wrote it — `ls` it to confirm).
2. `apps/dashboard/src/lib/observatory/<organ>/<organ>-pass.ts` EXISTS on disk
   (you wrote the FramePass + WGSL — `ls` it to confirm).
3. `apps/dashboard/src/routes/(app)/<organ>/+page.svelte` is MODIFIED to mount
   `<RouteStage organ="<organ>" passes={...} scene={...} .../>` full-bleed
   (`git diff` it to confirm your edit is there).
4. `pnpm --filter @vestige/dashboard check` prints `0 ERRORS`. You RAN it and
   pasted the real output.
5. `pnpm --filter @vestige/dashboard build` succeeded. You RAN it.
In your completion summary you MUST paste: the `ls -la` of your organ dir, the
`git diff --stat` showing your changed files, and the tail of the real check +
build output. NO files + no gate output = the verifier bounces you as RED.

## HOW TO BUILD IT (do the work, don't just plan it)
1. FIRST copy the working reference to steal its exact structure:
   `cp apps/dashboard/src/lib/observatory/reasoning/reasoning-theater-pass.ts
   apps/dashboard/src/lib/observatory/<organ>/<organ>-pass.ts` then rewrite it
   for your organ's hero + real data. Same for the scene adapter (copy
   reasoning-scene.ts). This guarantees the WebGPU-safe pipeline (all render
   passes, no storage textures, split modules, no reserved words, real struct
   fields) instead of reinventing it and re-hitting the traps.
2. You MAY use `delegate_task` to PARALLELIZE the WRITING — but each subagent
   must WRITE ITS FILE, not return prose. E.g. delegate: "write the full
   contents of <organ>-scene.ts to disk and confirm with ls" — not "recommend a
   design." If a subagent returns a design instead of a written file, YOU write
   the file from its design. The card is not done until the files are on disk.
3. Then YOU integrate, wire the route, and RUN the gate. Paste the output.
The bar is "people lose their minds" AND it renders — but step 0 is the files
must exist. A brilliant design with no files is a RED card. GO WRITE CODE.

READ FIRST, in order:
1. `/Users/entity002/vestige/.council/ROUND3-FINAL-PLAN.md` — the full plan +
   §6b traps T1-T6 (BINDING).
2. `/Users/entity002/vestige/.council/gpt55-round2-research.md` — the WGSL
   recipes (metaball field D1, spline advection D2, retrograde axon D3).
3. WORKING REFERENCE CODE — copy the proven patterns, don't reinvent:
   - `apps/dashboard/src/lib/observatory/reasoning/reasoning-theater-pass.ts`
     (the metaball field: additive splat -> render-pass blur -> membrane, all
     render passes, no storage textures; compute spline advection).
   - `apps/dashboard/src/lib/observatory/contradictions/contradictions-pass.ts`
     (dual-channel signed field, scarlet seam).
   - `apps/dashboard/src/lib/observatory/RouteStage.svelte` (the organ shell).
   - `apps/dashboard/src/lib/observatory/route-scene.ts` (RouteSceneModel +
     assertProvenance) and `cognitive-palette.ts` (the color language).

## HARD RULES (a card is RED if it violates any)
- NEVER touch: `MemoryCinema.svelte`, `src/lib/graph/cinema/*`,
  `observatory/types.ts` NodeState (64 bytes), the 8 existing observatory
  shaders, the Graph field. REUSE the engine, don't fork it.
- Every visual primitive carries real `Provenance` (memory/event/receipt/pair/
  trace/pr/pattern/scalar). Run `assertProvenance(scene)` in dev. NO
  Math.random() as a semantic input. The Math.random() discipline test governs.
- Reuse `cognitive-palette.ts`. magenta `#FF2DF7` = RSB retrograde causality
  ONLY. indigo `#7C6CFF` = bitemporal ONLY. scarlet/immune reds for
  contradiction/veto/suppression. blackwater `#020307` base, never purple.

## WEBGPU TRAPS (all found live in Organ 1/3 — DO NOT repeat)
- T3: field ping-pong textures are `rgba16float` with usage `RENDER_ATTACHMENT
  | TEXTURE_BINDING` (NO STORAGE_BINDING). Do separable blur as FULLSCREEN
  RENDER PASSES, never `texture_storage_2d<...,write>` + textureStore (not
  portable). The whole field pipeline = render passes: splat(additive) ->
  blur-H -> blur-V -> membrane. Copy reasoning-theater-pass.ts.
- T6: split WGSL into per-pipeline modules. A render pipeline's module must NOT
  declare `var<storage, read_write>` or write storage textures (compute-only).
  Render-stage storage buffers are `var<storage, read>` only.
- WGSL reserved words: NEVER name a var OR STRUCT FIELD `active`, `meta`,
  `filter`, `common`, `sample`, `override`, `enable`, `const`, `handle`,
  `input`, `output`, `texture`, `binding`, `access`, `layout`. (`active` broke
  Organ 1, `meta` broke Blackbox — both found live.) The WGSL reserved list is
  long — pick descriptive safe names like `beat`/`info`/`data2`.
- STRUCT-FIELD MATCH: every `struct.field` you reference in WGSL must EXIST in
  the struct (Organ 3's bug was `p.signals` vs a struct declaring `meta`). TS
  build does NOT catch this.
- FramePass sequencing: the engine calls ALL `compute(encoder, frame)` then
  opens ONE main HDR scene pass and calls ALL `render(pass, frame)`. So a field
  pass encodes its own offscreen splat + blur render passes INSIDE
  `compute(encoder)` (it gets a `GPUCommandEncoder`), then draws the membrane in
  `render()`. Route field textures need their own `ensure(w,h)` on resize.
- Pipeline-layout scoping (web-verified vs W3C spec): a pipeline's explicit
  layout need only cover the bindings the SELECTED entry point statically
  reaches — a shared multi-entry-point module is fine. Layout must be a SUPERSET
  of the used bindings. Watch texture sampleType `float` vs a filtering sampler
  (common creation error).
- One-shot backend events (e.g. DeepReferenceCompleted) become LABELED
  deterministic replays, never fake streaming. A stage/element lights only if
  it has real output.

## LIFECYCLE CONTRACT (every organ ships all of it)
loading / ready / stale / empty / error / reduced-motion / paused. WebGPU
absent -> DOM/SVG fallback, NEVER a black canvas. Adapter-null -> honest metric
snapshot. RouteStage already inherits pause + reduced-motion — reuse it.

## PER-CARD DELIVERABLE
1. `<organ>/<organ>-scene.ts` — adapter: real API/event -> RouteSceneModel,
   run assertProvenance. Handle the honest empty state (scene.alive=false).
2. `<organ>/<organ>-pass.ts` — the FramePass(es) implementing the hero (see the
   organ's spec in ROUND3-FINAL-PLAN §3/§4).
3. Wire `src/routes/(app)/<organ>/+page.svelte`: mount RouteStage full-bleed
   with the pass; keep DOM detail/list panels for accessibility ON TOP; wire the
   primary click to a real receipt/action.

## GATE (do NOT self-assess — RUN these; the verifier will re-run them)
- `pnpm --filter @vestige/dashboard check` -> 0 errors
- `pnpm --filter @vestige/dashboard build` -> success
Both must pass before you mark the card done. Note in your completion metadata:
the files you created, and confirm you did NOT touch protected files.

## NOTE ON LIVE-GPU BUGS
check+build CANNOT catch invalid-pipeline runtime errors. After building, if you
can, sanity-check your WGSL modules with `getCompilationInfo()`. The human
conductor (Claude) will do the final live-GPU audit against the running app.

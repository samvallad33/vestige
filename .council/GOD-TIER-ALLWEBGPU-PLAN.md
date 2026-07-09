# GOD-TIER PLAN — The All-WebGPU Cognitive OS Dashboard
### Vestige · HN launch July 14 2026 · written July 9 (T-5 days)

> **THE ONE RULE THAT OVERRIDES EVERYTHING:** The ENTIRE dashboard renders in
> WebGPU. **Text and all. Zero DOM chrome.** Every pixel a user sees — labels,
> memory content, receipt IDs, numbers, buttons, the cursor's effect — is drawn
> by the GPU into one canvas. This is not negotiable and it is the whole point:
> **a full-screen WebGPU canvas is the only artifact that survives a screen
> recording, an OS screenshot, or a phone camera pointed at the monitor — which
> is the only thing that makes people share it.** A DOM dashboard is a website.
> An all-WebGPU dashboard is footage.
>
> Sam, verbatim: *"EVERYTHING needs to be WebGPU/RawGPU so the visuals can be to
> the ABSOLUTE MAX"* and *"I WANT THIS WHOLE DASHBOARD AS WEBGPU BECAUSE THAT'S
> THE ONLY THING THAT WILL GET USERS TO SHARE THIS."*

DOM's ONLY remaining job: a single invisible accessibility mirror (offscreen
`aria-live` + focusable proxies) so screen-readers/keyboard work. It is never
seen. It carries zero visual styling. `app.css` becomes ~empty.

Branch: `feat/dashboard-live-max`. Repo: `/Users/entity002/vestige`. Reuse the
raw-WebGPU `ObservatoryEngine` + `PostChain`. Never touch Memory Cinema or the
Graph field.

---

## 0. WHY THIS IS THE SHAREABLE MOVE (the launch thesis)

Verified precedents (Vestige memory, HN Algolia, July 2026):
- ghostty-blackhole (raytraced black hole whose SIZE = context fill) went viral
  because **the visual IS a live gauge** and it's pure canvas — remixable.
- XorDev singularity shader (381 chars) → ~60M views, remixed into shadcn/godot.
- pilot flight-globes 1539pts, neal.fun, wplace.live 0→1M users in 4 days.
- **The 5 transferable mechanics:** (1) personalization — render THEIR data;
  (2) one absurd verifiable number in the title; (3) receipts — every flourish
  is a gauge of real state; (4) deterministic 12s perfect loop = remixable clip;
  (5) chain moments into one journey ending in a personal payoff.

A DOM dashboard fails all five the moment someone screen-records it (the DOM
chrome screams "web app"). An all-WebGPU organism passes all five: it looks like
a living brain, it's a single shareable surface, every glow is a real gauge, and
the 720-frame loop is already deterministic.

**The moat is the conjunction:** raw-WebGPU engine (ours) × real Vestige-only
data (FSRS decay, RSB causal receipts, contradiction pairs, suppression, live
events) × the discipline test (swap real data for Math.random and the viewer can
tell). Nobody else has all three. Rendering the TEXT in-canvas too is what makes
it uncopyable *and* unscreenshottable-as-a-webpage.

---

## 1. WHAT ALREADY EXISTS (verified in-repo, reuse — do not rebuild)

- `lib/observatory/engine.ts` — `ObservatoryEngine`: FramePass plugin system
  (`addPass`, `compute(encoder)` + `render(pass)`), params uniform buffer,
  deterministic `DemoClock` (720-frame loop), DPR clamp, `setPaused`,
  `preFrameHook`, `wallNowMs`, `totalFrames`, GPU picking.
- `post/post-chain.ts` — HDR chain: `SCENE_FORMAT='rgba16float'` scene → mip
  bloom → Khronos PBR neutral tonemap → grain → vignette. **Text drawn into the
  scene texture gets bloom for free** — that's how labels "glow."
- `live-bridge.ts` — real WebSocket `VestigeEvent` → GPU mutation, with the live
  lanes (liveKind/liveFrame/liveEnergy/projectionDays) + the metaball/firewall/
  dream/causal-recall machinery.
- `cognitive-palette.ts` — the invented color language (blackwater base, FSRS
  oxygen ramp, magenta=RSB-only, scarlet=immune, indigo=bitemporal, cyan accent).
- `route-scene.ts` — `RouteSceneModel` + `Provenance` + `assertProvenance` (the
  discipline test as a type constraint).
- Working organ passes to copy: `reasoning/reasoning-theater-pass.ts`,
  `contradictions/contradictions-pass.ts`, `blackbox/blackbox-pass.ts`.
- `RouteStage.svelte` — the organ shell (engine lifecycle, WebGPU fallback,
  pause, reduced-motion). **Must be rebuilt to host the text layer + input layer
  and remove all DOM panels.**

**THE ONE MISSING PIECE (the linchpin):** there is NO text pipeline in the
observatory. All-WebGPU text = we build an MSDF glyph renderer. This is the
critical path. Everything else is variation on proven passes.

---

## 2. WEBGPU PLATFORM WE CAN RELY ON (verified July 5 2026, Chrome stable)

SHIPPED (use freely): HDR **extended-range canvas** (`toneMapping:{mode:'extended'}`
+ rgba16float + display-p3 — ignitions brighter than page-white on M1 Max XDR —
the "gasp" nobody else has), subgroups (134/144/145), shader-f16 (120),
dual-source-blending (130), timestamp-query (121, dev perf HUD only),
texture-formats tier1/tier2, IMMEDIATES/push-constants (149-150).
NOT shipped (do NOT depend on): multi-draw-indirect, bindless.
Cross-browser: Safari 26 all Apple, Firefox 141+ Windows/145+ AS-Mac. **WebGPU
absent → the accessibility DOM mirror shows a static "open in Chrome/Safari for
the live view" + the last server-rendered OG still. Never a black screen.**

**Capture/share gotchas (verified, launch-critical):** WebGPU canvases IGNORE
`preserveDrawingBuffer` — snapshot must copy `getCurrentTexture()` IN THE SAME
TASK as the render (gpuweb#1781). Clip export: `mediabunny` (WebCodecs
canvas→MP4). Safari clipboard: pass `Promise<Blob>` inside `ClipboardItem`. Bake
the wordmark into the POST CHAIN (not DOM) so it survives OS screen recordings
(the Loom/tldraw viral-watermark move).

---

## 3. THE ARCHITECTURE — one canvas, four GPU layers, zero DOM chrome

Every route = `ObservatoryEngine` rendering into the HDR scene texture, composed
in this Z order, all in WebGPU:

```
  L0  BLACKWATER FIELD      full-bleed metaball/organism substrate (the "tissue")
  L1  ORGAN PASS(es)        the route's bespoke hero (chambers/synapse/rings/…)
  L2  MSDF TEXT LAYER       every label/number/content, drawn in-scene (gets bloom)
  L3  INTERACTION LAYER     GPU cursor field, click shockwaves, hover ignition
  →   POST CHAIN            bloom + tonemap + grain + vignette + baked wordmark
```

Input: ONE transparent full-canvas hit-layer. Pointer/keyboard events →
CPU hit-test against a GPU-picking ID buffer (already have picking) OR a CPU
rect/quadtree of the laid-out text/controls → drive `live-bridge` + API calls.
No `<button>`, no `<div>` panel. A "button" is an MSDF label + a pickable quad +
a click-shockwave. Selection/focus is a GPU highlight, not a CSS outline.

Accessibility mirror (offscreen, visually `sr-only`, never rendered): mirror the
scene model into `aria-live` regions + focusable proxies so keyboard/SR users
get the data and can trigger the same actions. This is the ONLY DOM.

---

## 4. THE UNIFYING VISUAL LANGUAGE (locked — "Causal Bioluminescent Cortex")

A dark local brain in a jar; routes are organs of one organism. From
`cognitive-palette.ts` (source of truth):
- **Base:** blackwater `#020307`. Never purple. Ever.
- **Retention = oxygen:** luciferin `#E9FFB7` (fresh) → healthy `#A8FF5E` →
  amber-debt `#8A4B18` → sediment `#2A160B` (forgotten). Radius = √stability.
- **Cyan `#22C7DE`** = the single interactive accent (focus, hover, selection).
- **Scarlet `#FF3B30`/`#B90D2B`** = immune (contradiction/veto/suppression scar).
- **Magenta `#FF2DF7`** = RSB retrograde causal axon ONLY. Never chrome.
- **Indigo `#7C6CFF`** = bitemporal transaction-time parallax ONLY.
- **Text is bioluminescent, not chrome:** labels are **etched glowing scars** —
  they materialize glyph-by-glyph when backed by a real memory/event/receipt,
  colored by their semantic (a memory's label glows at its retention color; a
  veto reason glows scarlet; a causal receipt id glows magenta). Text drawn into
  the HDR scene BEFORE bloom, so it glows like the tissue.

Motion grammar: chemotaxis, elastic axons, immune clamping, retrograde firing,
metabolic breathing, sedimentation, scar persistence, **click-as-incision**
(every click cuts/probes real tissue → receipt-backed wave).

---

## 5. THE CRITICAL PATH — MSDF Text Engine (build FIRST, everything needs it)

Reference: Red Blob "SDF Fonts" + WebGPU-samples `textRenderingMsdf`
(`msdfText.wgsl`, median-of-RGB distance, screen-space width via `fwidth`/
derivatives, smoothstep alpha). Verified real, spec-current.

Deliverables:
1. **Checked-in MSDF atlas** for ONE crisp mono font (JetBrains Mono / IBM Plex
   Mono). Generate offline with `msdf-atlas-gen` → a PNG atlas + JSON
   (`planeBounds`/`atlasBounds` per glyph). Commit both to
   `apps/dashboard/static/msdf/`. NEVER generate at runtime (toolchain risk).
2. `lib/observatory/text/msdf-atlas.ts` — loads the atlas PNG as a `texture_2d`
   + parses the JSON into a glyph metrics map.
3. `lib/observatory/text/text-layer.ts` — a `FramePass`. CPU lays out strings →
   per-glyph instance buffer: `{glyphRect, atlasRect, worldAnchor,
   semanticColorRGBA, ageFrame, confidence}`. GPU draws one instanced quad per
   glyph, samples the MSDF atlas, applies `smoothstep` AA, tints by
   `semanticColor`, and (signature move) reveals glyph-by-glyph over `ageFrame`
   for the "etched scar materializes" effect. Renders into the scene texture
   (pre-bloom) so text glows.
4. `shaders/msdf-text.wgsl.ts` — the vertex (place glyph quad at worldAnchor +
   glyphRect) + fragment (median MSDF distance → alpha → semantic color * glow).
   Single module; render-only; storage buffers `var<storage, read>` (trap T6).
   No reserved-word struct fields (traps: not `meta`, not `active`).
5. A `layout` helper: wrap/truncate/align strings to a world-space box, return
   glyph instances. This replaces every DOM `<p>`/`<span>`/label.

Acceptance: a test route renders "hello · 5de3e41f · trust 51%" as glowing
in-canvas text, selectable via pick, materializing glyph-by-glyph, tinted by
semantic color. Once this lands, all text everywhere is this layer.

---

## 6. THE INTERACTION ENGINE (every click means something, in-canvas)

`lib/observatory/interaction/` — reuse the existing GPU picking + click-shockwave:
- **Cursor field:** the pointer is a soft light in the tissue; nearby cells lean
  toward it (chemotaxis) — the field *flinches* on movement. Pure GPU, cursor
  pos as a uniform.
- **Hover = ignition:** hovering a pickable object (memory/event/receipt) ignites
  it + its real neighbors (spreading activation), and its MSDF label
  materializes. Drives off the pick ID buffer.
- **Click = incision + receipt:** click fires a shockwave from the exact object;
  for a real action (suppress/promote/merge/appeal) the wave + state change play
  AFTER the API succeeds; for inspect, a receipt panel (MSDF text) etches in.
  No optimistic fake pulses.
- **Controls are organisms:** a "button" = MSDF label + pickable quad + hover
  ignition + click shockwave. A slider (e.g. duplicate threshold, forgetting
  horizon) = a draggable pickable node that physically moves the field.
- Keyboard: the accessibility mirror's focusable proxies map 1:1 to pickable
  objects; Tab moves a GPU focus-ring (cyan) through them; Enter = click.

---

## 7. PER-ROUTE ORGANS (all-WebGPU, real data, discipline test governs)

Each = `RouteStage` mounting: L0 field + L1 organ pass + L2 text + L3 interaction.
The route's `*-scene.ts` adapter turns real API/events → `RouteSceneModel`
(`assertProvenance`). DONE organs need their DOM panels replaced by the text
layer.

**Flagships (highest share value, do first):**
1. **Reasoning — Eight-Stage Thought Organ.** `/deep_reference` replays as 8
   glowing chambers; evidence cells flow chamber→chamber on compute Bezier
   splines; a stage lights only if it has real output; contradiction/supersession
   interrupts cut the path; the answer + evidence render as MSDF scars in-scene.
   Click a chamber → its receipt etches in. *(pass exists — swap DOM answer panel
   for MSDF text.)*
2. **Blackbox — Agent Flight Recorder.** Real `/traces` as a nervous-system
   trace: tool calls = impulses on lanes, retrievals = green branches,
   suppressions = red clamps, writes = cell births, vetoes = immune gates,
   receipt IDs = MSDF beads. Run list + event log rendered in-canvas. *(pass
   exists — the run list + event log must become MSDF text, not DOM.)*
3. **Contradictions — Immune Synapse Arena.** Dual-channel signed metaball field;
   higher-trust membrane thickens; unresolved pairs spark scarlet arcs; the pair
   text + verdict etch in scarlet. *(pass exists.)*
4. **Duplicates — Synaptic Fusion Chamber.** Similarity slider physically pushes
   twin cells together (neck = smoothstep(0.78,0.98,sim)); mismatch tokens =
   glowing filaments (MSDF); real merge fuses nuclei + receipt ring. *(pass
   exists.)*
5. **Timeline — Bitemporal Growth Rings.** Cut cross-section of the brain:
   valid-time rings + indigo transaction-time shadows; supersessions cut seams;
   MSDF date ticks engraved in the rings. *(pass exists.)*

**Shared effect:** RSB retrograde magenta axon (research D3) — reusable pass,
used by Reasoning + Blackbox; target→cause backward wavefront + permanent
cause-latch brightening. Magenta lives only here.

**Sweep organs (same L0 field + text + real-metric adapter, not bespoke heroes):**
feed (event bloodstream), schedule (forgetting-debt orrery), patterns
(cross-project mycelium/physarum), memories (cellular atlas), explore
(chemoattractant probe), importance (salience furnace), dreams (REM forge over
the live dream storm), intentions (future-tense germline seeds), memory-prs
(immune review capsules), stats (metabolic vitals). Each: `*-scene.ts` adapter +
the shared field + MSDF readouts. Nothing static, nothing purple, nothing DOM.

**The graph route + Memory Cinema:** UNTOUCHED (already all-WebGPU + protected).

---

## 8. THE SHARE LOOP — "PLANT A THOUGHT" (the viral wedge, all-WebGPU)

One public URL, no install. Type a thought/name/handle. Your text is
simultaneously the **seed** (phantomBrain: deterministic one-of-a-kind
constellation from your text) AND the **label** (MSDF-rendered — your words
glowing in the void). The birth engine fires: dust converges → your thought
ignites in thin-film iridescence brighter than page-white on XDR → edges engrave
to a phantom constellation → settles to your mind-signature. 12s seamless loop,
footage-grade through the post chain. Share mints
`vestige.dev/mind?t=your-thought` (deterministic replay forever + OG still from
`getCurrentTexture` same-task copy). Every share visually unique by construction.
End-card (MSDF): *"This is a toy mind. Vestige renders your agent's REAL one."* →
install → the real Observatory. This reuses the birth choreography + seeded
determinism that already exist; new work = text→seed+label + share/OG mechanics.

---

## 9. THE HARD TRAPS (all found live — every worker card carries these)

- **T-DOM:** ZERO visible DOM. If a route renders a styled `<div>`/`<p>`/`<button>`
  the user can see, it's a RED card. Text = MSDF layer. Controls = pickable
  quads. The only DOM is the invisible a11y mirror.
- **T3 (rgba16float):** field ping-pong = `rgba16float` render-attachment
  textures; separable blur = FULLSCREEN RENDER PASSES; NEVER
  `texture_storage_2d<...,write>`+textureStore (not portable). Copy the proven
  field pipeline from reasoning-theater-pass.ts.
- **T6 (module split):** a render pipeline's WGSL module must not declare
  `read_write` storage or write storage textures (compute-only). Render-stage
  storage buffers are `var<storage, read>`. Split modules per pipeline.
- **WGSL reserved words:** never a var OR struct field named `active`, `meta`,
  `filter`, `sample`, `texture`, `binding`, `common`, `override`. (`active` broke
  Organ 1; `meta` broke Blackbox.) Pick names like `beat`/`info`.
- **Struct-field match:** every `struct.field` referenced in WGSL must EXIST in
  the struct. TS build does not catch this.
- **Pipeline-layout scoping (web-verified vs W3C spec):** a pipeline's layout
  need only cover the SELECTED entry point's statically-reached bindings; a
  shared multi-entry-point module is fine. Watch texture sampleType `float` vs a
  filtering sampler (creation error).
- **FramePass order:** engine calls ALL `compute(encoder)` then ONE main scene
  pass calls ALL `render(pass)`. Field passes encode their own offscreen
  splat+blur inside `compute(encoder)`; draw membrane + text in `render()`.
- **One-shot events = labeled deterministic replay,** never fake streaming.
- **Capture:** snapshot copies `getCurrentTexture()` in the same task (WebGPU
  ignores preserveDrawingBuffer). Wordmark baked in the post chain.

## 9b. LIFECYCLE CONTRACT (every organ, all in-canvas)
loading / ready / stale / empty / error / reduced-motion / paused — all rendered
as GPU states (a "loading" organ breathes dim; "empty" = calm blackwater +
honest MSDF line; "error" = scarlet pulse + MSDF message). WebGPU absent → the
a11y DOM mirror + a static message + last OG still. Reduced-motion freezes
ambient drift but KEEPS discrete event pulses. Persistent GPU pause control.

## 9c. PROTECTED (never touch)
`MemoryCinema.svelte`, `graph/cinema/*`, the Graph field's 64-byte NodeState +
the 8 existing observatory shaders. Reuse the engine; don't fork it.

## 9d. GATE (every card, run — don't self-assess)
`pnpm --filter @vestige/dashboard check` = 0 errors; `build` succeeds; **AND the
CONDUCTOR does a live-GPU audit** (Preview console + getCompilationInfo +
screenshot) — because check/build cannot catch invalid-pipeline runtime errors
or "still looks like slop." "Renders with real data" ≠ "looks category-of-one" —
hold BOTH bars.

---

## 10. BUILD ORDER — 5 days to launch (T-5)

**Day 1 (today) — THE FOUNDATION (blocks everything):**
1. MSDF text engine (§5): atlas + msdf-atlas.ts + text-layer.ts + msdf-text.wgsl
   + layout helper. This is the critical path — nothing all-WebGPU works without
   it. De-risk it FIRST with a one-line test route.
2. Rebuild `RouteStage.svelte`: remove ALL DOM panels; host L0 field + L1 organ +
   L2 text + L3 interaction + the invisible a11y mirror. One canvas.
3. Interaction engine (§6): cursor field + hover ignition + click shockwave +
   pickable-quad "controls", reusing GPU picking.

**Day 2 — CONVERT THE 3 DONE FLAGSHIPS TO ALL-WEBGPU:**
Reasoning, Contradictions, Blackbox already have organ passes — strip their DOM
panels, render all their text/controls via the MSDF + interaction layers. These
become the proof the all-WebGPU pattern works end-to-end. Live-GPU audit each.

**Day 3 — THE OTHER 2 FLAGSHIPS + RSB axon:**
Duplicates + Timeline organs, all-WebGPU. Build the shared RSB retrograde-axon
pass; wire into Reasoning + Blackbox.

**Day 4 — THE SWEEP ORGANS (parallel swarm, all-WebGPU):**
The 10 remaining routes through the same shell: L0 field + MSDF readouts + real
adapter. GPT-5.5 swarm (gateway dispatcher, verifier gate) — cards MUST say "all
WebGPU, zero DOM, spawn subagents, build the files, gate." Conductor live-audits
each.

**Day 5 — "PLANT A THOUGHT" + SHARE + POLISH:**
The public share page (§8), OG-still capture, mediabunny clip export, baked
wordmark, the extended-range HDR "gasp", the mega-PR, final live-audit sweep of
every route (no DOM, no purple, gates green, looks insane). Launch July 14.

---

## 11. HOW WE EXECUTE (Opus conducts, GPT-5.5 builds, Vestige remembers)

- **Opus = conductor/auditor:** recall Vestige FIRST before every non-trivial
  move (the failure this session was substituting faded memory for the record);
  set precise cards; live-GPU audit every organ; protect Cinema/Graph field;
  hold the visual bar; commit only what's verified.
- **GPT-5.5 = builder** on `--yolo`, xhigh, spawning its own `delegate_task`
  subagents; for design/visual work give it the `claude-design` skill +
  computer-use so it iterates by LOOKING at the real page. Swarm (kanban +
  gateway dispatcher) for parallel organs; verifier profile gates on real exit
  codes.
- **Vestige = memory:** every decision/correction/finding saved the moment it
  lands; recall before diagnosing. The dashboard IS the demo of this — it must
  never again be built against a forgotten decision.
- **ONE THING AT A TIME.** Stop the swarm before switching workstreams. No
  parallel contradictory builds.

---

## 12. THE ACCEPTANCE TEST (are we done?)
Open any route. Screen-record it on a phone. Does the recording look like a
living brain nobody else has — or like a web dashboard? If any DOM chrome, any
purple, any static panel is visible, it's not done. Every glyph glows, every
click cuts tissue and returns a receipt, every color means a real Vestige-only
fact, and the whole thing is one shareable WebGPU surface. That — and only that —
is the launch.
```
```

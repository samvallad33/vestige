The prompt is the deliverable — I'll return it directly as my final response (the parent agent parses my text output, not a file). Here it is, ready to paste:

---

# BUILD BRIEF: The Vestige Dashboard is a Living Nervous System With Receipts

You are making the Vestige main dashboard the single most uncopyable memory-product UI in existence, for the July 14 HN launch. Read this whole brief before touching code. Every move here is verified against the real repo — file paths and line references are load-bearing, not suggestions.

## THE THESIS (this is the whole point — internalize it)

The dashboard is not "a graph plus some pretty background." It is a **live nervous system with receipts**: every pixel of motion is driven by a real cognitive event the Vestige backend computes and broadcasts, and *nothing else in the category can do this*. Competitors are either a static force-directed hairball (Obsidian, Cognee, Logseq) or a stack of text panels (Letta, mem0 v3, Zep). Not one renders per-memory forgetting state, live decay, causal receipts, or a contradiction firewall — because none of them have the substrate. Vestige uniquely (a) *computes* the data (per-memory FSRS state, RSB causal backfill, bitemporal state, contradiction pairs, suppression-with-reason), (b) *broadcasts* it live over WebSocket (~20 distinct `VestigeEvent` variants), and (c) already *owns* a raw-WebGPU engine (8 WGSL shaders, 5 renderers, an HDR post-chain with mip bloom + Khronos PBR tonemap). That three-way overlap exists nowhere else on Earth.

**The discipline test, tape it to the monitor.** For every visual you build, ask: *"If I swapped the real backend value for `Math.random()`, could the viewer tell?"* If NO, it is a screensaver and a competitor ships it in a weekend — do not build it as a hero. If YES, it is the moat. Curl-noise-behind-glass fails this test. Live FSRS decay, causal wavefronts, and the contradiction firewall pass it. **Build the data-backed moves as heroes; ship exactly one cheap ambient base coat and stop.**

## WHY NOBODY ELSE CAN BUILD THIS (the exact substrate dependencies)

The engine is NOT the moat — curl noise, radiance cascades, reaction-diffusion are all public repos. The moat is the **data feeding the engine**, and every hero below is load-bearing on a Vestige-only substrate:

- **Per-memory FSRS state** — `stability` + `last_accessed` on every node (`crates/vestige-core/src/node.rs`), and the closed-form retrievability curve `r=(1+factor*elapsed/stability)^(-w20)` (`crates/vestige-core/src/fsrs/algorithm.rs:101`, live personalized `w20=0.01036`). mem0 stores atomic facts with no decay; Zep has bitemporal *validity* booleans but no forgetting curve; Letta has char-count limits. To render a decay field a competitor first rebuilds spaced-repetition scheduling into their core — a two-quarter project before frame one.
- **RSB causal backfill** — `crates/vestige-core/src/advanced/retroactive_backfill.rs` produces real backward-only, shared-entity causal paths. `ConnectionType::Causal` exists (`chains.rs:59`). No competitor computes causation; they store undirected co-occurrence.
- **Suppression-with-reason** — `VestigeEvent::MemorySuppressed` carries a real `estimated_cascade` neighbor count and a `reversible_until` labile window (`crates/vestige-mcp/src/dashboard/events.rs:44-45`). Real top-down inhibitory active-forgetting. Competitors just delete.
- **Contradiction pairs** — `DeepReferenceCompleted` carries `contradiction_pairs: Vec<(String,String)>` (`events.rs:82`) from the composition engine. Nobody else has a contradiction detector wired to the graph.
- **The ~20-variant live cognitive event stream** — `MemoryCreated`, `MemorySuppressed`, `DreamStarted/Progress/Completed`, `ConnectionDiscovered`, `Rac1CascadeSwept`, etc. Competitor backends have maybe three states worth broadcasting (created/updated/deleted).

The one-line version: **the visuals below CANNOT EXIST without that data. Build them and the competition is locked out for two quarters.**

## THE ENGINE YOU'RE REUSING (do not add new deps, do not rebuild)

Everything lives under `apps/dashboard/src/lib/observatory/`:
- `engine.ts` — `ObservatoryEngine`, a `FramePass` plug-in system (`addPass`) + a params uniform buffer + the reusable HDR post-chain (`post/post-chain.ts`: rgba16float scene, mip bloom, Khronos PBR tonemap, grain, vignette).
- `shaders/` — 8 WGSL: `simulate.wgsl` (edge-spring force integrator + O(N²) repulsion + damping, lines ~121-135), `render-nodes.wgsl` (already dims by retention: `intensity *= 0.45 + 0.55*retention`, line 183), `render-edges`, `render-path`, `birth-particles`, `firewall`, `forgetting`, `rescue`.
- 5 renderers: `node-renderer.ts` (has `setPathSteps()` at :142 — the hot buffer-destroy→recreate→`createPipeline` rebuild pattern you'll clone for `setEdges`/`appendNode`), `birth-renderer`, `firewall-renderer`, `forgetting-renderer`, `rescue-renderer`.
- `ObservatoryStage.svelte` — the mount point. **It does NOT subscribe to the live WebSocket today** (verified: zero websocket refs in `lib/observatory/*`). That subscription is the central new integration work.
- `graph/+page.svelte` — dual renderer: `renderMode==='field'` mounts `<ObservatoryStage demo="recall-path">` (prop-driven, zero live events); `renderMode==='classic'` mounts `<Graph3D events={$eventFeed}>` (the old Three.js path that today owns all live-event wiring). The field is the banger; make it live.

## GUARDRAILS (non-negotiable — these make the ambition WORK, not shrink it)

1. **Reuse the observatory engine. Zero new rendering deps.** Every hero rides existing WGSL/renderers/post-chain. New work is app-code plumbing (event bridge, buffer regrow, one serializer field, one pathfinder branch), not new frameworks.
2. **Memory Cinema is untouched.** Do not reach into `graph/cinema/*` or `MemoryCinema.svelte`. It is finished and loved. All work is the graph field, the dead routes, or a sibling ambient canvas.
3. **Earned by real data, never decoration.** Every flourish maps to a real lifecycle event/metric or it does not ship. This is the discipline test above, enforced.
4. **Purple breathes, it does not sit static.** Replace flat brand purple with an FSRS-indexed perceptual ramp: high-retention = bright cyan-shifted accent (#22C7DE hot end), decaying = deep desaturated indigo, dead = near-black. Purple becomes a *retention gauge*. On the dead routes, invert to a neutral graphite substrate and spend purple + motion ONLY on live events and FSRS threshold crossings (ISA-101 / mission-control color-singleton discipline) — aliveness from contrast, not more glow.
5. **prefers-reduced-motion + WebGPU fallback + perf budget, all first-class:**
   - `prefers-reduced-motion: reduce` → freeze ambient drift/sim, but KEEP discrete event pulses (they are information). Ship a persistent on-page pause control (WCAG requires it for >5s motion).
   - WebGPU absent/`requestAdapter()` null/context-create throw → degrade to the existing flat-glass look (`EngineStatus 'unsupported'/'error'`). Never a black canvas. `navigator.gpu` existing is NOT sufficient — check the adapter.
   - DPR clamp (`min(devicePixelRatio, 1.5)` desktop, 2 mobile) as a runtime uniform. Page-Visibility + IntersectionObserver gate the rAF loop (stop GPU submits when unseen). Mobile-first particle budget; scale UP on desktop. Event-driven render when settled — calm at rest, spike on real events.
6. **Verify in the real running app with screenshots + gates green before claiming done** (see final section).

---

## BUILD ORDER — most-jaw-drop-first, each independently shippable

Execute incrementally. Each phase lands, verifies, and can ship alone. Do NOT stack all five into one unverifiable blob — land, screenshot, gate, then next.

### PHASE 0 — The live event bridge (foundation for everything; do this first)

The field is inert because `ObservatoryStage` never hears the backend. Build the bridge once; every hero rides it.

- Subscribe `ObservatoryStage.svelte` to the `eventFeed` store (`$stores/websocket`) — the same store `Graph3D` already consumes.
- Add a lock-free ingestion path: WebSocket event → engine param/buffer mutation for the current frame. Keep it allocation-free (ring buffer / typed arrays).
- This unlocks Phases 1-4. Ship nothing user-visible yet; verify with a console assertion that live events reach the engine.

### PHASE 1 — Live FSRS decay: watch memories forget (the #1 moat, hero)

Each node continuously dims/shrinks on its real FSRS retrievability curve. The shader path already exists (`render-nodes.wgsl:183` dims by retention; radius is retention-weighted in `graph-upload.ts:105`; retention lives in `vel_retention.w`). The ONE gap is data: the live-graph payload `get_graph` (`handlers.rs:1237`) ships `retention` + timestamps but NOT `stability` and NOT `lastAccessed` per node (they already exist for single-node handlers at `handlers.rs:305/345`, proving it's a serializer add, not new data).

- **Backend:** add `stability` + `lastAccessed` to the `get_graph` per-node JSON (`handlers.rs` ~1237).
- **Types:** add both to `GraphNode` (`types/index.ts:68`) + `ObservatoryNode` (`observatory/types.ts:119`).
- **Engine:** each frame, compute `retrievability(stability, (now - lastAccessed)/86400)` → write `vel_retention.w`. Pure closed-form arithmetic, no per-node server round-trip. Reuses every existing shader path.
- **Honesty (protect the demo, stay honest):** with live `w20=0.01036` + multi-day stabilities, real-time drift is imperceptibly slow over a viewing session. Ship a **scrubbable "project forward N days" control** — recompute `elapsed` at `t+N` on the same true curve. Fully honest (it IS the real curve), and it makes decay legible without faking. Or tie it to the existing `?demo=3` forgetting-horizon choreography.

### PHASE 2 — Contradiction firewall on camera (flagship live moment, hero)

A contradicted memory flares crimson and a microglial firewall arc quarantines it live. The crimson firewall VISUAL already exists (`firewall.wgsl.ts` + `firewall-renderer.ts` + `firewall-plan.ts`) but is hard-locked to the deterministic demo-clock (compute gates on `params[9]===4`, writes are pure functions of frame + loop_phase). Make it data-driven.

- Build a live `FirewallPlan` constructor from a real event's target node + its real graph neighbors (reuse the plan's fire-word structure; drive an event-triggered frame counter instead of `loop_phase`).
- Fire on two Vestige-only substrates: `MemorySuppressed` (real `estimated_cascade` + `reversible_until`, `events.rs:44`) and `DeepReferenceCompleted.contradiction_pairs` (`events.rs:82`).
- **Honesty:** the live DB has 0 standing contradictions at rest (they're produced per `deep_reference` run, not a resting pool), and suppression only fires on an actual suppress. So a live demo must **trigger a suppress or a contradiction-surfacing query on camera** rather than wait for ambient events. When it fires, every frame is earned by a real cognitive event.

### PHASE 3 — Dream consolidation storm (visible metabolic event, hero)

When the real dream pipeline streams in, the whole field enters a consolidation storm: edges rewire, clusters merge. All mechanics exist. `simulate.wgsl` (~121-135) already runs edge-spring forces + repulsion + damping every frame, so "clusters merge" is the emergent settle when new springs are added — no new physics. The server dream handler (`handlers.rs` ~1400-1491) emits a REAL sequence backed by `save_connection` DB writes: `DreamStarted{memory_count}` → `ConnectionDiscovered{source,target,weight,type}` ×N → `DreamCompleted`.

- Add `setEdges()` to `node-renderer.ts` — a `setPathSteps()` clone (`:142`): rebuild the sized-to-graph edge index buffer, bump `params.edge_count`, rebuild pipeline. This is the real work; the machinery is proven.
- On `DreamStarted`: raise a global agitation param + loosen damping (params-driven, no shader rewrite).
- Append each `ConnectionDiscovered` pair to the edge buffer live — springs auto-pull the clusters together.
- On `DreamCompleted`: settle back.
- **Honesty:** `DreamProgress` is defined + handled client-side but is NEVER emitted server-side. Ride `DreamStarted` + the `ConnectionDiscovered` burst + `DreamCompleted` — do not depend on Progress.

### PHASE 4 — RSB causal recall wavefront (scoped, hero — ship the honest version)

Recall lights the backward causal path through the graph. The backward wavefront ALREADY renders on the field: `ObservatoryStage demo="recall-path"` → `node-renderer` calls `buildRecallPath` (`path-builder.ts`) → `PathStep` buffer scanned by `simulate.wgsl`. Raw WebGPU, no library. Three gaps make the literal "every recall, real causal edges" pitch false today — close them, don't over-claim:

- **Real trigger:** a live recall emits `SearchPerformed{result_ids}` and `DeepReferenceCompleted{primary/supporting/contradicting_ids}`, but the graph handler just random-pulses all nodes. Wire a new `BackfillFired`/`CausalReceipt` WebSocket event (carrying the backward causal path + shared-entity join keys) and drive `buildRecallPath` from it instead of the DemoClock.
- **Real causal edges:** `planCinemaPath` (`cinema/pathfinder.ts`) is a greedy edge-weight walk — add a branch that prefers `ConnectionType::Causal` edges (the type exists, `chains.rs:59`).
- **Scope discipline:** ship "recall lights the backward causal path" on the graph page FIRST (highest EV, engine already there, wired to the real event so the wavefront is *earned*, not scripted). Then generalize the recall-wavefront to the other routes as they get the canvas.
- **Marketing honesty:** do NOT claim "nobody does causal/backward memory" (Datadog Watchdog / Pernosco / git-bisect exist). The uncopyable claim is the *conjunction as persistent memory* — RSB's failure-triggered, backward-only, shared-entity causal promotion + per-memory FSRS + the causal edge type + the raw-WebGPU engine. Do not market "every recall reaches backward" until the trigger + causal pathfinder land.

### PHASE 5 — The one ambient base coat (cheap, ONE day, then STOP)

Kill "purple and static" on the 10 dead routes with a single ambient WGSL `FramePass` behind them, added to the observatory engine and composited through the existing bloom. `engine.ts` exposes `addPass`; the post-chain is core WebGPU (no feature gate); it already degrades to flat-glass on `unsupported`/`error`.

- One shared full-bleed field, per-route metric-uniform block, driven by the already-live APIs (`stats`/`health`/`retentionDistribution`/`timeline`/`duplicates`/`contradictions`/`crossProjectPatterns`/`importance`/`intentions` in `lib/stores/api.ts`).
- **Acceptance gate is "the field legibly READS the backend," not "the noise looks alive."** Endangered-count storms the decay field; contradiction pairs visibly fracture it; due-for-review pulses it. If it ships as generic pretty noise loosely tinted by one metric, it is copyable, off-thesis, and violates the earned-by-data rule — cut it.
- **Do not exceed one day or one substrate here.** Curl noise / reaction-diffusion / physarum are all public repos; behind a card at low opacity the metric-binding is invisible, so extra investment subsidizes a competitor's catch-up. This is a base coat, not a hero. The dead routes (`stats`, `memories`, `timeline`, `patterns`, `duplicates`, `contradictions`, `intentions`, `importance`, `explore`, `schedule`) are the target; render once via single-canvas + `setViewport`/`setScissorRect` (a raw-WebGPU advantage Three.js dashboards drop).

---

## EXPLICITLY OUT OF SCOPE (do not spend launch week here)

Per adversarial teardown, these fail the `Math.random()` test or are `npm install`-able — a competitor ships them in three weeks:
- Radiance-cascade GI ("importance is a light source") — highest effort-to-legibility ratio; the penumbra looks identical whether the light is `importance_score` or a random float. The importance signal is already served by the existing emissive-mask bloom. **Defer past launch.**
- More than one reaction-diffusion / physarum / metaball substrate — the drive is unverifiable, so it's wallpaper. One base coat max (Phase 5).
- ChartGPU as a "moat" — it's an MIT library; use it if useful but it's a commodity, not a differentiator.
- MSDF text, View Transitions, dual-Kawase bloom retune, GPU picking polish — good hygiene, background chores, never headline features.
- Birth-on-write as a hero — the pixels ("sparkles fly in on a websocket event") are a generic dataviz trope, copyable in a day, and today's `MemoryCreated` payload lacks position/neighbors/retention so a naive live birth would invent the neighborhood (decoration). It only becomes uncopyable if it resolves the new node's REAL retention-ranked neighbors and engraves true incident edges. Defer it behind the four data-backed heroes; if built, do the engine extension honestly (`appendNode` with buffer regrow + pipeline rebuild, reusing the `setPathSteps` machinery + `birth-plan` golden-angle placement) and resolve real associations — never a firework.

---

## VERIFICATION — required before you claim any phase done

Run the real app and show proof. No "done" without it.

1. `pnpm --filter @vestige/dashboard build && pnpm --filter @vestige/dashboard check` — green.
2. `cargo test --workspace --no-fail-fast && cargo clippy --workspace -- -D warnings` — green (for the backend serializer + event changes).
3. Launch the running dashboard, connect the live backend, and for each hero: trigger the real event (project-forward decay scrub; on-camera suppress/contradiction query; a real `dream` run; a real recall) and **capture screenshots proving the field responds to real data**, not the demo clock. Confirm the `Math.random()` test by eye: the motion must be legibly wrong if you fed it noise.
4. Confirm graceful degradation: force WebGPU off → flat-glass, never black. Toggle `prefers-reduced-motion` → ambient stops, event pulses remain, pause control present.
5. Confirm Memory Cinema is byte-for-byte untouched.

Save each landed phase (decision/finding) to Vestige as it lands. Land most-jaw-drop-first: Phase 0 bridge → Phase 1 live decay → Phase 2 firewall → Phase 3 dream storm → Phase 4 causal wavefront → Phase 5 base coat. Ship the ones that pass. Let's go.

---

**File paths returned for the executing session (all absolute):**
- Engine: `/Users/entity002/vestige/apps/dashboard/src/lib/observatory/engine.ts`
- Mount + event bridge target: `/Users/entity002/vestige/apps/dashboard/src/lib/observatory/ObservatoryStage.svelte`
- Node renderer (`setPathSteps` → clone for `setEdges`/`appendNode`): `/Users/entity002/vestige/apps/dashboard/src/lib/observatory/node-renderer.ts`
- Shaders: `/Users/entity002/vestige/apps/dashboard/src/lib/observatory/shaders/` (`render-nodes.wgsl.ts:183`, `simulate.wgsl.ts:121`, `firewall.wgsl.ts`)
- Path/firewall/plans: `path-builder.ts`, `firewall-plan.ts`, `graph/cinema/pathfinder.ts` (add Causal branch), `graph-upload.ts:105`
- Graph route (dual renderer): `/Users/entity002/vestige/apps/dashboard/src/routes/(app)/graph/+page.svelte`
- Types: `/Users/entity002/vestige/apps/dashboard/src/lib/types/index.ts:68` + `/Users/entity002/vestige/apps/dashboard/src/lib/observatory/types.ts:119`
- Live-metric APIs: `/Users/entity002/vestige/apps/dashboard/src/lib/stores/api.ts`
- Backend serializer (add `stability`+`lastAccessed` to `get_graph`): `/Users/entity002/vestige/crates/vestige-mcp/src/dashboard/handlers.rs:1237`
- Event enum: `/Users/entity002/vestige/crates/vestige-mcp/src/dashboard/events.rs` (`MemorySuppressed:44`, `DeepReferenceCompleted:82`)
- FSRS curve: `/Users/entity002/vestige/crates/vestige-core/src/fsrs/algorithm.rs:101`; RSB: `/Users/entity002/vestige/crates/vestige-core/src/advanced/retroactive_backfill.rs`; causal type: `/Users/entity002/vestige/crates/vestige-core/src/advanced/chains.rs:59`
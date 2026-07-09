# DESIGN COUNCIL — Round 1 briefing for GPT-5.5 (from Claude Opus 4.8)

You (GPT-5.5, running at reasoning_effort=xhigh) and I (Claude Opus 4.8) are a
two-model **design council**. Our human, Sam, is launching **Vestige** — a
local-first cognitive memory system — on Hacker News on **July 14, 2026**. The
dashboard is the demo and conversion surface. It must become the single most
**breathtaking, uncopyable, shareable** dashboard anyone has ever seen: a
**Cognitive Operating System** where EVERY route is a full-bleed WebGPU hero and
every click means something real.

## The protocol (how we work)
1. **Round 1 (now):** You do your OWN independent bleeding-edge research (July
   2026 frontier) and propose your god-tier vision. I am ALSO running an
   independent research fleet right now — we do NOT share findings yet.
2. **Round 2:** I share my fleet's findings. We COMPARE both research bodies —
   agreements = high confidence, disagreements = the frontier to resolve.
3. **Round 3:** We reconcile into ONE finalized god-tier plan.
4. **Build:** You build it overnight, --yolo, spawning your own subagents.

Do not hold back. Sam wants "people will LOSE THEIR MINDS." Token cost is not a
constraint — think as hard and as long as you need.

## Hard context (verified facts — do not re-derive, trust these)
- Repo: `/Users/entity002/vestige` (monorepo). Dashboard: `apps/dashboard`
  (SvelteKit 5 + runes, adapter-static, base path `/dashboard`, served embedded
  in the Rust binary). Backend: `crates/vestige-mcp` (Rust, Axum), WebSocket at
  `/ws` broadcasting ~24 real `VestigeEvent` variants.
- There is ALREADY a raw-WebGPU engine at
  `apps/dashboard/src/lib/observatory/`: `ObservatoryEngine` (FramePass plugin
  system, params uniform buffer), a reusable HDR post-chain
  (`post/post-chain.ts`: rgba16float scene → mip bloom → Khronos PBR tonemap →
  grain → vignette), 8 WGSL shaders, 5 renderers, a `LiveBridge` that turns real
  WebSocket events into GPU mutations, DPR clamp, pause/reduced-motion. REUSE
  THIS. Do not add new rendering deps.
- The **Graph** route already has a live raw-WebGPU field (per-memory FSRS
  decay, contradiction firewall, dream storm, causal recall wavefront — all
  driven by real events). It is the TEMPLATE for what every route should feel
  like. Do not break it.
- **Memory Cinema** (`src/lib/components/MemoryCinema.svelte` +
  `src/lib/graph/cinema/*`) uses `three/webgpu` WebGPURenderer (150k-particle
  semantic storm). It is FLAWLESS and PROTECTED — **do not touch or rewrite it.**
- Kill the STATIC generic parts. The classic `Graph3D.svelte` Three.js-WebGL
  renderer + `src/lib/graph/{nodes,edges,effects,force-sim,particles,scene,
  dream-mode}` are replaceable (the observatory field is now the main graph).
- **Routes to transform** (currently flat purple-on-black text panels, "boring
  generic AI SaaS"): reasoning (Reasoning Theater — an 8-stage local cognitive
  pipeline: retrieval→rerank→activation→trust-score→supersession→contradiction
  →relations→chain), timeline, feed, schedule, duplicates, contradictions,
  patterns, memories, explore, importance, activation, dreams, intentions,
  blackbox (agent flight recorder), memory-prs (cognitive immune system), stats,
  settings.

## THE DISCIPLINE TEST (non-negotiable, this is the moat)
For every visual: "If I swapped the real backend value for Math.random(), could
the viewer tell?" If NO → it's a screensaver a competitor ships in a weekend,
DO NOT build it as a hero. If YES → it's the moat. Vestige uniquely computes
per-memory FSRS forgetting state, RSB causal backfill, bitemporal state,
contradiction pairs, suppression-with-reason, and broadcasts them live. Every
hero must be load-bearing on that Vestige-only substrate.

## COLOR/MOTION mandate (from Sam, verbatim)
NOT purple-breathe, NOT per-route-accent, NOT graphite — those are too safe.
Invent **something BRAND NEW that has never been shipped in a dashboard**.
Combine novel ideas into ONE coherent system so the OS reads as one organism
with distinct organs. Category-of-one.

## YOUR ROUND-1 DELIVERABLE (write it to `/Users/entity002/vestige/.council/gpt55-round1.md`)
1. **Independent frontier research** — use your web tools + delegate subagents to
   find the July-2026 bleeding edge of WebGPU dataviz: compute particle systems,
   SDF/metaball merging, volumetric/raymarched light transport, reaction-
   diffusion / Lenia / neural-cellular-automata / physarum, MSDF living text, GPU
   interaction/shockwave feedback, WebGPU-specific 2026 features (subgroups,
   timestamp queries, multi-draw indirect, etc.). Cite sources with dates. Be
   specific at the WGSL level.
2. **The invented unifying visual+motion LANGUAGE** — the never-shipped idea:
   the metaphor, exact color/light encoding (hex values + the rule mapping FSRS
   state / trust / event-type to visuals), the motion grammar. Justify why it's
   category-of-one.
3. **Per-route hero spec** — for each route above: the hero concept, the REAL
   data signal that drives it (must pass the discipline test), the frontier
   technique it uses, and what the primary click MEANS.
4. **Build order** — most-jaw-drop / most-shareable first, grouped into shippable
   increments, and the shared engine extensions to build once (reusable
   RouteStage, MSDF text layer, click-shockwave system).
5. **The ONE thing to prototype first** to de-risk the whole system.

Save your deliverable to that file. Be bold, specific, and buildable. LFG.

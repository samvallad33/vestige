# DEEP RESEARCH BRIEF — Transform a live WebGPU "memory operating system" dashboard into the most visually stunning thing on the web (bleeding edge, JULY 2026)

> Paste this whole brief into your deep-research agent (ChatGPT Deep Research, Gemini Deep Research, Perplexity, Grok DeepSearch, Claude Research, etc.). It is self-contained. Do the research; return the deliverable exactly as specified at the end.

## 0. Your mission, in one sentence

We are shipping **Vestige** — a local-first AI *memory operating system* with an embedded, entirely GPU-rendered dashboard — in **~30 hours from now (13 July 2026)**. Find the **absolute bleeding edge of real-time web graphics as of THIS MONTH (June–July 2026)** and tell us, concretely, how to fuse those brand-new techniques with our exact engine and our exact data to make something **nobody else has or can easily copy** — jaw-dropping, and *meaningful* (every pixel must encode a real cognitive/memory signal, never decoration).

## 1. RECENCY IS THE ENTIRE POINT — non-negotiable rules

- Today is **13 July 2026**. Prioritize things that are **brand new as of the last 4–6 weeks (≈ 15 Jun – 13 Jul 2026)**. Second priority: anything from 2026. **Discard pre-2025 material** unless it is the *direct foundation* a brand-new 2026 technique builds on (then label it "foundation for X").
- For **every** finding you MUST provide: the **date** (month + year, day if you can), the **primary source URL** (spec, release notes, paper, author's own post — not a listicle), and its **browser/library support status as of July 2026** (shipped / origin-trial / behind-flag / research-only).
- If you cannot date something to 2026, mark it **LOW-CONFIDENCE** and say why. Do not present undated training-data recall as current fact. **Verify live; cite.**
- Hunt these sources specifically: Chrome/Edge **"What's New in WebGPU"** for the current milestone(s); **Safari Technology Preview** + iOS 26 release notes; **Firefox / wgpu** release notes; the **WebGPU + WGSL spec** changelog and W3C GPU-for-the-Web GitHub; **SIGGRAPH 2026** (August) technical-papers preprints and Real-Time Live; **arXiv** cs.GR / cs.HC from the last 6 weeks; **Compute.toys**, **Shadertoy** trending, **X/Bluesky** graphics community, **Hacker News** front page + Show HN, **Awwwards SOTD / FWA / CSS Design Awards / Codrops**, **GitHub Trending** (TypeScript/Rust/WGSL), and release feeds for **three.js (WebGPURenderer/TSL)**, **TypeGPU**, **Babylon.js WebGPU**, **transformers.js / web-llm / WebNN**.

## 2. WHAT WE HAVE LOCALLY — transform THIS, not a generic engine

**Stack (exact):** SvelteKit 2.20 + Svelte 5 (runes), Vite 6, `@sveltejs/adapter-static` (ships as a static site embedded in a Rust binary), Tailwind 4. Backend: Rust + Axum, local-first, exposes a JSON `/api/*` + a WebSocket event stream. The dashboard is served at a base path and must also deploy to GitHub Pages.

**Rendering engine (exact — hand-written, NO framework for the main dashboard):**
- Raw **WebGPU**. `requestAdapter()` / `requestDevice()` are called with **NO `requiredFeatures` and NO `requiredLimits`** today → we are on the **baseline WebGPU feature set**, so opting into optional features (shader-f16, timestamp-query, etc.) is available headroom we have not spent.
- ~**10 compute pipelines + 8 render pipelines**. An **HDR offscreen scene target in `rgba16float`**, composited by a **bloom + tonemap PostChain**.
- A signature **"living field"**: additive **splat → separable blur → membrane → cell** passes (organic, breathing, biological). Storage buffers hold per-cell / per-node state; a per-frame uniform `params` block drives everything; a **deterministic fixed-60 Hz sim clock** (rAF only schedules, never drives sim) makes every frame reproducible (`?frame=N` gives identical pixels).
- **All text is GPU-rendered**: a custom **MSDF/MTSDF** text layer (JetBrains Mono atlas, premultiplied-over), aspect-corrected in-shader, with per-glyph reveal + depth-of-field + cursor-reactive swell. The dashboard is **zero-DOM** on the main organs — the only non-canvas element is a single `<canvas>` (plus a small DOM mobile-nav overlay).
- **GPU picking** via CPU-reprojection; `prefers-reduced-motion` freezes the field; **DPR-clamped** resize.
- 15 dedicated GPU pass modules already exist, e.g.: `living-field-pass`, `text-layer`, `node-renderer`, `post-chain`, `palace-node-pass` (a bespoke 3D constellation with its own close-orbit camera + CPU-reprojection picking), `reasoning-theater-pass` (an 8-stage decision-trace theater), `timeline-pass` (bitemporal growth rings), `duplicates-pass`, `contradictions-pass`, `blackbox-pass`, plus scripted cinematic renderers (`birth`, `firewall`, `forgetting`, `rescue`).
- **Protected, do not propose replacing:** "Memory Cinema" — a Three.js/WebGL2 cinematography engine (particle-storm + shot director + camera auteur) that is finished and loved.

**Targets & degradation (a technique is only useful if it fits ALL of this):**
- Desktop Chrome/Edge + **Safari 18+**, and **mobile** (iOS 26 Safari on by default; Android Chrome 121+ conditional). Must hit **60 fps on a mid-tier GPU** and stay bandwidth-sane on **mobile tile GPUs**.
- Must **degrade gracefully**: reduced-motion path, and a **no-WebGPU SVG fallback** (a real slice of phones have no WebGPU).

**The data every visual MUST encode (this is our unfair advantage — real cognitive state, ~1,346 live memories):**
Per memory we have real **FSRS-6** + dual-strength fields: `retention_strength`, `retrieval_strength`, `storage_strength`, `stability`, `difficulty`, `reps`, `lapses`, `next_review` (schedule urgency), `last_accessed`, `sentiment_score`, `sentiment_magnitude`, plus tags, source, embeddings (256-D), and a causal graph (edges with type + weight). Live events stream over WebSocket: `MemorySuppressed`, `ConnectionDiscovered`, `DreamStarted/DreamCompleted`, `DeepReferenceCompleted` (an 8-stage recall with a causal path + contradictions), `SearchPerformed`, `MemoryCreated`. The product thesis is **postdiction / retroactive salience** — "a root cause never looks like the bug it creates" — the graph reaches *backward* to surface causes a vector search misses.

**Hard rule for every recommendation:** it must map to a REAL field/event above. The litmus test: *if you swapped the driving value for `Math.random()`, the motion would be legibly wrong.* Decorative-only ideas are rejected.

## 3. Scour these frontiers (be exhaustive; date + cite everything)

For each, tell us **what is BRAND NEW as of Jun–Jul 2026**, and whether it runs on our stack:
1. **WebGPU browser/spec deltas this month** — new WGSL features, adapter features, limits, `subgroups`, `shader-f16`, `timestamp-query`, storage-texture / read-write-storage-texture, multi-draw-indirect, compute-driven vertex pulling, WebGPU in workers / OffscreenCanvas news, `navigator.gpu` on more mobile.
2. **GPU compute simulation** — particle-life / **neural cellular automata**, **GPU force-directed graph layout** at 1k–100k nodes, XPBD/position-based soft bodies, **MLS-MPM**, flow fields, reaction-diffusion, 1M+ boids — with the SOTA WebGPU implementation and fps.
3. **Volumetric / point / splat rendering** — **3D & 2D Gaussian splatting** web viewers, compute-rasterized splats, real-time **SDF raymarching**, order-independent transparency, volumetric fog / godrays.
4. **Post-processing & color** — next-gen HDR tonemapping (**AgX** and successors), bloom/lens advances, **temporal upscaling / TAA** in WebGPU, and **browser HDR / wide-gamut** (Display-P3, rec2100, `dynamic-range-limit`, HDR `<canvas>`) landing now.
5. **GPU typography** — MSDF/MTSDF advances, **on-GPU vector/curve** text (Slug-like), variable fonts on GPU, kinetic typography, **text-as-particles**, glyph morphing/decay. How to make the *memory text itself* the stunning element.
6. **Procedural shader art** — brand-new noise (gabor/curl/domain-warp), flow-field advection, iridescence/thin-film, caustics, aurora/nebula fields — link the actual Shadertoy/Compute.toys pieces posted this month.
7. **Interaction & input** — WebGPU picking tricks, pointer/pressure/tilt, **scroll-driven animations**, **View Transitions** (same- + cross-document), the new CSS scroll/anchor primitives, WebXR/hand-tracking, haptics — what makes a memory OS feel physical *right now*.
8. **Audio-reactive / generative audio-visual** — WebAudio→WebGPU compute bridges, FFT-driven fields, data sonification, any new lib this month.
9. **Libraries & tooling** — three.js **TSL / WebGPURenderer**, **TypeGPU**, Babylon WebGPU, wgpu-in-wasm, compute helpers — exact version + date + what's newly possible.
10. **Beautiful data/graph/memory viz frontier** — GPU node-link at scale, **edge bundling**, hypergraph/simplicial rendering, temporal/hive/river layouts, **in-browser UMAP/embeddings**, "living knowledge graph" art.
11. **On-device generative AI × real-time visuals** — **WebNN**, transformers.js, **web-llm**, neural fields / tiny diffusion as backdrops, LLM-driven procedural visuals at runtime. What can an *AI memory product* uniquely do here that others can't?
12. **Award-winning interactive experiences of Jun–Jul 2026** — dissect the SPECIFIC techniques behind this month's Awwwards SOTD / FWA / Codrops pieces (links + the trick).
13. **Neuroscience/cognitive visual metaphors (2026 papers/tools)** — connectome rendering, neural-avalanche / criticality, hippocampal replay, engram imaging, oscillation/phase viz — that we can translate into a memory field.

## 4. Then COMBINE — the part only we can do

From your findings, propose **fusions nobody has shipped for a memory system**: cross a brand-new sim/render/text/AI technique with our real FSRS physics (decay, retrieval, review urgency, suppression scars, contradiction firewall, dream replay, backward causal recall). For each fusion state: the 2–4 July-2026 techniques it fuses, **which real data field drives it**, why nobody's done it, rough effort (S/M/L/XL), and a 1–10 wow score. Then design **ONE signature "nobody but us" hero moment** for launch.

## 5. Deliverable — return EXACTLY this, nothing generic

**A. Frontier Ledger** — a table of your strongest ~25–40 findings: `technique | date | source URL | what's new (Jul 2026) | support status | runs on our stack? (Y/N + why)`.

**B. Ship-Now (buildable in < 1 day, on our stack)** — 5–10 concrete upgrades. For each: the technique (dated+sourced), the **exact pass/file to add or modify** given our architecture (e.g. "new compute pass feeding the additive splat", "PostChain tonemap swap to AgX", "opt into `shader-f16` in `requestDevice`"), the **real data field it encodes**, effort, wow, and mobile/no-WebGPU degradation.

**C. Signature Hero Moment** — the single most stunning, uniquely-ours launch centerpiece, described concretely enough to start coding, with the July-2026 techniques it stands on.

**D. Post-Launch Ambition** — 3–5 bigger bets (XL) worth doing after launch, with why they're moats.

**E. Kill List** — bleeding-edge things that look tempting but WON'T work on our stack / mobile / no-WebGPU, and the one-line reason each is a trap.

Rank everything by **(wow × recency) ÷ effort**. Be specific enough that an engineer can start today. Cite dates and URLs throughout. If you are unsure whether something shipped, say so — do not bluff.

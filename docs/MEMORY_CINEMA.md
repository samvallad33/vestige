# Memory Cinema — Complete Feature Reference

Memory Cinema turns your real memory graph into a directed, narrated, infinitely-
diving cinematic experience rendered as a 150,000-particle WebGPU compute storm.
It is the dashboard's signature pillar.

The whole thing is **dynamically imported only on launch** — the heavy WebGPU/TSL
bundles never load for normal dashboard use. It boots a **separate WebGPU canvas**
so the underlying WebGL graph (every current user's experience) is never touched —
zero regression by construction.

## Architecture — the 8 modules

| Module | Role |
|---|---|
| `pathfinder.ts` | Plans the narrative path through the real graph (a story, not a BFS dump) |
| `topology.ts` | Extracts graph signals (betweenness, contradictions, surprise, decay) |
| `auteur.ts` | The director's brain — a typed cinematography grammar + shot-plan contract |
| `narrator.ts` | Captions — 3-tier narration (LLM / local / deterministic) |
| `director.ts` | The camera runtime — executes the shot plan frame-by-frame |
| `sandbox.ts` | The isolated WebGPU stage — renderer, camera, selective bloom |
| `storm.ts` | The 150k-particle GPU compute storm — physics, geometry, color, immersion |
| `components/MemoryCinema.svelte` | The orchestration overlay — input, dream mode, UI |

## Tier 1 — the narrative path (`pathfinder.ts`)

The bulletproof core that always runs, using only the nodes + edges the backend
returns. It plans a story: start at the origin (focused memory) → visit its
strongest-weighted connections → detour to a contradiction edge if one exists
(tension = interesting) → end on a recently-created node ("where the mind is now").
Each stop is a beat with a `kind` (origin/connection/contradiction/recent/bridge/
surprise). Falls back to weighted BFS. No LLM, no WebGPU, no network — if
everything else fails this alone produces a coherent, watchable flythrough.

## Tier 2 — graph signal extraction (`topology.ts`)

Pure statistics over the real `/api/graph` data, computed once per launch:
- **Brandes betweenness centrality** — the most load-bearing memory (graph keystone)
- **Connected-component clustering**
- **Contradiction detection** + **merge/supersede detection**
- **Surprise score** (shared neighbors yet low edge weight = non-obvious link)
- **Recency rank**, **FSRS retention**, **suppression pressure**

## The Auteur — the director's brain (`auteur.ts`)

A real cinematography grammar applied to your memories. The director (LLM Tier-1 or
a deterministic rule table Tier-2) produces a `DirectorPlan`: typed `Shot`s, each
grounded in a real node and justified by a real metric.
- **Moves:** push_in, pull_back, orbit, crane, whip_pan, rack_focus, hold
- **Angles:** low = look up (power), high = look down (decay), eye
- **Cuts:** fly, hard_cut, match_cut
- **Per shot:** dutch roll, standoff, flight/dwell seconds, spring half-life,
  intensity, tension (0–1)
- **Three acts** (I/II/III), an **emotional arc** (man_in_hole, rags_to_riches,
  icarus, cinderella, oedipus, flat), a **caption tone**, a **score cue**
- A **required `why`** per shot citing the real graph metric
- **Carry-forward semantics:** sparse/half-hallucinated plans always resolve to a
  coherent film.

## Narration (`narrator.ts`)

- **Tier 1:** backend LLM (`/api/narrative`) or opt-in on-device model (lazy-loaded
  only when "Local AI" enabled; never downloads weights unprompted).
- **Tier 2:** deterministic structured captions from real data — instant, no network.
- **Tier 3:** can't fail — falls back to Tier 2.
- Optional voice via `speechSynthesis`; typewriter caption stream (instant under
  reduced-motion).

## The 150k-particle storm (`storm.ts`)

150,000 particles whose physics run entirely on the GPU via Three Shading Language
compute nodes. One particle pool, one compute kernel, ~32 uniforms.

### The 7-beat world journey
Each beat is a unique world — particles aren't swapped, only the forces (uWorld
selects; uBlend crossfades over ~1s):
0 nebula mist (curl-noise flow) · 1 orbital anchor (cross-product spin) · 2 strange
attractor (Thomas) · 3 detonation void · 4 crystal lattice (voxel snap) · 5 fluid
galaxy (curl + swirl) · 6 phyllotaxis bloom (Vogel sunflower).

### The impossible-geometry form pack (dream worlds 7–11)
Forms nobody renders as living particles, mapped over a (u,v) manifold grid (387²)
so they read as a sculpted skin, not spaghetti:
7 supershape (3D superformula) · 8 Calabi–Yau (6D string-theory manifold, 4D→3D
projection; α rotates through the 4th dimension) · 9 Boy's surface (non-orientable
minimal immersion) · 10 Aizawa attractor shell · 11 gyroid↔Schwarz-D (triply-
periodic minimal surface morph). Inline complex-math + hyperbolics (sinh/cosh
expanded via exp; absent in three@0.172).

### Color
- Full-spectrum iridescent palette (per-particle phase + radial shells + spatial
  bands + time + global drift).
- Per-world cosine (IQ) palettes — each world a distinct identity.
- **The Color Blast:** a long-lived uBlast envelope (~2.8s, decoupled from the fast
  physics burst so color outlives the shockwave) drives an outward spectral-
  dispersion wave (uBlastTime; prism order) over a blackbody ember core. Spectrum
  dominates (reads rainbow, not white); capped at 0.6 mix.
- **Jarring inner/outer clash:** the nested figure and shell use opposing color
  universes (ice↔fire, acid↔blood, gold↔violet, mint↔crimson, electric↔gold);
  uClash cycles the pair every beat.

### 3D-within-3D nesting
~34% of particles form a second, smaller, counter-rotating figure (a different
world, ~52% scale) inside the outer shell — a figure within a figure.

### Anti-white-out / "solid" systems
- Rim glow (Fresnel): bright edges, dim readable center.
- Emissive routing: the rainbow goes to BOTH colorNode and emissiveNode (the
  selective bloom reads emissive — the original white-out was an unset emissive).
- Hollow-shell spawn (the old tiny dense ball flashed white on frame 0).
- Act/beat-aware brightness (uActDim): beats 0/1 fade in soft, Acts II/III blaze.

### The immersion stack
- **Infinite Droste zoom** (uZoomPeriod/uLambda/uZoomOn): the cloud dives inward
  forever. Two layers ride offset phases of fract(uTime/T); the outer grows
  pow(λ, phase) then snaps back invisibly (λ=1.923=1/0.52 makes inner→outer exact);
  the inner promotes by λ to become the next shell; a fresh inner spawns inside. A
  sin(phase·π) seam cross-fade in rimFactor makes each layer transparent at its
  snap → zero pop, seamless loop. Particle-space, not a camera dolly.
- **Velocity-stretch flythrough** (uCamVelView/uStreak/uMaxStretch): sprites stretch
  along screen-space camera velocity into warp streaks; the camera clamp floor
  relaxes (30→6) so it plunges inside the shell.
- **DOF + volumetric fog** (uFocus/uFocusRange/uDofDim/uFogDensity): off-focus
  particles dim (bokeh under bloom) + exp-falloff fog tints the far field → 3D
  atmospheric depth; rack-focus tracks the dive. All folded into a single depthFade
  Fn (three@0.172 constraint: positionView may be read from only ONE Fn per
  material output or the node-builder stack-overflows).
- **Interactive parallax** (overlay): pointer orbits, scroll/pinch zooms, damped;
  composed onto the director's base pose; idle eases back to 0; the sandbox re-clamp
  means framing can't break.

### Physics safety
Hard velocity clamp, strong damping, boundary snap, dt clamp, serialized compute
dispatches.

## Endless dream mode

When the 7-beat tour ends, instead of freezing it drops into an infinite generative
loop (`dreamBeat()` every ~5.5s): a fresh random procedural figure (worlds 7–11, new
uMorphSeed → never the same twice), uChaos ramps up so each is wilder than the last,
a random clash pair, full color blast, infinite zoom + flythrough on. Overlay shows
"∞ Dreaming." Never idle.

## The stage (`sandbox.ts`)

- Separate WebGPU renderer + scene + PerspectiveCamera (clamped to a safe distance
  band, always lookAt(origin) → the storm can never leave frame).
- Selective MRT bloom — blooms only the emissive channel against a clean void; falls
  back to a plain pass if MRT is unavailable on a driver.
- Per-frame camera-velocity tracking (one Vector3, zero compute) feeding the streak.
- Pass-throughs: setZoom, setStreak, setFlythrough, setCameraVel, setContainRadius.

## Overlay & UX (`MemoryCinema.svelte`)

- Fullscreen launch, staged status (idle → planning → playing → done).
- Pre-roll "Director's Plan" card (logline + arc).
- Live captions, beat chips, director's-note ("why," citing a real metric), act
  indicator, tension sparkline, progress bar.
- WebGPU / Auteur / Live-captions badges; Voice + Local AI toggles.
- `H` hides all chrome for clean demo capture (faint restore hint; `body.cinema-open`
  hides the graph page's stats pill; overlay z-index 200).
- Replay on completion.
- Graceful degradation: if WebGPU is unavailable or render fails 3× consecutively,
  drops to camera-only (captions still play) — never stalls.
- Reduced-motion fully honored: no parallax listeners, zoom/flythrough/streak gated
  off, jump-cuts instead of flights, instant captions.

## Protection contract

On the shared graph page, `<MemoryCinema>` is wrapped in a clearly-commented
`PROTECTED MEMORY CINEMA — DO NOT MODIFY` boundary so the graph-control overhaul and
the cinema engine stay visibly distinct.

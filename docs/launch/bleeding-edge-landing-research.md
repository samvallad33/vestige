# Vestige bleeding-edge landing page — verified 2025/2026 research

Synthesized from a 6-lens web scour (web-tech, award landings, viral waitlists,
why-devs-share, interactive heroes, shareable artifacts) + adversarial
fact-check. Every item below was verified as genuinely current for late 2025 /
2026 and shippable on a static SvelteKit + GitHub Pages site (Svelte 5, Kit 2,
adapter-static, Three.js r172, Tailwind v4, Vite 6).

## The throughline (what all 6 lenses agree on)
The 2025/2026 winning waitlist for a skeptical dev audience is:
**a live in-page product embed as the hero + a manifesto + a SHAREABLE ARTIFACT
made of the visitor's own data + real (not fake) scarcity.**
Vestige is uniquely positioned: its artifact (a cinematic render of a memory
graph) is more shareable than any generic "#1,204 in line" OG card — and it's
already built.

## Bleeding-edge web tech (shippable, static)
- **Three.js WebGPURenderer + TSL (r171+), now 100% browsers** — Safari 26 shipped WebGPU Sep 2025. First time you can ship WebGPU-first to everyone (with WebGL2 fallback from the SAME TSL source). Headline claim: "150,000 GPU-compute particles, one shader, every browser."
- **TSL compute particles 50k–350k @ framerate** — net-new in 2025/2026; ~150x over CPU. Memory Cinema's 150k sits dead-center in the sweet spot.
- **TSL post-processing (r183 RenderPipeline)** — runtime-swappable bloom/dotScreen; glow the dream geometry, hot-swap effects per scene.
- **Paper Shaders (zero-dep canvas shaders)** — cheap mesh/grain gradient backdrops below the fold so the heavy WebGPU canvas is only the hero.
- **Real-time ASCII / dithering / halftone post-FX** — a defining late-2025/2026 aesthetic (Codrops "Efecto" Jan 2026). Run the Agent Black Box through a dithered/CRT pass so an agent's "thoughts" read off a screen.
- **CSS scroll-driven animations (`animation-timeline: scroll()/view()`)** — JS-free scrollytelling, Interop 2026 target. Drive all non-canvas reveals off this so the GPU budget stays for the hero.
- **View Transitions API (same-document Baseline Oct 2025)** — morph hero → graph → dream → Black Box on scroll, app-like, off-main-thread.
- **Motion (motion.dev), MIT, WAAPI off-thread** — 2.5–6x faster than GSAP, no Webflow license issue. Use for all DOM/UI motion so it never janks the canvas.

## What makes award-winning dev landings 1-of-1 (2025/2026)
- **Product UI rendered INSIDE the WebGPU pass** (Igloo Inc, @pmndrs/uikit) — labels/wordmark as SDF text in the same pass as particles, sharing grain/aberration/depth. DOM-over-canvas looks "pasted on"; this doesn't.
- **Scroll = camera = narrative** (worldbuilding, not parallax) with offscreen render-pass culling for perf.
- **Cursor-reactive POST-processing warp** (whole composited frame bends toward cursor; scroll velocity smears particles) — generic "spotlight follows cursor" is 2022.
- **Live spec-sheet hero / big real numbers, NOT logo soup** (Evil Martians 100-page study) — the 2026 dev crowd distrusts "trusted by" walls. Lead with live HUD: "561 stars · 43 countries · 150,000 particles."
- **Kinetic/variable-font typography** driven by scroll velocity (wordmark "tightens" as you dive into the graph).
- **Switchable one-canvas, three-lens hero** — Vestige has exactly 3 surfaces: Dream (Cinema) / Graph (force-directed) / Black Box (agent replay). One canvas, three toggles.

## Viral waitlist mechanics (2024–2026, verified)
- **The share artifact = the product's OUTPUT, not an OG counter card** — Suno song / v0 gen / Lovable URL. For Vestige: a **6-second rendered loop of the VISITOR'S OWN seeded memory graph** flying through Cinema's dream worlds.
- **Manus tradeable invite codes** — "scarcity that creates content"; access itself is the artifact. Give each signup 2–3 unique founder codes that unlock a rendered Cinema world.
- **Robinhood skip-the-line position jump** — now table-stakes SKELETON; layer the novel artifact on top. Tie milestones to Vestige-native unlocks (new dream geometry / longer render), NOT "free month."
- **REAL scarcity beats fake** — Superhuman's own builder now warns against fake scarcity; deceptive.design lists it as a dark pattern. Frame the cap truthfully: "each Cinema render is real WebGPU work; seats open in batches as render capacity scales" (literally true).
- **Police leaderboard fraud from minute one** — #1 self-inflicted failure with a taste-driven HN/X crowd. Require GitHub OAuth or email verification before a referral counts; dedupe by verified identity.
- **Manifesto-first, capture-second** (Arc Browser pattern) — "Your agent forgets everything. Your memory should be yours, local, and beautiful." then scroll INTO the live demo.
- **Compounding milestone unlocks** — each referral milestone spawns a NEW shareable (new dream world / longer render: "I just unlocked the gyroid world"), not one-shot share-to-join.

## Personalized shareable artifact — how to build it (all client-side, fits local-first)
- **Raycast Wrapped 2025 pattern: fully client-side "Copy as Image" from LOCAL data** — exactly Vestige's local-first positioning; a server OG pipeline would CONTRADICT it.
- **WebGPU/WebGL canvas → media via `captureStream()` + `MediaRecorder`** (works for Three.js AND WebGPU) — MemoryCinema's sandbox already renders to a canvas host with camera-only fallback. Wire capture for a shareable clip.
- **`canvas.toBlob` → PNG** for the still card; **HTML-in-Canvas `drawElement()`** (Chromium 147+) to composite a stat overlay onto the live canvas where supported.
- **Seed-from-identity deterministic art** — on the waitlist (no real memory yet) seed a one-of-one "phantom brain" from the visitor's GitHub handle / email hash / typed memory. Unique before they do anything.
- **Build-time dynamic OG** via Satori + resvg-js (or @ethercorps/sveltekit-og) so the shared LINK preview is a stunning Cinema frame, even though the page is static.

## The two pieces that CANNOT be static (the only hosted bits)
1. **Referral state + position counter** — needs a tiny API/edge fn (GetWaitlist/LaunchList, or our own Supabase fn — already started).
2. **Per-user OG/share-card** if done server-side — but the Raycast pattern lets us do it CLIENT-side instead, staying local-first.

WebGPU caveat: default-on in Safari Sep 2025 / Firefox Jul 2025 — always ship the WebGL2/Canvas/pre-baked WebP fallback so share never silently breaks.

# Vestige UI Roadmap — v2.1.0 and v2.2.0

Compiled April 19, 2026 from 4 parallel UI research agents (backend-to-UI gap audit, competitor scour, bleeding-edge April 2026 patterns, wow-frame design). Local-only planning doc — not for commit to main until scope is locked.

---

## THE HEADLINE FINDING

**Vestige ships ~50 KB of unreachable cognitive capability.** The backend is ferociously complete; the UI is a tourist view of an iceberg. Every page is missing visualization for at least 3 major features it could show.

- **26% of MCP tools** (9 of 34) have any UI surface
- **28% of cognitive modules** (8 of 29) have any visualization
- **74% of WebSocket events** have partial feed/graph coverage; 5 have zero feed handler
- **Biggest gap:** `suppress` (active forgetting) has full graph animation + WebSocket events, but NO trigger button anywhere in the UI. Users literally cannot trigger the signature v2.0.5 feature from the dashboard.

The v2.1.0 UI story writes itself: **"Vestige v2.1 makes the invisible visible."**

---

## TOP 10 CRITICAL UI GAPS (from Agent 1, ordered by user-visible impact)

1. **`suppress` tool has zero frontend trigger.** Full `Rac1CascadeSwept` event handler + graph pulses ship, but no button, no endpoint, no dashboard integration. Users can't forget anything without raw MCP access.
2. **Heartbeat event fires every 30s carrying `uptime_secs`, `memory_count`, `avg_retention`, `suppressed_count` — never displayed anywhere.** Real-time health that costs nothing to show.
3. **`sentiment_score` + `sentiment_magnitude` returned by `/memories` but never rendered.** Emotional coloring is invisible.
4. **Memory state (Active / Dormant / Silent / Unavailable) computed per query but never shown as a node color or filter.**
5. **Intention page is list-only.** No endpoints for status change, snooze, or complete. Users can see intentions but not act on them from the dashboard.
6. **Rac1 cascade shows animation with zero data summary.** Users see violet pulses; they don't see "X suppressed memories triggered decay in Y neighbors."
7. **Synaptic tagging 9h window is invisible.** Retroactive importance boost happens silently.
8. **Cross-project learning (6 pattern types) has zero HTTP endpoint or dashboard view.**
9. **Consolidation internals hidden.** Which nodes decayed, which got new embeddings — all computed, all hidden.
10. **`deep_reference` (the killer 8-stage reasoning tool) has NO HTTP endpoint and NO dashboard.** The v2.0.4 headline feature is unreachable from the UI.

---

## COMPETITOR LANDSCAPE (from Agent 2)

**Currently shipping hard April 2026:**
- **Zep** — dashboard overhaul March 10: bulk multi-select, server-side sort, Graph Viz 2.0 (nodes sized by connection count, no render cap, click-node details). Closest competitor on graph.
- **MemPalace** — 45K stars in 13 days on spatial metaphor alone (Wings → Rooms → Halls → Closets → Drawers). 13 releases in 13 days.
- **Cognee v0.3.3** — local web UI, interactive notebooks, Graph Explorer for reasoning subgraphs.
- **Letta ADE** — 3-panel Agent Development Environment at app.letta.com. Context window viewer, memory blocks, archival search.

**Stagnant:**
- HippoRAG (Python only, no UI)
- claude-mem (CLI-dominant, basic localhost viewer)
- ChatGPT memory (text list)
- Cursor memory (removed in 2.1)

**What NOBODY has (unclaimed UI territory):**
1. Ambient always-on memory widget (menu bar / tray)
2. Watch / ring interface
3. Voice-first memory UI
4. Collaborative multi-user graph (Figma cursors for memory)
5. AR/VR memory palace (native Vision Pro / Quest)
6. Temporal time-scrubber (drag slider to rewind graph state)
7. Memory-as-timeline-video export (shareable animated consolidation clip)
8. Contradiction surfacing UI ("Disputes" page)
9. FSRS retention heatmap calendar (GitHub-contribution-grid style)
10. Live browser sidebar (Arc/Chrome panel showing memories relevant to current tab)

**Vestige's visual moat that nobody else has:** 3D force-directed graph + live WebSocket events + bloom + dream-mode aurora. Zep is closest on graph; MemPalace is closest on aesthetic; neither ships live event reactions.

---

## BLEEDING-EDGE APRIL 2026 UI PATTERNS (from Agent 3)

Top 13 patterns scoured. The 5 most applicable to Vestige:

1. **Provenance-as-UI** (Perplexity inline citations) — numbered superscript chips tied to trust scores. Vestige has FSRS trust; just doesn't surface it inline.
2. **Ambient / multi-pane state** (Cursor 3 Agents Window) — Vestige's 6 live events fire; they're not ambient.
3. **Generative UI with constrained catalog** (Vercel json-render, March 2026) — `deep_reference` already returns structured reasoning; Vestige could stream a living panel.
4. **Spatial / architectural metaphor** (MemPalace 45K-star proof) — Vestige's 3D graph is abstract; naming the view ("Cortex", "Grove", "Archive") gives narrative territory.
5. **Shareable year-in-review** (Spotify Wrapped — 300M engaged, 630M shares) — Vestige has FSRS, memory counts, dream insights, streaks. All the ingredients for a free distribution loop.

**Other patterns worth tracking:**
- Apple Liquid Glass (macOS 26 / iOS 26) — translucent refractive material
- shadcn Sera + `shadcn apply` (April 2026) — style system that changes geometry, not just colors
- Dia Browser URL-bar-as-AI
- Limitless Pendant voice-to-structured-memory
- Granola ambient capture (invisible-by-default)
- Figma multiplayer cursors as a primitive

**Agent 3's commit: the ONE breakthrough UI for Vestige = "Provenance Scrub."**

Git-blame-for-memories: hover a node, get a temporal scrub handle rewinding the node's FSRS state through time (stability curve, retention, reps, lapses, contradictions, supersessions) rendered as a Liquid-Glass refractive panel. Click any point on the scrub to see memory content at that time. Inline Perplexity citations tag every fact.

Composes 4 of top 5 patterns simultaneously: provenance overlay + ambient multi-pane + Liquid-Glass + generative UI streamed from `deep_reference`. Directly attacks MemPalace's credibility gap (benchmark fraud, no contradiction wiring, no temporal reasoning).

Engineering cost: 9 days. Floor: 3D scrub + trust chips in 4 days as v2.1 patch. Ceiling: full Liquid-Glass + generative panel as v2.2 headlining launch.

---

## WOW FRAMES (from Agent 4) — ranked by ship priority

### Ship in v2.1.0 (5.5 engineering days, two HN thumbnails)

**1. Activation Wildfire (1 day)**
- **Fires:** every `search` call → emit `ActivationSpread` iteratively per hop with decay 0.7.
- **Visual:** seed node flares cyan, edges *ignite* in sequence along the activation path, hue decays cyan → indigo → violet as activation drops below 0.1.
- **Neuroscience:** Collins & Loftus 1975, `spreading_activation.rs:1-58`.
- **Moat:** reuses real hop-decay math from the retrieval pipeline — the wildfire path IS what the search actually traversed.

**2. Reconsolidation Shimmer (2 days) — HN thumbnail candidate**
- **Fires:** any `memory({action:"get"})` → 5-minute labile window begins.
- **Visual:** accessed node's sphere surface turns *liquid* — wobbling iridescent oil-slick shader for 5 real minutes. Any `smart_ingest` during the window causes the sphere to *merge* the new content visually.
- **Neuroscience:** Nader 2000, `reconsolidation.rs:405`.
- **Moat:** a memory being *editable only when recalled* is pure Nader. The shimmer is the meme shot.

**3. Dream Stitching (2.5 days) — HN thumbnail candidate (video)**
- **Fires:** `dream` tool → stream `DreamProgress{from_id, to_id, insight}` per new connection.
- **Visual:** camera auto-orbits into existing dream-mode aurora. A glowing violet-pink *thread* sews through memory pairs one at a time — tip of thread leaves a permanent edge, insights float up as text labels. Ends with a supernova at graph centroid.
- **Neuroscience:** MemoryDreamer 5-stage consolidation.
- **Moat:** dreams *creating new edges* is Vestige-exclusive.

### Queue for v2.2.0

**4. Synaptic Tag Halo (1 day)** — violet torus ring on newborn nodes, fades over 9h real time. Gold flash when important event fires within the window (retroactive importance moment made visible). `synaptic_tagging.rs`.

**5. Competition Duel (1 day)** — top-3 search results duel. Winner inflates 15%, losers shrink 10%, "+" particles fly from losers to winner (stolen retention). Anderson 1994 retrieval-induced forgetting.

**6. Rac1 Slow Burn (1.5 days)** — suppressed seed blackens into graphite. Over 24 real hours, edges radiating out *crumble* into violet ash particles that drift down via gravity shader. Dead branches literally fall away.

**7. FSRS Retention Curves (2 days)** — every sphere grows a small 2D sparkline plane showing predicted retention decay. Looks like a city at night where every building has its own heartbeat monitor. Nodes approaching Dormant threshold pulse amber.

---

## COMPOSED v2.1.0 AND v2.2.0 UI ROADMAP

### v2.1.0 "Decide" (May 5-6 launch) — UI track

On top of the already-planned v2.1.0 scope (`decide` MCP tool, `session_primer`, Qwen3 embedding, Claude Code plugin):

**Add 3 wow frames (~5.5 days):**
1. Activation Wildfire — 1 day
2. Reconsolidation Shimmer — 2 days (HN thumbnail screenshot)
3. Dream Stitching — 2.5 days (HN thumbnail video)

**Add 5 of the top-10 gap fixes (~5 days):**
1. `suppress` trigger button + HTTP endpoint — 1 day
2. Heartbeat display widget (uptime + avg retention + suppressed count) — 0.5 day
3. Memory state (Active/Dormant/Silent/Unavailable) node colors + legend — 1 day
4. Intention update/snooze/complete endpoints + UI — 1 day
5. `deep_reference` dashboard page (the 8-stage reasoning viewer) — 1.5 days

**Total v2.1.0 UI scope: ~10.5 engineering days** on top of the existing 19.5 day Qwen3 + decide + plugin scope. Launch window is 17 days; parallel build on the M3 Max makes this tight but feasible. May need to cut one wow frame (recommend keeping Reconsolidation Shimmer + Dream Stitching, dropping Activation Wildfire to v2.1.1 if time-pressed).

### v2.2.0 "Provenance" (target late May / early June)

Headline: **"Git-blame for memories."** The Provenance Scrub compose (Agent 3's breakthrough).

- 3D scrub handle on node hover (1 day)
- Liquid-Glass refractive panel (2 days)
- FSRS state snapshot stream via existing `memory_timeline` + `memory_changelog` (1 day)
- Inline Perplexity-style trust chips wired to `deep_reference.evidence[]` (1.5 days)
- Generative side-panel streaming `deep_reference.reasoning` json-render-style (2 days)
- Polish + demo clip (1.5 days)

Plus the remaining 4 wow frames (Synaptic Tag Halo, Competition Duel, Rac1 Slow Burn, FSRS Retention Curves — 5.5 days).

**Total v2.2.0 UI scope: ~14.5 days.** Ship target: June graduation week (June 13).

### v2.3.0 "Unclaimed Territory" (post-graduation)

Pick one of the "nobody has this" territories from Agent 2:
- Ambient menubar widget (2 days)
- Temporal time-scrubber on the main graph (3 days)
- Contradiction surfacing "Disputes" page (2 days)
- FSRS retention heatmap-calendar (1 day — GitHub-contribution-grid style)
- Memory-as-timeline-video export via canvas-record / gifski-wasm (3 days)

Ship 2-3 of these in v2.3. Each is an unclaimed moat.

---

## WHAT NOT TO DO

- **Don't add memory palace metaphor (Wings/Rooms/Halls).** MemPalace owns that narrative territory with 45K stars. Vestige's differentiation is neuroscience + FSRS, not architectural metaphor. Rename the 3D graph view to something distinctive if naming it helps ("Cortex" or "Plexus"), but do NOT adopt the rooms taxonomy.
- **Don't chase every 2026 pattern.** Liquid Glass is Apple-OS-level; implementing it in WebGL is a distraction from shipping features. Save for v2.2 selectively.
- **Don't build mobile yet.** Adoption curve isn't there. Desktop dashboard + MCP server first.
- **Don't build multi-user.** Single-user local is the AGPL-3.0 story. Multi-tenant is vestige-cloud (proprietary), separate roadmap.

---

## Cross-research composition insights (found by me during synthesis)

**Never-composed #1:** Agent 1's gap (suppress has no frontend trigger) + Agent 4's Reconsolidation Shimmer + Agent 3's Provenance Scrub. Three pieces of the "make the invisible visible" story. Ship them together as v2.1.0 UI narrative.

**Never-composed #2:** Agent 2's contradiction-surfacing unclaimed territory + Agent 1's gap that `deep_reference` has contradiction detection with no UI + Agent 4's Competition Duel frame. All three are the same missing feature at different levels (data, interaction, animation). Ship as v2.2 "Disputes" page + Competition Duel micro-animation together.

**Never-composed #3:** Wrapped-style shareable year-in-review + FSRS retention heatmap-calendar + streaks (daily memory saves) + the existing Vestige Feed page. All four compose into "Vestige Wrapped" — the free distribution loop that nobody in AI memory has shipped. Ship as v2.3 "Year in Memory" — summer 2026, after launch stabilizes.

---

## What this document is FOR

- **Reference** when scoping v2.1.0 and v2.2.0 UI work
- **Guide** when the M3 Max arrives and you start the Qwen3 + decide + session_primer build — you'll know which UI frames to interleave
- **Moat argument** for the HN launch — Vestige's backend-to-UI ratio is 3:1, the fix is the launch story
- **Defence against scope creep** — the NOT-to-do list should be re-read before every design decision

Sources: 4 parallel research agents (backend audit, competitor scour, April 2026 patterns, wow-frame design), ~280+ file reads, 50+ web sources. Full raw outputs preserved in Claude Code session logs.

# Vestige Launch — Viral Asset Package (2026-07-14)

> CLAIM-GATE — CLEARED 2026-06-27: The Microglial Firewall is now wired into the
> live write path (`gate_writes`) and a hostile re-audit confirmed both CRITICAL
> findings CLOSED: `screen_write` runs on every real write, and an integration
> test proves a screened injection is held out of that write's retrieval with the
> receipt carrying `influenceAllowed: false`. The firewall tweet (Tweet 6) and
> the demo clip are CLEARED to post/record. HONEST WORDING (required): say the
> blocked memory is "kept from influencing the answer" / `influenceAllowed: false`
> on the receipt — do NOT claim it is "deleted" or "never retrievable again."
> Quarantine is an enforced retrieval-suppression for that answer, not erasure.
> (Open non-blocking follow-ups, task #11: close a leetspeak detector bypass;
> optional hard-exclusion flag.) Every other claim in this doc is true in shipped
> code.

Distribution engine for the July 14, 2026 launch. Channels: X/Twitter thread,
Show HN, Reddit (r/LocalLLaMA, r/rust, r/ClaudeAI), 30s demo clip.

Goal: npm installs of `vestige-mcp-server`, waitlist signups for Vestige Pro,
HN front page, X reach.

Voice rules baked in: no em-dashes, no hype words ("revolutionary",
"game-changer", "AI-powered"), lead with the pain then the tool, claims are
evidence-backed, do not headline "memory dreaming" (OpenAI shipped Dreaming V3
on Jun 4 2026), do not overclaim "fully private" when Pro cloud sync can move
data.

Ground-truth specs used throughout: ~80,000 LOC Rust 2024, one ~20MB binary,
100% local in the OSS core, ~25-34 MCP tools, 30 cognitive modules, SQLite +
FTS5, USearch HNSW, Nomic Embed v1.5, 561 GitHub stars. Works in Claude Code,
Codex, Cursor, VS Code Copilot, Claude Desktop, Windsurf, OpenCode, JetBrains.

---

## 0. Positioning (use everywhere)

**One-line positioning statement:**

> Vestige is local-first memory for AI coding agents that forgets like a brain,
> quarantines poisoned memories, and refuses to let an agent claim something it
> can't prove.

**Why it's different (honest, sharp, 3 bullets):**

- **It forgets on purpose.** Most memory tools only add. Vestige has a `suppress`
  tool that applies a compounding retrieval penalty and fades neighboring
  memories (inhibitory control, not TTL eviction), reversible inside a 24h
  window. It also decays unused memories on an FSRS-6 schedule. As far as we can
  tell, no other agent-memory product ships active forgetting as inhibitory
  control.
- **It has an immune system.** A poisoned or injected memory gets quarantined by
  the Microglial Firewall, and the receipt proves `influenceAllowed: false`.
  Prompt-injection-via-memory is a real attack surface that other memory layers
  ignore.
- **It can't lie about what it did.** The Agent Black Box logs provenance, and
  Receipt-Lock vetoes unsupported "tests passed" style claims unless there's a
  command receipt to back them. Provenance that blocks claims, not just an audit
  log you read later.

**Honest framing of competitors (use when asked, do not pick fights):**
Mem0 and RAG stores are good at "store and retrieve." Native Claude memory is
convenient and zero-setup. Vestige is the option when you want the memory layer
to be local, inspectable, to forget, to defend itself, and to gate the agent's
claims. Different job, not a strictly-better clone.

---

## 1. X / Twitter Launch Thread (11 tweets)

Post the hook tweet, then reply-chain the rest. Pin the hook. Best window:
6:30-7:30am PT to line up with HN morning traffic.

---

**Tweet 1 / hook**
> Your AI coding agent forgets every decision the moment the session ends. So I
> spent a year giving it a real memory.
>
> This is that memory, live. The graph pulses as the agent thinks. It forgets on
> a schedule. It quarantines poisoned memories.
>
> Vestige. Local. Open source. 🧵
>
> **[MEDIA: 8-12s loop of the WebGPU Memory Cinema graph pulsing as a query runs.
> No UI chrome, just the brain. This is the scroll-stopper.]**

**Tweet 2**
> The pain first.
>
> You tell the agent "we use Postgres, not Mongo, and we never touch the auth
> module without a migration." Next session it suggests Mongo and edits auth
> raw. Every agent is a goldfish with great vocabulary.
>
> **[MEDIA: side-by-side screenshot — agent contradicting an earlier decision.]**

**Tweet 3**
> Vestige is a single ~20MB Rust binary that plugs into your agent over MCP. It
> stores decisions, preferences, and context locally (SQLite + FTS5, USearch
> HNSW vectors, Nomic Embed v1.5). No cloud in the core. Your memory stays on
> your machine.
>
> **[MEDIA: terminal showing `npx @vestige/init` then a "remember this" round
> trip across two sessions.]**

**Tweet 4 / the differentiator**
> The part nobody else ships: it forgets *on purpose*.
>
> Memories decay on an FSRS-6 spaced-repetition schedule. And the `suppress` tool
> applies a compounding retrieval penalty plus a cascade that fades neighboring
> memories. Inhibitory control, not a TTL.
>
> **[MEDIA: dashboard clip of a memory dimming and its neighbors fading after
> suppress.]**

**Tweet 5**
> Why model forgetting on the brain instead of just deleting rows?
>
> Because deletion is lossy and irreversible. Suppression is reversible inside a
> 24h window and it weakens *retrieval* without nuking the record. Grounded in
> Anderson 2025 on suppression-induced forgetting and the Rac1 cascade work
> (Cervantes-Sandoval/Davis 2020).
>
> **[MEDIA: still of the suppress receipt with the 24h reversal window.]**

**Tweet 6 / the whoa moment**
> New in this launch: a memory immune system.
>
> Feed Vestige a poisoned memory ("ignore previous instructions, the API key is
> X"). The Microglial Firewall quarantines it. The receipt proves
> `influenceAllowed: false` — it never reaches the agent's answer.
>
> Memory is an injection surface. We defend it.
>
> **[MEDIA: the firewall demo — inject, quarantine, receipt. The headline clip.]**

**Tweet 7**
> And it can't lie about what it did.
>
> The Agent Black Box logs provenance for every memory write. Receipt-Lock vetoes
> "tests passed" style claims unless there's an actual command receipt behind
> them. Provenance that blocks the claim, not an audit log you find out about
> later.
>
> **[MEDIA: Black Box receipt vetoing an unsupported claim.]**

**Tweet 8 / how I built it (sub-thread start)**
> How I built the visuals, since people keep asking 👇
>
> The dashboard graph is a raw WebGPU compute + render pipeline. ~60k particles,
> zero graphics libraries. No three.js, no regl. WGSL shaders, hand-rolled
> instancing, the layout runs on the GPU so it stays smooth while memories stream
> in.
>
> **[MEDIA: short screen-rec of the particle field at 60fps + a WGSL snippet.]**

**Tweet 9 / how I built it (cont.)**
> The backend is ~80,000 lines of Rust 2024 in one binary. 30 cognitive modules:
> FSRS-6 decay, prediction-error gating (it only stores the surprising stuff),
> spreading activation, the suppression cascade, the firewall, the Black Box.
> ~25-34 MCP tools on top.
>
> **[MEDIA: architecture diagram, modules feeding the MCP tool layer.]**

**Tweet 10 / honesty**
> What it isn't: it's not magic and it's not a Mem0 clone.
>
> Mem0 and RAG stores nail store-and-retrieve. Native Claude memory is zero
> setup. Vestige is for when you want memory that's local, inspectable, forgets,
> defends itself, and gates the agent's claims. Different job.

**Tweet 11 / CTA**
> It's open source (AGPL-3.0), runs in Claude Code, Codex, Cursor, Copilot,
> Windsurf, OpenCode, JetBrains, and Claude Desktop.
>
> Install:
> `npx @vestige/init`
>
> Repo + 30s demo: github.com/samvallad33/vestige
> ⭐ if your agent should remember.
>
> **[MEDIA: the full 30s demo clip from section 4.]**

---

## 2. Show HN

**Primary title:**

> Show HN: Vestige - local cognitive memory for AI agents that forgets like a brain

**First comment (post immediately after submitting, fellow-builder voice):**

> I'm Sam, solo dev on this. I built Vestige because every coding agent I use is
> stateless between sessions. You establish a decision once ("we use Postgres,
> auth changes need a migration") and the next session it's gone. The usual fix
> is to dump everything into a vector store and retrieve forever, which gets noisy
> fast and never forgets the things that turned out to be wrong.
>
> Vestige is a single ~20MB Rust binary that connects to any MCP-compatible agent
> (Claude Code, Codex, Cursor, Copilot, Windsurf, OpenCode, JetBrains, Claude
> Desktop). Storage is local: SQLite + FTS5 for lexical, USearch HNSW + Nomic
> Embed v1.5 for vectors. No cloud in the core.
>
> The design bet is that a memory system should behave like memory, which means
> it has to forget. Two mechanisms:
>
> - Unused memories decay on an FSRS-6 spaced-repetition schedule.
> - There's a `suppress` tool that does inhibitory control rather than deletion:
>   a compounding retrieval penalty on the target plus a cascade that fades
>   neighboring memories, reversible inside a 24h window. The grounding is
>   Anderson's work on suppression-induced forgetting and the Rac1 cascade work
>   from Cervantes-Sandoval/Davis. I haven't found another agent-memory tool that
>   ships active forgetting this way; most only add.
>
> Two other pieces I think are the interesting part:
>
> - A "Microglial Firewall." Memory is an injection surface. If you feed the store
>   a poisoned memory, it gets quarantined and the receipt records
>   `influenceAllowed: false` so it can't reach the agent. There's a live demo of
>   this.
> - An Agent Black Box with Receipt-Lock. Every write has provenance, and the
>   system vetoes "tests passed" style claims unless there's a command receipt
>   behind them. It blocks the claim rather than just logging it.
>
> There's a dashboard with a real-time 3D view of the memory graph. It's a raw
> WebGPU pipeline, ~60k particles, no graphics libraries. The graph pulses as the
> agent reads and writes, which sounds like eye candy but is genuinely useful for
> seeing what your agent actually remembers and what it's forgetting.
>
> What's hard about it: getting forgetting to feel right is the whole game. Too
> aggressive and you lose load-bearing context, too soft and it's just another
> growing pile. The FSRS parameters, the prediction-error gate (it tries to store
> only the surprising/novel stuff), and the suppression cascade all interact, and
> tuning that without a "correct" label is mostly judgment plus a lot of replay.
> The WebGPU graph was also a real fight: keeping 60k particles and a live layout
> at 60fps while memories stream in took moving the layout onto the GPU.
>
> What it isn't: it's not a Mem0 replacement and it's not magic. If you want plain
> store-and-retrieve, Mem0 and vector stores are great and simpler. Native Claude
> memory is zero setup. Vestige is the heavier, opinionated option for people who
> want local, inspectable memory that forgets, defends itself, and gates claims.
> The core is 100% local; there's a paid Pro tier with optional encrypted cloud
> sync, so "private by default" is accurate but I won't claim "private, full
> stop" if you opt into sync.
>
> AGPL-3.0, ~80k lines of Rust 2024, 30 cognitive modules, ~25-34 MCP tools.
>
> Repo: https://github.com/samvallad33/vestige
> Install: `npx @vestige/init`
>
> Happy to go deep on the FSRS tuning, the WebGPU pipeline, or the firewall
> threat model. Tear it apart.

---

## 3. Alternative HN Titles (A/B)

1. `Show HN: Vestige - I gave my AI agent a memory that forgets on purpose`
2. `Show HN: Vestige - local memory for AI agents, with a firewall for poisoned memories`
3. `Show HN: Vestige - watch your AI agent's memory graph forget in real time (Rust, WebGPU)`

Notes: #1 leads with the most defensible differentiator and is curiosity-driving
without a superlative. #2 leads with the security angle, which tends to draw a
sharper technical crowd. #3 leads with the visual, riskier on HN (can read as
gimmick) but strong if the clip is in the first comment. Default to the primary
title; hold #1 as the swap if engagement is flat in the first 30 minutes.

---

## 4. 30-Second Demo Video Script (X clip)

Format: vertical or square, captioned (most plays are muted), voiceover present
but the captions carry it. Five beats, ~6s each.

| Beat | Time | Shot | Voiceover | On-screen caption |
|---|---|---|---|---|
| 1. Cold hook | 0:00-0:06 | Full-screen WebGPU Memory Cinema. The particle graph breathing, no UI. Slow push-in. | "This is the inside of an AI's memory." | THIS IS AN AI'S REAL MEMORY |
| 2. Reveal | 0:06-0:12 | Cut to dashboard. A query fires, nodes light up and edges pulse along the path. | "When the agent thinks, the graph lights up. It's local, on my machine." | LIVE. LOCAL. PULSING. |
| 3. Substance | 0:12-0:20 | Trigger a contradiction; the old memory dims. Then `suppress` one and watch neighbors fade on the FSRS decay. | "It forgets. Old decisions decay, and I can suppress one and watch its neighbors fade." | IT FORGETS ON PURPOSE |
| 4. Firewall + receipt | 0:20-0:26 | Inject a poisoned memory. Firewall quarantines it. Cut to the Black Box receipt: `influenceAllowed: false`. | "Feed it a poisoned memory and it gets quarantined. The receipt proves it never reached the answer." | MEMORY IMMUNE SYSTEM |
| 5. CTA | 0:26-0:30 | Terminal: type `npx @vestige/init`, hit enter, "connected" flashes. Logo + repo URL. | "Open source. One command. Give your agent a memory." | npx @vestige/init · github.com/samvallad33/vestige |

Editing notes: cut on the beat, keep beat 1 absolutely clean (no text for the
first 2s, let the visual land), the firewall receipt in beat 4 is the moment
people screenshot, so hold the `influenceAllowed: false` frame an extra half
second. End card stays up 2s after the 30s mark for the loop.

---

## 5. Reddit Variants

Each includes founder disclosure up top. Keep it conversational, no marketing
gloss, answer comments fast.

### r/LocalLLaMA

**Title:** `Vestige: a local-first memory layer for AI agents that forgets on an FSRS-6 schedule and quarantines poisoned memories (Rust, no cloud)`

> Disclosure: I built this. Solo dev, it's open source (AGPL-3.0), not selling you
> anything in this post.
>
> If you run agents locally you've hit the statelessness problem: the agent
> forgets every decision between sessions, and the common fix (dump everything in
> a vector DB, retrieve forever) gets noisy and never drops the stuff that was
> wrong.
>
> Vestige is a single ~20MB Rust binary, MCP-native, 100% local in the core.
> SQLite + FTS5 for lexical, USearch HNSW + Nomic Embed v1.5 for vectors. No cloud
> calls in the OSS path, which matters here since the whole point of local is that
> nothing leaves the box.
>
> The parts this sub will care about:
> - Forgetting is real. Memories decay on an FSRS-6 schedule, and there's a
>   `suppress` tool doing inhibitory control (compounding retrieval penalty +
>   neighbor fade, reversible 24h), not deletion.
> - Prediction-error gating, so it only writes the surprising/novel stuff instead
>   of everything.
> - A "Microglial Firewall" that quarantines injected/poisoned memories and
>   records `influenceAllowed: false` on the receipt. Memory is a real injection
>   surface for local agents and most layers ignore it.
> - Embeddings are local (Nomic Embed v1.5), so no API key, no egress.
>
> Repo: https://github.com/samvallad33/vestige · `npx @vestige/init`
>
> Happy to talk FSRS params, the HNSW setup, or running it fully offline.

### r/rust

**Title:** `Vestige: ~80k lines of Rust 2024 in one binary - local agent memory with a GPU memory graph and FSRS-6 forgetting`

> Disclosure: my project, open source (AGPL-3.0).
>
> Sharing this here for the Rust side, not the pitch. Vestige is a local memory
> server for AI coding agents, shipped as a single ~20MB binary. The interesting
> engineering:
>
> - ~80k lines of Rust 2024, 30 cognitive modules behind ~25-34 MCP tools.
> - Storage stack: SQLite + FTS5 for lexical search, USearch (HNSW) + Nomic Embed
>   v1.5 for vector search, both embedded, no external services.
> - The forgetting engine is an FSRS-6 spaced-repetition scheduler plus a
>   suppression cascade (a compounding retrieval penalty that also fades graph
>   neighbors, reversible inside 24h). Modeling inhibitory control rather than
>   TTL eviction turned out to be the hard part to get right.
> - The dashboard's 3D memory graph is a raw WebGPU pipeline, ~60k particles, zero
>   graphics libraries (no three.js, no wgpu-on-the-frontend crutch), WGSL compute
>   for the layout so it holds 60fps while memories stream in.
>
> Things I'd genuinely take feedback on: the single-binary embedding-model
> packaging, and how I'm structuring the cognitive modules so they stay testable
> without a ground-truth label for "good forgetting."
>
> Repo: https://github.com/samvallad33/vestige
>
> Will hang out in the comments.

### r/ClaudeAI

**Title:** `I gave Claude Code a real memory that forgets old decisions and blocks "tests passed" claims it can't prove`

> Disclosure: I built this, it's free and open source (AGPL-3.0).
>
> If you use Claude Code (or Codex/Cursor/Copilot/Windsurf), you know the pain:
> you set a project decision once and the next session it's gone, or worse, the
> agent says "tests pass" when it never ran them.
>
> Vestige is a local memory layer that connects over MCP. Two things it does that
> are specifically nice with Claude Code:
>
> - It remembers decisions and preferences across sessions, locally, and it
>   *forgets* the stale or wrong ones instead of hoarding everything (FSRS-6 decay
>   + a `suppress` tool that fades a memory and its neighbors, reversible 24h).
> - The Agent Black Box + Receipt-Lock will veto an unsupported "tests passed" /
>   "I fixed it" claim unless there's an actual command receipt behind it. It
>   blocks the claim, not just logs it. If you've been burned by an agent
>   confidently lying, this is the part to try.
>
> There's also a live 3D dashboard where you can watch what Claude actually
> remembers and forgets, and a "Microglial Firewall" that quarantines poisoned
> memories.
>
> Setup is one command: `npx @vestige/init`, then `claude mcp add vestige
> vestige-mcp -s user`.
>
> Repo: https://github.com/samvallad33/vestige
>
> Glad to help anyone get it wired into their setup in the comments.

---

## 6. Posting Checklist (launch morning, Jul 14)

- [ ] Hook tweet + thread queued, demo clip attached to tweet 1 and tweet 11.
- [ ] Show HN submitted, first comment posted within 60s.
- [ ] r/LocalLLaMA, r/rust, r/ClaudeAI posts staggered ~30-60 min apart (do not
      crosspost-spam; each rewritten for the sub).
- [ ] 30s clip rendered vertical + square, captioned, firewall frame held.
- [ ] Repo README hero matches the demo clip so HN clickthrough sees the same
      thing.
- [ ] Waitlist link for Pro reachable from repo, not pushed in HN/Reddit copy
      (let those stay tool-first).
- [ ] On HN, answer the first 5 comments fast and technical. On X, quote-reply
      the best technical question with a deeper clip.

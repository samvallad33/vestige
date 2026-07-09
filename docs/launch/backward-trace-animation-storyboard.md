# Backward-Trace Animation — Storyboard ("The X Rocket")

**Purpose:** the single most shareable launch artifact. A ~15s looping clip,
pinned to the top of the launch thread on X and used as the Show HN demo. It
shows the ONE thing nothing else does: when a failure hits, Vestige's arrow
snaps *backward through time* past the confounders to the quiet change that
actually caused it — while a vector search sits stuck on the symptom.

**Design law (from 2026 viral-demo research):** animate the *mechanism*
legibly, no narration needed, payoff visible in the first 1.5s, loops clean, no
logo intro. The tldraw / X-algorithm-viz / CodeReel pattern: the demo IS the
pitch.

---

## The case shown (canonical CauseBench archetype)

Uses the exact incident SHAPE the benchmark is built on (see
`benchmarks/causebench/data/real/*.meta.json`: a config/limit/cert/migration/flag
change that shares an *entity* with a later failure but *none of its words*, with
a more-recent confounder planted to defeat naive recency).

Timeline of stored memories (left = 3 weeks ago, right = today):

| When | Memory | Role |
|---|---|---|
| 21 days ago | `Lowered connection-pool max from 100 → 20 in db.config.yaml` | **the true cause** (shares entity `db.config.yaml` / the DB, not words) |
| 9 days ago | `Renamed UserService to AccountService across the API` | confounder (loud, recent, unrelated) |
| 4 days ago | `Bumped Postgres driver to 5.2` | confounder (shares "Postgres", more recent than cause) |
| **today** | `PagerDuty: checkout timing out under load — "connection acquisition timeout"` | **the failure (symptom)** |

Vector search ranks by resemblance → it surfaces the driver bump and the
timeout log (they share words like "connection", "Postgres"). It ranks the
pool-size change near the *bottom* — that memory shares no vocabulary with
"timeout under load". Vestige's Retroactive Salience Backfill reaches backward
along the shared entity (the database / `db.config.yaml`) and promotes the
pool-size change to #1.

---

## Frame-by-frame (15s, 30fps, loops)

**Beat 0 — 0.0s to 1.5s · THE HOOK (payoff visible immediately)**
- A red failure card slams in at the right edge (today): `checkout timing out — connection acquisition timeout`. Subtle shake.
- Caption, one line, monospace: `your agent's bug, today →`
- (Why first: a scroller must see the payoff-shape in 1.5s or they're gone.)

**Beat 1 — 1.5s to 4.0s · THE TIMELINE FILLS**
- A horizontal time axis draws left→right. Four memory cards fade in at their dates: the pool-size change (far left, dim/dormant), the two confounders (mid), the failure (right, red).
- The dormant cause is visibly *faded* — small, low-contrast. It looks unimportant. That's the point.

**Beat 2 — 4.0s to 6.5s · VECTOR SEARCH TRIES (and fails)**
- Label appears: `vector search` with a small magnifying-glass icon.
- Three thin gray "similarity" beams shoot from the failure card to the cards that *share words*: the driver bump and (self) the log. A rank badge appears: the true cause gets tagged `#7` in gray, sinking to the bottom.
- Caption: `finds what looks like the bug`
- Beat ends with a gray ✕ over the search — it never touched the real cause.

**Beat 3 — 6.5s to 10.5s · THE ARROW SNAPS BACK (the money moment)**
- Everything else desaturates. A single bright teal arrow launches from the failure card and travels *right-to-left, backward in time*, deliberately skipping the two confounders (each pings faintly and is passed over — "shares words, not entity" micro-label flickers).
- The arrow lands on the dormant pool-size card 21 days back. On contact the card **ignites**: scales up, fills teal, snaps from dim to bright. A link line labeled `same entity: db.config.yaml` connects them.
- Caption: `Vestige reaches back to what caused it`
- This is the frame that gets clipped and reshared. Make the snap fast and physical (ease-in, slight overshoot).

**Beat 4 — 10.5s to 13.0s · THE VERDICT**
- The promoted cause card shows a teal `#1` badge; the vector `#7` badge greys beside it for contrast.
- Two-line stat, big: `root-cause recall@1` / `Vestige 60% · vector search 0%`
- Small honest sub-label: `on CauseBench · reproducible`

**Beat 5 — 13.0s to 15.0s · SIGNATURE + LOOP RESET**
- Wordmark `vestige` fades in bottom-left, tiny. Tagline: `memory that finds the cause, not the resemblance.`
- Everything gently fades and the failure card is already sliding back in at the right — the loop restart is seamless (no hard cut).

---

## Production notes

- **Format:** build in HTML/CSS/SVG (animated), screen-record to MP4 + GIF.
  Alternatively Remotion (React) if you want a clean MP4 export pipeline — but
  the animated-SVG prototype below is enough to record from directly.
- **Palette:** teal `#1d9e75` = Vestige/cause/truth; gray `#888780` = vector
  search/confounders/noise; red `#e24b4a` = the failure only. Two-color
  discipline + one alarm color. Works on light and dark.
- **Text:** monospace for the memory-card content (feels like real logs);
  sans for captions. No narration — captions carry it, so it works muted in a
  feed (most X video autoplays silent).
- **Honesty guardrails (non-negotiable, same as the chart):** the stat frame
  must say `60%` with `on CauseBench` attached, and pair it with `vector search
  0%`. Never imply an industry benchmark. Never show the FALSE cross-session
  claim.
- **Loop length:** 15s is the sweet spot for X autoplay + a clean GIF under
  ~8MB. If the GIF is too heavy, cut Beat 1 to 2s and land at 13s total.

---

## The X post it pins to

> Every AI memory tool is built on vector search. Vector search finds what
> *looks like* your bug.
>
> But a root cause never looks like the bug it creates.
>
> So I built memory that reaches *backward in time* to the change that actually
> caused it. 60% root-cause recall where vector search scores 0%. 🧵👇
>
> [pinned: the 15s clip]

(Then the thread: the wall-of-zeros chart, the repro command, the "here's where
it's weak" honesty tweet.)

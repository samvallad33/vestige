# Backward-Trace Animation — Storyboard ("The X Rocket")

**Purpose:** the single most shareable launch artifact. A ~15s looping clip,
pinned to the top of the launch thread on X and used as the Show HN demo. It
shows the one interaction ordinary similarity search cannot provide: when a
failure hits, Vestige can follow its *recorded evidence backward through time*
past confounders to an earlier upstream **candidate** — while a similarity
baseline remains on the symptom. The replay is evidence, not an automatic proof
of causation.

**Design law (from 2026 viral-demo research):** animate the *mechanism*
legibly, no narration needed, payoff visible in the first 1.5s, loops clean, no
logo intro. The tldraw / X-algorithm-viz / CodeReel pattern: the demo IS the
pitch.

---

## The case shown (canonical reproducible fixture)

Use a committed, reproducible incident fixture: a config/limit/cert/migration/flag
change shares a *recorded entity* with a later failure but none of its words, with
a more-recent confounder planted to defeat naive recency. The film must identify
it as a fixture unless a versioned benchmark harness and results are available.

Timeline of stored memories (left = 3 weeks ago, right = today):

| When | Memory | Role |
|---|---|---|
| 21 days ago | `Lowered connection-pool max from 100 → 20 in db.config.yaml` | **recorded upstream candidate** (shares entity `db.config.yaml` / the DB, not words) |
| 9 days ago | `Renamed UserService to AccountService across the API` | confounder (loud, recent, unrelated) |
| 4 days ago | `Bumped Postgres driver to 5.2` | confounder (shares "Postgres", more recent than cause) |
| **today** | `PagerDuty: checkout timing out under load — "connection acquisition timeout"` | **the failure (symptom)** |

The similarity baseline ranks by resemblance and may surface the driver bump or
timeout log because they share words such as "connection" and "Postgres". The
film must render the **actual candidate list and rank from that run's receipt**;
it must not pre-bake a losing rank. Backfill then follows the recorded entity
(`db.config.yaml`) backward and presents the pool-size change as an upstream
candidate, with the join key, path, and timestamp for a human to verify.

---

## Frame-by-frame (15s, 30fps, loops)

**Beat 0 — 0.0s to 1.5s · THE HOOK (payoff visible immediately)**
- A red failure card slams in at the right edge (today): `checkout timing out — connection acquisition timeout`. Subtle shake.
- Caption, one line, monospace: `your agent's bug, today →`
- (Why first: a scroller must see the payoff-shape in 1.5s or they're gone.)

**Beat 1 — 1.5s to 4.0s · THE TIMELINE FILLS**
- A horizontal time axis draws left→right. Four memory cards fade in at their dates: the pool-size change (far left, dim/dormant), the two confounders (mid), the failure (right, red).
- The dormant cause is visibly *faded* — small, low-contrast. It looks unimportant. That's the point.

**Beat 2 — 4.0s to 6.5s · SIMILARITY BASELINE**
- Label appears: `similarity baseline` with a small magnifying-glass icon.
- Thin gray "similarity" beams shoot from the failure card only to the baseline candidates named in the receipt. A rank badge shows the upstream candidate's **actual baseline rank** from that same receipt; omit the badge when the rank was not measured.
- Caption: `finds what looks like the bug`
- Do not draw a failure cross. The baseline and Backfill answer different
  questions; the receipt makes that difference inspectable.

**Beat 3 — 6.5s to 10.5s · THE ARROW SNAPS BACK (the money moment)**
- Everything else desaturates. A single bright teal arrow launches from the failure card and travels *right-to-left, backward in time*, deliberately skipping the two confounders (each pings faintly and is passed over — "not on this recorded path" micro-label flickers).
- The arrow lands on the dormant pool-size card 21 days back. On contact the card **ignites**: scales up, fills teal, snaps from dim to bright. A link line labeled `recorded entity: db.config.yaml` connects them.
- Caption: `Vestige reaches back to recorded upstream evidence`
- This is the frame that gets clipped and reshared. Make the snap fast and physical (ease-in, slight overshoot).

**Beat 4 — 10.5s to 13.0s · THE RECEIPT**
- The promoted card is labeled `recorded upstream candidate`, not "root cause."
- Show a receipt card with the exact `run ID`, `shared entity`, `time gap`,
  `baseline rank` (when measured), and recorded `path IDs`.
- A claim of "recorded upstream cause" is permitted only when the receipt
  contains an explicit persisted causal edge. Otherwise the verdict remains
  `candidate cause — verify linked change`.
- Do not show CauseBench, `60%`, or `0%` in launch footage until the committed
  benchmark harness, baselines, and versioned results reproduce those numbers.

**Beat 5 — 13.0s to 15.0s · SIGNATURE + LOOP RESET**
- Wordmark `vestige` fades in bottom-left, tiny. Tagline: `memory that follows evidence, not resemblance.`
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
- **Honesty guardrails (non-negotiable):** source every visible candidate,
  rank, entity, timestamp, and path from one Backfill receipt. Label a seeded
  demonstration as a fixture. Publish benchmark numbers only with their
  committed harness, baselines, dataset version, and reproducible output.
- **Loop length:** 15s is the sweet spot for X autoplay + a clean GIF under
  ~8MB. If the GIF is too heavy, cut Beat 1 to 2s and land at 13s total.

---

## The X post it pins to

> Similarity search finds what *looks like* your bug.
>
> But an upstream change does not always resemble the failure it precedes.
>
> So I built memory that reaches *backward in time* through recorded entities
> and shows you the receipt for the upstream candidate it found. 🧵👇
>
> [pinned: the 15s clip]

(Then the thread: the wall-of-zeros chart, the repro command, the "here's where
it's weak" honesty tweet.)

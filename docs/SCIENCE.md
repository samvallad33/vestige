# The Science

> The research Vestige is built on, and exactly how much of it the 4.x engine implements

Vestige is the Causal Proof Engine and the operating system for AI agents. Its kernel
is Strata, an append-only, hash-chained, signed log. In the engine (recall, ranking,
pairing, explanation) there are no vectors, no RAG and no similarity. Every output
carries its proof: a record id, an edge path, or a receipt.

The research is still here. Memory science supplied the ideas behind how records
strengthen, fade, replay and get hidden. What changed in 4.x is how each idea is
computed: from recorded state and fixed formulas, so the same log gives the same answer
on any machine.

## Honesty convention

Every mechanism below carries one of three labels.

- **Implemented.** The code follows a published formula. Where the constants are
  Vestige's own approximations, the section says so.
- **Inspired by.** The mechanism borrows the idea. The computation is an engineering
  rule, not a model of the brain.
- **Heuristic.** A threshold or rule chosen because it is useful, with research as
  motivation only.

Nothing here is a claim that Vestige models a brain. The labels are the claim.

## What the 4.x engine uses

| Mechanism | Research basis | In 4.x | Label |
|-----------|----------------|--------|-------|
| Spaced-repetition scheduling | [FSRS](https://github.com/open-spaced-repetition/fsrs4anki), power-law forgetting | Every record has an FSRS card in the Strata kernel. Retrievability is derived on read and never stored | Implemented (formula shape, pinned constants) |
| Accessibility states | Multi-store models of memory | Active, Dormant, Silent and Unavailable are bands over retrievability | Heuristic |
| Explicit review | Testing effect, Roediger & Karpicke, 2006 | `promote` is an FSRS review rated Easy and `demote` one rated Again. A read is never a review | Inspired by |
| Dreaming | Sleep replay and consolidation, Diekelmann & Born, 2010; synaptic downscaling, Tononi & Cirelli, 2006 | `maintain` actions `dream` and `dream_compile` replay recorded edges and re-weight them | Inspired by |
| Suppression | Top-down inhibitory control of memory, Anderson et al., 2025 | `suppress` takes a record out of every read and keeps its bytes | Inspired by |
| Causal walk | None. Provenance and lineage tracing | A bounded backward search over recorded edges | Engineering |
| GhostLink | None. Graph distance and set overlap | Bridge and divergent lenses over recorded edges, with a deterministic sampler | Engineering |

The last two rows are not neuroscience, and the table says so. They are deterministic
graph algorithms over what the log recorded.

---

## FSRS scheduling in the Strata kernel

FSRS (the Free Spaced Repetition Scheduler) comes from the open-spaced-repetition
project. FSRS-6 is its sixth version: 21 parameters, and a power-law forgetting curve
whose shape is trainable. Power-law forgetting fits review data better than the
exponential curve older schedulers use. Vestige uses the FSRS formula shape. It does not
bundle the project's optimizer or its trained default parameters.

What the kernel (`strata-kernel`, module `fsrs`) does:

```
R(t, S) = (1 + FACTOR × t / S) ^ decay          decay = -(0.5 + w20)
```

- `R` is retrievability, the chance the record is still recalled. `S` is stability, the
  time in days for `R` to fall to about 90%. `t` is the time since the last review.
- A card holds stability, difficulty (1 to 10), a review count, a lapse count and a phase.
  A review rated Again, Hard, Good or Easy updates stability and difficulty with the
  FSRS recall and forget formulas.
- **Pinned constants.** The 21 weights are fixed in the source as milli-unit integers and
  versioned (`ALGO_V1`, `ALGO_V2`). A change means a new version, never an edit in place,
  so an old log replays under the constants it was written with. The store folds reviews
  and reads retrievability under version 2, where `FACTOR` is 0.242 and `w20` is 0, so
  the decay is -0.5.
- **Deterministic arithmetic.** Stability and difficulty are stored as Q32.32 fixed-point
  integers and evolved with a software math library, so replaying the log on another
  machine gives the same bits. Retrievability is derived only. It is never written to the
  log, and reading it appends nothing.
- **Time.** Reading retrievability measures whole days since the last explicit review, or
  since the record was created if it has only its ingest review. If neither clock exists,
  it falls back to the distance in log positions.

What this is not, so the label stays honest:

- The constants are Vestige's pinned approximations of the FSRS shape. They are not the
  optimizer-trained FSRS-6 defaults, and nothing is fitted to your reviews. v3 fitted a
  personal curve. After an upgrade, retention drifts slightly from what v3 showed (about
  0.06 lower after a week), because scheduling now follows this fixed curve. Imported v3
  cards keep their stability and difficulty.
- A review update measures elapsed time in **log positions**, not days. The wall-clock
  days enter only when retrievability is read. The gain from a review therefore depends on
  how much the log has moved since the card's last review.

Read the formulas in `crates/strata-kernel/src/fsrs.rs`.

## Accessibility states

`memory` action `state` reports one of four bands: **Active** at 0.7 and above,
**Dormant** from 0.4, **Silent** from 0.1, **Unavailable** below that. The thresholds are
heuristic. On a Strata log the three strengths the response lists (storage, retrieval,
retention) all equal the card's retrievability, so the band is a band over retrievability.

Nothing is deleted when a record fades. It is still found by its id and tags.

## Review, the testing effect, and why a read is not a review

The testing effect is the finding that retrieving information strengthens memory more
than re-studying it (Roediger & Karpicke, 2006).

Vestige keeps the idea and removes the self-reinforcement. A result being shown is not
proof that it was right or useful, so reads never touch a card. The only reviews are
explicit:

- `memory` `promote` folds a review rated Easy.
- `memory` `demote` folds a review rated Again. The record is not deleted. It fades
  faster from the lower stability.
- Both record an endorsement event for that exact content revision, with a receipt.

Label: Inspired by. This is a policy about which events count as practice.

## Dreaming

Sleep research suggests that replaying recent activity consolidates it
(Diekelmann & Born, 2010) and that connections get scaled back so that important ones
stand out (Tononi & Cirelli, 2006). 4.x uses those two ideas as fixed rules over recorded
edges:

- **`dream`** replays one page of a scope's live records (5 to 500, default 50). It
  considers only recorded edges at or above a strength floor (default 0.5) that join two
  records on the page. For each endpoint it folds one FSRS review whose rating comes from
  that card's own retention, stability and lapses, never from the record's content.
- **`dream_compile`** ranks the top records by retrievability and replays their recorded
  edges. An edge whose two ends were both replayed gains 0.1 strength (cap 1.0). A weak
  edge (under 0.5) with one end outside the replay set is scaled by 0.95. `corrects`
  edges among replayed records count as contradictions, and `derived_from` and
  `evidence_of` edges count as insights. No record is rewritten.

Both are deterministic: the same log and arguments give the same result. Neither
discovers new connections, since that would need similarity. Their `discovery` field says
`unavailable` and a replay with no recorded edge says why in `emptyBecause`.

Label: Inspired by. The phase names in `dream_compile` output (`NREM1_Triage`,
`NREM3_Consolidation`, `REM_Creative`, `Integration`) are labels for steps, not a
simulation of sleep stages.

## Suppression

Anderson et al. (2025, *Nature Reviews Neuroscience*) review how the brain actively
suppresses retrieval. `suppress` takes the idea in one respect: a suppressed record is
hidden from every read, and the log keeps its bytes. On a Strata log it cannot be undone,
and it is not erasure. Suppressing a GhostLink member withdraws the records composed from
it, each with its own receipt.

Label: Inspired by. The v3 behavior built on the same paper (a 24-hour reversal window
and a 72-hour neighbor-fading cascade, after Cervantes-Sandoval & Davis, 2020) is v3
engine behavior and is not part of a Strata log.

## Bounded causal walks

A walk starts at a record that holds a symptom, which you name, and goes backward over
edges the log recorded. This is lineage tracing, not neuroscience.

- Edges followed: from a record to what it is `derived_from`, and to the records that are
  `evidence_of` it, that it closed (`closed_by`), or that `touched` it. `forgotten_lesson`
  also follows `corrects`.
- Bounds: 8 hops and 500 nodes. The search is breadth-first, so the shallowest route wins,
  and ties break by id.
- No inferred edges, no shared-name matching, and nothing is written. An empty walk says
  why, from the edges the log holds.
- Results are hypotheses, not proven causes.

`forgotten_lesson` then ranks the fix or lesson records it reached by FSRS retrievability,
lowest first, and reports those below 0.5. `selftest` plants a known cause in a throwaway
copy and checks that the walk finds it and ignores decoys.

## GhostLink's deterministic sampler

GhostLink proposes pairs of records nobody has combined. Admission, ranking and
explanation use recorded structure only: ids, exact scope, type and tag identity, typed
edges, woven outcomes and FSRS state. Content text is shown as reading material and is
never read by a lens.

- **Bridge lens.** A pair is admitted when one member reaches the other within 3
  undirected hops over recorded `touched`, `derived_from` or `closed_by` edges and the
  pair was never woven. The shortest path (ties by id) is the proof. The score is
  `(1.5 + 1/hops) + 2/hops + 1.5 × novelty + trust + outcomeAdjustment`, where novelty is
  the mean of `1 / (1 + weaveDegree)` over the two members and trust is their mean FSRS
  retention.
- **Divergent lens.** A pair is eligible only if no recorded edge of any kind joins it.
  `Path_min` is the shortest undirected path over every recorded edge within 6 hops (a
  pair beyond that counts as 7). Divergence is
  `1 - |shared typed neighbors| / sqrt(|a's typed neighbors| × |b's typed neighbors|)`,
  computed on typed neighbor **ids**, not on any text, and the score is
  `min(Path_min, 7) × divergence`.
- **Forced juxtaposition.** If either member has no typed neighbors, nothing can be
  measured, so the pair gets no score. A deterministic sampler picks it. Members are
  ordered never-woven first, then by retention (read at the log's own head clock), then by
  id, grouped by exact scope, type and creation month, and interleaved round-robin. A fixed
  pairing schedule then walks the group so that every unordered pair appears exactly once
  and no member appears twice on a page. A page is a function of the log head and the
  filter alone. Nothing is argmaxed and nothing is random.
- Imported v3 links (`legacy_inferred`) can only shorten a path or dampen a score. They
  never admit a pair and never raise a score.
- Every pair carries a claim boundary: a bridge pair is connected by recorded edges, and a
  divergent pair is joined by none. Neither proves causality or novelty beyond this log.
  Weave the outcome once you have tested it, and later proposals learn from it.

Label: Engineering. The divergence term has the same form as a set cosine, but it runs on
sets of record ids that the log recorded, never on vectors or words.

---

## v3 engine (not used by the 4.x default build)

Earlier versions of this page, and Vestige v3, described a retrieval engine that ranked
by resemblance. The code still exists behind the `legacy-sqlite` and `v3-engine` features
for the harnesses that test it, and none of it runs on a Strata log:

- **Prediction-error gating** (create, merge or reinforce by similarity thresholds):
  needs embeddings. A 4.x save never merges; the only reinforcement is an exact repeat of
  the same text, found by its canonical hash, never by resemblance.
- **Spreading activation** (Collins & Loftus, 1975) over embedding similarity: needs
  embeddings. 4.x follows only recorded edges.
- **Hybrid search with reciprocal rank fusion**, BM25 and FTS5 keyword search, the Nomic
  embedding model and the HNSW vector index: removed. Recall is by exact handle.
- **Synaptic tagging and capture** (Frey & Morris, 1997) with a retroactive time window:
  a Strata log records no capture events. `maintain` `importance_score` still scores text
  with the v3 heuristics but writes nothing.
- **Context-dependent retrieval** (Tulving & Thomson, 1973) by topic weights: it ranked
  search results. 4.x partitions by exact scope instead.
- **Dual-strength memory** (Bjork & Bjork, 1992) as separate storage and retrieval
  strengths: a Strata log has one derived value, retrievability, plus FSRS stability and
  difficulty. The stability-versus-retrievability split in FSRS is the nearest relative.
- **Retroactive Salience Backfill** and post-retrieval failure feedback: they joined
  records by shared names. `causal_walk` replaces them.
- **`recall` modes `reason` and `contradictions`** (trust scoring and contradiction
  detection by topic overlap): they return `similarity_disabled`.

## References

- FSRS: <https://github.com/open-spaced-repetition/fsrs4anki>
- Roediger, H. L. & Karpicke, J. D. (2006). Test-enhanced learning: taking memory tests improves long-term retention. *Psychological Science*, 17(3), 249-255.
- Diekelmann, S. & Born, J. (2010). The memory function of sleep. *Nature Reviews Neuroscience*, 11.
- Tononi, G. & Cirelli, C. (2006). Sleep function and synaptic homeostasis. *Sleep Medicine Reviews*, 10(1).
- Anderson, M. C. et al. (2025). Brain mechanisms underlying the inhibitory control of thought. *Nature Reviews Neuroscience*. DOI 10.1038/s41583-025-00929-y
- Cervantes-Sandoval, I. & Davis, R. L. (2020). Rac1 impairs forgetting-induced cellular plasticity. *Frontiers in Cellular Neuroscience* (v3 behavior only).
- Bjork, R. A. & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. [PDF](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_EBjork_1992.pdf) (v3 only).
- Tulving, E. & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. [Record](https://psycnet.apa.org/record/1973-31800-001) (v3 only).
- Frey, U. & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. [*Nature*, 385](https://www.nature.com/articles/385533a0) (v3 only).
- Collins, A. M. & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review*, 82 (v3 only).

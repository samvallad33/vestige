# Pre-registration: the Silent Rotation ablation

**Committed before the experiment was run.** Check `git log` on this file against the commit date of
`ABLATION.md` and `results/ablation-*/`. If the interpretation below was written after the numbers came
back, the commit order will say so.

This document exists because the current result has a confound I have not closed, and because the
interpretation of an experiment is worth nothing if it is chosen after the experiment.

---

## 1. The confound, stated against me

The `sync` (Vestige) arm differs from the `rag` arm on **three** axes at once, not one. Read
`harness/run-sync.sh` against `harness/run-rag.sh`:

| Axis | `sync` | `rag` |
|---|---|---|
| **ANCHOR** | reaches backward from the failure event (`vestige_backfill`, no query argument) | embeds a query the agent typed itself |
| **EDGE** | traverses a causal join authored by the harness in `prepare_trial.py` | no edge |
| **BUS** | writes findings back to a shared store (`vestige_log`) so the fleet can converge | none — `run-rag.sh` says so in its own header: *"There is NO shared bus and NO coordination"* |

The published caveats admit the EDGE. **They did not admit the BUS.** They do now.

A reviewer is entitled to read the current headline as *"coordinated agents beat uncoordinated agents"*
and conclude the memory architecture is doing none of the work. That reading is available from the
source, and I would rather name it than be handed it.

### What is already known about the bus, and what is not

Transcript forensics argue the bus did **not** supply the answer:

- Kimi K3 trial 1: all three `sync` agents called `vestige_backfill` at **turn 0** and the correct key
  was in that turn-0 result. The first `vestige_log` write by any agent was **turn 4**.
- Across 15 DeepSeek `sync` agent runs, no backfill output contained a peer-written finding.
- DeepSeek trial 1 agent `a0` had `used_vestige_log=False`, called backfill once, and won.

That is evidence, not a control. **The ablation is the control.** Forensics can be argued with; an arm
cannot.

---

## 2. The three new arms

Each isolates one axis. All run on GLM-5.2, 5 trials each, on the same trial set as the existing arms.

| Arm | Anchor | Edge | Bus | Isolates |
|---|---|---|---|---|
| `rag-bus` | typed query | no | **yes** | pure coordination — the reviewer's arm |
| `sync-noedge` | event | **no** | yes | the hand-authored causal edge |
| `anchor-dense` | event | no | no | event-anchoring alone, on the competitor's retrieval stack |

`anchor-dense` is the one I most want to see. It embeds the failure event and takes top-3 cosine — no
Vestige, no edge, no bus. If it matches `sync`, the finding belongs to nobody and is larger for it.

---

## 3. The decision table

Committed before any of these arms has been run. Thresholds are stated as a fraction of `sync`'s
converged-correct rate on the same trials.

| Observed | Pre-committed interpretation |
|---|---|
| `rag-bus` correct **≥ 80%** of `sync` | **The coordination bus explains most of the win.** The headline finding is rewritten. Vestige's claim narrows to "one implementation of a shared write bus." Stated in the README's first paragraph, not a footnote. |
| `anchor-dense` correct **≥ 80%** of `sync` | **Event-anchoring beats querying, independently of Vestige.** This is promoted to the headline and the product becomes an implementation detail. Best outcome for the work, worst for the vendor. |
| `sync-noedge` correct **≥ 80%** of `sync` | The hand-authored causal edge is not carrying the result. That claim is deleted; event-anchoring is kept. |
| All three **< 50%** of `sync` | The bundle is load-bearing and no single axis explains it. The mechanism is a conjunction, and I say so — this is a weaker and more honest claim than "the memory architecture did it." |
| Anything between 50% and 80% | Reported as indeterminate at this n. No narrative is constructed to bridge it. |

**If two conditions fire at once, both are reported.** No cherry-picking the flattering one.

---

## 4. Stopping rule

- 5 trials per arm. **Fixed in advance.**
- No extending an arm because the result is close to a threshold.
- No dropping a trial except for a *documented harness failure*, and any drop is published in
  `results/WITHHELD-contaminated/` with the reason, as previous exclusions were.
- Arm liveness (`tests/verify-arms.sh`, `tests/arm_liveness.py`) is asserted **before** any spend. An arm
  with zero successful retrievals is recorded as `memory_unavailable`, never as a loss.

## 5. What this does not test

- Corpus scale. Still 8 documents.
- Whether Vestige can *discover* a causal edge rather than traverse one. It cannot, and this does not
  change that.
- Anything at production scale.
- Model breadth for the new arms — GLM-5.2 only, chosen for cost.

---

## 6. Commitment

Whatever comes back gets published, in this repository, with the interpretation above applied as
written — including the outcomes that demote my own system. The trial where Vestige loses is already in
this dataset because I caught myself excluding it. This is the same commitment, made in advance instead
of in hindsight.

— Sam Valladares, 2026-07-22

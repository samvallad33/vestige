# The ablation: which axis of the Vestige arm carries the result

Companion to `PREREGISTRATION.md`. The interpretation applied below is the one committed in that
file **before** these arms ran. Check `git log` on `PREREGISTRATION.md` (committed Jul 22) against
the commit that added the ablation arms (`8bc4c5b`, before any result JSON) and against the commit
that adds this file. The order is the audit.

Every number here was read out of `results/ablation-glm-5.2/trial-*/{arm}.json`, not from memory.

---

## What the pre-registration asked

The `sync` (Vestige) arm differs from the plain `rag` arm on **three** axes at once: it anchors on
the failure event rather than a typed query, it traverses a hand-authored causal edge, and it writes
to a shared coordination bus. A reviewer is entitled to read the headline as "coordinated agents beat
uncoordinated agents" and conclude the memory architecture did none of the work. Three arms isolate
the three axes. All ran on GLM-5.2, 5 trials, on the same trial set (seed 2026, correct keys
k_yildun, k_bellatrix, k_wren, k_indus, k_dorado) as the published `sync` run.

| Arm | Anchor | Edge | Bus | Isolates |
|---|---|---|---|---|
| `rag-bus` | typed query | no | yes | pure coordination — the reviewer's arm |
| `sync-noedge` | event | **no** | yes | the hand-authored causal edge |
| `anchor-dense` | event | no | no | event-anchoring alone, on the competitor's stack |

## The result

Scored the same way as the main benchmark: an arm passes a trial only if the merged tree is green
**and** an independent production replay passes **and** the key is correct (`fleet_verdict ==
fixed_correctly`). Every arm's memory layer was alive on all five trials (`memory_layer_alive: true`,
zero retrieval errors), so none of these is a dead-backend non-measurement.

| Arm | correct + prod-safe | vs published `sync` (5/5, same trials) |
|---|---|---|
| `rag-bus` | **0/5** | `rag` was already 0/5; adding the bus changed nothing |
| `sync-noedge` | **0/5** | the full `sync` arm was 5/5 |
| `anchor-dense` | **0/5** | — |

Per trial, with each fleet's converged key:

| trial | correct key | `rag-bus` | `sync-noedge` | `anchor-dense` |
|---|---|---|---|---|
| 1 | k_yildun | green_but_voids_prod (all k_wezen) | failed_merge_conflict (k_pavo/k_pavo/k_sirius) | green_but_voids_prod (all k_wezen) |
| 2 | k_bellatrix | green_but_voids_prod (k_atlas/unset/k_atlas) | green_but_voids_prod (all k_vela) | failed_still_red (all unset) |
| 3 | k_wren | green_but_voids_prod (all k_vela) | failed_merge_conflict (k_lyra/k_sirius/k_lyra) | green_but_voids_prod (all k_vela) |
| 4 | k_indus | green_but_voids_prod (all k_draco) | green_but_voids_prod (all k_talc) | failed_merge_conflict (k_talc/unset/k_atlas) |
| 5 | k_dorado | green_but_voids_prod (all k_quartz) | green_but_voids_prod (all k_polaris) | failed_still_red (all unset) |

Cost: $1.5837 across the three arms (1,924,040 tokens; rag-bus $0.4655, sync-noedge $0.6125,
anchor-dense $0.5058).

## The pre-committed interpretation that fires

Each arm scored 0/5, which is 0% of `sync`'s converged-correct rate. The decision table has one row
for that, written in advance:

> All three **< 50%** of `sync`: the bundle is load-bearing and no single axis explains it. The
> mechanism is a conjunction, and I say so — this is a weaker and more honest claim than "the memory
> architecture did it."

So the pre-registered conclusion is **not** "the causal edge wins." It is: **anchoring, the causal
edge, and the coordination bus are load-bearing together, and no single one of them reproduces the
full system.** None of the three exoneration rows fired. In particular, the `sync-noedge ≥ 80%` row —
the one that would have said the causal edge is not carrying the result and deleted that claim — did
not fire, so the edge remains implicated rather than cleared.

## What the transcripts add, without overriding the table

Two mechanistic facts are visible in the arm JSONs and are worth stating, because they say *how* the
axes fail, not just that they do:

- **The causal edge is necessary.** `sync-noedge` removes only the edge (the cause memory is seeded
  without the shared `active_key` tag; everything else, including the bus, is identical to the winning
  arm). It collapsed from 5/5 to 0/5, and `vestige_backfill` returned `"causes": []` — "surfaced 0
  causal memories" — on every trial. Strip the edge and backfill goes blind. This is the falsification
  test from `FINDING.md` section, run as an arm.
- **Coordination alone amplifies the wrong answer.** `rag-bus` gave the similarity arm the same shared
  write bus `sync` uses. It did not help; all three agents converged on the decoy in 4 of 5 trials
  (k_wezen, k_vela, k_draco, k_quartz — unanimous) and shipped a green-but-voids-prod fix. The bus made
  the fleet agree faster on a wrong key. A shared bus over similarity retrieval is a coordination
  amplifier, not a correctness mechanism.

These are consistent with the conjunction conclusion: the edge is *necessary* (removing it alone kills
the result), but it is not *sufficient*, because `anchor-dense` (event anchoring, no edge, no bus) and
`rag-bus` (coordination, no edge) each also went 0/5. Necessary is a narrower claim than "the edge is
the win," and it is the one the data supports.

## What this does not show

- **n is small and the model is one.** 5 trials per arm, GLM-5.2 only. Both were fixed in the
  pre-registration; this is a mechanism probe, not a powered comparison.
- **The baseline was a separate execution.** The `sync` 5/5 this is measured against is the published
  GLM-5.2 run on the same seed and trial set, not a re-run inside this batch. The setup is deterministic
  (same seed, same keys, same corpus), but the model calls were a different execution. A fully
  self-contained batch (anarchy + rag + sync re-run alongside these three) would remove the last
  execution-provenance objection; it has not been run.
- **The causal edge is still hand-authored.** This ablation confirms the edge matters. It does not
  show Vestige can *discover* the edge from prose. The extractor does not derive the `active_key` join
  from the cause text; strip the tag and there is nothing to traverse. Traversal, not discovery, as
  disclosed throughout.

## Two things the harness caught during this run, published because they are the point

- The leak audit refused to run every trial until `.vestige-seed-noedge.sh` — a new file this ablation
  introduced that names the correct key — was added to the checkout-ignore list. The integrity gate
  caught its own author adding an answer-bearing file. Fixed in `runner.py`.
- `reset-repo.sh` was missing from the public harness entirely (the wrappers reference it; it lived only
  in the private working tree). Anyone reproducing from this repo would have hit it before the first
  API call. Fixed in `1abeb65`.

## Reproduce it

```
cd harness
PROVIDER=openrouter MODEL=z-ai/glm-5.2 MASTER_SEED=2026 N_TRIALS=5 \
  ARMS="rag-bus sync-noedge anchor-dense" TORTURE_REPO=<your torture-v3.5> \
  bash run-experiment.sh
```

Each arm's per-trial JSON carries `memory_layer_alive` and `retrieval_err_total`; check them before
quoting any score, as with every arm in this benchmark. The seed randomizes the correct key per trial,
so a different `MASTER_SEED` moves the answer and the leak audit re-runs.

## The short version

The Vestige arm's advantage is not one trick. Take away the causal edge and it goes blind; take away
the anchor or the bus and it also fails. The three together carry the result, the edge is necessary
within that bundle, and the edge is hand-authored. That is a smaller and more honest claim than the
headline, which is why it is the one the pre-registration committed to before the numbers came back.

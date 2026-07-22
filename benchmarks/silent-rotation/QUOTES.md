# What the agents actually said, and what it proves

Every quotation below is verbatim from a transcript in `results/`. The file it comes from is named
next to it. Nothing here is paraphrased.

The point of this file is that the numbers in `FINDING.md` are a consequence, and this is the cause.

---

## 1. The agent that got it wrong was not reasoning badly

This is the single most important thing in the dataset, and it is why the raw transcripts ship instead
of a summary table.

**GPT-5.6 Sol, run `gpt-5.6-sol-trial-1`, correct key `k_yildun`, dense cosine arm.**
All three agents converged on `k_wezen`, a decoy. From `transcript-rag-a1.json`:

> "I'm weighing whether to fix the memory to say that the key k_wezen is staging only. The note
> mentions a closed ticket about resetting the key id for k_wezen in staging without any rotation,
> which seems intentional... I've noticed that my memory indicates a reset to k_wezen, so I must set
> both configs to that"

**It read the words "staging only" and used the key anyway.** Not carelessness. Nothing it retrieved
contradicted the decoy, and a caveated answer beats no answer. The chain of inference is correct. The
evidence set was incomplete.

**Kimi K3, run `runB-trial-2`, correct key `k_regulus`, dense cosine arm.**
From `transcript-rag-a1.json`, committing to `k_sirius`:

> "Memory: `k_sirius` is the key id verification recovered with (OPS-522). **No other memory suggests a
> different key.**"

**It checked for a contradiction.** It went looking for something that would tell it otherwise. Nothing
came back, so it proceeded.

You cannot prompt-engineer around either of these. Both agents did the thing you would want an agent to
do.

---

## 2. Two agents on the same arm, in the same trial, disagreeing because they saw different pasts

**Kimi K3, run `runA-trial-1`, correct key `k_nashira`, mem0 arm.**

Agent a0 never surfaced the cause. From `transcript-mem0-a0.json`:

> "Both configs have blank key ids, left blank by an 'externalize secrets' refactor. The memory said:
> OPS-522 the key id was reset to `k_antares` and verification recovered. That suggests `k_antares` is
> the key the platform shipped with / should use."

Agent a2, same arm, same trial, got a richer retrieval and became suspicious of the same document. From
`transcript-mem0-a2.json`:

> "Hmm, that says k_antares was staging-only. That memory may be a hint that the production key id is
> something else, or it might be a red herring. Let me search memory more specifically."

**a0 wrote `k_antares`. a1 and a2 wrote `k_nashira`. Two merge conflicts. Nothing shipped.**

They were not disagreeing about how to interpret a shared reality. Their memory layer handed them
different realities.

---

## 3. An agent giving up on its memory system entirely

**Kimi K3, run `runB-trial-3`, correct key `k_orion`, mem0 arm.** From `transcript-mem0-a1.json`, after
two failed queries:

> "The memory isn't directly relevant to 'no active signing key selected'. **Let me look at the code.**"

It concluded the memory layer had nothing and went to the source. It needed six queries in total and
first surfaced the correct key on call four.

This is what a 6.1% first-call hit rate feels like from inside the agent.

---

## 4. The trap working exactly as designed, and one agent escaping it

**Kimi K3, run `runB-trial-3`, dense cosine arm.** From `transcript-rag-a2.json`:

> "Team memory: OPS-522 says the key id was reset to `k_antares` and verification recovered. But that
> was a staging bo..."

That agent argued itself out and got the right answer. In `runB-trial-2`, the identical reasoning on the
identical decoy is what killed the fleet. **The difference between a pass and a merge conflict was
whether one agent finished a sentence.**

---

## 5. What the winning arm's agents said, and what is missing from it

**Kimi K3, run `runB-trial-3`, Vestige arm.** All three agents, from `transcript-sync-a{0,1,2}.json`:

> "The failure is 'no active signing key selected' from `identity-service/src/signer.ts:31`. Let me read
> the relevant files and check the shared memory for context."

Then one `vestige_backfill` call each, no query argument, correct key on the first call for all three.

**There is no passage anywhere in the Vestige transcripts where an agent weighs a decoy.** Not because
the agents were better. Because no decoy arrived. In `runB-trial-3` the backfill results contained zero
decoy keys across all three agents, while the dense cosine and supermemory arms received a decoy in
nearly every result.

---

## 6. The receipt that reports its own blind spot

The Vestige tool returns this alongside the answer, from `runA-trial-1/transcript-sync-a0.json`:

```json
{
  "content_preview": "Rotation runbook, Q3: we migrated the live signer to k_nashira ...",
  "age_days_before_failure": 4.0,
  "shared_entities": ["active_key"],
  "similarity_rank": 6,
  "backfill_score": 1.51,
  "reason": "Reached back 4.0d to a quiet memory sharing 1 entity (active_key) with the
             failure; it ranked #6 on similarity, so semantic search would have missed it."
}
```

`similarity_rank: 6` is the tool stating that the fact it just returned is one a similarity search would
have buried. That number is self-reported by Vestige, so treat it as a claim rather than a measurement.
The independent check is `tests/bm25_baseline.py`, which reproduces the ranking from the corpus without
involving Vestige at all, and puts the cause at 7th of 8.

---

## 7. The harness bug that hurt two competitors, and what it looked like

**Kimi K3, run `runB-trial-2`, correct key `k_regulus`, Zep arm (now in `results/WITHHELD-contaminated/`).** All three agents received this graph
fact:

> "[1] k_antares was decommissioned in favor of k_nashira as the live signer."

Neither `k_antares` nor `k_nashira` exists in that trial's corpus. Both are keys from the **previous**
trial. My harness never flushed Zep's graph between trials, so a superseded fact was asserted as current
and every agent believed it.

That cell is withheld. It is a measurement of my harness, not of Zep. After fixing the flush and adding
a per-trial graph namespace, Zep was re-run clean in `runB-trial-3`: two of three agents wrote a decoy,
the third wrote nothing, tests passed, production replay failed.

The same class of bug hit mem0 through a second store at `~/.mem0` that persists alongside the
configured one. Fixed and re-run: mem0 converged correctly in `runB-trial-3` and split in `runB-trial-2`.

---

## What all of this proves, stated carefully

**It proves:** at turn zero an agent knows only the symptom, so it queries the symptom. In this incident
class the symptom query returns a document that resembles the failure and buries the one that explains
it, and that holds under dense cosine and BM25 alike. Across 6 models and 25 trials, a query-based arm's
first memory call surfaced the deciding fact 7 times out of 114. Anchoring on the failure event instead
of a query did it 65 out of 65. In a fleet, that first-call gap becomes either a merge conflict or a
unanimous wrong answer that passes tests and breaks production.

**It does not prove:** that Vestige discovers causal relationships. The edge it traverses is authored by
the harness at `harness/agent/prepare_trial.py:187`. It does not prove Vestige is better at retrieval,
because a plain dense cosine baseline costs less per run and, on Kimi K3, never converged wrong. It does not prove anything at
production scale, because the retrievable corpus is 8 documents. And it does not isolate which of
Vestige's three properties (event anchoring, causal traversal, temporal validity) is doing the work,
because that ablation has not been run.

**The one sentence:** the first query fails for almost everyone, almost every time. The only thing that
varied was whether the memory layer required a first query at all.

# Every claim, and where to check it

Companion to `FINDING.md`. Each row states a claim and the file that proves it. All transcripts are in
`results/`. Nothing below is summarized from memory; every number was read out of the JSON.

Five models. Twenty-two trials. Two hundred and ten agent transcripts.

---

## 1. The headline table

Three agents work the same failing test. The fix needs a signing key id that is randomized per trial
from a 50-key keyring and appears in no file the agents can read. It exists only in the memory layer.

For every arm in every trial we asked one question: **did all three agents write the same key, and was
it the right one?**

| arm | converged on the CORRECT key | converged on a WRONG key | split, no consensus | n |
|---|---|---|---|---|
| anarchy (no memory) | 0/22 | **16/22** | 6/22 | 22 |
| rag (dense cosine) | 4/21 | **10/21** | 7/21 | 21 |
| supermemory | 5/6 | 0/6 | 1/6 | 6 |
| mem0 (clean cells only) | 1/2 | 0/2 | 1/2 | 2 |
| hindsight | 0/3 | 0/3 | 3/3 | 3 |
| Zep/Graphiti (clean cell only) | 0/1 | 0/1 | 1/1 | 1 |
| **sync (Vestige)** | **19/21** | **0/21** | 2/21 | 21 |

Models: Kimi K3, MiniMax M3, GLM 5.2, GPT-5.6 Sol, DeepSeek V4 Flash.
Reproduce: read `fix_directions` and `correct_kid` out of every `trial-*/{arm}.json` and `manifest.json`.

### There are two failure modes and only one of them is visible

**Split.** The agents disagree, write conflicting changes, the merge fractures, nothing ships. Loud.
You find out immediately.

**Converged wrong.** The agents all agree, on a decoy. The tests go green. The merge is clean. The
production replay fails. **This is the one that ships.**

Look at that middle column again. A memoryless fleet converged on a wrong key **16 times out of 22**.
The dense-cosine fleet did it **10 times out of 21**. Vestige did it **zero times out of 21**.

Not "rarely". Not "less often". Zero.

---

## 2. The agents were not reasoning badly. They were reasoning correctly from wrong evidence.

This is the part that does not show up in any aggregate score, and it is why the transcripts are
published rather than a summary table.

### GPT-5.6 Sol, trial 1, correct key `k_yildun`, dense-cosine arm

All three agents converged on `k_wezen`. Here is agent a1 at the moment it decided, verbatim from
`raw-sol/trial-1/transcript-rag-a1.json`:

> "I'm weighing whether to fix the memory to say that the key k_wezen is staging only. The note
> mentions a closed ticket about resetting the key id for k_wezen in staging without any rotation,
> which seems intentional. There's a need to configure both to k_wezen while considering possible
> environment fallback... I've noticed that my memory indicates a reset to k_wezen, so I must set both
> configs to that"

Read that twice. **The agent noticed the memory said "staging only" and used it anyway.** Not because
it was careless, but because nothing it retrieved contradicted the decoy, and a caveated answer beats
no answer. It reasoned its way to the wrong key through a correct chain of inference over an incomplete
evidence set.

### GLM 5.2, trial 1, correct key `k_yildun`, dense-cosine arm

All three converged on `k_wezen`. From `raw-glm/trial-1/transcript-rag-a1.json`:

> "The memory mentions a key id `k_wezen` in config."

That is the whole deliberation. The retrieval returned one key-shaped fact and the agent took it.

### GLM 5.2, trial 3, correct key `k_wren`, dense-cosine arm

All three converged on `k_vela`. From `raw-glm/trial-3/transcript-rag-a0.json`:

> "The first memory mentions 'reset the key id to k_vela'."

Different trial, different decoy, identical shape.

### Kimi K3, trial 1, correct key `k_nashira`, mem0 arm

Here the fleet split rather than converging, and you can watch exactly why. Agent a0 never surfaced the
cause, from `experiment-1337-5/trial-1/transcript-mem0-a0.json`:

> "Both configs have blank key ids, left blank by an 'externalize secrets' refactor. The memory said:
> OPS-522 the key id was reset to `k_antares` and verification recovered. That suggests `k_antares` is
> the key the platform shipped with / should use."

Its teammate a2 got a richer retrieval and became suspicious of the same document:

> "Memory said: 'OPS-522: a staging box failed token verification after a hand-edited config; on-call
> reset the key id to k_antares and verification recovered, noting it was staging-only with no
> rotation.' Hmm, that says k_antares was staging-only. That memory may be a hint that the production
> key id is something else, or it might be a red herring. Let me search memory more specifically."

**a0 picked `k_antares`. a1 and a2 picked `k_nashira`. Two conflicts. Nothing shipped.**

They did not disagree about how to interpret a shared reality. Their memory layer gave them different
realities. One of them was working from a past in which `k_antares` was the answer.

### What the Vestige agents said on the same task, same models

Nothing like the above. From `raw-glm/trial-3/transcript-sync-a1.json`:

> "The issue is that `activeKid` is blank in identity-service config. The 'externalize-secrets refactor'
> pulled out the concrete key id. I need to find which key id the verifier expects."

Then it called `vestige_backfill` once and went to the fix. **There is no passage anywhere in the sync
transcripts where an agent weighs a decoy**, because no decoy arrived.

---

## 3. Why the symptom query fails

Every agent in every arm opened by describing the symptom. Here is what the very first retrieval
returned, verbatim with the tool's own similarity scores, from
`experiment-1337-5/trial-1/transcript-rag-a0.json`:

> Top-3 memories by semantic similarity (cosine, nomic-embed-text):
> **[1] (similarity 0.641)** Closed ticket OPS-522: a staging box failed token verification after a
> hand-edited config; on-call reset the key id to k_antares and verification recovered. Staging-only,
> nothing rotated.
> **[2] (similarity 0.629)** Ledger verifier threw a 500 on charge when the token had no 'kid' header...

The top hit for "my charge test is failing on token verification" is a document about a token
verification failure fixed by setting a key id. Excellent match for the symptom. Wrong answer.

The document that actually explains the failure reads:

> "Rotation runbook, Q3: we migrated the live signer to k_nashira and let the old material age out.
> Every credential minted since the cutover carries the new kid."

It never mentions charges, tests, or failures. It is a routine operational note written four days before
anything broke.

Measured against trial 1's own eight-document corpus, using the exact queries the agents typed
(`tests/bm25_baseline.py`, runnable, no API key needed):

| query (verbatim from a transcript) | who asked it | dense cosine | BM25 |
|---|---|---|---|
| `end-to-end charge test failing identity gateway ledger session token verify` | rag a0, first query | 7 of 8 | 7 of 8 |
| `end-to-end charge test failing session token identity gateway ledger` | zep a0, first query | 7 of 8 | 7 of 8 |
| `default key kid keyring shipped platform active key` | zep a0, third query | 3 of 8 | 7 of 8 |
| `which key id active signing key k_antares externalize secrets refactor blank config` | mem0 a0, second | 3 of 8 | 4 of 8 |
| `no active signing key selected signing key rotation identity-service keyring` | supermemory a0, first | 4 of 8 | 6 of 8 |

**When you describe the symptom, the cause ranks last.**

We ran BM25 specifically because the answer is a rare literal token, which is supposed to be
exact-match's home ground. It ties dense on the symptom queries and is one to four ranks worse on the
queries that actually worked. The decoys share the target's exact vocabulary, so lexical scoring rewards
them just as much. What separates the right key from the wrong one is *which one is current*, and
recency has no lexical signal.

This is not a tuning problem. The corpus contains look-alike documents that are **more** similar to the
failure than the cause is, because they describe the same symptom. Sharpening the embedding raises the
look-alikes. Better similarity matching makes an anti-correlation worse.

---

## 4. The same trial, run twice

`experiment-1337-5/trial-1` and `experiment-1337-3/trial-1` are the same trial. Same master seed, same
trial index, same correct key `k_nashira`, same nine-memory corpus, same model, same 8-turn budget.
Only the run differs.

| | dense-cosine arm | Vestige arm |
|---|---|---|
| Run A | `fixed_correctly`, all three on `k_nashira` | `fixed_correctly`, all three on `k_nashira` |
| Run B | **`failed_merge_conflict`**, a0 `k_nashira`, a1 `k_nashira`, a2 **`k_antares`** | `fixed_correctly`, all three on `k_nashira` |

Plain cosine won the trial once and lost the identical trial the next time. One agent of three phrased
its queries slightly differently, never surfaced the cause, took the decoy, and voided the fleet.

Vestige did the same thing both times: one call per agent, same answer, zero conflicts.

**That is the finding, not a caveat.** A result that flips on rerun is a coin flip, and a coin flip is
what you are deploying when retrieval depends on query phrasing.

---

## 5. Why the model matters, and why that helps rather than hurts

Number of times each model's dense-cosine agents re-queried before committing:

| model | rag_search calls per agent | rag outcome |
|---|---|---|
| GLM 5.2 | 1.00 | 0/5 |
| DeepSeek V4 Flash | 1.00 | 1/5 |
| GPT-5.6 Sol | 1.13 | 0/5 |
| Kimi K3 | 2.17 | wins some, loses some |

**Similarity retrieval's success is explained almost entirely by whether the model bothers to
reformulate.** Ask once and accept the answer, and you get the decoy. Kimi K3 re-queries twice on
average and sometimes escapes.

Vestige is **1 call per agent on every model tested**, and the key was surfaced on 47 of 48 successful
calls, because there is no query to reformulate. `vestige_backfill` takes no query argument. It anchors
on the failure event and walks the causal edge backward.

So the honest cross-model statement is not "this works everywhere." It is: **the memory layer's
behaviour was invariant across five models and the baseline's was not.**

---

## 6. What this does not show

Stated plainly, because an interesting result is worth more than an overclaim. Every one of these was
found by attacking our own work today.

1. **The causal edge is hand authored.** `prepare_trial.py:187` tags the cause
   `keyring_rotation,active_key`, `:198` tags the failure `active_key,crash`, and none of the seven
   distractors carry it. Remove that string and backfill returns `causes: []`. **This measures traversal
   of an existing edge, not discovery of one.** Vestige's extractor does not derive that edge from prose.
2. **We are not the first to run multiple agents on one repo.** CooperBench (arXiv 2601.13295) runs two
   coding agents concurrently across 652 tasks and scores merge compatibility, finding roughly 30% lower
   success working together than alone. MultiAgentBench (ACL 2025) runs 1/3/5/7 agents over an explicit
   shared memory layer. Ask WhAI (arXiv 2511.14780) measures belief divergence across agents on a shared
   record. The narrow thing that appears to be unmeasured is **treating the memory layer itself as the
   independent variable**: same fleet, same task, same model, swap only the memory backend, score
   inter-agent agreement.
3. **The dense-cosine baseline is not a strawman and sometimes beats us.** It converged correctly 4 times
   and was cheaper than Vestige in both Kimi K3 trials measured ($0.4429 vs $0.4918 in trial 1).
4. **The retrievable corpus is 8 documents.** Nothing here is measured at production scale, and the two
   scaling axes run in opposite directions: more documents push a cause that already ranks 7th further
   down a similarity list, which makes N=8 the *easy* case for the RAG arms, while Vestige's backfill
   degrades as more memories share the join entity, which is 2 of 9 here.
5. **`similarity_rank` in the Vestige receipt is self-reported by Vestige.** The independent check is
   that the RAG arm's own tool output shows the decoy outranking the cause, which is visible in the
   transcripts.
6. **The Vestige arm has a `vestige_log` write tool no competitor arm was given.** It did not produce the
   result: in every agent run, the key came from that agent's own single backfill call, and every log
   write happened strictly later in the transcript.
7. **The competitor corpus is built content-only**, tags discarded (`runner.py:399`). Restoring tags was
   measured and flips zero outcomes; the cause still never reaches rank 1 and still ranks 7th or 8th on
   symptom queries in both conditions.
8. **mem0's cells in the pre-fix trials are contaminated.** mem0 persists to `~/.mem0` in addition to its
   configured store, which the harness cleared but `~/.mem0` was not, so it carried facts across trials.
   That disadvantaged mem0. Fixed, and flagged wherever those cells appear.
9. **One arm ran below its documented behaviour.** The tool description shown to Zep's agents said
   "graph RRF", which overstates what `EDGE_HYBRID_SEARCH_RRF` does. Corrected after these runs.
10. **The author built the benchmark, the harness, and one of the products measured.** Which is why the
    raw transcripts ship rather than a summary.

---

## 7. Check it yourself

```
harness/                 the runner, the fleet driver, the trial generator
tests/bm25_baseline.py   reproduces the ranking table, no API key needed
results/trial-1/         all 7 arms, all 21 agent transcripts
```

Start here if you are auditing: every arm JSON carries `memory_layer_alive` and `retrieval_err_total`.
An arm whose backend never answered is a missing measurement, not a retrieval loss, and should never be
read as a score. Where that telemetry predates a run, liveness was reconstructed from the transcripts by
counting successful tool results, and that is stated wherever it applies.

---

## The short version

A memory system can be right for most of your agents and still be unusable, because a fleet is a
conjunction and the statistic that governs it is a minimum, not a mean.

And the failure you should actually fear is not the one where your agents argue. It is the one where
they all agree, the tests pass, and the key is wrong.

**Zero out of twenty-one.**

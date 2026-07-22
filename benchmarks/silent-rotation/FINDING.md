# One misled agent voids a three-agent fleet

**What happens when three AI agents fix the same bug, and only two of them know the truth.**

Trial 1 of the Silent Rotation benchmark. Kimi K3, seven memory configurations, one randomized secret.
Every number and every quotation below comes from the raw agent transcripts, which are published
alongside this document in `results/trial-1/`.

---

## The result in one paragraph

Three agents work the same failing test. The fix requires knowing which cryptographic signing key is
currently live. That key is randomized per trial and appears in no file the agents can read. It exists
only in the memory layer.

Four of the six memory configurations put the correct answer in front of at least two of the three
agents. Three of them lost anyway. In each losing case exactly one agent out of three never received
the correct fact, confidently adopted a planted decoy, and wrote a conflicting change. The merge
fractured and nothing shipped.

Two agents right plus one misled equals zero production-safe fleets.

---

## Why this is not a retrieval story

The reflex when a memory system loses is to say its retrieval was bad. That is not what happened here.
In every losing arm the correct memory was in the store, and most agents found it.

Here is the complete trial-1 outcome table. `fixed_correctly` means the tests went green **and** an
independent production replay passed **and** the key was right.

| arm | what it is | verdict | retrievals | fleet cost | wall clock |
|---|---|---|---|---|---|
| anarchy | no memory tool at all (control) | green_but_voids_prod | 0 | $0.5668 | 162.04s |
| rag | dense cosine, nomic-embed-text, top 3 | **fixed_correctly** | 6 | $0.4429 | 140.68s |
| sync | Vestige causal backfill | **fixed_correctly** | 3 | $0.4918 | 137.29s |
| mem0 | mem0ai OSS, LLM extraction on ingest | failed_merge_conflict | 7 | $0.4888 | 154.25s |
| supermemory | supermemory, vendor defaults | **fixed_correctly** | 7 | $0.5119 | 143.81s |
| hindsight | semantic + BM25 + cross-encoder rerank | failed_merge_conflict | 6 | $0.4974 | 147.78s |
| zep | Graphiti temporal knowledge graph | failed_merge_conflict | 13 | $0.5341 | 135.28s |

Read the losing rows by agent instead of by arm and the pattern is identical every time.

| arm | agent a0 | agent a1 | agent a2 | conflicts |
|---|---|---|---|---|
| mem0 | **k_antares** | k_nashira | k_nashira | 2 |
| hindsight | k_nashira | **k_antares** | k_nashira | 2 |
| zep | **k_antares** | k_nashira | k_nashira | 2 |

The correct key was `k_nashira`. In all three arms, two agents out of three got it. In all three arms,
the third agent took `k_antares`, a decoy planted in a closed staging ticket. Different arm, different
agent index, same shape.

---

## The agent that failed was not reasoning badly

This is the part that matters, and it is why the transcripts are published rather than summarized.

The agent that got it wrong was not sloppy. It reasoned carefully and correctly from the evidence it
was handed. Here is mem0's agent a0, in its own words, at the moment it committed:

> "Both configs have blank key ids, left blank by an 'externalize secrets' refactor. The memory said:
> OPS-522 the key id was reset to `k_antares` and verification recovered. That suggests `k_antares` is
> the key the platform shipped with / should use."

That is a good inference. The retrieval had returned exactly one key-related fact, and that fact said
`k_antares` fixed a verification failure. a0 searched twice. Neither search returned the rotation
runbook naming `k_nashira`.

Now here is its teammate a2, on the same arm, in the same trial, working from a retrieval that did
surface more:

> "Memory said: 'OPS-522: a staging box failed token verification after a hand-edited config; on-call
> reset the key id to k_antares and verification recovered, noting it was staging-only with no
> rotation.' Hmm, that says k_antares was staging-only. That memory may be a hint that the production
> key id is something else, or it might be a red herring. Let me search memory more specifically."

a2 got suspicious and kept looking. a0 had no reason to be suspicious, because nothing it retrieved
contradicted the decoy.

The two agents did not disagree about how to interpret a shared reality. **Their memory layer gave them
different realities.** One of them was working from a past in which `k_antares` was the answer.

hindsight's a1 shows the same thing from the other side. It searched twice, and its second query was
well formed:

> query: `externalize secrets refactor blanked key id which kid did the platform ship with activeKid trustedKid`

That query never surfaced `k_nashira` either. Its reasoning then walks straight into the decoy:

> "The question: which kid should both sides use? The config comment says 'entry the platform shipped
> with' dangling comment. The hindsight memory says: 'Closed ticket OPS-522 ... on-call reset the key
> id to k_antares and verification recovered.'"

Same arm, same corpus, same model, same turn budget. Its two teammates asked differently and got the
truth.

---

## Why the symptom query fails

Every agent in every arm opened with a query describing the symptom. Here is what the very first
retrieval returned for the RAG arm's a0, verbatim, with the similarity scores the tool reported:

> Top-3 memories by semantic similarity (cosine, nomic-embed-text):
> **[1] (similarity 0.641)** Closed ticket OPS-522: a staging box failed token verification after a
> hand-edited config; on-call reset the key id to k_antares and verification recovered. Staging-only,
> nothing rotated.
> **[2] (similarity 0.629)** Ledger verifier threw a 500 on charge when the token had no 'kid' header...

The top hit for "my charge test is failing on token verification" is a document about a token
verification failure that was fixed by setting a key id. It is an excellent match for the symptom. It
is also the wrong answer.

The document that actually explains the failure reads:

> "Rotation runbook, Q3: we migrated the live signer to k_nashira and let the old material age out.
> Every credential minted since the cutover carries the new kid."

Notice that it shares almost no surface vocabulary with "end-to-end charge test failing on session
token verify." It does not mention charges, tests, or failures. It is a routine operational note
written four days before anything broke.

We measured this directly. Against the eight-document retrievable corpus, using the exact queries the
agents actually typed:

| query (verbatim from a transcript) | who asked it | dense cosine | BM25 |
|---|---|---|---|
| `end-to-end charge test failing identity gateway ledger session token verify` | rag a0, first query | 7 of 8 | 7 of 8 |
| `end-to-end charge test failing session token identity gateway ledger` | zep a0, first query | 7 of 8 | 7 of 8 |
| `default key kid keyring shipped platform active key` | zep a0, third query | 3 of 8 | 7 of 8 |
| `which key id active signing key k_antares externalize secrets refactor blank config` | mem0 a0, second query | 3 of 8 | 4 of 8 |
| `no active signing key selected signing key rotation identity-service keyring` | supermemory a0, first query | 4 of 8 | 6 of 8 |

Measured against trial 1's own eight-document corpus, exported from the seeded store. Okapi BM25 with
k1=1.5 and b=0.75; dense is nomic-embed-text, the same embedder the RAG arm used.

**When you describe the symptom, the cause ranks last.** Not sixth. Last, or next to last, out of
eight.

This is not a tuning problem, and it is worth being precise about why. The cause is not merely
dissimilar to the failure. The corpus contains look-alike documents that are *more* similar to the
failure than the cause is, because they describe the same symptom. Sharpening the embedding pushes the
look-alikes up, not the cause. **Better similarity matching makes an anti-correlation worse, not
better.**

We ran the BM25 lexical baseline specifically because the answer is a rare literal token, which is
supposed to be exact-match's home ground. It is not better. It ties dense cosine on the symptom queries,
where both bury the cause at rank 7 of 8, and it is worse by one to four ranks on every query that
actually worked. The decoys share the target's exact vocabulary, so lexical scoring rewards them just
as much. The thing that separates the right key from the wrong one is *which one is current*, and
recency has no lexical signal at all.

There is also a practical limit on exact match here. The live key is randomized per trial from a
50-key keyring and appears in no readable file, so you can only type it into a query after you already
know the answer.

---

## What the winning arms actually did

Three arms shipped a correct, production-safe fix. They did it in visibly different ways.

**rag** and **supermemory** won by reformulating. Every one of their agents opened with a symptom query,
got the decoy, and then wrote a second query that described the *shape of the answer* rather than the
shape of the problem. supermemory's a0 opened with `no active signing key selected signing key rotation
identity-service keyring`. It had already guessed that a rotation was involved before it searched.

That works. It also means the model is spending reasoning budget compensating for retrieval, and that
it has to suspect the answer's category before it can find it.

**sync**, the Vestige arm, did not formulate a query. `vestige_backfill` takes no query argument. Each
agent called it exactly once, and the tool returned this:

```json
{
  "causes": [{
    "content_preview": "Rotation runbook, Q3: we migrated the live signer to k_nashira ...",
    "age_days_before_failure": 4.0,
    "shared_entities": ["active_key"],
    "similarity_rank": 6,
    "backfill_score": 1.51,
    "reason": "Reached back 4.0d to a quiet memory sharing 1 entity (active_key) with the
               failure; it ranked #6 on similarity, so semantic search would have missed it."
  }]
}
```

Three agents, three calls, three identical results. Zero merge conflicts. The receipt states its own
similarity rank, which is the tool reporting that the fact it just returned was one a similarity search
would have buried.

**Be precise about what that demonstrates.** The `active_key` entity link between the failure and the
cause is authored by the benchmark harness (`prepare_trial.py:187` and `:198`). Vestige's extractor does
not derive it from the prose. So this shows that *traversing an existing causal edge* returns the cause
in one call with no query formulation. It does **not** show that the edge can be discovered
automatically. That is a separate, unproven claim, and this benchmark does not test it.

---

## The control that makes the point

The most useful row in the table is the one with no memory at all.

**anarchy** had all three agents converge unanimously on `k_olivine`. Zero merge conflicts. Perfect
agreement. The tests went green. The production replay failed.

So convergence is not the prize. A fleet can agree perfectly and be perfectly wrong. What the memory
layer has to deliver is convergence *on the truth*, and the failing arms show that delivering it to two
out of three agents is worth exactly as much as delivering it to none.

anarchy also did something worth its own study: with no memory, it did not fail loudly. It shipped a
green test suite that breaks production. That result has nothing to do with any memory product, which
is why we find it credible.

---

## Why the usual metrics cannot see this

Retrieval quality is normally reported as recall@k, MRR, or nDCG. All three are **means over queries**.

Fleet success is not a mean. It is a conjunction. Every agent editing shared state has to hold the same
correct belief, or the merge breaks. The statistic that governs the outcome is the **minimum** over
agents, not the average.

If each agent independently has a 90% chance of retrieving the deciding fact:

| fleet size | probability every agent gets it |
|---|---|
| 1 | 0.90 |
| 3 | 0.73 |
| 5 | 0.59 |
| 10 | 0.35 |

A memory layer that benchmarks at 90% recall looks excellent and produces a coin flip at fleet size 5.
Nothing in the standard metric suite surfaces that, because the standard metric suite was designed for
one agent asking one question.

The measurements a fleet actually needs are different:

- **Correct convergence.** Did *every* agent independently reach the production-safe answer?
- **Retrieval consistency.** How many distinct purported truths did the memory layer hand out across
  agents working the same incident?
- **Query sensitivity.** How much did the returned evidence change with small changes in wording?
- **Fleet-safe completion.** Did the merged result stay conflict free and pass an external replay?
- **Cost per converged safe fix.** Retrievals, turns, tokens, seconds and dollars until the whole fleet
  was correct.

---

## What this does not show

Stated plainly, because the interesting result is worth more than an overclaim.

1. **The causal edge is hand authored.** `prepare_trial.py:187` tags the cause `keyring_rotation,active_key`
   and `:198` tags the failure `active_key,crash`. None of the seven distractors carry it. Remove that
   string and backfill returns `causes: []`. This measures traversal, not discovery.
2. **The outcome column does not separate the winners.** rag, sync and supermemory all scored
   `fixed_correctly`. Plain dense cosine over eight documents matched the memory products on the original Kimi K3 trials, and it was the
   **cheapest arm in the trial** at $0.4429. There is no "Vestige beats RAG" result here to defend.
3. **The retrievable corpus is eight documents.** Nothing here is measured at production scale, and the
   two scaling axes run in opposite directions. More documents can only push a cause that already ranks
   seventh further down a similarity list, which makes N=8 the *easy* case for the RAG arms. Vestige's
   backfill degrades as more memories share the join entity, which is two of nine here.
4. **This is one trial of one scenario on one model.** The per-agent divergence pattern repeated across
   three independent arms within it, and reproduced in a later run, but n is small and it is one bug
   shape.
5. **`similarity_rank` in the receipt is self-reported by Vestige.** The independent check on it is that
   the RAG arm's own tool output showed the decoy outranking the cause on symptom queries, which is
   visible in the transcripts.
6. **The Vestige arm has a write-back coordination tool** (`vestige_log`) that no competitor arm was
   given. It did not produce the result: in all six agent runs across both trials, each agent got the
   key from its own single backfill call, and every log write happened strictly later in the transcript.
7. **The competitor corpus is built content-only**, with the `tags` field discarded (`runner.py:399`).
   Restoring tags to competitors was measured and flips zero outcomes; the cause still never reaches
   rank 1, and on symptom queries it still ranks seventh or eighth in both conditions.
8. **The author built the benchmark, the harness, and one of the products measured.** Which is why the
   raw transcripts are published rather than a summary table.

---

## Reproducing it

Everything needed to check this is in the repository.

```
harness/            the runner, the fleet driver, and the trial generator
harness/agent/prepare_trial.py     seeds the corpus and randomizes the key
results/trial-1/    all 7 arm results and all 21 agent transcripts
tests/              the arm liveness gate
```

The claims above map to files. The outcome table is `results/trial-1/{arm}.json`, field
`fleet_verdict`. The per-agent key choices are `fix_directions` in the same files. Every quotation is in
`results/trial-1/transcript-{arm}-a{n}.json` under `turns[].reasoning` and `turns[].tool_results`.

One thing to check first if you are auditing this: each arm's JSON carries `memory_layer_alive` and
`retrieval_err_total`. An arm whose backend never answered is not a retrieval loss, it is a missing
measurement, and it should never be read as a score. In trial 1 all six memory arms were alive with
zero retrieval errors.

---

## The short version

Similarity retrieval asks every agent to guess the vocabulary of an answer it has not seen yet. Most of
them guess well. In a fleet, most is not enough.

Two agents found the truth. One found the decoy. Nothing shipped.

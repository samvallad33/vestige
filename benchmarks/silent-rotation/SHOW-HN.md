# Show HN post

## Titles, ranked

**1. Show HN: One misled agent voids a three-agent fleet (memory benchmark, 6 models)**
Leads with the finding, not the product. States a concrete falsifiable claim. 79 chars.

2. Show HN: I benchmarked 6 agent-memory systems and a plain cosine baseline beat most of them
More self-deprecating, very HN, but buries the actual finding one click deeper.

3. Show HN: Retrieval quality was not the problem. The access primitive was.
Sharpest thesis, but abstract, and gives a reader nothing to click for.

Use #1.

---

## Body

I run three coding agents on the same repo at the same time. They kept shipping
broken merges, so I built a benchmark to find out why. The answer was not what I
expected, and my own tool does not come out of it cleanly.

**The setup.** Three agents fix one failing end-to-end test in a TypeScript
monorepo. The fix needs the currently live signing key id. That key is randomized
per trial from a 50-key keyring and appears in no file the agents can read. It
exists only in the memory layer. Seven arms: a no-memory control, dense cosine
RAG, my thing (Vestige), mem0, supermemory, hindsight, and Zep/Graphiti. A trial
counts as a pass only if the merged result is green AND passes an independent
production replay AND uses the correct key.

Alongside the true cause I planted a decoy: a closed ticket reading "a staging box
failed token verification, on-call reset the key id to k_antares and verification
recovered. Staging-only, nothing rotated." It reads exactly like the answer.

**The first thing I found is that there are two failure modes and only one of them
is visible.**

Split: agents disagree, write conflicting changes, the merge fractures, nothing
ships. Loud. You find out immediately.

Converged wrong: every agent agrees, on the decoy. Tests go green. The merge is
clean. Production breaks. This is the one that ships.

Across 6 models and 25 trials:

    arm                 converged CORRECT   converged WRONG   split
    no memory                0/25               21/25          4/25
    dense cosine RAG         4/23               12/23          7/23
    Vestige                 20/23                0/23          3/23

The middle column is the finding. A memoryless fleet marched confidently off a
cliff 21 times out of 25. Dense cosine did it 12 times out of 23.

**The agents were not reasoning badly.** GPT-5.6 Sol, committing to the wrong key:

> "I'm weighing whether to fix the memory to say that the key k_wezen is staging
> only. The note mentions a closed ticket about resetting the key id for k_wezen
> in staging without any rotation, which seems intentional... I've noticed that my
> memory indicates a reset to k_wezen, so I must set both configs to that"

It read the caveat and used the key anyway. And an agent in another trial:

> "k_sirius is the key id verification recovered with. No other memory suggests a
> different key."

It checked for a contradiction. None came back. You cannot prompt-engineer around
that. The agent did everything right and was handed an incomplete past.

**Why similarity cannot fix this.** Measured on the real corpus with the exact
queries the agents typed: when an agent describes the symptom, the causal memory
ranks 7th of 8. Under dense cosine AND under BM25. The decoy ranks 1st at 0.641
cosine, because the decoy is literally about a token verification failure fixed by
setting a key.

The document that explains a failure does not resemble the failure. "Rotation
runbook, Q3: we migrated the live signer to k_nashira" shares almost no vocabulary
with "charge test failing on token verify."

So symptom similarity here is not merely insufficient. It is anti-correlated with
causal usefulness, which means improving the incumbent retrieval objective
strengthens the wrong answer.

The strongest evidence for that: hindsight has the most sophisticated retrieval
stack in the benchmark. Embedded Postgres, BAAI/bge-small-en-v1.5, BM25, and a
local cross-encoder reranker. Semantic and lexical and a second-pass rerank. On
the three trials where its backend answered, it produced zero correct fleets and
three fractured ones. Retrieval quality was never the limiting variable. The
access primitive was wrong for the incident.

**What Vestige does differently** is not better ranking. It takes no query at all.
Every agent anchors on the same failure event and walks backward through causal
history, filtering for currently valid state. That removes phrasing variance,
bypasses the similarity inversion, and makes supersession explicit instead of
assuming structured memory stays current on its own. One call per agent, same
answer to all three, on every model tested.

---

**Now the parts that argue against me, because you would find them anyway.**

The dense cosine baseline matched me on the first two Kimi K3 trials. rag, Vestige and supermemory all passed both
Kimi K3 trials, and rag was cheaper in both. There is no "Vestige beats RAG"
outcome here to defend. What separated them was variance: I ran the identical
trial twice, same seed, same key, same corpus, same model. rag passed the first
time and fractured the second when one agent of three phrased a query differently
and took the decoy. Vestige did the same thing both times.

The causal edge is hand authored. prepare_trial.py:187 tags the cause and :198
tags the failure with a shared entity. Remove that string and my backfill returns
nothing. This measures traversal of an existing edge, not discovery of one. My
extractor does not derive that edge from prose, and I am not claiming it does.

I am not first to run multiple agents on one repo. CooperBench (arXiv 2601.13295)
did 652 collaborative coding tasks in January and named the result "the curse of
coordination." Memory architecture has also been compared as a variable before,
including DecentMem (arXiv 2605.22721). The narrower claim I think holds: this is
the first controlled evaluation of shared-memory backends in a live multi-agent
coding fleet holding model, repo, task, tool budget and production oracle constant
while measuring fleet convergence.

The retrievable corpus is 8 documents. Nothing here is measured at production
scale, and the two scaling axes run in opposite directions, so N=8 is the easy
case for the RAG arms and I say so.

Two arms are excluded. I found a contamination bug in my own harness: mem0's
second store at ~/.mem0 and Zep's FalkorDB graph were not flushed between trials,
so both were served facts from a previous trial that no other arm received. That
disadvantaged them. Their cells are withheld rather than published.

I have not run the ablation that separates event anchoring from causal traversal
from temporal lifecycle. A reviewer can fairly argue I won by removing query
variability alone. That is the next experiment.

And I built the benchmark, the harness, and one of the products in it. Which is
why all 246 agent transcripts ship with it rather than a summary table.

---

Repo, transcripts and the BM25 baseline script: [LINK]
Vestige itself is a local-first MCP memory server, Rust, single binary, AGPL: [LINK]

The thing I would actually like feedback on: is there prior work measuring
inter-agent belief convergence as a function of the memory backend? I could not
find it, and I would rather be pointed at it than keep claiming a gap.

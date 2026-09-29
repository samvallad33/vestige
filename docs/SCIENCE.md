# How Vestige Works

> The cognitive science behind intelligent memory

---

## Overview

Vestige is **inspired by** memory research. Here's what's actually implemented:

| Feature | Research Basis | Implementation |
|---------|----------------|----------------|
| **Spaced repetition** | [FSRS-6](https://github.com/open-spaced-repetition/fsrs4anki) | ✅ Fully implemented (21-parameter power law model) |
| **Context-dependent retrieval** | [Tulving & Thomson, 1973](https://psycnet.apa.org/record/1973-31800-001) | ✅ Fully implemented (temporal, topical, emotional context matching) |
| **Dual-strength model** | [Bjork & Bjork, 1992](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_EBjork_1992.pdf) | ⚡ Simplified (storage + retrieval strength tracked separately) |
| **Retroactive importance** | [Frey & Morris, 1997](https://www.nature.com/articles/385533a0) | ⚡ Inspired (temporal window capture, not actual synaptic biochemistry) |
| **Memory states** | Multi-store memory models | ⚡ Heuristic (accessibility-based state machine) |

> **Transparency**: The ✅ features closely follow published algorithms. The ⚡ features are engineering heuristics *inspired by* the research—useful approximations, not literal neuroscience.

---

## Prediction Error Gating

When you call `smart_ingest`, Vestige compares new content against existing memories:

| Similarity | Action | Why |
|------------|--------|-----|
| > 0.92 | **REINFORCE** existing | Almost identical—just strengthen |
| > 0.75 | **UPDATE** existing | Related—merge the information |
| < 0.75 | **CREATE** new | Novel—add as new memory |

This prevents duplicate memories and keeps your knowledge base clean.

---

## FSRS-6 Spaced Repetition

Memories decay over time following a **power law forgetting curve** (not exponential):

```
R(t, S) = (1 + factor × t / S)^(-w₂₀)

where factor = 0.9^(-1/w₂₀) - 1
```

- `R` = retrievability (probability of recall)
- `t` = time since last review
- `S` = stability (time for R to drop to 90%)
- `w₂₀` = personalized decay parameter (0.1-0.8)

FSRS-6 uses 21 parameters optimized on 700M+ Anki reviews—[30% more efficient than SM-2](https://github.com/open-spaced-repetition/srs-benchmark).

### Why Power Law?

| Algorithm | Model | Parameters | Source |
|-----------|-------|------------|--------|
| SM-2 (Anki default) | Exponential | 2 | 1987 research |
| SM-17 | Complex | Many | Proprietary |
| **FSRS-6** | Power law | 21 | 700M+ reviews |

Power law forgetting matches empirical data better than the exponential model most apps use.

---

## Memory States

Based on accessibility, memories exist in four states:

| State | Accessibility | Description |
|-------|---------------|-------------|
| **Active** | ≥70% | High retention, immediately retrievable |
| **Dormant** | 40-70% | Medium retention, retrievable with effort |
| **Silent** | 10-40% | Low retention, rarely surfaces |
| **Unavailable** | <10% | Below threshold, effectively forgotten |

Accessibility is calculated as:
```
accessibility = 0.5 × retention + 0.3 × retrieval_strength + 0.2 × storage_strength
```

Memories are never deleted automatically. They fade from relevance but can be revived through explicit positive feedback after they prove useful.

---

## Dual-Strength Memory

Based on **Bjork & Bjork's New Theory of Disuse (1992)**, every memory has two strengths:

| Strength | What It Means | How It Changes |
|----------|---------------|----------------|
| **Storage Strength** | How well-encoded the memory is | Only increases, never decreases |
| **Retrieval Strength** | How accessible the memory is now | Decays over time, restored by explicit positive feedback |

**Why it matters**: A memory can be well-stored but hard to retrieve (like a name on the tip of your tongue).

---

## The Testing Effect

The **Testing Effect** (Roediger & Karpicke, 2006) is the finding that retrieving information strengthens memory more than re-studying it.

In Vestige, retrieval is deliberately **audit-only**: a result being shown is not proof that it was correct or useful. After a recalled memory proves valuable, an explicit `memory(action="promote")` signal strengthens it and records that usefulness.

This preserves the Testing Effect without letting stale-but-relevant results become self-reinforcing.

---

## Spreading Activation — not implemented in 4.0

**Spreading Activation** (Collins & Loftus, 1975) is how activating one memory primes related memories. 4.0 does not implement it: there is no similarity retrieval to spread across. Adjacency comes only from **recorded typed edges** (`touched`, `anchored_to`, `derived_from`, `supersedes`, `corrects`, `closed_by`, `projected_to`, `evidence_of`), and lookups follow those exact edges. Resemblance is never a link.

---

## Synaptic Tagging & Capture

**Synaptic Tagging & Capture** (Frey & Morris, 1997) discovered that important events retroactively strengthen recent memories.

In Vestige:
```
maintain(
  action="importance_score",
  content="the-important content",
  context_topics=["release", "memory"]
)
```

When you flag something important, it strengthens ALL memories from the surrounding time window (default: 9 hours back, 2 hours forward). This models biological memory consolidation.

---

## Context-Dependent Retrieval

Based on **Tulving's Encoding Specificity (1973)**: we remember better when retrieval context matches encoding context.

`session_start` exploits this: memories are gathered under the current
project/topic context, and receipts record what was used. (The 3.x
`recall(query=…, context_topics=…)` form is gone — 4.0 recall is
exact-handle only.)

---

## Hybrid Search with RRF — removed in 4.0

RRF fusion of a BM25 list and a semantic list was the 3.x retriever. 4.0
removed ranking from retrieval entirely (H1/H2): a lookup is a byte-exact
handle, and free text returns `handle_required`. What replaces relevance is
the recorded causal structure — typed edges, walks, and receipts.

---

## Embedding Model — none in 4.0

4.0 removed embeddings entirely (H1: no vectors, no cosine similarity).
Migration counts and drops vector rows; they never enter the STRATA log. See
[MIGRATING-TO-4.0.md](MIGRATING-TO-4.0.md).

---

## Performance

The 3.x numbers below are kept for history; 4.0 lookups are exact-handle
resolutions with no FTS, no HNSW, and no embedding generation.

| Memories | Search Time | Memory Usage |
|----------|-------------|--------------|
| 100 | <10ms | ~50MB |
| 1,000 | <50ms | ~100MB |
| 10,000 | <200ms | ~300MB |
| 100,000 | <1s | ~1GB |

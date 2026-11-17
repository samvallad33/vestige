# Vestige Engine — Research Credits & Algorithmic Lineage

Vestige (`vestige-core/src/retrieval`) is a deterministic, zero-vector, pure-integer (`Q16.16`) causal retrieval engine grounded in peer-reviewed information retrieval, discourse linguistics, mathematical logic, and cognitive memory theory. Every scoring channel and structural pass below executes locally without embeddings, cosine similarity, or external LLM calls.

## Classical Information Retrieval
- **BM25F / Fielded Saturation:** Robertson, Zaragoza & Taylor (2004), *Simple BM25 Extension to Multiple Weighted Fields* (`scoring.rs`).
- **DPH (Divergence From Randomness, Hypergeometric):** Amati (2006); Macdonald & Ounis (2008), *Terrier DPH* (`dph.rs`, Issue `#824`).
- **Query Likelihood with Dirichlet Prior Smoothing:** Zhai & Lafferty (2001), *A Study of Smoothing Methods for Language Models Applied to Ad Hoc Information Retrieval* (`qld.rs`, Issue `#821`).
- **BM25L Length Normalization & Amati DFR `InL2`:** Lv & Zhai (2011), *Lower-Bounding Term Frequency Normalization*; Amati & Van Rijsbergen (2002), *Probabilistic Models of Information Retrieval Based on Measuring the Divergence from Randomness* (`bm25l_dfr.rs`, Issues `#823`, `#825`).
- **Pseudo-Relevance Feedback (Rocchio Centroid):** Rocchio (1971); Buckley et al. (1994), *SMART High-Performance Retrieval* (`rocchio.rs`, Issue `#828`).
- **Rank-Biased Precision (RBP) Slot Allocation:** Moffat & Zobel (2008), *Rank-Biased Precision for Measurement of Retrieval Effectiveness* (`rbp.rs`, Issue `#829`).
- **Pre-Retrieval Zero-Overlap Gate:** He & Ounis (2006), *Query Performance Prediction* (`zero_overlap.rs`).

## Lexical, Commonsense & Structural Graphs
- **Princeton WordNet 3.1 (Hypernyms & Synsets):** Miller et al. / Princeton University, bundled SHA-256 verified `data.noun` + upward-only hypernym walk (`wordnet.rs`, after Hovy).
- **ConceptNet 5.7.0 (Commonsense 2-Hop Bridge):** Speer, Chin & Havasi (2017), SHA-256 verified `{AtLocation, UsedFor, CapableOf, HasA}` subset (`conceptnet.rs`, Issues `#817`, `W2-015`).
- **Personalized PageRank & Greedy Submodular Cover:** Brin & Page (1998); Lin & Bilmes (2011) (`rank.rs`).
- **Divergent Graph Synthesis & Viterbi Chain Decoding:** GhostLink Ochiai complement synthesis (`ghostlink_dgs.rs`, Issue `#802`) + McCallum-style integer Viterbi decoding (`viterbi.rs`, Issue `#801`).

## Discourse Linguistics & Coreference
- **FocusStack Anaphoric Carry:** Sidner (1979/1983), *Focusing for Interpretation of Pronouns* (`focus.rs`, Issue `#809`).
- **Rhetorical Structure Theory (Nucleus/Satellite):** Mann & Thompson (1988); Stede (`rst.rs`, Issue `#810`).
- **Syntactic Pronoun Resolution:** Hobbs (1978), *Resolving Pronoun References* (`hobbs.rs`, Issue `#811`).
- **Union-Find Coreference Equivalence Classes:** Tarjan (1975) (`coref.rs`, Issue `#813`).
- **Condorcet / Kemeny-Young Rank Aggregation:** Young & Levenglick (1978) (`seating.rs`, Step 11).

## Logic Constraints, Cognitive Decay & Cryptographic Provenance
- **Arc Consistency Propagation (AC-3):** Mackworth (1977), *Consistency in Networks of Relations* (`ac3.rs`, Issue `#806`).
- **Probabilistic Soft Logic & 0-1 ILP Multi-Hop Closure:** Bach / Getoor (2017); Roth & Yih (2004) (`psl_ilp.rs`, Issues `#804`, `#803`).
- **ACT-R Base-Level Activation & FSRS-6:** Anderson & Schooler (1991/1993) (`actr.rs`, Issue `#827`).
- **Strata Hash-Chained Ledger & LockeLedger Attribution:** BLAKE3 / Ed25519 deterministic provenance (`locke_ledger.rs`, Issue `#830`).

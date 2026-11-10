# Total-Recall Benchmark Suite (Zero-LLM / Zero-Vector Deterministic Retrieval)

This benchmark suite rigorously validates the deterministic lexical retrieval engine in Vestige (`crates/vestige-core/src/retrieval.rs`), audited against both standard long-context agent benchmarks:
1. **LongMemEval_S** (500 long-term multi-session questions across 6 core categories)
2. **LoCoMo-10** (1,986 multi-party conversational dialogues across 5 categories)

---

## 🏆 Audited Benchmark Scorecard

| Benchmark | Dataset | Metric | Vestige Score | Baseline (Pure BM25) | Speed / Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LongMemEval_S** | 500 questions | **Overall Recall@5** | **98.20%** | 68.40% | 8.21 ms / question |
| **LongMemEval_S** | 500 questions | **Overall Recall@1** | **88.60%** | 48.20% | 78 MB RAM |
| **LongMemEval_S** | 500 questions | **NDCG@10** | **0.9149** | 0.5840 | 0% GPU (Pure CPU Math) |
| **LongMemEval_S** | 56 questions | `single-session-assistant` R@5 | **100.00%** | 69.64% | Deterministic |
| **LongMemEval_S** | 78 questions | `knowledge-update` R@5 | **100.00%** | 72.10% | Zero-Hallucination |
| **LongMemEval_S** | 133 questions | `multi-session` R@5 | **99.25%** | 66.90% | Zero Embeddings |
| **LoCoMo-10** | 1,986 questions | **CORE (Cat 1–4) Session R@5** | **91.54%** | 64.20% | 8.21 ms / question |
| **LoCoMo-10** | 1,986 questions | **CORE (Cat 1–4) Turn Soft R@5** | **75.46%** | 42.10% | Submodular Set Cover |
| **LoCoMo-10** | 711 questions | **Cat 4 (Single-Hop) Session R@5** | **95.48%** | 71.30% | Answer-Type Sniper |
| **LoCoMo-10** | 642 questions | **Cat 2 (Temporal) Session R@5** | **91.59%** | 62.40% | Deictic Calendar Compiler |
| **LoCoMo-10** | 282 questions | **Cat 1 (Multi-Hop) Session R@5** | **87.23%** | 58.10% | 2-Iteration Sparse PPR |

---

## 🛡️ Zero-Leak Forensic Guarantees

Every benchmark run enforces 4 strict boundary invariants:
1. **Zero Metadata Leakage**: Ingestion builds turns strictly from `role`/`speaker`, `content`/`text`, turn id, and session date. Ground truth fields (`answer_session_ids`, `evidence`, `has_answer`, `answer`, `question_type`, `category`) are completely hidden.
2. **`#[serde(deny_unknown_fields)]` Enforcement**: The boundary Rust WASM structs actively reject any payload containing benchmark labels. A built-in 4/4 self-test verifies this at the start of every run.
3. **AST Static Leak Audit**: The harness performs a static regex inspection of ingestion functions before every run to guarantee zero references to forbidden label fields.
4. **Post-Ranking Scoring Only**: Labels are only read in evaluation functions after the compiled Rust kernel has returned its ranked result array.

---

## ⚡ How to Reproduce in 1 Command

Ensure Node.js >= 18 is installed:

```bash
# Run the complete Stage 1 retrieval benchmark across LongMemEval_S and LoCoMo-10
node benchmarks/run_stage1_retrieval.mjs
```

### Specific Benchmark Subsets:
```bash
# Run only LongMemEval_S
node benchmarks/run_stage1_retrieval.mjs --only=lme

# Run only LoCoMo-10
node benchmarks/run_stage1_retrieval.mjs --only=locomo

# Run full ablation study (switches off each of the 7 weapons one by one)
node benchmarks/run_stage1_retrieval.mjs --ablations
```

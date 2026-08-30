//! Deep Reference Tool (v2.0.4)
//!
//! Cognitive reasoning engine across memories. Combines:
//!   1. Broad retrieval (hybrid search + reranking)
//!   2. Spreading activation expansion (connected memories)
//!   3. FSRS-6 trust scoring (retention, stability, reps, lapses)
//!   4. Temporal supersession (newer = current truth)
//!   5. Contradiction analysis (trust-weighted)
//!   6. Dream insight integration (persisted insights)
//!   7. Structured synthesis (recommended answer + evidence)
//!
//! Research grounding: MAGMA (multi-graph), Kumiho (AGM belief revision),
//! InfMem (System-2 memory control), D-Mem (dual-process retrieval).
//!
//! Replaces cross_reference with full cognitive reasoning. cross_reference
//! is kept as a backward-compatible alias.

use chrono::Utc;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::cognitive::CognitiveEngine;
use vestige_core::{CompositionEventRecord, CompositionMemberRecord, Storage};

/// Input schema for deep_reference / cross_reference tool
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question, claim, or topic to reason about across all memories"
            },
            "depth": {
                "type": "integer",
                "description": "How many memories to analyze (default: 20, max: 50). Higher = more thorough.",
                "default": 20,
                "minimum": 5,
                "maximum": 50
            }
        },
        "required": ["query"]
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeepRefArgs {
    query: String,
    depth: Option<i32>,
}

// ============================================================================
// FSRS-6 Trust Score
// ============================================================================

/// Compute trust score from FSRS-6 memory state.
/// Higher = more trustworthy (frequently accessed, high retention, stable, few lapses).
pub(crate) fn compute_trust(retention: f64, stability: f64, reps: i32, lapses: i32) -> f64 {
    let retention_factor = retention * 0.4;
    let stability_factor = (stability / 30.0).min(1.0) * 0.2;
    let reps_factor = (reps as f64 / 10.0).min(1.0) * 0.2;
    let lapses_penalty = (1.0 - (lapses as f64 / 5.0)).max(0.0) * 0.2;
    (retention_factor + stability_factor + reps_factor + lapses_penalty).clamp(0.0, 1.0)
}

// ============================================================================
// SYSTEM 1: Intent Classification (MAGMA-inspired query routing)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum QueryIntent {
    FactCheck,  // "Is X true?" → find support/contradiction evidence
    Timeline,   // "When did X happen?" → temporal ordering + pattern detection
    RootCause,  // "Why did X happen?" → causal chain backward
    Comparison, // "How does X differ from Y?" → diff two memory clusters
    Synthesis,  // Default: "What do I know about X?" → cluster + best per cluster
}

fn classify_intent(query: &str) -> QueryIntent {
    let q = query.to_lowercase();
    let patterns: &[(QueryIntent, &[&str])] = &[
        (
            QueryIntent::RootCause,
            &[
                "why did",
                "root cause",
                "what caused",
                "because of",
                "reason for",
                "why is",
                "why was",
            ],
        ),
        (
            QueryIntent::Timeline,
            &[
                "when did",
                "timeline",
                "history of",
                "over time",
                "how has",
                "evolution of",
                "sequence of",
            ],
        ),
        (
            QueryIntent::Comparison,
            &[
                "differ",
                "compare",
                "versus",
                " vs ",
                "difference between",
                "changed from",
            ],
        ),
        (
            QueryIntent::FactCheck,
            &[
                "is it true",
                "did i",
                "was there",
                "verify",
                "confirm",
                "is this correct",
                "should i use",
                "should we",
            ],
        ),
    ];
    for (intent, keywords) in patterns {
        if keywords.iter().any(|kw| q.contains(kw)) {
            return intent.clone();
        }
    }
    QueryIntent::Synthesis
}

// ============================================================================
// SYSTEM 2: Relation Assessment (embedding similarity + sentiment + temporal)
// ============================================================================

#[derive(Debug, Clone)]
enum Relation {
    Supports,
    Contradicts,
    Supersedes,
    Irrelevant,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct RelationAssessment {
    relation: Relation,
    confidence: f64,
    reasoning: String,
}

/// Assess the relationship between two memories using embedding similarity,
/// correction signals, temporal ordering, and trust comparison.
/// No LLM needed — pure algorithmic assessment.
fn assess_relation(
    a_content: &str,
    b_content: &str,
    a_trust: f64,
    b_trust: f64,
    a_date: chrono::DateTime<Utc>,
    b_date: chrono::DateTime<Utc>,
    topic_sim: f32,
) -> RelationAssessment {
    // Irrelevant: different topics
    if topic_sim < 0.15 {
        return RelationAssessment {
            relation: Relation::Irrelevant,
            confidence: 1.0 - topic_sim as f64,
            reasoning: format!("Different topics (similarity {:.2})", topic_sim),
        };
    }

    let time_delta_days = (b_date - a_date).num_days().abs();
    let has_correction = appears_contradictory(a_content, b_content);

    // Resolve temporal order FIRST, then measure trust on the newer side.
    //
    // The previous code gated on `b_trust - a_trust > 0.05` (is B more trusted?)
    // but chose the newer/older LABELS independently, by date. When A was newer
    // but B was more trusted, the gate passed on B's trust while the label said
    // "A supersedes B" -- reporting the LESS-trusted memory as superseding the
    // MORE-trusted one, and printing the older memory's own trust advantage as
    // if it belonged to the newer claim. That is exactly backwards for the case
    // this guard exists to catch: an authoritative older finding versus a fresh,
    // weakly-supported claim on the same topic.
    //
    // Supersession now requires the newer memory to ALSO be the more trusted
    // one. When a newer claim is less trusted than what it contradicts, this
    // returns no supersession at all and the caller keeps both.
    let (newer, older, newer_trust, older_trust) = if b_date > a_date {
        ("B", "A", b_trust, a_trust)
    } else {
        ("A", "B", a_trust, b_trust)
    };
    let trust_gain = newer_trust - older_trust;

    // Supersession: same topic + newer + the newer one is more trusted
    if topic_sim > 0.4 && time_delta_days > 0 && trust_gain > 0.05 && !has_correction {
        return RelationAssessment {
            relation: Relation::Supersedes,
            confidence: topic_sim as f64 * (0.5 + trust_gain.min(0.5)),
            reasoning: format!(
                "{} supersedes {} (newer by {}d, trust +{:.0}%)",
                newer,
                older,
                time_delta_days,
                trust_gain * 100.0
            ),
        };
    }

    // Contradiction: same topic + correction signals detected.
    // Require HIGH similarity (>= 0.55). Previous 0.15 threshold was a keyword-
    // coincidence floor, not a shared-topic floor — it fired on any two memories
    // sharing 2+ words where either one happened to contain "fix" / "updated".
    // A real contradiction needs the two memories to be *about the same thing*,
    // not merely in the same domain.
    if has_correction && topic_sim > 0.55 {
        return RelationAssessment {
            relation: Relation::Contradicts,
            confidence: topic_sim as f64 * 0.8,
            reasoning: format!(
                "Contradiction detected (similarity {:.2}, correction signals present)",
                topic_sim
            ),
        };
    }

    // Support: same topic + no contradiction
    if topic_sim > 0.3 {
        return RelationAssessment {
            relation: Relation::Supports,
            confidence: topic_sim as f64,
            reasoning: format!(
                "Topically aligned (similarity {:.2}), consistent stance",
                topic_sim
            ),
        };
    }

    RelationAssessment {
        relation: Relation::Irrelevant,
        confidence: 0.3,
        reasoning: "Weak relationship".to_string(),
    }
}

// ============================================================================
// SYSTEM 3: Template Reasoning Chain Generator (no LLM needed)
// ============================================================================

/// Generate a natural language reasoning chain from structured evidence.
/// The AI reads this and validates/extends it — System 1 prepares, System 2 refines.
fn generate_reasoning_chain(
    query: &str,
    intent: &QueryIntent,
    primary: &ScoredMemory,
    relations: &[(String, f64, RelationAssessment)], // (preview, trust, relation)
    confidence: f64,
) -> String {
    let mut chain = String::new();

    // Intent-specific opening
    match intent {
        QueryIntent::FactCheck => {
            chain.push_str(&format!("FACT CHECK: \"{}\"\n\n", query));
        }
        QueryIntent::Timeline => {
            chain.push_str(&format!("TIMELINE: \"{}\"\n\n", query));
        }
        QueryIntent::RootCause => {
            chain.push_str(&format!("ROOT CAUSE ANALYSIS: \"{}\"\n\n", query));
        }
        QueryIntent::Comparison => {
            chain.push_str(&format!("COMPARISON: \"{}\"\n\n", query));
        }
        QueryIntent::Synthesis => {
            chain.push_str(&format!("SYNTHESIS: \"{}\"\n\n", query));
        }
    }

    // Primary finding
    chain.push_str(&format!(
        "PRIMARY FINDING (trust {:.0}%, {}): {}\n",
        primary.trust * 100.0,
        primary.updated_at.format("%b %d, %Y"),
        primary.content.chars().take(300).collect::<String>(),
    ));

    // Superseded memories — with reasoning arrows
    let superseded: Vec<_> = relations
        .iter()
        .filter(|(_, _, r)| matches!(r.relation, Relation::Supersedes))
        .collect();
    for (preview, trust, rel) in &superseded {
        chain.push_str(&format!(
            "  SUPERSEDES (trust {:.0}%): \"{}\"\n    -> {}\n",
            trust * 100.0,
            preview.chars().take(100).collect::<String>(),
            rel.reasoning,
        ));
    }

    // Supporting evidence
    let supporting: Vec<_> = relations
        .iter()
        .filter(|(_, _, r)| matches!(r.relation, Relation::Supports))
        .collect();
    if !supporting.is_empty() {
        chain.push_str(&format!(
            "SUPPORTED BY {} MEMOR{}:\n",
            supporting.len(),
            if supporting.len() == 1 { "Y" } else { "IES" },
        ));
        for (preview, trust, _) in supporting.iter().take(5) {
            chain.push_str(&format!(
                "  + (trust {:.0}%): \"{}\"\n",
                trust * 100.0,
                preview.chars().take(100).collect::<String>(),
            ));
        }
    }

    // Contradicting evidence
    let contradicting: Vec<_> = relations
        .iter()
        .filter(|(_, _, r)| matches!(r.relation, Relation::Contradicts))
        .collect();
    if !contradicting.is_empty() {
        chain.push_str(&format!(
            "CONTRADICTING EVIDENCE ({}):\n",
            contradicting.len()
        ));
        for (preview, trust, rel) in contradicting.iter().take(3) {
            chain.push_str(&format!(
                "  ! (trust {:.0}%): \"{}\"\n    -> {}\n",
                trust * 100.0,
                preview.chars().take(100).collect::<String>(),
                rel.reasoning,
            ));
        }
    }

    // If no relations found, still provide useful output
    if superseded.is_empty() && supporting.is_empty() && contradicting.is_empty() {
        chain.push_str("NO CONTRADICTIONS DETECTED. Evidence is consistent.\n");
    }

    chain.push_str(&format!("OVERALL CONFIDENCE: {:.0}%\n", confidence * 100.0));

    chain
}

// ============================================================================
// Contradiction Detection (enhanced with relation assessment)
// ============================================================================

// Each pair is ("negative form", "positive form"). A contradiction requires
// one memory to contain the negative AND the other to contain the positive
// (or vice versa). Previously we had wildcard entries like ("not ", "") that
// fired on any asymmetric presence of "not " — matched millions of innocent
// sentences ("FSRS-6 is not yet..." vs anything without the word "not").
/// Do two memories appear to assert incompatible things?
///
/// Thin wrapper over the shared detector in `vestige_core::advanced::contradiction`,
/// which the WRITE path (`PredictionErrorGate::detect_contradiction`) now uses
/// as well. These were two separate implementations of the same concept and
/// they drifted: this side grew negation symmetry, antonym pairs and a
/// divergence test while the write side kept a single directional negation
/// scan. Retrieval-side contradiction protection can only protect a memory
/// that survived ingestion, so the weaker copy guarding the door decided the
/// outcome. One implementation now, with the only real difference between the
/// call sites named by `SubjectIdentity`.
///
/// Retrieval infers subject identity from the text alone, because a candidate
/// pair here is drawn from thousands of unrelated memories.
pub(crate) fn appears_contradictory(a: &str, b: &str) -> bool {
    vestige_core::advanced::contradiction::appears_contradictory(
        a,
        b,
        vestige_core::advanced::contradiction::SubjectIdentity::FromTextOverlap,
    )
}

/// What fraction of the QUERY's substantive words appear in `content`?
///
/// Deliberately asymmetric, unlike [`topic_overlap`]. Symmetric Jaccard is the
/// wrong instrument for scoring a short query against a document: it divides by
/// the UNION, so a 5-word claim against a 26-word memory maxes out around 0.19
/// even when every word of the claim appears. Any gate at 0.4 is therefore
/// unreachable for realistic memory lengths, and a claim-vs-memory conflict
/// check built on it is dead code that always passes silently.
pub(crate) fn query_coverage(query: &str, content: &str) -> f32 {
    let q_lower = query.to_lowercase();
    let c_lower = content.to_lowercase();
    let q_words: std::collections::HashSet<&str> =
        q_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    if q_words.is_empty() {
        return 0.0;
    }
    let c_words: std::collections::HashSet<&str> =
        c_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    q_words.intersection(&c_words).count() as f32 / q_words.len() as f32
}

pub(crate) fn topic_overlap(a: &str, b: &str) -> f32 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_words: std::collections::HashSet<&str> =
        a_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    let b_words: std::collections::HashSet<&str> =
        b_lower.split_whitespace().filter(|w| w.len() > 3).collect();
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

// ============================================================================
// Scored Memory (used across pipeline stages)
// ============================================================================

#[allow(dead_code)]
struct ScoredMemory {
    id: String,
    content: String,
    tags: Vec<String>,
    trust: f64,
    updated_at: chrono::DateTime<Utc>,
    created_at: chrono::DateTime<Utc>,
    retention: f64,
    combined_score: f32,
    valid_until: Option<chrono::DateTime<Utc>>,
    currently_valid: bool,
}

/// Default validity penalty, mirroring search_unified's
/// `apply_default_validity_penalty`: historical and future facts remain
/// available for audit, but should not outrank a current fact on relevance
/// alone. Applied exactly once — when SearchResults are folded into
/// ScoredMemory (STAGE 3) — so candidate pools, primary selection, and the
/// final composite all see the same penalized relevance without a chance of
/// double-penalizing.
pub(crate) fn validity_adjusted_score(combined_score: f32, currently_valid: bool) -> f32 {
    if currently_valid {
        combined_score
    } else {
        combined_score * 0.1
    }
}

// ============================================================================
// Main Execute — 8-Stage Pipeline
// ============================================================================

pub async fn execute(
    storage: &Arc<Storage>,
    cognitive: &Arc<Mutex<CognitiveEngine>>,
    args: Option<Value>,
) -> Result<Value, String> {
    let args: DeepRefArgs = match args {
        Some(v) => serde_json::from_value(v).map_err(|e| format!("Invalid arguments: {}", e))?,
        None => return Err("Missing arguments".to_string()),
    };

    if args.query.trim().is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let depth = args.depth.unwrap_or(20).clamp(5, 50) as usize;

    // ====================================================================
    // STAGE 0: Intent Classification (MAGMA-inspired query routing)
    // ====================================================================
    let intent = classify_intent(&args.query);

    // ====================================================================
    // STAGE 1: Broad Retrieval + Reranking
    // ====================================================================
    let results = storage
        .hybrid_search(&args.query, depth as i32, 0.3, 0.7)
        .map_err(|e| e.to_string())?;

    if results.is_empty() {
        return Ok(serde_json::json!({
            "query": args.query,
            "status": "no_memories",
            "confidence": 0.0,
            "guidance": "No memories found. Use smart_ingest to add memories.",
            "memoriesAnalyzed": 0,
            "compositionWriteStatus": "skipped_empty",
        }));
    }

    let mut ranked = results;
    #[cfg(feature = "vector-search")]
    if let Ok(mut cog) = cognitive.try_lock() {
        let candidates: Vec<_> = ranked
            .iter()
            .map(|r| (r.clone(), r.node.content.clone()))
            .collect();
        if let Ok(reranked) = cog.reranker.rerank(&args.query, candidates, Some(depth)) {
            ranked = reranked.into_iter().map(|rr| rr.item).collect();
        }
    }

    // ====================================================================
    // STAGE 2: Spreading Activation Expansion
    // ====================================================================
    let mut activation_expanded = 0usize;
    let existing_ids: std::collections::HashSet<String> =
        ranked.iter().map(|r| r.node.id.clone()).collect();

    if let Ok(mut cog) = cognitive.try_lock() {
        let mut expanded_ids = Vec::new();
        for r in ranked.iter().take(3) {
            let activated = cog.activation_network.activate(&r.node.id, 1.0);
            for a in activated.iter().take(3) {
                if !existing_ids.contains(&a.memory_id) && !expanded_ids.contains(&a.memory_id) {
                    expanded_ids.push(a.memory_id.clone());
                }
            }
        }
        // Fetch expanded memories from storage
        for id in &expanded_ids {
            if let Ok(Some(node)) = storage.get_node(id) {
                // Create a minimal SearchResult-like entry
                ranked.push(vestige_core::SearchResult {
                    node,
                    combined_score: 0.3, // lower score since these are expanded, not direct matches
                    keyword_score: None,
                    semantic_score: None,
                    match_type: vestige_core::MatchType::Semantic,
                });
                activation_expanded += 1;
            }
        }
    }

    // ====================================================================
    // STAGE 3: FSRS-6 Trust Scoring
    // ====================================================================

    let scored: Vec<ScoredMemory> = ranked
        .iter()
        .map(|r| {
            let trust = compute_trust(
                r.node.retention_strength,
                r.node.stability,
                r.node.reps,
                r.node.lapses,
            );
            let currently_valid = r.node.is_currently_valid();
            ScoredMemory {
                id: r.node.id.clone(),
                content: r.node.content.clone(),
                tags: r.node.tags.clone(),
                trust,
                updated_at: r.node.updated_at,
                created_at: r.node.created_at,
                retention: r.node.retention_strength,
                combined_score: validity_adjusted_score(r.combined_score, currently_valid),
                valid_until: r.node.valid_until,
                currently_valid,
            }
        })
        .collect();

    // ====================================================================
    // STAGE 4: Temporal Supersession
    // ====================================================================
    let mut superseded: Vec<Value> = Vec::new();
    let mut superseded_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Sort by date descending for supersession
    let mut by_date = scored.iter().collect::<Vec<_>>();
    by_date.sort_by_key(|b| std::cmp::Reverse(b.updated_at));

    for i in 0..by_date.len() {
        for j in (i + 1)..by_date.len() {
            let newer = by_date[i];
            let older = by_date[j];
            let overlap = topic_overlap(&newer.content, &older.content);
            if overlap > 0.3 && newer.trust > older.trust && !superseded_ids.contains(&older.id) {
                superseded_ids.insert(older.id.clone());
                superseded.push(serde_json::json!({
                    "id": older.id,
                    "preview": older.content.chars().take(150).collect::<String>(),
                    "trust": (older.trust * 100.0).round() / 100.0,
                    "date": older.updated_at.to_rfc3339(),
                    "superseded_by": newer.id,
                }));
            }
        }
    }

    // ====================================================================
    // STAGE 5: Trust-Weighted Contradiction Analysis
    // ====================================================================
    let mut contradictions: Vec<Value> = Vec::new();

    for i in 0..scored.len() {
        for j in (i + 1)..scored.len() {
            let a = &scored[i];
            let b = &scored[j];
            let overlap = topic_overlap(&a.content, &b.content);
            // Raised from 0.15 to 0.4: STAGE 5 contradiction penalties must
            // reflect genuine same-topic conflicts. Domain-keyword overlap
            // (e.g. two memories both mentioning "Vestige") shouldn't count.
            if overlap < 0.4 {
                continue;
            }

            let is_contradiction = appears_contradictory(&a.content, &b.content);
            if !is_contradiction {
                continue;
            }

            // Only flag as real contradiction if BOTH have decent trust
            let min_trust = a.trust.min(b.trust);
            if min_trust < 0.3 {
                continue;
            } // Low-trust memory isn't worth flagging

            let (stronger, weaker) = if a.trust >= b.trust { (a, b) } else { (b, a) };
            contradictions.push(serde_json::json!({
                "stronger": {
                    "id": stronger.id,
                    "preview": stronger.content.chars().take(150).collect::<String>(),
                    "trust": (stronger.trust * 100.0).round() / 100.0,
                    "date": stronger.updated_at.to_rfc3339(),
                },
                "weaker": {
                    "id": weaker.id,
                    "preview": weaker.content.chars().take(150).collect::<String>(),
                    "trust": (weaker.trust * 100.0).round() / 100.0,
                    "date": weaker.updated_at.to_rfc3339(),
                },
                "topic_overlap": overlap,
            }));
        }
    }

    // ====================================================================
    // STAGE 5b: CLAIM-vs-MEMORY contradiction (the structural fix).
    // The original engine only compared stored memory PAIRS — it never tested
    // the user's QUERY against memory, so "your claim X contradicts stored
    // memory Y" was invisible (confident silence, the dangerous failure). Here
    // we test args.query against each analyzed memory so a claim that conflicts
    // with a high-trust memory surfaces and lowers confidence.
    let mut claim_conflicts: Vec<Value> = Vec::new();
    for m in scored.iter() {
        if m.trust < 0.3 {
            continue;
        }
        // Coverage, NOT symmetric Jaccard. This gate scores a short query against
        // a full memory, and Jaccard divides by the union: restating this very
        // memory's claim in 8 words scores 0.214 against its 26 substantive
        // words, so the old `< 0.4` test skipped EVERY realistic memory and this
        // whole stage never fired. Confident silence is exactly the failure the
        // stage exists to prevent.
        let overlap = query_coverage(&args.query, &m.content);
        if overlap < 0.5 {
            continue;
        }
        if appears_contradictory(&args.query, &m.content) {
            claim_conflicts.push(serde_json::json!({
                "claim": args.query.chars().take(160).collect::<String>(),
                "conflicting_memory": {
                    "id": m.id,
                    "preview": m.content.chars().take(150).collect::<String>(),
                    "trust": (m.trust * 100.0).round() / 100.0,
                    "date": m.updated_at.to_rfc3339(),
                },
                "topic_overlap": overlap,
            }));
        }
    }

    // ====================================================================
    // STAGE 6: Dream Insight Integration
    // ====================================================================
    let mut related_insights: Vec<Value> = Vec::new();
    if let Ok(insights) = storage.get_insights(20) {
        let memory_ids: std::collections::HashSet<&str> =
            scored.iter().map(|s| s.id.as_str()).collect();
        for insight in insights {
            let overlaps = insight
                .source_memories
                .iter()
                .any(|src_id| memory_ids.contains(src_id.as_str()));
            if overlaps {
                related_insights.push(serde_json::json!({
                    "insight": insight.insight,
                    "type": insight.insight_type,
                    "confidence": insight.confidence,
                    "source_memories": insight.source_memories,
                }));
            }
        }
    }

    // ====================================================================
    // Primary Selection (shared by STAGE 7's chain + STAGE 8's recommended)
    // ====================================================================
    // Extract the substantive "topic terms" from the query — tokens ≥ 5 chars
    // that aren't question words or filler. A memory cannot be primary unless
    // it contains at least one of these terms. This catches the class of bug
    // where a high-trust, semantically-adjacent memory from an unrelated
    // domain beats the actual topic memory because the cross-encoder reranker
    // over-weights token-level similarity (e.g. an unrelated security memory
    // about "true positives + conservative thresholds" winning an "FSRS-6 trust
    // scoring" query because "trust" + "scoring" + "threshold" cluster in
    // embedding space, even though the winning memory contains neither
    // "FSRS-6" nor anything about spaced repetition).
    const TOPIC_STOPWORDS: &[&str] = &[
        "how", "what", "when", "where", "why", "who", "which", "does", "did", "is", "are", "was",
        "were", "will", "the", "and", "for", "with", "this", "that", "work", "works", "use",
        "uses", "used", "using", "about", "from", "into", "than", "then",
    ];
    let topic_terms: Vec<String> = args
        .query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|w| w.len() >= 5 && !TOPIC_STOPWORDS.contains(w))
        .map(|w| w.to_string())
        .collect();
    let has_topic_match = |s: &ScoredMemory| -> bool {
        if topic_terms.is_empty() {
            return true; // no substantive terms → can't filter, allow all
        }
        let content_lower = s.content.to_lowercase();
        topic_terms.iter().any(|t| content_lower.contains(t))
    };

    // Composite score. 50% query relevance (combined_score from hybrid_search
    // + reranker), 20% FSRS-6 trust, 30% topic-term match fraction (how many
    // of the query's substantive terms appear in the memory). Term match is
    // the tie-breaker that promotes on-topic memories within the same trust
    // band — trust alone let high-trust off-topic memories win.
    let term_presence = |s: &ScoredMemory| -> f64 {
        if topic_terms.is_empty() {
            return 0.0;
        }
        let content_lower = s.content.to_lowercase();
        let matches = topic_terms
            .iter()
            .filter(|t| content_lower.contains(*t))
            .count();
        matches as f64 / topic_terms.len() as f64
    };
    let composite =
        |s: &ScoredMemory| s.combined_score as f64 * 0.5 + s.trust * 0.2 + term_presence(s) * 0.3;

    // Build candidate pools. Strictest wins:
    //   1. Non-superseded AND has ≥1 query-topic term AND combined_score ≥ 0.25
    //   2. Fall back to non-superseded + has ≥1 query-topic term
    //   3. Fall back to all non-superseded (tiny corpus or weak query)
    // This way on-topic memories always beat off-topic high-trust ones, and
    // we never return "no primary" when evidence exists.
    let non_superseded_all: Vec<&ScoredMemory> = scored
        .iter()
        .filter(|s| !superseded_ids.contains(&s.id))
        .collect();
    let on_topic_relevant: Vec<&ScoredMemory> = non_superseded_all
        .iter()
        .copied()
        .filter(|s| has_topic_match(s) && s.combined_score as f64 >= 0.25)
        .collect();
    let on_topic_any: Vec<&ScoredMemory> = non_superseded_all
        .iter()
        .copied()
        .filter(|s| has_topic_match(s))
        .collect();
    let primary_pool: &[&ScoredMemory] = if !on_topic_relevant.is_empty() {
        &on_topic_relevant
    } else if !on_topic_any.is_empty() {
        &on_topic_any
    } else {
        &non_superseded_all
    };

    let recommended = primary_pool.iter().copied().max_by(|a, b| {
        composite(a)
            .partial_cmp(&composite(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.updated_at.cmp(&b.updated_at))
    });

    // ====================================================================
    // STAGE 7: Relation Assessment (per-pair, using trust + temporal + similarity)
    // ====================================================================
    let mut pair_relations: Vec<(String, f64, RelationAssessment)> = Vec::new();
    if let Some(primary) = recommended {
        for other in scored.iter().filter(|s| s.id != primary.id).take(15) {
            // Use combined_score as a proxy for semantic similarity (already reranked)
            // Fall back to topic_overlap for keyword-level comparison
            let sim = topic_overlap(&primary.content, &other.content);
            let effective_sim = if other.combined_score > 0.2 {
                sim.max(0.3)
            } else {
                sim
            };
            let rel = assess_relation(
                &primary.content,
                &other.content,
                primary.trust,
                other.trust,
                primary.updated_at,
                other.updated_at,
                effective_sim,
            );
            if !matches!(rel.relation, Relation::Irrelevant) {
                pair_relations.push((other.content.chars().take(100).collect(), other.trust, rel));
            }
        }
    }

    // ====================================================================
    // STAGE 8: Synthesis + Reasoning Chain Generation
    // ====================================================================
    // `composite` and `recommended` were computed above (shared with STAGE 7
    // so the chain's PRIMARY FINDING and the citation card's Primary Source
    // are always the same memory).

    // Build evidence list (top memories by composite, not superseded)
    let mut non_superseded: Vec<&ScoredMemory> = non_superseded_all.clone();
    non_superseded.sort_by(|a, b| {
        composite(b)
            .partial_cmp(&composite(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let evidence: Vec<Value> = non_superseded
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, s)| {
            serde_json::json!({
                "id": s.id,
                "preview": s.content.chars().take(200).collect::<String>(),
                "trust": (s.trust * 100.0).round() / 100.0,
                "relevanceScore": ((composite(s) * 100.0).round() / 100.0),
                "date": s.updated_at.to_rfc3339(),
                "validUntil": s.valid_until.map(|dt| dt.to_rfc3339()),
                "currentlyValid": s.currently_valid,
                "role": if i == 0 { "primary" } else { "supporting" },
            })
        })
        .collect();

    // Build evolution timeline
    let mut evolution: Vec<Value> = by_date
        .iter()
        .rev()
        .map(|s| {
            serde_json::json!({
                "date": s.updated_at.format("%b %d, %Y").to_string(),
                "preview": s.content.chars().take(100).collect::<String>(),
                "trust": (s.trust * 100.0).round() / 100.0,
            })
        })
        .collect();
    evolution.truncate(15); // cap timeline length

    // Confidence scoring: derived from the same composite as `recommended`,
    // so confidence actually moves with query relevance instead of being a
    // function of trust + corpus size alone.
    let base_confidence = recommended.map(composite).unwrap_or(0.0);
    let agreement_boost = (evidence.len() as f64 * 0.03).min(0.2);
    // A claim that conflicts with a stored memory is the strongest possible signal
    // to lower confidence (heavier penalty than an inter-memory disagreement).
    let contradiction_penalty =
        (contradictions.len() as f64 * 0.1) + (claim_conflicts.len() as f64 * 0.2);
    let confidence = (base_confidence + agreement_boost - contradiction_penalty).clamp(0.0, 1.0);

    let status = if !claim_conflicts.is_empty() {
        // The claim itself conflicts with stored memory — never report "resolved".
        "claim_contradicts_memory"
    } else if contradictions.is_empty() && confidence > 0.7 {
        "resolved"
    } else if !contradictions.is_empty() {
        "contradictions_found"
    } else if scored.is_empty() {
        "no_evidence"
    } else {
        "partial_evidence"
    };

    let guidance = if !claim_conflicts.is_empty() {
        format!(
            "CAUTION: your claim conflicts with {} stored memor{}. Do NOT treat this as resolved — review the conflicting memory(ies) below before acting.",
            claim_conflicts.len(),
            if claim_conflicts.len() == 1 {
                "y"
            } else {
                "ies"
            }
        )
    } else if let Some(rec) = recommended {
        if contradictions.is_empty() {
            format!(
                "High confidence ({:.0}%). Recommended memory (trust {:.0}%, {}) is the most reliable source.",
                confidence * 100.0,
                rec.trust * 100.0,
                rec.updated_at.format("%b %d, %Y")
            )
        } else {
            format!(
                "WARNING: {} contradiction(s) detected. Recommended memory has trust {:.0}% but conflicts exist. Review contradictions below.",
                contradictions.len(),
                rec.trust * 100.0
            )
        }
    } else {
        "No strong evidence found. Verify with external sources.".to_string()
    };

    // Evidence shown to a caller is not automatically evidence of usefulness.
    let ids: Vec<&str> = scored.iter().map(|s| s.id.as_str()).collect();
    let _ = storage.record_batch_retrieval(&ids);

    // Generate reasoning chain (the key differentiator — no LLM needed)
    let reasoning_chain = if let Some(rec) = recommended {
        generate_reasoning_chain(&args.query, &intent, rec, &pair_relations, confidence)
    } else {
        "No strong evidence found for reasoning.".to_string()
    };

    // Build response
    let mut response = serde_json::json!({
        "query": args.query,
        "intent": format!("{:?}", intent),
        "status": status,
        "confidence": (confidence * 100.0).round() / 100.0,
        "reasoning": reasoning_chain,
        "guidance": guidance,
        "memoriesAnalyzed": scored.len(),
        "activationExpanded": activation_expanded,
    });

    if !claim_conflicts.is_empty() {
        response["claim_conflicts"] = serde_json::json!(claim_conflicts);
    }

    if let Some(rec) = recommended {
        response["recommended"] = serde_json::json!({
            "answer_preview": rec.content.chars().take(300).collect::<String>(),
            "memory_id": rec.id,
            "trust_score": (rec.trust * 100.0).round() / 100.0,
            "date": rec.updated_at.to_rfc3339(),
            "validUntil": rec.valid_until.map(|dt| dt.to_rfc3339()),
            "currentlyValid": rec.currently_valid,
        });
    }

    if !evidence.is_empty() {
        response["evidence"] = serde_json::json!(evidence);
    }
    if !contradictions.is_empty() {
        response["contradictions"] = serde_json::json!(contradictions);
    }
    if !superseded.is_empty() {
        response["superseded"] = serde_json::json!(superseded);
    }
    if !evolution.is_empty() {
        response["evolution"] = serde_json::json!(evolution);
    }
    if !related_insights.is_empty() {
        response["related_insights"] = serde_json::json!(related_insights);
    }

    match persist_deep_reference_composition(storage, &args.query, &intent, &response) {
        Ok(Some(event_id)) => {
            response["composition_event_id"] = serde_json::json!(event_id);
            response["compositionWriteStatus"] = serde_json::json!("persisted");
        }
        Ok(None) => {
            response["compositionWriteStatus"] = serde_json::json!("skipped_empty");
        }
        Err(err) => {
            tracing::warn!(
                "Failed to persist deep_reference composition event: {}",
                err
            );
            response["compositionWriteStatus"] = serde_json::json!("failed");
        }
    }

    Ok(response)
}

fn persist_deep_reference_composition(
    storage: &Arc<Storage>,
    query: &str,
    intent: &QueryIntent,
    response: &Value,
) -> Result<Option<String>, String> {
    let event_id = Uuid::new_v4().to_string();
    let event = CompositionEventRecord {
        id: event_id.clone(),
        created_at: Utc::now(),
        tool: "deep_reference".to_string(),
        mode: "deep_reference".to_string(),
        query: Some(query.to_string()),
        query_hash: Some(query_hash(query)),
        confidence: response.get("confidence").and_then(|v| v.as_f64()),
        status: response
            .get("status")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        output_preview: response
            .get("guidance")
            .and_then(|v| v.as_str())
            .map(|value| preview_text(value, 280)),
        metadata: serde_json::json!({
            "intent": format!("{:?}", intent),
            "memoriesAnalyzed": response.get("memoriesAnalyzed").and_then(|v| v.as_u64()).unwrap_or(0),
            "activationExpanded": response.get("activationExpanded").and_then(|v| v.as_u64()).unwrap_or(0),
            "reasoningPreview": response.get("reasoning").and_then(|v| v.as_str()).map(|value| preview_text(value, 600)),
        }),
    };

    let mut members = Vec::new();
    if let Some(evidence) = response.get("evidence").and_then(|v| v.as_array()) {
        for (idx, item) in evidence.iter().enumerate() {
            let Some(memory_id) = item.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let role = item
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or(if idx == 0 { "primary" } else { "supporting" });
            members.push(CompositionMemberRecord {
                event_id: event_id.clone(),
                memory_id: memory_id.to_string(),
                role: role.to_string(),
                rank: idx as i32,
                trust: item.get("trust").and_then(|v| v.as_f64()),
                score: item
                    .get("relevanceScore")
                    .or_else(|| item.get("relevance_score"))
                    .and_then(|v| v.as_f64()),
                preview: None,
                metadata: serde_json::json!({
                    "roleSource": "deep_reference_evidence",
                    "evidenceRank": idx,
                    "date": item.get("date").and_then(|v| v.as_str()),
                }),
            });
        }
    }

    if let Some(contradictions) = response.get("contradictions").and_then(|v| v.as_array()) {
        for (idx, contradiction) in contradictions.iter().enumerate() {
            for side in ["stronger", "weaker"] {
                let Some(item) = contradiction.get(side) else {
                    continue;
                };
                let Some(memory_id) = item.get("id").and_then(|v| v.as_str()) else {
                    continue;
                };
                members.push(CompositionMemberRecord {
                    event_id: event_id.clone(),
                    memory_id: memory_id.to_string(),
                    role: "contradicting".to_string(),
                    rank: idx as i32,
                    trust: item.get("trust").and_then(|v| v.as_f64()),
                    score: contradiction.get("topic_overlap").and_then(|v| v.as_f64()),
                    preview: None,
                    metadata: serde_json::json!({
                        "roleSource": "deep_reference_contradiction",
                        "side": side,
                        "date": item.get("date").and_then(|v| v.as_str()),
                    }),
                });
            }
        }
    }

    if let Some(superseded) = response.get("superseded").and_then(|v| v.as_array()) {
        for (idx, item) in superseded.iter().enumerate() {
            let Some(memory_id) = item.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            members.push(CompositionMemberRecord {
                event_id: event_id.clone(),
                memory_id: memory_id.to_string(),
                role: "superseded".to_string(),
                rank: idx as i32,
                trust: item.get("trust").and_then(|v| v.as_f64()),
                score: None,
                preview: None,
                metadata: serde_json::json!({
                    "roleSource": "deep_reference_superseded",
                    "superseded_by": item.get("superseded_by").and_then(|v| v.as_str()),
                    "date": item.get("date").and_then(|v| v.as_str()),
                }),
            });
        }
    }

    if members.is_empty() {
        return Ok(None);
    }

    storage
        .save_composition(&event, &members, &[])
        .map_err(|e| e.to_string())?;
    Ok(Some(event_id))
}

fn query_hash(query: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in query.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn preview_text(value: &str, max: usize) -> String {
    let collapsed = value.replace('\n', " ");
    if collapsed.len() <= max {
        return collapsed;
    }
    format!("{}...", &collapsed[..collapsed.floor_char_boundary(max)])
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{query_coverage, topic_overlap};

    /// The AIMO3 shape: two memories asserting OPPOSITE DIRECTIONS about the same
    /// subject, with no negation word in either. The negation scan structurally
    /// cannot see this, and it is the exact case where acting on the wrong side
    /// suppresses the memory that would have corrected it.
    #[test]
    fn opposite_direction_claims_are_contradictions_without_any_negation() {
        let hurts = "Prompt diversity monotonically hurts AIMO3 accuracy at temperature \
                     0.6 and above on GPT-OSS-120B during the competition run";
        let improves = "Prompt diversity monotonically improves AIMO3 accuracy at temperature \
                        0.6 and above on GPT-OSS-120B during the competition run";
        assert!(
            super::appears_contradictory(hurts, improves),
            "opposite-direction claims about the same subject must be contradictory"
        );

        // Neither memory contains a negation word, which is why the older
        // negation-asymmetry scan missed this entirely.
        for text in [hurts, improves] {
            let t = text.to_lowercase();
            assert!(!t.contains("never") && !t.contains("don't") && !t.contains("avoid"));
        }
    }

    /// Mutually exclusive values for the same attribute -- the shape MemConflict
    /// measures and the one both other detection paths structurally miss. Verified
    /// against the live server: both memories were retrievable via lookup while
    /// recall(mode="contradictions") returned ZERO pairs.
    #[test]
    fn same_attribute_different_value_is_a_contradiction() {
        let bachelor = "Priya holds a Bachelor degree in computer science from Leeds University";
        let master = "Priya holds a Master degree in computer science from Leeds University";
        assert!(
            super::appears_contradictory(bachelor, master),
            "two incompatible values for the same attribute must be contradictory"
        );

        // No negation and no antonym in either -- this is why the other two paths
        // cannot see it.
        for t in [bachelor, master] {
            let l = t.to_lowercase();
            assert!(!l.contains("never") && !l.contains("not ") && !l.contains("hurts"));
        }
    }

    /// Elaboration is not contradiction. A memory that ADDS detail to another must
    /// never be flagged, or every refinement becomes a conflict.
    #[test]
    fn elaboration_is_not_flagged_as_contradiction() {
        let base = "Priya works in the Leeds office on the payments platform team";
        let more =
            "Priya works in the Leeds office on the payments platform team and mentors interns";
        assert!(
            !super::appears_contradictory(base, more),
            "a superset elaboration must not be a contradiction"
        );
    }

    /// The antonym test must not fire on agreement, or on two memories that merely
    /// share a domain. Both sides must be present in OPPOSITE memories.
    #[test]
    fn antonym_detection_does_not_fire_on_agreement_or_unrelated_text() {
        let a = "Prompt diversity monotonically hurts AIMO3 accuracy at temperature \
                 0.6 and above on GPT-OSS-120B";
        let same = "Prompt diversity monotonically hurts AIMO3 accuracy at temperature \
                    0.6 and above on GPT-OSS-120B, confirmed twice";
        assert!(
            !super::appears_contradictory(a, same),
            "two memories agreeing must not be flagged as contradictory"
        );

        let unrelated = "The dashboard accent colour improves legibility in dark mode";
        assert!(
            !super::appears_contradictory(a, unrelated),
            "different subjects must not be flagged merely because antonyms appear"
        );
    }

    /// STAGE 5b's claim-vs-memory conflict gate was built on symmetric Jaccard,
    /// which divides by the UNION. A short claim can therefore never clear a 0.4
    /// bar against a normal-length memory, so the stage never fired and a query
    /// contradicting a high-trust memory passed in silence. This pins the
    /// arithmetic so the gate cannot quietly die again.
    #[test]
    fn claim_conflict_gate_is_reachable_for_a_real_claim() {
        let memory = "Paper arxiv 2603.27844 tested on GPT-OSS-120B for AIMO3 on H100 and \
                      found that prompt diversity monotonically hurts accuracy at temperature \
                      0.6 and above; every intervention fails, so submit the unmodified \
                      baseline repeatedly instead of stacking changes.";
        let claim = "prompt diversity improves accuracy at temperature 0.6 and above on \
                     GPT-OSS-120B for AIMO3";

        // The old instrument: unreachable, nowhere near the 0.4 gate it was tested against.
        let jaccard = topic_overlap(claim, memory);
        assert!(
            jaccard < 0.4,
            "symmetric Jaccard should be unreachable here, got {jaccard}"
        );

        // The correct instrument: most of the claim's substantive words are present.
        let coverage = query_coverage(claim, memory);
        assert!(
            coverage >= 0.5,
            "a claim restating this memory must clear the coverage gate, got {coverage}"
        );
    }

    /// Coverage must still reject a claim that simply is not about the memory.
    #[test]
    fn claim_conflict_gate_still_rejects_unrelated_claims() {
        let memory = "Paper arxiv 2603.27844 tested prompt diversity on AIMO3 and found it hurts.";
        let unrelated = "the dashboard accent colour should be cyan instead of indigo";
        assert!(query_coverage(unrelated, memory) < 0.5);
    }

    use super::*;
    use crate::cognitive::CognitiveEngine;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::Mutex;
    use vestige_core::Storage;

    fn test_cognitive() -> Arc<Mutex<CognitiveEngine>> {
        Arc::new(Mutex::new(CognitiveEngine::new()))
    }

    async fn test_storage() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        (Arc::new(storage), dir)
    }

    async fn ingest_one(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> String {
        storage
            .ingest(vestige_core::IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: tags.iter().map(|s| s.to_string()).collect(),
                valid_from: None,
                valid_until: None,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    async fn ingest_with_validity(
        storage: &Arc<Storage>,
        content: &str,
        tags: &[&str],
        valid_until: Option<chrono::DateTime<Utc>>,
    ) -> String {
        storage
            .ingest(vestige_core::IngestInput {
                content: content.to_string(),
                node_type: "fact".to_string(),
                source: None,
                sentiment_score: 0.0,
                sentiment_magnitude: 0.0,
                tags: tags.iter().map(|s| s.to_string()).collect(),
                valid_from: None,
                valid_until,
                validity_inferred: false,
                source_envelope: None,
            })
            .unwrap()
            .id
    }

    fn evidence_entry<'a>(result: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
        result["evidence"]
            .as_array()
            .expect("evidence array should be present")
            .iter()
            .find(|e| e["id"].as_str() == Some(id))
            .unwrap_or_else(|| panic!("memory {} missing from evidence", id))
    }

    // ========================================================================
    // BUG A: `recommended` is picked by FSRS trust only, ignoring query relevance.
    // ========================================================================
    #[tokio::test]
    async fn test_recommended_uses_query_relevance_not_just_trust() {
        let (storage, _dir) = test_storage().await;

        let id_a = ingest_one(
            &storage,
            "PostgreSQL connection pooling with pgbouncer transaction mode \
             requires careful tuning of max_connections and pool_mode settings.",
            &["postgres", "database"],
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let _id_b = ingest_one(
            &storage,
            "Making sourdough bread requires a mature starter, long bulk \
             fermentation, and attention to dough hydration levels.",
            &["baking", "bread"],
        )
        .await;

        let args = serde_json::json!({
            "query": "PostgreSQL connection pooling pgbouncer max_connections"
        });
        let result = execute(&storage, &test_cognitive(), Some(args))
            .await
            .expect("execute should succeed");

        assert_eq!(
            result["recommended"]["memory_id"].as_str(),
            Some(id_a.as_str()),
            "Expected recommended={} (matches query). Got {:?}. \
             Root cause: lines 565-572 select `recommended` by trust only, \
             discarding the combined_score signal from hybrid_search + reranker.",
            id_a,
            result["recommended"]["memory_id"]
        );
    }

    #[tokio::test]
    async fn test_deep_reference_persists_composition_event() {
        let (storage, _dir) = test_storage().await;

        let primary_id = ingest_one(
            &storage,
            "ProtocolGate control-plane composition tracks global invariant local gate bypasses.",
            &["protocolgate", "boundary-scope"],
        )
        .await;
        let supporting_id = ingest_one(
            &storage,
            "ProtocolGate global invariant local gate research used Aave account-global health factor and route-local validation.",
            &["protocolgate", "boundary-scope"],
        )
        .await;

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "ProtocolGate global invariant local gate",
                "depth": 10
            })),
        )
        .await
        .expect("execute should succeed");

        let event_id = result["composition_event_id"]
            .as_str()
            .expect("deep_reference should return persisted event id");
        assert_eq!(result["compositionWriteStatus"].as_str(), Some("persisted"));

        let event = storage
            .get_composition_event(event_id)
            .unwrap()
            .expect("composition event should be stored");
        assert_eq!(event.tool, "deep_reference");
        assert_eq!(
            event.query.as_deref(),
            Some("ProtocolGate global invariant local gate")
        );

        let members = storage.get_composition_members(event_id).unwrap();
        assert!(members.iter().any(|member| member.memory_id == primary_id));
        assert!(
            members
                .iter()
                .any(|member| member.memory_id == supporting_id)
        );
        assert!(members.iter().any(|member| member.role == "primary"));
        assert!(
            members.iter().any(|member| {
                member.memory_id == primary_id
                    && member.score.is_some()
                    && member.metadata["roleSource"] == "deep_reference_evidence"
            }),
            "persisted members should retain relevance score and role source"
        );
    }

    #[tokio::test]
    async fn test_deep_reference_skips_empty_composition_event() {
        let (storage, _dir) = test_storage().await;

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "no memories exist for this query",
                "depth": 10
            })),
        )
        .await
        .expect("execute should succeed");

        assert_eq!(
            result["compositionWriteStatus"].as_str(),
            Some("skipped_empty")
        );
        assert!(
            result.get("composition_event_id").is_none(),
            "empty evidence should not create a composition event"
        );
        assert!(
            storage
                .get_recent_composition_events(10)
                .unwrap()
                .is_empty(),
            "ledger should stay empty when no memories participated"
        );
    }

    // ========================================================================
    // Confidence sanity: must vary with query relevance.
    // ========================================================================
    #[tokio::test]
    async fn test_confidence_varies_with_query_relevance() {
        let (storage, _dir) = test_storage().await;

        ingest_one(
            &storage,
            "The Borrow Checker enforces Rust's ownership rules at compile time, \
             preventing data races and use-after-free without a garbage collector.",
            &["rust"],
        )
        .await;

        let relevant = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "Rust borrow checker ownership compile time"
            })),
        )
        .await
        .expect("execute should succeed");

        let irrelevant = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "18th century Dutch maritime trade routes"
            })),
        )
        .await
        .expect("execute should succeed");

        let rel_conf = relevant["confidence"].as_f64().unwrap_or(0.0);
        let irr_conf = irrelevant["confidence"].as_f64().unwrap_or(0.0);

        assert!(
            rel_conf > irr_conf,
            "Confidence should be higher for a relevant query. Got \
             relevant={}, irrelevant={}. Currently `confidence` derives from \
             recommended.trust + evidence count (lines 602-605), both \
             invariant under query changes.",
            rel_conf,
            irr_conf
        );
    }

    #[test]
    fn test_schema_structure() {
        let s = schema();
        assert!(s["properties"]["query"].is_object());
        assert!(s["properties"]["depth"].is_object());
        assert_eq!(s["required"], serde_json::json!(["query"]));
    }

    #[test]
    fn test_trust_score_high() {
        // High retention, high stability, many reps, no lapses → high trust
        let trust = compute_trust(0.95, 60.0, 20, 0);
        assert!(trust > 0.8, "Expected >0.8, got {}", trust);
    }

    #[test]
    fn test_trust_score_low() {
        // Low retention, low stability, few reps, many lapses → low trust
        let trust = compute_trust(0.2, 1.0, 1, 10);
        assert!(trust < 0.3, "Expected <0.3, got {}", trust);
    }

    #[test]
    fn test_trust_score_medium() {
        // Medium everything
        let trust = compute_trust(0.6, 15.0, 5, 2);
        assert!(
            trust > 0.4 && trust < 0.7,
            "Expected 0.4-0.7, got {}",
            trust
        );
    }

    #[test]
    fn test_trust_score_clamped() {
        // Even extreme values stay in [0, 1]
        assert!(compute_trust(1.0, 1000.0, 100, 0) <= 1.0);
        assert!(compute_trust(0.0, 0.0, 0, 100) >= 0.0);
    }

    #[test]
    fn test_contradiction_requires_shared_words() {
        assert!(!appears_contradictory(
            "not sure about weather",
            "Rust is fast"
        ));
    }

    #[test]
    fn test_contradiction_with_shared_context() {
        assert!(appears_contradictory(
            "Don't use FAISS for vector search in production",
            "Use FAISS for vector search in production always"
        ));
    }

    // ========================================================================
    // STAGE 5b AUDIT: a NON-contradicting claim must NOT set
    // status=claim_contradicts_memory; a contradicting claim MUST.
    // ========================================================================
    #[tokio::test]
    async fn audit_stage5b_noncontradicting_claim_is_not_flagged() {
        let (storage, _dir) = test_storage().await;

        // High-overlap, AGREEING memory: same subject, same stance.
        ingest_one(
            &storage,
            "Vestige uses USearch HNSW for vector search with cosine similarity \
             and Matryoshka truncation to 256 dimensions for storage savings.",
            &["vestige", "vector-search"],
        )
        .await;

        // Claim that AGREES (no negation, no correction marker, same subject).
        let args = serde_json::json!({
            "query": "Vestige uses USearch HNSW for vector search with cosine \
                      similarity and Matryoshka truncation to 256 dimensions"
        });
        let result = execute(&storage, &test_cognitive(), Some(args))
            .await
            .expect("execute should succeed");

        // Non-vacuous: the memory MUST have been retrieved (else the assertion
        // below would pass trivially via the no_memories early-return).
        assert!(
            result["memoriesAnalyzed"].as_i64().unwrap_or(0) >= 1,
            "Expected the agreeing memory to be retrieved (memoriesAnalyzed>=1). Got {:?}",
            result["memoriesAnalyzed"]
        );
        assert_ne!(
            result["status"].as_str(),
            Some("claim_contradicts_memory"),
            "A NON-contradicting (agreeing) claim must not be flagged. Got status={:?}, claim_conflicts={:?}",
            result["status"],
            result.get("claim_conflicts")
        );
        assert!(
            result.get("claim_conflicts").is_none(),
            "No claim_conflicts array should be present for an agreeing claim. Got {:?}",
            result.get("claim_conflicts")
        );
    }

    // STAGE 5b decision predicate, tested directly. The end-to-end `execute`
    // path cannot surface a genuinely-contradicting claim in a test env with no
    // embeddings model loaded, because keyword retrieval is implicit-AND and a
    // contradicting claim by construction carries a stance word the memory
    // lacks. This asserts the exact gate STAGE 5b applies once a memory is
    // retrieved: topic_overlap >= 0.4 AND appears_contradictory(query, memory).
    #[test]
    fn audit_stage5b_gate_predicate_distinguishes_agree_vs_contradict() {
        let memory = "USearch HNSW vector search Vestige production cosine similarity \
                      recall correct should always be enabled because it is fast";

        // Agreeing claim: high overlap, NO stance flip → must NOT trip the gate.
        let agree = "USearch HNSW vector search Vestige production cosine similarity \
                     recall correct should always be enabled because it is fast";
        assert!(
            topic_overlap(agree, memory) >= 0.4,
            "agree/memory should share topic"
        );
        assert!(
            !appears_contradictory(agree, memory),
            "An agreeing claim must NOT be flagged as contradictory (false-positive guard)"
        );

        // Contradicting claim: same subject + a negation marker ("never"/"avoid")
        // present in exactly one side → must trip the gate.
        let contradict = "USearch HNSW vector search Vestige production cosine similarity \
                          recall avoid never enabled";
        assert!(
            topic_overlap(contradict, memory) >= 0.4,
            "contradict/memory should share topic"
        );
        assert!(
            appears_contradictory(contradict, memory),
            "A same-subject negated claim MUST be flagged as contradictory"
        );
    }

    #[test]
    fn test_topic_overlap_similar() {
        let overlap = topic_overlap(
            "Vestige uses USearch for vector search",
            "Vestige vector search powered by USearch HNSW",
        );
        assert!(overlap > 0.3);
    }

    #[test]
    fn test_topic_overlap_different() {
        let overlap = topic_overlap("The weather is sunny today", "Rust compile times improving");
        assert!(overlap < 0.15);
    }

    #[test]
    fn test_depth_clamped() {
        let s = schema();
        assert_eq!(s["properties"]["depth"]["minimum"], 5);
        assert_eq!(s["properties"]["depth"]["maximum"], 50);
    }

    // === Intent Classification Tests ===

    #[test]
    fn test_intent_fact_check() {
        assert_eq!(
            classify_intent("Is it true that Vestige uses USearch?"),
            QueryIntent::FactCheck
        );
        assert_eq!(
            classify_intent("Did I switch to port 3002?"),
            QueryIntent::FactCheck
        );
        assert_eq!(
            classify_intent("Should I use prefix caching?"),
            QueryIntent::FactCheck
        );
    }

    #[test]
    fn test_intent_timeline() {
        assert_eq!(
            classify_intent("When did the port change happen?"),
            QueryIntent::Timeline
        );
        assert_eq!(
            classify_intent("How has the benchmark score evolved over time?"),
            QueryIntent::Timeline
        );
    }

    #[test]
    fn test_intent_root_cause() {
        assert_eq!(
            classify_intent("Why did the build fail?"),
            QueryIntent::RootCause
        );
        assert_eq!(
            classify_intent("What caused the score regression?"),
            QueryIntent::RootCause
        );
    }

    #[test]
    fn test_intent_comparison() {
        assert_eq!(
            classify_intent("How does USearch differ from FAISS?"),
            QueryIntent::Comparison
        );
        assert_eq!(
            classify_intent("Compare FSRS versus SM-2"),
            QueryIntent::Comparison
        );
    }

    #[test]
    fn test_intent_synthesis_default() {
        assert_eq!(
            classify_intent("Tell me about the user's projects"),
            QueryIntent::Synthesis
        );
        assert_eq!(classify_intent("What is Vestige?"), QueryIntent::Synthesis);
    }

    // === Relation Assessment Tests ===

    #[test]
    fn test_relation_irrelevant() {
        let rel = assess_relation(
            "Rust is fast",
            "The weather is nice",
            0.8,
            0.8,
            Utc::now(),
            Utc::now(),
            0.05,
        );
        assert!(matches!(rel.relation, Relation::Irrelevant));
    }

    #[test]
    fn test_relation_supports() {
        let rel = assess_relation(
            "Vestige uses USearch for vector search",
            "USearch provides fast HNSW indexing for Vestige",
            0.8,
            0.7,
            Utc::now(),
            Utc::now(),
            0.6,
        );
        assert!(matches!(rel.relation, Relation::Supports));
    }

    #[test]
    fn test_relation_contradicts() {
        let rel = assess_relation(
            "Don't use FAISS for vector search in production anymore",
            "Use FAISS for vector search in production always",
            0.8,
            0.5,
            Utc::now(),
            Utc::now(),
            0.7,
        );
        assert!(matches!(rel.relation, Relation::Contradicts));
    }

    // ========================================================================
    // VALIDITY (issue #156 Ask 1): expired facts must not compose at full rank.
    // Mirrors search_unified's default validity penalty on the reason path.
    // ========================================================================

    #[test]
    fn test_validity_adjusted_score_only_penalizes_invalid() {
        assert_eq!(validity_adjusted_score(0.8, true), 0.8);
        let penalized = validity_adjusted_score(0.8, false);
        assert!(
            (penalized - 0.08).abs() < 1e-6,
            "invalid facts must be downranked to 0.1x, got {}",
            penalized
        );
    }

    #[tokio::test]
    async fn test_expired_memory_ranked_below_current_and_flagged() {
        let (storage, _dir) = test_storage().await;

        let expired_id = ingest_with_validity(
            &storage,
            "Kubernetes ingress gateway timeout policy for the payments cluster \
             is ninety seconds.",
            &["kubernetes", "payments"],
            Some(Utc::now() - chrono::Duration::days(30)),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let current_id = ingest_with_validity(
            &storage,
            "Kubernetes ingress gateway timeout policy for the payments cluster \
             is thirty seconds.",
            &["kubernetes", "payments"],
            None,
        )
        .await;

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "Kubernetes ingress gateway timeout policy payments cluster"
            })),
        )
        .await
        .expect("execute should succeed");

        let expired = evidence_entry(&result, &expired_id);
        let current = evidence_entry(&result, &current_id);

        assert_eq!(
            expired["currentlyValid"].as_bool(),
            Some(false),
            "expired memory must carry currentlyValid=false: {:?}",
            expired
        );
        assert!(
            expired["validUntil"].as_str().is_some(),
            "expired memory must surface its validUntil (RFC3339): {:?}",
            expired
        );
        assert_eq!(
            current["currentlyValid"].as_bool(),
            Some(true),
            "current memory must carry currentlyValid=true: {:?}",
            current
        );

        let expired_score = expired["relevanceScore"].as_f64().unwrap();
        let current_score = current["relevanceScore"].as_f64().unwrap();
        assert!(
            expired_score < current_score,
            "an expired fact must rank below its current replacement \
             (expired={}, current={}). Without the validity penalty, both \
             carry identical trust/terms and the expired one can win.",
            expired_score,
            current_score
        );
        assert_eq!(
            result["recommended"]["memory_id"].as_str(),
            Some(current_id.as_str()),
            "the current fact must win primary selection over the expired one"
        );
        assert_eq!(
            result["recommended"]["currentlyValid"].as_bool(),
            Some(true),
            "recommended block must surface validity too"
        );
    }

    #[tokio::test]
    async fn test_future_valid_until_not_penalized() {
        let (storage, _dir) = test_storage().await;

        let expired_id = ingest_with_validity(
            &storage,
            "Redis cache eviction policy for the checkout service keeps entries \
             for sixty minutes.",
            &["redis", "checkout"],
            Some(Utc::now() - chrono::Duration::days(30)),
        )
        .await;

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let future_id = ingest_with_validity(
            &storage,
            "Redis cache eviction policy for the checkout service keeps entries \
             for ninety minutes.",
            &["redis", "checkout"],
            Some(Utc::now() + chrono::Duration::days(365)),
        )
        .await;

        let result = execute(
            &storage,
            &test_cognitive(),
            Some(serde_json::json!({
                "query": "Redis cache eviction policy checkout service entries"
            })),
        )
        .await
        .expect("execute should succeed");

        let future = evidence_entry(&result, &future_id);
        let expired = evidence_entry(&result, &expired_id);

        assert_eq!(
            future["currentlyValid"].as_bool(),
            Some(true),
            "a fact whose valid_until is in the FUTURE is currently valid: {:?}",
            future
        );
        assert!(
            future["validUntil"].as_str().is_some(),
            "future validUntil must still be surfaced for the reasoning layer: {:?}",
            future
        );

        let future_score = future["relevanceScore"].as_f64().unwrap();
        let expired_score = expired["relevanceScore"].as_f64().unwrap();
        // If future validity were wrongly penalized (e.g. keying the penalty on
        // valid_until.is_some() instead of is_currently_valid()), both twins
        // would sit in the same 0.1x band and the gap collapses to <=~0.03.
        // Unpenalized, the future-valid fact keeps its full 0.5-weighted
        // relevance slot: the gap over the penalized expired twin is >=~0.09.
        assert!(
            future_score - expired_score >= 0.05,
            "a future-valid fact must NOT be penalized (future={}, expired={})",
            future_score,
            expired_score
        );
        assert_eq!(
            result["recommended"]["memory_id"].as_str(),
            Some(future_id.as_str()),
            "the future-valid fact must win primary selection over the expired one"
        );
    }
}

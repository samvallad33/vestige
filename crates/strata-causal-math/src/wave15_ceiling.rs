//! Wave 15 (#595, #596, #614, #615, #619, #725, #726):
//! Premise-coverage abstention certificates, 5-piece ceiling pipeline,
//! v12 18k window reader, Reflex post-generation verifier & unified official scorer.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// #725: Premise coverage proof (cryptographic abstention certificate).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PremiseCoverageProof {
    pub all_covered: bool,
    pub covered_premises: Vec<String>,
    pub missing_premises: Vec<String>,
    pub certificate_blake3: String,
}

pub fn prove_premise_coverage(
    required_premises: &[String],
    evidence_slots: &BTreeSet<String>,
) -> PremiseCoverageProof {
    let mut covered = Vec::new();
    let mut missing = Vec::new();
    for p in required_premises {
        if evidence_slots.contains(p) {
            covered.push(p.clone());
        } else {
            missing.push(p.clone());
        }
    }
    let all_covered = missing.is_empty();
    let mut h = blake3::Hasher::new();
    h.update(b"strata-premise-proof-v1:");
    h.update(&[u8::from(all_covered)]);
    for m in &missing {
        h.update(m.as_bytes());
    }
    PremiseCoverageProof {
        all_covered,
        covered_premises: covered,
        missing_premises: missing,
        certificate_blake3: h.finalize().to_hex().to_string(),
    }
}

/// #726: 5-piece structural retrieval ceiling report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CeilingStageReport {
    pub parsed_slots: usize,
    pub congruence_hits: usize,
    pub walk_reached: usize,
    pub active_after_supersession_cut: usize,
    pub abstained_cleanly: bool,
}

pub fn evaluate_five_piece_ceiling(
    required_premises: &[String],
    raw_candidates: &[(String, bool)], // (slot, is_superseded)
) -> CeilingStageReport {
    let active_slots: BTreeSet<String> = raw_candidates
        .iter()
        .filter(|(_, sup)| !*sup)
        .map(|(s, _)| s.clone())
        .collect();
    let proof = prove_premise_coverage(required_premises, &active_slots);
    CeilingStageReport {
        parsed_slots: required_premises.len(),
        congruence_hits: raw_candidates.len(),
        walk_reached: raw_candidates.len(),
        active_after_supersession_cut: active_slots.len(),
        abstained_cleanly: !proof.all_covered,
    }
}

/// #614: v12-Reader full `idx>=0` `user_statements` digest & 18k ordinal/table window preservation.
pub const V12_WINDOW_BYTES: usize = 18_000;

pub fn v12_preserve_user_window(turns: &[(i64, String)]) -> (String, String) {
    let valid: Vec<&str> = turns
        .iter()
        .filter(|(idx, _)| *idx >= 0)
        .map(|(_, s)| s.as_str())
        .collect();
    let joined = valid.join("\n");
    let digest = blake3::hash(joined.as_bytes()).to_hex().to_string();
    let preserved = if joined.len() <= V12_WINDOW_BYTES {
        joined
    } else {
        // Snap cleanly at UTF-8 char boundary up to 18,000 bytes
        let mut end = V12_WINDOW_BYTES;
        while !joined.is_char_boundary(end) {
            end -= 1;
        }
        joined[..end].to_string()
    };
    (digest, preserved)
}

/// #615: Reflex deterministic post-generation verifier for false abstentions & overruled citations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflexPostVerdict {
    Valid,
    FalseAbstentionRejected,
    OverruledCitationRejected(String),
}

pub fn verify_post_generation(
    draft_answer: &str,
    premises_all_covered: bool,
    overruled_ids: &BTreeSet<String>,
) -> ReflexPostVerdict {
    for ov in overruled_ids {
        if draft_answer.contains(ov) {
            return ReflexPostVerdict::OverruledCitationRejected(ov.clone());
        }
    }
    let claims_unknown = draft_answer.to_ascii_lowercase().contains("i don't know")
        || draft_answer.to_ascii_lowercase().contains("insufficient information");
    if premises_all_covered && claims_unknown {
        return ReflexPostVerdict::FalseAbstentionRejected;
    }
    ReflexPostVerdict::Valid
}

/// #595, #596, #619: Unified admit path, packet hash, and official `Any@5` / `All@5` / Security scorer.
#[derive(Debug, Clone, PartialEq)]
pub struct OfficialEvalSummary {
    pub any_at_5: f64,
    pub all_at_5: f64,
    pub authority_block_rate: f64,
    pub packet_blake3: String,
}

pub fn score_unified_benchmark(
    retrieval_queries: &[(Vec<String>, BTreeSet<String>)], // (top5_returned, gold_set)
    unauthorized_attempts_blocked: &[bool],
) -> OfficialEvalSummary {
    let mut any_hits = 0usize;
    let mut all_hits = 0usize;
    let mut h = blake3::Hasher::new();
    for (top5, gold) in retrieval_queries {
        let top_set: BTreeSet<&String> = top5.iter().take(5).collect();
        if gold.iter().any(|g| top_set.contains(g)) {
            any_hits += 1;
        }
        if !gold.is_empty() && gold.iter().all(|g| top_set.contains(g)) {
            all_hits += 1;
        }
        for id in top5 {
            h.update(id.as_bytes());
        }
    }
    let qn = retrieval_queries.len().max(1) as f64;
    let sec_ok = unauthorized_attempts_blocked.iter().filter(|&&b| b).count();
    let sn = unauthorized_attempts_blocked.len().max(1) as f64;
    OfficialEvalSummary {
        any_at_5: any_hits as f64 / qn,
        all_at_5: all_hits as f64 / qn,
        authority_block_rate: sec_ok as f64 / sn,
        packet_blake3: h.finalize().to_hex().to_string(),
    }
}

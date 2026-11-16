//! src/kernel/engine.rs — File 3: The ZOOM, BRIDGES, & VARIETY File
//! 1. ZOOM: 3-tier hierarchical index (session_bags -> window_bags -> turn_bags)
//! 2. BRIDGES: 2-iteration sparse Personalized PageRank over rare bridge entities
//! 3. VARIETY: Greedy Submodular Set Cover for maximal information diversity

use std::collections::{HashMap, HashSet};
use super::intern::{StringInterner, PackedFeature, TokenSequence};
use super::bitmask::{QueryMask, uncovered_query_term_count, marginal_information_ratio};

// -------------------------------------------------------------------------
// 1. ZOOM: Three-Tier Hierarchical Index
// -------------------------------------------------------------------------
pub struct HierarchicalIndex {
    pub interner: StringInterner,
    
    // Macro Tier: Whole-session BM25F bag of words
    pub session_bags: Vec<HashMap<u32, f32>>,
    
    // Meso Tier: 3-turn sliding window bags with proximity bonus
    pub window_bags: Vec<HashMap<u32, f32>>,
    
    // Micro Tier: Turn-level bags with question/answer dyad coupling
    pub turn_bags: Vec<HashMap<u32, f32>>,
    pub turn_masks: Vec<QueryMask>,
    pub turn_sessions: Vec<usize>,
}

// -------------------------------------------------------------------------
// 2. BRIDGES: 2-Iteration Sparse Personalized PageRank
// -------------------------------------------------------------------------
pub fn spread_activation_bridges(
    seed_turns: &[usize],
    turn_scores: &mut [f32],
    turn_content: &[HashSet<u32>],
    turn_masks: &[QueryMask],
    doc_frequency: &HashMap<u32, u32>,
    total_docs: f32,
) {
    if seed_turns.is_empty() {
        return;
    }
    let t0 = seed_turns[0];
    let t0_mask = turn_masks[t0];
    let mut activation: HashMap<u32, f32> = HashMap::new();

    // Pass 1: Turn -> Rare Bridge Entities (IDF Filtered)
    for &s in seed_turns {
        let s_score = turn_scores[s];
        for &w in &turn_content[s] {
            if let Some(&df) = doc_frequency.get(&w) {
                // Must appear in >= 2 documents, but <= 20% of the entire corpus
                if df >= 2 && (df as f32) <= 0.20 * total_docs {
                    let idf_term = 1.0 / ((df as f32 + 1.1).ln());
                    *activation.entry(w).or_insert(0.0) += s_score * idf_term;
                }
            }
        }
    }

    // Pass 2: Bridge Entities -> 2nd-Hop Context Lines
    for ti in 0..turn_scores.len() {
        let mut ppr_sum = 0.0f32;
        for &w in &turn_content[ti] {
            if let Some(&act) = activation.get(&w) {
                ppr_sum += act;
            }
        }
        if ppr_sum > 0.0 {
            let mut boost = 0.45 * ppr_sum;
            // Bridge Bonus: 1.75x if this turn provides missing query terms
            if (turn_masks[ti] & !t0_mask) != 0 {
                boost *= 1.75;
            }
            turn_scores[ti] += boost;
        }
    }
}

// -------------------------------------------------------------------------
// 3. VARIETY: Greedy Submodular Set Cover for Top-K
// -------------------------------------------------------------------------
pub fn select_diverse_top_k(
    candidate_scores: &[f32],
    turn_masks: &[QueryMask],
    turn_sessions: &[usize],
    total_query_groups: usize,
    k: usize,
) -> Vec<usize> {
    let mut selected_turns: Vec<usize> = Vec::with_capacity(k);
    let mut covered_query_terms: QueryMask = 0;
    let mut session_counts: HashMap<usize, usize> = HashMap::new();
    let mut remaining: HashSet<usize> = (0..candidate_scores.len())
        .filter(|&ti| candidate_scores[ti] > 0.0)
        .collect();

    while selected_turns.len() < k && !remaining.is_empty() {
        let mut best_turn = None;
        let mut best_gain = -1.0f32;

        for &ti in &remaining {
            let base_score = candidate_scores[ti];
            let sess = turn_sessions[ti];
            let count = session_counts.get(&sess).copied().unwrap_or(0);

            // Submodular Session Penalty: penalizes echo/duplicate context
            let diversity_factor = match count {
                0 => 1.0,
                1 => 0.70,
                _ => 0.40,
            };

            // Hardware POPCNT calculation of net-new information
            let uncovered_ratio = marginal_information_ratio(
                turn_masks[ti],
                covered_query_terms,
                total_query_groups,
            );

            // Marginal gain combines score affinity, novel query coverage, and diversity
            let gain = base_score * (1.0 + 0.60 * uncovered_ratio) * diversity_factor;
            if gain > best_gain {
                best_gain = gain;
                best_turn = Some(ti);
            }
        }

        if let Some(chosen) = best_turn {
            selected_turns.push(chosen);
            remaining.remove(&chosen);
            covered_query_terms |= turn_masks[chosen];
            *session_counts.entry(turn_sessions[chosen]).or_insert(0) += 1;
        } else {
            break;
        }
    }

    selected_turns
}

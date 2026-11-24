//! Query-Anchored Session-Summary Silhouette (#5) & Multi-Episode Milestone Lattice (#4).
//!
//! Used by `stage12_session_sync_and_fusion` to pick substantive aspect-broad user initiative
//! turns per session on `summarization` queries and spread `event_ordering` Top-K across
//! chronological conversation phases (early / middle / late).

use crate::retrieval::{DetHashSet, Index, raw_tokens, turn_pos};

/// Returns `true` when the query asks for a chronological progression / ordering of events.
pub(in crate::retrieval) fn is_chronological_progression_query(q_text: &str) -> bool {
    let low = q_text.trim().to_ascii_lowercase();
    low.contains("order in which")
        || low.contains("walk me through")
        || low.contains("chronological")
        || low.contains("over time")
        || low.contains("sequence of")
        || low.contains("timeline")
}

/// Score a turn for `summarization` / `event_ordering` representative selection:
/// rewards substantive user initiative turns (`field == 0`, opening/declarative, broad unique
/// high-IC vocabulary + exact query group coverage) and penalizes short ack/clarification turns
/// (`"ok cool, do i need..."`).
pub(in crate::retrieval) fn summary_turn_silhouette_score(
    ix: &Index,
    ti: usize,
    t_scene: &[f32],
    exact_bits: u32,
) -> (u8, u32, usize, u64) {
    let t = &ix.turns[ti];
    let is_substantive = u8::from(
        !t.is_backchannel
            && !t.is_standing_policy
            && ((t.field == 0
                && (!t.is_question || t.content.len() >= 10)
                && t.body_for_headers.len() >= 36)
                || (t.field == 1
                    && !t.is_question
                    && exact_bits >= 2
                    && t.body_for_headers.len() >= 72)),
    );
    let is_opening = usize::from(t.field == 0 && turn_pos(ix, ti) == Some(0));
    let breadth = t.content.len().min(40) + is_opening * 6;
    let sc_q16 = (t_scene.get(ti).copied().unwrap_or(0.0).max(0.0) * 65_536.0) as u64;
    (is_substantive, exact_bits, breadth, sc_q16)
}

/// Spread `sessions` across chronological conversation thirds (`early`, `mid`, `late`) when
/// the query asks for a multi-episode progression/order, keeping the #1 anchor at index 0
/// and interleaving unseen chronological phases in slots `1..target_k`.
pub(in crate::retrieval) fn interleave_chronological_milestones(
    ix: &Index,
    sessions: &[usize],
    target_k: usize,
) -> Vec<usize> {
    if sessions.len() <= 2 || ix.session_turns.len() < 3 {
        return sessions.to_vec();
    }
    let n_sess = ix.session_turns.len();
    let mut bucket: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for &si in sessions {
        let phase = (si.saturating_mul(3) / n_sess).min(2);
        bucket[phase].push(si);
    }

    let mut out: Vec<usize> = Vec::with_capacity(sessions.len());
    let mut seen: DetHashSet<usize> = DetHashSet::default();
    // Always preserve top-ranked session first so Any@1 isn't disrupted.
    if let Some(&first) = sessions.first() {
        out.push(first);
        seen.insert(first);
    }

    // Round-robin across chronological phases `[0, 1, 2]` up to `target_k.max(5)`.
    let cap = target_k.max(5).min(sessions.len());
    let mut idx = [0usize; 3];
    while out.len() < cap {
        let mut progressed = false;
        for ph in 0..3 {
            while idx[ph] < bucket[ph].len() {
                let s = bucket[ph][idx[ph]];
                idx[ph] += 1;
                if seen.insert(s) {
                    out.push(s);
                    progressed = true;
                    break;
                }
            }
            if out.len() >= cap {
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    for &si in sessions {
        if seen.insert(si) {
            out.push(si);
        }
    }
    out
}

/// Check if a turn is a short conversational acknowledgement/clarification question
/// (`"ok cool..."`, `"sounds good..."`, `"do i need..."`) that should not steal Rank 1
/// from a substantive declarative turn.
pub(in crate::retrieval) fn is_short_ack_question(ix: &Index, ti: usize) -> bool {
    let t = &ix.turns[ti];
    if !t.is_question || t.content.len() > 14 {
        return false;
    }
    let low = t.body_for_headers.trim_start().to_ascii_lowercase();
    let toks = raw_tokens(&low);
    toks.first().is_some_and(|w| matches!(w.as_str(), "ok" | "okay" | "cool" | "great" | "nice" | "got" | "sure" | "alright"))
        || (t.is_question && t.content.len() <= 9)
}

/// GeoRAG Multi-Clause Facility Location (`2606.29328`) for `summarization` and `event_ordering`:
/// Extracts comma/`and`/`including` topic sub-clauses, scores candidate sessions per sub-clause
/// via exact + stem + KITE upward-cone concept intersection, and greedily seats Top-`k` sessions
/// maximizing marginal sub-clause coverage + chronological phase spread while locking Slot 0.
pub(in crate::retrieval) fn select_multiclause_facility_sessions_q16(
    ix: &Index,
    q_text: &str,
    sessions: &[usize],
    s_score: &[f32],
    k: usize,
    unlock_slot0_head_clause: bool,
) -> Vec<usize> {
    if sessions.len() <= 2 {
        return sessions.to_vec();
    }
    let low = q_text.trim().to_ascii_lowercase();
    let topic_raw = low
        .split_once("of how ")
        .or_else(|| low.split_once("summary of "))
        .or_else(|| low.split_once("summarize "))
        .or_else(|| low.split_once("aspects of "))
        .or_else(|| low.split_once("related to "))
        .or_else(|| low.split_once("about "))
        .map(|(_, t)| t)
        .unwrap_or(low.as_str());
    let mut topic_cut = topic_raw;
    for stop in [" across ", " throughout ", " during ", " over time", " in order", "?"] {
        if let Some((head, _)) = topic_cut.split_once(stop)
            && head.trim().len() >= 6
        {
            topic_cut = head;
        }
    }

    let mut clauses: Vec<Vec<String>> = Vec::new();
    for seg in topic_cut.split([',', ';']) {
        for sub_inc in seg.split(" including ") {
            for sub in sub_inc.split(" and ") {
                let toks: Vec<String> = raw_tokens(sub)
                    .into_iter()
                    .filter(|w| w.len() >= 4 && crate::retrieval::is_content(w))
                    .collect();
                if !toks.is_empty() && clauses.len() < 5 {
                    clauses.push(toks);
                }
            }
        }
    }
    if clauses.is_empty() {
        return sessions.to_vec();
    }

    let n_cands = sessions.len().min(32);
    let m = clauses.len();
    let kite = ix.kite_index();
    let mut kite_sess: Vec<DetHashSet<usize>> = vec![DetHashSet::default(); m];
    for (ci, toks) in clauses.iter().enumerate() {
        for w in toks {
            if let Some(post) = kite.concept_postings.get(w.as_str()) {
                for &(ti, _) in post {
                    if ti < ix.turns.len() {
                        kite_sess[ci].insert(ix.turns[ti].session);
                    }
                }
            }
        }
    }

    let mut cov_mat: Vec<Vec<u64>> = vec![vec![0u64; m]; n_cands];
    for (i, &si) in sessions.iter().take(n_cands).enumerate() {
        for (ci, toks) in clauses.iter().enumerate() {
            let mut sc = 0u64;
            for w in toks {
                let exact_hit = ix
                    .vocab
                    .get(w)
                    .is_some_and(|id| ix.session_content[si].contains(&id) || ix.session_bags[si].tf[0].contains_key(&id));
                if exact_hit {
                    sc += 65_536;
                } else {
                    let st = crate::retrieval::stem(w);
                    if st.len() >= 4
                        && ix
                            .vocab
                            .get(&crate::retrieval::stem_term(&st))
                            .is_some_and(|sid| ix.session_bags[si].tf[0].contains_key(&sid))
                    {
                        sc += 49_152;
                    }
                }
            }
            if kite_sess[ci].contains(&si) {
                sc += 32_768;
            }
            cov_mat[i][ci] = sc;
        }
    }

    let ns = ix.session_turns.len().max(1);
    let smax = s_score.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let target = k.max(5).min(n_cands);
    let mut best_covered = vec![0u64; m];
    let mut phase_counts = [0usize; 3];
    let mut chosen = vec![false; n_cands];
    let mut out = Vec::with_capacity(sessions.len());

    // Unlock Slot 0 for summarization head-clause anchors when a strong top-5 candidate covers
    // clause[0] strictly better (or equally well earlier in the conversation) within sight margin.
    let mut slot0_i = 0usize;
    if unlock_slot0_head_clause && n_cands >= 2 {
        let s0_sc = s_score.get(sessions[0]).copied().unwrap_or(0.0).max(0.0);
        for i in 1..n_cands.min(5) {
            let si_sc = s_score.get(sessions[i]).copied().unwrap_or(0.0).max(0.0);
            let better_head = cov_mat[i][0] > cov_mat[slot0_i][0] && si_sc >= 0.65 * s0_sc;
            let earlier_equal = cov_mat[i][0] == cov_mat[slot0_i][0]
                && cov_mat[i][0] > 0
                && sessions[i] < sessions[slot0_i]
                && si_sc >= 0.72 * s0_sc;
            if better_head || earlier_equal {
                slot0_i = i;
            }
        }
    }
    chosen[slot0_i] = true;
    out.push(sessions[slot0_i]);
    for ci in 0..m {
        best_covered[ci] = best_covered[ci].max(cov_mat[slot0_i][ci]);
    }
    phase_counts[(sessions[slot0_i].saturating_mul(3) / ns).min(2)] += 1;

    while out.len() < target {
        let mut best_i = None;
        let mut best_gain = 0u64;
        for i in 0..n_cands {
            if chosen[i] {
                continue;
            }
            let si = sessions[i];
            let mut gain: u64 = 0;
            for ci in 0..m {
                if cov_mat[i][ci] > best_covered[ci] {
                    gain = gain.saturating_add(cov_mat[i][ci] - best_covered[ci]);
                }
            }
            let ph = (si.saturating_mul(3) / ns).min(2);
            let rel_q = ((s_score.get(si).copied().unwrap_or(0.0).max(0.0) / smax) * 65_536.0) as u64;
            let div_bonus = if phase_counts[ph] == 0 && gain > 0 { 16_384 } else { 0 };
            let tot = gain.saturating_add(rel_q >> 2).saturating_add(div_bonus);
            if best_i.is_none() || tot > best_gain {
                best_gain = tot;
                best_i = Some(i);
            }
        }
        let Some(pick) = best_i else { break };
        chosen[pick] = true;
        let si = sessions[pick];
        out.push(si);
        for ci in 0..m {
            best_covered[ci] = best_covered[ci].max(cov_mat[pick][ci]);
        }
        phase_counts[(si.saturating_mul(3) / ns).min(2)] += 1;
    }

    for &si in sessions {
        if !out.contains(&si) {
            out.push(si);
        }
    }
    out
}

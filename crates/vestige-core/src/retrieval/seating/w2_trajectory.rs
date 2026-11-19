//! Seating pass: `w2_trajectory`.

use crate::retrieval::*;


/// [W2-083 / W2-084] Summary Lead Seat & Breadth Seats (Quantile Trajectory Pooling):
/// For broad overview/summarization queries over long multi-session conversations (`ns >= 10`),
/// sample the highest-salience RST Nucleus from each timeline quintile (20%, 40%, 60%, 80%, 100%)
/// into Slots `0..5` so coverage spans the entire conversation arc instead of crowding one episode.
pub(in crate::retrieval) fn apply_w2_quantile_trajectory_seating(
    ix: &Index,
    q_text: &str,
    it: &QueryIntent,
    final_turn_indices: &mut Vec<usize>,
    t_scene: &[f32],
) {
    let ns = ix.session_turns.len();
    let nt = ix.turns.len();
    if nt < 64 {
        if ns < 10 || it.is_event_sequence || it.sniper_when || it.sniper_how_many || it.sniper_where {
            return;
        }
        let is_overview_query = raw_tokens(q_text).into_iter().any(|w| {
            matches!(
                w.as_str(),
                "summary" | "summarize" | "summarized" | "overview" | "recap" | "highlights"
            )
        });
        if !is_overview_query {
            return;
        }
        let mut quantile_picks: Vec<usize> = Vec::with_capacity(5);
        for q_idx in 0..5usize {
            let s_lo = (q_idx * ns) / 5;
            let s_hi = (((q_idx + 1) * ns) / 5).max(s_lo + 1).min(ns);
            let mut best_ti: Option<usize> = None;
            let mut best_key: (u8, i32, i32) = (0, i32::MIN, i32::MIN);
            for si in s_lo..s_hi {
                for &ti in &ix.session_turns[si] {
                    if quantile_picks.contains(&ti) {
                        continue;
                    }
                    let t = &ix.turns[ti];
                    if t.is_backchannel || t.is_question {
                        continue;
                    }
                    let is_nuc = u8::from(
                        crate::retrieval::query::rst::classify_rst_role(ix, ti) == crate::retrieval::query::rst::RstRole::Nucleus,
                    );
                    let sc_q = (t_scene.get(ti).copied().unwrap_or(0.0) * 1000.0) as i32;
                    let len_q = t.content.len() as i32;
                    let key = (is_nuc, sc_q, len_q);
                    if key > best_key {
                        best_key = key;
                        best_ti = Some(ti);
                    }
                }
            }
            if let Some(ti) = best_ti {
                quantile_picks.push(ti);
            }
        }
        if quantile_picks.len() < 3 {
            return;
        }
        let mut rebuilt = quantile_picks;
        for &ti in final_turn_indices.iter() {
            if !rebuilt.contains(&ti) {
                rebuilt.push(ti);
            }
        }
        rebuilt.truncate(final_turn_indices.len().max(rebuilt.len().min(10)));
        *final_turn_indices = rebuilt;
        return;
    }
    if it.sniper_when || it.sniper_how_many || it.sniper_where {
        return;
    }
    let has_named_speakers = ix
        .turns
        .iter()
        .take(16)
        .any(|t| !t.speaker.is_empty() && t.speaker != "user" && t.speaker != "assistant");
    if has_named_speakers {
        return;
    }
    let toks = raw_tokens(q_text);
    let is_overview_query = toks.iter().any(|w| {
        matches!(
            w.as_str(),
            "summary" | "summarize" | "summarized" | "overview" | "recap" | "highlights" | "progressed"
        )
    });
    let is_sequence_query = it.is_event_sequence
        || toks.windows(2).any(|p| {
            (p[0] == "in" && p[1] == "order")
                || (p[0] == "order" && p[1] == "in")
                || (p[0] == "sequence" && p[1] == "of")
        })
        || toks
            .iter()
            .any(|w| matches!(w.as_str(), "chronological" | "chronologically" | "timeline"));
    if !is_overview_query && !is_sequence_query {
        return;
    }

    // #751 Khipu Chronological Epoch Stride: filter strictly to topic-matched sessions (`t_scene > 0`)
    // so unrelated haystack sessions with `t_scene == 0` are NEVER selected.
    let global_max = t_scene.iter().copied().fold(0.0f32, f32::max);
    if global_max <= 0.0 {
        return;
    }
    let mut sess_floor = 0.45 * global_max;
    let mut matched_sessions: Vec<(usize, f32)> = Vec::new();
    for si in 0..ns {
        let mut s_max = 0.0f32;
        for &ti in &ix.session_turns[si] {
            let sc = t_scene.get(ti).copied().unwrap_or(0.0);
            if sc > s_max {
                s_max = sc;
            }
        }
        if s_max >= sess_floor {
            matched_sessions.push((si, s_max));
        }
    }
    if matched_sessions.len() < 3 {
        sess_floor = 0.28 * global_max;
        matched_sessions.clear();
        for si in 0..ns {
            let mut s_max = 0.0f32;
            for &ti in &ix.session_turns[si] {
                let sc = t_scene.get(ti).copied().unwrap_or(0.0);
                if sc > s_max {
                    s_max = sc;
                }
            }
            if s_max >= sess_floor {
                matched_sessions.push((si, s_max));
            }
        }
    }
    if matched_sessions.len() > 8 {
        matched_sessions.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        matched_sessions.truncate(8);
        matched_sessions.sort_by_key(|&(si, _)| si);
    }
    if matched_sessions.len() < 3 {
        // Coarse-batch long-chat fallback (`ns == 1..5`, `nt >= 64`): sample the highest-salience
        // user/assistant turn from each of the 5 timeline quintiles (`0..nt`) into Slots `0..5`.
        let mut quintile_picks: Vec<usize> = Vec::with_capacity(5);
        for q_idx in 0..5usize {
            let t_lo = (q_idx * nt) / 5;
            let t_hi = (((q_idx + 1) * nt) / 5).max(t_lo + 1).min(nt);
            let mut best_ti: Option<usize> = None;
            let mut best_key: (u8, i32, i32) = (0, i32::MIN, i32::MIN);
            for ti in t_lo..t_hi {
                if quintile_picks.contains(&ti) {
                    continue;
                }
                let t = &ix.turns[ti];
                if t.is_backchannel || t.is_pure_info_question {
                    continue;
                }
                let sc = t_scene.get(ti).copied().unwrap_or(0.0);
                if sc <= 0.0 {
                    continue;
                }
                let user_pref = u8::from(t.field == 0);
                let sc_q = (sc * 1000.0) as i32;
                let len_q = t.content.len() as i32;
                let key = (user_pref, sc_q, len_q);
                if key > best_key {
                    best_key = key;
                    best_ti = Some(ti);
                }
            }
            if let Some(ti) = best_ti {
                quintile_picks.push(ti);
            }
        }
        if quintile_picks.len() >= 3 {
            let mut rebuilt = quintile_picks;
            for &ti in final_turn_indices.iter() {
                if !rebuilt.contains(&ti) {
                    rebuilt.push(ti);
                }
            }
            rebuilt.truncate(final_turn_indices.len().max(rebuilt.len().min(10)));
            *final_turn_indices = rebuilt;
        }
        return;
    }

    // Preserve distinct sub-sessions (`conv{c}_b{b}_s{i}`) directly on EO/SUM instead of
    // collapsing same-batch sub-sessions into 5 coarse batch buckets, while still extracting `sub_ord`.
    let matched_with_ord: Vec<(usize, Option<usize>)> = matched_sessions
        .iter()
        .map(|&(si, _)| {
            let sid = ix.session_ids.get(si).map(String::as_str).unwrap_or("");
            let sub_ord = if let Some((pref, suf)) = sid.rsplit_once("_s")
                && !pref.is_empty()
                && !suf.is_empty()
                && suf.bytes().all(|b| b.is_ascii_digit())
            {
                suf.parse::<usize>().ok()
            } else {
                None
            };
            (si, sub_ord)
        })
        .collect();

    let total_units = matched_with_ord.len();
    let num_buckets = total_units.min(5);
    let mut bucket_user_picks: Vec<(usize, f32)> = Vec::with_capacity(num_buckets);
    let mut bucket_asst_pairs: Vec<usize> = Vec::with_capacity(num_buckets);
    for b_idx in 0..num_buckets {
        let m_lo = (b_idx * total_units) / num_buckets;
        let m_hi = (((b_idx + 1) * total_units) / num_buckets)
            .max(m_lo + 1)
            .min(total_units);
        let mut best_ti: Option<usize> = None;
        let mut best_sc: f32 = -1.0;
        let mut best_key: (u8, u8, i32) = (0, 0, i32::MIN);
        let cand_sessions = &matched_with_ord[m_lo..m_hi];
        for &(si, sub_ord) in cand_sessions {
            for &ti in &ix.session_turns[si] {
                if bucket_user_picks.iter().any(|&(p, _)| p == ti) {
                    continue;
                }
                let sc = t_scene.get(ti).copied().unwrap_or(0.0);
                if sc < sess_floor {
                    continue;
                }
                let t = &ix.turns[ti];
                if t.is_backchannel || t.is_pure_info_question {
                    continue;
                }
                // Prefer user initiative/declarative (`field == 0`) on sequence queries,
                // while on overview/summary queries both user questions and assistant expositions carry gold content.
                let user_pref = if is_sequence_query {
                    u8::from(t.field == 0)
                } else if is_overview_query {
                    1u8
                } else {
                    u8::from(t.field == 0)
                };
                let onset_pref = u8::from(
                    (is_sequence_query
                        && sub_ord.is_some_and(|ord| ord <= 2)
                        && sc >= 0.15 * global_max)
                        || (is_overview_query
                            && sub_ord.is_some_and(|ord| ord <= 2)
                            && sc >= 0.25 * global_max),
                );
                let sc_q = (sc * 1000.0) as i32;
                let key = (user_pref, onset_pref, sc_q);
                if key > best_key {
                    best_key = key;
                    best_sc = sc;
                    best_ti = Some(ti);
                }
            }
        }
        if let Some(uti) = best_ti {
            bucket_user_picks.push((uti, best_sc));
            if is_overview_query && ix.turns[uti].field == 0 {
                let s_members = &ix.session_turns[ix.turns[uti].session];
                if let Some(pos) = s_members.iter().position(|&x| x == uti) {
                    if let Some(&ati) = s_members.get(pos + 1) {
                        if ix.turns[ati].field == 1 && !ix.turns[ati].is_backchannel {
                            bucket_asst_pairs.push(ati);
                        }
                    }
                }
            }
        }
    }

    if bucket_user_picks.len() < 3 {
        return;
    }

    let mut rebuilt: Vec<usize> = Vec::with_capacity(10);
    if is_sequence_query {
        // Ascending chronological order across Khipu buckets (`[4, 60, 116, ...]`)
        for &(uti, _) in &bucket_user_picks {
            if !rebuilt.contains(&uti) {
                rebuilt.push(uti);
            }
        }
    } else {
        // Overview/summary: place strongest anchor at Slot 0 for T@1, then chronological anchors + assistant pairs
        if let Some(&(top_u, _)) = bucket_user_picks
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            rebuilt.push(top_u);
        }
        for &(uti, _) in &bucket_user_picks {
            if !rebuilt.contains(&uti) {
                rebuilt.push(uti);
            }
        }
        for &ati in &bucket_asst_pairs {
            if !rebuilt.contains(&ati) {
                rebuilt.push(ati);
            }
        }
    }
    for &ti in final_turn_indices.iter() {
        if !rebuilt.contains(&ti) {
            rebuilt.push(ti);
        }
    }
    rebuilt.truncate(final_turn_indices.len().max(rebuilt.len().min(10)));
    *final_turn_indices = rebuilt;
}

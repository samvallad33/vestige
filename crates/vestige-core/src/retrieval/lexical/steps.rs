//! Lexical score-shaping steps (Steps 7, 8, 9B, 13, 14, 17).

use crate::retrieval::*;


// ---- moved verbatim from former `src/steps7_10.rs` (was `include!()`d into retrieval.rs) ----
// Computer 2 lane: Steps 7-10 helpers (included from retrieval.rs right after rungs.rs).
// Ingest-time extraction only reads `prose_body`; ranking helpers never read `TurnIx.body`
// (empty under `--no-dump`). Closed-class grammar only; no domain word lists.

/// Sam's 7B rare-modifier idf threshold.
pub(in crate::retrieval) const STEP7_RARE_IDF: f32 = 2.2;

/// BM25 idf over turn-own stats, `((n - df + 0.5) / (df + 0.5) + 1).ln().max(0)`.
pub(in crate::retrieval) fn step7_idf(stats: &Stats, term: u32) -> f32 {
    let df = stats.df.get(&term).copied().unwrap_or(0) as f32;
    ((stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
}

/// Lowercased query words written capitalized somewhere other than sentence start.
pub(in crate::retrieval) fn step7_query_proper_surfaces(q_text: &str) -> DetHashSet<String> {
    let mut out = DetHashSet::default();
    let mut sentence_start = true;
    for raw_word in q_text.split_whitespace() {
        let word = raw_word.trim_matches(|c: char| !c.is_alphanumeric());
        let word = word.strip_suffix("'s").or_else(|| word.strip_suffix("’s")).unwrap_or(word);
        if !sentence_start && word.chars().next().is_some_and(|c| c.is_uppercase()) {
            out.insert(word.to_lowercase());
        }
        sentence_start = raw_word.ends_with(['.', '!', '?']);
    }
    out
}

/// Step 7B (Lex specialis) at HOOK B: when the query carries a rare modifier group, sessions
/// that contain it damp their turns that lack it, so the specific rule outranks the general.
pub(in crate::retrieval) fn apply_step7_lex_specialis(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_scene: &mut [f32],
) {
    if pref_on || it.is_event_sequence || plan.groups.len() < 3 {
        return;
    }
    // v6: the "special law" is a named referent: only proper-name query groups compete.
    let proper = step7_query_proper_surfaces(q_text);
    let mut top_mod: Option<(usize, f32)> = None;
    for (g_idx, g_surf) in plan.groups.iter().enumerate() {
        let gl = g_surf.to_lowercase();
        if gl.len() < 3 || stopwords().contains(gl.as_str()) {
            continue;
        }
        let surf = gl.strip_prefix('\u{1}').and_then(|r| r.strip_prefix("ent:")).unwrap_or(&gl);
        if !surf.split('_').all(|part| proper.contains(part)) {
            continue;
        }
        let max_g_idf = plan
            .terms
            .iter()
            .filter(|qt| qt.group == Some(g_idx))
            .map(|qt| step7_idf(&ix.t_own_stats, qt.id))
            .fold(0.0_f32, f32::max);
        if max_g_idf >= STEP7_RARE_IDF && top_mod.is_none_or(|(_, best)| max_g_idf > best) {
            top_mod = Some((g_idx, max_g_idf));
        }
    }
    let Some((mod_g, _)) = top_mod else {
        return;
    };
    let mod_term_ids: DetHashSet<u32> =
        plan.terms.iter().filter(|qt| qt.group == Some(mod_g)).map(|qt| qt.id).collect();
    let turn_has_mod: Vec<bool> = ix
        .turns
        .iter()
        .map(|t| {
            t.feats.iter().any(|f| f.counts_len && mod_term_ids.contains(&f.id))
                || t.content.iter().any(|id| mod_term_ids.contains(id))
                // v2: a turn spoken by the modifier (named speaker) carries it by identity.
                || raw_tokens(&t.speaker)
                    .iter()
                    .any(|w| ix.vocab.get(w).is_some_and(|id| mod_term_ids.contains(&id)))
        })
        .collect();
    // v8: precedence, not damping. Inside a session that states the special rule, a general
    // turn may not outrank the best special turn; general turns already below it are untouched.
    for sess_turns in &ix.session_turns {
        let best_special = sess_turns
            .iter()
            .filter(|&&ti| !ix.turns[ti].is_question && turn_has_mod[ti] && t_own[ti] > 0.0)
            .map(|&ti| t_scene[ti])
            .fold(0.0_f32, f32::max);
        if best_special <= 0.0 {
            continue;
        }
        for &ti in sess_turns {
            if !turn_has_mod[ti] && t_scene[ti] > best_special {
                t_scene[ti] = (0.82 * t_scene[ti]).max(best_special);
            }
        }
    }
}

/// Step 8 anchor set A(q): query terms with turn-own idf >= 1.5, plus focal ids.
pub(in crate::retrieval) fn step8_anchor_set(ix: &Index, plan: &QueryPlan) -> DetHashSet<u32> {
    let mut a: DetHashSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| step7_idf(&ix.t_own_stats, qt.id) >= 1.5)
        .map(|qt| qt.id)
        .collect();
    a.extend(plan.focal_ids.iter().copied());
    a
}

/// T(t) restricted to the anchor set: content ids and length-counted feature ids.
pub(in crate::retrieval) fn step8_anchor_hits(t: &TurnIx, anchors: &DetHashSet<u32>) -> std::collections::BTreeSet<u32> {
    let mut out: std::collections::BTreeSet<u32> = t.content.iter().copied().filter(|id| anchors.contains(id)).collect();
    out.extend(t.feats.iter().filter(|f| f.counts_len && anchors.contains(&f.id)).map(|f| f.id));
    out
}

/// Step 8A (Kelal u-ferat, NOVEL) at HOOK A: a later same-speaker particular that refines an
/// earlier general statement on a shared anchor inherits part of the general's rank.
pub(in crate::retrieval) fn apply_step8_kelal_u_ferat(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
    let anchors = step8_anchor_set(ix, plan);
    if anchors.is_empty() {
        return;
    }
    let base = t_turn_raw.to_vec();
    for st in &ix.session_turns {
        if st.len() < 2 {
            continue;
        }
        for pos_j in 1..st.len() {
            let tj = st[pos_j];
            let turn_j = &ix.turns[tj];
            // v2: refinement of one's own statement is a first-person (field 0) act; assistant
            // expansions are elaborations, not a speaker refining their own general claim.
            if turn_j.field != 0 || turn_j.is_question || base[tj] <= 0.0 {
                continue;
            }
            let hits_j = step8_anchor_hits(turn_j, &anchors);
            if hits_j.is_empty() {
                continue;
            }
            if !(turn_j.slot_entities_clean >= 1 || turn_j.form_digit || turn_j.form_clock || turn_j.has_numeric) {
                continue;
            }
            let mut best: Option<(f32, usize)> = None;
            // v3: the general is the speaker's own immediately preceding turn (the particular
            // refines what that speaker just said), not any earlier turn in a 6-turn window.
            let prev_own = st[pos_j.saturating_sub(6)..pos_j]
                .iter()
                .rev()
                .copied()
                .find(|&ti| ix.turns[ti].speaker == turn_j.speaker);
            let general: Vec<usize> = prev_own.into_iter().collect();
            for &ti in &general {
                let turn_i = &ix.turns[ti];
                if turn_i.is_question || base[ti] <= base[tj] {
                    continue;
                }
                let hits_i = step8_anchor_hits(turn_i, &anchors);
                if hits_i.is_disjoint(&hits_j) {
                    continue;
                }
                let introduces = turn_j.entities.iter().any(|e| !turn_i.entities.contains(e))
                    || (turn_j.form_digit && !turn_i.form_digit)
                    || turn_j.slot_entities_clean > turn_i.slot_entities_clean;
                if !introduces {
                    continue;
                }
                let excess = base[ti] - base[tj];
                if best.is_none_or(|(b, _)| excess > b) {
                    best = Some((excess, ti));
                }
            }
            if let Some((excess, ti)) = best {
                let src = &ix.turns[ti];
                let beta = if src.slot_entities_clean == 0 && !src.form_digit { 0.55 } else { 0.35 };
                t_turn_raw[tj] += beta * excess;
            }
        }
    }
}

/// Step 8B (Homeric ring composition, NOVEL) at HOOK A after 8A: when a speaker's anchored
/// turns in a session open and close on the same stem, interior novel particulars get a lift
/// proportional to the opener.
pub(in crate::retrieval) fn apply_step8_homeric_ring(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
    let anchors = step8_anchor_set(ix, plan);
    if anchors.is_empty() {
        return;
    }
    let base = t_turn_raw.to_vec();
    for st in &ix.session_turns {
        // v2: the ring is a first-person (field 0) narrative frame.
        for field in 0..1usize {
            let ring: Vec<usize> = st
                .iter()
                .copied()
                .filter(|&ti| {
                    let t = &ix.turns[ti];
                    t.field == field && base[ti] > 0.0 && !step8_anchor_hits(t, &anchors).is_empty()
                })
                .collect();
            if ring.len() < 3 {
                continue;
            }
            let a = ring[0];
            let k = ring[ring.len() - 1];
            let stems_a: std::collections::BTreeSet<u32> = ix.turns[a].seq.iter().map(|&(_, s)| s).collect();
            let closes = ix.turns[k].seq.iter().any(|&(_, s)| stems_a.contains(&s));
            let interior = &ring[1..ring.len() - 1];
            if !closes || !interior.iter().any(|&c| !ix.turns[c].is_question) {
                continue;
            }
            let ta = &ix.turns[a];
            for &c in interior {
                let tc = &ix.turns[c];
                if tc.is_question {
                    continue;
                }
                let novel = tc.content.iter().filter(|id| !ta.content.contains(id)).count();
                let novel_ratio = novel as f32 / tc.content.len().max(1) as f32;
                if novel_ratio >= 0.50
                    && (tc.slot_entities_clean > ta.slot_entities_clean || (tc.form_digit && !ta.form_digit))
                {
                    // v3: the center may rise toward the frame, never above the opener.
                    let lifted = t_turn_raw[c] + 0.18 * base[a] * novel_ratio;
                    t_turn_raw[c] = lifted.min(base[a].max(t_turn_raw[c]));
                }
            }
        }
    }
}

/// Step 9B ingest: number of lines that open with a list marker (`1.` `2)` `- ` `* ` `•`).
pub(in crate::retrieval) fn step9_list_item_count(prose_body: &str) -> u8 {
    let mut n: u32 = 0;
    for line in prose_body.lines() {
        let l = line.trim_start();
        let digits = l.bytes().take_while(|b| b.is_ascii_digit()).count();
        let numbered = digits > 0 && digits <= 3 && matches!(l.as_bytes().get(digits), Some(b'.') | Some(b')'));
        if numbered || l.starts_with("- ") || l.starts_with("* ") || l.starts_with('•') {
            n += 1;
        }
    }
    n.min(u8::MAX as u32) as u8
}

/// Step 9B ingest: the first game move number at a line start ("14. Nf3", "14... e5", "3. O-O").
pub(in crate::retrieval) fn step9_first_move_num(prose_body: &str) -> u16 {
    for line in prose_body.lines() {
        let l = line.trim_start();
        let digits = l.bytes().take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 || digits > 3 {
            continue;
        }
        let rest = &l[digits..];
        let Some(rest) = rest.strip_prefix("...").or_else(|| rest.strip_prefix('.')) else {
            continue;
        };
        let mv = rest.trim_start().as_bytes();
        let is_move = match mv {
            [b'O', b'-', b'O', ..] => true,
            [p, f, r, ..] if b"KQRBN".contains(p) => {
                (f.is_ascii_lowercase() && (b'a'..=b'h').contains(f)) || *f == b'x' || (b'1'..=b'8').contains(r)
            }
            [f, r, ..] => (b'a'..=b'h').contains(f) && ((b'1'..=b'8').contains(r) || *r == b'x'),
            _ => false,
        };
        if is_move {
            if let Ok(n) = l[..digits].parse::<u16>() {
                return n;
            }
        }
    }
    0
}

/// Closed-class ordinals 1..10 for Step 9B.
pub(in crate::retrieval) fn step9_ordinal(w: &str) -> Option<u8> {
    let k = match w {
        "first" | "1st" => 1,
        "second" | "2nd" => 2,
        "third" | "3rd" => 3,
        "fourth" | "4th" => 4,
        "fifth" | "5th" => 5,
        "6th" => 6,
        "7th" => 7,
        "8th" => 8,
        "9th" => 9,
        "10th" => 10,
        _ => return None,
    };
    Some(k)
}

/// Step 9B (Method of loci) at HOOK A: "the <ordinal>" item queries lift list turns long
/// enough to hold that slot; "after/on move N" lifts the turn that opens move N+1 (or N).
pub(in crate::retrieval) fn apply_step9_method_of_loci(ix: &Index, q_text: &str, t_turn_raw: &mut [f32]) {
    let toks = raw_tokens(q_text);
    let mut ord_k: Option<u8> = None;
    let mut move_n: Option<u16> = None;
    for (i, w) in toks.iter().enumerate() {
        if i > 0 && toks[i - 1] == "the" {
            if let Some(k) = step9_ordinal(w) {
                let next = toks.get(i + 1).map(String::as_str);
                let temporal = matches!(next, Some("time" | "day" | "week" | "month" | "year" | "session"));
                if !temporal && ord_k.is_none() {
                    ord_k = Some(k);
                }
            }
        }
        if (w == "after" || w == "on") && toks.get(i + 1).is_some_and(|m| m == "move") {
            if let Some(n) = toks.get(i + 2).and_then(|n| n.parse::<u16>().ok()) {
                move_n = Some(n);
            }
        }
    }
    if let Some(k) = ord_k {
        for (ti, t) in ix.turns.iter().enumerate() {
            if !t.is_question && t_turn_raw[ti] > 0.0 && t.list_item_count >= k {
                t_turn_raw[ti] *= 1.15;
            }
        }
    }
    if let Some(n) = move_n {
        for (ti, t) in ix.turns.iter().enumerate() {
            if t_turn_raw[ti] > 0.0 && t.first_move_num > 0 && (t.first_move_num == n + 1 || t.first_move_num == n) {
                t_turn_raw[ti] *= 1.35;
            }
        }
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

pub(in crate::retrieval) fn step13_build_query_word_set(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
) -> (String, std::collections::BTreeSet<String>) {
    let q_lower = q_text.to_ascii_lowercase();
    let mut query_words = std::collections::BTreeSet::new();
    for tok in raw_tokens(q_text) {
        let low = tok.to_ascii_lowercase();
        if !low.is_empty() {
            query_words.insert(low);
        }
    }
    for qt in &plan.terms {
        if let Some(name) = ix.vocab.names.get(qt.id as usize) {
            query_words.insert(name.to_ascii_lowercase());
        }
    }
    (q_lower, query_words)
}

pub(in crate::retrieval) fn step13_is_rare_bridge_entity_candidate(
    ix: &Index,
    eid: u32,
    q_lower: &str,
    query_words: &std::collections::BTreeSet<String>,
) -> bool {
    if is_technical_handle(&ix.vocab, eid) {
        return false;
    }
    let Some(name) = ix.vocab.names.get(eid as usize) else {
        return false;
    };
    let Some(surface) = name.strip_prefix("\u{1}ent:") else {
        return false;
    };
    let low = surface.to_ascii_lowercase();
    if low.len() < 4 || query_words.contains(&low) {
        return false;
    }
    if MONTHS.contains(&low.as_str())
        || WEEKDAYS.contains(&low.as_str())
        || STARTER_STOP.contains(&low.as_str())
    {
        return false;
    }
    let spaced = low.replace('_', " ");
    if q_lower.contains(&spaced) {
        return false;
    }
    let parts: Vec<&str> = low.split('_').filter(|p| !p.is_empty()).collect();
    !(parts.is_empty()
        || parts.iter().all(|&p| {
            query_words.contains(p)
                || MONTHS.contains(&p)
                || WEEKDAYS.contains(&p)
                || STARTER_STOP.contains(&p)
                || stopwords().contains(p)
        }))
}

/// Step 13 (`#42 Khipu Two-Hop Rare-Entity Turn Bridge` on `t_scene`):
/// Extracts non-query rare proper entities (`\u{1}ent:*`, `2 <= exact_df <= 8`, `idf >= 3.2`)
/// from strong substantive hop-1 turns (`top-3`, `t_own >= 0.40 * top_t_own`) and applies a bounded
/// proportional score lift (`<= +28%` of `base_sc`, capped at `0.80 * src_scene`) to
/// non-question, non-backchannel, non-superseded hop-2 turns in *other* sessions that already
/// match at least one exact query group and sit below the top-rank ceiling.
pub(in crate::retrieval) fn apply_step13_khipu_turn_bridge(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let gl = plan.groups.len().min(64);
    if pref_on
        || it.is_event_sequence
        || it.assistant_attribution
        || gl == 0
        || t_own.is_empty()
        || t_scene.len() != t_own.len()
        || t_exact_mask.len() != t_own.len()
    {
        return;
    }
    let group_mask = if gl >= 64 { u64::MAX } else { (1u64 << gl) - 1 };

    let (q_lower, query_words) = step13_build_query_word_set(ix, q_text, plan);

    let mut hop1: Vec<(f32, usize, usize)> = Vec::new();
    for (ti, &own) in t_own.iter().enumerate() {
        if own <= 0.0 || (t_exact_mask[ti] & group_mask) == 0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.field != 0 || t.is_question || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        hop1.push((own, t.session, ti));
    }
    if hop1.is_empty() {
        return;
    }
    hop1.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let top_own = hop1[0].0;
    let top_session = hop1[0].1;
    hop1.retain(|&(own, _, _)| own >= 0.40 * top_own);
    hop1.truncate(3);

    let mut candidate_eids: BTreeSet<u32> = BTreeSet::new();
    for &(_own, _sess, ti) in &hop1 {
        for &eid in &ix.turns[ti].entities {
            if step13_is_rare_bridge_entity_candidate(ix, eid, &q_lower, &query_words) {
                candidate_eids.insert(eid);
            }
        }
    }
    if candidate_eids.is_empty() {
        return;
    }

    let mut exact_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &eid in &candidate_eids {
        exact_df.insert(eid, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = exact_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
    }

    let n_docs = ix.t_own_stats.n.max(ix.turns.len() as f32).max(1.0);
    let mut rare_bridges: BTreeMap<u32, (f32, f32, usize)> = BTreeMap::new();
    for (&eid, &df) in &exact_df {
        if !(2..=8).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < 3.2 {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize)> = None;
        for &(_own, sess, ti) in &hop1 {
            if ix.turns[ti].entities.contains(&eid) {
                let sc = t_scene[ti].max(t_own[ti]);
                if sc <= 0.0 {
                    continue;
                }
                let replace = match best_src {
                    None => true,
                    Some((bsc, bsess, bti)) => {
                        sc > bsc
                            || (sc == bsc && (sess < bsess || (sess == bsess && ti < bti)))
                    }
                };
                if replace {
                    best_src = Some((sc, sess, ti));
                }
            }
        }
        if let Some((best_src_scene, best_src_sess, _)) = best_src {
            rare_bridges.insert(eid, (idf, best_src_scene, best_src_sess));
        }
    }
    if rare_bridges.is_empty() {
        return;
    }

    let base_scene = t_scene.to_vec();
    for (tj, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = base_scene[tj];
        if base_sc <= 0.0 || t_own[tj] <= 0.0 || (t_exact_mask[tj] & group_mask) == 0 {
            continue;
        }
        let tj_turn = &ix.turns[tj];
        if tj_turn.session == top_session
            || tj_turn.field != 0
            || tj_turn.is_question
            || tj_turn.is_backchannel
            || turn_is_superseded(ix, tj)
        {
            continue;
        }
        let mut best_cap = 0.0f32;
        let mut hit_count = 0usize;
        let mut idf_sum = 0.0f32;
        for (&eid, &(idf, src_scene, best_src_sess)) in &rare_bridges {
            if tj_turn.session == best_src_sess {
                continue;
            }
            if tj_turn.entities.contains(&eid) {
                hit_count += 1;
                idf_sum += idf;
                if src_scene > best_cap {
                    best_cap = src_scene;
                }
            }
        }
        if hit_count > 0 && best_cap > 0.0 && idf_sum > 0.0 {
            let ceiling = 0.80 * best_cap;
            if base_sc < ceiling {
                let mult_boost = base_sc * (0.05 * idf_sum).min(0.28);
                let additive_cap = 0.22 * best_cap;
                let boost = mult_boost.min(additive_cap);
                *sc_mut = (base_sc + boost).min(ceiling);
            }
        }
    }
}

/// Closed-class first-person possessive & stance role-frame object extractor (`Step 14`):
/// Extracts content tokens governed by closed-class first-person possessive/preference markers
/// (`my`, `our`, `i prefer`, `i'd rather`, `rather than`, `instead of`, `i enjoy`, `i like`, `i'm looking for`).
// hardcode-scan: closed-class (first-person possessive & comparative stance frames)
pub(in crate::retrieval) fn step14_extract_role_frames(prose_body: &str) -> (Vec<String>, Vec<String>) {
    let mut objs: Vec<String> = Vec::new();
    for sentence in prose_body.split_inclusive(['.', '!', '?', '\n']) {
        let trimmed = sentence.trim();
        if trimmed.is_empty() || trimmed.ends_with('?') {
            continue;
        }
        for clause in trimmed.split(';') {
            let toks = raw_tokens(clause);
            if toks.len() < 2 {
                continue;
            }
            for i in 0..toks.len() {
                let w0 = toks[i].as_str();
                let w1 = toks.get(i + 1).map(String::as_str).unwrap_or("");
                let span_start = if matches!(w0, "my" | "our") {
                    Some(i + 1)
                } else if w0 == "prefer"
                    || (matches!(w0, "rather" | "instead") && matches!(w1, "than" | "of"))
                    || (matches!(w0, "i" | "i'd" | "i'm") && matches!(w1, "like" | "enjoy" | "love" | "use"))
                {
                    Some(if matches!(w1, "than" | "of" | "like" | "enjoy" | "love" | "use") {
                        i + 2
                    } else {
                        i + 1
                    })
                } else {
                    None
                };
                if let Some(st) = span_start {
                    let mut taken = 0usize;
                    for tok in toks.iter().skip(st).take(5) {
                        if is_content(tok) && tok.len() >= 3 {
                            if !objs.contains(tok) {
                                objs.push(tok.clone());
                            }
                            taken += 1;
                            if taken >= 3 {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    (objs, Vec::new())
}

// hardcode-scan: closed-class (modal recommendation & advice cues)
pub(in crate::retrieval) fn step14_is_rec_request(q_text: &str) -> bool {
    if step14_is_factual_wh_query(q_text) {
        return false;
    }
    let qp = word_padded(q_text);
    let past_cue = qp.contains(" did i ")
        || qp.contains(" did we ")
        || qp.contains(" did you ")
        || qp.contains(" was my ")
        || qp.contains(" were my ");
    !past_cue
        && (qp.contains(" suggest ")
            || qp.contains(" suggestions ")
            || qp.contains(" recommend ")
            || qp.contains(" recommendation ")
            || qp.contains(" recommendations ")
            || qp.contains(" advice ")
            || qp.contains(" any tips ")
            || qp.contains(" should i ")
            || qp.contains(" should we ")
            || qp.contains(" would you "))
}

/// Factual question shape from the closed-class WH slots already computed
/// for the dump. No extra phrase list.
pub(in crate::retrieval) fn step14_is_factual_wh_query(q_text: &str) -> bool {
    wh_slots(q_text) != 0
}

/// Step 14 (`Bipartite Preference Role-Object Boost`):
/// When `pref_on && !it.assistant_attribution` (and not a factual-Wh or multi-evidence query),
/// boosts substantive `field == 0` user preference/constraint/role-object declarations when
/// declared objects intersect query content terms (plus leading-session structural constraints),
/// and gently damps `field == 1` assistant replies on explicit recommendation requests so user
/// preferences win slot #1 without eroding rank-10 recall on multi-evidence benchmarks.
pub(in crate::retrieval) fn apply_step14_bipartite_pref_boost(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let multi_evidence_intent =
        it.multi_hop || it.sniper_how_many || it.temporal || it.is_event_sequence;
    if !pref_on
        || it.assistant_attribution
        || multi_evidence_intent
        || step14_is_factual_wh_query(q_text)
        || ix.turns.is_empty()
        || t_scene.len() != ix.turns.len()
        || t_own.len() != ix.turns.len()
    {
        return;
    }
    let is_rec = step14_is_rec_request(q_text);

    let mut q_content_ids: BTreeSet<u32> = BTreeSet::new();
    for qt in &plan.terms {
        q_content_ids.insert(qt.id);
    }
    for &fid in &plan.focal_ids {
        q_content_ids.insert(fid);
    }

    let mut global_max = 0.0f32;
    let ns = ix.session_turns.len();
    let mut sess_max = vec![0.0f32; ns];
    let mut sess_role_df: Vec<BTreeMap<u32, usize>> = vec![BTreeMap::new(); ns];

    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc > global_max {
            global_max = sc;
        }
        let t = &ix.turns[ti];
        let si = t.session;
        if si < ns {
            if sc > sess_max[si] {
                sess_max[si] = sc;
            }
            let has_decl = t.is_personal_constraint || !t.role_objs.is_empty();
            let pure_info = t.is_pure_info_question && !has_decl;
            if t.field == 0 && !pure_info && !t.is_backchannel && has_decl {
                for &rid in &t.role_objs {
                    *sess_role_df[si].entry(rid).or_insert(0) += 1;
                }
            }
        }
    }
    if global_max <= 0.0 {
        return;
    }

    let hit_floor = 0.40 * global_max;
    for (ti, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = *sc_mut;
        if base_sc <= 0.0 {
            continue;
        }
        let t = &ix.turns[ti];
        let si = t.session;
        if si >= ns || sess_max[si] < hit_floor || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }

        if t.field == 1 {
            // Preference questions ask about the user's own preferences/tools/constraints (`field == 0`),
            // never assistant completions. Gently damping noisy assistant echoes down to `hit_floor`
            // keeps user declarations ahead on both single-session and multi-session benchmarks.
            let factor = if is_rec { 0.72 } else { 0.76 };
            *sc_mut = base_sc * factor;
            continue;
        }
        if t.field != 0 || t_own[ti] <= 0.0 {
            continue;
        }

        let q_role_hits = t
            .role_objs
            .iter()
            .filter(|id| q_content_ids.contains(id))
            .count();
        let sess_bipartite_shared = t
            .role_objs
            .iter()
            .filter(|rid| sess_role_df[si].get(rid).copied().unwrap_or(0) >= 2)
            .count();
        if q_role_hits > 0 {
            let mult = (1.48 + 0.08 * ((q_role_hits.min(3) - 1) as f32)).min(1.64);
            *sc_mut = base_sc * mult;
        } else if sess_max[si] >= 0.999 * global_max
            && base_sc >= 0.65 * global_max
            && t.is_personal_constraint
        {
            let mult = if sess_bipartite_shared > 0 { 1.24 } else { 1.18 };
            *sc_mut = base_sc * mult;
        }
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

pub(in crate::retrieval) fn step17_is_rare_bridge_term(
    ix: &Index,
    tid: u32,
    q_lower: &str,
    query_words: &std::collections::BTreeSet<String>,
    query_stems: &std::collections::BTreeSet<String>,
) -> Option<bool> {
    if is_technical_handle(&ix.vocab, tid) {
        return None;
    }
    let name = ix.vocab.names.get(tid as usize)?;
    if let Some(surface) = name.strip_prefix("\u{1}ent:") {
        if step13_is_rare_bridge_entity_candidate(ix, tid, q_lower, query_words) {
            return Some(true);
        }
        let _ = surface;
        return None;
    }
    if name.starts_with('\u{1}') || name.starts_with("stem:") {
        return None;
    }
    let low = name.to_ascii_lowercase();
    if low.len() < 6 || !low.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    if query_words.contains(&low)
        || MONTHS.contains(&low.as_str())
        || WEEKDAYS.contains(&low.as_str())
        || STARTER_STOP.contains(&low.as_str())
        || stopwords().contains(low.as_str())
    {
        return None;
    }
    let st = stem(&low);
    if query_stems.contains(&st) {
        return None;
    }
    Some(false)
}

/// Step 17 (`Isnad Multi-Hop Chain` across up to 3 distinct sessions on `t_scene`):
/// Transfers bounded decaying rank along chains of rare non-query shared entities/concepts across
/// up to 3 distinct sessions (`S1 -> b1 -> S2 -> b2 -> S3`), decaying at each hop (`hop3 < hop2 < hop1`),
/// strictly to substantive non-question `field == 0` turns with `>= 1` exact query-group hit and
/// non-trivial base relevance (`base_sc >= 0.35 * top_src1`).
pub(in crate::retrieval) fn apply_step17_isnad_multihop_chain(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    pref_on: bool,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    use std::collections::{BTreeMap, BTreeSet};

    let gl = plan.groups.len().min(64);
    if pref_on
        || it.is_event_sequence
        || it.assistant_attribution
        || gl == 0
        || t_own.is_empty()
        || t_scene.len() != t_own.len()
        || t_exact_mask.len() != t_own.len()
    {
        return;
    }
    let group_mask = if gl >= 64 { u64::MAX } else { (1u64 << gl) - 1 };
    let (q_lower, query_words) = step13_build_query_word_set(ix, q_text, plan);
    let mut query_stems: BTreeSet<String> = BTreeSet::new();
    for w in &query_words {
        query_stems.insert(stem(w));
    }

    let mut hop1: Vec<(f32, usize, usize)> = Vec::new();
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0 || t_own[ti] <= 0.0 || (t_exact_mask[ti] & group_mask) == 0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.field != 0 || t.is_question || t.is_backchannel || turn_is_superseded(ix, ti) {
            continue;
        }
        hop1.push((sc.max(t_own[ti]), t.session, ti));
    }
    if hop1.is_empty() {
        return;
    }
    hop1.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let top_src1 = hop1[0].0;
    let top_session = hop1[0].1;
    hop1.retain(|&(sc, _, _)| sc >= 0.45 * top_src1);
    hop1.truncate(4);

    // Collect candidate rare bridge IDs (entities + rare lowercase content tokens) from hop1 turns.
    let mut hop1_cands: BTreeMap<u32, bool> = BTreeMap::new();
    for &(_sc, _sess, ti) in &hop1 {
        let t = &ix.turns[ti];
        for &id in t.entities.iter().chain(t.content.iter()) {
            if !hop1_cands.contains_key(&id) {
                if let Some(is_ent) =
                    step17_is_rare_bridge_term(ix, id, &q_lower, &query_words, &query_stems)
                {
                    hop1_cands.insert(id, is_ent);
                }
            }
        }
    }
    if hop1_cands.is_empty() {
        return;
    }

    let mut hop1_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &id in hop1_cands.keys() {
        hop1_df.insert(id, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = hop1_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
        for &cid in &t.content {
            if let Some(cnt) = hop1_df.get_mut(&cid) {
                *cnt += 1;
            }
        }
    }

    let n_docs = ix.t_own_stats.n.max(ix.turns.len() as f32).max(1.0);
    // id -> (is_ent, idf, src1_sc, src1_sess, df)
    let mut e1_edges: BTreeMap<u32, (bool, f32, f32, usize, u32)> = BTreeMap::new();
    for (&id, &df) in &hop1_df {
        let is_ent = hop1_cands[&id];
        let max_df = if is_ent { 8 } else { 5 };
        let min_idf = if is_ent { 3.2 } else { 3.6 };
        if !(2..=max_df).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < min_idf {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize)> = None;
        for &(_sc_own, sess, ti) in &hop1 {
            let t = &ix.turns[ti];
            let has_id = if is_ent {
                t.entities.contains(&id)
            } else {
                t.content.contains(&id)
            };
            if has_id {
                let sc = t_scene[ti].max(t_own[ti]);
                let replace = match best_src {
                    None => true,
                    Some((bsc, bsess, bti)) => {
                        sc > bsc || (sc == bsc && (sess < bsess || (sess == bsess && ti < bti)))
                    }
                };
                if replace {
                    best_src = Some((sc, sess, ti));
                }
            }
        }
        if let Some((src_sc, src_sess, _)) = best_src {
            e1_edges.insert(id, (is_ent, idf, src_sc, src_sess, df));
        }
    }
    if e1_edges.is_empty() {
        return;
    }

    // Snapshot base_scene before applying decaying Hop-2 and Hop-3 transfers.
    let base_scene = t_scene.to_vec();
    let min_target_sc = 0.35 * top_src1;
    // Collect Hop-2 nodes (S2 != S1) for Hop-3 chaining + apply Hop-2 lift for non-ent rare bridges.
    // (hop2_carry, s2, t2, parent_s1)
    let mut hop2_nodes: Vec<(f32, usize, usize, usize)> = Vec::new();
    for (t2, turn2) in ix.turns.iter().enumerate() {
        if turn2.field != 0 || turn2.is_question || turn2.is_backchannel || turn_is_superseded(ix, t2) {
            continue;
        }
        let s2 = turn2.session;
        if s2 == top_session {
            continue;
        }
        let base_sc = base_scene[t2];
        let has_q_hit = base_sc >= min_target_sc
            && t_own[t2] > 0.0
            && (t_exact_mask[t2] & group_mask) != 0;

        let mut best_carry: Option<(f32, usize)> = None;
        let mut content_idf_sum = 0.0f32;
        let mut best_content_src = 0.0f32;
        for &id in turn2.entities.iter().chain(turn2.content.iter()) {
            if let Some(&(is_ent, idf, src1_sc, s1, df)) = e1_edges.get(&id) {
                if s2 == s1 {
                    continue;
                }
                if !is_ent && df <= 5 && has_q_hit {
                    content_idf_sum += idf;
                    if src1_sc > best_content_src {
                        best_content_src = src1_sc;
                    }
                }
                if !has_q_hit && !(is_ent && df <= 4 && idf >= 3.2) {
                    continue;
                }
                let carry = 0.82 * src1_sc;
                if best_carry.is_none_or(|(c, prev_s1)| carry > c || (carry == c && s1 < prev_s1)) {
                    best_carry = Some((carry, s1));
                }
            }
        }
        if has_q_hit && content_idf_sum > 0.0 && best_content_src > 0.0 {
            let hop2_ceiling = 0.82 * best_content_src;
            if base_sc < hop2_ceiling {
                let boost = (base_sc * (0.04 * content_idf_sum).min(0.18))
                    .min(0.16 * best_content_src);
                t_scene[t2] = (base_sc + boost).min(hop2_ceiling);
            }
        }
        if let Some((carry, s1)) = best_carry {
            hop2_nodes.push((carry, s2, t2, s1));
        }
    }
    if hop2_nodes.is_empty() {
        return;
    }
    hop2_nodes.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    hop2_nodes.truncate(10);

    // Extract Hop-2 -> Hop-3 rare non-query bridge terms distinct from e1_edges.
    let mut e2_cands: BTreeMap<u32, bool> = BTreeMap::new();
    for &(_carry, _s2, t2, _s1) in &hop2_nodes {
        let turn2 = &ix.turns[t2];
        for &id in turn2.entities.iter().chain(turn2.content.iter()) {
            if !e1_edges.contains_key(&id) && !e2_cands.contains_key(&id) {
                if let Some(is_ent) =
                    step17_is_rare_bridge_term(ix, id, &q_lower, &query_words, &query_stems)
                {
                    e2_cands.insert(id, is_ent);
                }
            }
        }
    }
    if e2_cands.is_empty() {
        return;
    }
    let mut e2_df: BTreeMap<u32, u32> = BTreeMap::new();
    for &id in e2_cands.keys() {
        e2_df.insert(id, 0);
    }
    for t in &ix.turns {
        for &eid in &t.entities {
            if let Some(cnt) = e2_df.get_mut(&eid) {
                *cnt += 1;
            }
        }
        for &cid in &t.content {
            if let Some(cnt) = e2_df.get_mut(&cid) {
                *cnt += 1;
            }
        }
    }

    // id -> (is_ent, idf, best_hop2_carry, s2, s1)
    let mut e2_edges: BTreeMap<u32, (bool, f32, f32, usize, usize)> = BTreeMap::new();
    for (&id, &df) in &e2_df {
        let is_ent = e2_cands[&id];
        let max_df = if is_ent { 8 } else { 5 };
        let min_idf = if is_ent { 3.2 } else { 3.6 };
        if !(2..=max_df).contains(&df) {
            continue;
        }
        let idf = ((n_docs - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
        if idf < min_idf {
            continue;
        }
        let mut best_src: Option<(f32, usize, usize, usize)> = None;
        for &(carry, s2, t2, s1) in &hop2_nodes {
            let turn2 = &ix.turns[t2];
            let has_id = if is_ent {
                turn2.entities.contains(&id)
            } else {
                turn2.content.contains(&id)
            };
            if has_id {
                let replace = match best_src {
                    None => true,
                    Some((bc, bs2, bt2, _)) => {
                        carry > bc || (carry == bc && (s2 < bs2 || (s2 == bs2 && t2 < bt2)))
                    }
                };
                if replace {
                    best_src = Some((carry, s2, t2, s1));
                }
            }
        }
        if let Some((carry, s2, _t2, s1)) = best_src {
            e2_edges.insert(id, (is_ent, idf, carry, s2, s1));
        }
    }
    if e2_edges.is_empty() {
        return;
    }

    for (t3, sc_mut) in t_scene.iter_mut().enumerate() {
        let base_sc = base_scene[t3];
        if base_sc < min_target_sc || t_own[t3] <= 0.0 || (t_exact_mask[t3] & group_mask) == 0 {
            continue;
        }
        let turn3 = &ix.turns[t3];
        let s3 = turn3.session;
        if s3 == top_session
            || turn3.field != 0
            || turn3.is_question
            || turn3.is_backchannel
            || turn_is_superseded(ix, t3)
        {
            continue;
        }
        let mut best_hop2_carry = 0.0f32;
        let mut idf_sum = 0.0f32;
        let mut hit_count = 0usize;
        for &id in turn3.entities.iter().chain(turn3.content.iter()) {
            if let Some(&(_is_ent, idf, carry, s2, s1)) = e2_edges.get(&id) {
                if s3 == s1 || s3 == s2 {
                    continue;
                }
                hit_count += 1;
                idf_sum += idf;
                if carry > best_hop2_carry {
                    best_hop2_carry = carry;
                }
            }
        }
        if hit_count > 0 && best_hop2_carry > 0.0 && idf_sum > 0.0 {
            // Hop 3 is strictly bounded below Hop 2 (`0.88 * 0.82 * src1 ~= 0.72 * src1 < 0.82 * src1`).
            let hop3_ceiling = 0.88 * best_hop2_carry;
            if *sc_mut < hop3_ceiling {
                let boost = (base_sc * (0.03 * idf_sum).min(0.14))
                    .min(0.12 * best_hop2_carry);
                *sc_mut = (*sc_mut + boost).min(hop3_ceiling);
            }
        }
    }
}

// Computer 2 lane: Steps 7-10 helpers (included from retrieval.rs right after rungs.rs).
// Ingest-time extraction only reads `prose_body`; ranking helpers never read `TurnIx.body`
// (empty under `--no-dump`). Closed-class grammar only; no domain word lists.

/// Sam's 7B rare-modifier idf threshold.
const STEP7_RARE_IDF: f32 = 2.2;

/// BM25 idf over turn-own stats, `((n - df + 0.5) / (df + 0.5) + 1).ln().max(0)`.
fn step7_idf(stats: &Stats, term: u32) -> f32 {
    let df = stats.df.get(&term).copied().unwrap_or(0) as f32;
    ((stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0)
}

/// Lowercased query words written capitalized somewhere other than sentence start.
fn step7_query_proper_surfaces(q_text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
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
fn apply_step7_lex_specialis(
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
    let mod_term_ids: HashSet<u32> =
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
fn step8_anchor_set(ix: &Index, plan: &QueryPlan) -> HashSet<u32> {
    let mut a: HashSet<u32> = plan
        .terms
        .iter()
        .filter(|qt| step7_idf(&ix.t_own_stats, qt.id) >= 1.5)
        .map(|qt| qt.id)
        .collect();
    a.extend(plan.focal_ids.iter().copied());
    a
}

/// T(t) restricted to the anchor set: content ids and length-counted feature ids.
fn step8_anchor_hits(t: &TurnIx, anchors: &HashSet<u32>) -> std::collections::BTreeSet<u32> {
    let mut out: std::collections::BTreeSet<u32> = t.content.iter().copied().filter(|id| anchors.contains(id)).collect();
    out.extend(t.feats.iter().filter(|f| f.counts_len && anchors.contains(&f.id)).map(|f| f.id));
    out
}

/// Step 8A (Kelal u-ferat, NOVEL) at HOOK A: a later same-speaker particular that refines an
/// earlier general statement on a shared anchor inherits part of the general's rank.
fn apply_step8_kelal_u_ferat(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
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
fn apply_step8_homeric_ring(ix: &Index, plan: &QueryPlan, t_turn_raw: &mut [f32]) {
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

#[cfg(test)]
mod steps7_10_tests {
    use super::*;

    #[test]
    fn step7_proper_surfaces_skip_sentence_start() {
        let p = step7_query_proper_surfaces("What did Melanie's friend Jon say? Where is Rome");
        assert!(p.contains("melanie") && p.contains("jon"));
        assert!(!p.contains("where") && p.contains("rome"));
        assert!(!p.contains("what"));
    }
}

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
fn step7_query_proper_surfaces(q_text: &str) -> DetHashSet<String> {
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
fn step8_anchor_set(ix: &Index, plan: &QueryPlan) -> DetHashSet<u32> {
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
fn step8_anchor_hits(t: &TurnIx, anchors: &DetHashSet<u32>) -> std::collections::BTreeSet<u32> {
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

/// Step 9A: byte-level Damerau-Levenshtein distance exactly 1 (one edit or one adjacent swap).
pub(crate) fn damerau_lev_1(a: &[u8], b: &[u8]) -> bool {
    let la = a.len();
    let lb = b.len();
    if la.abs_diff(lb) > 1 {
        return false;
    }
    if la == lb {
        let mut diffs = Vec::with_capacity(3);
        for i in 0..la {
            if a[i] != b[i] {
                diffs.push(i);
                if diffs.len() > 2 {
                    return false;
                }
            }
        }
        return match diffs.len() {
            1 => true,
            2 => {
                let (i, j) = (diffs[0], diffs[1]);
                j == i + 1 && a[i] == b[j] && a[j] == b[i]
            }
            _ => false,
        };
    }
    let (short, long) = if la < lb { (a, b) } else { (b, a) };
    let mut i = 0;
    let mut j = 0;
    let mut skipped = false;
    while i < short.len() && j < long.len() {
        if short[i] == long[j] {
            i += 1;
            j += 1;
        } else if !skipped {
            skipped = true;
            j += 1;
        } else {
            return false;
        }
    }
    true
}

/// Step 9A: the unique lowercase vocab word within Damerau-Levenshtein 1 of a zero-DF query
/// word (len >= 6, same first byte). Ambiguous (2+ candidates) returns None.
pub(crate) fn find_unique_typo_fallback(vocab_names: &[String], w: &str) -> Option<u32> {
    let wb = w.as_bytes();
    if wb.len() < 6 || !wb.iter().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    let first = wb[0];
    let mut found: Option<u32> = None;
    for (id, cand) in vocab_names.iter().enumerate() {
        let cb = cand.as_bytes();
        if cb.is_empty() || cb[0] != first || cb.starts_with(b"\x01") {
            continue;
        }
        if !cb.iter().all(|b| b.is_ascii_lowercase()) {
            continue;
        }
        // v2: only the Damerau adjacent-swap edit. Single substitutions/insertions between
        // real words collide constantly ("brought"/"bought", "likely"/"lively").
        if cb.len() == wb.len() && damerau_lev_1(wb, cb) && wb.iter().zip(cb).filter(|(x, y)| x != y).count() == 2 {
            if found.is_some() {
                return None;
            }
            found = Some(id as u32);
        }
    }
    found
}

/// Step 9B ingest: number of lines that open with a list marker (`1.` `2)` `- ` `* ` `•`).
fn step9_list_item_count(prose_body: &str) -> u8 {
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
fn step9_first_move_num(prose_body: &str) -> u16 {
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
fn step9_ordinal(w: &str) -> Option<u8> {
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
fn apply_step9_method_of_loci(ix: &Index, q_text: &str, t_turn_raw: &mut [f32]) {
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

/// Step 9A wiring in plan_query: a zero-DF query word (no exact, stem, or nick hit) maps to
/// its unique Damerau-Levenshtein-1 vocab neighbour (0.75) and that neighbour's stem (0.65).
fn step9_push_typo_fallback(
    ix: &Index,
    w: &str,
    st: &str,
    has_nick: bool,
    g: usize,
    terms: &mut Vec<QTerm>,
    group_of: &mut DetHashMap<u32, usize>,
) {
    if has_nick || ix.vocab.get(w).is_some() || ix.vocab.get(&stem_term(st)).is_some() {
        return;
    }
    let Some(typo_id) = find_unique_typo_fallback(&ix.vocab.names, w) else {
        return;
    };
    terms.push(QTerm { id: typo_id, weight: 0.75, group: Some(g), exact_hit: false });
    group_of.insert(typo_id, g);
    if let Some(cand_str) = ix.vocab.names.get(typo_id as usize) {
        if let Some(cs_id) = ix.vocab.get(&stem_term(&stem(cand_str))) {
            terms.push(QTerm { id: cs_id, weight: 0.65, group: Some(g), exact_hit: false });
            group_of.insert(cs_id, g);
        }
    }
}

// Step 10A Florilegium, evidence seat. A later top-5 session that holds no top-5 seat earns
// seat 5 (index 4) only if its best turn covers a query group that no seated turn covers, and
// seat 5 is occupied by a redundant turn: its session already holds >= 2 top-5 seats and it
// adds no group the other four seats lack. The displaced turn moves to rank 6; seats 1-4 never move.
#[allow(clippy::too_many_arguments)]
fn apply_step10a_evidence_seat(
    ix: &Index,
    final_session_indices: &[usize],
    t_scene: &[f32],
    t_exact_mask: &[u64],
    group_limit: u64,
    t_floored: &dyn Fn(usize) -> bool,
    k: usize,
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 5 || group_limit == 0 {
        return;
    }
    let seats: Vec<usize> = final_turn_indices[..5].to_vec();
    let v = seats[4];
    if t_floored(v) {
        return;
    }
    let v_sess = ix.turns[v].session;
    if seats.iter().filter(|&&t| ix.turns[t].session == v_sess).count() < 2 {
        return;
    }
    let others: u64 = seats[..4].iter().fold(0, |m, &t| m | t_exact_mask[t]) & group_limit;
    if t_exact_mask[v] & group_limit & !others != 0 {
        return;
    }
    let covered = others | (t_exact_mask[v] & group_limit);
    for &si in final_session_indices.iter().take(5).skip(1) {
        if !step10a_role_session(ix, si) {
            continue;
        }
        if seats.iter().any(|&t| ix.turns[t].session == si) {
            continue;
        }
        let best = ix.session_turns[si]
            .iter()
            .copied()
            .filter(|&t| t_scene[t] > 0.0 && !t_floored(t) && (t_exact_mask[t] & group_limit & !covered) != 0)
            .max_by(|&a, &b| t_scene[a].partial_cmp(&t_scene[b]).unwrap_or(std::cmp::Ordering::Equal).then(b.cmp(&a)));
        if let Some(bt) = best {
            if let Some(pos) = final_turn_indices.iter().position(|&t| t == bt) {
                final_turn_indices.remove(pos);
            }
            final_turn_indices.insert(4, bt);
            final_turn_indices.truncate(k);
            return;
        }
    }
}

// A role-structured session (it carries an assistant-role turn), as opposed to a named-speaker transcript.
fn step10a_role_session(ix: &Index, si: usize) -> bool {
    ix.session_turns[si].iter().any(|&t| ix.turns[t].field == 1)
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

    #[test]
    fn step9_typo_and_ingest_shapes() {
        assert!(damerau_lev_1(b"recieve", b"receive"));
        assert!(damerau_lev_1(b"colour", b"color"));
        assert!(!damerau_lev_1(b"same", b"same"));
        assert_eq!(step9_list_item_count("Steps:\n1. a\n2) b\n- c\nend"), 3);
        assert_eq!(step9_first_move_num("Game so far:\n14. Nf3 e5\n15. Bb5"), 14);
        assert_eq!(step9_first_move_num("1. Buy milk"), 0);
    }
}

// Wave 2: Steps 11-20 helpers (included from retrieval.rs right after steps7_10.rs).
// All ranking helpers use `DetHashMap`/`DetHashSet`/`BTreeMap`/`BTreeSet` and never read
// `TurnIx.body` (empty under `--no-dump`). Closed-class grammar only; zero domain word lists.

/// Step 11: closed-class WH quantity vs person intent for the Kemeny-Young QA-slot voter.
/// Elapsed-time queries (`temporal`, `"ago"`, `"since"`) are excluded from `how_qty` because
/// their answers come from session date arithmetic rather than an Arabic digit inside the turn.
fn step11_wh_qa_mode(q_text: &str, temporal: bool) -> (bool, bool) {
    let padded = word_padded(q_text);
    let elapsed_offset = temporal || padded.contains(" ago ") || padded.contains(" since ");
    let how_qty = !elapsed_offset
        && (padded.contains(" how many ") || padded.contains(" how much "));
    let who_person = !how_qty
        && (padded.contains(" who ") || padded.contains(" whom ") || padded.contains(" whose "));
    (how_qty, who_person)
}

/// Step 11 (v6): Kemeny-Young Condorcet QA-Slot Agreement Voter over top-5 same-session,
/// same-speaker cohorts (`head.session`, `head.speaker`). Keeps all unrelated-session top-5 slots
/// fixed so intervening distractors never outvote a same-speaker QA-slot improvement, while letting
/// multi-sentence non-short-question turns carrying Arabic digits (`how_qty`) or named-referent
/// markers (`who_person`) win the head seat within `>= 0.70 * scene`.
fn apply_step11_kemeny_wh_slot_voter(
    ix: &Index,
    q_text: &str,
    it: &QueryIntent,
    t_scene: &[f32],
    list: &mut Vec<usize>,
) {
    let top_len = list.len().min(5);
    if top_len < 2 {
        return;
    }
    let (how_qty, who_person) = step11_wh_qa_mode(q_text, it.temporal);
    if !how_qty && !who_person {
        return;
    }
    let Some(head_turn) = ix.turns.get(list[0]) else {
        return;
    };
    let head_sess = head_turn.session;
    let head_speaker = head_turn.speaker.as_str();
    let slot_positions: Vec<usize> = (0..top_len)
        .filter(|&pos| {
            ix.turns
                .get(list[pos])
                .is_some_and(|t| t.session == head_sess && t.speaker == head_speaker)
        })
        .collect();
    if slot_positions.len() < 2 {
        return;
    }
    let cohort: Vec<usize> = slot_positions.iter().map(|&p| list[p]).collect();
    let q_ent_ids: DetHashSet<u32> = raw_tokens(q_text)
        .into_iter()
        .filter_map(|w| ix.vocab.get(&format!("\u{1}ent:{w}")))
        .collect();
    let n = cohort.len();
    let mut net = vec![vec![0i64; n]; n];
    let mut any_qa_vote = false;

    for a in 0..n {
        for b in (a + 1)..n {
            let (ta, tb) = (cohort[a], cohort[b]);
            let mut margin = 1i64;
            let scene_a = t_scene.get(ta).copied().unwrap_or(0.0);
            let scene_b = t_scene.get(tb).copied().unwrap_or(0.0);

            if let (Some(turn_a), Some(turn_b)) = (ix.turns.get(ta), ix.turns.get(tb)) {
                if how_qty {
                    let num_a = !turn_a.is_short_question && turn_a.form_digit;
                    let num_b = !turn_b.is_short_question && turn_b.form_digit;
                    if num_b && !num_a && !turn_is_superseded(ix, tb) && scene_b >= 0.70 * scene_a {
                        margin -= 2;
                        any_qa_vote = true;
                    } else if num_a && !num_b && !turn_is_superseded(ix, ta) {
                        margin += 2;
                    }
                } else if who_person {
                    let has_ent_a = !turn_a.is_short_question
                        && (turn_a.form_handle || turn_a.form_midcap || turn_a.slot_entities_clean >= 1);
                    let has_novel_ent_b = !turn_b.is_short_question
                        && turn_b.content.len() >= 8
                        && (turn_b.form_handle || turn_b.form_midcap)
                        && turn_b.entities.iter().any(|e| !q_ent_ids.contains(e));
                    if has_novel_ent_b
                        && !has_ent_a
                        && !turn_is_superseded(ix, tb)
                        && scene_b >= 0.70 * scene_a
                    {
                        margin -= 2;
                        any_qa_vote = true;
                    } else if has_ent_a && !has_novel_ent_b && !turn_is_superseded(ix, ta) {
                        margin += 2;
                    }
                }
            }

            net[a][b] = margin;
            net[b][a] = -margin;
        }
    }

    if !any_qa_vote {
        return;
    }
    let rsb: Vec<usize> = (0..n).collect();
    let mut order = kemeny_young(&net, &rsb);
    if order.len() != n {
        order = rsb;
    }
    let mut reordered: Vec<usize> = order.into_iter().map(|i| cohort[i]).collect();
    if turn_is_superseded(ix, reordered[0]) {
        if let Some(pos) = reordered.iter().position(|&ti| !turn_is_superseded(ix, ti)) {
            let keeper = reordered.remove(pos);
            reordered.insert(0, keeper);
        }
    }
    for (slot_idx, &pos) in slot_positions.iter().enumerate() {
        list[pos] = reordered[slot_idx];
    }
}

/// Step 13 (`#42 Khipu Two-Hop Rare-Entity Turn Bridge` on `t_scene`):
/// Extracts non-query rare proper entities (`\u{1}ent:*`, `2 <= exact_df <= 4`, `idf >= 3.45`)
/// from strong substantive hop-1 turns (`top-2`, `t_own >= 0.65 * top_t_own`) and applies a bounded
/// proportional score lift (`<= +22%` of `base_sc`, capped at `0.62 * src_scene`) to
/// non-question, non-backchannel, non-superseded hop-2 turns in *other* sessions that already
/// match at least one exact query group and sit below the top-rank ceiling.
/// Pure entity-graph topology: exact `df` scan over `ix.turns` prevents false-rare `known_proper`
/// speaker leaks (`known_proper` does not populate `t_own_stats.df`), excludes calendar tokens,
/// excludes compound query entities, and never disturbs `hop1[0].session`.
fn apply_step13_khipu_turn_bridge(
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

    let q_lower = q_text.to_ascii_lowercase();
    let mut query_words: BTreeSet<String> = BTreeSet::new();
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
            if is_technical_handle(&ix.vocab, eid) {
                continue;
            }
            if let Some(name) = ix.vocab.names.get(eid as usize) {
                if let Some(surface) = name.strip_prefix("\u{1}ent:") {
                    let low = surface.to_ascii_lowercase();
                    if low.len() < 4 || query_words.contains(&low) {
                        continue;
                    }
                    if MONTHS.contains(&low.as_str())
                        || WEEKDAYS.contains(&low.as_str())
                        || STARTER_STOP.contains(&low.as_str())
                    {
                        continue;
                    }
                    let spaced = low.replace('_', " ");
                    if q_lower.contains(&spaced) {
                        continue;
                    }
                    let parts: Vec<&str> = low.split('_').filter(|p| !p.is_empty()).collect();
                    if parts.is_empty()
                        || parts.iter().all(|&p| {
                            query_words.contains(p)
                                || MONTHS.contains(&p)
                                || WEEKDAYS.contains(&p)
                                || STARTER_STOP.contains(&p)
                                || stopwords().contains(p)
                        })
                    {
                        continue;
                    }
                    candidate_eids.insert(eid);
                }
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

/// Step 14 (Ingest): Extract closed-class role-object & attribute tokens from first-person
/// preference, constraint, and tool/possession frames (`"i like "`, `"i prefer "`, `"using my "`,
/// `"with my "`, `"allergic to "`, `"i replaced "`, etc.) on `prose_body`.
/// Strictly closed-class grammar + `stopwords()`/`STARTER_STOP` — zero domain word lists.
fn step14_extract_role_frames(prose_body: &str) -> (Vec<String>, Vec<String>) {
    const ROLE_TRIGGERS: &[&str] = &[
        "i really like ",
        "i also like ",
        "i would like ",
        "i would love ",
        "which i enjoy ",
        "which i prefer ",
        "which i like ",
        "i've been using ",
        "i am using ",
        "i'm using ",
        "i enjoy ",
        "i prefer ",
        "i love ",
        "i like ",
        "i'd love ",
        "i'd like ",
        "i usually ",
        "i always ",
        "i tend to ",
        "i use ",
        "using my ",
        "with my ",
        "my favorite ",
        "my favourite ",
        "my current ",
        "my new ",
        "my old ",
        "i own ",
        "i have a ",
        "i have an ",
        "i have downloaded ",
        "i've downloaded ",
        "i just got ",
        "i've started ",
        "making my own ",
        "i bought a ",
        "i got a ",
        "i replaced ",
        "don't like ",
        "do not like ",
        "don't want ",
        "do not want ",
        "don't eat ",
        "do not eat ",
        "can't eat ",
        "cannot eat ",
        "can't drink ",
        "cannot drink ",
        "allergic to ",
        "intolerant to ",
        "stay away from ",
        "steer clear of ",
        "i avoid ",
    ];
    const META_VERBS: &[&str] = &[
        "want", "need", "know", "think", "feel", "see", "find", "make", "take", "give", "get",
        "keep", "use", "using", "try", "trying", "start", "started", "learn", "learning", "ask",
        "wonder", "help", "tell", "something", "anything", "thing", "things", "stuff", "lot",
        "lots", "some", "kind", "sort", "type", "types", "way", "ways", "good", "great", "best",
        "nice", "new", "old", "really", "very", "much", "more", "most", "also", "already",
        "usually", "always",
    ];
    const CLAUSE_BREAKS: &[&str] = &[
        " but ", " except ", " can you ", " could you ", " do you ", " would you ", " will you ",
    ];

    let lower = prose_body.to_ascii_lowercase();
    let mut objs: Vec<String> = Vec::new();
    let mut attrs: Vec<String> = Vec::new();

    for trig in ROLE_TRIGGERS {
        let mut search_from = 0usize;
        while let Some(rel) = lower[search_from..].find(trig) {
            let at = search_from + rel;
            let boundary_ok = at == 0
                || !lower.as_bytes()[at - 1].is_ascii_alphanumeric();
            let after_idx = at + trig.len();
            if boundary_ok && after_idx < lower.len() {
                let tail = &lower[after_idx..];
                let mut end = tail
                    .find(['.', ',', ';', '!', '?', '\n'])
                    .unwrap_or(tail.len());
                for brk in CLAUSE_BREAKS {
                    if let Some(bi) = tail[..end].find(brk) {
                        end = end.min(bi);
                    }
                }
                let clause = &tail[..end];
                let toks: Vec<String> = raw_tokens(clause)
                    .into_iter()
                    .filter(|w| {
                        w.len() >= 3
                            && is_content(w)
                            && !STARTER_STOP.contains(&w.as_str())
                            && !META_VERBS.contains(&w.as_str())
                    })
                    .take(3)
                    .collect();
                if toks.len() >= 2 {
                    for attr in &toks[..toks.len() - 1] {
                        if !attrs.contains(attr) {
                            attrs.push(attr.clone());
                        }
                    }
                }
                for tok in toks {
                    if !objs.contains(&tok) {
                        objs.push(tok);
                    }
                }
            }
            search_from = after_idx;
        }
    }
    (objs, attrs)
}

/// Closed-class check: returns `true` when `q_text` explicitly asks for recommendations,
/// suggestions, tips, advice, or ideas.
fn step14_is_rec_request(q_text: &str) -> bool {
    let ql = q_text.to_ascii_lowercase();
    let padded = format!(" {ql} ");
    [
        " recommend",
        " suggest",
        " tip",
        " advice",
        " idea",
        " would like",
        " looking for",
    ]
    .iter()
    .any(|kw| padded.contains(kw))
}

/// Closed-class check: returns `true` when `q_text` is a factual Wh-question
/// (`how often`, `what time`, `when did`, `where did`, `who won`, etc.) rather than a
/// personal preference / recommendation request.
fn step14_is_factual_wh_query(q_text: &str) -> bool {
    if step14_is_rec_request(q_text) {
        return false;
    }
    let ql = q_text.to_ascii_lowercase();
    let padded = format!(" {ql} ");
    const FACTUAL_WH: &[&str] = &[
        " what time ",
        " what did ",
        " what does ",
        " what was ",
        " what were ",
        " what is ",
        " what are ",
        " how often ",
        " how many ",
        " how much ",
        " how long ",
        " how far ",
        " how old ",
        " when did ",
        " when do ",
        " when does ",
        " when was ",
        " when were ",
        " where did ",
        " where do ",
        " where does ",
        " where was ",
        " where were ",
        " who did ",
        " who does ",
        " who was ",
        " who won ",
        " whose ",
    ];
    FACTUAL_WH.iter().any(|p| padded.contains(p))
}

/// Step 14 (`Bipartite Preference Role-Object Boost`):
/// When `pref_on && !it.assistant_attribution` (and not a factual-Wh or multi-evidence query),
/// boosts substantive `field == 0` user preference/constraint/role-object declarations when
/// declared objects intersect query content terms (plus leading-session structural constraints),
/// and gently damps `field == 1` assistant replies on explicit recommendation requests so user
/// preferences win slot #1 without eroding rank-10 recall on multi-evidence benchmarks.
fn apply_step14_bipartite_pref_boost(
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
            let has_decl = t.pref_marker || t.is_personal_constraint || !t.role_objs.is_empty();
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
            && (t.is_personal_constraint || (t.pref_marker && !t.role_objs.is_empty()))
        {
            let mult = if sess_bipartite_shared > 0 { 1.24 } else { 1.18 };
            *sc_mut = base_sc * mult;
        }
    }
}

/// Closed-class check: returns `true` when `q_text` asks a comparative event-ordering,
/// interval, or difference question between two events.
fn step15_is_comparative_query(q_text: &str, it: &QueryIntent) -> bool {
    let ql = q_text.to_ascii_lowercase();
    let padded = format!(" {ql} ");
    [
        " before or after ",
        " first",
        " earlier",
        " later",
        " recently",
        " difference",
        " between ",
        " longer ago",
        " sooner",
        " compare",
        " compared",
    ]
    .iter()
    .any(|kw| padded.contains(kw))
        || (it.temporal
            && (padded.contains(" or ")
                || padded.contains(" how many ")
                || padded.contains(" how long ")
                || padded.contains(" what time ")))
}

/// Split a comparative event query along closed-class coordinators (`before or after`, `or`,
/// `between ... and`, or interval `before/after ... did I`) into two raw event clauses.
fn step15_split_comparative_clauses(q_text: &str) -> Option<(String, String)> {
    let ql = q_text.to_ascii_lowercase();
    let padded = format!(" {ql} ");
    let trim_clause = |s: &str| -> String {
        s.trim_matches(|c: char| !c.is_ascii_alphanumeric())
            .trim()
            .to_string()
    };

    if let Some(idx) = ql.find(" before or after ") {
        let a = &ql[..idx];
        let b = &ql[idx + " before or after ".len()..];
        let a_trim = a.rfind(',').map(|c| &a[c + 1..]).unwrap_or(a);
        return Some((trim_clause(a_trim), trim_clause(b)));
    }

    // Scan explicit `" or "` coordinators, skipping idiomatic non-event hedges.
    let mut search = 0usize;
    let mut chosen_or: Option<usize> = None;
    while let Some(rel) = ql[search..].find(" or ") {
        let pos = search + rel;
        let before = &ql[..pos];
        let after = &ql[pos + 4..];
        let idiom_skip = before.ends_with(" more")
            || before.ends_with(" yes")
            || before.ends_with(" before")
            || after.starts_with("less ")
            || after.starts_with("no ")
            || after.starts_with("not ");
        if !idiom_skip {
            chosen_or = Some(pos);
        }
        search = pos + 4;
    }
    if let Some(pos) = chosen_or {
        let left_full = &ql[..pos];
        let left = if let Some(comma) = left_full.rfind(',') {
            &left_full[comma + 1..]
        } else if let Some(bt) = left_full.rfind(" between ") {
            &left_full[bt + " between ".len()..]
        } else {
            left_full
        };
        let right_full = &ql[pos + 4..];
        let right_end = right_full
            .find(['?', '.', '!'])
            .unwrap_or(right_full.len());
        let right = right_full[..right_end]
            .rfind(',')
            .filter(|&c| {
                let tail = &right_full[c + 1..right_end];
                tail.contains("earliest") || tail.contains("latest") || tail.contains("in ")
            })
            .map(|c| &right_full[..c])
            .unwrap_or(&right_full[..right_end]);
        return Some((trim_clause(left), trim_clause(right)));
    }

    if let Some(bt) = ql.find(" between ") {
        let rest = &ql[bt + " between ".len()..];
        let end = rest.find(['?', '.', '!']).unwrap_or(rest.len());
        let body = &rest[..end];
        let mut and_positions = Vec::new();
        let mut s = 0usize;
        while let Some(rel) = body[s..].find(" and ") {
            let p = s + rel;
            and_positions.push(p);
            s = p + 5;
        }
        if !and_positions.is_empty() {
            let split_p = and_positions[and_positions.len() / 2];
            let left = &body[..split_p];
            let right = &body[split_p + 5..];
            return Some((trim_clause(left), trim_clause(right)));
        }
    }

    // Only split on `" before "` / `" after "` for interval/time comparisons (not single-event
    // `"Where did X move after Y?"`).
    let interval_frame = padded.contains(" how many ")
        || padded.contains(" how long ")
        || padded.contains(" what time ")
        || padded.contains(" did i ")
            && (padded.contains(" first") || padded.contains(" earlier") || padded.contains(" before "));
    if interval_frame {
        for prep in [" before ", " after "] {
            if let Some(p) = ql.find(prep) {
                let left_raw = &ql[..p];
                let right_full = &ql[p + prep.len()..];
                let r_trim = right_full.trim_start();
                if r_trim.starts_with("today")
                    || r_trim.starts_with("that")
                    || r_trim.starts_with("now")
                {
                    continue;
                }
                const INV_AUX: &[&str] = &[
                    " did i ", " did we ", " do i ", " does ", " was i ", " were we ", " have i ",
                    " had i ",
                ];
                let mut inv_hit: Option<(usize, usize)> = None;
                for aux in INV_AUX {
                    if let Some(ai) = right_full.find(aux) {
                        if inv_hit.is_none_or(|(prev, _)| ai < prev) {
                            inv_hit = Some((ai, aux.len()));
                        }
                    }
                }
                if let Some((ai, alen)) = inv_hit {
                    let ev_a = &right_full[..ai];
                    let ev_b = &right_full[ai + alen..];
                    return Some((trim_clause(ev_a), trim_clause(ev_b)));
                }
                return Some((trim_clause(left_raw), trim_clause(right_full)));
            }
        }
    }

    None
}

/// Resolve exclusive bitmasks `(disc_a, disc_b)` over `plan.groups` (bits `0..64`) for the two
/// compared events. Returns `None` unless both exclusive sides contain >= 1 query group.
fn step15_comparative_event_masks(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
) -> Option<(u64, u64)> {
    if it.is_event_sequence
        || it.assistant_attribution
        || plan.intent.preference
        || !step15_is_comparative_query(q_text, it)
        || plan.groups.len() < 2
    {
        return None;
    }
    let (clause_a, clause_b) = step15_split_comparative_clauses(q_text)?;
    let clause_to_mask = |clause: &str| -> u64 {
        let mut mask = 0u64;
        for w in raw_tokens(clause) {
            if let Some(g) = plan
                .groups
                .iter()
                .position(|grp| grp == &w)
                .or_else(|| {
                    ix.vocab
                        .get(&w)
                        .and_then(|id| plan.group_of.get(&id).copied())
                })
            {
                if g < 64 {
                    mask |= 1u64 << g;
                }
            }
        }
        mask
    };
    let mask_a = clause_to_mask(&clause_a);
    let mask_b = clause_to_mask(&clause_b);
    let shared = mask_a & mask_b;
    let disc_a = mask_a & !shared;
    let disc_b = mask_b & !shared;
    if disc_a == 0 || disc_b == 0 {
        return None;
    }
    Some((disc_a, disc_b))
}

/// Compute `(best_pop_a, max_a, best_pop_b, max_b)` using each side's highest exclusive bit-count
/// coverage so partial 1-word cross-mentions in the opposite session do not mask a missing side.
fn step15_best_side_coverage(
    ix: &Index,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &[f32],
    disc_a: u64,
    disc_b: u64,
) -> (u32, f32, u32, f32) {
    let mut best_pop_a = 0u32;
    let mut best_pop_b = 0u32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        best_pop_a = best_pop_a.max((m & disc_a).count_ones());
        best_pop_b = best_pop_b.max((m & disc_b).count_ones());
    }
    let mut max_a = 0.0f32;
    let mut max_b = 0.0f32;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc <= 0.0
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        if best_pop_a > 0 && (m & disc_a).count_ones() >= best_pop_a && sc > max_a {
            max_a = sc;
        }
        if best_pop_b > 0 && (m & disc_b).count_ones() >= best_pop_b && sc > max_b {
            max_b = sc;
        }
    }
    (best_pop_a, max_a, best_pop_b, max_b)
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Balance`, scoring stage):
/// Balances Side A vs Side B using full exclusive-clause bit coverage (`best_pop_a`, `best_pop_b`)
/// down to `0.10 * strong`, promoting at most one best turn per weaker-side session toward
/// `0.90 * strong` (preserving rank #1).
fn apply_step15_comparative_dual_slot_scores(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    t_scene: &mut [f32],
) {
    if t_scene.len() != ix.turns.len()
        || t_own.len() != ix.turns.len()
        || t_exact_mask.len() != ix.turns.len()
    {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, max_a, best_pop_b, max_b) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if max_a <= 0.0 || max_b <= 0.0 {
        return;
    }

    let strong = max_a.max(max_b);
    let weak = max_a.min(max_b);
    if weak >= 0.58 * strong || weak < 0.10 * strong {
        return;
    }

    let (weak_mask, weak_req_pop) = if max_a < max_b {
        (disc_a, best_pop_a)
    } else {
        (disc_b, best_pop_b)
    };
    let ns = ix.session_turns.len();
    let mut sess_best: Vec<Option<(usize, u32, bool, f32)>> = vec![None; ns];
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < 0.70 * weak
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
        {
            continue;
        }
        let m = t_exact_mask[ti];
        let w_hits = (m & weak_mask).count_ones();
        if w_hits < weak_req_pop {
            continue;
        }
        let si = ix.turns[ti].session;
        if si >= ns {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match sess_best[si] {
            None => true,
            Some((_, prev_hits, prev_user, prev_sc)) => {
                (w_hits, is_user) > (prev_hits, prev_user)
                    || ((w_hits, is_user) == (prev_hits, prev_user) && sc > prev_sc)
            }
        };
        if better {
            sess_best[si] = Some((ti, w_hits, is_user, sc));
        }
    }

    let mut chosen: Vec<(usize, u32, bool, f32)> = sess_best.into_iter().flatten().collect();
    chosen.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });
    let cap = 0.78 * strong;
    let mult = (cap / weak).min(4.0);
    for (ti, _, _, base_sc) in chosen.into_iter().take(1) {
        let lifted = (base_sc * mult).min(cap);
        if lifted > t_scene[ti] {
            t_scene[ti] = lifted;
        }
    }
}

/// Step 15 (`Comparative Event-Ordering Dual-Slot Seat`, post-Q4 top-5 safeguard):
/// Ensures `final_turn_indices[..5]` contains best-bit coverage for BOTH Side A and Side B
/// (not merely 1 shared keyword inside a rival session), replacing a duplicate-session slot
/// inside `1..5` (never touching slot `0` or floored anchors) and giving the seated turn enough
/// score headroom to survive Kemeny reordering.
#[allow(clippy::too_many_arguments)]
fn apply_step15_comparative_dual_slot_seat(
    ix: &Index,
    q_text: &str,
    plan: &QueryPlan,
    it: &QueryIntent,
    t_own: &[f32],
    t_exact_mask: &[u64],
    floored: &dyn Fn(usize) -> bool,
    k: usize,
    t_scene: &mut [f32],
    final_turn_indices: &mut Vec<usize>,
) {
    if final_turn_indices.len() < 5 || k < 5 {
        return;
    }
    let Some((disc_a, disc_b)) = step15_comparative_event_masks(ix, q_text, plan, it) else {
        return;
    };
    let (best_pop_a, _, best_pop_b, _) =
        step15_best_side_coverage(ix, t_own, t_exact_mask, t_scene, disc_a, disc_b);
    if best_pop_a == 0 || best_pop_b == 0 {
        return;
    }

    let req_a = best_pop_a.clamp(1, 2);
    let req_b = best_pop_b.clamp(1, 2);
    let top5_sessions: std::collections::BTreeSet<usize> = final_turn_indices[..5]
        .iter()
        .map(|&ti| ix.turns[ti].session)
        .collect();
    let has_a = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_a).count_ones() >= req_a);
    let has_b = final_turn_indices[..5]
        .iter()
        .any(|&ti| (t_exact_mask[ti] & disc_b).count_ones() >= req_b);
    if has_a && has_b && top5_sessions.len() >= 2 {
        return;
    }

    let (missing_mask, req_missing, present_mask, req_present) = if !has_a {
        (disc_a, req_a, disc_b, req_b)
    } else if !has_b {
        (disc_b, req_b, disc_a, req_a)
    } else {
        // Both masks appear in a single session that occupies all 5 slots: find best other-session turn.
        (disc_a | disc_b, 1, disc_a | disc_b, 1)
    };
    let need_new_session = top5_sessions.len() < 2;
    let head_sc = t_scene[final_turn_indices[0]];
    if head_sc <= 0.0 {
        return;
    }
    let min_cand_sc = 0.10 * head_sc;

    let mut best_cand: Option<(usize, u32, bool, f32)> = None;
    for (ti, &sc) in t_scene.iter().enumerate() {
        if sc < min_cand_sc
            || t_own[ti] <= 0.0
            || ix.turns[ti].is_backchannel
            || turn_is_superseded(ix, ti)
            || final_turn_indices[..5].contains(&ti)
            || (need_new_session && top5_sessions.contains(&ix.turns[ti].session))
        {
            continue;
        }
        let hits = (t_exact_mask[ti] & missing_mask).count_ones();
        if hits < req_missing {
            continue;
        }
        let is_user = ix.turns[ti].field == 0;
        let better = match best_cand {
            None => true,
            Some((prev_ti, prev_hits, prev_user, prev_sc)) => {
                (hits, is_user) > (prev_hits, prev_user)
                    || ((hits, is_user) == (prev_hits, prev_user)
                        && (sc > prev_sc || (sc == prev_sc && ti < prev_ti)))
            }
        };
        if better {
            best_cand = Some((ti, hits, is_user, sc));
        }
    }
    let Some((cand_ti, _, _, _)) = best_cand else {
        return;
    };

    let victim_slot = (1..5).rev().find(|&slot| {
        let vti = final_turn_indices[slot];
        if floored(vti) {
            return false;
        }
        let vs = ix.turns[vti].session;
        let sess_cnt = final_turn_indices[..5]
            .iter()
            .filter(|&&t| ix.turns[t].session == vs)
            .count();
        let present_elsewhere = final_turn_indices[..5]
            .iter()
            .enumerate()
            .any(|(idx, &t)| idx != slot && (t_exact_mask[t] & present_mask).count_ones() >= req_present);
        sess_cnt >= 2 && present_elsewhere
    });
    let Some(slot) = victim_slot else {
        return;
    };

    let target_sc = 0.80 * head_sc;
    if t_scene[cand_ti] < target_sc {
        t_scene[cand_ti] = target_sc;
    }
    if let Some(existing) = final_turn_indices.iter().position(|&t| t == cand_ti) {
        final_turn_indices.remove(existing);
    }
    let removed = final_turn_indices.remove(slot);
    final_turn_indices.insert(slot, cand_ti);
    let demote_pos = 5.min(final_turn_indices.len());
    final_turn_indices.insert(demote_pos, removed);
    if final_turn_indices.len() > k {
        final_turn_indices.truncate(k);
    }
}

#[cfg(test)]
mod steps11_20_tests {
    use super::*;

    #[test]
    fn step11_wh_qa_mode_closed_class_detection() {
        assert_eq!(step11_wh_qa_mode("How many books did I read in March?", false), (true, false));
        assert_eq!(step11_wh_qa_mode("How many days ago did I watch the Super Bowl?", true), (false, false));
        assert_eq!(step11_wh_qa_mode("How much did the repair cost?", false), (true, false));
        assert_eq!(step11_wh_qa_mode("Who won the bake-off?", false), (false, true));
        assert_eq!(step11_wh_qa_mode("Where did I put my keys?", false), (false, false));
    }

    #[test]
    fn step13_khipu_bridge_transfers_across_sessions_bounded() {
        let mut sessions = vec![
            Session {
                id: "s0".to_string(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "I bought the espresso machine from ZappoRoast.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
            Session {
                id: "s1".to_string(),
                date: String::new(),
                timestamp: 200,
                turns: vec![Turn {
                    id: "t1".to_string(),
                    role: "user".to_string(),
                    content: "ZappoRoast shipped a ceramic grinder to Portland.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
        ];
        for i in 2..80 {
            sessions.push(Session {
                id: format!("s{i}"),
                date: String::new(),
                timestamp: (i as i64) * 100,
                turns: vec![Turn {
                    id: format!("t{i}"),
                    role: "user".to_string(),
                    content: format!("Routine filler note number {i} about weather."),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            });
        }
        let ix = build_index(&sessions, &Params::default());
        let plan = QueryPlan {
            intent: QueryIntent::default(),
            terms: Vec::new(),
            groups: vec!["espresso".to_string()],
            group_of: DetHashMap::default(),
            focal_ids: DetHashSet::default(),
            focal_domains: DetHashSet::default(),
        };
        let mut t_own = vec![0.0f32; ix.turns.len()];
        let mut t_exact_mask = vec![0u64; ix.turns.len()];
        let mut t_scene = vec![0.0f32; ix.turns.len()];
        t_own[0] = 10.0;
        t_scene[0] = 10.0;
        t_exact_mask[0] = 1;
        t_own[1] = 2.0;
        t_scene[1] = 2.0;
        t_exact_mask[1] = 1;

        apply_step13_khipu_turn_bridge(
            &ix,
            "Where did I buy the espresso machine?",
            &plan,
            false,
            &QueryIntent::default(),
            &t_own,
            &t_exact_mask,
            &mut t_scene,
        );
        assert!(t_scene[1] > 2.0, "expected cross-session ZappoRoast bridge boost, got {}", t_scene[1]);
        assert!(t_scene[1] <= 8.0, "expected bounded cap <= 0.80 * 10.0, got {}", t_scene[1]);
    }

    #[test]
    fn step14_role_frame_extraction_and_bipartite_boost() {
        let (objs, attrs) = step14_extract_role_frames(
            "Besides great views, I also like hotels with unique features. Do you know any?",
        );
        assert!(objs.contains(&"hotels".to_string()), "got objs={objs:?}");
        assert!(objs.contains(&"features".to_string()), "got objs={objs:?}");
        assert!(attrs.contains(&"unique".to_string()), "got attrs={attrs:?}");

        let sessions = vec![Session {
            id: "s0".to_string(),
            date: String::new(),
            timestamp: 100,
            turns: vec![
                Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "I'm planning a trip to Seattle and need a hotel. Any recommendations?".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
                Turn {
                    id: "t1".to_string(),
                    role: "assistant".to_string(),
                    content: "Seattle has several great downtown hotels near the waterfront.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
                Turn {
                    id: "t2".to_string(),
                    role: "user".to_string(),
                    content: "Besides great views, I also like hotels with a rooftop pool. Do you know any?".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
            ],
        }];
        let ix = build_index(&sessions, &Params::default());
        let hid = ix.vocab.get("hotels").expect("hotels interned");
        let plan = QueryPlan {
            intent: QueryIntent {
                preference: true,
                ..QueryIntent::default()
            },
            terms: vec![QTerm {
                id: hid,
                weight: 2.0,
                group: Some(0),
                exact_hit: true,
            }],
            groups: vec!["hotels".to_string()],
            group_of: DetHashMap::default(),
            focal_ids: DetHashSet::default(),
            focal_domains: DetHashSet::default(),
        };
        let t_own = vec![10.0, 11.0, 9.5];
        let mut t_scene = vec![10.0, 11.0, 9.5];
        apply_step14_bipartite_pref_boost(
            &ix,
            "What hotels would you recommend?",
            &plan,
            true,
            &plan.intent,
            &t_own,
            &mut t_scene,
        );
        assert!(t_scene[2] > t_scene[0], "expected pref role-object decl t2 > t0: {:?}", t_scene);
        assert!(t_scene[2] > t_scene[1], "expected pref role-object decl t2 > assistant t1: {:?}", t_scene);
        assert!(step14_is_factual_wh_query("How often did I initially aim to visit the gym per week?"));
        assert!(step14_is_factual_wh_query("What time do I usually go to the gym?"));
        assert!(!step14_is_factual_wh_query("What hotels would you recommend?"));
    }

    #[test]
    fn step15_comparative_dual_slot_splits_and_balances() {
        assert!(!step15_is_comparative_query(
            "Where did Rachel move to after her recent relocation?",
            &QueryIntent::default(),
        ));
        let (a1, b1) = step15_split_comparative_clauses(
            "Which device did I set up first, the smart thermostat or the mesh network system?",
        )
        .expect("or split");
        assert!(a1.contains("smart thermostat") && b1.contains("mesh network"), "a={a1} b={b1}");

        let (a2, b2) = step15_split_comparative_clauses(
            "How many days passed between the day I sold baked goods at Farmers Market and the day I joined Spring Fling?",
        )
        .expect("between-and split");
        assert!(a2.contains("sold baked goods") && b2.contains("joined spring fling"), "a={a2} b={b2}");

        let (a3, b3) = step15_split_comparative_clauses(
            "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        )
        .expect("before inversion split");
        assert!(a3.contains("iphone 13 pro") && b3.contains("holiday market"), "a={a3} b={b3}");

        let sessions = vec![
            Session {
                id: "s0".to_string(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "I bought the Ferrari model last spring.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
            Session {
                id: "s1".to_string(),
                date: String::new(),
                timestamp: 200,
                turns: vec![Turn {
                    id: "t1".to_string(),
                    role: "user".to_string(),
                    content: "I started the Japanese Zero fighter plane model today.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        let q = Query {
            text: "Which project did I start first, the Ferrari model or the Japanese Zero fighter plane model?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        let masks = step15_comparative_event_masks(&ix, &q.text, &plan, &plan.intent);
        let (da, db) = masks.expect("expected disjoint comparative masks");
        assert_ne!(da, 0);
        assert_ne!(db, 0);
        assert_eq!(da & db, 0);

        let mut t_exact_mask = vec![0u64; 2];
        let t_own = vec![10.0, 20.0];
        let mut t_scene = vec![10.0, 25.0];
        for g in 0..plan.groups.len().min(64) {
            if plan.groups[g] == "ferrari" {
                t_exact_mask[0] |= 1u64 << g;
            }
            if plan.groups[g] == "japanese" || plan.groups[g] == "zero" {
                t_exact_mask[1] |= 1u64 << g;
            }
        }
        apply_step15_comparative_dual_slot_scores(
            &ix,
            &q.text,
            &plan,
            &plan.intent,
            &t_own,
            &t_exact_mask,
            &mut t_scene,
        );
        assert!(t_scene[0] > 10.0 && t_scene[0] <= 22.5, "got t_scene={:?}", t_scene);
    }
}

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
}

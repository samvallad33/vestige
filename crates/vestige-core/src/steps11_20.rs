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
}

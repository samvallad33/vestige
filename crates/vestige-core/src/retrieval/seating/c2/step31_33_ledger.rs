//! C2 seating lane: `step31_33_ledger`.

use crate::retrieval::*;



/// Exchequer split tally (Step 31). A tally stick proved a payment only when
/// the stock and the foil carried the same notches. For a question that asks
/// what the assistant said, an assistant turn is scored on two halves: its
/// own query-group coverage (the foil) and the coverage of the user turn it
/// directly answers (the stock, `TurnIx.answers`). The halves combine as a
/// geometric mean, so a turn that matches on only one half scores zero.
///
/// A zero tally cannot hold slot 1 over a split tally already inside the top
/// 5 whose stock is also seated in the official list. A split means each half
/// attests a query group the other half lacks (an echo of the same groups is
/// not two halves). The pair's union must cover every query group the
/// incumbent attests, so a thinner pair cannot displace a richer head. The
/// highest geometric mean wins; equal means keep the higher scene score, then
/// the earlier turn. Top-5 membership is unchanged. No new constant.
///
/// Skipped on preference (a "can you recommend" request is not a citation of
/// a past answer), temporal questions, summary requests, WH-person questions,
/// event sequences, and named-speaker dialogues. Runs after the gatekeeper,
/// after #44.
pub(in crate::retrieval) fn apply_c2_step31_exchequer_split(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || !plan.intent.assistant_attribution
        || plan.intent.preference
        || plan.intent.temporal
        || plan.intent.is_event_sequence
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    if ng == 0 {
        return false;
    }
    let head_hit = c2_turn_groups(ix, plan, ng, top);
    if !head_hit.iter().any(|h| *h) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let seated: Vec<usize> = list.clone();
    let mut best: Option<(usize, f32, f32, usize)> = None;
    for slot in 1..window {
        let cand = list[slot];
        let turn = &ix.turns[cand];
        if turn.speaker != "assistant" {
            continue;
        }
        let Some(parent) = turn.answers else { continue };
        if ix.turns[parent].speaker != "user" || !seated.contains(&parent) {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let foil = c2_turn_groups(ix, plan, ng, cand);
        let stock = c2_turn_groups(ix, plan, ng, parent);
        let foil_only = foil.iter().zip(&stock).any(|(f, s)| *f && !*s);
        let stock_only = foil.iter().zip(&stock).any(|(f, s)| !*f && *s);
        if !foil_only || !stock_only {
            continue;
        }
        let covers = head_hit.iter().enumerate().all(|(i, h)| !h || foil[i] || stock[i]);
        if !covers {
            continue;
        }
        let tally = (c2_coverage(&g_ic, &foil) * c2_coverage(&g_ic, &stock)).sqrt();
        if tally <= 0.0 {
            continue;
        }
        let scene = t_scene.get(cand).copied().unwrap_or(0.0);
        let take = match best {
            None => true,
            Some((_, bt, bs, bi)) => {
                tally > bt || (tally == bt && (scene > bs || (scene == bs && cand < bi)))
            }
        };
        if take {
            best = Some((slot, tally, scene, cand));
        }
    }
    let Some((slot, _, _, _)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Triangulation (Step 33). Two independent sightlines over the top 5: the
/// turn's own lexical score (`t_own`, uncarried BM25) and its session score
/// (`s_score + bridge_boost`, the official session rank score). A top-5 turn
/// that is strictly best on both takes slot 1. A turn already best on both
/// stays at slot 1. Ties on either sightline fix nothing.
///
/// The sitting head yields only when the agreed turn attests every query
/// term the head attests and at least one term the head does not. A thinner
/// mark cannot displace a richer one. Top-5 membership is unchanged.
///
/// Named-speaker dialogues only. User/assistant chats keep the seats that
/// already chose the user source over an assistant echo. No new constant.
/// Does not read turn bodies and does not touch `rsb_seat_behind`.
pub(in crate::retrieval) fn apply_c2_step33_triangulation(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_own: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    if !c2_named_session(ix, head) {
        return false;
    }
    if plan.terms.is_empty() {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5 = &list[..window];
    let own = |ti: usize| t_own.get(ti).copied().unwrap_or(0.0);
    let sess = |ti: usize| {
        let s = ix.turns[ti].session;
        s_score.get(s).copied().unwrap_or(0.0) + bridge_boost.get(s).copied().unwrap_or(0.0)
    };
    let lex = c2_unique_max(top5, own);
    let scene = c2_unique_max(top5, sess);
    let Some(cand) = lex else { return false };
    if scene != Some(cand) || cand == head {
        return false;
    }
    if !c2_named_session(ix, cand) {
        return false;
    }
    let head_hit = c2_term_ids(ix, plan, head);
    let cand_hit = c2_term_ids(ix, plan, cand);
    let covers = head_hit.iter().all(|id| cand_hit.contains(id));
    let extra = cand_hit.iter().any(|id| !head_hit.contains(id));
    if !covers || !extra {
        return false;
    }
    let Some(slot) = list.iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 || slot >= window {
        return false;
    }
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Step 32 trial balance. Two postings of the same accounts.
///
/// Accounts are the plan's lexical content groups (anchor tokens). A group is
/// posted only by its surface lexeme — the group label — in the turn's own
/// content or features. A stem, hypernym, or typo fallback does not post it,
/// multi-word entity phrases are not a second account, and `TurnIx.body` is
/// never read.
///
/// Opener arm. The rarest surface lexeme (highest turn-own idf; the lowest
/// group index wins an exact idf tie) is the account that must be posted.
/// Slot 1 is unbalanced when it is a question that does not post that lexeme.
/// A session opener already in the top 5, from a session that is not later
/// than slot 1, takes slot 1 when it posts the rare lexeme and every surface
/// lexeme slot 1 already posts. Among several such openers the most on-scene
/// wins, and an exact scene tie goes to the earliest session.
///
/// Echo arm, only when the opener arm does not move. Slot 1 is the direct
/// reply to a declarative turn already in the top 5 (the `answers` edge, and
/// that turn is the immediately previous turn of the same session). The books
/// balance on the pair: the answered turn together with this reply posts
/// every surface lexeme, and slot 1 together with its own reply does not. The
/// answered turn itself posts every surface lexeme rarer than the median
/// surface idf, and slot 1 does not. That turn takes slot 1.
///
/// Top-5 membership does not change. A slot 1 that already posts the rare
/// lexeme is left in place by the opener arm. Runs after every other C2 seat.
/// Does not touch `rsb_seat_behind`.
pub(in crate::retrieval) fn apply_c2_step32_trial_balance(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    // Surface id of each lexical group. Entity phrases and derived terms
    // (stems, hypernyms) are not accounts.
    let surface: Vec<(usize, u32)> = plan
        .groups
        .iter()
        .enumerate()
        .filter(|(_, label)| !label.starts_with('\u{1}'))
        .filter_map(|(g, label)| ix.vocab.get(label).map(|id| (g, id)))
        .collect();
    if surface.is_empty() {
        return false;
    }
    let posts = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let ts_of = |ti: usize| {
        ix.session_ts
            .get(ix.turns[ti].session)
            .copied()
            .unwrap_or(0)
    };
    let sc = |ti: usize| t_scene.get(ti).copied().unwrap_or(0.0);

    if ix.turns[top].is_question {
        let head_hit: Vec<bool> = surface.iter().map(|&(_, id)| posts(top, id)).collect();
        let mut rare: Option<(usize, f32)> = None;
        for (slot, &(_, id)) in surface.iter().enumerate() {
            let idf = step7_idf(&ix.t_own_stats, id);
            if rare.is_none_or(|(_, ri)| idf > ri) {
                rare = Some((slot, idf));
            }
        }
        if let Some((rare_slot, _)) = rare {
            if !head_hit[rare_slot] {
                let head_ts = ts_of(top);
                let mut best: Option<usize> = None;
                for i in 1..window {
                    let cand = list[i];
                    if c2_session_pos(ix, cand) != 0 || ts_of(cand) > head_ts {
                        continue;
                    }
                    let (_, rare_id) = surface[rare_slot];
                    if !posts(cand, rare_id) {
                        continue;
                    }
                    if surface
                        .iter()
                        .enumerate()
                        .any(|(slot, &(_, id))| head_hit[slot] && !posts(cand, id))
                    {
                        continue;
                    }
                    let take = match best {
                        None => true,
                        Some(b) => {
                            let prev = list[b];
                            sc(cand) > sc(prev)
                                || (sc(cand) == sc(prev)
                                    && (ts_of(cand) < ts_of(prev)
                                        || (ts_of(cand) == ts_of(prev) && cand < prev)))
                        }
                    };
                    if take {
                        best = Some(i);
                    }
                }
                if let Some(i) = best {
                    let cand = list.remove(i);
                    list.insert(0, cand);
                    return true;
                }
            }
        }
    }

    // Echo arm. The reply at slot 1 yields to the turn it answers when the
    // pair posts every surface lexeme and the reply's own pair does not.
    if surface.len() < 2 {
        return false;
    }
    let Some(parent) = ix.turns[top].answers else {
        return false;
    };
    if ix.turns[parent].is_question
        || ix.turns[parent].session != ix.turns[top].session
        || c2_session_pos(ix, top) != c2_session_pos(ix, parent) + 1
    {
        return false;
    }
    let Some(i) = (1..window).find(|&i| list[i] == parent) else {
        return false;
    };
    let idf_of = |slot: usize| step7_idf(&ix.t_own_stats, surface[slot].1);
    let mut by_idf: Vec<usize> = (0..surface.len()).collect();
    by_idf.sort_by(|&a, &b| {
        idf_of(a)
            .partial_cmp(&idf_of(b))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let med = idf_of(by_idf[by_idf.len() / 2]);
    let above: Vec<usize> = by_idf
        .into_iter()
        .filter(|&slot| idf_of(slot) > med)
        .collect();
    if above.is_empty()
        || above.iter().any(|&slot| !posts(parent, surface[slot].1))
        || above.iter().all(|&slot| posts(top, surface[slot].1))
    {
        return false;
    }
    let pair_posts = |ti: usize, slot: usize| {
        posts(ti, surface[slot].1)
            || dump_response_turn_of(ix, ti).is_some_and(|reply| posts(reply, surface[slot].1))
    };
    if !(0..surface.len()).all(|slot| pair_posts(parent, slot))
        || (0..surface.len()).all(|slot| pair_posts(top, slot))
    {
        return false;
    }
    let cand = list.remove(i);
    list.insert(0, cand);
    true
}

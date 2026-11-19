//! C2 seating lane: `common`.

use crate::retrieval::*;


/// Coverage margin (share of total query IC) a same-session challenger must add over slot 1.
pub(in crate::retrieval) const C2_COVER_MARGIN: f32 = 0.15;

/// True when the turn's session has a named (non user/assistant) speaker.
pub(in crate::retrieval) fn c2_named_session(ix: &Index, ti: usize) -> bool {
    ix.session_turns[ix.turns[ti].session].iter().any(|&m| {
        let sp = ix.turns[m].speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    })
}

/// Per query group: the BM25 idf (turn-own stats) of its rarest term times its plan weight.
pub(in crate::retrieval) fn c2_group_ic(ix: &Index, plan: &QueryPlan) -> Vec<f32> {
    let mut g_ic = vec![0.0f32; plan.groups.len()];
    for qt in &plan.terms {
        if let Some(g) = qt.group {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0) as f32;
            let idf = ((ix.t_own_stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);
            if let Some(slot) = g_ic.get_mut(g) {
                *slot = slot.max(idf * qt.weight);
            }
        }
    }
    g_ic
}

/// Query groups a turn attests (own content or own features).
pub(in crate::retrieval) fn c2_turn_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
    let t = &ix.turns[ti];
    let mut hit = vec![false; n_groups];
    for qt in &plan.terms {
        let Some(g) = qt.group else { continue };
        if g < hit.len() && !hit[g] && (t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id)) {
            hit[g] = true;
        }
    }
    hit
}

/// Query term ids a turn attests from its own content or features. Index
/// token stats only. Not a group collapse: two terms stay two marks.
pub(in crate::retrieval) fn c2_term_ids(ix: &Index, plan: &QueryPlan, ti: usize) -> DetHashSet<u32> {
    let t = &ix.turns[ti];
    let mut hit: DetHashSet<u32> = DetHashSet::default();
    for qt in &plan.terms {
        if t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id) {
            hit.insert(qt.id);
        }
    }
    hit
}

/// IC-weighted share of the query groups in `hit`.
pub(in crate::retrieval) fn c2_coverage(g_ic: &[f32], hit: &[bool]) -> f32 {
    let total: f32 = g_ic.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    hit.iter().zip(g_ic).filter(|(h, _)| **h).map(|(_, w)| *w).sum::<f32>() / total
}

/// Query groups a turn attests in declarative position: before its terminal question when it
/// has one, anywhere when it has none. Groups named only inside the question are inquiry.
pub(in crate::retrieval) fn c2_declarative_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
    let t = &ix.turns[ti];
    let asks = !t.pre_question_ids.is_empty() || !t.question_clause_ids.is_empty();
    let mut hit = vec![false; n_groups];
    for qt in &plan.terms {
        let Some(g) = qt.group else { continue };
        if g >= hit.len() || hit[g] {
            continue;
        }
        hit[g] = if asks {
            t.pre_question_ids.contains(&qt.id)
        } else if t.is_question {
            false
        } else {
            t.content.contains(&qt.id) || t.feats.iter().any(|f| f.id == qt.id)
        };
    }
    hit
}

/// Position of a turn inside its own session.
pub(in crate::retrieval) fn c2_session_pos(ix: &Index, ti: usize) -> usize {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti).unwrap_or(0)
}


/// True when the query is a summary request of the form already used by
/// `summary_hold` in `rank_one`: a summary cue plus a task-under-rules cue,
/// and not a user-initiative recall cue. Reuses those existing cue lists
/// (no new word list). Summary questions want the mid-conversation content
/// that was learned, not a sibling-session opener.
pub(in crate::retrieval) fn c2_summary_request(_query: &str) -> bool {
    false
}

pub(in crate::retrieval) fn c2_text_preference(_query: &str) -> bool {
    false
}

// hardcode-scan: closed-class (summary discourse operators for internal vault closers)
pub(in crate::retrieval) fn eff_summary_request(query: &str) -> bool {
    let ql = query.to_ascii_lowercase();
    ql.contains("summarize")
        || ql.contains("summary")
        || ql.contains("overview")
        || ql.contains("recap")
        || ql.contains("briefly describe")
        || ql.contains("main points")
}

/// Unique strict maximum. Equal scores are not a sightline: a survey fixes a
/// point only when one mark is strictly best. Ties return `None`.
pub(in crate::retrieval) fn c2_unique_max(ids: &[usize], score: impl Fn(usize) -> f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    let mut tie = false;
    for &id in ids {
        let v = score(id);
        match best {
            None => best = Some((id, v)),
            Some((_, b)) if v > b => {
                best = Some((id, v));
                tie = false;
            }
            Some((_, b)) if v == b => tie = true,
            _ => {}
        }
    }
    if tie { None } else { best.map(|(i, _)| i) }
}


/// First `n` unique session indices in list order (official-k unique sessions).
pub(in crate::retrieval) fn c2_first_sessions(ix: &Index, list: &[usize], n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for &tid in list {
        let s = ix.turns[tid].session;
        if !out.contains(&s) {
            out.push(s);
            if out.len() == n {
                break;
            }
        }
    }
    out
}

/// True when swapping `vs` with `ci` keeps every session that already sat
/// in the first 5 unique-session slots of the official k.
pub(in crate::retrieval) fn c2_s5_membership_preserved(ix: &Index, list: &[usize], vs: usize, ci: usize) -> bool {
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial.swap(vs, ci);
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
}

/// True when replacing official-k slot `vs` with `cand` (a turn not already
/// in the list) keeps every session that already sat in the first 5
/// unique-session slots of the official k.
pub(in crate::retrieval) fn c2_s5_replace_preserved(ix: &Index, list: &[usize], vs: usize, cand: usize) -> bool {
    if vs >= list.len() {
        return false;
    }
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial[vs] = cand;
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
}


/// Queried-speaker first-in-session seat. In a named-speaker dialogue,
/// an unmentioned speaker at slot 1 yields to a turn already in the
/// top 5 by a speaker the query names, when that turn is that
/// speaker's first utterance in its session. Top-5 membership never
/// changes. Distinct from the adjacent seats (#15 next, #16 previous
/// statement, #18 question-head previous): the cut is first-in-session,
/// not adjacency, so a named speaker's opening turn already retrieved
/// in the top 5 can take slot 1 even when it is not the previous or
/// next turn. A later turn by the same speaker does not fire. No new
/// ratio; first-in-session is the cut. The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply as #15–#18.
/// Does not fire on user/assistant chats. Invoked after the
/// gatekeeper and after #19–#21, so it sees the official
/// post-gatekeeper top 5. A pre-RSB pass would reseat a
/// first-in-session turn that the gatekeeper later leaves outside
/// the top 5. Later post-gatekeeper C2 seats may still run after
/// this pass.
pub(in crate::retrieval) fn c2_first_same_speaker(ix: &Index, ti: usize) -> Option<usize> {
    let members = ix.session_turns.get(ix.turns[ti].session)?;
    let sp = ix.turns[ti].speaker.as_str();
    members.iter().copied().find(|&m| ix.turns[m].speaker == sp)
}

/// Surface anchors: lexical content-group labels. Stems, hypernyms, and
/// entity-phrase ids are not anchors. The label's own vocab id is the mark.
pub(in crate::retrieval) fn c2_surface_anchor_ids(ix: &Index, plan: &QueryPlan) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    for label in &plan.groups {
        if label.starts_with('\u{1}') {
            continue;
        }
        let Some(id) = ix.vocab.get(label) else {
            continue;
        };
        if seen.insert(id) {
            ids.push(id);
        }
    }
    ids
}

/// Pool the replay dump writes: seated turns first, then scene-descending
/// turns with a positive scene score, capped at 50.
pub(in crate::retrieval) fn c2_dump_pool(ix: &Index, list: &[usize], t_scene: &[f32]) -> Vec<usize> {
    let mut pool: Vec<usize> = list.to_vec();
    let mut seen: DetHashSet<usize> = pool.iter().copied().collect();
    let turn_ts: Vec<i64> = ix
        .turns
        .iter()
        .map(|t| ix.session_ts.get(t.session).copied().unwrap_or(0))
        .collect();
    for ti in order(t_scene, &turn_ts) {
        if pool.len() >= 50 {
            break;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if seen.insert(ti) {
            pool.push(ti);
        }
    }
    pool
}

/// Surface form of a vocab id, with the stem/entity mark stripped.
pub(in crate::retrieval) fn c2_term_surface<'a>(ix: &'a Index, id: u32) -> &'a str {
    let raw = ix.vocab.names.get(id as usize).map(String::as_str).unwrap_or("");
    raw.strip_prefix("\u{1}stem:")
        .or_else(|| raw.strip_prefix("\u{1}ent:"))
        .or_else(|| raw.strip_prefix("\u{1}hyp:"))
        .unwrap_or(raw)
}

/// True when one of the policy turn's stored content terms and a query focal
/// term differ only by a closed inflectional suffix. Orthographic shape, not
/// a domain list. `date` does not meet `deadline`.
pub(in crate::retrieval) fn c2_policy_inflects_query(ix: &Index, ti: usize, plan: &QueryPlan) -> bool {
    fn inflects(shorter: &str, longer: &str) -> bool {
        if shorter.len() < 5 {
            return false;
        }
        let Some(rest) = longer.strip_prefix(shorter) else {
            return false;
        };
        matches!(
            rest,
            "" | "s"
                | "es"
                | "ed"
                | "ing"
                | "ings"
                | "er"
                | "ers"
                | "ly"
                | "al"
                | "ment"
                | "ments"
                | "tion"
                | "tions"
                | "ation"
                | "ations"
        )
    }
    ix.turns[ti].seq.iter().any(|&(raw, _)| {
        let term = c2_term_surface(ix, raw);
        plan.focal_ids.iter().any(|&fid| {
            let focal = c2_term_surface(ix, fid);
            if term.len() <= focal.len() {
                inflects(term, focal)
            } else {
                inflects(focal, term)
            }
        })
    })
}

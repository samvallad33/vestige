// Computer 2 algorithm lane (perf/c2-algos): one structural mechanism per commit.
// Included from retrieval.rs right after steps7_10.rs. Ranking helpers never read
// `TurnIx.body` (empty under `--no-dump`); they use ingest-time flags only.
// No domain word lists: closed-class structure and orthographic shape only.

/// Window (top-k slots) that the numeral seat may draw a challenger from.
const C2_BULLAE_WINDOW: usize = 5;
/// Challenger must hold at least this share of the head's scene score.
const C2_BULLAE_RATIO: f32 = 0.7;

/// Clay-token bullae (numeral attestation seat). A sealed bulla carried tokens that
/// counted the goods; the token, not the label, was the evidence. When slot 1 is a
/// digit-free turn and a same-speaker turn already in the top 5 carries a digit and
/// holds >= 70% of slot 1's scene score, the counted (digit-bearing) attestation takes
/// slot 1 and the former head follows it. Top-5 membership never changes. Runs after
/// hearsay and before `rsb_seat_behind`, which then seats behind the new head.
/// Skipped in named-speaker sessions.
fn apply_c2_bullae_numeral_seat(ix: &Index, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 {
        return false;
    }
    let top = list[0];
    let head = &ix.turns[top];
    if head.form_digit {
        return false;
    }
    // v2: user/assistant transcripts only. Named-speaker chats keep their own score path
    // (scene pooling over uncarried t_own), where a digit is not counted evidence.
    let named = ix.session_turns[head.session].iter().any(|&m| {
        let sp = ix.turns[m].speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    });
    if named {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != head.speaker || !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) >= C2_BULLAE_RATIO * sc_top {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}

/// Coverage margin (share of total query IC) a same-session challenger must add over slot 1.
const C2_COVER_MARGIN: f32 = 0.15;

/// True when the turn's session has a named (non user/assistant) speaker.
fn c2_named_session(ix: &Index, ti: usize) -> bool {
    ix.session_turns[ix.turns[ti].session].iter().any(|&m| {
        let sp = ix.turns[m].speaker.as_str();
        !sp.is_empty() && sp != "user" && sp != "assistant"
    })
}

/// Per query group: the BM25 idf (turn-own stats) of its rarest term times its plan weight.
fn c2_group_ic(ix: &Index, plan: &QueryPlan) -> Vec<f32> {
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
fn c2_turn_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
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

/// IC-weighted share of the query groups in `hit`.
fn c2_coverage(g_ic: &[f32], hit: &[bool]) -> f32 {
    let total: f32 = g_ic.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    hit.iter().zip(g_ic).filter(|(h, _)| **h).map(|(_, w)| *w).sum::<f32>() / total
}

/// Patent anticipation ("all elements in a single reference"), same-session form. Inside
/// slot 1's own session, a same-speaker top-5 turn that attests the query's elements by a
/// clear IC margin (>= 15% of total query IC more than slot 1) while holding every element
/// slot 1 holds is the anticipating
/// reference and takes slot 1. Top-5 membership never changes. Runs before RSB.
fn apply_c2_coverage_margin_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>) -> bool {
    if list.len() < 2 || plan.groups.len() < 2 {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    let top_hit = c2_turn_groups(ix, plan, ng, top);
    let cov_top = c2_coverage(&g_ic, &top_hit);
    let (sess, speaker) = (ix.turns[top].session, ix.turns[top].speaker.as_str());
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut best: Option<(usize, f32)> = None;
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.session != sess || t.speaker != speaker {
            continue;
        }
        let hit = c2_turn_groups(ix, plan, ng, cand);
        // v2: anticipation proper. The challenger must hold every element slot 1 holds.
        if top_hit.iter().zip(&hit).any(|(t, c)| *t && !*c) {
            continue;
        }
        let cov = c2_coverage(&g_ic, &hit);
        if cov >= cov_top + C2_COVER_MARGIN && best.is_none_or(|(_, b)| cov > b) {
            best = Some((i, cov));
        }
    }
    if let Some((i, _)) = best {
        let cand = list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Transfer-appropriate processing (Morris, Bransford & Franks 1977), when/where arms only:
/// retrieval succeeds when the test form matches the encoding form. For a when-query the
/// answer form is a temporal turn (`has_temporal` / `form_clock`); for a where-query a
/// spatial turn (`has_spatial`). Inside a session that holds a form-matching turn, a
/// non-matching turn may not stay above 90% of the best matching one. Cap only, never a
/// boost. HOOK B, after Lex specialis. (The how-many arm belongs to Steps 11-20.)
fn apply_c2_tap_form_precedence(ix: &Index, it: &QueryIntent, pref_on: bool, t_scene: &mut [f32]) {
    if pref_on || it.is_event_sequence || !(it.sniper_when || it.sniper_where) {
        return;
    }
    let form_ok = |t: &TurnIx| -> bool {
        (!it.sniper_when || t.has_temporal || t.form_clock) && (!it.sniper_where || t.has_spatial)
    };
    for turns in &ix.session_turns {
        // v2: named-speaker chats only. User/assistant transcripts state dates in session
        // metadata, not in the turn, so a temporal/spatial surface form is not the answer form.
        let named = turns.iter().any(|&m| {
            let sp = ix.turns[m].speaker.as_str();
            !sp.is_empty() && sp != "user" && sp != "assistant"
        });
        if !named {
            continue;
        }
        let best_ok = turns
            .iter()
            .filter(|&&ti| form_ok(&ix.turns[ti]))
            .map(|&ti| t_scene[ti])
            .fold(0.0f32, f32::max);
        if best_ok <= 0.0 {
            continue;
        }
        for &ti in turns {
            if !form_ok(&ix.turns[ti]) && t_scene[ti] > 0.9 * best_ok {
                t_scene[ti] = 0.9 * best_ok;
            }
        }
    }
}

/// Near-tie band (share of slot 1's scene score) inside which the identification key decides.
const C2_KEY_TIE: f32 = 0.03;

/// Dichotomous identification key (Lamarck 1778): split the remaining candidates on the single
/// most decisive character. When slots 1 and 2 are within 3% on scene score, the query term
/// with the highest turn-own IDF that exactly one of them attests decides: its holder takes
/// slot 1. Deterministic (IDF desc, term id asc). Runs before RSB.
fn apply_c2_dichotomous_key(ix: &Index, plan: &QueryPlan, list: &mut [usize], t_scene: &[f32]) -> bool {
    if list.len() < 2 {
        return false;
    }
    let (a, b) = (list[0], list[1]);
    let (sa, sb) = (t_scene.get(a).copied().unwrap_or(0.0), t_scene.get(b).copied().unwrap_or(0.0));
    if sa <= 0.0 || (sa - sb).abs() >= C2_KEY_TIE * sa {
        return false;
    }
    // v2: named-speaker chats only (user/assistant near-ties are settled by the seats above).
    if !c2_named_session(ix, a) || !c2_named_session(ix, b) {
        return false;
    }
    let has = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let mut terms: Vec<(f32, u32)> = plan
        .terms
        .iter()
        .filter(|q| q.group.is_some() && has(a, q.id) != has(b, q.id))
        .map(|q| {
            let df = ix.t_own_stats.df.get(&q.id).copied().unwrap_or(0) as f32;
            (((ix.t_own_stats.n - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0) * q.weight, q.id)
        })
        .collect();
    terms.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    if let Some(&(_, w)) = terms.first() {
        if has(b, w) {
            list.swap(0, 1);
            return true;
        }
    }
    false
}

/// Near-tie band for the Mimamsa precedence order (share of slot 1's scene score).
const C2_MIMAMSA_BAND: f32 = 0.10;
/// Mimamsa viniyoga precedence key, two strongest means: sruti (focal query terms the turn
/// states) then linga (focal query terms it carries as named entities).
fn c2_mimamsa_key(ix: &Index, plan: &QueryPlan, ti: usize) -> (u16, u16) {
    let t = &ix.turns[ti];
    let sruti = plan.focal_ids.iter().filter(|f| t.content.contains(f)).count() as u16;
    let linga = plan.focal_ids.iter().filter(|f| t.entities.contains(f)).count() as u16;
    (sruti, linga)
}

/// Mimamsa viniyoga hierarchy (Jaimini 3.3.14): direct statement beats indicatory mark, which
/// beats syntactic connection, then position; a weaker means never overrules a stronger one.
/// In named-speaker chats, among slot 1 and its same-session same-speaker top-5 rivals within
/// 10% of its scene score, a strictly stronger (sruti, linga) key takes slot 1; weaker means
/// (connection, position) are left to the scored order. Top-5 membership never changes.
/// Runs before RSB.
fn apply_c2_mimamsa_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.focal_ids.is_empty() {
        return false;
    }
    let top = list[0];
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    // v3: named-speaker chats only; there a named referent is the sruti of the scene.
    if !c2_named_session(ix, top) {
        return false;
    }
    let (sess, speaker) = (ix.turns[top].session, ix.turns[top].speaker.as_str());
    let mut best = (0usize, c2_mimamsa_key(ix, plan, top));
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.session != sess || t.speaker != speaker {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < (1.0 - C2_MIMAMSA_BAND) * sc_top {
            continue;
        }
        let k = c2_mimamsa_key(ix, plan, cand);
        if k > best.1 {
            best = (i, k);
        }
    }
    if best.0 > 0 {
        let cand = list.remove(best.0);
        list.insert(0, cand);
        return true;
    }
    false
}

/// Share of slot 1's scene score a primacy-testimony challenger must keep.
const C2_TESTIMONY_RATIO: f32 = 0.5;

/// Query groups a turn attests in declarative position: before its terminal question when it
/// has one, anywhere when it has none. Groups named only inside the question are inquiry.
fn c2_declarative_groups(ix: &Index, plan: &QueryPlan, n_groups: usize, ti: usize) -> Vec<bool> {
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
fn c2_session_pos(ix: &Index, ti: usize) -> usize {
    ix.session_turns[ix.turns[ti].session].iter().position(|&m| m == ti).unwrap_or(0)
}

/// Primacy testimony seat (first testimony: the account a witness gives first, before the
/// conversation drifts, is the one entered in the record). A same-role top-5 turn, from any
/// user/assistant session, that sits in the first half of slot 1's own session depth (its
/// session position x2 < slot 1's position), states in declarative position at least the query
/// IC slot 1 states, and keeps >= 50% of slot 1's scene score, takes slot 1 (first in list
/// order wins). Top-5 membership never changes. Runs before RSB. Skipped in named sessions.
fn apply_c2_testimony_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    // v3: testimony is the user's own account; assistant replies are not first accounts.
    if ix.turns[top].speaker != "user" || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let g_ic = c2_group_ic(ix, plan);
    let ng = g_ic.len();
    let d_top = c2_coverage(&g_ic, &c2_declarative_groups(ix, plan, ng, top));
    let pos_top = c2_session_pos(ix, top);
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != ix.turns[top].speaker || c2_named_session(ix, cand) {
            continue;
        }
        // v2: the first account must hold every surface form slot 1 holds (a counted digit).
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if c2_session_pos(ix, cand) * 2 >= pos_top {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_TESTIMONY_RATIO * sc_top {
            continue;
        }
        let d = c2_coverage(&g_ic, &c2_declarative_groups(ix, plan, ng, cand));
        if d > 0.0 && d >= d_top {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}

/// Share of slot 1's scene score a cross-session earlier witness must keep.
const C2_XWITNESS_RATIO: f32 = 0.7;

/// Earlier-witness seat across sessions (the deposition taken closest to the start of a
/// proceeding outranks a later restatement elsewhere). Not for temporal questions. When slot 1
/// is a user turn, a user turn from a different user/assistant session in the top 5 that sits
/// strictly earlier in its own session than slot 1 sits in slot 1's, keeps slot 1's digit form,
/// attests at least one query group, and holds >= 70% of slot 1's scene score takes slot 1.
/// Top-5 membership never changes. Runs before RSB, after the primacy testimony seat.
fn apply_c2_cross_witness_seat(ix: &Index, plan: &QueryPlan, list: &mut Vec<usize>, t_scene: &[f32]) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let top = list[0];
    if plan.intent.temporal || ix.turns[top].speaker != "user" || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let ng = plan.groups.len();
    let pos_top = c2_session_pos(ix, top);
    let window = list.len().min(C2_BULLAE_WINDOW);
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user" || t.session == ix.turns[top].session || c2_named_session(ix, cand) {
            continue;
        }
        if (ix.turns[top].form_digit && !t.form_digit) || c2_session_pos(ix, cand) >= pos_top {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        if c2_turn_groups(ix, plan, ng, cand).iter().any(|h| *h) {
            list.remove(i);
            list.insert(0, cand);
            return true;
        }
    }
    false
}


/// Challenger must hold at least this share of the evicted assistant's scene score.
const C2_REALLOC_RATIO: f32 = 0.5;

/// Same-session pool-fill reallocation: a user source that attests a strict superset
/// of an assistant restatement's query groups takes that assistant's top-5 seat.
/// Mined from the 51 LME Turn@5 misses at 877804cf, where gold often sat in the
/// already-chosen session but lost its seat to a same-session assistant (session
/// cap / greedy scene order). Distinct from ReplySource (slot-1 only, previous
/// user already in the top 5) and from C2 coverage-margin (slot-1, top-5 membership
/// unchanged, same speaker). Top-5 membership may change; the evicted assistant
/// is pushed behind the top 5. One swap. Skips named-speaker sessions. Runs
/// before RSB. Not used on assistant-attribution or event-sequence queries
/// (caller).
fn apply_c2_source_superset_realloc(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.groups.is_empty() {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if window < 2 {
        return false;
    }
    let top = list[0];
    if c2_named_session(ix, top) {
        return false;
    }
    let ng = plan.groups.len();
    let g_ic = c2_group_ic(ix, plan);
    let top5 = &list[..window];
    let mut sess_ok: DetHashSet<usize> = DetHashSet::default();
    for &ti in top5 {
        sess_ok.insert(ix.turns[ti].session);
    }
    // Best leftover user in a top-5 session: coverage desc, scene desc, index asc.
    let mut best_c: Option<(f32, f32, usize)> = None;
    let mut best_hit: Vec<bool> = Vec::new();
    for &si in &sess_ok {
        let Some(members) = ix.session_turns.get(si) else {
            continue;
        };
        for &ti in members {
            if top5.contains(&ti) {
                continue;
            }
            let t = &ix.turns[ti];
            if t.speaker != "user" {
                continue;
            }
            let sc = t_scene.get(ti).copied().unwrap_or(0.0);
            if sc <= 0.0 {
                continue;
            }
            let hit = c2_turn_groups(ix, plan, ng, ti);
            if !hit.iter().any(|h| *h) {
                continue;
            }
            let cov = c2_coverage(&g_ic, &hit);
            let take = match best_c {
                None => true,
                Some((bc, bsc, bi)) => cov > bc || (cov == bc && (sc > bsc || (sc == bsc && ti < bi))),
            };
            if take {
                best_c = Some((cov, sc, ti));
                best_hit = hit;
            }
        }
    }
    let Some((_, c_sc, cand)) = best_c else {
        return false;
    };
    let csess = ix.turns[cand].session;
    // Weakest same-session assistant in the top 5 that the user strictly supersets.
    let mut best_w: Option<(f32, f32, usize, usize)> = None; // cov, scene, slot, ti
    for (slot, &wi) in top5.iter().enumerate() {
        let t = &ix.turns[wi];
        if t.session != csess || t.speaker != "assistant" {
            continue;
        }
        let w_hit = c2_turn_groups(ix, plan, ng, wi);
        // Strict superset: every group W holds, plus at least one more.
        if w_hit.iter().zip(&best_hit).any(|(w, c)| *w && !*c) {
            continue;
        }
        if !best_hit.iter().zip(&w_hit).any(|(c, w)| *c && !*w) {
            continue;
        }
        let w_sc = t_scene.get(wi).copied().unwrap_or(0.0);
        if c_sc < C2_REALLOC_RATIO * w_sc {
            continue;
        }
        let w_cov = c2_coverage(&g_ic, &w_hit);
        let take = match best_w {
            None => true,
            Some((bc, bsc, bslot, _)) => {
                w_cov < bc || (w_cov == bc && (w_sc < bsc || (w_sc == bsc && slot < bslot)))
            }
        };
        if take {
            best_w = Some((w_cov, w_sc, slot, wi));
        }
    }
    let Some((_, _, slot, victim)) = best_w else {
        return false;
    };
    let k = list.len();
    if let Some(pos) = list.iter().position(|&t| t == cand) {
        list.remove(pos);
        let slot = if pos < slot { slot - 1 } else { slot };
        let vpos = list.iter().position(|&t| t == victim).unwrap_or(slot);
        list.remove(vpos);
        list.insert(vpos, cand);
        list.push(victim);
    } else {
        list.remove(slot);
        list.insert(slot, cand);
        list.push(victim);
    }
    if list.len() > k {
        list.truncate(k);
    }
    true
}

#[cfg(test)]
mod algos_c2_tests {
    use super::*;

    fn sess(id: &str, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.into(),
            date: String::new(),
            timestamp: 0,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (r, c))| Turn { id: format!("{id}#{i}"), role: (*r).into(), content: (*c).into(), ..Turn::default() })
                .collect(),
        }
    }

    #[test]
    fn bullae_seats_digit_turn_over_digit_free_head() {
        let ix = build_index(
            &[sess("s", &[("user", "I love my bikes a lot."), ("user", "I own 3 bikes now."), ("assistant", "Nice.")])],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 2];
        let t_scene = vec![1.0f32, 0.8, 0.5];
        assert!(apply_c2_bullae_numeral_seat(&ix, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // Below the 70% share: no seat.
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_bullae_numeral_seat(&ix, &mut list, &[1.0, 0.5, 0.5]));
        assert_eq!(list, vec![0, 1, 2]);
    }

    #[test]
    fn bullae_skips_named_speaker_sessions() {
        let ix = build_index(
            &[sess("D1", &[("Caroline", "I love my bikes a lot."), ("Caroline", "I own 3 bikes now.")])],
            &Params::default(),
        );
        let mut list = vec![0usize, 1];
        assert!(!apply_c2_bullae_numeral_seat(&ix, &mut list, &[1.0, 0.9]));
        assert_eq!(list, vec![0, 1]);
    }

    #[test]
    fn coverage_margin_needs_superset_of_head_elements() {
        let ix = build_index(
            &[sess("s", &[("user", "I like hiking."), ("user", "My hiking boots came from Denver."), ("user", "Denver trip soon.")])],
            &Params::default(),
        );
        let q = Query { text: "Where did I buy my hiking boots?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(apply_c2_coverage_margin_seat(&ix, &plan, &mut list));
        assert_eq!(list[0], 1);
        // Turn 2 lacks "hiking" that slot 1 holds: never seated over turn 0.
        let mut list = vec![0usize, 2];
        assert!(!apply_c2_coverage_margin_seat(&ix, &plan, &mut list));
    }

    #[test]
    fn tap_caps_formless_turn_in_named_session_only() {
        let ix = build_index(
            &[
                sess("D1", &[("Caroline", "We talked about the concert."), ("Caroline", "The concert was on 7 May 2023.")]),
                sess("u", &[("user", "We talked about the concert."), ("user", "The concert was on 7 May 2023.")]),
            ],
            &Params::default(),
        );
        assert!(ix.turns[1].has_temporal && !ix.turns[0].has_temporal);
        let it = QueryIntent { sniper_when: true, ..QueryIntent::default() };
        let mut t_scene = vec![1.0f32, 0.5, 1.0, 0.5];
        apply_c2_tap_form_precedence(&ix, &it, false, &mut t_scene);
        assert!((t_scene[0] - 0.45).abs() < 1e-6);
        assert_eq!(t_scene[2], 1.0);
        let mut t_scene = vec![1.0f32, 0.5, 1.0, 0.5];
        apply_c2_tap_form_precedence(&ix, &it, true, &mut t_scene);
        assert_eq!(t_scene[0], 1.0);
    }

    #[test]
    fn dichotomous_key_breaks_named_near_tie_on_rarest_split_term() {
        let ix = build_index(
            &[sess("D1", &[("Caroline", "I went hiking yesterday."), ("Melanie", "I went hiking in Yosemite yesterday.")])],
            &Params::default(),
        );
        let q = Query { text: "When did Melanie go hiking in Yosemite?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        let mut list = vec![0usize, 1];
        assert!(apply_c2_dichotomous_key(&ix, &plan, &mut list, &[1.0, 0.99]));
        assert_eq!(list, vec![1, 0]);
        // Outside the 3% band: no change.
        let mut list = vec![0usize, 1];
        assert!(!apply_c2_dichotomous_key(&ix, &plan, &mut list, &[1.0, 0.9]));
    }

    #[test]
    fn mimamsa_seats_stronger_sruti_in_named_session() {
        let ix = build_index(
            &[sess("D1", &[("Caroline", "We went hiking last week."), ("Caroline", "Yosemite hiking was the best trip.")])],
            &Params::default(),
        );
        let q = Query { text: "What did Caroline say about Yosemite hiking?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        let k0 = c2_mimamsa_key(&ix, &plan, 0);
        let k1 = c2_mimamsa_key(&ix, &plan, 1);
        let mut list = vec![0usize, 1];
        let moved = apply_c2_mimamsa_seat(&ix, &plan, &mut list, &[1.0, 0.95]);
        assert_eq!(moved, k1 > k0);
        // Outside the 10% band: never moved.
        let mut list = vec![0usize, 1];
        assert!(!apply_c2_mimamsa_seat(&ix, &plan, &mut list, &[1.0, 0.5]));
    }

    #[test]
    fn primacy_testimony_seats_early_user_statement() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "My cat Luna loves the window seat."),
                    ("assistant", "Cats love sun."),
                    ("user", "Unrelated note."),
                    ("assistant", "Sure."),
                    ("user", "Do you know cat names for a cat?"),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "What is the name of my cat?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        // Slot 1 = turn 4 (position 4), challenger turn 0 (position 0 x2 < 4), scene share 0.6.
        let mut list = vec![4usize, 0, 1];
        assert!(apply_c2_testimony_seat(&ix, &plan, &mut list, &[0.6, 0.1, 0.0, 0.0, 1.0]));
        assert_eq!(list[0], 0);
        // Below the 50% share: no seat.
        let mut list = vec![4usize, 0, 1];
        assert!(!apply_c2_testimony_seat(&ix, &plan, &mut list, &[0.4, 0.1, 0.0, 0.0, 1.0]));
        // Assistant heads are never displaced.
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_testimony_seat(&ix, &plan, &mut list, &[0.9, 1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn cross_witness_seats_earlier_user_turn_from_other_session() {
        let ix = build_index(
            &[
                sess("a", &[("user", "I adopted my cat Luna."), ("assistant", "Lovely.")]),
                sess("b", &[("user", "Hi."), ("assistant", "Hello."), ("user", "My cat needs toys.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "What is my cat called?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.temporal);
        // Slot 1 = turn 4 (session b, pos 2); challenger turn 0 (session a, pos 0) at 0.8 share.
        let mut list = vec![4usize, 0];
        assert!(apply_c2_cross_witness_seat(&ix, &plan, &mut list, &[0.8, 0.0, 0.0, 0.0, 1.0]));
        assert_eq!(list, vec![0, 4]);
        let mut list = vec![4usize, 0];
        assert!(!apply_c2_cross_witness_seat(&ix, &plan, &mut list, &[0.6, 0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn source_superset_realloc_seats_user_over_assistant() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I went to Hawaii with my family."),
                    ("assistant", "Family time is wonderful and has unique benefits."),
                    ("user", "Unrelated note about cooking."),
                    ("assistant", "Nice."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "Where did I go with my family to Hawaii?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        // Assistant at slot 2 (turn 1) attests "family"; leftover user (turn 0) is a strict superset (family+hawaii).
        let mut list = vec![2usize, 1];
        let t_scene = vec![0.7f32, 1.0, 0.4, 0.3];
        assert!(apply_c2_source_superset_realloc(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0]);
        // Below the 50% scene share: no swap.
        let mut list = vec![2usize, 1];
        assert!(!apply_c2_source_superset_realloc(&ix, &plan, &mut list, &[0.4, 1.0, 0.4, 0.3]));
        // Named-speaker sessions are left alone.
        let ix = build_index(
            &[sess("D1", &[("Caroline", "I went to Hawaii with my family."), ("Melanie", "Family time is wonderful.")])],
            &Params::default(),
        );
        let plan = plan_query(&ix, &q, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_source_superset_realloc(&ix, &plan, &mut list, &[0.8, 1.0]));
    }

}

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
/// Challenger must hold at least this share of the victim assistant's scene score.
const C2_UNIQUE_GROUP_RATIO: f32 = 0.5;

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


/// Same-session unique-group realloc. A user turn already in the retrieved
/// window that attests a query group none of the already-seated same-session
/// top-5 turns attest takes a same-session assistant's top-5 seat, provided it
/// holds >= 50% of that assistant's scene score and the swap does not drop a
/// query group the remaining top 5 (plus the challenger) would otherwise lose.
/// The evicted assistant stays in the window (pushed behind top 5). Challengers
/// outside the current k-list are ignored, so top-k membership does not change.
/// Complementary to source-superset realloc: the challenger need not be a
/// superset of the assistant, only uniquely attest something the session's
/// current representatives missed. Named-speaker sessions are skipped. One swap.
/// Runs after source-superset realloc, before RSB.
fn apply_c2_unique_group_realloc(
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
    struct Pair {
        n_unique: usize,
        cov: f32,
        sc: f32,
        cand: usize,
        w_cov: f32,
        w_sc: f32,
        slot: usize,
        victim: usize,
    }
    let mut best: Option<Pair> = None;
    for &si in &sess_ok {
        let Some(members) = ix.session_turns.get(si) else {
            continue;
        };
        let sibs: Vec<usize> = top5
            .iter()
            .copied()
            .filter(|&ti| ix.turns[ti].session == si)
            .collect();
        if sibs.is_empty() {
            continue;
        }
        let mut union = vec![false; ng];
        for &ti in &sibs {
            let hit = c2_turn_groups(ix, plan, ng, ti);
            for (u, h) in union.iter_mut().zip(&hit) {
                *u |= *h;
            }
        }
        for &ti in members {
            if top5.contains(&ti) {
                continue;
            }
            // Already in the retrieved window (slots 6..k). Outside-list turns
            // would grow the window or drop the evicted assistant from top-k.
            if !list.contains(&ti) {
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
            let n_unique = hit.iter().zip(&union).filter(|(c, u)| **c && !**u).count();
            if n_unique == 0 {
                continue;
            }
            let cov = c2_coverage(&g_ic, &hit);
            for (slot, &wi) in top5.iter().enumerate() {
                let w = &ix.turns[wi];
                if w.session != si || w.speaker != "assistant" {
                    continue;
                }
                let w_sc = t_scene.get(wi).copied().unwrap_or(0.0);
                if sc < C2_UNIQUE_GROUP_RATIO * w_sc {
                    continue;
                }
                let w_hit = c2_turn_groups(ix, plan, ng, wi);
                // Do not evict an assistant that uniquely attests a group the
                // remaining top 5 plus the challenger would lose.
                let mut remain = hit.clone();
                for &oi in top5 {
                    if oi == wi {
                        continue;
                    }
                    let oh = c2_turn_groups(ix, plan, ng, oi);
                    for (r, h) in remain.iter_mut().zip(&oh) {
                        *r |= *h;
                    }
                }
                if w_hit.iter().zip(&remain).any(|(w_h, r)| *w_h && !*r) {
                    continue;
                }
                let w_cov = c2_coverage(&g_ic, &w_hit);
                let take = match &best {
                    None => true,
                    Some(b) => {
                        if n_unique != b.n_unique {
                            n_unique > b.n_unique
                        } else if cov != b.cov {
                            cov > b.cov
                        } else if sc != b.sc {
                            sc > b.sc
                        } else if ti != b.cand {
                            ti < b.cand
                        } else if w_cov != b.w_cov {
                            w_cov < b.w_cov
                        } else if w_sc != b.w_sc {
                            w_sc < b.w_sc
                        } else {
                            slot < b.slot
                        }
                    }
                };
                if take {
                    best = Some(Pair {
                        n_unique,
                        cov,
                        sc,
                        cand: ti,
                        w_cov,
                        w_sc,
                        slot,
                        victim: wi,
                    });
                }
            }
        }
    }
    let Some(pair) = best else {
        return false;
    };
    let cand = pair.cand;
    let victim = pair.victim;
    let slot = pair.slot;
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


/// Share of slot 1's scene score a subsequent-answer challenger must keep.
const C2_SUBSEQUENT_RATIO: f32 = 0.5;

/// First later user turn in the same session, if any.
fn c2_next_user(ix: &Index, ti: usize) -> Option<usize> {
    let members = ix.session_turns.get(ix.turns[ti].session)?;
    let pos = members.iter().position(|&m| m == ti)?;
    members[pos + 1..]
        .iter()
        .copied()
        .find(|&m| ix.turns[m].speaker == "user")
}

/// Subsequent-answer seat. A mid-session user question at slot 1 often holds
/// the scene-setting inquiry ("by the way, …?"); the next user turn already
/// in the top 5 is the already-retrieved answer. When that follow-up is
/// declarative, keeps >= 50% of slot 1's scene score, and the head is not
/// the session opener, it takes slot 1. Top-5 membership never changes.
/// Session openers stay put (their follow-up is often more planning).
/// Skipped on temporal questions (those keep the existing timeline path),
/// event sequences and assistant-attribution (caller), and named-speaker
/// sessions. Distinct from primacy testimony (which promotes an earlier
/// first-half account) and from ReplySource (assistant at slot 1, previous
/// user). Runs after unique-group realloc, before RSB. Slot 1 is then
/// frozen by rank1_gatekeeper_keep_rsb_head, so this reseat sticks; it is
/// not a Kemeny ballot (a tail vote cannot move T@1).
fn apply_c2_subsequent_answer_seat(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.temporal {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || !ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(cand) = c2_next_user(ix, top) else {
        return false;
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    let t = &ix.turns[cand];
    if t.speaker != "user" || t.is_question {
        return false;
    }
    if t_scene.get(cand).copied().unwrap_or(0.0) < C2_SUBSEQUENT_RATIO * sc_top {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}


/// True when the query cites a prior assistant utterance (collective chat,
/// "remind me", "you recommended", …) rather than only asking for a new
/// recommendation. The present-tense cue "you recommend" is excluded because
/// it also matches preference requests ("can you recommend...").
fn c2_retrospective_attribution(text: &str) -> bool {
    let padded = word_padded(text);
    if has_any(&padded, COLLECTIVE_CHAT_CUES) {
        return true;
    }
    ASSISTANT_ATTRIBUTION_CUES
        .iter()
        .any(|c| *c != "you recommend" && has_any(&padded, &[*c]))
}

/// Preference source seat. ReplySource already yields an assistant at slot 1
/// to its previous user turn when that source is already in the top 5, but it
/// is skipped on assistant-attribution queries. Preference requests of the
/// form "can you recommend..." are tagged assistant_attribution by the
/// present-tense "you recommend" cue, so the user's preference-stating
/// source never takes slot 1. When the query is a preference intent and is
/// not a retrospective citation of a prior assistant utterance, apply the
/// same membership-preserving source move (position and role only). Skipped
/// on temporal questions, event sequences (caller), and named-speaker
/// sessions. Distinct from ReplySource (unchanged; still skipped on
/// assistant_attribution) and from subsequent-answer (#10; a user question
/// at slot 1). Runs after the other C2 seats, before RSB. Slot 1 is then
/// frozen by rank1_gatekeeper_keep_rsb_head.
fn apply_c2_preference_source_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
) -> bool {
    if list.len() < 2 || plan.intent.temporal || !plan.intent.preference {
        return false;
    }
    if c2_retrospective_attribution(query) {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "assistant" || c2_named_session(ix, top) {
        return false;
    }
    let members = &ix.session_turns[ix.turns[top].session];
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    if pos == 0 {
        return false;
    }
    let prev = members[pos - 1];
    if ix.turns[prev].speaker != "user" {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == prev) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, prev);
    true
}


/// Next-user elaboration seat. A mid-session declarative user turn at
/// slot 1 is often a list or scene overview; the next same-session user
/// turn already in the top 5, also declarative, with a scene score at
/// least that of the head, is the elaboration and takes slot 1. Top-5
/// membership never changes. Session openers stay put. Skipped on
/// temporal questions, preference questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. Sibling
/// of subsequent-answer (#10), which fires when the mid-session head is
/// a question and the follow-up holds >= 50% of its scene score. This
/// seat requires a declarative head and a follow-up that is at least as
/// on-scene as that head. Runs after subsequent-answer, before RSB.
/// Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
fn apply_c2_elaboration_seat(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.temporal || plan.intent.preference {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(cand) = c2_next_user(ix, top) else {
        return false;
    };
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    let t = &ix.turns[cand];
    if t.speaker != "user" || t.is_question {
        return false;
    }
    if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
    true
}


/// True when the query is a summary request of the form already used by
/// `summary_hold` in `rank_one`: a summary cue plus a task-under-rules cue,
/// and not a user-initiative recall cue. Reuses those existing cue lists
/// (no new word list). Summary questions want the mid-conversation content
/// that was learned, not a sibling-session opener.
fn c2_summary_request(query: &str) -> bool {
    let padded = word_padded(query);
    has_any(&padded, SUMMARY_REQUEST_CUES)
        && has_any(&padded, TASK_UNDER_RULES_CUES)
        && !has_any(&padded, USER_INITIATIVE_CUES)
}

/// Cross-session opener seat. When slot 1 is a mid-session user turn, a
/// user session-opener from a different user/assistant session already in
/// the top 5, that keeps slot 1's digit form and holds >= 70% of slot 1's
/// scene score, takes slot 1. The conversation start is the lead when the
/// sitting head has already drifted mid-topic. Top-5 membership never
/// changes. Skipped on summary requests (existing `summary_hold` shape:
/// those questions want the mid-conversation content, not an opener),
/// preference and temporal questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. Distinct
/// from cross-witness (any earlier session position, requires a query
/// group) and from SessionOpenerLift (requires the opener to cover every
/// query cue the head matched). Reuses `C2_XWITNESS_RATIO`. Runs after
/// the elaboration seat, before RSB. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head.
fn apply_c2_cross_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.preference || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
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
        if t.speaker != "user"
            || t.session == ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Same-session opener seat. A mid-session declarative user turn at
/// slot 1 is often a later aside that still mentions the fact; the same
/// session's user opener already in the top 5, that keeps slot 1's
/// digit form and holds >= 70% of its scene score, is the conversation
/// start and takes slot 1. Top-5 membership never changes. Skipped on
/// summary requests (existing `summary_hold` shape), preference and
/// temporal questions, event sequences and assistant-attribution
/// (caller), and named-speaker sessions. Sibling of the cross-session
/// opener seat (#13), which requires a different session and allows a
/// question head. This seat requires the opener to be in the same
/// session and both the head and the opener to be declarative: a
/// question head already asking the query should stay, and a
/// question opener is still asking, not stating the fact. Reuses
/// `C2_XWITNESS_RATIO`. Runs after
/// the cross-session opener seat, before RSB. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head.
fn apply_c2_same_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.preference || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) < 1
    {
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
        if t.speaker != "user"
            || t.is_question
            || t.session != ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if ix.turns[top].form_digit && !t.form_digit {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// True when the padded query contains the speaker's name as a whole token.
/// Speakers are the conversation's own role labels (no name list). User and
/// assistant roles never count.
fn c2_query_mentions_speaker(query: &str, speaker: &str) -> bool {
    if speaker.is_empty() || speaker == "user" || speaker == "assistant" {
        return false;
    }
    let padded = word_padded(query);
    let sp = speaker.to_ascii_lowercase();
    // word_padded keeps a possessive apostrophe ("Melanie's" -> " melanie's ").
    padded.contains(&format!(" {sp} "))
        || padded.contains(&format!(" {sp}'s "))
        || padded.contains(&format!(" {sp}’s "))
}

/// True when a mentioned speaker is conjoined with someone else ("X and Y").
/// Those questions are often answered by a third-party report, not by X's
/// next reply.
fn c2_speaker_conjoined(query: &str, speaker: &str) -> bool {
    if speaker.is_empty() {
        return false;
    }
    let padded = word_padded(query);
    let sp = speaker.to_ascii_lowercase();
    padded.contains(&format!(" {sp} and ")) || padded.contains(&format!(" and {sp} "))
}

/// Queried-speaker reply seat. In a named-speaker dialogue, an unmentioned
/// speaker's question at slot 1 is often the prompt; the immediately
/// following same-session turn, by a speaker the query names, already in
/// the top 5, is that speaker's reply and takes slot 1. Top-5 membership
/// never changes. Skipped when the query conjoins the named speaker with
/// someone else ("X and Y": the answer is often a third-party report),
/// on WH-person questions (existing `wh_slots`, so this does not compete
/// with Gemini Step 11), preference and temporal questions, summary
/// requests, event sequences and assistant-attribution (caller). Does
/// not fire on user/assistant chats. Runs after the same-session opener
/// seat, before RSB. Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
fn apply_c2_queried_speaker_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || plan.intent.preference
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty()
        || head_sp == "user"
        || head_sp == "assistant"
        || !ix.turns[top].is_question
        || c2_query_mentions_speaker(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(pos) = members.iter().position(|&m| m == top) else {
        return false;
    };
    let Some(&cand) = members.get(pos + 1) else {
        return false;
    };
    let csp = ix.turns[cand].speaker.as_str();
    if !c2_query_mentions_speaker(query, csp) || c2_speaker_conjoined(query, csp) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(slot) = list[..window].iter().position(|&t| t == cand) else {
        return false;
    };
    if slot == 0 {
        return false;
    }
    list.remove(slot);
    list.insert(0, cand);
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

    #[test]
    fn unique_group_realloc_seats_user_over_assistant() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "Are there movie festivals this year?"),
                    ("assistant", "There are plenty of film festivals around."),
                    ("user", "I attended two last year on my own."),
                    ("assistant", "Nice."),
                    ("user", "Unrelated note about cooking dinner."),
                    ("assistant", "Sounds good."),
                    ("user", "I also bought groceries today."),
                    ("user", "The weather is mild."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "How many movie festivals that I attended?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        // Assistant at slot 2 (turn 1) attests "festival"; user turn 2 is already
        // in the window (slot 6) and uniquely attests "attended" (not a superset).
        // Top-5 fillers are users so source-superset realloc does not fire on an
        // empty-group assistant.
        let mut list = vec![0usize, 1, 4, 6, 7, 2];
        let t_scene = vec![1.0f32, 1.0, 0.7, 0.3, 0.2, 0.2, 0.2, 0.2];
        assert!(!apply_c2_source_superset_realloc(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 6, 7, 2]);
        assert!(apply_c2_unique_group_realloc(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 4, 6, 7, 1]);
        // Below the 50% scene share: no swap.
        let mut list = vec![0usize, 1, 4, 6, 7, 2];
        assert!(!apply_c2_unique_group_realloc(
            &ix,
            &plan,
            &mut list,
            &[1.0, 1.0, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2]
        ));
        // Outside the retrieved window: no swap (membership must stay the same).
        let mut list = vec![0usize, 1, 4, 6, 7];
        assert!(!apply_c2_unique_group_realloc(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 6, 7]);
        // No unique group: leftover only restates "festival".
        let ix2 = build_index(
            &[sess(
                "s",
                &[
                    ("user", "Are there movie festivals this year?"),
                    ("assistant", "There are plenty of film festivals around."),
                    ("user", "Festivals are fun to watch."),
                    ("assistant", "Nice."),
                    ("user", "Unrelated note about cooking dinner."),
                    ("assistant", "Sounds good."),
                    ("user", "I also bought groceries today."),
                    ("user", "The weather is mild."),
                ],
            )],
            &Params::default(),
        );
        let plan2 = plan_query(&ix2, &q, &Params::default());
        let mut list = vec![0usize, 1, 4, 6, 7, 2];
        assert!(!apply_c2_unique_group_realloc(&ix2, &plan2, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 6, 7, 2]);
        // Named-speaker sessions are left alone.
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("Caroline", "Are there movie festivals this year?"),
                    ("Melanie", "There are plenty of film festivals around."),
                    ("Caroline", "I attended two last year on my own."),
                ],
            )],
            &Params::default(),
        );
        let plan = plan_query(&ix, &q, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_unique_group_realloc(&ix, &plan, &mut list, &[1.0, 1.0, 0.7]));
    }

    #[test]
    fn subsequent_answer_seat_promotes_next_user_over_mid_session_question() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I'm planning a hike this weekend."),
                    ("assistant", "Nice."),
                    ("user", "By the way, do you know any good yoga apps?"),
                    ("assistant", "There are several good ones."),
                    ("user", "I've been using Down Dog, especially when I can't make it to Serenity Yoga."),
                    ("user", "Unrelated note about cooking dinner."),
                    ("assistant", "Sounds good."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "Where do I take yoga classes?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.temporal);
        assert!(ix.turns[2].is_question);
        assert!(!ix.turns[4].is_question);
        // Slot 1 is the mid-session question (turn 2, session pos 2). The next
        // user turn (turn 4) is already in the top 5 and is declarative.
        let mut list = vec![2usize, 3, 4, 5, 6];
        let t_scene = vec![0.5f32, 0.4, 1.0, 0.8, 0.64, 0.3, 0.2];
        assert!(apply_c2_subsequent_answer_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![4, 2, 3, 5, 6]);
        // Below the 50% scene share: no swap.
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_subsequent_answer_seat(
            &ix,
            &plan,
            &mut list,
            &[0.5, 0.4, 1.0, 0.8, 0.4, 0.3, 0.2]
        ));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Session opener at slot 1: no swap.
        let mut list = vec![0usize, 1, 4, 5, 6];
        assert!(!apply_c2_subsequent_answer_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 6]);
        // Next user not already in the top 5: membership must not change.
        let mut list = vec![2usize, 3, 5, 6, 1];
        assert!(!apply_c2_subsequent_answer_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 5, 6, 1]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I take yoga classes?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_subsequent_answer_seat(&ix, &tplan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[sess(
                "D1",
                &[
                    ("Caroline", "I'm planning a hike this weekend."),
                    ("Melanie", "Nice."),
                    ("Caroline", "By the way, do you know any good yoga apps?"),
                    ("Melanie", "There are several good ones."),
                    ("Caroline", "I've been using Down Dog at Serenity Yoga."),
                ],
            )],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![2usize, 3, 4];
        assert!(!apply_c2_subsequent_answer_seat(&ixn, &plann, &mut list, &[0.5, 0.4, 1.0, 0.8, 0.64]));
        assert_eq!(list, vec![2, 3, 4]);
    }

    #[test]
    fn preference_source_seat_promotes_prev_user_on_can_you_recommend() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I'm trying to learn more about video editing with Premiere, which I enjoy to use."),
                    ("assistant", "Adobe Premiere Pro is a powerful video editing software with many resources."),
                    ("user", "What about color grading?"),
                    ("assistant", "For color grading, start with a simple LUT."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "Can you recommend some resources where I can learn more about video editing?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(plan.intent.preference);
        assert!(plan.intent.assistant_attribution);
        assert!(!c2_retrospective_attribution(&q.text));
        // Assistant at slot 1; previous user already in the top 5.
        let mut list = vec![1usize, 2, 0, 3];
        assert!(apply_c2_preference_source_seat(&ix, &plan, &q.text, &mut list));
        assert_eq!(list, vec![0, 1, 2, 3]);
        // Previous user not already in the top 5: membership must not change.
        let mut list = vec![1usize, 2, 3];
        assert!(!apply_c2_preference_source_seat(&ix, &plan, &q.text, &mut list));
        assert_eq!(list, vec![1, 2, 3]);
        // Retrospective "remind me / you recommended": no swap (SSA path).
        let rq = Query { text: "Can you remind me of the resources you recommended for video editing?".into(), date: String::new() };
        let rplan = plan_query(&ix, &rq, &Params::default());
        assert!(c2_retrospective_attribution(&rq.text));
        let mut list = vec![1usize, 2, 0, 3];
        assert!(!apply_c2_preference_source_seat(&ix, &rplan, &rq.text, &mut list));
        assert_eq!(list, vec![1, 2, 0, 3]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[sess(
                "D1",
                &[
                    ("Caroline", "I'm trying to learn more about video editing with Premiere."),
                    ("Melanie", "Adobe Premiere Pro is a powerful video editing software."),
                ],
            )],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_preference_source_seat(&ixn, &plann, &q.text, &mut list));
        assert_eq!(list, vec![1, 0]);
        // Non-preference query: no swap.
        let nq = Query { text: "What degree did I graduate with?".into(), date: String::new() };
        let nplan = plan_query(&ix, &nq, &Params::default());
        assert!(!nplan.intent.preference);
        let mut list = vec![1usize, 2, 0, 3];
        assert!(!apply_c2_preference_source_seat(&ix, &nplan, &nq.text, &mut list));
        assert_eq!(list, vec![1, 2, 0, 3]);
    }


    #[test]
    fn elaboration_seat_promotes_next_user_over_mid_session_overview() {
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I'm planning gifts this year."),
                    ("assistant", "Nice."),
                    ("user", "Here's a start - I've bought gifts for my sister's birthday, my mom, and my neighbor."),
                    ("assistant", "That's a good list."),
                    ("user", "For my sister's birthday, I got her a yellow dress and a pair of earrings to match."),
                    ("user", "I also got mom some flowers."),
                    ("assistant", "Lovely."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "What did I buy for my sister's birthday gift?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.temporal);
        assert!(!plan.intent.preference);
        assert!(!ix.turns[2].is_question);
        assert!(!ix.turns[4].is_question);
        // Slot 1 is the mid-session overview (turn 2). The next user (turn 4)
        // is already in the top 5, declarative, and at least as on-scene.
        let mut list = vec![2usize, 3, 4, 5, 6];
        let t_scene = vec![0.5f32, 0.4, 1.0, 0.8, 1.05, 0.9, 0.2];
        assert!(apply_c2_elaboration_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![4, 2, 3, 5, 6]);
        // Equal scene is enough (at least as on-scene as the head).
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(apply_c2_elaboration_seat(&ix, &plan, &mut list, &[0.5, 0.4, 1.0, 0.8, 1.0, 0.9, 0.2]));
        assert_eq!(list, vec![4, 2, 3, 5, 6]);
        // Below the head's scene score: no swap.
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(
            &ix,
            &plan,
            &mut list,
            &[0.5, 0.4, 1.0, 0.8, 0.99, 0.9, 0.2]
        ));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Session opener at slot 1: no swap.
        let mut list = vec![0usize, 1, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 6]);
        // Next user not already in the top 5: membership must not change.
        let mut list = vec![2usize, 3, 5, 6, 1];
        assert!(!apply_c2_elaboration_seat(&ix, &plan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 5, 6, 1]);
        // Preference query: no swap.
        let pq = Query { text: "Can you recommend a birthday gift for my sister?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &pplan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When did I buy my sister's birthday gift?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &tplan, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[sess(
                "D1",
                &[
                    ("Caroline", "I'm planning gifts this year."),
                    ("Melanie", "Nice."),
                    ("Caroline", "Here's a start - I've bought gifts for my sister's birthday, my mom, and my neighbor."),
                    ("Melanie", "That's a good list."),
                    ("Caroline", "For my sister's birthday, I got her a yellow dress and a pair of earrings to match."),
                ],
            )],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![2usize, 3, 4];
        assert!(!apply_c2_elaboration_seat(&ixn, &plann, &mut list, &[0.5, 0.4, 1.0, 0.8, 1.05]));
        assert_eq!(list, vec![2, 3, 4]);
    }


    #[test]
    fn cross_opener_seat_promotes_sibling_session_opener() {
        let ix = build_index(
            &[
                sess(
                    "a",
                    &[
                        ("user", "I'm looking for cocktail ideas."),
                        ("assistant", "A gimlet is a classic."),
                        ("user", "I like the sound of the whiskey sour. Can you tell me more about honey syrup?"),
                    ],
                ),
                sess(
                    "b",
                    &[
                        ("user", "I'm looking for a refreshing summer cocktail that pairs well with paella. I recently made a cucumber gimlet."),
                        ("assistant", "A cucumber gimlet is a great start."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "How many different types of citrus fruits have I used in my cocktail recipes?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(!c2_summary_request(&q.text));
        // a#2 is mid-session; b#0 is a user opener already in the top 5.
        let mut list = vec![2usize, 0, 1, 3, 4];
        let t_scene = vec![0.8f32, 0.7, 1.0, 0.75, 0.6];
        assert!(apply_c2_cross_opener_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![3, 2, 0, 1, 4]);
        // Below the 70% share: no seat.
        let mut list = vec![2usize, 0, 1, 3, 4];
        assert!(!apply_c2_cross_opener_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[0.8, 0.7, 1.0, 0.6, 0.6]
        ));
        assert_eq!(list, vec![2, 0, 1, 3, 4]);
        // Session opener at slot 1: no seat.
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_cross_opener_seat(&ix, &plan, &q.text, &mut list, &[1.0, 0.9, 0.9]));
        assert_eq!(list, vec![0, 2, 3]);
        // Same-session only: no seat.
        let mut list = vec![2usize, 0, 1];
        assert!(!apply_c2_cross_opener_seat(&ix, &plan, &q.text, &mut list, &[0.8, 0.7, 1.0]));
        assert_eq!(list, vec![2, 0, 1]);
        // Digit-bearing head, digit-free opener: keep form, no seat.
        let ixd = build_index(
            &[
                sess(
                    "a",
                    &[
                        ("user", "I'm looking for cocktail ideas."),
                        ("assistant", "A gimlet is a classic."),
                        ("user", "I spent 45 dollars on citrus for the whiskey sour. Can you tell me more about honey syrup?"),
                    ],
                ),
                sess(
                    "b",
                    &[("user", "I'm looking for a refreshing summer cocktail that pairs well with paella.")],
                ),
            ],
            &Params::default(),
        );
        assert!(ixd.turns[2].form_digit);
        assert!(!ixd.turns[3].form_digit);
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_cross_opener_seat(
            &ixd,
            &plan,
            &q.text,
            &mut list,
            &[0.8, 0.7, 1.0, 0.75]
        ));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Summary request: no seat (BEAM summarization shape).
        let sq = Query {
            text: "Can you summarize what I learned about cocktail recipes?".into(),
            date: String::new(),
        };
        assert!(c2_summary_request(&sq.text));
        let splan = plan_query(&ix, &sq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3, 4];
        assert!(!apply_c2_cross_opener_seat(&ix, &splan, &sq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 4]);
        // Preference query: no seat.
        let pq = Query { text: "Can you recommend a cocktail recipe?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3, 4];
        assert!(!apply_c2_cross_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 4]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("Caroline", "I'm looking for cocktail ideas."),
                        ("Melanie", "A gimlet is a classic."),
                        ("Caroline", "I like the sound of the whiskey sour. Can you tell me more about honey syrup?"),
                    ],
                ),
                sess(
                    "D2",
                    &[("Caroline", "I'm looking for a refreshing summer cocktail that pairs well with paella.")],
                ),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_cross_opener_seat(&ixn, &plann, &q.text, &mut list, &[0.8, 0.7, 1.0, 0.75]));
        assert_eq!(list, vec![2, 0, 1, 3]);
    }

    #[test]
    fn same_opener_seat_promotes_same_session_opener() {
        let ix = build_index(
            &[
                sess(
                    "a",
                    &[
                        ("user", "Last Wednesday I took a taxi that cost $12, and the train fare was $5."),
                        ("assistant", "That's a big difference for a daily commute."),
                        ("user", "I'd rather not talk about the book I bought. Last Wednesday's taxi cost me $12, which was more than the train."),
                    ],
                ),
                sess(
                    "b",
                    &[("user", "I spent $40 on a new novel at the bookstore.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "For my daily commute, how much more expensive was the taxi ride compared to the train fare?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(!c2_summary_request(&q.text));
        assert!(!ix.turns[2].is_question);
        assert!(ix.turns[0].form_digit && ix.turns[2].form_digit);
        // a#2 is mid-session declarative; a#0 is the same-session opener already in the top 5.
        let mut list = vec![2usize, 0, 1, 3];
        let t_scene = vec![0.85f32, 0.5, 1.0, 0.7];
        assert!(apply_c2_same_opener_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 1, 3]);
        // Below the 70% share: no seat.
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_same_opener_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[0.6, 0.5, 1.0, 0.7]
        ));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Session opener at slot 1: no seat.
        let mut list = vec![0usize, 2, 1];
        assert!(!apply_c2_same_opener_seat(&ix, &plan, &q.text, &mut list, &[1.0, 0.5, 0.9]));
        assert_eq!(list, vec![0, 2, 1]);
        // Cross-session opener only: no seat.
        let mut list = vec![2usize, 3, 1];
        assert!(!apply_c2_same_opener_seat(&ix, &plan, &q.text, &mut list, &[0.7, 0.85, 1.0]));
        assert_eq!(list, vec![2, 3, 1]);
        // Question head: no seat.
        let ixq = build_index(
            &[sess(
                "a",
                &[
                    ("user", "Last Wednesday I took a taxi that cost $12, and the train fare was $5."),
                    ("assistant", "That's a big difference for a daily commute."),
                    ("user", "How much did last Wednesday's $12 taxi cost compared to the train?"),
                ],
            )],
            &Params::default(),
        );
        assert!(ixq.turns[2].is_question);
        let mut list = vec![2usize, 0, 1];
        assert!(!apply_c2_same_opener_seat(&ixq, &plan, &q.text, &mut list, &[0.85, 0.5, 1.0]));
        assert_eq!(list, vec![2, 0, 1]);
        // Question opener: no seat. A later named declarative (WH-person shape)
        // must not yield to an opening question.
        let ixo = build_index(
            &[sess(
                "a",
                &[
                    ("user", "Did you say we could work on the design project at the studio tomorrow?"),
                    ("assistant", "Yes, we can."),
                    ("user", "I planned to work on the design project with Elena Morales at the studio."),
                ],
            )],
            &Params::default(),
        );
        assert!(ixo.turns[0].is_question);
        assert!(!ixo.turns[2].is_question);
        let mut list = vec![2usize, 0, 1];
        assert!(!apply_c2_same_opener_seat(&ixo, &plan, &q.text, &mut list, &[0.85, 0.5, 1.0]));
        assert_eq!(list, vec![2, 0, 1]);
        // Digit-bearing head, digit-free opener: keep form, no seat.
        let ixd = build_index(
            &[sess(
                "a",
                &[
                    ("user", "Last Wednesday I took a taxi, and the train fare was cheaper."),
                    ("assistant", "That's a big difference for a daily commute."),
                    ("user", "I'd rather not talk about the book I bought. Last Wednesday's taxi cost me $12, which was more than the train."),
                ],
            )],
            &Params::default(),
        );
        assert!(ixd.turns[2].form_digit);
        assert!(!ixd.turns[0].form_digit);
        let mut list = vec![2usize, 0, 1];
        assert!(!apply_c2_same_opener_seat(&ixd, &plan, &q.text, &mut list, &[0.85, 0.5, 1.0]));
        assert_eq!(list, vec![2, 0, 1]);
        // Summary request: no seat.
        let sq = Query {
            text: "Can you summarize what I learned about my commute costs?".into(),
            date: String::new(),
        };
        assert!(c2_summary_request(&sq.text));
        let splan = plan_query(&ix, &sq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_same_opener_seat(&ix, &splan, &sq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Preference query: no seat.
        let pq = Query { text: "Can you recommend a cheaper commute?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_same_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[sess(
                "D1",
                &[
                    ("Caroline", "Last Wednesday I took a taxi that cost $12, and the train fare was $5."),
                    ("Melanie", "That's a big difference for a daily commute."),
                    ("Caroline", "I'd rather not talk about the book I bought. Last Wednesday's taxi cost me $12, which was more than the train."),
                ],
            )],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![2usize, 0, 1];
        assert!(!apply_c2_same_opener_seat(&ixn, &plann, &q.text, &mut list, &[0.85, 0.5, 1.0]));
        assert_eq!(list, vec![2, 0, 1]);
    }

    #[test]
    fn queried_speaker_seat_promotes_named_speakers_next_reply() {
        let ix = build_index(
            &[sess(
                "D7",
                &[
                    ("caroline", "Do you have any pets? What are their names?"),
                    ("melanie", "I have Oliver, Luna, and Bailey."),
                    ("caroline", "Those are lovely names."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "What are Melanie's pets' names?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(ix.turns[0].is_question);
        assert!(!c2_query_mentions_speaker(&q.text, "caroline"));
        assert!(c2_query_mentions_speaker(&q.text, "melanie"));
        assert!(q.text.contains("Melanie's"));
        let mut list = vec![0usize, 1, 2];
        let t_scene = vec![1.0f32, 0.8, 0.5];
        assert!(apply_c2_queried_speaker_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // Declarative head: no seat.
        let ixd = build_index(
            &[sess(
                "D7",
                &[
                    ("caroline", "I was just asking about pets earlier."),
                    ("melanie", "I have Oliver, Luna, and Bailey."),
                ],
            )],
            &Params::default(),
        );
        assert!(!ixd.turns[0].is_question);
        let mut list = vec![0usize, 1];
        assert!(!apply_c2_queried_speaker_seat(&ixd, &plan, &q.text, &mut list, &[1.0, 0.8]));
        assert_eq!(list, vec![0, 1]);
        // Head speaker is already named in the query: no seat.
        let hq = Query { text: "What did Caroline ask about Melanie's pets?".into(), date: String::new() };
        let hplan = plan_query(&ix, &hq, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_queried_speaker_seat(&ix, &hplan, &hq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);
        // Conjoined speaker ("X and Y"): no seat.
        let aq = Query {
            text: "What did Melanie and Anna discuss while watching the sunset?".into(),
            date: String::new(),
        };
        assert!(c2_speaker_conjoined(&aq.text, "melanie"));
        let aplan = plan_query(&ix, &aq, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_queried_speaker_seat(&ix, &aplan, &aq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);
        // Preference query: no seat.
        let pq = Query { text: "What is Melanie's favorite pet?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_queried_speaker_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);
        // WH-person: no seat.
        let wq = Query { text: "Who told Melanie about the pets?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_queried_speaker_seat(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);
        // user/assistant chats are left alone.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "Do you have any pets? What are their names?"),
                    ("assistant", "I have Oliver, Luna, and Bailey."),
                ],
            )],
            &Params::default(),
        );
        let uq = Query { text: "What are Melanie's pets' names?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![0usize, 1];
        assert!(!apply_c2_queried_speaker_seat(&ixu, &uplan, &uq.text, &mut list, &[1.0, 0.8]));
        assert_eq!(list, vec![0, 1]);
        // Next turn not in the top 5: no seat.
        let mut list = vec![0usize, 2];
        assert!(!apply_c2_queried_speaker_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2]);
    }

}

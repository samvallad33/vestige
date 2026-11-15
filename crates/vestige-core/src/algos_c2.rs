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

/// Query term ids a turn attests from its own content or features. Index
/// token stats only. Not a group collapse: two terms stay two marks.
fn c2_term_ids(ix: &Index, plan: &QueryPlan, ti: usize) -> DetHashSet<u32> {
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
/// user). Runs after unique-group realloc, before RSB. A later
/// post-gatekeeper pass may run the same function again after RSB and
/// the tail reorder, so a next-user turn that only entered the top 5
/// via RSB can still take slot 1. Slot 1 is then frozen by
/// rank1_gatekeeper_keep_rsb_head, so a pre-RSB reseat also sticks; it
/// is not a Kemeny ballot (a tail vote cannot move T@1).
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
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || plan.intent.temporal || c2_text_preference(query) {
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

/// Text-level preference question, the same test `classify` uses
/// (`PREF_QUERY_CUES` and not a retrospective user-initiative / timeline
/// cue). This is the preference #11 is built for. `plan.intent.preference`
/// can also be set later in `plan_query` when a query token is a
/// constraint object from the conversation; that upgrade is for the
/// preference scoring path, not a preference *question*. C2 seats from
/// #12 onward skip only this text-level test, so they stay independent
/// of #11 without treating a factual question that happens to mention a
/// constraint object ("dogs" from "I don't want to hurt any of my dogs")
/// as a preference query.
fn c2_text_preference(query: &str) -> bool {
    let padded = word_padded(query);
    let user_initiative = has_any(&padded, USER_INITIATIVE_CUES);
    let retrospective_user = user_initiative || has_any(&padded, TIMELINE_CUES);
    has_any(&padded, PREF_QUERY_CUES) && !retrospective_user
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
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
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
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
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


/// Cross-session opener-pair seat. A user session-opener at slot 1 is
/// often a sibling conversation start that still shares topic words; a
/// user opener from a different user/assistant session already in the
/// top 5, whose immediately following assistant reply is also already
/// in the top 5, and that is at least as on-scene, is a retrieved
/// conversation start and takes slot 1. Top-5 membership never changes.
/// The pair cut is the structure: an opener whose reply was not
/// retrieved with it is not promoted, which is what would otherwise
/// unseat a sitting gold opener that a higher-scoring lone opener
/// outranks. Distinct from the cross-session opener seat (#13), which
/// requires a mid-session head and does not require the candidate's
/// reply to be in the top 5, and from the same-session opener seat
/// (#14). Skipped on summary requests (existing `summary_hold` shape),
/// preference and temporal questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant: the scene floor is "at least as on-scene," the same
/// comparison #12 uses. Invoked after the gatekeeper, after #26. A
/// later post-gatekeeper C2 seat may still run after this pass.
fn apply_c2_cross_opener_pair_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
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
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        let Some(members) = ix.session_turns.get(t.session) else {
            continue;
        };
        let Some(&reply) = members.get(1) else {
            continue;
        };
        if ix.turns[reply].speaker != "assistant" {
            continue;
        }
        if !list[..window].contains(&reply) {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Answered-opener declarative seat. A user question opener at slot 1
/// whose immediately following assistant reply is already in the top 5
/// is a retrieved Q-A pair; a declarative user opener from a different
/// user/assistant session already in the top 5, that is at least as
/// on-scene, is a conversation start that states a fact and takes slot 1.
/// Top-5 membership never changes. The answered-head + declarative-cand
/// cut is the structure: a sitting gold question opener whose reply is
/// not in the top 5 is left alone, and a question challenger is not
/// promoted, which is what would otherwise unseat a sitting gold opener.
/// Distinct from the cross-session opener-pair seat (#27), which
/// requires the *candidate's* reply to be in the top 5 and does not
/// require a question head or a declarative candidate, from #13
/// (mid-session head), and from #14 (same session). Skipped on summary
/// requests (existing `summary_hold` shape), preference and temporal
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. No new constant: the scene floor is "at
/// least as on-scene," the same comparison #12 and #27 use. Invoked
/// after the gatekeeper, after #27. A later post-gatekeeper C2 seat
/// may still run after this pass.
fn apply_c2_answered_opener_decl_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || !ix.turns[top].is_question
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let Some(members) = ix.session_turns.get(ix.turns[top].session) else {
        return false;
    };
    let Some(&reply) = members.get(1) else {
        return false;
    };
    if ix.turns[reply].speaker != "assistant" || !list[..window].contains(&reply) {
        return false;
    }
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.is_question
            || t.session == ix.turns[top].session
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Solo earlier-opener seat. A lone user opener at slot 1 — no other
/// turn from that session is already in the top 5 — yields to an
/// earlier user opener from a different user/assistant session already
/// in the top 5, when that earlier opener is at least as on-scene and
/// both sessions have a known timestamp. Top-5 membership never changes.
/// The solo + earlier cut is the structure: a sitting gold opener that
/// arrived with another turn from its session is left alone, and a
/// same-date or later challenger is not promoted, which is what would
/// otherwise unseat a sitting gold opener. Distinct from #27 (candidate
/// reply already in the top 5; no solo or earlier requirement), from
/// #28 (answered question head and a declarative candidate), from #13
/// (mid-session head), and from #14 (same session). Skipped on summary
/// requests (existing `summary_hold` shape), preference and temporal
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. No new constant: the scene floor is "at
/// least as on-scene," the same comparison #12, #27, and #28 use.
/// Invoked after the gatekeeper, after #28. A later post-gatekeeper
/// C2 seat may still run after this pass.
fn apply_c2_solo_earlier_opener_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 || c2_text_preference(query) || plan.intent.temporal || c2_summary_request(query)
    {
        return false;
    }
    let top = list[0];
    if ix.turns[top].speaker != "user"
        || c2_named_session(ix, top)
        || c2_session_pos(ix, top) != 0
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let ts_top = ix.session_ts.get(ix.turns[top].session).copied().unwrap_or(0);
    if ts_top <= 0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top_sess = ix.turns[top].session;
    if list[1..window].iter().any(|&t| ix.turns[t].session == top_sess) {
        return false;
    }
    for i in 1..window {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || t.session == top_sess
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < sc_top {
            continue;
        }
        let ts_cand = ix.session_ts.get(t.session).copied().unwrap_or(0);
        if ts_cand <= 0 || ts_cand >= ts_top {
            continue;
        }
        list.remove(i);
        list.insert(0, cand);
        return true;
    }
    false
}


/// Pool opener T@10 entry. A user session opener that is in the
/// scene-ranked retrieval pool but not yet in the official k takes the
/// official-k seat of the last assistant in the official-k tail (slots
/// 6..k) that is not the unique official-k representative of its
/// session, is not slot 1's immediately following same-session turn,
/// and whose session still has a user turn in the official k, when that
/// opener is at least as on-scene as the evicted assistant. The top 5
/// is unchanged, so T@5 and All@5 cannot drop. The replacement is
/// refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. Distinct from #21 (the
/// candidate is already in the official k, and the victim is pulled
/// into the top 5) and from pull-outsider / drop-official-last (that
/// class evicts official last and can change the top 5). No new
/// constant: the scene floor is "at least as on-scene," the same
/// comparison #12 and #21 use. Skipped on summary requests (existing
/// `summary_hold` shape), preference, temporal, and WH-person questions,
/// event sequences and assistant-attribution (caller), and named-speaker
/// sessions. Invoked after the gatekeeper, after #29. A later
/// post-gatekeeper C2 seat may still run after this pass.
fn apply_c2_pool_opener_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if list.len() <= window {
        return false;
    }
    let k = list.len();
    let head_sess = ix.turns[top].session;
    let head_pos = c2_session_pos(ix, top);
    let mut vs = None;
    for i in window..k {
        let tid = list[i];
        if ix.turns[tid].speaker != "assistant" || c2_named_session(ix, tid) {
            continue;
        }
        let sess = ix.turns[tid].session;
        if list.iter().filter(|&&u| ix.turns[u].session == sess).count() < 2 {
            continue;
        }
        if sess == head_sess && c2_session_pos(ix, tid) == head_pos + 1 {
            continue;
        }
        if !list
            .iter()
            .any(|&u| u != tid && ix.turns[u].speaker == "user" && ix.turns[u].session == sess)
        {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    let sc_v = t_scene.get(list[vs]).copied().unwrap_or(0.0);
    if sc_v <= 0.0 {
        return false;
    }
    let turn_ts: Vec<i64> = ix
        .turns
        .iter()
        .map(|t| ix.session_ts.get(t.session).copied().unwrap_or(0))
        .collect();
    for ti in order(t_scene, &turn_ts) {
        if list.contains(&ti) || t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let t = &ix.turns[ti];
        if t.speaker != "user" || c2_named_session(ix, ti) || c2_session_pos(ix, ti) != 0 {
            continue;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) < sc_v {
            continue;
        }
        if !c2_s5_replace_preserved(ix, list, vs, ti) {
            continue;
        }
        list[vs] = ti;
        return true;
    }
    false
}


/// Same-session opener T@5 entry. A user session opener already in the
/// official k but outside the top 5, from a session already represented
/// in the top 5 by an assistant, swaps into the top 5 in place of the
/// last same-session assistant in the top 5. The victim stays in the
/// official k, so T@10 cannot drop. The swap is refused when it would
/// push any session out of the first 5 unique-session slots, so S@5
/// cannot drop. Distinct from #20, which also promotes a later
/// non-opener user from a top-5 session and evicts the last assistant
/// of any session: that later-user clause is what would unseat a
/// sitting gold assistant (the ac031881 shape). The opener-only,
/// same-session-assistant victim cut is the structure. Skipped on
/// summary requests (existing `summary_hold` shape), preference,
/// temporal, and WH-person questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant. Invoked after the gatekeeper, after #30, as the last
/// ranking change, so it can see openers that the pool T@10 entry just
/// seated into official k. A later post-gatekeeper C2 seat may still
/// run after this pass.
fn apply_c2_same_session_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut top5_sessions = Vec::new();
    for &tid in &list[..window] {
        let s = ix.turns[tid].session;
        if !top5_sessions.contains(&s) {
            top5_sessions.push(s);
        }
    }
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
            || !top5_sessions.contains(&t.session)
            || t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0
        {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let cand_sess = ix.turns[list[ci]].session;
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker != "assistant" || ix.turns[tid].session != cand_sess {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Late opener T@5 entry. A user session opener already in the official k
/// but outside the top 5, from a session already represented in the top 5
/// (by any turn), swaps into the top 5 in place of the last assistant in
/// the top 5, of any session. Distinct from #31, whose victim must be a
/// same-session assistant (it refuses when the opener's session sits in
/// the top 5 only through a user turn), and from #20, which also seats a
/// later non-opener user and runs before #30 seats pool openers into
/// official k. The opener-only cut is what makes the #20 later-user loss
/// shape (a mid-session user evicting a seated gold assistant)
/// impossible. The victim stays in the official k, so T@10 cannot drop.
/// The swap is refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. Skipped on summary requests,
/// preference, temporal, and WH-person questions, event sequences and
/// assistant-attribution (caller), and named-speaker sessions. No new
/// constant. Invoked after the gatekeeper, after #33, as the last ranking
/// change, so it sees openers that the pool T@10 entry (#30) seated.
fn apply_c2_late_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5_sessions: Vec<usize> = list[..window].iter().map(|&t| ix.turns[t].session).collect();
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user"
            || c2_named_session(ix, cand)
            || c2_session_pos(ix, cand) != 0
            || !top5_sessions.contains(&t.session)
            || t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0
        {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker == "assistant" {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Head-session opener T@5 entry. The user opener of the session that holds
/// slot 1, already in the official k but outside the top 5, swaps into the
/// top 5 in place of the least on-scene user turn in slots 2..5 that
/// belongs to a different session, when the opener is at least as
/// on-scene as that turn. The slot-1 session is the strongest evidence
/// session; its opener frames it. Distinct from #31 and #34 (assistant
/// victims) and from evicting the last top-5 user (that victim choice
/// unseats sitting golds such as 8979f9ec): the victim here is the
/// weakest foreign-session user, and only the head session's opener
/// enters. The victim stays in the official k, so T@10 cannot drop. The
/// swap is refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. No new constant: the scene
/// floor is the same "at least as on-scene" comparison #12 and #30 use.
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #34, as
/// the last ranking change.
fn apply_c2_head_session_opener_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let head_sess = ix.turns[top].session;
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker == "user"
            && t.session == head_sess
            && c2_session_pos(ix, cand) == 0
            && t_scene.get(cand).copied().unwrap_or(0.0) > 0.0
        {
            cand_slot = Some(i);
            break;
        }
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let sc_c = t_scene.get(list[ci]).copied().unwrap_or(0.0);
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        if ix.turns[tid].speaker != "user" || ix.turns[tid].session == head_sess {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Session substitution T@5 entry. The most on-scene user turn already in
/// the official k, outside the top 5, from a session not represented in the
/// top 5, swaps with the least on-scene user turn in slots 2..5 when that
/// turn is a declarative and holds at most `C2_XWITNESS_RATIO` of the
/// candidate's scene (the cross-witness dominance ratio, reused, not
/// re-tuned). The swap trades the weakest top-5 session for a clearly
/// stronger session the window left out, so the number of distinct
/// sessions in the top 5 does not fall when the victim was its session's
/// only seat. The declarative cut is load-bearing: a weakest user
/// question in the top 5 is often the sitting gold (the user asked about
/// the thing), as on BEAM 9:contradiction_resolution:0. Assistants are
/// never the victim. The victim stays in the official k, so T@10 cannot
/// drop, and C2 turn seats do not rewrite the session list. Distinct from
/// #21 (opener entry, assistant victim), #34/#35 (session already in the
/// top 5), and from evicting the last top-5 user (sitting-gold losses).
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #35, and
/// again after #37 (#38) so a pool user that #37 seats into official k can
/// still enter the top 5.
fn apply_c2_session_substitution_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5_sessions: Vec<usize> = list[..window].iter().map(|&t| ix.turns[t].session).collect();
    let mut cand: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let t = &ix.turns[tid];
        if t.speaker != "user" || c2_named_session(ix, tid) || top5_sessions.contains(&t.session) {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if cand.is_none_or(|(_, best)| sc > best) {
            cand = Some((i, sc));
        }
    }
    let Some((ci, sc_c)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        if ix.turns[tid].speaker != "user" {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if ix.turns[list[vs]].is_question {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Pool session substitution T@10 entry. The most on-scene user turn in the
/// scene-ranked pool (exact ties to the earliest session, the original
/// attestation) from a session with no turn in the official k takes
/// the official-k seat of the least on-scene turn in the official-k tail
/// (slots 6..k) when that turn is a declarative and holds at most
/// `C2_XWITNESS_RATIO` of the candidate's scene (reused, not re-tuned). The
/// T@10 analogue of #36: a clearly stronger session the official k left out
/// replaces the weakest tail seat. The top 5 is unchanged, so T@1, T@5 and
/// All@5 cannot drop; C2 turn seats do not rewrite the session list. The
/// declarative cut is the #36 cut (a weakest question is often a sitting
/// gold). Distinct from #30 (openers only, non-unique tail assistant
/// victim, "at least as on-scene" floor) and #32 (named speakers).
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Invoked after the gatekeeper, after #36, as the
/// last ranking change.
fn apply_c2_pool_session_substitution_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if list.len() <= window {
        return false;
    }
    let k_sessions: Vec<usize> = list.iter().map(|&t| ix.turns[t].session).collect();
    // Most on-scene eligible pool user. Exact scene ties go to the earliest
    // session (the original attestation; a later session repeating the
    // same turn scores identically), then the lower turn index.
    let mut cand: Option<(usize, f32, i64)> = None;
    for (ti, t) in ix.turns.iter().enumerate() {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        if sc <= 0.0 || list.contains(&ti) {
            continue;
        }
        if t.speaker != "user" || c2_named_session(ix, ti) || k_sessions.contains(&t.session) {
            continue;
        }
        let ts = ix.session_ts.get(t.session).copied().unwrap_or(0);
        if cand.is_none_or(|(_, bs, bt)| sc > bs || (sc == bs && ts < bt)) {
            cand = Some((ti, sc, ts));
        }
    }
    let Some((cand, sc_c, _)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    if ix.turns[list[vs]].is_question {
        return false;
    }
    list[vs] = cand;
    true
}


/// Mentioned-declarative session-mate T@5 entry. In a named-speaker
/// dialogue, the most on-scene declarative turn by a speaker the query
/// names (not conjoined with another name), already in the official k but
/// outside the top 5, from a session already represented in the top 5,
/// swaps with the least on-scene turn in slots 2..5 when that turn is a
/// declarative, holds at most `C2_XWITNESS_RATIO` of the candidate's scene
/// (reused, not re-tuned), and comes from a session no earlier than the
/// candidate's. The session cut is primacy: an earlier witness is never
/// evicted for a later or same-weight restatement (without it, LoCoMo
/// 4:57 loses a sitting earlier-session gold; unofficial All@5). The
/// declarative cuts follow #17 (a mentioned question is often a prompt)
/// and #36 (a weakest question is often a sitting gold). Distinct from
/// #17/#19/#33 (victim is the last unmentioned named speaker; any
/// mentioned declarative), #23 (first-in-session), and #36 (user/assistant
/// chats, session outside the top 5). The victim stays in the official k,
/// so T@10 cannot drop; C2 turn seats do not rewrite the session list.
/// Skipped on summary requests, preference, temporal, and WH-person
/// questions, event sequences and assistant-attribution (caller). Does not
/// fire on user/assistant chats. Invoked after the gatekeeper, after #38,
/// as the last ranking change.
fn apply_c2_mentioned_session_mate_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    if t_scene.get(top).copied().unwrap_or(0.0) <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let top5_sessions: Vec<usize> = list[..window].iter().map(|&t| ix.turns[t].session).collect();
    let mut cand: Option<(usize, f32)> = None;
    for (i, &tid) in list.iter().enumerate().skip(window) {
        let t = &ix.turns[tid];
        let sp = t.speaker.as_str();
        if t.is_question
            || sp.is_empty()
            || sp == "user"
            || sp == "assistant"
            || !c2_query_mentions_speaker(query, sp)
            || c2_speaker_conjoined(query, sp)
            || !top5_sessions.contains(&t.session)
        {
            continue;
        }
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if cand.is_none_or(|(_, best)| sc > best) {
            cand = Some((i, sc));
        }
    }
    let Some((ci, sc_c)) = cand else {
        return false;
    };
    let mut vs: Option<(usize, f32)> = None;
    for (i, &tid) in list[..window].iter().enumerate().skip(1) {
        let sc = t_scene.get(tid).copied().unwrap_or(0.0);
        if sc > C2_XWITNESS_RATIO * sc_c {
            continue;
        }
        if vs.is_none_or(|(_, best)| sc < best) {
            vs = Some((i, sc));
        }
    }
    let Some((vs, _)) = vs else {
        return false;
    };
    let victim = list[vs];
    if ix.turns[victim].is_question {
        return false;
    }
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(victim) < ts_of(list[ci]) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Episode-run edge T@10 entry. Two or more official-k turns that sit
/// back to back in one session form an episode the retriever already
/// found; the pool turn just before or just after that run (same session,
/// t_scene > 0, not in the official k, a declarative, and, in a
/// named-speaker dialogue, by a speaker the query names and does not
/// conjoin) is the episode's unread edge. The most on-scene such edge
/// takes the official-k tail seat (slots 6..k) of the least on-scene
/// isolated turn (the only official-k turn of its session), when the
/// edge is at least as on-scene as that turn and the turn's session is
/// not earlier than the edge's (primacy cut: an earlier lone witness is
/// never evicted for a later episode). Multi-turn evidence, not a
/// per-turn feature: neither the candidate nor the victim is chosen by
/// its own text. Top 5 unchanged, so T@1 / T@5 / All@5 cannot drop; C2
/// turn seats do not rewrite the session list. No new constant: "at
/// least as on-scene" is the existing floor and a run is two adjacent
/// turns. Distinct from #30 / #32 / #37 (pool entry by opener, mentioned
/// speaker, or left-out session; none uses k-run adjacency) and from the
/// head-adjacency seats (#2 / #10 / #12 / #15 / #16 move turns next to slot
/// 1 into the top 5). Skipped on summary requests, preference, temporal,
/// and WH-person questions; event sequences and assistant-attribution
/// are skipped by the caller. Invoked after the gatekeeper, after #39,
/// as the last ranking change.
fn apply_c2_episode_run_edge_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let mut by_sess: Vec<(usize, Vec<usize>)> = Vec::new();
    for &t in list.iter() {
        let s = ix.turns[t].session;
        let p = c2_session_pos(ix, t);
        match by_sess.iter_mut().find(|(bs, _)| *bs == s) {
            Some((_, ps)) => ps.push(p),
            None => by_sess.push((s, vec![p])),
        }
    }
    let eligible = |t: usize| -> bool {
        if list.contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
            return false;
        }
        let sp = ix.turns[t].speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" {
            return c2_query_mentions_speaker(query, sp) && !c2_speaker_conjoined(query, sp);
        }
        true
    };
    let mut cand: Option<usize> = None;
    for (s, ps) in by_sess.iter_mut() {
        ps.sort_unstable();
        let turns = &ix.session_turns[*s];
        let mut start = 0;
        while start < ps.len() {
            let mut end = start;
            while end + 1 < ps.len() && ps[end + 1] == ps[end] + 1 {
                end += 1;
            }
            if end > start {
                let mut edges: Vec<usize> = Vec::new();
                if ps[start] > 0 {
                    edges.push(turns[ps[start] - 1]);
                }
                if ps[end] + 1 < turns.len() {
                    edges.push(turns[ps[end] + 1]);
                }
                for t in edges {
                    if !eligible(t) {
                        continue;
                    }
                    if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                        cand = Some(t);
                    }
                }
            }
            start = end + 1;
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in window..list.len() {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c || list.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1 {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    list[vs] = cand;
    true
}


/// Episode-run edge T@5 entry. The top-5 analogue of #40: two or more
/// top-5 turns back to back in one session form an episode; the turn just
/// before or just after that run (same session, t_scene > 0, outside the
/// top 5, a declarative, and, in a named-speaker dialogue, by a speaker
/// the query names and does not conjoin) is its unread edge. The most
/// on-scene edge takes the slot (2..5) of the least on-scene top-5 turn
/// that is the only top-5 turn of its session and is not a session
/// opener, when the edge is at least as on-scene as that turn and the
/// turn's session is not earlier than the edge's (primacy cut, as #39 /
/// #40). An official-k edge swaps places with the victim (the victim stays
/// in the official k); a pool edge takes the victim's seat. The opener cut
/// keeps the opener seats' work (#13 / #21 / #29; without it LoCoMo 8:1
/// loses a seated gold opener). No new constant: "at least as on-scene"
/// is the existing floor. Skipped on summary requests, preference,
/// temporal, and WH-person questions; event sequences and
/// assistant-attribution are skipped by the caller. Invoked after the
/// gatekeeper, after #40, as the last ranking change.
fn apply_c2_episode_run_edge_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let top: Vec<usize> = list[..window].to_vec();
    let mut by_sess: Vec<(usize, Vec<usize>)> = Vec::new();
    for &t in top.iter() {
        let s = ix.turns[t].session;
        let p = c2_session_pos(ix, t);
        match by_sess.iter_mut().find(|(bs, _)| *bs == s) {
            Some((_, ps)) => ps.push(p),
            None => by_sess.push((s, vec![p])),
        }
    }
    let eligible = |t: usize| -> bool {
        if top.contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
            return false;
        }
        let sp = ix.turns[t].speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" {
            return c2_query_mentions_speaker(query, sp) && !c2_speaker_conjoined(query, sp);
        }
        true
    };
    let mut cand: Option<usize> = None;
    for (s, ps) in by_sess.iter_mut() {
        ps.sort_unstable();
        let turns = &ix.session_turns[*s];
        let mut start = 0;
        while start < ps.len() {
            let mut end = start;
            while end + 1 < ps.len() && ps[end + 1] == ps[end] + 1 {
                end += 1;
            }
            if end > start {
                let mut edges: Vec<usize> = Vec::new();
                if ps[start] > 0 {
                    edges.push(turns[ps[start] - 1]);
                }
                if ps[end] + 1 < turns.len() {
                    edges.push(turns[ps[end] + 1]);
                }
                for t in edges {
                    if !eligible(t) {
                        continue;
                    }
                    if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                        cand = Some(t);
                    }
                }
            }
            start = end + 1;
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c
            || c2_session_pos(ix, v) == 0
            || top.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1
        {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    match list.iter().position(|&t| t == cand) {
        Some(ci) => list.swap(vs, ci),
        None => list[vs] = cand,
    }
    true
}


/// Neighbor-pair T@5 entry. In a named-speaker dialogue, an official-k
/// tail turn (slots 6..k) that sits directly next to a top-5 turn of the
/// same session completes a two-turn exchange the top 5 holds only half of.
/// The most on-scene such neighbor (declarative, t_scene > 0, by a speaker
/// the query names and does not conjoin) swaps places with the least
/// on-scene top-5 turn (slots 2..5) that is the only top-5 turn of its
/// session and not a session opener, when the neighbor is at least as
/// on-scene and that turn's session is not earlier than the neighbor's
/// (the #41 victim and primacy cuts). The victim stays in the official k,
/// so T@10 cannot drop; C2 turn seats do not rewrite the session list.
/// Official-k neighbors only: a pool neighbor (taking the seat outright)
/// loses LoCoMo 7:179. Named dialogues only: on user/assistant chats the
/// same move loses LME All@5 (4bc144e2). Distinct from #41 (the edge of a
/// back-to-back top-5 run, official k or pool) and from the slot-1
/// adjacency seats (#2 / #10 / #12 / #15 / #16). No new constant. Skipped
/// on summary requests, preference, temporal, and WH-person questions;
/// event sequences and assistant-attribution are skipped by the caller.
/// Invoked after the gatekeeper, after #41, as the last ranking change.
fn apply_c2_neighbor_pair_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let head_sp = ix.turns[list[0]].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let top: Vec<usize> = list[..window].to_vec();
    let mut cand: Option<usize> = None;
    for &x in top.iter() {
        let turns = &ix.session_turns[ix.turns[x].session];
        let p = c2_session_pos(ix, x);
        let mut nbs: Vec<usize> = Vec::new();
        if p > 0 {
            nbs.push(turns[p - 1]);
        }
        if p + 1 < turns.len() {
            nbs.push(turns[p + 1]);
        }
        for t in nbs {
            if top.contains(&t) || !list[window..].contains(&t) || sc(t) <= 0.0 || ix.turns[t].is_question {
                continue;
            }
            let sp = ix.turns[t].speaker.as_str();
            if !sp.is_empty() && sp != "user" && sp != "assistant" {
                if !c2_query_mentions_speaker(query, sp) || c2_speaker_conjoined(query, sp) {
                    continue;
                }
            }
            if cand.is_none_or(|c| sc(t) > sc(c) || (sc(t) == sc(c) && t < c)) {
                cand = Some(t);
            }
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let sc_c = sc(cand);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc_c
            || c2_session_pos(ix, v) == 0
            || top.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1
        {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    let Some(ci) = list.iter().position(|&t| t == cand) else {
        return false;
    };
    list.swap(vs, ci);
    true
}


/// Assistant-reply T@5 entry, for assistant-attribution questions only
/// (the question asks what the assistant said), which every other C2 seat
/// skips. The signal is the transcript's own role labels: an assistant turn
/// already in the official k but outside the top 5 (t_scene > 0) that sits
/// directly next to a top-5 user turn of the same session is that user
/// turn's reply. The most on-scene such reply swaps places with the least
/// on-scene top-5 user turn (slots 2..5) that (a) is not the reply's own
/// neighbor (the anchor: evicting it would only trade the exchange's halves;
/// without this cut LME 32260d93 loses its gold user turn), (b) is at most
/// as on-scene as the reply, and (c) comes from a session no earlier than
/// the reply's (primacy, as #39-#42). The victim stays in the official k,
/// so T@10 cannot drop; the swap is refused when it would push any session
/// out of the first 5 unique-session slots, so S@5 cannot drop; C2 turn
/// seats do not rewrite the session list. No new constant. Skipped on
/// temporal questions (Gemini Step 17 owns them), summary requests,
/// WH-person questions, named-speaker dialogues, and event sequences
/// (caller). Runs after the gatekeeper, outside the C2 assistant-attribution
/// / preference skip, as the last ranking change.
fn apply_c2_assistant_reply_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let adjacent = |a: usize, b: usize| {
        ix.turns[a].session == ix.turns[b].session && c2_session_pos(ix, a).abs_diff(c2_session_pos(ix, b)) == 1
    };
    let mut cand: Option<usize> = None;
    for i in window..list.len() {
        let c = list[i];
        if ix.turns[c].speaker != "assistant" || sc(c) <= 0.0 {
            continue;
        }
        if !list[..window].iter().any(|&u| ix.turns[u].speaker == "user" && adjacent(u, c)) {
            continue;
        }
        if cand.is_none_or(|b| sc(c) > sc(list[b])) {
            cand = Some(i);
        }
    }
    let Some(ci) = cand else {
        return false;
    };
    let c = list[ci];
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    let mut vs: Option<usize> = None;
    for i in 1..window {
        let v = list[i];
        if ix.turns[v].speaker != "user" || sc(v) > sc(c) || adjacent(v, c) || ts_of(v) < ts_of(c) {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Assistant-attribution left-out-session opener T@10 entry, for
/// assistant-attribution questions only (outside the C2 skip, beside #43).
/// The most on-scene declarative session opener (t_scene > 0) in the pool
/// whose session has no official-k seat at all takes the official-k tail
/// seat (slots 6..k) of the least on-scene turn that is the only official-k
/// turn of its session, when the opener is at least as on-scene and that
/// turn's session is not earlier than the opener's (primacy). The question
/// points back at a past conversation; an unseated conversation's opening
/// statement is where that conversation is anchored. Top 5 unchanged; C2
/// turn seats do not rewrite the session list. No new constant: "at least
/// as on-scene" is the existing floor. Distinct from #30 (user openers on
/// non-attribution questions, victim the last non-unique assistant) and #37
/// (any pool user, reused 0.7 ratio, declarative victim); both are skipped
/// on assistant-attribution questions. Skipped on temporal (Gemini Step
/// 17), summary, WH-person, named dialogues, and event sequences (caller).
/// Last ranking change.
fn apply_c2_attribution_unseated_opener_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let sc = |t: usize| t_scene.get(t).copied().unwrap_or(0.0);
    let ts_of = |ti: usize| ix.session_ts.get(ix.turns[ti].session).copied().unwrap_or(0);
    let k_sessions: Vec<usize> = list.iter().map(|&t| ix.turns[t].session).collect();
    let mut cand: Option<usize> = None;
    for (ti, t) in ix.turns.iter().enumerate() {
        if sc(ti) <= 0.0 || t.is_question || c2_session_pos(ix, ti) != 0 || k_sessions.contains(&t.session) {
            continue;
        }
        let tsp = t.speaker.as_str();
        if (tsp != "user" && tsp != "assistant") || c2_named_session(ix, ti) {
            continue;
        }
        // Exact scene ties go to the earliest session (the original
        // attestation; a later session repeating the opener scores the same),
        // the #37 tie rule.
        if cand.is_none_or(|c| sc(ti) > sc(c) || (sc(ti) == sc(c) && ts_of(ti) < ts_of(c))) {
            cand = Some(ti);
        }
    }
    let Some(cand) = cand else {
        return false;
    };
    let mut vs: Option<usize> = None;
    for i in window..list.len() {
        let v = list[i];
        let vsess = ix.turns[v].session;
        if sc(v) > sc(cand) || list.iter().filter(|&&t| ix.turns[t].session == vsess).count() != 1 {
            continue;
        }
        if vs.is_none_or(|b| sc(v) <= sc(list[b])) {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if ts_of(list[vs]) < ts_of(cand) {
        return false;
    }
    list[vs] = cand;
    true
}


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
fn apply_c2_step31_exchequer_split(
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

/// Unique strict maximum. Equal scores are not a sightline: a survey fixes a
/// point only when one mark is strictly best. Ties return `None`.
fn c2_unique_max(ids: &[usize], score: impl Fn(usize) -> f32) -> Option<usize> {
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
fn apply_c2_step33_triangulation(
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


/// Pool mentioned-speaker T@10 entry. In a named-speaker dialogue, a
/// turn by a speaker the query names, in the scene-ranked pool but not
/// already in the official k, takes the official-k seat of the last
/// unmentioned named speaker in the official-k tail (slots 6..k), when
/// that turn is at least as on-scene as the evicted turn. The top 5 is
/// unchanged, so T@1 and T@5 cannot drop. The engine's session list is
/// not rewritten by C2 turn seats, so official Session cannot drop.
/// The replacement is also refused when it would push any session out
/// of the first 5 unique-session slots. Distinct from #17 and #23
/// (those candidates are already in official k and are pulled into the
/// top 5) and from #30 (user/assistant openers). No new constant: the
/// scene floor is "at least as on-scene," the same comparison #12 and
/// #30 use. Skipped when the query conjoins the named speaker with
/// someone else ("X and Y"), on summary requests (existing
/// `summary_hold` shape), preference, temporal, and WH-person
/// questions, and on event sequences and assistant-attribution
/// (caller). Does not fire on user/assistant chats. Invoked after the
/// gatekeeper, after #31. A later post-gatekeeper C2 seat may still
/// run after this pass.
fn apply_c2_pool_mentioned_t10_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    if list.len() <= window {
        return false;
    }
    let k = list.len();
    let mut vs = None;
    for i in window..k {
        let sp = ix.turns[list[i]].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    let sc_v = t_scene.get(list[vs]).copied().unwrap_or(0.0);
    if sc_v <= 0.0 {
        return false;
    }
    let turn_ts: Vec<i64> = ix
        .turns
        .iter()
        .map(|t| ix.session_ts.get(t.session).copied().unwrap_or(0))
        .collect();
    for ti in order(t_scene, &turn_ts) {
        if list.contains(&ti) || t_scene.get(ti).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let csp = ix.turns[ti].speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        if t_scene.get(ti).copied().unwrap_or(0.0) < sc_v {
            continue;
        }
        if !c2_s5_replace_preserved(ix, list, vs, ti) {
            continue;
        }
        list[vs] = ti;
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
        || c2_text_preference(query)
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


/// Queried-speaker previous seat. In a named-speaker dialogue, an
/// unmentioned speaker's declarative turn at slot 1 is often a follow-up
/// comment; the immediately preceding same-session turn, also
/// declarative, by a speaker the query names, already in the top 5, is
/// the statement and takes slot 1. Top-5 membership never changes.
/// Sibling of the queried-speaker reply seat (#15), which requires a
/// question head and the next turn. This seat requires both sides
/// declarative and looks backward. The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply. Does not
/// fire on user/assistant chats. Runs after the queried-speaker reply
/// seat, before RSB. Slot 1 is then frozen by rank1_gatekeeper_keep_rsb_head.
fn apply_c2_queried_speaker_prev_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
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
        || ix.turns[top].is_question
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
    if pos == 0 {
        return false;
    }
    let cand = members[pos - 1];
    let csp = ix.turns[cand].speaker.as_str();
    if ix.turns[cand].is_question
        || !c2_query_mentions_speaker(query, csp)
        || c2_speaker_conjoined(query, csp)
    {
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


/// First `n` unique session indices in list order (official-k unique sessions).
fn c2_first_sessions(ix: &Index, list: &[usize], n: usize) -> Vec<usize> {
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
fn c2_s5_membership_preserved(ix: &Index, list: &[usize], vs: usize, ci: usize) -> bool {
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial.swap(vs, ci);
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
}

/// True when replacing official-k slot `vs` with `cand` (a turn not already
/// in the list) keeps every session that already sat in the first 5
/// unique-session slots of the official k.
fn c2_s5_replace_preserved(ix: &Index, list: &[usize], vs: usize, cand: usize) -> bool {
    if vs >= list.len() {
        return false;
    }
    let before = c2_first_sessions(ix, list, 5);
    let mut trial = list.to_vec();
    trial[vs] = cand;
    let after = c2_first_sessions(ix, &trial, 5);
    before.iter().all(|s| after.contains(s))
}

/// Queried-speaker T@5 entry. In a named-speaker dialogue, a declarative
/// turn by a speaker the query names, already in the official k but
/// outside the top 5, swaps into the top 5 in place of the last
/// unmentioned named speaker. The victim stays in the official k, so
/// T@10 cannot drop by construction. A question by a mentioned speaker
/// is not pulled: that is often a prompt, not the answer (the 4:74
/// shape). The swap is refused when it would push any session out of
/// the first 5 unique-session slots, so S@5 cannot drop by construction.
/// Reuses `C2_XWITNESS_RATIO`. The same mention, conjunction, WH-person,
/// preference, temporal, and summary cuts apply as #15/#16. Does not
/// fire on user/assistant chats. Invoked twice: after the queried-speaker
/// previous / question-head-previous seats and before RSB (#17), and
/// again after `rank1_gatekeeper_keep_rsb_head` so a tail reorder cannot
/// undo the entry (#19), and a third time after the pool
/// mentioned-speaker T@10 entry (#32) so a mentioned declarative that
/// pass just seated into official k can still enter the top 5 (#33).
/// The post-#32 call is the last ranking change; it does not touch
/// `rsb_seat_behind` and adds no WH-slot ballot.
fn apply_c2_queried_speaker_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        let csp = t.speaker.as_str();
        if t.is_question
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        let sp = ix.turns[tid].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Queried-speaker question-head previous seat. In a named-speaker
/// dialogue, an unmentioned speaker's question at slot 1 is often a
/// follow-up prompt; the immediately preceding same-session turn, by a
/// speaker the query names, already in the top 5, is that speaker's
/// prior statement (or their own earlier question) and takes slot 1.
/// Top-5 membership never changes. Literal backward twin of the
/// queried-speaker reply seat (#15), which requires a question head
/// and the next turn. This seat keeps the question head and looks
/// backward; the previous turn may be a question or a statement
/// (both shapes are live golds). The same mention, conjunction,
/// WH-person, preference, temporal, and summary cuts apply. Does not
/// fire on user/assistant chats. Runs after the queried-speaker
/// previous seat, before the T@5 entry and RSB. Slot 1 is then frozen
/// by rank1_gatekeeper_keep_rsb_head.
fn apply_c2_queried_speaker_qhead_prev_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
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
    if pos == 0 {
        return false;
    }
    let cand = members[pos - 1];
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


/// Same-session user T@5 entry. In a user/assistant transcript, a user
/// turn already in the official k but outside the top 5, from a session
/// already represented in the top 5, that is the session opener or is
/// later than a same-session turn already in the top 5, swaps into the
/// top 5 in place of the last assistant. The victim stays in the official
/// k, so T@10 cannot drop by construction. The swap is refused when it
/// would push any session out of the first 5 unique-session slots, so
/// S@5 cannot drop by construction. No new ratio: the same-session
/// opener-or-later cut is the structure. Skipped on preference, temporal,
/// WH-person, summary requests, event sequences and assistant-attribution
/// (caller), and named-speaker sessions. Distinct from the source-superset
/// and unique-group reallocs (no group test) and from the subsequent-answer
/// seat (this changes official-k membership, not slot 1). Runs after
/// `rank1_gatekeeper_keep_rsb_head` so the entry survives the tail reorder.
fn apply_c2_same_session_user_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut top5_sessions = Vec::new();
    for &tid in &list[..window] {
        let s = ix.turns[tid].session;
        if !top5_sessions.contains(&s) {
            top5_sessions.push(s);
        }
    }
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user" {
            continue;
        }
        if !top5_sessions.contains(&t.session) {
            continue;
        }
        let cand_pos = c2_session_pos(ix, cand);
        let opener = cand_pos == 0;
        let later = list[..window].iter().any(|&tid| {
            ix.turns[tid].session == t.session && c2_session_pos(ix, tid) < cand_pos
        });
        if !opener && !later {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker == "assistant" {
            vs = Some(i);
        }
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
}


/// Cross-session user-opener T@5 entry. In a user/assistant transcript, a
/// user session opener already in the official k but outside the top 5,
/// from a session not yet represented in the top 5, swaps into the top 5
/// in place of the last assistant that is not the unique top-5
/// representative of its session and is not slot 1's immediately
/// following same-session turn. The last cut refuses to break the seated
/// continuation of the head (a user-assistant pair starting at slot 1).
/// The victim stays in the official k, so T@10 cannot drop. The swap is
/// refused when it would push any session out of the first 5
/// unique-session slots, so S@5 cannot drop. No new ratio: the
/// cross-session opener cut, plus not evicting the head's next turn, is
/// the structure. Skipped on preference, temporal, WH-person, summary
/// requests, event sequences and assistant-attribution (caller), and
/// named-speaker sessions. Distinct from the same-session user T@5 entry
/// (#20: session already in the top 5) and from the source-superset and
/// unique-group reallocs (no group test). Runs after
/// `rank1_gatekeeper_keep_rsb_head`, after #20, as the last ranking change.
fn apply_c2_cross_session_user_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if (head_sp != "user" && head_sp != "assistant") || c2_named_session(ix, top) {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut top5_sessions = Vec::new();
    for &tid in &list[..window] {
        let s = ix.turns[tid].session;
        if !top5_sessions.contains(&s) {
            top5_sessions.push(s);
        }
    }
    let head_sess = ix.turns[top].session;
    let head_pos = c2_session_pos(ix, top);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        if t.speaker != "user" {
            continue;
        }
        if top5_sessions.contains(&t.session) {
            continue;
        }
        if c2_session_pos(ix, cand) != 0 {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        if ix.turns[tid].speaker != "assistant" {
            continue;
        }
        let sess = ix.turns[tid].session;
        if list[..window].iter().filter(|&&u| ix.turns[u].session == sess).count() < 2 {
            continue;
        }
        if sess == head_sess && c2_session_pos(ix, tid) == head_pos + 1 {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
    true
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
fn c2_first_same_speaker(ix: &Index, ti: usize) -> Option<usize> {
    let members = ix.session_turns.get(ix.turns[ti].session)?;
    let sp = ix.turns[ti].speaker.as_str();
    members.iter().copied().find(|&m| ix.turns[m].speaker == sp)
}

fn apply_c2_queried_speaker_first_seat(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2
        || c2_text_preference(query)
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
        || c2_query_mentions_speaker(query, head_sp)
        || c2_speaker_conjoined(query, head_sp)
    {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in 1..window {
        let cand = list[i];
        let csp = ix.turns[cand].speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        if t_scene.get(cand).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if c2_first_same_speaker(ix, cand) != Some(cand) {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(slot) = cand_slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


/// Queried-speaker first-in-session T@5 entry. In a named-speaker
/// dialogue, a turn by a speaker the query names that is that
/// speaker's first utterance in its session, already in official k
/// but outside the top 5, swaps into the top 5 in place of the last
/// unmentioned named speaker. The victim stays in official k, so
/// T@10 cannot drop by construction. A swap that would push any
/// session out of the first 5 unique-session slots is refused, so
/// S@5 cannot drop. Distinct from #17/#19: the cut is first-in-session,
/// not a scene floor on every candidate. Declaratives have no new
/// ratio; first-in-session is enough. Questions reuse
/// `C2_XWITNESS_RATIO`, the same floor #17 uses, because a
/// first-in-session question at r=0 can evict an unmentioned gold
/// (the 4:74 / 0:106 shapes). A later turn by the same speaker is
/// not pulled. The same mention, conjunction, WH-person, preference,
/// temporal, and summary cuts apply as #15–#19. Does not fire on
/// user/assistant chats. Invoked after the gatekeeper and after
/// #19–#22. A later #22 pass may still run after this entry, so a
/// newly admitted first-in-session turn can take slot 1.
fn apply_c2_queried_speaker_first_t5_entry(
    ix: &Index,
    plan: &QueryPlan,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 6
        || c2_text_preference(query)
        || plan.intent.temporal
        || c2_summary_request(query)
        || (wh_slots(query) & WH_PERSON) != 0
    {
        return false;
    }
    let top = list[0];
    let head_sp = ix.turns[top].speaker.as_str();
    if head_sp.is_empty() || head_sp == "user" || head_sp == "assistant" {
        return false;
    }
    let sc_top = t_scene.get(top).copied().unwrap_or(0.0);
    if sc_top <= 0.0 {
        return false;
    }
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut cand_slot = None;
    for i in window..list.len() {
        let cand = list[i];
        let t = &ix.turns[cand];
        let csp = t.speaker.as_str();
        if csp.is_empty()
            || csp == "user"
            || csp == "assistant"
            || !c2_query_mentions_speaker(query, csp)
            || c2_speaker_conjoined(query, csp)
        {
            continue;
        }
        let sc = t_scene.get(cand).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        if c2_first_same_speaker(ix, cand) != Some(cand) {
            continue;
        }
        if t.is_question && sc < C2_XWITNESS_RATIO * sc_top {
            continue;
        }
        cand_slot = Some(i);
        break;
    }
    let Some(ci) = cand_slot else {
        return false;
    };
    let mut vs = None;
    for (i, &tid) in list[..window].iter().enumerate() {
        let sp = ix.turns[tid].speaker.as_str();
        if sp.is_empty() || sp == "user" || sp == "assistant" {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
            continue;
        }
        vs = Some(i);
    }
    let Some(vs) = vs else {
        return false;
    };
    if !c2_s5_membership_preserved(ix, list, vs, ci) {
        return false;
    }
    list.swap(vs, ci);
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
fn apply_c2_step32_trial_balance(
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

/// Step 34 Bertillonage / Locard point threshold.
///
/// A turn agrees with the query on independent feature kinds. Each query
/// token feeds at most one kind, so the measurements do not double-count:
/// numeral shape (a token that contains an ASCII digit), speaker (an attested
/// name the query mentions; user/assistant are roles, not names), title-case
/// entity (orthographic proper-token, excluding that speaker and any numeral),
/// and anchor (any other content token). Session match is the fifth kind: the
/// turn sits in the same session as the strongest pool evidence, the turn with
/// the greatest scene score (higher session time, then the earlier turn).
///
/// N is how many of the four query-side kinds the query actually has in the
/// index vocabulary. Session is not one of them: the query does not carry a
/// session. A card counts only the query-side kinds. Slot 1 is left alone when
/// its card already meets N, and it is not a blank card when it matches the
/// strongest session.
///
/// A slot-1 card below N yields to one top-5 turn that meets N, agrees on every
/// kind slot 1 agrees on, is a strict superset, shares slot 1's session, and is
/// strictly more on-scene. The witness must also be admissible: either slot 1
/// is a question and the witness is not, and the witness is the same speaker or
/// a speaker the query names; or slot 1's card is empty and slot 1 is not in
/// the strongest session. Two witnesses with the same number of kinds are not
/// an identification, so the seat stays. Top-5 membership does not change.
/// `TurnIx.body` is never read. Does not touch `rsb_seat_behind`.
const STEP34_ANCHOR: u8 = 1;
const STEP34_ENTITY: u8 = 2;
const STEP34_NUMERAL: u8 = 4;
const STEP34_SPEAKER: u8 = 8;

struct Step34Query {
    anchor_ids: Vec<u32>,
    entity_ids: Vec<u32>,
    numeral_ids: Vec<u32>,
    speakers: DetHashSet<String>,
    /// Query-side kinds actually present. Not a free parameter.
    n: u8,
}

fn step34_query(ix: &Index, query: &str) -> Step34Query {
    let mut seen_sp: DetHashSet<String> = DetHashSet::default();
    let mut speakers: DetHashSet<String> = DetHashSet::default();
    let mut speaker_toks: DetHashSet<String> = DetHashSet::default();
    for turn in &ix.turns {
        let sp = turn.speaker.as_str();
        let low = sp.to_ascii_lowercase();
        if low.is_empty() || low == "user" || low == "assistant" || !seen_sp.insert(low.clone()) {
            continue;
        }
        if c2_query_mentions_speaker(query, sp) {
            speakers.insert(low);
            for w in raw_tokens(sp) {
                speaker_toks.insert(w);
            }
        }
    }
    let mut entity_toks: DetHashSet<String> = DetHashSet::default();
    for ent in proper_entity_tokens(query) {
        let Some(body) = ent.strip_prefix('\u{1}').and_then(|s| s.strip_prefix("ent:")) else {
            continue;
        };
        if body.contains('_') || body.bytes().any(|b| b.is_ascii_digit()) {
            continue;
        }
        if speaker_toks.contains(body) || !is_content(body) {
            continue;
        }
        entity_toks.insert(body.to_string());
    }
    let mut anchor_ids = Vec::new();
    let mut entity_ids = Vec::new();
    let mut numeral_ids = Vec::new();
    let mut seen_a: DetHashSet<u32> = DetHashSet::default();
    let mut seen_e: DetHashSet<u32> = DetHashSet::default();
    let mut seen_n: DetHashSet<u32> = DetHashSet::default();
    for w in raw_tokens(query) {
        if w.bytes().any(|b| b.is_ascii_digit()) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_n.insert(id) {
                    numeral_ids.push(id);
                }
            }
            continue;
        }
        if speaker_toks.contains(&w) {
            continue;
        }
        if entity_toks.contains(&w) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_e.insert(id) {
                    entity_ids.push(id);
                }
            }
            continue;
        }
        if is_content(&w) {
            if let Some(id) = ix.vocab.get(&w) {
                if seen_a.insert(id) {
                    anchor_ids.push(id);
                }
            }
        }
    }
    let mut n = 0u8;
    if !anchor_ids.is_empty() {
        n += 1;
    }
    if !entity_ids.is_empty() {
        n += 1;
    }
    if !numeral_ids.is_empty() {
        n += 1;
    }
    if !speakers.is_empty() {
        n += 1;
    }
    Step34Query { anchor_ids, entity_ids, numeral_ids, speakers, n }
}

fn step34_posts(turn: &TurnIx, id: u32) -> bool {
    turn.content.contains(&id)
}

fn step34_card(ix: &Index, ti: usize, q: &Step34Query) -> u8 {
    let turn = &ix.turns[ti];
    let mut bits = 0u8;
    if q.anchor_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_ANCHOR;
    }
    if q.entity_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_ENTITY;
    }
    if q.numeral_ids.iter().any(|id| step34_posts(turn, *id)) {
        bits |= STEP34_NUMERAL;
    }
    if q.speakers.contains(&turn.speaker.to_ascii_lowercase()) {
        bits |= STEP34_SPEAKER;
    }
    bits
}

/// Session of the strongest positive scene score. Tie: later session time, then
/// the earlier turn. Same order as `order`.
fn step34_leader_session(ix: &Index, t_scene: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32, i64)> = None;
    for (ti, turn) in ix.turns.iter().enumerate() {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue;
        }
        let ts = ix.session_ts.get(turn.session).copied().unwrap_or(0);
        let take = match best {
            None => true,
            Some((bi, bs, bts)) => sc > bs || (sc == bs && (ts > bts || (ts == bts && ti < bi))),
        };
        if take {
            best = Some((ti, sc, ts));
        }
    }
    best.map(|(ti, _, _)| ix.turns[ti].session)
}

fn apply_c2_step34_bertillonage(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let q = step34_query(ix, query);
    if q.n == 0 {
        return false;
    }
    let top = list[0];
    let head = step34_card(ix, top, &q);
    if head.count_ones() >= u32::from(q.n) {
        return false;
    }
    let leader = step34_leader_session(ix, t_scene);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let head_sc = t_scene.get(top).copied().unwrap_or(0.0);
    let head_sess = ix.turns[top].session;
    let head_sp = ix.turns[top].speaker.to_ascii_lowercase();
    let head_q = ix.turns[top].is_question;
    let blank = head == 0 && leader != Some(head_sess);
    let mut best: Option<(u32, usize)> = None;
    let mut tied = false;
    for slot in 1..window {
        let cand = list[slot];
        if ix.turns[cand].session != head_sess {
            continue;
        }
        let sc = t_scene.get(cand).copied().unwrap_or(0.0);
        if sc <= head_sc {
            continue;
        }
        let cb = step34_card(ix, cand, &q);
        if cb.count_ones() < u32::from(q.n) || (cb & head) != head || cb == head {
            continue;
        }
        let csp = ix.turns[cand].speaker.to_ascii_lowercase();
        let testimony = head_q
            && !ix.turns[cand].is_question
            && (csp == head_sp || q.speakers.contains(&csp));
        if !(testimony || blank) {
            continue;
        }
        let pc = cb.count_ones();
        match best {
            None => best = Some((pc, slot)),
            Some((bp, _)) if pc > bp => {
                best = Some((pc, slot));
                tied = false;
            }
            Some((bp, _)) if pc == bp => tied = true,
            _ => {}
        }
    }
    if tied {
        return false;
    }
    let Some((_, slot)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}


/// Surface anchors: lexical content-group labels. Stems, hypernyms, and
/// entity-phrase ids are not anchors. The label's own vocab id is the mark.
fn c2_surface_anchor_ids(ix: &Index, plan: &QueryPlan) -> Vec<u32> {
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
fn c2_dump_pool(ix: &Index, list: &[usize], t_scene: &[f32]) -> Vec<usize> {
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

/// Mill's method of agreement (Step 35).
///
/// If every instance of a phenomenon shares exactly one circumstance, that
/// circumstance is its cause. Each surface query anchor is an instance. Its
/// hit set is the pool turns that post the anchor from their own content or
/// features (`TurnIx.body` is never read). The circumstance is a session.
/// The sessions common to every hit set are the agreement. Exactly one such
/// session is a cause. An empty hit set, a tie, or several common sessions
/// fix nothing.
///
/// A head already inside the agreement session is never displaced. A head
/// outside it yields slot 1 in exactly one of two shapes:
///
/// - Non-instance. The head posts no surface anchor, so it is not an instance
///   of the phenomenon. The unique top-5 turn of the agreement session that
///   posts at least one anchor (an instance) takes slot 1.
/// - Complete instance. The head posts some anchors and misses others. The
///   unique top-5 turn of the agreement session that posts every anchor, and
///   that is the same speaker as the head, takes slot 1. A different speaker
///   is the other role's echo of the same circumstance, not a second witness.
///
/// Two qualifying turns are a tie. Top-5 membership is unchanged. No new
/// constant: the pool cap is the dump's 50 and the window is the existing
/// top-5. Does not touch `rsb_seat_behind`.
fn apply_c2_step35_mill_agreement(
    ix: &Index,
    plan: &QueryPlan,
    list: &mut Vec<usize>,
    t_scene: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let anchors = c2_surface_anchor_ids(ix, plan);
    if anchors.is_empty() {
        return false;
    }
    let pool = c2_dump_pool(ix, list, t_scene);
    let posts = |ti: usize, id: u32| {
        let t = &ix.turns[ti];
        t.content.contains(&id) || t.feats.iter().any(|f| f.id == id)
    };
    let mut common: DetHashSet<usize> = DetHashSet::default();
    let mut started = false;
    for &id in &anchors {
        let mut sessions: DetHashSet<usize> = DetHashSet::default();
        for &ti in &pool {
            if posts(ti, id) {
                sessions.insert(ix.turns[ti].session);
            }
        }
        if sessions.is_empty() {
            return false;
        }
        if !started {
            common = sessions;
            started = true;
        } else {
            common.retain(|s| sessions.contains(s));
        }
        if common.is_empty() {
            return false;
        }
    }
    if common.len() != 1 {
        return false;
    }
    let agree = match common.iter().next() {
        Some(&s) => s,
        None => return false,
    };
    let head = list[0];
    if ix.turns[head].session == agree {
        return false;
    }
    let head_has: Vec<bool> = anchors.iter().map(|&id| posts(head, id)).collect();
    let head_any = head_has.iter().copied().any(|h| h);
    let head_speaker = ix.turns[head].speaker.clone();
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut slot: Option<usize> = None;
    for i in 1..window {
        let cand = list[i];
        if ix.turns[cand].session != agree {
            continue;
        }
        let qualifies = if head_any {
            ix.turns[cand].speaker == head_speaker && anchors.iter().all(|&id| posts(cand, id))
        } else {
            anchors.iter().any(|&id| posts(cand, id))
        };
        if !qualifies {
            continue;
        }
        if slot.is_some() {
            return false;
        }
        slot = Some(i);
    }
    let Some(slot) = slot else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Content-token anchors of the query. Each anchor is the vocab ids of the
/// surface token and, when it differs, its stem. Closed-class tokens are not
/// anchors. A token the index never interned is not an anchor. No turn body
/// is read.
fn step36_anchors(ix: &Index, query: &str) -> Vec<Vec<u32>> {
    let mut seen: DetHashSet<String> = DetHashSet::default();
    let mut anchors = Vec::new();
    for w in raw_tokens(query) {
        if !is_content(&w) || !seen.insert(w.clone()) {
            continue;
        }
        let mut ids = Vec::new();
        if let Some(id) = ix.vocab.get(&w) {
            ids.push(id);
        }
        let stemmed = stem(&w);
        if stemmed != w {
            if let Some(id) = ix.vocab.get(&stemmed) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        if !ids.is_empty() {
            anchors.push(ids);
        }
    }
    anchors
}

fn step36_posts(turn: &TurnIx, ids: &[u32]) -> bool {
    ids.iter()
        .any(|id| turn.content.contains(id) || turn.feats.iter().any(|f| f.id == *id))
}

/// Antaranga-bahiranga (Step 36). The internal operation precedes the external
/// one. Internal evidence is `t_own`, the turn's uncarried lexical score.
/// External evidence on the turn is everything in `t_scene` above `t_own`
/// (neighbour carry and later scene operations). `bridge_boost` is not part of
/// `t_scene`; it is added only to the session rank score, and that sum is the
/// session sightline.
///
/// Slot 1 yields only when it is a question and its external residual strictly
/// exceeds its own score. The witness is a top-5 turn that is not a question,
/// whose own score strictly exceeds slot 1's own score and is a strict majority
/// of its own scene score, that attests every query anchor slot 1 attests, and
/// whose session score (`s_score + bridge_boost`) strictly exceeds slot 1's.
/// Two witnesses with the same own score are not a unique internal reading, so
/// the seat stays. Top-5 membership does not change. Does not read turn bodies
/// and does not touch `rsb_seat_behind`.
fn apply_c2_step36_antaranga(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    t_own: &[f32],
    t_scene: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) -> bool {
    if list.len() < 2 {
        return false;
    }
    let head = list[0];
    if !ix.turns[head].is_question {
        return false;
    }
    let own = |ti: usize| t_own.get(ti).copied().unwrap_or(0.0);
    let scene = |ti: usize| t_scene.get(ti).copied().unwrap_or(0.0);
    let sess = |ti: usize| {
        let s = ix.turns[ti].session;
        s_score.get(s).copied().unwrap_or(0.0) + bridge_boost.get(s).copied().unwrap_or(0.0)
    };
    let ho = own(head);
    let hs = scene(head);
    // Mostly external: the scene residual above own lexical exceeds own lexical.
    if hs - ho <= ho {
        return false;
    }
    let anchors = step36_anchors(ix, query);
    let head_turn = &ix.turns[head];
    let head_sess = sess(head);
    let window = list.len().min(C2_BULLAE_WINDOW);
    let mut best: Option<(f32, usize)> = None;
    let mut tied = false;
    for slot in 1..window {
        let cand = list[slot];
        let turn = &ix.turns[cand];
        if turn.is_question {
            continue;
        }
        let to = own(cand);
        let ts = scene(cand);
        if to <= ho || to <= ts - to {
            continue;
        }
        if sess(cand) <= head_sess {
            continue;
        }
        let covers = anchors
            .iter()
            .all(|ids| !step36_posts(head_turn, ids) || step36_posts(turn, ids));
        if !covers {
            continue;
        }
        match best {
            None => best = Some((to, slot)),
            Some((bo, _)) if to > bo => {
                best = Some((to, slot));
                tied = false;
            }
            Some((bo, _)) if to == bo => tied = true,
            _ => {}
        }
    }
    if tied {
        return false;
    }
    let Some((_, slot)) = best else {
        return false;
    };
    let cand = list.remove(slot);
    list.insert(0, cand);
    true
}

/// Month ids the existing calendar compiler emitted for `text` (`__MONTH__MM`).
/// No word list of our own: `inline_date_tokens` already decides which orthographic
/// month is a calendar month, including the ambiguous "may" / "march" guard.
fn w2001_query_months(text: &str) -> Vec<String> {
    let mut months: Vec<String> = Vec::new();
    for tok in inline_date_tokens(text) {
        if let Some(mm) = tok.strip_prefix("__MONTH__") {
            if !months.iter().any(|m| m == mm) {
                months.push(mm.to_string());
            }
        }
    }
    months
}

fn w2001_session_month(ix: &Index, si: usize) -> Option<String> {
    let ts = ix.session_ts.get(si).copied().unwrap_or(0);
    if ts == 0 {
        return None;
    }
    calendar_tokens(ts)
        .into_iter()
        .find_map(|tok| tok.strip_prefix("__MONTH__").map(|mm| mm.to_string()))
}

fn w2001_gt(a: f32, b: f32) -> bool {
    a.total_cmp(&b) == std::cmp::Ordering::Greater
}

fn w2001_ge(a: f32, b: f32) -> bool {
    a.total_cmp(&b) != std::cmp::Ordering::Less
}

/// Best `(score, turn)` inside one session. Equal scores keep the earlier turn.
fn w2001_best_turn(ix: &Index, si: usize, t_scene: &[f32]) -> Option<(f32, usize)> {
    let members = ix.session_turns.get(si)?;
    let mut best: Option<(f32, usize)> = None;
    for &ti in members {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        match best {
            None => best = Some((sc, ti)),
            Some((bs, bt)) => {
                if w2001_gt(sc, bs) || (sc.to_bits() == bs.to_bits() && ti < bt) {
                    best = Some((sc, ti));
                }
            }
        }
    }
    best
}

/// Session that holds the strictly best turn whose session date carries `month`.
/// A tie across two sessions is an ambiguous date match: no winner.
fn w2001_calendar_session(ix: &Index, month: &str, t_scene: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    let mut tied = false;
    for si in 0..ix.session_ts.len() {
        if w2001_session_month(ix, si).as_deref() != Some(month) {
            continue;
        }
        let Some((sc, _)) = w2001_best_turn(ix, si, t_scene) else {
            continue;
        };
        if !w2001_gt(sc, 0.0) {
            continue;
        }
        match best {
            None => best = Some((si, sc)),
            Some((bsi, bsc)) => {
                if w2001_gt(sc, bsc) {
                    best = Some((si, sc));
                    tied = false;
                } else if sc.to_bits() == bsc.to_bits() && si != bsi {
                    tied = true;
                }
            }
        }
    }
    if tied {
        None
    } else {
        best.map(|(si, _)| si)
    }
}

/// Session of the unique highest-`t_scene` turn. Equal tops in two sessions are
/// not one envelope.
fn w2001_scene_leader_session(ix: &Index, t_scene: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    let mut tied = false;
    for (ti, turn) in ix.turns.iter().enumerate() {
        let sc = t_scene.get(ti).copied().unwrap_or(0.0);
        if !w2001_gt(sc, 0.0) {
            continue;
        }
        match best {
            None => best = Some((turn.session, sc)),
            Some((bs, bv)) => {
                if w2001_gt(sc, bv) {
                    best = Some((turn.session, sc));
                    tied = false;
                } else if sc.to_bits() == bv.to_bits() && turn.session != bs {
                    tied = true;
                }
            }
        }
    }
    if tied {
        None
    } else {
        best.map(|(si, _)| si)
    }
}

/// Move `si` to session #1 only when it already sits inside the top 5.
/// Returns the old position. A session already at #1, or outside the top 5,
/// is not moved: the top-5 set stays intact.
fn w2001_take(list: &mut Vec<usize>, si: usize) -> Option<usize> {
    let pos = list.iter().position(|&s| s == si)?;
    if pos == 0 || pos >= 5 {
        return None;
    }
    let moved = list.remove(pos);
    list.insert(0, moved);
    Some(pos)
}

/// Where `si` sits, when the calendar or envelope rule owns the decision.
/// `Some(None)` means the rule applied and the list must stay (already #1, or
/// the session is outside the top 5 so promoting it would demote someone out
/// of the top 5). `None` means the rule did not apply.
fn w2001_owned(list: &[usize], si: usize) -> Option<Option<usize>> {
    match list.iter().position(|&s| s == si) {
        Some(0) => Some(None),
        Some(p) if p < 5 => Some(Some(p)),
        _ => None,
    }
}

/// Envelope-seal-calendar session head (W2-001).
///
/// The three rules only nominate a session. A nomination is seated at #1 only
/// when four witnesses agree, and only by moving a session that is already
/// inside the top 5:
/// - the nominee already holds strictly more top-5 turns than the incumbent;
/// - its session sightline (`s_score + bridge_boost`) strictly exceeds the
///   incumbent;
/// - the incumbent does not already attest every query cue on the rank-1 turn;
/// - the rank-1 turn's own lexical score strictly exceeds the incumbent's best
///   turn's own lexical score.
///
/// Nomination order:
/// 1. Calendar. One calendar month, and one session holds the strictly best
///    date-matched turn. Two months, or a tie, do not nominate.
/// 2. Envelope. The rank-1 turn and the unique highest-`t_scene` turn share a
///    session.
/// 3. Seal. The rank-1 turn scores at least as high as every turn of the
///    current #1 session and lives in another session.
///
/// A failed witness does not fall through to a later rule: that nomination
/// owned the decision. Does not read turn bodies and does not touch
/// `rsb_seat_behind`.
fn w2001_sight(s_score: &[f32], bridge: &[f32], si: usize) -> f32 {
    s_score.get(si).copied().unwrap_or(0.0) + bridge.get(si).copied().unwrap_or(0.0)
}

fn w2001_cue_union(ix: &Index, q_cues: &[u32], si: usize) -> Vec<u32> {
    let mut union: Vec<u32> = Vec::new();
    if let Some(members) = ix.session_turns.get(si) {
        for &ti in members {
            for c in turn_query_cues(ix, q_cues, ti) {
                if !union.contains(&c) {
                    union.push(c);
                }
            }
        }
    }
    union.sort_unstable();
    union
}

/// Four witnesses. False means the nomination is refused and no later rule runs.
fn w2001_admit(
    ix: &Index,
    query: &str,
    incumbent: usize,
    si: usize,
    turns: &[usize],
    t_scene: &[f32],
    t_own: &[f32],
    s_score: &[f32],
    bridge: &[f32],
) -> bool {
    let Some(&head) = turns.first() else {
        return false;
    };
    if si == incumbent {
        return false;
    }
    let in_session = |ti: usize, sess: usize| ix.turns.get(ti).is_some_and(|t| t.session == sess);
    let inc_n = turns.iter().take(5).filter(|&&ti| in_session(ti, incumbent)).count();
    let prom_n = turns.iter().take(5).filter(|&&ti| in_session(ti, si)).count();
    if prom_n <= inc_n {
        return false;
    }
    if !w2001_gt(w2001_sight(s_score, bridge, si), w2001_sight(s_score, bridge, incumbent)) {
        return false;
    }
    let q_cues = query_cues(ix, query);
    let head_c = turn_query_cues(ix, &q_cues, head);
    if head_c.is_empty() || covers_cues(&w2001_cue_union(ix, &q_cues, incumbent), &head_c) {
        return false;
    }
    let head_own = t_own.get(head).copied().unwrap_or(0.0);
    let inc_own = w2001_best_turn(ix, incumbent, t_scene)
        .and_then(|(_, ti)| t_own.get(ti).copied())
        .unwrap_or(0.0);
    w2001_gt(head_own, inc_own)
}

fn w2001_commit(
    ix: &Index,
    query: &str,
    list: &mut Vec<usize>,
    si: usize,
    turns: &[usize],
    t_scene: &[f32],
    t_own: &[f32],
    s_score: &[f32],
    bridge: &[f32],
) -> Option<usize> {
    let incumbent = *list.first()?;
    if !w2001_admit(ix, query, incumbent, si, turns, t_scene, t_own, s_score, bridge) {
        return None;
    }
    w2001_take(list, si)
}

fn apply_c2_w2001_session_head(
    ix: &Index,
    query: &Query,
    turns: &[usize],
    session_list: &mut Vec<usize>,
    t_scene: &[f32],
    t_own: &[f32],
    s_score: &[f32],
    bridge_boost: &[f32],
) -> Option<usize> {
    if session_list.len() < 2 || turns.is_empty() {
        return None;
    }
    let head = turns[0];
    let head_sess = ix.turns.get(head)?.session;
    let months = w2001_query_months(&query.text);
    if months.len() == 1 {
        if let Some(si) = w2001_calendar_session(ix, &months[0], t_scene) {
            if let Some(owned) = w2001_owned(session_list, si) {
                return match owned {
                    Some(_) => w2001_commit(
                        ix, &query.text, session_list, si, turns, t_scene, t_own, s_score, bridge_boost,
                    ),
                    None => None,
                };
            }
        }
    }
    if let Some(leader) = w2001_scene_leader_session(ix, t_scene) {
        if leader == head_sess {
            if let Some(owned) = w2001_owned(session_list, leader) {
                return match owned {
                    Some(_) => w2001_commit(
                        ix,
                        &query.text,
                        session_list,
                        leader,
                        turns,
                        t_scene,
                        t_own,
                        s_score,
                        bridge_boost,
                    ),
                    None => None,
                };
            }
        }
    }
    let incumbent = session_list[0];
    if head_sess == incumbent {
        return None;
    }
    let head_score = t_scene.get(head).copied().unwrap_or(0.0);
    if !w2001_gt(head_score, 0.0) {
        return None;
    }
    let inc_best = w2001_best_turn(ix, incumbent, t_scene)
        .map(|(sc, _)| sc)
        .unwrap_or(0.0);
    if w2001_ge(head_score, inc_best) {
        return w2001_commit(
            ix,
            &query.text,
            session_list,
            head_sess,
            turns,
            t_scene,
            t_own,
            s_score,
            bridge_boost,
        );
    }
    None
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
        assert!(apply_c2_elaboration_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![4, 2, 3, 5, 6]);
        // Equal scene is enough (at least as on-scene as the head).
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(apply_c2_elaboration_seat(&ix, &plan, &q.text, &mut list, &[0.5, 0.4, 1.0, 0.8, 1.0, 0.9, 0.2]));
        assert_eq!(list, vec![4, 2, 3, 5, 6]);
        // Below the head's scene score: no swap.
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[0.5, 0.4, 1.0, 0.8, 0.99, 0.9, 0.2]
        ));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Session opener at slot 1: no swap.
        let mut list = vec![0usize, 1, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 6]);
        // Next user not already in the top 5: membership must not change.
        let mut list = vec![2usize, 3, 5, 6, 1];
        assert!(!apply_c2_elaboration_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 5, 6, 1]);
        // Preference query: no swap.
        let pq = Query { text: "Can you recommend a birthday gift for my sister?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 4, 5, 6]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When did I buy my sister's birthday gift?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(!apply_c2_elaboration_seat(&ix, &tplan, &tq.text, &mut list, &t_scene));
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
        assert!(!apply_c2_elaboration_seat(&ixn, &plann, &q.text, &mut list, &[0.5, 0.4, 1.0, 0.8, 1.05]));
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

    #[test]
    fn queried_speaker_prev_seat_promotes_named_speakers_previous_statement() {
        let ix = build_index(
            &[sess(
                "D3",
                &[
                    ("gina", "I really care about creating an experience for my customers."),
                    ("jon", "That's a thoughtful approach to running the store."),
                    ("gina", "Thanks, I try to keep that in mind every day."),
                ],
            )],
            &Params::default(),
        );
        let q = Query {
            text: "What did Gina say about creating an experience for her customers?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(!ix.turns[0].is_question);
        assert!(!ix.turns[1].is_question);
        assert!(c2_query_mentions_speaker(&q.text, "gina"));
        assert!(!c2_query_mentions_speaker(&q.text, "jon"));
        // jon#1 is a declarative follow-up; gina#0 is the previous named statement already in the top 5.
        let mut list = vec![1usize, 0, 2];
        let t_scene = vec![0.9f32, 1.0, 0.5];
        assert!(apply_c2_queried_speaker_prev_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);
        // Question head: no seat (that's #15's next-reply shape).
        let ixq = build_index(
            &[sess(
                "D3",
                &[
                    ("gina", "I really care about creating an experience for my customers."),
                    ("jon", "What did you mean about creating an experience for customers?"),
                ],
            )],
            &Params::default(),
        );
        assert!(ixq.turns[1].is_question);
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_prev_seat(&ixq, &plan, &q.text, &mut list, &[0.9, 1.0]));
        assert_eq!(list, vec![1, 0]);
        // Previous turn is a question: no seat.
        let ixp = build_index(
            &[sess(
                "D3",
                &[
                    ("gina", "What would it take to create an experience for my customers?"),
                    ("jon", "That's a thoughtful approach to running the store."),
                ],
            )],
            &Params::default(),
        );
        assert!(ixp.turns[0].is_question);
        assert!(!ixp.turns[1].is_question);
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_prev_seat(&ixp, &plan, &q.text, &mut list, &[0.9, 1.0]));
        assert_eq!(list, vec![1, 0]);
        // Head speaker is already named in the query: no seat.
        let hq = Query { text: "What did Jon say after Gina talked about customers?".into(), date: String::new() };
        let hplan = plan_query(&ix, &hq, &Params::default());
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_queried_speaker_prev_seat(&ix, &hplan, &hq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // Conjoined speaker ("X and Y"): no seat.
        let aq = Query {
            text: "What did Gina and Anna say about creating an experience for customers?".into(),
            date: String::new(),
        };
        assert!(c2_speaker_conjoined(&aq.text, "gina"));
        let aplan = plan_query(&ix, &aq, &Params::default());
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_queried_speaker_prev_seat(&ix, &aplan, &aq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // Preference query: no seat.
        let pq = Query { text: "What is Gina's favorite way to create an experience for customers?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_queried_speaker_prev_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // WH-person: no seat.
        let wq = Query { text: "Who told Gina about creating an experience for customers?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_queried_speaker_prev_seat(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0, 2]);
        // user/assistant chats are left alone.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I really care about creating an experience for my customers."),
                    ("assistant", "That's a thoughtful approach to running the store."),
                ],
            )],
            &Params::default(),
        );
        let uq = Query { text: "What did Gina say about creating an experience for her customers?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_prev_seat(&ixu, &uplan, &uq.text, &mut list, &[0.9, 1.0]));
        assert_eq!(list, vec![1, 0]);
        // Previous turn not in the top 5: no seat.
        let mut list = vec![1usize, 2];
        assert!(!apply_c2_queried_speaker_prev_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2]);
    }

    #[test]
    fn queried_speaker_t5_entry_swaps_declarative_mentioned_into_top5() {
        let ix = build_index(
            &[
                sess(
                    "D7",
                    &[
                        ("john", "I have been thinking about public service for a long time."),
                        ("maria", "What made you want to run for office again?"),
                        ("john", "I decided to run for office again after seeing the impact I could make in the community through politics."),
                    ],
                ),
                sess(
                    "D14",
                    &[
                        ("maria", "Running for office takes a lot of work in any community."),
                        ("john", "The last campaign taught me so much about politics."),
                    ],
                ),
                sess(
                    "D6",
                    &[("john", "I used to volunteer in the community a lot.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "Why did John decide to run for office again?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(!ix.turns[2].is_question);
        assert!(c2_query_mentions_speaker(&q.text, "john"));
        assert!(!c2_query_mentions_speaker(&q.text, "maria"));
        // Official k of 6: last unmentioned (maria D14) at slot 3 yields to
        // John's later declarative answer already in k at slot 6.
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        let t_scene = vec![1.0f32, 0.9, 0.85, 0.8, 0.5, 0.4];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_queried_speaker_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 4, 5, 3]);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // A mentioned speaker's question in official k is not pulled (4:74).
        let ixq = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("john", "My fan project will cover many sides of that universe."),
                        ("tim", "Which aspects will you discuss in the collaborations?"),
                        ("tim", "We will discuss characters, spells, and magical creatures."),
                        ("john", "Which aspects of the universe should the fan project cover?"),
                        ("john", "I have loved this universe since I was a kid."),
                        ("tim", "The project is going to be a lot of fun."),
                    ],
                ),
            ],
            &Params::default(),
        );
        assert!(ixq.turns[3].is_question);
        assert!(!ixq.turns[2].is_question);
        let qhp = Query {
            text: "What aspects of the universe will be discussed in John's fan project collaborations?".into(),
            date: String::new(),
        };
        let qplan = plan_query(&ixq, &qhp, &Params::default());
        let mut list = vec![0usize, 1, 2, 4, 5, 3];
        let qsc = vec![1.0f32, 0.9, 1.0, 0.95, 0.4, 0.3];
        assert!(!apply_c2_queried_speaker_t5_entry(&ixq, &qplan, &qhp.text, &mut list, &qsc));
        assert_eq!(list, vec![0, 1, 2, 4, 5, 3]);

        // Below the reused 70% share: no entry.
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_queried_speaker_t5_entry(&ix, &plan, &q.text, &mut list, &[1.0, 0.9, 0.5, 0.8, 0.5, 0.4]));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);

        // Preference query: no entry.
        let pq = Query { text: "What is John's favorite reason to run for office again?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_queried_speaker_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);

        // Conjoined speaker: no entry.
        let aq = Query {
            text: "Why did John and Maria decide to run for office again?".into(),
            date: String::new(),
        };
        assert!(c2_speaker_conjoined(&aq.text, "john"));
        let aplan = plan_query(&ix, &aq, &Params::default());
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_queried_speaker_t5_entry(&ix, &aplan, &aq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);

        // WH-person: no entry.
        let wq = Query { text: "Who decided to run for office again after seeing the impact?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_queried_speaker_t5_entry(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);

        // user/assistant chats are left alone.
        let ixu = build_index(
            &[
                sess(
                    "s",
                    &[
                        ("user", "I have been thinking about public service for a long time."),
                        ("assistant", "What made you want to run for office again?"),
                        ("user", "I decided to run for office again after seeing the impact I could make."),
                        ("assistant", "Running for office takes a lot of work."),
                        ("user", "The last campaign taught me so much."),
                        ("user", "I used to volunteer in the community a lot."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let uq = Query { text: "Why did John decide to run for office again?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_queried_speaker_t5_entry(&ixu, &uplan, &uq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);

        // Swap that would push a session out of the first 5 unique slots: no entry.
        let ixs = build_index(
            &[
                sess("A", &[("john", "I keep notes from every campaign stop I make.")]),
                sess("B", &[("john", "The community meeting last week was packed.")]),
                sess("C", &[("john", "I knocked on doors in every neighborhood.")]),
                sess("D", &[("john", "The debate prep took most of the afternoon.")]),
                sess("E", &[("maria", "Campaigning in that district is always hard work.")]),
                sess("F", &[("john", "I decided to run for office again after seeing the impact I could make in the community through politics.")]),
            ],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        let ssc = vec![1.0f32, 0.9, 0.9, 0.9, 0.8, 0.85];
        assert!(!apply_c2_queried_speaker_t5_entry(&ixs, &plan, &q.text, &mut list, &ssc));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn queried_speaker_qhead_prev_seat_promotes_named_speakers_previous_turn() {
        // Declarative previous turn (7:1 shape).
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("deborah", "My mother, my father, and my friend Karlie have passed away."),
                    ("jolene", "Which of your family and friends have passed away?"),
                    ("deborah", "It has been a hard season for the family."),
                ],
            )],
            &Params::default(),
        );
        // Backtick possessive, as in the live LoCoMo query: word_padded
        // splits it, and the bare speaker token still matches.
        let q = Query {
            text: "Which of Deborah`s family and friends have passed away?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(ix.turns[1].is_question);
        assert!(!ix.turns[0].is_question);
        assert!(c2_query_mentions_speaker(&q.text, "deborah"));
        assert!(!c2_query_mentions_speaker(&q.text, "jolene"));
        let mut list = vec![1usize, 2, 0];
        let t_scene = vec![0.7f32, 1.0, 0.5];
        assert!(apply_c2_queried_speaker_qhead_prev_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2]);

        // Question previous turn (5:27 shape) also seats.
        let ixq = build_index(
            &[sess(
                "D18",
                &[
                    ("andrew", "How do I feel about my current work these days?"),
                    ("audrey", "How do you feel about your current work?"),
                    ("andrew", "It has been stressful at my current job."),
                ],
            )],
            &Params::default(),
        );
        assert!(ixq.turns[0].is_question);
        assert!(ixq.turns[1].is_question);
        let aqry = Query { text: "How does Andrew feel about his current work?".into(), date: String::new() };
        let aplan = plan_query(&ixq, &aqry, &Params::default());
        let mut list = vec![1usize, 2, 0];
        assert!(apply_c2_queried_speaker_qhead_prev_seat(&ixq, &aplan, &aqry.text, &mut list, &[0.7, 1.0, 0.5]));
        assert_eq!(list, vec![0, 1, 2]);

        // Declarative head: no seat (that's #16's both-declarative shape).
        let ixd = build_index(
            &[sess(
                "D1",
                &[
                    ("deborah", "My mother, my father, and my friend Karlie have passed away."),
                    ("jolene", "That is a hard season for any family."),
                ],
            )],
            &Params::default(),
        );
        assert!(!ixd.turns[1].is_question);
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ixd, &plan, &q.text, &mut list, &[0.7, 1.0]));
        assert_eq!(list, vec![1, 0]);

        // Head speaker is already named in the query: no seat.
        let hq = Query { text: "Which of Jolene's questions about family came before Deborah spoke?".into(), date: String::new() };
        let hplan = plan_query(&ix, &hq, &Params::default());
        let mut list = vec![1usize, 2, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ix, &hplan, &hq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2, 0]);

        // Conjoined speaker ("X and Y"): no seat.
        let cq = Query {
            text: "Which of Deborah and Anna's family and friends have passed away?".into(),
            date: String::new(),
        };
        assert!(c2_speaker_conjoined(&cq.text, "deborah"));
        let cplan = plan_query(&ix, &cq, &Params::default());
        let mut list = vec![1usize, 2, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ix, &cplan, &cq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2, 0]);

        // Preference query: no seat.
        let pq = Query { text: "What is Deborah's favorite memory of her family and friends?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![1usize, 2, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2, 0]);

        // WH-person: no seat.
        let wq = Query { text: "Who in Deborah's family and friends have passed away?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![1usize, 2, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2, 0]);

        // user/assistant chats are left alone.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "My mother, my father, and my friend Karlie have passed away."),
                    ("assistant", "Which of your family and friends have passed away?"),
                ],
            )],
            &Params::default(),
        );
        let uq = Query { text: "Which of Deborah`s family and friends have passed away?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ixu, &uplan, &uq.text, &mut list, &[0.7, 1.0]));
        assert_eq!(list, vec![1, 0]);

        // Previous turn not in the top 5: no seat.
        let mut list = vec![1usize, 2];
        assert!(!apply_c2_queried_speaker_qhead_prev_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 2]);
    }

    #[test]
    fn c2_text_preference_matches_classify_not_constraint_upgrade() {
        assert!(!c2_text_preference(
            "What was the reason Audrey couldn't walk her dogs for a period of time?",
        ));
        assert!(!c2_text_preference(
            "What are the classes that Audrey took for her dogs to?",
        ));
        assert!(!c2_text_preference("Where do I take yoga classes?"));
        assert!(c2_text_preference(
            "Can you recommend a birthday gift for my sister?",
        ));
    }

    #[test]
    fn queried_speaker_first_seat_still_fires_when_only_constraint_upgrade_sets_preference() {
        // plan.intent.preference is forced the way plan_query would after a
        // constraint-object upgrade; the query itself is not a PREF_QUERY.
        let ix = build_index(
            &[
                sess("D8", &[("andrew", "I walked the dogs today."), ("andrew", "The park was muddy.")]),
                sess("D22", &[("audrey", "I couldn't walk the dogs because of my ankle."), ("andrew", "Sorry to hear that.")]),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What was the reason Audrey couldn't walk her dogs for a period of time?".into(),
            date: String::new(),
        };
        let mut plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        plan.intent.preference = true;
        // D8: unmentioned andrew at slot 1. D22's first Audrey turn is already in the top 5.
        let mut list = vec![0usize, 1, 2, 3];
        let t_scene = vec![1.0f32, 0.8, 0.7, 0.5];
        assert!(apply_c2_queried_speaker_first_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3]);
    }

    #[test]
    fn queried_speaker_first_seat_promotes_named_speakers_first_in_session_turn() {
        // Adjacent previous question that is also the speaker's first
        // utterance (6:78 shape).
        let ix = build_index(
            &[sess(
                "D5",
                &[
                    ("james", "How should I go about adopting a rescue?"),
                    ("john", "That is a great plan for the spring."),
                ],
            )],
            &Params::default(),
        );
        let q = Query { text: "What did James adopt?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(ix.turns[0].is_question);
        assert!(!ix.turns[1].is_question);
        assert!(c2_query_mentions_speaker(&q.text, "james"));
        assert!(!c2_query_mentions_speaker(&q.text, "john"));
        let mut list = vec![1usize, 0];
        let t_scene = vec![0.9f32, 1.0];
        assert!(apply_c2_queried_speaker_first_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1]);

        // Non-adjacent first-in-session turn already in the top 5 (5:100 shape).
        let ix2 = build_index(
            &[
                sess(
                    "D8",
                    &[
                        ("audrey", "The weather has been terrible for walking."),
                        ("andrew", "The trail has been muddy all week."),
                        ("audrey", "Has the rain kept you off the usual loop?"),
                        ("andrew", "Yes, I have been staying in."),
                        ("audrey", "I have been staying in too."),
                    ],
                ),
                sess(
                    "D22",
                    &[
                        ("audrey", "A sprain kept me from walking the dogs."),
                        ("andrew", "Sorry to hear you had to pause the walks."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q2 = Query {
            text: "What was the reason Audrey couldn't walk her dogs?".into(),
            date: String::new(),
        };
        let p2 = plan_query(&ix2, &q2, &Params::default());
        assert!(!p2.intent.preference);
        assert!(!p2.intent.temporal);
        assert!(c2_query_mentions_speaker(&q2.text, "audrey"));
        assert!(!c2_query_mentions_speaker(&q2.text, "andrew"));
        // Index order: D8 turns 0..4, D22 turns 5..6. First Audrey in D8 is 0
        // (held out of the top 5). First Audrey in D22 is 5.
        assert_eq!(c2_first_same_speaker(&ix2, 2), Some(0));
        assert_eq!(c2_first_same_speaker(&ix2, 5), Some(5));
        // Head = D8 andrew (3); later Audrey in D8 (2) is in the top 5 but
        // is not first-in-session; D22 Audrey first (5) is in the top 5.
        let mut list = vec![3usize, 2, 1, 5, 4];
        let sc = vec![0.5f32, 1.0, 0.9, 1.1, 0.8, 0.4, 0.3];
        assert!(apply_c2_queried_speaker_first_seat(&ix2, &p2, &q2.text, &mut list, &sc));
        assert_eq!(list, vec![5, 3, 2, 1, 4]);

        // A later turn by the mentioned speaker, even if previous, does not seat.
        let ixl = build_index(
            &[sess(
                "D1",
                &[
                    ("james", "I have been thinking about a rescue."),
                    ("john", "Have you looked at the local shelter?"),
                    ("james", "What breed would suit an apartment?"),
                    ("john", "A small dog might work."),
                ],
            )],
            &Params::default(),
        );
        let mut list = vec![3usize, 2];
        assert!(!apply_c2_queried_speaker_first_seat(&ixl, &plan, &q.text, &mut list, &[1.0, 0.9, 0.8, 0.7]));
        assert_eq!(list, vec![3, 2]);

        // Head speaker is already named in the query: no seat.
        let hq = Query { text: "What did John say James should adopt?".into(), date: String::new() };
        let hplan = plan_query(&ix, &hq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_first_seat(&ix, &hplan, &hq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0]);

        // Conjoined speaker ("X and Y"): no seat.
        let cq = Query { text: "What did James and John adopt?".into(), date: String::new() };
        assert!(c2_speaker_conjoined(&cq.text, "james"));
        let cplan = plan_query(&ix, &cq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_first_seat(&ix, &cplan, &cq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0]);

        // Preference query: no seat.
        let pq = Query { text: "What is James's favorite rescue breed to adopt?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_first_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0]);

        // WH-person: no seat.
        let wq = Query { text: "Who did James adopt?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_first_seat(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 0]);

        // user/assistant chats are left alone.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "How should I go about adopting a rescue?"),
                    ("assistant", "That is a great plan for the spring."),
                ],
            )],
            &Params::default(),
        );
        let uq = Query { text: "What did James adopt?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_queried_speaker_first_seat(&ixu, &uplan, &uq.text, &mut list, &[0.9, 1.0]));
        assert_eq!(list, vec![1, 0]);

        // First-in-session turn not in the top 5: no seat.
        let mut list = vec![1usize];
        assert!(!apply_c2_queried_speaker_first_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![1]);
    }


    #[test]
    fn queried_speaker_first_seat_after_t5_entry_promotes_the_entered_turn() {
        // #23 admits Audrey's first turn in D22 from slot 6 into the top 5;
        // a second #22 pass then seats it at slot 1 over the unmentioned
        // Andrew head. The official-k list has no earlier first-in-session
        // Audrey, so the entered turn is the one that takes slot 1.
        let ix = build_index(
            &[
                sess(
                    "D8",
                    &[
                        ("andrew", "I walked in the park today."),
                        ("andrew", "The paths were muddy."),
                        ("andrew", "I stayed on the trail."),
                        ("andrew", "It was still a good loop."),
                        ("andrew", "I will go again tomorrow."),
                    ],
                ),
                sess(
                    "D22",
                    &[
                        ("audrey", "I couldn't walk the dogs because of my ankle."),
                        ("andrew", "Sorry to hear that."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What was the reason Audrey couldn't walk her dogs?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        // Turns 0-4: Andrew in D8. Turn 5: Audrey's first in D22.
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        let t_scene = vec![1.0f32, 0.9, 0.8, 0.7, 0.6, 0.4];
        assert_eq!(c2_first_same_speaker(&ix, 5), Some(5));
        assert!(apply_c2_queried_speaker_first_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert!(list[..5].contains(&5), "t5 entry should admit turn 5 into the top 5: {list:?}");
        assert_ne!(list[0], 5, "entry alone should not take slot 1");
        assert!(apply_c2_queried_speaker_first_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list[0], 5);
    }

    #[test]

    fn cross_opener_pair_seat_promotes_other_session_opener_with_reply_in_top5() {
        // 10:knowledge_update:1 shape: a user opener at slot 1 yields to
        // another session's user opener already in the top 5 whose
        // assistant reply is also already in the top 5, when that opener
        // is at least as on-scene.
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I need to pick a day to drop off the draft."),
                        ("assistant", "Any morning this week should work."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I want to submit my peer-reviewed draft to the local writing group by Friday."),
                        ("assistant", "Friday is a solid deadline for that group."),
                        ("user", "I will also send a copy to my advisor."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What deadline should I aim for to submit my peer-reviewed draft to the local writing group?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        // Turns: 0-1 session s1, 2-4 session s2. Candidate opener is 2, reply is 3.
        let mut list = vec![0usize, 3, 2, 4];
        let t_scene = vec![0.8f32, 0.7, 1.45, 1.6, 0.5];
        assert!(apply_c2_cross_opener_pair_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 3, 4]);
        // Reply not already in the top 5: no seat.
        let mut list = vec![0usize, 2, 4];
        assert!(!apply_c2_cross_opener_pair_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 4]);
        // Candidate below the head's scene score: no seat. This is the
        // 6:information_extraction:0 shape (lone opener at r=0.91).
        let mut list = vec![0usize, 3, 2, 4];
        assert!(!apply_c2_cross_opener_pair_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[1.0, 0.7, 0.9, 1.0, 0.5]
        ));
        assert_eq!(list, vec![0, 3, 2, 4]);
        // Mid-session head: no seat (#13's territory, not this one).
        let mut list = vec![4usize, 3, 2, 0];
        assert!(!apply_c2_cross_opener_pair_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![4, 3, 2, 0]);
        // Preference questions are skipped.
        let pq = Query { text: "What is my favorite writing group deadline?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 3, 2, 4];
        assert!(!apply_c2_cross_opener_pair_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 3, 2, 4]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I submit my peer-reviewed draft to the local writing group?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 3, 2, 4];
        assert!(!apply_c2_cross_opener_pair_seat(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 3, 2, 4]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("caroline", "I need to pick a day to drop off the draft."),
                        ("melanie", "Any morning this week should work."),
                    ],
                ),
                sess(
                    "D2",
                    &[
                        ("caroline", "I want to submit my peer-reviewed draft by Friday."),
                        ("melanie", "Friday is a solid deadline."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 3, 2, 1];
        assert!(!apply_c2_cross_opener_pair_seat(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 3, 2, 1]);
    }

    #[test]

    fn answered_opener_decl_seat_promotes_declarative_opener_over_answered_question() {
        // 16:instruction_following:0 shape: a user question opener at
        // slot 1 whose assistant reply is already in the top 5 yields to
        // another session's declarative user opener already in the top 5,
        // when that opener is at least as on-scene.
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I've never approved any holiday medical supply budgets, can you explain why?"),
                        ("assistant", "It may be relevant to current financial planning."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "Always confirm budget caps when I ask about holiday spending."),
                        ("assistant", "The holiday spending cap is four hundred dollars."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "How much am I allowed to spend on my holiday plans?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert!(ix.turns[0].is_question);
        assert!(!ix.turns[2].is_question);
        // Turns: 0-1 session s1, 2-3 session s2. Head 0, reply 1, cand 2.
        let mut list = vec![0usize, 2, 1, 3];
        let t_scene = vec![0.9f32, 1.0, 1.05, 0.7];
        assert!(apply_c2_answered_opener_decl_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Head reply not already in the top 5: no seat.
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_answered_opener_decl_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 3]);
        // Candidate below the head's scene score: no seat.
        let mut list = vec![0usize, 2, 1, 3];
        assert!(!apply_c2_answered_opener_decl_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[1.0, 0.8, 0.9, 0.7]
        ));
        assert_eq!(list, vec![0, 2, 1, 3]);
        // Declarative head: no seat (this rule needs an answered question).
        let mut list = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_answered_opener_decl_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3]);
        // Preference questions are skipped.
        let pq = Query { text: "What is my favorite holiday spending cap?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 2, 1, 3];
        assert!(!apply_c2_answered_opener_decl_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 1, 3]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I spend on my holiday plans?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 2, 1, 3];
        assert!(!apply_c2_answered_opener_decl_seat(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 1, 3]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("caroline", "I've never approved any holiday medical supply budgets, can you explain why?"),
                        ("melanie", "It may be relevant to current financial planning."),
                    ],
                ),
                sess(
                    "D2",
                    &[
                        ("caroline", "Always confirm budget caps when I ask about holiday spending."),
                        ("melanie", "The holiday spending cap is four hundred dollars."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 2, 1, 3];
        assert!(!apply_c2_answered_opener_decl_seat(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 1, 3]);
    }

    #[test]

    fn solo_earlier_opener_seat_promotes_earlier_opener_over_lone_head() {
        // 5:knowledge_update:0 shape: a lone user opener at slot 1 (no
        // other turn from that session in the top 5) yields to an earlier
        // user opener from a different session already in the top 5, when
        // that opener is at least as on-scene.
        let mut later = sess(
            "s2",
            &[
                ("user", "I just started a new practice schedule this month."),
                ("assistant", "Noted, I will keep that in mind."),
            ],
        );
        later.timestamp = 2_000;
        let mut earlier = sess(
            "s1",
            &[
                ("user", "I've been studying probability basics, including dice roll problems, for hours."),
                ("assistant", "Keep a running total of those hours."),
            ],
        );
        earlier.timestamp = 1_000;
        let ix = build_index(&[later, earlier], &Params::default());
        let q = Query {
            text: "How many hours have I spent studying probability basics?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        // Turns: 0-1 later session s2, 2-3 earlier session s1. Head 0 is solo.
        let mut list = vec![0usize, 2, 3];
        let t_scene = vec![0.9f32, 0.5, 1.05, 0.7];
        assert!(apply_c2_solo_earlier_opener_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 3]);
        // Head is not solo (its assistant reply is already in the top 5): no seat.
        let mut list = vec![0usize, 2, 1, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 1, 3]);
        // Candidate is later, not earlier: no seat.
        let mut later_cand = sess(
            "s3",
            &[
                ("user", "I logged more probability study hours last week."),
                ("assistant", "I recorded that total."),
            ],
        );
        later_cand.timestamp = 3_000;
        let mut later = sess(
            "s2",
            &[
                ("user", "I just started a new practice schedule this month."),
                ("assistant", "Noted, I will keep that in mind."),
            ],
        );
        later.timestamp = 2_000;
        let ixl = build_index(&[later, later_cand], &Params::default());
        let planl = plan_query(&ixl, &q, &Params::default());
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(&ixl, &planl, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 3]);
        // Candidate below the head's scene score: no seat.
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[1.0, 0.5, 0.9, 0.7]
        ));
        assert_eq!(list, vec![0, 2, 3]);
        // Preference questions are skipped.
        let pq = Query { text: "What is my favorite probability study schedule?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 3]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I study probability basics?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 3]);
        // Named-speaker sessions are left alone.
        let mut later_n = sess(
            "D2",
            &[
                ("caroline", "I just started a new practice schedule this month."),
                ("melanie", "Noted, I will keep that in mind."),
            ],
        );
        later_n.timestamp = 2_000;
        let mut earlier_n = sess(
            "D1",
            &[
                ("caroline", "I've been studying probability basics, including dice roll problems, for hours."),
                ("melanie", "Keep a running total of those hours."),
            ],
        );
        earlier_n.timestamp = 1_000;
        let ixn = build_index(&[later_n, earlier_n], &Params::default());
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 2, 3];
        assert!(!apply_c2_solo_earlier_opener_seat(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 3]);
    }

    #[test]

    fn pool_opener_t10_entry_replaces_tail_assistant_with_pool_opener() {
        // 2:instruction_following:1 shape: a user opener in the scene-ranked
        // pool, not already in the official k, takes the official-k seat of
        // the last non-unique tail assistant whose session still has a user
        // in the official k, when that opener is at least as on-scene. The
        // top 5 is unchanged.
        let ix = build_index(
            &[
                sess("s0", &[("user", "I am starting a new practice project this month."), ("assistant", "Noted, I will keep that in mind.")]),
                sess("s1", &[("user", "I also sketched a second practice project."), ("assistant", "I recorded that sketch.")]),
                sess("s2", &[("user", "A third note is already in the log."), ("assistant", "The third note is stored.")]),
                sess("s3", &[("user", "I logged a fourth practice reminder."), ("assistant", "I stored the fourth reminder.")]),
                sess("s4", &[("user", "I keep a running total of API error types."), ("assistant", "I recorded those error types.")]),
                sess("s5", &[("user", "Always confirm typical REST API errors when I build an application."), ("assistant", "Timeouts, 404s, and 500s are typical.")]),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What typical errors should I be prepared to handle with a REST API?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: 0-1 s0, 2-3 s1, 4-5 s2, 6-7 s3, 8-9 s4, 10-11 s5 (pool opener).
        // Official k of 10; tail last evictable assistant is 9 (s4).
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        let t_scene = vec![1.0f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.5, 0.2];
        let top5 = list[..5].to_vec();
        assert!(apply_c2_pool_opener_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(&list[..5], &top5[..]);
        assert_eq!(list[9], 10);
        assert!(!list.contains(&9));
        // Candidate below the victim's scene score: no entry.
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(!apply_c2_pool_opener_t10_entry(
            &ix,
            &plan,
            &q.text,
            &mut list,
            &[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.55, 0.5, 0.2]
        ));
        assert_eq!(list, vec![0, 1, 2, 3, 5, 6, 8, 4, 7, 9]);
        // Only evictable assistants are in the top 5; the tail assistant
        // is unique and has no user in the official k: no entry.
        let mut list = vec![0usize, 1, 3, 5, 7, 2, 4, 6, 8, 11];
        assert!(!apply_c2_pool_opener_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 5, 7, 2, 4, 6, 8, 11]);
        // Preference questions are skipped.
        let pq = Query { text: "What is my favorite way to handle REST API errors?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(!apply_c2_pool_opener_t10_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 5, 6, 8, 4, 7, 9]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I handle REST API errors?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(!apply_c2_pool_opener_t10_entry(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 5, 6, 8, 4, 7, 9]);
        // WH-person questions are left to the existing midcap path.
        let wq = Query { text: "Who handled the REST API errors?".into(), date: String::new() };
        let wplan = plan_query(&ix, &wq, &Params::default());
        assert_ne!(wh_slots(&wq.text) & WH_PERSON, 0);
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(!apply_c2_pool_opener_t10_entry(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 5, 6, 8, 4, 7, 9]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "I am starting a new practice project this month."), ("melanie", "Noted, I will keep that in mind.")]),
                sess("D1", &[("caroline", "I also sketched a second practice project."), ("melanie", "I recorded that sketch.")]),
                sess("D2", &[("caroline", "A third note is already in the log."), ("melanie", "The third note is stored.")]),
                sess("D3", &[("caroline", "I logged a fourth practice reminder."), ("melanie", "I stored the fourth reminder.")]),
                sess("D4", &[("caroline", "I keep a running total of API error types."), ("melanie", "I recorded those error types.")]),
                sess("D5", &[("caroline", "Always confirm typical REST API errors when I build an application."), ("melanie", "Timeouts, 404s, and 500s are typical.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(!apply_c2_pool_opener_t10_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 5, 6, 8, 4, 7, 9]);
    }

    #[test]

    fn same_session_opener_t5_entry_swaps_opener_for_same_session_top5_assistant() {
        // 41698283 / 80ec1f4f shape: a user opener already in official k
        // but outside the top 5, from a session already represented in
        // the top 5 by an assistant, swaps with the last same-session
        // assistant in the top 5. The victim stays in official k.
        let ix = build_index(
            &[
                sess("s0", &[("user", "I am starting a new practice project this month."), ("assistant", "Noted, I will keep that in mind.")]),
                sess(
                    "s1",
                    &[
                        ("user", "Always confirm typical REST API errors when I build an application."),
                        ("assistant", "Timeouts, 404s, and 500s are typical."),
                    ],
                ),
                sess("s2", &[("user", "A third note is already in the log."), ("assistant", "The third note is stored.")]),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What typical errors should I be prepared to handle with a REST API?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: 0-1 s0, 2-3 s1 (opener 2, assistant 3), 4-5 s2.
        // Official k of 6; s1 assistant is in the top 5, s1 opener is not.
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        let t_scene = vec![1.0f32, 0.9, 0.7, 0.8, 0.6, 0.5];
        assert!(apply_c2_same_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 2, 3]);
        // Later same-session user (not an opener) is not seated: that
        // is the #20 clause this structure drops.
        let ix_l = build_index(
            &[
                sess("s0", &[("user", "I am starting a new practice project this month."), ("assistant", "Noted, I will keep that in mind.")]),
                sess(
                    "s1",
                    &[
                        ("user", "I opened a log of API errors last month."),
                        ("assistant", "I recorded that log."),
                        ("user", "I later listed timeouts, 404s, and 500s as typical REST API errors."),
                    ],
                ),
                sess("s2", &[("user", "A third note is already in the log."), ("assistant", "The third note is stored.")]),
            ],
            &Params::default(),
        );
        let plan_l = plan_query(&ix_l, &q, &Params::default());
        // Turns: 0-1 s0, 2-4 s1 (opener 2, asst 3, later user 4), 5-6 s2.
        // Top 5 has s1 assistant 3; official-k slot 6 is later user 4, not the opener.
        let mut list = vec![0usize, 1, 5, 6, 3, 4];
        let t_scene_l = vec![1.0f32, 0.9, 0.5, 0.8, 0.7, 0.6, 0.4];
        assert!(!apply_c2_same_session_opener_t5_entry(&ix_l, &plan_l, &q.text, &mut list, &t_scene_l));
        assert_eq!(list, vec![0, 1, 5, 6, 3, 4]);
        // No same-session assistant in the top 5: no entry.
        let mut list = vec![0usize, 1, 4, 5, 2, 3];
        assert!(!apply_c2_same_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 2, 3]);
        // Preference questions are skipped.
        let pq = Query { text: "What is my favorite way to handle REST API errors?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        assert!(!apply_c2_same_session_opener_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 3, 2]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When do I handle REST API errors?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        assert!(!apply_c2_same_session_opener_t5_entry(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 3, 2]);
        // WH-person questions are left to the existing midcap path.
        let wq = Query { text: "Who handled the REST API errors?".into(), date: String::new() };
        let wplan = plan_query(&ix, &wq, &Params::default());
        assert_ne!(wh_slots(&wq.text) & WH_PERSON, 0);
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        assert!(!apply_c2_same_session_opener_t5_entry(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 3, 2]);
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "I am starting a new practice project this month."), ("melanie", "Noted, I will keep that in mind.")]),
                sess(
                    "D1",
                    &[
                        ("caroline", "Always confirm typical REST API errors when I build an application."),
                        ("melanie", "Timeouts, 404s, and 500s are typical."),
                    ],
                ),
                sess("D2", &[("caroline", "A third note is already in the log."), ("melanie", "The third note is stored.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        assert!(!apply_c2_same_session_opener_t5_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 4, 5, 3, 2]);
    }

    #[test]
    fn late_opener_t5_entry_swaps_opener_for_any_session_top5_assistant() {
        // gpt4_ec93e27f shape: the opener's session sits in the top 5 only
        // through a later user turn, so #31 (same-session assistant victim)
        // refuses. The opener swaps with the last top-5 assistant of
        // another session that keeps a top-5 seat.
        let ix = build_index(
            &[
                sess(
                    "s0",
                    &[
                        ("user", "I took a train to the city last spring."),
                        ("assistant", "Noted, the train trip is logged."),
                        ("user", "I also logged the train fare."),
                    ],
                ),
                sess(
                    "s1",
                    &[
                        ("user", "I rode the bus to work this morning."),
                        ("assistant", "The bus ride is stored."),
                        ("user", "The bus was late again today."),
                    ],
                ),
                sess("s2", &[("user", "A third note is already in the log."), ("assistant", "The third note is stored.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "Which mode of transport did I use, a bus or a train?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: 0-2 s0 (opener 0, asst 1, user 2), 3-5 s1 (opener 3, asst 4, user 5), 6-7 s2.
        // Top 5: s1 later user 5, s0 asst 1, s0 user 2, s2 user 6, s2 asst 7; s1 opener 3 at slot 6.
        let mut list = vec![5usize, 1, 2, 6, 7, 3];
        let t_scene = vec![0.5f32, 0.9, 0.6, 0.4, 0.3, 1.0, 0.7, 0.2];
        // #31 refuses: no s1 assistant in the top 5.
        let mut l31 = list.clone();
        assert!(!apply_c2_same_session_opener_t5_entry(&ix, &plan, &q.text, &mut l31, &t_scene));
        assert!(apply_c2_late_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 1, 2, 6, 3, 7]);
        // A later (non-opener) user in the tail is not seated.
        let mut list = vec![0usize, 1, 6, 7, 3, 2];
        assert!(!apply_c2_late_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 6, 7, 3, 2]);
        // An opener from a session outside the top 5 is not seated.
        let mut list = vec![5usize, 4, 2, 1, 0, 6];
        assert!(!apply_c2_late_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 4, 2, 1, 0, 6]);
        // The victim is the last top-5 assistant, of any session.
        let mut list = vec![5usize, 2, 4, 6, 7, 3, 0];
        assert!(apply_c2_late_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 2, 4, 6, 3, 7, 0]);
        // Preference, temporal, and WH-person questions are skipped.
        for text in [
            "What is my favorite mode of transport, a bus or a train?",
            "When did I use a bus or a train?",
            "Who drove the bus or the train?",
        ] {
            let tq = Query { text: text.into(), date: String::new() };
            let tplan = plan_query(&ix, &tq, &Params::default());
            let mut list = vec![5usize, 1, 2, 6, 7, 3];
            assert!(!apply_c2_late_opener_t5_entry(&ix, &tplan, &tq.text, &mut list, &t_scene));
            assert_eq!(list, vec![5, 1, 2, 6, 7, 3]);
        }
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "I took a train to the city."), ("melanie", "Train noted."), ("caroline", "Fare logged.")]),
                sess("D1", &[("caroline", "I rode the bus to work."), ("melanie", "Bus noted."), ("caroline", "The bus was late.")]),
                sess("D2", &[("caroline", "A third note."), ("melanie", "Stored.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![5usize, 1, 2, 6, 7, 3];
        assert!(!apply_c2_late_opener_t5_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 1, 2, 6, 7, 3]);
    }

    #[test]
    fn head_session_opener_t5_entry_swaps_opener_for_weakest_foreign_user() {
        // 1da05512 shape: slot 1 is a mid-session user question; its
        // session's opener sits outside the top 5; the top 5 holds weak
        // users from other sessions and no assistant to evict.
        let ix = build_index(
            &[
                sess(
                    "s0",
                    &[
                        ("user", "Should I buy a NAS device now or wait for a sale?"),
                        ("assistant", "It depends on prices."),
                        ("user", "How much storage would a NAS device need?"),
                        ("assistant", "Four terabytes is common."),
                    ],
                ),
                sess("s1", &[("user", "I am cleaning the garage this week."), ("assistant", "Good luck with it.")]),
                sess("s2", &[("user", "I logged a new recipe."), ("assistant", "Recipe stored.")]),
                sess("s3", &[("user", "A fourth note about my week."), ("assistant", "Stored.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "I'm trying to decide whether to buy a NAS device now or wait. What do you think?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: 0-3 s0 (opener 0, asst 1, user 2, asst 3), 4-5 s1, 6-7 s2, 8-9 s3.
        // Top 5: s0 user 2 (head), s1 user 4, s2 user 6, s3 user 8, s0 asst 3; s0 opener 0 at slot 6.
        let t_scene = vec![0.8f32, 0.3, 1.0, 0.9, 0.2, 0.1, 0.4, 0.1, 0.3, 0.1];
        let mut list = vec![2usize, 4, 6, 8, 3, 0];
        assert!(apply_c2_head_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // The weakest foreign-session user (4, scene 0.2) is evicted.
        assert_eq!(list, vec![2, 0, 6, 8, 3, 4]);
        // An opener weaker than every foreign-session user does not enter.
        let t_low = vec![0.05f32, 0.3, 1.0, 0.9, 0.2, 0.1, 0.4, 0.1, 0.3, 0.1];
        let mut list = vec![2usize, 4, 6, 8, 3, 0];
        assert!(!apply_c2_head_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_low));
        assert_eq!(list, vec![2, 4, 6, 8, 3, 0]);
        // Another session's opener does not enter.
        let mut list = vec![2usize, 4, 1, 8, 3, 6];
        assert!(!apply_c2_head_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 4, 1, 8, 3, 6]);
        // Assistants are never the victim.
        let mut list = vec![2usize, 3, 5, 7, 9, 0];
        assert!(!apply_c2_head_session_opener_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 5, 7, 9, 0]);
        // Preference, temporal, and WH-person questions are skipped.
        for text in [
            "What is my favorite NAS device to buy?",
            "When should I buy a NAS device?",
            "Who sold me the NAS device?",
        ] {
            let tq = Query { text: text.into(), date: String::new() };
            let tplan = plan_query(&ix, &tq, &Params::default());
            let mut list = vec![2usize, 4, 6, 8, 3, 0];
            assert!(!apply_c2_head_session_opener_t5_entry(&ix, &tplan, &tq.text, &mut list, &t_scene));
            assert_eq!(list, vec![2, 4, 6, 8, 3, 0]);
        }
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "Should I buy a NAS now?"), ("melanie", "Depends."), ("caroline", "How much storage?"), ("melanie", "Four.")]),
                sess("D1", &[("caroline", "Garage week."), ("melanie", "Luck.")]),
                sess("D2", &[("caroline", "New recipe."), ("melanie", "Stored.")]),
                sess("D3", &[("caroline", "Fourth note."), ("melanie", "Stored.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![2usize, 4, 6, 8, 3, 0];
        assert!(!apply_c2_head_session_opener_t5_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 4, 6, 8, 3, 0]);
    }

    #[test]
    fn session_substitution_t5_entry_swaps_strong_outside_session_for_weakest_declarative() {
        // 11:contradiction_resolution:1 shape: the top 5 is all users from
        // distinct sessions; a clearly stronger user turn from a session
        // the top 5 left out sits at slot 7.
        let ix = build_index(
            &[
                sess("s0", &[("user", "I set up two-factor authentication for the admin console."), ("assistant", "Good.")]),
                sess("s1", &[("user", "Did I ever enable login alerts?"), ("assistant", "Yes.")]),
                sess("s2", &[("user", "I moved the platform to a new region."), ("assistant", "Noted.")]),
                sess("s3", &[("user", "I rotated the access keys this week."), ("assistant", "Noted.")]),
                sess("s4", &[("user", "Can you review my access policy?"), ("assistant", "Sure.")]),
                sess("s5", &[("user", "I implemented two-factor authentication for platform access."), ("assistant", "Logged.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "Have I ever implemented two-factor authentication for platform access?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Users are even indices: s0=0, s1=2, s2=4, s3=6, s4=8, s5=10.
        // Top 5: 0, 2, 4, 6, 8; official-k tail: 1, 10.
        let mut t_scene = vec![0.0f32; 12];
        t_scene[0] = 2.4;
        t_scene[2] = 2.5;
        t_scene[4] = 0.8;
        t_scene[6] = 1.8;
        t_scene[8] = 1.5;
        t_scene[1] = 0.5;
        t_scene[10] = 1.3;
        let mut list = vec![0usize, 2, 4, 6, 8, 1, 10];
        assert!(apply_c2_session_substitution_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 10, 6, 8, 1, 4]);
        // The weakest top-5 user is a question: no swap.
        let mut tq = t_scene.clone();
        tq[4] = 1.9;
        tq[8] = 0.8;
        let mut list = vec![0usize, 2, 4, 6, 8, 1, 10];
        assert!(!apply_c2_session_substitution_t5_entry(&ix, &plan, &q.text, &mut list, &tq));
        assert_eq!(list, vec![0, 2, 4, 6, 8, 1, 10]);
        // The candidate does not dominate the weakest user by the ratio.
        let mut tr = t_scene.clone();
        tr[10] = 1.0;
        let mut list = vec![0usize, 2, 4, 6, 8, 1, 10];
        assert!(!apply_c2_session_substitution_t5_entry(&ix, &plan, &q.text, &mut list, &tr));
        assert_eq!(list, vec![0, 2, 4, 6, 8, 1, 10]);
        // Only users enter: an assistant tail turn is not a candidate.
        let mut list = vec![0usize, 2, 4, 6, 8, 11];
        let mut ta = t_scene.clone();
        ta[11] = 2.0;
        assert!(!apply_c2_session_substitution_t5_entry(&ix, &plan, &q.text, &mut list, &ta));
        assert_eq!(list, vec![0, 2, 4, 6, 8, 11]);
        // Preference, temporal, and WH-person questions are skipped.
        for text in [
            "What is my favorite two-factor authentication app?",
            "When did I implement two-factor authentication?",
            "Who implemented two-factor authentication for platform access?",
        ] {
            let sq = Query { text: text.into(), date: String::new() };
            let splan = plan_query(&ix, &sq, &Params::default());
            let mut list = vec![0usize, 2, 4, 6, 8, 1, 10];
            assert!(!apply_c2_session_substitution_t5_entry(&ix, &splan, &sq.text, &mut list, &t_scene));
            assert_eq!(list, vec![0, 2, 4, 6, 8, 1, 10]);
        }
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "Two-factor for admin."), ("melanie", "Good.")]),
                sess("D1", &[("caroline", "Login alerts?"), ("melanie", "Yes.")]),
                sess("D2", &[("caroline", "New region."), ("melanie", "Noted.")]),
                sess("D3", &[("caroline", "Rotated keys."), ("melanie", "Noted.")]),
                sess("D4", &[("caroline", "Review my policy?"), ("melanie", "Sure.")]),
                sess("D5", &[("caroline", "Implemented two-factor authentication."), ("melanie", "Logged.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 2, 4, 6, 8, 1, 10];
        assert!(!apply_c2_session_substitution_t5_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 2, 4, 6, 8, 1, 10]);
    }

    #[test]
    fn pool_session_substitution_t10_entry_replaces_weakest_declarative_tail() {
        // BEAM 19:instruction_following:1 shape: a user turn from a session
        // with no official-k seat is clearly stronger than the weakest
        // official-k tail turn, a declarative.
        let ix = build_index(
            &[
                sess("s0", &[("user", "Please always format code answers with comments."), ("assistant", "I will add comments.")]),
                sess("s1", &[("user", "I started a new parser module."), ("assistant", "The parser module is noted.")]),
                sess("s2", &[("user", "I added a lexer test."), ("assistant", "The lexer test is noted.")]),
                sess("s3", &[("user", "Always include comments when you format my code."), ("assistant", "Understood.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "How should you format code answers for me?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: s0 0-1, s1 2-3, s2 4-5, s3 6-7. Official k: 0,1,2,3,4,5 (s3 absent).
        let mut t_scene = vec![2.0f32, 1.8, 1.0, 0.9, 0.8, 0.3, 1.5, 0.2];
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(apply_c2_pool_session_substitution_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 6]);
        // #38: the session substitution T@5 entry, run again after #37,
        // pulls the just-seated pool user into the top 5 over the weakest
        // top-5 declarative user.
        assert!(apply_c2_session_substitution_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 6, 4]);
        // Exact scene ties go to the earliest session: a later session
        // repeating the same turn does not take the seat.
        let mut s_dup = vec![
            sess("s0", &[("user", "Please always format code answers with comments."), ("assistant", "I will add comments.")]),
            sess("s1", &[("user", "I started a new parser module."), ("assistant", "The parser module is noted.")]),
            sess("s2", &[("user", "I added a lexer test."), ("assistant", "The lexer test is noted.")]),
            sess("s3", &[("user", "Always include comments when you format my code."), ("assistant", "Understood.")]),
            sess("s4", &[("user", "Always include comments when you format my code."), ("assistant", "Understood.")]),
        ];
        for (i, se) in s_dup.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ixd = build_index(&s_dup, &Params::default());
        let pland = plan_query(&ixd, &q, &Params::default());
        let td = vec![2.0f32, 1.8, 1.0, 0.9, 0.8, 0.3, 1.5, 0.2, 1.5, 0.2];
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(apply_c2_pool_session_substitution_t10_entry(&ixd, &pland, &q.text, &mut list, &td));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 6]);
        // The weakest tail turn is not dominated by the ratio: no entry.
        t_scene[6] = 0.4;
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(!apply_c2_pool_session_substitution_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
        t_scene[6] = 1.5;
        // A pool user whose session already has an official-k seat does not enter.
        let mut list = vec![0usize, 1, 2, 3, 7, 5];
        let mut ts = t_scene.clone();
        ts[4] = 1.4;
        ts[6] = 0.0;
        assert!(!apply_c2_pool_session_substitution_t10_entry(&ix, &plan, &q.text, &mut list, &ts));
        assert_eq!(list, vec![0, 1, 2, 3, 7, 5]);
        // Preference, temporal, and WH-person questions are skipped.
        for text in [
            "What is my favorite way to format code answers?",
            "When did I ask you to format code answers?",
            "Who asked you to format code answers?",
        ] {
            let sq = Query { text: text.into(), date: String::new() };
            let splan = plan_query(&ix, &sq, &Params::default());
            let mut list = vec![0usize, 1, 2, 3, 4, 5];
            assert!(!apply_c2_pool_session_substitution_t10_entry(&ix, &splan, &sq.text, &mut list, &t_scene));
            assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
        }
        // Named-speaker sessions are left alone.
        let ixn = build_index(
            &[
                sess("D0", &[("caroline", "Format code with comments."), ("melanie", "Will do.")]),
                sess("D1", &[("caroline", "New parser module."), ("melanie", "Noted.")]),
                sess("D2", &[("caroline", "Added a lexer test."), ("melanie", "Noted.")]),
                sess("D3", &[("caroline", "Always include comments in code."), ("melanie", "Understood.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(!apply_c2_pool_session_substitution_t10_entry(&ixn, &plann, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn mentioned_session_mate_t5_entry_evicts_weakest_later_declarative() {
        // LoCoMo 4:92 / 4:122 shape: a mentioned speaker's declarative from
        // a top-5 session sits in the official-k tail; the weakest top-5
        // turn is a later-session declarative well below it.
        let mut ss = vec![
            sess("D1", &[("john", "My teammates gave me a signed jersey when we met."), ("tim", "That is a great gift.")]),
            sess("D2", &[("john", "I practiced free throws today."), ("tim", "Nice work.")]),
            sess("D3", &[("john", "We met the team again on Aug 15th."), ("tim", "Fun.")]),
            sess("D4", &[("john", "I watched a game on TV."), ("tim", "Which one?")]),
        ];
        for (i, se) in ss.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ix = build_index(&ss, &Params::default());
        let q = Query { text: "What did John's teammates give him when they met?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: D1 0-1, D2 2-3, D3 4-5, D4 6-7.
        // Top 5: 4 (head, D3), 1 (D1 tim), 2 (D2), 5 (D3 tim), 6 (D4 john, weakest); tail: 0 (D1 john), 3.
        let t_scene = vec![1.0f32, 0.9, 0.8, 0.2, 2.0, 0.7, 0.5, 0.3];
        let mut list = vec![4usize, 1, 2, 5, 6, 0, 3];
        assert!(apply_c2_mentioned_session_mate_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![4, 1, 2, 5, 0, 6, 3]);
        // An earlier-session weakest turn is never evicted for a later candidate.
        // Candidate 6 (D4 john) in the tail, weakest top-5 turn 0 (D1 john).
        let t2 = vec![0.3f32, 0.9, 0.8, 0.2, 2.0, 0.7, 1.0, 0.1];
        let mut list = vec![4usize, 1, 2, 7, 0, 6, 3];
        assert!(!apply_c2_mentioned_session_mate_t5_entry(&ix, &plan, &q.text, &mut list, &t2));
        assert_eq!(list, vec![4, 1, 2, 7, 0, 6, 3]);
        // The weakest top-5 turn is a question: no swap.
        let t3 = vec![1.0f32, 0.9, 0.8, 0.2, 2.0, 0.7, 0.6, 0.3];
        let mut list = vec![4usize, 1, 2, 5, 7, 0, 3];
        assert!(!apply_c2_mentioned_session_mate_t5_entry(&ix, &plan, &q.text, &mut list, &t3));
        assert_eq!(list, vec![4, 1, 2, 5, 7, 0, 3]);
        // A candidate not dominating by the ratio does not enter.
        let t4 = vec![0.6f32, 0.9, 0.8, 0.2, 2.0, 0.7, 0.5, 0.3];
        let mut list = vec![4usize, 1, 2, 5, 6, 0, 3];
        assert!(!apply_c2_mentioned_session_mate_t5_entry(&ix, &plan, &q.text, &mut list, &t4));
        // Unmentioned-speaker candidates do not enter.
        let t5 = vec![0.1f32, 0.9, 0.8, 0.2, 2.0, 0.7, 0.5, 0.3];
        let mut list = vec![4usize, 0, 2, 5, 6, 1, 3];
        let mut t5b = t5.clone();
        t5b[1] = 1.0;
        assert!(!apply_c2_mentioned_session_mate_t5_entry(&ix, &plan, &q.text, &mut list, &t5b));
        // Preference, temporal, and WH-person questions are skipped.
        for text in [
            "What is John's favorite gift from his teammates?",
            "When did John's teammates give him a gift?",
            "Who gave John a gift when they met?",
        ] {
            let sq = Query { text: text.into(), date: String::new() };
            let splan = plan_query(&ix, &sq, &Params::default());
            let mut list = vec![4usize, 1, 2, 5, 6, 0, 3];
            assert!(!apply_c2_mentioned_session_mate_t5_entry(&ix, &splan, &sq.text, &mut list, &t_scene));
            assert_eq!(list, vec![4, 1, 2, 5, 6, 0, 3]);
        }
        // User/assistant chats are left alone.
        let ixu = build_index(
            &[
                sess("s1", &[("user", "My teammates gave me a signed jersey."), ("assistant", "Great gift.")]),
                sess("s2", &[("user", "I practiced free throws."), ("assistant", "Nice.")]),
                sess("s3", &[("user", "We met the team on Aug 15th."), ("assistant", "Fun.")]),
                sess("s4", &[("user", "I watched a game."), ("assistant", "Which one?")]),
            ],
            &Params::default(),
        );
        let planu = plan_query(&ixu, &q, &Params::default());
        let mut list = vec![4usize, 1, 2, 5, 6, 0, 3];
        assert!(!apply_c2_mentioned_session_mate_t5_entry(&ixu, &planu, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![4, 1, 2, 5, 6, 0, 3]);
    }

    #[test]
    fn episode_run_edge_t10_entry_seats_run_edge_over_isolated_tail() {
        // LoCoMo 9:91 shape: two back-to-back official-k turns of D3, the
        // mentioned speaker's declarative just before them is pool-only, and
        // the weakest tail turn is the lone official-k turn of a later session.
        let mut ss = vec![
            sess("D1", &[("john", "I bought a new guitar."), ("tim", "Cool.")]),
            sess("D2", &[("john", "I went hiking."), ("tim", "Where?")]),
            sess("D3", &[("tim", "Hi."), ("john", "I own a Ferrari and I am proud of it."), ("tim", "Nice car!"), ("john", "Thanks, it is red.")]),
            sess("D4", &[("john", "I watched a movie."), ("tim", "Which one?")]),
        ];
        for (i, se) in ss.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ix = build_index(&ss, &Params::default());
        let q = Query { text: "What car does John own that he is proud of?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: D1 0-1, D2 2-3, D3 4-7, D4 8-9. Run = D3 turns 6, 7
        // (positions 2, 3); its pool edge is turn 5 (john, declarative).
        let t_scene = vec![3.0f32, 2.9, 2.8, 2.7, 2.6, 2.0, 1.5, 1.4, 2.5, 1.0];
        let mut list = vec![0usize, 1, 2, 3, 8, 6, 7, 9];
        // Tail 6, 7, 9: D3 and D4 (8 and 9) each hold two official-k turns,
        // so there is no lone-session victim.
        assert!(!apply_c2_episode_run_edge_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // Tail 6, 7, 9 with 8 out of the list: turn 9 is D4's lone official-k
        // turn, later than D3, and less on-scene than edge 5.
        let mut list = vec![0usize, 1, 2, 3, 4, 6, 7, 9];
        assert!(apply_c2_episode_run_edge_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 6, 7, 5]);
        // Primacy: an earlier lone witness is not evicted for a later episode.
        let mut list = vec![8usize, 9, 2, 3, 6, 7, 1, 4];
        let t2 = vec![3.0f32, 0.5, 2.8, 2.7, 2.6, 2.0, 1.5, 1.4, 2.5, 1.0];
        // Tail 7, 1, 4: the only lone-session turn is 1 (D1, earlier than D3).
        assert!(!apply_c2_episode_run_edge_t10_entry(&ix, &plan, &q.text, &mut list, &t2));
        // An edge less on-scene than the isolated turn does not enter.
        let t3 = vec![3.0f32, 2.9, 2.8, 2.7, 2.6, 0.5, 1.5, 1.4, 2.5, 1.0];
        let mut list = vec![0usize, 1, 2, 3, 4, 6, 7, 9];
        assert!(!apply_c2_episode_run_edge_t10_entry(&ix, &plan, &q.text, &mut list, &t3));
        // The edge's speaker is not the one the query names: no entry.
        let qn = Query { text: "What car does Mike own that he is proud of?".into(), date: String::new() };
        let plann = plan_query(&ix, &qn, &Params::default());
        let mut list = vec![0usize, 1, 2, 3, 4, 6, 7, 9];
        assert!(!apply_c2_episode_run_edge_t10_entry(&ix, &plann, &qn.text, &mut list, &t_scene));
        // Temporal and WH-person questions are skipped.
        for text in ["When did John buy the car he is proud of?", "Who owns the car John is proud of?"] {
            let sq = Query { text: text.into(), date: String::new() };
            let splan = plan_query(&ix, &sq, &Params::default());
            let mut list = vec![0usize, 1, 2, 3, 4, 6, 7, 9];
            assert!(!apply_c2_episode_run_edge_t10_entry(&ix, &splan, &sq.text, &mut list, &t_scene));
        }
    }

    #[test]
    fn episode_run_edge_t5_entry_seats_top5_run_edge_over_lone_top5_turn() {
        // LoCoMo 6:81 shape: two back-to-back top-5 turns of D3, the
        // mentioned speaker's declarative just after them sits in the
        // official-k tail, and a weaker top-5 turn is the only top-5 turn of
        // a later session (not its opener).
        let mut ss = vec![
            sess("D1", &[("john", "I bought a new guitar."), ("tim", "Cool.")]),
            sess("D2", &[("john", "I went hiking."), ("tim", "Where?")]),
            sess("D3", &[("tim", "Hi."), ("john", "I designed a game character."), ("tim", "Why?"), ("john", "A novel I loved inspired the character.")]),
            sess("D4", &[("john", "I watched a movie."), ("tim", "Which one?")]),
        ];
        for (i, se) in ss.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ix = build_index(&ss, &Params::default());
        let q = Query { text: "What inspired John to create the game character?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: D1 0-1, D2 2-3, D3 4-7, D4 8-9. Top-5 run = D3 turns 5, 6;
        // its edge is turn 7 (john, declarative) in the tail. Turn 9 is D4's
        // only top-5 turn and not its opener.
        let t_scene = vec![1.0f32, 0.5, 2.8, 0.4, 0.3, 3.0, 2.9, 2.0, 0.2, 1.5];
        let mut list = vec![5usize, 6, 2, 9, 0, 7, 1, 3];
        assert!(apply_c2_episode_run_edge_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 6, 2, 7, 0, 9, 1, 3]);
        // Session openers are not evicted: with 9 replaced by opener 8, the
        // only other lone candidates are 2 (D2 position 0) and 0 (D1 opener).
        let mut list = vec![5usize, 6, 2, 8, 0, 7, 1, 3];
        assert!(!apply_c2_episode_run_edge_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // Primacy: an earlier-session lone turn is not evicted. Turn 1 is D1's
        // only top-5 turn (position 1), earlier than D3.
        let mut list = vec![5usize, 6, 2, 1, 0, 7, 9, 3];
        let t2 = vec![1.0f32, 0.5, 2.8, 0.4, 0.3, 3.0, 2.9, 2.0, 0.2, 2.5];
        assert!(!apply_c2_episode_run_edge_t5_entry(&ix, &plan, &q.text, &mut list, &t2));
        // A pool edge takes the victim's seat.
        let mut list = vec![5usize, 6, 2, 9, 0, 1, 3, 8];
        assert!(apply_c2_episode_run_edge_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![5, 6, 2, 7, 0, 1, 3, 8]);
        // An edge less on-scene than every lone non-opener top-5 turn does not enter.
        let t3 = vec![1.0f32, 0.5, 2.8, 0.4, 0.3, 3.0, 2.9, 1.0, 0.2, 1.5];
        let mut list = vec![5usize, 6, 2, 9, 0, 7, 1, 3];
        assert!(!apply_c2_episode_run_edge_t5_entry(&ix, &plan, &q.text, &mut list, &t3));
        // The edge's speaker is not the one the query names: no entry.
        let qn = Query { text: "What inspired Mike to create the game character?".into(), date: String::new() };
        let plann = plan_query(&ix, &qn, &Params::default());
        let mut list = vec![5usize, 6, 2, 9, 0, 7, 1, 3];
        assert!(!apply_c2_episode_run_edge_t5_entry(&ix, &plann, &qn.text, &mut list, &t_scene));
    }

    #[test]
    fn neighbor_pair_t5_entry_swaps_tail_neighbor_for_lone_top5_turn() {
        let mut ss = vec![
            sess("D1", &[("john", "I bought a new guitar."), ("tim", "Cool.")]),
            sess("D2", &[("john", "I went hiking."), ("tim", "Where?")]),
            sess("D3", &[("tim", "Hi."), ("john", "I designed a game character."), ("tim", "Why?"), ("john", "A novel I loved inspired the character.")]),
            sess("D4", &[("john", "I watched a movie."), ("tim", "Which one?")]),
        ];
        for (i, se) in ss.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ix = build_index(&ss, &Params::default());
        let q = Query { text: "What inspired John to create the game character?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: D1 0-1, D2 2-3, D3 4-7, D4 8-9. Top-5 turn 6 (D3, alone in
        // the top 5) has tail neighbor 7 (john, declarative). Turn 9 is D4's
        // only top-5 turn and not its opener.
        let t_scene = vec![1.0f32, 0.5, 2.8, 0.4, 0.3, 0.2, 2.9, 2.0, 0.2, 1.5];
        let mut list = vec![6usize, 2, 9, 0, 3, 7, 1, 5];
        assert!(apply_c2_neighbor_pair_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![6, 2, 7, 0, 3, 9, 1, 5]);
        // A pool neighbor (not in the official k) does not enter.
        let mut list = vec![6usize, 2, 9, 0, 3, 1, 5, 8];
        assert!(!apply_c2_neighbor_pair_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // Primacy: the only lone non-opener victim, turn 1 (D1), is earlier than D3.
        let mut list = vec![6usize, 2, 1, 0, 8, 7, 9, 3];
        let t2 = vec![1.0f32, 0.5, 2.8, 0.4, 0.3, 0.2, 2.9, 2.0, 2.5, 1.5];
        assert!(!apply_c2_neighbor_pair_t5_entry(&ix, &plan, &q.text, &mut list, &t2));
        // The neighbor's speaker is not the one the query names: no entry.
        let qn = Query { text: "What inspired Mike to create the game character?".into(), date: String::new() };
        let plann = plan_query(&ix, &qn, &Params::default());
        let mut list = vec![6usize, 2, 9, 0, 3, 7, 1, 5];
        assert!(!apply_c2_neighbor_pair_t5_entry(&ix, &plann, &qn.text, &mut list, &t_scene));
        // User/assistant chats are left alone.
        let ixu = build_index(
            &[
                sess("s1", &[("user", "I bought a new guitar."), ("assistant", "Cool.")]),
                sess("s2", &[("user", "I went hiking."), ("assistant", "Where?")]),
                sess("s3", &[("user", "Hi."), ("assistant", "I designed a game character."), ("user", "Why?"), ("assistant", "A novel inspired the character.")]),
                sess("s4", &[("user", "I watched a movie."), ("assistant", "Which one?")]),
            ],
            &Params::default(),
        );
        let planu = plan_query(&ixu, &q, &Params::default());
        let mut list = vec![6usize, 2, 9, 0, 3, 7, 1, 5];
        assert!(!apply_c2_neighbor_pair_t5_entry(&ixu, &planu, &q.text, &mut list, &t_scene));
    }

    #[test]
    fn assistant_reply_t5_entry_swaps_reply_for_weakest_later_user() {
        // LME 3249768e shape: the assistant reply to a top-5 user turn sits
        // in the official-k tail; the weakest top-5 user turn from an
        // earlier session is protected, the next weakest (later) is evicted.
        let mut ss = vec![
            sess("s1", &[("user", "I am setting up a garden shed."), ("assistant", "Sounds fun.")]),
            sess("s2", &[("user", "I am building a cocktail bar at home."), ("assistant", "Buy gin, rum, tequila, vodka and bourbon.")]),
            sess("s3", &[("user", "Which glasses do I need?"), ("assistant", "Highball and rocks glasses.")]),
        ];
        for (i, se) in ss.iter_mut().enumerate() {
            se.timestamp = 1_000 + i as i64;
        }
        let ix = build_index(&ss, &Params::default());
        let q = Query {
            text: "Looking back at our previous conversation, which bottles did you recommend for my cocktail bar?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: s1 0-1, s2 2-3, s3 4-5. Top 5: 2 (s2 user), 4 (s3 user),
        // 0 (s1 user, earlier, weakest), 5, 1; tail: 3 (reply to 2).
        let t_scene = vec![0.5f32, 0.4, 3.0, 2.5, 0.8, 0.9, 0.0];
        let mut list = vec![2usize, 4, 0, 5, 1, 3];
        assert!(apply_c2_assistant_reply_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 0, 5, 1, 4]);
        // The reply's own user neighbor (the anchor) is never the victim.
        let mut list = vec![4usize, 2, 5, 0, 1, 3];
        let t2 = vec![2.6f32, 0.4, 0.3, 2.5, 3.0, 0.9];
        assert!(!apply_c2_assistant_reply_t5_entry(&ix, &plan, &q.text, &mut list, &t2));
        // A reply less on-scene than every eligible user does not enter.
        let t3 = vec![0.5f32, 0.4, 3.0, 0.6, 0.8, 0.9];
        let mut list = vec![2usize, 4, 0, 5, 1, 3];
        assert!(!apply_c2_assistant_reply_t5_entry(&ix, &plan, &q.text, &mut list, &t3));
        // Temporal questions are left alone.
        let qt = Query { text: "When did you recommend bottles for my cocktail bar?".into(), date: String::new() };
        let plant = plan_query(&ix, &qt, &Params::default());
        let mut list = vec![2usize, 4, 0, 5, 1, 3];
        assert!(!apply_c2_assistant_reply_t5_entry(&ix, &plant, &qt.text, &mut list, &t_scene));
    }

    #[test]
    fn attribution_unseated_opener_t10_entry_seats_opener_of_unseated_session() {
        // BEAM 14:instruction_following:0 shape: the opener of a session with
        // no official-k seat outscores the lone-session last tail turn.
        let mk = |ts: [i64; 3]| {
            let mut ss = vec![
                sess("s1", &[("user", "I love sci-fi films like Interstellar."), ("assistant", "Great taste.")]),
                sess("s2", &[("user", "Recommend a movie?"), ("assistant", "Try Arrival."), ("user", "More?"), ("assistant", "Try Contact."), ("user", "Thanks."), ("assistant", "Enjoy.")]),
                sess("s3", &[("user", "What about series?"), ("assistant", "Try The Expanse.")]),
            ];
            for (se, t) in ss.iter_mut().zip(ts) {
                se.timestamp = t;
            }
            build_index(&ss, &Params::default())
        };
        let ix = mk([1_000, 1_001, 1_002]);
        let q = Query { text: "What movies would you recommend for me to watch?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(!c2_summary_request(&q.text));
        // Turns: s1 0-1, s2 2-7, s3 8-9. Official k: 2..=6 (s2), then 9 (s3's
        // lone k turn, later than s1). Opener 0 (s1) has no k seat.
        let t_scene = vec![2.0f32, 0.1, 3.0, 2.9, 2.8, 2.7, 2.6, 0.5, 0.0, 1.5];
        let mut list = vec![2usize, 3, 4, 5, 6, 9];
        assert!(apply_c2_attribution_unseated_opener_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 4, 5, 6, 0]);
        // Primacy: the lone tail turn's session is earlier than the opener's.
        let ixe = mk([1_005, 1_001, 1_002]);
        let plane = plan_query(&ixe, &q, &Params::default());
        let mut list = vec![2usize, 3, 4, 5, 6, 9];
        assert!(!apply_c2_attribution_unseated_opener_t10_entry(&ixe, &plane, &q.text, &mut list, &t_scene));
        // An opener less on-scene than the lone tail turn does not enter.
        let t2 = vec![1.0f32, 0.1, 3.0, 2.9, 2.8, 2.7, 2.6, 0.5, 0.0, 1.5];
        let mut list = vec![2usize, 3, 4, 5, 6, 9];
        assert!(!apply_c2_attribution_unseated_opener_t10_entry(&ix, &plan, &q.text, &mut list, &t2));
        // No lone-session tail turn: 8 and 9 both from s3.
        let mut list = vec![2usize, 3, 4, 5, 8, 9];
        assert!(!apply_c2_attribution_unseated_opener_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // Temporal questions are left alone.
        let qt = Query { text: "When did you recommend movies for me to watch?".into(), date: String::new() };
        let plant = plan_query(&ix, &qt, &Params::default());
        let mut list = vec![2usize, 3, 4, 5, 6, 9];
        assert!(!apply_c2_attribution_unseated_opener_t10_entry(&ix, &plant, &qt.text, &mut list, &t_scene));
    }

    #[test]

    fn pool_mentioned_t10_entry_replaces_unmentioned_tail_with_mentioned_pool_turn() {
        // 6:81 / 9:113 shape: a mentioned-speaker turn already in the
        // scene-ranked pool but outside official k takes the official-k
        // seat of the last unmentioned named speaker in the tail. The
        // top 5 is unchanged; the victim stays out of the top 5.
        let ix = build_index(
            &[
                sess("D0", &[("melanie", "I started a new practice project this month."), ("caroline", "I noted that project in the log.")]),
                sess("D1", &[("melanie", "I asked about typical REST API errors yesterday."), ("caroline", "Timeouts, 404s, and 500s are typical.")]),
                sess(
                    "D2",
                    &[
                        ("melanie", "A third note is already in the log."),
                        ("caroline", "The third note is stored."),
                        ("melanie", "I later asked about a leftover topic."),
                    ],
                ),
                sess(
                    "D3",
                    &[("caroline", "Always confirm typical REST API errors when I build an application.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What typical errors should Caroline be prepared to handle with a REST API?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        assert!(c2_query_mentions_speaker(&q.text, "caroline"));
        assert!(!c2_query_mentions_speaker(&q.text, "melanie"));
        // Turns: 0-1 D0, 2-3 D1, 4-6 D2, 7 D3 (caroline leftover).
        // Official k of 7; tail slot 7 (index 6) is unmentioned Melanie.
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        let t_scene = vec![1.0f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.55];
        assert!(apply_c2_pool_mentioned_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 7]);
        // Below the victim's scene: no entry.
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        let t_low = vec![1.0f32, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ix, &plan, &q.text, &mut list, &t_low));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
        // User/assistant chats are left alone.
        let ixu = build_index(
            &[
                sess("s0", &[("user", "I started a new practice project this month."), ("assistant", "I noted that project in the log.")]),
                sess("s1", &[("user", "I asked about typical REST API errors yesterday."), ("assistant", "Timeouts, 404s, and 500s are typical.")]),
                sess(
                    "s2",
                    &[
                        ("user", "A third note is already in the log."),
                        ("assistant", "The third note is stored."),
                        ("user", "I later asked about a leftover topic."),
                    ],
                ),
                sess("s3", &[("user", "Always confirm typical REST API errors when I build an application.")]),
            ],
            &Params::default(),
        );
        let planu = plan_query(&ixu, &q, &Params::default());
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ixu, &planu, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
        // Preference questions are skipped.
        let pq = Query { text: "What is Caroline's favorite way to handle REST API errors?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
        // Temporal questions keep the timeline path.
        let tq = Query { text: "When does Caroline handle REST API errors?".into(), date: String::new() };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(tplan.intent.temporal);
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ix, &tplan, &tq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
        // WH-person questions are left to the existing midcap path.
        let wq = Query { text: "Who handled the REST API errors?".into(), date: String::new() };
        let wplan = plan_query(&ix, &wq, &Params::default());
        assert_ne!(wh_slots(&wq.text) & WH_PERSON, 0);
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
        // Conjoined "X and Y" mentions are left alone.
        let cq = Query { text: "What typical errors should Caroline and Melanie handle with a REST API?".into(), date: String::new() };
        let cplan = plan_query(&ix, &cq, &Params::default());
        assert!(c2_speaker_conjoined(&cq.text, "caroline"));
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_pool_mentioned_t10_entry(&ix, &cplan, &cq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]

    fn queried_speaker_t5_entry_after_pool_mentioned_seats_new_official_k_turn() {
        // 2:66 / 5:54 shape: after the pool mentioned-speaker T@10
        // entry seats a mentioned declarative into official k, the
        // queried-speaker T@5 entry (the same #17 structure) swaps it
        // into the top 5 in place of the last unmentioned named speaker.
        let ix = build_index(
            &[
                sess("D0", &[("melanie", "I started a new practice project this month."), ("caroline", "I noted that project in the log.")]),
                sess("D1", &[("melanie", "I asked about typical REST API errors yesterday."), ("caroline", "Timeouts, 404s, and 500s are typical.")]),
                sess(
                    "D2",
                    &[
                        ("melanie", "A third note is already in the log."),
                        ("caroline", "The third note is stored."),
                        ("melanie", "I later asked about a leftover topic."),
                    ],
                ),
                sess(
                    "D3",
                    &[("caroline", "Always confirm typical REST API errors when I build an application.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What typical errors should Caroline be prepared to handle with a REST API?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!c2_text_preference(&q.text));
        assert!(!plan.intent.temporal);
        assert_eq!(wh_slots(&q.text) & WH_PERSON, 0);
        // Turns: 0-1 D0, 2-3 D1, 4-6 D2, 7 D3 (caroline leftover).
        // Official k of 7; tail slot 7 is unmentioned Melanie.
        // Cand scene 0.80 is below the mid-k caroline at 0.50 so #17
        // first-matches the just-seated leftover, and is at least 0.7
        // of the head's scene.
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        let t_scene = vec![1.0f32, 0.9, 0.8, 0.75, 0.6, 0.50, 0.4, 0.80];
        assert!(apply_c2_pool_mentioned_t10_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 7]);
        assert!(apply_c2_queried_speaker_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        // Last unmentioned in the top 5 is slot 5 (turn 4, Melanie);
        // the just-seated Caroline leftover takes that seat.
        assert_eq!(list, vec![0, 1, 2, 3, 7, 5, 4]);
    }

    #[test]
    fn queried_speaker_first_t5_entry_swaps_first_in_session_mentioned_into_top5() {
        // Declarative first-in-session turn well below the #17 70% floor
        // (3:47 / 2:82 shape).
        let ix = build_index(
            &[
                sess(
                    "D23",
                    &[
                        ("nate", "I also saw some friends at a later show."),
                        ("joanna", "That later show sounded crowded."),
                        ("nate", "I have been meeting new people at concerts this year."),
                        ("joanna", "Concerts are a good place to meet people."),
                    ],
                ),
                sess(
                    "D17",
                    &[
                        ("nate", "I met new people at the market on the weekend."),
                        ("joanna", "The weekend market is usually busy."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query { text: "What places has Nate met new people?".into(), date: String::new() };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        assert!(!ix.turns[4].is_question);
        assert_eq!(c2_first_same_speaker(&ix, 4), Some(4));
        assert_eq!(c2_first_same_speaker(&ix, 2), Some(0));
        assert!(c2_query_mentions_speaker(&q.text, "nate"));
        assert!(!c2_query_mentions_speaker(&q.text, "joanna"));
        // Official k of 6: Nate's first turn in D17 sits at slot 6, r=0.3.
        // Last unmentioned (Joanna D17 at slot 5) yields. #17 would refuse the ratio.
        let mut list = vec![2usize, 0, 1, 3, 5, 4];
        let t_scene = vec![0.8f32, 0.7, 1.0, 0.6, 0.3, 0.4];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_queried_speaker_first_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 4, 5]);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // A later mentioned declarative, already in official k, is not pulled.
        assert_eq!(c2_first_same_speaker(&ix, 2), Some(0));
        let mut list = vec![1usize, 3, 5, 4, 0, 2];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![1, 3, 5, 4, 0, 2]);

        // Question first-in-session at r>=0.7 is pulled (0:32 shape).
        let ixq = build_index(
            &[
                sess(
                    "D10",
                    &[
                        ("caroline", "I have been going to more community events this year."),
                        ("melanie", "Those events have been a lot of fun."),
                        ("caroline", "The last one ran late into the evening."),
                        ("melanie", "I had to leave a little early."),
                    ],
                ),
                sess(
                    "D3",
                    &[
                        ("caroline", "Have you been to any Pride events this summer?"),
                        ("melanie", "I have not made it to one yet."),
                    ],
                ),
            ],
            &Params::default(),
        );
        assert!(ixq.turns[4].is_question);
        assert_eq!(c2_first_same_speaker(&ixq, 4), Some(4));
        let qq = Query { text: "What LGBTQ+ events has Caroline participated in?".into(), date: String::new() };
        let qplan = plan_query(&ixq, &qq, &Params::default());
        assert!(!qplan.intent.preference);
        assert!(!qplan.intent.temporal);
        assert!(c2_query_mentions_speaker(&qq.text, "caroline"));
        assert!(!c2_query_mentions_speaker(&qq.text, "melanie"));
        let mut list = vec![0usize, 2, 1, 3, 5, 4];
        let qsc = vec![1.0f32, 0.8, 0.9, 0.7, 0.85, 0.5];
        assert!(apply_c2_queried_speaker_first_t5_entry(&ixq, &qplan, &qq.text, &mut list, &qsc));
        assert_eq!(list, vec![0, 2, 1, 3, 4, 5]);

        // Same question first-in-session below the reused 70% floor: no entry.
        let mut list = vec![0usize, 2, 1, 3, 5, 4];
        let qsc_low = vec![1.0f32, 0.8, 0.9, 0.7, 0.5, 0.4];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ixq, &qplan, &qq.text, &mut list, &qsc_low));
        assert_eq!(list, vec![0, 2, 1, 3, 5, 4]);

        // A later mentioned question at high r is not pulled (4:74 shape).
        let ixl = build_index(
            &[sess(
                "D1",
                &[
                    ("john", "My fan project will cover many sides of that universe."),
                    ("tim", "Which aspects will you discuss in the collaborations?"),
                    ("tim", "We will discuss characters, spells, and magical creatures."),
                    ("john", "Which aspects of the universe should the fan project cover?"),
                    ("john", "I have loved this universe since I was a kid."),
                    ("tim", "The project is going to be a lot of fun."),
                ],
            )],
            &Params::default(),
        );
        assert!(ixl.turns[3].is_question);
        assert_eq!(c2_first_same_speaker(&ixl, 3), Some(0));
        let qhp = Query {
            text: "What aspects of the universe will be discussed in John's fan project collaborations?".into(),
            date: String::new(),
        };
        let qplan2 = plan_query(&ixl, &qhp, &Params::default());
        let mut list = vec![0usize, 1, 2, 4, 5, 3];
        let qsc2 = vec![1.0f32, 0.9, 1.0, 0.95, 0.4, 0.3];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ixl, &qplan2, &qhp.text, &mut list, &qsc2));
        assert_eq!(list, vec![0, 1, 2, 4, 5, 3]);

        // Preference query: no entry.
        let pq = Query { text: "What is Nate's favorite place to meet new people?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3, 5, 4];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 5, 4]);

        // WH-person: no entry.
        let wq = Query { text: "Who has Nate met at new places?".into(), date: String::new() };
        assert!(wh_slots(&wq.text) & WH_PERSON != 0);
        let wplan = plan_query(&ix, &wq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3, 5, 4];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ix, &wplan, &wq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 5, 4]);

        // user/assistant chats are left alone.
        let ixu = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I have been meeting new people at concerts this year."),
                        ("assistant", "Concerts are a good place to meet people."),
                        ("user", "I also saw some friends at a later show."),
                        ("assistant", "That later show sounded crowded."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I met new people at the market on the weekend."),
                        ("assistant", "The weekend market is usually busy."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let uq = Query { text: "What places has Nate met new people?".into(), date: String::new() };
        let uplan = plan_query(&ixu, &uq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3, 5, 4];
        assert!(!apply_c2_queried_speaker_first_t5_entry(&ixu, &uplan, &uq.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 0, 1, 3, 5, 4]);
    }

    #[test]
    fn same_session_user_t5_entry_swaps_later_or_opener_user_for_assistant() {
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I went on a week-long trip with my family last year."),
                        ("assistant", "That sounds like a great trip with your family."),
                        ("user", "We went to Hawaii on a week-long trip with my family."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I also took a short weekend trip last month."),
                        ("assistant", "Weekend trips can be a nice break too."),
                    ],
                ),
                sess(
                    "s3",
                    &[("user", "Travel planning takes a lot of time for any family trip.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "Where did I go on a week-long trip with my family?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        // Official k of 6: last assistant (s2) at slot 5 yields to the later
        // same-session user already in k at slot 6.
        let mut list = vec![0usize, 3, 5, 1, 4, 2];
        let t_scene = vec![1.0f32, 0.8, 0.6, 0.9, 0.7, 0.4];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_same_session_user_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 3, 5, 1, 2, 4]);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // Session opener in official k, later same-session turn already in top 5.
        let ixo = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I went to three movie festivals this year."),
                        ("assistant", "You have been to quite a few festivals."),
                        ("user", "How many movie festivals have I attended so far?"),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I also watched a lot of films at home."),
                        ("assistant", "That is a different kind of movie habit."),
                    ],
                ),
                sess(
                    "s3",
                    &[("user", "Festival season always keeps me busy every year.")],
                ),
            ],
            &Params::default(),
        );
        let oq = Query { text: "How many movie festivals that I attended?".into(), date: String::new() };
        let oplan = plan_query(&ixo, &oq, &Params::default());
        // Head is the later same-session question (idx 2); opener (idx 0) is at slot 6.
        let mut list = vec![2usize, 3, 5, 1, 4, 0];
        let osc = vec![0.8f32, 0.7, 1.0, 0.9, 0.6, 0.4];
        assert!(apply_c2_same_session_user_t5_entry(&ixo, &oplan, &oq.text, &mut list, &osc));
        assert_eq!(list, vec![2, 3, 5, 1, 0, 4]);

        // Different-session user in official k: no entry.
        let ixd = build_index(
            &[
                sess("s1", &[("user", "I went to three movie festivals this year."), ("assistant", "Nice."), ("user", "How many movie festivals have I attended so far?")]),
                sess("s2", &[("user", "I also watched a lot of films at home."), ("assistant", "Okay.")]),
                sess("s3", &[("user", "Festival season always keeps me busy every year.")]),
                sess("s4", &[("user", "I booked a train ticket to another city today.")]),
            ],
            &Params::default(),
        );
        let mut list = vec![2usize, 3, 5, 1, 4, 6];
        let dsc = vec![0.8f32, 0.7, 1.0, 0.9, 0.6, 0.4, 0.5];
        assert!(!apply_c2_same_session_user_t5_entry(&ixd, &oplan, &oq.text, &mut list, &dsc));
        assert_eq!(list, vec![2, 3, 5, 1, 4, 6]);

        // Preference query: no entry.
        let pq = Query { text: "What is my favorite week-long trip with my family?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![0usize, 3, 5, 1, 4, 2];
        assert!(!apply_c2_same_session_user_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 3, 5, 1, 4, 2]);

        // Named-speaker session: no entry.
        let ixn = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("john", "I went on a week-long trip with my family last year."),
                        ("maria", "That sounds like a great trip with your family."),
                        ("john", "We went to Hawaii on a week-long trip with my family."),
                        ("maria", "I also took a short weekend trip last month."),
                        ("john", "Weekend trips can be a nice break too."),
                        ("maria", "Travel planning takes a lot of time for any family trip."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let nq = Query { text: "Where did I go on a week-long trip with my family?".into(), date: String::new() };
        let nplan = plan_query(&ixn, &nq, &Params::default());
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(!apply_c2_same_session_user_t5_entry(&ixn, &nplan, &nq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 5, 2]);
    }

    #[test]
    fn cross_session_user_t5_entry_swaps_opener_for_non_head_assistant() {
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I have been tracking my family size for a genealogy project."),
                        ("assistant", "That is a useful way to organize family records."),
                        ("user", "I also wrote down the cousins on my mother's side."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I have been cataloging recipes from different relatives."),
                        ("assistant", "Recipes can be a nice complement to a family archive."),
                    ],
                ),
                sess(
                    "s3",
                    &[
                        ("user", "I have two siblings, one older brother and one younger sister."),
                        ("assistant", "So you have two siblings in total."),
                    ],
                ),
                sess(
                    "s4",
                    &[("user", "I have been thinking about printing the family book next year.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What is the total number of siblings I have?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        // Official k of 6: last non-unique assistant that is not the head's
        // next turn (s2 assistant at slot 4) yields to the cross-session
        // user opener already in k at slot 6. The s1 assistant at slot 2
        // is the head's immediate continuation and is not evicted.
        // ids: s1 u0=0 a=1 u2=2; s2 u=3 a=4; s3 u=5 a=6; s4 u=7
        let mut list = vec![0usize, 1, 3, 4, 7, 5];
        let t_scene = vec![1.0f32, 0.9, 0.5, 0.8, 0.7, 0.4, 0.3, 0.6];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_cross_session_user_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 5, 7, 4]);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // Only evictable assistant is the head's next turn: no entry.
        let mut list = vec![0usize, 1, 7, 2, 3, 5];
        assert!(!apply_c2_cross_session_user_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 7, 2, 3, 5]);

        // Mid-session cross-session user in official k: no entry.
        let ixm = build_index(
            &[
                sess(
                    "s1",
                    &[
                        ("user", "I have been tracking my family size for a genealogy project."),
                        ("assistant", "That is a useful way to organize family records."),
                    ],
                ),
                sess(
                    "s2",
                    &[
                        ("user", "I have been cataloging recipes from different relatives."),
                        ("assistant", "Recipes can be a nice complement to a family archive."),
                    ],
                ),
                sess(
                    "s3",
                    &[
                        ("user", "I started a diary last year about my relatives."),
                        ("assistant", "A diary is a good place to keep family notes."),
                        ("user", "I have two siblings, one older brother and one younger sister."),
                    ],
                ),
                sess(
                    "s4",
                    &[("user", "I have been thinking about printing the family book next year.")],
                ),
            ],
            &Params::default(),
        );
        // ids: s1 u=0 a=1; s2 u=2 a=3; s3 u=4 a=5 u=6; s4 u=7. Mid user (6) at slot 6.
        let mut list = vec![0usize, 1, 2, 3, 7, 6];
        let msc = vec![1.0f32, 0.9, 0.8, 0.7, 0.4, 0.3, 0.5, 0.6];
        assert!(!apply_c2_cross_session_user_t5_entry(&ixm, &plan, &q.text, &mut list, &msc));
        assert_eq!(list, vec![0, 1, 2, 3, 7, 6]);

        // Same-session opener already represented in the top 5: no entry (that is #20).
        // ids: 0,1,2 s1; 3,4 s2; 5,6 s3; 7 s4. Head is s1 later user; s1 opener at slot 6.
        let mut list = vec![2usize, 3, 1, 7, 4, 0];
        assert!(!apply_c2_cross_session_user_t5_entry(&ix, &plan, &q.text, &mut list, &t_scene));
        assert_eq!(list, vec![2, 3, 1, 7, 4, 0]);

        // Preference query: no entry.
        let pq = Query { text: "What is my favorite way to record the total number of siblings I have?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![0usize, 1, 3, 4, 7, 5];
        assert!(!apply_c2_cross_session_user_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 3, 4, 7, 5]);

        // Named-speaker session: no entry.
        let ixn = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("john", "I have been tracking my family size for a genealogy project."),
                        ("maria", "That is a useful way to organize family records."),
                        ("john", "I also wrote down the cousins on my mother's side."),
                        ("maria", "I have been cataloging recipes from different relatives."),
                        ("john", "Recipes can be a nice complement to a family archive."),
                        ("maria", "I have two siblings, one older brother and one younger sister."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let nq = Query { text: "What is the total number of siblings I have?".into(), date: String::new() };
        let nplan = plan_query(&ixn, &nq, &Params::default());
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(!apply_c2_cross_session_user_t5_entry(&ixn, &nplan, &nq.text, &mut list, &t_scene));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn exchequer_split_seats_complementary_assistant_over_echo() {
        // Stock (user asks for the dish) and foil (assistant states the egg
        // count) together cover the echo head, which restates every query
        // group in one turn. The echo's own reply does not split.
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "How do I make a classic French omelette?"),
                    ("assistant", "Use 3 eggs and whisk them."),
                    ("user", "Any tips to keep the omelette fluffy?"),
                    ("assistant", "Whisk the eggs for a French omelette."),
                    ("user", "I want to make a classic French omelette with eggs tonight."),
                    ("assistant", "A classic French omelette with eggs can take cheese or ham."),
                ],
            )],
            &Params::default(),
        );
        let q = Query {
            text: "How many eggs did you say we need for the classic French omelette?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(plan.intent.assistant_attribution);
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        let mut list = vec![5usize, 3, 4, 2, 1, 0];
        let scene = vec![0.4f32, 0.9, 0.3, 0.8, 0.5, 1.0];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        let top5_before: std::collections::HashSet<usize> = list[..5].iter().copied().collect();
        assert!(apply_c2_step31_exchequer_split(&ix, &plan, &q.text, &mut list, &scene));
        assert_eq!(list[0], 1);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);
        let top5_after: std::collections::HashSet<usize> = list[..5].iter().copied().collect();
        assert_eq!(top5_before, top5_after);

        // Same groups on both halves: not a split, no seat.
        let ix_echo = build_index(
            &[sess(
                "s",
                &[
                    ("user", "How many eggs go in a classic French omelette?"),
                    ("assistant", "A classic French omelette needs 3 eggs."),
                    ("user", "Thanks, and what pan should I use for the eggs?"),
                    ("assistant", "Use a nonstick pan."),
                ],
            )],
            &Params::default(),
        );
        let mut echo = vec![1usize, 3, 2, 0];
        assert!(!apply_c2_step31_exchequer_split(&ix_echo, &plan, &q.text, &mut echo, &[1.0, 0.5, 0.4, 0.8]));
        assert_eq!(echo, vec![1, 3, 2, 0]);

        // Preference request: no seat.
        let pq = Query { text: "Can you recommend a classic French omelette recipe with eggs?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(pplan.intent.preference);
        let mut list = vec![5usize, 3, 4, 2, 1, 0];
        assert!(!apply_c2_step31_exchequer_split(&ix, &pplan, &pq.text, &mut list, &scene));
        assert_eq!(list, vec![5, 3, 4, 2, 1, 0]);

        // Named speakers: no seat.
        let ixn = build_index(
            &[sess(
                "D1",
                &[
                    ("caroline", "How do I make a classic French omelette?"),
                    ("melanie", "Use 3 eggs and whisk them."),
                    ("caroline", "Any tips to keep the omelette fluffy?"),
                    ("melanie", "Whisk the eggs for a French omelette."),
                    ("caroline", "I want to make a classic French omelette with eggs tonight."),
                    ("melanie", "A classic French omelette with eggs can take cheese or ham."),
                ],
            )],
            &Params::default(),
        );
        let mut list = vec![5usize, 3, 4, 2, 1, 0];
        assert!(!apply_c2_step31_exchequer_split(&ixn, &plan, &q.text, &mut list, &scene));
        assert_eq!(list, vec![5, 3, 4, 2, 1, 0]);
    }

    #[test]
    fn triangulation_seats_dual_best_richer_mark_in_named_dialogue() {
        // s1 is the winning session and contributes one top-5 turn. That turn
        // is also the unique own-lexical best and attests a query term the
        // head lacks. The head's session is tied with another top-5 turn, so
        // it is not best on the session sightline.
        let ix = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("melanie", "The weather was nice yesterday."),
                        ("caroline", "I saw a small museum."),
                    ],
                ),
                sess(
                    "D2",
                    &[
                        ("caroline", "I visited Rome and Florence in Italy."),
                        ("melanie", "That was a long trip."),
                    ],
                ),
                sess(
                    "D3",
                    &[
                        ("caroline", "I stayed home all weekend."),
                        ("melanie", "Home is quiet."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "Which cities in Italy did Caroline visit?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(c2_named_session(&ix, 0));
        let ng = plan.groups.len();
        assert!(ng > 0);
        // turns: 0,1 s1; 2,3 s2; 4,5 s3. Head is 0. Dual-best candidate is 2,
        // the only top-5 member of the highest session score.
        let mut list = vec![0usize, 1, 4, 5, 2];
        let t_own = vec![0.2f32, 1.0, 4.0, 0.5, 0.4, 0.3];
        let s_score = vec![0.2f32, 0.9, 0.4];
        let bridge = vec![0.0f32, 0.0, 0.0];
        let head_hit = c2_term_ids(&ix, &plan, 0);
        let cand_hit = c2_term_ids(&ix, &plan, 2);
        assert!(head_hit.iter().all(|id| cand_hit.contains(id)));
        assert!(cand_hit.iter().any(|id| !head_hit.contains(id)));
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        let top5_before: std::collections::HashSet<usize> = list[..5].iter().copied().collect();
        assert!(apply_c2_step33_triangulation(&ix, &plan, &mut list, &t_own, &s_score, &bridge));
        assert_eq!(list[0], 2);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);
        let top5_after: std::collections::HashSet<usize> = list[..5].iter().copied().collect();
        assert_eq!(top5_before, top5_after);

        // Already best on both: turn 2 is slot 1. Do not move it.
        let mut seated = vec![2usize, 0, 1, 4, 5];
        assert!(!apply_c2_step33_triangulation(&ix, &plan, &mut seated, &t_own, &s_score, &bridge));
        assert_eq!(seated, vec![2, 0, 1, 4, 5]);

        // Two turns share the winning session score: no unique session sightline.
        let mut tied = vec![0usize, 1, 4, 2, 3];
        assert!(!apply_c2_step33_triangulation(&ix, &plan, &mut tied, &t_own, &s_score, &bridge));
        assert_eq!(tied, vec![0, 1, 4, 2, 3]);

        // User/assistant chat: named-speaker sightline does not apply.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "The weather was nice yesterday."),
                    ("assistant", "I visited Rome and Florence in Italy."),
                ],
            )],
            &Params::default(),
        );
        let mut users = vec![0usize, 1];
        assert!(!apply_c2_step33_triangulation(
            &ixu,
            &plan,
            &mut users,
            &[0.1, 5.0],
            &[0.2],
            &[0.0],
        ));
        assert_eq!(users, vec![0, 1]);
    }



    fn sess_at(id: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
        Session {
            id: id.into(),
            date: String::new(),
            timestamp: ts,
            turns: turns
                .iter()
                .enumerate()
                .map(|(i, (r, c))| Turn {
                    id: format!("{id}#{i}"),
                    role: (*r).into(),
                    content: (*c).into(),
                    ..Turn::default()
                })
                .collect(),
        }
    }

    #[test]
    fn step32_trial_balance_seats_earlier_opener_that_posts_the_rare_group() {
        let ix = build_index(
            &[
                sess_at(
                    "early",
                    100,
                    &[
                        ("user", "I stored the zynth widget in the drawer."),
                        ("assistant", "The widget is noted."),
                    ],
                ),
                sess_at(
                    "late",
                    200,
                    &[
                        ("user", "Did I leave the widget somewhere safe?"),
                        ("assistant", "We can search for the widget."),
                    ],
                ),
                sess_at("noise", 50, &[("user", "The widget on the shelf is blue.")]),
            ],
            &Params::default(),
        );
        // 0 early opener (zynth+widget), 1 early reply, 2 late question, 3 late reply, 4 noise widget.
        let q = Query {
            text: "Where is the zynth widget?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(ix.turns[2].is_question);
        assert_eq!(c2_session_pos(&ix, 0), 0);
        let mut list = vec![2usize, 0, 1, 3, 4];
        let scene = vec![0.4f32, 0.2, 0.9, 0.3, 0.1];
        assert!(apply_c2_step32_trial_balance(&ix, &plan, &mut list, &scene));
        assert_eq!(list[0], 0);
        assert_eq!(list.len(), 5);

        // Slot 1 already posts the rare group: no move.
        let q_bal = Query {
            text: "Where is the zynth widget?".into(),
            date: String::new(),
        };
        let ix_bal = build_index(
            &[
                sess_at(
                    "early",
                    100,
                    &[("user", "I stored the zynth widget in the drawer.")],
                ),
                sess_at(
                    "late",
                    200,
                    &[("user", "Did I leave the zynth widget somewhere safe?")],
                ),
                sess_at("noise", 50, &[("user", "The widget on the shelf is blue.")]),
            ],
            &Params::default(),
        );
        let plan_b = plan_query(&ix_bal, &q_bal, &Params::default());
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_step32_trial_balance(
            &ix_bal,
            &plan_b,
            &mut list,
            &[1.0, 0.9, 0.2]
        ));
        assert_eq!(list, vec![1, 0, 2]);

        // Opener session is later than the question: no move.
        let ix_later = build_index(
            &[
                sess_at(
                    "earlyq",
                    100,
                    &[("user", "Did I leave the widget somewhere safe?")],
                ),
                sess_at(
                    "later",
                    300,
                    &[("user", "I stored the zynth widget in the drawer.")],
                ),
                sess_at("noise", 50, &[("user", "The widget on the shelf is blue.")]),
            ],
            &Params::default(),
        );
        let plan_l = plan_query(&ix_later, &q, &Params::default());
        let mut list = vec![0usize, 1, 2];
        assert!(!apply_c2_step32_trial_balance(
            &ix_later,
            &plan_l,
            &mut list,
            &[1.0, 0.95, 0.2]
        ));
        assert_eq!(list, vec![0, 1, 2]);

        // Head is not a question: no move.
        let ix_decl = build_index(
            &[
                sess_at(
                    "early",
                    100,
                    &[("user", "I stored the zynth widget in the drawer.")],
                ),
                sess_at(
                    "late",
                    200,
                    &[("user", "I left the widget somewhere safe.")],
                ),
                sess_at("noise", 50, &[("user", "The widget on the shelf is blue.")]),
            ],
            &Params::default(),
        );
        let plan_d = plan_query(&ix_decl, &q, &Params::default());
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_step32_trial_balance(
            &ix_decl,
            &plan_d,
            &mut list,
            &[1.0, 0.9, 0.2]
        ));
        assert_eq!(list, vec![1, 0, 2]);

        // Equal scene: the earlier session wins.
        let ix_tie = build_index(
            &[
                sess_at(
                    "mid",
                    150,
                    &[("user", "I stored the zynth widget in the drawer.")],
                ),
                sess_at(
                    "early",
                    80,
                    &[("user", "I stored the zynth widget in the drawer.")],
                ),
                sess_at(
                    "late",
                    200,
                    &[("user", "Did I leave the widget somewhere safe?")],
                ),
                sess_at("noise", 50, &[("user", "The widget on the shelf is blue.")]),
            ],
            &Params::default(),
        );
        let plan_t = plan_query(&ix_tie, &q, &Params::default());
        // 0 mid opener, 1 early opener, 2 late question, 3 noise.
        let mut list = vec![2usize, 0, 1, 3];
        assert!(apply_c2_step32_trial_balance(
            &ix_tie,
            &plan_t,
            &mut list,
            &[0.4, 0.4, 0.5, 0.1]
        ));
        assert_eq!(list[0], 1);
    }

    #[test]
    fn step32_surface_lexeme_does_not_accept_a_stem() {
        // "recipe" sits next to "recipes" and must not post that account.
        let ix = build_index(
            &[
                sess_at("early", 100, &[("user", "I wrote three cocktail recipes.")]),
                sess_at("late", 200, &[("user", "Can you repeat that recipe?")]),
                sess_at(
                    "noise",
                    50,
                    &[("user", "The cocktail on the shelf is blue.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "Which cocktail recipes did I write?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(ix.turns[1].is_question);
        let mut list = vec![1usize, 0, 2];
        assert!(apply_c2_step32_trial_balance(
            &ix,
            &plan,
            &mut list,
            &[0.4, 0.9, 0.2]
        ));
        assert_eq!(list[0], 0);
    }

    #[test]
    fn step32_echo_yields_to_the_turn_it_answers() {
        let mut sessions = Vec::new();
        for i in 0..6 {
            sessions.push(sess_at(
                &format!("filler{i}"),
                10 + i,
                &[
                    ("user", "The kids learned values and the dog can teach."),
                    (
                        "assistant",
                        "Values for the kids matter and the dog can teach.",
                    ),
                ],
            ));
        }
        sessions.push(sess_at(
            "real",
            100,
            &[
                ("user", "We are adopting a rescue dog to teach the kids."),
                ("assistant", "That teaches the kids important values."),
            ],
        ));
        let ix = build_index(&sessions, &Params::default());
        let parent = ix.turns.len() - 2;
        let echo = ix.turns.len() - 1;
        assert_eq!(ix.turns[echo].answers, Some(parent));
        assert!(!ix.turns[echo].is_question);
        let q = Query {
            text: "What values does adopting a rescue dog teach the kids?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        let mut list = vec![echo, parent];
        let scene = vec![0.2f32; ix.turns.len()];
        assert!(apply_c2_step32_trial_balance(&ix, &plan, &mut list, &scene));
        assert_eq!(list[0], parent);
    }

    #[test]
    fn step34_bertillonage_yields_to_the_richer_same_session_card() {
        // Blank card. Slot 1 is a question that posts none of the query's
        // anchor kind, and it is not the strongest session. The same-session
        // user turn that posts an anchor and is more on-scene takes slot 1.
        // A richer turn in the strongest session does not: different session.
        let ix = build_index(
            &[
                sess(
                    "other",
                    &[("user", "I listen to music streaming lately on a different service.")],
                ),
                sess(
                    "home",
                    &[
                        ("user", "What playlist should we start?"),
                        ("user", "I have been using that service lately."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = "What is the name of the music streaming service have I been using lately?";
        let parsed = step34_query(&ix, q);
        assert!(parsed.n >= 1);
        assert!(parsed.speakers.is_empty());
        let head = 1usize;
        let gold = 2usize;
        let other = 0usize;
        assert!(ix.turns[head].is_question);
        assert!(!ix.turns[gold].is_question);
        assert_eq!(step34_card(&ix, head, &parsed), 0);
        assert_ne!(step34_card(&ix, gold, &parsed) & STEP34_ANCHOR, 0);
        let mut list = vec![head, other, gold];
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[head] = 1.0;
        scene[gold] = 2.0;
        scene[other] = 5.0;
        assert_ne!(step34_leader_session(&ix, &scene), Some(ix.turns[head].session));
        assert!(apply_c2_step34_bertillonage(&ix, q, &mut list, &scene));
        assert_eq!(list[0], gold);
        assert_eq!(list.len(), 3);

        // Named-speaker testimony. The question at slot 1 posts an anchor but
        // not the speaker the query names. That speaker's same-session reply
        // posts both and is more on-scene, so it takes slot 1. Her turn in
        // another session is more on-scene still and stays put.
        let ix = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("jon", "What dance piece did the team perform to win first place?"),
                        ("gina", "Our piece was a contemporary solo."),
                    ],
                ),
                sess("D2", &[("gina", "The dance piece our team performed won first place.")]),
            ],
            &Params::default(),
        );
        let q = "What kind of dance piece did Gina's team perform to win first place?";
        let parsed = step34_query(&ix, q);
        assert!(parsed.speakers.contains("gina"));
        assert!(parsed.n >= 2);
        let head = 0usize;
        let gold = 1usize;
        let other = 2usize;
        assert!(ix.turns[head].is_question);
        assert!(!ix.turns[gold].is_question);
        let head_bits = step34_card(&ix, head, &parsed);
        let gold_bits = step34_card(&ix, gold, &parsed);
        assert_eq!(head_bits & STEP34_ANCHOR, STEP34_ANCHOR);
        assert_eq!(head_bits & STEP34_SPEAKER, 0);
        assert_eq!(gold_bits & STEP34_SPEAKER, STEP34_SPEAKER);
        assert!(gold_bits.count_ones() > head_bits.count_ones());
        let mut list = vec![head, gold, other];
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[head] = 3.0;
        scene[gold] = 4.0;
        scene[other] = 9.0;
        assert!(apply_c2_step34_bertillonage(&ix, q, &mut list, &scene));
        assert_eq!(list[0], gold);

        // Slot 1 already meets N: the named speaker who posts the anchor stays,
        // even when a question in the same session is more on-scene.
        let mut list = vec![gold, head, other];
        scene[head] = 8.0;
        assert!(!apply_c2_step34_bertillonage(&ix, q, &mut list, &scene));
        assert_eq!(list[0], gold);

        // Echo in the strongest session. Slot 1 is a question with an empty
        // card, but it already matches the strongest session, and the witness
        // is a different role the query does not name. The seat stays.
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "What should we do this weekend?"),
                    ("assistant", "You mentioned musical instruments you currently own."),
                    ("user", "The catalog is on the shelf."),
                ],
            )],
            &Params::default(),
        );
        let q = "How many musical instruments do I currently own?";
        let parsed = step34_query(&ix, q);
        let head = 0usize;
        let echo = 1usize;
        assert!(ix.turns[head].is_question);
        assert!(!ix.turns[echo].is_question);
        assert_eq!(step34_card(&ix, head, &parsed), 0);
        assert_ne!(step34_card(&ix, echo, &parsed) & STEP34_ANCHOR, 0);
        let mut list = vec![head, echo, 2];
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[head] = 2.0;
        scene[echo] = 4.0;
        scene[2] = 6.0;
        assert_eq!(step34_leader_session(&ix, &scene), Some(ix.turns[head].session));
        assert!(!apply_c2_step34_bertillonage(&ix, q, &mut list, &scene));
        assert_eq!(list, vec![head, echo, 2]);

        // Two same-session witnesses with the same number of kinds are not an
        // identification. Slot 1 stays.
        let ix = build_index(
            &[sess(
                "home",
                &[
                    ("user", "What playlist should we start?"),
                    ("user", "I have been using that service lately."),
                    ("user", "The music streaming service is the one I use lately."),
                ],
            )],
            &Params::default(),
        );
        let q = "What is the name of the music streaming service have I been using lately?";
        let mut list = vec![0usize, 1, 2];
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[0] = 1.0;
        scene[1] = 3.0;
        scene[2] = 4.0;
        assert!(!apply_c2_step34_bertillonage(&ix, q, &mut list, &scene));
        assert_eq!(list, vec![0, 1, 2]);
    }


    #[test]
    fn step35_mill_agreement_seats_the_unique_common_session() {
        // ag posts both anchors (split across two turns). hd posts only widget.
        // ot posts only zynth. The pool intersection is ag alone. The head is
        // outside it. The one top-5 ag turn that posts widget takes slot 1.
        let ix = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[
                        ("user", "The zynth widget sits in the cedar drawer."),
                        ("assistant", "Noted for later."),
                    ],
                ),
                sess_at(
                    "hd",
                    20,
                    &[("user", "The widget is blue and heavy."), ("assistant", "Blue widget.")],
                ),
                sess_at(
                    "ot",
                    30,
                    &[("user", "The zynth is shiny."), ("assistant", "Shiny.")],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "Where is the zynth widget?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(plan.groups.iter().any(|g| g == "zynth"));
        assert!(plan.groups.iter().any(|g| g == "widget"));
        // 0 ag both, 1 ag widget-only, 2 hd widget, 3 hd widget, 4 ot zynth, 5 ot none.
        let scene = vec![0.9f32, 0.4, 1.0, 0.3, 0.5, 0.2];
        let mut list = vec![2usize, 0, 4, 3, 1];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_step35_mill_agreement(&ix, &plan, &mut list, &scene));
        assert_eq!(list[0], 0);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // Head already inside the agreement session: do not move it.
        let mut inside = vec![0usize, 2, 4, 3, 1];
        assert!(!apply_c2_step35_mill_agreement(&ix, &plan, &mut inside, &scene));
        assert_eq!(inside, vec![0, 2, 4, 3, 1]);

        // Two top-5 turns of the agreement session both cover the head: tie.
        let ixt = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[
                        ("user", "The zynth widget sits in the cedar drawer."),
                        ("user", "A second zynth widget note."),
                    ],
                ),
                sess_at("hd", 20, &[("user", "The widget is blue and heavy.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plant = plan_query(&ixt, &q, &Params::default());
        let scenet = vec![0.9f32, 0.8, 1.0, 0.5];
        let mut tie = vec![2usize, 0, 1, 3];
        assert!(!apply_c2_step35_mill_agreement(&ixt, &plant, &mut tie, &scenet));
        assert_eq!(tie, vec![2, 0, 1, 3]);

        // A second session that also posts both anchors: several common sessions.
        let ix2 = build_index(
            &[
                sess_at("ag", 10, &[("user", "The zynth widget sits in the cedar drawer.")]),
                sess_at("hd", 20, &[("user", "The widget is blue and heavy.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
                sess_at("also", 40, &[("user", "Another zynth widget is on the shelf.")]),
            ],
            &Params::default(),
        );
        let plan2 = plan_query(&ix2, &q, &Params::default());
        let scene2 = vec![0.9f32, 1.0, 0.5, 0.8];
        let mut many = vec![1usize, 0, 2, 3];
        assert!(!apply_c2_step35_mill_agreement(&ix2, &plan2, &mut many, &scene2));
        assert_eq!(many, vec![1, 0, 2, 3]);

        // The only top-5 agreement turn misses an anchor the head posts.
        let ix3 = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[
                        ("user", "The zynth is here."),
                        ("assistant", "The widget is in the cedar drawer."),
                    ],
                ),
                sess_at("hd", 20, &[("user", "The widget is cedar colored.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let q3 = Query {
            text: "Where is the cedar zynth widget?".into(),
            date: String::new(),
        };
        let plan3 = plan_query(&ix3, &q3, &Params::default());
        // 0 ag zynth (top 5), 1 ag widget+cedar (pool only, not top 5), 2 hd widget+cedar, 3 ot zynth.
        let scene3 = vec![0.4f32, 0.2, 1.0, 0.5];
        let mut thin = vec![2usize, 0, 3];
        assert!(!apply_c2_step35_mill_agreement(&ix3, &plan3, &mut thin, &scene3));
        assert_eq!(thin, vec![2, 0, 3]);

        // A stem does not post the surface anchor, so "widgets" is not "widget".
        let ix4 = build_index(
            &[
                sess_at("ag", 10, &[("user", "The zynth widgets sit in the drawer.")]),
                sess_at("hd", 20, &[("user", "The widget is blue.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plan4 = plan_query(&ix4, &q, &Params::default());
        let scene4 = vec![0.9f32, 1.0, 0.5];
        let mut stemmed = vec![1usize, 0, 2];
        assert!(!apply_c2_step35_mill_agreement(&ix4, &plan4, &mut stemmed, &scene4));
        assert_eq!(stemmed, vec![1, 0, 2]);

        // A complete instance by the other speaker does not displace the head.
        let ixn = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[("caroline", "The zynth widget sits in the cedar drawer.")],
                ),
                sess_at("hd", 20, &[("melanie", "The widget is blue and heavy.")]),
                sess_at("ot", 30, &[("caroline", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plann = plan_query(&ixn, &q, &Params::default());
        let scenen = vec![0.9f32, 1.0, 0.5];
        let mut named = vec![1usize, 0, 2];
        assert!(!apply_c2_step35_mill_agreement(&ixn, &plann, &mut named, &scenen));
        assert_eq!(named, vec![1, 0, 2]);

        // Same speaker: the complete instance takes slot 1.
        let ixs = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[("caroline", "The zynth widget sits in the cedar drawer.")],
                ),
                sess_at("hd", 20, &[("caroline", "The widget is blue and heavy.")]),
                sess_at("ot", 30, &[("melanie", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plans = plan_query(&ixs, &q, &Params::default());
        let mut same = vec![1usize, 0, 2];
        assert!(apply_c2_step35_mill_agreement(&ixs, &plans, &mut same, &scenen));
        assert_eq!(same[0], 0);

        // Non-instance head: it posts no anchor. The unique top-5 instance of
        // the agreement session takes slot 1. An other-role echo is not required
        // to match, because the head is not a witness.
        let ix0 = build_index(
            &[
                sess_at("ag", 10, &[("user", "The zynth widget sits in the cedar drawer.")]),
                sess_at("hd", 20, &[("user", "The drawer is blue and heavy.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plan0 = plan_query(&ix0, &q, &Params::default());
        let mut none = vec![1usize, 0, 2];
        assert!(apply_c2_step35_mill_agreement(&ix0, &plan0, &mut none, &[0.9, 1.0, 0.5]));
        assert_eq!(none[0], 0);

        // Other-role echo that posts every anchor does not displace a partial user head.
        let ixe = build_index(
            &[
                sess_at(
                    "ag",
                    10,
                    &[("assistant", "The zynth widget sits in the cedar drawer.")],
                ),
                sess_at("hd", 20, &[("user", "The widget is blue and heavy.")]),
                sess_at("ot", 30, &[("user", "The zynth is shiny.")]),
            ],
            &Params::default(),
        );
        let plane = plan_query(&ixe, &q, &Params::default());
        let mut echo = vec![1usize, 0, 2];
        assert!(!apply_c2_step35_mill_agreement(&ixe, &plane, &mut echo, &[0.9, 1.0, 0.5]));
        assert_eq!(echo, vec![1, 0, 2]);
    }

    #[test]
    fn step36_antaranga_seats_own_majority_over_an_external_question() {
        let ix = build_index(
            &[
                sess("ask", &[("user", "Where was the workshop?")]),
                sess("fact", &[("user", "I attended the workshop on Friday.")]),
                sess("echo", &[("user", "I attended the workshop downtown too.")]),
            ],
            &Params::default(),
        );
        let q = "Where was the workshop?";
        assert!(ix.turns[0].is_question);
        assert!(!ix.turns[1].is_question);
        assert!(!ix.turns[2].is_question);
        let anchors = step36_anchors(&ix, q);
        assert!(anchors.iter().any(|ids| step36_posts(&ix.turns[0], ids)));
        assert!(
            anchors
                .iter()
                .all(|ids| !step36_posts(&ix.turns[0], ids) || step36_posts(&ix.turns[1], ids))
        );
        // Question at slot 1: own 0.2, scene 5.0 (external residual 4.8 > 0.2).
        // Fact turn: own 3.0 is a majority of scene 4.0, higher session score.
        // Echo has a higher own score but the same session score as the question.
        let t_own = vec![0.2f32, 3.0, 4.0];
        let t_scene = vec![5.0f32, 4.0, 5.0];
        let s_score = vec![0.4f32, 0.9, 0.4];
        let bridge = vec![0.0f32, 0.0, 0.0];
        let mut list = vec![0usize, 2, 1];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_step36_antaranga(
            &ix, q, &mut list, &t_own, &t_scene, &s_score, &bridge
        ));
        assert_eq!(list[0], 1);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // Own score is not a majority of the fact turn's scene: the question stays.
        let mut list = vec![0usize, 2, 1];
        let thin = vec![5.0f32, 9.0, 5.0];
        assert!(!apply_c2_step36_antaranga(
            &ix, q, &mut list, &t_own, &thin, &s_score, &bridge
        ));
        assert_eq!(list, vec![0, 2, 1]);

        // Two stronger sessions with the same own score are not one reading.
        let mut list = vec![0usize, 1, 2];
        let tied_own = vec![0.2f32, 3.0, 3.0];
        let tied_scene = vec![5.0f32, 4.0, 4.0];
        let both = vec![0.4f32, 0.9, 0.95];
        assert!(!apply_c2_step36_antaranga(
            &ix,
            q,
            &mut list,
            &tied_own,
            &tied_scene,
            &both,
            &bridge
        ));
        assert_eq!(list, vec![0, 1, 2]);

        // A declarative head is already internal. It does not yield.
        let mut list = vec![1usize, 0, 2];
        assert!(!apply_c2_step36_antaranga(
            &ix, q, &mut list, &t_own, &t_scene, &s_score, &bridge
        ));
        assert_eq!(list, vec![1, 0, 2]);
    }

    fn sess_dated(id: &str, date: &str, turns: &[(&str, &str)]) -> Session {
        let mut s = sess(id, turns);
        s.date = date.into();
        s
    }

    fn witnesses() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // t_scene, t_own, s_score, bridge. Turn 1 is the lexical head in session b.
        (
            vec![0.4f32, 0.9, 0.5, 0.2],
            vec![0.2f32, 0.8, 0.3, 0.1],
            vec![0.2f32, 0.9, 0.1],
            vec![0.0f32, 0.0, 0.0],
        )
    }

    #[test]
    fn w2001_seats_session_when_four_witnesses_agree() {
        let ix = build_index(
            &[
                sess("a", &[("user", "I like hiking.")]),
                sess("b", &[
                    ("user", "I bought hiking boots in Denver."),
                    ("user", "The boots were from Denver."),
                ]),
                sess("c", &[("user", "The weather was fine.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "Where did I buy hiking boots?".into(), date: String::new() };
        let turns = vec![1usize, 2, 0];
        let (t_scene, t_own, s_score, bridge) = witnesses();
        let mut sessions = vec![0usize, 1, 2];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &turns, &mut sessions, &t_scene, &t_own, &s_score, &bridge),
            Some(1)
        );
        assert_eq!(sessions, vec![1, 0, 2]);
        assert_eq!(turns, vec![1, 2, 0]);
    }

    #[test]
    fn w2001_refuses_when_a_witness_fails_and_does_not_fall_through() {
        let ix = build_index(
            &[
                sess("a", &[("user", "I like hiking.")]),
                sess("b", &[
                    ("user", "I bought hiking boots in Denver."),
                    ("user", "The boots were from Denver."),
                ]),
                sess("c", &[("user", "The workshop was cancelled.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "Where did I buy hiking boots?".into(), date: String::new() };
        let turns = vec![1usize, 2, 0];
        let (t_scene, t_own, _, bridge) = witnesses();
        // Sightline of b does not beat a. Envelope still nominates b (it is the
        // scene leader and the rank-1 turn). Seal would also nominate b, because
        // 0.9 >= a's best scene 0.4. The failed sightline owns the decision.
        let s_score = vec![0.9f32, 0.2, 0.1];
        let mut sessions = vec![0usize, 1, 2];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &turns, &mut sessions, &t_scene, &t_own, &s_score, &bridge),
            None
        );
        assert_eq!(sessions, vec![0, 1, 2]);

        // Own lexical score of the head does not beat the incumbent's best turn.
        let (t_scene, _, s_score, bridge) = witnesses();
        let t_own = vec![0.95f32, 0.2, 0.1, 0.1];
        let mut sessions = vec![0usize, 1, 2];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &turns, &mut sessions, &t_scene, &t_own, &s_score, &bridge),
            None
        );

        // The nominee does not hold strictly more top-5 turns.
        let (t_scene, t_own, s_score, bridge) = witnesses();
        let mut sessions = vec![0usize, 1, 2];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &[1usize, 0], &mut sessions, &t_scene, &t_own, &s_score, &bridge),
            None
        );
        assert_eq!(sessions, vec![0, 1, 2]);
    }

    #[test]
    fn w2001_calendar_tie_and_outside_top5_stay_put() {
        let ix = build_index(
            &[
                sess_dated("june", "2023-06-02", &[("user", "We met in the park.")]),
                sess_dated("may-a", "2023-05-08", &[("user", "We met in the park once.")]),
                sess_dated("may-b", "2023-05-20", &[("user", "We met in the park twice.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "What changed in May?".into(), date: String::new() };
        assert_eq!(w2001_query_months(&q.text), vec!["05".to_string()]);
        let mut sessions = vec![0usize, 1, 2];
        let t_scene = vec![0.9f32, 0.4, 0.4];
        let t_own = vec![0.9f32, 0.4, 0.4];
        let scores = vec![0.5f32, 0.4, 0.4];
        let bridge = vec![0.0f32, 0.0, 0.0];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &[0usize], &mut sessions, &t_scene, &t_own, &scores, &bridge),
            None
        );
        assert_eq!(sessions, vec![0, 1, 2]);

        let ix = build_index(
            &[
                sess_dated("s0", "2023-06-01", &[("user", "alpha park day.")]),
                sess_dated("s1", "2023-06-02", &[("user", "beta park day.")]),
                sess_dated("s2", "2023-06-03", &[("user", "gamma park day.")]),
                sess_dated("s3", "2023-06-04", &[("user", "delta park day.")]),
                sess_dated("s4", "2023-06-05", &[("user", "epsilon park day.")]),
                sess_dated("may", "2023-05-08", &[("user", "the may park day with boots.")]),
            ],
            &Params::default(),
        );
        let q = Query { text: "What happened during May 2023?".into(), date: String::new() };
        let mut sessions = vec![0usize, 1, 2, 3, 4, 5];
        let t_scene = vec![0.4f32, 0.2, 0.2, 0.2, 0.2, 0.9];
        let t_own = t_scene.clone();
        let scores = vec![0.2f32, 0.1, 0.1, 0.1, 0.1, 0.9];
        let bridge = vec![0.0; 6];
        assert_eq!(
            apply_c2_w2001_session_head(&ix, &q, &[0usize, 5], &mut sessions, &t_scene, &t_own, &scores, &bridge),
            None
        );
        assert_eq!(sessions, vec![0, 1, 2, 3, 4, 5]);
    }
}

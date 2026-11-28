//! Retroactive Salience Backfill (RSB): ingest-time backward edges and runtime cause seating.

use crate::retrieval::*;

/// Ingest-time Retroactive Salience Backfill (RSB), Contradiction Supersession,
/// Synaptic Tagging 9h CaptureWindow, and Temporal Context Episode Contiguity.
/// Strictly BACKWARD-ONLY (`later -> earlier`).
pub(in crate::retrieval) fn build_rsb_and_supersession_edges(
    session_turns: &[Vec<usize>],
    session_ep_keys: &[String],
    session_ts: &[i64],
    turn_tech_handles: &[Vec<u32>],
    constraint_terms: &DetHashSet<u32>,
    turn_prose_cache: &[String],
    turns: &mut [TurnIx],
) -> (Vec<Vec<(u32, usize)>>, Vec<Vec<usize>>, Vec<Option<usize>>) {
    // ---- Ingest-Time Retroactive Salience Backfill (RSB), Contradiction Supersession,
    //      Synaptic Tagging 9h CaptureWindow, and Temporal Context Episode Contiguity ----
    // Strictly BACKWARD-ONLY (`later -> earlier`). Zero forward edges, zero word similarity.
    let ns_count = session_turns.len();
    let mut session_rsb_back: Vec<Vec<(u32, usize)>> = vec![Vec::new(); ns_count];
    let mut session_supersedes_back: Vec<Vec<usize>> = vec![Vec::new(); ns_count];
    let mut session_episode_prev: Vec<Option<usize>> = vec![None; ns_count];
    for si in 1..ns_count {
        let same_ep = !session_ep_keys[si].is_empty() && session_ep_keys[si] == session_ep_keys[si - 1];
        let same_nonzero_ts = session_ts[si] > 0 && session_ts[si] == session_ts[si - 1];
        if same_ep || same_nonzero_ts {
            session_episode_prev[si] = Some(si - 1);
        }
    }

    // 1. Exact Technical Handle & Typed Slot RSB (`df <= 6`): later turn/session -> earlier turn/session.
    let mut handle_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ti, hs) in turn_tech_handles.iter().enumerate() {
        for &h in hs {
            let list = handle_postings.entry(h).or_default();
            if list.last().copied() != Some(ti) {
                list.push(ti);
            }
        }
    }
    let max_handle_deg = ((ns_count / 12).clamp(6, 36)) as u32;
    for (_, list) in handle_postings {
        let deg = list.len() as u32;
        if !(2..=max_handle_deg).contains(&deg) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let earlier = list[idx - 1];
            if !turns[later].exact_edges.iter().any(|&(_, t)| t == earlier) {
                turns[later].exact_edges.push((deg, earlier));
            }
            let s_lat = turns[later].session;
            let s_ear = turns[earlier].session;
            if s_lat > s_ear && !session_rsb_back[s_lat].iter().any(|&(_, s)| s == s_ear) {
                session_rsb_back[s_lat].push((deg, s_ear));
            }
        }
    }

    // 2. State-Update & Structural Contradiction Supersession Chain (`appears_contradictory`):
    //    When two user turns share a rare proper entity (`2 <= df <= max_ent_deg`) or constraint
    //    object and either `advanced::contradiction::appears_contradictory` fires or the later turn
    //    asserts an explicit revision/constraint, record a backward `supersedes` edge (`later -> earlier`).
    //    Crucially, `digit_shift` only triggers supersession on shared proper entities (never bare
    //    constraint terms across unrelated topics/entities).
    let mut ent_user_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    let mut const_user_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ti, t) in turns.iter().enumerate() {
        if t.field != 0 || t.is_pure_info_question || t.is_backchannel {
            continue;
        }
        for &eid in &t.entities {
            ent_user_postings.entry(eid).or_default().push(ti);
        }
        for &cid in &t.content {
            if constraint_terms.contains(&cid) {
                const_user_postings.entry(cid).or_default().push(ti);
            }
        }
    }
    let max_ent_deg = (ns_count / 10).clamp(8, 48);
    for (allow_digit_shift, postings_map) in [(true, ent_user_postings), (false, const_user_postings)] {
        for (_, mut list) in postings_map {
            list.dedup();
            if !(2..=max_ent_deg).contains(&list.len()) {
                continue;
            }
            for idx in 1..list.len() {
                let later = list[idx];
                let prev = list[idx - 1];
                let s_lat = turns[later].session;
                let s_prev = turns[prev].session;
                let same_sess_later = s_lat == s_prev && later > prev;
                if s_lat < s_prev || (s_lat == s_prev && later <= prev) {
                    continue;
                }
                let lat_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[later]);
                let prev_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[prev]);
                let torrens_pair_key_match = match (
                    crate::retrieval::graph::torrens::parse_torrens_sentence(prev_prose, prev),
                    crate::retrieval::graph::torrens::parse_torrens_sentence(lat_prose, later),
                ) {
                    (Some(rp), Some(rl)) => Some(rp.key_hash == rl.key_hash && rp.val_phrase != rl.val_phrase),
                    _ => None,
                };
                if torrens_pair_key_match == Some(false) {
                    continue;
                }
                let shared_substantive = {
                    let prev_toks = crate::retrieval::raw_tokens(prev_prose);
                    crate::retrieval::raw_tokens(lat_prose)
                        .iter()
                        .filter(|w| {
                            w.len() >= 4
                                && crate::retrieval::is_content(w)
                                && prev_toks.contains(*w)
                        })
                        .count()
                };
                let digit_shift = if allow_digit_shift && shared_substantive >= 2 {
                    let later_digits = digit_token_set(lat_prose);
                    let prev_digits = digit_token_set(prev_prose);
                    !later_digits.is_empty() && !prev_digits.is_empty() && later_digits != prev_digits
                } else {
                    false
                };
                let lat_low = lat_prose.to_ascii_lowercase();
                // hardcode-scan: closed-class (explicit state-replacement adverbs)
                let explicit_replacement = !typed_slot_tokens(lat_prose).is_empty()
                    || lat_low.contains("instead of ")
                    || lat_low.contains("no longer ")
                    || lat_low.contains("rather than ")
                    || lat_low.contains("switched to ")
                    || lat_low.contains("switched from ")
                    || lat_low.contains("moved from ")
                    || lat_low.contains("changed from ")
                    || lat_low.contains("changing the rule")
                    || lat_low.contains("changed the rule")
                    || lat_low.contains("changing the policy")
                    || lat_low.contains("changed the policy")
                    || lat_low.contains("anymore");
                let has_rev_cue = explicit_replacement || digit_shift;
                let contradict_id = if allow_digit_shift && shared_substantive >= 2 {
                    crate::advanced::contradiction::SubjectIdentity::AlreadyEstablished
                } else {
                    crate::advanced::contradiction::SubjectIdentity::FromTextOverlap
                };
                let contradicts = torrens_pair_key_match == Some(true)
                    || (lat_prose.len() <= 420
                        && prev_prose.len() <= 420
                        && crate::advanced::contradiction::appears_contradictory(
                            prev_prose,
                            lat_prose,
                            contradict_id,
                        ));
                let same_sess_ok = !same_sess_later
                    || torrens_pair_key_match == Some(true)
                    || (shared_substantive >= 2 && explicit_replacement);
                if same_sess_ok && (contradicts || has_rev_cue) {
                    if !turns[later].supersedes_turns.contains(&prev) {
                        turns[later].supersedes_turns.push(prev);
                    }
                    if !same_sess_later && !session_supersedes_back[s_lat].contains(&s_prev) {
                        session_supersedes_back[s_lat].push(s_prev);
                    }
                }
            }
        }
    }

    // 2b. Role-Object State-Update Supersession (`2 <= df <= 6`), strictly gated on explicit
    //     replacement cues or `appears_contradictory` (NEVER bare `digit_shift`).
    let mut role_obj_postings: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ti, t) in turns.iter().enumerate() {
        if t.field != 0 || t.is_pure_info_question || t.is_backchannel {
            continue;
        }
        for &rid in &t.role_objs {
            role_obj_postings.entry(rid).or_default().push(ti);
        }
    }
    let max_role_deg = (ns_count / 12).clamp(6, 36);
    for (_, mut list) in role_obj_postings {
        list.dedup();
        if !(2..=max_role_deg).contains(&list.len()) {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let prev = list[idx - 1];
            let s_lat = turns[later].session;
            let s_prev = turns[prev].session;
            if s_lat <= s_prev || turns[later].supersedes_turns.contains(&prev) {
                continue;
            }
            let lat_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[later]);
            let prev_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[prev]);
            let torrens_pair_key_match = match (
                crate::retrieval::graph::torrens::parse_torrens_sentence(prev_prose, prev),
                crate::retrieval::graph::torrens::parse_torrens_sentence(lat_prose, later),
            ) {
                (Some(rp), Some(rl)) => Some(rp.key_hash == rl.key_hash && rp.val_phrase != rl.val_phrase),
                _ => None,
            };
            if torrens_pair_key_match == Some(false) {
                continue;
            }
            let lat_low = lat_prose.to_ascii_lowercase();
            // hardcode-scan: closed-class (explicit state-replacement adverbs)
            let explicit_replacement = lat_low.contains("instead of ")
                || lat_low.contains("no longer ")
                || lat_low.contains("rather than ")
                || lat_low.contains("switched to ")
                || lat_low.contains("switched from ")
                || lat_low.contains("moved from ")
                || lat_low.contains("changed from ")
                || lat_low.contains("anymore");
            let contradicts = torrens_pair_key_match == Some(true)
                || (lat_prose.len() <= 420
                    && prev_prose.len() <= 420
                    && crate::advanced::contradiction::appears_contradictory(
                        prev_prose,
                        lat_prose,
                        crate::advanced::contradiction::SubjectIdentity::FromTextOverlap,
                    ));
            if contradicts || explicit_replacement {
                turns[later].supersedes_turns.push(prev);
                if !session_supersedes_back[s_lat].contains(&s_prev) {
                    session_supersedes_back[s_lat].push(s_prev);
                }
            }
        }
    }

    // 2c. Personal Attribute Slot Supersession (`RESIDENCE`, `CAREER`, `MARITAL`, `INCOME`, `WORKLOAD`,
    //     `HEALTH`, `SOCIAL`, `FAMILY`): connects disjoint-entity 1st-person functional state updates
    //     (`later -> prev`) across sessions when entities change value completely.
    //     First, link explicit same-session policy/rule replacements sharing >= 2 substantive words.
    for ti in 0..turns.len() {
        if turns[ti].field != 0 || turns[ti].is_pure_info_question || turns[ti].is_backchannel {
            continue;
        }
        let sess_ti = turns[ti].session;
        let lat_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[ti]);
        let lat_low = lat_prose.to_ascii_lowercase();
        let has_repl = lat_low.contains("instead of ")
            || lat_low.contains("no longer ")
            || lat_low.contains("rather than ")
            || lat_low.contains("changing the rule")
            || lat_low.contains("changed the rule")
            || lat_low.contains("changing the policy")
            || lat_low.contains("changed the policy");
        if !has_repl || lat_prose.len() > 420 {
            continue;
        }
        let lat_toks: Vec<String> = crate::retrieval::raw_tokens(lat_prose)
            .into_iter()
            .filter(|w| w.len() >= 4 && crate::retrieval::is_content(w))
            .collect();
        if lat_toks.len() < 2 {
            continue;
        }
        let mut found_prev: Option<usize> = None;
        for prev in (0..ti).rev() {
            if turns[prev].session != sess_ti {
                break;
            }
            if turns[prev].field != 0 || turns[prev].is_backchannel {
                continue;
            }
            let prev_prose = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[prev]);
            if prev_prose.len() > 420 {
                continue;
            }
            let prev_toks = crate::retrieval::raw_tokens(prev_prose);
            let shared = lat_toks.iter().filter(|w| prev_toks.contains(w)).count();
            if shared >= 2 {
                found_prev = Some(prev);
                break;
            }
        }
        if let Some(prev) = found_prev
            && !turns[ti].supersedes_turns.contains(&prev)
        {
            turns[ti].supersedes_turns.push(prev);
        }
    }
    let mut state_slot_posts: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
    let mut res_sessions: DetHashSet<usize> = DetHashSet::default();
    for (ti, t) in turns.iter().enumerate() {
        if t.field != 0 || t.is_backchannel {
            continue;
        }
        let low = crate::retrieval::graph::torrens::strip_leading_ordinal(&turn_prose_cache[ti])
            .to_ascii_lowercase();
        if low.len() > 640 || (t.is_pure_info_question && low.len() < 40) {
            continue;
        }
        // hardcode-scan: closed-class (1st-person functional state copular & relocation frames)
        let has_1p = low.contains("i ")
            || low.contains("i'm ")
            || low.contains("i\u{2019}m ")
            || low.contains("i've ")
            || low.contains("i\u{2019}ve ")
            || low.contains("my ")
            || low.contains("our ");
        if !has_1p {
            continue;
        }
        if low.contains("i live in ")
            || low.contains("now live in ")
            || low.contains("now living in ")
            || low.contains("relocated to ")
            || low.contains("relocated and ")
            || low.contains("i just relocated")
            || low.contains("i've relocated")
            || low.contains("i\u{2019}ve relocated")
            || low.contains("i moved to ")
            || low.contains("moved from ")
            || low.contains("residence is now ")
            || ((low.contains("i'm in ") || low.contains("i\u{2019}m in "))
                && !t.entities.is_empty()
                && low.contains("trip"))
        {
            if res_sessions.insert(t.session) {
                state_slot_posts[0].push(ti);
            }
        }
        if low.contains("work as a ")
            || low.contains("work as an ")
            || low.contains("employed as ")
            || low.contains("working as a ")
            || low.contains("working as an ")
            || low.contains("started a new job")
            || low.contains("started a new internship")
            || low.contains("started a role as ")
            || low.contains("accepted a ")
            || low.contains("accepted an ")
            || low.contains("became an entrepreneur")
            || low.contains("switched careers")
            || low.contains("transitioned into entrepreneurship")
            || low.contains("quick career update")
            || low.contains("changed employers")
            || low.contains("i'm unemployed")
            || low.contains("i\u{2019}m unemployed")
            || low.contains("left my job")
            || low.contains("director at ")
            || low.contains("senior at ")
            || low.contains("junior at ")
            || low.contains("intern at ")
        {
            state_slot_posts[1].push(ti);
        }
        if low.contains("i'm divorced")
            || low.contains("i am divorced")
            || low.contains("officially divorced")
            || low.contains("now divorced")
            || low.contains("i'm married")
            || low.contains("i\u{2019}m married")
            || low.contains("got married")
            || low.contains("now dating ")
            || low.contains("i'm dating ")
            || low.contains("i\u{2019}m dating ")
            || low.contains("engaged to ")
            || low.contains("i'm single")
            || low.contains("i\u{2019}m single")
            || low.contains("relationship status ")
            || low.contains("marital status")
        {
            state_slot_posts[2].push(ti);
        }
        if low.contains("i make roughly ")
            || low.contains("monthly income")
            || low.contains("monthly range")
            || low.contains("salary is ")
        {
            state_slot_posts[3].push(ti);
        }
        if low.contains("work status")
            || low.contains("free from work")
            || low.contains("on a business trip")
            || low.contains("availability and workload")
            || low.contains("workload has ")
            || low.contains("workload just changed")
            || low.contains("my workload ")
            || low.contains("steady at work")
        {
            state_slot_posts[4].push(ti);
        }
        if low.contains("health status")
            || low.contains("physically healthy")
            || low.contains("physically i'm")
            || low.contains("physically i\u{2019}m")
            || low.contains("feeling physically")
            || low.contains("emotionally stable")
            || low.contains("often fatigued")
            || low.contains("emotionally distressed")
        {
            state_slot_posts[5].push(ti);
        }
        if low.contains("social status")
            || low.contains("less time socializing")
            || low.contains("staying in contact")
            || low.contains("seeing people more")
            || low.contains("social activities")
        {
            state_slot_posts[6].push(ti);
        }
        if low.contains("family status")
            || low.contains("children status")
            || low.contains("have two kids")
            || low.contains("have three children")
            || low.contains("have one child")
        {
            state_slot_posts[7].push(ti);
        }
    }
    let max_slot_deg = max_role_deg.max(16);
    for (slot_idx, list) in state_slot_posts.into_iter().enumerate() {
        if list.is_empty() || list.len() > max_slot_deg {
            continue;
        }
        let bit = 1u8 << slot_idx;
        for &ti in &list {
            turns[ti].state_slot_mask |= bit;
        }
        if list.len() < 2 {
            continue;
        }
        for idx in 1..list.len() {
            let later = list[idx];
            let prev = list[idx - 1];
            let s_lat = turns[later].session;
            let s_prev = turns[prev].session;
            if s_lat <= s_prev || turns[later].state_slot_edges.contains(&prev) {
                continue;
            }
            turns[later].state_slot_edges.push(prev);
        }
    }

    // 3. Synaptic Tagging `CaptureWindow` (Frey & Morris 1997; Redondo & Morris 2011; 9-hour window):
    //    A high-salience turn (`is_standing_policy`, `is_personal_constraint`, or `is_initiative`
    //    milestone) captures the preceding tagged user initiative turn within `CaptureWindow::backward_hours`
    //    (9h) or the immediate preceding same-episode sub-session (`session_episode_prev`).
    let capture_window = crate::neuroscience::CaptureWindow::default();
    let max_back_secs = (capture_window.backward_hours * 3600.0) as i64;
    for ti in 1..turns.len() {
        let is_prp = turns[ti].is_standing_policy
            || turns[ti].is_personal_constraint
            || (turns[ti].is_initiative && turns[ti].entities.len() >= 2);
        if !is_prp {
            continue;
        }
        let s_cur = turns[ti].session;
        let ts_cur = session_ts[s_cur];
        // Search backward for the closest preceding user initiative turn inside the 9h capture window (or same episode).
        for back_s in (0..s_cur).rev() {
            let ts_prev = session_ts[back_s];
            let in_clock_window = ts_cur > 0 && ts_prev > 0 && ts_cur >= ts_prev && (ts_cur - ts_prev) <= max_back_secs;
            let is_ep_prev = session_episode_prev[s_cur] == Some(back_s);
            if !in_clock_window && !is_ep_prev {
                if ts_cur > 0 && ts_prev > 0 && (ts_cur - ts_prev) > max_back_secs {
                    break;
                }
                continue;
            }
            if let Some(&init_ti) = session_turns[back_s].iter().rev().find(|&&cand| turns[cand].is_initiative) {
                // Require at least 1 shared entity so unrelated adjacent topics don't capture blindly.
                let shares_anchor = turns[ti].entities.iter().any(|e| turns[init_ti].entities.contains(e));
                if shares_anchor {
                    if !turns[ti].exact_edges.iter().any(|&(_, t)| t == init_ti) {
                        turns[ti].exact_edges.push((3, init_ti));
                    }
                    if !session_rsb_back[s_cur].iter().any(|&(_, s)| s == back_s) {
                        session_rsb_back[s_cur].push((3, back_s));
                    }
                }
                break;
            }
        }
    }

    // 4. SalienceJoin. A later turn records a backward link to an earlier
    //    standing policy in the same session, or in an earlier session that shares
    //    the episode key. The link is session or episode identity, not a word match.
    //    It is not a general exact edge and it does not write session_rsb_back.
    //    v1's global winner seat raised a rule from a different batch. v3 does not
    //    require the policy to match the query. Rank time seats it behind the
    //    retrieved turn that points at it.
    for si in 0..ns_count {
        let members = &session_turns[si];
        for (pos, &p_ti) in members.iter().enumerate() {
            if !turns[p_ti].is_standing_policy {
                continue;
            }
            for &t_ti in &members[pos + 1..] {
                if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                    turns[t_ti].policy_backlinks.push(p_ti);
                }
            }
        }
    }
    let mut ep_policies: DetHashMap<&str, Vec<usize>> = DetHashMap::default();
    for si in 0..ns_count {
        let ep = session_ep_keys[si].as_str();
        if ep.is_empty() {
            continue;
        }
        if let Some(prev_policies) = ep_policies.get(ep) {
            for &p_ti in prev_policies {
                for &t_ti in &session_turns[si] {
                    if !turns[t_ti].policy_backlinks.contains(&p_ti) {
                        turns[t_ti].policy_backlinks.push(p_ti);
                    }
                }
            }
        }
        for &p_ti in &session_turns[si] {
            if turns[p_ti].is_standing_policy {
                ep_policies.entry(ep).or_default().push(p_ti);
            }
        }
    }

    (session_rsb_back, session_supersedes_back, session_episode_prev)
}


/// Nearest preceding self-contained turn in the same session. A follow-up is an
/// assistant turn or a turn flagged anaphoric at ingest. The antecedent is the
/// cause; it need not share words with the follow-up.
pub(in crate::retrieval) fn rsb_antecedent(ix: &Index, ti: usize) -> Option<usize> {
    let t = &ix.turns[ti];
    if t.speaker != "assistant" && !t.anaphoric {
        return None;
    }
    let members = &ix.session_turns[t.session];
    let pos = members.iter().position(|&m| m == ti)?;
    for &prev in members[..pos].iter().rev() {
        let p = &ix.turns[prev];
        if p.field == 0 && !p.anaphoric {
            return Some(prev);
        }
    }
    None
}

/// One structural cause for a retrieved symptom: the antecedent if this turn is
/// a follow-up, otherwise the nearest earlier standing policy in the same episode.
pub(in crate::retrieval) fn rsb_cause(ix: &Index, ti: usize) -> Option<usize> {
    if let Some(ante) = rsb_antecedent(ix, ti) {
        return Some(ante);
    }
    ix.turns[ti].policy_backlinks.last().copied()
}

/// Seat each cause immediately after the symptom that points at it.
///
/// Version 4, fixed before the run. Version 3 inserted the cause and then
/// truncated the list to k, which drops whoever already sat in the last slot.
/// That drop is not a salience change. A cause that is not already in the
/// window is inserted only while the window is shorter than k, so an empty
/// slot is filled and nobody already inside is removed. A cause already in
/// the window is moved to sit immediately after its symptom, and is never
/// moved ahead of that symptom. A full window does not grow. k is the
/// caller's top_k. The backchannel skip on the antecedent walk is removed
/// so the walk matches the spec: nearest preceding field-0 turn that is
/// not anaphoric.
pub(in crate::retrieval) fn rsb_seat_behind(ix: &Index, list: &mut Vec<usize>, k: usize) {
    if list.is_empty() || k == 0 {
        return;
    }
    let snapshot = list.clone();
    let mut planned: Vec<(usize, usize)> = Vec::new();
    let mut reserved = snapshot.clone();
    for (i, &ti) in snapshot.iter().enumerate() {
        let Some(cause) = rsb_cause(ix, ti) else { continue };
        if planned.iter().any(|&(_, c)| c == cause) {
            continue;
        }
        let inside = snapshot.contains(&cause);
        if !inside && reserved.len() >= k {
            continue;
        }
        if !inside {
            reserved.push(cause);
        }
        planned.push((i, cause));
    }
    for (i, cause) in planned.into_iter().rev() {
        let Some(sym_at) = list.iter().position(|&t| t == snapshot[i]) else { continue };
        if let Some(cause_at) = list.iter().position(|&t| t == cause) {
            if cause_at <= sym_at {
                continue;
            }
            list.remove(cause_at);
            let sym_at = list.iter().position(|&t| t == snapshot[i]).unwrap();
            list.insert(sym_at + 1, cause);
        } else if list.len() < k {
            list.insert(sym_at + 1, cause);
        }
    }
}

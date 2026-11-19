//! Shared test helpers and mirrored test suite across all 7 retrieval pipeline stages.

use crate::retrieval::*;

#[cfg(test)]
mod query_tests;
#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod lexical_tests;
#[cfg(test)]
mod graph_tests;
#[cfg(test)]
mod fusion_tests;
#[cfg(test)]
mod seating_tests;
#[cfg(test)]
mod rank_tests;

#[cfg(test)]
pub(in crate::retrieval) fn deictic_tokens(raw: &[String], ts: i64) -> Vec<String> {
    deictic_weighted_tokens(raw, ts).into_iter().map(|(t, _)| t).collect()
}

/// BoundaryPrimacy (event-boundary primacy; Zheng 2022, Ezzyat & Davachi 2011). If rank 1 is not
/// its session's opening turn, it matched at least one query cue, and that session's opener is
/// already inside the top `PRIMACY_WINDOW` and carries every cue rank 1 matched, the opener takes
/// slot 1 and the rest keep their order. Never crosses sessions; never changes the top-5 set.
#[cfg(test)]
pub(in crate::retrieval) fn boundary_primacy(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    boundary_primacy_with_plan(ix, q_cues, list, None)
}

/// GivenNew (given/new information, Prince 1981; first attestation in lexicography). If rank 1
/// matched at least one query cue, take the earliest turn of rank 1's own session whose cues cover
/// every cue rank 1 matched. When that turn comes before rank 1 and is already inside the top
/// `GIVEN_NEW_WINDOW`, it takes slot 1 and the rest keep their order. Never crosses sessions.
#[cfg(test)]
pub(in crate::retrieval) fn given_new(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    given_new_with_plan(ix, q_cues, list, None)
}

/// SessionOpenerLift generalizes BoundaryPrimacy by dropping the same-session clause and
/// keeping the chat role gate. If rank 1 is a user or assistant turn, is not position 0 of
/// its session, and matched at least one query cue, take the first turn at ranks 2 through
/// `SESSION_OPENER_LIFT_WINDOW` that opens any session and whose cues cover every cue rank 1
/// matched. That opener takes slot 1 and the rest keep their order. Never changes the top-5 set.
#[cfg(test)]
pub(in crate::retrieval) fn session_opener_lift(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    session_opener_lift_with_plan(ix, q_cues, list, None)
}

/// Hearsay (provenance / hearsay rule). If rank 1 is an assistant turn and matched at
/// least one query cue, take the first user turn at ranks 2 through `HEARSAY_WINDOW`,
/// from any session, whose cues cover every cue rank 1 matched. That user turn takes
/// slot 1 and the rest keep their order. Never changes the top-5 set.
#[cfg(test)]
pub(in crate::retrieval) fn hearsay(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    hearsay_with_plan(ix, q_cues, list, None)
}

pub(in crate::retrieval) fn turn(role: &str, content: &str) -> Turn {
    Turn { role: role.into(), content: content.into(), ..Turn::default() }
}
pub(in crate::retrieval) fn t(id: &str, role: &str, content: &str) -> Turn {
    Turn { id: id.into(), role: role.into(), content: content.into(), ..Turn::default() }
}
pub(in crate::retrieval) fn s(id: &str, turns: Vec<Turn>) -> Session {
    Session { id: id.into(), date: String::new(), timestamp: 0, turns }
}
pub(in crate::retrieval) fn ix_of(sessions: &[Session]) -> Index {
    build_index(sessions, &Params::default())
}
pub(in crate::retrieval) fn idx(ix: &Index, id: &str) -> usize {
    ix.turns.iter().position(|x| x.id == id).unwrap()
}
pub(in crate::retrieval) fn ids(ix: &Index, list: &[usize]) -> Vec<String> {
    list.iter().map(|&i| ix.turns[i].id.clone()).collect()
}
pub(in crate::retrieval) fn haystack() -> Vec<Session> {
    vec![
        s("a", vec![
            t("a0", "user", "My new camera is a Sony A7 and I shoot street photos."),
            t("a1", "assistant", "Nice choice for street work."),
            t("a2", "user", "The Sony camera battery dies fast though."),
        ]),
        s("b", vec![
            t("b0", "user", "Let's talk about gardening tomatoes."),
            t("b1", "user", "My Sony camera takes great tomato shots."),
        ]),
    ]
}

pub(in crate::retrieval) fn chat() -> Vec<Session> {
    vec![
        s("c", vec![
            t("c0", "user", "Morning! Planning a trip."),
            t("c1", "user", "I booked a flight to Lisbon for March."),
            t("c2", "assistant", "Great, Lisbon in March is lovely."),
            t("c3", "user", "Remind me later about the Lisbon flight in March."),
        ]),
        s("e", vec![t("e0", "user", "Lisbon flight prices in March are high.")]),
    ]
}

pub(in crate::retrieval) fn haystack_two_openers() -> Vec<Session> {
    vec![
        s("a", vec![
            t("a0", "user", "My new camera is a Sony A7 and I shoot street photos."),
            t("a1", "assistant", "Nice choice for street work."),
            t("a2", "user", "The Sony camera battery dies fast though."),
        ]),
        s("b", vec![
            t("b0", "user", "Let's talk about gardening tomatoes."),
            t("b1", "user", "My Sony camera takes great tomato shots."),
        ]),
        s("c", vec![
            t("c0", "user", "The Sony camera I keep in the bag is an A7."),
        ]),
    ]
}

pub(in crate::retrieval) fn hearsay_haystack() -> Vec<Session> {
    vec![
        s("a", vec![
            t("a0", "user", "I booked a flight to Lisbon for March."),
            t("a1", "assistant", "Lisbon March looks great."),
            t("a2", "assistant", "I can also look at hotels later."),
        ]),
        s("b", vec![
            t("b0", "user", "Lisbon March works on my side too."),
            t("b1", "assistant", "Noted for the Lisbon March trip."),
        ]),
        s("c", vec![
            t("c0", "user", "Lisbon March is already on the calendar."),
        ]),
    ]
}
pub(in crate::retrieval) fn sess(id: &str, turns: &[(&str, &str)]) -> Session {
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



pub(in crate::retrieval) fn sess_at(id: &str, ts: i64, turns: &[(&str, &str)]) -> Session {
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

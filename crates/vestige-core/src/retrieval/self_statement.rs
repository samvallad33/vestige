//! Self-statement channel: what the user has told us about themselves.
//!
//! People keep the core facts someone shared about themselves on hand. You do
//! not need to hear the word "asthma" again to remember a friend has asthma
//! when they ask about getting a cat. Vestige mirrors that with a small
//! stratified channel, a sibling of `standing_rules`: salient self-disclosures
//! ride alongside every recall without taking an episodic `turns` slot.
//!
//! A turn enters the channel by the shape of the dialogue around it, never by
//! its words. There is no word, cue, pronoun or stopword list here.
//!
//! * **Speaker role.** The turn is the user's, and the very next turn of the
//!   same session is the assistant replying to it (the `answers` adjacency
//!   edge built at ingest).
//! * **Dialogue position.** The turn is not a question and does not answer a
//!   question. Both flags come from the data's own punctuation. Standing
//!   policies are excluded because they already have their own channel.
//! * **How the assistant responds.** The reply is not a question, and it is
//!   an acknowledgement rather than an answer: it carries no more content
//!   terms than the lower quartile of this corpus's assistant turns. An
//!   answer adds information; an acknowledgement files it away. The cut is a
//!   quantile of each corpus's own distribution, so it is the same rule, with
//!   no tuned value, on every corpus.
//!
//! Channel order is salience by corpus specificity: the mean inverse document
//! frequency of the turn's own content terms, so a rare, specific disclosure
//! outranks a generic one regardless of length. Ties break by recency, then
//! by index, so the order is total and deterministic.

use super::TurnIx;

/// Most self-statements returned per query.
///
/// The channel is meant to be small next to the caller's top-k, like a short
/// working set of facts about the person. Three is a fixed, corpus-independent
/// constant; every returned item is reported against k by the bench scorer.
pub(super) const SELF_STATEMENT_CAP: usize = 3;

/// Quantile of the corpus's assistant content-term counts at or below which a
/// reply counts as an acknowledgement. The lower quartile is the standard
/// descriptive cut for "short for this corpus" (Tukey's hinge).
const ACK_QUANTILE_DENOM: usize = 4;

/// Nearest-rank lower quantile `1 / ACK_QUANTILE_DENOM` of `xs`. `None` when empty.
fn lower_quantile(mut xs: Vec<usize>) -> Option<usize> {
    if xs.is_empty() {
        return None;
    }
    xs.sort_unstable();
    let rank = xs.len().div_ceil(ACK_QUANTILE_DENOM).max(1);
    Some(xs[rank - 1])
}

/// True when the user turn `u` is a statement that `r`, the assistant's direct
/// reply, only acknowledges. `ack_cut` is the corpus acknowledgement cut.
fn is_acknowledged_self_statement(turns: &[TurnIx], u: usize, r: usize, ack_cut: usize) -> bool {
    let (user, reply) = (&turns[u], &turns[r]);
    let user_states = user.field == 0
        && !user.is_question
        && !user.answers_question
        && !user.is_standing_policy
        && !user.content.is_empty();
    let reply_acknowledges = reply.field == 1
        && reply.answers == Some(u)
        && !reply.is_question
        && reply.content.len() <= ack_cut;
    user_states && reply_acknowledges
}

/// Ingest-time detection. Returns every self-statement turn index, in channel
/// order (most specific first). `idf` maps a content-term id to its corpus IDF.
pub(super) fn detect(
    turns: &[TurnIx],
    session_turns: &[Vec<usize>],
    session_ts: &[i64],
    idf: impl Fn(u32) -> f32,
) -> Vec<usize> {
    let assistant_sizes = turns
        .iter()
        .filter(|t| t.field == 1)
        .map(|t| t.content.len())
        .collect();
    let Some(ack_cut) = lower_quantile(assistant_sizes) else {
        return Vec::new();
    };
    let mut found: Vec<(usize, f32, i64)> = Vec::new();
    for members in session_turns {
        for pair in members.windows(2) {
            let (u, r) = (pair[0], pair[1]);
            if !is_acknowledged_self_statement(turns, u, r, ack_cut) {
                continue;
            }
            let mut terms: Vec<u32> = turns[u].content.iter().copied().collect();
            terms.sort_unstable();
            let n_terms = terms.len() as f32;
            let specificity = terms.into_iter().map(&idf).sum::<f32>() / n_terms;
            let ts = session_ts.get(turns[u].session).copied().unwrap_or(0);
            found.push((u, specificity, ts));
        }
    }
    found.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.cmp(&a.2)).then(b.0.cmp(&a.0)));
    found.into_iter().map(|(u, _, _)| u).collect()
}

#[cfg(test)]
mod tests {
    use super::super::{rank, Params, Query, RankRequest, RankResult, Session, Turn};
    use super::lower_quantile;

    fn turn(id: &str, role: &str, content: &str) -> Turn {
        Turn {
            id: id.into(),
            role: role.into(),
            content: content.into(),
            ..Turn::default()
        }
    }

    /// A long assistant answer, so acknowledgements stand out as short for this corpus.
    const ANSWER: &str = "Here is a detailed plan covering warm up routines, interval pacing, \
        hydration timing, recovery stretches, sleep scheduling, footwear rotation, \
        terrain variety, weekly mileage progression and tapering before race day.";

    fn ranked(sessions: Vec<Session>, query: &str) -> RankResult {
        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: query.into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        rank(&req).results.remove(0)
    }

    fn corpus() -> Vec<Session> {
        let mut sessions = Vec::new();
        for i in 0..4 {
            sessions.push(Session {
                id: format!("run{i}"),
                date: format!("2024/01/0{} (Mon) 09:00", i + 1),
                timestamp: 0,
                turns: vec![
                    turn(
                        &format!("run{i}#0"),
                        "user",
                        "How should I train for a marathon this spring?",
                    ),
                    turn(&format!("run{i}#1"), "assistant", ANSWER),
                ],
            });
        }
        sessions.push(Session {
            id: "health".into(),
            date: "2024/02/01 (Thu) 09:00".into(),
            timestamp: 0,
            turns: vec![
                turn("health#0", "user", "I've been diagnosed with asthma."),
                turn("health#1", "assistant", "Understood, noted."),
                turn("health#2", "user", "Is pollen season starting soon?"),
                turn("health#3", "assistant", "Yes, soon."),
            ],
        });
        sessions
    }

    #[test]
    fn acknowledged_self_statement_rides_along_with_zero_word_overlap() {
        let r = ranked(corpus(), "My family wants a pet cat, any advice?");
        let ids: Vec<&str> = r.self_statements.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, vec!["health#0"], "self channel: {ids:?}");
    }

    #[test]
    fn question_and_assistant_turns_never_enter_the_channel() {
        let r = ranked(corpus(), "What do you know about my health?");
        for t in &r.self_statements {
            assert_ne!(t.id, "health#2", "a question entered the channel");
            assert!(
                !t.id.ends_with("#1") && !t.id.ends_with("#3"),
                "an assistant turn entered the channel: {}",
                t.id
            );
            assert!(
                !t.id.starts_with("run"),
                "a question answered at length entered the channel: {}",
                t.id
            );
        }
    }

    #[test]
    fn statement_answered_at_length_is_not_a_self_statement() {
        let mut sessions = corpus();
        let longer = format!("{ANSWER} Asthma also changes how you should pace long runs.");
        sessions[4].turns[1] = turn("health#1", "assistant", &longer);
        let r = ranked(sessions, "My family wants a pet cat, any advice?");
        assert!(r.self_statements.is_empty(), "{:?}", r.self_statements);
    }

    #[test]
    fn sidecar_channels_never_change_the_recalled_turns() {
        let r = ranked(corpus(), "How should I train for a marathon?");
        assert!(!r.turns.is_empty());
        let chan: Vec<&str> = r.self_statements.iter().map(|t| t.id.as_str()).collect();
        assert!(chan.len() <= super::SELF_STATEMENT_CAP);
        assert!(r.graph_walk.len() <= super::super::graph_walk::WALK_CAP);
        let recalled: Vec<&str> = r.turns.iter().map(|t| t.id.as_str()).collect();
        for w in &r.graph_walk_topk {
            assert!(
                !recalled.contains(&w.id.as_str()),
                "walk returned a seed: {}",
                w.id
            );
        }
    }

    #[test]
    fn lower_quantile_is_nearest_rank() {
        assert_eq!(lower_quantile(vec![]), None);
        assert_eq!(lower_quantile(vec![7]), Some(7));
        assert_eq!(lower_quantile(vec![4, 1, 3, 2]), Some(1));
        assert_eq!(lower_quantile(vec![8, 1, 3, 2, 5, 6, 7, 4]), Some(2));
    }
}

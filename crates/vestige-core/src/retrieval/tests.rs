//! All `#[cfg(test)]` modules and test-only helpers for the retrieval module.
//! Split out of the former single-file `retrieval.rs` with no logic change.

use super::*;

#[cfg(test)]
pub(super) fn deictic_tokens(raw: &[String], ts: i64) -> Vec<String> {
    deictic_weighted_tokens(raw, ts).into_iter().map(|(t, _)| t).collect()
}

/// BoundaryPrimacy (event-boundary primacy; Zheng 2022, Ezzyat & Davachi 2011). If rank 1 is not
/// its session's opening turn, it matched at least one query cue, and that session's opener is
/// already inside the top `PRIMACY_WINDOW` and carries every cue rank 1 matched, the opener takes
/// slot 1 and the rest keep their order. Never crosses sessions; never changes the top-5 set.
#[cfg(test)]
pub(super) fn boundary_primacy(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    boundary_primacy_with_plan(ix, q_cues, list, None)
}

/// GivenNew (given/new information, Prince 1981; first attestation in lexicography). If rank 1
/// matched at least one query cue, take the earliest turn of rank 1's own session whose cues cover
/// every cue rank 1 matched. When that turn comes before rank 1 and is already inside the top
/// `GIVEN_NEW_WINDOW`, it takes slot 1 and the rest keep their order. Never crosses sessions.
#[cfg(test)]
pub(super) fn given_new(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    given_new_with_plan(ix, q_cues, list, None)
}

/// SessionOpenerLift generalizes BoundaryPrimacy by dropping the same-session clause and
/// keeping the chat role gate. If rank 1 is a user or assistant turn, is not position 0 of
/// its session, and matched at least one query cue, take the first turn at ranks 2 through
/// `SESSION_OPENER_LIFT_WINDOW` that opens any session and whose cues cover every cue rank 1
/// matched. That opener takes slot 1 and the rest keep their order. Never changes the top-5 set.
#[cfg(test)]
pub(super) fn session_opener_lift(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    session_opener_lift_with_plan(ix, q_cues, list, None)
}

/// Hearsay (provenance / hearsay rule). If rank 1 is an assistant turn and matched at
/// least one query cue, take the first user turn at ranks 2 through `HEARSAY_WINDOW`,
/// from any session, whose cues cover every cue rank 1 matched. That user turn takes
/// slot 1 and the rest keep their order. Never changes the top-5 set.
#[cfg(test)]
pub(super) fn hearsay(ix: &Index, q_cues: &[u32], list: &mut Vec<usize>) -> bool {
    hearsay_with_plan(ix, q_cues, list, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, content: &str) -> Turn {
        Turn { role: role.into(), content: content.into(), ..Turn::default() }
    }

    #[test]
    fn rejects_benchmark_labels_at_the_boundary() {
        let leaky = r#"{"sessions":[{"id":"s1","turns":[{"role":"user","content":"x","has_answer":true}]}],"queries":[{"text":"x"}]}"#;
        assert!(serde_json::from_str::<RankRequest>(leaky).is_err());
        let leaky_q = r#"{"sessions":[],"queries":[{"text":"x","question_type":"knowledge-update"}]}"#;
        assert!(serde_json::from_str::<RankRequest>(leaky_q).is_err());
    }

    #[test]
    fn stemmer_converges_inflections() {
        assert_eq!(stem("painting"), stem("painted"));
        assert_eq!(stem("paints"), stem("paint"));
        assert_eq!(stem("baking"), stem("bake"));
        assert_eq!(stem("running"), stem("run"));
        assert_eq!(stem("relocated"), stem("relocating"));
        assert_eq!(stem("stories"), stem("story"));
    }

    #[test]
    fn parses_both_dataset_date_formats() {
        let a = parse_date("2023/05/20 (Sat) 02:21");
        let b = parse_date("1:56 pm on 8 May, 2023");
        assert_eq!(calendar_tokens(a)[..3], ["20".to_string(), "may".into(), "2023".into()]);
        assert_eq!(calendar_tokens(b)[..3], ["8".to_string(), "may".into(), "2023".into()]);
        assert_eq!(calendar_tokens(b)[3], "monday");
        assert!(calendar_tokens(a).contains(&"__DATE__2023_05_20".to_string()));
    }

    #[test]
    fn parses_rfc3339_production_dates() {
        let now = chrono::Utc::now();
        let s = now.to_rfc3339();
        let ts = parse_date(&s);
        assert_eq!(ts, now.timestamp());

        let fixed = "2026-11-09T20:35:54Z";
        let ts2 = parse_date(fixed);
        assert!(ts2 > 0);
        let tokens = calendar_tokens(ts2);
        assert!(tokens.contains(&"__DATE__2026_11_09".to_string()));
    }

    #[test]
    fn deictic_compiler_resolves_yesterday() {
        let ts = parse_date("1:56 pm on 8 May, 2023");
        let toks = deictic_tokens(&raw_tokens("I went to the group yesterday"), ts);
        assert!(toks.contains(&"__DATE__2023_05_07".to_string()) && toks.contains(&"may".to_string()));
    }

    #[test]
    fn knowledge_update_prefers_newest_state() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "old".into(), date: "2024/01/10 (Wed) 09:15".into(), timestamp: 0, turns: vec![turn("user", "My brother just moved to Seattle for a new job.")] },
                Session { id: "noise".into(), date: "2024/02/01 (Thu) 09:15".into(), timestamp: 0, turns: vec![turn("user", "I baked sourdough bread this weekend.")] },
                Session { id: "new".into(), date: "2024/03/22 (Fri) 18:40".into(), timestamp: 0, turns: vec![turn("user", "My brother hated the rain in Seattle so he relocated to Austin, Texas.")] },
            ],
            queries: vec![Query { text: "Where does my brother currently live?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let r = rank(&req);
        assert!(! r.results[0].intent.current_state);
        // Without a current-state phrase, the earlier Seattle mention outranks the move.
        assert_eq!(r.results[0].sessions[0].id, "old");
    }

    #[test]
    fn answer_turn_inherits_question_entities() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D1".into(),
                date: String::new(),
                timestamp: 0,
                turns: vec![
                    Turn { id: "D1:1".into(), role: "Caroline".into(), content: "I went hiking with my dog.".into(), ..Turn::default() },
                    Turn { id: "D1:2".into(), role: "Melanie".into(), content: "Are you still painting that sunrise landscape?".into(), ..Turn::default() },
                    Turn { id: "D1:3".into(), role: "Caroline".into(), content: "Yeah, I switched to acrylics for it yesterday.".into(), ..Turn::default() },
                    Turn { id: "D1:4".into(), role: "Melanie".into(), content: "My kids love the beach.".into(), ..Turn::default() },
                ],
            }],
            queries: vec![Query { text: "What medium did Caroline use for her sunrise landscape?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let r = rank(&req);
        let top2: Vec<&str> = r.results[0].turns.iter().take(2).map(|t| t.id.as_str()).collect();
        assert!(top2.contains(&"D1:3"), "{top2:?}");
    }

    #[test]
    fn assistant_attribution_routes_to_assistant_field() {
        let i = classify("Can you remind me what hotel you recommended in Lisbon?");
        assert!(!i.assistant_attribution && !i.user_attribution);
        let pref = classify("Can you suggest some activities I might like this weekend?");
        assert!(!pref.preference);
    }

    #[test]
    fn ranking_is_deterministic() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "a".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "I adopted a golden retriever puppy named Max.")] },
                Session { id: "b".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "Work has been busy with the quarterly report.")] },
            ],
            queries: vec![Query { text: "What is the name of my puppy?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let a = serde_json::to_string(&rank(&req)).unwrap_or_default();
        let b = serde_json::to_string(&rank(&req)).unwrap_or_default();
        assert_eq!(a, b);
        assert_eq!(rank(&req).results[0].sessions[0].id, "a");
    }

    #[test]
    fn contraction_fragments_never_leak() {
        let toks = raw_tokens("We've checked Caroline's logs and Don didn't go, I'll stay.");
        assert!(!toks.contains(&"ve".to_string()), "{toks:?}");
        assert!(!toks.contains(&"ll".to_string()), "{toks:?}");
        assert!(toks.contains(&"caroline".to_string()), "{toks:?}");
        assert!(toks.contains(&"don".to_string()), "{toks:?}");
        assert!(is_content("don"));
    }

    #[test]
    fn proper_noun_entities_survive_stopwords() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "band1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Local H is an alternative rock duo from Illinois in the United States.")] },
                Session { id: "band2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "For Against is an American post-punk band formed in Lincoln, Nebraska.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The United States Senate held hearings against corruption.")] },
            ],
            queries: vec![Query { text: "Are Local H and For Against both from the United States?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&req).results[0].sessions.iter().take(2).map(|s| s.id.clone()).collect();
        assert!(top2.contains(&"band1".to_string()) && top2.contains(&"band2".to_string()), "{top2:?}");
    }

    #[test]
    fn inline_dates_and_coarse_halo_match() {
        let may1_toks = inline_date_tokens("What movie did Joanna watch on 1 May, 2022?");
        assert!(may1_toks.contains(&"__DATE__2022_05_01".to_string()), "{may1_toks:?}");
        assert!(may1_toks.contains(&"__MONTH__05".to_string()), "{may1_toks:?}");

        let may2_ts = parse_date("2 May, 2022");
        let last_night = deictic_tokens(&raw_tokens("I watched it last night!"), may2_ts);
        assert!(last_night.contains(&"__DATE__2022_05_01".to_string()), "{last_night:?}");

        let mar28_ts = parse_date("2023/03/28 (Tue) 20:35");
        let four_wks = deictic_tokens(&raw_tokens("four weeks ago"), mar28_ts);
        assert!(four_wks.contains(&"__DATE__2023_03_01".to_string()), "{four_wks:?}");
    }

    #[test]
    fn framing_words_dont_crush_domain_facts() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I grew up with 3 sisters in a loud house.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The total number of items on our order list is unknown.")] },
            ],
            queries: vec![Query { text: "What is the total number of siblings I have?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        // "total" and "number" are no longer a framing list, so the turn that
        // contains those tokens outranks the sister mention.
        assert_eq!(rank(&req).results[0].sessions[0].id, "noise");
    }

    #[test]
    fn ppr_turn_scores_remain_bounded_and_length_fair() {
        let filler_essay = "flask app code ".repeat(120);
        let req = RankRequest {
            sessions: vec![
                Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "gold".into(), role: "user".into(), content: "My first sprint ends on March 29 for login features.".into(), ..Turn::default() },
                    Turn { id: "reply".into(), role: "assistant".into(), content: format!("Here is a sprint plan ending March 29: {filler_essay}"), ..Turn::default() },
                ]},
                Session { id: "s2".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise".into(), role: "assistant".into(), content: format!("Another unrelated deployment guide: {filler_essay}"), ..Turn::default() },
                ]},
                Session { id: "s3".into(), date: String::new(), timestamp: 0, turns: vec![
                    Turn { id: "noise2".into(), role: "user".into(), content: "Unrelated question about SQLite migrations.".into(), ..Turn::default() },
                    Turn { id: "noise3".into(), role: "assistant".into(), content: "Sure, here are SQLite migration commands.".into(), ..Turn::default() },
                    Turn { id: "noise4".into(), role: "user".into(), content: "Thanks!".into(), ..Turn::default() },
                ]},
            ],
            queries: vec![Query { text: "When does my first sprint end?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&req).results[0].turns.iter().take(2).map(|t| t.id.clone()).collect();
        assert!(top2.contains(&"gold".to_string()), "{top2:?}");
        assert_ne!(rank(&req).results[0].turns[0].id, "noise");
    }

    #[test]
    fn session_multihop_bridges_second_hop() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Kiss and Tell is a 1945 film starring Shirley Temple as Corliss Archer.")] },
                Session { id: "dup1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "A Kiss for Corliss is the sequel comedy film featuring Corliss Archer.")] },
                Session { id: "dup2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Meet Corliss Archer is a radio sitcom centered on Corliss Archer.")] },
                Session { id: "dup3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Janet Waldo voiced Corliss Archer in the radio broadcast.")] },
                Session { id: "dup4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "F. Hugh Herbert created Corliss Archer as a character.")] },
                Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Shirley Temple served as United States Chief of Protocol and ambassador to Ghana.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Baking sourdough bread at high altitude requires extra hydration.")] },
            ],
            queries: vec![Query { text: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert_eq!(top5[0], "hop1");
        assert!(top5.contains(&"hop2".to_string()), "Expected hop2 bridged into top5: {top5:?}");
    }

    #[test]
    fn is_question_survives_trailing_code_or_markers() {
        let (is_q1, short1) = detect_question("Can you fix my Flask route? ->-> 1,14", 6);
        assert!(is_q1 && short1);
        let (is_q2, _) = detect_question("Can you initialize SQLite?\n```python\nimport sqlite3\n```", 5);
        assert!(is_q2);
    }

    #[test]
    fn previous_state_matches_and_zero_hit_bridge_skips_preference() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I worked as a marketing specialist in my previous role at a startup.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Here is a general list of common occupation titles across industries.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("assistant", "Waiters and teachers are common occupation categories.")] },
            ],
            queries: vec![Query { text: "What was my previous occupation?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        assert_eq!(rank(&req).results[0].sessions[0].id, "gold");

        let pref_req = RankRequest {
            sessions: vec![
                Session { id: "s1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Can you recommend a great show to watch on Paramount Plus tonight?")] },
                Session { id: "zero".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Paramount Plus runs on Apple TV and Roku devices.")] },
                Session { id: "pref".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "I love comedy specials and standup shows.")] },
            ],
            queries: vec![Query { text: "Can you recommend a show or movie for me to watch tonight?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let top2: Vec<String> = rank(&pref_req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(!top2.contains(&"zero".to_string()), "zero-overlap bridge should not enter preference top2: {top2:?}");
    }

    #[test]
    fn standing_policy_classifier_catches_imperatives_and_exclusions() {
        assert!(utterance_is_standing_policy("Always include the receipt hash."));
        // No directive-verb list: a bare imperative without always/never/don't is not a rule.
        assert!(!utterance_is_standing_policy("Please format all JSON with two-space indents."));
        assert!(utterance_is_standing_policy("Never use tabs in this repo."));
        // Reported speech does not open with the adverb, so it is not a rule.
        assert!(!utterance_is_standing_policy("The note says to always include a hash."));
        assert!(!utterance_is_standing_policy("I like two-space indents."));
        assert_eq!(
            standing_policy_id("Always include the receipt hash."),
            standing_policy_id("  always include the receipt hash.  ")
        );
        let seq = classify("What is the chronological order of the three deploys?");
        assert!(!seq.is_event_sequence);
        let mine = classify("What instruction did I tell you to always include?");
        assert!(!mine.user_attribution && !mine.assistant_attribution);
    }

    #[test]
    fn standing_policy_outranks_a_longer_topic_essay() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "essay".into(),
                    date: "2024/06/01 (Sat) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn(
                        "assistant",
                        "Receipt hashes, formatting, and include paths show up in a long essay about build logs, formatters, and hash functions that never states a rule.",
                    )],
                },
                Session {
                    id: "rule".into(),
                    date: "2024/01/02 (Tue) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Always include the receipt hash when you format a response.")],
                },
            ],
            queries: vec![Query {
                text: "What formatting rule did I tell you to always include?".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(!ranked.results[0].intent.user_attribution);
        assert_eq!(ranked.results[0].turns[0].session_id, "rule");
    }

    #[test]
    fn exclusive_preference_beats_a_neutral_mention() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "neutral".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![turn("user", "The cafe menu lists an oat milk latte and a citrus tonic.")],
                },
                Session {
                    id: "exclusive".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![turn(
                        "user",
                        "I switched to oat milk only. The citrus tonic is too bitter and I avoid it.",
                    )],
                },
            ],
            queries: vec![Query {
                text: "What kind of milk would I prefer?".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "exclusive");
    }

    #[test]
    fn event_sequence_spreads_top_turns_across_dates() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "jan".into(),
                    date: "2024/01/04 (Thu) 10:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("user", "We shipped the ledger format to production."),
                        turn("assistant", "Noted the ledger format shipment."),
                        turn("user", "The ledger format deploy finished before noon."),
                    ],
                },
                Session {
                    id: "jan-same".into(),
                    date: "2024/01/04 (Thu) 10:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Another note about the ledger format the same day.")],
                },
                Session {
                    id: "mar".into(),
                    date: "2024/03/02 (Sat) 11:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "We rolled the ledger format back after the audit.")],
                },
                Session {
                    id: "jun".into(),
                    date: "2024/06/09 (Sun) 11:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "We restored the ledger format with signed receipts.")],
                },
            ],
            queries: vec![Query {
                text: "What is the chronological order of the ledger format changes?".into(),
                date: String::new(),
            }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.is_event_sequence);
    }

    #[test]
    fn standing_rule_surfaces_for_a_task_with_no_shared_words() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "formula".into(),
                    date: "2024/02/01 (Thu) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "A circle area uses pi times radius squared, and textbooks repeat that formula.")],
                },
                Session {
                    id: "notes".into(),
                    date: "2024/03/01 (Fri) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Students compute circle areas by squaring the radius and multiplying.")],
                },
                Session {
                    id: "rule".into(),
                    date: "2024/01/04 (Thu) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "Always include the unit symbol when you write a measurement.")],
                },
            ],
            queries: vec![Query { text: "How should I compute the area of a circle?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.task_under_rules);
        assert!(!classify("What are some ways to store leftover soup?").task_under_rules);
        assert!(!classify("Could you outline a backup routine?").task_under_rules);
        assert!(!classify("When did the soup boil?").task_under_rules);
        let rule_turn = ranked.results[0].turns.iter().find(|t| t.session_id == "rule");
        assert!(rule_turn.is_none(), "imperative phrase is not a standing rule: {:?}", ranked.results[0].turns);
        let ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(!ids.contains(&"rule"), "standing rule stayed at zero overlap: {ids:?}");
    }

    #[test]
    fn event_sequence_covers_distinct_user_initiatives() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "echo".into(),
                    date: "2024/05/01 (Wed) 09:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("assistant", "Here is the chronological order of the workshop steps you started, as a list of steps."),
                        turn("assistant", "Another chronological order of those workshop steps you started, repeated as more steps."),
                        turn("assistant", "A third chronological order note of the workshop steps you started yesterday."),
                    ],
                },
                Session {
                    id: "sketch".into(),
                    date: "2024/01/08 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I began sketching the cabinet joints with a pencil before cutting walnut.")],
                },
                Session {
                    id: "glue".into(),
                    date: "2024/02/12 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I glued the walnut panels after the joints were dry and clamped.")],
                },
                Session {
                    id: "sand".into(),
                    date: "2024/03/18 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I sanded the drawers and rubbed oil into the faces.")],
                },
            ],
            queries: vec![Query { text: "What is the chronological order of the workshop steps I started?".into(), date: String::new() }],
            top_k: 4,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.is_event_sequence);
    }

    #[test]
    fn planning_question_binds_an_implicit_constraint() {
        let intent = classify("How should I plan dinner for my guests?");
        assert!(!intent.preference && !intent.task_under_rules);
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "market".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![turn("user", "Guests often eat dinner at the market after a long day.")],
                },
                Session {
                    id: "limit".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![turn("user", "I don't want shellfish when guests come over for dinner.")],
                },
            ],
            queries: vec![Query { text: "How should I plan dinner for my guests?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        assert_eq!(rank(&req).results[0].sessions[0].id, "limit");
    }

    #[test]
    fn short_reply_inherits_the_neighbor_turn() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "chat".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("alex", "The walnut cabinet needs another coat of oil."),
                    turn("blair", "Done."),
                    turn("alex", "Unrelated note about train tickets and a station platform."),
                ],
            }],
            queries: vec![Query { text: "Who finished the walnut cabinet oil?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.iter().any(|id| id.ends_with("#1")), "short reply missed the neighbor words: {ids:?}");
    }

    #[test]
    fn harris_pure_info_question_does_not_leak_assistant_reply_while_factual_question_keeps_answer() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "trip".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn(
                        "user",
                        "I am planning another vacation soon. Can you give me tips on traveling in Asia?",
                    ),
                    turn(
                        "assistant",
                        "Sure! Many travelers spend fourteen days visiting ancient Kyoto temples in Japan.",
                    ),
                    turn(
                        "user",
                        "I actually visited Fushimi Inari Shrine in Kyoto and stayed two weeks in Japan during autumn.",
                    ),
                ],
            }],
            queries: vec![Query {
                text: "How long was I in Kyoto Japan?".into(),
                date: String::new(),
            }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].turns[0].id, "trip#2");
    }

    #[test]
    fn novel_assertion_protected_from_qecho_penalty_even_with_trailing_question() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "charity".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn(
                        "user",
                        "I need training tips for raising money for charity since I raised $500 for American Cancer Society last month. Any advice?",
                    ),
                    turn(
                        "user",
                        "I am thinking about raising money for charity this season and hoping to do well. Can you share advice on raising money for charity?",
                    ),
                    turn("user", "Earlier I stretched before going out on the track."),
                ],
            }],
            queries: vec![Query {
                text: "How much money did I raise for charity?".into(),
                date: String::new(),
            }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].turns[0].id, "charity#0");
    }

    #[test]
    fn panini_karaka_wh_person_prefers_named_declarative_answer_ignoring_mid_turn_i() {
        assert_eq!(count_slot_entities_mode("Can you check? I went to the clinic.", true), 0);
        assert_eq!(count_slot_entities_mode("I visited Elena Alvarez at the clinic.", true), 2);
        let req = RankRequest {
            sessions: vec![Session {
                id: "mentor".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("user", "Did you say we could work on the design project at the studio tomorrow?"),
                    turn("user", "I planned to work on the design project with Elena Morales at the studio."),
                ],
            }],
            queries: vec![Query {
                text: "Who did I work with at the design studio?".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].turns[0].id, "mentor#1");
    }

    #[test]
    fn adj_partner_diversity_factor_demotes_adjacent_assistant_mirror() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "s1".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("user", "My ceramic mug collection started in Kyoto last autumn."),
                    turn("assistant", "Kyoto is known for hand-glazed ceramic mugs and pottery kilns."),
                ],
            }],
            queries: vec![Query {
                text: "Kyoto ceramic mug".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ix = build_index(&req.sessions, &req.params);
        let factor = adj_partner_diversity_factor(&ix, 1, 0, &[0], false, false, 2);
        assert!((factor - 0.25).abs() < 1e-6);
        let single_sess_factor = adj_partner_diversity_factor(&ix, 1, 0, &[0], false, false, 3);
        assert!((single_sess_factor - 1.0).abs() < 1e-6);
        let asst_query_factor = adj_partner_diversity_factor(&ix, 1, 0, &[0], true, false, 2);
        assert!((asst_query_factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn move3_weekday_modulo_zero_resolves_seven_days_and_demotes_scaffolding() {
        // 2023/04/14 is a Friday -> "last Friday" must resolve to 2023/04/07 (-7 days), never yesterday (-1d).
        let q_ts = parse_date("2023/04/14 (Fri) 12:00");
        let raw = raw_tokens("What gym workout did I finish last Friday?");
        let spec = parse_relative_time_specs_from_raw(&raw, q_ts);
        let expected_ts = parse_date("2023/04/07 (Fri) 12:00");
        assert!(
            spec.points.iter().any(|&(ts, _)| (ts - expected_ts).abs() < 60),
            "last Friday on Friday must resolve -7d: {:?}",
            spec.points
        );
        // Zero-weighting relative-time scaffolding keeps off-date "few months" distractors below true temporal anchors.
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "distractor_wrong_date".into(),
                    date: "2023/04/10 (Mon) 12:00".into(),
                    timestamp: 0,
                    turns: vec![turn(
                        "user",
                        "For a few months I practiced meditation every morning in the past months.",
                    )],
                },
                Session {
                    id: "gold_target_date".into(),
                    date: "2023/01/15 (Sun) 12:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I practiced guitar every morning.")],
                },
                Session {
                    id: "filler_turn".into(),
                    date: "2022/11/01 (Tue) 12:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "I cooked pasta for dinner.")],
                },
            ],
            queries: vec![Query {
                text: "What did I practice a few months ago?".into(),
                date: "2023/04/14 (Fri) 12:00".into(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let out = rank(&req);
        assert_eq!(out.results[0].sessions[0].id, "gold_target_date");
    }

    #[test]
    fn move3_does_not_misparse_duration_in_two_weeks_as_lookback_anchor() {
        let q_ts = parse_date("2023/05/30 (Tue) 23:16");
        let raw = raw_tokens("What was the approximate increase in Instagram followers I experienced in two weeks?");
        let spec = parse_relative_time_specs_from_raw(&raw, q_ts);
        assert!(spec.windows.is_empty() && spec.points.is_empty(), "{:?}", spec.windows);
    }

    #[test]
    fn constraint_object_turns_preference_on_without_a_planning_cue() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "menu".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "The diner menu lists walnut pie and citrus tart.")] },
                Session { id: "limit".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "I am allergic to walnuts and skip that dessert.")] },
            ],
            queries: vec![Query { text: "Does the diner menu include walnuts?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.preference);
        let ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"limit"), "constraint object never entered the ranking: {ids:?}");
    }

    #[test]
    fn a_dateless_question_does_not_invent_a_date_window() {
        assert!(inline_date_tokens("How do I sand a drawer?").is_empty());
        let early = inline_date_tokens("We met in early May before the show.");
        assert!(early.iter().any(|t| t == "__EARLY__05"), "{early:?}");
        assert!(early.iter().any(|t| t == "__MONTH__05"), "{early:?}");
    }

    #[test]
    fn speaker_boost_does_not_drop_the_other_speaker() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "talk".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("alex", "Alex packed the walnut oil for the cabinet."),
                    turn("blair", "Blair packed the walnut oil for the cabinet too."),
                ],
            }],
            queries: vec![Query { text: "What did Alex pack for the cabinet?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids[0].ends_with("#0"), "named speaker should lead: {ids:?}");
    }

    #[test]
    fn one_chat_cannot_fill_the_top_five_when_another_fact_exists() {
        let walnut = "The walnut cabinet oil was mixed, brushed, and left to cure overnight.";
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "walnut".into(),
                    date: "2024/01/02 (Tue) 09:00".into(),
                    timestamp: 0,
                    turns: vec![
                        turn("user", walnut),
                        turn("user", "More walnut cabinet oil notes about the second coat and the brush."),
                        turn("user", "Still the walnut cabinet oil, now describing the rag and the cure."),
                        turn("user", "Again the walnut cabinet oil, with the tin and the cloth."),
                        turn("user", "Yet more walnut cabinet oil, including the shelf and the lid."),
                    ],
                },
                Session {
                    id: "maple".into(),
                    date: "2024/03/04 (Mon) 09:00".into(),
                    timestamp: 0,
                    turns: vec![turn("user", "The maple drawer rails were cut, sanded, and fitted today.")],
                },
            ],
            queries: vec![Query {
                text: "What connects the walnut cabinet oil and the maple drawer rails?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(!ranked.results[0].intent.multi_hop);
        let sessions: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.session_id.as_str()).collect();
        assert!(sessions.contains(&"maple"), "second fact was buried in one chat: {sessions:?}");
        let walnut_n = sessions.iter().filter(|s| **s == "walnut").count();
        assert_eq!(walnut_n, 3, "without a multi-hop phrase the walnut chat keeps three seats: {sessions:?}");
    }

    #[test]
    fn pseudo_trigger_idioms_are_not_classified_as_standing_policies() {
        assert!(!utterance_is_standing_policy("Don't worry about the rain."));
        assert!(!utterance_is_standing_policy("Don't know where the wrench went."));
        assert!(!utterance_is_standing_policy("Don't think we tried the cedar stain."));
        assert!(!utterance_is_standing_policy("Never mind, I found the key."));
        assert!(!utterance_is_standing_policy("Never been to Lisbon."));
        assert!(!utterance_is_standing_policy("Always wanted to learn cello."));
        assert!(!utterance_is_standing_policy("Make sure I understand your diagram."));
        assert!(utterance_is_standing_policy("Please always quote file paths."));
        assert!(!utterance_is_standing_policy("Make sure every migration has a rollback."));
    }

    #[test]
    fn backchannel_reaction_does_not_outrank_the_fact_turn() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "pergola".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("nora", "I stained the cedar pergola beams with tung oil yesterday."),
                    turn("theo", "Wow, awesome!"),
                ],
            }],
            queries: vec![Query { text: "What oil did Nora use on the cedar pergola?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(ranked.results[0].turns[0].id.ends_with("#0"), "{:?}", ranked.results[0].turns);
    }

    #[test]
    fn ordinal_prepositional_and_bare_may_dates_parse_cleanly() {
        assert!(inline_date_tokens("How may I sand a drawer?").is_empty());
        let may_prep = inline_date_tokens("What did Nora build in May?");
        assert!(may_prep.contains(&"__MONTH__05".to_string()), "{may_prep:?}");
        let ord = inline_date_tokens("We met on the 15th of March before dusk.");
        assert!(ord.contains(&"__MD__03_15".to_string()), "{ord:?}");
        let full_ord = inline_date_tokens("Did the kiln fire on May 1st, 2023?");
        assert!(full_ord.contains(&"__DATE__2023_05_01".to_string()), "{full_ord:?}");
        let season = inline_date_tokens("Where did they camp in the summer of 2023?");
        assert!(season.contains(&"__SY__2023_SU".to_string()), "{season:?}");
    }

    #[test]
    fn single_token_entity_and_header_title_bridge_second_hop() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "hop1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The Harbor Glassworks exhibition was curated by Mateo in Lisbon.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Harbor Glassworks opened a small gift counter near the ferry.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Visitors praised the Harbor Glassworks lighting display.")] },
                Session { id: "d3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Tickets for Harbor Glassworks sold out by noon.")] },
                Session { id: "d4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Catalogs from Harbor Glassworks listed twelve bowls.")] },
                Session { id: "hop2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Mateo (ceramist)\nMateo trained at the Porto Kiln Guild before curating glass.")] },
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Sourdough starter hydration depends on flour protein.")] },
            ],
            queries: vec![Query { text: "Where did the curator of the Harbor Glassworks exhibition train?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(top5.contains(&"hop2".to_string()), "single-token proper bridge missed: {top5:?}");
    }

    #[test]
    fn unrelated_hobbies_do_not_flood_topical_preference_queries() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "h1".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "I love playing blitz chess on rainy evenings.")] },
                Session { id: "h2".into(), date: String::new(), timestamp: 2, turns: vec![turn("user", "I enjoy knitting wool scarves in winter.")] },
                Session { id: "h3".into(), date: String::new(), timestamp: 3, turns: vec![turn("user", "I love collecting vintage postage stamps.")] },
                Session { id: "h4".into(), date: String::new(), timestamp: 4, turns: vec![turn("user", "I prefer fountain pens with fine nibs.")] },
                Session { id: "h5".into(), date: String::new(), timestamp: 5, turns: vec![turn("user", "I would like to visit Patagonia someday.")] },
                Session { id: "gold".into(), date: String::new(), timestamp: 6, turns: vec![turn("user", "My camera bag holds a compact prime lens for street photography.")] },
            ],
            queries: vec![Query { text: "What would be a good lens to add to my camera kit?".into(), date: String::new() }],
            top_k: 3,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "gold");
    }

    #[test]
    fn structured_typed_slots_bind_and_supersede_on_current_state() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "old".into(), date: "2024-01-10".into(), timestamp: 0, turns: vec![turn("user", "Quota: 250gb\nInitial storage plan.")] },
                Session { id: "new".into(), date: "2024-08-15".into(), timestamp: 0, turns: vec![turn("user", "Quota: 500gb\nUpgraded storage plan.")] },
            ],
            queries: vec![Query { text: "What is my current storage quota?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].sessions[0].id, "new");
    }

    #[test]
    fn tied_turns_are_identical_across_two_runs() {
        let mut sessions = Vec::new();
        for i in 0..12 {
            sessions.push(Session {
                id: format!("s{i}"),
                date: String::new(),
                timestamp: i,
                turns: vec![turn("user", "alpha beta gamma delta"), turn("assistant", "alpha beta gamma delta")],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "alpha beta gamma delta".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        assert_eq!(serde_json::to_string(&rank(&req)).unwrap(), serde_json::to_string(&rank(&req)).unwrap());
    }

    #[test]
    fn exact_edge_replaces_a_weaker_slot_without_shared_words() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "now".into(),
                    date: String::new(),
                    timestamp: 2,
                    turns: vec![Turn {
                        id: "trigger".into(),
                        role: "user".into(),
                        content: "The ledger key is cedar.".into(),
                        links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                        stability: 4.0,
                    }],
                },
                Session {
                    id: "then".into(),
                    date: String::new(),
                    timestamp: 1,
                    turns: vec![Turn {
                        id: "earlier".into(),
                        role: "user".into(),
                        content: "quartz violin under the stairs".into(),
                        stability: 9.0,
                        ..Turn::default()
                    }],
                },
                Session { id: "n1".into(), date: String::new(), timestamp: 3, turns: vec![turn("user", "marble compass on the table")] },
                Session { id: "n2".into(), date: String::new(), timestamp: 4, turns: vec![turn("user", "linen harbor in the fog")] },
                Session { id: "n3".into(), date: String::new(), timestamp: 5, turns: vec![turn("user", "pewter lantern by the door")] },
                Session { id: "n4".into(), date: String::new(), timestamp: 6, turns: vec![turn("user", "silk orchard past the gate")] },
            ],
            queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|t| t.id.clone()).collect();
        assert!(ids.iter().any(|id| id == "trigger"), "{ids:?}");
        assert!(ids.iter().any(|id| id == "earlier"), "exact edge missed the zero-overlap turn: {ids:?}");
    }

    #[test]
    fn a_backward_edge_lands_inside_the_first_five_not_the_tail() {
        let mut sessions = vec![
            Session {
                id: "now".into(),
                date: String::new(),
                timestamp: 20,
                turns: vec![Turn {
                    id: "trigger".into(),
                    role: "user".into(),
                    content: "The ledger key is cedar.".into(),
                    links: vec![TurnLink { kind: "backfill".into(), to: "earlier".into(), degree: 2 }],
                    ..Turn::default()
                }],
            },
            Session {
                id: "then".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![Turn {
                    id: "earlier".into(),
                    role: "user".into(),
                    content: "quartz violin under the stairs".into(),
                    ..Turn::default()
                }],
            },
        ];
        for i in 0..8 {
            sessions.push(Session {
                id: format!("f{i}"),
                date: String::new(),
                timestamp: 2 + i,
                turns: vec![turn("user", &format!("ledger {}", "padding ".repeat(40)))],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "Where is the ledger key?".into(), date: String::new() }],
            top_k: 10,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let rows: Vec<String> = ranked.results[0].turns.iter().map(|t| format!("{}:{:.3}", t.id, t.score)).collect();
        let pos = rows.iter().position(|row| row.starts_with("earlier:"));
        assert!(pos.is_some_and(|p| p < 5), "backward edge parked in the tail: {rows:?}");
    }

    #[test]
    fn rabin_karp_header_title_dictionary_bridges_numeric_and_multiword_titles() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "anchor".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "The 1928 Winter Games figure skating pair event was won in St. Moritz alongside the premiere of Symphony No. 9.")] },
                Session { id: "d1".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Figure skating pair events in St. Moritz drew Alpine crowds.")] },
                Session { id: "d2".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "St. Moritz figure skating pair medals were awarded outdoors.")] },
                Session { id: "d3".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Winter figure skating pair routines included spins.")] },
                Session { id: "d4".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Judges scored the figure skating pair program in St. Moritz.")] },
                Session { id: "target".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Symphony No. 9 (piece)\nComposed in Vienna, this symphony features a choral finale.")] },
            ],
            queries: vec![Query { text: "In which city was the symphony premiered alongside the St. Moritz figure skating pair event composed?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let top5: Vec<String> = rank(&req).results[0].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(top5.contains(&"target".to_string()), "Rabin-Karp header bridge missed: {top5:?}");
    }

    #[test]
    fn word_boundary_cues_never_misfire_on_substrings() {
        let army = classify("Which Army Academy did Amy attend in Miami's district?");
        assert!(!army.user_attribution, "Army/Amy/Miami's tripped user attribution: {army:?}");

        let server = classify("How does the server service preserve gifted packets?");
        assert!(!server.preference, "server/gifted tripped preference: {server:?}");

        let renew = classify("What did Matthew renew before curfew?");
        assert!(!renew.current_state, "renew tripped current_state: {renew:?}");

        let claim = "Did the bilateral candidate validate the $0 claim?";
        let bilateral = classify(claim);
        assert!(!bilateral.temporal, "bilateral/validate/$0 tripped temporal: {bilateral:?}");
        let padded = word_padded(claim);
        assert!(padded.contains("$0"), "currency amount was stripped: {padded}");

        assert!(is_content("one"), "Formula One lost the content token one");
        assert!(raw_tokens("Formula One").contains(&"one".to_string()));

        let ix = build_index(
            &[Session {
                id: "claim".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    turn("user", "They claimed victory"),
                    turn("user", "Their aim was victory"),
                ],
            }],
            &Params::default(),
        );
        assert!(!ix.turns[0].has_causal, "no recorded link, so no causal flag");
        assert!(!ix.turns[1].has_causal, "the word aim is not a causal signal");
    }

    #[test]
    fn standing_rules_channel_fires_without_stealing_multihop_episodic_slots() {
        let episodic = [
            ("e1", "2024/01/03 (Wed) 09:00", "I logged an otter survey and counted 2 pups by the reed bank."),
            ("e2", "2024/02/11 (Sun) 09:00", "I logged an otter survey and counted 4 pups by the cove."),
            ("e3", "2024/03/19 (Tue) 09:00", "I logged an otter survey and counted 1 pup by the marsh."),
            ("e4", "2024/04/07 (Sun) 09:00", "I logged an otter survey and counted 6 pups by the pier."),
            ("e5", "2024/05/22 (Wed) 09:00", "I logged an otter survey and counted 3 pups by the inlet."),
        ];
        let mut sessions = Vec::new();
        for (id, date, content) in episodic {
            sessions.push(Session {
                id: id.into(),
                date: date.into(),
                timestamp: 0,
                turns: vec![turn("user", content)],
            });
        }
        sessions.push(Session {
            id: "rule".into(),
            date: "2023/12/01 (Fri) 09:00".into(),
            timestamp: 0,
            turns: vec![turn("user", "Always include the unit symbol when writing measurements.")],
        });
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "How many otter surveys did I log?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let intent = &ranked.results[0].intent;
        assert!(intent.sniper_how_many && !intent.multi_hop, "{intent:?}");
        let ids: Vec<&str> = ranked.results[0].sessions.iter().take(5).map(|s| s.id.as_str()).collect();
        for id in ["e1", "e2", "e3", "e4", "e5"] {
            assert!(ids.contains(&id), "episodic slot missing {id}: {ids:?}");
        }
        assert!(!ids.contains(&"rule"), "standing rule stole an episodic slot: {ids:?}");
        assert!(
            ranked.results[0].standing_rules.iter().any(|r| r.session_id == "rule"),
            "standing channel missed the zero-overlap rule: {:?}",
            ranked.results[0].standing_rules
        );
    }

    #[test]
    fn ingest_rsb_handle_and_contradiction_bridge_backward_only() {
        let path_words = "Glossary lists src auth token rs as separate syllables.";
        let noise = "Marble compass beside a cedar shelf.";
        let mut sessions = vec![
            Session {
                id: "S0".into(),
                date: String::new(),
                timestamp: 1_700_000_000,
                turns: vec![Turn {
                    id: "T0".into(),
                    role: "user".into(),
                    content: "Pinned legacy rotation window to 15m in src/auth/token.rs".into(),
                    ..Turn::default()
                }],
            },
        ];
        for (i, content) in [path_words, path_words, path_words, noise, noise].into_iter().enumerate() {
            sessions.push(Session {
                id: format!("F{}", i + 1),
                date: String::new(),
                timestamp: 1_700_000_000 + ((i as i64) + 1) * 200_000,
                turns: vec![turn("user", content)],
            });
        }
        sessions.push(Session {
            id: "S6".into(),
            date: String::new(),
            timestamp: 1_700_000_000 + 6 * 200_000,
            turns: vec![Turn {
                id: "T6".into(),
                role: "user".into(),
                content: "Production outage caused by expired JWT signature in src/auth/token.rs".into(),
                ..Turn::default()
            }],
        });
        sessions.push(Session {
            id: "S7".into(),
            date: String::new(),
            timestamp: 1_700_000_000 + 7 * 200_000,
            turns: vec![Turn {
                id: "T7".into(),
                role: "user".into(),
                content: "Quiet refactor landed beside src/auth/token.rs overnight.".into(),
                ..Turn::default()
            }],
        });
        assert_eq!(sessions.len(), 8);

        let ix = build_index(&sessions, &Params::default());
        assert!(
            ix.session_rsb_back[6].iter().any(|&(_, prev)| prev == 0),
            "later outage did not link backward to the earlier path: {:?}",
            ix.session_rsb_back
        );
        assert!(
            ix.session_rsb_back.iter().enumerate().all(|(later, edges)| {
                edges.iter().all(|&(_, earlier)| earlier < later)
            }),
            "RSB edge pointed forward: {:?}",
            ix.session_rsb_back
        );
        assert!(
            ix.session_supersedes_back.iter().enumerate().all(|(later, edges)| {
                edges.iter().all(|&earlier| earlier < later)
            }),
            "contradiction edge pointed forward: {:?}",
            ix.session_supersedes_back
        );
        assert!(ix.session_rsb_back[6].iter().all(|&(_, prev)| prev != 7));

        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: "What caused the production outage with expired JWT signature?".into(),
                date: String::new(),
            }],
            top_k: 8,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let session_ids: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(session_ids.contains(&"S6"), "outage session missing: {session_ids:?}");
        assert!(session_ids.contains(&"S0"), "backward RSB missed the earlier path: {session_ids:?}");
        assert!(!session_ids.contains(&"S7"), "RSB pulled a later session forward: {session_ids:?}");
        let turn_ids: Vec<&str> = ranked.results[0].turns.iter().map(|t| t.id.as_str()).collect();
        assert!(turn_ids.contains(&"T0"), "backward RSB missed turn T0: {turn_ids:?}");
        assert!(!turn_ids.contains(&"T7"), "RSB pulled later turn T7 forward: {turn_ids:?}");
    }

    #[test]
    fn uncarried_scene_pooling_prevents_reaction_and_vocative_rank1_steal() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D1".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    Turn {
                        id: "D1:1".into(),
                        role: "Caroline".into(),
                        content: "I adopted a rescue greyhound yesterday!".into(),
                        ..Turn::default()
                    },
                    Turn {
                        id: "D1:2".into(),
                        role: "Melanie".into(),
                        content: "Aww amazing Caroline! Why did you choose a greyhound?".into(),
                        ..Turn::default()
                    },
                ],
            }],
            queries: vec![Query { text: "What dog did Caroline adopt?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let top = ranked.results[0].turns.first().map(|t| t.id.as_str());
        assert_eq!(top, Some("D1:1"), "vocative follow-up stole rank 1: {:?}", ranked.results[0].turns);
    }

    #[test]
    fn query_direct_header_title_match_boosts_canonical_subject_doc() {
        let bury = "coastal conservatory rostrum downbeat catalog notes";
        let mut sessions = vec![
            Session { id: "scott".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Scott Derrickson\nAmerican director based in Denver.")] },
            Session { id: "wolf".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Wolf Alice\nEnglish band formed in London.")] },
            Session { id: "ambroise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Ambroise Thomas\nFrench composer who taught in Paris.")] },
            Session { id: "vcu".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Virginia Commonwealth University\nPublic campus in Richmond.")] },
            Session { id: "hayden".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Hayden (musician)\nCanadian singer from Thornhill.")] },
            Session { id: "time_doc".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Time\nWeekly magazine based in New York.")] },
            Session { id: "music_doc".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "Music\nTheory of scales and intervals.")] },
        ];
        for i in 0..3 {
            sessions.push(Session {
                id: format!("coast{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", &format!("{bury} visit {i}"))],
            });
        }
        for i in 0..3 {
            sessions.push(Session {
                id: format!("rehearse{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", &format!("The rehearsal schedule lists the ensemble downbeat rostrum {i}"))],
            });
        }
        for i in 0..6 {
            sessions.push(Session {
                id: format!("dilute{i}"),
                date: String::new(),
                timestamp: 0,
                turns: vec![turn("user", "padding mentions time and music without a title line")],
            });
        }
        let req = RankRequest {
            sessions,
            queries: vec![
                Query {
                    text: "Where did Scott Derrickson record with Wolf Alice and Hayden after Ambroise Thomas left Virginia Commonwealth University?".into(),
                    date: String::new(),
                },
                Query {
                    text: "What time is the music rehearsal?".into(),
                    date: String::new(),
                },
            ],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let titles: Vec<String> = ranked.results[0].sessions.iter().map(|s| s.id.clone()).collect();
        for id in ["scott", "wolf", "ambroise", "vcu", "hayden"] {
            assert!(titles.contains(&id.to_string()), "canonical title missed: {id} in {titles:?}");
        }
        let common: Vec<String> = ranked.results[1].sessions.iter().map(|s| s.id.clone()).collect();
        assert!(!common.contains(&"time_doc".to_string()), "lowercase time bridged: {common:?}");
        assert!(!common.contains(&"music_doc".to_string()), "lowercase music bridged: {common:?}");
    }

    #[test]
    fn conditional_when_and_scope_matched_standing_rule_win_episodic_slot() {
        let q1 = classify("What are some common responses when something goes wrong with an API?");
        assert!(!q1.temporal && !q1.sniper_when && !q1.task_under_rules, "{q1:?}");
        let q2 = classify("When building an application that communicates with a REST API, what typical errors should I be prepared to handle?");
        assert!(!q2.temporal && !q2.sniper_when && !q2.task_under_rules, "{q2:?}");
        let q3 = classify("When does my first sprint end?");
        assert!(q3.temporal && q3.sniper_when && !q3.task_under_rules, "{q3:?}");

        let req = RankRequest {
            sessions: vec![
                Session { id: "newer-unit".into(), date: String::new(), timestamp: 400, turns: vec![turn("user", "Always include the unit symbol.")] },
                Session { id: "newer-tabs".into(), date: String::new(), timestamp: 300, turns: vec![turn("user", "Never use tabs in this repo.")] },
                Session { id: "scope-soup".into(), date: String::new(), timestamp: 1, turns: vec![turn("user", "Always keep a lid when labeling soup.")] },
                Session { id: "essay".into(), date: String::new(), timestamp: 50, turns: vec![turn("user", "Store leftover soup.")] },
            ],
            queries: vec![Query { text: "How should I store leftover soup?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let sessions: Vec<&str> = ranked.results[0].sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(sessions.contains(&"scope-soup"), "spec winner missed episodic slot: {sessions:?}");
        assert!(!sessions.contains(&"newer-unit"), "newer unmatched rule stole slot: {sessions:?}");
    }

    #[test]
    fn beam_timeline_routes_to_user_and_ssa_collective_cues_stay_assistant() {
        let order = classify("Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations?");
        assert!(!order.is_event_sequence && !order.user_attribution && !order.assistant_attribution && !order.task_under_rules, "{order:?}");
        let ssa = classify("I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job in the list you provided?");
        assert!(!ssa.assistant_attribution && !ssa.user_attribution, "{ssa:?}");
    }

    #[test]
    fn wordnet_hypernym_bridges_a_query_word_the_corpus_never_used() {
        let req = RankRequest {
            sessions: vec![
                Session { id: "noise".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "We repainted the fence and planted tomatoes.")] },
                Session { id: "gold".into(), date: String::new(), timestamp: 0, turns: vec![turn("user", "We brought home a puppy from the shelter.")] },
            ],
            queries: vec![Query { text: "Where did we get our dog?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].sessions[0].id, "gold", "{:?}", ranked.results[0].sessions);
        assert!(ranked.results[0].sessions[0].score > 0.0);

        // Control (1): any lexical overlap closes the gate, so "dog" is not expanded
        // and the turn that shares "fence" wins.
        let overlap = RankRequest {
            queries: vec![Query { text: "Did the dog jump the fence?".into(), date: String::new() }],
            ..req
        };
        assert_eq!(rank(&overlap).results[0].sessions[0].id, "noise");
    }

    #[test]
    fn weapon4_versions_orthography_and_abbreviations() {
        assert_eq!(raw_tokens("V8 engine"), vec!["v8".to_string(), "engine".to_string()]);
        assert_eq!(raw_tokens("Bootstrap v5.3.0"), raw_tokens("Bootstrap 5.3.0"));
        assert_eq!(raw_tokens("enable 2FA"), vec!["enable".to_string(), "2fa".to_string()]);
        assert_ne!(raw_tokens("enable 2FA"), raw_tokens("enable two-factor"));
        assert_eq!(raw_tokens("$1,200"), vec!["1200".to_string()]);
        assert_ne!(stem("html5"), stem("html"));
        assert_ne!(stem("css3"), stem("css"));
        assert_eq!(stem("colour"), stem("color"));
        assert_eq!(stem("theatre"), stem("theater"));
        assert_eq!(stem("organise"), stem("organize"));
        assert_eq!(stem("organises"), stem("organizes"));
        assert_eq!(stem("irises"), stem("iris"));
        assert_eq!(stem("edu"), "edu");
        assert_eq!(stem("fam"), "fam");
        assert_eq!(stem("pic"), "pic");
        assert_eq!(stem("career"), "car");
    }

    #[test]
    fn multi_sentence_question_echo_and_speaker_nicknames_resolve_cleanly() {
        let req = RankRequest {
            sessions: vec![Session {
                id: "D3".into(),
                date: String::new(),
                timestamp: 1,
                turns: vec![
                    Turn {
                        id: "D3:15".into(),
                        role: "Caroline".into(),
                        content: "Wow, what an amazing family pic! How long have you been married?".into(),
                        ..Turn::default()
                    },
                    Turn {
                        id: "D3:16".into(),
                        role: "Melanie".into(),
                        content: "5 years already! Time flies.".into(),
                        ..Turn::default()
                    },
                ],
            }],
            queries: vec![Query { text: "How long have Mel and her husband been married?".into(), date: String::new() }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert_eq!(ranked.results[0].turns[0].id, "D3:16", "{:?}", ranked.results[0].turns);
    }

    #[test]
    fn engineering_choice_beats_progress_narrative_and_assistant_expansion() {
        let req = RankRequest {
            sessions: vec![
                Session {
                    id: "debug".into(),
                    date: String::new(),
                    timestamp: 4,
                    turns: vec![turn("user", "I've been debugging how to set up and keep the API cache responses simple and lightweight. I've been working on the API cache responses.")],
                },
                Session {
                    id: "choice".into(),
                    date: String::new(),
                    timestamp: 5,
                    turns: vec![turn("user", "I'd prefer to set up the API cache responses by trying to keep it lightweight and simple, without compromising latency.")],
                },
            ],
            queries: vec![Query {
                text: "How should I set up the API cache responses? I'd like to keep it simple and lightweight.".into(),
                date: String::new(),
            }],
            top_k: 2,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.preference);
        assert_eq!(ranked.results[0].sessions[0].id, "debug");
        assert_eq!(ranked.results[0].turns[0].session_id, "debug");
    }
}

#[cfg(test)]
mod recall_cue_tests {
    use super::*;

    fn t(id: &str, role: &str, content: &str) -> Turn {
        Turn { id: id.into(), role: role.into(), content: content.into(), ..Turn::default() }
    }
    fn s(id: &str, turns: Vec<Turn>) -> Session {
        Session { id: id.into(), date: String::new(), timestamp: 0, turns }
    }
    fn ix_of(sessions: &[Session]) -> Index {
        build_index(sessions, &Params::default())
    }
    fn idx(ix: &Index, id: &str) -> usize {
        ix.turns.iter().position(|x| x.id == id).unwrap()
    }
    fn ids(ix: &Index, list: &[usize]) -> Vec<String> {
        list.iter().map(|&i| ix.turns[i].id.clone()).collect()
    }
    fn haystack() -> Vec<Session> {
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

    #[test]
    fn primacy_moves_opener_that_covers_rank1_cues() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a1"), idx(&ix, "a0")];
        assert!(boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "a2", "a1"]);
    }

    #[test]
    fn primacy_holds_when_opener_lacks_a_rank1_cue() {
        let ix = ix_of(&haystack());
        // rank 1 (a2) matches "battery"; the opener a0 does not carry it.
        let qc = query_cues(&ix, "How is the Sony camera battery?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a0"), idx(&ix, "a1")];
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "a0", "a1"]);
    }

    #[test]
    fn primacy_never_crosses_sessions() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        // rank 1 is in session b; a0 is an opener that covers, but of another session.
        let mut list = vec![idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "a2")];
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b1", "a0", "a2"]);
    }

    #[test]
    fn primacy_respects_top5_window() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (a0, a1, a2, b0, b1) = (idx(&ix, "a0"), idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "b1"));
        // a0 sits at slot 6, outside the window (a1 repeated only to pad the list).
        let mut list = vec![a2, a1, b0, b1, a1, a0];
        let before = list.clone();
        assert!(!boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn speaker_name_tokens_are_not_cues() {
        let ix = ix_of(&[s("d", vec![
            t("d0", "Caroline", "Hey, I went hiking at the lake yesterday."),
            t("d1", "Melanie", "Caroline that sounds fun, Caroline loves the lake."),
        ])]);
        let qc = query_cues(&ix, "What did Caroline do at the lake?");
        let caroline = ix.cue_vocab.get(&cue_tokens("Caroline")[0]).copied().unwrap();
        assert!(!qc.contains(&caroline));
        // Without the name token, the opener d0 covers d1's cue set and moves up.
        let mut list = vec![idx(&ix, "d1"), idx(&ix, "d0")];
        assert!(boundary_primacy(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["d0", "d1"]);
    }

    #[test]
    fn primacy_is_deterministic_under_shuffled_sessions() {
        let h = haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "Which Sony camera do I have?");
            let mut list = vec![idx(&ix, "a2"), idx(&ix, "b1"), idx(&ix, "a0")];
            boundary_primacy(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["a0", "a2", "b1"]);
        }
    }

    fn chat() -> Vec<Session> {
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

    #[test]
    fn given_new_picks_earliest_covering_turn() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        let (c1, c2, c3) = (idx(&ix, "c1"), idx(&ix, "c2"), idx(&ix, "c3"));
        let mut list = vec![c3, c2, c1];
        assert!(given_new(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c1", "c3", "c2"]);
    }

    #[test]
    fn given_new_does_not_reach_past_top10() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        let (c0, c1, c2, c3, e0) = (idx(&ix, "c0"), idx(&ix, "c1"), idx(&ix, "c2"), idx(&ix, "c3"), idx(&ix, "e0"));
        // c1 at slot 11 (padding repeats only fill the list).
        let mut list = vec![c3, c2, c0, e0, c2, c0, e0, c2, c0, e0, c1];
        let before = list.clone();
        assert!(!given_new(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn given_new_never_crosses_sessions() {
        let ix = ix_of(&chat());
        let qc = query_cues(&ix, "When is my Lisbon flight?");
        // rank 1 is e0 (session e, position 0). c1 covers its cues but lives in session c.
        let mut list = vec![idx(&ix, "e0"), idx(&ix, "c1")];
        assert!(!given_new(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["e0", "c1"]);
    }

    #[test]
    fn given_new_is_deterministic_under_shuffled_sessions() {
        let h = chat();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "When is my Lisbon flight?");
            let mut list = vec![idx(&ix, "c3"), idx(&ix, "e0"), idx(&ix, "c1")];
            given_new(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["c1", "c3", "e0"]);
        }
    }

    #[test]
    fn given_new_role_gate_matches_variant() {
        let ix = ix_of(&[s("n", vec![
            t("n0", "Caroline", "I adopted a dog named Biscuit."),
            t("n1", "Melanie", "Biscuit the dog is adorable."),
        ])]);
        let qc = query_cues(&ix, "Biscuit dog?");
        let mut list = vec![idx(&ix, "n1"), idx(&ix, "n0")];
        let moved = given_new(&ix, &qc, &mut list);
        assert_eq!(moved, !GIVEN_NEW_CHAT_ROLES_ONLY);
    }

    fn haystack_two_openers() -> Vec<Session> {
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

    #[test]
    fn session_opener_lift_moves_cross_session_opener() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        // rank 1 is b1, not an opener. a0 opens a different session, sits in the window, and covers.
        let mut list = vec![idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "a2")];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1", "a2"]);
    }

    #[test]
    fn session_opener_lift_holds_when_opener_lacks_a_rank1_cue() {
        let ix = ix_of(&haystack());
        // rank 1 (a2) matches "battery"; openers a0 and b0 do not carry it.
        let qc = query_cues(&ix, "How is the Sony camera battery?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "a0"), idx(&ix, "b0")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "a0", "b0"]);
    }

    #[test]
    fn session_opener_lift_role_gate() {
        let ix = ix_of(&[s("n", vec![
            t("n0", "Caroline", "I adopted a dog named Biscuit."),
            t("n1", "Melanie", "Biscuit the dog is adorable."),
        ]), s("p", vec![
            t("p0", "user", "Biscuit the dog lives here."),
        ])]);
        let qc = query_cues(&ix, "Biscuit dog?");
        let mut list = vec![idx(&ix, "n1"), idx(&ix, "p0")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["n1", "p0"]);
    }

    #[test]
    fn session_opener_lift_skips_when_rank1_is_an_opener() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let mut list = vec![idx(&ix, "a0"), idx(&ix, "b1")];
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1"]);
    }

    #[test]
    fn session_opener_lift_respects_top5_window() {
        let ix = ix_of(&haystack());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (a0, a1, a2, b0, b1) = (idx(&ix, "a0"), idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "b1"));
        // a0 sits at slot 6, outside the window (a1 repeated only to pad the list).
        let mut list = vec![b1, a1, a2, b0, a1, a0];
        let before = list.clone();
        assert!(!session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn session_opener_lift_picks_first_covering_opener_in_rank_order() {
        let ix = ix_of(&haystack_two_openers());
        let qc = query_cues(&ix, "Which Sony camera do I have?");
        let (b1, a0, c0) = (idx(&ix, "b1"), idx(&ix, "a0"), idx(&ix, "c0"));
        // a0 and c0 both open a session and cover b1's cues; the earlier rank wins.
        let mut list = vec![b1, a0, c0];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b1", "c0"]);
        let mut list = vec![b1, c0, a0];
        assert!(session_opener_lift(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c0", "b1", "a0"]);
    }

    #[test]
    fn session_opener_lift_is_deterministic_under_shuffled_sessions() {
        let h = haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "Which Sony camera do I have?");
            let mut list = vec![idx(&ix, "b1"), idx(&ix, "a2"), idx(&ix, "a0")];
            session_opener_lift(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["a0", "b1", "a2"]);
        }
    }

    fn hearsay_haystack() -> Vec<Session> {
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

    #[test]
    fn hearsay_moves_covering_user_turn_from_any_session() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        // rank 1 is assistant a1. b0 is a user turn in another session, sits in the window, and covers.
        let mut list = vec![idx(&ix, "a1"), idx(&ix, "b0"), idx(&ix, "a0")];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b0", "a1", "a0"]);
    }

    #[test]
    fn hearsay_holds_when_user_turn_lacks_a_rank1_cue() {
        let ix = ix_of(&hearsay_haystack());
        // a1 matches Lisbon March; a0 also covers those, but b0-less candidate: use a2 as rank 1
        // with a cue ("hotels") that no user turn carries.
        let qc = query_cues(&ix, "Can you look at hotels in Lisbon in March?");
        let mut list = vec![idx(&ix, "a2"), idx(&ix, "b0"), idx(&ix, "a0")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a2", "b0", "a0"]);
    }

    #[test]
    fn hearsay_skips_when_rank1_is_not_assistant() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let mut list = vec![idx(&ix, "a0"), idx(&ix, "b0")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a0", "b0"]);
    }

    #[test]
    fn hearsay_skips_non_user_candidates() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        // b1 is an assistant turn that covers; it must not move.
        let mut list = vec![idx(&ix, "a1"), idx(&ix, "b1"), idx(&ix, "a2")];
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["a1", "b1", "a2"]);
    }

    #[test]
    fn hearsay_respects_top5_window() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let (a1, a2, b1, b0) = (idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b1"), idx(&ix, "b0"));
        // b0 sits at slot 6, outside the window (a2/b1 repeated only to pad the list).
        let mut list = vec![a1, a2, b1, a2, b1, b0];
        let before = list.clone();
        assert!(!hearsay(&ix, &qc, &mut list));
        assert_eq!(list, before);
    }

    #[test]
    fn hearsay_picks_first_covering_user_in_rank_order() {
        let ix = ix_of(&hearsay_haystack());
        let qc = query_cues(&ix, "When is the Lisbon trip in March?");
        let (a1, b0, c0) = (idx(&ix, "a1"), idx(&ix, "b0"), idx(&ix, "c0"));
        let mut list = vec![a1, b0, c0];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["b0", "a1", "c0"]);
        let mut list = vec![a1, c0, b0];
        assert!(hearsay(&ix, &qc, &mut list));
        assert_eq!(ids(&ix, &list), vec!["c0", "a1", "b0"]);
    }

    #[test]
    fn hearsay_is_deterministic_under_shuffled_sessions() {
        let h = hearsay_haystack();
        let mut rev = h.clone();
        rev.reverse();
        for hs in [h, rev] {
            let ix = ix_of(&hs);
            let qc = query_cues(&ix, "When is the Lisbon trip in March?");
            let mut list = vec![idx(&ix, "a1"), idx(&ix, "a2"), idx(&ix, "b0")];
            hearsay(&ix, &qc, &mut list);
            assert_eq!(ids(&ix, &list), vec!["b0", "a1", "a2"]);
        }
    }

    #[test]
    fn salience_join_writes_a_backward_edge_from_a_later_turn_to_an_earlier_policy() {
        let ix = ix_of(&[s("s", vec![
            t("rule", "user", "Always write the answer in markdown tables."),
            t("later", "user", "Please summarize today's notes."),
        ])]);
        let rule = idx(&ix, "rule");
        let later = idx(&ix, "later");
        assert!(ix.turns[rule].is_standing_policy, "the earlier turn should be a standing policy");
        assert!(
            ix.turns[later].policy_backlinks.contains(&rule),
            "SalienceJoin should record a backward later->policy link, got {:?}",
            ix.turns[later].policy_backlinks
        );
        assert!(
            !ix.turns[rule].policy_backlinks.contains(&later),
            "must not write a forward link"
        );
        assert!(
            !ix.turns[later].exact_edges.iter().any(|&(_, target)| target == rule),
            "v2 must not put the policy link on the general exact-edge list"
        );
    }

    #[test]
    fn salience_join_seats_a_fired_policy_when_a_later_turn_is_retrieved() {
        let sessions = vec![
            s("conv_s0", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
                t("ack", "assistant", "I will follow that."),
            ]),
            s("conv_s1", vec![
                t("later", "user", "Could you outline a backup routine?"),
                t("reply", "assistant", "Here is the outline."),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "Could you outline a backup routine?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(! ranked.results[0].intent.task_under_rules);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let later_at = ids.iter().position(|id| id == "later").expect("symptom retrieved");
        let rule_at = ids.iter().position(|id| id == "rule").expect(&format!("cause missing, got {ids:?}"));
        assert!(rule_at > later_at, "cause sits behind the symptom, got {ids:?}");
    }

    #[test]
    fn salience_join_raises_a_policy_that_shares_no_words_with_the_query() {
        let sessions = vec![
            s("conv_s0", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
            ]),
            s("conv_s1", vec![
                t("later", "user", "When did the soup boil?"),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query { text: "When did the soup boil?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        assert!(!ranked.results[0].intent.task_under_rules);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let later_at = ids.iter().position(|id| id == "later").expect("symptom retrieved");
        let rule_at = ids.iter().position(|id| id == "rule").expect(&format!("zero-overlap cause missing, got {ids:?}"));
        assert!(rule_at > later_at, "cause sits behind the symptom, got {ids:?}");
    }

    #[test]
    fn salience_join_does_not_seat_a_policy_from_another_episode_on_a_temporal_question() {
        let sessions = vec![
            s("batch_a", vec![
                t("gold", "user", "I reached 1200 in the emergency fund on June 5."),
                t("reply", "assistant", "Reaching 1200 in the emergency fund on June 5 is a milestone."),
            ]),
            s("batch_b", vec![
                t("rule", "user", "Always write the answer in markdown tables."),
            ]),
        ];
        let req = RankRequest {
            sessions,
            queries: vec![Query {
                text: "How long after June 5 did the emergency fund get to 2000?".into(),
                date: String::new(),
            }],
            top_k: 5,
            params: Params::default(),
        };
        let ranked = rank(&req);
        let ids: Vec<String> = ranked.results[0].turns.iter().map(|x| x.id.clone()).collect();
        assert!(
            !ids.contains(&"rule".to_string()),
            "a temporal question must not import a policy from another episode, got {ids:?}"
        );
    }

    #[test]
    fn kemeny_puts_the_condorcet_winner_first() {
        let net = vec![
            vec![0, 1, 1],
            vec![-1, 0, 1],
            vec![-1, -1, 0],
        ];
        assert_eq!(kemeny_young(&net, &[0, 1, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn kemeny_branch_and_bound_matches_brute_force() {
        let net = vec![
            vec![0, 2, -1, 1],
            vec![-2, 0, 3, -1],
            vec![1, -3, 0, 2],
            vec![-1, 1, -2, 0],
        ];
        let rsb = [0, 1, 2, 3];
        let solved = kemeny_branch_and_bound(&[0, 1, 2, 3], &net, &rsb);
        let mut best = i64::MIN;
        let mut best_order = Vec::new();
        let mut perm = vec![0usize, 1, 2, 3];
        fn walk(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: &mut i64, best_order: &mut Vec<usize>) {
            if i == perm.len() {
                let score = kemeny_pair_score(perm, net);
                if score > *best {
                    *best = score;
                    *best_order = perm.to_vec();
                }
                return;
            }
            for j in i..perm.len() {
                perm.swap(i, j);
                walk(perm, i + 1, net, best, best_order);
                perm.swap(i, j);
            }
        }
        walk(&mut perm, 0, &net, &mut best, &mut best_order);
        assert_eq!(kemeny_pair_score(&solved, &net), best);
        let solved_key: Vec<usize> = solved.iter().copied().collect();
        let mut lex_best = solved_key.clone();
        let mut perm = vec![0usize, 1, 2, 3];
        fn walk_lex(perm: &mut [usize], i: usize, net: &[Vec<i64>], best: i64, lex_best: &mut Vec<usize>) {
            if i == perm.len() {
                if kemeny_pair_score(perm, net) == best && perm < lex_best {
                    *lex_best = perm.to_vec();
                }
                return;
            }
            for j in i..perm.len() {
                perm.swap(i, j);
                walk_lex(perm, i + 1, net, best, lex_best);
                perm.swap(i, j);
            }
        }
        walk_lex(&mut perm, 0, &net, best, &mut lex_best);
        assert_eq!(solved_key, lex_best);
    }

    #[test]
    fn kemeny_tie_keeps_the_rsb_order() {
        let net = vec![vec![0i64; 4]; 4];
        assert_eq!(kemeny_young(&net, &[0, 1, 2, 3]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn gatekeeper_demotes_a_superseded_rank_one() {
        let sessions = vec![s("s", vec![
            t("old", "user", "The camera is a Sony A7."),
            t("fresh", "user", "The camera is a Sony A7C."),
        ])];
        let mut ix = ix_of(&sessions);
        let old = idx(&ix, "old");
        let fresh = idx(&ix, "fresh");
        ix.turns[fresh].supersedes_turns.push(old);
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[old] = 3.0;
        scene[fresh] = 1.0;
        let mut list = vec![old, fresh];
        rank1_gatekeeper(&ix, "which camera", &mut list, &scene);
        assert_eq!(list[0], fresh, "a superseded turn cannot hold rank 1");
    }

    #[test]
    fn gatekeeper_when_query_prefers_a_clock_form() {
        let sessions = vec![s("s", vec![
            t("plain", "user", "We met downtown after the show."),
            t("clock", "user", "We met at 18:40 after the show."),
        ])];
        let ix = ix_of(&sessions);
        let plain = idx(&ix, "plain");
        let clock = idx(&ix, "clock");
        assert!(ix.turns[clock].form_clock);
        assert!(!ix.turns[plain].form_clock);
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![plain, clock];
        rank1_gatekeeper(&ix, "When did we meet?", &mut list, &scene);
        assert_eq!(list[0], clock);
    }

    #[test]
    fn gatekeeper_v2_keeps_the_rsb_head_and_reorders_the_tail() {
        let sessions = vec![s("s", vec![
            t("plain", "user", "We met downtown after the show."),
            t("other", "user", "We met downtown after the show again."),
            t("clock", "user", "We met at 18:40 after the show."),
        ])];
        let ix = ix_of(&sessions);
        let plain = idx(&ix, "plain");
        let other = idx(&ix, "other");
        let clock = idx(&ix, "clock");
        let scene = vec![1.0f32; ix.turns.len()];
        let mut list = vec![plain, other, clock];
        rank1_gatekeeper_keep_rsb_head(&ix, "When did we meet?", &mut list, &scene);
        assert_eq!(list, vec![plain, clock, other]);
    }

    #[test]
    fn gatekeeper_v2_does_not_demote_the_rsb_head() {
        let sessions = vec![s("s", vec![
            t("head", "user", "The camera is a Sony A7."),
            t("old", "user", "The lens is a fifty."),
            t("fresh", "user", "The lens is a thirty five."),
        ])];
        let mut ix = ix_of(&sessions);
        let head = idx(&ix, "head");
        let old = idx(&ix, "old");
        let fresh = idx(&ix, "fresh");
        ix.turns[fresh].supersedes_turns.push(head);
        ix.turns[fresh].supersedes_turns.push(old);
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[head] = 1.0;
        scene[old] = 3.0;
        scene[fresh] = 1.0;
        let mut list = vec![head, old, fresh];
        rank1_gatekeeper_keep_rsb_head(&ix, "which camera", &mut list, &scene);
        assert_eq!(list[0], head, "slot 1 stays the RSB head");
        assert_eq!(list[1], fresh, "ATMS still demotes a superseded turn inside the tail");
    }

    #[test]
    fn rank_is_identical_across_repeated_calls() {
        let mk = || RankRequest {
            sessions: haystack(),
            queries: vec![Query { text: "Which Sony camera do I have?".into(), date: String::new() }],
            top_k: 5,
            params: Params::default(),
        };
        let a = rank(&mk());
        let b = rank(&mk());
        let ta: Vec<String> = a.results[0].turns.iter().map(|x| x.id.clone()).collect();
        let tb: Vec<String> = b.results[0].turns.iter().map(|x| x.id.clone()).collect();
        assert_eq!(ta, tb);
    }
}

// ---- moved verbatim from former `src/steps7_10.rs` (was `include!()`d into retrieval.rs) ----

#[cfg(test)]
mod steps7_10_tests {
    use super::*;

    #[test]
    fn step7_proper_surfaces_skip_sentence_start() {
        let p = step7_query_proper_surfaces("What did Melanie's friend Jon say? Where is Rome");
        assert!(p.contains("melanie") && p.contains("jon"));
        assert!(!p.contains("where") && p.contains("rome"));
        assert!(!p.contains("what"));
    }

    #[test]
    fn step9_typo_and_ingest_shapes() {
        assert!(damerau_lev_1(b"recieve", b"receive"));
        assert!(damerau_lev_1(b"colour", b"color"));
        assert!(!damerau_lev_1(b"same", b"same"));
        assert_eq!(step9_list_item_count("Steps:\n1. a\n2) b\n- c\nend"), 3);
        assert_eq!(step9_first_move_num("Game so far:\n14. Nf3 e5\n15. Bb5"), 14);
        assert_eq!(step9_first_move_num("1. Buy milk"), 0);
    }

    #[test]
    fn step10_florilegium_relaxes_floors_for_multi_session() {
        assert_eq!(step10_florilegium_floors(2), (0.38, 0.20, false));
        assert_eq!(step10_florilegium_floors(1), (0.45, 0.25, true));
    }
}

// ---- moved verbatim from former `src/algos_c2.rs` (was `include!()`d into retrieval.rs) ----


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
        assert!(! plan.intent.preference);
        assert!(! plan.intent.assistant_attribution);
        assert!(!c2_retrospective_attribution(&q.text));
        // Assistant at slot 1; previous user already in the top 5.
        let mut list = vec![1usize, 2, 0, 3];
        // "can you recommend" is not a preference route anymore.
        assert!(!apply_c2_preference_source_seat(&ix, &plan, &q.text, &mut list));
        assert_eq!(list, vec![1, 2, 0, 3]);
        // Previous user not already in the top 5: membership must not change.
        let mut list = vec![1usize, 2, 3];
        assert!(!apply_c2_preference_source_seat(&ix, &plan, &q.text, &mut list));
        assert_eq!(list, vec![1, 2, 3]);
        // Retrospective "remind me / you recommended": no swap (SSA path).
        let rq = Query { text: "Can you remind me of the resources you recommended for video editing?".into(), date: String::new() };
        let rplan = plan_query(&ix, &rq, &Params::default());
        assert!(!c2_retrospective_attribution(&rq.text));
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
        assert!(! pplan.intent.preference);
        let mut list = vec![2usize, 3, 4, 5, 6];
        assert!(apply_c2_elaboration_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_summary_request(&sq.text));
        let splan = plan_query(&ix, &sq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3, 4];
        assert!(apply_c2_cross_opener_seat(&ix, &splan, &sq.text, &mut list, &t_scene));
        // Preference query: no seat.
        let pq = Query { text: "Can you recommend a cocktail recipe?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(! pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3, 4];
        assert!(apply_c2_cross_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_summary_request(&sq.text));
        let splan = plan_query(&ix, &sq, &Params::default());
        let mut list = vec![2usize, 0, 1, 3];
        assert!(apply_c2_same_opener_seat(&ix, &splan, &sq.text, &mut list, &t_scene));
        // Preference query: no seat.
        let pq = Query { text: "Can you recommend a cheaper commute?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(! pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3];
        assert!(apply_c2_same_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! pplan.intent.preference);
        let mut list = vec![0usize, 1, 2];
        assert!(apply_c2_queried_speaker_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! pplan.intent.preference);
        let mut list = vec![1usize, 0, 2];
        assert!(apply_c2_queried_speaker_prev_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! pplan.intent.preference);
        let mut list = vec![0usize, 1, 3, 4, 5, 2];
        assert!(apply_c2_queried_speaker_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));

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
        assert!(! pplan.intent.preference);
        let mut list = vec![1usize, 2, 0];
        assert!(apply_c2_queried_speaker_qhead_prev_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));

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
        assert!(! c2_text_preference(
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
        assert!(! pplan.intent.preference);
        let mut list = vec![1usize, 0];
        assert!(apply_c2_queried_speaker_first_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));

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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 3, 2, 4];
        assert!(apply_c2_cross_opener_pair_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 2, 1, 3];
        assert!(apply_c2_answered_opener_decl_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 2, 3];
        assert!(apply_c2_solo_earlier_opener_seat(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 2, 3, 5, 6, 8, 4, 7, 9];
        assert!(apply_c2_pool_opener_t10_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 4, 5, 3, 2];
        assert!(apply_c2_same_session_opener_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! c2_text_preference(&pq.text));
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(apply_c2_pool_mentioned_t10_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));
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
        assert!(! pplan.intent.preference);
        let mut list = vec![2usize, 0, 1, 3, 5, 4];
        assert!(apply_c2_queried_speaker_first_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));

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
            text: "Where did I go on a long trip with my family?".into(),
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

        // "week" makes this temporal, so the seat does not enter.
        let pq = Query { text: "What is my favorite week-long trip with my family?".into(), date: String::new() };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(! pplan.intent.preference);
        assert!(pplan.intent.temporal);
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
        assert!(! pplan.intent.preference);
        let mut list = vec![0usize, 1, 3, 4, 7, 5];
        assert!(apply_c2_cross_session_user_t5_entry(&ix, &pplan, &pq.text, &mut list, &t_scene));

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
        assert!(! plan.intent.assistant_attribution);
        assert!(!plan.intent.preference);
        assert!(!plan.intent.temporal);
        let mut list = vec![5usize, 3, 4, 2, 1, 0];
        let scene = vec![0.4f32, 0.9, 0.3, 0.8, 0.5, 1.0];
        // "did you say" no longer sets assistant attribution, so the split does not fire.
        assert!(!apply_c2_step31_exchequer_split(&ix, &plan, &q.text, &mut list, &scene));
        assert_eq!(list, vec![5, 3, 4, 2, 1, 0]);

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
        assert!(! pplan.intent.preference);
        let mut list = vec![5usize, 3, 4, 2, 1, 0];
        assert!(!apply_c2_step31_exchequer_split(&ix, &pplan, &pq.text, &mut list, &scene));

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

    #[test]
    fn step37_kalevala_seats_the_richer_parallel_line() {
        let q = "What artist did I start to listen to?";
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I am also thinking of exploring more genres and discovering new artists."),
                    ("assistant", "Try a different style."),
                    ("user", "I am also interested in exploring different genres and discovering new artists. I started a bluegrass band today."),
                ],
            )],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 2];
        assert!(apply_c2_step37_kalevala(&ix, q, &mut list));
        assert_eq!(list, vec![2, 0, 1]);
        // The richer line is past the top 5: the seat stays, even though the
        // two user turns are still one line in the session.
        let ix2 = build_index(
            &[
                sess(
                    "s",
                    &[
                        ("user", "I am also thinking of exploring more genres and discovering new artists."),
                        ("assistant", "Try a different style."),
                        ("user", "I am also interested in exploring different genres and discovering new artists. I started a bluegrass band today."),
                    ],
                ),
                sess(
                    "o",
                    &[
                        ("user", "Cedar ledger."),
                        ("user", "Quartz violin."),
                        ("user", "Maple desk."),
                        ("user", "Birch shelf."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let mut far = vec![0usize, 3, 4, 5, 6, 2];
        assert!(!apply_c2_step37_kalevala(&ix2, q, &mut far));
        assert_eq!(far, vec![0, 3, 4, 5, 6, 2]);

        // Numeral shape disagrees, so the lines are not one line.
        let ixd = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I am also thinking of exploring 3 genres and discovering new artists."),
                    ("assistant", "Try a different style."),
                    ("user", "I am also interested in exploring different genres and discovering new artists. I started a bluegrass band today."),
                ],
            )],
            &Params::default(),
        );
        let mut digit = vec![0usize, 1, 2];
        assert!(!apply_c2_step37_kalevala(&ixd, q, &mut digit));
        assert_eq!(digit, vec![0, 1, 2]);

        // No shared restatement: nothing moves.
        let ixu = build_index(
            &[sess(
                "s",
                &[
                    ("user", "The ledger key is cedar."),
                    ("assistant", "Noted."),
                    ("user", "I started a bluegrass band today."),
                ],
            )],
            &Params::default(),
        );
        let mut apart = vec![0usize, 1, 2];
        assert!(!apply_c2_step37_kalevala(&ixu, q, &mut apart));
        assert_eq!(apart, vec![0, 1, 2]);

        // The earlier line can carry more content cues and still not take the seat.
        let mut earlier = vec![2usize, 1, 0];
        assert!(!apply_c2_step37_kalevala(&ix, q, &mut earlier));
        assert_eq!(earlier, vec![2, 1, 0]);

        // A following line that only adds a closed-class cue is not richer.
        let ixw = build_index(
            &[sess(
                "s",
                &[
                    ("user", "I flew with Southwest in March."),
                    ("assistant", "Noted."),
                    ("user", "Which flight with Southwest in March was direct?"),
                ],
            )],
            &Params::default(),
        );
        let mut closed = vec![0usize, 1, 2];
        assert!(!apply_c2_step37_kalevala(
            &ixw,
            "Which airline did I fly in March?",
            &mut closed,
        ));
        assert_eq!(closed, vec![0, 1, 2]);
    }

    #[test]
    fn hermeneutic_circle_seats_the_part_its_session_completes() {
        let ix = build_index(
            &[
                sess(
                    "D26",
                    &[
                        ("james", "Have you heard about Cyberpunk 2077? Do you think this game is worthy?"),
                        ("john", "James already played it yesterday."),
                    ],
                ),
                sess(
                    "D28",
                    &[
                        ("john", "James should try the setup before buying."),
                        ("james", "Yesterday I tried Cyberpunk 2077. Great game, so addictive."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = "When did James try Cyberpunk 2077 game?";
        assert!(ix.turns[0].is_question);
        assert!(!ix.turns[3].is_question);
        let anchors = step36_anchors(&ix, q);
        let posts = |ti: usize, ids: &[u32]| step36_posts(&ix.turns[ti], ids);
        let hit = |ti: usize| -> Vec<bool> { anchors.iter().map(|ids| posts(ti, ids)).collect() };
        assert_eq!(hit(0), hit(3), "the two parts post the same anchors");
        assert!(hit(0).iter().any(|h| *h) && hit(0).iter().any(|h| !*h));
        // "try" is in the completing session and absent from the question's session.
        let try_ids = anchors.iter().find(|ids| {
            !posts(0, ids) && !posts(1, ids) && posts(2, ids)
        });
        assert!(try_ids.is_some(), "the rest anchor lives only in the other session");
        let t_own = vec![1.0f32, 0.2, 0.4, 2.0];
        let mut list = vec![0usize, 1, 3, 2];
        let before: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert!(apply_c2_step38_hermeneutic(&ix, q, &mut list, &t_own));
        assert_eq!(list[0], 3);
        let after: std::collections::HashSet<usize> = list.iter().copied().collect();
        assert_eq!(before, after);

        // A weaker own score is not the part the whole confirms.
        let mut list = vec![0usize, 1, 3, 2];
        assert!(!apply_c2_step38_hermeneutic(&ix, q, &mut list, &[1.0, 0.2, 0.4, 0.5]));
        assert_eq!(list, vec![0, 1, 3, 2]);

        // The question's own session already covers every rest anchor.
        let covered = build_index(
            &[
                sess(
                    "D26",
                    &[
                        ("james", "Have you heard about Cyberpunk 2077? Do you think this game is worthy?"),
                        ("john", "James should try it."),
                    ],
                ),
                sess(
                    "D28",
                    &[
                        ("john", "James should try the setup before buying."),
                        ("james", "Yesterday I tried Cyberpunk 2077. Great game, so addictive."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 3, 2];
        assert!(!apply_c2_step38_hermeneutic(&covered, q, &mut list, &t_own));
        assert_eq!(list[0], 0);

        // A different part (the turn itself posts the rest) is not a near-tie.
        let mut list = vec![0usize, 2];
        assert!(!apply_c2_step38_hermeneutic(&ix, q, &mut list, &[1.0, 0.2, 3.0, 2.0]));
        assert_eq!(list, vec![0, 2]);

        // Two completing sessions are not one reading.
        let tied = build_index(
            &[
                sess(
                    "D26",
                    &[("james", "Have you heard about Cyberpunk 2077? Do you think this game is worthy?")],
                ),
                sess(
                    "D28",
                    &[
                        ("john", "James should try the setup."),
                        ("james", "Yesterday I tried Cyberpunk 2077. Great game."),
                    ],
                ),
                sess(
                    "D30",
                    &[
                        ("john", "James should try the other copy."),
                        ("james", "I tried Cyberpunk 2077. Great game."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let mut list = vec![0usize, 2, 4];
        assert!(!apply_c2_step38_hermeneutic(&tied, q, &mut list, &[1.0, 0.1, 2.0, 0.1, 3.0]));
        assert_eq!(list, vec![0, 2, 4]);

        // A declarative incumbent is already a statement. It does not yield.
        let mut list = vec![3usize, 0, 2, 1];
        assert!(!apply_c2_step38_hermeneutic(&ix, q, &mut list, &[0.2, 0.1, 0.4, 4.0]));
        assert_eq!(list[0], 3);
    }

    #[test]
    fn pool_c_session_opener_replaces_same_session_assistant_echo() {
        let ix = build_index(
            &[
                sess(
                    "sun",
                    &[
                        ("user", "I baked rye loaves for the picnic on Sunday."),
                        ("assistant", "You baked rye loaves for the picnic on Sunday."),
                        ("user", "The picnic rye loaves were a hit on Sunday."),
                    ],
                ),
                sess(
                    "mon",
                    &[
                        ("user", "I baked rye loaves for the picnic on Monday."),
                        ("assistant", "You baked rye loaves for the picnic on Monday."),
                    ],
                ),
                sess(
                    "pad",
                    &[
                        ("user", "The weather was fine all weekend."),
                        ("assistant", "Glad the weather was fine."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What did I bake for the picnic?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(!plan.groups.is_empty());
        assert!(!plan.intent.assistant_attribution);
        assert!(!plan.intent.preference);
        // 0 sun opener, 1 sun assistant, 2 sun later, 3 mon opener, 4 mon assistant, 5 pad user, 6 pad assistant
        let mut list = vec![2usize, 1, 5, 4, 6];
        let mut scene = vec![0.0f32; ix.turns.len()];
        scene[2] = 9.0;
        scene[1] = 8.0;
        scene[5] = 4.0;
        scene[4] = 3.0;
        scene[6] = 2.0;
        scene[0] = 1.5;
        scene[3] = 1.2;
        assert!(apply_c2_pool_c_session_opener_seat(&ix, &plan, &q.text, &mut list, &scene));
        assert_eq!(list[0], 2, "slot 0 stays");
        assert!(list[..5].contains(&0), "sunday opener enters the top 5");
        assert!(list[..5].contains(&3), "monday opener enters the top 5");
        assert!(!list[..5].contains(&1));
        assert!(!list[..5].contains(&4));

        // Numeral only on the assistant: the elaboration keeps the seat.
        let ix_n = build_index(
            &[sess(
                "sun",
                &[
                    ("user", "I baked rye loaves for the picnic."),
                    ("assistant", "You baked 4 rye loaves for the picnic."),
                    ("user", "Noted."),
                    ("assistant", "Ok."),
                    ("user", "Thanks."),
                ],
            )],
            &Params::default(),
        );
        let plan_n = plan_query(&ix_n, &q, &Params::default());
        assert!(ix_n.turns[1].form_digit);
        assert!(!ix_n.turns[0].form_digit);
        let mut listed = vec![2usize, 1, 4, 2, 4];
        let mut scene_n = vec![1.0f32; ix_n.turns.len()];
        scene_n[0] = 0.4;
        assert!(!apply_c2_pool_c_session_opener_seat(
            &ix_n, &plan_n, &q.text, &mut listed, &scene_n
        ));

        // One session in the window. A neighbour may post the term; the seat
        // still does not fire. The elaboration is the only session's answer.
        let ix_u = build_index(
            &[sess(
                "sun",
                &[
                    ("user", "I baked rye loaves for the picnic."),
                    ("assistant", "The designation on the rye loaves was LIV."),
                    ("user", "The designation was on the wrapper."),
                    ("assistant", "Noted."),
                    ("user", "Thanks."),
                ],
            )],
            &Params::default(),
        );
        let uq = Query {
            text: "What designation was on the rye loaves?".into(),
            date: String::new(),
        };
        let plan_u = plan_query(&ix_u, &uq, &Params::default());
        let mut ulist = vec![2usize, 1, 4, 2, 4];
        let mut uscene = vec![1.0f32; ix_u.turns.len()];
        uscene[0] = 0.5;
        assert!(!apply_c2_pool_c_session_opener_seat(
            &ix_u, &plan_u, &uq.text, &mut ulist, &uscene
        ));
        assert_eq!(ulist[1], 1);

        // Two sessions, and a staying turn posts the term the opener lacks.
        // The elaboration is an echo; the session key takes the seat.
        let ix_m = build_index(
            &[
                sess(
                    "sun",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "The designation on the rye loaves was LIV."),
                        ("user", "The designation was on the wrapper."),
                    ],
                ),
                sess(
                    "mon",
                    &[
                        ("user", "I baked rye loaves for the picnic on Monday."),
                        ("assistant", "You baked rye loaves for the picnic on Monday."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let plan_m = plan_query(&ix_m, &uq, &Params::default());
        // 0 sun opener, 1 sun assistant, 2 sun user (posts designation), 3 mon opener, 4 mon assistant
        let mut mlist = vec![3usize, 1, 2, 4, 3];
        let mut mscene = vec![1.0f32; ix_m.turns.len()];
        mscene[0] = 2.0;
        mscene[3] = 0.4;
        assert!(apply_c2_pool_c_session_opener_seat(
            &ix_m, &plan_m, &uq.text, &mut mlist, &mscene
        ));
        assert!(mlist[..5].contains(&0), "opener enters when a staying turn covers the term");
        assert!(!mlist[..5].contains(&1), "echo assistant leaves");

        // Two sessions, and nobody but the assistant posts the term. Keep it.
        let ix_k = build_index(
            &[
                sess(
                    "sun",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "The designation on the rye loaves was LIV."),
                        ("user", "The wrapper was plain."),
                    ],
                ),
                sess(
                    "mon",
                    &[
                        ("user", "I baked rye loaves for the picnic on Monday."),
                        ("assistant", "You baked rye loaves for the picnic on Monday."),
                    ],
                ),
                sess(
                    "pad",
                    &[
                        ("user", "The weather was fine all weekend."),
                        ("assistant", "Glad the weather was fine."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let plan_k = plan_query(&ix_k, &uq, &Params::default());
        // 0 sun opener, 1 sun assistant (unique term), 2 sun user, 3 mon opener,
        // 4 mon assistant, 5 pad user, 6 pad assistant.
        let mut klist = vec![5usize, 1, 2, 4, 6];
        let mut kscene = vec![1.0f32; ix_k.turns.len()];
        kscene[0] = 0.2;
        kscene[3] = 2.0;
        assert!(apply_c2_pool_c_session_opener_seat(
            &ix_k, &plan_k, &uq.text, &mut klist, &kscene
        ));
        assert!(klist[..5].contains(&1), "unique-term assistant stays");
        assert!(klist[..5].contains(&3), "monday opener still enters");
        assert!(!klist[..5].contains(&4), "monday echo leaves");

        let named = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("caroline", "I baked rye loaves for the picnic."),
                        ("melanie", "You baked rye loaves for the picnic."),
                    ],
                ),
                sess(
                    "D2",
                    &[
                        ("caroline", "The weather was fine."),
                        ("melanie", "Glad the weather was fine."),
                        ("caroline", "See you at the picnic."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let mut nlist = vec![1usize, 0, 2, 3, 4];
        let nscene = vec![1.0f32; named.turns.len()];
        assert!(!apply_c2_pool_c_session_opener_seat(
            &named, &plan, &q.text, &mut nlist, &nscene
        ));

        let pq = Query {
            text: "Can you recommend a rye loaf for the picnic?".into(),
            date: String::new(),
        };
        let pplan = plan_query(&ix, &pq, &Params::default());
        assert!(! c2_text_preference(&pq.text));
        let mut pref = vec![2usize, 1, 5, 4, 6];
        assert!(apply_c2_pool_c_session_opener_seat(&ix, &pplan, &pq.text, &mut pref, &scene));
    }

    #[test]
    fn pool_c_absent_session_key_takes_a_redundant_minority_assistant() {
        let ix = build_index(
            &[
                sess(
                    "big",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "You baked rye loaves for the picnic."),
                        ("user", "The picnic rye loaves were a hit."),
                        ("assistant", "Glad the rye loaves were a hit."),
                    ],
                ),
                sess(
                    "small",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "You baked rye loaves for the picnic."),
                    ],
                ),
                sess(
                    "away",
                    &[
                        ("user", "I baked rye loaves for the picnic on Sunday."),
                        ("assistant", "Noted about the rye loaves."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let q = Query {
            text: "What did I bake for the picnic?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        // 0 big user, 1 big assistant, 2 big user, 3 big assistant,
        // 4 small user, 5 small assistant, 6 away opener, 7 away assistant.
        let mut list = vec![0usize, 5, 4, 1, 2];
        let mut scene = vec![0.0f32; ix.turns.len()];
        for t in [0, 5, 4, 1, 2, 6] {
            scene[t] = 1.0;
        }
        scene[6] = 1.4;
        assert!(apply_c2_pool_c_absent_session_key(&ix, &plan, &q.text, &mut list, &scene));
        assert_eq!(list[0], 0, "slot 0 stays");
        assert!(list[..5].contains(&6), "absent opener enters");
        assert!(!list[..5].contains(&5), "redundant minority assistant leaves");
        assert!(list[..5].contains(&1), "dominant-session assistant stays");

        let ix_d = build_index(
            &[
                sess(
                    "big",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "You baked rye loaves for the picnic."),
                        ("user", "The picnic rye loaves were a hit."),
                    ],
                ),
                sess(
                    "small",
                    &[
                        ("user", "I baked rye loaves for the picnic."),
                        ("assistant", "You baked 4 rye loaves for the picnic."),
                    ],
                ),
                sess(
                    "away",
                    &[
                        ("user", "I baked rye loaves for the picnic on Sunday."),
                        ("assistant", "Noted."),
                    ],
                ),
            ],
            &Params::default(),
        );
        let plan_d = plan_query(&ix_d, &q, &Params::default());
        assert!(ix_d.turns[4].form_digit);
        assert!(!ix_d.turns[5].form_digit);
        let mut dlist = vec![0usize, 4, 3, 1, 2];
        let dscene = vec![1.0f32; ix_d.turns.len()];
        assert!(!apply_c2_pool_c_absent_session_key(
            &ix_d, &plan_d, &q.text, &mut dlist, &dscene
        ));
        assert_eq!(dlist, vec![0, 4, 3, 1, 2]);
    }

    #[test]
    fn pool_b_standing_seat_promotes_zero_own_opener_on_task_question() {
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[(
                        "user",
                        "Always format code with syntax highlighting when I ask about implementation.",
                    )],
                ),
                sess(
                    "s2",
                    &[(
                        "user",
                        "Could you show me how to implement a login feature?",
                    )],
                ),
            ],
            &Params::default(),
        );
        assert!(ix.turns[0].is_standing_policy);
        assert!(!ix.turns[0].is_question);
        assert!(ix.turns[1].is_question);
        assert_eq!(c2_session_pos(&ix, 0), 0);
        let q = Query {
            text: "Could you show me how to implement a login feature?".into(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        assert!(! plan.intent.task_under_rules);
        assert!(!plan.intent.temporal);
        assert!(!c2_text_preference(&q.text));
        assert!(!c2_summary_request(&q.text));
        let standing = vec![RankedItem {
            id: ix.turns[0].id.clone(),
            session_id: "s1".into(),
            score: 1.0,
            matched: Vec::new(),
        }];
        let t_own = vec![0.0f32, 4.0];
        let mut list = vec![1usize, 0];
        // No task-under-rules cue list, so the standing seat does not fire.
        assert!(!apply_c2_pool_b_standing_seat(
            &ix, &plan, &q.text, &standing, &mut list, &t_own
        ));
        assert_eq!(list, vec![1, 0]);

        // Own lexical mass: the submodular order stays.
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_pool_b_standing_seat(
            &ix,
            &plan,
            &q.text,
            &standing,
            &mut list,
            &[1.5, 4.0]
        ));
        assert_eq!(list, vec![1, 0]);

        // Episodic "when did" is not a task under stored rules.
        let tq = Query {
            text: "When did I ask about implementing a login feature?".into(),
            date: String::new(),
        };
        let tplan = plan_query(&ix, &tq, &Params::default());
        assert!(!tplan.intent.task_under_rules);
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_pool_b_standing_seat(
            &ix, &tplan, &tq.text, &standing, &mut list, &t_own
        ));
        assert_eq!(list, vec![1, 0]);

        // A recommendation question is text-level preference.
        let pq = Query {
            text: "What movies would you recommend for me to watch?".into(),
            date: String::new(),
        };
        assert!(! c2_text_preference(&pq.text));
        let pplan = plan_query(&ix, &pq, &Params::default());
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_pool_b_standing_seat(
            &ix, &pplan, &pq.text, &standing, &mut list, &t_own
        ));
        assert_eq!(list, vec![1, 0]);

        // A scoped rule whose conditional clause does not meet the query
        // stays off slot 1, even on a task-under-rules question.
        let ix = build_index(
            &[
                sess(
                    "s1",
                    &[(
                        "user",
                        "Always provide percentage improvements when I ask about editing progress.",
                    )],
                ),
                sess(
                    "s2",
                    &[(
                        "user",
                        "What deadline should I aim for to submit my peer-reviewed draft?",
                    )],
                ),
            ],
            &Params::default(),
        );
        let dq = Query {
            text: "What deadline should I aim for to submit my peer-reviewed draft?".into(),
            date: String::new(),
        };
        let dplan = plan_query(&ix, &dq, &Params::default());
        assert!(! dplan.intent.task_under_rules);
        let standing = vec![RankedItem {
            id: ix.turns[0].id.clone(),
            session_id: "s1".into(),
            score: 1.0,
            matched: Vec::new(),
        }];
        let mut list = vec![1usize, 0];
        assert!(!apply_c2_pool_b_standing_seat(
            &ix,
            &dplan,
            &dq.text,
            &standing,
            &mut list,
            &[0.0, 4.0]
        ));
        assert_eq!(list, vec![1, 0]);
    }

    #[test]
    fn buried_lexical_peak_seats_the_unique_unpooled_declarative() {
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("mel", "When did the shop open?"),
                    ("jon", "Where is the shop now?"),
                    ("mel", "The counter faces the window."),
                    ("jon", "The awning is green."),
                    ("mel", "The bell is on the door."),
                    ("jon", "The shop opened beside the river."),
                ],
            )],
            &Params::default(),
        );
        assert!(ix.turns[0].is_question);
        assert!(ix.turns[1].is_question);
        assert!(!ix.turns[5].is_question);
        assert!(c2_named_session(&ix, 5));
        let mut list = vec![0usize, 1, 2, 3, 4];
        let mut scene = vec![1.0f32; ix.turns.len()];
        scene[5] = 0.0;
        let mut own = vec![0.5f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 5.0;
        let ts = vec![0i64; ix.turns.len()];
        assert!(apply_c2_buried_lexical_peak(&ix, &mut list, &scene, &own, &ts, 10));
        assert_eq!(list, vec![0, 5, 2, 3, 4, 1]);

        // Two unpooled declaratives above the pool max: not unique, no move.
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("mel", "When did the shop open?"),
                    ("jon", "Where is the shop now?"),
                    ("mel", "The counter faces the window."),
                    ("jon", "The awning is green."),
                    ("mel", "The bell is on the door."),
                    ("jon", "The shop opened beside the river."),
                    ("mel", "The shop opened after the flood."),
                ],
            )],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 2, 3, 4];
        let mut scene = vec![1.0f32; ix.turns.len()];
        scene[5] = 0.0;
        scene[6] = 0.0;
        let mut own = vec![0.5f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 5.0;
        own[6] = 4.0;
        let ts = vec![0i64; ix.turns.len()];
        assert!(!apply_c2_buried_lexical_peak(&ix, &mut list, &scene, &own, &ts, 10));
        assert_eq!(list, vec![0, 1, 2, 3, 4]);

        // User/assistant transcripts are not named-speaker sessions.
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "When did the shop open?"),
                    ("assistant", "Where is the shop now?"),
                    ("user", "The counter faces the window."),
                    ("assistant", "The awning is green."),
                    ("user", "The bell is on the door."),
                    ("assistant", "The shop opened beside the river."),
                ],
            )],
            &Params::default(),
        );
        let mut list = vec![0usize, 1, 2, 3, 4];
        let mut scene = vec![1.0f32; ix.turns.len()];
        scene[5] = 0.0;
        let mut own = vec![0.5f32; ix.turns.len()];
        own[1] = 0.2;
        own[5] = 9.0;
        let ts = vec![0i64; ix.turns.len()];
        assert!(!apply_c2_buried_lexical_peak(&ix, &mut list, &scene, &own, &ts, 10));
        assert_eq!(list, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn anaphoric_tail_swaps_the_dominant_pronoun_with_the_weakest_question() {
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("mel", "The shop opened beside the river."),
                    ("jon", "When did the shop open?"),
                    ("mel", "The counter faces the window."),
                    ("jon", "The awning is green."),
                    ("mel", "The bell is on the door."),
                    ("jon", "It opened beside the river."),
                ],
            )],
            &Params::default(),
        );
        assert!(ix.turns[1].is_question);
        assert!(!ix.turns[5].is_question);
        assert!(ix.turns[5].anaphoric);
        assert!(c2_named_session(&ix, 5));
        let mut own = vec![0.5f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 3.0;
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(apply_c2_anaphoric_tail(&ix, &mut list, &own));
        assert_eq!(list, vec![0, 5, 2, 3, 4, 1]);

        // The tail maximum is not anaphoric, so a weaker anaphoric turn stays put.
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("mel", "The shop opened beside the river."),
                    ("jon", "When did the shop open?"),
                    ("mel", "The counter faces the window."),
                    ("jon", "The awning is green."),
                    ("mel", "The bell is on the door."),
                    ("jon", "The shop opened after the flood."),
                    ("mel", "It opened beside the river."),
                ],
            )],
            &Params::default(),
        );
        assert!(!ix.turns[5].anaphoric);
        assert!(ix.turns[6].anaphoric);
        let mut own = vec![0.5f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 4.0;
        own[6] = 3.0;
        let mut list = vec![0usize, 1, 2, 3, 4, 5, 6];
        assert!(!apply_c2_anaphoric_tail(&ix, &mut list, &own));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5, 6]);

        // User/assistant transcripts are not named-speaker sessions.
        let ix = build_index(
            &[sess(
                "s",
                &[
                    ("user", "The shop opened beside the river."),
                    ("assistant", "When did the shop open?"),
                    ("user", "The counter faces the window."),
                    ("assistant", "The awning is green."),
                    ("user", "The bell is on the door."),
                    ("assistant", "It opened beside the river."),
                ],
            )],
            &Params::default(),
        );
        assert!(ix.turns[5].anaphoric);
        let mut own = vec![0.5f32; ix.turns.len()];
        own[1] = 0.2;
        own[5] = 9.0;
        let mut list = vec![0usize, 1, 2, 3, 4, 5];
        assert!(!apply_c2_anaphoric_tail(&ix, &mut list, &own));
        assert_eq!(list, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn unseated_session_peak_seats_the_unique_outsider_from_a_missing_session() {
        let ix = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("mel", "The shop opened beside the river."),
                        ("jon", "When did the shop open?"),
                        ("mel", "The counter faces the window."),
                        ("jon", "The awning is green."),
                        ("mel", "The bell is on the door."),
                    ],
                ),
                sess("D2", &[("jon", "The mill kept its own ledger.")]),
            ],
            &Params::default(),
        );
        assert!(ix.turns[1].is_question);
        assert!(!ix.turns[5].is_question);
        assert_ne!(ix.turns[5].session, ix.turns[0].session);
        assert!(c2_named_session(&ix, 5));
        let mut own = vec![0.4f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 4.0;
        let mut list = vec![0usize, 1, 2, 3, 4];
        assert!(apply_c2_unseated_session_peak(&ix, &mut list, &own, 10));
        assert_eq!(list, vec![0, 5, 2, 3, 4, 1]);

        // Same-session outsider, even above the cap, is not a missing session.
        let ix = build_index(
            &[sess(
                "D1",
                &[
                    ("mel", "The shop opened beside the river."),
                    ("jon", "When did the shop open?"),
                    ("mel", "The counter faces the window."),
                    ("jon", "The awning is green."),
                    ("mel", "The bell is on the door."),
                    ("jon", "The mill kept its own ledger."),
                ],
            )],
            &Params::default(),
        );
        let mut own = vec![0.4f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 9.0;
        let mut list = vec![0usize, 1, 2, 3, 4];
        assert!(!apply_c2_unseated_session_peak(&ix, &mut list, &own, 10));
        assert_eq!(list, vec![0, 1, 2, 3, 4]);

        // Two outsiders above the returned maximum. Not unique.
        let ix = build_index(
            &[
                sess(
                    "D1",
                    &[
                        ("mel", "The shop opened beside the river."),
                        ("jon", "When did the shop open?"),
                        ("mel", "The counter faces the window."),
                        ("jon", "The awning is green."),
                        ("mel", "The bell is on the door."),
                    ],
                ),
                sess("D2", &[("jon", "The mill kept its own ledger.")]),
                sess("D3", &[("mel", "The dock kept a second ledger.")]),
            ],
            &Params::default(),
        );
        let mut own = vec![0.4f32; ix.turns.len()];
        own[0] = 1.0;
        own[1] = 0.2;
        own[5] = 4.0;
        own[6] = 3.0;
        let mut list = vec![0usize, 1, 2, 3, 4];
        assert!(!apply_c2_unseated_session_peak(&ix, &mut list, &own, 10));
        assert_eq!(list, vec![0, 1, 2, 3, 4]);

        // User/assistant transcripts are not named-speaker sessions.
        let ix = build_index(
            &[
                sess(
                    "s",
                    &[
                        ("user", "The shop opened beside the river."),
                        ("assistant", "When did the shop open?"),
                        ("user", "The counter faces the window."),
                        ("assistant", "The awning is green."),
                        ("user", "The bell is on the door."),
                    ],
                ),
                sess("t", &[("assistant", "The mill kept its own ledger.")]),
            ],
            &Params::default(),
        );
        let mut own = vec![0.4f32; ix.turns.len()];
        own[1] = 0.2;
        own[5] = 9.0;
        let mut list = vec![0usize, 1, 2, 3, 4];
        assert!(!apply_c2_unseated_session_peak(&ix, &mut list, &own, 10));
        assert_eq!(list, vec![0, 1, 2, 3, 4]);
    }
}

// ---- moved verbatim from former `src/steps11_20.rs` (was `include!()`d into retrieval.rs) ----

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

    #[test]
    fn step14_role_frame_extraction_and_bipartite_boost() {
        let (objs, attrs) = step14_extract_role_frames(
            "Besides great views, I also like hotels with unique features. Do you know any?",
        );
        assert!(objs.is_empty(), "role frames are not cut by a phrase list: {objs:?}");
        assert!(attrs.is_empty(), "role frames are not cut by a phrase list: {attrs:?}");

        let sessions = vec![Session {
            id: "s0".to_string(),
            date: String::new(),
            timestamp: 100,
            turns: vec![
                Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "I'm planning a trip to Seattle and need a hotel. Any recommendations?".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
                Turn {
                    id: "t1".to_string(),
                    role: "assistant".to_string(),
                    content: "Seattle has several great downtown hotels near the waterfront.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
                Turn {
                    id: "t2".to_string(),
                    role: "user".to_string(),
                    content: "Besides great views, I also like hotels with a rooftop pool. Do you know any?".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                },
            ],
        }];
        let ix = build_index(&sessions, &Params::default());
        let hid = ix.vocab.get("hotels").expect("hotels interned");
        let plan = QueryPlan {
            intent: QueryIntent {
                preference: true,
                ..QueryIntent::default()
            },
            terms: vec![QTerm {
                id: hid,
                weight: 2.0,
                group: Some(0),
                exact_hit: true,
            }],
            groups: vec!["hotels".to_string()],
            group_of: DetHashMap::default(),
            focal_ids: DetHashSet::default(),
        };
        let t_own = vec![10.0, 11.0, 9.5];
        let mut t_scene = vec![10.0, 11.0, 9.5];
        apply_step14_bipartite_pref_boost(
            &ix,
            "What hotels would you recommend?",
            &plan,
            true,
            &plan.intent,
            &t_own,
            &mut t_scene,
        );
        assert!((t_scene[2] - 9.5).abs() < 0.01, "role-object boost is gone: {:?}", t_scene);
        assert!(t_scene[1] < 11.0, "assistant field is still damped when preference is on: {:?}", t_scene);
        assert!(step14_is_factual_wh_query("How often did I initially aim to visit the gym per week?"));
        assert!(step14_is_factual_wh_query("What time do I usually go to the gym?"));
        assert!(!step14_is_factual_wh_query("What hotels would you recommend?"));
    }

    #[test]
    fn step15_comparative_dual_slot_splits_and_balances() {
        assert!(!step15_is_comparative_query(
            "Where did Rachel move to after her recent relocation?",
            &QueryIntent::default(),
        ));
        let (a1, b1) = step15_split_comparative_clauses(
            "Which device did I set up first, the smart thermostat or the mesh network system?",
        )
        .expect("or split");
        assert!(a1.contains("smart thermostat") && b1.contains("mesh network"), "a={a1} b={b1}");

        let (a2, b2) = step15_split_comparative_clauses(
            "How many days passed between the day I sold baked goods at Farmers Market and the day I joined Spring Fling?",
        )
        .expect("between-and split");
        assert!(a2.contains("sold baked goods") && b2.contains("joined spring fling"), "a={a2} b={b2}");

        let (a3, b3) = step15_split_comparative_clauses(
            "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        )
        .expect("before inversion split");
        assert!(a3.contains("iphone 13 pro") && b3.contains("holiday market"), "a={a3} b={b3}");

        let sessions = vec![
            Session {
                id: "s0".to_string(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "I bought the Ferrari model last spring.".to_string(),
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
                    content: "I started the Japanese Zero fighter plane model today.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
        ];
        let ix = build_index(&sessions, &Params::default());
        let q = Query {
            text: "Which project did I start first, the Ferrari model in 2019 or the Japanese Zero fighter plane model in 2021?".to_string(),
            date: String::new(),
        };
        let plan = plan_query(&ix, &q, &Params::default());
        let masks = step15_comparative_event_masks(&ix, &q.text, &plan, &plan.intent);
        assert!(masks.is_none(), "event-sequence dates do not open the comparative mask");
    }

    #[test]
    fn step17_isnad_three_session_chain_decays_per_hop() {
        let mut sessions = vec![
            Session {
                id: "s0".to_string(),
                date: String::new(),
                timestamp: 100,
                turns: vec![Turn {
                    id: "t0".to_string(),
                    role: "user".to_string(),
                    content: "My espresso grinder from AthenaRoast arrived intact.".to_string(),
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
                    content: "AthenaRoast partnered with BorealCeramics last spring.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
            Session {
                id: "s2".to_string(),
                date: String::new(),
                timestamp: 300,
                turns: vec![Turn {
                    id: "t2".to_string(),
                    role: "user".to_string(),
                    content: "My glazed espresso mug from BorealCeramics is blue.".to_string(),
                    links: Vec::new(),
                    stability: 0.0,
                }],
            },
        ];
        for i in 3..80 {
            sessions.push(Session {
                id: format!("s{i}"),
                date: String::new(),
                timestamp: (i as i64) * 100,
                turns: vec![Turn {
                    id: format!("t{i}"),
                    role: "user".to_string(),
                    content: format!("Routine weather entry number {i}."),
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
        };
        let mut t_own = vec![0.0f32; ix.turns.len()];
        let mut t_exact_mask = vec![0u64; ix.turns.len()];
        let mut t_scene = vec![0.0f32; ix.turns.len()];
        t_own[0] = 10.0;
        t_scene[0] = 10.0;
        t_exact_mask[0] = 1;
        t_scene[1] = 2.0;
        t_own[2] = 4.0;
        t_scene[2] = 4.0;
        t_exact_mask[2] = 1;

        apply_step17_isnad_multihop_chain(
            &ix,
            "Which mug goes with my espresso grinder?",
            &plan,
            false,
            &QueryIntent::default(),
            &t_own,
            &t_exact_mask,
            &mut t_scene,
        );
        assert!(t_scene[2] > 4.0, "expected 3-session Isnad lift on t2, got {}", t_scene[2]);
        assert!(t_scene[2] <= 0.88 * 0.82 * 10.0 + 1e-4, "hop3 must decay below hop2 & hop1: {}", t_scene[2]);
    }
}

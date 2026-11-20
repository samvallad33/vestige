//! Mirrored unit tests: `rank_tests` (16 tests).

use super::*;


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

#[test]
fn retrospective_assistant_reminder_routes_to_assistant_while_classify_stays_false() {
    let q_text = "I'm planning a trip to Kyoto; remind me what you recommended for day two?";
    let raw_it = classify(q_text);
    assert!(
        !raw_it.assistant_attribution,
        "query::intent::classify() must remain unmodified: {raw_it:?}"
    );
    let mut sessions = Vec::new();
    for i in 0..16 {
        sessions.push(Session {
            id: format!("noise-{i}"),
            date: String::new(),
            timestamp: 100 + i as i64,
            turns: vec![
                turn("user", &format!("General planning note {i} about packing luggage and checking weather.")),
                turn("assistant", &format!("Sure, note {i} about standard travel checklists.")),
            ],
        });
    }
    sessions.push(Session {
        id: "kyoto-rec".into(),
        date: String::new(),
        timestamp: 500,
        turns: vec![
            turn("user", "Can you suggest a Kyoto itinerary for my upcoming trip?"),
            turn("assistant", "For day two in Kyoto, visit Arashiyama Bamboo Grove early, then Tenryu-ji Temple and Philosopher's Path."),
        ],
    });
    let ranked = rank(&RankRequest {
        sessions,
        queries: vec![Query { text: q_text.into(), date: String::new() }],
        top_k: 5,
        params: Params::default(),
    });
    assert_eq!(ranked.results[0].sessions[0].id, "kyoto-rec");
    assert!(
        ranked.results[0].turns[0].id.contains("t1") || ranked.results[0].turns[0].session_id == "kyoto-rec",
        "expected assistant Kyoto recommendation turn at rank 1: {:?}",
        ranked.results[0].turns
    );
}

#[test]
fn topical_preference_beats_unrelated_standing_rule_without_hobby_flood() {
    let mut sessions = Vec::new();
    for i in 0..15 {
        sessions.push(Session {
            id: format!("hobby-{i}"),
            date: String::new(),
            timestamp: 100 + i as i64,
            turns: vec![
                turn("user", &format!("I enjoy hiking trail {i} and love outdoor photography on weekends.")),
                turn("assistant", "Sounds like a wonderful outdoor hobby!"),
            ],
        });
    }
    sessions.push(Session {
        id: "rule-date".into(),
        date: String::new(),
        timestamp: 200,
        turns: vec![
            turn("user", "Always use YYYY-MM-DD when writing calendar dates."),
            turn("assistant", "Understood, I will use ISO date format."),
        ],
    });
    sessions.push(Session {
        id: "coffee-pref".into(),
        date: String::new(),
        timestamp: 300,
        turns: vec![
            turn("user", "I prefer single-origin Ethiopian light roast pour-over coffee and dislike dark roast espresso. What beans should I try?"),
            turn("assistant", "Try washed Yirgacheffe or Guji light roasts."),
        ],
    });
    let ranked = rank(&RankRequest {
        sessions,
        queries: vec![Query {
            text: "Can you recommend a coffee roaster for my morning pour-over?".into(),
            date: String::new(),
        }],
        top_k: 5,
        params: Params::default(),
    });
    assert_eq!(
        ranked.results[0].sessions[0].id,
        "coffee-pref",
        "topical coffee preference must take Slot 0 ahead of unrelated standing rule & hobbies: {:?}",
        ranked.results[0].sessions
    );
}

#[test]
fn self_payload_turn_beats_carried_echo_opener_at_slot0() {
    let req = RankRequest {
        sessions: vec![
            Session {
                id: "s-echo".into(),
                date: "2024-03-10".into(),
                timestamp: 100,
                turns: vec![
                    Turn {
                        id: "s-echo:t0".into(),
                        ..turn("user", "Do you remember the museum we talked about earlier?")
                    },
                    Turn {
                        id: "s-echo:t1".into(),
                        ..turn("user", "I visited the Rijksmuseum in Amsterdam on March 10 and saw Rembrandt's Night Watch.")
                    },
                    Turn {
                        id: "s-echo:t2".into(),
                        ..turn("assistant", "The Night Watch at the Rijksmuseum is iconic!")
                    },
                ],
            },
            Session {
                id: "s-other".into(),
                date: "2024-03-12".into(),
                timestamp: 200,
                turns: vec![turn("user", "I bought groceries at the corner market today.")],
            },
        ],
        queries: vec![Query {
            text: "Which museum in Amsterdam did I visit?".into(),
            date: "2024-03-20".into(),
        }],
        top_k: 5,
        params: Params::default(),
    };
    let ranked = rank(&req);
    assert_eq!(
        ranked.results[0].turns[0].id, "s-echo:t1",
        "originating self-payload turn must win Slot 0 over echo question opener: {:?}",
        ranked.results[0].turns
    );
}

#[test]
fn prepositional_condition_standing_rule_extracts_condition_and_ranks_first() {
    let cond = crate::retrieval::extract_rule_condition("For API errors, always include HTTP status codes.");
    assert!(
        cond.iter().any(|w| w == "api") && cond.iter().any(|w| w == "errors"),
        "expected fronted prepositional condition tokens, got: {cond:?}"
    );
    let req = RankRequest {
        sessions: vec![
            Session {
                id: "rule-api".into(),
                date: String::new(),
                timestamp: 100,
                turns: vec![turn("user", "For API errors, always include HTTP status codes.")],
            },
            Session {
                id: "rule-other".into(),
                date: String::new(),
                timestamp: 200,
                turns: vec![turn("user", "For meeting notes, always use bullet lists.")],
            },
        ],
        queries: vec![Query {
            text: "How should I document API errors?".into(),
            date: String::new(),
        }],
        top_k: 2,
        params: Params::default(),
    };
    let ranked = rank(&req);
    assert_eq!(
        ranked.results[0].standing_rules.first().map(|r| r.session_id.as_str()),
        Some("rule-api"),
        "prepositional condition rule must rank #1: {:?}",
        ranked.results[0].standing_rules
    );
}

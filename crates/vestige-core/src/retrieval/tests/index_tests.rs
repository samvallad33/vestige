//! Mirrored unit tests: `index_tests` (10 tests).

use super::*;


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
fn step9_typo_and_ingest_shapes() {
    assert!(damerau_lev_1(b"recieve", b"receive"));
    assert!(damerau_lev_1(b"colour", b"color"));
    assert!(!damerau_lev_1(b"same", b"same"));
    assert_eq!(step9_list_item_count("Steps:\n1. a\n2) b\n- c\nend"), 3);
    assert_eq!(step9_first_move_num("Game so far:\n14. Nf3 e5\n15. Bb5"), 14);
    assert_eq!(step9_first_move_num("1. Buy milk"), 0);
}

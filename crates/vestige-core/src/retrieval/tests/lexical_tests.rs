//! Mirrored unit tests: `lexical_tests` (13 tests).

use super::*;


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
fn step7_proper_surfaces_skip_sentence_start() {
    let p = step7_query_proper_surfaces("What did Melanie's friend Jon say? Where is Rome");
    assert!(p.contains("melanie") && p.contains("jon"));
    assert!(!p.contains("where") && p.contains("rome"));
    assert!(!p.contains("what"));
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

//! Mirrored unit tests: `query_tests` (18 tests).

use super::*;


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
fn is_question_survives_trailing_code_or_markers() {
    let (is_q1, short1) = detect_question("Can you fix my Flask route? ->-> 1,14", 6);
    assert!(is_q1 && short1);
    let (is_q2, _) = detect_question("Can you initialize SQLite?\n```python\nimport sqlite3\n```", 5);
    assert!(is_q2);
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
fn a_dateless_question_does_not_invent_a_date_window() {
    assert!(inline_date_tokens("How do I sand a drawer?").is_empty());
    let early = inline_date_tokens("We met in early May before the show.");
    assert!(early.iter().any(|t| t == "__EARLY__05"), "{early:?}");
    assert!(early.iter().any(|t| t == "__MONTH__05"), "{early:?}");
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
fn beam_timeline_routes_to_user_and_ssa_collective_cues_stay_assistant() {
    let order = classify("Can you list the order in which I brought up different aspects of developing my personal budget tracker throughout our conversations?");
    assert!(!order.is_event_sequence && !order.user_attribution && !order.assistant_attribution && !order.task_under_rules, "{order:?}");
    let ssa = classify("I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job in the list you provided?");
    assert!(!ssa.assistant_attribution && !ssa.user_attribution, "{ssa:?}");
}
#[test]
fn step11_wh_qa_mode_closed_class_detection() {
    assert_eq!(step11_wh_qa_mode("How many books did I read in March?", false), (true, false));
    assert_eq!(step11_wh_qa_mode("How many days ago did I watch the Super Bowl?", true), (false, false));
    assert_eq!(step11_wh_qa_mode("How much did the repair cost?", false), (true, false));
    assert_eq!(step11_wh_qa_mode("Who won the bake-off?", false), (false, true));
    assert_eq!(step11_wh_qa_mode("Where did I put my keys?", false), (false, false));
}

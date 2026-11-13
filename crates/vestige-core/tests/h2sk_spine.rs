use std::collections::BTreeMap;
use vestige_core::h2sk::*;
fn event(id: &str, seq: u64, value: &str) -> LogEvent {
    LogEvent {
        id: id.into(),
        seq,
        text: format!("Mia lives in {value}."),
        tags: vec![
            "symbol:Mia".into(),
            "slot:Mia:residence".into(),
            format!("lit:{value}"),
        ],
        source_date: Some(format!("2023-03-{:02}", seq + 1)),
        stability_q: Some(4294967296),
    }
}
fn prefix() -> LogPrefix {
    LogPrefix {
        offset: 5,
        source_root: "synthetic-only".into(),
        events: vec![
            event("A", 1, "Austin"),
            event("B", 2, "Boston"),
            event("C", 3, "Denver"),
        ],
        edges: vec![
            LogEdge {
                from: "B".into(),
                to: "A".into(),
                kind: EdgeKind::Corrects,
                seq: 4,
            },
            LogEdge {
                from: "C".into(),
                to: "B".into(),
                kind: EdgeKind::Supersedes,
                seq: 5,
            },
        ],
        previous_receipt_hash: Some("prior".into()),
    }
}
fn options() -> CompileOptions {
    CompileOptions {
        query: "Where does Mia live now?".into(),
        symbols: vec!["symbol:Mia".into()],
        today: Some("2023-03-25".into()),
        bindings: BTreeMap::new(),
        view_ids: None,
    }
}
#[test]
fn apex_horosphere_and_off_spine_follow_only_recorded_edges() {
    let r = compile_log_prefix(&prefix(), &options()).unwrap();
    let row = &r.currency[0];
    assert_eq!(row.apex.as_deref(), Some("C"));
    assert_eq!(row.horosphere, ["B"]);
    assert_eq!(row.off_spine, ["A"]);
    assert!(verify_receipt(&r));
}
#[test]
fn unlinked_values_remain_ambiguous() {
    let mut p = prefix();
    p.edges.clear();
    let r = compile_log_prefix(&p, &options()).unwrap();
    assert_eq!(r.currency[0].status, CurrencyStatus::NoUniqueApex);
    assert!(r.section.iter().all(|s| s.banner.is_none()));
}
#[test]
fn closed_vocabulary_refuses_unknown_edges() {
    assert!(serde_json::from_str::<EdgeKind>("\"InventedSimilarity\"").is_err());
}
#[test]
fn cycles_and_malformed_endpoint_edges_are_rejected() {
    let mut p = prefix();
    p.edges.push(LogEdge {
        from: "A".into(),
        to: "C".into(),
        kind: EdgeKind::Supersedes,
        seq: 5,
    });
    assert_eq!(
        compile_log_prefix(&p, &options()).unwrap_err(),
        "currency_cycle"
    );
    p.edges[2].to = "missing".into();
    assert_eq!(
        compile_log_prefix(&p, &options()).unwrap_err(),
        "edge_endpoint_missing"
    );
}
#[test]
fn current_and_historical_citations_are_distinct() {
    let r = compile_log_prefix(&prefix(), &options()).unwrap();
    assert!(r.section[0].banner.as_ref().unwrap().contains("OVERRULED"));
    assert!(
        r.section[2]
            .banner
            .as_ref()
            .unwrap()
            .contains("CONTROLLING")
    );
    assert!(matches!(
        verify_draft(
            &r,
            &options(),
            &GateInput {
                draft: "Mia lives in Boston.".into(),
                ..Default::default()
            }
        ),
        GateVerdict::StaleLiteral { .. }
    ));
    assert_eq!(
        verify_draft(
            &r,
            &options(),
            &GateInput {
                draft: "Mia previously lived in Boston, and now lives in Denver.".into(),
                ..Default::default()
            }
        ),
        GateVerdict::Allow
    );
}
#[test]
fn tampering_and_different_slice_are_rejected() {
    let mut r = compile_log_prefix(&prefix(), &options()).unwrap();
    r.section[0].quote = "wrong".into();
    assert!(!verify_receipt(&r));
    assert_eq!(
        verify_draft(&r, &options(), &GateInput::default()),
        GateVerdict::SliceMismatch
    );
}
#[test]
fn impulse_and_hologram_are_bound_to_source() {
    let r = compile_log_prefix(&prefix(), &options()).unwrap();
    assert_eq!(r.impulse.len(), 3);
    assert!(
        r.impulse
            .iter()
            .all(|i| i.previous_receipt_hash.as_deref() == Some("prior"))
    );
    assert!(r.hologram.unwrap().bins > 0);
    assert!(
        r.section[0]
            .deltas
            .iter()
            .any(|d| d.days == 23 && d.whole_weeks == 3 && d.remainder_days == 2)
    );
}
#[test]
fn empty_symbols_produce_exact_abstention() {
    let mut o = options();
    o.symbols = vec!["symbol:Noah".into()];
    let r = compile_log_prefix(&prefix(), &o).unwrap();
    assert!(matches!(r.gate, GateVerdict::Abstain { .. }));
    assert!(matches!(
        verify_draft(&r, &o, &GateInput::default()),
        GateVerdict::Abstain { .. }
    ));
}
#[test]
fn gate_catches_false_abstention() {
    let r = compile_log_prefix(&prefix(), &options()).unwrap();
    assert!(matches!(
        verify_draft(
            &r,
            &options(),
            &GateInput {
                draft: "I do not have enough information.".into(),
                ..Default::default()
            }
        ),
        GateVerdict::FalseAbstention { .. }
    ));
}
#[test]
fn deterministic_admission_does_not_turn_wishes_into_facts() {
    let mut e = event("A", 1, "Austin");
    e.text = "I wish Mia lived in a cabin.".into();
    e.tags.clear();
    let (e, edges) = admit::admit_statement(e, &[], &[]);
    assert_eq!(admit::modality(&e.text, &e.tags), Modality::Hypothetical);
    assert!(!e.tags.iter().any(|s| s.starts_with("slot:")));
    assert!(edges.is_empty());
}
#[test]
fn one_admitter_emits_only_explicit_exact_corrections() {
    let mut a = event("A", 1, "Austin");
    a.tags.clear();
    let (a, _) = admit::admit_statement(a, &[], &[]);
    let mut b = event("B", 2, "Boston");
    b.tags.clear();
    b.text = "Correction: Mia lives in Boston.".into();
    let (b, edges) = admit::admit_statement(b, &[a], &[]);
    assert!(b.tags.contains(&"slot:Mia:residence".into()));
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].kind, EdgeKind::Corrects);
}
#[test]
fn temporal_resolver_handles_leaps_weekdays_unknown_dates_and_unicode() {
    let b = BTreeMap::from([("she".into(), "Mia".into())]);
    let s = pragmatics::canonicalize(
        "İstanbul: she went two weeks ago and last Friday.",
        Some("2024-03-01"),
        &b,
    );
    assert!(s.contains("she [= Mia]"));
    assert!(s.contains("2024-02-16"));
    assert!(s.contains("2024-02-23"));
    assert_eq!(
        pragmatics::canonicalize("two weeks ago", None, &BTreeMap::new()),
        "two weeks ago"
    );
}
#[test]
fn compile_is_reproducible_and_does_not_mutate_source() {
    let p = prefix();
    let before = serde_json::to_string(&p).unwrap();
    assert_eq!(
        compile_log_prefix(&p, &options()).unwrap().hash,
        compile_log_prefix(&p, &options()).unwrap().hash
    );
    assert_eq!(before, serde_json::to_string(&p).unwrap());
}
#[test]
fn selected_old_hit_still_carries_its_exact_controlling_successor() {
    let mut o = options();
    o.view_ids = Some(vec!["B".into()]);
    let r = compile_log_prefix(&prefix(), &o).unwrap();
    assert_eq!(r.currency[0].apex.as_deref(), Some("C"));
    assert!(r.section.iter().any(|s| s.id == "C"));
}
#[test]
fn empty_selected_view_does_not_fall_back_to_salient_history() {
    let mut o = options();
    o.view_ids = Some(vec![]);
    let r = compile_log_prefix(&prefix(), &o).unwrap();
    assert!(r.section.is_empty());
    assert!(matches!(r.gate, GateVerdict::Abstain { .. }));
}

#[test]
fn identical_unlinked_records_are_not_falsely_stamped_overruled() {
    let mut p = prefix();
    p.events[1] = event("B", 2, "Austin");
    p.events.truncate(2);
    p.edges.clear();
    let r = compile_log_prefix(&p, &options()).unwrap();
    assert!(
        r.section
            .iter()
            .all(|s| !s.banner.as_deref().unwrap_or("").contains("OVERRULED"))
    );
}

#[test]
fn a_changed_question_is_a_different_receipt_slice() {
    let r = compile_log_prefix(&prefix(), &options()).unwrap();
    let mut o = options();
    o.query = "Where did Mia live initially?".into();
    assert_eq!(
        verify_draft(&r, &o, &GateInput::default()),
        GateVerdict::SliceMismatch
    );
}

#[test]
fn active_preference_supports_advice_and_pointers_are_not_empty() {
    let mut p = prefix();
    p.events = vec![LogEvent {
        id: "gear".into(),
        seq: 1,
        text: "I use a small camera.".into(),
        tags: vec!["gear:camera".into()],
        source_date: None,
        stability_q: None,
    }];
    p.edges.clear();
    let o = CompileOptions {
        query: "Any tips for my camera?".into(),
        ..Default::default()
    };
    let r = compile_log_prefix(&p, &o).unwrap();
    assert_eq!(
        verify_draft(
            &r,
            &o,
            &GateInput {
                draft: "Not mentioned.".into(),
                ..Default::default()
            }
        ),
        GateVerdict::FalseAbstention {
            controlling: vec!["gear".into()]
        }
    );
}

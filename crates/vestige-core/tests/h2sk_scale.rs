use vestige_core::h2sk::*;
fn event(id: &str, seq: u64, text: &str) -> LogEvent {
    LogEvent {
        id: id.into(),
        seq,
        text: text.into(),
        tags: vec![],
        source_date: None,
        stability_q: None,
    }
}
#[test]
fn admission_preserves_words_and_records_anchored_relative_dates() {
    let mut e = event("dated", 1, "I saw Mia two weeks ago.");
    e.source_date = Some("2023-03-25".into());
    let (out, _) = admit::admit_statement(e, &[], &[]);
    assert_eq!(out.text, "I saw Mia two weeks ago.");
    assert!(out.tags.contains(&"deixis:two-weeks-ago=2023-03-11".into()));
}
#[test]
fn one_admission_path_handles_correction_then_replacement() {
    let (a, _) = admit::admit_statement(event("A", 1, "Mia lives in Austin."), &[], &[]);
    let (b, mut edges) = admit::admit_statement(
        event("B", 2, "correction: Mia lives in Boston."),
        std::slice::from_ref(&a),
        &[],
    );
    let (c, next) = admit::admit_statement(
        event("C", 3, "Mia moved to Denver."),
        &[a.clone(), b.clone()],
        &edges,
    );
    edges.extend(next);
    let p = LogPrefix {
        offset: 3,
        events: vec![a, b, c],
        edges,
        ..Default::default()
    };
    let r = compile_log_prefix(&p, &CompileOptions::default()).unwrap();
    assert_eq!(r.currency[0].apex.as_deref(), Some("C"));
    assert_eq!(r.currency[0].horosphere, vec!["B"]);
}
#[test]
fn selected_view_stays_small_in_fifty_thousand_records() {
    let (a, _) = admit::admit_statement(event("A", 1, "Mia lives in Austin."), &[], &[]);
    let (b, edges) = admit::admit_statement(
        event("B", 2, "Correction: Mia lives in Denver."),
        std::slice::from_ref(&a),
        &[],
    );
    let mut p = LogPrefix {
        offset: 50_003,
        events: vec![a, b],
        edges,
        ..Default::default()
    };
    for n in 0..50_000 {
        p.events.push(event(
            &format!("noise-{n}"),
            3 + n,
            "An unrelated synthetic archive entry.",
        ));
    }
    let o = CompileOptions {
        query: "Where does Mia live now?".into(),
        view_ids: Some(vec!["A".into()]),
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let r = compile_log_prefix(&p, &o).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(r.section.len(), 2);
    assert_eq!(r.currency[0].apex.as_deref(), Some("B"));
    let payload =
        compile::prepare_payload(&r, &std::collections::BTreeSet::from(["A".into()]), 7000);
    assert!(payload.contains("Denver"));
    assert!(payload.contains("CONTROLLING"));
    eprintln!(
        "50k source projection: {elapsed:?}; compiled sources={}",
        r.section.len()
    );
}
#[test]
fn exact_number_date_and_id_values_are_named_without_a_second_extractor() {
    let (number, edges) = admit::admit_statement(event("n", 1, "Mia owns 2."), &[], &[]);
    assert!(number.tags.iter().any(|tag| tag == "num:2"), "{number:?}");
    assert!(edges.is_empty());
    let (next, edges) =
        admit::admit_statement(event("n2", 2, "Correction: Mia owns 3."), &[number], &[]);
    assert!(next.tags.iter().any(|tag| tag == "num:3"));
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].kind, EdgeKind::Corrects);
    let (dated, _) = admit::admit_statement(event("d", 3, "Mia works at 2024-05-01."), &[], &[]);
    assert!(dated.tags.iter().any(|tag| tag == "date:2024-05-01"));
    let (ident, _) = admit::admit_statement(event("i", 4, "Mia likes id:gear-7."), &[], &[]);
    assert!(ident.tags.iter().any(|tag| tag == "id:gear-7"));
}
#[test]
fn generic_slots_never_combine_different_people() {
    let mut mia = event("mia", 1, "Mia works at A.");
    mia.tags = vec!["slot:job".into(), "person:Mia".into(), "lit:A".into()];
    let mut noah = event("noah", 2, "Noah works at B.");
    noah.tags = vec!["slot:job".into(), "person:Noah".into(), "lit:B".into()];
    let p = LogPrefix {
        offset: 2,
        events: vec![mia, noah],
        ..Default::default()
    };
    let r = compile_log_prefix(&p, &CompileOptions::default()).unwrap();
    assert_eq!(r.currency.len(), 2);
    assert!(r.currency.iter().all(|row| row.values.len() == 1));
}

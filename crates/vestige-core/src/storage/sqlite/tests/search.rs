//! Tests for `sqlite/search.rs`: keyword filters, hybrid and concrete search.

use super::*;

#[test]
fn test_parse_timestamp_accepts_rfc3339_and_sqlite_native() {
    use chrono::TimeZone;

    // Canonical writer: RFC 3339 with fractional seconds + offset.
    let rfc = Storage::parse_timestamp("2026-06-12T15:07:59.730+00:00", "last_accessed").unwrap();
    assert_eq!(rfc.to_rfc3339(), "2026-06-12T15:07:59.730+00:00");

    // External writer: SQLite-native `datetime('now')` (space separator,
    // no timezone, no fraction) — must be tolerated, assumed UTC.
    let sqlite = Storage::parse_timestamp("2026-06-12 15:07:59", "last_accessed").unwrap();
    assert_eq!(
        sqlite,
        Utc.with_ymd_and_hms(2026, 6, 12, 15, 7, 59).unwrap()
    );

    // SQLite-native with fractional seconds.
    let sqlite_frac = Storage::parse_timestamp("2026-06-12 15:07:59.730", "last_accessed").unwrap();
    assert_eq!(sqlite_frac.timestamp_subsec_millis(), 730);

    // Genuinely malformed input still errors.
    assert!(Storage::parse_timestamp("not-a-timestamp", "last_accessed").is_err());
}

#[test]
fn test_search() {
    let storage = create_test_storage();

    let input = IngestInput {
        content: "The mitochondria is the powerhouse of the cell".to_string(),
        node_type: "fact".to_string(),
        ..Default::default()
    };

    storage.ingest(input).unwrap();

    let results = storage.search("mitochondria", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("mitochondria"));
}

#[test]
fn test_keyword_search_with_include_types() {
    let storage = create_test_storage();

    // Ingest nodes of different types all containing the word "quantum"
    storage
        .ingest(IngestInput {
            content: "Quantum mechanics is fundamental to physics".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Quantum computing uses qubits for calculation".to_string(),
            node_type: "concept".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Quantum entanglement was demonstrated in the lab".to_string(),
            node_type: "event".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search with include_types = ["fact"] — should only return the fact
    let include = vec!["fact".to_string()];
    let results = storage
        .hybrid_search_filtered("quantum", 10, 0.3, 0.7, Some(&include), None)
        .unwrap();

    assert!(!results.is_empty(), "should return at least one result");
    for r in &results {
        assert_eq!(
            r.node.node_type, "fact",
            "include_types=[fact] should only return facts, got: {}",
            r.node.node_type
        );
    }
}

#[test]
fn test_keyword_search_with_exclude_types() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Photosynthesis converts sunlight to energy".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Photosynthesis is a complex biochemical process".to_string(),
            node_type: "reflection".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search with exclude_types = ["reflection"] — should skip the reflection
    let exclude = vec!["reflection".to_string()];
    let results = storage
        .hybrid_search_filtered("photosynthesis", 10, 0.3, 0.7, None, Some(&exclude))
        .unwrap();

    assert!(!results.is_empty(), "should return at least one result");
    for r in &results {
        assert_ne!(
            r.node.node_type, "reflection",
            "exclude_types=[reflection] should not return reflections"
        );
    }
}

#[test]
fn test_include_types_takes_precedence_over_exclude() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Gravity holds planets in orbit around stars".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "Gravity waves were first detected by LIGO".to_string(),
            node_type: "event".to_string(),
            ..Default::default()
        })
        .unwrap();

    // When both are provided, include_types wins
    let include = vec!["fact".to_string()];
    let exclude = vec!["fact".to_string()];
    let results = storage
        .hybrid_search_filtered("gravity", 10, 0.3, 0.7, Some(&include), Some(&exclude))
        .unwrap();

    // include_types takes precedence — facts should be returned
    assert!(!results.is_empty());
    for r in &results {
        assert_eq!(r.node.node_type, "fact");
    }
}

#[test]
fn test_type_filter_with_no_matches_returns_empty() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "DNA carries genetic information in cells".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    // Search for a type that doesn't exist among matches
    let include = vec!["person".to_string()];
    let results = storage
        .hybrid_search_filtered("DNA", 10, 0.3, 0.7, Some(&include), None)
        .unwrap();

    assert!(
        results.is_empty(),
        "filtering for a non-matching type should return empty results"
    );
}

#[test]
fn test_hybrid_search_backward_compat() {
    // Ensure the original hybrid_search (no type filters) still works
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "Neurons transmit electrical signals in the brain".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage.hybrid_search("neurons", 10, 0.3, 0.7).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].node.content.contains("Neurons"));
}

#[test]
fn test_concrete_search_literal_identifier_lands_first() {
    let storage = create_test_storage();

    storage
        .ingest(IngestInput {
            content: "General OpenAI API setup notes without the exact env var".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    let target = storage
        .ingest(IngestInput {
            content: "Set OPENAI_API_KEY before running the release smoke tests".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    storage
        .ingest(IngestInput {
            content: "API keys should be handled carefully in shell profiles".to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage
        .concrete_search_filtered("OPENAI_API_KEY", 10, None, None)
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].node.id, target.id);
    assert_eq!(results[0].match_type, MatchType::Keyword);
    assert!(results[0].semantic_score.is_none());
}

/// A memory that merely CITES an identifier must not outrank the memory that
/// IS it. Raw BM25 magnitude is unbounded while literal_match_score is capped
/// at 3.0, and both fed the same combined_score: measured on a 202-document
/// corpus, a note citing a UUID three times scored 27.5 against the exact
/// match's 3.0. That inverted the documented exact-lookup guarantee.
/// A corrupt FTS index must NOT strand the user's memories. `knowledge_fts`
/// is declared `content='knowledge_nodes'`, so it is derived state and is
/// always reconstructible. This reproduces the field failure: a store with
/// intact memories became unopenable because one fts5 blob was damaged.
#[test]
fn corrupt_fts_index_is_rebuilt_instead_of_bricking_the_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vestige.db");

    // Seed a store and close it cleanly.
    {
        let storage = Storage::new(Some(path.clone())).expect("open");
        for i in 0..5 {
            storage
                .ingest(IngestInput {
                    content: format!("memory number {i} about deployment"),
                    node_type: "fact".to_string(),
                    ..Default::default()
                })
                .expect("ingest");
        }
    }

    // Corrupt the FTS index the way an interrupted rebuild does. The
    // damage is a FIXED byte pattern, not randomblob(): an unseeded random
    // block sometimes wrecks the segment so badly that quick_check itself
    // fails with SQLITE_NOMEM before the heal gate can classify the rows,
    // and the store (correctly) refuses loudly. That path is documented
    // shipped behavior, but it made this test flake in CI; a deterministic
    // pattern exercises the rebuild path every run.
    {
        let conn = Connection::open(&path).expect("raw open");
        let pattern = "A5".repeat(200);
        conn.execute_batch(&format!(
            "UPDATE knowledge_fts_data SET block = x'{pattern}' \
                 WHERE id = (SELECT id FROM knowledge_fts_data WHERE id > 1 LIMIT 1);"
        ))
        .expect("corrupt");
        let corrupt = conn
            .execute_batch("INSERT INTO knowledge_fts(knowledge_fts) VALUES('integrity-check');")
            .is_err();
        assert!(corrupt, "the fixture must actually corrupt the index");
    }

    // Reopening must succeed by rebuilding, not fail.
    let storage = Storage::new(Some(path.clone()))
        .expect("a corrupt DERIVED index must not prevent opening the store");
    let all = storage.get_all_nodes(100, 0).expect("list nodes");
    assert_eq!(all.len(), 5, "every memory must survive the rebuild");

    // And the rebuilt index must actually be usable again.
    let hits = storage
        .concrete_search_filtered("deployment", 10, None, None)
        .expect("keyword search after rebuild");
    assert!(
        !hits.is_empty(),
        "the rebuilt index must find the seeded memories"
    );
}

#[test]
fn test_concrete_search_exact_match_beats_a_doc_that_only_cites_it() {
    let storage = create_test_storage();

    // Filler so BM25's IDF term is meaningful rather than degenerate.
    for i in 0..40 {
        storage
            .ingest(IngestInput {
                content: format!("Routine note {i} about deployment pipelines and review"),
                node_type: "fact".to_string(),
                ..Default::default()
            })
            .unwrap();
    }

    let needle = "PAYMENTS_REDIS_URL";
    let target = storage
        .ingest(IngestInput {
            content: needle.to_string(),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();
    // Cites the identifier repeatedly -> large BM25 magnitude.
    storage
        .ingest(IngestInput {
            content: format!(
                "See {needle} for the rollout; {needle} was rotated in review, and \
                     {needle} supersedes the older connection note entirely"
            ),
            node_type: "fact".to_string(),
            ..Default::default()
        })
        .unwrap();

    let results = storage
        .concrete_search_filtered(needle, 10, None, None)
        .unwrap();
    assert!(!results.is_empty(), "exact lookup must return something");
    assert_eq!(
        results[0].node.id, target.id,
        "the memory that IS the identifier must rank first, not the one citing it"
    );
}

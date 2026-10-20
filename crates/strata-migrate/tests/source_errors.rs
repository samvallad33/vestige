//! A read the v3 source cannot answer is an error, never a default (#496).
//!
//! `schema_version` and the dropped-vector count land in the signed
//! MIGRATION_RECEIPT (the version in GENESIS and PARAMS too), and
//! `skipped_tables` tells the user what stayed behind. An out-of-range or
//! missing version used to migrate as 0, and a table whose row count could
//! not be read, or whose name holds a double quote, was left out of the
//! report.

use std::path::Path;

use strata::StrataLog;
use strata_migrate::{migrate, read_snapshot, MigrationError};

const SCHEMA: &str = r#"
    CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
    CREATE TABLE knowledge_nodes (
        id TEXT PRIMARY KEY, content TEXT NOT NULL, node_type TEXT NOT NULL DEFAULT 'fact',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_accessed TEXT NOT NULL,
        stability REAL NOT NULL DEFAULT 1.0, difficulty REAL NOT NULL DEFAULT 5.0,
        reps INTEGER NOT NULL DEFAULT 0, lapses INTEGER NOT NULL DEFAULT 0,
        learning_state TEXT NOT NULL DEFAULT 'new', sentiment_magnitude REAL NOT NULL DEFAULT 0.0,
        tags TEXT DEFAULT '[]', source TEXT, superseded_by TEXT
    );
    INSERT INTO knowledge_nodes (id, content, created_at, updated_at, last_accessed)
        VALUES ('11111111-1111-4111-8111-111111111111', 'a v3 memory',
                '2026-09-28T00:00:00Z', '2026-09-28T00:00:00Z', '2026-09-28T00:00:00Z');
"#;

const VERSION_38: &str = "INSERT INTO schema_version VALUES (38, '2026-09-28T00:00:00Z');";

/// A table sqlite_master lists whose rows no reader can count: a virtual
/// table over a module this build does not have.
fn ghost_table(name: &str) -> String {
    format!(
        "PRAGMA writable_schema = ON;
         INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql)
             VALUES ('table', '{name}', '{name}', 0,
                     'CREATE VIRTUAL TABLE {name} USING nosuchmodule(x)');
         PRAGMA writable_schema = OFF;"
    )
}

fn build(db: &Path, extra: &str) {
    let conn = rusqlite::Connection::open(db).expect("open");
    conn.execute_batch(SCHEMA).expect("schema");
    conn.execute_batch(extra).expect("extra");
}

/// Migrate a store built from `SCHEMA` plus `extra`.
fn migrate_store(
    extra: &str,
) -> (
    tempfile::TempDir,
    Result<strata_migrate::MigrationReport, MigrationError>,
) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    build(&db, extra);
    let result = migrate(&db, &tmp.path().join("strata"));
    (tmp, result)
}

/// The schema version as GENESIS, PARAMS and the signed receipt carry it.
fn signed_versions(strata_dir: &Path) -> (u32, u32, u32) {
    let log = StrataLog::open(strata_dir).expect("reopen log");
    let snapshot = read_snapshot(&log).expect("read snapshot");
    (
        snapshot.genesis.expect("genesis").schema_version,
        snapshot.params.expect("params").schema_version,
        snapshot.receipt.expect("receipt").body.schema_version,
    )
}

#[test]
fn a_valid_schema_version_is_signed_as_it_is() {
    let (tmp, result) = migrate_store(VERSION_38);
    let report = result.expect("migrate");
    assert!(report.receipt_verified);
    assert_eq!(signed_versions(&tmp.path().join("strata")), (38, 38, 38));
}

#[test]
fn a_schema_version_outside_u32_is_refused_not_signed_as_zero() {
    for value in ["-1", "4294967296"] {
        let (tmp, result) = migrate_store(&format!(
            "INSERT INTO schema_version VALUES ({value}, '2026-09-28T00:00:00Z');"
        ));
        let err = result.expect_err(value);
        assert!(matches!(err, MigrationError::Corrupt(_)), "{value}: {err}");
        assert!(
            err.to_string()
                .contains(&format!("schema_version {value} ")),
            "{value}: {err}"
        );
        assert!(
            !tmp.path().join("strata").exists(),
            "{value}: a refused source created the destination"
        );
    }
}

#[test]
fn a_store_with_memories_and_no_schema_version_is_refused() {
    for (label, extra) in [
        ("an empty schema_version table", ""),
        ("no schema_version table", "DROP TABLE schema_version;"),
    ] {
        let (tmp, result) = migrate_store(extra);
        let err = result.expect_err(label);
        assert!(!tmp.path().join("strata").exists(), "{label}");
        assert!(
            matches!(err, MigrationError::UnsupportedSource(_)),
            "{label}: {err}"
        );
        assert!(
            err.to_string().contains("records no schema_version"),
            "{label}: {err}"
        );
    }
}

#[test]
fn a_store_without_memories_or_a_version_migrates_as_version_0() {
    let (tmp, result) = migrate_store("DELETE FROM knowledge_nodes; DROP TABLE schema_version;");
    let report = result.expect("an empty store has nothing to misdate");
    assert_eq!(report.nodes, 0);
    assert_eq!(signed_versions(&tmp.path().join("strata")), (0, 0, 0));
}

#[test]
fn every_nonempty_table_is_named_even_when_its_name_or_count_is_awkward() {
    let extra = format!(
        r#"{VERSION_38}
        CREATE TABLE side_notes (x TEXT);
        INSERT INTO side_notes VALUES ('kept out of strata');
        CREATE TABLE "odd""name" (x TEXT);
        INSERT INTO "odd""name" VALUES ('a table name with a double quote');
        {}"#,
        ghost_table("ghost_vtab")
    );
    let (_tmp, result) = migrate_store(&extra);
    let report = result.expect("migrate");
    let mut skipped = report.skipped_tables.clone();
    skipped.sort();
    assert_eq!(
        skipped,
        ["ghost_vtab", "odd\"name", "schema_version", "side_notes"],
        "{report:?}"
    );
}

#[test]
fn an_uncountable_node_embeddings_table_is_refused_not_signed_as_zero() {
    let (_tmp, result) = migrate_store(&format!("{VERSION_38}{}", ghost_table("node_embeddings")));
    let err = result.expect_err("the receipt would name 0 dropped vectors");
    assert!(matches!(err, MigrationError::Source(_)), "{err}");
    assert!(err.to_string().contains("count node_embeddings"), "{err}");
}

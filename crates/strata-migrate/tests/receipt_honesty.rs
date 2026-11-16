//! #496: a SQLite read that fails must not become `schema_version` 0 or a
//! skipped-tables list that omits the table.
//!
//! These are the receipt inputs. A bad version aborts before any frame is
//! appended. A table whose `COUNT(*)` fails is still named.

use strata::StrataLog;
use strata_migrate::source::load_snapshot;
use strata_migrate::source::{GuardFiles, SourceFiles};
use strata_migrate::{migrate, read_snapshot, MigrationError};

fn files(db: &std::path::Path) -> SourceFiles {
    SourceFiles {
        db: db.to_path_buf(),
        wal: None,
        shm: None,
        guard: GuardFiles {
            db: db.to_path_buf(),
            wal: None,
            shm: None,
        },
    }
}

fn open_db(path: &std::path::Path) -> rusqlite::Connection {
    rusqlite::Connection::open(path).expect("create sqlite")
}

fn load_err(db: &std::path::Path) -> MigrationError {
    match load_snapshot(&files(db)) {
        Ok(_) => panic!("expected a corrupt source, got a snapshot"),
        Err(err) => err,
    }
}

#[test]
fn empty_database_reports_schema_version_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("empty.db");
    open_db(&db)
        .execute_batch("PRAGMA user_version = 0;")
        .expect("write a sqlite header");

    let snapshot = load_snapshot(&files(&db)).expect("empty database is version 0");
    assert_eq!(snapshot.schema_version, 0);
    assert!(snapshot.all_nonempty_tables.is_empty());

    let strata_dir = tmp.path().join("strata");
    let report = migrate(&db, &strata_dir).expect("empty database migrates");
    assert!(
        report.skipped_tables.is_empty(),
        "{:?}",
        report.skipped_tables
    );
    let log = StrataLog::open(&strata_dir).expect("open log");
    let written = read_snapshot(&log).expect("snapshot");
    assert_eq!(written.genesis.expect("genesis").schema_version, 0);
}

#[test]
fn negative_schema_version_is_corrupt_and_names_the_value() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    open_db(&db)
        .execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
             INSERT INTO schema_version (version, applied_at) VALUES (-1, '2026-09-28T00:00:00Z');
             CREATE TABLE notes (body TEXT NOT NULL);
             INSERT INTO notes (body) VALUES ('kept');",
        )
        .expect("plant negative version");

    let message = load_err(&db).to_string();
    assert!(
        message.contains("schema_version") && message.contains("-1"),
        "the error names the bad value: {message}"
    );
    assert!(
        !message.contains("schema_version value 0"),
        "the failure must not be rewritten as version 0: {message}"
    );

    let migrated = migrate(&db, &tmp.path().join("strata"));
    let message = migrated
        .expect_err("no receipt is sealed for a bad version")
        .to_string();
    assert!(message.contains("-1"), "{message}");
}

#[test]
fn text_and_oversized_schema_versions_are_corrupt() {
    let tmp = tempfile::tempdir().expect("tempdir");

    let text_db = tmp.path().join("text.db");
    open_db(&text_db)
        .execute_batch(
            "CREATE TABLE schema_version (version TEXT, applied_at TEXT);
             INSERT INTO schema_version (version, applied_at) VALUES ('nope', 't');",
        )
        .expect("plant text version");
    let message = load_err(&text_db).to_string();
    assert!(
        message.contains("nope"),
        "the error names the text value: {message}"
    );

    let wide_db = tmp.path().join("wide.db");
    open_db(&wide_db)
        .execute_batch(
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT);
             INSERT INTO schema_version (version, applied_at) VALUES (4294967296, 't');",
        )
        .expect("plant oversized version");
    let message = load_err(&wide_db).to_string();
    assert!(
        message.contains("4294967296"),
        "the error names the oversized value: {message}"
    );
}

#[test]
fn missing_schema_version_on_a_nonempty_database_is_corrupt() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    open_db(&db)
        .execute_batch(
            "CREATE TABLE notes (body TEXT NOT NULL);
             INSERT INTO notes (body) VALUES ('kept');",
        )
        .expect("plant notes without schema_version");

    let message = load_err(&db).to_string();
    assert!(
        message.contains("schema_version") && message.contains("missing"),
        "{message}"
    );
}

#[test]
fn empty_schema_version_table_with_other_tables_is_corrupt() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    open_db(&db)
        .execute_batch(
            "CREATE TABLE schema_version (version INTEGER, applied_at TEXT);
             CREATE TABLE notes (body TEXT NOT NULL);
             INSERT INTO notes (body) VALUES ('kept');",
        )
        .expect("plant empty schema_version");

    let message = load_err(&db).to_string();
    assert!(
        message.contains("schema_version") && message.contains("no version row"),
        "{message}"
    );
}

#[test]
fn an_unreadable_table_is_still_named_in_skipped_tables() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = tmp.path().join("vestige.db");
    let conn = open_db(&db);
    conn.execute_batch(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
         INSERT INTO schema_version (version, applied_at) VALUES (38, '2026-09-28T00:00:00Z');
         CREATE TABLE audit_notes (body TEXT NOT NULL);
         INSERT INTO audit_notes (body) VALUES ('left in the source');
         PRAGMA writable_schema = ON;",
    )
    .expect("plant schema");
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql)
         VALUES ('table', 'left_behind', 'left_behind', 0, 'CREATE TABLE left_behind (id INTEGER)')",
        [],
    )
    .expect("plant a table whose pages are not readable");
    conn.execute_batch("PRAGMA writable_schema = OFF;")
        .expect("reset writable schema");
    drop(conn);

    let snapshot = load_snapshot(&files(&db)).unwrap_or_else(|err| {
        panic!("an unreadable side table must not abort the snapshot: {err}")
    });
    assert_eq!(snapshot.schema_version, 38);
    assert!(
        snapshot
            .all_nonempty_tables
            .iter()
            .any(|(name, count)| name == "left_behind" && count.is_none()),
        "an unreadable table stays on the list with an unknown count: {:?}",
        snapshot.all_nonempty_tables
    );
    assert!(
        snapshot
            .all_nonempty_tables
            .iter()
            .any(|(name, count)| name == "audit_notes" && *count == Some(1)),
        "a readable side table is still counted: {:?}",
        snapshot.all_nonempty_tables
    );

    let report = migrate(&db, &tmp.path().join("strata")).unwrap_or_else(|err| {
        panic!("an unreadable side table must not abort the migration: {err}")
    });
    assert!(
        report
            .skipped_tables
            .iter()
            .any(|name| name == "left_behind"),
        "skipped tables omitted the unreadable table: {:?}",
        report.skipped_tables
    );
    assert!(
        report
            .skipped_tables
            .iter()
            .any(|name| name == "audit_notes"),
        "skipped tables omitted the readable side table: {:?}",
        report.skipped_tables
    );
}

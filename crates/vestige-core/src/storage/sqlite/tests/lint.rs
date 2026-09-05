//! Policy lint: every writer transaction in the storage layer must begin
//! IMMEDIATE.
//!
//! A DEFERRED transaction that reads before it writes can fail with
//! `SQLITE_BUSY_SNAPSHOT` the moment another process (the CLI next to the MCP
//! server) commits in between, and SQLite does not consult the busy handler
//! for that upgrade. `BEGIN IMMEDIATE` takes the write lock up front, where
//! `busy_timeout` applies, and SQLite then guarantees no `SQLITE_BUSY` until
//! COMMIT. Read-only transactions on the reader connection stay DEFERRED.

/// Every storage module that can open a transaction on a writer
/// connection. The rule is module-wide, not file-wide: the first version
/// of this lint read the single-file `sqlite.rs` alone, and two writers
/// drifted DEFERRED in the blind spot (`trace_store.rs`'s memory-PR decide
/// path, and the store's own `unchecked_transaction` in the open-time FK
/// repair). The store is now a directory of modules, so every file under
/// `storage/sqlite/` is listed here, tests included; a file that is not in
/// this list is a file the lint cannot see.
const STORAGE_SOURCES: [(&str, &str); 33] = [
    ("sqlite/mod.rs", include_str!("../mod.rs")),
    ("sqlite/admin.rs", include_str!("../admin.rs")),
    ("sqlite/embeddings.rs", include_str!("../embeddings.rs")),
    ("sqlite/search.rs", include_str!("../search.rs")),
    ("sqlite/ingest.rs", include_str!("../ingest.rs")),
    ("sqlite/lifecycle.rs", include_str!("../lifecycle.rs")),
    ("sqlite/merge.rs", include_str!("../merge.rs")),
    ("sqlite/purge.rs", include_str!("../purge.rs")),
    ("sqlite/sync.rs", include_str!("../sync.rs")),
    ("sqlite/records.rs", include_str!("../records.rs")),
    ("sqlite/connectors.rs", include_str!("../connectors.rs")),
    ("sqlite/store_trait.rs", include_str!("../store_trait.rs")),
    ("sqlite/tests/mod.rs", include_str!("mod.rs")),
    ("sqlite/tests/admin.rs", include_str!("admin.rs")),
    ("sqlite/tests/connectors.rs", include_str!("connectors.rs")),
    ("sqlite/tests/embeddings.rs", include_str!("embeddings.rs")),
    ("sqlite/tests/ingest.rs", include_str!("ingest.rs")),
    ("sqlite/tests/lifecycle.rs", include_str!("lifecycle.rs")),
    ("sqlite/tests/merge.rs", include_str!("merge.rs")),
    ("sqlite/tests/purge.rs", include_str!("purge.rs")),
    ("sqlite/tests/records.rs", include_str!("records.rs")),
    ("sqlite/tests/search.rs", include_str!("search.rs")),
    (
        "sqlite/tests/store_trait.rs",
        include_str!("store_trait.rs"),
    ),
    ("sqlite/tests/sync.rs", include_str!("sync.rs")),
    ("sqlite/tests/lint.rs", include_str!("lint.rs")),
    ("migrations.rs", include_str!("../../migrations.rs")),
    ("trace_store.rs", include_str!("../../trace_store.rs")),
    ("synaptic_store.rs", include_str!("../../synaptic_store.rs")),
    ("replay_store.rs", include_str!("../../replay_store.rs")),
    (
        "attestation_store.rs",
        include_str!("../../attestation_store.rs"),
    ),
    (
        "unlearning_store.rs",
        include_str!("../../unlearning_store.rs"),
    ),
    ("memory_store.rs", include_str!("../../memory_store.rs")),
    ("portable.rs", include_str!("../../portable.rs")),
];

/// Modules whose writers must additionally route through the shared
/// helper, so a BUSY past the busy timeout is retried and logged rather
/// than surfacing to the caller on the first refusal. Beginning IMMEDIATE
/// by hand is correct but silent: it takes the write lock up front and
/// then gives up on the first refusal past the 5 s busy timeout, with
/// nothing in the log to say a writer lost a race. Every `sqlite/` file is
/// helper-routed (see [`helper_routed`]); these are the siblings that are.
///
/// The needle matches the single-line form production used (the guard
/// receiver and the behaviour call on one line). Test fixtures that
/// genuinely need to drive a transaction by hand (a rollback or
/// lock-contention harness on their own connection) build it across lines
/// and are deliberately not caught. Note this comment cannot spell the
/// needle out: the lint reads this file, so a literal spelling would flag
/// itself, which is exactly what it did on the first draft of this text.
const HELPER_ROUTED_SIBLINGS: [&str; 4] = [
    "trace_store.rs",
    "synaptic_store.rs",
    "replay_store.rs",
    "attestation_store.rs",
];

fn helper_routed(name: &str) -> bool {
    name.starts_with("sqlite/") || HELPER_ROUTED_SIBLINGS.contains(&name)
}

/// Production transactions propagate with `?`; test fixtures `.unwrap()`
/// on their own in-memory connections. The `?` suffix is what separates
/// the two here, and it is the convention the storage layer already uses.
#[test]
fn writer_transactions_begin_immediate() {
    // Assembled at runtime so this lint never matches its own source lines.
    let deferred_writer = ["writer.", "transaction()?"].concat();
    let deferred_unchecked = ["unchecked_", "transaction()?"].concat();
    let bypasses_helper = ["writer.", "transaction_with_behavior("].concat();
    let snapshot_on_writer = ["begin_read_", "snapshot(&writer"].concat();

    let mut offenders: Vec<String> = Vec::new();
    for (name, source) in STORAGE_SOURCES {
        for (index, line) in source.lines().enumerate() {
            let number = index + 1;
            if line.contains(&deferred_writer) || line.contains(&deferred_unchecked) {
                offenders.push(format!(
                    "{name}:{number} opens a DEFERRED writer transaction; a read-then-write \
                     DEFERRED transaction can fail with SQLITE_BUSY_SNAPSHOT and SQLite does \
                     not consult the busy handler for that upgrade"
                ));
            }
            if helper_routed(name) && line.contains(&bypasses_helper) {
                offenders.push(format!(
                    "{name}:{number} opens a writer transaction directly; use \
                     SqliteMemoryStore::begin_write_transaction so BUSY retries are logged"
                ));
            }
            if line.contains(&snapshot_on_writer) {
                offenders.push(format!(
                    "{name}:{number} opens a DEFERRED read snapshot on the writer connection; \
                     snapshots belong on the reader, writers begin IMMEDIATE"
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "writer transactions must begin IMMEDIATE:\n{}",
        offenders.join("\n")
    );
}

/// The scan above is only as wide as [`STORAGE_SOURCES`]. Every `.rs` file
/// under `storage/sqlite/` (this directory and `tests/`) must be listed, or a
/// new module is a new blind spot.
#[test]
fn every_sqlite_module_is_scanned() {
    let sqlite_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/storage/sqlite");
    let mut on_disk: Vec<String> = Vec::new();
    for (dir, prefix) in [
        (sqlite_dir.clone(), "sqlite/"),
        (sqlite_dir.join("tests"), "sqlite/tests/"),
    ] {
        for entry in std::fs::read_dir(&dir).expect("storage/sqlite is readable") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "rs") {
                on_disk.push(format!(
                    "{prefix}{}",
                    path.file_name().unwrap().to_string_lossy()
                ));
            }
        }
    }
    on_disk.sort();
    let mut listed: Vec<String> = STORAGE_SOURCES
        .iter()
        .map(|(name, _)| name.to_string())
        .filter(|name| name.starts_with("sqlite/"))
        .collect();
    listed.sort();
    assert_eq!(
        on_disk, listed,
        "every file under storage/sqlite/ must appear in STORAGE_SOURCES"
    );
}

#[test]
fn the_write_transaction_helper_exists_and_is_shared() {
    let source = include_str!("../mod.rs");
    assert!(
        source.contains("fn begin_write_transaction"),
        "the write-transaction helper must exist in sqlite/mod.rs"
    );
    // Sibling storage modules are not descendants of this one, so the
    // helper has to stay at least `pub(super)` for them to reach it. That
    // only means `storage` while the helper lives in `sqlite/mod.rs`; from a
    // submodule, `pub(super)` would mean `sqlite` and the siblings would
    // lose it.
    assert!(
        source.contains(
            ["pub(super) fn begin_write_", "transaction"]
                .concat()
                .as_str()
        ),
        "the helper must stay visible to sibling storage modules"
    );
}

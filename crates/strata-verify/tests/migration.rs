//! PR-0a blocker 3: strata-verify must pass on an untouched migrated log
//! and fail on flipped bytes, dropped frames, truncation, or the wrong key.

use std::path::Path;

use strata_verify::migration::verify_migrated_log;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite"
);

fn build_log(dir: &Path) {
    std::fs::write(dir.parent().unwrap().join("receipt-signing.key"), [7u8; 32]).unwrap();
    strata_migrate::migrate_with_options(
        Path::new(FIXTURE),
        dir,
        strata_migrate::MigrateOptions {
            seed: Some([7u8; 32]),
            ..Default::default()
        },
    )
    .expect("migration for the verify fixture");
}

/// v3.1.1 fixture: inferred links are `legacy_inferred`, counts match the
/// source, the fixture sha256 is unchanged, and `verify_path` passes.
#[test]
fn v3_fixture_inferred_edges_are_legacy_inferred() {
    let sha_before = sha256_file(Path::new(FIXTURE));
    assert_eq!(
        sha_before,
        "961f12d1750dbd2f6e6a8fc365c4a4bb42dd1b7e49cbf465985a36b665e10479"
    );
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    assert_eq!(sha256_file(Path::new(FIXTURE)), sha_before);

    let report = strata_verify::verify_path(&log_dir);
    assert!(
        report.ok,
        "strata-verify must pass: {}",
        report.failures.join("; ")
    );

    let snapshot =
        strata_migrate::read_snapshot(&strata::StrataLog::open(&log_dir).unwrap()).unwrap();
    let source_edges = snapshot
        .receipt
        .as_ref()
        .expect("receipt")
        .body
        .counts
        .iter()
        .find(|(name, _)| name == "memory_connections")
        .expect("memory_connections count")
        .1;
    assert_eq!(snapshot.edges.len() as u64, source_edges);
    assert_eq!(source_edges, 3);
    let causal = strata_migrate::STRATA_EDGE_VOCABULARY;
    for edge in &snapshot.edges {
        if edge.legacy_inferred {
            assert_eq!(edge.link_type, strata_migrate::LEGACY_INFERRED_KIND);
            assert!(!causal.contains(&edge.link_type.as_str()));
        } else {
            assert!(causal.contains(&edge.link_type.as_str()));
        }
    }
    assert_eq!(
        snapshot.edges.iter().filter(|e| e.legacy_inferred).count(),
        2
    );
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("read for sha256");
    let digest = Sha256::digest(&bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Untouched log: chain + receipt checksum + signature + counts all pass.
#[test]
fn migrated_log_untouched_passes() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    let report = verify_migrated_log(&log_dir).expect("verification runs");
    assert!(
        report.failures.is_empty(),
        "untouched log must verify: {report:?}"
    );
    assert!(report.checksum_ok && report.signature_ok && report.counts_match);
}

/// A flipped byte in a sealed segment halts the log open (chain check).
#[test]
fn migrated_log_flipped_byte_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    // Flip one byte in the first (sealed) segment.
    let seg = std::fs::read_dir(&log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "seg"))
        .expect("segment file");
    let mut bytes = std::fs::read(&seg).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&seg, &bytes).unwrap();
    assert!(
        verify_migrated_log(&log_dir).is_err(),
        "a flipped byte must fail verification"
    );
}

/// Truncation of the log fails the open.
#[test]
fn migrated_log_truncated_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    let seg = std::fs::read_dir(&log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "seg"))
        .expect("segment file");
    let bytes = std::fs::read(&seg).unwrap();
    std::fs::write(&seg, &bytes[..bytes.len() / 2]).unwrap();
    assert!(
        verify_migrated_log(&log_dir).is_err(),
        "truncation must fail verification"
    );
}

/// A receipt whose verifying key does not match the signature (the
/// wrong-key forge) fails the authorship check.
#[test]
fn migrated_log_wrong_receipt_key_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    // Forge at the API level: a receipt that claims someone else's key.
    let log = strata::StrataLog::open(&log_dir).unwrap();
    let frames = log.read_frames(1).unwrap();
    let receipt_frame = frames
        .iter()
        .find(|f| f.kind == strata_migrate::records::KIND_MIGRATION_RECEIPT)
        .expect("receipt frame");
    let mut receipt = strata_migrate::records::decode_receipt(&receipt_frame.payload).unwrap();
    let other = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
    receipt.verifying_key = other.verifying_key().to_bytes();
    assert!(
        !receipt.verify_signature(),
        "a receipt claiming the wrong key must fail authorship"
    );
    assert!(receipt.verify_checksum(), "checksum stays key-independent");
}

/// The upgrade admits carried intentions after the receipt, and every later
/// store write lands there too. Kinds 0x20/0x21 after the receipt are
/// STORE_WRITE/STORE_CHECKPOINT, so the receipt's counts still match.
#[test]
fn store_writes_after_the_receipt_keep_the_counts() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("log");
    build_log(&log_dir);
    let mut store = strata_store::StrataStore::open(dir.path()).unwrap();
    store
        .upsert_intentions(vec![strata_store::IntentionRecord {
            id: "int-after-receipt".into(),
            content: "carried".into(),
            trigger_type: "manual".into(),
            trigger_data: "{}".into(),
            priority: 2,
            status: "active".into(),
            created_at_ms: 1_700_000_000_000,
            deadline_ms: None,
            fulfilled_at_ms: None,
            reminder_count: 0,
            last_reminded_at_ms: None,
            notes: None,
            tags: Vec::new(),
            related_memories: Vec::new(),
            snoozed_until_ms: None,
            source_type: "mcp".into(),
            source_data: None,
            scope: None,
        }])
        .unwrap();
    store
        .ingest(strata_store::IngestInput {
            content: "written after the upgrade".into(),
            ..Default::default()
        })
        .unwrap();
    drop(store);

    let report = verify_migrated_log(&log_dir).expect("verification runs");
    assert!(report.ok, "{report:?}");
    assert!(report.counts_match);
    assert!(strata_verify::verify_path(&log_dir).ok);
}

/// `strata-verify <data-dir>` is the form the README documents. For an
/// upgraded store it must check the migration receipt in `log/` exactly as
/// `strata-verify <data-dir>/log` does. It used to scan only the chain, so a
/// swapped receipt-signing key beside the log still printed OK.
#[test]
fn data_dir_form_verifies_the_migration_receipt() {
    let dir = tempfile::tempdir().unwrap();
    build_log(&dir.path().join("log"));

    let report = strata_verify::verify_path(dir.path());
    assert!(report.ok, "{}", report.failures.join("; "));
    let json: serde_json::Value = serde_json::from_str(&report.json).unwrap();
    assert_eq!(json["signature_ok"], true, "{}", report.json);
    assert_eq!(json["counts_match"], true, "{}", report.json);
    assert_eq!(json["key_pin"], "receipt-signing.key", "{}", report.json);

    std::fs::write(dir.path().join("receipt-signing.key"), [9u8; 32]).unwrap();
    let swapped = strata_verify::verify_path(dir.path());
    assert!(
        !swapped.ok,
        "a receipt key that no longer matches the pin must fail the data-dir form: {}",
        swapped.json
    );
}

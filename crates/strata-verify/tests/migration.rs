//! PR-0a blocker 3: strata-verify must pass on an untouched migrated log
//! and fail on flipped bytes, dropped frames, truncation, or the wrong key.

use std::io::Read;
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

/// The log signing key on disk does not verify the segment trailers.
#[test]
fn migrated_log_wrong_key_on_disk_fails() {
    let dir = tempfile::tempdir().unwrap();
    let log_dir = dir.path().join("strata");
    build_log(&log_dir);
    let key_path = strata_migrate::log_signing_key_path(&log_dir);
    let original = std::fs::read(&key_path).expect("external log key");
    assert_eq!(original.len(), 32);
    assert!(
        !log_dir.join("strata.key").exists(),
        "the private key must not live inside --to"
    );
    let mut wrong = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut wrong)
        .unwrap();
    if wrong.as_slice() == original.as_slice() {
        wrong[0] ^= 0xff;
    }
    std::fs::write(&key_path, wrong).unwrap();
    let err = verify_migrated_log(&log_dir).expect_err("wrong on-disk key must fail");
    assert!(
        !err.contains("kernel.log"),
        "wrong key is a log failure, not a missing kernel.log: {err}"
    );
}

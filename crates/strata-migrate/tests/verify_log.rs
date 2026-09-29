//! Sealed-segment verification: frames_verified covers every frame, and a
//! flipped byte, a deleted frame, or a truncated segment is a hard error.

use std::path::{Path, PathBuf};

use strata_migrate::{migrate_with_options, verify_migrated_dir, MigrateOptions};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v3.1.1-sample.sqlite")
}

fn migrated() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("src.sqlite");
    std::fs::copy(fixture(), &db).unwrap();
    let dest = dir.path().join("strata");
    let report = migrate_with_options(
        &db,
        &dest,
        MigrateOptions {
            seed: Some([9u8; 32]),
            ..Default::default()
        },
    )
    .expect("migration");
    assert!(report.verify_passed);
    assert!(report.frames_total > 0, "the log must contain frames");
    assert_eq!(
        report.frames_verified, report.frames_total,
        "frames_verified must cover every frame, not the empty post-seal segment"
    );
    (dir, dest)
}

fn sealed_segment(dir: &Path) -> PathBuf {
    let mut segs: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "seg"))
        .collect();
    segs.sort();
    segs.into_iter()
        .find(|p| std::fs::metadata(p).unwrap().len() > 58 + 104)
        .expect("a sealed segment with a trailer")
}

#[test]
fn untouched_log_frames_verified_equals_total() {
    let (_dir, dest) = migrated();
    let again = verify_migrated_dir(&dest).expect("reopen verify");
    assert_eq!(again.frames_verified, again.frames_total);
    assert!(again.frames_verified > 0);
}

#[test]
fn flipped_byte_in_sealed_segment_fails() {
    let (_dir, dest) = migrated();
    let seg = sealed_segment(&dest);
    let mut bytes = std::fs::read(&seg).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&seg, bytes).unwrap();
    let err = verify_migrated_dir(&dest).expect_err("flipped byte");
    let msg = err.to_string();
    assert!(
        msg.contains("verification") || msg.contains("log open") || msg.contains("damaged"),
        "{msg}"
    );
}

#[test]
fn deleted_frame_fails() {
    let (_dir, dest) = migrated();
    let seg = sealed_segment(&dest);
    let bytes = std::fs::read(&seg).unwrap();
    // Drop a slice out of the frame region (after the 58-byte header, before
    // the 104-byte trailer) so one frame is gone and the chain cannot close.
    let cut = 58 + 40;
    let mut torn = bytes[..cut].to_vec();
    torn.extend_from_slice(&bytes[cut + 30..]);
    std::fs::write(&seg, torn).unwrap();
    assert!(
        verify_migrated_dir(&dest).is_err(),
        "a deleted frame must fail verification"
    );
}

#[test]
fn truncated_segment_fails() {
    let (_dir, dest) = migrated();
    let seg = sealed_segment(&dest);
    let bytes = std::fs::read(&seg).unwrap();
    std::fs::write(&seg, &bytes[..bytes.len() / 2]).unwrap();
    assert!(
        verify_migrated_dir(&dest).is_err(),
        "truncation must fail verification"
    );
}

//! Sealed-segment verification: frames_verified covers every frame, and a
//! flipped byte, a deleted frame, or a truncated segment is a hard error.

use std::path::{Path, PathBuf};

use std::os::unix::fs::PermissionsExt;

use strata_migrate::{
    log_signing_key_path, migrate_with_options, open_migrated, verify_migrated_dir, MigrateOptions,
};

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
fn log_key_lives_outside_dest_and_pubkey_is_in_params() {
    let (dir, dest) = migrated();
    assert!(
        !dest.join("strata.key").exists(),
        "private log key must not sit inside --to"
    );
    let key_path = log_signing_key_path(&dest);
    assert!(key_path.exists());
    assert!(
        !key_path.starts_with(&dest),
        "key path {} is inside {}",
        key_path.display(),
        dest.display()
    );
    let mode = std::fs::metadata(&key_path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "log key mode {mode:o}");
    let seed: [u8; 32] = std::fs::read(&key_path).unwrap().try_into().unwrap();
    let vk = ed25519_dalek::SigningKey::from_bytes(&seed)
        .verifying_key()
        .to_bytes();
    let opened = open_migrated(&dest).unwrap();
    let params = strata_migrate::read_snapshot(&opened.log)
        .unwrap()
        .params
        .unwrap();
    assert_eq!(params.log_verifying_key, vk);
    let _ = dir;
}

#[test]
fn source_hash_is_not_the_log_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("src.sqlite");
    std::fs::copy(fixture(), &db).unwrap();
    let dest = dir.path().join("strata");
    let report = migrate_with_options(&db, &dest, MigrateOptions::default()).unwrap();
    let key = std::fs::read(log_signing_key_path(&dest)).unwrap();
    let mut hash = [0u8; 32];
    for i in 0..32 {
        hash[i] = u8::from_str_radix(&report.source_blake3[i * 2..i * 2 + 2], 16).unwrap();
    }
    assert_ne!(key.as_slice(), &hash);
    std::fs::write(log_signing_key_path(&dest), hash).unwrap();
    assert!(
        verify_migrated_dir(&dest).is_err(),
        "a key recomputed from the published source hash must not verify"
    );
}

#[test]
fn two_runs_differ_in_key_not_in_data_frames() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let db_a = dir_a.path().join("src.sqlite");
    let db_b = dir_b.path().join("src.sqlite");
    std::fs::copy(fixture(), &db_a).unwrap();
    std::fs::copy(fixture(), &db_b).unwrap();
    let dest_a = dir_a.path().join("strata");
    let dest_b = dir_b.path().join("strata");
    migrate_with_options(&db_a, &dest_a, MigrateOptions::default()).unwrap();
    migrate_with_options(&db_b, &dest_b, MigrateOptions::default()).unwrap();
    let key_a = std::fs::read(log_signing_key_path(&dest_a)).unwrap();
    let key_b = std::fs::read(log_signing_key_path(&dest_b)).unwrap();
    assert_ne!(key_a, key_b, "OS entropy must not repeat the log key");

    let open_a = open_migrated(&dest_a).unwrap();
    let open_b = open_migrated(&dest_b).unwrap();
    let frames_a = open_a.log.read_frames(1).unwrap();
    let frames_b = open_b.log.read_frames(1).unwrap();
    let data = |frames: &[strata::FrameRecord]| {
        frames
            .iter()
            .filter(|f| {
                f.kind != strata_migrate::records::KIND_PARAMS
                    && f.kind != strata_migrate::records::KIND_MIGRATION_RECEIPT
            })
            .map(|f| (f.kind, f.payload.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(data(&frames_a), data(&frames_b));
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

//! A live strata-store folder verifies without a `kernel.log`.

use std::process::Command;

use strata_store::{IngestInput, StrataStore};

fn populate(dir: &std::path::Path) {
    let mut store = StrataStore::open(dir).expect("open store");
    store
        .ingest(IngestInput {
            content: "causal proof lives in the segment log".into(),
            ..Default::default()
        })
        .expect("ingest");
    store.seal_checkpoint().expect("checkpoint");
    store.log().seal().expect("seal segment");
}

#[test]
fn live_store_folder_verifies() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("store");
    populate(&dir);
    assert!(!dir.join("kernel.log").exists());
    assert!(dir.join("store.meta").is_file());
    assert!(dir.join("log").is_dir());

    let report = strata_verify::live::verify_live_store(&dir).expect("verify");
    assert!(report.ok, "{report:?}");
    assert!(report.sweep_clear && report.verdicts_match);
    assert!(report.frames_total > 0);

    let bin = env!("CARGO_BIN_EXE_strata-verify");
    let out = Command::new(bin).arg(&dir).output().expect("run cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "live store must verify\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("OK"), "{stdout}");
    assert!(!stderr.contains("kernel.log"), "{stderr}");
}

#[test]
fn live_store_flipped_sealed_segment_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("store");
    populate(&dir);
    let log_dir = dir.join("log");
    let mut segs: Vec<_> = std::fs::read_dir(&log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "seg"))
        .collect();
    segs.sort();
    assert!(
        segs.len() >= 2,
        "seal leaves a sealed segment and an empty tail"
    );
    let sealed = &segs[0];
    let mut bytes = std::fs::read(sealed).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(sealed, &bytes).unwrap();
    assert!(
        strata_verify::live::verify_live_store(&dir).is_err(),
        "a flipped sealed segment must fail"
    );
}

#[test]
fn cli_accepts_migrated_log() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir = tmp.path().join("strata");
    std::fs::write(tmp.path().join("receipt-signing.key"), [7u8; 32]).unwrap();
    strata_migrate::migrate_with_options(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite"
        )),
        &log_dir,
        strata_migrate::MigrateOptions {
            seed: Some([7u8; 32]),
            ..Default::default()
        },
    )
    .expect("migrate");
    let bin = env!("CARGO_BIN_EXE_strata-verify");
    let out = Command::new(bin).arg(&log_dir).output().expect("run cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "migrated log must verify\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("OK"), "{stdout}");
}

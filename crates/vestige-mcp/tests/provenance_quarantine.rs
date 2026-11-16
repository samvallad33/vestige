//! #774 against a real Strata log: quarantine, release, identical re-run.
//!
//! The iroh half is `mesh_sync::orphan_quarantine_syncs_over_loopback_iroh_and_releases_to_one_mesh_digest`.

#![forbid(unsafe_code)]

use strata_store::StrataStore;
use vestige_mcp::provenance_log::{quarantine_store, release_store, status_store};

const SUMMARY: &str = "orphan memory with no anchor";
const ANCHOR: &str = "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b";

fn open() -> (tempfile::TempDir, StrataStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = StrataStore::open(dir.path()).unwrap();
    (dir, store)
}

fn payloads(store: &StrataStore) -> Vec<(u8, Vec<u8>)> {
    store
        .log()
        .read_frames(1)
        .unwrap()
        .into_iter()
        .map(|frame| (frame.kind, frame.payload))
        .collect()
}

#[test]
fn orphan_is_quarantined_then_freed_without_deleting_the_receipt() {
    let (_dir, mut store) = open();
    let first = quarantine_store(&mut store, SUMMARY).unwrap();
    assert_eq!(first["state"], "quarantined");
    assert_eq!(first["appended"], true);
    let receipt = first["receipt_digest"].as_str().unwrap().to_string();
    let digest = first["mesh_digest"].as_str().unwrap().to_string();

    let status = status_store(&store);
    assert_eq!(status["quarantined"][0]["receipt_digest"], receipt);
    assert_eq!(status["quarantined"][0]["summary"], SUMMARY);
    assert_eq!(status["retained_quarantine_bodies"], 1);
    assert!(status["released"].as_array().unwrap().is_empty());

    let second = quarantine_store(&mut store, SUMMARY).unwrap();
    assert_eq!(second["appended"], false);
    assert_eq!(second["receipt_digest"], receipt);
    assert_eq!(second["mesh_digest"], digest);

    let released = release_store(&mut store, &receipt, ANCHOR).unwrap();
    assert_eq!(released["state"], "released");
    assert_eq!(released["appended"], true);
    assert_ne!(released["mesh_digest"], digest);

    let after = status_store(&store);
    assert!(after["quarantined"].as_array().unwrap().is_empty());
    assert_eq!(after["retained_quarantine_bodies"], 1);
    assert_eq!(after["released"][0]["receipt_digest"], receipt);
    assert_eq!(after["released"][0]["anchor_digest"], ANCHOR);
    assert_eq!(after["released"][0]["summary"], SUMMARY);
    assert!(store.nodes().iter().any(|node| {
        node.is_live() && node.content.contains("vestige.provenance.quarantine.v1")
    }));
    assert!(
        store
            .nodes()
            .iter()
            .any(|node| node.is_live() && node.content == SUMMARY)
    );

    let again = release_store(&mut store, &receipt, ANCHOR).unwrap();
    assert_eq!(again["appended"], false);
    assert_eq!(again["mesh_digest"], released["mesh_digest"]);
}

#[test]
fn two_fresh_stores_write_identical_quarantine_payloads() {
    let (_left_dir, mut left) = open();
    let (_right_dir, mut right) = open();
    quarantine_store(&mut left, SUMMARY).unwrap();
    quarantine_store(&mut right, SUMMARY).unwrap();
    let left_payloads = payloads(&left);
    let right_payloads = payloads(&right);
    assert_eq!(left_payloads, right_payloads);
    assert!(
        left_payloads
            .iter()
            .any(|(kind, payload)| *kind == strata_store::KIND_STORE_WRITE && !payload.is_empty())
    );
    assert_eq!(left.nodes(), right.nodes());
}

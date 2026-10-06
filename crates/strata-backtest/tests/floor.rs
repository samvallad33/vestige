//! Copy-only open, cut manifest, and the origin firewall.

use std::fs;
use std::path::Path;

use strata_backtest::canon::hex_bytes;
use strata_backtest::cuts::eligible_bounds;
use strata_backtest::manifest::prereg_blake3;
use strata_backtest::mechanism::{IdOrder, Mechanism, Query, admitted_events, project_prefix};
use strata_backtest::{TABLE_ID, harness_manifest, open_copy, require_firewall};
use strata_store::{ConnectionRecord, EdgeKind, IngestInput, StrataStore};

fn script(store: &mut StrataStore) {
    let mut ids = Vec::new();
    for i in 0..4 {
        let id = store
            .ingest_in_scope(
                IngestInput {
                    content: format!("n{i}"),
                    created_at_ms: Some(1_700_000_000_000 + i),
                    ..IngestInput::default()
                },
                "floor",
            )
            .unwrap();
        ids.push(id);
    }
    for i in 0..4 {
        store
            .save_connection(&ConnectionRecord {
                source_id: ids[i].clone(),
                target_id: ids[(i + 1) % 4].clone(),
                link_type: EdgeKind::DerivedFrom.as_str().to_string(),
                created_at_ms: 1_700_000_000_000,
                ..ConnectionRecord::default()
            })
            .unwrap();
    }
    store
        .review_at(&ids[3], 4, Some(1_700_000_000_000))
        .unwrap();
    store
        .review_at(&ids[0], 3, Some(1_700_000_003_600))
        .unwrap();
}

fn tree_digest(root: &Path) -> [u8; 32] {
    let mut files = Vec::new();
    fn walk(path: &Path, root: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            if entry.file_name() == "strata.lock" {
                continue;
            }
            let child = entry.path();
            if child.is_dir() {
                walk(&child, root, out);
            } else {
                let rel = child
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, fs::read(&child).unwrap()));
            }
        }
    }
    walk(root, root, &mut files);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (rel, bytes) in files {
        for byte in rel.bytes().chain(bytes) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash.to_le_bytes().repeat(4).try_into().unwrap()
}

#[test]
fn seeded_copies_pin_the_same_manifest() {
    let prereg = include_bytes!("../../../docs/benchmarks/MATTAR-EVB-PREREGISTRATION.md");
    let seed = [0x5A_u8; 32];
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    {
        let mut store = StrataStore::open_seeded(left.path(), seed).unwrap();
        script(&mut store);
    }
    {
        let mut store = StrataStore::open_seeded(right.path(), seed).unwrap();
        script(&mut store);
    }
    let before = tree_digest(left.path());
    let mut copied = open_copy(left.path()).unwrap();
    let other = open_copy(right.path()).unwrap();
    let manifest = harness_manifest(&copied.store, prereg, &[]).unwrap();
    let again = harness_manifest(&other.store, prereg, &[]).unwrap();
    assert_eq!(manifest, again);
    assert!(manifest.starts_with("{\"cuts\":[{"));
    assert!(manifest.ends_with('\n'));
    assert!(manifest.contains(&format!("\"table_id\":\"{TABLE_ID}\"")));
    let hash = hex_bytes(&prereg_blake3(prereg));
    assert!(manifest.contains(&hash));

    let events = admitted_events(&copied.store).unwrap();
    let bounds = eligible_bounds(&events, &[]);
    assert!(!bounds.is_empty());
    let fold = copied.store.as_of(bounds[0]).unwrap();
    assert!(manifest.contains(&hex_bytes(&fold.state_digest())));
    assert!(manifest.contains(&hex_bytes(&fold.prefix_head_frame_hash())));
    let prefix = project_prefix(&fold, &events, bounds[0], &[]).unwrap();
    let ranked = IdOrder.rank(
        &prefix,
        &Query {
            corpus_id: "floor".into(),
            bound_seq: bounds[0],
            horizon: strata_backtest::protocol::HORIZON,
        },
    );
    require_firewall(bounds[0], &ranked).unwrap();
    assert!(ranked.len() >= 4);

    let mut scratch = copied.store.as_of(bounds[0]).unwrap();
    let refused = scratch.ingest(IngestInput {
        content: "nope".into(),
        ..IngestInput::default()
    });
    assert!(refused.unwrap_err().to_string().contains("read-only"));

    copied
        .store
        .ingest(IngestInput {
            content: "on the copy".into(),
            ..IngestInput::default()
        })
        .unwrap();
    assert_eq!(tree_digest(left.path()), before);
}

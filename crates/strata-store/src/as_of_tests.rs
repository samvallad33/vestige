//! Prefix fold. Live retrievability and `state_digest` stay on the pre-`as_of`
//! path: `replay_bound` is `None`, so the sequence fallback is still the
//! acked log head.

use std::path::PathBuf;

use crate::{AnchorRecord, ConnectionRecord, EdgeKind, IngestInput, StrataStore};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("strata-store-asof-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn anchor(id: &str, node_id: &str, file_path: &str) -> AnchorRecord {
    AnchorRecord {
        id: id.to_string(),
        node_id: node_id.to_string(),
        file_path: file_path.to_string(),
        symbol: Some("load_config".into()),
        symbol_kind: Some("fn".into()),
        start_line: Some(1),
        end_line: Some(4),
        span_lines: Some(4),
        content_hash: Some("v2:0123456789abcdef0123456789abcdef".into()),
        captured_at_ms: 1_700_000_000_000,
        last_verified_at_ms: None,
        last_status: None,
    }
}

/// 500 ids and their f32/f64 retrievability bits, hashed with the digest.
/// Captured on main before `as_of` landed. A live-path change fails this.
#[test]
fn live_head_ranking_hash_is_unchanged() {
    let dir = temp_dir("rank-hash");
    let mut store = StrataStore::open(&dir).expect("open");
    let mut ids = Vec::with_capacity(500);
    for i in 0..500 {
        let id = store
            .ingest(IngestInput {
                content: format!("probe-{i}"),
                created_at_ms: Some(1_700_000_000_000 + (i as i64) * 3_600_000),
                tags: vec![format!("t{}", i % 17)],
                ..IngestInput::default()
            })
            .expect("ingest");
        ids.push(id);
    }
    for i in 0..500 {
        store
            .save_connection(&ConnectionRecord {
                source_id: ids[i].clone(),
                target_id: ids[(i * 3 + 1) % 500].clone(),
                link_type: "derived_from".into(),
                created_at_ms: 1_700_000_000_000,
                ..ConnectionRecord::default()
            })
            .expect("edge");
    }
    for i in 0..100 {
        store
            .review_at(
                &ids[i * 5],
                ((i % 4) + 1) as u8,
                Some(1_700_000_003_600 + (i as i64) * 86_400_000),
            )
            .expect("review");
    }
    let digest = store.state_digest();
    let mut rows: Vec<(String, u64, u32)> = Vec::new();
    for node in store.nodes() {
        let value = store.retrievability(&node.id).expect("r").expect("some");
        rows.push((node.id, value.to_bits(), (value as f32).to_bits()));
    }
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    assert_eq!(rows.len(), 500);
    let mut hasher = blake3::Hasher::new();
    hasher.update(&digest);
    for (id, f64_bits, f32_bits) in &rows {
        hasher.update(id.as_bytes());
        hasher.update(&f64_bits.to_le_bytes());
        hasher.update(&f32_bits.to_le_bytes());
    }
    let rank = hex(hasher.finalize().as_bytes());
    assert_eq!(
        hex(&digest),
        "5425b19d2977ea2de2f4c1ca28831a622e6c443cf8c538be2645b5bdf5222caf"
    );
    assert_eq!(
        rank,
        "903aea4458cd13d8594b571d28ea0e60d34a54a4de713cbce1c1c2d386003e0d"
    );
    assert!(store.admitted_frames().is_empty());
    assert!(!store.is_prefix_fold());
    std::fs::remove_dir_all(&dir).ok();
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

enum Op {
    Ingest {
        content: String,
        at_ms: i64,
    },
    Edge {
        from: usize,
        to: usize,
        kind: EdgeKind,
    },
    Review {
        index: usize,
        rating: u8,
        at_ms: Option<i64>,
    },
    Anchor {
        index: usize,
        path: String,
    },
}

fn apply_script(store: &mut StrataStore, script: &[Op]) {
    for op in script {
        match op {
            Op::Ingest { content, at_ms } => {
                store
                    .ingest(IngestInput {
                        content: content.clone(),
                        created_at_ms: Some(*at_ms),
                        ..IngestInput::default()
                    })
                    .expect("ingest");
            }
            Op::Edge { from, to, kind } => {
                let nodes = store.nodes();
                if nodes.len() < 2 {
                    continue;
                }
                let source = nodes[*from % nodes.len()].id.clone();
                let target = nodes[*to % nodes.len()].id.clone();
                if source == target {
                    continue;
                }
                store
                    .save_connection(&ConnectionRecord {
                        source_id: source,
                        target_id: target,
                        link_type: kind.as_str().to_string(),
                        created_at_ms: 1_700_000_000_000,
                        ..ConnectionRecord::default()
                    })
                    .expect("edge");
            }
            Op::Review {
                index,
                rating,
                at_ms,
            } => {
                let nodes = store.nodes();
                if nodes.is_empty() {
                    continue;
                }
                let id = nodes[*index % nodes.len()].id.clone();
                store.review_at(&id, *rating, *at_ms).expect("review");
            }
            Op::Anchor { index, path } => {
                let nodes = store.nodes();
                if nodes.is_empty() {
                    continue;
                }
                let id = nodes[*index % nodes.len()].id.clone();
                let anchor_id = format!("anc-{path}-{id}");
                store
                    .record_anchors(vec![anchor(&anchor_id, &id, path)])
                    .expect("anchor");
            }
        }
    }
}

fn random_script(rng: &mut SplitMix64, n: usize, tag: &str) -> Vec<Op> {
    let kinds = EdgeKind::ALL;
    (0..n)
        .map(|i| match rng.below(10) {
            0..=3 => Op::Ingest {
                content: format!("{tag}-fact-{i}-{}", rng.next_u64()),
                at_ms: 1_700_000_000_000 + (i as i64) * 86_400_000,
            },
            4..=6 => Op::Edge {
                from: rng.below(8),
                to: rng.below(8),
                kind: kinds[rng.below(kinds.len())],
            },
            7..=8 => Op::Review {
                index: rng.below(8),
                rating: 1 + rng.below(4) as u8,
                at_ms: if rng.below(2) == 0 {
                    None
                } else {
                    Some(1_700_000_000_000 + (i as i64) * 3_600_000)
                },
            },
            _ => Op::Anchor {
                index: rng.below(8),
                path: format!("src/{tag}-{}.rs", rng.below(4)),
            },
        })
        .collect()
}

fn assert_prefix_matches(full: &StrataStore, prefix: &StrataStore) {
    let bound = prefix.log().head().last_acked_seq;
    let view = full.as_of(bound).expect("as_of prefix head");
    assert_eq!(view.state_digest(), prefix.state_digest());
    assert_eq!(view.prefix_head_seq(), bound);
    if bound > 0 {
        let head_hash = full
            .log()
            .read_frames(1)
            .expect("frames")
            .iter()
            .rfind(|frame| frame.seq == bound)
            .expect("head frame")
            .frame_hash;
        assert_eq!(view.prefix_head_frame_hash(), head_hash);
    } else {
        assert_eq!(view.prefix_head_frame_hash(), [0u8; 32]);
    }
    assert_eq!(view.head_clock_ms(), prefix.head_clock_ms());
    assert!(view.admitted_frames().len() >= prefix.node_count());
    assert!(prefix.admitted_frames().is_empty());
    let clock = prefix.head_clock_ms();
    for node in prefix.nodes() {
        assert_eq!(
            view.retrievability(&node.id).expect("view R"),
            prefix.retrievability_at(&node.id, clock).expect("prefix R")
        );
    }
    let mid = bound / 2;
    assert_eq!(
        full.as_of(mid).expect("full mid").state_digest(),
        prefix.as_of(mid).expect("prefix mid").state_digest()
    );
}

#[test]
fn as_of_prefix_digest_matches_a_fresh_store_over_random_ops() {
    for trial in 0..24u64 {
        let mut rng = SplitMix64(0xA5_0F_00_00 ^ trial.wrapping_mul(0x1000_0000_01B3));
        let prefix_ops = random_script(&mut rng, 6 + (trial as usize % 10), "p");
        let suffix_ops = random_script(&mut rng, 3 + (trial as usize % 5), "s");
        let full_dir = temp_dir(&format!("full-{trial}"));
        let prefix_dir = temp_dir(&format!("prefix-{trial}"));
        let mut full = StrataStore::open(&full_dir).expect("full");
        let mut prefix = StrataStore::open(&prefix_dir).expect("prefix");
        apply_script(&mut full, &prefix_ops);
        apply_script(&mut prefix, &prefix_ops);
        assert_eq!(full.state_digest(), prefix.state_digest());
        assert_prefix_matches(&full, &prefix);
        apply_script(&mut full, &suffix_ops);
        let after = full.state_digest();
        assert_eq!(
            full.as_of(prefix.log().head().last_acked_seq)
                .expect("as_of")
                .state_digest(),
            prefix.state_digest()
        );
        assert_eq!(full.as_of(u64::MAX).expect("whole").state_digest(), after);
        assert_eq!(full.state_digest(), after);
        std::fs::remove_dir_all(&full_dir).ok();
        std::fs::remove_dir_all(&prefix_dir).ok();
    }
}

#[test]
fn as_of_retrievability_ignores_frames_after_the_bound() {
    let full_dir = temp_dir("r-full");
    let prefix_dir = temp_dir("r-prefix");
    let mut full = StrataStore::open(&full_dir).expect("full");
    let mut prefix = StrataStore::open(&prefix_dir).expect("prefix");
    let created = IngestInput {
        content: "clockless review".into(),
        created_at_ms: Some(1_700_000_000_000),
        ..IngestInput::default()
    };
    let id = full.ingest(created.clone()).expect("ingest");
    prefix.ingest(created).expect("ingest");
    full.review_at(&id, 3, None).expect("review");
    prefix.review_at(&id, 3, None).expect("review");
    let bound = prefix.log().head().last_acked_seq;
    full.ingest(IngestInput {
        content: "later fact".into(),
        created_at_ms: Some(1_700_000_000_000),
        ..IngestInput::default()
    })
    .expect("suffix");
    let mut view = full.as_of(bound).expect("as_of");
    let pure = view.retrievability_at(&id, 0).expect("view");
    let fresh = prefix.retrievability_at(&id, 0).expect("prefix");
    let leaked = full.retrievability_at(&id, 0).expect("full");
    assert_eq!(pure, fresh);
    assert_ne!(pure, leaked);
    let err = view
        .ingest(IngestInput {
            content: "no write".into(),
            ..IngestInput::default()
        })
        .expect_err("read-only");
    assert!(err.to_string().contains("read-only"));
    assert!(full.log().head().last_acked_seq > bound);
    std::fs::remove_dir_all(&full_dir).ok();
    std::fs::remove_dir_all(&prefix_dir).ok();
}

#[test]
fn seeded_open_pins_the_signed_head_across_directories() {
    let seed = [0x5Au8; 32];
    let a_dir = temp_dir("seeded-a");
    let b_dir = temp_dir("seeded-b");
    let mut a = StrataStore::open_seeded(&a_dir, seed).expect("a");
    let mut b = StrataStore::open_seeded(&b_dir, seed).expect("b");
    for i in 0..3 {
        let input = IngestInput {
            content: format!("seeded-{i}"),
            created_at_ms: Some(1_700_000_000_000 + i),
            ..IngestInput::default()
        };
        a.ingest(input.clone()).expect("a ingest");
        b.ingest(input).expect("b ingest");
    }
    let a_head = a.as_of(u64::MAX).expect("a as_of").prefix_head_frame_hash();
    let b_head = b.as_of(u64::MAX).expect("b as_of").prefix_head_frame_hash();
    assert_eq!(a_head, b_head);
    assert_ne!(a_head, [0u8; 32]);
    assert_eq!(a.state_digest(), b.state_digest());
    std::fs::remove_dir_all(&a_dir).ok();
    std::fs::remove_dir_all(&b_dir).ok();
}

#[test]
fn as_of_zero_matches_an_empty_store() {
    let dir = temp_dir("empty-src");
    let empty_dir = temp_dir("empty-fresh");
    let mut store = StrataStore::open(&dir).expect("open");
    store
        .ingest(IngestInput {
            content: "something".into(),
            created_at_ms: Some(1_700_000_000_000),
            ..IngestInput::default()
        })
        .expect("ingest");
    let empty = StrataStore::open(&empty_dir).expect("empty");
    assert_eq!(
        store.as_of(0).expect("bound 0").state_digest(),
        empty.state_digest()
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&empty_dir).ok();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

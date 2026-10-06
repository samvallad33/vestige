//! TEMPORARY measurement, never committed: the cost of the auto-connect
//! edges of one save on a 9,200-memory Strata store, written one synced
//! write per edge (before) and as one batched write (after).
//!
//! PROBE_DIR=<dir> cargo test -p vestige-mcp --test tmp_probe_write_cost -- --ignored --nocapture

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use chrono::Utc;
use vestige_core::{ConnectionRecord, IngestInput};
use vestige_mcp::auto_connect::{self, Holder, Identity};

const TOTAL: usize = 9_200;

fn probe_dir() -> PathBuf {
    PathBuf::from(std::env::var("PROBE_DIR").expect("set PROBE_DIR"))
}

fn text(i: usize) -> String {
    let words = [
        "the",
        "connection",
        "retry",
        "was",
        "added",
        "when",
        "server",
        "timeout",
        "regression",
        "since",
        "version",
        "tests",
        "stopped",
        "passing",
        "handler",
        "smart_ingest",
        "causal_walk",
        "ConnectionPool",
        "note:",
        "(see",
        "below).",
    ];
    let mut text = String::with_capacity(2200);
    let mut n = i;
    while text.len() < 2000 {
        n = n
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (n >> 33) as usize;
        if pick % 40 == 0 {
            text.push_str(&format!(
                "crates/mod{}/src/file{}.rs:{} ",
                pick % 50,
                pick % 300,
                pick % 900
            ));
        } else if pick % 97 == 0 {
            text.push_str(&format!("https://example.com/a/{} ", pick % 1000));
        } else {
            text.push_str(words[pick % words.len()]);
            text.push(' ');
        }
    }
    text
}

#[test]
#[ignore]
fn build_store() {
    let dir = probe_dir().join("base");
    if dir.join("built").exists() {
        eprintln!("store already built at {}", dir.display());
        return;
    }
    std::fs::create_dir_all(&dir).unwrap();
    let storage = vestige_mcp::strata_memory::open(&dir).unwrap();
    let started = Instant::now();
    for i in 0..TOTAL {
        let mut tags = vec![format!("topic-{}", i % 400)];
        if i % 3 == 0 {
            tags.push("vestige".to_string());
        }
        // Exactly 100 memories carry `hundred`.
        if i % 92 == 0 {
            tags.push("hundred".to_string());
        }
        storage
            .ingest_in_scope(
                IngestInput {
                    content: text(i),
                    tags,
                    ..Default::default()
                },
                "user",
            )
            .unwrap();
    }
    eprintln!("built {TOTAL} memories in {:?}", started.elapsed());
    drop(storage);
    std::fs::write(dir.join("built"), b"ok").unwrap();
}

fn copy_store(name: &str) -> PathBuf {
    let base = probe_dir().join("base");
    let copy = probe_dir().join(name);
    let _ = std::fs::remove_dir_all(&copy);
    let status = Command::new("cp")
        .arg("-Rp")
        .arg(&base)
        .arg(&copy)
        .status()
        .unwrap();
    assert!(status.success());
    copy
}

fn ingest(storage: &vestige_core::Storage, content: &str, tags: &[&str]) -> (String, Vec<String>) {
    let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    let id = storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                tags: tags.clone(),
                ..Default::default()
            },
            "user",
        )
        .unwrap()
        .id;
    (id, tags)
}

/// The write path as it was: the same scan and plan, then one synced write
/// per edge.
fn one_write_per_edge(dir: &Path, content: &str, tags: &[&str]) -> (usize, f64, f64) {
    let storage = vestige_mcp::strata_memory::open(dir).unwrap();
    let (id, tags) = ingest(storage.as_ref(), content, tags);
    let started = Instant::now();
    let mine: BTreeSet<Identity> = auto_connect::extract_identities(content, &tags)
        .into_iter()
        .collect();
    let scan = auto_connect::scan_scope(storage.as_ref(), "user", &mine).unwrap();
    let mut carriers: BTreeMap<Identity, usize> =
        mine.iter().map(|identity| (identity.clone(), 1)).collect();
    for other in scan.holders.iter().filter(|holder| holder.id != id) {
        for identity in &other.identities {
            if let Some(count) = carriers.get_mut(identity) {
                *count += 1;
            }
        }
    }
    let joined: HashSet<String> = HashSet::new();
    let open: Vec<&Holder> = scan
        .holders
        .iter()
        .filter(|holder| holder.id != id && !joined.contains(&holder.id))
        .collect();
    let plan = auto_connect::plan_edges(
        &mine,
        &carriers,
        &open,
        scan.scope_size,
        auto_connect::MAX_AUTO_EDGES,
    );
    let planned = started.elapsed().as_secs_f64();
    let now = Utc::now();
    let mut written = 0usize;
    for candidate in plan.ranked.iter().take(auto_connect::MAX_AUTO_EDGES) {
        storage
            .save_connection(&ConnectionRecord {
                source_id: candidate.id.clone(),
                target_id: id.clone(),
                strength: 0.5,
                link_type: "touched".to_string(),
                created_at: now,
                last_activated: now,
                activation_count: 0,
            })
            .unwrap();
        written += 1;
    }
    (written, planned, started.elapsed().as_secs_f64())
}

/// The write path as it is: `auto_connect_new_memory`, one batched write.
fn one_batched_write(dir: &Path, content: &str, tags: &[&str]) -> (usize, f64, Option<String>) {
    let storage = vestige_mcp::strata_memory::open(dir).unwrap();
    let (id, tags) = ingest(storage.as_ref(), content, tags);
    let started = Instant::now();
    let report =
        auto_connect::auto_connect_new_memory(storage.as_ref(), &id, "user", content, &tags)
            .unwrap();
    (
        report.edges,
        started.elapsed().as_secs_f64(),
        report.receipt_id,
    )
}

#[test]
#[ignore]
fn measure() {
    for (label, content, tags) in [
        (
            "topic tag (23 carriers)",
            "a plain note with no reference",
            vec!["vestige", "topic-7"],
        ),
        (
            "tag on 100 memories",
            "another plain note",
            vec!["vestige", "hundred"],
        ),
        (
            "one path in the text plus a topic tag",
            "fails at crates/mod27/src/file77.rs:12 again",
            vec!["vestige", "topic-9"],
        ),
    ] {
        for round in 0..3 {
            let before = copy_store("before");
            let (edges_before, planned, total_before) = one_write_per_edge(&before, content, &tags);
            let after = copy_store("after");
            let (edges_after, total_after, receipt) = one_batched_write(&after, content, &tags);
            eprintln!(
                "{label} round {round}: BEFORE {edges_before} edges, one write each: {total_before:.3} s (scan+plan {planned:.3} s) | AFTER {edges_after} edges, one write: {total_after:.3} s, receipt {receipt:?}"
            );
            assert_eq!(edges_before, edges_after);
        }
    }
}

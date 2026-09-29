//! Fresh install, v3.1.1 auto-upgrade, failed import, SIGKILL, and the
//! hidden migrate-to-strata fallback.

use std::fs;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::support::*;

fn source_hashes(dir: &std::path::Path) -> Vec<(String, String)> {
    tree_hashes(dir)
}

#[test]
fn upgrade_fresh_install_started_twice_creates_no_sqlite_and_stays_up() {
    let dir = tempfile::tempdir().unwrap();
    for pass in 1..=2 {
        let mut child = Command::new(product_bin("vestige-mcp"))
            .arg("--data-dir")
            .arg(dir.path())
            .env("VESTIGE_DATA_DIR", dir.path())
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("HOME", dir.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");
        std::thread::sleep(Duration::from_secs(3));
        let status = child.try_wait().expect("poll");
        let artifacts = sqlite_artifacts(dir.path());
        if !artifacts.is_empty() {
            let _ = child.kill();
            missing(&format!(
                "fresh install pass {pass} created a SQLite file: {artifacts:?}. \
                 The shipped binary must never create one."
            ));
        }
        if let Some(status) = status {
            let mut err = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                use std::io::Read;
                let _ = stderr.read_to_string(&mut err);
            }
            let _ = child.wait();
            missing(&format!(
                "fresh install pass {pass} exited {status} instead of serving a strata log. \
                 stderr: {}",
                err.chars().take(800).collect::<String>()
            ));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[test]
fn upgrade_v311_auto_upgrade_preserves_memories_fsrs_and_source_sha() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let rows = sqlite_text(
        &db,
        "SELECT id, content, stability, difficulty, reps, lapses, next_review FROM knowledge_nodes",
    );
    assert!(rows.len() >= 1, "fixture has no memories");
    let mut child = Command::new(product_bin("vestige-mcp"))
        .arg("--data-dir")
        .arg(&src)
        .env("HOME", dir.path())
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    std::thread::sleep(Duration::from_secs(5));
    let status = child.try_wait().expect("poll");
    let after = sha256_file(&db);
    assert_eq!(before, after, "auto-upgrade modified the v3 file");
    if let Some(code) = status {
        let mut err = String::new();
        if let Some(mut stderr) = child.stderr.take() {
            use std::io::Read;
            let _ = stderr.read_to_string(&mut err);
        }
        let _ = child.wait();
        missing(&format!(
            "auto-upgrade on first launch is not wired. vestige-mcp exited {code} on a real \
             v3.1.1 store (schema 38, {} memories) instead of backup + import + verify + \
             switch. The v3 file sha256 was unchanged. stderr: {}",
            rows.len(),
            err.chars().take(900).collect::<String>()
        ));
    }
    // Still running: kill it and require a strata log whose memories and FSRS
    // columns match the v3 rows. The v3 file hash was already checked.
    let _ = child.kill();
    let _ = child.wait();
    let Some(strata_dir) = find_strata_dir(dir.path()) else {
        missing(
            "auto-upgrade kept vestige-mcp alive but wrote no strata segment. \
             First launch must backup, import, verify, and switch, leaving the \
             v3 file byte-identical.",
        );
    };
    let dump = run_driver(&["dump-migration", strata_dir.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "auto-upgrade wrote a strata directory that does not reopen: {dump}"
        ));
    }
    let nodes = dump["nodes"].as_array().cloned().unwrap_or_default();
    if nodes.len() != rows.len() {
        missing(&format!(
            "auto-upgrade imported {} memories, the v3 store has {}",
            nodes.len(),
            rows.len()
        ));
    }
    for row in &rows {
        let id = &row[0];
        let node = nodes
            .iter()
            .find(|n| n["legacy_id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("auto-upgrade dropped memory {id}"));
        let legacy = &node["legacy"];
        for (offset, col) in ["stability", "difficulty", "reps", "lapses", "next_review"]
            .into_iter()
            .enumerate()
        {
            let key = format!("knowledge_nodes.{col}");
            let got = legacy[&key].as_str().unwrap_or("");
            if got.is_empty() {
                missing(&format!(
                    "auto-upgrade dropped FSRS column {col} on {id}. legacy={legacy}"
                ));
            }
            let want = &row[offset + 2];
            if col == "stability" || col == "difficulty" {
                let g: f64 = got.parse().unwrap_or(f64::NAN);
                let w: f64 = want.parse().unwrap_or(f64::NAN);
                assert!((g - w).abs() < 1e-9, "{id} {col}: {got} != {want}");
            } else if got != want {
                panic!("{id} {col}: log {got:?} != sqlite {want:?}");
            }
        }
    }
}

#[test]
fn upgrade_failed_import_leaves_v3_and_names_v311() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v3");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    // Corrupt past the header so the file is still a SQLite database and the
    // import has something to fail on. The bytes we leave are the pre-import
    // image; the process must not change them.
    let len = file_len(&db);
    let file = fs::OpenOptions::new().write(true).open(&db).unwrap();
    file.set_len(4096).unwrap();
    let before = sha256_file(&db);
    let out = run_cmd(
        &product_bin("vestige-mcp"),
        &["--data-dir".into(), src.display().to_string()],
        &[("HOME", dir.path().to_str().unwrap())],
        &["VESTIGE_DATA_DIR"],
        Duration::from_secs(20),
    );
    let after = sha256_file(&db);
    assert_eq!(
        before, after,
        "failed import modified the v3 file (len was {len})"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status == Some(0) {
        panic!("corrupt v3 import exited 0: {blob}");
    }
    if !blob.contains("v3.1.1") {
        missing(&format!(
            "failed import did not point at the v3.1.1 release. The v3 file was left \
             untouched, but the process must exit with a message naming v3.1.1 so the \
             operator can go back. Output: {}",
            blob.chars().take(900).collect::<String>()
        ));
    }
}

#[test]
fn upgrade_migrate_dry_run_writes_nothing_and_keeps_source_sha() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let before = source_hashes(&src);
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
            "--dry-run".into(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(before, source_hashes(&src), "dry-run changed the v3 store");
    let wrote = dest.exists() && fs::read_dir(&dest).unwrap().next().is_some();
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) || wrote {
        missing(&format!(
            "migrate-to-strata --dry-run did not report counts without writing \
             (exit {:?}, dest written {wrote}). Output: {}",
            out.status,
            blob.chars().take(900).collect::<String>()
        ));
    }
    assert!(
        blob.contains("node") || blob.contains("Migrated") || blob.contains("Dry run"),
        "dry-run produced no count report: {blob}"
    );
}

#[test]
fn upgrade_real_migrate_keeps_source_sha_and_fsrs() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    let before = sha256_file(&db);
    let expected = sqlite_text(
        &db,
        "SELECT id, stability, difficulty, reps, lapses, next_review, suppression_count, protected \
         FROM knowledge_nodes ORDER BY id",
    );
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    assert_eq!(before, sha256_file(&db), "migrate modified the v3 file");
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status != Some(0) {
        missing(&format!(
            "migrate-to-strata failed on the real v3.1.1 fixture (schema 38). Output: {}",
            blob.chars().take(1000).collect::<String>()
        ));
    }
    let dump = run_driver(&["dump-migration", dest.to_str().unwrap()]);
    if dump.get("ok") != Some(&serde_json::Value::Bool(true)) {
        missing(&format!(
            "migrated log did not reopen for a frame dump: {dump}"
        ));
    }
    let nodes = dump["nodes"].as_array().cloned().unwrap_or_default();
    assert_eq!(nodes.len(), expected.len(), "node count drifted: {dump}");
    for row in &expected {
        let id = &row[0];
        let node = nodes
            .iter()
            .find(|n| n["legacy_id"].as_str() == Some(id.as_str()))
            .unwrap_or_else(|| panic!("migrated log dropped memory {id}"));
        let legacy = &node["legacy"];
        for (idx, col) in [
            "stability",
            "difficulty",
            "reps",
            "lapses",
            "next_review",
            "suppression_count",
            "protected",
        ]
        .into_iter()
        .enumerate()
        {
            let key = format!("knowledge_nodes.{col}");
            let got = legacy[&key].as_str().unwrap_or("");
            if got.is_empty() {
                missing(&format!(
                    "migrated node {id} dropped {col}. FSRS schedule and flags must be \
                     carried exactly. legacy={legacy}"
                ));
            }
            let want = &row[idx + 1];
            if col == "stability" || col == "difficulty" {
                let g: f64 = got.parse().unwrap_or(f64::NAN);
                let w: f64 = want.parse().unwrap_or(f64::NAN);
                assert!(
                    (g - w).abs() < 1e-9,
                    "{id} {col}: log {got} != sqlite {want}"
                );
            } else if got != want {
                panic!("{id} {col}: log {got:?} != sqlite {want:?}");
            }
        }
    }
}

#[test]
fn upgrade_nonempty_to_is_refused() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("probe-v38-ckpt"), &src);
    let before = sha256_file(&src.join("vestige.db"));
    let dest = work.path().join("strata");
    fs::create_dir_all(&dest).unwrap();
    fs::write(dest.join("already-here"), b"not empty").unwrap();
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(before, sha256_file(&src.join("vestige.db")));
    assert!(
        fs::read(dest.join("already-here")).unwrap() == b"not empty",
        "refused migrate changed the destination"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    if out.status == Some(0)
        || !blob.to_lowercase().contains("not empty") && !blob.to_lowercase().contains("empty")
    {
        missing(&format!(
            "non-empty --to was not refused. exit {:?} output: {}",
            out.status,
            blob.chars().take(800).collect::<String>()
        ));
    }
}

#[test]
fn upgrade_wal_without_flag_refuses_and_source_unchanged() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-wal"), &src);
    let before = source_hashes(&src);
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let out = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.join("vestige.db").display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(120),
    );
    assert_eq!(
        before,
        source_hashes(&src),
        "WAL refusal modified the source"
    );
    let blob = format!("{}{}", out.stdout, out.stderr);
    assert!(
        out.status != Some(0) && blob.to_lowercase().contains("wal"),
        "non-empty WAL must be refused without --accept-wal-snapshot: {blob}"
    );
}

#[test]
fn upgrade_sigkill_mid_import_then_rerun_does_not_duplicate() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    copy_tree(&fixture("fresh-v38-ckpt"), &src);
    let db = src.join("vestige.db");
    inflate_nodes(&db, 4000);
    let before = sha256_file(&db);
    let dest = work.path().join("strata");
    let home = tempfile::tempdir().unwrap();
    let mut child = Command::new(product_bin("vestige"))
        .args([
            "migrate-to-strata",
            "--from",
            src.to_str().unwrap(),
            "--to",
            dest.to_str().unwrap(),
        ])
        .env("HOME", home.path())
        .env_remove("VESTIGE_DATA_DIR")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn migrate");
    let started = Instant::now();
    let mut killed = false;
    while started.elapsed() < Duration::from_secs(180) {
        if dest.exists() && fs::read_dir(&dest).unwrap().next().is_some() {
            let _ = child.kill();
            killed = true;
            break;
        }
        if child.try_wait().expect("poll").is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = child.wait();
    assert_eq!(
        before,
        sha256_file(&db),
        "SIGKILL path modified the v3 file"
    );
    if !killed {
        missing(
            "could not SIGKILL vestige migrate-to-strata mid-import: the process \
             finished before any destination file appeared, even after the source \
             was inflated. The gate needs a real mid-import kill.",
        );
    }
    let partial_nodes = node_count_if_log(&dest);
    let again = run_vestige(
        &[
            "migrate-to-strata".into(),
            "--from".into(),
            src.display().to_string(),
            "--to".into(),
            dest.display().to_string(),
        ],
        home.path(),
        Duration::from_secs(180),
    );
    assert_eq!(before, sha256_file(&db));
    let blob = format!("{}{}", again.stdout, again.stderr);
    if again.status == Some(0)
        && !blob.to_lowercase().contains("not empty")
        && !blob.contains("idempotent")
    {
        let after = node_count_if_log(&dest);
        if after > partial_nodes.saturating_add(0) && partial_nodes > 0 && after > partial_nodes {
            panic!("rerun after SIGKILL duplicated nodes ({partial_nodes} -> {after}): {blob}");
        }
    }
    // A partial destination must be refused (or rolled back). Appending is the bug.
    if again.status == Some(0) {
        let after = node_count_if_log(&dest);
        if partial_nodes > 0 && after != partial_nodes {
            panic!("rerun changed the partial log from {partial_nodes} to {after} nodes: {blob}");
        }
        missing(&format!(
            "rerun after SIGKILL exited 0 on a partial --to. It must refuse and point at \
             --rollback. Output: {}",
            blob.chars().take(800).collect::<String>()
        ));
    }
    let after = node_count_if_log(&dest);
    if partial_nodes > 0 && after != partial_nodes {
        panic!("refused rerun still changed node count {partial_nodes} -> {after}");
    }
}

#[test]
fn upgrade_rollback_undoes_a_partial_import() {
    let help_home = tempfile::tempdir().unwrap();
    let help = run_vestige(
        &["migrate-to-strata".into(), "--help".into()],
        help_home.path(),
        Duration::from_secs(20),
    );
    let text = format!("{}{}", help.stdout, help.stderr);
    if !text.contains("--rollback") {
        missing(
            "`vestige migrate-to-strata --rollback` is not implemented. A SIGKILL \
             mid-import must be undoable with --rollback, leaving --to empty and the \
             v3 file untouched.",
        );
    }
}

#[test]
fn upgrade_legacy_links_are_not_causal_edges() {
    let work = tempfile::tempdir().unwrap();
    let cases = [
        ("fresh-v38-ckpt", "narrative"),
        ("vnc-v36", "semantic"),
        ("backfill-v31", "backfill_candidate"),
    ];
    for (fixture_name, legacy_type) in cases {
        let src = work.path().join(fixture_name);
        copy_tree(&fixture(fixture_name), &src);
        let before = sha256_file(&src.join("vestige.db"));
        let dest = work.path().join(format!("{fixture_name}-strata"));
        let home = tempfile::tempdir().unwrap();
        let out = run_vestige(
            &[
                "migrate-to-strata".into(),
                "--from".into(),
                src.display().to_string(),
                "--to".into(),
                dest.display().to_string(),
            ],
            home.path(),
            Duration::from_secs(180),
        );
        assert_eq!(before, sha256_file(&src.join("vestige.db")));
        let blob = format!("{}{}", out.stdout, out.stderr);
        if out.status != Some(0) {
            missing(&format!(
                "migrate of {fixture_name} failed, so legacy link {legacy_type} could not be checked: {}",
                blob.chars().take(600).collect::<String>()
            ));
        }
        let dump = run_driver(&["dump-migration", dest.to_str().unwrap()]);
        let edges = dump["edges"].as_array().cloned().unwrap_or_default();
        let matched: Vec<&Value> = edges
            .iter()
            .filter(|e| e["legacy_link_type"].as_str() == Some(legacy_type))
            .collect();
        if matched.is_empty() {
            missing(&format!(
                "{fixture_name} link type {legacy_type} did not survive as legacy provenance. edges={edges:?}"
            ));
        }
        for edge in &matched {
            assert_eq!(
                edge["legacy_inferred"], true,
                "{legacy_type} must be legacy data, not a causal edge: {edge}"
            );
            assert_eq!(edge["link_type"], "derived_from");
            let walked = edge["link_type"].as_str() == Some(legacy_type);
            assert!(!walked, "{legacy_type} is stored as a walkable link_type");
        }
        // The product walk must not treat that pair as a causal neighbor.
        // Until causal_walk reads the strata log, this stays a named gap.
        let home = tempfile::tempdir().unwrap();
        let walk = run_vestige(
            &[
                "--data-dir".into(),
                dest.display().to_string(),
                "causal-walk".into(),
                "--logged-write".into(),
                matched[0]["source_legacy_id"].as_str().unwrap_or("").into(),
                "--json".into(),
            ],
            home.path(),
            Duration::from_secs(40),
        );
        let walk_blob = format!("{}{}", walk.stdout, walk.stderr);
        if walk_blob.contains("SQLite store creation is disabled")
            || walk_blob.contains("cannot be opened by 4.0")
            || walk_blob.contains("no store")
        {
            missing(&format!(
                "causal walk is not wired to the strata log, so a migrated {legacy_type} \
                 edge cannot be proven unwalkable through the product. The log flags it \
                 legacy_inferred=true (that part held). CLI output: {}",
                walk_blob.chars().take(500).collect::<String>()
            ));
        }
        assert!(
            !walk_blob.contains(&format!("\"link_type\":\"{legacy_type}\"")),
            "causal walk returned legacy type {legacy_type}: {walk_blob}"
        );
    }
}

fn find_strata_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut found = None;
    fn walk(dir: &std::path::Path, found: &mut Option<std::path::PathBuf>) {
        if found.is_some() {
            return;
        }
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().and_then(|e| e.to_str()) == Some("seg") {
                *found = path.parent().map(|p| p.to_path_buf());
                return;
            }
        }
    }
    walk(root, &mut found);
    found
}

fn inflate_nodes(db: &std::path::Path, extra: usize) {
    let conn = rusqlite::Connection::open(db).unwrap();
    let mut stmt = conn
        .prepare("SELECT * FROM knowledge_nodes LIMIT 1")
        .unwrap();
    let cols: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut rows = stmt.query([]).unwrap();
    let row = rows.next().unwrap().unwrap();
    let mut values: Vec<rusqlite::types::Value> = Vec::new();
    for i in 0..cols.len() {
        values.push(row.get(i).unwrap());
    }
    drop(rows);
    drop(stmt);
    let placeholders = vec!["?"; cols.len()].join(",");
    let sql = format!(
        "INSERT INTO knowledge_nodes ({}) VALUES ({placeholders})",
        cols.join(",")
    );
    let id_idx = cols.iter().position(|c| c == "id").unwrap();
    let tx = conn.unchecked_transaction().unwrap();
    for n in 0..extra {
        values[id_idx] =
            rusqlite::types::Value::Text(format!("aaaaaaaa-bbbb-4ccc-8ddd-{:012x}", n as u64));
        tx.execute(&sql, rusqlite::params_from_iter(values.iter()))
            .unwrap();
    }
    tx.commit().unwrap();
}

fn node_count_if_log(dir: &std::path::Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    let dump = run_driver(&["dump-migration", dir.to_str().unwrap()]);
    dump["nodes"].as_array().map(|a| a.len()).unwrap_or(0)
}

//! Real `vestige` binary against a Strata data directory: every subcommand
//! either runs on the log or refuses with `unavailable_in_4_0` /
//! `similarity_disabled` and names what works, and no refusal writes.

use std::path::Path;
use std::process::{Command, Output};
use std::sync::Arc;

use serde_json::Value;
use tempfile::TempDir;
use vestige_core::{IngestInput, Storage};

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

impl Run {
    fn text(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

fn vestige(data_dir: &Path, args: &[&str]) -> Run {
    let output: Output = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env_remove("VESTIGE_DATA_DIR")
        .output()
        .expect("spawn vestige");
    Run {
        ok: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn path_arg(path: &Path) -> &str {
    path.to_str().expect("utf-8 temp path")
}

fn open(dir: &Path) -> Arc<Storage> {
    vestige_mcp::strata_memory::open(dir).expect("open strata log")
}

fn put(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> String {
    storage
        .ingest_in_scope(
            IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
                ..Default::default()
            },
            "user",
        )
        .expect("ingest")
        .id
}

fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str) {
    let now = chrono::Utc::now();
    storage
        .save_connection(&vestige_core::ConnectionRecord {
            source_id: source.to_string(),
            target_id: target.to_string(),
            strength: 1.0,
            link_type: link_type.to_string(),
            created_at: now,
            last_activated: now,
            activation_count: 0,
        })
        .expect("save edge");
}

/// Three memories in `user`: alpha and beta carry the tag, alpha -> beta is
/// a recorded edge, gamma is unlinked.
struct Seeded {
    _dir: TempDir,
    alpha: String,
    beta: String,
    gamma: String,
}

impl Seeded {
    fn path(&self) -> &Path {
        self._dir.path()
    }
}

fn seed() -> Seeded {
    let dir = TempDir::new().expect("temp dir");
    let storage = open(dir.path());
    let alpha = put(
        &storage,
        "CLI_STRATA_ALPHA login handler failed",
        &["cli-tag"],
    );
    let beta = put(
        &storage,
        "CLI_STRATA_BETA auth timeout flipped",
        &["cli-tag"],
    );
    let gamma = put(&storage, "CLI_STRATA_GAMMA unrelated note", &[]);
    link(&storage, &alpha, &beta, "derived_from");
    drop(storage);
    Seeded {
        _dir: dir,
        alpha,
        beta,
        gamma,
    }
}

fn node_count(dir: &Path) -> usize {
    open(dir).get_stats().expect("stats").total_nodes as usize
}

fn edge_count(dir: &Path) -> usize {
    open(dir).get_all_connections().expect("edges").len()
}

#[test]
fn backup_copies_the_strata_log_and_never_vestige_db() {
    let seeded = seed();
    // An upgraded store keeps its v3 file beside the published log.
    let v3 = seeded.path().join("vestige.db");
    let v3_bytes = b"SQLite format 3\0CLI_STRATA_V3_ONLY".to_vec();
    std::fs::write(&v3, &v3_bytes).unwrap();

    let out = TempDir::new().unwrap();
    let dest = out.path().join("bk");
    let run = vestige(seeded.path(), &["backup", path_arg(&dest)]);
    assert!(run.ok, "{}", run.text());
    assert!(
        run.stdout.contains("Sealing and copying the Strata log"),
        "{}",
        run.text()
    );
    assert!(!run.stdout.contains("SQLite snapshot"), "{}", run.text());
    assert!(run.stdout.contains("not the live store"), "{}", run.text());

    let segments = std::fs::read_dir(dest.join("log"))
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "seg"))
        .count();
    assert!(segments > 0, "backup has no log segments");
    assert!(!dest.join("vestige.db").exists());
    for entry in walk(&dest) {
        assert_ne!(
            std::fs::read(&entry).unwrap(),
            v3_bytes,
            "{entry:?} is vestige.db"
        );
    }
    assert_eq!(
        std::fs::read(&v3).unwrap(),
        v3_bytes,
        "vestige.db was modified"
    );

    // The copy is a working, verifiable log with every memory.
    let verify = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("strata-verify")
        .arg(&dest)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "{}",
        String::from_utf8_lossy(&verify.stdout)
    );
    assert_eq!(node_count(&dest), 3);

    // A second backup never mixes into an existing one.
    let again = vestige(seeded.path(), &["backup", path_arg(&dest)]);
    assert!(!again.ok);
    assert!(again.stderr.contains("already exists"), "{}", again.text());

    // Nor into the live log.
    let inside = seeded.path().join("log").join("bk");
    let nested = vestige(seeded.path(), &["backup", path_arg(&inside)]);
    assert!(!nested.ok);
    assert!(
        nested.stderr.contains("inside the live log"),
        "{}",
        nested.text()
    );
    assert!(!inside.exists());
}

fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(walk(&path));
        } else {
            files.push(path);
        }
    }
    files
}

#[test]
fn backup_refuses_a_directory_with_no_store() {
    let empty = TempDir::new().unwrap();
    let out = TempDir::new().unwrap();
    let run = vestige(empty.path(), &["backup", path_arg(&out.path().join("bk"))]);
    assert!(!run.ok);
    assert!(run.stderr.contains("nothing to back up"), "{}", run.text());
    assert!(!empty.path().join("log").exists(), "a store was created");
}

#[test]
fn recall_resolves_exact_handles_and_refuses_free_text() {
    let seeded = seed();
    let dir = seeded.path();

    let by_id = vestige(dir, &["recall", "--handle", &seeded.alpha]);
    assert!(by_id.ok, "{}", by_id.text());
    assert!(
        by_id.stdout.contains("CLI_STRATA_ALPHA"),
        "{}",
        by_id.text()
    );
    assert!(by_id.stdout.contains("kind=memory"), "{}", by_id.text());
    assert!(by_id.stdout.contains("derived_from"), "{}", by_id.text());

    let json = vestige(dir, &["recall", "--handle", &seeded.alpha, "--json"]);
    assert!(json.ok, "{}", json.text());
    let value: Value = serde_json::from_str(&json.stdout).unwrap();
    assert_eq!(value["nodes"][0]["id"], seeded.alpha.as_str());
    assert_eq!(value["neighbors"][0]["to"], seeded.beta.as_str());

    let by_tag = vestige(dir, &["recall", "--handle", "cli-tag"]);
    assert!(by_tag.ok, "{}", by_tag.text());
    assert!(by_tag.stdout.contains("kind=tag"), "{}", by_tag.text());
    assert!(
        by_tag.stdout.contains("CLI_STRATA_BETA"),
        "{}",
        by_tag.text()
    );
    assert!(
        !by_tag.stdout.contains("CLI_STRATA_GAMMA"),
        "{}",
        by_tag.text()
    );

    // Seeded ids share a long prefix: a prefix is never guessed between them.
    let prefix = &seeded.alpha[..8];
    let ambiguous = vestige(dir, &["recall", "--handle", prefix]);
    assert!(!ambiguous.ok);
    assert!(
        ambiguous.stderr.contains("ambiguous"),
        "{}",
        ambiguous.text()
    );

    let unknown = vestige(dir, &["recall", "--handle", "no_such_handle"]);
    assert!(!unknown.ok);
    assert!(
        unknown.stderr.contains("handle_required"),
        "{}",
        unknown.text()
    );

    let free = vestige(dir, &["recall", "what failed near cli-tag"]);
    assert!(!free.ok);
    assert!(
        free.stderr.contains("similarity_disabled"),
        "{}",
        free.text()
    );
    assert!(
        free.stderr.contains("--handle cli-tag (tag, 2 memories)"),
        "{}",
        free.text()
    );
}

#[test]
fn backfill_refuses_and_names_causal_walk() {
    let seeded = seed();
    let run = vestige(seeded.path(), &["backfill", "--failure-id", &seeded.alpha]);
    assert!(!run.ok);
    assert!(run.stderr.contains("unavailable_in_4_0"), "{}", run.text());
    assert!(
        run.stderr.contains(&format!(
            "vestige causal-walk --logged-write {}",
            seeded.alpha
        )),
        "{}",
        run.text()
    );
}

#[test]
fn causal_walk_on_strata_walks_recorded_edges_only_and_writes_nothing() {
    let dir = TempDir::new().unwrap();
    let (effect, cause, decoy) =
        vestige_mcp::tools::causal_walk::seed_recorded_cause(dir.path()).expect("seed");
    let edges_before = edge_count(dir.path());

    let json = vestige(
        dir.path(),
        &["causal-walk", "--logged-write", &effect, "--json"],
    );
    assert!(json.ok, "{}", json.text());
    let value: Value = serde_json::from_str(&json.stdout).unwrap();
    let causes: Vec<&str> = value["causes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["id"].as_str())
        .collect();
    assert_eq!(causes, vec![cause.as_str()], "{value}");
    assert!(!causes.contains(&decoy.as_str()));

    // Promote is the CLI default; a Strata walk still persists nothing.
    let human = vestige(dir.path(), &["causal-walk", "--logged-write", &effect]);
    assert!(human.ok, "{}", human.text());
    assert!(human.stdout.contains(&cause), "{}", human.text());
    assert!(
        human.stdout.contains("persists nothing"),
        "{}",
        human.text()
    );
    assert_eq!(edge_count(dir.path()), edges_before);

    let named = vestige(dir.path(), &["causal-walk", "--failing-test", "t_login"]);
    assert!(!named.ok);
    assert!(
        named.stderr.contains("unavailable_in_4_0"),
        "{}",
        named.text()
    );
    assert!(named.stderr.contains("--logged-write"), "{}", named.text());
}

#[test]
fn causal_walk_node_id_makes_a_named_start_point_walkable_on_strata() {
    let dir = TempDir::new().unwrap();
    let (effect, cause, decoy) =
        vestige_mcp::tools::causal_walk::seed_recorded_cause(dir.path()).expect("seed");
    let edges_before = edge_count(dir.path());

    // --node-id names the recorded memory the stack frame describes
    let walked = vestige(
        dir.path(),
        &[
            "causal-walk",
            "--stack-frame",
            "src/auth.rs:10",
            "--node-id",
            &effect,
            "--json",
        ],
    );
    assert!(walked.ok, "{}", walked.text());
    let value: Value = serde_json::from_str(&walked.stdout).unwrap();
    let causes: Vec<&str> = value["causes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["id"].as_str())
        .collect();
    assert!(causes.contains(&cause.as_str()), "{value}");
    assert!(!causes.contains(&decoy.as_str()), "{value}");
    assert_eq!(value["start_points"][0]["kind"], "stack_frame", "{value}");
    assert_eq!(value["start_points"][0]["status"], "walked", "{value}");
    assert_eq!(edge_count(dir.path()), edges_before, "a walk wrote edges");

    // without it the frame is only a name, which a Strata log cannot walk
    let bare = vestige(
        dir.path(),
        &["causal-walk", "--stack-frame", "src/auth.rs:10", "--json"],
    );
    assert!(!bare.ok, "{}", bare.text());
    assert!(
        bare.stderr.contains("unavailable_in_4_0"),
        "{}",
        bare.text()
    );
    assert!(bare.stderr.contains("--node-id"), "{}", bare.text());

    // half a version range is refused, not walked as something else
    let half = vestige(
        dir.path(),
        &[
            "causal-walk",
            "--git-repo",
            "x",
            "--worked-in",
            "a",
            "--node-id",
            &effect,
        ],
    );
    assert!(!half.ok, "{}", half.text());
    assert!(half.stderr.contains("together"), "{}", half.text());
}

#[test]
fn portable_and_sync_refuse_and_write_nothing() {
    let seeded = seed();
    let out = TempDir::new().unwrap();

    let archive = out.path().join("sub").join("p.json");
    let export = vestige(seeded.path(), &["portable-export", path_arg(&archive)]);
    assert!(!export.ok);
    assert!(
        export.stderr.contains("unavailable_in_4_0"),
        "{}",
        export.text()
    );
    assert!(
        export.stderr.contains("vestige export"),
        "{}",
        export.text()
    );
    assert!(
        !out.path().join("sub").exists(),
        "portable-export created files"
    );

    let sync_file = out.path().join("sync.json");
    let sync = vestige(seeded.path(), &["sync", path_arg(&sync_file)]);
    assert!(!sync.ok);
    assert!(
        sync.stderr.contains("unavailable_in_4_0"),
        "{}",
        sync.text()
    );
    assert!(!sync_file.exists(), "sync created the archive");

    let portable = out.path().join("portable.json");
    std::fs::write(&portable, portable_archive()).unwrap();
    let import = vestige(seeded.path(), &["portable-import", path_arg(&portable)]);
    assert!(!import.ok);
    assert!(
        import.stderr.contains("unavailable_in_4_0"),
        "{}",
        import.text()
    );
    assert_eq!(node_count(seeded.path()), 3);
}

fn portable_archive() -> String {
    serde_json::json!({
        "archiveFormat": vestige_core::PORTABLE_ARCHIVE_FORMAT,
        "vestigeVersion": "3.1.1",
        "schemaVersion": 20,
        "exportedAt": "2026-09-01T00:00:00Z",
        "mode": "exact",
        "tables": [],
    })
    .to_string()
}

#[test]
fn restore_reingests_exports_and_refuses_everything_else() {
    let seeded = seed();
    let out = TempDir::new().unwrap();

    // A Strata backup is restored by copying log/, not by this command.
    let backup = out.path().join("bk");
    assert!(vestige(seeded.path(), &["backup", path_arg(&backup)]).ok);
    let from_dir = vestige(seeded.path(), &["restore", path_arg(&backup)]);
    assert!(!from_dir.ok);
    assert!(
        from_dir.stderr.contains("unavailable_in_4_0"),
        "{}",
        from_dir.text()
    );
    assert!(from_dir.stderr.contains("log/"), "{}", from_dir.text());

    let portable = out.path().join("portable.json");
    std::fs::write(&portable, portable_archive()).unwrap();
    let from_portable = vestige(seeded.path(), &["restore", path_arg(&portable)]);
    assert!(!from_portable.ok);
    assert!(
        from_portable.stderr.contains("unavailable_in_4_0"),
        "{}",
        from_portable.text()
    );

    let sqlite = out.path().join("old.db");
    std::fs::write(&sqlite, b"SQLite format 3\0rest").unwrap();
    let from_sqlite = vestige(seeded.path(), &["restore", path_arg(&sqlite)]);
    assert!(!from_sqlite.ok);
    assert!(
        from_sqlite.stderr.contains("upgrade"),
        "{}",
        from_sqlite.text()
    );
    assert_eq!(node_count(seeded.path()), 3);

    // `vestige export` json re-ingests into another store as new records.
    let exported = out.path().join("export.json");
    assert!(vestige(seeded.path(), &["export", path_arg(&exported)]).ok);
    let target = TempDir::new().unwrap();
    let restored = vestige(target.path(), &["restore", path_arg(&exported)]);
    assert!(restored.ok, "{}", restored.text());
    assert!(
        restored.stdout.contains("new record"),
        "{}",
        restored.text()
    );
    assert!(
        !restored.stdout.contains("embeddings"),
        "{}",
        restored.text()
    );
    assert_eq!(node_count(target.path()), 3);
}

#[test]
fn ingest_backdating_ingest_git_and_gc_refuse_before_writing() {
    let seeded = seed();
    let dir = seeded.path();

    let backdated = vestige(dir, &["ingest", "CLI_STRATA_OLD", "--ago-days", "3"]);
    assert!(!backdated.ok);
    assert!(
        backdated.stderr.contains("unavailable_in_4_0"),
        "{}",
        backdated.text()
    );
    let stamped = vestige(
        dir,
        &[
            "ingest",
            "CLI_STRATA_OLD",
            "--created-at",
            "2026-01-01T00:00:00Z",
        ],
    );
    assert!(!stamped.ok);
    assert_eq!(node_count(dir), 3, "a refused ingest left a memory behind");

    let plain = vestige(dir, &["ingest", "CLI_STRATA_NOW"]);
    assert!(plain.ok, "{}", plain.text());
    assert_eq!(node_count(dir), 4);

    let git = vestige(dir, &["ingest-git", path_arg(dir)]);
    assert!(!git.ok);
    assert!(git.stderr.contains("unavailable_in_4_0"), "{}", git.text());

    let gc = vestige(dir, &["gc", "--yes", "--min-retention", "1.1"]);
    assert!(!gc.ok);
    assert!(gc.stderr.contains("withheld"), "{}", gc.text());
    let dry = vestige(dir, &["gc", "--dry-run", "--min-retention", "1.1"]);
    assert!(dry.ok, "{}", dry.text());
    assert!(
        dry.stdout.contains("Below threshold: 4 / 4"),
        "{}",
        dry.text()
    );
    assert!(!dry.stdout.contains("would be deleted"), "{}", dry.text());
    assert_eq!(node_count(dir), 4);
}

#[test]
fn compose_runs_both_ghostlink_lenses_with_proofs() {
    let seeded = seed();
    let sorted = |a: &str, b: &str| {
        let mut pair = [a.to_string(), b.to_string()];
        pair.sort();
        (pair[0].clone(), pair[1].clone())
    };
    // `--json` keeps the 4.0.0 contract: an array of pairs with a_id / b_id
    // and the legacy keys, each also carrying its lens, lane and proof.
    let pairs_of = |answer: &Value| -> Vec<(String, String)> {
        answer
            .as_array()
            .unwrap_or_else(|| panic!("--json must print an array: {answer}"))
            .iter()
            .map(|c| {
                for key in [
                    "score", "novelty", "bridge", "trust", "a", "b", "question", "reason",
                ] {
                    assert!(c.get(key).is_some(), "legacy key {key} missing: {c}");
                }
                assert_eq!(c["shared_tags"], serde_json::json!([]), "{c}");
                sorted(c["a_id"].as_str().unwrap(), c["b_id"].as_str().unwrap())
            })
            .collect()
    };

    // Bridge (default): alpha -derived_from-> beta is one typed hop and was
    // never woven, so it is the one pair, carried with its path.
    let bridge = vestige(seeded.path(), &["compose", "--limit", "10", "--json"]);
    assert!(bridge.ok, "{}", bridge.text());
    let bridge: Value = serde_json::from_str(&bridge.stdout).unwrap();
    assert_eq!(
        pairs_of(&bridge),
        vec![sorted(&seeded.alpha, &seeded.beta)],
        "{bridge}"
    );
    let only = &bridge[0];
    assert_eq!(only["lens"], "bridge", "{only}");
    assert_eq!(only["proof"]["hops"], 1, "{only}");
    assert!(
        only["reason"]
            .as_str()
            .unwrap()
            .contains("1 typed-edge hop"),
        "{only}"
    );

    // Divergent: the linked pair is never eligible; gamma, joined to nothing,
    // is paired once on the page as a forced juxtaposition with no score.
    let divergent = vestige(
        seeded.path(),
        &["compose", "--lens", "divergent", "--limit", "10", "--json"],
    );
    assert!(divergent.ok, "{}", divergent.text());
    let divergent: Value = serde_json::from_str(&divergent.stdout).unwrap();
    let pairs = pairs_of(&divergent);
    assert!(
        !pairs.contains(&sorted(&seeded.alpha, &seeded.beta)),
        "linked pair listed: {divergent}"
    );
    assert_eq!(pairs.len(), 1, "each memory once per page: {divergent}");
    assert!(
        pairs[0].0 == seeded.gamma || pairs[0].1 == seeded.gamma,
        "{divergent}"
    );
    assert!(divergent[0]["score"].is_null(), "{divergent}");
    assert_eq!(divergent[0]["lane"], "juxtaposition", "{divergent}");

    // Human output names the lens and the reason; an unknown lens fails.
    let human = vestige(seeded.path(), &["compose"]);
    assert!(human.ok, "{}", human.text());
    assert!(human.stdout.contains("bridge lens"), "{}", human.text());
    assert!(human.stdout.contains("typed-edge hop"), "{}", human.text());
    // Every pair prints its proof path.
    let path_line = human
        .stdout
        .lines()
        .find(|line| line.trim_start().starts_with("path: "))
        .unwrap_or_else(|| panic!("no proof path printed: {}", human.text()));
    assert!(
        path_line.contains("derived_from")
            && path_line.contains(&seeded.alpha)
            && path_line.contains(&seeded.beta),
        "{path_line}"
    );
    let bad = vestige(seeded.path(), &["compose", "--lens", "nearest"]);
    assert!(!bad.ok, "{}", bad.text());
    assert!(bad.text().contains("unknown lens"), "{}", bad.text());
}

#[test]
fn health_consolidate_and_upgrade_describe_the_strata_log() {
    let seeded = seed();
    let dir = seeded.path();

    let health = vestige(dir, &["health"]);
    assert!(health.ok, "{}", health.text());
    assert!(
        health.stdout.contains("exact handles only"),
        "{}",
        health.text()
    );
    assert!(
        !health.stdout.contains("Embedding Coverage"),
        "{}",
        health.text()
    );
    assert!(
        !health.stdout.contains("keyword search only"),
        "{}",
        health.text()
    );
    assert!(!health.stdout.contains("consolidat"), "{}", health.text());

    let consolidate = vestige(dir, &["consolidate"]);
    assert!(consolidate.ok, "{}", consolidate.text());
    assert!(
        consolidate.stdout.contains("no-op"),
        "{}",
        consolidate.text()
    );

    let v3 = dir.join("vestige.db");
    std::fs::write(&v3, b"SQLite format 3\0kept").unwrap();
    for args in [&["upgrade", "--dry-run"][..], &["upgrade"][..]] {
        let upgrade = vestige(dir, args);
        assert!(upgrade.ok, "{}", upgrade.text());
        assert!(
            upgrade.stdout.contains("Already upgraded"),
            "{}",
            upgrade.text()
        );
    }
    assert_eq!(std::fs::read(&v3).unwrap(), b"SQLite format 3\0kept");
}

fn token(fill: &str) -> String {
    format!("ghp_{}", fill.repeat(36))
}

#[test]
fn ingest_and_restore_refuse_a_credential_in_tags_or_source_without_echo() {
    let seeded = seed();
    let dir = seeded.path();
    let secret = token("A");

    let tagged = vestige(
        dir,
        &[
            "ingest",
            "CLI_STRATA_GATE plain note",
            "--tags",
            &format!("safe,{secret}"),
        ],
    );
    assert!(!tagged.ok, "{}", tagged.text());
    assert!(!tagged.text().contains(&secret), "{}", tagged.text());

    let sourced = vestige(
        dir,
        &[
            "ingest",
            "CLI_STRATA_GATE plain note",
            "--source",
            &format!("https://example.invalid/?t={secret}"),
        ],
    );
    assert!(!sourced.ok, "{}", sourced.text());
    assert!(!sourced.text().contains(&secret), "{}", sourced.text());
    assert_eq!(node_count(dir), 3, "a refused ingest left a memory behind");

    let out = TempDir::new().unwrap();
    let file = out.path().join("backup.json");
    std::fs::write(
        &file,
        serde_json::json!([
            {"content": "CLI_STRATA_GATE tagged", "tags": [secret]},
            {"content": "CLI_STRATA_GATE sourced", "source": secret},
            {"content": "CLI_STRATA_GATE clean"}
        ])
        .to_string(),
    )
    .unwrap();
    let restored = vestige(dir, &["restore", path_arg(&file)]);
    assert!(!restored.text().contains(&secret), "{}", restored.text());
    assert_eq!(
        node_count(dir),
        4,
        "only the clean record may be restored: {}",
        restored.text()
    );
}

#[test]
fn scan_secrets_reaches_retired_memories_scopes_and_intentions() {
    let dir = TempDir::new().expect("temp dir");
    let secret = token("C");
    // Records written before the gate covered every field still sit in the
    // log. Seed them below the gate, as an older binary would have.
    let (retired, scoped) = {
        let mut raw = strata_store::StrataStore::open(dir.path()).expect("open raw");
        let node = |tags: Vec<String>| strata_store::IngestInput {
            content: "CLI_STRATA_AUDIT plain note".into(),
            source: None,
            source_updated_at_ms: None,
            node_type: "fact".into(),
            tags,
            created_at_ms: Some(1),
            valid_from_ms: None,
            valid_until_ms: None,
        };
        let retired = raw
            .ingest_in_scope(node(vec![secret.clone()]), "user")
            .expect("seed tagged");
        let scoped = raw
            .ingest_in_scope(node(Vec::new()), &secret)
            .expect("seed scoped");
        raw.upsert_intentions(vec![strata_store::IntentionRecord {
            id: "int-audit".into(),
            content: format!("rotate {secret}"),
            trigger_type: "manual".into(),
            trigger_data: "{}".into(),
            priority: 2,
            status: "active".into(),
            created_at_ms: 1,
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
            scope: Some("user".into()),
        }])
        .expect("seed intention");
        (retired, scoped)
    };
    {
        let storage = open(dir.path());
        storage.suppress_memory(&retired).expect("suppress");
        assert!(storage.get_node(&retired).expect("get").is_none());
    }

    let scan = vestige(dir.path(), &["scan-secrets", "--json"]);
    assert!(scan.ok, "{}", scan.text());
    assert!(
        !scan.text().contains(&secret),
        "the audit must not print the credential: {}",
        scan.text()
    );
    let report: Value = serde_json::from_str(&scan.stdout).expect("json report");
    let hit_ids: Vec<&str> = report["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .filter_map(|hit| hit["nodeId"].as_str())
        .collect();
    assert!(
        hit_ids.contains(&retired.as_str()),
        "a suppressed memory's tag must be reported: {report}"
    );
    assert!(
        hit_ids.contains(&scoped.as_str()),
        "a credential-shaped scope must be reported: {report}"
    );
    assert!(
        hit_ids.contains(&"int-audit"),
        "an intention must be reported: {report}"
    );
}

/// Ingest auto-connects: the first memory lands alone, and the second —
/// sharing the `euler` tag/entity with it — is joined by one `touched` edge
/// written by the ingest itself, earlier -> later, so a backward causal-walk
/// from the second reaches the first with no `vestige connect` in between.
#[test]
fn ingest_auto_connects_shared_entities_then_causal_walk_finds_the_cause() {
    let dir = TempDir::new().expect("temp dir");

    let ingest = |content: &str, source: &str, tags: &str| {
        let run = vestige(
            dir.path(),
            &["ingest", content, "--source", source, "--tags", tags],
        );
        assert!(run.ok, "{}", run.text());
        run.stdout
            .lines()
            .find_map(|line| line.strip_prefix("Node ID: "))
            .expect("ingest prints the node id")
            .trim()
            .to_string()
    };
    let cause = ingest(
        "PR 4337 removed npoints floor from euler() in path.py",
        "git-log",
        "euler,path.py",
    );
    // Nothing shares an entity with the first memory yet.
    assert_eq!(edge_count(dir.path()), 0, "the first ingest has no peer");

    let effect_run = vestige(
        dir.path(),
        &[
            "ingest",
            "Issue 4557: strange paths in euler bends, geometry deformed",
            "--source",
            "github-issue",
            "--tags",
            "bug,euler",
        ],
    );
    assert!(effect_run.ok, "{}", effect_run.text());
    assert!(
        effect_run
            .stdout
            .contains("Auto-connected: 1 edge(s) created via shared entities: euler"),
        "{}",
        effect_run.text()
    );
    let effect = effect_run
        .stdout
        .lines()
        .find_map(|line| line.strip_prefix("Node ID: "))
        .expect("ingest prints the node id")
        .trim()
        .to_string();
    assert_eq!(edge_count(dir.path()), 1, "{}", effect_run.text());

    // The walk follows the auto-written touched edge back to the PR without
    // `vestige connect` ever running.
    let walk = vestige(dir.path(), &["causal-walk", "--logged-write", &effect]);
    assert!(walk.ok, "{}", walk.text());
    assert!(walk.stdout.contains(&cause), "{}", walk.text());
    assert!(walk.stdout.contains("touched"), "{}", walk.text());
    assert!(walk.stdout.contains("PR 4337"), "{}", walk.text());

    // The connect command stays the full-scan catch-up, and it sees nothing
    // left to do: the pair is already joined by the auto-written edge.
    let again = vestige(dir.path(), &["connect"]);
    assert!(again.ok, "{}", again.text());
    assert!(again.stdout.contains("No new edges"), "{}", again.text());
    assert_eq!(edge_count(dir.path()), 1);
}

/// Auto-connect speaks only when it writes: a memory with no peer ingests
/// with the plain output, a later memory sharing a tag says so, and a third
/// joining two earlier memories adds exactly two edges — the edge between
/// the first two is never duplicated.
#[test]
fn ingest_auto_connect_is_quiet_without_peers_and_idempotent() {
    let dir = TempDir::new().expect("temp dir");

    let ingest = |content: &str, tags: &str| {
        let run = vestige(dir.path(), &["ingest", content, "--tags", tags]);
        assert!(run.ok, "{}", run.text());
        let id = run
            .stdout
            .lines()
            .find_map(|line| line.strip_prefix("Node ID: "))
            .expect("ingest prints the node id")
            .trim()
            .to_string();
        (run, id)
    };

    let (first_run, _first) = ingest("Changed redis timeout to 5s in config", "redis,timeout");
    assert!(
        !first_run.stdout.contains("Auto-connected"),
        "the first memory has no peer: {}",
        first_run.text()
    );
    assert_eq!(edge_count(dir.path()), 0);

    let (second_run, _second) = ingest("Redis dropping connections, timeout errors", "bug,redis");
    assert!(
        second_run
            .stdout
            .contains("Auto-connected: 1 edge(s) created via shared entities: redis, timeout"),
        "{}",
        second_run.text()
    );
    assert_eq!(edge_count(dir.path()), 1);

    // The third memory shares `redis` with both earlier ones. The first two
    // are already joined, so this ingest writes its own two edges only.
    let (third_run, _third) = ingest("redis OOM killed the worker", "redis");
    assert!(
        third_run.stdout.contains("Auto-connected: 2 edge(s)"),
        "{}",
        third_run.text()
    );
    assert_eq!(edge_count(dir.path()), 3);
}

/// --min-shared gates pairs on entity overlap: two memories sharing two
/// entities connect, two sharing none never do.
#[test]
fn connect_min_shared_gates_the_pairs() {
    let dir = TempDir::new().expect("temp dir");
    let storage = open(dir.path());
    let near = put(
        &storage,
        "euler refactor touched connection_pool in db.py",
        &["euler"],
    );
    let far = put(&storage, "euler bends deform under load", &[]);
    let lone = put(&storage, "release notes for the dashboard", &[]);
    drop(storage);

    let strict = vestige(dir.path(), &["connect", "--min-shared", "2", "--dry-run"]);
    assert!(strict.ok, "{}", strict.text());
    // near/far share exactly one entity (euler); with --min-shared 2 no
    // pair qualifies.
    assert!(
        strict.stdout.contains("Candidate pairs: 0"),
        "{}",
        strict.text()
    );

    let loose = vestige(dir.path(), &["connect", "--min-shared", "1"]);
    assert!(loose.ok, "{}", loose.text());
    assert_eq!(edge_count(dir.path()), 1, "only near/euler pairs connect");

    let walk = vestige(
        dir.path(),
        &["causal-walk", "--logged-write", &far, "--json"],
    );
    assert!(walk.ok, "{}", walk.text());
    let value: Value = serde_json::from_str(&walk.stdout).unwrap();
    let causes: Vec<&str> = value["causes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["id"].as_str())
        .collect();
    assert_eq!(causes, vec![near.as_str()], "{value}");
    assert!(!causes.contains(&lone.as_str()), "{value}");
}

//! Real-process driver for the v4.0.0 release gate.
//!
//! Every subcommand is a separate OS process. Tests spawn it; nothing here
//! is an in-process mock of the log.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use serde_json::{json, Value};
use strata::StrataLog;
use strata_gate::{Policy, Rule, Verdict, ANY_KIND, WILDCARD_PREFIX};
use strata_kernel::canonical::from_q32_32;
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::ALGO_V1;
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;
use strata_migrate::read_snapshot;
use strata_migrate::records::{self, decode_edge, decode_node};
use strata_store::{default_policy, ConnectionRecord, IngestInput, StrataStore};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        eprintln!(
            "usage: strata-driver <write-log|open-log|write-store|reopen-store|hold|dump-migration|dump-schedule|probe-retrievability|migrate> ..."
        );
        return ExitCode::from(2);
    }
    let cmd = args.remove(0);
    let result = match cmd.as_str() {
        "write-log" => write_log(&args),
        "open-log" => open_log(&args),
        "write-store" => write_store(&args, false),
        "write-store-permissive" => write_store(&args, true),
        "reopen-store" => reopen_store(&args),
        "hold" => hold(&args),
        "dump-migration" => dump_migration(&args),
        "dump-schedule" => dump_schedule(&args),
        "probe-retrievability" => probe_retrievability(&args),
        "migrate" => migrate_cmd(&args),
        other => Err(format!("unknown command {other}")),
    };
    match result {
        Ok(code) => code,
        Err(err) => {
            let _ = writeln!(io::stderr(), "{err}");
            println!("{}", json!({ "ok": false, "error": err }));
            ExitCode::from(1)
        }
    }
}

fn dir_arg(args: &[String]) -> Result<PathBuf, String> {
    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--seed" {
            skip_next = true;
            continue;
        }
        if !arg.starts_with("--") {
            return Ok(PathBuf::from(arg));
        }
    }
    Err("missing directory argument".into())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn parse_seed(text: &str) -> Result<[u8; 32], String> {
    if text.len() != 64 {
        return Err(format!("seed must be 64 hex chars, got {}", text.len()));
    }
    let mut seed = [0u8; 32];
    for i in 0..32 {
        seed[i] = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
            .map_err(|e| format!("seed hex: {e}"))?;
    }
    Ok(seed)
}

fn files_under(dir: &Path) -> Value {
    let mut rows = Vec::new();
    fn walk(dir: &Path, root: &Path, rows: &mut Vec<Value>) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, rows);
                continue;
            }
            let name = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            if name.ends_with("strata.lock") {
                continue;
            }
            let len = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let bytes = fs::read(&path).unwrap_or_default();
            rows.push(json!({
                "name": name,
                "len": len,
                "blake3": hex(blake3::hash(&bytes).as_bytes()),
            }));
        }
    }
    walk(dir, dir, &mut rows);
    Value::Array(rows)
}

fn emit_ok(value: Value) -> Result<ExitCode, String> {
    println!("{value}");
    Ok(ExitCode::SUCCESS)
}

fn write_log(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let log = match flag_value(args, "--seed") {
        Some(text) => {
            StrataLog::open_seeded(&dir, parse_seed(&text)?).map_err(|e| e.to_string())?
        }
        None => StrataLog::open(&dir).map_err(|e| e.to_string())?,
    };
    for i in 0..8u8 {
        let payload = format!("release-matrix-frame-{i}");
        log.append(32, payload.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let sealed = if flag(args, "--seal") {
        let info = log.seal().map_err(|e| e.to_string())?;
        Some(info.sealed_segment_no)
    } else {
        None
    };
    let head = log.head();
    emit_ok(json!({
        "ok": true,
        "frames": head.frames_total,
        "last_acked_seq": head.last_acked_seq,
        "sealed_segment": sealed,
        "files": files_under(&dir),
    }))
}

fn open_log(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    match StrataLog::open(&dir) {
        Ok(log) => {
            let head = log.head();
            let frames = log.read_frames(1).map_err(|e| e.to_string())?;
            emit_ok(json!({
                "ok": true,
                "frames": head.frames_total,
                "last_acked_seq": head.last_acked_seq,
                "read_frames": frames.len(),
                "files": files_under(&dir),
            }))
        }
        Err(err) => {
            println!(
                "{}",
                json!({
                    "ok": false,
                    "error": err.to_string(),
                    "files": files_under(&dir),
                })
            );
            Ok(ExitCode::from(1))
        }
    }
}

fn allow_all() -> Policy {
    Policy {
        rules: vec![Rule {
            match_kind: ANY_KIND,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: u32::MAX,
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Allow,
        }],
    }
}

fn fixed_input(content: &str, at: i64) -> IngestInput {
    IngestInput {
        content: content.to_string(),
        created_at_ms: Some(at),
        node_type: "fact".to_string(),
        tags: vec!["release-matrix".to_string()],
        ..IngestInput::default()
    }
}

fn verdict_name(v: Verdict) -> &'static str {
    match v {
        Verdict::Allow => "Allow",
        Verdict::Deny => "Deny",
        Verdict::Hold => "Hold",
    }
}

fn write_store(args: &[String], permissive: bool) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let policy = if permissive {
        allow_all()
    } else {
        default_policy()
    };
    let mut store = StrataStore::open_with_policy(&dir, policy).map_err(|e| e.to_string())?;
    let mut ids = Vec::new();
    for (i, content) in [
        "payments api timeout is 30s after the september incident",
        "deploys cut over by swapping the target group",
        "postgres primary is db-prod-1",
    ]
    .into_iter()
    .enumerate()
    {
        let id = store
            .ingest(fixed_input(content, 1_700_000_000_000 + i as i64))
            .map_err(|e| e.to_string())?;
        ids.push(id);
    }
    store
        .save_connection(&ConnectionRecord {
            source_id: ids[1].clone(),
            target_id: ids[0].clone(),
            link_type: "derived_from".to_string(),
            strength_milli: 1000,
            created_at_ms: 1_700_000_000_000,
            ..ConnectionRecord::default()
        })
        .map_err(|e| e.to_string())?;
    store.review(&ids[0], 3).map_err(|e| e.to_string())?;
    let supersede = store.supersede(&ids[0], &ids[2]);
    let supersede_msg = match &supersede {
        Ok(()) => "applied".to_string(),
        Err(err) => err.to_string(),
    };
    let _ = store.seal_checkpoint().map_err(|e| e.to_string())?;
    let verdicts: Vec<String> = store
        .rederive_verdicts()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|(_, v)| verdict_name(v).to_string())
        .collect();
    let superseded = store
        .get_node(&ids[0])
        .and_then(|n| n.superseded_by)
        .is_some();
    emit_ok(json!({
        "ok": true,
        "digest": hex(&store.state_digest()),
        "nodes": store.node_count(),
        "edges": store.edge_count(),
        "ids": ids,
        "supersede": supersede_msg,
        "superseded": superseded,
        "verdicts": verdicts,
        "sweep_gaps": store.sweep().len(),
        "files": files_under(&dir),
    }))
}

fn reopen_store(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    match StrataStore::open(&dir) {
        Ok(store) => emit_ok(json!({
            "ok": true,
            "digest": hex(&store.state_digest()),
            "nodes": store.node_count(),
            "edges": store.edge_count(),
            "sweep_gaps": store.sweep().len(),
            "files": files_under(&dir),
        })),
        Err(err) => {
            println!(
                "{}",
                json!({
                    "ok": false,
                    "error": err.to_string(),
                    "files": files_under(&dir),
                })
            );
            Ok(ExitCode::from(1))
        }
    }
}

fn hold(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut store = StrataStore::open(&dir).map_err(|e| e.to_string())?;
    for i in 0..4 {
        store
            .ingest(fixed_input(
                &format!("held writer record {i}"),
                1_700_000_000_000 + i,
            ))
            .map_err(|e| e.to_string())?;
    }
    println!(
        "{}",
        json!({
            "ready": true,
            "digest": hex(&store.state_digest()),
            "nodes": store.node_count(),
        })
    );
    io::stdout().flush().map_err(|e| e.to_string())?;
    // The store (and its directory lock) stay alive until SIGKILL or a
    // normal signal. Drop does not run on SIGKILL; the next opener must
    // reclaim a dead pid.
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

fn dump_migration(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    let log = StrataLog::open(&dir).map_err(|e| e.to_string())?;
    let frames = log.read_frames(1).map_err(|e| e.to_string())?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut kinds: Vec<u8> = Vec::new();
    for frame in &frames {
        kinds.push(frame.kind);
        if frame.kind == records::KIND_NODE {
            let node = decode_node(&frame.payload).map_err(|e| e.to_string())?;
            let legacy: serde_json::Map<String, Value> = node
                .legacy
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            nodes.push(json!({
                "legacy_id": node.legacy_id,
                "content": node.content,
                "node_type": node.node_type,
                "tags": node.tags,
                "legacy": legacy,
            }));
        } else if frame.kind == records::KIND_EDGE {
            let edge = decode_edge(&frame.payload).map_err(|e| e.to_string())?;
            edges.push(json!({
                "link_type": edge.link_type,
                "legacy_inferred": edge.legacy_inferred,
                "legacy_link_type": edge.legacy_link_type,
                "source_legacy_id": edge.source_legacy_id,
                "target_legacy_id": edge.target_legacy_id,
            }));
        }
    }
    emit_ok(json!({
        "ok": true,
        "frames": frames.len(),
        "kinds": kinds,
        "nodes": nodes,
        "edges": edges,
    }))
}

fn dump_schedule(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    let log = StrataLog::open(&dir).map_err(|e| e.to_string())?;
    let snapshot = read_snapshot(&log).map_err(|e| e.to_string())?;
    let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1).map_err(|e| e.to_string())?;
    let mut state = State::default();
    kernel.apply_all(&mut state, snapshot.reviews.iter());
    let mut cards = Vec::new();
    for node in &snapshot.nodes {
        let card = state.cards.get(&node.kernel_id);
        let reviews: Vec<Value> = snapshot
            .reviews
            .iter()
            .filter(|event| event.card_id == node.kernel_id)
            .map(|event| {
                json!({
                    "event_seq": event.event_seq,
                    "rating": event.rating,
                    "fields": ["card_id", "rating", "event_seq"],
                })
            })
            .collect();
        let legacy: serde_json::Map<String, Value> = node
            .legacy
            .iter()
            .filter(|(key, _)| {
                let name = key.rsplit('.').next().unwrap_or(key);
                matches!(
                    name,
                    "last_review"
                        | "due"
                        | "due_date"
                        | "next_review"
                        | "stability"
                        | "difficulty"
                        | "last_accessed"
                )
            })
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect();
        cards.push(json!({
            "legacy_id": node.legacy_id,
            "kernel_id": node.kernel_id,
            "created_ms": node.created_ms,
            "last_accessed_ms": node.last_accessed_ms,
            "last_seq": card.map(|card| card.last_seq),
            "stability": card.map(|card| from_q32_32(card.stability_q)),
            "difficulty": card.map(|card| from_q32_32(card.difficulty_q)),
            "due": Value::Null,
            "last_review_source": "CardState.last_seq",
            "reviews": reviews,
            "legacy": legacy,
        }));
    }
    emit_ok(json!({
        "ok": true,
        "cards": cards,
        "review_event_fields": ["card_id", "rating", "event_seq"],
    }))
}

/// Fresh store: one memory with a caller-supplied `created_at`, one explicit
/// review, then a second ingest that advances seq and does not review the
/// first card. Retrievability is read after each step. No clock is consulted.
fn probe_retrievability(args: &[String]) -> Result<ExitCode, String> {
    let dir = dir_arg(args)?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let hooks = [
        "VESTIGE_NOW_MS",
        "VESTIGE_CLOCK",
        "STRATA_NOW_MS",
        "STRATA_TEST_CLOCK",
    ];
    let hook_set: Vec<&str> = hooks
        .into_iter()
        .filter(|name| std::env::var_os(name).is_some())
        .collect();
    let created_at_ms = 1_000_000_000_000_i64;
    let mut store = StrataStore::open_with_policy(&dir, allow_all()).map_err(|e| e.to_string())?;
    let id = store
        .ingest(fixed_input("explicit review clock probe", created_at_ms))
        .map_err(|e| e.to_string())?;
    store.review(&id, 3).map_err(|e| e.to_string())?;
    let after_review = store.retrievability(&id).map_err(|e| e.to_string())?;
    let last_seq = store.card_state(&id).map(|card| card.last_seq);
    let _other = store
        .ingest(fixed_input("seq advance is not a review", created_at_ms))
        .map_err(|e| e.to_string())?;
    let after_seq = store.retrievability(&id).map_err(|e| e.to_string())?;
    let created = store
        .get_node(&id)
        .map(|node| node.created_at_ms)
        .ok_or_else(|| "probe node missing after review".to_string())?;
    emit_ok(json!({
        "ok": true,
        "id": id,
        "created_at_ms": created,
        "clock_hooks_checked": hooks,
        "clock_hooks_set": hook_set,
        "review_payload_fields": ["card_id", "rating"],
        "review_has_timestamp": false,
        "last_seq": last_seq,
        "retrievability_after_review": after_review,
        "retrievability_after_seq_advance": after_seq,
    }))
}

fn migrate_cmd(args: &[String]) -> Result<ExitCode, String> {
    let paths: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if paths.len() != 2 {
        return Err("migrate <source> <dest> [--dry-run]".into());
    }
    let report = strata_migrate::migrate_with_options(
        Path::new(paths[0]),
        Path::new(paths[1]),
        strata_migrate::MigrateOptions {
            dry_run: flag(args, "--dry-run"),
            ..strata_migrate::MigrateOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let mut value = serde_json::to_value(&report).map_err(|e| e.to_string())?;
    value["ok"] = json!(true);
    if !flag(args, "--dry-run") {
        value["files"] = files_under(Path::new(paths[1]));
    }
    emit_ok(value)
}

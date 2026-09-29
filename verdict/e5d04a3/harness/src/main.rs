//! Read-only inspection and lane-E probes for the e5d04a3 verdict run.
//! Does not modify the product tree. Opens of migration logs are done on
//! copies by the shell driver; `dump` itself calls `StrataLog::open`, which
//! is not a pure reader.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use borsh::{BorshDeserialize, BorshSerialize};
use strata::StrataLog;
use strata_store::{ConnectionRecord, IngestInput, StrataStore};

const KIND_GENESIS: u8 = 0x1F;
const KIND_NODE: u8 = 0x20;
const KIND_EDGE: u8 = 0x21;
const KIND_FSRS: u8 = 0x22;
const KIND_TOMBSTONE: u8 = 0x23;
const KIND_SUPERSESSION: u8 = 0x24;
const KIND_CHECKPOINT: u8 = 0x25;
const KIND_PARAMS: u8 = 0x26;
const KIND_RECEIPT: u8 = 46;

const RECEIPT_CTX: &[u8] = b"vestige strata migration receipt v1";
const OLD_SEED_CTX: &str = "vestige strata migrate log seed v1";

#[derive(Debug, BorshDeserialize)]
struct ParamsRecord {
    _record_version: u16,
    params_id: String,
    schema_version: u32,
    source_blake3: String,
    envelope_head: String,
}

#[derive(Debug, BorshDeserialize)]
struct NodeRecord {
    _record_version: u16,
    legacy_id: String,
    kernel_id: u64,
    content: String,
    node_type: String,
    tags: Vec<String>,
    created_ms: i64,
    updated_ms: i64,
    last_accessed_ms: i64,
    legacy: Vec<(String, String)>,
}

#[derive(Debug, BorshDeserialize)]
struct EdgeRecord {
    _record_version: u16,
    source_kernel_id: u64,
    target_kernel_id: u64,
    source_legacy_id: String,
    target_legacy_id: String,
    link_type: String,
    legacy_inferred: bool,
    legacy_link_type: String,
    strength_q32: i64,
    created_ms: i64,
    last_activated_ms: i64,
    activation_count: i32,
    legacy: Vec<(String, String)>,
}

#[derive(Debug, BorshDeserialize)]
struct TombstoneRecord {
    _record_version: u16,
    origin_table: String,
    source_table: Option<String>,
    row_id: String,
    deleted_ms: i64,
    reason: Option<String>,
    node_type: Option<String>,
    tags: Vec<String>,
}

#[derive(Debug, BorshDeserialize)]
struct SupersessionRecord {
    _record_version: u16,
    superseded_legacy_id: String,
    superseded_by_legacy_id: String,
    superseded_kernel_id: u64,
    superseded_by_kernel_id: u64,
}

#[derive(Debug, BorshDeserialize)]
struct ReceiptBody {
    _record_version: u16,
    source_blake3_before: String,
    source_blake3_after: String,
    schema_version: u32,
    envelope_head: String,
    counts: Vec<(String, u64)>,
    dropped_vectors: u64,
    dropped_columns: Vec<String>,
    signing_key_id: String,
}

#[derive(Debug, BorshDeserialize)]
struct MigrationReceipt {
    _record_version: u16,
    body: ReceiptBody,
    verifying_key: [u8; 32],
    signature: [u8; 64],
    checksum: [u8; 32],
}

fn kind_name(kind: u8) -> String {
    match kind {
        KIND_GENESIS => "GENESIS".into(),
        KIND_NODE => "NODE".into(),
        KIND_EDGE => "EDGE".into(),
        KIND_FSRS => "FSRS_REVIEW".into(),
        KIND_TOMBSTONE => "TOMBSTONE".into(),
        KIND_SUPERSESSION => "SUPERSESSION".into(),
        KIND_CHECKPOINT => "CHECKPOINT".into(),
        KIND_PARAMS => "PARAMS".into(),
        KIND_RECEIPT => "MIGRATION_RECEIPT".into(),
        other => format!("KIND_{other}"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn mode_of(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).ok().map(|m| m.permissions().mode() & 0o777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn verify_receipt(r: &MigrationReceipt) -> (bool, bool) {
    let body_bytes = encode_body(&r.body);
    let checksum_ok = *blake3::hash(&body_bytes).as_bytes() == r.checksum;
    let signature_ok = {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};
        match VerifyingKey::from_bytes(&r.verifying_key) {
            Ok(vk) => {
                let mut msg = Vec::with_capacity(body_bytes.len() + RECEIPT_CTX.len());
                msg.extend_from_slice(RECEIPT_CTX);
                msg.extend_from_slice(&body_bytes);
                vk.verify(&msg, &Signature::from_bytes(&r.signature)).is_ok()
            }
            Err(_) => false,
        }
    };
    (checksum_ok, signature_ok)
}

fn encode_body(body: &ReceiptBody) -> Vec<u8> {
    #[derive(borsh::BorshSerialize)]
    struct Mirror<'a> {
        record_version: u16,
        source_blake3_before: &'a str,
        source_blake3_after: &'a str,
        schema_version: u32,
        envelope_head: &'a str,
        counts: &'a [(String, u64)],
        dropped_vectors: u64,
        dropped_columns: &'a [String],
        signing_key_id: &'a str,
    }
    borsh::to_vec(&Mirror {
        record_version: 1,
        source_blake3_before: &body.source_blake3_before,
        source_blake3_after: &body.source_blake3_after,
        schema_version: body.schema_version,
        envelope_head: &body.envelope_head,
        counts: &body.counts,
        dropped_vectors: body.dropped_vectors,
        dropped_columns: &body.dropped_columns,
        signing_key_id: &body.signing_key_id,
    })
    .expect("encode receipt body")
}

fn list_files(dir: &Path) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut walk = |root: &Path| {
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let rel = path.strip_prefix(dir).unwrap_or(&path);
                let len = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                out.push(serde_json::json!({
                    "path": rel.to_string_lossy(),
                    "bytes": len,
                    "mode": mode_of(&path),
                    "sha256": real_sha256(&path),
                }));
            }
        }
    };
    walk(dir);
    if dir.join("log").is_dir() {
        walk(&dir.join("log"));
    }
    out.sort_by(|a, b| {
        a["path"]
            .as_str()
            .unwrap_or("")
            .cmp(b["path"].as_str().unwrap_or(""))
    });
    out
}

fn real_sha256(path: &Path) -> String {
    // Use the sha256sum binary so the report matches the operator's hashes.
    let out = std::process::Command::new("sha256sum")
        .arg(path)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.split_whitespace().next().unwrap_or("").to_string()
        }
        _ => String::new(),
    }
}

fn dump(dir: &Path) -> Result<serde_json::Value, String> {
    let before = list_files(dir);
    let log = StrataLog::open(dir).map_err(|e| format!("open: {e}"))?;
    let frames = log.read_frames(1).map_err(|e| format!("read: {e}"))?;
    let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut tombstones = Vec::new();
    let mut supersessions = Vec::new();
    let mut params = None;
    let mut receipt_json = None;
    let mut decode_errors = Vec::new();
    for frame in &frames {
        *counts.entry(kind_name(frame.kind)).or_default() += 1;
        match frame.kind {
            KIND_NODE => match NodeRecord::try_from_slice(&frame.payload) {
                Ok(n) => nodes.push(serde_json::json!({
                    "legacy_id": n.legacy_id,
                    "kernel_id": n.kernel_id,
                    "content": n.content,
                    "node_type": n.node_type,
                    "tags": n.tags,
                    "created_ms": n.created_ms,
                    "updated_ms": n.updated_ms,
                    "last_accessed_ms": n.last_accessed_ms,
                    "legacy": n.legacy,
                })),
                Err(e) => decode_errors.push(format!("node seq {}: {e}", frame.seq)),
            },
            KIND_EDGE => match EdgeRecord::try_from_slice(&frame.payload) {
                Ok(e) => edges.push(serde_json::json!({
                    "source_kernel_id": e.source_kernel_id,
                    "target_kernel_id": e.target_kernel_id,
                    "source_legacy_id": e.source_legacy_id,
                    "target_legacy_id": e.target_legacy_id,
                    "link_type": e.link_type,
                    "legacy_inferred": e.legacy_inferred,
                    "legacy_link_type": e.legacy_link_type,
                    "strength_q32": e.strength_q32,
                    "created_ms": e.created_ms,
                    "last_activated_ms": e.last_activated_ms,
                    "activation_count": e.activation_count,
                    "legacy": e.legacy,
                })),
                Err(e) => decode_errors.push(format!("edge seq {}: {e}", frame.seq)),
            },
            KIND_TOMBSTONE => match TombstoneRecord::try_from_slice(&frame.payload) {
                Ok(t) => tombstones.push(serde_json::json!({
                    "origin_table": t.origin_table,
                    "source_table": t.source_table,
                    "row_id": t.row_id,
                    "deleted_ms": t.deleted_ms,
                    "reason": t.reason,
                    "node_type": t.node_type,
                    "tags": t.tags,
                })),
                Err(e) => decode_errors.push(format!("tombstone seq {}: {e}", frame.seq)),
            },
            KIND_SUPERSESSION => match SupersessionRecord::try_from_slice(&frame.payload) {
                Ok(s) => supersessions.push(serde_json::json!({
                    "superseded_legacy_id": s.superseded_legacy_id,
                    "superseded_by_legacy_id": s.superseded_by_legacy_id,
                    "superseded_kernel_id": s.superseded_kernel_id,
                    "superseded_by_kernel_id": s.superseded_by_kernel_id,
                })),
                Err(e) => decode_errors.push(format!("supersession seq {}: {e}", frame.seq)),
            },
            KIND_PARAMS => match ParamsRecord::try_from_slice(&frame.payload) {
                Ok(p) => {
                    params = Some(serde_json::json!({
                        "params_id": p.params_id,
                        "schema_version": p.schema_version,
                        "source_blake3": p.source_blake3,
                        "envelope_head": p.envelope_head,
                    }))
                }
                Err(e) => decode_errors.push(format!("params: {e}")),
            },
            KIND_RECEIPT => match MigrationReceipt::try_from_slice(&frame.payload) {
                Ok(r) => {
                    let (checksum_ok, signature_ok) = verify_receipt(&r);
                    let source = r.body.source_blake3_before.clone();
                    let derived = blake3::derive_key(OLD_SEED_CTX, source.as_bytes());
                    receipt_json = Some(serde_json::json!({
                        "schema_version": r.body.schema_version,
                        "source_blake3_before": r.body.source_blake3_before,
                        "source_blake3_after": r.body.source_blake3_after,
                        "envelope_head": r.body.envelope_head,
                        "counts": r.body.counts,
                        "dropped_vectors": r.body.dropped_vectors,
                        "dropped_columns": r.body.dropped_columns,
                        "signing_key_id": r.body.signing_key_id,
                        "verifying_key": hex(&r.verifying_key),
                        "checksum_ok": checksum_ok,
                        "signature_ok": signature_ok,
                        "old_derived_seed": hex(&derived),
                    }));
                }
                Err(e) => decode_errors.push(format!("receipt: {e}")),
            },
            _ => {}
        }
    }
    drop(log);
    let after = list_files(dir);
    let key_path = dir.join("strata.key");
    let key_bytes = fs::read(&key_path).unwrap_or_default();
    let parent_key = dir.parent().map(|p| p.join("receipt-signing.key"));
    let receipt_key_meta = parent_key.as_ref().map(|p| {
        serde_json::json!({
            "path": p.display().to_string(),
            "exists": p.exists(),
            "mode": mode_of(p),
            "bytes": fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            "inside_to": p.starts_with(dir),
        })
    });
    let derived_matches_key = receipt_json.as_ref().and_then(|r| {
        r.get("old_derived_seed")
            .and_then(|v| v.as_str())
            .map(|d| d == hex(&key_bytes))
    });
    Ok(serde_json::json!({
        "dir": dir.display().to_string(),
        "frames_total": frames.len(),
        "kind_counts": counts,
        "nodes": nodes,
        "edges": edges,
        "tombstones": tombstones,
        "supersessions": supersessions,
        "params": params,
        "receipt": receipt_json,
        "decode_errors": decode_errors,
        "strata_key": {
            "path": key_path.display().to_string(),
            "exists": key_path.exists(),
            "mode": mode_of(&key_path),
            "sha256": real_sha256(&key_path),
            "hex": hex(&key_bytes),
            "equals_old_source_hash_derivation": derived_matches_key,
        },
        "receipt_key": receipt_key_meta,
        "files_before_open": before,
        "files_after_open": after,
    }))
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn segments(log_dir: &Path) -> Vec<PathBuf> {
    let mut segs: Vec<PathBuf> = fs::read_dir(log_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("seg"))
        .collect();
    segs.sort();
    segs
}

fn seg_meta(log_dir: &Path) -> Vec<serde_json::Value> {
    segments(log_dir)
        .into_iter()
        .map(|p| {
            serde_json::json!({
                "name": p.file_name().unwrap().to_string_lossy(),
                "bytes": fs::metadata(&p).map(|m| m.len()).unwrap_or(0),
                "sha256": real_sha256(&p),
            })
        })
        .collect()
}

fn flip_mid(path: &Path) -> std::io::Result<u64> {
    let mut bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "segment too small",
        ));
    }
    let off = bytes.len() / 2;
    bytes[off] ^= 0xff;
    fs::write(path, &bytes)?;
    Ok(off as u64)
}

fn try_open_store(dir: &Path) -> Result<serde_json::Value, String> {
    match StrataStore::open(dir) {
        Ok(store) => Ok(serde_json::json!({
            "opened": true,
            "nodes": store.node_count(),
            "edges": store.edge_count(),
            "reviews": store.review_event_count(),
            "checkpoints": store.checkpoints().len(),
            "digest": hex(&store.state_digest()),
        })),
        Err(e) => Err(e.to_string()),
    }
}

fn try_open_log(dir: &Path) -> Result<u64, String> {
    let log = StrataLog::open(dir).map_err(|e| e.to_string())?;
    let frames = log.read_frames(1).map_err(|e| e.to_string())?;
    Ok(frames.len() as u64)
}

fn populate(dir: &Path, n: usize, seal_log: bool) -> Result<(), String> {
    let mut store = StrataStore::open(dir).map_err(|e| e.to_string())?;
    for i in 0..n {
        let id = store
            .ingest(IngestInput {
                content: format!("verdict node {i}: payments timeout cafe naive"),
                node_type: "fact".into(),
                tags: vec!["verdict".into(), format!("n{i}")],
                created_at_ms: Some(1_700_000_000_000 + i as i64),
                valid_from_ms: None,
                valid_until_ms: None,
            })
            .map_err(|e| e.to_string())?;
        if i % 5 == 0 {
            let _ = store.review(&id, 3);
        }
        if i > 0 && i % 7 == 0 {
            let _ = store.save_connection(&ConnectionRecord {
                source_id: id.clone(),
                target_id: format!("anchor-{i}"),
                strength_milli: 500,
                link_type: "anchored_to".into(),
                meta_sha: None,
                created_at_ms: 1_700_000_000_000,
                activation_count: 1,
            });
        }
    }
    store.seal_checkpoint().map_err(|e| e.to_string())?;
    if seal_log {
        store.log().seal().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn probe_damage(root: &Path, name: &str, seal_log: bool, which: &str) -> serde_json::Value {
    let dir = root.join(name);
    let _ = fs::remove_dir_all(&dir);
    let populated = populate(&dir, 24, seal_log);
    if let Err(e) = populated {
        return serde_json::json!({"name": name, "setup_error": e});
    }
    let log_dir = dir.join("log");
    let before = seg_meta(&log_dir);
    let head = log_dir.join("head.state");
    let head_existed = head.exists();
    let _ = fs::remove_file(&head);
    let segs = segments(&log_dir);
    let target = if which == "first" {
        segs.first().cloned()
    } else {
        segs.last().cloned()
    };
    let Some(target) = target else {
        return serde_json::json!({"name": name, "setup_error": "no segment"});
    };
    let flipped = flip_mid(&target).map_err(|e| e.to_string());
    let open = try_open_store(&dir);
    let after = seg_meta(&log_dir);
    let truncated = segment_truncated(&before, &after);
    serde_json::json!({
        "name": name,
        "seal_log": seal_log,
        "head_state_existed": head_existed,
        "flipped": target.file_name().unwrap().to_string_lossy(),
        "flip_offset": flipped.as_ref().ok(),
        "flip_error": flipped.as_ref().err(),
        "segments_before": before,
        "segments_after": after,
        "open_ok": open.as_ref().ok(),
        "open_err": open.as_ref().err(),
        "requirement": "open must fail and segment bytes must not be truncated",
        "refused": open.is_err(),
        "truncated": truncated,
    })
}

fn segment_truncated(before: &[serde_json::Value], after: &[serde_json::Value]) -> bool {
    before.iter().any(|bv| {
        let name = bv["name"].as_str().unwrap_or("");
        let blen = bv["bytes"].as_u64().unwrap_or(0);
        match after.iter().find(|av| av["name"].as_str() == Some(name)) {
            Some(av) => av["bytes"].as_u64().unwrap_or(0) < blen,
            None => blen > 0,
        }
    })
}

fn probe_missing_key(root: &Path) -> serde_json::Value {
    let dir = root.join("missing-key-unsealed");
    let _ = fs::remove_dir_all(&dir);
    if let Err(e) = populate(&dir, 8, false) {
        return serde_json::json!({"setup_error": e});
    }
    let key = dir.join("log/strata.key");
    let before = fs::read(&key).ok();
    let _ = fs::remove_file(&key);
    let open = try_open_store(&dir);
    let after = fs::read(&key).ok();
    serde_json::json!({
        "key_existed": before.as_ref().map(|b| b.len()),
        "key_deleted": !key.exists() || after.is_some(),
        "open_ok": open.as_ref().ok(),
        "open_err": open.as_ref().err(),
        "key_reappeared": after.is_some(),
        "key_bytes_changed": match (&before, &after) {
            (Some(b), Some(a)) => b != a,
            (Some(_), None) => false,
            (None, Some(_)) => true,
            _ => false,
        },
        "requirement": "missing key on an unsealed log must refuse, not mint a new key",
        "refused": open.is_err(),
        "silently_regenerated": open.is_ok() && after.is_some(),
    })
}

fn probe_missing_meta(root: &Path) -> serde_json::Value {
    let dir = root.join("missing-meta");
    let _ = fs::remove_dir_all(&dir);
    if let Err(e) = populate(&dir, 8, false) {
        return serde_json::json!({"setup_error": e});
    }
    let meta = dir.join("store.meta");
    let existed = meta.exists();
    let meta_bytes = fs::read(&meta).unwrap_or_default().len();
    let _ = fs::remove_file(&meta);
    let open = try_open_store(&dir);
    serde_json::json!({
        "store_meta_existed": existed,
        "store_meta_bytes": meta_bytes,
        "open_ok": open.as_ref().ok(),
        "open_err": open.as_ref().err(),
        "requirement": "missing store.meta on a populated store must error",
        "refused": open.is_err() && existed,
    })
}

fn harden(root: &Path) -> serde_json::Value {
    let _ = fs::create_dir_all(root);
    serde_json::json!({
        "sealed_segment_head_deleted": probe_damage(root, "sealed-damage", true, "first"),
        "unsealed_segment_head_deleted": probe_damage(root, "unsealed-damage", false, "first"),
        "missing_key_unsealed": probe_missing_key(root),
        "missing_store_meta": probe_missing_meta(root),
    })
}

fn mutate_copy(src: &Path, dst: &Path, op: &str) -> Result<(), String> {
    if dst.exists() {
        fs::remove_dir_all(dst).map_err(|e| e.to_string())?;
    }
    copy_dir(src, dst).map_err(|e| e.to_string())?;
    match op {
        "flip" => {
            let segs = segments(dst);
            // Prefer a non-last segment (sealed history). Fall back to the only segment.
            let target = if segs.len() >= 2 {
                segs[0].clone()
            } else {
                segs.first()
                    .cloned()
                    .ok_or_else(|| "no segment to flip".to_string())?
            };
            let off = flip_mid(&target).map_err(|e| e.to_string())?;
            println!("flipped {} at {off}", target.display());
        }
        "replace-derived" => {
            let log = StrataLog::open(dst).map_err(|e| e.to_string())?;
            let frames = log.read_frames(1).map_err(|e| e.to_string())?;
            drop(log);
            let source = frames
                .iter()
                .find(|f| f.kind == KIND_PARAMS)
                .ok_or_else(|| "no params".to_string())?;
            let params = ParamsRecord::try_from_slice(&source.payload).map_err(|e| e.to_string())?;
            let derived = blake3::derive_key(OLD_SEED_CTX, params.source_blake3.as_bytes());
            let key = dst.join("strata.key");
            {
                use std::os::unix::fs::OpenOptionsExt;
                let mut f = fs::OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .create(true)
                    .mode(0o600)
                    .open(&key)
                    .map_err(|e| e.to_string())?;
                f.write_all(&derived).map_err(|e| e.to_string())?;
            }
            println!("replaced strata.key with derive_key({OLD_SEED_CTX:?}, source_blake3)");
        }
        "replace-random" => {
            let mut seed = [0u8; 32];
            seed[0] = 0x5a;
            seed[31] = 0xa5;
            let key = dst.join("strata.key");
            fs::write(&key, &seed).map_err(|e| e.to_string())?;
            println!("replaced strata.key with a fixed non-matching seed");
        }
        other => return Err(format!("unknown op {other}")),
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let result = match cmd {
        "dump" => {
            let dir = PathBuf::from(args.get(1).expect("dir"));
            dump(&dir).map(|v| {
                println!("{}", serde_json::to_string_pretty(&v).unwrap());
            })
        }
        "open-log" => {
            let dir = PathBuf::from(args.get(1).expect("dir"));
            try_open_log(&dir).map(|n| println!("OPEN_OK frames={n}"))
        }
        "harden" => {
            let root = PathBuf::from(args.get(1).expect("root"));
            let v = harden(&root);
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            Ok(())
        }
        "mutate" => {
            let op = args.get(1).expect("op").as_str();
            let src = PathBuf::from(args.get(2).expect("src"));
            let dst = PathBuf::from(args.get(3).expect("dst"));
            mutate_copy(&src, &dst, op).map(|_| {
                match try_open_log(&dst) {
                    Ok(n) => println!("OPEN_OK frames={n}"),
                    Err(e) => {
                        println!("OPEN_FAIL {e}");
                        std::process::exit(1);
                    }
                }
            })
        }
        "live-store" => {
            let dir = PathBuf::from(args.get(1).expect("dir"));
            populate(&dir, 12, true).map(|_| println!("LIVE_STORE {}", dir.display()))
        }
        "probe-nometa" => (|| -> Result<(), String> {
            let dir = PathBuf::from(args.get(1).expect("dir"));
            let _ = fs::remove_dir_all(&dir);
            {
                let mut store = StrataStore::open(&dir).map_err(|e| e.to_string())?;
                for i in 0..20 {
                    store
                        .ingest(IngestInput {
                            content: format!("nometa node {i}"),
                            node_type: "fact".into(),
                            tags: vec!["n".into()],
                            created_at_ms: Some(1_700_000_000_000),
                            valid_from_ms: None,
                            valid_until_ms: None,
                        })
                        .map_err(|e| e.to_string())?;
                }
            }
            let seg = segments(&dir.join("log"))
                .into_iter()
                .next()
                .ok_or_else(|| "no segment".to_string())?;
            let before = fs::metadata(&seg).map_err(|e| e.to_string())?.len();
            let _ = fs::remove_file(dir.join("log/head.state"));
            let _ = fs::remove_file(dir.join("store.meta"));
            let off = flip_mid(&seg).map_err(|e| e.to_string())?;
            let open = try_open_store(&dir);
            let after = fs::metadata(&seg).map(|m| m.len()).unwrap_or(0);
            println!(
                "{}",
                serde_json::json!({
                    "before": before,
                    "after": after,
                    "flip_offset": off,
                    "truncated": after < before,
                    "open_ok": open.as_ref().ok(),
                    "open_err": open.as_ref().err(),
                })
            );
            Ok(())
        })(),
        _ => Err(
            "usage: dump <logdir> | open-log <logdir> | harden <root> | mutate flip|replace-derived|replace-random <src> <dst> | live-store <dir>"
                .into(),
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("harness error: {e}");
            ExitCode::from(2)
        }
    }
}

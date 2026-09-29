//! Live probes for the e5d04a3 verdict. Prints one JSON object to stdout.
//! Mutating commands operate only on the directory they are given.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use strata::StrataLog;
use strata_migrate::records::{
    decode_edge, decode_node, decode_receipt, decode_review, KIND_EDGE, KIND_FSRS_REVIEW,
    KIND_MIGRATION_RECEIPT, KIND_NODE,
};
use strata_store::{IngestInput, StrataStore};

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| usage());
    let rest: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "dump" => dump(Path::new(&need(&rest, 0))),
        "live" => live(Path::new(&need(&rest, 0))),
        "check7-truncate" => check7_truncate(Path::new(&need(&rest, 0))),
        "check7-key" => check7_key(Path::new(&need(&rest, 0))),
        "check7-meta" => check7_meta(Path::new(&need(&rest, 0))),
        "forge-replaced-key" => forge_replaced_key(Path::new(&need(&rest, 0))),
        "forge-resigned-receipt" => forge_resigned_receipt(Path::new(&need(&rest, 0))),
        other => {
            eprintln!("unknown command {other}");
            usage();
        }
    };
    match result {
        Ok(value) => {
            println!("{value}");
        }
        Err(err) => {
            eprintln!("harness error: {err}");
            std::process::exit(1);
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: verdict-harness <dump|live|check7-truncate|check7-key|check7-meta|forge-replaced-key|forge-resigned-receipt> <dir>"
    );
    std::process::exit(2);
}

fn need(args: &[String], idx: usize) -> String {
    args.get(idx).cloned().unwrap_or_else(|| usage())
}

fn dump(dir: &Path) -> Result<String, String> {
    let log = StrataLog::open(dir).map_err(|e| format!("open: {e}"))?;
    let frames = log.read_frames(1).map_err(|e| format!("read: {e}"))?;
    let mut kind_counts = serde_json::Map::new();
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut fsrs = Vec::new();
    let mut receipt = serde_json::Value::Null;
    for frame in &frames {
        let key = frame.kind.to_string();
        let entry = kind_counts.entry(key).or_insert(serde_json::json!(0));
        *entry = serde_json::json!(entry.as_u64().unwrap_or(0) + 1);
        match frame.kind {
            KIND_NODE => {
                let node = decode_node(&frame.payload).map_err(|e| e.to_string())?;
                let legacy_keys: Vec<&str> = node.legacy.iter().map(|(k, _)| k.as_str()).collect();
                let fsrs_legacy: Vec<serde_json::Value> = node
                    .legacy
                    .iter()
                    .filter(|(k, _)| k.starts_with("fsrs_cards."))
                    .map(|(k, v)| serde_json::json!({"k": k, "v": v}))
                    .collect();
                nodes.push(serde_json::json!({
                    "legacy_id": node.legacy_id,
                    "kernel_id": node.kernel_id,
                    "node_type": node.node_type,
                    "tags": node.tags,
                    "legacy_keys": legacy_keys,
                    "fsrs_legacy": fsrs_legacy,
                }));
            }
            KIND_EDGE => {
                let edge = decode_edge(&frame.payload).map_err(|e| e.to_string())?;
                edges.push(serde_json::json!({
                    "link_type": edge.link_type,
                    "legacy_inferred": edge.legacy_inferred,
                    "legacy_link_type": edge.legacy_link_type,
                    "legacy_keys": edge.legacy.iter().map(|(k, _)| k).collect::<Vec<_>>(),
                }));
            }
            KIND_FSRS_REVIEW => {
                let review = decode_review(&frame.payload).map_err(|e| e.to_string())?;
                fsrs.push(serde_json::json!({
                    "card_id": review.card_id,
                    "rating": review.rating,
                    "event_seq": review.event_seq,
                }));
            }
            KIND_MIGRATION_RECEIPT => {
                let rec = decode_receipt(&frame.payload).map_err(|e| e.to_string())?;
                receipt = serde_json::json!({
                    "schema_version": rec.body.schema_version,
                    "counts": rec.body.counts,
                    "dropped_columns": rec.body.dropped_columns,
                    "dropped_vectors": rec.body.dropped_vectors,
                    "checksum_ok": rec.verify_checksum(),
                    "signature_ok": rec.verify_signature(),
                    "source_blake3_before": rec.body.source_blake3_before,
                    "source_blake3_after": rec.body.source_blake3_after,
                    "verifying_key_hex": hex(&rec.verifying_key),
                });
            }
            _ => {}
        }
    }
    Ok(serde_json::json!({
        "frames": frames.len(),
        "kind_counts": kind_counts,
        "nodes": nodes,
        "edges": edges,
        "fsrs_reviews": fsrs,
        "receipt": receipt,
    })
    .to_string())
}

fn live(dir: &Path) -> Result<String, String> {
    let mut store = StrataStore::open(dir).map_err(|e| format!("open: {e}"))?;
    let id = store
        .ingest(IngestInput {
            content: "verdict live store row".into(),
            node_type: String::new(),
            tags: vec!["verdict".into()],
            created_at_ms: Some(1_700_000_000_000),
            valid_from_ms: None,
            valid_until_ms: None,
        })
        .map_err(|e| format!("ingest: {e}"))?;
    let cp = store
        .seal_checkpoint()
        .map_err(|e| format!("checkpoint: {e}"))?;
    let entries: Vec<String> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    Ok(serde_json::json!({
        "node_id": id,
        "checkpoint_log_seq": cp.log_seq,
        "top_entries": entries,
        "has_store_meta": dir.join("store.meta").is_file(),
        "log_dir": dir.join("log").is_dir(),
    })
    .to_string())
}

/// Delete head.state on an unsealed log that already acked frames, then
/// flip one payload byte. Report whether reopen refuses or truncates.
fn check7_truncate(dir: &Path) -> Result<String, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let log = StrataLog::open(dir).map_err(|e| format!("open: {e}"))?;
    for i in 0..5 {
        log.append(1, format!("acked-frame-{i}").as_bytes())
            .map_err(|e| format!("append: {e}"))?;
    }
    drop(log);
    let head = dir.join("head.state");
    let head_existed = head.is_file();
    fs::remove_file(&head).map_err(|e| format!("remove head.state: {e}"))?;
    let seg = first_seg(dir)?;
    let before = fs::read(&seg).map_err(|e| e.to_string())?;
    let mut damaged = before.clone();
    // Flip a byte inside the first frame payload, past the fixed header.
    let flip_at = 80.min(damaged.len().saturating_sub(1));
    damaged[flip_at] ^= 0xff;
    fs::write(&seg, &damaged).map_err(|e| e.to_string())?;
    let len_before = damaged.len();
    let reopen = StrataLog::open(dir);
    let len_after = fs::metadata(&seg).map(|m| m.len()).unwrap_or(0);
    match reopen {
        Ok(log) => {
            let frames = log.read_frames(1).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "head_state_existed_before_delete": head_existed,
                "reopen": "ok",
                "frames_after": frames.len(),
                "segment_len_before": len_before,
                "segment_len_after": len_after,
                "truncated": len_after < len_before as u64,
                "verdict": if len_after < len_before as u64 { "SILENT_TRUNCATE" } else { "OPENED_WITHOUT_TRUNCATE" },
            })
            .to_string())
        }
        Err(err) => Ok(serde_json::json!({
            "head_state_existed_before_delete": head_existed,
            "reopen": "err",
            "error": err.to_string(),
            "segment_len_before": len_before,
            "segment_len_after": len_after,
            "truncated": len_after < len_before as u64,
            "verdict": "REFUSED",
        })
        .to_string()),
    }
}

/// Delete strata.key on an unsealed log and reopen.
fn check7_key(dir: &Path) -> Result<String, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let log = StrataLog::open(dir).map_err(|e| format!("open: {e}"))?;
    log.append(1, b"unsealed-payload")
        .map_err(|e| format!("append: {e}"))?;
    drop(log);
    let key_path = dir.join("strata.key");
    let original = fs::read(&key_path).map_err(|e| e.to_string())?;
    fs::remove_file(&key_path).map_err(|e| e.to_string())?;
    let reopen = StrataLog::open(dir);
    let key_reappeared = key_path.is_file();
    let new_key = fs::read(&key_path).unwrap_or_default();
    match reopen {
        Ok(log) => {
            let frames = log.read_frames(1).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "reopen": "ok",
                "frames_after": frames.len(),
                "key_reappeared": key_reappeared,
                "key_changed": new_key != original,
                "verdict": if key_reappeared && new_key != original { "SILENT_REGEN" } else { "OPENED" },
            })
            .to_string())
        }
        Err(err) => Ok(serde_json::json!({
            "reopen": "err",
            "error": err.to_string(),
            "key_reappeared": key_reappeared,
            "verdict": "REFUSED",
        })
        .to_string()),
    }
}

/// Checkpoint a store, delete store.meta, reopen.
fn check7_meta(dir: &Path) -> Result<String, String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    {
        let mut store = StrataStore::open(dir).map_err(|e| format!("open: {e}"))?;
        store
            .ingest(IngestInput {
                content: "meta probe".into(),
                node_type: String::new(),
                tags: vec![],
                created_at_ms: Some(1),
                valid_from_ms: None,
                valid_until_ms: None,
            })
            .map_err(|e| format!("ingest: {e}"))?;
        store
            .seal_checkpoint()
            .map_err(|e| format!("checkpoint: {e}"))?;
    }
    let meta = dir.join("store.meta");
    if !meta.is_file() {
        return Err("store.meta was not written by seal_checkpoint".into());
    }
    fs::remove_file(&meta).map_err(|e| e.to_string())?;
    match StrataStore::open(dir) {
        Ok(store) => Ok(serde_json::json!({
            "reopen": "ok",
            "checkpoints": store.checkpoints().len(),
            "verdict": "MISSING_META_ACCEPTED",
        })
        .to_string()),
        Err(err) => Ok(serde_json::json!({
            "reopen": "err",
            "error": err.to_string(),
            "verdict": "REFUSED",
        })
        .to_string()),
    }
}

/// Replace strata.key with a fresh random key and with blake3(source hash).
fn forge_replaced_key(dir: &Path) -> Result<String, String> {
    let key_path = dir.join("strata.key");
    let original = fs::read(&key_path).map_err(|e| format!("read key: {e}"))?;
    let log = StrataLog::open(dir).map_err(|e| format!("baseline open: {e}"))?;
    let frames = log.read_frames(1).map_err(|e| e.to_string())?;
    drop(log);
    let source_blake3 = frames
        .iter()
        .find(|f| f.kind == KIND_MIGRATION_RECEIPT)
        .and_then(|f| decode_receipt(&f.payload).ok())
        .map(|r| r.body.source_blake3_before)
        .unwrap_or_default();

    let random_key = random_32()?;
    write_key(&key_path, &random_key)?;
    let random_open = StrataLog::open(dir).map(|l| l.read_frames(1).map(|f| f.len()).unwrap_or(0));

    let derived = *blake3::hash(source_blake3.as_bytes()).as_bytes();
    write_key(&key_path, &derived)?;
    let derived_open = StrataLog::open(dir).map(|l| l.read_frames(1).map(|f| f.len()).unwrap_or(0));

    // Restore so the copy is not left with a forged key if the caller inspects it.
    write_key(&key_path, &original)?;
    let _ = StrataLog::open(dir);

    Ok(serde_json::json!({
        "source_blake3": source_blake3,
        "random_key_open": match &random_open {
            Ok(n) => format!("ok frames={n}"),
            Err(e) => format!("err {e}"),
        },
        "blake3_source_derived_key_open": match &derived_open {
            Ok(n) => format!("ok frames={n}"),
            Err(e) => format!("err {e}"),
        },
        "random_refused": random_open.is_err(),
        "derived_refused": derived_open.is_err(),
    })
    .to_string())
}

/// Re-sign the receipt body with a brand-new key (new verifying key + signature)
/// inside the segment bytes, without the log's strata.key. Verify must fail.
fn forge_resigned_receipt(dir: &Path) -> Result<String, String> {
    let log = StrataLog::open(dir).map_err(|e| format!("baseline open: {e}"))?;
    let frames = log.read_frames(1).map_err(|e| e.to_string())?;
    drop(log);
    let receipt_frame = frames
        .iter()
        .find(|f| f.kind == KIND_MIGRATION_RECEIPT)
        .ok_or("no receipt")?;
    let receipt = decode_receipt(&receipt_frame.payload).map_err(|e| e.to_string())?;
    let mut seed = [0u8; 32];
    fill_urandom(&mut seed)?;
    let signing = ed25519_dalek::SigningKey::from_bytes(&seed);
    let forged = strata_migrate::records::MigrationReceipt::seal(receipt.body.clone(), &signing);
    let forged_bytes = borsh::to_vec(&forged).map_err(|e| e.to_string())?;
    let seg = largest_seg(dir)?;
    let raw = fs::read(&seg).map_err(|e| e.to_string())?;
    let original_payload = &receipt_frame.payload;
    let pos = find_subslice(&raw, original_payload).ok_or("receipt payload bytes not found in segment")?;
    if forged_bytes.len() != original_payload.len() {
        return Ok(serde_json::json!({
            "replaced_in_place": false,
            "reason": "forged receipt length differs; in-place splice would desync framing",
            "original_len": original_payload.len(),
            "forged_len": forged_bytes.len(),
            "forged_signature_self_verifies": forged.verify_signature(),
            "note": "self-verify of a detached forged receipt is expected true; the log must still reject it",
        })
        .to_string());
    }
    let mut patched = raw.clone();
    patched[pos..pos + forged_bytes.len()].copy_from_slice(&forged_bytes);
    fs::write(&seg, &patched).map_err(|e| e.to_string())?;
    let reopen = StrataLog::open(dir);
    Ok(serde_json::json!({
        "replaced_in_place": true,
        "reopen": match &reopen {
            Ok(_) => "ok".to_string(),
            Err(e) => format!("err {e}"),
        },
        "refused": reopen.is_err(),
    })
    .to_string())
}

fn first_seg(dir: &Path) -> Result<PathBuf, String> {
    let mut segs: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("seg"))
        .collect();
    segs.sort();
    segs.into_iter().next().ok_or_else(|| "no segment".into())
}

fn largest_seg(dir: &Path) -> Result<PathBuf, String> {
    let mut segs: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("seg"))
        .collect();
    segs.sort_by_key(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    segs.pop().ok_or_else(|| "no segment".into())
}

fn write_key(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    Ok(())
}

fn random_32() -> Result<[u8; 32], String> {
    let mut buf = [0u8; 32];
    fill_urandom(&mut buf)?;
    Ok(buf)
}

fn fill_urandom(buf: &mut [u8]) -> Result<(), String> {
    use std::io::Read;
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(buf))
        .map_err(|e| e.to_string())
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

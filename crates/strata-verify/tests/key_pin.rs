//! A rewritten receipt that carries a new embedded key must fail the folder
//! pin. Swapping `receipt-signing.key` as well does not satisfy `--expect-key`
//! of the original fingerprint.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use strata::{
    HEADER_WIRE_SIZE, SegmentHeader, SegmentTrailer, TRAILER_WIRE_SIZE, frame_hash, header_hash,
    merkle_root, parse_frame, payload_blake3, signature_message,
};
use strata_migrate::records::{KIND_MIGRATION_RECEIPT, MigrationReceipt, decode_receipt};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../strata-migrate/tests/fixtures/v3.1.1-sample.sqlite"
);

fn fingerprint(public_key: &[u8; 32]) -> String {
    blake3::hash(public_key).to_hex().to_string()
}

fn build_log(dir: &Path, seed: [u8; 32]) {
    std::fs::write(dir.parent().unwrap().join("receipt-signing.key"), seed).unwrap();
    strata_migrate::migrate_with_options(
        Path::new(FIXTURE),
        dir,
        strata_migrate::MigrateOptions {
            seed: Some(seed),
            ..Default::default()
        },
    )
    .expect("migration");
}

/// Replace the receipt with one sealed by `new_key`, and re-sign the segment
/// trailer with the on-disk `strata.key`. The log format is unchanged.
fn segment_has_receipt(path: &Path) -> bool {
    let bytes = std::fs::read(path).unwrap();
    if bytes.len() <= HEADER_WIRE_SIZE {
        return false;
    }
    let mut rest = &bytes[HEADER_WIRE_SIZE..];
    while rest.len() > TRAILER_WIRE_SIZE {
        let Ok((frame, used)) = parse_frame(rest) else {
            return false;
        };
        if frame.kind == KIND_MIGRATION_RECEIPT {
            return true;
        }
        rest = &rest[used..];
    }
    false
}

fn resign_receipt(log_dir: &Path, new_key: &SigningKey) {
    let seg = std::fs::read_dir(log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "seg"))
        .find(|path| segment_has_receipt(path))
        .expect("segment with a receipt");
    let bytes = std::fs::read(&seg).unwrap();
    let header: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).unwrap();
    let mut rest = &bytes[HEADER_WIRE_SIZE..];
    let mut frames = Vec::new();
    while rest.len() > TRAILER_WIRE_SIZE {
        let (frame, used) = parse_frame(rest).unwrap();
        rest = &rest[used..];
        frames.push(frame);
    }
    assert_eq!(rest.len(), TRAILER_WIRE_SIZE);
    let idx = frames
        .iter()
        .position(|frame| frame.kind == KIND_MIGRATION_RECEIPT)
        .expect("receipt frame");
    let old = decode_receipt(&frames[idx].payload).unwrap();
    let receipt = MigrationReceipt::seal(old.body, new_key);
    frames[idx].payload = borsh::to_vec(&receipt).unwrap();

    let mut prev = header_hash(&header);
    let mut leaves = Vec::new();
    for frame in &mut frames {
        frame.payload_blake3 = payload_blake3(frame.kind, &frame.payload);
        frame.prev_frame_hash = prev;
        prev = frame_hash(frame);
        leaves.push(frame.payload_blake3);
    }
    let root = merkle_root(&leaves);
    let strata_seed: [u8; 32] = std::fs::read(log_dir.join("strata.key"))
        .unwrap()
        .try_into()
        .unwrap();
    let strata_key = SigningKey::from_bytes(&strata_seed);
    let msg = signature_message(&header.segment_id, &header.prev_segment_hash, &root);
    let trailer = SegmentTrailer {
        frame_count: frames.len() as u64,
        merkle_root: root,
        signature: strata_key.sign(&msg).to_bytes(),
    };
    let mut out = borsh::to_vec(&header).unwrap();
    for frame in &frames {
        out.extend(borsh::to_vec(frame).unwrap());
    }
    out.extend(borsh::to_vec(&trailer).unwrap());
    std::fs::write(&seg, &out).unwrap();
    // seal() opens the next segment. Its prev hash must follow the rewritten file.
    let segment_hash = *blake3::hash(&out).as_bytes();
    let mut later: Vec<PathBuf> = std::fs::read_dir(log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "seg") && path != &seg)
        .collect();
    later.sort();
    for path in later {
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            bytes.len(),
            HEADER_WIRE_SIZE,
            "only the empty segment after seal is rewritten"
        );
        let mut header: SegmentHeader = borsh::from_slice(&bytes).unwrap();
        header.prev_segment_hash = segment_hash;
        bytes = borsh::to_vec(&header).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}

fn vestige_if_set(log_dir: &Path, expect: Option<&str>, want_ok: bool) {
    let Ok(bin) = std::env::var("VESTIGE_BIN") else {
        return;
    };
    let output = run_verify(&bin, log_dir, expect, true);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("--- vestige strata-verify expect={expect:?} ---\n{stdout}{stderr}");
    assert_eq!(
        output.status.success(),
        want_ok,
        "vestige exit {:?}",
        output.status.code()
    );
}

fn run_verify(
    bin: &str,
    log_dir: &Path,
    expect: Option<&str>,
    subcommand: bool,
) -> std::process::Output {
    let mut cmd = Command::new(bin);
    if subcommand {
        cmd.arg("strata-verify");
    }
    if let Some(expect) = expect {
        cmd.arg("--expect-key").arg(expect);
    }
    cmd.arg(log_dir).output().unwrap()
}

#[test]
fn honest_log_prints_the_pinned_fingerprint() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir = tmp.path().join("strata");
    let seed = [9u8; 32];
    build_log(&log_dir, seed);
    let expected = fingerprint(&SigningKey::from_bytes(&seed).verifying_key().to_bytes());
    let report = strata_verify::verify_path(&log_dir);
    assert!(report.ok, "{}", report.json);
    assert_eq!(report.key_fingerprint, expected);
    assert!(report.json.contains("\"key_pin\": \"receipt-signing.key\""));

    let output = run_verify(env!("CARGO_BIN_EXE_strata-verify"), &log_dir, None, false);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains(&format!("key fingerprint: {expected}")));
    vestige_if_set(&log_dir, None, true);
    vestige_if_set(&log_dir, Some(&expected), true);
}

#[test]
fn resigned_embedded_key_fails_the_folder_pin() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir = tmp.path().join("strata");
    let original = [9u8; 32];
    build_log(&log_dir, original);
    let original_fp = fingerprint(&SigningKey::from_bytes(&original).verifying_key().to_bytes());
    let fresh = SigningKey::from_bytes(&[4u8; 32]);
    resign_receipt(&log_dir, &fresh);

    let report = strata_verify::verify_path(&log_dir);
    assert!(!report.ok, "rewritten receipt must fail: {}", report.json);
    assert!(
        report
            .failures
            .iter()
            .any(|f| f.contains("does not match pinned receipt-signing.key")),
        "{:?}",
        report.failures
    );
    vestige_if_set(&log_dir, None, false);

    let fresh_fp = fingerprint(&fresh.verifying_key().to_bytes());
    // Folder key swapped to the new signer: the pin matches, --expect-key of
    // the original fingerprint still fails, and the new fingerprint passes.
    std::fs::write(tmp.path().join("receipt-signing.key"), [4u8; 32]).unwrap();
    let swapped = strata_verify::verify_path(&log_dir);
    assert!(
        swapped.ok,
        "swapped pin matches the new embedded key: {}",
        swapped.json
    );
    assert_eq!(swapped.key_fingerprint, fresh_fp);

    let rejected = run_verify(
        env!("CARGO_BIN_EXE_strata-verify"),
        &log_dir,
        Some(&original_fp),
        false,
    );
    assert!(!rejected.status.success());
    let err = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        err.contains("does not match --expect-key"),
        "{err}\n{}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    vestige_if_set(&log_dir, Some(&original_fp), false);

    let accepted = run_verify(
        env!("CARGO_BIN_EXE_strata-verify"),
        &log_dir,
        Some(&fresh_fp),
        false,
    );
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stdout)
    );
    vestige_if_set(&log_dir, Some(&fresh_fp), true);
}

/// A store opened by the real stdio server has no migration receipt and no
/// `receipt-signing.key`. The folder pin is `log/strata.key`. The active
/// segment is unsealed, so it carries no embedded verifying key. Signing
/// that segment with a different key fails the pin; swapping `strata.key`
/// afterwards still fails `--expect-key` of the original fingerprint.
#[test]
fn live_stdio_store_pins_strata_key() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let nodes = write_live_store(dir);
    assert!(nodes.len() >= 2, "gated writes: {nodes:?}");
    assert!(
        !dir.join("receipt-signing.key").exists(),
        "a live store must not mint a migration receipt key"
    );
    let log_dir = dir.join("log");
    let seed: [u8; 32] = std::fs::read(log_dir.join("strata.key"))
        .unwrap()
        .try_into()
        .unwrap();
    let original_fp = fingerprint(&SigningKey::from_bytes(&seed).verifying_key().to_bytes());

    let honest = run_verify(&vestige_bin().to_string_lossy(), dir, None, true);
    let stdout = String::from_utf8_lossy(&honest.stdout);
    let stderr = String::from_utf8_lossy(&honest.stderr);
    eprintln!("--- live strata-verify ---\n{stdout}{stderr}");
    assert!(honest.status.success(), "{stdout}{stderr}");
    assert!(stdout.contains(&format!("key fingerprint: {original_fp}")));
    assert!(stdout.contains("\"key_pin\": \"strata.key\""));
    let report = strata_verify::verify_path(dir);
    assert!(report.ok, "{}", report.json);
    assert_eq!(report.key_fingerprint, original_fp);

    let fresh = SigningKey::from_bytes(&[4u8; 32]);
    sign_unsealed_segment(&log_dir, &fresh);
    let resigned = strata_verify::verify_path(dir);
    assert!(!resigned.ok, "foreign trailer must fail: {}", resigned.json);
    assert!(
        resigned
            .failures
            .iter()
            .any(|f| f.contains("does not match pinned strata.key")),
        "{:?}",
        resigned.failures
    );
    let rejected_bin = run_verify(&vestige_bin().to_string_lossy(), dir, None, true);
    assert!(!rejected_bin.status.success());

    std::fs::write(log_dir.join("strata.key"), fresh.to_bytes()).unwrap();
    let fresh_fp = fingerprint(&fresh.verifying_key().to_bytes());
    let swapped = strata_verify::verify_path(dir);
    assert!(
        swapped.ok,
        "swapped strata.key matches the new trailer: {}",
        swapped.json
    );
    assert_eq!(swapped.key_fingerprint, fresh_fp);
    let expect_old = run_verify(
        &vestige_bin().to_string_lossy(),
        dir,
        Some(&original_fp),
        true,
    );
    let err = String::from_utf8_lossy(&expect_old.stderr);
    assert!(
        !expect_old.status.success(),
        "{}",
        String::from_utf8_lossy(&expect_old.stdout)
    );
    assert!(
        err.contains("does not match --expect-key"),
        "{err}\n{}",
        String::from_utf8_lossy(&expect_old.stdout)
    );
    let expect_new = run_verify(&vestige_bin().to_string_lossy(), dir, Some(&fresh_fp), true);
    assert!(
        expect_new.status.success(),
        "{}",
        String::from_utf8_lossy(&expect_new.stdout)
    );
}

fn vestige_bin() -> PathBuf {
    if let Ok(path) = std::env::var("VESTIGE_BIN") {
        return PathBuf::from(path);
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/debug/vestige{}", std::env::consts::EXE_SUFFIX));
    assert!(path.is_file(), "build vestige at {}", path.display());
    path
}

fn mcp_bin() -> PathBuf {
    if let Ok(path) = std::env::var("VESTIGE_MCP_BIN") {
        return PathBuf::from(path);
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/debug/vestige-mcp{}", std::env::consts::EXE_SUFFIX));
    assert!(path.is_file(), "build vestige-mcp at {}", path.display());
    path
}

struct StdioServer {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl StdioServer {
    fn spawn(dir: &Path) -> Self {
        let mut child = Command::new(mcp_bin())
            .env("VESTIGE_DATA_DIR", dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let raw_err = child.stderr.take().unwrap();
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&stderr);
        std::thread::spawn(move || {
            for line in BufReader::new(raw_err).lines().map_while(Result::ok) {
                if let Ok(mut sink) = sink.lock() {
                    sink.push(line);
                }
            }
        });
        Self {
            child,
            stdin,
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn send(&mut self, value: serde_json::Value) {
        let mut line = value.to_string();
        line.push('\n');
        self.stdin
            .as_mut()
            .unwrap()
            .write_all(line.as_bytes())
            .and_then(|()| self.stdin.as_mut().unwrap().flush())
            .unwrap_or_else(|err| {
                panic!(
                    "stdio write: {err}; stderr {:?}",
                    self.stderr.lock().unwrap()
                )
            });
    }

    fn call(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }));
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).expect("stdio read");
            assert!(n > 0, "stdio eof; stderr {:?}", self.stderr.lock().unwrap());
            let value: serde_json::Value = serde_json::from_str(&line).expect("json");
            if value.get("id").and_then(|v| v.as_u64()) == Some(id) {
                return value;
            }
        }
    }

    fn stop(mut self) {
        self.stdin.take();
        let status = self.child.wait().expect("wait vestige-mcp");
        assert!(
            status.success(),
            "stdio exit {status:?}; stderr {:?}",
            self.stderr.lock().unwrap()
        );
    }
}

impl Drop for StdioServer {
    fn drop(&mut self) {
        self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn tool_body(response: &serde_json::Value) -> serde_json::Value {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    serde_json::from_str(text).unwrap_or_else(|_| response.clone())
}

fn write_live_store(dir: &Path) -> Vec<String> {
    let mut server = StdioServer::spawn(dir);
    let init = server.call(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "key-pin", "version": "1" },
        }),
    );
    assert!(init.get("result").is_some(), "{init}");
    server.send(serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }));
    let mut nodes = Vec::new();
    for content in [
        "Pin the live strata key in the store folder, not inside a frame.",
        "A gated ingest returns an effect receipt the verifier can name.",
        "Segment trailers are checked against strata.key on disk.",
    ] {
        let res = server.call(
            "tools/call",
            serde_json::json!({
                "name": "smart_ingest",
                "arguments": { "content": content, "forceCreate": true, "tags": ["pin-probe"] },
            }),
        );
        let body = tool_body(&res);
        assert_eq!(body["success"], serde_json::json!(true), "{body}");
        let node = body["nodeId"].as_str().expect("node id").to_string();
        let receipt = server.call(
            "tools/call",
            serde_json::json!({
                "name": "receipt",
                "arguments": { "action": "get", "receipt_id": node },
            }),
        );
        let receipt_body = tool_body(&receipt);
        let receipt_id = receipt_body["receipt"]["receipt_id"].as_str().unwrap_or("");
        assert!(
            receipt_id.starts_with("eff-"),
            "gated write must return an effect receipt: {receipt_body}"
        );
        nodes.push(node);
    }
    server.stop();
    nodes
}

/// Sign the unsealed live segment with `key`. The frames stay as the server
/// wrote them; only a trailer is appended. The log format is unchanged.
fn sign_unsealed_segment(log_dir: &Path, key: &SigningKey) {
    let seg = std::fs::read_dir(log_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "seg"))
        .expect("live segment");
    let bytes = std::fs::read(&seg).unwrap();
    let header: SegmentHeader = borsh::from_slice(&bytes[..HEADER_WIRE_SIZE]).unwrap();
    let mut rest = &bytes[HEADER_WIRE_SIZE..];
    let mut leaves = Vec::new();
    while !rest.is_empty() {
        assert_ne!(
            rest.len(),
            TRAILER_WIRE_SIZE,
            "live segment is already sealed"
        );
        let (frame, used) = parse_frame(rest).unwrap();
        rest = &rest[used..];
        leaves.push(frame.payload_blake3);
    }
    assert!(!leaves.is_empty());
    let root = merkle_root(&leaves);
    let msg = signature_message(&header.segment_id, &header.prev_segment_hash, &root);
    let trailer = SegmentTrailer {
        frame_count: leaves.len() as u64,
        merkle_root: root,
        signature: key.sign(&msg).to_bytes(),
    };
    let mut out = bytes;
    out.extend(borsh::to_vec(&trailer).unwrap());
    std::fs::write(seg, out).unwrap();
}

#[test]
fn migrated_sample_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let log_dir: PathBuf = tmp.path().join("strata");
    build_log(&log_dir, [7u8; 32]);
    let output = run_verify(env!("CARGO_BIN_EXE_strata-verify"), &log_dir, None, false);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{stdout}");
    assert!(stdout.contains("\"ok\": true"));
    assert!(stdout.contains("key fingerprint:"));
    vestige_if_set(&log_dir, None, true);
}

//! Process harness shared by the release matrix.
//!
//! Product binaries (`vestige`, `vestige-mcp`, `strata-verify`) are the ones
//! the workflow builds. `strata-driver` is this crate's own binary, spawned
//! as a separate process for log and store operations.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const MISSING_HANDLE: &str = "mem:00000000-0000-4000-8000-000000000000";

pub const EXPECTED_TOOLS: &[&str] = &[
    "causal_walk",
    "codebase",
    "dedup",
    "forgotten_lesson",
    "graph",
    "intention",
    "maintain",
    "memory",
    "memory_status",
    "project",
    "purge",
    "recall",
    "receipt",
    "selftest",
    "session_start",
    "smart_ingest",
    "source_sync",
    "suppress",
];

pub fn missing(what: &str) -> ! {
    panic!("MISSING: {what}");
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("tests/release-e2e lives two levels under the repo root")
        .to_path_buf()
}

pub fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/v3")
        .join(name)
}

pub fn driver_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_strata-driver"))
}

pub fn product_bin(name: &str) -> PathBuf {
    let key = match name {
        "vestige" => "VESTIGE_BIN",
        "vestige-mcp" => "VESTIGE_MCP_BIN",
        "strata-verify" => "STRATA_VERIFY_BIN",
        other => panic!("HARNESS: unknown product binary {other}"),
    };
    if let Ok(raw) = std::env::var(key) {
        let path = PathBuf::from(&raw);
        if path.is_file() {
            return path;
        }
        panic!("HARNESS: {key}={raw} is not a file");
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target"));
    for profile in ["debug", "release"] {
        let path = target.join(profile).join(name);
        if path.is_file() {
            return path;
        }
    }
    panic!(
        "HARNESS: {name} was not found under {target}. Build it before the matrix \
         (cargo build -p vestige-mcp --bins; cargo build --manifest-path crates/strata-verify/Cargo.toml --bin strata-verify) \
         or set {key}.",
        target = target.display()
    );
}

pub struct CmdOut {
    pub status: Option<i32>,
    pub signal: bool,
    pub stdout: String,
    pub stderr: String,
}

pub fn run_cmd(
    bin: &Path,
    args: &[String],
    env: &[(&str, &str)],
    clear_env: &[&str],
    timeout: Duration,
) -> CmdOut {
    let mut command = Command::new(bin);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("RUST_LOG", "error");
    for key in clear_env {
        command.env_remove(key);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .unwrap_or_else(|e| panic!("HARNESS: failed to spawn {}: {e}", bin.display()));
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let (out_rx, err_rx) = (pipe_to_string(stdout), pipe_to_string(stderr));
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return CmdOut {
                status: None,
                signal: true,
                stdout: recv_all(out_rx),
                stderr: recv_all(err_rx) + "\n[killed: timeout]",
            };
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    CmdOut {
        signal: false,
        status: status.code(),
        stdout: recv_all(out_rx),
        stderr: recv_all(err_rx),
    }
}

fn pipe_to_string(mut reader: impl Read + Send + 'static) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = reader.read_to_string(&mut buf);
        let _ = tx.send(buf);
    });
    rx
}

fn recv_all(rx: Receiver<String>) -> String {
    rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default()
}

pub fn run_vestige(args: &[String], home: &Path, timeout: Duration) -> CmdOut {
    run_cmd(
        &product_bin("vestige"),
        args,
        &[("HOME", home.to_str().unwrap())],
        &["VESTIGE_DATA_DIR"],
        timeout,
    )
}

pub fn run_driver(args: &[&str]) -> Value {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let out = run_cmd(&driver_bin(), &owned, &[], &[], Duration::from_secs(60));
    let text = if out.stdout.trim().is_empty() {
        out.stderr.clone()
    } else {
        out.stdout.clone()
    };
    serde_json::from_str(text.lines().last().unwrap_or("{}")).unwrap_or_else(|_| {
        json!({
            "ok": false,
            "error": "driver emitted no JSON",
            "stdout": out.stdout,
            "stderr": out.stderr,
            "status": out.status,
        })
    })
}

pub fn sha256_file(path: &Path) -> String {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hex_encode(&hasher.finalize())
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

pub fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &to);
        } else {
            fs::copy(entry.path(), &to).unwrap();
        }
    }
}

pub fn sqlite_artifacts(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.ends_with(".sqlite")
                || name.ends_with(".sqlite3")
                || name.ends_with(".db")
                || name.ends_with(".db-wal")
                || name.ends_with(".db-shm")
                || name.contains("sqlite")
            {
                out.push(path);
            }
        }
    }
    walk(dir, &mut out);
    out
}

pub fn tree_hashes(dir: &Path) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    fn walk(dir: &Path, root: &Path, rows: &mut Vec<(String, String)>) {
        let mut entries: Vec<_> = fs::read_dir(dir).unwrap().flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, rows);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if rel.ends_with("strata.lock") {
                    continue;
                }
                rows.push((rel, sha256_file(&path)));
            }
        }
    }
    if dir.exists() {
        walk(dir, dir, &mut rows);
    }
    rows
}

pub fn flip_byte(path: &Path, offset: u64) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xff;
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&byte).unwrap();
}

pub fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

pub fn seg_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("seg") {
                out.push(path);
            }
        }
    }
    walk(dir, &mut out);
    out.sort();
    out
}

pub fn refuse_sqlite_creation(blob: &str, what: &str) {
    if blob.contains("SQLite store creation is disabled")
        || blob.contains("cannot be opened by 4.0")
        || blob.contains("built without legacy-sqlite")
        || blob.contains("STRATA default lands")
    {
        missing(&format!(
            "{what} is not running on a strata store. The process refused SQLite \
             instead of opening a strata log. Wiring is the strata runtime boot \
             (after #300). Output: {}",
            blob.chars().take(900).collect::<String>()
        ));
    }
}

pub struct Server {
    child: Child,
    stdin: Option<std::process::ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
    pub data_dir: PathBuf,
}

impl Server {
    pub fn spawn(data_dir: &Path) -> Self {
        let mut command = Command::new(product_bin("vestige-mcp"));
        command
            .arg("--data-dir")
            .arg(data_dir)
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("HOME", data_dir)
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().expect("HARNESS: spawn vestige-mcp");
        let stdin = child.stdin.take().unwrap();
        let raw_stdout = child.stdout.take().unwrap();
        let raw_stderr = child.stderr.take().unwrap();
        let (tx, stdout) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(raw_stdout);
            for line in std::io::BufRead::lines(reader) {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            return;
                        }
                    }
                    Err(_) => return,
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(raw_stderr);
                for line in std::io::BufRead::lines(reader).map_while(Result::ok) {
                    if let Ok(mut sink) = sink.lock() {
                        sink.push(line);
                    }
                }
            });
        }
        Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
            data_dir: data_dir.to_path_buf(),
        }
    }

    pub fn stderr_text(&self) -> String {
        self.stderr
            .lock()
            .map(|lines| lines.join("\n"))
            .unwrap_or_default()
    }

    pub fn boot_strata(data_dir: &Path) -> Self {
        let mut server = Self::spawn(data_dir);
        // Storage init happens before the first read. Give a failing process
        // a moment to exit so the handshake does not burn the full RPC budget
        // on a binary that already refused to boot.
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            if let Ok(Some(status)) = server.child.try_wait() {
                let err = server.stderr_text();
                let artifacts = sqlite_artifacts(data_dir);
                if !artifacts.is_empty() {
                    missing(&format!(
                        "vestige-mcp created a SQLite file on an empty data dir ({artifacts:?}) \
                         and then exited {status}. 4.0 must not create SQLite. stderr: {err}"
                    ));
                }
                missing(&format!(
                    "strata MCP boot is not wired. vestige-mcp exited {status} on an empty \
                     data dir instead of serving a strata log over stdio. A fresh 4.0 install \
                     must stay up, create no SQLite file, and answer initialize. This lands \
                     with the strata runtime boot (after #300). stderr: {err}"
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
            // Still alive after a beat: try the handshake. If it dies during
            // handshake, read_line reports it as MISSING.
            if deadline.saturating_duration_since(Instant::now()) < Duration::from_secs(6) {
                break;
            }
        }
        if !sqlite_artifacts(data_dir).is_empty() {
            missing(&format!(
                "vestige-mcp created a SQLite file on an empty data dir: {:?}. \
                 The shipped 4.0 server must boot a strata log and never create SQLite.",
                sqlite_artifacts(data_dir)
            ));
        }
        match server.try_initialize() {
            Ok(_) => server,
            Err(err) => {
                let stderr = server.stderr_text();
                missing(&format!(
                    "strata MCP boot is not wired. initialize failed: {err}. stderr: {stderr}"
                ));
            }
        }
    }

    fn write_line(&mut self, line: &str) -> Result<(), String> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| "stdin closed".to_string())?;
        stdin
            .write_all(line.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("stdin write failed: {e}; stderr: {}", self.stderr_text()))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let timeout = Duration::from_secs(45);
        loop {
            match self.stdout.recv_timeout(timeout) {
                Ok(line) => {
                    if let Ok(value) = serde_json::from_str::<Value>(&line) {
                        if value.get("id").is_none() && value.get("method").is_some() {
                            continue;
                        }
                    }
                    return Ok(line);
                }
                Err(RecvTimeoutError::Timeout) => {
                    let status = self.child.try_wait().ok().flatten();
                    return Err(format!(
                        "no response in {timeout:?} (process {status:?}). stderr: {}",
                        self.stderr_text()
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(format!("stdout closed. stderr: {}", self.stderr_text()));
                }
            }
        }
    }

    fn request(&mut self, method: &str, params: Option<Value>) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string())?;
        let line = self.read_line()?;
        serde_json::from_str(&line).map_err(|e| format!("non-JSON response {line:?}: {e}"))
    }

    fn try_initialize(&mut self) -> Result<Value, String> {
        let response = self.request(
            "initialize",
            Some(json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "release-matrix", "version": "4.0.0" },
            })),
        )?;
        if response.get("error").is_some() {
            return Err(format!("initialize error: {response}"));
        }
        self.write_line(
            &json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })
            .to_string(),
        )?;
        Ok(response["result"].clone())
    }

    pub fn call(&mut self, method: &str, params: Option<Value>) -> Result<Value, String> {
        self.request(method, params)
    }

    pub fn tool(&mut self, name: &str, arguments: Value) -> Result<ToolReply, String> {
        let response = self.request(
            "tools/call",
            Some(json!({ "name": name, "arguments": arguments })),
        )?;
        if let Some(error) = response.get("error") {
            return Ok(ToolReply {
                protocol_error: Some(error.clone()),
                raw: Value::Null,
                body: error.clone(),
            });
        }
        let raw = response["result"].clone();
        let body = if let Some(structured) = raw.get("structuredContent") {
            structured.clone()
        } else if let Some(text) = raw["content"][0]["text"].as_str() {
            serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
        } else {
            raw.clone()
        };
        Ok(ToolReply {
            protocol_error: None,
            raw,
            body,
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ToolReply {
    pub protocol_error: Option<Value>,
    pub raw: Value,
    pub body: Value,
}

impl ToolReply {
    pub fn rejected(&self) -> bool {
        self.protocol_error.is_some()
            || self.raw.get("isError") == Some(&json!(true))
            || self.body.get("error").is_some()
            || self.body.get("isError") == Some(&json!(true))
    }

    pub fn handle_required(&self) -> bool {
        self.body.get("error").and_then(|v| v.as_str()) == Some("handle_required")
            || self.raw.to_string().contains("\"handle_required\"")
            || self.body.to_string().contains("\"handle_required\"")
    }

    pub fn blob(&self) -> String {
        format!("{} {}", self.raw, self.body)
    }
}

pub fn assert_handle_required(tool: &str, reply: &ToolReply) {
    if !reply.handle_required() {
        missing(&format!(
            "`{tool}` exact-handle miss must return error \"handle_required\" \
             (no keyword, BM25, name, or cosine fallback). Got: {}",
            reply.blob().chars().take(800).collect::<String>()
        ));
    }
}

pub fn has_gate_receipt(reply: &ToolReply) -> bool {
    fn walk(v: &Value) -> bool {
        match v {
            Value::Object(map) => {
                if map.contains_key("receipt")
                    || map.contains_key("receipts")
                    || map.contains_key("verdict")
                    || map.contains_key("admission")
                    || map.contains_key("admission_receipt")
                {
                    return true;
                }
                map.values().any(walk)
            }
            Value::Array(items) => items.iter().any(walk),
            _ => false,
        }
    }
    walk(&reply.body) || walk(&reply.raw)
}

pub fn assert_receipt(tool: &str, reply: &ToolReply) {
    if !has_gate_receipt(reply) {
        missing(&format!(
            "`{tool}` write/destructive result has no gate verdict or receipt. \
             Every write and destructive action must carry one. Got: {}",
            reply.blob().chars().take(800).collect::<String>()
        ));
    }
}

/// Boot a strata server or fail naming the missing runtime, then run `body`.
pub fn with_server(body: impl FnOnce(&mut Server)) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut server = Server::boot_strata(dir.path());
    body(&mut server);
}

pub fn sqlite_text(db: &Path, sql: &str) -> Vec<Vec<String>> {
    let conn = rusqlite::Connection::open_with_flags(
        db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .unwrap_or_else(|e| panic!("open {}: {e}", db.display()));
    let mut stmt = conn.prepare(sql).unwrap();
    let cols = stmt.column_count();
    let mut rows = stmt.query([]).unwrap();
    let mut out = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        let mut cells = Vec::new();
        for i in 0..cols {
            let cell = row
                .get_ref(i)
                .unwrap()
                .as_str()
                .map(|s| s.to_string())
                .or_else(|_| row.get::<_, i64>(i).map(|n| n.to_string()))
                .or_else(|_| row.get::<_, f64>(i).map(|n| n.to_string()))
                .unwrap_or_default();
            cells.push(cell);
        }
        out.push(cells);
    }
    out
}

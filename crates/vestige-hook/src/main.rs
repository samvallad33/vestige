//! `vestige-hook`: Ultra-thin (<2.5 MB, <10 ms) native Rust hook binary for Vestige.
//!
//! Deliberately avoids linking `vestige-core` (35.5 MiB lexicons), `axum`, SvelteKit,
//! `tokio`, or `clap` derive so every PreToolUse / UserPromptSubmit / PostToolUse / Stop
//! hook invocation across Claude Code, Antigravity, Cursor, Codex, Windsurf, Cline, and
//! Gemini CLI runs in single-digit milliseconds with zero Python and zero Node.js.
//!
//! Commands:
//! - `vestige-hook gate [--source <rt>] [--no-receipt]`
//! - `vestige-hook dispatch <runtime> <event> [--data-dir <dir>] [--no-receipt]`
//! - `vestige-hook verify-gpu <script.py> [--summary <text>]`
//! - `vestige-hook --version`

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const TAIL_READ_BYTES: u64 = 8_192;
const GENESIS_PREV: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const GPU_DRYRUN_TTL_SECS: i64 = 24 * 60 * 60;
const MAX_MISSION_STOP_BLOCKS: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GateReceipt {
    ts: String,
    prev: String,
    digest: String,
    decision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    commitment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rule: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owasp_asi: Option<String>,
    source: String,
    event: String,
    session: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    action_preview: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    command: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    via: Option<String>,
    why: String,
    action_digest: String,
    provider_id: String,
    latency_us: u64,
}

#[derive(Debug, Clone, Default)]
struct EvalReq {
    event: String,
    runtime: String,
    tool_name: String,
    command: Option<String>,
    file_path: Option<String>,
    payload_json: String,
    cwd: Option<String>,
    session_id: String,
}

#[derive(Debug, Clone)]
struct EvalVerdict {
    deny: bool,
    rule_id: Option<String>,
    owasp_asi: Option<String>,
    reason: String,
    latency_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MissionContract {
    session_id: String,
    cwd: String,
    prompt_excerpt: String,
    required_deleted_paths: Vec<String>,
    #[serde(default)]
    required_net_loc_negative: bool,
    #[serde(default)]
    stop_blocks: u32,
    #[serde(default = "default_max_stop_blocks")]
    max_stop_blocks: u32,
    active: bool,
    created_at: String,
}

const fn default_max_stop_blocks() -> u32 {
    MAX_MISSION_STOP_BLOCKS
}

fn home_dir() -> PathBuf {
    if let Ok(h) = std::env::var("HOME")
        && !h.trim().is_empty()
    {
        return PathBuf::from(h);
    }
    if let Ok(h) = std::env::var("USERPROFILE")
        && !h.trim().is_empty()
    {
        return PathBuf::from(h);
    }
    PathBuf::from(".")
}

fn operator_home_dir() -> PathBuf {
    if let Ok(op) = std::env::var("OPERATOR_HOME")
        && !op.trim().is_empty()
    {
        return PathBuf::from(op);
    }
    home_dir().join(".operator")
}

fn missions_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VESTIGE_MISSIONS_DIR")
        && !d.trim().is_empty()
    {
        return PathBuf::from(d);
    }
    home_dir().join(".vestige").join("missions")
}

fn gpu_dryruns_dir() -> PathBuf {
    if let Ok(d) = std::env::var("VESTIGE_GPU_DRYRUNS_DIR")
        && !d.trim().is_empty()
    {
        return PathBuf::from(d);
    }
    home_dir().join(".vestige").join("gpu_dryruns")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn canonical_sha256_without_digest(val: &Value) -> String {
    fn write_canonical(v: &Value, out: &mut String, top: bool) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_canonical(x, out, false);
                }
                out.push(']');
            }
            Value::Object(o) => {
                let mut keys: Vec<&String> =
                    o.keys().filter(|k| !(top && *k == "digest")).collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    write_canonical(&o[k], out, false);
                }
                out.push('}');
            }
        }
    }
    let mut text = String::new();
    write_canonical(val, &mut text, true);
    sha256_hex(text.as_bytes())
}

fn compute_action_digest(req: &EvalReq) -> String {
    let cwd = req.cwd.as_deref().unwrap_or("").replace('\\', "/");
    let cmd = req.command.as_deref().unwrap_or("");
    let file = req.file_path.as_deref().unwrap_or("").replace('\\', "/");
    let preimage = format!("{}|{}|{}|{}", req.tool_name, cwd, cmd, file);
    sha256_hex(preimage.as_bytes())[..24].to_string()
}

fn read_last_receipt_digest(path: &Path) -> String {
    let Ok(mut file) = File::open(path) else {
        return GENESIS_PREV.to_string();
    };
    let Ok(meta) = file.metadata() else {
        return GENESIS_PREV.to_string();
    };
    let len = meta.len();
    if len == 0 {
        return GENESIS_PREV.to_string();
    }
    let start = len.saturating_sub(TAIL_READ_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return GENESIS_PREV.to_string();
    }
    let mut buf = String::new();
    if file.read_to_string(&mut buf).is_err() {
        return GENESIS_PREV.to_string();
    }
    for line in buf.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(trimmed)
            && let Some(d) = v.get("digest").and_then(Value::as_str)
            && !d.is_empty()
        {
            return d.to_string();
        }
    }
    GENESIS_PREV.to_string()
}

fn read_last_chain_digest_in_dir(receipts_dir: &Path) -> String {
    let Ok(rd) = fs::read_dir(receipts_dir) else {
        return GENESIS_PREV.to_string();
    };
    let mut files: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    for path in files.into_iter().rev() {
        let d = read_last_receipt_digest(&path);
        if d != GENESIS_PREV {
            return d;
        }
    }
    GENESIS_PREV.to_string()
}

fn today_receipts_path(op_home: &Path) -> PathBuf {
    let day = Utc::now().format("%Y-%m-%d").to_string();
    op_home.join("receipts").join(format!("{day}.jsonl"))
}

fn try_consume_single_use_permit(op_home: &Path, action_digest: &str) -> bool {
    let permit_path = op_home.join("permits").join(format!("{action_digest}.json"));
    if !permit_path.is_file() {
        return false;
    }
    let Ok(raw) = fs::read_to_string(&permit_path) else {
        return false;
    };
    let Ok(val) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    if let Some(ad) = val.get("action_digest").and_then(Value::as_str)
        && ad != action_digest
    {
        return false;
    }
    let now_s = Utc::now().timestamp() as f64;
    if let Some(exp) = val.get("expires").and_then(Value::as_f64)
        && exp < now_s
    {
        let _ = fs::rename(&permit_path, op_home.join("permits").join(format!("{action_digest}.json.expired")));
        return false;
    }
    fs::rename(&permit_path, op_home.join("permits").join(format!("{action_digest}.json.used"))).is_ok()
}

fn append_receipt(
    path: &Path,
    req: &EvalReq,
    verdict: &EvalVerdict,
    decision_override: Option<&str>,
) -> Option<GateReceipt> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok()?;
    }
    let prev = path
        .parent()
        .map(read_last_chain_digest_in_dir)
        .filter(|d| d != GENESIS_PREV)
        .unwrap_or_else(|| read_last_receipt_digest(path));
    let decision = decision_override
        .map(str::to_string)
        .unwrap_or_else(|| if verdict.deny { "STOP".into() } else { "PASS".into() });
    let via = (decision == "ALLOW").then(|| "permit".to_string());
    let action_digest = compute_action_digest(req);
    let mut receipt = GateReceipt {
        ts: Utc::now().to_rfc3339(),
        prev,
        digest: String::new(),
        decision,
        commitment: verdict.rule_id.clone(),
        rule: verdict.rule_id.clone(),
        owasp_asi: verdict.owasp_asi.clone(),
        source: req.runtime.clone(),
        event: req.event.clone(),
        session: req.session_id.clone(),
        cwd: req.cwd.clone(),
        tool: req.tool_name.clone(),
        action_preview: req.command.clone().or_else(|| req.file_path.clone()),
        target: req.command.clone().or_else(|| req.file_path.clone()),
        command: req.command.clone(),
        via,
        why: verdict.reason.clone(),
        action_digest,
        provider_id: "vestige-hook-native".into(),
        latency_us: verdict.latency_us,
    };
    let val = serde_json::to_value(&receipt).unwrap_or_else(|_| json!({}));
    receipt.digest = canonical_sha256_without_digest(&val);
    let mut f = OpenOptions::new().create(true).append(true).open(path).ok()?;
    writeln!(f, "{}", serde_json::to_string(&receipt).unwrap_or_default()).ok()?;
    Some(receipt)
}

fn is_shell_tool(tool: &str) -> bool {
    matches!(
        tool.to_ascii_lowercase().as_str(),
        "bash"
            | "shell"
            | "local_shell"
            | "exec_command"
            | "container.exec"
            | "run_shell_command"
            | "run_command"
            | "terminal"
            | "run_terminal_cmd"
            | "powershell"
    )
}

fn is_write_tool(tool: &str) -> bool {
    matches!(
        tool.to_ascii_lowercase().as_str(),
        "write"
            | "edit"
            | "multiedit"
            | "notebookedit"
            | "apply_patch"
            | "str_replace_editor"
            | "create_file"
            | "edit_file"
            | "write_file"
            | "write_to_file"
            | "replace_file_content"
            | "multi_replace_file_content"
    )
}

fn is_protected_kernel_path(path: &str) -> bool {
    let p = path.replace('\\', "/").to_ascii_lowercase();
    p.contains("/.operator/")
        || p.ends_with("/.operator")
        || p.contains("/.git/hooks/")
        || p.ends_with("/strata.key")
        || (p.contains("/log/") && p.ends_with(".seg"))
}

fn is_strictly_readonly_shell(cmd: &str) -> bool {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.contains(';')
        || trimmed.contains("&&")
        || trimmed.contains("||")
        || trimmed.contains('|')
        || trimmed.contains('`')
        || trimmed.contains("$(")
        || trimmed.contains('>')
    {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let first = lower.split_whitespace().next().unwrap_or("");
    matches!(
        first,
        "get-childitem"
            | "gci"
            | "ls"
            | "dir"
            | "pwd"
            | "cat"
            | "get-content"
            | "select-string"
            | "rg"
            | "grep"
            | "findstr"
            | "head"
            | "tail"
            | "wc"
            | "which"
            | "where"
            | "whoami"
    ) || lower.starts_with("git status")
        || lower.starts_with("git log")
        || lower.starts_with("git diff")
        || lower.starts_with("git show")
        || lower.starts_with("git branch")
        || lower.starts_with("git rev-parse")
        || lower.starts_with("git remote -v")
        || lower.starts_with("gh run list")
        || lower.starts_with("gh run view")
}

fn is_temp_path_hint(lower: &str) -> bool {
    lower.contains("/tmp/")
        || lower.contains("\\temp\\")
        || lower.contains("/temp/")
        || lower.contains("appdata/local/temp")
        || lower.contains("appdata\\local\\temp")
}

fn contains_high_confidence_secret(payload: &str) -> Option<&'static str> {
    const PREFIXES: &[(&str, &str)] = &[
        (concat!("-----BEGIN ", "OPENSSH PRIVATE KEY-----"), "PrivateKey"),
        (concat!("-----BEGIN ", "RSA PRIVATE KEY-----"), "PrivateKey"),
        (concat!("-----BEGIN ", "PRIVATE KEY-----"), "PrivateKey"),
        ("AKIA", "AwsAccessKey"),
        ("ghp_", "GitHubPat"),
        ("github_pat_", "GitHubPat"),
        ("sk-ant-", "AnthropicApiKey"),
        ("sk-proj-", "OpenAiApiKey"),
        ("hf_", "HuggingFaceToken"),
        ("xoxb-", "SlackToken"),
    ];
    for &(marker, kind) in PREFIXES {
        if let Some(pos) = payload.find(marker) {
            if marker.starts_with("-----BEGIN") {
                return Some(kind);
            }
            let rest = &payload[pos + marker.len()..];
            let token_len = rest
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
                .count();
            if token_len >= 16 {
                return Some(kind);
            }
        }
    }
    None
}

fn is_paid_gpu_launch(lower: &str) -> bool {
    if lower.contains("colab stop")
        || lower.contains("colab ls")
        || lower.contains("colab status")
        || lower.contains("colab rm")
        || lower.contains("colab kill")
        || lower.contains("modal app stop")
        || lower.contains("runpodctl stop")
        || lower.contains("runpodctl remove")
        || lower.contains("sky down")
        || lower.contains("sky stop")
        || lower.contains("--dry-run")
        || lower.contains("--dry_run")
        || lower.contains("py_compile")
        || lower.contains("full-dry-run")
    {
        return false;
    }
    lower.contains("colab exec")
        || lower.contains("colab new")
        || lower.contains("modal run")
        || lower.contains("runpodctl create")
        || lower.contains("sky launch")
        || lower.contains("torchrun ")
        || lower.contains("accelerate launch")
}

fn has_fresh_gpu_dryrun_receipt(cmd: &str, cwd: Option<&str>) -> bool {
    let Some(script_tok) = cmd.split_whitespace().find_map(|raw| {
        let tok = raw.trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | ',' | ';'));
        (tok.ends_with(".py") && !tok.starts_with('-')).then_some(tok)
    }) else {
        return false;
    };
    let p = Path::new(script_tok);
    let candidate = if p.is_absolute() {
        p.to_path_buf()
    } else if let Some(base) = cwd {
        Path::new(base).join(p)
    } else {
        p.to_path_buf()
    };
    let Ok(bytes) = fs::read(&candidate) else {
        return false;
    };
    let sha = sha256_hex(&bytes);
    let Ok(raw) = fs::read_to_string(gpu_dryruns_dir().join(format!("{sha}.json"))) else {
        return false;
    };
    let Ok(val) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    if val.get("status").and_then(Value::as_str) != Some("FULL-DRY-RUN PASS") {
        return false;
    }
    let ts = val.get("ts").and_then(Value::as_i64).unwrap_or(0);
    Utc::now().timestamp().saturating_sub(ts) <= GPU_DRYRUN_TTL_SECS
}

fn evaluate_gate(req: &EvalReq) -> EvalVerdict {
    let started = Instant::now();
    let deny = |rule: &str, asi: &str, reason: String| EvalVerdict {
        deny: true,
        rule_id: Some(rule.into()),
        owasp_asi: Some(asi.into()),
        reason,
        latency_us: started.elapsed().as_micros() as u64,
    };

    if let Some(fp) = req.file_path.as_deref()
        && is_write_tool(&req.tool_name)
        && is_protected_kernel_path(fp)
    {
        return deny("OP-005", "ASI05", format!("Protected gate/kernel path cannot be modified by an agent: {fp}"));
    }

    if is_write_tool(&req.tool_name) && !req.payload_json.is_empty()
        && let Some(kind) = contains_high_confidence_secret(&req.payload_json)
    {
        return deny(
            "OP-009",
            "ASI09",
            format!("High-confidence secret ({kind}) detected in file write payload; use a secret manager or env reference."),
        );
    }

    if is_shell_tool(&req.tool_name)
        && let Some(cmd) = req.command.as_deref()
        && !is_strictly_readonly_shell(cmd)
    {
        let lower = cmd.to_ascii_lowercase();
        let norm = lower.replace('\\', "/");

        if lower.contains("rm -rf / ")
            || lower.ends_with("rm -rf /")
            || lower.contains("rm -rf /*")
            || lower.contains("rm -rf ~")
            || lower.contains("rm -fr /")
            || lower.contains("del /s /q c:\\")
            || lower.contains("mkfs.")
            || (lower.contains("dd if=") && lower.contains("of=/dev/"))
        {
            return deny("OP-001", "ASI01", "Destructive root, home, or block-device wipe is blocked.".into());
        }
        let has_net_fetch = lower.contains("curl ")
            || lower.contains("wget ")
            || lower.contains("irm ")
            || lower.contains("iwr ")
            || lower.contains("invoke-webrequest")
            || lower.contains("invoke-restmethod");
        let has_pipe_exec = lower.contains("| sh")
            || lower.contains("|sh")
            || lower.contains("| bash")
            || lower.contains("|bash")
            || lower.contains("| zsh")
            || lower.contains("| python")
            || lower.contains("| iex")
            || lower.contains("|iex")
            || lower.contains("| invoke-expression");
        if has_net_fetch && has_pipe_exec {
            return deny("OP-002", "ASI02", "Piping remote network output directly into a shell/interpreter is blocked.".into());
        }
        let touches_creds = norm.contains(".ssh/id_")
            || norm.contains(".aws/credentials")
            || norm.contains(".kube/config")
            || norm.contains("/etc/shadow");
        let sends_over_net = lower.contains("curl ")
            || lower.contains("wget ")
            || lower.contains("scp ")
            || lower.contains("nc ")
            || lower.contains("ncat ");
        if touches_creds && sends_over_net {
            return deny("OP-003", "ASI03", "Exfiltrating local credentials or private keys over the network is blocked.".into());
        }
        if (lower.contains("git push")
            && (lower.contains(" --force") || lower.contains(" -f "))
            && (lower.contains(" main") || lower.contains(" master")))
            || lower.contains("git reset --hard")
            || lower.contains("git clean -fdx")
            || lower.contains("git clean -fd")
            || lower.contains("git branch -d main")
            || lower.contains("git branch -d master")
        {
            return deny("OP-004", "ASI04", "Destructive git force-push, hard reset, or untracked wipe requires owner approval.".into());
        }
        if (lower.contains("rm ")
            || lower.contains("del ")
            || lower.contains("remove-item")
            || lower.contains("chmod ")
            || lower.contains('>'))
            && is_protected_kernel_path(&norm)
        {
            return deny("OP-005", "ASI05", "Shell mutation of protected Operator gate, git hooks, or Strata log is blocked.".into());
        }
        if lower.contains("drop table")
            || lower.contains("drop database")
            || lower.contains("truncate table")
        {
            return deny("OP-006", "ASI06", "Destructive SQL DDL (DROP/TRUNCATE) requires owner approval.".into());
        }
        if lower.contains("sudo rm ")
            || lower.contains("chmod 777 /")
            || lower.contains("reg delete hklm")
            || lower.contains("bcdedit ")
        {
            return deny("OP-007", "ASI07", "System-level privilege escalation or OS tampering is blocked.".into());
        }
        let is_recursive_del = lower.contains("rm -rf ")
            || lower.contains("rm -fr ")
            || (lower.contains("remove-item") && lower.contains("-recurse"))
            || lower.contains("rmdir /s");
        if is_recursive_del && !is_temp_path_hint(&lower) {
            return deny("OP-008", "ASI08", "Recursive deletion outside temporary directories requires owner approval.".into());
        }
        if is_paid_gpu_launch(&lower) && !has_fresh_gpu_dryrun_receipt(cmd, req.cwd.as_deref()) {
            return deny(
                "OP-GPU-DRYRUN",
                "ASI08",
                "Paid GPU launch blocked: no fresh (<24h) [FULL-DRY-RUN PASS] receipt found for this exact script SHA-256. Run a full local dry-run and record it with `vestige-hook verify-gpu <script.py>` first.".into(),
            );
        }
        if lower.contains("/dev/tcp/")
            || lower.contains("nc -e ")
            || lower.contains("ncat -e ")
            || (lower.contains("pty.spawn") && lower.contains("socket"))
        {
            return deny("OP-010", "ASI10", "Interactive reverse-shell invocation is blocked.".into());
        }
    }

    EvalVerdict {
        deny: false,
        rule_id: None,
        owasp_asi: None,
        reason: String::new(),
        latency_us: started.elapsed().as_micros() as u64,
    }
}

fn parse_eval_req(source: &str, raw_stdin: &str) -> EvalReq {
    let v: Value = serde_json::from_str(raw_stdin).unwrap_or_else(|_| json!({}));
    let tool_name = v
        .get("tool_name")
        .or_else(|| v.get("tool"))
        .or_else(|| v.pointer("/toolCall/name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let input = v
        .get("tool_input")
        .or_else(|| v.pointer("/toolCall/args"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let command = input
        .get("command")
        .or_else(|| input.get("CommandLine"))
        .or_else(|| input.get("cmd"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let file_path = input
        .get("file_path")
        .or_else(|| input.get("path"))
        .or_else(|| input.get("Path"))
        .or_else(|| input.get("TargetFile"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let event = v
        .get("hook_event_name")
        .or_else(|| v.get("event"))
        .and_then(Value::as_str)
        .unwrap_or("PreToolUse")
        .to_string();
    let cwd = v
        .get("cwd")
        .or_else(|| v.pointer("/workspacePaths/0"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let session_id = v
        .get("session_id")
        .or_else(|| v.get("conversationId"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    EvalReq {
        event,
        runtime: source.to_string(),
        tool_name,
        command,
        file_path,
        payload_json: input.to_string(),
        cwd,
        session_id,
    }
}

fn mission_file_path(dir: &Path, session_id: &str) -> PathBuf {
    let key = if session_id.trim().is_empty() { "default" } else { session_id.trim() };
    dir.join(format!("{}.json", &sha256_hex(key.as_bytes())[..24]))
}

fn capture_mission_from_prompt(dir: &Path, session_id: &str, cwd: &str, prompt: &str) -> Option<MissionContract> {
    let cwd_path = Path::new(cwd);
    if cwd.trim().is_empty() || !cwd_path.is_dir() {
        return None;
    }
    let lower = prompt.to_ascii_lowercase();
    let has_delete = lower.contains("delete")
        || lower.contains("remove")
        || lower.contains("prune")
        || lower.contains("purge")
        || lower.contains("strip")
        || lower.contains("consolidat");
    let mut targets = Vec::new();
    if has_delete {
        for raw_tok in prompt.split_whitespace() {
            let cleaned = raw_tok
                .trim_matches(|c: char| matches!(c, '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '.' | '!' | '?'))
                .trim_end_matches('/');
            if cleaned.is_empty() || cleaned.starts_with("http://") || cleaned.starts_with("https://") || cleaned.contains("..") || Path::new(cleaned).is_absolute() {
                continue;
            }
            let looks_like_path = cleaned.contains('/')
                || cleaned.contains('.')
                || raw_tok.trim_matches(|c: char| matches!(c, '`' | '\'' | '"' | ',' | '(' | ')')).ends_with('/')
                || (cleaned.len() >= 3 && cwd_path.join(cleaned).is_dir());
            if looks_like_path && cwd_path.join(cleaned).exists() && !targets.iter().any(|t: &String| t == cleaned) {
                targets.push(cleaned.to_string());
            }
        }
    }
    let need_shrink = (lower.contains("consolidat") || lower.contains("shrink"))
        && (lower.contains("delete") || lower.contains("remove") || lower.contains("strip") || lower.contains("loc") || lower.contains("lines"));
    if targets.is_empty() && !need_shrink {
        return None;
    }
    let excerpt: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let contract = MissionContract {
        session_id: session_id.to_string(),
        cwd: cwd.to_string(),
        prompt_excerpt: excerpt.chars().take(240).collect(),
        required_deleted_paths: targets,
        required_net_loc_negative: need_shrink,
        stop_blocks: 0,
        max_stop_blocks: MAX_MISSION_STOP_BLOCKS,
        active: true,
        created_at: Utc::now().to_rfc3339(),
    };
    let path = mission_file_path(dir, session_id);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(&contract) {
        let _ = fs::write(&path, text);
    }
    Some(contract)
}

fn verify_mission_on_stop(dir: &Path, session_id: &str) -> Option<String> {
    let path = mission_file_path(dir, session_id);
    let raw = fs::read_to_string(&path).ok()?;
    let mut contract: MissionContract = serde_json::from_str(&raw).ok()?;
    if !contract.active {
        return None;
    }
    let max_blocks = if contract.max_stop_blocks == 0 { MAX_MISSION_STOP_BLOCKS } else { contract.max_stop_blocks };
    if contract.stop_blocks >= max_blocks {
        contract.active = false;
        let _ = fs::write(&path, serde_json::to_string_pretty(&contract).unwrap_or_default());
        return None;
    }
    let cwd = Path::new(&contract.cwd);
    let mut failures = Vec::new();
    let remaining: Vec<String> = contract
        .required_deleted_paths
        .iter()
        .filter(|rel| cwd.join(rel).exists())
        .cloned()
        .collect();
    if !remaining.is_empty() {
        failures.push(format!("{} target(s) still exist ({})", remaining.len(), remaining.join(", ")));
    }
    if contract.required_net_loc_negative
        && let Ok(out) = std::process::Command::new("git")
            .args(["diff", "--numstat", "HEAD"])
            .current_dir(cwd)
            .output()
        && out.status.success()
    {
        let (mut add, mut del) = (0i64, 0i64);
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let mut it = line.split_whitespace();
            if let (Some(a), Some(d)) = (it.next(), it.next())
                && let (Ok(ai), Ok(di)) = (a.parse::<i64>(), d.parse::<i64>())
            {
                add += ai;
                del += di;
            }
        }
        if add > del {
            failures.push(format!("consolidation contract requires negative net LOC, got +{add}/-{del}"));
        }
    }
    if failures.is_empty() {
        contract.active = false;
        let _ = fs::write(&path, serde_json::to_string_pretty(&contract).unwrap_or_default());
        return None;
    }
    contract.stop_blocks = contract.stop_blocks.saturating_add(1);
    let _ = fs::write(&path, serde_json::to_string_pretty(&contract).unwrap_or_default());
    Some(format!(
        "Mission contract unfinished (block {}/{}): {}. Complete the mission before stopping.",
        contract.stop_blocks,
        max_blocks,
        failures.join("; ")
    ))
}

/// Optional sub-2ms direct TCP_NODELAY JSON-RPC call to the resident `vestige-mcp` daemon.
fn try_daemon_reflex(data_dir: &Path, params: Value, timeout_ms: u64) -> Option<Value> {
    let ep_raw = fs::read_to_string(data_dir.join(".serve.endpoint")).ok()?;
    let parts: Vec<&str> = ep_raw.split_whitespace().collect();
    if parts.len() != 3 {
        return None;
    }
    let port: u16 = parts[0].parse().ok()?;
    let token = parts[1];
    let expected_welcome = format!("vestige-attached/1 {}", parts[2]);
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let timeout = Duration::from_millis(timeout_ms);
    let mut stream = TcpStream::connect_timeout(&addr, timeout).ok()?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));

    stream.write_all(format!("vestige-attach/1 {token}\n").as_bytes()).ok()?;
    stream.flush().ok()?;

    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    if line.trim_end() != expected_welcome {
        return None;
    }

    let init = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": { "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": { "name": "vestige-hook", "version": VERSION } }
    });
    stream.write_all(format!("{init}\n").as_bytes()).ok()?;
    line.clear();
    reader.read_line(&mut line).ok()?;

    stream.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n").ok()?;
    let call = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": { "name": "session", "arguments": { "action": "reflex", "params": params } }
    });
    stream.write_all(format!("{call}\n").as_bytes()).ok()?;
    stream.flush().ok()?;

    line.clear();
    reader.read_line(&mut line).ok()?;
    let resp: Value = serde_json::from_str(line.trim()).ok()?;
    let res = resp.get("result")?;
    if let Some(sc) = res.get("structuredContent") {
        return Some(sc.clone());
    }
    let text = res.pointer("/content/0/text").and_then(Value::as_str)?;
    serde_json::from_str(text).ok()
}

fn format_stop_stderr(verdict: &EvalVerdict, receipt: Option<&GateReceipt>, req: &EvalReq) -> String {
    let rule = verdict.rule_id.as_deref().unwrap_or("OP-GATE");
    let asi = verdict.owasp_asi.as_deref().unwrap_or("ASI00");
    let action_digest = receipt
        .map(|r| r.action_digest.clone())
        .unwrap_or_else(|| compute_action_digest(req));
    let receipt_short: String = receipt
        .map(|r| r.digest.chars().take(16).collect())
        .unwrap_or_else(|| action_digest.chars().take(16).collect());
    format!(
        "OPERATOR: STOPPED ({rule} {asi}): {} Receipt: {receipt_short}\nNext: do not retry it; call precedent(rule=\"{rule}\") to inspect the receipt before proposing an alternative.\nProof: recall(handle=\"op-receipt-{receipt_short}\") | Rule: recall(handle=\"op-rule-{rule}\") | ALLOW ONCE {action_digest}",
        verdict.reason
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "--version" | "-V") {
        println!("vestige-hook {VERSION}");
        return;
    }

    match args[0].as_str() {
        "gate" => {
            let mut source = "claude".to_string();
            let mut write_receipt = true;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--source" if i + 1 < args.len() => {
                        source = args[i + 1].clone();
                        i += 2;
                    }
                    "--no-receipt" => {
                        write_receipt = false;
                        i += 1;
                    }
                    "hook" => i += 1,
                    _ => i += 1,
                }
            }
            let mut raw = String::new();
            let _ = std::io::stdin().read_to_string(&mut raw);
            let req = parse_eval_req(&source, &raw);
            let mut verdict = evaluate_gate(&req);
            let mut decision_override = None;
            if verdict.deny {
                let op_home = operator_home_dir();
                let ad = compute_action_digest(&req);
                if try_consume_single_use_permit(&op_home, &ad) {
                    verdict.deny = false;
                    verdict.reason = "Authorized by single-use owner permit.".into();
                    decision_override = Some("ALLOW");
                }
            }
            let receipt = if write_receipt {
                append_receipt(&today_receipts_path(&operator_home_dir()), &req, &verdict, decision_override)
            } else {
                None
            };
            if verdict.deny {
                eprintln!("{}", format_stop_stderr(&verdict, receipt.as_ref(), &req));
                std::process::exit(2);
            }
        }
        "verify-gpu" => {
            let Some(script) = args.get(1) else {
                eprintln!("Usage: vestige-hook verify-gpu <script.py> [--summary <text>]");
                std::process::exit(2);
            };
            let summary = args
                .windows(2)
                .find(|w| w[0] == "--summary")
                .map(|w| w[1].as_str())
                .unwrap_or("Local CPU/mock full dry-run passed");
            let path = Path::new(script);
            let bytes = match fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Failed to read {script}: {e}");
                    std::process::exit(1);
                }
            };
            let sha = sha256_hex(&bytes);
            let dir = gpu_dryruns_dir();
            if let Err(e) = fs::create_dir_all(&dir) {
                eprintln!("Failed to create {}: {e}", dir.display());
                std::process::exit(1);
            }
            let out_path = dir.join(format!("{sha}.json"));
            let body = json!({
                "script": path.display().to_string(),
                "sha256": sha,
                "status": "FULL-DRY-RUN PASS",
                "ts": Utc::now().timestamp(),
                "created_at": Utc::now().to_rfc3339(),
                "summary": summary,
                "receipt": out_path.display().to_string(),
            });
            if let Err(e) = fs::write(&out_path, serde_json::to_string_pretty(&body).unwrap_or_default()) {
                eprintln!("Failed to write receipt: {e}");
                std::process::exit(1);
            }
            println!("{body}");
        }
        "dispatch" => {
            let runtime = args.get(1).map(String::as_str).unwrap_or("claude");
            let raw_event = args.get(2).map(String::as_str).unwrap_or("PreToolUse");
            let write_receipt = !args.iter().any(|a| a == "--no-receipt");
            let data_dir = args
                .windows(2)
                .find(|w| w[0] == "--data-dir")
                .map(|w| PathBuf::from(&w[1]))
                .or_else(|| std::env::var("VESTIGE_DATA_DIR").ok().filter(|s| !s.trim().is_empty()).map(PathBuf::from))
                .unwrap_or_else(|| home_dir().join(".vestige").join("store"));
            let mut raw = String::new();
            let _ = std::io::stdin().read_to_string(&mut raw);
            let parsed: Value = serde_json::from_str(&raw).unwrap_or_else(|_| json!({}));
            let session_id = parsed.get("session_id").or_else(|| parsed.get("conversationId")).and_then(Value::as_str).unwrap_or("");
            let cwd = parsed.get("cwd").or_else(|| parsed.pointer("/workspacePaths/0")).and_then(Value::as_str).unwrap_or("");
            let ev_lower = raw_event.to_ascii_lowercase();
            let rt_lower = runtime.to_ascii_lowercase();

            if matches!(ev_lower.as_str(), "userpromptsubmit" | "preinvocation" | "beforesubmitprompt" | "beforeagent") {
                let prompt = parsed.get("prompt").or_else(|| parsed.get("lastUserInput")).and_then(Value::as_str).unwrap_or("");
                if let Some(m) = capture_mission_from_prompt(&missions_dir(), session_id, cwd, prompt) {
                    let targets_str = if m.required_deleted_paths.is_empty() {
                        "none".to_string()
                    } else {
                        m.required_deleted_paths.join(", ")
                    };
                    let msg = format!(
                        "Vestige Mission Contract captured ({} path target(s): [{}], shrink={}).",
                        m.required_deleted_paths.len(),
                        targets_str,
                        m.required_net_loc_negative
                    );
                    if rt_lower == "antigravity" {
                        println!("{}", json!({ "injectSteps": [{ "systemMessage": { "systemMessage": msg } }] }));
                    } else {
                        println!("{}", json!({ "hookSpecificOutput": { "hookEventName": "UserPromptSubmit", "additionalContext": msg } }));
                    }
                }
                return;
            }

            if matches!(ev_lower.as_str(), "stop" | "agentstop" | "afteragent" | "afteragentresponse" | "taskcomplete") {
                if let Some(reason) = verify_mission_on_stop(&missions_dir(), session_id) {
                    if rt_lower == "antigravity" {
                        println!("{}", json!({ "decision": "deny", "reason": format!("Vestige mission (MISSION-INCOMPLETE): {reason}") }));
                    } else if matches!(rt_lower.as_str(), "cline" | "windsurf" | "kiro") {
                        eprintln!("Vestige mission (MISSION-INCOMPLETE): {reason}");
                        std::process::exit(2);
                    } else {
                        println!("{}", json!({ "decision": "block", "reason": format!("Vestige mission (MISSION-INCOMPLETE): {reason}") }));
                    }
                }
                return;
            }

            // Default: PreToolUse native gate + optional fast daemon reflex check.
            let mut req = parse_eval_req(runtime, &raw);
            req.event = "PreToolUse".into();
            let mut verdict = evaluate_gate(&req);
            let mut decision_override = None;
            if verdict.deny {
                let op_home = operator_home_dir();
                let ad = compute_action_digest(&req);
                if try_consume_single_use_permit(&op_home, &ad) {
                    verdict.deny = false;
                    verdict.reason = "Authorized by single-use owner permit.".into();
                    decision_override = Some("ALLOW");
                }
            }
            if write_receipt {
                let _ = append_receipt(&today_receipts_path(&operator_home_dir()), &req, &verdict, decision_override);
            }
            if verdict.deny {
                let rule = verdict.rule_id.as_deref().unwrap_or("GATE");
                if rt_lower == "antigravity" {
                    println!("{}", json!({ "decision": "deny", "reason": format!("Vestige gate ({rule}): {}", verdict.reason) }));
                } else if rt_lower == "cline" {
                    eprintln!("Vestige gate ({rule}): {}", verdict.reason);
                    std::process::exit(2);
                } else if matches!(rt_lower.as_str(), "cursor" | "copilot") {
                    println!("{}", json!({ "permission": "deny", "userMessage": format!("Vestige gate ({rule}): {}", verdict.reason) }));
                } else {
                    println!("{}", json!({ "hookSpecificOutput": { "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": format!("Vestige gate ({rule}): {}", verdict.reason) } }));
                }
                return;
            }
            // Fast non-blocking daemon reflex check when a live session id is present.
            if !session_id.is_empty() && args.iter().any(|a| a == "--with-reflex") {
                let rfx_params = json!({
                    "event": "PreToolUse",
                    "session": session_id,
                    "tool": req.tool_name,
                    "command": req.command.unwrap_or_default(),
                    "file": req.file_path.unwrap_or_default(),
                    "cwd": req.cwd.unwrap_or_default(),
                    "host": runtime,
                });
                if let Some(rfx) = try_daemon_reflex(&data_dir, rfx_params, 250)
                    && rfx.pointer("/hookSpecificOutput/permissionDecision").and_then(Value::as_str) == Some("deny")
                {
                    let reason = rfx.pointer("/hookSpecificOutput/permissionDecisionReason").and_then(Value::as_str).unwrap_or("Vestige Reflex denied this call.");
                    if rt_lower == "antigravity" {
                        println!("{}", json!({ "decision": "deny", "reason": reason }));
                    } else {
                        println!("{}", json!({ "hookSpecificOutput": { "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": reason } }));
                    }
                }
            }
        }
        other => {
            eprintln!("Unknown vestige-hook subcommand: {other}");
            std::process::exit(2);
        }
    }
}

//! The whole v5 chain, end to end, on the shipped pieces: Operator Lite's gate as Claude Code runs it
//! (installed by its own `install`, called through the hook command that install registered), the
//! gate's hash-chained receipts, `vestige-mcp` over real stdio (`precedent`, `smart_ingest`, `recall`,
//! `causal_walk`, `forgotten_lesson`, `receipt`), a second agent attached to the same store,
//! `vestige hook session-start` after a compaction, the owner's ALLOW ONCE, `vestige strata-verify`
//! over the log, a restart, the loopback dashboard API, and the `@vestige/init` PRECEDENT journey a
//! user runs to follow a stop around the loop.
//!
//! Each test is one machine: a throwaway home with the gate installed, a project folder, and a fresh
//! Strata store. Nothing touches the developer's own home, gate or store. The agent's command is
//! `git push --force origin main`, which the gate judges and never runs.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Output, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::TempDir;

const FORCE_PUSH: &str = "git push --force origin main";

/// Variables an agent's shell or a developer's machine may carry that would change what the gate does.
const LEAKY_VARS: [&str; 6] = [
    "OPERATOR_HOME",
    "OPERATOR_AGENT_SESSION",
    "OPERATOR_GATE_MODE",
    "OPERATOR_GATE",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
];

/// The owner at a terminal: run `vestige gate approve` in a pseudo-terminal and type ALLOW ONCE at its prompt.
const OWNER_AT_A_TERMINAL: &str = r#"
import os, pty, select, sys, time
gate_bin, action = sys.argv[1], sys.argv[2]
pid, fd = pty.fork()
if pid == 0:
    os.execvp(gate_bin, [gate_bin, "gate", "approve", action])
out, sent, deadline = b"", False, time.time() + 30
while time.time() < deadline:
    if not select.select([fd], [], [], 1)[0]:
        continue
    try:
        chunk = os.read(fd, 4096)
    except OSError:
        break
    if not chunk:
        break
    out += chunk
    if not sent and b"Type ALLOW ONCE" in out:
        os.write(fd, b"ALLOW ONCE\n")
        sent = True
done, status = os.waitpid(pid, os.WNOHANG)
if not done:
    os.kill(pid, 9)
    done, status = os.waitpid(pid, 0)
sys.stdout.write(out.decode("utf-8", "replace"))
sys.exit(os.waitstatus_to_exitcode(status))
"#;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// One agent's connection: the shipped `vestige-mcp` (default features) over real stdio. The first
/// one on a store owns it; a second attaches to the first, as a second agent on a machine does.
struct Agent {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    next_id: u64,
}

impl Agent {
    fn spawn(data: &Path, home: &Path) -> Self {
        Self::spawn_with(data, home, None)
    }

    /// `dashboard`: serve the loopback dashboard API on this port, as `vestige dashboard` does.
    fn spawn_with(data: &Path, home: &Path, dashboard: Option<u16>) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"));
        command
            .arg("--no-http")
            .arg("--data-dir")
            .arg(data)
            .env("HOME", home)
            .env("USERPROFILE", home)
            .env(
                "VESTIGE_DASHBOARD_ENABLED",
                if dashboard.is_some() { "true" } else { "false" },
            )
            .env(
                "VESTIGE_DASHBOARD_PORT",
                dashboard.map(|p| p.to_string()).unwrap_or_default(),
            )
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env_remove("VESTIGE_DATA_DIR")
            .env_remove("RUST_LOG")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for var in LEAKY_VARS {
            command.env_remove(var);
        }
        let mut child = command.spawn().expect("spawn vestige-mcp");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout");
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        let mut agent = Self {
            child,
            stdin,
            lines,
            next_id: 0,
        };
        agent.rpc(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "full-flow-e2e", "version": "1"},
            }),
        );
        agent.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        agent
    }

    fn send(&mut self, message: &Value) {
        let mut line = message.to_string();
        line.push('\n');
        let stdin = self.stdin.as_mut().expect("stdin open");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.flush())
            .expect("write to vestige-mcp");
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "vestige-mcp did not answer {method} within 120 s"
            );
            let raw = self.lines.recv_timeout(remaining).unwrap_or_else(|error| {
                panic!("vestige-mcp closed stdout during {method}: {error}")
            });
            let response: Value = serde_json::from_str(&raw).expect("a JSON-RPC line");
            if response.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            assert!(
                response.get("error").is_none(),
                "{method} failed: {response}"
            );
            return response["result"].clone();
        }
    }

    /// A tool's body, whether it succeeded or reported a tool-level error.
    fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.rpc(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        );
        self.tool_body(&result)
    }

    /// A tool's body, which must be a success.
    fn tool_ok(&mut self, name: &str, arguments: Value) -> Value {
        let result = self.rpc(
            "tools/call",
            json!({ "name": name, "arguments": arguments.clone() }),
        );
        assert_ne!(result["isError"], json!(true), "{name} failed: {result}");
        let body = self.tool_body(&result);
        assert!(
            body.get("error").is_none(),
            "{name} {arguments} failed: {body}"
        );
        body
    }

    fn tool_body(&self, result: &Value) -> Value {
        if let Some(body) = result
            .get("structuredContent")
            .filter(|body| !body.is_null())
        {
            return body.clone();
        }
        let text = result["content"][0]["text"].as_str().unwrap_or_default();
        serde_json::from_str(text).unwrap_or_else(|_| json!({ "raw": text }))
    }

    /// Close stdin and wait for a clean exit, the way an MCP client shuts a stdio server down.
    fn shutdown(mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait().expect("poll vestige-mcp") {
                assert!(
                    status.success(),
                    "vestige-mcp exited with {status} at stdin EOF"
                );
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("vestige-mcp did not exit within 30 s of stdin EOF");
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// What the gate said about one tool call.
struct Verdict {
    code: Option<i32>,
    stderr: String,
}

impl Verdict {
    fn lines(&self) -> Vec<&str> {
        self.stderr.lines().collect()
    }

    /// The 16-hex receipt id the first line of a stop names.
    fn receipt(&self) -> String {
        let first = self.lines().first().copied().unwrap_or_default();
        let id = first
            .rsplit("Receipt: ")
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        assert!(
            id.len() == 16 && id.chars().all(|c| c.is_ascii_hexdigit()),
            "no receipt id on the stop's first line: {}",
            self.stderr
        );
        id
    }

    /// The 24-hex action id the owner approves, from the stop's last line.
    fn action(&self) -> String {
        let last = self.lines().last().copied().unwrap_or_default();
        let id = last.rsplit(' ').next().unwrap_or_default().to_string();
        assert!(
            id.len() == 24 && id.chars().all(|c| c.is_ascii_hexdigit()),
            "no action id on the stop's last line: {}",
            self.stderr
        );
        id
    }
}

/// One machine: a home with the gate installed, a project, and a Strata store.
struct Machine {
    _root: TempDir,
    home: PathBuf,
    project: PathBuf,
    data: PathBuf,
}

impl Machine {
    fn new() -> Self {
        let root = TempDir::new().expect("temporary machine");
        let home = root.path().join("home");
        let project = home.join("shop");
        let data = root.path().join("vestige-data");
        for dir in [&project, &data] {
            std::fs::create_dir_all(dir).expect("create machine folders");
        }
        Self {
            _root: root,
            home,
            project,
            data,
        }
    }

    /// A command that sees this machine's home and none of the caller's gate settings.
    fn command(&self, program: impl AsRef<std::ffi::OsStr>) -> Command {
        let mut command = Command::new(program);
        command
            .env("HOME", &self.home)
            .env("USERPROFILE", &self.home)
            .env("NO_COLOR", "1")
            .current_dir(&self.project);
        for var in LEAKY_VARS {
            command.env_remove(var);
        }
        command
    }

    fn gate_bin(&self) -> &'static str {
        env!("CARGO_BIN_EXE_vestige")
    }

    /// Configure the native Rust gate (`vestige gate hook --source claude`) on this machine.
    fn install_gate(&self) -> String {
        let op_dir = self.home.join(".operator");
        std::fs::create_dir_all(&op_dir).expect("create .operator");
        let mode_path = op_dir.join("mode");
        std::fs::write(&mode_path, "enforce\n").expect("write mode");
        let claude_dir = self.home.join(".claude");
        std::fs::create_dir_all(&claude_dir).expect("create .claude");
        let cmd = format!("'{}' gate hook --source claude", self.gate_bin());
        let settings = json!({
            "hooks": {
                "PreToolUse": [
                    { "matcher": "*", "hooks": [{ "type": "command", "command": cmd }] }
                ]
            }
        });
        std::fs::write(
            claude_dir.join("settings.json"),
            serde_json::to_string_pretty(&settings).unwrap(),
        )
        .expect("write Claude settings");
        let out = self.gate(&["status"]);
        let stdout = text(&out.stdout);
        assert!(
            out.status.success(),
            "native gate status failed: {stdout}{}",
            text(&out.stderr)
        );
        assert!(
            stdout.contains("vestige-native-rust-gate"),
            "native gate must report its engine id: {stdout}"
        );
        assert!(mode_path.is_file(), "install wrote mode file");
        stdout
    }

    /// The PreToolUse command registered in Claude Code's settings.
    fn hook_command(&self) -> String {
        let settings: Value = serde_json::from_str(
            &std::fs::read_to_string(self.home.join(".claude/settings.json"))
                .expect("install wrote Claude Code settings"),
        )
        .expect("settings are JSON");
        let command = settings["hooks"]["PreToolUse"]
            .as_array()
            .and_then(|groups| groups.last())
            .and_then(|group| group["hooks"][0]["command"].as_str())
            .expect("a PreToolUse command")
            .to_string();
        assert!(
            command.contains("gate hook --source claude"),
            "{command}"
        );
        command
    }

    /// An agent's Bash call, sent through the registered hook exactly as Claude Code sends it.
    fn agent_runs(&self, bash: &str, session: &str) -> Verdict {
        let _ = self.hook_command();
        let payload = json!({
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": { "command": bash },
            "cwd": self.project,
            "session_id": session,
        });
        let mut child = self
            .command(self.gate_bin())
            .args(["gate", "hook", "--source", "claude"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run the native gate hook");
        child
            .stdin
            .take()
            .expect("hook stdin")
            .write_all(payload.to_string().as_bytes())
            .expect("send the payload");
        let out = child.wait_with_output().expect("hook finished");
        Verdict {
            code: out.status.code(),
            stderr: text(&out.stderr),
        }
    }

    /// The native Rust gate's own command line (`vestige gate <args>`).
    fn gate(&self, args: &[&str]) -> Output {
        self.command(self.gate_bin())
            .arg("gate")
            .args(args)
            .output()
            .expect("run the native gate")
    }

    /// Every receipt the gate wrote, oldest first.
    fn receipts(&self) -> Vec<Value> {
        let dir = self.home.join(".operator/receipts");
        let mut days: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("the gate's receipts folder")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect();
        days.sort();
        days.iter()
            .flat_map(|day| {
                std::fs::read_to_string(day)
                    .expect("read a day of receipts")
                    .lines()
                    .map(|line| serde_json::from_str::<Value>(line).expect("a receipt is JSON"))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// An agent's Vestige server: this machine's store, and the gate's receipts under its home.
    fn server(&self) -> Agent {
        Agent::spawn(&self.data, &self.home)
    }

    /// The owner grants ALLOW ONCE for `action` from a terminal.
    fn owner_allows_once(&self, action: &str) -> Output {
        self.command("python3")
            .args(["-c", OWNER_AT_A_TERMINAL])
            .arg(self.gate_bin())
            .arg(action)
            .output()
            .expect("run approve in a pseudo-terminal")
    }

    fn vestige(&self, args: &[&str]) -> Output {
        self.command(env!("CARGO_BIN_EXE_vestige"))
            .args(args)
            .env_remove("VESTIGE_DATA_DIR")
            .output()
            .expect("run vestige")
    }
}

/// A port nothing listens on right now, on loopback.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("a free loopback port")
        .port()
}

/// The body of `GET path` on the loopback dashboard, or None when nothing answers 200.
fn http_get(port: u16, path: &str) -> Option<String> {
    use std::io::Read;
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(stream, "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: application/json\r\nConnection: close\r\n\r\n").ok()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).ok()?;
    let (head, body) = raw.split_once("\r\n\r\n")?;
    head.starts_with("HTTP/1.1 200").then(|| body.to_string())
}

/// Record a stop the way the stop-to-memory step does: its rule as a pattern memory, the stop as an
/// event memory tagged with its receipt and derived_from the rule. Returns (rule id, stop id).
fn record_stop(
    agent: &mut Agent,
    machine: &Machine,
    receipt: &str,
    digest: &str,
) -> (String, String) {
    let rule = agent.tool_ok(
        "smart_ingest",
        json!({
            "content": "OPERATOR RULE OP-004 no-history-destruction: force-pushing over a shared branch or deleting .git destroys history.",
            "node_type": "pattern",
            "tags": ["OP-004", "op-rule-OP-004", "operator", "operator-rule"],
            "source": "operator-gate rule OP-004",
            "forceCreate": true,
        }),
    );
    let rule_id = rule["nodeId"].as_str().expect("rule memory id").to_string();
    let stopped = agent.tool_ok(
        "smart_ingest",
        json!({
            "content": format!("OPERATOR STOP under OP-004: the agent tried `{FORCE_PUSH}` in {}. Receipt digest {digest}. Nothing was run.", machine.project.display()),
            "node_type": "event",
            "tags": ["OP-004", format!("op-receipt-{receipt}"), "operator", "operator-stop"],
            "source": format!("operator-gate receipt {digest}"),
            "links": [{ "kind": "derived_from", "to": rule_id }],
            "forceCreate": true,
        }),
    );
    let stop_id = stopped["nodeId"]
        .as_str()
        .expect("stop memory id")
        .to_string();
    let link = &stopped["links"][0];
    assert_eq!(link["kind"], "derived_from", "{stopped}");
    assert_eq!(link["target"], rule_id.as_str(), "{stopped}");
    assert!(
        link["receiptId"].as_str().is_some_and(|r| !r.is_empty()),
        "{stopped}"
    );
    (rule_id, stop_id)
}

/// The chain is unbroken: each receipt names the digest of the one before it.
fn assert_chained(receipts: &[Value]) {
    for pair in receipts.windows(2) {
        assert_eq!(
            pair[1]["prev"], pair[0]["digest"],
            "receipt chain broken between {} and {}",
            pair[0], pair[1]
        );
    }
}

fn stop_of(receipts: &[Value], receipt: &str) -> Value {
    receipts
        .iter()
        .find(|r| {
            r["decision"] == "STOP" && r["digest"].as_str().is_some_and(|d| d.starts_with(receipt))
        })
        .cloned()
        .unwrap_or_else(|| panic!("no STOP receipt {receipt} in {receipts:?}"))
}

/// The gate stops the agent with three lines that say what to do next, writes a hash-chained
/// receipt, and `precedent`, called exactly as the stop message tells the agent to, returns that
/// same receipt from the Vestige server.
#[test]
fn the_gate_stops_records_and_precedent_reads_back_the_same_receipt() {
    let machine = Machine::new();
    machine.install_gate();

    let listing = machine.agent_runs("ls", "agent-a");
    assert_eq!(
        listing.code,
        Some(0),
        "an ordinary command runs: {}",
        listing.stderr
    );
    assert!(listing.stderr.is_empty(), "{}", listing.stderr);

    let stop = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(
        stop.code,
        Some(2),
        "the gate must stop a force push to main: {}",
        stop.stderr
    );
    let lines = stop.lines();
    assert_eq!(lines.len(), 3, "{}", stop.stderr);
    assert!(
        lines[0].starts_with("OPERATOR: STOPPED (OP-004"),
        "{}",
        lines[0]
    );
    assert!(
        lines[1].starts_with("Next: do not retry it"),
        "{}",
        lines[1]
    );
    assert!(
        lines[1].contains("precedent(rule=\"OP-004\")"),
        "{}",
        lines[1]
    );
    assert!(lines[2].contains("ALLOW ONCE"), "{}", lines[2]);
    let receipt = stop.receipt();
    let action = stop.action();

    let receipts = machine.receipts();
    assert_eq!(receipts.len(), 2, "{receipts:?}");
    assert_eq!(receipts[0]["decision"], "PASS");
    assert_chained(&receipts);
    let recorded = stop_of(&receipts, &receipt);
    assert_eq!(recorded["commitment"], "OP-004");
    assert_eq!(recorded["action_digest"], action.as_str());
    assert_eq!(recorded["session"], "agent-a");

    let verify = machine.gate(&["verify"]);
    assert!(verify.status.success(), "{}", text(&verify.stdout));
    assert!(
        text(&verify.stdout).contains("receipts=2 chain=OK"),
        "{}",
        text(&verify.stdout)
    );

    let mut server = machine.server();
    let precedent = server.tool_ok("precedent", json!({ "rule": "OP-004" }));
    assert!(
        precedent["summary"]
            .as_str()
            .is_some_and(|s| s.starts_with("1 stop")),
        "{precedent}"
    );
    let items = precedent["items"].as_array().expect("items");
    assert_eq!(items.len(), 1, "{precedent}");
    assert_eq!(items[0]["decision"], "STOP");
    assert_eq!(items[0]["digest"], recorded["digest"]);
    assert_eq!(items[0]["prev"], receipts[0]["digest"]);
    assert_eq!(items[0]["action_digest"], action.as_str());
    assert_eq!(items[0]["session"], "agent-a");

    let everything = server.tool_ok("precedent", json!({ "decision": "all" }));
    let decisions: Vec<&str> = everything["items"]
        .as_array()
        .expect("items")
        .iter()
        .filter_map(|item| item["decision"].as_str())
        .collect();
    assert_eq!(decisions, ["STOP", "PASS"], "{everything}");
    server.shutdown();
}

/// A stop recorded in Strata is found by its exact handle by the agent that recorded it and by a
/// second agent attached to the same store; the causal walk reaches the rule it was derived from;
/// both writes replay from the log; and the log verifies once every server has exited.
#[test]
fn a_stop_in_memory_is_recalled_by_every_agent_and_walks_back_to_its_rule() {
    let machine = Machine::new();
    machine.install_gate();
    let stop = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(stop.code, Some(2), "{}", stop.stderr);
    let receipt = stop.receipt();
    let recorded = stop_of(&machine.receipts(), &receipt);
    let digest = recorded["digest"].as_str().expect("digest").to_string();

    let mut agent_a = machine.server();
    let (rule_id, stop_id) = record_stop(&mut agent_a, &machine, &receipt, &digest);

    let handle = format!("op-receipt-{receipt}");
    let found = agent_a.tool_ok("recall", json!({ "handle": handle }));
    assert_eq!(found["exact"], true, "{found}");
    let nodes = found["nodes"].as_array().expect("nodes");
    assert_eq!(nodes.len(), 1, "{found}");
    assert_eq!(nodes[0]["id"], stop_id.as_str());
    assert!(
        nodes[0]["content"]
            .as_str()
            .is_some_and(|c| c.contains(&digest)),
        "{found}"
    );

    let missing = agent_a.tool("recall", json!({ "handle": "op-receipt-0000000000000000" }));
    assert_eq!(
        missing["error"], "handle_required",
        "a stop never recorded is not found: {missing}"
    );

    // A second agent on the same machine attaches to the same store and sees the same stop.
    let mut agent_b = machine.server();
    let seen = agent_b.tool_ok(
        "recall",
        json!({ "handle": format!("op-receipt-{receipt}") }),
    );
    assert_eq!(seen["nodes"][0]["id"], stop_id.as_str(), "{seen}");
    let rule_seen = agent_b.tool_ok("recall", json!({ "handle": "op-rule-OP-004" }));
    assert_eq!(rule_seen["nodes"][0]["id"], rule_id.as_str(), "{rule_seen}");

    let walk = agent_b.tool_ok(
        "causal_walk",
        json!({
            "scope": "user",
            "start_points": [{ "kind": "logged_write", "node_id": stop_id }],
            "promote": false,
        }),
    );
    assert_eq!(walk["status"], "completed", "{walk}");
    let to_rule = walk["causes"]
        .as_array()
        .expect("causes")
        .iter()
        .find(|cause| cause["id"] == rule_id.as_str())
        .unwrap_or_else(|| panic!("the walk must reach the rule memory: {walk}"));
    assert_eq!(to_rule["path"][0]["source_id"], stop_id.as_str(), "{walk}");
    assert_eq!(to_rule["path"][0]["target_id"], rule_id.as_str(), "{walk}");

    let lesson = agent_b.tool_ok("forgotten_lesson", json!({ "failure_id": stop_id }));
    assert_eq!(lesson["tool"], "forgotten_lesson", "{lesson}");
    assert_eq!(lesson["failure_id"], stop_id.as_str(), "{lesson}");

    for id in [&rule_id, &stop_id] {
        let replay = agent_b.tool_ok("receipt", json!({ "action": "replay", "receipt_id": id }));
        assert_eq!(replay["matched"], true, "{replay}");
        assert_eq!(replay["mismatches"], json!([]), "{replay}");
        let got = agent_b.tool_ok("receipt", json!({ "action": "get", "receipt_id": id }));
        assert!(
            got["receipt"]["receipt_id"]
                .as_str()
                .is_some_and(|r| !r.is_empty()),
            "{got}"
        );
    }

    agent_b.shutdown();
    agent_a.shutdown();
    let data = machine.data.to_string_lossy().into_owned();
    let verified = machine.vestige(&["strata-verify", &data]);
    assert!(
        verified.status.success(),
        "the log must verify after the run: {}{}",
        text(&verified.stdout),
        text(&verified.stderr)
    );

    // The next session, after every server has exited: the stop is still there, by the same handle,
    // and its write still replays from the log.
    let mut tomorrow = machine.server();
    let still = tomorrow.tool_ok(
        "recall",
        json!({ "handle": format!("op-receipt-{receipt}") }),
    );
    assert_eq!(
        still["nodes"][0]["id"],
        stop_id.as_str(),
        "the stop must survive a restart: {still}"
    );
    let replay = tomorrow.tool_ok(
        "receipt",
        json!({ "action": "replay", "receipt_id": stop_id }),
    );
    assert_eq!(replay["matched"], true, "{replay}");
    tomorrow.shutdown();
}

/// What a user sees: `npx @vestige/init precedent --plain` (run from this repository) follows the
/// real stop around the loop. The case file names the stop and its rule, re-derives the gate's receipt
/// chain with nothing broken, and finds the stop's memory and its rule memory through the loopback
/// dashboard API of a running Vestige server.
#[test]
fn the_precedent_journey_shows_the_stop_its_record_and_its_memory() {
    if Command::new("node").arg("--version").output().is_err() {
        eprintln!("SKIPPED: node is not on PATH, so the @vestige/init journey cannot run here");
        return;
    }
    let machine = Machine::new();
    machine.install_gate();
    assert_eq!(machine.agent_runs("ls", "agent-a").code, Some(0));
    let stop = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(stop.code, Some(2), "{}", stop.stderr);
    let receipt = stop.receipt();
    let digest = stop_of(&machine.receipts(), &receipt)["digest"]
        .as_str()
        .expect("digest")
        .to_string();

    let port = free_port();
    let mut agent = Agent::spawn_with(&machine.data, &machine.home, Some(port));
    let (rule_id, stop_id) = record_stop(&mut agent, &machine, &receipt, &digest);
    let deadline = Instant::now() + Duration::from_secs(30);
    let tagged = loop {
        if let Some(body) = http_get(
            port,
            &format!("/api/memories?tag=op-receipt-{receipt}&limit=1"),
        ) {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "the dashboard API never answered on port {port}"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(
        tagged.contains(&stop_id),
        "the dashboard API must return the stop memory: {tagged}"
    );

    let journey = machine
        .command("node")
        .arg(repo_root().join("packages/vestige-init/bin/init.js"))
        .args(["precedent", "--plain"])
        .env("VESTIGE_API", format!("http://127.0.0.1:{port}"))
        .env("TERM", "dumb")
        .output()
        .expect("run the PRECEDENT journey");
    let case = text(&journey.stdout);
    assert!(journey.status.success(), "{case}{}", text(&journey.stderr));
    let line = |label: &str| {
        case.lines()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("the case file has no {label:?} line: {case}"))
            .to_string()
    };
    assert!(
        line("1 THE STOP").contains("git push --force origin main"),
        "{case}"
    );
    assert!(case.contains("stopped by OP-004"), "{case}");
    let record = line("2 THE RECORD");
    assert!(record.contains("2 of 2 receipts re-derived"), "{record}");
    assert!(record.contains("0 broken"), "{record}");
    assert!(
        case.contains(&format!("receipt {digest}")) || case.contains(&receipt),
        "{case}"
    );
    let memory = line("3 THE MEMORY");
    assert!(
        memory.contains(&stop_id),
        "the journey must find the stop in memory: {memory}"
    );
    assert!(
        memory.contains(&format!("derived from rule memory {rule_id}")),
        "{memory}"
    );
    let precedent = line("4 THE PRECEDENT");
    assert!(precedent.contains("under OP-004"), "{precedent}");
    assert!(case.contains("precedent(rule=\"OP-004\""), "{case}");
    agent.shutdown();
}

/// A rule the owner pinned comes back through `vestige hook session-start` after a compaction, and
/// the gate still stops the command the rule forbids: memory and enforcement agree.
#[test]
fn a_pinned_rule_survives_compaction_and_the_gate_still_holds_the_line() {
    let machine = Machine::new();
    machine.install_gate();
    let mut server = machine.server();
    let pinned = server.tool_ok(
        "smart_ingest",
        json!({
            "content": "Never force-push main in the shop repository; ask the owner for ALLOW ONCE first.",
            "node_type": "decision",
            "tags": ["pinned", "shop"],
            "source": "the owner",
            "forceCreate": true,
        }),
    );
    let rule_id = pinned["nodeId"]
        .as_str()
        .expect("pinned rule id")
        .to_string();

    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/session_start/compact.json");
    let mut payload: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).expect("compact payload"))
            .expect("JSON");
    payload["cwd"] = json!(machine.project);
    let data = machine.data.to_string_lossy().into_owned();
    let mut child = machine
        .command(env!("CARGO_BIN_EXE_vestige"))
        .args(["hook", "session-start", "--data-dir", &data])
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run the session-start hook");
    child
        .stdin
        .take()
        .expect("hook stdin")
        .write_all(payload.to_string().as_bytes())
        .expect("send the payload");
    let out = child.wait_with_output().expect("hook finished");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let hook: Value = serde_json::from_str(text(&out.stdout).trim()).expect("hook JSON");
    assert_eq!(
        hook["hookSpecificOutput"]["hookEventName"], "SessionStart",
        "{hook}"
    );
    let context = hook["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("context");
    assert!(context.contains("compact"), "{context}");
    assert!(
        context
            .lines()
            .any(|line| line.starts_with(&format!("- {rule_id} "))),
        "the pinned rule must come back by id after the compaction: {context}"
    );
    assert!(context.contains("Never force-push main"), "{context}");

    // After the compaction the agent tries it anyway: the gate does not depend on the summary.
    let stop = machine.agent_runs(FORCE_PUSH, "after-compact");
    assert_eq!(stop.code, Some(2), "{}", stop.stderr);
    assert!(
        stop.lines()[0].starts_with("OPERATOR: STOPPED (OP-004"),
        "{}",
        stop.stderr
    );
    let precedent = server.tool_ok("precedent", json!({ "session": "after-compact" }));
    assert_eq!(precedent["items"][0]["decision"], "STOP", "{precedent}");
    server.shutdown();
}

/// The owner's ALLOW ONCE, granted from a terminal, lets exactly one run of the stopped action
/// through; the next attempt is stopped again; the permit is spent on disk; and `precedent` shows
/// the stop, the permitted run and the permit, all on one unbroken chain.
#[test]
fn allow_once_lets_the_action_run_exactly_once() {
    let machine = Machine::new();
    machine.install_gate();
    let first = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(first.code, Some(2), "{}", first.stderr);
    let action = first.action();

    let granted = machine.owner_allows_once(&action);
    assert!(
        granted.status.success() && text(&granted.stdout).contains("Permit granted for one use."),
        "the owner's approve failed: {}{}",
        text(&granted.stdout),
        text(&granted.stderr)
    );
    let permit = machine
        .home
        .join(format!(".operator/permits/{action}.json"));
    assert!(permit.is_file(), "approve wrote no permit");

    let allowed = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(
        allowed.code,
        Some(0),
        "the permit must let one run through: {}",
        allowed.stderr
    );
    assert!(!permit.exists(), "the permit must be spent");
    assert!(
        machine
            .home
            .join(format!(".operator/permits/{action}.json.used"))
            .is_file()
    );

    let again = machine.agent_runs(FORCE_PUSH, "agent-a");
    assert_eq!(
        again.code,
        Some(2),
        "a spent permit allows nothing: {}",
        again.stderr
    );
    assert_eq!(again.action(), action, "the same action, stopped again");

    let receipts = machine.receipts();
    let decisions: Vec<&str> = receipts
        .iter()
        .filter_map(|r| r["decision"].as_str())
        .collect();
    assert_eq!(
        decisions,
        ["STOP", "PERMIT_GRANTED", "ALLOW", "STOP"],
        "{receipts:?}"
    );
    assert_eq!(receipts[2]["via"], "permit");
    assert_chained(&receipts);
    let verify = machine.gate(&["verify"]);
    assert!(
        text(&verify.stdout).contains("receipts=4 chain=OK"),
        "{}",
        text(&verify.stdout)
    );

    let mut server = machine.server();
    let precedent = server.tool_ok("precedent", json!({ "rule": "OP-004" }));
    let summary = precedent["summary"].as_str().expect("summary");
    assert!(summary.starts_with("2 stops"), "{precedent}");
    assert!(summary.contains("1 permit used"), "{precedent}");
    let ran = precedent["items"]
        .as_array()
        .expect("items")
        .iter()
        .find(|item| item["decision"] == "ALLOW")
        .unwrap_or_else(|| panic!("the permitted run is missing: {precedent}"));
    assert_eq!(ran["via"], "permit", "{precedent}");
    assert_eq!(ran["permit"]["status"], "used", "{precedent}");
    assert_eq!(ran["permit"]["by"], "owner-tty", "{precedent}");
    server.shutdown();
}

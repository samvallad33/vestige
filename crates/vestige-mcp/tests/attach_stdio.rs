//! Real binaries: every MCP client on a machine can use one Vestige store.
//!
//! The Strata log has one writer. These tests start `vestige-mcp` the way an
//! MCP client does (a child process speaking line-framed JSON-RPC on stdio)
//! several times against one data directory, and check what a user sees:
//! every client connects, every client reads what the others wrote, a client
//! keeps working when the process that owned the store is killed, and the
//! `vestige` CLI backs up through a running server instead of opening the log
//! as a second writer.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// Bound on one JSON-RPC answer. A hang is a failure, never a wedged job.
const RPC_TIMEOUT: Duration = Duration::from_secs(90);

struct Client {
    name: &'static str,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
}

impl Client {
    fn spawn(name: &'static str, data_dir: &Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vestige-mcp"))
            .env("VESTIGE_DATA_DIR", data_dir)
            .env("VESTIGE_DASHBOARD_ENABLED", "false")
            .env("VESTIGE_HTTP_ENABLED", "0")
            .env("VESTIGE_AUTOPILOT_ENABLED", "0")
            .env("RUST_LOG", "info")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige-mcp");
        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || {
                for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                    sink.lock().unwrap().push(line);
                }
            });
        }
        Self {
            name,
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
        }
    }

    fn stderr(&self) -> String {
        self.stderr.lock().unwrap().join("\n")
    }

    fn send(&mut self, message: &Value) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        writeln!(stdin, "{message}").expect("write request");
        stdin.flush().expect("flush request");
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let deadline = Instant::now() + RPC_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.stdout.recv_timeout(left) {
                Ok(line) => {
                    let message: Value = serde_json::from_str(&line).unwrap_or_else(|e| {
                        panic!("{}: stdout line is not JSON ({e}): {line}", self.name)
                    });
                    if message.get("method").is_none() && message["id"] == json!(id) {
                        return message;
                    }
                    assert!(
                        message.get("method").is_some(),
                        "{}: answer for an id nobody asked about: {line}",
                        self.name
                    );
                }
                Err(RecvTimeoutError::Timeout) => panic!(
                    "{}: no answer to {method} within {RPC_TIMEOUT:?}\nstderr:\n{}",
                    self.name,
                    self.stderr()
                ),
                Err(RecvTimeoutError::Disconnected) => panic!(
                    "{}: stdout closed before answering {method}\nstderr:\n{}",
                    self.name,
                    self.stderr()
                ),
            }
        }
    }

    fn initialize(&mut self) {
        let answer = self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": self.name, "version": "1"}
            }),
        );
        assert!(
            answer["result"]["serverInfo"]["name"].is_string(),
            "{}: initialize failed: {answer}",
            self.name
        );
        self.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    /// A tool call's structured result; panics on any error answer.
    fn call(&mut self, tool: &str, arguments: Value) -> Value {
        let answer = self.request("tools/call", json!({"name": tool, "arguments": arguments}));
        let result = &answer["result"];
        assert!(
            answer.get("error").is_none() && result["isError"] != json!(true),
            "{}: {tool} failed: {answer}\nstderr:\n{}",
            self.name,
            self.stderr()
        );
        if !result["structuredContent"].is_null() {
            return result["structuredContent"].clone();
        }
        let text = result["content"][0]["text"].as_str().unwrap_or("null");
        serde_json::from_str(text).unwrap_or(Value::String(text.to_string()))
    }

    fn remember(&mut self, content: &str) -> String {
        let saved = self.call(
            "smart_ingest",
            json!({"content": content, "forceCreate": true, "tags": ["attach-test"]}),
        );
        saved["nodeId"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: smart_ingest gave no nodeId: {saved}", self.name))
            .to_string()
    }

    fn sees(&mut self, id: &str) -> bool {
        let got = self.call("memory", json!({"action": "get", "id": id}));
        got["found"] != json!(false) && got.to_string().contains(id)
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn wait_exit(&mut self, within: Duration) -> ExitStatus {
        let deadline = Instant::now() + within;
        loop {
            if let Some(status) = self.child.try_wait().expect("poll child") {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "{}: did not exit within {within:?}\nstderr:\n{}",
                self.name,
                self.stderr()
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    fn running(&mut self) -> bool {
        self.child.try_wait().expect("poll child").is_none()
    }
}

impl Client {
    fn send_raw(&mut self, bytes: &[u8]) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        stdin.write_all(bytes).expect("write raw");
        stdin.flush().expect("flush raw");
    }

    fn wait_stderr(&self, needle: &str, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if self.stderr().contains(needle) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    /// Every stdout line that arrives within `within`.
    fn drain(&mut self, within: Duration) -> Vec<Value> {
        let deadline = Instant::now() + within;
        let mut out = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.stdout.recv_timeout(left) {
                Ok(line) => out.push(serde_json::from_str(&line).unwrap_or(Value::String(line))),
                Err(_) => return out,
            }
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn cli(data_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env_remove("VESTIGE_DATA_DIR")
        .env("NO_COLOR", "1")
        .output()
        .expect("run vestige CLI")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn three_clients_share_one_store_through_one_owner() {
    let dir = tempfile::tempdir().unwrap();
    let mut owner = Client::spawn("owner", dir.path());
    owner.initialize();

    // Before the attach layer, the second server blocked on the lock forever.
    let started = Instant::now();
    let mut second = Client::spawn("second", dir.path());
    second.initialize();
    let mut third = Client::spawn("third", dir.path());
    third.initialize();
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "attaching took {:?}",
        started.elapsed()
    );
    assert!(
        second.stderr().contains("relaying this client"),
        "{}",
        second.stderr()
    );
    assert!(
        third.stderr().contains("relaying this client"),
        "{}",
        third.stderr()
    );

    // Unix listens on a socket in the data directory; only Windows (or a
    // path too long for a socket address) uses loopback TCP.
    let endpoint = std::fs::read_to_string(dir.path().join(".serve.endpoint")).unwrap();
    if cfg!(unix) {
        assert!(endpoint.starts_with("unix "), "{endpoint}");
        assert!(dir.path().join(".serve.sock").exists());
    } else {
        assert!(
            endpoint.split(' ').next().unwrap().parse::<u16>().is_ok(),
            "{endpoint}"
        );
    }

    let by_owner = owner.remember("attach test: written by the owning process");
    let by_second = second.remember("attach test: written by the second client");
    let by_third = third.remember("attach test: written by the third client");
    for id in [&by_owner, &by_second, &by_third] {
        assert!(owner.sees(id), "owner cannot read {id}");
        assert!(second.sees(id), "second cannot read {id}");
        assert!(third.sees(id), "third cannot read {id}");
    }

    // The owner's own client leaves; the owner keeps serving the others.
    owner.close_stdin();
    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        owner.running(),
        "the owner exited with clients still attached"
    );
    let late = second.remember("attach test: written after the owner's client left");
    assert!(third.sees(&late));

    second.close_stdin();
    third.close_stdin();
    assert!(second.wait_exit(Duration::from_secs(30)).success());
    assert!(third.wait_exit(Duration::from_secs(30)).success());
    assert!(owner.wait_exit(Duration::from_secs(30)).success());
    assert!(
        !dir.path().join(".serve.endpoint").exists(),
        "a clean exit retires the endpoint file"
    );
    assert!(
        !dir.path().join(".serve.sock").exists(),
        "a clean exit removes the socket"
    );

    // Everything any client wrote is in the log a fresh process opens.
    let mut reopened = Client::spawn("reopened", dir.path());
    reopened.initialize();
    for id in [&by_owner, &by_second, &by_third, &late] {
        assert!(reopened.sees(id), "{id} did not survive a restart");
    }
}

#[test]
fn an_attached_client_survives_its_owner_being_killed() {
    let dir = tempfile::tempdir().unwrap();
    let mut owner = Client::spawn("owner", dir.path());
    owner.initialize();
    let mut survivor = Client::spawn("survivor", dir.path());
    survivor.initialize();
    let before = survivor.remember("attach test: written before the owner died");

    // An MCP client killing its server on exit looks exactly like this.
    owner.child.kill().expect("kill the owner");
    owner.child.wait().expect("reap the owner");

    // The same session, no new handshake from the client: the survivor
    // takes the lock and serves itself. A request sent in the instant before
    // the relay notices the owner is gone is answered with the lost-owner
    // error by design (it may have reached the owner), so the next call
    // waits for the takeover.
    assert!(
        survivor.wait_stderr("now serves", Duration::from_secs(30)),
        "{}",
        survivor.stderr()
    );
    let after = survivor.remember("attach test: written after the owner died");
    assert!(
        survivor.sees(&before),
        "a memory acknowledged before the kill was lost"
    );
    assert!(survivor.sees(&after));
    assert!(
        survivor.stderr().contains("now serves"),
        "{}",
        survivor.stderr()
    );

    // A client started now attaches to the survivor.
    let mut newcomer = Client::spawn("newcomer", dir.path());
    newcomer.initialize();
    assert!(
        newcomer.stderr().contains("relaying this client"),
        "{}",
        newcomer.stderr()
    );
    assert!(newcomer.sees(&before));
    assert!(newcomer.sees(&after));

    newcomer.close_stdin();
    survivor.close_stdin();
    assert!(newcomer.wait_exit(Duration::from_secs(30)).success());
    assert!(survivor.wait_exit(Duration::from_secs(30)).success());

    let log = dir.path().join("log");
    let verify = cli(dir.path(), &["strata-verify", &log.to_string_lossy()]);
    assert!(
        verify.status.success(),
        "the log fails verification:\n{}",
        text(&verify)
    );
}

#[test]
fn cli_backs_up_through_a_running_server_and_never_opens_a_served_log() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let id = server.remember("attach test: must be in the backup");

    // A command that opens the log directly refuses instead of becoming a
    // second writer.
    let stats = cli(dir.path(), &["stats"]);
    assert!(
        !stats.status.success(),
        "stats opened a served log:\n{}",
        text(&stats)
    );
    assert!(text(&stats).contains("is serving"), "{}", text(&stats));

    // Backup goes through the server.
    let out = dir.path().join("nightly").join("backup.strata");
    let backup = cli(dir.path(), &["backup", &out.to_string_lossy()]);
    assert!(backup.status.success(), "backup failed:\n{}", text(&backup));
    assert!(
        text(&backup).contains("backing up through it"),
        "{}",
        text(&backup)
    );
    assert!(out.join("log").is_dir(), "no log/ in the backup");
    let leftovers = std::fs::read_dir(dir.path().join("backups"))
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(
        leftovers, 0,
        "the server-side copy was left behind after the move"
    );

    // The server keeps working after the backup.
    assert!(server.sees(&id));
    server.close_stdin();
    assert!(server.wait_exit(Duration::from_secs(30)).success());

    // With no server running, the CLI opens the store itself.
    let stats = cli(dir.path(), &["stats"]);
    assert!(
        stats.status.success(),
        "stats failed with no server:\n{}",
        text(&stats)
    );

    // The backup is a working store on its own.
    let mut restored = Client::spawn("restored", &out);
    restored.initialize();
    assert!(restored.sees(&id), "the backup does not hold the memory");
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Status line of a plain HTTP GET, or `None` when nothing answers.
fn http_status(port: u16, path: &str) -> Option<String> {
    use std::io::Read;
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    response.lines().next().map(str::to_string)
}

#[test]
fn dashboard_command_is_served_by_the_running_server() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let id = server.remember("attach test: visible while the dashboard runs");

    let port = free_port();
    let mut dashboard = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(dir.path())
        .args(["dashboard", "--port", &port.to_string(), "--no-open"])
        .env_remove("VESTIGE_DATA_DIR")
        .env("NO_COLOR", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run vestige dashboard");
    let (tx, lines) = channel();
    let stdout = dashboard.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let deadline = Instant::now() + RPC_TIMEOUT;
    let mut seen = Vec::new();
    loop {
        let line = lines
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_else(|_| panic!("vestige dashboard printed no URL:\n{}", seen.join("\n")));
        seen.push(line.clone());
        if line.contains("served by vestige-mcp") {
            break;
        }
    }
    assert!(
        seen.iter()
            .any(|line| line.contains(&format!("127.0.0.1:{port}"))),
        "{}",
        seen.join("\n")
    );
    let status = http_status(port, "/api/health").expect("the dashboard answers");
    assert!(status.contains(" 200"), "GET /api/health: {status}");

    // MCP keeps working beside the dashboard.
    assert!(server.sees(&id));

    // The agent's client leaves; the dashboard's lease keeps the store served.
    server.close_stdin();
    std::thread::sleep(Duration::from_millis(1500));
    assert!(server.running(), "the owner exited under an open dashboard");
    let status = http_status(port, "/api/health").expect("still answering");
    assert!(status.contains(" 200"), "GET /api/health: {status}");

    // Closing the dashboard releases the owner.
    dashboard.kill().unwrap();
    dashboard.wait().unwrap();
    assert!(server.wait_exit(Duration::from_secs(30)).success());
}

/// `vestige dashboard --port <port>` against a store `vestige-mcp` holds, read
/// up to the line naming the process that serves it. Returns the child and
/// every stdout line seen so far.
fn leased_dashboard(dir: &Path, port: u16) -> (Child, Vec<String>) {
    let mut dashboard = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(dir)
        .args(["dashboard", "--port", &port.to_string(), "--no-open"])
        .env_remove("VESTIGE_DATA_DIR")
        .env("NO_COLOR", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run vestige dashboard");
    let (tx, lines) = channel();
    let stdout = dashboard.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let deadline = Instant::now() + RPC_TIMEOUT;
    let mut seen = Vec::new();
    loop {
        let Ok(line) = lines.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        else {
            let _ = dashboard.kill();
            let _ = dashboard.wait();
            panic!("vestige dashboard printed no URL:\n{}", seen.join("\n"));
        };
        seen.push(line.clone());
        if line.contains("Press Ctrl+C") {
            return (dashboard, seen);
        }
    }
}

/// Wait until nothing answers on `port`, or panic after `within`.
fn wait_until_closed(port: u16, within: Duration) {
    let deadline = Instant::now() + within;
    while http_status(port, "/api/health").is_some() {
        assert!(
            Instant::now() < deadline,
            "the dashboard still answers on {port} {within:?} after its last lease closed"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// README: the shared server serves the dashboard "until you press Ctrl+C".
/// The agent's own session keeps the server running, so the dashboard must
/// stop with the last `vestige dashboard` rather than with the agent. A later
/// `--port` that the running dashboard cannot honour is reported.
#[test]
fn the_dashboard_stops_when_the_last_vestige_dashboard_exits() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let id = server.remember("attach test: the agent outlives the dashboard");

    let port = free_port();
    let (mut first, seen) = leased_dashboard(dir.path(), port);
    assert!(
        seen.iter()
            .any(|line| line.contains("served by vestige-mcp")),
        "{}",
        seen.join("\n")
    );
    let status = http_status(port, "/api/health").expect("the dashboard answers");
    assert!(status.contains(" 200"), "GET /api/health: {status}");

    // A second `vestige dashboard` asking for another port shares the
    // running one and says so.
    let other = free_port();
    let (mut second, seen) = leased_dashboard(dir.path(), other);
    let said = seen.join("\n");
    assert!(said.contains(&format!("127.0.0.1:{port}")), "{said}");
    assert!(
        said.contains(&format!("--port {other}")),
        "a second dashboard's --port was ignored silently:\n{said}"
    );
    assert!(http_status(other, "/api/health").is_none());

    // One lease left: still served.
    second.kill().unwrap();
    second.wait().unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let status = http_status(port, "/api/health").expect("one lease still holds it");
    assert!(status.contains(" 200"), "GET /api/health: {status}");

    // The last lease closes: the listener goes away, the agent stays.
    first.kill().unwrap();
    first.wait().unwrap();
    wait_until_closed(port, Duration::from_secs(10));
    assert!(
        server.running(),
        "closing the dashboard ended the agent's server"
    );
    assert!(server.sees(&id));

    // The same port serves again on the next request.
    let (mut again, _) = leased_dashboard(dir.path(), port);
    let status = http_status(port, "/api/health").expect("the dashboard answers again");
    assert!(status.contains(" 200"), "GET /api/health: {status}");
    again.kill().unwrap();
    again.wait().unwrap();
    wait_until_closed(port, Duration::from_secs(10));

    server.close_stdin();
    assert!(server.wait_exit(Duration::from_secs(30)).success());
}

#[test]
fn a_data_dir_too_long_for_a_socket_attaches_over_loopback_tcp() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("a".repeat(60)).join("b".repeat(60));
    std::fs::create_dir_all(&dir).unwrap();
    let mut owner = Client::spawn("owner", &dir);
    owner.initialize();
    let mut second = Client::spawn("second", &dir);
    second.initialize();

    let endpoint = std::fs::read_to_string(dir.join(".serve.endpoint")).unwrap();
    let port = endpoint.split(' ').next().unwrap();
    assert!(
        port.parse::<u16>().is_ok(),
        "expected a TCP port: {endpoint}"
    );
    assert!(!dir.join(".serve.sock").exists());

    let id = second.remember("attach test: over loopback TCP");
    assert!(owner.sees(&id));
    second.close_stdin();
    owner.close_stdin();
    assert!(second.wait_exit(Duration::from_secs(30)).success());
    assert!(owner.wait_exit(Duration::from_secs(30)).success());
}

/// A request the relay had half read when its owner died reaches the server
/// the relay turns into, and is answered.
#[test]
fn promotion_keeps_a_half_read_request() {
    let dir = tempfile::tempdir().unwrap();
    let mut owner = Client::spawn("owner", dir.path());
    owner.initialize();
    let mut proxy = Client::spawn("proxy", dir.path());
    proxy.initialize();

    let request = json!({"jsonrpc": "2.0", "id": 99, "method": "tools/call",
        "params": {"name": "memory_status", "arguments": {"view": "health"}}})
    .to_string();
    let (head, tail) = request.split_at(request.len() / 2);
    proxy.send_raw(head.as_bytes());
    std::thread::sleep(Duration::from_millis(500));

    owner.child.kill().unwrap();
    owner.child.wait().unwrap();
    assert!(
        proxy.wait_stderr("now serves", Duration::from_secs(30)),
        "{}",
        proxy.stderr()
    );
    proxy.send_raw(format!("{tail}\n").as_bytes());
    let seen = proxy.drain(Duration::from_secs(10));
    assert!(
        seen.iter()
            .any(|m| m["id"] == json!(99) && m.get("result").is_some()),
        "request 99 was never answered; got {seen:?}"
    );
}

/// One line of invalid UTF-8 does not end a relayed session.
#[test]
fn a_relayed_session_survives_an_invalid_utf8_line() {
    let dir = tempfile::tempdir().unwrap();
    let mut owner = Client::spawn("owner", dir.path());
    owner.initialize();
    let mut proxy = Client::spawn("proxy", dir.path());
    proxy.initialize();

    proxy.send_raw(b"\xff\xfe garbage\n");
    std::thread::sleep(Duration::from_millis(500));
    assert!(proxy.running(), "the relayed session ended on one bad line");
    let answer = proxy.request("ping", json!({}));
    assert!(answer.get("result").is_some(), "proxy: {answer}");
}

/// A definite refusal from the serving process (its dashboard port is taken)
/// is reported at once, not retried for the whole election wait.
#[test]
fn a_dashboard_refusal_is_reported_promptly() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = busy.local_addr().unwrap().port();
    let started = Instant::now();
    let out = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(dir.path())
        .args(["dashboard", "--port", &port.to_string(), "--no-open"])
        .env_remove("VESTIGE_DATA_DIR")
        .env("VESTIGE_ATTACH_WAIT_SECS", "15")
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    let took = started.elapsed();
    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        took < Duration::from_secs(5),
        "a refusal took {took:?} to report"
    );
    assert!(
        text(&out).contains("could not start the dashboard"),
        "{}",
        text(&out)
    );
    drop(busy);
}

/// vestige-restore writes to the log directly, so it refuses a served store
/// instead of becoming a second writer.
#[test]
fn restore_refuses_a_store_a_server_holds() {
    let dir = tempfile::tempdir().unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let backup = dir.path().join("backup.json");
    let inner = json!({"results": [{"content": "restored while served"}]}).to_string();
    std::fs::write(
        &backup,
        json!([{"type": "text", "text": inner}]).to_string(),
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_vestige-restore"))
        .arg(&backup)
        .env("VESTIGE_DATA_DIR", dir.path())
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "restore ran beside a server:\n{}",
        text(&out)
    );
    assert!(text(&out).contains("is serving"), "{}", text(&out));

    server.close_stdin();
    assert!(server.wait_exit(Duration::from_secs(30)).success());
    let out = Command::new(env!("CARGO_BIN_EXE_vestige-restore"))
        .arg(&backup)
        .env("VESTIGE_DATA_DIR", dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "restore failed with no server:\n{}",
        text(&out)
    );
}

/// The lock file and the data directory are owner-only: another user who
/// could read the lock could hold it and lock the owner out.
#[cfg(unix)]
#[test]
fn the_lock_file_and_data_dir_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut server = Client::spawn("server", dir.path());
    server.initialize();
    let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&dir.path().join(".serve.lock")), 0o600);
    assert_eq!(mode(&dir.path().join(".serve.sock")), 0o600);
    assert_eq!(mode(&dir.path().join(".serve.endpoint")), 0o600);
    assert_eq!(mode(dir.path()), 0o700);
}

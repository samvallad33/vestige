//! A default-feature `vestige-mcp` boot opens no outbound socket and does no DNS.
//!
//! Linux drives the real binary under `strace -f -e trace=network,connect`:
//! initialize, tools/list, then one tools/call. The removed npm client waited
//! up to 2s, so the process stays up past that before the trace is read.

#[cfg(target_os = "linux")]
use serde_json::{Value, json};

fn is_outbound_syscall(line: &str) -> bool {
    // strace attach/detach chatter is not a syscall.
    if line.contains("+++") || line.contains("---") || line.contains("strace:") {
        return false;
    }
    let hay = line.to_ascii_lowercase();
    // AF_INET6 contains AF_INET. nscd is glibc's first step in getaddrinfo.
    // AF_UNIX and AF_NETLINK are local; AF_UNSPEC is a UDP disconnect.
    line.contains("AF_INET")
        || line.contains("AF_PACKET")
        || hay.contains("nscd")
        || hay.contains("npmjs")
        || hay.contains("htons(53)")
        || hay.contains("io.systemd.resolve")
}

#[test]
fn outbound_trace_detector_flags_dns_and_ignores_local_sockets() {
    assert!(!is_outbound_syscall(
        r#"socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 3"#
    ));
    assert!(!is_outbound_syscall(
        r#"socket(AF_NETLINK, SOCK_RAW|SOCK_CLOEXEC, NETLINK_ROUTE) = 5"#
    ));
    assert!(!is_outbound_syscall(
        r#"connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0"#
    ));
    assert!(is_outbound_syscall(
        r#"connect(5, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (No such file or directory)"#
    ));
    assert!(is_outbound_syscall(
        r#"socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC|SOCK_NONBLOCK, IPPROTO_IP) = 5"#
    ));
    assert!(is_outbound_syscall(
        r#"connect(5, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("198.18.0.53")}, 16) = 0"#
    ));
    assert!(is_outbound_syscall(
        r#"connect(5, {sa_family=AF_INET6, sin6_port=htons(443), inet_pton(AF_INET6, "::1", &sin6_addr)}, 28) = -1"#
    ));
    assert!(is_outbound_syscall(
        r#"sendmmsg(5, [{iov_base="\10registry\5npmjs\3org\0"}], 1, MSG_NOSIGNAL) = 1"#
    ));
}

#[cfg(target_os = "linux")]
use std::io::{BufRead, BufReader, Write};
#[cfg(target_os = "linux")]
use std::process::{Child, ChildStdin, Command, Stdio};
#[cfg(target_os = "linux")]
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};
#[cfg(target_os = "linux")]
use std::time::Duration;

#[cfg(target_os = "linux")]
const RPC_TIMEOUT: Duration = Duration::from_secs(120);
/// Longer than the removed registry client's 2s timeout.
#[cfg(target_os = "linux")]
const POST_CALL_WINDOW: Duration = Duration::from_secs(3);

#[cfg(target_os = "linux")]
#[test]
fn default_boot_opens_no_outbound_socket_or_dns() {
    use std::os::unix::fs::PermissionsExt;

    let data = tempfile::tempdir().expect("data dir");
    let scratch = tempfile::tempdir().expect("scratch dir");
    let binary = scratch.path().join("vestige-mcp");
    std::fs::copy(env!("CARGO_BIN_EXE_vestige-mcp"), &binary).expect("copy vestige-mcp");
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))
        .expect("mark vestige-mcp executable");
    let trace_path = scratch.path().join("boot.strace");

    let mut command = Command::new("strace");
    command
        .arg("-f")
        .arg("-s")
        .arg("256")
        .arg("-e")
        .arg("trace=network,connect")
        .arg("-o")
        .arg(&trace_path)
        .arg("--")
        .arg(&binary)
        .arg("--no-http")
        .arg("--data-dir")
        .arg(data.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if name.starts_with("VESTIGE_")
            || name.starts_with("REDMINE_")
            || name.starts_with("GITHUB_")
        {
            command.env_remove(key);
        }
    }
    command
        .env("VESTIGE_DASHBOARD_ENABLED", "false")
        .env("VESTIGE_HTTP_ENABLED", "false")
        .env("RUST_LOG", "error");
    // strace and vestige-mcp share a group so a failed test can kill both.
    // Dropping strace alone detaches and leaves the server running.
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }

    let child = command.spawn().unwrap_or_else(|error| {
        panic!("spawn strace -f -e trace=network,connect (install strace): {error}")
    });
    let mut guard = KillGroup::new(child);
    let mut session = Session::from_child(guard.child_mut());

    let initialized = session.result(
        "initialize",
        Some(json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "boot-no-network", "version": "1" },
        })),
    );
    assert_eq!(initialized["serverInfo"]["name"], json!("vestige"));
    session.notify("notifications/initialized");

    let listed = session.result("tools/list", None);
    let names: Vec<&str> = listed["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools/list has no tools array: {listed}"))
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect();
    assert!(
        names.contains(&"memory_status"),
        "tools/list missing memory_status: {names:?}"
    );
    assert!(
        !names.contains(&"source_sync"),
        "source_sync is absent unless --features connectors: {names:?}"
    );
    assert_eq!(names.len(), 17, "default tools/list: {names:?}");

    let called = session.result(
        "tools/call",
        Some(json!({
            "name": "memory_status",
            "arguments": { "view": "tools" },
        })),
    );
    assert_ne!(
        called["isError"],
        json!(true),
        "memory_status failed: {called}"
    );
    let body = tool_body(&called);
    assert_eq!(
        body["compiledFeatures"]["cloudSync"],
        json!(false),
        "4.0 default features must not compile cloud-sync: {body}"
    );
    assert_eq!(
        body["compiledFeatures"]["connectors"],
        json!(false),
        "4.0 default features must not compile connectors: {body}"
    );

    let unknown = session.exchange(
        "tools/call",
        Some(json!({
            "name": "source_sync",
            "arguments": { "source": "gitlab", "repo": "a/b" },
        })),
    );
    assert!(
        unknown.get("result").is_none(),
        "source_sync must not produce a result body: {unknown}"
    );
    assert_eq!(unknown["error"]["code"], json!(-32602), "{unknown}");
    let message = unknown["error"]["message"].as_str().unwrap_or("");
    assert!(
        message.contains("Unknown tool") && message.contains("source_sync"),
        "{unknown}"
    );

    for (label, value) in [
        ("initialize", &initialized),
        ("tools/list", &listed),
        ("tools/call", &called),
    ] {
        assert_no_update_surface(label, &value.to_string());
    }

    // Stay up past the old registry timeout so a reintroduced background
    // lookup cannot exit before strace records it.
    std::thread::sleep(POST_CALL_WINDOW);
    session.drain_notifications();
    for note in &session.notifications {
        assert_no_update_surface("notification", &note.to_string());
    }

    session.close_stdin();
    let status = guard.wait_exit();
    assert!(
        status.success(),
        "stdio EOF must be a clean exit, got {status}. stderr: {:?}",
        session.stderr_lines()
    );

    let trace = std::fs::read_to_string(&trace_path)
        .unwrap_or_else(|error| panic!("read strace log {}: {error}", trace_path.display()));
    assert!(
        trace.contains("+++ exited"),
        "strace wrote no exit record; the trace was not captured:\n{trace}"
    );
    let outbound: Vec<&str> = trace
        .lines()
        .filter(|line| is_outbound_syscall(line))
        .collect();
    assert!(
        outbound.is_empty(),
        "boot opened an outbound socket or DNS lookup:\n{}",
        outbound.join("\n")
    );
}

#[cfg(target_os = "linux")]
fn assert_no_update_surface(label: &str, text: &str) {
    // Split so crate sources do not contain the removed host or function name.
    let registry = concat!("registry.", "npmjs.org");
    let function = concat!("latest_", "npm_version");
    for needle in [
        registry,
        function,
        "newer_version_available",
        "vestige.update",
        "updateAvailable",
        "update_available",
    ] {
        assert!(
            !text.contains(needle),
            "{label} still surfaces {needle}: {text}"
        );
    }
}

#[cfg(target_os = "linux")]
fn tool_body(result: &Value) -> Value {
    if let Some(structured) = result.get("structuredContent") {
        return structured.clone();
    }
    let text = result["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("tool returned no text: {result}"));
    serde_json::from_str(text)
        .unwrap_or_else(|error| panic!("tool text is not JSON ({error}): {text}"))
}

#[cfg(target_os = "linux")]
struct KillGroup {
    child: Option<Child>,
    pid: u32,
}

#[cfg(target_os = "linux")]
impl KillGroup {
    fn new(child: Child) -> Self {
        Self {
            pid: child.id(),
            child: Some(child),
        }
    }

    fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("child still running")
    }

    fn wait_exit(&mut self) -> std::process::ExitStatus {
        let deadline = std::time::Instant::now() + Duration::from_secs(45);
        loop {
            let status = self
                .child
                .as_mut()
                .expect("child still running")
                .try_wait()
                .expect("poll strace");
            if let Some(status) = status {
                self.child.take();
                return status;
            }
            if std::time::Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("strace did not exit within 45s");
    }
}

#[cfg(target_os = "linux")]
impl Drop for KillGroup {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{}", self.pid)])
                .status();
            if let Some(child) = self.child.as_mut() {
                let _ = child.wait();
            }
        }
    }
}

#[cfg(target_os = "linux")]
struct Session {
    stdin: Option<ChildStdin>,
    stdout: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
    notifications: Vec<Value>,
}

#[cfg(target_os = "linux")]
impl Session {
    fn from_child(child: &mut Child) -> Self {
        let stdin = child.stdin.take().expect("stdin");
        let raw_stdout = child.stdout.take().expect("stdout");
        let raw_stderr = child.stderr.take().expect("stderr");
        let (tx, stdout) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(raw_stdout).lines() {
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
                for line in BufReader::new(raw_stderr).lines().map_while(Result::ok) {
                    if let Ok(mut sink) = sink.lock() {
                        sink.push(line);
                    }
                }
            });
        }
        Self {
            stdin: Some(stdin),
            stdout,
            stderr,
            next_id: 0,
            notifications: Vec::new(),
        }
    }

    fn stderr_lines(&self) -> Vec<String> {
        self.stderr.lock().expect("stderr").clone()
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn write_line(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin open");
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| {
                panic!(
                    "write to vestige-mcp failed ({error}). stderr: {:?}",
                    self.stderr_lines()
                )
            });
    }

    fn read_line(&mut self) -> String {
        match self.stdout.recv_timeout(RPC_TIMEOUT) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => panic!(
                "no response within {RPC_TIMEOUT:?}. stderr: {:?}",
                self.stderr_lines()
            ),
            Err(RecvTimeoutError::Disconnected) => {
                panic!("stdout closed. stderr: {:?}", self.stderr_lines())
            }
        }
    }

    fn notify(&mut self, method: &str) {
        self.write_line(&json!({ "jsonrpc": "2.0", "method": method }).to_string());
    }

    fn exchange(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let mut message = json!({ "jsonrpc": "2.0", "id": id, "method": method });
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_line(&message.to_string());
        loop {
            let line = self.read_line();
            let value: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("non-JSON from vestige-mcp ({error}): {line}"));
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return value;
            }
            if value.get("id").is_none() && value.get("method").is_some() {
                self.notifications.push(value);
                continue;
            }
            panic!("unexpected stdout line: {line}");
        }
    }

    fn result(&mut self, method: &str, params: Option<Value>) -> Value {
        let value = self.exchange(method, params);
        assert!(
            value.get("error").is_none(),
            "{method} returned an error: {value}. stderr: {:?}",
            self.stderr_lines()
        );
        value["result"].clone()
    }

    fn drain_notifications(&mut self) {
        while let Ok(line) = self.stdout.try_recv() {
            if let Ok(value) = serde_json::from_str::<Value>(&line)
                && value.get("id").is_none()
                && value.get("method").is_some()
            {
                self.notifications.push(value);
            }
        }
    }
}

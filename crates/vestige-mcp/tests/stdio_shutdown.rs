//! Regression coverage for IDEs that close an MCP process's stderr before
//! closing stdin (for example, Zed restarting a stdio server).

#[cfg(unix)]
#[test]
fn stdio_server_exits_cleanly_when_stderr_is_already_closed() {
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::{Duration, Instant};

    let data_dir = tempfile::tempdir().expect("temporary Vestige data directory");
    let mut child = Command::new("sh")
        // Closing fd 2 in the child reproduces the shutdown order reported by
        // Zed without mutating this test process's stderr.
        .arg("-c")
        .arg("exec 2>&-; exec \"$1\"")
        .arg("vestige-mcp-closed-stderr")
        .arg(env!("CARGO_BIN_EXE_vestige-mcp"))
        .env("VESTIGE_DATA_DIR", data_dir.path())
        .env("VESTIGE_DASHBOARD_ENABLED", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .expect("start vestige-mcp with stderr closed");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().expect("poll vestige-mcp") {
            assert!(
                status.success(),
                "closed stderr plus stdin EOF must be a clean shutdown, got {status}"
            );
            return;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("vestige-mcp did not exit after stdin EOF");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

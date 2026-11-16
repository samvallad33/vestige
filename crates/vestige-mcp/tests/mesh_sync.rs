//! Two data directories pair, sync over loopback iroh, and share a mesh digest.
//!
//! SIGKILL variants kill the receiver, then the sender, after a seeded number
//! of durable acks. Every acked `(seq, frame hash)` is still in the replica
//! after restart, and the resumed sync reaches one digest.

use std::fs::{self, File};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

fn vestige(data_dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env("NO_COLOR", "1")
        .env("VESTIGE_MESH_TEST_CONFIRM", "1")
        .env_remove("VESTIGE_DATA_DIR")
        .env_remove("VESTIGE_MESH_PACE_MS")
        .output()
        .expect("spawn vestige")
}

fn spawn(data_dir: &Path, args: &[&str], pace_ms: Option<u64>) -> Child {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vestige"));
    command
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env("NO_COLOR", "1")
        .env("VESTIGE_MESH_TEST_CONFIRM", "1")
        .env_remove("VESTIGE_DATA_DIR")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match pace_ms {
        Some(ms) => {
            command.env("VESTIGE_MESH_PACE_MS", ms.to_string());
        }
        None => {
            command.env_remove("VESTIGE_MESH_PACE_MS");
        }
    }
    command.spawn().expect("spawn vestige")
}

fn ingest(dir: &Path, content: &str) {
    let output = vestige(dir, &["ingest", content]);
    assert!(
        output.status.success(),
        "ingest failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn status(dir: &Path) -> Value {
    let output = vestige(dir, &["mesh", "status"]);
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("status json")
}

fn digest(dir: &Path) -> String {
    status(dir)["mesh_digest"]
        .as_str()
        .expect("digest")
        .to_string()
}

fn pair(left: &Path, right: &Path) {
    let scratch = tempfile::tempdir().unwrap();
    let ticket = scratch.path().join("ticket");
    let response = scratch.path().join("response");
    let offer_out = scratch.path().join("pair.out");
    let mut offering = spawn_to(
        left,
        &[
            "mesh",
            "pair",
            "--code",
            "mesh-test",
            "--ticket-file",
            ticket.to_str().unwrap(),
            "--wait-file",
            response.to_str().unwrap(),
        ],
        &offer_out,
    );
    let ticket_text = wait_file(&ticket, Duration::from_secs(30));
    assert!(ticket_text.starts_with("vm1."), "{ticket_text}");
    let joining = vestige(
        right,
        &[
            "mesh",
            "join",
            "--code",
            "mesh-test",
            "--ticket-file",
            ticket.to_str().unwrap(),
            "--response-file",
            response.to_str().unwrap(),
        ],
    );
    assert!(
        joining.status.success(),
        "join failed: {}",
        String::from_utf8_lossy(&joining.stderr)
    );
    let joined: Value = serde_json::from_slice(&joining.stdout).expect("join json");
    let offer_status = wait_child(&mut offering, Duration::from_secs(30));
    let offer_text = fs::read_to_string(&offer_out).unwrap_or_default();
    assert!(offer_status.success(), "pair failed: {offer_text}");
    let offered: Value = serde_json::from_slice(offer_text.as_bytes()).expect("pair json");
    assert_eq!(offered["sas"], joined["sas"]);
    assert!(offered["qr"].as_str().is_some_and(|qr| !qr.is_empty()));
}

fn sync_both(left: &Path, right: &Path) {
    let mut listening = spawn(right, &["mesh", "sync", "--listen", "--once"], None);
    let dial = vestige(left, &["mesh", "sync"]);
    let listen_status = wait_child(&mut listening, Duration::from_secs(60));
    assert!(
        dial.status.success(),
        "dial failed: {}",
        String::from_utf8_lossy(&dial.stderr)
    );
    assert!(listen_status.success(), "listen failed");
    let dialed: Value = serde_json::from_slice(&dial.stdout).unwrap();
    assert!(dialed["merges"].as_array().is_some_and(|rows| !rows.is_empty()));
}

fn wait_file(path: &Path, limit: Duration) -> String {
    let start = Instant::now();
    loop {
        if let Ok(text) = fs::read_to_string(path) {
            if !text.trim().is_empty() {
                return text.trim().to_string();
            }
        }
        if start.elapsed() > limit {
            panic!("timed out waiting for {}", path.display());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn wait_child(child: &mut Child, limit: Duration) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("child did not exit");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn spawn_to(data_dir: &Path, args: &[&str], stdout: &Path) -> Child {
    let file = File::create(stdout).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_vestige"));
    command
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .env("NO_COLOR", "1")
        .env("VESTIGE_MESH_TEST_CONFIRM", "1")
        .env_remove("VESTIGE_DATA_DIR")
        .env_remove("VESTIGE_MESH_PACE_MS")
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::inherit());
    command.spawn().expect("spawn vestige")
}

fn ack_lines(path: &Path) -> Vec<(u64, String)> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let (seq, hash) = line.split_once(' ')?;
            Some((seq.parse().ok()?, hash.to_string()))
        })
        .collect()
}

fn kill(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn two_devices_pair_over_loopback_and_share_a_mesh_digest() {
    let left_dir = tempfile::tempdir().unwrap();
    let right_dir = tempfile::tempdir().unwrap();
    ingest(left_dir.path(), "alpha keeps a private fact");
    ingest(right_dir.path(), "beta keeps a private fact");
    ingest(left_dir.path(), "the shared sentence");
    ingest(right_dir.path(), "the shared sentence");
    assert_ne!(digest(left_dir.path()), digest(right_dir.path()));

    pair(left_dir.path(), right_dir.path());
    sync_both(left_dir.path(), right_dir.path());

    let left = status(left_dir.path());
    let right = status(right_dir.path());
    assert_eq!(left["mesh_digest"], right["mesh_digest"]);
    assert_eq!(left["relay"], "disabled");
    assert_eq!(right["discovery"], "off");
    assert!(left["merges"].as_array().unwrap().iter().any(|row| {
        row["receipt"].as_str().unwrap().starts_with("eff-")
    }));
    assert!(!left_dir.path().join("mesh").join("chains").join("strata.key").exists());
    assert!(
        fs::metadata(left_dir.path().join("mesh/transport.key"))
            .unwrap()
            .len()
            == 32
    );
}

#[test]
fn sigkill_receiver_then_sender_keeps_acked_frames() {
    let left_dir = tempfile::tempdir().unwrap();
    let right_dir = tempfile::tempdir().unwrap();
    for n in 0..24 {
        ingest(left_dir.path(), &format!("left frame body {n}"));
        ingest(right_dir.path(), &format!("right frame body {n}"));
    }
    pair(left_dir.path(), right_dir.path());

    kill_during_sync(left_dir.path(), right_dir.path(), true);
    let acked = ack_lines(&right_dir.path().join("mesh/acks.log"));
    assert!(
        acked.len() >= 2,
        "receiver recorded too few durable acks: {acked:?}"
    );
    assert_acked_frames_survive(right_dir.path(), &acked);

    let mut listening = spawn(
        right_dir.path(),
        &["mesh", "sync", "--listen", "--once"],
        None,
    );
    let dial = vestige(left_dir.path(), &["mesh", "sync"]);
    assert!(wait_child(&mut listening, Duration::from_secs(90)).success());
    assert!(
        dial.status.success(),
        "resume dial failed: {}",
        String::from_utf8_lossy(&dial.stderr)
    );
    assert_eq!(digest(left_dir.path()), digest(right_dir.path()));

    // A second exchange is receipt-only. Kill the sender after the receiver
    // has acked, then finish again. The digest does not move.
    let settled = digest(left_dir.path());
    let mut listening = spawn(
        right_dir.path(),
        &["mesh", "sync", "--listen", "--once"],
        Some(40),
    );
    let mut dialing = spawn(left_dir.path(), &["mesh", "sync"], Some(40));
    let start = Instant::now();
    let before = ack_lines(&right_dir.path().join("mesh/acks.log")).len();
    while ack_lines(&right_dir.path().join("mesh/acks.log")).len() <= before
        && start.elapsed() < Duration::from_secs(20)
    {
        if dialing.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(15));
    }
    kill(&mut dialing);
    kill(&mut listening);
    let mut listening = spawn(
        right_dir.path(),
        &["mesh", "sync", "--listen", "--once"],
        None,
    );
    let dial = vestige(left_dir.path(), &["mesh", "sync"]);
    assert!(wait_child(&mut listening, Duration::from_secs(60)).success());
    assert!(dial.status.success(), "{}", String::from_utf8_lossy(&dial.stderr));
    assert_eq!(digest(left_dir.path()), settled);
    assert_eq!(digest(right_dir.path()), settled);
}

fn kill_during_sync(left: &Path, right: &Path, kill_receiver: bool) {
    let _ = fs::remove_file(right.join("mesh/acks.log"));
    let mut listening = spawn(right, &["mesh", "sync", "--listen", "--once"], Some(30));
    let mut dialing = spawn(left, &["mesh", "sync"], Some(30));
    let start = Instant::now();
    let mut saw = false;
    while start.elapsed() < Duration::from_secs(40) {
        if ack_lines(&right.join("mesh/acks.log")).len() >= 2 {
            saw = true;
            break;
        }
        if listening.try_wait().unwrap().is_some() || dialing.try_wait().unwrap().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(saw, "sync finished before two durable acks");
    if kill_receiver {
        kill(&mut listening);
        kill(&mut dialing);
    } else {
        kill(&mut dialing);
        kill(&mut listening);
    }
}

fn assert_acked_frames_survive(data: &Path, acked: &[(u64, String)]) {
    let peers: Vec<Value> =
        serde_json::from_slice(&fs::read(data.join("mesh/peers.json")).unwrap()).unwrap();
    let hex = peers[0]["verifying_key_hex"].as_str().unwrap();
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    let store = strata_store::StrataStore::open(data).expect("reopen after kill");
    let head = store
        .replica_head(key)
        .expect("replica verifies")
        .expect("replica has frames");
    for (seq, hash) in acked {
        assert!(*seq <= head.0, "acked seq {seq} is past the replica head");
        if *seq == head.0 {
            assert_eq!(hex_encode(&head.1), *hash, "acked head hash changed");
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

#[cfg(target_os = "linux")]
#[test]
fn loopback_sync_makes_no_dns_lookup() {
    let left_dir = tempfile::tempdir().unwrap();
    let right_dir = tempfile::tempdir().unwrap();
    ingest(left_dir.path(), "trace left");
    ingest(right_dir.path(), "trace right");
    pair(left_dir.path(), right_dir.path());
    let peer_port = bind_port(left_dir.path());
    let own_port = bind_port(right_dir.path());

    let trace = tempfile::NamedTempFile::new().unwrap();
    let mut listening = spawn(right_dir.path(), &["mesh", "sync", "--listen", "--once"], None);
    let mut command = Command::new("strace");
    command
        .arg("-f")
        .arg("-s")
        .arg("128")
        .arg("-e")
        .arg("trace=network,connect")
        .arg("-o")
        .arg(trace.path())
        .arg("--")
        .arg(env!("CARGO_BIN_EXE_vestige"))
        .arg("--data-dir")
        .arg(left_dir.path())
        .arg("mesh")
        .arg("sync")
        .env("NO_COLOR", "1")
        .env("VESTIGE_MESH_TEST_CONFIRM", "1")
        .env_remove("VESTIGE_DATA_DIR");
    let dial = command.output().expect("strace");
    assert!(wait_child(&mut listening, Duration::from_secs(60)).success());
    assert!(dial.status.success(), "{}", String::from_utf8_lossy(&dial.stderr));
    let lines = fs::read_to_string(trace.path()).unwrap();
    for line in lines.lines() {
        if is_dns(line) {
            panic!("mesh sync did a DNS lookup: {line}");
        }
        if is_payload_send(line) && mentions_non_loopback(line) {
            panic!("mesh sync sent bytes off loopback: {line}");
        }
        if is_foreign_inet(line, peer_port, own_port) && !is_source_selection_probe(line) {
            panic!("mesh sync connected somewhere other than the paired peer: {line}");
        }
    }
}

fn bind_port(dir: &Path) -> u16 {
    let text = fs::read_to_string(dir.join("mesh/bind.txt")).unwrap();
    text.trim()
        .rsplit(':')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn is_dns(line: &str) -> bool {
    let hay = line.to_ascii_lowercase();
    hay.contains("nscd") || hay.contains("htons(53)") || hay.contains("io.systemd.resolve")
}

/// `netdev` (pulled in by iroh's netmon, `gateway` feature on by default)
/// UDP-connects to a non-routable address so the kernel names the source
/// address of the default route. That `connect` transmits nothing. A later
/// `sendto` / `sendmsg` to the same address still fails the test.
fn is_source_selection_probe(line: &str) -> bool {
    if !line.contains("connect(") || !line.contains("htons(1)") {
        return false;
    }
    line.contains("10.254.254.254")
        || line.contains("fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")
}

fn is_payload_send(line: &str) -> bool {
    line.contains("sendto(") || line.contains("sendmsg(") || line.contains("sendmmsg(")
}

fn mentions_non_loopback(line: &str) -> bool {
    if !line.contains("sin_addr") && !line.contains("sin6_addr") {
        return false;
    }
    !line.contains("127.0.0.1") && !line.contains("::1")
}

fn is_foreign_inet(line: &str, peer_port: u16, own_port: u16) -> bool {
    if !line.contains("AF_INET") {
        return false;
    }
    if line.contains("AF_INET6") && !line.contains("127.0.0.1") && line.contains("connect(") {
        return true;
    }
    if line.contains("connect(") && line.contains("sin_addr") && !line.contains("127.0.0.1") {
        return true;
    }
    if line.contains("connect(") && line.contains("127.0.0.1") {
        let peer = format!("htons({peer_port})");
        let own = format!("htons({own_port})");
        return !line.contains(&peer) && !line.contains(&own);
    }
    false
}


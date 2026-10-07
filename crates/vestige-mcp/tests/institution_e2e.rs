//! THE INSTITUTION end-to-end: spawns the real `vestige` binary and drives it
//! through host-shaped stdin JSON, with a sandboxed HOME and a dead dashboard
//! port so behavior is fully deterministic. State files are inspected on disk.

use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_vestige")
}

struct Sandbox {
    home: PathBuf,
    session: String,
}

impl Sandbox {
    fn new(name: &str) -> Self {
        let home = std::env::temp_dir().join(format!("inst-e2e-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        Sandbox {
            home,
            session: format!("e2e-{name}"),
        }
    }

    fn run(&self, payload: &serde_json::Value) -> String {
        let mut child = Command::new(bin())
            .arg("hook")
            .env("HOME", &self.home)
            .env("VESTIGE_API", "http://127.0.0.1:1")
            .env("VESTIGE_HOOK_MAX_GAP", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn vestige hook");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(payload.to_string().as_bytes())
            .unwrap();
        String::from_utf8_lossy(&child.wait_with_output().unwrap().stdout).to_string()
    }

    fn pre(&self, tool: &str, command: &str) -> String {
        self.run(&serde_json::json!({
            "hook_event_name": "PreToolUse", "session_id": self.session,
            "cwd": "/tmp", "tool_name": tool, "tool_input": {"command": command}
        }))
    }

    fn post(&self, command: &str, code: i64, failed_event: bool) -> String {
        let mut p = serde_json::json!({
            "hook_event_name": if failed_event { "PostToolUseFailure" } else { "PostToolUse" },
            "session_id": self.session, "cwd": "/tmp",
            "tool_name": "Bash", "tool_input": {"command": command}, "tool_exit_code": code
        });
        if failed_event {
            p["error"] = serde_json::json!("spawn failed");
        }
        self.run(&p)
    }

    fn vestige(&self, tool: &str, input: serde_json::Value) {
        let _ = self.run(&serde_json::json!({
            "hook_event_name": "PreToolUse", "session_id": self.session, "cwd": "/tmp",
            "tool_name": format!("mcp__vestige__{tool}"), "tool_input": input
        }));
    }
}

#[test]
fn e2e_fail_open_garbage_and_unknown_event() {
    let sb = Sandbox::new("failopen");
    for junk in ["not json", "", "{\"hook_event_name\":\"Nonsense\"}"] {
        let mut child = Command::new(bin())
            .arg("hook")
            .env("HOME", &sb.home)
            .env("VESTIGE_API", "http://127.0.0.1:1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(junk.as_bytes())
            .unwrap();
        let o = child.wait_with_output().unwrap();
        assert_eq!(o.status.code(), Some(0));
        assert!(o.stdout.is_empty(), "garbage produces no output");
    }
}

#[test]
fn e2e_session_start_emits_contract_and_writes_state() {
    let sb = Sandbox::new("ss");
    let out = sb.run(&serde_json::json!({
        "hook_event_name": "SessionStart", "session_id": sb.session, "cwd": "/tmp", "source": "startup"
    }));
    assert!(out.contains("VESTIGE MEMORY IS ENFORCED"));
    assert!(out.contains("did not answer"));
    let st = sb
        .home
        .join(".vestige/hooks")
        .join(format!("institution-{}.json", sb.session));
    assert!(st.exists(), "state file persisted");
}

#[test]
fn e2e_timeout_card_full_cycle() {
    let sb = Sandbox::new("card");
    let first = sb.pre("Bash", "git push --force origin main");
    assert!(first.contains("TIME-OUT CARD") && first.contains("blast radius"));
    let second = sb.pre("Bash", "git push --force origin main");
    assert!(!second.contains("TIME-OUT CARD"), "repeat passes");
    let lease = sb.pre("Bash", "git push --force-with-lease origin main");
    assert!(!lease.contains("TIME-OUT CARD"), "lease exempt");
}

#[test]
fn e2e_search_gate_recall_satisfies() {
    let sb = Sandbox::new("search");
    let denied = sb.pre("Bash", "rg pattern src/");
    assert!(denied.contains("SEARCH-BY-ACT"));
    sb.vestige("recall", serde_json::json!({"handle": "institution"}));
    let allowed = sb.pre("Bash", "rg pattern src/");
    assert!(!allowed.contains("SEARCH-BY-ACT"), "recall satisfies");
    sb.vestige("session_start", serde_json::json!({}));
    let denied2 = sb.pre("Bash", "grep -rn TODO .");
    assert!(
        denied2.contains("SEARCH-BY-ACT"),
        "session_start does not satisfy"
    );
}

#[test]
fn e2e_foqa_secret_exceedance_on_third_touch() {
    let sb = Sandbox::new("foqa");
    assert!(!sb.post("cat ~/.ssh/kaggle.json", 0, false).contains("FOQA"));
    assert!(!sb.post("cat ~/.ssh/kaggle.json", 0, false).contains("FOQA"));
    let third = sb.post("cat ~/.ssh/kaggle.json", 0, false);
    assert!(third.contains("FOQA") && third.contains("secret-adjacent"));
    let fourth = sb.post("cat ~/.ssh/kaggle.json", 0, false);
    assert!(!fourth.contains("secret-adjacent"), "filed once");
}

#[test]
fn e2e_ripple_on_success_then_failure() {
    let sb = Sandbox::new("ripple");
    assert!(!sb.post("make build", 0, false).contains("RIPPLE"));
    assert!(sb.post("make build", 1, false).contains("RIPPLE TAG"));
}

#[test]
fn e2e_drawdown_postmortem_gate_and_lift() {
    let sb = Sandbox::new("dd");
    sb.vestige("session_start", serde_json::json!({}));
    for _ in 0..5 {
        let _ = sb.post("make all", 2, false);
    }
    let denied = sb.pre("Bash", "make all");
    assert!(
        denied.contains("DRAWDOWN LIMIT"),
        "post-mortem demanded: {denied}"
    );
    sb.vestige(
        "smart_ingest",
        serde_json::json!({"content": "post-mortem"}),
    );
    let allowed = sb.pre("Bash", "make all");
    assert!(!allowed.contains("DRAWDOWN"), "write lifts");
}

#[test]
fn e2e_stop_save_guard_blocks_then_asks_once() {
    let sb = Sandbox::new("stop");
    sb.vestige("session_start", serde_json::json!({}));
    for c in ["touch /tmp/e2e-a", "touch /tmp/e2e-b", "touch /tmp/e2e-c"] {
        let o = sb.pre("Bash", c);
        assert!(!o.contains("permissionDecision"), "offline pass: {c}");
    }
    let stop = sb.run(
        &serde_json::json!({"hook_event_name": "Stop", "session_id": sb.session, "cwd": "/tmp"}),
    );
    assert!(
        stop.contains("saved nothing to Vestige"),
        "guard fires: {stop}"
    );
    let stop2 = sb.run(
        &serde_json::json!({"hook_event_name": "Stop", "session_id": sb.session, "cwd": "/tmp"}),
    );
    assert!(!stop2.contains("saved nothing"), "asked once");
}

#[test]
fn e2e_stop_work_from_seeded_registry_and_override_receipt() {
    let sb = Sandbox::new("sw");
    let dir = sb.home.join(".vestige/hooks");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("institution-{}.json", sb.session)),
        serde_json::json!({"started": true, "stopwork": [
            {"id": "mem-e2e", "pattern": "kubectl\\s+delete\\s+namespace\\s+prod", "reason": "prod ns deletion forbidden"}
        ]}).to_string()).unwrap();
    let denied = sb.pre("Bash", "kubectl delete namespace prod --now");
    assert!(denied.contains("STOP-WORK AUTHORITY") && denied.contains("mem-e2e"));
    let over = "OVERRIDE-STOPWORK kubectl delete namespace prod --now";
    let _ = sb.pre("Bash", over);
    let next = sb.post("echo after", 0, false);
    assert!(
        next.contains("STOP-WORK OVERRIDE"),
        "receipt surfaced: {next}"
    );
}

#[test]
fn e2e_prompt_triggers_surface_laws() {
    let sb = Sandbox::new("prompt");
    let out = sb.run(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit", "session_id": sb.session, "cwd": "/tmp",
        "prompt": "that was a near miss, log it"
    }));
    assert!(out.contains("NEAR-MISS"));
    let out2 = sb.run(&serde_json::json!({
        "hook_event_name": "UserPromptSubmit", "session_id": sb.session, "cwd": "/tmp",
        "prompt": "ok deploy it"
    }));
    assert!(out2.contains("PREREGISTER"));
}

#[test]
fn e2e_hooks_install_rewrites_zcode_config_with_backup() {
    let sb = Sandbox::new("install");
    let zdir = sb.home.join(".zcode/cli");
    std::fs::create_dir_all(&zdir).unwrap();
    std::fs::write(zdir.join("config.json"), serde_json::json!({
        "hooks": {"enabled": false, "events": {"PreToolUse": [{"hooks": [
            {"type": "command", "command": "/usr/bin/python3 /x/vestige-memory.py --host zcode", "timeout": 10}
        ]}]}},
        "mcp": {"keep": true}
    }).to_string()).unwrap();
    let out = Command::new(bin())
        .arg("hooks")
        .arg("install")
        .arg("zcode")
        .env("HOME", &sb.home)
        .output()
        .unwrap();
    assert!(out.status.success());
    let rw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(zdir.join("config.json")).unwrap()).unwrap();
    assert_eq!(rw["hooks"]["enabled"], serde_json::json!(true));
    let cmd = rw["hooks"]["events"]["PreToolUse"][0]["hooks"][0]["command"]
        .as_str()
        .unwrap();
    assert!(cmd.ends_with("vestige hook") && !cmd.contains("vestige-memory.py"));
    assert_eq!(rw["mcp"]["keep"], serde_json::json!(true));
    assert!(zdir.join("config.json.bak-institution").exists());
}

#[test]
fn e2e_hooks_install_unknown_host_fails_cleanly() {
    let sb = Sandbox::new("badhost");
    let out = Command::new(bin())
        .arg("hooks")
        .arg("install")
        .arg("plan9")
        .env("HOME", &sb.home)
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn e2e_denial_ledger_written_and_graduation_promotes() {
    let s = Sandbox::new("ledger");
    // three separate sessions each deny the same force-push once
    for i in 1..=3 {
        let sid = format!("{}-{i}", s.session);
        s.run(&serde_json::json!({"hook_event_name":"SessionStart","session_id":sid,"cwd":"/tmp","source":"startup"}));
        let out = s.run(
            &serde_json::json!({"hook_event_name":"PreToolUse","session_id":sid,"cwd":"/tmp",
            "tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}),
        );
        assert!(out.contains("TIME-OUT CARD"), "session {i} denies: {out}");
    }
    let ledger = std::fs::read_to_string(s.home.join(".vestige/hooks/denials.jsonl"))
        .expect("three denied force-pushes must leave three ledger lines");
    assert_eq!(ledger.lines().count(), 3, "ledger: {ledger}");
    assert!(ledger.contains("\"gate\":\"timeout-card\""));
    // the fourth session's start surfaces the graduation prompt
    let start = s.run(&serde_json::json!({"hook_event_name":"SessionStart","session_id":format!("{}-4", s.session),"cwd":"/tmp","source":"startup"}));
    assert!(
        start.contains("GRADUATE TO LAW"),
        "three same-shape denials graduate: {start}"
    );
}

#[test]
fn e2e_master_off_switch_silences_everything() {
    let s = Sandbox::new("offsw");
    std::fs::create_dir_all(s.home.join(".claude/hooks"))
        .expect("sandbox off-switch dir must be creatable");
    std::fs::write(s.home.join(".claude/hooks/VESTIGE_MEMORY_OFF"), b"")
        .expect("off-switch touch file must be writable");
    let out = s.run(
        &serde_json::json!({"hook_event_name":"PreToolUse","session_id":"off","cwd":"/tmp",
        "tool_name":"Bash","tool_input":{"command":"git push --force origin main"}}),
    );
    assert_eq!(
        out.trim(),
        "",
        "off switch silences the hook entirely: {out}"
    );
    let start = s.run(&serde_json::json!({"hook_event_name":"SessionStart","session_id":"off","cwd":"/tmp","source":"startup"}));
    assert_eq!(start.trim(), "");
}

#[test]
fn e2e_pivot_gate_requires_search_before_family_switch() {
    let s = Sandbox::new("pivot");
    s.run(&serde_json::json!({"hook_event_name":"SessionStart","session_id":s.session,"cwd":"/tmp","source":"startup"}));
    // a failure in one command family...
    s.post("cargo build --release", 1, true);
    // ...then a DIFFERENT family with no external search is denied once
    let denied = s.pre("Bash", "npm install left-pad");
    assert!(
        denied.contains("PIVOT GATE"),
        "pivot without search denied: {denied}"
    );
    // a WebSearch tool call is the reality check that clears the requirement
    let _ = s.run(
        &serde_json::json!({"hook_event_name":"PreToolUse","session_id":s.session,"cwd":"/tmp",
        "tool_name":"WebSearch","tool_input":{"query":"current best practice for this problem"}}),
    );
    let pass = s.pre("Bash", "npm install left-pad");
    assert!(
        !pass.contains("PIVOT GATE"),
        "after searching, the pivot proceeds: {pass}"
    );
    // and the denial was filed in the ledger under its own gate
    let ledger =
        std::fs::read_to_string(s.home.join(".vestige/hooks/denials.jsonl")).unwrap_or_default();
    assert!(
        ledger.contains("\"gate\":\"pivot\""),
        "pivot denial is flight data: {ledger}"
    );
}

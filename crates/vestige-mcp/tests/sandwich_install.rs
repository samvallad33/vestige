//! Integration tests for #396 (SessionStart command hooks for startup,
//! resume, clear, compact and fork), event-preserving sandwich install, and
//! #520 `vestige coach --replay`.

use std::path::PathBuf;
use std::process::Command;

use serde_json::{Value, json};
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn enable_capsule_wires_a_sessionstart_command_hook() {
    let home = TempDir::new().unwrap();
    let claude = home.path().join(".claude");
    std::fs::create_dir_all(&claude).unwrap();
    let settings = claude.join("settings.json");
    let user = json!({
        "hooks": {
            "SessionStart": [{
                "matcher": "startup",
                "hooks": [{"type": "command", "command": "echo mine", "timeout": 1}]
            }]
        }
    });
    std::fs::write(&settings, serde_json::to_vec_pretty(&user).unwrap()).unwrap();
    let root = repo_root();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_vestige"))
            .env("HOME", home.path())
            .env_remove("VESTIGE_DATA_DIR")
            .args(args)
            .output()
            .unwrap()
    };

    let installed = run(&[
        "sandwich",
        "install",
        "--src",
        root.to_str().unwrap(),
        "--enable-capsule",
    ]);
    assert!(
        installed.status.success(),
        "stdout {}\nstderr {}",
        String::from_utf8_lossy(&installed.stdout),
        String::from_utf8_lossy(&installed.stderr)
    );

    let text = std::fs::read_to_string(&settings).unwrap();
    let value: Value = serde_json::from_str(&text).unwrap();
    let groups = value["hooks"]["SessionStart"].as_array().unwrap();
    assert_eq!(groups[0]["hooks"][0]["command"], "echo mine");
    let capsule = groups
        .iter()
        .find(|group| {
            group["hooks"][0]["command"]
                == "vestige capsule --harness claude-code --event session-start"
        })
        .expect(&text);
    assert_eq!(capsule["matcher"], "startup|resume|clear|compact|fork");
    assert_eq!(capsule["hooks"][0]["type"], "command");

    // Second install must be byte-identical.
    let again = run(&[
        "sandwich",
        "install",
        "--src",
        root.to_str().unwrap(),
        "--enable-capsule",
    ]);
    assert!(
        again.status.success(),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert_eq!(std::fs::read(&settings).unwrap(), text.into_bytes());

    // Re-running without --enable-capsule removes only the Vestige capsule entry.
    let removed = run(&["sandwich", "install", "--src", root.to_str().unwrap()]);
    assert!(
        removed.status.success(),
        "{}",
        String::from_utf8_lossy(&removed.stderr)
    );
    let after: Value = serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
    let left = after["hooks"]["SessionStart"].as_array().unwrap();
    assert!(left.iter().all(|group| {
        group["hooks"][0]["command"]
            != "vestige capsule --harness claude-code --event session-start"
    }));
    assert_eq!(left[0]["hooks"][0]["command"], "echo mine");
}

#[test]
fn sandwich_install_preserves_user_hooks_across_preflight_and_sanhedrin() {
    let home = TempDir::new().unwrap();
    let claude = home.path().join(".claude");
    std::fs::create_dir_all(&claude).unwrap();
    let settings = claude.join("settings.json");
    let user = json!({
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": "echo user-prompt-hook", "timeout": 2}]
            }],
            "Stop": [{
                "hooks": [{"type": "command", "command": "echo user-stop-hook", "timeout": 2}]
            }]
        }
    });
    std::fs::write(&settings, serde_json::to_vec_pretty(&user).unwrap()).unwrap();
    let root = repo_root();
    let installed = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .env("HOME", home.path())
        .env_remove("VESTIGE_DATA_DIR")
        .args([
            "sandwich",
            "install",
            "--src",
            root.to_str().unwrap(),
            "--enable-preflight",
            "--enable-sanhedrin",
        ])
        .output()
        .unwrap();
    assert!(
        installed.status.success(),
        "{}",
        String::from_utf8_lossy(&installed.stderr)
    );

    let after: Value = serde_json::from_str(&std::fs::read_to_string(&settings).unwrap()).unwrap();
    let prompts = after["hooks"]["UserPromptSubmit"].as_array().unwrap();
    assert!(prompts.iter().any(|g| g["hooks"][0]["command"] == "echo user-prompt-hook"));
    assert!(prompts.iter().any(|g| {
        g["hooks"].as_array().is_some_and(|hs| {
            hs.iter()
                .any(|h| h["command"].as_str().is_some_and(|c| c.contains("synthesis-preflight.sh")))
        })
    }));
    let stops = after["hooks"]["Stop"].as_array().unwrap();
    assert!(stops.iter().any(|g| g["hooks"][0]["command"] == "echo user-stop-hook"));
    assert!(stops.iter().any(|g| {
        g["hooks"].as_array().is_some_and(|hs| {
            hs.iter()
                .any(|h| h["command"].as_str().is_some_and(|c| c.contains("veto-detector.sh")))
        })
    }));
}

#[test]
fn coach_replay_cli_reports_prevented_failures_and_saved_chars() {
    let dir = TempDir::new().unwrap();
    let log = dir.path().join("calls.jsonl");
    std::fs::write(
        &log,
        concat!(
            "{\"receipt_id\":\"c1\",\"tool_name\":\"mcp__vestige__curate\",\"tool_input\":{\"action\":\"demote\",\"memory_id\":\"mem-0000000000000001\"}}\n",
            "{\"receipt_id\":\"c2\",\"tool_name\":\"mcp__vestige__recall\",\"tool_input\":{\"query\":\"auth\"}}\n",
            "{\"receipt_id\":\"c3\",\"tool_name\":\"mcp__vestige__curate\",\"tool_input\":{\"action\":\"purge\",\"id\":\"mem-0000000000000001\"}}\n"
        ),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vestige"))
        .args(["coach", "--replay", log.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let parsed: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(parsed["total_calls"], 3);
    let rules = parsed["rules"].as_array().unwrap();
    assert!(rules.iter().any(|r| r["rule"] == "CC-001" && r["prevented_failures"] == 1));
    assert!(rules.iter().any(|r| r["rule"] == "CC-003" && r["saved_chars"].as_u64().unwrap() > 0));
    assert!(rules.iter().any(|r| r["rule"] == "CC-010" && r["prevented_failures"] == 1));
}

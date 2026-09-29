//! Whole-session network ban on the default build.
//!
//! `cloud-sync` and `connectors` are opt-in. The default session does not
//! call `source_sync` and does not run `vestige sync --cloud`. `source_sync`
//! must be absent from `tools/list`. Pass is an empty default `reqwest`
//! inverse tree (`hyper` via the axum dashboard server is allowed) and zero
//! `connect()` or DNS to anything other than `AF_UNIX`.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

use super::support::*;

const HTTP_CRATES: &[&str] = &["reqwest", "hyper", "ureq", "hyper-util"];
const BINS: &[&str] = &["vestige", "vestige-mcp"];

fn clear_network_env() -> Vec<String> {
    let mut names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "TOGETHER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OLLAMA_HOST",
        "GITHUB_TOKEN",
        "VESTIGE_GITHUB_TOKEN",
        "REDMINE_API_KEY",
        "VESTIGE_REDMINE_API_KEY",
        "REDMINE_URL",
        "VESTIGE_REDMINE_URL",
        "VESTIGE_CLOUD_ENDPOINT",
        "VESTIGE_CLOUD_SYNC_KEY",
        "VESTIGE_CLOUD_ENCRYPTION_KEY",
        "VESTIGE_SANHEDRIN_ENABLED",
        "VESTIGE_SANHEDRIN_ENDPOINT",
        "VESTIGE_SANHEDRIN_MODEL",
        "VESTIGE_SANHEDRIN_CLAIM_MODE",
        "VESTIGE_SANHEDRIN_OUTPUT",
        "VESTIGE_SANHEDRIN_STATE_DIR",
        "VESTIGE_SANHEDRIN_API_KEY",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let root = repo_root();
    for rel in ["crates/vestige-mcp/src", "crates/vestige-core/src"] {
        let mut stack = vec![root.join(rel)];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = fs::read_to_string(&path).unwrap_or_default();
                for needle in ["env::var(\"", "env::var_os(\"", "var(\"", "var_os(\""] {
                    let mut rest = text.as_str();
                    while let Some(at) = rest.find(needle) {
                        rest = &rest[at + needle.len()..];
                        let Some(end) = rest.find('"') else { break };
                        let name = &rest[..end];
                        let upper = name.to_ascii_uppercase();
                        if upper.contains("KEY")
                            || upper.contains("TOKEN")
                            || upper.contains("SECRET")
                            || upper.contains("SANHEDRIN")
                            || upper.contains("OPENAI")
                            || upper.contains("ANTHROPIC")
                            || upper.contains("OLLAMA")
                            || upper.contains("CLOUD")
                            || upper.contains("GITHUB")
                            || upper.contains("REDMINE")
                        {
                            names.push(name.to_string());
                        }
                        rest = &rest[end..];
                    }
                }
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// `cargo tree` in this toolchain has no `--bin`. Both binaries are the
/// `vestige-mcp` package under default features, so one inverse tree is the
/// default build of each.
fn cargo_tree(spec: &str) -> String {
    let manifest = repo_root().join("Cargo.toml");
    let out = Command::new("cargo")
        .current_dir(repo_root())
        .args([
            "tree",
            "--manifest-path",
            manifest.to_str().unwrap_or("Cargo.toml"),
            "-p",
            "vestige-mcp",
            "-e",
            "normal",
            "-i",
            spec,
            "--locked",
            "--offline",
        ])
        .output();
    match out {
        Ok(out) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let text = text.trim().to_string();
            if text.is_empty() {
                "(empty)".to_string()
            } else {
                text
            }
        }
        Err(err) => format!("failed to spawn cargo tree: {err}"),
    }
}

fn feature_lines(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .map(str::trim)
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            (lower.contains("reqwest")
                || lower.starts_with("connectors ")
                || lower.starts_with("connectors=")
                || lower.starts_with("cloud-sync ")
                || lower.starts_with("cloud-sync="))
                && !line.starts_with('#')
        })
        .map(|line| format!("{}: {line}", path.display()))
        .collect()
}

fn reqwest_absent(text: &str) -> bool {
    text.contains("did not match any package")
}

fn http_feature_report(trees: &[(String, String)]) -> String {
    let reqwest = trees
        .iter()
        .find(|(spec, _)| spec == "reqwest")
        .map(|(_, text)| text.as_str())
        .unwrap_or("");
    let root = repo_root();
    let mut lines = feature_lines(&root.join("crates/vestige-mcp/Cargo.toml"));
    lines.extend(feature_lines(&root.join("crates/vestige-core/Cargo.toml")));
    format!(
        "default reqwest tree is empty: {}\n\
         hyper and hyper-util may remain through unconditional axum \
         (crates/vestige-mcp/Cargo.toml, the dashboard server).\n\
         connectors and cloud-sync are opt-in; either one forwards to \
         vestige-core `dep:reqwest` (crates/vestige-core/Cargo.toml).\n\
         manifest lines:\n{}",
        reqwest_absent(reqwest),
        if lines.is_empty() {
            "(none)".to_string()
        } else {
            lines.join("\n")
        }
    )
}

fn symbol_hits(bin: &Path) -> String {
    let nm = Command::new("nm")
        .args(["-a"])
        .arg(bin)
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    let mut nm_hits = Vec::new();
    for line in nm.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("reqwest") || lower.contains("hyper") || lower.contains("getaddrinfo") {
            nm_hits.push(line.trim().to_string());
        }
    }
    let strings = Command::new("strings")
        .args(["-a", "-n", "6"])
        .arg(bin)
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    let mut string_hits = Vec::new();
    for line in strings.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("reqwest") || lower.contains("hyper") || lower.contains("getaddrinfo") {
            string_hits.push(line.trim().to_string());
        }
    }
    let sample = |hits: &[String]| -> String {
        if hits.is_empty() {
            return "(none)".to_string();
        }
        let mut shown = hits.iter().take(12).cloned().collect::<Vec<_>>();
        if hits.len() > shown.len() {
            shown.push(format!("… {} more", hits.len() - shown.len()));
        }
        shown.join("\n")
    };
    format!(
        "{} nm hits {} (getaddrinfo/reqwest/hyper)\n{}\nstrings hits {}\n{}",
        bin.display(),
        nm_hits.len(),
        sample(&nm_hits),
        string_hits.len(),
        sample(&string_hits)
    )
}

fn link_report() -> String {
    let mut trees = Vec::new();
    let mut blocks = Vec::new();
    for spec in HTTP_CRATES {
        let text = cargo_tree(spec);
        blocks.push(format!(
            "cargo tree -p vestige-mcp -e normal -i {spec}\n\
             (default features; vestige and vestige-mcp are this package; \
             this cargo has no --bin on `cargo tree`)\n{text}"
        ));
        trees.push(((*spec).to_string(), text));
    }
    let mut symbols = Vec::new();
    for name in BINS {
        symbols.push(symbol_hits(&product_bin(name)));
    }
    format!(
        "{}\n\n{}\n\n{}",
        blocks.join("\n\n"),
        http_feature_report(&trees),
        symbols.join("\n\n")
    )
}

fn namespace_prefix(trace: &Path) -> Vec<String> {
    let ok = |args: &[&str]| {
        Command::new("unshare")
            .args(args)
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    };
    let net = if ok(&["-n", "true"]) {
        vec!["unshare".to_string(), "-n".into(), "--".into()]
    } else if ok(&["-Urn", "true"]) {
        vec!["unshare".to_string(), "-Urn".into(), "--".into()]
    } else {
        panic!(
            "FAIL: no network namespace. `unshare -n` and `unshare -Urn` both failed, \
             so no_network_whole_session cannot prove the offline run."
        );
    };
    // strace is the parent. Inside `unshare -Urn`, PTRACE_TRACEME is EPERM.
    [
        "strace".to_string(),
        "-f".into(),
        "-e".into(),
        "trace=connect,sendto,sendmsg,sendmmsg,recvfrom,recvmsg".into(),
        "-o".into(),
        trace.display().to_string(),
        "-s".into(),
        "160".into(),
        "--".into(),
    ]
    .into_iter()
    .chain(net)
    .collect()
}

fn is_syscall(line: &str) -> bool {
    line.contains("connect(")
        || line.contains("sendto(")
        || line.contains("sendmsg(")
        || line.contains("sendmmsg(")
        || line.contains("recvfrom(")
        || line.contains("recvmsg(")
}

/// `connect()` to anything but `AF_UNIX`, plus DNS send/recv on INET.
fn non_unix_network(trace: &str) -> Vec<String> {
    trace
        .lines()
        .filter(|line| {
            if !is_syscall(line) {
                return false;
            }
            if line.contains("AF_INET") || line.contains("AF_INET6") || line.contains("inet_addr(")
            {
                return true;
            }
            false
        })
        .take(24)
        .map(|line| line.trim().to_string())
        .collect()
}

fn tool_names(listed: &Value) -> Vec<String> {
    listed["result"]["tools"]
        .as_array()
        .or_else(|| listed["tools"].as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| tool["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn memory_handle(id: &str) -> String {
    if id.starts_with("mem-") || id.contains(':') {
        id.to_string()
    } else {
        format!("mem:{id}")
    }
}

/// Schema-valid arguments. `None` means this advertised name has no call yet.
fn tool_args(name: &str, id: &str) -> Option<Value> {
    let handle = memory_handle(id);
    Some(match name {
        "smart_ingest" => json!({
            "content": "no-network-session-canary-7c2e",
            "forceCreate": true
        }),
        "memory" => json!({ "action": "get", "id": id }),
        "memory_status" => json!({ "view": "health" }),
        "purge" => json!({ "id": id }),
        "recall" => json!({ "handle": handle }),
        "receipt" => json!({ "action": "get", "receipt_id": "wr_no_network_session" }),
        "codebase" => json!({ "action": "get_context" }),
        "project" => json!({ "action": "preview" }),
        "intention" => json!({ "action": "list" }),
        "maintain" => json!({ "action": "gc", "dry_run": true }),
        "dedup" => json!({ "action": "scan" }),
        "graph" => json!({ "action": "recent" }),
        "session_start" => json!({ "queries": ["no-network-session-canary-7c2e"] }),
        "suppress" => json!({ "id": id, "reason": "no-network-session" }),
        "causal_walk" => json!({
            "start_points": [{ "kind": "failing_test", "name": "no_network_whole_session" }]
        }),
        "selftest" => json!({}),
        "forgotten_lesson" => json!({ "failure_id": id }),
        _ => return None,
    })
}

#[test]
fn no_network_whole_session() {
    let report = link_report();
    println!("{report}");

    let reqwest_tree = cargo_tree("reqwest");
    let mut problems = Vec::new();
    if !reqwest_absent(&reqwest_tree) {
        problems.push(format!(
            "cargo tree -p vestige-mcp -e normal -i reqwest must print \
             'did not match any package' on the default build. hyper via axum \
             is allowed; reqwest is not.\n{reqwest_tree}"
        ));
    }

    let clear_owned = clear_network_env();
    let clear: Vec<&str> = clear_owned.iter().map(String::as_str).collect();

    let dir = tempfile::tempdir().unwrap();
    let mcp_trace =
        std::env::temp_dir().join(format!("vestige-session-mcp-{}.strace", std::process::id()));
    let _ = fs::remove_file(&mcp_trace);

    let prefix = namespace_prefix(&mcp_trace);
    let mut server = Server::boot_wrapped(dir.path(), &prefix, &clear);

    let listed = match server.call("tools/list", None) {
        Ok(value) => value,
        Err(err) => missing(&format!("tools/list failed: {err}")),
    };
    let names = tool_names(&listed);
    if names.is_empty() {
        problems.push("tools/list advertised no tools".to_string());
    }
    if names.iter().any(|name| name == "source_sync") {
        problems.push(
            "source_sync is in the default tools/list. connectors are opt-in and \
             the default build must not advertise it"
                .to_string(),
        );
    }

    let created = server
        .tool("smart_ingest", tool_args("smart_ingest", "").expect("args"))
        .expect("rpc");
    let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
    if created.rejected() || id.is_empty() {
        problems.push(format!(
            "gated write smart_ingest was not admitted: {}",
            created.blob().chars().take(500).collect::<String>()
        ));
    }
    let id = if id.is_empty() {
        "mem:missing-no-network".to_string()
    } else {
        id
    };

    let mut called = vec!["smart_ingest".to_string()];
    for name in &names {
        if name == "smart_ingest" || name == "source_sync" {
            continue;
        }
        let Some(args) = tool_args(name, &id) else {
            problems.push(format!(
                "tools/list advertised `{name}` and this row has no schema-valid arguments for it"
            ));
            continue;
        };
        match server.tool(name, args) {
            Ok(_) => called.push(name.clone()),
            Err(err) => problems.push(format!("`{name}` rpc failed: {err}")),
        }
    }
    for name in &names {
        if !called.iter().any(|called| called == name) && tool_args(name, &id).is_some() {
            problems.push(format!("`{name}` was listed and not called"));
        }
    }

    let blocked = server
        .tool("memory", json!({ "action": "purge", "id": id }))
        .expect("rpc");
    if !blocked.rejected() {
        problems.push(format!(
            "blocked write (memory purge without confirm) was not rejected: {}",
            blocked.blob().chars().take(500).collect::<String>()
        ));
    }
    if has_gate_receipt(&blocked) {
        problems.push(
            "blocked write returned a gate receipt; purge without confirm must not".to_string(),
        );
    }

    drop(server);

    let mut sockets = Vec::new();
    let text = fs::read_to_string(&mcp_trace).unwrap_or_default();
    if text.is_empty() {
        problems.push(format!(
            "vestige-mcp strace is empty ({})",
            mcp_trace.display()
        ));
    }
    for hit in non_unix_network(&text) {
        sockets.push(format!("vestige-mcp: {hit}"));
    }
    let _ = fs::remove_file(&mcp_trace);
    if !sockets.is_empty() {
        problems.push(format!(
            "a process connected to something other than a local unix socket:\n{}",
            sockets.join("\n")
        ));
    }

    if !problems.is_empty() {
        panic!(
            "FAIL: no_network_whole_session\n{}\n\n{}",
            problems.join("\n"),
            report
        );
    }
}

//! The gate decides with no model and no network.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{json, Value};

use super::support::*;

/// Roots of the decision path. A hit inside one of these functions is a call
/// site that can reach Sanhedrin or an LLM client from the gate.
const DECISION_FNS: &[(&str, &str)] = &[
    (
        "crates/vestige-mcp/src/server.rs",
        "gate_pending_memory_mutation",
    ),
    (
        "crates/vestige-mcp/src/trace_recorder.rs",
        "gate_pending_memory_mutation",
    ),
    ("crates/vestige-mcp/src/trace_recorder.rs", "gate_writes"),
    (
        "crates/vestige-mcp/src/trace_recorder.rs",
        "pending_memory_mutation",
    ),
    ("crates/vestige-mcp/src/trace_recorder.rs", "extract_veto"),
    ("crates/vestige-core/src/trace/review.rs", "classify_write"),
    ("crates/vestige-core/src/trace/review.rs", "collect_signals"),
    ("crates/strata-gate/src/policy.rs", "gate_verdict"),
    ("crates/strata-gate/src/policy.rs", "evaluate_detailed"),
    ("crates/strata-gate/src/policy.rs", "evaluate"),
    ("crates/strata-gate/src/runtime.rs", "commit_gate"),
    ("crates/strata-gate/src/runtime.rs", "commit_propose"),
    ("crates/strata-store/src/store.rs", "admit_write"),
    ("crates/strata-store/src/store.rs", "supersede"),
];

const LLM_NEEDLES: &[&str] = &[
    "sanhedrin",
    "chat/completions",
    "VESTIGE_SANHEDRIN",
    "reqwest",
    "openai",
    "ollama",
    "mlx_lm",
    "vllm",
    "anthropic.com",
];

/// Shipped crates. Tests and the dashboard's display of a receipt the hook
/// already wrote are listed separately from the decision-path hits.
const SHIPPED: &[&str] = &[
    "crates/vestige-mcp/src",
    "crates/vestige-core/src",
    "crates/strata/src",
    "crates/strata-store/src",
    "crates/strata-gate/src",
    "crates/strata-kernel/src",
    "crates/strata-migrate/src",
    "crates/strata-verify/src",
];

fn is_uuid_at(s: &str, i: usize) -> bool {
    let rest = &s[i..];
    if rest.len() < 36 {
        return false;
    }
    let bytes = rest.as_bytes();
    let dashes = [8usize, 13, 18, 23];
    bytes.iter().take(36).enumerate().all(|(n, c)| {
        if dashes.contains(&n) {
            *c == b'-'
        } else {
            c.is_ascii_hexdigit()
        }
    })
}

fn scrub(s: &str) -> String {
    if s.len() >= 20 {
        let b = s.as_bytes();
        if b.len() >= 11
            && b.get(4) == Some(&b'-')
            && b.get(7) == Some(&b'-')
            && b.get(10) == Some(&b'T')
            && b[..4].iter().all(|c| c.is_ascii_digit())
        {
            return "<time>".to_string();
        }
    }
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        if s.is_char_boundary(i) && is_uuid_at(s, i) {
            out.push_str("<id>");
            i += 36;
            continue;
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn scrub_value(value: &Value) -> Value {
    match value {
        Value::String(s) => Value::String(scrub(s)),
        Value::Array(items) => Value::Array(items.iter().map(scrub_value).collect()),
        Value::Object(map) => {
            let mut next = serde_json::Map::new();
            for (k, v) in map {
                next.insert(k.clone(), scrub_value(v));
            }
            Value::Object(next)
        }
        other => other.clone(),
    }
}

fn llm_env_names() -> Vec<String> {
    let mut names = BTreeSet::from([
        "OPENAI_API_KEY".to_string(),
        "ANTHROPIC_API_KEY".to_string(),
        "AZURE_OPENAI_API_KEY".to_string(),
        "GEMINI_API_KEY".to_string(),
        "GROQ_API_KEY".to_string(),
        "MISTRAL_API_KEY".to_string(),
        "COHERE_API_KEY".to_string(),
        "TOGETHER_API_KEY".to_string(),
        "HF_TOKEN".to_string(),
        "HUGGING_FACE_HUB_TOKEN".to_string(),
        "OLLAMA_HOST".to_string(),
        "VESTIGE_SANHEDRIN_ENABLED".to_string(),
        "VESTIGE_SANHEDRIN_ENDPOINT".to_string(),
        "VESTIGE_SANHEDRIN_MODEL".to_string(),
        "VESTIGE_SANHEDRIN_CLAIM_MODE".to_string(),
        "VESTIGE_SANHEDRIN_OUTPUT".to_string(),
        "VESTIGE_SANHEDRIN_STATE_DIR".to_string(),
        "VESTIGE_SANHEDRIN_API_KEY".to_string(),
    ]);
    let root = repo_root();
    for rel in SHIPPED {
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
                            || upper.contains("AZURE")
                            || upper.contains("GEMINI")
                            || upper.contains("MISTRAL")
                            || upper.contains("GROQ")
                            || upper.contains("COHERE")
                            || upper.contains("HUGGING")
                        {
                            names.insert(name.to_string());
                        }
                        rest = &rest[end..];
                    }
                }
            }
        }
    }
    names.into_iter().collect()
}

fn fn_at(text: &str, name: &str) -> Option<usize> {
    let marker = format!("fn {name}");
    let mut from = 0;
    while let Some(at) = text[from..].find(&marker) {
        let abs = from + at;
        let after = abs + marker.len();
        let boundary = match text[after..].chars().next() {
            Some(ch) if ch.is_ascii_alphanumeric() || ch == '_' => false,
            _ => true,
        };
        if boundary {
            return Some(abs);
        }
        from = after;
    }
    None
}

fn fn_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let start = fn_at(text, name)?;
    let after = &text[start + format!("fn {name}").len()..];
    let brace = after.find('{')?;
    let mut depth = 0i32;
    let body = &after[brace..];
    for (i, ch) in body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&body[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn line_of(text: &str, offset: usize) -> usize {
    text[..offset].bytes().filter(|b| *b == b'\n').count() + 1
}

fn decision_path_hits() -> Vec<String> {
    let root = repo_root();
    let mut hits = Vec::new();
    for (rel, name) in DECISION_FNS {
        let path = root.join(rel);
        let text = fs::read_to_string(&path).unwrap_or_default();
        let Some(body) = fn_body(&text, name) else {
            hits.push(format!("{rel}: fn {name} not found"));
            continue;
        };
        let body_at = text.find(body).unwrap_or(0);
        for needle in LLM_NEEDLES {
            let mut rest = body;
            let mut local = 0usize;
            while let Some(at) = rest.to_ascii_lowercase().find(&needle.to_ascii_lowercase()) {
                let abs = body_at + local + at;
                let line = line_of(&text, abs);
                let src = text.lines().nth(line - 1).unwrap_or("").trim();
                hits.push(format!("{rel}:{line} {name} matches `{needle}`: {src}"));
                let step = at + needle.len();
                rest = &rest[step..];
                local += step;
            }
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

fn adjacent_llm_sites() -> Vec<String> {
    let root = repo_root();
    let mut hits = Vec::new();
    for rel in SHIPPED {
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
                for (n, line) in text.lines().enumerate() {
                    let lower = line.to_ascii_lowercase();
                    if lower.contains("sanhedrin")
                        || lower.contains("chat/completions")
                        || lower.contains("vestige_sanhedrin")
                    {
                        let file = path.strip_prefix(&root).unwrap_or(&path);
                        hits.push(format!("{}:{} {}", file.display(), n + 1, line.trim()));
                    }
                }
            }
        }
    }
    hits.sort();
    hits.dedup();
    hits
}

fn call_site_report() -> String {
    let decision = decision_path_hits();
    let adjacent = adjacent_llm_sites();
    format!(
        "decision-path LLM/Sanhedrin hits ({}):\n{}\n\n\
         shipped-crate sanhedrin / chat-completions lines ({}):\n{}",
        decision.len(),
        if decision.is_empty() {
            "(none)".to_string()
        } else {
            decision.join("\n")
        },
        adjacent.len(),
        if adjacent.is_empty() {
            "(none)".to_string()
        } else {
            adjacent.join("\n")
        }
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
             so gate_decides_without_llm cannot prove the offline run."
        );
    };
    // strace is the parent. `strace` inside `unshare -Urn` dies with
    // PTRACE_TRACEME EPERM, which would hide every connect. `-f` follows
    // the binary into the network namespace, so a connect there is still
    // in this trace.
    [
        "strace".to_string(),
        "-f".into(),
        "-e".into(),
        "trace=connect,sendto,sendmsg,sendmmsg".into(),
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

fn network_connects(trace: &str) -> Vec<String> {
    trace
        .lines()
        .filter(|line| {
            (line.contains("connect(")
                || line.contains("sendto(")
                || line.contains("sendmsg(")
                || line.contains("sendmmsg("))
                && (line.contains("AF_INET") || line.contains("AF_INET6"))
        })
        .take(12)
        .map(|line| line.trim().to_string())
        .collect()
}

struct Decisions {
    safe: Value,
    blocked: Value,
    safe_receipt: bool,
    blocked_receipt: bool,
    safe_rejected: bool,
    blocked_rejected: bool,
}

fn drive(server: &mut Server) -> Decisions {
    let created = server
        .tool(
            "smart_ingest",
            json!({
                "content": "gate-offline-canary-7c2e",
                "forceCreate": true
            }),
        )
        .expect("rpc");
    if created.rejected() {
        missing(&format!(
            "gate row needs one memory before the blocked purge. smart_ingest failed: {}",
            created.blob().chars().take(500).collect::<String>()
        ));
    }
    let id = created.body["nodeId"].as_str().unwrap_or("").to_string();
    if id.is_empty() {
        missing("smart_ingest returned no nodeId for the gate row");
    }
    let safe = server
        .tool("memory_status", json!({ "view": "health" }))
        .expect("rpc");
    let blocked = server
        .tool("memory", json!({ "action": "purge", "id": id }))
        .expect("rpc");
    Decisions {
        safe_rejected: safe.rejected(),
        blocked_rejected: blocked.rejected(),
        safe_receipt: has_gate_receipt(&safe),
        blocked_receipt: has_gate_receipt(&blocked),
        safe: scrub_value(&safe.body),
        blocked: scrub_value(&blocked.body),
    }
}

#[test]
fn gate_decides_without_llm() {
    let report = call_site_report();
    println!("{report}");

    let clear_owned = llm_env_names();
    let clear: Vec<&str> = clear_owned.iter().map(String::as_str).collect();

    let net_dir = tempfile::tempdir().unwrap();
    let mut net = Server::boot_wrapped(net_dir.path(), &[], &clear);
    let online = drive(&mut net);
    drop(net);

    let iso_dir = tempfile::tempdir().unwrap();
    let trace_path = std::env::temp_dir().join(format!(
        "vestige-gate-connect-{}.strace",
        std::process::id()
    ));
    let _ = fs::remove_file(&trace_path);
    let prefix = namespace_prefix(&trace_path);
    let mut iso = Server::boot_wrapped(iso_dir.path(), &prefix, &clear);
    let offline = drive(&mut iso);
    drop(iso);

    let trace = fs::read_to_string(&trace_path).unwrap_or_default();
    let connects = network_connects(&trace);
    let _ = fs::remove_file(&trace_path);

    let mut problems = Vec::new();
    if online.safe_rejected {
        problems.push("networked memory_status was rejected".to_string());
    }
    if !online.blocked_rejected {
        problems.push("networked purge without confirm was not rejected".to_string());
    }
    if offline.safe_rejected {
        problems.push("offline memory_status was rejected".to_string());
    }
    if !offline.blocked_rejected {
        problems.push("offline purge without confirm was not rejected".to_string());
    }
    if online.safe != offline.safe || online.safe_receipt != offline.safe_receipt {
        problems.push(format!(
            "safe decision diverged. receipt {} vs {}. online={} offline={}",
            online.safe_receipt, offline.safe_receipt, online.safe, offline.safe
        ));
    }
    if online.blocked != offline.blocked || online.blocked_receipt != offline.blocked_receipt {
        problems.push(format!(
            "blocked purge diverged. receipt {} vs {}. online={} offline={}",
            online.blocked_receipt, offline.blocked_receipt, online.blocked, offline.blocked
        ));
    }
    if !connects.is_empty() {
        problems.push(format!(
            "a process opened a network socket:\n{}",
            connects.join("\n")
        ));
    }
    if !problems.is_empty() {
        panic!(
            "FAIL: gate_decides_without_llm\n{}\n\n{}",
            problems.join("\n"),
            report
        );
    }
}

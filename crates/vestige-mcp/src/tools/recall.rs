//! Unified `recall` Tool (4.0 — handle-only, H5/H2/H8)
//!
//! `recall` accepts ONLY exact handles. Free text is never searched: any
//! input that is not a byte-exact handle — or a handle that resolves to
//! zero nodes — returns the `handle_required` payload. There is no mode
//! dispatch, no FTS path, no token mining, no candidates field.
//!
//! Handle grammar (byte-exact; no trimming, case folding or globbing):
//! `mem:<uuid-v4 lowercase>|mem-<seq>`, `sha:<repo>@<40|64 hex>`,
//! `path:<repo>@<full sha>:<path>`, `line:…#L<n>[-L<m>]`,
//! `sym:…::<parser symbol path>`, `test:<framework>:<id>`,
//! `run:<forge>:<run>[/<job>]`, `call:<session>/<call>`, `issue:<repo>#<n>`,
//! `pr:<repo>!<n>`, `purl:…`, `node:<64hex>`, `receipt:<64hex>`,
//! `tag:` (filter only), `session:`; `repo := <forge>/<numeric id>`.
//!
//! The real handle walk (strata-store, signed RECALL receipt) lands in PR 1;
//! this PR only flips the default and removes the text paths.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use vestige_core::{KnowledgeNode, OutputConfig, Storage};

use crate::cognitive::CognitiveEngine;

/// Handle schemes accepted by `recall`, in the order reported by
/// `handle_required`.
pub const ACCEPTED_PREFIXES: [&str; 15] = [
    "mem:", "sha:", "line:", "sym:", "path:", "test:", "run:", "call:", "issue:", "pr:", "purl:",
    "node:", "receipt:", "tag:", "session:",
];

/// The 8-type STRATA edge vocabulary (H4); `edge_types` must be a subset.
pub const EDGE_TYPES: [&str; 8] = [
    "touched",
    "anchored_to",
    "derived_from",
    "supersedes",
    "corrects",
    "closed_by",
    "projected_to",
    "evidence_of",
];

/// Maximum number of handles accepted in one call.
pub const MAX_HANDLES: usize = 16;

/// Maximum walk breadth accepted by this tool (the walk itself lands in PR 1).
pub const MAX_K: u32 = 3;

/// Discriminated schema for the handle-only `recall` tool.
pub fn schema() -> Value {
    serde_json::json!({
        "type": "object",
        "description": "Look up by exact handle (mem:, sha:, path:, sym:, test:, run:, call:, session:…). Returns the causal neighborhood with a signed RECALL receipt. Free text returns handle_required.",
        "required": ["handle"],
        "properties": {
            "handle": {
                "oneOf": [
                    { "type": "string" },
                    { "type": "array", "items": { "type": "string" }, "maxItems": MAX_HANDLES }
                ],
                "description": "Exact handle(s). Byte-exact grammar: mem:<uuid-v4>|mem-<seq>, sha:<forge>/<id>@<40|64hex>, path:<forge>/<id>@<sha>:<path>, line:…#L<n>[-L<m>], sym:…::<symbol>, test:<framework>:<id>, run:<forge>:<run>[/<job>], call:<session>/<call>, issue:<repo>#<n>, pr:<repo>!<n>, purl:…, node:<64hex>, receipt:<64hex>, tag:<name> (filter only), session:<id>. No prefix, case fold, glob, or free text."
            },
            "as_of": {
                "type": "string",
                "description": "Caller-supplied as-of point for the causal neighborhood (RFC3339). Defaults to the store head."
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_K,
                "description": "Maximum causal neighborhood breadth (default 1, at most 3)."
            },
            "edge_types": {
                "type": "array",
                "items": { "type": "string", "enum": EDGE_TYPES },
                "description": "Optional subset of the 8 typed edges to walk (default: all 8)."
            }
        }
    })
}

/// Unified dispatcher for `recall`. Handle-only: no mode, no text path.
pub async fn execute(
    storage: &Arc<Storage>,
    _cognitive: &Arc<Mutex<CognitiveEngine>>,
    _output_config: &OutputConfig,
    args: Option<Value>,
) -> Result<Value, String> {
    let object = match args.as_ref().and_then(Value::as_object) {
        Some(object) => object,
        None => return Ok(handle_required_payload("")),
    };
    if !object.contains_key("handle") {
        // No handle at all: free text (or nothing) is never searched.
        return Ok(handle_required_payload(
            object.get("query").and_then(Value::as_str).unwrap_or(""),
        ));
    }

    // Optional controls (validated, unused until the PR 1 walk).
    if let Some(k) = object.get("k").and_then(Value::as_u64) {
        if k > MAX_K as u64 {
            return Err(format!("recall k must be at most {MAX_K}"));
        }
    }
    if let Some(types) = object.get("edge_types").and_then(Value::as_array) {
        for t in types {
            match t.as_str() {
                Some(name) if EDGE_TYPES.contains(&name) => {}
                _ => return Err("recall edge_types must be a subset of the 8 typed edges".into()),
            }
        }
    }

    let handles: Vec<String> = match object.get("handle") {
        Some(Value::String(one)) => vec![one.clone()],
        Some(Value::Array(many)) => many
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect(),
        Some(_) => return Ok(handle_required_payload("")),
        None => return Ok(handle_required_payload("")),
    };
    if handles.len() > MAX_HANDLES {
        return Ok(handle_required_payload(&handles.join(" ")));
    }

    let mut resolved: Vec<Value> = Vec::new();
    for raw in &handles {
        match resolve_one(storage, raw) {
            Some(payload) => resolved.push(payload),
            // Parse failure or zero resolved nodes: fail closed.
            None => return Ok(handle_required_payload(raw)),
        }
    }
    Ok(serde_json::json!({ "handles": resolved }))
}

// ============================================================================
// Handle grammar — byte-exact, no trimming, case folding, or globbing.
// ============================================================================

fn is_lower_hex(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_sha(s: &str) -> bool {
    (s.len() == 40 || s.len() == 64) && is_lower_hex(s)
}

fn is_uuid_v4_lower(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (i, b) in bytes.iter().enumerate() {
        match i {
            8 | 13 | 18 | 23 => {
                if *b != b'-' {
                    return false;
                }
            }
            14 => {
                if *b != b'4' {
                    return false;
                }
            }
            19 => {
                if !matches!(b, b'8' | b'9' | b'a' | b'b') {
                    return false;
                }
            }
            _ => {
                if !b.is_ascii_hexdigit() || bytes[i].is_ascii_uppercase() {
                    return false;
                }
            }
        }
    }
    true
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn is_forge(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'.' || b == b'_'
        })
}

/// `repo := <forge>/<numeric id>`
fn is_repo(s: &str) -> bool {
    let (forge, id) = match s.split_once('/') {
        Some(parts) => parts,
        None => return false,
    };
    is_forge(forge) && is_digits(id)
}

fn is_plain(s: &str) -> bool {
    !s.is_empty() && !s.chars().any(char::is_whitespace) && !s.contains('#')
}

/// Like [`is_plain`] but `#` is allowed (test ids, run/job ids).
fn no_ws(s: &str) -> bool {
    !s.is_empty() && !s.chars().any(char::is_whitespace)
}

/// Validate `raw` against the handle grammar. Returns the scheme on success.
pub fn handle_scheme(raw: &str) -> Option<&'static str> {
    // `mem-<seq>` carries no colon: the strata sequence id IS the handle.
    if let Some(seq) = raw.strip_prefix("mem-") {
        return is_digits(seq).then_some("mem:");
    }

    let (scheme, rest) = raw.split_once(':')?;

    let ok = match scheme {
        "mem" => is_uuid_v4_lower(rest) || (rest.starts_with("mem-") && is_digits(&rest[4..])),
        "sha" => {
            let (repo, sha) = rest.split_once('@')?;
            is_repo(repo) && is_sha(sha)
        }
        "path" => {
            let (repo_sha, path) = rest.split_once(':')?;
            let (repo, sha) = repo_sha.split_once('@')?;
            is_repo(repo) && is_sha(sha) && is_plain(path)
        }
        "line" => {
            let (repo_sha_path, lines) = rest.split_once("#L")?;
            let (repo_sha, path) = repo_sha_path.split_once(':')?;
            let (repo, sha) = repo_sha.split_once('@')?;
            is_repo(repo) && is_sha(sha) && is_plain(path) && {
                let (n, m) = match lines.split_once("-L") {
                    Some((n, m)) => (n, Some(m)),
                    None => (lines, None),
                };
                is_digits(n) && m.map(is_digits).unwrap_or(true)
            }
        }
        "sym" => {
            let (repo_sha, symbol) = rest.split_once("::")?;
            let (repo, sha) = repo_sha.split_once('@')?;
            is_repo(repo) && is_sha(sha) && is_plain(symbol)
        }
        "test" => {
            let (framework, id) = rest.split_once(':')?;
            is_plain(framework) && no_ws(id)
        }
        "run" => {
            let (forge, rest) = rest.split_once(':')?;
            is_forge(forge)
                && match rest.split_once('/') {
                    Some((run, job)) => is_plain(run) && is_plain(job),
                    None => is_plain(rest),
                }
        }
        "call" => {
            let (session, call) = rest.split_once('/')?;
            is_plain(session) && is_plain(call)
        }
        "issue" => {
            let (repo, n) = rest.split_once('#')?;
            is_repo(repo) && is_digits(n)
        }
        "pr" => {
            let (repo, n) = rest.split_once('!')?;
            is_repo(repo) && is_digits(n)
        }
        "purl" => is_plain(rest),
        "node" | "receipt" => is_sha(rest),
        "tag" => is_plain(rest),
        "session" => is_plain(rest),
        _ => false,
    };
    if !ok {
        return None;
    }
    Some(match scheme {
        "mem" => "mem:",
        "sha" => "sha:",
        "line" => "line:",
        "sym" => "sym:",
        "path" => "path:",
        "test" => "test:",
        "run" => "run:",
        "call" => "call:",
        "issue" => "issue:",
        "pr" => "pr:",
        "purl" => "purl:",
        "node" => "node:",
        "receipt" => "receipt:",
        "tag" => "tag:",
        "session" => "session:",
        _ => unreachable!("scheme matched above"),
    })
}

/// Resolve one grammar-valid handle to a payload, or `None` when the input
/// is not a handle or resolves to zero nodes (both fail closed to
/// `handle_required`).
fn resolve_one(storage: &Arc<Storage>, raw: &str) -> Option<Value> {
    let scheme = handle_scheme(raw)?;

    // `mem:`/`mem-<seq>` resolve by exact id against the store. Every other
    // scheme goes through the byte-exact resolver and must return exactly one
    // exact hit; prefixes, ambiguity, and misses all fail closed.
    let mem_id: Option<String> = if let Some(seq) = raw.strip_prefix("mem-") {
        is_digits(seq).then(|| raw.to_string())
    } else {
        let rest = raw.split_once(':')?.1;
        (scheme == "mem:").then(|| rest.to_string())
    };

    let ids: Vec<String> = if let Some(id) = mem_id {
        match storage.get_node(&id) {
            Ok(Some(_)) => vec![id],
            _ => Vec::new(),
        }
    } else {
        let rest = raw.split_once(':')?.1;
        let inner = match scheme {
            "sha:" | "path:" | "sym:" | "test:" | "run:" | "call:" | "tag:" => {
                // Strip repo@/framework/forge decorations: the resolver wants
                // the bare sha/path/symbol/id bytes.
                match rest.split_once('@') {
                    Some((_, tail)) => match tail.split_once(':') {
                        Some((_, bare)) => bare,
                        None => tail,
                    },
                    None => rest,
                }
            }
            _ => rest,
        };
        let resolution = storage.resolve_handle(inner);
        if resolution.exact && resolution.ids.len() == 1 {
            resolution.ids
        } else {
            Vec::new()
        }
    };

    if ids.is_empty() {
        return None;
    }
    let nodes: Vec<Value> = ids
        .iter()
        .filter_map(|id| storage.get_node(id).ok().flatten())
        .map(node_payload)
        .collect();
    Some(serde_json::json!({
        "handle": raw,
        "kind": scheme,
        "nodes": nodes,
    }))
}

/// The single fail-closed payload for every non-handle input.
fn handle_required_payload(got: &str) -> Value {
    let mut echo = got.to_string();
    if echo.len() > 200 {
        echo = echo.chars().take(200).collect();
    }
    serde_json::json!({
        "error": "handle_required",
        "got": echo,
        "accepted": ACCEPTED_PREFIXES,
        "session_handles": [],
        "hint": "pass an exact handle; free text is never searched",
    })
}

/// Lean node payload for handle responses.
fn node_payload(node: KnowledgeNode) -> Value {
    serde_json::json!({
        "id": node.id,
        "type": node.node_type,
        "content": node.content,
        "tags": node.tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::IngestInput;

    async fn store() -> (Arc<Storage>, tempfile::TempDir, String) {
        let dir = tempfile::TempDir::new().unwrap();
        let storage = vestige_core::open_storage(Some(dir.path().join("handle.db"))).unwrap();
        let memory = storage
            .ingest(IngestInput {
                content: "Set API_TIMEOUT=2 in the deploy env".into(),
                tags: vec!["deploy-env".into()],
                ..Default::default()
            })
            .unwrap();
        let memory_id = memory.id.clone();
        (storage, dir, memory_id)
    }

    async fn run(storage: &Arc<Storage>, args: Value) -> Result<Value, String> {
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        execute(storage, &cognitive, &oc, Some(args)).await
    }

    // Spec: recall_free_text_returns_handle_required
    #[tokio::test]
    async fn recall_free_text_returns_handle_required() {
        let (storage, _dir, _id) = store().await;
        let out = run(
            &storage,
            serde_json::json!({ "query": "what did API_TIMEOUT change" }),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert_eq!(out["got"], "what did API_TIMEOUT change");
        assert_eq!(out["accepted"].as_array().unwrap().len(), 15);
        assert_eq!(out["session_handles"].as_array().unwrap().len(), 0);
        assert!(out["hint"].as_str().unwrap().contains("never searched"));
        assert!(out.get("candidates").is_none(), "no candidates field");
    }

    // Spec: recall_empty_handle_returns_handle_required
    #[tokio::test]
    async fn recall_empty_handle_returns_handle_required() {
        let (storage, _dir, _id) = store().await;
        let out = run(&storage, serde_json::json!({ "handle": "" }))
            .await
            .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert!(out.get("candidates").is_none());
    }

    // Spec: recall_handle_required_never_mines_tokens — a sentence containing
    // an existing memory's exact id still returns handle_required, with no
    // candidates field.
    #[tokio::test]
    async fn recall_handle_required_never_mines_tokens() {
        let (storage, _dir, id) = store().await;
        let sentence = format!("please look at {id} and tell me what changed");
        let out = run(&storage, serde_json::json!({ "query": sentence }))
            .await
            .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert!(out.get("candidates").is_none(), "token mining must be gone");
        assert!(out["nodes"].is_null(), "nothing may resolve from prose");
    }

    // Spec: recall_short_sha_returns_handle_required_with_hint
    #[tokio::test]
    async fn recall_short_sha_returns_handle_required_with_hint() {
        let (storage, _dir, _id) = store().await;
        let out = run(&storage, serde_json::json!({ "handle": "0123456" }))
            .await
            .unwrap();
        assert_eq!(out["error"], "handle_required");
        assert_eq!(out["got"], "0123456");
        assert!(out["hint"].as_str().unwrap().contains("exact handle"));
        // A bare full sha is also outside the grammar (sha: needs repo@).
        let out = run(
            &storage,
            serde_json::json!({ "handle": "0123456789abcdef0123456789abcdef01234567" }),
        )
        .await
        .unwrap();
        assert_eq!(out["error"], "handle_required");
    }

    // Spec: recall_search_alias_removed
    #[tokio::test]
    async fn recall_search_alias_removed() {
        let (storage, _dir, _id) = store().await;
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        let oc = OutputConfig::default();
        // The `mode` field is no longer part of recall at all.
        let out = execute(
            &storage,
            &cognitive,
            &oc,
            Some(serde_json::json!({ "mode": "lookup", "query": "anything" })),
        )
        .await
        .unwrap();
        assert_eq!(
            out["error"], "handle_required",
            "mode/query are ignored: free text is never searched"
        );
    }

    // Spec: recall_never_touches_fts — works on a store with no knowledge_fts.
    #[tokio::test]
    async fn recall_never_touches_fts() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = dir.path().join("nofts.db");
        let storage = vestige_core::open_storage(Some(db.clone())).unwrap();
        let memory = storage
            .ingest(IngestInput {
                content: "handle-only recall target".into(),
                ..Default::default()
            })
            .unwrap();
        drop(storage);
        // Strip the FTS table entirely.
        #[cfg(feature = "legacy-sqlite")]
        {
            let conn = rusqlite::Connection::open(&db).unwrap();
            for suffix in ["", "_data", "_idx", "_docsize", "_config"] {
                conn.execute_batch(&format!("DROP TABLE IF EXISTS knowledge_fts{suffix};"))
                    .unwrap();
            }
        }
        let storage = vestige_core::open_storage(Some(db)).unwrap();
        let out = run(
            &storage,
            serde_json::json!({ "handle": format!("mem:{}", memory.id) }),
        )
        .await
        .unwrap();
        assert!(
            out["handles"].is_array(),
            "handle resolution works without FTS: {out}"
        );
    }

    // Exact mem: handle resolves; grammar-valid but unknown handles fail closed.
    #[tokio::test]
    async fn mem_handle_resolves_and_unknown_fails_closed() {
        let (storage, _dir, id) = store().await;
        // A bare uuid is free text; the handle form is mem:<uuid>.
        let bare = run(&storage, serde_json::json!({ "handle": id }))
            .await
            .unwrap();
        assert_eq!(
            bare["error"], "handle_required",
            "bare uuid without mem: must fail closed"
        );
        let out = run(
            &storage,
            serde_json::json!({ "handle": format!("mem:{id}") }),
        )
        .await
        .unwrap();
        let handles = out["handles"].as_array().unwrap();
        assert_eq!(handles[0]["kind"], "mem:");
        assert_eq!(handles[0]["nodes"][0]["id"], serde_json::json!(id));

        let out = run(
            &storage,
            serde_json::json!({ "handle": "mem-000000000001" }),
        )
        .await
        .unwrap();
        assert_eq!(
            out["error"], "handle_required",
            "grammar-valid but unresolved fails closed"
        );
    }

    // Grammar unit tests: byte-exact acceptance.
    #[test]
    fn handle_grammar_is_byte_exact() {
        assert_eq!(
            handle_scheme("mem:3d2f0e1a-7b8c-4d9e-a1b2-c3d4e5f60718"),
            Some("mem:")
        );
        assert_eq!(handle_scheme("mem-000000000042"), Some("mem:"));
        assert!(
            handle_scheme("mem:3D2F0E1A-7B8C-4D9E-A1B2-C3D4E5F60718").is_none(),
            "no case fold"
        );
        assert_eq!(
            handle_scheme("sha:github/1234@0123456789abcdef0123456789abcdef01234567"),
            Some("sha:")
        );
        assert!(
            handle_scheme("sha:github/1234@0123456").is_none(),
            "no short sha"
        );
        assert!(
            handle_scheme("sha:github/abc@0123456789abcdef0123456789abcdef01234567").is_none(),
            "repo id is numeric"
        );
        assert_eq!(
            handle_scheme("path:github/1@0123456789abcdef0123456789abcdef01234567:src/main.rs"),
            Some("path:")
        );
        assert_eq!(
            handle_scheme(
                "line:github/1@0123456789abcdef0123456789abcdef01234567:src/main.rs#L40-L52"
            ),
            Some("line:")
        );
        assert_eq!(
            handle_scheme(
                "sym:github/1@0123456789abcdef0123456789abcdef01234567::vestige::core::ingest"
            ),
            Some("sym:")
        );
        assert_eq!(handle_scheme("test:junit:com.x.YTest#run"), Some("test:"));
        assert_eq!(handle_scheme("run:github:998877/1234"), Some("run:"));
        assert_eq!(handle_scheme("call:sess-01/call-09"), Some("call:"));
        assert_eq!(handle_scheme("issue:github/42#17"), Some("issue:"));
        assert_eq!(handle_scheme("pr:github/42!9"), Some("pr:"));
        assert_eq!(
            handle_scheme("node:0011223344556677889900112233445566778899001122334455667788990011"),
            Some("node:")
        );
        assert_eq!(
            handle_scheme(
                "receipt:0011223344556677889900112233445566778899001122334455667788990011"
            ),
            Some("receipt:")
        );
        assert_eq!(handle_scheme("tag:deploy-env"), Some("tag:"));
        assert_eq!(handle_scheme("session:s-123"), Some("session:"));
        assert!(handle_scheme("free text with spaces").is_none());
        assert!(handle_scheme("mem:hello").is_none());
        assert!(
            handle_scheme("SHA:github/1@0123456789abcdef0123456789abcdef01234567").is_none(),
            "scheme is case-sensitive"
        );
    }
}

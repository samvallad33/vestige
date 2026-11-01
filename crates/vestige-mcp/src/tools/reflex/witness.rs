//! Witness receipts. When an agent runs a test command that is on the
//! allowlist, word for word, Reflex records one signed witness: the command,
//! its exit code, the SHA-256 of the output the host reported, and the git
//! tree of the working directory taken just before the run and again just
//! after it. The record is signed by the store's actor key as a DSSE envelope
//! and saved through the gate as a Strata memory tagged `tree-<tree>`, so
//! `receipt(action='prove')` finds every witness of one tree by exact handle.
//!
//! The allowlist is the `[witness]` table of the owner's `vestige.toml`, plus
//! the repository's own `.vestige/witness.toml`. A command that is not on it,
//! a background run and an interrupted run are never witnessed. Trees are
//! taken through a scratch copy of the index, so the user's index is never
//! written. No model is involved anywhere in this path.

use super::*;
use std::time::SystemTime;

use vestige_core::actor::ProcessActor;
use vestige_proof::config::WitnessConfig;
use vestige_proof::witness::{WITNESS_TYPE, WitnessRecord, exit_code_in, output_digest};
use vestige_proof::{git as proof_git, tests_count};

/// Witness runs a session started and has not finished.
const PENDING_KEEP: usize = 32;
/// Largest config file read.
const CONFIG_MAX_BYTES: u64 = 1 << 20;
/// Largest junit report read.
const JUNIT_MAX_BYTES: u64 = 64 << 20;

/// An allowlisted command between its PreToolUse and its result.
#[derive(Clone, Debug)]
pub(super) struct Pending {
    pub call: String,
    pub agent: String,
    pub tree_before: Option<String>,
    pub started_at: String,
    pub started: SystemTime,
}

fn read_config(path: &Path) -> WitnessConfig {
    std::fs::metadata(path)
        .ok()
        .filter(|m| m.is_file() && m.len() <= CONFIG_MAX_BYTES)
        .and_then(|_| std::fs::read_to_string(path).ok())
        .map(|t| WitnessConfig::parse(&t))
        .unwrap_or_default()
}

/// The allowlist in force for a repository: the owner's `vestige.toml`, then
/// the repository's `.vestige/witness.toml`.
pub(crate) fn config_for(data_dir: &Path, root: Option<&Path>) -> WitnessConfig {
    let owner = read_config(&data_dir.join(vestige_core::config::CONFIG_FILE));
    match root {
        Some(r) => owner.merge(read_config(&r.join(vestige_proof::REPO_CONFIG))),
        None => owner,
    }
}

/// The repository root above `cwd`, found from git's own files without
/// running git, so a call that is not witnessed costs no process.
fn repo_root(cwd: &Path) -> Option<PathBuf> {
    let mut dir = cwd.to_path_buf();
    for _ in 0..64 {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
    None
}

/// The tag every Reflex memory of one host session carries.
pub(crate) fn session_tag(session: &str) -> String {
    let safe: String = session
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .take(96)
        .collect();
    format!("reflex-session-{safe}")
}

/// The store's actor, the key every witness and bundle is signed with.
pub(crate) fn actor_for(storage: &Arc<Storage>) -> Option<Arc<ProcessActor>> {
    static ACTORS: OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<ProcessActor>>>> =
        OnceLock::new();
    let dir = storage.data_dir().to_path_buf();
    let mut actors = ACTORS.get_or_init(Default::default).lock().ok()?;
    if let Some(a) = actors.get(&dir) {
        return Some(Arc::clone(a));
    }
    let path = vestige_core::actor::actor_key_path_for_data_dir(&dir);
    match ProcessActor::load_or_mint(&path) {
        Ok(a) => {
            let a = Arc::new(a);
            actors.insert(dir, Arc::clone(&a));
            Some(a)
        }
        Err(e) => {
            tracing::warn!(error = %e, "reflex cannot sign a witness without the actor key");
            None
        }
    }
}

/// Run blocking git work off the async worker.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    if tokio::runtime::Handle::try_current().is_err() {
        return Some(f());
    }
    tokio::task::spawn_blocking(f).await.ok()
}

fn now_millis() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// The tests a run collected, from its configured junit report when that
/// report was written during the run, else from JSON on its output.
fn count_tests(
    cfg: &WitnessConfig,
    command: &str,
    root: &Path,
    pending: Option<&Pending>,
    output: &str,
) -> Option<tests_count::TestCount> {
    if let Some(rel) = cfg.junit_for(command) {
        let path = root.join(rel);
        let fresh = std::fs::metadata(&path).ok().filter(|m| {
            m.is_file()
                && m.len() <= JUNIT_MAX_BYTES
                && pending
                    .and_then(|p| m.modified().ok().map(|t| t >= p.started))
                    .unwrap_or(false)
        });
        return fresh
            .and_then(|_| std::fs::read_to_string(&path).ok())
            .and_then(|xml| tests_count::from_junit(&xml));
    }
    tests_count::from_output(output)
}

fn status_word(r: &WitnessRecord) -> &'static str {
    match vestige_proof::claims::status_of(r) {
        vestige_proof::claims::Status::ZeroTests => "with zero tests run",
        vestige_proof::claims::Status::Green => "green",
        _ => "red",
    }
}

fn witness_item(inp: &Input, r: &WitnessRecord, envelope: &str) -> Value {
    let tests = r.tests.as_ref().map_or_else(
        || "not reported in JSON or junit".to_string(),
        |t| format!("{} ({})", t.collected, t.source),
    );
    let bound = if r.bound_to(&r.tree) {
        "the same tree before and after the run"
    } else {
        "a tree that changed during the run, so it binds to no tree"
    };
    json!({
        "node_type": "event",
        "tags": ["reflex-witness", format!("tree-{}", r.tree), session_tag(&r.session)],
        "source": format!("reflex {VERSION}, {} {}", inp.event, inp.call),
        "content": format!(
            "TEST WITNESS, recorded by Reflex. When: {}. `{}` finished {}, exit code {}, at tree {} ({bound}). \
             Tests collected: {tests}. Output SHA-256 {} over {} bytes. Session {}. \
             The signed record is the DSSE envelope on the next line; receipt(action='prove') reads only that.\n{envelope}",
            r.ended_at,
            r.command,
            status_word(r),
            r.exit.map_or_else(|| "none".to_string(), |e| e.to_string()),
            r.tree,
            r.output_sha256,
            r.output_bytes,
            or_unknown(&r.session.chars().take(8).collect::<String>()),
        ),
    })
}

/// Witness one firing: remember the tree before an allowlisted command, and
/// record the signed witness when its result arrives.
pub(super) async fn on_event(
    global: &Arc<std::sync::Mutex<Global>>,
    ctx: &Ctx<'_>,
    inp: &Input,
) -> pairing::Paired {
    let mut out = pairing::Paired::default();
    let event = inp.event.as_str();
    if inp.session.is_empty()
        || inp.cwd.is_empty()
        || !pairing::is_shell(&inp.tool)
        || !matches!(event, "PreToolUse" | "PostToolUse" | "PostToolUseFailure")
    {
        return out;
    }
    let command = inp.command.trim().to_string();
    let cwd = PathBuf::from(&inp.cwd);
    let Some(root) = repo_root(&cwd) else {
        return out;
    };
    let cfg = config_for(ctx.storage.data_dir(), Some(&root));
    if !cfg.is_allowed(&command) {
        return out;
    }
    if event == "PreToolUse" {
        if inp.call.is_empty() {
            return out;
        }
        let started = SystemTime::now();
        let started_at = now_millis();
        let dir = cwd.clone();
        let tree_before = blocking(move || proof_git::working_tree(&dir))
            .await
            .and_then(Result::ok);
        if let Ok(mut g) = global.lock() {
            let s = g.session(&inp.session);
            s.witness_pending.retain(|p| p.call != inp.call);
            s.witness_pending.push(Pending {
                call: inp.call.clone(),
                agent: inp.agent.clone(),
                tree_before,
                started_at,
                started,
            });
            while s.witness_pending.len() > PENDING_KEEP {
                s.witness_pending.remove(0);
            }
        }
        return out;
    }
    let pending = global.lock().ok().and_then(|mut g| {
        let s = g.session(&inp.session);
        let pos = s
            .witness_pending
            .iter()
            .position(|p| p.call == inp.call && p.agent == inp.agent)?;
        Some(s.witness_pending.remove(pos))
    });
    if inp.background || inp.interrupt {
        return out;
    }
    let dir = cwd.clone();
    let Some((tree, head)) = blocking(move || {
        let tree = proof_git::working_tree(&dir);
        let head = proof_git::commit_id(&dir, "HEAD").ok();
        (tree, head)
    })
    .await
    .and_then(|(t, h)| t.ok().map(|t| (t, h))) else {
        tracing::warn!("reflex could not hash the working tree for a witness");
        return out;
    };
    let failed = event == "PostToolUseFailure";
    let (exit, error) = if failed {
        (exit_code_in(&inp.error_raw), Some(inp.error_raw.as_str()))
    } else {
        (Some(0), None)
    };
    let (output_sha256, output_bytes) = output_digest(&inp.stdout, &inp.stderr, error);
    let printed = if failed {
        inp.error_raw
            .split_once('\n')
            .map(|(_, rest)| rest)
            .unwrap_or("")
    } else {
        inp.stdout.as_str()
    };
    let tests = count_tests(&cfg, &command, &root, pending.as_ref(), printed);
    let rel = std::fs::canonicalize(&cwd)
        .ok()
        .zip(std::fs::canonicalize(&root).ok())
        .and_then(|(c, r)| c.strip_prefix(&r).ok().map(Path::to_path_buf))
        .unwrap_or_default();
    let record = WitnessRecord {
        kind: WITNESS_TYPE.to_string(),
        command: command.clone(),
        exit,
        output_sha256,
        output_bytes,
        tree,
        tree_before: pending.as_ref().and_then(|p| p.tree_before.clone()),
        head,
        dir: rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/"),
        tests,
        host: inp.host.clone(),
        session: inp.session.clone(),
        call: inp.call.clone(),
        agent: inp.agent.clone(),
        started_at: pending.as_ref().map(|p| p.started_at.clone()),
        ended_at: now_millis(),
        producer: format!("vestige reflex {VERSION}"),
    };
    let Some(actor) = actor_for(ctx.storage) else {
        return out;
    };
    let envelope = record.seal(actor.did(), |m| actor.sign(m));
    let envelope = serde_json::to_string(&envelope).unwrap_or_default();
    match ingest(ctx, witness_item(inp, &record, &envelope)).await {
        Ok(node) => {
            let short: String = record.tree.chars().take(12).collect();
            out.tell.push(format!(
                "Reflex: witnessed `{}` {} at tree {short}, signed receipt {node}.",
                record.command,
                status_word(&record),
            ));
            out.wrote.push(json!({
                "kind": "witness", "memory": node, "tree": record.tree,
                "bound": record.bound_to(&record.tree), "exit": record.exit,
                "tests": record.tests.as_ref().map(|t| t.collected), "new": true,
            }));
        }
        Err(e) => tracing::warn!(error = %e, "reflex could not record a witness"),
    }
    out
}

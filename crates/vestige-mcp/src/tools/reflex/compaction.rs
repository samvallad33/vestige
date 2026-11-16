//! Compaction survival. Compaction rewrites the conversation into a summary,
//! and whatever the summary drops is gone for the rest of the session. Reflex
//! makes that checkable and repairs it by handle:
//!
//! - PreCompact: the memories the main agent holds are saved as a checkpoint
//!   memory with a receipt, listing their exact ids.
//! - PostCompact: the summary is read for those exact ids. What it kept and
//!   what it dropped are saved as an audit memory derived from the checkpoint.
//! - After compaction: the dropped memories come back by handle, decisions
//!   first, each as its current version, at the first firing that can add
//!   context once SessionStart has marked the compaction. Claude Code sends
//!   SessionStart before PostCompact, so waiting lets the audit's exact
//!   dropped set be used. With no audit, every checkpointed memory comes back.
//!
//! The ids are read as exact tokens; a paraphrased or cut id counts as dropped.

use super::*;

/// Memories re-introduced after one compaction, at most.
const BACK_MAX: usize = 12;

/// What the main agent held when compaction started.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct Checkpoint {
    pub ids: Vec<String>,
    pub turn: u32,
    pub node: Option<String>,
    /// Set by the PostCompact audit: the ids the summary did not keep.
    pub dropped: Option<Vec<String>>,
    pub audit: Option<String>,
    /// Set by SessionStart after compaction: give the memories back at the
    /// next firing that can add context.
    #[serde(default)]
    pub ready: bool,
    /// Finished edits and runs in this session, kept across the compaction
    /// so the next session does not redo them.
    #[serde(default)]
    pub done: Vec<DoneEntry>,
}

/// One finished tool effect. The receipt is the compaction checkpoint or
/// audit memory that lists it.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct DoneEntry {
    pub tool: String,
    pub file: String,
    pub command: String,
    pub call: String,
}

/// Every exact memory id written in a text, in order, once each.
pub(super) fn ids_in(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
        if is_memory_id(token) && !out.iter().any(|o| o == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn checkpoint_item(inp: &Input, ids: &[String]) -> Value {
    json!({
        "node_type": "event",
        "tags": ["reflex-compaction", "compaction-checkpoint"],
        "source": format!("reflex {VERSION}, PreCompact"),
        "content": format!(
            "COMPACTION CHECKPOINT, recorded by Reflex. When: {} UTC. Session {}, trigger {}. The agent held {} \
             memories when compaction started, by exact id: {}. Manifest {}.",
            Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            or_unknown(&inp.session.chars().take(8).collect::<String>()),
            or_unknown(&inp.trigger),
            ids.len(),
            ids.join(", "),
            manifest_digest(ids),
        ),
    })
}

fn audit_item(inp: &Input, checkpoint: &str, kept: &[String], dropped: &[String]) -> Value {
    let mut tags = vec![
        "reflex-compaction".to_string(),
        "compaction-audit".to_string(),
    ];
    for id in dropped {
        tags.push(format!("dropped:{id}"));
    }
    json!({
        "node_type": "event",
        "tags": tags,
        "source": format!("reflex {VERSION}, PostCompact"),
        "links": [{"kind": "derived_from", "to": checkpoint}],
        "content": format!(
            "COMPACTION AUDIT, recorded by Reflex. When: {} UTC. Session {}. Of the memories in checkpoint {checkpoint}, \
             the summary kept {} by exact id: {}. It dropped {}: {}. The dropped ones come back by handle when the \
             session starts again.",
            Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            or_unknown(&inp.session.chars().take(8).collect::<String>()),
            kept.len(),
            if kept.is_empty() { "none".to_string() } else { kept.join(", ") },
            dropped.len(),
            if dropped.is_empty() { "none".to_string() } else { dropped.join(", ") },
        ),
    })
}

/// A memory's current version: follow recorded retirements to the successor.
fn current(id: &str, retired: &[stale::Retirement]) -> Option<String> {
    let mut id = id.to_string();
    for _ in 0..8 {
        match retired.iter().rev().find(|r| r.old == id) {
            // A retirement with no successor means the memory is gone.
            Some(r) => id = r.new.clone()?,
            None => return Some(id),
        }
    }
    Some(id)
}

/// The type and opening of one memory, read by its exact id.
async fn describe(ctx: &Ctx<'_>, id: &str) -> Option<(String, String)> {
    let out = super::super::recall::execute(
        ctx.storage,
        ctx.cognitive,
        ctx.output_config,
        Some(json!({"handle": id, "detail_level": "summary", "limit": 1})),
    )
    .await
    .ok()?;
    let node = out.get("nodes").and_then(Value::as_array)?.first()?.clone();
    let kind = node
        .get("type")
        .or_else(|| node.get("nodeType"))
        .and_then(Value::as_str)
        .unwrap_or("memory")
        .to_string();
    let content = node.get("content").and_then(Value::as_str).unwrap_or("");
    let opening: String = content.chars().take(140).collect();
    Some((kind, opening))
}

fn rank(kind: &str) -> u8 {
    match kind {
        "decision" => 0,
        "correction" => 1,
        "fact" => 2,
        _ => 3,
    }
}

/// The lines that give back what compaction dropped, and how many memories.
async fn restore_lines(
    ctx: &Ctx<'_>,
    cp: &Checkpoint,
    retired: &[stale::Retirement],
) -> (Vec<String>, usize) {
    let wanted = cp.dropped.clone().unwrap_or_else(|| cp.ids.clone());
    let mut back: Vec<(u8, String)> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for id in wanted.iter().rev() {
        if back.len() >= BACK_MAX {
            break;
        }
        let Some(cur) = current(id, retired) else {
            continue;
        };
        if seen.contains(&cur) {
            continue;
        }
        seen.push(cur.clone());
        if let Some((kind, opening)) = describe(ctx, &cur).await {
            let note = if cur != *id {
                format!(" (current version of {id})")
            } else {
                String::new()
            };
            back.push((rank(&kind), format!("- {cur} [{kind}]{note}: {opening}")));
        }
    }
    back.sort_by_key(|(r, _)| *r);
    let mut lines = Vec::new();
    if !back.is_empty() {
        lines.push(format!(
            "Reflex: compaction {} {} memor{} you held. Here they are by handle; recall(handle=...) reads one whole{}:",
            if cp.dropped.is_some() { "dropped" } else { "may have dropped" },
            back.len(),
            if back.len() == 1 { "y" } else { "ies" },
            cp.audit
                .as_deref()
                .map(|a| format!(", audit {a}"))
                .unwrap_or_default(),
        ));
        lines.extend(back.into_iter().map(|(_, l)| l));
    }
    let told = seen.len();
    (lines, told)
}

/// Compaction survival for one firing. Returns the ledger entries and the
/// lines to tell the agent.
fn note_done(global: &Arc<std::sync::Mutex<Global>>, inp: &Input) {
    if inp.event != "PostToolUse" || inp.interrupt || !inp.error.is_empty() {
        return;
    }
    if inp.file.is_empty() && inp.command.is_empty() {
        return;
    }
    let entry = DoneEntry {
        tool: inp.tool.clone(),
        file: inp.file.clone(),
        command: inp.command.clone(),
        call: inp.call.clone(),
    };
    let Ok(mut g) = global.lock() else {
        return;
    };
    let pushed = {
        let session = g.session(&inp.session);
        let checkpoint = session.checkpoint.get_or_insert_with(Checkpoint::default);
        let duplicate =
            !entry.call.is_empty() && checkpoint.done.iter().any(|done| done.call == entry.call);
        if duplicate || checkpoint.done.len() >= 32 {
            false
        } else {
            checkpoint.done.push(entry);
            true
        }
    };
    if pushed {
        g.dirty = true;
    }
}

fn redo_line(global: &Arc<std::sync::Mutex<Global>>, inp: &Input) -> Option<String> {
    if inp.event != "PreToolUse" || inp.file.is_empty() {
        return None;
    }
    let g = global.lock().ok()?;
    let checkpoint = g.sessions.get(&inp.session)?.checkpoint.as_ref()?;
    let hit = checkpoint.done.iter().find(|done| done.file == inp.file)?;
    let receipt = checkpoint
        .audit
        .clone()
        .or_else(|| checkpoint.node.clone())
        .unwrap_or_else(|| "unreceipted".into());
    Some(format!(
        "Redo flagged: `{}` is already finished ({}, call {}). Receipt `{receipt}`.",
        hit.file, hit.tool, hit.call
    ))
}

fn ledger_lines(cp: &Checkpoint) -> Vec<String> {
    if cp.done.is_empty() {
        return Vec::new();
    }
    let receipt = cp
        .audit
        .clone()
        .or_else(|| cp.node.clone())
        .unwrap_or_else(|| "unreceipted".into());
    let mut lines = vec![format!(
        "Done ledger: {} finished effect{} in this session. Do not redo them. Receipt `{receipt}`.",
        cp.done.len(),
        if cp.done.len() == 1 { "" } else { "s" }
    )];
    for done in &cp.done {
        if done.file.is_empty() {
            lines.push(format!(
                "- run `{}` tool {} call {}",
                done.command, done.tool, done.call
            ));
        } else {
            lines.push(format!(
                "- edit `{}` tool {} call {}",
                done.file, done.tool, done.call
            ));
        }
    }
    lines
}

pub(super) async fn on_event(
    global: &Arc<std::sync::Mutex<Global>>,
    ctx: &Ctx<'_>,
    inp: &Input,
) -> (Vec<Value>, Vec<String>) {
    let mut wrote = Vec::new();
    let mut tell = Vec::new();
    if inp.session.is_empty() {
        return (wrote, tell);
    }
    if let Some(line) = redo_line(global, inp) {
        tell.push(line);
    }
    note_done(global, inp);
    match inp.event.as_str() {
        "PreCompact" => {
            let (ids, turn, prior_done) = {
                let Ok(mut g) = global.lock() else {
                    return (wrote, tell);
                };
                let s = g.session(&inp.session);
                let ids = s.contexts.get("").map(Context::ids).unwrap_or_default();
                let prior_done = s.checkpoint.clone().map(|cp| cp.done).unwrap_or_default();
                (ids, s.turn, prior_done)
            };
            if ids.is_empty() {
                return (wrote, tell);
            }
            let node = ingest(ctx, checkpoint_item(inp, &ids)).await.ok();
            if let Some(n) = &node {
                wrote.push(json!({"kind": "compaction-checkpoint", "memory": n, "held": ids.len(), "new": true}));
            }
            if let Ok(mut g) = global.lock() {
                g.session(&inp.session).checkpoint = Some(Checkpoint {
                    ids,
                    turn,
                    node,
                    done: prior_done,
                    ..Checkpoint::default()
                });
                g.dirty = true;
            }
        }
        "PostCompact" => {
            let cp = {
                let Ok(mut g) = global.lock() else {
                    return (wrote, tell);
                };
                g.session(&inp.session).checkpoint.clone()
            };
            let Some(cp) = cp else {
                return (wrote, tell);
            };
            if inp.summary.is_empty() {
                return (wrote, tell);
            }
            let found = ids_in(&inp.summary);
            let (kept, dropped): (Vec<String>, Vec<String>) =
                cp.ids.iter().cloned().partition(|id| found.contains(id));
            let audit = match &cp.node {
                Some(n) => ingest(ctx, audit_item(inp, n, &kept, &dropped)).await.ok(),
                None => None,
            };
            if let Some(a) = &audit {
                wrote.push(json!({"kind": "compaction-audit", "memory": a, "kept": kept.len(), "dropped": dropped.len(), "new": true}));
            }
            if let Ok(mut g) = global.lock() {
                if let Some(c) = g.session(&inp.session).checkpoint.as_mut() {
                    c.dropped = Some(dropped);
                    c.audit = audit;
                }
                g.dirty = true;
            }
        }
        "SessionStart" if inp.source == "compact" => {
            if let Ok(mut g) = global.lock()
                && let Some(c) = g.session(&inp.session).checkpoint.as_mut()
            {
                c.ready = true;
                g.dirty = true;
            }
        }
        event if event_budget(event) > 0 => {
            let (cp, retired) = {
                let Ok(mut g) = global.lock() else {
                    return (wrote, tell);
                };
                let retired = g.p.retired.clone();
                let s = g.session(&inp.session);
                if !s.checkpoint.as_ref().is_some_and(|c| c.ready) {
                    return (wrote, tell);
                }
                (s.checkpoint.take(), retired)
            };
            let Some(cp) = cp else {
                return (wrote, tell);
            };
            let (lines, told) = restore_lines(ctx, &cp, &retired).await;
            tell.extend(lines);
            tell.extend(ledger_lines(&cp));
            if !cp.done.is_empty()
                && let Ok(mut g) = global.lock()
            {
                g.session(&inp.session).checkpoint = Some(Checkpoint {
                    done: cp.done.clone(),
                    node: cp.node.clone(),
                    audit: cp.audit.clone(),
                    ..Checkpoint::default()
                });
                g.dirty = true;
            }
            wrote.push(json!({"kind": "compaction-restore", "told": told, "audit": cp.audit, "new": false}));
        }
        _ => {}
    }
    (wrote, tell)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ids_are_read_from_a_summary_and_a_cut_id_is_not_one() {
        let text = "Kept mem-00000000000000a1 and (mem-00000000000000b2). Cut: mem-00000000000000c. \
                    Again mem-00000000000000a1, uuid 68540393-203d-428b-9a1a-854da288e655.";
        assert_eq!(
            ids_in(text),
            vec![
                "mem-00000000000000a1".to_string(),
                "mem-00000000000000b2".to_string(),
                "68540393-203d-428b-9a1a-854da288e655".to_string()
            ]
        );
    }

    #[test]
    fn the_current_version_follows_recorded_successors() {
        let r = |old: &str, new: Option<&str>| stale::Retirement {
            old: old.into(),
            new: new.map(str::to_string),
            ..stale::Retirement::default()
        };
        let retired = vec![r("a", Some("b")), r("b", Some("c")), r("x", None)];
        assert_eq!(current("a", &retired).as_deref(), Some("c"));
        assert_eq!(current("z", &retired).as_deref(), Some("z"));
        assert_eq!(
            current("x", &retired),
            None,
            "a suppressed memory does not come back"
        );
    }

    #[tokio::test]
    async fn the_done_ledger_survives_compaction_and_a_redo_is_flagged() {
        let dir = tempfile::tempdir().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        let home = tempfile::tempdir().unwrap();
        let cog = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::cognitive::CognitiveEngine::new(),
        ));
        let oc = std::sync::Arc::new(vestige_core::OutputConfig::default());
        let conn = super::super::new_connection();
        let fire = |args: serde_json::Value| {
            let storage = std::sync::Arc::clone(&storage);
            let cog = std::sync::Arc::clone(&cog);
            let oc = std::sync::Arc::clone(&oc);
            let conn = std::sync::Arc::clone(&conn);
            let home = home.path().to_path_buf();
            async move {
                super::super::execute_with(&storage, &cog, &oc, &conn, Some(args), &home)
                    .await
                    .expect("reflex answers")
            }
        };
        fire(json!({
            "event": "PostToolUse", "session": "s-done", "tool": "Edit",
            "call": "c1", "file": "src/a.rs"
        }))
        .await;
        fire(json!({
            "event": "PostToolUse", "session": "s-done", "tool": "Bash",
            "call": "c2", "command": "cargo test -p vestige-mcp"
        }))
        .await;
        fire(json!({"event": "PreCompact", "session": "s-done", "trigger": "auto"})).await;
        fire(json!({"event": "SessionStart", "session": "s-done", "source": "compact"})).await;
        let restored = fire(json!({"event": "UserPromptSubmit", "session": "s-done"})).await;
        let text = restored["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or("");
        assert!(text.contains("Done ledger"), "{text}");
        assert!(text.contains("src/a.rs"), "{text}");
        assert!(text.contains("cargo test -p vestige-mcp"), "{text}");
        let redo = fire(json!({
            "event": "PreToolUse", "session": "s-done", "tool": "Edit",
            "call": "c3", "file": "src/a.rs"
        }))
        .await;
        let redo_text = redo["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or("");
        assert!(redo_text.contains("Redo flagged"), "{redo_text}");
        assert!(redo_text.contains("src/a.rs"), "{redo_text}");
    }
}

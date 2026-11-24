//! Stale-belief push. Every retirement any agent makes on this store is
//! recorded in order: an edit that admits a successor, a memory superseded at
//! ingest, a demotion, a suppression. Before an agent's next action, each
//! retirement of a memory it already held is told to it once, with the old id,
//! the successor, who changed it and the receipt. The retired id then leaves
//! the agent's context, so only current versions come back after compaction.

use super::*;

/// Retirements kept, newest last.
pub(super) const RETIRED_KEEP: usize = 500;

/// One memory that stopped being the current belief.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(super) struct Retirement {
    pub seq: u64,
    pub old: String,
    pub new: Option<String>,
    pub how: String,
    pub ts: String,
    pub session: String,
    #[serde(default)]
    pub agent: String,
    pub receipt: Option<String>,
    /// The successor's first sentence, when the change carried its text.
    pub first: String,
}

fn first_sentence(text: &str) -> String {
    let t = text.trim();
    let end = t
        .find(". ")
        .map(|i| i + 1)
        .or_else(|| t.find('\n'))
        .unwrap_or(t.len())
        .min(160);
    t[..t.floor_char_boundary(end)].to_string()
}

/// The retirements one successful tool result records, read from its JSON keys.
pub(super) fn retirements_of(tool: &str, result: &Value) -> Vec<Retirement> {
    let id = |k: &str| {
        result
            .get(k)
            .and_then(Value::as_str)
            .filter(|s| is_memory_id(s))
            .map(str::to_string)
    };
    let receipt = result
        .get("receiptId")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut out = Vec::new();
    match (tool, result.get("action").and_then(Value::as_str)) {
        ("memory", Some("edit")) | ("curate", Some("edit" | "supersede")) => {
            if let Some(old) = id("supersedes") {
                out.push(Retirement {
                    old,
                    new: id("nodeId"),
                    how: "edited into a successor".into(),
                    receipt,
                    first: first_sentence(
                        result
                            .get("newContentPreview")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                    ),
                    ..Retirement::default()
                });
            }
        }
        ("memory", Some("demoted")) | ("curate", Some("demote" | "demoted")) => {
            if let Some(old) = id("nodeId").or_else(|| id("id")) {
                out.push(Retirement {
                    old,
                    how: "demoted as wrong or unhelpful".into(),
                    receipt,
                    ..Retirement::default()
                });
            }
        }
        ("suppress", Some("suppress")) | ("curate", Some("suppress" | "suppressed")) => {
            if let Some(old) = id("id").or_else(|| id("nodeId")) {
                out.push(Retirement {
                    old,
                    how: "suppressed".into(),
                    receipt: receipt.clone(),
                    ..Retirement::default()
                });
            }
            let targets = result
                .pointer("/cascadeDerivedFrom/targets")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for t in targets {
                if t.get("outcome").and_then(Value::as_str) == Some("suppressed")
                    && let Some(old) = t
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| is_memory_id(s))
                {
                    out.push(Retirement {
                        old: old.to_string(),
                        how: "suppressed with what it was derived from".into(),
                        receipt: receipt.clone(),
                        ..Retirement::default()
                    });
                }
            }
        }
        ("smart_ingest", _) | ("remember", _) => {
            let mut links: Vec<Value> = result
                .get("links")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for r in result
                .get("results")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                links.extend(
                    r.get("links")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                );
            }
            for l in links {
                let kind = l
                    .get("edge")
                    .or_else(|| l.get("kind"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if kind != "supersedes" {
                    continue;
                }
                let end = |k: &str| {
                    l.get(k)
                        .and_then(Value::as_str)
                        .filter(|s| is_memory_id(s))
                        .map(str::to_string)
                };
                if let Some(old) = end("target") {
                    out.push(Retirement {
                        old,
                        new: end("source"),
                        how: "superseded by a newer memory".into(),
                        receipt: l
                            .get("receiptId")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        ..Retirement::default()
                    });
                }
            }
        }
        _ => {}
    }
    out
}

/// Record what a successful Vestige call retired, whichever connection made it.
pub fn record(conn: &ConnectionState, storage: &Arc<Storage>, tool: &str, result: &Value) {
    let found = retirements_of(tool, result);
    if found.is_empty() {
        return;
    }
    let session = conn
        .lock()
        .ok()
        .and_then(|c| c.session.clone())
        .unwrap_or_default();
    let global = global_for(storage);
    let Ok(mut g) = global.lock() else {
        return;
    };
    let ts = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut manifests: std::collections::BTreeMap<String, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for (sid, sess) in &g.sessions {
        for (aid, ctx) in &sess.contexts {
            let key = format!("{sid}:{aid}");
            let held: std::collections::BTreeSet<String> =
                ctx.held.iter().map(|(id, _)| id.clone()).collect();
            manifests.insert(key, held);
        }
    }
    let mut retired_ids = Vec::new();
    for mut r in found {
        g.p.retire_seq += 1;
        r.seq = g.p.retire_seq;
        r.ts = ts.clone();
        r.session = session.clone();
        let ripple = crate::cognitive_kernel::compute_correction_ripple(&r.old, &manifests);
        if !ripple.is_empty() {
            tracing::debug!(
                memory_id = %r.old,
                targets = ?ripple,
                "Reflex: queued live correction ripple to active agent manifests"
            );
        }
        retired_ids.push(r.old.clone());
        g.p.retired.push(r);
    }
    if g.p.retired.len() > RETIRED_KEEP {
        let drop = g.p.retired.len() - RETIRED_KEEP;
        g.p.retired.drain(..drop);
    }
    g.dirty = true;
    drop(g);
    for id in retired_ids {
        crate::circuit_index::mark_zset_delta(storage.as_ref(), &id, -1);
    }
}

/// One line for each retirement of a memory this context held before it was
/// retired, made by any other session or sibling subagent. Returns the line and the old id.
pub(super) fn lines(
    retired: &[Retirement],
    session: &str,
    agent: &str,
    ctx: &Context,
) -> Vec<(String, String)> {
    retired
        .iter()
        .filter(|r| {
            r.session != session
                || (!agent.is_empty() && r.agent != agent)
                || (agent.is_empty() && !r.agent.is_empty())
        })
        .filter(|r| ctx.get(&r.old).is_some_and(|h| r.seq > h.seq))
        .map(|r| {
            let mut line = format!(
                "Reflex: a memory you hold was {} at {} in another session: {}",
                r.how,
                hhmm(&r.ts),
                r.old
            );
            if let Some(new) = &r.new {
                line.push_str(&format!(" is now {new}"));
            }
            if let Some(rc) = &r.receipt {
                line.push_str(&format!(" (receipt {rc})"));
            }
            line.push('.');
            if !r.first.is_empty() {
                line.push_str(&format!(" It now reads: {}", r.first));
            }
            if r.new.is_some() {
                line.push_str(" Read the new one before acting on the old.");
            } else {
                line.push_str(" Do not act on it as current.");
            }
            (line, r.old.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_demotions_suppressions_and_supersessions_are_read_from_their_results() {
        let a = "mem-00000000000000a1";
        let b = "mem-00000000000000b2";
        let edit = retirements_of(
            "memory",
            &json!({"action": "edit", "nodeId": b, "supersedes": a, "newContentPreview": "Use port 4000. Not 3000.", "receiptId": "eff-1"}),
        );
        assert_eq!(edit.len(), 1);
        assert_eq!((edit[0].old.as_str(), edit[0].new.as_deref()), (a, Some(b)));
        assert_eq!(edit[0].first, "Use port 4000.");
        assert_eq!(
            retirements_of("memory", &json!({"action": "demoted", "nodeId": a})).len(),
            1
        );
        let sup = retirements_of(
            "suppress",
            &json!({"action": "suppress", "id": a, "cascadeDerivedFrom": {"targets": [
                {"id": b, "outcome": "suppressed"}, {"id": "mem-00000000000000c3", "outcome": "pending_review"}]}}),
        );
        assert_eq!(sup.len(), 2, "held targets are not retired");
        let ing = retirements_of(
            "smart_ingest",
            &json!({"nodeId": b, "links": [{"edge": "supersedes", "source": b, "target": a, "receiptId": "eff-2"}]}),
        );
        assert_eq!((ing[0].old.as_str(), ing[0].new.as_deref()), (a, Some(b)));
        let cur_edit = retirements_of(
            "curate",
            &json!({"action": "edit", "nodeId": b, "supersedes": a, "receiptId": "eff-3"}),
        );
        assert_eq!((cur_edit[0].old.as_str(), cur_edit[0].new.as_deref()), (a, Some(b)));
        assert_eq!(
            retirements_of("curate", &json!({"action": "demote", "id": a})).len(),
            1
        );
        assert_eq!(
            retirements_of("curate", &json!({"action": "suppress", "id": a})).len(),
            1
        );
        assert_eq!(
            retirements_of(
                "remember",
                &json!({"nodeId": b, "links": [{"kind": "supersedes", "source": b, "target": a}]})
            )
            .len(),
            1
        );
        assert!(retirements_of("recall", &json!({"supersedes": a})).is_empty());
        assert!(retirements_of("memory", &json!({"action": "promoted", "nodeId": a})).is_empty());
    }
}

//! `receipt(action='prove')`: the proof bundle for one commit.
//!
//! Everything here is found by exact handle. The commit's tree (without
//! `.vestige/proof`) names the tag `tree-<tree>`, which holds every witness
//! Reflex recorded at that tree; each witness's session names the tag
//! `reflex-session-<session>`, which holds that session's Failure Pairing
//! failures and fixes and its Operator stops. Pairings and stop resolutions are
//! the `closed_by` edges the log recorded between those memories.
//!
//! Each witness envelope is opened against the store's actor key before it is
//! used, a witness is kept only when it is bound to the tree, and the claims
//! are derived from the kept witnesses by `vestige_proof::claims::derive`. The
//! statement is signed by the same key and written to
//! `.vestige/proof/<tree>.intoto.jsonl`, as a git note under
//! `refs/notes/vestige-proof`, or both.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use vestige_core::Storage;
use vestige_core::storage::HandleKind;
use vestige_proof::didkey::TrustedKey;
use vestige_proof::dsse::Envelope;
use vestige_proof::statement::{
    CLAIM_BOUNDARY, Pairing, Predicate, Statement, Stop, WitnessEntry, bundle_text,
};
use vestige_proof::witness::WitnessRecord;
use vestige_proof::{NOTES_REF, PROOF_DIR, bundle_file_name, claims, config, git};

use super::reflex;

/// Where `prove` puts the bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    File,
    Notes,
    Both,
    None,
}

impl Out {
    pub fn parse(s: Option<&str>) -> Result<Out, String> {
        match s.unwrap_or("file") {
            "file" => Ok(Out::File),
            "notes" => Ok(Out::Notes),
            "both" => Ok(Out::Both),
            "none" => Ok(Out::None),
            other => Err(format!(
                "out must be file, notes, both or none, not {other:?}"
            )),
        }
    }
}

/// Every memory carrying exactly this tag.
fn tagged(storage: &Arc<Storage>, tag: &str) -> Vec<String> {
    let r = storage.resolve_handle(tag);
    if r.kind == HandleKind::Tag && r.exact {
        r.ids
    } else {
        Vec::new()
    }
}

/// The Strata payload digest of the write that admitted this memory.
fn strata_digest(storage: &Arc<Storage>, id: &str) -> Option<String> {
    let receipt = storage.get_receipt(id).ok()??;
    receipt.mutations.iter().find_map(|m| {
        m.note
            .as_deref()?
            .split_whitespace()
            .find_map(|part| part.strip_prefix("digest="))
            .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(str::to_string)
    })
}

/// The envelope a witness memory holds on its last line.
fn envelope_of(content: &str) -> Option<Envelope> {
    serde_json::from_str(content.lines().last()?.trim()).ok()
}

/// `(closed, closer)` for every `closes` edge touching `id`.
fn closes_edges(storage: &Arc<Storage>, id: &str) -> Vec<(String, String)> {
    storage
        .get_connections_for_memory(id)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|e| match e.link_type.as_str() {
            "closed_by" => Some((e.source_id, e.target_id)),
            "closes" => Some((e.target_id, e.source_id)),
            _ => None,
        })
        .collect()
}

/// The proof for `rev` in the repository at `repo`. Blocking: run it off the
/// async worker.
pub fn prove(storage: &Arc<Storage>, repo: &Path, rev: &str, out: Out) -> Result<Value, String> {
    let root = git::toplevel(repo).map_err(|e| e.to_string())?;
    let commit = git::commit_id(&root, rev).map_err(|e| e.to_string())?;
    let tree = git::commit_tree(&root, &commit).map_err(|e| e.to_string())?;
    let actor = reflex::actor_for(storage).ok_or("the store's actor key is not available")?;
    let trusted = [TrustedKey::parse(actor.did()).map_err(|e| e.to_string())?];

    let mut entries = Vec::new();
    let mut records: Vec<(String, WitnessRecord)> = Vec::new();
    let (mut unbound, mut unverified) = (0u32, Vec::new());
    for id in tagged(storage, &format!("tree-{tree}")) {
        let Some(node) = storage.get_node(&id).ok().flatten() else {
            continue;
        };
        let opened = envelope_of(&node.content)
            .ok_or_else(|| "no envelope".to_string())
            .and_then(|env| {
                WitnessRecord::open(&env, &trusted)
                    .map(|(r, _)| (env, r))
                    .map_err(|e| e.to_string())
            });
        let (env, record) = match opened {
            Ok(x) => x,
            Err(why) => {
                unverified.push(json!({"memory": id, "why": why}));
                continue;
            }
        };
        if !record.bound_to(&tree) {
            unbound += 1;
            continue;
        }
        entries.push(WitnessEntry {
            memory: id.clone(),
            strata_digest: strata_digest(storage, &id),
            envelope: env,
        });
        records.push((id, record));
    }
    entries.sort_by(|a, b| a.memory.cmp(&b.memory));
    records.sort_by(|a, b| a.0.cmp(&b.0));

    let sessions: BTreeSet<String> = records.iter().map(|(_, r)| r.session.clone()).collect();
    let mut pairings: BTreeMap<(String, String), Pairing> = BTreeMap::new();
    let mut stops: BTreeMap<String, Stop> = BTreeMap::new();
    for session in &sessions {
        for id in tagged(storage, &reflex::session_tag(session)) {
            let Some(node) = storage.get_node(&id).ok().flatten() else {
                continue;
            };
            let has = |t: &str| node.tags.iter().any(|x| x == t);
            if has("reflex-failure") || has("reflex-fix") {
                for (failure, fix) in closes_edges(storage, &id) {
                    let Some(f) = storage.get_node(&failure).ok().flatten() else {
                        continue;
                    };
                    let fixed_by_fix = storage
                        .get_node(&fix)
                        .ok()
                        .flatten()
                        .is_some_and(|n| n.tags.iter().any(|t| t == "reflex-fix"));
                    if !f.tags.iter().any(|t| t == "reflex-failure") || !fixed_by_fix {
                        continue;
                    }
                    let signature = f.tags.iter().find(|t| t.starts_with("fail-")).cloned();
                    pairings
                        .entry((failure.clone(), fix.clone()))
                        .or_insert_with(|| Pairing {
                            failure_digest: strata_digest(storage, &failure),
                            fix_digest: strata_digest(storage, &fix),
                            failure,
                            fix,
                            signature,
                            session: session.clone(),
                        });
                }
            }
            if has("operator-stop") {
                let receipt = node
                    .tags
                    .iter()
                    .find(|t| t.starts_with("op-receipt-"))
                    .cloned()
                    .unwrap_or_default();
                let rule = node
                    .tags
                    .iter()
                    .find(|t| {
                        !t.starts_with("op-receipt-")
                            && !t.starts_with("reflex-session-")
                            && !matches!(t.as_str(), "operator" | "operator-stop")
                    })
                    .cloned()
                    .unwrap_or_default();
                let closed_by = closes_edges(storage, &id)
                    .into_iter()
                    .find(|(closed, _)| *closed == id)
                    .map(|(_, closer)| closer);
                stops.entry(id.clone()).or_insert_with(|| Stop {
                    strata_digest: strata_digest(storage, &id),
                    memory: id.clone(),
                    receipt,
                    rule,
                    session: session.clone(),
                    closed_by,
                });
            }
        }
    }

    let repo_cfg = git::show_file(&root, &commit, vestige_proof::REPO_CONFIG)
        .map(|t| config::WitnessConfig::parse(&t))
        .unwrap_or_default();
    let allowlist = reflex::config_for(storage.data_dir(), None)
        .merge(repo_cfg)
        .commands;
    let claims = claims::derive(&tree, &records, &allowlist);
    let statement = Statement::new(Predicate {
        commit: commit.clone(),
        tree: tree.clone(),
        claims: claims.clone(),
        allowlist,
        witnesses: entries,
        pairings: pairings.into_values().collect(),
        stops: stops.into_values().collect(),
        sessions: sessions.into_iter().collect(),
        producer: format!("vestige {}", env!("CARGO_PKG_VERSION")),
        generated_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        claim_boundary: CLAIM_BOUNDARY.to_string(),
    });
    let envelope = statement.seal(actor.did(), |m| actor.sign(m));
    let text = bundle_text(&envelope);

    let mut written = json!({});
    if matches!(out, Out::File | Out::Both) {
        let dir = root.join(PROOF_DIR);
        std::fs::create_dir_all(&dir).map_err(|e| format!("could not create {PROOF_DIR}: {e}"))?;
        let path = dir.join(bundle_file_name(&tree));
        std::fs::write(&path, &text).map_err(|e| format!("could not write the bundle: {e}"))?;
        written["file"] = json!(format!("{PROOF_DIR}/{}", bundle_file_name(&tree)));
    }
    if matches!(out, Out::Notes | Out::Both) {
        git::write_note(&root, &commit, text.trim_end()).map_err(|e| e.to_string())?;
        written["note"] = json!(format!("{NOTES_REF} on {commit}"));
    }

    let p = &statement.predicate;
    let count = |s: claims::Status| claims.iter().filter(|c| c.status == s).count();
    Ok(json!({
        "action": "prove",
        "commit": commit,
        "tree": tree,
        "signer": actor.did(),
        "claims": claims,
        "summary": {
            "green": count(claims::Status::Green),
            "red": count(claims::Status::Red),
            "zeroTests": count(claims::Status::ZeroTests),
            "unwitnessed": count(claims::Status::Unwitnessed),
            "witnesses": p.witnesses.len(),
            "unboundWitnesses": unbound,
            "pairings": p.pairings.len(),
            "stops": p.stops.len(),
            "sessions": p.sessions.len(),
        },
        "unverifiedWitnesses": unverified,
        "written": written,
        "bundleSha256": vestige_proof::sha256_hex(text.as_bytes()),
        "verify": format!(
            "vestige-verify-proof --repo . --commit {commit} --trust {}",
            actor.did()
        ),
        "claimBoundary": CLAIM_BOUNDARY,
    }))
}

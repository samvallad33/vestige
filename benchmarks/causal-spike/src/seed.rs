//! Official seeded dataset. Content is derived from the seed; SQLite row IDs
//! come from `Storage::ingest` UUID v4 (core API) so the DB file itself is not
//! byte-identical across runs. Manifest records the assigned IDs. `run` never
//! reads this file.

use crate::rng::SplitMix64;
use crate::types::{
    CAUSE_LAG_MAX_DAYS, CAUSE_LAG_MIN_DAYS, DISTRACTORS_PER_KIND, FailuresFile, Manifest, N_STORES,
    PAIRS_PER_STORE, PREREGISTERED, PROTOCOL_PATH, PairManifest, StoreManifest, bridge_name,
    dataset_id_for, entity_name, is_multihop_pair, t0,
};
use anyhow::{Context, Result, bail};
use chrono::{Duration, Utc};
use std::fs;
use std::path::{Path, PathBuf};
use vestige_core::advanced::retroactive_backfill::{extract_entities, looks_like_failure};
use vestige_core::{IngestInput, Storage};

const CLAIM_BOUNDARY: &str =
    "receipt-backed upstream candidates that similarity search misses — NEVER automatic root cause";

const SEMANTIC_DECOYS: [&str; DISTRACTORS_PER_KIND] = [
    "Service crashed with Internal Server Error on the auth endpoint",
    "The process hit a panic during the nightly run",
    "A regression broke the checkout flow this week",
    "Latency spiked until the instance was saturated",
    "The worker hung and the queue stalled",
];

const NEUTRALS: [&str; DISTRACTORS_PER_KIND] = [
    "Reviewed the weekly notes for the docs pass",
    "Filed the meeting summary in the shared folder",
    "Updated the readme with the setup steps",
    "Sorted the inbox before the planning session",
    "Copied the agenda into the team notes",
];

#[derive(Clone)]
struct Pending {
    content: String,
    node_type: &'static str,
    tags: Vec<String>,
    when: chrono::DateTime<Utc>,
}

pub fn seed(out: &Path, seed: u64) -> Result<PathBuf> {
    if out.exists() && contains_store(out) {
        bail!(
            "refusing to write into {out:?}: path already contains a store \
             (Jul 15 live-corpus incident). Delete it or pick a new --out."
        );
    }
    fs::create_dir_all(out).with_context(|| format!("create {out:?}"))?;

    let epoch = t0();
    let mut stores = Vec::with_capacity(N_STORES);

    for store_idx in 0..N_STORES {
        let store_id = format!("store-{store_idx}");
        let store_dir = out.join(&store_id);
        fs::create_dir_all(&store_dir)?;
        let db_path = store_dir.join("vestige.db");
        if db_path.exists() {
            bail!("refusing to overwrite existing store {db_path:?}");
        }

        let storage = Storage::new(Some(db_path.clone()))
            .with_context(|| format!("open fresh store {db_path:?}"))?;
        let mut rng = SplitMix64::mix_with(seed, store_idx as u64 + 1);
        let mut pairs = Vec::with_capacity(PAIRS_PER_STORE);

        for pair_idx in 0..PAIRS_PER_STORE {
            let planted = plant_pair(store_idx, pair_idx, epoch, &mut rng)?;
            let mut ids: Vec<String> = Vec::new();
            for item in &planted.items {
                let node = storage.ingest(IngestInput {
                    content: item.content.clone(),
                    node_type: item.node_type.to_string(),
                    tags: item.tags.clone(),
                    source: Some("causal-spike-seed".into()),
                    ..Default::default()
                })?;
                storage.set_created_at(&node.id, item.when)?;
                ids.push(node.id);
            }

            let failure_id = ids[planted.failure_slot].clone();
            let cause_id = ids[planted.cause_slot].clone();
            let root_id = planted.root_slot.map(|i| ids[i].clone());
            pairs.push(PairManifest {
                pair_id: format!("s{store_idx}-p{pair_idx:02}"),
                cause_id,
                failure_id,
                root_id,
                entity: planted.entity,
                bridge_entity: planted.bridge,
                cause_lag_days: planted.cause_lag_days,
                multihop: planted.multihop,
            });
        }

        stores.push(StoreManifest {
            id: store_id.clone(),
            db: format!("{store_id}/vestige.db"),
            failures_file: format!("{store_id}/failures.json"),
            pairs,
        });
    }

    let all_failure_ids: Vec<String> = stores
        .iter()
        .flat_map(|s| s.pairs.iter().map(|p| p.failure_id.clone()))
        .collect();
    let dataset_id = dataset_id_for(&all_failure_ids);

    for store in &stores {
        let failure_ids: Vec<String> = store.pairs.iter().map(|p| p.failure_id.clone()).collect();
        write_json(
            &out.join(&store.failures_file),
            &FailuresFile {
                store_id: store.id.clone(),
                failure_ids,
                dataset_id: dataset_id.clone(),
            },
        )?;
    }

    let manifest = Manifest {
        protocol: PROTOCOL_PATH.into(),
        preregistered: PREREGISTERED.into(),
        seed,
        t0: epoch,
        claim_boundary: CLAIM_BOUNDARY.into(),
        dataset_id,
        stores,
    };
    let manifest_path = out.join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    Ok(manifest_path)
}

struct PlantedPair {
    items: Vec<Pending>,
    cause_slot: usize,
    failure_slot: usize,
    root_slot: Option<usize>,
    entity: String,
    bridge: Option<String>,
    cause_lag_days: i64,
    multihop: bool,
}

fn plant_pair(
    store_idx: usize,
    pair_idx: usize,
    epoch: chrono::DateTime<Utc>,
    rng: &mut SplitMix64,
) -> Result<PlantedPair> {
    let entity = entity_name(store_idx, pair_idx);
    let multihop = is_multihop_pair(store_idx, pair_idx);
    let cause_lag_days = rng.gen_range(CAUSE_LAG_MIN_DAYS, CAUSE_LAG_MAX_DAYS) as i64;
    let value = rng.gen_range(2, 20);

    let (cause_content, failure_content, bridge, root_content) = if multihop {
        let bridge = bridge_name(store_idx, pair_idx);
        let root = format!("Recorded {entity} in the toolchain file during the weekly pass");
        let cause = format!("Copied {entity} into {bridge} for the deploy env");
        let failure = failure_text(pair_idx);
        (cause, failure, Some(bridge), Some(root))
    } else {
        let cause = format!("Set {entity}={value} in the deploy env for faster cold starts");
        let failure = failure_text(pair_idx);
        (cause, failure, None, None)
    };

    let cause_when = epoch - Duration::days(cause_lag_days);
    let mut items = Vec::new();

    let root_slot = if let Some(root) = root_content {
        assert_quiet("root", &root, &[])?;
        let root_lag = rng.gen_range(3, 10) as i64;
        items.push(Pending {
            content: root,
            node_type: "decision",
            tags: vec![entity.clone()],
            when: cause_when - Duration::days(root_lag),
        });
        Some(0)
    } else {
        None
    };

    let cause_tags = match &bridge {
        Some(b) => vec![entity.clone(), b.clone()],
        None => vec![entity.clone()],
    };
    assert_quiet("cause", &cause_content, &cause_tags)?;
    assert_has_entity("cause", &cause_content, &cause_tags, &entity)?;
    let cause_slot = items.len();
    items.push(Pending {
        content: cause_content,
        node_type: "decision",
        tags: cause_tags,
        when: cause_when,
    });

    let fail_entity = bridge.as_ref().unwrap_or(&entity).clone();
    let fail_tags = vec![fail_entity.clone(), "crash".into()];
    assert_failure("failure", &failure_content, &fail_tags)?;
    assert_has_entity("failure", &failure_content, &fail_tags, &fail_entity)?;
    if failure_content.contains(&entity)
        || bridge.as_ref().is_some_and(|b| failure_content.contains(b))
    {
        bail!("failure content must not mention the identifier (Arm A would cheat via FTS)");
    }
    // Official 20260831 templates still share stopwords (`the`/`deploy`/`env`
    // on cause vs crash vocabulary on failure). Do not rewrite planted text
    // after seeing scores; future generations must assert this for real.

    for (i, text) in SEMANTIC_DECOYS.iter().enumerate() {
        assert_failure("decoy", text, &[])?;
        assert_no_entities("decoy", text, &[])?;
        let lag = rng.gen_range(1, 6) as i64;
        items.push(Pending {
            content: format!("{text} (pair {pair_idx} decoy {i})"),
            node_type: "event",
            tags: vec![],
            when: epoch - Duration::days(lag),
        });
        // Re-check after suffix: suffix must not introduce identifiers or strip failure.
        let last = items.last().unwrap();
        assert_failure("decoy+suffix", &last.content, &last.tags)?;
        assert_no_entities("decoy+suffix", &last.content, &last.tags)?;
    }

    for i in 0..DISTRACTORS_PER_KIND {
        let chatter =
            format!("Noted {entity} still listed in the env catalog after the deploy window ({i})");
        assert_quiet("chatter", &chatter, std::slice::from_ref(&entity))?;
        assert_has_entity("chatter", &chatter, std::slice::from_ref(&entity), &entity)?;
        let ahead = rng.gen_range(1, 5) as i64;
        items.push(Pending {
            content: chatter,
            node_type: "note",
            tags: vec![entity.clone()],
            when: epoch + Duration::days(ahead),
        });
    }

    for (i, text) in NEUTRALS.iter().enumerate() {
        assert_quiet("neutral", text, &[])?;
        assert_no_entities("neutral", text, &[])?;
        let lag = rng.gen_range(2, 20) as i64;
        items.push(Pending {
            content: format!("{text} (pair {pair_idx} bg {i})"),
            node_type: "note",
            tags: vec![],
            when: epoch - Duration::days(lag),
        });
        let last = items.last().unwrap();
        assert_quiet("neutral+suffix", &last.content, &last.tags)?;
        assert_no_entities("neutral+suffix", &last.content, &last.tags)?;
    }

    let failure_slot = items.len();
    items.push(Pending {
        content: failure_content,
        node_type: "event",
        tags: fail_tags,
        when: epoch,
    });

    Ok(PlantedPair {
        items,
        cause_slot,
        failure_slot,
        root_slot,
        entity,
        bridge,
        cause_lag_days,
        multihop,
    })
}

pub(crate) fn failure_text(pair_idx: usize) -> String {
    // Identifier lives in tags only. Putting SPIKE*_CFG_* in the body lets
    // hybrid_search / FTS recover the cause by token overlap — the thing Arm A
    // is supposed to miss.
    format!("Process crashed with Internal Server Error on the auth endpoint ({pair_idx})")
}

fn assert_quiet(role: &str, content: &str, tags: &[String]) -> Result<()> {
    if looks_like_failure(content, tags) {
        bail!("{role} must carry ZERO failure vocabulary, got: {content:?} tags={tags:?}");
    }
    Ok(())
}

fn assert_failure(role: &str, content: &str, tags: &[String]) -> Result<()> {
    if !looks_like_failure(content, tags) {
        bail!("{role} must look like a failure, got: {content:?} tags={tags:?}");
    }
    Ok(())
}

fn assert_no_entities(role: &str, content: &str, tags: &[String]) -> Result<()> {
    let ents = extract_entities(content, tags);
    if !ents.is_empty() {
        bail!("{role} must have ZERO identifier entities, got {ents:?} from {content:?}");
    }
    Ok(())
}

fn assert_has_entity(role: &str, content: &str, tags: &[String], want: &str) -> Result<()> {
    let ents = extract_entities(content, tags);
    let want_lc = want.to_lowercase();
    if !ents.iter().any(|e| e == &want_lc) {
        bail!("{role} missing entity {want} (extracted {ents:?}) from {content:?}");
    }
    Ok(())
}

fn contains_store(path: &Path) -> bool {
    contains_store_inner(path, 0)
}

fn contains_store_inner(path: &Path, depth: usize) -> bool {
    if depth > 4 || !path.exists() {
        return false;
    }
    if path.is_file() {
        return path.extension().and_then(|e| e.to_str()) == Some("db");
    }
    let Ok(walk) = fs::read_dir(path) else {
        return false;
    };
    for entry in walk.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("db") {
            return true;
        }
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir && contains_store_inner(&p, depth + 1) {
            return true;
        }
    }
    false
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestige_core::advanced::retroactive_backfill::{extract_entities, looks_like_failure};

    #[test]
    fn official_templates_obey_entity_and_failure_invariants() {
        let entity = entity_name(0, 0);
        let bridge = bridge_name(0, 0);
        let cause = format!("Set {entity}=7 in the deploy env for faster cold starts");
        let root = format!("Recorded {entity} in the toolchain file during the weekly pass");
        let mid = format!("Copied {entity} into {bridge} for the deploy env");
        let failure = failure_text(0);
        let hop_fail = failure_text(1);
        assert!(!failure.contains(&entity));
        assert!(!hop_fail.contains(&bridge));

        assert!(!looks_like_failure(&cause, std::slice::from_ref(&entity)));
        assert!(!looks_like_failure(&root, std::slice::from_ref(&entity)));
        assert!(!looks_like_failure(&mid, &[entity.clone(), bridge.clone()]));
        assert!(looks_like_failure(
            &failure,
            &[entity.clone(), "crash".into()]
        ));
        assert!(looks_like_failure(
            &hop_fail,
            &[bridge.clone(), "crash".into()]
        ));

        let cause_ents = extract_entities(&cause, std::slice::from_ref(&entity));
        assert!(cause_ents.contains(&entity.to_lowercase()));
        let decoy_ents = extract_entities(SEMANTIC_DECOYS[0], &[]);
        assert!(decoy_ents.is_empty(), "{decoy_ents:?}");
        for n in NEUTRALS {
            assert!(!looks_like_failure(n, &[]));
            assert!(extract_entities(n, &[]).is_empty());
        }
        for d in SEMANTIC_DECOYS {
            assert!(looks_like_failure(d, &[]), "{d}");
            assert!(extract_entities(d, &[]).is_empty(), "{d}");
        }
    }

    #[test]
    fn ten_of_ninety_pairs_are_multihop() {
        let n = (0..N_STORES)
            .flat_map(|s| (0..PAIRS_PER_STORE).map(move |p| (s, p)))
            .filter(|(s, p)| is_multihop_pair(*s, *p))
            .count();
        assert_eq!(n, 10);
    }

    #[test]
    fn refuse_existing_store() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("vestige.db"), b"nope").unwrap();
        assert!(contains_store(dir.path()));
    }

    #[test]
    fn live_backfill_ranks_planted_cause_above_semantic_decoy() {
        use crate::arms::backfill::{engine, rank_backfill};
        use crate::arms::load_all_nodes;
        use vestige_core::IngestInput;

        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(Some(dir.path().join("test.db"))).unwrap();
        let epoch = t0();
        let entity = entity_name(0, 0);

        let cause = storage
            .ingest(IngestInput {
                content: format!("Set {entity}=7 in the deploy env for faster cold starts"),
                node_type: "decision".into(),
                tags: vec![entity.clone()],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&cause.id, epoch - chrono::Duration::days(12))
            .unwrap();

        let decoy = storage
            .ingest(IngestInput {
                content: SEMANTIC_DECOYS[0].into(),
                node_type: "event".into(),
                tags: vec![],
                ..Default::default()
            })
            .unwrap();
        storage
            .set_created_at(&decoy.id, epoch - chrono::Duration::days(3))
            .unwrap();

        let failure = storage
            .ingest(IngestInput {
                content: failure_text(0),
                node_type: "event".into(),
                tags: vec![entity, "crash".into()],
                ..Default::default()
            })
            .unwrap();
        storage.set_created_at(&failure.id, epoch).unwrap();

        let all = load_all_nodes(&storage).unwrap();
        let failure_node = all.iter().find(|n| n.id == failure.id).unwrap();
        let ranked = rank_backfill(&storage, &engine(), failure_node, &all, false).unwrap();
        assert!(
            !ranked.is_empty(),
            "backfill must surface the planted cause"
        );
        assert_eq!(
            ranked[0].0, cause.id,
            "top rank must be the quiet cause, got {ranked:?}"
        );
        assert!(
            ranked.iter().all(|(id, _)| id != &decoy.id)
                || ranked.iter().position(|(id, _)| id == &cause.id)
                    < ranked.iter().position(|(id, _)| id == &decoy.id),
            "cause must outrank the semantic decoy: {ranked:?}"
        );
    }
}

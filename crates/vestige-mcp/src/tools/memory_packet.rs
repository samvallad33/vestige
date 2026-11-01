//! Read-only packets for local lifecycle hooks.
//!
//! A prompt is ranked by exact-token BM25 over stored memory text. An empty
//! prompt (session open, no words yet) still returns the highest-salience
//! records. Full record text is packed into about 3_000 tokens. A prompt that
//! names a recorded person does not pull in a different person's records.
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use vestige_core::{KnowledgeNode, Storage};

use super::lexical::{self, HOOK_CHAR_BUDGET};

pub(crate) fn scores(storage: &Storage) -> BTreeMap<String, i128> {
    let mut scores = BTreeMap::new();
    for row in storage.salience_receipts().unwrap_or_default() {
        if let (Some(id), Some(old), Some(new)) = (
            row["memory_id"].as_str(),
            row["old_stability_q"].as_i64(),
            row["new_stability_q"].as_i64(),
        ) {
            *scores.entry(id.to_owned()).or_default() += i128::from(new) - i128::from(old);
        }
    }
    scores
}

pub(crate) fn compare(
    a: &KnowledgeNode,
    b: &KnowledgeNode,
    scores: &BTreeMap<String, i128>,
) -> std::cmp::Ordering {
    scores
        .get(&b.id)
        .copied()
        .unwrap_or_default()
        .cmp(&scores.get(&a.id).copied().unwrap_or_default())
        .then_with(|| b.stability.total_cmp(&a.stability))
        .then_with(|| b.created_at.cmp(&a.created_at))
        .then_with(|| b.id.cmp(&a.id))
}

fn phrase(text: &str, surface: &str) -> bool {
    let text = text.to_lowercase();
    let surface = surface.to_lowercase();
    if surface.is_empty() {
        return false;
    }
    text.match_indices(&surface).any(|(at, _)| {
        let word = |c: char| c.is_alphanumeric() || c == '_';
        text[..at].chars().next_back().is_none_or(|c| !word(c))
            && text[at + surface.len()..]
                .chars()
                .next()
                .is_none_or(|c| !word(c))
    })
}

pub(crate) fn packet(storage: &Storage, scope: &str, args: &Value) -> Result<Value, String> {
    let prompt = args["prompt"].as_str().unwrap_or_default();
    if prompt.len() > 64 * 1024 {
        return Err("hook prompt exceeds 64 KiB".into());
    }
    let mut nodes = storage
        .get_all_nodes_in_scope(scope, i32::MAX, 0)
        .map_err(|e| e.to_string())?;
    let retired = storage.superseded_node_ids().map_err(|e| e.to_string())?;
    let now = chrono::Utc::now();
    nodes.retain(|n| {
        !retired.contains(&n.id)
            && n.valid_from.is_none_or(|t| t <= now)
            && n.valid_until.is_none_or(|t| t > now)
    });
    let count = nodes.len();
    let scores = scores(storage);
    let tie = |node: &KnowledgeNode| scores.get(&node.id).copied().unwrap_or_default();
    let requested = args["limit"].as_u64().unwrap_or(8).clamp(1, 24) as usize;
    let named = lexical::named_people(&nodes, prompt);
    let entities: BTreeSet<String> = nodes
        .iter()
        .flat_map(|n| n.tags.iter())
        .filter(|tag| {
            ["person:", "place:", "food:", "file:", "service:", "symbol:"]
                .iter()
                .any(|prefix| tag.starts_with(prefix))
        })
        .filter(|tag| {
            phrase(prompt, tag)
                || tag
                    .split_once(':')
                    .is_some_and(|(_, surface)| phrase(prompt, surface))
        })
        .cloned()
        .collect();
    let (selected, selection, ranking): (Vec<&KnowledgeNode>, &str, &str) =
        if prompt.trim().is_empty() {
            nodes.sort_by(|a, b| compare(a, b, &scores));
            let packed_hits: Vec<lexical::LexicalHit> = (0..nodes.len())
                .map(|index| lexical::LexicalHit {
                    index,
                    score_micro: 1,
                })
                .collect();
            (
                lexical::pack(&nodes, &packed_hits, HOOK_CHAR_BUDGET, requested.min(8)),
                "top_salience",
                "receipt_salience_then_stability_then_recency",
            )
        } else {
            let hits = lexical::rank(&nodes, prompt, tie);
            let score_of = |id: &str| {
                hits.iter()
                    .find(|hit| nodes[hit.index].id == id)
                    .map(|hit| hit.score_micro)
                    .unwrap_or(0)
            };
            let mut indexes: Vec<usize> = hits
                .iter()
                .map(|hit| hit.index)
                .filter(|index| lexical::allows_person(&nodes[*index], &named))
                .collect();
            // A recorded identity the prompt names still qualifies when the
            // memory's other words do not repeat the prompt.
            for (index, node) in nodes.iter().enumerate() {
                if !entities.is_empty()
                    && node.tags.iter().any(|tag| entities.contains(tag))
                    && lexical::allows_person(node, &named)
                    && !indexes.contains(&index)
                {
                    indexes.push(index);
                }
            }
            indexes.sort_by(|a, b| {
                score_of(&nodes[*b].id)
                    .cmp(&score_of(&nodes[*a].id))
                    .then_with(|| compare(&nodes[*a], &nodes[*b], &scores))
            });
            let packed_hits: Vec<lexical::LexicalHit> = indexes
                .into_iter()
                .map(|index| lexical::LexicalHit {
                    index,
                    score_micro: 1,
                })
                .collect();
            let packed = lexical::pack(&nodes, &packed_hits, HOOK_CHAR_BUDGET, requested);
            let selection = if hits.is_empty() && !entities.is_empty() {
                "exact_recorded_entities"
            } else {
                "exact_token_bm25"
            };
            (
                packed,
                selection,
                "exact_token_bm25_then_salience_then_recency",
            )
        };
    let ids: BTreeSet<&str> = selected.iter().map(|n| n.id.as_str()).collect();
    let seed_indexes: Vec<usize> = selected
        .iter()
        .filter_map(|node| nodes.iter().position(|candidate| candidate.id == node.id))
        .collect();
    let bridge_rows = lexical::bridges(&nodes, &seed_indexes, 4);
    let bridges: Vec<Value> = bridge_rows
        .iter()
        .map(|bridge| {
            let node = &nodes[bridge.index];
            json!({
                "id": node.id,
                "content": node.content,
                "tags": node.tags,
                "path": bridge.path,
            })
        })
        .collect();
    let receipts: Vec<Value> = storage
        .salience_receipts()
        .map_err(|e| e.to_string())?
        .into_iter()
        .rev()
        .filter(|r| r["memory_id"].as_str().is_some_and(|id| ids.contains(id)))
        .take(8)
        .map(|receipt| {
            json!({
                "receipt_id": receipt["receipt_id"],
                "memory_id": receipt["memory_id"],
                "trigger_id": receipt["trigger_id"],
                "reason": receipt["reason"],
                "direction": raise_direction(&receipt),
                "old_score": receipt["old_score"],
                "new_score": receipt["new_score"],
                "old_stability_q": receipt["old_stability_q"],
                "new_stability_q": receipt["new_stability_q"],
            })
        })
        .collect();
    Ok(json!({
        "memory_count": count,
        "entities": entities,
        "memories": selected.iter().map(|n| json!({
            "id": n.id,
            "content": n.content,
            "tags": n.tags,
            "stability": n.stability,
            "salience_delta_q": scores.get(&n.id).copied().unwrap_or_default().to_string(),
        })).collect::<Vec<_>>(),
        "bridges": bridges,
        "backfill_receipts": receipts,
        "backfill_autofire": super::backfill::autofire_enabled(),
        "selection": selection,
        "token_budget": lexical::HOOK_TOKEN_BUDGET,
        "ranking_signal": ranking,
    }))
}

fn raise_direction(receipt: &Value) -> &'static str {
    let raised = match (
        receipt["new_stability_q"].as_i64(),
        receipt["old_stability_q"].as_i64(),
    ) {
        (Some(new), Some(old)) => new > old,
        _ => receipt["new_score"].as_f64() > receipt["old_score"].as_f64(),
    };
    if raised { "raise" } else { "lower" }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn entity_surfaces_use_case_insensitive_whole_boundaries() {
        assert!(phrase("Where should MAYA go?", "Maya"));
        assert!(!phrase("Mayakovsky", "Maya"));
        assert!(phrase("try Bangkok Garden?", "Bangkok Garden"));
        assert!(!phrase("Bangkok Gardens", "Bangkok Garden"));
    }
}

//! Reopen triangle (#560). Rows are the week a Failure Pairing `closes`
//! (`closed_by`) edge was written; a cell counts later failures on the same
//! exact `fail-…` handle recorded as a `reopened` edge. Age-to-age factors
//! come from the chain-ladder method. The expected reopen count for weeks
//! that are still developing uses the Cape Cod / Bornhuetter-Ferguson form
//! of those factors so a recent week with no reopens yet still gets a
//! count from the developed frequency, from counts and time only.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use vestige_core::Storage;

/// Seven 24-hour days, in milliseconds. Origin weeks and ages are counted
/// from the Unix epoch with this width so the triangle does not read a clock.
pub const WEEK_MS: i64 = 7 * 24 * 60 * 60 * 1000;

fn week_of(ms: i64) -> i64 {
    ms.div_euclid(WEEK_MS)
}

fn edge_ms(created_at: DateTime<Utc>) -> i64 {
    created_at.timestamp_millis()
}

fn handle_of(tags: &[String]) -> Option<String> {
    tags.iter()
        .find(|tag| tag.starts_with("fail-") && tag.len() > 5)
        .cloned()
}

fn cwd_of(content: &str) -> Option<&str> {
    let rest = content.split("Working directory: ").nth(1)?;
    let value = rest.split('.').next().unwrap_or(rest).trim();
    if value.is_empty() || value == "unknown" {
        None
    } else {
        Some(value)
    }
}

fn path_matches(value: &str, repository: &str) -> bool {
    value == repository
        || value.starts_with(&(repository.to_string() + "/"))
        || repository.starts_with(&(value.to_string() + "/"))
}

fn in_repository(tags: &[String], content: &str, repository: &str) -> bool {
    if tags
        .iter()
        .any(|tag| tag == repository || tag == &format!("repo:{repository}"))
    {
        return true;
    }
    if tags.iter().any(|tag| {
        tag.strip_prefix("repo:")
            .is_some_and(|root| path_matches(root, repository))
    }) {
        return true;
    }
    cwd_of(content).is_some_and(|cwd| path_matches(cwd, repository))
}

fn in_agent(tags: &[String], agent: Option<&str>) -> bool {
    match agent {
        None => true,
        Some(name) => tags
            .iter()
            .any(|tag| tag == name || tag == &format!("agent:{name}")),
    }
}

#[derive(Clone, Debug)]
struct Close {
    week: i64,
    at_ms: i64,
    handle: String,
    failure: String,
    fix: String,
}

#[derive(Clone, Debug)]
struct Reopen {
    at_ms: i64,
    handle: String,
    failure: String,
    fix: String,
}

struct Matched {
    close: Close,
    age: i64,
}

fn collect(
    storage: &Arc<Storage>,
    repository: &str,
    agent: Option<&str>,
) -> Result<(Vec<Close>, Vec<Reopen>), String> {
    let edges = storage.get_all_connections().map_err(|e| e.to_string())?;
    let mut closes = Vec::new();
    let mut reopens = Vec::new();
    for edge in edges {
        let (kind, failure_id, fix_id) = match edge.link_type.as_str() {
            "closed_by" => ("close", edge.source_id.clone(), edge.target_id.clone()),
            "closes" => ("close", edge.target_id.clone(), edge.source_id.clone()),
            "reopened" => ("reopen", edge.source_id.clone(), edge.target_id.clone()),
            _ => continue,
        };
        let Some(failure) = storage.get_node(&failure_id).ok().flatten() else {
            continue;
        };
        let fix = storage.get_node(&fix_id).ok().flatten();
        let tags = {
            let mut tags = failure.tags.clone();
            if let Some(fix) = &fix {
                tags.extend(fix.tags.iter().cloned());
            }
            tags
        };
        let mut content = failure.content.clone();
        if let Some(fix) = &fix {
            content.push('\n');
            content.push_str(&fix.content);
        }
        if !in_repository(&tags, &content, repository) {
            continue;
        }
        if !in_agent(&tags, agent) {
            continue;
        }
        let Some(handle) = handle_of(&failure.tags).or_else(|| {
            fix.as_ref()
                .and_then(|node| handle_of(&node.tags))
                .map(|tag| tag.replacen("fix-", "fail-", 1))
        }) else {
            continue;
        };
        let at_ms = edge_ms(edge.created_at);
        if kind == "close" {
            closes.push(Close {
                week: week_of(at_ms),
                at_ms,
                handle,
                failure: failure_id,
                fix: fix_id,
            });
        } else {
            reopens.push(Reopen {
                at_ms,
                handle,
                failure: failure_id,
                fix: fix_id,
            });
        }
    }
    closes.sort_by_key(|close| (close.at_ms, close.failure.clone(), close.fix.clone()));
    reopens.sort_by_key(|reopen| (reopen.at_ms, reopen.failure.clone()));
    Ok((closes, reopens))
}

fn match_reopens(closes: &[Close], reopens: &[Reopen]) -> Vec<Matched> {
    let mut out = Vec::new();
    for reopen in reopens {
        let Some(close) = closes
            .iter()
            .filter(|close| {
                close.handle == reopen.handle
                    && close.at_ms < reopen.at_ms
                    && (reopen.fix.is_empty() || close.fix == reopen.fix)
            })
            .max_by_key(|close| close.at_ms)
        else {
            continue;
        };
        let age = week_of(reopen.at_ms) - close.week;
        if age < 0 {
            continue;
        }
        out.push(Matched {
            close: close.clone(),
            age,
        });
    }
    out
}

struct Projection {
    weeks: Vec<i64>,
    cells: Vec<Vec<i64>>,
    factors: Vec<f64>,
    remaining: Vec<f64>,
    ultimate: Vec<f64>,
    expected: f64,
}

/// Chain-ladder age-to-age factors on cumulative counts, plus Cape Cod
/// expected remaining reopens.
fn project(closes: &[Close], matched: &[Matched], as_of_ms: i64) -> Projection {
    let as_of_week = week_of(as_of_ms);
    let mut weeks: Vec<i64> = closes.iter().map(|close| close.week).collect();
    weeks.sort_unstable();
    weeks.dedup();
    if weeks.is_empty() {
        return Projection {
            weeks,
            cells: Vec::new(),
            factors: Vec::new(),
            remaining: Vec::new(),
            ultimate: Vec::new(),
            expected: 0.0,
        };
    }
    let max_age = weeks
        .iter()
        .map(|week| (as_of_week - week).max(0))
        .max()
        .unwrap_or(0);
    let ages = max_age + 1;
    let mut cells = vec![vec![0i64; ages as usize]; weeks.len()];
    let mut exposure = vec![0i64; weeks.len()];
    for close in closes {
        if let Ok(row) = weeks.binary_search(&close.week) {
            exposure[row] += 1;
        }
    }
    for item in matched {
        if let Ok(row) = weeks.binary_search(&item.close.week)
            && item.age <= max_age
        {
            cells[row][item.age as usize] += 1;
        }
    }
    let mut cumulative = vec![vec![0i64; ages as usize]; weeks.len()];
    for (row, line) in cells.iter().enumerate() {
        let mut total = 0;
        for (age, count) in line.iter().enumerate() {
            total += count;
            cumulative[row][age] = total;
        }
    }
    let mut factors = vec![1.0; max_age.max(0) as usize];
    for age in 0..max_age as usize {
        let mut num = 0i64;
        let mut den = 0i64;
        for (row, week) in weeks.iter().enumerate() {
            let observed = as_of_week - week;
            if observed > age as i64 {
                num += cumulative[row][age + 1];
                den += cumulative[row][age];
            }
        }
        factors[age] = if den == 0 {
            1.0
        } else {
            num as f64 / den as f64
        };
    }
    let ldf: Vec<f64> = (0..=max_age as usize)
        .map(|age| factors[age..].iter().product::<f64>())
        .collect();
    let mut used_exposure = 0.0;
    let mut observed_total = 0.0;
    let mut remaining = vec![0.0; weeks.len()];
    let mut ultimate = vec![0.0; weeks.len()];
    for (row, week) in weeks.iter().enumerate() {
        let age = (as_of_week - week).clamp(0, max_age) as usize;
        let reported = cumulative[row][age] as f64;
        let developed = if ldf[age] == 0.0 { 1.0 } else { 1.0 / ldf[age] };
        used_exposure += exposure[row] as f64 * developed;
        observed_total += reported;
        ultimate[row] = reported * ldf[age];
        remaining[row] = (ultimate[row] - reported).max(0.0);
    }
    let mu = if used_exposure == 0.0 {
        0.0
    } else {
        observed_total / used_exposure
    };
    let mut expected = 0.0;
    for (row, week) in weeks.iter().enumerate() {
        let age = (as_of_week - week).clamp(0, max_age) as usize;
        let developed = if ldf[age] == 0.0 { 1.0 } else { 1.0 / ldf[age] };
        let cape = (exposure[row] as f64 * mu * (1.0 - developed)).max(0.0);
        // Prefer Cape Cod when the current week has no observed reopens yet;
        // classic chain-ladder would project zero from a zero start.
        if cumulative[row][age] == 0 {
            remaining[row] = cape;
            ultimate[row] = cape;
        }
        expected += remaining[row];
    }
    Projection {
        weeks,
        cells,
        factors,
        remaining,
        ultimate,
        expected,
    }
}

fn likely_handles(
    closes: &[Close],
    matched: &[Matched],
    remaining_by_week: &BTreeMap<i64, f64>,
    exposure_by_week: &BTreeMap<i64, i64>,
) -> Vec<Value> {
    #[derive(Default)]
    struct Row {
        closes: i64,
        reopens: i64,
        failure_ids: Vec<String>,
        fix_ids: Vec<String>,
        last_close_ms: i64,
        last_close_week: i64,
        last_fix: String,
        last_failure: String,
        reopened_after_last: bool,
    }
    let mut by_handle: BTreeMap<String, Row> = BTreeMap::new();
    for close in closes {
        let row = by_handle.entry(close.handle.clone()).or_default();
        row.closes += 1;
        row.failure_ids.push(close.failure.clone());
        row.fix_ids.push(close.fix.clone());
        if close.at_ms >= row.last_close_ms {
            row.last_close_ms = close.at_ms;
            row.last_close_week = close.week;
            row.last_fix = close.fix.clone();
            row.last_failure = close.failure.clone();
            row.reopened_after_last = false;
        }
    }
    for item in matched {
        let row = by_handle.entry(item.close.handle.clone()).or_default();
        row.reopens += 1;
        if item.close.at_ms == row.last_close_ms && item.close.fix == row.last_fix {
            row.reopened_after_last = true;
        }
    }
    let mut rows: Vec<(String, Row, f64, f64)> = by_handle
        .into_iter()
        .filter(|(_, row)| !row.reopened_after_last && row.closes > 0)
        .map(|(handle, row)| {
            let rate = if row.closes == 0 {
                0.0
            } else {
                row.reopens as f64 / row.closes as f64
            };
            let week_remaining = remaining_by_week
                .get(&row.last_close_week)
                .copied()
                .unwrap_or(0.0);
            let exposure = exposure_by_week
                .get(&row.last_close_week)
                .copied()
                .unwrap_or(0);
            let residual = if exposure == 0 {
                0.0
            } else {
                week_remaining / exposure as f64
            };
            (handle, row, rate, residual)
        })
        .collect();
    rows.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.reopens.cmp(&a.1.reopens))
            .then(b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });
    rows.into_iter()
        .take(20)
        .map(|(handle, row, rate, _)| {
            json!({
                "handle": handle,
                "closes": row.closes,
                "reopens": row.reopens,
                "reopen_rate": rate,
                "open_fix": row.last_fix,
                "open_failure": row.last_failure,
                "failure_ids": row.failure_ids,
                "fix_ids": row.fix_ids,
            })
        })
        .collect()
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// Render the reopen triangle for one repository, optionally one agent.
pub fn execute(storage: &Arc<Storage>, args: Option<Value>) -> Result<Value, String> {
    let raw = args.unwrap_or_else(|| json!({}));
    let repository = raw
        .get("repository")
        .or_else(|| raw.get("cwd"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "reopen_triangle needs a repository (a git root, a path prefix, or a repo: tag)"
                .to_string()
        })?;
    let agent = raw
        .get("agent")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let (closes, reopens) = collect(storage, repository, agent)?;
    let matched = match_reopens(&closes, &reopens);
    let as_of_ms = closes
        .iter()
        .map(|close| close.at_ms)
        .chain(reopens.iter().map(|reopen| reopen.at_ms))
        .max()
        .unwrap_or(0);
    let projection = project(&closes, &matched, as_of_ms);
    let remaining_by_week: BTreeMap<i64, f64> = projection
        .weeks
        .iter()
        .copied()
        .zip(projection.remaining.iter().copied())
        .collect();
    let mut exposure_by_week: BTreeMap<i64, i64> = BTreeMap::new();
    for close in &closes {
        *exposure_by_week.entry(close.week).or_insert(0) += 1;
    }
    let handles = likely_handles(&closes, &matched, &remaining_by_week, &exposure_by_week);
    let origin_weeks: Vec<Value> = projection
        .weeks
        .iter()
        .enumerate()
        .map(|(row, week)| {
            json!({
                "week": week,
                "week_start_ms": week * WEEK_MS,
                "closes": exposure_by_week.get(week).copied().unwrap_or(0),
                "cells": projection.cells[row],
                "expected_remaining": round4(projection.remaining[row]),
                "expected_ultimate": round4(projection.ultimate[row]),
            })
        })
        .collect();
    Ok(json!({
        "tool": "precedent",
        "query": "reopen_triangle",
        "repository": repository,
        "agent": agent,
        "as_of_ms": as_of_ms,
        "as_of_week": week_of(as_of_ms),
        "week_ms": WEEK_MS,
        "closes": closes.len(),
        "reopens": matched.len(),
        "origin_weeks": origin_weeks,
        "age_to_age_factors": projection.factors.iter().copied().map(round4).collect::<Vec<_>>(),
        "expected_reopen_count": round4(projection.expected),
        "handles_most_likely_to_reopen": handles,
        "note": "Rows are origin weeks of Failure Pairing closes edges. A cell counts reopened edges on the same fail- handle, by age in weeks. Age-to-age factors are chain-ladder on cumulative reopen counts. expected_reopen_count is the Cape Cod remaining, so a recent week with no reopens yet still gets a projection from the developed frequency.",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use vestige_core::{ConnectionRecord, IngestInput, KnowledgeNode};

    fn open() -> (Arc<Storage>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = crate::strata_memory::open(dir.path()).unwrap();
        (storage, dir)
    }

    fn ingest(storage: &Arc<Storage>, content: &str, tags: &[&str]) -> KnowledgeNode {
        storage
            .ingest(IngestInput {
                content: content.to_string(),
                tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
                node_type: "event".into(),
                ..Default::default()
            })
            .unwrap()
    }

    fn link(storage: &Arc<Storage>, source: &str, target: &str, link_type: &str, at_ms: i64) {
        let created = DateTime::<Utc>::from_timestamp_millis(at_ms).unwrap();
        storage
            .save_connection(&ConnectionRecord {
                source_id: source.to_string(),
                target_id: target.to_string(),
                strength: 1.0,
                link_type: link_type.to_string(),
                created_at: created,
                last_activated: created,
                activation_count: 0,
            })
            .unwrap();
    }

    /// A known triangle: week 0 has 10 closes and reopens (1, 2, 1) at ages
    /// 0..2; week 1 has 8 closes and reopens (1, 1); week 2 has 5 closes and
    /// no reopens yet. Factors are 2.5 and 4/3. Week 1 keeps classic
    /// chain-ladder remaining 2/3; week 2 uses Cape Cod remaining 1.2.
    #[test]
    fn chain_ladder_factors_and_cape_cod_expected_count() {
        let (storage, _dir) = open();
        let repo = "/tmp/reopen-triangle-demo";
        let week0 = 100 * WEEK_MS;
        let mut ids = Vec::new();
        for i in 0..10 {
            let failure = ingest(
                &storage,
                &format!("TOOL FAILURE. Working directory: {repo}."),
                &["fail-aaaa", "repo:/tmp/reopen-triangle-demo"],
            );
            let fix = ingest(
                &storage,
                &format!("FIX. Working directory: {repo}."),
                &["fix-aaaa", "repo:/tmp/reopen-triangle-demo"],
            );
            link(&storage, &failure.id, &fix.id, "closed_by", week0 + i);
            ids.push((failure.id, fix.id));
        }
        // Reopens at ages 0, 1, 1, 2 for the first four closes of week 0,
        // plus two more at ages 1 and 0 so cells are 1, 2, 1.
        let later = |age: i64, index: usize| {
            let (failure, fix) = &ids[index];
            let new_failure = ingest(
                &storage,
                &format!("TOOL FAILURE. Working directory: {repo}."),
                &["fail-aaaa", "repo:/tmp/reopen-triangle-demo"],
            );
            link(
                &storage,
                &new_failure.id,
                fix,
                "reopened",
                week0 + age * WEEK_MS + 100,
            );
            let _ = failure;
        };
        later(0, 0);
        later(1, 1);
        later(1, 2);
        later(2, 3);

        let week1 = 101 * WEEK_MS;
        let mut week1_ids = Vec::new();
        for i in 0..8 {
            let failure = ingest(
                &storage,
                &format!("TOOL FAILURE. Working directory: {repo}."),
                &["fail-bbbb", "repo:/tmp/reopen-triangle-demo"],
            );
            let fix = ingest(
                &storage,
                &format!("FIX. Working directory: {repo}."),
                &["fix-bbbb", "repo:/tmp/reopen-triangle-demo"],
            );
            link(&storage, &failure.id, &fix.id, "closed_by", week1 + i);
            week1_ids.push((failure.id, fix.id));
        }
        for (age, index) in [(0, 0), (1, 1)] {
            let new_failure = ingest(
                &storage,
                &format!("TOOL FAILURE. Working directory: {repo}."),
                &["fail-bbbb", "repo:/tmp/reopen-triangle-demo"],
            );
            link(
                &storage,
                &new_failure.id,
                &week1_ids[index].1,
                "reopened",
                week1 + age * WEEK_MS + 100,
            );
        }

        let week2 = 102 * WEEK_MS;
        for i in 0..5 {
            let failure = ingest(
                &storage,
                &format!("TOOL FAILURE. Working directory: {repo}."),
                &["fail-cccc", "repo:/tmp/reopen-triangle-demo"],
            );
            let fix = ingest(
                &storage,
                &format!("FIX. Working directory: {repo}."),
                &["fix-cccc", "repo:/tmp/reopen-triangle-demo"],
            );
            link(&storage, &failure.id, &fix.id, "closed_by", week2 + i);
        }

        let out = execute(&storage, Some(json!({"repository": repo}))).unwrap();
        assert_eq!(out["closes"], 23, "{out}");
        assert_eq!(out["reopens"], 6, "{out}");
        assert_eq!(out["age_to_age_factors"][0], 2.5, "{out}");
        assert_eq!(out["age_to_age_factors"][1], 1.3333, "{out}");
        // Week 1 classic remaining 2/3; week 2 Cape Cod 1.2; total 1.8667.
        let weeks = out["origin_weeks"].as_array().unwrap();
        assert_eq!(weeks.len(), 3, "{out}");
        assert_eq!(weeks[0]["cells"], json!([1, 2, 1]), "{out}");
        assert_eq!(weeks[1]["cells"], json!([1, 1, 0]), "{out}");
        assert_eq!(weeks[2]["cells"], json!([0, 0, 0]), "{out}");
        assert_eq!(out["expected_reopen_count"], 1.8667, "{out}");
        assert_eq!(weeks[1]["expected_remaining"], 0.6667, "{out}");
        assert_eq!(weeks[2]["expected_remaining"], 1.2, "{out}");
        let handles = out["handles_most_likely_to_reopen"].as_array().unwrap();
        assert!(
            handles
                .iter()
                .any(|row| row["handle"] == "fail-aaaa" && row["reopens"] == 4),
            "{out}"
        );
        // A repository with no matching edges still renders.
        let empty = execute(&storage, Some(json!({"repository": "/tmp/other"}))).unwrap();
        assert_eq!(empty["closes"], 0, "{empty}");
        assert_eq!(empty["expected_reopen_count"], 0.0, "{empty}");
        assert!(execute(&storage, Some(json!({}))).is_err());
    }

    #[test]
    fn agent_filter_keeps_only_that_agents_edges() {
        let (storage, _dir) = open();
        let repo = "/tmp/reopen-agent";
        let at = 50 * WEEK_MS;
        let failure = ingest(
            &storage,
            &format!("TOOL FAILURE. Working directory: {repo}."),
            &["fail-zzzz", "repo:/tmp/reopen-agent", "agent:worker"],
        );
        let fix = ingest(
            &storage,
            &format!("FIX. Working directory: {repo}."),
            &["fix-zzzz", "repo:/tmp/reopen-agent", "agent:worker"],
        );
        link(&storage, &failure.id, &fix.id, "closed_by", at);
        let out = execute(
            &storage,
            Some(json!({"repository": repo, "agent": "worker"})),
        )
        .unwrap();
        assert_eq!(out["closes"], 1, "{out}");
        let other = execute(
            &storage,
            Some(json!({"repository": repo, "agent": "other"})),
        )
        .unwrap();
        assert_eq!(other["closes"], 0, "{other}");
    }
}

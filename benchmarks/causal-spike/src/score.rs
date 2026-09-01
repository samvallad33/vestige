//! Score is the only stage that opens the manifest. Missing arms, duplicate
//! queries, and unparseable run files are hard errors — never silent skips.

use crate::git::git_head;
use crate::types::{
    Arm, ArmMetrics, GateChecks, GateVerdict, Manifest, PairManifest, RunOutput, dataset_id_for,
};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub fn score(runs_dir: &Path, manifest_path: &Path) -> Result<PathBuf> {
    let manifest: Manifest = serde_json::from_slice(
        &fs::read(manifest_path).with_context(|| format!("read manifest {manifest_path:?}"))?,
    )?;
    let mut runs: Vec<RunOutput> = Vec::new();
    for entry in fs::read_dir(runs_dir).with_context(|| format!("read runs dir {runs_dir:?}"))? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read(&path).with_context(|| format!("read {path:?}"))?;
        let run: RunOutput =
            serde_json::from_slice(&text).with_context(|| format!("parse run JSON {path:?}"))?;
        runs.push(run);
    }
    if runs.is_empty() {
        bail!("no run JSON files found in {runs_dir:?}");
    }

    let pairs: Vec<&PairManifest> = manifest
        .stores
        .iter()
        .flat_map(|s| s.pairs.iter())
        .collect();
    let expected: Vec<String> = pairs.iter().map(|p| p.failure_id.clone()).collect();
    let expected_dataset = if manifest.dataset_id.is_empty() {
        dataset_id_for(&expected)
    } else {
        manifest.dataset_id.clone()
    };
    for run in &runs {
        if !run.dataset_id.is_empty() && run.dataset_id != expected_dataset {
            bail!(
                "run dataset_id {} does not match manifest dataset_id {expected_dataset}",
                run.dataset_id
            );
        }
    }

    let mut by_arm: HashMap<String, Vec<&RunOutput>> = HashMap::new();
    for run in &runs {
        by_arm
            .entry(run.arm.as_str().to_string())
            .or_default()
            .push(run);
    }

    for arm in [Arm::Lexical, Arm::Backfill, Arm::CausalGraph] {
        if !by_arm.contains_key(arm.as_str()) {
            bail!("missing protocol arm {arm}; scored comparison must include ALL locked arms");
        }
    }

    let results_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results");
    fs::create_dir_all(&results_dir)?;
    let commit = git_head();
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");

    let mut metrics = Vec::new();
    let mut scored_arms = vec![Arm::Lexical, Arm::Backfill, Arm::CausalGraph];
    for extra in [Arm::LexicalAnd, Arm::LexicalOr, Arm::LexicalEmbed, Arm::CausalGraphV2] {
        if by_arm.contains_key(extra.as_str()) {
            scored_arms.push(extra);
        }
    }
    for arm in scored_arms {
        let arm_runs = by_arm.get(arm.as_str()).expect("checked");
        let queries = index_queries(arm_runs)?;
        for id in &expected {
            if !queries.contains_key(id) {
                bail!("arm {arm} missing failure {id}");
            }
        }
        for id in queries.keys() {
            if !expected.iter().any(|e| e == id) {
                bail!("arm {arm} has extra failure {id} not in the manifest");
            }
        }
        let m = score_arm(&manifest, &pairs, arm_runs, &by_arm)?;
        let path = results_dir.join(format!("{}-{:.7}-{stamp}.json", arm.as_str(), commit));
        write_json(&path, &m)?;
        eprintln!("wrote {}", path.display());
        metrics.push(m);
    }

    let lexical = metrics
        .iter()
        .find(|m| m.arm == Arm::Lexical)
        .cloned()
        .expect("lexical required");
    let backfill = metrics
        .iter()
        .find(|m| m.arm == Arm::Backfill)
        .cloned()
        .expect("backfill required");
    let causal_graph = metrics
        .iter()
        .find(|m| m.arm == Arm::CausalGraph)
        .cloned()
        .expect("causal-graph required");
    let lexical_or = metrics.iter().find(|m| m.arm == Arm::LexicalOr).cloned();
    let lexical_and = metrics.iter().find(|m| m.arm == Arm::LexicalAnd).cloned();
    let lexical_embed = metrics.iter().find(|m| m.arm == Arm::LexicalEmbed).cloned();

    let sep = causal_graph.separation_rate_vs_lexical.unwrap_or(0.0);
    let mh = causal_graph.multihop_recall_at_3.unwrap_or(0.0);
    let gate = GateChecks {
        c_recall_at_1_ge_0_60: causal_graph.recall_at_1 >= 0.60,
        c_recall_at_3_ge_0_80: causal_graph.recall_at_3 >= 0.80,
        c_separation_vs_a_ge_0_40: sep >= 0.40,
        c_recall_at_3_ge_b: causal_graph.recall_at_3 >= backfill.recall_at_3,
        c_multihop_recall_at_3_ge_0_50: mh >= 0.50,
    };
    let pass = gate.c_recall_at_1_ge_0_60
        && gate.c_recall_at_3_ge_0_80
        && gate.c_separation_vs_a_ge_0_40
        && gate.c_recall_at_3_ge_b
        && gate.c_multihop_recall_at_3_ge_0_50;
    let outcome = verdict_outcome(pass, &lexical).to_string();
    let verdict = GateVerdict {
        commit: commit.clone(),
        seed: manifest.seed,
        lexical,
        backfill,
        causal_graph,
        lexical_or,
        lexical_and,
        lexical_embed,
        gate,
        outcome,
        claim_licensed_if_pass: manifest.claim_boundary.clone(),
        claim_never_licensed: "automatic root cause".into(),
        protocol: manifest.protocol.clone(),
        dataset_id: expected_dataset,
    };
    let c2_verdict = metrics
        .iter()
        .find(|m| m.arm == Arm::CausalGraphV2)
        .cloned()
        .map(|c2| {
            let sep2 = c2.separation_rate_vs_lexical.unwrap_or(0.0);
            let mh2 = c2.multihop_recall_at_3.unwrap_or(0.0);
            let gate2 = GateChecks {
                c_recall_at_1_ge_0_60: c2.recall_at_1 >= 0.60,
                c_recall_at_3_ge_0_80: c2.recall_at_3 >= 0.80,
                c_separation_vs_a_ge_0_40: sep2 >= 0.40,
                c_recall_at_3_ge_b: c2.recall_at_3 >= verdict.backfill.recall_at_3,
                c_multihop_recall_at_3_ge_0_50: mh2 >= 0.50,
            };
            let pass2 = gate2.c_recall_at_1_ge_0_60
                && gate2.c_recall_at_3_ge_0_80
                && gate2.c_separation_vs_a_ge_0_40
                && gate2.c_recall_at_3_ge_b
                && gate2.c_multihop_recall_at_3_ge_0_50;
            let outcome2 = verdict_outcome(pass2, &verdict.lexical).to_string();
            crate::types::GateVerdictC2 {
                causal_graph_v2: c2,
                gate: gate2,
                outcome: outcome2,
            }
        });
    let verdict = crate::types::GateVerdictFile {
        base: verdict,
        c2: c2_verdict,
    };
    let path = results_dir.join(format!("verdict-{:.7}-{stamp}.json", commit));
    write_json(&path, &verdict)?;
    eprintln!(
        "verdict C1={}{} → {}",
        verdict.base.outcome,
        verdict
            .c2
            .as_ref()
            .map(|c| format!(" C2={}", c.outcome))
            .unwrap_or_default(),
        path.display()
    );
    Ok(path)
}

fn score_arm(
    manifest: &Manifest,
    pairs: &[&PairManifest],
    arm_runs: &[&RunOutput],
    by_arm: &HashMap<String, Vec<&RunOutput>>,
) -> Result<ArmMetrics> {
    let mut hits1 = 0usize;
    let mut hits3 = 0usize;
    let mut rr_sum = 0.0;
    let mut clock = 0.0;
    let mut n_clock = 0usize;
    let mut mh_hits = 0usize;
    let mut mh_n = 0usize;
    let mut sep_hits = 0usize;
    let mut sep_or_hits = 0usize;
    let mut list_len = 0usize;
    let mut empty = 0usize;
    let mut answered = 0usize;

    let query_index = index_queries(arm_runs)?;
    let lexical_index = by_arm
        .get(Arm::Lexical.as_str())
        .map(|runs| index_queries(runs))
        .transpose()?;
    let lexical_or_index = by_arm
        .get(Arm::LexicalOr.as_str())
        .map(|runs| index_queries(runs))
        .transpose()?;

    for pair in pairs.iter().copied() {
        let Some(q) = query_index.get(&pair.failure_id) else {
            continue;
        };
        answered += 1;
        clock += q.wall_clock_ms;
        n_clock += 1;
        list_len += q.ranked_ids.len();
        if q.ranked_ids.is_empty() {
            empty += 1;
        }
        if let Some(rank) = q
            .ranked_ids
            .iter()
            .position(|id| id == &pair.cause_id)
            .map(|i| i + 1)
        {
            if rank == 1 {
                hits1 += 1;
            }
            if rank <= 3 {
                hits3 += 1;
            }
            rr_sum += 1.0 / rank as f64;
        }
        if pair.multihop
            && let Some(root_id) = &pair.root_id
        {
            mh_n += 1;
            if q.ranked_ids.iter().take(3).any(|id| id == root_id) {
                mh_hits += 1;
            }
        }
        if let Some(lex) = &lexical_index
            && let Some(lq) = lex.get(&pair.failure_id)
        {
            let c_ok = q.ranked_ids.iter().take(3).any(|id| id == &pair.cause_id);
            let a_ok = lq.ranked_ids.iter().take(3).any(|id| id == &pair.cause_id);
            if c_ok && !a_ok {
                sep_hits += 1;
            }
        }
        if let Some(lex) = &lexical_or_index
            && let Some(lq) = lex.get(&pair.failure_id)
        {
            let c_ok = q.ranked_ids.iter().take(3).any(|id| id == &pair.cause_id);
            let a_ok = lq.ranked_ids.iter().take(3).any(|id| id == &pair.cause_id);
            if c_ok && !a_ok {
                sep_or_hits += 1;
            }
        }
    }

    let n = pairs.len().max(1);
    let arm = arm_runs[0].arm;
    let acc = if arm_runs.iter().any(|r| r.accumulation_ms.is_some()) {
        Some(arm_runs.iter().filter_map(|r| r.accumulation_ms).sum())
    } else {
        None
    };
    let amortized = acc.map(|a| {
        if n_clock == 0 {
            a
        } else {
            (a + clock) / n_clock as f64
        }
    });
    Ok(ArmMetrics {
        arm,
        commit: arm_runs[0].commit.clone(),
        seed: manifest.seed,
        n_pairs: pairs.len(),
        recall_at_1: hits1 as f64 / n as f64,
        recall_at_3: hits3 as f64 / n as f64,
        mrr: rr_sum / n as f64,
        mean_wall_clock_ms: if n_clock == 0 {
            0.0
        } else {
            clock / n_clock as f64
        },
        search_mode: arm_runs[0].search_mode.clone(),
        n_multihop: mh_n,
        multihop_recall_at_3: if mh_n > 0 {
            Some(mh_hits as f64 / mh_n as f64)
        } else {
            None
        },
        separation_rate_vs_lexical: if arm != Arm::Lexical && lexical_index.is_some() {
            Some(sep_hits as f64 / n as f64)
        } else {
            None
        },
        separation_rate_vs_lexical_or: if !matches!(arm, Arm::Lexical | Arm::LexicalOr)
            && lexical_or_index.is_some()
        {
            Some(sep_or_hits as f64 / n as f64)
        } else {
            None
        },
        accumulation_ms: acc,
        amortized_ms_per_query: amortized,
        mean_list_len: if answered == 0 {
            0.0
        } else {
            list_len as f64 / answered as f64
        },
        empty_list_rate: if answered == 0 {
            0.0
        } else {
            empty as f64 / answered as f64
        },
        n_answered_pairs: answered,
    })
}

fn verdict_outcome(pass: bool, lexical: &ArmMetrics) -> &'static str {
    // A perfect separation rate against a baseline that returned zero
    // candidates is not a measurement. Do not print PASS in that shape.
    if lexical.n_answered_pairs == 0 || lexical.empty_list_rate == 1.0 {
        "INVALID-BASELINE"
    } else if pass {
        "PASS"
    } else {
        "FAIL"
    }
}

fn index_queries(runs: &[&RunOutput]) -> Result<HashMap<String, crate::types::QueryResult>> {
    let mut map = HashMap::new();
    for run in runs {
        for q in &run.queries {
            if map.insert(q.failure_id.clone(), q.clone()).is_some() {
                bail!(
                    "duplicate query for failure {} on arm {}",
                    q.failure_id,
                    run.arm
                );
            }
        }
    }
    Ok(map)
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QueryResult;
    use chrono::Utc;

    fn pair(cause: &str, failure: &str, multihop: bool, root: Option<&str>) -> PairManifest {
        PairManifest {
            pair_id: "p".into(),
            cause_id: cause.into(),
            failure_id: failure.into(),
            root_id: root.map(|s| s.into()),
            entity: "SPIKE0_CFG_00".into(),
            bridge_entity: None,
            cause_lag_days: 10,
            multihop,
            identifier_in_failure_content: false,
        }
    }

    fn run(arm: Arm, failure: &str, ranked: &[&str]) -> RunOutput {
        RunOutput {
            arm,
            store: "x".into(),
            store_id: "store-0".into(),
            commit: "abc".into(),
            search_mode: "test".into(),
            lookback_days: Some(60),
            top_k: 10,
            started_at: Utc::now(),
            finished_at: Utc::now(),
            queries: vec![QueryResult {
                failure_id: failure.into(),
                ranked_ids: ranked.iter().map(|s| (*s).into()).collect(),
                scores: ranked.iter().map(|_| 1.0).collect(),
                wall_clock_ms: 1.0,
            }],
            accumulation_ms: None,
            dataset_id: "d".into(),
            scratch_store: None,
            embedding_ready: false,
        }
    }

    #[test]
    fn recall_and_mrr_and_separation() {
        let pairs = [pair("cause", "fail", false, None)];
        let refs: Vec<&PairManifest> = pairs.iter().collect();
        let c = run(Arm::CausalGraph, "fail", &["cause", "x"]);
        let a = run(Arm::Lexical, "fail", &["noise", "x"]);
        let c_ref = [&c];
        let a_ref = [&a];
        let mut by = HashMap::new();
        by.insert("causal-graph".into(), vec![&c]);
        by.insert("lexical".into(), vec![&a]);
        let manifest = Manifest {
            protocol: "".into(),
            preregistered: "".into(),
            seed: 1,
            t0: Utc::now(),
            claim_boundary: "".into(),
            dataset_id: "".into(),
            generation: "v2".into(),
            stores: vec![],
        };
        let m = score_arm(&manifest, &refs, &c_ref, &by).unwrap();
        assert_eq!(m.recall_at_1, 1.0);
        assert_eq!(m.recall_at_3, 1.0);
        assert_eq!(m.mrr, 1.0);
        assert_eq!(m.separation_rate_vs_lexical, Some(1.0));
        assert_eq!(m.mean_list_len, 2.0);
        let ma = score_arm(&manifest, &refs, &a_ref, &by).unwrap();
        assert_eq!(ma.recall_at_1, 0.0);
        assert_eq!(ma.separation_rate_vs_lexical, None);
        assert_eq!(ma.multihop_recall_at_3, None);
    }

    #[test]
    fn duplicate_query_is_an_error() {
        let a = run(Arm::Lexical, "fail", &["x"]);
        let b = run(Arm::Lexical, "fail", &["y"]);
        let runs = [&a, &b];
        assert!(index_queries(&runs).is_err());
    }

    #[test]
    fn empty_lexical_lists_cannot_print_pass() {
        let lexical = ArmMetrics {
            arm: Arm::Lexical,
            commit: "x".into(),
            seed: 1,
            n_pairs: 90,
            recall_at_1: 0.0,
            recall_at_3: 0.0,
            mrr: 0.0,
            mean_wall_clock_ms: 1.0,
            search_mode: "fts_keyword_and".into(),
            n_multihop: 10,
            multihop_recall_at_3: Some(0.0),
            separation_rate_vs_lexical: None,
            separation_rate_vs_lexical_or: None,
            accumulation_ms: None,
            amortized_ms_per_query: None,
            mean_list_len: 0.0,
            empty_list_rate: 1.0,
            n_answered_pairs: 90,
        };
        assert_eq!(verdict_outcome(true, &lexical), "INVALID-BASELINE");
        let mut nonempty = lexical.clone();
        nonempty.empty_list_rate = 0.0;
        nonempty.mean_list_len = 10.0;
        nonempty.recall_at_3 = 0.0;
        assert_eq!(verdict_outcome(true, &nonempty), "PASS");
        assert_eq!(verdict_outcome(false, &nonempty), "FAIL");
    }
}

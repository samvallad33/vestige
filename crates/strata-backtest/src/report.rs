//! Canonical results for both corpora.
//!
//! The numbers are whatever the frozen protocol produces. This module does
//! not retune a constant, a corpus, or a claim threshold.

use strata_store::StoreError;

use crate::canon::{Json, hex_bytes, q_decimal, quantize};
use crate::cuts::{eligible_bounds, thin_cuts};
use crate::evb::{aggregate_evb, run_arm};
use crate::manifest::{CutHead, prereg_blake3};
use crate::mechanism::{
    Mechanism, Prefix, Query, firewall_accepts, project_prefix, require_firewall,
};
use crate::need::{
    Degree, FsrsR, Recency, SeededRandom, SrNeed, StationaryNeed, Uniform, aggregate_need,
    predict_cut,
};
use crate::protocol::{DEVIATIONS, EVB_ARMS, HORIZON, RECORDED_CORPUS, SYNTH_CORPUS, TABLE_ID};
use crate::rng::log_seed;

use crate::corpus::load_corpus;

/// JSON and markdown rendered from one run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendered {
    /// `mattar-evb-v1.json` bytes, including the trailing newline.
    pub json: String,
    /// `mattar-evb-v1.md` bytes.
    pub markdown: String,
    /// Thinned cuts on the synthetic log.
    pub synth_cuts: u64,
    /// Thinned cuts on the recorded-operations log.
    pub recorded_cuts: u64,
}

/// Fold both corpora and render the results files.
pub fn render_results(prereg: &[u8]) -> Result<Rendered, String> {
    let synth = score_corpus(SYNTH_CORPUS)?;
    let recorded = score_corpus(RECORDED_CORPUS)?;
    let synth_cuts = synth.n_cuts;
    let recorded_cuts = recorded.n_cuts;
    let json = render_json(prereg, &synth, &recorded);
    let markdown = render_markdown(prereg, &synth, &recorded);
    Ok(Rendered {
        json,
        markdown,
        synth_cuts,
        recorded_cuts,
    })
}

struct Scored {
    id: &'static str,
    n_cuts: u64,
    cuts: Vec<CutHead>,
    need: crate::need::NeedReport,
    evb: crate::evb::EvbReport,
}

fn score_corpus(id: &'static str) -> Result<Scored, String> {
    let loaded = load_corpus(id).map_err(store_err)?;
    let bounds = thin_cuts(&eligible_bounds(&loaded.events, &[]));
    let mut cuts = Vec::with_capacity(bounds.len());
    let mut predictions = Vec::with_capacity(bounds.len());
    let mut runs = Vec::with_capacity(bounds.len());
    for bound in bounds {
        let fold = loaded.copy.store.as_of(bound).map_err(store_err)?;
        let prefix = project_prefix(&fold, &loaded.events, bound, &[]).map_err(store_err)?;
        check_firewall(&prefix, id)?;
        predictions.push(predict_cut(&prefix, &loaded.events, id));
        let mut arm_runs = Vec::with_capacity(EVB_ARMS.len());
        for arm in EVB_ARMS {
            let run = run_arm(arm, &prefix, &loaded.events, id);
            for proof in &run.proofs {
                if !firewall_accepts(proof, bound) {
                    return Err(format!(
                        "firewall: {id} arm {arm} at {bound} cites a frame past the cut"
                    ));
                }
            }
            arm_runs.push(run);
        }
        cuts.push(CutHead {
            bound_seq: prefix.bound_seq,
            prefix_head_seq: prefix.head_seq,
            prefix_head_frame_hash: prefix.head_frame_hash,
            state_digest: prefix.state_digest,
        });
        runs.push(arm_runs);
    }
    let n_cuts = u64::try_from(cuts.len()).unwrap_or(u64::MAX);
    Ok(Scored {
        id,
        n_cuts,
        cuts,
        need: aggregate_need(id, &predictions),
        evb: aggregate_evb(id, &runs),
    })
}

fn check_firewall(prefix: &Prefix, corpus_id: &str) -> Result<(), String> {
    let query = Query {
        corpus_id: corpus_id.to_string(),
        bound_seq: prefix.bound_seq,
        horizon: HORIZON,
    };
    let mechanisms: [&dyn Mechanism; 7] = [
        &SrNeed,
        &StationaryNeed,
        &FsrsR,
        &Recency,
        &Degree,
        &Uniform,
        &SeededRandom,
    ];
    for mechanism in mechanisms {
        let ranked = mechanism.rank(prefix, &query);
        require_firewall(prefix.bound_seq, &ranked)?;
    }
    Ok(())
}

fn store_err(err: StoreError) -> String {
    err.to_string()
}

fn q(value: f64) -> Json {
    Json::Int(quantize(value))
}

fn render_json(prereg: &[u8], synth: &Scored, recorded: &Scored) -> String {
    Json::obj(vec![
        (
            "corpora",
            Json::Arr(vec![corpus_json(synth), corpus_json(recorded)]),
        ),
        (
            "deviations",
            Json::Arr(
                DEVIATIONS
                    .iter()
                    .map(|id| Json::Str((*id).to_string()))
                    .collect(),
            ),
        ),
        ("dream_windows", Json::Arr(Vec::new())),
        (
            "prereg_blake3",
            Json::Str(hex_bytes(&prereg_blake3(prereg))),
        ),
        ("table_id", Json::Str(TABLE_ID.to_string())),
    ])
    .canonical()
}

fn corpus_json(scored: &Scored) -> Json {
    Json::obj(vec![
        (
            "cuts",
            Json::Arr(scored.cuts.iter().map(cut_json).collect()),
        ),
        ("evb", evb_json(&scored.evb)),
        ("id", Json::Str(scored.id.to_string())),
        ("log_seed", Json::Str(hex_bytes(&log_seed(scored.id)))),
        ("n_cuts", Json::U64(scored.n_cuts)),
        ("need", need_json(&scored.need)),
    ])
}

fn cut_json(cut: &CutHead) -> Json {
    Json::obj(vec![
        ("bound_seq", Json::U64(cut.bound_seq)),
        (
            "prefix_head_frame_hash",
            Json::Str(hex_bytes(&cut.prefix_head_frame_hash)),
        ),
        ("prefix_head_seq", Json::U64(cut.prefix_head_seq)),
        ("state_digest", Json::Str(hex_bytes(&cut.state_digest))),
    ])
}

fn need_json(report: &crate::need::NeedReport) -> Json {
    let arms = report
        .arms
        .iter()
        .map(|arm| {
            Json::obj(vec![
                ("auc_ci_hi_q", q(arm.auc_hi)),
                ("auc_ci_lo_q", q(arm.auc_lo)),
                ("auc_q", q(arm.auc)),
                ("id", Json::Str(arm.id.to_string())),
                ("ll_q", q(arm.ll)),
                ("n", Json::U64(arm.n)),
                ("ndcg_q", q(arm.ndcg)),
            ])
        })
        .collect();
    let fig6c = report
        .fig6c
        .iter()
        .map(|(bin, n, entropy)| {
            Json::obj(vec![
                ("bin", Json::Str((*bin).to_string())),
                ("entropy_q", q(*entropy)),
                ("n", Json::U64(*n)),
            ])
        })
        .collect();
    Json::obj(vec![
        ("arms", Json::Arr(arms)),
        ("claim", Json::Str(report.claim.to_string())),
        ("fig6c", Json::Arr(fig6c)),
        ("sr_minus_fsrs_auc_q", q(report.sr_minus_fsrs)),
        ("sr_minus_fsrs_ci_hi_q", q(report.sr_minus_hi)),
        ("sr_minus_fsrs_ci_lo_q", q(report.sr_minus_lo)),
    ])
}

fn evb_json(report: &crate::evb::EvbReport) -> Json {
    let arms = report
        .arms
        .iter()
        .map(|arm| {
            let by_budget = arm
                .by_budget
                .iter()
                .map(|(budget, ret, lo, hi)| {
                    Json::obj(vec![
                        (
                            "budget",
                            Json::U64(u64::try_from(*budget).unwrap_or(u64::MAX)),
                        ),
                        ("ci_hi_q", q(*hi)),
                        ("ci_lo_q", q(*lo)),
                        ("return_q", q(*ret)),
                    ])
                })
                .collect();
            Json::obj(vec![
                ("by_budget", Json::Arr(by_budget)),
                ("id", Json::Str(arm.id.to_string())),
                ("mean_ci_hi_q", q(arm.mean_hi)),
                ("mean_ci_lo_q", q(arm.mean_lo)),
                ("mean_over_budgets_q", q(arm.mean)),
            ])
        })
        .collect();
    Json::obj(vec![
        ("arms", Json::Arr(arms)),
        ("claim", Json::Str(report.claim.to_string())),
        ("evb_minus_gain_only_ci_hi_q", q(report.diff_hi)),
        ("evb_minus_gain_only_ci_lo_q", q(report.diff_lo)),
        ("evb_minus_gain_only_q", q(report.diff)),
        ("fraction_reached_q", q(report.fraction_reached)),
        ("mean_budget_to_90_q", q(report.mean_budget_to_90)),
        ("n_reached", Json::U64(report.n_reached)),
    ])
}

fn render_markdown(prereg: &[u8], synth: &Scored, recorded: &Scored) -> String {
    let mut out = String::new();
    out.push_str("# Mattar–Daw EVB results (`mattar-evb-v1`)\n\n");
    out.push_str("Q values are Q32.32, shown with eight truncated fraction digits. The JSON integer is the exact published value. The Q layer is the model under test.\n\n");
    out.push_str(&format!(
        "Preregistration blake3 `{hash}`.\n\n",
        hash = hex_bytes(&prereg_blake3(prereg))
    ));
    out.push_str("Deviations, in protocol order:\n\n");
    for (index, id) in DEVIATIONS.iter().enumerate() {
        out.push_str(&format!("{}. `{id}`\n", index + 1));
    }
    out.push_str("\nDream windows: none.\n");
    write_corpus(&mut out, synth);
    write_corpus(&mut out, recorded);
    out
}

fn write_corpus(out: &mut String, scored: &Scored) {
    out.push_str(&format!(
        "\n## {id}\n\nLog seed `{seed}`. Cuts: {n}.\n\n",
        id = scored.id,
        seed = hex_bytes(&log_seed(scored.id)),
        n = scored.n_cuts
    ));
    out.push_str("### Need\n\n");
    out.push_str(&format!(
        "Claim `{claim}`. SR Need AUC minus FSRS R: {diff} [{lo}, {hi}].\n\n",
        claim = scored.need.claim,
        diff = dec(scored.need.sr_minus_fsrs),
        lo = dec(scored.need.sr_minus_lo),
        hi = dec(scored.need.sr_minus_hi)
    ));
    out.push_str("| arm | n | AUC | CI low | CI high | NDCG@5 | LL |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for arm in &scored.need.arms {
        out.push_str(&format!(
            "| `{id}` | {n} | {auc} | {lo} | {hi} | {ndcg} | {ll} |\n",
            id = arm.id,
            n = arm.n,
            auc = dec(arm.auc),
            lo = dec(arm.auc_lo),
            hi = dec(arm.auc_hi),
            ndcg = dec(arm.ndcg),
            ll = dec(arm.ll)
        ));
    }
    out.push_str("\nFig. 6c, mean Need-row entropy by maximum outgoing transition. No claim.\n\n");
    out.push_str("| bin | n | entropy |\n");
    out.push_str("| --- | ---: | ---: |\n");
    for (bin, n, entropy) in &scored.need.fig6c {
        out.push_str(&format!(
            "| `{bin}` | {n} | {entropy} |\n",
            entropy = dec(*entropy)
        ));
    }
    out.push_str("\n### Gain × Need\n\n");
    out.push_str(&format!(
        "Claim `{claim}`. EVB minus Gain-only: {diff} [{lo}, {hi}].\n\n",
        claim = scored.evb.claim,
        diff = dec(scored.evb.diff),
        lo = dec(scored.evb.diff_lo),
        hi = dec(scored.evb.diff_hi)
    ));
    out.push_str(&format!(
        "EVB reached 90% of the oracle on {n} cuts (fraction {fraction}). Mean budget among those cuts: {budget}.\n\n",
        n = scored.evb.n_reached,
        fraction = dec(scored.evb.fraction_reached),
        budget = dec(scored.evb.mean_budget_to_90)
    ));
    out.push_str("| arm | mean | CI low | CI high | b0 | b1 | b2 | b4 | b8 | b12 | b20 |\n");
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for arm in &scored.evb.arms {
        out.push_str(&format!(
            "| `{id}` | {mean} | {lo} | {hi} |",
            id = arm.id,
            mean = dec(arm.mean),
            lo = dec(arm.mean_lo),
            hi = dec(arm.mean_hi)
        ));
        for (_, ret, _, _) in &arm.by_budget {
            out.push_str(&format!(" {ret} |", ret = dec(*ret)));
        }
        out.push('\n');
    }
}

fn dec(value: f64) -> String {
    q_decimal(quantize(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_are_byte_stable_with_at_least_four_cuts_and_make_no_winner_assertion() {
        let prereg = include_bytes!("../../../docs/benchmarks/MATTAR-EVB-PREREGISTRATION.md");
        let left = render_results(prereg).unwrap();
        let right = render_results(prereg).unwrap();
        assert_eq!(left.json, right.json);
        assert_eq!(left.markdown, right.markdown);
        assert!(left.synth_cuts >= 4, "synth cuts {}", left.synth_cuts);
        assert!(
            left.recorded_cuts >= 4,
            "recorded cuts {}",
            left.recorded_cuts
        );
        assert!(left.json.contains("\"id\":\"evb\""));
        assert!(left.json.contains("\"id\":\"indicator_need\""));
        assert!(left.json.contains("\"id\":\"sr_need\""));
        assert!(left.json.contains("one_step_backups_pr3_not_run"));
        assert!(left.json.ends_with('\n'));
        assert!(!left.json.contains("null"));
        assert_eq!(left.json.matches("\"id\":\"evb\"").count(), 2);
    }
}

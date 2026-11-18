//! Waves 20–22 (#540, #551–#554, #556–#558, #561, #562, #564, #572):
//! Counterfactual replay, Ochiai fault spectrum, lagged causes, co-allocation,
//! reperformance bounds, citator flags, outbreak alarms, reconsolidation,
//! crash simulation, ZK replay commitments, MAST failure subtypes & root-cause benchmarks.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// #540: Counterfactual replay — what decision would fire without `excluded_mem_id`?
pub fn counterfactual_replay(
    active_rules: &[(String, String)], // (mem_id, verdict)
    excluded_mem_id: &str,
) -> (&'static str, Option<String>) {
    for (id, verdict) in active_rules {
        if id != excluded_mem_id && verdict == "STOP" {
            return ("STOP", Some(id.clone()));
        }
    }
    ("ALLOW", None)
}

/// #551: Ochiai fault-localization failure spectrum score in `[0.0, 1.0]`.
pub fn ochiai_failure_spectrum(ef: u64, nf: u64, ep: u64) -> f64 {
    if ef == 0 {
        return 0.0;
    }
    let denom = ((ef + nf) as f64 * (ef + ep) as f64).sqrt();
    if denom == 0.0 { 0.0 } else { (ef as f64) / denom }
}

/// #552: Lagged cause finding — proposes `candidate_cause` edges with lift statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateCauseStat {
    pub event_sig: String,
    pub failure_sig: String,
    pub support: usize,
    pub lift_delta: f64,
}

pub fn propose_lagged_candidate_causes(
    sessions: &[(BTreeSet<String>, bool)], // (events_in_session, failed)
    failure_sig: &str,
    min_support: usize,
    min_lift: f64,
) -> Vec<CandidateCauseStat> {
    let mut all_events = BTreeSet::new();
    for (evs, _) in sessions {
        for e in evs {
            all_events.insert(e.clone());
        }
    }
    let mut out = Vec::new();
    for ev in all_events {
        let mut with_ev = (0usize, 0usize); // (total, fail)
        let mut without_ev = (0usize, 0usize);
        for (evs, failed) in sessions {
            if evs.contains(&ev) {
                with_ev.0 += 1;
                if *failed {
                    with_ev.1 += 1;
                }
            } else {
                without_ev.0 += 1;
                if *failed {
                    without_ev.1 += 1;
                }
            }
        }
        if with_ev.1 >= min_support && with_ev.0 > 0 {
            let p1 = with_ev.1 as f64 / with_ev.0 as f64;
            let p0 = if without_ev.0 > 0 {
                without_ev.1 as f64 / without_ev.0 as f64
            } else {
                0.0
            };
            let lift = p1 - p0;
            if lift >= min_lift {
                out.push(CandidateCauseStat {
                    event_sig: ev,
                    failure_sig: failure_sig.to_string(),
                    support: with_ev.1,
                    lift_delta: lift,
                });
            }
        }
    }
    out
}

/// #553: Co-allocation edges between memories written close in time in one session.
pub fn mint_coallocation_edges(
    writes: &[(String, u64)], // (mem_id, timestamp_ms)
    max_gap_ms: u64,
) -> Vec<(String, String)> {
    let mut edges = Vec::new();
    for i in 0..writes.len() {
        for j in (i + 1)..writes.len() {
            if writes[j].1.abs_diff(writes[i].1) <= max_gap_ms {
                edges.push((writes[i].0.clone(), writes[j].0.clone()));
            }
        }
    }
    edges
}

/// #554: Reperformance sampling upper error bound (`-ln(alpha) / n` for 0 observed errors).
pub fn reperformance_zero_defect_upper_bound(sample_size: usize, alpha: f64) -> f64 {
    if sample_size == 0 {
        return 1.0;
    }
    (-alpha.ln() / (sample_size as f64)).min(1.0)
}

/// #556: Legal citator flags on architectural decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CitatorFlag {
    Followed,
    Distinguished,
    Questioned,
    Overruled,
}

/// #557: Outbreak alarm when a failure signature spikes across distinct agents.
pub fn detect_outbreak_alarm(
    events: &[(String, String, u64)], // (agent_id, failure_sig, ts_ms)
    window_ms: u64,
    min_distinct_agents: usize,
) -> Vec<String> {
    let mut by_sig: BTreeMap<&str, Vec<(&str, u64)>> = BTreeMap::new();
    for (agent, sig, ts) in events {
        by_sig.entry(sig.as_str()).or_default().push((agent.as_str(), *ts));
    }
    let mut alarms = Vec::new();
    for (sig, mut obs) in by_sig {
        obs.sort_by_key(|(_, ts)| *ts);
        if let (Some((_, first)), Some((_, last))) = (obs.first(), obs.last())
            && last.saturating_sub(*first) <= window_ms
        {
            let agents: BTreeSet<&str> = obs.iter().map(|(a, _)| *a).collect();
            if agents.len() >= min_distinct_agents {
                alarms.push(sig.to_string());
            }
        }
    }
    alarms
}

/// #558: Reconsolidation window — in-place update allowed ONLY right after recall + failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconsolidationGate {
    pub recalled_at_seq: Option<u64>,
    pub failure_after_recall: bool,
    pub window_seqs: u64,
}

impl ReconsolidationGate {
    pub fn may_reconsolidate(&self, current_seq: u64) -> bool {
        match self.recalled_at_seq {
            Some(r_seq) => {
                self.failure_after_recall
                    && current_seq >= r_seq
                    && (current_seq - r_seq) <= self.window_seqs
            }
            None => false,
        }
    }
}

/// #561: Deterministic simulation of crash + owner takeover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimStep {
    AppendDurable(String),
    TornUnackedCrash(String),
    OwnerTakeoverOverride(String),
}

pub fn simulate_crash_and_takeover(steps: &[SimStep]) -> Vec<String> {
    let mut durable_log = Vec::new();
    for s in steps {
        match s {
            SimStep::AppendDurable(v) => durable_log.push(v.clone()),
            SimStep::TornUnackedCrash(_) => {
                // Torn unacked tail is discarded cleanly on recovery; durable prefix survives!
            }
            SimStep::OwnerTakeoverOverride(v) => durable_log.push(format!("OWNER:{v}")),
        }
    }
    durable_log
}

/// #562: Zero-knowledge hiding replay receipt commitment `H(trace || blinding_factor)`.
pub fn commit_zk_replay_receipt(
    private_trace: &[u8],
    blinding_nonce: &[u8; 32],
    policy_hash: &[u8; 32],
    passed: bool,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new_keyed(blinding_nonce);
    h.update(b"strata-zk-replay-v1:");
    h.update(policy_hash);
    h.update(&[u8::from(passed)]);
    h.update(private_trace);
    *h.finalize().as_bytes()
}

/// #564: Multi-agent failure subtype classifier (published MAST taxonomy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiAgentFailureSubtype {
    StepRepetitionLoop,
    UnverifiedTermination,
    SubagentContextLoss,
    PrivilegeEscalationAttempt,
    SpecificationDrift,
}

pub fn classify_failure_subtype(
    repeated_calls: usize,
    missing_exec_receipt: bool,
    dropped_handles: usize,
    denied_escalation: bool,
) -> MultiAgentFailureSubtype {
    if denied_escalation {
        MultiAgentFailureSubtype::PrivilegeEscalationAttempt
    } else if repeated_calls >= 3 {
        MultiAgentFailureSubtype::StepRepetitionLoop
    } else if dropped_handles > 0 {
        MultiAgentFailureSubtype::SubagentContextLoss
    } else if missing_exec_receipt {
        MultiAgentFailureSubtype::UnverifiedTermination
    } else {
        MultiAgentFailureSubtype::SpecificationDrift
    }
}

/// #572: Benchmark `causal_walk` on labeled root-cause incident graphs (`Recall@1` & `MRR`).
pub fn benchmark_causal_walk_mrr(
    cases: &[(Vec<String>, String)], // (ranked_walk_roots, true_root_cause)
) -> (f64, f64) {
    if cases.is_empty() {
        return (0.0, 0.0);
    }
    let mut r1 = 0usize;
    let mut rr_sum = 0.0;
    for (ranked, truth) in cases {
        if let Some(pos) = ranked.iter().position(|x| x == truth) {
            if pos == 0 {
                r1 += 1;
            }
            rr_sum += 1.0 / ((pos + 1) as f64);
        }
    }
    let n = cases.len() as f64;
    (r1 as f64 / n, rr_sum / n)
}

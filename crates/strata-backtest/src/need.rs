//! Successor-representation Need over recorded transitions.
//!
//! T is learned only from the prefix. The first observation of a row is a
//! one-hot. Later observations use the delta rule `α_T = 0.9`. Unobserved
//! rows are self-loops. Need is a truncated Neumann series, `i = 0..=K`.

use std::collections::{BTreeMap, BTreeSet};

use crate::events::{Body, Rec, candidates, card_handle, in_dream, suffix_subjects};
use crate::mechanism::{Mechanism, Prefix, Proof, Query, Subject, proof_of};
use crate::metrics::{
    auc, bootstrap_mean, bootstrap_paired_diff, claim_delta, ln, ndcg_at_k, order_by_score,
    suffix_log_likelihood,
};
use crate::protocol::{
    ALPHA_T, FIG6C_BINS, GAMMA, HORIZON, K_NEUMANN, NEED_ARMS, STATIONARY_ITERS, is_typed_link,
};
use crate::rng::{SplitMix64, fisher_yates, subseed};

/// Learned transition model for one prefix.
#[derive(Clone, Debug)]
pub struct TransitionModel {
    subjects: Vec<Subject>,
    index: BTreeMap<Subject, usize>,
    /// `t[from][to]`.
    t: Vec<Vec<f64>>,
    observed: Vec<bool>,
    proof: Proof,
}

/// SR Need from `current`. Missing or unknown current state ⇒ all zeros.
pub fn sr_need(model: &TransitionModel, current: Option<&Subject>) -> BTreeMap<Subject, f64> {
    let Some(current) = current else {
        return zeros(model);
    };
    let Some(&row) = model.index.get(current) else {
        return zeros(model);
    };
    map_row(model, &neumann(&model.t, row))
}

/// Stationary Need. Independent of the current state.
pub fn stationary_need(model: &TransitionModel) -> BTreeMap<Subject, f64> {
    let n = model.subjects.len();
    if n == 0 {
        return BTreeMap::new();
    }
    let mut mu = vec![1.0 / n as f64; n];
    for _ in 0..STATIONARY_ITERS {
        mu = matvec(&model.t, &mu);
        let sum: f64 = mu.iter().sum();
        if sum > 0.0 {
            for value in &mut mu {
                *value /= sum;
            }
        }
    }
    let mass = geometric_mass();
    for value in &mut mu {
        *value *= mass;
    }
    map_row(model, &mu)
}

/// `(p_max, entropy)` for every row that was actually observed.
pub fn fig6c_observations(model: &TransitionModel) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for (row, seen) in model.observed.iter().enumerate() {
        if !seen {
            continue;
        }
        let mut p_max = 0.0;
        for value in &model.t[row] {
            if *value > p_max {
                p_max = *value;
            }
        }
        let need = neumann(&model.t, row);
        if let Some(entropy) = shannon(&need) {
            out.push((p_max, entropy));
        }
    }
    out
}

/// Bin id for a maximum outgoing probability.
pub fn fig6c_bin(p_max: f64) -> &'static str {
    if p_max < 0.4 {
        FIG6C_BINS[0]
    } else if p_max < 0.7 {
        FIG6C_BINS[1]
    } else {
        FIG6C_BINS[2]
    }
}

/// Last prefix visit, dream reviews removed.
pub fn current_state(prefix: &Prefix) -> Option<Subject> {
    let cards = card_map(&prefix.events);
    let scopes = scope_map(&prefix.events);
    let mut last: Option<(u64, u32, Subject)> = None;
    for rec in &prefix.events {
        let subject = visit_subject(rec, &cards, &scopes, &prefix.windows);
        let Some(subject) = subject else {
            continue;
        };
        let replace = match &last {
            None => true,
            Some((seq, ord, _)) => {
                rec.frame_seq > *seq || (rec.frame_seq == *seq && rec.ord >= *ord)
            }
        };
        if replace {
            last = Some((rec.frame_seq, rec.ord, subject));
        }
    }
    last.map(|(_, _, subject)| subject)
}

/// Build T from the prefix. Anchor paths do not need an edge; node pairs do.
pub fn model_from_prefix(prefix: &Prefix) -> TransitionModel {
    let subjects: Vec<Subject> = candidates(&prefix.events, prefix.bound_seq)
        .into_iter()
        .collect();
    let index: BTreeMap<Subject, usize> = subjects
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, subject)| (subject, i))
        .collect();
    let n = subjects.len();
    let mut t = vec![vec![0.0; n]; n];
    let mut observed = vec![false; n];
    let mut cited = Vec::new();
    for step in transitions(prefix) {
        let Some(&from) = index.get(&step.from) else {
            continue;
        };
        let Some(&to) = index.get(&step.to) else {
            continue;
        };
        cited.push(step.from_seq);
        cited.push(step.to_seq);
        if let Some(edge_seq) = step.edge_seq {
            cited.push(edge_seq);
        }
        if !observed[from] {
            t[from][to] = 1.0;
            observed[from] = true;
        } else {
            let keep = 1.0 - ALPHA_T;
            for value in &mut t[from] {
                *value *= keep;
            }
            t[from][to] += ALPHA_T;
        }
    }
    for row in 0..n {
        if !observed[row] {
            t[row][row] = 1.0;
        }
    }
    TransitionModel {
        subjects,
        index,
        t,
        observed,
        proof: proof_of(cited),
    }
}

/// Scores for one Need arm. Unknown ids yield an empty map.
pub fn arm_scores(arm: &str, prefix: &Prefix, corpus_id: &str) -> BTreeMap<Subject, f64> {
    let cand = candidates(&prefix.events, prefix.bound_seq);
    match arm {
        "sr_need" => sr_need(&model_from_prefix(prefix), current_state(prefix).as_ref()),
        "stationary_need" => stationary_need(&model_from_prefix(prefix)),
        "fsrs_r" => cand
            .into_iter()
            .map(|subject| {
                let score = match &subject {
                    Subject::Node(id) => prefix.retrievability.get(id).copied().unwrap_or(0.0),
                    Subject::Path(_) => 0.0,
                };
                (subject, score)
            })
            .collect(),
        "recency" => cand
            .into_iter()
            .map(|subject| {
                let score = recency(&prefix.events, &subject) as f64;
                (subject, score)
            })
            .collect(),
        "degree" => {
            let counts = degrees(&prefix.events);
            cand.into_iter()
                .map(|subject| {
                    let score = counts.get(&subject).copied().unwrap_or(0) as f64;
                    (subject, score)
                })
                .collect()
        }
        "uniform" => cand.into_iter().map(|subject| (subject, 0.0)).collect(),
        "random" => random_scores(&cand, corpus_id, prefix.bound_seq),
        _ => BTreeMap::new(),
    }
}

/// SR Need, as a [`Mechanism`].
#[derive(Clone, Debug, Default)]
pub struct SrNeed;

/// Stationary Need.
#[derive(Clone, Debug, Default)]
pub struct StationaryNeed;

/// FSRS retrievability at the prefix head clock.
#[derive(Clone, Debug, Default)]
pub struct FsrsR;

/// Last mentioning frame seq.
#[derive(Clone, Debug, Default)]
pub struct Recency;

/// Typed-edge degree.
#[derive(Clone, Debug, Default)]
pub struct Degree;

/// Identical scores, canon order.
#[derive(Clone, Debug, Default)]
pub struct Uniform;

/// Seeded Fisher–Yates.
#[derive(Clone, Debug, Default)]
pub struct SeededRandom;

impl Mechanism for SrNeed {
    fn id(&self) -> &'static str {
        "sr_need"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_with_model_proof("sr_need", prefix, query)
    }
}

impl Mechanism for StationaryNeed {
    fn id(&self) -> &'static str {
        "stationary_need"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_with_model_proof("stationary_need", prefix, query)
    }
}

impl Mechanism for FsrsR {
    fn id(&self) -> &'static str {
        "fsrs_r"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_scored("fsrs_r", prefix, query, |prefix, subject| {
            proof_of(history_seqs(prefix, subject))
        })
    }
}

impl Mechanism for Recency {
    fn id(&self) -> &'static str {
        "recency"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_scored("recency", prefix, query, |prefix, subject| {
            proof_of([recency(&prefix.events, subject)])
        })
    }
}

impl Mechanism for Degree {
    fn id(&self) -> &'static str {
        "degree"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_scored("degree", prefix, query, |prefix, subject| {
            proof_of(degree_seqs(&prefix.events, subject))
        })
    }
}

impl Mechanism for Uniform {
    fn id(&self) -> &'static str {
        "uniform"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_scored("uniform", prefix, query, |prefix, subject| {
            proof_of(existence(&prefix.events, subject))
        })
    }
}

impl Mechanism for SeededRandom {
    fn id(&self) -> &'static str {
        "random"
    }
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
        rank_scored("random", prefix, query, |prefix, subject| {
            proof_of(existence(&prefix.events, subject))
        })
    }
}

/// One cut's predictive numbers, in [`NEED_ARMS`] order. `None` is undefined.
#[derive(Clone, Debug)]
pub struct CutPrediction {
    /// AUC per arm.
    pub auc: Vec<Option<f64>>,
    /// NDCG@5 per arm.
    pub ndcg: Vec<Option<f64>>,
    /// Suffix log-likelihood per arm.
    pub ll: Vec<Option<f64>>,
    /// Fig. 6c observations from this cut's observed rows.
    pub fig6c: Vec<(f64, f64)>,
}

/// Score every Need arm at this prefix against suffix occupancy.
pub fn predict_cut(prefix: &Prefix, all_events: &[Rec], corpus_id: &str) -> CutPrediction {
    let cand = candidates(&prefix.events, prefix.bound_seq);
    let positives: BTreeSet<Subject> =
        suffix_subjects(all_events, prefix.bound_seq, HORIZON, &prefix.windows)
            .into_iter()
            .filter(|subject| cand.contains(subject))
            .collect();
    let model = model_from_prefix(prefix);
    let mut aucs = Vec::with_capacity(NEED_ARMS.len());
    let mut ndcgs = Vec::with_capacity(NEED_ARMS.len());
    let mut lls = Vec::with_capacity(NEED_ARMS.len());
    for arm in NEED_ARMS {
        let scores = match arm {
            "sr_need" => sr_need(&model, current_state(prefix).as_ref()),
            "stationary_need" => stationary_need(&model),
            other => arm_scores(other, prefix, corpus_id),
        };
        let order = order_by_score(&scores);
        aucs.push(auc(&scores, &positives, &cand));
        ndcgs.push(ndcg_at_k(&order, &positives, &cand));
        lls.push(suffix_log_likelihood(&order, &positives));
    }
    CutPrediction {
        auc: aucs,
        ndcg: ndcgs,
        ll: lls,
        fig6c: fig6c_observations(&model),
    }
}

/// Aggregated Need comparison for one corpus.
#[derive(Clone, Debug)]
pub struct NeedReport {
    /// Arm id, then mean and CI for AUC, NDCG, and LL. `n` is defined AUC cuts.
    pub arms: Vec<NeedArmReport>,
    /// `sr_need` AUC minus `fsrs_r` AUC on cuts where both exist.
    pub sr_minus_fsrs: f64,
    /// CI lower bound of that difference.
    pub sr_minus_lo: f64,
    /// CI upper bound of that difference.
    pub sr_minus_hi: f64,
    /// Claim id from the preregistered rule.
    pub claim: &'static str,
    /// `(bin, n, mean entropy)` in bin order. Entropy is 0 when `n` is 0.
    pub fig6c: Vec<(&'static str, u64, f64)>,
}

/// One arm's aggregated predictive metrics.
#[derive(Clone, Debug)]
pub struct NeedArmReport {
    /// Arm id.
    pub id: &'static str,
    /// Mean AUC. 0 when `n` is 0.
    pub auc: f64,
    /// AUC CI lower bound.
    pub auc_lo: f64,
    /// AUC CI upper bound.
    pub auc_hi: f64,
    /// Mean NDCG.
    pub ndcg: f64,
    /// NDCG CI lower bound.
    pub ndcg_lo: f64,
    /// NDCG CI upper bound.
    pub ndcg_hi: f64,
    /// Mean suffix log-likelihood.
    pub ll: f64,
    /// LL CI lower bound.
    pub ll_lo: f64,
    /// LL CI upper bound.
    pub ll_hi: f64,
    /// Cuts with a defined AUC.
    pub n: u64,
}

/// Bootstrap the per-cut predictions. `corpus_id` names the subseeds.
pub fn aggregate_need(corpus_id: &str, cuts: &[CutPrediction]) -> NeedReport {
    let mut arms = Vec::with_capacity(NEED_ARMS.len());
    for (index, id) in NEED_ARMS.iter().enumerate() {
        let auc_vals = defined(cuts, |cut| cut.auc[index]);
        let ndcg_vals = defined(cuts, |cut| cut.ndcg[index]);
        let ll_vals = defined(cuts, |cut| cut.ll[index]);
        let (auc_mean, auc_lo, auc_hi) =
            bootstrap_mean(&auc_vals, &format!("boot/{corpus_id}/need/{id}/auc"));
        let (ndcg_mean, ndcg_lo, ndcg_hi) =
            bootstrap_mean(&ndcg_vals, &format!("boot/{corpus_id}/need/{id}/ndcg"));
        let (ll_mean, ll_lo, ll_hi) =
            bootstrap_mean(&ll_vals, &format!("boot/{corpus_id}/need/{id}/ll"));
        arms.push(NeedArmReport {
            id,
            auc: auc_mean,
            auc_lo,
            auc_hi,
            ndcg: ndcg_mean,
            ndcg_lo,
            ndcg_hi,
            ll: ll_mean,
            ll_lo,
            ll_hi,
            n: u64::try_from(auc_vals.len()).unwrap_or(u64::MAX),
        });
    }
    let sr = NEED_ARMS
        .iter()
        .position(|id| *id == "sr_need")
        .unwrap_or(0);
    let fsrs = NEED_ARMS.iter().position(|id| *id == "fsrs_r").unwrap_or(0);
    let mut sr_vals = Vec::new();
    let mut fsrs_vals = Vec::new();
    for cut in cuts {
        if let (Some(left), Some(right)) = (cut.auc[sr], cut.auc[fsrs]) {
            sr_vals.push(left);
            fsrs_vals.push(right);
        }
    }
    let (diff, lo, hi) = bootstrap_paired_diff(
        &sr_vals,
        &fsrs_vals,
        &format!("boot/{corpus_id}/need/sr_minus_fsrs_auc"),
    );
    let claim = claim_delta(
        diff,
        lo,
        hi,
        "sr_need_beats_fsrs_r",
        "fsrs_r_beats_sr_need",
        sr_vals.len(),
    );
    let mut observations = Vec::new();
    for cut in cuts {
        observations.extend_from_slice(&cut.fig6c);
    }
    let mut fig6c = Vec::with_capacity(FIG6C_BINS.len());
    for bin in FIG6C_BINS {
        let values: Vec<f64> = observations
            .iter()
            .filter(|(p_max, _)| fig6c_bin(*p_max) == bin)
            .map(|(_, entropy)| *entropy)
            .collect();
        let n = u64::try_from(values.len()).unwrap_or(u64::MAX);
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        fig6c.push((bin, n, mean));
    }
    NeedReport {
        arms,
        sr_minus_fsrs: diff,
        sr_minus_lo: lo,
        sr_minus_hi: hi,
        claim,
        fig6c,
    }
}

fn defined(cuts: &[CutPrediction], pick: impl Fn(&CutPrediction) -> Option<f64>) -> Vec<f64> {
    cuts.iter().filter_map(&pick).collect()
}

fn rank_with_model_proof(arm: &str, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)> {
    let model = model_from_prefix(prefix);
    let scores = arm_scores(arm, prefix, &query.corpus_id);
    order_by_score(&scores)
        .into_iter()
        .map(|subject| (subject, model.proof.clone()))
        .collect()
}

fn rank_scored(
    arm: &str,
    prefix: &Prefix,
    query: &Query,
    proof: impl Fn(&Prefix, &Subject) -> Proof,
) -> Vec<(Subject, Proof)> {
    let scores = arm_scores(arm, prefix, &query.corpus_id);
    order_by_score(&scores)
        .into_iter()
        .map(|subject| {
            let cited = proof(prefix, &subject);
            (subject, cited)
        })
        .collect()
}

fn zeros(model: &TransitionModel) -> BTreeMap<Subject, f64> {
    model
        .subjects
        .iter()
        .cloned()
        .map(|subject| (subject, 0.0))
        .collect()
}

fn map_row(model: &TransitionModel, row: &[f64]) -> BTreeMap<Subject, f64> {
    model
        .subjects
        .iter()
        .cloned()
        .zip(row.iter().copied())
        .collect()
}

fn geometric_mass() -> f64 {
    let mut mass = 0.0;
    let mut gamma = 1.0;
    for _ in 0..=K_NEUMANN {
        mass += gamma;
        gamma *= GAMMA;
    }
    mass
}

fn neumann(t: &[Vec<f64>], row: usize) -> Vec<f64> {
    let n = t.len();
    let mut v = vec![0.0; n];
    if n == 0 {
        return v;
    }
    v[row] = 1.0;
    let mut acc = v.clone();
    let mut gamma = 1.0;
    for _ in 0..K_NEUMANN {
        gamma *= GAMMA;
        v = matvec(t, &v);
        for (slot, value) in acc.iter_mut().zip(v.iter()) {
            *slot += gamma * value;
        }
    }
    acc
}

fn matvec(t: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0; n];
    for target in 0..n {
        let mut acc = 0.0;
        for source in 0..n {
            acc += v[source] * t[source][target];
        }
        out[target] = acc;
    }
    out
}

fn shannon(need: &[f64]) -> Option<f64> {
    let sum: f64 = need.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    let mut entropy = 0.0;
    for value in need {
        if *value > 0.0 {
            let p = value / sum;
            entropy -= p * ln(p);
        }
    }
    Some(entropy)
}

struct Step {
    from_seq: u64,
    to_seq: u64,
    to_ord: u32,
    from: Subject,
    to: Subject,
    edge_seq: Option<u64>,
}

fn transitions(prefix: &Prefix) -> Vec<Step> {
    let mut steps = Vec::new();
    let cards = card_map(&prefix.events);
    let scopes = scope_map(&prefix.events);
    let mut by_scope: BTreeMap<String, Vec<(u64, u32, Subject)>> = BTreeMap::new();
    let mut paths = Vec::new();
    for rec in &prefix.events {
        match &rec.body {
            Body::Anchor { path, .. } => {
                paths.push((rec.frame_seq, rec.ord, Subject::Path(path.clone())));
            }
            Body::Upsert { .. } | Body::Review { .. } => {
                if let Some(subject) = visit_subject(rec, &cards, &scopes, &prefix.windows) {
                    if let Subject::Node(id) = &subject {
                        let scope = scopes.get(id).cloned().unwrap_or_default();
                        by_scope
                            .entry(scope)
                            .or_default()
                            .push((rec.frame_seq, rec.ord, subject));
                    }
                }
            }
            Body::Edge { .. } => {}
        }
    }
    for visits in by_scope.values() {
        push_distinct(visits, true, &prefix.events, &mut steps);
    }
    push_distinct(&paths, false, &prefix.events, &mut steps);
    steps.sort_by(|left, right| {
        left.to_seq
            .cmp(&right.to_seq)
            .then(left.to_ord.cmp(&right.to_ord))
            .then(left.from.cmp(&right.from))
            .then(left.to.cmp(&right.to))
    });
    steps
}

fn push_distinct(
    visits: &[(u64, u32, Subject)],
    require_edge: bool,
    events: &[Rec],
    out: &mut Vec<Step>,
) {
    let mut prev: Option<(u64, u32, Subject)> = None;
    for visit in visits {
        if let Some((from_seq, _, from)) = &prev {
            if from != &visit.2 {
                let edge_seq = if require_edge {
                    joining_edge(events, from, &visit.2, visit.0)
                } else {
                    None
                };
                if !require_edge || edge_seq.is_some() {
                    out.push(Step {
                        from_seq: *from_seq,
                        to_seq: visit.0,
                        to_ord: visit.1,
                        from: from.clone(),
                        to: visit.2.clone(),
                        edge_seq,
                    });
                }
            }
        }
        prev = Some(visit.clone());
    }
}

fn joining_edge(events: &[Rec], from: &Subject, to: &Subject, later_seq: u64) -> Option<u64> {
    let (Subject::Node(left), Subject::Node(right)) = (from, to) else {
        return None;
    };
    let mut best: Option<u64> = None;
    for rec in events {
        if rec.frame_seq > later_seq {
            break;
        }
        if let Body::Edge {
            source,
            target,
            link,
        } = &rec.body
        {
            if !is_typed_link(link) {
                continue;
            }
            let joins = (source == left && target == right) || (source == right && target == left);
            if joins {
                best = Some(rec.frame_seq);
            }
        }
    }
    best
}

fn visit_subject(
    rec: &Rec,
    cards: &BTreeMap<u64, String>,
    scopes: &BTreeMap<String, String>,
    windows: &[crate::events::DreamWindow],
) -> Option<Subject> {
    match &rec.body {
        Body::Upsert { id, .. } => Some(Subject::Node(id.clone())),
        Body::Review { card_id, .. } => {
            if in_dream(rec.frame_seq, windows) {
                return None;
            }
            let id = cards.get(card_id)?;
            if !scopes.contains_key(id) {
                return None;
            }
            Some(Subject::Node(id.clone()))
        }
        Body::Anchor { path, .. } => Some(Subject::Path(path.clone())),
        Body::Edge { .. } => None,
    }
}

fn card_map(events: &[Rec]) -> BTreeMap<u64, String> {
    let mut map = BTreeMap::new();
    for rec in events {
        if let Body::Upsert { id, .. } = &rec.body {
            map.insert(card_handle(id), id.clone());
        }
    }
    map
}

fn scope_map(events: &[Rec]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for rec in events {
        if let Body::Upsert { id, scope, .. } = &rec.body {
            map.insert(id.clone(), scope.clone());
        }
    }
    map
}

fn recency(events: &[Rec], subject: &Subject) -> u64 {
    let mut best = 0;
    for rec in events {
        if mentions(rec, subject) && rec.frame_seq >= best {
            best = rec.frame_seq;
        }
    }
    best
}

fn mentions(rec: &Rec, subject: &Subject) -> bool {
    match (&rec.body, subject) {
        (Body::Upsert { id, .. }, Subject::Node(node)) => id == node,
        (Body::Review { card_id, .. }, Subject::Node(node)) => *card_id == card_handle(node),
        (Body::Edge { source, target, .. }, Subject::Node(node)) => {
            source == node || target == node
        }
        (Body::Edge { source, target, .. }, Subject::Path(path)) => {
            source == path || target == path
        }
        (Body::Anchor { node_id, .. }, Subject::Node(node)) => node_id == node,
        (Body::Anchor { path, .. }, Subject::Path(want)) => path == want,
        _ => false,
    }
}

fn degrees(events: &[Rec]) -> BTreeMap<Subject, u64> {
    let nodes: BTreeSet<String> = events
        .iter()
        .filter_map(|rec| match &rec.body {
            Body::Upsert { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    let mut counts = BTreeMap::new();
    for rec in events {
        if let Body::Edge {
            source,
            target,
            link,
        } = &rec.body
        {
            if !is_typed_link(link) {
                continue;
            }
            let bump = |counts: &mut BTreeMap<Subject, u64>, id: &str| {
                let subject = if nodes.contains(id) {
                    Subject::Node(id.to_string())
                } else {
                    Subject::Path(id.to_string())
                };
                *counts.entry(subject).or_insert(0) += 1;
            };
            bump(&mut counts, source);
            if source != target {
                bump(&mut counts, target);
            }
        }
    }
    counts
}

fn degree_seqs(events: &[Rec], subject: &Subject) -> Vec<u64> {
    let mut seqs = Vec::new();
    for rec in events {
        if let Body::Edge {
            source,
            target,
            link,
        } = &rec.body
        {
            if is_typed_link(link) && touches(subject, source, target, events) {
                seqs.push(rec.frame_seq);
            }
        }
    }
    seqs
}

fn touches(subject: &Subject, source: &str, target: &str, events: &[Rec]) -> bool {
    match subject {
        Subject::Node(id) => source == id || target == id,
        Subject::Path(path) => {
            let nodes: BTreeSet<&str> = events
                .iter()
                .filter_map(|rec| match &rec.body {
                    Body::Upsert { id, .. } => Some(id.as_str()),
                    _ => None,
                })
                .collect();
            (source == path && !nodes.contains(source))
                || (target == path && !nodes.contains(target))
        }
    }
}

fn history_seqs(prefix: &Prefix, subject: &Subject) -> Vec<u64> {
    let Subject::Node(id) = subject else {
        return Vec::new();
    };
    let handle = card_handle(id);
    prefix
        .events
        .iter()
        .filter_map(|rec| match &rec.body {
            Body::Upsert { id: node, .. } if node == id => Some(rec.frame_seq),
            Body::Review { card_id, .. } if *card_id == handle => Some(rec.frame_seq),
            _ => None,
        })
        .collect()
}

fn existence(events: &[Rec], subject: &Subject) -> Vec<u64> {
    events
        .iter()
        .filter_map(|rec| mentions(rec, subject).then_some(rec.frame_seq))
        .take(1)
        .collect()
}

fn random_scores(cand: &BTreeSet<Subject>, corpus_id: &str, bound: u64) -> BTreeMap<Subject, f64> {
    let mut items: Vec<Subject> = cand.iter().cloned().collect();
    let tag = format!("need-random/{corpus_id}/{bound}");
    fisher_yates(&mut items, &mut SplitMix64::new(subseed(&tag)));
    let n = items.len();
    items
        .into_iter()
        .enumerate()
        .map(|(rank, subject)| (subject, (n - rank) as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::quantize;
    use crate::events::DreamWindow;
    use crate::protocol::K_NEUMANN;
    use strata_kernel::canonical::Q_ONE;

    fn upsert(seq: u64, id: &str) -> Rec {
        Rec {
            frame_seq: seq,
            ord: 0,
            body: Body::Upsert {
                id: id.into(),
                scope: "s".into(),
                node_type: "fact".into(),
                tags: Vec::new(),
            },
        }
    }

    fn edge(seq: u64, source: &str, target: &str) -> Rec {
        Rec {
            frame_seq: seq,
            ord: 0,
            body: Body::Edge {
                source: source.into(),
                target: target.into(),
                link: "derived_from".into(),
            },
        }
    }

    fn review(seq: u64, id: &str, rating: u8) -> Rec {
        Rec {
            frame_seq: seq,
            ord: 0,
            body: Body::Review {
                card_id: card_handle(id),
                rating,
            },
        }
    }

    fn prefix(events: Vec<Rec>) -> Prefix {
        let bound = events.last().map(|rec| rec.frame_seq).unwrap_or(0);
        Prefix {
            bound_seq: bound,
            head_seq: bound,
            head_frame_hash: [0; 32],
            state_digest: [0; 32],
            head_clock_ms: 0,
            events,
            retrievability: BTreeMap::new(),
            windows: Vec::new(),
        }
    }

    #[test]
    fn two_state_need_matches_the_closed_form() {
        let model = model_from_prefix(&prefix(vec![
            upsert(1, "A"),
            edge(2, "A", "B"),
            upsert(3, "B"),
        ]));
        let need = sr_need(&model, Some(&Subject::Node("A".into())));
        let mut expect_b = 0.0;
        let mut gamma = GAMMA;
        for _ in 1..=K_NEUMANN {
            expect_b += gamma;
            gamma *= GAMMA;
        }
        assert_eq!(quantize(need[&Subject::Node("A".into())]), Q_ONE);
        assert_eq!(
            quantize(need[&Subject::Node("B".into())]),
            quantize(expect_b)
        );
        let row_a = &model.t[model.index[&Subject::Node("A".into())]];
        let row_b = &model.t[model.index[&Subject::Node("B".into())]];
        assert_eq!(row_a, &vec![0.0, 1.0]);
        assert_eq!(row_b, &vec![0.0, 1.0]);
    }

    #[test]
    fn second_distinct_successor_applies_the_delta_rule() {
        let model = model_from_prefix(&prefix(vec![
            upsert(1, "A"),
            edge(2, "A", "B"),
            upsert(3, "B"),
            edge(4, "A", "C"),
            upsert(5, "C"),
            review(6, "A", 3),
            review(7, "C", 3),
        ]));
        let idx = &model.index;
        let row = &model.t[idx[&Subject::Node("A".into())]];
        let b = idx[&Subject::Node("B".into())];
        let c = idx[&Subject::Node("C".into())];
        assert!((row[b] - 0.1).abs() < 1e-12);
        assert!((row[c] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn repeated_visits_do_not_invent_a_self_transition() {
        let model = model_from_prefix(&prefix(vec![
            upsert(1, "A"),
            edge(2, "A", "A"),
            review(3, "A", 3),
        ]));
        assert!(!model.observed[model.index[&Subject::Node("A".into())]]);
    }

    #[test]
    fn anchor_paths_transition_without_an_edge() {
        let model = model_from_prefix(&prefix(vec![
            Rec {
                frame_seq: 1,
                ord: 0,
                body: Body::Anchor {
                    node_id: "n".into(),
                    path: "src/a.rs".into(),
                },
            },
            Rec {
                frame_seq: 1,
                ord: 1,
                body: Body::Anchor {
                    node_id: "n".into(),
                    path: "src/b.rs".into(),
                },
            },
            upsert(2, "n"),
        ]));
        let from = model.index[&Subject::Path("src/a.rs".into())];
        let to = model.index[&Subject::Path("src/b.rs".into())];
        assert!((model.t[from][to] - 1.0).abs() < 1e-12);
        assert!(model.proof.cited_seqs.iter().all(|seq| *seq <= 2));
    }

    #[test]
    fn dream_review_is_not_a_visit() {
        let mut built = prefix(vec![
            upsert(1, "A"),
            edge(2, "A", "B"),
            upsert(3, "B"),
            review(4, "A", 3),
        ]);
        built.windows = vec![DreamWindow {
            start_seq: 4,
            end_seq: 5,
        }];
        assert_eq!(current_state(&built), Some(Subject::Node("B".into())));
        built.windows.clear();
        assert_eq!(current_state(&built), Some(Subject::Node("A".into())));
    }

    #[test]
    fn sr_need_ranks_the_recorded_successor_above_a_stranger() {
        let built = prefix(vec![
            upsert(1, "S"),
            upsert(2, "Z"),
            edge(3, "S", "A"),
            upsert(4, "A"),
            review(5, "S", 3),
            review(6, "A", 3),
            review(7, "S", 3),
        ]);
        let scores = sr_need(&model_from_prefix(&built), current_state(&built).as_ref());
        assert!(scores[&Subject::Node("A".into())] > scores[&Subject::Node("Z".into())]);
        assert_eq!(current_state(&built), Some(Subject::Node("S".into())));
    }

    #[test]
    fn stationary_need_ignores_which_state_is_current() {
        let built = prefix(vec![upsert(1, "A"), edge(2, "A", "B"), upsert(3, "B")]);
        let once = stationary_need(&model_from_prefix(&built));
        let twice = stationary_need(&model_from_prefix(&built));
        assert_eq!(
            quantize(once[&Subject::Node("A".into())]),
            quantize(twice[&Subject::Node("A".into())])
        );
        let observations = fig6c_observations(&model_from_prefix(&built));
        assert!(
            observations
                .iter()
                .any(|(p_max, _)| fig6c_bin(*p_max) == "ge_0.7")
        );
    }

    #[test]
    fn firewall_holds_on_every_need_arm() {
        let built = prefix(vec![upsert(1, "A"), edge(2, "A", "B"), upsert(3, "B")]);
        let query = Query {
            corpus_id: "unit".into(),
            bound_seq: built.bound_seq,
            horizon: HORIZON,
        };
        for mechanism in [
            Box::new(SrNeed) as Box<dyn Mechanism>,
            Box::new(StationaryNeed),
            Box::new(FsrsR),
            Box::new(Recency),
            Box::new(Degree),
            Box::new(Uniform),
            Box::new(SeededRandom),
        ] {
            let ranked = mechanism.rank(&built, &query);
            crate::mechanism::require_firewall(built.bound_seq, &ranked).unwrap();
            assert_eq!(ranked.len(), 2);
        }
    }
}

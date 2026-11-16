//! Gain × Need scheduler. The Q layer is the model under test.
//!
//! A backup is the paper's Bellman operator `Q(s,a) ← r + γ max Q(s', ·)`,
//! not a policy-expectation backup. Gain is equation 5. Every backup has
//! length 1.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::events::{Body, Rec, card_handle, in_dream, suffix_review_nodes};
use crate::mechanism::{Prefix, Proof, Subject, proof_of};
use crate::metrics::{bootstrap_mean, bootstrap_paired_diff, claim_delta};
use crate::need::{current_state, model_from_prefix, sr_need};
use crate::protocol::{
    BETA, BUDGETS, COMPOSITION_NODE_TYPE, EVB_ARMS, EXPERIENCE_LINKS, GAMMA, GHOSTLINK_TAG,
    MEAN_BUDGETS, MIN_GAIN, OUTCOME_TAG_PREFIX, WEAVE_TAG, outcome_sign,
};
use crate::rng::{SplitMix64, subseed};

/// One recorded experience the scheduler can back up.
#[derive(Clone, Debug)]
pub struct Experience {
    /// Edge frame.
    pub frame_seq: u64,
    /// `link_type`.
    pub link: String,
    /// State the backup updates.
    pub state: Subject,
    /// Surfaced successor.
    pub action: Subject,
    /// Successor state. Equal to `action` under the pinned orientation.
    pub successor: Subject,
    /// Kind bonus plus the successor's node reward.
    pub reward: f64,
    /// Tie-break id.
    pub id: String,
}

/// Node reward from prefix reviews, weaves, and penalty edges.
pub fn node_rewards(prefix: &Prefix) -> BTreeMap<String, i32> {
    let cards = cards_of(&prefix.events);
    let mut reward: BTreeMap<String, i32> = BTreeMap::new();
    for rec in &prefix.events {
        if in_dream(rec.frame_seq, &prefix.windows) {
            continue;
        }
        if let Body::Review { card_id, rating } = &rec.body {
            let Some(id) = cards.get(card_id) else {
                continue;
            };
            match rating {
                4 => {
                    reward.insert(id.clone(), 1);
                }
                1 => {
                    reward.insert(id.clone(), -1);
                }
                _ => {}
            }
        }
    }
    for (id, sign) in weave_signs(&prefix.events) {
        *reward.entry(id).or_insert(0) += sign;
    }
    for rec in &prefix.events {
        if let Body::Edge { target, link, .. } = &rec.body {
            if link == "corrects" || link == "supersedes" {
                *reward.entry(target.clone()).or_insert(0) -= 1;
            }
        }
    }
    reward
}

/// Experiences in log order. Suffix frames are absent from `prefix.events`.
pub fn experiences(prefix: &Prefix) -> Vec<Experience> {
    let rewards = node_rewards(prefix);
    let nodes = node_set(&prefix.events);
    let mut out = Vec::new();
    for rec in &prefix.events {
        let Body::Edge {
            source,
            target,
            link,
        } = &rec.body
        else {
            continue;
        };
        if !EXPERIENCE_LINKS.contains(&link.as_str()) {
            continue;
        }
        let source_s = endpoint(source, &nodes);
        let target_s = endpoint(target, &nodes);
        let (state, action, successor) = if link == "derived_from" {
            (source_s, target_s.clone(), target_s)
        } else {
            (target_s, source_s.clone(), source_s)
        };
        let kind = if link == "closed_by" { 1.0 } else { 0.0 };
        let node = match &successor {
            Subject::Node(id) => rewards.get(id).copied().unwrap_or(0) as f64,
            Subject::Path(_) => 0.0,
        };
        let id = format!(
            "{:016x}|{link}|{}|{}",
            rec.frame_seq,
            state.canon(),
            action.canon()
        );
        out.push(Experience {
            frame_seq: rec.frame_seq,
            link: link.clone(),
            state,
            action,
            successor,
            reward: kind + node,
            id,
        });
    }
    out
}

/// Gain equation 5 for one hypothetical backup. One action ⇒ 0. NaN stays NaN.
pub fn gain_of(
    q: &BTreeMap<(Subject, Subject), f64>,
    exp: &Experience,
    actions: &BTreeMap<Subject, Vec<Subject>>,
) -> f64 {
    let Some(list) = actions.get(&exp.state) else {
        return 0.0;
    };
    if list.len() <= 1 {
        return 0.0;
    }
    let Some(idx) = list.iter().position(|action| action == &exp.action) else {
        return 0.0;
    };
    let q_old: Vec<f64> = list
        .iter()
        .map(|action| q_at(q, &exp.state, action))
        .collect();
    let mut q_new = q_old.clone();
    q_new[idx] = exp.reward + GAMMA * max_q(q, actions, &exp.successor);
    let pi_old = softmax(&q_old);
    let pi_new = softmax(&q_new);
    let mut gain = 0.0;
    for i in 0..list.len() {
        gain += q_new[i] * (pi_new[i] - pi_old[i]);
    }
    gain
}

/// `max(gain, 1e-10)`, and `1e-10` when `gain` is not finite.
pub fn effective_gain(gain: f64) -> f64 {
    if !gain.is_finite() || gain < MIN_GAIN {
        MIN_GAIN
    } else {
        gain
    }
}

/// Returns at each budget in [`BUDGETS`], plus the proofs of the backups.
#[derive(Clone, Debug)]
pub struct ArmRun {
    /// Aligned with [`BUDGETS`].
    pub returns: [f64; 7],
    /// Proofs of the backups that were applied, in order.
    pub proofs: Vec<Proof>,
    /// Eval pairs at this cut. Same for every arm.
    pub n_pairs: usize,
}

/// Run one arm for the full budget grid. The random generator is not reseeded.
pub fn run_arm(arm: &str, prefix: &Prefix, all_events: &[Rec], corpus_id: &str) -> ArmRun {
    let exps = experiences(prefix);
    let actions = action_sets(&exps);
    let reviews = suffix_review_nodes(
        all_events,
        prefix.bound_seq,
        crate::protocol::HORIZON,
        &prefix.windows,
    );
    let current = current_state(prefix);
    let pairs = eval_pairs(current.clone(), &reviews);
    let model = model_from_prefix(prefix);
    let sr = sr_need(&model, current.as_ref());
    let mut q: BTreeMap<(Subject, Subject), f64> = BTreeMap::new();
    let mut touched: BTreeSet<Subject> = BTreeSet::new();
    let mut returns = [0.0; 7];
    returns[0] = discounted(&q, &touched, &actions, &pairs);
    let mut proofs = Vec::new();
    if arm == "no_replay" || exps.is_empty() {
        return ArmRun {
            returns,
            proofs,
            n_pairs: pairs.len(),
        };
    }
    let static_float = static_float_priority(arm, prefix, &exps, &sr, &current);
    let static_int = static_int_priority(arm, &prefix.events, &exps);
    let mut rng = SplitMix64::new(subseed(&format!(
        "evb-random/{corpus_id}/{}",
        prefix.bound_seq
    )));
    let uniform_order = uniform_index(&exps);
    for step in 0..crate::protocol::BACKUPS {
        let choice = choose(
            &Choice {
                arm,
                exps: &exps,
                actions: &actions,
                q: &q,
                sr: &sr,
                current: &current,
                static_float: &static_float,
                static_int: &static_int,
                uniform_order: &uniform_order,
            },
            step,
            &mut rng,
        );
        let exp = &exps[choice];
        let target = exp.reward + GAMMA * max_q(&q, &actions, &exp.successor);
        q.insert((exp.state.clone(), exp.action.clone()), target);
        touched.insert(exp.state.clone());
        proofs.push(backup_proof(prefix, exp, model.proof()));
        let done = step + 1;
        if let Some(slot) = BUDGETS.iter().position(|budget| *budget == done) {
            returns[slot] = discounted(&q, &touched, &actions, &pairs);
        }
    }
    ArmRun {
        returns,
        proofs,
        n_pairs: pairs.len(),
    }
}

/// Mean of the returns on [`MEAN_BUDGETS`].
pub fn mean_return(returns: &[f64; 7]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for (slot, budget) in BUDGETS.iter().enumerate() {
        if MEAN_BUDGETS.contains(budget) {
            sum += returns[slot];
            n += 1.0;
        }
    }
    if n == 0.0 { 0.0 } else { sum / n }
}

/// Smallest budget whose return reaches 90% of the oracle, if any pair exists.
pub fn budget_to_90(returns: &[f64; 7], n_pairs: usize) -> Option<usize> {
    if n_pairs == 0 {
        return None;
    }
    let oracle = oracle_return(n_pairs);
    if oracle == 0.0 {
        return None;
    }
    let threshold = 0.9 * oracle;
    BUDGETS
        .iter()
        .copied()
        .enumerate()
        .find(|(slot, _)| returns[*slot] >= threshold)
        .map(|(_, budget)| budget)
}

/// `Σ γ^i` over `n_pairs`.
pub fn oracle_return(n_pairs: usize) -> f64 {
    let mut sum = 0.0;
    let mut weight = 1.0;
    for _ in 0..n_pairs {
        sum += weight;
        weight *= GAMMA;
    }
    sum
}

/// Per-arm aggregated return curve.
#[derive(Clone, Debug)]
pub struct EvbArmReport {
    /// Arm id.
    pub id: &'static str,
    /// Mean return, CI low, CI high, one row per budget.
    pub by_budget: Vec<(usize, f64, f64, f64)>,
    /// Mean over the primary budgets.
    pub mean: f64,
    /// CI low of that mean.
    pub mean_lo: f64,
    /// CI high of that mean.
    pub mean_hi: f64,
}

/// Corpus-level EVB comparison.
#[derive(Clone, Debug)]
pub struct EvbReport {
    /// Arms in [`EVB_ARMS`] order.
    pub arms: Vec<EvbArmReport>,
    /// Primary difference, EVB minus Gain-only.
    pub diff: f64,
    /// CI low of the difference.
    pub diff_lo: f64,
    /// CI high of the difference.
    pub diff_hi: f64,
    /// Claim id.
    pub claim: &'static str,
    /// Cuts whose EVB return reached 90% of the oracle.
    pub n_reached: u64,
    /// `n_reached / cuts with an eval pair`, or 0 when there are none.
    pub fraction_reached: f64,
    /// Mean reaching budget. 0 when `n_reached` is 0.
    pub mean_budget_to_90: f64,
}

/// Bootstrap the per-cut arm runs. `runs[cut][arm]`.
pub fn aggregate_evb(corpus_id: &str, runs: &[Vec<ArmRun>]) -> EvbReport {
    let mut arms = Vec::with_capacity(EVB_ARMS.len());
    let mut mean_by_arm: Vec<Vec<f64>> = Vec::new();
    for (index, id) in EVB_ARMS.iter().enumerate() {
        let means: Vec<f64> = runs
            .iter()
            .map(|cut| mean_return(&cut[index].returns))
            .collect();
        let (mean, mean_lo, mean_hi) =
            bootstrap_mean(&means, &format!("boot/{corpus_id}/evb/{id}/mean"));
        let mut by_budget = Vec::with_capacity(BUDGETS.len());
        for (slot, budget) in BUDGETS.iter().enumerate() {
            let values: Vec<f64> = runs.iter().map(|cut| cut[index].returns[slot]).collect();
            let (ret, lo, hi) =
                bootstrap_mean(&values, &format!("boot/{corpus_id}/evb/{id}/b{budget}"));
            by_budget.push((*budget, ret, lo, hi));
        }
        mean_by_arm.push(means);
        arms.push(EvbArmReport {
            id,
            by_budget,
            mean,
            mean_lo,
            mean_hi,
        });
    }
    let evb_idx = EVB_ARMS.iter().position(|id| *id == "evb").unwrap_or(0);
    let gain_idx = EVB_ARMS
        .iter()
        .position(|id| *id == "gain_only")
        .unwrap_or(1);
    let (diff, diff_lo, diff_hi) = bootstrap_paired_diff(
        &mean_by_arm[evb_idx],
        &mean_by_arm[gain_idx],
        &format!("boot/{corpus_id}/evb/evb_minus_gain_only"),
    );
    let claim = claim_delta(
        diff,
        diff_lo,
        diff_hi,
        "evb_beats_gain_only",
        "gain_only_beats_evb",
        runs.len(),
    );
    let mut n_with_pairs = 0u64;
    let mut n_reached = 0u64;
    let mut budget_sum = 0.0;
    for cut in runs {
        let n_pairs = cut[evb_idx].n_pairs;
        if n_pairs == 0 {
            continue;
        }
        n_with_pairs += 1;
        if let Some(budget) = budget_to_90(&cut[evb_idx].returns, n_pairs) {
            n_reached += 1;
            budget_sum += budget as f64;
        }
    }
    let fraction_reached = if n_with_pairs == 0 {
        0.0
    } else {
        n_reached as f64 / n_with_pairs as f64
    };
    let mean_budget_to_90 = if n_reached == 0 {
        0.0
    } else {
        budget_sum / n_reached as f64
    };
    EvbReport {
        arms,
        diff,
        diff_lo,
        diff_hi,
        claim,
        n_reached,
        fraction_reached,
        mean_budget_to_90,
    }
}

fn eval_pairs(current: Option<Subject>, reviews: &[Subject]) -> Vec<(Subject, Subject)> {
    let Some(mut prev) = current else {
        return Vec::new();
    };
    let mut pairs = Vec::with_capacity(reviews.len());
    for used in reviews {
        pairs.push((prev.clone(), used.clone()));
        prev = used.clone();
    }
    pairs
}

fn discounted(
    q: &BTreeMap<(Subject, Subject), f64>,
    touched: &BTreeSet<Subject>,
    actions: &BTreeMap<Subject, Vec<Subject>>,
    pairs: &[(Subject, Subject)],
) -> f64 {
    let mut ret = 0.0;
    let mut weight = 1.0;
    for (state, used) in pairs {
        if top1(q, touched, actions, state).is_some_and(|action| &action == used) {
            ret += weight;
        }
        weight *= GAMMA;
    }
    ret
}

fn top1(
    q: &BTreeMap<(Subject, Subject), f64>,
    touched: &BTreeSet<Subject>,
    actions: &BTreeMap<Subject, Vec<Subject>>,
    state: &Subject,
) -> Option<Subject> {
    if !touched.contains(state) {
        return None;
    }
    let list = actions.get(state)?;
    let mut best: Option<&Subject> = None;
    let mut best_q = 0.0;
    for action in list {
        let value = q_at(q, state, action);
        match &best {
            None => {
                best = Some(action);
                best_q = value;
            }
            Some(current) => match value.total_cmp(&best_q) {
                Ordering::Greater => {
                    best = Some(action);
                    best_q = value;
                }
                Ordering::Equal if action < current => {
                    best = Some(action);
                    best_q = value;
                }
                _ => {}
            },
        }
    }
    best.cloned()
}

struct Choice<'a> {
    arm: &'a str,
    exps: &'a [Experience],
    actions: &'a BTreeMap<Subject, Vec<Subject>>,
    q: &'a BTreeMap<(Subject, Subject), f64>,
    sr: &'a BTreeMap<Subject, f64>,
    current: &'a Option<Subject>,
    static_float: &'a [f64],
    static_int: &'a [u64],
    uniform_order: &'a [usize],
}

fn choose(choice: &Choice<'_>, step: usize, rng: &mut SplitMix64) -> usize {
    match choice.arm {
        "random" => (rng.next_u64() % choice.exps.len() as u64) as usize,
        "uniform" => choice.uniform_order[step % choice.uniform_order.len()],
        "recency" => argmax_int(choice.exps, choice.static_int),
        "need_only" | "fsrs_r" => argmax_float(choice.exps, choice.static_float),
        "evb" | "gain_only" | "indicator_need" => {
            let priority = (0..choice.exps.len())
                .map(|index| {
                    live_priority(
                        choice.arm,
                        &choice.exps[index],
                        choice.actions,
                        choice.q,
                        choice.sr,
                        choice.current,
                    )
                })
                .collect::<Vec<_>>();
            argmax_float(choice.exps, &priority)
        }
        _ => 0,
    }
}

fn live_priority(
    arm: &str,
    exp: &Experience,
    actions: &BTreeMap<Subject, Vec<Subject>>,
    q: &BTreeMap<(Subject, Subject), f64>,
    sr: &BTreeMap<Subject, f64>,
    current: &Option<Subject>,
) -> f64 {
    let gain = effective_gain(gain_of(q, exp, actions));
    let need = match arm {
        "gain_only" => return gain,
        "indicator_need" => {
            if current.as_ref() == Some(&exp.state) {
                1.0
            } else {
                0.0
            }
        }
        _ => sr.get(&exp.state).copied().unwrap_or(0.0),
    };
    if need == 0.0 { 0.0 } else { gain * need }
}

fn static_float_priority(
    arm: &str,
    prefix: &Prefix,
    exps: &[Experience],
    sr: &BTreeMap<Subject, f64>,
    current: &Option<Subject>,
) -> Vec<f64> {
    exps.iter()
        .map(|exp| match arm {
            "need_only" => sr.get(&exp.state).copied().unwrap_or(0.0),
            "fsrs_r" => match &exp.state {
                Subject::Node(id) => prefix.retrievability.get(id).copied().unwrap_or(0.0),
                Subject::Path(_) => 0.0,
            },
            "indicator_need" if current.as_ref() == Some(&exp.state) => 1.0,
            "indicator_need" => 0.0,
            _ => 0.0,
        })
        .collect()
}

fn static_int_priority(arm: &str, events: &[Rec], exps: &[Experience]) -> Vec<u64> {
    if arm != "recency" {
        return vec![0; exps.len()];
    }
    exps.iter()
        .map(|exp| state_recency(events, &exp.state))
        .collect()
}

/// Greatest frame seq of a prefix event that mentions `subject`.
fn state_recency(events: &[Rec], subject: &Subject) -> u64 {
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

fn argmax_float(exps: &[Experience], priority: &[f64]) -> usize {
    let mut best = 0usize;
    for index in 1..exps.len() {
        match priority[index].total_cmp(&priority[best]) {
            Ordering::Greater => best = index,
            Ordering::Equal if exps[index].id < exps[best].id => best = index,
            _ => {}
        }
    }
    best
}

fn argmax_int(exps: &[Experience], priority: &[u64]) -> usize {
    let mut best = 0usize;
    for index in 1..exps.len() {
        if priority[index] > priority[best]
            || (priority[index] == priority[best] && exps[index].id < exps[best].id)
        {
            best = index;
        }
    }
    best
}

fn uniform_index(exps: &[Experience]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..exps.len()).collect();
    order.sort_by(|&left, &right| exps[left].id.cmp(&exps[right].id));
    order
}

fn action_sets(exps: &[Experience]) -> BTreeMap<Subject, Vec<Subject>> {
    let mut sets: BTreeMap<Subject, BTreeSet<Subject>> = BTreeMap::new();
    for exp in exps {
        sets.entry(exp.state.clone())
            .or_default()
            .insert(exp.action.clone());
    }
    sets.into_iter()
        .map(|(state, actions)| (state, actions.into_iter().collect()))
        .collect()
}

fn q_at(q: &BTreeMap<(Subject, Subject), f64>, state: &Subject, action: &Subject) -> f64 {
    q.get(&(state.clone(), action.clone()))
        .copied()
        .unwrap_or(0.0)
}

fn max_q(
    q: &BTreeMap<(Subject, Subject), f64>,
    actions: &BTreeMap<Subject, Vec<Subject>>,
    state: &Subject,
) -> f64 {
    let Some(list) = actions.get(state) else {
        return 0.0;
    };
    let mut best: Option<f64> = None;
    for action in list {
        let value = q_at(q, state, action);
        best = Some(match best {
            None => value,
            Some(current) if value > current => value,
            Some(current) => current,
        });
    }
    best.unwrap_or(0.0)
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let mut peak = values[0];
    for value in values.iter().skip(1) {
        if *value > peak {
            peak = *value;
        }
    }
    let mut weights = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    for value in values {
        let weight = libm::exp(BETA * (value - peak));
        weights.push(weight);
        sum += weight;
    }
    if sum == 0.0 {
        let share = 1.0 / values.len() as f64;
        return vec![share; values.len()];
    }
    weights.into_iter().map(|weight| weight / sum).collect()
}

fn backup_proof(prefix: &Prefix, exp: &Experience, model_proof: &Proof) -> Proof {
    let mut seqs = model_proof.cited_seqs.clone();
    seqs.push(exp.frame_seq);
    seqs.extend(reward_seqs(prefix, exp));
    proof_of(seqs)
}

fn reward_seqs(prefix: &Prefix, exp: &Experience) -> Vec<u64> {
    let Subject::Node(id) = &exp.successor else {
        return Vec::new();
    };
    let handle = card_handle(id);
    let mut seqs = Vec::new();
    for rec in &prefix.events {
        match &rec.body {
            Body::Review { card_id, .. } if *card_id == handle => seqs.push(rec.frame_seq),
            Body::Edge { target, link, .. }
                if target == id && (link == "corrects" || link == "supersedes") =>
            {
                seqs.push(rec.frame_seq);
            }
            _ => {}
        }
    }
    seqs.extend(weave_seqs_for(&prefix.events, id));
    seqs
}

fn weave_signs(events: &[Rec]) -> Vec<(String, i32)> {
    let mut out = Vec::new();
    for rec in events {
        let Body::Upsert {
            id,
            node_type,
            tags,
            ..
        } = &rec.body
        else {
            continue;
        };
        if !is_weave(node_type, tags) {
            continue;
        }
        let sign = weave_tag_sign(tags);
        for target in weave_targets(events, id) {
            out.push((target, sign));
        }
    }
    out
}

fn weave_seqs_for(events: &[Rec], member: &str) -> Vec<u64> {
    let mut seqs = Vec::new();
    for rec in events {
        let Body::Upsert {
            id,
            node_type,
            tags,
            ..
        } = &rec.body
        else {
            continue;
        };
        if is_weave(node_type, tags)
            && weave_targets(events, id)
                .iter()
                .any(|target| target == member)
        {
            seqs.push(rec.frame_seq);
        }
    }
    seqs
}

fn weave_targets(events: &[Rec], weave_id: &str) -> Vec<String> {
    events
        .iter()
        .filter_map(|rec| match &rec.body {
            Body::Edge {
                source,
                target,
                link,
            } if source == weave_id && link == "derived_from" => Some(target.clone()),
            _ => None,
        })
        .collect()
}

fn is_weave(node_type: &str, tags: &[String]) -> bool {
    node_type == COMPOSITION_NODE_TYPE
        && tags.iter().any(|tag| tag == GHOSTLINK_TAG)
        && tags.iter().any(|tag| tag == WEAVE_TAG)
}

fn weave_tag_sign(tags: &[String]) -> i32 {
    let mut sign = 0;
    for tag in tags {
        if let Some(outcome) = tag.strip_prefix(OUTCOME_TAG_PREFIX) {
            if let Some(value) = outcome_sign(outcome) {
                sign += value;
            }
        }
    }
    sign
}

fn endpoint(id: &str, nodes: &BTreeSet<String>) -> Subject {
    if nodes.contains(id) {
        Subject::Node(id.to_string())
    } else {
        Subject::Path(id.to_string())
    }
}

fn node_set(events: &[Rec]) -> BTreeSet<String> {
    events
        .iter()
        .filter_map(|rec| match &rec.body {
            Body::Upsert { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn cards_of(events: &[Rec]) -> BTreeMap<u64, String> {
    let mut map = BTreeMap::new();
    for rec in events {
        if let Body::Upsert { id, .. } = &rec.body {
            map.insert(card_handle(id), id.clone());
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::DreamWindow;

    fn node_exp(state: &str, action: &str, reward: f64, seq: u64) -> Experience {
        let state_s = Subject::Node(state.into());
        let action_s = Subject::Node(action.into());
        Experience {
            frame_seq: seq,
            link: "derived_from".into(),
            id: format!(
                "{seq:016x}|derived_from|{}|{}",
                state_s.canon(),
                action_s.canon()
            ),
            state: state_s,
            action: action_s.clone(),
            successor: action_s,
            reward,
        }
    }

    #[test]
    fn positive_and_negative_two_action_gains_sit_in_the_same_band() {
        let good = node_exp("s", "good", 1.0, 2);
        let bad = node_exp("s", "bad", 0.0, 1);
        let actions = action_sets(&[good.clone(), bad.clone()]);
        let gain = gain_of(&BTreeMap::new(), &good, &actions);
        assert!(gain > 0.4 && gain < 0.5, "{gain}");
        let avoid = node_exp("s", "bad", -1.0, 1);
        let gain_neg = gain_of(
            &BTreeMap::new(),
            &avoid,
            &action_sets(&[good, avoid.clone()]),
        );
        assert!(gain_neg > 0.4 && gain_neg < 0.5, "{gain_neg}");
    }

    #[test]
    fn a_single_action_has_zero_gain_before_the_floor() {
        let only = node_exp("s", "only", 1.0, 1);
        let actions = action_sets(std::slice::from_ref(&only));
        assert_eq!(gain_of(&BTreeMap::new(), &only, &actions), 0.0);
        assert_eq!(effective_gain(0.0), MIN_GAIN);
        assert_eq!(effective_gain(f64::NAN), MIN_GAIN);
    }

    #[test]
    fn indicator_need_is_zero_off_the_current_state() {
        let exp = node_exp("other", "a", 1.0, 1);
        let actions = action_sets(&[exp.clone(), node_exp("other", "b", 0.0, 2)]);
        let current = Some(Subject::Node("here".into()));
        let priority = live_priority(
            "indicator_need",
            &exp,
            &actions,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &current,
        );
        assert_eq!(priority, 0.0);
        let on = node_exp("here", "a", 1.0, 1);
        let on_actions = action_sets(&[on.clone(), node_exp("here", "b", 0.0, 2)]);
        let on_priority = live_priority(
            "indicator_need",
            &on,
            &on_actions,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &current,
        );
        assert!(on_priority > 0.0);
    }

    fn bare_prefix(events: Vec<Rec>) -> Prefix {
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

    fn upsert(seq: u64, id: &str, node_type: &str, tags: &[&str]) -> Rec {
        Rec {
            frame_seq: seq,
            ord: 0,
            body: Body::Upsert {
                id: id.into(),
                scope: "s".into(),
                node_type: node_type.into(),
                tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
            },
        }
    }

    #[test]
    fn closed_by_adds_the_kind_bonus_on_a_hand_built_list() {
        let built = bare_prefix(vec![
            upsert(1, "door", "fact", &[]),
            upsert(2, "key", "fact", &[]),
            Rec {
                frame_seq: 3,
                ord: 0,
                body: Body::Edge {
                    source: "key".into(),
                    target: "door".into(),
                    link: "closed_by".into(),
                },
            },
        ]);
        let exps = experiences(&built);
        assert_eq!(exps.len(), 1);
        assert_eq!(exps[0].state, Subject::Node("door".into()));
        assert_eq!(exps[0].action, Subject::Node("key".into()));
        assert_eq!(exps[0].reward, 1.0);
    }

    #[test]
    fn weave_and_corrects_and_a_later_rating_three_follow_the_table() {
        let built = bare_prefix(vec![
            upsert(1, "w1", "fact", &[]),
            upsert(2, "w2", "fact", &[]),
            upsert(
                3,
                "weave",
                "composition",
                &["ghostlink", "ghostlink-weave", "outcome:dead_end"],
            ),
            Rec {
                frame_seq: 4,
                ord: 0,
                body: Body::Edge {
                    source: "weave".into(),
                    target: "w1".into(),
                    link: "derived_from".into(),
                },
            },
            Rec {
                frame_seq: 5,
                ord: 0,
                body: Body::Edge {
                    source: "weave".into(),
                    target: "w2".into(),
                    link: "derived_from".into(),
                },
            },
            upsert(6, "scrap", "fact", &[]),
            upsert(7, "pen", "fact", &[]),
            Rec {
                frame_seq: 8,
                ord: 0,
                body: Body::Edge {
                    source: "pen".into(),
                    target: "scrap".into(),
                    link: "corrects".into(),
                },
            },
            upsert(9, "leaf", "fact", &[]),
            Rec {
                frame_seq: 10,
                ord: 0,
                body: Body::Review {
                    card_id: card_handle("leaf"),
                    rating: 4,
                },
            },
            Rec {
                frame_seq: 11,
                ord: 0,
                body: Body::Review {
                    card_id: card_handle("leaf"),
                    rating: 3,
                },
            },
        ]);
        let rewards = node_rewards(&built);
        assert_eq!(rewards["w1"], -1);
        assert_eq!(rewards["w2"], -1);
        assert_eq!(rewards["scrap"], -1);
        assert_eq!(rewards["leaf"], 1);
        let mut dreamed = built.clone();
        dreamed.windows = vec![DreamWindow {
            start_seq: 10,
            end_seq: 12,
        }];
        assert_eq!(node_rewards(&dreamed).get("leaf"), None);
        assert!(
            experiences(&built)
                .iter()
                .all(|exp| exp.successor != Subject::Node("scrap".into()))
        );
    }

    #[test]
    fn an_untouched_state_scores_zero_and_a_zero_tie_picks_the_smaller_id() {
        let actions = action_sets(&[node_exp("s", "m", 0.0, 2), node_exp("s", "a", 0.0, 1)]);
        let q = BTreeMap::new();
        let touched = BTreeSet::new();
        let pairs = [(Subject::Node("s".into()), Subject::Node("m".into()))];
        assert_eq!(discounted(&q, &touched, &actions, &pairs), 0.0);
        let mut touched = BTreeSet::new();
        touched.insert(Subject::Node("s".into()));
        assert_eq!(discounted(&q, &touched, &actions, &pairs), 0.0);
        let picked = top1(&q, &touched, &actions, &Subject::Node("s".into())).unwrap();
        assert_eq!(picked, Subject::Node("a".into()));
    }

    #[test]
    fn recency_is_the_latest_mention_of_the_state() {
        let events = vec![
            upsert(1, "n", "fact", &[]),
            Rec {
                frame_seq: 4,
                ord: 0,
                body: Body::Edge {
                    source: "n".into(),
                    target: "z".into(),
                    link: "derived_from".into(),
                },
            },
            Rec {
                frame_seq: 9,
                ord: 0,
                body: Body::Review {
                    card_id: card_handle("n"),
                    rating: 3,
                },
            },
        ];
        assert_eq!(state_recency(&events, &Subject::Node("n".into())), 9);
        assert_eq!(state_recency(&events, &Subject::Node("z".into())), 4);
    }
}

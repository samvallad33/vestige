# Mattar–Daw Gain × Need: preregistered Strata replay backtest

**Status: protocol frozen in this file. No result is written here.**

Computed tables live only under `docs/benchmarks/results/` and only from a
commit whose history already contains this file. A number quoted from this
backtest was produced by the rules below, or the deviation is an amendment at
the bottom written before that number is quoted.

Vestige is a cognitive, deterministic memory-transaction security OS for AI
agents. This backtest does not change the live binaries. The Q and V layer
below is the model under test. Vestige v4.1.1 does not implement Gain, Need,
or EVB. Its shipped dream orderings are two of the baseline arms (FSRS
retrievability, and id order).

Paper: Mattar, M. G. & Daw, N. D. (2018). Prioritized memory access explains
planning and hippocampal replay. *Nature Neuroscience*, 21, 1609–1617.
Definitions follow the Methods (backup with α = 1, Gain equation 5, Need
equation 6, softmax β = 5, γ = 0.9, 20 planning steps, minimum gain 1e-10,
successor-representation delta rule α_T = 0.9).

Table id: `mattar-evb-v1`.

## 1. What is in this stack, and what is not run

| piece | where | this stack |
| --- | --- | --- |
| Bounded prefix fold, harness, this file | `StrataStore::as_of`, `crates/strata-backtest` | run |
| Successor-representation Need vs simpler predictors | `crates/strata-backtest/src/need.rs` | run |
| Gain × Need scheduler, nine arms, Fig. 1d analog | `crates/strata-backtest/src/evb.rs` | run |
| n-step sequences, forward/reverse classification | not in this stack | not run |
| Six ingested-repo regressions | listed below, not executed | not run |

Case list, fixed and not executed: tokio #6714, cargo #10682, uv #10186,
prometheus, kubernetes, grafana. No fixture for these cases is committed.

## 2. Hard rules

- States are node ids or exact anchor paths. No text, keyword, BM25, fuzzy,
  entity-name, shared-name, embedding, or cosine input. Content strings stored
  by the corpora are unread by every ranker, reward rule, and metric.
- Transitions come only from recorded edges and recorded events, as defined
  in section 5. The harness never writes a cause link or an outcome label.
  Outcomes are recorded events.
- No SQLite. No LLM call. Arithmetic is `libm` plus IEEE-754 `+ - * /` in a
  fixed iteration order (Rust does not contract FMA by default). Every
  published real is an `i64` Q32.32 from `strata_kernel::canonical::to_q32_32`
  (round ties to even via `libm::rint`).
- Constants below are the only constants. They are not refit after a result.
- Live `state_digest` of a store opened without `as_of` stays the digest of
  the unbounded fold. `as_of` is a read-only scratch. Dream windows do not
  delete frames from that fold.
- Real logs are opened only as copies. The copy skips `strata.lock` and never
  calls `backup_to`. The corpora in section 8 are built in-process through the
  store write API. No production database is read or committed.

## 3. Pinned constants

| symbol | value |
| --- | --- |
| `table_id` | `mattar-evb-v1` |
| γ | 0.9 |
| β (policy softmax) | 5 |
| α_T (transition delta rule) | 0.9 |
| α backup | 1 (replace) |
| K (Neumann degree, terms i = 0..=K) | 64 |
| stationary power iterations L | 256, start uniform, renormalize every step |
| minimum gain | 1e-10 |
| backups per cut N | 20 |
| budget grid | 0, 1, 2, 4, 8, 12, 20 |
| budgets in the primary mean | 1, 2, 4, 8, 12, 20 (0 is on the curve only) |
| suffix horizon H | 512 frames |
| minimum typed edges at a cut | 4 |
| minimum states at a cut | 4 |
| minimum rating-4 reviews at a cut | 1 |
| maximum cuts | 32 |
| NDCG k | 5 |
| log-likelihood temperature β_ll | 1, on rank scores, not on raw R or Need |
| bootstrap draws B | 2000 |
| bootstrap seed | `0x2018_4D41_5454_4152` |
| minimum delta | 0.02 absolute |
| clock origin | `1_700_000_000_000` ms |
| clock step | `3_600_000` ms per recorded operation |

Subseed of a tag: the first 8 bytes, little-endian, of
`blake3( b"mattar-evb-v1\0" || seed_le_u64 || tag_utf8 )`.
Log seed of a corpus id: `blake3` of the utf-8 string
`mattar-evb-v1/log/{corpus_id}`.

CI index on a length-B sample sorted ascending by `total_cmp`:
`lo = floor(0.025 * (B-1))`, `hi = ceil(0.975 * (B-1))`.
With B = 2000 that is index 49 and index 1950, computed in integers as
`(25 * (B-1)) / 1000` and `(975 * (B-1) + 999) / 1000`.

Cut thinning, only when more than 32 eligible bounds exist. For
`i in 0..32`, keep index `(i as u128) * (len - 1) / 31`. Both endpoints
are included. If the cap were 1 the first bound would be kept; the cap is 32.

## 4. Claims (fixed before any number)

Primary EVB endpoint: mean, over cuts, of the per-cut mean of discounted
top-1 return on the budget grid excluding 0.

Let D be that quantity for `evb` minus the same quantity for `gain_only`,
with a paired bootstrap over cuts (section 11).

- `evb_beats_gain_only` only if D ≥ 0.02 and the 95% CI lower bound is > 0.
- `gain_only_beats_evb` only if D ≤ −0.02 and the 95% CI upper bound is < 0.
- otherwise `not_separated`.
- `undefined_no_cuts` if there is no cut.

Budget 20 is reported and is not the headline. A saturated budget can hide
an ordering effect.

Primary Need endpoint: AUC(`sr_need`) − AUC(`fsrs_r`) on the cuts where both
AUCs are defined, same delta and the same CI rule.

- `sr_need_beats_fsrs_r`, `fsrs_r_beats_sr_need`, `not_separated`,
  `undefined_no_cuts`.

A null or a negative result is a result. No arm is declared the winner by a
test. Tests check shape, prefix purity, and byte identity only.

Secondary, no claim: smallest budget in the grid whose return is at least
90% of the oracle return (section 10). Report the mean of that budget over
cuts that reach it, and the fraction of cuts with at least one eval pair
that reach it.

Fig. 6c analog, no claim: mean entropy of the normalized Need row, binned by
the row's maximum transition probability (section 6). Empty bins stay empty.

## 5. States, transitions, current state

A subject is `node:{id}` or `path:{path}`, compared in that canon order
(nodes before paths, then bytewise).

Candidate set at a cut: every node id upserted at `seq ≤ bound`, every
anchor `file_path` recorded at `seq ≤ bound`, and every edge endpoint at
`seq ≤ bound` that is not a node id (a path subject, exact string).

Visits, in `(frame_seq, ord)` order:

- an upsert of a node is a node visit in that node's scope;
- a `ReviewNode` that is not inside a dream window is a node visit in the
  scope of the upserted node whose card handle matches;
- an anchor row is a path visit.

Card handle: first 8 bytes of `blake3(node_id)`, little-endian. The last
upsert with that handle wins a collision.

Current state: the last visit in the prefix. Iteration order is the total
order, so a later row at the same `(seq, ord)` replaces the earlier one.

Node transitions, one stream per scope: successive **distinct** node visits.
The pair `(u, v)` is kept only when some typed edge (any of the eight
vocabulary strings) has those two ids as its endpoints, either orientation,
and that edge's frame seq is ≤ v's visit seq. The latest such edge is the
one cited. Direction of T is visit order, not edge orientation. An edge that
appears only after v does not join the pair at this cut.

Anchor transitions, one stream: successive distinct exact `file_path` values
in anchor-op order (`RecordAnchors` and `ReplaceAnchors`, row order is
`ord`). No edge-join requirement. That is a disclosed modeling choice.

Both kinds of transition update one successor representation. There is no
invented node↔path transition.

Write occupancy is not decision occupancy. The log does not record reads.

## 6. Successor representation

T is a row-stochastic matrix indexed in subject order.

Delta rule, α_T = 0.9, applied to each kept transition in time order:

- the first observation of a row sets that row to a one-hot on the
  successor (it does not scale a zero row by α_T);
- every later observation multiplies the row by `(1 − α_T)` and then adds
  α_T on the observed successor.

After all observations, a row that was never observed becomes a self-loop.

Need from current state c, for target s:

`Need(s) = Σ_{i=0..=K} γ^i (T^i)_{c,s}`

with `(T^0) = I`. The row vector is multiplied on the right. Inner sums
walk source index, then target index, ascending.

Stationary variant: μ starts uniform. For L = 256 steps, μ ← μ T, then
divide by the sum, in index order. `Need_stat(s) = μ_s * Σ_{i=0..=K} γ^i`.
The geometric sum is a `libm`-free loop of multiplies by γ. This variant
does not depend on the current state.

Fig. 6c observations: at each cut, for each state whose row was actually
observed (not merely completed as a self-loop), let p_max be the maximum
entry of that row. Treat that state as the current state, normalize its
Need row (skip the row if the sum is 0), and compute Shannon entropy
`−Σ p ln p` with `libm::ln`, skipping zeros. Bins of p_max: `< 0.4`,
`[0.4, 0.7)`, `≥ 0.7`. The reported entropy is the mean over
`(cut, state)` observations in the bin.

Every Need value's proof cites the visit seqs and the joining-edge seqs
that built T. The firewall rejects a proof that cites any seq `> bound`.

## 7. Need arms and predictive metrics

Rank score descending, subject canon ascending for ties. AUC treats equal
scores as ties (0.5) and does **not** use the subject tie-break. NDCG and
log-likelihood use the tie-broken order.

| id | score |
| --- | --- |
| `sr_need` | section 6, from the current state. Missing current state ⇒ all zeros |
| `stationary_need` | section 6 |
| `fsrs_r` | `retrievability` of that node at the prefix head clock. Paths and cards with no value score 0. This is context-free R, the Anderson & Milson analogue, not Need |
| `recency` | greatest frame seq of a prefix event that mentions the subject (upsert, review, either edge endpoint, anchor node id, or anchor path). Higher is more recent. Compared as an integer, never as `f64` |
| `degree` | count of typed edges at `seq ≤ bound` that touch the subject. A non-node endpoint is a path subject |
| `uniform` | identical scores; order is subject canon ascending |
| `random` | Fisher–Yates shuffle of the candidate vector. For `i` from `len-1` down to `1`, swap `i` with `next_u64() % (i+1)`. Seed tag `need-random/{corpus}/{bound}` with `{bound}` in decimal. Modulo bias is accepted |

Suffix positives: subjects produced by section 9 whose frame lies in
`(bound, bound+H]`, intersected with the prefix candidate set. AUC and
NDCG are undefined on a cut with no positive or no negative, and that cut
is dropped from those means. `n` is the number of cuts that remain.

AUC is the Mann–Whitney probability a positive outranks a negative, 0.5
on `total_cmp` equality.

NDCG@5: binary relevance, discount `1 / log2(i+2)` via `libm::log2`, `i`
0-based, ideal DCG from the same formula on the `min(k, n_pos)` positives.

Suffix log-likelihood: rank 0 is best, score `(n_candidates − rank)`,
softmax at β_ll = 1 implemented as `p(rank) ∝ exp(−rank)` with `libm::exp`
and a denominator summed in rank order. The cut's LL is the mean of
`libm::ln(p)` over positive candidates. Raw R and raw Need never enter
this softmax, so scale cannot decide the comparison.

Bootstrap tags (each statistic has its own subseed):

- `boot/{corpus}/need/{arm}/auc`
- `boot/{corpus}/need/{arm}/ndcg`
- `boot/{corpus}/need/{arm}/ll`
- `boot/{corpus}/need/sr_minus_fsrs_auc` (paired, shared indices, only cuts
  where both AUCs exist)

## 8. Corpora

Both corpora use an empty dream-window list. Operation `i` (0-based) is
stamped `created_at_ms = origin + i * step`. Reviews pass that stamp as
`reviewed_at_ms`. Edge strength is 1000 milli, activation count 0, no
meta sha. Scope is `synth` or `agent`. Content strings exist only because
ingest rejects empty content.

### 8.1 `synth-track-v1`

1. Ingest `dead`.
2. For `i` in `0..8`: ingest `d{i}`, ingest `p{i}`, `derived_from` `d{i} → p{i}`, `derived_from` `d{i} → dead`, review `p{i}` at rating 4.
3. Ingest `hub`, `a1`, `a2`, `a3`, `a4`, `a5`.
4. `derived_from` edges, in order: `hub→a1`, `a1→a2`, `a2→a3`, `a3→a4`, `a4→a5`, `a5→hub`, then each of `hub`, `a1`, `a2`, `a3`, `a4`, `a5` `→ dead`.
5. Review `a5` at rating 4. No other rating-4 review is placed on the chain, and no weave or `closed_by` is added on the chain. Stacking those would change the leaf reward and is not this corpus.
6. Three episodes of rating-3 reviews: `hub`, `a1`, `a2`, `a3`, `a4`, `a5`.
7. One suffix episode of the same six rating-3 reviews.

Decoy components are disconnected from the evaluation cycle except for
edges into `dead`, which are not visit-transitions back onto the decoys
(section 5). Decoy payoff edges are admitted before the chain, so equal
Gain ties break toward them. `dead` is ingested first, so its id is the
smallest among those nodes; an all-zero touched state therefore does not
count as a hit on the chain (section 10).

### 8.2 `recorded-ops-v1`

Built through `ingest_in_scope`, `save_connection`, `review_at`, and
`record_anchors`. Not a copy of a production store.

1. Ingest `dead`.
2. For `i` in `0..6`: the same decoy pattern as section 8.1 (payoff review rating 4 on `p{i}`).
3. Ingest `failure`, `lesson`, `fix`.
4. `derived_from` edges, in order: `failure→lesson`, `failure→dead`, `lesson→fix`, `fix→failure`, `lesson→dead`, `fix→dead`.
5. Review `fix` at rating 4 only. No `closed_by` and no helpful weave on `fix`.
6. Ingest `w1`, `w2`. Ingest a composition node, `node_type` `composition`, tags `ghostlink`, `ghostlink-weave`, `outcome:dead_end`. `derived_from` from that composition node to `w1` and to `w2`.
7. Ingest `pen` and `scrap`. One `corrects` edge `pen → scrap`.
8. One `RecordAnchors` batch on `fix`, row order: id `anc-attach` path `src/attach.rs`, then id `anc-receipt` path `src/receipt.rs`. Other anchor fields are empty.
9. Three episodes of rating-3 reviews: `failure`, `lesson`, `fix`.
10. Suffix episode, rating 3: `failure`, `lesson`, `fix`.
11. Suffix anchor batch on `fix`: id `anc-receipt-suffix`, path `src/receipt.rs` only.

`scrap` is the penalty target on purpose. Under Gain equation 5 a backup
whose reward is −1 has **positive** Gain: the policy improves by avoiding
that action. Putting the penalty on `dead` would insert avoidance backups
into the decision states. `scrap` is not the successor of any experience,
so the penalty table is live and is not an extra prioritized-sweeping
target. A unit test still checks that a two-action 0→−1 backup has
positive Gain. The corpora contain no `closed_by` edge; the +1 kind bonus
is unit-tested on a hand-built event list.

`closed_by` is not placed on the used leaf. Visit reviews are rating 3 so
they do not clear the rating-4 promote (section 9).

## 9. Experiences, reward, backup, Gain

An experience is a recorded edge whose `link_type` is `derived_from`,
`closed_by`, `evidence_of`, or `touched`, with `seq ≤ bound`.

Orientation, the same one `upstream_end` uses for walks:

| link | state | action | successor |
| --- | --- | --- | --- |
| `derived_from` | source | target | target |
| `closed_by`, `evidence_of`, `touched` | target | source | source |

`corrects` and `supersedes` are not experiences. Each such edge adds −1 to
the target node's reward. `SupersedeNode` frames are not rewards.

Experience id, bytewise: `{frame_seq:016x}|{link}|{state.canon}|{action.canon}`.

Node reward, prefix events only, dream-window reviews ignored:

- reviews in time order: rating 4 sets +1, rating 1 sets −1, ratings 2 and 3 leave the value unchanged;
- each weave outcome sign, summed;
- each `corrects` or `supersedes` penalty, summed.

A weave is an upsert with `node_type` `composition` and both tags
`ghostlink` and `ghostlink-weave`. Each tag `outcome:{type}` whose type is
in the table below contributes its sign to every target of a
`derived_from` edge sourced at that composition node. Unknown outcome
strings contribute 0. Signs are +1 or −1, not the fractional composition
adjustments of any other crate.

| sign | outcome type |
| --- | --- |
| +1 | `helpful`, `accepted`, `submitted`, `user_promoted` |
| −1 | `dead_end`, `rejected`, `duplicate_risk`, `needs_poc`, `bad_severity`, `user_demoted`, `closed_by_scope`, `closed_by_duplicate`, `closed_by_false_assumption`, `closed_by_user`, `expired_lane` |

Experience reward = kind bonus + node reward of the successor.
Kind bonus is +1 for `closed_by` and 0 otherwise. A path successor has
node reward 0.

Action set of a state: the sorted unique experience actions at that state.

Policy: softmax over that action set,
`π(a) ∝ exp(β (Q(s,a) − max Q))`, `libm::exp`, denominator in action
order. Subtracting the max is the same function as `exp(β Q)`. One action
⇒ π = 1 ⇒ Gain = 0 before the floor. Missing Q is 0.

Backup, the paper's Bellman operator, not a policy expectation:
`Q(s,a) ← r + γ * max_{a'} Q(s', a')`, or `r` when `s'` has no action.
The maximum's value does not depend on which tied action attains it.

Gain (equation 5), using Q before and after the hypothetical backup:
`Σ_a Q_new(s,a) * (π_new(a|s) − π_old(a|s))`.

`effective_gain` is `max(gain, 1e-10)`. A NaN gain becomes `1e-10`.
This floor is why every experience remains eligible. It is not a drop
filter. EVB = 0 when Need is 0, and the product itself is not floored.
Otherwise EVB = `effective_gain * Need(state)`.

The mapping sentence "V(src) ← r + γ Σ_a π(a|src) Q" is the policy
expectation inside Gain, not the backup target. This protocol uses the
paper's max backup. n-step extension is not this stack: every backup has
length 1, so the shorter-sequence tie-break is vacuous and the experience
id breaks the tie.

## 10. Scheduler and return

Q starts at 0 inside each arm, independently. Need is fixed for the cut
(it does not depend on Q). `evb`, `gain_only`, and `indicator_need`
recompute Gain from Q before every backup. The other arms have a static
key, except `random`.

| id | what is maximized each step |
| --- | --- |
| `evb` | EVB, Need = SR Need of the experience's state |
| `gain_only` | `effective_gain`. Prioritized sweeping |
| `need_only` | SR Need of the state. Static |
| `fsrs_r` | R of the state node at the prefix head clock; paths score 0. This is the current dream-compile triage |
| `recency` | recency of the state, integer |
| `random` | uniform with replacement, index `next_u64() % n`. Seed tag `evb-random/{corpus}/{bound}`. Original Dyna. Modulo bias accepted |
| `uniform` | experiences sorted by id, step `t` selects index `t % n`. This is the current id-ordered dream |
| `no_replay` | no backup |
| `indicator_need` | EVB with Need = 1 iff `experience.state` is the current state, else 0. Mattar & Daw supplementary indicator ablation |

Static keys are re-selected every step, with replacement. A flat key
repeats the same experience. That is argmax, not a without-replacement
queue.

Tie-break, after the priority: shorter sequence first (always 1), then
experience id ascending. Priority comparison is `total_cmp`.

Applying a backup marks its **state** touched, even if Q does not change.

Top-1 at a state is undefined unless that state has been touched. An
undefined top-1 scores 0. Otherwise top-1 is the action with greatest Q,
tie broken by smaller subject canon. `no_replay` never touches a state
and scores 0. Because `dead` is the earliest node, a touched state whose
actions are still all zero does not hit the later chain action.

Eval pairs, suffix `ReviewNode` events only, in `(bound, bound+H]`, dream
windows excluded, in log order. The used subject is the reviewed node.
The predecessor of the first pair is the prefix current state. The
predecessor of each later pair is the previous suffix review. Anchor
paths are not return pairs. Discount weight of pair `i` (from 0) is
`γ^i`, by iterative multiply. Reward is 1 when top-1 equals the used
subject and that subject is an action at the predecessor; otherwise 0.
A pair whose predecessor has no action scores 0 for every arm.

Return = `Σ_i γ^i * reward_i`. Oracle = `Σ_i γ^i` over the same pairs.
A cut with no pair is left out of the secondary fraction's denominator.

The first `k` backups of a 20-step run are the `k`-budget policy for the
deterministic arms, and for `random` because the generator is not
reseeded between budgets. Snapshots are taken at every budget in the grid
inside that one run.

## 11. Bootstrap

Paired resampling over the cuts that enter the statistic, in cut order.
`SplitMix64` starts at the statistic's subseed:

`state += 0x9E3779B97F4A7C15`;
`z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`;
`z = (z ^ (z >> 27)) * 0x94D049BB133111EB`;
`return z ^ (z >> 31)`.

Each draw: `n` times, add the value at index `next_u64() % n`, then
divide by `n`. Sort with `total_cmp`. Read the CI indices from section 3.

Tags:

- `boot/{corpus}/evb/{arm}/b{budget}`
- `boot/{corpus}/evb/{arm}/mean`
- `boot/{corpus}/evb/evb_minus_gain_only`

`{corpus}` is `synth-track-v1` or `recorded-ops-v1`. `{budget}` is
decimal. `{arm}` is the id in section 7 or section 10.

## 12. Dream windows and the origin firewall

A dream window is a half-open frame-seq interval `[start, end)`. Reviews
inside any window are excluded from transitions, rewards, the rating-4
eligibility count, and suffix outcomes. They stay in the folded FSRS
state, because the fold is the log. Both corpora pass an empty list.
Edges inside a window are not excluded.

`rank` proofs and backup proofs must cite only seqs `≤ bound`. A violation
fails the run. Suffix labels are taken only from seqs `> bound`. Reward
and T are built from the prefix event list only.

## 13. Bytes

Harness manifest (library, no arm results), canonical JSON, trailing
newline, keys sorted at every object:

- `cuts[]` with `bound_seq`, `prefix_head_frame_hash`, `prefix_head_seq`, `state_digest`
- `dream_windows[]` with `end_seq`, `start_seq`
- `prereg_blake3` (lowercase hex of this file's bytes)
- `table_id`

`prefix_head_frame_hash` is the chain hash of the last folded frame of
that seeded log. It is comparable across copies opened from the same
seed and not across two unseeded logs. `state_digest` is comparable
either way. The prefix-purity property is
`state_digest(as_of(T)) == digest(fresh store built from the prefix)`,
tested in `strata-store`.

Results file `docs/benchmarks/results/mattar-evb-v1.json`, same canonical
rules. Integers only (`i64` Q32.32 and `u64`). No `f64`, no `null`.
When a mean has `n = 0`, its Q fields are 0 and the claim is
`undefined_no_cuts` or, for a per-arm AUC, the arm's `n` is 0.

Top-level keys: `corpora`, `deviations`, `dream_windows`, `prereg_blake3`,
`table_id`.

`deviations`, in this order:

1. `one_step_backups_pr3_not_run`
2. `backup_is_bellman_max_not_policy_expectation`
3. `corpus_b_in_process_recorded_ops_corpus_c_not_run`
4. `dream_windows_empty`
5. `anchor_path_transitions_without_edge_join`
6. `write_occupancy_not_decision_occupancy`
7. `q_is_the_model_under_test`

`corpora`, in order `synth-track-v1` then `recorded-ops-v1`. Each corpus:
`cuts`, `evb`, `id`, `log_seed`, `n_cuts`, `need`.

`need.arms`, in the section 7 order. Each arm: `auc_ci_hi_q`,
`auc_ci_lo_q`, `auc_q`, `id`, `ll_q`, `n`, `ndcg_q`.
Also `claim`, `fig6c` (bins `lt_0.4`, `mid_0.4_0.7`, `ge_0.7`, each with
`bin`, `entropy_q`, `n`), `sr_minus_fsrs_auc_q`, `sr_minus_fsrs_ci_hi_q`,
`sr_minus_fsrs_ci_lo_q`.

`evb.arms`, in the section 10 order. Each arm: `by_budget[]` of
`budget`, `ci_hi_q`, `ci_lo_q`, `return_q` for the seven budgets; `id`;
`mean_ci_hi_q`; `mean_ci_lo_q`; `mean_over_budgets_q`.
Also `claim`, `evb_minus_gain_only_ci_hi_q`, `evb_minus_gain_only_ci_lo_q`,
`evb_minus_gain_only_q`, `fraction_reached_q`, `mean_budget_to_90_q`,
`n_reached`. `mean_budget_to_90_q` is 0 when `n_reached` is 0.
`fraction_reached_q` is 0 when no cut has an eval pair.

Markdown sibling `mattar-evb-v1.md` is rendered from those integers.
Eight fraction digits: for absolute value `ip * 2^32 + frac`, the digits
are `(frac * 10^8) / 2^32` truncated toward zero. The integer is exact.
The markdown is byte-checked with the JSON.

Regenerate, from the repository root:

```sh
cargo run --quiet --release --manifest-path crates/strata-backtest/Cargo.toml --bin strata-backtest -- --prereg docs/benchmarks/MATTAR-EVB-PREREGISTRATION.md --out docs/benchmarks/results
```

Check the committed bytes:

```sh
cargo run --quiet --release --manifest-path crates/strata-backtest/Cargo.toml --bin strata-backtest -- --prereg docs/benchmarks/MATTAR-EVB-PREREGISTRATION.md --check docs/benchmarks/results/mattar-evb-v1.json
```

`--check` also compares `mattar-evb-v1.md` beside that JSON. The committed
files are the release-binary bytes.

## 14. What this will not claim

- That Vestige already replays by Gain × Need.
- That a separated result on these two constructed logs is a result about
  the six named repositories, or about animal replay.
- That write occupancy is the agent's decision occupancy.
- That n-step forward or reverse sequences emerge. They are not computed.
- A winner when the CI covers 0 or the point estimate is inside ±0.02.

## 15. Amendments

None.

//! # Cross-engine differential harness (SQLite vs STRATA)
//!
//! The determinism/parity gate for the SQLite→STRATA default flip.
//!
//! A driver takes an operation script (a JSON list of `{ingest, edge, review,
//! suppress}` ops) and replays it against BOTH engines:
//!
//! * **vestige-core SQLite** — the real `Storage` (ingest / typed edges /
//!   `mark_reviewed` / `suppress_memory`) plus the real `FSRSScheduler`
//!   driven as a pure fold with script-controlled elapsed time (the wall
//!   clock only enters `mark_reviewed`, which is exactly the non-determinism
//!   the flip retires; the shadow fold pins it).
//! * **strata** — the sibling crates directly (the `strata-store` sibling may
//!   not be merged yet, so ops are scripted at the sibling-crate level):
//!   every mutating op goes through the `GateRuntime`
//!   propose→gate→effect discipline over `MemLog`, reviews fold through the
//!   version-pinned strata kernel, and the resulting log+checkpoints+gate
//!   frames are materialized into the `strata-verify` store layout and
//!   verified with `verify_store`.
//!
//! ## What parity is asserted (and why)
//!
//! Node ids are engine-generated (SQLite mints UUIDs; strata fact ids are
//! gate-log seqs), so the harness asserts instead:
//!
//! * same node COUNTS and the same CONTENT DIGESTS in sorted order;
//! * same edge sets (sorted `(digest_src, digest_dst, link)` triples);
//! * same suppressed content sets;
//! * same integer FSRS trajectory per card — review counts, lapse counts,
//!   and (for scripts that avoid the documented Hard-review phase delta)
//!   learning phase.
//!
//! Cross-engine numeric equality of stability/difficulty is NOT asserted and
//! CANNOT be: the engines pin different weight tables by design
//! (`FSRS6_WEIGHTS` floats in vestige-core vs `V1_WEIGHTS_MILLI` in the
//! kernel) and different time bases (wall-clock days vs SEQ). What IS
//! asserted is quantized determinism: each engine's FSRS stream, projected
//! through the kernel's Q32.32 quantizer, must be byte-identical across two
//! seeded runs of that engine — that is the property the default flip needs.
//!
//! ## Known engine delta (documented, not masked)
//!
//! A Hard (rating 2) review at elapsed ≥ 1 day moves a card to `Review` in
//! vestige-core but preserves its current phase in the strata kernel. The
//! fuzz generator may emit Hard reviews; phase parity is asserted on the
//! fixed script (ratings in {1,3,4}) and reported elsewhere.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};
use strata_gate::log::MemLog;
use strata_gate::policy::{ANY_KIND, Rule, WILDCARD_PREFIX};
use strata_gate::record::{EffectRecord, Propose, Verdict};
use strata_gate::{GateRuntime, Policy};
use strata_kernel::canonical::to_q32_32;
use strata_kernel::checkpoint::Checkpoint;
use strata_kernel::event::ReviewEvent;
use strata_kernel::fsrs::{ALGO_V1, CardPhase};
use strata_kernel::kernel::Kernel;
use strata_kernel::state::State;
use strata_verify::VerifyReport;
use strata_verify::layout::{GateFrame, KernelRecord, StoreFiles, gate_frame_hash, write_store};
use vestige_core::storage::{EdgeDirection, TypedEdge};
use vestige_core::SqliteMemoryStore;
use vestige_core::{FSRSScheduler, FSRSState, IngestInput, LearningState, Rating};

// ============================================================================
// Operation script
// ============================================================================

/// One operation in a differential script. `ingest` indices are implicit
/// (the i-th ingest creates node i); `edge`/`review`/`suppress` reference
/// nodes by that index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Create a memory node.
    Ingest {
        /// Node content (unique per script by construction).
        content: String,
        /// Node type (`fact`, `decision`, ...).
        node_type: String,
        /// Tags.
        tags: Vec<String>,
    },
    /// Declare a typed edge between two previously ingested nodes.
    Edge {
        /// Source node index (ingest order).
        src: usize,
        /// Target node index.
        dst: usize,
        /// Typed-edge vocabulary member (`derived_from`, ...).
        link: String,
    },
    /// Review a node. `rating` is the FSRS 1..=4 convention;
    /// `elapsed_seq` is the strata SEQ gap and, on the SQLite side, maps
    /// 1:1 to elapsed DAYS (1 seq = 1 day; always >= 1 so both engines take
    /// the non-same-day path and lapse counting aligns).
    Review {
        /// Node index.
        node: usize,
        /// Rating 1..=4 (again/hard/good/easy).
        rating: u8,
        /// Elapsed time in SEQ units (>= 1).
        elapsed_seq: u64,
    },
    /// Suppress a node (SQLite `suppress_memory`; strata RETIRE effect).
    Suppress {
        /// Node index.
        node: usize,
    },
}

/// A seeded operation script.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Script {
    /// The seed the script was generated from (informational for fixed
    /// scripts; the driver itself is fully deterministic given `ops`).
    pub seed: u64,
    /// Operations in replay order.
    pub ops: Vec<Op>,
}

impl Script {
    /// Parse a script from JSON — either `{"seed": N, "ops": [...]}` or a
    /// bare array of ops.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("invalid script JSON: {e}"))?;
        match value {
            serde_json::Value::Array(ops) => Ok(Script {
                seed: 0,
                ops: serde_json::from_value(serde_json::Value::Array(ops))
                    .map_err(|e| format!("invalid ops: {e}"))?,
            }),
            other => serde_json::from_value(other).map_err(|e| format!("invalid script: {e}")),
        }
    }
}

// ============================================================================
// Seeded generator (SplitMix64 — no external RNG dependency)
// ============================================================================

/// SplitMix64: tiny, deterministic, seedable.
pub struct Rng(u64);

impl Rng {
    /// New generator from a seed.
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish pick in `0..bound` (bound > 0).
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Generate a seeded script with at least `min_ops` operations (always
/// >= 20 by construction of the per-class minimums). Contents are unique
/// > (index-stamped), ratings span 1..=4, elapsed gaps are 1..=9 SEQ.
pub fn gen_script(seed: u64, min_ops: usize) -> Script {
    let mut rng = Rng::new(seed);
    let ingests = 8 + rng.below(5); // 8..=12
    let edges_wanted = 6 + rng.below(5); // 6..=10
    let reviews_wanted = 10 + rng.below(7); // 10..=16
    let suppresses_wanted = 1 + rng.below(3); // 1..=3

    // Phase 1: ingests.
    let ops: Vec<Op> = (0..ingests)
        .map(|i| Op::Ingest {
            content: format!("differential node {i} (seed {seed})"),
            node_type: if i % 2 == 0 { "fact" } else { "decision" }.to_string(),
            tags: vec!["diff".to_string(), format!("n{i}")],
        })
        .collect();

    // Phase 2: interleave the remaining op classes at random positions,
    // respecting script invariants (edges/reviews/suppresses target live,
    // existing, un-suppressed nodes; suppress only un-suppressed nodes).
    let mut suppressed: BTreeSet<usize> = BTreeSet::new();
    let live = |suppressed: &BTreeSet<usize>| -> Vec<usize> {
        (0..ingests).filter(|i| !suppressed.contains(i)).collect()
    };

    let mut to_emit: Vec<Op> = Vec::new();
    for _ in 0..edges_wanted {
        let a = rng.below(ingests);
        let mut b = rng.below(ingests);
        if b == a {
            b = (b + 1) % ingests;
        }
        const LINKS: [&str; 4] = ["derived_from", "touched", "evidence_of", "corrects"];
        to_emit.push(Op::Edge {
            src: a,
            dst: b,
            link: LINKS[rng.below(LINKS.len())].to_string(),
        });
    }
    for _ in 0..reviews_wanted {
        let candidates = live(&suppressed);
        let node = candidates[rng.below(candidates.len())];
        to_emit.push(Op::Review {
            node,
            rating: (1 + rng.below(4)) as u8,
            elapsed_seq: 1 + rng.below(9) as u64,
        });
    }
    for _ in 0..suppresses_wanted {
        let candidates = live(&suppressed);
        let node = candidates[rng.below(candidates.len())];
        suppressed.insert(node);
        to_emit.push(Op::Suppress { node });
    }

    // Shuffle the non-ingest ops by inserting each at a random position
    // after the ingest block, then replay-adjust: reviews/suppresses must
    // respect the LIVE set at execution time, so filter at replay instead —
    // the driver SKIPS ops whose target was already suppressed (deterministically
    // on both engines), which keeps scripts valid while exercising ordering.
    let mut ordered: Vec<Op> = ops.clone();
    for op in to_emit {
        let pos = ingests.min(ordered.len());
        let at = pos + rng.below(ordered.len() - pos + 1);
        ordered.insert(at, op);
    }

    // Guarantee the minimum op count by padding with reviews (always valid
    // on some live node).
    while ordered.len() < min_ops {
        let candidates = live(&suppressed);
        let node = candidates[rng.below(candidates.len())];
        ordered.push(Op::Review {
            node,
            rating: (1 + rng.below(4)) as u8,
            elapsed_seq: 1 + rng.below(9) as u64,
        });
    }

    Script { seed, ops: ordered }
}

// ============================================================================
// Engine snapshot (the parity artifact)
// ============================================================================

/// Integer FSRS trajectory facts asserted across engines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReviewTrajectory {
    /// Total reviews applied.
    pub review_count: u32,
    /// Rating-1 lapses.
    pub lapse_count: u32,
    /// `learning` / `review` / `relearning`.
    pub phase: String,
}

/// Everything one engine observed for one script, as plain sorted data.
#[derive(Debug, Clone, Serialize)]
pub struct EngineSnapshot {
    /// `sqlite` or `strata`.
    pub engine: &'static str,
    /// Number of ingested nodes.
    pub node_count: usize,
    /// blake3(content) hex, sorted.
    pub content_digests_sorted: Vec<String>,
    /// `(digest_src, digest_dst, link)` sorted.
    pub edges_sorted: Vec<String>,
    /// Suppressed content digests, sorted.
    pub suppressed_sorted: Vec<String>,
    /// Per-content-digest integer trajectory, sorted by digest.
    pub reviews: BTreeMap<String, ReviewTrajectory>,
    /// Per-content-digest Q32.32-quantized stability/difficulty, sorted.
    /// Engine-specific (pinned weight tables differ by design); compared
    /// run-over-run per engine, NOT across engines.
    pub fsrs_quantized: Vec<(String, i64, i64)>,
    /// Engine-specific extras (strata: verify outcome, frame/checkpoint
    /// counts; sqlite: store counter cross-checks), sorted keys.
    pub extras: BTreeMap<String, String>,
}

impl EngineSnapshot {
    /// Canonical byte encoding: one `key\tvalue...` line per fact, sorted
    /// fields, LF-joined. Two runs of the same engine on the same script
    /// must produce identical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("engine\t{}", self.engine));
        lines.push(format!("node_count\t{}", self.node_count));
        for d in &self.content_digests_sorted {
            lines.push(format!("digest\t{d}"));
        }
        for e in &self.edges_sorted {
            lines.push(format!("edge\t{e}"));
        }
        for s in &self.suppressed_sorted {
            lines.push(format!("suppressed\t{s}"));
        }
        for (digest, t) in &self.reviews {
            lines.push(format!(
                "review\t{digest}\t{}\t{}\t{}",
                t.review_count, t.lapse_count, t.phase
            ));
        }
        for (digest, s, d) in &self.fsrs_quantized {
            lines.push(format!("quantized\t{digest}\t{s}\t{d}"));
        }
        for (k, v) in &self.extras {
            lines.push(format!("extra\t{k}\t{v}"));
        }
        lines.sort();
        let mut out = lines.join("\n").into_bytes();
        out.push(b'\n');
        out
    }

    /// Assert cross-engine parity of every engine-comparable field.
    /// Panics with a field-named message on the first divergence.
    pub fn assert_parity(&self, other: &EngineSnapshot) {
        assert_eq!(
            self.node_count, other.node_count,
            "parity: node counts differ ({} vs {})",
            self.engine, other.engine
        );
        assert_eq!(
            self.content_digests_sorted, other.content_digests_sorted,
            "parity: sorted content digests differ ({} vs {})",
            self.engine, other.engine
        );
        assert_eq!(
            self.edges_sorted, other.edges_sorted,
            "parity: sorted edge sets differ ({} vs {})",
            self.engine, other.engine
        );
        assert_eq!(
            self.suppressed_sorted, other.suppressed_sorted,
            "parity: suppressed sets differ ({} vs {})",
            self.engine, other.engine
        );
        // Integer FSRS trajectory: counts always; phase only when the caller
        // has verified the script avoids the Hard-review delta — phase is
        // compared separately via assert_phase_parity.
        let strip_phase = |s: &EngineSnapshot| -> BTreeMap<String, (u32, u32)> {
            s.reviews
                .iter()
                .map(|(d, t)| (d.clone(), (t.review_count, t.lapse_count)))
                .collect()
        };
        assert_eq!(
            strip_phase(self),
            strip_phase(other),
            "parity: review/lapse counts per digest differ ({} vs {})",
            self.engine,
            other.engine
        );
    }

    /// Phase-level parity (call only for scripts that avoid rating-2
    /// reviews on non-Review cards — see the module docs).
    pub fn assert_phase_parity(&self, other: &EngineSnapshot) {
        assert_eq!(
            self.reviews, other.reviews,
            "phase parity: trajectories (incl. phase) differ ({} vs {})",
            self.engine, other.engine
        );
    }
}

fn digest_hex(content: &str) -> String {
    hex(blake3::hash(content.as_bytes()).as_bytes())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sqlite_phase(state: LearningState) -> &'static str {
    match state {
        LearningState::Learning => "learning",
        LearningState::Relearning => "relearning",
        LearningState::Review => "review",
        // `LearningState` is `#[non_exhaustive]`; treat anything else as new.
        _ => "new",
    }
}

fn strata_phase(phase: CardPhase) -> &'static str {
    match phase {
        CardPhase::Learning => "learning",
        CardPhase::Relearning => "relearning",
        CardPhase::Review => "review",
    }
}

// ============================================================================
// Engine A: vestige-core SQLite Storage
// ============================================================================

/// Replay `script` against the real SQLite engine rooted at `root` (a fresh
/// directory). Reviews go through BOTH the real `mark_reviewed` store write
/// AND a pure `FSRSScheduler` shadow fold with the script's elapsed time;
/// the store's integer counters must agree with the shadow after every
/// review (the shadow is the deterministic reference; the store's stability
/// column is wall-clock-fed and excluded from the artifact by design).
pub fn run_sqlite(script: &Script, root: &Path) -> EngineSnapshot {
    let storage =
        SqliteMemoryStore::new(Some(root.join("sqlite.db"))).expect("open sqlite store in fresh dir");

    let mut node_ids: Vec<String> = Vec::new();
    let mut digests: Vec<String> = Vec::new();
    let mut suppressed: BTreeSet<usize> = BTreeSet::new();
    let mut edges: BTreeMap<(usize, usize), String> = BTreeMap::new();
    let mut shadow: HashMap<usize, FSRSState> = HashMap::new();
    // The shared kernel clock (1 seq = 1 day): every review advances it by
    // the op's `elapsed_seq`, so the per-card elapsed the SQLite scheduler
    // core sees is IDENTICAL to the strata kernel's `event_seq - last_seq`.
    let mut clock: u64 = 0;
    let mut card_last: HashMap<usize, u64> = HashMap::new();
    // Wall-clock shape of the STORE's FSRS path (see the tie comment above):
    // under fast test cadence every post-first review is same-day, so the
    // store's total lapses under-count vs the pinned-time fold.
    let mut store_lapses_total: u32 = 0;
    let scheduler = FSRSScheduler::default();

    for op in &script.ops {
        match op {
            Op::Ingest {
                content,
                node_type,
                tags,
            } => {
                let node = storage
                    .ingest(IngestInput {
                        content: content.clone(),
                        node_type: node_type.clone(),
                        tags: tags.clone(),
                        ..Default::default()
                    })
                    .expect("sqlite ingest");
                node_ids.push(node.id);
                digests.push(digest_hex(content));
            }
            Op::Edge { src, dst, link } => {
                if suppressed.contains(src) || suppressed.contains(dst) {
                    continue; // deterministic skip on both engines
                }
                storage
                    .save_typed_edge(&TypedEdge::new(
                        node_ids[*src].clone(),
                        node_ids[*dst].clone(),
                        link,
                    ))
                    .expect("sqlite typed edge save");
                edges.insert((*src, *dst), link.clone());
            }
            Op::Review {
                node,
                rating,
                elapsed_seq,
            } => {
                if suppressed.contains(node) {
                    continue;
                }
                assert!(
                    *elapsed_seq >= 1,
                    "script invariant: elapsed_seq >= 1 keeps both engines off the same-day path"
                );
                clock += elapsed_seq;
                let elapsed_days = clock - card_last.get(node).copied().unwrap_or(0);
                card_last.insert(*node, clock);
                let rating = Rating::from_i32(i32::from(*rating)).expect("rating 1..=4");
                // 1) Real store write (wall-clock elapsed inside).
                let stored = storage
                    .mark_reviewed(&node_ids[*node], rating)
                    .expect("sqlite mark_reviewed");
                // 2) Deterministic shadow fold through the SAME scheduler
                //    core with the shared-clock elapsed time.
                // `FSRSState::default()` is a New card (reps 0); `review` routes New
                // cards through the first-review path, so the default's
                // numeric stand-ins never influence the fold.
                let state = shadow.entry(*node).or_default();
                let result = scheduler.review(state, rating, elapsed_days as f64, None);
                *state = result.state;
                // 3) Integer counter tie between store and shadow — reps
                //    only. LAPSES deliberately NOT tied: `mark_reviewed`
                //    derives elapsed from the WALL CLOCK, so rapid
                //    successive reviews classify as same-day and the store
                //    never increments lapses on them, while the pinned-time
                //    shadow (and strata) do. That divergence is the
                //    non-determinism the default flip retires; it is
                //    recorded in `extras.store_lapses_total` instead of
                //    being asserted away.
                assert_eq!(
                    stored.reps, state.reps,
                    "sqlite store reps must match the deterministic shadow"
                );
                store_lapses_total = u32::try_from(stored.lapses).expect("lapses fits u32");
            }
            Op::Suppress { node } => {
                if suppressed.contains(node) {
                    continue;
                }
                storage
                    .suppress_memory(&node_ids[*node])
                    .expect("sqlite suppress");
                suppressed.insert(*node);
            }
        }
    }

    // Edge read-back: prove the store persisted exactly the declared set.
    let mut id_to_idx: HashMap<&str, usize> = HashMap::new();
    for (idx, id) in node_ids.iter().enumerate() {
        id_to_idx.insert(id.as_str(), idx);
    }
    let mut read_back: BTreeSet<String> = BTreeSet::new();
    for id in &node_ids {
        for edge in storage
            .edges_for(id, EdgeDirection::Both)
            .expect("edge read-back")
        {
            let (Some(s), Some(d)) = (
                id_to_idx.get(edge.source_id.as_str()),
                id_to_idx.get(edge.target_id.as_str()),
            ) else {
                continue;
            };
            read_back.insert(format!(
                "{}\t{}\t{}",
                digests[*s], digests[*d], edge.link_type
            ));
        }
    }
    let declared: BTreeSet<String> = edges
        .iter()
        .map(|((s, d), link)| format!("{}\t{}\t{}", digests[*s], digests[*d], link))
        .collect();
    assert_eq!(
        read_back, declared,
        "sqlite edge read-back must equal the declared edge set"
    );

    let mut reviews = BTreeMap::new();
    let mut fsrs_quantized = Vec::new();
    for (idx, digest) in digests.iter().enumerate() {
        if let Some(state) = shadow.get(&idx) {
            reviews.insert(
                digest.clone(),
                ReviewTrajectory {
                    review_count: u32::try_from(state.reps).expect("reps fits u32"),
                    lapse_count: u32::try_from(state.lapses).expect("lapses fits u32"),
                    phase: sqlite_phase(state.state).to_string(),
                },
            );
            fsrs_quantized.push((
                digest.clone(),
                to_q32_32(state.stability),
                to_q32_32(state.difficulty),
            ));
        }
    }
    fsrs_quantized.sort();

    let mut extras = BTreeMap::new();
    extras.insert(
        "store_lapses_total".to_string(),
        store_lapses_total.to_string(),
    );
    extras.insert(
        "store_nodes".to_string(),
        storage
            .get_all_nodes(10_000, 0)
            .expect("node count read-back")
            .len()
            .to_string(),
    );

    EngineSnapshot {
        engine: "sqlite",
        node_count: node_ids.len(),
        content_digests_sorted: {
            let mut d = digests.clone();
            d.sort();
            d
        },
        edges_sorted: read_back.into_iter().collect(),
        suppressed_sorted: suppressed.iter().map(|i| digests[*i].clone()).collect(),
        reviews,
        fsrs_quantized,
        extras,
    }
}

// ============================================================================
// Engine B: strata (kernel + gate directly, materialized + verified)
// ============================================================================

/// The permissive harness policy (one Allow rule, unbounded blast radius).
pub fn permissive_policy() -> Policy {
    Policy {
        rules: vec![Rule {
            match_kind: ANY_KIND,
            match_params_hash_prefix: WILDCARD_PREFIX,
            max_blast_radius: u32::MAX,
            forbid_forgotten_lessons: false,
            require_human: false,
            verdict: Verdict::Allow,
        }],
    }
}

/// Replay `script` against the strata engine (gate runtime + kernel fold),
/// materialize the store into `<root>/strata-store`, verify it with
/// `verify_store`, and return the snapshot plus the verify report.
pub fn run_strata(script: &Script, root: &Path) -> (EngineSnapshot, VerifyReport) {
    let policy = permissive_policy();
    let mut runtime = GateRuntime::new(MemLog::new(), policy.clone());
    let kernel = Kernel::<ReviewEvent>::for_version(ALGO_V1).expect("v1 kernel resolves");
    let mut state = State::default();
    let mut kernel_seq: u64 = 0;

    let mut fact_ids: Vec<u64> = Vec::new(); // node idx -> WRITE effect seq
    let mut digests: Vec<String> = Vec::new();
    let mut suppressed: BTreeSet<usize> = BTreeSet::new();
    let mut edges: BTreeMap<(usize, usize), String> = BTreeMap::new();
    let mut kernel_records: Vec<KernelRecord> = Vec::new();

    // Gated propose -> gate -> effect for a mutating op.
    fn commit_gated(runtime: &mut GateRuntime<MemLog>, propose: Propose) -> (u64, u64, u64) {
        let action_hash = propose.action_hash;
        let payload_digest = propose.params_hash;
        let ack = runtime.commit_propose(propose);
        let gate = runtime.commit_gate(ack.seq).expect("gate commits");
        let effect = runtime
            .commit_effect(EffectRecord {
                propose_seq: ack.seq,
                gate_seq: gate.seq,
                action_hash,
                payload_digest,
            })
            .expect("admission allows under the harness policy");
        (ack.seq, gate.seq, effect.seq)
    }

    for op in &script.ops {
        match op {
            Op::Ingest {
                content,
                node_type,
                tags,
            } => {
                let digest = blake3::hash(content.as_bytes());
                let mut params = blake3::Hasher::new();
                params.update(node_type.as_bytes());
                for tag in tags {
                    params.update(tag.as_bytes());
                }
                let (_, _, effect_seq) = commit_gated(
                    &mut runtime,
                    Propose {
                        action_hash: *digest.as_bytes(),
                        action_kind: strata_gate::ActionKindCode::WRITE,
                        params_hash: *params.finalize().as_bytes(),
                        context: vec![],
                    },
                );
                fact_ids.push(effect_seq);
                digests.push(hex(digest.as_bytes()));
            }
            Op::Edge { src, dst, link } => {
                if suppressed.contains(src) || suppressed.contains(dst) {
                    continue;
                }
                let (_, _, _) = commit_gated(
                    &mut runtime,
                    Propose {
                        action_hash: *blake3::hash(
                            format!("{link}:{}:{}", digests[*src], digests[*dst]).as_bytes(),
                        )
                        .as_bytes(),
                        action_kind: strata_gate::ActionKindCode::EFFECT,
                        params_hash: *blake3::hash(link.as_bytes()).as_bytes(),
                        context: vec![fact_ids[*src], fact_ids[*dst]],
                    },
                );
                edges.insert((*src, *dst), link.clone());
            }
            Op::Review {
                node,
                rating,
                elapsed_seq,
            } => {
                if suppressed.contains(node) {
                    continue;
                }
                assert!(
                    *elapsed_seq >= 1,
                    "script invariant: elapsed_seq >= 1 keeps both engines off the same-day path"
                );
                // Same shared clock as the SQLite engine: the strata event
                // seq IS time, so card elapsed = event_seq - last_seq matches
                // the SQLite shadow's elapsed days exactly.
                kernel_seq += elapsed_seq;
                let event = ReviewEvent {
                    card_id: fact_ids[*node],
                    rating: *rating,
                    event_seq: kernel_seq,
                };
                kernel.apply(&mut state, &event);
                let payload = borsh::to_vec(&event).expect("borsh ReviewEvent is infallible");
                kernel_records.push(KernelRecord {
                    seq: event.event_seq,
                    event_hash: *blake3::hash(&payload).as_bytes(),
                    payload,
                });
            }
            Op::Suppress { node } => {
                if suppressed.contains(node) {
                    continue;
                }
                commit_gated(
                    &mut runtime,
                    Propose {
                        action_hash: *blake3::hash(format!("retire:{}", digests[*node]).as_bytes())
                            .as_bytes(),
                        action_kind: strata_gate::ActionKindCode::RETIRE,
                        params_hash: [0; 32],
                        context: vec![fact_ids[*node]],
                    },
                );
                suppressed.insert(*node);
            }
        }
    }

    let log = runtime.into_inner();

    // Seal the kernel side: genesis + one checkpoint per 3 review events
    // (multiple segments exercise segment-versioned replay) + the tail.
    // The tail may coincide with a %3 boundary, so never seal the same
    // log_seq twice (strictly increasing is a verify invariant).
    let mut checkpoints = vec![Checkpoint::genesis(ALGO_V1)];
    {
        let mut fold = State::default();
        let total = kernel_records.len();
        for (i, record) in kernel_records.iter().enumerate() {
            kernel.apply(
                &mut fold,
                &borsh::from_slice::<ReviewEvent>(&record.payload).expect("record payload decodes"),
            );
            let is_tail = i + 1 == total;
            if record.seq % 3 == 0 || is_tail {
                let last_log_seq = checkpoints.last().expect("genesis present").log_seq;
                if record.seq > last_log_seq {
                    let prev = checkpoints.last().expect("genesis present").hash();
                    checkpoints.push(Checkpoint::seal(ALGO_V1, record.seq, prev, &fold));
                }
            }
        }
    }

    let gate_frames: Vec<GateFrame> = log
        .all()
        .iter()
        .map(|ev| GateFrame {
            seq: ev.seq,
            kind: ev.kind,
            payload: ev.payload.clone(),
        })
        .collect();
    let last = gate_frames.last().cloned();
    let gate_head = last
        .as_ref()
        .map(|f| gate_frame_hash(f.seq, f.kind, &f.payload))
        .unwrap_or(*blake3::hash(b"").as_bytes());

    let store_dir = root.join("strata-store");
    write_store(
        &store_dir,
        &StoreFiles {
            kernel_records: kernel_records.clone(),
            checkpoints: checkpoints.clone(),
            kernel_head: checkpoints.last().expect("genesis at minimum").hash(),
            gate_frames: gate_frames.clone(),
            policy,
            gate_head,
        },
    )
    .expect("materialize strata store");
    let report = strata_verify::verify_store(&store_dir);

    // Snapshot from the kernel fold state (the same state the checkpoints
    // seal — and the report re-verifies on disk).
    let mut reviews = BTreeMap::new();
    let mut fsrs_quantized = Vec::new();
    for (idx, digest) in digests.iter().enumerate() {
        if let Some(card) = state.cards.get(&fact_ids[idx]) {
            reviews.insert(
                digest.clone(),
                ReviewTrajectory {
                    review_count: card.review_count,
                    lapse_count: card.lapse_count,
                    phase: strata_phase(card.phase).to_string(),
                },
            );
            fsrs_quantized.push((digest.clone(), card.stability_q, card.difficulty_q));
        }
    }
    fsrs_quantized.sort();

    let mut extras = BTreeMap::new();
    extras.insert("verify_ok".to_string(), report.ok().to_string());
    extras.insert("gate_frames".to_string(), gate_frames.len().to_string());
    extras.insert("checkpoints".to_string(), checkpoints.len().to_string());

    let snapshot = EngineSnapshot {
        engine: "strata",
        node_count: fact_ids.len(),
        content_digests_sorted: {
            let mut d = digests.clone();
            d.sort();
            d
        },
        edges_sorted: edges
            .iter()
            .map(|((s, d), link)| format!("{}\t{}\t{}", digests[*s], digests[*d], link))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        suppressed_sorted: suppressed.iter().map(|i| digests[*i].clone()).collect(),
        reviews,
        fsrs_quantized,
        extras,
    };
    (snapshot, report)
}

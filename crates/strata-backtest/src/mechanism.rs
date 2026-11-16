//! Rank trait, subjects, and the origin firewall.
//!
//! A mechanism sees a prefix. It does not see suffix frames. The firewall
//! rejects a proof that cites a seq past the cut.

use std::collections::BTreeMap;

use strata_store::{StoreError, StrataStore};

use crate::canon::hex_bytes;
use crate::events::{Body, DreamWindow, Rec, events_from_admitted};

/// A state: a node id or an exact anchor path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Subject {
    /// Memory id.
    Node(String),
    /// Exact file path, or a non-node edge endpoint.
    Path(String),
}

impl Subject {
    /// `node:{id}` or `path:{path}`.
    pub fn canon(&self) -> String {
        match self {
            Subject::Node(id) => format!("node:{id}"),
            Subject::Path(path) => format!("path:{path}"),
        }
    }
}

/// Frame seqs that justify a rank or a backup. Sorted, unique.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// Cited store-frame seqs.
    pub cited_seqs: Vec<u64>,
}

/// Build a proof, dropping duplicates.
pub fn proof_of(seqs: impl IntoIterator<Item = u64>) -> Proof {
    let mut cited_seqs: Vec<u64> = seqs.into_iter().collect();
    cited_seqs.sort_unstable();
    cited_seqs.dedup();
    Proof { cited_seqs }
}

/// Every cited seq is inside the prefix.
pub fn firewall_accepts(proof: &Proof, bound: u64) -> bool {
    proof.cited_seqs.iter().all(|seq| *seq <= bound)
}

/// Reject the run when any proof looks past the cut.
pub fn require_firewall(bound: u64, ranked: &[(Subject, Proof)]) -> Result<(), String> {
    for (subject, proof) in ranked {
        if !firewall_accepts(proof, bound) {
            return Err(format!(
                "firewall: {} cites a frame past {bound}",
                subject.canon()
            ));
        }
    }
    Ok(())
}

/// What a mechanism is allowed to condition on.
#[derive(Clone, Debug)]
pub struct Query {
    /// Corpus id, used only to name a seeded shuffle.
    pub corpus_id: String,
    /// Cut bound. The prefix is already folded at this seq.
    pub bound_seq: u64,
    /// Suffix horizon. Rankers must not read suffix events from this.
    pub horizon: u64,
}

/// Prefix fold plus the events that built it.
#[derive(Clone, Debug)]
pub struct Prefix {
    /// Cut bound.
    pub bound_seq: u64,
    /// Greatest folded frame seq.
    pub head_seq: u64,
    /// Chain hash of [`Self::head_seq`] in this log.
    pub head_frame_hash: [u8; 32],
    /// Borsh digest of the folded prefix.
    pub state_digest: [u8; 32],
    /// Clock the prefix fold uses for retrievability.
    pub head_clock_ms: i64,
    /// Admitted rows with `frame_seq <= bound_seq`.
    pub events: Vec<Rec>,
    /// Node id → retrievability at [`Self::head_clock_ms`].
    pub retrievability: BTreeMap<String, f64>,
    /// Dream windows. Empty for both corpora.
    pub windows: Vec<DreamWindow>,
}

impl Prefix {
    /// Lowercase hex of the signed head, for manifests.
    pub fn head_hex(&self) -> String {
        hex_bytes(&self.head_frame_hash)
    }
}

/// `rank` returns subjects best-first, each with a proof.
pub trait Mechanism {
    /// Stable arm id.
    fn id(&self) -> &'static str;

    /// Order the prefix. Must not read frames after `prefix.bound_seq`.
    fn rank(&self, prefix: &Prefix, query: &Query) -> Vec<(Subject, Proof)>;
}

/// Reference arm: subject canon ascending. Proves the trait and the firewall.
#[derive(Clone, Debug, Default)]
pub struct IdOrder;

impl Mechanism for IdOrder {
    fn id(&self) -> &'static str {
        "id_order"
    }

    fn rank(&self, prefix: &Prefix, _query: &Query) -> Vec<(Subject, Proof)> {
        let mut subjects: Vec<Subject> =
            crate::events::candidates(&prefix.events, prefix.bound_seq)
                .into_iter()
                .collect();
        subjects.sort();
        subjects
            .into_iter()
            .map(|subject| {
                let seq = existence_seq(&prefix.events, &subject).unwrap_or(0);
                (subject, proof_of([seq]))
            })
            .collect()
    }
}

fn existence_seq(events: &[Rec], subject: &Subject) -> Option<u64> {
    for rec in events {
        let hit = match (&rec.body, subject) {
            (Body::Upsert { id, .. }, Subject::Node(node)) => id == node,
            (Body::Anchor { path, .. }, Subject::Path(want)) => path == want,
            (Body::Edge { source, target, .. }, Subject::Path(want)) => {
                source == want || target == want
            }
            (Body::Edge { source, target, .. }, Subject::Node(node)) => {
                source == node || target == node
            }
            (Body::Review { card_id, .. }, Subject::Node(node)) => {
                *card_id == crate::events::card_handle(node)
            }
            _ => false,
        };
        if hit {
            return Some(rec.frame_seq);
        }
    }
    None
}

/// Events of a store, via `as_of(u64::MAX)` so a live open's empty
/// `admitted_frames` buffer is not mistaken for an empty log.
pub fn admitted_events(store: &StrataStore) -> Result<Vec<Rec>, StoreError> {
    let full = store.as_of(u64::MAX)?;
    Ok(events_from_admitted(full.admitted_frames()))
}

/// Project `events` onto an already-folded `as_of(bound)` scratch.
pub fn project_prefix(
    fold: &StrataStore,
    events: &[Rec],
    bound: u64,
    windows: &[DreamWindow],
) -> Result<Prefix, StoreError> {
    let mut retrievability = BTreeMap::new();
    for node in fold.nodes() {
        if let Some(value) = fold.retrievability(&node.id)? {
            retrievability.insert(node.id, value);
        }
    }
    let prefix_events = events
        .iter()
        .filter(|rec| rec.frame_seq <= bound)
        .cloned()
        .collect();
    Ok(Prefix {
        bound_seq: bound,
        head_seq: fold.prefix_head_seq(),
        head_frame_hash: fold.prefix_head_frame_hash(),
        state_digest: fold.state_digest(),
        head_clock_ms: fold.head_clock_ms(),
        events: prefix_events,
        retrievability,
        windows: windows.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firewall_rejects_a_seq_past_the_bound() {
        let ok = proof_of([1, 4, 4]);
        assert_eq!(ok.cited_seqs, vec![1, 4]);
        assert!(firewall_accepts(&ok, 4));
        let bad = proof_of([5]);
        assert!(!firewall_accepts(&bad, 4));
        let err = require_firewall(4, &[(Subject::Node("m".into()), bad)]).unwrap_err();
        assert!(err.contains("firewall"));
    }

    #[test]
    fn subject_ord_matches_canon_order() {
        let node = Subject::Node("z".into());
        let path = Subject::Path("a".into());
        assert!(node < path);
        assert!(node.canon() < path.canon());
        let a = Subject::Node("a".into());
        let b = Subject::Node("b".into());
        assert!(a < b);
        assert!(a.canon() < b.canon());
    }
}

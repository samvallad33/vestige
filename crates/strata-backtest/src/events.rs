//! Recorded events projected out of admitted store frames.
//!
//! Content bytes are not copied. Rankers never see them.

use std::collections::{BTreeMap, BTreeSet};

use strata_store::{AdmittedFrame, StoreOp};

use crate::mechanism::Subject;
use crate::protocol::{HORIZON, SUFFIX_EDGE_LINKS, is_typed_link};

/// Half-open frame interval `[start_seq, end_seq)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DreamWindow {
    /// First excluded review seq.
    pub start_seq: u64,
    /// First seq after the window.
    pub end_seq: u64,
}

/// Whether `seq` falls inside any dream window.
pub fn in_dream(seq: u64, windows: &[DreamWindow]) -> bool {
    windows
        .iter()
        .any(|window| seq >= window.start_seq && seq < window.end_seq)
}

/// One recorded fact a ranker is allowed to see.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Body {
    /// Node upsert. Content is omitted.
    Upsert {
        /// Node id.
        id: String,
        /// Scope string, exact.
        scope: String,
        /// `node_type`, exact.
        node_type: String,
        /// Tags as stored (already sorted by the fold).
        tags: Vec<String>,
    },
    /// Explicit review. Dream-window filtering is the caller's.
    Review {
        /// Card handle (`blake3` of the node id).
        card_id: u64,
        /// Rating 1..=4.
        rating: u8,
    },
    /// Typed edge.
    Edge {
        /// Source id, exact.
        source: String,
        /// Target id, exact. A non-node target is a path subject.
        target: String,
        /// `link_type`, exact.
        link: String,
    },
    /// One anchor row.
    Anchor {
        /// Memory the anchor names.
        node_id: String,
        /// Exact file path.
        path: String,
    },
}

/// An admitted row. `ord` breaks ties inside one frame (anchor batches).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rec {
    /// Store-frame seq.
    pub frame_seq: u64,
    /// Row index inside the frame.
    pub ord: u32,
    /// Payload the rankers may read.
    pub body: Body,
}

/// Card handle: first 8 bytes of `blake3(id)`, little-endian.
pub fn card_handle(id: &str) -> u64 {
    let digest = blake3::hash(id.as_bytes());
    u64::from_le_bytes(
        digest.as_bytes()[0..8]
            .try_into()
            .expect("blake3 is 32 bytes"),
    )
}

/// Project admitted frames. Frames that are not visits, edges, or anchors are dropped.
pub fn events_from_admitted(frames: &[AdmittedFrame]) -> Vec<Rec> {
    let mut out = Vec::new();
    for frame in frames {
        match &frame.op {
            StoreOp::UpsertNode { record } => out.push(Rec {
                frame_seq: frame.frame_seq,
                ord: 0,
                body: Body::Upsert {
                    id: record.id.clone(),
                    scope: record.scope.clone(),
                    node_type: record.node_type.clone(),
                    tags: record.tags.clone(),
                },
            }),
            StoreOp::ReviewNode {
                card_id,
                rating,
                reviewed_at_ms: _,
            } => out.push(Rec {
                frame_seq: frame.frame_seq,
                ord: 0,
                body: Body::Review {
                    card_id: *card_id,
                    rating: *rating,
                },
            }),
            StoreOp::SaveEdge { edge } => out.push(Rec {
                frame_seq: frame.frame_seq,
                ord: 0,
                body: Body::Edge {
                    source: edge.source_id.clone(),
                    target: edge.target_id.clone(),
                    link: edge.link_type.clone(),
                },
            }),
            StoreOp::RecordAnchors { anchors } | StoreOp::ReplaceAnchors { anchors, .. } => {
                for (ord, anchor) in anchors.iter().enumerate() {
                    out.push(Rec {
                        frame_seq: frame.frame_seq,
                        ord: u32::try_from(ord).unwrap_or(u32::MAX),
                        body: Body::Anchor {
                            node_id: anchor.node_id.clone(),
                            path: anchor.file_path.clone(),
                        },
                    });
                }
            }
            StoreOp::SupersedeNode { .. }
            | StoreOp::UpsertIntentions { .. }
            | StoreOp::RecordAnchorVerdict { .. } => {}
        }
    }
    out
}

/// Node id for a card, using the latest upsert at `seq <= bound` (or any seq
/// when `bound` is `None`).
pub fn card_nodes(events: &[Rec], bound: Option<u64>) -> BTreeMap<u64, String> {
    let mut map = BTreeMap::new();
    for rec in events {
        if bound.is_some_and(|limit| rec.frame_seq > limit) {
            break;
        }
        if let Body::Upsert { id, .. } = &rec.body {
            map.insert(card_handle(id), id.clone());
        }
    }
    map
}

fn node_ids_through(events: &[Rec], bound: u64) -> BTreeSet<String> {
    let mut nodes = BTreeSet::new();
    for rec in events {
        if rec.frame_seq > bound {
            break;
        }
        if let Body::Upsert { id, .. } = &rec.body {
            nodes.insert(id.clone());
        }
    }
    nodes
}

fn endpoint(id: &str, nodes: &BTreeSet<String>) -> Subject {
    if nodes.contains(id) {
        Subject::Node(id.to_string())
    } else {
        Subject::Path(id.to_string())
    }
}

/// Prefix candidate set (section 5 of the preregistration).
pub fn candidates(events: &[Rec], bound: u64) -> BTreeSet<Subject> {
    let nodes = node_ids_through(events, bound);
    let mut out = BTreeSet::new();
    for id in &nodes {
        out.insert(Subject::Node(id.clone()));
    }
    for rec in events {
        if rec.frame_seq > bound {
            break;
        }
        match &rec.body {
            Body::Anchor { path, .. } => {
                out.insert(Subject::Path(path.clone()));
            }
            Body::Edge { source, target, .. } => {
                out.insert(endpoint(source, &nodes));
                out.insert(endpoint(target, &nodes));
            }
            Body::Upsert { .. } | Body::Review { .. } => {}
        }
    }
    out
}

/// Typed edges with `seq <= bound`.
pub fn typed_edge_count(events: &[Rec], bound: u64) -> usize {
    events
        .iter()
        .take_while(|rec| rec.frame_seq <= bound)
        .filter(|rec| match &rec.body {
            Body::Edge { link, .. } => is_typed_link(link),
            _ => false,
        })
        .count()
}

/// Rating-4 reviews with `seq <= bound`, dream windows removed.
pub fn rating4_count(events: &[Rec], bound: u64, windows: &[DreamWindow]) -> usize {
    events
        .iter()
        .take_while(|rec| rec.frame_seq <= bound)
        .filter(|rec| {
            matches!(rec.body, Body::Review { rating: 4, .. }) && !in_dream(rec.frame_seq, windows)
        })
        .count()
}

/// Suffix occupancy labels in log order. Exact ids only.
pub fn suffix_subjects(
    events: &[Rec],
    bound: u64,
    horizon: u64,
    windows: &[DreamWindow],
) -> Vec<Subject> {
    let end = bound.saturating_add(horizon);
    let mut cards: BTreeMap<u64, String> = BTreeMap::new();
    let mut nodes = BTreeSet::new();
    let mut out = Vec::new();
    for rec in events {
        if rec.frame_seq > end {
            break;
        }
        let in_suffix = rec.frame_seq > bound;
        match &rec.body {
            Body::Upsert { id, .. } => {
                cards.insert(card_handle(id), id.clone());
                nodes.insert(id.clone());
                if in_suffix {
                    out.push(Subject::Node(id.clone()));
                }
            }
            Body::Review { card_id, .. } => {
                if in_suffix && !in_dream(rec.frame_seq, windows) {
                    if let Some(id) = cards.get(card_id) {
                        out.push(Subject::Node(id.clone()));
                    }
                }
            }
            Body::Edge {
                source,
                target,
                link,
            } => {
                if in_suffix && SUFFIX_EDGE_LINKS.contains(&link.as_str()) {
                    out.push(endpoint(source, &nodes));
                    out.push(endpoint(target, &nodes));
                }
            }
            Body::Anchor { path, .. } => {
                if in_suffix {
                    out.push(Subject::Path(path.clone()));
                }
            }
        }
    }
    out
}

/// Suffix reviews only, dream windows removed. This is the return sequence.
pub fn suffix_review_nodes(
    events: &[Rec],
    bound: u64,
    horizon: u64,
    windows: &[DreamWindow],
) -> Vec<Subject> {
    let end = bound.saturating_add(horizon);
    let cards = card_nodes(events, None);
    let mut out = Vec::new();
    for rec in events {
        if rec.frame_seq > end {
            break;
        }
        if rec.frame_seq <= bound || in_dream(rec.frame_seq, windows) {
            continue;
        }
        if let Body::Review { card_id, .. } = &rec.body {
            if let Some(id) = cards.get(card_id) {
                out.push(Subject::Node(id.clone()));
            }
        }
    }
    out
}

/// Default horizon from the pinned table.
pub fn suffix_subjects_default(
    events: &[Rec],
    bound: u64,
    windows: &[DreamWindow],
) -> Vec<Subject> {
    suffix_subjects(events, bound, HORIZON, windows)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn dream_window_strips_the_review_and_not_the_edge() {
        let events = vec![
            upsert(1, "mem-a"),
            review(10, "mem-a", 4),
            Rec {
                frame_seq: 10,
                ord: 1,
                body: Body::Edge {
                    source: "mem-a".into(),
                    target: "dangling-path".into(),
                    link: "touched".into(),
                },
            },
        ];
        let windows = [DreamWindow {
            start_seq: 10,
            end_seq: 11,
        }];
        assert_eq!(rating4_count(&events, 10, &windows), 0);
        assert_eq!(rating4_count(&events, 10, &[]), 1);
        let labels = suffix_subjects(&events, 1, 100, &windows);
        assert_eq!(
            labels,
            vec![
                Subject::Node("mem-a".into()),
                Subject::Path("dangling-path".into()),
            ]
        );
        let with_review = suffix_subjects(&events, 1, 100, &[]);
        assert_eq!(
            with_review
                .iter()
                .filter(|subject| matches!(subject, Subject::Node(_)))
                .count(),
            2
        );
        let reviews = suffix_review_nodes(&events, 0, 100, &windows);
        assert!(reviews.is_empty());
    }

    #[test]
    fn suffix_review_is_the_exact_node_id() {
        let events = vec![
            upsert(1, "mem-keep"),
            upsert(2, "mem-other"),
            review(9, "mem-keep", 3),
        ];
        let labels = suffix_review_nodes(&events, 2, 100, &[]);
        assert_eq!(labels, vec![Subject::Node("mem-keep".into())]);
    }

    #[test]
    fn card_handle_matches_a_known_vector() {
        let first = card_handle("mem-a");
        let second = card_handle("mem-a");
        assert_eq!(first, second);
        assert_ne!(first, card_handle("mem-b"));
    }
}

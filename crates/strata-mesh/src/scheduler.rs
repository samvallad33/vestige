//! Demand over cost. A hint can reorder a transfer. It cannot change admission.

use std::cmp::Ordering;

/// One missing contiguous range on a writer. Cost includes the prefix that
/// has to arrive before the range can fold.
#[derive(Debug, Clone)]
pub struct MissingRange {
    /// Writer did.
    pub writer: String,
    /// First missing seq.
    pub from_seq: u64,
    /// Last missing seq.
    pub to_seq: u64,
    /// Segment number the sender named.
    pub segment: u64,
    /// Bytes in the range.
    pub bytes: u64,
    /// Missing bytes earlier on this writer's chain.
    pub prefix_bytes: u64,
}

/// A recorded local reference into a missing range.
#[derive(Debug, Clone)]
pub struct Demand {
    /// Record that names the handle. This is the proof.
    pub record: String,
    /// Qualified handle whose origin is in some missing range.
    pub handle: String,
    /// Writer the handle belongs to.
    pub writer: String,
}

/// A sender's claim that `handle` lives in `segment`.
#[derive(Debug, Clone)]
pub struct IndexHint {
    /// Sender did.
    pub sender: String,
    /// Segment number.
    pub segment: u64,
    /// Handle the sender placed in that segment.
    pub handle: String,
}

/// One scheduling choice and the records that justified it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleDecision {
    /// Writer did.
    pub writer: String,
    /// Segment number.
    pub segment: u64,
    /// demand / cost, scaled by the other side's cost so it stays integral.
    pub score: u128,
    /// Referencing records, in the order they were given.
    pub proof: Vec<String>,
}

/// Rank missing ranges by demand / (range bytes + prefix bytes).
/// Ties break by writer did bytes, then segment number.
pub fn schedule(ranges: &[MissingRange], demand: &[Demand], pulls: &[String]) -> Vec<ScheduleDecision> {
    let mut ranked: Vec<ScheduleDecision> = ranges
        .iter()
        .map(|range| {
            let proof: Vec<String> = demand
                .iter()
                .filter(|item| item.writer == range.writer && range_contains(range, &item.handle))
                .map(|item| item.record.clone())
                .chain(pulls.iter().filter(|handle| range_contains(range, handle)).cloned())
                .collect();
            let demand_n = proof.len() as u128;
            let cost = range.bytes.saturating_add(range.prefix_bytes).max(1) as u128;
            ScheduleDecision {
                writer: range.writer.clone(),
                segment: range.segment,
                score: demand_n.saturating_mul(1_000_000) / cost,
                proof,
            }
        })
        .collect();
    ranked.sort_by(cmp_decision);
    ranked
}

fn range_contains(range: &MissingRange, handle: &str) -> bool {
    handle.contains(&range.writer)
}

/// `true` when `handle` occurs in the segment payload bytes.
pub fn segment_contains(segment: &[u8], handle: &str) -> bool {
    let needle = handle.as_bytes();
    segment.windows(needle.len()).any(|window| window == needle)
}

/// A hint that names a handle the segment does not contain.
pub fn index_lies(hint: &IndexHint, segment: &[u8]) -> bool {
    !segment_contains(segment, &hint.handle)
}

/// Compare two scores the way the sort does, for tests that want the raw order.
pub fn cmp_decision(left: &ScheduleDecision, right: &ScheduleDecision) -> Ordering {
    right
        .score
        .cmp(&left.score)
        .then_with(|| left.writer.as_bytes().cmp(right.writer.as_bytes()))
        .then(left.segment.cmp(&right.segment))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(writer: &str, segment: u64, bytes: u64, prefix: u64) -> MissingRange {
        MissingRange {
            writer: writer.into(),
            from_seq: 1,
            to_seq: 2,
            segment,
            bytes,
            prefix_bytes: prefix,
        }
    }

    #[test]
    fn a_demanded_range_jumps_the_queue_and_ties_break_by_writer() {
        let ranges = vec![
            range("did:key:zb", 2, 100, 0),
            range("did:key:za", 9, 100, 0),
            range("did:key:za", 1, 10_000, 0),
        ];
        let demand = vec![Demand {
            record: "edge-1".into(),
            handle: "did:key:za/mem-1".into(),
            writer: "did:key:za".into(),
        }];
        let ranked = schedule(&ranges, &demand, &[]);
        assert_eq!(ranked[0].segment, 9, "the demanded short range is first");
        assert_eq!(ranked[0].proof, vec!["edge-1".to_string()]);
        assert_eq!(ranked[1].writer, "did:key:za");
        assert_eq!(ranked[1].segment, 1);
        assert_eq!(ranked[2].writer, "did:key:zb");
    }

    #[test]
    fn equal_demand_breaks_ties_by_writer_bytes_then_segment() {
        let ranges = vec![
            range("did:key:zb", 1, 50, 50),
            range("did:key:za", 2, 80, 20),
            range("did:key:za", 1, 100, 0),
        ];
        let ranked = schedule(&ranges, &[], &[]);
        assert!(ranked.iter().all(|item| item.score == ranked[0].score));
        assert_eq!(ranked[0].writer, "did:key:za");
        assert_eq!(ranked[0].segment, 1);
        assert_eq!(ranked[1].segment, 2);
        assert_eq!(ranked[2].writer, "did:key:zb");
    }

    #[test]
    fn a_lying_index_is_detected_without_changing_bytes() {
        let segment = b"frame-bytes-for-mem-real";
        let hint = IndexHint {
            sender: "did:key:zpeer".into(),
            segment: 3,
            handle: "mem-missing".into(),
        };
        assert!(index_lies(&hint, segment));
        assert!(!index_lies(
            &IndexHint {
                handle: "mem-real".into(),
                ..hint.clone()
            },
            segment
        ));
        assert_eq!(segment, b"frame-bytes-for-mem-real");
    }
}

//! Eligible cut bounds and the pinned integer thinning.

use crate::events::{DreamWindow, Rec, candidates, rating4_count, typed_edge_count};
use crate::protocol::{HORIZON, MAX_CUTS, MIN_EDGES, MIN_PROMOTES, MIN_STATES};

/// Event-frame seqs that satisfy the prefix floors and still have a suffix
/// event inside the horizon.
pub fn eligible_bounds(events: &[Rec], windows: &[DreamWindow]) -> Vec<u64> {
    let mut seqs = Vec::new();
    for rec in events {
        if seqs.last().copied() != Some(rec.frame_seq) {
            seqs.push(rec.frame_seq);
        }
    }
    let mut out = Vec::new();
    for bound in seqs {
        if typed_edge_count(events, bound) < MIN_EDGES {
            continue;
        }
        if candidates(events, bound).len() < MIN_STATES {
            continue;
        }
        if rating4_count(events, bound, windows) < MIN_PROMOTES {
            continue;
        }
        let end = bound.saturating_add(HORIZON);
        let has_suffix = events
            .iter()
            .any(|rec| rec.frame_seq > bound && rec.frame_seq <= end);
        if has_suffix {
            out.push(bound);
        }
    }
    out
}

/// Keep at most `max` bounds. Above the cap, index
/// `(i * (len - 1)) / (max - 1)` using `u128`.
pub fn thin(bounds: &[u64], max: usize) -> Vec<u64> {
    if bounds.len() <= max || max == 0 {
        return bounds.to_vec();
    }
    if max == 1 {
        return vec![bounds[0]];
    }
    let last = (bounds.len() - 1) as u128;
    let denom = (max as u128) - 1;
    let mut out = Vec::with_capacity(max);
    for i in 0..max {
        let idx = ((i as u128) * last) / denom;
        out.push(bounds[idx as usize]);
    }
    out
}

/// [`thin`] at [`MAX_CUTS`].
pub fn thin_cuts(bounds: &[u64]) -> Vec<u64> {
    thin(bounds, MAX_CUTS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Body;

    #[test]
    fn thinning_keeps_both_endpoints() {
        let bounds: Vec<u64> = (0..33).collect();
        let kept = thin(&bounds, 32);
        assert_eq!(kept.len(), 32);
        assert_eq!(*kept.first().unwrap(), 0);
        assert_eq!(*kept.last().unwrap(), 32);
        let unchanged = thin(&bounds[..32], 32);
        assert_eq!(unchanged, bounds[..32]);
        assert_eq!(thin(&bounds, 1), vec![0]);
    }

    #[test]
    fn a_short_log_has_no_eligible_cut() {
        let events = vec![Rec {
            frame_seq: 1,
            ord: 0,
            body: Body::Upsert {
                id: "only".into(),
                scope: String::new(),
                node_type: "fact".into(),
                tags: Vec::new(),
            },
        }];
        assert!(eligible_bounds(&events, &[]).is_empty());
    }
}

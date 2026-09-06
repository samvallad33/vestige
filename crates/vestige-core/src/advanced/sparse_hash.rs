//! Deterministic sparse random-hyperplane candidate filtering.
//!
//! Hash collisions nominate pairs for exact cosine scoring; they never prove
//! that two facts are interchangeable. The filter is approximate and can miss
//! near duplicates. Small scans and thresholds below 0.5 use exhaustive pairs.
//! Buckets and per-row candidate sets use O(n) space, even for identical vectors.

use std::collections::{BTreeSet, HashMap};

const BANDS: usize = 32;
const BITS: usize = 8;
const EXACT_SCAN_LIMIT: usize = 128;

/// Ephemeral index: rebuilt from the current embeddings for each scan. No
/// persisted signatures, model coupling, new dependency, or database migration.
pub struct SparseHashIndex {
    signatures: Vec<Option<(usize, [u8; BANDS])>>,
    buckets: HashMap<(usize, usize, u8), Vec<usize>>,
    approximate: bool,
    mask: u8,
}

impl SparseHashIndex {
    /// `minimum_cosine` is the lowest exact cosine the caller could accept.
    /// Lower thresholds use six-bit bands to improve candidate recall.
    pub fn new(vectors: &[&[f32]], minimum_cosine: f32) -> Self {
        let approximate =
            vectors.len() > EXACT_SCAN_LIMIT && minimum_cosine.is_finite() && minimum_cosine >= 0.5;
        let mask = if minimum_cosine < 0.8 { 0x3f } else { 0xff };
        let mut projections = HashMap::new();
        let mut signatures = Vec::with_capacity(vectors.len());
        let mut buckets: HashMap<_, Vec<usize>> = HashMap::new();
        for (index, vector) in vectors.iter().enumerate() {
            let valid = !vector.is_empty()
                && vector.iter().all(|v| v.is_finite())
                && vector.iter().any(|v| *v != 0.0);
            if !valid {
                signatures.push(None);
                continue;
            }
            let mut signature = [0u8; BANDS];
            if approximate {
                let planes = projections
                    .entry(vector.len())
                    .or_insert_with(|| sparse_planes(vector.len()));
                for (bit, plane) in planes.iter().enumerate() {
                    let dot: f64 = plane.iter().map(|&(i, sign)| vector[i] as f64 * sign).sum();
                    if dot > 0.0 {
                        signature[bit / BITS] |= 1 << (bit % BITS);
                    }
                }
                for (band, &value) in signature.iter().enumerate() {
                    buckets
                        .entry((vector.len(), band, value & mask))
                        .or_default()
                        .push(index);
                }
            }
            signatures.push(Some((vector.len(), signature)));
        }
        Self {
            signatures,
            buckets,
            approximate,
            mask,
        }
    }

    pub fn is_approximate(&self) -> bool {
        self.approximate
    }

    /// Unique, ascending indices greater than `index`. Invalid vectors and
    /// different dimensions never reach the cosine pass.
    pub fn candidates(&self, index: usize) -> Vec<usize> {
        let Some(Some((dimension, signature))) = self.signatures.get(index) else {
            return vec![];
        };
        if !self.approximate {
            return ((index + 1)..self.signatures.len())
                .filter(|&j| {
                    self.signatures[j]
                        .as_ref()
                        .is_some_and(|(d, _)| d == dimension)
                })
                .collect();
        }
        let mut candidates = BTreeSet::new();
        for (band, &value) in signature.iter().enumerate() {
            if let Some(bucket) = self.buckets.get(&(*dimension, band, value & self.mask)) {
                let start = bucket.partition_point(|&j| j <= index);
                candidates.extend(&bucket[start..]);
            }
        }
        candidates.into_iter().collect()
    }
}

// Sparse signed projections, with about 1/16 of coordinates nonzero. SplitMix64
// supplies reproducible sampling only; it is not a cryptographic primitive.
fn sparse_planes(dimension: usize) -> Vec<Vec<(usize, f64)>> {
    let mut state = 0x7665_7374_6967_6501u64 ^ dimension as u64;
    (0..BANDS * BITS)
        .map(|_| {
            let mut plane = Vec::new();
            for i in 0..dimension {
                // Sparse planes degenerate on very short vectors: a zero
                // coordinate can dominate their sign. Dense Gaussian planes
                // are cheap here and retain angular discrimination.
                if dimension <= 32 {
                    let u1 = ((next_random(&mut state) >> 11) as f64 + 1.0)
                        / ((1u64 << 53) as f64 + 1.0);
                    let u2 = (next_random(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                    let weight = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                    plane.push((i, weight));
                    continue;
                }
                let random = next_random(&mut state);
                if random & 15 == 0 {
                    plane.push((i, if random & 16 == 0 { 1.0 } else { -1.0 }));
                }
            }
            if plane.is_empty() {
                plane.push(((next_random(&mut state) % dimension as u64) as usize, 1.0));
            }
            plane
        })
        .collect()
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Automatic equivalence deliberately requires nonempty, byte-identical text.
/// Case, punctuation, numbers, negation and word order can all change behavior.
/// Paraphrases stay available for an explicit, reversible merge after review.
pub fn same_memory_text(a: &str, b: &str) -> bool {
    !a.trim().is_empty() && a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_and_mixed_dimensions_never_become_candidates() {
        let vectors: Vec<&[f32]> = vec![
            &[],
            &[0.0],
            &[f32::NAN],
            &[f32::INFINITY],
            &[1.0],
            &[1.0, 0.0],
            &[2.0],
        ];
        let index = SparseHashIndex::new(&vectors, 0.85);
        assert!(!index.is_approximate());
        assert_eq!(index.candidates(4), vec![6]);
        for i in [0, 1, 2, 3, 5, 6, 99] {
            assert!(index.candidates(i).is_empty());
        }
    }

    #[test]
    fn dense_duplicate_bucket_is_complete_unique_and_ordered() {
        let vectors = vec![vec![1.0, -2.0, 3.0]; 2001];
        let refs: Vec<_> = vectors.iter().map(Vec::as_slice).collect();
        let index = SparseHashIndex::new(&refs, 0.85);
        assert!(index.is_approximate());
        assert_eq!(index.candidates(0), (1..2001).collect::<Vec<_>>());
        assert_eq!(index.candidates(1999), vec![2000]);
        assert!(index.candidates(2000).is_empty());
    }

    #[test]
    fn short_vectors_with_zero_coordinates_retain_close_pairs() {
        let mut vectors = vec![vec![0.0, 1.0]; 129];
        vectors[0] = vec![1.0, 0.0];
        vectors[1] = vec![0.92, (1.0f32 - 0.92 * 0.92).sqrt()];
        let refs: Vec<_> = vectors.iter().map(Vec::as_slice).collect();
        let index = SparseHashIndex::new(&refs, 0.85);
        assert!(index.is_approximate());
        assert!(index.candidates(0).contains(&1));
    }

    #[test]
    fn changed_facts_are_never_automatic_equivalence() {
        for (a, b) in [
            ("Enable cache", "Disable cache"),
            ("timeout = 30", "timeout = 300"),
            ("Use /Prod/config", "Use /prod/config"),
            ("A depends on B", "B depends on A"),
            ("Access is allowed", "Access is not allowed"),
            ("x <= 5", "x < 5"),
            ("", ""),
            ("  ", "  "),
        ] {
            assert!(!same_memory_text(a, b));
        }
        assert!(same_memory_text("Use Redis", "Use Redis"));
    }

    #[test]
    fn seeded_recall_and_comparison_reduction_against_known_pairs() {
        // Independent synthetic vectors, plus 200 constructed cosine=0.92
        // pairs. This measures this corpus, not a universal recall guarantee.
        let mut state = 123u64;
        let mut vectors: Vec<Vec<f32>> = (0..800)
            .map(|_| {
                let mut vector: Vec<f32> = (0..256)
                    .map(|_| {
                        (next_random(&mut state) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
                    })
                    .collect();
                let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                for v in &mut vector {
                    *v /= norm;
                }
                vector
            })
            .collect();
        for i in 0..200 {
            let a = &vectors[i];
            let noise = &vectors[i + 400];
            let dot: f32 = a.iter().zip(noise).map(|(a, b)| a * b).sum();
            let mut orthogonal: Vec<_> = a.iter().zip(noise).map(|(a, b)| b - dot * a).collect();
            let norm = orthogonal.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in &mut orthogonal {
                *v /= norm;
            }
            vectors.push(
                a.iter()
                    .zip(orthogonal)
                    .map(|(a, b)| 0.92 * a + (1.0f32 - 0.92 * 0.92).sqrt() * b)
                    .collect(),
            );
        }
        let refs: Vec<_> = vectors.iter().map(Vec::as_slice).collect();
        let index = SparseHashIndex::new(&refs, 0.85);
        let repeat = SparseHashIndex::new(&refs, 0.85);
        let mut candidates = 0;
        let mut recovered = 0;
        for i in 0..vectors.len() {
            let row = index.candidates(i);
            assert_eq!(row, repeat.candidates(i));
            candidates += row.len();
            if i < 200 && row.contains(&(800 + i)) {
                recovered += 1;
            }
        }
        let exhaustive = vectors.len() * (vectors.len() - 1) / 2;
        eprintln!(
            "sparse-hash fixture: {recovered}/200 planted pairs; {candidates}/{exhaustive} cosine candidates"
        );
        assert!(
            recovered >= 196,
            "recall on the seeded fixture fell below 98%"
        );
        assert!(
            candidates < exhaustive / 3,
            "filter should skip over two thirds of this fixture"
        );
    }
}

//! SplitMix64 and the preregistered subseed.

use crate::protocol::BOOTSTRAP_SEED;

/// Deterministic generator named by the preregistration.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Start at `seed`. The first call mixes before it returns.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// `blake3("mattar-evb-v1\\0" || seed_le || tag)` , first 8 bytes, little-endian.
pub fn subseed(tag: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"mattar-evb-v1\0");
    hasher.update(&BOOTSTRAP_SEED.to_le_bytes());
    hasher.update(tag.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[0..8].try_into().expect("8 bytes"))
}

/// Log seed: `blake3` of `mattar-evb-v1/log/{corpus_id}`.
pub fn log_seed(corpus_id: &str) -> [u8; 32] {
    let label = format!("mattar-evb-v1/log/{corpus_id}");
    *blake3::hash(label.as_bytes()).as_bytes()
}

/// Fisher–Yates. Index `j = next_u64() % (i + 1)`. Modulo bias is accepted.
pub fn fisher_yates<T>(items: &mut [T], rng: &mut SplitMix64) {
    let len = items.len();
    if len < 2 {
        return;
    }
    for i in (1..len).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subseed_and_shuffle_are_stable() {
        assert_eq!(
            subseed("need-random/synth-track-v1/1"),
            subseed("need-random/synth-track-v1/1")
        );
        assert_ne!(subseed("a"), subseed("b"));
        let mut left = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut right = left.clone();
        fisher_yates(&mut left, &mut SplitMix64::new(subseed("shuffle")));
        fisher_yates(&mut right, &mut SplitMix64::new(subseed("shuffle")));
        assert_eq!(left, right);
        let mut other = vec![0, 1, 2, 3, 4, 5, 6, 7];
        fisher_yates(&mut other, &mut SplitMix64::new(subseed("other")));
        assert_ne!(left, other);
    }

    #[test]
    fn log_seed_depends_only_on_the_corpus_id() {
        assert_eq!(log_seed("synth-track-v1"), log_seed("synth-track-v1"));
        assert_ne!(log_seed("synth-track-v1"), log_seed("recorded-ops-v1"));
    }
}

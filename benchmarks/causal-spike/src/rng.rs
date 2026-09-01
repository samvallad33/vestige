//! Deterministic SplitMix64. No thread RNG. Same seed → same integer stream.

#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn mix_with(seed: u64, lane: u64) -> Self {
        Self::new(seed ^ lane.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Inclusive range. `hi` must be >= `lo`.
    pub fn gen_range(&mut self, lo: u64, hi: u64) -> u64 {
        debug_assert!(hi >= lo);
        let span = hi.saturating_sub(lo).saturating_add(1);
        lo + self.next_u64() % span
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::new(20260831);
        let mut b = SplitMix64::new(20260831);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = SplitMix64::new(20260831);
        let mut b = SplitMix64::new(20260832);
        assert_ne!(a.next_u64(), b.next_u64());
    }
}

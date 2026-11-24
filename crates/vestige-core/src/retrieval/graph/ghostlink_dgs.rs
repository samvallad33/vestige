//! GhostLink Edge-Physics Core (`Q16.16` integer arithmetic, zero floats):
//!
//! Preserves the shared fixed-point math primitives (`Q16`, `ALPHA_Q16`,
//! `ONE_MINUS_ALPHA_Q16`, `integer_sqrt`, `ochiai_divergence_q16`) used by
//! `ghostcurrent_mdl_q16`, `pit_commensuration_q16`, `taxa`,
//! `temporal_allen`, and `torrens`.

pub(in crate::retrieval) const Q16: u64 = 1 << 16;

#[inline]
pub(in crate::retrieval) fn integer_sqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }
    x
}

/// Integer Ochiai graph divergence `1 - inter / sqrt(|a| * |b|)` in Q16.16 `[0, 65536]`.
/// Maximal (`Q16`) when `(phi_i, phi_j)` has never co-occurred (edge in complement graph G_bar).
pub(in crate::retrieval) fn ochiai_divergence_q16(a: &[u32], b: &[u32]) -> u64 {
    if a.is_empty() || b.is_empty() {
        return Q16;
    }
    let (mut i, mut j, mut inter) = (0usize, 0usize, 0u64);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                inter += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    let denom = integer_sqrt((a.len() as u64).saturating_mul(b.len() as u64)).max(1);
    let sim_q16 = (inter.saturating_mul(Q16) / denom).min(Q16);
    Q16.saturating_sub(sim_q16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ochiai_divergence_is_maximal_in_complement_graph() {
        let a = [1u32, 3, 5];
        let b = [2u32, 4, 6];
        assert_eq!(ochiai_divergence_q16(&a, &b), Q16);
        assert_eq!(ochiai_divergence_q16(&a, &a), 0);
    }
}

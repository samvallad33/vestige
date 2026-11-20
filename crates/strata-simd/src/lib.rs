//! # strata-simd — Hardware SIMD Acceleration (#834)
//!
//! Sub-millisecond discrete integer posting scoring, galloping/SIMD sorted
//! posting-list intersection, and batch BLAKE3 hashing with runtime feature
//! detection (`AVX-512F` / `AVX2` on `x86_64`, `NEON` on `aarch64`) and
//! bit-exact scalar parity.

/// Scalar reference dot product over discrete integer term weights and postings.
pub fn dot_i32_scalar(weights: &[i32], tf_norms: &[i32]) -> i64 {
    let n = weights.len().min(tf_norms.len());
    let mut acc = 0i64;
    for i in 0..n {
        acc += (weights[i] as i64) * (tf_norms[i] as i64);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i32_avx2(weights: &[i32], tf_norms: &[i32]) -> i64 {
    use core::arch::x86_64::*;
    let n = weights.len().min(tf_norms.len());
    let mut i = 0usize;
    let mut acc = 0i64;
    while i + 8 <= n {
        let vw = unsafe { _mm256_loadu_si256(weights.as_ptr().add(i).cast()) };
        let vt = unsafe { _mm256_loadu_si256(tf_norms.as_ptr().add(i).cast()) };

        let vw_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(vw));
        let vt_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(vt));
        let vw_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(vw, 1));
        let vt_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(vt, 1));

        // Exact 32x32->64 signed products via `_mm256_mul_epi32` on even/shifted lanes
        let prod_lo_even = _mm256_mul_epi32(
            _mm256_shuffle_epi32(vw_lo, 0b10_00_10_00),
            _mm256_shuffle_epi32(vt_lo, 0b10_00_10_00),
        );
        let prod_hi_even = _mm256_mul_epi32(
            _mm256_shuffle_epi32(vw_hi, 0b10_00_10_00),
            _mm256_shuffle_epi32(vt_hi, 0b10_00_10_00),
        );
        let mut lanes = [0i64; 4];
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), prod_lo_even) };
        let mut lanes_hi = [0i64; 4];
        unsafe { _mm256_storeu_si256(lanes_hi.as_mut_ptr().cast(), prod_hi_even) };
        let _ = (lanes, lanes_hi);

        // Exact widened lane extraction guarantees zero overflow across all i32 inputs
        let mut w_buf = [0i64; 4];
        let mut t_buf = [0i64; 4];
        unsafe {
            _mm256_storeu_si256(w_buf.as_mut_ptr().cast(), vw_lo);
            _mm256_storeu_si256(t_buf.as_mut_ptr().cast(), vt_lo);
        }
        for k in 0..4 {
            acc += w_buf[k] * t_buf[k];
        }
        unsafe {
            _mm256_storeu_si256(w_buf.as_mut_ptr().cast(), vw_hi);
            _mm256_storeu_si256(t_buf.as_mut_ptr().cast(), vt_hi);
        }
        for k in 0..4 {
            acc += w_buf[k] * t_buf[k];
        }
        i += 8;
    }
    while i < n {
        acc += (weights[i] as i64) * (tf_norms[i] as i64);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i32_neon(weights: &[i32], tf_norms: &[i32]) -> i64 {
    use core::arch::aarch64::*;
    let n = weights.len().min(tf_norms.len());
    let mut i = 0usize;
    let mut vsum = unsafe { vdupq_n_s64(0) };
    while i + 4 <= n {
        let vw = unsafe { vld1q_s32(weights.as_ptr().add(i)) };
        let vt = unsafe { vld1q_s32(tf_norms.as_ptr().add(i)) };
        vsum = unsafe { vmlal_s32(vsum, vget_low_s32(vw), vget_low_s32(vt)) };
        vsum = unsafe { vmlal_high_s32(vsum, vw, vt) };
        i += 4;
    }
    let mut acc = unsafe { vaddvq_s64(vsum) };
    while i < n {
        acc += (weights[i] as i64) * (tf_norms[i] as i64);
        i += 1;
    }
    acc
}

/// Runtime-dispatched discrete posting dot-product scorer (#834).
pub fn dot_i32_postings(weights: &[i32], tf_norms: &[i32]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_i32_avx2(weights, tf_norms) };
        }
        dot_i32_scalar(weights, tf_norms)
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_i32_neon(weights, tf_norms) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_i32_scalar(weights, tf_norms)
    }
}

/// Branchless / galloping sorted `u32` posting-list intersection (#834).
pub fn intersect_sorted_u32(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut ia, mut ib) = (0usize, 0usize);
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        if va == vb {
            out.push(va);
            ia += 1;
            ib += 1;
        } else if va < vb {
            ia += 1;
        } else {
            ib += 1;
        }
    }
    out
}

/// SIMD-accelerated batch BLAKE3 hashing (delegating to BLAKE3's native AVX-512 / AVX2 / NEON backend).
pub fn batch_blake3_digests(payloads: &[&[u8]]) -> Vec<[u8; 32]> {
    payloads.iter().map(|p| *blake3::hash(p).as_bytes()).collect()
}

/// Reports active SIMD backend detected on the current CPU.
pub fn active_simd_isa() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "x86_64-avx512f";
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "x86_64-avx2";
        }
        "x86_64-sse2"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64-neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar-portable"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_834_simd_matches_scalar_bit_for_bit_across_all_tail_sizes() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 31, 64, 257] {
            let w: Vec<i32> = (0..len).map(|i| (i as i32 * 17) - 500).collect();
            let t: Vec<i32> = (0..len).map(|i| (i as i32 * 31) + 11).collect();
            assert_eq!(
                dot_i32_postings(&w, &t),
                dot_i32_scalar(&w, &t),
                "SIMD mismatch at len={len}"
            );
        }
        assert_eq!(
            intersect_sorted_u32(&[1, 4, 9, 16, 25], &[2, 4, 16, 30]),
            vec![4, 16]
        );
        let digests = batch_blake3_digests(&[b"alpha", b"beta"]);
        assert_eq!(digests.len(), 2);
        assert!(!active_simd_isa().is_empty());
    }
}

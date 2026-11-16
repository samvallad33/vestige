#![forbid(unsafe_code)]

//! Linear-time $O(N)$ SA-IS suffix array construction over `u16` symbols.
//!
//! Alphabet encoding:
//! - `0`: global EOF sentinel (strictly smallest, appears once at the end)
//! - `1`: document boundary separator (prevents any pattern match from crossing documents)
//! - `2..=257`: raw byte `(b as u16) + 2` (supports all 256 byte values cleanly)

pub const SYM_EOF: u16 = 0;
pub const SYM_DOC_SEP: u16 = 1;
pub const BYTE_OFFSET: u16 = 2;
pub const ALPHABET_SIZE: usize = 258;

/// Encodes raw documents into the unified `u16` symbol stream with `SYM_DOC_SEP`
/// between documents and `SYM_EOF` at the end.
pub fn encode_documents(docs: &[(u32, &[u8])]) -> (Vec<u16>, Vec<(u32, u32, u32)>) {
    let total_raw: usize = docs.iter().map(|(_, bytes)| bytes.len()).sum();
    let mut syms = Vec::with_capacity(total_raw + docs.len() + 1);
    let mut spans = Vec::with_capacity(docs.len());

    for (idx, &(doc_id, bytes)) in docs.iter().enumerate() {
        let start = syms.len() as u32;
        for &b in bytes {
            syms.push(u16::from(b) + BYTE_OFFSET);
        }
        let len = bytes.len() as u32;
        spans.push((doc_id, start, len));
        if idx + 1 < docs.len() {
            syms.push(SYM_DOC_SEP);
        } else {
            syms.push(SYM_EOF);
        }
    }

    (syms, spans)
}

/// Builds the Suffix Array of `syms` (where `syms.last() == Some(&0)` and `0` is unique)
/// in $O(N)$ time using SA-IS.
pub fn build_suffix_array(syms: &[u16]) -> Vec<u32> {
    let input_u32: Vec<u32> = syms.iter().map(|&v| u32::from(v)).collect();
    let sa = sais_u32(&input_u32, ALPHABET_SIZE);
    sa.into_iter().map(|v| v as u32).collect()
}

fn sais_u32(s: &[u32], alphabet_size: usize) -> Vec<usize> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    if n == 2 {
        return if s[1] < s[0] { vec![1, 0] } else { vec![0, 1] };
    }

    // `is_s[i] == true` for S-type suffixes, `false` for L-type suffixes.
    let mut is_s = vec![false; n];
    is_s[n - 1] = true;
    for i in (0..n - 1).rev() {
        is_s[i] = if s[i] < s[i + 1] {
            true
        } else if s[i] > s[i + 1] {
            false
        } else {
            is_s[i + 1]
        };
    }

    let is_lms = |is_s: &[bool], i: usize| -> bool { i > 0 && is_s[i] && !is_s[i - 1] };

    let mut counts = vec![0usize; alphabet_size];
    for &c in s {
        counts[c as usize] += 1;
    }

    let compute_buckets = |end: bool| -> Vec<usize> {
        let mut bkt = vec![0usize; alphabet_size];
        let mut acc = 0usize;
        for (c, &cnt) in counts.iter().enumerate() {
            acc += cnt;
            bkt[c] = if end { acc } else { acc - cnt };
        }
        bkt
    };

    let induce_sort = |sa: &mut [isize]| {
        let mut bkt = compute_buckets(false);
        for i in 0..n {
            let val = sa[i];
            if val > 0 {
                let j = (val - 1) as usize;
                if !is_s[j] {
                    let c = s[j] as usize;
                    sa[bkt[c]] = j as isize;
                    bkt[c] += 1;
                }
            }
        }
        let mut bkt = compute_buckets(true);
        for i in (0..n).rev() {
            let val = sa[i];
            if val > 0 {
                let j = (val - 1) as usize;
                if is_s[j] {
                    let c = s[j] as usize;
                    bkt[c] -= 1;
                    sa[bkt[c]] = j as isize;
                }
            }
        }
    };

    // Stage 1: place LMS suffixes at the ends of their buckets and induce sort.
    let mut sa = vec![-1isize; n];
    let mut bkt = compute_buckets(true);
    for (i, &sym) in s.iter().enumerate().skip(1) {
        if is_lms(&is_s, i) {
            let c = sym as usize;
            bkt[c] -= 1;
            sa[bkt[c]] = i as isize;
        }
    }
    induce_sort(&mut sa);

    // Compact sorted LMS substrings into `sa[0..n1]`.
    let mut n1 = 0usize;
    for i in 0..n {
        let pos = sa[i] as usize;
        if is_lms(&is_s, pos) {
            sa[n1] = pos as isize;
            n1 += 1;
        }
    }

    for slot in &mut sa[n1..n] {
        *slot = -1;
    }

    let mut name = 0usize;
    let mut prev: isize = -1;
    for i in 0..n1 {
        let pos = sa[i] as usize;
        let mut diff = false;
        for d in 0..n {
            if prev < 0 {
                diff = true;
                break;
            }
            let prev_u = prev as usize;
            if s[pos + d] != s[prev_u + d] || is_s[pos + d] != is_s[prev_u + d] {
                diff = true;
                break;
            }
            if d > 0 && (is_lms(&is_s, pos + d) || is_lms(&is_s, prev_u + d)) {
                break;
            }
        }
        if diff {
            name += 1;
            prev = pos as isize;
        }
        sa[n1 + (pos / 2)] = (name - 1) as isize;
    }

    let mut lms_positions = Vec::with_capacity(n1);
    let mut s1 = Vec::with_capacity(n1);
    for i in 1..n {
        if is_lms(&is_s, i) {
            lms_positions.push(i);
            s1.push(sa[n1 + (i / 2)] as u32);
        }
    }

    // Stage 2: recursively solve reduced LMS problem if names are not unique.
    let sa1 = if name < n1 {
        sais_u32(&s1, name)
    } else {
        let mut direct = vec![0usize; n1];
        for (i, &v) in s1.iter().enumerate() {
            direct[v as usize] = i;
        }
        direct
    };

    // Stage 3: place LMS suffixes in exact sorted order and induce final SA.
    sa.fill(-1);
    let mut bkt = compute_buckets(true);
    for &idx in sa1.iter().rev() {
        let j = lms_positions[idx];
        let c = s[j] as usize;
        bkt[c] -= 1;
        sa[bkt[c]] = j as isize;
    }
    induce_sort(&mut sa);

    sa.into_iter().map(|v| v as usize).collect()
}

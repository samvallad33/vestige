use vestige_fmindex::wavelet::{popcount_scalar, popcount_u64};
use vestige_fmindex::{FmError, FmIndex, Hit};

fn naive_locate(docs: &[(u32, &[u8])], pattern: &[u8]) -> Vec<Hit> {
    if pattern.is_empty() {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for &(doc_id, text) in docs {
        if pattern.len() > text.len() {
            continue;
        }
        for (offset, window) in text.windows(pattern.len()).enumerate() {
            if window == pattern {
                hits.push(Hit {
                    byte_offset: offset as u32,
                    doc_id,
                });
            }
        }
    }
    hits.sort();
    hits
}

#[test]
fn test_popcount_scalar_matches_fast_path_for_all_patterns() {
    let probes = [
        0u64,
        1,
        u64::MAX,
        0x5555_5555_5555_5555,
        0xAAAA_AAAA_AAAA_AAAA,
        0xDEAD_BEEF_CAFE_BABE,
        0x0123_4567_89AB_CDEF,
    ];
    for w in probes {
        assert_eq!(popcount_u64(w), popcount_scalar(w));
    }
}

#[test]
fn test_count_locate_and_extract_match_naive_on_adversarial_and_all_256_bytes() {
    let all_bytes: Vec<u8> = (0u8..=255).collect();
    let repeated = vec![b'a'; 128];
    let boundary_left = b"prefix_BOUNDARY_".to_vec();
    let boundary_right = b"CONTINUATION_suffix".to_vec();

    let docs: Vec<(u32, &[u8])> = vec![
        (10, &all_bytes),
        (20, &repeated),
        (30, &boundary_left),
        (40, &boundary_right),
    ];

    let idx = FmIndex::build_with_sample_rate(&docs, 8).expect("build index");

    // Empty pattern -> 0 matches.
    assert_eq!(idx.count(b""), 0);
    assert!(idx.locate(b"", 100).is_empty());

    // Pattern longer than corpus -> 0 matches.
    let too_long = vec![b'a'; 2000];
    assert_eq!(idx.count(&too_long), 0);
    assert!(idx.locate(&too_long, 100).is_empty());

    // Pattern spanning a document boundary must NOT match.
    assert_eq!(idx.count(b"BOUNDARY_CONTINUATION"), 0);
    assert!(idx.locate(b"BOUNDARY_CONTINUATION", 10).is_empty());

    // Every single byte 0..=255 matches naive scan.
    for b in 0u8..=255 {
        let pat = [b];
        let expected = naive_locate(&docs, &pat);
        assert_eq!(idx.count(&pat), expected.len() as u64);
        assert_eq!(idx.locate(&pat, usize::MAX), expected);
    }

    // Repeated substring overlaps.
    for pat in [b"a".as_slice(), b"aa", b"aaa", b"aaaa", &[0x00, 0x01, 0x02], &[0xFE, 0xFF]] {
        let expected = naive_locate(&docs, pat);
        assert_eq!(idx.count(pat), expected.len() as u64);
        assert_eq!(idx.locate(pat, usize::MAX), expected);
    }

    // Exact extract round-trip for every document including binary 0..=255.
    assert_eq!(idx.extract(10, 0, 256).expect("extract doc 10"), all_bytes);
    assert_eq!(idx.extract(20, 17, 45).expect("extract doc 20"), repeated[17..62]);
    assert_eq!(idx.extract(30, 7, 8).expect("extract doc 30"), b"BOUNDARY");
    assert!(matches!(
        idx.extract(999, 0, 1),
        Err(FmError::InvalidDocId(999))
    ));
    assert!(matches!(
        idx.extract(30, 10, 100),
        Err(FmError::OutOfBounds { .. })
    ));
}

#[test]
fn test_random_corpus_and_exact_technical_tokens_uuids_env_paths_stacktraces() {
    let doc0 = b"2026-11-16T02:00:00Z INFO uuid=550e8400-e29b-41d4-a716-446655440000 \
                 RUST_BACKTRACE=1 path=crates/vestige-core/src/chain_fold.rs:142\n\
                 at vestige_core::chain_fold::fold_writer_chains";
    let doc1 = b"VESTIGE_REFLEX_DIR=/var/tmp/reflex-9a8b7c6d \
                 uuid=550e8400-e29b-41d4-a716-446655440000 \
                 panicked at crates/vestige-fmindex/src/wavelet.rs:88:13";

    // Pseudo-random binary + ASCII slice via deterministic xorshift64.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand_doc = Vec::with_capacity(512);
    for _ in 0..512 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rand_doc.push((rng & 0xFF) as u8);
    }

    let docs: Vec<(u32, &[u8])> = vec![(1, doc0), (2, doc1), (3, &rand_doc)];
    let idx = FmIndex::build(&docs).expect("build");

    for token in [
        b"550e8400-e29b-41d4-a716-446655440000".as_slice(),
        b"RUST_BACKTRACE=1",
        b"VESTIGE_REFLEX_DIR=",
        b"crates/vestige-core/src/chain_fold.rs:142",
        b"at vestige_core::chain_fold::fold_writer_chains",
        b"crates/vestige-fmindex/src/wavelet.rs:88:13",
        &rand_doc[120..135],
    ] {
        let expected = naive_locate(&docs, token);
        assert!(!expected.is_empty());
        assert_eq!(idx.count(token), expected.len() as u64);
        assert_eq!(idx.locate(token, usize::MAX), expected);
    }
}

#[test]
fn test_deterministic_digest_and_serialization_roundtrip() {
    let d1 = b"fn fold_writer_chains(frames: &[SignedWriterFrame])";
    let d2 = b"biscuit-auth offline attenuation + iroh-blobs bao";
    let docs: Vec<(u32, &[u8])> = vec![(1, d1), (2, d2)];

    let idx_a = FmIndex::build(&docs).expect("build a");
    let idx_b = FmIndex::build(&docs).expect("build b");
    assert_eq!(idx_a.digest(), idx_b.digest());

    let bytes = idx_a.to_bytes();
    let idx_restored = FmIndex::from_bytes(&bytes).expect("round-trip");
    assert_eq!(idx_a.digest(), idx_restored.digest());
    assert_eq!(
        idx_a.locate(b"SignedWriterFrame", 10),
        idx_restored.locate(b"SignedWriterFrame", 10)
    );
    assert_eq!(
        idx_restored.extract(2, 13, 19).expect("extract restored"),
        b"offline attenuation"
    );

    // Flipping a byte in serialized bytes is caught by digest verification.
    let mut corrupted = bytes;
    corrupted[20] ^= 0x01;
    assert!(matches!(
        FmIndex::from_bytes(&corrupted),
        Err(FmError::CorruptIndex(_))
    ));
}

#[test]
#[ignore = "builds >10MB repetitive log corpus and measures index size vs raw UTF-8"]
fn bench_ten_megabyte_repetitive_log_corpus_smaller_than_raw() {
    let mut docs_owned = Vec::new();
    for shard in 0u32..40 {
        let mut buf = Vec::with_capacity(270_000);
        for line in 0u32..2_500 {
            let entry = format!(
                "2026-11-16T02:40:{:02}Z INFO shard={shard:02} op=chain_fold \
                 writer=did:key:ed25519:vestige_node_{:02} seq={line:04} status=ADMITTED\n",
                line % 60,
                shard % 8
            );
            buf.extend_from_slice(entry.as_bytes());
        }
        docs_owned.push((shard, buf));
    }

    let raw_bytes: usize = docs_owned.iter().map(|(_, b)| b.len()).sum();
    assert!(raw_bytes > 10_000_000, "raw_bytes={raw_bytes} must exceed 10MB");

    let doc_refs: Vec<(u32, &[u8])> = docs_owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    let idx = FmIndex::build_with_sample_rate(&doc_refs, 64).expect("build 10MB+ FmIndex");
    let serialized = idx.to_bytes();

    let count_pat = b"shard=17 op=chain_fold";
    let t_count = std::time::Instant::now();
    let count_hits = idx.count(count_pat);
    let count_latency = t_count.elapsed();

    let t_locate = std::time::Instant::now();
    let located = idx.locate(count_pat, usize::MAX);
    let locate_latency = t_locate.elapsed();

    let t_count_rare = std::time::Instant::now();
    let rare_hits = idx.count(b"seq=1234 status=ADMITTED");
    let rare_count_latency = t_count_rare.elapsed();

    let proc_status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let vm_rss = proc_status
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .unwrap_or("VmRSS: unavailable");
    let vm_hwm = proc_status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .unwrap_or("VmHWM: unavailable");

    eprintln!(
        "10MB+ Repetitive Log Corpus: raw_bytes={} ({:.2} MB), index_bytes={} ({:.2} MB), ratio={:.3}x, rlbwt_runs={}, count_latency={count_latency:?}, locate_latency={locate_latency:?}, locate_hits={}, rare_count_latency={rare_count_latency:?}, {vm_rss}, {vm_hwm}",
        raw_bytes,
        raw_bytes as f64 / 1_048_576.0,
        serialized.len(),
        serialized.len() as f64 / 1_048_576.0,
        serialized.len() as f64 / raw_bytes as f64,
        idx.num_runs(),
        located.len(),
    );

    assert!(
        serialized.len() < raw_bytes,
        "index ({}) must use less memory than raw text ({raw_bytes})",
        serialized.len()
    );
    assert_eq!(count_hits, 2_500);
    assert_eq!(located.len(), 2_500);
    assert_eq!(rare_hits, 40);
}

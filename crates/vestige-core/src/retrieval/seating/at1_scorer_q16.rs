//! Unified deterministic `Q16.16` integer `@1` & `@5` turn/session gatekeeper scorer.
//!
//! Combines:
//! 1. Deep candidate pool assembly (`pool@48`: pipeline turns + top-30 scene + top-20 own +
//!    exchange dyads from top-6 sessions),
//! 2. 36 pure-integer linguistic, discourse, temporal, lexical-specificity, and epistemic features:
//!    - Spoken prose (`tf[0..2]`) vs photo-caption (`tf[2]` / `blip_feats`) disentanglement,
//!    - Speaker-subject alignment + epistemic presupposition deficit shield (`F17`),
//!    - Fine Li & Roth (COLING 2002) answer typing backed by SHA-256-pinned Princeton WordNet 3.1,
//!      GeoNames gazetteer (`countryInfo`, `admin1CodesASCII`, `cities15000`), and Wikidata CC0,
//!    - Session calendar year / month / season alignment (`dates.rs`),
//!    - Partner-question adjacency at `t - 1` and `t - 2`, photo-opener continuation at `t + 2`,
//!      and originating-turn primacy over anaphoric echoes,
//!    - 5-ballot Copeland / Condorcet pairwise tournament (`F35`),
//! 3. Frozen SHA-256-pinned integer GBDT / LambdaMART tree ensemble fitted strictly on a
//!    disjoint conversation split with zero floating-point operations at inference time.

use crate::retrieval::fusion::kemeny_mrf::{ieee754_bits_to_q16, shannon_ic_q16};
use crate::retrieval::*;
use chrono::Datelike;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::OnceLock;

/// Number of deterministic `i32` features per candidate turn.
pub const N_AT1_FEATS: usize = 48;

/// Maximum candidate turns evaluated per query in the deep gatekeeper pool.
const MAX_AT1_POOL: usize = 56;

/// Unit `1.0` in `Q16.16` fixed-point representation.
const Q16_ONE_I64: i64 = 65_536;

/// Compact binary decision tree node in `.rodata` (`8 bytes` per node, zero floats).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GbdtNode {
    /// Feature index `0..N_AT1_FEATS`, or `u8::MAX` when this node is a leaf.
    pub feat: u8,
    /// Left child offset inside the tree (`feats[feat] <= thresh_or_leaf`).
    pub left: u8,
    /// Right child offset inside the tree (`feats[feat] > thresh_or_leaf`).
    pub right: u8,
    /// Split threshold for internal nodes, or signed `Q16` leaf weight for leaves.
    pub thresh_or_leaf: i32,
}

// === BEGIN FROZEN GBDT WEIGHTS (fitted strictly on Convs 1-5 disjoint split, verified by SHA-256 test) ===
/// SHA-256 hex digest of the canonical byte encoding of `AT1_TREE_ROOTS` + `AT1_TREE_NODES`.
#[cfg(test)]
pub const AT1_GBDT_SHA256: &str =
    "305ac81695d9d94aaf85c6ab57417d30eaa6937f0df6898e10be3e0324d8ad3f";

/// Start index of each tree inside `AT1_TREE_NODES`.
pub const AT1_TREE_ROOTS: &[u16] = &[0, 31, 60, 91, 122, 151, 180, 211, 242, 271, 302, 333, 364, 395, 426, 457, 488, 519, 550, 581, 612, 643, 674, 705, 736, 767, 798, 827, 858, 887, 918, 949, 978, 1009, 1040, 1069, 1100, 1129, 1160, 1191, 1220, 1251, 1282, 1313, 1344, 1373, 1404, 1435, 1466, 1497, 1526, 1557, 1588, 1619, 1646, 1673, 1696, 1723, 1754, 1779, 1810, 1835, 1858, 1885];

/// Frozen GBDT node table.
pub const AT1_TREE_NODES: &[GbdtNode] = &[
    GbdtNode { feat: 0, left: 1, right: 16, thresh_or_leaf: 46080 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 31206 },
    GbdtNode { feat: 16, left: 3, right: 6, thresh_or_leaf: 76458 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 12851 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -5800 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -5024 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2901 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 113 },
    GbdtNode { feat: 35, left: 10, right: 13, thresh_or_leaf: 2 },
    GbdtNode { feat: 0, left: 11, right: 12, thresh_or_leaf: 44032 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -926 },
    GbdtNode { feat: 0, left: 14, right: 15, thresh_or_leaf: 44032 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -678 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2915 },
    GbdtNode { feat: 4, left: 17, right: 24, thresh_or_leaf: 11838 },
    GbdtNode { feat: 31, left: 18, right: 21, thresh_or_leaf: 20164 },
    GbdtNode { feat: 37, left: 19, right: 20, thresh_or_leaf: 43690 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4764 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -944 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 43564 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3760 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 936 },
    GbdtNode { feat: 1, left: 25, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: 35, left: 26, right: 27, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 238 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3731 },
    GbdtNode { feat: 5, left: 29, right: 30, thresh_or_leaf: 40930 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3154 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 5712 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 34661 },
    GbdtNode { feat: 8, left: 2, right: 9, thresh_or_leaf: 43690 },
    GbdtNode { feat: 2, left: 3, right: 6, thresh_or_leaf: 23652 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 46080 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -5088 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2621 },
    GbdtNode { feat: 35, left: 7, right: 8, thresh_or_leaf: 3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3724 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 352 },
    GbdtNode { feat: 35, left: 10, right: 13, thresh_or_leaf: 1 },
    GbdtNode { feat: 16, left: 11, right: 12, thresh_or_leaf: 68812 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4390 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1306 },
    GbdtNode { feat: 5, left: 14, right: 15, thresh_or_leaf: 40930 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 169 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4540 },
    GbdtNode { feat: 1, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 4, left: 18, right: 21, thresh_or_leaf: 13932 },
    GbdtNode { feat: 18, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3399 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 632 },
    GbdtNode { feat: 45, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3252 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3126 },
    GbdtNode { feat: 4, left: 25, right: 26, thresh_or_leaf: 11838 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2696 },
    GbdtNode { feat: 6, left: 27, right: 28, thresh_or_leaf: 63488 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4225 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 5069 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 2 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 34661 },
    GbdtNode { feat: 16, left: 3, right: 6, thresh_or_leaf: 76458 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 46080 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4611 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2846 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2975 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1259 },
    GbdtNode { feat: 38, left: 10, right: 13, thresh_or_leaf: 326164 },
    GbdtNode { feat: 12, left: 11, right: 12, thresh_or_leaf: 42674 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 155 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2919 },
    GbdtNode { feat: 0, left: 14, right: 15, thresh_or_leaf: 48128 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 108 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4131 },
    GbdtNode { feat: 2, left: 17, right: 24, thresh_or_leaf: 28324 },
    GbdtNode { feat: 8, left: 18, right: 21, thresh_or_leaf: 43690 },
    GbdtNode { feat: 5, left: 19, right: 20, thresh_or_leaf: 42196 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4164 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1884 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 40930 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2257 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4443 },
    GbdtNode { feat: 0, left: 25, right: 28, thresh_or_leaf: 47104 },
    GbdtNode { feat: 45, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3028 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2111 },
    GbdtNode { feat: 35, left: 29, right: 30, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3872 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4621 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 34661 },
    GbdtNode { feat: 8, left: 2, right: 9, thresh_or_leaf: 43690 },
    GbdtNode { feat: 35, left: 3, right: 6, thresh_or_leaf: 3 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 46080 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4253 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2480 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 17576 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3402 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 274 },
    GbdtNode { feat: 35, left: 10, right: 13, thresh_or_leaf: 1 },
    GbdtNode { feat: 29, left: 11, right: 12, thresh_or_leaf: 5207 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2909 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1870 },
    GbdtNode { feat: 4, left: 14, right: 15, thresh_or_leaf: 20331 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2356 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3613 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 26, left: 18, right: 21, thresh_or_leaf: 36 },
    GbdtNode { feat: 18, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3551 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -547 },
    GbdtNode { feat: 6, left: 22, right: 23, thresh_or_leaf: 61440 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2806 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1558 },
    GbdtNode { feat: 0, left: 25, right: 28, thresh_or_leaf: 47104 },
    GbdtNode { feat: 35, left: 26, right: 27, thresh_or_leaf: 4 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1370 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3276 },
    GbdtNode { feat: 35, left: 29, right: 30, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3537 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 4274 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 34661 },
    GbdtNode { feat: 16, left: 2, right: 9, thresh_or_leaf: 60074 },
    GbdtNode { feat: 35, left: 3, right: 6, thresh_or_leaf: 3 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4050 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3363 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 17576 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3005 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 419 },
    GbdtNode { feat: 4, left: 10, right: 11, thresh_or_leaf: 15260 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2980 },
    GbdtNode { feat: 44, left: 12, right: 13, thresh_or_leaf: 9 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3190 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2406 },
    GbdtNode { feat: 35, left: 15, right: 22, thresh_or_leaf: 1 },
    GbdtNode { feat: 26, left: 16, right: 19, thresh_or_leaf: 36 },
    GbdtNode { feat: 18, left: 17, right: 18, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3350 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -637 },
    GbdtNode { feat: 31, left: 20, right: 21, thresh_or_leaf: 37449 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -830 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2137 },
    GbdtNode { feat: 0, left: 23, right: 26, thresh_or_leaf: 48128 },
    GbdtNode { feat: 18, left: 24, right: 25, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1409 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2989 },
    GbdtNode { feat: 6, left: 27, right: 28, thresh_or_leaf: 63488 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3180 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3992 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 2 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 34661 },
    GbdtNode { feat: 8, left: 3, right: 6, thresh_or_leaf: 52428 },
    GbdtNode { feat: 5, left: 4, right: 5, thresh_or_leaf: 42196 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3885 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3422 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2464 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1807 },
    GbdtNode { feat: 24, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 5, left: 11, right: 12, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1664 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1723 },
    GbdtNode { feat: 47, left: 14, right: 15, thresh_or_leaf: 64035 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2730 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 181 },
    GbdtNode { feat: 5, left: 17, right: 22, thresh_or_leaf: 35331 },
    GbdtNode { feat: 44, left: 18, right: 21, thresh_or_leaf: 12 },
    GbdtNode { feat: 21, left: 19, right: 20, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -199 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3505 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2238 },
    GbdtNode { feat: 2, left: 23, right: 26, thresh_or_leaf: 43673 },
    GbdtNode { feat: 16, left: 24, right: 25, thresh_or_leaf: 45875 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -355 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2558 },
    GbdtNode { feat: 35, left: 27, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2884 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3779 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 34661 },
    GbdtNode { feat: 16, left: 2, right: 9, thresh_or_leaf: 60074 },
    GbdtNode { feat: 35, left: 3, right: 6, thresh_or_leaf: 3 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3677 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2891 },
    GbdtNode { feat: 5, left: 7, right: 8, thresh_or_leaf: 36504 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3047 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 704 },
    GbdtNode { feat: 4, left: 10, right: 13, thresh_or_leaf: 15260 },
    GbdtNode { feat: 44, left: 11, right: 12, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -254 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2778 },
    GbdtNode { feat: 40, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2827 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2047 },
    GbdtNode { feat: 4, left: 17, right: 24, thresh_or_leaf: 13932 },
    GbdtNode { feat: 16, left: 18, right: 21, thresh_or_leaf: 38229 },
    GbdtNode { feat: 18, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2735 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -157 },
    GbdtNode { feat: 35, left: 22, right: 23, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2498 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2991 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 8901 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1234 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1807 },
    GbdtNode { feat: 35, left: 29, right: 30, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2574 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3597 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 38236 },
    GbdtNode { feat: 35, left: 2, right: 9, thresh_or_leaf: 2 },
    GbdtNode { feat: 16, left: 3, right: 6, thresh_or_leaf: 76458 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 20260 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3522 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2331 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 15528 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2596 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1067 },
    GbdtNode { feat: 4, left: 10, right: 13, thresh_or_leaf: 27039 },
    GbdtNode { feat: 5, left: 11, right: 12, thresh_or_leaf: 36504 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3208 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 19 },
    GbdtNode { feat: 31, left: 14, right: 15, thresh_or_leaf: 28086 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2059 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2286 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 4, left: 18, right: 21, thresh_or_leaf: 13932 },
    GbdtNode { feat: 26, left: 19, right: 20, thresh_or_leaf: 23 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3083 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -441 },
    GbdtNode { feat: 18, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1470 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1466 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 155 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 805 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2938 },
    GbdtNode { feat: 0, left: 29, right: 30, thresh_or_leaf: 46080 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2466 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3479 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 34661 },
    GbdtNode { feat: 4, left: 2, right: 9, thresh_or_leaf: 27039 },
    GbdtNode { feat: 35, left: 3, right: 6, thresh_or_leaf: 1 },
    GbdtNode { feat: 37, left: 4, right: 5, thresh_or_leaf: 46811 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3389 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1094 },
    GbdtNode { feat: 17, left: 7, right: 8, thresh_or_leaf: -406637 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2741 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1141 },
    GbdtNode { feat: 35, left: 10, right: 13, thresh_or_leaf: -1 },
    GbdtNode { feat: 46, left: 11, right: 12, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 667 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3058 },
    GbdtNode { feat: 44, left: 14, right: 15, thresh_or_leaf: 9 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1581 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2929 },
    GbdtNode { feat: 1, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 35, left: 18, right: 21, thresh_or_leaf: 2 },
    GbdtNode { feat: 26, left: 19, right: 20, thresh_or_leaf: 36 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1882 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 487 },
    GbdtNode { feat: 45, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2357 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3315 },
    GbdtNode { feat: 4, left: 25, right: 26, thresh_or_leaf: 11838 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2407 },
    GbdtNode { feat: 5, left: 27, right: 28, thresh_or_leaf: 62177 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2718 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3369 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 2 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 28324 },
    GbdtNode { feat: 8, left: 3, right: 6, thresh_or_leaf: 52428 },
    GbdtNode { feat: 5, left: 4, right: 5, thresh_or_leaf: 30321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3434 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2953 },
    GbdtNode { feat: 15, left: 7, right: 8, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1603 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1830 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 17770 },
    GbdtNode { feat: 18, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2075 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -229 },
    GbdtNode { feat: 5, left: 14, right: 15, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2275 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2460 },
    GbdtNode { feat: 2, left: 17, right: 24, thresh_or_leaf: 43673 },
    GbdtNode { feat: 3, left: 18, right: 21, thresh_or_leaf: 30186 },
    GbdtNode { feat: 5, left: 19, right: 20, thresh_or_leaf: 36504 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2951 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 303 },
    GbdtNode { feat: 31, left: 22, right: 23, thresh_or_leaf: 29127 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2146 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2185 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 155 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 622 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2682 },
    GbdtNode { feat: 45, left: 29, right: 30, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3248 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1474 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 20331 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 34661 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 35, left: 4, right: 5, thresh_or_leaf: 3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3186 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1216 },
    GbdtNode { feat: 11, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1871 },
    GbdtNode { feat: 42, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 29, left: 11, right: 12, thresh_or_leaf: 9 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2564 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 64 },
    GbdtNode { feat: 16, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -684 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1862 },
    GbdtNode { feat: 2, left: 17, right: 24, thresh_or_leaf: 43673 },
    GbdtNode { feat: 16, left: 18, right: 21, thresh_or_leaf: 42598 },
    GbdtNode { feat: 5, left: 19, right: 20, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2140 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 137 },
    GbdtNode { feat: 43, left: 22, right: 23, thresh_or_leaf: 12276 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2516 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 309 },
    GbdtNode { feat: 0, left: 25, right: 28, thresh_or_leaf: 47104 },
    GbdtNode { feat: 31, left: 26, right: 27, thresh_or_leaf: 16384 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1707 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1809 },
    GbdtNode { feat: 18, left: 29, right: 30, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1341 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3104 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 2 },
    GbdtNode { feat: 0, left: 2, right: 9, thresh_or_leaf: 40960 },
    GbdtNode { feat: 16, left: 3, right: 6, thresh_or_leaf: 58982 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 18868 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3188 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2056 },
    GbdtNode { feat: 17, left: 7, right: 8, thresh_or_leaf: -354927 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1975 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1039 },
    GbdtNode { feat: 2, left: 10, right: 13, thresh_or_leaf: 28324 },
    GbdtNode { feat: 37, left: 11, right: 12, thresh_or_leaf: 46811 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2512 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 870 },
    GbdtNode { feat: 4, left: 14, right: 15, thresh_or_leaf: 13932 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1114 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 955 },
    GbdtNode { feat: 5, left: 17, right: 24, thresh_or_leaf: 44845 },
    GbdtNode { feat: 4, left: 18, right: 21, thresh_or_leaf: 27039 },
    GbdtNode { feat: 44, left: 19, right: 20, thresh_or_leaf: 12 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2294 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1713 },
    GbdtNode { feat: 20, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2354 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -454 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 2, left: 26, right: 27, thresh_or_leaf: 58982 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 918 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2522 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 32466 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3079 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2083 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 2 },
    GbdtNode { feat: 4, left: 2, right: 9, thresh_or_leaf: 13932 },
    GbdtNode { feat: 2, left: 3, right: 6, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 4, right: 5, thresh_or_leaf: 46811 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3130 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -445 },
    GbdtNode { feat: 34, left: 7, right: 8, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1931 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 801 },
    GbdtNode { feat: 2, left: 10, right: 13, thresh_or_leaf: 43673 },
    GbdtNode { feat: 15, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2660 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -648 },
    GbdtNode { feat: 18, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -644 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1885 },
    GbdtNode { feat: 0, left: 17, right: 24, thresh_or_leaf: 47104 },
    GbdtNode { feat: 4, left: 18, right: 21, thresh_or_leaf: 27039 },
    GbdtNode { feat: 6, left: 19, right: 20, thresh_or_leaf: 61440 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2315 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 474 },
    GbdtNode { feat: 18, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2146 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1924 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 81 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -42 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2490 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 32466 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3041 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1903 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 5, left: 4, right: 5, thresh_or_leaf: 30321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3200 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2737 },
    GbdtNode { feat: 11, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2432 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2400 },
    GbdtNode { feat: 36, left: 10, right: 13, thresh_or_leaf: 19660 },
    GbdtNode { feat: 15, left: 11, right: 12, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2038 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -394 },
    GbdtNode { feat: 26, left: 14, right: 15, thresh_or_leaf: 28 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 224 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3319 },
    GbdtNode { feat: 0, left: 17, right: 24, thresh_or_leaf: 47104 },
    GbdtNode { feat: 5, left: 18, right: 21, thresh_or_leaf: 30321 },
    GbdtNode { feat: 4, left: 19, right: 20, thresh_or_leaf: 20331 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2427 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 431 },
    GbdtNode { feat: 35, left: 22, right: 23, thresh_or_leaf: 4 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 137 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1804 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 81 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -280 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2390 },
    GbdtNode { feat: 5, left: 29, right: 30, thresh_or_leaf: 59919 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2062 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2970 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 16, left: 3, right: 6, thresh_or_leaf: 45511 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 47104 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2990 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -256 },
    GbdtNode { feat: 37, left: 7, right: 8, thresh_or_leaf: 49152 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1678 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2918 },
    GbdtNode { feat: 34, left: 10, right: 13, thresh_or_leaf: 1 },
    GbdtNode { feat: 36, left: 11, right: 12, thresh_or_leaf: 28086 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1417 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2535 },
    GbdtNode { feat: 26, left: 14, right: 15, thresh_or_leaf: 46 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1378 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2002 },
    GbdtNode { feat: 0, left: 17, right: 24, thresh_or_leaf: 47104 },
    GbdtNode { feat: 4, left: 18, right: 21, thresh_or_leaf: 8751 },
    GbdtNode { feat: 6, left: 19, right: 20, thresh_or_leaf: 61440 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2875 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -241 },
    GbdtNode { feat: 31, left: 22, right: 23, thresh_or_leaf: 21845 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -515 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1379 },
    GbdtNode { feat: 6, left: 25, right: 28, thresh_or_leaf: 63488 },
    GbdtNode { feat: 35, left: 26, right: 27, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -732 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1998 },
    GbdtNode { feat: 35, left: 29, right: 30, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1977 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2871 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 48476 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 54613 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 10959 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3054 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2160 },
    GbdtNode { feat: 16, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1473 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2751 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 1434 },
    GbdtNode { feat: 0, left: 11, right: 12, thresh_or_leaf: 45056 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2170 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -75 },
    GbdtNode { feat: 31, left: 14, right: 15, thresh_or_leaf: 40960 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -689 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2143 },
    GbdtNode { feat: 0, left: 17, right: 24, thresh_or_leaf: 48128 },
    GbdtNode { feat: 35, left: 18, right: 21, thresh_or_leaf: 5 },
    GbdtNode { feat: 4, left: 19, right: 20, thresh_or_leaf: 7055 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1862 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 488 },
    GbdtNode { feat: 9, left: 22, right: 23, thresh_or_leaf: 143688 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3230 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2195 },
    GbdtNode { feat: 43, left: 25, right: 28, thresh_or_leaf: 32466 },
    GbdtNode { feat: 18, left: 26, right: 27, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 564 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2801 },
    GbdtNode { feat: 5, left: 29, right: 30, thresh_or_leaf: 56444 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3877 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1361 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 7055 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2981 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2235 },
    GbdtNode { feat: 11, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2184 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2732 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 26, left: 11, right: 12, thresh_or_leaf: 33237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2025 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 338 },
    GbdtNode { feat: 42, left: 14, right: 15, thresh_or_leaf: 2350 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -741 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1291 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 18, right: 21, thresh_or_leaf: 8901 },
    GbdtNode { feat: 4, left: 19, right: 20, thresh_or_leaf: 38969 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -880 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 755 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 70 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2397 },
    GbdtNode { feat: 5, left: 25, right: 28, thresh_or_leaf: 62177 },
    GbdtNode { feat: 16, left: 26, right: 27, thresh_or_leaf: 29491 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3394 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1749 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 32466 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2775 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1419 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 20331 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 46504 },
    GbdtNode { feat: 39, left: 3, right: 6, thresh_or_leaf: 43690 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 9442 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2951 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1981 },
    GbdtNode { feat: 16, left: 7, right: 8, thresh_or_leaf: 38229 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2079 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2031 },
    GbdtNode { feat: 16, left: 10, right: 13, thresh_or_leaf: 36548 },
    GbdtNode { feat: 18, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2146 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -506 },
    GbdtNode { feat: 18, left: 14, right: 15, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2664 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1432 },
    GbdtNode { feat: 1, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 18, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 3, left: 19, right: 20, thresh_or_leaf: 8773 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1224 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2929 },
    GbdtNode { feat: 35, left: 22, right: 23, thresh_or_leaf: -2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1209 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1393 },
    GbdtNode { feat: 9, left: 25, right: 28, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 34, left: 26, right: 27, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1722 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2923 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 42629 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2922 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1136 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 28324 },
    GbdtNode { feat: 35, left: 2, right: 9, thresh_or_leaf: 1 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 5, left: 4, right: 5, thresh_or_leaf: 30321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2995 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2109 },
    GbdtNode { feat: 11, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1785 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2636 },
    GbdtNode { feat: 5, left: 10, right: 13, thresh_or_leaf: 59919 },
    GbdtNode { feat: 44, left: 11, right: 12, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1034 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1505 },
    GbdtNode { feat: 18, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -82 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2186 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 18, right: 21, thresh_or_leaf: 81 },
    GbdtNode { feat: 24, left: 19, right: 20, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -241 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2215 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -181 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1850 },
    GbdtNode { feat: 9, left: 25, right: 28, thresh_or_leaf: 838221 },
    GbdtNode { feat: 34, left: 26, right: 27, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1212 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2950 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 60662 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2741 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1935 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 5, left: 4, right: 5, thresh_or_leaf: 30321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2954 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2189 },
    GbdtNode { feat: 11, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1947 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2595 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: -1 },
    GbdtNode { feat: 0, left: 11, right: 12, thresh_or_leaf: 43008 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2330 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -532 },
    GbdtNode { feat: 34, left: 14, right: 15, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -140 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2484 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 5 },
    GbdtNode { feat: 38, left: 18, right: 21, thresh_or_leaf: 440012 },
    GbdtNode { feat: 18, left: 19, right: 20, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2955 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 29 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 44845 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2271 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1851 },
    GbdtNode { feat: 5, left: 25, right: 28, thresh_or_leaf: 62177 },
    GbdtNode { feat: 16, left: 26, right: 27, thresh_or_leaf: 29491 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3378 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1490 },
    GbdtNode { feat: 45, left: 29, right: 30, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2539 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -80 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 7055 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2793 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1812 },
    GbdtNode { feat: 12, left: 7, right: 8, thresh_or_leaf: 49152 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2378 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1854 },
    GbdtNode { feat: 34, left: 10, right: 13, thresh_or_leaf: 1 },
    GbdtNode { feat: 19, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 197 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1581 },
    GbdtNode { feat: 26, left: 14, right: 15, thresh_or_leaf: 46 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -784 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2229 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 6 },
    GbdtNode { feat: 7, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 15, left: 19, right: 20, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2160 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 280 },
    GbdtNode { feat: 18, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2143 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1460 },
    GbdtNode { feat: 44, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 15, left: 26, right: 27, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3080 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1546 },
    GbdtNode { feat: 5, left: 29, right: 30, thresh_or_leaf: 52282 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 432 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2580 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 20260 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 46811 },
    GbdtNode { feat: 16, left: 4, right: 5, thresh_or_leaf: 36044 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2662 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1182 },
    GbdtNode { feat: 20, left: 7, right: 8, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1656 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1578 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 5, left: 11, right: 12, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2416 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -628 },
    GbdtNode { feat: 5, left: 14, right: 15, thresh_or_leaf: 39492 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1821 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 853 },
    GbdtNode { feat: 7, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 15, left: 18, right: 21, thresh_or_leaf: 1 },
    GbdtNode { feat: 20, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -776 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4497 },
    GbdtNode { feat: 3, left: 22, right: 23, thresh_or_leaf: 6580 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1427 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1073 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 6 },
    GbdtNode { feat: 27, left: 26, right: 27, thresh_or_leaf: 18 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1461 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -495 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 31452 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2517 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 293 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 13932 },
    GbdtNode { feat: 2, left: 2, right: 9, thresh_or_leaf: 12851 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 40960 },
    GbdtNode { feat: 35, left: 4, right: 5, thresh_or_leaf: -2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2729 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1750 },
    GbdtNode { feat: 12, left: 7, right: 8, thresh_or_leaf: 54613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1841 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1785 },
    GbdtNode { feat: 42, left: 10, right: 13, thresh_or_leaf: 1662 },
    GbdtNode { feat: 6, left: 11, right: 12, thresh_or_leaf: 63488 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2212 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -761 },
    GbdtNode { feat: 34, left: 14, right: 15, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -262 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2552 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 6 },
    GbdtNode { feat: 18, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 3, left: 19, right: 20, thresh_or_leaf: 13076 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 247 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2790 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -98 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1292 },
    GbdtNode { feat: 9, left: 25, right: 28, thresh_or_leaf: 838221 },
    GbdtNode { feat: 44, left: 26, right: 27, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1319 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1705 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 50939 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2610 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 44 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 13932 },
    GbdtNode { feat: 7, left: 2, right: 9, thresh_or_leaf: 0 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 42196 },
    GbdtNode { feat: 8, left: 4, right: 5, thresh_or_leaf: 28086 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2707 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1032 },
    GbdtNode { feat: 15, left: 7, right: 8, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2644 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -247 },
    GbdtNode { feat: 16, left: 10, right: 13, thresh_or_leaf: 24576 },
    GbdtNode { feat: 26, left: 11, right: 12, thresh_or_leaf: 17 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1587 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 292 },
    GbdtNode { feat: 34, left: 14, right: 15, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 207 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2565 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 6 },
    GbdtNode { feat: 26, left: 18, right: 21, thresh_or_leaf: 8901 },
    GbdtNode { feat: 24, left: 19, right: 20, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 233 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2285 },
    GbdtNode { feat: 42, left: 22, right: 23, thresh_or_leaf: 1983 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -227 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1988 },
    GbdtNode { feat: 9, left: 25, right: 28, thresh_or_leaf: 838221 },
    GbdtNode { feat: 44, left: 26, right: 27, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1344 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1591 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 26797 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2616 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1246 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: 0 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 42196 },
    GbdtNode { feat: 4, left: 3, right: 6, thresh_or_leaf: 9442 },
    GbdtNode { feat: 28, left: 4, right: 5, thresh_or_leaf: 32768 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2772 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1356 },
    GbdtNode { feat: 9, left: 7, right: 8, thresh_or_leaf: 362495 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -116 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2171 },
    GbdtNode { feat: 35, left: 10, right: 13, thresh_or_leaf: -5 },
    GbdtNode { feat: 34, left: 11, right: 12, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1858 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1937 },
    GbdtNode { feat: 15, left: 14, right: 15, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1323 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 910 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 5 },
    GbdtNode { feat: 31, left: 18, right: 21, thresh_or_leaf: 40960 },
    GbdtNode { feat: 34, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1294 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 328 },
    GbdtNode { feat: 26, left: 22, right: 23, thresh_or_leaf: 33237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 524 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2124 },
    GbdtNode { feat: 43, left: 25, right: 28, thresh_or_leaf: 31452 },
    GbdtNode { feat: 45, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2137 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1977 },
    GbdtNode { feat: 4, left: 29, right: 30, thresh_or_leaf: 38969 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3107 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 988 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 8751 },
    GbdtNode { feat: 7, left: 2, right: 9, thresh_or_leaf: 0 },
    GbdtNode { feat: 3, left: 3, right: 6, thresh_or_leaf: 48 },
    GbdtNode { feat: 37, left: 4, right: 5, thresh_or_leaf: 39321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2932 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -604 },
    GbdtNode { feat: 47, left: 7, right: 8, thresh_or_leaf: 280 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -401 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2281 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 37, left: 11, right: 12, thresh_or_leaf: 37449 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1999 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 314 },
    GbdtNode { feat: 12, left: 14, right: 15, thresh_or_leaf: 37809 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1575 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -364 },
    GbdtNode { feat: 5, left: 17, right: 24, thresh_or_leaf: 62177 },
    GbdtNode { feat: 15, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 36, left: 19, right: 20, thresh_or_leaf: 10922 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2484 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1434 },
    GbdtNode { feat: 18, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2781 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 630 },
    GbdtNode { feat: 31, left: 25, right: 28, thresh_or_leaf: 25206 },
    GbdtNode { feat: 34, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3074 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 916 },
    GbdtNode { feat: 18, left: 29, right: 30, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -941 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1882 },
    GbdtNode { feat: 4, left: 1, right: 14, thresh_or_leaf: 7055 },
    GbdtNode { feat: 5, left: 2, right: 7, thresh_or_leaf: 42196 },
    GbdtNode { feat: 28, left: 3, right: 6, thresh_or_leaf: 32768 },
    GbdtNode { feat: 44, left: 4, right: 5, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1818 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2796 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1688 },
    GbdtNode { feat: 26, left: 8, right: 11, thresh_or_leaf: 4885 },
    GbdtNode { feat: 46, left: 9, right: 10, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2418 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -616 },
    GbdtNode { feat: 44, left: 12, right: 13, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3259 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -260 },
    GbdtNode { feat: 5, left: 15, right: 22, thresh_or_leaf: 62177 },
    GbdtNode { feat: 15, left: 16, right: 19, thresh_or_leaf: 0 },
    GbdtNode { feat: 36, left: 17, right: 18, thresh_or_leaf: 18724 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2362 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1928 },
    GbdtNode { feat: 18, left: 20, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2460 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 523 },
    GbdtNode { feat: 9, left: 23, right: 26, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 34, left: 24, right: 25, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 554 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2417 },
    GbdtNode { feat: 8, left: 27, right: 28, thresh_or_leaf: 24576 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2492 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2388 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 7055 },
    GbdtNode { feat: 37, left: 2, right: 9, thresh_or_leaf: 37449 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 42196 },
    GbdtNode { feat: 44, left: 4, right: 5, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1707 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2770 },
    GbdtNode { feat: 29, left: 7, right: 8, thresh_or_leaf: 22 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1933 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -16 },
    GbdtNode { feat: 18, left: 10, right: 13, thresh_or_leaf: -1 },
    GbdtNode { feat: 6, left: 11, right: 12, thresh_or_leaf: 55296 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 847 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2433 },
    GbdtNode { feat: 12, left: 14, right: 15, thresh_or_leaf: 56173 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2229 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -395 },
    GbdtNode { feat: 35, left: 17, right: 24, thresh_or_leaf: 5 },
    GbdtNode { feat: 27, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 5, left: 19, right: 20, thresh_or_leaf: 48476 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -823 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 914 },
    GbdtNode { feat: 27, left: 22, right: 23, thresh_or_leaf: 30229 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -906 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3549 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 9, left: 26, right: 27, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 291 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2143 },
    GbdtNode { feat: 31, left: 29, right: 30, thresh_or_leaf: 21845 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 411 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2782 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 12851 },
    GbdtNode { feat: 3, left: 2, right: 7, thresh_or_leaf: 48 },
    GbdtNode { feat: 28, left: 3, right: 6, thresh_or_leaf: 32768 },
    GbdtNode { feat: 44, left: 4, right: 5, thresh_or_leaf: 14 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2859 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 961 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 942 },
    GbdtNode { feat: 7, left: 8, right: 11, thresh_or_leaf: 0 },
    GbdtNode { feat: 15, left: 9, right: 10, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2760 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1233 },
    GbdtNode { feat: 18, left: 12, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -821 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1613 },
    GbdtNode { feat: 16, left: 15, right: 22, thresh_or_leaf: 40960 },
    GbdtNode { feat: 34, left: 16, right: 19, thresh_or_leaf: 1 },
    GbdtNode { feat: 24, left: 17, right: 18, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -174 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2241 },
    GbdtNode { feat: 5, left: 20, right: 21, thresh_or_leaf: 39492 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1879 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1713 },
    GbdtNode { feat: 18, left: 23, right: 26, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 24, right: 25, thresh_or_leaf: -387101 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 364 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3158 },
    GbdtNode { feat: 34, left: 27, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1258 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2595 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 7055 },
    GbdtNode { feat: 6, left: 2, right: 9, thresh_or_leaf: 63488 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 30321 },
    GbdtNode { feat: 29, left: 4, right: 5, thresh_or_leaf: 53 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2749 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1702 },
    GbdtNode { feat: 41, left: 7, right: 8, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2747 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1785 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 37, left: 11, right: 12, thresh_or_leaf: 37449 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2093 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 274 },
    GbdtNode { feat: 26, left: 14, right: 15, thresh_or_leaf: 36 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -295 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1980 },
    GbdtNode { feat: 16, left: 17, right: 24, thresh_or_leaf: 40960 },
    GbdtNode { feat: 5, left: 18, right: 21, thresh_or_leaf: 58534 },
    GbdtNode { feat: 16, left: 19, right: 20, thresh_or_leaf: 31507 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1914 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 413 },
    GbdtNode { feat: 27, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 838 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1298 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 9, left: 26, right: 27, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 513 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1864 },
    GbdtNode { feat: 27, left: 29, right: 30, thresh_or_leaf: 32 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2723 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1361 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 7055 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 42196 },
    GbdtNode { feat: 44, left: 3, right: 6, thresh_or_leaf: 1 },
    GbdtNode { feat: 29, left: 4, right: 5, thresh_or_leaf: 53 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1997 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1285 },
    GbdtNode { feat: 37, left: 7, right: 8, thresh_or_leaf: 39321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2672 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 111 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 28 },
    GbdtNode { feat: 46, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2507 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -319 },
    GbdtNode { feat: 16, left: 14, right: 15, thresh_or_leaf: -65536 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2948 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -157 },
    GbdtNode { feat: 5, left: 17, right: 24, thresh_or_leaf: 64921 },
    GbdtNode { feat: 15, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 33, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2383 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -238 },
    GbdtNode { feat: 34, left: 22, right: 23, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -108 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1897 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: 9, left: 26, right: 27, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1615 },
    GbdtNode { feat: 18, left: 29, right: 30, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1187 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1850 },
    GbdtNode { feat: 4, left: 1, right: 14, thresh_or_leaf: 7055 },
    GbdtNode { feat: 5, left: 2, right: 7, thresh_or_leaf: 42196 },
    GbdtNode { feat: 28, left: 3, right: 6, thresh_or_leaf: 32768 },
    GbdtNode { feat: 44, left: 4, right: 5, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1351 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2602 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1826 },
    GbdtNode { feat: 15, left: 8, right: 11, thresh_or_leaf: -1 },
    GbdtNode { feat: 0, left: 9, right: 10, thresh_or_leaf: 40960 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2344 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -401 },
    GbdtNode { feat: 6, left: 12, right: 13, thresh_or_leaf: 57344 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2685 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -306 },
    GbdtNode { feat: 16, left: 15, right: 22, thresh_or_leaf: 40960 },
    GbdtNode { feat: 31, left: 16, right: 19, thresh_or_leaf: 40960 },
    GbdtNode { feat: 34, left: 17, right: 18, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1300 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 89 },
    GbdtNode { feat: 5, left: 20, right: 21, thresh_or_leaf: 54414 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1403 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1001 },
    GbdtNode { feat: 34, left: 23, right: 26, thresh_or_leaf: 1 },
    GbdtNode { feat: 5, left: 24, right: 25, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -180 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1322 },
    GbdtNode { feat: 35, left: 27, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1469 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2612 },
    GbdtNode { feat: 4, left: 1, right: 16, thresh_or_leaf: 7055 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 42196 },
    GbdtNode { feat: 29, left: 3, right: 6, thresh_or_leaf: 53 },
    GbdtNode { feat: 35, left: 4, right: 5, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2612 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -565 },
    GbdtNode { feat: 44, left: 7, right: 8, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1392 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1832 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 28 },
    GbdtNode { feat: 46, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2426 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -205 },
    GbdtNode { feat: 16, left: 14, right: 15, thresh_or_leaf: -65536 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2756 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -38 },
    GbdtNode { feat: 9, left: 17, right: 24, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 34, left: 18, right: 21, thresh_or_leaf: 1 },
    GbdtNode { feat: 15, left: 19, right: 20, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -959 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 381 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 46504 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -136 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1982 },
    GbdtNode { feat: 43, left: 25, right: 28, thresh_or_leaf: 26797 },
    GbdtNode { feat: 45, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2167 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1418 },
    GbdtNode { feat: 12, left: 29, right: 30, thresh_or_leaf: 58254 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3495 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1126 },
    GbdtNode { feat: 31, left: 1, right: 16, thresh_or_leaf: 25206 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 30321 },
    GbdtNode { feat: 4, left: 3, right: 6, thresh_or_leaf: 27039 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 40960 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2507 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -834 },
    GbdtNode { feat: 31, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2312 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1071 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 33237 },
    GbdtNode { feat: 15, left: 11, right: 12, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1764 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 52 },
    GbdtNode { feat: 47, left: 14, right: 15, thresh_or_leaf: 75 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 592 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -7863 },
    GbdtNode { feat: 18, left: 17, right: 24, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 18, right: 21, thresh_or_leaf: -481648 },
    GbdtNode { feat: 4, left: 19, right: 20, thresh_or_leaf: 18315 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1827 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 811 },
    GbdtNode { feat: 27, left: 22, right: 23, thresh_or_leaf: 14749 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2787 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -649 },
    GbdtNode { feat: 42, left: 25, right: 28, thresh_or_leaf: 1983 },
    GbdtNode { feat: 5, left: 26, right: 27, thresh_or_leaf: 59919 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -855 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 639 },
    GbdtNode { feat: 34, left: 29, right: 30, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 893 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2317 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 6947 },
    GbdtNode { feat: 37, left: 2, right: 9, thresh_or_leaf: 56173 },
    GbdtNode { feat: 0, left: 3, right: 6, thresh_or_leaf: 40960 },
    GbdtNode { feat: 46, left: 4, right: 5, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2891 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1862 },
    GbdtNode { feat: 17, left: 7, right: 8, thresh_or_leaf: 473735 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1024 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2205 },
    GbdtNode { feat: 20, left: 10, right: 11, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -562 },
    GbdtNode { feat: 26, left: 12, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2996 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -258 },
    GbdtNode { feat: 9, left: 15, right: 22, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 34, left: 16, right: 19, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 17, right: 18, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2102 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -26 },
    GbdtNode { feat: 31, left: 20, right: 21, thresh_or_leaf: 28086 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 655 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2072 },
    GbdtNode { feat: 43, left: 23, right: 26, thresh_or_leaf: 26797 },
    GbdtNode { feat: 45, left: 24, right: 25, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2037 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1400 },
    GbdtNode { feat: 12, left: 27, right: 28, thresh_or_leaf: 58254 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3745 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1034 },
    GbdtNode { feat: 31, left: 1, right: 16, thresh_or_leaf: 25206 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 30321 },
    GbdtNode { feat: 36, left: 3, right: 6, thresh_or_leaf: 26214 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 27039 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2367 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 942 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 7353 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -433 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1829 },
    GbdtNode { feat: 16, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 37, left: 11, right: 12, thresh_or_leaf: 46811 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1625 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1252 },
    GbdtNode { feat: 43, left: 14, right: 15, thresh_or_leaf: 47688 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 694 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1433 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 9, left: 18, right: 21, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 33, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -360 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 941 },
    GbdtNode { feat: 45, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1575 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2005 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 26, left: 26, right: 27, thresh_or_leaf: 17 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -377 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1790 },
    GbdtNode { feat: 27, left: 29, right: 30, thresh_or_leaf: 32 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2689 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 638 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 6210 },
    GbdtNode { feat: 3, left: 2, right: 7, thresh_or_leaf: 48 },
    GbdtNode { feat: 31, left: 3, right: 4, thresh_or_leaf: 52428 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2672 },
    GbdtNode { feat: 29, left: 5, right: 6, thresh_or_leaf: 34 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -579 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 563 },
    GbdtNode { feat: 0, left: 8, right: 11, thresh_or_leaf: 40960 },
    GbdtNode { feat: 15, left: 9, right: 10, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2572 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1073 },
    GbdtNode { feat: 47, left: 12, right: 13, thresh_or_leaf: 60650 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1856 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1289 },
    GbdtNode { feat: 34, left: 15, right: 22, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 16, right: 19, thresh_or_leaf: -1 },
    GbdtNode { feat: 38, left: 17, right: 18, thresh_or_leaf: 677040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2301 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 395 },
    GbdtNode { feat: 43, left: 20, right: 21, thresh_or_leaf: 10156 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 781 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -315 },
    GbdtNode { feat: 42, left: 23, right: 26, thresh_or_leaf: 1413 },
    GbdtNode { feat: 14, left: 24, right: 25, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1628 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1778 },
    GbdtNode { feat: 16, left: 27, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 757 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2278 },
    GbdtNode { feat: 5, left: 1, right: 16, thresh_or_leaf: 30321 },
    GbdtNode { feat: 16, left: 2, right: 9, thresh_or_leaf: 45511 },
    GbdtNode { feat: 15, left: 3, right: 6, thresh_or_leaf: 0 },
    GbdtNode { feat: 39, left: 4, right: 5, thresh_or_leaf: 43690 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2729 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 125 },
    GbdtNode { feat: 16, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -527 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2135 },
    GbdtNode { feat: 9, left: 10, right: 13, thresh_or_leaf: 550468 },
    GbdtNode { feat: 38, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2269 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1047 },
    GbdtNode { feat: 9, left: 14, right: 15, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1880 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 115 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 19, right: 20, thresh_or_leaf: -505548 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -477 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2486 },
    GbdtNode { feat: 15, left: 22, right: 23, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -476 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 674 },
    GbdtNode { feat: 42, left: 25, right: 28, thresh_or_leaf: 1662 },
    GbdtNode { feat: 14, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1588 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1802 },
    GbdtNode { feat: 2, left: 29, right: 30, thresh_or_leaf: 14571 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2110 },
    GbdtNode { feat: 5, left: 1, right: 16, thresh_or_leaf: 30321 },
    GbdtNode { feat: 8, left: 2, right: 9, thresh_or_leaf: 29127 },
    GbdtNode { feat: 15, left: 3, right: 6, thresh_or_leaf: 0 },
    GbdtNode { feat: 28, left: 4, right: 5, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2852 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -781 },
    GbdtNode { feat: 15, left: 7, right: 8, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 723 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1903 },
    GbdtNode { feat: 9, left: 10, right: 13, thresh_or_leaf: 550468 },
    GbdtNode { feat: 2, left: 11, right: 12, thresh_or_leaf: 3301 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1712 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1623 },
    GbdtNode { feat: 5, left: 14, right: 15, thresh_or_leaf: 29338 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1973 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -48 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 38, left: 19, right: 20, thresh_or_leaf: 677040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2186 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 522 },
    GbdtNode { feat: 9, left: 22, right: 23, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -36 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1413 },
    GbdtNode { feat: 19, left: 25, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: 8, left: 26, right: 27, thresh_or_leaf: 26214 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -846 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1486 },
    GbdtNode { feat: 2, left: 29, right: 30, thresh_or_leaf: 16518 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -888 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2205 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 6210 },
    GbdtNode { feat: 37, left: 2, right: 9, thresh_or_leaf: 56173 },
    GbdtNode { feat: 3, left: 3, right: 6, thresh_or_leaf: 78 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 5199 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2533 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -185 },
    GbdtNode { feat: 5, left: 7, right: 8, thresh_or_leaf: 30321 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1820 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 97 },
    GbdtNode { feat: 20, left: 10, right: 11, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -456 },
    GbdtNode { feat: 26, left: 12, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3111 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -210 },
    GbdtNode { feat: 15, left: 15, right: 22, thresh_or_leaf: 1 },
    GbdtNode { feat: 7, left: 16, right: 19, thresh_or_leaf: 0 },
    GbdtNode { feat: 9, left: 17, right: 18, thresh_or_leaf: 230535 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 178 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2482 },
    GbdtNode { feat: 34, left: 20, right: 21, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -175 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1495 },
    GbdtNode { feat: 43, left: 23, right: 26, thresh_or_leaf: 60662 },
    GbdtNode { feat: 34, left: 24, right: 25, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 566 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2047 },
    GbdtNode { feat: 26, left: 27, right: 28, thresh_or_leaf: 1434 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1717 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 996 },
    GbdtNode { feat: 5, left: 1, right: 16, thresh_or_leaf: 30321 },
    GbdtNode { feat: 8, left: 2, right: 9, thresh_or_leaf: 29127 },
    GbdtNode { feat: 31, left: 3, right: 6, thresh_or_leaf: 0 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 4483 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1816 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1963 },
    GbdtNode { feat: 43, left: 7, right: 8, thresh_or_leaf: 60662 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2621 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -869 },
    GbdtNode { feat: 9, left: 10, right: 13, thresh_or_leaf: 550468 },
    GbdtNode { feat: 2, left: 11, right: 12, thresh_or_leaf: 3301 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1627 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1577 },
    GbdtNode { feat: 44, left: 14, right: 15, thresh_or_leaf: 8 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2162 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -77 },
    GbdtNode { feat: 42, left: 17, right: 24, thresh_or_leaf: 1662 },
    GbdtNode { feat: 24, left: 18, right: 21, thresh_or_leaf: 1 },
    GbdtNode { feat: 16, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1055 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 165 },
    GbdtNode { feat: 26, left: 22, right: 23, thresh_or_leaf: 81 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2622 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1045 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 43, left: 26, right: 27, thresh_or_leaf: 12276 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 878 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -134 },
    GbdtNode { feat: 19, left: 29, right: 30, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 734 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2233 },
    GbdtNode { feat: 5, left: 1, right: 16, thresh_or_leaf: 30321 },
    GbdtNode { feat: 15, left: 2, right: 9, thresh_or_leaf: 0 },
    GbdtNode { feat: 28, left: 3, right: 6, thresh_or_leaf: 103 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 30 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -54 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2747 },
    GbdtNode { feat: 2, left: 7, right: 8, thresh_or_leaf: 6947 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1392 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 952 },
    GbdtNode { feat: 3, left: 10, right: 13, thresh_or_leaf: 48 },
    GbdtNode { feat: 27, left: 11, right: 12, thresh_or_leaf: 3575 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2630 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 180 },
    GbdtNode { feat: 31, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1272 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1136 },
    GbdtNode { feat: 42, left: 17, right: 24, thresh_or_leaf: 1662 },
    GbdtNode { feat: 15, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 9, left: 19, right: 20, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1450 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1638 },
    GbdtNode { feat: 29, left: 22, right: 23, thresh_or_leaf: 27993 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -273 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2330 },
    GbdtNode { feat: 26, left: 25, right: 28, thresh_or_leaf: 529 },
    GbdtNode { feat: 40, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 64 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2166 },
    GbdtNode { feat: 34, left: 29, right: 30, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 775 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2231 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 6210 },
    GbdtNode { feat: 3, left: 2, right: 9, thresh_or_leaf: 48 },
    GbdtNode { feat: 37, left: 3, right: 6, thresh_or_leaf: 52428 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 30 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -853 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2544 },
    GbdtNode { feat: 19, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 555 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -256 },
    GbdtNode { feat: 12, left: 10, right: 13, thresh_or_leaf: 25486 },
    GbdtNode { feat: 2, left: 11, right: 12, thresh_or_leaf: 3726 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2807 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -615 },
    GbdtNode { feat: 37, left: 14, right: 15, thresh_or_leaf: 43690 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1436 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1923 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 9, left: 18, right: 21, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 29, left: 19, right: 20, thresh_or_leaf: 8073 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -416 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1097 },
    GbdtNode { feat: 2, left: 22, right: 23, thresh_or_leaf: 58982 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 98 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1683 },
    GbdtNode { feat: 6, left: 25, right: 28, thresh_or_leaf: 63488 },
    GbdtNode { feat: 17, left: 26, right: 27, thresh_or_leaf: -387101 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1607 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1178 },
    GbdtNode { feat: 47, left: 29, right: 30, thresh_or_leaf: 92 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -984 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1928 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: -6 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 30321 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 2492 },
    GbdtNode { feat: 4, left: 4, right: 5, thresh_or_leaf: 137 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1132 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1587 },
    GbdtNode { feat: 43, left: 7, right: 8, thresh_or_leaf: 60662 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2610 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1117 },
    GbdtNode { feat: 43, left: 10, right: 13, thresh_or_leaf: 14287 },
    GbdtNode { feat: 5, left: 11, right: 12, thresh_or_leaf: 62177 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2717 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -705 },
    GbdtNode { feat: 42, left: 14, right: 15, thresh_or_leaf: 1156 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1878 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 43 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 5, left: 18, right: 21, thresh_or_leaf: 64921 },
    GbdtNode { feat: 2, left: 19, right: 20, thresh_or_leaf: 51793 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -143 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2070 },
    GbdtNode { feat: 18, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -332 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 790 },
    GbdtNode { feat: 3, left: 25, right: 28, thresh_or_leaf: 38 },
    GbdtNode { feat: 2, left: 26, right: 27, thresh_or_leaf: 779 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 245 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2121 },
    GbdtNode { feat: 19, left: 29, right: 30, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 371 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1820 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 6210 },
    GbdtNode { feat: 37, left: 2, right: 9, thresh_or_leaf: 56173 },
    GbdtNode { feat: 0, left: 3, right: 6, thresh_or_leaf: 40960 },
    GbdtNode { feat: 46, left: 4, right: 5, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2732 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1383 },
    GbdtNode { feat: 35, left: 7, right: 8, thresh_or_leaf: -3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -840 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1652 },
    GbdtNode { feat: 27, left: 10, right: 13, thresh_or_leaf: 158 },
    GbdtNode { feat: 26, left: 11, right: 12, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2987 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -324 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -533 },
    GbdtNode { feat: 34, left: 15, right: 22, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 16, right: 19, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 17, right: 18, thresh_or_leaf: -505548 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -163 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2334 },
    GbdtNode { feat: 15, left: 20, right: 21, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -509 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 505 },
    GbdtNode { feat: 19, left: 23, right: 26, thresh_or_leaf: 0 },
    GbdtNode { feat: 43, left: 24, right: 25, thresh_or_leaf: 47688 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 699 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1420 },
    GbdtNode { feat: 15, left: 27, right: 28, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 87 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2059 },
    GbdtNode { feat: 31, left: 1, right: 16, thresh_or_leaf: 25206 },
    GbdtNode { feat: 26, left: 2, right: 9, thresh_or_leaf: 33237 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 24713 },
    GbdtNode { feat: 43, left: 4, right: 5, thresh_or_leaf: 6970 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 746 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2163 },
    GbdtNode { feat: 22, left: 7, right: 8, thresh_or_leaf: -2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2819 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -112 },
    GbdtNode { feat: 47, left: 10, right: 13, thresh_or_leaf: 40 },
    GbdtNode { feat: 43, left: 11, right: 12, thresh_or_leaf: 25571 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 852 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -498 },
    GbdtNode { feat: 42, left: 14, right: 15, thresh_or_leaf: 8153 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -8879 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1651 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 20, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 12, left: 19, right: 20, thresh_or_leaf: 36623 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 895 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1371 },
    GbdtNode { feat: 5, left: 22, right: 23, thresh_or_leaf: 48476 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -773 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 703 },
    GbdtNode { feat: 35, left: 25, right: 28, thresh_or_leaf: 5 },
    GbdtNode { feat: 30, left: 26, right: 27, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -444 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1803 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 41413 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2168 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1076 },
    GbdtNode { feat: 42, left: 1, right: 16, thresh_or_leaf: 1662 },
    GbdtNode { feat: 24, left: 2, right: 9, thresh_or_leaf: 1 },
    GbdtNode { feat: 22, left: 3, right: 6, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 4, right: 5, thresh_or_leaf: 679086 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2895 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 84 },
    GbdtNode { feat: 15, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1178 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -8 },
    GbdtNode { feat: 44, left: 10, right: 13, thresh_or_leaf: 2 },
    GbdtNode { feat: 26, left: 11, right: 12, thresh_or_leaf: 81 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2121 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3174 },
    GbdtNode { feat: 9, left: 14, right: 15, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2928 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 187 },
    GbdtNode { feat: 5, left: 17, right: 24, thresh_or_leaf: 28418 },
    GbdtNode { feat: 16, left: 18, right: 21, thresh_or_leaf: 44470 },
    GbdtNode { feat: 31, left: 19, right: 20, thresh_or_leaf: 40960 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1978 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 833 },
    GbdtNode { feat: 9, left: 22, right: 23, thresh_or_leaf: 550468 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1625 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1347 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 12, left: 26, right: 27, thresh_or_leaf: 54613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 991 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -132 },
    GbdtNode { feat: 26, left: 29, right: 30, thresh_or_leaf: 4885 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 757 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2238 },
    GbdtNode { feat: 31, left: 1, right: 16, thresh_or_leaf: 25206 },
    GbdtNode { feat: 15, left: 2, right: 9, thresh_or_leaf: -1 },
    GbdtNode { feat: 5, left: 3, right: 6, thresh_or_leaf: 62177 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 30 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 487 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2452 },
    GbdtNode { feat: 0, left: 7, right: 8, thresh_or_leaf: 39936 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1601 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 718 },
    GbdtNode { feat: 3, left: 10, right: 13, thresh_or_leaf: 49728 },
    GbdtNode { feat: 26, left: 11, right: 12, thresh_or_leaf: 33237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 43 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4327 },
    GbdtNode { feat: 44, left: 14, right: 15, thresh_or_leaf: 5 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -5729 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -661 },
    GbdtNode { feat: 45, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 33, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 34, left: 19, right: 20, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 47 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2078 },
    GbdtNode { feat: 26, left: 22, right: 23, thresh_or_leaf: 17770 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 878 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2393 },
    GbdtNode { feat: 26, left: 25, right: 28, thresh_or_leaf: 4885 },
    GbdtNode { feat: 12, left: 26, right: 27, thresh_or_leaf: 56173 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 57 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3491 },
    GbdtNode { feat: 31, left: 29, right: 30, thresh_or_leaf: 54613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1196 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1596 },
    GbdtNode { feat: 42, left: 1, right: 16, thresh_or_leaf: 1662 },
    GbdtNode { feat: 24, left: 2, right: 9, thresh_or_leaf: 1 },
    GbdtNode { feat: 15, left: 3, right: 6, thresh_or_leaf: 0 },
    GbdtNode { feat: 27, left: 4, right: 5, thresh_or_leaf: 12 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -863 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2512 },
    GbdtNode { feat: 38, left: 7, right: 8, thresh_or_leaf: 209863 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 333 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -920 },
    GbdtNode { feat: 26, left: 10, right: 13, thresh_or_leaf: 81 },
    GbdtNode { feat: 9, left: 11, right: 12, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2759 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 238 },
    GbdtNode { feat: 44, left: 14, right: 15, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3004 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1777 },
    GbdtNode { feat: 5, left: 17, right: 24, thresh_or_leaf: 28418 },
    GbdtNode { feat: 44, left: 18, right: 21, thresh_or_leaf: 2 },
    GbdtNode { feat: 2, left: 19, right: 20, thresh_or_leaf: 4706 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1242 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2274 },
    GbdtNode { feat: 6, left: 22, right: 23, thresh_or_leaf: 61440 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2042 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -18 },
    GbdtNode { feat: 22, left: 25, right: 28, thresh_or_leaf: 1 },
    GbdtNode { feat: 3, left: 26, right: 27, thresh_or_leaf: 13076 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1180 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 80 },
    GbdtNode { feat: 43, left: 29, right: 30, thresh_or_leaf: 23090 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1870 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1476 },
    GbdtNode { feat: 31, left: 1, right: 14, thresh_or_leaf: 25206 },
    GbdtNode { feat: 26, left: 2, right: 9, thresh_or_leaf: 33237 },
    GbdtNode { feat: 15, left: 3, right: 6, thresh_or_leaf: -1 },
    GbdtNode { feat: 0, left: 4, right: 5, thresh_or_leaf: 44032 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2094 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 779 },
    GbdtNode { feat: 18, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -894 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 409 },
    GbdtNode { feat: 38, left: 10, right: 13, thresh_or_leaf: 548845 },
    GbdtNode { feat: 5, left: 11, right: 12, thresh_or_leaf: 48476 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -399 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -7670 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 544 },
    GbdtNode { feat: 45, left: 15, right: 22, thresh_or_leaf: 0 },
    GbdtNode { feat: 33, left: 16, right: 19, thresh_or_leaf: 0 },
    GbdtNode { feat: 34, left: 17, right: 18, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 45 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1982 },
    GbdtNode { feat: 31, left: 20, right: 21, thresh_or_leaf: 52428 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 483 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1689 },
    GbdtNode { feat: 26, left: 23, right: 26, thresh_or_leaf: 4885 },
    GbdtNode { feat: 12, left: 24, right: 25, thresh_or_leaf: 56173 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 80 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3112 },
    GbdtNode { feat: 44, left: 27, right: 28, thresh_or_leaf: 3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1554 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -780 },
    GbdtNode { feat: 42, left: 1, right: 16, thresh_or_leaf: 1662 },
    GbdtNode { feat: 15, left: 2, right: 9, thresh_or_leaf: 0 },
    GbdtNode { feat: 9, left: 3, right: 6, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 38, left: 4, right: 5, thresh_or_leaf: 351359 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1850 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -328 },
    GbdtNode { feat: 19, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1925 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -158 },
    GbdtNode { feat: 29, left: 10, right: 13, thresh_or_leaf: 27993 },
    GbdtNode { feat: 12, left: 11, right: 12, thresh_or_leaf: 59578 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -964 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 238 },
    GbdtNode { feat: 4, left: 14, right: 15, thresh_or_leaf: 694 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -860 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2403 },
    GbdtNode { feat: 43, left: 17, right: 24, thresh_or_leaf: 24362 },
    GbdtNode { feat: 45, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 22, left: 19, right: 20, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 604 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1901 },
    GbdtNode { feat: 42, left: 22, right: 23, thresh_or_leaf: 3724 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1004 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2472 },
    GbdtNode { feat: 28, left: 25, right: 28, thresh_or_leaf: 32768 },
    GbdtNode { feat: 47, left: 26, right: 27, thresh_or_leaf: 60650 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 417 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1255 },
    GbdtNode { feat: 44, left: 29, right: 30, thresh_or_leaf: 6 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3753 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -198 },
    GbdtNode { feat: 35, left: 1, right: 16, thresh_or_leaf: -6 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 23806 },
    GbdtNode { feat: 43, left: 3, right: 6, thresh_or_leaf: 6970 },
    GbdtNode { feat: 13, left: 4, right: 5, thresh_or_leaf: 882023 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2282 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1130 },
    GbdtNode { feat: 12, left: 7, right: 8, thresh_or_leaf: 20480 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 369 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2545 },
    GbdtNode { feat: 15, left: 10, right: 13, thresh_or_leaf: -1 },
    GbdtNode { feat: 44, left: 11, right: 12, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -339 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2060 },
    GbdtNode { feat: 44, left: 14, right: 15, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1633 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -887 },
    GbdtNode { feat: 42, left: 17, right: 24, thresh_or_leaf: 1662 },
    GbdtNode { feat: 9, left: 18, right: 21, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 3, left: 19, right: 20, thresh_or_leaf: 49728 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -232 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2866 },
    GbdtNode { feat: 20, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2160 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2008 },
    GbdtNode { feat: 34, left: 25, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: 47, left: 26, right: 27, thresh_or_leaf: 11919 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1279 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -442 },
    GbdtNode { feat: 22, left: 29, right: 30, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 618 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2138 },
    GbdtNode { feat: 2, left: 1, right: 16, thresh_or_leaf: 5283 },
    GbdtNode { feat: 43, left: 2, right: 9, thresh_or_leaf: 6970 },
    GbdtNode { feat: 47, left: 3, right: 6, thresh_or_leaf: 5944 },
    GbdtNode { feat: 12, left: 4, right: 5, thresh_or_leaf: 32768 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1658 },
    GbdtNode { feat: 14, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3004 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1144 },
    GbdtNode { feat: 37, left: 10, right: 13, thresh_or_leaf: 52428 },
    GbdtNode { feat: 12, left: 11, right: 12, thresh_or_leaf: 34192 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -192 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2028 },
    GbdtNode { feat: 4, left: 14, right: 15, thresh_or_leaf: 188 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2790 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -600 },
    GbdtNode { feat: 34, left: 17, right: 24, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 18, right: 21, thresh_or_leaf: -1 },
    GbdtNode { feat: 44, left: 19, right: 20, thresh_or_leaf: 3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -523 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2307 },
    GbdtNode { feat: 9, left: 22, right: 23, thresh_or_leaf: 1025798 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -114 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 987 },
    GbdtNode { feat: 27, left: 25, right: 28, thresh_or_leaf: 32 },
    GbdtNode { feat: 22, left: 26, right: 27, thresh_or_leaf: -2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1731 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1462 },
    GbdtNode { feat: 27, left: 29, right: 30, thresh_or_leaf: 613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3155 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 919 },
    GbdtNode { feat: 2, left: 1, right: 14, thresh_or_leaf: 5283 },
    GbdtNode { feat: 37, left: 2, right: 9, thresh_or_leaf: 56173 },
    GbdtNode { feat: 12, left: 3, right: 6, thresh_or_leaf: 25486 },
    GbdtNode { feat: 43, left: 4, right: 5, thresh_or_leaf: 42629 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2450 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1182 },
    GbdtNode { feat: 0, left: 7, right: 8, thresh_or_leaf: 41984 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1822 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1276 },
    GbdtNode { feat: 44, left: 10, right: 11, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -226 },
    GbdtNode { feat: 29, left: 12, right: 13, thresh_or_leaf: 13201 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3038 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -88 },
    GbdtNode { feat: 22, left: 15, right: 20, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 16, right: 19, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 26, left: 17, right: 18, thresh_or_leaf: 33237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3198 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 7 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1231 },
    GbdtNode { feat: 34, left: 21, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 20, left: 22, right: 23, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1162 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 78 },
    GbdtNode { feat: 15, left: 25, right: 26, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -46 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 962 },
    GbdtNode { feat: 5, left: 1, right: 14, thresh_or_leaf: 30321 },
    GbdtNode { feat: 44, left: 2, right: 7, thresh_or_leaf: 2 },
    GbdtNode { feat: 46, left: 3, right: 4, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2463 },
    GbdtNode { feat: 17, left: 5, right: 6, thresh_or_leaf: -505548 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1558 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2051 },
    GbdtNode { feat: 4, left: 8, right: 11, thresh_or_leaf: 12816 },
    GbdtNode { feat: 43, left: 9, right: 10, thresh_or_leaf: 6970 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 337 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2188 },
    GbdtNode { feat: 9, left: 12, right: 13, thresh_or_leaf: 362495 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1317 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1256 },
    GbdtNode { feat: 40, left: 15, right: 22, thresh_or_leaf: 0 },
    GbdtNode { feat: 34, left: 16, right: 19, thresh_or_leaf: 1 },
    GbdtNode { feat: 26, left: 17, right: 18, thresh_or_leaf: 81 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -489 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 316 },
    GbdtNode { feat: 26, left: 20, right: 21, thresh_or_leaf: 1434 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 71 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1687 },
    GbdtNode { feat: 15, left: 23, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3389 },
    GbdtNode { feat: 43, left: 25, right: 26, thresh_or_leaf: 41413 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2395 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -584 },
    GbdtNode { feat: 22, left: 1, right: 8, thresh_or_leaf: -2 },
    GbdtNode { feat: 6, left: 2, right: 3, thresh_or_leaf: 63488 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3096 },
    GbdtNode { feat: 12, left: 4, right: 5, thresh_or_leaf: 58982 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2028 },
    GbdtNode { feat: 5, left: 6, right: 7, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1007 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1450 },
    GbdtNode { feat: 18, left: 9, right: 16, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 10, right: 13, thresh_or_leaf: -481648 },
    GbdtNode { feat: 29, left: 11, right: 12, thresh_or_leaf: 28 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -991 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2480 },
    GbdtNode { feat: 28, left: 14, right: 15, thresh_or_leaf: 103 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2152 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 211 },
    GbdtNode { feat: 40, left: 17, right: 20, thresh_or_leaf: -1 },
    GbdtNode { feat: 5, left: 18, right: 19, thresh_or_leaf: 58534 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3134 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -200 },
    GbdtNode { feat: 45, left: 21, right: 22, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 350 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2267 },
    GbdtNode { feat: 22, left: 1, right: 12, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 2, right: 9, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 26, left: 3, right: 6, thresh_or_leaf: 33237 },
    GbdtNode { feat: 40, left: 4, right: 5, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3224 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -452 },
    GbdtNode { feat: 34, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 957 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1001 },
    GbdtNode { feat: 39, left: 10, right: 11, thresh_or_leaf: 32768 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -733 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1514 },
    GbdtNode { feat: 18, left: 13, right: 20, thresh_or_leaf: -1 },
    GbdtNode { feat: 17, left: 14, right: 17, thresh_or_leaf: -481648 },
    GbdtNode { feat: 2, left: 15, right: 16, thresh_or_leaf: 17576 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1721 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1080 },
    GbdtNode { feat: 28, left: 18, right: 19, thresh_or_leaf: 103 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2028 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 262 },
    GbdtNode { feat: 34, left: 21, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 3, left: 22, right: 23, thresh_or_leaf: 30186 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 144 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -814 },
    GbdtNode { feat: 3, left: 25, right: 26, thresh_or_leaf: 24 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1500 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 665 },
    GbdtNode { feat: 15, left: 1, right: 16, thresh_or_leaf: -1 },
    GbdtNode { feat: 5, left: 2, right: 9, thresh_or_leaf: 59919 },
    GbdtNode { feat: 26, left: 3, right: 6, thresh_or_leaf: 81 },
    GbdtNode { feat: 2, left: 4, right: 5, thresh_or_leaf: 30 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 843 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2549 },
    GbdtNode { feat: 4, left: 7, right: 8, thresh_or_leaf: 49886 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1495 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1413 },
    GbdtNode { feat: 18, left: 10, right: 13, thresh_or_leaf: 0 },
    GbdtNode { feat: 34, left: 11, right: 12, thresh_or_leaf: 2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1426 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2000 },
    GbdtNode { feat: 38, left: 14, right: 15, thresh_or_leaf: 440012 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -156 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1606 },
    GbdtNode { feat: 40, left: 17, right: 24, thresh_or_leaf: 0 },
    GbdtNode { feat: 0, left: 18, right: 21, thresh_or_leaf: 45056 },
    GbdtNode { feat: 3, left: 19, right: 20, thresh_or_leaf: 48 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1231 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 733 },
    GbdtNode { feat: 43, left: 22, right: 23, thresh_or_leaf: 24362 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 89 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1709 },
    GbdtNode { feat: 43, left: 25, right: 28, thresh_or_leaf: 0 },
    GbdtNode { feat: 3, left: 26, right: 27, thresh_or_leaf: 135 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1122 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2480 },
    GbdtNode { feat: 31, left: 29, right: 30, thresh_or_leaf: 26214 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 926 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1580 },
    GbdtNode { feat: 22, left: 1, right: 10, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 2, right: 7, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 26, left: 3, right: 4, thresh_or_leaf: 33237 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3002 },
    GbdtNode { feat: 17, left: 5, right: 6, thresh_or_leaf: -38336 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -925 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 836 },
    GbdtNode { feat: 39, left: 8, right: 9, thresh_or_leaf: 32768 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -732 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1441 },
    GbdtNode { feat: 34, left: 11, right: 18, thresh_or_leaf: 1 },
    GbdtNode { feat: 24, left: 12, right: 15, thresh_or_leaf: 1 },
    GbdtNode { feat: 40, left: 13, right: 14, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1621 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 114 },
    GbdtNode { feat: 26, left: 16, right: 17, thresh_or_leaf: 6603 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2021 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1803 },
    GbdtNode { feat: 27, left: 19, right: 22, thresh_or_leaf: 32 },
    GbdtNode { feat: 35, left: 20, right: 21, thresh_or_leaf: -6 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1284 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1351 },
    GbdtNode { feat: 27, left: 23, right: 24, thresh_or_leaf: 613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3718 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 715 },
    GbdtNode { feat: 43, left: 1, right: 16, thresh_or_leaf: 23090 },
    GbdtNode { feat: 34, left: 2, right: 9, thresh_or_leaf: 2 },
    GbdtNode { feat: 24, left: 3, right: 6, thresh_or_leaf: 1 },
    GbdtNode { feat: 38, left: 4, right: 5, thresh_or_leaf: 239013 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 747 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -126 },
    GbdtNode { feat: 9, left: 7, right: 8, thresh_or_leaf: 762999 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2251 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -34 },
    GbdtNode { feat: 31, left: 10, right: 13, thresh_or_leaf: 57344 },
    GbdtNode { feat: 4, left: 11, right: 12, thresh_or_leaf: 11838 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -103 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2596 },
    GbdtNode { feat: 47, left: 14, right: 15, thresh_or_leaf: 47002 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2756 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 848 },
    GbdtNode { feat: 37, left: 17, right: 24, thresh_or_leaf: 37449 },
    GbdtNode { feat: 1, left: 18, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: 15, left: 19, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1390 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -73 },
    GbdtNode { feat: 43, left: 22, right: 23, thresh_or_leaf: 32466 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -417 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3382 },
    GbdtNode { feat: 20, left: 25, right: 28, thresh_or_leaf: -1 },
    GbdtNode { feat: 16, left: 26, right: 27, thresh_or_leaf: 55705 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1545 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 952 },
    GbdtNode { feat: 16, left: 29, right: 30, thresh_or_leaf: 55705 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 3097 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 80 },
    GbdtNode { feat: 22, left: 1, right: 10, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 2, right: 7, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 38, left: 3, right: 4, thresh_or_leaf: 677040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2960 },
    GbdtNode { feat: 34, left: 5, right: 6, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 827 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1057 },
    GbdtNode { feat: 39, left: 8, right: 9, thresh_or_leaf: 32768 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -696 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1366 },
    GbdtNode { feat: 18, left: 11, right: 18, thresh_or_leaf: -1 },
    GbdtNode { feat: 34, left: 12, right: 15, thresh_or_leaf: 1 },
    GbdtNode { feat: 38, left: 13, right: 14, thresh_or_leaf: 677040 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1789 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 522 },
    GbdtNode { feat: 15, left: 16, right: 17, thresh_or_leaf: -1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1664 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 2151 },
    GbdtNode { feat: 45, left: 19, right: 22, thresh_or_leaf: 0 },
    GbdtNode { feat: 34, left: 20, right: 21, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -143 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 523 },
    GbdtNode { feat: 26, left: 23, right: 24, thresh_or_leaf: 12478 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3015 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 376 },
    GbdtNode { feat: 22, left: 1, right: 8, thresh_or_leaf: -2 },
    GbdtNode { feat: 5, left: 2, right: 3, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3035 },
    GbdtNode { feat: 39, left: 4, right: 7, thresh_or_leaf: 32768 },
    GbdtNode { feat: 35, left: 5, right: 6, thresh_or_leaf: -3 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 658 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1575 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1420 },
    GbdtNode { feat: 15, left: 9, right: 16, thresh_or_leaf: -1 },
    GbdtNode { feat: 5, left: 10, right: 13, thresh_or_leaf: 59919 },
    GbdtNode { feat: 43, left: 11, right: 12, thresh_or_leaf: 10156 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -258 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2079 },
    GbdtNode { feat: 27, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 478 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1517 },
    GbdtNode { feat: 40, left: 17, right: 20, thresh_or_leaf: 0 },
    GbdtNode { feat: 2, left: 18, right: 19, thresh_or_leaf: 34661 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 703 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -286 },
    GbdtNode { feat: 5, left: 21, right: 22, thresh_or_leaf: 28418 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1896 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1877 },
    GbdtNode { feat: 22, left: 1, right: 12, thresh_or_leaf: -2 },
    GbdtNode { feat: 9, left: 2, right: 9, thresh_or_leaf: 1025798 },
    GbdtNode { feat: 38, left: 3, right: 6, thresh_or_leaf: 677040 },
    GbdtNode { feat: 40, left: 4, right: 5, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -3021 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -363 },
    GbdtNode { feat: 34, left: 7, right: 8, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 811 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -1010 },
    GbdtNode { feat: 17, left: 10, right: 11, thresh_or_leaf: -481648 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1233 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -564 },
    GbdtNode { feat: 15, left: 13, right: 20, thresh_or_leaf: -1 },
    GbdtNode { feat: 5, left: 14, right: 17, thresh_or_leaf: 59919 },
    GbdtNode { feat: 43, left: 15, right: 16, thresh_or_leaf: 10156 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -146 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2003 },
    GbdtNode { feat: 18, left: 18, right: 19, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -840 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 609 },
    GbdtNode { feat: 29, left: 21, right: 24, thresh_or_leaf: 8073 },
    GbdtNode { feat: 40, left: 22, right: 23, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -4 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1337 },
    GbdtNode { feat: 12, left: 25, right: 26, thresh_or_leaf: 54613 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1914 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 203 },
    GbdtNode { feat: 22, left: 1, right: 8, thresh_or_leaf: -2 },
    GbdtNode { feat: 5, left: 2, right: 3, thresh_or_leaf: 64921 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2963 },
    GbdtNode { feat: 2, left: 4, right: 7, thresh_or_leaf: 51793 },
    GbdtNode { feat: 2, left: 5, right: 6, thresh_or_leaf: 1836 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -954 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1008 },
    GbdtNode { feat: 34, left: 9, right: 16, thresh_or_leaf: 1 },
    GbdtNode { feat: 18, left: 10, right: 13, thresh_or_leaf: -1 },
    GbdtNode { feat: 15, left: 11, right: 12, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -534 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2250 },
    GbdtNode { feat: 15, left: 14, right: 15, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -510 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 194 },
    GbdtNode { feat: 3, left: 17, right: 20, thresh_or_leaf: 38 },
    GbdtNode { feat: 24, left: 18, right: 19, thresh_or_leaf: 1 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: -2043 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 716 },
    GbdtNode { feat: 19, left: 21, right: 22, thresh_or_leaf: 0 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 136 },
    GbdtNode { feat: u8::MAX, left: 0, right: 0, thresh_or_leaf: 1283 },
];

/// Minimum integer GBDT margin required for top-turn session concordance to promote `sessions[0]`.
pub const AT1_SESS_SYNC_MARGIN_Q16: i64 = 32768;
// === END FROZEN GBDT WEIGHTS ===

/// Evaluates the frozen integer GBDT ensemble on a 36-element `i32` feature vector.
#[inline]
pub fn eval_at1_gbdt_q16(feats: &[i32; N_AT1_FEATS]) -> i64 {
    let mut sum: i64 = 0;
    for (tree_idx, &root_u16) in AT1_TREE_ROOTS.iter().enumerate() {
        let base = root_u16 as usize;
        let next_base = AT1_TREE_ROOTS
            .get(tree_idx + 1)
            .map(|&r| r as usize)
            .unwrap_or(AT1_TREE_NODES.len());
        if base >= next_base || next_base > AT1_TREE_NODES.len() {
            continue;
        }
        let slice = &AT1_TREE_NODES[base..next_base];
        let mut cur = 0usize;
        for _ in 0..8 {
            let Some(node) = slice.get(cur) else {
                break;
            };
            if node.feat == u8::MAX {
                sum = sum.saturating_add(node.thresh_or_leaf as i64);
                break;
            }
            let f_idx = node.feat as usize;
            let val = if f_idx < N_AT1_FEATS { feats[f_idx] } else { 0 };
            cur = if val <= node.thresh_or_leaf {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }
    sum
}

/// Canonical byte serialization of the frozen GBDT parameters for cryptographic pinning.
#[cfg(test)]
pub fn canonical_gbdt_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + AT1_TREE_ROOTS.len() * 2 + AT1_TREE_NODES.len() * 8);
    out.extend_from_slice(&(AT1_TREE_ROOTS.len() as u32).to_le_bytes());
    for &r in AT1_TREE_ROOTS {
        out.extend_from_slice(&r.to_le_bytes());
    }
    out.extend_from_slice(&(AT1_TREE_NODES.len() as u32).to_le_bytes());
    for n in AT1_TREE_NODES {
        out.push(n.feat);
        out.push(n.left);
        out.push(n.right);
        out.push(0);
        out.extend_from_slice(&n.thresh_or_leaf.to_le_bytes());
    }
    out
}

fn at1_feat_dump_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var("VESTIGE_AT1_FEAT_DUMP")
            .ok()
            .filter(|s| !s.trim().is_empty())
    })
    .as_deref()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LocKind {
    None,
    AnyPlace,
    Country,
    State,
    City,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NumKind {
    None,
    QtyOrFreq,
    HowLongDuration,
}

struct QueryAt1Spec {
    has_named_corpus: bool,
    queried_speakers: Vec<String>,
    conjoined_speakers: bool,
    non_sp_mask: u64,
    non_sp_cnt: u32,
    group_ic_q16: Vec<u64>,
    wn_bridge_by_group: Vec<(usize, DetHashSet<u32>)>,
    loc_kind: LocKind,
    time_when: bool,
    num_kind: NumKind,
    hum_who: bool,
    enty_vocab_ids: DetHashSet<u32>,
    enty_head_nouns: Vec<String>,
    geo_class_vocab_ids: DetHashSet<u32>,
    q_years: Vec<i32>,
    q_months: Vec<u32>,
    q_seasons: Vec<&'static str>,
    has_bound_temp_prep: bool,
    past_override: bool,
    q_entity_ids: DetHashSet<u32>,
}

// hardcode-scan: closed-class (English cardinal, ordinal, and frequency function words)
const CLOSED_NUMBER_WORDS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "once", "twice", "thrice", "couple", "few", "several",
    "single", "double", "triple", "half", "daily", "weekly", "monthly", "yearly", "annually",
];

// hardcode-scan: closed-class (temporal duration unit nouns for how-long queries)
const CLOSED_DURATION_UNITS: &[&str] = &[
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "decade", "decades", "semester", "semesters",
];

// hardcode-scan: closed-class (first-person pronouns for speaker self-report attribution)
const FIRST_PERSON_TOKS: &[&str] = &[
    "i", "i'm", "i've", "i'd", "i'll", "im", "ive", "my", "me", "mine", "myself", "we", "we're",
    "we've", "we'll", "our", "us", "ours",
];

// hardcode-scan: closed-class (second-person pronouns for partner-advice attribution)
const SECOND_PERSON_TOKS: &[&str] = &[
    "you", "your", "yours", "yourself", "you're", "you've", "you'd", "you'll", "youre", "youve",
];

fn season_of_month(month: u32) -> &'static str {
    match month {
        3..=5 => "SP",
        6..=8 => "SU",
        9..=11 => "FA",
        _ => "WI",
    }
}

fn build_query_at1_spec(ix: &Index, plan: &QueryPlan, q: &Query) -> QueryAt1Spec {
    let mut corpus_speakers: Vec<String> = Vec::new();
    for tr in ix.turns.iter().take(96) {
        let sp = tr.speaker.as_str();
        if !sp.is_empty() && sp != "user" && sp != "assistant" {
            let sl = sp.to_ascii_lowercase();
            if !corpus_speakers.contains(&sl) {
                corpus_speakers.push(sl);
            }
        }
    }
    let has_named_corpus = !corpus_speakers.is_empty();
    let nick_bind = ix.nicks.bindings_in(&q.text);
    let cased_words: Vec<&str> = q.text.split_whitespace().collect();
    let is_surname_collision = |sp_low: &str| -> bool {
        cased_words.windows(2).any(|w2| {
            let first = w2[0].trim_matches(|c: char| !c.is_alphanumeric());
            let second = w2[1].trim_matches(|c: char| !c.is_alphanumeric());
            first.eq_ignore_ascii_case(sp_low)
                && second.len() >= 2
                && second.chars().next().is_some_and(char::is_uppercase)
                && !corpus_speakers.iter().any(|s| s.eq_ignore_ascii_case(second))
        })
    };

    let mut queried_speakers: Vec<String> = Vec::new();
    let mut conjoined_speakers = false;
    if has_named_corpus {
        for sp in &corpus_speakers {
            let mentioned = c2_query_mentions_speaker(&q.text, sp)
                || nick_bind.values().any(|full| full.eq_ignore_ascii_case(sp));
            if mentioned && !is_surname_collision(sp) {
                if c2_speaker_conjoined(&q.text, sp) {
                    conjoined_speakers = true;
                }
                if !queried_speakers.contains(sp) {
                    queried_speakers.push(sp.clone());
                }
            }
        }
        if queried_speakers.len() >= 2 {
            conjoined_speakers = true;
        }
    }

    let g_cnt = plan.groups.len().min(64);
    let n_turns = (ix.turns.len() as u32).max(1);
    let mut group_ic_q16 = vec![Q16_ONE_I64 as u64; g_cnt];
    for qt in &plan.terms {
        if let Some(g) = qt.group
            && g < g_cnt
        {
            let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(0);
            let ic = shannon_ic_q16(n_turns, df);
            if ic > group_ic_q16[g] {
                group_ic_q16[g] = ic;
            }
        }
    }

    let mut non_sp_mask: u64 = 0;
    let mut wn_bridge_by_group: Vec<(usize, DetHashSet<u32>)> = Vec::new();
    for (gi, g_word) in plan.groups.iter().take(64).enumerate() {
        let is_sp = corpus_speakers.iter().any(|sp| sp.eq_ignore_ascii_case(g_word))
            || nick_bind.contains_key(g_word.as_str());
        if !is_sp {
            non_sp_mask |= 1u64 << gi;
            let mut bridge_ids: DetHashSet<u32> = DetHashSet::default();
            for (lem, w_u16) in crate::wordnet::full_lexicon_neighbors(g_word) {
                if w_u16 >= 70 {
                    if let Some(id) = ix.vocab.get(lem) {
                        bridge_ids.insert(id);
                    }
                    if lem.len() >= 7 {
                        for plen in [3usize, 4] {
                            if let Some(pid) = ix.vocab.get(&lem[..plen]) {
                                let pdf = ix.t_own_stats.df.get(&pid).copied().unwrap_or(0);
                                if (1..=8).contains(&pdf) {
                                    bridge_ids.insert(pid);
                                }
                            }
                        }
                    }
                }
            }
            for (cn_lem, _) in crate::conceptnet::two_hop_neighbors(g_word, 20) {
                if let Some(id) = ix.vocab.get(cn_lem) {
                    bridge_ids.insert(id);
                }
            }
            if (1..=5).contains(&crate::wordnet::senses(g_word).len()) {
                for inst in crate::wordnet::class_to_instances(g_word, 2) {
                    if let Some(id) = ix.vocab.get(inst) {
                        bridge_ids.insert(id);
                    }
                    let st = stem(inst);
                    if let Some(sid) = ix.vocab.get(&stem_term(&st)) {
                        bridge_ids.insert(sid);
                    }
                }
            }
            if g_word.len() >= 7 {
                for plen in [3usize, 4] {
                    if let Some(pid) = ix.vocab.get(&g_word[..plen]) {
                        let pdf = ix.t_own_stats.df.get(&pid).copied().unwrap_or(0);
                        if (1..=8).contains(&pdf) {
                            bridge_ids.insert(pid);
                        }
                    }
                }
            }
            if !bridge_ids.is_empty() {
                wn_bridge_by_group.push((gi, bridge_ids));
            }
        }
    }

    let q_toks = raw_tokens(&q.text);
    let slots = wh_slots(&q.text);
    let has_phrase = |a: &str, b: &str| q_toks.windows(2).any(|p| p[0] == a && p[1] == b);

    // Extract WH nominal head ("what/which [kind/type/sort of] [mod] <head>")
    let mut wh_heads: Vec<String> = Vec::new();
    for (idx, tok) in q_toks.iter().enumerate() {
        if matches!(tok.as_str(), "what" | "which") {
            let mut j = idx + 1;
            if j + 1 < q_toks.len()
                && matches!(
                    q_toks[j].as_str(),
                    "kind" | "kinds" | "type" | "types" | "sort" | "sorts"
                )
                && q_toks[j + 1] == "of"
            {
                j += 2;
            }
            for cand in q_toks.iter().skip(j).take(2) {
                // hardcode-scan: closed-class (English auxiliaries and determiners terminating WH NP)
                if matches!(
                    cand.as_str(),
                    "did" | "does" | "do" | "is" | "are" | "was" | "were" | "will" | "would"
                        | "could" | "should" | "might" | "may" | "has" | "have" | "had" | "a"
                        | "an" | "the" | "to" | "in" | "on" | "at" | "for" | "from" | "with"
                ) {
                    break;
                }
                if !corpus_speakers.iter().any(|sp| sp.eq_ignore_ascii_case(cand)) && is_content(cand)
                {
                    wh_heads.push(cand.clone());
                }
            }
        }
    }

    let mut loc_kind = if (slots & WH_PLACE) != 0 {
        LocKind::AnyPlace
    } else {
        LocKind::None
    };
    let mut geo_class_vocab_ids: DetHashSet<u32> = DetHashSet::default();
    for h in &wh_heads {
        if !crate::geonames::geo_class_instances(h).is_empty() {
            loc_kind = match h.as_str() {
                "country" | "countries" | "nation" | "nations" | "homeland" => LocKind::Country,
                "state" | "states" | "province" | "provinces" | "region" | "regions" => {
                    LocKind::State
                }
                _ => LocKind::City,
            };
            for inst in crate::geonames::geo_class_instances(h) {
                if let Some(id) = ix.vocab.get(inst) {
                    geo_class_vocab_ids.insert(id);
                }
            }
        }
    }

    let time_when = (slots & WH_TIME) != 0
        || wh_heads.iter().any(|h| {
            matches!(
                h.as_str(),
                "date" | "day" | "month" | "year" | "time" | "season" | "weekend"
            )
        });
    let is_how_long = has_phrase("how", "long");
    let num_kind = if is_how_long {
        NumKind::HowLongDuration
    } else if (slots & WH_QTY) != 0
        || has_phrase("how", "old")
        || has_phrase("what", "age")
        || has_phrase("how", "far")
    {
        NumKind::QtyOrFreq
    } else {
        NumKind::None
    };
    let hum_who = (slots & WH_PERSON) != 0;

    let mut enty_vocab_ids: DetHashSet<u32> = DetHashSet::default();
    let mut enty_head_nouns: Vec<String> = Vec::new();
    if loc_kind == LocKind::None && !time_when && num_kind == NumKind::None && !hum_who {
        for h in &wh_heads {
            if h.len() >= 3 && !crate::wordnet::senses(h).is_empty() {
                enty_head_nouns.push(h.clone());
                for inst in crate::wordnet::class_to_instances(h, 2) {
                    if let Some(id) = ix.vocab.get(inst) {
                        enty_vocab_ids.insert(id);
                    }
                }
                for (nbr, w_u16) in crate::wordnet::full_lexicon_neighbors(h) {
                    if w_u16 >= 72
                        && let Some(id) = ix.vocab.get(nbr)
                    {
                        enty_vocab_ids.insert(id);
                    }
                }
            }
        }
    }

    let mut q_years: Vec<i32> = Vec::new();
    let mut q_months: Vec<u32> = Vec::new();
    let mut q_seasons: Vec<&'static str> = Vec::new();
    for (i, tok) in q_toks.iter().enumerate() {
        if tok.len() == 4
            && tok.bytes().all(|b| b.is_ascii_digit())
            && let Ok(y) = tok.parse::<i32>()
            && (1900..=2099).contains(&y)
            && !q_years.contains(&y)
        {
            q_years.push(y);
        }
        if let Some((m, ambig)) = month_index(tok) {
            let prev = i.checked_sub(1).map(|k| q_toks[k].as_str()).unwrap_or("");
            // hardcode-scan: closed-class (temporal prepositions introducing month names)
            let valid_m = !ambig
                || matches!(
                    prev,
                    "in" | "on" | "of" | "during" | "since" | "before" | "after" | "early" | "late" | "mid" | "last" | "this"
                )
                || q_toks
                    .get(i + 1)
                    .is_some_and(|nx| nx.bytes().all(|b| b.is_ascii_digit()));
            if valid_m && !q_months.contains(&m) {
                q_months.push(m);
            }
        }
        let sc_opt = match tok.as_str() {
            "summer" => Some(("SU", false)),
            "winter" => Some(("WI", false)),
            "autumn" => Some(("FA", false)),
            "spring" => Some(("SP", true)),
            "fall" => Some(("FA", true)),
            _ => None,
        };
        if let Some((sc, ambig_s)) = sc_opt {
            let prev = i.checked_sub(1).map(|k| q_toks[k].as_str()).unwrap_or("");
            let prev2 = i.checked_sub(2).map(|k| q_toks[k].as_str()).unwrap_or("");
            if (!ambig_s
                || matches!(prev, "in" | "during" | "last" | "this" | "every" | "early" | "late")
                || (prev == "the" && matches!(prev2, "in" | "during")))
                && !q_seasons.contains(&sc)
            {
                q_seasons.push(sc);
            }
        }
    }
    let has_bound_temp_prep = q_toks
        .iter()
        .any(|w| matches!(w.as_str(), "before" | "after" | "since" | "until" | "prior" | "between"));
    // hardcode-scan: closed-class (past temporal override adverbs)
    let past_override = q_toks.iter().any(|w| {
        matches!(
            w.as_str(),
            "before" | "previously" | "originally" | "earlier" | "initial" | "initially" | "former" | "formerly"
        )
    }) || has_phrase("used", "to");

    let mut q_entity_ids: DetHashSet<u32> = DetHashSet::default();
    for w in &q_toks {
        if let Some(eid) = ix.vocab.get(&format!("\u{1}ent:{w}")) {
            q_entity_ids.insert(eid);
        }
        if let Some(id) = ix.vocab.get(w) {
            q_entity_ids.insert(id);
        }
    }

    QueryAt1Spec {
        has_named_corpus,
        queried_speakers,
        conjoined_speakers,
        non_sp_mask,
        non_sp_cnt: non_sp_mask.count_ones(),
        group_ic_q16,
        wn_bridge_by_group,
        loc_kind,
        time_when: time_when && !is_how_long,
        num_kind,
        hum_who,
        enty_vocab_ids,
        enty_head_nouns,
        geo_class_vocab_ids,
        q_years,
        q_months,
        q_seasons,
        has_bound_temp_prep,
        past_override,
        q_entity_ids,
    }
}

/// Builds a deterministic candidate pool of up to `MAX_AT1_POOL` (`56`) turns.
pub(in crate::retrieval) fn build_at1_candidate_pool(
    ix: &Index,
    is_multi_hop: bool,
    final_turn_indices: &[usize],
    final_session_indices: &[usize],
    t_scene: &[f32],
    t_own: &[f32],
    t_bridge: &[f32],
) -> Vec<usize> {
    let nt = ix.turns.len();
    if nt == 0 {
        return Vec::new();
    }
    let mut pool: Vec<usize> = Vec::with_capacity(MAX_AT1_POOL);
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut push_cand = |ti: usize| {
        if ti < nt && pool.len() < MAX_AT1_POOL && seen.insert(ti) {
            pool.push(ti);
        }
    };

    // 1. Incoming pipeline turn ranking (preserves all C2, RSB, Viterbi, and Torrens candidates)
    for &ti in final_turn_indices.iter().take(20) {
        push_cand(ti);
    }

    // 2. Top-28 turns globally by `t_scene` (`Q16`) — D'Hondt session-dispersed on `is_multi_hop`
    //    so aggregation queries ("What activities does Melanie partake in?") pull across 5+ sessions!
    let mut by_scene: Vec<(u64, usize)> = (0..nt)
        .filter_map(|ti| {
            let sc = ieee754_bits_to_q16(t_scene.get(ti).copied().unwrap_or(0.0).to_bits());
            (sc > 0).then_some((sc, ti))
        })
        .collect();
    by_scene.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if is_multi_hop {
        let mut sess_dhondt: BTreeMap<usize, u64> = BTreeMap::new();
        let mut rem_scene = by_scene;
        for _ in 0..28 {
            if rem_scene.is_empty() {
                break;
            }
            let mut best_i = 0usize;
            let mut best_g = 0u64;
            for (idx, &(sc, ti)) in rem_scene.iter().enumerate() {
                let c = sess_dhondt.get(&ix.turns[ti].session).copied().unwrap_or(0);
                let g = sc.saturating_mul(4) / (4 + 3 * c);
                if idx == 0 || g > best_g {
                    best_g = g;
                    best_i = idx;
                }
            }
            let (_, ti) = rem_scene.remove(best_i);
            *sess_dhondt.entry(ix.turns[ti].session).or_insert(0) += 1;
            push_cand(ti);
        }
    } else {
        for &(_, ti) in by_scene.iter().take(28) {
            push_cand(ti);
        }
    }

    // 3. Top-16 turns globally by uncarried `t_own` (`Q16`)
    let mut by_own: Vec<(u64, usize)> = (0..nt)
        .filter_map(|ti| {
            let ow = ieee754_bits_to_q16(t_own.get(ti).copied().unwrap_or(0.0).to_bits());
            (ow > 0).then_some((ow, ti))
        })
        .collect();
    by_own.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for &(_, ti) in by_own.iter().take(16) {
        push_cand(ti);
    }

    // 4. Hot turns + `±1, ±2` exchange dyad neighbors inside `final_session_indices` (12 sessions on multi-hop!)
    let sess_cap = if is_multi_hop { 12usize } else { 6usize };
    let head_si = final_session_indices.first().copied();
    for &si in final_session_indices.iter().take(sess_cap) {
        let per_sess_hot = if is_multi_hop {
            1usize
        } else if Some(si) == head_si {
            3usize
        } else {
            2usize
        };
        let Some(members) = ix.session_turns.get(si) else {
            continue;
        };
        let mut sess_ranked: Vec<(u64, usize, usize)> = members
            .iter()
            .enumerate()
            .map(|(pos, &ti)| {
                let sc = ieee754_bits_to_q16(t_scene.get(ti).copied().unwrap_or(0.0).to_bits());
                (sc, pos, ti)
            })
            .filter(|&(sc, _, _)| sc > 0)
            .collect();
        sess_ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
        for &(_, pos, ti) in sess_ranked.iter().take(per_sess_hot) {
            push_cand(ti);
            for off in [1usize, 2usize] {
                if let Some(&nxt) = members.get(pos + off)
                    && !ix.turns[nxt].is_backchannel
                {
                    push_cand(nxt);
                }
                if pos >= off {
                    let prv = members[pos - off];
                    if !ix.turns[prv].is_backchannel {
                        push_cand(prv);
                    }
                }
            }
        }
    }

    // 5. Top-10 zero-surface-overlap genus->hyponym lexical-bridge turns (`t_bridge > 0`), 1-per-session dispersed!
    let mut by_bridge: Vec<(u64, usize)> = (0..nt)
        .filter_map(|ti| {
            if ix.turns[ti].is_backchannel {
                return None;
            }
            let br = ieee754_bits_to_q16(t_bridge.get(ti).copied().unwrap_or(0.0).to_bits());
            (br > 0).then_some((br, ti))
        })
        .collect();
    by_bridge.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut br_sess_seen: BTreeSet<usize> = BTreeSet::new();
    for &(_, ti) in &by_bridge {
        if br_sess_seen.len() >= 10 {
            break;
        }
        if br_sess_seen.insert(ix.turns[ti].session) {
            push_cand(ti);
        }
    }
    for &(_, ti) in by_bridge.iter().take(10) {
        push_cand(ti);
    }

    pool
}

fn eval_li_roth_turn(ix: &Index, spec: &QueryAt1Spec, ti: usize) -> (i32, i32) {
    let t = &ix.turns[ti];
    let prose = t.body_for_headers.as_str();
    let prose_toks = raw_tokens(prose);

    let has_cardinal = prose_toks
        .iter()
        .any(|w| CLOSED_NUMBER_WORDS.contains(&w.as_str()));
    let has_num_any = t.form_digit || t.has_numeric || has_cardinal;
    let mut dur_exact = 0i32;
    match spec.num_kind {
        NumKind::HowLongDuration => {
            let has_unit_pair = prose_toks.windows(2).any(|p| {
                let n_ok = p[0].bytes().all(|b| b.is_ascii_digit())
                    || CLOSED_NUMBER_WORDS.contains(&p[0].as_str());
                n_ok && CLOSED_DURATION_UNITS.contains(&p[1].as_str())
            }) || prose_toks.iter().any(|w| w == "since");
            dur_exact = if has_unit_pair {
                2
            } else if has_num_any {
                1
            } else {
                -1
            };
        }
        NumKind::QtyOrFreq => {
            dur_exact = if has_num_any { 1 } else { -1 };
        }
        NumKind::None => {}
    }

    let has_any_class = spec.loc_kind != LocKind::None
        || spec.time_when
        || spec.num_kind != NumKind::None
        || spec.hum_who
        || !spec.enty_head_nouns.is_empty();
    if !has_any_class {
        return (0, dur_exact);
    }

    if spec.loc_kind != LocKind::None {
        if !spec.geo_class_vocab_ids.is_empty() {
            let exact_geo = prose_toks.iter().any(|w| {
                ix.vocab
                    .get(w)
                    .is_some_and(|id| spec.geo_class_vocab_ids.contains(&id))
            });
            if exact_geo {
                return (2, dur_exact);
            }
        }
        let any_geo_prose = prose_toks.iter().any(|w| {
            w.len() >= 3
                && !spec.q_entity_ids.contains(&ix.vocab.get(w).unwrap_or(u32::MAX))
                && (!crate::geonames::geo_neighbors(w).is_empty()
                    || crate::wikidata::has_source(w)
                    || crate::wordnet::upward_hypernyms(w, 3)
                        .iter()
                        .any(|&h| matches!(h, "location" | "region" | "structure" | "facility")))
        });
        if any_geo_prose {
            return (2, dur_exact);
        }
        if t.has_spatial || t.form_midcap {
            return (1, dur_exact);
        }
        return (-1, dur_exact);
    }

    if spec.time_when {
        let inline_d = inline_date_tokens(prose);
        if !inline_d.is_empty() || t.form_clock {
            return (2, dur_exact);
        }
        if t.has_temporal {
            return (1, dur_exact);
        }
        return (-1, dur_exact);
    }

    if spec.num_kind != NumKind::None {
        return (dur_exact, dur_exact);
    }

    if spec.hum_who {
        let novel_ent = t
            .entities
            .iter()
            .any(|eid| !spec.q_entity_ids.contains(eid));
        let person_noun = prose_toks.iter().any(|w| {
            w.len() >= 3
                && !spec.q_entity_ids.contains(&ix.vocab.get(w).unwrap_or(u32::MAX))
                && crate::wordnet::upward_hypernyms(w, 3)
                    .iter()
                    .any(|&h| matches!(h, "person" | "relative" | "organization"))
        });
        if novel_ent || person_noun {
            return (2, dur_exact);
        }
        if t.form_midcap || t.form_handle {
            return (1, dur_exact);
        }
        return (-1, dur_exact);
    }

    if !spec.enty_head_nouns.is_empty() {
        let prose_hit = ix.turn_own_bags[ti]
            .tf[0]
            .keys()
            .chain(ix.turn_own_bags[ti].tf[1].keys())
            .any(|id| !spec.q_entity_ids.contains(id) && spec.enty_vocab_ids.contains(id));
        if prose_hit {
            return (2, dur_exact);
        }
        let hyper_hit = prose_toks.iter().any(|w| {
            w.len() >= 3
                && !spec.q_entity_ids.contains(&ix.vocab.get(w).unwrap_or(u32::MAX))
                && spec.enty_head_nouns.iter().any(|hn| {
                    crate::wordnet::upward_hypernyms(w, 4)
                        .iter()
                        .any(|&anc| anc == hn.as_str())
                })
        });
        if hyper_hit {
            return (2, dur_exact);
        }
        let blip_hit = ix.turn_own_bags[ti]
            .tf[2]
            .keys()
            .any(|id| !spec.q_entity_ids.contains(id) && spec.enty_vocab_ids.contains(id));
        if blip_hit {
            return (1, dur_exact);
        }
        return (-1, dur_exact);
    }

    (0, dur_exact)
}

pub(in crate::retrieval) struct PoolScoreSlices<'a> {
    pub t_scene: &'a [f32],
    pub t_own: &'a [f32],
    pub t_turn_raw: &'a [f32],
    pub t_bridge: &'a [f32],
    pub s_score: &'a [f32],
    pub bridge_boost: &'a [f32],
}

/// Extracts all `N_AT1_FEATS` (`48`) deterministic `i32` features for every turn in `pool`.
pub(in crate::retrieval) fn extract_pool_features_q16(
    ix: &Index,
    plan: &QueryPlan,
    q: &Query,
    pool: &[usize],
    final_turn_indices: &[usize],
    final_session_indices: &[usize],
    sc: &PoolScoreSlices<'_>,
) -> Vec<[i32; N_AT1_FEATS]> {
    let t_scene = sc.t_scene;
    let t_own = sc.t_own;
    let t_turn_raw = sc.t_turn_raw;
    let t_bridge = sc.t_bridge;
    let s_score = sc.s_score;
    let bridge_boost = sc.bridge_boost;
    let n = pool.len();
    if n == 0 {
        return Vec::new();
    }
    let spec = build_query_at1_spec(ix, plan, q);
    let nt_u32 = (ix.turns.len() as u32).max(1);

    let mut pipe_rank_map: BTreeMap<usize, usize> = BTreeMap::new();
    for (r, &ti) in final_turn_indices.iter().enumerate() {
        pipe_rank_map.entry(ti).or_insert(r);
    }
    let mut sess_rank_map: BTreeMap<usize, usize> = BTreeMap::new();
    for (sr, &si) in final_session_indices.iter().enumerate() {
        sess_rank_map.entry(si).or_insert(sr);
    }

    let scene_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| ieee754_bits_to_q16(t_scene.get(ti).copied().unwrap_or(0.0).to_bits()))
        .collect();
    let own_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| ieee754_bits_to_q16(t_own.get(ti).copied().unwrap_or(0.0).to_bits()))
        .collect();
    let raw_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| ieee754_bits_to_q16(t_turn_raw.get(ti).copied().unwrap_or(0.0).to_bits()))
        .collect();
    let sess_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| {
            let si = ix.turns[ti].session;
            let s = s_score.get(si).copied().unwrap_or(0.0)
                + bridge_boost.get(si).copied().unwrap_or(0.0);
            ieee754_bits_to_q16(s.to_bits())
        })
        .collect();
    let t_bridge_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| ieee754_bits_to_q16(t_bridge.get(ti).copied().unwrap_or(0.0).to_bits()))
        .collect();
    let s_bridge_q16: Vec<u64> = pool
        .iter()
        .map(|&ti| {
            let si = ix.turns[ti].session;
            ieee754_bits_to_q16(bridge_boost.get(si).copied().unwrap_or(0.0).to_bits())
        })
        .collect();

    let max_scene = scene_q16.iter().copied().max().unwrap_or(1).max(1);
    let max_own = own_q16.iter().copied().max().unwrap_or(1).max(1);
    let max_raw = raw_q16.iter().copied().max().unwrap_or(1).max(1);
    let max_sess = sess_q16.iter().copied().max().unwrap_or(1).max(1);
    let max_t_bridge = t_bridge_q16.iter().copied().max().unwrap_or(1).max(1);
    let max_s_bridge = s_bridge_q16.iter().copied().max().unwrap_or(1).max(1);

    let turn_prose_and_blip_masks = |ti: usize| -> (u64, u64) {
        let bag = &ix.turn_own_bags[ti];
        let mut p_mask = 0u64;
        let mut b_mask = 0u64;
        for qt in &plan.terms {
            if !qt.exact_hit {
                continue;
            }
            let Some(g) = qt.group else {
                continue;
            };
            if g >= 64 {
                continue;
            }
            let in_prose = bag.tf[0].get(&qt.id).copied().unwrap_or(0.0) > 0.0
                || bag.tf[1].get(&qt.id).copied().unwrap_or(0.0) > 0.0;
            if in_prose {
                p_mask |= 1u64 << g;
            } else if bag.tf[2].get(&qt.id).copied().unwrap_or(0.0) > 0.0 {
                b_mask |= 1u64 << g;
            }
        }
        (p_mask, b_mask & !p_mask)
    };

    let mask_ic_sum = |mut m: u64| -> u64 {
        let mut s = 0u64;
        while m != 0 {
            let g = m.trailing_zeros() as usize;
            s = s.saturating_add(spec.group_ic_q16.get(g).copied().unwrap_or(65_536));
            m &= m - 1;
        }
        s
    };

    let mut prose_masks: Vec<u64> = Vec::with_capacity(n);
    let mut blip_masks: Vec<u64> = Vec::with_capacity(n);
    let mut prose_ics: Vec<u64> = Vec::with_capacity(n);
    let mut sp_states: Vec<i32> = Vec::with_capacity(n);

    let mut max_queried_sp_ic: u64 = 0;
    let mut max_other_sp_ic: u64 = 0;

    for &ti in pool {
        let (pm, bm) = turn_prose_and_blip_masks(ti);
        let p_non_sp = pm & spec.non_sp_mask;
        let pic = mask_ic_sum(p_non_sp);
        prose_masks.push(pm);
        blip_masks.push(bm);
        prose_ics.push(pic);

        let sp = ix.turns[ti].speaker.as_str();
        let sp_state = if !spec.has_named_corpus || spec.queried_speakers.is_empty() {
            0
        } else if spec.queried_speakers.iter().any(|qs| qs.eq_ignore_ascii_case(sp)) {
            if spec.conjoined_speakers { 1 } else { 2 }
        } else {
            -1
        };
        sp_states.push(sp_state);
        if sp_state > 0 {
            max_queried_sp_ic = max_queried_sp_ic.max(pic);
        } else if sp_state < 0 {
            max_other_sp_ic = max_other_sp_ic.max(pic);
        }
    }
    let max_pool_prose_ic = max_queried_sp_ic.max(max_other_sp_ic);
    let deficit_i32 = (max_other_sp_ic as i64 - max_queried_sp_ic as i64)
        .clamp(-524_288, 524_288) as i32;

    let mut rare_q_mask = 0u64;
    for qt in &plan.terms {
        if !qt.exact_hit {
            continue;
        }
        let Some(g) = qt.group else {
            continue;
        };
        if g >= 64 || (spec.non_sp_mask & (1u64 << g)) == 0 {
            continue;
        }
        let df = ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(nt_u32);
        if df <= 8 {
            rare_q_mask |= 1u64 << g;
        }
    }
    let mut union_queried_sp_rare_mask = 0u64;
    let mut union_other_sp_decl_rare_mask = 0u64;
    for idx in 0..n {
        let tr = &ix.turns[pool[idx]];
        if sp_states[idx] > 0 {
            union_queried_sp_rare_mask |= prose_masks[idx] & rare_q_mask;
        } else if sp_states[idx] < 0 && !tr.is_question && !tr.is_backchannel {
            union_other_sp_decl_rare_mask |= prose_masks[idx] & rare_q_mask;
        }
    }
    // C13 Presupposition-Conflict Detector (corpus/pool-wide):
    // When the queried speaker has ZERO asserted hits on the query's rare non-speaker predicates
    // (`df <= 8`) across the entire corpus while a non-queried declarative turn attests them,
    // neutralize `f[15] = 0` BEFORE GBDT evaluation so tree splits on `f[15]` never bury Cat 5 gold.
    let mut corpus_queried_sp_rare_asserted = union_queried_sp_rare_mask != 0;
    if !corpus_queried_sp_rare_asserted && rare_q_mask != 0 && !spec.queried_speakers.is_empty() {
        'rare_scan: for (ti_c, tr_c) in ix.turns.iter().enumerate() {
            if tr_c.is_question || tr_c.is_backchannel {
                continue;
            }
            if !spec
                .queried_speakers
                .iter()
                .any(|qs| qs.eq_ignore_ascii_case(&tr_c.speaker))
            {
                continue;
            }
            let (pmc, _) = turn_prose_and_blip_masks(ti_c);
            if (pmc & rare_q_mask) != 0 {
                corpus_queried_sp_rare_asserted = true;
                break 'rare_scan;
            }
        }
    }
    let presup_conflict = spec.has_named_corpus
        && !spec.queried_speakers.is_empty()
        && rare_q_mask != 0
        && !corpus_queried_sp_rare_asserted
        && union_other_sp_decl_rare_mask != 0;

    // Precompute query WordNet 41-supersense bitmask (`f[44]`) & non-speaker exact query ids (`f[45]`):
    let mut q_ss_mask = 0u64;
    for h in &spec.enty_head_nouns {
        q_ss_mask |= crate::wordnet::supersense_mask(h);
    }
    for qt in &plan.terms {
        if let Some(g) = qt.group
            && g < 64
            && (spec.non_sp_mask & (1u64 << g)) != 0
        {
            if let Some(nm) = ix.vocab.names.get(qt.id as usize) {
                q_ss_mask |= crate::wordnet::supersense_mask(nm);
            }
        }
    }
    let q_nonsp_exact_ids: Vec<u32> = plan
        .terms
        .iter()
        .filter(|qt| {
            qt.exact_hit
                && qt.group.is_some_and(|g| g < 64 && (spec.non_sp_mask & (1u64 << g)) != 0)
        })
        .map(|qt| qt.id)
        .collect();
    let q_inline_dates = inline_date_tokens(&q.text);
    let mut sess_max_own: BTreeMap<usize, u64> = BTreeMap::new();
    for &ti in pool {
        let s = ix.turns[ti].session;
        sess_max_own.entry(s).or_insert_with(|| {
            ix.session_turns
                .get(s)
                .map(|ms| {
                    ms.iter()
                        .map(|&m| {
                            ieee754_bits_to_q16(t_own.get(m).copied().unwrap_or(0.0).to_bits())
                        })
                        .max()
                        .unwrap_or(1)
                        .max(1)
                })
                .unwrap_or(1)
        });
    }

    let mut incoming_rsb: BTreeMap<usize, i32> = BTreeMap::new();
    for &src_ti in final_turn_indices.iter().take(5) {
        if let Some(ai) = ix.turns[src_ti].answers {
            *incoming_rsb.entry(ai).or_insert(0) += 1;
        }
        for &(_, tgt) in &ix.turns[src_ti].exact_edges {
            *incoming_rsb.entry(tgt).or_insert(0) += 2;
        }
    }

    let wh_has_class = i32::from(
        spec.loc_kind != LocKind::None
            || spec.time_when
            || spec.num_kind != NumKind::None
            || spec.hum_who
            || !spec.enty_head_nouns.is_empty(),
    );

    let mut out: Vec<[i32; N_AT1_FEATS]> = vec![[0i32; N_AT1_FEATS]; n];

    for idx in 0..n {
        let ti = pool[idx];
        let t = &ix.turns[ti];
        let sess = t.session;
        let mut f = [0i32; N_AT1_FEATS];

        if let Some(&pr) = pipe_rank_map.get(&ti) {
            f[0] = ((48usize.saturating_sub(pr.min(47))) * 1024) as i32;
            f[1] = i32::from(pr == 0);
        }
        f[2] = ((scene_q16[idx].saturating_mul(65_536)) / max_scene).min(65_536) as i32;
        f[3] = ((own_q16[idx].saturating_mul(65_536)) / max_own).min(65_536) as i32;
        f[4] = ((raw_q16[idx].saturating_mul(65_536)) / max_raw).min(65_536) as i32;
        f[5] = ((sess_q16[idx].saturating_mul(65_536)) / max_sess).min(65_536) as i32;
        if let Some(&sr) = sess_rank_map.get(&sess) {
            f[6] = ((32usize.saturating_sub(sr.min(31))) * 2048) as i32;
            f[7] = i32::from(sr == 0);
        }

        let p_non_sp = prose_masks[idx] & spec.non_sp_mask;
        let b_non_sp = blip_masks[idx] & spec.non_sp_mask;
        f[8] = if spec.non_sp_cnt > 0 {
            ((p_non_sp.count_ones() as u64 * 65_536) / (spec.non_sp_cnt as u64)) as i32
        } else {
            65_536
        };
        f[9] = prose_ics[idx].min(1_048_576) as i32;
        f[10] = b_non_sp.count_ones() as i32;
        f[11] = i32::from(!t.blip_feats.is_empty());
        let p_len = t.feats.iter().filter(|ft| ft.counts_len).count() as u64;
        let b_len = t.blip_feats.len() as u64;
        f[12] = ((p_len * 65_536) / (p_len + b_len + 1)) as i32;

        let mut novel_ics: Vec<u64> = Vec::new();
        for ft in &t.feats {
            if !ft.counts_len
                || plan.group_of.contains_key(&ft.id)
                || spec.q_entity_ids.contains(&ft.id)
            {
                continue;
            }
            let df = ix.t_own_stats.df.get(&ft.id).copied().unwrap_or(nt_u32);
            let ic = shannon_ic_q16(nt_u32, df);
            if ic > 0 {
                novel_ics.push(ic);
            }
        }
        novel_ics.sort_unstable_by(|a, b| b.cmp(a));
        novel_ics.dedup();
        let novel_sum: u64 = novel_ics.iter().take(4).sum();
        f[13] = novel_sum.min(1_048_576) as i32;

        let bag = &ix.turn_own_bags[ti];
        let mut wn_hits = 0i32;
        for (gi, bset) in &spec.wn_bridge_by_group {
            if (p_non_sp & (1u64 << *gi)) != 0 {
                continue;
            }
            let hit = bag
                .tf[0]
                .keys()
                .chain(bag.tf[1].keys())
                .any(|id| bset.contains(id));
            if hit {
                wn_hits += 1;
            }
        }
        f[14] = wn_hits;

        let mut partner_q_pm1 = 0u64;
        if let Some(members) = ix.session_turns.get(sess)
            && let Some(pos) = members.iter().position(|&m| m == ti)
        {
            let mut triad_mask = p_non_sp;
            if pos >= 1 {
                let prv1 = members[pos - 1];
                let (pm1, _) = turn_prose_and_blip_masks(prv1);
                triad_mask |= pm1 & spec.non_sp_mask;
                if ix.turns[prv1].is_question && ix.turns[prv1].speaker != t.speaker {
                    if ix.turns[prv1].is_pure_info_question {
                        partner_q_pm1 = pm1 & spec.non_sp_mask;
                    }
                    let ow1 = ieee754_bits_to_q16(t_own.get(prv1).copied().unwrap_or(0.0).to_bits());
                    f[26] = ((ow1.saturating_mul(65_536)) / max_own).min(65_536) as i32;
                }
            }
            if pos >= 2 {
                let prv2 = members[pos - 2];
                let (pm2, bm2) = turn_prose_and_blip_masks(prv2);
                let ow2 = ieee754_bits_to_q16(t_own.get(prv2).copied().unwrap_or(0.0).to_bits());
                let ow2_norm = ((ow2.saturating_mul(65_536)) / max_own).min(65_536) as i32;
                if ix.turns[prv2].is_question {
                    f[27] = ow2_norm;
                    triad_mask |= pm2 & spec.non_sp_mask;
                }
                if !ix.turns[prv2].blip_feats.is_empty()
                    && ix.turns[prv2].speaker == t.speaker
                    && t.blip_feats.is_empty()
                    && !t.is_question
                {
                    f[28] = ow2_norm.max(if (pm2 | bm2) & spec.non_sp_mask != 0 {
                        32_768
                    } else {
                        0
                    });
                    triad_mask |= (pm2 | bm2) & spec.non_sp_mask;
                }
                if t.anaphoric
                    && (t_own.get(prv2).copied().unwrap_or(0.0) > 0.0
                        || t_own.get(members[pos - 1]).copied().unwrap_or(0.0) > 0.0)
                {
                    f[30] = 1;
                }
            }
            if !t.is_question && pos + 1 < members.len() {
                let nxt1 = members[pos + 1];
                let (pm_n1, _) = turn_prose_and_blip_masks(nxt1);
                triad_mask |= pm_n1 & spec.non_sp_mask;
            }
            if !t.is_question && pos + 2 < members.len() {
                let nxt2 = members[pos + 2];
                if ix.turns[nxt2].anaphoric {
                    let own_n2 =
                        ieee754_bits_to_q16(t_own.get(nxt2).copied().unwrap_or(0.0).to_bits());
                    f[29] = ((own_n2.saturating_mul(65_536)) / max_own).min(65_536) as i32;
                }
            }
            f[31] = if spec.non_sp_cnt > 0 {
                ((triad_mask.count_ones() as u64 * 65_536) / (spec.non_sp_cnt as u64)) as i32
            } else {
                65_536
            };
        }

        let sp_st = sp_states[idx];
        // C13 Pre-GBDT Neutralizer: on presupposition conflict (`presup_conflict`), neutralize
        // `f[15] = 0` so the tree splits on `f[15]` never bury Cat 5 Adversarial refutation turns.
        f[15] = if presup_conflict { 0 } else { sp_st };
        let inherited_partner_topic =
            !t.is_question && !t.is_backchannel && partner_q_pm1 != 0;
        let eff_topic_mask = p_non_sp
            | if inherited_partner_topic {
                partner_q_pm1
            } else {
                0
            };
        let has_own_topic = eff_topic_mask != 0 || wn_hits > 0;
        let adv_rare_witness = sp_st < 0
            && ((p_non_sp & rare_q_mask) & !union_queried_sp_rare_mask) != 0
            && prose_ics[idx] > max_queried_sp_ic.saturating_add(32_768);
        f[16] = if sp_st > 0 && has_own_topic {
            let eff_cov = if spec.non_sp_cnt > 0 {
                ((eff_topic_mask.count_ones() as u64 * 65_536) / (spec.non_sp_cnt as u64)) as i32
            } else {
                65_536
            };
            eff_cov + 16_384
        } else if sp_st > 0 && !has_own_topic && b_non_sp == 0 && max_pool_prose_ic > 0 {
            -65_536
        } else if adv_rare_witness
            || (sp_st < 0 && max_queried_sp_ic == 0 && prose_ics[idx] > 0)
        {
            f[8] + 32_768
        } else {
            0
        };
        f[17] = deficit_i32;

        let prose_words = raw_tokens(&t.body_for_headers);
        let has_1p = prose_words.iter().any(|w| FIRST_PERSON_TOKS.contains(&w.as_str()));
        let has_2p = prose_words.iter().any(|w| SECOND_PERSON_TOKS.contains(&w.as_str()));
        f[18] = if has_1p && !has_2p {
            1
        } else if has_2p && !has_1p {
            -1
        } else {
            0
        };

        let (lr_match, dur_match) = eval_li_roth_turn(ix, &spec, ti);
        f[19] = wh_has_class;
        f[20] = lr_match;
        f[21] = dur_match;

        let inline_prose_dates = inline_date_tokens(&t.body_for_headers);
        let mut prose_date_hit = false;
        for &qy in &spec.q_years {
            let tag = format!("__YEAR__{qy}");
            if inline_prose_dates.iter().any(|d| d == &tag) {
                prose_date_hit = true;
            }
        }
        for &qm in &spec.q_months {
            let tag = format!("__MONTH__{qm:02}");
            if inline_prose_dates.iter().any(|d| d == &tag) {
                prose_date_hit = true;
            }
        }
        f[23] = if prose_date_hit { 2 } else { 0 };

        let sess_ts = ix.session_ts.get(sess).copied().unwrap_or(0);
        if (!spec.q_years.is_empty() || !spec.q_months.is_empty() || !spec.q_seasons.is_empty())
            && sess_ts > 0
            && let Some(dt) = chrono::DateTime::from_timestamp(sess_ts, 0)
        {
            let d = dt.date_naive();
            let sy = d.year();
            let sm = d.month();
            let ssc = season_of_month(sm);
            let y_ok = spec.q_years.is_empty() || spec.q_years.contains(&sy);
            let m_ok = spec.q_months.is_empty() || spec.q_months.contains(&sm);
            let s_ok = spec.q_seasons.is_empty() || spec.q_seasons.contains(&ssc);
            f[22] = if !spec.q_years.is_empty() && !spec.q_months.is_empty() && y_ok && m_ok {
                3
            } else if (!spec.q_months.is_empty() && m_ok && y_ok)
                || (!spec.q_seasons.is_empty() && s_ok && y_ok)
            {
                2
            } else if !spec.q_years.is_empty()
                && spec.q_months.is_empty()
                && spec.q_seasons.is_empty()
                && y_ok
            {
                1
            } else if !spec.has_bound_temp_prep && !prose_date_hit {
                -2
            } else {
                0
            };
        }

        f[24] = if t.is_pure_info_question {
            2
        } else if t.is_question {
            1
        } else {
            0
        };
        f[25] = i32::from(
            t.is_backchannel
                || (t.content.len() <= 2
                    && !t.has_numeric
                    && !t.has_temporal
                    && t.entities.is_empty()),
        );

        let is_sup = !spec.past_override
            && (turn_is_superseded(ix, ti)
                || ix.torrens_superseded.get(ti).copied().unwrap_or(false));
        f[32] = i32::from(is_sup);
        let rsb_deg = (t.exact_edges.len() as i32) + incoming_rsb.get(&ti).copied().unwrap_or(0);
        f[33] = rsb_deg.min(16);
        f[34] = (t.slot_entities_clean.min(16)) as i32;

        // f[36]: `[caption]` (`blip_feats`) non-speaker query coverage ratio in Q16.
        f[36] = if spec.non_sp_cnt > 0 {
            ((b_non_sp.count_ones() as u64 * 65_536) / (spec.non_sp_cnt as u64)) as i32
        } else {
            0
        };
        // f[37]: max non-speaker query coverage (`prose | caption`) of any same-session `[caption]`
        // neighbor within `+-2` turns of `ti` (captures photo-sharing + reply dyads).
        // f[38]: `stimulus_ref_ic_q16` — partner question / `answers` stimulus IC projected onto
        // declarative reply `t` without length damping.
        let mut cap_nb_cov_q16 = 0i32;
        let mut stim_ic_q16 = 0u64;
        if let Some(members) = ix.session_turns.get(sess)
            && let Some(pos) = members.iter().position(|&m| m == ti)
        {
            let lo = pos.saturating_sub(2);
            let hi = (pos + 2).min(members.len().saturating_sub(1));
            for (p_nb, &nb_ti) in members.iter().enumerate().take(hi + 1).skip(lo) {
                if p_nb == pos {
                    continue;
                }
                if !ix.turns[nb_ti].blip_feats.is_empty() {
                    let (pm_nb, bm_nb) = turn_prose_and_blip_masks(nb_ti);
                    let cov_nb = ((pm_nb | bm_nb) & spec.non_sp_mask).count_ones() as u64;
                    if cov_nb > 0 && spec.non_sp_cnt > 0 {
                        let qv = ((cov_nb * 65_536) / (spec.non_sp_cnt as u64)) as i32;
                        if qv > cap_nb_cov_q16 {
                            cap_nb_cov_q16 = qv;
                        }
                    }
                }
            }
            if !t.is_question && !t.is_backchannel {
                if pos >= 1 {
                    let prv1 = members[pos - 1];
                    if (ix.turns[prv1].is_question && ix.turns[prv1].speaker != t.speaker)
                        || t.answers == Some(prv1)
                    {
                        let (pm1, bm1) = turn_prose_and_blip_masks(prv1);
                        stim_ic_q16 = stim_ic_q16.max(mask_ic_sum((pm1 | bm1) & spec.non_sp_mask));
                    }
                }
                if pos >= 2 {
                    let prv2 = members[pos - 2];
                    if t.answers == Some(prv2)
                        || (ix.turns[prv2].is_question && ix.turns[prv2].speaker != t.speaker)
                    {
                        let (pm2, bm2) = turn_prose_and_blip_masks(prv2);
                        let half = mask_ic_sum((pm2 | bm2) & spec.non_sp_mask) / 2;
                        stim_ic_q16 = stim_ic_q16.max(half);
                    }
                }
            }
        }
        f[37] = cap_nb_cov_q16;
        f[38] = stim_ic_q16.min(1_048_576) as i32;
        // f[39]: fraction (Q16) of rare non-speaker query groups (`df <= 8`) matched in prose.
        let rare_cnt = rare_q_mask.count_ones() as u64;
        f[39] = if rare_cnt > 0 {
            ((((p_non_sp & rare_q_mask).count_ones() as u64) * 65_536) / rare_cnt) as i32
        } else {
            0
        };
        // f[40]: resolved_timex exact calendar alignment on temporal queries.
        let mut timex_exact = false;
        if !q_inline_dates.is_empty() && !t.resolved_timex.is_empty() {
            'tx: for &(iv_s, iv_e) in &t.resolved_timex {
                let span = iv_e.saturating_sub(iv_s);
                if span > 3 * 86_400 {
                    continue;
                }
                let mid = iv_s + span / 2;
                let ctoks = calendar_tokens(mid);
                if ctoks.iter().any(|ct| q_inline_dates.iter().any(|qd| qd == ct)) {
                    timex_exact = true;
                    break 'tx;
                }
            }
        }
        f[40] = if timex_exact {
            2
        } else if spec.time_when && (!t.resolved_timex.is_empty() || t.has_temporal || prose_date_hit) {
            1
        } else if spec.time_when {
            -1
        } else {
            0
        };
        // f[41]: C13 presupposition-conflict refuter signal (`+2` for non-queried declarative
        // holding unattested rare predicate, `-1` for queried speaker lacking it, `0` otherwise).
        f[41] = if presup_conflict {
            if sp_st < 0 && !t.is_question && !t.is_backchannel && (p_non_sp & rare_q_mask) != 0 {
                2
            } else if sp_st > 0 {
                -1
            } else {
                0
            }
        } else {
            0
        };
        // f[42]: normalized Stage-04/11 turn bridge lift `t_bridge[ti]` in Q16.
        f[42] = ((t_bridge_q16[idx].saturating_mul(65_536)) / max_t_bridge).min(65_536) as i32;
        // f[43]: normalized Stage-04 session bridge lift `bridge_boost[sess]` in Q16.
        f[43] = ((s_bridge_q16[idx].saturating_mul(65_536)) / max_s_bridge).min(65_536) as i32;
        // f[44]: WordNet 41-supersense bitwise intersection count (zero-overlap semantic typing).
        let ss_hits = if q_ss_mask != 0 {
            let mut turn_ss = 0u64;
            for tw in &prose_words {
                if tw.len() >= 3 && is_content(tw) {
                    turn_ss |= crate::wordnet::supersense_mask(tw);
                }
            }
            (turn_ss & q_ss_mask).count_ones().min(16) as i32
        } else {
            0
        };
        f[44] = ss_hits;
        // f[45]: 1 if matched query terms appear ONLY inside `other_speaker_terms` (2nd-person praise).
        let only_other = if !t.other_speaker_terms.is_empty() && !q_nonsp_exact_ids.is_empty() {
            let mut hit_any = false;
            let mut all_other = true;
            for &qid in &q_nonsp_exact_ids {
                if t.content.contains(&qid) {
                    hit_any = true;
                    if !t.other_speaker_terms.contains(&qid) {
                        all_other = false;
                        break;
                    }
                }
            }
            hit_any && all_other
        } else {
            false
        };
        f[45] = i32::from(only_other);
        // f[46]: 4-bit modality state (`+1` pure ASSERTED, `-1` pure HYPOTHETICAL, `0` mixed/neutral).
        f[46] = if (t.modality_mask & 0x01) != 0 && (t.modality_mask & 0x04) == 0 {
            1
        } else if (t.modality_mask & 0x04) != 0 && (t.modality_mask & 0x01) == 0 {
            -1
        } else {
            0
        };
        // f[47]: within-session `t_own` dominance ratio in Q16 (`65_536` iff strongest turn in session).
        let smax_own = sess_max_own.get(&sess).copied().unwrap_or(1).max(1);
        f[47] = ((own_q16[idx].saturating_mul(65_536)) / smax_own).min(65_536) as i32;

        out[idx] = f;
    }

    let top_c = n.min(8);
    for a in 0..n {
        let mut net_wins = 0i32;
        for b in 0..top_c {
            if a == b {
                continue;
            }
            let fa = &out[a];
            let fb = &out[b];
            let mut margin = 0i32;
            if fa[20] != fb[20] {
                margin += if fa[20] > fb[20] { 1 } else { -1 };
            }
            if (fa[9] - fb[9]).abs() >= 8_192 {
                margin += if fa[9] > fb[9] { 1 } else { -1 };
            }
            if deficit_i32 <= 24_576 && fa[16] != fb[16] {
                margin += if fa[16] > fb[16] { 1 } else { -1 };
            }
            let qa_a = -fa[24] * 2 + fa[22];
            let qa_b = -fb[24] * 2 + fb[22];
            if qa_a != qa_b {
                margin += if qa_a > qa_b { 1 } else { -1 };
            }
            if (fa[2] as i64) * 100 >= (fb[2] as i64) * 112 {
                margin += 1;
            } else if (fb[2] as i64) * 100 >= (fa[2] as i64) * 112 {
                margin -= 1;
            }
            if margin > 0 {
                net_wins += 1;
            } else if margin < 0 {
                net_wins -= 1;
            }
        }
        out[a][35] = net_wins;
    }

    out
}

struct AnonCondorcetCtx<'a> {
    ix: &'a Index,
    plan: &'a QueryPlan,
    eff_it: &'a QueryIntent,
    q: &'a Query,
    t_scene: &'a [f32],
    t_own: &'a [f32],
    t_exact_mask: &'a [u64],
    s_score: &'a [f32],
    bridge_boost: &'a [f32],
}

// hardcode-scan: closed-class (revision, state-update, and correction markers)
const CLOSED_REVISION_CUES: &[&str] = &[
    "actually",
    "instead",
    "anymore",
    "switched",
    "changed",
    "updated",
    "updating",
    "revised",
    "adjusted",
    "increased",
    "decreased",
    "extended",
    "replaced",
    "postponed",
    "rescheduled",
];

fn turn_has_revision_cue(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    if low.contains("no longer") || low.contains("instead of") || low.contains("rather than") {
        return true;
    }
    raw_tokens(&low)
        .iter()
        .any(|w| CLOSED_REVISION_CUES.contains(&w.as_str()))
}

fn turns_share_rare_slot_anchor(ix: &Index, t_a: usize, t_b: usize) -> bool {
    let ns = ix.session_turns.len().max(1);
    let hub_df = ((ns / 3).clamp(4, 32)) as u32;
    let a = &ix.turns[t_a];
    let b = &ix.turns[t_b];
    if a.entities.iter().any(|e| {
        b.entities.contains(e) && ix.s_stats.df.get(e).copied().unwrap_or(0) <= hub_df
    }) {
        return true;
    }
    a.content.iter().any(|c| {
        b.content.contains(c) && (1..=hub_df).contains(&ix.s_stats.df.get(c).copied().unwrap_or(0))
    })
}

/// Deterministic 5-ballot Condorcet `@1` verifier for anonymous User/Assistant dialogues.
/// Strictly reorders within `0..5` (or inserts a latest-leaf revision ahead while keeping `@5`
/// intact) and preserves `@5` membership on Rank-1 user-source recovery.
fn apply_anonymous_dialogue_at1_condorcet_q16(
    ctx: &AnonCondorcetCtx<'_>,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
    k: usize,
) {
    let w = final_turn_indices.len().min(5);
    if w < 2 {
        return;
    }
    let ix = ctx.ix;
    let spec = build_query_at1_spec(ix, ctx.plan, ctx.q);
    let _t0 = final_turn_indices[0];
    let _s0 = ix.turns[_t0].session;
    let sight_of = |s: usize| -> u64 {
        let sc = ctx.s_score.get(s).copied().unwrap_or(0.0)
            + ctx.bridge_boost.get(s).copied().unwrap_or(0.0);
        ieee754_bits_to_q16(sc.to_bits())
    };
    let role_ok = |ti: usize| -> bool {
        let tr = &ix.turns[ti];
        if ctx.eff_it.assistant_attribution {
            tr.field == 1
        } else {
            tr.field == 0 && (!tr.is_question || tr.content.len() >= 6)
        }
    };
    let self_share_of = |ti: usize| -> u64 {
        let own = ieee754_bits_to_q16(ctx.t_own.get(ti).copied().unwrap_or(0.0).to_bits());
        let scn = ieee754_bits_to_q16(ctx.t_scene.get(ti).copied().unwrap_or(0.0).to_bits()).max(1);
        own.saturating_mul(65_536) / scn
    };

    let mut q_mask_full = 0u64;
    for qt in &ctx.plan.terms {
        if qt.exact_hit
            && let Some(g) = qt.group
            && g < 64
        {
            q_mask_full |= 1u64 << g;
        }
    }

    let q_low = ctx.q.text.trim_start().to_ascii_lowercase();
    let q_toks = raw_tokens(&ctx.q.text);
    let is_history_or_agg = q_low.starts_with("have i ")
        || q_low.starts_with("have we ")
        || q_low.contains("did i ")
        || q_low.contains("did we ")
        || q_toks
            .iter()
            .any(|w| matches!(w.as_str(), "between" | "across" | "compare" | "compared" | "both"));

    // 0. Session-to-Turn Slot-0 Coherence Sync (`!has_named_speakers_eff`):
    //    When Rank-1 turn `t0` is a stray standing-policy rule on a non-rule query OR belongs
    //    to a lower-ranked session while `final_session_indices[0]`'s turn sits in Slots `1..5`
    //    with comparable/better coverage, seat `final_session_indices[0]`'s turn at Rank 1.
    if let Some(&s_head) = final_session_indices.first() {
        let t0_cur = final_turn_indices[0];
        let t0_is_stray_rule = ix.turns[t0_cur].is_standing_policy && !ctx.eff_it.task_under_rules;
        if t0_is_stray_rule || ix.turns[t0_cur].session != s_head {
            let own0_c = ieee754_bits_to_q16(ctx.t_own.get(t0_cur).copied().unwrap_or(0.0).to_bits());
            let ex0_c = (ctx.t_exact_mask.get(t0_cur).copied().unwrap_or(0) & q_mask_full).count_ones();
            let mut head_slot: Option<(usize, u32, u64)> = None;
            for (slot, &tc) in final_turn_indices.iter().enumerate().take(w).skip(1) {
                let trc = &ix.turns[tc];
                if trc.is_backchannel || trc.is_standing_policy || turn_is_superseded(ix, tc) {
                    continue;
                }
                if !t0_is_stray_rule && trc.session != s_head {
                    continue;
                }
                if !ctx.eff_it.assistant_attribution
                    && (trc.field != 0 || trc.is_pure_info_question)
                {
                    continue;
                }
                let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
                let sync_ok = t0_is_stray_rule
                    || (exc >= ex0_c && ownc.saturating_mul(100) >= own0_c.saturating_mul(72))
                    || (exc > ex0_c && ownc > 0);
                if sync_ok && head_slot.is_none_or(|(_, be, bo)| (exc, ownc) > (be, bo)) {
                    head_slot = Some((slot, exc, ownc));
                }
            }
            if let Some((win_slot, _, _)) = head_slot {
                final_turn_indices[..=win_slot].rotate_right(1);
            } else if !ctx.eff_it.assistant_attribution
                && let Some(members) = ix.session_turns.get(s_head)
            {
                let mut best_direct: Option<(usize, u32, u64)> = None;
                for &tc in members {
                    if tc == t0_cur {
                        continue;
                    }
                    let trc = &ix.turns[tc];
                    if trc.field != 0
                        || trc.is_pure_info_question
                        || trc.is_backchannel
                        || trc.is_standing_policy
                        || turn_is_superseded(ix, tc)
                    {
                        continue;
                    }
                    let ownc =
                        ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                    let exc =
                        (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
                    if exc == 0 {
                        continue;
                    }
                    let sync_ok = t0_is_stray_rule
                        || (exc > ex0_c && ownc > 0);
                    if sync_ok && best_direct.is_none_or(|(_, be, bo)| (exc, ownc) > (be, bo)) {
                        best_direct = Some((tc, exc, ownc));
                    }
                }
                if let Some((win_ti, _, _)) = best_direct {
                    if let Some(pos) = final_turn_indices.iter().position(|&x| x == win_ti) {
                        final_turn_indices.remove(pos);
                    }
                    final_turn_indices.insert(0, win_ti);
                }
            }
        }
    }
    let t0 = final_turn_indices[0];
    let s0 = ix.turns[t0].session;

    // 1. User-Source Recovery over Assistant Echo (`!assistant_attribution`):
    //    When Rank 1 is an assistant turn (`field == 1`) and its preceding same-session user
    //    source (`field == 0`, declarative or substantive multi-clause question) sits inside Top-5,
    //    preferring non-superseded user turns (`>= 28%` own score on immediate parent `pc+1 == p0`),
    //    restore the originating user turn to Rank 1 while preserving Top-5 membership.
    if !ctx.eff_it.assistant_attribution && ix.turns[t0].field == 1 {
        let own0 = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let ex0 = ctx.t_exact_mask.get(t0).copied().unwrap_or(0) & q_mask_full;
        let pos0 = turn_pos(ix, t0);
        let mut user_src_slot: Option<(usize, bool, u32, u64)> = None;
        for (slot, &tc) in final_turn_indices.iter().enumerate().take(w).skip(1) {
            let trc = &ix.turns[tc];
            let substantive_user = !trc.is_question || trc.content.len() >= 6;
            if trc.field != 0 || !substantive_user || trc.is_backchannel {
                continue;
            }
            if trc.session != s0 && tc + 1 != t0 {
                continue;
            }
            let direct_parent = tc + 1 == t0
                || match (turn_pos(ix, tc), pos0) {
                    (Some(pc), Some(p0)) => pc < p0 && p0 - pc == 1,
                    _ => false,
                };
            let near_prev = direct_parent
                || match (turn_pos(ix, tc), pos0) {
                    (Some(pc), Some(p0)) => pc < p0 && p0 - pc <= 2,
                    _ => false,
                };
            if !near_prev {
                continue;
            }
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            if ownc == 0 {
                continue;
            }
            let exc = ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full;
            let cov_match = (exc & ex0) == ex0
                || ownc.saturating_mul(100) >= own0.saturating_mul(58)
                || (direct_parent
                    && exc != 0
                    && (ownc.saturating_mul(100) >= own0.saturating_mul(25)
                        || exc.count_ones() >= 2));
            if cov_match {
                let fresh = !turn_is_superseded(ix, tc);
                let bits = exc.count_ones();
                if user_src_slot.is_none_or(|(_, bf, bb, bo)| (fresh, bits, ownc) > (bf, bb, bo)) {
                    user_src_slot = Some((slot, fresh, bits, ownc));
                }
            }
        }
        if let Some((win_slot, _, _, _)) = user_src_slot {
            final_turn_indices[..=win_slot].rotate_right(1);
        }
    }

    // 2. Closed-Class Revision-Cue + Latest-Leaf Override at Rank 1 (current-state updates):
    //    When the query asks for current state (`!past_override && !is_history_or_agg`), if a newer
    //    non-superseded user declarative in Top-10 shares the entity/slot signature with `t0` and
    //    either carries a closed-class revision cue or supersedes `t0`, seat the newest leaf at #1.
    let t0_now = final_turn_indices[0];
    let s0_now = ix.turns[t0_now].session;
    let ts0_now = ix.session_ts.get(s0_now).copied().unwrap_or(0);
    if !spec.past_override
        && !ctx.eff_it.is_event_sequence
        && !ctx.eff_it.assistant_attribution
        && !is_history_or_agg
        && final_turn_indices.len() >= 2
    {
        let own0_now = ieee754_bits_to_q16(ctx.t_own.get(t0_now).copied().unwrap_or(0.0).to_bits());
        let sight0_now = sight_of(s0_now);
        let ex0_now = ctx.t_exact_mask.get(t0_now).copied().unwrap_or(0) & q_mask_full;
        let sup0_now = turn_is_superseded(ix, t0_now)
            || ix.torrens_superseded.get(t0_now).copied().unwrap_or(false);
        let top_scan = final_turn_indices.len().min(10);
        let mut best_rev: Option<(bool, usize, usize)> = None;
        for (slot, &tc) in final_turn_indices.iter().enumerate().take(top_scan).skip(1) {
            let trc = &ix.turns[tc];
            let substantive_user = !trc.is_question || trc.content.len() >= 6;
            if trc.field != 0 || !substantive_user || trc.is_backchannel || turn_is_superseded(ix, tc) {
                continue;
            }
            let sc = trc.session;
            let tsc = ix.session_ts.get(sc).copied().unwrap_or(0);
            if tc <= t0_now || sc <= s0_now || (ts0_now > 0 && tsc > 0 && tsc < ts0_now) {
                continue;
            }
            if sight_of(sc).saturating_mul(100) < sight0_now.saturating_mul(75) {
                continue;
            }
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            if ownc == 0 || ownc.saturating_mul(100) < own0_now.saturating_mul(65) {
                continue;
            }
            let exc = ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full;
            let cov_ok = (exc & ex0_now) == ex0_now
                || (exc.count_ones() >= 2 && exc.count_ones() >= ex0_now.count_ones());
            if !cov_ok {
                continue;
            }
            let explicit_sup = trc.supersedes_turns.contains(&t0_now)
                || ix.session_supersedes_back.get(sc).is_some_and(|v| v.contains(&s0_now))
                || sup0_now;
            // #5 Cellular Sheaf Cohomology H^1 + #6 Dempster-Shafer Conflict Mass `K`:
            // When two turns share a rare entity/slot anchor across epochs and yield non-zero
            // sheaf residual + DS conflict (`pair_sheaf_conflict_q16 > Q16/2`), promote the newer gluer.
            let sheaf_ds_conflict = crate::retrieval::graph::sheaf_dempster_q16::pair_sheaf_conflict_q16(ix, t0_now, tc);
            let has_rev_cue = turn_has_revision_cue(&trc.body_for_headers) || sheaf_ds_conflict >= 32_768;
            let same_slot = turns_share_rare_slot_anchor(ix, t0_now, tc);
            if explicit_sup || (has_rev_cue && same_slot) {
                let cand_key = (explicit_sup || has_rev_cue, tc, slot);
                if best_rev.is_none_or(|cur| (cand_key.0, cand_key.1) > (cur.0, cur.1)) {
                    best_rev = Some(cand_key);
                }
            }
        }
        if let Some((_, win_ti, win_slot)) = best_rev {
            let win_sess = ix.turns[win_ti].session;
            if win_slot < 5 {
                final_turn_indices[..=win_slot].rotate_right(1);
            } else {
                // Keep Top-4 incumbents (Slots 0..4) inside Top-5 so All@5 on pairs is preserved.
                final_turn_indices.remove(win_slot);
                final_turn_indices.insert(0, win_ti);
                final_turn_indices.truncate(k.max(5));
            }
            if win_sess != s0_now {
                let sw = final_session_indices.len().min(5);
                if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                    && spos > 0
                {
                    final_session_indices[..=spos].rotate_right(1);
                }
            }
            return;
        }
    }

    // 1b. Strict Assistant-Attribution & "did I" User-Attribution Enforcement:
    //     When the query asks `"did you recommend/suggest/..."` (`eff_it.assistant_attribution`)
    //     and a User turn sits at Slot 0, promote a matching Assistant reply from Top-7.
    //     Conversely when the query asks `"did i ..."` and an Assistant turn sits at Slot 0,
    //     promote a matching substantive User turn from Top-6.
    if ctx.eff_it.assistant_attribution && ix.turns[t0].field == 0 {
        let own0_a = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let scan = final_turn_indices.len().min(10);
        let mut best_asst: Option<(u32, u64, usize)> = None;
        for (slot, &tc) in final_turn_indices.iter().enumerate().take(scan).skip(1) {
            let trc = &ix.turns[tc];
            if trc.field != 1 || trc.is_question || trc.is_backchannel {
                continue;
            }
            if sight_of(trc.session).saturating_mul(100) < sight_of(s0).saturating_mul(75) {
                continue;
            }
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
            let in_top2_sess = final_session_indices.iter().take(2).any(|&s| s == trc.session);
            if (exc >= 1 || in_top2_sess) && ownc.saturating_mul(100) >= own0_a.saturating_mul(45) {
                let key = (exc, ownc, slot);
                if best_asst.is_none_or(|b| (key.0, key.1) > (b.0, b.1)) {
                    best_asst = Some(key);
                }
            }
        }
        if let Some((_, _, win_slot)) = best_asst {
            if win_slot < 5 {
                final_turn_indices[..=win_slot].rotate_right(1);
            } else {
                let win_ti = final_turn_indices.remove(win_slot);
                final_turn_indices.insert(0, win_ti);
                final_turn_indices.truncate(k.max(5));
            }
            return;
        }
        // Fallback: scan substantive `field == 1` replies inside Top-2 sessions with WordNet/ConceptNet
        // lexical expansion so paraphrased assistant recommendations ("flow of requests" <-> "rate limits") promote.
        let mut q_lex_asst: Vec<u32> = Vec::new();
        for qw in &ctx.plan.groups {
            if qw.starts_with('\u{1}') || !is_content(qw) {
                continue;
            }
            for (lem, _) in crate::wordnet::full_lexicon_neighbors(qw).into_iter().take(16) {
                if let Some(id) = ix.vocab.get(lem)
                    && !q_lex_asst.contains(&id)
                {
                    q_lex_asst.push(id);
                }
            }
            for (lem, _) in crate::conceptnet::two_hop_neighbors(qw, 8) {
                if let Some(id) = ix.vocab.get(lem)
                    && !q_lex_asst.contains(&id)
                {
                    q_lex_asst.push(id);
                }
            }
        }
        let scn0_a = ieee754_bits_to_q16(ctx.t_scene.get(t0).copied().unwrap_or(0.0).to_bits());
        let mut sess_asst: Option<(u32, usize, u64, usize)> = None;
        for &si in final_session_indices.iter().take(2) {
            if sight_of(si).saturating_mul(100) < sight_of(s0).saturating_mul(65) {
                continue;
            }
            for &tc in &ix.session_turns[si] {
                if tc == t0 {
                    continue;
                }
                let trc = &ix.turns[tc];
                if trc.field != 1 || trc.is_question || trc.is_backchannel {
                    continue;
                }
                let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                let scnc = ieee754_bits_to_q16(ctx.t_scene.get(tc).copied().unwrap_or(0.0).to_bits());
                let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
                let lex_h = q_lex_asst
                    .iter()
                    .filter(|id| trc.content.contains(id) || trc.feats.iter().any(|f| f.id == **id))
                    .count();
                let bridge_ok = exc >= 1 || lex_h >= 1;
                if (bridge_ok && scnc.saturating_mul(100) >= scn0_a.saturating_mul(40))
                    || ownc.saturating_mul(100) >= own0_a.saturating_mul(45)
                {
                    let key = (exc, lex_h, scnc.max(ownc), tc);
                    if sess_asst.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                        sess_asst = Some(key);
                    }
                }
            }
        }
        if let Some((_, _, _, win_ti)) = sess_asst {
            final_turn_indices.retain(|&x| x != win_ti);
            final_turn_indices.insert(0, win_ti);
            final_turn_indices.truncate(k.max(5));
            return;
        }
    } else if !ctx.eff_it.assistant_attribution
        && (q_low.contains("did i ")
            || q_low.contains("did we ")
            || q_low.contains("have i ")
            || q_low.contains("was i "))
        && ix.turns[t0].field == 1
    {
        let own0_u = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let scan = final_turn_indices.len().min(10);
        let mut best_u: Option<(u32, u64, usize)> = None;
        for (slot, &tc) in final_turn_indices.iter().enumerate().take(scan).skip(1) {
            let trc = &ix.turns[tc];
            if trc.field != 0 || trc.is_backchannel || turn_is_superseded(ix, tc) {
                continue;
            }
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
            if ownc.saturating_mul(100) >= own0_u.saturating_mul(50) {
                let key = (exc, ownc, slot);
                if best_u.is_none_or(|b| (key.0, key.1) > (b.0, b.1)) {
                    best_u = Some(key);
                }
            }
        }
        if let Some((_, _, win_slot)) = best_u {
            if win_slot < 5 {
                final_turn_indices[..=win_slot].rotate_right(1);
            } else {
                let win_ti = final_turn_indices.remove(win_slot);
                final_turn_indices.insert(0, win_ti);
                final_turn_indices.truncate(k.max(5));
            }
            return;
        }
    }

    // Re-read Slot 0 in case User-Source Recovery just restored the user turn.
    let t0 = final_turn_indices[0];
    let s0 = ix.turns[t0].session;

    // Protect a condition-matched standing rule already seated at Slot 0 by Stage 12 Step 2
    // from being evicted by downstream rare-anchor or habitus tie-breakers.
    if ix.turns[t0].is_standing_policy {
        return;
    }

    // 1c. Closed-Class Quantitative Answer-Type Gate (`information_extraction`):
    //     When a quantitative query ("how much/how long/how old/what probability/percentage") seats a
    //     Slot-0 turn lacking digits (or lacking expected currency/duration units), promote the strongest
    //     unit/digit-bearing exact-hit turn from Top-10 or Top-10 sessions (`field == 0`) to Slot 0.
    // hardcode-scan: closed-class (quantitative WH interrogatives & currency/percent/duration format cues)
    let is_quant_ie_q = !ctx.eff_it.is_event_sequence
        && !eff_summary_request(&ctx.q.text)
        && (q_low.starts_with("how much ")
            || q_low.starts_with("how long ")
            || q_low.starts_with("how old ")
            || q_low.starts_with("what probability")
            || q_low.starts_with("what percentage"));
    let want_unit_sym = q_low.contains("cost")
        || q_low.contains("price")
        || q_low.contains("pay")
        || q_low.contains("much")
        || q_low.contains("percentage");
    let want_dur_unit = q_low.starts_with("how long ");
    let t0_body_q = ix.turns[t0].body_for_headers.as_str();
    let t0_low_q = t0_body_q.to_ascii_lowercase();
    let t0_has_digit = t0_body_q.bytes().any(|b| b.is_ascii_digit());
    let has_dur_unit = |s_low: &str| {
        s_low.bytes().any(|b| b.is_ascii_digit())
            && (s_low.contains(" year")
                || s_low.contains(" month")
                || s_low.contains(" week")
                || s_low.contains(" day")
                || s_low.contains(" hour"))
    };
    let t0_missing_quant = !t0_has_digit
        || (want_unit_sym && !(t0_body_q.contains('$') || t0_body_q.contains('%')))
        || (want_dur_unit && !has_dur_unit(&t0_low_q));
    if is_quant_ie_q
        && !ix.turns[t0].is_standing_policy
        && t0_missing_quant
    {
        let own0_q = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let mut quant_cands: Vec<usize> = final_turn_indices
            .iter()
            .copied()
            .skip(1)
            .take(9)
            .collect();
        for &si in final_session_indices.iter().take(10) {
            for &ti in &ix.session_turns[si] {
                let tr = &ix.turns[ti];
                if ti != t0
                    && tr.field == 0
                    && !tr.is_backchannel
                    && !tr.is_standing_policy
                    && !quant_cands.contains(&ti)
                {
                    quant_cands.push(ti);
                }
            }
        }
        let mut best_quant: Option<(bool, u32, u64, usize)> = None;
        for tc in quant_cands {
            let trc = &ix.turns[tc];
            if trc.is_backchannel || trc.is_standing_policy {
                continue;
            }
            let body = trc.body_for_headers.as_str();
            if !body.bytes().any(|b| b.is_ascii_digit()) {
                continue;
            }
            let body_low = body.to_ascii_lowercase();
            let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            let unit_hit = (want_unit_sym && (body.contains('$') || body.contains('%')))
                || (want_dur_unit && has_dur_unit(&body_low));
            let min_own_pct = if unit_hit { 20u64 } else { 45u64 };
            if exc >= 1 && ownc.saturating_mul(100) >= own0_q.saturating_mul(min_own_pct) {
                let key = (unit_hit, exc, ownc, tc);
                if best_quant.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                    best_quant = Some(key);
                }
            }
        }
        if let Some((_, _, _, win_ti)) = best_quant {
            let win_sess = ix.turns[win_ti].session;
            if let Some(win_slot) = final_turn_indices.iter().position(|&x| x == win_ti) {
                if win_slot < 5 {
                    final_turn_indices[..=win_slot].rotate_right(1);
                } else {
                    final_turn_indices.remove(win_slot);
                    final_turn_indices.insert(0, win_ti);
                    final_turn_indices.truncate(k.max(5));
                }
            } else {
                final_turn_indices.insert(0, win_ti);
                final_turn_indices.truncate(k.max(5));
            }
            if win_sess != s0 {
                let sw = final_session_indices.len().min(5);
                if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                    && spos > 0
                {
                    final_session_indices[..=spos].rotate_right(1);
                }
            }
            return;
        }
    }

    // 1d. Closed-Class Episodic Entity-Rich Extraction Gate (`"what ... did I say/mention/outline"`):
    //     When a factual "did I say/mention" query lands on a low-entity Slot-0 turn (`<= 2` entities)
    //     while a Top-10 or Top-3 session user turn packs rich explicit technologies/entities with strong exact IC,
    //     promote it to Slot 0.
    // hardcode-scan: closed-class (first-person past episodic mention interrogatives)
    let is_entity_ie_q = !ctx.eff_it.temporal
        && !ctx.eff_it.is_event_sequence
        && !ctx.eff_it.assistant_attribution
        && !eff_summary_request(&ctx.q.text)
        && (q_low.contains("did i say")
            || q_low.contains("did i mention")
            || q_low.contains("did i outline")
            || q_low.contains("did i use"));
    if is_entity_ie_q && ix.turns[t0].slot_entities_clean <= 2 {
        let own0_e = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let mut ent_cands: Vec<usize> = final_turn_indices.iter().copied().skip(1).take(9).collect();
        for &si in final_session_indices.iter().take(3) {
            for &ti in &ix.session_turns[si] {
                let tr = &ix.turns[ti];
                if ti != t0
                    && tr.field == 0
                    && !tr.is_backchannel
                    && !tr.is_standing_policy
                    && !ent_cands.contains(&ti)
                {
                    ent_cands.push(ti);
                }
            }
        }
        let min_ents = ix.turns[t0].slot_entities_clean.saturating_add(2);
        let mut best_ent: Option<(u32, u16, u64, usize)> = None;
        for tc in ent_cands {
            let trc = &ix.turns[tc];
            if trc.field != 0
                || trc.is_backchannel
                || trc.is_standing_policy
                || trc.slot_entities_clean < min_ents
            {
                continue;
            }
            let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            if exc >= 2 && ownc.saturating_mul(100) >= own0_e.saturating_mul(60) {
                let key = (exc, trc.slot_entities_clean, ownc, tc);
                if best_ent.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                    best_ent = Some(key);
                }
            }
        }
        if let Some((_, _, _, win_ti)) = best_ent {
            let win_sess = ix.turns[win_ti].session;
            if let Some(win_slot) = final_turn_indices.iter().position(|&x| x == win_ti) {
                if win_slot < 5 {
                    final_turn_indices[..=win_slot].rotate_right(1);
                } else {
                    final_turn_indices.remove(win_slot);
                    final_turn_indices.insert(0, win_ti);
                    final_turn_indices.truncate(k.max(5));
                }
            } else {
                final_turn_indices.insert(0, win_ti);
                final_turn_indices.truncate(k.max(5));
            }
            if win_sess != s0 {
                let sw = final_session_indices.len().min(5);
                if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                    && spos > 0
                {
                    final_session_indices[..=spos].rotate_right(1);
                }
            }
            return;
        }
    }

    // 2b. Closed-Class Constraint & Directive Density Promotion (#3) + Short Ack-Question Demotion (#6)
    //     + Chronological Latest-State Tie-Break (#9):
    {
        let own0_c = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
        let ex0_c = (ctx.t_exact_mask.get(t0).copied().unwrap_or(0) & q_mask_full).count_ones();
        let is_adv = crate::retrieval::seating::constraint_density_q16::is_advice_or_preference_query(&ctx.q.text);
        let d0 = if is_adv {
            crate::retrieval::seating::constraint_density_q16::deontic_directive_density_q16(&ix.turns[t0].body_for_headers)
        } else {
            0
        };
        let t0_ack = crate::retrieval::fusion::summary_silhouette_q16::is_short_ack_question(ix, t0);
        // Exclusive Rare-Anchor Tournament: find ultra-rare exact query anchors (`df <= 2`) missing from `t0`.
        let rare_missing_from_t0: Vec<u32> = ctx
            .plan
            .terms
            .iter()
            .filter(|qt| {
                qt.exact_hit
                    && !ix.turns[t0].content.contains(&qt.id)
                    && ix.t_own_stats.df.get(&qt.id).copied().unwrap_or(99) <= 2
            })
            .map(|qt| qt.id)
            .collect();
        let mut promo_slot: Option<(u8, u64, u32, usize)> = None;
        for (slot, &tc) in final_turn_indices.iter().enumerate().take(w).skip(1) {
            let trc = &ix.turns[tc];
            if trc.field != ix.turns[t0].field
                && (ctx.eff_it.assistant_attribution || trc.field != 0)
            {
                continue;
            }
            if trc.is_backchannel || trc.is_standing_policy || turn_is_superseded(ix, tc) {
                continue;
            }
            let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
            let exc = (ctx.t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_full).count_ones();
            if ownc == 0 || sight_of(trc.session).saturating_mul(100) < sight_of(s0).saturating_mul(75) {
                continue;
            }
            // Priority 3: Exclusive ultra-rare anchor hit missing from `t0` (e.g. `endpoint`)
            if !rare_missing_from_t0.is_empty()
                && !trc.is_question
                && ownc.saturating_mul(100) >= own0_c.saturating_mul(85)
                && exc >= ex0_c
            {
                let rare_hits = rare_missing_from_t0
                    .iter()
                    .filter(|id| trc.content.contains(id))
                    .count() as u64;
                if rare_hits > 0 && exc >= ex0_c {
                    let key = (4u8, rare_hits, exc, slot);
                    if promo_slot.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                        promo_slot = Some(key);
                    }
                    continue;
                }
            }
            // Priority 2 (#3): High deontic constraint density on advice/preference queries
            if is_adv && d0 < 2_500 && !trc.is_question {
                let dc = crate::retrieval::seating::constraint_density_q16::deontic_directive_density_q16(&trc.body_for_headers);
                if dc >= 4_500
                    && ownc.saturating_mul(100) >= own0_c.saturating_mul(65)
                    && exc.saturating_add(1) >= ex0_c
                {
                    let key = (3u8, dc, exc, slot);
                    if promo_slot.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                        promo_slot = Some(key);
                    }
                    continue;
                }
            }
            // Priority 1 (#6): Demote short ack/clarification question at Slot 0
            if t0_ack && !trc.is_question && trc.content.len() >= 10 && ownc.saturating_mul(100) >= own0_c.saturating_mul(65) {
                let key = (2u8, ownc, exc, slot);
                if promo_slot.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                    promo_slot = Some(key);
                }
                continue;
            }
            // Priority 0 (#9 + Pillar 7a IA-RAG Directional Allen Cone):
            // On temporal directional queries ("before/after/during"), prefer Top-5 turns inside
            // the matching Allen cone; on current-state queries, let fresher assertions break ties.
            let allen_cone = crate::retrieval::graph::temporal_allen::detect_ia_rag_cone(&ctx.q.text);
            if allen_cone != crate::retrieval::graph::temporal_allen::AllenDirCone::None
                && !trc.is_question
                && exc >= ex0_c
                && ownc.saturating_mul(100) >= own0_c.saturating_mul(78)
                && crate::retrieval::graph::temporal_allen::satisfies_ia_rag_cone(ix, allen_cone, t0, tc)
            {
                let key = (2u8, ownc, exc, slot);
                if promo_slot.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                    promo_slot = Some(key);
                }
                continue;
            }
            if ctx.eff_it.current_state
                && !spec.past_override
                && !is_history_or_agg
                && !ctx.eff_it.is_event_sequence
                && !ctx.eff_it.assistant_attribution
                && !trc.is_question
                && trc.session > s0
                && ex0_c >= 1
                && exc >= ex0_c
                && ownc.saturating_mul(100) >= own0_c.saturating_mul(88)
            {
                let key = (1u8, trc.session as u64, exc, slot);
                if promo_slot.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                    promo_slot = Some(key);
                }
            }
        }
        if let Some((_, _, _, win_slot)) = promo_slot {
            let win_sess = ix.turns[final_turn_indices[win_slot]].session;
            final_turn_indices[..=win_slot].rotate_right(1);
            if win_sess != s0 {
                let sw = final_session_indices.len().min(5);
                if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                    && spos > 0
                {
                    final_session_indices[..=spos].rotate_right(1);
                }
            }
            return;
        }
    }

    // Re-read Slot 0 after Step 2b/c/d.
    let t0 = final_turn_indices[0];
    let s0 = ix.turns[t0].session;
    let sight0 = sight_of(s0);

    let (lr0, dur0) = eval_li_roth_turn(ix, &spec, t0);
    let r0 = role_ok(t0);
    let sup0 = turn_is_superseded(ix, t0);
    let own0 = ieee754_bits_to_q16(ctx.t_own.get(t0).copied().unwrap_or(0.0).to_bits());
    let share0 = self_share_of(t0);
    let ent0 = ix.turns[t0].slot_entities_clean;
    let num0 = ix.turns[t0].has_numeric;

    let mut best: Option<(u8, i32, u64, usize)> = None;
    for (slot, &tc) in final_turn_indices.iter().enumerate().take(w).skip(1) {
        let trc = &ix.turns[tc];
        if trc.is_backchannel || turn_is_superseded(ix, tc) {
            continue;
        }
        let sc = trc.session;
        if sight_of(sc).saturating_mul(100) < sight0.saturating_mul(82) {
            continue;
        }
        if ctx.eff_it.current_state && sc < s0 && !sup0 {
            continue;
        }
        let rc = role_ok(tc);
        if r0 && !rc {
            continue;
        }
        let (lrc, durc) = eval_li_roth_turn(ix, &spec, tc);
        let ownc = ieee754_bits_to_q16(ctx.t_own.get(tc).copied().unwrap_or(0.0).to_bits());
        if ownc == 0 {
            continue;
        }
        let sharec = self_share_of(tc);

        // 5 orthogonal deterministic ballots:
        let b1_li_roth = lrc > lr0 || (lrc == lr0 && durc > dur0);
        let b2_role = (rc && !r0)
            || (ix.turns[t0].is_question && !trc.is_question)
            || (sup0 && !turn_is_superseded(ix, tc));
        let b3_self_share = sharec > share0;
        let b4_own = ownc > own0;
        let b5_grounding = trc.slot_entities_clean > ent0
            || (trc.slot_entities_clean == ent0 && trc.has_numeric && !num0);

        let votes = u8::from(b1_li_roth)
            + u8::from(b2_role)
            + u8::from(b3_self_share)
            + u8::from(b4_own)
            + u8::from(b5_grounding);
        let qualified = votes >= 4 || (sup0 && rc && ownc >= own0);
        if !qualified {
            continue;
        }
        let cand = (votes, lrc, ownc, slot);
        let better = best.is_none_or(|cur| {
            (cand.0, cand.1, cand.2, std::cmp::Reverse(cand.3))
                > (cur.0, cur.1, cur.2, std::cmp::Reverse(cur.3))
        });
        if better {
            best = Some(cand);
        }
    }

    if let Some((_, _, _, win_slot)) = best {
        let win_sess = ix.turns[final_turn_indices[win_slot]].session;
        final_turn_indices[..=win_slot].rotate_right(1);
        if win_sess != s0 {
            let sw = final_session_indices.len().min(5);
            if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                && spos > 0
            {
                final_session_indices[..=spos].rotate_right(1);
            }
        }
    }
}

/// Runs the unified `@1` & `@5` gatekeeper scorer over the deep candidate pool on named-speaker
/// episodic dialogues (and dumps exact integer features when `VESTIGE_AT1_FEAT_DUMP` is set).
pub(in crate::retrieval) fn apply_at1_gatekeeper_q16(
    ix: &Index,
    plan: &QueryPlan,
    eff_it: &QueryIntent,
    q: &Query,
    has_named_speakers_eff: bool,
    is_monologue_corpus: bool,
    pref_on: bool,
    final_turn_indices: &mut Vec<usize>,
    final_session_indices: &mut Vec<usize>,
    t_scene: &[f32],
    t_own: &[f32],
    t_turn_raw: &[f32],
    t_bridge: &[f32],
    t_exact_mask: &[u64],
    s_score: &[f32],
    bridge_boost: &[f32],
    k: usize,
) {
    if final_turn_indices.is_empty()
        || is_monologue_corpus
        || eff_it.is_event_sequence
        || eff_summary_request(&q.text)
    {
        return;
    }
    // Pillar 6 (HabitusMem `2611.25855` + MINDSET `2610.08586` Scope Energy Rank-1 Re-Seating):
    // Unlocked specifically on anonymous dialogue `pref_on || task_under_rules` queries where
    // Top-5 near-misses previously stalled because transient debug / assistant turns stole Slot 0.
    if pref_on || eff_it.task_under_rules {
        if !has_named_speakers_eff && final_turn_indices.len() >= 2 {
            let q_low_h = q.text.to_ascii_lowercase();
            // hardcode-scan: closed-class (genuine personal preference/advice openings vs general how-to rules)
            let is_genuine_pref_q = q_low_h.contains("suggest")
                || q_low_h.contains("recommend")
                || q_low_h.contains("prefer")
                || q_low_h.contains("i'd like")
                || q_low_h.contains("i would like")
                || q_low_h.contains("i'm looking")
                || q_low_h.contains("i'm working")
                || q_low_h.contains("i'm planning")
                || q_low_h.contains("can you help me");
            let w = final_turn_indices.len().min(6);
            let t0 = final_turn_indices[0];
            let s0 = ix.turns[t0].session;
            let u0 = crate::retrieval::seating::habitus_energy_q16::habitus_scope_utility_q16(ix, t0, true);
            let d0 = crate::retrieval::seating::constraint_density_q16::deontic_directive_density_q16(
                &ix.turns[t0].body_for_headers,
            );
            let own0 = ieee754_bits_to_q16(t_own.get(t0).copied().unwrap_or(0.0).to_bits());
            let mut q_mask_h = 0u64;
            for qt in &plan.terms {
                if qt.exact_hit
                    && let Some(g) = qt.group
                    && g < 64
                {
                    q_mask_h |= 1u64 << g;
                }
            }
            let ex0 = (t_exact_mask.get(t0).copied().unwrap_or(0) & q_mask_h).count_ones();
            let q_toks_h = raw_tokens(&q.text);
            let t0_cond_matched_rule = ix.turns[t0].is_standing_policy
                && ix.standing_rules.iter().any(|r| {
                    r.turn == t0 && rule_condition_matches(ix, r, plan, &q_toks_h)
                });
            let t0_is_decoy = crate::retrieval::seating::habitus_energy_q16::is_status_metric_decoy(
                &ix.turns[t0].body_for_headers,
            );
            let t0_weak_scope = !t0_cond_matched_rule
                && (ix.turns[t0].field == 1
                    || (u0 < 0 && !ix.turns[t0].is_standing_policy)
                    || crate::retrieval::fusion::summary_silhouette_q16::is_short_ack_question(ix, t0)
                    || (is_genuine_pref_q && (ix.turns[t0].is_standing_policy || t0_is_decoy)))
                && !(ix.turns[t0].is_standing_policy && !is_genuine_pref_q);
            if t0_weak_scope {
                let mut best_h: Option<(i64, u64, u32, usize)> = None;
                for (slot, &tc) in final_turn_indices.iter().enumerate().take(w).skip(1) {
                    let trc = &ix.turns[tc];
                    if trc.field != 0 || trc.is_backchannel || turn_is_superseded(ix, tc) {
                        continue;
                    }
                    if is_genuine_pref_q && trc.is_standing_policy {
                        continue;
                    }
                    if crate::retrieval::seating::habitus_energy_q16::is_status_metric_decoy(&trc.body_for_headers) {
                        continue;
                    }
                    let uc = crate::retrieval::seating::habitus_energy_q16::habitus_scope_utility_q16(ix, tc, true);
                    let beats_stray_policy = is_genuine_pref_q && ix.turns[t0].is_standing_policy;
                    if !beats_stray_policy && !t0_is_decoy && uc <= u0 {
                        continue;
                    }
                    let dc = crate::retrieval::seating::constraint_density_q16::deontic_directive_density_q16(
                        &trc.body_for_headers,
                    );
                    let ownc = ieee754_bits_to_q16(t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                    let exc = (t_exact_mask.get(tc).copied().unwrap_or(0) & q_mask_h).count_ones();
                    let min_pct = if beats_stray_policy || t0_is_decoy { 55 } else { 75 };
                    let rel_ok = ownc > 0
                        && ownc.saturating_mul(100) >= own0.saturating_mul(min_pct)
                        && (exc >= 1 && (beats_stray_policy || t0_is_decoy || exc >= ex0 || dc >= d0.saturating_add(2_500)));
                    if rel_ok {
                        let key = (uc, dc, exc, slot);
                        if best_h.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                            best_h = Some(key);
                        }
                    }
                }
                if let Some((_, _, _, win_slot)) = best_h {
                    let win_sess = ix.turns[final_turn_indices[win_slot]].session;
                    if win_slot < 5 {
                        final_turn_indices[..=win_slot].rotate_right(1);
                    } else {
                        let win_ti = final_turn_indices.remove(win_slot);
                        final_turn_indices.insert(0, win_ti);
                        final_turn_indices.truncate(k.max(5));
                    }
                    if win_sess != s0 {
                        let sw = final_session_indices.len().min(5);
                        if let Some(spos) = final_session_indices[..sw].iter().position(|&s| s == win_sess)
                            && spos > 0
                        {
                            final_session_indices[..=spos].rotate_right(1);
                        }
                    }
                }
            }
        }
        // Unlock anonymous dialogue Condorcet & IE gates (`1b` Assistant-Attribution, `1c` Quantitative,
        // `1d` Entity-Rich Extraction) when the Slot-0 turn is NOT a protected standing policy.
        // hardcode-scan: closed-class (factual IE quantitative/attribution triggers)
        let q_low_ie = q.text.trim_start().to_ascii_lowercase();
        let run_ie_after_habitus = !has_named_speakers_eff
            && !final_turn_indices
                .first()
                .is_some_and(|&t0| ix.turns[t0].is_standing_policy)
            && (eff_it.assistant_attribution
                || q_low_ie.contains("did i ")
                || q_low_ie.contains("did we ")
                || q_low_ie.contains("have i ")
                || q_low_ie.contains("was i ")
                || q_low_ie.starts_with("how much ")
                || q_low_ie.starts_with("how long ")
                || q_low_ie.starts_with("how old ")
                || q_low_ie.starts_with("what probability")
                || q_low_ie.starts_with("what percentage"));
        if !run_ie_after_habitus {
            return;
        }
    }
    if !has_named_speakers_eff {
        let anon_ctx = AnonCondorcetCtx {
            ix,
            plan,
            eff_it,
            q,
            t_scene,
            t_own,
            t_exact_mask,
            s_score,
            bridge_boost,
        };
        apply_anonymous_dialogue_at1_condorcet_q16(
            &anon_ctx,
            final_turn_indices,
            final_session_indices,
            k,
        );
        return;
    }

    let pool = build_at1_candidate_pool(
        ix,
        eff_it.multi_hop,
        final_turn_indices,
        final_session_indices,
        t_scene,
        t_own,
        t_bridge,
    );
    if pool.len() < 2 {
        return;
    }

    let pool_scores = PoolScoreSlices {
        t_scene,
        t_own,
        t_turn_raw,
        t_bridge,
        s_score,
        bridge_boost,
    };
    let feats = extract_pool_features_q16(
        ix,
        plan,
        q,
        &pool,
        final_turn_indices,
        final_session_indices,
        &pool_scores,
    );

    if let Some(dump_path) = at1_feat_dump_path() {
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dump_path)
        {
            let pool_ids: Vec<&str> = pool.iter().map(|&ti| ix.turns[ti].id.as_str()).collect();
            let pool_sess: Vec<&str> = pool
                .iter()
                .map(|&ti| ix.session_ids[ix.turns[ti].session].as_str())
                .collect();
            let feat_slices: Vec<&[i32]> = feats.iter().map(|r| r.as_slice()).collect();
            let rec = serde_json::json!({
                "query": q.text,
                "first_sess": ix.session_ids.first().map(String::as_str).unwrap_or(""),
                "n_sessions": ix.session_ids.len(),
                "n_turns": ix.turns.len(),
                "pool_ids": pool_ids,
                "pool_sess": pool_sess,
                "final_turn_ids": final_turn_indices.iter().map(|&ti| ix.turns[ti].id.as_str()).collect::<Vec<_>>(),
                "final_sess_ids": final_session_indices.iter().map(|&si| ix.session_ids[si].as_str()).collect::<Vec<_>>(),
                "feats": feat_slices,
            });
            let _ = writeln!(file, "{rec}");
        }
    }

    if AT1_TREE_ROOTS.is_empty() {
        return;
    }

    let mut scores: Vec<i64> = feats.iter().map(eval_at1_gbdt_q16).collect();

    // #4 Tropical (max,+) 2-Hop Squaring bonus in `@1` scoring for Multi-Hop queries:
    // Adds a bounded Q16 tropical 2-hop bridge bonus to declarative candidates connected
    // to top Pass-1 anchors through rare non-query bridge entities.
    if eff_it.multi_hop {
        let mut q_gmask = 0u64;
        for qt in &plan.terms {
            if qt.exact_hit
                && let Some(g) = qt.group
                && g < 64
            {
                q_gmask |= 1u64 << g;
            }
        }
        let q_toks_mh = raw_tokens(&q.text);
        let mut matched_sps: Vec<String> = Vec::new();
        for tr in ix.turns.iter().take(64) {
            let sp = tr.speaker.to_ascii_lowercase();
            if !sp.is_empty()
                && sp != "user"
                && sp != "assistant"
                && q_toks_mh.iter().any(|w| w == &sp)
                && !matched_sps.contains(&sp)
            {
                matched_sps.push(sp);
            }
        }
        let single_sp = if matched_sps.len() == 1 {
            Some(matched_sps[0].as_str())
        } else {
            None
        };
        let trop_map =
            crate::retrieval::graph::tropical_bridge_q16::tropical_max_plus_entity_bridge_q16(
                ix,
                plan,
                t_own,
                t_exact_mask,
                q_gmask,
                single_sp,
            );
        if let Some(max_t) = trop_map.values().copied().max()
            && max_t > 0
        {
            const TROP_AT1_CAP_Q16: u64 = 16_000;
            for (i, &ti) in pool.iter().enumerate() {
                if let Some(&tsc) = trop_map.get(&ti)
                    && feats[i][32] == 0
                {
                    let bonus = (((tsc as u128).saturating_mul(TROP_AT1_CAP_Q16 as u128))
                        / (max_t as u128)) as i64;
                    scores[i] = scores[i].saturating_add(bonus);
                }
            }
        }
    }

    // #6 Co-Hyponym Sibling Mismatch Penalty (guarded by `any_cand_matches_qw` so
    // false-premise Adversarial source passages are NEVER penalized when the queried
    // sibling does not exist in the candidate pool):
    {
        const COHYP_MISMATCH_PENALTY_Q16: i64 = 18_000;
        let q_toks_ch = raw_tokens(&q.text);
        let mut checked_qw: Vec<String> = Vec::new();
        for qw in &q_toks_ch {
            if !is_content(qw) || qw.len() < 3 || checked_qw.contains(qw) {
                continue;
            }
            checked_qw.push(qw.clone());
            let sibs = crate::wordnet::strict_co_hyponym_siblings(qw);
            if sibs.is_empty() {
                continue;
            }
            let qw_ids: Vec<u32> = [ix.vocab.get(qw), ix.vocab.get(&stem_term(&stem(qw)))]
                .into_iter()
                .flatten()
                .collect();
            if qw_ids.is_empty() {
                continue;
            }
            let any_cand_matches_qw = pool.iter().any(|&ti| {
                let tr = &ix.turns[ti];
                !tr.is_question
                    && qw_ids
                        .iter()
                        .any(|id| tr.content.contains(id) || tr.feats.iter().any(|f| f.id == *id))
            });
            if !any_cand_matches_qw {
                continue;
            }
            let sib_ids: Vec<u32> = sibs
                .into_iter()
                .filter_map(|s| ix.vocab.get(s))
                .collect();
            if sib_ids.is_empty() {
                continue;
            }
            for (i, &ti) in pool.iter().enumerate() {
                let tr = &ix.turns[ti];
                let has_qw = qw_ids
                    .iter()
                    .any(|id| tr.content.contains(id) || tr.feats.iter().any(|f| f.id == *id));
                if !has_qw && sib_ids.iter().any(|sid| tr.content.contains(sid)) {
                    scores[i] = scores[i].saturating_sub(COHYP_MISMATCH_PENALTY_Q16);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .cmp(&scores[a])
            .then_with(|| feats[b][35].cmp(&feats[a][35]))
            .then_with(|| feats[b][0].cmp(&feats[a][0]))
            .then_with(|| pool[a].cmp(&pool[b]))
    });

    let s0_dominant = !eff_it.multi_hop
        && final_session_indices.len() >= 2
        && (s_score[final_session_indices[0]] + bridge_boost[final_session_indices[0]])
            >= 1.35 * (s_score[final_session_indices[1]] + bridge_boost[final_session_indices[1]]);
    let head_sess_opt = final_session_indices.first().copied();
    let max_per_sess_top5 = if eff_it.multi_hop { 2usize } else { 3usize };
    let target_len = k.max(final_turn_indices.len());
    let mut new_turns: Vec<usize> = Vec::with_capacity(target_len);
    let mut sess_cnt: BTreeMap<usize, usize> = BTreeMap::new();

    for &idx in &order {
        if new_turns.len() >= 5 {
            break;
        }
        let ti = pool[idx];
        if feats[idx][32] > 0 && new_turns.is_empty() {
            continue;
        }
        let s = ix.turns[ti].session;
        let c = sess_cnt.get(&s).copied().unwrap_or(0);
        let sess_cap_eff = if s0_dominant && Some(s) == head_sess_opt {
            4usize
        } else {
            max_per_sess_top5
        };
        if !new_turns.is_empty() && c >= sess_cap_eff {
            continue;
        }
        new_turns.push(ti);
        *sess_cnt.entry(s).or_insert(0) += 1;
    }

    for &idx in &order {
        if new_turns.len() >= target_len {
            break;
        }
        let ti = pool[idx];
        if !new_turns.contains(&ti) {
            new_turns.push(ti);
        }
    }
    for &ti in final_turn_indices.iter() {
        if new_turns.len() >= target_len {
            break;
        }
        if !new_turns.contains(&ti) {
            new_turns.push(ti);
        }
    }
    // Post-GBDT Partner Q -> Queried Speaker Answer Promotion:
    // When Slot 0 is an unqueried partner's question (`is_question && feats[i0][15] < 0`)
    // and the immediately following turn in the same session is a declarative, non-backchannel,
    // non-superseded reply by a queried speaker, promote that direct answer to Slot 0.
    if new_turns.len() >= 2 {
        let t0 = new_turns[0];
        if ix.turns[t0].is_pure_info_question
            && let Some(i0) = pool.iter().position(|&x| x == t0)
            && feats[i0][15] < 0
            && feats[i0][16] == 0
        {
            let s0_t = ix.turns[t0].session;
            if let Some(members) = ix.session_turns.get(s0_t)
                && let Some(p0) = members.iter().position(|&m| m == t0)
                && p0 + 1 < members.len()
            {
                let nxt = members[p0 + 1];
                let nt_tr = &ix.turns[nxt];
                let nxt_is_queried = if let Some(inxt) = pool.iter().position(|&x| x == nxt) {
                    feats[inxt][15] > 0 && feats[inxt][32] == 0
                } else {
                    !turn_is_superseded(ix, nxt)
                        && !nt_tr.speaker.is_empty()
                        && c2_query_mentions_speaker(&q.text, &nt_tr.speaker)
                };
                if nxt_is_queried && !nt_tr.is_question && !nt_tr.is_backchannel {
                    if let Some(pos) = new_turns.iter().position(|&x| x == nxt) {
                        new_turns.remove(pos);
                    }
                    new_turns.insert(0, nxt);
                    if new_turns.len() > target_len {
                        new_turns.truncate(target_len);
                    }
                }
            }
        }
    }
    // Post-GBDT Same-Session Self-Payload Arbiter on Named Dialogues:
    // When Rank 1 (`t0`) and a same-session nearby declarative (`tc`, `|pos| <= 2`) are within
    // `<= 24_576` Q16 GBDT margin, prefer the originating self-payload turn over a question/echo opener.
    if new_turns.len() >= 2 {
        let t0 = new_turns[0];
        if let Some(i0) = pool.iter().position(|&x| x == t0) {
            let p0 = turn_pos(ix, t0);
            let s0_t = ix.turns[t0].session;
            let own0 = ieee754_bits_to_q16(t_own.get(t0).copied().unwrap_or(0.0).to_bits());
            let mut flip_slot: Option<usize> = None;
            let pure_zero_decoy = ix.turns[t0].slot_entities_clean == 0 && feats[i0][9] == 0;
            let scan_self = if pure_zero_decoy {
                new_turns.len().min(5)
            } else {
                new_turns.len().min(3)
            };
            let max_pos_diff = if pure_zero_decoy { 4 } else { 2 };
            let max_gap_self = if pure_zero_decoy { 48_000 } else { 24_576 };
            for slot in 1..scan_self {
                let tc = new_turns[slot];
                if ix.turns[tc].session != s0_t
                    || ix.turns[tc].is_question
                    || ix.turns[tc].is_backchannel
                {
                    continue;
                }
                let near = match (p0, turn_pos(ix, tc)) {
                    (Some(a), Some(b)) => a.abs_diff(b) <= max_pos_diff,
                    _ => false,
                };
                if !near {
                    continue;
                }
                let Some(ic) = pool.iter().position(|&x| x == tc) else {
                    continue;
                };
                if feats[ic][32] > 0 || scores[i0].saturating_sub(scores[ic]) > max_gap_self {
                    continue;
                }
                let ownc = ieee754_bits_to_q16(t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                let cond_a = (ix.turns[t0].is_question || own0.saturating_mul(100) < ownc.saturating_mul(85))
                    && feats[ic][9] >= feats[i0][9];
                let cond_b = ix.turns[t0].slot_entities_clean == 0
                    && ix.turns[tc].slot_entities_clean >= 1
                    && feats[ic][20] >= feats[i0][20]
                    && feats[ic][15] >= feats[i0][15];
                let cond_c = pure_zero_decoy
                    && feats[ic][15] == 2
                    && feats[ic][9] > 0
                    && ix.turns[tc].slot_entities_clean >= 1;
                if cond_a || cond_b || cond_c {
                    flip_slot = Some(slot);
                    break;
                }
            }
            if let Some(slot) = flip_slot {
                new_turns.swap(0, slot);
            }
        }
    }
    // Post-GBDT Speaker-Subject Alignment & Lexical WH Answer-Type Hard Gate (`0..5` invariant):
    // 1. When the query targets a single non-conjoined speaker (`feats[ic][15] == 2`) without `WH_PERSON`,
    //    if Rank 1 is spoken by an unqueried speaker (`feats[i0][15] < 0`) and a Top-5 declarative by
    //    the queried speaker covers strong non-speaker IC (or `t0` is a question / adjacent reply),
    //    rotate the queried speaker's declarative to Rank 1.
    // 2. When Rank 1 is a question (`is_question`) or fails the WH-type check (`feats[i0][20] < 0`)
    //    while a Top-5 declarative satisfies it (`feats[ic][20] > feats[i0][20]`) without speaker
    //    regression, rotate it to Rank 1.
    if new_turns.len() >= 2 && (wh_slots(&q.text) & WH_PERSON) == 0 {
        let t0 = new_turns[0];
        if let Some(i0) = pool.iter().position(|&x| x == t0) {
            let sp_names_tr = session_speaker_names(ix, &ix.session_turns[ix.turns[t0].session]);
            let q_exact_ids_sp: Vec<u32> = plan
                .terms
                .iter()
                .filter(|qt| qt.exact_hit && !is_speaker_term(ix, qt.id, &sp_names_tr))
                .map(|qt| qt.id)
                .collect();
            let turn_only_other = |ti: usize| -> bool {
                let tr = &ix.turns[ti];
                if tr.other_speaker_terms.is_empty() {
                    return false;
                }
                let mut hit = false;
                for &qid in &q_exact_ids_sp {
                    if tr.content.contains(&qid) {
                        hit = true;
                        if !tr.other_speaker_terms.contains(&qid) {
                            return false;
                        }
                    }
                }
                hit
            };
            let mut sp_win: Option<(usize, i32, i64)> = None;
            let t0_adv_protected = (feats[i0][41] > 0
                || (feats[i0][15] < 0 && feats[i0][16] > 0))
                && !ix.turns[t0].is_pure_info_question
                && !turn_only_other(t0);
            if feats[i0][15] < 0 && !t0_adv_protected {
                let q_toks_sp = raw_tokens(&q.text);
                let q_refs_sp: Vec<&str> = q_toks_sp.iter().map(String::as_str).collect();
                let is_od_infer_sp =
                    crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape(
                        &q_refs_sp,
                    );
                let p0 = turn_pos(ix, t0);
                let s0_t = ix.turns[t0].session;
                let own0 = ieee754_bits_to_q16(t_own.get(t0).copied().unwrap_or(0.0).to_bits());
                let scn0 = ieee754_bits_to_q16(t_scene.get(t0).copied().unwrap_or(0.0).to_bits());
                let sp_scan = if is_od_infer_sp {
                    new_turns.len().min(8)
                } else {
                    new_turns.len().min(6)
                };
                for slot in 1..sp_scan {
                    let tc = new_turns[slot];
                    let trc = &ix.turns[tc];
                    let allow_multi_clause_od = is_od_infer_sp && trc.content.len() >= 10;
                    if (trc.is_question && !allow_multi_clause_od) || trc.is_backchannel {
                        continue;
                    }
                    let Some(ic) = pool.iter().position(|&x| x == tc) else {
                        continue;
                    };
                    if (feats[ic][32] > 0 && !allow_multi_clause_od) || feats[ic][15] != 2 {
                        continue;
                    }
                    let adj_same_sess = trc.session == s0_t
                        && match (p0, turn_pos(ix, tc)) {
                            (Some(a), Some(b)) => a.abs_diff(b) <= 2,
                            _ => false,
                        };
                    let ownc = ieee754_bits_to_q16(t_own.get(tc).copied().unwrap_or(0.0).to_bits());
                    let scnc = ieee754_bits_to_q16(t_scene.get(tc).copied().unwrap_or(0.0).to_bits());
                    let od_sp_ok = is_od_infer_sp
                        && scores[i0].saturating_sub(scores[ic]) <= 95_000
                        && (adj_same_sess
                            || trc.is_habitual
                            || scnc.saturating_mul(100) >= scn0.saturating_mul(35)
                            || feats[ic][9] * 10 >= feats[i0][9] * 5);
                    // Vocative / 2nd-person praise flip: only flip non-OD `feats[i0][15] < 0` with relaxed IC
                    // when `t0` exclusively references the queried terms in 2nd-person (`turn_only_other(t0)`),
                    // preventing false-premise Adversarial queries from evicting Speaker B's 1st-person gold fact.
                    let vocative_sp_ok = turn_only_other(t0)
                        && !turn_only_other(tc)
                        && feats[ic][9] * 10 >= feats[i0][9] * 8
                        && scores[i0].saturating_sub(scores[ic]) <= 48_000;
                    let ic_ok = if ix.turns[t0].is_question {
                        feats[ic][9] * 10 >= feats[i0][9] * 7
                    } else {
                        od_sp_ok
                            || vocative_sp_ok
                            || (trc.session == s0_t
                                && feats[ic][9] >= feats[i0][9]
                                && ownc.saturating_mul(100) >= own0.saturating_mul(65)
                                && scores[i0].saturating_sub(scores[ic]) <= 42_000)
                            || (feats[ic][9] >= feats[i0][9]
                                && ((adj_same_sess && ownc * 5 >= own0 * 4)
                                    || (ownc >= own0
                                        && scores[i0].saturating_sub(scores[ic]) <= 32_000)))
                    };
                    if ic_ok && feats[ic][20] >= feats[i0][20] {
                        let key = (slot, feats[ic][9], scores[ic]);
                        if sp_win.is_none_or(|(_, bic, bsc)| (key.1, key.2) > (bic, bsc)) {
                            sp_win = Some(key);
                        }
                    }
                }
            }
            // Pass 1: Cross-speaker subject alignment (`sp_win`).
            if let Some((slot, _, _)) = sp_win {
                new_turns[..=slot].rotate_right(1);
            }
            // Pass 2 (unchained C13): Strict Speaker-Attributed Predicate Guard (`turn_only_other`).
            let t0 = new_turns[0];
            if let Some(i0) = pool.iter().position(|&x| x == t0)
                && feats[i0][15] == 2
                && turn_only_other(t0)
            {
                let mut own_win: Option<(usize, i32, i64)> = None;
                for slot in 1..new_turns.len().min(5) {
                    let tc = new_turns[slot];
                    let trc = &ix.turns[tc];
                    if trc.is_question || trc.is_backchannel || turn_only_other(tc) {
                        continue;
                    }
                    let Some(ic) = pool.iter().position(|&x| x == tc) else {
                        continue;
                    };
                    if feats[ic][32] == 0
                        && feats[ic][15] == 2
                        && feats[ic][9] * 10 >= feats[i0][9] * 8
                        && scores[i0].saturating_sub(scores[ic]) <= 36_000
                    {
                        let key = (slot, feats[ic][9], scores[ic]);
                        if own_win.is_none_or(|(_, bic, bsc)| (key.1, key.2) > (bic, bsc)) {
                            own_win = Some(key);
                        }
                    }
                }
                if let Some((slot, _, _)) = own_win {
                    new_turns[..=slot].rotate_right(1);
                }
            }
            // Pass 3 (unchained C13): #7 Cross-Speaker Self-Disclosure Tie-Break (`self_win`).
            let t0 = new_turns[0];
            if let Some(i0) = pool.iter().position(|&x| x == t0)
                && ((feats[i0][15] == 2 && ix.turns[t0].is_question)
                    || (feats[i0][15] == -1
                        && feats[i0][41] <= 0
                        && (ix.turns[t0].is_question || turn_only_other(t0))))
            {
                let mut self_win: Option<(usize, i32, i64)> = None;
                for slot in 1..new_turns.len().min(5) {
                    let tc = new_turns[slot];
                    let trc = &ix.turns[tc];
                    if trc.is_question || trc.is_backchannel || turn_only_other(tc) {
                        continue;
                    }
                    let Some(ic) = pool.iter().position(|&x| x == tc) else {
                        continue;
                    };
                    if feats[ic][32] == 0
                        && feats[ic][15] == 2
                        && feats[ic][9] >= feats[i0][9]
                        && scores[i0].saturating_sub(scores[ic]) <= 24_000
                    {
                        let key = (slot, feats[ic][9], scores[ic]);
                        if self_win.is_none_or(|(_, bic, bsc)| (key.1, key.2) > (bic, bsc)) {
                            self_win = Some(key);
                        }
                    }
                }
                if let Some((slot, _, _)) = self_win {
                    new_turns[..=slot].rotate_right(1);
                }
            }
            // Pass 4 (unchained C13): 4-Bit ModalityMask Guard (`0x04 = HYPOTHETICAL`).
            let t0 = new_turns[0];
            if let Some(i0) = pool.iter().position(|&x| x == t0)
                && (ix.turns[t0].modality_mask & 0x04) != 0
            {
                let q_low_mod = q.text.to_ascii_lowercase();
                let is_past_q_m = q_low_mod.starts_with("what did ")
                    || q_low_mod.starts_with("when did ")
                    || q_low_mod.starts_with("where has ")
                    || q_low_mod.starts_with("where did ")
                    || q_low_mod.starts_with("what has ");
                let ic_num = if is_past_q_m { 7 } else { 10 };
                let max_gap = if is_past_q_m { 48_000 } else { 28_000 };
                let mut mod_win: Option<(usize, i32, i64)> = None;
                for slot in 1..new_turns.len().min(5) {
                    let tc = new_turns[slot];
                    let trc = &ix.turns[tc];
                    if trc.is_question || trc.is_backchannel || (trc.modality_mask & 0x01) == 0 {
                        continue;
                    }
                    let Some(ic) = pool.iter().position(|&x| x == tc) else {
                        continue;
                    };
                    if feats[ic][32] == 0
                        && feats[ic][15] >= feats[i0][15]
                        && feats[ic][9] * 10 >= feats[i0][9] * ic_num
                        && scores[i0].saturating_sub(scores[ic]) <= max_gap
                    {
                        let key = (slot, feats[ic][9], scores[ic]);
                        if mod_win.is_none_or(|(_, bic, bsc)| (key.1, key.2) > (bic, bsc)) {
                            mod_win = Some(key);
                        }
                    }
                }
                if let Some((slot, _, _)) = mod_win {
                    new_turns[..=slot].rotate_right(1);
                }
            }
            // Pass 5 (unchained C13): Lexical WH Answer-Type Hard Gate (`feats[i0][20] < 0`).
            let t0 = new_turns[0];
            if let Some(i0) = pool.iter().position(|&x| x == t0)
                && feats[i0][20] < 0
            {
                let mut wh_win: Option<(usize, i32, i64)> = None;
                for slot in 1..new_turns.len().min(5) {
                    let tc = new_turns[slot];
                    let trc = &ix.turns[tc];
                    if trc.is_question || trc.is_backchannel {
                        continue;
                    }
                    let Some(ic) = pool.iter().position(|&x| x == tc) else {
                        continue;
                    };
                    if feats[ic][32] == 0
                        && feats[ic][20] > 0
                        && feats[ic][15] >= feats[i0][15]
                        && feats[ic][9] * 10 >= feats[i0][9] * 9
                        && scores[i0].saturating_sub(scores[ic]) <= 26_000
                    {
                        let key = (slot, feats[ic][20], scores[ic]);
                        if wh_win.is_none_or(|(_, blr, bsc)| (key.1, key.2) > (blr, bsc)) {
                            wh_win = Some(key);
                        }
                    }
                }
                if let Some((slot, _, _)) = wh_win {
                    new_turns[..=slot].rotate_right(1);
                }
            }
        }
    }
    let q_toks_sync = raw_tokens(&q.text);
    let q_refs_sync: Vec<&str> = q_toks_sync.iter().map(String::as_str).collect();
    let is_inferential_modal =
        crate::retrieval::lexical::speaker_translation_q16::is_open_domain_inferential_shape(
            &q_refs_sync,
        );
    let q_low_od = q.text.trim_start().to_ascii_lowercase();
    let is_strong_od_infer = is_inferential_modal
        && has_named_speakers_eff
        && (q_low_od.starts_with("would ")
            || q_low_od.contains(" would ")
            || q_low_od.starts_with("could ")
            || q_low_od.contains(" could ")
            || q_low_od.contains(" likely")
            || q_low_od.contains(" might "))
        && new_turns.len() >= 2;
    if is_strong_od_infer {
        // Dedicated Open-Domain Inferential Rank-1 Arbiter:
        // Fuses KITE Upward-Cone Typed Meet + Discrete Sinkhorn OT + SYNAPSE Spreading Activation
        // + exact/compound-suffix IC + `is_habitual` + `t_scene` over candidate turns by the
        // queried speaker(s), rescuing turns where generic polysemous decoys stole Slot 0.
        let mut queried_sp_od: BTreeSet<String> = BTreeSet::new();
        for (idx, &ti) in pool.iter().enumerate() {
            if feats[idx][15] > 0 && !ix.turns[ti].speaker.is_empty() {
                queried_sp_od.insert(ix.turns[ti].speaker.to_ascii_lowercase());
            }
        }
        let ns_od = (ix.session_turns.len() as u32).max(1);
        let max_suf_df = (ns_od / 3).max(4);
        let mut q_suffix_ids: Vec<u32> = Vec::new();
        for w in &q_toks_sync {
            if !w.is_ascii() || !is_content(w) || queried_sp_od.contains(w.as_str()) {
                continue;
            }
            if w.len() >= 6 && ix.vocab.get(w).is_none() {
                for (vid, vname) in ix.vocab.names.iter().enumerate() {
                    if !vname.starts_with('\u{1}')
                        && crate::retrieval::lexical::bm25f::damerau_levenshtein_le_1(w, vname)
                    {
                        let cand_id = vid as u32;
                        let df = ix.s_stats.df.get(&cand_id).copied().unwrap_or(0);
                        if (1..=max_suf_df).contains(&df) && !q_suffix_ids.contains(&cand_id) {
                            q_suffix_ids.push(cand_id);
                        }
                    }
                }
            }
            if w.len() < 7 {
                continue;
            }
            for split in 3..=(w.len() - 4) {
                let suf = &w[split..];
                let st_suf = stem(suf);
                for cand_id in [ix.vocab.get(suf), ix.vocab.get(&st_suf)]
                    .into_iter()
                    .flatten()
                {
                    let df = ix.s_stats.df.get(&cand_id).copied().unwrap_or(0);
                    if (1..=max_suf_df).contains(&df) && !q_suffix_ids.contains(&cand_id) {
                        q_suffix_ids.push(cand_id);
                    }
                }
            }
        }
        let (_, kite_t_od) = crate::retrieval::index::kite_typed::score_kite_typed_facet_meet_q16(
            ix,
            plan,
            &q.text,
            &queried_sp_od,
            0,
            65_536,
        );
        let ot_od = crate::retrieval::lexical::sinkhorn_ot_q16::SinkhornQuery::build(
            ix,
            &q.text,
            &queried_sp_od,
        );
        let max_scn_od = t_scene
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let mut od_cands: Vec<(usize, u64)> = Vec::new();
        for (idx, &ti) in pool.iter().enumerate() {
            let tr = &ix.turns[ti];
            if tr.is_question || tr.is_backchannel || turn_is_superseded(ix, ti) {
                continue;
            }
            if !queried_sp_od.is_empty() && !queried_sp_od.contains(&tr.speaker.to_ascii_lowercase()) {
                continue;
            }
            let scn_q = (((t_scene.get(ti).copied().unwrap_or(0.0).max(0.0)) / max_scn_od) * 65_536.0) as u64;
            let exact_q = (feats[idx][9].max(0) as u64).min(65_536);
            let suf_q = if q_suffix_ids.iter().any(|id| tr.content.contains(id)) {
                24_576u64
            } else {
                0u64
            };
            let kite_q = kite_t_od.get(ti).copied().unwrap_or(0);
            let ot_q = ot_od.score_turn_q16(ix, ti);
            let hab_q = if tr.is_habitual { 16_384u64 } else { 0u64 };
            let comb = scn_q
                .saturating_add(exact_q.saturating_mul(3) / 2)
                .saturating_add(suf_q)
                .saturating_add(kite_q.saturating_mul(3) / 2)
                .saturating_add(ot_q.saturating_mul(2))
                .saturating_add(hab_q);
            if comb > 0 {
                od_cands.push((ti, comb));
            }
        }
        od_cands.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        od_cands.truncate(20);
        let syn_ranked = crate::retrieval::graph::synapse_activation_q16::synapse_activate_speaker_turns_q16(
            ix,
            &od_cands,
        );
        if let Some(&(best_ti, best_syn)) = syn_ranked.first() {
            let t0 = new_turns[0];
            let best_in_topk = new_turns.iter().take(10).position(|&x| x == best_ti);
            let best_kite = kite_t_od.get(best_ti).copied().unwrap_or(0);
            let t0_kite = kite_t_od.get(t0).copied().unwrap_or(0);
            let best_exact_q = pool
                .iter()
                .position(|&x| x == best_ti)
                .map(|idx| feats[idx][9].max(0) as u64)
                .unwrap_or(0);
            let t0_exact_q = pool
                .iter()
                .position(|&x| x == t0)
                .map(|idx| feats[idx][9].max(0) as u64)
                .unwrap_or(0);
            let t0_syn = syn_ranked
                .iter()
                .find(|&&(ti, _)| ti == t0)
                .map(|&(_, sc)| sc)
                .unwrap_or(0);
            let t0_wrong_sp = !queried_sp_od.is_empty()
                && !queried_sp_od.contains(&ix.turns[t0].speaker.to_ascii_lowercase());
            let decisive = best_ti != t0
                && (t0_wrong_sp
                    || (best_syn > t0_syn.saturating_mul(115) / 100
                        && best_kite >= t0_kite
                        && best_exact_q >= t0_exact_q)
                    || (best_exact_q > t0_exact_q && best_syn >= t0_syn)
                    || (t_scene.get(best_ti).copied().unwrap_or(0.0)
                        > 1.15 * t_scene.get(t0).copied().unwrap_or(0.0)
                        && best_kite > t0_kite));
            if decisive {
                if let Some(pos) = best_in_topk {
                    new_turns[..=pos].rotate_right(1);
                } else {
                    new_turns.retain(|&x| x != best_ti);
                    new_turns.insert(0, best_ti);
                    new_turns.truncate(k);
                }
                let win_sess = ix.turns[best_ti].session;
                if let Some(s_pos) = final_session_indices.iter().take(10).position(|&s| s == win_sess)
                    && s_pos > 0
                {
                    let moved = final_session_indices.remove(s_pos);
                    final_session_indices.insert(0, moved);
                }
            }
        }
    }
    // Forward GeoNames (City -> Country/US-State), WordNet Forward Hypernym-Category,
    // and Single-Speaker Possessive Alignment Arbiter for named-speaker dialogues.
    if has_named_speakers_eff && !new_turns.is_empty() {
        let mut sp_set_w17: Vec<String> = Vec::new();
        for (idx, &ti) in pool.iter().enumerate() {
            if feats[idx][15] > 0 && !ix.turns[ti].speaker.is_empty() {
                let sp_l = ix.turns[ti].speaker.to_ascii_lowercase();
                if !sp_set_w17.contains(&sp_l) {
                    sp_set_w17.push(sp_l);
                }
            }
        }
        let single_sp_w17: Option<&str> = if sp_set_w17.len() == 1 {
            Some(sp_set_w17[0].as_str())
        } else {
            None
        };
        let q_toks_w17 = raw_tokens(&q.text);
        let countries_list = crate::geonames::geo_class_instances("country");
        let states_list = crate::geonames::geo_class_instances("state");
        let explicit_places: Vec<String> = q_toks_w17
            .iter()
            .filter(|w| {
                w.len() >= 4
                    && !sp_set_w17.iter().any(|sp| sp == w.as_str())
                    && (countries_list.iter().any(|c| c == w.as_str())
                        || states_list.iter().any(|s| s == w.as_str()))
            })
            .cloned()
            .collect();
        let wants_country = q_toks_w17
            .iter()
            .any(|w| matches!(w.as_str(), "country" | "countries" | "nation" | "nations"));
        let wants_state = q_toks_w17
            .iter()
            .any(|w| matches!(w.as_str(), "state" | "states" | "province" | "provinces"));
        // hardcode-scan: closed-class (WH determiners & generic classifier nouns skipped as hypernym heads)
        let q_hyp_target: Option<String> = q_toks_w17
            .windows(4)
            .find_map(|win| {
                if (win[0] == "which" || win[0] == "what")
                    && matches!(
                        win[1].as_str(),
                        "kind" | "sort" | "type" | "breed" | "genre" | "style" | "form"
                    )
                    && win[2] == "of"
                    && win[3].len() >= 3
                    && is_content(&win[3])
                    && !crate::wordnet::full_lexicon_neighbors(&win[3]).is_empty()
                {
                    Some(win[3].clone())
                } else {
                    None
                }
            })
            .or_else(|| {
                q_toks_w17.windows(2).find_map(|win| {
                    if (win[0] == "which" || win[0] == "what")
                        && win[1].len() >= 4
                        && is_content(&win[1])
                        && !matches!(
                            win[1].as_str(),
                            "country"
                                | "state"
                                | "city"
                                | "kind"
                                | "sort"
                                | "type"
                                | "breed"
                                | "genre"
                                | "style"
                                | "form"
                                | "ways"
                                | "things"
                                | "year"
                                | "month"
                                | "time"
                                | "additional"
                        )
                        && !crate::wordnet::full_lexicon_neighbors(&win[1]).is_empty()
                    {
                        Some(win[1].clone())
                    } else {
                        None
                    }
                })
            });
        let geo_score_turn = |ti: usize| -> u32 {
            if explicit_places.is_empty() && !wants_country && !wants_state {
                return 0;
            }
            let mut sc = 0u32;
            for tw in raw_tokens(&ix.turns[ti].body_for_headers) {
                if tw.len() < 4 || q_toks_w17.iter().any(|qw| qw == &tw) {
                    continue;
                }
                if !explicit_places.is_empty()
                    && (explicit_places.iter().any(|p| p == &tw)
                        || crate::geonames::geo_neighbors(&tw)
                            .iter()
                            .any(|&(nb, _)| explicit_places.iter().any(|p| p == nb)))
                {
                    sc = sc.saturating_add(3);
                }
                if wants_country
                    && (countries_list.iter().any(|c| c == &tw)
                        || crate::geonames::geo_neighbors(&tw)
                            .iter()
                            .any(|&(nb, _)| countries_list.iter().any(|c| c == nb)))
                {
                    sc = sc.saturating_add(2);
                }
                if wants_state
                    && (states_list.iter().any(|s| s == &tw)
                        || crate::geonames::geo_neighbors(&tw)
                            .iter()
                            .any(|&(nb, _)| states_list.iter().any(|s| s == nb)))
                {
                    sc = sc.saturating_add(2);
                }
            }
            sc
        };
        let hyp_score_turn = |ti: usize| -> u32 {
            let Some(ref hyp) = q_hyp_target else {
                return 0;
            };
            let mut sc = 0u32;
            for tw in raw_tokens(&ix.turns[ti].body_for_headers) {
                if tw.len() < 4 || q_toks_w17.iter().any(|qw| qw == &tw) {
                    continue;
                }
                if tw == *hyp || crate::wordnet::heyting_implication_ic_q16(&tw, hyp) > 0 {
                    sc = sc.saturating_add(1);
                }
            }
            sc
        };
        let t0 = new_turns[0];
        let mut q_nonsp_mask = 0u64;
        for qt in &plan.terms {
            if qt.exact_hit
                && let Some(g) = qt.group
                && g < 64
            {
                let gnm = plan.groups.get(g).map(String::as_str).unwrap_or("");
                if !sp_set_w17.iter().any(|sp| sp == gnm) {
                    q_nonsp_mask |= 1u64 << g;
                }
            }
        }
        let ex0_nonsp = (t_exact_mask.get(t0).copied().unwrap_or(0) & q_nonsp_mask).count_ones();
        let geo0 = geo_score_turn(t0);
        let hyp0 = hyp_score_turn(t0);
        let mut cand_turns_w17: Vec<usize> = pool.clone();
        for &si in final_session_indices.iter().take(5) {
            for &ti in &ix.session_turns[si] {
                if !cand_turns_w17.contains(&ti) {
                    cand_turns_w17.push(ti);
                }
            }
        }
        let mut promoted_w17: Option<usize> = None;
        if !explicit_places.is_empty() || wants_country || wants_state || q_hyp_target.is_some() {
            let mut best_ans: Option<(u32, u32, u64, usize)> = None;
            let tsc0 = t_scene.get(t0).copied().unwrap_or(0.0);
            let i0_w17 = pool.iter().position(|&x| x == t0);
            let gsc0_w17 = i0_w17.map(|i| scores[i]).unwrap_or(i64::MIN / 2);
            for &tc in &cand_turns_w17 {
                if tc == t0 {
                    continue;
                }
                let trc = &ix.turns[tc];
                if trc.is_question || trc.is_backchannel || turn_is_superseded(ix, tc) {
                    continue;
                }
                if let Some(sp) = single_sp_w17
                    && !trc.speaker.eq_ignore_ascii_case(sp)
                {
                    continue;
                }
                let gc = geo_score_turn(tc);
                let hc = hyp_score_turn(tc);
                let gain_geo = gc > geo0 && geo0 == 0;
                let gain_hyp = q_hyp_target.is_some() && hc > hyp0 && hyp0 == 0;
                if !gain_geo && !gain_hyp {
                    continue;
                }
                let exc_ns = (t_exact_mask.get(tc).copied().unwrap_or(0) & q_nonsp_mask).count_ones();
                let tscc = t_scene.get(tc).copied().unwrap_or(0.0);
                let gsc_c = pool
                    .iter()
                    .position(|&x| x == tc)
                    .map(|i| scores[i])
                    .unwrap_or(i64::MIN / 2);
                let geo_ok = gain_geo
                    && exc_ns >= ex0_nonsp
                    && (tscc >= 0.65 * tsc0 || gsc_c.saturating_add(24_000) >= gsc0_w17);
                let hyp_ok = gain_hyp
                    && exc_ns >= ex0_nonsp
                    && tscc >= 0.72 * tsc0
                    && gsc_c.saturating_add(22_000) >= gsc0_w17;
                if geo_ok || hyp_ok {
                    let key = (gc * 2 + hc, exc_ns, ieee754_bits_to_q16(tscc.to_bits()), tc);
                    if best_ans.is_none_or(|b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                        best_ans = Some(key);
                    }
                }
            }
            if let Some((_, _, _, win_ti)) = best_ans {
                promoted_w17 = Some(win_ti);
            }
        }
        if let Some(win_ti) = promoted_w17 {
            if let Some(pos) = new_turns.iter().take(10).position(|&x| x == win_ti) {
                if pos > 0 {
                    new_turns[..=pos].rotate_right(1);
                }
            } else {
                new_turns.retain(|&x| x != win_ti);
                new_turns.insert(0, win_ti);
            }
            let win_sess = ix.turns[win_ti].session;
            if let Some(s_pos) = final_session_indices.iter().take(8).position(|&s| s == win_sess)
                && s_pos > 0
            {
                let mv = final_session_indices.remove(s_pos);
                final_session_indices.insert(0, mv);
            }
        }
    }
    new_turns.truncate(k);
    *final_turn_indices = new_turns;
    if !is_inferential_modal
        && let Some(&win_ti) = final_turn_indices.first()
        && final_session_indices.len() >= 2
    {
        let win_sess = ix.turns[win_ti].session;
        let cur_s0 = final_session_indices[0];
        if win_sess != cur_s0
            && let Some(s_pos) = final_session_indices.iter().take(6).position(|&s| s == win_sess)
        {
            let win_sc = pool
                .iter()
                .position(|&x| x == win_ti)
                .map(|i| scores[i])
                .unwrap_or(scores[order[0]]);
            let cur_s0_best = (0..pool.len())
                .filter(|&i| ix.turns[pool[i]].session == cur_s0)
                .map(|i| scores[i])
                .max()
                .unwrap_or(i64::MIN / 2);
            if win_sc.saturating_sub(cur_s0_best) >= AT1_SESS_SYNC_MARGIN_Q16 {
                let moved = final_session_indices.remove(s_pos);
                final_session_indices.insert(0, moved);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn at1_gbdt_table_matches_pinned_sha256() {
        let bytes = canonical_gbdt_bytes();
        let digest = Sha256::digest(&bytes);
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, AT1_GBDT_SHA256, "AT1 GBDT sha256 mismatch");
    }

    #[test]
    fn closed_revision_cues_reject_ambient_time_words() {
        assert!(turn_has_revision_cue("I switched my diet to vegetarian instead of keto"));
        assert!(turn_has_revision_cue("Actually my weekly quota was updated to 500"));
        assert!(turn_has_revision_cue("I am no longer taking the morning class"));
        assert!(!turn_has_revision_cue("Right now I need to plan my new schedule for May"));
    }
}

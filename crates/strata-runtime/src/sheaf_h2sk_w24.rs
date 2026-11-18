//! Wave 24A–24L (#590–#594, #610–#613, #616–#618): H²SK Sheaf, Hyperbolic,
//! Holographic Reduced Representations (HRR), Liquid Time-Constant (LTC)
//! Dynamics, Legal Shepardizing, Epistemic Modality & In-Packet Receipts.

use serde::{Deserialize, Serialize};

/// #590: Poincaré ball hyperbolic distance + bounded Ising spin energy.
pub fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();
    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let denom = (1.0 - norm_u_sq).max(1e-9) * (1.0 - norm_v_sq).max(1e-9);
    let arg = 1.0 + 2.0 * diff_sq / denom;
    arg.max(1.0).acosh()
}

pub fn ising_spin_energy(spins: &[i8], couplings: &[(usize, usize, f64)]) -> f64 {
    let mut e = 0.0;
    for &(i, j, j_val) in couplings {
        e -= j_val * (spins[i] as f64) * (spins[j] as f64);
    }
    e
}

/// #591: Cellular-sheaf coboundary residual `sum ||r_{u->e} x_u - r_{v->e} x_v||^2`.
/// Non-zero residual witnesses global cohomology inconsistency across local beliefs!
pub fn cellular_sheaf_residual(
    stalks: &[f64],
    restriction_edges: &[(usize, f64, usize, f64)], // (u, r_u, v, r_v)
) -> f64 {
    restriction_edges
        .iter()
        .map(|&(u, r_u, v, r_v)| (r_u * stalks[u] - r_v * stalks[v]).powi(2))
        .sum()
}

/// #592: Plate's Holographic Reduced Representations (circular convolution & correlation).
pub fn hrr_circular_bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut out = vec![0.0; n];
    for k in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[j] * b[(k + n - j) % n];
        }
        out[k] = s;
    }
    out
}

pub fn hrr_circular_unbind(key: &[f64], trace: &[f64]) -> Vec<f64> {
    let n = key.len();
    let mut inv = vec![0.0; n];
    if n > 0 {
        inv[0] = key[0];
        for i in 1..n {
            inv[i] = key[n - i];
        }
    }
    hrr_circular_bind(&inv, trace)
}

/// #593: Continuous-time Liquid Time-Constant (LTC) semi-implicit Euler step.
pub fn ltc_step(x: f64, dt_sec: f64, tau_sec: f64, drive_a: f64) -> f64 {
    let rate = dt_sec / tau_sec.max(1e-6);
    (x + rate * drive_a) / (1.0 + rate)
}

/// #594: Unified H²SK experimental kernel diagnostic summary.
#[derive(Debug, Clone, PartialEq)]
pub struct H2skDiagnostic {
    pub hyperbolic_depth: f64,
    pub sheaf_residual: f64,
    pub ltc_activation: f64,
}

pub fn evaluate_h2sk_kernel(point: &[f64], origin: &[f64], sheaf_res: f64, ltc_val: f64) -> H2skDiagnostic {
    H2skDiagnostic {
        hyperbolic_depth: poincare_distance(origin, point),
        sheaf_residual: sheaf_res,
        ltc_activation: ltc_val,
    }
}

/// #610: Legal Shepardizing compiler — stamps `OVERRULED` vs `CONTROLLING` banners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShepardBanner {
    Controlling,
    QuestionedBySheafResidual,
    Overruled,
}

pub fn compile_shepard_banner(is_superseded: bool, sheaf_residual: f64, tol: f64) -> ShepardBanner {
    if is_superseded {
        ShepardBanner::Overruled
    } else if sheaf_residual > tol {
        ShepardBanner::QuestionedBySheafResidual
    } else {
        ShepardBanner::Controlling
    }
}

/// #611: Zero-deixis canonicalization & exact LTC delta-t matrix header.
pub fn canonicalize_zero_deixis(text: &str, session_iso_date: &str) -> String {
    text.replace("today", &format!("[DATE:{session_iso_date}]"))
}

/// #612: Spectral HRR entity auto-tally dot-product detector.
pub fn hrr_entity_similarity(entity_vec: &[f64], superposition: &[f64]) -> f64 {
    entity_vec
        .iter()
        .zip(superposition.iter())
        .map(|(a, b)| a * b)
        .sum()
}

/// #613: Epistemic modality tags (`EXECUTED` vs `HYPOTHETICAL`) & Gricean implicature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicModality {
    ExecutedFact,
    HypotheticalProposal,
}

pub fn classify_epistemic_modality(text: &str, has_exec_receipt: bool) -> EpistemicModality {
    let low = text.to_ascii_lowercase();
    if !has_exec_receipt || low.contains("what if") || low.contains("could try") || low.contains("might") {
        EpistemicModality::HypotheticalProposal
    } else {
        EpistemicModality::ExecutedFact
    }
}

/// #616: Admit exact slots and the closed edge vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClosedEdgeKind {
    Causes,
    Fixes,
    Supersedes,
    Contradicts,
    Touched,
    Calls,
    Cites,
}

pub fn parse_closed_edge_vocab(raw: &str) -> Result<ClosedEdgeKind, &'static str> {
    match raw {
        "causes" => Ok(ClosedEdgeKind::Causes),
        "fixes" => Ok(ClosedEdgeKind::Fixes),
        "supersedes" => Ok(ClosedEdgeKind::Supersedes),
        "contradicts" => Ok(ClosedEdgeKind::Contradicts),
        "touched" => Ok(ClosedEdgeKind::Touched),
        "calls" => Ok(ClosedEdgeKind::Calls),
        "cites" => Ok(ClosedEdgeKind::Cites),
        _ => Err("EdgeKindOutsideClosedVocabulary"),
    }
}

/// #617 & #618: Receipt compiler (`impulse`, `hologram`, `gate`) embedded inside
/// the memory packet and replayable deterministically from precedent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledH2skReceipt {
    pub impulse_milli: u64,
    pub hologram_blake3: String,
    pub gate_verdict: String,
    pub receipt_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptCarryingPacket {
    pub memory_id: String,
    pub content: String,
    pub embedded_receipt: CompiledH2skReceipt,
}

pub fn compile_packet_with_receipt(
    memory_id: &str,
    content: &str,
    impulse_milli: u64,
    hologram_bytes: &[u8],
    gate_verdict: &str,
) -> ReceiptCarryingPacket {
    let hologram_blake3 = blake3::hash(hologram_bytes).to_hex().to_string();
    let mut h = blake3::Hasher::new();
    h.update(memory_id.as_bytes());
    h.update(content.as_bytes());
    h.update(&impulse_milli.to_le_bytes());
    h.update(hologram_blake3.as_bytes());
    h.update(gate_verdict.as_bytes());
    let receipt = CompiledH2skReceipt {
        impulse_milli,
        hologram_blake3,
        gate_verdict: gate_verdict.to_string(),
        receipt_hash: h.finalize().to_hex().to_string(),
    };
    ReceiptCarryingPacket {
        memory_id: memory_id.to_string(),
        content: content.to_string(),
        embedded_receipt: receipt,
    }
}

pub fn replay_packet_receipt(pkt: &ReceiptCarryingPacket, hologram_bytes: &[u8]) -> bool {
    let recomputed = compile_packet_with_receipt(
        &pkt.memory_id,
        &pkt.content,
        pkt.embedded_receipt.impulse_milli,
        hologram_bytes,
        &pkt.embedded_receipt.gate_verdict,
    );
    recomputed.embedded_receipt == pkt.embedded_receipt
}

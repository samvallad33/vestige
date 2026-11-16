//! Bounded Fourier diagnostic over source hashes, never a retrieval store.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumReceipt {
    pub source_hash: String,
    pub bins: usize,
    pub power_micros: u64,
    pub reconstruction_noise_micros: u64,
    pub snr_db_milli: Option<i64>,
}
pub fn project(hashes: &[u8]) -> Option<SpectrumReceipt> {
    if hashes.is_empty() {
        return None;
    }
    let signal: Vec<f64> = hashes
        .iter()
        .take(64)
        .map(|b| f64::from(*b) / 255.0)
        .collect();
    let n = signal.len();
    let mut spectrum = vec![(0.0, 0.0); n];
    for (k, bin) in spectrum.iter_mut().enumerate() {
        for (j, value) in signal.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            bin.0 += value * libm::cos(angle);
            bin.1 -= value * libm::sin(angle);
        }
    }
    let mut noise = 0.0;
    let power: f64 = signal.iter().map(|v| v * v).sum();
    for (j, original) in signal.iter().enumerate() {
        let restored: f64 = spectrum
            .iter()
            .enumerate()
            .map(|(k, (r, i))| {
                let a = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                r * libm::cos(a) - i * libm::sin(a)
            })
            .sum::<f64>()
            / n as f64;
        noise += (restored - original).powi(2);
    }
    Some(SpectrumReceipt {
        source_hash: blake3::hash(hashes).to_hex().to_string(),
        bins: n,
        power_micros: (power * 1e6).round() as u64,
        reconstruction_noise_micros: (noise * 1e6).round() as u64,
        snr_db_milli: if noise > 0.0 && power > 0.0 {
            Some((10000.0 * libm::log10(power / noise)).round() as i64)
        } else {
            None
        },
    })
}

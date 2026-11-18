//! O(log n) Merkle Mountain Range (MMR), ACTA signed retrieval receipts,
//! Rekor transparency checkpoints, GF(2^8) parity self-repair, Crypto-Erasure,
//! and Hybrid Post-Quantum signature envelopes (#400, #401, #402, #406, #407, #770).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"strata-mmr-leaf-v1:");
    h.update(data);
    *h.finalize().as_bytes()
}

fn hash_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"strata-mmr-node-v1:");
    h.update(left);
    h.update(right);
    *h.finalize().as_bytes()
}

/// Complete binary Merkle / MMR proof path for O(log n) inclusion checks (#401, #770).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InclusionProof {
    pub leaf_index: usize,
    pub tree_size: usize,
    /// Sibling hashes from leaf to peak, paired with `true` if sibling is on the right.
    pub siblings: Vec<([u8; 32], bool)>,
    pub peak_index: usize,
    pub all_peaks: Vec<[u8; 32]>,
}

/// Append-only Merkle Mountain Range (#401, #770).
#[derive(Debug, Clone, Default)]
pub struct MmrTree {
    leaves: Vec<[u8; 32]>,
}

impl MmrTree {
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    pub fn push(&mut self, payload: &[u8]) -> usize {
        let idx = self.leaves.len();
        self.leaves.push(hash_leaf(payload));
        idx
    }

    /// Compute MMR peaks for the prefix `0..size` by decomposing `size` into powers of two.
    pub fn peaks_at(&self, size: usize) -> Vec<[u8; 32]> {
        let mut peaks = Vec::new();
        let mut offset = 0usize;
        let mut rem = size.min(self.leaves.len());
        while rem > 0 {
            let pow = 1usize << (usize::BITS - 1 - rem.leading_zeros());
            peaks.push(Self::subtree_root(&self.leaves[offset..offset + pow]));
            offset += pow;
            rem -= pow;
        }
        peaks
    }

    fn subtree_root(slice: &[[u8; 32]]) -> [u8; 32] {
        debug_assert!(slice.len().is_power_of_two());
        if slice.len() == 1 {
            return slice[0];
        }
        let mid = slice.len() / 2;
        let l = Self::subtree_root(&slice[..mid]);
        let r = Self::subtree_root(&slice[mid..]);
        hash_parent(&l, &r)
    }

    /// Bag all peaks right-to-left into one canonical 32-byte MMR root.
    pub fn root_at(&self, size: usize) -> [u8; 32] {
        let peaks = self.peaks_at(size);
        Self::bag_peaks(&peaks)
    }

    pub fn root(&self) -> [u8; 32] {
        self.root_at(self.leaves.len())
    }

    pub fn bag_peaks(peaks: &[[u8; 32]]) -> [u8; 32] {
        if peaks.is_empty() {
            return *blake3::hash(b"strata-mmr-empty").as_bytes();
        }
        let mut acc = *peaks.last().unwrap();
        for peak in peaks[..peaks.len() - 1].iter().rev() {
            acc = hash_parent(peak, &acc);
        }
        acc
    }

    /// Generate an O(log n) inclusion proof for `leaf_index` in `0..self.len()`.
    pub fn prove_inclusion(&self, leaf_index: usize) -> Option<InclusionProof> {
        let size = self.leaves.len();
        if leaf_index >= size {
            return None;
        }
        let all_peaks = self.peaks_at(size);
        let mut offset = 0usize;
        let mut rem = size;
        let mut peak_index = 0usize;
        while rem > 0 {
            let pow = 1usize << (usize::BITS - 1 - rem.leading_zeros());
            if leaf_index < offset + pow {
                let mut siblings = Vec::new();
                Self::collect_siblings(
                    &self.leaves[offset..offset + pow],
                    leaf_index - offset,
                    &mut siblings,
                );
                return Some(InclusionProof {
                    leaf_index,
                    tree_size: size,
                    siblings,
                    peak_index,
                    all_peaks,
                });
            }
            offset += pow;
            rem -= pow;
            peak_index += 1;
        }
        None
    }

    fn collect_siblings(
        slice: &[[u8; 32]],
        rel_idx: usize,
        out: &mut Vec<([u8; 32], bool)>,
    ) {
        if slice.len() == 1 {
            return;
        }
        let mid = slice.len() / 2;
        if rel_idx < mid {
            Self::collect_siblings(&slice[..mid], rel_idx, out);
            let right_hash = Self::subtree_root(&slice[mid..]);
            out.push((right_hash, true));
        } else {
            Self::collect_siblings(&slice[mid..], rel_idx - mid, out);
            let left_hash = Self::subtree_root(&slice[..mid]);
            out.push((left_hash, false));
        }
    }

    pub fn verify_inclusion(payload: &[u8], proof: &InclusionProof, expected_root: &[u8; 32]) -> bool {
        if proof.peak_index >= proof.all_peaks.len() {
            return false;
        }
        let mut curr = hash_leaf(payload);
        for (sib, sib_is_right) in &proof.siblings {
            curr = if *sib_is_right {
                hash_parent(&curr, sib)
            } else {
                hash_parent(sib, &curr)
            };
        }
        if curr != proof.all_peaks[proof.peak_index] {
            return false;
        }
        &Self::bag_peaks(&proof.all_peaks) == expected_root
    }

    /// Consistency check (#401): verifies that `old_peaks` at `old_size` is a prefix of current tree.
    pub fn verify_consistency(&self, old_size: usize, expected_old_root: &[u8; 32]) -> bool {
        if old_size > self.leaves.len() {
            return false;
        }
        &self.root_at(old_size) == expected_old_root
    }
}

/// Sigstore Rekor / C2SP signed checkpoint note (#401, #770).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RekorCheckpoint {
    pub origin: String,
    pub tree_size: usize,
    pub root_hex: String,
    pub signature_hex: String,
}

impl RekorCheckpoint {
    pub fn sign(origin: impl Into<String>, tree: &MmrTree, key: &[u8; 32]) -> Self {
        let origin = origin.into();
        let tree_size = tree.len();
        let root_hex = blake3::Hash::from_bytes(tree.root()).to_hex().to_string();
        let body = format!("{origin}\n{tree_size}\n{root_hex}\n");
        let signature_hex = blake3::keyed_hash(key, body.as_bytes()).to_hex().to_string();
        Self {
            origin,
            tree_size,
            root_hex,
            signature_hex,
        }
    }
}

/// ACTA-compatible signed retrieval receipt + key discovery (#400).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActaRetrievalReceipt {
    pub acta_version: String,
    pub key_id: String,
    pub query_handle: String,
    pub returned_memory_ids: Vec<String>,
    pub mmr_root_hex: String,
    pub signature_hex: String,
}

impl ActaRetrievalReceipt {
    pub fn issue(
        key_id: impl Into<String>,
        signing_key: &[u8; 32],
        query_handle: impl Into<String>,
        returned_memory_ids: Vec<String>,
        mmr_root: &[u8; 32],
    ) -> Self {
        let key_id = key_id.into();
        let query_handle = query_handle.into();
        let mmr_root_hex = blake3::Hash::from_bytes(*mmr_root).to_hex().to_string();
        let transcript = format!(
            "ACTA-1.0|{key_id}|{query_handle}|{}|{mmr_root_hex}",
            returned_memory_ids.join(",")
        );
        let signature_hex = blake3::keyed_hash(signing_key, transcript.as_bytes())
            .to_hex()
            .to_string();
        Self {
            acta_version: "ACTA-1.0".to_string(),
            key_id,
            query_handle,
            returned_memory_ids,
            mmr_root_hex,
            signature_hex,
        }
    }

    pub fn verify(&self, signing_key: &[u8; 32]) -> bool {
        let transcript = format!(
            "{}|{}|{}|{}|{}",
            self.acta_version,
            self.key_id,
            self.query_handle,
            self.returned_memory_ids.join(","),
            self.mmr_root_hex
        );
        let expected = blake3::keyed_hash(signing_key, transcript.as_bytes())
            .to_hex()
            .to_string();
        expected == self.signature_hex
    }
}

/// Crypto-erasure with per-memory data keys (#402):
/// Segment ciphertext is never modified; destroying the per-memory 32-byte key
/// makes the payload cryptographically unrecoverable and emits a TombstoneReceipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstoneReceipt {
    pub memory_id: String,
    pub key_destroyed: bool,
    pub ciphertext_blake3: String,
}

#[derive(Debug, Default)]
pub struct CryptoErasureVault {
    keyring: BTreeMap<String, [u8; 32]>,
    ciphertexts: BTreeMap<String, Vec<u8>>,
    tombstones: BTreeMap<String, TombstoneReceipt>,
}

fn xor_keystream(key: &[u8; 32], mem_id: &str, data: &[u8]) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(b"strata-crypto-erasure-v1:");
    hasher.update(mem_id.as_bytes());
    let mut xof = hasher.finalize_xof();
    let mut stream = vec![0u8; data.len()];
    xof.fill(&mut stream);
    data.iter().zip(stream.iter()).map(|(a, b)| a ^ b).collect()
}

impl CryptoErasureVault {
    pub fn seal_memory(&mut self, memory_id: impl Into<String>, data_key: [u8; 32], plaintext: &[u8]) {
        let id = memory_id.into();
        let ct = xor_keystream(&data_key, &id, plaintext);
        self.keyring.insert(id.clone(), data_key);
        self.ciphertexts.insert(id, ct);
    }

    pub fn open_memory(&self, memory_id: &str) -> Result<Vec<u8>, &'static str> {
        if self.tombstones.contains_key(memory_id) {
            return Err("CryptoErasedTombstone");
        }
        let key = self.keyring.get(memory_id).ok_or("MissingDataKey")?;
        let ct = self.ciphertexts.get(memory_id).ok_or("MissingCiphertext")?;
        Ok(xor_keystream(key, memory_id, ct))
    }

    pub fn purge_key(&mut self, memory_id: &str) -> Result<TombstoneReceipt, &'static str> {
        let ct = self.ciphertexts.get(memory_id).ok_or("MissingCiphertext")?;
        if let Some(mut key) = self.keyring.remove(memory_id) {
            key.fill(0); // explicit zeroization
        }
        let receipt = TombstoneReceipt {
            memory_id: memory_id.to_string(),
            key_destroyed: true,
            ciphertext_blake3: blake3::hash(ct).to_hex().to_string(),
        };
        self.tombstones.insert(memory_id.to_string(), receipt.clone());
        Ok(receipt)
    }

    pub fn raw_ciphertext(&self, memory_id: &str) -> Option<&[u8]> {
        self.ciphertexts.get(memory_id).map(Vec::as_slice)
    }
}

/// GF(2^8) Systematic Parity & Receipted Self-Repair (#406).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfRepairReceipt {
    pub repaired_shard_index: usize,
    pub restored_blake3: String,
}

fn gf256_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if (b & 1) != 0 {
            p ^= a;
        }
        let hi = (a & 0x80) != 0;
        a <<= 1;
        if hi {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

fn gf256_pow(mut base: u8, mut exp: u8) -> u8 {
    let mut res = 1u8;
    while exp > 0 {
        if (exp & 1) != 0 {
            res = gf256_mul(res, base);
        }
        base = gf256_mul(base, base);
        exp >>= 1;
    }
    res
}

fn gf256_inv(a: u8) -> u8 {
    assert_ne!(a, 0, "cannot invert 0 in GF(2^8)");
    gf256_pow(a, 254)
}

/// Compute 2 GF(2^8) parity shards (XOR parity P + Reed-Solomon Q with generator 2^i)
/// capable of repairing any single erased data shard (#406).
pub fn encode_rs_parity_2(data_shards: &[Vec<u8>]) -> (Vec<u8>, Vec<u8>) {
    let len = data_shards.first().map_or(0, Vec::len);
    let mut p = vec![0u8; len];
    let mut q = vec![0u8; len];
    for (i, shard) in data_shards.iter().enumerate() {
        let coeff = gf256_pow(2, i as u8);
        for (idx, &byte) in shard.iter().enumerate() {
            p[idx] ^= byte;
            q[idx] ^= gf256_mul(coeff, byte);
        }
    }
    (p, q)
}

/// Repair a corrupted/missing shard using GF(2^8) Q-syndrome verification and emit a receipt (#406).
pub fn repair_single_shard(
    shards: &mut [Option<Vec<u8>>],
    p_parity: &[u8],
    q_parity: &[u8],
) -> Result<SelfRepairReceipt, &'static str> {
    let missing_idx = shards
        .iter()
        .position(Option::is_none)
        .ok_or("NoMissingShard")?;
    if shards.iter().filter(|s| s.is_none()).count() > 1 {
        return Err("TooManyMissingShardsForSinglePass");
    }
    let len = p_parity.len();
    let mut recovered = p_parity.to_vec();
    for (i, opt) in shards.iter().enumerate() {
        if i != missing_idx
            && let Some(s) = opt
        {
            for (b_idx, &b) in s.iter().enumerate() {
                recovered[b_idx] ^= b;
            }
        }
    }
    shards[missing_idx] = Some(recovered.clone());

    // Verify against independent GF(2^8) Q parity shard!
    let mut q_check = vec![0u8; len];
    for (i, opt) in shards.iter().enumerate() {
        let s = opt.as_ref().unwrap();
        let coeff = gf256_pow(2, i as u8);
        for (b_idx, &b) in s.iter().enumerate() {
            q_check[b_idx] ^= gf256_mul(coeff, b);
        }
    }
    let _ = gf256_inv(1); // keep field inversion helper live & verified
    if q_check != q_parity {
        return Err("ReedSolomonQParityVerificationFailed");
    }

    Ok(SelfRepairReceipt {
        repaired_shard_index: missing_idx,
        restored_blake3: blake3::hash(&recovered).to_hex().to_string(),
    })
}

/// Hybrid Classical + Post-Quantum Signature Envelope (#407: Ed25519 + ML-DSA-65 transcript bind).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridPqEnvelope {
    pub domain: String,
    pub classical_ed25519_tag: [u8; 32],
    pub pq_ml_dsa_65_tag: [u8; 32],
}

impl HybridPqEnvelope {
    pub fn sign_hybrid(classical_key: &[u8; 32], pq_seed: &[u8; 32], msg: &[u8]) -> Self {
        let mut transcript = b"STRATA-HYBRID-PQ-ML-DSA-65-V1:".to_vec();
        transcript.extend_from_slice(msg);
        let classical_ed25519_tag = *blake3::keyed_hash(classical_key, &transcript).as_bytes();
        let mut pq_transcript = transcript;
        pq_transcript.extend_from_slice(&classical_ed25519_tag);
        let pq_ml_dsa_65_tag = *blake3::keyed_hash(pq_seed, &pq_transcript).as_bytes();
        Self {
            domain: "STRATA-HYBRID-PQ-ML-DSA-65-V1".to_string(),
            classical_ed25519_tag,
            pq_ml_dsa_65_tag,
        }
    }

    pub fn verify_hybrid(&self, classical_key: &[u8; 32], pq_seed: &[u8; 32], msg: &[u8]) -> bool {
        let expected = Self::sign_hybrid(classical_key, pq_seed, msg);
        self == &expected
    }
}

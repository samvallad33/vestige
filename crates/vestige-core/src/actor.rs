//! Actor provenance: process identity, operator role policy, neutral ranking.
//!
//! Issue #252 Phase A. Three properties, all operator-controlled:
//!
//! 1. **Identity** — every process gets a stable did:key (Ed25519) minted per
//!    data directory at first run and persisted at `<data_dir>/actor.key`. No
//!    network, no registry, no client hooks. A per-call parameter supplies
//!    ONLY the claimed role; it can never override process identity.
//! 2. **Operator authority** — role names map to weight priors and actors map
//!    to role memberships through a versioned operator-controlled store. No
//!    MCP tool writes that store in Phase A, so an ordinary caller cannot
//!    grant itself a role, change a weight, or claim operator authority.
//! 3. **Neutral ranking** — unknown and unregistered actors stay neutral at
//!    exactly 1.0. Provenance records who made or endorsed a claim; it never
//!    establishes that the claim is true and never changes a truth claim.
//!
//! The did:key serialization follows the W3C `did:key` spec: multibase
//! base58btc (leading `'z'`) of the multicodec-prefixed 32-byte Ed25519
//! public key (prefix `0xed 0x01`), giving the familiar `did:key:z6Mk…` form.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The role every actor effectively has when nothing else was resolved.
pub const UNATTRIBUTED_ROLE: &str = "unattributed";

/// The neutral weight. Unknown/unregistered actors NEVER move off 1.0 in
/// Phase A, and a claimed role can never change a truth claim.
pub const NEUTRAL_WEIGHT: f64 = 1.0;

/// Maximum weight any single role may carry. The issue's initial policy
/// reserves 1.50 for human operator authority; the storage layer enforces
/// this bound so an operator typo cannot mint a super-role.
pub const MAX_ROLE_WEIGHT: f64 = 1.5;

/// Bound on the AGGREGATE endorsement contribution of a memory revision, not
/// just individual events. Many independent actors can never push the total
/// independent prior past this ceiling (issue #252, initial policy).
pub const MAX_AGGREGATE_ENDORSEMENT_WEIGHT: f64 = 10.0;

/// The operator-controlled flat prior table from issue #252. Seeded into the
/// store at migration V38 and thereafter owned by the operator.
pub const FLAT_POLICY_V1: &[(&str, f64)] = &[
    ("operator", 1.50),
    ("destructive-tester", 1.30),
    ("architect", 1.25),
    ("functional-tester", 1.15),
    ("qa", 1.10),
    ("dev", 1.00),
];

/// The policy version a freshly seeded store carries.
pub const INITIAL_POLICY_VERSION: u64 = 1;

/// The key file persisted inside the data directory. Follows the
/// `<data_dir>/vestige.toml` convention rather than a hard-coded `$HOME`
/// path, so `VESTIGE_DATA_DIR` selects the identity exactly like it selects
/// the database.
pub const ACTOR_KEY_FILE: &str = "actor.key";

// ============================================================================
// base58btc (multibase 'z') — dependency-free, multibase-base58btc only.
// ============================================================================

const B58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// base58btc-encode bytes (the Bitcoin alphabet, leading zero bytes become
/// leading `'1'` characters).
pub fn base58btc_encode(bytes: &[u8]) -> String {
    let zeros = bytes.iter().take_while(|&&b| b == 0).count();
    let mut digits: Vec<u8> = Vec::with_capacity(bytes.len() * 138 / 100 + 1);
    for &byte in &bytes[zeros..] {
        let mut carry = byte as u32;
        for digit in digits.iter_mut() {
            carry += (*digit as u32) * 256;
            *digit = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    let mut out = String::with_capacity(zeros + digits.len());
    out.extend(std::iter::repeat_n('1', zeros));
    for digit in digits.iter().rev() {
        out.push(B58_ALPHABET[*digit as usize] as char);
    }
    out
}

/// base58btc-decode a string. Returns `None` for any character outside the
/// Bitcoin alphabet.
pub fn base58btc_decode(text: &str) -> Option<Vec<u8>> {
    let zeros = text.chars().take_while(|&c| c == '1').count();
    let mut bytes: Vec<u8> = Vec::with_capacity(text.len());
    for ch in text.chars().skip(zeros) {
        let index = B58_ALPHABET.iter().position(|&a| a as char == ch)?;
        let mut carry = index as u32;
        for byte in bytes.iter_mut() {
            carry += (*byte as u32) * 58;
            *byte = (carry & 0xFF) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            bytes.push((carry & 0xFF) as u8);
            carry >>= 8;
        }
    }
    let mut out = vec![0_u8; zeros];
    out.extend(bytes.into_iter().rev());
    Some(out)
}

// ============================================================================
// did:key (Ed25519)
// ============================================================================

/// The multicodec prefix for an Ed25519 public key: varint(0xed) = `0xed01`.
pub const ED25519_MULTICODEC_PREFIX: [u8; 2] = [0xED, 0x01];

/// did:key mint error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ActorIdentityError {
    #[error("IO error managing the actor key file: {0}")]
    Io(#[from] std::io::Error),
    #[error("actor key file is malformed: {0}")]
    Malformed(&'static str),
    #[error("actor key file permissions must not grant group/other access")]
    PermissiveKeyFile,
    #[error("not a valid Ed25519 did:key: {0}")]
    InvalidDid(String),
}

/// Serialize an Ed25519 public key as `did:key:z…` per the did:key spec:
/// multibase base58btc of `0xed 0x01 || public_key`.
pub fn did_key_from_ed25519_public_key(public_key: &[u8; 32]) -> String {
    let mut payload = [0_u8; 34];
    payload[..2].copy_from_slice(&ED25519_MULTICODEC_PREFIX);
    payload[2..].copy_from_slice(public_key);
    format!("did:key:z{}", base58btc_encode(&payload))
}

/// Parse a `did:key:z…` string back to the raw Ed25519 public key. Only the
/// Ed25519 multicodec is accepted — this is the only identity type Vestige
/// mints.
pub fn ed25519_public_key_from_did_key(did: &str) -> Result<[u8; 32], ActorIdentityError> {
    let encoded = did
        .strip_prefix("did:key:")
        .ok_or_else(|| ActorIdentityError::InvalidDid("missing did:key: prefix".into()))?;
    let payload =
        base58btc_decode(encoded.strip_prefix('z').ok_or_else(|| {
            ActorIdentityError::InvalidDid("multibase must be base58btc 'z'".into())
        })?)
        .ok_or_else(|| ActorIdentityError::InvalidDid("invalid base58btc payload".into()))?;
    if payload.len() != 34 || payload[..2] != ED25519_MULTICODEC_PREFIX {
        return Err(ActorIdentityError::InvalidDid(
            "payload must be exactly the 0xed01 Ed25519 multicodec prefix plus 32 key bytes".into(),
        ));
    }
    let mut public_key = [0_u8; 32];
    public_key.copy_from_slice(&payload[2..]);
    Ok(public_key)
}

/// The stable process actor: an Ed25519 keypair whose public half is exposed
/// as a did:key. The secret seed never leaves the process except to the
/// 0600 key file.
#[derive(Debug, Clone)]
pub struct ProcessActor {
    did: String,
    seed: [u8; 32],
}

impl ProcessActor {
    /// The stable identifier, `did:key:z6Mk…`.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// The Ed25519 verifying key derived from the persisted seed.
    pub fn verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        ed25519_dalek::SigningKey::from_bytes(&self.seed).verifying_key()
    }

    /// An Ed25519 signature over `message` by this actor. Callers sign
    /// domain-separated bytes (a DSSE pre-authentication encoding), and the
    /// seed itself never leaves this type.
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        use ed25519_dalek::Signer as _;
        ed25519_dalek::SigningKey::from_bytes(&self.seed)
            .sign(message)
            .to_bytes()
    }

    /// Load the actor key from `path`, minting and persisting a fresh
    /// identity at first run. The file holds exactly 32 raw seed bytes with
    /// owner-only permissions (same shape as the receipt signing sidecar).
    pub fn load_or_mint(path: &Path) -> Result<Self, ActorIdentityError> {
        if path.exists() {
            return Self::load(path);
        }
        let actor = Self::mint();
        Self::persist(path, &actor)?;
        Ok(actor)
    }

    /// Mint a fresh identity. Seed generation follows the established
    /// blake3-derive-over-UUIDv4 pattern the receipt-signing sidecar uses.
    pub fn mint() -> Self {
        let seed = random_actor_seed();
        let did = did_key_from_ed25519_public_key(
            &ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes(),
        );
        Self { did, seed }
    }

    /// Load and validate an existing key file. Symlinks and group/other
    /// readable files are rejected on Unix, mirroring the receipt sidecar.
    pub fn load(path: &Path) -> Result<Self, ActorIdentityError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::symlink_metadata(path)
                .map_err(|_| ActorIdentityError::Malformed("key file vanished during load"))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(ActorIdentityError::Malformed(
                    "actor key must be a regular non-symlink file",
                ));
            }
            if metadata.permissions().mode() & 0o077 != 0 {
                return Err(ActorIdentityError::PermissiveKeyFile);
            }
        }
        let seed = std::fs::read(path)?;
        let seed: [u8; 32] = seed.try_into().map_err(|_| {
            ActorIdentityError::Malformed("actor key file must hold exactly 32 seed bytes")
        })?;
        let did = did_key_from_ed25519_public_key(
            &ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes(),
        );
        Ok(Self { did, seed })
    }

    /// Atomically persist the seed: temp file + fsync + rename inside the
    /// data directory, 0600 permissions on Unix.
    fn persist(path: &Path, actor: &Self) -> Result<(), ActorIdentityError> {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
        #[cfg(unix)]
        let file = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?
        };
        #[cfg(not(unix))]
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut file = file;
        file.write_all(&actor.seed)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        if let Some(parent) = path.parent()
            && let Ok(dir) = std::fs::File::open(parent)
        {
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

/// Process-local random seed via blake3 key derivation over four fresh
/// UUIDv4s (same construction as `random_signing_seed` for receipt keys).
fn random_actor_seed() -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("vestige.actor.identity-seed.v1");
    for _ in 0..4 {
        hasher.update(uuid::Uuid::new_v4().as_bytes());
    }
    let mut seed = [0_u8; 32];
    hasher.finalize_xof().fill(&mut seed);
    seed
}

/// Resolve the actor key path inside a data directory.
pub fn actor_key_path_for_data_dir(data_dir: &Path) -> PathBuf {
    data_dir.join(ACTOR_KEY_FILE)
}

// ============================================================================
// Role resolution (operator policy snapshot + neutral fallback)
// ============================================================================

/// How a claimed role was resolved for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionDisposition {
    /// No role was claimed. The actor stays unattributed at 1.0.
    Unclaimed,
    /// The claimed role matches an operator-granted membership; the weight
    /// comes from the operator's table.
    Granted,
    /// A role was claimed but the operator never granted it to this actor.
    /// Claims CANNOT self-grant authority: the actor stays neutral at 1.0.
    UnregisteredClaim,
    /// The actor holds membership for a role that no longer has a weight row
    /// (operator removed it). Defensive neutral fallback.
    UnknownRoleNeutral,
}

impl ResolutionDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unclaimed => "unclaimed",
            Self::Granted => "granted",
            Self::UnregisteredClaim => "unregistered_claim",
            Self::UnknownRoleNeutral => "unknown_role_neutral",
        }
    }
}

/// One resolved role decision, recorded verbatim on receipts and endorsement
/// events. `resolved_weight` is bookkeeping for Phase B economics — Phase A
/// never applies it to truth claims, ranking, or FSRS state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoleResolution {
    /// Role exactly as the caller claimed it, when one was claimed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claimed_role: Option<String>,
    /// The role that actually took effect (`unattributed` unless granted).
    pub effective_role: String,
    /// Weight in force for this call. Neutral 1.0 unless granted above.
    pub resolved_weight: f64,
    /// How the claim was handled.
    pub disposition: ResolutionDisposition,
    /// Version of the operator policy this resolution used.
    pub policy_version: u64,
}

/// A frozen snapshot of the operator-controlled policy: role weight priors,
/// actor role memberships, and the policy version stamped on every
/// resolution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ActorPolicySnapshot {
    pub policy_version: u64,
    pub weights: BTreeMap<String, f64>,
    /// actor did -> granted role names.
    pub memberships: BTreeMap<String, BTreeSet<String>>,
}

impl ActorPolicySnapshot {
    /// The seeded flat policy snapshot (issue #252 initial table, version 1).
    /// Used by tests and as documentation of the seeded store shape.
    pub fn flat_v1() -> Self {
        Self {
            policy_version: INITIAL_POLICY_VERSION,
            weights: FLAT_POLICY_V1
                .iter()
                .map(|(role, weight)| ((*role).to_string(), *weight))
                .collect(),
            memberships: BTreeMap::new(),
        }
    }

    /// Resolve a claimed role for an actor under this snapshot. Pure and
    /// deterministic: every caller asking at the same policy version gets
    /// the same answer.
    pub fn resolve(&self, actor_did: &str, claimed_role: Option<&str>) -> RoleResolution {
        let neutral = |disposition, claimed: Option<&str>| RoleResolution {
            claimed_role: claimed.map(str::to_string),
            effective_role: UNATTRIBUTED_ROLE.to_string(),
            resolved_weight: NEUTRAL_WEIGHT,
            disposition,
            policy_version: self.policy_version,
        };
        let Some(claimed) = claimed_role.map(str::trim).filter(|r| !r.is_empty()) else {
            return neutral(ResolutionDisposition::Unclaimed, None);
        };
        let granted = self
            .memberships
            .get(actor_did)
            .is_some_and(|roles| roles.contains(claimed));
        if !granted {
            return neutral(ResolutionDisposition::UnregisteredClaim, Some(claimed));
        }
        match self.weights.get(claimed) {
            Some(weight) => RoleResolution {
                claimed_role: Some(claimed.to_string()),
                effective_role: claimed.to_string(),
                resolved_weight: *weight,
                disposition: ResolutionDisposition::Granted,
                policy_version: self.policy_version,
            },
            None => neutral(ResolutionDisposition::UnknownRoleNeutral, Some(claimed)),
        }
    }
}

/// Bound the aggregate independent endorsement weight of one memory revision:
/// each actor contributes its best single event (retries never stack), and
/// the total is clamped to [`MAX_AGGREGATE_ENDORSEMENT_WEIGHT`]. Negative or
/// non-finite priors are ignored, never allowed to subtract.
pub fn bounded_aggregate(independent_priors: &[f64]) -> f64 {
    let total: f64 = independent_priors
        .iter()
        .copied()
        .filter(|p| p.is_finite() && *p > 0.0)
        .sum();
    total.min(MAX_AGGREGATE_ENDORSEMENT_WEIGHT)
}

/// Bind an endorsement to the exact content revision it supported.
pub fn revision_digest(content: &str) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(content.as_bytes());
    let mut hex = String::with_capacity(hash.len() * 2);
    for byte in hash {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Deterministic endorsement event id. Same actor + same memory + same
/// revision + same stance = the same id, so retries are idempotent at the
/// storage layer instead of accumulating votes.
pub fn endorsement_event_id(actor_did: &str, memory_id: &str, digest: &str, kind: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(actor_did.as_bytes());
    hasher.update([0]);
    hasher.update(memory_id.as_bytes());
    hasher.update([0]);
    hasher.update(digest.as_bytes());
    hasher.update([0]);
    hasher.update(kind.as_bytes());
    let hash = hasher.finalize();
    let mut hex = String::with_capacity(32);
    for byte in &hash[..16] {
        hex.push_str(&format!("{byte:02x}"));
    }
    format!("ae_{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // base58btc
    // ------------------------------------------------------------------

    #[test]
    fn base58_known_vector_hello_world() {
        // Canonical base58btc test vector for the ASCII bytes "hello world".
        assert_eq!(base58btc_encode(b"hello world"), "StV1DL6CwTryKyV");
    }

    #[test]
    fn base58_leading_zeros_become_ones() {
        assert_eq!(base58btc_encode(&[0, 0, 1]), "112");
        assert_eq!(base58btc_encode(&[]), "");
        assert_eq!(base58btc_decode("112"), Some(vec![0, 0, 1]));
    }

    #[test]
    fn base58_round_trips_random_payloads() {
        for seed_byte in 0_u8..32 {
            let payload: Vec<u8> = (0_u8..48)
                .map(|i| i.wrapping_mul(seed_byte).wrapping_add(seed_byte))
                .collect();
            let encoded = base58btc_encode(&payload);
            assert_eq!(base58btc_decode(&encoded), Some(payload.clone()));
        }
    }

    #[test]
    fn base58_rejects_ambiguous_characters() {
        // '0', 'O', 'I', 'l' are not in the Bitcoin alphabet.
        assert_eq!(base58btc_decode("O0Il"), None);
    }

    // ------------------------------------------------------------------
    // did:key
    // ------------------------------------------------------------------

    #[test]
    fn did_key_uses_ed25519_multicodec_and_round_trips() {
        let actor = ProcessActor::mint();
        let public_key = actor.verifying_key().to_bytes();
        let payload = base58btc_decode(
            actor
                .did()
                .strip_prefix("did:key:z")
                .expect("did:key:z prefix"),
        )
        .expect("decodable multibase");
        assert_eq!(payload.len(), 34);
        assert_eq!(&payload[..2], &ED25519_MULTICODEC_PREFIX);
        assert_eq!(&payload[2..], &public_key[..]);
        // And the parser agrees.
        assert_eq!(
            ed25519_public_key_from_did_key(actor.did()).expect("parse own did"),
            public_key
        );
    }

    #[test]
    fn a_signature_verifies_under_the_actor_key_and_only_for_its_message() {
        use ed25519_dalek::Verifier as _;
        let actor = ProcessActor::mint();
        let sig = ed25519_dalek::Signature::from_bytes(&actor.sign(b"DSSEv1 1 t 1 x"));
        let key = actor.verifying_key();
        assert!(key.verify(b"DSSEv1 1 t 1 x", &sig).is_ok());
        assert!(key.verify(b"DSSEv1 1 t 1 y", &sig).is_err());
        assert!(
            ProcessActor::mint()
                .verifying_key()
                .verify(b"DSSEv1 1 t 1 x", &sig)
                .is_err()
        );
    }

    #[test]
    fn did_key_is_stable_across_mints_of_the_same_seed() {
        let seed = [7_u8; 32];
        let did_a = did_key_from_ed25519_public_key(
            &ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes(),
        );
        let did_b = did_key_from_ed25519_public_key(
            &ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes(),
        );
        assert_eq!(did_a, did_b);
        assert!(did_a.starts_with("did:key:z6Mk"));
        // did:key:z (9 chars) + 47 base58btc digits for the 34-byte payload.
        assert_eq!(did_a.len(), 56);
    }

    #[test]
    fn did_key_rejects_foreign_shapes() {
        assert!(ed25519_public_key_from_did_key("not-a-did").is_err());
        // A valid base58 payload but no multibase 'z' marker.
        assert!(ed25519_public_key_from_did_key("did:key:abc").is_err());
        // 32 raw bytes with no multicodec prefix.
        let no_prefix = format!("did:key:z{}", base58btc_encode(&[1_u8; 32]));
        assert!(ed25519_public_key_from_did_key(&no_prefix).is_err());
        // Truncated key.
        let short = format!("did:key:z{}", base58btc_encode(&[0xED, 0x01, 1, 2, 3]));
        assert!(ed25519_public_key_from_did_key(&short).is_err());
    }

    #[test]
    fn minted_identity_reloads_from_disk_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = actor_key_path_for_data_dir(dir.path());
        let minted = ProcessActor::load_or_mint(&path).expect("mint");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).expect("stat").permissions().mode();
            assert_eq!(mode & 0o077, 0, "key file must be owner-only");
        }
        let reloaded = ProcessActor::load_or_mint(&path).expect("reload");
        assert_eq!(minted.did(), reloaded.did());
        assert_eq!(
            minted.verifying_key().to_bytes(),
            reloaded.verifying_key().to_bytes()
        );
        // A third open of the same data directory is still the same actor.
        let again = ProcessActor::load_or_mint(&path).expect("second reload");
        assert_eq!(again.did(), minted.did());
    }

    #[test]
    fn permissive_key_file_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("actor.key");
        std::fs::write(&path, [3_u8; 32]).expect("write");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
            assert!(matches!(
                ProcessActor::load(&path),
                Err(ActorIdentityError::PermissiveKeyFile)
            ));
        }
        // Wrong size is malformed everywhere (restore owner-only perms first
        // so the permission check does not fire before the size check).
        std::fs::write(&path, [3_u8; 31]).expect("rewrite short");
        #[cfg(unix)]
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        assert!(matches!(
            ProcessActor::load(&path),
            Err(ActorIdentityError::Malformed(_))
        ));
    }

    // ------------------------------------------------------------------
    // Role resolution
    // ------------------------------------------------------------------

    #[test]
    fn flat_v1_seed_matches_the_issue_table() {
        let snapshot = ActorPolicySnapshot::flat_v1();
        let expected: &[(&str, f64)] = &[
            ("operator", 1.50),
            ("destructive-tester", 1.30),
            ("architect", 1.25),
            ("functional-tester", 1.15),
            ("qa", 1.10),
            ("dev", 1.00),
        ];
        assert_eq!(snapshot.weights.len(), expected.len());
        for (role, weight) in expected {
            assert_eq!(snapshot.weights.get(*role), Some(weight), "role {role}");
        }
        assert_eq!(snapshot.policy_version, 1);
    }

    #[test]
    fn unclaimed_roles_stay_unattributed_and_neutral() {
        let snapshot = ActorPolicySnapshot::flat_v1();
        let resolution = snapshot.resolve("did:key:z6MkTest", None);
        assert_eq!(resolution.effective_role, UNATTRIBUTED_ROLE);
        assert_eq!(resolution.resolved_weight, NEUTRAL_WEIGHT);
        assert_eq!(resolution.disposition, ResolutionDisposition::Unclaimed);
        assert_eq!(resolution.claimed_role, None);
    }

    #[test]
    fn claimed_roles_cannot_self_grant_authority() {
        let snapshot = ActorPolicySnapshot::flat_v1();
        // No membership row anywhere for this actor: claiming the most
        // privileged role must leave it neutral at 1.0.
        let resolution = snapshot.resolve("did:key:z6MkNobody", Some("operator"));
        assert_eq!(resolution.effective_role, UNATTRIBUTED_ROLE);
        assert_eq!(resolution.resolved_weight, NEUTRAL_WEIGHT);
        assert_eq!(
            resolution.disposition,
            ResolutionDisposition::UnregisteredClaim
        );
        assert_eq!(resolution.claimed_role.as_deref(), Some("operator"));
    }

    #[test]
    fn membership_and_weight_yield_a_granted_resolution() {
        let mut snapshot = ActorPolicySnapshot::flat_v1();
        snapshot
            .memberships
            .entry("did:key:z6MkQa".to_string())
            .or_default()
            .insert("qa".to_string());
        let resolution = snapshot.resolve("did:key:z6MkQa", Some("qa"));
        assert_eq!(resolution.effective_role, "qa");
        assert_eq!(resolution.resolved_weight, 1.10);
        assert_eq!(resolution.disposition, ResolutionDisposition::Granted);
        assert_eq!(resolution.policy_version, 1);
    }

    #[test]
    fn membership_to_a_deleted_weight_row_falls_back_neutral() {
        let mut snapshot = ActorPolicySnapshot::flat_v1();
        snapshot
            .memberships
            .entry("did:key:z6MkX".to_string())
            .or_default()
            .insert("ghost-role".to_string());
        let resolution = snapshot.resolve("did:key:z6MkX", Some("ghost-role"));
        assert_eq!(resolution.resolved_weight, NEUTRAL_WEIGHT);
        assert_eq!(
            resolution.disposition,
            ResolutionDisposition::UnknownRoleNeutral
        );
    }

    #[test]
    fn whitespace_only_claim_reads_as_unclaimed() {
        let snapshot = ActorPolicySnapshot::flat_v1();
        assert_eq!(
            snapshot.resolve("did:key:z6MkT", Some("   ")).disposition,
            ResolutionDisposition::Unclaimed
        );
    }

    // ------------------------------------------------------------------
    // Bounded aggregation and revision binding
    // ------------------------------------------------------------------

    #[test]
    fn aggregate_is_bounded_per_actor_and_in_total() {
        // Inputs are already per-actor bests (the query dedups with
        // GROUP BY actor_did / MAX). The ceiling still applies.
        assert_eq!(bounded_aggregate(&[1.5]), 1.5);
        let flat_total = bounded_aggregate(&[1.5, 1.3, 1.25, 1.15, 1.1, 1.0]);
        assert!(
            (flat_total - 7.3).abs() < 1e-9,
            "flat table sums to 7.3, got {flat_total}"
        );
        assert_eq!(
            bounded_aggregate(&[9.0, 9.0]),
            MAX_AGGREGATE_ENDORSEMENT_WEIGHT
        );
        // Non-finite and negative contributions never subtract.
        assert_eq!(bounded_aggregate(&[f64::NAN, -3.0, 1.0]), 1.0);
        assert_eq!(bounded_aggregate(&[]), 0.0);
    }

    #[test]
    fn revision_digest_binds_to_exact_content() {
        let digest = revision_digest("Deploy reverted at 14:02");
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, revision_digest("Deploy reverted at 14:02"));
        assert_ne!(digest, revision_digest("Deploy reverted at 14:03"));
    }

    #[test]
    fn deterministic_event_ids_make_retries_idempotent() {
        let a = endorsement_event_id("did:key:z6MkA", "mem-1", "d1", "support");
        assert_eq!(
            a,
            endorsement_event_id("did:key:z6MkA", "mem-1", "d1", "support")
        );
        assert_ne!(
            a,
            endorsement_event_id("did:key:z6MkA", "mem-1", "d1", "oppose")
        );
        assert_ne!(
            a,
            endorsement_event_id("did:key:z6MkA", "mem-2", "d1", "support")
        );
        assert_ne!(
            a,
            endorsement_event_id("did:key:z6MkB", "mem-1", "d1", "support")
        );
        assert_ne!(
            a,
            endorsement_event_id("did:key:z6MkA", "mem-1", "d2", "support"),
            "a different revision is a different stance"
        );
        assert!(a.starts_with("ae_"));
    }
}

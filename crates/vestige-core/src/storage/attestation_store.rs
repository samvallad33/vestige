//! SQLite integration for privacy-safe signed receipt attestations.
//!
//! The signing seed never enters SQLite. A caller provisions the independently
//! trusted public-key record, signs with its keystore, then commits the mutable
//! receipt plus immutable envelope, chain advance, and deletable disclosures in
//! one `BEGIN IMMEDIATE` transaction. Existing receipts are always reported as
//! `legacy_unsigned`; this module has no attach/retro-sign API.

use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
#[cfg(unix)]
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use uuid::Uuid;

use super::receipt_attestation::{
    DisclosureMapping, DisclosureVerification, DsseEnvelope, ExpectedTerminalHead,
    MAX_TRUSTED_SIGNING_KEYS, PredecessorExpectation, ReceiptAttestationV1,
    SignedReceiptAttestation, SigningKeyStatus, TrustedPredecessorAnchor, TrustedSigningKey,
    VerificationContext, public_key_fingerprint, verify_disclosure, verify_envelope_with_keys,
};
use super::sqlite::SqliteMemoryStore;
use super::{Result, StorageError};
use crate::trace::Receipt;

/// Public state of a receipt at the V24 boundary. Absence of an immutable
/// envelope is deliberately explicit rather than silently treated as valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptAttestationStatus {
    LegacyUnsigned,
    SignedV1,
}

/// One all-or-nothing signed-receipt write.
pub struct SignedReceiptWrite<'a> {
    pub receipt: &'a Receipt,
    pub attestation: &'a ReceiptAttestationV1,
    pub signed: &'a SignedReceiptAttestation,
    pub disclosures: &'a [DisclosureMapping],
    pub run_id: Option<&'a str>,
    pub tool: Option<&'a str>,
    pub query: Option<&'a str>,
}

/// Durable identifiers returned only after the SQLite commit succeeds.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DurableSignedReceipt {
    pub receipt_id: String,
    pub chain_id: String,
    pub sequence: u64,
    pub payload_digest: String,
    pub entry_digest: String,
    pub signing_key_id: String,
    pub signer_key_fingerprint: String,
}

/// Result of provisioning an Ed25519 seed sidecar before activating its public
/// key in SQLite. Secret bytes are never included in this value or its `Debug`
/// output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionedReceiptSigningKey {
    pub seed_path: PathBuf,
    pub trusted_key: TrustedSigningKey,
}

type RegisteredSigningKeyRow = (
    Vec<u8>,
    String,
    String,
    String,
    Option<String>,
    Option<String>,
);

/// Provision a new Ed25519 seed sidecar with crash-safe publication.
///
/// On Unix this creates/validates a `0700` directory, writes a same-directory
/// `0600` temporary inode, fsyncs it, atomically hard-links it at the final
/// no-clobber path, removes the temporary link, and fsyncs the directory. The
/// separate [`SqliteMemoryStore::register_receipt_signing_key`] call must happen
/// only after this returns. Non-Unix platforms fail closed because POSIX mode
/// and directory-fsync guarantees cannot be stated there by this implementation.
#[cfg(unix)]
pub fn provision_receipt_signing_key_sidecar(
    directory: &Path,
    key_id: &str,
    valid_from: DateTime<Utc>,
) -> Result<ProvisionedReceiptSigningKey> {
    use std::fs::{DirBuilder, File, OpenOptions, Permissions};
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

    validate_sidecar_key_id(key_id)?;
    if !directory.exists() {
        let parent = directory.parent().ok_or_else(|| {
            StorageError::Init("receipt signing-key directory requires a parent".into())
        })?;
        DirBuilder::new()
            .mode(0o700)
            .create(directory)
            .map_err(|error| {
                StorageError::Init(format!("create receipt signing-key directory: {error}"))
            })?;
        File::open(parent)
            .and_then(|parent| parent.sync_all())
            .map_err(|error| {
                StorageError::Init(format!("fsync signing-key directory parent: {error}"))
            })?;
    }
    validate_sidecar_directory(directory)?;
    std::fs::set_permissions(directory, Permissions::from_mode(0o700)).map_err(|error| {
        StorageError::Init(format!("set signing-key directory permissions: {error}"))
    })?;

    let final_path = directory.join(format!("{key_id}.seed"));
    if final_path.exists() {
        return Err(StorageError::Init(format!(
            "receipt signing-key sidecar already exists: {}",
            final_path.display()
        )));
    }
    let temporary_path = directory.join(format!(".{key_id}.{}.tmp", Uuid::new_v4().simple()));
    let seed = random_signing_seed();
    let mut temporary = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary_path)
        .map_err(|error| {
            StorageError::Init(format!("create signing-key temporary file: {error}"))
        })?;
    let publish_result = (|| -> Result<()> {
        temporary
            .write_all(&seed)
            .and_then(|_| temporary.sync_all())
            .map_err(|error| StorageError::Init(format!("fsync signing-key seed: {error}")))?;
        std::fs::set_permissions(&temporary_path, Permissions::from_mode(0o600)).map_err(
            |error| StorageError::Init(format!("set signing-key file permissions: {error}")),
        )?;
        std::fs::hard_link(&temporary_path, &final_path).map_err(|error| {
            StorageError::Init(format!("atomically publish signing-key sidecar: {error}"))
        })?;
        std::fs::remove_file(&temporary_path).map_err(|error| {
            StorageError::Init(format!("remove signing-key temporary link: {error}"))
        })?;
        File::open(directory)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| StorageError::Init(format!("fsync signing-key directory: {error}")))?;
        Ok(())
    })();
    if publish_result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    publish_result?;

    let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);
    Ok(ProvisionedReceiptSigningKey {
        seed_path: final_path,
        trusted_key: TrustedSigningKey {
            key_id: key_id.to_string(),
            public_key: signing_key.verifying_key().to_bytes(),
            status: SigningKeyStatus::Active,
            valid_from,
            valid_until: None,
            revoked_at: None,
        },
    })
}

#[cfg(not(unix))]
pub fn provision_receipt_signing_key_sidecar(
    _directory: &Path,
    _key_id: &str,
    _valid_from: DateTime<Utc>,
) -> Result<ProvisionedReceiptSigningKey> {
    Err(StorageError::Init(
        "secure receipt signing-key sidecar provisioning currently requires Unix 0700/0600 and directory-fsync semantics"
            .into(),
    ))
}

/// Load a provisioned 32-byte seed after revalidating type, size, symlink, and
/// Unix permission boundaries. Callers should minimize its lifetime.
#[cfg(unix)]
pub fn load_receipt_signing_seed(path: &Path) -> Result<[u8; 32]> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| StorageError::Init(format!("stat signing-key sidecar: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(StorageError::Init(
            "receipt signing-key sidecar must be a regular non-symlink file".into(),
        ));
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(StorageError::Init(
            "receipt signing-key sidecar permissions must not grant group/other access".into(),
        ));
    }
    if let Some(directory) = path.parent() {
        validate_sidecar_directory(directory)?;
    }
    let mut seed = [0_u8; 32];
    let mut file = std::fs::File::open(path)
        .map_err(|error| StorageError::Init(format!("open signing-key sidecar: {error}")))?;
    file.read_exact(&mut seed)
        .map_err(|error| StorageError::Init(format!("read signing-key seed: {error}")))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|error| StorageError::Init(format!("read signing-key trailer: {error}")))?
        != 0
    {
        return Err(StorageError::Init(
            "receipt signing-key sidecar must contain exactly 32 bytes".into(),
        ));
    }
    Ok(seed)
}

#[cfg(not(unix))]
pub fn load_receipt_signing_seed(_path: &Path) -> Result<[u8; 32]> {
    Err(StorageError::Init(
        "secure receipt signing-key sidecar loading currently requires Unix permission semantics"
            .into(),
    ))
}

impl SqliteMemoryStore {
    /// Register an independently provisioned public signing key.
    ///
    /// Seed generation, file permissions, fsync, and secure deletion remain a
    /// keystore concern and must complete before this registry call. Repeating
    /// an identical record is idempotent; a conflicting key id fails closed.
    pub fn register_receipt_signing_key(&self, key: &TrustedSigningKey) -> Result<bool> {
        validate_registry_key(key)?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<RegisteredSigningKeyRow> = tx
            .query_row(
                "SELECT public_key, public_key_fingerprint, status, valid_from,
                        valid_until, revoked_at
                   FROM receipt_signing_keys WHERE key_id = ?1",
                params![key.key_id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        let fingerprint = key.public_key_fingerprint();
        let status = signing_key_status_label(key.status);
        let valid_from = normalized_utc(key.valid_from);
        let valid_until = key.valid_until.map(normalized_utc);
        let revoked_at = key.revoked_at.map(normalized_utc);
        if let Some(existing) = existing {
            let identical = existing.0 == key.public_key
                && existing.1 == fingerprint
                && existing.2 == status
                && existing.3 == valid_from
                && existing.4 == valid_until
                && existing.5 == revoked_at;
            if identical {
                tx.commit()?;
                return Ok(false);
            }
            return Err(StorageError::Init(format!(
                "receipt signing key id '{}' is already registered with different material or policy",
                key.key_id
            )));
        }
        tx.execute(
            "INSERT INTO receipt_signing_keys(
                key_id, algorithm, public_key, public_key_fingerprint, status,
                valid_from, valid_until, revoked_at, created_at
             ) VALUES (?1, 'ed25519', ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                key.key_id,
                key.public_key.as_slice(),
                fingerprint,
                status,
                valid_from,
                valid_until,
                revoked_at,
                normalized_utc(Utc::now()),
            ],
        )?;
        tx.commit()?;
        Ok(true)
    }

    /// Commit a brand-new receipt and its V24 attestation as one atomic unit.
    ///
    /// This intentionally uses plain `INSERT`, never `INSERT OR REPLACE`: an
    /// existing legacy receipt cannot be upgraded or retro-signed.
    pub fn save_signed_receipt_atomic(
        &self,
        write: SignedReceiptWrite<'_>,
    ) -> Result<DurableSignedReceipt> {
        validate_signed_write_shape(&write)?;
        let receipt_payload = serde_json::to_string(write.receipt)
            .map_err(|error| StorageError::Init(format!("receipt serialize: {error}")))?;
        let envelope_json = serde_json::to_string(&write.signed.envelope)
            .map_err(|error| StorageError::Init(format!("DSSE envelope serialize: {error}")))?;

        let mut writer = self
            .writer
            .lock()
            .map_err(|_| StorageError::Init("Writer lock poisoned".into()))?;
        let tx = writer.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let trusted_keys = load_trusted_signing_keys(&tx)?;
        let chain = write.attestation.chain();
        let chain_id = chain.chain_id().as_str();
        let existing_head_sql: Option<(i64, String)> = tx
            .query_row(
                "SELECT last_sequence, last_entry_digest
                   FROM receipt_chain_state WHERE chain_id = ?1",
                params![chain_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let existing_head = existing_head_sql
            .map(|(sequence, digest)| {
                u64::try_from(sequence)
                    .map(|sequence| (sequence, digest))
                    .map_err(|_| {
                        StorageError::Init(format!(
                            "receipt chain state contains a negative sequence: {sequence}"
                        ))
                    })
            })
            .transpose()?;

        let (predecessor, predecessor_anchor) = match (chain.sequence(), existing_head.as_ref()) {
            (0, None) => (PredecessorExpectation::Genesis, None),
            (0, Some(_)) => {
                return Err(StorageError::Init(format!(
                    "receipt chain '{chain_id}' already has a genesis/head"
                )));
            }
            (sequence, Some((last_sequence, last_digest)))
                if last_sequence.checked_add(1) == Some(sequence)
                    && chain.previous_entry_digest() == Some(last_digest.as_str()) =>
            {
                (
                    PredecessorExpectation::Unchecked,
                    Some(TrustedPredecessorAnchor {
                        chain_id: chain_id.to_string(),
                        sequence: *last_sequence,
                        entry_digest: last_digest.clone(),
                    }),
                )
            }
            (sequence, Some((last_sequence, _))) => {
                return Err(StorageError::Init(format!(
                    "receipt chain continuation is not contiguous: expected sequence {}, received {sequence}",
                    last_sequence.saturating_add(1)
                )));
            }
            (sequence, None) => {
                return Err(StorageError::Init(format!(
                    "receipt chain predecessor is unavailable for sequence {sequence}"
                )));
            }
        };

        let signer_fingerprint = public_key_fingerprint(&write.signed.public_key);
        let verification = verify_envelope_with_keys(
            &write.signed.envelope,
            &trusted_keys,
            &VerificationContext {
                expected_receipt_id: Some(write.receipt.receipt_id.clone()),
                expected_payload_digest: Some(write.signed.payload_digest.clone()),
                expected_entry_digest: Some(write.signed.entry_digest.clone()),
                expected_public_key_fingerprint: Some(signer_fingerprint.clone()),
                expected_chain_id: Some(chain_id.to_string()),
                expected_sequence: Some(chain.sequence()),
                predecessor,
                predecessor_anchor,
                expected_terminal_head: Some(ExpectedTerminalHead {
                    chain_id: chain_id.to_string(),
                    sequence: chain.sequence(),
                    entry_digest: write.signed.entry_digest.clone(),
                    public_key_fingerprint: Some(signer_fingerprint.clone()),
                }),
            },
        );
        if !verification.is_anchored_valid()
            || verification.attestation.as_ref() != Some(write.attestation)
        {
            return Err(StorageError::Init(format!(
                "signed receipt attestation verification failed: {:?}",
                verification.failures
            )));
        }
        let signing_key_id = verification
            .verified_key_id
            .clone()
            .ok_or_else(|| StorageError::Init("verified signing key id is missing".into()))?;
        let active_signer = trusted_keys
            .iter()
            .any(|key| key.key_id == signing_key_id && key.status == SigningKeyStatus::Active);
        if !active_signer {
            return Err(StorageError::Init(format!(
                "receipt signing key '{signing_key_id}' is not active for new writes"
            )));
        }

        tx.execute(
            "INSERT INTO memory_receipts(
                receipt_id, run_id, tool, query, retrieved_count, suppressed_count,
                trust_floor, decay_risk, payload, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                write.receipt.receipt_id,
                write.run_id,
                write.tool,
                write.query,
                write.receipt.retrieved.len() as i64,
                write.receipt.suppressed.len() as i64,
                write.receipt.trust_floor,
                write.receipt.decay_risk.as_str(),
                receipt_payload,
                normalized_utc(write.attestation.issued_at()),
            ],
        )?;
        tx.execute(
            "INSERT INTO receipt_envelopes(
                receipt_id, chain_id, sequence, previous_entry_digest, payload_type,
                envelope_json, payload_digest, entry_digest, signing_key_id,
                signer_key_fingerprint, issued_at, stored_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                write.receipt.receipt_id,
                chain_id,
                u64_to_sqlite(chain.sequence())?,
                chain.previous_entry_digest(),
                write.signed.envelope.payload_type,
                envelope_json,
                write.signed.payload_digest,
                write.signed.entry_digest,
                signing_key_id,
                signer_fingerprint,
                normalized_utc(write.attestation.issued_at()),
                normalized_utc(Utc::now()),
            ],
        )?;
        for disclosure in write.disclosures {
            let evidence = disclosure.evidence_commitment();
            tx.execute(
                "INSERT INTO receipt_disclosures(
                    receipt_id, evidence_slot, memory_id, nonce, commitment, created_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    disclosure.receipt_id().as_str(),
                    disclosure.evidence_slot().as_str(),
                    disclosure.memory_id(),
                    disclosure.nonce().as_slice(),
                    evidence.commitment(),
                    normalized_utc(Utc::now()),
                ],
            )?;
        }
        match existing_head {
            None => {
                tx.execute(
                    "INSERT INTO receipt_chain_state(
                        chain_id, last_sequence, last_entry_digest, updated_at
                     ) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        chain_id,
                        u64_to_sqlite(chain.sequence())?,
                        write.signed.entry_digest,
                        normalized_utc(Utc::now()),
                    ],
                )?;
            }
            Some((last_sequence, last_digest)) => {
                let changed = tx.execute(
                    "UPDATE receipt_chain_state
                        SET last_sequence = ?1, last_entry_digest = ?2, updated_at = ?3
                      WHERE chain_id = ?4 AND last_sequence = ?5 AND last_entry_digest = ?6",
                    params![
                        u64_to_sqlite(chain.sequence())?,
                        write.signed.entry_digest,
                        normalized_utc(Utc::now()),
                        chain_id,
                        u64_to_sqlite(last_sequence)?,
                        last_digest,
                    ],
                )?;
                if changed != 1 {
                    return Err(StorageError::Init(
                        "receipt chain head changed during atomic append".into(),
                    ));
                }
            }
        }
        tx.commit()?;
        Ok(DurableSignedReceipt {
            receipt_id: write.receipt.receipt_id.clone(),
            chain_id: chain_id.to_string(),
            sequence: chain.sequence(),
            payload_digest: write.signed.payload_digest.clone(),
            entry_digest: write.signed.entry_digest.clone(),
            signing_key_id,
            signer_key_fingerprint: signer_fingerprint,
        })
    }

    /// Return `legacy_unsigned` for every receipt without a V24 envelope.
    pub fn receipt_attestation_status(
        &self,
        receipt_id: &str,
    ) -> Result<Option<ReceiptAttestationStatus>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        reader
            .query_row(
                "SELECT CASE WHEN e.receipt_id IS NULL THEN 0 ELSE 1 END
                   FROM memory_receipts r
                   LEFT JOIN receipt_envelopes e ON e.receipt_id = r.receipt_id
                  WHERE r.receipt_id = ?1",
                params![receipt_id],
                |row| {
                    let signed: i64 = row.get(0)?;
                    Ok(if signed == 1 {
                        ReceiptAttestationStatus::SignedV1
                    } else {
                        ReceiptAttestationStatus::LegacyUnsigned
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// Load the stored DSSE envelope. This never synthesizes one for legacy
    /// receipts.
    pub fn get_receipt_attestation_envelope(
        &self,
        receipt_id: &str,
    ) -> Result<Option<DsseEnvelope>> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let envelope_json: Option<String> = reader
            .query_row(
                "SELECT envelope_json FROM receipt_envelopes WHERE receipt_id = ?1",
                params![receipt_id],
                |row| row.get(0),
            )
            .optional()?;
        envelope_json
            .map(|json| {
                serde_json::from_str(&json).map_err(|error| {
                    StorageError::Init(format!("stored DSSE envelope deserialize: {error}"))
                })
            })
            .transpose()
    }

    /// Privacy-safe disclosure cardinality for diagnostics/tests.
    pub fn receipt_disclosure_count(&self, receipt_id: &str) -> Result<u64> {
        let reader = self
            .reader
            .lock()
            .map_err(|_| StorageError::Init("Reader lock poisoned".into()))?;
        let count: i64 = reader.query_row(
            "SELECT COUNT(*) FROM receipt_disclosures WHERE receipt_id = ?1",
            params![receipt_id],
            |row| row.get(0),
        )?;
        Ok(count.max(0) as u64)
    }
}

fn validate_signed_write_shape(write: &SignedReceiptWrite<'_>) -> Result<()> {
    if write.receipt.receipt_id != write.attestation.receipt_id().as_str() {
        return Err(StorageError::Init(
            "mutable receipt id must equal the immutable attestation receipt id".into(),
        ));
    }
    write
        .attestation
        .validate()
        .map_err(|error| StorageError::Init(format!("attestation invalid: {error}")))?;
    if write.disclosures.len() != write.attestation.predicate().evidence().len() {
        return Err(StorageError::Init(
            "every attested evidence slot requires exactly one deletable disclosure".into(),
        ));
    }
    let mut slots = std::collections::HashSet::with_capacity(write.disclosures.len());
    for disclosure in write.disclosures {
        let slot = disclosure.evidence_slot().as_str();
        if !slots.insert(slot)
            || verify_disclosure(write.attestation, slot, Some(disclosure))
                != DisclosureVerification::Verified
        {
            return Err(StorageError::Init(format!(
                "invalid or duplicate receipt disclosure for slot '{slot}'"
            )));
        }
    }
    Ok(())
}

fn load_trusted_signing_keys(tx: &Transaction<'_>) -> Result<Vec<TrustedSigningKey>> {
    let mut stmt = tx.prepare(
        "SELECT key_id, public_key, status, valid_from, valid_until, revoked_at
           FROM receipt_signing_keys ORDER BY key_id LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![(MAX_TRUSTED_SIGNING_KEYS + 1) as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Vec<u8>>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
        ))
    })?;
    let mut keys = Vec::new();
    for row in rows {
        let (key_id, public_key, status, valid_from, valid_until, revoked_at) = row?;
        let public_key: [u8; 32] = public_key
            .try_into()
            .map_err(|_| StorageError::Init("registered Ed25519 key is not 32 bytes".into()))?;
        keys.push(TrustedSigningKey {
            key_id,
            public_key,
            status: parse_signing_key_status(&status)?,
            valid_from: parse_utc(&valid_from)?,
            valid_until: valid_until.as_deref().map(parse_utc).transpose()?,
            revoked_at: revoked_at.as_deref().map(parse_utc).transpose()?,
        });
    }
    if keys.len() > MAX_TRUSTED_SIGNING_KEYS {
        return Err(StorageError::Init(format!(
            "receipt signing-key registry exceeds the verification bound of {MAX_TRUSTED_SIGNING_KEYS}"
        )));
    }
    Ok(keys)
}

#[cfg(unix)]
fn validate_sidecar_key_id(key_id: &str) -> Result<()> {
    if key_id.is_empty()
        || key_id.len() > 128
        || !key_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(StorageError::Init(
            "sidecar key id must be a 1..=128 byte ASCII code using only letters, digits, '.', '_', or '-'"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn validate_sidecar_directory(directory: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = std::fs::symlink_metadata(directory).map_err(|error| {
        StorageError::Init(format!("stat receipt signing-key directory: {error}"))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(StorageError::Init(
            "receipt signing-key directory must be a non-symlink directory".into(),
        ));
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(StorageError::Init(
            "receipt signing-key directory permissions must not grant group/other access".into(),
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn random_signing_seed() -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("vestige.receipt.signing-seed.v1");
    for _ in 0..4 {
        hasher.update(Uuid::new_v4().as_bytes());
    }
    let mut seed = [0_u8; 32];
    hasher.finalize_xof().fill(&mut seed);
    seed
}

fn validate_registry_key(key: &TrustedSigningKey) -> Result<()> {
    if key.key_id.trim().is_empty() || key.key_id.len() > 128 {
        return Err(StorageError::Init(
            "receipt signing key id must contain 1..=128 bytes".into(),
        ));
    }
    ed25519_dalek::VerifyingKey::from_bytes(&key.public_key)
        .map_err(|_| StorageError::Init("malformed Ed25519 public key".into()))?;
    if key.valid_until.is_some_and(|until| until <= key.valid_from) {
        return Err(StorageError::Init(
            "receipt signing key valid_until must be after valid_from".into(),
        ));
    }
    Ok(())
}

const fn signing_key_status_label(status: SigningKeyStatus) -> &'static str {
    match status {
        SigningKeyStatus::Active => "active",
        SigningKeyStatus::Retired => "retired",
        SigningKeyStatus::Revoked => "revoked",
        SigningKeyStatus::Disabled => "disabled",
    }
}

fn parse_signing_key_status(value: &str) -> Result<SigningKeyStatus> {
    match value {
        "active" => Ok(SigningKeyStatus::Active),
        "retired" => Ok(SigningKeyStatus::Retired),
        "revoked" => Ok(SigningKeyStatus::Revoked),
        "disabled" => Ok(SigningKeyStatus::Disabled),
        _ => Err(StorageError::Init(format!(
            "unknown receipt signing-key status '{value}'"
        ))),
    }
}

fn normalized_utc(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::AutoSi, true)
}

fn parse_utc(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| StorageError::Init(format!("invalid signing-key timestamp: {error}")))
}

fn u64_to_sqlite(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StorageError::Init(format!("chain sequence {value} exceeds SQLite INTEGER")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IngestInput;
    use crate::storage::receipt_attestation::{
        AttestationChainPosition, CaptureDirection, ProducerIdentity,
        RedactionSafeDecisionProjectionV1, sign_attestation,
    };
    use crate::trace::{DecayRisk, Receipt};
    use ed25519_dalek::SigningKey;

    const SEED_1: [u8; 32] = [41; 32];
    const SEED_2: [u8; 32] = [42; 32];

    fn store() -> SqliteMemoryStore {
        let dir = tempfile::tempdir().expect("tempdir");
        SqliteMemoryStore::new(Some(dir.path().join("attestation.db"))).expect("test store")
    }

    fn trusted_key(id: &str, seed: &[u8; 32]) -> TrustedSigningKey {
        TrustedSigningKey {
            key_id: id.to_string(),
            public_key: SigningKey::from_bytes(seed).verifying_key().to_bytes(),
            status: SigningKeyStatus::Active,
            valid_from: DateTime::<Utc>::UNIX_EPOCH,
            valid_until: None,
            revoked_at: None,
        }
    }

    fn receipt(receipt_id: &str, memories: &[String]) -> Receipt {
        Receipt {
            receipt_id: receipt_id.to_string(),
            retrieved: memories.to_vec(),
            suppressed: vec![],
            activation_path: vec![],
            trust_floor: 0.8,
            decay_risk: DecayRisk::Low,
            mutations: vec![],
            evidence: None,
        }
    }

    #[test]
    fn signed_receipt_chain_continues_atomically_across_key_rotation() {
        let store = store();
        let memory = store
            .ingest(IngestInput {
                content: "attested memory".into(),
                ..Default::default()
            })
            .unwrap();
        let key_1 = trusted_key("key-1", &SEED_1);
        let key_2 = trusted_key("key-2", &SEED_2);
        assert!(store.register_receipt_signing_key(&key_1).unwrap());
        assert!(store.register_receipt_signing_key(&key_2).unwrap());
        assert!(!store.register_receipt_signing_key(&key_1).unwrap());

        let first = ReceiptAttestationV1::build(
            Utc::now(),
            ProducerIdentity::new("vestige-core", "2.3.0", "v24-test").unwrap(),
            AttestationChainPosition::Genesis,
            "synaptic-capture-v1",
            RedactionSafeDecisionProjectionV1::SynapticCapture {
                direction: CaptureDirection::Backward,
                evaluated_count: 1,
                captured_count: 1,
                withheld_count: 0,
            },
            [memory.id.as_str()],
        )
        .unwrap();
        let (first_attestation, first_disclosures) = first.into_parts();
        let first_signed = sign_attestation(&first_attestation, "key-1", &SEED_1).unwrap();
        let first_receipt = receipt(
            first_attestation.receipt_id().as_str(),
            std::slice::from_ref(&memory.id),
        );
        let durable_first = store
            .save_signed_receipt_atomic(SignedReceiptWrite {
                receipt: &first_receipt,
                attestation: &first_attestation,
                signed: &first_signed,
                disclosures: &first_disclosures,
                run_id: Some("run-v24"),
                tool: Some("test"),
                query: None,
            })
            .unwrap();
        assert_eq!(durable_first.sequence, 0);

        let second = ReceiptAttestationV1::build(
            Utc::now(),
            ProducerIdentity::new("vestige-core", "2.3.0", "v24-test").unwrap(),
            AttestationChainPosition::Successor(first_signed.chain_entry()),
            "synaptic-capture-v1",
            RedactionSafeDecisionProjectionV1::SynapticCapture {
                direction: CaptureDirection::Forward,
                evaluated_count: 1,
                captured_count: 1,
                withheld_count: 0,
            },
            [memory.id.as_str()],
        )
        .unwrap();
        let (second_attestation, second_disclosures) = second.into_parts();
        let second_signed = sign_attestation(&second_attestation, "key-2", &SEED_2).unwrap();
        let second_receipt = receipt(
            second_attestation.receipt_id().as_str(),
            std::slice::from_ref(&memory.id),
        );
        let durable_second = store
            .save_signed_receipt_atomic(SignedReceiptWrite {
                receipt: &second_receipt,
                attestation: &second_attestation,
                signed: &second_signed,
                disclosures: &second_disclosures,
                run_id: Some("run-v24"),
                tool: Some("test"),
                query: None,
            })
            .unwrap();
        assert_eq!(durable_second.sequence, 1);
        assert_eq!(durable_second.signing_key_id, "key-2");
        assert_eq!(
            store
                .receipt_attestation_status(&second_receipt.receipt_id)
                .unwrap(),
            Some(ReceiptAttestationStatus::SignedV1)
        );
        assert_eq!(
            store
                .get_receipt_attestation_envelope(&second_receipt.receipt_id)
                .unwrap(),
            Some(second_signed.envelope.clone())
        );
    }

    #[test]
    fn purge_deletes_only_the_target_disclosure_and_keeps_envelope_immutable() {
        let store = store();
        let first_memory = store
            .ingest(IngestInput {
                content: "first".into(),
                ..Default::default()
            })
            .unwrap();
        let second_memory = store
            .ingest(IngestInput {
                content: "second".into(),
                ..Default::default()
            })
            .unwrap();
        let key = trusted_key("purge-key", &SEED_1);
        store.register_receipt_signing_key(&key).unwrap();
        let memory_ids = vec![first_memory.id.clone(), second_memory.id.clone()];
        let prepared = ReceiptAttestationV1::build(
            Utc::now(),
            ProducerIdentity::new("vestige-core", "2.3.0", "v24-purge").unwrap(),
            AttestationChainPosition::Genesis,
            "replay-v1",
            RedactionSafeDecisionProjectionV1::CounterfactualReplayInfluence {
                baseline_count: 2,
                counterfactual_count: 1,
                membership_changed: true,
                ordering_changed: false,
                decision_changed: true,
                withheld_slot_count: 0,
            },
            memory_ids.iter().map(String::as_str),
        )
        .unwrap();
        let (attestation, disclosures) = prepared.into_parts();
        let signed = sign_attestation(&attestation, "purge-key", &SEED_1).unwrap();
        let mutable_receipt = receipt(attestation.receipt_id().as_str(), &memory_ids);
        store
            .save_signed_receipt_atomic(SignedReceiptWrite {
                receipt: &mutable_receipt,
                attestation: &attestation,
                signed: &signed,
                disclosures: &disclosures,
                run_id: None,
                tool: Some("test"),
                query: None,
            })
            .unwrap();
        assert_eq!(
            store
                .receipt_disclosure_count(&mutable_receipt.receipt_id)
                .unwrap(),
            2
        );
        store
            .purge_node(&first_memory.id, Some("V24 disclosure test"))
            .unwrap();
        assert_eq!(
            store
                .receipt_disclosure_count(&mutable_receipt.receipt_id)
                .unwrap(),
            1
        );
        assert!(
            store
                .get_receipt_attestation_envelope(&mutable_receipt.receipt_id)
                .unwrap()
                .is_some()
        );

        let writer = store.writer.lock().unwrap();
        let update = writer.execute(
            "UPDATE receipt_envelopes SET payload_digest = ?1 WHERE receipt_id = ?2",
            params!["0".repeat(64), mutable_receipt.receipt_id],
        );
        assert!(
            update.is_err(),
            "immutable envelope trigger must reject updates"
        );
    }

    #[test]
    fn legacy_receipts_remain_unsigned_and_failed_bundle_is_atomic() {
        let store = store();
        let legacy = receipt("legacy-receipt", &[]);
        store
            .save_receipt(&legacy, None, Some("test"), None)
            .unwrap();
        assert_eq!(
            store.receipt_attestation_status("legacy-receipt").unwrap(),
            Some(ReceiptAttestationStatus::LegacyUnsigned)
        );

        let key = trusted_key("atomic-key", &SEED_1);
        store.register_receipt_signing_key(&key).unwrap();
        let prepared = ReceiptAttestationV1::build(
            Utc::now(),
            ProducerIdentity::new("vestige-core", "2.3.0", "v24-atomic").unwrap(),
            AttestationChainPosition::Genesis,
            "synaptic-capture-v1",
            RedactionSafeDecisionProjectionV1::SynapticCapture {
                direction: CaptureDirection::Backward,
                evaluated_count: 1,
                captured_count: 1,
                withheld_count: 0,
            },
            ["missing-memory"],
        )
        .unwrap();
        let (attestation, disclosures) = prepared.into_parts();
        let signed = sign_attestation(&attestation, "atomic-key", &SEED_1).unwrap();
        let mutable_receipt = receipt(attestation.receipt_id().as_str(), &[]);
        assert!(
            store
                .save_signed_receipt_atomic(SignedReceiptWrite {
                    receipt: &mutable_receipt,
                    attestation: &attestation,
                    signed: &signed,
                    disclosures: &disclosures,
                    run_id: None,
                    tool: None,
                    query: None,
                })
                .is_err()
        );
        assert_eq!(
            store
                .receipt_attestation_status(&mutable_receipt.receipt_id)
                .unwrap(),
            None,
            "failed disclosure FK must roll back the mutable receipt and envelope"
        );
    }

    #[cfg(unix)]
    #[test]
    fn sidecar_provisioning_is_no_clobber_fsynced_and_permission_bounded() {
        use std::os::unix::fs::PermissionsExt;

        let parent = tempfile::tempdir().expect("sidecar parent");
        let key_directory = parent.path().join("receipt-keys");
        let provisioned = provision_receipt_signing_key_sidecar(
            &key_directory,
            "sidecar-key-1",
            DateTime::<Utc>::UNIX_EPOCH,
        )
        .expect("provision sidecar");
        let directory_mode = std::fs::metadata(&key_directory)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let file_mode = std::fs::metadata(&provisioned.seed_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(directory_mode, 0o700);
        assert_eq!(file_mode, 0o600);
        let seed = load_receipt_signing_seed(&provisioned.seed_path).unwrap();
        assert_eq!(
            SigningKey::from_bytes(&seed).verifying_key().to_bytes(),
            provisioned.trusted_key.public_key
        );
        assert!(
            provision_receipt_signing_key_sidecar(
                &key_directory,
                "sidecar-key-1",
                DateTime::<Utc>::UNIX_EPOCH,
            )
            .is_err(),
            "provisioning must never overwrite an existing seed"
        );
    }
}

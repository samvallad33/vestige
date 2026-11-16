//! Pairing ticket, SPAKE2, and the short authentication string.
//!
//! The ticket carries public identity and one SPAKE2 message. The low-entropy
//! code is the PAKE password. Both sides show the same authentication string
//! and a human confirms it before either device writes a grant. The test hook
//! `VESTIGE_MESH_TEST_CONFIRM=1` confirms without a terminal.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use spake2::{Ed25519Group, Identity, Password, Spake2};
use strata_store::{mesh_grant_context, StrataStore};

use crate::protocol::MeshError;

/// Domain separated from the log's segment signatures.
pub const BINDING_DOMAIN: &[u8] = b"vestige.mesh.endpoint.v1\0";
const TICKET_PREFIX: &str = "vm1.";

/// One paired peer. Public fields only. No seed, no PAKE password.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerRecord {
    /// did:key of the peer's log verifying key.
    pub writer_did: String,
    /// Hex of the 32-byte verifying key.
    pub verifying_key_hex: String,
    /// iroh endpoint id, hex. Empty when the pair was offline.
    pub endpoint_id: String,
    /// Direct addresses. Empty when relay and discovery are off and no
    /// address was learned.
    pub addresses: Vec<String>,
    /// `disabled`, `self-hosted`, or `public`.
    pub relay: String,
}

/// What `vestige mesh pair` shows.
#[derive(Debug, Clone)]
pub struct PairingOffer {
    /// Ticket the other device types or scans.
    pub ticket: String,
    /// Short authentication string. Both screens must show the same value.
    pub sas: String,
    /// ASCII QR of the ticket.
    pub qr: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TicketBody {
    writer_did: String,
    verifying_key: [u8; 32],
    endpoint_id: String,
    addresses: Vec<String>,
    pake: Vec<u8>,
    binding: Vec<u8>,
}

/// Start a pairing. `code` is the low-entropy password the human typed on
/// both devices, or the test hook's fixed code.
pub fn offer_ticket(
    store: &StrataStore,
    code: &str,
    endpoint_id: &str,
    addresses: &[String],
) -> Result<(PairingOffer, Spake2<Ed25519Group>), MeshError> {
    let (state, pake) = Spake2::<Ed25519Group>::start_a(
        &Password::new(code.as_bytes()),
        &Identity::new(b"vestige-mesh-a"),
        &Identity::new(b"vestige-mesh-b"),
    );
    let binding = binding_message(endpoint_id);
    let signature = store.sign_bytes(&binding);
    let body = TicketBody {
        writer_did: store.writer_did(),
        verifying_key: store_key(store)?,
        endpoint_id: endpoint_id.to_string(),
        addresses: addresses.to_vec(),
        pake,
        binding: signature.to_vec(),
    };
    let ticket = format!(
        "{TICKET_PREFIX}{}",
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(&body).map_err(|err| MeshError::Export(err.to_string()))?)
    );
    let qr = render_qr(&ticket);
    Ok((
        PairingOffer {
            ticket,
            sas: String::new(),
            qr,
        },
        state,
    ))
}

/// Finish the initiator's side once the joiner's PAKE message arrives.
pub fn finish_offer(
    state: Spake2<Ed25519Group>,
    their_message: &[u8],
) -> Result<String, MeshError> {
    let key = state
        .finish(their_message)
        .map_err(|err| MeshError::Export(format!("pake: {err}")))?;
    Ok(sas(&key))
}

/// The joiner's answer: a ticket whose PAKE message finishes the initiator's
/// state, plus the grant the joiner wrote when `confirm` is set.
pub struct PairingReply {
    /// Short authentication string. The initiator must show the same value.
    pub sas: String,
    /// Ticket the initiator feeds to [`accept_response`].
    pub response: String,
    /// The initiator, as the joiner recorded them.
    pub peer: PeerRecord,
}

/// Build the joiner's response and, when confirmed, grant the initiator.
pub fn respond_ticket(
    store: &mut StrataStore,
    ticket: &str,
    code: &str,
    confirm: bool,
    endpoint_id: &str,
    addresses: &[String],
) -> Result<PairingReply, MeshError> {
    let body = decode_ticket(ticket)?;
    let (state, ours) = Spake2::<Ed25519Group>::start_b(
        &Password::new(code.as_bytes()),
        &Identity::new(b"vestige-mesh-a"),
        &Identity::new(b"vestige-mesh-b"),
    );
    let shared = state
        .finish(&body.pake)
        .map_err(|err| MeshError::Export(format!("pake: {err}")))?;
    let sas = sas(&shared);
    if !confirm && !test_confirm() {
        return Err(MeshError::Export(
            "pairing was not confirmed; the grant was not written".into(),
        ));
    }
    verify_binding(&body)?;
    store.grant_writer(
        body.verifying_key,
        Vec::new(),
        body.binding.clone(),
        &mesh_grant_context(),
    )?;
    let peer = PeerRecord {
        writer_did: body.writer_did.clone(),
        verifying_key_hex: hex32(&body.verifying_key),
        endpoint_id: body.endpoint_id.clone(),
        addresses: body.addresses.clone(),
        relay: "disabled".into(),
    };
    remember_peer(store, &peer)?;
    let response = signed_ticket(store, endpoint_id, addresses, ours)?;
    Ok(PairingReply {
        sas,
        response,
        peer,
    })
}

/// Finish the initiator once the joiner's response ticket arrives.
pub fn accept_response(
    store: &mut StrataStore,
    state: Spake2<Ed25519Group>,
    response: &str,
    confirm: bool,
) -> Result<(String, PeerRecord), MeshError> {
    let body = decode_ticket(response)?;
    let sas = finish_offer(state, &body.pake)?;
    if !confirm && !test_confirm() {
        return Err(MeshError::Export(
            "pairing was not confirmed; the grant was not written".into(),
        ));
    }
    verify_binding(&body)?;
    store.grant_writer(
        body.verifying_key,
        Vec::new(),
        body.binding.clone(),
        &mesh_grant_context(),
    )?;
    let peer = PeerRecord {
        writer_did: body.writer_did,
        verifying_key_hex: hex32(&body.verifying_key),
        endpoint_id: body.endpoint_id,
        addresses: body.addresses,
        relay: "disabled".into(),
    };
    remember_peer(store, &peer)?;
    Ok((sas, peer))
}

fn signed_ticket(
    store: &StrataStore,
    endpoint_id: &str,
    addresses: &[String],
    pake: Vec<u8>,
) -> Result<String, MeshError> {
    let binding = binding_message(endpoint_id);
    let signature = store.sign_bytes(&binding);
    let body = TicketBody {
        writer_did: store.writer_did(),
        verifying_key: store_key(store)?,
        endpoint_id: endpoint_id.to_string(),
        addresses: addresses.to_vec(),
        pake,
        binding: signature.to_vec(),
    };
    Ok(format!(
        "{TICKET_PREFIX}{}",
        URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&body).map_err(|err| MeshError::Export(err.to_string()))?
        )
    ))
}

/// Accept a ticket. Returns the SAS and the peer record. The grant is written
/// only when `confirm` is true or the test hook is set.
pub fn accept_ticket(
    store: &mut StrataStore,
    ticket: &str,
    code: &str,
    confirm: bool,
) -> Result<(String, PeerRecord), MeshError> {
    let body = decode_ticket(ticket)?;
    let (state, ours) = Spake2::<Ed25519Group>::start_b(
        &Password::new(code.as_bytes()),
        &Identity::new(b"vestige-mesh-a"),
        &Identity::new(b"vestige-mesh-b"),
    );
    let shared = state
        .finish(&body.pake)
        .map_err(|err| MeshError::Export(format!("pake: {err}")))?;
    let sas = sas(&shared);
    if !confirm && !test_confirm() {
        return Err(MeshError::Export(
            "pairing was not confirmed; the grant was not written".into(),
        ));
    }
    verify_binding(&body)?;
    store
        .grant_writer(
            body.verifying_key,
            Vec::new(),
            body.binding.clone(),
            &mesh_grant_context(),
        )
        .map_err(MeshError::Store)?;
    let peer = PeerRecord {
        writer_did: body.writer_did,
        verifying_key_hex: hex32(&body.verifying_key),
        endpoint_id: body.endpoint_id,
        addresses: body.addresses,
        relay: "disabled".into(),
    };
    remember_peer(store, &peer)?;
    let _ = ours;
    Ok((sas, peer))
}

/// The joiner's PAKE message, so the initiator can compute the same SAS and grant.
pub fn joiner_message(code: &str, initiator_pake: &[u8]) -> Result<(Vec<u8>, String), MeshError> {
    let (state, ours) = Spake2::<Ed25519Group>::start_b(
        &Password::new(code.as_bytes()),
        &Identity::new(b"vestige-mesh-a"),
        &Identity::new(b"vestige-mesh-b"),
    );
    let shared = state
        .finish(initiator_pake)
        .map_err(|err| MeshError::Export(format!("pake: {err}")))?;
    Ok((ours, sas(&shared)))
}

/// True when the process is the non-interactive test hook.
pub fn test_confirm() -> bool {
    std::env::var("VESTIGE_MESH_TEST_CONFIRM").ok().as_deref() == Some("1")
}

/// Peers recorded for this data dir.
pub fn load_peers(dir: &std::path::Path) -> Vec<PeerRecord> {
    let path = dir.join("mesh/peers.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

/// Replace the peer list. Public fields only.
pub fn save_peers(dir: &std::path::Path, peers: &[PeerRecord]) -> Result<(), MeshError> {
    let path = dir.join("mesh/peers.json");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(peers).map_err(|err| MeshError::Export(err.to_string()))?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Drop one peer from the list. The grant stays in the log; a log is append-only.
pub fn unpair(dir: &std::path::Path, did: &str) -> Result<(), MeshError> {
    let peers: Vec<PeerRecord> = load_peers(dir)
        .into_iter()
        .filter(|peer| peer.writer_did != did)
        .collect();
    save_peers(dir, &peers)
}

fn remember_peer(store: &StrataStore, peer: &PeerRecord) -> Result<(), MeshError> {
    let mut peers = load_peers(store.data_dir());
    peers.retain(|existing| existing.writer_did != peer.writer_did);
    peers.push(peer.clone());
    save_peers(store.data_dir(), &peers)
}

/// The initiator's SPAKE2 message, pulled out of a ticket.
pub fn ticket_pake(ticket: &str) -> Result<Vec<u8>, MeshError> {
    Ok(decode_ticket(ticket)?.pake)
}

fn decode_ticket(ticket: &str) -> Result<TicketBody, MeshError> {
    let encoded = ticket
        .strip_prefix(TICKET_PREFIX)
        .ok_or_else(|| MeshError::Export("ticket prefix is vm1.".into()))?;
    let bytes = URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|err| MeshError::Export(format!("ticket encoding: {err}")))?;
    serde_json::from_slice(&bytes).map_err(|err| MeshError::Export(err.to_string()))
}

fn verify_binding(body: &TicketBody) -> Result<(), MeshError> {
    let key = VerifyingKey::from_bytes(&body.verifying_key)
        .map_err(|_| MeshError::Export("pairing key is not ed25519".into()))?;
    let signature = Signature::from_slice(&body.binding)
        .map_err(|_| MeshError::Export("pairing binding is not a signature".into()))?;
    let message = binding_message(&body.endpoint_id);
    key.verify(&message, &signature)
        .map_err(|_| MeshError::Export("endpoint binding did not verify".into()))?;
    Ok(())
}

fn binding_message(endpoint_id: &str) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(BINDING_DOMAIN);
    message.extend_from_slice(endpoint_id.as_bytes());
    message
}

fn store_key(store: &StrataStore) -> Result<[u8; 32], MeshError> {
    Ok(store.verifying_key_bytes())
}

fn sas(shared: &[u8]) -> String {
    let digest = blake3::hash(shared);
    hex32(&digest.as_bytes()[..4].try_into().unwrap_or([0; 4]))
}

fn hex32(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn render_qr(text: &str) -> String {
    qrcode::QrCode::new(text.as_bytes())
        .map(|code| {
            code.render::<char>()
                .quiet_zone(false)
                .module_dimensions(1, 1)
                .build()
        })
        .unwrap_or_else(|_| text.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use strata_store::StrataStore;

    use super::*;

    fn dir(name: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("strata-mesh-pair-{n}-{name}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn confirmed_code_grants_both_sides_and_a_bare_join_does_not() {
        let mut alice = StrataStore::open(dir("alice")).unwrap();
        let mut bob = StrataStore::open(dir("bob")).unwrap();
        let (offer, state) = offer_ticket(&alice, "desk", "endpoint-a", &["127.0.0.1:1".into()]).unwrap();
        assert!(offer.ticket.starts_with("vm1."));
        assert!(!offer.qr.is_empty());
        let refused = respond_ticket(
            &mut bob,
            &offer.ticket,
            "desk",
            false,
            "endpoint-b",
            &["127.0.0.1:2".into()],
        );
        assert!(refused.is_err());
        assert!(bob.mesh_grants().is_empty());

        let reply = respond_ticket(
            &mut bob,
            &offer.ticket,
            "desk",
            true,
            "endpoint-b",
            &["127.0.0.1:2".into()],
        )
        .unwrap();
        let (sas, peer) = accept_response(&mut alice, state, &reply.response, true).unwrap();
        assert_eq!(sas, reply.sas);
        assert_eq!(peer.writer_did, bob.writer_did());
        assert_eq!(alice.mesh_grants(), vec![bob.writer_did()]);
        assert_eq!(bob.mesh_grants(), vec![alice.writer_did()]);
        assert!(std::fs::read(alice.data_dir().join("log/strata.key")).is_ok());
        assert!(!bob.data_dir().join("mesh/chains").join("strata.key").exists());
    }

    #[test]
    fn mcp_context_cannot_grant() {
        let mut store = StrataStore::open(dir("mcp")).unwrap();
        let peer = StrataStore::open(dir("peer")).unwrap();
        let err = store.grant_writer(
            peer.verifying_key_bytes(),
            Vec::new(),
            Vec::new(),
            &strata_store::AdmissionContext::default(),
        );
        assert!(err.is_err());
        assert!(store.mesh_grants().is_empty());
    }
}

//! Strata Mesh CLI and maintain actions.
//!
//! Pairing, unpairing, fork, export and import run in the process that holds
//! the log and set the human-confirmed grant context. `mesh_status` and
//! `mesh_sync` do not. An unpaired store never reaches [`open_session`], so
//! it opens no socket.

use std::sync::Arc;

use serde_json::{json, Value};
use vestige_core::Storage;

/// Dispatch a mesh command. `action` is `status`, `sync`, `pair`, `join`,
/// `unpair`, `fork`, `export`, `import`, or `pull`.
pub async fn dispatch(storage: &Arc<Storage>, args: &Value) -> Result<Value, String> {
    #[cfg(feature = "mesh")]
    {
        enabled::dispatch(storage, args).await
    }
    #[cfg(not(feature = "mesh"))]
    {
        let _ = (storage, args);
        Err(
            "this build was compiled without the `mesh` feature. Nothing was written and no socket was opened."
                .into(),
        )
    }
}

/// Status for an already-open store. Does not bind.
pub async fn status(storage: &Arc<Storage>) -> Result<Value, String> {
    dispatch(storage, &json!({"action": "status"})).await
}

/// Sync already-paired peers. `listen` accepts one connection when `once` is set.
pub async fn sync(
    storage: &Arc<Storage>,
    listen: bool,
    once: bool,
    handle: Option<&str>,
) -> Result<Value, String> {
    dispatch(
        storage,
        &json!({
            "action": "sync",
            "listen": listen,
            "once": once,
            "handle": handle,
        }),
    )
    .await
}

#[cfg(feature = "mesh")]
mod enabled {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::net::{SocketAddr, SocketAddrV4};
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::strata_memory::with_store;

    use strata_mesh::{
        accept_response, apply_export, load_peers, offer_ticket, pack_export, peer_addr,
        respond_ticket, serve, unpair, IrohSession, RelaySetting,
    };
    use strata_store::mesh_grant_context;

    const TICKET_WAIT: Duration = Duration::from_secs(90);

    pub async fn dispatch(storage: &Arc<Storage>, args: &Value) -> Result<Value, String> {
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("status");
        match action {
            "status" => Ok(status_of(storage)?),
            "sync" | "pull" => {
                let listen = args.get("listen").and_then(Value::as_bool).unwrap_or(false);
                let once = args.get("once").and_then(Value::as_bool).unwrap_or(listen);
                let handle = args.get("handle").and_then(Value::as_str).map(str::to_string);
                sync_peers(storage, listen, once, handle).await
            }
            "pair" => {
                pair(
                    storage,
                    required(args, "code")?,
                    args.get("bind").and_then(Value::as_str),
                    path_arg(args, "ticket_file")?,
                    path_arg(args, "wait_file")?,
                    args.get("confirm").and_then(Value::as_bool).unwrap_or(false),
                )
                .await
            }
            "join" => {
                join(
                    storage,
                    required(args, "code")?,
                    args.get("bind").and_then(Value::as_str),
                    path_arg(args, "ticket_file")?,
                    path_arg(args, "response_file")?,
                    args.get("ticket").and_then(Value::as_str).map(str::to_string),
                    args.get("confirm").and_then(Value::as_bool).unwrap_or(false),
                )
                .await
            }
            "unpair" => unpair_peer(storage, required(args, "did")?),
            "fork" => fork(storage),
            "export" => export_chain(storage, PathBuf::from(required(args, "dir")?)),
            "import" => import_chain(
                storage,
                PathBuf::from(required(args, "dir")?),
                args.get("confirm").and_then(Value::as_bool).unwrap_or(false),
            ),
            other => Err(format!("unknown mesh action {other}")),
        }
    }

    fn required<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
        args.get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("mesh requires {key}"))
    }

    fn path_arg(args: &Value, key: &str) -> Result<Option<PathBuf>, String> {
        match args.get(key).and_then(Value::as_str) {
            Some(value) if !value.is_empty() => Ok(Some(PathBuf::from(value))),
            _ => Ok(None),
        }
    }

    fn data_dir(storage: &Arc<Storage>) -> Result<PathBuf, String> {
        with_store(storage.as_ref(), |store| store.data_dir().to_path_buf())
    }

    fn status_of(storage: &Arc<Storage>) -> Result<Value, String> {
        with_store(storage.as_ref(), |store| {
            let peers = load_peers(store.data_dir());
            let merges: Vec<Value> = store
                .mesh_merges()
                .into_iter()
                .map(|(writer, from_seq, to_seq, effect_seq)| {
                    json!({
                        "writer": writer,
                        "from_seq": from_seq,
                        "to_seq": to_seq,
                        "receipt": strata_store::effect_receipt_id(effect_seq),
                    })
                })
                .collect();
            json!({
                "mesh_digest": hex(&store.mesh_digest()),
                "frontier": store.mesh_frontier(),
                "peers": peers,
                "relay": relay_setting().as_str(),
                "discovery": "off",
                "merges": merges,
                "conflicts": store.mesh_conflict_receipts().len(),
                "quarantine": store.mesh_grants().into_iter().filter(|did| store.mesh_quarantined(did)).collect::<Vec<_>>(),
                "paired": !peers.is_empty(),
            })
        })
    }

    fn relay_setting() -> RelaySetting {
        match std::env::var("VESTIGE_MESH_RELAY").ok().as_deref() {
            Some("public") => RelaySetting::Public,
            Some("self-hosted") => RelaySetting::SelfHosted,
            _ => RelaySetting::Disabled,
        }
    }

    async fn sync_peers(
        storage: &Arc<Storage>,
        listen: bool,
        once: bool,
        handle: Option<String>,
    ) -> Result<Value, String> {
        let peers = with_store(storage.as_ref(), |store| load_peers(store.data_dir()))?;
        if peers.is_empty() {
            return Err(
                "no paired peer. `vestige mesh pair` and `vestige mesh join` admit a grant; until then this store opens no socket."
                    .into(),
            );
        }
        let dir = data_dir(storage)?;
        let session = open_session(&dir, None).await?;
        let ack = dir.join("mesh/acks.log");
        if listen {
            let mut served = 0u32;
            loop {
                session
                    .accept_one(|request| {
                        with_store(storage.as_ref(), |store| {
                            serve(store, request, Some(&ack)).map_err(|err| err.to_string())
                        })
                        .and_then(|result| result)
                    })
                    .await
                    .map_err(|err| err.to_string())?;
                served += 1;
                if once || served >= peers.len() as u32 {
                    break;
                }
            }
        } else {
            for peer in &peers {
                let addr = peer_addr(&peer.endpoint_id, &peer.addresses)
                    .map_err(|err| format!("peer {}: {err}", peer.writer_did))?;
                let bytes = with_store(storage.as_ref(), |store| pack_export(store))?
                    .map_err(|err| err.to_string())?;
                let response = session
                    .roundtrip(addr, &bytes)
                    .await
                    .map_err(|err| err.to_string())?;
                with_store(storage.as_ref(), |store| {
                    apply_export(store, &response, Some(&ack), false).map_err(|err| err.to_string())
                })?
                .map_err(|err| err)?;
            }
        }
        let mut status = status_of(storage)?;
        if let Some(handle) = handle {
            status["pull"] = json!(handle);
        }
        status["bound"] = json!(session.bound.iter().map(|addr| addr.to_string()).collect::<Vec<_>>());
        status["relay"] = json!(session.relay().as_str());
        Ok(status)
    }

    async fn pair(
        storage: &Arc<Storage>,
        code: &str,
        bind: Option<&str>,
        ticket_file: Option<PathBuf>,
        wait_file: Option<PathBuf>,
        confirm: bool,
    ) -> Result<Value, String> {
        let dir = data_dir(storage)?;
        let session = open_session(&dir, bind_addr(bind)?).await?;
        let addresses = addresses_of(&session);
        let (offer, state) = with_store(storage.as_ref(), |store| {
            offer_ticket(store, code, &session.endpoint_id(), &addresses)
        })?
        .map_err(|err| err.to_string())?;
        if let Some(path) = &ticket_file {
            write_text(path, &offer.ticket)?;
        }
        let Some(wait_file) = wait_file else {
            return Ok(json!({
                "ticket": offer.ticket,
                "qr": offer.qr,
                "endpoint": session.endpoint_id(),
                "bound": addresses,
                "relay": session.relay().as_str(),
                "discovery": "off",
            }));
        };
        let response = wait_text(&wait_file).await?;
        let (sas, peer) = with_store(storage.as_ref(), |store| {
            accept_response(store, state, &response, confirm)
        })?
        .map_err(|err| err.to_string())?;
        Ok(json!({
            "sas": sas,
            "peer": peer.writer_did,
            "ticket": offer.ticket,
            "qr": offer.qr,
            "relay": session.relay().as_str(),
            "discovery": "off",
        }))
    }

    async fn join(
        storage: &Arc<Storage>,
        code: &str,
        bind: Option<&str>,
        ticket_file: Option<PathBuf>,
        response_file: Option<PathBuf>,
        ticket: Option<String>,
        confirm: bool,
    ) -> Result<Value, String> {
        let ticket = match ticket {
            Some(ticket) => ticket,
            None => {
                let path = ticket_file.ok_or("join needs --ticket or --ticket-file")?;
                wait_text(&path).await?
            }
        };
        let dir = data_dir(storage)?;
        let session = open_session(&dir, bind_addr(bind)?).await?;
        let addresses = addresses_of(&session);
        let reply = with_store(storage.as_ref(), |store| {
            respond_ticket(store, &ticket, code, confirm, &session.endpoint_id(), &addresses)
        })?
        .map_err(|err| err.to_string())?;
        if let Some(path) = response_file {
            write_text(&path, &reply.response)?;
        }
        Ok(json!({
            "sas": reply.sas,
            "peer": reply.peer.writer_did,
            "response": reply.response,
            "relay": session.relay().as_str(),
            "discovery": "off",
        }))
    }

    fn unpair_peer(storage: &Arc<Storage>, did: &str) -> Result<Value, String> {
        with_store(storage.as_ref(), |store| {
            unpair(store.data_dir(), did).map_err(|err| err.to_string())?;
            Ok(json!({
                "unpaired": did,
                "note": "the grant remains in the log; a Strata log does not erase it",
            }))
        })?
    }

    fn fork(storage: &Arc<Storage>) -> Result<Value, String> {
        with_store(storage.as_ref(), |store| {
            let did = store.fork_identity().map_err(|err| err.to_string())?;
            Ok(json!({
                "writer": did,
                "note": "the next open of this data directory uses mesh/fork-log",
            }))
        })?
    }

    fn export_chain(storage: &Arc<Storage>, dir: PathBuf) -> Result<Value, String> {
        with_store(storage.as_ref(), |store| {
            store.export_chain(&dir).map_err(|err| err.to_string())?;
            if dir.join("strata.key").exists() {
                return Err("export wrote strata.key".into());
            }
            Ok(json!({
                "dir": dir.display().to_string(),
                "writer": store.writer_did(),
                "mesh_digest": hex(&store.mesh_digest()),
            }))
        })?
    }

    fn import_chain(storage: &Arc<Storage>, dir: PathBuf, confirm: bool) -> Result<Value, String> {
        with_store(storage.as_ref(), |store| {
            let ctx = if confirm || strata_mesh::test_confirm() {
                mesh_grant_context()
            } else {
                strata_store::AdmissionContext::default()
            };
            let report = store.import_chain(&dir, &ctx).map_err(|err| err.to_string())?;
            Ok(json!({
                "mesh_digest": hex(&report.mesh_digest),
                "receipt_seq": report.receipt_seq,
                "receipt": report.receipt_seq.map(strata_store::effect_receipt_id),
                "receipt_only": report.receipt_only,
                "denied": report.denied,
                "policy_mismatch": report.policy_mismatch,
                "equivocation": report.equivocation,
            }))
        })?
    }

    fn addresses_of(session: &IrohSession) -> Vec<String> {
        session.bound.iter().map(|addr| addr.to_string()).collect()
    }

    fn bind_addr(text: Option<&str>) -> Result<Option<SocketAddrV4>, String> {
        let Some(text) = text else {
            return Ok(None);
        };
        let socket: SocketAddr = text
            .parse()
            .map_err(|_| format!("bind address {text} is not a socket"))?;
        match socket {
            SocketAddr::V4(addr) => Ok(Some(addr)),
            SocketAddr::V6(_) => Err("mesh binds an ipv4 loopback address".into()),
        }
    }

    async fn open_session(dir: &Path, bind: Option<SocketAddrV4>) -> Result<IrohSession, String> {
        let secret = load_secret(dir)?;
        let saved = saved_bind(dir)?;
        let addr = bind.or(saved).unwrap_or(SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, 0));
        let session = IrohSession::bind(relay_setting(), Some(addr), secret)
            .await
            .map_err(|err| err.to_string())?;
        store_secret(dir, &session.secret_bytes())?;
        if let Some(bound) = session.bound.iter().find(|addr| addr.is_ipv4()) {
            write_text(&dir.join("mesh/bind.txt"), &bound.to_string())?;
        }
        Ok(session)
    }

    fn saved_bind(dir: &Path) -> Result<Option<SocketAddrV4>, String> {
        let path = dir.join("mesh/bind.txt");
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).map_err(|err| err.to_string())?;
        let socket: SocketAddr = text
            .trim()
            .parse()
            .map_err(|_| "mesh/bind.txt is not a socket address".to_string())?;
        match socket {
            SocketAddr::V4(addr) => Ok(Some(addr)),
            SocketAddr::V6(_) => Err("mesh/bind.txt is not ipv4".into()),
        }
    }

    fn load_secret(dir: &Path) -> Result<Option<[u8; 32]>, String> {
        let path = dir.join("mesh/transport.key");
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).map_err(|err| err.to_string())?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "mesh transport key has the wrong length".to_string())?;
        Ok(Some(bytes))
    }

    fn store_secret(dir: &Path, secret: &[u8; 32]) -> Result<(), String> {
        let path = dir.join("mesh/transport.key");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let mut file = {
            let mut options = OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options.open(&path).map_err(|err| err.to_string())?
        };
        file.write_all(secret).map_err(|err| err.to_string())?;
        file.sync_all().map_err(|err| err.to_string())?;
        Ok(())
    }

    fn write_text(path: &Path, text: &str) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        std::fs::write(path, text).map_err(|err| err.to_string())
    }

    async fn wait_text(path: &Path) -> Result<String, String> {
        let start = std::time::Instant::now();
        loop {
            if path.is_file() {
                let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
                if !text.trim().is_empty() {
                    return Ok(text.trim().to_string());
                }
            }
            if start.elapsed() > TICKET_WAIT {
                return Err(format!("timed out waiting for {}", path.display()));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0xf) as usize] as char);
        }
        out
    }
}

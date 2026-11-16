//! In-memory pipe and the iroh endpoint.
//!
//! [`IrohSession::bind`] is the only constructor that opens a socket. An
//! unpaired process never calls it. The endpoint is built with relays off,
//! address lookup cleared, the portmapper off, and (by default) a loopback
//! bind. iroh supplies the QUIC crypto.

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

use iroh::endpoint::presets;

use crate::protocol::MeshError;

/// What status reports. The default is disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaySetting {
    /// No relay. The default.
    Disabled,
    /// An operator-chosen relay URL. This build still starts with relays off
    /// until that URL is configured on the endpoint.
    SelfHosted,
    /// iroh's public relays. Explicit opt-in.
    Public,
}

impl RelaySetting {
    /// Status word.
    pub fn as_str(self) -> &'static str {
        match self {
            RelaySetting::Disabled => "disabled",
            RelaySetting::SelfHosted => "self-hosted",
            RelaySetting::Public => "public",
        }
    }

    /// Parse a status word. Unknown values stay disabled.
    pub fn parse(text: &str) -> Self {
        match text {
            "public" => RelaySetting::Public,
            "self-hosted" => RelaySetting::SelfHosted,
            _ => RelaySetting::Disabled,
        }
    }
}

/// One request, one response.
pub trait MeshTransport {
    /// Send `request` and return the peer's bytes.
    fn roundtrip(&mut self, request: &[u8]) -> Result<Vec<u8>, String>;
}

/// A pipe whose other end is a function. No socket.
pub struct MemoryPipe<F> {
    respond: F,
}

impl<F> MemoryPipe<F>
where
    F: FnMut(&[u8]) -> Result<Vec<u8>, String>,
{
    /// `respond` is the peer.
    pub fn new(respond: F) -> Self {
        Self { respond }
    }
}

impl<F> MeshTransport for MemoryPipe<F>
where
    F: FnMut(&[u8]) -> Result<Vec<u8>, String>,
{
    fn roundtrip(&mut self, request: &[u8]) -> Result<Vec<u8>, String> {
        (self.respond)(request)
    }
}

/// A loopback iroh endpoint. Relays and discovery stay off unless asked.
pub struct IrohSession {
    endpoint: iroh::Endpoint,
    /// Bound addresses.
    pub bound: Vec<SocketAddr>,
    relay: RelaySetting,
}

impl IrohSession {
    /// Bind `127.0.0.1:0`. No relay, no discovery, no portmapper.
    pub async fn bind_loopback() -> Result<Self, MeshError> {
        Self::bind(RelaySetting::Disabled, Some(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)), None).await
    }

    /// Bind `addr` when set, otherwise the preset's default sockets.
    ///
    /// `secret` is the iroh transport key, not a Strata signing seed. `None`
    /// lets iroh generate one for this process.
    pub async fn bind(
        relay: RelaySetting,
        addr: Option<SocketAddrV4>,
        secret: Option<[u8; 32]>,
    ) -> Result<Self, MeshError> {
        let mut builder = iroh::Endpoint::builder(presets::Minimal)
            .relay_mode(match relay {
                RelaySetting::Public => iroh::RelayMode::Default,
                RelaySetting::Disabled | RelaySetting::SelfHosted => iroh::RelayMode::Disabled,
            })
            .clear_address_lookup()
            .portmapper_config(iroh::endpoint::PortmapperConfig::Disabled)
            .net_report_config(iroh::endpoint::NetReportConfig::minimal())
            .alpns(vec![ALPN.to_vec()]);
        if let Some(secret) = secret {
            builder = builder.secret_key(iroh::SecretKey::from_bytes(&secret));
        }
        if addr.is_some() {
            builder = builder.clear_ip_transports();
        }
        if let Some(addr) = addr {
            builder = builder
                .bind_addr(addr)
                .map_err(|err| MeshError::Transport(err.to_string()))?;
        }
        let endpoint = builder
            .bind()
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let bound = endpoint.bound_sockets();
        Ok(Self {
            endpoint,
            bound,
            relay,
        })
    }

    /// Relay mode this endpoint was built with.
    pub fn relay(&self) -> RelaySetting {
        self.relay
    }

    /// Endpoint id. Not a Strata signing key.
    pub fn endpoint_id(&self) -> String {
        self.endpoint.id().to_string()
    }

    /// The transport secret, so the next bind keeps the same endpoint id.
    /// Callers write it to a mode-0600 file and do not log it.
    pub fn secret_bytes(&self) -> [u8; 32] {
        self.endpoint.secret_key().to_bytes()
    }

    /// Send `request` to `addr` and read one response.
    pub async fn roundtrip(
        &self,
        addr: impl Into<iroh::EndpointAddr>,
        request: &[u8],
    ) -> Result<Vec<u8>, MeshError> {
        let connection = self
            .endpoint
            .connect(addr, ALPN)
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        send.write_all(request)
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        send.finish()
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let response = recv
            .read_to_end(32 * 1024 * 1024)
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let _ = send.stopped().await;
        Ok(response)
    }

    /// Accept one bidi stream, hand the bytes to `respond`, and write the answer.
    pub async fn accept_one<F>(&self, respond: F) -> Result<(), MeshError>
    where
        F: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| MeshError::Transport("endpoint closed".into()))?;
        let connection = incoming
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let (mut send, mut recv) = connection
            .accept_bi()
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let request = recv
            .read_to_end(32 * 1024 * 1024)
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        let response = respond(&request).map_err(MeshError::Transport)?;
        send.write_all(&response)
            .await
            .map_err(|err| MeshError::Transport(err.to_string()))?;
        finish_send(&mut send).await?;
        Ok(())
    }

    /// Address a peer dials: this endpoint id plus the bound sockets.
    pub fn dial_addr(&self) -> iroh::EndpointAddr {
        let mut addr = iroh::EndpointAddr::new(self.endpoint.id());
        for socket in &self.bound {
            addr = addr.with_ip_addr(*socket);
        }
        addr
    }
}

async fn finish_send(send: &mut iroh::endpoint::SendStream) -> Result<(), MeshError> {
    send.finish()
        .map_err(|err| MeshError::Transport(err.to_string()))?;
    match send.stopped().await {
        Ok(_) => Ok(()),
        Err(err) => Err(MeshError::Transport(err.to_string())),
    }
}

const ALPN: &[u8] = b"vestige-mesh/1";

/// Direct address for a paired peer. No relay URL is added.
pub fn peer_addr(endpoint_id: &str, addresses: &[String]) -> Result<iroh::EndpointAddr, MeshError> {
    let id: iroh::EndpointId = endpoint_id
        .parse()
        .map_err(|_| MeshError::Transport("endpoint id is not an iroh key".into()))?;
    let mut addr = iroh::EndpointAddr::new(id);
    for text in addresses {
        let socket: std::net::SocketAddr = text
            .parse()
            .map_err(|_| MeshError::Transport(format!("address {text} is not a socket")))?;
        addr = addr.with_ip_addr(socket);
    }
    if addr.is_empty() {
        return Err(MeshError::Transport(
            "peer has no direct address".into(),
        ));
    }
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loopback_bind_is_local_and_has_no_relay() {
        let session = IrohSession::bind_loopback().await.expect("bind");
        assert_eq!(session.relay(), RelaySetting::Disabled);
        assert!(!session.bound.is_empty());
        assert!(session.bound.iter().all(|addr| addr.ip().is_loopback()));
    }

    #[tokio::test]
    async fn loopback_roundtrip_stays_on_the_pipe() {
        let left = IrohSession::bind_loopback().await.expect("left");
        let right = IrohSession::bind_loopback().await.expect("right");
        let addr = right.dial_addr();
        let server = tokio::spawn(async move {
            right
                .accept_one(|request| Ok(request.to_vec()))
                .await
                .expect("accept");
        });
        let got = left.roundtrip(addr, b"ping").await.expect("roundtrip");
        assert_eq!(got, b"ping");
        server.await.expect("server");
    }
}

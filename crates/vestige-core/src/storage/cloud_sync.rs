//! Hosted managed-sync backend (Vestige Cloud).
//!
//! This module is only compiled with the `cloud-sync` feature. It provides
//! [`HttpPortableSyncBackend`], an HTTP implementation of the
//! [`PortableSyncBackend`](super::sqlite::PortableSyncBackend) trait that
//! pull-merge-pushes the portable archive to a hosted blob endpoint.
//!
//! The merge/conflict engine is unchanged: this backend only moves bytes. The
//! authoritative `key -> namespace` mapping and per-user isolation live in the
//! hosted service; the client just presents an opaque sync key as a bearer
//! token. The default local-first build never links an HTTP client.
//!
//! ## Concurrency
//!
//! Two devices can each pull → merge → push. To avoid a lost update in the
//! GET↔PUT window, the backend uses optimistic concurrency: it captures the
//! object `ETag` on read and sends it as `If-Match` on write. The generic
//! [`sync_portable_archive`](super::sqlite::SqliteMemoryStore::sync_portable_archive)
//! driver calls `read_archive` then `write_archive` exactly once, so the ETag
//! captured during the pull is the precondition for the push. A
//! `412 Precondition Failed` means another device wrote in between; the caller
//! re-runs sync (the merge is idempotent and converges by `updated_at`).

use std::cell::RefCell;
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, ETAG, IF_MATCH};
use reqwest::StatusCode;

use super::portable::PortableArchive;
use super::sqlite::{PortableSyncBackend, Result, StorageError};

/// Default request timeout for cloud sync HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Blob path on the hosted service. One opaque blob per sync key (the service
/// derives the namespace from the key), so the client uses a fixed path.
const BLOB_PATH: &str = "/v1/blob";

/// HTTP-backed portable sync backend for Vestige Cloud.
///
/// Mirrors the shape of
/// [`FilePortableSyncBackend`](super::sqlite::FilePortableSyncBackend) but reads
/// and writes the archive over HTTPS with a per-user bearer key.
#[derive(Debug)]
pub struct HttpPortableSyncBackend {
    /// Base endpoint, e.g. `https://sync.vestige.dev`. No trailing slash.
    endpoint: String,
    /// Per-user sync key, presented as `Authorization: Bearer <key>`.
    sync_key: String,
    /// Blocking HTTP client (the trait is synchronous).
    client: Client,
    /// ETag captured on the most recent successful read, used as the `If-Match`
    /// precondition on the next write. `None` until the first read, or when the
    /// remote had no archive yet.
    last_etag: RefCell<Option<String>>,
}

impl HttpPortableSyncBackend {
    /// Build a cloud sync backend for `endpoint` authenticated with `sync_key`.
    ///
    /// A trailing slash on `endpoint` is trimmed so URL joining is predictable.
    pub fn new(endpoint: impl Into<String>, sync_key: impl Into<String>) -> Result<Self> {
        let endpoint = endpoint.into().trim_end_matches('/').to_string();
        let sync_key = sync_key.into();
        if endpoint.is_empty() {
            return Err(StorageError::Init(
                "cloud sync endpoint is empty (set VESTIGE_CLOUD_ENDPOINT)".to_string(),
            ));
        }
        if sync_key.is_empty() {
            return Err(StorageError::Init(
                "cloud sync key is empty (set VESTIGE_CLOUD_SYNC_KEY)".to_string(),
            ));
        }
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(concat!("vestige-cloud-sync/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| StorageError::Init(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            endpoint,
            sync_key,
            client,
            last_etag: RefCell::new(None),
        })
    }

    /// Full blob URL for this backend.
    fn blob_url(&self) -> String {
        format!("{}{}", self.endpoint, BLOB_PATH)
    }
}

impl PortableSyncBackend for HttpPortableSyncBackend {
    fn label(&self) -> String {
        format!("cloud:{}", self.endpoint)
    }

    fn read_archive(&self) -> Result<Option<PortableArchive>> {
        let resp = self
            .client
            .get(self.blob_url())
            .header(AUTHORIZATION, format!("Bearer {}", self.sync_key))
            .send()
            .map_err(|e| StorageError::Init(format!("cloud sync read failed: {e}")))?;

        match resp.status() {
            StatusCode::NOT_FOUND => {
                // No remote archive yet — first sync for this key.
                *self.last_etag.borrow_mut() = None;
                Ok(None)
            }
            StatusCode::OK => {
                // Capture the ETag for the matching If-Match write.
                let etag = resp
                    .headers()
                    .get(ETAG)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string());
                *self.last_etag.borrow_mut() = etag;

                let bytes = resp
                    .bytes()
                    .map_err(|e| StorageError::Init(format!("cloud sync read body failed: {e}")))?;
                let archive: PortableArchive = serde_json::from_slice(&bytes).map_err(|e| {
                    StorageError::Init(format!("failed to parse cloud sync archive: {e}"))
                })?;
                Ok(Some(archive))
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(StorageError::Init(
                "cloud sync rejected the sync key (401/403). Check your subscription and \
                 VESTIGE_CLOUD_SYNC_KEY."
                    .to_string(),
            )),
            other => Err(StorageError::Init(format!(
                "cloud sync read returned unexpected status {other}"
            ))),
        }
    }

    fn write_archive(&self, archive: &PortableArchive) -> Result<()> {
        let body = serde_json::to_vec(archive)
            .map_err(|e| StorageError::Init(format!("failed to serialize archive: {e}")))?;

        let mut req = self
            .client
            .put(self.blob_url())
            .header(AUTHORIZATION, format!("Bearer {}", self.sync_key))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);

        // Optimistic concurrency: only overwrite the object we pulled. If the
        // remote had no archive, require that it still doesn't exist (`If-Match: *`
        // would require existence, so we omit the header to allow first create).
        if let Some(etag) = self.last_etag.borrow_mut().take() {
            req = req.header(IF_MATCH, etag);
        }

        let resp = req
            .send()
            .map_err(|e| StorageError::Init(format!("cloud sync write failed: {e}")))?;

        match resp.status() {
            StatusCode::OK | StatusCode::CREATED | StatusCode::NO_CONTENT => Ok(()),
            StatusCode::PRECONDITION_FAILED => Err(StorageError::Init(
                "cloud sync conflict: another device updated your memory in between. \
                 Run `vestige sync --cloud` again to merge and retry."
                    .to_string(),
            )),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(StorageError::Init(
                "cloud sync rejected the sync key (401/403). Check your subscription and \
                 VESTIGE_CLOUD_SYNC_KEY."
                    .to_string(),
            )),
            StatusCode::PAYLOAD_TOO_LARGE => Err(StorageError::Init(
                "cloud sync archive too large for the hosted plan limit".to_string(),
            )),
            other => Err(StorageError::Init(format!(
                "cloud sync write returned unexpected status {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::portable::{PortableArchive, PortableTable, PORTABLE_ARCHIVE_FORMAT};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    fn sample_archive() -> PortableArchive {
        PortableArchive {
            archive_format: PORTABLE_ARCHIVE_FORMAT.to_string(),
            vestige_version: "test".to_string(),
            schema_version: 1,
            exported_at: chrono::Utc::now(),
            mode: "exact".to_string(),
            tables: vec![PortableTable {
                name: "knowledge_nodes".to_string(),
                columns: vec!["id".to_string()],
                rows: vec![],
            }],
        }
    }

    /// A captured request the mock observed, surfaced to the test thread.
    #[derive(Debug, Default, Clone)]
    struct CapturedRequest {
        method: String,
        authorization: Option<String>,
        if_match: Option<String>,
    }

    /// Minimal one-shot HTTP mock. `responder` builds the raw HTTP response
    /// string for the request line + headers it parsed. Returns the bound base
    /// URL and a receiver for the captured request.
    fn spawn_mock<F>(responder: F) -> (String, mpsc::Receiver<CapturedRequest>)
    where
        F: Fn(&CapturedRequest) -> String + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let text = String::from_utf8_lossy(&buf[..n]);
                let mut cap = CapturedRequest::default();
                for (i, line) in text.lines().enumerate() {
                    if i == 0 {
                        cap.method = line.split_whitespace().next().unwrap_or("").to_string();
                    } else if let Some(v) = line.strip_prefix("authorization: ") {
                        cap.authorization = Some(v.trim().to_string());
                    } else if let Some(v) = line.strip_prefix("if-match: ") {
                        cap.if_match = Some(v.trim().to_string());
                    }
                }
                let response = responder(&cap);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
                let _ = tx.send(cap);
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn http_response(status: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn new_rejects_empty_endpoint_and_key() {
        assert!(HttpPortableSyncBackend::new("", "key").is_err());
        assert!(HttpPortableSyncBackend::new("https://x", "").is_err());
        assert!(HttpPortableSyncBackend::new("https://x", "key").is_ok());
    }

    #[test]
    fn endpoint_trailing_slash_trimmed() {
        let be = HttpPortableSyncBackend::new("https://sync.example/", "k").unwrap();
        assert_eq!(be.blob_url(), "https://sync.example/v1/blob");
    }

    #[test]
    fn read_404_returns_none() {
        let (base, rx) = spawn_mock(|_| http_response("404 Not Found", "", ""));
        let be = HttpPortableSyncBackend::new(base, "secret").unwrap();
        let got = be.read_archive().expect("read ok");
        assert!(got.is_none());
        let cap = rx.recv().unwrap();
        assert_eq!(cap.method, "GET");
        assert_eq!(cap.authorization.as_deref(), Some("Bearer secret"));
    }

    #[test]
    fn read_200_parses_and_captures_etag() {
        let archive = sample_archive();
        let body = serde_json::to_string(&archive).unwrap();
        let (base, _rx) = spawn_mock(move |_| {
            http_response("200 OK", "ETag: \"v1-abc\"\r\n", &body)
        });
        let be = HttpPortableSyncBackend::new(base, "secret").unwrap();
        let got = be.read_archive().expect("read ok").expect("some archive");
        assert_eq!(got.archive_format, PORTABLE_ARCHIVE_FORMAT);
        // ETag captured for the next If-Match write.
        assert_eq!(be.last_etag.borrow().as_deref(), Some("\"v1-abc\""));
    }

    #[test]
    fn read_401_is_error() {
        let (base, _rx) = spawn_mock(|_| http_response("401 Unauthorized", "", ""));
        let be = HttpPortableSyncBackend::new(base, "bad").unwrap();
        assert!(be.read_archive().is_err());
    }

    #[test]
    fn write_sends_if_match_when_etag_present() {
        // Seed an etag as if a prior read happened.
        let (base, rx) = spawn_mock(|_| http_response("200 OK", "", ""));
        let be = HttpPortableSyncBackend::new(base, "secret").unwrap();
        *be.last_etag.borrow_mut() = Some("\"v1-abc\"".to_string());
        be.write_archive(&sample_archive()).expect("write ok");
        let cap = rx.recv().unwrap();
        assert_eq!(cap.method, "PUT");
        assert_eq!(cap.authorization.as_deref(), Some("Bearer secret"));
        assert_eq!(cap.if_match.as_deref(), Some("\"v1-abc\""));
    }

    #[test]
    fn write_omits_if_match_for_first_create() {
        let (base, rx) = spawn_mock(|_| http_response("201 Created", "", ""));
        let be = HttpPortableSyncBackend::new(base, "secret").unwrap();
        // No prior read → no etag → no If-Match (allow create).
        be.write_archive(&sample_archive()).expect("write ok");
        let cap = rx.recv().unwrap();
        assert_eq!(cap.method, "PUT");
        assert!(cap.if_match.is_none());
    }

    #[test]
    fn write_412_is_conflict_error() {
        let (base, _rx) = spawn_mock(|_| http_response("412 Precondition Failed", "", ""));
        let be = HttpPortableSyncBackend::new(base, "secret").unwrap();
        *be.last_etag.borrow_mut() = Some("\"stale\"".to_string());
        let err = be.write_archive(&sample_archive()).unwrap_err();
        assert!(err.to_string().contains("conflict"));
    }
}

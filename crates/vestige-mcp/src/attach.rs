//! One process serves a data directory; every other `vestige-mcp` attaches.
//!
//! The Strata log has exactly one writer. `.serve.lock` is an OS file lock
//! (`File::try_lock`) that the kernel releases when its holder dies, SIGKILL
//! included, so holding it is what makes a process the OWNER. The owner opens
//! the log, serves its own client, and listens for other local processes on
//! `.serve.sock`, a Unix socket in the data directory. A `vestige-mcp` that
//! finds the lock taken becomes a PROXY: it connects there, presents the
//! token from the endpoint file, and from then on copies JSON-RPC lines
//! between its own stdio and the owner. The owner runs one MCP session per
//! attached connection over the same storage, so every agent on a machine
//! shares one store and one writer.
//!
//! Windows, and a data directory whose path is too long for a socket
//! address, listen on a loopback TCP port instead. No transport ever leaves
//! the machine.
//!
//! `.serve.endpoint` holds `unix|<port> token pid`. It is created 0600 and
//! replaced by rename; the socket is 0600 too. Whoever can read the endpoint
//! file can already read the log, so the token adds no access beyond that;
//! it keeps other local users, and local processes that merely find the
//! port, from speaking MCP to the store. The proxy also checks it reached
//! that owner: on a Unix socket the kernel's peer credentials must name
//! this user and that pid, and the welcome line's pid must match the
//! endpoint. A listener that answers as a different pid is not the owner.
//!
//! FAILOVER. The owner is usually some agent's own server process, and that
//! agent's client kills it when the agent exits. Each proxy then answers the
//! requests it was still waiting on with an error (a tool call may already
//! have taken effect, so it is never resent) and runs the election again. The
//! first to take the lock becomes the owner in place; the others attach to
//! it. The client's `initialize` is replayed to the new owner under a private
//! id whose response is dropped, so the client keeps its session.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, TryLockError};
use std::io::{self, Write};
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use subtle::ConstantTimeEq;
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
    Lines,
};
use tokio::net::{TcpListener, TcpStream};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::protocol::stdio::{CappedLines, CappedRead, MAX_REQUEST_LINE, run_io};
use crate::server::McpServer;

/// The OS lock that elects the owner. `vestige-upgrade` holds it too.
pub const LOCK_FILE: &str = ".serve.lock";
/// `unix|<port> token pid` of the owner's attach listener.
pub const ENDPOINT_FILE: &str = ".serve.endpoint";
/// The owner's Unix socket, in the data directory.
pub const SOCKET_FILE: &str = ".serve.sock";
/// The canonical reflex IPC socket for hooks and CLI commands.
pub const REFLEX_SOCKET_FILE: &str = "reflex.sock";
/// A socket address holds 104 bytes on macOS and 108 on Linux, NUL
/// included. A longer socket path falls back to loopback TCP.
#[cfg(unix)]
const SOCKET_PATH_MAX: usize = 100;

const HELLO: &str = "vestige-attach/1";
const WELCOME: &str = "vestige-attached/1";
const REFUSED: &str = "vestige-refused/1";
/// Bound on each side of the attach handshake.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the owner waits for a new connection's hello. Clients send it
/// right after connecting.
const HELLO_TIMEOUT: Duration = Duration::from_secs(2);
/// A handshake line longer than this is not one.
const HANDSHAKE_LINE_MAX: u64 = 256;
/// Connections still in their handshake. A newer one evicts the oldest.
const MAX_HANDSHAKES: usize = 16;
/// Attached sessions one owner serves at once.
const MAX_ATTACHED: usize = 256;
const ELECTION_POLL: Duration = Duration::from_millis(100);
/// How long a new process waits for the lock holder to accept an attach. A
/// first launch on a v3 store holds the lock for the whole import.
pub const DEFAULT_ELECTION_WAIT: Duration = Duration::from_secs(120);
const ELECTION_WAIT_ENV: &str = "VESTIGE_ATTACH_WAIT_SECS";
/// Lines queued toward the owner before the proxy stops reading stdin.
const OWNER_QUEUE: usize = 1024;
/// Lines queued toward the client's stdout.
const STDOUT_QUEUE: usize = 256;
/// After the client closes stdin, how long the proxy keeps relaying the
/// owner's last answers.
const CLOSE_DRAIN: Duration = Duration::from_secs(20);
/// Bound on one tool call made through [`call_tool`].
const CALL_TIMEOUT: Duration = Duration::from_secs(600);
/// When the owner is lost, how long the proxy keeps reading answers the
/// owner had already sent.
const OWNER_LOST_DRAIN: Duration = Duration::from_secs(2);
/// Consecutive stdin read errors before a proxy gives up, as `run_io` does.
const MAX_STDIN_ERRORS: u32 = 5;

// ============================================================================
// Election
// ============================================================================

/// Take `.serve.lock` without waiting. `None` when another process holds it.
///
/// The lock file is owner-only: an exclusive lock can be taken through a
/// read-only descriptor, so a lock file other users can read is one they can
/// hold to lock the owner out. The data directory is tightened to 0700 too,
/// as v3 did for its token file, since the log holds every memory.
pub fn try_serve_lock(data_dir: &Path) -> io::Result<Option<File>> {
    let path = data_dir.join(LOCK_FILE);
    let mut options = File::options();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(&path)?;
    #[cfg(unix)]
    {
        owner_only(&path, 0o600);
        owner_only(data_dir, 0o700);
    }
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(err)) => Err(err),
    }
}

/// Drop group and other permission bits from a path this user owns. Never
/// loosens, and leaves a path owned by someone else alone.
#[cfg(unix)]
fn owner_only(path: &Path, mode: u32) {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let Ok(meta) = fs::metadata(path) else {
        return;
    };
    // Safety: geteuid has no preconditions and cannot fail.
    let me = unsafe { libc::geteuid() };
    if meta.uid() == me && meta.mode() & 0o077 != 0 {
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
}

/// What the election made this process.
pub enum Role {
    /// This process holds the lock and serves the store.
    Owner(File),
    /// Another process serves the store; this is a connection to it.
    Attached(Attachment),
}

/// `VESTIGE_ATTACH_WAIT_SECS`, else [`DEFAULT_ELECTION_WAIT`].
pub fn election_wait() -> Duration {
    std::env::var(ELECTION_WAIT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_ELECTION_WAIT)
}

/// Take the serve lock, or attach to the process that holds it.
///
/// The lock is tried first, every round: when the holder is gone the kernel
/// has already released it, whatever its endpoint file still says.
pub async fn elect(data_dir: &Path, wait: Duration) -> io::Result<Role> {
    let deadline = Instant::now() + wait;
    let mut announced = false;
    loop {
        if let Some(lock) = try_serve_lock(data_dir)? {
            return Ok(Role::Owner(lock));
        }
        if let Some(attachment) = attach(data_dir).await {
            return Ok(Role::Attached(attachment));
        }
        if Instant::now() >= deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "another Vestige process holds {} and did not accept an attach within {}s. \
                     It may be vestige-upgrade importing a v3 store, a vestige CLI command, or a \
                     server still opening the log. Set {ELECTION_WAIT_ENV} to wait longer.",
                    data_dir.join(LOCK_FILE).display(),
                    wait.as_secs()
                ),
            ));
        }
        if !announced {
            info!(
                "{} is held by another Vestige process; waiting to attach to it",
                data_dir.join(LOCK_FILE).display()
            );
            announced = true;
        }
        tokio::time::sleep(ELECTION_POLL).await;
    }
}

// ============================================================================
// Transport
// ============================================================================

/// One side of an attach connection, over a Unix socket or loopback TCP.
type ReadHalf = Box<dyn AsyncRead + Send + Unpin>;
type WriteHalf = Box<dyn AsyncWrite + Send + Unpin>;

/// How the owner is reached.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Addr {
    /// `.serve.sock` in the data directory.
    Unix,
    /// Windows Named Pipe (`\\.\pipe\vestige-<hash>`), zero firewall prompts & zero TIME_WAIT.
    Pipe(String),
    /// 127.0.0.1 on this port.
    Tcp(u16),
}

/// Deterministic local Named Pipe path for `data_dir` on Windows.
#[cfg(windows)]
fn named_pipe_path(data_dir: &Path) -> String {
    let canonical = fs::canonicalize(data_dir).unwrap_or_else(|_| data_dir.to_path_buf());
    let hash = blake3::hash(canonical.to_string_lossy().as_bytes());
    let hex = hash.to_hex();
    format!(r"\\.\pipe\vestige-{}", &hex[..16])
}

/// The socket path, when it fits in a socket address.
#[cfg(unix)]
fn socket_path(data_dir: &Path) -> Option<PathBuf> {
    let path = data_dir.join(SOCKET_FILE);
    (path.as_os_str().len() <= SOCKET_PATH_MAX).then_some(path)
}

enum Listener {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(UnixListener),
    #[cfg(windows)]
    Pipe {
        name: String,
        server: tokio::net::windows::named_pipe::NamedPipeServer,
    },
}

impl Listener {
    /// Listen on the data directory's Unix socket (Unix), local Named Pipe (Windows),
    /// else on a free loopback port. Called only while holding the serve lock.
    async fn bind(data_dir: &Path) -> io::Result<(Self, Addr)> {
        #[cfg(unix)]
        if let Some(path) = socket_path(data_dir) {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::remove_file(&path);
            let listener = UnixListener::bind(&path)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
            return Ok((Self::Unix(listener), Addr::Unix));
        }
        #[cfg(windows)]
        if std::env::var("VESTIGE_ATTACH_TCP").as_deref() != Ok("1") {
            let name = named_pipe_path(data_dir);
            match tokio::net::windows::named_pipe::ServerOptions::new()
                .first_pipe_instance(true)
                .reject_remote_clients(true)
                .create(&name)
            {
                Ok(server) => {
                    return Ok((
                        Self::Pipe {
                            name: name.clone(),
                            server,
                        },
                        Addr::Pipe(name),
                    ));
                }
                Err(err) => {
                    warn!("named pipe bind failed ({err}); falling back to loopback TCP");
                }
            }
        }
        #[cfg(not(any(unix, windows)))]
        let _ = data_dir;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let port = listener.local_addr()?.port();
        Ok((Self::Tcp(listener), Addr::Tcp(port)))
    }

    async fn accept(&mut self) -> io::Result<Option<(ReadHalf, WriteHalf)>> {
        match self {
            Self::Tcp(listener) => {
                let (stream, _) = listener.accept().await?;
                stream.set_nodelay(true)?;
                let (read, write) = stream.into_split();
                Ok(Some((Box::new(read), Box::new(write))))
            }
            #[cfg(unix)]
            Self::Unix(listener) => {
                let (stream, _) = listener.accept().await?;
                if !unix_peer_is_us(&stream) {
                    debug!("attach refused: the connecting process is another user");
                    return Ok(None);
                }
                let (read, write) = stream.into_split();
                Ok(Some((Box::new(read), Box::new(write))))
            }
            #[cfg(windows)]
            Self::Pipe { name, server } => {
                server.connect().await?;
                let next = tokio::net::windows::named_pipe::ServerOptions::new()
                    .reject_remote_clients(true)
                    .create(&*name)?;
                let connected = std::mem::replace(server, next);
                let (read, write) = tokio::io::split(connected);
                Ok(Some((Box::new(read), Box::new(write))))
            }
        }
    }
}

/// This user's uid. The attach socket is not a capability for other accounts.
#[cfg(unix)]
fn our_uid() -> u32 {
    // Safety: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

/// The process on the other end of a Unix socket is this user.
#[cfg(unix)]
fn unix_peer_is_us(stream: &tokio::net::UnixStream) -> bool {
    match stream.peer_cred() {
        Ok(cred) => cred.uid() == our_uid(),
        Err(err) => {
            debug!("attach refused: peer credentials are unreadable: {err}");
            false
        }
    }
}

/// The process on the other end is the owner named in the endpoint file.
///
/// `pid()` is `None` on platforms that do not report a peer pid. The welcome
/// line is still checked against the endpoint pid.
#[cfg(unix)]
fn unix_peer_is_owner(stream: &tokio::net::UnixStream, owner_pid: u32) -> bool {
    if !unix_peer_is_us(stream) {
        return false;
    }
    match stream.peer_cred() {
        Ok(cred) => match cred.pid() {
            Some(pid) if u32::try_from(pid).ok() == Some(owner_pid) => true,
            Some(pid) => {
                debug!("attach refused: socket peer pid {pid} is not the owner pid {owner_pid}");
                false
            }
            None => true,
        },
        Err(err) => {
            debug!("attach refused: peer credentials are unreadable: {err}");
            false
        }
    }
}

async fn connect(data_dir: &Path, endpoint: &Endpoint) -> io::Result<(ReadHalf, WriteHalf)> {
    match &endpoint.addr {
        Addr::Tcp(port) => {
            let stream = TcpStream::connect((Ipv4Addr::LOCALHOST, *port)).await?;
            stream.set_nodelay(true)?;
            let (read, write) = stream.into_split();
            Ok((Box::new(read), Box::new(write)))
        }
        #[cfg(windows)]
        Addr::Pipe(name) => {
            let _ = data_dir;
            let mut attempts = 0u32;
            let client = loop {
                match tokio::net::windows::named_pipe::ClientOptions::new().open(name) {
                    Ok(c) => break c,
                    Err(err) if err.raw_os_error() == Some(231) && attempts < 20 => {
                        attempts += 1;
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                    Err(err) => return Err(err),
                }
            };
            let (read, write) = tokio::io::split(client);
            Ok((Box::new(read), Box::new(write)))
        }
        #[cfg(not(windows))]
        Addr::Pipe(_) => {
            let _ = data_dir;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "the endpoint names a Windows named pipe, which this platform does not have",
            ))
        }
        #[cfg(unix)]
        Addr::Unix => {
            let stream = UnixStream::connect(data_dir.join(SOCKET_FILE)).await?;
            if !unix_peer_is_owner(&stream, endpoint.pid) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "the socket is not held by the owner named in the endpoint file",
                ));
            }
            let (read, write) = stream.into_split();
            Ok((Box::new(read), Box::new(write)))
        }
        #[cfg(not(unix))]
        Addr::Unix => {
            let _ = data_dir;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "the endpoint names a Unix socket, which this platform does not have",
            ))
        }
    }
}

// ============================================================================
// Endpoint file
// ============================================================================

/// Where the owner accepts attached sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    addr: Addr,
    token: String,
    pub pid: u32,
}

impl Endpoint {
    fn render(&self) -> String {
        let addr = match &self.addr {
            Addr::Unix => "unix".to_string(),
            Addr::Pipe(name) => format!("pipe:{name}"),
            Addr::Tcp(port) => port.to_string(),
        };
        format!("{addr} {} {}\n", self.token, self.pid)
    }

    fn parse(text: &str) -> Option<Self> {
        let mut parts = text.split_whitespace();
        let addr = match parts.next()? {
            "unix" => Addr::Unix,
            raw if raw.starts_with("pipe:") && raw.len() > 5 => Addr::Pipe(raw[5..].to_string()),
            port => Addr::Tcp(port.parse().ok().filter(|port: &u16| *port != 0)?),
        };
        let token = parts.next()?.to_string();
        let pid = parts.next()?.parse().ok()?;
        let well_formed = parts.next().is_none()
            && token.len() == 64
            && token.bytes().all(|byte| byte.is_ascii_hexdigit());
        well_formed.then_some(Self { addr, token, pid })
    }
}

/// The endpoint the current (or last) owner published, if it parses.
pub fn read_endpoint(data_dir: &Path) -> Option<Endpoint> {
    fs::read_to_string(data_dir.join(ENDPOINT_FILE))
        .ok()
        .and_then(|text| Endpoint::parse(&text))
}

fn new_token() -> io::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes)
        .map_err(|err| io::Error::other(format!("the OS random source failed: {err}")))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Write the endpoint file: 0600 from creation, then renamed into place so a
/// reader never sees half of it. Only the lock holder calls this.
fn publish_endpoint(data_dir: &Path, endpoint: &Endpoint) -> io::Result<()> {
    let staged = data_dir.join(format!("{ENDPOINT_FILE}.{}.tmp", std::process::id()));
    let _ = fs::remove_file(&staged);
    let mut options = File::options();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let written = (|| {
        let mut file = options.open(&staged)?;
        file.write_all(endpoint.render().as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&staged, data_dir.join(ENDPOINT_FILE))
    })();
    if written.is_err() {
        let _ = fs::remove_file(&staged);
        return written;
    }
    #[cfg(unix)]
    if endpoint.addr == Addr::Unix {
        let reflex_path = data_dir.join(REFLEX_SOCKET_FILE);
        let _ = fs::remove_file(&reflex_path);
        let _ = std::os::unix::fs::symlink(SOCKET_FILE, &reflex_path);
        if let Some(base) = directories::BaseDirs::new() {
            let user_vestige = base.home_dir().join(".vestige");
            if user_vestige.is_dir() && user_vestige != data_dir {
                let user_reflex = user_vestige.join(REFLEX_SOCKET_FILE);
                let _ = fs::remove_file(&user_reflex);
                let _ = std::os::unix::fs::symlink(data_dir.join(SOCKET_FILE), &user_reflex);
            }
        }
    }
    written
}

/// Remove the endpoint file while it still names `endpoint`, and this
/// owner's socket. The caller holds the lock, so no other owner can have
/// published in between.
fn retire_endpoint(data_dir: &Path, endpoint: &Endpoint) {
    if read_endpoint(data_dir).as_ref() == Some(endpoint) {
        let _ = fs::remove_file(data_dir.join(ENDPOINT_FILE));
    }
    if endpoint.addr == Addr::Unix {
        let _ = fs::remove_file(data_dir.join(SOCKET_FILE));
        let _ = fs::remove_file(data_dir.join(REFLEX_SOCKET_FILE));
        if let Some(base) = directories::BaseDirs::new() {
            let user_vestige = base.home_dir().join(".vestige");
            if user_vestige.is_dir() && user_vestige != data_dir {
                let _ = fs::remove_file(user_vestige.join(REFLEX_SOCKET_FILE));
            }
        }
    }
}

async fn read_handshake_line<R>(reader: &mut R) -> Option<String>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = String::new();
    let read = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        (&mut *reader).take(HANDSHAKE_LINE_MAX).read_line(&mut line),
    )
    .await;
    match read {
        Ok(Ok(n)) if n > 0 && line.ends_with('\n') => Some(line.trim_end().to_string()),
        _ => None,
    }
}

// ============================================================================
// Owner side
// ============================================================================

#[derive(Default)]
struct GateState {
    attached: usize,
    closing: bool,
}

/// Counts attached sessions, and closes the door only once none remain.
#[derive(Default)]
struct Gate {
    state: Mutex<GateState>,
    idle: Notify,
}

struct SessionGuard(Arc<Gate>);

impl Drop for SessionGuard {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.attached -= 1;
        if state.attached == 0 {
            self.0.idle.notify_waiters();
        }
    }
}

impl Gate {
    fn try_enter(self: &Arc<Self>) -> Option<SessionGuard> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closing || state.attached >= MAX_ATTACHED {
            return None;
        }
        state.attached += 1;
        Some(SessionGuard(Arc::clone(self)))
    }

    fn attached(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .attached
    }

    /// Wait until no session is attached, then refuse every later one. The
    /// check and the refusal happen under one lock, so no session can slip in
    /// between them.
    async fn close_when_idle(&self) {
        loop {
            // Registered before the check: `notify_waiters` reaches a
            // `Notified` from the moment it is created.
            let notified = self.idle.notified();
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if state.attached == 0 {
                    state.closing = true;
                    return;
                }
            }
            notified.await;
        }
    }
}

/// Starts (or finds) the owner's dashboard on the asked-for port for one
/// `vestige dashboard` run, or says why it cannot.
pub type DashboardStarter =
    Arc<dyn Fn(u16) -> BoxFuture<'static, Result<DashboardGrant, String>> + Send + Sync>;

/// The owner's dashboard as one `vestige dashboard` run holds it.
pub struct DashboardGrant {
    /// Where the dashboard answers. Its port is the running dashboard's,
    /// which may not be the one that run asked for.
    pub url: String,
    /// Keeps the dashboard serving while that run lasts. Dropping it gives
    /// the claim back; the owner stops a dashboard no run holds any more
    /// (unless it started the dashboard for itself).
    pub hold: Box<dyn std::any::Any + Send + Sync>,
}

/// What an owner offers attached connections.
struct Services {
    make_server: Box<dyn Fn() -> McpServer + Send + Sync>,
    dashboard: Option<DashboardStarter>,
}

/// The owner's side: a loopback listener plus the endpoint file naming it.
pub struct AttachPoint {
    data_dir: PathBuf,
    endpoint: Endpoint,
    gate: Arc<Gate>,
    accept: JoinHandle<()>,
}

impl AttachPoint {
    /// Bind 127.0.0.1 on a free port, publish the endpoint, and start
    /// accepting. `make_server` builds the MCP session for each attachment;
    /// `dashboard`, when given, serves `vestige dashboard` requests. Call only
    /// while holding the serve lock.
    pub async fn open<F>(
        data_dir: &Path,
        make_server: F,
        dashboard: Option<DashboardStarter>,
    ) -> io::Result<Self>
    where
        F: Fn() -> McpServer + Send + Sync + 'static,
    {
        let (listener, addr) = Listener::bind(data_dir).await?;
        let endpoint = Endpoint {
            addr,
            token: new_token()?,
            pid: std::process::id(),
        };
        publish_endpoint(data_dir, &endpoint)?;
        let gate = Arc::new(Gate::default());
        let accept = tokio::spawn(accept_loop(
            listener,
            Arc::from(endpoint.token.as_str()),
            Arc::clone(&gate),
            Arc::new(Services {
                make_server: Box::new(make_server),
                dashboard,
            }),
        ));
        info!(
            endpoint = ?endpoint.addr,
            "other Vestige clients on this machine attach to this server"
        );
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            endpoint,
            gate,
            accept,
        })
    }

    /// Sessions attached right now.
    pub fn attached(&self) -> usize {
        self.gate.attached()
    }

    /// Call once this process's own client is gone. Serves the attached
    /// sessions until the last one closes, then stops accepting and removes
    /// the endpoint file (on drop). The serve lock is still held throughout.
    pub async fn close(self) {
        let attached = self.gate.attached();
        if attached > 0 {
            info!(
                attached,
                "own client closed; still serving attached sessions until they close"
            );
        }
        self.gate.close_when_idle().await;
    }
}

impl Drop for AttachPoint {
    /// Stop accepting and retire the endpoint, whether or not `close` ran: a
    /// long-running CLI command drops its attach point without closing it.
    fn drop(&mut self) {
        self.accept.abort();
        retire_endpoint(&self.data_dir, &self.endpoint);
    }
}

/// Handshakes in progress, oldest first: (connection id, its eviction signal).
type PendingHandshakes = Arc<Mutex<VecDeque<(u64, Arc<Notify>)>>>;

async fn accept_loop(
    mut listener: Listener,
    token: Arc<str>,
    gate: Arc<Gate>,
    services: Arc<Services>,
) {
    // Connections still in their handshake, oldest first. Past the cap the
    // oldest is evicted rather than the newest refused, so connections that
    // never say hello cannot lock real clients out.
    let pending: PendingHandshakes = Arc::default();
    let mut next_id: u64 = 0;
    loop {
        let (read, write) = match listener.accept().await {
            Ok(Some(halves)) => halves,
            Ok(None) => continue,
            Err(err) => {
                warn!("attach listener could not accept a connection: {err}");
                tokio::time::sleep(ELECTION_POLL).await;
                continue;
            }
        };
        next_id += 1;
        let id = next_id;
        let evict = Arc::new(Notify::new());
        {
            let mut queue = pending.lock().unwrap_or_else(PoisonError::into_inner);
            while queue.len() >= MAX_HANDSHAKES {
                if let Some((_, oldest)) = queue.pop_front() {
                    oldest.notify_one();
                }
            }
            queue.push_back((id, Arc::clone(&evict)));
        }
        let pending = Arc::clone(&pending);
        let token = Arc::clone(&token);
        let gate = Arc::clone(&gate);
        let services = Arc::clone(&services);
        tokio::spawn(async move {
            let admitted = tokio::select! {
                admitted = admit(read, write, &token, &gate) => admitted,
                _ = evict.notified() => None,
            };
            pending
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|(entry, _)| *entry != id);
            let Some(admitted) = admitted else {
                return;
            };
            let pid = std::process::id();
            let Admitted {
                request,
                mut reader,
                mut writer,
                guard,
            } = admitted;
            match request {
                Request::Session => {
                    let welcome = format!("{WELCOME} {pid}\n");
                    if writer.write_all(welcome.as_bytes()).await.is_err()
                        || writer.flush().await.is_err()
                    {
                        return;
                    }
                    if let Err(err) = run_io((services.make_server)(), None, reader, writer).await {
                        debug!("attached session ended with an I/O error: {err}");
                    }
                }
                Request::Dashboard { port } => {
                    let started = match &services.dashboard {
                        Some(start) => start(port).await,
                        None => Err("this Vestige process does not serve a dashboard".to_string()),
                    };
                    let answer = match &started {
                        Ok(grant) => format!("{WELCOME} {pid} {}\n", grant.url),
                        Err(reason) => format!("{REFUSED} {}\n", reason.replace('\n', " ")),
                    };
                    if writer.write_all(answer.as_bytes()).await.is_err()
                        || writer.flush().await.is_err()
                        || started.is_err()
                    {
                        return;
                    }
                    // The lease stays open until `vestige dashboard` exits. It
                    // counts as an attached session, so this owner keeps
                    // running even after its own client leaves, and the
                    // grant keeps the dashboard serving. Dropping the grant
                    // at EOF lets the owner stop a dashboard nobody holds.
                    let mut sink = [0u8; 256];
                    while matches!(reader.read(&mut sink).await, Ok(n) if n > 0) {}
                    drop(started);
                }
            }
            drop(guard);
        });
    }
}

/// What an attaching connection asked for.
enum Request {
    /// An MCP session over this connection.
    Session,
    /// Start (or find) the dashboard, then hold the connection as a lease.
    Dashboard { port: u16 },
}

struct Admitted {
    request: Request,
    reader: BufReader<ReadHalf>,
    writer: WriteHalf,
    guard: SessionGuard,
}

/// Check the token, read what the connection wants, and count it.
async fn admit(
    read: ReadHalf,
    writer: WriteHalf,
    token: &str,
    gate: &Arc<Gate>,
) -> Option<Admitted> {
    let mut reader = BufReader::new(read);
    let hello = tokio::time::timeout(HELLO_TIMEOUT, read_handshake_line(&mut reader))
        .await
        .ok()??;
    let mut words = hello
        .strip_prefix(HELLO)
        .and_then(|rest| rest.strip_prefix(' '))
        .unwrap_or("")
        .split(' ');
    let presented = words.next().unwrap_or("");
    if !bool::from(presented.as_bytes().ct_eq(token.as_bytes())) {
        debug!("attach refused: wrong or missing token");
        return None;
    }
    let request = match (words.next(), words.next(), words.next()) {
        (None, _, _) => Request::Session,
        (Some("dashboard"), Some(port), None) => Request::Dashboard {
            port: port.parse().ok()?,
        },
        _ => {
            // The line carries the token; it stays out of the log.
            debug!("attach refused: unknown request");
            return None;
        }
    };
    let guard = gate.try_enter()?;
    Some(Admitted {
        request,
        reader,
        writer,
        guard,
    })
}

// ============================================================================
// Proxy side
// ============================================================================

/// A connection to the owner whose handshake has completed.
pub struct Attachment {
    lines: Lines<BufReader<ReadHalf>>,
    writer: WriteHalf,
    /// The owner's process id, from its handshake answer.
    pub owner_pid: u32,
}

/// The welcome names the owner the endpoint file published.
///
/// A listener that answers with any other pid is not that owner. The check
/// is the whole of owner authentication on TCP, where the kernel does not
/// report a peer pid; on a Unix socket [`connect`] has already required the
/// peer pid as well.
fn authenticated_owner_pid(endpoint: &Endpoint, welcome: &str) -> Option<u32> {
    let rest = welcome.strip_prefix(WELCOME)?.strip_prefix(' ')?;
    let pid = rest.split_whitespace().next()?.parse().ok()?;
    (pid == endpoint.pid).then_some(pid)
}

/// Connect to the published endpoint, send the token plus `request` (empty
/// for an MCP session), and return the owner's one-line answer plus the
/// endpoint the answer has to match.
async fn greet(
    data_dir: &Path,
    request: &str,
    answer_within: Duration,
) -> Option<(BufReader<ReadHalf>, WriteHalf, String, Endpoint)> {
    let endpoint = read_endpoint(data_dir)?;
    let (read, mut writer) = tokio::time::timeout(HANDSHAKE_TIMEOUT, connect(data_dir, &endpoint))
        .await
        .ok()?
        .ok()?;
    let hello = format!("{HELLO} {}{request}\n", endpoint.token);
    writer.write_all(hello.as_bytes()).await.ok()?;
    writer.flush().await.ok()?;
    let mut reader = BufReader::new(read);
    let mut answer = String::new();
    let read = tokio::time::timeout(
        answer_within,
        (&mut reader)
            .take(HANDSHAKE_LINE_MAX)
            .read_line(&mut answer),
    )
    .await;
    match read {
        Ok(Ok(n)) if n > 0 && answer.ends_with('\n') => {
            Some((reader, writer, answer.trim_end().to_string(), endpoint))
        }
        _ => None,
    }
}

async fn attach(data_dir: &Path) -> Option<Attachment> {
    let (reader, writer, welcome, endpoint) = greet(data_dir, "", HANDSHAKE_TIMEOUT).await?;
    let owner_pid = authenticated_owner_pid(&endpoint, &welcome)?;
    Some(Attachment {
        lines: reader.lines(),
        writer,
        owner_pid,
    })
}

/// What the client has said that a new owner must hear again, and the
/// requests still waiting on an answer.
struct ClientSession {
    initialize: Option<Value>,
    initialize_answered: bool,
    initialized: bool,
    in_flight: HashMap<String, Value>,
    /// Id of the replayed `initialize`; its response never reaches the client.
    replay_id: Value,
}

impl ClientSession {
    fn new() -> Self {
        let nonce = new_token()
            .map(|token| token[..16].to_string())
            .unwrap_or_default();
        Self {
            initialize: None,
            initialize_answered: false,
            initialized: false,
            in_flight: HashMap::new(),
            replay_id: Value::String(format!(
                "vestige-attach-replay-{}-{nonce}",
                std::process::id()
            )),
        }
    }

    /// Record a line on its way from the client to the owner.
    fn client_line(&mut self, line: &str) {
        let Ok(Value::Object(message)) = serde_json::from_str::<Value>(line) else {
            return;
        };
        // No method: the client answering a request from the server.
        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return;
        };
        match method {
            "initialize" => {
                self.initialize = Some(Value::Object(message));
                self.initialize_answered = false;
            }
            "notifications/initialized" => self.initialized = true,
            _ => {
                if let Some(id) = message.get("id").filter(|id| !id.is_null()) {
                    self.in_flight.insert(id.to_string(), id.clone());
                }
            }
        }
    }

    /// Record a line on its way from the owner to the client. `false` for the
    /// answer to a replayed handshake, which the client must not see.
    fn owner_line(&mut self, line: &str) -> bool {
        let Ok(Value::Object(message)) = serde_json::from_str::<Value>(line) else {
            return true;
        };
        if message.contains_key("method") {
            return true;
        }
        let Some(id) = message.get("id") else {
            return true;
        };
        if *id == self.replay_id {
            return false;
        }
        if self
            .initialize
            .as_ref()
            .and_then(|init| init.get("id"))
            .is_some_and(|init_id| init_id == id)
        {
            self.initialize_answered = true;
        }
        self.in_flight.remove(&id.to_string());
        true
    }

    /// Error answers for every request the lost owner never answered.
    fn abandon_in_flight(&mut self) -> Vec<String> {
        self.in_flight
            .drain()
            .map(|(_, id)| {
                let error = json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32603,
                        "message": "The Vestige server this session was attached to stopped \
                                    before answering. The call may or may not have taken \
                                    effect; check before retrying it."
                    }
                });
                format!("{error}\n")
            })
            .collect()
    }

    /// Lines that give a new owner this client's handshake. An `initialize`
    /// the client already has an answer to goes under the replay id; one it
    /// is still waiting on keeps its own id, so the new owner answers it.
    fn replay(&self) -> Vec<String> {
        let Some(init) = &self.initialize else {
            return Vec::new();
        };
        let mut init = init.clone();
        if self.initialize_answered {
            init["id"] = self.replay_id.clone();
        }
        let mut lines = vec![format!("{init}\n")];
        if self.initialized {
            lines.push("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n".into());
        }
        lines
    }
}

/// How a proxied session ended.
pub enum ProxyEnd {
    /// The client closed stdin (or stopped reading stdout).
    Closed,
    /// The owner went away and this process took the lock. Serve the store
    /// in place with [`PromotedClient::serve`].
    Promoted { lock: File, client: PromotedClient },
}

/// Lines from stdin that survive cancellation. Bytes already taken live in
/// [`CappedLines`], which outlives a cancelled call, so a line the relay's
/// other branches interrupt is completed on the next call, or handed whole
/// to a promoted server. A line over [`MAX_REQUEST_LINE`] is discarded
/// through its newline instead of being buffered.
struct StdinLines {
    lines: CappedLines<BufReader<tokio::io::Stdin>>,
}

enum StdinEvent {
    Line(String),
    Eof,
    TooLong,
}

impl StdinLines {
    fn new() -> Self {
        Self {
            lines: CappedLines::new(BufReader::new(tokio::io::stdin())),
        }
    }

    /// The next stdin event. Invalid UTF-8 is an `InvalidData` error, which
    /// the relay counts and skips, as `run_io` does for its own stdin.
    async fn next_line(&mut self) -> io::Result<StdinEvent> {
        Ok(match self.lines.next().await? {
            CappedRead::Line(line) => StdinEvent::Line(line),
            CappedRead::Eof => StdinEvent::Eof,
            CappedRead::TooLong => StdinEvent::TooLong,
        })
    }

    async fn finish_discard(&mut self) -> io::Result<()> {
        self.lines.finish_discard().await
    }

    fn take_pending(&mut self) -> Vec<u8> {
        self.lines.take_pending()
    }

    fn into_inner(self) -> BufReader<tokio::io::Stdin> {
        self.lines.into_inner()
    }
}

/// A proxied client whose server now runs in this process.
pub struct PromotedClient {
    stdin: BufReader<tokio::io::Stdin>,
    /// The replayed handshake, then any partly read client line.
    prefix: Vec<u8>,
    replay_id: Value,
    stdout: mpsc::Sender<String>,
    stdout_writer: JoinHandle<io::Result<()>>,
}

enum Relayed {
    ClientClosed,
    OwnerLost,
}

async fn write_lines<W>(mut sink: W, mut lines: mpsc::Receiver<String>) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    while let Some(line) = lines.recv().await {
        sink.write_all(line.as_bytes()).await?;
        sink.flush().await?;
    }
    sink.shutdown().await
}

/// Copy this process's stdio to and from the owner until the client leaves
/// or the owner does. On owner loss, elect again and carry on.
pub async fn proxy_stdio(
    first: Attachment,
    data_dir: &Path,
    wait: Duration,
) -> io::Result<ProxyEnd> {
    let (stdout, stdout_rx) = mpsc::channel::<String>(STDOUT_QUEUE);
    let stdout_writer = tokio::spawn(write_lines(tokio::io::stdout(), stdout_rx));
    let mut stdin = StdinLines::new();
    let mut session = ClientSession::new();
    let mut attachment = first;

    loop {
        let owner_pid = attachment.owner_pid;
        match relay(&mut stdin, attachment, &stdout, &mut session).await {
            Relayed::ClientClosed => {
                drop(stdout);
                let _ = tokio::time::timeout(CLOSE_DRAIN, stdout_writer).await;
                return Ok(ProxyEnd::Closed);
            }
            Relayed::OwnerLost => {
                warn!(
                    owner_pid,
                    "the Vestige server this client was attached to went away; electing another"
                );
                for line in session.abandon_in_flight() {
                    if stdout.send(line).await.is_err() {
                        return Ok(ProxyEnd::Closed);
                    }
                }
                match elect(data_dir, wait).await? {
                    Role::Attached(next) => {
                        info!(owner_pid = next.owner_pid, "reattached to the new owner");
                        attachment = next;
                    }
                    Role::Owner(lock) => {
                        if let Err(err) = stdin.finish_discard().await {
                            warn!(
                                "could not finish an oversized stdin line before taking over: {err}"
                            );
                        }
                        let mut prefix = session.replay().concat().into_bytes();
                        prefix.append(&mut stdin.take_pending());
                        return Ok(ProxyEnd::Promoted {
                            lock,
                            client: PromotedClient {
                                stdin: stdin.into_inner(),
                                prefix,
                                replay_id: session.replay_id.clone(),
                                stdout,
                                stdout_writer,
                            },
                        });
                    }
                }
            }
        }
    }
}

/// One attachment's worth of relaying.
async fn relay(
    stdin: &mut StdinLines,
    attachment: Attachment,
    stdout: &mpsc::Sender<String>,
    session: &mut ClientSession,
) -> Relayed {
    let Attachment {
        lines: mut owner_lines,
        writer,
        ..
    } = attachment;
    let (owner_tx, owner_rx) = mpsc::channel::<String>(OWNER_QUEUE);
    let owner_writer = tokio::spawn(write_lines(writer, owner_rx));
    for line in session.replay() {
        // A fresh queue holds the two replay lines.
        let _ = owner_tx.try_send(line);
    }
    // A client line the owner queue had no room for; stdin is not read again
    // until it is queued.
    let mut blocked: Option<String> = None;
    let mut read_errors: u32 = 0;

    let outcome = loop {
        tokio::select! {
            line = stdin.next_line(), if blocked.is_none() => match line {
                Ok(StdinEvent::Line(line)) => {
                    read_errors = 0;
                    if line.trim().is_empty() {
                        continue;
                    }
                    // Recorded once queued (or once the owner is known to be
                    // gone). A request the owner never got is then answered by
                    // `abandon_in_flight`, like one it got and did not answer.
                    let line = format!("{line}\n");
                    match owner_tx.try_send(line.clone()) {
                        Ok(()) => session.client_line(&line),
                        Err(mpsc::error::TrySendError::Full(line)) => blocked = Some(line),
                        Err(mpsc::error::TrySendError::Closed(line)) => {
                            session.client_line(&line);
                            break Relayed::OwnerLost;
                        }
                    }
                }
                Ok(StdinEvent::Eof) => break Relayed::ClientClosed,
                Ok(StdinEvent::TooLong) => {
                    read_errors = 0;
                    warn!("refusing a proxied request line longer than {MAX_REQUEST_LINE} bytes");
                    let notice = format!(
                        "{}\n",
                        json!({
                            "jsonrpc": "2.0",
                            "id": null,
                            "error": {
                                "code": -32600,
                                "message": format!(
                                    "request line exceeds {MAX_REQUEST_LINE} bytes"
                                ),
                            }
                        })
                    );
                    if stdout.send(notice).await.is_err() {
                        break Relayed::ClientClosed;
                    }
                }
                Err(err) => {
                    read_errors += 1;
                    warn!("reading stdin failed ({read_errors}/{MAX_STDIN_ERRORS}): {err}");
                    if read_errors >= MAX_STDIN_ERRORS {
                        break Relayed::ClientClosed;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
            permit = owner_tx.reserve(), if blocked.is_some() => match permit {
                Ok(permit) => {
                    let line = blocked.take().unwrap_or_default();
                    session.client_line(&line);
                    permit.send(line);
                }
                Err(_) => {
                    if let Some(line) = blocked.take() {
                        session.client_line(&line);
                    }
                    break Relayed::OwnerLost;
                }
            },
            line = owner_lines.next_line() => match line {
                Ok(Some(line)) => {
                    if session.owner_line(&line) && stdout.send(format!("{line}\n")).await.is_err() {
                        // Nobody reads our stdout any more.
                        break Relayed::ClientClosed;
                    }
                }
                Ok(None) | Err(_) => {
                    if let Some(line) = blocked.take() {
                        session.client_line(&line);
                    }
                    break Relayed::OwnerLost;
                }
            },
        }
    };

    match outcome {
        Relayed::ClientClosed => {
            // Closing the queue ends the writer, which half-closes the socket:
            // the owner sees EOF, finishes what is in flight, and closes.
            drop(owner_tx);
            let drained = tokio::time::timeout(CLOSE_DRAIN, async {
                while let Ok(Some(line)) = owner_lines.next_line().await {
                    if session.owner_line(&line) && stdout.send(format!("{line}\n")).await.is_err()
                    {
                        break;
                    }
                }
            })
            .await;
            if drained.is_err() {
                warn!(
                    "the Vestige server did not finish answering within {CLOSE_DRAIN:?} of stdin EOF"
                );
            }
            owner_writer.abort();
            Relayed::ClientClosed
        }
        Relayed::OwnerLost => {
            // The owner may have answered before it went (the writer can fail
            // first). Those answers are real and reach the client, instead of
            // the lost-owner error for their ids.
            let _ = tokio::time::timeout(OWNER_LOST_DRAIN, async {
                while let Ok(Some(line)) = owner_lines.next_line().await {
                    if session.owner_line(&line) && stdout.send(format!("{line}\n")).await.is_err()
                    {
                        break;
                    }
                }
            })
            .await;
            owner_writer.abort();
            Relayed::OwnerLost
        }
    }
}

impl PromotedClient {
    /// Serve this client from `server`, which runs in this process: the
    /// replayed handshake first, then the rest of stdin. The answer to a
    /// replayed `initialize` is dropped before it reaches stdout.
    pub async fn serve(self, server: McpServer) -> io::Result<()> {
        let Self {
            stdin,
            prefix,
            replay_id,
            stdout,
            stdout_writer,
        } = self;
        let reader = std::io::Cursor::new(prefix).chain(stdin);
        let (server_out, filter_in) = tokio::io::duplex(64 * 1024);
        let filter = tokio::spawn(async move {
            let mut lines = BufReader::new(filter_in).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if answers(&line, &replay_id) {
                    continue;
                }
                if stdout.send(format!("{line}\n")).await.is_err() {
                    break;
                }
            }
        });
        let result = run_io(server, None, reader, server_out).await;
        let _ = tokio::time::timeout(CLOSE_DRAIN, filter).await;
        let _ = tokio::time::timeout(CLOSE_DRAIN, stdout_writer).await;
        result
    }
}

/// `line` is a response (no method) whose id is `id`.
fn answers(line: &str, id: &Value) -> bool {
    matches!(
        serde_json::from_str::<Value>(line),
        Ok(Value::Object(message)) if !message.contains_key("method") && message.get("id") == Some(id)
    )
}

// ============================================================================
// One-shot calls (CLI)
// ============================================================================

/// The owner's dashboard, held open for as long as this lease lives.
pub struct DashboardLease {
    /// Where the dashboard answers.
    pub url: String,
    /// The process serving it.
    pub owner_pid: u32,
    reader: BufReader<ReadHalf>,
    _writer: WriteHalf,
}

impl DashboardLease {
    /// Wait until the owner goes away (the connection closes).
    pub async fn closed(mut self) {
        let mut sink = [0u8; 256];
        while matches!(self.reader.read(&mut sink).await, Ok(n) if n > 0) {}
    }
}

/// Ask the process serving `data_dir` to serve its dashboard on `port` (or
/// to name the one it already runs). The lease keeps that process serving.
pub async fn request_dashboard(data_dir: &Path, port: u16) -> io::Result<DashboardLease> {
    let request = format!(" dashboard {port}");
    let Some((reader, writer, answer, endpoint)) =
        greet(data_dir, &request, Duration::from_secs(30)).await
    else {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            format!(
                "no Vestige server accepted an attach for {}",
                data_dir.display()
            ),
        ));
    };
    if let Some(reason) = answer
        .strip_prefix(REFUSED)
        .map(|rest| rest.trim().to_string())
    {
        // A definite answer: callers stop asking instead of retrying.
        return Err(io::Error::new(io::ErrorKind::ConnectionRefused, reason));
    }
    let owner_pid = authenticated_owner_pid(&endpoint, &answer);
    let url = answer
        .strip_prefix(WELCOME)
        .and_then(|rest| rest.strip_prefix(' '))
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .map(str::to_string);
    match (owner_pid, url) {
        (Some(owner_pid), Some(url)) => Ok(DashboardLease {
            url,
            owner_pid,
            reader,
            _writer: writer,
        }),
        _ => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "the process that answered is not the owner named by the endpoint file: {answer}"
            ),
        )),
    }
}

/// The pid of the process serving `data_dir`, when it answers the attach
/// handshake. The endpoint file alone can name a process that is gone.
pub async fn probe_owner(data_dir: &Path) -> Option<u32> {
    let (_reader, _writer, welcome, endpoint) = greet(data_dir, "", HANDSHAKE_TIMEOUT).await?;
    authenticated_owner_pid(&endpoint, &welcome)
}

/// [`probe_owner`] for synchronous code, on its own thread and runtime so it
/// works whether or not the caller is inside one.
pub fn probe_owner_blocking(data_dir: &Path) -> Option<u32> {
    let dir = data_dir.to_path_buf();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()?
            .block_on(probe_owner(&dir))
    })
    .join()
    .ok()
    .flatten()
}

/// Call one MCP tool through the process serving `data_dir`, as a
/// short-lived client. Returns the tool's structured result.
pub async fn call_tool(data_dir: &Path, name: &str, arguments: Value) -> io::Result<Value> {
    let Some(attachment) = attach(data_dir).await else {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            format!(
                "no Vestige server accepted an attach for {}",
                data_dir.display()
            ),
        ));
    };
    tokio::time::timeout(CALL_TIMEOUT, call_over(attachment, name, arguments))
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                format!("{name} did not answer within {}s", CALL_TIMEOUT.as_secs()),
            )
        })?
}

/// Synchronous blocking wrapper for [`call_tool`].
pub fn call_tool_blocking(data_dir: &Path, name: &str, arguments: Value) -> io::Result<Value> {
    let dir = data_dir.to_path_buf();
    let name = name.to_string();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(io::Error::other)?
            .block_on(call_tool(&dir, &name, arguments))
    })
    .join()
    .map_err(|_| io::Error::other("call_tool thread panicked"))?
}

async fn call_over(attachment: Attachment, name: &str, arguments: Value) -> io::Result<Value> {
    let Attachment {
        mut lines,
        mut writer,
        ..
    } = attachment;
    // hardcode-scan: protocol (JSON-RPC / JSON-schema keys)
    let requests = [
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "vestige-cli", "version": env!("CARGO_PKG_VERSION")}
            }
        }),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }),
    ];
    for request in &requests {
        writer.write_all(format!("{request}\n").as_bytes()).await?;
    }
    writer.flush().await?;
    let answer = loop {
        let Some(line) = lines.next_line().await? else {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("the Vestige server closed the connection before answering {name}"),
            ));
        };
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if message.get("method").is_none() && message.get("id") == Some(&json!(2)) {
            break message;
        }
    };
    let _ = writer.shutdown().await;
    tool_result(name, answer)
}

/// The structured result of a `tools/call` answer, or its error text.
fn tool_result(name: &str, answer: Value) -> io::Result<Value> {
    if let Some(error) = answer.get("error") {
        return Err(io::Error::other(format!("{name} failed: {error}")));
    }
    let result = answer.get("result").cloned().unwrap_or(Value::Null);
    let text = result
        .get("content")
        .and_then(Value::as_array)
        .and_then(|content| content.first())
        .and_then(|item| item.get("text"))
        .and_then(Value::as_str)
        .map(str::to_string);
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Err(io::Error::other(format!(
            "{name} failed: {}",
            text.unwrap_or_else(|| result.to_string())
        )));
    }
    if let Some(structured) = result.get("structuredContent")
        && !structured.is_null()
    {
        return Ok(structured.clone());
    }
    match text {
        Some(text) => Ok(serde_json::from_str(&text).unwrap_or(Value::String(text))),
        None => Ok(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn token() -> String {
        "ab".repeat(32)
    }

    #[test]
    fn endpoint_round_trips_and_rejects_malformed_text() {
        for addr in [Addr::Tcp(43_210), Addr::Unix] {
            let endpoint = Endpoint {
                addr,
                token: token(),
                pid: 77,
            };
            assert_eq!(Endpoint::parse(&endpoint.render()), Some(endpoint));
        }
        for bad in [
            "",
            "0 {t} 1",
            "43210 short 1",
            "43210 {t}",
            "43210 {t} 1 extra",
            "port {t} 1",
            "43210 {z} 1",
        ] {
            let text = bad
                .replace("{t}", &token())
                .replace("{z}", &"zz".repeat(32));
            assert_eq!(Endpoint::parse(&text), None, "{text:?} must not parse");
        }
    }

    #[test]
    fn endpoint_file_is_published_owner_only_and_retired_only_while_it_is_ours() {
        let dir = tempfile::tempdir().unwrap();
        let ours = Endpoint {
            addr: Addr::Tcp(1234),
            token: token(),
            pid: 1,
        };
        publish_endpoint(dir.path(), &ours).unwrap();
        assert_eq!(read_endpoint(dir.path()), Some(ours.clone()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(dir.path().join(ENDPOINT_FILE))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        let theirs = Endpoint {
            addr: Addr::Tcp(4321),
            token: "cd".repeat(32),
            pid: 2,
        };
        publish_endpoint(dir.path(), &theirs).unwrap();
        retire_endpoint(dir.path(), &ours);
        assert_eq!(read_endpoint(dir.path()), Some(theirs.clone()));
        retire_endpoint(dir.path(), &theirs);
        assert_eq!(read_endpoint(dir.path()), None);
    }

    #[test]
    fn serve_lock_is_exclusive_and_released_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let held = try_serve_lock(dir.path()).unwrap().expect("first lock");
        assert!(try_serve_lock(dir.path()).unwrap().is_none());
        drop(held);
        assert!(try_serve_lock(dir.path()).unwrap().is_some());
    }

    #[test]
    fn replay_keeps_an_unanswered_initialize_and_hides_an_answered_one() {
        let mut session = ClientSession::new();
        session.client_line(r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}"#);
        let replay = session.replay();
        assert_eq!(replay.len(), 1);
        let init: Value = serde_json::from_str(&replay[0]).unwrap();
        assert_eq!(init["id"], 0, "an unanswered initialize keeps its own id");

        assert!(session.owner_line(r#"{"jsonrpc":"2.0","id":0,"result":{}}"#));
        session.client_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        let replay = session.replay();
        assert_eq!(replay.len(), 2);
        let init: Value = serde_json::from_str(&replay[0]).unwrap();
        assert_eq!(init["id"], session.replay_id);
        assert_eq!(init["method"], "initialize");
        assert!(replay[1].contains("notifications/initialized"));

        let echo = format!(
            r#"{{"jsonrpc":"2.0","id":{},"result":{{}}}}"#,
            session.replay_id
        );
        assert!(
            !session.owner_line(&echo),
            "the replayed handshake's answer must not reach the client"
        );
    }

    #[test]
    fn unanswered_requests_are_abandoned_with_errors_and_answered_ones_are_not() {
        let mut session = ClientSession::new();
        session.client_line(r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{}}"#);
        session.client_line(r#"{"jsonrpc":"2.0","id":"x","method":"tools/list"}"#);
        session.client_line(r#"{"jsonrpc":"2.0","method":"notifications/cancelled"}"#);
        session.client_line(r#"{"jsonrpc":"2.0","id":9,"result":{}}"#);
        assert!(session.owner_line(r#"{"jsonrpc":"2.0","id":"x","result":{}}"#));
        assert!(
            session.owner_line(r#"{"jsonrpc":"2.0","method":"notifications/message","params":{}}"#)
        );

        let errors = session.abandon_in_flight();
        assert_eq!(errors.len(), 1);
        let error: Value = serde_json::from_str(&errors[0]).unwrap();
        assert_eq!(error["id"], 7);
        assert_eq!(error["error"]["code"], -32603);
        assert!(session.abandon_in_flight().is_empty());
    }

    #[test]
    fn tool_result_prefers_structured_content_and_surfaces_errors() {
        let ok = json!({"result": {
            "content": [{"type": "text", "text": "{\"path\":\"a\"}"}],
            "structuredContent": {"path": "b"},
            "isError": false
        }});
        assert_eq!(tool_result("t", ok).unwrap()["path"], "b");
        let text_only =
            json!({"result": {"content": [{"type": "text", "text": "{\"path\":\"a\"}"}]}});
        assert_eq!(tool_result("t", text_only).unwrap()["path"], "a");
        let failed =
            json!({"result": {"content": [{"type": "text", "text": "nope"}], "isError": true}});
        assert!(
            tool_result("t", failed)
                .unwrap_err()
                .to_string()
                .contains("nope")
        );
        let rpc_error = json!({"error": {"code": -32601, "message": "missing"}});
        assert!(tool_result("t", rpc_error).is_err());
    }

    #[tokio::test]
    async fn gate_refuses_new_sessions_only_after_the_last_one_leaves() {
        let gate = Arc::new(Gate::default());
        let first = gate.try_enter().expect("open gate admits");
        let closer = {
            let gate = Arc::clone(&gate);
            tokio::spawn(async move { gate.close_when_idle().await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!closer.is_finished(), "must wait for the attached session");
        let second = gate
            .try_enter()
            .expect("still open while a session is attached");
        drop(first);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!closer.is_finished());
        drop(second);
        tokio::time::timeout(Duration::from_secs(5), closer)
            .await
            .expect("closes once idle")
            .unwrap();
        assert!(gate.try_enter().is_none(), "closed gate admits nobody");
    }

    #[test]
    fn welcome_pid_must_be_the_endpoint_pid() {
        let endpoint = Endpoint {
            addr: Addr::Tcp(9),
            token: token(),
            pid: 4242,
        };
        assert_eq!(
            authenticated_owner_pid(&endpoint, "vestige-attached/1 4242"),
            Some(4242)
        );
        assert_eq!(
            authenticated_owner_pid(&endpoint, "vestige-attached/1 4242 http://127.0.0.1:1"),
            Some(4242)
        );
        assert_eq!(
            authenticated_owner_pid(&endpoint, "vestige-attached/1 99999"),
            None,
            "a different pid is not the published owner"
        );
        assert_eq!(
            authenticated_owner_pid(&endpoint, "vestige-refused/1 no dashboard"),
            None
        );
    }

    /// #370 B18. The proxy used to trust whatever pid the listener wrote
    /// into the welcome line.
    #[tokio::test]
    async fn attach_refuses_a_welcome_from_a_different_pid() {
        let dir = tempfile::tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let endpoint = Endpoint {
            addr: Addr::Tcp(port),
            token: token(),
            pid: 4242,
        };
        publish_endpoint(dir.path(), &endpoint).unwrap();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 512];
            let _ = sock.read(&mut buf).await.unwrap();
            sock.write_all(b"vestige-attached/1 99999\n").await.unwrap();
            let _ = done_tx.send(());
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let attached = attach(dir.path()).await;
        done_rx.await.expect("the impostor did answer");
        assert!(
            attached.is_none(),
            "a welcome pid that is not the endpoint's pid is not the owner"
        );
    }

    #[tokio::test]
    async fn attach_accepts_the_published_owner_pid() {
        let dir = tempfile::tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let owner_pid = 4242u32;
        let endpoint = Endpoint {
            addr: Addr::Tcp(port),
            token: token(),
            pid: owner_pid,
        };
        publish_endpoint(dir.path(), &endpoint).unwrap();
        let expect = endpoint.token.clone();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 512];
            let n = sock.read(&mut buf).await.unwrap();
            let hello = std::str::from_utf8(&buf[..n]).unwrap();
            assert!(
                hello.contains(&expect),
                "the proxy presents the endpoint token: {hello}"
            );
            let line = format!("vestige-attached/1 {owner_pid}\n");
            sock.write_all(line.as_bytes()).await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let attached = attach(dir.path())
            .await
            .expect("the published pid is the owner");
        assert_eq!(attached.owner_pid, owner_pid);
    }
}

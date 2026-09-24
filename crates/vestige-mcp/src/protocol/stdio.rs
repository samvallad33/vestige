//! stdio Transport for MCP
//!
//! Handles JSON-RPC communication over stdin/stdout.
//! v1.9.2: Async tokio I/O with error resilience.
//! Requests are dispatched concurrently: the read loop never awaits a handler
//! before reading the next line. See [`run_io`].

use serde_json::Value;
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

use super::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::server::McpServer;

/// Maximum consecutive I/O errors before giving up
const MAX_CONSECUTIVE_ERRORS: u32 = 5;

/// Requests served concurrently before a newly spawned one waits for a permit.
///
/// INVARIANT this number is chosen against: the process must not put more
/// blocking work on the tokio runtime than the runtime has worker threads.
/// `main` builds `Builder::new_multi_thread()` with no `worker_threads`,
/// which sizes the pool to the CPU count, so the floor is a 4-core machine.
/// The concurrent consumers of those threads are: these MAX_INFLIGHT stdio
/// handlers, the dashboard's axum server (the HTTP MCP transport in
/// `protocol/http.rs` is itself capped at `CONCURRENCY_LIMIT`), at most one
/// consolidation worker (`SqliteMemoryStore::run_consolidation` is
/// synchronous and occupies one worker for its whole run;
/// `McpServer::claim_consolidation` is what holds it to one, and before that
/// claim existed several handlers could each spawn their own), plus the
/// single writer task here and the reader.
///
/// The cap also bounds how many blocking `std::sync::Mutex<Connection>` guards
/// can queue on the one reader/writer connection pair in `vestige-core`
/// (`storage/sqlite/mod.rs`), which is the residual serialization this
/// transport change does NOT remove.
const MAX_INFLIGHT: usize = 16;

/// Requests read off stdin and not yet finished, before the read loop waits.
///
/// The in-flight cap alone does not bound this: a task waiting for a permit
/// still holds its parsed request. Without this second cap a client can push
/// work into the process faster than it is served and the queue grows in
/// memory rather than in the pipe. Reaching it reinstates head-of-line waiting
/// at MAX_PENDING outstanding requests.
///
/// The wait at this cap polls stdin EOF and the writer's death as well as the
/// task set, because both can happen while it waits and neither would be seen
/// otherwise. What it cannot see is EOF sitting behind input the client has
/// already written and this loop has not read: EOF is only reachable once that
/// input is consumed, and consuming it is what the cap exists to stop. That
/// case ends when the tasks finish or the writer dies, both of which the wait
/// does observe. [`queue_line`] carries the same limit, for the same reason.
const MAX_PENDING: usize = MAX_INFLIGHT * 4;

/// Response lines queued for the writer task before a sender waits.
///
/// Bounded so that a client which stops reading stops this process accepting
/// work. See the channel's construction in [`run_io`] for what an unbounded
/// one measured.
const WRITER_QUEUE_LIMIT: usize = MAX_INFLIGHT * 4;

/// How long in-flight requests get to finish after stdin closes, then how long
/// the abort that follows gets to land, then how long the writer gets to drain
/// what is queued.
///
/// A handler parked on a lock no living client can release must not keep the
/// process, and the SQLite file it holds, alive after its client is gone.
/// Whatever has not finished by the first bound is aborted; whatever has not
/// stopped by the second is abandoned and [`run_io`] returns anyway. A task
/// that is cancelled drops its [`ResponseGuard`] and its id is answered. A task
/// with no await point to be cancelled at, which is what a handler inside a
/// blocking `std::sync::Mutex<Connection>` is, keeps its guard and its id is
/// not answered.
const EOF_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Emitted when a response cannot be serialized and the id is unrecoverable.
const FALLBACK_INTERNAL_ERROR: &str = "{\"jsonrpc\":\"2.0\",\"id\":null,\"error\":{\"code\":-32603,\"message\":\"Internal error\"}}\n";

/// One response, serialized into the single line this transport writes.
fn response_line(response: JsonRpcResponse) -> String {
    match serde_json::to_string(&response) {
        Ok(json) => {
            debug!("Sending: {} bytes", json.len());
            format!("{json}\n")
        }
        Err(e) => {
            error!("Failed to serialize response: {}", e);
            FALLBACK_INTERNAL_ERROR.to_string()
        }
    }
}

/// Queue one response line for the single writer task, from inside a request
/// task.
///
/// Waits when the queue is full. That wait is the backpressure, and it ends
/// when the client reads or when the writer dies (the writer owns the
/// receiver, so its death closes the channel and this returns at once). A task
/// still waiting here when stdin closes is aborted by the EOF bound in
/// [`run_io_with`]. The read loop queues its own lines through [`queue_line`],
/// which watches stdin while it waits.
async fn send_response(tx: &mpsc::Sender<String>, response: JsonRpcResponse) {
    let _ = tx.send(response_line(response)).await;
}

/// How an attempt by the read loop to queue a line ended.
enum InlineQueue {
    /// The line is queued for the writer.
    Queued,
    /// The writer is gone, so the channel is closed and the line is dropped.
    /// The read loop's own writer arm carries the outcome out.
    Closed,
    /// stdin is at EOF and the queue is still full, so the loop should end.
    Eof,
}

/// Queue one line the READ LOOP itself produces, while still watching stdin.
///
/// The loop queues three kinds of line without spawning a task: a parse error,
/// the handshake response, and a notification. Each of those awaits the
/// bounded writer channel, and while it does, the loop's own `select!` is not
/// running, so nothing else it watches is observed. A dead writer needs no arm
/// here, because the writer task owns the receiver and its death closes the
/// channel. Stdin EOF does need one, and it is observable on exactly the terms
/// the wait at [`MAX_PENDING`] has: only when the client has left nothing
/// unread, because EOF sits behind whatever it has already written.
async fn queue_line<R>(
    tx: &mpsc::Sender<String>,
    line: String,
    lines: &mut tokio::io::Lines<R>,
) -> InlineQueue
where
    R: AsyncBufRead + Unpin,
{
    let mut pending = Some(line);
    let mut input_pending = false;
    loop {
        tokio::select! {
            reserved = tx.reserve() => {
                return match reserved {
                    Ok(permit) => {
                        permit.send(pending.take().expect("the line is queued once"));
                        InlineQueue::Queued
                    }
                    Err(_) => InlineQueue::Closed,
                };
            }
            // `fill_buf` reads without consuming and is cancel safe (tokio
            // docs), so polling it here cannot take a line the read arm would
            // otherwise get. An empty buffer is EOF; a non-empty one means EOF
            // is not reachable yet, and the branch is disabled rather than
            // left ready forever.
            filled = lines.get_mut().fill_buf(), if !input_pending => {
                match filled {
                    Ok([]) => return InlineQueue::Eof,
                    Ok(_) | Err(_) => input_pending = true,
                }
            }
        }
    }
}

/// Emits a `-32603` for `id` if the task holding it is dropped before the
/// handler returns, so a client is never left waiting forever on a request that
/// will never produce a response.
///
/// SCOPE — this is not panic isolation in a release binary. The workspace sets
/// `[profile.release] panic = "abort"`, so a panicking handler aborts the
/// process without unwinding and no destructor runs at all. What the guard does
/// cover is the drop/cancel path: a task dropped while the `JoinSet` is torn
/// down, any future cancellation arm, and a panic under the unwinding dev/test
/// profile, which is how the unit test below exercises it.
struct ResponseGuard {
    id: Option<Value>,
    tx: mpsc::Sender<String>,
    armed: bool,
}

impl ResponseGuard {
    /// A guard for `id`.
    ///
    /// A guard built for a JSON-RPC notification is inert. A notification
    /// carries no id and MUST NOT be answered, and `JsonRpcResponse` skips a
    /// `None` id when it serializes, so an armed guard would emit
    /// `{"jsonrpc":"2.0","error":{..}}` with no id at all, which is not a
    /// response object.
    fn new(id: Option<Value>, tx: mpsc::Sender<String>) -> Self {
        let armed = id.is_some();
        Self { id, tx, armed }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ResponseGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let response = JsonRpcResponse::error(
            self.id.take(),
            JsonRpcError::internal_error("request handler did not complete"),
        );
        let line = match serde_json::to_string(&response) {
            Ok(json) => format!("{json}\n"),
            Err(_) => FALLBACK_INTERNAL_ERROR.to_string(),
        };
        // `try_send`, because `Drop` cannot await. A full queue means the
        // client has already stopped reading, so this is a line it would not
        // have seen either way.
        let _ = self.tx.try_send(line);
    }
}

/// Handler behaviour driven from the real dispatch arm by a request method, so
/// that the guard's drop paths and the EOF bound are exercised through the
/// production code rather than a copy of it. Compiled out of every build but
/// this crate's own unit tests.
#[cfg(test)]
#[derive(Clone, Copy)]
enum HandlerProbe {
    None,
    /// Panics inside the spawned task.
    Panic,
    /// Parks on a blocking `std::sync::Mutex`, which is the lock kind
    /// `vestige-core`'s `SqliteMemoryStore` reader and writer connections use.
    /// A task parked there has no await point and cannot be aborted.
    BlockingPark,
}

#[cfg(test)]
impl HandlerProbe {
    fn for_method(method: &str) -> Self {
        match method {
            "test/panic" => Self::Panic,
            "test/block" => Self::BlockingPark,
            _ => Self::None,
        }
    }

    fn run(self) {
        match self {
            Self::None => {}
            Self::Panic => panic!("test/panic: handler panicked on purpose"),
            Self::BlockingPark => {
                let _parked = tests::BLOCKING_PARK
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }
}

/// Handle that background work (a first-run model download, a reranker load)
/// uses to push server-initiated `notifications/message` lines onto stdout
/// between responses. Clients that render MCP logging show them; others
/// ignore them. Sending never blocks and never fails the sender.
#[derive(Clone)]
pub struct Notifier {
    tx: mpsc::UnboundedSender<Value>,
}

impl Notifier {
    /// Queue one logging notification. `level` is an MCP log level
    /// (`info`, `warning`, ...), `logger` names the subsystem.
    pub fn log(&self, level: &str, logger: &str, data: Value) {
        let _ = self.tx.send(Self::message(level, logger, data));
    }

    /// The wire shape of a logging notification, kept pure for tests.
    pub fn message(level: &str, logger: &str, data: Value) -> Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": { "level": level, "logger": logger, "data": data }
        })
    }
}

/// Resolve the next queued notification, or wait forever when the transport
/// has no notification channel (or the channel closed and was dropped).
async fn next_notification(rx: &mut Option<mpsc::UnboundedReceiver<Value>>) -> Option<Value> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// stdio Transport for MCP server
pub struct StdioTransport {
    notifications: Option<mpsc::UnboundedReceiver<Value>>,
}

impl StdioTransport {
    pub fn new() -> Self {
        Self {
            notifications: None,
        }
    }

    /// A transport plus the [`Notifier`] that feeds it.
    pub fn with_notifications() -> (Self, Notifier) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                notifications: Some(rx),
            },
            Notifier { tx },
        )
    }

    /// Run the MCP server over stdio with error resilience.
    pub async fn run(self, server: McpServer) -> Result<(), io::Error> {
        run_io(
            server,
            self.notifications,
            BufReader::new(tokio::io::stdin()),
            tokio::io::stdout(),
        )
        .await
    }
}

/// The transport loop, over any reader/writer pair so it is testable.
///
/// Each request is dispatched as its own task, so one slow handler no longer
/// stalls every other request on the same connection. Previously
/// `handle_request(..).await` sat inline in the read loop, which meant the next
/// line was not even read until the current request finished: a client's `ping`
/// queued behind a long embedding call waited for that call to return, and N
/// concurrent requests completed in an N x service-time staircase.
///
/// Responses are returned out of order. That is well formed: JSON-RPC
/// correlates a response to its request by `id`, and the MCP stdio transport
/// imposes no ordering beyond one JSON document per line.
///
/// `Lines::next_line` is cancel safe (tokio docs); `read_line` is not, and a
/// cancelled `read_line` loses the bytes it had already pulled off stdin.
///
/// Backpressure is kept: the writer channel is bounded, the number of requests
/// read but not finished is capped at MAX_PENDING, and a writer that can no
/// longer write ends the loop and is reported to the caller.
///
/// TERMINATION, stated as what is guaranteed. Once stdin EOF or the writer's
/// death is observed, this function returns after at most three
/// `EOF_DRAIN_TIMEOUT` bounds, whatever the handlers are doing: the drain is
/// bounded, the abort that follows it is bounded, the final wait for the
/// writer is bounded, and what is still running after that is detached rather
/// than waited for.
///
/// Two things are NOT guaranteed.
///
/// First, a handler that cannot be cancelled keeps running on the runtime after
/// this returns, so the caller has to stop the runtime without joining it (see
/// `shutdown_runtime` in `main.rs`) for the process to exit.
///
/// Second, the loop can block on the bounded writer channel, and while it does
/// it observes stdin EOF only when the client has left nothing unread. That
/// covers EVERY place the loop waits on that channel: the wait at
/// [`MAX_PENDING`], and the three lines the loop queues itself through
/// [`queue_line`] rather than from a task, which are a parse error, the
/// handshake response and a notification. In each of them EOF is only reachable
/// once the input already written has been consumed, and each is left when the
/// client reads, when the writer dies, or on that EOF. A request TASK blocked
/// on the same channel does not hold the loop at all, and is bounded by the EOF
/// drain below.
pub(crate) async fn run_io<R, W>(
    server: McpServer,
    notifications: Option<mpsc::UnboundedReceiver<Value>>,
    reader: R,
    writer: W,
) -> Result<(), io::Error>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    run_io_with(server, notifications, reader, writer, EOF_DRAIN_TIMEOUT).await
}

/// [`run_io`] with the post-EOF drain bound as a parameter, so a test can pick
/// a short one.
pub(crate) async fn run_io_with<R, W>(
    server: McpServer,
    mut notifications: Option<mpsc::UnboundedReceiver<Value>>,
    reader: R,
    writer: W,
    drain_timeout: Duration,
) -> Result<(), io::Error>
where
    R: AsyncBufRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let server = Arc::new(server);

    // ONE writer task owns the sink; nothing else may write to it. Two tasks
    // interleaving `write_all` could splice two JSON documents into one line,
    // which the stdio transport forbids ("Messages are delimited by newlines,
    // and MUST NOT contain embedded newlines").
    //
    // BOUNDED, so a client that stops reading stops this process accepting
    // work. The serial loop got that for free by writing each response inline:
    // a blocked stdout blocked the reader. Spawning removed that, and an
    // unbounded channel here removed what was left of it. Measured over a
    // duplex pair with the sink blocked at 64 bytes and an unbounded channel,
    // run_io accepted 20,000 requests off stdin and held every response in
    // memory. A handler that already holds a permit can now wait on this
    // channel. Bounded in exchange: MAX_INFLIGHT handlers, WRITER_QUEUE_LIMIT
    // queued lines, and MAX_PENDING requests read but not finished.
    let (tx, mut rx) = mpsc::channel::<String>(WRITER_QUEUE_LIMIT);
    let mut writer_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(line) = rx.recv().await {
            writer.write_all(line.as_bytes()).await?;
            // Flush per line: a client is blocked on this exact response.
            writer.flush().await?;
        }
        Ok::<(), io::Error>(())
    });
    // Set when the writer stops before stdin does, so the loop below neither
    // polls a finished `JoinHandle` again nor awaits it twice.
    let mut writer_outcome: Option<Result<(), io::Error>> = None;

    let mut lines = reader.lines();
    let permits = Arc::new(Semaphore::new(MAX_INFLIGHT));
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut consecutive_errors: u32 = 0;

    'read: loop {
        tokio::select! {
            // A sink that can no longer be written to ends the loop. Without
            // this arm the reader keeps accepting requests and the server
            // keeps serving them into a void, and `run_io` returns `Ok(())`
            // while its client sees nothing.
            joined = &mut writer_task => {
                writer_outcome = Some(match joined {
                    Ok(result) => result,
                    Err(e) => Err(io::Error::other(format!("stdout writer task failed: {e}"))),
                });
                error!("stdout writer stopped; ending the read loop");
                break;
            }
            result = lines.next_line() => {
                match result {
                    Ok(None) => {
                        // Clean EOF — stdin closed
                        info!("stdin closed (EOF), shutting down");
                        break;
                    }
                    Ok(Some(raw)) => {
                        consecutive_errors = 0;
                        let line = raw.trim();

                        if line.is_empty() {
                            continue;
                        }

                        debug!("Received: {} bytes", line.len());

                        // Parse JSON-RPC request
                        let request: JsonRpcRequest = match serde_json::from_str(line) {
                            Ok(r) => r,
                            Err(e) => {
                                warn!("Failed to parse request: {}", e);
                                let queued = queue_line(
                                    &tx,
                                    response_line(JsonRpcResponse::error(
                                        None,
                                        JsonRpcError::parse_error(),
                                    )),
                                    &mut lines,
                                )
                                .await;
                                if let InlineQueue::Eof = queued {
                                    info!("stdin closed (EOF) while queueing a parse error");
                                    break 'read;
                                }
                                continue;
                            }
                        };

                        // ORDERING: the handshake is handled INLINE, before
                        // anything is spawned. `McpServer::handle_request`
                        // rejects every other method with
                        // `server_not_initialized` until `initialize`
                        // completes -- `server/discover` is the one exemption,
                        // documented at that check -- so a spawned
                        // `tools/call` could otherwise overtake the handshake
                        // and be refused. It is also what flips
                        // `is_initialized`, which gates the notification arm
                        // below. Everything after it is order-independent.
                        if request.method == "initialize"
                            || request.method == "notifications/initialized"
                        {
                            if let Some(response) = server.handle_request(request).await
                                && let InlineQueue::Eof =
                                    queue_line(&tx, response_line(response), &mut lines).await
                            {
                                info!("stdin closed (EOF) while queueing the handshake response");
                                break 'read;
                            }
                            continue;
                        }

                        // Reap what has finished, without blocking.
                        while let Some(joined) = tasks.try_join_next() {
                            if let Err(e) = joined {
                                error!("request task failed: {}", e);
                            }
                        }

                        let srv = Arc::clone(&server);
                        let out = tx.clone();
                        let permits = Arc::clone(&permits);
                        #[cfg(test)]
                        let probe = HandlerProbe::for_method(&request.method);
                        // The guard is built BEFORE the cap wait below, so a
                        // request this loop has parsed and then abandons --
                        // because the wait ended in EOF or a dead writer -- is
                        // still answered as the guard drops.
                        let mut guard = ResponseGuard::new(request.id.clone(), out.clone());

                        // PENDING CAP. Reading further while MAX_PENDING
                        // requests are already outstanding would let a client
                        // queue unbounded work inside this process instead of
                        // leaving it in the pipe. This wait is the
                        // backpressure. The defect this change removes was a
                        // wait on every single request, at every depth.
                        //
                        // The wait polls stdin and the writer too. A stalled
                        // sink can hold the tasks here indefinitely, and with
                        // only `join_next` to wait on, a client that closed
                        // stdin or a writer that died went unnoticed for as
                        // long as that lasted. Measured before this arm
                        // existed: with the sink blocked and stdin closed,
                        // `run_io` had not returned 15 s later.
                        let mut input_pending = false;
                        while tasks.len() >= MAX_PENDING {
                            tokio::select! {
                                joined = tasks.join_next() => {
                                    match joined {
                                        Some(Err(e)) => error!("request task failed: {}", e),
                                        Some(Ok(())) => {}
                                        None => break,
                                    }
                                }
                                joined = &mut writer_task => {
                                    writer_outcome = Some(match joined {
                                        Ok(result) => result,
                                        Err(e) => Err(io::Error::other(
                                            format!("stdout writer task failed: {e}")
                                        )),
                                    });
                                    error!(
                                        "stdout writer stopped while {} requests were outstanding",
                                        tasks.len()
                                    );
                                    break 'read;
                                }
                                // `fill_buf` reads without consuming, and is
                                // cancel safe (tokio docs), so polling it here
                                // cannot take a line the read arm would
                                // otherwise get. An empty buffer is EOF. A
                                // non-empty one means EOF is not reachable yet,
                                // and the branch is disabled rather than left
                                // to return ready forever.
                                filled = lines.get_mut().fill_buf(), if !input_pending => {
                                    match filled {
                                        Ok([]) => {
                                            info!(
                                                "stdin closed (EOF) with {} requests outstanding",
                                                tasks.len()
                                            );
                                            break 'read;
                                        }
                                        // The read arm handles both of these
                                        // when the cap clears and it runs again.
                                        Ok(_) | Err(_) => input_pending = true,
                                    }
                                }
                            }
                        }

                        tasks.spawn(async move {
                            // The permit is acquired INSIDE the task, never in
                            // the read loop. Waiting for it out there would
                            // stop draining stdin and reintroduce exactly the
                            // head-of-line blocking this change removes, and it
                            // would stop a cheap call from overtaking queued
                            // slow ones.
                            let _permit = match permits.acquire_owned().await {
                                Ok(permit) => permit,
                                // Only on a closed semaphore, i.e. shutdown.
                                // The guard answers the id as it drops.
                                Err(_) => return,
                            };
                            // Drives the guard's drop paths through the real
                            // dispatch arm. Compiled out of every build but
                            // this crate's own unit tests.
                            #[cfg(test)]
                            probe.run();
                            let response = srv.handle_request(request).await;
                            if let Some(response) = response {
                                send_response(&out, response).await;
                            }
                            // Disarmed only once the response is QUEUED. The
                            // wait inside `send_response` is a cancellation
                            // point, and it is exactly where the EOF abort
                            // finds a task whose client stopped reading.
                            // Disarming before it would leave that id with
                            // neither its response nor a `-32603`.
                            guard.disarm();
                        });
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        warn!(
                            "I/O error reading stdin ({}/{}): {}",
                            consecutive_errors, MAX_CONSECUTIVE_ERRORS, e
                        );
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                            error!(
                                "Too many consecutive I/O errors ({}), shutting down",
                                consecutive_errors
                            );
                            break;
                        }
                        // Brief pause before retrying
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
            }
            // Held until the handshake completes: a logging line before the
            // initialize response would desync a client that reads the next
            // line as its answer. The channel buffers meanwhile.
            notification = next_notification(&mut notifications), if server.is_initialized() => {
                match notification {
                    Some(notification) if !server.logging_allows(&notification) => {},
                    Some(notification) => match serde_json::to_string(&notification) {
                        Ok(json) => {
                            if let InlineQueue::Eof =
                                queue_line(&tx, format!("{json}\n"), &mut lines).await
                            {
                                info!("stdin closed (EOF) while queueing a notification");
                                break 'read;
                            }
                        }
                        Err(e) => warn!("Failed to serialize notification: {}", e),
                    },
                    // Every sender is gone: park the branch instead of
                    // spinning on a closed channel.
                    None => notifications = None,
                }
            }
        }
    }

    // EOF drain, BOUNDED: finish the in-flight requests if they finish
    // quickly, then close the channel and wait for the writer, so the last
    // response reaches the client before exit. An unbounded drain here waits
    // on a handler parked in a lock, which outlives the client that asked for
    // it: this server's own consolidation pass has been measured holding the
    // storage writer mutex for minutes.
    let drain = async {
        while let Some(joined) = tasks.join_next().await {
            if let Err(e) = joined {
                error!("request task failed: {}", e);
            }
        }
    };
    if tokio::time::timeout(drain_timeout, drain).await.is_err() {
        warn!(
            "in-flight requests did not finish within {:?} of stdin EOF; aborting them",
            drain_timeout
        );
        // A cancelled task drops its ResponseGuard, which queues a -32603 for
        // its id while the channel is still open. Two cases are not answered:
        // a task that cannot be cancelled keeps its guard, and a guard whose
        // queue is full discards the line, because `Drop` cannot wait.
        //
        // THE ABORT IS BOUNDED TOO. `JoinSet::shutdown` aborts every task and
        // then waits for each to stop, and an abort only lands at an await
        // point. A handler inside a blocking `std::sync::Mutex<Connection>`
        // has none, so that wait lasts as long as the lock is held. Measured
        // with a handler parked on a `std::sync::Mutex`: an unbounded
        // `shutdown().await` had not returned 10 s after EOF, and returned
        // only when the lock was released.
        if tokio::time::timeout(drain_timeout, tasks.shutdown())
            .await
            .is_err()
        {
            warn!(
                "{} request task(s) could not be cancelled; abandoning them",
                tasks.len()
            );
            // They are already aborted and will stop at their next await
            // point, if they reach one. Detaching means this function neither
            // waits for them nor aborts them again as the set drops.
            tasks.detach_all();
        }
    }
    drop(tx);

    match writer_outcome {
        Some(outcome) => outcome,
        // BOUNDED for the same reason the abort above is. An abandoned task
        // still holds a sender, so the channel never closes, so the writer
        // never reaches the end of its loop. Everything queued before this
        // point has been written by the time the bound expires; what is left
        // is a channel that will not close.
        None => match tokio::time::timeout(drain_timeout, &mut writer_task).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(e)) => Err(io::Error::other(format!("stdout writer task failed: {e}"))),
            Err(_) => {
                warn!(
                    "stdout writer did not finish within {:?} of the last response; \
                     ending anyway",
                    drain_timeout
                );
                writer_task.abort();
                Ok(())
            }
        },
    }
}

impl Default for StdioTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::CognitiveEngine;
    use serde_json::json;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::io::AsyncReadExt;
    use tokio::sync::Mutex;
    use vestige_core::Storage;

    /// The lock the `test/block` probe parks on. One test takes it, and that
    /// test releases it before it returns, so this test binary's runtimes are
    /// never dropped while a worker is parked here.
    pub(super) static BLOCKING_PARK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn test_parts() -> (Arc<Storage>, Arc<Mutex<CognitiveEngine>>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = Arc::new(Storage::new(Some(dir.path().join("test.db"))).unwrap());
        let cognitive = Arc::new(Mutex::new(CognitiveEngine::new()));
        (storage, cognitive, dir)
    }

    fn test_server() -> (McpServer, TempDir) {
        let (storage, cognitive, dir) = test_parts();
        (McpServer::new(storage, cognitive), dir)
    }

    fn init_line() -> String {
        json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"}
            }
        })
        .to_string()
            + "\n"
    }

    /// Feed `input` to `run_io` and return the lines it wrote, in write order.
    async fn drive(input: String) -> Vec<Value> {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, mut client_r) = tokio::io::duplex(1 << 20);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );
        client_w.write_all(input.as_bytes()).await.unwrap();
        // Dropping the client's write half is the EOF that ends the loop; the
        // loop must still drain everything already in flight.
        drop(client_w);
        let mut buf = String::new();
        client_r.read_to_string(&mut buf).await.unwrap();
        handle.await.unwrap().unwrap();
        buf.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Value>(l).expect("every stdout line is one JSON doc"))
            .collect()
    }

    /// Read one newline-delimited JSON document from a reader.
    async fn read_one<R: AsyncBufRead + Unpin>(reader: &mut R) -> Value {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        assert!(n > 0, "the server closed the stream instead of answering");
        serde_json::from_str(line.trim()).expect("one JSON doc per line")
    }

    /// THE HEAD-OF-LINE TEST. A request parked inside its handler must not stop
    /// the next request from being read and answered.
    ///
    /// The slow request is made slow deterministically rather than by timing:
    /// `graph { action: "predict" }` takes the cognitive engine's mutex as its
    /// first act, and this test holds that mutex, so request 1's handler parks
    /// there until the test releases it. A `ping` needs neither the engine nor
    /// storage.
    ///
    /// With the old inline `handle_request(..).await` in the read loop, the
    /// loop itself parks on request 1 and never reads request 2, so no line
    /// arrives and the timeout below fires.
    #[tokio::test]
    async fn a_parked_request_does_not_block_the_next_one() {
        let (storage, cognitive, _dir) = test_parts();
        let server = McpServer::new(Arc::clone(&storage), Arc::clone(&cognitive));
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, client_r) = tokio::io::duplex(1 << 20);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );

        // Complete the handshake first, and observe its response, so the
        // measurement below cannot be confused with handshake ordering.
        let mut reader = BufReader::new(client_r);
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let init = read_one(&mut reader).await;
        assert_eq!(init["id"], json!(0), "initialize answered: {init}");

        // Park the engine the slow handler needs.
        let held = cognitive.lock().await;

        let slow = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "graph", "arguments": {"action": "predict"}}
        })
        .to_string()
            + "\n";
        let fast = json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}).to_string() + "\n";
        client_w.write_all(slow.as_bytes()).await.unwrap();
        client_w.write_all(fast.as_bytes()).await.unwrap();

        let answered = tokio::time::timeout(Duration::from_secs(10), read_one(&mut reader))
            .await
            .expect("the ping is answered while request 1 is parked in its handler");
        assert_eq!(
            answered["id"],
            json!(2),
            "the ping overtook the parked request: {answered}"
        );
        assert!(answered.get("result").is_some(), "ping: {answered}");

        // Release the engine; the parked request must still be answered.
        drop(held);
        let slow_response = tokio::time::timeout(Duration::from_secs(60), read_one(&mut reader))
            .await
            .expect("the parked request completes once the engine is free");
        assert_eq!(slow_response["id"], json!(1), "{slow_response}");

        drop(client_w);
        drop(reader);
        let _ = tokio::time::timeout(Duration::from_secs(60), handle).await;
    }

    /// Responses may arrive in any order; every request id must appear exactly
    /// once, and correlation is by id rather than by position.
    #[tokio::test]
    async fn out_of_order_responses_correlate_by_id() {
        let mut input = init_line();
        for id in 1..=12 {
            input += &(json!({"jsonrpc": "2.0", "id": id, "method": "ping"}).to_string() + "\n");
        }
        let out = drive(input).await;
        let mut ids: Vec<i64> = out
            .iter()
            .filter_map(|v| v.get("id").and_then(Value::as_i64))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..=12).collect::<Vec<i64>>());
        for v in &out {
            let id = v["id"].as_i64().unwrap();
            assert!(v.get("result").is_some(), "id {id} must have a result: {v}");
        }
    }

    /// A `tools/call` written immediately after `initialize`, with no wait in
    /// between, must not be refused with `server_not_initialized`. This is the
    /// ordering the inline handshake arm exists to preserve.
    #[tokio::test]
    async fn initialize_is_ordered_before_a_following_tools_call() {
        let input = init_line()
            + &(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string()
                + "\n")
            + &(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).to_string() + "\n")
            + &(json!({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "memory_status", "arguments": {"view": "health"}}
            })
            .to_string()
                + "\n");
        let out = drive(input).await;
        for v in &out {
            if let Some(err) = v.get("error") {
                assert_ne!(
                    err["code"].as_i64(),
                    Some(-32003),
                    "no response may be server_not_initialized: {v}"
                );
            }
        }
        let listed = out
            .iter()
            .find(|v| v["id"] == json!(1))
            .expect("tools/list");
        assert!(listed.get("result").is_some(), "tools/list: {listed}");
    }

    /// Every line on stdout is exactly one complete JSON document even when
    /// many handlers finish at once — the single-writer invariant.
    #[tokio::test]
    async fn responses_never_interleave_under_concurrent_load() {
        let mut input = init_line();
        for id in 1..=40 {
            input +=
                &(json!({"jsonrpc": "2.0", "id": id, "method": "tools/list"}).to_string() + "\n");
        }
        let out = drive(input).await; // drive() parses every line or panics
        assert_eq!(out.len(), 41, "one response per request");
        for v in &out {
            assert_eq!(v["jsonrpc"], json!("2.0"));
        }
    }

    /// EOF must not truncate work already accepted: every accepted request gets
    /// its response before `run_io` returns.
    #[tokio::test]
    async fn eof_drains_in_flight_responses() {
        let mut input = init_line();
        for id in 1..=25 {
            input += &(json!({
                "jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "memory_status", "arguments": {"view": "health"}}
            })
            .to_string()
                + "\n");
        }
        let out = drive(input).await;
        let mut ids: Vec<i64> = out
            .iter()
            .filter_map(|v| v.get("id").and_then(Value::as_i64))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..=25).collect::<Vec<i64>>());
    }

    /// A malformed line is answered and the loop keeps serving after it.
    #[tokio::test]
    async fn a_bad_line_does_not_stop_the_loop() {
        let input = init_line()
            + "{ this is not json\n"
            + &(json!({"jsonrpc": "2.0", "id": 7, "method": "ping"}).to_string() + "\n");
        let out = drive(input).await;
        assert!(
            out.iter()
                .any(|v| v["error"]["code"] == json!(-32700) && v["id"].is_null()),
            "a parse error is reported: {out:?}"
        );
        assert!(
            out.iter()
                .any(|v| v["id"] == json!(7) && v.get("result").is_some()),
            "the request after the bad line is still served: {out:?}"
        );
    }

    /// An armed guard that is dropped without disarming emits a -32603 carrying
    /// the request id, so a dropped or cancelled task cannot hang a client.
    ///
    /// This is the drop path, not the panic path: `[profile.release]` sets
    /// `panic = "abort"`, so in a release binary a panicking handler kills the
    /// process and no destructor runs. See ResponseGuard's doc comment.
    #[tokio::test]
    async fn dropping_an_armed_guard_emits_an_internal_error_for_that_id() {
        let (tx, mut rx) = mpsc::channel::<String>(WRITER_QUEUE_LIMIT);
        {
            let _guard = ResponseGuard::new(Some(json!(42)), tx.clone());
        }
        let line = rx.try_recv().expect("the guard emitted a response");
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["id"], json!(42));
        assert_eq!(v["error"]["code"], json!(-32603));

        // A disarmed guard emits nothing.
        {
            let mut guard = ResponseGuard::new(Some(json!(43)), tx.clone());
            guard.disarm();
        }
        assert!(rx.try_recv().is_err(), "a disarmed guard stays silent");
    }

    /// A guard built for a JSON-RPC notification must stay silent. A
    /// notification carries no id, `JsonRpcResponse` omits a `None` id when it
    /// serializes, and an error object with no id is not a response object.
    #[tokio::test]
    async fn a_guard_for_an_id_less_request_emits_nothing() {
        let (tx, mut rx) = mpsc::channel::<String>(WRITER_QUEUE_LIMIT);
        {
            let _guard = ResponseGuard::new(None, tx.clone());
        }
        assert!(
            rx.try_recv().is_err(),
            "a notification must not be answered, not even with an error"
        );
    }

    /// BACKPRESSURE. A client that stops reading must stop this process
    /// accepting requests off stdin.
    ///
    /// The serial loop had this by construction: it wrote each response inline,
    /// so a blocked sink blocked the reader. With an unbounded writer channel
    /// and no cap on requests read but not finished, `run_io` took everything
    /// offered and held it in memory: measured at 20,000 requests and 888,894
    /// bytes against a sink blocked at 64 bytes.
    #[tokio::test]
    async fn a_client_that_stops_reading_stops_stdin_being_drained() {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1024);
        // The client never reads this side, so the writer blocks on the first
        // response that does not fit.
        let (server_w, client_r) = tokio::io::duplex(64);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );

        client_w.write_all(init_line().as_bytes()).await.unwrap();

        let mut load = String::new();
        for id in 1..=5_000 {
            load += &(json!({"jsonrpc": "2.0", "id": id, "method": "ping"}).to_string() + "\n");
        }
        let offered = load.len();
        let accepted =
            tokio::time::timeout(Duration::from_secs(5), client_w.write_all(load.as_bytes())).await;
        assert!(
            accepted.is_err(),
            "the server took all {offered} bytes while its output was blocked, \
             so nothing bounds what a client can queue inside the process"
        );

        drop(client_r);
        drop(client_w);
        let _ = tokio::time::timeout(Duration::from_secs(30), handle).await;
    }

    /// stdin EOF must end the loop even when a handler is parked.
    ///
    /// An unbounded drain waits for every in-flight handler, and a handler can
    /// be parked on a lock no living client will release, so the process
    /// outlives its client while still holding the database file.
    #[tokio::test]
    async fn stdin_eof_ends_the_loop_even_with_a_handler_parked() {
        let (storage, cognitive, _dir) = test_parts();
        let server = McpServer::new(Arc::clone(&storage), Arc::clone(&cognitive));
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            run_io_with(
                server,
                None,
                BufReader::new(server_r),
                server_w,
                Duration::from_millis(250),
            )
            .await
        });

        let mut reader = BufReader::new(client_r);
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let init = read_one(&mut reader).await;
        assert_eq!(init["id"], json!(0), "initialize answered: {init}");

        // Park the handler on a mutex nothing will release.
        let held = cognitive.lock().await;
        let slow = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "graph", "arguments": {"action": "predict"}}
        })
        .to_string()
            + "\n";
        client_w.write_all(slow.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;

        // The client is gone.
        drop(client_w);

        let returned = tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run_io returns after stdin EOF instead of waiting for a parked handler");
        returned.unwrap().unwrap();

        // The aborted handler still answered its id before the channel closed.
        let answered = tokio::time::timeout(Duration::from_secs(5), read_one(&mut reader))
            .await
            .expect("the abandoned request is answered");
        assert_eq!(answered["id"], json!(1), "{answered}");
        assert_eq!(answered["error"]["code"], json!(-32603), "{answered}");

        drop(held);
    }

    /// stdin EOF must end the loop even when the parked handler cannot be
    /// aborted.
    ///
    /// `JoinSet::shutdown` aborts and then waits, and an abort lands only at an
    /// await point. `vestige-core`'s storage takes `std::sync::Mutex<Connection>`
    /// for both reads and writes, and a handler inside one has no await point,
    /// so an unbounded `shutdown().await` waits for the lock instead. Measured
    /// with the probe below and no second bound: `run_io_with` had not returned
    /// 10 s after EOF, and returned only once the lock was released. The test
    /// asserts termination, not that the abandoned id is answered: a task that
    /// is never cancelled keeps its guard.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stdin_eof_ends_the_loop_when_a_handler_is_parked_in_a_blocking_lock() {
        // The lock is held on a plain thread rather than in this async body,
        // so nothing holds a blocking guard across an await. The holder
        // releases on its own after 20 s, which bounds the run when the
        // assertion below fails.
        let (parked_tx, parked_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            let _held = BLOCKING_PARK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            parked_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(20));
        });
        parked_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the holder thread took the park");

        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            run_io_with(
                server,
                None,
                BufReader::new(server_r),
                server_w,
                Duration::from_millis(250),
            )
            .await
        });

        let mut reader = BufReader::new(client_r);
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let init = read_one(&mut reader).await;
        assert_eq!(init["id"], json!(0), "initialize answered: {init}");

        let blocked = json!({"jsonrpc": "2.0", "id": 1, "method": "test/block"}).to_string() + "\n";
        client_w.write_all(blocked.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(250)).await;

        // The client is gone, and the handler is inside a lock it cannot be
        // aborted out of.
        drop(client_w);
        let returned = tokio::time::timeout(Duration::from_secs(10), handle).await;

        // Release before asserting, so a failure does not leave a parked
        // worker for this test's runtime to join.
        let _ = release_tx.send(());
        holder.join().unwrap();

        returned
            .expect(
                "run_io returns after stdin EOF even though the parked handler is not abortable",
            )
            .unwrap()
            .unwrap();
    }

    /// stdin EOF must be observed while the read loop waits at the pending cap.
    ///
    /// MAX_PENDING handlers park on the cognitive mutex this test holds, so
    /// nothing finishes and the wait at the cap is where the loop sits. With
    /// only `join_next` to wait on, EOF arrived and was never seen.
    #[tokio::test]
    async fn stdin_eof_is_observed_while_the_read_loop_waits_at_the_pending_cap() {
        let (storage, cognitive, _dir) = test_parts();
        let server = McpServer::new(Arc::clone(&storage), Arc::clone(&cognitive));
        let (mut client_w, server_r) = tokio::io::duplex(1 << 18);
        let (server_w, client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            run_io_with(
                server,
                None,
                BufReader::new(server_r),
                server_w,
                Duration::from_millis(250),
            )
            .await
        });

        let mut reader = BufReader::new(client_r);
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let init = read_one(&mut reader).await;
        assert_eq!(init["id"], json!(0), "initialize answered: {init}");

        let held = cognitive.lock().await;
        // MAX_PENDING requests park in their handlers and one more is read and
        // then held at the cap. Every byte offered is consumed, which is what
        // makes EOF reachable at all; see MAX_PENDING for the case where it is
        // not.
        let last_id = MAX_PENDING + 1;
        let mut load = String::new();
        for id in 1..=last_id {
            load += &(json!({
                "jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "graph", "arguments": {"action": "predict"}}
            })
            .to_string()
                + "\n");
        }
        client_w.write_all(load.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        drop(client_w);

        let returned = tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run_io observes stdin EOF while it waits at the pending cap");
        returned.unwrap().unwrap();

        // The request read but never dispatched is answered from the guard the
        // read loop built before it waited.
        let mut rest = String::new();
        reader.read_to_string(&mut rest).await.unwrap();
        let answered: Vec<Value> = rest
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("one JSON doc per line"))
            .collect();
        assert!(
            answered
                .iter()
                .any(|v| v["id"] == json!(last_id) && v["error"]["code"] == json!(-32603)),
            "the request abandoned at the cap is answered: {answered:?}"
        );

        drop(held);
    }

    /// A writer that dies while the read loop waits at the pending cap must
    /// end the loop. The outer select's writer arm is not polled there, and
    /// the handlers cannot finish, so nothing else would notice.
    #[tokio::test]
    async fn a_dead_writer_is_observed_while_the_read_loop_waits_at_the_pending_cap() {
        let (storage, cognitive, _dir) = test_parts();
        let server = McpServer::new(Arc::clone(&storage), Arc::clone(&cognitive));
        let (mut client_w, server_r) = tokio::io::duplex(1 << 18);
        // A 64-byte sink the client never reads: the writer blocks inside
        // `write_all` on the initialize response and stays there.
        let (server_w, client_r) = tokio::io::duplex(64);
        let handle = tokio::spawn(async move {
            run_io_with(
                server,
                None,
                BufReader::new(server_r),
                server_w,
                Duration::from_millis(250),
            )
            .await
        });

        let held = cognitive.lock().await;
        let mut load = init_line();
        for id in 1..=(MAX_PENDING + 1) {
            load += &(json!({
                "jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "graph", "arguments": {"action": "predict"}}
            })
            .to_string()
                + "\n");
        }
        client_w.write_all(load.as_bytes()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        // stdin stays open. Only the sink dies.
        drop(client_r);

        let returned = tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run_io observes a dead writer while it waits at the pending cap");
        let err = returned
            .unwrap()
            .expect_err("a dead sink is reported to the caller as an io error");
        assert_eq!(
            err.kind(),
            io::ErrorKind::BrokenPipe,
            "the underlying write error is carried: {err}"
        );

        drop(held);
        drop(client_w);
    }

    /// KNOWN LIMIT, pinned here so a change to it is deliberate: stdin EOF
    /// behind input this loop has not read is NOT observed at the pending cap.
    /// EOF is only reachable once that input is consumed, and consuming it is
    /// what the cap exists to stop. The loop leaves the cap when the tasks
    /// finish or the writer dies; here the client resumes reading, which is
    /// what lets both happen.
    #[tokio::test]
    async fn eof_behind_unread_input_waits_for_the_pending_cap_to_clear() {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 18);
        let (server_w, mut client_r) = tokio::io::duplex(64);
        let mut handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );

        let mut load = init_line();
        for id in 1..=500 {
            load += &(json!({"jsonrpc": "2.0", "id": id, "method": "ping"}).to_string() + "\n");
        }
        client_w.write_all(load.as_bytes()).await.unwrap();
        // EOF, with hundreds of lines still unread in the pipe.
        drop(client_w);
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert!(
            tokio::time::timeout(Duration::from_secs(1), &mut handle)
                .await
                .is_err(),
            "EOF behind unread input is not observed while the loop waits at the cap"
        );

        let mut out = String::new();
        client_r.read_to_string(&mut out).await.unwrap();
        tokio::time::timeout(Duration::from_secs(30), handle)
            .await
            .expect("the loop reaches EOF once the client reads again")
            .unwrap()
            .unwrap();
        assert!(
            out.lines().filter(|l| !l.trim().is_empty()).count() >= 500,
            "every request offered is answered once the client reads again"
        );
    }

    /// The read loop queues three kinds of line itself rather than from a
    /// task, and each of those awaits the bounded writer channel. While it
    /// does, the loop's `select!` is not running, so stdin EOF is observed
    /// only if the queue attempt watches for it.
    ///
    /// Driven through the parse-error arm, which is the cheapest of the three
    /// to reach. The sink is filled so the writer blocks mid-write, holding
    /// exactly one line off the channel and taking no more; the channel then
    /// has exactly `WRITER_QUEUE_LIMIT` free slots, and one line past that has
    /// nowhere to go and nothing unread behind it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stdin_eof_is_observed_while_the_read_loop_waits_to_queue_a_line() {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 18);
        let (server_w, mut client_r) = tokio::io::duplex(64);
        let handle = tokio::spawn(async move {
            run_io_with(
                server,
                None,
                BufReader::new(server_r),
                server_w,
                Duration::from_millis(250),
            )
            .await
        });

        // Reading one byte is the proof that the writer has taken the
        // handshake response OFF the channel and is now blocked inside its
        // `write_all`, so it will not take another line.
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let mut one = [0u8; 1];
        tokio::time::timeout(Duration::from_secs(30), client_r.read_exact(&mut one))
            .await
            .expect("the writer starts writing the handshake response")
            .unwrap();

        let mut flood = String::new();
        for _ in 0..=WRITER_QUEUE_LIMIT {
            flood.push_str("{not json\n");
        }
        client_w.write_all(flood.as_bytes()).await.unwrap();
        // EOF with every line consumed: the loop is inside its last queue
        // attempt and there is nothing left in the pipe.
        drop(client_w);

        let ended = tokio::time::timeout(Duration::from_secs(20), handle).await;
        assert!(
            ended.is_ok(),
            "run_io observes stdin EOF while it waits to queue a line: {ended:?}"
        );
        ended.unwrap().unwrap().unwrap();
    }

    /// A sink that can no longer be written to must end the loop and be
    /// reported, so `main` exits non-zero rather than serving into a void.
    #[tokio::test]
    async fn a_dead_writer_ends_the_loop_and_is_reported() {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, client_r) = tokio::io::duplex(1 << 16);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );

        // The client's read half is gone, so every write to the sink fails.
        // stdin stays OPEN, so nothing else can end the loop.
        drop(client_r);
        for id in 0..200 {
            let line = if id == 0 {
                init_line()
            } else {
                json!({"jsonrpc": "2.0", "id": id, "method": "ping"}).to_string() + "\n"
            };
            if client_w.write_all(line.as_bytes()).await.is_err() {
                break;
            }
        }

        let returned = tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("run_io returns once its output is gone, without waiting for stdin EOF");
        let err = returned
            .unwrap()
            .expect_err("a dead sink is reported to the caller as an io error");
        assert_eq!(
            err.kind(),
            io::ErrorKind::BrokenPipe,
            "the underlying write error is carried: {err}"
        );
    }

    /// A panicking handler is answered from its guard and the loop keeps
    /// serving. This is the guard's panic path, reachable only under the
    /// unwinding dev/test profile; the release profile sets `panic = "abort"`
    /// and takes the process down instead.
    #[tokio::test]
    async fn a_panicking_handler_is_answered_and_the_loop_keeps_serving() {
        let input = init_line()
            + &(json!({"jsonrpc": "2.0", "id": 99, "method": "test/panic"}).to_string() + "\n")
            + &(json!({"jsonrpc": "2.0", "id": 100, "method": "ping"}).to_string() + "\n");
        let out = drive(input).await;

        let panicked = out
            .iter()
            .find(|v| v["id"] == json!(99))
            .unwrap_or_else(|| panic!("the panicking request is answered: {out:?}"));
        assert_eq!(panicked["error"]["code"], json!(-32603), "{panicked}");
        assert!(
            out.iter()
                .any(|v| v["id"] == json!(100) && v.get("result").is_some()),
            "the request after the panic is still served: {out:?}"
        );
    }

    /// Offered load above the in-flight cap must not stop stdin draining: a
    /// cheap `ping` written after 3x MAX_INFLIGHT heavier calls still gets
    /// served. With the permit acquired in the read loop instead of inside the
    /// task, the reader would block on the cap and this would hang.
    #[tokio::test]
    async fn a_ping_is_served_when_offered_load_exceeds_the_cap() {
        let (server, _dir) = test_server();
        let (mut client_w, server_r) = tokio::io::duplex(1 << 18);
        let (server_w, client_r) = tokio::io::duplex(1 << 22);
        let handle =
            tokio::spawn(
                async move { run_io(server, None, BufReader::new(server_r), server_w).await },
            );
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let mut load = String::new();
        for id in 1..=(MAX_INFLIGHT * 3) {
            load += &(json!({
                "jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "memory_status", "arguments": {"view": "health"}}
            })
            .to_string()
                + "\n");
        }
        client_w.write_all(load.as_bytes()).await.unwrap();
        let sentinel = MAX_INFLIGHT * 3 + 1;
        client_w
            .write_all(
                (json!({"jsonrpc": "2.0", "id": sentinel, "method": "ping"}).to_string() + "\n")
                    .as_bytes(),
            )
            .await
            .unwrap();

        // Read until the sentinel appears. The reader must have consumed it
        // while the heavy calls were still queued behind the cap.
        let mut reader = BufReader::new(client_r);
        let mut seen = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
        while tokio::time::Instant::now() < deadline {
            let mut line = String::new();
            let n = tokio::time::timeout(Duration::from_secs(30), reader.read_line(&mut line))
                .await
                .expect("the server keeps answering")
                .unwrap();
            if n == 0 {
                break;
            }
            let v: Value = serde_json::from_str(line.trim()).unwrap();
            if v["id"] == json!(sentinel) {
                assert!(v.get("result").is_some(), "the ping succeeded: {v}");
                seen = true;
                break;
            }
        }
        assert!(seen, "the ping past the in-flight cap was served");
        drop(client_w);
        drop(reader);
        let _ = tokio::time::timeout(Duration::from_secs(180), handle).await;
    }

    /// A notification queued before the handshake is still held until after the
    /// initialize response, and then reaches the client through the same single
    /// writer as the responses.
    #[tokio::test]
    async fn notifications_are_held_until_after_the_handshake() {
        let (server, _dir) = test_server();
        let (notification_tx, notification_rx) = mpsc::unbounded_channel::<Value>();
        let notifier = Notifier {
            tx: notification_tx.clone(),
        };
        notifier.log("warning", "t", json!({ "n": 1 }));

        let (mut client_w, server_r) = tokio::io::duplex(1 << 16);
        let (server_w, client_r) = tokio::io::duplex(1 << 20);
        let handle = tokio::spawn(async move {
            run_io(
                server,
                Some(notification_rx),
                BufReader::new(server_r),
                server_w,
            )
            .await
        });

        let mut reader = BufReader::new(client_r);
        client_w.write_all(init_line().as_bytes()).await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(10), read_one(&mut reader))
            .await
            .expect("initialize is answered");
        assert_eq!(
            first["id"],
            json!(0),
            "the initialize response precedes any notification: {first}"
        );

        let second = tokio::time::timeout(Duration::from_secs(10), read_one(&mut reader))
            .await
            .expect("the held notification is delivered after the handshake");
        assert_eq!(second["method"], json!("notifications/message"), "{second}");
        assert_eq!(second["params"]["data"]["n"], json!(1), "{second}");

        drop(notification_tx);
        drop(client_w);
        drop(reader);
        let _ = tokio::time::timeout(Duration::from_secs(60), handle).await;
    }

    #[test]
    fn logging_notification_has_the_mcp_wire_shape() {
        let message = Notifier::message(
            "info",
            "vestige.embeddings",
            serde_json::json!({ "event": "model_download_started" }),
        );
        assert_eq!(message["jsonrpc"], "2.0");
        assert_eq!(message["method"], "notifications/message");
        assert_eq!(message["params"]["level"], "info");
        assert_eq!(message["params"]["logger"], "vestige.embeddings");
        assert_eq!(message["params"]["data"]["event"], "model_download_started");
        assert!(message.get("id").is_none(), "notifications carry no id");
    }

    #[tokio::test]
    async fn queued_notifications_are_delivered_in_order() {
        let (mut transport, notifier) = super::StdioTransport::with_notifications();
        notifier.log("info", "t", serde_json::json!({ "n": 1 }));
        notifier.log("info", "t", serde_json::json!({ "n": 2 }));
        let first = super::next_notification(&mut transport.notifications)
            .await
            .unwrap();
        let second = super::next_notification(&mut transport.notifications)
            .await
            .unwrap();
        assert_eq!(first["params"]["data"]["n"], 1);
        assert_eq!(second["params"]["data"]["n"], 2);
    }
}

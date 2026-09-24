//! Vestige MCP Server - local cognitive memory for MCP agents.
//!
//! A bleeding-edge Rust MCP (Model Context Protocol) server that provides
//! Claude and other AI assistants with long-term memory capabilities
//! powered by 130 years of memory research.
//!
//! Core Features:
//! - FSRS-6 spaced repetition algorithm (21 parameters, 30% more efficient than SM-2)
//! - Bjork dual-strength memory model
//! - Local semantic embeddings (768-dim BGE, no external API)
//! - HNSW vector search (20x faster than FAISS)
//! - Hybrid search (BM25 + semantic + RRF fusion)
//!
//! Neuroscience Features:
//! - Synaptic Tagging & Capture (retroactive importance)
//! - Spreading Activation Networks (multi-hop associations)
//! - Hippocampal Indexing (two-phase retrieval)
//! - Memory States (active/dormant/silent/unavailable)
//! - Context-Dependent Memory (encoding specificity)
//! - Multi-Channel Importance Signals
//! - Predictive Retrieval
//! - Prospective Memory (intentions with triggers)
//!
//! Advanced Features:
//! - Memory Dreams (insight generation during consolidation)
//! - Memory Compression
//! - Reconsolidation (memories editable on retrieval)
//! - Memory Chains (reasoning paths)

// Supplies the `__isoc23_*` and `__cxa_call_terminate` symbols that the
// statically linked ONNX Runtime archive imports from glibc >= 2.38 and
// libstdc++ >= GCC 13. Compiled into each binary root rather than into the
// library so the definitions are always part of the final link instead of
// being subject to archive member selection. See the module docs.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[path = "glibc_compat.rs"]
mod glibc_compat;

use vestige_mcp::cognitive;
use vestige_mcp::protocol;
use vestige_mcp::server;

use directories::BaseDirs;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Component, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{Level, debug, error, info, warn};
use tracing_subscriber::EnvFilter;

// Use vestige-core for the cognitive science engine
use vestige_core::Storage;

use protocol::stdio::{Notifier, StdioTransport};
use server::McpServer;

const DATA_DIR_ENV: &str = "VESTIGE_DATA_DIR";
const DATABASE_FILE: &str = "vestige.db";

/// Parsed CLI configuration.
struct Config {
    data_dir: Option<PathBuf>,
    http_port: u16,
    http_enabled: bool,
    dashboard_enabled: bool,
}

fn data_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(DATA_DIR_ENV).and_then(|value| {
        if value.as_os_str().is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

/// Parse command-line arguments into a `Config`.
/// Exits the process if `--help` or `--version` is requested.
fn parse_args() -> Config {
    parse_args_from(std::env::args_os().collect(), data_dir_from_env())
}

fn parse_args_from(args: Vec<OsString>, env_data_dir: Option<PathBuf>) -> Config {
    let mut data_dir = env_data_dir;
    let mut http_port: u16 = std::env::var("VESTIGE_HTTP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3928);
    let mut http_enabled = std::env::var("VESTIGE_HTTP_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);
    let dashboard_enabled = std::env::var("VESTIGE_DASHBOARD_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);
    let mut i = 1;

    while i < args.len() {
        let arg = args[i].to_string_lossy();
        match arg.as_ref() {
            "--help" | "-h" => {
                println!("Vestige MCP Server v{}", env!("CARGO_PKG_VERSION"));
                println!();
                println!("FSRS-6 powered AI memory server using the Model Context Protocol.");
                println!();
                println!("USAGE:");
                println!("    vestige-mcp [OPTIONS]");
                println!();
                println!("OPTIONS:");
                println!("    -h, --help              Print help information");
                println!("    -V, --version           Print version information");
                println!(
                    "    --data-dir <PATH>       Custom data directory (overrides VESTIGE_DATA_DIR)"
                );
                println!("    --http                  Enable Streamable HTTP transport");
                println!("    --no-http               Disable Streamable HTTP transport");
                println!("    --http-port <PORT>      HTTP transport port (also enables HTTP)");
                println!();
                println!("ENVIRONMENT:");
                println!(
                    "    VESTIGE_DATA_DIR          Data directory fallback (stores vestige.db inside)"
                );
                println!(
                    "    RUST_LOG                  Log level filter (e.g., debug, info, warn, error)"
                );
                println!(
                    "    VESTIGE_AUTH_TOKEN       Override the bearer token for HTTP transport"
                );
                println!("    VESTIGE_HTTP_ENABLED     Enable HTTP transport (default: false)");
                println!("    VESTIGE_HTTP_PORT        HTTP transport port (default: 3928)");
                println!(
                    "    VESTIGE_HTTP_ALLOWED_ORIGINS  Comma-separated browser origins allowed for HTTP"
                );
                println!("    VESTIGE_DASHBOARD_ENABLED Enable dashboard (default: disabled)");
                println!("    VESTIGE_DASHBOARD_PORT     Dashboard port (default: 3927)");
                println!(
                    "    VESTIGE_SYSTEM_PROMPT_MODE Inject the full composition mandate into every MCP session (minimal|full, default: minimal)"
                );
                println!();
                println!("EXAMPLES:");
                println!("    vestige-mcp");
                println!("    vestige-mcp --data-dir /custom/path");
                println!("    VESTIGE_DATA_DIR=~/.vestige vestige-mcp");
                println!("    vestige-mcp --http --http-port 8080");
                println!("    RUST_LOG=debug vestige-mcp");
                std::process::exit(0);
            }
            "--version" | "-V" => {
                println!("vestige-mcp {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--data-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --data-dir requires a path argument");
                    eprintln!("Usage: vestige-mcp --data-dir <PATH>");
                    std::process::exit(1);
                }
                if args[i].as_os_str().is_empty() {
                    eprintln!("error: --data-dir requires a non-empty path argument");
                    eprintln!("Usage: vestige-mcp --data-dir <PATH>");
                    std::process::exit(1);
                }
                data_dir = Some(PathBuf::from(&args[i]));
            }
            arg if arg.starts_with("--data-dir=") => {
                // Safe: we just verified the prefix exists with starts_with
                let path = arg.strip_prefix("--data-dir=").unwrap_or("");
                if path.is_empty() {
                    eprintln!("error: --data-dir requires a path argument");
                    eprintln!("Usage: vestige-mcp --data-dir <PATH>");
                    std::process::exit(1);
                }
                data_dir = Some(PathBuf::from(path));
            }
            "--http" => {
                http_enabled = true;
            }
            "--no-http" => {
                http_enabled = false;
            }
            "--http-port" => {
                http_enabled = true;
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --http-port requires a port number");
                    eprintln!("Usage: vestige-mcp --http-port <PORT>");
                    std::process::exit(1);
                }
                let port = args[i].to_string_lossy();
                http_port = match port.parse() {
                    Ok(p) => p,
                    Err(_) => {
                        eprintln!("error: invalid port number '{}'", port);
                        std::process::exit(1);
                    }
                };
            }
            arg if arg.starts_with("--http-port=") => {
                http_enabled = true;
                let val = arg.strip_prefix("--http-port=").unwrap_or("");
                http_port = match val.parse() {
                    Ok(p) => p,
                    Err(_) => {
                        eprintln!("error: invalid port number '{}'", val);
                        std::process::exit(1);
                    }
                };
            }
            arg => {
                eprintln!("error: unknown argument '{}'", arg);
                eprintln!("Usage: vestige-mcp [OPTIONS]");
                eprintln!("Try 'vestige-mcp --help' for more information.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Config {
        data_dir,
        http_port,
        http_enabled,
        dashboard_enabled,
    }
}

fn expand_tilde(path: PathBuf) -> PathBuf {
    let rest = {
        let mut components = path.components();
        match components.next() {
            Some(Component::Normal(first)) if first == "~" => {
                Some(components.as_path().to_path_buf())
            }
            _ => None,
        }
    };

    match rest {
        Some(rest) => BaseDirs::new()
            .map(|dirs| dirs.home_dir().join(rest))
            .unwrap_or(path),
        None => path,
    }
}

fn prepare_storage_path(data_dir: Option<PathBuf>) -> io::Result<Option<PathBuf>> {
    let Some(data_dir) = data_dir else {
        return Ok(None);
    };

    let data_dir = expand_tilde(data_dir);

    // Check if path exists and is a file (not a directory)
    if data_dir.exists() && !data_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Data directory path exists but is not a directory: {}",
                data_dir.display()
            ),
        ));
    }

    // Only create if it doesn't exist (avoids "File exists" error on existing directories)
    let created = !data_dir.exists();
    if created {
        fs::create_dir_all(&data_dir)?;
    }

    #[cfg(unix)]
    if created {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&data_dir, fs::Permissions::from_mode(0o700));
    }

    Ok(Some(data_dir.join(DATABASE_FILE)))
}

/// How long the runtime gets to stop its workers before the process leaves
/// them where they are.
const RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);

/// Stop the runtime, and report whether every runtime thread was joined.
///
/// Dropping a multi-thread runtime joins its worker threads, and a worker
/// running a task that is parked in a blocking `std::sync::Mutex` never gets to
/// the point where it can be joined. The stdio transport bounds its own EOF
/// drain and then abandons whatever it could not cancel (see
/// `protocol::stdio::run_io`), so a handler inside `vestige-core`'s
/// `Mutex<Connection>` can still be running when the transport returns. With
/// `#[tokio::main]`, which drops the runtime at the end of `main`, that drop is
/// where the process stopped instead of exiting. Measured: with one task
/// holding such a lock, a plain drop had not returned 10 s later, while this
/// bounded call returns inside its timeout.
///
/// The drop runs on its own thread rather than through
/// `Runtime::shutdown_timeout`, because that call does not report what it did.
/// `BlockingPool::shutdown` puts every join behind
/// `if self.shutdown_rx.wait(timeout)`, so one thread that misses the bound
/// leaves every thread unjoined, including threads that finished long before,
/// and the caller cannot tell that case from a clean stop. `false` here means
/// at least one runtime thread is still running, and [`stop_runtime`] leaves
/// the process without exit-time teardown on that verdict.
fn join_runtime(runtime: tokio::runtime::Runtime, timeout: Duration) -> bool {
    let (joined_tx, joined_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        drop(runtime);
        let _ = joined_tx.send(());
    });
    joined_rx.recv_timeout(timeout).is_ok()
}

/// Leave the process without running exit-time teardown.
///
/// Returning from `main` reaches libc `exit()`, and so does
/// `std::process::exit`. Both run atexit handlers and the static destructors of
/// everything linked in, including the statically linked ONNX Runtime's
/// `onnx::OpSchemaRegistry` op-schema map. `serve` warms embeddings and the
/// cross-encoder reranker on `spawn_blocking` threads, and each spends seconds
/// inside `OrtApis::CreateSession`. When [`join_runtime`] reports the runtime
/// threads unjoined, one of those threads can still be inside ONNX Runtime, and
/// running its destructors underneath it reads freed memory.
///
/// Measured on this branch before this call existed: `cargo test -p vestige-mcp
/// --test e2e_real_binary` raised `stdin EOF must be a clean shutdown, got
/// signal: 11 (SIGSEGV)` in 7 of 15 runs, with the faulting thread in
/// `onnx::OpSchemaRegistry::Schema` under `OrtApis::CreateSession` and the main
/// thread in `__run_exit_handlers`. Deleting the call to this function brings
/// that flaky SIGSEGV back and produces no compile error, which is why
/// [`stop_runtime`] takes it as an argument and is tested through it.
///
/// `serve` uses it for the server-error exit as well. That path leaves with the
/// whole runtime live, including any warm-up still inside ONNX Runtime, so it
/// has the same teardown race and keeps its exit status of 1.
fn leave_without_running_exit_handlers(code: i32) -> ! {
    use std::io::Write;

    // `_exit` skips the flush that Rust's normal exit path performs.
    let _ = io::stdout().flush();
    let _ = io::stderr().flush();

    #[cfg(unix)]
    // SAFETY: `_exit` ends this process at once. It runs no user code and does
    // not return, so there is nothing for a still-running thread to race.
    unsafe {
        unsafe extern "C" {
            fn _exit(code: i32) -> !;
        }
        _exit(code)
    }
    // Off unix there is no teardown-free exit here: `std::process::exit` still
    // runs the CRT's atexit list and the DLL detach routines. That narrows the
    // window rather than closing it, and no occurrence has been observed there.
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// Stop the runtime, and leave the process when it cannot be stopped.
///
/// [`join_runtime`] produces the verdict and this function acts on it. The
/// acting half is a single call that compiles cleanly when it is removed, and
/// what it prevents showed up in 7 of 15 integration runs rather than in every
/// one, so it sits here with `leave_without_teardown` as an argument and the
/// tests below assert both directions: the call runs when a worker is parked,
/// and it does not run when the runtime joins.
fn stop_runtime(
    runtime: tokio::runtime::Runtime,
    timeout: Duration,
    leave_without_teardown: fn(i32) -> !,
) {
    if join_runtime(runtime, timeout) {
        return;
    }

    warn!(
        "A runtime thread was still running {timeout:?} after shutdown began, \
         so this process is leaving without exit-time teardown"
    );
    leave_without_teardown(0);
}

fn main() {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("Failed to start the tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    runtime.block_on(serve());
    stop_runtime(
        runtime,
        RUNTIME_SHUTDOWN_TIMEOUT,
        leave_without_running_exit_handlers,
    );
}

async fn serve() {
    // Parse CLI arguments first (before logging init, so --help/--version work cleanly)
    let config = parse_args();

    // Initialize logging to stderr (stdout is for JSON-RPC)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .with_writer(io::stderr)
        .with_target(false)
        .with_ansi(false)
        // MCP stdio clients can close stderr before stdin.  The formatter's
        // default fallback tries `eprintln!` after a failed stderr write,
        // which panics on that same closed descriptor during shutdown.
        .log_internal_errors(false)
        .init();

    info!(
        "Vestige MCP Server v{} starting...",
        env!("CARGO_PKG_VERSION")
    );

    let storage_path = match prepare_storage_path(config.data_dir) {
        Ok(path) => path,
        Err(e) => {
            error!("Failed to prepare storage data directory: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize storage with optional custom data directory.
    // Storage::new(Some(...)) expects a DB file path, so map data dirs to vestige.db here.
    let storage = match Storage::new(storage_path) {
        Ok(s) => {
            info!("Storage initialized successfully");
            Arc::new(s)
        }
        Err(e) => {
            error!("Failed to initialize storage: {}", e);
            std::process::exit(1);
        }
    };

    // Preserve the released Nomic default in the background so MCP clients can
    // finish their stdio handshake before a first-run model download. Optional
    // profiles reject this compatibility path: their artifact verification,
    // evaluation, migration, and activation remain explicit local operations.
    // The stdio transport and the notifier that lets background work tell the
    // client what it is doing. Created before the warm-up tasks so a first-run
    // model download can announce itself as MCP logging instead of dying on
    // stderr, which stdio clients hide.
    let (transport, notifier) = StdioTransport::with_notifications();
    // In builds without an embedding runtime nothing warms up, so the notifier
    // has no sender beyond this scope; dropping it parks the channel.
    let _notifier: Notifier = notifier.clone();

    #[cfg(feature = "embeddings")]
    {
        let storage_clone = Arc::clone(&storage);
        let notifier = notifier.clone();
        tokio::task::spawn_blocking(move || {
            let first_run = !vestige_core::embeddings::embedding_model_cached();
            notifier.log(
                "info",
                "vestige.embeddings",
                serde_json::json!({
                    "event": if first_run { "model_download_started" } else { "model_loading" },
                    "model": "nomic-ai/nomic-embed-text-v1.5",
                    "approxBytes": if first_run { Some(130_000_000u64) } else { None },
                    "effect": "recall answers by keyword and smart_ingest stores without a vector until the runtime is ready; those responses carry a `warming` block meanwhile",
                }),
            );
            if let Err(error) = storage_clone.init_embeddings() {
                tracing::debug!(%error, "No legacy Nomic embedding runtime started");
                notifier.log(
                    "warning",
                    "vestige.embeddings",
                    serde_json::json!({ "event": "embedding_runtime_unavailable", "error": error.to_string() }),
                );
                return;
            }
            info!("Legacy Nomic embedding service initialized successfully");
            notifier.log(
                "info",
                "vestige.embeddings",
                serde_json::json!({ "event": "embedding_runtime_ready" }),
            );

            #[cfg(feature = "vector-search")]
            match storage_clone.generate_embeddings(None, false) {
                Ok(result) if result.successful > 0 || result.failed > 0 => info!(
                    embeddings_generated = result.successful,
                    embeddings_failed = result.failed,
                    embeddings_skipped = result.skipped,
                    "Background legacy Nomic embedding backfill complete"
                ),
                Ok(_) => {}
                Err(error) => warn!(%error, "Background legacy Nomic embedding backfill failed"),
            }
        });
    }

    // Startup hygiene: sweep Black Box traces past VESTIGE_TRACE_RETENTION_DAYS
    // now, not only when the consolidation cycle next runs. Best-effort.
    match storage.prune_agent_traces() {
        Ok(deleted) if deleted > 0 => info!(deleted, "Pruned expired agent trace events at startup"),
        Ok(_) => {}
        Err(e) => warn!("Startup trace retention sweep failed: {}", e),
    }

    // Periodic WAL checkpoint. `wal_autocheckpoint` already runs on commit, but
    // a PASSIVE checkpoint every 60 s (never blocks readers or writers) folds
    // the .wal back into the main file across long uptimes and reports a WAL
    // that keeps growing instead of letting it reach gigabytes unnoticed.
    {
        let storage_clone = storage.clone();
        tokio::spawn(async move {
            // Roughly 40 MB at the 4 KiB default page size, 80 MB at 8 KiB.
            const WAL_WARN_FRAMES: i64 = 10_000;
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await; // the first tick completes immediately; skip it
            loop {
                ticker.tick().await;
                let storage = storage_clone.clone();
                let result = tokio::task::spawn_blocking(move || {
                    storage.checkpoint_wal(vestige_core::storage::WalCheckpointMode::Passive)
                })
                .await;
                match result {
                    Ok(Ok(status)) if status.log_frames > WAL_WARN_FRAMES => warn!(
                        log_frames = status.log_frames,
                        checkpointed_frames = status.checkpointed_frames,
                        busy = status.busy,
                        "WAL is large after a passive checkpoint; a long-running reader may be pinning it"
                    ),
                    Ok(Ok(status)) => debug!(
                        log_frames = status.log_frames,
                        checkpointed_frames = status.checkpointed_frames,
                        "Periodic WAL checkpoint"
                    ),
                    Ok(Err(e)) => warn!("Periodic WAL checkpoint failed: {}", e),
                    Err(e) => warn!("Periodic WAL checkpoint task failed to run: {}", e),
                }
            }
        });
    }

    // Spawn periodic auto-consolidation so FSRS-6 decay scores stay fresh.
    // Runs on startup (if needed) and then every N hours (default: 6).
    // Configurable via VESTIGE_CONSOLIDATION_INTERVAL_HOURS env var.
    {
        let storage_clone = storage.clone();
        tokio::spawn(async move {
            let interval_hours: u64 = std::env::var("VESTIGE_CONSOLIDATION_INTERVAL_HOURS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(6);

            // Small delay so we don't block server startup / stdio handshake
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;

            loop {
                // Check whether consolidation is actually needed
                let should_run = match storage_clone.get_last_consolidation() {
                    Ok(Some(last)) => {
                        let elapsed = chrono::Utc::now() - last;
                        let stale = elapsed > chrono::Duration::hours(interval_hours as i64);
                        if !stale {
                            info!(
                                last_consolidation = %last,
                                "Skipping auto-consolidation (last run was < {} hours ago)",
                                interval_hours
                            );
                        }
                        stale
                    }
                    Ok(None) => {
                        info!("No previous consolidation found — running first auto-consolidation");
                        true
                    }
                    Err(e) => {
                        warn!(
                            "Could not read consolidation history: {} — running anyway",
                            e
                        );
                        true
                    }
                };

                if should_run {
                    match storage_clone.run_consolidation() {
                        Ok(result) => {
                            info!(
                                nodes_processed = result.nodes_processed,
                                decay_applied = result.decay_applied,
                                embeddings_generated = result.embeddings_generated,
                                duplicates_merged = result.duplicates_merged,
                                activations_computed = result.activations_computed,
                                duration_ms = result.duration_ms,
                                "Periodic auto-consolidation complete"
                            );
                        }
                        Err(e) => {
                            warn!("Periodic auto-consolidation failed: {}", e);
                        }
                    }

                    // v2.0.5: Rac1 cascade sweep — walk recently-suppressed
                    // memories and fade their co-activated neighbors
                    // (Cervantes-Sandoval & Davis 2020, PMC7477079).
                    match storage_clone.run_rac1_cascade_sweep() {
                        Ok((seeds, affected)) if seeds > 0 || affected > 0 => {
                            info!(
                                suppressed_seeds = seeds,
                                neighbors_affected = affected,
                                "Rac1 cascade sweep complete"
                            );
                        }
                        Ok(_) => {}
                        Err(e) => {
                            warn!("Rac1 cascade sweep failed: {}", e);
                        }
                    }
                }

                // Sleep until next check
                tokio::time::sleep(std::time::Duration::from_secs(interval_hours * 3600)).await;
            }
        });
    }

    // Create cognitive engine (stateful neuroscience modules)
    let cognitive = Arc::new(Mutex::new(cognitive::CognitiveEngine::new()));
    // Hydrate cognitive modules from persisted connections
    {
        let mut cog = cognitive.lock().await;
        cog.hydrate(&storage);
    }
    info!("CognitiveEngine initialized and hydrated");

    // Create shared event broadcast channel for dashboard <-> MCP tool events
    let (event_tx, _) =
        tokio::sync::broadcast::channel::<vestige_mcp::dashboard::events::VestigeEvent>(
            vestige_mcp::dashboard::state::EVENT_CHANNEL_CAPACITY,
        );

    // v2.0.9 "Autopilot" — spawn the backend event-subscriber that routes
    // every live WebSocket event into the cognitive modules that already
    // have trigger methods implemented. Without this, the 20 event types
    // terminate at the dashboard and the cognitive engine is a passive
    // library that only responds to MCP tool queries.
    //
    // See `crates/vestige-mcp/src/autopilot.rs` for the routing table and
    // `docs/VESTIGE_STATE_AND_PLAN.md` §15 for the architectural rationale.
    vestige_mcp::autopilot::spawn(
        Arc::clone(&cognitive),
        Arc::clone(&storage),
        event_tx.clone(),
    );

    // Spawn dashboard HTTP server alongside MCP server (now with CognitiveEngine access)
    if config.dashboard_enabled {
        let dashboard_port = std::env::var("VESTIGE_DASHBOARD_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(3927);
        let dashboard_storage = Arc::clone(&storage);
        let dashboard_cognitive = Arc::clone(&cognitive);
        let dashboard_event_tx = event_tx.clone();
        tokio::spawn(async move {
            match vestige_mcp::dashboard::start_background_with_event_tx(
                dashboard_storage,
                Some(dashboard_cognitive),
                dashboard_event_tx,
                dashboard_port,
            )
            .await
            {
                Ok(_state) => {
                    info!("Dashboard started with WebSocket + CognitiveEngine + shared event bus");
                }
                Err(e) => {
                    warn!("Dashboard failed to start: {}", e);
                }
            }
        });
    } else {
        info!("Dashboard disabled by VESTIGE_DASHBOARD_ENABLED=false");
    }

    // Start optional HTTP MCP transport for clients that need Streamable HTTP.
    if config.http_enabled {
        let http_storage = Arc::clone(&storage);
        let http_cognitive = Arc::clone(&cognitive);
        let http_event_tx = event_tx.clone();
        let http_port = config.http_port;

        match protocol::auth::get_or_create_auth_token() {
            Ok(token) => {
                let bind =
                    std::env::var("VESTIGE_HTTP_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
                eprintln!("Vestige HTTP transport: http://{}:{}/mcp", bind, http_port);
                if let Ok(path) = protocol::auth::token_path() {
                    eprintln!("Auth token file: {}", path.display());
                }
                tokio::spawn(async move {
                    if let Err(e) = protocol::http::start_http_transport(
                        http_storage,
                        http_cognitive,
                        http_event_tx,
                        token,
                        http_port,
                    )
                    .await
                    {
                        warn!("HTTP transport failed to start: {}", e);
                    }
                });
            }
            Err(e) => {
                warn!(
                    "Could not create auth token, HTTP transport disabled: {}",
                    e
                );
            }
        }
    } else {
        info!("HTTP MCP transport disabled; set VESTIGE_HTTP_ENABLED=1 or pass --http to enable");
    }

    // Load cross-encoder reranker in the background (downloads ~150MB on first run)
    #[cfg(all(feature = "vector-search", feature = "embeddings"))]
    {
        let cog_clone = Arc::clone(&cognitive);
        let notifier = notifier.clone();
        tokio::spawn(async move {
            // Small delay so we don't block the stdio handshake
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            notifier.log(
                "info",
                "vestige.reranker",
                serde_json::json!({
                    "event": "reranker_loading",
                    "note": "a first run downloads about 150 MB; recall ranks by BM25 until it is ready",
                }),
            );
            // The model load is synchronous and downloads ~150MB on a fresh
            // install. Doing it under the CognitiveEngine mutex stalled every
            // tool that shares that lock (explore, predict, session_context,
            // memory_unified, dream, and the rest) for the whole download, on
            // every startup rather than on demand, and it blocked a tokio
            // worker thread while it ran. Load on the blocking pool holding
            // nothing, then take the lock only to install the result.
            let loaded = tokio::task::spawn_blocking(
                vestige_core::search::Reranker::load_cross_encoder,
            )
            .await;
            match loaded {
                Ok(Some(model)) => {
                    let mut cog = cog_clone.lock().await;
                    cog.reranker.install_cross_encoder(model);
                    notifier.log(
                        "info",
                        "vestige.reranker",
                        serde_json::json!({ "event": "reranker_ready" }),
                    );
                }
                // `None` already logged its reason; BM25 fallback stands.
                Ok(None) => notifier.log(
                    "warning",
                    "vestige.reranker",
                    serde_json::json!({ "event": "reranker_unavailable", "effect": "BM25 ranking stands" }),
                ),
                Err(e) => warn!("Cross-encoder load task failed: {e}"),
            }
        });
    }

    // Create MCP server with shared event channel for dashboard broadcasts
    let server = McpServer::new_with_events(storage, cognitive, event_tx);

    info!("Starting MCP server on stdio...");

    // Run the server
    if let Err(e) = transport.run(server).await {
        error!("Server error: {}", e);
        // Not `std::process::exit`: this runs on a runtime thread with the
        // warm-up tasks possibly still inside ONNX Runtime, which is the state
        // `leave_without_running_exit_handlers` documents. The status stays 1.
        leave_without_running_exit_handlers(1);
    }

    info!("Vestige MCP Server shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os_args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    /// The exit handed to `stop_runtime` by the two tests below. It has to
    /// diverge, because the real one does, so it reports by panicking and the
    /// tests read that through `catch_unwind`.
    fn panic_instead_of_leaving(code: i32) -> ! {
        panic!("the teardown-free exit ran with status {code}");
    }

    /// A worker parked in a blocking lock is the state the stdio transport can
    /// leave behind when its EOF bound abandons a handler it could not cancel.
    /// `stop_runtime` has to take the teardown-free exit there. Returning
    /// instead reaches libc's exit path, whose static destructors ran
    /// underneath a live ONNX Runtime warm-up in 7 of 15 runs of
    /// `e2e_real_binary` on this branch.
    ///
    /// The assertion covers both halves of the fix: `join_runtime` reporting
    /// the parked worker as unjoined, and the call that acts on that verdict.
    /// Deleting either one leaves this test failing and the build clean.
    #[test]
    fn stop_runtime_leaves_without_teardown_when_a_worker_is_parked_in_a_blocking_lock() {
        static PARK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        let held = PARK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let (parked_tx, parked_rx) = std::sync::mpsc::channel::<()>();
        let (released_tx, released_rx) = std::sync::mpsc::channel::<()>();
        runtime.spawn(async move {
            parked_tx.send(()).unwrap();
            let _parked = PARK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = released_tx.send(());
        });
        parked_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the parked task started");
        // The task is on its way to the lock; give it the moment it needs to
        // get there, so the worker is genuinely parked.
        std::thread::sleep(Duration::from_millis(200));

        let left = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stop_runtime(runtime, RUNTIME_SHUTDOWN_TIMEOUT, panic_instead_of_leaving);
        }))
        .is_err();

        // Release the park whatever the outcome, so the thread `join_runtime`
        // left dropping the runtime can finish before this test returns.
        drop(held);
        let _ = released_rx.recv_timeout(Duration::from_secs(10));

        assert!(
            left,
            "stop_runtime returned while a worker was parked in a blocking lock, \
             so main would return into libc's exit path with that thread still \
             running"
        );
    }

    /// The positive control for the test above: with nothing parked, the same
    /// bound joins everything, and `stop_runtime` has to return so that `main`
    /// returns and the exit handlers still run, the coverage profile writer
    /// among them.
    #[test]
    fn stop_runtime_returns_without_leaving_when_the_runtime_joins() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {});

        let left = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stop_runtime(runtime, Duration::from_secs(10), panic_instead_of_leaving);
        }))
        .is_err();

        assert!(
            !left,
            "stop_runtime took the teardown-free exit after joining an idle \
             runtime, which would skip every atexit handler on a clean shutdown"
        );
    }

    #[test]
    fn vestige_data_dir_env_is_used_when_cli_data_dir_is_absent() {
        let config = parse_args_from(
            os_args(&["vestige-mcp"]),
            Some(PathBuf::from("/tmp/vestige-env")),
        );

        assert_eq!(config.data_dir, Some(PathBuf::from("/tmp/vestige-env")));
        assert!(!config.http_enabled);
    }

    #[test]
    fn cli_data_dir_takes_precedence_over_env_data_dir() {
        let config = parse_args_from(
            os_args(&["vestige-mcp", "--data-dir", "/tmp/vestige-cli"]),
            Some(PathBuf::from("/tmp/vestige-env")),
        );

        assert_eq!(config.data_dir, Some(PathBuf::from("/tmp/vestige-cli")));
    }

    #[test]
    fn http_is_opt_in_and_port_flag_enables_it() {
        let disabled = parse_args_from(os_args(&["vestige-mcp"]), None);
        assert!(!disabled.http_enabled);

        let enabled = parse_args_from(os_args(&["vestige-mcp", "--http-port", "8080"]), None);
        assert!(enabled.http_enabled);
        assert_eq!(enabled.http_port, 8080);
    }

    #[test]
    fn prepare_storage_path_creates_dir_and_points_to_vestige_db() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("nested").join("vestige");

        let db_path = prepare_storage_path(Some(data_dir.clone())).unwrap();

        assert!(data_dir.is_dir());
        assert_eq!(db_path, Some(data_dir.join(DATABASE_FILE)));
    }

    #[test]
    fn prepare_storage_path_reuses_existing_data_dir() {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("existing");
        fs::create_dir_all(&data_dir).unwrap();

        let db_path = prepare_storage_path(Some(data_dir.clone())).unwrap();

        assert_eq!(db_path, Some(data_dir.join(DATABASE_FILE)));
    }

    #[cfg(unix)]
    #[test]
    fn prepare_storage_path_preserves_existing_data_dir_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("shared");
        fs::create_dir_all(&data_dir).unwrap();
        fs::set_permissions(&data_dir, fs::Permissions::from_mode(0o755)).unwrap();

        let db_path = prepare_storage_path(Some(data_dir.clone())).unwrap();
        let mode = fs::metadata(&data_dir).unwrap().permissions().mode() & 0o777;

        assert_eq!(db_path, Some(data_dir.join(DATABASE_FILE)));
        assert_eq!(mode, 0o755);
    }

    #[test]
    fn expand_tilde_expands_current_users_home_only() {
        let home = BaseDirs::new().unwrap().home_dir().to_path_buf();

        assert_eq!(
            expand_tilde(PathBuf::from("~/vestige")),
            home.join("vestige")
        );
        assert_eq!(
            expand_tilde(PathBuf::from("~other/vestige")),
            PathBuf::from("~other/vestige")
        );
    }
}

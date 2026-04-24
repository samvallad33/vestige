//! Vestige MCP Server v1.0 - Cognitive Memory for Claude
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

use vestige_mcp::cognitive;
use vestige_mcp::protocol;
use vestige_mcp::server;

use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{Level, error, info, warn};
use tracing_subscriber::EnvFilter;

// Use vestige-core for the cognitive science engine
use vestige_core::Storage;

use protocol::stdio::StdioTransport;
use server::McpServer;

/// Parsed CLI configuration.
struct Config {
    data_dir: Option<PathBuf>,
    http_port: u16,
    dashboard_enabled: bool,
}

/// Parse command-line arguments into a `Config`.
/// Exits the process if `--help` or `--version` is requested.
fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir: Option<PathBuf> = None;
    let mut http_port: u16 = std::env::var("VESTIGE_HTTP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3928);
    let dashboard_enabled = std::env::var("VESTIGE_DASHBOARD_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false);
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
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
                println!("    --data-dir <PATH>       Custom data directory");
                println!("    --http-port <PORT>      HTTP transport port (default: 3928)");
                println!();
                println!("ENVIRONMENT:");
                println!(
                    "    RUST_LOG                  Log level filter (e.g., debug, info, warn, error)"
                );
                println!(
                    "    VESTIGE_AUTH_TOKEN         Override the bearer token for HTTP transport"
                );
                println!("    VESTIGE_HTTP_PORT          HTTP transport port (default: 3928)");
                println!("    VESTIGE_DASHBOARD_ENABLED     Enable dashboard (default: disabled)");
                println!("    VESTIGE_DASHBOARD_PORT     Dashboard port (default: 3927)");
                println!(
                    "    VESTIGE_SYSTEM_PROMPT_MODE Inject the full composition mandate into every MCP session (minimal|full, default: minimal)"
                );
                println!();
                println!("EXAMPLES:");
                println!("    vestige-mcp");
                println!("    vestige-mcp --data-dir /custom/path");
                println!("    vestige-mcp --http-port 8080");
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
            "--http-port" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --http-port requires a port number");
                    eprintln!("Usage: vestige-mcp --http-port <PORT>");
                    std::process::exit(1);
                }
                http_port = match args[i].parse() {
                    Ok(p) => p,
                    Err(_) => {
                        eprintln!("error: invalid port number '{}'", args[i]);
                        std::process::exit(1);
                    }
                };
            }
            arg if arg.starts_with("--http-port=") => {
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
        dashboard_enabled,
    }
}

#[tokio::main]
async fn main() {
    // Parse CLI arguments first (before logging init, so --help/--version work cleanly)
    let config = parse_args();

    // Initialize logging to stderr (stdout is for JSON-RPC)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .with_writer(io::stderr)
        .with_target(false)
        .with_ansi(false)
        .init();

    info!(
        "Vestige MCP Server v{} starting...",
        env!("CARGO_PKG_VERSION")
    );

    // Initialize storage with optional custom data directory
    let storage = match Storage::new(config.data_dir) {
        Ok(s) => {
            info!("Storage initialized successfully");

            // Try to initialize embeddings early and log any issues
            #[cfg(feature = "embeddings")]
            {
                if let Err(e) = s.init_embeddings() {
                    error!("Failed to initialize embedding service: {}", e);
                    error!("Smart ingest will fall back to regular ingest without deduplication");
                    error!(
                        "Hint: Check FASTEMBED_CACHE_PATH or ensure ~/.cache/vestige/fastembed is writable"
                    );
                } else {
                    info!("Embedding service initialized successfully");
                }
            }

            Arc::new(s)
        }
        Err(e) => {
            error!("Failed to initialize storage: {}", e);
            std::process::exit(1);
        }
    };

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
        tokio::sync::broadcast::channel::<vestige_mcp::dashboard::events::VestigeEvent>(1024);

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

    // Start HTTP MCP transport (Streamable HTTP for Claude.ai / remote clients)
    {
        let http_storage = Arc::clone(&storage);
        let http_cognitive = Arc::clone(&cognitive);
        let http_event_tx = event_tx.clone();
        let http_port = config.http_port;

        match protocol::auth::get_or_create_auth_token() {
            Ok(token) => {
                let bind =
                    std::env::var("VESTIGE_HTTP_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
                eprintln!("Vestige HTTP transport: http://{}:{}/mcp", bind, http_port);
                eprintln!("Auth token: {}...", &token[..token.len().min(8)]);
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
    }

    // Load cross-encoder reranker in the background (downloads ~150MB on first run)
    #[cfg(feature = "embeddings")]
    {
        let cog_clone = Arc::clone(&cognitive);
        tokio::spawn(async move {
            // Small delay so we don't block the stdio handshake
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let mut cog = cog_clone.lock().await;
            cog.reranker.init_cross_encoder();
        });
    }

    // Create MCP server with shared event channel for dashboard broadcasts
    let server = McpServer::new_with_events(storage, cognitive, event_tx);

    // Create stdio transport
    let transport = StdioTransport::new();

    info!("Starting MCP server on stdio...");

    // Run the server
    if let Err(e) = transport.run(server).await {
        error!("Server error: {}", e);
        std::process::exit(1);
    }

    info!("Vestige MCP Server shutting down");
}

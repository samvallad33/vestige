//! Vestige CLI
//!
//! Command-line interface for managing cognitive memory system.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use chrono::{NaiveDate, Utc};
use clap::{Args, Parser, Subcommand};
use colored::Colorize;
use vestige_core::{IngestInput, PortableImportMode, Storage};

/// Vestige - Cognitive Memory System CLI
#[derive(Parser)]
#[command(name = "vestige")]
#[command(author = "samvallad33")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CLI for the Vestige cognitive memory system")]
#[command(
    long_about = "Vestige is a cognitive memory system based on 130 years of memory research.\n\nIt implements FSRS-6, spreading activation, synaptic tagging, and more."
)]
struct Cli {
    /// Use a specific Vestige data directory for this command.
    #[arg(long, global = true, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

static CLI_DB_PATH: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Clone, Default, Args)]
struct SandwichInstallOptions {
    /// Overwrite existing staged Vestige hook and agent files.
    #[arg(long)]
    force: bool,

    /// Wire optional UserPromptSubmit preflight hooks.
    #[arg(long)]
    enable_preflight: bool,

    /// Wire both optional preflight hooks and the optional Sanhedrin verifier.
    #[arg(long)]
    enable_sandwich: bool,

    /// Wire optional Sanhedrin Stop hook.
    #[arg(long)]
    enable_sanhedrin: bool,

    /// On Apple Silicon, auto-start the local MLX Sanhedrin backend.
    #[arg(long)]
    with_launchd: bool,

    /// Also stage the large memory-loader hook file.
    #[arg(long)]
    include_memory_loader: bool,

    /// OpenAI-compatible chat completions endpoint for optional Sanhedrin.
    #[arg(long, value_name = "URL")]
    sanhedrin_endpoint: Option<String>,

    /// Model name passed to the optional Sanhedrin endpoint.
    #[arg(long, value_name = "MODEL")]
    sanhedrin_model: Option<String>,

    /// Use a local checkout/release root containing hooks/ and agents/.
    #[arg(long, value_name = "DIR", hide = true)]
    src: Option<PathBuf>,
}

#[derive(Subcommand)]
enum SandwichCommands {
    /// Install/update Cognitive Sandwich companion files without enabling hooks by default.
    Install {
        /// Install files from a specific release tag instead of latest.
        #[arg(long)]
        version: Option<String>,

        #[command(flatten)]
        options: SandwichInstallOptions,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Show memory statistics
    Stats {
        /// Show tagging/retention distribution
        #[arg(long)]
        tagging: bool,

        /// Show cognitive state distribution
        #[arg(long)]
        states: bool,
    },

    /// Run health check with warnings and recommendations
    Health,

    /// Run memory consolidation cycle
    Consolidate,

    /// Update Vestige binaries from the latest GitHub release
    Update {
        /// Install a specific release tag instead of latest (example: v2.1.21)
        #[arg(long)]
        version: Option<String>,

        /// Override install directory (defaults to the current vestige binary's directory)
        #[arg(long)]
        install_dir: Option<PathBuf>,

        /// Print what would be updated without changing files
        #[arg(long)]
        dry_run: bool,

        /// Deprecated: companion updates are skipped by default.
        #[arg(long)]
        no_sandwich: bool,

        /// Also refresh optional Claude Code Cognitive Sandwich companion files.
        #[arg(long)]
        sandwich_companion: bool,

        #[command(flatten)]
        sandwich: SandwichInstallOptions,
    },

    /// Manage optional Claude Code Cognitive Sandwich companion files.
    Sandwich {
        #[command(subcommand)]
        command: SandwichCommands,
    },

    /// Restore memories from backup file
    Restore {
        /// Path to backup JSON file
        file: PathBuf,
    },

    /// Create a full backup of the SQLite database
    Backup {
        /// Output file path for the backup
        output: PathBuf,
    },

    /// Export memories in JSON or JSONL format
    Export {
        /// Output file path
        output: PathBuf,
        /// Export format: json or jsonl
        #[arg(long, default_value = "json")]
        format: String,
        /// Filter by tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Only export memories created after this date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,
    },

    /// Export an exact portable archive for Vestige-to-Vestige transfer
    PortableExport {
        /// Output archive path
        output: PathBuf,
    },

    /// Import an exact portable archive
    PortableImport {
        /// Input archive path
        input: PathBuf,
        /// Merge into the current database instead of requiring an empty target
        #[arg(long)]
        merge: bool,
    },

    /// Two-way sync with a file-backed portable archive, or Vestige Cloud
    Sync {
        /// Sync archive path, often in Dropbox/iCloud/Syncthing/Git.
        /// Omit when using --cloud.
        archive: Option<PathBuf>,
        /// Sync with the hosted Vestige Cloud managed-sync service instead of a
        /// file. Requires a sync key (VESTIGE_CLOUD_SYNC_KEY) and endpoint
        /// (--endpoint or VESTIGE_CLOUD_ENDPOINT).
        #[arg(long)]
        cloud: bool,
        /// Vestige Cloud base endpoint (e.g. https://sync.vestige.dev).
        /// Defaults to the VESTIGE_CLOUD_ENDPOINT env var.
        #[arg(long)]
        endpoint: Option<String>,
    },

    /// Garbage collect stale memories below retention threshold
    Gc {
        /// Minimum retention strength to keep (delete below this)
        #[arg(long, default_value = "0.1")]
        min_retention: f64,
        /// Maximum age in days (delete memories older than this AND below retention threshold)
        #[arg(long)]
        max_age_days: Option<u64>,
        /// Dry run - show what would be deleted without actually deleting
        #[arg(long)]
        dry_run: bool,
        /// Skip confirmation prompt
        #[arg(long)]
        yes: bool,
    },

    /// Launch the memory web dashboard
    Dashboard {
        /// Port to bind the dashboard server to
        #[arg(long, default_value = "3927")]
        port: u16,
        /// Don't automatically open the browser
        #[arg(long)]
        no_open: bool,
    },

    /// Ingest a memory (routes through Prediction Error Gating)
    Ingest {
        /// Content to remember
        content: String,
        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,
        /// Node type (fact, concept, event, person, place, note, pattern, decision)
        #[arg(long, default_value = "fact")]
        node_type: String,
        /// Source reference
        #[arg(long)]
        source: Option<String>,
    },

    /// Start standalone HTTP MCP server (no stdio, for remote access)
    Serve {
        /// HTTP transport port
        #[arg(long, default_value = "3928")]
        port: u16,
        /// Also start the dashboard
        #[arg(long)]
        dashboard: bool,
        /// Dashboard port
        #[arg(long, default_value = "3927")]
        dashboard_port: u16,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(data_dir) = cli.data_dir {
        let db_path = Storage::db_path_for_data_dir(data_dir)?;
        CLI_DB_PATH
            .set(db_path)
            .map_err(|_| anyhow::anyhow!("data directory was initialized more than once"))?;
    }

    match cli.command {
        Commands::Stats { tagging, states } => run_stats(tagging, states),
        Commands::Health => run_health(),
        Commands::Consolidate => run_consolidate(),
        Commands::Update {
            version,
            install_dir,
            dry_run,
            no_sandwich,
            sandwich_companion,
            sandwich,
        } => run_update(
            version,
            install_dir,
            dry_run,
            no_sandwich,
            sandwich_companion,
            sandwich,
        ),
        Commands::Sandwich { command } => match command {
            SandwichCommands::Install { version, options } => {
                run_sandwich_install(version.as_deref(), &options)
            }
        },
        Commands::Restore { file } => run_restore(file),
        Commands::Backup { output } => run_backup(output),
        Commands::Export {
            output,
            format,
            tags,
            since,
        } => run_export(output, format, tags, since),
        Commands::PortableExport { output } => run_portable_export(output),
        Commands::PortableImport { input, merge } => run_portable_import(input, merge),
        Commands::Sync {
            archive,
            cloud,
            endpoint,
        } => run_sync(archive, cloud, endpoint),
        Commands::Gc {
            min_retention,
            max_age_days,
            dry_run,
            yes,
        } => run_gc(min_retention, max_age_days, dry_run, yes),
        Commands::Dashboard { port, no_open } => run_dashboard(port, !no_open),
        Commands::Ingest {
            content,
            tags,
            node_type,
            source,
        } => run_ingest(content, tags, node_type, source),
        Commands::Serve {
            port,
            dashboard,
            dashboard_port,
        } => run_serve(port, dashboard, dashboard_port),
    }
}

#[derive(Debug, Clone, Copy)]
struct ReleaseAsset {
    target: &'static str,
    archive_ext: &'static str,
    binary_suffix: &'static str,
}

struct UpdateTempDir {
    path: PathBuf,
}

impl UpdateTempDir {
    fn create() -> anyhow::Result<Self> {
        let path = env::temp_dir().join(format!(
            "vestige-update-{}-{}",
            std::process::id(),
            Utc::now().timestamp_millis()
        ));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create temp directory {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for UpdateTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn release_asset_for(os: &str, arch: &str) -> anyhow::Result<ReleaseAsset> {
    match (os, arch) {
        ("macos", "aarch64") => Ok(ReleaseAsset {
            target: "aarch64-apple-darwin",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("macos", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-apple-darwin",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("linux", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-unknown-linux-gnu",
            archive_ext: "tar.gz",
            binary_suffix: "",
        }),
        ("windows", "x86_64") => Ok(ReleaseAsset {
            target: "x86_64-pc-windows-msvc",
            archive_ext: "zip",
            binary_suffix: ".exe",
        }),
        _ => anyhow::bail!(
            "unsupported platform for vestige update: {}-{}. Download manually from https://github.com/samvallad33/vestige/releases",
            os,
            arch
        ),
    }
}

fn current_release_asset() -> anyhow::Result<ReleaseAsset> {
    release_asset_for(env::consts::OS, env::consts::ARCH)
}

fn release_download_url(asset: ReleaseAsset, version: Option<&str>) -> String {
    let archive_name = format!("vestige-mcp-{}.{}", asset.target, asset.archive_ext);
    match version {
        Some(version) => {
            let tag = normalize_release_tag(version);
            format!(
                "https://github.com/samvallad33/vestige/releases/download/{}/{}",
                tag, archive_name
            )
        }
        None => format!(
            "https://github.com/samvallad33/vestige/releases/latest/download/{}",
            archive_name
        ),
    }
}

fn normalize_release_tag(version: &str) -> String {
    if version.starts_with('v') {
        version.to_string()
    } else {
        format!("v{}", version)
    }
}

fn source_archive_url(tag: &str) -> String {
    format!(
        "https://github.com/samvallad33/vestige/archive/refs/tags/{}.tar.gz",
        tag
    )
}

fn download_file(url: &str, output: &Path, action: &str) -> anyhow::Result<()> {
    run_command(
        Command::new("curl")
            .arg("-fsSL")
            .arg("-A")
            .arg("vestige-cli")
            .arg(url)
            .arg("-o")
            .arg(output),
        action,
    )
}

fn parse_sha256(text: &str) -> anyhow::Result<String> {
    let hash = text
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow::anyhow!("checksum file is empty"))?
        .to_ascii_lowercase();
    if hash.len() != 64 || !hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        anyhow::bail!("checksum file does not contain a valid SHA-256 hash");
    }
    Ok(hash)
}

fn sha256_from_command(command: &mut Command) -> anyhow::Result<Option<String>> {
    match command.output() {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            Ok(Some(parse_sha256(&text)?))
        }
        Ok(_) => Ok(None),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).context("failed to run checksum command"),
    }
}

fn compute_sha256(path: &Path) -> anyhow::Result<String> {
    #[cfg(windows)]
    {
        if let Some(hash) = sha256_from_command(
            Command::new("powershell")
                .arg("-NoProfile")
                .arg("-Command")
                .arg("(Get-FileHash -Algorithm SHA256 -LiteralPath $args[0]).Hash.ToLowerInvariant()")
                .arg(path),
        )? {
            return Ok(hash);
        }
    }

    #[cfg(not(windows))]
    {
        if let Some(hash) =
            sha256_from_command(Command::new("shasum").arg("-a").arg("256").arg(path))?
        {
            return Ok(hash);
        }
        if let Some(hash) = sha256_from_command(Command::new("sha256sum").arg(path))? {
            return Ok(hash);
        }
    }

    anyhow::bail!("no SHA-256 command available to verify release archive");
}

fn verify_release_checksum(archive_path: &Path, checksum_path: &Path) -> anyhow::Result<()> {
    let expected = parse_sha256(&fs::read_to_string(checksum_path).with_context(|| {
        format!(
            "failed to read release checksum file {}",
            checksum_path.display()
        )
    })?)?;
    let actual = compute_sha256(archive_path)?;
    if actual != expected {
        anyhow::bail!(
            "release archive checksum mismatch for {}",
            archive_path.display()
        );
    }
    Ok(())
}

fn latest_release_tag() -> anyhow::Result<String> {
    let temp_dir = UpdateTempDir::create()?;
    let metadata_path = temp_dir.path.join("latest-release.json");
    download_file(
        "https://api.github.com/repos/samvallad33/vestige/releases/latest",
        &metadata_path,
        "checking latest Vestige release",
    )?;
    let file = fs::File::open(&metadata_path)?;
    let metadata: serde_json::Value =
        serde_json::from_reader(file).context("failed to parse latest Vestige release metadata")?;
    metadata
        .get("tag_name")
        .and_then(|tag| tag.as_str())
        .map(|tag| tag.to_string())
        .ok_or_else(|| anyhow::anyhow!("latest Vestige release metadata did not include tag_name"))
}

fn release_tag_for_source(version: Option<&str>) -> anyhow::Result<String> {
    match version {
        Some(version) => Ok(normalize_release_tag(version)),
        None => latest_release_tag(),
    }
}

fn find_sandwich_source_root(root: &Path) -> Option<PathBuf> {
    if root.join("hooks").is_dir() && root.join("agents").is_dir() {
        return Some(root.to_path_buf());
    }

    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("hooks").is_dir() && path.join("agents").is_dir() {
            return Some(path);
        }
    }

    None
}

fn download_sandwich_source(version: Option<&str>, output_dir: &Path) -> anyhow::Result<PathBuf> {
    let tag = release_tag_for_source(version)?;
    let archive_path = output_dir.join(format!("vestige-source-{}.tar.gz", tag));
    let url = source_archive_url(&tag);

    println!("{}: {}", "Sandwich source".white().bold(), tag);
    download_file(&url, &archive_path, "downloading Vestige source archive")?;
    extract_source_archive(&archive_path, output_dir)?;
    find_sandwich_source_root(output_dir).ok_or_else(|| {
        anyhow::anyhow!("Vestige source archive did not contain hooks/ and agents/ directories")
    })
}

fn home_dir() -> anyhow::Result<PathBuf> {
    directories::BaseDirs::new()
        .map(|dirs| dirs.home_dir().to_path_buf())
        .ok_or_else(|| anyhow::anyhow!("failed to locate home directory"))
}

fn is_vestige_hook_command(command: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "synthesis-preflight.sh",
        "cwd-state-injector.sh",
        "vestige-pulse-daemon.sh",
        "preflight-swarm.sh",
        "load-all-memory.sh",
        "veto-detector.sh",
        "sanhedrin.sh",
        "synthesis-stop-validator.sh",
        "synthesis-gate.sh",
    ];
    NEEDLES.iter().any(|needle| command.contains(needle))
}

fn scrub_vestige_hooks(settings: &mut serde_json::Value) {
    let Some(hooks) = settings
        .get_mut("hooks")
        .and_then(|hooks| hooks.as_object_mut())
    else {
        return;
    };

    for event_name in ["UserPromptSubmit", "Stop"] {
        let Some(groups) = hooks
            .get_mut(event_name)
            .and_then(|groups| groups.as_array_mut())
        else {
            continue;
        };

        for group in groups.iter_mut() {
            if let Some(commands) = group
                .get_mut("hooks")
                .and_then(|hooks| hooks.as_array_mut())
            {
                commands.retain(|hook| {
                    !hook
                        .get("command")
                        .and_then(|command| command.as_str())
                        .is_some_and(is_vestige_hook_command)
                });
            }
        }

        groups.retain(|group| {
            group
                .get("hooks")
                .and_then(|hooks| hooks.as_array())
                .is_some_and(|hooks| !hooks.is_empty())
        });
    }

    hooks.retain(|_, value| match value {
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(items) => !items.is_empty(),
        serde_json::Value::Null => false,
        _ => true,
    });

    if hooks.is_empty()
        && let Some(root) = settings.as_object_mut()
    {
        root.remove("hooks");
    }
}

fn merge_json(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base), serde_json::Value::Object(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(existing) => merge_json(existing, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

fn merge_settings_fragment(
    settings: &mut serde_json::Value,
    fragment_path: &Path,
) -> anyhow::Result<()> {
    let file = fs::File::open(fragment_path)
        .with_context(|| format!("failed to open {}", fragment_path.display()))?;
    let fragment: serde_json::Value = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", fragment_path.display()))?;
    merge_json(settings, fragment);
    Ok(())
}

fn copy_companion_files(
    source_dir: &Path,
    destination_dir: &Path,
    allowed_extensions: &[&str],
    _mode: u32,
    options: &SandwichInstallOptions,
) -> anyhow::Result<(usize, usize)> {
    fs::create_dir_all(destination_dir)?;
    let mut copied = 0;
    let mut skipped = 0;

    for entry in fs::read_dir(source_dir)
        .with_context(|| format!("failed to read {}", source_dir.display()))?
    {
        let entry = entry?;
        let source = entry.path();
        if !source.is_file() {
            continue;
        }

        let extension = source
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        if !allowed_extensions.contains(&extension) {
            continue;
        }

        let Some(file_name) = source.file_name() else {
            continue;
        };
        if file_name.to_string_lossy() == "load-all-memory.sh" && !options.include_memory_loader {
            continue;
        }

        let destination = destination_dir.join(file_name);
        if destination.exists() && !options.force {
            skipped += 1;
            continue;
        }

        fs::copy(&source, &destination).with_context(|| {
            format!(
                "failed to copy {} to {}",
                source.display(),
                destination.display()
            )
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&destination)?.permissions();
            perms.set_mode(_mode);
            fs::set_permissions(&destination, perms)?;
        }

        copied += 1;
    }

    Ok((copied, skipped))
}

fn quote_shell_env(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn write_sanhedrin_env(
    hooks_dir: &Path,
    endpoint: &str,
    model: &str,
    dashboard_port: &str,
) -> anyhow::Result<()> {
    let env_path = hooks_dir.join("vestige-sanhedrin.env");
    let contents = format!(
        "VESTIGE_SANHEDRIN_ENABLED=1\nVESTIGE_SANHEDRIN_ENDPOINT={}\nVESTIGE_SANHEDRIN_MODEL={}\nVESTIGE_DASHBOARD_PORT={}\nVESTIGE_SANHEDRIN_CLAIM_MODE=1\nVESTIGE_SANHEDRIN_OUTPUT=json\n",
        quote_shell_env(endpoint),
        quote_shell_env(model),
        quote_shell_env(dashboard_port)
    );
    fs::write(&env_path, contents)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&env_path)?.permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&env_path, perms)?;
    }

    println!("{}: {}", "Sanhedrin env".white().bold(), env_path.display());
    Ok(())
}

fn install_launchd_job(source_root: &Path, home: &Path, model: &str) -> anyhow::Result<()> {
    let launchd_dir = home.join("Library").join("LaunchAgents");
    fs::create_dir_all(&launchd_dir)?;

    let template_path = source_root
        .join("launchd")
        .join("com.vestige.mlx-server.plist.template");
    let template = fs::read_to_string(&template_path)
        .with_context(|| format!("failed to read {}", template_path.display()))?;
    let rendered = template
        .replace("__HOME__", &home.display().to_string())
        .replace("__MODEL__", model);

    let plist = launchd_dir.join("com.vestige.mlx-server.plist");
    fs::write(&plist, rendered)?;
    let _ = Command::new("launchctl").arg("unload").arg(&plist).status();
    run_command(
        Command::new("launchctl").arg("load").arg(&plist),
        "loading Vestige MLX launchd job",
    )?;
    println!("{}: {}", "launchd".white().bold(), plist.display());
    Ok(())
}

fn remove_legacy_launchd_job(home: &Path) {
    if env::consts::OS != "macos" {
        return;
    }

    let plist = home
        .join("Library")
        .join("LaunchAgents")
        .join("com.vestige.mlx-server.plist");
    if plist.exists() {
        let _ = Command::new("launchctl").arg("unload").arg(&plist).status();
        if fs::remove_file(&plist).is_ok() {
            println!(
                "{}: removed old Sanhedrin launchd job",
                "launchd".white().bold()
            );
        }
    }
}

fn install_sandwich_from_source(
    source_root: &Path,
    options: &SandwichInstallOptions,
) -> anyhow::Result<()> {
    let home = home_dir()?;
    let claude_dir = home.join(".claude");
    let hooks_dir = claude_dir.join("hooks");
    let agents_dir = claude_dir.join("agents");
    let settings_path = claude_dir.join("settings.json");
    let source_root =
        find_sandwich_source_root(source_root).unwrap_or_else(|| source_root.to_path_buf());

    if !source_root.join("hooks").is_dir() || !source_root.join("agents").is_dir() {
        anyhow::bail!(
            "Cognitive Sandwich source missing hooks/ or agents/: {}",
            source_root.display()
        );
    }

    let enable_preflight = options.enable_preflight || options.enable_sandwich;
    let mut enable_sanhedrin =
        options.enable_sanhedrin || options.enable_sandwich || options.with_launchd;
    let mut with_launchd = options.with_launchd;

    if with_launchd && (env::consts::OS != "macos" || env::consts::ARCH != "aarch64") {
        println!(
            "{}",
            "--with-launchd is Apple Silicon only; using endpoint-backed Sanhedrin instead."
                .yellow()
        );
        with_launchd = false;
        enable_sanhedrin = true;
    }

    fs::create_dir_all(&claude_dir)?;
    let (hooks_copied, hooks_skipped) = copy_companion_files(
        &source_root.join("hooks"),
        &hooks_dir,
        &["sh", "py"],
        0o755,
        options,
    )?;
    let (json_copied, json_skipped) = copy_companion_files(
        &source_root.join("hooks"),
        &hooks_dir,
        &["json"],
        0o644,
        options,
    )?;
    let (agents_copied, agents_skipped) = copy_companion_files(
        &source_root.join("agents"),
        &agents_dir,
        &["md"],
        0o644,
        options,
    )?;

    println!(
        "{}: {} installed, {} skipped",
        "Hooks".white().bold(),
        hooks_copied + json_copied,
        hooks_skipped + json_skipped
    );
    println!(
        "{}: {} installed, {} skipped",
        "Agents".white().bold(),
        agents_copied,
        agents_skipped
    );

    if !with_launchd {
        remove_legacy_launchd_job(&home);
    }

    let dashboard_port = env::var("VESTIGE_DASHBOARD_PORT").unwrap_or_else(|_| "3927".to_string());
    let mut endpoint = options
        .sanhedrin_endpoint
        .clone()
        .or_else(|| env::var("VESTIGE_SANHEDRIN_ENDPOINT").ok())
        .or_else(|| env::var("MLX_ENDPOINT").ok())
        .unwrap_or_default()
        .trim_end_matches('/')
        .to_string();
    let mut model = options
        .sanhedrin_model
        .clone()
        .or_else(|| env::var("VESTIGE_SANHEDRIN_MODEL").ok())
        .or_else(|| env::var("VESTIGE_SANDWICH_MODEL").ok())
        .unwrap_or_default();
    if with_launchd {
        if endpoint.is_empty() {
            endpoint = "http://127.0.0.1:8080/v1/chat/completions".to_string();
        }
        if model.is_empty() {
            model = "mlx-community/Qwen3.6-35B-A3B-4bit".to_string();
        }
    }

    if enable_sanhedrin {
        if endpoint.is_empty() || model.is_empty() {
            println!(
                "{}",
                "Sanhedrin enabled without a verifier model; it will fail open until VESTIGE_SANHEDRIN_ENDPOINT and VESTIGE_SANHEDRIN_MODEL are set."
                    .yellow()
            );
        }
        write_sanhedrin_env(&hooks_dir, &endpoint, &model, &dashboard_port)?;
    }
    if with_launchd {
        install_launchd_job(&source_root, &home, &model)?;
    }

    if !settings_path.exists() {
        fs::write(&settings_path, "{}\n")?;
    }
    let backup_path = claude_dir.join("settings.json.bak.pre-sandwich");
    if !backup_path.exists() {
        fs::copy(&settings_path, &backup_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&backup_path)?.permissions();
            perms.set_mode(0o600);
            fs::set_permissions(&backup_path, perms)?;
        }
    }

    let settings_file = fs::File::open(&settings_path)?;
    let mut settings: serde_json::Value =
        serde_json::from_reader(settings_file).unwrap_or_else(|_| serde_json::json!({}));
    scrub_vestige_hooks(&mut settings);

    if enable_preflight {
        merge_settings_fragment(
            &mut settings,
            &source_root
                .join("hooks")
                .join("settings.preflight.fragment.json"),
        )?;
    }
    if enable_sanhedrin {
        merge_settings_fragment(
            &mut settings,
            &source_root
                .join("hooks")
                .join("settings.sanhedrin.fragment.json"),
        )?;
    }

    let mut settings_file = fs::File::create(&settings_path)?;
    serde_json::to_writer_pretty(&mut settings_file, &settings)?;
    writeln!(settings_file)?;

    if enable_preflight || enable_sanhedrin {
        let mut layers = Vec::new();
        if enable_preflight {
            layers.push("preflight");
        }
        if enable_sanhedrin {
            layers.push("sanhedrin");
        }
        println!(
            "{}: enabled optional layer(s): {}",
            "Settings".white().bold(),
            layers.join(", ")
        );
    } else {
        println!(
            "{}: no Vestige Claude Code hooks enabled by default",
            "Settings".white().bold()
        );
    }

    Ok(())
}

fn run_sandwich_install(
    version: Option<&str>,
    options: &SandwichInstallOptions,
) -> anyhow::Result<()> {
    println!(
        "{}",
        "=== Vestige Cognitive Sandwich Install ===".cyan().bold()
    );
    println!();

    if let Some(source_root) = &options.src {
        install_sandwich_from_source(source_root, options)?;
    } else {
        let temp_dir = UpdateTempDir::create()?;
        let source_root = download_sandwich_source(version, &temp_dir.path)?;
        install_sandwich_from_source(&source_root, options)?;
    }

    println!();
    let optional_layers_enabled = options.enable_preflight
        || options.enable_sandwich
        || options.enable_sanhedrin
        || options.with_launchd;
    let message = if optional_layers_enabled {
        "Cognitive Sandwich files updated. Restart Claude Code to use enabled optional hooks."
    } else {
        "Cognitive Sandwich files updated. No hooks enabled; no automatic model calls."
    };
    println!("{}", message.green().bold());
    Ok(())
}

fn run_command(command: &mut Command, action: &str) -> anyhow::Result<()> {
    let status = command
        .status()
        .with_context(|| format!("failed to start {}", action))?;
    if !status.success() {
        anyhow::bail!("{} failed with status {}", action, status);
    }
    Ok(())
}

fn create_private_file(path: &Path) -> std::io::Result<fs::File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
    }

    #[cfg(not(unix))]
    {
        fs::File::create(path)
    }
}

fn command_output(command: &mut Command, action: &str) -> anyhow::Result<String> {
    let output = command
        .output()
        .with_context(|| format!("failed to start {}", action))?;
    if !output.status.success() {
        anyhow::bail!("{} failed with status {}", action, output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn powershell_quote(value: &Path) -> String {
    format!("'{}'", value.display().to_string().replace('\'', "''"))
}

fn normalize_archive_entry(entry: &str) -> anyhow::Result<String> {
    let normalized = entry.trim().replace('\\', "/");
    let normalized = normalized.strip_prefix("./").unwrap_or(&normalized);
    if normalized.is_empty()
        || normalized.starts_with('/')
        || normalized.get(1..2) == Some(":")
        || normalized
            .split('/')
            .any(|part| part.is_empty() || part == "..")
    {
        anyhow::bail!("archive contains unsafe entry: {}", entry);
    }
    Ok(normalized.to_string())
}

fn archive_listing(archive_path: &Path, archive_ext: &str) -> anyhow::Result<String> {
    let listing = match archive_ext {
        "tar.gz" => command_output(
            Command::new("tar").arg("-tzf").arg(archive_path),
            "listing Vestige archive with tar",
        )?,
        "zip" => {
            let script = format!(
                "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                 $zip = [System.IO.Compression.ZipFile]::OpenRead({}); \
                 try {{ $zip.Entries | ForEach-Object {{ $_.FullName }} }} finally {{ $zip.Dispose() }}",
                powershell_quote(archive_path)
            );
            command_output(
                Command::new("powershell")
                    .arg("-NoProfile")
                    .arg("-Command")
                    .arg(script),
                "listing Vestige archive with PowerShell",
            )?
        }
        other => anyhow::bail!("unsupported release archive extension: {}", other),
    };
    Ok(listing)
}

fn validate_archive_safety(archive_path: &Path, archive_ext: &str) -> anyhow::Result<()> {
    let listing = archive_listing(archive_path, archive_ext)?;
    for entry in listing.lines().filter(|line| !line.trim().is_empty()) {
        normalize_archive_entry(entry)?;
    }
    Ok(())
}

fn validate_archive_entries(
    archive_path: &Path,
    archive_ext: &str,
    expected_members: &[String],
) -> anyhow::Result<()> {
    let listing = archive_listing(archive_path, archive_ext)?;

    let expected: HashSet<&str> = expected_members.iter().map(String::as_str).collect();
    for entry in listing.lines().filter(|line| !line.trim().is_empty()) {
        let normalized = normalize_archive_entry(entry)?;
        if !expected.contains(normalized.as_str()) {
            anyhow::bail!("release archive contains unexpected entry: {}", entry);
        }
    }
    Ok(())
}

fn extract_source_archive(archive_path: &Path, output_dir: &Path) -> anyhow::Result<()> {
    validate_archive_safety(archive_path, "tar.gz")?;
    run_command(
        Command::new("tar")
            .arg("-xzf")
            .arg(archive_path)
            .arg("-C")
            .arg(output_dir),
        "extracting Vestige source archive with tar",
    )
}

fn extract_archive(
    archive_path: &Path,
    output_dir: &Path,
    archive_ext: &str,
    expected_members: &[String],
) -> anyhow::Result<()> {
    validate_archive_entries(archive_path, archive_ext, expected_members)?;
    match archive_ext {
        "tar.gz" => run_command(
            Command::new("tar")
                .arg("-xzf")
                .arg(archive_path)
                .arg("-C")
                .arg(output_dir),
            "extracting Vestige release archive with tar",
        ),
        "zip" => run_command(
            Command::new("powershell")
                .arg("-NoProfile")
                .arg("-Command")
                .arg(format!(
                    "Expand-Archive -LiteralPath {} -DestinationPath {} -Force",
                    powershell_quote(archive_path),
                    powershell_quote(output_dir)
                )),
            "extracting Vestige release archive with PowerShell",
        ),
        other => anyhow::bail!("unsupported release archive extension: {}", other),
    }
}

fn replace_binary(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid destination path {}", destination.display()))?;
    let temp_destination = destination.with_file_name(format!(
        ".{}.vestige-update-{}",
        file_name,
        std::process::id()
    ));

    fs::copy(source, &temp_destination).with_context(|| {
        format!(
            "failed to stage {} for install at {}",
            source.display(),
            temp_destination.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&temp_destination)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&temp_destination, perms)?;
    }

    #[cfg(windows)]
    if destination.exists() {
        fs::remove_file(destination).with_context(|| {
            format!(
                "failed to replace {}. Close running Vestige processes and retry",
                destination.display()
            )
        })?;
    }

    fs::rename(&temp_destination, destination).with_context(|| {
        let _ = fs::remove_file(&temp_destination);
        format!(
            "failed to install {}. If this is a system directory, retry with: sudo vestige update",
            destination.display()
        )
    })?;

    Ok(())
}

fn run_update(
    version: Option<String>,
    install_dir: Option<PathBuf>,
    dry_run: bool,
    no_sandwich: bool,
    sandwich_companion: bool,
    sandwich: SandwichInstallOptions,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Update ===".cyan().bold());
    println!();

    let asset = current_release_asset()?;
    let current_exe = env::current_exe().context("failed to locate current vestige executable")?;
    let install_dir = match install_dir {
        Some(path) => path,
        None => current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("current executable has no parent directory"))?
            .to_path_buf(),
    };

    let url = release_download_url(asset, version.as_deref());
    let archive_name = format!("vestige-mcp-{}.{}", asset.target, asset.archive_ext);

    println!(
        "{}: {}",
        "Current version".white().bold(),
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "{}: {}",
        "Release".white().bold(),
        version.as_deref().unwrap_or("latest")
    );
    println!("{}: {}", "Target".white().bold(), asset.target);
    println!(
        "{}: {}",
        "Install dir".white().bold(),
        install_dir.display()
    );
    println!("{}: {}", "Download".white().bold(), url);

    if dry_run {
        println!();
        println!("{}", "Dry run: no files changed.".yellow().bold());
        return Ok(());
    }

    fs::create_dir_all(&install_dir).with_context(|| {
        format!(
            "failed to create install directory {}",
            install_dir.display()
        )
    })?;

    let temp_dir = UpdateTempDir::create()?;
    let archive_path = temp_dir.path.join(&archive_name);
    let checksum_path = temp_dir.path.join(format!("{}.sha256", archive_name));

    println!();
    println!("{}", "Downloading release archive...".cyan());
    download_file(&url, &archive_path, "downloading Vestige release archive")?;
    download_file(
        &format!("{}.sha256", url),
        &checksum_path,
        "downloading Vestige release checksum",
    )?;
    verify_release_checksum(&archive_path, &checksum_path)?;

    let binaries = ["vestige", "vestige-mcp", "vestige-restore"];
    let mut expected_members = binaries
        .iter()
        .map(|binary| format!("{}{}", binary, asset.binary_suffix))
        .collect::<Vec<_>>();
    if asset.target == "x86_64-apple-darwin" {
        expected_members.push("INSTALL-INTEL-MAC.md".to_string());
    }

    println!("{}", "Extracting release archive...".cyan());
    extract_archive(
        &archive_path,
        &temp_dir.path,
        asset.archive_ext,
        &expected_members,
    )?;

    for binary in binaries {
        let filename = format!("{}{}", binary, asset.binary_suffix);
        let source = temp_dir.path.join(&filename);
        if !source.exists() {
            anyhow::bail!("release archive is missing expected binary: {}", filename);
        }

        let destination = install_dir.join(&filename);
        println!("  {} {}", "install".dimmed(), destination.display());
        replace_binary(&source, &destination)?;
    }

    println!();
    let installed_mcp = install_dir.join(format!("vestige-mcp{}", asset.binary_suffix));
    if let Ok(output) = Command::new(&installed_mcp).arg("--version").output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !version.is_empty() {
            println!("{}: {}", "Installed".white().bold(), version.green());
        }
    }

    println!(
        "{}",
        "Binary update complete. Restart your MCP client to pick up the new binary."
            .green()
            .bold()
    );

    if sandwich_companion && !no_sandwich {
        println!();
        println!(
            "{}",
            "Updating Cognitive Sandwich companion files...".cyan()
        );
        run_sandwich_install(version.as_deref(), &sandwich)?;
    } else if no_sandwich {
        println!(
            "{}",
            "Skipped Cognitive Sandwich companion update (--no-sandwich).".yellow()
        );
    } else {
        println!(
            "{}",
            "Skipped Cognitive Sandwich companion update (default). Pass --sandwich-companion to refresh Claude Code companion files."
                .yellow()
        );
    }

    Ok(())
}

/// Run stats command
fn run_stats(show_tagging: bool, show_states: bool) -> anyhow::Result<()> {
    let storage = open_storage()?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Memory Statistics ===".cyan().bold());
    println!();

    // Basic stats
    println!("{}: {}", "Total Memories".white().bold(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white().bold(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white().bold(),
        stats.average_retention * 100.0
    );
    println!(
        "{}: {:.2}",
        "Average Storage Strength".white().bold(),
        stats.average_storage_strength
    );
    println!(
        "{}: {:.2}",
        "Average Retrieval Strength".white().bold(),
        stats.average_retrieval_strength
    );
    println!(
        "{}: {}",
        "With Embeddings".white().bold(),
        stats.nodes_with_embeddings
    );

    if let Some(model) = &stats.active_embedding_model {
        println!("{}: {}", "Active Embedding Model".white().bold(), model);
    }

    if let Some(model) = &stats.embedding_model {
        println!("{}: {}", "Stored Embedding Model".white().bold(), model);
    }

    if stats.nodes_with_mismatched_embeddings > 0 {
        println!(
            "{}: {}",
            "Mismatched Embeddings".white().bold(),
            stats.nodes_with_mismatched_embeddings
        );
    }

    if let Some(oldest) = stats.oldest_memory {
        println!(
            "{}: {}",
            "Oldest Memory".white().bold(),
            oldest.format("%Y-%m-%d %H:%M:%S")
        );
    }
    if let Some(newest) = stats.newest_memory {
        println!(
            "{}: {}",
            "Newest Memory".white().bold(),
            newest.format("%Y-%m-%d %H:%M:%S")
        );
    }

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{}: {:.1}%",
        "Active Embedding Coverage".white().bold(),
        embedding_coverage
    );

    // Tagging distribution (retention levels)
    if show_tagging {
        println!();
        println!("{}", "=== Retention Distribution ===".yellow().bold());

        let memories = storage.get_all_nodes(500, 0)?;
        let total = memories.len();

        if total > 0 {
            let high = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.7)
                .count();
            let medium = memories
                .iter()
                .filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7)
                .count();
            let low = memories
                .iter()
                .filter(|m| m.retention_strength < 0.4)
                .count();

            print_distribution_bar("High (>=70%)", high, total, "green");
            print_distribution_bar("Medium (40-70%)", medium, total, "yellow");
            print_distribution_bar("Low (<40%)", low, total, "red");
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    // State distribution
    if show_states {
        println!();
        println!(
            "{}",
            "=== Cognitive State Distribution ===".magenta().bold()
        );

        let memories = storage.get_all_nodes(500, 0)?;
        let total = memories.len();

        if total > 0 {
            let (active, dormant, silent, unavailable) = compute_state_distribution(&memories);

            print_distribution_bar("Active", active, total, "green");
            print_distribution_bar("Dormant", dormant, total, "yellow");
            print_distribution_bar("Silent", silent, total, "red");
            print_distribution_bar("Unavailable", unavailable, total, "magenta");

            println!();
            println!("{}", "State Thresholds:".dimmed());
            println!("  {} >= 0.70 accessibility", "Active".green());
            println!("  {} >= 0.40 accessibility", "Dormant".yellow());
            println!("  {} >= 0.10 accessibility", "Silent".red());
            println!("  {} < 0.10 accessibility", "Unavailable".magenta());
        } else {
            println!("{}", "No memories found.".dimmed());
        }
    }

    Ok(())
}

/// Compute cognitive state distribution for memories
fn compute_state_distribution(
    memories: &[vestige_core::KnowledgeNode],
) -> (usize, usize, usize, usize) {
    let mut active = 0;
    let mut dormant = 0;
    let mut silent = 0;
    let mut unavailable = 0;

    for memory in memories {
        // Accessibility = 0.5*retention + 0.3*retrieval + 0.2*storage
        let accessibility = memory.retention_strength * 0.5
            + memory.retrieval_strength * 0.3
            + memory.storage_strength * 0.2;

        if accessibility >= 0.7 {
            active += 1;
        } else if accessibility >= 0.4 {
            dormant += 1;
        } else if accessibility >= 0.1 {
            silent += 1;
        } else {
            unavailable += 1;
        }
    }

    (active, dormant, silent, unavailable)
}

/// Print a distribution bar
fn print_distribution_bar(label: &str, count: usize, total: usize, color: &str) {
    let percentage = if total > 0 {
        (count as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let bar_width: usize = 30;
    let filled = ((percentage / 100.0) * bar_width as f64) as usize;
    let empty = bar_width.saturating_sub(filled);

    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(empty));
    let colored_bar = match color {
        "green" => bar.green(),
        "yellow" => bar.yellow(),
        "red" => bar.red(),
        "magenta" => bar.magenta(),
        _ => bar.white(),
    };

    println!(
        "  {:15} [{:30}] {:>4} ({:>5.1}%)",
        label, colored_bar, count, percentage
    );
}

/// Run health check
fn run_health() -> anyhow::Result<()> {
    let storage = open_storage()?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Health Check ===".cyan().bold());
    println!();

    // Determine health status
    let (status, status_color) = if stats.total_nodes == 0 {
        ("EMPTY", "white")
    } else if stats.average_retention < 0.3 {
        ("CRITICAL", "red")
    } else if stats.average_retention < 0.5 {
        ("DEGRADED", "yellow")
    } else {
        ("HEALTHY", "green")
    };

    let colored_status = match status_color {
        "green" => status.green().bold(),
        "yellow" => status.yellow().bold(),
        "red" => status.red().bold(),
        _ => status.white().bold(),
    };

    println!("{}: {}", "Status".white().bold(), colored_status);
    println!("{}: {}", "Total Memories".white(), stats.total_nodes);
    println!(
        "{}: {}",
        "Due for Review".white(),
        stats.nodes_due_for_review
    );
    println!(
        "{}: {:.1}%",
        "Average Retention".white(),
        stats.average_retention * 100.0
    );

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_active_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "{}: {:.1}%",
        "Active Embedding Coverage".white(),
        embedding_coverage
    );
    println!(
        "{}: {}",
        "Embedding Service".white(),
        if storage.is_embedding_ready() {
            "Ready".green()
        } else {
            "Not Ready".red()
        }
    );

    // Warnings
    let mut warnings = Vec::new();

    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings
            .push("Low average retention - consider running consolidation or reviewing memories");
    }

    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }

    if stats.total_nodes > 0 && stats.nodes_with_active_embeddings == 0 {
        warnings.push("No active-model embeddings generated - semantic search unavailable");
    }

    if embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push("Low embedding coverage - run consolidation to improve semantic search");
    }

    if stats.nodes_with_mismatched_embeddings > 0 {
        warnings.push("Stored embeddings from another model are present - run consolidation after changing embedding models");
    }

    if !warnings.is_empty() {
        println!();
        println!("{}", "Warnings:".yellow().bold());
        for warning in &warnings {
            println!("  {} {}", "!".yellow().bold(), warning.yellow());
        }
    }

    // Recommendations
    let mut recommendations = Vec::new();

    if status == "CRITICAL" {
        recommendations
            .push("CRITICAL: Many memories have very low retention. Review important memories.");
    }

    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }

    if stats.nodes_with_active_embeddings < stats.total_nodes {
        recommendations
            .push("Run 'vestige consolidate' to generate active-model embeddings for better semantic search.");
    }

    if stats.total_nodes > 100 && stats.average_retention < 0.7 {
        recommendations.push("Consider running periodic consolidation to maintain memory health.");
    }

    if recommendations.is_empty() && status == "HEALTHY" {
        recommendations.push("Memory system is healthy!");
    }

    println!();
    println!("{}", "Recommendations:".cyan().bold());
    for rec in &recommendations {
        let icon = if rec.starts_with("CRITICAL") {
            "!".red().bold()
        } else {
            ">".cyan()
        };
        let text = if rec.starts_with("CRITICAL") {
            rec.red().to_string()
        } else {
            rec.to_string()
        };
        println!("  {} {}", icon, text);
    }

    Ok(())
}

/// Run consolidation cycle
fn run_consolidate() -> anyhow::Result<()> {
    println!("{}", "=== Vestige Consolidation ===".cyan().bold());
    println!();
    println!("Running memory consolidation cycle...");
    println!();

    let storage = open_storage()?;
    let result = storage.run_consolidation()?;

    println!(
        "{}: {}",
        "Nodes Processed".white().bold(),
        result.nodes_processed
    );
    println!(
        "{}: {}",
        "Nodes Promoted".white().bold(),
        result.nodes_promoted
    );
    println!("{}: {}", "Nodes Pruned".white().bold(), result.nodes_pruned);
    println!(
        "{}: {}",
        "Decay Applied".white().bold(),
        result.decay_applied
    );
    println!(
        "{}: {}",
        "Embeddings Generated".white().bold(),
        result.embeddings_generated
    );
    println!("{}: {}ms", "Duration".white().bold(), result.duration_ms);

    println!();
    println!(
        "{}",
        format!(
            "Consolidation complete: {} nodes processed, {} embeddings generated in {}ms",
            result.nodes_processed, result.embeddings_generated, result.duration_ms
        )
        .green()
    );

    Ok(())
}

/// Run restore from backup
fn run_restore(backup_path: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Restore ===".cyan().bold());
    println!();
    println!("Loading backup from: {}", backup_path.display());

    // Read and parse backup
    let backup_bytes = std::fs::read(&backup_path)?;
    if backup_bytes.starts_with(b"SQLite format 3\0") {
        anyhow::bail!(
            "{} is a raw SQLite database backup, not a JSON restore file. Use portable-export/portable-import for cross-device transfer, or replace the database file manually while Vestige is stopped.",
            backup_path.display()
        );
    }
    let backup_content = String::from_utf8(backup_bytes)
        .with_context(|| format!("{} is not UTF-8 JSON", backup_path.display()))?;

    if let Ok(archive) = serde_json::from_str::<vestige_core::PortableArchive>(&backup_content)
        && archive.archive_format == vestige_core::PORTABLE_ARCHIVE_FORMAT
    {
        println!("Detected portable archive.");
        println!("{}: {}", "Format".white().bold(), archive.archive_format);
        println!("{}: {}", "Schema".white().bold(), archive.schema_version);
        println!("{}: {}", "Tables".white().bold(), archive.tables.len());
        println!("{}: {}", "Rows".white().bold(), archive.total_rows());
        println!();

        let storage = open_storage()?;
        let report = storage.import_portable_archive(&archive, PortableImportMode::EmptyOnly)?;

        println!(
            "{}: {}",
            "Tables imported".white().bold(),
            report.tables_imported
        );
        println!(
            "{}: {}",
            "Rows imported".white().bold(),
            report.rows_imported
        );
        println!(
            "{}: {}",
            "Tables skipped".white().bold(),
            report.tables_skipped
        );
        println!("{}: {}", "FTS rebuilt".white().bold(), report.fts_rebuilt);
        println!();
        println!("{}", "Portable restore complete.".green().bold());
        return Ok(());
    }

    #[derive(serde::Deserialize)]
    struct BackupWrapper {
        #[serde(rename = "type")]
        _type: String,
        text: String,
    }

    #[derive(serde::Deserialize)]
    struct RecallResult {
        results: Vec<MemoryBackup>,
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct MemoryBackup {
        content: String,
        node_type: Option<String>,
        tags: Option<Vec<String>>,
        source: Option<String>,
    }

    let memories = if let Ok(wrapper) = serde_json::from_str::<Vec<BackupWrapper>>(&backup_content)
    {
        let first = wrapper.first().context("backup wrapper is empty")?;
        let recall_result: RecallResult = serde_json::from_str(&first.text)?;
        recall_result.results
    } else if let Ok(recall_result) = serde_json::from_str::<RecallResult>(&backup_content) {
        recall_result.results
    } else if let Ok(memories) = serde_json::from_str::<Vec<MemoryBackup>>(&backup_content) {
        memories
    } else {
        anyhow::bail!(
            "Unrecognized backup format. Expected portable archive, MCP wrapper, RecallResult, or array of memories."
        );
    };

    println!("Found {} memories to restore", memories.len());
    println!();

    // Initialize storage
    println!("Initializing storage...");
    let storage = open_storage()?;

    println!("Generating embeddings and ingesting memories...");
    println!();

    let total = memories.len();
    let mut success_count = 0;

    for (i, memory) in memories.into_iter().enumerate() {
        let input = IngestInput {
            content: memory.content.clone(),
            node_type: memory.node_type.unwrap_or_else(|| "fact".to_string()),
            source: memory.source,
            sentiment_score: 0.0,
            sentiment_magnitude: 0.0,
            tags: memory.tags.unwrap_or_default(),
            valid_from: None,
            valid_until: None,
            source_envelope: None,
        };

        match storage.ingest(input) {
            Ok(_node) => {
                success_count += 1;
                println!(
                    "[{}/{}] {} {}",
                    i + 1,
                    total,
                    "OK".green(),
                    truncate(&memory.content, 60)
                );
            }
            Err(e) => {
                println!("[{}/{}] {} {}", i + 1, total, "FAIL".red(), e);
            }
        }
    }

    println!();
    println!(
        "Restore complete: {}/{} memories restored",
        success_count.to_string().green().bold(),
        total
    );

    // Show stats
    let stats = storage.get_stats()?;
    println!();
    println!("{}: {}", "Total Nodes".white(), stats.total_nodes);
    println!(
        "{}: {}",
        "Active Embeddings".white(),
        stats.nodes_with_active_embeddings
    );

    Ok(())
}

/// Get the default database path
fn get_default_db_path() -> anyhow::Result<PathBuf> {
    if let Some(path) = CLI_DB_PATH.get() {
        Ok(path.clone())
    } else {
        Ok(Storage::default_db_path()?)
    }
}

/// Open storage using the CLI-selected data directory, if one was provided.
fn open_storage() -> anyhow::Result<Storage> {
    if let Some(path) = CLI_DB_PATH.get() {
        Ok(Storage::new(Some(path.clone()))?)
    } else {
        Ok(Storage::new(None)?)
    }
}

/// Fetch all nodes from storage using pagination
fn fetch_all_nodes(storage: &Storage) -> anyhow::Result<Vec<vestige_core::KnowledgeNode>> {
    let mut all_nodes = Vec::new();
    let page_size = 500;
    let mut offset = 0;

    loop {
        let batch = storage.get_all_nodes(page_size, offset)?;
        let batch_len = batch.len();
        all_nodes.extend(batch);
        if batch_len < page_size as usize {
            break;
        }
        offset += page_size;
    }

    Ok(all_nodes)
}

/// Run backup command - copies the SQLite database file
fn run_backup(output: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Backup ===".cyan().bold());
    println!();

    let db_path = get_default_db_path()?;

    if !db_path.exists() {
        anyhow::bail!("Database not found at: {}", db_path.display());
    }

    // Open storage to flush WAL before copying
    println!("Flushing WAL checkpoint...");
    {
        let storage = open_storage()?;
        // get_stats triggers a read so the connection is active, then drop flushes
        let _ = storage.get_stats()?;
    }

    // Also flush WAL directly via a separate connection for safety
    {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    }

    // Create parent directories if needed
    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    // Copy the database file
    println!("Copying database...");
    println!("  {} {}", "From:".dimmed(), db_path.display());
    println!("  {}   {}", "To:".dimmed(), output.display());

    std::fs::copy(&db_path, &output)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&output)?.permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&output, perms)?;
    }

    let file_size = std::fs::metadata(&output)?.len();
    let size_display = if file_size >= 1024 * 1024 {
        format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
    } else if file_size >= 1024 {
        format!("{:.1} KB", file_size as f64 / 1024.0)
    } else {
        format!("{} bytes", file_size)
    };

    println!();
    println!(
        "{}",
        format!("Backup complete: {} ({})", output.display(), size_display)
            .green()
            .bold()
    );

    Ok(())
}

/// Run export command - exports memories in JSON or JSONL format
fn run_export(
    output: PathBuf,
    format: String,
    tags: Option<String>,
    since: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Export ===".cyan().bold());
    println!();

    // Validate format
    if format != "json" && format != "jsonl" {
        anyhow::bail!("Invalid format '{}'. Must be 'json' or 'jsonl'.", format);
    }

    // Parse since date if provided
    let since_date = match &since {
        Some(date_str) => {
            let naive = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|e| {
                anyhow::anyhow!("Invalid date '{}': {}. Use YYYY-MM-DD format.", date_str, e)
            })?;
            Some(
                naive
                    .and_hms_opt(0, 0, 0)
                    .expect("midnight is always valid")
                    .and_utc(),
            )
        }
        None => None,
    };

    // Parse tags filter
    let tag_filter: Vec<String> = tags
        .as_deref()
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let storage = open_storage()?;
    let all_nodes = fetch_all_nodes(&storage)?;

    // Apply filters
    let filtered: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            // Date filter
            if let Some(ref since_dt) = since_date
                && node.created_at < *since_dt
            {
                return false;
            }
            // Tag filter: node must contain ALL specified tags
            if !tag_filter.is_empty() {
                for tag in &tag_filter {
                    if !node.tags.iter().any(|t| t == tag) {
                        return false;
                    }
                }
            }
            true
        })
        .collect();

    println!("{}: {}", "Format".white().bold(), format);
    if !tag_filter.is_empty() {
        println!("{}: {}", "Tag filter".white().bold(), tag_filter.join(", "));
    }
    if let Some(ref date_str) = since {
        println!("{}: {}", "Since".white().bold(), date_str);
    }
    println!(
        "{}: {} / {} total",
        "Matching".white().bold(),
        filtered.len(),
        all_nodes.len()
    );
    println!();

    // Create parent directories if needed
    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = create_private_file(&output)?;
    let mut writer = BufWriter::new(file);

    match format.as_str() {
        "json" => {
            serde_json::to_writer_pretty(&mut writer, &filtered)?;
            writer.write_all(b"\n")?;
        }
        "jsonl" => {
            for node in &filtered {
                serde_json::to_writer(&mut writer, node)?;
                writer.write_all(b"\n")?;
            }
        }
        _ => unreachable!(),
    }

    writer.flush()?;

    let file_size = std::fs::metadata(&output)?.len();
    let size_display = if file_size >= 1024 * 1024 {
        format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
    } else if file_size >= 1024 {
        format!("{:.1} KB", file_size as f64 / 1024.0)
    } else {
        format!("{} bytes", file_size)
    };

    println!(
        "{}",
        format!(
            "Exported {} memories to {} ({}, {})",
            filtered.len(),
            output.display(),
            format,
            size_display
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Run exact portable archive export.
fn run_portable_export(output: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Portable Export ===".cyan().bold());
    println!();

    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let storage = open_storage()?;
    let archive = storage.export_portable_archive_to_path(&output)?;

    let file_size = std::fs::metadata(&output)?.len();
    let size_display = if file_size >= 1024 * 1024 {
        format!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0))
    } else if file_size >= 1024 {
        format!("{:.1} KB", file_size as f64 / 1024.0)
    } else {
        format!("{} bytes", file_size)
    };

    println!("{}: {}", "Archive".white().bold(), output.display());
    println!("{}: {}", "Format".white().bold(), archive.archive_format);
    println!("{}: {}", "Schema".white().bold(), archive.schema_version);
    println!("{}: {}", "Tables".white().bold(), archive.tables.len());
    println!("{}: {}", "Rows".white().bold(), archive.total_rows());
    println!();
    println!(
        "{}",
        format!(
            "Portable export complete: {} ({})",
            output.display(),
            size_display
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Run exact portable archive import.
fn run_portable_import(input: PathBuf, merge: bool) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Portable Import ===".cyan().bold());
    println!();
    println!("{}: {}", "Archive".white().bold(), input.display());
    let mode = if merge {
        PortableImportMode::Merge
    } else {
        PortableImportMode::EmptyOnly
    };
    println!(
        "{}",
        if merge {
            "Mode: merge into existing database".yellow()
        } else {
            "Mode: empty database only".yellow()
        }
    );
    println!();

    let storage = open_storage()?;
    let report = storage.import_portable_archive_from_path(&input, mode)?;

    println!(
        "{}: {}",
        "Tables imported".white().bold(),
        report.tables_imported
    );
    println!(
        "{}: {}",
        "Rows imported".white().bold(),
        report.rows_imported
    );
    println!(
        "{}: {}",
        "Tables skipped".white().bold(),
        report.tables_skipped
    );
    println!("{}: {}", "FTS rebuilt".white().bold(), report.fts_rebuilt);
    if merge {
        println!(
            "{}: {} inserted, {} updated, {} deleted, {} skipped, {} kept local",
            "Merge".white().bold(),
            report.rows_inserted,
            report.rows_updated,
            report.rows_deleted,
            report.rows_skipped,
            report.conflicts_kept_local
        );
    }
    println!();
    println!("{}", "Portable import complete.".green().bold());

    Ok(())
}

/// Run file-backed two-way sync.
fn run_sync(
    archive: Option<PathBuf>,
    cloud: bool,
    endpoint: Option<String>,
) -> anyhow::Result<()> {
    if cloud {
        run_sync_cloud(endpoint)
    } else {
        let archive = archive.ok_or_else(|| {
            anyhow::anyhow!(
                "no sync target: pass an archive path for file sync, or --cloud for Vestige Cloud"
            )
        })?;
        run_sync_file(archive)
    }
}

fn run_sync_file(archive: PathBuf) -> anyhow::Result<()> {
    println!("{}", "=== Vestige File Sync ===".cyan().bold());
    println!();
    println!("{}: {}", "Archive".white().bold(), archive.display());

    let storage = open_storage()?;
    let report = storage.sync_portable_archive_file(&archive)?;
    print_sync_report(&report);
    Ok(())
}

#[cfg(feature = "cloud-sync")]
fn run_sync_cloud(endpoint: Option<String>) -> anyhow::Result<()> {
    let endpoint = endpoint
        .or_else(|| std::env::var("VESTIGE_CLOUD_ENDPOINT").ok())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no cloud endpoint: pass --endpoint or set VESTIGE_CLOUD_ENDPOINT \
                 (e.g. https://sync.vestige.dev)"
            )
        })?;
    let sync_key = std::env::var("VESTIGE_CLOUD_SYNC_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no sync key: set VESTIGE_CLOUD_SYNC_KEY (issued when you subscribe to \
                 Vestige Cloud)"
            )
        })?;

    // Optional zero-knowledge encryption passphrase. Never sent to the server.
    let encryption_key = std::env::var("VESTIGE_CLOUD_ENCRYPTION_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty());

    println!("{}", "=== Vestige Cloud Sync ===".cyan().bold());
    println!();
    println!("{}: {}", "Endpoint".white().bold(), endpoint);
    if encryption_key.is_some() {
        println!(
            "{}: {}",
            "Encryption".white().bold(),
            "zero-knowledge (XChaCha20-Poly1305) — your data is encrypted before upload".green()
        );
    } else {
        println!(
            "{}: {}",
            "Encryption".white().bold(),
            "OFF — set VESTIGE_CLOUD_ENCRYPTION_KEY for zero-knowledge sync".yellow()
        );
    }

    let storage = open_storage()?;
    let report = storage.sync_portable_archive_cloud(&endpoint, &sync_key, encryption_key)?;
    print_sync_report(&report);
    Ok(())
}

#[cfg(not(feature = "cloud-sync"))]
fn run_sync_cloud(_endpoint: Option<String>) -> anyhow::Result<()> {
    anyhow::bail!(
        "this build was compiled without the `cloud-sync` feature; rebuild with \
         --features cloud-sync to use Vestige Cloud"
    )
}

fn print_sync_report(report: &vestige_core::PortableSyncReport) {
    if let Some(pull) = &report.pull {
        println!("{}", "Pull: merged remote archive".yellow());
        println!(
            "  {} inserted, {} updated, {} deleted, {} skipped, {} kept local",
            pull.rows_inserted,
            pull.rows_updated,
            pull.rows_deleted,
            pull.rows_skipped,
            pull.conflicts_kept_local
        );
    } else {
        println!(
            "{}",
            "Pull: archive does not exist yet; creating it".yellow()
        );
    }

    println!("{}", "Push: wrote merged local state".yellow());
    println!(
        "{}",
        format!(
            "Sync complete: {} tables, {} rows",
            report.pushed_tables, report.pushed_rows
        )
        .green()
        .bold()
    );
}

/// Run garbage collection command
fn run_gc(
    min_retention: f64,
    max_age_days: Option<u64>,
    dry_run: bool,
    yes: bool,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Garbage Collection ===".cyan().bold());
    println!();

    let storage = open_storage()?;
    let all_nodes = fetch_all_nodes(&storage)?;
    let now = Utc::now();

    // Find candidates for deletion
    let candidates: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            // Must be below retention threshold
            if node.retention_strength >= min_retention {
                return false;
            }
            // If max_age_days specified, must also be older than that
            if let Some(max_days) = max_age_days {
                let age_days = (now - node.created_at).num_days();
                if age_days < 0 || (age_days as u64) < max_days {
                    return false;
                }
            }
            true
        })
        .collect();

    println!(
        "{}: {}",
        "Min retention threshold".white().bold(),
        min_retention
    );
    if let Some(max_days) = max_age_days {
        println!("{}: {} days", "Max age".white().bold(), max_days);
    }
    println!(
        "{}: {} / {} total",
        "Candidates for deletion".white().bold(),
        candidates.len(),
        all_nodes.len()
    );

    if candidates.is_empty() {
        println!();
        println!(
            "{}",
            "No memories match the garbage collection criteria.".green()
        );
        return Ok(());
    }

    // Show sample of what would be deleted
    println!();
    println!("{}", "Sample of memories to be removed:".yellow().bold());
    let sample_count = candidates.len().min(10);
    for node in candidates.iter().take(sample_count) {
        let age_days = (now - node.created_at).num_days();
        println!(
            "  {} [ret={:.3}, age={}d] {}",
            node.id[..8].dimmed(),
            node.retention_strength,
            age_days,
            truncate(&node.content, 60).dimmed()
        );
    }
    if candidates.len() > sample_count {
        println!(
            "  {} ... and {} more",
            "".dimmed(),
            candidates.len() - sample_count
        );
    }

    if dry_run {
        println!();
        println!(
            "{}",
            format!(
                "Dry run: {} memories would be deleted. Re-run without --dry-run to delete.",
                candidates.len()
            )
            .yellow()
            .bold()
        );
        return Ok(());
    }

    // Confirmation prompt (unless --yes)
    if !yes {
        println!();
        print!(
            "{} Delete {} memories? This cannot be undone. [y/N] ",
            "WARNING:".red().bold(),
            candidates.len()
        );
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_lowercase();

        if input != "y" && input != "yes" {
            println!("{}", "Aborted.".yellow());
            return Ok(());
        }
    }

    // Perform deletion
    let mut deleted = 0;
    let mut errors = 0;
    let total_candidates = candidates.len();

    for node in &candidates {
        match storage.delete_node(&node.id) {
            Ok(true) => deleted += 1,
            Ok(false) => errors += 1, // node was already gone
            Err(e) => {
                eprintln!(
                    "  {} Failed to delete {}: {}",
                    "ERR".red(),
                    &node.id[..8],
                    e
                );
                errors += 1;
            }
        }
    }

    println!();
    println!(
        "{}",
        format!(
            "Garbage collection complete: {}/{} memories deleted{}",
            deleted,
            total_candidates,
            if errors > 0 {
                format!(" ({} errors)", errors)
            } else {
                String::new()
            }
        )
        .green()
        .bold()
    );

    Ok(())
}

/// Ingest a memory via CLI (routes through smart_ingest / PE Gating)
fn run_ingest(
    content: String,
    tags: Option<String>,
    node_type: String,
    source: Option<String>,
) -> anyhow::Result<()> {
    if content.trim().is_empty() {
        anyhow::bail!("Content cannot be empty");
    }

    let tag_list: Vec<String> = tags
        .as_deref()
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let input = IngestInput {
        content: content.clone(),
        node_type,
        source,
        sentiment_score: 0.0,
        sentiment_magnitude: 0.0,
        tags: tag_list,
        valid_from: None,
        valid_until: None,
        source_envelope: None,
    };

    let storage = open_storage()?;

    // Try smart_ingest (PE Gating) if available, otherwise regular ingest
    #[cfg(all(feature = "embeddings", feature = "vector-search"))]
    {
        let result = storage.smart_ingest(input)?;
        println!("{}", "=== Vestige Ingest ===".cyan().bold());
        println!();
        println!("{}: {}", "Decision".white().bold(), result.decision.green());
        println!("{}: {}", "Node ID".white().bold(), result.node.id);
        if let Some(sim) = result.similarity {
            println!("{}: {:.3}", "Similarity".white().bold(), sim);
        }
        if let Some(pe) = result.prediction_error {
            println!("{}: {:.3}", "Prediction Error".white().bold(), pe);
        }
        println!("{}: {}", "Reason".white().bold(), result.reason);
        println!();
        println!(
            "{}",
            format!("Memory {} ({})", result.decision, truncate(&content, 60))
                .green()
                .bold()
        );
    }

    #[cfg(not(all(feature = "embeddings", feature = "vector-search")))]
    {
        let node = storage.ingest(input)?;
        println!("{}", "=== Vestige Ingest ===".cyan().bold());
        println!();
        println!("{}: create", "Decision".white().bold());
        println!("{}: {}", "Node ID".white().bold(), node.id);
        println!();
        println!(
            "{}",
            format!("Memory created ({})", truncate(&content, 60))
                .green()
                .bold()
        );
    }

    Ok(())
}

/// Run the dashboard web server
fn run_dashboard(port: u16, open_browser: bool) -> anyhow::Result<()> {
    use vestige_mcp::cognitive::CognitiveEngine;

    println!("{}", "=== Vestige Dashboard ===".cyan().bold());
    println!();
    println!(
        "Starting dashboard at {}...",
        format!("http://127.0.0.1:{}", port).cyan()
    );

    let storage = open_storage()?;

    // Try to initialize embeddings for search support
    #[cfg(feature = "embeddings")]
    {
        if let Err(e) = storage.init_embeddings() {
            println!(
                "  {} Embeddings unavailable: {} (search will use keyword-only)",
                "!".yellow(),
                e
            );
        }
    }

    let storage = std::sync::Arc::new(storage);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        // Initialize cognitive engine for dream and other cognitive features
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        {
            let mut cog = cognitive.lock().await;
            cog.hydrate(&storage); // Load persisted connections
        }

        vestige_mcp::dashboard::start_dashboard(storage, Some(cognitive), port, open_browser)
            .await
            .map_err(|e| anyhow::anyhow!("Dashboard error: {}", e))
    })
}

/// Start standalone HTTP MCP server (no stdio transport)
fn run_serve(port: u16, with_dashboard: bool, dashboard_port: u16) -> anyhow::Result<()> {
    use vestige_mcp::cognitive::CognitiveEngine;

    println!("{}", "=== Vestige HTTP Server ===".cyan().bold());
    println!();

    let storage = open_storage()?;

    #[cfg(feature = "embeddings")]
    {
        if let Err(e) = storage.init_embeddings() {
            println!(
                "  {} Embeddings unavailable: {} (search will use keyword-only)",
                "!".yellow(),
                e
            );
        }
    }

    let storage = Arc::new(storage);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let cognitive = Arc::new(tokio::sync::Mutex::new(CognitiveEngine::new()));
        {
            let mut cog = cognitive.lock().await;
            cog.hydrate(&storage);
        }

        let (event_tx, _) =
            tokio::sync::broadcast::channel::<vestige_mcp::dashboard::events::VestigeEvent>(1024);

        // Optionally start dashboard
        if with_dashboard {
            let ds = Arc::clone(&storage);
            let dc = Arc::clone(&cognitive);
            let dtx = event_tx.clone();
            tokio::spawn(async move {
                match vestige_mcp::dashboard::start_background_with_event_tx(
                    ds,
                    Some(dc),
                    dtx,
                    dashboard_port,
                )
                .await
                {
                    Ok(_) => println!(
                        "  {} Dashboard: http://127.0.0.1:{}",
                        ">".cyan(),
                        dashboard_port
                    ),
                    Err(e) => eprintln!("  {} Dashboard failed: {}", "!".yellow(), e),
                }
            });
        }

        // Get auth token
        let token = vestige_mcp::protocol::auth::get_or_create_auth_token()
            .map_err(|e| anyhow::anyhow!("Failed to create auth token: {}", e))?;

        let bind = std::env::var("VESTIGE_HTTP_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
        println!(
            "  {} HTTP transport: http://{}:{}/mcp",
            ">".cyan(),
            bind,
            port
        );
        if let Ok(path) = vestige_mcp::protocol::auth::token_path() {
            println!("  {} Auth token file: {}", ">".cyan(), path.display());
        }
        println!();
        println!("{}", "Press Ctrl+C to stop.".dimmed());

        // Start HTTP transport (blocks on the server, no stdio)
        vestige_mcp::protocol::http::start_http_transport(
            Arc::clone(&storage),
            Arc::clone(&cognitive),
            event_tx,
            token,
            port,
        )
        .await
        .map_err(|e| anyhow::anyhow!("HTTP transport failed: {}", e))?;

        // Keep the process alive (the HTTP server runs in a spawned task)
        tokio::signal::ctrl_c().await.ok();
        println!();
        println!("{}", "Shutting down...".dimmed());

        Ok(())
    })
}

/// Truncate a string for display (UTF-8 safe)
fn truncate(s: &str, max_chars: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max_chars {
        s
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_asset_mapping_matches_release_names() {
        let mac_arm = release_asset_for("macos", "aarch64").unwrap();
        assert_eq!(mac_arm.target, "aarch64-apple-darwin");
        assert_eq!(mac_arm.archive_ext, "tar.gz");
        assert_eq!(mac_arm.binary_suffix, "");

        let linux = release_asset_for("linux", "x86_64").unwrap();
        assert_eq!(linux.target, "x86_64-unknown-linux-gnu");
        assert_eq!(linux.archive_ext, "tar.gz");

        let windows = release_asset_for("windows", "x86_64").unwrap();
        assert_eq!(windows.target, "x86_64-pc-windows-msvc");
        assert_eq!(windows.archive_ext, "zip");
        assert_eq!(windows.binary_suffix, ".exe");
    }

    #[test]
    fn update_url_uses_latest_or_normalized_tag() {
        let asset = release_asset_for("macos", "aarch64").unwrap();
        assert_eq!(
            release_download_url(asset, None),
            "https://github.com/samvallad33/vestige/releases/latest/download/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            release_download_url(asset, Some("2.1.0")),
            "https://github.com/samvallad33/vestige/releases/download/v2.1.0/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(
            release_download_url(asset, Some("v2.1.0")),
            "https://github.com/samvallad33/vestige/releases/download/v2.1.0/vestige-mcp-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    fn source_archive_url_uses_normalized_tag() {
        assert_eq!(normalize_release_tag("2.1.1"), "v2.1.1");
        assert_eq!(normalize_release_tag("v2.1.1"), "v2.1.1");
        assert_eq!(
            source_archive_url("v2.1.1"),
            "https://github.com/samvallad33/vestige/archive/refs/tags/v2.1.1.tar.gz"
        );
    }

    #[test]
    fn scrub_vestige_hooks_removes_only_vestige_commands() {
        let mut settings = serde_json::json!({
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            { "type": "command", "command": "/tmp/synthesis-preflight.sh" },
                            { "type": "command", "command": "/tmp/custom-user-hook.sh" }
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            { "type": "command", "command": "/tmp/sanhedrin.sh" }
                        ]
                    }
                ]
            },
            "other": true
        });

        scrub_vestige_hooks(&mut settings);

        let user_hooks = settings["hooks"]["UserPromptSubmit"][0]["hooks"]
            .as_array()
            .unwrap();
        assert_eq!(user_hooks.len(), 1);
        assert_eq!(user_hooks[0]["command"], "/tmp/custom-user-hook.sh");
        assert!(settings["hooks"].get("Stop").is_none());
        assert_eq!(settings["other"], true);
    }
}

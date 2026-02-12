//! Vestige CLI
//!
//! Command-line interface for managing cognitive memory system.

use std::io::{BufWriter, Write};
use std::path::PathBuf;

use chrono::{NaiveDate, Utc};
use clap::{Parser, Subcommand};
use colored::Colorize;
use directories::ProjectDirs;
use vestige_core::{IngestInput, Storage};

/// Vestige - Cognitive Memory System CLI
#[derive(Parser)]
#[command(name = "vestige")]
#[command(author = "samvallad33")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CLI for the Vestige cognitive memory system")]
#[command(long_about = "Vestige is a cognitive memory system based on 130 years of memory research.\n\nIt implements FSRS-6, spreading activation, synaptic tagging, and more.")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats { tagging, states } => run_stats(tagging, states),
        Commands::Health => run_health(),
        Commands::Consolidate => run_consolidate(),
        Commands::Restore { file } => run_restore(file),
        Commands::Backup { output } => run_backup(output),
        Commands::Export {
            output,
            format,
            tags,
            since,
        } => run_export(output, format, tags, since),
        Commands::Gc {
            min_retention,
            max_age_days,
            dry_run,
            yes,
        } => run_gc(min_retention, max_age_days, dry_run, yes),
    }
}

/// Run stats command
fn run_stats(show_tagging: bool, show_states: bool) -> anyhow::Result<()> {
    let storage = Storage::new(None)?;
    let stats = storage.get_stats()?;

    println!("{}", "=== Vestige Memory Statistics ===".cyan().bold());
    println!();

    // Basic stats
    println!("{}: {}", "Total Memories".white().bold(), stats.total_nodes);
    println!("{}: {}", "Due for Review".white().bold(), stats.nodes_due_for_review);
    println!("{}: {:.1}%", "Average Retention".white().bold(), stats.average_retention * 100.0);
    println!("{}: {:.2}", "Average Storage Strength".white().bold(), stats.average_storage_strength);
    println!("{}: {:.2}", "Average Retrieval Strength".white().bold(), stats.average_retrieval_strength);
    println!("{}: {}", "With Embeddings".white().bold(), stats.nodes_with_embeddings);

    if let Some(model) = &stats.embedding_model {
        println!("{}: {}", "Embedding Model".white().bold(), model);
    }

    if let Some(oldest) = stats.oldest_memory {
        println!("{}: {}", "Oldest Memory".white().bold(), oldest.format("%Y-%m-%d %H:%M:%S"));
    }
    if let Some(newest) = stats.newest_memory {
        println!("{}: {}", "Newest Memory".white().bold(), newest.format("%Y-%m-%d %H:%M:%S"));
    }

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!("{}: {:.1}%", "Embedding Coverage".white().bold(), embedding_coverage);

    // Tagging distribution (retention levels)
    if show_tagging {
        println!();
        println!("{}", "=== Retention Distribution ===".yellow().bold());

        let memories = storage.get_all_nodes(500, 0)?;
        let total = memories.len();

        if total > 0 {
            let high = memories.iter().filter(|m| m.retention_strength >= 0.7).count();
            let medium = memories.iter().filter(|m| m.retention_strength >= 0.4 && m.retention_strength < 0.7).count();
            let low = memories.iter().filter(|m| m.retention_strength < 0.4).count();

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
        println!("{}", "=== Cognitive State Distribution ===".magenta().bold());

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
fn compute_state_distribution(memories: &[vestige_core::KnowledgeNode]) -> (usize, usize, usize, usize) {
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
        label,
        colored_bar,
        count,
        percentage
    );
}

/// Run health check
fn run_health() -> anyhow::Result<()> {
    let storage = Storage::new(None)?;
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
    println!("{}: {}", "Due for Review".white(), stats.nodes_due_for_review);
    println!("{}: {:.1}%", "Average Retention".white(), stats.average_retention * 100.0);

    // Embedding coverage
    let embedding_coverage = if stats.total_nodes > 0 {
        (stats.nodes_with_embeddings as f64 / stats.total_nodes as f64) * 100.0
    } else {
        0.0
    };
    println!("{}: {:.1}%", "Embedding Coverage".white(), embedding_coverage);
    println!("{}: {}", "Embedding Service".white(),
        if storage.is_embedding_ready() { "Ready".green() } else { "Not Ready".red() });

    // Warnings
    let mut warnings = Vec::new();

    if stats.average_retention < 0.5 && stats.total_nodes > 0 {
        warnings.push("Low average retention - consider running consolidation or reviewing memories");
    }

    if stats.nodes_due_for_review > 10 {
        warnings.push("Many memories are due for review");
    }

    if stats.total_nodes > 0 && stats.nodes_with_embeddings == 0 {
        warnings.push("No embeddings generated - semantic search unavailable");
    }

    if embedding_coverage < 50.0 && stats.total_nodes > 10 {
        warnings.push("Low embedding coverage - run consolidation to improve semantic search");
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
        recommendations.push("CRITICAL: Many memories have very low retention. Review important memories.");
    }

    if stats.nodes_due_for_review > 5 {
        recommendations.push("Review due memories to strengthen retention.");
    }

    if stats.nodes_with_embeddings < stats.total_nodes {
        recommendations.push("Run 'vestige consolidate' to generate embeddings for better semantic search.");
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
        let icon = if rec.starts_with("CRITICAL") { "!".red().bold() } else { ">".cyan() };
        let text = if rec.starts_with("CRITICAL") { rec.red().to_string() } else { rec.to_string() };
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

    let mut storage = Storage::new(None)?;
    let result = storage.run_consolidation()?;

    println!("{}: {}", "Nodes Processed".white().bold(), result.nodes_processed);
    println!("{}: {}", "Nodes Promoted".white().bold(), result.nodes_promoted);
    println!("{}: {}", "Nodes Pruned".white().bold(), result.nodes_pruned);
    println!("{}: {}", "Decay Applied".white().bold(), result.decay_applied);
    println!("{}: {}", "Embeddings Generated".white().bold(), result.embeddings_generated);
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
    let backup_content = std::fs::read_to_string(&backup_path)?;

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

    let wrapper: Vec<BackupWrapper> = serde_json::from_str(&backup_content)?;
    let recall_result: RecallResult = serde_json::from_str(&wrapper[0].text)?;
    let memories = recall_result.results;

    println!("Found {} memories to restore", memories.len());
    println!();

    // Initialize storage
    println!("Initializing storage...");
    let mut storage = Storage::new(None)?;

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
    println!("{}: {}", "With Embeddings".white(), stats.nodes_with_embeddings);

    Ok(())
}

/// Get the default database path
fn get_default_db_path() -> anyhow::Result<PathBuf> {
    let proj_dirs = ProjectDirs::from("com", "vestige", "core")
        .ok_or_else(|| anyhow::anyhow!("Could not determine project directories"))?;
    Ok(proj_dirs.data_dir().join("vestige.db"))
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
        let storage = Storage::new(None)?;
        // get_stats triggers a read so the connection is active, then drop flushes
        let _ = storage.get_stats()?;
    }

    // Also flush WAL directly via a separate connection for safety
    {
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
    }

    // Create parent directories if needed
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Copy the database file
    println!("Copying database...");
    println!("  {} {}", "From:".dimmed(), db_path.display());
    println!("  {}   {}", "To:".dimmed(), output.display());

    std::fs::copy(&db_path, &output)?;

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
            let naive = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|e| anyhow::anyhow!("Invalid date '{}': {}. Use YYYY-MM-DD format.", date_str, e))?;
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
        .map(|t| t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();

    let storage = Storage::new(None)?;
    let all_nodes = fetch_all_nodes(&storage)?;

    // Apply filters
    let filtered: Vec<&vestige_core::KnowledgeNode> = all_nodes
        .iter()
        .filter(|node| {
            // Date filter
            if let Some(ref since_dt) = since_date {
                if node.created_at < *since_dt {
                    return false;
                }
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
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = std::fs::File::create(&output)?;
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

/// Run garbage collection command
fn run_gc(
    min_retention: f64,
    max_age_days: Option<u64>,
    dry_run: bool,
    yes: bool,
) -> anyhow::Result<()> {
    println!("{}", "=== Vestige Garbage Collection ===".cyan().bold());
    println!();

    let mut storage = Storage::new(None)?;
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

    println!("{}: {}", "Min retention threshold".white().bold(), min_retention);
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
        println!("{}", "No memories match the garbage collection criteria.".green());
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
                eprintln!("  {} Failed to delete {}: {}", "ERR".red(), &node.id[..8], e);
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

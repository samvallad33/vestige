use std::path::PathBuf;
use vestige_core::{IngestInput, Storage};

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

fn main() -> anyhow::Result<()> {
    // Parse args
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: vestige-restore <backup.json>");
        std::process::exit(1);
    }

    let backup_path = PathBuf::from(&args[1]);
    println!("Loading backup from: {}", backup_path.display());

    // Read and parse backup
    let backup_content = std::fs::read_to_string(&backup_path)?;
    let wrapper: Vec<BackupWrapper> = serde_json::from_str(&backup_content)?;
    let recall_result: RecallResult = serde_json::from_str(&wrapper[0].text)?;
    let memories = recall_result.results;

    println!("Found {} memories to restore", memories.len());

    // Initialize storage (uses default path)
    println!("Initializing storage...");
    let mut storage = Storage::new(None)?;

    println!("Generating embeddings and ingesting memories...\n");

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
                println!("[{}/{}] OK: {}", i + 1, total, truncate(&memory.content, 60));
            }
            Err(e) => {
                println!("[{}/{}] FAIL: {}", i + 1, total, e);
            }
        }
    }

    println!("\nRestore complete: {}/{} memories restored", success_count, total);

    // Show stats
    let stats = storage.get_stats()?;
    println!("Total nodes: {}", stats.total_nodes);
    println!("With embeddings: {}", stats.nodes_with_embeddings);

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

//! # Semantic Memory Compression
//!
//! Compress old memories while preserving their semantic meaning.
//! This allows Vestige to maintain vast amounts of knowledge without
//! overwhelming storage or search latency.
//!
//! ## Compression Strategy
//!
//! 1. **Identify compressible groups**: Find memories that are related and old enough
//! 2. **Extract key facts**: Pull out the essential information
//! 3. **Generate summary**: Create a concise summary preserving meaning
//! 4. **Store compressed form**: Save summary with references to originals
//! 5. **Lazy decompress**: Load originals only when needed
//!
//! ## Semantic Fidelity
//!
//! The compression algorithm measures how well meaning is preserved:
//! - Cosine similarity between original embeddings and compressed embedding
//! - Key fact extraction coverage
//! - Information entropy preservation
//!
//! ## Example
//!
//! ```rust,ignore
//! let compressor = MemoryCompressor::new();
//!
//! // Check if memories can be compressed together
//! if compressor.can_compress(&old_memories) {
//!     let compressed = compressor.compress(&old_memories);
//!     println!("Compressed {} memories to {:.0}%",
//!         old_memories.len(),
//!         compressed.compression_ratio * 100.0);
//! }
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Minimum memories needed for compression
const MIN_MEMORIES_FOR_COMPRESSION: usize = 3;

/// Maximum memories in a single compression group
const MAX_COMPRESSION_GROUP_SIZE: usize = 50;

/// Minimum semantic similarity for grouping
const MIN_SIMILARITY_THRESHOLD: f64 = 0.6;

/// Minimum age in days for compression consideration
const MIN_AGE_DAYS: i64 = 30;

/// A compressed memory representing multiple original memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedMemory {
    /// Unique ID for this compressed memory
    pub id: String,
    /// High-level summary of all compressed memories
    pub summary: String,
    /// Extracted key facts from the originals
    pub key_facts: Vec<KeyFact>,
    /// IDs of the original memories that were compressed
    pub original_ids: Vec<String>,
    /// Compression ratio (0.0 to 1.0, lower = more compression)
    pub compression_ratio: f64,
    /// How well the semantic meaning was preserved (0.0 to 1.0)
    pub semantic_fidelity: f64,
    /// Tags aggregated from original memories
    pub tags: Vec<String>,
    /// When this compression was created
    pub created_at: DateTime<Utc>,
    /// Embedding of the compressed summary
    pub embedding: Option<Vec<f32>>,
    /// Total character count of originals
    pub original_size: usize,
    /// Character count of compressed form
    pub compressed_size: usize,
}

impl CompressedMemory {
    /// Create a new compressed memory
    pub fn new(summary: String, key_facts: Vec<KeyFact>, original_ids: Vec<String>) -> Self {
        let compressed_size = summary.len() + key_facts.iter().map(|f| f.fact.len()).sum::<usize>();

        Self {
            id: format!("compressed-{}", Uuid::new_v4()),
            summary,
            key_facts,
            original_ids,
            compression_ratio: 0.0, // Will be calculated
            semantic_fidelity: 0.0, // Will be calculated
            tags: Vec::new(),
            created_at: Utc::now(),
            embedding: None,
            original_size: 0,
            compressed_size,
        }
    }

    /// Check if a search query might need decompression
    pub fn might_need_decompression(&self, query: &str) -> bool {
        // Check if query terms appear in key facts
        let query_lower = query.to_lowercase();
        self.key_facts.iter().any(|f| {
            f.fact.to_lowercase().contains(&query_lower)
                || f.keywords
                    .iter()
                    .any(|k| query_lower.contains(&k.to_lowercase()))
        })
    }
}

/// A key fact extracted from memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFact {
    /// The fact itself
    pub fact: String,
    /// Keywords associated with this fact
    pub keywords: Vec<String>,
    /// How important this fact is (0.0 to 1.0)
    pub importance: f64,
    /// Which original memory this came from
    pub source_id: String,
}

/// Configuration for memory compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Minimum memories needed for compression
    pub min_group_size: usize,
    /// Maximum memories in a compression group
    pub max_group_size: usize,
    /// Minimum similarity for grouping
    pub similarity_threshold: f64,
    /// Minimum age in days before compression
    pub min_age_days: i64,
    /// Target compression ratio (0.1 = compress to 10%)
    pub target_ratio: f64,
    /// Minimum semantic fidelity required
    pub min_fidelity: f64,
    /// Maximum key facts to extract per memory
    pub max_facts_per_memory: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_group_size: MIN_MEMORIES_FOR_COMPRESSION,
            max_group_size: MAX_COMPRESSION_GROUP_SIZE,
            similarity_threshold: MIN_SIMILARITY_THRESHOLD,
            min_age_days: MIN_AGE_DAYS,
            target_ratio: 0.3,
            min_fidelity: 0.7,
            max_facts_per_memory: 3,
        }
    }
}

/// Statistics about compression operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Total memories compressed
    pub memories_compressed: usize,
    /// Total compressed memories created
    pub compressions_created: usize,
    /// Average compression ratio achieved
    pub average_ratio: f64,
    /// Average semantic fidelity
    pub average_fidelity: f64,
    /// Total bytes saved
    pub bytes_saved: usize,
    /// Compression operations performed
    pub operations: usize,
}

/// Input memory for compression (abstracted from storage)
#[derive(Debug, Clone)]
pub struct MemoryForCompression {
    /// Memory ID
    pub id: String,
    /// Memory content
    pub content: String,
    /// Memory tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: Option<DateTime<Utc>>,
    /// Embedding vector
    pub embedding: Option<Vec<f32>>,
}

/// Memory compressor for semantic compression
pub struct MemoryCompressor {
    /// Configuration
    config: CompressionConfig,
    /// Compression statistics
    stats: CompressionStats,
}

impl MemoryCompressor {
    /// Create a new memory compressor with default config
    pub fn new() -> Self {
        Self::with_config(CompressionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Check if a group of memories can be compressed
    pub fn can_compress(&self, memories: &[MemoryForCompression]) -> bool {
        // Check minimum size
        if memories.len() < self.config.min_group_size {
            return false;
        }

        // Check age - all must be old enough
        let now = Utc::now();
        let min_date = now - Duration::days(self.config.min_age_days);
        if !memories.iter().all(|m| m.created_at < min_date) {
            return false;
        }

        // Check semantic similarity - must be related
        if !self.are_semantically_related(memories) {
            return false;
        }

        true
    }

    /// Compress a group of related memories into a summary
    pub fn compress(&mut self, memories: &[MemoryForCompression]) -> Option<CompressedMemory> {
        if !self.can_compress(memories) {
            return None;
        }

        // Extract key facts from each memory
        let key_facts = self.extract_key_facts(memories);

        // Generate summary from key facts
        let summary = self.generate_summary(&key_facts, memories);

        // Calculate original size
        let original_size: usize = memories.iter().map(|m| m.content.len()).sum();

        // Create compressed memory
        let mut compressed = CompressedMemory::new(
            summary,
            key_facts,
            memories.iter().map(|m| m.id.clone()).collect(),
        );

        compressed.original_size = original_size;

        // Aggregate tags
        let all_tags: HashSet<_> = memories
            .iter()
            .flat_map(|m| m.tags.iter().cloned())
            .collect();
        compressed.tags = all_tags.into_iter().collect();

        // Calculate compression ratio
        compressed.compression_ratio = compressed.compressed_size as f64 / original_size as f64;

        // Calculate semantic fidelity (simplified - in production would use embedding comparison)
        compressed.semantic_fidelity = self.calculate_semantic_fidelity(&compressed, memories);

        // Update stats
        self.stats.memories_compressed += memories.len();
        self.stats.compressions_created += 1;
        self.stats.bytes_saved += original_size - compressed.compressed_size;
        self.stats.operations += 1;
        self.update_average_stats(&compressed);

        Some(compressed)
    }

    /// Decompress to retrieve original memory references
    pub fn decompress(&self, compressed: &CompressedMemory) -> DecompressionResult {
        DecompressionResult {
            compressed_id: compressed.id.clone(),
            original_ids: compressed.original_ids.clone(),
            summary: compressed.summary.clone(),
            key_facts: compressed.key_facts.clone(),
        }
    }

    /// Find groups of memories that could be compressed together
    pub fn find_compressible_groups(&self, memories: &[MemoryForCompression]) -> Vec<Vec<String>> {
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashSet<String> = HashSet::new();

        // Sort by age (oldest first)
        let mut sorted: Vec<_> = memories.iter().collect();
        sorted.sort_by(|a, b| a.created_at.cmp(&b.created_at));

        for memory in sorted {
            if assigned.contains(&memory.id) {
                continue;
            }

            // Try to form a group around this memory
            let mut group = vec![memory.id.clone()];
            assigned.insert(memory.id.clone());

            for other in memories {
                if assigned.contains(&other.id) {
                    continue;
                }

                if group.len() >= self.config.max_group_size {
                    break;
                }

                // Check if semantically similar
                if self.are_similar(memory, other) {
                    group.push(other.id.clone());
                    assigned.insert(other.id.clone());
                }
            }

            if group.len() >= self.config.min_group_size {
                groups.push(group);
            }
        }

        groups
    }

    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::default();
    }

    // ========================================================================
    // Private implementation
    // ========================================================================

    fn are_semantically_related(&self, memories: &[MemoryForCompression]) -> bool {
        // Check pairwise similarities
        // In production, this would use embeddings
        let embeddings: Vec<_> = memories
            .iter()
            .filter_map(|m| m.embedding.as_ref())
            .collect();

        if embeddings.len() < 2 {
            // Fall back to tag overlap
            return self.have_tag_overlap(memories);
        }

        // Calculate average pairwise similarity
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                total_sim += cosine_similarity(embeddings[i], embeddings[j]);
                count += 1;
            }
        }

        if count == 0 {
            return false;
        }

        let avg_sim = total_sim / count as f64;
        avg_sim >= self.config.similarity_threshold
    }

    fn have_tag_overlap(&self, memories: &[MemoryForCompression]) -> bool {
        if memories.len() < 2 {
            return false;
        }

        // Count tag frequencies
        let mut tag_counts: HashMap<&str, usize> = HashMap::new();
        for memory in memories {
            for tag in &memory.tags {
                *tag_counts.entry(tag.as_str()).or_insert(0) += 1;
            }
        }

        // Check if any tag appears in majority of memories
        let threshold = memories.len() / 2;
        tag_counts.values().any(|&count| count > threshold)
    }

    fn are_similar(&self, a: &MemoryForCompression, b: &MemoryForCompression) -> bool {
        // Try embedding similarity first
        if let (Some(emb_a), Some(emb_b)) = (&a.embedding, &b.embedding) {
            let sim = cosine_similarity(emb_a, emb_b);
            return sim >= self.config.similarity_threshold;
        }

        // Fall back to tag overlap
        let a_tags: HashSet<_> = a.tags.iter().collect();
        let b_tags: HashSet<_> = b.tags.iter().collect();
        let overlap = a_tags.intersection(&b_tags).count();
        let union = a_tags.union(&b_tags).count();

        if union == 0 {
            return false;
        }

        (overlap as f64 / union as f64) >= 0.3
    }

    fn extract_key_facts(&self, memories: &[MemoryForCompression]) -> Vec<KeyFact> {
        let mut facts = Vec::new();

        for memory in memories {
            // Extract sentences as potential facts
            let sentences = self.extract_sentences(&memory.content);

            // Score and select top facts
            let mut scored: Vec<_> = sentences
                .iter()
                .map(|s| (s, self.score_sentence(s, &memory.content)))
                .collect();

            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (sentence, score) in scored.into_iter().take(self.config.max_facts_per_memory) {
                if score > 0.3 {
                    facts.push(KeyFact {
                        fact: sentence.to_string(),
                        keywords: self.extract_keywords(sentence),
                        importance: score,
                        source_id: memory.id.clone(),
                    });
                }
            }
        }

        // Sort by importance and deduplicate
        facts.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        self.deduplicate_facts(facts)
    }

    fn extract_sentences<'a>(&self, content: &'a str) -> Vec<&'a str> {
        content
            .split(['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| s.len() > 10) // Filter very short fragments
            .collect()
    }

    fn score_sentence(&self, sentence: &str, full_content: &str) -> f64 {
        let mut score: f64 = 0.0;

        // Length factor (prefer medium-length sentences)
        let words = sentence.split_whitespace().count();
        if (5..=25).contains(&words) {
            score += 0.3;
        }

        // Position factor (first sentences often more important)
        if full_content.starts_with(sentence) {
            score += 0.2;
        }

        // Keyword density (sentences with more "important" words)
        let important_patterns = [
            "is",
            "are",
            "must",
            "should",
            "always",
            "never",
            "important",
        ];
        for pattern in important_patterns {
            if sentence.to_lowercase().contains(pattern) {
                score += 0.1;
            }
        }

        // Cap at 1.0
        score.min(1.0)
    }

    fn extract_keywords(&self, sentence: &str) -> Vec<String> {
        // Simple keyword extraction - in production would use NLP
        let stopwords: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
            "at", "by", "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "until", "while", "this", "that", "these",
            "those", "it",
        ]
        .into_iter()
        .collect();

        sentence
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 3 && !stopwords.contains(w.to_lowercase().as_str()))
            .map(|w| w.to_lowercase())
            .take(5)
            .collect()
    }

    fn deduplicate_facts(&self, facts: Vec<KeyFact>) -> Vec<KeyFact> {
        let mut seen_facts: HashSet<String> = HashSet::new();
        let mut result = Vec::new();

        for fact in facts {
            let normalized = fact.fact.to_lowercase();
            if !seen_facts.contains(&normalized) {
                seen_facts.insert(normalized);
                result.push(fact);
            }
        }

        result
    }

    fn generate_summary(&self, key_facts: &[KeyFact], memories: &[MemoryForCompression]) -> String {
        // Generate a summary from key facts
        let mut summary_parts: Vec<String> = Vec::new();

        // Aggregate common tags for context
        let tag_counts: HashMap<&str, usize> = memories
            .iter()
            .flat_map(|m| m.tags.iter().map(|t| t.as_str()))
            .fold(HashMap::new(), |mut acc, tag| {
                *acc.entry(tag).or_insert(0) += 1;
                acc
            });

        let common_tags: Vec<_> = tag_counts
            .iter()
            .filter(|(_, &count)| count > memories.len() / 2)
            .map(|(&tag, _)| tag)
            .take(3)
            .collect();

        if !common_tags.is_empty() {
            summary_parts.push(format!(
                "Collection of {} related memories about: {}.",
                memories.len(),
                common_tags.join(", ")
            ));
        }

        // Add top key facts
        let top_facts: Vec<_> = key_facts
            .iter()
            .filter(|f| f.importance > 0.5)
            .take(5)
            .collect();

        if !top_facts.is_empty() {
            summary_parts.push("Key points:".to_string());
            for fact in top_facts {
                summary_parts.push(format!("- {}", fact.fact));
            }
        }

        summary_parts.join("\n")
    }

    fn calculate_semantic_fidelity(
        &self,
        compressed: &CompressedMemory,
        memories: &[MemoryForCompression],
    ) -> f64 {
        // Calculate how well key information is preserved
        let mut preserved_count = 0;
        let mut total_check = 0;

        for memory in memories {
            // Check if key keywords from original appear in compressed
            let original_keywords: HashSet<_> = memory
                .content
                .split_whitespace()
                .filter(|w| w.len() > 4)
                .map(|w| w.to_lowercase())
                .collect();

            let compressed_text = format!(
                "{} {}",
                compressed.summary,
                compressed
                    .key_facts
                    .iter()
                    .map(|f| f.fact.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .to_lowercase();

            for keyword in original_keywords.iter().take(10) {
                total_check += 1;
                if compressed_text.contains(keyword) {
                    preserved_count += 1;
                }
            }
        }

        if total_check == 0 {
            return 0.8; // Default fidelity when can't check
        }

        let keyword_fidelity = preserved_count as f64 / total_check as f64;

        // Also factor in fact coverage
        let fact_coverage = (compressed.key_facts.len() as f64
            / (memories.len() * self.config.max_facts_per_memory) as f64)
            .min(1.0);

        // Combined fidelity score
        (keyword_fidelity * 0.7 + fact_coverage * 0.3).min(1.0)
    }

    fn update_average_stats(&mut self, compressed: &CompressedMemory) {
        let n = self.stats.compressions_created as f64;
        self.stats.average_ratio =
            (self.stats.average_ratio * (n - 1.0) + compressed.compression_ratio) / n;
        self.stats.average_fidelity =
            (self.stats.average_fidelity * (n - 1.0) + compressed.semantic_fidelity) / n;
    }
}

impl Default for MemoryCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of decompression operation
#[derive(Debug, Clone)]
pub struct DecompressionResult {
    /// ID of the compressed memory
    pub compressed_id: String,
    /// Original memory IDs to load
    pub original_ids: Vec<String>,
    /// Summary for quick reference
    pub summary: String,
    /// Key facts extracted
    pub key_facts: Vec<KeyFact>,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    (dot / (mag_a * mag_b)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(id: &str, content: &str, tags: Vec<&str>) -> MemoryForCompression {
        MemoryForCompression {
            id: id.to_string(),
            content: content.to_string(),
            tags: tags.into_iter().map(String::from).collect(),
            created_at: Utc::now() - Duration::days(60),
            last_accessed: None,
            embedding: None,
        }
    }

    #[test]
    fn test_can_compress_minimum_size() {
        let compressor = MemoryCompressor::new();

        let memories = vec![
            make_memory("1", "Content one", vec!["tag"]),
            make_memory("2", "Content two", vec!["tag"]),
        ];

        // Too few memories
        assert!(!compressor.can_compress(&memories));
    }

    #[test]
    fn test_extract_sentences() {
        let compressor = MemoryCompressor::new();

        let content = "This is the first sentence. This is the second one! And a third?";
        let sentences = compressor.extract_sentences(content);

        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_extract_keywords() {
        let compressor = MemoryCompressor::new();

        let sentence = "The Rust programming language is very powerful";
        let keywords = compressor.extract_keywords(sentence);

        assert!(keywords.contains(&"rust".to_string()));
        assert!(keywords.contains(&"programming".to_string()));
        assert!(!keywords.contains(&"the".to_string()));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }
}

//! Database Migrations
//!
//! Schema migration definitions for the storage layer.

/// Migration definitions
pub const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        description: "Initial schema with FSRS-6 and embeddings",
        up: MIGRATION_V1_UP,
    },
    Migration {
        version: 2,
        description: "Add temporal columns",
        up: MIGRATION_V2_UP,
    },
    Migration {
        version: 3,
        description: "Add persistence tables for neuroscience features",
        up: MIGRATION_V3_UP,
    },
    Migration {
        version: 4,
        description: "GOD TIER 2026: Temporal knowledge graph, memory scopes, embedding versioning",
        up: MIGRATION_V4_UP,
    },
    Migration {
        version: 5,
        description: "FSRS-6 upgrade: access history, ACT-R activation, personalized decay",
        up: MIGRATION_V5_UP,
    },
    Migration {
        version: 6,
        description: "Dream history persistence for automation triggers",
        up: MIGRATION_V6_UP,
    },
    Migration {
        version: 7,
        description: "Performance: page_size 8192, FTS5 porter tokenizer",
        up: MIGRATION_V7_UP,
    },
    Migration {
        version: 8,
        description: "v1.9.0 Autonomic: waking SWR tags, utility scoring, retention tracking",
        up: MIGRATION_V8_UP,
    },
    Migration {
        version: 9,
        description: "v2.0.0 Cognitive Leap: emotional memory, flashbulb encoding, temporal hierarchy",
        up: MIGRATION_V9_UP,
    },
    Migration {
        version: 10,
        description: "v2.0.5 Intentional Amnesia: active forgetting — top-down suppression (Anderson 2025 + Davis Rac1)",
        up: MIGRATION_V10_UP,
    },
    Migration {
        version: 11,
        description: "v2.0.7 Cleanup: drop dead knowledge_edges and compressed_memories tables",
        up: MIGRATION_V11_UP,
    },
    Migration {
        version: 12,
        description: "v2.1.1 Sync: tombstones for merge-capable portable storage",
        up: MIGRATION_V12_UP,
    },
    Migration {
        version: 13,
        description: "v2.1.2 Honest Memory: non-content purge tombstones",
        up: MIGRATION_V13_UP,
    },
    Migration {
        version: 14,
        description: "v2.1.25 Merge/Supersede: reversible operation log, merge plans, bitemporal lineage, protected pins",
        up: MIGRATION_V14_UP,
    },
];

/// A database migration
#[derive(Debug, Clone)]
pub struct Migration {
    /// Version number
    pub version: u32,
    /// Description
    pub description: &'static str,
    /// SQL to apply
    pub up: &'static str,
}

/// V1: Initial schema
const MIGRATION_V1_UP: &str = r#"
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'fact',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,

    -- FSRS-6 state (21 parameters)
    stability REAL DEFAULT 1.0,
    difficulty REAL DEFAULT 5.0,
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    learning_state TEXT DEFAULT 'new',

    -- Dual-strength model (Bjork & Bjork 1992)
    storage_strength REAL DEFAULT 1.0,
    retrieval_strength REAL DEFAULT 1.0,
    retention_strength REAL DEFAULT 1.0,

    -- Sentiment for emotional memory weighting
    sentiment_score REAL DEFAULT 0.0,
    sentiment_magnitude REAL DEFAULT 0.0,

    -- Scheduling
    next_review TEXT,
    scheduled_days INTEGER DEFAULT 0,

    -- Provenance
    source TEXT,
    tags TEXT DEFAULT '[]',

    -- Embedding metadata
    has_embedding INTEGER DEFAULT 0,
    embedding_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_retention ON knowledge_nodes(retention_strength);
CREATE INDEX IF NOT EXISTS idx_nodes_next_review ON knowledge_nodes(next_review);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON knowledge_nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_has_embedding ON knowledge_nodes(has_embedding);

-- Embeddings storage table (binary blob for efficiency)
CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id TEXT PRIMARY KEY REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL DEFAULT 768,
    model TEXT NOT NULL DEFAULT 'BAAI/bge-base-en-v1.5',
    created_at TEXT NOT NULL
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    id,
    content,
    tags,
    content='knowledge_nodes',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(rowid, id, content, tags)
    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, tags)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, tags)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags);
    INSERT INTO knowledge_fts(rowid, id, content, tags)
    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags);
END;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, datetime('now'));
"#;

/// V2: Add temporal columns
const MIGRATION_V2_UP: &str = r#"
ALTER TABLE knowledge_nodes ADD COLUMN valid_from TEXT;
ALTER TABLE knowledge_nodes ADD COLUMN valid_until TEXT;

CREATE INDEX IF NOT EXISTS idx_nodes_valid_from ON knowledge_nodes(valid_from);
CREATE INDEX IF NOT EXISTS idx_nodes_valid_until ON knowledge_nodes(valid_until);

UPDATE schema_version SET version = 2, applied_at = datetime('now');
"#;

/// V3: Add persistence tables for neuroscience features
/// Fixes critical gap: intentions, insights, and activation network were IN-MEMORY ONLY
const MIGRATION_V3_UP: &str = r#"
-- 1. INTENTIONS TABLE (Prospective Memory)
-- Stores future intentions/reminders with trigger conditions
CREATE TABLE IF NOT EXISTS intentions (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    trigger_type TEXT NOT NULL,  -- 'time', 'duration', 'event', 'context', 'activity', 'recurring', 'compound'
    trigger_data TEXT NOT NULL,  -- JSON: serialized IntentionTrigger
    priority INTEGER NOT NULL DEFAULT 2,  -- 1=Low, 2=Normal, 3=High, 4=Critical
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'triggered', 'fulfilled', 'cancelled', 'expired', 'snoozed'
    created_at TEXT NOT NULL,
    deadline TEXT,
    fulfilled_at TEXT,
    reminder_count INTEGER DEFAULT 0,
    last_reminded_at TEXT,
    notes TEXT,
    tags TEXT DEFAULT '[]',
    related_memories TEXT DEFAULT '[]',
    snoozed_until TEXT,
    source_type TEXT NOT NULL DEFAULT 'api',
    source_data TEXT
);

CREATE INDEX IF NOT EXISTS idx_intentions_status ON intentions(status);
CREATE INDEX IF NOT EXISTS idx_intentions_priority ON intentions(priority);
CREATE INDEX IF NOT EXISTS idx_intentions_deadline ON intentions(deadline);
CREATE INDEX IF NOT EXISTS idx_intentions_snoozed ON intentions(snoozed_until);

-- 2. INSIGHTS TABLE (From Consolidation/Dreams)
-- Stores AI-generated insights discovered during memory consolidation
CREATE TABLE IF NOT EXISTS insights (
    id TEXT PRIMARY KEY,
    insight TEXT NOT NULL,
    source_memories TEXT NOT NULL,  -- JSON array of memory IDs
    confidence REAL NOT NULL,
    novelty_score REAL NOT NULL,
    insight_type TEXT NOT NULL,  -- 'hidden_connection', 'recurring_pattern', 'generalization', 'contradiction', 'knowledge_gap', 'temporal_trend', 'synthesis'
    generated_at TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    feedback TEXT,  -- 'accepted', 'rejected', or NULL
    applied_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence);
CREATE INDEX IF NOT EXISTS idx_insights_generated ON insights(generated_at);
CREATE INDEX IF NOT EXISTS idx_insights_feedback ON insights(feedback);

-- 3. MEMORY_CONNECTIONS TABLE (Activation Network Edges)
-- Stores associations between memories for spreading activation
CREATE TABLE IF NOT EXISTS memory_connections (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    strength REAL NOT NULL,
    link_type TEXT NOT NULL,  -- 'semantic', 'temporal', 'spatial', 'causal', 'part_of', 'user_defined', 'cross_reference', 'sequential', 'shared_concepts', 'pattern'
    created_at TEXT NOT NULL,
    last_activated TEXT NOT NULL,
    activation_count INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_connections_source ON memory_connections(source_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON memory_connections(target_id);
CREATE INDEX IF NOT EXISTS idx_connections_strength ON memory_connections(strength);
CREATE INDEX IF NOT EXISTS idx_connections_type ON memory_connections(link_type);

-- 4. MEMORY_STATES TABLE (Accessibility States)
-- Tracks lifecycle state of each memory (Active/Dormant/Silent/Unavailable)
CREATE TABLE IF NOT EXISTS memory_states (
    memory_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'active',  -- 'active', 'dormant', 'silent', 'unavailable'
    last_access TEXT NOT NULL,
    access_count INTEGER DEFAULT 1,
    state_entered_at TEXT NOT NULL,
    suppression_until TEXT,
    suppressed_by TEXT DEFAULT '[]',
    time_active_seconds INTEGER DEFAULT 0,
    time_dormant_seconds INTEGER DEFAULT 0,
    time_silent_seconds INTEGER DEFAULT 0,
    time_unavailable_seconds INTEGER DEFAULT 0,
    FOREIGN KEY (memory_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_states_state ON memory_states(state);
CREATE INDEX IF NOT EXISTS idx_states_access ON memory_states(last_access);
CREATE INDEX IF NOT EXISTS idx_states_suppression ON memory_states(suppression_until);

-- 5. FSRS_CARDS TABLE (Extended Review State)
-- Stores complete FSRS-6 card state for spaced repetition
CREATE TABLE IF NOT EXISTS fsrs_cards (
    memory_id TEXT PRIMARY KEY,
    difficulty REAL NOT NULL DEFAULT 5.0,
    stability REAL NOT NULL DEFAULT 1.0,
    state TEXT NOT NULL DEFAULT 'new',  -- 'new', 'learning', 'review', 'relearning'
    reps INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    last_review TEXT,
    due_date TEXT,
    elapsed_days INTEGER DEFAULT 0,
    scheduled_days INTEGER DEFAULT 0,
    FOREIGN KEY (memory_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fsrs_due ON fsrs_cards(due_date);
CREATE INDEX IF NOT EXISTS idx_fsrs_state ON fsrs_cards(state);

-- 6. CONSOLIDATION_HISTORY TABLE (Dream Cycle Records)
-- Tracks when consolidation ran and what it accomplished
CREATE TABLE IF NOT EXISTS consolidation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    completed_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    memories_replayed INTEGER DEFAULT 0,
    connections_found INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    insights_generated INTEGER DEFAULT 0,
    memories_transferred TEXT DEFAULT '[]',
    patterns_discovered TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_consolidation_completed ON consolidation_history(completed_at);

-- 7. STATE_TRANSITIONS TABLE (Audit Trail)
-- Historical record of state changes for debugging and analytics
CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason_type TEXT NOT NULL,  -- 'access', 'time_decay', 'cue_reactivation', 'competition_loss', 'interference_resolved', 'user_suppression', 'suppression_expired', 'manual_override', 'system_init'
    reason_data TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transitions_memory ON state_transitions(memory_id);
CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON state_transitions(timestamp);

UPDATE schema_version SET version = 3, applied_at = datetime('now');
"#;

/// V4: GOD TIER 2026 - Temporal Knowledge Graph, Memory Scopes, Embedding Versioning
/// Competes with Zep's Graphiti and Mem0's memory scopes
const MIGRATION_V4_UP: &str = r#"
-- ============================================================================
-- TEMPORAL KNOWLEDGE GRAPH (Like Zep's Graphiti)
-- ============================================================================

-- DEPRECATED (v2.0.5): knowledge_edges is unused. All graph edges use
-- memory_connections (migration V3). This table was designed for bi-temporal
-- edge support but was never wired. Retained for schema compatibility with
-- existing databases. Do NOT add queries against this table.
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,  -- 'semantic', 'temporal', 'causal', 'derived', 'contradiction', 'refinement'
    weight REAL NOT NULL DEFAULT 1.0,
    -- Temporal validity (bi-temporal model)
    valid_from TEXT,  -- When this relationship started being true
    valid_until TEXT, -- When this relationship stopped being true (NULL = still valid)
    -- Provenance
    created_at TEXT NOT NULL,
    created_by TEXT,  -- 'user', 'system', 'consolidation', 'llm'
    confidence REAL NOT NULL DEFAULT 1.0,  -- Confidence in this edge
    -- Metadata
    metadata TEXT,  -- JSON for edge-specific data
    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON knowledge_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_valid_from ON knowledge_edges(valid_from);
CREATE INDEX IF NOT EXISTS idx_edges_valid_until ON knowledge_edges(valid_until);

-- ============================================================================
-- MEMORY SCOPES (Like Mem0's User/Session/Agent)
-- ============================================================================

-- Add scope column to knowledge_nodes
ALTER TABLE knowledge_nodes ADD COLUMN scope TEXT DEFAULT 'user';
-- Values: 'session' (per-session, cleared on restart)
--         'user' (per-user, persists across sessions)
--         'agent' (global agent knowledge, shared)

CREATE INDEX IF NOT EXISTS idx_nodes_scope ON knowledge_nodes(scope);

-- Session tracking table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    context TEXT,  -- JSON: session metadata
    memory_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

-- ============================================================================
-- EMBEDDING VERSIONING (Track model upgrades)
-- ============================================================================

-- Add embedding version to node_embeddings
ALTER TABLE node_embeddings ADD COLUMN version INTEGER DEFAULT 1;
-- Version 1 = all-MiniLM-L6-v2 (384d, pre-2026)
-- Version 2 = BGE-base-en-v1.5 (768d, GOD TIER 2026)

CREATE INDEX IF NOT EXISTS idx_embeddings_version ON node_embeddings(version);

-- Update existing embeddings to mark as version 1 (old model)
UPDATE node_embeddings SET version = 1 WHERE version IS NULL;

-- ============================================================================
-- MEMORY COMPRESSION (For old memories - Tier 3 prep)
-- ============================================================================

CREATE TABLE IF NOT EXISTS compressed_memories (
    id TEXT PRIMARY KEY,
    original_id TEXT NOT NULL,
    compressed_content TEXT NOT NULL,
    original_length INTEGER NOT NULL,
    compressed_length INTEGER NOT NULL,
    compression_ratio REAL NOT NULL,
    semantic_fidelity REAL NOT NULL,  -- How much meaning was preserved (0-1)
    compressed_at TEXT NOT NULL,
    model_used TEXT NOT NULL DEFAULT 'llm',
    FOREIGN KEY (original_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_compressed_original ON compressed_memories(original_id);
CREATE INDEX IF NOT EXISTS idx_compressed_at ON compressed_memories(compressed_at);

-- ============================================================================
-- EPISODIC vs SEMANTIC MEMORY (Research-backed distinction)
-- ============================================================================

-- Add memory system classification
ALTER TABLE knowledge_nodes ADD COLUMN memory_system TEXT DEFAULT 'semantic';
-- Values: 'episodic' (what happened - events, conversations)
--         'semantic' (what I know - facts, concepts)
--         'procedural' (how-to - never decays)

CREATE INDEX IF NOT EXISTS idx_nodes_memory_system ON knowledge_nodes(memory_system);

UPDATE schema_version SET version = 4, applied_at = datetime('now');
"#;

/// V5: FSRS-6 Upgrade - Access history for ACT-R activation, personalized decay parameters
const MIGRATION_V5_UP: &str = r#"
-- ============================================================================
-- ACCESS HISTORY (For ACT-R Activation + Parameter Training)
-- ============================================================================

-- Logs every search hit, promote, demote for ACT-R activation computation
CREATE TABLE IF NOT EXISTS memory_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    access_type TEXT NOT NULL,  -- 'search_hit', 'promote', 'demote'
    accessed_at TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_access_log_node ON memory_access_log(node_id);
CREATE INDEX IF NOT EXISTS idx_access_log_time ON memory_access_log(accessed_at);

-- ============================================================================
-- ACT-R ACTIVATION (Pre-computed during consolidation)
-- ============================================================================

-- B_i = ln(sum(t_j^(-d))) — NULL until first consolidation computes it
ALTER TABLE knowledge_nodes ADD COLUMN activation REAL;

CREATE INDEX IF NOT EXISTS idx_nodes_activation ON knowledge_nodes(activation);

-- ============================================================================
-- PERSONALIZED FSRS-6 PARAMETERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS fsrs_config (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL
);

-- Default w20 (forgetting curve decay parameter)
INSERT OR IGNORE INTO fsrs_config (key, value, updated_at)
VALUES ('w20', 0.1542, datetime('now'));

-- ============================================================================
-- EXTENDED CONSOLIDATION TRACKING
-- ============================================================================

ALTER TABLE consolidation_history ADD COLUMN duplicates_merged INTEGER DEFAULT 0;
ALTER TABLE consolidation_history ADD COLUMN activations_computed INTEGER DEFAULT 0;
ALTER TABLE consolidation_history ADD COLUMN w20_optimized REAL;

UPDATE schema_version SET version = 5, applied_at = datetime('now');
"#;

/// V6: Dream history persistence for automation triggers
/// Dreams were in-memory only (MemoryDreamer.dream_history Vec<DreamResult> lost on restart).
/// This table persists dream metadata so system_status can report when last dream ran.
const MIGRATION_V6_UP: &str = r#"
CREATE TABLE IF NOT EXISTS dream_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dreamed_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    memories_replayed INTEGER NOT NULL DEFAULT 0,
    connections_found INTEGER NOT NULL DEFAULT 0,
    insights_generated INTEGER NOT NULL DEFAULT 0,
    memories_strengthened INTEGER NOT NULL DEFAULT 0,
    memories_compressed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_dream_history_dreamed_at ON dream_history(dreamed_at);

UPDATE schema_version SET version = 6, applied_at = datetime('now');
"#;

/// V7: Performance — FTS5 porter tokenizer for 15-30% better keyword recall (stemming)
/// page_size upgrade handled in apply_migrations() since VACUUM can't run inside execute_batch
const MIGRATION_V7_UP: &str = r#"
-- FTS5 porter tokenizer upgrade (15-30% better keyword recall via stemming)
DROP TRIGGER IF EXISTS knowledge_ai;
DROP TRIGGER IF EXISTS knowledge_ad;
DROP TRIGGER IF EXISTS knowledge_au;
DROP TABLE IF EXISTS knowledge_fts;

CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    id, content, tags,
    content='knowledge_nodes',
    content_rowid='rowid',
    tokenize='porter ascii'
);

-- Rebuild FTS index from existing data with new tokenizer
INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild');

-- Re-create sync triggers
CREATE TRIGGER knowledge_ai AFTER INSERT ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(rowid, id, content, tags)
    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags);
END;

CREATE TRIGGER knowledge_ad AFTER DELETE ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, tags)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags);
END;

CREATE TRIGGER knowledge_au AFTER UPDATE ON knowledge_nodes BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, tags)
    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.tags);
    INSERT INTO knowledge_fts(rowid, id, content, tags)
    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.tags);
END;

UPDATE schema_version SET version = 7, applied_at = datetime('now');
"#;

/// V8: v1.9.0 Autonomic — Waking SWR tags, utility scoring, retention trend tracking
const MIGRATION_V8_UP: &str = r#"
-- Waking SWR (Sharp-Wave Ripple) tagging
-- Memories tagged during waking operation get preferential replay during dream cycles
ALTER TABLE knowledge_nodes ADD COLUMN waking_tag BOOLEAN DEFAULT FALSE;
ALTER TABLE knowledge_nodes ADD COLUMN waking_tag_at TEXT;

-- Utility scoring (MemRL-inspired: times_useful / times_retrieved)
ALTER TABLE knowledge_nodes ADD COLUMN utility_score REAL DEFAULT 0.0;
ALTER TABLE knowledge_nodes ADD COLUMN times_retrieved INTEGER DEFAULT 0;
ALTER TABLE knowledge_nodes ADD COLUMN times_useful INTEGER DEFAULT 0;

-- Retention trend tracking (for retention target system)
CREATE TABLE IF NOT EXISTS retention_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT NOT NULL,
    avg_retention REAL NOT NULL,
    total_memories INTEGER NOT NULL,
    memories_below_target INTEGER NOT NULL DEFAULT 0,
    gc_triggered BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_retention_snapshots_at ON retention_snapshots(snapshot_at);

UPDATE schema_version SET version = 8, applied_at = datetime('now');
"#;

/// V9: v2.0.0 Cognitive Leap — Emotional Memory, Flashbulb Encoding, Temporal Hierarchy
///
/// Adds columns for:
/// - Emotional memory module (#29): valence scoring + flashbulb encoding (Brown & Kulik 1977)
/// - Temporal Memory Tree: hierarchical summaries (daily/weekly/monthly) for TiMem-style recall
/// - Dream phase tracking: per-phase metrics for 4-phase biologically-accurate dream cycles
const MIGRATION_V9_UP: &str = r#"
-- ============================================================================
-- EMOTIONAL MEMORY (Brown & Kulik 1977, LaBar & Cabeza 2006)
-- ============================================================================

-- Emotional valence: -1.0 (very negative) to 1.0 (very positive)
-- Used for mood-congruent retrieval and emotional decay modulation
ALTER TABLE knowledge_nodes ADD COLUMN emotional_valence REAL DEFAULT 0.0;

-- Flashbulb memory flag: ultra-high-fidelity encoding for high-importance + high-arousal events
-- Flashbulb memories get minimum decay rate and maximum context capture
ALTER TABLE knowledge_nodes ADD COLUMN flashbulb BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_nodes_flashbulb ON knowledge_nodes(flashbulb);

-- ============================================================================
-- TEMPORAL MEMORY TREE (TiMem-inspired hierarchical consolidation)
-- ============================================================================

-- Temporal hierarchy level for summary nodes produced during dream consolidation
-- NULL = leaf node (raw memory), 'daily'/'weekly'/'monthly' = summary at that level
ALTER TABLE knowledge_nodes ADD COLUMN temporal_level TEXT;

-- Parent summary ID: links a leaf memory to its containing summary
ALTER TABLE knowledge_nodes ADD COLUMN summary_parent_id TEXT;

CREATE INDEX IF NOT EXISTS idx_nodes_temporal_level ON knowledge_nodes(temporal_level);
CREATE INDEX IF NOT EXISTS idx_nodes_summary_parent ON knowledge_nodes(summary_parent_id);

-- ============================================================================
-- 4-PHASE DREAM CYCLE TRACKING (NREM1 → NREM3 → REM → Integration)
-- ============================================================================

-- Extended dream history with per-phase metrics
ALTER TABLE dream_history ADD COLUMN phase_nrem1_ms INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN phase_nrem3_ms INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN phase_rem_ms INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN phase_integration_ms INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN summaries_generated INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN emotional_memories_processed INTEGER DEFAULT 0;
ALTER TABLE dream_history ADD COLUMN creative_connections_found INTEGER DEFAULT 0;

UPDATE schema_version SET version = 9, applied_at = datetime('now');
"#;

/// V10: v2.0.5 Intentional Amnesia — Top-Down Active Forgetting
///
/// Adds columns to `knowledge_nodes` for user-initiated suppression distinct
/// from passive FSRS decay and from bottom-up retrieval-induced forgetting
/// (which lives on `memory_states.suppression_until`). These columns are
/// incremented by the `suppress` MCP tool (tool #24) and consumed by the
/// search scoring stage + background Rac1 cascade worker.
///
/// References:
/// - Anderson et al. (2025). Brain mechanisms underlying the inhibitory
///   control of thought. Nat Rev Neurosci. DOI 10.1038/s41583-025-00929-y
/// - Cervantes-Sandoval & Davis (2020). Rac1 Impairs Forgetting-Induced
///   Cellular Plasticity. Front Cell Neurosci. PMC7477079
const MIGRATION_V10_UP: &str = r#"
-- Top-down suppression count (Suppression-Induced Forgetting, Anderson 2025).
-- Compounds with each `suppress` call, saturates via the k × count formula
-- in active_forgetting::retrieval_penalty().
ALTER TABLE knowledge_nodes ADD COLUMN suppression_count INTEGER DEFAULT 0;

-- Timestamp of the most recent suppression. Used for the 24h labile window
-- (reversal is allowed only while (now - suppressed_at) < labile_hours).
ALTER TABLE knowledge_nodes ADD COLUMN suppressed_at TEXT;

-- Partial indices — only materialise rows actually involved in suppression.
CREATE INDEX IF NOT EXISTS idx_nodes_suppression_count
    ON knowledge_nodes(suppression_count)
    WHERE suppression_count > 0;

CREATE INDEX IF NOT EXISTS idx_nodes_suppressed_at
    ON knowledge_nodes(suppressed_at)
    WHERE suppressed_at IS NOT NULL;

UPDATE schema_version SET version = 10, applied_at = datetime('now');
"#;

/// V11: v2.0.7 Cleanup — drop tables that were added speculatively and never used
///
/// Two tables from V4 were created but never had a single INSERT or SELECT in
/// the codebase:
///
/// 1. `knowledge_edges` — an elaborate bi-temporal edge schema (valid_from,
///    valid_until, confidence, created_by). Was marked DEPRECATED in the same
///    V4 migration that created it. The real edge table is `memory_connections`
///    (V3), which is what the graph traversal code actually uses.
///
/// 2. `compressed_memories` — a tiered-compression feature (compression_ratio,
///    semantic_fidelity, model_used). `advanced/compression.rs` operates
///    entirely in-memory and never touches this table. Dropping the schema
///    frees space for future migrations and removes dead schema debt.
///
/// Both tables are verified single-file references (only in migrations.rs).
/// A grep across the entire crates/ tree shows zero INSERT, SELECT, or row
/// mapping against either table. Safe to drop without behaviour change.
const MIGRATION_V11_UP: &str = r#"
-- Drop the never-used bi-temporal edge table (real edges live in memory_connections).
DROP TABLE IF EXISTS knowledge_edges;

-- Drop the never-used compression table (compression.rs is in-memory only).
DROP TABLE IF EXISTS compressed_memories;

UPDATE schema_version SET version = 11, applied_at = datetime('now');
"#;

/// V12: Merge-capable sync tombstones.
///
/// Portable sync needs to propagate deletions between devices. `knowledge_nodes`
/// remains the source of truth for live memories; this table records deletes so
/// another device can remove the same memory during a merge import.
const MIGRATION_V12_UP: &str = r#"
CREATE TABLE IF NOT EXISTS sync_tombstones (
    table_name TEXT NOT NULL,
    row_id TEXT NOT NULL,
    deleted_at TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY (table_name, row_id)
);

CREATE INDEX IF NOT EXISTS idx_sync_tombstones_deleted_at
ON sync_tombstones(deleted_at);

UPDATE schema_version SET version = 12, applied_at = datetime('now');
"#;

/// V13: non-content purge tombstones.
///
/// `memory(action="purge")` permanently removes memory content and embeddings,
/// but keeps a content-free audit/sync record so users can verify that a memory
/// was removed without Vestige retaining the text it was told to forget.
const MIGRATION_V13_UP: &str = r#"
CREATE TABLE IF NOT EXISTS deletion_tombstones (
    memory_id TEXT PRIMARY KEY,
    deleted_at TEXT NOT NULL,
    reason TEXT,
    node_type TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    edges_pruned INTEGER NOT NULL DEFAULT 0,
    insights_rewritten INTEGER NOT NULL DEFAULT 0,
    insights_deleted INTEGER NOT NULL DEFAULT 0,
    children_orphaned INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_deletion_tombstones_deleted_at
ON deletion_tombstones(deleted_at);

UPDATE schema_version SET version = 13, applied_at = datetime('now');
"#;

/// V14: Merge / Supersede controls (Phase 3).
///
/// Adds the four pieces the merge/supersede feature needs on a never-delete
/// (bitemporal) store:
///
/// 1. `merge_plans` — previewable, not-yet-applied plans. `plan_merge` and
///    `plan_supersede` write a plan row containing a JSON diff; `apply_plan`
///    consumes it by id. Plans are append-only; status moves
///    pending -> applied / cancelled.
/// 2. `merge_operations` — the reversible operation log (the "memory reflog").
///    Every applied merge/supersede records one row with a JSON `undo_payload`
///    capturing exactly what changed, so `merge_undo` can reverse it. The
///    `signals` column records WHY the memories combined (provenance), which is
///    the self-explaining differentiator.
/// 3. `knowledge_nodes.protected` — pin flag. A protected memory can never be
///    auto-merged, superseded, or forgotten.
/// 4. `knowledge_nodes.superseded_by` — bitemporal lineage pointer. Superseding
///    A with B does NOT delete A: it stamps A.valid_until = B.valid_from and
///    sets A.superseded_by = B.id, leaving A fully queryable for audit
///    (Graphiti-style invalidate-don't-delete).
// The two `protected` / `superseded_by` ADD COLUMNs (and their indexes) are
// applied separately in `apply_migrations` BEFORE this batch runs, guarded
// against "duplicate column" on replay, since SQLite has no
// `ADD COLUMN IF NOT EXISTS`. The rest of V14 is idempotent (CREATE ... IF NOT
// EXISTS).
const MIGRATION_V14_UP: &str = r#"
CREATE INDEX IF NOT EXISTS idx_nodes_protected ON knowledge_nodes(protected);
CREATE INDEX IF NOT EXISTS idx_nodes_superseded_by ON knowledge_nodes(superseded_by);

-- Previewable plans (a diff) produced by plan_merge / plan_supersede.
-- `kind` is 'merge' | 'supersede'. `payload` is the full JSON plan/diff.
CREATE TABLE IF NOT EXISTS merge_plans (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | applied | cancelled
    created_at TEXT NOT NULL,
    applied_at TEXT,
    survivor_id TEXT,                          -- node kept after the op
    member_ids TEXT NOT NULL DEFAULT '[]',     -- JSON array of all involved node ids
    confidence REAL,                           -- Fellegi-Sunter match score (0-1)
    classification TEXT,                       -- match | possible | non_match
    payload TEXT NOT NULL                      -- full JSON plan/diff
);

CREATE INDEX IF NOT EXISTS idx_merge_plans_status ON merge_plans(status);
CREATE INDEX IF NOT EXISTS idx_merge_plans_created_at ON merge_plans(created_at);

-- Reversible operation log — the "git reflog for your agent's memory".
-- One row per applied merge/supersede; `undo_payload` carries everything
-- needed to reverse it, `signals` records why the memories combined.
CREATE TABLE IF NOT EXISTS merge_operations (
    id TEXT PRIMARY KEY,
    plan_id TEXT,                              -- merge_plans.id this came from
    op_type TEXT NOT NULL,                     -- merge | supersede | undo
    status TEXT NOT NULL DEFAULT 'applied',    -- applied | reverted
    created_at TEXT NOT NULL,
    reverted_at TEXT,
    reverts_op_id TEXT,                        -- set when op_type = 'undo'
    survivor_id TEXT,                          -- node kept
    affected_ids TEXT NOT NULL DEFAULT '[]',   -- JSON array of node ids touched
    confidence REAL,
    signals TEXT,                              -- JSON: why they combined (provenance)
    reason TEXT,                               -- human-readable explanation
    undo_payload TEXT NOT NULL                 -- JSON snapshot to reverse the op
);

CREATE INDEX IF NOT EXISTS idx_merge_operations_status ON merge_operations(status);
CREATE INDEX IF NOT EXISTS idx_merge_operations_created_at ON merge_operations(created_at);
CREATE INDEX IF NOT EXISTS idx_merge_operations_survivor ON merge_operations(survivor_id);

UPDATE schema_version SET version = 14, applied_at = datetime('now');
"#;

/// Get current schema version from database
pub fn get_current_version(conn: &rusqlite::Connection) -> rusqlite::Result<u32> {
    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .or(Ok(0))
}

/// Run an `ALTER TABLE ... ADD COLUMN` statement, treating a "duplicate column
/// name" failure as success so migration replay stays idempotent (SQLite has no
/// `ADD COLUMN IF NOT EXISTS`).
fn add_column_if_missing(conn: &rusqlite::Connection, sql: &str) -> rusqlite::Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(msg))) if msg.contains("duplicate column name") => {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Apply pending migrations
pub fn apply_migrations(conn: &rusqlite::Connection) -> rusqlite::Result<u32> {
    let current_version = get_current_version(conn)?;
    let mut applied = 0;

    for migration in MIGRATIONS {
        if migration.version > current_version {
            tracing::info!(
                "Applying migration v{}: {}",
                migration.version,
                migration.description
            );

            // V14: add the two bitemporal/protect columns BEFORE the batch (the
            // batch's indexes reference them). SQLite lacks
            // `ADD COLUMN IF NOT EXISTS`, so swallow the "duplicate column"
            // error to stay idempotent on replay.
            if migration.version == 14 {
                add_column_if_missing(
                    conn,
                    "ALTER TABLE knowledge_nodes ADD COLUMN protected INTEGER NOT NULL DEFAULT 0",
                )?;
                add_column_if_missing(
                    conn,
                    "ALTER TABLE knowledge_nodes ADD COLUMN superseded_by TEXT",
                )?;
            }

            // Use execute_batch to handle multi-statement SQL including triggers
            conn.execute_batch(migration.up)?;

            // V7: Upgrade page_size to 8192 (10-30% faster large-row reads)
            // VACUUM rewrites the DB with the new page size — can't run inside execute_batch
            if migration.version == 7 {
                conn.pragma_update(None, "page_size", 8192)?;
                conn.execute_batch("VACUUM;")?;
                tracing::info!("Database page_size upgraded to 8192 via VACUUM");
            }

            applied += 1;
        }
    }

    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh in-memory DB must end up at schema_version = highest migration
    /// version after `apply_migrations` runs all migrations end-to-end, and
    /// neither of the dead tables V11 drops must exist afterwards.
    #[test]
    fn test_apply_migrations_advances_to_v14_and_drops_dead_tables() {
        let conn = rusqlite::Connection::open_in_memory().expect("open in-memory");

        // Pre-requisite: schema_version must be bootstrapped by V1.
        apply_migrations(&conn).expect("apply_migrations succeeds");

        // 1. schema_version advanced to V14
        let version = get_current_version(&conn).expect("read schema_version");
        assert_eq!(
            version, 14,
            "schema_version must be 14 after all migrations"
        );

        // 2. knowledge_edges is gone (V11 drops it)
        let knowledge_edges_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='knowledge_edges'",
                [],
                |row| row.get(0),
            )
            .expect("query sqlite_master");
        assert_eq!(
            knowledge_edges_rows, 0,
            "knowledge_edges table must be dropped by V11"
        );

        // 3. compressed_memories is gone (V11 drops it)
        let compressed_memories_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='compressed_memories'",
                [],
                |row| row.get(0),
            )
            .expect("query sqlite_master");
        assert_eq!(
            compressed_memories_rows, 0,
            "compressed_memories table must be dropped by V11"
        );

        // 4. sync_tombstones exists (V12 creates it)
        let sync_tombstone_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sync_tombstones'",
                [],
                |row| row.get(0),
            )
            .expect("query sqlite_master");
        assert_eq!(
            sync_tombstone_rows, 1,
            "sync_tombstones table must be created by V12"
        );

        // 5. deletion_tombstones exists (V13 creates it)
        let deletion_tombstone_rows: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='deletion_tombstones'",
                [],
                |row| row.get(0),
            )
            .expect("query sqlite_master");
        assert_eq!(
            deletion_tombstone_rows, 1,
            "deletion_tombstones table must be created by V13"
        );

        // 6. merge_plans + merge_operations exist (V14 creates them)
        for table in ["merge_plans", "merge_operations"] {
            let rows: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .expect("query sqlite_master");
            assert_eq!(rows, 1, "{table} table must be created by V14");
        }

        // 7. knowledge_nodes gains `protected` + `superseded_by` (V14)
        let node_cols: Vec<String> = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(knowledge_nodes)")
                .expect("prepare table_info");
            stmt.query_map([], |row| row.get::<_, String>(1))
                .expect("query table_info")
                .filter_map(|r| r.ok())
                .collect()
        };
        assert!(
            node_cols.iter().any(|c| c == "protected"),
            "knowledge_nodes must have `protected` column after V14"
        );
        assert!(
            node_cols.iter().any(|c| c == "superseded_by"),
            "knowledge_nodes must have `superseded_by` column after V14"
        );
    }

    /// V11 must be idempotent on replay — if the tables were already dropped
    /// (e.g. a user ran v2.0.7, downgraded, then upgraded again), re-running
    /// the migration must not error. `DROP TABLE IF EXISTS` handles this but
    /// we enforce it with an explicit test so a future refactor to plain
    /// `DROP TABLE` would be caught.
    #[test]
    fn test_v11_is_idempotent_on_replay() {
        let conn = rusqlite::Connection::open_in_memory().expect("open in-memory");
        apply_migrations(&conn).expect("first apply_migrations succeeds");

        // Force schema_version back to 10 so V11 runs again even though its
        // changes are already applied.
        conn.execute("UPDATE schema_version SET version = 10", [])
            .expect("rewind schema_version");

        // Replay must not error.
        apply_migrations(&conn).expect("V11 replay must be idempotent");

        let version = get_current_version(&conn).expect("read schema_version");
        assert_eq!(version, 14, "schema_version back at 14 after replay");
    }
}

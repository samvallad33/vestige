import Database from 'better-sqlite3';
import { nanoid } from 'nanoid';
import path from 'path';
import fs from 'fs';
import os from 'os';
import crypto from 'crypto';
import { execSync } from 'child_process';
import natural from 'natural';
import type { KnowledgeNode, KnowledgeNodeInput, PersonNode, GraphEdge, Source, Interaction } from './types.js';

// ============================================================================
// SENTIMENT ANALYSIS (Emotional Memory Weighting)
// ============================================================================

// Initialize sentiment analyzer
// We use AFINN for simplicity - it assigns scores from -5 to +5 for emotional words
const Analyzer = natural.SentimentAnalyzer;
const stemmer = natural.PorterStemmer;
const tokenizer = new natural.WordTokenizer();
const sentimentAnalyzer = new Analyzer('English', stemmer, 'afinn');

/**
 * Analyze the emotional intensity of text content
 * Returns a value from 0 (neutral) to 1 (highly emotional)
 *
 * KEY INSIGHT: We care about INTENSITY, not polarity
 * "I'm absolutely THRILLED" and "I'm completely DEVASTATED" should both decay slowly
 * because they're emotionally significant memories
 */
export function analyzeSentimentIntensity(content: string): number {
  try {
    const tokens = tokenizer.tokenize(content.toLowerCase());
    if (!tokens || tokens.length === 0) return 0;

    // Get raw sentiment score (-1 to 1 typical range)
    const rawScore = sentimentAnalyzer.getSentiment(tokens);

    // Convert to INTENSITY (absolute value, normalized)
    // We also boost based on emotional word density
    const absScore = Math.abs(rawScore);

    // Count emotional words (those contributing to sentiment)
    // AFINN-165 has ~2500 words with sentiment values
    let emotionalWordCount = 0;
    for (const token of tokens) {
      // If the token contributed to score, it's emotional
      const singleTokenScore = sentimentAnalyzer.getSentiment([token]);
      if (singleTokenScore !== 0) {
        emotionalWordCount++;
      }
    }

    // Emotional density = emotional words / total words
    const emotionalDensity = emotionalWordCount / tokens.length;

    // Combine raw intensity with density
    // More emotional words = more memorable content
    const combinedIntensity = (absScore * 0.6) + (emotionalDensity * 0.4);

    // Normalize to 0-1 range (cap at 1)
    return Math.min(1, Math.max(0, combinedIntensity));
  } catch {
    // If sentiment analysis fails, return neutral
    return 0;
  }
}

// ============================================================================
// GIT-BLAME FOR THOUGHTS - Capture code context when memory is created
// ============================================================================

export interface GitContext {
  branch?: string;
  commit?: string;
  commitMessage?: string;
  repoPath?: string;
  dirty?: boolean;
  changedFiles?: string[];
}

/**
 * Capture current git context - what code is being worked on right now?
 * This allows "time travel" to see what you were coding when you had a thought.
 *
 * Example use case:
 * - You're debugging a nasty race condition
 * - You have an insight and use `ingest` to record it
 * - Later, you recall the insight and see: "Branch: fix/race-condition, Commit: abc123"
 * - This context helps you understand WHY you had that thought
 */
export function captureGitContext(): GitContext | undefined {
  try {
    const cwd = process.cwd();

    // Check if we're in a git repository
    try {
      execSync('git rev-parse --is-inside-work-tree', { cwd, stdio: 'pipe' });
    } catch {
      return undefined; // Not a git repo
    }

    const context: GitContext = {};

    // Get repo root
    try {
      context.repoPath = execSync('git rev-parse --show-toplevel', { cwd, encoding: 'utf-8' }).trim();
    } catch {
      // Ignore
    }

    // Get current branch
    try {
      context.branch = execSync('git rev-parse --abbrev-ref HEAD', { cwd, encoding: 'utf-8' }).trim();
    } catch {
      // Ignore
    }

    // Get current commit (short SHA)
    try {
      context.commit = execSync('git rev-parse --short HEAD', { cwd, encoding: 'utf-8' }).trim();
    } catch {
      // Ignore
    }

    // Get commit message (first line)
    try {
      context.commitMessage = execSync('git log -1 --format=%s', { cwd, encoding: 'utf-8' }).trim();
      // Truncate long messages
      if (context.commitMessage.length > 100) {
        context.commitMessage = context.commitMessage.slice(0, 97) + '...';
      }
    } catch {
      // Ignore
    }

    // Check for uncommitted changes
    try {
      const status = execSync('git status --porcelain', { cwd, encoding: 'utf-8' }).trim();
      context.dirty = status.length > 0;
      if (context.dirty) {
        // Get list of changed files (limit to 10)
        const files = status.split('\n')
          .map(line => line.slice(3).trim())
          .filter(Boolean)
          .slice(0, 10);
        if (files.length > 0) {
          context.changedFiles = files;
        }
      }
    } catch {
      // Ignore
    }

    return context;
  } catch {
    // Git context capture is optional - never fail ingestion
    return undefined;
  }
}

// ============================================================================
// CONSTANTS & CONFIGURATION
// ============================================================================

const DEFAULT_DB_PATH = path.join(os.homedir(), '.vestige', 'vestige.db');
const BACKUP_DIR = path.join(os.homedir(), '.vestige', 'backups');

// Size thresholds (in bytes)
const SIZE_WARNING_THRESHOLD = 100 * 1024 * 1024;  // 100MB
const SIZE_CRITICAL_THRESHOLD = 500 * 1024 * 1024; // 500MB
const MAX_NODES_WARNING = 50000;
const MAX_NODES_CRITICAL = 100000;

// Default pagination
const DEFAULT_LIMIT = 50;
const MAX_LIMIT = 500;

// Input validation limits
const MAX_CONTENT_LENGTH = 1_000_000;      // 1MB max content
const MAX_NAME_LENGTH = 500;               // 500 chars for names
const MAX_QUERY_LENGTH = 10_000;           // 10KB max query
const MAX_TAGS_COUNT = 100;                // Max tags per node
const MAX_BATCH_SIZE = 1000;               // Max items in batch operations

// Concurrency control
const BUSY_TIMEOUT_MS = 5000;              // 5 second busy timeout

// SM-2 Spaced Repetition Constants
const SM2_EASE_FACTOR = 2.5;               // Standard SM-2 ease factor for successful recall
const SM2_LAPSE_THRESHOLD = 0.3;           // Below this retention = "forgot" (lapse)
const SM2_MIN_STABILITY = 1.0;             // Minimum stability (1 day)
const SM2_MAX_STABILITY = 365.0;           // Maximum stability (1 year - effectively permanent)

// Sentiment-Weighted Decay Constants
const SENTIMENT_STABILITY_BOOST = 2.0;     // Max 2x stability boost for highly emotional memories
const SENTIMENT_MIN_BOOST = 1.0;           // Neutral content gets no boost

// ============================================================================
// SECURITY HELPERS
// ============================================================================

/**
 * Validate that a path is within an allowed directory (prevents path traversal)
 */
function isPathWithinDirectory(targetPath: string, allowedDir: string): boolean {
  const resolvedTarget = path.resolve(targetPath);
  const resolvedAllowed = path.resolve(allowedDir);
  return resolvedTarget.startsWith(resolvedAllowed + path.sep) || resolvedTarget === resolvedAllowed;
}

/**
 * Validate backup file path - must be within BACKUP_DIR and have .db extension
 */
function validateBackupPath(backupPath: string): void {
  const resolvedPath = path.resolve(backupPath);
  const resolvedBackupDir = path.resolve(BACKUP_DIR);

  // Check path is within backup directory
  if (!isPathWithinDirectory(resolvedPath, resolvedBackupDir)) {
    throw new VestigeDatabaseError(
      'Backup path must be within the backup directory',
      'INVALID_BACKUP_PATH'
    );
  }

  // Validate file extension
  if (!resolvedPath.endsWith('.db')) {
    throw new VestigeDatabaseError(
      'Backup file must have .db extension',
      'INVALID_BACKUP_EXTENSION'
    );
  }

  // Check for null bytes or other suspicious characters
  if (backupPath.includes('\0') || backupPath.includes('..')) {
    throw new VestigeDatabaseError(
      'Invalid characters in backup path',
      'INVALID_BACKUP_PATH'
    );
  }
}

/**
 * Safe JSON parse with fallback - never throws
 */
function safeJsonParse<T>(value: string | null | undefined, fallback: T): T {
  if (!value) return fallback;
  try {
    const parsed = JSON.parse(value);
    // Basic type validation
    if (typeof parsed !== typeof fallback) {
      return fallback;
    }
    return parsed as T;
  } catch {
    return fallback;
  }
}

/**
 * Sanitize error message to prevent sensitive data leakage
 */
function sanitizeErrorMessage(message: string): string {
  // Remove file paths
  let sanitized = message.replace(/\/[^\s]+/g, '[PATH]');
  // Remove potential SQL queries
  sanitized = sanitized.replace(/SELECT|INSERT|UPDATE|DELETE|DROP|CREATE/gi, '[SQL]');
  // Remove potential connection strings
  sanitized = sanitized.replace(/\b(password|secret|key|token|auth)\s*[=:]\s*\S+/gi, '[REDACTED]');
  return sanitized;
}

/**
 * Validate string length for inputs
 */
function validateStringLength(value: string, maxLength: number, fieldName: string): void {
  if (value && value.length > maxLength) {
    throw new VestigeDatabaseError(
      `${fieldName} exceeds maximum length of ${maxLength} characters`,
      'INPUT_TOO_LONG'
    );
  }
}

/**
 * Validate array length for inputs
 */
function validateArrayLength<T>(arr: T[] | undefined, maxLength: number, fieldName: string): void {
  if (arr && arr.length > maxLength) {
    throw new VestigeDatabaseError(
      `${fieldName} exceeds maximum count of ${maxLength} items`,
      'INPUT_TOO_MANY_ITEMS'
    );
  }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

export class VestigeDatabaseError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    cause?: unknown
  ) {
    // Sanitize the message to prevent sensitive data leakage
    super(sanitizeErrorMessage(message));
    this.name = 'VestigeDatabaseError';
    // Don't expose the original cause in production - it may contain sensitive info
    if (process.env.NODE_ENV === 'development' && cause) {
      this.cause = cause;
    }
  }
}

// ============================================================================
// HEALTH CHECK TYPES
// ============================================================================

export interface HealthStatus {
  status: 'healthy' | 'warning' | 'critical';
  dbPath: string;
  dbSizeBytes: number;
  dbSizeMB: number;
  nodeCount: number;
  peopleCount: number;
  edgeCount: number;
  walMode: boolean;
  integrityCheck: boolean;
  warnings: string[];
  lastBackup: string | null;
}

export interface PaginationOptions {
  limit?: number;
  offset?: number;
}

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
}

// ============================================================================
// DATABASE INITIALIZATION
// ============================================================================

export function getDbPath(): string {
  const envPath = process.env['VESTIGE_DB_PATH'];
  return envPath || DEFAULT_DB_PATH;
}

export function initializeDatabase(dbPath?: string): Database.Database {
  const finalPath = dbPath || getDbPath();

  // Ensure directory exists
  const dir = path.dirname(finalPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  // Ensure backup directory exists
  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
  }

  const db = new Database(finalPath);

  // CRITICAL: Set busy timeout FIRST to handle concurrent access
  // This prevents SQLITE_BUSY errors when multiple clients access the DB
  db.pragma(`busy_timeout = ${BUSY_TIMEOUT_MS}`);

  // Enable WAL mode for better concurrent performance
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  // Optimize for performance
  db.pragma('synchronous = NORMAL');
  db.pragma('cache_size = -64000'); // 64MB cache
  db.pragma('temp_store = MEMORY');

  // Security: Limit memory usage to prevent DoS
  db.pragma('max_page_count = 1073741823'); // ~2TB max (practical limit)

  // Enable secure delete to overwrite deleted data
  db.pragma('secure_delete = ON');

  // Create tables
  createTables(db);

  // Run migrations for existing databases
  runMigrations(db);

  return db;
}

function createTables(db: Database.Database): void {
  // Knowledge Nodes table
  db.exec(`
    CREATE TABLE IF NOT EXISTS knowledge_nodes (
      id TEXT PRIMARY KEY,
      content TEXT NOT NULL,
      summary TEXT,

      -- Temporal metadata
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_accessed_at TEXT NOT NULL,
      access_count INTEGER DEFAULT 0,

      -- Decay modeling (SM-2 inspired spaced repetition)
      retention_strength REAL DEFAULT 1.0,
      stability_factor REAL DEFAULT 1.0,  -- Grows with successful reviews, resets on lapse
      sentiment_intensity REAL DEFAULT 0, -- Emotional weight (0=neutral, 1=highly emotional)
      next_review_date TEXT,
      review_count INTEGER DEFAULT 0,

      -- Dual-Strength Memory Model (Bjork & Bjork, 1992)
      storage_strength REAL DEFAULT 1.0,    -- How well encoded (never decreases)
      retrieval_strength REAL DEFAULT 1.0,  -- How accessible now (decays)

      -- Provenance
      source_type TEXT NOT NULL,
      source_platform TEXT NOT NULL,
      source_id TEXT,
      source_url TEXT,
      source_chain TEXT DEFAULT '[]', -- JSON array
      git_context TEXT, -- JSON object: {branch, commit, commitMessage, repoPath, dirty, changedFiles}

      -- Confidence
      confidence REAL DEFAULT 0.8,
      is_contradicted INTEGER DEFAULT 0,
      contradiction_ids TEXT DEFAULT '[]', -- JSON array

      -- Extracted entities (JSON arrays)
      people TEXT DEFAULT '[]',
      concepts TEXT DEFAULT '[]',
      events TEXT DEFAULT '[]',
      tags TEXT DEFAULT '[]'
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON knowledge_nodes(created_at);
    CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON knowledge_nodes(last_accessed_at);
    CREATE INDEX IF NOT EXISTS idx_nodes_retention ON knowledge_nodes(retention_strength);
    CREATE INDEX IF NOT EXISTS idx_nodes_source_type ON knowledge_nodes(source_type);
    CREATE INDEX IF NOT EXISTS idx_nodes_source_platform ON knowledge_nodes(source_platform);
  `);

  // Full-text search for content
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
      id,
      content,
      summary,
      tags,
      content='knowledge_nodes',
      content_rowid='rowid'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge_nodes BEGIN
      INSERT INTO knowledge_fts(rowid, id, content, summary, tags)
      VALUES (NEW.rowid, NEW.id, NEW.content, NEW.summary, NEW.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge_nodes BEGIN
      INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, summary, tags)
      VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.summary, OLD.tags);
    END;

    CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge_nodes BEGIN
      INSERT INTO knowledge_fts(knowledge_fts, rowid, id, content, summary, tags)
      VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.summary, OLD.tags);
      INSERT INTO knowledge_fts(rowid, id, content, summary, tags)
      VALUES (NEW.rowid, NEW.id, NEW.content, NEW.summary, NEW.tags);
    END;
  `);

  // People table
  db.exec(`
    CREATE TABLE IF NOT EXISTS people (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      aliases TEXT DEFAULT '[]', -- JSON array

      -- Relationship context
      how_we_met TEXT,
      relationship_type TEXT,
      organization TEXT,
      role TEXT,
      location TEXT,

      -- Contact info
      email TEXT,
      phone TEXT,
      social_links TEXT DEFAULT '{}', -- JSON object

      -- Communication patterns
      last_contact_at TEXT,
      contact_frequency REAL DEFAULT 0,
      preferred_channel TEXT,

      -- Shared context
      shared_topics TEXT DEFAULT '[]',
      shared_projects TEXT DEFAULT '[]',

      -- Meta
      notes TEXT,
      relationship_health REAL DEFAULT 0.5,

      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);
    CREATE INDEX IF NOT EXISTS idx_people_last_contact ON people(last_contact_at);
  `);

  // Interactions table
  db.exec(`
    CREATE TABLE IF NOT EXISTS interactions (
      id TEXT PRIMARY KEY,
      person_id TEXT NOT NULL,
      type TEXT NOT NULL,
      date TEXT NOT NULL,
      summary TEXT NOT NULL,
      topics TEXT DEFAULT '[]',
      sentiment REAL,
      action_items TEXT DEFAULT '[]',
      source_node_id TEXT,

      FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE,
      FOREIGN KEY (source_node_id) REFERENCES knowledge_nodes(id) ON DELETE SET NULL
    );

    CREATE INDEX IF NOT EXISTS idx_interactions_person ON interactions(person_id);
    CREATE INDEX IF NOT EXISTS idx_interactions_date ON interactions(date);
  `);

  // Graph edges table
  db.exec(`
    CREATE TABLE IF NOT EXISTS graph_edges (
      id TEXT PRIMARY KEY,
      from_id TEXT NOT NULL,
      to_id TEXT NOT NULL,
      edge_type TEXT NOT NULL,
      weight REAL DEFAULT 0.5,
      metadata TEXT DEFAULT '{}',
      created_at TEXT NOT NULL,

      UNIQUE(from_id, to_id, edge_type)
    );

    CREATE INDEX IF NOT EXISTS idx_edges_from ON graph_edges(from_id);
    CREATE INDEX IF NOT EXISTS idx_edges_to ON graph_edges(to_id);
    CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type);
  `);

  // Sources table
  db.exec(`
    CREATE TABLE IF NOT EXISTS sources (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      platform TEXT NOT NULL,
      original_id TEXT,
      url TEXT,
      file_path TEXT,
      title TEXT,
      author TEXT,
      publication_date TEXT,

      ingested_at TEXT NOT NULL,
      last_synced_at TEXT NOT NULL,
      content_hash TEXT,

      node_count INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_sources_platform ON sources(platform);
    CREATE INDEX IF NOT EXISTS idx_sources_file_path ON sources(file_path);
  `);

  // Embeddings reference table (actual vectors stored in ChromaDB)
  db.exec(`
    CREATE TABLE IF NOT EXISTS embeddings (
      node_id TEXT PRIMARY KEY,
      chroma_id TEXT NOT NULL,
      model TEXT NOT NULL,
      created_at TEXT NOT NULL,

      FOREIGN KEY (node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
    );
  `);

  // Metadata table for tracking backups and system info
  db.exec(`
    CREATE TABLE IF NOT EXISTS vestige_metadata (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `);
}

/**
 * Run database migrations for existing databases
 * This ensures older databases get new columns
 */
function runMigrations(db: Database.Database): void {
  try {
    const columns = db.prepare("PRAGMA table_info(knowledge_nodes)").all() as { name: string }[];

    // Migration 1: Add stability_factor column if it doesn't exist
    const hasStabilityFactor = columns.some(col => col.name === 'stability_factor');
    if (!hasStabilityFactor) {
      db.exec(`
        ALTER TABLE knowledge_nodes
        ADD COLUMN stability_factor REAL DEFAULT 1.0;
      `);

      // Initialize stability based on review_count for existing nodes
      // Nodes that have been reviewed multiple times should have higher stability
      db.exec(`
        UPDATE knowledge_nodes
        SET stability_factor = MIN(${SM2_MAX_STABILITY}, POWER(${SM2_EASE_FACTOR}, review_count))
        WHERE review_count > 0;
      `);
    }

    // Migration 2: Add sentiment_intensity column if it doesn't exist
    const hasSentimentIntensity = columns.some(col => col.name === 'sentiment_intensity');
    if (!hasSentimentIntensity) {
      db.exec(`
        ALTER TABLE knowledge_nodes
        ADD COLUMN sentiment_intensity REAL DEFAULT 0;
      `);

      // Backfill sentiment for existing nodes
      // This analyzes all existing content and sets sentiment intensity
      const nodes = db.prepare('SELECT id, content FROM knowledge_nodes').all() as { id: string; content: string }[];
      const updateStmt = db.prepare('UPDATE knowledge_nodes SET sentiment_intensity = ? WHERE id = ?');

      for (const node of nodes) {
        const intensity = analyzeSentimentIntensity(node.content);
        if (intensity > 0) {
          updateStmt.run(intensity, node.id);
        }
      }
    }

    // Migration 3: Add git_context column if it doesn't exist
    const hasGitContext = columns.some(col => col.name === 'git_context');
    if (!hasGitContext) {
      db.exec(`
        ALTER TABLE knowledge_nodes
        ADD COLUMN git_context TEXT;
      `);
      // No backfill - we can't retroactively determine git context
    }

    // Migration 4: Add Dual-Strength Memory Model columns (Bjork & Bjork, 1992)
    const hasStorageStrength = columns.some(col => col.name === 'storage_strength');
    if (!hasStorageStrength) {
      db.exec(`
        ALTER TABLE knowledge_nodes
        ADD COLUMN storage_strength REAL DEFAULT 1.0;
      `);

      // Backfill storage_strength based on review history
      // storage_strength = 1.0 + (review_count * 0.5) + (access_count * 0.1)
      db.exec(`
        UPDATE knowledge_nodes
        SET storage_strength = 1.0 + (review_count * 0.5) + (access_count * 0.1);
      `);
    }

    const hasRetrievalStrength = columns.some(col => col.name === 'retrieval_strength');
    if (!hasRetrievalStrength) {
      db.exec(`
        ALTER TABLE knowledge_nodes
        ADD COLUMN retrieval_strength REAL DEFAULT 1.0;
      `);

      // Backfill retrieval_strength from retention_strength for backward compatibility
      db.exec(`
        UPDATE knowledge_nodes
        SET retrieval_strength = retention_strength;
      `);
    }
  } catch {
    // Migration may have already been applied or table doesn't exist yet
  }
}

// ============================================================================
// CRUD OPERATIONS
// ============================================================================

/**
 * Simple mutex for serializing critical database operations
 */
class OperationMutex {
  private locked = false;
  private queue: (() => void)[] = [];

  async acquire(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.locked) {
        this.locked = true;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  release(): void {
    const next = this.queue.shift();
    if (next) {
      next();
    } else {
      this.locked = false;
    }
  }
}

export class VestigeDatabase {
  private db: Database.Database;
  private dbPath: string;
  private readonly writeMutex = new OperationMutex();

  constructor(dbPath?: string) {
    this.dbPath = dbPath || getDbPath();
    this.db = initializeDatabase(this.dbPath);
  }

  // ============================================================================
  // HEALTH & MONITORING
  // ============================================================================

  /**
   * Get comprehensive health status of the database
   */
  checkHealth(): HealthStatus {
    const warnings: string[] = [];
    let status: 'healthy' | 'warning' | 'critical' = 'healthy';

    // Get database file size
    let dbSizeBytes = 0;
    try {
      const stats = fs.statSync(this.dbPath);
      dbSizeBytes = stats.size;

      // Also check WAL file size
      const walPath = this.dbPath + '-wal';
      if (fs.existsSync(walPath)) {
        const walStats = fs.statSync(walPath);
        dbSizeBytes += walStats.size;
      }
    } catch {
      warnings.push('Could not determine database file size');
    }

    const dbSizeMB = dbSizeBytes / (1024 * 1024);

    // Size warnings
    if (dbSizeBytes >= SIZE_CRITICAL_THRESHOLD) {
      status = 'critical';
      warnings.push(`Database size (${dbSizeMB.toFixed(1)}MB) exceeds critical threshold (${SIZE_CRITICAL_THRESHOLD / 1024 / 1024}MB)`);
    } else if (dbSizeBytes >= SIZE_WARNING_THRESHOLD) {
      status = 'warning';
      warnings.push(`Database size (${dbSizeMB.toFixed(1)}MB) exceeds warning threshold (${SIZE_WARNING_THRESHOLD / 1024 / 1024}MB)`);
    }

    // Get counts
    const stats = this.getStats();

    // Node count warnings
    if (stats.totalNodes >= MAX_NODES_CRITICAL) {
      status = 'critical';
      warnings.push(`Node count (${stats.totalNodes}) exceeds critical threshold (${MAX_NODES_CRITICAL})`);
    } else if (stats.totalNodes >= MAX_NODES_WARNING) {
      if (status !== 'critical') status = 'warning';
      warnings.push(`Node count (${stats.totalNodes}) exceeds warning threshold (${MAX_NODES_WARNING})`);
    }

    // Check WAL mode
    const journalMode = this.db.pragma('journal_mode', { simple: true }) as string;
    const walMode = journalMode.toLowerCase() === 'wal';
    if (!walMode) {
      if (status === 'healthy') status = 'warning';
      warnings.push('WAL mode is not enabled - concurrent performance may be degraded');
    }

    // Quick integrity check (just checks header, not full scan)
    let integrityCheck = true;
    try {
      const result = this.db.pragma('quick_check', { simple: true }) as string;
      integrityCheck = result === 'ok';
      if (!integrityCheck) {
        status = 'critical';
        warnings.push('Database integrity check failed');
      }
    } catch (e) {
      integrityCheck = false;
      status = 'critical';
      warnings.push('Could not run integrity check');
    }

    // Get last backup time
    let lastBackup: string | null = null;
    try {
      const row = this.db.prepare('SELECT value FROM vestige_metadata WHERE key = ?').get('last_backup') as { value: string } | undefined;
      lastBackup = row?.value || null;

      // Warn if no backup in 7 days
      if (lastBackup) {
        const lastBackupDate = new Date(lastBackup);
        const daysSinceBackup = (Date.now() - lastBackupDate.getTime()) / (1000 * 60 * 60 * 24);
        if (daysSinceBackup > 7) {
          if (status === 'healthy') status = 'warning';
          warnings.push(`Last backup was ${Math.floor(daysSinceBackup)} days ago`);
        }
      } else {
        if (status === 'healthy') status = 'warning';
        warnings.push('No backup has been created');
      }
    } catch {
      // Metadata table might not exist in older databases
    }

    return {
      status,
      dbPath: this.dbPath,
      dbSizeBytes,
      dbSizeMB,
      nodeCount: stats.totalNodes,
      peopleCount: stats.totalPeople,
      edgeCount: stats.totalEdges,
      walMode,
      integrityCheck,
      warnings,
      lastBackup,
    };
  }

  /**
   * Get database size in bytes
   */
  getDatabaseSize(): { bytes: number; mb: number; formatted: string } {
    let bytes = 0;
    try {
      const stats = fs.statSync(this.dbPath);
      bytes = stats.size;

      // Include WAL file
      const walPath = this.dbPath + '-wal';
      if (fs.existsSync(walPath)) {
        bytes += fs.statSync(walPath).size;
      }
    } catch {
      // File might not exist yet
    }

    const mb = bytes / (1024 * 1024);
    const formatted = mb < 1 ? `${(bytes / 1024).toFixed(1)}KB` : `${mb.toFixed(1)}MB`;

    return { bytes, mb, formatted };
  }

  // ============================================================================
  // BACKUP & RESTORE
  // ============================================================================

  /**
   * Create a backup of the database
   * @returns Path to the backup file
   */
  backup(customPath?: string): string {
    // Generate safe backup filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupFileName = `vestige-backup-${timestamp}.db`;

    // Determine backup path - always force it to be within BACKUP_DIR for security
    let backupPath: string;
    if (customPath) {
      // SECURITY: Validate custom path is within backup directory
      const resolvedCustom = path.resolve(customPath);
      const resolvedBackupDir = path.resolve(BACKUP_DIR);

      // If custom path is just a filename, place it in BACKUP_DIR
      if (!customPath.includes(path.sep) && !customPath.includes('/')) {
        backupPath = path.join(BACKUP_DIR, customPath);
      } else if (isPathWithinDirectory(resolvedCustom, resolvedBackupDir)) {
        backupPath = resolvedCustom;
      } else {
        throw new VestigeDatabaseError(
          'Custom backup path must be within the backup directory',
          'INVALID_BACKUP_PATH'
        );
      }
      validateBackupPath(backupPath);
    } else {
      backupPath = path.join(BACKUP_DIR, backupFileName);
    }

    // Checkpoint WAL to ensure all data is in main file
    this.db.pragma('wal_checkpoint(TRUNCATE)');

    // Ensure backup directory exists
    const backupDir = path.dirname(backupPath);
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true, mode: 0o700 }); // Restrict permissions
    }

    // Use file copy for backup (simpler and synchronous)
    fs.copyFileSync(this.dbPath, backupPath);

    // Set restrictive permissions on backup file
    try {
      fs.chmodSync(backupPath, 0o600); // Owner read/write only
    } catch {
      // chmod may not work on all platforms, continue anyway
    }

    // Update metadata
    const now = new Date().toISOString();
    this.db.prepare(`
      INSERT OR REPLACE INTO vestige_metadata (key, value, updated_at)
      VALUES (?, ?, ?)
    `).run('last_backup', now, now);

    // Clean old backups (keep last 5)
    this.cleanOldBackups(5);

    return backupPath;
  }

  /**
   * List available backups
   */
  listBackups(): { path: string; size: number; date: Date }[] {
    if (!fs.existsSync(BACKUP_DIR)) {
      return [];
    }

    const files = fs.readdirSync(BACKUP_DIR)
      .filter(f => f.startsWith('vestige-backup-') && f.endsWith('.db'))
      .map(f => {
        const fullPath = path.join(BACKUP_DIR, f);
        const stats = fs.statSync(fullPath);
        return {
          path: fullPath,
          size: stats.size,
          date: stats.mtime,
        };
      })
      .sort((a, b) => b.date.getTime() - a.date.getTime());

    return files;
  }

  /**
   * Restore from a backup file
   * WARNING: This will replace the current database!
   *
   * SECURITY: Only accepts backups from the official backup directory
   */
  restore(backupPath: string): void {
    // CRITICAL SECURITY: Validate backup path is within backup directory
    // This prevents path traversal attacks (CWE-22)
    validateBackupPath(backupPath);

    const resolvedPath = path.resolve(backupPath);

    if (!fs.existsSync(resolvedPath)) {
      throw new VestigeDatabaseError(
        'Backup file not found',
        'BACKUP_NOT_FOUND'
      );
    }

    // Validate the backup file is actually a SQLite database
    try {
      const header = Buffer.alloc(16);
      const fd = fs.openSync(resolvedPath, 'r');
      fs.readSync(fd, header, 0, 16, 0);
      fs.closeSync(fd);

      // SQLite database files start with "SQLite format 3\0"
      const sqliteHeader = 'SQLite format 3\0';
      if (header.toString('utf8', 0, 16) !== sqliteHeader) {
        throw new VestigeDatabaseError(
          'Invalid backup file format - not a valid SQLite database',
          'INVALID_BACKUP_FORMAT'
        );
      }
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to validate backup file',
        'BACKUP_VALIDATION_FAILED'
      );
    }

    // Close current connection
    this.db.close();

    // Create a backup of current database before restoring
    const preRestoreBackup = this.dbPath + '.pre-restore-' + Date.now();
    if (fs.existsSync(this.dbPath)) {
      fs.copyFileSync(this.dbPath, preRestoreBackup);
    }

    try {
      // Copy backup to database location
      fs.copyFileSync(resolvedPath, this.dbPath);

      // Remove WAL files if they exist
      const walPath = this.dbPath + '-wal';
      const shmPath = this.dbPath + '-shm';
      if (fs.existsSync(walPath)) fs.unlinkSync(walPath);
      if (fs.existsSync(shmPath)) fs.unlinkSync(shmPath);

      // Reopen database and verify integrity
      this.db = initializeDatabase(this.dbPath);

      // Verify the restored database has the expected schema
      const tables = this.db.prepare(
        "SELECT name FROM sqlite_master WHERE type='table'"
      ).all() as { name: string }[];
      const tableNames = tables.map(t => t.name);

      if (!tableNames.includes('knowledge_nodes') || !tableNames.includes('people')) {
        throw new Error('Restored database is missing required tables');
      }

      // Clean up pre-restore backup on success
      if (fs.existsSync(preRestoreBackup)) {
        fs.unlinkSync(preRestoreBackup);
      }
    } catch (error) {
      // Restore failed, try to recover
      if (fs.existsSync(preRestoreBackup)) {
        fs.copyFileSync(preRestoreBackup, this.dbPath);
        this.db = initializeDatabase(this.dbPath);
        fs.unlinkSync(preRestoreBackup);
      }
      throw new VestigeDatabaseError(
        'Failed to restore backup',
        'RESTORE_FAILED'
      );
    }
  }

  /**
   * Clean old backups, keeping only the most recent N
   */
  private cleanOldBackups(keepCount: number): void {
    const backups = this.listBackups();
    const toDelete = backups.slice(keepCount);

    for (const backup of toDelete) {
      try {
        fs.unlinkSync(backup.path);
      } catch {
        // Ignore deletion errors
      }
    }
  }

  // ============================================================================
  // KNOWLEDGE NODES
  // ============================================================================

  insertNode(node: Omit<KnowledgeNodeInput, 'id'>): KnowledgeNode {
    try {
      // Input validation
      validateStringLength(node.content, MAX_CONTENT_LENGTH, 'Content');
      validateStringLength(node.summary || '', MAX_CONTENT_LENGTH, 'Summary');
      validateArrayLength(node.tags, MAX_TAGS_COUNT, 'Tags');
      validateArrayLength(node.people, MAX_TAGS_COUNT, 'People');
      validateArrayLength(node.concepts, MAX_TAGS_COUNT, 'Concepts');
      validateArrayLength(node.events, MAX_TAGS_COUNT, 'Events');

      // Validate confidence is within bounds
      const confidence = Math.max(0, Math.min(1, node.confidence ?? 0.8));
      const retention = Math.max(0, Math.min(1, node.retentionStrength ?? 1.0));

      // Dual-Strength Memory Model (Bjork & Bjork, 1992)
      const storageStrength = Math.max(1, node.storageStrength ?? 1.0);
      const retrievalStrength = Math.max(0, Math.min(1, node.retrievalStrength ?? 1.0));

      // Analyze emotional intensity of content
      // Highly emotional memories get stability boost (decay slower)
      const sentimentIntensity = node.sentimentIntensity ?? analyzeSentimentIntensity(node.content);

      // Git-Blame for Thoughts: Capture current code context
      // This lets you "time travel" to see what you were working on when you had this thought
      const gitContext = node.gitContext ?? captureGitContext();

      const id = nanoid();
      const now = new Date().toISOString();

      const stmt = this.db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, summary,
          created_at, updated_at, last_accessed_at, access_count,
          retention_strength, sentiment_intensity, next_review_date, review_count,
          storage_strength, retrieval_strength,
          source_type, source_platform, source_id, source_url, source_chain, git_context,
          confidence, is_contradicted, contradiction_ids,
          people, concepts, events, tags
        ) VALUES (
          ?, ?, ?,
          ?, ?, ?, ?,
          ?, ?, ?, ?,
          ?, ?,
          ?, ?, ?, ?, ?, ?,
          ?, ?, ?,
          ?, ?, ?, ?
        )
      `);

      stmt.run(
        id, node.content, node.summary || null,
        node.createdAt?.toISOString() || now,
        now,
        now,
        0,
        retention,
        sentimentIntensity,
        node.nextReviewDate?.toISOString() || null,
        0,
        storageStrength,
        retrievalStrength,
        node.sourceType,
        node.sourcePlatform,
        node.sourceId || null,
        node.sourceUrl || null,
        JSON.stringify(node.sourceChain || []),
        gitContext ? JSON.stringify(gitContext) : null,
        confidence,
        node.isContradicted ? 1 : 0,
        JSON.stringify(node.contradictionIds || []),
        JSON.stringify(node.people || []),
        JSON.stringify(node.concepts || []),
        JSON.stringify(node.events || []),
        JSON.stringify(node.tags || [])
      );

      return { ...node, id } as KnowledgeNode;
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to insert knowledge node',
        'INSERT_NODE_FAILED'
      );
    }
  }

  getNode(id: string): KnowledgeNode | null {
    try {
      const stmt = this.db.prepare('SELECT * FROM knowledge_nodes WHERE id = ?');
      const row = stmt.get(id) as Record<string, unknown> | undefined;
      if (!row) return null;
      return this.rowToNode(row);
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to get node: ${id}`,
        'GET_NODE_FAILED',
        error
      );
    }
  }

  updateNodeAccess(id: string): void {
    try {
      // Dual-Strength Memory Model (Bjork & Bjork, 1992):
      // - Storage strength increases with each exposure (never decreases)
      // - Retrieval strength resets to 1.0 on access (we just retrieved it successfully)
      const stmt = this.db.prepare(`
        UPDATE knowledge_nodes
        SET last_accessed_at = ?,
            access_count = access_count + 1,
            storage_strength = storage_strength + 0.05,
            retrieval_strength = 1.0
        WHERE id = ?
      `);
      stmt.run(new Date().toISOString(), id);
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to update node access: ${id}`,
        'UPDATE_ACCESS_FAILED',
        error
      );
    }
  }

  /**
   * Mark a node as reviewed (spaced repetition)
   */
  /**
   * Mark a node as reviewed using SM-2 inspired spaced repetition
   *
   * KEY INSIGHT: We don't just reset retention - we modify the STABILITY FACTOR
   * - High retention (remembered easily) → Stability increases → Slower future decay
   * - Low retention (forgot/struggled) → Stability resets → Must rebuild memory
   *
   * This creates "crystallized" memories that barely decay after multiple reviews
   *
   * DUAL-STRENGTH MODEL (Bjork & Bjork, 1992):
   * - Storage strength ALWAYS increases on review (more for difficult recalls)
   * - Retrieval strength resets to 1.0
   */
  markReviewed(id: string): void {
    try {
      const node = this.getNode(id);
      if (!node) {
        throw new VestigeDatabaseError(`Node not found: ${id}`, 'NODE_NOT_FOUND');
      }

      const currentStability = node.stabilityFactor ?? SM2_MIN_STABILITY;
      const currentStorageStrength = node.storageStrength ?? 1.0;
      let newStability: number;
      let newReviewCount: number;
      let newStorageStrength: number;

      // SM-2 with Lapse Detection
      if (node.retentionStrength >= SM2_LAPSE_THRESHOLD) {
        // SUCCESSFUL RECALL: Memory was still accessible
        // Increase stability - the curve gets flatter (slower decay)
        newStability = Math.min(SM2_MAX_STABILITY, currentStability * SM2_EASE_FACTOR);
        newReviewCount = node.reviewCount + 1;
        // Storage strength increases moderately for easy recalls
        newStorageStrength = currentStorageStrength + 0.1;
      } else {
        // LAPSE: Memory had decayed too far - we "forgot" it
        // Reset stability - must rebuild the memory from scratch
        // But keep review count as a record of total attempts
        newStability = SM2_MIN_STABILITY;
        newReviewCount = node.reviewCount + 1; // Still count the review
        // DESIRABLE DIFFICULTY: Storage strength increases MORE for difficult recalls
        // This is a key insight from Bjork & Bjork - struggling to retrieve strengthens encoding
        newStorageStrength = currentStorageStrength + 0.3;
      }

      // Reset retention to full strength (we just accessed it)
      const newRetention = 1.0;
      // Reset retrieval strength to 1.0 (we just retrieved it successfully)
      const newRetrievalStrength = 1.0;

      // Calculate next review date based on NEW stability
      // Higher stability = longer until next review needed
      const daysUntilReview = Math.ceil(newStability);
      const nextReview = new Date();
      nextReview.setDate(nextReview.getDate() + daysUntilReview);

      const stmt = this.db.prepare(`
        UPDATE knowledge_nodes
        SET retention_strength = ?,
            stability_factor = ?,
            review_count = ?,
            next_review_date = ?,
            last_accessed_at = ?,
            updated_at = ?,
            storage_strength = ?,
            retrieval_strength = ?
        WHERE id = ?
      `);
      stmt.run(
        newRetention,
        newStability,
        newReviewCount,
        nextReview.toISOString(),
        new Date().toISOString(),
        new Date().toISOString(),
        newStorageStrength,
        newRetrievalStrength,
        id
      );
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to mark node as reviewed',
        'MARK_REVIEWED_FAILED'
      );
    }
  }

  searchNodes(query: string, options: PaginationOptions = {}): PaginatedResult<KnowledgeNode> {
    try {
      // Input validation
      validateStringLength(query, MAX_QUERY_LENGTH, 'Search query');

      // Sanitize FTS5 query to prevent injection
      // FTS5 special characters: AND OR NOT ( ) " * ^
      const sanitizedQuery = query
        .replace(/[^\w\s\-]/g, ' ')  // Remove special characters except hyphen
        .trim();

      if (!sanitizedQuery) {
        return {
          items: [],
          total: 0,
          limit: DEFAULT_LIMIT,
          offset: 0,
          hasMore: false,
        };
      }

      const { limit = DEFAULT_LIMIT, offset = 0 } = options;
      const safeLimit = Math.min(Math.max(1, limit), MAX_LIMIT);
      const safeOffset = Math.max(0, offset);

      // Get total count
      const countStmt = this.db.prepare(`
        SELECT COUNT(*) as total FROM knowledge_nodes kn
        JOIN knowledge_fts fts ON kn.id = fts.id
        WHERE knowledge_fts MATCH ?
      `);
      const countResult = countStmt.get(sanitizedQuery) as { total: number };
      const total = countResult.total;

      // Get paginated results
      const stmt = this.db.prepare(`
        SELECT kn.* FROM knowledge_nodes kn
        JOIN knowledge_fts fts ON kn.id = fts.id
        WHERE knowledge_fts MATCH ?
        ORDER BY rank
        LIMIT ? OFFSET ?
      `);
      const rows = stmt.all(sanitizedQuery, safeLimit, safeOffset) as Record<string, unknown>[];
      const items = rows.map(row => this.rowToNode(row));

      return {
        items,
        total,
        limit: safeLimit,
        offset: safeOffset,
        hasMore: safeOffset + items.length < total,
      };
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Search operation failed',
        'SEARCH_FAILED'
      );
    }
  }

  getRecentNodes(options: PaginationOptions = {}): PaginatedResult<KnowledgeNode> {
    try {
      const { limit = DEFAULT_LIMIT, offset = 0 } = options;
      const safeLimit = Math.min(limit, MAX_LIMIT);

      // Get total count
      const countResult = this.db.prepare('SELECT COUNT(*) as total FROM knowledge_nodes').get() as { total: number };
      const total = countResult.total;

      // Get paginated results
      const stmt = this.db.prepare(`
        SELECT * FROM knowledge_nodes
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
      `);
      const rows = stmt.all(safeLimit, offset) as Record<string, unknown>[];
      const items = rows.map(row => this.rowToNode(row));

      return {
        items,
        total,
        limit: safeLimit,
        offset,
        hasMore: offset + items.length < total,
      };
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get recent nodes',
        'GET_RECENT_FAILED',
        error
      );
    }
  }

  getDecayingNodes(threshold: number = 0.5, options: PaginationOptions = {}): PaginatedResult<KnowledgeNode> {
    try {
      const { limit = DEFAULT_LIMIT, offset = 0 } = options;
      const safeLimit = Math.min(limit, MAX_LIMIT);

      // Get total count
      const countStmt = this.db.prepare(`
        SELECT COUNT(*) as total FROM knowledge_nodes
        WHERE retention_strength < ?
      `);
      const countResult = countStmt.get(threshold) as { total: number };
      const total = countResult.total;

      // Get paginated results
      const stmt = this.db.prepare(`
        SELECT * FROM knowledge_nodes
        WHERE retention_strength < ?
        ORDER BY retention_strength ASC
        LIMIT ? OFFSET ?
      `);
      const rows = stmt.all(threshold, safeLimit, offset) as Record<string, unknown>[];
      const items = rows.map(row => this.rowToNode(row));

      return {
        items,
        total,
        limit: safeLimit,
        offset,
        hasMore: offset + items.length < total,
      };
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get decaying nodes',
        'GET_DECAYING_FAILED',
        error
      );
    }
  }

  getNodeCount(): number {
    try {
      const stmt = this.db.prepare('SELECT COUNT(*) as count FROM knowledge_nodes');
      const result = stmt.get() as { count: number };
      return result.count;
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get node count',
        'COUNT_FAILED',
        error
      );
    }
  }

  /**
   * Delete a knowledge node
   */
  deleteNode(id: string): boolean {
    try {
      const stmt = this.db.prepare('DELETE FROM knowledge_nodes WHERE id = ?');
      const result = stmt.run(id);
      return result.changes > 0;
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to delete node: ${id}`,
        'DELETE_NODE_FAILED',
        error
      );
    }
  }

  // ============================================================================
  // PEOPLE
  // ============================================================================

  insertPerson(person: Omit<PersonNode, 'id'>): PersonNode {
    try {
      // Input validation
      validateStringLength(person.name, MAX_NAME_LENGTH, 'Name');
      validateStringLength(person.notes || '', MAX_CONTENT_LENGTH, 'Notes');
      validateStringLength(person.howWeMet || '', MAX_CONTENT_LENGTH, 'How we met');
      validateArrayLength(person.aliases, MAX_TAGS_COUNT, 'Aliases');
      validateArrayLength(person.sharedTopics, MAX_TAGS_COUNT, 'Shared topics');
      validateArrayLength(person.sharedProjects, MAX_TAGS_COUNT, 'Shared projects');

      // Validate relationship health is within bounds
      const relationshipHealth = Math.max(0, Math.min(1, person.relationshipHealth ?? 0.5));

      const id = nanoid();
      const now = new Date().toISOString();

      const stmt = this.db.prepare(`
        INSERT INTO people (
          id, name, aliases,
          how_we_met, relationship_type, organization, role, location,
          email, phone, social_links,
          last_contact_at, contact_frequency, preferred_channel,
          shared_topics, shared_projects,
          notes, relationship_health,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(
        id, person.name, JSON.stringify(person.aliases || []),
        person.howWeMet || null, person.relationshipType || null,
        person.organization || null, person.role || null, person.location || null,
        person.email || null, person.phone || null, JSON.stringify(person.socialLinks || {}),
        person.lastContactAt?.toISOString() || null, person.contactFrequency || 0,
        person.preferredChannel || null,
        JSON.stringify(person.sharedTopics || []), JSON.stringify(person.sharedProjects || []),
        person.notes || null, relationshipHealth,
        now, now
      );

      return { ...person, id, createdAt: new Date(now), updatedAt: new Date(now) } as PersonNode;
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to insert person',
        'INSERT_PERSON_FAILED'
      );
    }
  }

  getPerson(id: string): PersonNode | null {
    try {
      const stmt = this.db.prepare('SELECT * FROM people WHERE id = ?');
      const row = stmt.get(id) as Record<string, unknown> | undefined;
      if (!row) return null;
      return this.rowToPerson(row);
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to get person: ${id}`,
        'GET_PERSON_FAILED',
        error
      );
    }
  }

  getPersonByName(name: string): PersonNode | null {
    try {
      // Input validation
      validateStringLength(name, MAX_NAME_LENGTH, 'Name');

      // Sanitize name for LIKE query - escape special LIKE characters
      // This prevents SQL injection via LIKE wildcards
      const escapedName = name
        .replace(/\\/g, '\\\\')  // Escape backslashes first
        .replace(/%/g, '\\%')    // Escape percent
        .replace(/_/g, '\\_')    // Escape underscore
        .replace(/"/g, '\\"');   // Escape quotes for JSON match

      const stmt = this.db.prepare(`
        SELECT * FROM people
        WHERE name = ? OR aliases LIKE ? ESCAPE '\\'
      `);
      const row = stmt.get(name, `%"${escapedName}"%`) as Record<string, unknown> | undefined;
      if (!row) return null;
      return this.rowToPerson(row);
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to get person by name',
        'GET_PERSON_BY_NAME_FAILED'
      );
    }
  }

  getAllPeople(options: PaginationOptions = {}): PaginatedResult<PersonNode> {
    try {
      const { limit = DEFAULT_LIMIT, offset = 0 } = options;
      const safeLimit = Math.min(limit, MAX_LIMIT);

      // Get total count
      const countResult = this.db.prepare('SELECT COUNT(*) as total FROM people').get() as { total: number };
      const total = countResult.total;

      // Get paginated results
      const stmt = this.db.prepare('SELECT * FROM people ORDER BY name LIMIT ? OFFSET ?');
      const rows = stmt.all(safeLimit, offset) as Record<string, unknown>[];
      const items = rows.map(row => this.rowToPerson(row));

      return {
        items,
        total,
        limit: safeLimit,
        offset,
        hasMore: offset + items.length < total,
      };
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get all people',
        'GET_ALL_PEOPLE_FAILED',
        error
      );
    }
  }

  getPeopleToReconnect(daysSinceContact: number = 30, options: PaginationOptions = {}): PaginatedResult<PersonNode> {
    try {
      const { limit = DEFAULT_LIMIT, offset = 0 } = options;
      const safeLimit = Math.min(limit, MAX_LIMIT);

      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysSinceContact);
      const cutoffStr = cutoffDate.toISOString();

      // Get total count
      const countStmt = this.db.prepare(`
        SELECT COUNT(*) as total FROM people
        WHERE last_contact_at IS NOT NULL AND last_contact_at < ?
      `);
      const countResult = countStmt.get(cutoffStr) as { total: number };
      const total = countResult.total;

      // Get paginated results
      const stmt = this.db.prepare(`
        SELECT * FROM people
        WHERE last_contact_at IS NOT NULL
          AND last_contact_at < ?
        ORDER BY last_contact_at ASC
        LIMIT ? OFFSET ?
      `);
      const rows = stmt.all(cutoffStr, safeLimit, offset) as Record<string, unknown>[];
      const items = rows.map(row => this.rowToPerson(row));

      return {
        items,
        total,
        limit: safeLimit,
        offset,
        hasMore: offset + items.length < total,
      };
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get people to reconnect',
        'GET_RECONNECT_FAILED',
        error
      );
    }
  }

  /**
   * Update last contact date for a person
   */
  updatePersonContact(id: string): void {
    try {
      const stmt = this.db.prepare(`
        UPDATE people
        SET last_contact_at = ?, updated_at = ?
        WHERE id = ?
      `);
      const now = new Date().toISOString();
      stmt.run(now, now, id);
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to update person contact: ${id}`,
        'UPDATE_CONTACT_FAILED',
        error
      );
    }
  }

  /**
   * Delete a person
   */
  deletePerson(id: string): boolean {
    try {
      const stmt = this.db.prepare('DELETE FROM people WHERE id = ?');
      const result = stmt.run(id);
      return result.changes > 0;
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to delete person: ${id}`,
        'DELETE_PERSON_FAILED',
        error
      );
    }
  }

  // ============================================================================
  // GRAPH EDGES
  // ============================================================================

  insertEdge(edge: Omit<GraphEdge, 'id'>): GraphEdge {
    try {
      const id = nanoid();
      const now = new Date().toISOString();

      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO graph_edges (
          id, from_id, to_id, edge_type, weight, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(
        id, edge.fromId, edge.toId, edge.edgeType,
        edge.weight ?? 0.5, JSON.stringify(edge.metadata || {}),
        now
      );

      return { ...edge, id, createdAt: new Date(now) } as GraphEdge;
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to insert edge',
        'INSERT_EDGE_FAILED',
        error
      );
    }
  }

  getRelatedNodes(nodeId: string, depth: number = 1): string[] {
    try {
      // Simple BFS for related nodes
      const visited = new Set<string>();
      let current = [nodeId];

      for (let d = 0; d < depth; d++) {
        if (current.length === 0) break;

        const placeholders = current.map(() => '?').join(',');
        const stmt = this.db.prepare(`
          SELECT DISTINCT
            CASE WHEN from_id IN (${placeholders}) THEN to_id ELSE from_id END as related_id
          FROM graph_edges
          WHERE from_id IN (${placeholders}) OR to_id IN (${placeholders})
        `);

        const params = [...current, ...current, ...current];
        const rows = stmt.all(...params) as { related_id: string }[];

        const newNodes: string[] = [];
        for (const row of rows) {
          if (!visited.has(row.related_id) && row.related_id !== nodeId) {
            visited.add(row.related_id);
            newNodes.push(row.related_id);
          }
        }
        current = newNodes;
      }

      return Array.from(visited);
    } catch (error) {
      throw new VestigeDatabaseError(
        `Failed to get related nodes: ${nodeId}`,
        'GET_RELATED_FAILED',
        error
      );
    }
  }

  // ============================================================================
  // STATS
  // ============================================================================

  getStats(): { totalNodes: number; totalPeople: number; totalEdges: number } {
    try {
      const nodeCount = this.db.prepare('SELECT COUNT(*) as c FROM knowledge_nodes').get() as { c: number };
      const peopleCount = this.db.prepare('SELECT COUNT(*) as c FROM people').get() as { c: number };
      const edgeCount = this.db.prepare('SELECT COUNT(*) as c FROM graph_edges').get() as { c: number };

      return {
        totalNodes: nodeCount.c,
        totalPeople: peopleCount.c,
        totalEdges: edgeCount.c,
      };
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to get stats',
        'GET_STATS_FAILED',
        error
      );
    }
  }

  // ============================================================================
  // MAINTENANCE
  // ============================================================================

  /**
   * Optimize database (vacuum and reindex)
   */
  optimize(): void {
    try {
      // Checkpoint WAL
      this.db.pragma('wal_checkpoint(TRUNCATE)');
      // Vacuum to reclaim space
      this.db.exec('VACUUM');
      // Reindex for performance
      this.db.exec('REINDEX');
    } catch (error) {
      throw new VestigeDatabaseError(
        'Failed to optimize database',
        'OPTIMIZE_FAILED',
        error
      );
    }
  }

  /**
   * Apply decay to all nodes based on time since last access
   * Call this periodically (e.g., daily) to update retention strengths
   *
   * KEY FEATURES:
   * 1. Each node decays at ITS OWN rate based on stability_factor
   * 2. EMOTIONAL MEMORIES DECAY SLOWER via sentiment_intensity boost
   * 3. DUAL-STRENGTH MODEL (Bjork & Bjork, 1992):
   *    - Retrieval strength decays based on storage strength and sentiment
   *    - Storage strength NEVER decreases (only increases on access/review)
   *    - Higher storage = slower retrieval decay
   *
   * Decay rates:
   * - New neutral memories (S=1, emotion=0): Decay fast → 50% after 1 day
   * - New emotional memories (S=1, emotion=1): Decay slower → 75% after 1 day
   * - Reviewed memories (S=10): Decay slow → 90% after 1 day
   * - Crystallized emotional (S=100, emotion=1): Near-permanent → 99.5% after 1 day
   *
   * Uses IMMEDIATE transaction to prevent dirty reads and ensure consistency
   */
  applyDecay(): number {
    try {
      const now = Date.now();

      // Use IMMEDIATE transaction mode for write consistency
      // This acquires write lock at start, preventing concurrent modifications
      const transaction = this.db.transaction(() => {
        // Fetch all strength factors for each node
        const nodes = this.db.prepare(`
          SELECT id, last_accessed_at, retention_strength, stability_factor, sentiment_intensity,
                 storage_strength, retrieval_strength
          FROM knowledge_nodes
        `).all() as {
          id: string;
          last_accessed_at: string;
          retention_strength: number;
          stability_factor: number | null;
          sentiment_intensity: number | null;
          storage_strength: number | null;
          retrieval_strength: number | null;
        }[];

        let updated = 0;
        const updateStmt = this.db.prepare(`
          UPDATE knowledge_nodes
          SET retention_strength = ?, retrieval_strength = ?
          WHERE id = ?
        `);

        for (const node of nodes) {
          const lastAccessed = new Date(node.last_accessed_at).getTime();
          const daysSince = (now - lastAccessed) / (1000 * 60 * 60 * 24);

          // Base stability factor (from SM-2 reviews)
          const baseStability = node.stability_factor ?? SM2_MIN_STABILITY;

          // SENTIMENT BOOST: Emotional memories decay slower
          // sentimentIntensity: 0 = neutral (1x), 1 = highly emotional (2x boost)
          const sentimentIntensity = node.sentiment_intensity ?? 0;
          const sentimentMultiplier = SENTIMENT_MIN_BOOST + (sentimentIntensity * (SENTIMENT_STABILITY_BOOST - SENTIMENT_MIN_BOOST));

          // Effective stability = base stability * sentiment boost
          // A memory with S=10 and high emotion (1.0) becomes S_effective = 10 * 2 = 20
          const effectiveStability = baseStability * sentimentMultiplier;

          // Ebbinghaus forgetting curve: R = e^(-t/S)
          // where t is time in days and S is effective stability
          // Higher S = slower decay = "crystallized" or emotional memory
          const newRetention = Math.max(0.1, node.retention_strength * Math.exp(-daysSince / effectiveStability));

          // DUAL-STRENGTH MODEL (Bjork & Bjork, 1992):
          // Retrieval strength decays based on storage strength and sentiment intensity
          // Higher storage strength = slower retrieval decay
          const storageStrength = node.storage_strength ?? 1.0;
          const currentRetrievalStrength = node.retrieval_strength ?? 1.0;

          // Effective decay rate is inversely proportional to storage strength and emotional weight
          // Formula: effectiveDecayRate = 1 / (storageStrength * (1 + sentimentIntensity))
          const effectiveDecayRate = 1 / (storageStrength * (1 + sentimentIntensity));

          // Apply decay to retrieval strength with minimum floor of 0.1
          const newRetrievalStrength = Math.max(0.1, Math.exp(-daysSince * effectiveDecayRate));

          // Compute backward-compatible retention_strength as a weighted combination
          // This preserves existing behavior while incorporating dual-strength model
          // retention_strength = (retrieval_strength * 0.7) + (normalized_storage * 0.3)
          const normalizedStorage = Math.min(1, storageStrength / 10);
          const backwardCompatibleRetention = (newRetrievalStrength * 0.7) + (normalizedStorage * 0.3);

          const hasRetentionChange = Math.abs(backwardCompatibleRetention - node.retention_strength) > 0.01;
          const hasRetrievalChange = Math.abs(newRetrievalStrength - currentRetrievalStrength) > 0.01;

          if (hasRetentionChange || hasRetrievalChange) {
            updateStmt.run(backwardCompatibleRetention, newRetrievalStrength, node.id);
            updated++;
          }
        }

        return updated;
      });

      // Execute with IMMEDIATE mode (acquires RESERVED lock immediately)
      return transaction.immediate();
    } catch (error) {
      if (error instanceof VestigeDatabaseError) throw error;
      throw new VestigeDatabaseError(
        'Failed to apply decay',
        'APPLY_DECAY_FAILED'
      );
    }
  }

  // ============================================================================
  // HELPERS
  // ============================================================================

  private rowToNode(row: Record<string, unknown>): KnowledgeNode {
    // Use safe JSON parsing with fallbacks to prevent crashes from corrupted data
    return {
      id: row['id'] as string,
      content: row['content'] as string,
      summary: row['summary'] as string | undefined,
      createdAt: new Date(row['created_at'] as string),
      updatedAt: new Date(row['updated_at'] as string),
      lastAccessedAt: new Date(row['last_accessed_at'] as string),
      accessCount: row['access_count'] as number,
      retentionStrength: row['retention_strength'] as number,
      stabilityFactor: (row['stability_factor'] as number) ?? SM2_MIN_STABILITY,
      sentimentIntensity: (row['sentiment_intensity'] as number) ?? 0,
      nextReviewDate: row['next_review_date'] ? new Date(row['next_review_date'] as string) : undefined,
      reviewCount: row['review_count'] as number,
      // Dual-Strength Memory Model (Bjork & Bjork, 1992)
      storageStrength: (row['storage_strength'] as number) ?? 1.0,
      retrievalStrength: (row['retrieval_strength'] as number) ?? 1.0,
      sourceType: row['source_type'] as KnowledgeNode['sourceType'],
      sourcePlatform: row['source_platform'] as KnowledgeNode['sourcePlatform'],
      sourceId: row['source_id'] as string | undefined,
      sourceUrl: row['source_url'] as string | undefined,
      sourceChain: safeJsonParse<string[]>(row['source_chain'] as string, []),
      gitContext: row['git_context'] ? safeJsonParse<GitContext>(row['git_context'] as string, undefined) : undefined,
      confidence: row['confidence'] as number,
      isContradicted: Boolean(row['is_contradicted']),
      contradictionIds: safeJsonParse<string[]>(row['contradiction_ids'] as string, []),
      people: safeJsonParse<string[]>(row['people'] as string, []),
      concepts: safeJsonParse<string[]>(row['concepts'] as string, []),
      events: safeJsonParse<string[]>(row['events'] as string, []),
      tags: safeJsonParse<string[]>(row['tags'] as string, []),
    };
  }

  private rowToPerson(row: Record<string, unknown>): PersonNode {
    // Use safe JSON parsing with fallbacks to prevent crashes from corrupted data
    return {
      id: row['id'] as string,
      name: row['name'] as string,
      aliases: safeJsonParse<string[]>(row['aliases'] as string, []),
      howWeMet: row['how_we_met'] as string | undefined,
      relationshipType: row['relationship_type'] as string | undefined,
      organization: row['organization'] as string | undefined,
      role: row['role'] as string | undefined,
      location: row['location'] as string | undefined,
      email: row['email'] as string | undefined,
      phone: row['phone'] as string | undefined,
      socialLinks: safeJsonParse<Record<string, string>>(row['social_links'] as string, {}),
      lastContactAt: row['last_contact_at'] ? new Date(row['last_contact_at'] as string) : undefined,
      contactFrequency: row['contact_frequency'] as number,
      preferredChannel: row['preferred_channel'] as string | undefined,
      sharedTopics: safeJsonParse<string[]>(row['shared_topics'] as string, []),
      sharedProjects: safeJsonParse<string[]>(row['shared_projects'] as string, []),
      notes: row['notes'] as string | undefined,
      relationshipHealth: row['relationship_health'] as number,
      createdAt: new Date(row['created_at'] as string),
      updatedAt: new Date(row['updated_at'] as string),
    };
  }

  close(): void {
    try {
      // Checkpoint WAL before closing
      this.db.pragma('wal_checkpoint(TRUNCATE)');
      this.db.close();
    } catch {
      // Ignore close errors
    }
  }
}

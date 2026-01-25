import Database from 'better-sqlite3';
import type { KnowledgeNodeInput, PersonNode, GraphEdge } from '../core/types.js';

/**
 * Create an in-memory database for testing
 */
export function createTestDatabase(): Database.Database {
  const db = new Database(':memory:');
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  // Initialize tables (from database.ts initializeSchema)
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
      stability_factor REAL DEFAULT 1.0,
      sentiment_intensity REAL DEFAULT 0,
      next_review_date TEXT,
      review_count INTEGER DEFAULT 0,

      -- Dual-Strength Memory Model (Bjork & Bjork, 1992)
      storage_strength REAL DEFAULT 1.0,
      retrieval_strength REAL DEFAULT 1.0,

      -- Provenance
      source_type TEXT NOT NULL,
      source_platform TEXT NOT NULL,
      source_id TEXT,
      source_url TEXT,
      source_chain TEXT DEFAULT '[]',
      git_context TEXT,

      -- Confidence
      confidence REAL DEFAULT 0.8,
      is_contradicted INTEGER DEFAULT 0,
      contradiction_ids TEXT DEFAULT '[]',

      -- Extracted entities (JSON arrays)
      people TEXT DEFAULT '[]',
      concepts TEXT DEFAULT '[]',
      events TEXT DEFAULT '[]',
      tags TEXT DEFAULT '[]'
    );

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
      aliases TEXT DEFAULT '[]',

      -- Relationship context
      how_we_met TEXT,
      relationship_type TEXT,
      organization TEXT,
      role TEXT,
      location TEXT,

      -- Contact info
      email TEXT,
      phone TEXT,
      social_links TEXT DEFAULT '{}',

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

  // Embeddings reference table
  db.exec(`
    CREATE TABLE IF NOT EXISTS embeddings (
      node_id TEXT PRIMARY KEY,
      chroma_id TEXT NOT NULL,
      model TEXT NOT NULL,
      created_at TEXT NOT NULL,

      FOREIGN KEY (node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
    );
  `);

  // Metadata table
  db.exec(`
    CREATE TABLE IF NOT EXISTS vestige_metadata (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );
  `);

  return db;
}

/**
 * Create test fixtures for knowledge nodes
 */
export function createTestNode(overrides: Partial<Omit<KnowledgeNodeInput, 'id'>> = {}): Omit<KnowledgeNodeInput, 'id'> {
  return {
    content: 'Test content for knowledge node',
    sourceType: 'manual',
    sourcePlatform: 'manual',
    tags: [],
    people: [],
    concepts: [],
    events: [],
    ...overrides,
  };
}

/**
 * Create test fixtures for people
 */
export function createTestPerson(overrides: Partial<Omit<PersonNode, 'id' | 'createdAt' | 'updatedAt'>> = {}): Omit<PersonNode, 'id' | 'createdAt' | 'updatedAt'> {
  return {
    name: 'Test Person',
    relationshipType: 'colleague',
    aliases: [],
    socialLinks: {},
    contactFrequency: 0,
    sharedTopics: [],
    sharedProjects: [],
    relationshipHealth: 0.5,
    ...overrides,
  };
}

/**
 * Create test fixtures for graph edges
 */
export function createTestEdge(fromId: string, toId: string, overrides: Partial<Omit<GraphEdge, 'id' | 'createdAt'>> = {}): Omit<GraphEdge, 'id' | 'createdAt'> {
  return {
    fromId,
    toId,
    edgeType: 'relates_to',
    weight: 0.5,
    metadata: {},
    ...overrides,
  };
}

/**
 * Clean up test database
 */
export function cleanupTestDatabase(db: Database.Database): void {
  try {
    db.close();
  } catch {
    // Ignore close errors
  }
}

/**
 * Wait for a specified amount of time (useful for async tests)
 */
export function wait(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Generate a unique test ID
 */
export function generateTestId(): string {
  return `test-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Create a mock timestamp for consistent testing
 */
export function mockTimestamp(daysAgo: number = 0): Date {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date;
}

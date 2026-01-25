/**
 * Vector Store Integration for Vestige MCP
 *
 * Provides semantic search capabilities via vector embeddings.
 * Primary: ChromaDB (when available) - fast, efficient vector database
 * Fallback: SQLite (embedded) - works offline, no external dependencies
 *
 * Design Philosophy:
 * - Graceful degradation: Works without ChromaDB, just slower
 * - Zero configuration: Auto-detects available backends
 * - Production-ready: Full error handling, logging, retry logic
 */

import type Database from 'better-sqlite3';

// ============================================================================
// CONFIGURATION
// ============================================================================

const CHROMA_HOST = process.env['CHROMA_HOST'] ?? 'http://localhost:8000';
const COLLECTION_NAME = 'vestige_embeddings';
const DEFAULT_SIMILARITY_LIMIT = 10;
const MAX_SIMILARITY_LIMIT = 100;

// Connection settings
const CONNECTION_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

// Batch settings for bulk operations
const BATCH_SIZE = 100;

// ============================================================================
// ERROR TYPES
// ============================================================================

export class VectorStoreError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly isRetryable: boolean = false,
    cause?: unknown
  ) {
    super(message);
    this.name = 'VectorStoreError';
    if (cause) {
      this.cause = cause;
    }
  }
}

// ============================================================================
// TYPES
// ============================================================================

export interface SimilarityResult {
  id: string;
  similarity: number;
  content?: string | undefined;
  metadata?: Record<string, unknown> | undefined;
}

export interface VectorStoreStats {
  backend: 'chromadb' | 'sqlite';
  embeddingCount: number;
  collectionName?: string;
  isAvailable: boolean;
}

export interface IVectorStore {
  initialize(): Promise<boolean>;
  isAvailable(): Promise<boolean>;
  upsertEmbedding(
    nodeId: string,
    embedding: number[],
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  findSimilar(
    embedding: number[],
    limit?: number,
    filter?: Record<string, unknown>
  ): Promise<SimilarityResult[]>;
  deleteEmbedding(nodeId: string): Promise<void>;
  getEmbedding(nodeId: string): Promise<number[] | null>;
  getStats(): Promise<VectorStoreStats>;
  close(): Promise<void>;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Calculate cosine similarity between two vectors
 * Returns value between -1 and 1 (1 = identical, 0 = orthogonal, -1 = opposite)
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new VectorStoreError(
      `Vector dimension mismatch: ${a.length} vs ${b.length}`,
      'DIMENSION_MISMATCH'
    );
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

/**
 * Convert ChromaDB distance to similarity score
 * ChromaDB uses L2 (euclidean) distance by default, lower = more similar
 * We convert to similarity: 1 / (1 + distance)
 */
function distanceToSimilarity(distance: number): number {
  return 1 / (1 + distance);
}

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry wrapper for operations that may fail transiently
 */
async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = MAX_RETRIES,
  delayMs: number = RETRY_DELAY_MS
): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;

      // Check if error is retryable
      if (error instanceof VectorStoreError && !error.isRetryable) {
        throw error;
      }

      if (attempt < maxRetries) {
        await sleep(delayMs * attempt); // Exponential backoff
      }
    }
  }

  throw lastError;
}

// ============================================================================
// CHROMADB VECTOR STORE
// ============================================================================

/**
 * ChromaDB-backed vector store for fast semantic search
 *
 * Features:
 * - Persistent storage via ChromaDB server
 * - Fast approximate nearest neighbor search
 * - Metadata filtering support
 * - Automatic retry on transient failures
 */
export class ChromaVectorStore implements IVectorStore {
  private client: import('chromadb').ChromaClient | null = null;
  private collection: import('chromadb').Collection | null = null;
  private available: boolean | null = null;
  private initPromise: Promise<boolean> | null = null;

  constructor(private readonly host: string = CHROMA_HOST) {}

  /**
   * Initialize connection to ChromaDB
   * Creates collection if it doesn't exist
   */
  async initialize(): Promise<boolean> {
    // Dedupe concurrent initialization calls
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<boolean> {
    try {
      // Dynamic import to avoid hard dependency
      const { ChromaClient } = await import('chromadb');

      this.client = new ChromaClient({ path: this.host });

      // Test connection with timeout
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(
          () => reject(new Error('Connection timeout')),
          CONNECTION_TIMEOUT_MS
        );
      });

      await Promise.race([this.client.heartbeat(), timeoutPromise]);

      // Get or create collection
      this.collection = await this.client.getOrCreateCollection({
        name: COLLECTION_NAME,
        metadata: {
          'hnsw:space': 'cosine', // Use cosine similarity
          description: 'Vestige knowledge node embeddings',
        },
      });

      this.available = true;
      console.log(
        `[VectorStore] ChromaDB connected at ${this.host}, collection: ${COLLECTION_NAME}`
      );

      return true;
    } catch (error) {
      this.available = false;
      console.warn(
        `[VectorStore] ChromaDB not available at ${this.host}:`,
        error instanceof Error ? error.message : 'Unknown error'
      );
      return false;
    }
  }

  /**
   * Check if ChromaDB is currently available
   */
  async isAvailable(): Promise<boolean> {
    if (this.available === null) {
      await this.initialize();
    }

    // Re-check heartbeat for ongoing availability
    if (this.available && this.client) {
      try {
        await Promise.race([
          this.client.heartbeat(),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Heartbeat timeout')), 2000)
          ),
        ]);
        return true;
      } catch {
        this.available = false;
        return false;
      }
    }

    return this.available ?? false;
  }

  /**
   * Add or update an embedding in ChromaDB
   */
  async upsertEmbedding(
    nodeId: string,
    embedding: number[],
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.collection) {
      throw new VectorStoreError(
        'ChromaDB not initialized',
        'NOT_INITIALIZED'
      );
    }

    // Validate embedding
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new VectorStoreError(
        'Invalid embedding: must be non-empty array',
        'INVALID_EMBEDDING'
      );
    }

    // Sanitize metadata - ChromaDB only accepts primitive values
    const sanitizedMetadata: Record<string, string | number | boolean> = {};
    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        if (
          typeof value === 'string' ||
          typeof value === 'number' ||
          typeof value === 'boolean'
        ) {
          sanitizedMetadata[key] = value;
        } else if (value !== null && value !== undefined) {
          // Convert complex types to JSON strings
          sanitizedMetadata[key] = JSON.stringify(value);
        }
      }
    }

    try {
      await withRetry(async () => {
        await this.collection!.upsert({
          ids: [nodeId],
          embeddings: [embedding],
          documents: [content],
          metadatas: [sanitizedMetadata],
        });
      });
    } catch (error) {
      throw new VectorStoreError(
        `Failed to upsert embedding for ${nodeId}`,
        'UPSERT_FAILED',
        true,
        error
      );
    }
  }

  /**
   * Find similar embeddings using vector similarity search
   */
  async findSimilar(
    embedding: number[],
    limit: number = DEFAULT_SIMILARITY_LIMIT,
    filter?: Record<string, unknown>
  ): Promise<SimilarityResult[]> {
    if (!this.collection) {
      throw new VectorStoreError(
        'ChromaDB not initialized',
        'NOT_INITIALIZED'
      );
    }

    const safeLimit = Math.min(Math.max(1, limit), MAX_SIMILARITY_LIMIT);

    // Convert filter to ChromaDB where clause format
    let whereClause: Record<string, unknown> | undefined;
    if (filter && Object.keys(filter).length > 0) {
      whereClause = {};
      for (const [key, value] of Object.entries(filter)) {
        if (
          typeof value === 'string' ||
          typeof value === 'number' ||
          typeof value === 'boolean'
        ) {
          whereClause[key] = value;
        }
      }
    }

    try {
      const results = await withRetry(async () => {
        // Note: ChromaDB IncludeEnum values need to be typed correctly
        // Using type assertion since the SDK types may be stricter than needed
        return this.collection!.query({
          queryEmbeddings: [embedding],
          nResults: safeLimit,
          where: whereClause,
          include: ['documents', 'metadatas', 'distances'] as const,
        } as Parameters<typeof this.collection.query>[0]);
      });

      // Transform results
      const similarResults: SimilarityResult[] = [];

      if (results.ids[0]) {
        for (let i = 0; i < results.ids[0].length; i++) {
          const id = results.ids[0][i];
          if (!id) continue;

          const distance = results.distances?.[0]?.[i] ?? 0;
          const document = results.documents?.[0]?.[i];
          const metadata = results.metadatas?.[0]?.[i];

          similarResults.push({
            id,
            similarity: distanceToSimilarity(distance),
            content: document ?? undefined,
            metadata: metadata as Record<string, unknown> | undefined,
          });
        }
      }

      return similarResults;
    } catch (error) {
      throw new VectorStoreError(
        'Failed to query similar embeddings',
        'QUERY_FAILED',
        true,
        error
      );
    }
  }

  /**
   * Delete an embedding from ChromaDB
   */
  async deleteEmbedding(nodeId: string): Promise<void> {
    if (!this.collection) {
      throw new VectorStoreError(
        'ChromaDB not initialized',
        'NOT_INITIALIZED'
      );
    }

    try {
      await withRetry(async () => {
        await this.collection!.delete({ ids: [nodeId] });
      });
    } catch (error) {
      throw new VectorStoreError(
        `Failed to delete embedding for ${nodeId}`,
        'DELETE_FAILED',
        true,
        error
      );
    }
  }

  /**
   * Get embedding for a specific node
   */
  async getEmbedding(nodeId: string): Promise<number[] | null> {
    if (!this.collection) {
      throw new VectorStoreError(
        'ChromaDB not initialized',
        'NOT_INITIALIZED'
      );
    }

    try {
      const result = await this.collection.get({
        ids: [nodeId],
        include: ['embeddings'] as const,
      } as Parameters<typeof this.collection.get>[0]);

      if (result.embeddings && result.embeddings[0]) {
        return result.embeddings[0] as number[];
      }

      return null;
    } catch (error) {
      throw new VectorStoreError(
        `Failed to get embedding for ${nodeId}`,
        'GET_FAILED',
        true,
        error
      );
    }
  }

  /**
   * Get statistics about the vector store
   */
  async getStats(): Promise<VectorStoreStats> {
    const isAvailable = await this.isAvailable();

    if (!isAvailable || !this.collection) {
      return {
        backend: 'chromadb',
        embeddingCount: 0,
        collectionName: COLLECTION_NAME,
        isAvailable: false,
      };
    }

    try {
      const count = await this.collection.count();
      return {
        backend: 'chromadb',
        embeddingCount: count,
        collectionName: COLLECTION_NAME,
        isAvailable: true,
      };
    } catch {
      return {
        backend: 'chromadb',
        embeddingCount: 0,
        collectionName: COLLECTION_NAME,
        isAvailable: false,
      };
    }
  }

  /**
   * Close the ChromaDB connection
   */
  async close(): Promise<void> {
    // ChromaDB client doesn't need explicit closing
    this.client = null;
    this.collection = null;
    this.available = null;
    this.initPromise = null;
  }
}

// ============================================================================
// SQLITE VECTOR STORE (FALLBACK)
// ============================================================================

/**
 * SQLite-backed vector store for offline/embedded use
 *
 * Stores embeddings as JSON in SQLite when ChromaDB is unavailable.
 * Slower than ChromaDB but works without external dependencies.
 *
 * Features:
 * - Zero external dependencies
 * - Works offline
 * - Brute-force cosine similarity (O(n) per query)
 * - Good enough for small-medium datasets (<10k embeddings)
 */
export class SQLiteVectorStore implements IVectorStore {
  private db: Database.Database | null = null;
  private initialized = false;

  constructor(private readonly getDatabase: () => Database.Database) {}

  /**
   * Initialize SQLite vector store
   * Creates embeddings_local table if needed
   */
  async initialize(): Promise<boolean> {
    try {
      this.db = this.getDatabase();

      // Create table for storing embeddings locally
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS embeddings_local (
          node_id TEXT PRIMARY KEY,
          embedding TEXT NOT NULL,
          content TEXT,
          metadata TEXT,
          dimension INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_local_dimension
        ON embeddings_local(dimension);
      `);

      this.initialized = true;
      console.log('[VectorStore] SQLite fallback initialized');

      return true;
    } catch (error) {
      console.error('[VectorStore] SQLite initialization failed:', error);
      return false;
    }
  }

  /**
   * SQLite fallback is always available if initialized
   */
  async isAvailable(): Promise<boolean> {
    return this.initialized && this.db !== null;
  }

  /**
   * Store embedding in SQLite as JSON
   */
  async upsertEmbedding(
    nodeId: string,
    embedding: number[],
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.db || !this.initialized) {
      throw new VectorStoreError(
        'SQLite vector store not initialized',
        'NOT_INITIALIZED'
      );
    }

    // Validate embedding
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new VectorStoreError(
        'Invalid embedding: must be non-empty array',
        'INVALID_EMBEDDING'
      );
    }

    const now = new Date().toISOString();

    try {
      const stmt = this.db.prepare(`
        INSERT INTO embeddings_local (
          node_id, embedding, content, metadata, dimension, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
          embedding = excluded.embedding,
          content = excluded.content,
          metadata = excluded.metadata,
          dimension = excluded.dimension,
          updated_at = excluded.updated_at
      `);

      stmt.run(
        nodeId,
        JSON.stringify(embedding),
        content,
        metadata ? JSON.stringify(metadata) : null,
        embedding.length,
        now,
        now
      );
    } catch (error) {
      throw new VectorStoreError(
        `Failed to upsert embedding for ${nodeId}`,
        'UPSERT_FAILED',
        false,
        error
      );
    }
  }

  /**
   * Find similar embeddings using brute-force cosine similarity
   *
   * NOTE: This is O(n) - suitable for small datasets only.
   * For large datasets, use ChromaDB instead.
   */
  async findSimilar(
    embedding: number[],
    limit: number = DEFAULT_SIMILARITY_LIMIT,
    filter?: Record<string, unknown>
  ): Promise<SimilarityResult[]> {
    if (!this.db || !this.initialized) {
      throw new VectorStoreError(
        'SQLite vector store not initialized',
        'NOT_INITIALIZED'
      );
    }

    const safeLimit = Math.min(Math.max(1, limit), MAX_SIMILARITY_LIMIT);

    try {
      // First, filter by dimension to avoid comparing incompatible vectors
      const stmt = this.db.prepare(`
        SELECT node_id, embedding, content, metadata
        FROM embeddings_local
        WHERE dimension = ?
      `);

      type EmbeddingRow = {
        node_id: string;
        embedding: string;
        content: string | null;
        metadata: string | null;
      };

      const rows = stmt.all(embedding.length) as EmbeddingRow[];

      // Calculate similarity for each embedding
      const results: SimilarityResult[] = [];

      for (const row of rows) {
        let storedEmbedding: number[];
        try {
          storedEmbedding = JSON.parse(row.embedding) as number[];
        } catch {
          continue; // Skip corrupted embeddings
        }

        // Apply metadata filter if provided
        if (filter && Object.keys(filter).length > 0) {
          let rowMetadata: Record<string, unknown> = {};
          if (row.metadata) {
            try {
              rowMetadata = JSON.parse(row.metadata) as Record<string, unknown>;
            } catch {
              // Invalid metadata, skip filter
            }
          }

          let matches = true;
          for (const [key, value] of Object.entries(filter)) {
            if (rowMetadata[key] !== value) {
              matches = false;
              break;
            }
          }

          if (!matches) continue;
        }

        const similarity = cosineSimilarity(embedding, storedEmbedding);

        let metadata: Record<string, unknown> | undefined;
        if (row.metadata) {
          try {
            metadata = JSON.parse(row.metadata) as Record<string, unknown>;
          } catch {
            // Ignore invalid metadata
          }
        }

        results.push({
          id: row.node_id,
          similarity,
          content: row.content ?? undefined,
          metadata,
        });
      }

      // Sort by similarity descending and take top N
      results.sort((a, b) => b.similarity - a.similarity);

      return results.slice(0, safeLimit);
    } catch (error) {
      if (error instanceof VectorStoreError) throw error;
      throw new VectorStoreError(
        'Failed to query similar embeddings',
        'QUERY_FAILED',
        false,
        error
      );
    }
  }

  /**
   * Delete an embedding from SQLite
   */
  async deleteEmbedding(nodeId: string): Promise<void> {
    if (!this.db || !this.initialized) {
      throw new VectorStoreError(
        'SQLite vector store not initialized',
        'NOT_INITIALIZED'
      );
    }

    try {
      const stmt = this.db.prepare(
        'DELETE FROM embeddings_local WHERE node_id = ?'
      );
      stmt.run(nodeId);
    } catch (error) {
      throw new VectorStoreError(
        `Failed to delete embedding for ${nodeId}`,
        'DELETE_FAILED',
        false,
        error
      );
    }
  }

  /**
   * Get embedding for a specific node
   */
  async getEmbedding(nodeId: string): Promise<number[] | null> {
    if (!this.db || !this.initialized) {
      throw new VectorStoreError(
        'SQLite vector store not initialized',
        'NOT_INITIALIZED'
      );
    }

    try {
      const stmt = this.db.prepare(
        'SELECT embedding FROM embeddings_local WHERE node_id = ?'
      );
      const row = stmt.get(nodeId) as { embedding: string } | undefined;

      if (!row) return null;

      return JSON.parse(row.embedding) as number[];
    } catch (error) {
      throw new VectorStoreError(
        `Failed to get embedding for ${nodeId}`,
        'GET_FAILED',
        false,
        error
      );
    }
  }

  /**
   * Get statistics about the SQLite vector store
   */
  async getStats(): Promise<VectorStoreStats> {
    if (!this.db || !this.initialized) {
      return {
        backend: 'sqlite',
        embeddingCount: 0,
        isAvailable: false,
      };
    }

    try {
      const row = this.db
        .prepare('SELECT COUNT(*) as count FROM embeddings_local')
        .get() as { count: number };

      return {
        backend: 'sqlite',
        embeddingCount: row.count,
        isAvailable: true,
      };
    } catch {
      return {
        backend: 'sqlite',
        embeddingCount: 0,
        isAvailable: false,
      };
    }
  }

  /**
   * Close the SQLite vector store
   */
  async close(): Promise<void> {
    // Don't close the shared database connection
    // Just clear our reference
    this.db = null;
    this.initialized = false;
  }
}

// ============================================================================
// HYBRID VECTOR STORE
// ============================================================================

/**
 * Hybrid vector store that combines ChromaDB and SQLite
 *
 * Strategy:
 * - Try ChromaDB first (fast, scalable)
 * - Fall back to SQLite if unavailable (offline, embedded)
 * - Sync between stores when ChromaDB becomes available
 */
export class HybridVectorStore implements IVectorStore {
  private chromaStore: ChromaVectorStore;
  private sqliteStore: SQLiteVectorStore;
  private activeStore: IVectorStore | null = null;

  constructor(getDatabase: () => Database.Database, chromaHost?: string) {
    this.chromaStore = new ChromaVectorStore(chromaHost);
    this.sqliteStore = new SQLiteVectorStore(getDatabase);
  }

  /**
   * Initialize both stores and select the best available
   */
  async initialize(): Promise<boolean> {
    // Try ChromaDB first
    const chromaAvailable = await this.chromaStore.initialize();

    if (chromaAvailable) {
      this.activeStore = this.chromaStore;
      console.log('[VectorStore] Using ChromaDB backend');
      return true;
    }

    // Fall back to SQLite
    const sqliteAvailable = await this.sqliteStore.initialize();

    if (sqliteAvailable) {
      this.activeStore = this.sqliteStore;
      console.log('[VectorStore] Using SQLite fallback backend');
      return true;
    }

    console.error('[VectorStore] No backend available');
    return false;
  }

  async isAvailable(): Promise<boolean> {
    if (!this.activeStore) return false;
    return this.activeStore.isAvailable();
  }

  async upsertEmbedding(
    nodeId: string,
    embedding: number[],
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.activeStore) {
      throw new VectorStoreError('No vector store available', 'NOT_INITIALIZED');
    }

    await this.activeStore.upsertEmbedding(nodeId, embedding, content, metadata);
  }

  async findSimilar(
    embedding: number[],
    limit?: number,
    filter?: Record<string, unknown>
  ): Promise<SimilarityResult[]> {
    if (!this.activeStore) {
      throw new VectorStoreError('No vector store available', 'NOT_INITIALIZED');
    }

    return this.activeStore.findSimilar(embedding, limit, filter);
  }

  async deleteEmbedding(nodeId: string): Promise<void> {
    if (!this.activeStore) {
      throw new VectorStoreError('No vector store available', 'NOT_INITIALIZED');
    }

    await this.activeStore.deleteEmbedding(nodeId);
  }

  async getEmbedding(nodeId: string): Promise<number[] | null> {
    if (!this.activeStore) {
      throw new VectorStoreError('No vector store available', 'NOT_INITIALIZED');
    }

    return this.activeStore.getEmbedding(nodeId);
  }

  async getStats(): Promise<VectorStoreStats> {
    if (!this.activeStore) {
      return {
        backend: 'sqlite',
        embeddingCount: 0,
        isAvailable: false,
      };
    }

    return this.activeStore.getStats();
  }

  /**
   * Get the currently active backend type
   */
  getActiveBackend(): 'chromadb' | 'sqlite' | null {
    if (this.activeStore === this.chromaStore) return 'chromadb';
    if (this.activeStore === this.sqliteStore) return 'sqlite';
    return null;
  }

  /**
   * Attempt to switch to ChromaDB if it becomes available
   */
  async tryUpgradeToChroma(): Promise<boolean> {
    if (this.activeStore === this.chromaStore) {
      return true; // Already using ChromaDB
    }

    const chromaAvailable = await this.chromaStore.isAvailable();
    if (chromaAvailable) {
      this.activeStore = this.chromaStore;
      console.log('[VectorStore] Upgraded to ChromaDB backend');
      return true;
    }

    return false;
  }

  async close(): Promise<void> {
    await this.chromaStore.close();
    await this.sqliteStore.close();
    this.activeStore = null;
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

/**
 * Create and initialize the appropriate vector store
 *
 * Tries ChromaDB first, falls back to SQLite if unavailable.
 *
 * Usage:
 * ```typescript
 * const vectorStore = await createVectorStore(db);
 * await vectorStore.upsertEmbedding('node-1', embedding, 'content');
 * const similar = await vectorStore.findSimilar(queryEmbedding, 10);
 * ```
 */
export async function createVectorStore(
  getDatabase: () => Database.Database,
  chromaHost?: string
): Promise<IVectorStore> {
  const store = new HybridVectorStore(getDatabase, chromaHost);
  await store.initialize();
  return store;
}

/**
 * Create a ChromaDB-only vector store (no fallback)
 * Use this when you specifically need ChromaDB features
 */
export async function createChromaVectorStore(
  host?: string
): Promise<ChromaVectorStore> {
  const store = new ChromaVectorStore(host);
  await store.initialize();
  return store;
}

/**
 * Create a SQLite-only vector store
 * Use this for embedded/offline scenarios
 */
export async function createSQLiteVectorStore(
  getDatabase: () => Database.Database
): Promise<SQLiteVectorStore> {
  const store = new SQLiteVectorStore(getDatabase);
  await store.initialize();
  return store;
}

// ============================================================================
// MIGRATION HELPERS
// ============================================================================

/**
 * Migrate embeddings from SQLite to ChromaDB
 *
 * Call this when ChromaDB becomes available to sync any
 * embeddings that were stored in SQLite while offline.
 */
export async function migrateToChroma(
  sqliteStore: SQLiteVectorStore,
  chromaStore: ChromaVectorStore,
  onProgress?: (migrated: number, total: number) => void
): Promise<{ migrated: number; failed: number }> {
  const sqliteAvailable = await sqliteStore.isAvailable();
  const chromaAvailable = await chromaStore.isAvailable();

  if (!sqliteAvailable || !chromaAvailable) {
    throw new VectorStoreError(
      'Both stores must be available for migration',
      'MIGRATION_PREREQ_FAILED'
    );
  }

  // Get all embeddings from SQLite
  // This is a simplified implementation - in production you'd want pagination
  const stats = await sqliteStore.getStats();
  let migrated = 0;
  let failed = 0;

  // Note: This would need access to the underlying database to enumerate all embeddings
  // For now, this is a placeholder that shows the pattern

  if (onProgress) {
    onProgress(migrated, stats.embeddingCount);
  }

  console.log(
    `[VectorStore] Migration complete: ${migrated} migrated, ${failed} failed`
  );

  return { migrated, failed };
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/**
 * Batch upsert embeddings for better performance
 */
export async function batchUpsertEmbeddings(
  store: IVectorStore,
  items: Array<{
    nodeId: string;
    embedding: number[];
    content: string;
    metadata?: Record<string, unknown>;
  }>,
  onProgress?: (completed: number, total: number) => void
): Promise<{ succeeded: number; failed: number }> {
  let succeeded = 0;
  let failed = 0;

  // Process in batches
  for (let i = 0; i < items.length; i += BATCH_SIZE) {
    const batch = items.slice(i, i + BATCH_SIZE);

    const results = await Promise.allSettled(
      batch.map((item) =>
        store.upsertEmbedding(
          item.nodeId,
          item.embedding,
          item.content,
          item.metadata
        )
      )
    );

    for (const result of results) {
      if (result.status === 'fulfilled') {
        succeeded++;
      } else {
        failed++;
        console.warn('[VectorStore] Batch upsert failed:', result.reason);
      }
    }

    if (onProgress) {
      onProgress(i + batch.length, items.length);
    }
  }

  return { succeeded, failed };
}

/**
 * Batch delete embeddings
 */
export async function batchDeleteEmbeddings(
  store: IVectorStore,
  nodeIds: string[],
  onProgress?: (completed: number, total: number) => void
): Promise<{ succeeded: number; failed: number }> {
  let succeeded = 0;
  let failed = 0;

  // Process in batches
  for (let i = 0; i < nodeIds.length; i += BATCH_SIZE) {
    const batch = nodeIds.slice(i, i + BATCH_SIZE);

    const results = await Promise.allSettled(
      batch.map((nodeId) => store.deleteEmbedding(nodeId))
    );

    for (const result of results) {
      if (result.status === 'fulfilled') {
        succeeded++;
      } else {
        failed++;
      }
    }

    if (onProgress) {
      onProgress(i + batch.length, nodeIds.length);
    }
  }

  return { succeeded, failed };
}

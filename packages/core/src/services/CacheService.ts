import type { KnowledgeNode, PersonNode } from '../core/types.js';
import type { PaginatedResult } from '../repositories/PersonRepository.js';

// ============================================================================
// TYPES
// ============================================================================

/**
 * Represents a single entry in the cache with metadata for TTL and LRU eviction.
 */
export interface CacheEntry<T> {
  /** The cached value */
  value: T;
  /** Unix timestamp (ms) when this entry expires */
  expiresAt: number;
  /** Number of times this entry has been accessed */
  accessCount: number;
  /** Unix timestamp (ms) of the last access */
  lastAccessed: number;
  /** Estimated size in bytes (optional, for memory-based eviction) */
  size?: number;
}

/**
 * Configuration options for the cache service.
 */
export interface CacheOptions {
  /** Maximum number of entries in the cache */
  maxSize: number;
  /** Maximum memory usage in bytes (optional) */
  maxMemory?: number;
  /** Default TTL in milliseconds */
  defaultTTL: number;
  /** Interval in milliseconds for automatic cleanup of expired entries */
  cleanupInterval: number;
}

/**
 * Statistics about cache performance and state.
 */
export interface CacheStats {
  /** Current number of entries in the cache */
  size: number;
  /** Hit rate as a ratio (0-1) */
  hitRate: number;
  /** Estimated memory usage in bytes */
  memoryUsage: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const DEFAULT_OPTIONS: CacheOptions = {
  maxSize: 10000,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
  cleanupInterval: 60 * 1000, // 1 minute
};

// ============================================================================
// CACHE SERVICE
// ============================================================================

/**
 * A generic in-memory cache service with TTL support and LRU eviction.
 *
 * Features:
 * - Time-based expiration (TTL)
 * - LRU eviction when max size is reached
 * - Memory-based eviction (optional)
 * - Automatic cleanup of expired entries
 * - Pattern-based invalidation
 * - Cache-aside pattern support (getOrCompute)
 * - Hit rate tracking
 *
 * @template T The type of values stored in the cache
 */
export class CacheService<T = unknown> {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private options: CacheOptions;
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;
  private hits = 0;
  private misses = 0;
  private totalMemory = 0;

  constructor(options?: Partial<CacheOptions>) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.startCleanupTimer();
  }

  // --------------------------------------------------------------------------
  // PUBLIC METHODS
  // --------------------------------------------------------------------------

  /**
   * Get a value from cache.
   * Updates access metadata if the entry exists and is not expired.
   *
   * @param key The cache key
   * @returns The cached value, or undefined if not found or expired
   */
  get(key: string): T | undefined {
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return undefined;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.deleteEntry(key, entry);
      this.misses++;
      return undefined;
    }

    // Update access metadata
    entry.accessCount++;
    entry.lastAccessed = Date.now();
    this.hits++;

    return entry.value;
  }

  /**
   * Set a value in cache.
   * Performs LRU eviction if the cache is at capacity.
   *
   * @param key The cache key
   * @param value The value to cache
   * @param ttl Optional TTL in milliseconds (defaults to configured defaultTTL)
   */
  set(key: string, value: T, ttl?: number): void {
    const now = Date.now();
    const effectiveTTL = ttl ?? this.options.defaultTTL;
    const size = this.estimateSize(value);

    // If key already exists, remove old entry's size from total
    const existingEntry = this.cache.get(key);
    if (existingEntry) {
      this.totalMemory -= existingEntry.size ?? 0;
    }

    // Evict entries if needed (before adding new entry)
    this.evictIfNeeded(size);

    const entry: CacheEntry<T> = {
      value,
      expiresAt: now + effectiveTTL,
      accessCount: 0,
      lastAccessed: now,
      size,
    };

    this.cache.set(key, entry);
    this.totalMemory += size;
  }

  /**
   * Delete a key from cache.
   *
   * @param key The cache key to delete
   * @returns true if the key was deleted, false if it didn't exist
   */
  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (entry) {
      this.deleteEntry(key, entry);
      return true;
    }
    return false;
  }

  /**
   * Check if a key exists in cache and is not expired.
   *
   * @param key The cache key
   * @returns true if the key exists and is not expired
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    if (Date.now() > entry.expiresAt) {
      this.deleteEntry(key, entry);
      return false;
    }

    return true;
  }

  /**
   * Invalidate all keys matching a pattern.
   *
   * @param pattern A RegExp pattern to match keys against
   * @returns The number of keys invalidated
   */
  invalidatePattern(pattern: RegExp): number {
    let count = 0;
    const keysToDelete: string[] = [];

    for (const key of this.cache.keys()) {
      if (pattern.test(key)) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      const entry = this.cache.get(key);
      if (entry) {
        this.deleteEntry(key, entry);
        count++;
      }
    }

    return count;
  }

  /**
   * Clear all entries from the cache.
   */
  clear(): void {
    this.cache.clear();
    this.totalMemory = 0;
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Get or compute a value (cache-aside pattern).
   * If the key exists and is not expired, returns the cached value.
   * Otherwise, computes the value using the provided function, caches it, and returns it.
   *
   * @param key The cache key
   * @param compute A function that computes the value if not cached
   * @param ttl Optional TTL in milliseconds
   * @returns The cached or computed value
   */
  async getOrCompute(
    key: string,
    compute: () => Promise<T>,
    ttl?: number
  ): Promise<T> {
    // Try to get from cache first
    const cached = this.get(key);
    if (cached !== undefined) {
      return cached;
    }

    // Compute the value
    const value = await compute();

    // Cache and return
    this.set(key, value, ttl);
    return value;
  }

  /**
   * Get cache statistics.
   *
   * @returns Statistics about cache performance and state
   */
  stats(): CacheStats {
    const totalRequests = this.hits + this.misses;
    return {
      size: this.cache.size,
      hitRate: totalRequests > 0 ? this.hits / totalRequests : 0,
      memoryUsage: this.totalMemory,
    };
  }

  /**
   * Stop the cleanup timer and release resources.
   * Call this when the cache is no longer needed to prevent memory leaks.
   */
  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
    this.clear();
  }

  // --------------------------------------------------------------------------
  // PRIVATE METHODS
  // --------------------------------------------------------------------------

  /**
   * Start the automatic cleanup timer.
   */
  private startCleanupTimer(): void {
    if (this.options.cleanupInterval > 0) {
      this.cleanupTimer = setInterval(() => {
        this.cleanup();
      }, this.options.cleanupInterval);

      // Don't prevent Node.js from exiting if this is the only timer
      if (this.cleanupTimer.unref) {
        this.cleanupTimer.unref();
      }
    }
  }

  /**
   * Remove expired entries from the cache.
   */
  private cleanup(): void {
    const now = Date.now();
    const keysToDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      const entry = this.cache.get(key);
      if (entry) {
        this.deleteEntry(key, entry);
      }
    }
  }

  /**
   * Delete an entry and update memory tracking.
   */
  private deleteEntry(key: string, entry: CacheEntry<T>): void {
    this.totalMemory -= entry.size ?? 0;
    this.cache.delete(key);
  }

  /**
   * Evict entries if the cache is at capacity.
   * Uses LRU eviction strategy based on lastAccessed timestamp.
   * Also considers memory limits if configured.
   */
  private evictIfNeeded(incomingSize: number): void {
    // Evict for size limit
    while (this.cache.size >= this.options.maxSize) {
      this.evictLRU();
    }

    // Evict for memory limit if configured
    if (this.options.maxMemory) {
      while (
        this.totalMemory + incomingSize > this.options.maxMemory &&
        this.cache.size > 0
      ) {
        this.evictLRU();
      }
    }
  }

  /**
   * Evict the least recently used entry.
   * Finds the entry with the oldest lastAccessed timestamp and removes it.
   */
  private evictLRU(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache.entries()) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldestKey = key;
      }
    }

    if (oldestKey !== null) {
      const entry = this.cache.get(oldestKey);
      if (entry) {
        this.deleteEntry(oldestKey, entry);
      }
    }
  }

  /**
   * Estimate the memory size of a value in bytes.
   * This is a rough approximation for memory tracking purposes.
   */
  private estimateSize(value: T): number {
    if (value === null || value === undefined) {
      return 8;
    }

    const type = typeof value;

    if (type === 'boolean') {
      return 4;
    }

    if (type === 'number') {
      return 8;
    }

    if (type === 'string') {
      return (value as string).length * 2 + 40; // 2 bytes per char + overhead
    }

    if (Array.isArray(value)) {
      // For arrays, estimate based on length
      // This is a rough approximation
      return 40 + (value as unknown[]).length * 8;
    }

    if (type === 'object') {
      // For objects, use JSON serialization as a rough estimate
      try {
        const json = JSON.stringify(value);
        return json.length * 2 + 40;
      } catch {
        return 1024; // Default size for non-serializable objects
      }
    }

    return 8;
  }
}

// ============================================================================
// CACHE KEY HELPERS
// ============================================================================

/**
 * Standard cache key patterns for Vestige MCP.
 * These functions generate consistent cache keys for different entity types.
 */
export const CACHE_KEYS = {
  /** Cache key for a knowledge node by ID */
  node: (id: string): string => `node:${id}`,

  /** Cache key for a person by ID */
  person: (id: string): string => `person:${id}`,

  /** Cache key for search results */
  search: (query: string, opts: string): string => `search:${query}:${opts}`,

  /** Cache key for embeddings by node ID */
  embedding: (nodeId: string): string => `embedding:${nodeId}`,

  /** Cache key for related nodes */
  related: (nodeId: string, depth: number): string => `related:${nodeId}:${depth}`,

  /** Cache key for person by name */
  personByName: (name: string): string => `person:name:${name.toLowerCase()}`,

  /** Cache key for daily brief by date */
  dailyBrief: (date: string): string => `daily-brief:${date}`,
};

/**
 * Pattern matchers for cache invalidation.
 */
export const CACHE_PATTERNS = {
  /** All node-related entries */
  allNodes: /^node:/,

  /** All person-related entries */
  allPeople: /^person:/,

  /** All search results */
  allSearches: /^search:/,

  /** All embeddings */
  allEmbeddings: /^embedding:/,

  /** All related node entries */
  allRelated: /^related:/,

  /** Entries for a specific node and its related data */
  nodeAndRelated: (nodeId: string): RegExp =>
    new RegExp(`^(node:${nodeId}|related:${nodeId}|embedding:${nodeId})`),

  /** Entries for a specific person and related data */
  personAndRelated: (personId: string): RegExp =>
    new RegExp(`^person:(${personId}|name:)`),
};

// ============================================================================
// SPECIALIZED CACHE INSTANCES
// ============================================================================

/**
 * Cache for KnowledgeNode entities.
 * Longer TTL since nodes don't change frequently.
 */
export const nodeCache = new CacheService<KnowledgeNode>({
  maxSize: 5000,
  defaultTTL: 10 * 60 * 1000, // 10 minutes
  cleanupInterval: 2 * 60 * 1000, // 2 minutes
});

/**
 * Cache for search results.
 * Shorter TTL since search results can change with new data.
 */
export const searchCache = new CacheService<PaginatedResult<KnowledgeNode>>({
  maxSize: 1000,
  defaultTTL: 60 * 1000, // 1 minute
  cleanupInterval: 30 * 1000, // 30 seconds
});

/**
 * Cache for embedding vectors.
 * Longer TTL since embeddings don't change for existing content.
 */
export const embeddingCache = new CacheService<number[]>({
  maxSize: 10000,
  defaultTTL: 60 * 60 * 1000, // 1 hour
  cleanupInterval: 5 * 60 * 1000, // 5 minutes
});

/**
 * Cache for PersonNode entities.
 */
export const personCache = new CacheService<PersonNode>({
  maxSize: 2000,
  defaultTTL: 10 * 60 * 1000, // 10 minutes
  cleanupInterval: 2 * 60 * 1000, // 2 minutes
});

/**
 * Cache for related nodes queries.
 */
export const relatedCache = new CacheService<KnowledgeNode[]>({
  maxSize: 2000,
  defaultTTL: 5 * 60 * 1000, // 5 minutes
  cleanupInterval: 60 * 1000, // 1 minute
});

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Invalidate all caches related to a specific node.
 * Call this when a node is created, updated, or deleted.
 *
 * @param nodeId The ID of the node that changed
 */
export function invalidateNodeCaches(nodeId: string): void {
  nodeCache.delete(CACHE_KEYS.node(nodeId));
  embeddingCache.delete(CACHE_KEYS.embedding(nodeId));

  // Invalidate related entries and search results
  relatedCache.invalidatePattern(new RegExp(`^related:${nodeId}`));
  searchCache.clear(); // Search results may be affected
}

/**
 * Invalidate all caches related to a specific person.
 * Call this when a person is created, updated, or deleted.
 *
 * @param personId The ID of the person that changed
 * @param name Optional name to also invalidate name-based lookups
 */
export function invalidatePersonCaches(personId: string, name?: string): void {
  personCache.delete(CACHE_KEYS.person(personId));

  if (name) {
    personCache.delete(CACHE_KEYS.personByName(name));
  }

  // Search results may reference this person
  searchCache.clear();
}

/**
 * Clear all caches. Useful for testing or when major data changes occur.
 */
export function clearAllCaches(): void {
  nodeCache.clear();
  searchCache.clear();
  embeddingCache.clear();
  personCache.clear();
  relatedCache.clear();
}

/**
 * Get aggregated statistics from all caches.
 */
export function getAllCacheStats(): Record<string, CacheStats> {
  return {
    node: nodeCache.stats(),
    search: searchCache.stats(),
    embedding: embeddingCache.stats(),
    person: personCache.stats(),
    related: relatedCache.stats(),
  };
}

/**
 * Destroy all cache instances and stop cleanup timers.
 * Call this during application shutdown.
 */
export function destroyAllCaches(): void {
  nodeCache.destroy();
  searchCache.destroy();
  embeddingCache.destroy();
  personCache.destroy();
  relatedCache.destroy();
}

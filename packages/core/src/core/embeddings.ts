/**
 * Embeddings Service - Semantic Understanding for Vestige
 *
 * Provides vector embeddings for knowledge nodes using Ollama.
 * Embeddings enable semantic similarity search and connection discovery.
 *
 * Features:
 * - Ollama integration with nomic-embed-text model (768-dim, fast, high quality)
 * - Graceful fallback to TF-IDF when Ollama unavailable
 * - Availability caching to reduce connection overhead
 * - Batch embedding support for efficiency
 * - Utility functions for similarity search
 */

import { Ollama } from 'ollama';

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Ollama API endpoint. Defaults to local installation.
 */
const OLLAMA_HOST = process.env['OLLAMA_HOST'] || 'http://localhost:11434';

/**
 * Embedding model to use. nomic-embed-text provides:
 * - 768 dimensions
 * - Fast inference
 * - High quality embeddings for semantic search
 * - 8192 token context window
 */
const EMBEDDING_MODEL = process.env['VESTIGE_EMBEDDING_MODEL'] || 'nomic-embed-text';

/**
 * Maximum characters to embed. nomic-embed-text supports ~8192 tokens,
 * but we truncate to 8000 chars for safety margin.
 */
const MAX_TEXT_LENGTH = 8000;

/**
 * Cache duration for availability check (5 minutes in ms)
 */
const AVAILABILITY_CACHE_TTL = 5 * 60 * 1000;

/**
 * Default request timeout in milliseconds
 */
const DEFAULT_TIMEOUT = 30000;

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Service interface for generating and comparing text embeddings.
 * Provides semantic similarity capabilities for knowledge retrieval.
 */
export interface EmbeddingService {
  /**
   * Generate an embedding vector for the given text.
   * @param text - The text to embed
   * @returns A promise resolving to a numeric vector
   */
  generateEmbedding(text: string): Promise<number[]>;

  /**
   * Generate embeddings for multiple texts in a single batch.
   * More efficient than calling generateEmbedding multiple times.
   * @param texts - Array of texts to embed
   * @returns A promise resolving to an array of embedding vectors
   */
  batchEmbeddings(texts: string[]): Promise<number[][]>;

  /**
   * Calculate similarity between two embedding vectors.
   * @param embA - First embedding vector
   * @param embB - Second embedding vector
   * @returns Similarity score between 0 and 1
   */
  getSimilarity(embA: number[], embB: number[]): number;

  /**
   * Check if the embedding service is available and ready.
   * @returns A promise resolving to true if the service is available
   */
  isAvailable(): Promise<boolean>;
}

/**
 * Configuration options for embedding services.
 */
export interface EmbeddingServiceConfig {
  /** Ollama host URL (default: http://localhost:11434) */
  host?: string;
  /** Embedding model to use (default: nomic-embed-text) */
  model?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
}

/**
 * Result from embedding generation with metadata.
 */
export interface EmbeddingResult {
  embedding: number[];
  model: string;
  dimension: number;
}

// ============================================================================
// COSINE SIMILARITY
// ============================================================================

/**
 * Calculate cosine similarity between two vectors.
 * Returns a value between -1 and 1, where:
 * - 1 means identical direction
 * - 0 means orthogonal (unrelated)
 * - -1 means opposite direction
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Cosine similarity score
 * @throws Error if vectors have different lengths or are empty
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) {
    throw new Error('Cannot compute cosine similarity of empty vectors');
  }

  if (a.length !== b.length) {
    throw new Error(
      `Vector dimension mismatch: ${a.length} vs ${b.length}`
    );
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    const aVal = a[i]!;
    const bVal = b[i]!;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);

  if (magnitude === 0) {
    return 0;
  }

  return dotProduct / magnitude;
}

/**
 * Normalize cosine similarity from [-1, 1] to [0, 1] range.
 * Useful when you need a percentage-like similarity score.
 *
 * @param similarity - Cosine similarity value
 * @returns Normalized similarity between 0 and 1
 */
export function normalizedSimilarity(similarity: number): number {
  return (similarity + 1) / 2;
}

/**
 * Calculate Euclidean distance between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns Euclidean distance (lower = more similar)
 */
export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i]! - b[i]!;
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

// ============================================================================
// OLLAMA EMBEDDING SERVICE
// ============================================================================

/**
 * Production embedding service using Ollama with nomic-embed-text model.
 * Provides high-quality semantic embeddings for knowledge retrieval.
 *
 * Features:
 * - Automatic text truncation for long inputs
 * - Availability caching to reduce connection overhead
 * - Graceful error handling with informative messages
 * - Batch embedding support for efficiency
 */
export class OllamaEmbeddingService implements EmbeddingService {
  private client: Ollama;
  private availabilityCache: { available: boolean; timestamp: number } | null = null;
  private readonly model: string;
  private readonly timeout: number;

  constructor(config: EmbeddingServiceConfig = {}) {
    const {
      host = OLLAMA_HOST,
      model = EMBEDDING_MODEL,
      timeout = DEFAULT_TIMEOUT,
    } = config;

    this.client = new Ollama({ host });
    this.model = model;
    this.timeout = timeout;
  }

  /**
   * Check if Ollama is running and the embedding model is available.
   * Results are cached for 5 minutes to reduce overhead.
   */
  async isAvailable(): Promise<boolean> {
    // Check cache first
    if (
      this.availabilityCache &&
      Date.now() - this.availabilityCache.timestamp < AVAILABILITY_CACHE_TTL
    ) {
      return this.availabilityCache.available;
    }

    try {
      // Try to list models to verify connection with timeout
      const response = await Promise.race([
        this.client.list(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), this.timeout)
        ),
      ]);

      const modelNames = response.models.map((m) => m.name);

      // Check if our model is available (handle both "model" and "model:latest" formats)
      const modelBase = this.model.split(':')[0];
      const available = modelNames.some(
        (name) => name === this.model ||
                  name.startsWith(`${this.model}:`) ||
                  name.split(':')[0] === modelBase
      );

      if (!available) {
        console.warn(
          `Ollama is running but model '${this.model}' not found. ` +
            `Available models: ${modelNames.join(', ') || 'none'}. ` +
            `Run 'ollama pull ${this.model}' to install.`
        );
      }

      this.availabilityCache = { available, timestamp: Date.now() };
      return available;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`Ollama not available: ${message}`);
      this.availabilityCache = { available: false, timestamp: Date.now() };
      return false;
    }
  }

  /**
   * Truncate text to fit within the model's context window.
   */
  private truncateText(text: string): string {
    if (text.length <= MAX_TEXT_LENGTH) {
      return text;
    }
    console.warn(
      `Text truncated from ${text.length} to ${MAX_TEXT_LENGTH} characters`
    );
    return text.slice(0, MAX_TEXT_LENGTH);
  }

  /**
   * Generate an embedding for the given text.
   */
  async generateEmbedding(text: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error('Cannot generate embedding for empty text');
    }

    const truncatedText = this.truncateText(text.trim());

    try {
      const response = await Promise.race([
        this.client.embed({
          model: this.model,
          input: truncatedText,
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Embedding timeout')), this.timeout)
        ),
      ]);

      // Response contains array of embeddings, we want the first one
      if (!response.embeddings || response.embeddings.length === 0) {
        throw new Error('No embeddings returned from Ollama');
      }

      const embedding = response.embeddings[0];
      if (!embedding) {
        throw new Error('No embedding returned from Ollama');
      }

      return embedding;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to generate embedding: ${message}`);
    }
  }

  /**
   * Generate embeddings for multiple texts in a batch.
   * More efficient than individual calls for bulk operations.
   */
  async batchEmbeddings(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) {
      return [];
    }

    // Filter and truncate texts
    const validTexts = texts
      .filter((t) => t && t.trim().length > 0)
      .map((t) => this.truncateText(t.trim()));

    if (validTexts.length === 0) {
      return [];
    }

    try {
      const response = await Promise.race([
        this.client.embed({
          model: this.model,
          input: validTexts,
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Batch embedding timeout')), this.timeout * 2)
        ),
      ]);

      if (!response.embeddings || response.embeddings.length !== validTexts.length) {
        throw new Error(
          `Expected ${validTexts.length} embeddings, got ${response.embeddings?.length ?? 0}`
        );
      }

      return response.embeddings;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to generate batch embeddings: ${message}`);
    }
  }

  /**
   * Calculate similarity between two embedding vectors using cosine similarity.
   */
  getSimilarity(embA: number[], embB: number[]): number {
    return cosineSimilarity(embA, embB);
  }

  /**
   * Get the model being used.
   */
  getModel(): string {
    return this.model;
  }

  /**
   * Clear the availability cache, forcing a fresh check on next call.
   */
  clearCache(): void {
    this.availabilityCache = null;
  }
}

// ============================================================================
// FALLBACK EMBEDDING SERVICE
// ============================================================================

/**
 * Default vocabulary size for fallback TF-IDF style embeddings.
 */
const DEFAULT_VOCAB_SIZE = 512;

/**
 * Fallback embedding service using TF-IDF style word frequency vectors.
 * Used when Ollama is not available. Provides basic keyword-based
 * similarity that works offline with no dependencies.
 *
 * Limitations compared to Ollama:
 * - No semantic understanding (only keyword matching)
 * - Fixed vocabulary may miss domain-specific terms
 * - Lower quality similarity scores
 */
export class FallbackEmbeddingService implements EmbeddingService {
  private readonly dimensions: number;
  private readonly vocabulary: Map<string, number>;
  private documentFrequency: Map<string, number>;
  private documentCount: number;

  constructor(vocabSize: number = DEFAULT_VOCAB_SIZE) {
    this.dimensions = vocabSize;
    this.vocabulary = new Map();
    this.documentFrequency = new Map();
    this.documentCount = 0;
  }

  /**
   * Fallback service is always available (runs locally with no dependencies).
   */
  async isAvailable(): Promise<boolean> {
    return true;
  }

  /**
   * Tokenize text into normalized words.
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter((word) => word.length > 2 && word.length < 30);
  }

  /**
   * Get or assign a vocabulary index for a word.
   * Uses hash-based assignment for consistent but bounded vocabulary.
   */
  private getWordIndex(word: string): number {
    if (this.vocabulary.has(word)) {
      return this.vocabulary.get(word)!;
    }

    // Simple hash function for consistent word-to-index mapping
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      const char = word.charCodeAt(i);
      hash = ((hash << 5) - hash + char) | 0;
    }
    const index = Math.abs(hash) % this.dimensions;
    this.vocabulary.set(word, index);
    return index;
  }

  /**
   * Generate a TF-IDF style embedding vector.
   * Uses term frequency weighted by inverse document frequency approximation.
   */
  async generateEmbedding(text: string): Promise<number[]> {
    if (!text || text.trim().length === 0) {
      throw new Error('Cannot generate embedding for empty text');
    }

    const tokens = this.tokenize(text);
    if (tokens.length === 0) {
      // Return zero vector for text with no valid tokens
      return new Array(this.dimensions).fill(0);
    }

    // Calculate term frequency
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) || 0) + 1);
    }

    // Update document frequency for IDF
    this.documentCount++;
    const seenWords = new Set<string>();
    for (const token of tokens) {
      if (!seenWords.has(token)) {
        this.documentFrequency.set(
          token,
          (this.documentFrequency.get(token) || 0) + 1
        );
        seenWords.add(token);
      }
    }

    // Build embedding vector
    const embedding = new Array(this.dimensions).fill(0);
    const maxFreq = Math.max(...termFreq.values());

    for (const [word, freq] of termFreq) {
      const index = this.getWordIndex(word);

      // TF: normalized term frequency (prevents bias towards long documents)
      const tf = freq / maxFreq;

      // IDF: inverse document frequency (common words get lower weight)
      const df = this.documentFrequency.get(word) || 1;
      const idf = Math.log((this.documentCount + 1) / (df + 1)) + 1;

      // TF-IDF score (may have collisions, add to handle)
      embedding[index] += tf * idf;
    }

    // L2 normalize the vector
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }

  /**
   * Generate embeddings for multiple texts.
   */
  async batchEmbeddings(texts: string[]): Promise<number[][]> {
    const embeddings: number[][] = [];
    for (const text of texts) {
      if (text && text.trim().length > 0) {
        embeddings.push(await this.generateEmbedding(text));
      }
    }
    return embeddings;
  }

  /**
   * Calculate similarity between two embedding vectors.
   */
  getSimilarity(embA: number[], embB: number[]): number {
    return cosineSimilarity(embA, embB);
  }

  /**
   * Reset the document frequency statistics.
   * Useful when starting fresh with a new corpus.
   */
  reset(): void {
    this.vocabulary.clear();
    this.documentFrequency.clear();
    this.documentCount = 0;
  }

  /**
   * Get the dimensionality of embeddings.
   */
  getDimensions(): number {
    return this.dimensions;
  }
}

// ============================================================================
// EMBEDDING CACHE
// ============================================================================

/**
 * Simple in-memory cache for embeddings.
 * Reduces redundant API calls during REM cycles.
 */
export class EmbeddingCache {
  private cache: Map<string, { embedding: number[]; timestamp: number }> = new Map();
  private maxSize: number;
  private ttlMs: number;

  constructor(maxSize: number = 1000, ttlMinutes: number = 60) {
    this.maxSize = maxSize;
    this.ttlMs = ttlMinutes * 60 * 1000;
  }

  /**
   * Get a cached embedding by node ID.
   */
  get(nodeId: string): number[] | null {
    const entry = this.cache.get(nodeId);
    if (!entry) return null;

    // Check if expired
    if (Date.now() - entry.timestamp > this.ttlMs) {
      this.cache.delete(nodeId);
      return null;
    }

    return entry.embedding;
  }

  /**
   * Cache an embedding for a node ID.
   */
  set(nodeId: string, embedding: number[]): void {
    // Evict oldest if at capacity
    if (this.cache.size >= this.maxSize) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(nodeId, {
      embedding,
      timestamp: Date.now(),
    });
  }

  /**
   * Check if a node ID has a cached embedding.
   */
  has(nodeId: string): boolean {
    return this.get(nodeId) !== null;
  }

  /**
   * Clear all cached embeddings.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get the number of cached embeddings.
   */
  size(): number {
    return this.cache.size;
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

let defaultService: EmbeddingService | null = null;

/**
 * Get the default embedding service (singleton).
 * Uses cached instance for efficiency.
 */
export function getEmbeddingService(config?: EmbeddingServiceConfig): EmbeddingService {
  if (!defaultService) {
    defaultService = new OllamaEmbeddingService(config);
  }
  return defaultService;
}

/**
 * Create an embedding service with automatic fallback.
 *
 * Attempts to use Ollama with nomic-embed-text for high-quality semantic
 * embeddings. Falls back to TF-IDF based keyword similarity if Ollama
 * is not available.
 *
 * @param config - Optional configuration for the Ollama service
 * @returns A promise resolving to an EmbeddingService instance
 *
 * @example
 * ```typescript
 * const embeddings = await createEmbeddingService();
 *
 * const vec1 = await embeddings.generateEmbedding("TypeScript is great");
 * const vec2 = await embeddings.generateEmbedding("JavaScript is popular");
 *
 * const similarity = embeddings.getSimilarity(vec1, vec2);
 * console.log(`Similarity: ${similarity}`);
 * ```
 */
export async function createEmbeddingService(
  config?: EmbeddingServiceConfig
): Promise<EmbeddingService> {
  const ollama = new OllamaEmbeddingService(config);

  if (await ollama.isAvailable()) {
    console.log(`Using Ollama embedding service with model: ${config?.model || EMBEDDING_MODEL}`);
    return ollama;
  }

  console.warn(
    'Ollama not available, using fallback keyword similarity. ' +
      'For better results, install Ollama and run: ollama pull nomic-embed-text'
  );
  return new FallbackEmbeddingService();
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Find the top K most similar items to a query embedding.
 *
 * @param queryEmbedding - The embedding to search for
 * @param candidates - Array of items with embeddings
 * @param k - Number of results to return
 * @returns Top K items sorted by similarity (highest first)
 *
 * @example
 * ```typescript
 * const results = findTopK(queryVec, documents, 10);
 * results.forEach(({ item, similarity }) => {
 *   console.log(`${item.title}: ${similarity.toFixed(3)}`);
 * });
 * ```
 */
export function findTopK<T extends { embedding: number[] }>(
  queryEmbedding: number[],
  candidates: T[],
  k: number
): Array<T & { similarity: number }> {
  const scored = candidates.map((item) => ({
    ...item,
    similarity: cosineSimilarity(queryEmbedding, item.embedding),
  }));

  scored.sort((a, b) => b.similarity - a.similarity);

  return scored.slice(0, k);
}

/**
 * Filter items by minimum similarity threshold.
 *
 * @param queryEmbedding - The embedding to search for
 * @param candidates - Array of items with embeddings
 * @param minSimilarity - Minimum similarity score (-1 to 1)
 * @returns Items with similarity >= minSimilarity, sorted by similarity
 *
 * @example
 * ```typescript
 * const relevant = filterBySimilarity(queryVec, documents, 0.7);
 * console.log(`Found ${relevant.length} relevant documents`);
 * ```
 */
export function filterBySimilarity<T extends { embedding: number[] }>(
  queryEmbedding: number[],
  candidates: T[],
  minSimilarity: number
): Array<T & { similarity: number }> {
  const scored = candidates
    .map((item) => ({
      ...item,
      similarity: cosineSimilarity(queryEmbedding, item.embedding),
    }))
    .filter((item) => item.similarity >= minSimilarity);

  scored.sort((a, b) => b.similarity - a.similarity);

  return scored;
}

/**
 * Compute average embedding from multiple vectors.
 * Useful for combining multiple documents into a single representation.
 *
 * @param embeddings - Array of embedding vectors
 * @returns Average embedding vector
 */
export function averageEmbedding(embeddings: number[][]): number[] {
  if (embeddings.length === 0) {
    throw new Error('Cannot compute average of empty embedding array');
  }

  const firstEmbedding = embeddings[0];
  if (!firstEmbedding) {
    throw new Error('Cannot compute average of empty embedding array');
  }

  const dimensions = firstEmbedding.length;
  const result = new Array<number>(dimensions).fill(0);

  for (const embedding of embeddings) {
    if (embedding.length !== dimensions) {
      throw new Error('All embeddings must have the same dimensions');
    }
    for (let i = 0; i < dimensions; i++) {
      result[i]! += embedding[i]!;
    }
  }

  for (let i = 0; i < dimensions; i++) {
    result[i]! /= embeddings.length;
  }

  return result;
}

/**
 * Configuration Management for Vestige MCP
 *
 * Provides centralized configuration with:
 * - Zod schema validation
 * - File-based configuration (~/.vestige/config.json)
 * - Environment variable overrides
 * - Type-safe accessors for all config sections
 *
 * Configuration priority (highest to lowest):
 * 1. Environment variables
 * 2. Config file
 * 3. Default values
 */

import { z } from 'zod';
import path from 'path';
import os from 'os';
import fs from 'fs';

// ============================================================================
// CONFIGURATION SCHEMA
// ============================================================================

/**
 * Database configuration schema
 */
const DatabaseConfigSchema = z.object({
  /** Path to the SQLite database file */
  path: z.string().default(path.join(os.homedir(), '.vestige', 'vestige.db')),
  /** Directory for database backups */
  backupDir: z.string().default(path.join(os.homedir(), '.vestige', 'backups')),
  /** SQLite busy timeout in milliseconds */
  busyTimeout: z.number().default(5000),
  /** SQLite cache size in pages (negative = KB) */
  cacheSize: z.number().default(64000),
  /** Maximum number of backup files to retain */
  maxBackups: z.number().default(5),
}).default({});

/**
 * FSRS (Free Spaced Repetition Scheduler) algorithm configuration
 * Named with 'Config' prefix to avoid collision with FSRSConfigSchema in fsrs.ts
 */
const ConfigFSRSSchema = z.object({
  /** Target retention rate (0.7 to 0.99) */
  desiredRetention: z.number().min(0.7).max(0.99).default(0.9),
  /** Custom FSRS-5 weights (19 values). If not provided, uses defaults. */
  weights: z.array(z.number()).length(19).optional(),
  /** Enable personalized scheduling based on review history */
  enablePersonalization: z.boolean().default(false),
}).default({});

/**
 * Dual-strength memory model configuration
 * Based on the distinction between storage strength and retrieval strength
 */
const MemoryConfigSchema = z.object({
  /** Storage strength boost on passive access (read) */
  storageBoostOnAccess: z.number().default(0.05),
  /** Storage strength boost on active review */
  storageBoostOnReview: z.number().default(0.1),
  /** Half-life for retrieval strength decay in days */
  retrievalDecayHalfLife: z.number().default(7),
  /** Minimum retention strength before memory is considered weak */
  minRetentionStrength: z.number().default(0.1),
}).default({});

/**
 * Sentiment analysis configuration for emotional memory weighting
 */
const SentimentConfigSchema = z.object({
  /** Stability multiplier for highly emotional memories */
  stabilityBoost: z.number().default(2.0),
  /** Minimum boost applied to any memory */
  minBoost: z.number().default(1.0),
}).default({});

/**
 * REM (Rapid Eye Movement) cycle configuration
 * Handles memory consolidation and connection discovery
 */
const REMConfigSchema = z.object({
  /** Enable REM cycle processing */
  enabled: z.boolean().default(true),
  /** Maximum number of memories to analyze per cycle */
  maxAnalyze: z.number().default(50),
  /** Minimum connection strength to create an edge */
  minConnectionStrength: z.number().default(0.3),
  /** Half-life for temporal proximity weighting in days */
  temporalHalfLifeDays: z.number().default(7),
  /** Decay factor for spreading activation (0-1) */
  spreadingActivationDecay: z.number().default(0.8),
}).default({});

/**
 * Memory consolidation configuration
 * Controls the background process that strengthens important memories
 */
const ConsolidationConfigSchema = z.object({
  /** Enable automatic consolidation */
  enabled: z.boolean().default(true),
  /** Hour of day to run consolidation (0-23) */
  scheduleHour: z.number().min(0).max(23).default(3),
  /** Window in hours for short-term memory processing */
  shortTermWindowHours: z.number().default(24),
  /** Minimum importance score for consolidation */
  importanceThreshold: z.number().default(0.5),
  /** Threshold below which memories may be pruned */
  pruneThreshold: z.number().default(0.2),
}).default({});

/**
 * Embeddings service configuration
 */
const EmbeddingsConfigSchema = z.object({
  /** Embedding provider to use */
  provider: z.enum(['ollama', 'fallback']).default('ollama'),
  /** Ollama API host URL */
  ollamaHost: z.string().default('http://localhost:11434'),
  /** Embedding model name */
  model: z.string().default('nomic-embed-text'),
  /** Maximum text length to embed (characters) */
  maxTextLength: z.number().default(8000),
}).default({});

/**
 * Vector store configuration for semantic search
 */
const VectorStoreConfigSchema = z.object({
  /** Vector store provider */
  provider: z.enum(['chromadb', 'sqlite']).default('chromadb'),
  /** ChromaDB host URL */
  chromaHost: z.string().default('http://localhost:8000'),
  /** Name of the embeddings collection */
  collectionName: z.string().default('vestige_embeddings'),
}).default({});

/**
 * Cache configuration
 */
const CacheConfigSchema = z.object({
  /** Enable caching */
  enabled: z.boolean().default(true),
  /** Maximum number of items in cache */
  maxSize: z.number().default(10000),
  /** Default time-to-live in milliseconds */
  defaultTTLMs: z.number().default(5 * 60 * 1000),
}).default({});

/**
 * Logging configuration
 */
const LoggingConfigSchema = z.object({
  /** Minimum log level */
  level: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  /** Use structured JSON logging */
  structured: z.boolean().default(true),
}).default({});

/**
 * Input/output limits configuration
 */
const LimitsConfigSchema = z.object({
  /** Maximum content length in characters */
  maxContentLength: z.number().default(1_000_000),
  /** Maximum name/title length in characters */
  maxNameLength: z.number().default(500),
  /** Maximum query length in characters */
  maxQueryLength: z.number().default(10_000),
  /** Maximum number of tags per item */
  maxTagsCount: z.number().default(100),
  /** Maximum items per batch operation */
  maxBatchSize: z.number().default(1000),
  /** Default pagination limit */
  paginationDefault: z.number().default(50),
  /** Maximum pagination limit */
  paginationMax: z.number().default(500),
}).default({});

/**
 * Main configuration schema combining all sections
 */
const ConfigSchema = z.object({
  database: DatabaseConfigSchema,
  fsrs: ConfigFSRSSchema,
  memory: MemoryConfigSchema,
  sentiment: SentimentConfigSchema,
  rem: REMConfigSchema,
  consolidation: ConsolidationConfigSchema,
  embeddings: EmbeddingsConfigSchema,
  vectorStore: VectorStoreConfigSchema,
  cache: CacheConfigSchema,
  logging: LoggingConfigSchema,
  limits: LimitsConfigSchema,
});

/**
 * Inferred TypeScript type from the Zod schema
 */
export type VestigeConfig = z.infer<typeof ConfigSchema>;

// ============================================================================
// CONFIGURATION LOADING
// ============================================================================

/**
 * Singleton configuration instance
 */
let config: VestigeConfig | null = null;

/**
 * Partial configuration type for environment overrides
 */
interface PartialVestigeConfig {
  database?: {
    path?: string;
    backupDir?: string;
  };
  logging?: {
    level?: string;
  };
  embeddings?: {
    ollamaHost?: string;
    model?: string;
  };
  vectorStore?: {
    chromaHost?: string;
  };
  fsrs?: {
    desiredRetention?: number;
  };
  rem?: {
    enabled?: boolean;
  };
  consolidation?: {
    enabled?: boolean;
  };
}

/**
 * Load environment variable overrides
 * Environment variables take precedence over file configuration
 */
function loadEnvConfig(): PartialVestigeConfig {
  const env: PartialVestigeConfig = {};

  // Database configuration
  const dbPath = process.env['VESTIGE_DB_PATH'];
  const backupDir = process.env['VESTIGE_BACKUP_DIR'];
  if (dbPath || backupDir) {
    env.database = {};
    if (dbPath) env.database.path = dbPath;
    if (backupDir) env.database.backupDir = backupDir;
  }

  // Logging configuration
  const logLevel = process.env['VESTIGE_LOG_LEVEL'];
  if (logLevel) {
    env.logging = { level: logLevel };
  }

  // Embeddings configuration
  const ollamaHost = process.env['OLLAMA_HOST'];
  const embeddingModel = process.env['VESTIGE_EMBEDDING_MODEL'];
  if (ollamaHost || embeddingModel) {
    env.embeddings = {};
    if (ollamaHost) env.embeddings.ollamaHost = ollamaHost;
    if (embeddingModel) env.embeddings.model = embeddingModel;
  }

  // Vector store configuration
  const chromaHost = process.env['CHROMA_HOST'];
  if (chromaHost) {
    env.vectorStore = { chromaHost };
  }

  // FSRS configuration
  const desiredRetention = process.env['VESTIGE_DESIRED_RETENTION'];
  if (desiredRetention) {
    const retention = parseFloat(desiredRetention);
    if (!isNaN(retention)) {
      env.fsrs = { desiredRetention: retention };
    }
  }

  // REM configuration
  const remEnabled = process.env['VESTIGE_REM_ENABLED'];
  if (remEnabled) {
    const enabled = remEnabled.toLowerCase() === 'true';
    env.rem = { enabled };
  }

  // Consolidation configuration
  const consolidationEnabled = process.env['VESTIGE_CONSOLIDATION_ENABLED'];
  if (consolidationEnabled) {
    const enabled = consolidationEnabled.toLowerCase() === 'true';
    env.consolidation = { enabled };
  }

  return env;
}

/**
 * Deep merge two objects, with source taking precedence
 */
function deepMerge<T extends Record<string, unknown>>(target: T, source: Partial<T>): T {
  const result = { ...target };

  for (const key of Object.keys(source) as (keyof T)[]) {
    const sourceValue = source[key];
    const targetValue = result[key];

    if (
      sourceValue !== undefined &&
      typeof sourceValue === 'object' &&
      sourceValue !== null &&
      !Array.isArray(sourceValue) &&
      typeof targetValue === 'object' &&
      targetValue !== null &&
      !Array.isArray(targetValue)
    ) {
      result[key] = deepMerge(
        targetValue as Record<string, unknown>,
        sourceValue as Record<string, unknown>
      ) as T[keyof T];
    } else if (sourceValue !== undefined) {
      result[key] = sourceValue as T[keyof T];
    }
  }

  return result;
}

/**
 * Load configuration from file and environment variables
 *
 * @param customPath - Optional custom path to config file
 * @returns Validated configuration object
 */
export function loadConfig(customPath?: string): VestigeConfig {
  if (config) return config;

  const configPath = customPath || path.join(os.homedir(), '.vestige', 'config.json');
  let fileConfig: Record<string, unknown> = {};

  // Load from file if it exists
  if (fs.existsSync(configPath)) {
    try {
      const content = fs.readFileSync(configPath, 'utf-8');
      fileConfig = JSON.parse(content) as Record<string, unknown>;
    } catch (error) {
      console.warn(`Failed to load config from ${configPath}:`, error);
    }
  }

  // Load environment variable overrides
  const envConfig = loadEnvConfig();

  // Merge configs: file config first, then env overrides
  const mergedConfig = deepMerge(fileConfig, envConfig as Record<string, unknown>);

  // Validate and parse with Zod (applies defaults)
  config = ConfigSchema.parse(mergedConfig);

  return config;
}

/**
 * Get the current configuration, loading it if necessary
 *
 * @returns The current configuration object
 */
export function getConfig(): VestigeConfig {
  if (!config) {
    return loadConfig();
  }
  return config;
}

/**
 * Reset the configuration singleton (useful for testing)
 */
export function resetConfig(): void {
  config = null;
}

// ============================================================================
// CONFIGURATION ACCESSORS
// ============================================================================

/**
 * Get database configuration
 */
export const getDatabaseConfig = () => getConfig().database;

/**
 * Get FSRS algorithm configuration
 */
export const getFSRSConfig = () => getConfig().fsrs;

/**
 * Get memory model configuration
 */
export const getMemoryConfig = () => getConfig().memory;

/**
 * Get sentiment analysis configuration
 */
export const getSentimentConfig = () => getConfig().sentiment;

/**
 * Get REM cycle configuration
 */
export const getREMConfig = () => getConfig().rem;

/**
 * Get consolidation configuration
 */
export const getConsolidationConfig = () => getConfig().consolidation;

/**
 * Get embeddings service configuration
 */
export const getEmbeddingsConfig = () => getConfig().embeddings;

/**
 * Get vector store configuration
 */
export const getVectorStoreConfig = () => getConfig().vectorStore;

/**
 * Get cache configuration
 */
export const getCacheConfig = () => getConfig().cache;

/**
 * Get logging configuration
 */
export const getLoggingConfig = () => getConfig().logging;

/**
 * Get limits configuration
 */
export const getLimitsConfig = () => getConfig().limits;

// ============================================================================
// CONFIGURATION VALIDATION
// ============================================================================

/**
 * Validate an unknown config object against the schema
 *
 * @param configObj - Unknown object to validate
 * @returns Validated configuration object
 * @throws ZodError if validation fails
 */
export function validateConfig(configObj: unknown): VestigeConfig {
  return ConfigSchema.parse(configObj);
}

/**
 * Get the Zod schema for configuration validation
 *
 * @returns The Zod configuration schema
 */
export function getConfigSchema() {
  return ConfigSchema;
}

// ============================================================================
// EXPORTS
// ============================================================================

// Export individual schemas for external use
export {
  ConfigSchema,
  DatabaseConfigSchema,
  ConfigFSRSSchema,
  MemoryConfigSchema,
  SentimentConfigSchema,
  REMConfigSchema,
  ConsolidationConfigSchema,
  EmbeddingsConfigSchema,
  VectorStoreConfigSchema,
  CacheConfigSchema,
  LoggingConfigSchema,
  LimitsConfigSchema,
};

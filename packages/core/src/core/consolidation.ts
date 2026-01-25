/**
 * Sleep Consolidation Simulation
 *
 * "The brain that consolidates while you sleep."
 *
 * This module simulates how the human brain consolidates memories during sleep.
 * Based on cognitive science research on memory consolidation, it implements:
 *
 * KEY FEATURES:
 * 1. Short-term Memory Processing - Identifies recent memories for consolidation
 * 2. Importance-based Promotion - Promotes significant memories to long-term storage
 * 3. REM Cycle Integration - Discovers new connections via semantic analysis
 * 4. Synaptic Homeostasis - Prunes weak connections to prevent memory overload
 * 5. Decay Application - Applies natural memory decay based on forgetting curve
 *
 * COGNITIVE SCIENCE BASIS:
 * - Active Systems Consolidation: Hippocampus replays memories during sleep
 * - Synaptic Homeostasis Hypothesis: Weak connections are pruned during sleep
 * - Emotional Memory Enhancement: Emotional memories are preferentially consolidated
 * - Spreading Activation: Related memories are co-activated and strengthened
 */

import { VestigeDatabase } from './database.js';
import { runREMCycle } from './rem-cycle.js';
import type { KnowledgeNode } from './types.js';
import { logger } from '../utils/logger.js';

// ============================================================================
// TYPES
// ============================================================================

export interface ConsolidationResult {
  /** Number of short-term memories processed */
  shortTermProcessed: number;
  /** Number of memories promoted to long-term storage */
  promotedToLongTerm: number;
  /** Number of new connections discovered via REM cycle */
  connectionsDiscovered: number;
  /** Number of weak edges pruned (synaptic homeostasis) */
  edgesPruned: number;
  /** Number of memories that had decay applied */
  decayApplied: number;
  /** Duration of consolidation cycle in milliseconds */
  duration: number;
}

export interface ConsolidationOptions {
  /** Hours to look back for short-term memories. Default: 24 */
  shortTermWindowHours?: number;
  /** Minimum importance score to promote to long-term. Default: 0.5 */
  importanceThreshold?: number;
  /** Edge weight below which connections are pruned. Default: 0.2 */
  pruneThreshold?: number;
  /** Maximum number of memories to analyze in REM cycle. Default: 100 */
  maxAnalyze?: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default short-term memory window (24 hours) */
const DEFAULT_SHORT_TERM_WINDOW_HOURS = 24;

/** Default importance threshold for long-term promotion */
const DEFAULT_IMPORTANCE_THRESHOLD = 0.5;

/** Default edge weight threshold for pruning */
const DEFAULT_PRUNE_THRESHOLD = 0.2;

/** Default max memories to analyze */
const DEFAULT_MAX_ANALYZE = 100;

/** Weight factors for importance calculation */
const EMOTION_WEIGHT = 0.4;
const ACCESS_WEIGHT = 0.3;
const CONNECTION_WEIGHT = 0.3;

/** Maximum values for normalization */
const MAX_ACCESSES_FOR_IMPORTANCE = 5;
const MAX_CONNECTIONS_FOR_IMPORTANCE = 5;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get memories created within the short-term window
 * These are candidates for consolidation processing
 */
async function getShortTermMemories(
  db: VestigeDatabase,
  windowHours: number
): Promise<KnowledgeNode[]> {
  const windowStart = new Date(Date.now() - windowHours * 60 * 60 * 1000);
  const recentNodes = db.getRecentNodes({ limit: 500 }).items;
  return recentNodes.filter(node => node.createdAt >= windowStart);
}

/**
 * Calculate importance score for a memory
 *
 * Importance = f(emotion, access_count, connection_count)
 *
 * The formula weights three factors:
 * - Emotional intensity (40%): Emotionally charged memories are more important
 * - Access count (30%): Frequently accessed memories are more important
 * - Connection count (30%): Well-connected memories are more important
 *
 * @returns Importance score from 0 to 1
 */
function calculateImportance(db: VestigeDatabase, memory: KnowledgeNode): number {
  // Get connection count for this memory
  const connections = db.getRelatedNodes(memory.id, 1).length;

  // Get emotional intensity (0 to 1)
  const emotion = memory.sentimentIntensity || 0;

  // Get access count
  const accesses = memory.accessCount;

  // Weighted importance formula
  // Each component is normalized to 0-1 range
  const emotionScore = emotion * EMOTION_WEIGHT;
  const accessScore =
    (Math.min(MAX_ACCESSES_FOR_IMPORTANCE, accesses) / MAX_ACCESSES_FOR_IMPORTANCE) *
    ACCESS_WEIGHT;
  const connectionScore =
    (Math.min(MAX_CONNECTIONS_FOR_IMPORTANCE, connections) / MAX_CONNECTIONS_FOR_IMPORTANCE) *
    CONNECTION_WEIGHT;

  const importanceScore = emotionScore + accessScore + connectionScore;

  return importanceScore;
}

/**
 * Promote a memory to long-term storage
 *
 * This boosts the storage strength proportional to importance.
 * Based on the Dual-Strength Memory Model (Bjork & Bjork, 1992),
 * storage strength represents how well the memory is encoded.
 *
 * Boost factor ranges from 1x (importance=0) to 3x (importance=1)
 */
async function promoteToLongTerm(
  db: VestigeDatabase,
  nodeId: string,
  importance: number
): Promise<void> {
  // Calculate boost factor: 1x to 3x based on importance
  const boost = 1 + importance * 2;

  // Access the internal database connection
  // Note: This uses internal access pattern for direct SQL operations
  const internalDb = (db as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } }).db;

  internalDb
    .prepare(
      `
    UPDATE knowledge_nodes
    SET storage_strength = storage_strength * ?,
        stability_factor = stability_factor * ?
    WHERE id = ?
  `
    )
    .run(boost, boost, nodeId);
}

/**
 * Prune weak connections discovered by REM cycle
 *
 * This implements synaptic homeostasis - the brain's process of
 * removing weak synaptic connections during sleep to:
 * 1. Prevent memory overload
 * 2. Improve signal-to-noise ratio
 * 3. Conserve metabolic resources
 *
 * Only auto-discovered connections (from REM cycle) are pruned.
 * User-created connections are preserved regardless of weight.
 */
async function pruneWeakConnections(
  db: VestigeDatabase,
  threshold: number
): Promise<number> {
  // Access the internal database connection
  const internalDb = (db as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => { changes: number } } } }).db;

  // Remove edges below threshold that were auto-discovered by REM cycle
  const result = internalDb
    .prepare(
      `
    DELETE FROM graph_edges
    WHERE weight < ?
      AND json_extract(metadata, '$.discoveredBy') = 'rem_cycle'
  `
    )
    .run(threshold);

  return result.changes;
}

// ============================================================================
// MAIN CONSOLIDATION FUNCTION
// ============================================================================

/**
 * Run Sleep Consolidation Simulation
 *
 * Based on cognitive science research on memory consolidation:
 *
 * PHASE 1: Identify short-term memories
 *   - Collect memories created within the specified window
 *   - These represent the "inbox" of memories to process
 *
 * PHASE 2: Calculate importance and promote
 *   - Score each memory based on emotion, access, connections
 *   - Memories above threshold are "promoted" (strengthened)
 *   - This simulates hippocampal replay during sleep
 *
 * PHASE 3: Run REM cycle for connection discovery
 *   - Analyze memories for semantic similarity
 *   - Discover new connections between related memories
 *   - Apply spreading activation for transitive connections
 *
 * PHASE 4: Prune weak connections (synaptic homeostasis)
 *   - Remove auto-discovered edges below weight threshold
 *   - Preserves signal-to-noise ratio in memory network
 *
 * PHASE 5: Apply decay to all memories
 *   - Apply Ebbinghaus forgetting curve
 *   - Emotional memories decay slower
 *   - Well-encoded memories (high storage strength) decay slower
 *
 * @param db - VestigeDatabase instance
 * @param options - Consolidation configuration options
 * @returns Results of the consolidation cycle
 */
export async function runConsolidation(
  db: VestigeDatabase,
  options: ConsolidationOptions = {}
): Promise<ConsolidationResult> {
  const startTime = Date.now();
  const {
    shortTermWindowHours = DEFAULT_SHORT_TERM_WINDOW_HOURS,
    importanceThreshold = DEFAULT_IMPORTANCE_THRESHOLD,
    pruneThreshold = DEFAULT_PRUNE_THRESHOLD,
    maxAnalyze = DEFAULT_MAX_ANALYZE,
  } = options;

  const result: ConsolidationResult = {
    shortTermProcessed: 0,
    promotedToLongTerm: 0,
    connectionsDiscovered: 0,
    edgesPruned: 0,
    decayApplied: 0,
    duration: 0,
  };

  logger.info('Starting consolidation cycle', {
    shortTermWindowHours,
    importanceThreshold,
    pruneThreshold,
    maxAnalyze,
  });

  // PHASE 1: Identify short-term memories
  // These are memories created within the window that need processing
  const shortTermMemories = await getShortTermMemories(db, shortTermWindowHours);
  result.shortTermProcessed = shortTermMemories.length;

  logger.debug('Phase 1: Identified short-term memories', {
    count: shortTermMemories.length,
  });

  // PHASE 2: Calculate importance and promote to long-term
  // This simulates the hippocampal replay that occurs during sleep
  for (const memory of shortTermMemories) {
    const importance = calculateImportance(db, memory);
    if (importance >= importanceThreshold) {
      await promoteToLongTerm(db, memory.id, importance);
      result.promotedToLongTerm++;
    }
  }

  logger.debug('Phase 2: Promoted memories to long-term storage', {
    promoted: result.promotedToLongTerm,
    threshold: importanceThreshold,
  });

  // PHASE 3: Run REM cycle for connection discovery
  // This discovers semantic connections between memories
  const remResult = await runREMCycle(db, { maxAnalyze });
  result.connectionsDiscovered = remResult.connectionsCreated;

  logger.debug('Phase 3: REM cycle complete', {
    connectionsDiscovered: remResult.connectionsDiscovered,
    connectionsCreated: remResult.connectionsCreated,
    spreadingActivationEdges: remResult.spreadingActivationEdges,
  });

  // PHASE 4: Prune weak connections (synaptic homeostasis)
  // Remove auto-discovered connections that are below the threshold
  result.edgesPruned = await pruneWeakConnections(db, pruneThreshold);

  logger.debug('Phase 4: Pruned weak connections', {
    edgesPruned: result.edgesPruned,
    threshold: pruneThreshold,
  });

  // PHASE 5: Apply decay to all memories
  // Uses Ebbinghaus forgetting curve with emotional weighting
  result.decayApplied = db.applyDecay();

  logger.debug('Phase 5: Applied memory decay', {
    memoriesAffected: result.decayApplied,
  });

  result.duration = Date.now() - startTime;
  logger.info('Consolidation cycle complete', { ...result });

  return result;
}

// ============================================================================
// SCHEDULING HELPER
// ============================================================================

/**
 * Get recommended next consolidation time
 *
 * Returns the next occurrence of 3 AM local time.
 * This is based on research showing that:
 * - Deep sleep (when consolidation occurs) typically happens 3-4 AM
 * - System resources are usually free at this time
 * - Users are unlikely to be actively using the system
 *
 * @returns Date object representing the next recommended consolidation time
 */
export function getNextConsolidationTime(): Date {
  const now = new Date();
  const next = new Date(now);

  // Schedule for 3 AM next day
  next.setDate(next.getDate() + 1);
  next.setHours(3, 0, 0, 0);

  return next;
}

/**
 * Preview consolidation results without making changes
 *
 * Useful for understanding what would happen during consolidation
 * without actually modifying the database.
 *
 * Note: This still runs the analysis phases but skips the
 * actual modification phases.
 */
export async function previewConsolidation(
  db: VestigeDatabase,
  options: ConsolidationOptions = {}
): Promise<{
  shortTermCount: number;
  wouldPromote: number;
  potentialConnections: number;
  weakEdgeCount: number;
}> {
  const {
    shortTermWindowHours = DEFAULT_SHORT_TERM_WINDOW_HOURS,
    importanceThreshold = DEFAULT_IMPORTANCE_THRESHOLD,
    pruneThreshold = DEFAULT_PRUNE_THRESHOLD,
    maxAnalyze = DEFAULT_MAX_ANALYZE,
  } = options;

  // Get short-term memories
  const shortTermMemories = await getShortTermMemories(db, shortTermWindowHours);

  // Count how many would be promoted
  let wouldPromote = 0;
  for (const memory of shortTermMemories) {
    const importance = calculateImportance(db, memory);
    if (importance >= importanceThreshold) {
      wouldPromote++;
    }
  }

  // Preview REM cycle (dry run)
  const remPreview = await runREMCycle(db, { maxAnalyze, dryRun: true });

  // Count weak edges that would be pruned
  const internalDb = (db as unknown as { db: { prepare: (sql: string) => { get: (...args: unknown[]) => { count: number } } } }).db;
  const weakEdgeResult = internalDb
    .prepare(
      `
    SELECT COUNT(*) as count FROM graph_edges
    WHERE weight < ?
      AND json_extract(metadata, '$.discoveredBy') = 'rem_cycle'
  `
    )
    .get(pruneThreshold) as { count: number };

  return {
    shortTermCount: shortTermMemories.length,
    wouldPromote,
    potentialConnections: remPreview.connectionsDiscovered,
    weakEdgeCount: weakEdgeResult.count,
  };
}

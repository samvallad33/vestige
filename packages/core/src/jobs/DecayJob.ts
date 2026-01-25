/**
 * DecayJob - Memory Decay Processing
 *
 * Applies the Ebbinghaus forgetting curve to all knowledge nodes,
 * updating their retention strength based on time since last access.
 *
 * Designed to run as a scheduled background job (e.g., daily at 3 AM).
 *
 * @module jobs/DecayJob
 */

import type { VestigeDatabase } from '../core/database.js';
import type { Job, JobHandler } from './JobQueue.js';

// ============================================================================
// TYPES
// ============================================================================

export interface DecayJobData {
  /** Optional: Minimum retention threshold to skip already-decayed nodes */
  minRetention?: number;
  /** Optional: Maximum number of nodes to process in one batch */
  batchSize?: number;
}

export interface DecayJobResult {
  /** Number of nodes whose retention was updated */
  updatedCount: number;
  /** Total time taken in milliseconds */
  processingTime: number;
  /** Timestamp when the job ran */
  timestamp: Date;
}

// ============================================================================
// JOB HANDLER FACTORY
// ============================================================================

/**
 * Create a decay job handler
 *
 * @param db - VestigeDatabase instance
 * @returns Job handler function
 *
 * @example
 * ```typescript
 * const db = new VestigeDatabase();
 * const queue = new JobQueue();
 *
 * queue.register('decay', createDecayJobHandler(db), {
 *   concurrency: 1, // Only one decay job at a time
 *   retryDelay: 60000, // Wait 1 minute before retry
 * });
 *
 * // Schedule to run daily at 3 AM
 * queue.schedule('decay', '0 3 * * *', {});
 * ```
 */
export function createDecayJobHandler(
  db: VestigeDatabase
): JobHandler<DecayJobData, DecayJobResult> {
  return async (job: Job<DecayJobData>): Promise<DecayJobResult> => {
    const startTime = Date.now();

    // Apply decay to all nodes
    // The database method handles the Ebbinghaus curve calculation
    const updatedCount = db.applyDecay();

    const result: DecayJobResult = {
      updatedCount,
      processingTime: Date.now() - startTime,
      timestamp: new Date(),
    };

    return result;
  };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get nodes that are critically decayed (retention < threshold)
 * Useful for generating review notifications
 */
export async function getCriticallyDecayedNodes(
  db: VestigeDatabase,
  threshold: number = 0.3
): Promise<{ nodeId: string; retention: number; content: string }[]> {
  const result = db.getDecayingNodes(threshold, { limit: 50 });

  return result.items.map(node => ({
    nodeId: node.id,
    retention: node.retentionStrength,
    content: node.content.slice(0, 100),
  }));
}

/**
 * REMCycleJob - Connection Discovery Processing
 *
 * Runs the REM (Rapid Eye Movement) cycle to discover hidden connections
 * between knowledge nodes using semantic similarity, shared concepts,
 * and keyword overlap analysis.
 *
 * Designed to run as a scheduled background job (e.g., every 6 hours).
 *
 * @module jobs/REMCycleJob
 */

import type { VestigeDatabase } from '../core/database.js';
import { runREMCycle } from '../core/rem-cycle.js';
import type { Job, JobHandler } from './JobQueue.js';

// ============================================================================
// TYPES
// ============================================================================

export interface REMCycleJobData {
  /** Maximum number of nodes to analyze per cycle. Default: 50 */
  maxAnalyze?: number;
  /** Minimum connection strength threshold (0-1). Default: 0.3 */
  minStrength?: number;
  /** If true, only discover but don't create edges. Default: false */
  dryRun?: boolean;
}

export interface REMCycleJobResult {
  /** Number of nodes analyzed */
  nodesAnalyzed: number;
  /** Number of potential connections discovered */
  connectionsDiscovered: number;
  /** Number of graph edges actually created */
  connectionsCreated: number;
  /** Time taken in milliseconds */
  duration: number;
  /** Details of discovered connections */
  discoveries: Array<{
    nodeA: string;
    nodeB: string;
    reason: string;
  }>;
  /** Timestamp when the job ran */
  timestamp: Date;
}

// ============================================================================
// JOB HANDLER FACTORY
// ============================================================================

/**
 * Create a REM cycle job handler
 *
 * @param db - VestigeDatabase instance
 * @returns Job handler function
 *
 * @example
 * ```typescript
 * const db = new VestigeDatabase();
 * const queue = new JobQueue();
 *
 * queue.register('rem-cycle', createREMCycleJobHandler(db), {
 *   concurrency: 1, // Only one REM cycle at a time
 *   retryDelay: 300000, // Wait 5 minutes before retry
 * });
 *
 * // Schedule to run every 6 hours
 * queue.schedule('rem-cycle', '0 *\/6 * * *', { maxAnalyze: 100 });
 * ```
 */
export function createREMCycleJobHandler(
  db: VestigeDatabase
): JobHandler<REMCycleJobData, REMCycleJobResult> {
  return async (job: Job<REMCycleJobData>): Promise<REMCycleJobResult> => {
    const options = {
      maxAnalyze: job.data.maxAnalyze ?? 50,
      minStrength: job.data.minStrength ?? 0.3,
      dryRun: job.data.dryRun ?? false,
    };

    // Run the REM cycle (async)
    const cycleResult = await runREMCycle(db, options);

    const result: REMCycleJobResult = {
      nodesAnalyzed: cycleResult.nodesAnalyzed,
      connectionsDiscovered: cycleResult.connectionsDiscovered,
      connectionsCreated: cycleResult.connectionsCreated,
      duration: cycleResult.duration,
      discoveries: cycleResult.discoveries.map(d => ({
        nodeA: d.nodeA,
        nodeB: d.nodeB,
        reason: d.reason,
      })),
      timestamp: new Date(),
    };

    return result;
  };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Preview what connections would be discovered without creating them
 * Useful for testing or showing users potential discoveries
 */
export async function previewREMCycleJob(
  db: VestigeDatabase,
  maxAnalyze: number = 100
): Promise<REMCycleJobResult> {
  const cycleResult = await runREMCycle(db, {
    maxAnalyze,
    dryRun: true,
  });

  return {
    nodesAnalyzed: cycleResult.nodesAnalyzed,
    connectionsDiscovered: cycleResult.connectionsDiscovered,
    connectionsCreated: 0,
    duration: cycleResult.duration,
    discoveries: cycleResult.discoveries.map(d => ({
      nodeA: d.nodeA,
      nodeB: d.nodeB,
      reason: d.reason,
    })),
    timestamp: new Date(),
  };
}

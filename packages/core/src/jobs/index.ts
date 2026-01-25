/**
 * Jobs Module - Background Job Processing for Vestige MCP
 *
 * This module provides a production-ready job queue system with:
 * - Priority-based scheduling
 * - Retry logic with exponential backoff
 * - Concurrency control
 * - Cron-like recurring job scheduling
 * - Event-driven architecture
 *
 * @module jobs
 *
 * @example
 * ```typescript
 * import {
 *   JobQueue,
 *   createDecayJobHandler,
 *   createREMCycleJobHandler,
 *   createConsolidationJobHandler,
 * } from './jobs';
 * import { VestigeDatabase } from './core';
 *
 * // Initialize
 * const db = new VestigeDatabase();
 * const queue = new JobQueue();
 *
 * // Register job handlers
 * queue.register('decay', createDecayJobHandler(db), { concurrency: 1 });
 * queue.register('rem-cycle', createREMCycleJobHandler(db), { concurrency: 1 });
 * queue.register('consolidation', createConsolidationJobHandler(db), { concurrency: 1 });
 *
 * // Schedule recurring jobs
 * queue.schedule('decay', '0 3 * * *', {}); // Daily at 3 AM
 * queue.schedule('rem-cycle', '0 *\/6 * * *', {}); // Every 6 hours
 * queue.schedule('consolidation', '0 4 * * 0', {}); // Weekly on Sunday at 4 AM
 *
 * // Start processing
 * queue.start();
 *
 * // Listen to events
 * queue.on('job:completed', (job, result) => {
 *   console.log(`Job ${job.name} completed:`, result);
 * });
 *
 * queue.on('job:failed', (job, error) => {
 *   console.error(`Job ${job.name} failed:`, error);
 * });
 *
 * // Add one-off jobs
 * queue.add('rem-cycle', { maxAnalyze: 200 }, { priority: 10 });
 *
 * // Graceful shutdown
 * process.on('SIGTERM', async () => {
 *   await queue.shutdown();
 *   db.close();
 * });
 * ```
 */

// Core job queue
export {
  JobQueue,
  getDefaultQueue,
  resetDefaultQueue,
  type Job,
  type JobResult,
  type JobHandler,
  type JobOptions,
  type JobDefinition,
  type JobStatus,
  type ScheduledJob,
  type QueueStats,
  type JobQueueEvents,
} from './JobQueue.js';

// Decay job
export {
  createDecayJobHandler,
  getCriticallyDecayedNodes,
  type DecayJobData,
  type DecayJobResult,
} from './DecayJob.js';

// REM cycle job
export {
  createREMCycleJobHandler,
  previewREMCycleJob,
  type REMCycleJobData,
  type REMCycleJobResult,
} from './REMCycleJob.js';

// Consolidation job
export {
  createConsolidationJobHandler,
  previewConsolidation,
  getConsolidationMetrics,
  type ConsolidationJobData,
  type ConsolidationJobResult,
} from './ConsolidationJob.js';

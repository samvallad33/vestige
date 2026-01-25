/**
 * JobQueue - Background Job Processing for Vestige MCP
 *
 * A production-ready in-memory job queue with:
 * - Priority-based job scheduling
 * - Retry logic with exponential backoff
 * - Concurrency control per job type
 * - Event-driven architecture
 * - Cron-like scheduling support
 *
 * @module jobs/JobQueue
 */

import { EventEmitter } from 'events';
import { nanoid } from 'nanoid';

// ============================================================================
// TYPES
// ============================================================================

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface Job<T = unknown> {
  id: string;
  name: string;
  data: T;
  priority: number;
  createdAt: Date;
  scheduledAt?: Date;
  startedAt?: Date;
  completedAt?: Date;
  retryCount: number;
  maxRetries: number;
  status: JobStatus;
  error?: string;
}

export interface JobResult<R = unknown> {
  jobId: string;
  success: boolean;
  result?: R;
  error?: Error;
  duration: number;
}

export type JobHandler<T, R> = (job: Job<T>) => Promise<R>;

export interface JobOptions {
  /** Priority (higher = processed first). Default: 0 */
  priority?: number;
  /** Delay in milliseconds before job becomes eligible. Default: 0 */
  delay?: number;
  /** Maximum retry attempts on failure. Default: 3 */
  maxRetries?: number;
}

export interface JobDefinition<T = unknown, R = unknown> {
  name: string;
  handler: JobHandler<T, R>;
  concurrency: number;
  retryDelay: number;
}

export interface ScheduledJob {
  name: string;
  cronExpression: string;
  data: unknown;
  lastRun?: Date;
  nextRun?: Date;
}

export interface QueueStats {
  pending: number;
  running: number;
  completed: number;
  failed: number;
  total: number;
}

// ============================================================================
// JOB QUEUE EVENTS
// ============================================================================

export interface JobQueueEvents {
  'job:added': (job: Job) => void;
  'job:started': (job: Job) => void;
  'job:completed': (job: Job, result: JobResult) => void;
  'job:failed': (job: Job, error: Error) => void;
  'job:retry': (job: Job, attempt: number, error: Error) => void;
  'queue:drained': () => void;
  'queue:error': (error: Error) => void;
}

// ============================================================================
// CRON PARSER (Simple Implementation)
// ============================================================================

interface CronFields {
  minute: number[];
  hour: number[];
  dayOfMonth: number[];
  month: number[];
  dayOfWeek: number[];
}

/**
 * Parse a simple cron expression
 * Format: minute hour day-of-month month day-of-week
 * Supports: numbers, *, /step, ranges (-)
 */
function parseCronField(field: string, min: number, max: number): number[] {
  const values: number[] = [];

  // Handle wildcard
  if (field === '*') {
    for (let i = min; i <= max; i++) {
      values.push(i);
    }
    return values;
  }

  // Handle step values (*/n or n/m)
  if (field.includes('/')) {
    const [range, stepStr] = field.split('/');
    const step = parseInt(stepStr || '1', 10);
    let start = min;
    let end = max;

    if (range && range !== '*') {
      if (range.includes('-')) {
        const [s, e] = range.split('-');
        start = parseInt(s || String(min), 10);
        end = parseInt(e || String(max), 10);
      } else {
        start = parseInt(range, 10);
      }
    }

    for (let i = start; i <= end; i += step) {
      values.push(i);
    }
    return values;
  }

  // Handle ranges (n-m)
  if (field.includes('-')) {
    const [start, end] = field.split('-');
    const s = parseInt(start || String(min), 10);
    const e = parseInt(end || String(max), 10);
    for (let i = s; i <= e; i++) {
      values.push(i);
    }
    return values;
  }

  // Handle comma-separated values
  if (field.includes(',')) {
    return field.split(',').map(v => parseInt(v.trim(), 10));
  }

  // Single value
  values.push(parseInt(field, 10));
  return values;
}

function parseCronExpression(expression: string): CronFields {
  const parts = expression.trim().split(/\s+/);

  if (parts.length !== 5) {
    throw new Error(`Invalid cron expression: ${expression}. Expected 5 fields.`);
  }

  return {
    minute: parseCronField(parts[0] || '*', 0, 59),
    hour: parseCronField(parts[1] || '*', 0, 23),
    dayOfMonth: parseCronField(parts[2] || '*', 1, 31),
    month: parseCronField(parts[3] || '*', 1, 12),
    dayOfWeek: parseCronField(parts[4] || '*', 0, 6),
  };
}

function getNextCronDate(expression: string, after: Date = new Date()): Date {
  const fields = parseCronExpression(expression);
  const next = new Date(after);
  next.setSeconds(0);
  next.setMilliseconds(0);

  // Start from next minute
  next.setMinutes(next.getMinutes() + 1);

  // Find next matching time (limit iterations to prevent infinite loops)
  for (let iterations = 0; iterations < 525600; iterations++) { // Max 1 year of minutes
    const minute = next.getMinutes();
    const hour = next.getHours();
    const dayOfMonth = next.getDate();
    const month = next.getMonth() + 1; // JS months are 0-indexed
    const dayOfWeek = next.getDay();

    // Check if current time matches cron expression
    if (
      fields.minute.includes(minute) &&
      fields.hour.includes(hour) &&
      fields.dayOfMonth.includes(dayOfMonth) &&
      fields.month.includes(month) &&
      fields.dayOfWeek.includes(dayOfWeek)
    ) {
      return next;
    }

    // Advance by one minute
    next.setMinutes(next.getMinutes() + 1);
  }

  throw new Error(`Could not find next cron date within 1 year for: ${expression}`);
}

// ============================================================================
// JOB QUEUE IMPLEMENTATION
// ============================================================================

export class JobQueue extends EventEmitter {
  private jobs: Map<string, Job> = new Map();
  private handlers: Map<string, JobDefinition> = new Map();
  private running: Map<string, number> = new Map();
  private interval: NodeJS.Timeout | null = null;
  private scheduledJobs: Map<string, ScheduledJob> = new Map();
  private schedulerInterval: NodeJS.Timeout | null = null;
  private isProcessing = false;
  private isPaused = false;

  // Completed/failed job history (limited size)
  private readonly maxHistorySize = 1000;
  private completedJobIds: Set<string> = new Set();
  private failedJobIds: Set<string> = new Set();

  constructor() {
    super();
    this.setMaxListeners(100);
  }

  // ============================================================================
  // HANDLER REGISTRATION
  // ============================================================================

  /**
   * Register a job handler
   *
   * @param name - Unique job type name
   * @param handler - Async function to process the job
   * @param options - Handler options (concurrency, retryDelay)
   *
   * @example
   * ```typescript
   * queue.register('send-email', async (job) => {
   *   await sendEmail(job.data);
   *   return { sent: true };
   * }, { concurrency: 5, retryDelay: 5000 });
   * ```
   */
  register<T, R>(
    name: string,
    handler: JobHandler<T, R>,
    options?: { concurrency?: number; retryDelay?: number }
  ): void {
    if (this.handlers.has(name)) {
      throw new Error(`Handler already registered for job type: ${name}`);
    }

    // Store as JobDefinition<unknown, unknown> since we type-erase at runtime
    // The type safety is maintained at the call site (add/register)
    const definition: JobDefinition = {
      name,
      handler: handler as unknown as JobHandler<unknown, unknown>,
      concurrency: options?.concurrency ?? 1,
      retryDelay: options?.retryDelay ?? 1000,
    };

    this.handlers.set(name, definition);
    this.running.set(name, 0);
  }

  /**
   * Unregister a job handler
   */
  unregister(name: string): boolean {
    const deleted = this.handlers.delete(name);
    this.running.delete(name);
    return deleted;
  }

  // ============================================================================
  // JOB MANAGEMENT
  // ============================================================================

  /**
   * Add a job to the queue
   *
   * @param name - Job type name (must have registered handler)
   * @param data - Job data payload
   * @param options - Job options (priority, delay, maxRetries)
   * @returns Job ID
   *
   * @example
   * ```typescript
   * const jobId = queue.add('send-email', {
   *   to: 'user@example.com',
   *   subject: 'Hello'
   * }, { priority: 10, maxRetries: 5 });
   * ```
   */
  add<T>(
    name: string,
    data: T,
    options?: JobOptions
  ): string {
    if (!this.handlers.has(name)) {
      throw new Error(`No handler registered for job type: ${name}`);
    }

    const id = nanoid();
    const now = new Date();

    let scheduledAt: Date | undefined;
    if (options?.delay && options.delay > 0) {
      scheduledAt = new Date(now.getTime() + options.delay);
    }

    const job: Job<T> = {
      id,
      name,
      data,
      priority: options?.priority ?? 0,
      createdAt: now,
      scheduledAt,
      retryCount: 0,
      maxRetries: options?.maxRetries ?? 3,
      status: 'pending',
    };

    this.jobs.set(id, job as Job);
    this.emit('job:added', job);

    // Trigger processing if running
    if (this.isProcessing && !this.isPaused) {
      this.processNextJobs();
    }

    return id;
  }

  /**
   * Get a job by ID
   */
  getJob(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  /**
   * Get all jobs matching a filter
   */
  getJobs(filter?: { name?: string; status?: JobStatus }): Job[] {
    let jobs = Array.from(this.jobs.values());

    if (filter?.name) {
      jobs = jobs.filter(j => j.name === filter.name);
    }

    if (filter?.status) {
      jobs = jobs.filter(j => j.status === filter.status);
    }

    return jobs;
  }

  /**
   * Remove a job from the queue
   * Can only remove pending jobs
   */
  removeJob(id: string): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    if (job.status === 'running') {
      throw new Error('Cannot remove a running job');
    }

    return this.jobs.delete(id);
  }

  /**
   * Clear all completed/failed jobs from history
   */
  clearHistory(): void {
    for (const id of this.completedJobIds) {
      this.jobs.delete(id);
    }
    for (const id of this.failedJobIds) {
      this.jobs.delete(id);
    }
    this.completedJobIds.clear();
    this.failedJobIds.clear();
  }

  // ============================================================================
  // QUEUE STATISTICS
  // ============================================================================

  /**
   * Get queue statistics
   */
  getStats(): QueueStats {
    let pending = 0;
    let running = 0;
    let completed = 0;
    let failed = 0;

    for (const job of this.jobs.values()) {
      switch (job.status) {
        case 'pending':
          pending++;
          break;
        case 'running':
          running++;
          break;
        case 'completed':
          completed++;
          break;
        case 'failed':
          failed++;
          break;
      }
    }

    return {
      pending,
      running,
      completed,
      failed,
      total: this.jobs.size,
    };
  }

  /**
   * Check if queue is empty (no pending or running jobs)
   */
  isEmpty(): boolean {
    for (const job of this.jobs.values()) {
      if (job.status === 'pending' || job.status === 'running') {
        return false;
      }
    }
    return true;
  }

  // ============================================================================
  // PROCESSING
  // ============================================================================

  /**
   * Start processing jobs
   *
   * @param pollInterval - How often to check for new jobs (ms). Default: 100
   */
  start(pollInterval: number = 100): void {
    if (this.isProcessing) {
      return;
    }

    this.isProcessing = true;
    this.isPaused = false;

    this.interval = setInterval(() => {
      if (!this.isPaused) {
        this.processNextJobs();
      }
    }, pollInterval);

    // Start scheduler for cron jobs
    this.startScheduler();

    // Process immediately
    this.processNextJobs();
  }

  /**
   * Stop processing jobs
   */
  stop(): void {
    this.isProcessing = false;

    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }

    this.stopScheduler();
  }

  /**
   * Pause processing (jobs stay in queue)
   */
  pause(): void {
    this.isPaused = true;
  }

  /**
   * Resume processing
   */
  resume(): void {
    this.isPaused = false;
    this.processNextJobs();
  }

  /**
   * Wait for all pending jobs to complete
   */
  async drain(): Promise<void> {
    return new Promise((resolve) => {
      const check = () => {
        if (this.isEmpty()) {
          resolve();
        } else {
          setTimeout(check, 50);
        }
      };
      check();
    });
  }

  /**
   * Process next eligible jobs
   */
  private processNextJobs(): void {
    const now = new Date();

    // Get pending jobs sorted by priority (descending)
    const pendingJobs = Array.from(this.jobs.values())
      .filter(job => {
        if (job.status !== 'pending') return false;
        if (job.scheduledAt && job.scheduledAt > now) return false;
        return true;
      })
      .sort((a, b) => b.priority - a.priority);

    // Process jobs respecting concurrency limits
    for (const job of pendingJobs) {
      const definition = this.handlers.get(job.name);
      if (!definition) continue;

      const currentRunning = this.running.get(job.name) ?? 0;
      if (currentRunning >= definition.concurrency) continue;

      // Start processing this job
      this.processJob(job, definition);
    }
  }

  /**
   * Process a single job
   */
  private async processJob(job: Job, definition: JobDefinition): Promise<void> {
    // Update job status
    job.status = 'running';
    job.startedAt = new Date();

    // Track running count
    const currentRunning = this.running.get(job.name) ?? 0;
    this.running.set(job.name, currentRunning + 1);

    this.emit('job:started', job);

    const startTime = Date.now();

    try {
      const result = await definition.handler(job);

      // Job completed successfully
      job.status = 'completed';
      job.completedAt = new Date();

      const jobResult: JobResult = {
        jobId: job.id,
        success: true,
        result,
        duration: Date.now() - startTime,
      };

      this.emit('job:completed', job, jobResult);

      // Track in history
      this.addToHistory(job.id, 'completed');

    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));

      // Check if we should retry
      if (job.retryCount < job.maxRetries) {
        job.retryCount++;
        job.status = 'pending';

        // Schedule retry with exponential backoff
        const backoffDelay = definition.retryDelay * Math.pow(2, job.retryCount - 1);
        job.scheduledAt = new Date(Date.now() + backoffDelay);

        this.emit('job:retry', job, job.retryCount, err);

      } else {
        // Max retries exceeded - mark as failed
        job.status = 'failed';
        job.completedAt = new Date();
        job.error = err.message;

        this.emit('job:failed', job, err);

        // Track in history
        this.addToHistory(job.id, 'failed');
      }

    } finally {
      // Update running count
      const runningCount = this.running.get(job.name) ?? 1;
      this.running.set(job.name, Math.max(0, runningCount - 1));

      // Check if queue is drained
      if (this.isEmpty()) {
        this.emit('queue:drained');
      }
    }
  }

  /**
   * Add job to history tracking (with size limit)
   */
  private addToHistory(jobId: string, type: 'completed' | 'failed'): void {
    const targetSet = type === 'completed' ? this.completedJobIds : this.failedJobIds;
    targetSet.add(jobId);

    // Trim history if too large
    if (targetSet.size > this.maxHistorySize) {
      const iterator = targetSet.values();
      const firstValue = iterator.next().value;
      if (firstValue) {
        targetSet.delete(firstValue);
        this.jobs.delete(firstValue);
      }
    }
  }

  // ============================================================================
  // SCHEDULING (CRON-LIKE)
  // ============================================================================

  /**
   * Schedule a recurring job
   *
   * @param name - Job type name
   * @param cronExpression - Cron expression (minute hour day-of-month month day-of-week)
   * @param data - Job data payload
   *
   * @example
   * ```typescript
   * // Run decay at 3 AM daily
   * queue.schedule('decay', '0 3 * * *', {});
   *
   * // Run REM cycle every 6 hours
   * queue.schedule('rem-cycle', '0 *\\/6 * * *', {});
   * ```
   */
  schedule<T>(name: string, cronExpression: string, data: T): void {
    if (!this.handlers.has(name)) {
      throw new Error(`No handler registered for job type: ${name}`);
    }

    // Validate cron expression by parsing it
    try {
      parseCronExpression(cronExpression);
    } catch (error) {
      throw new Error(`Invalid cron expression for ${name}: ${cronExpression}`);
    }

    const scheduledJob: ScheduledJob = {
      name,
      cronExpression,
      data,
      nextRun: getNextCronDate(cronExpression),
    };

    this.scheduledJobs.set(name, scheduledJob);
  }

  /**
   * Remove a scheduled job
   */
  unschedule(name: string): boolean {
    return this.scheduledJobs.delete(name);
  }

  /**
   * Get all scheduled jobs
   */
  getScheduledJobs(): ScheduledJob[] {
    return Array.from(this.scheduledJobs.values());
  }

  /**
   * Start the scheduler
   */
  private startScheduler(): void {
    if (this.schedulerInterval) return;

    // Check every minute for scheduled jobs
    this.schedulerInterval = setInterval(() => {
      this.checkScheduledJobs();
    }, 60000);

    // Also check immediately
    this.checkScheduledJobs();
  }

  /**
   * Stop the scheduler
   */
  private stopScheduler(): void {
    if (this.schedulerInterval) {
      clearInterval(this.schedulerInterval);
      this.schedulerInterval = null;
    }
  }

  /**
   * Check and trigger scheduled jobs
   */
  private checkScheduledJobs(): void {
    const now = new Date();

    for (const [name, scheduled] of this.scheduledJobs) {
      if (scheduled.nextRun && scheduled.nextRun <= now) {
        try {
          // Add the job
          this.add(name, scheduled.data);

          // Update last run and calculate next run
          scheduled.lastRun = now;
          scheduled.nextRun = getNextCronDate(scheduled.cronExpression, now);
        } catch (error) {
          const err = error instanceof Error ? error : new Error(String(error));
          this.emit('queue:error', err);
        }
      }
    }
  }

  // ============================================================================
  // CLEANUP
  // ============================================================================

  /**
   * Graceful shutdown
   */
  async shutdown(timeout: number = 30000): Promise<void> {
    this.stop();
    this.isPaused = true;

    // Wait for running jobs to complete (with timeout)
    const waitStart = Date.now();

    while (Date.now() - waitStart < timeout) {
      const stats = this.getStats();
      if (stats.running === 0) {
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Clear all jobs
    this.jobs.clear();
    this.completedJobIds.clear();
    this.failedJobIds.clear();
    this.scheduledJobs.clear();

    this.removeAllListeners();
  }
}

// ============================================================================
// SINGLETON INSTANCE (Optional)
// ============================================================================

let defaultQueue: JobQueue | null = null;

/**
 * Get the default job queue instance
 */
export function getDefaultQueue(): JobQueue {
  if (!defaultQueue) {
    defaultQueue = new JobQueue();
  }
  return defaultQueue;
}

/**
 * Reset the default queue (for testing)
 */
export function resetDefaultQueue(): void {
  if (defaultQueue) {
    defaultQueue.shutdown().catch(() => {});
    defaultQueue = null;
  }
}

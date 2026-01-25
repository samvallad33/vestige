/**
 * Concurrency utilities for Vestige MCP
 *
 * Provides synchronization primitives for managing concurrent access
 * to shared resources like database connections.
 */

/**
 * Error thrown when an operation times out
 */
export class TimeoutError extends Error {
  constructor(message = "Operation timed out") {
    super(message);
    this.name = "TimeoutError";
  }
}

/**
 * Reader-Writer Lock for concurrent database access.
 * Allows multiple concurrent readers OR one exclusive writer.
 *
 * This implementation uses writer preference with reader batching
 * to prevent writer starvation while still allowing good read throughput.
 */
export class RWLock {
  private readers = 0;
  private writer = false;
  private writerQueue: (() => void)[] = [];
  private readerQueue: (() => void)[] = [];

  /**
   * Execute a function with read lock (allows concurrent readers)
   */
  async withReadLock<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquireRead();
    try {
      return await fn();
    } finally {
      this.releaseRead();
    }
  }

  /**
   * Execute a function with write lock (exclusive access)
   */
  async withWriteLock<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquireWrite();
    try {
      return await fn();
    } finally {
      this.releaseWrite();
    }
  }

  private acquireRead(): Promise<void> {
    return new Promise<void>((resolve) => {
      // If no writer and no writers waiting, grant immediately
      if (!this.writer && this.writerQueue.length === 0) {
        this.readers++;
        resolve();
      } else {
        // Queue the reader
        this.readerQueue.push(() => {
          this.readers++;
          resolve();
        });
      }
    });
  }

  private releaseRead(): void {
    this.readers--;

    // If no more readers, wake up waiting writer
    if (this.readers === 0 && this.writerQueue.length > 0) {
      const nextWriter = this.writerQueue.shift();
      if (nextWriter) {
        this.writer = true;
        nextWriter();
      }
    }
  }

  private acquireWrite(): Promise<void> {
    return new Promise<void>((resolve) => {
      // If no readers and no writer, grant immediately
      if (this.readers === 0 && !this.writer) {
        this.writer = true;
        resolve();
      } else {
        // Queue the writer
        this.writerQueue.push(resolve);
      }
    });
  }

  private releaseWrite(): void {
    this.writer = false;

    // Prefer waking readers over writers to prevent starvation
    // Wake all waiting readers as a batch
    if (this.readerQueue.length > 0) {
      const readers = this.readerQueue.splice(0, this.readerQueue.length);
      for (const reader of readers) {
        reader();
      }
    } else if (this.writerQueue.length > 0) {
      // No waiting readers, wake next writer
      const nextWriter = this.writerQueue.shift();
      if (nextWriter) {
        this.writer = true;
        nextWriter();
      }
    }
  }

  /**
   * Get current lock state (for debugging/monitoring)
   */
  getState(): { readers: number; hasWriter: boolean; pendingReaders: number; pendingWriters: number } {
    return {
      readers: this.readers,
      hasWriter: this.writer,
      pendingReaders: this.readerQueue.length,
      pendingWriters: this.writerQueue.length,
    };
  }
}

/**
 * Simple mutex for exclusive access
 */
export class Mutex {
  private locked = false;
  private queue: (() => void)[] = [];

  /**
   * Execute a function with exclusive lock
   */
  async withLock<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    return new Promise<void>((resolve) => {
      if (!this.locked) {
        this.locked = true;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  private release(): void {
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      if (next) {
        next();
      }
    } else {
      this.locked = false;
    }
  }

  /**
   * Check if the mutex is currently locked
   */
  isLocked(): boolean {
    return this.locked;
  }

  /**
   * Get the number of waiters in the queue
   */
  getQueueLength(): number {
    return this.queue.length;
  }
}

/**
 * Semaphore for limiting concurrent operations
 */
export class Semaphore {
  private permits: number;
  private available: number;
  private queue: (() => void)[] = [];

  constructor(permits: number) {
    if (permits < 1) {
      throw new Error("Semaphore must have at least 1 permit");
    }
    this.permits = permits;
    this.available = permits;
  }

  /**
   * Execute a function with a permit from the semaphore
   */
  async withPermit<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }

  /**
   * Execute multiple functions concurrently, respecting the semaphore limit
   */
  async map<T, R>(items: T[], fn: (item: T) => Promise<R>): Promise<R[]> {
    return Promise.all(items.map((item) => this.withPermit(() => fn(item))));
  }

  private acquire(): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.available > 0) {
        this.available--;
        resolve();
      } else {
        this.queue.push(resolve);
      }
    });
  }

  private release(): void {
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      if (next) {
        next();
      }
    } else {
      this.available++;
    }
  }

  /**
   * Get the number of available permits
   */
  getAvailable(): number {
    return this.available;
  }

  /**
   * Get the total number of permits
   */
  getTotal(): number {
    return this.permits;
  }

  /**
   * Get the number of waiters in the queue
   */
  getQueueLength(): number {
    return this.queue.length;
  }
}

/**
 * Add timeout to any promise
 *
 * @param promise - The promise to wrap with a timeout
 * @param ms - Timeout in milliseconds
 * @param message - Optional custom error message
 * @returns The result of the promise if it completes in time
 * @throws TimeoutError if the timeout is exceeded
 */
export function withTimeout<T>(promise: Promise<T>, ms: number, message?: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new TimeoutError(message ?? `Operation timed out after ${ms}ms`));
    }, ms);

    promise
      .then((result) => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

/**
 * Options for retry with exponential backoff
 */
export interface RetryOptions {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Initial delay in milliseconds (default: 100) */
  initialDelay?: number;
  /** Maximum delay in milliseconds (default: 5000) */
  maxDelay?: number;
  /** Backoff multiplier (default: 2) */
  backoffFactor?: number;
  /** Optional function to determine if an error is retryable */
  isRetryable?: (error: unknown) => boolean;
  /** Optional callback called before each retry */
  onRetry?: (error: unknown, attempt: number, delay: number) => void;
}

/**
 * Retry function with exponential backoff
 *
 * @param fn - The async function to retry
 * @param options - Retry configuration options
 * @returns The result of the function if it succeeds
 * @throws The last error if all retries are exhausted
 */
export async function retry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 100,
    maxDelay = 5000,
    backoffFactor = 2,
    isRetryable = () => true,
    onRetry,
  } = options;

  let lastError: unknown;
  let delay = initialDelay;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Check if we've exhausted retries
      if (attempt >= maxRetries) {
        throw error;
      }

      // Check if the error is retryable
      if (!isRetryable(error)) {
        throw error;
      }

      // Calculate delay with jitter (0.5 to 1.5 of calculated delay)
      const jitter = 0.5 + Math.random();
      const actualDelay = Math.min(delay * jitter, maxDelay);

      // Call onRetry callback if provided
      if (onRetry) {
        onRetry(error, attempt + 1, actualDelay);
      }

      // Wait before next attempt
      await sleep(actualDelay);

      // Increase delay for next attempt
      delay = Math.min(delay * backoffFactor, maxDelay);
    }
  }

  // This should never be reached, but TypeScript needs it
  throw lastError;
}

/**
 * Sleep for a specified duration
 *
 * @param ms - Duration in milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Debounce a function - only execute after the specified delay
 * has passed without another call
 *
 * @param fn - The function to debounce
 * @param delay - Delay in milliseconds
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn(...args);
      timeoutId = null;
    }, delay);
  };
}

/**
 * Throttle a function - execute at most once per specified interval
 *
 * @param fn - The function to throttle
 * @param interval - Minimum interval between executions in milliseconds
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  interval: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;

    if (timeSinceLastCall >= interval) {
      lastCall = now;
      fn(...args);
    } else if (!timeoutId) {
      timeoutId = setTimeout(
        () => {
          lastCall = Date.now();
          fn(...args);
          timeoutId = null;
        },
        interval - timeSinceLastCall
      );
    }
  };
}

/**
 * Create a deferred promise that can be resolved/rejected externally
 */
export function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
}

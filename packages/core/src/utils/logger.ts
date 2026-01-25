/**
 * Centralized logging system for Vestige MCP
 *
 * Provides structured JSON logging with:
 * - Log levels (debug, info, warn, error)
 * - Child loggers for subsystems
 * - Request context tracking via AsyncLocalStorage
 * - Performance logging utilities
 */

import { AsyncLocalStorage } from 'async_hooks';
import { nanoid } from 'nanoid';

// ============================================================================
// Types
// ============================================================================

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  logger: string;
  message: string;
  context?: Record<string, unknown>;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
}

export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, error?: Error, context?: Record<string, unknown>): void;
  child(name: string): Logger;
}

// ============================================================================
// Request Context (AsyncLocalStorage)
// ============================================================================

interface RequestContext {
  requestId: string;
  startTime: number;
}

export const requestContext = new AsyncLocalStorage<RequestContext>();

/**
 * Run a function within a request context for tracing
 */
export function withRequestContext<T>(fn: () => T): T {
  const ctx: RequestContext = {
    requestId: nanoid(8),
    startTime: Date.now(),
  };
  return requestContext.run(ctx, fn);
}

/**
 * Run an async function within a request context for tracing
 */
export function withRequestContextAsync<T>(fn: () => Promise<T>): Promise<T> {
  const ctx: RequestContext = {
    requestId: nanoid(8),
    startTime: Date.now(),
  };
  return requestContext.run(ctx, fn);
}

/**
 * Enrich context with request tracing information if available
 */
function enrichContext(context?: Record<string, unknown>): Record<string, unknown> {
  const ctx = requestContext.getStore();
  if (ctx) {
    return {
      ...context,
      requestId: ctx.requestId,
      elapsed: Date.now() - ctx.startTime,
    };
  }
  return context || {};
}

// ============================================================================
// Logger Implementation
// ============================================================================

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Create a structured JSON logger
 *
 * @param name - Logger name (used as prefix for child loggers)
 * @param minLevel - Minimum log level to output (default: 'info')
 * @returns Logger instance
 */
export function createLogger(name: string, minLevel: LogLevel = 'info'): Logger {
  const minLevelValue = LOG_LEVELS[minLevel];

  function log(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    error?: Error
  ): void {
    if (LOG_LEVELS[level] < minLevelValue) return;

    const enrichedContext = enrichContext(context);

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      logger: name,
      message,
    };

    // Only include context if it has properties
    if (Object.keys(enrichedContext).length > 0) {
      entry.context = enrichedContext;
    }

    if (error) {
      entry.error = {
        name: error.name,
        message: error.message,
        ...(error.stack !== undefined && { stack: error.stack }),
      };
    }

    const output = JSON.stringify(entry);

    if (level === 'error') {
      console.error(output);
    } else {
      console.log(output);
    }
  }

  return {
    debug: (message, context) => log('debug', message, context),
    info: (message, context) => log('info', message, context),
    warn: (message, context) => log('warn', message, context),
    error: (message, error, context) => log('error', message, context, error),
    child: (childName) => createLogger(`${name}:${childName}`, minLevel),
  };
}

// ============================================================================
// Global Logger Instances
// ============================================================================

// Get log level from environment
function getLogLevelFromEnv(): LogLevel {
  const envLevel = process.env['VESTIGE_LOG_LEVEL']?.toLowerCase();
  if (envLevel && envLevel in LOG_LEVELS) {
    return envLevel as LogLevel;
  }
  return 'info';
}

const LOG_LEVEL = getLogLevelFromEnv();

// Root logger
export const logger = createLogger('vestige', LOG_LEVEL);

// Pre-configured child loggers for subsystems
export const dbLogger = logger.child('database');
export const mcpLogger = logger.child('mcp');
export const remLogger = logger.child('rem-cycle');
export const embeddingLogger = logger.child('embeddings');
export const cacheLogger = logger.child('cache');
export const jobLogger = logger.child('jobs');

// ============================================================================
// Performance Logging
// ============================================================================

/**
 * Wrap a function to log its execution time
 *
 * @param logger - Logger instance to use
 * @param operationName - Name of the operation for logging
 * @param fn - Async function to wrap
 * @returns Wrapped function that logs performance
 *
 * @example
 * const wrappedFetch = logPerformance(dbLogger, 'fetchNodes', fetchNodes);
 * const nodes = await wrappedFetch(query);
 */
export function logPerformance<T extends (...args: unknown[]) => Promise<unknown>>(
  logger: Logger,
  operationName: string,
  fn: T
): T {
  return (async (...args: Parameters<T>) => {
    const start = Date.now();
    try {
      const result = await fn(...args);
      logger.info(`${operationName} completed`, {
        duration: Date.now() - start,
      });
      return result;
    } catch (error) {
      logger.error(`${operationName} failed`, error as Error, {
        duration: Date.now() - start,
      });
      throw error;
    }
  }) as T;
}

/**
 * Log performance of a single async operation
 *
 * @param logger - Logger instance to use
 * @param operationName - Name of the operation for logging
 * @param fn - Async function to execute and measure
 * @returns Result of the function
 *
 * @example
 * const result = await timedOperation(dbLogger, 'query', async () => {
 *   return await db.query(sql);
 * });
 */
export async function timedOperation<T>(
  logger: Logger,
  operationName: string,
  fn: () => Promise<T>
): Promise<T> {
  const start = Date.now();
  try {
    const result = await fn();
    logger.info(`${operationName} completed`, {
      duration: Date.now() - start,
    });
    return result;
  } catch (error) {
    logger.error(`${operationName} failed`, error as Error, {
      duration: Date.now() - start,
    });
    throw error;
  }
}

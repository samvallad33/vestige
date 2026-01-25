/**
 * Vestige Error Types
 *
 * A comprehensive hierarchy of errors for proper error handling and reporting.
 * Includes type guards, utilities, and a Result type for functional error handling.
 */

// =============================================================================
// Error Sanitization
// =============================================================================

/**
 * Sanitize error messages to prevent information leakage
 */
export function sanitizeErrorMessage(message: string): string {
  let sanitized = message;
  // Remove file paths
  sanitized = sanitized.replace(/\/[^\s]+/g, '[PATH]');
  // Remove SQL keywords
  sanitized = sanitized.replace(/SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER/gi, '[SQL]');
  // Redact credentials
  sanitized = sanitized.replace(
    /\b(password|secret|key|token|auth)\s*[=:]\s*\S+/gi,
    '[REDACTED]'
  );
  return sanitized;
}

// =============================================================================
// Base Error Class
// =============================================================================

/**
 * Base error class for all Vestige errors
 */
export class VestigeError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 500,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'VestigeError';
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON(): {
    name: string;
    code: string;
    message: string;
    statusCode: number;
    details?: Record<string, unknown>;
  } {
    const result: {
      name: string;
      code: string;
      message: string;
      statusCode: number;
      details?: Record<string, unknown>;
    } = {
      name: this.name,
      code: this.code,
      message: this.message,
      statusCode: this.statusCode,
    };
    if (this.details !== undefined) {
      result.details = this.details;
    }
    return result;
  }
}

// =============================================================================
// Specific Error Types
// =============================================================================

/**
 * Validation errors (400)
 */
export class ValidationError extends VestigeError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', 400, details);
    this.name = 'ValidationError';
  }
}

/**
 * Resource not found (404)
 */
export class NotFoundError extends VestigeError {
  constructor(resource: string, id?: string) {
    super(
      id ? `${resource} not found: ${id}` : `${resource} not found`,
      'NOT_FOUND',
      404,
      { resource, id }
    );
    this.name = 'NotFoundError';
  }
}

/**
 * Conflict errors (409)
 */
export class ConflictError extends VestigeError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'CONFLICT', 409, details);
    this.name = 'ConflictError';
  }
}

/**
 * Database operation errors (500)
 */
export class DatabaseError extends VestigeError {
  constructor(message: string, cause?: unknown) {
    super(sanitizeErrorMessage(message), 'DATABASE_ERROR', 500, {
      cause: String(cause),
    });
    this.name = 'DatabaseError';
  }
}

/**
 * Security-related errors (403)
 */
export class SecurityError extends VestigeError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'SECURITY_ERROR', 403, details);
    this.name = 'SecurityError';
  }
}

/**
 * Configuration errors (500)
 */
export class ConfigurationError extends VestigeError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, 'CONFIGURATION_ERROR', 500, details);
    this.name = 'ConfigurationError';
  }
}

/**
 * Timeout errors (408)
 */
export class TimeoutError extends VestigeError {
  constructor(operation: string, timeoutMs: number) {
    super(`Operation timed out: ${operation}`, 'TIMEOUT', 408, {
      operation,
      timeoutMs,
    });
    this.name = 'TimeoutError';
  }
}

/**
 * Embedding service errors
 */
export class EmbeddingError extends VestigeError {
  constructor(message: string, cause?: unknown) {
    super(message, 'EMBEDDING_ERROR', 500, { cause: String(cause) });
    this.name = 'EmbeddingError';
  }
}

/**
 * Concurrency/locking errors (409)
 */
export class ConcurrencyError extends VestigeError {
  constructor(message: string = 'Operation failed due to concurrent access') {
    super(message, 'CONCURRENCY_ERROR', 409);
    this.name = 'ConcurrencyError';
  }
}

/**
 * Rate limit errors (429)
 */
export class RateLimitError extends VestigeError {
  constructor(message: string, retryAfterMs?: number) {
    super(message, 'RATE_LIMIT', 429, { retryAfterMs });
    this.name = 'RateLimitError';
  }
}

/**
 * Authentication errors (401)
 */
export class AuthenticationError extends VestigeError {
  constructor(message: string = 'Authentication required') {
    super(message, 'AUTHENTICATION_ERROR', 401);
    this.name = 'AuthenticationError';
  }
}

// =============================================================================
// Error Handling Utilities
// =============================================================================

/**
 * Type guard for VestigeError
 */
export function isVestigeError(error: unknown): error is VestigeError {
  return error instanceof VestigeError;
}

/**
 * Convert unknown error to VestigeError
 */
export function toVestigeError(error: unknown): VestigeError {
  if (isVestigeError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new VestigeError(
      sanitizeErrorMessage(error.message),
      'UNKNOWN_ERROR',
      500,
      { originalName: error.name }
    );
  }

  if (typeof error === 'string') {
    return new VestigeError(sanitizeErrorMessage(error), 'UNKNOWN_ERROR', 500);
  }

  return new VestigeError('An unknown error occurred', 'UNKNOWN_ERROR', 500, {
    errorType: typeof error,
  });
}

/**
 * Wrap function to catch and transform errors
 */
export function wrapError<T extends (...args: unknown[]) => Promise<unknown>>(
  fn: T,
  errorTransform?: (error: unknown) => VestigeError
): T {
  const wrapped = async (...args: Parameters<T>): Promise<ReturnType<T>> => {
    try {
      return (await fn(...args)) as ReturnType<T>;
    } catch (error) {
      if (errorTransform) {
        throw errorTransform(error);
      }
      throw toVestigeError(error);
    }
  };
  return wrapped as T;
}

/**
 * Execute a function with error transformation
 */
export async function withErrorHandling<T>(
  fn: () => Promise<T>,
  errorTransform?: (error: unknown) => VestigeError
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (errorTransform) {
      throw errorTransform(error);
    }
    throw toVestigeError(error);
  }
}

/**
 * Retry a function with exponential backoff
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelayMs?: number;
    maxDelayMs?: number;
    shouldRetry?: (error: unknown) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelayMs = 100,
    maxDelayMs = 5000,
    shouldRetry = () => true,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries || !shouldRetry(error)) {
        throw toVestigeError(error);
      }

      const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  throw toVestigeError(lastError);
}

// =============================================================================
// Result Type (Optional Pattern)
// =============================================================================

/**
 * Result type for functional error handling
 */
export type Result<T, E = VestigeError> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Create a success result
 */
export function ok<T>(data: T): Result<T, never> {
  return { success: true, data };
}

/**
 * Create an error result
 */
export function err<E = VestigeError>(error: E): Result<never, E> {
  return { success: false, error };
}

/**
 * Check if result is success
 */
export function isOk<T, E>(result: Result<T, E>): result is { success: true; data: T } {
  return result.success;
}

/**
 * Check if result is error
 */
export function isErr<T, E>(result: Result<T, E>): result is { success: false; error: E } {
  return !result.success;
}

/**
 * Unwrap a result, throwing if it's an error
 */
export function unwrap<T, E>(result: Result<T, E>): T {
  if (result.success) {
    return result.data;
  }
  throw (result as { success: false; error: E }).error;
}

/**
 * Unwrap a result with a default value
 */
export function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  if (result.success) {
    return result.data;
  }
  return defaultValue;
}

/**
 * Map over a successful result
 */
export function mapResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => U
): Result<U, E> {
  if (result.success) {
    return ok(fn(result.data));
  }
  return result as { success: false; error: E };
}

/**
 * Map over an error result
 */
export function mapError<T, E, F>(
  result: Result<T, E>,
  fn: (error: E) => F
): Result<T, F> {
  if (!result.success) {
    return err(fn((result as { success: false; error: E }).error));
  }
  return result as { success: true; data: T };
}

/**
 * Execute a function and return a Result
 */
export async function tryCatch<T>(
  fn: () => Promise<T>
): Promise<Result<T, VestigeError>> {
  try {
    const data = await fn();
    return ok(data);
  } catch (error) {
    return err(toVestigeError(error));
  }
}

/**
 * Execute a synchronous function and return a Result
 */
export function tryCatchSync<T>(fn: () => T): Result<T, VestigeError> {
  try {
    const data = fn();
    return ok(data);
  } catch (error) {
    return err(toVestigeError(error));
  }
}

// =============================================================================
// Error Assertion Helpers
// =============================================================================

/**
 * Assert a condition, throwing ValidationError if false
 */
export function assertValid(
  condition: boolean,
  message: string,
  details?: Record<string, unknown>
): asserts condition {
  if (!condition) {
    throw new ValidationError(message, details);
  }
}

/**
 * Assert a value is not null or undefined
 */
export function assertDefined<T>(
  value: T | null | undefined,
  resource: string,
  id?: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new NotFoundError(resource, id);
  }
}

/**
 * Assert a value exists, returning it if so
 */
export function requireDefined<T>(
  value: T | null | undefined,
  resource: string,
  id?: string
): T {
  assertDefined(value, resource, id);
  return value;
}

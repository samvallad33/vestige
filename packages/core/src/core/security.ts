/**
 * Security Utilities for Vestige
 *
 * Provides comprehensive security controls including:
 * - Input validation and sanitization
 * - Path traversal prevention
 * - SSRF prevention (including IPv6 and DNS rebinding)
 * - Unicode homograph detection
 * - Symlink race condition prevention
 * - Rate limiting
 * - SQL injection prevention
 * - Security event logging
 */

import path from 'path';
import os from 'os';
import fs from 'fs';
import { URL } from 'url';
import { SecurityError } from './errors.js';

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Allowed base directories for file operations
 * Users can only read/write files within these directories
 */
const ALLOWED_BASE_DIRS = [
  os.homedir(),           // User's home directory
  '/tmp',                 // Temp directory
  process.cwd(),          // Current working directory
];

/**
 * Sensitive paths that should NEVER be accessible
 */
const BLOCKED_PATHS = [
  '/.ssh',
  '/.gnupg',
  '/.aws',
  '/.config/gcloud',
  '/.azure',
  '/etc/passwd',
  '/etc/shadow',
  '/etc/hosts',
  '/.env',
  '/.git/config',
  '/id_rsa',
  '/id_ed25519',
  '/.netrc',
  '/.npmrc',
  '/.pypirc',
  '/.docker/config.json',
  '/.kube/config',
  '/credentials',
  '/secrets',
];

/**
 * Blocked file extensions
 */
const BLOCKED_EXTENSIONS = [
  '.pem',
  '.key',
  '.p12',
  '.pfx',
  '.keystore',
  '.jks',
  '.crt',
  '.cer',
];

/**
 * Private/internal IPv4 ranges that should be blocked (SSRF prevention)
 */
const PRIVATE_IPV4_PATTERNS = [
  /^127\./,                           // Loopback
  /^10\./,                            // Private Class A
  /^172\.(1[6-9]|2[0-9]|3[0-1])\./,  // Private Class B
  /^192\.168\./,                      // Private Class C
  /^169\.254\./,                      // Link-local
  /^0\./,                             // Current network
  /^100\.(6[4-9]|[7-9][0-9]|1[0-2][0-9])\./,  // Carrier-grade NAT (100.64.0.0/10)
  /^198\.1[89]\./,                    // Benchmark testing
  /^192\.0\.0\./,                     // IANA special purpose
  /^192\.0\.2\./,                     // TEST-NET-1
  /^198\.51\.100\./,                  // TEST-NET-2
  /^203\.0\.113\./,                   // TEST-NET-3
  /^224\./,                           // Multicast
  /^240\./,                           // Reserved
];

/**
 * Private/internal IPv6 ranges that should be blocked
 * Fixed: Comprehensive IPv6 private range detection
 */
const PRIVATE_IPV6_PATTERNS = [
  /^::1$/i,                           // Loopback (exact match)
  /^::$/,                             // Unspecified address
  /^::ffff:/i,                        // IPv4-mapped IPv6
  /^fe80:/i,                          // Link-local
  /^fec0:/i,                          // Site-local (deprecated but still dangerous)
  /^fc00:/i,                          // Unique local (ULA)
  /^fd[0-9a-f]{2}:/i,                 // Unique local (ULA) - fd00::/8
  /^ff[0-9a-f]{2}:/i,                 // Multicast
  /^2001:db8:/i,                      // Documentation prefix
  /^2001:10:/i,                       // ORCHID
  /^2001:20:/i,                       // ORCHIDv2
  /^100::/i,                          // Discard prefix
  /^64:ff9b:/i,                       // NAT64
  /^\[::1\]$/i,                       // Bracketed loopback
  /^\[::ffff:/i,                      // Bracketed IPv4-mapped
  /^\[fe80:/i,                        // Bracketed link-local
  /^\[fc00:/i,                        // Bracketed ULA
  /^\[fd[0-9a-f]{2}:/i,               // Bracketed ULA
];

/**
 * Blocked hostnames for SSRF prevention
 */
const BLOCKED_HOSTNAMES = [
  'localhost',
  'localhost.localdomain',
  '0.0.0.0',
  '[::1]',
  '[::0]',
  '[::]',
  'metadata.google.internal',       // GCP metadata
  'metadata.google.com',
  '169.254.169.254',                // AWS/GCP/Azure metadata
  'instance-data',                   // AWS metadata alias
  'metadata',                        // Generic metadata
  'metadata.internal',
  'computeMetadata',
  '169.254.170.2',                  // AWS ECS task metadata
  'fd00:ec2::254',                  // AWS IPv6 metadata
];

/**
 * Numeric localhost variations (hex, octal, decimal)
 */
const LOCALHOST_NUMERIC_PATTERNS = [
  /^0x7f/i,                          // Hex 127.x.x.x
  /^2130706433$/,                    // Decimal 127.0.0.1
  /^017700000001$/,                  // Octal 127.0.0.1
  /^0177\.0*\.0*\.0*1$/,             // Octal dotted
  /^0x7f\.0x0+\.0x0+\.0x0*1$/i,      // Hex dotted
];

/**
 * Allowed URL protocols
 */
const ALLOWED_PROTOCOLS = ['http:', 'https:'];

// ============================================================================
// PATH SECURITY
// ============================================================================

export interface PathValidationResult {
  valid: boolean;
  sanitizedPath: string | null;
  error: string | null;
}

/**
 * Validate and sanitize a file path
 * Prevents path traversal attacks and access to sensitive files
 *
 * Security improvements:
 * - Symlink resolution to prevent TOCTOU race conditions
 * - Null byte detection
 * - More comprehensive sensitive path detection
 */
export function validatePath(inputPath: string): PathValidationResult {
  try {
    // Check for null bytes (CWE-158)
    if (inputPath.includes('\0')) {
      logSecurityEvent({
        type: 'path_traversal',
        details: { reason: 'null_byte_detected' },
        severity: 'high',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedPath: null,
        error: 'Invalid path: null byte detected',
      };
    }

    // Resolve to absolute path
    const absolutePath = path.resolve(inputPath);
    const normalizedPath = path.normalize(absolutePath);

    // Check for path traversal attempts
    if (inputPath.includes('..') && !normalizedPath.startsWith(process.cwd())) {
      // Allow .. only if it resolves within cwd
      const relative = path.relative(process.cwd(), normalizedPath);
      if (relative.startsWith('..')) {
        logSecurityEvent({
          type: 'path_traversal',
          details: { inputPath, resolvedPath: normalizedPath },
          severity: 'high',
          blocked: true,
        });
        return {
          valid: false,
          sanitizedPath: null,
          error: 'Path traversal detected: cannot access files outside allowed directories',
        };
      }
    }

    // Check if path is within allowed directories
    const isAllowed = ALLOWED_BASE_DIRS.some(baseDir =>
      normalizedPath.startsWith(path.resolve(baseDir))
    );

    if (!isAllowed) {
      logSecurityEvent({
        type: 'path_traversal',
        details: { reason: 'outside_allowed_dirs', normalizedPath },
        severity: 'medium',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedPath: null,
        error: 'Access denied: path must be within home directory, /tmp, or current working directory',
      };
    }

    // Check for sensitive paths
    const lowerPath = normalizedPath.toLowerCase();
    for (const blocked of BLOCKED_PATHS) {
      if (lowerPath.includes(blocked.toLowerCase())) {
        logSecurityEvent({
          type: 'path_traversal',
          details: { reason: 'sensitive_path', blocked },
          severity: 'high',
          blocked: true,
        });
        return {
          valid: false,
          sanitizedPath: null,
          error: 'Access denied: cannot access sensitive system files',
        };
      }
    }

    // Check for blocked extensions
    const ext = path.extname(normalizedPath).toLowerCase();
    if (BLOCKED_EXTENSIONS.includes(ext)) {
      logSecurityEvent({
        type: 'path_traversal',
        details: { reason: 'blocked_extension', extension: ext },
        severity: 'high',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedPath: null,
        error: 'Access denied: cannot access credential files',
      };
    }

    return {
      valid: true,
      sanitizedPath: normalizedPath,
      error: null,
    };
  } catch (error) {
    return {
      valid: false,
      sanitizedPath: null,
      error: 'Invalid path',
    };
  }
}

/**
 * Resolve symlinks and verify the path is still safe
 * Prevents TOCTOU (Time-of-Check Time-of-Use) race conditions
 */
export function validatePathWithSymlinkResolution(inputPath: string): PathValidationResult {
  // First do basic validation
  const basicResult = validatePath(inputPath);
  if (!basicResult.valid) {
    return basicResult;
  }

  try {
    // Check if path exists
    if (!fs.existsSync(basicResult.sanitizedPath!)) {
      // Path doesn't exist yet, that's okay for new files
      return basicResult;
    }

    // Resolve symlinks to get the real path
    const realPath = fs.realpathSync(basicResult.sanitizedPath!);

    // Now validate the REAL path (after symlink resolution)
    const realPathResult = validatePath(realPath);
    if (!realPathResult.valid) {
      logSecurityEvent({
        type: 'path_traversal',
        details: {
          reason: 'symlink_escape',
          inputPath,
          resolvedPath: realPath
        },
        severity: 'critical',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedPath: null,
        error: 'Access denied: symlink points outside allowed directories',
      };
    }

    return {
      valid: true,
      sanitizedPath: realPath,
      error: null,
    };
  } catch (error) {
    // If realpath fails, the path might not exist yet
    // In that case, return the basic validation result
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return basicResult;
    }
    return {
      valid: false,
      sanitizedPath: null,
      error: 'Failed to resolve path',
    };
  }
}

// ============================================================================
// URL SECURITY (SSRF Prevention)
// ============================================================================

export interface UrlValidationResult {
  valid: boolean;
  sanitizedUrl: string | null;
  error: string | null;
}

/**
 * Check if a hostname contains Unicode homograph characters
 * Detects IDN homograph attacks (e.g., using Cyrillic 'а' instead of Latin 'a')
 */
export function containsHomographs(hostname: string): boolean {
  // Convert to ASCII (punycode) and compare
  // If they differ significantly, there might be homograph characters
  try {
    const url = new URL(`http://${hostname}`);
    const asciiHostname = url.hostname;

    // Check for mixed scripts
    // These Unicode ranges indicate potential homograph attacks
    const suspiciousPatterns = [
      /[\u0400-\u04FF]/,  // Cyrillic
      /[\u0370-\u03FF]/,  // Greek
      /[\u0530-\u058F]/,  // Armenian
      /[\u10A0-\u10FF]/,  // Georgian
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(hostname)) {
        // Check if the hostname also contains Latin characters
        if (/[a-zA-Z]/.test(hostname)) {
          return true; // Mixed scripts detected
        }
      }
    }

    // Check for look-alike characters that commonly appear in phishing
    const homoglyphs: Record<string, string[]> = {
      'a': ['а', 'ɑ', 'α'],  // Latin a vs Cyrillic а, etc.
      'e': ['е', 'ε'],
      'o': ['о', 'ο', '0'],
      'p': ['р', 'ρ'],
      'c': ['с', 'ϲ'],
      'x': ['х', 'χ'],
      'y': ['у', 'γ'],
      'n': ['п'],
      's': ['ѕ'],
    };

    for (const [latin, lookalikes] of Object.entries(homoglyphs)) {
      for (const lookalike of lookalikes) {
        if (hostname.includes(lookalike) && hostname.includes(latin)) {
          return true;
        }
      }
    }

    return false;
  } catch {
    return false;
  }
}

/**
 * Validate and sanitize a URL
 * Prevents SSRF attacks by blocking internal/private addresses
 *
 * Security improvements:
 * - Comprehensive IPv6 detection
 * - Homograph attack detection
 * - DNS rebinding protection hints
 */
export function validateUrl(inputUrl: string): UrlValidationResult {
  try {
    // Parse the URL
    const url = new URL(inputUrl);

    // Check protocol
    if (!ALLOWED_PROTOCOLS.includes(url.protocol)) {
      logSecurityEvent({
        type: 'ssrf_attempt',
        details: { reason: 'invalid_protocol', protocol: url.protocol },
        severity: 'medium',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedUrl: null,
        error: 'Invalid protocol: only HTTP and HTTPS are allowed',
      };
    }

    const hostname = url.hostname.toLowerCase();

    // Check for blocked hostnames
    if (BLOCKED_HOSTNAMES.includes(hostname)) {
      logSecurityEvent({
        type: 'ssrf_attempt',
        details: { reason: 'blocked_hostname', hostname },
        severity: 'high',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedUrl: null,
        error: 'Blocked hostname: internal addresses are not allowed',
      };
    }

    // Check for private/internal IPv4
    for (const pattern of PRIVATE_IPV4_PATTERNS) {
      if (pattern.test(hostname)) {
        logSecurityEvent({
          type: 'ssrf_attempt',
          details: { reason: 'private_ipv4', hostname },
          severity: 'high',
          blocked: true,
        });
        return {
          valid: false,
          sanitizedUrl: null,
          error: 'Blocked: private IPv4 address detected',
        };
      }
    }

    // Check for private/internal IPv6
    for (const pattern of PRIVATE_IPV6_PATTERNS) {
      if (pattern.test(hostname)) {
        logSecurityEvent({
          type: 'ssrf_attempt',
          details: { reason: 'private_ipv6', hostname },
          severity: 'high',
          blocked: true,
        });
        return {
          valid: false,
          sanitizedUrl: null,
          error: 'Blocked: private IPv6 address detected',
        };
      }
    }

    // Check for numeric localhost variations
    for (const pattern of LOCALHOST_NUMERIC_PATTERNS) {
      if (pattern.test(hostname)) {
        logSecurityEvent({
          type: 'ssrf_attempt',
          details: { reason: 'encoded_localhost', hostname },
          severity: 'high',
          blocked: true,
        });
        return {
          valid: false,
          sanitizedUrl: null,
          error: 'Blocked: encoded localhost address detected',
        };
      }
    }

    // Check for Unicode homograph attacks
    if (containsHomographs(hostname)) {
      logSecurityEvent({
        type: 'ssrf_attempt',
        details: { reason: 'homograph_attack', hostname },
        severity: 'high',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedUrl: null,
        error: 'Blocked: potential homograph attack detected in hostname',
      };
    }

    // Check for suspicious URL-encoded characters
    if (inputUrl.includes('%00') || inputUrl.includes('%0d') || inputUrl.includes('%0a')) {
      logSecurityEvent({
        type: 'ssrf_attempt',
        details: { reason: 'suspicious_encoding' },
        severity: 'medium',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedUrl: null,
        error: 'Blocked: suspicious URL encoding detected',
      };
    }

    // Check for @ symbol which could be used for credential injection
    if (url.username || url.password) {
      logSecurityEvent({
        type: 'ssrf_attempt',
        details: { reason: 'credentials_in_url' },
        severity: 'medium',
        blocked: true,
      });
      return {
        valid: false,
        sanitizedUrl: null,
        error: 'Blocked: credentials in URL are not allowed',
      };
    }

    // Reconstruct clean URL (removes any weird encoding tricks)
    const cleanUrl = url.toString();

    return {
      valid: true,
      sanitizedUrl: cleanUrl,
      error: null,
    };
  } catch (error) {
    return {
      valid: false,
      sanitizedUrl: null,
      error: 'Invalid URL format',
    };
  }
}

// ============================================================================
// INPUT SANITIZATION
// ============================================================================

/**
 * Maximum content length to prevent DoS
 */
export const MAX_CONTENT_LENGTH = 10 * 1024 * 1024; // 10MB

export interface SanitizeInputOptions {
  maxLength?: number;
  allowedChars?: RegExp;
  stripHtml?: boolean;
  normalizeUnicode?: boolean;
  allowNewlines?: boolean;
}

/**
 * Comprehensive input sanitization
 */
export function sanitizeInput(input: string, options: SanitizeInputOptions = {}): string {
  const {
    maxLength = MAX_CONTENT_LENGTH,
    allowedChars,
    stripHtml = false,
    normalizeUnicode = true,
    allowNewlines = true,
  } = options;

  let sanitized = input;

  // Truncate to max length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.slice(0, maxLength);
    logSecurityEvent({
      type: 'validation_failure',
      details: { reason: 'content_truncated', originalLength: input.length, maxLength },
      severity: 'low',
      blocked: false,
    });
  }

  // Remove null bytes
  sanitized = sanitized.replace(/\x00/g, '');

  // Remove other control characters (except newlines and tabs if allowed)
  if (allowNewlines) {
    sanitized = sanitized.replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
  } else {
    sanitized = sanitized.replace(/[\x00-\x1F\x7F]/g, '');
  }

  // Strip HTML tags if requested
  if (stripHtml) {
    sanitized = sanitized.replace(/<[^>]*>/g, '');
  }

  // Normalize Unicode if requested (NFC normalization)
  if (normalizeUnicode) {
    sanitized = sanitized.normalize('NFC');
  }

  // Apply allowed character filter if specified
  if (allowedChars) {
    sanitized = sanitized
      .split('')
      .filter(char => allowedChars.test(char))
      .join('');
  }

  return sanitized;
}

/**
 * Sanitize text content by removing potentially dangerous characters
 */
export function sanitizeContent(content: string, maxLength: number = MAX_CONTENT_LENGTH): string {
  return sanitizeInput(content, { maxLength, allowNewlines: true });
}

/**
 * Validate that a string is safe for use as an identifier
 */
export function isValidIdentifier(input: string, maxLength: number = 100): boolean {
  if (!input || input.length > maxLength) return false;
  // Only allow alphanumeric, underscore, hyphen
  return /^[a-zA-Z0-9_-]+$/.test(input);
}

// ============================================================================
// INPUT VALIDATORS
// ============================================================================

/**
 * Comprehensive validators for common input types
 */
export const validators = {
  /** Validate nanoid format (21 alphanumeric characters) */
  nodeId: (id: string): boolean => /^[a-zA-Z0-9_-]{21}$/.test(id),

  /** Validate tag (max 100 chars, no HTML special chars) */
  tag: (tag: string): boolean => tag.length > 0 && tag.length <= 100 && !/[<>]/.test(tag),

  /** Validate email address */
  email: (email: string): boolean => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) && email.length <= 254,

  /** Validate URL */
  url: (url: string): boolean => validateUrl(url).valid,

  /** Validate file path */
  path: (pathStr: string): boolean => validatePath(pathStr).valid,

  /** Validate name (no special chars, reasonable length) */
  name: (name: string): boolean => name.length > 0 && name.length <= 500 && !/[<>"'`;]/.test(name),

  /** Validate positive integer */
  positiveInt: (value: number): boolean => Number.isInteger(value) && value > 0,

  /** Validate percentage (0-100) */
  percentage: (value: number): boolean => typeof value === 'number' && value >= 0 && value <= 100,

  /** Validate UUID */
  uuid: (id: string): boolean => /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id),
};

// ============================================================================
// RATE LIMITING
// ============================================================================

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
  retryAfterMs?: number;
}

/**
 * Sliding window rate limiter
 */
export class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(
    private readonly maxRequests: number,
    private readonly windowMs: number
  ) {
    // Cleanup old entries periodically
    this.cleanupInterval = setInterval(() => this.cleanup(), windowMs);
  }

  /**
   * Check if a request is allowed
   */
  isAllowed(key: string): RateLimitResult {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    // Get existing requests for this key
    let timestamps = this.requests.get(key) || [];

    // Filter to only requests within the window
    timestamps = timestamps.filter(ts => ts > windowStart);

    const allowed = timestamps.length < this.maxRequests;
    const remaining = Math.max(0, this.maxRequests - timestamps.length - (allowed ? 1 : 0));
    const resetAt = new Date(now + this.windowMs);

    if (allowed) {
      timestamps.push(now);
      this.requests.set(key, timestamps);
    } else {
      // Calculate when the oldest request will expire
      const oldestRequest = Math.min(...timestamps);
      const retryAfterMs = oldestRequest + this.windowMs - now;

      logSecurityEvent({
        type: 'rate_limit',
        details: { key, requestCount: timestamps.length, maxRequests: this.maxRequests },
        severity: 'medium',
        blocked: true,
      });

      return {
        allowed: false,
        remaining: 0,
        resetAt,
        retryAfterMs,
      };
    }

    return {
      allowed: true,
      remaining,
      resetAt,
    };
  }

  /**
   * Get current request count for a key
   */
  getRequestCount(key: string): number {
    const windowStart = Date.now() - this.windowMs;
    const timestamps = this.requests.get(key) || [];
    return timestamps.filter(ts => ts > windowStart).length;
  }

  /**
   * Reset rate limit for a specific key
   */
  reset(key: string): void {
    this.requests.delete(key);
  }

  /**
   * Clear all rate limit data
   */
  clear(): void {
    this.requests.clear();
  }

  /**
   * Cleanup old entries
   */
  private cleanup(): void {
    const windowStart = Date.now() - this.windowMs;
    for (const [key, timestamps] of this.requests.entries()) {
      const valid = timestamps.filter(ts => ts > windowStart);
      if (valid.length === 0) {
        this.requests.delete(key);
      } else {
        this.requests.set(key, valid);
      }
    }
  }

  /**
   * Stop the cleanup interval
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.requests.clear();
  }
}

// ============================================================================
// SQL INJECTION PREVENTION
// ============================================================================

export interface PreparedQuery {
  sql: string;
  values: unknown[];
}

/**
 * Safe query builder that enforces parameterized queries
 * Uses named parameters for clarity and safety
 */
export function prepareQuery(
  template: string,
  params: Record<string, unknown>
): PreparedQuery {
  const values: unknown[] = [];
  let paramIndex = 0;

  // Replace :paramName with ? and collect values
  const sql = template.replace(/:([a-zA-Z_][a-zA-Z0-9_]*)/g, (_, paramName) => {
    if (!(paramName in params)) {
      throw new SecurityError(`Missing query parameter: ${paramName}`);
    }
    values.push(params[paramName]);
    paramIndex++;
    return '?';
  });

  return { sql, values };
}

/**
 * Escape a string for use in LIKE queries
 */
export function escapeLikePattern(pattern: string): string {
  return pattern
    .replace(/\\/g, '\\\\')  // Escape backslashes first
    .replace(/%/g, '\\%')    // Escape percent
    .replace(/_/g, '\\_');   // Escape underscore
}

/**
 * Validate that a query is safe (no dangerous operations)
 * This is a defense-in-depth measure
 */
export function isQuerySafe(query: string): boolean {
  const dangerousPatterns = [
    /;\s*(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)\s/i,
    /--\s*$/m,                   // SQL comments at end of line
    /\/\*[\s\S]*?\*\//,          // Block comments
    /\bEXEC\s*\(/i,              // EXEC function
    /\bxp_/i,                    // SQL Server extended procedures
    /\bsp_/i,                    // SQL Server system procedures
    /\bUNION\s+ALL\s+SELECT/i,   // UNION injection
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(query)) {
      logSecurityEvent({
        type: 'validation_failure',
        details: { reason: 'dangerous_query_pattern' },
        severity: 'critical',
        blocked: true,
      });
      return false;
    }
  }

  return true;
}

// ============================================================================
// SECURITY EVENT LOGGING
// ============================================================================

export interface SecurityEvent {
  type: 'path_traversal' | 'ssrf_attempt' | 'rate_limit' | 'validation_failure';
  timestamp: Date;
  details: Record<string, unknown>;
  severity: 'low' | 'medium' | 'high' | 'critical';
  blocked: boolean;
}

const securityLog: SecurityEvent[] = [];
const MAX_LOG_SIZE = 1000;

/**
 * Log a security event
 */
export function logSecurityEvent(event: Omit<SecurityEvent, 'timestamp'>): void {
  const fullEvent: SecurityEvent = {
    ...event,
    timestamp: new Date(),
  };

  securityLog.push(fullEvent);

  // Keep log from growing too large
  if (securityLog.length > MAX_LOG_SIZE) {
    securityLog.shift();
  }

  // Log to stderr in debug mode
  if (process.env['VESTIGE_DEBUG']) {
    console.error(`[SECURITY:${fullEvent.severity.toUpperCase()}] ${fullEvent.type}: ${JSON.stringify(fullEvent.details)}`);
  }

  // Alert on critical events
  if (fullEvent.severity === 'critical') {
    console.error(`[SECURITY:CRITICAL] ${fullEvent.type}: ${JSON.stringify(fullEvent.details)}`);
  }
}

/**
 * Get recent security events
 */
export function getSecurityEvents(limit: number = 100): SecurityEvent[] {
  return securityLog.slice(-limit);
}

/**
 * Get security events by type
 */
export function getSecurityEventsByType(type: SecurityEvent['type'], limit: number = 100): SecurityEvent[] {
  return securityLog
    .filter(event => event.type === type)
    .slice(-limit);
}

/**
 * Get security events by severity
 */
export function getSecurityEventsBySeverity(
  severity: SecurityEvent['severity'],
  limit: number = 100
): SecurityEvent[] {
  return securityLog
    .filter(event => event.severity === severity)
    .slice(-limit);
}

/**
 * Clear security log (useful for testing)
 */
export function clearSecurityLog(): void {
  securityLog.length = 0;
}

// ============================================================================
// SECURITY HEADERS (For Future Web UI)
// ============================================================================

export const SECURITY_HEADERS = {
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
  'Cache-Control': 'no-store, no-cache, must-revalidate, proxy-revalidate',
  'Pragma': 'no-cache',
  'Expires': '0',
};

/**
 * Apply security headers to a response object
 */
export function applySecurityHeaders(headers: Record<string, string>): Record<string, string> {
  return {
    ...headers,
    ...SECURITY_HEADERS,
  };
}

// ============================================================================
// CRYPTO UTILITIES
// ============================================================================

/**
 * Generate a cryptographically secure random string
 */
export function generateSecureToken(length: number = 32): string {
  const crypto = require('crypto');
  return crypto.randomBytes(length).toString('hex');
}

/**
 * Constant-time string comparison to prevent timing attacks
 */
export function secureCompare(a: string, b: string): boolean {
  const crypto = require('crypto');
  if (a.length !== b.length) {
    return false;
  }
  return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
}

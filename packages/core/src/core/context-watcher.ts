/**
 * Ghost in the Shell - Context Watcher
 *
 * Watches the active window and clipboard to provide contextual awareness.
 * Vestige sees what you see.
 *
 * Features:
 * - Active window title detection (macOS via AppleScript)
 * - Clipboard monitoring
 * - Context file for MCP injection
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';

// ============================================================================
// TYPES
// ============================================================================

export interface SystemContext {
  timestamp: string;
  activeWindow: {
    app: string;
    title: string;
  } | null;
  clipboard: string | null;
  workingDirectory: string;
  gitBranch: string | null;
  recentFiles: string[];
}

// ============================================================================
// CONTEXT FILE LOCATION
// ============================================================================

const CONTEXT_FILE = path.join(os.homedir(), '.vestige', 'context.json');

// ============================================================================
// PLATFORM-SPECIFIC IMPLEMENTATIONS
// ============================================================================

/**
 * Get active window info on macOS using AppleScript
 */
function getActiveWindowMac(): { app: string; title: string } | null {
  try {
    // Get frontmost app name
    const appScript = `
      tell application "System Events"
        set frontApp to first application process whose frontmost is true
        return name of frontApp
      end tell
    `;
    const app = execSync(`osascript -e '${appScript}'`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    }).trim();

    // Get window title
    const titleScript = `
      tell application "System Events"
        tell (first application process whose frontmost is true)
          if (count of windows) > 0 then
            return name of front window
          else
            return ""
          end if
        end tell
      end tell
    `;
    const title = execSync(`osascript -e '${titleScript}'`, {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    }).trim();

    return { app, title };
  } catch {
    return null;
  }
}

/**
 * Get clipboard content on macOS
 */
function getClipboardMac(): string | null {
  try {
    const content = execSync('pbpaste', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
      maxBuffer: 1024 * 100, // 100KB max
    });
    // Truncate long clipboard content
    if (content.length > 2000) {
      return content.slice(0, 2000) + '\n... [truncated]';
    }
    return content || null;
  } catch {
    return null;
  }
}

/**
 * Get current git branch
 */
function getGitBranch(): string | null {
  try {
    return execSync('git rev-parse --abbrev-ref HEAD', {
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe'],
    }).trim();
  } catch {
    return null;
  }
}

/**
 * Get recently modified files in current directory
 */
function getRecentFiles(): string[] {
  try {
    // Get files modified in last hour, sorted by time
    const result = execSync(
      'find . -type f -mmin -60 -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/dist/*" 2>/dev/null | head -10',
      {
        encoding: 'utf-8',
        stdio: ['pipe', 'pipe', 'pipe'],
      }
    );
    return result
      .split('\n')
      .map(f => f.trim())
      .filter(Boolean)
      .slice(0, 10);
  } catch {
    return [];
  }
}

// ============================================================================
// CONTEXT CAPTURE
// ============================================================================

/**
 * Capture current system context
 */
export function captureContext(): SystemContext {
  const platform = process.platform;

  let activeWindow: { app: string; title: string } | null = null;
  let clipboard: string | null = null;

  if (platform === 'darwin') {
    activeWindow = getActiveWindowMac();
    clipboard = getClipboardMac();
  }
  // TODO: Add Windows and Linux support

  return {
    timestamp: new Date().toISOString(),
    activeWindow,
    clipboard,
    workingDirectory: process.cwd(),
    gitBranch: getGitBranch(),
    recentFiles: getRecentFiles(),
  };
}

/**
 * Format context for injection into Claude prompts
 */
export function formatContextForInjection(context: SystemContext): string {
  const parts: string[] = [];

  if (context.activeWindow) {
    parts.push(`Active: ${context.activeWindow.app} - ${context.activeWindow.title}`);
  }

  if (context.gitBranch) {
    parts.push(`Git: ${context.gitBranch}`);
  }

  if (context.recentFiles.length > 0) {
    parts.push(`Recent: ${context.recentFiles.slice(0, 3).join(', ')}`);
  }

  if (context.clipboard && context.clipboard.length < 500) {
    parts.push(`Clipboard: "${context.clipboard.slice(0, 200)}${context.clipboard.length > 200 ? '...' : ''}"`);
  }

  return parts.join(' | ');
}

/**
 * Save context to file for external consumption
 */
export function saveContext(context: SystemContext): void {
  try {
    const dir = path.dirname(CONTEXT_FILE);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONTEXT_FILE, JSON.stringify(context, null, 2));
  } catch {
    // Ignore file write errors
  }
}

/**
 * Read saved context from file
 */
export function readSavedContext(): SystemContext | null {
  try {
    if (fs.existsSync(CONTEXT_FILE)) {
      const content = fs.readFileSync(CONTEXT_FILE, 'utf-8');
      return JSON.parse(content) as SystemContext;
    }
  } catch {
    // Ignore read errors
  }
  return null;
}

// ============================================================================
// WATCHER DAEMON
// ============================================================================

let watcherInterval: NodeJS.Timeout | null = null;

/**
 * Start the context watcher daemon
 * Updates context every N seconds
 */
export function startContextWatcher(intervalMs: number = 5000): void {
  if (watcherInterval) {
    console.log('Context watcher already running');
    return;
  }

  console.log(`Starting context watcher (interval: ${intervalMs}ms)`);

  // Capture immediately
  const context = captureContext();
  saveContext(context);

  // Then update periodically
  watcherInterval = setInterval(() => {
    const ctx = captureContext();
    saveContext(ctx);
  }, intervalMs);
}

/**
 * Stop the context watcher daemon
 */
export function stopContextWatcher(): void {
  if (watcherInterval) {
    clearInterval(watcherInterval);
    watcherInterval = null;
    console.log('Context watcher stopped');
  }
}

/**
 * Check if watcher is running
 */
export function isWatcherRunning(): boolean {
  return watcherInterval !== null;
}

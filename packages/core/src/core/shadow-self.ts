/**
 * The Shadow Self - Unsolved Problems Queue
 *
 * "Your subconscious that keeps working while you're not looking."
 *
 * When you say "I don't know how to fix this," Vestige logs it.
 * The Shadow periodically re-attacks these problems with new context.
 *
 * This turns Vestige from a passive memory into an active problem-solver.
 */

import Database from 'better-sqlite3';
import { nanoid } from 'nanoid';
import path from 'path';
import fs from 'fs';
import os from 'os';

// ============================================================================
// TYPES
// ============================================================================

export interface UnsolvedProblem {
  id: string;
  description: string;
  context: string;         // Original context when problem was logged
  tags: string[];
  status: 'open' | 'investigating' | 'solved' | 'abandoned';
  priority: number;        // 1-5, higher = more urgent
  attempts: number;        // How many times Shadow has tried to solve it
  lastAttemptAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
  solution: string | null; // If solved, what was the solution?
  relatedNodeIds: string[]; // Knowledge nodes that might help
}

export interface ShadowInsight {
  problemId: string;
  insight: string;
  source: 'keyword_match' | 'new_knowledge' | 'pattern_recognition';
  confidence: number;
  relatedNodeIds: string[];
  createdAt: Date;
}

// ============================================================================
// DATABASE SETUP
// ============================================================================

const SHADOW_DB_PATH = path.join(os.homedir(), '.vestige', 'shadow.db');

function initializeShadowDb(): Database.Database {
  const dir = path.dirname(SHADOW_DB_PATH);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const db = new Database(SHADOW_DB_PATH);

  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 5000');

  // Unsolved problems table
  db.exec(`
    CREATE TABLE IF NOT EXISTS unsolved_problems (
      id TEXT PRIMARY KEY,
      description TEXT NOT NULL,
      context TEXT,
      tags TEXT DEFAULT '[]',
      status TEXT DEFAULT 'open',
      priority INTEGER DEFAULT 3,
      attempts INTEGER DEFAULT 0,
      last_attempt_at TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      solution TEXT,
      related_node_ids TEXT DEFAULT '[]'
    );

    CREATE INDEX IF NOT EXISTS idx_problems_status ON unsolved_problems(status);
    CREATE INDEX IF NOT EXISTS idx_problems_priority ON unsolved_problems(priority);
  `);

  // Insights discovered by Shadow
  db.exec(`
    CREATE TABLE IF NOT EXISTS shadow_insights (
      id TEXT PRIMARY KEY,
      problem_id TEXT NOT NULL,
      insight TEXT NOT NULL,
      source TEXT NOT NULL,
      confidence REAL DEFAULT 0.5,
      related_node_ids TEXT DEFAULT '[]',
      created_at TEXT NOT NULL,

      FOREIGN KEY (problem_id) REFERENCES unsolved_problems(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_insights_problem ON shadow_insights(problem_id);
  `);

  return db;
}

// ============================================================================
// SHADOW SELF CLASS
// ============================================================================

export class ShadowSelf {
  private db: Database.Database;

  constructor() {
    this.db = initializeShadowDb();
  }

  /**
   * Log a new unsolved problem
   */
  logProblem(description: string, options: {
    context?: string;
    tags?: string[];
    priority?: number;
  } = {}): UnsolvedProblem {
    const id = nanoid();
    const now = new Date().toISOString();

    const stmt = this.db.prepare(`
      INSERT INTO unsolved_problems (
        id, description, context, tags, status, priority,
        attempts, last_attempt_at, created_at, updated_at,
        solution, related_node_ids
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      id,
      description,
      options.context || '',
      JSON.stringify(options.tags || []),
      'open',
      options.priority || 3,
      0,
      null,
      now,
      now,
      null,
      '[]'
    );

    return this.getProblem(id)!;
  }

  /**
   * Get a specific problem
   */
  getProblem(id: string): UnsolvedProblem | null {
    const stmt = this.db.prepare('SELECT * FROM unsolved_problems WHERE id = ?');
    const row = stmt.get(id) as Record<string, unknown> | undefined;
    if (!row) return null;
    return this.rowToProblem(row);
  }

  /**
   * Get all open problems
   */
  getOpenProblems(): UnsolvedProblem[] {
    const stmt = this.db.prepare(`
      SELECT * FROM unsolved_problems
      WHERE status IN ('open', 'investigating')
      ORDER BY priority DESC, created_at ASC
    `);
    const rows = stmt.all() as Record<string, unknown>[];
    return rows.map(row => this.rowToProblem(row));
  }

  /**
   * Update problem status
   */
  updateStatus(id: string, status: UnsolvedProblem['status'], solution?: string): void {
    const now = new Date().toISOString();
    const stmt = this.db.prepare(`
      UPDATE unsolved_problems
      SET status = ?, solution = ?, updated_at = ?
      WHERE id = ?
    `);
    stmt.run(status, solution || null, now, id);
  }

  /**
   * Mark problem as solved
   */
  markSolved(id: string, solution: string): void {
    this.updateStatus(id, 'solved', solution);
  }

  /**
   * Add insight to a problem
   */
  addInsight(problemId: string, insight: string, options: {
    source?: ShadowInsight['source'];
    confidence?: number;
    relatedNodeIds?: string[];
  } = {}): ShadowInsight {
    const id = nanoid();
    const now = new Date().toISOString();

    const stmt = this.db.prepare(`
      INSERT INTO shadow_insights (
        id, problem_id, insight, source, confidence, related_node_ids, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      id,
      problemId,
      insight,
      options.source || 'keyword_match',
      options.confidence || 0.5,
      JSON.stringify(options.relatedNodeIds || []),
      now
    );

    // Update problem attempt count
    this.db.prepare(`
      UPDATE unsolved_problems
      SET attempts = attempts + 1,
          last_attempt_at = ?,
          status = 'investigating',
          updated_at = ?
      WHERE id = ?
    `).run(now, now, problemId);

    return {
      id,
      problemId,
      insight,
      source: options.source || 'keyword_match',
      confidence: options.confidence || 0.5,
      relatedNodeIds: options.relatedNodeIds || [],
      createdAt: new Date(now),
    };
  }

  /**
   * Get insights for a problem
   */
  getInsights(problemId: string): ShadowInsight[] {
    const stmt = this.db.prepare(`
      SELECT * FROM shadow_insights
      WHERE problem_id = ?
      ORDER BY created_at DESC
    `);
    const rows = stmt.all(problemId) as Record<string, unknown>[];

    return rows.map(row => ({
      id: row['id'] as string,
      problemId: row['problem_id'] as string,
      insight: row['insight'] as string,
      source: row['source'] as ShadowInsight['source'],
      confidence: row['confidence'] as number,
      relatedNodeIds: JSON.parse(row['related_node_ids'] as string || '[]'),
      createdAt: new Date(row['created_at'] as string),
    }));
  }

  /**
   * Get problems that haven't been worked on recently
   */
  getStaleProblems(hoursSinceLastAttempt: number = 24): UnsolvedProblem[] {
    const cutoff = new Date(Date.now() - hoursSinceLastAttempt * 60 * 60 * 1000);

    const stmt = this.db.prepare(`
      SELECT * FROM unsolved_problems
      WHERE status IN ('open', 'investigating')
        AND (last_attempt_at IS NULL OR last_attempt_at < ?)
      ORDER BY priority DESC
    `);
    const rows = stmt.all(cutoff.toISOString()) as Record<string, unknown>[];
    return rows.map(row => this.rowToProblem(row));
  }

  /**
   * Get statistics
   */
  getStats(): {
    total: number;
    open: number;
    investigating: number;
    solved: number;
    abandoned: number;
    totalInsights: number;
  } {
    const statusCounts = this.db.prepare(`
      SELECT status, COUNT(*) as count FROM unsolved_problems GROUP BY status
    `).all() as { status: string; count: number }[];

    const insightCount = this.db.prepare(`
      SELECT COUNT(*) as count FROM shadow_insights
    `).get() as { count: number };

    const stats = {
      total: 0,
      open: 0,
      investigating: 0,
      solved: 0,
      abandoned: 0,
      totalInsights: insightCount.count,
    };

    for (const { status, count } of statusCounts) {
      stats.total += count;
      if (status === 'open') stats.open = count;
      if (status === 'investigating') stats.investigating = count;
      if (status === 'solved') stats.solved = count;
      if (status === 'abandoned') stats.abandoned = count;
    }

    return stats;
  }

  private rowToProblem(row: Record<string, unknown>): UnsolvedProblem {
    return {
      id: row['id'] as string,
      description: row['description'] as string,
      context: row['context'] as string,
      tags: JSON.parse(row['tags'] as string || '[]'),
      status: row['status'] as UnsolvedProblem['status'],
      priority: row['priority'] as number,
      attempts: row['attempts'] as number,
      lastAttemptAt: row['last_attempt_at'] ? new Date(row['last_attempt_at'] as string) : null,
      createdAt: new Date(row['created_at'] as string),
      updatedAt: new Date(row['updated_at'] as string),
      solution: row['solution'] as string | null,
      relatedNodeIds: JSON.parse(row['related_node_ids'] as string || '[]'),
    };
  }

  close(): void {
    this.db.close();
  }
}

// ============================================================================
// SHADOW WORK - Background processing
// ============================================================================

import { VestigeDatabase } from './database.js';

/**
 * Run Shadow work cycle - look for new insights on unsolved problems
 */
export function runShadowCycle(shadow: ShadowSelf, vestige: VestigeDatabase): {
  problemsAnalyzed: number;
  insightsGenerated: number;
  insights: Array<{ problem: string; insight: string }>;
} {
  const result = {
    problemsAnalyzed: 0,
    insightsGenerated: 0,
    insights: [] as Array<{ problem: string; insight: string }>,
  };

  // Get stale problems that need attention
  const problems = shadow.getStaleProblems(1); // Haven't been worked on in 1 hour

  for (const problem of problems) {
    result.problemsAnalyzed++;

    // Extract keywords from problem description
    const keywords = problem.description
      .toLowerCase()
      .split(/\W+/)
      .filter(w => w.length > 4);

    // Search knowledge base for related content
    for (const keyword of keywords.slice(0, 5)) {
      try {
        const searchResult = vestige.searchNodes(keyword, { limit: 3 });

        for (const node of searchResult.items) {
          // Check if this node was added after the problem
          if (node.createdAt > problem.createdAt) {
            // New knowledge! This might help
            shadow.addInsight(problem.id, `New knowledge found: "${node.content.slice(0, 100)}..."`, {
              source: 'new_knowledge',
              confidence: 0.6,
              relatedNodeIds: [node.id],
            });

            result.insightsGenerated++;
            result.insights.push({
              problem: problem.description.slice(0, 50),
              insight: `Found related: ${node.content.slice(0, 50)}...`,
            });
          }
        }
      } catch {
        // Ignore search errors
      }
    }
  }

  return result;
}

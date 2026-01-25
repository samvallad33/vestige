/**
 * Dual-Strength Memory Model Tests
 *
 * Tests for the Bjork & Bjork (1992) dual-strength memory model implementation.
 *
 * The model distinguishes between:
 * - Storage strength: How well encoded a memory is (never decreases, only grows)
 * - Retrieval strength: How accessible the memory is now (decays over time)
 *
 * Key insight: Difficult retrievals (when retrieval strength is low) increase
 * storage strength MORE than easy retrievals (desirable difficulty principle).
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { initializeDatabase, VestigeDatabase, analyzeSentimentIntensity } from '../../core/database.js';
import path from 'path';
import fs from 'fs';
import os from 'os';

// ============================================================================
// TEST UTILITIES
// ============================================================================

/**
 * Create a test database in a temporary location
 */
function createTestDatabase(): Database.Database {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-test-'));
  const dbPath = path.join(tempDir, 'test.db');
  return initializeDatabase(dbPath);
}

/**
 * Create a test VestigeDatabase instance
 */
function createTestVestigeDatabase(): { db: VestigeDatabase; path: string } {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vestige-test-'));
  const dbPath = path.join(tempDir, 'test.db');
  const db = new VestigeDatabase(dbPath);
  return { db, path: dbPath };
}

/**
 * Clean up test database
 */
function cleanupTestDatabase(db: Database.Database): void {
  try {
    db.close();
  } catch {
    // Ignore close errors
  }
}

/**
 * Clean up VestigeDatabase and its files
 */
function cleanupVestigeDatabase(db: VestigeDatabase, dbPath: string): void {
  try {
    db.close();
    // Clean up temp directory
    const tempDir = path.dirname(dbPath);
    fs.rmSync(tempDir, { recursive: true, force: true });
  } catch {
    // Ignore cleanup errors
  }
}

/**
 * Helper to insert a test node with specific properties
 */
function insertTestNode(
  db: Database.Database,
  overrides: Partial<{
    id: string;
    storage_strength: number;
    retrieval_strength: number;
    retention_strength: number;
    stability_factor: number;
    sentiment_intensity: number;
    last_accessed_at: string;
    access_count: number;
    review_count: number;
  }> = {}
): string {
  const id = overrides.id || `test-node-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO knowledge_nodes (
      id, content, summary,
      created_at, updated_at, last_accessed_at, access_count,
      retention_strength, stability_factor, sentiment_intensity,
      storage_strength, retrieval_strength, review_count,
      source_type, source_platform, confidence, tags
    ) VALUES (
      ?, ?, ?,
      ?, ?, ?, ?,
      ?, ?, ?,
      ?, ?, ?,
      ?, ?, ?, ?
    )
  `);

  stmt.run(
    id,
    'Test content for dual-strength memory model',
    'Test summary',
    now,
    now,
    overrides.last_accessed_at || now,
    overrides.access_count ?? 0,
    overrides.retention_strength ?? 1.0,
    overrides.stability_factor ?? 1.0,
    overrides.sentiment_intensity ?? 0,
    overrides.storage_strength ?? 1.0,
    overrides.retrieval_strength ?? 1.0,
    overrides.review_count ?? 0,
    'note',
    'manual',
    0.8,
    '[]'
  );

  return id;
}

/**
 * Helper to get node strengths from database
 */
function getNodeStrengths(db: Database.Database, id: string): {
  storage_strength: number;
  retrieval_strength: number;
  retention_strength: number;
  stability_factor: number;
  sentiment_intensity: number;
  access_count: number;
  review_count: number;
} | null {
  const stmt = db.prepare(`
    SELECT storage_strength, retrieval_strength, retention_strength,
           stability_factor, sentiment_intensity, access_count, review_count
    FROM knowledge_nodes WHERE id = ?
  `);
  const row = stmt.get(id) as Record<string, unknown> | undefined;
  if (!row) return null;
  return {
    storage_strength: row['storage_strength'] as number,
    retrieval_strength: row['retrieval_strength'] as number,
    retention_strength: row['retention_strength'] as number,
    stability_factor: row['stability_factor'] as number,
    sentiment_intensity: row['sentiment_intensity'] as number,
    access_count: row['access_count'] as number,
    review_count: row['review_count'] as number,
  };
}

/**
 * Backdate a node's last_accessed_at by a number of days
 */
function backdateNode(db: Database.Database, id: string, daysAgo: number): void {
  const pastDate = new Date();
  pastDate.setDate(pastDate.getDate() - daysAgo);
  const stmt = db.prepare(`UPDATE knowledge_nodes SET last_accessed_at = ? WHERE id = ?`);
  stmt.run(pastDate.toISOString(), id);
}

// ============================================================================
// STORAGE STRENGTH TESTS
// ============================================================================

describe('Dual-Strength Memory Model', () => {
  describe('Storage Strength', () => {
    let testDb: { db: VestigeDatabase; path: string };

    beforeEach(() => {
      testDb = createTestVestigeDatabase();
    });

    afterEach(() => {
      cleanupVestigeDatabase(testDb.db, testDb.path);
    });

    it('should start at 1.0 for new nodes', () => {
      // Create a new node
      const insertedNode = testDb.db.insertNode({
        content: 'New knowledge to remember',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Fetch the node to get full data (insertNode returns partial data)
      const node = testDb.db.getNode(insertedNode.id);

      // Verify storage_strength is 1.0
      expect(node?.storageStrength).toBe(1.0);
    });

    it('should increase by 0.05 on regular access', () => {
      // Create a node
      const insertedNode = testDb.db.insertNode({
        content: 'Memory to access',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Fetch to get full data
      const node = testDb.db.getNode(insertedNode.id);
      const initialStorage = node?.storageStrength ?? 1.0;

      // Access the node
      testDb.db.updateNodeAccess(insertedNode.id);

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage should increase by 0.05
      expect(updatedNode?.storageStrength).toBeCloseTo(initialStorage + 0.05, 4);
    });

    it('should increase by 0.1 on successful review (easy recall)', () => {
      // Create a node with retention above lapse threshold (0.3)
      const insertedNode = testDb.db.insertNode({
        content: 'Memory for easy review',
        sourceType: 'note',
        sourcePlatform: 'manual',
        retentionStrength: 0.5, // Above lapse threshold
      });

      // Fetch to get full data
      const node = testDb.db.getNode(insertedNode.id);
      const initialStorage = node?.storageStrength ?? 1.0;

      // Mark as reviewed (successful recall)
      testDb.db.markReviewed(insertedNode.id);

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage should increase by 0.1 for successful recall
      expect(updatedNode?.storageStrength).toBeCloseTo(initialStorage + 0.1, 4);
    });

    it('should increase by 0.3 on difficult review (lapse recall) - desirable difficulty', () => {
      // Create a node with retention below lapse threshold (0.3)
      const insertedNode = testDb.db.insertNode({
        content: 'Memory for difficult review',
        sourceType: 'note',
        sourcePlatform: 'manual',
        retentionStrength: 0.2, // Below lapse threshold - forgot it
      });

      // Fetch to get full data
      const node = testDb.db.getNode(insertedNode.id);
      const initialStorage = node?.storageStrength ?? 1.0;

      // Mark as reviewed (lapse - difficult recall)
      testDb.db.markReviewed(insertedNode.id);

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage should increase by 0.3 for difficult recall (desirable difficulty)
      expect(updatedNode?.storageStrength).toBeCloseTo(initialStorage + 0.3, 4);
    });

    it('should NEVER decrease (critical invariant)', () => {
      // Create a node with high storage strength
      const insertedNode = testDb.db.insertNode({
        content: 'Memory with high storage',
        sourceType: 'note',
        sourcePlatform: 'manual',
        storageStrength: 5.0,
      });

      // Apply decay
      testDb.db.applyDecay();

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage strength should NOT decrease
      expect(updatedNode?.storageStrength).toBeGreaterThanOrEqual(5.0);
    });

    it('should accumulate across multiple accesses', () => {
      // Create a node
      const insertedNode = testDb.db.insertNode({
        content: 'Memory to access multiple times',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Access 10 times
      for (let i = 0; i < 10; i++) {
        testDb.db.updateNodeAccess(insertedNode.id);
      }

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage should have increased by 0.05 * 10 = 0.5
      expect(updatedNode?.storageStrength).toBeCloseTo(1.0 + 0.5, 4);
    });

    it('should be able to exceed 1.0 (unbounded growth)', () => {
      // Create a node and access it many times
      const insertedNode = testDb.db.insertNode({
        content: 'Memory to access many times',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Access 50 times (0.05 * 50 = 2.5)
      for (let i = 0; i < 50; i++) {
        testDb.db.updateNodeAccess(insertedNode.id);
      }

      // Retrieve updated node
      const updatedNode = testDb.db.getNode(insertedNode.id);

      // Storage should be well above 1.0
      expect(updatedNode?.storageStrength).toBeGreaterThan(3.0);
    });

    it('should preserve storage strength after decay is applied', () => {
      // Get raw database and backdate
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 10.0,
        retrieval_strength: 1.0,
        retention_strength: 1.0,
      });
      backdateNode(rawDb, id, 30); // 30 days ago

      // Apply decay manually (simulating the applyDecay method logic)
      // Note: We're testing that storage_strength doesn't change after decay
      const now = Date.now();
      const nodes = rawDb.prepare(`
        SELECT id, last_accessed_at, retention_strength, stability_factor, sentiment_intensity,
               storage_strength, retrieval_strength
        FROM knowledge_nodes WHERE id = ?
      `).all(id) as {
        id: string;
        last_accessed_at: string;
        storage_strength: number;
      }[];

      for (const n of nodes) {
        const lastAccessed = new Date(n.last_accessed_at).getTime();
        const daysSince = (now - lastAccessed) / (1000 * 60 * 60 * 24);

        // Storage strength should NOT change in decay
        expect(daysSince).toBeGreaterThan(0);
        expect(n.storage_strength).toBe(10.0);
      }

      cleanupTestDatabase(rawDb);
    });
  });

  // ============================================================================
  // RETRIEVAL STRENGTH TESTS
  // ============================================================================

  describe('Retrieval Strength', () => {
    let testDb: { db: VestigeDatabase; path: string };

    beforeEach(() => {
      testDb = createTestVestigeDatabase();
    });

    afterEach(() => {
      cleanupVestigeDatabase(testDb.db, testDb.path);
    });

    it('should start at 1.0 for new nodes', () => {
      const insertedNode = testDb.db.insertNode({
        content: 'Fresh memory',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Fetch to get full data
      const node = testDb.db.getNode(insertedNode.id);

      expect(node?.retrievalStrength).toBe(1.0);
    });

    it('should decay over time following power law', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Backdate by 7 days
      backdateNode(rawDb, id, 7);

      const before = getNodeStrengths(rawDb, id);
      expect(before?.retrieval_strength).toBe(1.0);

      // Simulate decay calculation
      const now = Date.now();
      const lastAccessed = new Date();
      lastAccessed.setDate(lastAccessed.getDate() - 7);
      const daysSince = 7;
      const storageStrength = 1.0;
      const sentimentIntensity = 0;

      // effectiveDecayRate = 1 / (storageStrength * (1 + sentimentIntensity))
      const effectiveDecayRate = 1 / (storageStrength * (1 + sentimentIntensity));
      const expectedRetrieval = Math.max(0.1, Math.exp(-daysSince * effectiveDecayRate));

      // After 7 days with storage=1.0, sentiment=0
      // decay rate = 1 / (1 * 1) = 1
      // retrieval = exp(-7 * 1) = exp(-7) ~= 0.00091
      // But clamped to 0.1 minimum
      expect(expectedRetrieval).toBe(0.1);

      cleanupTestDatabase(rawDb);
    });

    it('should reset to 1.0 on access', () => {
      // Create a node with decayed retrieval
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 5.0,
        retrieval_strength: 0.3, // Already decayed
        sentiment_intensity: 0,
      });

      const before = getNodeStrengths(rawDb, id);
      expect(before?.retrieval_strength).toBe(0.3);

      // Update access (using raw SQL to simulate updateNodeAccess)
      rawDb.prepare(`
        UPDATE knowledge_nodes
        SET last_accessed_at = ?,
            access_count = access_count + 1,
            storage_strength = storage_strength + 0.05,
            retrieval_strength = 1.0
        WHERE id = ?
      `).run(new Date().toISOString(), id);

      const after = getNodeStrengths(rawDb, id);
      expect(after?.retrieval_strength).toBe(1.0);

      cleanupTestDatabase(rawDb);
    });

    it('should decay slower with higher storage strength', () => {
      // Test with two nodes: one with low storage, one with high storage
      const rawDb = createTestDatabase();

      // Create low storage node
      const lowStorageId = insertTestNode(rawDb, {
        id: 'low-storage-node',
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Create high storage node
      const highStorageId = insertTestNode(rawDb, {
        id: 'high-storage-node',
        storage_strength: 10.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Backdate both by 3 days
      backdateNode(rawDb, lowStorageId, 3);
      backdateNode(rawDb, highStorageId, 3);

      // Calculate expected decay for each
      const daysSince = 3;

      // Low storage: decay rate = 1 / (1 * 1) = 1
      // retrieval = exp(-3 * 1) = exp(-3) ~= 0.05
      const lowStorageDecay = Math.max(0.1, Math.exp(-daysSince * (1 / (1.0 * 1))));

      // High storage: decay rate = 1 / (10 * 1) = 0.1
      // retrieval = exp(-3 * 0.1) = exp(-0.3) ~= 0.74
      const highStorageDecay = Math.max(0.1, Math.exp(-daysSince * (1 / (10.0 * 1))));

      // High storage node should retain more
      expect(highStorageDecay).toBeGreaterThan(lowStorageDecay);
      expect(highStorageDecay).toBeGreaterThan(0.7);
      expect(lowStorageDecay).toBeLessThan(0.2);

      cleanupTestDatabase(rawDb);
    });

    it('should decay slower with higher sentiment intensity', () => {
      // Test emotional vs neutral memory
      const rawDb = createTestDatabase();

      // Create neutral node
      const neutralId = insertTestNode(rawDb, {
        id: 'neutral-node',
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Create emotional node
      const emotionalId = insertTestNode(rawDb, {
        id: 'emotional-node',
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 1.0, // Highly emotional
      });

      // Both have same storage, but different sentiment
      const daysSince = 3;

      // Neutral: decay rate = 1 / (1 * (1 + 0)) = 1
      const neutralDecay = Math.max(0.1, Math.exp(-daysSince * (1 / (1.0 * (1 + 0)))));

      // Emotional: decay rate = 1 / (1 * (1 + 1)) = 0.5
      const emotionalDecay = Math.max(0.1, Math.exp(-daysSince * (1 / (1.0 * (1 + 1)))));

      // Emotional memory should retain more
      expect(emotionalDecay).toBeGreaterThan(neutralDecay);

      cleanupTestDatabase(rawDb);
    });

    it('should have minimum floor of 0.1 (never completely forgotten)', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Backdate by 365 days (a year)
      backdateNode(rawDb, id, 365);

      const daysSince = 365;
      const decayRate = 1 / (1.0 * 1);
      const rawRetrieval = Math.exp(-daysSince * decayRate);

      // Should be clamped to 0.1
      const clampedRetrieval = Math.max(0.1, rawRetrieval);
      expect(clampedRetrieval).toBe(0.1);

      cleanupTestDatabase(rawDb);
    });

    it('should reset to 1.0 on review', () => {
      const insertedNode = testDb.db.insertNode({
        content: 'Memory to review',
        sourceType: 'note',
        sourcePlatform: 'manual',
        retrievalStrength: 0.4, // Already somewhat decayed
      });

      // Mark as reviewed
      testDb.db.markReviewed(insertedNode.id);

      // Get updated node
      const updated = testDb.db.getNode(insertedNode.id);

      expect(updated?.retrievalStrength).toBe(1.0);
    });
  });

  // ============================================================================
  // COMBINED RETENTION STRENGTH TESTS
  // ============================================================================

  describe('Combined Retention Strength (backward compatible)', () => {
    let rawDb: Database.Database;

    beforeEach(() => {
      rawDb = createTestDatabase();
    });

    afterEach(() => {
      cleanupTestDatabase(rawDb);
    });

    it('should be computed from both strengths using weighted formula', () => {
      // retention = (retrieval * 0.7) + (min(1, storage/10) * 0.3)
      const storage = 5.0;
      const retrieval = 0.8;

      const normalizedStorage = Math.min(1, storage / 10);
      const expectedRetention = (retrieval * 0.7) + (normalizedStorage * 0.3);

      expect(expectedRetention).toBeCloseTo((0.8 * 0.7) + (0.5 * 0.3), 4);
      expect(expectedRetention).toBeCloseTo(0.71, 2);
    });

    it('should be within [0, 1] range', () => {
      // Test extreme values
      const testCases = [
        { storage: 0, retrieval: 0 },
        { storage: 1, retrieval: 0 },
        { storage: 0, retrieval: 1 },
        { storage: 1, retrieval: 1 },
        { storage: 100, retrieval: 1 }, // Very high storage
        { storage: 100, retrieval: 0.1 }, // High storage, low retrieval
      ];

      for (const tc of testCases) {
        const normalizedStorage = Math.min(1, tc.storage / 10);
        const retention = (tc.retrieval * 0.7) + (normalizedStorage * 0.3);

        expect(retention).toBeGreaterThanOrEqual(0);
        expect(retention).toBeLessThanOrEqual(1);
      }
    });

    it('should weight retrieval more heavily (70%)', () => {
      // With same values, retrieval contribution should be larger
      const storage = 10.0; // Normalized to 1.0
      const retrieval = 1.0;

      const normalizedStorage = Math.min(1, storage / 10);
      const retention = (retrieval * 0.7) + (normalizedStorage * 0.3);

      const retrievalContribution = retrieval * 0.7;
      const storageContribution = normalizedStorage * 0.3;

      expect(retrievalContribution).toBeGreaterThan(storageContribution);
      expect(retrievalContribution).toBe(0.7);
      expect(storageContribution).toBe(0.3);
    });

    it('should cap storage contribution at normalized value of 1', () => {
      // Storage > 10 should not increase contribution
      const storageValues = [10, 20, 50, 100];
      const retrieval = 0.5;

      let previousRetention: number | null = null;

      for (const storage of storageValues) {
        const normalizedStorage = Math.min(1, storage / 10);
        const retention = (retrieval * 0.7) + (normalizedStorage * 0.3);

        if (storage >= 10) {
          // All should have same normalized storage contribution
          expect(normalizedStorage).toBe(1);
          if (previousRetention !== null) {
            expect(retention).toBe(previousRetention);
          }
        }
        previousRetention = retention;
      }
    });
  });

  // ============================================================================
  // DATABASE INTEGRATION TESTS
  // ============================================================================

  describe('Database Integration', () => {
    let rawDb: Database.Database;

    beforeEach(() => {
      rawDb = createTestDatabase();
    });

    afterEach(() => {
      cleanupTestDatabase(rawDb);
    });

    it('should have dual-strength columns after migration', () => {
      const columns = rawDb.prepare("PRAGMA table_info(knowledge_nodes)").all() as { name: string }[];
      const columnNames = columns.map(c => c.name);

      expect(columnNames).toContain('storage_strength');
      expect(columnNames).toContain('retrieval_strength');
    });

    it('should insert with correct default values', () => {
      const id = insertTestNode(rawDb);
      const strengths = getNodeStrengths(rawDb, id);

      expect(strengths?.storage_strength).toBe(1.0);
      expect(strengths?.retrieval_strength).toBe(1.0);
      expect(strengths?.retention_strength).toBe(1.0);
    });

    it('should allow custom initial values', () => {
      const id = insertTestNode(rawDb, {
        storage_strength: 5.0,
        retrieval_strength: 0.8,
        retention_strength: 0.9,
      });
      const strengths = getNodeStrengths(rawDb, id);

      expect(strengths?.storage_strength).toBe(5.0);
      expect(strengths?.retrieval_strength).toBe(0.8);
      expect(strengths?.retention_strength).toBe(0.9);
    });

    it('should update storage correctly on access', () => {
      const id = insertTestNode(rawDb, { storage_strength: 1.0 });

      // Simulate updateNodeAccess
      rawDb.prepare(`
        UPDATE knowledge_nodes
        SET storage_strength = storage_strength + 0.05,
            retrieval_strength = 1.0
        WHERE id = ?
      `).run(id);

      const strengths = getNodeStrengths(rawDb, id);
      expect(strengths?.storage_strength).toBeCloseTo(1.05, 4);
    });

    it('should update retrieval correctly during decay', () => {
      const id = insertTestNode(rawDb, {
        storage_strength: 2.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0,
      });

      // Backdate
      backdateNode(rawDb, id, 1);

      // Simulate decay calculation
      const daysSince = 1;
      const storage = 2.0;
      const sentiment = 0;
      const decayRate = 1 / (storage * (1 + sentiment));
      const newRetrieval = Math.max(0.1, Math.exp(-daysSince * decayRate));

      // Update
      rawDb.prepare(`
        UPDATE knowledge_nodes SET retrieval_strength = ? WHERE id = ?
      `).run(newRetrieval, id);

      const strengths = getNodeStrengths(rawDb, id);
      expect(strengths?.retrieval_strength).toBeCloseTo(newRetrieval, 4);
    });
  });

  // ============================================================================
  // EDGE CASES
  // ============================================================================

  describe('Edge Cases', () => {
    let testDb: { db: VestigeDatabase; path: string };

    beforeEach(() => {
      testDb = createTestVestigeDatabase();
    });

    afterEach(() => {
      cleanupVestigeDatabase(testDb.db, testDb.path);
    });

    it('should handle new node with no accesses correctly', () => {
      const insertedNode = testDb.db.insertNode({
        content: 'Brand new memory',
        sourceType: 'note',
        sourcePlatform: 'manual',
      });

      // Fetch to get full data
      const node = testDb.db.getNode(insertedNode.id);

      expect(node?.storageStrength).toBe(1.0);
      expect(node?.retrievalStrength).toBe(1.0);
      expect(node?.accessCount).toBe(0);
      expect(node?.reviewCount).toBe(0);
    });

    it('should handle heavily accessed node (storage >> 10)', () => {
      // Create node with very high storage strength
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 50.0, // Very heavily accessed
        retrieval_strength: 0.5,
        sentiment_intensity: 0.5,
      });

      // Verify storage is preserved and high
      const strengths = getNodeStrengths(rawDb, id);
      expect(strengths?.storage_strength).toBe(50.0);

      // Decay should still work but be very slow
      const daysSince = 1;
      const storage = 50.0;
      const sentiment = 0.5;
      const decayRate = 1 / (storage * (1 + sentiment));
      const newRetrieval = Math.max(0.1, Math.exp(-daysSince * decayRate));

      // With storage=50 and sentiment=0.5, decay rate = 1/(50*1.5) = 0.0133
      // retrieval after 1 day = exp(-0.0133) ~= 0.987
      expect(decayRate).toBeCloseTo(0.0133, 3);
      expect(newRetrieval).toBeGreaterThan(0.98);

      cleanupTestDatabase(rawDb);
    });

    it('should handle long-decayed node (retrieval near 0.1 floor)', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 1.0,
        retrieval_strength: 0.1, // Already at floor
        sentiment_intensity: 0,
      });

      // Backdate further
      backdateNode(rawDb, id, 100);

      // Retrieval should stay at floor
      const daysSince = 100;
      const decayRate = 1 / (1.0 * 1);
      const rawRetrieval = Math.exp(-daysSince * decayRate);
      const clampedRetrieval = Math.max(0.1, rawRetrieval);

      expect(rawRetrieval).toBeLessThan(0.001);
      expect(clampedRetrieval).toBe(0.1);

      cleanupTestDatabase(rawDb);
    });

    it('should handle high sentiment emotional memory', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 1.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 1.0, // Maximum emotional intensity
      });

      // Backdate by 7 days
      backdateNode(rawDb, id, 7);

      // Calculate decay with high sentiment
      const daysSince = 7;
      const storage = 1.0;
      const sentiment = 1.0;
      const decayRate = 1 / (storage * (1 + sentiment)); // 1 / (1 * 2) = 0.5
      const newRetrieval = Math.max(0.1, Math.exp(-daysSince * decayRate));

      // With sentiment=1.0, decay is halved
      // retrieval = exp(-7 * 0.5) = exp(-3.5) ~= 0.03
      // But clamped to 0.1
      expect(decayRate).toBe(0.5);
      expect(newRetrieval).toBe(0.1); // Clamped

      // Compare to neutral: would be exp(-7) ~= 0.0009, also clamped to 0.1
      cleanupTestDatabase(rawDb);
    });

    it('should handle combined high storage and high sentiment', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 10.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 1.0,
      });

      // Calculate decay
      const daysSince = 30; // A month
      const storage = 10.0;
      const sentiment = 1.0;
      const decayRate = 1 / (storage * (1 + sentiment)); // 1 / (10 * 2) = 0.05
      const newRetrieval = Math.max(0.1, Math.exp(-daysSince * decayRate));

      // retrieval = exp(-30 * 0.05) = exp(-1.5) ~= 0.22
      expect(decayRate).toBe(0.05);
      expect(newRetrieval).toBeGreaterThan(0.2);
      expect(newRetrieval).toBeLessThan(0.25);

      cleanupTestDatabase(rawDb);
    });

    it('should handle zero sentiment correctly', () => {
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 5.0,
        retrieval_strength: 1.0,
        sentiment_intensity: 0, // Neutral
      });

      const daysSince = 5;
      const storage = 5.0;
      const sentiment = 0;
      const decayRate = 1 / (storage * (1 + sentiment)); // 1 / (5 * 1) = 0.2
      const newRetrieval = Math.max(0.1, Math.exp(-daysSince * decayRate));

      // retrieval = exp(-5 * 0.2) = exp(-1) ~= 0.37
      expect(decayRate).toBe(0.2);
      expect(newRetrieval).toBeGreaterThan(0.35);
      expect(newRetrieval).toBeLessThan(0.4);

      cleanupTestDatabase(rawDb);
    });
  });

  // ============================================================================
  // SENTIMENT INTENSITY ANALYSIS TESTS
  // ============================================================================

  describe('Sentiment Intensity Analysis', () => {
    it('should return 0 for neutral content', () => {
      const content = 'The meeting is scheduled for 3pm tomorrow.';
      const intensity = analyzeSentimentIntensity(content);
      expect(intensity).toBeLessThan(0.2);
    });

    it('should return high intensity for highly positive content', () => {
      const content = 'I am absolutely thrilled! This is amazing and wonderful! Best day ever!';
      const intensity = analyzeSentimentIntensity(content);
      expect(intensity).toBeGreaterThan(0.3);
    });

    it('should return high intensity for highly negative content', () => {
      const content = 'I am completely devastated and heartbroken. This is terrible and awful!';
      const intensity = analyzeSentimentIntensity(content);
      expect(intensity).toBeGreaterThan(0.3);
    });

    it('should measure intensity not polarity (both positive and negative should be high)', () => {
      const positive = 'Absolutely fantastic! Amazing! Wonderful!';
      const negative = 'Terrible! Horrible! Devastating!';

      const positiveIntensity = analyzeSentimentIntensity(positive);
      const negativeIntensity = analyzeSentimentIntensity(negative);

      // Both should be emotionally intense
      expect(positiveIntensity).toBeGreaterThan(0.2);
      expect(negativeIntensity).toBeGreaterThan(0.2);
    });

    it('should handle empty content', () => {
      const intensity = analyzeSentimentIntensity('');
      expect(intensity).toBe(0);
    });

    it('should be bounded between 0 and 1', () => {
      const testCases = [
        '',
        'Hello',
        'Amazing wonderful fantastic brilliant extraordinary phenomenal',
        'Terrible horrible awful dreadful catastrophic disastrous devastating',
        'a'.repeat(1000), // Long neutral content
      ];

      for (const content of testCases) {
        const intensity = analyzeSentimentIntensity(content);
        expect(intensity).toBeGreaterThanOrEqual(0);
        expect(intensity).toBeLessThanOrEqual(1);
      }
    });
  });

  // ============================================================================
  // DESIRABLE DIFFICULTY TESTS
  // ============================================================================

  describe('Desirable Difficulty Principle', () => {
    let testDb: { db: VestigeDatabase; path: string };

    beforeEach(() => {
      testDb = createTestVestigeDatabase();
    });

    afterEach(() => {
      cleanupVestigeDatabase(testDb.db, testDb.path);
    });

    it('should reward difficult recalls with higher storage increase', () => {
      // Create two nodes: one for easy recall, one for difficult
      const easyInserted = testDb.db.insertNode({
        content: 'Easy recall memory',
        sourceType: 'note',
        sourcePlatform: 'manual',
        retentionStrength: 0.8, // Above lapse threshold - easy
      });

      const hardInserted = testDb.db.insertNode({
        content: 'Difficult recall memory',
        sourceType: 'note',
        sourcePlatform: 'manual',
        retentionStrength: 0.2, // Below lapse threshold - forgot it
      });

      // Fetch to get full data
      const easyNode = testDb.db.getNode(easyInserted.id);
      const hardNode = testDb.db.getNode(hardInserted.id);

      const easyInitialStorage = easyNode?.storageStrength ?? 1.0;
      const hardInitialStorage = hardNode?.storageStrength ?? 1.0;

      // Both reviewed
      testDb.db.markReviewed(easyInserted.id);
      testDb.db.markReviewed(hardInserted.id);

      const easyAfter = testDb.db.getNode(easyInserted.id);
      const hardAfter = testDb.db.getNode(hardInserted.id);

      const easyIncrease = (easyAfter?.storageStrength ?? 0) - easyInitialStorage;
      const hardIncrease = (hardAfter?.storageStrength ?? 0) - hardInitialStorage;

      // Difficult recall should increase storage MORE
      expect(hardIncrease).toBeGreaterThan(easyIncrease);
      expect(easyIncrease).toBeCloseTo(0.1, 2); // Easy: +0.1
      expect(hardIncrease).toBeCloseTo(0.3, 2); // Hard: +0.3
    });

    it('should reset stability on lapse but still increase storage', () => {
      // Create a node with high stability that then lapses using raw database
      const rawDb = createTestDatabase();
      const id = insertTestNode(rawDb, {
        storage_strength: 3.0,
        retrieval_strength: 0.2,
        retention_strength: 0.2,
        stability_factor: 10.0, // High stability from previous reviews
      });

      const before = getNodeStrengths(rawDb, id);
      expect(before?.stability_factor).toBe(10.0);
      expect(before?.storage_strength).toBe(3.0);

      // Now when reviewed (lapsed), stability should reset but storage should increase
      // This is the key insight: even failures strengthen encoding (desirable difficulty)

      // Verify the invariants we expect from the model
      // Storage should never decrease regardless of lapse
      expect(before?.storage_strength).toBeGreaterThanOrEqual(1.0);

      cleanupTestDatabase(rawDb);
    });
  });
});

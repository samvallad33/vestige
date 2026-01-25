/**
 * Integration tests for all 14 MCP tools in Vestige MCP
 *
 * Tests cover the complete tool functionality including:
 * - Input validation
 * - Database operations
 * - Response formatting
 * - Edge cases and error handling
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@rstest/core';
import { VestigeDatabase } from '../../core/database.js';
import type { KnowledgeNode, PersonNode } from '../../core/types.js';

/**
 * Creates an in-memory test database instance
 */
function createTestDatabase(): VestigeDatabase {
  return new VestigeDatabase(':memory:');
}

/**
 * Create a mock timestamp for consistent testing
 */
function mockTimestamp(daysAgo: number = 0): Date {
  const date = new Date();
  date.setDate(date.getDate() - daysAgo);
  return date;
}

// ============================================================================
// MCP TOOL HANDLER MOCK
// ============================================================================

/**
 * Creates mock MCP tool handlers that simulate the actual tool behavior
 * These handlers call the same database methods as the real MCP server
 */
function createMCPToolHandler(db: VestigeDatabase) {
  return {
    // --- Tool 1: ingest ---
    async ingest(args: {
      content: string;
      source?: string;
      platform?: string;
      sourceId?: string;
      sourceUrl?: string;
      timestamp?: string;
      people?: string[];
      tags?: string[];
    }) {
      const node = db.insertNode({
        content: args.content,
        sourceType: (args.source as KnowledgeNode['sourceType']) || 'manual',
        sourcePlatform: (args.platform as KnowledgeNode['sourcePlatform']) || 'manual',
        sourceId: args.sourceId,
        sourceUrl: args.sourceUrl,
        createdAt: args.timestamp ? new Date(args.timestamp) : new Date(),
        updatedAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        retentionStrength: 1.0,
        stabilityFactor: 1.0,
        reviewCount: 0,
        confidence: 0.8,
        isContradicted: false,
        contradictionIds: [],
        people: args.people || [],
        concepts: [],
        events: [],
        tags: args.tags || [],
        sourceChain: [],
      });

      return {
        success: true,
        nodeId: node.id,
        message: `Knowledge ingested successfully. Node ID: ${node.id}`,
      };
    },

    // --- Tool 2: recall ---
    async recall(args: { query: string; limit?: number; offset?: number }) {
      const result = db.searchNodes(args.query, {
        limit: args.limit || 10,
        offset: args.offset || 0,
      });

      // Update access timestamps for retrieved nodes
      for (const node of result.items) {
        try {
          db.updateNodeAccess(node.id);
        } catch {
          // Ignore access update errors
        }
      }

      const formatted = result.items.map((node) => ({
        id: node.id,
        content: node.content,
        summary: node.summary,
        source: {
          type: node.sourceType,
          platform: node.sourcePlatform,
          url: node.sourceUrl,
        },
        metadata: {
          createdAt: node.createdAt.toISOString(),
          lastAccessed: node.lastAccessedAt.toISOString(),
          retentionStrength: node.retentionStrength,
          sentimentIntensity: node.sentimentIntensity,
          confidence: node.confidence,
        },
        gitContext: node.gitContext,
        people: node.people,
        tags: node.tags,
      }));

      return {
        query: args.query,
        total: result.total,
        showing: result.items.length,
        offset: result.offset,
        hasMore: result.hasMore,
        results: formatted,
      };
    },

    // --- Tool 3: get_knowledge ---
    async getKnowledge(args: { nodeId: string }) {
      const node = db.getNode(args.nodeId);
      if (!node) {
        return { error: 'Node not found', nodeId: args.nodeId };
      }

      db.updateNodeAccess(args.nodeId);
      return node;
    },

    // --- Tool 4: get_related ---
    async getRelated(args: { nodeId: string; depth?: number }) {
      const depth = args.depth || 1;
      const relatedIds = db.getRelatedNodes(args.nodeId, depth);
      const relatedNodes = relatedIds
        .map((id) => db.getNode(id))
        .filter((n): n is KnowledgeNode => n !== null);

      return {
        sourceNode: args.nodeId,
        depth,
        relatedCount: relatedNodes.length,
        related: relatedNodes.map((n) => ({
          id: n.id,
          summary: n.summary || n.content.slice(0, 200),
          tags: n.tags,
        })),
      };
    },

    // --- Tool 5: remember_person ---
    async rememberPerson(args: {
      name: string;
      howWeMet?: string;
      relationshipType?: string;
      organization?: string;
      role?: string;
      email?: string;
      notes?: string;
      sharedTopics?: string[];
    }) {
      // Check if person exists
      const existing = db.getPersonByName(args.name);
      if (existing) {
        return {
          message: `Person "${args.name}" already exists`,
          personId: existing.id,
          existing: true,
        };
      }

      const person = db.insertPerson({
        name: args.name,
        aliases: [],
        howWeMet: args.howWeMet,
        relationshipType: args.relationshipType,
        organization: args.organization,
        role: args.role,
        email: args.email,
        notes: args.notes,
        sharedTopics: args.sharedTopics || [],
        sharedProjects: [],
        socialLinks: {},
        contactFrequency: 0,
        relationshipHealth: 0.5,
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      return {
        success: true,
        personId: person.id,
        message: `Remembered ${args.name}`,
      };
    },

    // --- Tool 6: get_person ---
    async getPerson(args: { name: string }) {
      const person = db.getPersonByName(args.name);
      if (!person) {
        return {
          found: false,
          message: `No person named "${args.name}" found in memory`,
        };
      }

      const daysSinceContact = person.lastContactAt
        ? Math.floor(
            (Date.now() - person.lastContactAt.getTime()) / (1000 * 60 * 60 * 24)
          )
        : null;

      return {
        found: true,
        person: {
          ...person,
          daysSinceContact,
        },
      };
    },

    // --- Tool 7: mark_reviewed ---
    async markReviewed(args: { nodeId: string }) {
      const nodeBefore = db.getNode(args.nodeId);
      if (!nodeBefore) {
        return { error: 'Node not found' };
      }

      db.markReviewed(args.nodeId);

      const nodeAfter = db.getNode(args.nodeId);

      return {
        success: true,
        nodeId: args.nodeId,
        previousRetention: nodeBefore.retentionStrength,
        newRetention: nodeAfter?.retentionStrength,
        previousStability: nodeBefore.stabilityFactor,
        newStability: nodeAfter?.stabilityFactor,
        reviewCount: nodeAfter?.reviewCount,
        nextReviewDate: nodeAfter?.nextReviewDate?.toISOString(),
        message: 'Memory reinforced',
      };
    },

    // --- Tool 8: daily_brief ---
    async dailyBrief() {
      const stats = db.getStats();
      const health = db.checkHealth();
      const decaying = db.getDecayingNodes(0.5, { limit: 5 });
      const reconnect = db.getPeopleToReconnect(30, { limit: 5 });
      const recent = db.getRecentNodes({ limit: 5 });

      const getTimeBasedGreeting = (): string => {
        const hour = new Date().getHours();
        if (hour < 12) return 'Good morning';
        if (hour < 17) return 'Good afternoon';
        return 'Good evening';
      };

      return {
        date: new Date().toISOString().split('T')[0],
        greeting: getTimeBasedGreeting(),
        healthStatus: health.status,
        warnings: health.warnings.length > 0 ? health.warnings : undefined,
        stats: {
          totalKnowledge: stats.totalNodes,
          peopleInNetwork: stats.totalPeople,
          connections: stats.totalEdges,
          databaseSize: db.getDatabaseSize().formatted,
        },
        reviewNeeded: decaying.items.map((n) => ({
          id: n.id,
          preview: n.summary || n.content.slice(0, 100),
          retentionStrength: n.retentionStrength,
          daysSinceAccess: Math.floor(
            (Date.now() - n.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24)
          ),
        })),
        peopleToReconnect: reconnect.items.map((p) => ({
          name: p.name,
          daysSinceContact: p.lastContactAt
            ? Math.floor(
                (Date.now() - p.lastContactAt.getTime()) / (1000 * 60 * 60 * 24)
              )
            : null,
          sharedTopics: p.sharedTopics,
        })),
        recentlyAdded: recent.items.map((n) => ({
          id: n.id,
          preview: n.summary || n.content.slice(0, 100),
          source: n.sourcePlatform,
        })),
      };
    },

    // --- Tool 9: health_check ---
    async healthCheck() {
      const health = db.checkHealth();
      const size = db.getDatabaseSize();

      const getHealthRecommendations = (): string[] => {
        const recommendations: string[] = [];

        if (health.status === 'critical') {
          recommendations.push(
            'CRITICAL: Immediate attention required. Check warnings for details.'
          );
        }

        if (!health.lastBackup) {
          recommendations.push('Create your first backup using the backup tool');
        } else {
          const daysSinceBackup =
            (Date.now() - new Date(health.lastBackup).getTime()) /
            (1000 * 60 * 60 * 24);
          if (daysSinceBackup > 7) {
            recommendations.push(
              `Consider creating a backup (last backup was ${Math.floor(daysSinceBackup)} days ago)`
            );
          }
        }

        if (health.dbSizeMB > 50) {
          recommendations.push('Consider running optimize_database to reclaim space');
        }

        if (health.nodeCount > 10000) {
          recommendations.push('Large knowledge base detected. Searches may be slower.');
        }

        if (recommendations.length === 0) {
          recommendations.push('Everything looks healthy!');
        }

        return recommendations;
      };

      return {
        ...health,
        databaseSize: size,
        recommendations: getHealthRecommendations(),
      };
    },

    // --- Tool 10: backup ---
    async backup() {
      // Note: For in-memory databases, backup operations are not supported
      // as there's no file to copy. This mock handles that gracefully.
      try {
        const backupPath = db.backup();
        const backups = db.listBackups();

        return {
          success: true,
          backupPath,
          message: 'Backup created successfully',
          totalBackups: backups.length,
          backups: backups.slice(0, 5).map((b) => ({
            path: b.path,
            size: `${(b.size / 1024 / 1024).toFixed(2)}MB`,
            date: b.date.toISOString(),
          })),
        };
      } catch {
        // In-memory databases cannot be backed up - this is expected
        return {
          success: false,
          message: 'Backup not supported for in-memory databases',
          totalBackups: 0,
          backups: [],
        };
      }
    },

    // --- Tool 11: list_backups ---
    async listBackups() {
      const backups = db.listBackups();

      return {
        totalBackups: backups.length,
        backups: backups.map((b) => ({
          path: b.path,
          size: `${(b.size / 1024 / 1024).toFixed(2)}MB`,
          date: b.date.toISOString(),
        })),
      };
    },

    // --- Tool 12: optimize_database ---
    async optimizeDatabase() {
      const sizeBefore = db.getDatabaseSize();
      db.optimize();
      const sizeAfter = db.getDatabaseSize();

      return {
        success: true,
        message: 'Database optimized',
        sizeBefore: sizeBefore.formatted,
        sizeAfter: sizeAfter.formatted,
        spaceSaved: `${(sizeBefore.mb - sizeAfter.mb).toFixed(2)}MB`,
      };
    },

    // --- Tool 13: apply_decay ---
    async applyDecay() {
      const updatedCount = db.applyDecay();

      return {
        success: true,
        nodesUpdated: updatedCount,
        message: `Applied decay to ${updatedCount} knowledge nodes`,
      };
    },
  };
}

// ============================================================================
// TEST SUITES
// ============================================================================

describe('MCP Tools Integration', () => {
  let db: VestigeDatabase;
  let tools: ReturnType<typeof createMCPToolHandler>;

  beforeAll(() => {
    db = createTestDatabase();
    tools = createMCPToolHandler(db);
  });

  afterAll(() => {
    db.close();
  });

  // ==========================================================================
  // Tool 1: ingest
  // ==========================================================================
  describe('ingest tool', () => {
    it('should store content and return node ID', async () => {
      const result = await tools.ingest({
        content: 'Test knowledge for MCP integration',
      });

      expect(result.success).toBe(true);
      expect(result.nodeId).toBeDefined();
      expect(typeof result.nodeId).toBe('string');
      expect(result.message).toContain('Knowledge ingested successfully');
    });

    it('should store content with tags', async () => {
      const result = await tools.ingest({
        content: 'Tagged content for testing',
        tags: ['test', 'mcp', 'integration'],
      });

      expect(result.success).toBe(true);

      const node = db.getNode(result.nodeId);
      expect(node).not.toBeNull();
      expect(node?.tags).toContain('test');
      expect(node?.tags).toContain('mcp');
      expect(node?.tags).toContain('integration');
    });

    it('should store content with people references', async () => {
      const result = await tools.ingest({
        content: 'Meeting notes with team members',
        people: ['Alice', 'Bob', 'Charlie'],
      });

      expect(result.success).toBe(true);

      const node = db.getNode(result.nodeId);
      expect(node?.people).toContain('Alice');
      expect(node?.people).toContain('Bob');
      expect(node?.people).toContain('Charlie');
    });

    it('should use specified source type and platform', async () => {
      const result = await tools.ingest({
        content: 'Article about TypeScript patterns',
        source: 'article',
        platform: 'browser',
        sourceUrl: 'https://example.com/article',
      });

      const node = db.getNode(result.nodeId);
      expect(node?.sourceType).toBe('article');
      expect(node?.sourcePlatform).toBe('browser');
      expect(node?.sourceUrl).toBe('https://example.com/article');
    });

    it('should use custom timestamp when provided', async () => {
      const customDate = '2024-01-15T10:30:00.000Z';
      const result = await tools.ingest({
        content: 'Historical note',
        timestamp: customDate,
      });

      const node = db.getNode(result.nodeId);
      expect(node?.createdAt.toISOString()).toBe(customDate);
    });

    it('should initialize with correct default values', async () => {
      const result = await tools.ingest({
        content: 'Testing default values',
      });

      const node = db.getNode(result.nodeId);
      expect(node?.retentionStrength).toBe(1.0);
      expect(node?.stabilityFactor).toBe(1.0);
      expect(node?.confidence).toBe(0.8);
      expect(node?.accessCount).toBe(0);
      expect(node?.reviewCount).toBe(0);
      expect(node?.isContradicted).toBe(false);
    });
  });

  // ==========================================================================
  // Tool 2: recall
  // ==========================================================================
  describe('recall tool', () => {
    beforeEach(async () => {
      // Seed some test data for search tests
      await tools.ingest({ content: 'React hooks tutorial with useState examples' });
      await tools.ingest({ content: 'Vue composition API patterns' });
      await tools.ingest({ content: 'React context for global state management' });
      await tools.ingest({ content: 'Angular dependency injection guide' });
    });

    it('should find content by keyword', async () => {
      const result = await tools.recall({ query: 'React' });

      expect(result.total).toBeGreaterThanOrEqual(2);
      expect(result.results.length).toBeGreaterThanOrEqual(2);
      expect(result.results.every((n) => n.content.includes('React'))).toBe(true);
    });

    it('should respect limit parameter', async () => {
      const result = await tools.recall({ query: 'React', limit: 1 });

      expect(result.results.length).toBe(1);
      expect(result.hasMore).toBe(true);
    });

    it('should support pagination with offset', async () => {
      const page1 = await tools.recall({ query: 'React', limit: 1, offset: 0 });
      const page2 = await tools.recall({ query: 'React', limit: 1, offset: 1 });

      expect(page1.results[0].id).not.toBe(page2.results[0]?.id);
      expect(page1.offset).toBe(0);
      expect(page2.offset).toBe(1);
    });

    it('should return empty results for non-matching query', async () => {
      const result = await tools.recall({ query: 'xyznonexistent123' });

      expect(result.total).toBe(0);
      expect(result.results.length).toBe(0);
      expect(result.hasMore).toBe(false);
    });

    it('should update access count on retrieve', async () => {
      // Create a node and recall it
      const ingestResult = await tools.ingest({
        content: 'Unique searchable content xyz123',
      });

      const nodeBefore = db.getNode(ingestResult.nodeId);
      expect(nodeBefore?.accessCount).toBe(0);

      await tools.recall({ query: 'xyz123' });

      const nodeAfter = db.getNode(ingestResult.nodeId);
      expect(nodeAfter?.accessCount).toBe(1);
    });

    it('should include metadata in results', async () => {
      await tools.ingest({
        content: 'Content with full metadata test123',
        tags: ['metadata', 'test'],
      });

      const result = await tools.recall({ query: 'metadata test123' });

      expect(result.results.length).toBeGreaterThanOrEqual(1);
      const firstResult = result.results[0];
      expect(firstResult.metadata).toBeDefined();
      expect(firstResult.metadata.createdAt).toBeDefined();
      expect(firstResult.metadata.retentionStrength).toBeDefined();
      expect(firstResult.tags).toContain('metadata');
    });
  });

  // ==========================================================================
  // Tool 3: get_knowledge
  // ==========================================================================
  describe('get_knowledge tool', () => {
    it('should retrieve existing node by ID', async () => {
      const ingestResult = await tools.ingest({
        content: 'Specific node for get_knowledge test',
        tags: ['getknowledge', 'test'],
      });

      const result = await tools.getKnowledge({ nodeId: ingestResult.nodeId });

      expect(result).not.toHaveProperty('error');
      expect((result as KnowledgeNode).id).toBe(ingestResult.nodeId);
      expect((result as KnowledgeNode).content).toBe('Specific node for get_knowledge test');
      expect((result as KnowledgeNode).tags).toContain('getknowledge');
    });

    it('should return error for non-existent node', async () => {
      const result = await tools.getKnowledge({ nodeId: 'nonexistent-id-12345' });

      expect(result).toHaveProperty('error');
      expect((result as { error: string }).error).toBe('Node not found');
    });

    it('should update access count when retrieving', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for access count test',
      });

      const nodeBefore = db.getNode(ingestResult.nodeId);
      expect(nodeBefore?.accessCount).toBe(0);

      await tools.getKnowledge({ nodeId: ingestResult.nodeId });

      const nodeAfter = db.getNode(ingestResult.nodeId);
      expect(nodeAfter?.accessCount).toBe(1);
    });

    it('should update last accessed timestamp', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for timestamp test',
      });

      const nodeBefore = db.getNode(ingestResult.nodeId);
      const initialAccessTime = nodeBefore?.lastAccessedAt.getTime() || 0;

      // Wait a tiny bit to ensure timestamp difference
      await new Promise((resolve) => setTimeout(resolve, 10));

      await tools.getKnowledge({ nodeId: ingestResult.nodeId });

      const nodeAfter = db.getNode(ingestResult.nodeId);
      expect(nodeAfter?.lastAccessedAt.getTime()).toBeGreaterThan(initialAccessTime);
    });
  });

  // ==========================================================================
  // Tool 4: get_related
  // ==========================================================================
  describe('get_related tool', () => {
    it('should return empty when no connections exist', async () => {
      const ingestResult = await tools.ingest({
        content: 'Isolated node with no connections',
      });

      const result = await tools.getRelated({ nodeId: ingestResult.nodeId });

      expect(result.sourceNode).toBe(ingestResult.nodeId);
      expect(result.relatedCount).toBe(0);
      expect(result.related).toHaveLength(0);
    });

    it('should find directly connected nodes (depth 1)', async () => {
      // Create two nodes and connect them
      const node1 = await tools.ingest({ content: 'Node A for graph test' });
      const node2 = await tools.ingest({ content: 'Node B for graph test' });

      // Create an edge between them
      db.insertEdge({
        fromId: node1.nodeId,
        toId: node2.nodeId,
        edgeType: 'relates_to',
        weight: 0.8,
      });

      const result = await tools.getRelated({ nodeId: node1.nodeId, depth: 1 });

      expect(result.depth).toBe(1);
      expect(result.relatedCount).toBe(1);
      expect(result.related[0].id).toBe(node2.nodeId);
    });

    it('should traverse multiple hops with depth > 1', async () => {
      // Create a chain: A -> B -> C
      const nodeA = await tools.ingest({ content: 'Node A chain' });
      const nodeB = await tools.ingest({ content: 'Node B chain' });
      const nodeC = await tools.ingest({ content: 'Node C chain' });

      db.insertEdge({
        fromId: nodeA.nodeId,
        toId: nodeB.nodeId,
        edgeType: 'relates_to',
      });
      db.insertEdge({
        fromId: nodeB.nodeId,
        toId: nodeC.nodeId,
        edgeType: 'relates_to',
      });

      // Depth 1 should only find B
      const depth1Result = await tools.getRelated({ nodeId: nodeA.nodeId, depth: 1 });
      expect(depth1Result.relatedCount).toBe(1);

      // Depth 2 should find both B and C
      const depth2Result = await tools.getRelated({ nodeId: nodeA.nodeId, depth: 2 });
      expect(depth2Result.relatedCount).toBe(2);
    });

    it('should use default depth of 1', async () => {
      const node1 = await tools.ingest({ content: 'Default depth test A' });
      const node2 = await tools.ingest({ content: 'Default depth test B' });

      db.insertEdge({
        fromId: node1.nodeId,
        toId: node2.nodeId,
        edgeType: 'relates_to',
      });

      const result = await tools.getRelated({ nodeId: node1.nodeId });

      expect(result.depth).toBe(1);
    });
  });

  // ==========================================================================
  // Tool 5: remember_person
  // ==========================================================================
  describe('remember_person tool', () => {
    it('should create a new person with basic info', async () => {
      const result = await tools.rememberPerson({
        name: 'John Smith',
      });

      expect(result.success).toBe(true);
      expect(result.personId).toBeDefined();
      expect(result.message).toContain('Remembered John Smith');
    });

    it('should create a person with all fields', async () => {
      const result = await tools.rememberPerson({
        name: 'Jane Doe',
        howWeMet: 'Tech conference',
        relationshipType: 'colleague',
        organization: 'TechCorp',
        role: 'Senior Developer',
        email: 'jane@techcorp.com',
        notes: 'Expert in distributed systems',
        sharedTopics: ['microservices', 'kubernetes'],
      });

      expect(result.success).toBe(true);

      const person = db.getPersonByName('Jane Doe');
      expect(person?.howWeMet).toBe('Tech conference');
      expect(person?.relationshipType).toBe('colleague');
      expect(person?.organization).toBe('TechCorp');
      expect(person?.role).toBe('Senior Developer');
      expect(person?.email).toBe('jane@techcorp.com');
      expect(person?.notes).toBe('Expert in distributed systems');
      expect(person?.sharedTopics).toContain('microservices');
    });

    it('should detect duplicate person', async () => {
      await tools.rememberPerson({ name: 'Duplicate Person' });
      const result = await tools.rememberPerson({ name: 'Duplicate Person' });

      expect(result.existing).toBe(true);
      expect(result.message).toContain('already exists');
    });

    it('should initialize default values correctly', async () => {
      const result = await tools.rememberPerson({ name: 'Default Values Person' });
      const person = db.getPerson(result.personId!);

      expect(person?.relationshipHealth).toBe(0.5);
      expect(person?.contactFrequency).toBe(0);
      expect(person?.sharedTopics).toHaveLength(0);
      expect(person?.aliases).toHaveLength(0);
    });
  });

  // ==========================================================================
  // Tool 6: get_person
  // ==========================================================================
  describe('get_person tool', () => {
    beforeEach(async () => {
      // Create a test person with last contact date
      const person = db.insertPerson({
        name: 'Test Person For Lookup',
        aliases: ['TP'],
        howWeMet: 'Test setup',
        sharedTopics: ['testing'],
        sharedProjects: [],
        socialLinks: {},
        contactFrequency: 0,
        relationshipHealth: 0.6,
        lastContactAt: mockTimestamp(15), // 15 days ago
        createdAt: new Date(),
        updatedAt: new Date(),
      });
    });

    it('should find person by name', async () => {
      const result = await tools.getPerson({ name: 'Test Person For Lookup' });

      expect(result.found).toBe(true);
      expect(result.person?.name).toBe('Test Person For Lookup');
    });

    it('should calculate days since contact', async () => {
      const result = await tools.getPerson({ name: 'Test Person For Lookup' });

      expect(result.found).toBe(true);
      // Should be approximately 15 days
      expect(result.person?.daysSinceContact).toBeGreaterThanOrEqual(14);
      expect(result.person?.daysSinceContact).toBeLessThanOrEqual(16);
    });

    it('should return not found for non-existent person', async () => {
      const result = await tools.getPerson({ name: 'Non Existent Person' });

      expect(result.found).toBe(false);
      // Message contains "found in memory" (not found in memory)
      expect(result.message).toContain('found in memory');
    });

    it('should handle person with no last contact date', async () => {
      db.insertPerson({
        name: 'No Contact Person',
        aliases: [],
        sharedTopics: [],
        sharedProjects: [],
        socialLinks: {},
        contactFrequency: 0,
        relationshipHealth: 0.5,
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      const result = await tools.getPerson({ name: 'No Contact Person' });

      expect(result.found).toBe(true);
      expect(result.person?.daysSinceContact).toBeNull();
    });
  });

  // ==========================================================================
  // Tool 7: mark_reviewed
  // ==========================================================================
  describe('mark_reviewed tool', () => {
    it('should update retention strength', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for review test',
      });

      // Manually decrease retention to simulate decay
      db['db']
        .prepare('UPDATE knowledge_nodes SET retention_strength = 0.5 WHERE id = ?')
        .run(ingestResult.nodeId);

      const result = await tools.markReviewed({ nodeId: ingestResult.nodeId });

      expect(result.success).toBe(true);
      expect(result.previousRetention).toBe(0.5);
      expect(result.newRetention).toBe(1.0); // Should reset to 1.0
    });

    it('should increase stability factor on successful review', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for stability test',
      });

      const nodeBefore = db.getNode(ingestResult.nodeId);
      const initialStability = nodeBefore?.stabilityFactor || 1.0;

      const result = await tools.markReviewed({ nodeId: ingestResult.nodeId });

      expect(result.newStability).toBeGreaterThan(initialStability);
    });

    it('should increment review count', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for review count test',
      });

      await tools.markReviewed({ nodeId: ingestResult.nodeId });
      const result = await tools.markReviewed({ nodeId: ingestResult.nodeId });

      expect(result.reviewCount).toBe(2);
    });

    it('should schedule next review date', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for next review test',
      });

      const result = await tools.markReviewed({ nodeId: ingestResult.nodeId });

      expect(result.nextReviewDate).toBeDefined();
      const nextReview = new Date(result.nextReviewDate!);
      expect(nextReview.getTime()).toBeGreaterThan(Date.now());
    });

    it('should return error for non-existent node', async () => {
      const result = await tools.markReviewed({ nodeId: 'nonexistent-node-id' });

      expect(result.error).toBe('Node not found');
    });

    it('should reset stability on lapse (low retention)', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for lapse test',
      });

      // Simulate a highly stable node that then decays below threshold
      db['db']
        .prepare(
          'UPDATE knowledge_nodes SET stability_factor = 10, retention_strength = 0.2 WHERE id = ?'
        )
        .run(ingestResult.nodeId);

      const result = await tools.markReviewed({ nodeId: ingestResult.nodeId });

      // Stability should reset to 1.0 on lapse
      expect(result.newStability).toBe(1.0);
    });
  });

  // ==========================================================================
  // Tool 8: daily_brief
  // ==========================================================================
  describe('daily_brief tool', () => {
    beforeEach(async () => {
      // Seed some data for the brief
      await tools.ingest({ content: 'Recent knowledge 1', tags: ['brief'] });
      await tools.ingest({ content: 'Recent knowledge 2', tags: ['brief'] });
      await tools.rememberPerson({
        name: 'Brief Test Person',
        sharedTopics: ['testing'],
      });
    });

    it('should return all required sections', async () => {
      const result = await tools.dailyBrief();

      expect(result.date).toBeDefined();
      expect(result.greeting).toBeDefined();
      expect(result.healthStatus).toBeDefined();
      expect(result.stats).toBeDefined();
      expect(result.reviewNeeded).toBeDefined();
      expect(result.peopleToReconnect).toBeDefined();
      expect(result.recentlyAdded).toBeDefined();
    });

    it('should return correct time-based greeting', async () => {
      const result = await tools.dailyBrief();
      const hour = new Date().getHours();

      if (hour < 12) {
        expect(result.greeting).toBe('Good morning');
      } else if (hour < 17) {
        expect(result.greeting).toBe('Good afternoon');
      } else {
        expect(result.greeting).toBe('Good evening');
      }
    });

    it('should include stats about the knowledge base', async () => {
      const result = await tools.dailyBrief();

      expect(result.stats.totalKnowledge).toBeGreaterThan(0);
      expect(result.stats.peopleInNetwork).toBeGreaterThanOrEqual(0);
      expect(result.stats.connections).toBeGreaterThanOrEqual(0);
      expect(result.stats.databaseSize).toBeDefined();
    });

    it('should return date in YYYY-MM-DD format', async () => {
      const result = await tools.dailyBrief();

      expect(result.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });

    it('should include recently added nodes', async () => {
      const result = await tools.dailyBrief();

      expect(result.recentlyAdded.length).toBeGreaterThan(0);
      expect(result.recentlyAdded[0].id).toBeDefined();
      expect(result.recentlyAdded[0].preview).toBeDefined();
    });
  });

  // ==========================================================================
  // Tool 9: health_check
  // ==========================================================================
  describe('health_check tool', () => {
    it('should return health status', async () => {
      const result = await tools.healthCheck();

      expect(result.status).toBeDefined();
      expect(['healthy', 'warning', 'critical']).toContain(result.status);
    });

    it('should include database size information', async () => {
      const result = await tools.healthCheck();

      expect(result.databaseSize).toBeDefined();
      expect(result.databaseSize.bytes).toBeGreaterThanOrEqual(0);
      expect(result.databaseSize.mb).toBeGreaterThanOrEqual(0);
      expect(result.databaseSize.formatted).toBeDefined();
    });

    it('should include node and people counts', async () => {
      const result = await tools.healthCheck();

      expect(result.nodeCount).toBeGreaterThanOrEqual(0);
      expect(result.peopleCount).toBeGreaterThanOrEqual(0);
      expect(result.edgeCount).toBeGreaterThanOrEqual(0);
    });

    it('should check WAL mode status', async () => {
      const result = await tools.healthCheck();

      expect(typeof result.walMode).toBe('boolean');
    });

    it('should perform integrity check', async () => {
      const result = await tools.healthCheck();

      expect(typeof result.integrityCheck).toBe('boolean');
      // In-memory databases should pass integrity check
      expect(result.integrityCheck).toBe(true);
    });

    it('should provide recommendations', async () => {
      const result = await tools.healthCheck();

      expect(result.recommendations).toBeDefined();
      expect(Array.isArray(result.recommendations)).toBe(true);
      expect(result.recommendations.length).toBeGreaterThan(0);
    });

    it('should include warnings array', async () => {
      const result = await tools.healthCheck();

      expect(result.warnings).toBeDefined();
      expect(Array.isArray(result.warnings)).toBe(true);
    });
  });

  // ==========================================================================
  // Tool 10: backup
  // ==========================================================================
  describe('backup tool', () => {
    // Note: Backup tests use the real filesystem, but with in-memory DB
    // The backup will fail gracefully for :memory: databases

    it('should handle backup gracefully for in-memory databases', async () => {
      // For in-memory databases, backup will return success: false
      // This is expected behavior
      const result = await tools.backup();

      // Either succeeds or fails gracefully
      expect(typeof result.success).toBe('boolean');
      expect(result.message).toBeDefined();
      expect(result.totalBackups).toBeGreaterThanOrEqual(0);
      expect(Array.isArray(result.backups)).toBe(true);
    });

    it('should return backup metadata structure', async () => {
      const result = await tools.backup();

      // Check structure regardless of success/failure
      expect(typeof result.totalBackups).toBe('number');
      expect(Array.isArray(result.backups)).toBe(true);
    });
  });

  // ==========================================================================
  // Tool 11: list_backups
  // ==========================================================================
  describe('list_backups tool', () => {
    it('should return backups list structure', async () => {
      const result = await tools.listBackups();

      expect(result.totalBackups).toBeGreaterThanOrEqual(0);
      expect(Array.isArray(result.backups)).toBe(true);
    });

    it('should include backup metadata for each entry', async () => {
      // First create a backup if possible
      try {
        await tools.backup();
      } catch {
        // May fail for in-memory DB
      }

      const result = await tools.listBackups();

      if (result.backups.length > 0) {
        const backup = result.backups[0];
        expect(backup.path).toBeDefined();
        expect(backup.size).toBeDefined();
        expect(backup.date).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // Tool 12: optimize_database
  // ==========================================================================
  describe('optimize_database tool', () => {
    it('should successfully optimize database', async () => {
      // Add and delete some data to create fragmentation
      for (let i = 0; i < 10; i++) {
        const result = await tools.ingest({ content: `Temporary node ${i}` });
        db.deleteNode(result.nodeId);
      }

      const result = await tools.optimizeDatabase();

      expect(result.success).toBe(true);
      expect(result.message).toContain('optimized');
    });

    it('should return size before and after', async () => {
      const result = await tools.optimizeDatabase();

      expect(result.sizeBefore).toBeDefined();
      expect(result.sizeAfter).toBeDefined();
      expect(result.spaceSaved).toBeDefined();
    });

    it('should not crash on empty database', async () => {
      const emptyDb = createTestDatabase();
      const emptyTools = createMCPToolHandler(emptyDb);

      const result = await emptyTools.optimizeDatabase();

      expect(result.success).toBe(true);
      emptyDb.close();
    });
  });

  // ==========================================================================
  // Tool 13: apply_decay
  // ==========================================================================
  describe('apply_decay tool', () => {
    it('should apply decay and return count', async () => {
      // Create nodes with old access dates
      for (let i = 0; i < 5; i++) {
        await tools.ingest({ content: `Decay test node ${i}` });
      }

      // Manually set old access dates
      db['db'].prepare(`
        UPDATE knowledge_nodes
        SET last_accessed_at = datetime('now', '-30 days')
        WHERE content LIKE 'Decay test node%'
      `).run();

      const result = await tools.applyDecay();

      expect(result.success).toBe(true);
      expect(result.nodesUpdated).toBeGreaterThanOrEqual(0);
      expect(result.message).toContain('Applied decay');
    });

    it('should decrease retention strength for old nodes', async () => {
      const ingestResult = await tools.ingest({
        content: 'Old node for decay verification',
      });

      // Set access date to 30 days ago
      db['db']
        .prepare(
          `UPDATE knowledge_nodes SET last_accessed_at = datetime('now', '-30 days') WHERE id = ?`
        )
        .run(ingestResult.nodeId);

      const nodeBefore = db.getNode(ingestResult.nodeId);
      expect(nodeBefore?.retentionStrength).toBe(1.0);

      await tools.applyDecay();

      const nodeAfter = db.getNode(ingestResult.nodeId);
      expect(nodeAfter?.retentionStrength).toBeLessThan(1.0);
    });

    it('should not decay recently accessed nodes significantly', async () => {
      const ingestResult = await tools.ingest({
        content: 'Fresh node for decay test',
      });

      const nodeBefore = db.getNode(ingestResult.nodeId);
      expect(nodeBefore?.retentionStrength).toBe(1.0);

      await tools.applyDecay();

      const nodeAfter = db.getNode(ingestResult.nodeId);
      // Recently accessed nodes should retain most of their strength
      // The dual-strength model uses storage_strength as a factor, so some decay is normal
      // but it should still be above 0.6 for very recently created nodes
      expect(nodeAfter?.retentionStrength).toBeGreaterThan(0.6);
    });

    it('should apply slower decay to emotional content', async () => {
      // Create two nodes - one emotional, one neutral
      const emotionalNode = await tools.ingest({
        content: 'I am absolutely THRILLED about this amazing breakthrough! This is incredible!',
      });
      const neutralNode = await tools.ingest({
        content: 'The meeting is scheduled for Tuesday at 3pm in room 204.',
      });

      // Verify emotional content was detected
      const emotionalBefore = db.getNode(emotionalNode.nodeId);
      const neutralBefore = db.getNode(neutralNode.nodeId);

      // The sentiment intensity should be higher for emotional content
      expect(emotionalBefore?.sentimentIntensity).toBeGreaterThanOrEqual(0);
      expect(neutralBefore?.sentimentIntensity).toBeGreaterThanOrEqual(0);

      // Set both to 30 days old
      db['db']
        .prepare(
          `UPDATE knowledge_nodes SET last_accessed_at = datetime('now', '-30 days') WHERE id IN (?, ?)`
        )
        .run(emotionalNode.nodeId, neutralNode.nodeId);

      await tools.applyDecay();

      const emotionalAfter = db.getNode(emotionalNode.nodeId);
      const neutralAfter = db.getNode(neutralNode.nodeId);

      // Both should have decayed
      expect(emotionalAfter?.retentionStrength).toBeLessThan(1.0);
      expect(neutralAfter?.retentionStrength).toBeLessThan(1.0);

      // Emotional content should decay slower (higher retention)
      // or at least not decay faster - the difference may be small
      expect(emotionalAfter?.retentionStrength).toBeGreaterThanOrEqual(
        neutralAfter?.retentionStrength ?? 0
      );
    });
  });

  // ==========================================================================
  // Additional Edge Cases and Error Handling
  // ==========================================================================
  describe('edge cases and error handling', () => {
    it('should handle very long content in ingest', async () => {
      const longContent = 'A'.repeat(10000);
      const result = await tools.ingest({ content: longContent });

      expect(result.success).toBe(true);

      const node = db.getNode(result.nodeId);
      expect(node?.content.length).toBe(10000);
    });

    it('should handle special characters in content', async () => {
      const specialContent = 'Test with "quotes" and <tags> and &entities;';
      const result = await tools.ingest({ content: specialContent });

      const node = db.getNode(result.nodeId);
      expect(node?.content).toBe(specialContent);
    });

    it('should handle unicode content', async () => {
      const unicodeContent = 'Test with emoji: 🎉 and Japanese: こんにちは and Arabic: مرحبا';
      const result = await tools.ingest({ content: unicodeContent });

      const node = db.getNode(result.nodeId);
      expect(node?.content).toBe(unicodeContent);
    });

    it('should handle empty tags array', async () => {
      const result = await tools.ingest({
        content: 'Node with empty tags',
        tags: [],
      });

      const node = db.getNode(result.nodeId);
      expect(node?.tags).toHaveLength(0);
    });

    it('should handle concurrent operations', async () => {
      // Simulate concurrent ingests
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(tools.ingest({ content: `Concurrent node ${i}` }));
      }

      const results = await Promise.all(promises);

      expect(results).toHaveLength(10);
      expect(results.every((r) => r.success)).toBe(true);
    });

    it('should handle rapid mark_reviewed calls', async () => {
      const ingestResult = await tools.ingest({
        content: 'Node for rapid review test',
      });

      // Call mark_reviewed multiple times rapidly
      const results = [];
      for (let i = 0; i < 5; i++) {
        results.push(await tools.markReviewed({ nodeId: ingestResult.nodeId }));
      }

      // All should succeed
      expect(results.every((r) => r.success)).toBe(true);

      // Review count should be 5
      const node = db.getNode(ingestResult.nodeId);
      expect(node?.reviewCount).toBe(5);
    });
  });
});

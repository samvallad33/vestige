import { describe, it, expect, beforeEach, afterEach } from '@rstest/core';
import Database from 'better-sqlite3';
import { nanoid } from 'nanoid';
import {
  createTestDatabase,
  createTestNode,
  createTestPerson,
  createTestEdge,
  cleanupTestDatabase,
  generateTestId,
} from './setup.js';

describe('VestigeDatabase', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = createTestDatabase();
  });

  afterEach(() => {
    cleanupTestDatabase(db);
  });

  describe('Schema Setup', () => {
    it('should create all required tables', () => {
      const tables = db.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
      ).all() as { name: string }[];

      const tableNames = tables.map(t => t.name);

      expect(tableNames).toContain('knowledge_nodes');
      expect(tableNames).toContain('knowledge_fts');
      expect(tableNames).toContain('people');
      expect(tableNames).toContain('interactions');
      expect(tableNames).toContain('graph_edges');
      expect(tableNames).toContain('sources');
      expect(tableNames).toContain('embeddings');
      expect(tableNames).toContain('vestige_metadata');
    });

    it('should create required indexes', () => {
      const indexes = db.prepare(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
      ).all() as { name: string }[];

      const indexNames = indexes.map(i => i.name);

      expect(indexNames).toContain('idx_nodes_created_at');
      expect(indexNames).toContain('idx_nodes_last_accessed');
      expect(indexNames).toContain('idx_nodes_retention');
      expect(indexNames).toContain('idx_people_name');
      expect(indexNames).toContain('idx_edges_from');
      expect(indexNames).toContain('idx_edges_to');
    });
  });

  describe('insertNode', () => {
    it('should create a new knowledge node', () => {
      const id = nanoid();
      const now = new Date().toISOString();
      const nodeData = createTestNode({
        content: 'Test knowledge content',
        tags: ['test', 'knowledge'],
      });

      const stmt = db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, summary,
          created_at, updated_at, last_accessed_at, access_count,
          retention_strength, stability_factor, sentiment_intensity,
          source_type, source_platform,
          confidence, people, concepts, events, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(
        id,
        nodeData.content,
        null,
        now,
        now,
        now,
        0,
        1.0,
        1.0,
        0,
        nodeData.sourceType,
        nodeData.sourcePlatform,
        0.8,
        JSON.stringify(nodeData.people),
        JSON.stringify(nodeData.concepts),
        JSON.stringify(nodeData.events),
        JSON.stringify(nodeData.tags)
      );

      const result = db.prepare('SELECT * FROM knowledge_nodes WHERE id = ?').get(id) as Record<string, unknown>;

      expect(result).toBeDefined();
      expect(result['content']).toBe('Test knowledge content');
      expect(JSON.parse(result['tags'] as string)).toContain('test');
      expect(JSON.parse(result['tags'] as string)).toContain('knowledge');
    });

    it('should store retention and stability factors', () => {
      const id = nanoid();
      const now = new Date().toISOString();
      const nodeData = createTestNode();

      const stmt = db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content,
          created_at, updated_at, last_accessed_at,
          retention_strength, stability_factor, sentiment_intensity,
          storage_strength, retrieval_strength,
          source_type, source_platform,
          confidence, people, concepts, events, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(
        id,
        nodeData.content,
        now,
        now,
        now,
        0.85,
        2.5,
        0.7,
        1.5,
        0.9,
        nodeData.sourceType,
        nodeData.sourcePlatform,
        0.8,
        '[]',
        '[]',
        '[]',
        '[]'
      );

      const result = db.prepare('SELECT * FROM knowledge_nodes WHERE id = ?').get(id) as Record<string, unknown>;

      expect(result['retention_strength']).toBe(0.85);
      expect(result['stability_factor']).toBe(2.5);
      expect(result['sentiment_intensity']).toBe(0.7);
      expect(result['storage_strength']).toBe(1.5);
      expect(result['retrieval_strength']).toBe(0.9);
    });
  });

  describe('searchNodes', () => {
    beforeEach(() => {
      // Insert test nodes for searching
      const nodes = [
        { id: generateTestId(), content: 'TypeScript is a typed superset of JavaScript' },
        { id: generateTestId(), content: 'React is a JavaScript library for building user interfaces' },
        { id: generateTestId(), content: 'Python is a versatile programming language' },
      ];

      const stmt = db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, created_at, updated_at, last_accessed_at,
          source_type, source_platform, confidence, people, concepts, events, tags
        ) VALUES (?, ?, datetime('now'), datetime('now'), datetime('now'), 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
      `);

      for (const node of nodes) {
        stmt.run(node.id, node.content);
      }
    });

    it('should find nodes by keyword using FTS', () => {
      const results = db.prepare(`
        SELECT kn.* FROM knowledge_nodes kn
        JOIN knowledge_fts fts ON kn.id = fts.id
        WHERE knowledge_fts MATCH ?
        ORDER BY rank
      `).all('JavaScript') as Record<string, unknown>[];

      expect(results.length).toBe(2);
      expect(results.some(r => (r['content'] as string).includes('TypeScript'))).toBe(true);
      expect(results.some(r => (r['content'] as string).includes('React'))).toBe(true);
    });

    it('should not find unrelated content', () => {
      const results = db.prepare(`
        SELECT kn.* FROM knowledge_nodes kn
        JOIN knowledge_fts fts ON kn.id = fts.id
        WHERE knowledge_fts MATCH ?
      `).all('Rust') as Record<string, unknown>[];

      expect(results.length).toBe(0);
    });

    it('should find partial matches', () => {
      const results = db.prepare(`
        SELECT kn.* FROM knowledge_nodes kn
        JOIN knowledge_fts fts ON kn.id = fts.id
        WHERE knowledge_fts MATCH ?
      `).all('programming') as Record<string, unknown>[];

      expect(results.length).toBe(1);
      expect((results[0]['content'] as string)).toContain('Python');
    });
  });

  describe('People Operations', () => {
    it('should insert a person', () => {
      const id = nanoid();
      const now = new Date().toISOString();
      const personData = createTestPerson({
        name: 'John Doe',
        relationshipType: 'friend',
        organization: 'Acme Inc',
      });

      const stmt = db.prepare(`
        INSERT INTO people (
          id, name, aliases, relationship_type, organization,
          contact_frequency, shared_topics, shared_projects, relationship_health,
          social_links, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(
        id,
        personData.name,
        JSON.stringify(personData.aliases),
        personData.relationshipType,
        personData.organization,
        personData.contactFrequency,
        JSON.stringify(personData.sharedTopics),
        JSON.stringify(personData.sharedProjects),
        personData.relationshipHealth,
        JSON.stringify(personData.socialLinks),
        now,
        now
      );

      const result = db.prepare('SELECT * FROM people WHERE id = ?').get(id) as Record<string, unknown>;

      expect(result).toBeDefined();
      expect(result['name']).toBe('John Doe');
      expect(result['relationship_type']).toBe('friend');
      expect(result['organization']).toBe('Acme Inc');
    });

    it('should find person by name', () => {
      const id = nanoid();
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO people (id, name, aliases, social_links, shared_topics, shared_projects, created_at, updated_at)
        VALUES (?, ?, '[]', '{}', '[]', '[]', ?, ?)
      `).run(id, 'Jane Smith', now, now);

      const result = db.prepare('SELECT * FROM people WHERE name = ?').get('Jane Smith') as Record<string, unknown>;

      expect(result).toBeDefined();
      expect(result['id']).toBe(id);
    });

    it('should find person by alias', () => {
      const id = nanoid();
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO people (id, name, aliases, social_links, shared_topics, shared_projects, created_at, updated_at)
        VALUES (?, ?, ?, '{}', '[]', '[]', ?, ?)
      `).run(id, 'Robert Johnson', JSON.stringify(['Bob', 'Bobby']), now, now);

      const result = db.prepare(`
        SELECT * FROM people WHERE name = ? OR aliases LIKE ?
      `).get('Bob', '%"Bob"%') as Record<string, unknown>;

      expect(result).toBeDefined();
      expect(result['name']).toBe('Robert Johnson');
    });
  });

  describe('Graph Edges', () => {
    let nodeId1: string;
    let nodeId2: string;

    beforeEach(() => {
      nodeId1 = nanoid();
      nodeId2 = nanoid();
      const now = new Date().toISOString();

      // Create two nodes
      const stmt = db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, created_at, updated_at, last_accessed_at,
          source_type, source_platform, confidence, people, concepts, events, tags
        ) VALUES (?, ?, ?, ?, ?, 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
      `);

      stmt.run(nodeId1, 'Node 1 content', now, now, now);
      stmt.run(nodeId2, 'Node 2 content', now, now, now);
    });

    it('should create an edge between nodes', () => {
      const edgeId = nanoid();
      const now = new Date().toISOString();
      const edgeData = createTestEdge(nodeId1, nodeId2, {
        edgeType: 'relates_to',
        weight: 0.8,
      });

      db.prepare(`
        INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `).run(edgeId, edgeData.fromId, edgeData.toId, edgeData.edgeType, edgeData.weight, '{}', now);

      const result = db.prepare('SELECT * FROM graph_edges WHERE id = ?').get(edgeId) as Record<string, unknown>;

      expect(result).toBeDefined();
      expect(result['from_id']).toBe(nodeId1);
      expect(result['to_id']).toBe(nodeId2);
      expect(result['edge_type']).toBe('relates_to');
      expect(result['weight']).toBe(0.8);
    });

    it('should find related nodes', () => {
      const edgeId = nanoid();
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
        VALUES (?, ?, ?, 'relates_to', 0.5, '{}', ?)
      `).run(edgeId, nodeId1, nodeId2, now);

      const results = db.prepare(`
        SELECT DISTINCT
          CASE WHEN from_id = ? THEN to_id ELSE from_id END as related_id
        FROM graph_edges
        WHERE from_id = ? OR to_id = ?
      `).all(nodeId1, nodeId1, nodeId1) as { related_id: string }[];

      expect(results.length).toBe(1);
      expect(results[0].related_id).toBe(nodeId2);
    });

    it('should enforce unique constraint on from_id, to_id, edge_type', () => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
        VALUES (?, ?, ?, 'relates_to', 0.5, '{}', ?)
      `).run(nanoid(), nodeId1, nodeId2, now);

      // Attempting to insert duplicate should fail
      expect(() => {
        db.prepare(`
          INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
          VALUES (?, ?, ?, 'relates_to', 0.7, '{}', ?)
        `).run(nanoid(), nodeId1, nodeId2, now);
      }).toThrow();
    });
  });

  describe('Decay Simulation', () => {
    it('should be able to update retention strength', () => {
      const id = nanoid();
      const now = new Date().toISOString();

      // Insert a node with initial retention
      db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, created_at, updated_at, last_accessed_at,
          retention_strength, stability_factor,
          source_type, source_platform, confidence, people, concepts, events, tags
        ) VALUES (?, 'Test content', ?, ?, ?, 1.0, 1.0, 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
      `).run(id, now, now, now);

      // Simulate decay
      const newRetention = 0.75;
      db.prepare(`
        UPDATE knowledge_nodes SET retention_strength = ? WHERE id = ?
      `).run(newRetention, id);

      const result = db.prepare('SELECT retention_strength FROM knowledge_nodes WHERE id = ?').get(id) as { retention_strength: number };

      expect(result.retention_strength).toBe(0.75);
    });

    it('should track review count', () => {
      const id = nanoid();
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO knowledge_nodes (
          id, content, created_at, updated_at, last_accessed_at,
          review_count, source_type, source_platform, confidence, people, concepts, events, tags
        ) VALUES (?, 'Test content', ?, ?, ?, 0, 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
      `).run(id, now, now, now);

      // Simulate review
      db.prepare(`
        UPDATE knowledge_nodes
        SET review_count = review_count + 1,
            retention_strength = 1.0,
            last_accessed_at = ?
        WHERE id = ?
      `).run(new Date().toISOString(), id);

      const result = db.prepare('SELECT review_count, retention_strength FROM knowledge_nodes WHERE id = ?').get(id) as { review_count: number; retention_strength: number };

      expect(result.review_count).toBe(1);
      expect(result.retention_strength).toBe(1.0);
    });
  });

  describe('Statistics', () => {
    it('should count nodes correctly', () => {
      const now = new Date().toISOString();

      // Insert 3 nodes
      for (let i = 0; i < 3; i++) {
        db.prepare(`
          INSERT INTO knowledge_nodes (
            id, content, created_at, updated_at, last_accessed_at,
            source_type, source_platform, confidence, people, concepts, events, tags
          ) VALUES (?, ?, ?, ?, ?, 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
        `).run(nanoid(), `Node ${i}`, now, now, now);
      }

      const result = db.prepare('SELECT COUNT(*) as count FROM knowledge_nodes').get() as { count: number };
      expect(result.count).toBe(3);
    });

    it('should count people correctly', () => {
      const now = new Date().toISOString();

      // Insert 2 people
      for (let i = 0; i < 2; i++) {
        db.prepare(`
          INSERT INTO people (id, name, aliases, social_links, shared_topics, shared_projects, created_at, updated_at)
          VALUES (?, ?, '[]', '{}', '[]', '[]', ?, ?)
        `).run(nanoid(), `Person ${i}`, now, now);
      }

      const result = db.prepare('SELECT COUNT(*) as count FROM people').get() as { count: number };
      expect(result.count).toBe(2);
    });

    it('should count edges correctly', () => {
      const now = new Date().toISOString();

      // Create nodes first
      const nodeIds = [nanoid(), nanoid(), nanoid()];
      for (const id of nodeIds) {
        db.prepare(`
          INSERT INTO knowledge_nodes (
            id, content, created_at, updated_at, last_accessed_at,
            source_type, source_platform, confidence, people, concepts, events, tags
          ) VALUES (?, 'Content', ?, ?, ?, 'manual', 'manual', 0.8, '[]', '[]', '[]', '[]')
        `).run(id, now, now, now);
      }

      // Insert 2 edges
      db.prepare(`
        INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
        VALUES (?, ?, ?, 'relates_to', 0.5, '{}', ?)
      `).run(nanoid(), nodeIds[0], nodeIds[1], now);

      db.prepare(`
        INSERT INTO graph_edges (id, from_id, to_id, edge_type, weight, metadata, created_at)
        VALUES (?, ?, ?, 'supports', 0.7, '{}', ?)
      `).run(nanoid(), nodeIds[1], nodeIds[2], now);

      const result = db.prepare('SELECT COUNT(*) as count FROM graph_edges').get() as { count: number };
      expect(result.count).toBe(2);
    });
  });
});

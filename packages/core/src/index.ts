#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { VestigeDatabase, VestigeDatabaseError } from './core/database.js';
import {
  captureContext,
  formatContextForInjection,
  readSavedContext,
} from './core/context-watcher.js';
import {
  IngestInputSchema,
  RecallOptionsSchema,
  type KnowledgeNode,
} from './core/types.js';

// New imports for integrated features
import { FSRSScheduler, Grade, type ReviewGrade } from './core/fsrs.js';
import { createEmbeddingService, type EmbeddingService } from './core/embeddings.js';
import { createVectorStore, type IVectorStore } from './core/vector-store.js';
import { runConsolidation } from './core/consolidation.js';
import { getConfig, type VestigeConfig } from './core/config.js';
import { JobQueue } from './jobs/JobQueue.js';
import { createDecayJobHandler } from './jobs/DecayJob.js';
import { createREMCycleJobHandler } from './jobs/REMCycleJob.js';
import {
  CacheService,
  CACHE_KEYS,
  nodeCache,
  invalidateNodeCaches,
  destroyAllCaches,
} from './services/CacheService.js';
import { logger, mcpLogger } from './utils/logger.js';

// ============================================================================
// VESTIGE MCP SERVER
// ============================================================================

const server = new McpServer({
  name: 'vestige',
  version: '0.3.0',
});

// Initialize configuration
const config = getConfig();

// Initialize database
const db = new VestigeDatabase();

// Initialize FSRS scheduler
const fsrsScheduler = new FSRSScheduler({
  desiredRetention: config.fsrs.desiredRetention,
  ...(config.fsrs.weights ? { weights: config.fsrs.weights } : {}),
});

// Services initialized asynchronously
let embeddingService: EmbeddingService | null = null;
let vectorStore: IVectorStore | null = null;
let jobQueue: JobQueue | null = null;

// ============================================================================
// ASYNC SERVICE INITIALIZATION
// ============================================================================

async function initializeServices(): Promise<void> {
  logger.info('Initializing Vestige services...');

  // Initialize embedding service (with fallback)
  try {
    embeddingService = await createEmbeddingService({
      host: config.embeddings.ollamaHost,
      model: config.embeddings.model,
    });
    logger.info('Embedding service initialized');
  } catch (error) {
    logger.warn('Failed to initialize embedding service', { error: String(error) });
  }

  // Initialize vector store
  try {
    vectorStore = await createVectorStore(
      () => (db as unknown as { db: import('better-sqlite3').Database }).db,
      config.vectorStore.chromaHost
    );
    logger.info('Vector store initialized');
  } catch (error) {
    logger.warn('Failed to initialize vector store', { error: String(error) });
  }

  // Initialize job queue
  try {
    jobQueue = new JobQueue();

    // Register job handlers
    jobQueue.register('decay', createDecayJobHandler(db), {
      concurrency: 1,
      retryDelay: 60000, // 1 minute
    });
    jobQueue.register('rem-cycle', createREMCycleJobHandler(db), {
      concurrency: 1,
      retryDelay: 300000, // 5 minutes
    });

    // Schedule recurring jobs
    if (config.consolidation.enabled) {
      // Schedule decay at configured hour (default 3 AM)
      jobQueue.schedule('decay', `0 ${config.consolidation.scheduleHour} * * *`, {});
    }

    if (config.rem.enabled) {
      // Schedule REM cycle every 6 hours
      jobQueue.schedule('rem-cycle', '0 */6 * * *', {
        maxAnalyze: config.rem.maxAnalyze,
      });
    }

    // Start processing
    jobQueue.start();
    logger.info('Job queue initialized and started');
  } catch (error) {
    logger.warn('Failed to initialize job queue', { error: String(error) });
  }

  logger.info('Vestige services initialization complete');
}

// ============================================================================
// HELPER: Safe JSON response with error handling
// ============================================================================

function safeResponse(data: unknown): { content: Array<{ type: 'text'; text: string }> } {
  return {
    content: [{
      type: 'text',
      text: JSON.stringify(data, null, 2),
    }],
  };
}

function errorResponse(error: unknown): { content: Array<{ type: 'text'; text: string }> } {
  const message = error instanceof VestigeDatabaseError
    ? { error: error.message, code: error.code }
    : { error: error instanceof Error ? error.message : 'Unknown error' };

  mcpLogger.error('Tool handler error', error instanceof Error ? error : undefined, message);

  return {
    content: [{
      type: 'text',
      text: JSON.stringify(message, null, 2),
    }],
  };
}

/**
 * Wrap a tool handler with error handling
 */
function withErrorHandling<T>(handler: () => Promise<T>): Promise<T> {
  return handler().catch(error => {
    logger.error('Tool handler error', error instanceof Error ? error : undefined);
    throw error;
  });
}

// ============================================================================
// RESOURCES
// ============================================================================

server.resource(
  'memory://stats',
  'Knowledge base statistics and health status',
  async () => {
    try {
      const health = db.checkHealth();
      const dbSize = db.getDatabaseSize();

      // Add vector store stats if available
      let vectorStats = null;
      if (vectorStore) {
        try {
          vectorStats = await vectorStore.getStats();
        } catch {
          // Ignore vector store errors
        }
      }

      return {
        contents: [{
          uri: 'memory://stats',
          mimeType: 'application/json',
          text: JSON.stringify({
            status: health.status,
            totalKnowledgeNodes: health.nodeCount,
            totalPeople: health.peopleCount,
            totalConnections: health.edgeCount,
            databaseSize: dbSize.formatted,
            lastBackup: health.lastBackup,
            warnings: health.warnings,
            vectorStore: vectorStats,
            embeddingsAvailable: embeddingService ? await embeddingService.isAvailable() : false,
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://stats',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get stats' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://knowledge/recent',
  'Recently added knowledge',
  async () => {
    try {
      const result = db.getRecentNodes({ limit: 20 });

      const formatted = result.items.map(node => ({
        id: node.id,
        summary: node.summary || node.content.slice(0, 200),
        source: `${node.sourcePlatform}/${node.sourceType}`,
        createdAt: node.createdAt.toISOString(),
        tags: node.tags,
      }));

      return {
        contents: [{
          uri: 'memory://knowledge/recent',
          mimeType: 'application/json',
          text: JSON.stringify({
            total: result.total,
            showing: result.items.length,
            hasMore: result.hasMore,
            items: formatted,
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://knowledge/recent',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get recent knowledge' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://knowledge/decaying',
  'Knowledge at risk of being forgotten (low retention)',
  async () => {
    try {
      const result = db.getDecayingNodes(0.5, { limit: 20 });

      const formatted = result.items.map(node => ({
        id: node.id,
        summary: node.summary || node.content.slice(0, 200),
        retentionStrength: node.retentionStrength,
        lastAccessed: node.lastAccessedAt.toISOString(),
        daysSinceAccess: Math.floor(
          (Date.now() - node.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24)
        ),
      }));

      return {
        contents: [{
          uri: 'memory://knowledge/decaying',
          mimeType: 'application/json',
          text: JSON.stringify({
            total: result.total,
            showing: result.items.length,
            hasMore: result.hasMore,
            items: formatted,
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://knowledge/decaying',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get decaying knowledge' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://people/network',
  'Your relationship network',
  async () => {
    try {
      const result = db.getAllPeople({ limit: 50 });

      const formatted = result.items.map(person => ({
        id: person.id,
        name: person.name,
        organization: person.organization,
        relationshipType: person.relationshipType,
        sharedTopics: person.sharedTopics,
        lastContact: person.lastContactAt?.toISOString(),
        relationshipHealth: person.relationshipHealth,
      }));

      return {
        contents: [{
          uri: 'memory://people/network',
          mimeType: 'application/json',
          text: JSON.stringify({
            total: result.total,
            showing: result.items.length,
            hasMore: result.hasMore,
            items: formatted,
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://people/network',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get people network' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://people/reconnect',
  'People you should reconnect with',
  async () => {
    try {
      const result = db.getPeopleToReconnect(30, { limit: 10 });

      const formatted = result.items.map(person => {
        const daysSince = person.lastContactAt
          ? Math.floor((Date.now() - person.lastContactAt.getTime()) / (1000 * 60 * 60 * 24))
          : null;

        return {
          id: person.id,
          name: person.name,
          daysSinceContact: daysSince,
          sharedTopics: person.sharedTopics,
          howWeMet: person.howWeMet,
          suggestion: `Consider reaching out about ${person.sharedTopics[0] || 'catching up'}`,
        };
      });

      return {
        contents: [{
          uri: 'memory://people/reconnect',
          mimeType: 'application/json',
          text: JSON.stringify({
            total: result.total,
            showing: result.items.length,
            hasMore: result.hasMore,
            items: formatted,
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://people/reconnect',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get reconnect suggestions' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://health',
  'Detailed health status of the memory database',
  async () => {
    try {
      const health = db.checkHealth();

      return {
        contents: [{
          uri: 'memory://health',
          mimeType: 'application/json',
          text: JSON.stringify(health, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://health',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to get health status' }),
        }],
      };
    }
  }
);

server.resource(
  'memory://context',
  'Ghost in the Shell - Current system context (active window, clipboard, git)',
  async () => {
    try {
      // Try to read saved context first (from watcher daemon)
      let context = readSavedContext();

      // If no saved context or it's stale (>30 seconds old), capture fresh
      if (!context) {
        context = captureContext();
      } else {
        const age = Date.now() - new Date(context.timestamp).getTime();
        if (age > 30000) {
          context = captureContext();
        }
      }

      return {
        contents: [{
          uri: 'memory://context',
          mimeType: 'application/json',
          text: JSON.stringify({
            ...context,
            injectionString: formatContextForInjection(context),
            hint: 'Use injectionString as context prefix when responding to user',
          }, null, 2),
        }],
      };
    } catch (error) {
      return {
        contents: [{
          uri: 'memory://context',
          mimeType: 'application/json',
          text: JSON.stringify({ error: 'Failed to capture context' }),
        }],
      };
    }
  }
);

// ============================================================================
// TOOLS
// ============================================================================

// --- INGESTION ---

server.tool(
  'ingest',
  'Add new knowledge to the memory palace',
  IngestInputSchema.shape,
  async (args) => {
    try {
      const input = IngestInputSchema.parse(args);

      const node = db.insertNode({
        content: input.content,
        sourceType: input.source,
        sourcePlatform: input.platform,
        sourceId: input.sourceId,
        sourceUrl: input.sourceUrl,
        createdAt: input.timestamp ? new Date(input.timestamp) : new Date(),
        updatedAt: new Date(),
        lastAccessedAt: new Date(),
        accessCount: 0,
        retentionStrength: 1.0,
        stabilityFactor: 1.0,  // New memories start with stability=1 (fast decay)
        // sentimentIntensity auto-calculated from content in insertNode
        reviewCount: 0,
        confidence: 0.8,
        isContradicted: false,
        contradictionIds: [],
        people: input.people || [],
        concepts: [],
        events: [],
        tags: input.tags || [],
        sourceChain: [],
      });

      // Generate embedding if service is available
      if (embeddingService && vectorStore) {
        try {
          if (await embeddingService.isAvailable()) {
            const embedding = await embeddingService.generateEmbedding(node.content);
            await vectorStore.upsertEmbedding(node.id, embedding, node.content, {
              sourceType: node.sourceType,
              sourcePlatform: node.sourcePlatform,
              createdAt: node.createdAt.toISOString(),
            });
            mcpLogger.debug('Generated embedding for new node', { nodeId: node.id });
          }
        } catch (embeddingError) {
          // Log but don't fail the ingest
          mcpLogger.warn('Failed to generate embedding', {
            nodeId: node.id,
            error: String(embeddingError)
          });
        }
      }

      return safeResponse({
        success: true,
        nodeId: node.id,
        message: `Knowledge ingested successfully. Node ID: ${node.id}`,
        embeddingGenerated: embeddingService ? await embeddingService.isAvailable() : false,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- RETRIEVAL ---

server.tool(
  'recall',
  'Search and retrieve knowledge from memory',
  {
    query: z.string().describe('Search query'),
    limit: z.number().min(1).max(100).optional().default(10).describe('Maximum results'),
    offset: z.number().min(0).optional().default(0).describe('Offset for pagination'),
  },
  async (args) => {
    try {
      const { query, limit, offset } = args as { query: string; limit: number; offset: number };

      let searchMethod = 'fts'; // Full-text search
      let result = db.searchNodes(query, { limit, offset });

      // Try semantic search first if available and FTS returns few results
      if (embeddingService && vectorStore && result.items.length < limit / 2) {
        try {
          if (await embeddingService.isAvailable()) {
            const queryEmbedding = await embeddingService.generateEmbedding(query);
            const semanticResults = await vectorStore.findSimilar(queryEmbedding, limit);

            if (semanticResults.length > 0) {
              // Get full nodes for semantic results
              const semanticNodeIds = new Set(semanticResults.map(r => r.id));
              const existingIds = new Set(result.items.map(n => n.id));

              // Add semantic results that aren't already in FTS results
              for (const semanticResult of semanticResults) {
                if (!existingIds.has(semanticResult.id)) {
                  const node = db.getNode(semanticResult.id);
                  if (node) {
                    result.items.push(node);
                  }
                }
              }

              searchMethod = 'hybrid';
            }
          }
        } catch (semanticError) {
          mcpLogger.debug('Semantic search fallback failed', { error: String(semanticError) });
        }
      }

      // Update access timestamps for retrieved nodes
      for (const node of result.items) {
        try {
          db.updateNodeAccess(node.id);
        } catch {
          // Ignore access update errors
        }
      }

      const formatted = result.items.map(node => ({
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
          sentimentIntensity: node.sentimentIntensity,  // How emotional was this memory?
          confidence: node.confidence,
        },
        // Git-Blame for Thoughts: What code were you working on when you had this thought?
        gitContext: node.gitContext ? {
          branch: node.gitContext.branch,
          commit: node.gitContext.commit,
          message: node.gitContext.commitMessage,
          dirty: node.gitContext.dirty,
          changedFiles: node.gitContext.changedFiles,
        } : undefined,
        people: node.people,
        tags: node.tags,
      }));

      return safeResponse({
        query,
        total: result.total,
        showing: result.items.length,
        offset: result.offset,
        hasMore: result.hasMore,
        searchMethod,
        results: formatted,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'get_knowledge',
  'Get a specific knowledge node by ID',
  { nodeId: z.string().describe('The ID of the knowledge node to retrieve') },
  async (args) => {
    try {
      const { nodeId } = args as { nodeId: string };

      // Try cache first
      const cached = nodeCache.get(CACHE_KEYS.node(nodeId));
      if (cached) {
        db.updateNodeAccess(nodeId);
        return safeResponse(cached);
      }

      const node = db.getNode(nodeId);
      if (!node) {
        return safeResponse({ error: 'Node not found', nodeId });
      }

      db.updateNodeAccess(nodeId);

      // Cache the node
      nodeCache.set(CACHE_KEYS.node(nodeId), node);

      return safeResponse(node);
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'get_related',
  'Find knowledge related to a specific node',
  {
    nodeId: z.string().describe('The ID of the knowledge node'),
    depth: z.number().min(1).max(3).optional().default(1).describe('How many hops to traverse'),
  },
  async (args) => {
    try {
      const { nodeId, depth } = args as { nodeId: string; depth: number };

      const relatedIds = db.getRelatedNodes(nodeId, depth);
      const relatedNodes = relatedIds
        .map(id => db.getNode(id))
        .filter((n): n is KnowledgeNode => n !== null);

      return safeResponse({
        sourceNode: nodeId,
        depth,
        relatedCount: relatedNodes.length,
        related: relatedNodes.map(n => ({
          id: n.id,
          summary: n.summary || n.content.slice(0, 200),
          tags: n.tags,
        })),
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- SEMANTIC SEARCH ---

server.tool(
  'semantic_search',
  'Search memories using semantic similarity (requires embeddings)',
  {
    query: z.string().describe('Search query'),
    limit: z.number().min(1).max(50).optional().default(10).describe('Maximum results'),
  },
  async (args) => {
    try {
      const { query, limit } = args as { query: string; limit: number };

      if (!embeddingService || !await embeddingService.isAvailable()) {
        return safeResponse({
          error: 'Embedding service not available',
          hint: 'Install Ollama and run: ollama pull nomic-embed-text',
        });
      }

      if (!vectorStore) {
        return safeResponse({
          error: 'Vector store not available',
        });
      }

      const embedding = await embeddingService.generateEmbedding(query);
      const similar = await vectorStore.findSimilar(embedding, limit);

      // Get full nodes for results
      const results = await Promise.all(
        similar.map(async (s) => {
          const node = db.getNode(s.id);
          if (!node) return null;

          // Update access
          try {
            db.updateNodeAccess(s.id);
          } catch {
            // Ignore
          }

          return {
            id: s.id,
            similarity: s.similarity,
            content: node.content,
            summary: node.summary || node.content.slice(0, 200),
            source: {
              type: node.sourceType,
              platform: node.sourcePlatform,
            },
            tags: node.tags,
          };
        })
      );

      return safeResponse({
        query,
        method: 'semantic',
        results: results.filter(Boolean),
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- PEOPLE MEMORY ---

server.tool(
  'remember_person',
  'Add or update a person in your relationship memory',
  {
    name: z.string().describe('Person\'s name'),
    howWeMet: z.string().optional().describe('How you met this person'),
    relationshipType: z.string().optional().describe('Type of relationship (colleague, friend, mentor, etc.)'),
    organization: z.string().optional().describe('Their organization/company'),
    role: z.string().optional().describe('Their role/title'),
    email: z.string().optional().describe('Email address'),
    notes: z.string().optional().describe('Any notes about this person'),
    sharedTopics: z.array(z.string()).optional().describe('Topics you share interest in'),
  },
  async (args) => {
    try {
      const input = args as {
        name: string;
        howWeMet?: string;
        relationshipType?: string;
        organization?: string;
        role?: string;
        email?: string;
        notes?: string;
        sharedTopics?: string[];
      };

      // Check if person exists
      const existing = db.getPersonByName(input.name);
      if (existing) {
        return safeResponse({
          message: `Person "${input.name}" already exists`,
          personId: existing.id,
          existing: true,
        });
      }

      const person = db.insertPerson({
        name: input.name,
        aliases: [],
        howWeMet: input.howWeMet,
        relationshipType: input.relationshipType,
        organization: input.organization,
        role: input.role,
        email: input.email,
        notes: input.notes,
        sharedTopics: input.sharedTopics || [],
        sharedProjects: [],
        socialLinks: {},
        contactFrequency: 0,
        relationshipHealth: 0.5,
        createdAt: new Date(),
        updatedAt: new Date(),
      });

      return safeResponse({
        success: true,
        personId: person.id,
        message: `Remembered ${input.name}`,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'get_person',
  'Get information about a person from your memory',
  { name: z.string().describe('Person\'s name to look up') },
  async (args) => {
    try {
      const { name } = args as { name: string };

      const person = db.getPersonByName(name);
      if (!person) {
        return safeResponse({
          found: false,
          message: `No person named "${name}" found in memory`,
        });
      }

      const daysSinceContact = person.lastContactAt
        ? Math.floor((Date.now() - person.lastContactAt.getTime()) / (1000 * 60 * 60 * 24))
        : null;

      return safeResponse({
        found: true,
        person: {
          ...person,
          daysSinceContact,
        },
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- TEMPORAL / REVIEW ---

server.tool(
  'mark_reviewed',
  'Mark knowledge as reviewed with FSRS (reinforces memory, slows decay)',
  {
    nodeId: z.string().describe('The ID of the knowledge node'),
    grade: z.number().min(1).max(4).optional().default(3).describe('Review grade: 1=Again, 2=Hard, 3=Good, 4=Easy'),
  },
  async (args) => {
    try {
      const { nodeId, grade } = args as { nodeId: string; grade: number };

      const nodeBefore = db.getNode(nodeId);
      if (!nodeBefore) {
        return safeResponse({ error: 'Node not found' });
      }

      // Get current FSRS state or create new one
      let currentState = fsrsScheduler.newCard();

      // If we have previous review data, reconstruct state
      if (nodeBefore.reviewCount > 0 && nodeBefore.lastAccessedAt) {
        currentState = {
          ...currentState,
          reps: nodeBefore.reviewCount,
          lastReview: nodeBefore.lastAccessedAt,
          state: 'Review',
          // Estimate stability from retention strength
          stability: nodeBefore.stabilityFactor || 1,
        };
      }

      // Calculate elapsed days since last review
      const elapsedDays = nodeBefore.lastAccessedAt
        ? (Date.now() - nodeBefore.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24)
        : 0;

      // Apply FSRS review
      const reviewResult = fsrsScheduler.review(
        currentState,
        grade as ReviewGrade,
        elapsedDays,
        nodeBefore.sentimentIntensity // Apply sentiment boost
      );

      // Update node with FSRS results
      db.markReviewed(nodeId);

      // Update stability factor based on FSRS
      const internalDb = (db as unknown as { db: { prepare: (sql: string) => { run: (...args: unknown[]) => void } } }).db;
      internalDb.prepare(`
        UPDATE knowledge_nodes
        SET stability_factor = ?,
            next_review_date = ?
        WHERE id = ?
      `).run(
        reviewResult.state.stability,
        new Date(Date.now() + reviewResult.interval * 24 * 60 * 60 * 1000).toISOString(),
        nodeId
      );

      const nodeAfter = db.getNode(nodeId);

      // Invalidate cache
      invalidateNodeCaches(nodeId);

      return safeResponse({
        success: true,
        nodeId,
        grade: ['Again', 'Hard', 'Good', 'Easy'][grade - 1],
        fsrs: {
          newStability: reviewResult.state.stability,
          newDifficulty: reviewResult.state.difficulty,
          retrievability: reviewResult.retrievability,
          nextInterval: reviewResult.interval,
          isLapse: reviewResult.isLapse,
        },
        previousRetention: nodeBefore.retentionStrength,
        newRetention: nodeAfter?.retentionStrength,
        reviewCount: nodeAfter?.reviewCount,
        nextReviewDays: reviewResult.interval,
        message: 'Memory reinforced with FSRS',
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- CONSOLIDATION ---

server.tool(
  'run_consolidation',
  'Run sleep consolidation cycle to optimize memories',
  {},
  async () => {
    try {
      const result = await runConsolidation(db, {
        shortTermWindowHours: config.consolidation.shortTermWindowHours,
        importanceThreshold: config.consolidation.importanceThreshold,
        pruneThreshold: config.consolidation.pruneThreshold,
        maxAnalyze: config.rem.maxAnalyze,
      });

      return safeResponse({
        success: true,
        shortTermProcessed: result.shortTermProcessed,
        promoted: result.promotedToLongTerm,
        connections: result.connectionsDiscovered,
        pruned: result.edgesPruned,
        decayed: result.decayApplied,
        duration: `${result.duration}ms`,
        message: 'Consolidation cycle complete',
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- MEMORY STATS ---

server.tool(
  'get_memory_stats',
  'Get detailed statistics about memory health and distribution',
  {},
  async () => {
    try {
      const stats = db.getStats();
      const health = db.checkHealth();

      // Get retention strength distribution
      type SqliteStatement = { all: () => unknown[] };
      const internalDb = (db as unknown as { db: { prepare: (sql: string) => SqliteStatement } }).db;

      const retentionDist = internalDb.prepare(`
        SELECT
          CASE
            WHEN retention_strength >= 0.8 THEN 'strong'
            WHEN retention_strength >= 0.5 THEN 'moderate'
            WHEN retention_strength >= 0.3 THEN 'weak'
            ELSE 'critical'
          END as bucket,
          COUNT(*) as count
        FROM knowledge_nodes
        GROUP BY bucket
        ORDER BY
          CASE bucket
            WHEN 'strong' THEN 1
            WHEN 'moderate' THEN 2
            WHEN 'weak' THEN 3
            ELSE 4
          END
      `).all() as Array<{ bucket: string; count: number }>;

      // Get FSRS state distribution
      const stabilityDist = internalDb.prepare(`
        SELECT
          CASE
            WHEN stability_factor >= 30 THEN 'stable'
            WHEN stability_factor >= 7 THEN 'learning'
            WHEN stability_factor >= 1 THEN 'new'
            ELSE 'lapsed'
          END as bucket,
          COUNT(*) as count
        FROM knowledge_nodes
        GROUP BY bucket
      `).all() as Array<{ bucket: string; count: number }>;

      // Get edge statistics
      const edgeStatsRows = internalDb.prepare(`
        SELECT
          COUNT(*) as total,
          AVG(weight) as avg_weight,
          SUM(CASE WHEN json_extract(metadata, '$.discoveredBy') = 'rem_cycle' THEN 1 ELSE 0 END) as auto_discovered
        FROM graph_edges
      `).all() as Array<{ total: number; avg_weight: number; auto_discovered: number }>;
      const edgeStats = edgeStatsRows[0];

      // Get vector store stats if available
      let vectorStats = null;
      if (vectorStore) {
        try {
          vectorStats = await vectorStore.getStats();
        } catch {
          // Ignore
        }
      }

      // Get job queue stats if available
      let jobStats = null;
      if (jobQueue) {
        jobStats = jobQueue.getStats();
      }

      return safeResponse({
        overview: {
          totalNodes: stats.totalNodes,
          totalPeople: stats.totalPeople,
          totalConnections: stats.totalEdges,
          databaseSize: db.getDatabaseSize().formatted,
        },
        health: {
          status: health.status,
          warnings: health.warnings,
        },
        retention: {
          distribution: retentionDist,
        },
        stability: {
          distribution: stabilityDist,
        },
        connections: {
          total: edgeStats?.total || 0,
          averageWeight: edgeStats?.avg_weight || 0,
          autoDiscovered: edgeStats?.auto_discovered || 0,
        },
        vectorStore: vectorStats,
        jobQueue: jobStats,
        embeddingsAvailable: embeddingService ? await embeddingService.isAvailable() : false,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- DAILY BRIEF ---

server.tool(
  'daily_brief',
  'Get your daily knowledge brief',
  {},
  async () => {
    try {
      const stats = db.getStats();
      const health = db.checkHealth();
      const decaying = db.getDecayingNodes(0.5, { limit: 5 });
      const reconnect = db.getPeopleToReconnect(30, { limit: 5 });
      const recent = db.getRecentNodes({ limit: 5 });

      const brief = {
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
        reviewNeeded: decaying.items.map(n => ({
          id: n.id,
          preview: n.summary || n.content.slice(0, 100),
          retentionStrength: n.retentionStrength,
          daysSinceAccess: Math.floor(
            (Date.now() - n.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24)
          ),
        })),
        peopleToReconnect: reconnect.items.map(p => ({
          name: p.name,
          daysSinceContact: p.lastContactAt
            ? Math.floor((Date.now() - p.lastContactAt.getTime()) / (1000 * 60 * 60 * 24))
            : null,
          sharedTopics: p.sharedTopics,
        })),
        recentlyAdded: recent.items.map(n => ({
          id: n.id,
          preview: n.summary || n.content.slice(0, 100),
          source: n.sourcePlatform,
        })),
      };

      return safeResponse(brief);
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// --- HEALTH & MAINTENANCE ---

server.tool(
  'health_check',
  'Get detailed health status of the memory database',
  {},
  async () => {
    try {
      const health = db.checkHealth();
      const size = db.getDatabaseSize();

      return safeResponse({
        ...health,
        databaseSize: size,
        recommendations: getHealthRecommendations(health),
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'backup',
  'Create a backup of the memory database',
  {},
  async () => {
    try {
      const backupPath = db.backup();
      const backups = db.listBackups();

      return safeResponse({
        success: true,
        backupPath,
        message: 'Backup created successfully',
        totalBackups: backups.length,
        backups: backups.slice(0, 5).map(b => ({
          path: b.path,
          size: `${(b.size / 1024 / 1024).toFixed(2)}MB`,
          date: b.date.toISOString(),
        })),
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'list_backups',
  'List available database backups',
  {},
  async () => {
    try {
      const backups = db.listBackups();

      return safeResponse({
        totalBackups: backups.length,
        backups: backups.map(b => ({
          path: b.path,
          size: `${(b.size / 1024 / 1024).toFixed(2)}MB`,
          date: b.date.toISOString(),
        })),
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'optimize_database',
  'Optimize the database (vacuum, reindex) - use sparingly',
  {},
  async () => {
    try {
      const sizeBefore = db.getDatabaseSize();
      db.optimize();
      const sizeAfter = db.getDatabaseSize();

      return safeResponse({
        success: true,
        message: 'Database optimized',
        sizeBefore: sizeBefore.formatted,
        sizeAfter: sizeAfter.formatted,
        spaceSaved: `${(sizeBefore.mb - sizeAfter.mb).toFixed(2)}MB`,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

server.tool(
  'apply_decay',
  'Apply memory decay based on time since last access',
  {},
  async () => {
    try {
      const updatedCount = db.applyDecay();

      return safeResponse({
        success: true,
        nodesUpdated: updatedCount,
        message: `Applied decay to ${updatedCount} knowledge nodes`,
      });
    } catch (error) {
      return errorResponse(error);
    }
  }
);

// ============================================================================
// HELPERS
// ============================================================================

function getTimeBasedGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 17) return 'Good afternoon';
  return 'Good evening';
}

function getHealthRecommendations(health: ReturnType<typeof db.checkHealth>): string[] {
  const recommendations: string[] = [];

  if (health.status === 'critical') {
    recommendations.push('CRITICAL: Immediate attention required. Check warnings for details.');
  }

  if (!health.lastBackup) {
    recommendations.push('Create your first backup using the backup tool');
  } else {
    const daysSinceBackup = (Date.now() - new Date(health.lastBackup).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceBackup > 7) {
      recommendations.push(`Consider creating a backup (last backup was ${Math.floor(daysSinceBackup)} days ago)`);
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
}

// ============================================================================
// GRACEFUL SHUTDOWN
// ============================================================================

async function gracefulShutdown(): Promise<void> {
  logger.info('Shutting down Vestige...');

  // Stop job queue
  if (jobQueue) {
    try {
      await jobQueue.shutdown(10000); // 10 second timeout
      logger.info('Job queue stopped');
    } catch (error) {
      logger.warn('Error stopping job queue', { error: String(error) });
    }
  }

  // Close vector store
  if (vectorStore) {
    try {
      await vectorStore.close();
      logger.info('Vector store closed');
    } catch (error) {
      logger.warn('Error closing vector store', { error: String(error) });
    }
  }

  // Destroy all caches
  destroyAllCaches();
  logger.info('Caches destroyed');

  // Close database
  db.close();
  logger.info('Database closed');

  logger.info('Vestige shutdown complete');
}

process.on('SIGINT', async () => {
  await gracefulShutdown();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await gracefulShutdown();
  process.exit(0);
});

// ============================================================================
// START SERVER
// ============================================================================

async function main() {
  // Initialize async services
  await initializeServices();

  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info('Vestige MCP server v0.3.0 running');
}

main().catch((error) => {
  logger.error('Failed to start Vestige', error instanceof Error ? error : undefined);
  db.close();
  process.exit(1);
});

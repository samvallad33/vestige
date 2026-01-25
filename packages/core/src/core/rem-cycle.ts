/**
 * REM Cycle - Nocturnal Optimization with Semantic Understanding
 *
 * "The brain that dreams while you sleep."
 *
 * This module discovers connections between unconnected knowledge nodes
 * by analyzing semantic similarity, shared concepts, keyword overlap,
 * emotional resonance, and spreading activation patterns.
 *
 * KEY FEATURES:
 * 1. Semantic Similarity - Uses embeddings for deep understanding
 * 2. Emotional Weighting - Emotionally charged memories create stronger connections
 * 3. Spreading Activation - Discovers transitive relationships (A->B->C implies A~C)
 * 4. Reconsolidation - Accessing memories strengthens their connections
 * 5. Exponential Temporal Proximity - Time-based connection strength decay
 */

import { VestigeDatabase } from './database.js';
import type { KnowledgeNode } from './types.js';
import natural from 'natural';
import {
  createEmbeddingService,
  type EmbeddingService,
  EmbeddingCache,
  cosineSimilarity,
} from './embeddings.js';

// ============================================================================
// TYPES
// ============================================================================

type ConnectionType =
  | 'concept_overlap'
  | 'keyword_similarity'
  | 'entity_shared'
  | 'temporal_proximity'
  | 'semantic_similarity'
  | 'spreading_activation';

interface DiscoveredConnection {
  nodeA: KnowledgeNode;
  nodeB: KnowledgeNode;
  reason: string;
  strength: number; // 0-1
  connectionType: ConnectionType;
}

interface REMCycleResult {
  nodesAnalyzed: number;
  connectionsDiscovered: number;
  connectionsCreated: number;
  spreadingActivationEdges: number;
  reconsolidatedNodes: number;
  duration: number;
  semanticEnabled: boolean;
  discoveries: Array<{
    nodeA: string;
    nodeB: string;
    reason: string;
    type: ConnectionType;
  }>;
}

interface REMCycleOptions {
  maxAnalyze?: number;
  minStrength?: number;
  dryRun?: boolean;
  /** Enable semantic similarity analysis (requires Ollama) */
  enableSemantic?: boolean;
  /** Run spreading activation to discover transitive connections */
  enableSpreadingActivation?: boolean;
  /** Maximum depth for spreading activation */
  spreadingActivationDepth?: number;
  /** Node IDs that were recently accessed (for reconsolidation) */
  recentlyAccessedIds?: string[];
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Temporal half-life in days for exponential proximity decay */
const TEMPORAL_HALF_LIFE_DAYS = 7;

/** Semantic similarity thresholds */
const SEMANTIC_STRONG_THRESHOLD = 0.7;
const SEMANTIC_MODERATE_THRESHOLD = 0.5;

/** Weight decay for spreading activation (per hop) */
const SPREADING_ACTIVATION_DECAY = 0.8;

/** Reconsolidation strength boost (5%) */
const RECONSOLIDATION_BOOST = 0.05;

// ============================================================================
// SIMILARITY ANALYSIS
// ============================================================================

const tokenizer = new natural.WordTokenizer();

/**
 * Extract keywords from content using TF-IDF
 */
function extractKeywords(content: string): string[] {
  const tokens = tokenizer.tokenize(content.toLowerCase()) || [];

  // Filter out common stop words and short tokens
  const stopWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
  ]);

  return tokens.filter(token =>
    token.length > 3 &&
    !stopWords.has(token) &&
    !/^\d+$/.test(token) // Filter pure numbers
  );
}

/**
 * Calculate Jaccard similarity between two keyword sets
 */
function jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);

  if (union.size === 0) return 0;
  return intersection.size / union.size;
}

/**
 * Find shared concepts between two nodes
 */
function findSharedConcepts(nodeA: KnowledgeNode, nodeB: KnowledgeNode): string[] {
  const conceptsA = new Set([...nodeA.concepts, ...nodeA.tags]);
  const conceptsB = new Set([...nodeB.concepts, ...nodeB.tags]);

  return [...conceptsA].filter(c => conceptsB.has(c));
}

/**
 * Find shared people between two nodes
 */
function findSharedPeople(nodeA: KnowledgeNode, nodeB: KnowledgeNode): string[] {
  const peopleA = new Set(nodeA.people);
  const peopleB = new Set(nodeB.people);

  return [...peopleA].filter(p => peopleB.has(p));
}

/**
 * Calculate exponential temporal proximity weight
 * Uses half-life decay instead of binary same-day check
 */
function calculateTemporalProximity(nodeA: KnowledgeNode, nodeB: KnowledgeNode): number {
  const msPerDay = 24 * 60 * 60 * 1000;
  const diffMs = Math.abs(nodeA.createdAt.getTime() - nodeB.createdAt.getTime());
  const daysBetween = diffMs / msPerDay;

  // Exponential decay: weight = e^(-t/half_life)
  // At t=0: weight = 1.0
  // At t=half_life: weight = 0.5
  // At t=2*half_life: weight = 0.25
  return Math.exp(-daysBetween / TEMPORAL_HALF_LIFE_DAYS);
}

/**
 * Calculate emotional resonance between two nodes
 * Returns a boost multiplier (1.0 to 1.5) based on combined emotional intensity
 */
function calculateEmotionalBoost(nodeA: KnowledgeNode, nodeB: KnowledgeNode): number {
  const emotionalA = nodeA.sentimentIntensity || 0;
  const emotionalB = nodeB.sentimentIntensity || 0;

  // Average emotional intensity
  const emotionalResonance = (emotionalA + emotionalB) / 2;

  // Up to 1.5x boost for highly emotional content
  return 1 + (emotionalResonance * 0.5);
}

// ============================================================================
// SEMANTIC ANALYSIS
// ============================================================================

/**
 * Analyze semantic connection between two nodes using embeddings
 */
async function analyzeSemanticConnection(
  nodeA: KnowledgeNode,
  nodeB: KnowledgeNode,
  embeddingService: EmbeddingService,
  cache: EmbeddingCache
): Promise<DiscoveredConnection | null> {
  try {
    // Get or generate embeddings
    let embeddingA = cache.get(nodeA.id);
    let embeddingB = cache.get(nodeB.id);

    // Generate missing embeddings
    if (!embeddingA) {
      embeddingA = await embeddingService.generateEmbedding(nodeA.content);
      cache.set(nodeA.id, embeddingA);
    }

    if (!embeddingB) {
      embeddingB = await embeddingService.generateEmbedding(nodeB.content);
      cache.set(nodeB.id, embeddingB);
    }

    // Calculate cosine similarity
    const similarity = cosineSimilarity(embeddingA, embeddingB);

    // Apply emotional boost
    const emotionalBoost = calculateEmotionalBoost(nodeA, nodeB);
    const boostedSimilarity = Math.min(1, similarity * emotionalBoost);

    // Strong semantic connection
    if (similarity >= SEMANTIC_STRONG_THRESHOLD) {
      return {
        nodeA,
        nodeB,
        reason: `Strong semantic similarity (${(similarity * 100).toFixed(0)}%)`,
        strength: Math.min(1, boostedSimilarity + 0.2), // Boost for strong connections
        connectionType: 'semantic_similarity',
      };
    }

    // Moderate semantic connection
    if (similarity >= SEMANTIC_MODERATE_THRESHOLD) {
      return {
        nodeA,
        nodeB,
        reason: `Moderate semantic similarity (${(similarity * 100).toFixed(0)}%)`,
        strength: boostedSimilarity,
        connectionType: 'semantic_similarity',
      };
    }

    return null;
  } catch {
    // If embedding fails, return null to fall back to traditional analysis
    return null;
  }
}

// ============================================================================
// TRADITIONAL ANALYSIS (FALLBACK)
// ============================================================================

/**
 * Analyze potential connection between two nodes using traditional methods
 * Used as fallback when embeddings are unavailable
 */
function analyzeTraditionalConnection(
  nodeA: KnowledgeNode,
  nodeB: KnowledgeNode
): DiscoveredConnection | null {
  // Extract keywords from both nodes
  const keywordsA = new Set(extractKeywords(nodeA.content));
  const keywordsB = new Set(extractKeywords(nodeB.content));

  // Calculate keyword similarity
  const keywordSim = jaccardSimilarity(keywordsA, keywordsB);

  // Find shared concepts/tags
  const sharedConcepts = findSharedConcepts(nodeA, nodeB);

  // Find shared people
  const sharedPeople = findSharedPeople(nodeA, nodeB);

  // Calculate temporal proximity weight
  const temporalWeight = calculateTemporalProximity(nodeA, nodeB);

  // Calculate emotional boost
  const emotionalBoost = calculateEmotionalBoost(nodeA, nodeB);

  // Determine if there's a meaningful connection
  // Priority: shared entities > concept overlap > keyword similarity > temporal

  if (sharedPeople.length > 0) {
    const baseStrength = Math.min(1, 0.5 + sharedPeople.length * 0.2);
    return {
      nodeA,
      nodeB,
      reason: `Shared people: ${sharedPeople.join(', ')}`,
      strength: Math.min(1, baseStrength * emotionalBoost),
      connectionType: 'entity_shared',
    };
  }

  if (sharedConcepts.length >= 2) {
    const baseStrength = Math.min(1, 0.4 + sharedConcepts.length * 0.15);
    return {
      nodeA,
      nodeB,
      reason: `Shared concepts: ${sharedConcepts.slice(0, 3).join(', ')}`,
      strength: Math.min(1, baseStrength * emotionalBoost),
      connectionType: 'concept_overlap',
    };
  }

  if (keywordSim > 0.15) {
    // Find the actual overlapping keywords
    const overlap = [...keywordsA].filter(k => keywordsB.has(k)).slice(0, 5);
    const baseStrength = Math.min(1, keywordSim * 2);
    return {
      nodeA,
      nodeB,
      reason: `Keyword overlap (${(keywordSim * 100).toFixed(0)}%): ${overlap.join(', ')}`,
      strength: Math.min(1, baseStrength * emotionalBoost),
      connectionType: 'keyword_similarity',
    };
  }

  // Temporal proximity with related content
  if (temporalWeight > 0.5 && (sharedConcepts.length > 0 || keywordSim > 0.05)) {
    const baseStrength = 0.3 + (temporalWeight - 0.5) * 0.4; // Scale 0.3-0.5
    return {
      nodeA,
      nodeB,
      reason: `Created ${Math.round((1 - temporalWeight) * TEMPORAL_HALF_LIFE_DAYS * 2)} days apart with related content`,
      strength: Math.min(1, baseStrength * emotionalBoost),
      connectionType: 'temporal_proximity',
    };
  }

  return null;
}

// ============================================================================
// SPREADING ACTIVATION
// ============================================================================

interface SpreadingActivationResult {
  edgesCreated: number;
  paths: Array<{
    from: string;
    via: string;
    to: string;
    weight: number;
  }>;
}

/**
 * Apply spreading activation to discover transitive connections
 * If A -> B and B -> C exist, creates A -> C with decayed weight
 */
function applySpreadingActivation(
  db: VestigeDatabase,
  maxDepth: number = 2,
  minWeight: number = 0.2
): SpreadingActivationResult {
  const result: SpreadingActivationResult = {
    edgesCreated: 0,
    paths: [],
  };

  // Get all existing edges
  const edges = db['db'].prepare(`
    SELECT from_id, to_id, weight FROM graph_edges
    WHERE edge_type = 'similar_to'
  `).all() as { from_id: string; to_id: string; weight: number }[];

  // Build adjacency map (bidirectional)
  const adjacency = new Map<string, Map<string, number>>();

  for (const edge of edges) {
    // Forward direction
    if (!adjacency.has(edge.from_id)) {
      adjacency.set(edge.from_id, new Map());
    }
    adjacency.get(edge.from_id)!.set(edge.to_id, edge.weight);

    // Reverse direction (treat as undirected)
    if (!adjacency.has(edge.to_id)) {
      adjacency.set(edge.to_id, new Map());
    }
    adjacency.get(edge.to_id)!.set(edge.from_id, edge.weight);
  }

  // Find existing direct connections (to avoid duplicates)
  const existingConnections = new Set<string>();
  for (const edge of edges) {
    existingConnections.add(`${edge.from_id}-${edge.to_id}`);
    existingConnections.add(`${edge.to_id}-${edge.from_id}`);
  }

  // For each node, find 2-hop paths
  const newConnections: Array<{
    from: string;
    to: string;
    via: string;
    weight: number;
  }> = [];

  for (const [nodeA, neighborsA] of adjacency) {
    for (const [nodeB, weightAB] of neighborsA) {
      const neighborsB = adjacency.get(nodeB);
      if (!neighborsB) continue;

      for (const [nodeC, weightBC] of neighborsB) {
        // Skip if A == C or if direct connection already exists
        if (nodeA === nodeC) continue;

        const connectionKey = `${nodeA}-${nodeC}`;
        const reverseKey = `${nodeC}-${nodeA}`;

        if (existingConnections.has(connectionKey) || existingConnections.has(reverseKey)) {
          continue;
        }

        // Calculate transitive weight with decay
        const transitiveWeight = weightAB * weightBC * SPREADING_ACTIVATION_DECAY;

        if (transitiveWeight >= minWeight) {
          newConnections.push({
            from: nodeA,
            to: nodeC,
            via: nodeB,
            weight: transitiveWeight,
          });

          // Mark as existing to avoid duplicates
          existingConnections.add(connectionKey);
          existingConnections.add(reverseKey);
        }
      }
    }
  }

  // Create the new edges
  for (const conn of newConnections) {
    try {
      db.insertEdge({
        fromId: conn.from,
        toId: conn.to,
        edgeType: 'similar_to',
        weight: conn.weight,
        metadata: {
          discoveredBy: 'spreading_activation',
          viaNode: conn.via,
          connectionType: 'spreading_activation',
        },
        createdAt: new Date(),
      });

      result.edgesCreated++;
      result.paths.push(conn);
    } catch {
      // Edge might already exist, skip
    }
  }

  return result;
}

// ============================================================================
// RECONSOLIDATION
// ============================================================================

/**
 * Strengthen connections for recently accessed nodes
 * Implements memory reconsolidation - accessing memories makes them stronger
 */
function reconsolidateConnections(db: VestigeDatabase, nodeId: string): number {
  let strengthened = 0;

  try {
    // Get all edges involving this node
    const edges = db['db'].prepare(`
      SELECT id, weight FROM graph_edges
      WHERE from_id = ? OR to_id = ?
    `).all(nodeId, nodeId) as { id: string; weight: number }[];

    // Strengthen each edge by RECONSOLIDATION_BOOST (5%)
    const updateStmt = db['db'].prepare(`
      UPDATE graph_edges
      SET weight = MIN(1.0, weight * ?)
      WHERE id = ?
    `);

    for (const edge of edges) {
      const newWeight = Math.min(1.0, edge.weight * (1 + RECONSOLIDATION_BOOST));
      if (newWeight > edge.weight) {
        updateStmt.run(newWeight, edge.id);
        strengthened++;
      }
    }
  } catch {
    // Reconsolidation is optional, don't fail the cycle
  }

  return strengthened;
}

// ============================================================================
// REM CYCLE MAIN LOGIC
// ============================================================================

/**
 * Get nodes that have few or no connections
 */
function getDisconnectedNodes(db: VestigeDatabase, maxEdges: number = 1): KnowledgeNode[] {
  // Get all nodes
  const result = db.getRecentNodes({ limit: 500 });
  const allNodes = result.items;

  // Filter to nodes with few connections
  const disconnected: KnowledgeNode[] = [];

  for (const node of allNodes) {
    const related = db.getRelatedNodes(node.id, 1);
    if (related.length <= maxEdges) {
      disconnected.push(node);
    }
  }

  return disconnected;
}

/**
 * Run one REM cycle - discover and create connections
 *
 * The cycle performs these steps:
 * 1. Reconsolidate recently accessed nodes (strengthen existing connections)
 * 2. Find disconnected nodes
 * 3. Try semantic similarity first (if enabled and available)
 * 4. Fall back to traditional analysis (Jaccard, shared concepts, etc.)
 * 5. Apply emotional weighting to all connections
 * 6. Run spreading activation to find transitive connections
 */
export async function runREMCycle(
  db: VestigeDatabase,
  options: REMCycleOptions = {}
): Promise<REMCycleResult> {
  const startTime = Date.now();
  const {
    maxAnalyze = 50,
    minStrength = 0.3,
    dryRun = false,
    enableSemantic = true,
    enableSpreadingActivation = true,
    spreadingActivationDepth = 2,
    recentlyAccessedIds = [],
  } = options;

  const result: REMCycleResult = {
    nodesAnalyzed: 0,
    connectionsDiscovered: 0,
    connectionsCreated: 0,
    spreadingActivationEdges: 0,
    reconsolidatedNodes: 0,
    duration: 0,
    semanticEnabled: false,
    discoveries: [],
  };

  // Step 1: Reconsolidate recently accessed nodes
  if (!dryRun && recentlyAccessedIds.length > 0) {
    for (const nodeId of recentlyAccessedIds) {
      const strengthened = reconsolidateConnections(db, nodeId);
      if (strengthened > 0) {
        result.reconsolidatedNodes++;
      }
    }
  }

  // Step 2: Initialize embedding service if semantic analysis is enabled
  let embeddingService: EmbeddingService | null = null;
  let embeddingCache: EmbeddingCache | null = null;

  if (enableSemantic) {
    try {
      embeddingService = await createEmbeddingService();
      const isAvailable = await embeddingService.isAvailable();
      result.semanticEnabled = isAvailable;

      if (isAvailable) {
        embeddingCache = new EmbeddingCache(500, 30); // 500 entries, 30 min TTL
      }
    } catch {
      // Semantic analysis not available, continue without it
      result.semanticEnabled = false;
    }
  }

  // Step 3: Get disconnected nodes
  const disconnected = getDisconnectedNodes(db, 2);

  if (disconnected.length < 2) {
    result.duration = Date.now() - startTime;
    return result;
  }

  // Limit analysis
  const toAnalyze = disconnected.slice(0, maxAnalyze);
  result.nodesAnalyzed = toAnalyze.length;

  // Step 4: Compare pairs
  const discoveries: DiscoveredConnection[] = [];
  const analyzed = new Set<string>();

  for (let i = 0; i < toAnalyze.length; i++) {
    for (let j = i + 1; j < toAnalyze.length; j++) {
      const nodeA = toAnalyze[i];
      const nodeB = toAnalyze[j];

      if (!nodeA || !nodeB) continue;

      // Skip if already have an edge
      const pairKey = [nodeA.id, nodeB.id].sort().join('-');
      if (analyzed.has(pairKey)) continue;
      analyzed.add(pairKey);

      let connection: DiscoveredConnection | null = null;

      // Try semantic similarity first if available
      if (result.semanticEnabled && embeddingService && embeddingCache) {
        connection = await analyzeSemanticConnection(
          nodeA,
          nodeB,
          embeddingService,
          embeddingCache
        );
      }

      // Fall back to traditional analysis if no semantic connection found
      if (!connection) {
        connection = analyzeTraditionalConnection(nodeA, nodeB);
      }

      if (connection && connection.strength >= minStrength) {
        discoveries.push(connection);
      }
    }
  }

  result.connectionsDiscovered = discoveries.length;

  // Step 5: Create edges for discovered connections
  if (!dryRun) {
    for (const discovery of discoveries) {
      try {
        db.insertEdge({
          fromId: discovery.nodeA.id,
          toId: discovery.nodeB.id,
          edgeType: 'similar_to',
          weight: discovery.strength,
          metadata: {
            discoveredBy: 'rem_cycle',
            reason: discovery.reason,
            connectionType: discovery.connectionType,
          },
          createdAt: new Date(),
        });
        result.connectionsCreated++;

        result.discoveries.push({
          nodeA: discovery.nodeA.content.slice(0, 50),
          nodeB: discovery.nodeB.content.slice(0, 50),
          reason: discovery.reason,
          type: discovery.connectionType,
        });
      } catch {
        // Edge might already exist
      }
    }

    // Step 6: Apply spreading activation
    if (enableSpreadingActivation) {
      const spreadingResult = applySpreadingActivation(db, spreadingActivationDepth, minStrength);
      result.spreadingActivationEdges = spreadingResult.edgesCreated;

      // Add spreading activation discoveries to results
      for (const path of spreadingResult.paths) {
        result.discoveries.push({
          nodeA: path.from.slice(0, 20),
          nodeB: path.to.slice(0, 20),
          reason: `Transitive via ${path.via.slice(0, 20)} (${(path.weight * 100).toFixed(0)}%)`,
          type: 'spreading_activation',
        });
      }
    }
  } else {
    // Dry run - just record discoveries
    for (const discovery of discoveries) {
      result.discoveries.push({
        nodeA: discovery.nodeA.content.slice(0, 50),
        nodeB: discovery.nodeB.content.slice(0, 50),
        reason: discovery.reason,
        type: discovery.connectionType,
      });
    }
  }

  result.duration = Date.now() - startTime;
  return result;
}

/**
 * Get a summary of potential discoveries without creating edges
 */
export async function previewREMCycle(db: VestigeDatabase): Promise<REMCycleResult> {
  return runREMCycle(db, { dryRun: true, maxAnalyze: 100 });
}

/**
 * Trigger reconsolidation for a specific node
 * Call this when a node is accessed to strengthen its connections
 */
export function triggerReconsolidation(db: VestigeDatabase, nodeId: string): number {
  return reconsolidateConnections(db, nodeId);
}

#!/usr/bin/env node

/**
 * Vestige CLI - Management commands for the Memory Palace
 *
 * Usage:
 *   vestige stats           - Show knowledge base statistics and health
 *   vestige health          - Detailed health check
 *   vestige review          - Start a review session
 *   vestige people          - List people in your network
 *   vestige backup          - Create a backup
 *   vestige backups         - List available backups
 *   vestige restore <path>  - Restore from a backup
 *   vestige optimize        - Optimize the database
 *   vestige decay           - Apply memory decay
 *   vestige eat <url|path>  - Ingest documentation/content (Man Page Absorber)
 */

import { VestigeDatabase, VestigeDatabaseError } from './core/database.js';
import {
  captureContext,
  formatContextForInjection,
  startContextWatcher,
  readSavedContext,
} from './core/context-watcher.js';
import { runREMCycle, previewREMCycle } from './core/rem-cycle.js';
import { ShadowSelf, runShadowCycle } from './core/shadow-self.js';
import {
  validatePath,
  validateUrl,
  sanitizeContent,
  logSecurityEvent,
  MAX_CONTENT_LENGTH,
} from './core/security.js';
import { runConsolidation } from './core/consolidation.js';
import { createEmbeddingService, OllamaEmbeddingService } from './core/embeddings.js';
import { getConfig, resetConfig, loadConfig } from './core/config.js';
import { createVectorStore, ChromaVectorStore } from './core/vector-store.js';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { marked } from 'marked';

// ============================================================================
// MAN PAGE ABSORBER - Feed your brain
// ============================================================================

interface ContentChunk {
  title: string;
  content: string;
  section: string;
  index: number;
}

/**
 * Fetch content from URL (with SSRF protection)
 */
async function fetchUrl(url: string): Promise<string> {
  // Validate URL to prevent SSRF attacks
  const validation = validateUrl(url);
  if (!validation.valid) {
    logSecurityEvent({
      type: 'ssrf_attempt',
      details: { error: validation.error || 'URL validation failed', url: url.slice(0, 100) },
      severity: 'high',
      blocked: true,
    });
    throw new Error(`Security: ${validation.error}`);
  }

  const safeUrl = validation.sanitizedUrl!;
  const response = await fetch(safeUrl, {
    // Add timeout to prevent hanging on slow responses
    signal: AbortSignal.timeout(30000), // 30 second timeout
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ${safeUrl}: ${response.status} ${response.statusText}`);
  }

  // Check content length to prevent DoS
  const contentLength = response.headers.get('content-length');
  if (contentLength && parseInt(contentLength, 10) > MAX_CONTENT_LENGTH) {
    throw new Error(`Content too large: ${contentLength} bytes exceeds ${MAX_CONTENT_LENGTH} byte limit`);
  }

  const contentType = response.headers.get('content-type') || '';

  let content: string;
  if (contentType.includes('text/html')) {
    // Strip HTML to get text content
    const html = await response.text();
    content = stripHtml(html);
  } else {
    content = await response.text();
  }

  // Sanitize the content
  return sanitizeContent(content);
}

/**
 * Simple HTML stripper - extracts text content
 */
function stripHtml(html: string): string {
  // Remove script and style tags and their content
  let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
  text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');

  // Remove HTML tags
  text = text.replace(/<[^>]+>/g, ' ');

  // Decode HTML entities
  text = text.replace(/&nbsp;/g, ' ');
  text = text.replace(/&amp;/g, '&');
  text = text.replace(/&lt;/g, '<');
  text = text.replace(/&gt;/g, '>');
  text = text.replace(/&quot;/g, '"');
  text = text.replace(/&#39;/g, "'");

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim();

  return text;
}

/**
 * Read content from file (with path traversal protection)
 */
async function readFile(filePath: string): Promise<string> {
  // Validate path to prevent path traversal attacks
  const validation = validatePath(filePath);
  if (!validation.valid) {
    logSecurityEvent({
      type: 'path_traversal',
      details: { error: validation.error || 'Path validation failed', path: filePath.slice(0, 100) },
      severity: 'high',
      blocked: true,
    });
    throw new Error(`Security: ${validation.error}`);
  }

  const safePath = validation.sanitizedPath!;

  if (!fs.existsSync(safePath)) {
    throw new Error(`File not found: ${safePath}`);
  }

  // Check file size before reading
  const stats = fs.statSync(safePath);
  if (stats.size > MAX_CONTENT_LENGTH) {
    throw new Error(`File too large: ${stats.size} bytes exceeds ${MAX_CONTENT_LENGTH} byte limit`);
  }

  const content = fs.readFileSync(safePath, 'utf-8');

  // Sanitize the content
  return sanitizeContent(content);
}

/**
 * Chunk content intelligently
 * - Respects markdown headers as section boundaries
 * - Creates overlapping chunks for context preservation
 * - Targets ~500-1000 tokens per chunk
 */
function chunkContent(content: string, source: string): ContentChunk[] {
  const chunks: ContentChunk[] = [];

  // Try to detect if it's markdown
  const isMarkdown = content.includes('# ') || content.includes('## ') || content.includes('```');

  if (isMarkdown) {
    // Split by headers
    const sections = content.split(/^(#{1,3} .+)$/m);
    let currentSection = 'Introduction';
    let currentContent = '';
    let chunkIndex = 0;

    for (let i = 0; i < sections.length; i++) {
      const sectionRaw = sections[i];
      if (!sectionRaw) continue;
      const section = sectionRaw.trim();
      if (!section) continue;

      // Check if this is a header
      if (section.match(/^#{1,3} /)) {
        // Save previous section if it has content
        if (currentContent.trim()) {
          chunks.push(...splitLargeSection(currentContent, currentSection, chunkIndex, source));
          chunkIndex = chunks.length;
        }
        currentSection = section.replace(/^#{1,3} /, '').trim();
        currentContent = '';
      } else {
        currentContent += section + '\n\n';
      }
    }

    // Don't forget the last section
    if (currentContent.trim()) {
      chunks.push(...splitLargeSection(currentContent, currentSection, chunkIndex, source));
    }
  } else {
    // Plain text - split by paragraphs
    const paragraphs = content.split(/\n\n+/);
    let currentChunk = '';
    let chunkIndex = 0;

    for (const para of paragraphs) {
      const trimmed = para.trim();
      if (!trimmed) continue;

      if ((currentChunk + trimmed).length > 2000) {
        if (currentChunk) {
          chunks.push({
            title: `Section ${chunkIndex + 1}`,
            content: currentChunk.trim(),
            section: source,
            index: chunkIndex,
          });
          chunkIndex++;
        }
        currentChunk = trimmed + '\n\n';
      } else {
        currentChunk += trimmed + '\n\n';
      }
    }

    if (currentChunk.trim()) {
      chunks.push({
        title: `Section ${chunkIndex + 1}`,
        content: currentChunk.trim(),
        section: source,
        index: chunkIndex,
      });
    }
  }

  return chunks;
}

/**
 * Split large sections into smaller chunks
 */
function splitLargeSection(content: string, section: string, startIndex: number, source: string): ContentChunk[] {
  const MAX_CHUNK_SIZE = 2000; // ~500 tokens
  const chunks: ContentChunk[] = [];

  if (content.length <= MAX_CHUNK_SIZE) {
    chunks.push({
      title: section,
      content: content.trim(),
      section: source,
      index: startIndex,
    });
    return chunks;
  }

  // Split by paragraphs
  const paragraphs = content.split(/\n\n+/);
  let currentChunk = '';
  let partNumber = 1;

  for (const para of paragraphs) {
    const trimmed = para.trim();
    if (!trimmed) continue;

    if ((currentChunk + trimmed).length > MAX_CHUNK_SIZE) {
      if (currentChunk) {
        chunks.push({
          title: `${section} (Part ${partNumber})`,
          content: currentChunk.trim(),
          section: source,
          index: startIndex + chunks.length,
        });
        partNumber++;
      }
      currentChunk = trimmed + '\n\n';
    } else {
      currentChunk += trimmed + '\n\n';
    }
  }

  if (currentChunk.trim()) {
    chunks.push({
      title: partNumber > 1 ? `${section} (Part ${partNumber})` : section,
      content: currentChunk.trim(),
      section: source,
      index: startIndex + chunks.length,
    });
  }

  return chunks;
}

/**
 * Ingest content from URL or file path
 */
async function eatContent(source: string, db: VestigeDatabase): Promise<void> {
  console.log(`\n  Fetching content from: ${source}`);

  // Determine if URL or file
  const isUrl = source.startsWith('http://') || source.startsWith('https://');
  let content: string;
  let sourceType: 'webpage' | 'article' = 'webpage';
  let sourceName: string;

  if (isUrl) {
    content = await fetchUrl(source);
    sourceName = new URL(source).hostname + new URL(source).pathname;
  } else {
    content = await readFile(source);
    sourceName = path.basename(source);
    sourceType = 'article'; // Local files are treated as articles
  }

  console.log(`  Content length: ${content.length} characters`);

  // Chunk the content
  const chunks = chunkContent(content, sourceName);
  console.log(`  Created ${chunks.length} knowledge chunks`);

  if (chunks.length === 0) {
    console.log('  No content to ingest.\n');
    return;
  }

  // Ingest each chunk
  console.log('\n  Ingesting chunks:');
  const nodeIds: string[] = [];

  for (const chunk of chunks) {
    const node = db.insertNode({
      content: chunk.content,
      summary: chunk.title,
      sourceType: sourceType,
      sourcePlatform: isUrl ? 'browser' : 'manual',
      sourceUrl: isUrl ? source : undefined,
      createdAt: new Date(),
      updatedAt: new Date(),
      lastAccessedAt: new Date(),
      accessCount: 0,
      retentionStrength: 1.0,
      stabilityFactor: 1.0,
      reviewCount: 0,
      confidence: 0.9, // Ingested docs are high confidence
      isContradicted: false,
      contradictionIds: [],
      people: [],
      concepts: [],
      events: [],
      tags: ['ingested', chunk.section.toLowerCase().replace(/[^a-z0-9]+/g, '-')],
      sourceChain: [source],
    });

    nodeIds.push(node.id);
    console.log(`    [${node.id.slice(0, 8)}] ${chunk.title.slice(0, 50)}${chunk.title.length > 50 ? '...' : ''}`);
  }

  // Create edges between sequential chunks (they're related!)
  console.log('\n  Creating knowledge connections...');
  let edgesCreated = 0;
  for (let i = 0; i < nodeIds.length - 1; i++) {
    const fromId = nodeIds[i];
    const toId = nodeIds[i + 1];
    if (!fromId || !toId) continue;

    try {
      db.insertEdge({
        fromId,
        toId,
        edgeType: 'follows',
        weight: 0.8,
        metadata: { source: 'ingestion', order: i },
        createdAt: new Date(),
      });
      edgesCreated++;
    } catch {
      // Edge might already exist
    }
  }

  console.log(`  Created ${edgesCreated} sequential connections`);
  console.log(`\n  Successfully ingested ${chunks.length} chunks from ${sourceName}`);
  console.log(`  Use 'vestige recall' or ask Claude to find this knowledge.\n`);
}

const command = process.argv[2];
const args = process.argv.slice(3);

async function main() {
  const db = new VestigeDatabase();

  try {
    switch (command) {
      case 'stats': {
        const detailed = args[0] === 'detailed';
        const stats = db.getStats();
        const health = db.checkHealth();
        const size = db.getDatabaseSize();

        console.log('\n  Memory Statistics');
        console.log('  -----------------');
        console.log(`  Status:          ${getStatusEmoji(health.status)} ${health.status.toUpperCase()}`);
        console.log(`  Total nodes:     ${stats.totalNodes}`);
        console.log(`  Total people:    ${stats.totalPeople}`);
        console.log(`  Total connections: ${stats.totalEdges}`);
        console.log(`  Database Size:   ${size.formatted}`);
        console.log(`  Last Backup:     ${health.lastBackup || 'Never'}`);

        if (health.warnings.length > 0) {
          console.log('\n  Warnings:');
          for (const warning of health.warnings) {
            console.log(`    - ${warning}`);
          }
        }

        const decaying = db.getDecayingNodes(0.5, { limit: 100 });
        console.log(`\n  Knowledge needing review: ${decaying.total} nodes`);

        if (detailed) {
          // Retention strength distribution
          console.log('\n  Retention Strength Distribution');
          console.log('  --------------------------------');

          const allNodes = db.getRecentNodes({ limit: 10000 });
          const distribution = {
            strong: 0,   // 0.8-1.0
            good: 0,     // 0.6-0.8
            moderate: 0, // 0.4-0.6
            weak: 0,     // 0.2-0.4
            fading: 0,   // 0.0-0.2
          };

          for (const node of allNodes.items) {
            const strength = node.retentionStrength;
            if (strength >= 0.8) distribution.strong++;
            else if (strength >= 0.6) distribution.good++;
            else if (strength >= 0.4) distribution.moderate++;
            else if (strength >= 0.2) distribution.weak++;
            else distribution.fading++;
          }

          const total = allNodes.items.length || 1;
          console.log(`    Strong (80-100%):   ${distribution.strong} (${((distribution.strong / total) * 100).toFixed(1)}%)`);
          console.log(`    Good (60-80%):      ${distribution.good} (${((distribution.good / total) * 100).toFixed(1)}%)`);
          console.log(`    Moderate (40-60%):  ${distribution.moderate} (${((distribution.moderate / total) * 100).toFixed(1)}%)`);
          console.log(`    Weak (20-40%):      ${distribution.weak} (${((distribution.weak / total) * 100).toFixed(1)}%)`);
          console.log(`    Fading (0-20%):     ${distribution.fading} (${((distribution.fading / total) * 100).toFixed(1)}%)`);

          // Source type breakdown
          console.log('\n  Source Type Breakdown');
          console.log('  ---------------------');

          const sourceTypes: Record<string, number> = {};
          for (const node of allNodes.items) {
            const type = node.sourceType || 'unknown';
            sourceTypes[type] = (sourceTypes[type] || 0) + 1;
          }

          const sortedTypes = Object.entries(sourceTypes).sort((a, b) => b[1] - a[1]);
          for (const [type, count] of sortedTypes.slice(0, 10)) {
            console.log(`    ${type.padEnd(15)} ${count}`);
          }

          // Services status
          console.log('\n  Services Status');
          console.log('  ---------------');

          try {
            const embService = new OllamaEmbeddingService();
            const embAvailable = await embService.isAvailable();
            console.log(`    Embeddings:    ${embAvailable ? 'Available (Ollama)' : 'Fallback mode'}`);
          } catch {
            console.log('    Embeddings:    Check failed');
          }

          try {
            const chromaStore = new ChromaVectorStore();
            const vecAvailable = await chromaStore.isAvailable();
            if (vecAvailable) {
              const vecStats = await chromaStore.getStats();
              console.log(`    Vector Store:  ChromaDB (${vecStats.embeddingCount} embeddings)`);
            } else {
              console.log('    Vector Store:  SQLite fallback');
            }
            await chromaStore.close();
          } catch {
            console.log('    Vector Store:  Check failed');
          }

          // FSRS config
          const config = getConfig();
          console.log(`    FSRS Retention: ${(config.fsrs.desiredRetention * 100).toFixed(0)}%`);
        }

        console.log();
        break;
      }

      case 'health': {
        const health = db.checkHealth();
        const size = db.getDatabaseSize();

        console.log('\n  Vestige Health Check\n');
        console.log(`  Status:           ${getStatusEmoji(health.status)} ${health.status.toUpperCase()}`);
        console.log(`  Database Path:    ${health.dbPath}`);
        console.log(`  Database Size:    ${size.formatted}`);
        console.log(`  WAL Mode:         ${health.walMode ? 'Enabled' : 'Disabled'}`);
        console.log(`  Integrity Check:  ${health.integrityCheck ? 'Passed' : 'FAILED'}`);
        console.log(`  Node Count:       ${health.nodeCount}`);
        console.log(`  People Count:     ${health.peopleCount}`);
        console.log(`  Edge Count:       ${health.edgeCount}`);
        console.log(`  Last Backup:      ${health.lastBackup || 'Never'}`);

        if (health.warnings.length > 0) {
          console.log('\n  Warnings:');
          for (const warning of health.warnings) {
            console.log(`    - ${warning}`);
          }
        } else {
          console.log('\n  No warnings - everything looks good!');
        }
        console.log();
        break;
      }

      case 'review': {
        const decaying = db.getDecayingNodes(0.5, { limit: 10 });
        if (decaying.items.length === 0) {
          console.log('\n  No knowledge needs review right now!\n');
          break;
        }

        console.log('\n  Knowledge Due for Review\n');
        console.log(`  Showing ${decaying.items.length} of ${decaying.total} items\n`);

        for (const node of decaying.items) {
          console.log(`  [${node.id.slice(0, 8)}] ${node.content.slice(0, 80)}...`);
          console.log(`    Retention: ${(node.retentionStrength * 100).toFixed(1)}%`);
          const daysSince = Math.floor((Date.now() - node.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24));
          console.log(`    Last accessed: ${daysSince} days ago`);
          console.log();
        }

        if (decaying.hasMore) {
          console.log(`  ... and ${decaying.total - decaying.items.length} more items need review\n`);
        }
        break;
      }

      case 'people': {
        const result = db.getAllPeople({ limit: 50 });
        if (result.items.length === 0) {
          console.log('\n  No people in your network yet.\n');
          break;
        }

        console.log('\n  Your Network\n');
        console.log(`  Showing ${result.items.length} of ${result.total} people\n`);

        for (const person of result.items) {
          const daysSince = person.lastContactAt
            ? Math.floor((Date.now() - person.lastContactAt.getTime()) / (1000 * 60 * 60 * 24))
            : null;
          console.log(`  ${person.name}`);
          if (person.organization) console.log(`    Organization: ${person.organization}`);
          if (person.relationshipType) console.log(`    Relationship: ${person.relationshipType}`);
          if (daysSince !== null) console.log(`    Last contact: ${daysSince} days ago`);
          if (person.sharedTopics.length > 0) console.log(`    Topics: ${person.sharedTopics.join(', ')}`);
          console.log();
        }
        break;
      }

      case 'backup': {
        console.log('\n  Creating backup...');
        const backupPath = db.backup();
        console.log(`  Backup created: ${backupPath}`);

        const backups = db.listBackups();
        console.log(`\n  Total backups: ${backups.length}`);
        console.log('  Recent backups:');
        for (const backup of backups.slice(0, 3)) {
          console.log(`    - ${backup.path}`);
          console.log(`      Size: ${(backup.size / 1024 / 1024).toFixed(2)}MB`);
          console.log(`      Date: ${backup.date.toISOString()}`);
        }
        console.log();
        break;
      }

      case 'backups': {
        const backups = db.listBackups();
        if (backups.length === 0) {
          console.log('\n  No backups found. Create one with: vestige backup\n');
          break;
        }

        console.log('\n  Available Backups\n');
        for (const backup of backups) {
          console.log(`  ${backup.path}`);
          console.log(`    Size: ${(backup.size / 1024 / 1024).toFixed(2)}MB`);
          console.log(`    Date: ${backup.date.toISOString()}`);
          console.log();
        }
        break;
      }

      case 'restore': {
        const backupPath = args[0];
        if (!backupPath) {
          console.log('\n  Usage: vestige restore <backup-path>');
          console.log('  Use "vestige backups" to see available backups.\n');
          break;
        }

        // Validate path to prevent path traversal attacks
        const pathValidation = validatePath(backupPath);
        if (!pathValidation.valid) {
          logSecurityEvent({
            type: 'path_traversal',
            details: { error: pathValidation.error || 'Path validation failed', path: backupPath.slice(0, 100) },
            severity: 'high',
            blocked: true,
          });
          console.error(`\n  Security Error: ${pathValidation.error}\n`);
          break;
        }

        const safePath = pathValidation.sanitizedPath!;
        console.log(`\n  Restoring from: ${safePath}`);
        console.log('  WARNING: This will replace your current database!\n');

        // In a real CLI, you'd prompt for confirmation here
        // For now, we just do it
        try {
          db.restore(safePath);
          console.log('  Restore completed successfully!\n');
        } catch (error) {
          if (error instanceof VestigeDatabaseError) {
            console.error(`  Error: ${error.message} (${error.code})\n`);
          } else {
            console.error(`  Error: ${error instanceof Error ? error.message : 'Unknown error'}\n`);
          }
        }
        break;
      }

      case 'optimize': {
        console.log('\n  Optimizing database...');
        const sizeBefore = db.getDatabaseSize();
        db.optimize();
        const sizeAfter = db.getDatabaseSize();

        console.log(`  Size before: ${sizeBefore.formatted}`);
        console.log(`  Size after:  ${sizeAfter.formatted}`);
        console.log(`  Space saved: ${(sizeBefore.mb - sizeAfter.mb).toFixed(2)}MB`);
        console.log();
        break;
      }

      case 'decay': {
        console.log('\n  Applying memory decay...');
        const updated = db.applyDecay();
        console.log(`  Updated ${updated} knowledge nodes\n`);
        break;
      }

      case 'consolidate':
      case 'sleep': {
        console.log('\n  Running sleep consolidation cycle...\n');
        const consResult = await runConsolidation(db);
        console.log(`  Short-term processed: ${consResult.shortTermProcessed}`);
        console.log(`  Promoted to long-term: ${consResult.promotedToLongTerm}`);
        console.log(`  Connections discovered: ${consResult.connectionsDiscovered}`);
        console.log(`  Edges pruned: ${consResult.edgesPruned}`);
        console.log(`  Decay applied: ${consResult.decayApplied}`);
        console.log(`\n  Duration: ${consResult.duration}ms\n`);
        break;
      }

      case 'embeddings': {
        const embCmd = args[0];

        switch (embCmd) {
          case 'status': {
            console.log('\n  Embedding Service Status\n');
            const embService = new OllamaEmbeddingService();
            const available = await embService.isAvailable();
            console.log(`  Service:   ${available ? 'Available' : 'Not available'}`);
            if (available) {
              console.log(`  Provider:  Ollama`);
              console.log(`  Model:     ${embService.getModel()}`);
              console.log(`  Host:      ${process.env['OLLAMA_HOST'] || 'http://localhost:11434'}`);
            } else {
              console.log('\n  To enable embeddings:');
              console.log('    1. Install Ollama: https://ollama.ai');
              console.log('    2. Run: ollama pull nomic-embed-text');
              console.log('    3. Start Ollama service');
            }
            console.log();
            break;
          }

          case 'generate': {
            const nodeId = args[1];
            console.log('\n  Generating Embeddings\n');

            try {
              const embService = await createEmbeddingService();

              if (nodeId) {
                // Generate for specific node
                const node = db.getNode(nodeId);
                if (!node) {
                  console.log(`  Error: Node not found: ${nodeId}\n`);
                  break;
                }

                console.log(`  Generating embedding for node: ${nodeId.slice(0, 8)}...`);
                const embedding = await embService.generateEmbedding(node.content);
                console.log(`  Embedding generated: ${embedding.length} dimensions`);
                console.log(`  First 5 values: [${embedding.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);
              } else {
                // Generate for all nodes without embeddings
                console.log('  Generating embeddings for all nodes...');
                const allNodes = db.getRecentNodes({ limit: 1000 });
                let generated = 0;
                let failed = 0;

                for (const node of allNodes.items) {
                  try {
                    await embService.generateEmbedding(node.content);
                    generated++;
                    if (generated % 10 === 0) {
                      process.stdout.write(`\r  Progress: ${generated}/${allNodes.items.length}`);
                    }
                  } catch {
                    failed++;
                  }
                }
                console.log(`\n  Generated: ${generated}, Failed: ${failed}`);
              }
            } catch (error) {
              console.log(`  Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
            console.log();
            break;
          }

          case 'search': {
            const query = args.slice(1).join(' ');
            if (!query) {
              console.log('\n  Usage: vestige embeddings search "<query>"\n');
              break;
            }

            console.log(`\n  Semantic Search: "${query}"\n`);

            try {
              const embService = await createEmbeddingService();
              const queryEmbedding = await embService.generateEmbedding(query);

              // Get all nodes and compute similarity
              const allNodes = db.getRecentNodes({ limit: 500 });
              const results: Array<{ node: typeof allNodes.items[0]; similarity: number }> = [];

              for (const node of allNodes.items) {
                try {
                  const nodeEmbedding = await embService.generateEmbedding(node.content);
                  const similarity = embService.getSimilarity(queryEmbedding, nodeEmbedding);
                  results.push({ node, similarity });
                } catch {
                  // Skip nodes that fail to embed
                }
              }

              // Sort by similarity and show top 10
              results.sort((a, b) => b.similarity - a.similarity);
              const topResults = results.slice(0, 10);

              if (topResults.length === 0) {
                console.log('  No results found.\n');
                break;
              }

              console.log('  Top Results:');
              for (const { node, similarity } of topResults) {
                const preview = node.content.slice(0, 60).replace(/\n/g, ' ');
                console.log(`    [${(similarity * 100).toFixed(1)}%] ${preview}...`);
              }
            } catch (error) {
              console.log(`  Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            }
            console.log();
            break;
          }

          default:
            console.log(`
  Vestige Embeddings - Semantic Understanding

  Usage: vestige embeddings <command>

  Commands:
    status              Check embedding service availability
    generate [nodeId]   Generate embeddings (all nodes or specific)
    search "<query>"    Semantic similarity search

  Examples:
    vestige embeddings status
    vestige embeddings generate
    vestige embeddings generate abc12345
    vestige embeddings search "authentication flow"
`);
        }
        break;
      }

      case 'config': {
        const configCmd = args[0];
        const configPath = path.join(os.homedir(), '.vestige', 'config.json');

        switch (configCmd) {
          case 'show': {
            console.log('\n  Vestige Configuration\n');
            const config = getConfig();
            console.log(JSON.stringify(config, null, 2));
            console.log(`\n  Config file: ${configPath}\n`);
            break;
          }

          case 'set': {
            const key = args[1];
            const value = args.slice(2).join(' ');

            if (!key || !value) {
              console.log('\n  Usage: vestige config set <section.key> <value>');
              console.log('\n  Examples:');
              console.log('    vestige config set logging.level debug');
              console.log('    vestige config set fsrs.desiredRetention 0.85');
              console.log('    vestige config set rem.enabled false\n');
              break;
            }

            console.log(`\n  Setting ${key} = ${value}\n`);

            // Load existing config or create empty
            let fileConfig: Record<string, unknown> = {};
            if (fs.existsSync(configPath)) {
              try {
                fileConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
              } catch {
                console.log('  Warning: Could not parse existing config, starting fresh');
              }
            }

            // Parse the key path (e.g., "logging.level")
            const keyParts = key.split('.');
            let current: Record<string, unknown> = fileConfig;

            for (let i = 0; i < keyParts.length - 1; i++) {
              const part = keyParts[i]!;
              if (!(part in current) || typeof current[part] !== 'object') {
                current[part] = {};
              }
              current = current[part] as Record<string, unknown>;
            }

            // Parse value (try as JSON, fall back to string)
            let parsedValue: unknown = value;
            try {
              parsedValue = JSON.parse(value);
            } catch {
              // Keep as string
            }

            current[keyParts[keyParts.length - 1]!] = parsedValue;

            // Ensure directory exists
            const configDir = path.dirname(configPath);
            if (!fs.existsSync(configDir)) {
              fs.mkdirSync(configDir, { recursive: true });
            }

            // Write config
            fs.writeFileSync(configPath, JSON.stringify(fileConfig, null, 2));

            // Reset singleton to reload
            resetConfig();

            console.log(`  Configuration updated: ${key} = ${JSON.stringify(parsedValue)}`);
            console.log(`  Saved to: ${configPath}\n`);
            break;
          }

          case 'reset': {
            console.log('\n  Resetting configuration to defaults...\n');

            if (fs.existsSync(configPath)) {
              // Create backup before deleting
              const backupPath = `${configPath}.backup.${Date.now()}`;
              fs.copyFileSync(configPath, backupPath);
              console.log(`  Backup created: ${backupPath}`);

              fs.unlinkSync(configPath);
              console.log(`  Removed: ${configPath}`);
            }

            resetConfig();
            console.log('  Configuration reset to defaults.\n');
            break;
          }

          default:
            console.log(`
  Vestige Configuration Management

  Usage: vestige config <command>

  Commands:
    show                Display current configuration
    set <key> <value>   Update a configuration value
    reset               Reset to default configuration

  Examples:
    vestige config show
    vestige config set logging.level debug
    vestige config set fsrs.desiredRetention 0.85
    vestige config reset

  Configuration Sections:
    database      - Database paths and settings
    fsrs          - Spaced repetition algorithm
    memory        - Dual-strength memory model
    rem           - REM cycle settings
    consolidation - Sleep consolidation
    embeddings    - Embedding service
    vectorStore   - Vector database
    logging       - Log levels
    limits        - Size limits
`);
        }
        break;
      }

      case 'test': {
        console.log('\n  Vestige Self-Test Suite\n');
        console.log('  Running diagnostic tests...\n');

        let allPassed = true;

        // Test 1: Database
        try {
          const stats = db.getStats();
          console.log(`  [PASS] Database: ${stats.totalNodes} nodes, ${stats.totalPeople} people, ${stats.totalEdges} edges`);
        } catch (error) {
          console.log(`  [FAIL] Database: ${error instanceof Error ? error.message : 'Unknown error'}`);
          allPassed = false;
        }

        // Test 2: Embeddings
        try {
          const embService = new OllamaEmbeddingService();
          const embAvailable = await embService.isAvailable();
          if (embAvailable) {
            console.log(`  [PASS] Embeddings: Ollama available (${embService.getModel()})`);
          } else {
            console.log('  [WARN] Embeddings: Ollama not available (fallback will be used)');
          }
        } catch (error) {
          console.log(`  [WARN] Embeddings: ${error instanceof Error ? error.message : 'Check failed'}`);
        }

        // Test 3: Vector Store
        try {
          const chromaStore = new ChromaVectorStore();
          const vecAvailable = await chromaStore.isAvailable();
          if (vecAvailable) {
            const vecStats = await chromaStore.getStats();
            console.log(`  [PASS] Vector Store: ChromaDB available (${vecStats.embeddingCount} embeddings)`);
          } else {
            console.log('  [WARN] Vector Store: ChromaDB not available (SQLite fallback will be used)');
          }
          await chromaStore.close();
        } catch (error) {
          console.log(`  [WARN] Vector Store: ${error instanceof Error ? error.message : 'Check failed'}`);
        }

        // Test 4: Configuration
        try {
          const config = getConfig();
          console.log(`  [PASS] Configuration: Loaded (FSRS retention: ${config.fsrs.desiredRetention})`);
        } catch (error) {
          console.log(`  [FAIL] Configuration: ${error instanceof Error ? error.message : 'Load failed'}`);
          allPassed = false;
        }

        // Test 5: Database health
        try {
          const health = db.checkHealth();
          if (health.status === 'healthy') {
            console.log(`  [PASS] Health Check: ${health.status}`);
          } else if (health.status === 'warning') {
            console.log(`  [WARN] Health Check: ${health.warnings.length} warning(s)`);
          } else {
            console.log(`  [FAIL] Health Check: ${health.status}`);
            allPassed = false;
          }
        } catch (error) {
          console.log(`  [FAIL] Health Check: ${error instanceof Error ? error.message : 'Check failed'}`);
          allPassed = false;
        }

        console.log();
        if (allPassed) {
          console.log('  All core tests passed!\n');
        } else {
          console.log('  Some tests failed. Review the output above.\n');
        }
        break;
      }

      case 'ingest': {
        const content = args.join(' ');
        if (!content) {
          console.log('\n  Usage: vestige ingest "<content>"');
          console.log('\n  Store knowledge directly into Vestige.');
          console.log('\n  Examples:');
          console.log('    vestige ingest "API rate limit is 100 req/min"');
          console.log('    vestige ingest "Meeting with John: discussed Q4 roadmap"\n');
          break;
        }

        console.log('\n  Ingesting knowledge...\n');

        try {
          const node = db.insertNode({
            content: content,
            summary: content.slice(0, 100),
            sourceType: 'note',
            sourcePlatform: 'manual',
            createdAt: new Date(),
            updatedAt: new Date(),
            lastAccessedAt: new Date(),
            accessCount: 0,
            retentionStrength: 1.0,
            stabilityFactor: 1.0,
            reviewCount: 0,
            confidence: 0.9,
            isContradicted: false,
            contradictionIds: [],
            people: [],
            concepts: [],
            events: [],
            tags: ['cli-ingested'],
            sourceChain: ['cli'],
          });

          console.log(`  Stored as node: ${node.id}`);
          console.log(`  Content: "${content.slice(0, 60)}${content.length > 60 ? '...' : ''}"`);
          console.log('\n  Knowledge successfully ingested!\n');
        } catch (error) {
          console.error(`  Error: ${error instanceof Error ? error.message : 'Unknown error'}\n`);
        }
        break;
      }

      case 'recall': {
        const query = args.join(' ');
        if (!query) {
          console.log('\n  Usage: vestige recall "<query>"');
          console.log('\n  Search your memories.');
          console.log('\n  Examples:');
          console.log('    vestige recall "rate limit"');
          console.log('    vestige recall "meeting John"\n');
          break;
        }

        console.log(`\n  Searching memories for: "${query}"\n`);

        try {
          const result = db.searchNodes(query, { limit: 10 });

          if (result.items.length === 0) {
            console.log('  No memories found matching your query.\n');
            break;
          }

          console.log(`  Found ${result.total} memories (showing ${result.items.length}):\n`);

          for (const node of result.items) {
            const preview = node.content.slice(0, 80).replace(/\n/g, ' ');
            const daysSince = Math.floor((Date.now() - node.lastAccessedAt.getTime()) / (1000 * 60 * 60 * 24));
            console.log(`  [${node.id.slice(0, 8)}] ${preview}${node.content.length > 80 ? '...' : ''}`);
            console.log(`           Retention: ${(node.retentionStrength * 100).toFixed(1)}% | Last accessed: ${daysSince}d ago`);
            console.log();
          }
        } catch (error) {
          console.error(`  Error: ${error instanceof Error ? error.message : 'Unknown error'}\n`);
        }
        break;
      }

      case 'eat': {
        const source = args[0];
        if (!source) {
          console.log('\n  Usage: vestige eat <url|path>');
          console.log('\n  Examples:');
          console.log('    vestige eat https://docs.rs/tauri/latest/');
          console.log('    vestige eat ./README.md');
          console.log('    vestige eat ~/Documents/notes.txt');
          console.log('\n  The Man Page Absorber chunks content intelligently and');
          console.log('  creates interconnected knowledge nodes for retrieval.\n');
          break;
        }

        try {
          await eatContent(source, db);
        } catch (error) {
          console.error(`\n  Error ingesting content: ${error instanceof Error ? error.message : 'Unknown error'}\n`);
        }
        break;
      }

      case 'context': {
        console.log('\n  Ghost in the Shell - Current Context\n');
        const context = captureContext();

        if (context.activeWindow) {
          console.log(`  Active App:    ${context.activeWindow.app}`);
          console.log(`  Window Title:  ${context.activeWindow.title}`);
        } else {
          console.log('  Active Window: (unable to detect)');
        }

        console.log(`  Working Dir:   ${context.workingDirectory}`);

        if (context.gitBranch) {
          console.log(`  Git Branch:    ${context.gitBranch}`);
        }

        if (context.recentFiles.length > 0) {
          console.log('\n  Recent Files (last hour):');
          for (const file of context.recentFiles.slice(0, 5)) {
            console.log(`    - ${file}`);
          }
        }

        if (context.clipboard) {
          console.log('\n  Clipboard:');
          const preview = context.clipboard.slice(0, 200);
          console.log(`    "${preview}${context.clipboard.length > 200 ? '...' : ''}"`);
        }

        console.log('\n  Injection String:');
        console.log(`    ${formatContextForInjection(context)}`);
        console.log();
        break;
      }

      case 'watch': {
        console.log('\n  Starting Ghost in the Shell context watcher...');
        console.log('  Press Ctrl+C to stop.\n');

        startContextWatcher(5000);

        // Keep running until interrupted
        process.on('SIGINT', () => {
          console.log('\n  Stopping context watcher...');
          process.exit(0);
        });

        // Keep the process alive
        await new Promise(() => {}); // Never resolves
        break;
      }

      case 'dream': {
        console.log('\n  REM Cycle - Discovering Hidden Connections\n');

        // Preview first
        console.log('  Analyzing knowledge graph...');
        const preview = await previewREMCycle(db);

        if (preview.connectionsDiscovered === 0) {
          console.log('  No new connections discovered.');
          console.log('  Your knowledge graph is well-connected or needs more nodes.\n');
          break;
        }

        console.log(`  Found ${preview.connectionsDiscovered} potential connections!\n`);

        // Show previews
        console.log('  Discoveries:');
        for (const d of preview.discoveries.slice(0, 10)) {
          console.log(`    "${d.nodeA}..."`);
          console.log(`      <-> "${d.nodeB}..."`);
          console.log(`      Reason: ${d.reason}\n`);
        }

        if (preview.discoveries.length > 10) {
          console.log(`    ... and ${preview.discoveries.length - 10} more\n`);
        }

        // Actually create the connections
        if (args[0] !== '--dry-run') {
          console.log('  Creating connections...');
          const result = await runREMCycle(db);
          console.log(`  Created ${result.connectionsCreated} new edges in ${result.duration}ms\n`);
        } else {
          console.log('  (Dry run - no connections created. Remove --dry-run to create them)\n');
        }
        break;
      }

      case 'rem': {
        // Alias for dream
        console.log('\n  Starting REM Cycle (alias for "dream")...\n');
        const result = await runREMCycle(db);

        console.log(`  Analyzed: ${result.nodesAnalyzed} nodes`);
        console.log(`  Discovered: ${result.connectionsDiscovered} connections`);
        console.log(`  Created: ${result.connectionsCreated} edges`);
        console.log(`  Duration: ${result.duration}ms\n`);

        if (result.discoveries.length > 0) {
          console.log('  New connections:');
          for (const d of result.discoveries.slice(0, 5)) {
            console.log(`    - ${d.reason}`);
          }
          if (result.discoveries.length > 5) {
            console.log(`    ... and ${result.discoveries.length - 5} more`);
          }
        }
        console.log();
        break;
      }

      // ====================================================================
      // SHADOW SELF - Unsolved Problems Queue
      // ====================================================================

      case 'problem': {
        const description = args.join(' ');
        if (!description) {
          console.log('\n  Usage: vestige problem <description>');
          console.log('\n  Log an unsolved problem for your Shadow to work on.\n');
          console.log('  Examples:');
          console.log('    vestige problem "How to implement efficient graph traversal"');
          console.log('    vestige problem "Why is the memory leak happening in the worker"');
          console.log('\n  The Shadow Self will periodically revisit these problems');
          console.log('  when new knowledge might provide insights.\n');
          break;
        }

        const shadow = new ShadowSelf();
        try {
          const problem = shadow.logProblem(description, {
            context: formatContextForInjection(captureContext()),
            priority: 3,
          });
          console.log('\n  Problem logged to the Shadow Self\n');
          console.log(`  ID:          ${problem.id}`);
          console.log(`  Description: ${problem.description.slice(0, 60)}${problem.description.length > 60 ? '...' : ''}`);
          console.log(`  Priority:    ${problem.priority}`);
          console.log(`  Status:      ${problem.status}`);
          console.log('\n  Your Shadow will work on this while you rest.\n');
        } finally {
          shadow.close();
        }
        break;
      }

      case 'problems': {
        const shadow = new ShadowSelf();
        try {
          const problems = shadow.getOpenProblems();
          const stats = shadow.getStats();

          console.log('\n  Shadow Self - Unsolved Problems Queue\n');
          console.log(`  Total: ${stats.total} | Open: ${stats.open} | Investigating: ${stats.investigating} | Solved: ${stats.solved}\n`);

          if (problems.length === 0) {
            console.log('  No open problems. Your mind is at peace.\n');
            console.log('  Log a problem with: vestige problem "<description>"\n');
            break;
          }

          for (const p of problems) {
            const priority = '!'.repeat(p.priority);
            const daysSince = Math.floor((Date.now() - p.createdAt.getTime()) / (1000 * 60 * 60 * 24));
            console.log(`  [${p.id.slice(0, 8)}] ${priority.padEnd(5)} ${p.description.slice(0, 50)}${p.description.length > 50 ? '...' : ''}`);
            console.log(`             Status: ${p.status} | Attempts: ${p.attempts} | Age: ${daysSince}d`);

            // Show any insights
            const insights = shadow.getInsights(p.id);
            if (insights.length > 0) {
              console.log(`             Latest insight: "${insights[0]?.insight.slice(0, 40)}..."`);
            }
            console.log();
          }
        } finally {
          shadow.close();
        }
        break;
      }

      case 'solve': {
        const problemId = args[0];
        const solution = args.slice(1).join(' ');

        if (!problemId) {
          console.log('\n  Usage: vestige solve <problem-id> <solution>');
          console.log('\n  Mark a problem as solved with the solution.\n');
          console.log('  Example:');
          console.log('    vestige solve abc123 "Used memoization to optimize the traversal"');
          console.log('\n  Use "vestige problems" to see problem IDs.\n');
          break;
        }

        const shadow = new ShadowSelf();
        try {
          // Find the problem (match on prefix)
          const problems = shadow.getOpenProblems();
          const match = problems.find(p => p.id.startsWith(problemId));

          if (!match) {
            console.log(`\n  Problem not found: ${problemId}`);
            console.log('  Use "vestige problems" to see open problems.\n');
            break;
          }

          shadow.markSolved(match.id, solution || 'Solved (no details provided)');
          console.log('\n  Problem marked as SOLVED\n');
          console.log(`  Problem:  ${match.description.slice(0, 50)}...`);
          console.log(`  Solution: ${solution || '(no details)'}`);
          console.log(`  Attempts: ${match.attempts}`);
          console.log('\n  The Shadow rejoices.\n');
        } finally {
          shadow.close();
        }
        break;
      }

      case 'shadow': {
        console.log('\n  Shadow Self - Running Background Analysis\n');

        const shadow = new ShadowSelf();
        try {
          const stats = shadow.getStats();
          console.log(`  Problems: ${stats.open} open, ${stats.investigating} investigating, ${stats.solved} solved`);
          console.log(`  Total insights generated: ${stats.totalInsights}\n`);

          if (stats.open === 0 && stats.investigating === 0) {
            console.log('  No problems to work on. The Shadow rests.\n');
            break;
          }

          console.log('  Running shadow cycle...');
          const result = runShadowCycle(shadow, db);

          console.log(`  Analyzed: ${result.problemsAnalyzed} problems`);
          console.log(`  New insights: ${result.insightsGenerated}\n`);

          if (result.insights.length > 0) {
            console.log('  Discoveries:');
            for (const i of result.insights) {
              console.log(`    Problem: "${i.problem}..."`);
              console.log(`    Insight: ${i.insight}\n`);
            }
          } else {
            console.log('  No new insights yet. The Shadow continues to watch.\n');
          }
        } finally {
          shadow.close();
        }
        break;
      }

      case 'help':
      default:
        console.log(`
  Vestige CLI - Git Blame for AI Thoughts

  Usage: vestige <command> [options]

  Core Commands:
    ingest <content>     Store knowledge directly
    recall <query>       Search memories
    review               Review memories due for reinforcement
    stats [detailed]     Show memory statistics

  Memory Processing:
    dream                Run REM cycle (connection discovery)
    consolidate          Run sleep consolidation (alias: sleep)
    decay                Apply memory decay

  Embeddings:
    embeddings status    Check embedding service availability
    embeddings generate  Generate embeddings for all nodes
    embeddings search    Semantic similarity search

  Configuration:
    config show          Display current configuration
    config set <k> <v>   Update a configuration value
    config reset         Reset to default configuration
    test                 Run self-tests

  Ghost in the Shell:
    context              Show current system context
    watch                Start context watcher daemon
    eat <url|path>       Ingest docs/content

  Shadow Self (Unsolved Problems):
    problem <desc>       Log a new unsolved problem
    problems             List all open problems
    solve <id> <sol>     Mark a problem as solved
    shadow               Run shadow cycle for insights

  Maintenance:
    backup               Create database backup
    backups              List available backups
    restore <path>       Restore from backup
    optimize             Optimize database
    health               Detailed health check
    people               List people in your network

  Examples:
    vestige ingest "API rate limit is 100 req/min"
    vestige recall "rate limit"
    vestige stats detailed
    vestige embeddings search "authentication"
    vestige config set logging.level debug
    vestige eat https://docs.example.com/api

  The Vestige MCP server runs automatically when connected to Claude.
  Your brain gets smarter while you sleep.
`);
    }
  } finally {
    db.close();
  }
}

function getStatusEmoji(status: string): string {
  switch (status) {
    case 'healthy':
      return '(healthy)';
    case 'warning':
      return '(warning)';
    case 'critical':
      return '(CRITICAL)';
    default:
      return '';
  }
}

main().catch((error) => {
  console.error('Error:', error instanceof Error ? error.message : error);
  process.exit(1);
});

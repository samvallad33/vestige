// Vestige API Types — auto-matched to Rust backend

export interface Memory {
	id: string;
	content: string;
	nodeType: string;
	tags: string[];
	retentionStrength: number;
	storageStrength: number;
	retrievalStrength: number;
	createdAt: string;
	updatedAt: string;
	source?: string;
	reviewCount?: number;
	combinedScore?: number;
	sentimentScore?: number;
	sentimentMagnitude?: number;
	lastAccessedAt?: string;
	nextReviewAt?: string;
	validFrom?: string;
	validUntil?: string;
}

export interface SearchResult {
	query: string;
	total: number;
	durationMs: number;
	results: Memory[];
}

export interface MemoryListResponse {
	total: number;
	memories: Memory[];
}

export interface SystemStats {
	totalMemories: number;
	dueForReview: number;
	averageRetention: number;
	averageStorageStrength: number;
	averageRetrievalStrength: number;
	withEmbeddings: number;
	embeddingCoverage: number;
	embeddingModel: string;
	oldestMemory?: string;
	newestMemory?: string;
}

// Embedding profiles deliberately keep model contracts separate. A profile owns
// its encoder, vector space, local artifact state, and any migration receipt;
// vectors from different profiles are never comparable.
export type EmbeddingProfileStage =
	| 'available'
	| 'installing'
	| 'installed'
	| 'evaluating'
	| 'evaluated'
	| 'migrating'
	| 'ready'
	| 'active'
	| 'rollback_ready'
	| 'error';

export interface EmbeddingProfile {
	id: string;
	name: string;
	modelId: string;
	description?: string;
	stage: EmbeddingProfileStage;
	installed: boolean;
	active: boolean;
	dimensions: number;
	maxTokens?: number;
	modelBytes?: number;
	vectorBytes?: number;
	diskBytes?: number;
	hardware?: string;
	localOnly?: boolean;
	migration?: {
		state: 'not_started' | 'in_progress' | 'validating' | 'paused' | 'complete' | 'not_required' | 'failed' | 'cancelled';
		id?: string;
		total?: number;
		completed?: number;
		remaining?: number;
		updatedAt?: string;
	};
	evaluation?: {
		state: 'not_run' | 'running' | 'complete' | 'failed';
		score?: number;
		metric?: string;
		sampleSize?: number;
	};
	lastReceipt?: {
		id?: string;
		at?: string;
		summary?: string;
	};
}

export interface EmbeddingProfilesResponse {
	profiles: EmbeddingProfile[];
	activeProfileId?: string | null;
	rollbackProfileId?: string | null;
	localOnly?: boolean;
	available?: boolean;
}

export interface EmbeddingProfileActionResponse extends Partial<EmbeddingProfilesResponse> {
	accepted?: boolean;
	message?: string;
	receipt?: EmbeddingProfile['lastReceipt'];
	// Install, evaluation, and migration deliberately return this guidance as a
	// 409 response instead of accepting a filesystem path through the browser.
	cliRequired?: boolean;
	operation?: 'install' | 'evaluate' | 'migrate' | 'activate' | 'rollback';
	command?: string;
}

export interface HealthCheck {
	status: 'healthy' | 'degraded' | 'critical' | 'empty';
	totalMemories: number;
	averageRetention: number;
	version: string;
}

export interface TimelineDay {
	date: string;
	count: number;
	memories: Memory[];
}

export interface TimelineResponse {
	days: number;
	totalMemories: number;
	timeline: TimelineDay[];
}

export interface GraphNode {
	id: string;
	label: string;
	type: string;
	retention: number;
	tags: string[];
	createdAt: string;
	updatedAt: string;
	isCenter: boolean;
	// v2.0.5 Active Forgetting — top-down suppression state
	suppression_count?: number;
	suppressed_at?: string;
	// v2.3 living field — real FSRS state so the graph can render LIVE decay
	// (retrievability recomputed each frame on the true forgetting curve).
	// stability in days; lastAccessed is an ISO timestamp. Optional so an old
	// serializer (pre-2026-07) still parses.
	stability?: number;
	lastAccessed?: string;
}

export interface GraphEdge {
	source: string;
	target: string;
	weight: number;
	type: string;
}

export interface GraphResponse {
	nodes: GraphNode[];
	edges: GraphEdge[];
	center_id: string;
	depth: number;
	nodeCount: number;
	edgeCount: number;
}

export interface DreamResult {
	status: string;
	memoriesReplayed: number;
	connectionsPersisted: number;
	insights: DreamInsight[];
	stats: {
		newConnectionsFound: number;
		connectionsPersisted: number;
		memoriesStrengthened: number;
		memoriesCompressed: number;
		insightsGenerated: number;
		durationMs: number;
	};
}

export interface DreamInsight {
	type: string;
	insight: string;
	sourceMemories: string[];
	confidence: number;
	noveltyScore: number;
}

export interface ImportanceScore {
	composite: number;
	channels: {
		novelty: number;
		arousal: number;
		reward: number;
		attention: number;
	};
	recommendation: 'save' | 'skip';
}

export interface RetentionDistribution {
	distribution: { range: string; count: number }[];
	byType: Record<string, number>;
	endangered: Memory[];
	total: number;
}

export interface ConsolidationResult {
	nodesProcessed: number;
	decayApplied: number;
	embeddingsGenerated: number;
	duplicatesMerged: number;
	activationsComputed: number;
	durationMs: number;
}

// WebSocket event types
export type VestigeEventType =
	| 'Connected'
	| 'MemoryCreated'
	| 'MemoryUpdated'
	| 'MemoryDeleted'
	| 'MemoryPromoted'
	| 'MemoryDemoted'
	| 'MemorySuppressed'
	| 'MemoryUnsuppressed'
	| 'Rac1CascadeSwept'
	| 'SearchPerformed'
	| 'DreamStarted'
	| 'DreamProgress'
	| 'DreamCompleted'
	| 'ConsolidationStarted'
	| 'ConsolidationCompleted'
	| 'RetentionDecayed'
	| 'ConnectionDiscovered'
	| 'ActivationSpread'
	| 'ImportanceScored'
	| 'DeepReferenceCompleted'
	// v2.3 living field — RSB causal recall receipt (Phase 4): a failure-
	// triggered backward-only causal path with shared-entity join keys, so the
	// field lights the REAL cause chain instead of a random pulse.
	| 'BackfillFired'
	| 'CausalReceipt'
	| 'HookVerdictRecorded'
	| 'TraceEvent'
	| 'MemoryPrOpened'
	| 'MemoryPrDecided'
	| 'Heartbeat'
	// The server's broadcast buffer overflowed for this socket: `data.missed`
	// events were dropped. Explicit, never silent; consumers refetch state.
	| 'EventsDropped';

export interface VestigeEvent {
	type: VestigeEventType;
	data: Record<string, unknown>;
}

// v2.0.7: active-forgetting response shapes. Each suppress call COMPOUNDS;
// `suppressionCount` is the lifetime total. `reversibleUntil` is the ISO
// timestamp after which the labile window closes and the suppression locks in.
export interface SuppressResult {
	suppressed: true;
	id: string;
	suppressionCount: number;
	priorCount: number;
	retrievalPenalty: number;
	retentionStrength: number;
	retrievalStrength: number;
	stability: number;
	estimatedCascadeNeighbors: number;
	reversibleUntil: string;
	labileWindowHours: number;
	reason: string | null;
}

export interface UnsuppressResult {
	unsuppressed: true;
	id: string;
	suppressionCount: number;
	stillSuppressed: boolean;
	retentionStrength: number;
	retrievalStrength: number;
	stability: number;
}

export type VerdictLevel = 'PASS' | 'NOTE' | 'CAUTION' | 'VETO' | 'APPEALED';
export type SanhedrinAppealReason = 'stale' | 'wrong' | 'too_strict';

export interface SanhedrinAppealState {
	status: 'open' | 'appealed';
	actions?: SanhedrinAppealReason[];
	lastReason?: SanhedrinAppealReason;
	note?: string;
}

export interface SanhedrinPrecedent {
	type?: string;
	summary?: string;
	command?: string;
	exitCode?: number | null;
	evidence?: string;
}

export interface SanhedrinClaim {
	id: string;
	text: string;
	fingerprint: string;
	class: string;
	subject: string;
	risk: string;
	evidence_state: string;
	decision: string;
	precedent: SanhedrinPrecedent[];
	fix: string;
	appeal: SanhedrinAppealState;
}

export interface SanhedrinReceipt {
	schema: string;
	id: string;
	draftId: string;
	createdAt: string;
	overall: string;
	verdictBar: VerdictLevel;
	summary: string;
	draftPreview: string;
	claims: SanhedrinClaim[];
	receipts: Array<Record<string, unknown>>;
	source?: Record<string, unknown>;
}

export interface SanhedrinLatestResponse {
	receipt: SanhedrinReceipt | null;
	stateDir: string;
	receiptPath?: string;
	htmlPath?: string;
	schemaWarning?: string | null;
}

export interface SanhedrinAppealResponse {
	appeal: Record<string, unknown>;
	receipt: SanhedrinReceipt;
}

export interface SanhedrinDailyTelemetry {
	date: string;
	total: number;
	pass: number;
	note: number;
	caution: number;
	veto: number;
	appealed: number;
	failOpen: number;
}

export interface SanhedrinTelemetryResponse {
	days: number;
	stateDir: string;
	totalRuns: number;
	byVerdict: Partial<Record<VerdictLevel, number>>;
	byClass: Record<string, number>;
	appeals: number;
	failOpen: number;
	truncated?: boolean;
	lastRunAt?: string | null;
	daily: SanhedrinDailyTelemetry[];
}

// Intentions (prospective memory)
export interface IntentionItem {
	id: string;
	content: string;
	trigger_type: string;
	trigger_data: string; // JSON-encoded trigger payload (e.g. {"type":"time","at":"..."} )
	status: string;
	priority: number; // 1=low, 2=normal, 3=high, 4=critical
	created_at: string;
	deadline?: string | null;
	snoozed_until?: string | null;
}

// Memory Hygiene — GET /api/duplicates (cosine-similarity clusters)
export interface DuplicateClusterMemory {
	id: string;
	content: string;
	nodeType: string;
	tags: string[];
	retention: number;
	createdAt: string;
}

export interface DuplicateClusterGroup {
	similarity: number;
	suggestedAction: 'merge' | 'review';
	memories: DuplicateClusterMemory[];
}

export interface DuplicatesResponse {
	threshold: number;
	total: number;
	clusters: DuplicateClusterGroup[];
}

// Contradiction pairs — GET /api/contradictions. Field names mirror the
// Contradiction interface in $components/ContradictionArcs.svelte; a = older
// memory, b = newer. Sorted by similarity desc.
export interface ContradictionPair {
	memory_a_id: string;
	memory_b_id: string;
	memory_a_preview: string;
	memory_b_preview: string;
	memory_a_type?: string;
	memory_b_type?: string;
	memory_a_created?: string;
	memory_b_created?: string;
	memory_a_tags?: string[];
	memory_b_tags?: string[];
	trust_a: number;
	trust_b: number;
	similarity: number;
	date_diff_days: number;
	topic: string;
}

export interface ContradictionsResponse {
	memoriesAnalyzed: number;
	total: number;
	contradictions: ContradictionPair[];
}

// Cross-project pattern transfer — GET /api/patterns/cross-project.
// Only the six tracked categories ever cross the wire (backend drops others).
export type CrossProjectCategory =
	| 'ErrorHandling'
	| 'AsyncConcurrency'
	| 'Testing'
	| 'Architecture'
	| 'Performance'
	| 'Security';

export interface CrossProjectPattern {
	name: string;
	category: CrossProjectCategory;
	origin_project: string;
	transferred_to: string[];
	transfer_count: number;
	last_used: string;
	confidence: number;
}

export interface CrossProjectPatternsResponse {
	projects: string[];
	patterns: CrossProjectPattern[];
}

// Per-memory audit trail — GET /api/memories/{id}/audit. Events arrive
// newest-first; the action union matches AuditAction in
// $components/audit-trail-helpers.ts.
export type MemoryAuditAction =
	| 'created'
	| 'accessed'
	| 'promoted'
	| 'demoted'
	| 'edited'
	| 'suppressed'
	| 'dreamed'
	| 'reconsolidated';

export interface MemoryAuditEvent {
	action: MemoryAuditAction;
	timestamp: string; // RFC3339
	old_value?: number;
	new_value?: number;
	reason?: string;
	triggered_by?: string;
}

export interface MemoryAuditResponse {
	memoryId: string;
	events: MemoryAuditEvent[];
}

// Node type colors for visualization — bioluminescent palette
export const NODE_TYPE_COLORS: Record<string, string> = {
	fact: '#00A8FF',      // electric blue
	concept: '#1A8FA8',   // fossil teal — never purple
	event: '#FFB800',     // golden amber
	person: '#00FFD1',    // bioluminescent cyan
	place: '#00D4FF',     // bright cyan
	note: '#8B95A5',      // soft steel
	pattern: '#E07A3D',   // fossil ochre
	decision: '#FF4757',  // vivid red
};

export const EVENT_TYPE_COLORS: Record<string, string> = {
	MemoryCreated: '#00FFD1',
	MemoryUpdated: '#00A8FF',
	MemoryDeleted: '#FF4757',
	MemoryPromoted: '#00FF88',
	MemoryDemoted: '#FF6B35',
	MemorySuppressed: '#FF3B30',
	MemoryUnsuppressed: '#14E8C6',
	Rac1CascadeSwept: '#FF6B35',
	SearchPerformed: '#22C7DE',
	DeepReferenceCompleted: '#29F2A9',
	HookVerdictRecorded: '#F59E0B',
	DreamStarted: '#29F2A9',
	DreamProgress: '#1BD6FF',
	DreamCompleted: '#22C7DE',
	ConsolidationStarted: '#FFB800',
	ConsolidationCompleted: '#FF9500',
	RetentionDecayed: '#FF4757',
	ConnectionDiscovered: '#00D4FF',
	ActivationSpread: '#14E8C6',
	ImportanceScored: '#E07A3D',
	Heartbeat: '#8B95A5',
};

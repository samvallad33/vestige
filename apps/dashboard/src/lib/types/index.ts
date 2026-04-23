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
	| 'Heartbeat';

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

// Node type colors for visualization — bioluminescent palette
export const NODE_TYPE_COLORS: Record<string, string> = {
	fact: '#00A8FF',      // electric blue
	concept: '#9D00FF',   // deep violet
	event: '#FFB800',     // golden amber
	person: '#00FFD1',    // bioluminescent cyan
	place: '#00D4FF',     // bright cyan
	note: '#8B95A5',      // soft steel
	pattern: '#FF3CAC',   // hot pink
	decision: '#FF4757',  // vivid red
};

export const EVENT_TYPE_COLORS: Record<string, string> = {
	MemoryCreated: '#00FFD1',
	MemoryUpdated: '#00A8FF',
	MemoryDeleted: '#FF4757',
	MemoryPromoted: '#00FF88',
	MemoryDemoted: '#FF6B35',
	MemorySuppressed: '#A33FFF',
	MemoryUnsuppressed: '#14E8C6',
	Rac1CascadeSwept: '#6E3FFF',
	SearchPerformed: '#818CF8',
	DeepReferenceCompleted: '#C4B5FD',
	DreamStarted: '#9D00FF',
	DreamProgress: '#B44AFF',
	DreamCompleted: '#C084FC',
	ConsolidationStarted: '#FFB800',
	ConsolidationCompleted: '#FF9500',
	RetentionDecayed: '#FF4757',
	ConnectionDiscovered: '#00D4FF',
	ActivationSpread: '#14E8C6',
	ImportanceScored: '#FF3CAC',
	Heartbeat: '#8B95A5',
};

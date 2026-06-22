import type {
	MemoryListResponse,
	Memory,
	SearchResult,
	SystemStats,
	HealthCheck,
	TimelineResponse,
	GraphResponse,
	DreamResult,
	ImportanceScore,
	RetentionDistribution,
	ConsolidationResult,
	IntentionItem,
	SuppressResult,
	UnsuppressResult,
	SanhedrinAppealReason,
	SanhedrinAppealResponse,
	SanhedrinLatestResponse,
	SanhedrinTelemetryResponse
} from '$types';

const BASE = '/api';

async function fetcher<T>(path: string, options?: RequestInit): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		headers: { 'Content-Type': 'application/json' },
		...options
	});
	if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
	return res.json();
}

export const api = {
	// Memories
	memories: {
		list: (params?: Record<string, string>) => {
			const qs = params ? '?' + new URLSearchParams(params).toString() : '';
			return fetcher<MemoryListResponse>(`/memories${qs}`);
		},
		get: (id: string) => fetcher<Memory>(`/memories/${id}`),
		delete: (id: string) => fetcher<{ deleted: boolean }>(`/memories/${id}`, { method: 'DELETE' }),
		promote: (id: string) => fetcher<Memory>(`/memories/${id}/promote`, { method: 'POST' }),
		demote: (id: string) => fetcher<Memory>(`/memories/${id}/demote`, { method: 'POST' }),
		// v2.0.7: suppress + unsuppress. Anderson 2025 top-down inhibitory
		// control. Each suppress call compounds; reversible within 24h. The
		// backend emits MemorySuppressed / MemoryUnsuppressed so the 3D graph
		// plays the violet implosion / rainbow reversal.
		suppress: (id: string, reason?: string) =>
			fetcher<SuppressResult>(`/memories/${id}/suppress`, {
				method: 'POST',
				body: reason ? JSON.stringify({ reason }) : undefined
			}),
		unsuppress: (id: string) =>
			fetcher<UnsuppressResult>(`/memories/${id}/unsuppress`, { method: 'POST' })
	},

	// Search
	search: (q: string, limit = 20) =>
		fetcher<SearchResult>(`/search?q=${encodeURIComponent(q)}&limit=${limit}`),

	// Stats & Health
	stats: () => fetcher<SystemStats>('/stats'),
	health: () => fetcher<HealthCheck>('/health'),

	// Timeline
	timeline: (days = 7, limit = 200) =>
		fetcher<TimelineResponse>(`/timeline?days=${days}&limit=${limit}`),

	// Graph
	//
	// `sort` controls the default center when no query/center_id is given:
	//   - "recent" (default) — newest memory; matches user expectation of
	//     "show me what I just added". Previously the backend defaulted to
	//     "connected" which clustered on historical hotspots and hid
	//     fresh memories that hadn't accumulated edges yet.
	//   - "connected" — densest node; richer initial subgraph for a
	//     well-aged corpus. Exposed for a future UI toggle.
	graph: (params?: {
		query?: string;
		center_id?: string;
		depth?: number;
		max_nodes?: number;
		sort?: 'recent' | 'connected';
	}) => {
		const qs = params ? '?' + new URLSearchParams(
			Object.entries(params)
				.filter(([, v]) => v !== undefined)
				.map(([k, v]) => [k, String(v)])
		).toString() : '';
		return fetcher<GraphResponse>(`/graph${qs}`);
	},

	// Cognitive operations
	dream: () => fetcher<DreamResult>('/dream', { method: 'POST' }),

	explore: (fromId: string, action = 'associations', toId?: string, limit = 10) =>
		fetcher<Record<string, unknown>>('/explore', {
			method: 'POST',
			body: JSON.stringify({ from_id: fromId, action, to_id: toId, limit })
		}),

	predict: () => fetcher<Record<string, unknown>>('/predict', { method: 'POST' }),

	importance: (content: string) =>
		fetcher<ImportanceScore>('/importance', {
			method: 'POST',
			body: JSON.stringify({ content })
		}),

	consolidate: () => fetcher<ConsolidationResult>('/consolidate', { method: 'POST' }),

	retentionDistribution: () => fetcher<RetentionDistribution>('/retention-distribution'),

	// Intentions
	intentions: (status = 'active') =>
		fetcher<{ intentions: IntentionItem[]; total: number; filter: string }>(`/intentions?status=${status}`),

	// Reasoning Theater (v2.0.8): the 8-stage deep_reference cognitive pipeline.
	// Returns a reasoning chain + evidence + contradictions + supersession +
	// evolution + confidence. Emits DeepReferenceCompleted on the WebSocket so
	// the 3D graph can camera-glide + pulse + arc.
	deepReference: (query: string, depth = 20) =>
		fetcher<Record<string, unknown>>('/deep_reference', {
			method: 'POST',
			body: JSON.stringify({ query, depth })
		}),

	sanhedrin: {
		latest: () => fetcher<SanhedrinLatestResponse>('/sanhedrin/latest'),
		telemetry: (days = 7) => fetcher<SanhedrinTelemetryResponse>(`/sanhedrin/telemetry?days=${days}`),
		appeal: (reason: SanhedrinAppealReason, note?: string, claimId?: string, receiptId?: string) =>
			fetcher<SanhedrinAppealResponse>('/sanhedrin/appeal', {
				method: 'POST',
				body: JSON.stringify({ reason, note, claimId, receiptId })
			})
	},

	// Agent Black Box (v2.2): replayable agent-run traces. The runId in a tool
	// result threads through here unchanged — one id, end to end.
	traces: {
		list: (limit = 50) => fetcher<TraceRunListResponse>(`/traces?limit=${limit}`),
		get: (runId: string) => fetcher<TraceDetail>(`/traces/${encodeURIComponent(runId)}`),
		exportUrl: (runId: string) => `${BASE}/traces/${encodeURIComponent(runId)}/export`
	},

	// Memory Receipts (v2.2): the nutrition label for a retrieval.
	receipts: {
		list: (limit = 50) => fetcher<ReceiptListResponse>(`/receipts?limit=${limit}`),
		get: (receiptId: string) => fetcher<Receipt>(`/receipts/${encodeURIComponent(receiptId)}`)
	},

	// Memory PRs (v2.2): the risk-gated brain-change review queue.
	memoryPrs: {
		list: (status?: string, limit = 100) => {
			const qs = new URLSearchParams();
			if (status) qs.set('status', status);
			qs.set('limit', String(limit));
			return fetcher<MemoryPrListResponse>(`/memory-prs?${qs.toString()}`);
		},
		get: (id: string) => fetcher<MemoryPr>(`/memory-prs/${encodeURIComponent(id)}`),
		act: (id: string, action: MemoryPrAction) =>
			fetcher<Record<string, unknown>>(`/memory-prs/${encodeURIComponent(id)}/${action}`, {
				method: 'POST'
			}),
		getMode: () => fetcher<{ mode: ReviewMode; pendingCount: number }>('/memory-prs/mode'),
		setMode: (mode: ReviewMode) =>
			fetcher<{ mode: ReviewMode }>('/memory-prs/mode', {
				method: 'POST',
				body: JSON.stringify({ mode })
			})
	}
};

// ---------------------------------------------------------------------------
// Agent Black Box / Receipts / Memory PR types
// ---------------------------------------------------------------------------

export type TraceRunSummary = {
	runId: string;
	firstTool: string | null;
	eventCount: number;
	retrievedCount: number;
	suppressedCount: number;
	writeCount: number;
	vetoCount: number;
	startedAt: number;
	lastAt: number;
};

export type TraceRunListResponse = { total: number; runs: TraceRunSummary[] };

/** One trace event — discriminated on `type`, matching the Rust schema. */
export type TraceEvent =
	| { type: 'mcp.call'; runId: string; tool: string; argsHash: string; at: number }
	| { type: 'memory.retrieve'; runId: string; ids: string[]; activation: Record<string, number>; at: number }
	| { type: 'memory.suppress'; runId: string; id: string; reason: string; at: number }
	| { type: 'memory.write'; runId: string; id: string; diff: unknown; source: string; at: number }
	| { type: 'contradiction.detected'; runId: string; ids: string[]; winnerId?: string; detail: string; at: number }
	| { type: 'sanhedrin.veto'; runId: string; claim: string; evidenceIds: string[]; confidence: number; at: number }
	| { type: 'dream.patch'; runId: string; proposalIds: string[]; at: number };

export type TraceDetail = {
	runId: string;
	summary: Omit<TraceRunSummary, 'runId'> | null;
	events: TraceEvent[];
};

export type Receipt = {
	receipt_id: string;
	retrieved: string[];
	suppressed: { id: string; reason: string }[];
	activation_path: string[];
	trust_floor: number;
	decay_risk: 'low' | 'medium' | 'high';
	mutations: { id: string; kind: string; note?: string }[];
};

export type ReceiptListResponse = { total: number; receipts: Receipt[] };

export type MemoryPrAction =
	| 'promote'
	| 'merge'
	| 'supersede'
	| 'quarantine'
	| 'forget'
	| 'ask_agent_why';

export type ReviewMode = 'fast' | 'risk_gated' | 'paranoid';

export type MemoryPr = {
	id: string;
	kind: string;
	status: string;
	title: string;
	diff: Record<string, unknown>;
	signals: { code: string; detail: string }[];
	subject_id?: string;
	run_id?: string;
	created_at: string;
	decided_at?: string;
	decision?: string;
};

export type MemoryPrListResponse = {
	total: number;
	pendingCount: number;
	mode: ReviewMode;
	prs: MemoryPr[];
};

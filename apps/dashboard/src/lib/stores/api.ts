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
	UnsuppressResult
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
		})
};

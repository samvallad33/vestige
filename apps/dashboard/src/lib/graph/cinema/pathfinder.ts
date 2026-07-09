// Memory Cinema — Tier 3: the bulletproof pathfinder.
//
// Plans a cinematic tour through the REAL memory graph using nothing but the
// nodes + edges the backend already returns. This is the deterministic engine
// that ALWAYS drives the camera, regardless of which narration tier (backend
// LLM / local captions / none) is active. No WebGPU, no network, no LLM — if
// everything else fails, this still produces a coherent, watchable flythrough.
//
// The path is intentionally a STORY, not a raw BFS dump:
//   1. start at the center (the memory the graph is focused on)
//   2. visit its strongest-weighted connections (what it's most tied to)
//   3. detour to a contradiction edge if one exists (tension = interesting)
//   4. end on a recently-created node (where the mind is now)
// Falling back to plain weighted BFS when those signals are absent.

import type { GraphNode, GraphEdge } from '$types';

export interface CinemaBeat {
	/** Node this beat centers the camera on. */
	nodeId: string;
	/** The node payload, for the narrator + visuals. */
	node: GraphNode;
	/** Edge traversed to arrive here (null for the opening beat). */
	viaEdge: GraphEdge | null;
	/** Why this beat exists — drives the deterministic caption + visual emphasis. */
	kind: 'origin' | 'connection' | 'contradiction' | 'recent' | 'bridge' | 'surprise';
	/** 0..1 emphasis used by the sandbox to spike emissive/bloom on arrival. */
	intensity: number;
}

export interface CinemaPath {
	beats: CinemaBeat[];
	/** The node the requested centerId resolved to (which the tour actually
	 * starts from). May differ from the requested centerId when it was missing,
	 * in which case `pivoted` is true — callers can surface this if they care. */
	centerId: string;
	/** True when the requested centerId did not exist and we picked a start node. */
	pivoted: boolean;
	/** Edges that should visibly "flow" during the tour, in beat order. */
	flowEdges: GraphEdge[];
}

export interface Adjacency {
	[nodeId: string]: { edge: GraphEdge; otherId: string }[];
}

export function buildAdjacency(edges: GraphEdge[]): Adjacency {
	const adj: Adjacency = {};
	for (const edge of edges) {
		(adj[edge.source] ??= []).push({ edge, otherId: edge.target });
		(adj[edge.target] ??= []).push({ edge, otherId: edge.source });
	}
	// Strongest connections first so the tour visits the most meaningful ties.
	for (const id of Object.keys(adj)) {
		adj[id].sort((a, b) => (b.edge.weight ?? 0) - (a.edge.weight ?? 0));
	}
	return adj;
}

export function isContradictionEdge(edge: GraphEdge): boolean {
	const t = (edge.type ?? '').toLowerCase();
	return t.includes('contradict') || t.includes('conflict') || t.includes('supersede');
}

/**
 * A real CAUSAL edge (ConnectionType::Causal / DiscoveredConnectionType::
 * CausalChain, serialized as connection_type "causal"). Vestige is the only
 * memory product that computes directed causation (RSB backward backfill), so
 * a path that PREFERS these edges follows the true cause chain, not mere
 * co-occurrence — the moat behind the causal recall wavefront (Phase 4).
 */
export function isCausalEdge(edge: GraphEdge): boolean {
	const t = (edge.type ?? '').toLowerCase();
	return t === 'causal' || t.includes('causal') || t.includes('cause');
}

export function recencyOf(node: GraphNode): number {
	// Larger = more recent. Tolerates missing/invalid timestamps.
	const t = Date.parse(node.updatedAt || node.createdAt || '');
	return Number.isFinite(t) ? t : 0;
}

/**
 * Plan a cinematic path over the real graph.
 *
 * @param maxBeats hard cap on tour length (keeps the flythrough watchable).
 * Deterministic: same inputs always yield the same path (no randomness), so the
 * recorded launch GIF is reproducible.
 */
export function planCinemaPath(
	nodes: GraphNode[],
	edges: GraphEdge[],
	centerId: string,
	maxBeats = 7,
	opts: { preferCausal?: boolean } = {}
): CinemaPath {
	const preferCausal = opts.preferCausal ?? false;
	const byId = new Map(nodes.map((n) => [n.id, n]));
	const empty: CinemaPath = { beats: [], centerId, pivoted: false, flowEdges: [] };
	if (nodes.length === 0) return empty;

	// Resolve a real starting node: prefer centerId, else the explicit center
	// flag, else the most-connected node, else the first node. Track whether we
	// had to pivot off the requested centerId so callers can surface it.
	const adj = buildAdjacency(edges);
	const requestedExists = byId.has(centerId);
	let startId = requestedExists ? centerId : '';
	if (!startId) startId = nodes.find((n) => n.isCenter)?.id ?? '';
	if (!startId) {
		startId = nodes
			.map((n) => ({ id: n.id, deg: adj[n.id]?.length ?? 0 }))
			.sort((a, b) => b.deg - a.deg)[0].id;
	}
	const start = byId.get(startId);
	if (!start) return empty;
	const pivoted = !requestedExists;

	const visited = new Set<string>([startId]);
	const beats: CinemaBeat[] = [
		{ nodeId: startId, node: start, viaEdge: null, kind: 'origin', intensity: 1 },
	];
	const flowEdges: GraphEdge[] = [];

	// Greedy weighted walk: from the current frontier, step to the strongest
	// unvisited neighbour, with a one-time detour to a contradiction if reachable.
	let current = startId;
	let contradictionUsed = false;

	while (beats.length < maxBeats) {
		const neighbours = adj[current] ?? [];

		// Causal recall (Phase 4): when asked, PREFER a real causal edge every
		// step so the path traces the true cause chain (RSB substrate), not just
		// the strongest co-occurrence. neighbours are already weight-sorted, so
		// this picks the strongest causal tie first.
		let next: { edge: GraphEdge; otherId: string } | undefined;
		if (preferCausal) {
			next = neighbours.find((n) => !visited.has(n.otherId) && isCausalEdge(n.edge));
		}
		// Prefer an unused contradiction edge once — tension makes a better story.
		if (!next && !contradictionUsed) {
			next = neighbours.find((n) => !visited.has(n.otherId) && isContradictionEdge(n.edge));
			if (next) contradictionUsed = true;
		}
		// Otherwise the strongest unvisited tie.
		if (!next) next = neighbours.find((n) => !visited.has(n.otherId));

		// Dead end: hop to the most recent unvisited node anywhere (a "bridge"
		// cut) so the tour can keep going instead of stalling.
		if (!next) {
			const remaining = nodes
				.filter((n) => !visited.has(n.id))
				.sort((a, b) => recencyOf(b) - recencyOf(a));
			if (remaining.length === 0) break;
			const node = remaining[0];
			visited.add(node.id);
			beats.push({ nodeId: node.id, node, viaEdge: null, kind: 'bridge', intensity: 0.6 });
			current = node.id;
			continue;
		}

		const node = byId.get(next.otherId);
		if (!node) {
			visited.add(next.otherId);
			continue;
		}
		visited.add(node.id);
		flowEdges.push(next.edge);
		const causal = preferCausal && isCausalEdge(next.edge);
		beats.push({
			nodeId: node.id,
			node,
			viaEdge: next.edge,
			// Kind stays within the shared Cinema union (the narrator maps it) —
			// a causal step reads as a contradiction-style backward hop for the
			// wavefront; consumers that care read viaEdge.type directly.
			kind: isContradictionEdge(next.edge) ? 'contradiction' : 'connection',
			intensity: isContradictionEdge(next.edge) || causal
				? 1
				: Math.min(1, 0.55 + (next.edge.weight ?? 0) * 0.45),
		});
		current = node.id;
	}

	// Closing beat: end on the single most-recent node not already the finale,
	// so the tour lands on "where the memory is now". Only if it adds variety.
	if (beats.length < maxBeats) {
		const last = beats[beats.length - 1].nodeId;
		const recent = nodes
			.filter((n) => n.id !== last)
			.sort((a, b) => recencyOf(b) - recencyOf(a))[0];
		if (recent && !beats.some((b) => b.nodeId === recent.id)) {
			beats.push({ nodeId: recent.id, node: recent, viaEdge: null, kind: 'recent', intensity: 0.8 });
		}
	}

	return { beats, centerId: startId, pivoted, flowEdges };
}

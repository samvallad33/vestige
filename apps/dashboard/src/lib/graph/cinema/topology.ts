// The Auteur — graph signal extraction.
//
// Pure, dependency-free statistics over the REAL /api/graph data, computed once
// per Cinema launch. These signals are what gives the AI director something
// meaningful to direct: which memory is most load-bearing (betweenness), where
// tension lives (contradictions), what's surprising (distant-but-plausible
// links), what's fading (low retention / suppression). No LLM, no WebGPU, no
// network — fully headless-testable.

import type { GraphNode, GraphEdge } from '$types';
import { buildAdjacency, recencyOf, isContradictionEdge } from './pathfinder';

export interface NodeSignal {
	nodeId: string;
	/** Raw connection count. */
	degree: number;
	/** Brandes betweenness centrality, normalized 0..1 — how load-bearing this
	 * memory is as a bridge between clusters. The director favors high-betweenness
	 * nodes for hero shots. */
	betweenness: number;
	/** Connected-component id (which cluster of memory this belongs to). */
	clusterId: number;
	/** 0..1, 1 = most recent. */
	recencyRank: number;
	/** FSRS retention 0..1. */
	retention: number;
	/** Suppression pressure 0..1 (memory actively being forgotten). */
	suppression: number;
}

export interface EdgeSignal {
	source: string;
	target: string;
	isContradiction: boolean;
	isMergeSupersede: boolean;
	/** 0..1: high when endpoints share neighbors (plausible) yet the edge weight
	 * is low (distant) — a surprising, non-obvious connection. */
	surprise: number;
	weight: number;
}

export interface GraphSignals {
	nodes: Map<string, NodeSignal>;
	edges: EdgeSignal[];
	clusterCount: number;
	/** Node id with the single highest betweenness — the graph's keystone. */
	peakBetweennessId: string;
}

function isMergeSupersedeEdge(edge: GraphEdge): boolean {
	const t = (edge.type ?? '').toLowerCase();
	return t.includes('merge') || t.includes('supersede') || t.includes('duplicate');
}

/**
 * Brandes' algorithm for betweenness centrality on an unweighted, undirected
 * graph. O(V·E) — fine for /api/graph payloads. Returns raw (unnormalized)
 * scores keyed by node id; the caller normalizes.
 */
function brandesBetweenness(nodeIds: string[], adj: Record<string, { otherId: string }[]>): Map<string, number> {
	const cb = new Map<string, number>();
	for (const v of nodeIds) cb.set(v, 0);

	for (const s of nodeIds) {
		const stack: string[] = [];
		const pred = new Map<string, string[]>();
		const sigma = new Map<string, number>();
		const dist = new Map<string, number>();
		for (const v of nodeIds) {
			pred.set(v, []);
			sigma.set(v, 0);
			dist.set(v, -1);
		}
		sigma.set(s, 1);
		dist.set(s, 0);

		// BFS (unweighted shortest paths).
		const queue: string[] = [s];
		let head = 0;
		while (head < queue.length) {
			const v = queue[head++];
			stack.push(v);
			for (const { otherId: w } of adj[v] ?? []) {
				if ((dist.get(w) ?? -1) < 0) {
					dist.set(w, (dist.get(v) ?? 0) + 1);
					queue.push(w);
				}
				if ((dist.get(w) ?? -1) === (dist.get(v) ?? 0) + 1) {
					sigma.set(w, (sigma.get(w) ?? 0) + (sigma.get(v) ?? 0));
					pred.get(w)!.push(v);
				}
			}
		}

		// Accumulation (back-propagate dependencies).
		const delta = new Map<string, number>();
		for (const v of nodeIds) delta.set(v, 0);
		while (stack.length > 0) {
			const w = stack.pop()!;
			for (const v of pred.get(w) ?? []) {
				const c = ((sigma.get(v) ?? 0) / (sigma.get(w) || 1)) * (1 + (delta.get(w) ?? 0));
				delta.set(v, (delta.get(v) ?? 0) + c);
			}
			if (w !== s) cb.set(w, (cb.get(w) ?? 0) + (delta.get(w) ?? 0));
		}
	}
	return cb;
}

/** Union-find connected components → a cluster id per node. */
function components(nodeIds: string[], edges: GraphEdge[]): { clusterOf: Map<string, number>; count: number } {
	const parent = new Map<string, string>();
	for (const id of nodeIds) parent.set(id, id);
	const find = (x: string): string => {
		let root = x;
		while (parent.get(root) !== root) root = parent.get(root)!;
		// Path compression.
		let cur = x;
		while (parent.get(cur) !== root) {
			const next = parent.get(cur)!;
			parent.set(cur, root);
			cur = next;
		}
		return root;
	};
	const union = (a: string, b: string) => {
		const ra = find(a);
		const rb = find(b);
		if (ra !== rb) parent.set(ra, rb);
	};
	for (const e of edges) {
		if (parent.has(e.source) && parent.has(e.target)) union(e.source, e.target);
	}
	const rootToCluster = new Map<string, number>();
	const clusterOf = new Map<string, number>();
	let next = 0;
	for (const id of nodeIds) {
		const r = find(id);
		if (!rootToCluster.has(r)) rootToCluster.set(r, next++);
		clusterOf.set(id, rootToCluster.get(r)!);
	}
	return { clusterOf, count: next };
}

/**
 * Compute all director signals from the real graph. Pure; safe to call once at
 * launch. Caps betweenness work on very large graphs by limiting to the
 * top-degree subset (the only nodes that can carry meaningful centrality).
 */
export function computeSignals(nodes: GraphNode[], edges: GraphEdge[]): GraphSignals {
	const nodeIds = nodes.map((n) => n.id);
	const adj = buildAdjacency(edges);

	// Recency ranking (0..1, 1 = newest).
	const byRecency = [...nodes].sort((a, b) => recencyOf(a) - recencyOf(b));
	const recencyRank = new Map<string, number>();
	byRecency.forEach((n, i) => recencyRank.set(n.id, nodes.length > 1 ? i / (nodes.length - 1) : 1));

	// Betweenness — guard pathological sizes: above the cap, compute on the
	// top-degree subset (others get 0; they can't be meaningful bridges anyway).
	const BETWEENNESS_CAP = 600;
	let betweennessNodes = nodeIds;
	if (nodeIds.length > BETWEENNESS_CAP) {
		betweennessNodes = [...nodeIds]
			.sort((a, b) => (adj[b]?.length ?? 0) - (adj[a]?.length ?? 0))
			.slice(0, BETWEENNESS_CAP);
	}
	const rawBetween = brandesBetweenness(betweennessNodes, adj);
	let maxBetween = 0;
	for (const v of rawBetween.values()) maxBetween = Math.max(maxBetween, v);

	const { clusterOf, count: clusterCount } = components(nodeIds, edges);

	const maxSuppression = Math.max(1, ...nodes.map((n) => n.suppression_count ?? 0));

	const nodeSignals = new Map<string, NodeSignal>();
	let peakBetweennessId = nodeIds[0] ?? '';
	let peakVal = -1;
	for (const n of nodes) {
		const bt = maxBetween > 0 ? (rawBetween.get(n.id) ?? 0) / maxBetween : 0;
		if (bt > peakVal) {
			peakVal = bt;
			peakBetweennessId = n.id;
		}
		nodeSignals.set(n.id, {
			nodeId: n.id,
			degree: adj[n.id]?.length ?? 0,
			betweenness: bt,
			clusterId: clusterOf.get(n.id) ?? 0,
			recencyRank: recencyRank.get(n.id) ?? 0,
			retention: clamp01(n.retention ?? 0),
			suppression: clamp01((n.suppression_count ?? 0) / maxSuppression),
		});
	}

	// Edge signals incl. surprise (shared-neighbor overlap × edge distance).
	const neighborSets = new Map<string, Set<string>>();
	for (const id of nodeIds) neighborSets.set(id, new Set((adj[id] ?? []).map((a) => a.otherId)));
	const edgeSignals: EdgeSignal[] = edges.map((e) => {
		const a = neighborSets.get(e.source);
		const b = neighborSets.get(e.target);
		let shared = 0;
		if (a && b) {
			const [small, large] = a.size < b.size ? [a, b] : [b, a];
			for (const x of small) if (large.has(x)) shared++;
		}
		const union = (a?.size ?? 0) + (b?.size ?? 0) - shared || 1;
		const overlap = shared / union; // Jaccard: structural plausibility.
		const distance = 1 - clamp01(e.weight ?? 0); // low weight = semantically distant.
		return {
			source: e.source,
			target: e.target,
			isContradiction: isContradictionEdge(e),
			isMergeSupersede: isMergeSupersedeEdge(e),
			surprise: clamp01(overlap * distance * 2), // plausible AND distant = surprising.
			weight: e.weight ?? 0,
		};
	});

	return { nodes: nodeSignals, edges: edgeSignals, clusterCount, peakBetweennessId };
}

function clamp01(x: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
}

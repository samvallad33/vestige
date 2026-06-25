// Phantom Brain — deterministic "seed-from-identity" generative memory graph.
//
// The viral artifact engine: any string (GitHub handle, email, a typed memory)
// deterministically produces a unique, believable memory graph. Same input ->
// same brain, every time, on every device (no RNG, no backend). This is what
// makes the share artifact one-of-one per visitor while the page stays static.
//
// Output matches the dashboard's GraphNode/GraphEdge contract so it feeds the
// real MemoryCinema + 3D graph components unchanged.

import type { GraphNode, GraphEdge } from '$types';

// ---- deterministic hashing (FNV-1a 32-bit) + seeded PRNG (mulberry32) -------

function fnv1a(str: string): number {
	let h = 0x811c9dc5;
	for (let i = 0; i < str.length; i++) {
		h ^= str.charCodeAt(i);
		h = Math.imul(h, 0x01000193);
	}
	return h >>> 0;
}

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ---- believable memory vocabulary, themed by node type ----------------------

const NODE_TYPES = ['decision', 'fact', 'preference', 'pattern', 'insight', 'identity'] as const;

const FRAGMENTS: Record<string, string[]> = {
	decision: [
		'Chose Postgres over Mongo for the ledger',
		'Migrated auth to short-lived JWTs',
		'Adopted a monorepo with pnpm workspaces',
		'Picked Rust for the hot path',
		'Standardized on trunk-based deploys',
		'Moved embeddings on-device for privacy'
	],
	fact: [
		'The API rate-limits at 600 req/min',
		'Staging mirrors prod with seeded data',
		'CI runs the full suite in 4 minutes',
		'The retry budget is 3 with jitter',
		'Feature flags live in the edge config',
		'Cold starts dropped to 40ms after the rewrite'
	],
	preference: [
		'Prefers small, reviewable PRs',
		'Always writes the test first',
		'Hates implicit any',
		'Likes results over exceptions',
		'Wants logs structured, never printf',
		'Trusts source over memory'
	],
	pattern: [
		'Reaches for a state machine when flows branch',
		'Caches at the edge, invalidates on write',
		'Wraps external calls in a circuit breaker',
		'Names things for what they do, not how',
		'Pushes side effects to the boundary',
		'Treats deletes as tombstones, never hard'
	],
	insight: [
		'The flaky test was a clock, not the code',
		'Most latency was one N+1 query',
		'The bug only reproduced under real load',
		'The slow path was JSON, not the DB',
		'Two services were fighting the same lock',
		'The memory leak was an unclosed stream'
	],
	identity: [
		'Ships fast, refuses to ship broken',
		'Builds for the developer who comes after',
		'Optimizes for the next maintainer',
		'Treats taste as a feature',
		'Defaults to the ambitious version',
		'Owns the whole vertical slice'
	]
};

const TAGS = [
	'architecture', 'auth', 'performance', 'ci', 'database', 'privacy',
	'refactor', 'incident', 'design', 'workflow', 'security', 'launch'
];

export interface PhantomBrain {
	seed: string;
	nodes: GraphNode[];
	edges: GraphEdge[];
	stats: {
		memories: number;
		connections: number;
		topConcept: string;
		dominantType: string;
	};
}

/**
 * Build a deterministic phantom brain from any identity string.
 * @param identity GitHub handle, email, or a typed memory — anything.
 * @param size     node count (default scales nicely for the hero, 28).
 */
export function seedPhantomBrain(identity: string, size = 28): PhantomBrain {
	const clean = identity.trim().toLowerCase() || 'anonymous-builder';
	const rng = mulberry32(fnv1a(`vestige:${clean}`));
	const pick = <T>(arr: readonly T[]): T => arr[Math.floor(rng() * arr.length)];

	const now = Date.now();
	const typeCounts: Record<string, number> = {};
	const nodes: GraphNode[] = [];

	for (let i = 0; i < size; i++) {
		const type = pick(NODE_TYPES);
		typeCounts[type] = (typeCounts[type] ?? 0) + 1;
		const ageMs = Math.floor(rng() * 1000 * 60 * 60 * 24 * 120); // up to ~120 days
		const created = new Date(now - ageMs).toISOString();
		// Older memories decay; a few are freshly reinforced.
		const retention = Math.max(0.12, Math.min(1, 1 - (ageMs / (1000 * 60 * 60 * 24 * 140)) + (rng() - 0.5) * 0.3));
		const tagCount = 1 + Math.floor(rng() * 2);
		const tags = Array.from({ length: tagCount }, () => pick(TAGS));
		nodes.push({
			id: `n${i}`,
			label: pick(FRAGMENTS[type]),
			type,
			retention: Math.round(retention * 100) / 100,
			tags: [...new Set(tags)],
			createdAt: created,
			updatedAt: created,
			isCenter: i === 0
		});
	}

	// Edges: a connected backbone (so the graph never fragments) plus seeded
	// cross-links weighted toward shared tags / strong memories.
	const edges: GraphEdge[] = [];
	const edgeKey = new Set<string>();
	const addEdge = (a: number, b: number, type: string) => {
		if (a === b) return;
		const k = a < b ? `${a}-${b}` : `${b}-${a}`;
		if (edgeKey.has(k)) return;
		edgeKey.add(k);
		edges.push({
			source: `n${a}`,
			target: `n${b}`,
			weight: Math.round((0.3 + rng() * 0.7) * 100) / 100,
			type
		});
	};
	// backbone
	for (let i = 1; i < size; i++) addEdge(i, Math.floor(rng() * i), 'relates_to');
	// cross-links — denser for a richer constellation
	const extra = Math.floor(size * 1.4);
	for (let i = 0; i < extra; i++) {
		addEdge(Math.floor(rng() * size), Math.floor(rng() * size), pick(['supports', 'contradicts', 'relates_to', 'supersedes']));
	}

	const dominantType = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'fact';
	// Top concept = most-seen tag.
	const tagFreq: Record<string, number> = {};
	for (const n of nodes) for (const t of n.tags) tagFreq[t] = (tagFreq[t] ?? 0) + 1;
	const topConcept = Object.entries(tagFreq).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'architecture';

	return {
		seed: clean,
		nodes,
		edges,
		stats: {
			memories: nodes.length,
			connections: edges.length,
			topConcept,
			dominantType
		}
	};
}

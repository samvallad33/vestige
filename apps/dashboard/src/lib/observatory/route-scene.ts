/**
 * RouteSceneModel — the shared contract between a route's data adapter and its
 * RouteStage WebGPU organ (design council Opus 4.8 × GPT-5.5, Jul 8 2026).
 *
 * The discipline test is enforced HERE, in the type system: every visual
 * primitive MUST carry a `source` provenance (a real memory id, event, receipt,
 * or a named real scalar). A hero that can't name its source doesn't ship.
 * Swap the backend value for Math.random() and `source` becomes a lie the code
 * review catches.
 *
 * Adapters (`*-scene.ts`) turn API/WebSocket responses into this model; the
 * RouteStage renders it through the shared engine + cognitive field pass. Pure
 * data — no Svelte/DOM/GPU imports.
 */

/** The organs of the Cognitive OS — one per transformed route. */
export type RouteOrgan =
	| 'reasoning'
	| 'timeline'
	| 'feed'
	| 'schedule'
	| 'duplicates'
	| 'contradictions'
	| 'patterns'
	| 'memories'
	| 'explore'
	| 'importance'
	| 'activation'
	| 'dreams'
	| 'intentions'
	| 'blackbox'
	| 'witness'
	| 'memory-prs'
	| 'stats'
	| 'settings';

/**
 * Provenance for a primitive — the discipline-test receipt. Every node/edge/
 * event/label points to something REAL. `kind` says what backend fact backs it.
 */
export interface Provenance {
	kind: 'memory' | 'event' | 'receipt' | 'pair' | 'trace' | 'pr' | 'pattern' | 'scalar';
	/** The real id (memory id, receipt id, run id, pair key, …). */
	id: string;
	/** For kind:'scalar' — the named metric + its real value (e.g. 'dueForReview'). */
	scalar?: { name: string; value: number };
}

/** A living cell — a memory or record rendered in the organism. */
export interface RouteNode {
	source: Provenance;
	/** Stable index in the GPU buffer (assigned by the adapter). */
	index: number;
	label: string;
	/** FSRS retrievability 0..1 — the oxygen level. Drives hue + cracking. */
	retention: number;
	/** FSRS stability in days — drives radius (sqrt) + sedimentation depth. */
	stability?: number;
	/** ISO last-access — drives live decay recompute (fsrs.ts). */
	lastAccessed?: string;
	/** 0..1 activation concentration — halo opacity + chemotaxis. */
	activation?: number;
	/** 0..1 trust / verifier confidence — membrane thickness. */
	trust?: number;
	/** suppression count (>0 = scarred). */
	suppression?: number;
	tags: string[];
	type: string;
}

/** A tension-bearing fiber (axon), not a line. Weight tightens it. */
export interface RouteEdge {
	source: Provenance;
	sourceIndex: number;
	targetIndex: number;
	weight: number;
	/** Real connection type — 'causal' fires the retrograde grammar. */
	kind: string;
}

/** A discrete cognitive event to animate (from the live feed or a receipt). */
export interface RouteEvent {
	source: Provenance;
	/** VestigeEvent type — maps to an impulse color (cognitive-palette). */
	type: string;
	/** Target node index if the event binds to a cell, else -1. */
	targetIndex: number;
	/** Monotonic frame it fired (assigned by the LiveBridge / adapter). */
	frame: number;
	/** Free scalar (energy, confidence, cascade count, …). */
	energy: number;
}

/** A named proof the organ can etch as an MSDF scar / open on click. */
export interface RouteReceipt {
	source: Provenance;
	label: string;
	/** Node indices this receipt lights (evidence, path, pair). */
	nodeIndices: number[];
}

/**
 * The full scene an organ renders. `scalars` holds route-level real metrics
 * (endangered fraction, due count, similarity threshold, …) that drive global
 * field behavior. Every entry traces to a real backend fact.
 */
export interface RouteSceneModel {
	organ: RouteOrgan;
	nodes: RouteNode[];
	edges: RouteEdge[];
	events: RouteEvent[];
	receipts: RouteReceipt[];
	/** Route-level real metrics (name → value). */
	scalars: Record<string, number>;
	/** True when the organ has real data; false → honest empty state. */
	alive: boolean;
}

/** An empty, honest scene — the field breathes, nothing is invented. */
export function emptyScene(organ: RouteOrgan): RouteSceneModel {
	return { organ, nodes: [], edges: [], events: [], receipts: [], scalars: {}, alive: false };
}

/**
 * Dev-only discipline assertion: every node/edge/event/receipt must carry a
 * `source`. Call in adapters (behind import.meta.env.DEV) so a screensaver
 * primitive — one with no real provenance — trips the review before it ships.
 */
export function assertProvenance(scene: RouteSceneModel): void {
	const bad: string[] = [];
	const check = (arr: { source?: Provenance }[], what: string) => {
		arr.forEach((p, i) => {
			if (!p.source || !p.source.kind || (!p.source.id && !p.source.scalar)) {
				bad.push(`${what}[${i}]`);
			}
		});
	};
	check(scene.nodes, 'node');
	check(scene.edges, 'edge');
	check(scene.events, 'event');
	check(scene.receipts, 'receipt');
	if (bad.length > 0) {
		throw new Error(
			`[discipline-test] ${scene.organ}: ${bad.length} primitive(s) without real provenance: ${bad
				.slice(0, 8)
				.join(', ')}. Every hero primitive must point to a real memory/event/receipt/scalar.`
		);
	}
}

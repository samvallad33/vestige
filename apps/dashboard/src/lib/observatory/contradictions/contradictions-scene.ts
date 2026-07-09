import {
	assertProvenance,
	type Provenance,
	type RouteEdge,
	type RouteEvent,
	type RouteNode,
	type RouteReceipt,
	type RouteSceneModel
} from '$lib/observatory/route-scene';
import type { ContradictionPair, VestigeEvent } from '$types';

export interface ContradictionMemorySide {
	id: string;
	preview: string;
	trust: number;
	date: string;
	type?: string;
	tags?: string[];
	provenance: Provenance;
}

export interface ImmuneSynapsePair {
	id: string;
	stronger: ContradictionMemorySide;
	weaker: ContradictionMemorySide;
	topic_overlap: number;
	topic: string;
	trust_delta: number;
	date_diff_days: number;
	resolved: boolean;
	provenance: Provenance;
	receipt: {
		label: string;
		evidence: Record<string, unknown>;
	};
}

export interface ContradictionsScene extends RouteSceneModel {
	organ: 'contradictions';
	pairs: ImmuneSynapsePair[];
	raw: {
		apiPairs: unknown[];
		deepReferenceEvents: VestigeEvent[];
	};
}

function record(v: unknown): Record<string, unknown> {
	return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function text(v: unknown, fallback = ''): string {
	return typeof v === 'string' ? v : v == null ? fallback : String(v);
}

function num(v: unknown, fallback = 0): number {
	return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function clamp01(v: number): number {
	return Math.max(0, Math.min(1, v));
}

function confidence01(v: unknown): number {
	const n = num(v, 0);
	return clamp01(n > 1 ? n / 100 : n);
}

function source(kind: Provenance['kind'], id: string, scalar?: Provenance['scalar']): Provenance {
	return scalar ? { kind, id, scalar } : { kind, id: id || `${kind}:unknown` };
}

function scalarSource(name: string, value: number): Provenance {
	return { kind: 'scalar', id: `contradictions.${name}`, scalar: { name, value } };
}

function sideFromRecord(rawSide: Record<string, unknown>, fallbackId: string, fallbackPreview = ''): ContradictionMemorySide {
	const id = text(rawSide.id ?? rawSide.memory_id ?? fallbackId);
	const trust = confidence01(rawSide.trust ?? rawSide.trust_score ?? 0);
	return {
		id,
		preview: text(rawSide.preview ?? rawSide.content ?? fallbackPreview ?? id),
		trust,
		date: text(rawSide.date ?? rawSide.created_at ?? rawSide.createdAt ?? ''),
		type: rawSide.node_type ? text(rawSide.node_type) : rawSide.nodeType ? text(rawSide.nodeType) : undefined,
		tags: Array.isArray(rawSide.tags) ? rawSide.tags.map((t) => text(t)).filter(Boolean) : undefined,
		provenance: source('memory', id)
	};
}

function pairKey(a: string, b: string): string {
	return `contradiction:${a}:${b}`;
}

function normalizeApiPair(input: unknown, index: number): ImmuneSynapsePair | null {
	const c = record(input);
	const strongerRaw = record(c.stronger);
	const weakerRaw = record(c.weaker);

	let stronger: ContradictionMemorySide;
	let weaker: ContradictionMemorySide;
	let topicOverlap: number;
	let topic: string;
	let dateDiffDays: number;

	if (Object.keys(strongerRaw).length > 0 || Object.keys(weakerRaw).length > 0) {
		stronger = sideFromRecord(strongerRaw, '');
		weaker = sideFromRecord(weakerRaw, '');
		topicOverlap = clamp01(num(c.topic_overlap ?? c.similarity, 0));
		topic = text(c.topic ?? c.summary ?? 'trust-weighted contradiction');
		dateDiffDays = num(c.date_diff_days, 0);
	} else {
		const a = sideFromRecord(
			{
				id: c.memory_a_id ?? c.a_id,
				preview: c.memory_a_preview,
				trust: c.trust_a,
				date: c.memory_a_created,
				node_type: c.memory_a_type,
				tags: c.memory_a_tags
			},
			''
		);
		const b = sideFromRecord(
			{
				id: c.memory_b_id ?? c.b_id,
				preview: c.memory_b_preview,
				trust: c.trust_b,
				date: c.memory_b_created,
				node_type: c.memory_b_type,
				tags: c.memory_b_tags
			},
			''
		);
		// /api/contradictions currently exposes chronological a/b; Organ 3 needs
		// stronger/weaker membranes, so choose by real trust while retaining both IDs.
		[stronger, weaker] = a.trust >= b.trust ? [a, b] : [b, a];
		topicOverlap = clamp01(num(c.topic_overlap ?? c.similarity, 0));
		topic = text(c.topic ?? 'trust-weighted contradiction');
		dateDiffDays = num(c.date_diff_days, 0);
	}

	if (!stronger.id || !weaker.id) return null;
	const id = pairKey(stronger.id, weaker.id);
	const trustDelta = Math.abs(stronger.trust - weaker.trust);
	return {
		id,
		stronger,
		weaker,
		topic_overlap: topicOverlap,
		topic,
		trust_delta: trustDelta,
		date_diff_days: dateDiffDays,
		resolved: text(c.status).toLowerCase() === 'resolved' || Boolean(c.resolved),
		provenance: source('pair', id),
		receipt: {
			label: `immune synapse ${index + 1}: ${stronger.id.slice(0, 8)} ↔ ${weaker.id.slice(0, 8)}`,
			evidence: c
		}
	};
}

function normalizeDeepReferenceEvent(event: VestigeEvent, pairIndexBase: number): ImmuneSynapsePair[] {
	if (event.type !== 'DeepReferenceCompleted') return [];
	const pairs = Array.isArray(event.data?.contradiction_pairs) ? event.data.contradiction_pairs : [];
	return pairs
		.map((entry, i): ImmuneSynapsePair | null => {
			const tuple = Array.isArray(entry) ? entry : [];
			const a = text(tuple[0]);
			const b = text(tuple[1]);
			if (!a || !b) return null;
			const stronger = sideFromRecord({ id: a, preview: `DeepReference contradiction side ${a.slice(0, 8)}` }, a);
			const weaker = sideFromRecord({ id: b, preview: `DeepReference contradiction side ${b.slice(0, 8)}` }, b);
			const id = pairKey(a, b);
			return {
				id,
				stronger,
				weaker,
				topic_overlap: 0.64,
				topic: text(event.data.query ?? event.data.intent ?? 'deep_reference contradiction'),
				trust_delta: 0,
				date_diff_days: 0,
				resolved: false,
				provenance: source('event', `DeepReferenceCompleted:${text(event.data.timestamp, String(pairIndexBase + i))}:${id}`),
				receipt: {
					label: `DeepReference contradiction pair ${i + 1}`,
					evidence: {
						query: event.data.query,
						intent: event.data.intent,
						status: event.data.status,
						confidence: event.data.confidence,
						memories_analyzed: event.data.memories_analyzed,
						contradiction_pair: [a, b]
					}
				}
			};
		})
		.filter((p): p is ImmuneSynapsePair => Boolean(p));
}

export function normalizeContradictionsScene(input: {
	contradictions?: unknown[];
	total?: number;
	memoriesAnalyzed?: number;
	deepReferenceEvents?: VestigeEvent[];
}): ContradictionsScene {
	const apiPairs = input.contradictions ?? [];
	const events = (input.deepReferenceEvents ?? []).filter((e) => e.type === 'DeepReferenceCompleted');
	const pairsById = new Map<string, ImmuneSynapsePair>();

	apiPairs.forEach((p, i) => {
		const pair = normalizeApiPair(p, i);
		if (pair) pairsById.set(pair.id, pair);
	});
	for (const event of events) {
		for (const pair of normalizeDeepReferenceEvent(event, pairsById.size)) {
			if (!pairsById.has(pair.id)) pairsById.set(pair.id, pair);
		}
	}

	const pairs = Array.from(pairsById.values());
	const nodes: RouteNode[] = [];
	const nodeIndex = new Map<string, number>();
	const addNode = (side: ContradictionMemorySide, roleTag: string) => {
		if (nodeIndex.has(side.id)) return nodeIndex.get(side.id)!;
		const index = nodes.length;
		nodeIndex.set(side.id, index);
		nodes.push({
			source: side.provenance,
			index,
			label: side.preview || side.id.slice(0, 8),
			retention: side.trust,
			trust: side.trust,
			lastAccessed: side.date || undefined,
			tags: [roleTag, ...(side.tags ?? [])],
			type: side.type ?? 'memory'
		});
		return index;
	};

	const edges: RouteEdge[] = [];
	const receipts: RouteReceipt[] = [];
	pairs.forEach((pair) => {
		const a = addNode(pair.stronger, 'stronger');
		const b = addNode(pair.weaker, 'weaker');
		edges.push({
			source: pair.provenance,
			sourceIndex: a,
			targetIndex: b,
			weight: Math.max(0.1, pair.topic_overlap),
			kind: 'contradiction'
		});
		receipts.push({
			source: pair.provenance,
			label: pair.receipt.label,
			nodeIndices: [a, b]
		});
	});

	const unresolved = pairs.filter((p) => !p.resolved).length;
	const avgTrustDelta = pairs.length ? pairs.reduce((sum, p) => sum + p.trust_delta, 0) / pairs.length : 0;
	const eventsOut: RouteEvent[] = unresolved
		? [
				{
					source: source('scalar', 'contradictions.unresolved', { name: 'unresolved', value: unresolved }),
					type: 'ContradictionUnresolved',
					targetIndex: -1,
					frame: 80,
					energy: unresolved
				}
			]
		: [];

	const scene: ContradictionsScene = {
		organ: 'contradictions',
		nodes,
		edges,
		events: eventsOut,
		receipts,
		scalars: {
			total: num(input.total, pairs.length),
			memoriesAnalyzed: num(input.memoriesAnalyzed, nodes.length),
			pairCount: pairs.length,
			unresolved,
			avgTrustDelta
		},
		alive: pairs.length > 0,
		pairs,
		raw: { apiPairs, deepReferenceEvents: events }
	};

	if (import.meta.env.DEV) assertProvenance(scene);
	// Scalar provenance is not part of RouteSceneModel arrays, so force-create a
	// real scalar source here as a discipline marker for the empty/calm state.
	void scalarSource('pairCount', pairs.length);
	return scene;
}

export function pairToLegacyContradiction(pair: ImmuneSynapsePair): ContradictionPair {
	return {
		memory_a_id: pair.stronger.id,
		memory_b_id: pair.weaker.id,
		memory_a_preview: pair.stronger.preview,
		memory_b_preview: pair.weaker.preview,
		memory_a_type: pair.stronger.type,
		memory_b_type: pair.weaker.type,
		memory_a_created: pair.stronger.date,
		memory_b_created: pair.weaker.date,
		memory_a_tags: pair.stronger.tags,
		memory_b_tags: pair.weaker.tags,
		trust_a: pair.stronger.trust,
		trust_b: pair.weaker.trust,
		similarity: pair.topic_overlap,
		date_diff_days: pair.date_diff_days,
		topic: pair.topic
	};
}

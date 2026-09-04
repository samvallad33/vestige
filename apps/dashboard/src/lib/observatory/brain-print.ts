/**
 * Brain print — share mechanic #2.
 *
 * A deterministic particle-signature of a store's SHAPE, never its content.
 * Counts, type mix, retention histogram, edge density. No memory text, no
 * ids, no labels, no endangered snippets. The print id is the Observatory
 * seed (`?seed=vb1-<8 hex>`), so the field layout is the fingerprint and
 * the permalink is safe by construction.
 *
 * Same store twice → identical print. Two stores → different prints.
 *
 * Determinism rule: only STORED shape enters the print. `dueForReview` is a
 * wall-clock property (`next_review <= now`), so it is carried on the shape
 * for display but never hashed or scored — otherwise an untouched store would
 * re-key every time a card crossed its due date.
 */

import { api } from '$stores/api';
import type { RetentionDistribution, SystemStats } from '$types';

export const BRAIN_PRINT_VERSION = 1;
export const BRAIN_PRINT_PREFIX = 'vb1-';

/** 10% FSRS histogram lanes — matches /api/retention-distribution. */
export const RETENTION_RANGES = [
	'0-10%',
	'10-20%',
	'20-30%',
	'30-40%',
	'40-50%',
	'50-60%',
	'60-70%',
	'70-80%',
	'80-90%',
	'90-100%'
] as const;

/** Canonical Vestige node-type lanes (stable order, zeros kept). */
export const TYPE_LANES = [
	'concept',
	'decision',
	'event',
	'fact',
	'note',
	'pattern',
	'person',
	'place'
] as const;

export interface BrainTopology {
	nodeCount: number;
	edgeCount: number;
}

export interface BrainShape {
	totalMemories: number;
	dueForReview: number;
	averageRetention: number;
	embeddingCoverage: number;
	/** Count only — never the endangered Memory[] payload. */
	endangeredCount: number;
	byType: Record<string, number>;
	retentionBuckets: { range: string; count: number }[];
	nodeCount: number;
	edgeCount: number;
}

export interface BrainTrait {
	id: string;
	label: string;
}

export interface BrainPrint {
	printId: string;
	seed: string;
	traits: BrainTrait[];
	/** Quantized numeric lanes hashed into printId. Structure only. */
	vector: number[];
}

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/** FNV-1a 32-bit over UTF-8 bytes. Empty string → 0x811c9dc5. */
export function fnv1a32(input: string | Uint8Array): number {
	const bytes = typeof input === 'string' ? new TextEncoder().encode(input) : input;
	let hash = FNV_OFFSET;
	for (let i = 0; i < bytes.length; i++) {
		hash ^= bytes[i]!;
		hash = Math.imul(hash, FNV_PRIME);
	}
	return hash >>> 0;
}

export function isBrainPrintSeed(seed: string): boolean {
	return /^vb1-[0-9a-f]{8}$/.test(seed);
}

export function formatPrintId(hash: number): string {
	return `${BRAIN_PRINT_PREFIX}${(hash >>> 0).toString(16).padStart(8, '0')}`;
}

export function loopExportFilename(seed: string, demo: string): string {
	return isBrainPrintSeed(seed) ? `vestige-${seed}-loop.mp4` : `vestige-${demo}-loop.mp4`;
}

function intLane(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(0, Math.round(value));
}

function milliLane(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(0, Math.min(1000, Math.round(value * 1000)));
}

/** /api/stats embeddingCoverage is 0–100 percent, not a 0–1 fraction. */
function percentMilli(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(0, Math.min(1000, Math.round(value * 10)));
}

function slugType(raw: string): string {
	return raw.toLowerCase().replace(/[^a-z0-9_-]/g, '');
}

function bucketMap(buckets: { range: string; count: number }[]): Map<string, number> {
	const map = new Map<string, number>();
	for (const range of RETENTION_RANGES) map.set(range, 0);
	for (const bucket of buckets) {
		const range = bucket.range.trim();
		if (!range) continue;
		map.set(range, intLane(bucket.count));
	}
	return map;
}

function typeMap(byType: Record<string, number>): Map<string, number> {
	const map = new Map<string, number>();
	for (const lane of TYPE_LANES) map.set(lane, 0);
	for (const [raw, count] of Object.entries(byType)) {
		const key = slugType(raw);
		if (!key) continue;
		map.set(key, (map.get(key) ?? 0) + intLane(count));
	}
	return map;
}

/**
 * Quantized numeric fingerprint lanes. Extra (non-canonical) types are NOT in
 * this vector — they ride in the canonical payload string so they still
 * change the print without leaking memory text.
 */
export function encodeShapeVector(shape: BrainShape): number[] {
	const nodes = intLane(shape.nodeCount);
	const edges = intLane(shape.edgeCount);
	const densityMilli = nodes <= 0 ? 0 : Math.round((1000 * edges) / nodes);
	const types = typeMap(shape.byType);
	const buckets = bucketMap(shape.retentionBuckets);
	return [
		BRAIN_PRINT_VERSION,
		intLane(shape.totalMemories),
		// Lane 2 is reserved (always 0). It carried dueForReview until the Sep 2026
		// determinism fix; that leaked wall-clock time into the hash (header contract).
		0,
		milliLane(shape.averageRetention),
		percentMilli(shape.embeddingCoverage),
		intLane(shape.endangeredCount),
		nodes,
		edges,
		densityMilli,
		...TYPE_LANES.map((lane) => types.get(lane) ?? 0),
		...RETENTION_RANGES.map((range) => buckets.get(range) ?? 0)
	];
}

/**
 * Canonical ASCII payload. Stable across object-key order. Contains ZERO
 * memory text — type keys are schema slugs, ranges are bucket labels.
 */
export function canonicalShapePayload(shape: BrainShape): string {
	const vector = encodeShapeVector(shape);
	const types = typeMap(shape.byType);
	const extras = [...types.entries()]
		.filter(([key]) => !(TYPE_LANES as readonly string[]).includes(key))
		.filter(([, count]) => count > 0)
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([key, count]) => `${key}=${count}`)
		.join(',');
	const buckets = bucketMap(shape.retentionBuckets);
	const ret = RETENTION_RANGES.map((range) => `${range}=${buckets.get(range) ?? 0}`).join(',');
	const extraLane = extras.length ? `|extra:${extras}` : '';
	return `v${vector[0]}|t:${vector[1]}|d:${vector[2]}|r:${vector[3]}|c:${vector[4]}|z:${vector[5]}|n:${vector[6]}|g:${vector[7]}|x:${vector[8]}|types:${TYPE_LANES.map((lane) => `${lane}=${types.get(lane) ?? 0}`).join(',')}|ret:${ret}${extraLane}`;
}

interface ShapeMetrics {
	total: number;
	avgRet: number;
	coverage: number;
	endangeredRatio: number;
	edgeDensity: number;
	highRetFrac: number;
	lowRetFrac: number;
	typeEntropy: number;
	dominantType: string | null;
	dominantFrac: number;
	typeCount: number;
}

function fractionForRanges(shape: BrainShape, pred: (lo: number, hi: number) => boolean): number {
	const total = Math.max(1, intLane(shape.totalMemories) || intLane(shape.retentionBuckets.reduce((s, b) => s + intLane(b.count), 0)));
	let n = 0;
	for (const bucket of bucketMap(shape.retentionBuckets)) {
		const match = /^(\d+)\s*-\s*(\d+)%$/.exec(bucket[0]);
		if (!match) continue;
		const lo = Number(match[1]);
		const hi = Number(match[2]);
		if (pred(lo, hi)) n += bucket[1];
	}
	return n / total;
}

function metricsOf(shape: BrainShape): ShapeMetrics {
	const total = Math.max(0, intLane(shape.totalMemories));
	const denom = Math.max(1, total);
	const nodes = Math.max(0, intLane(shape.nodeCount));
	const edges = Math.max(0, intLane(shape.edgeCount));
	const types = typeMap(shape.byType);
	const present = [...types.entries()].filter(([, n]) => n > 0);
	const typeTotal = present.reduce((s, [, n]) => s + n, 0) || 1;
	let entropy = 0;
	for (const [, n] of present) {
		const p = n / typeTotal;
		entropy -= p * Math.log2(p);
	}
	let dominantType: string | null = null;
	let dominantFrac = 0;
	for (const [key, n] of present) {
		const frac = n / typeTotal;
		if (frac > dominantFrac || (frac === dominantFrac && key.localeCompare(dominantType ?? '') < 0)) {
			dominantType = key;
			dominantFrac = frac;
		}
	}
	return {
		total,
		avgRet: milliLane(shape.averageRetention) / 1000,
		coverage: percentMilli(shape.embeddingCoverage) / 1000,
		endangeredRatio: intLane(shape.endangeredCount) / denom,
		edgeDensity: nodes <= 0 ? 0 : edges / nodes,
		highRetFrac: fractionForRanges(shape, (lo) => lo >= 70),
		lowRetFrac: fractionForRanges(shape, (_lo, hi) => hi <= 30),
		typeEntropy: entropy,
		dominantType,
		dominantFrac,
		typeCount: present.length
	};
}

interface TraitRule {
	id: string;
	label: string;
	group: string;
	score: (m: ShapeMetrics) => number;
}

const TRAIT_RULES: TraitRule[] = [
	{
		id: 'dense-associative',
		label: 'dense associative field',
		group: 'density',
		score: (m) => (m.edgeDensity >= 1.2 ? Math.min(3, m.edgeDensity) : 0)
	},
	{
		id: 'sparse-lattice',
		label: 'sparse lattice',
		group: 'density',
		score: (m) => (m.edgeDensity > 0 && m.edgeDensity < 0.45 ? 1.4 - m.edgeDensity : 0)
	},
	{
		id: 'deep-archive',
		label: 'deep archive',
		group: 'vitality',
		score: (m) => (m.highRetFrac >= 0.4 || m.avgRet >= 0.72 ? 1.2 + m.highRetFrac : 0)
	},
	{
		id: 'sedimentary',
		label: 'sedimentary dark',
		group: 'vitality',
		score: (m) => (m.lowRetFrac >= 0.3 || m.endangeredRatio >= 0.22 ? 1.1 + m.lowRetFrac : 0)
	},
	{
		id: 'oxygen-rich',
		label: 'oxygen-rich field',
		group: 'vitality',
		score: (m) => (m.avgRet >= 0.78 && m.highRetFrac >= 0.35 ? m.avgRet : 0)
	},
	{
		id: 'typed-mosaic',
		label: 'typed mosaic',
		group: 'types',
		score: (m) => (m.typeCount >= 5 && m.typeEntropy >= 1.8 ? m.typeEntropy : 0)
	},
	{
		id: 'concept-dominant',
		label: 'concept-weighted',
		group: 'types',
		score: (m) => (m.dominantType === 'concept' && m.dominantFrac >= 0.4 ? m.dominantFrac : 0)
	},
	{
		id: 'event-forward',
		label: 'event-forward',
		group: 'types',
		score: (m) => (m.dominantType === 'event' && m.dominantFrac >= 0.4 ? m.dominantFrac : 0)
	},
	{
		id: 'decision-heavy',
		label: 'decision-heavy',
		group: 'types',
		score: (m) => (m.dominantType === 'decision' && m.dominantFrac >= 0.35 ? m.dominantFrac : 0)
	},
	{
		id: 'wide-field',
		label: 'wide-field archive',
		group: 'scale',
		score: (m) => (m.total >= 400 ? Math.log10(m.total) : 0)
	},
	{
		id: 'expanding-cortex',
		label: 'expanding cortex',
		group: 'scale',
		score: (m) => (m.total >= 80 && m.total < 400 ? 0.6 : 0)
	},
	{
		id: 'compact-nucleus',
		label: 'compact nucleus',
		group: 'scale',
		score: (m) => (m.total > 0 && m.total < 80 ? 0.55 : 0)
	},
	{
		id: 'covered-embeddings',
		label: 'fully embedded',
		group: 'coverage',
		score: (m) => (m.coverage >= 0.95 ? m.coverage : 0)
	}
];

const SCALE_FALLBACKS: BrainTrait[] = [
	{ id: 'wide-field', label: 'wide-field archive' },
	{ id: 'expanding-cortex', label: 'expanding cortex' },
	{ id: 'compact-nucleus', label: 'compact nucleus' }
];

export function deriveTraits(shape: BrainShape): BrainTrait[] {
	const m = metricsOf(shape);
	const ranked = TRAIT_RULES.map((rule) => ({ rule, score: rule.score(m) }))
		.filter((row) => row.score > 0)
		.sort((a, b) => b.score - a.score || a.rule.id.localeCompare(b.rule.id));

	const picked: BrainTrait[] = [];
	const usedGroups = new Set<string>();
	const usedIds = new Set<string>();
	for (const { rule } of ranked) {
		if (picked.length >= 3) break;
		if (usedGroups.has(rule.group)) continue;
		picked.push({ id: rule.id, label: rule.label });
		usedGroups.add(rule.group);
		usedIds.add(rule.id);
	}

	if (picked.length < 2) {
		const scale =
			m.total >= 400
				? SCALE_FALLBACKS[0]
				: m.total >= 80
					? SCALE_FALLBACKS[1]
					: SCALE_FALLBACKS[2];
		if (scale && !usedIds.has(scale.id)) {
			picked.push(scale);
			usedIds.add(scale.id);
		}
	}
	if (picked.length < 2) {
		picked.push({ id: 'structured-field', label: 'structured field' });
	}
	return picked.slice(0, 3);
}

export function computeBrainPrint(shape: BrainShape): BrainPrint {
	const payload = canonicalShapePayload(shape);
	const printId = formatPrintId(fnv1a32(payload));
	return {
		printId,
		seed: printId,
		traits: deriveTraits(shape),
		vector: encodeShapeVector(shape)
	};
}

/**
 * Lift API payloads into a shape. Reads endangered LENGTH only — never
 * content, ids, or labels from that array.
 */
export function shapeFromStore(input: {
	stats: Pick<SystemStats, 'totalMemories' | 'dueForReview' | 'averageRetention' | 'embeddingCoverage'>;
	retention: Pick<RetentionDistribution, 'distribution' | 'byType' | 'total'> & {
		endangered?: { length: number } | null;
	};
	topology?: BrainTopology;
}): BrainShape {
	const endangered = input.retention.endangered;
	return {
		totalMemories: intLane(input.stats.totalMemories),
		dueForReview: intLane(input.stats.dueForReview),
		averageRetention: input.stats.averageRetention,
		embeddingCoverage: input.stats.embeddingCoverage,
		endangeredCount: endangered ? intLane(endangered.length) : 0,
		byType: { ...input.retention.byType },
		retentionBuckets: input.retention.distribution.map((row) => ({
			range: row.range,
			count: intLane(row.count)
		})),
		nodeCount: intLane(input.topology?.nodeCount ?? 0),
		edgeCount: intLane(input.topology?.edgeCount ?? 0)
	};
}

/**
 * Fetch live shape and return the print. Pass hub graph counts when they
 * already represent the store's connected field; omit them (receipt-scoped
 * views, cold start) and this takes one cheap `api.graph` itself.
 */
export async function captureBrainPrint(topology?: BrainTopology): Promise<BrainPrint> {
	const [stats, retention, graph] = await Promise.all([
		api.stats(),
		api.retentionDistribution(),
		topology
			? Promise.resolve(null)
			: api.graph({ max_nodes: 200, depth: 3, sort: 'connected' }).catch(() => null)
	]);
	const topo =
		topology ??
		(graph ? { nodeCount: graph.nodeCount, edgeCount: graph.edgeCount } : undefined);
	return computeBrainPrint(shapeFromStore({ stats, retention, topology: topo }));
}

/** Absolute share URL: current location, demo + print seed, no capture/receipt. */
export function printPermalink(currentHref: string, demo: string, printId: string): string {
	const url = new URL(currentHref);
	url.searchParams.set('demo', demo);
	url.searchParams.set('seed', printId);
	url.searchParams.delete('frame');
	url.searchParams.delete('capture');
	url.searchParams.delete('receipt');
	url.searchParams.delete('brain');
	return url.toString();
}

/**
 * Compact structure-only `?brain=` payload. JSON→base64url of the quantized
 * vector + extra type lanes. A receiver rebuilds the SAME print + synthetic
 * field with ZERO backend round-trips and ZERO memory text.
 */
export function encodeBrainParam(shape: BrainShape): string {
	const vector = encodeShapeVector(shape);
	const types = typeMap(shape.byType);
	const extras: Record<string, number> = {};
	for (const [key, count] of types) {
		if (!(TYPE_LANES as readonly string[]).includes(key) && count > 0) extras[key] = count;
	}
	const json = JSON.stringify({ v: vector, e: extras });
	return `v1.${btoa(json).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')}`;
}

export function decodeBrainParam(raw: string): BrainShape | null {
	const m = /^v1\.([A-Za-z0-9_-]+)$/.exec(raw.trim());
	if (!m) return null;
	try {
		const b64 = m[1]!.replace(/-/g, '+').replace(/_/g, '/');
		const pad = b64.length % 4 === 0 ? '' : '='.repeat(4 - (b64.length % 4));
		const parsed = JSON.parse(atob(b64 + pad)) as { v?: number[]; e?: Record<string, number> };
		if (!Array.isArray(parsed.v) || parsed.v.length < 19) return null;
		const shape = shapeFromVector(parsed.v.map((n) => intLane(Number(n))));
		if (parsed.e && typeof parsed.e === 'object') {
			for (const [key, count] of Object.entries(parsed.e)) {
				const slug = slugType(key);
				if (!slug) continue;
				shape.byType[slug] = intLane(count);
			}
		}
		return shape;
	} catch {
		return null;
	}
}

/** Rebuild a BrainShape from the quantized vector lanes (no extras). */
export function shapeFromVector(vector: number[]): BrainShape {
	const v = vector.length >= 19 ? vector : [...vector, ...Array(19 - vector.length).fill(0)];
	const byType: Record<string, number> = {};
	TYPE_LANES.forEach((lane, i) => {
		const n = intLane(v[9 + i] ?? 0);
		if (n > 0) byType[lane] = n;
	});
	const retentionBuckets = RETENTION_RANGES.map((range, i) => ({
		range,
		count: intLane(v[17 + i] ?? 0)
	}));
	return {
		totalMemories: intLane(v[1] ?? 0),
		dueForReview: intLane(v[2] ?? 0),
		averageRetention: intLane(v[3] ?? 0) / 1000,
		embeddingCoverage: intLane(v[4] ?? 0) / 10,
		endangeredCount: intLane(v[5] ?? 0),
		byType,
		retentionBuckets,
		nodeCount: intLane(v[6] ?? 0),
		edgeCount: intLane(v[7] ?? 0)
	};
}

/**
 * Content-free synthetic graph from a shape — structure only. Labels are
 * type·index tokens; ids are deterministic syn-<printId>-N. Same shape +
 * print ⇒ identical topology for offline `?brain=` receivers.
 */
export function synthesizeStructureGraph(shape: BrainShape, printId: string): import('$types').GraphResponse {
	const targetNodes = Math.min(200, Math.max(0, intLane(shape.nodeCount) || intLane(shape.totalMemories)));
	const types = typeMap(shape.byType);
	const present = [...types.entries()].filter(([, n]) => n > 0);
	const typeTotal = present.reduce((s, [, n]) => s + n, 0) || 1;
	const buckets = [...bucketMap(shape.retentionBuckets).entries()];
	const bucketTotal = buckets.reduce((s, [, n]) => s + n, 0) || 1;

	const nodes: import('$types').GraphNode[] = [];
	for (let i = 0; i < targetNodes; i++) {
		// Pick type by proportional walk
		let tRoll = ((i * 2654435761) >>> 0) % typeTotal;
		let type = present[0]?.[0] ?? 'note';
		for (const [key, n] of present) {
			tRoll -= n;
			if (tRoll < 0) {
				type = key;
				break;
			}
		}
		let bRoll = ((i * 1597334677) >>> 0) % bucketTotal;
		let retention = 0.55;
		for (const [range, n] of buckets) {
			bRoll -= n;
			if (bRoll < 0) {
				const m = /^(\d+)\s*-\s*(\d+)/.exec(range);
				const lo = m ? Number(m[1]) : 50;
				const hi = m ? Number(m[2]) : 60;
				retention = (lo + hi) / 200;
				break;
			}
		}
		const id = `syn-${printId}-${i.toString(16).padStart(3, '0')}`;
		nodes.push({
			id,
			label: `${type}·${(i + 1).toString().padStart(3, '0')}`,
			type,
			retention,
			tags: [],
			createdAt: '1970-01-01T00:00:00.000Z',
			updatedAt: '1970-01-01T00:00:00.000Z',
			isCenter: i === 0
		});
	}

	const density = targetNodes <= 0 ? 0 : intLane(shape.edgeCount) / Math.max(1, intLane(shape.nodeCount) || targetNodes);
	const targetEdges = Math.min(
		targetNodes * 12,
		Math.round(targetNodes * Math.max(0.2, Math.min(8, density)))
	);
	const edges: import('$types').GraphEdge[] = [];
	const seen = new Set<string>();
	for (let e = 0; e < targetEdges && nodes.length > 1; e++) {
		const a = ((e * 1103515245 + 12345) >>> 0) % nodes.length;
		let b = ((e * 214013 + 2531011) >>> 0) % nodes.length;
		if (b === a) b = (b + 1) % nodes.length;
		const source = nodes[a]!.id;
		const target = nodes[b]!.id;
		const key = `${source}:${target}`;
		if (seen.has(key)) continue;
		seen.add(key);
		edges.push({ source, target, weight: 0.35 + ((e % 65) / 100), type: 'assoc' });
	}

	return {
		nodes,
		edges,
		center_id: nodes[0]?.id ?? '',
		depth: 3,
		nodeCount: nodes.length,
		edgeCount: edges.length
	};
}

/** Permalink that carries the full structure vector (Task C). */
export function brainPermalink(currentHref: string, demo: string, shape: BrainShape): string {
	const print = computeBrainPrint(shape);
	const url = new URL(currentHref);
	url.searchParams.set('demo', demo);
	url.searchParams.set('seed', print.printId);
	url.searchParams.set('brain', encodeBrainParam(shape));
	url.searchParams.delete('frame');
	url.searchParams.delete('capture');
	url.searchParams.delete('receipt');
	return url.toString();
}

export async function captureBrainBundle(topology?: BrainTopology): Promise<{
	shape: BrainShape;
	print: BrainPrint;
	archiveDays: number;
}> {
	const [stats, retention, graph] = await Promise.all([
		api.stats(),
		api.retentionDistribution(),
		topology
			? Promise.resolve(null)
			: api.graph({ max_nodes: 200, depth: 3, sort: 'connected' }).catch(() => null)
	]);
	const topo =
		topology ??
		(graph ? { nodeCount: graph.nodeCount, edgeCount: graph.edgeCount } : undefined);
	const shape = shapeFromStore({ stats, retention, topology: topo });
	const print = computeBrainPrint(shape);
	const oldest = stats.oldestMemory;
	const newest = stats.newestMemory;
	let archiveDays = 0;
	if (oldest && newest) {
		const a = Date.parse(oldest);
		const b = Date.parse(newest);
		if (Number.isFinite(a) && Number.isFinite(b)) {
			archiveDays = Math.max(0, Math.round(Math.abs(b - a) / 86_400_000));
		}
	}
	return { shape, print, archiveDays };
}

import { describe, it, expect } from 'vitest';
import {
	canonicalShapePayload,
	computeBrainPrint,
	deriveTraits,
	encodeShapeVector,
	fnv1a32,
	formatPrintId,
	isBrainPrintSeed,
	loopExportFilename,
	printPermalink,
	shapeFromStore,
	RETENTION_RANGES,
	TYPE_LANES,
	type BrainShape
} from '../brain-print';
import type { RetentionDistribution, SystemStats } from '$types';

function buckets(weights: number[]): BrainShape['retentionBuckets'] {
	const total = weights.reduce((s, n) => s + n, 0) || 1;
	return RETENTION_RANGES.map((range, i) => ({
		range,
		count: Math.round(((weights[i] ?? 0) / total) * 250)
	}));
}

function shape(partial: Partial<BrainShape> = {}): BrainShape {
	return {
		totalMemories: 250,
		dueForReview: 18,
		averageRetention: 0.64,
		embeddingCoverage: 100,
		endangeredCount: 12,
		byType: { fact: 80, concept: 60, event: 40, note: 30, decision: 20, pattern: 12, person: 5, place: 3 },
		retentionBuckets: buckets([8, 10, 12, 18, 22, 30, 40, 45, 35, 30]),
		nodeCount: 200,
		edgeCount: 310,
		...partial
	};
}

describe('fnv1a32', () => {
	it('matches published FNV-1a 32-bit vectors', () => {
		expect(fnv1a32('').toString(16)).toBe('811c9dc5');
		expect(fnv1a32('a').toString(16)).toBe('e40c292c');
		expect(fnv1a32('foobar').toString(16)).toBe('bf9cf968');
	});
});

describe('computeBrainPrint', () => {
	it('same shape twice → identical print id, seed, vector, traits', () => {
		const a = computeBrainPrint(shape());
		const b = computeBrainPrint(shape());
		expect(a).toEqual(b);
		expect(a.printId).toMatch(/^vb1-[0-9a-f]{8}$/);
		expect(a.seed).toBe(a.printId);
	});

	it('printId is vb1- + 8 hex of FNV-1a over the canonical payload', () => {
		const s = shape();
		const print = computeBrainPrint(s);
		expect(print.printId).toBe(formatPrintId(fnv1a32(canonicalShapePayload(s))));
	});

	it('two stores with different counts produce different prints', () => {
		const a = computeBrainPrint(shape({ totalMemories: 250 }));
		const b = computeBrainPrint(shape({ totalMemories: 251 }));
		expect(a.printId).not.toBe(b.printId);
	});

	it('type-mix changes the print even when totals match', () => {
		const a = computeBrainPrint(shape({ byType: { fact: 100, event: 50 } }));
		const b = computeBrainPrint(shape({ byType: { fact: 50, event: 100 } }));
		expect(a.printId).not.toBe(b.printId);
	});

	it('byType key order does not change the print', () => {
		const a = computeBrainPrint(shape({ byType: { fact: 10, concept: 20, event: 30 } }));
		const b = computeBrainPrint(shape({ byType: { event: 30, concept: 20, fact: 10 } }));
		expect(a.printId).toBe(b.printId);
	});

	it('float jitter below a milli does not change the print', () => {
		const a = computeBrainPrint(shape({ averageRetention: 0.6414 }));
		const b = computeBrainPrint(shape({ averageRetention: 0.64149 }));
		expect(a.printId).toBe(b.printId);
	});

	it('treats embeddingCoverage as 0–100 percent (live /api/stats contract)', () => {
		const full = computeBrainPrint(shape({ embeddingCoverage: 100 }));
		const onePercent = computeBrainPrint(shape({ embeddingCoverage: 1 }));
		expect(full.printId).not.toBe(onePercent.printId);
		expect(full.vector[4]).toBe(1000);
		expect(onePercent.vector[4]).toBe(10);
	});

	it('edge density changes the print', () => {
		const a = computeBrainPrint(shape({ nodeCount: 200, edgeCount: 80 }));
		const b = computeBrainPrint(shape({ nodeCount: 200, edgeCount: 400 }));
		expect(a.printId).not.toBe(b.printId);
	});

	it('canonical payload and vector contain zero memory text', () => {
		const secret = 'the launch token is hunter2 and Sam lives on Maple Street';
		const print = computeBrainPrint(
			shape({
				byType: { fact: 4, note: 2 }
			})
		);
		const payload = canonicalShapePayload(
			shape({
				byType: { fact: 4, note: 2 }
			})
		);
		expect(payload).not.toContain(secret);
		expect(print.printId).not.toContain(secret);
		expect(print.vector.every((n) => Number.isInteger(n))).toBe(true);
		expect(print.vector[0]).toBe(1);
		expect(print.vector.length).toBe(1 + 8 + TYPE_LANES.length + RETENTION_RANGES.length);
	});
});

describe('shapeFromStore privacy', () => {
	it('reads endangered LENGTH only — never content, ids, or labels', () => {
		const stats: Pick<SystemStats, 'totalMemories' | 'dueForReview' | 'averageRetention' | 'embeddingCoverage'> = {
			totalMemories: 12,
			dueForReview: 2,
			averageRetention: 0.5,
			embeddingCoverage: 100
		};
		const retention: Pick<RetentionDistribution, 'distribution' | 'byType' | 'total'> & {
			endangered: Array<{ id: string; content: string; label?: string }>;
		} = {
			total: 12,
			byType: { fact: 12 },
			distribution: RETENTION_RANGES.map((range) => ({ range, count: range === '20-30%' ? 12 : 0 })),
			endangered: [
				{
					id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
					content: 'PRIVATE: API key sk-live-do-not-hash-this',
					label: 'secret-memory-title'
				}
			]
		};
		const shaped = shapeFromStore({ stats, retention, topology: { nodeCount: 12, edgeCount: 4 } });
		const print = computeBrainPrint(shaped);
		const payload = canonicalShapePayload(shaped);
		expect(shaped.endangeredCount).toBe(1);
		expect(payload).not.toContain('PRIVATE');
		expect(payload).not.toContain('sk-live');
		expect(payload).not.toContain('aaaaaaaa-bbbb');
		expect(payload).not.toContain('secret-memory-title');
		expect(print.printId).not.toContain('sk-live');
		expect(JSON.stringify(print)).not.toContain('PRIVATE');
	});
});

describe('deriveTraits', () => {
	it('names a dense associative field from high edge density', () => {
		const traits = deriveTraits(shape({ nodeCount: 100, edgeCount: 180 }));
		expect(traits.map((t) => t.id)).toContain('dense-associative');
		expect(traits.map((t) => t.label)).toContain('dense associative field');
		expect(traits.length).toBeGreaterThanOrEqual(2);
		expect(traits.length).toBeLessThanOrEqual(3);
	});

	it('names a deep archive from high-retention mass', () => {
		const traits = deriveTraits(
			shape({
				averageRetention: 0.84,
				retentionBuckets: buckets([1, 1, 1, 2, 3, 5, 8, 20, 30, 40]),
				nodeCount: 80,
				edgeCount: 20
			})
		);
		expect(traits.map((t) => t.id)).toContain('deep-archive');
	});

	it('never emits both dense and sparse', () => {
		const traits = deriveTraits(shape({ nodeCount: 100, edgeCount: 200 }));
		const ids = new Set(traits.map((t) => t.id));
		expect(ids.has('dense-associative') && ids.has('sparse-lattice')).toBe(false);
	});

	it('always returns 2–3 structure labels, never memory text', () => {
		const traits = deriveTraits(shape({ totalMemories: 3, byType: { note: 3 }, nodeCount: 3, edgeCount: 0 }));
		expect(traits.length).toBeGreaterThanOrEqual(2);
		expect(traits.length).toBeLessThanOrEqual(3);
		for (const trait of traits) {
			expect(trait.label).toMatch(/^[a-z][a-z- ]+$/);
			expect(trait.label).not.toMatch(/memory-|content|sk-/i);
		}
	});
});

describe('permalink + export name', () => {
	it('isBrainPrintSeed accepts only vb1- + 8 lowercase hex', () => {
		expect(isBrainPrintSeed('vb1-deadbeef')).toBe(true);
		expect(isBrainPrintSeed('vb1-DEADBEEF')).toBe(false);
		expect(isBrainPrintSeed('vestige-observatory-v1')).toBe(false);
		expect(isBrainPrintSeed('vb1-abcd')).toBe(false);
	});

	it('names the clip vestige-<printId>-loop.mp4 when a print is the seed', () => {
		expect(loopExportFilename('vb1-c0ffee00', 'recall-path')).toBe('vestige-vb1-c0ffee00-loop.mp4');
		expect(loopExportFilename('vestige-observatory-v1', 'recall-path')).toBe('vestige-recall-path-loop.mp4');
	});

	it('permalink is demo + seed, stripping capture/receipt/frame', () => {
		const href = 'http://127.0.0.1:5230/dashboard/observatory?demo=firewall&frame=12&capture=1&receipt=r_x&seed=old';
		expect(printPermalink(href, 'engram-birth', 'vb1-abcd1234')).toBe(
			'http://127.0.0.1:5230/dashboard/observatory?demo=engram-birth&seed=vb1-abcd1234'
		);
	});
});

describe('encodeShapeVector', () => {
	it('keeps a stable lane count', () => {
		const vector = encodeShapeVector(shape());
		expect(vector.length).toBe(1 + 8 + TYPE_LANES.length + RETENTION_RANGES.length);
		expect(vector.every((n) => Number.isInteger(n) && n >= 0)).toBe(true);
	});
});

describe('brain print determinism: stored shape only, never wall-clock', () => {
	it('dueForReview does not change the print id, the vector, or the traits', () => {
		// Same store, captured before and after cards cross their due date. The
		// only difference is the live `next_review <= now` count, which is not
		// part of the store's shape. The print must not re-key.
		const morning = shape({ dueForReview: 0 });
		const evening = shape({ dueForReview: 41 });
		const a = computeBrainPrint(morning);
		const b = computeBrainPrint(evening);
		expect(a.printId).toBe(b.printId);
		expect(a.seed).toBe(b.seed);
		expect(a.vector).toEqual(b.vector);
		expect(a.traits).toEqual(b.traits);
		expect(canonicalShapePayload(morning)).toBe(canonicalShapePayload(evening));
		// Lane 2 is reserved and always zero.
		expect(a.vector[2]).toBe(0);
		expect(a.traits.map((t) => t.id)).not.toContain('review-pressure');
	});
});

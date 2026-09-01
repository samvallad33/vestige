import { describe, it, expect } from 'vitest';
import { archiveSpanDays } from '../export/wrapped-card';
import {
	computeBrainPrint,
	decodeBrainParam,
	encodeBrainParam,
	shapeFromVector,
	synthesizeStructureGraph,
	type BrainShape
} from '../brain-print';

function sample(): BrainShape {
	return {
		totalMemories: 250,
		dueForReview: 18,
		averageRetention: 0.64,
		embeddingCoverage: 100,
		endangeredCount: 12,
		byType: { fact: 80, concept: 60, correction: 7 },
		retentionBuckets: [
			{ range: '0-10%', count: 5 },
			{ range: '10-20%', count: 8 },
			{ range: '20-30%', count: 10 },
			{ range: '30-40%', count: 12 },
			{ range: '40-50%', count: 20 },
			{ range: '50-60%', count: 30 },
			{ range: '60-70%', count: 40 },
			{ range: '70-80%', count: 50 },
			{ range: '80-90%', count: 45 },
			{ range: '90-100%', count: 30 }
		],
		nodeCount: 200,
		edgeCount: 310
	};
}

describe('archiveSpanDays', () => {
	it('counts whole days between oldest and newest', () => {
		expect(archiveSpanDays('2026-01-01T00:00:00Z', '2026-01-31T00:00:00Z')).toBe(30);
	});
	it('returns 0 for missing dates', () => {
		expect(archiveSpanDays(null, '2026-01-01T00:00:00Z')).toBe(0);
	});
});

describe('?brain= round-trip', () => {
	it('encodes extras so print id is identical after decode', () => {
		const shape = sample();
		const a = computeBrainPrint(shape);
		const param = encodeBrainParam(shape);
		expect(param.startsWith('v1.')).toBe(true);
		const decoded = decodeBrainParam(param);
		expect(decoded).not.toBeNull();
		const b = computeBrainPrint(decoded!);
		expect(b.printId).toBe(a.printId);
		expect(decoded!.byType.correction).toBe(7);
	});

	it('rejects junk', () => {
		expect(decodeBrainParam('nope')).toBeNull();
		expect(decodeBrainParam('v1.%%%')).toBeNull();
	});

	it('synthetic graph is content-free and deterministic', () => {
		const shape = sample();
		const print = computeBrainPrint(shape);
		const g1 = synthesizeStructureGraph(shape, print.printId);
		const g2 = synthesizeStructureGraph(shape, print.printId);
		expect(g1.nodeCount).toBe(g2.nodeCount);
		expect(g1.nodes[0]?.id).toBe(g2.nodes[0]?.id);
		expect(g1.nodes.some((n) => /hunter2|secret|sk-live/i.test(n.label))).toBe(false);
		expect(g1.nodes[0]?.id.startsWith(`syn-${print.printId}-`)).toBe(true);
	});

	it('shapeFromVector reconstructs canonical lanes', () => {
		const shape = sample();
		const print = computeBrainPrint(shape);
		const rebuilt = shapeFromVector(print.vector);
		expect(rebuilt.totalMemories).toBe(250);
		expect(rebuilt.nodeCount).toBe(200);
		expect(rebuilt.byType.fact).toBe(80);
	});
});

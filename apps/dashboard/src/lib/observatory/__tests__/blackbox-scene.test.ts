import { describe, expect, it } from 'vitest';
import { normalizeBlackboxScene } from '../blackbox/blackbox-scene';

const run = {
	runId: 'run_alpha',
	summary: {
		firstTool: 'deep_reference',
		eventCount: 5,
		retrievedCount: 2,
		suppressedCount: 1,
		writeCount: 1,
		vetoCount: 1,
		startedAt: 1000,
		lastAt: 1600
	},
	events: [
		{ type: 'mcp.call', runId: 'run_alpha', tool: 'deep_reference', argsHash: 'abc123456789', at: 1000 },
		{ type: 'memory.retrieve', runId: 'run_alpha', ids: ['mem_a', 'mem_b'], activation: { mem_a: 0.9, mem_b: 0.35 }, at: 1120 },
		{ type: 'memory.suppress', runId: 'run_alpha', id: 'mem_b', reason: 'low_trust', at: 1260 },
		{ type: 'memory.write', runId: 'run_alpha', id: 'mem_c', diff: { decision: 'create' }, source: 'agent', at: 1410 },
		{ type: 'sanhedrin.veto', runId: 'run_alpha', claim: 'dangerous write', evidenceIds: ['mem_a'], confidence: 0.82, at: 1600 }
	]
};

const receipts = [
	{
		receipt_id: 'r_alpha',
		retrieved: ['mem_a', 'mem_b'],
		suppressed: [{ id: 'mem_b', reason: 'low_trust' }],
		activation_path: ['mem_a', 'mem_b'],
		trust_floor: 0.42,
		decay_risk: 'medium' as const,
		mutations: [{ id: 'mem_c', kind: 'write' }]
	}
];

describe('normalizeBlackboxScene', () => {
	it('turns a real trace run into lane events, memory nodes, receipts, and provenance', () => {
		const scene = normalizeBlackboxScene(run, receipts, 2);

		expect(scene.organ).toBe('blackbox');
		expect(scene.alive).toBe(true);
		expect(scene.runId).toBe('run_alpha');
		expect(scene.visibleEventCount).toBe(3);
		expect(scene.traceEvents).toHaveLength(5);
		expect(scene.traceEvents.map((e) => e.lane)).toEqual(['tool', 'retrieve', 'suppress', 'write', 'veto']);
		expect(scene.nodes.map((n) => n.source.id).sort()).toEqual(['mem_a', 'mem_b', 'mem_c']);
		expect(scene.receipts[0].source).toEqual({ kind: 'receipt', id: 'r_alpha' });
		expect(scene.events[1].source.id).toBe('run_alpha:event:1:memory.retrieve');
		expect(scene.traceEvents[1].activationPairs).toEqual([
			{ id: 'mem_a', activation: 0.9 },
			{ id: 'mem_b', activation: 0.35 }
		]);
		expect(scene.scalars.visibleEventCount).toBe(3);
	});

	it('returns an honest empty scene when no run is selected', () => {
		const scene = normalizeBlackboxScene(null, []);

		expect(scene.alive).toBe(false);
		expect(scene.nodes).toEqual([]);
		expect(scene.traceEvents).toEqual([]);
		expect(scene.scalars.eventCount).toBe(0);
	});
});

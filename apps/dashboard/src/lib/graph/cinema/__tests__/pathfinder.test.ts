import { describe, it, expect, beforeEach } from 'vitest';
import { planCinemaPath } from '../pathfinder';
import { makeNode, makeEdge, resetNodeCounter } from '../../__tests__/helpers';

describe('planCinemaPath', () => {
	beforeEach(() => resetNodeCounter());

	it('returns an empty path for no nodes', () => {
		const path = planCinemaPath([], [], 'missing');
		expect(path.beats).toEqual([]);
		expect(path.flowEdges).toEqual([]);
	});

	it('starts at the requested center when it exists', () => {
		const a = makeNode({ id: 'a' });
		const b = makeNode({ id: 'b' });
		const path = planCinemaPath([a, b], [makeEdge('a', 'b')], 'a');
		expect(path.beats[0].nodeId).toBe('a');
		expect(path.beats[0].kind).toBe('origin');
		expect(path.beats[0].viaEdge).toBeNull();
	});

	it('falls back to the most-connected node when center is missing', () => {
		const hub = makeNode({ id: 'hub' });
		const x = makeNode({ id: 'x' });
		const y = makeNode({ id: 'y' });
		const path = planCinemaPath(
			[x, hub, y],
			[makeEdge('hub', 'x'), makeEdge('hub', 'y')],
			'does-not-exist'
		);
		expect(path.beats[0].nodeId).toBe('hub');
	});

	it('visits the strongest-weighted connection first', () => {
		const a = makeNode({ id: 'a' });
		const weak = makeNode({ id: 'weak' });
		const strong = makeNode({ id: 'strong' });
		const path = planCinemaPath(
			[a, weak, strong],
			[makeEdge('a', 'weak', { weight: 0.1 }), makeEdge('a', 'strong', { weight: 0.9 })],
			'a'
		);
		expect(path.beats[1].nodeId).toBe('strong');
		expect(path.beats[1].kind).toBe('connection');
	});

	it('detours through a contradiction edge when reachable', () => {
		const a = makeNode({ id: 'a' });
		const normal = makeNode({ id: 'normal' });
		const conflict = makeNode({ id: 'conflict' });
		const path = planCinemaPath(
			[a, normal, conflict],
			[
				makeEdge('a', 'normal', { weight: 0.95, type: 'semantic' }),
				makeEdge('a', 'conflict', { weight: 0.2, type: 'contradiction' }),
			],
			'a'
		);
		const kinds = path.beats.map((b) => b.kind);
		expect(kinds).toContain('contradiction');
		// The contradiction beat carries max intensity.
		const c = path.beats.find((b) => b.kind === 'contradiction');
		expect(c?.intensity).toBe(1);
	});

	it('never exceeds maxBeats and never repeats a node', () => {
		const nodes = Array.from({ length: 20 }, (_, i) => makeNode({ id: `n${i}` }));
		const edges = nodes.slice(1).map((n) => makeEdge('n0', n.id, { weight: Math.random() }));
		const path = planCinemaPath(nodes, edges, 'n0', 5);
		expect(path.beats.length).toBeLessThanOrEqual(5);
		const ids = path.beats.map((b) => b.nodeId);
		expect(new Set(ids).size).toBe(ids.length);
	});

	it('is deterministic — same inputs yield the same path', () => {
		const nodes = [makeNode({ id: 'a' }), makeNode({ id: 'b' }), makeNode({ id: 'c' })];
		const edges = [makeEdge('a', 'b', { weight: 0.8 }), makeEdge('b', 'c', { weight: 0.6 })];
		const p1 = planCinemaPath(nodes, edges, 'a');
		const p2 = planCinemaPath(nodes, edges, 'a');
		expect(p1.beats.map((b) => b.nodeId)).toEqual(p2.beats.map((b) => b.nodeId));
	});

	it('records flowEdges for each traversed connection', () => {
		const a = makeNode({ id: 'a' });
		const b = makeNode({ id: 'b' });
		const path = planCinemaPath([a, b], [makeEdge('a', 'b', { weight: 0.7 })], 'a');
		expect(path.flowEdges.length).toBeGreaterThanOrEqual(1);
		expect(path.flowEdges[0].source === 'a' || path.flowEdges[0].target === 'a').toBe(true);
	});
});

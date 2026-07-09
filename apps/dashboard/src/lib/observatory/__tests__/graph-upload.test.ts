import { describe, it, expect } from 'vitest';
import { DemoClock } from '../demo-clock';
import {
	buildObservatoryGraph,
	buildNodeStateArray,
	buildEdgeIndexArray,
	hexToRgb01,
	nodeBaseColor
} from '../graph-upload';
import { FLOATS_PER_NODE, NODE_LANE, NODE_FLAG, toObservatoryNode } from '../types';
import type { GraphResponse, GraphNode } from '$types';

function node(partial: Partial<GraphNode> & { id: string }): GraphNode {
	return {
		label: partial.id,
		type: 'note',
		retention: 0.8,
		tags: [],
		createdAt: '2026-07-01T00:00:00Z',
		updatedAt: '2026-07-01T00:00:00Z',
		isCenter: false,
		...partial
	};
}

function response(nodes: GraphNode[], edges: GraphResponse['edges'] = []): GraphResponse {
	return {
		nodes,
		edges,
		center_id: nodes.find((n) => n.isCenter)?.id ?? '',
		depth: 3,
		nodeCount: nodes.length,
		edgeCount: edges.length
	};
}

describe('buildObservatoryGraph', () => {
	it('orders center first, then by id — independent of API order', () => {
		const a = response([node({ id: 'zz' }), node({ id: 'aa' }), node({ id: 'mm', isCenter: true })]);
		const b = response([node({ id: 'mm', isCenter: true }), node({ id: 'zz' }), node({ id: 'aa' })]);
		const ga = buildObservatoryGraph(a);
		const gb = buildObservatoryGraph(b);
		expect(ga.nodes.map((n) => n.id)).toEqual(['mm', 'aa', 'zz']);
		expect(gb.nodes.map((n) => n.id)).toEqual(['mm', 'aa', 'zz']);
		expect(ga.centerIndex).toBe(0);
	});

	it('maps edges to stable indices and drops dangling/self edges', () => {
		const g = buildObservatoryGraph(
			response(
				[node({ id: 'a' }), node({ id: 'b' })],
				[
					{ source: 'a', target: 'b', weight: 1, type: 'semantic' },
					{ source: 'a', target: 'ghost', weight: 1, type: 'semantic' },
					{ source: 'b', target: 'b', weight: 1, type: 'semantic' }
				]
			)
		);
		expect(g.edges).toHaveLength(1);
		expect(g.edges[0].sourceIndex).toBe(g.indexById.get('a'));
		expect(g.edges[0].targetIndex).toBe(g.indexById.get('b'));
	});
});

describe('buildNodeStateArray determinism', () => {
	const resp = response([
		node({ id: 'center', isCenter: true }),
		node({ id: 'n1', retention: 0.9 }),
		node({ id: 'n2', retention: 0.2 }),
		node({ id: 'n3', retention: 0.05 })
	]);

	it('same seed → identical field; different seed → different field', () => {
		const g = buildObservatoryGraph(resp);
		const a = buildNodeStateArray(g, new DemoClock({ seed: 'A' }).state.rng);
		const b = buildNodeStateArray(g, new DemoClock({ seed: 'A' }).state.rng);
		const c = buildNodeStateArray(g, new DemoClock({ seed: 'B' }).state.rng);
		expect(Array.from(a.data)).toEqual(Array.from(b.data));
		expect(Array.from(a.data)).not.toEqual(Array.from(c.data));
	});

	it('anchors the center node at the origin with the largest radius', () => {
		const g = buildObservatoryGraph(resp);
		const { data } = buildNodeStateArray(g, new DemoClock({ seed: 'A' }).state.rng);
		expect(data[NODE_LANE.posRadius]).toBe(0);
		expect(data[NODE_LANE.posRadius + 1]).toBe(0);
		expect(data[NODE_LANE.posRadius + 2]).toBe(0);
		const centerRadius = data[NODE_LANE.posRadius + 3];
		for (let i = 1; i < g.nodes.length; i++) {
			expect(centerRadius).toBeGreaterThan(data[i * FLOATS_PER_NODE + NODE_LANE.posRadius + 3]);
		}
	});

	it('stores retention and packs flags', () => {
		const g = buildObservatoryGraph(
			response([
				node({ id: 'c', isCenter: true }),
				node({ id: 's', suppression_count: 2, retention: 0.5 }),
				node({ id: 'aha', tags: ['aha'], retention: 0.6 })
			])
		);
		const { data } = buildNodeStateArray(g, new DemoClock({ seed: 'A' }).state.rng);
		const byId = (id: string) => g.indexById.get(id)! * FLOATS_PER_NODE;
		expect(data[byId('s') + NODE_LANE.velRetention + 3]).toBeCloseTo(0.5, 6);
		expect(data[byId('c') + NODE_LANE.colorFlags + 3] & NODE_FLAG.isCenter).toBeTruthy();
		expect(data[byId('s') + NODE_LANE.colorFlags + 3] & NODE_FLAG.suppressed).toBeTruthy();
		expect(data[byId('aha') + NODE_LANE.colorFlags + 3] & NODE_FLAG.isAha).toBeTruthy();
	});
});

describe('meaning-layer palette (visual DNA §7.1)', () => {
	it('maps FSRS retention buckets to the real dashboard palette', () => {
		const active = toObservatoryNode(node({ id: 'a', retention: 0.9 }), 0);
		const dormant = toObservatoryNode(node({ id: 'd', retention: 0.5 }), 1);
		const silent = toObservatoryNode(node({ id: 's', retention: 0.2 }), 2);
		const gone = toObservatoryNode(node({ id: 'u', retention: 0.01 }), 3);
		expect(nodeBaseColor(active)).toEqual(hexToRgb01('#10b981')); // emerald
		expect(nodeBaseColor(dormant)).toEqual(hexToRgb01('#f59e0b')); // amber
		expect(nodeBaseColor(silent)).toEqual(hexToRgb01('#8b5cf6')); // violet
		expect(nodeBaseColor(gone)).toEqual(hexToRgb01('#6b7280')); // slate
	});

	it('aha tag overrides with gold', () => {
		const aha = toObservatoryNode(node({ id: 'x', retention: 0.9, tags: ['aha'] }), 0);
		expect(nodeBaseColor(aha)).toEqual(hexToRgb01('#FFD700'));
	});

	it('hexToRgb01 falls back to slate on malformed input', () => {
		expect(hexToRgb01('not-a-color')).toEqual(hexToRgb01('#6b7280'));
	});
});

describe('buildEdgeIndexArray', () => {
	it('emits source/target index pairs', () => {
		const g = buildObservatoryGraph(
			response(
				[node({ id: 'a' }), node({ id: 'b' }), node({ id: 'c' })],
				[
					{ source: 'a', target: 'b', weight: 1, type: 'semantic' },
					{ source: 'b', target: 'c', weight: 1, type: 'semantic' }
				]
			)
		);
		const arr = buildEdgeIndexArray(g);
		expect(arr).toHaveLength(4);
		expect(arr[0]).toBe(g.indexById.get('a'));
		expect(arr[1]).toBe(g.indexById.get('b'));
	});
});

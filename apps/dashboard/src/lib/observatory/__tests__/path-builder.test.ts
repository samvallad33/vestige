import { describe, it, expect } from 'vitest';
import { buildRecallPath, beatFrameFor } from '../path-builder';
import { buildObservatoryGraph } from '../graph-upload';
import { UINTS_PER_PATHSTEP, PATH_KIND } from '../types';
import type { GraphResponse, GraphNode, GraphEdge } from '$types';

function node(id: string, extra: Partial<GraphNode> = {}): GraphNode {
	return {
		id,
		label: `label-${id}`,
		type: 'note',
		retention: 0.8,
		tags: [],
		createdAt: '2026-07-01T00:00:00Z',
		updatedAt: '2026-07-01T00:00:00Z',
		isCenter: false,
		...extra
	};
}

function edge(source: string, target: string, weight = 1, type = 'semantic'): GraphEdge {
	return { source, target, weight, type };
}

// A small star graph around a center with a spur — enough for a real story.
function response(): GraphResponse {
	const nodes = [
		node('center', { isCenter: true }),
		node('a'),
		node('b'),
		node('c'),
		node('d', { createdAt: '2026-07-02T00:00:00Z', updatedAt: '2026-07-02T00:00:00Z' })
	];
	const edges = [
		edge('center', 'a', 5),
		edge('center', 'b', 3),
		edge('a', 'c', 2),
		edge('c', 'd', 1)
	];
	return {
		nodes,
		edges,
		center_id: 'center',
		depth: 3,
		nodeCount: nodes.length,
		edgeCount: edges.length
	};
}

describe('beatFrameFor', () => {
	it('starts at frame 60 and advances 60 per beat', () => {
		expect(beatFrameFor(0)).toBe(60);
		expect(beatFrameFor(3)).toBe(240);
	});

	it('an 8-beat story + afterglow fits inside the 720-frame loop', () => {
		// last beat at 60 + 7·60 = 480; afterglow envelope ends ≤ +200 frames
		expect(beatFrameFor(7) + 200).toBeLessThan(720);
	});
});

describe('buildRecallPath', () => {
	it('produces steps that reference valid stable node indices', () => {
		const resp = response();
		const graph = buildObservatoryGraph(resp);
		const { steps, data } = buildRecallPath(resp, graph);

		expect(steps.length).toBeGreaterThan(1);
		for (const s of steps) {
			expect(s.targetIndex).toBeGreaterThanOrEqual(0);
			expect(s.targetIndex).toBeLessThan(graph.nodes.length);
			expect(s.sourceIndex).toBeGreaterThanOrEqual(0);
			expect(s.sourceIndex).toBeLessThan(graph.nodes.length);
		}
		expect(data.length).toBe(steps.length * UINTS_PER_PATHSTEP);
	});

	it('starts the story at the center with itself as source', () => {
		const resp = response();
		const graph = buildObservatoryGraph(resp);
		const { steps } = buildRecallPath(resp, graph);
		expect(steps[0].nodeId).toBe('center');
		expect(steps[0].sourceIndex).toBe(steps[0].targetIndex);
		expect(steps[0].beatFrame).toBe(60);
	});

	it('is deterministic — same data → identical steps', () => {
		const resp = response();
		const graph = buildObservatoryGraph(resp);
		const a = buildRecallPath(resp, graph);
		const b = buildRecallPath(resp, graph);
		expect(a.steps).toEqual(b.steps);
		expect(Array.from(a.data)).toEqual(Array.from(b.data));
	});

	it('marks contradiction beats as backward-cause hops', () => {
		const resp = response();
		// pathfinder treats 'contradicts' edge type as a contradiction
		resp.edges.push({ source: 'b', target: 'd', weight: 4, type: 'contradicts' });
		const graph = buildObservatoryGraph(resp);
		const { steps } = buildRecallPath(resp, graph);
		const kinds = new Set(steps.map((s) => s.beatKind));
		if (kinds.has('contradiction')) {
			const c = steps.find((s) => s.beatKind === 'contradiction')!;
			expect(c.kind).toBe(PATH_KIND.backwardCause);
		}
		// every step kind is one of the two GPU lanes either way
		for (const s of steps) {
			expect([PATH_KIND.recall, PATH_KIND.backwardCause]).toContain(s.kind);
		}
	});

	it('survives an empty graph', () => {
		const resp: GraphResponse = {
			nodes: [],
			edges: [],
			center_id: '',
			depth: 3,
			nodeCount: 0,
			edgeCount: 0
		};
		const graph = buildObservatoryGraph(resp);
		const { steps, data } = buildRecallPath(resp, graph);
		expect(steps).toHaveLength(0);
		expect(data.length).toBeGreaterThan(0); // placeholder lane, no zero-size buffer
	});
});

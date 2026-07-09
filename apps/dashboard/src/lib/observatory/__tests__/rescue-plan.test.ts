import { describe, it, expect } from 'vitest';

import {
	buildRescuePlan,
	rescueEnvelopes,
	pickFailureIndex,
	bfsFromFailure,
	pickCauseIndex,
	pickLookalikes,
	layoutPositions,
	hopSlotFor,
	waveArrivalFrame,
	lookalikeFrame,
	UNREACHED,
	MAX_WAVE_STEPS,
	RESCUE_K,
	ARC_FRAME,
	VERDICT_START,
	DETONATE_FRAME
} from '../rescue-plan';
import { buildObservatoryGraph } from '../graph-upload';
import { FLOATS_PER_NODE } from '../types';
import type { GraphNode, GraphEdge, GraphResponse } from '$types';

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function gn(id: string, opts: Partial<GraphNode> = {}): GraphNode {
	return {
		id,
		label: opts.label ?? `Memory ${id}`,
		type: 'note',
		retention: opts.retention ?? 0.5,
		tags: opts.tags ?? [],
		createdAt: opts.createdAt ?? '2026-01-15T00:00:00Z',
		updatedAt: '2026-01-15T00:00:00Z',
		isCenter: opts.isCenter ?? false,
		suppression_count: opts.suppression_count
	};
}

function ge(source: string, target: string, type = 'semantic'): GraphEdge {
	return { source, target, weight: 1, type };
}

function gr(nodes: GraphNode[], edges: GraphEdge[], centerId = 'center'): GraphResponse {
	return {
		nodes,
		edges,
		center_id: centerId,
		depth: 3,
		nodeCount: nodes.length,
		edgeCount: edges.length
	};
}

/**
 * Main fixture. Stable indices after buildObservatoryGraph (center first,
 * then id-sorted): center=0, cause=1, fail=2, h1=3, h1b=4, h2=5, l1..l4=6..9.
 *
 *   fail —(temporal)— h1 —(causal)— h2 —(causal)— cause   (cause depth 3)
 *   fail — h1b, fail — center, center — l1..l4
 */
function mainFixture(): GraphResponse {
	const nodes = [
		gn('center', { isCenter: true, retention: 0.9 }),
		gn('fail', { tags: ['failure'], retention: 0.8, label: 'checkout 500s on submit' }),
		gn('h1', { retention: 0.6 }),
		gn('h1b', { retention: 0.7 }),
		gn('h2', { retention: 0.55 }),
		gn('cause', {
			retention: 0.1,
			createdAt: '2025-03-01T12:00:00Z',
			label: 'schema migration dropped index'
		}),
		gn('l1', { retention: 0.62 }),
		gn('l2', { retention: 0.63 }),
		gn('l3', { retention: 0.64 }),
		gn('l4', { retention: 0.65 })
	];
	const edges = [
		ge('fail', 'h1', 'temporal'),
		ge('h1', 'h2', 'causal'),
		ge('h2', 'cause', 'causal'),
		ge('fail', 'h1b'),
		ge('fail', 'center'),
		ge('center', 'l1'),
		ge('center', 'l2'),
		ge('center', 'l3'),
		ge('center', 'l4')
	];
	return gr(nodes, edges);
}

const SEED = 'vestige-observatory-v1';

function planFor(response: GraphResponse, seed = SEED) {
	const graph = buildObservatoryGraph(response);
	return { graph, plan: buildRescuePlan(response, graph, seed) };
}

// ---------------------------------------------------------------------------
// 1. Determinism
// ---------------------------------------------------------------------------

describe('determinism', () => {
	it('same graph + seed → identical plan, byte-identical typed arrays', () => {
		const r = mainFixture();
		const { plan: p1 } = planFor(r);
		const { plan: p2 } = planFor(r);
		expect(p1).toEqual(p2);
		expect(Array.from(p1.waveData)).toEqual(Array.from(p2.waveData));
		expect(Array.from(p1.pathData)).toEqual(Array.from(p2.pathData));
		expect(Array.from(p1.hopDepths)).toEqual(Array.from(p2.hopDepths));
	});
});

// ---------------------------------------------------------------------------
// 2. Failure selection
// ---------------------------------------------------------------------------

describe('pickFailureIndex', () => {
	it('never the center; prefers failure-tagged well-connected node; degree ≥ 2 when available', () => {
		const r = mainFixture();
		const graph = buildObservatoryGraph(r);
		const positions = layoutPositions(graph, SEED);
		const failure = pickFailureIndex(graph, positions);
		expect(failure).not.toBe(graph.centerIndex);
		expect(graph.nodes[failure].id).toBe('fail'); // tagged 'failure', degree 3
	});

	it('falls back to a non-center node when nothing is tagged and degrees are low', () => {
		const r = gr(
			[gn('center', { isCenter: true }), gn('a'), gn('b')],
			[ge('a', 'b')]
		);
		const graph = buildObservatoryGraph(r);
		const failure = pickFailureIndex(graph, layoutPositions(graph, SEED));
		expect(failure).not.toBe(graph.centerIndex);
	});
});

// ---------------------------------------------------------------------------
// 3. BFS exactness
// ---------------------------------------------------------------------------

describe('bfsFromFailure', () => {
	it('exact depths on a chain, disconnected node UNREACHED and absent from pathData', () => {
		// chain: center — a(failure) — b — c — d — e, plus disconnected g
		const r = gr(
			[
				gn('center', { isCenter: true }),
				gn('a', { tags: ['failure'] }),
				gn('b'),
				gn('c'),
				gn('d', { retention: 0.9 }),
				gn('e', { retention: 0.2 }),
				gn('g')
			],
			[ge('center', 'a'), ge('a', 'b'), ge('b', 'c'), ge('c', 'd'), ge('d', 'e')]
		);
		const graph = buildObservatoryGraph(r);
		const idx = (id: string) => graph.indexById.get(id)!;
		const { depths } = bfsFromFailure(graph, idx('a'));
		expect(depths[idx('a')]).toBe(0);
		expect(depths[idx('center')]).toBe(1);
		expect(depths[idx('b')]).toBe(1);
		expect(depths[idx('c')]).toBe(2);
		expect(depths[idx('d')]).toBe(3);
		expect(depths[idx('e')]).toBe(4);
		expect(depths[idx('g')]).toBe(UNREACHED);

		const plan = buildRescuePlan(r, graph, SEED);
		expect(plan.viable).toBe(true);
		// A disconnected node can be a LOOKALIKE (nearest in layout — "looks
		// similar, causally unrelated" is the story), so kind-2 probe beams may
		// target it. But the backward wave walks REAL graph edges: no kind-1
		// wave/arc step may ever touch an unreached node.
		for (let s = 0; s < plan.pathData.length / 4; s++) {
			if (plan.pathData[s * 4 + 3] !== 1) continue;
			expect(plan.pathData[s * 4 + 0]).not.toBe(idx('g'));
			expect(plan.pathData[s * 4 + 1]).not.toBe(idx('g'));
		}
	});
});

// ---------------------------------------------------------------------------
// 4. Cause selection
// ---------------------------------------------------------------------------

describe('pickCauseIndex', () => {
	it('depth ≥ 3 when available; lowest retention among depth ≥ 3 wins', () => {
		const r = mainFixture();
		const graph = buildObservatoryGraph(r);
		const idx = (id: string) => graph.indexById.get(id)!;
		const { depths } = bfsFromFailure(graph, idx('fail'));
		const cause = pickCauseIndex(r, graph, depths, idx('fail'));
		expect(cause.index).toBe(idx('cause'));
		expect(cause.depth).toBe(3);
	});

	it('retention tie → older createdAt wins', () => {
		// two depth-3 candidates with equal retention, different ages
		const r = gr(
			[
				gn('center', { isCenter: true }),
				gn('f', { tags: ['failure'] }),
				gn('x'),
				gn('y'),
				gn('old', { retention: 0.2, createdAt: '2025-01-01T00:00:00Z' }),
				gn('new', { retention: 0.2, createdAt: '2026-06-01T00:00:00Z' })
			],
			[
				ge('center', 'f'),
				ge('f', 'x'),
				ge('x', 'y'),
				ge('y', 'old'),
				ge('y', 'new')
			]
		);
		const graph = buildObservatoryGraph(r);
		const idx = (id: string) => graph.indexById.get(id)!;
		const { depths } = bfsFromFailure(graph, idx('f'));
		expect(depths[idx('old')]).toBe(3);
		expect(depths[idx('new')]).toBe(3);
		const cause = pickCauseIndex(r, graph, depths, idx('f'));
		expect(cause.index).toBe(idx('old'));
	});
});

// ---------------------------------------------------------------------------
// 5. Edge-type-preferring parents
// ---------------------------------------------------------------------------

describe('edge-type parents', () => {
	it('equal-hop dual parents: the causal edge wins the parent chain', () => {
		// f — u1 (semantic) — v ; f — u2 (semantic) — v via causal edge u2—v
		const r = gr(
			[
				gn('center', { isCenter: true }),
				gn('f', { tags: ['failure'] }),
				gn('u1'),
				gn('u2'),
				gn('v', { retention: 0.2 })
			],
			[
				ge('center', 'f'),
				ge('f', 'u1', 'semantic'),
				ge('f', 'u2', 'semantic'),
				ge('u1', 'v', 'semantic'),
				ge('u2', 'v', 'causal')
			]
		);
		const graph = buildObservatoryGraph(r);
		const idx = (id: string) => graph.indexById.get(id)!;
		const { depths, parents } = bfsFromFailure(graph, idx('f'));
		expect(depths[idx('v')]).toBe(2);
		expect(parents[idx('v')]).toBe(idx('u2'));

		// pathData contains the (u2 → v) wave step
		const plan = buildRescuePlan(r, graph, SEED);
		let found = false;
		for (let s = 0; s < plan.pathData.length / 4; s++) {
			if (
				plan.pathData[s * 4 + 0] === idx('u2') &&
				plan.pathData[s * 4 + 1] === idx('v') &&
				plan.pathData[s * 4 + 3] === 1
			) {
				found = true;
			}
		}
		expect(found).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// 6. Relaxation ladder + non-viable
// ---------------------------------------------------------------------------

describe('cause relaxation ladder', () => {
	it('relaxes depth ≥ 3 → 2 on a short chain', () => {
		// center — f(failure) — x — y : deepest candidate at depth 2
		const r = gr(
			[
				gn('center', { isCenter: true }),
				gn('f', { tags: ['failure'] }),
				gn('x'),
				gn('y', { retention: 0.3 })
			],
			[ge('center', 'f'), ge('f', 'x'), ge('x', 'y')]
		);
		const { graph, plan } = planFor(r);
		expect(plan.viable).toBe(true);
		expect(plan.causeIndex).toBe(graph.indexById.get('y')!);
		expect(plan.causeDepth).toBe(2);
	});

	it('1-node graph → viable:false, no throw', () => {
		const r = gr([gn('center', { isCenter: true })], []);
		const { plan } = planFor(r);
		expect(plan.viable).toBe(false);
		expect(plan.pathMetas).toEqual([]);
		expect(plan.spineBeats).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
// 7. Lookalikes
// ---------------------------------------------------------------------------

describe('pickLookalikes', () => {
	it('K = min(4, eligible), layout distances non-decreasing, excludes failure/cause/center', () => {
		const r = mainFixture();
		const graph = buildObservatoryGraph(r);
		const plan = buildRescuePlan(r, graph, SEED);
		expect(plan.viable).toBe(true);
		expect(plan.lookalikeIndices.length).toBe(Math.min(RESCUE_K, graph.nodes.length - 3));

		// Recompute via the REAL layout function + same seed.
		const positions = layoutPositions(graph, SEED);
		const fi = plan.failureIndex;
		const d2 = (i: number) => {
			const dx = positions[i * FLOATS_PER_NODE + 0] - positions[fi * FLOATS_PER_NODE + 0];
			const dy = positions[i * FLOATS_PER_NODE + 1] - positions[fi * FLOATS_PER_NODE + 1];
			const dz = positions[i * FLOATS_PER_NODE + 2] - positions[fi * FLOATS_PER_NODE + 2];
			return dx * dx + dy * dy + dz * dz;
		};
		for (let k = 1; k < plan.lookalikeIndices.length; k++) {
			expect(d2(plan.lookalikeIndices[k])).toBeGreaterThanOrEqual(
				d2(plan.lookalikeIndices[k - 1])
			);
		}
		for (const li of plan.lookalikeIndices) {
			expect(li).not.toBe(plan.failureIndex);
			expect(li).not.toBe(plan.causeIndex);
			expect(li).not.toBe(graph.centerIndex);
		}

		// direct call agrees with the plan
		const direct = pickLookalikes(
			positions,
			graph.nodes.length,
			plan.failureIndex,
			plan.causeIndex,
			graph.centerIndex
		);
		expect(direct).toEqual(plan.lookalikeIndices);
	});
});

// ---------------------------------------------------------------------------
// 8. SEAM PROOF — every node, frames 0 and 719 all-zero
// ---------------------------------------------------------------------------

describe('loop seam', () => {
	it('rescueEnvelopes is exactly zero at frames 0 and 719 for EVERY packed word', () => {
		const r = mainFixture();
		const { plan } = planFor(r);
		expect(plan.viable).toBe(true);
		for (const packed of plan.waveData) {
			for (const frame of [0, 719]) {
				const e = rescueEnvelopes(frame, packed, plan.consts);
				expect(Math.abs(e.x)).toBeLessThan(1e-6);
				expect(Math.abs(e.y)).toBeLessThan(1e-6);
				expect(Math.abs(e.z)).toBeLessThan(1e-6);
				expect(Math.abs(e.w)).toBeLessThan(1e-6);
			}
		}
	});
});

// ---------------------------------------------------------------------------
// 9. Envelopes fire on the beat map
// ---------------------------------------------------------------------------

describe('envelopes fire', () => {
	it('y, x, z, w peak on their beats', () => {
		const r = mainFixture();
		const { plan } = planFor(r);
		const c = plan.consts;

		// searchlight: y > 0.9 at Fk on lookalike k
		plan.lookalikeIndices.forEach((li, k) => {
			const e = rescueEnvelopes(lookalikeFrame(k), plan.waveData[li], c);
			expect(e.y).toBeGreaterThan(0.9);
		});

		// cause ignition: x > 0.99 at frame 580
		const cw = plan.waveData[plan.causeIndex];
		expect(rescueEnvelopes(580, cw, c).x).toBeGreaterThan(0.99);

		// backward wave: max z > 0.7 within [W(d), W(d)+28] on a depth-1 node
		const d1 = plan.hopDepths.findIndex((d, i) => d === 1 && i !== plan.failureIndex);
		expect(d1).toBeGreaterThanOrEqual(0);
		const wd = waveArrivalFrame(1, plan.hopSlot);
		let zMax = 0;
		for (let f = wd; f <= wd + 28; f++) {
			zMax = Math.max(zMax, rescueEnvelopes(f, plan.waveData[d1], c).z);
		}
		expect(zMax).toBeGreaterThan(0.7);

		// detonation: w > 0.9 at frame 105 on the failure
		expect(rescueEnvelopes(105, plan.waveData[plan.failureIndex], c).w).toBeGreaterThan(0.9);
	});
});

// ---------------------------------------------------------------------------
// 10. Packing round-trip
// ---------------------------------------------------------------------------

describe('waveData packing', () => {
	it('depth/roles/k decode for failure, cause, lookalikes, plain and unreached nodes', () => {
		// use the BFS-chain fixture with a disconnected node
		const r = gr(
			[
				gn('center', { isCenter: true }),
				gn('a', { tags: ['failure'] }),
				gn('b'),
				gn('c'),
				gn('d', { retention: 0.9 }),
				gn('e', { retention: 0.2 }),
				gn('g')
			],
			[ge('center', 'a'), ge('a', 'b'), ge('b', 'c'), ge('c', 'd'), ge('d', 'e')]
		);
		const graph = buildObservatoryGraph(r);
		const idx = (id: string) => graph.indexById.get(id)!;
		const plan = buildRescuePlan(r, graph, SEED);
		expect(plan.viable).toBe(true);

		const fw = plan.waveData[plan.failureIndex];
		expect(fw & 0xffff).toBe(0);
		expect(fw & 0x10000).not.toBe(0);
		expect(fw & 0x20000).toBe(0);

		const cw = plan.waveData[plan.causeIndex];
		expect(cw & 0xffff).toBe(plan.causeDepth);
		expect(cw & 0x20000).not.toBe(0);
		expect(cw & 0x10000).toBe(0);

		plan.lookalikeIndices.forEach((li, k) => {
			const w = plan.waveData[li];
			expect(w & 0x40000).not.toBe(0);
			expect((w >>> 19) & 0x7).toBe(k);
			expect(w & 0xffff).toBe(plan.hopDepths[li]);
		});

		// unreached node round-trips 0xFFFF and is never failure/cause
		const gw = plan.waveData[idx('g')];
		expect(gw & 0xffff).toBe(UNREACHED);
		expect(gw & 0x30000).toBe(0);

		// plain node (no role bits at all) — main fixture has 3 non-role nodes
		const rich = mainFixture();
		const richGraph = buildObservatoryGraph(rich);
		const richPlan = buildRescuePlan(rich, richGraph, SEED);
		const roles = new Set([
			richPlan.failureIndex,
			richPlan.causeIndex,
			...richPlan.lookalikeIndices
		]);
		const plain = richGraph.nodes.findIndex(
			(n) => !roles.has(n.index) && n.index !== richGraph.centerIndex
		);
		expect(plain).toBeGreaterThanOrEqual(0);
		expect(richPlan.waveData[plain] & 0x7f0000).toBe(0);
		expect(richPlan.waveData[plain] & 0xffff).toBe(richPlan.hopDepths[plain]);
	});
});

// ---------------------------------------------------------------------------
// 11. Ribbon-window invariant + step ordering
// ---------------------------------------------------------------------------

describe('path steps', () => {
	it('every beat frame keeps its ribbon window inside [0, 719]; probes first, arc last', () => {
		const r = mainFixture();
		const { graph, plan } = planFor(r);
		const count = plan.pathData.length / 4;
		expect(count).toBeLessThanOrEqual(RESCUE_K + MAX_WAVE_STEPS + 1);

		for (let s = 0; s < count; s++) {
			const bf = plan.pathData[s * 4 + 2];
			expect(bf - 46).toBeGreaterThanOrEqual(0);
			expect(bf + 90).toBeLessThanOrEqual(719);
		}

		// probes first, kind 2, beats 138/166/194/222
		const K = plan.lookalikeIndices.length;
		for (let k = 0; k < K; k++) {
			expect(plan.pathData[k * 4 + 0]).toBe(plan.failureIndex);
			expect(plan.pathData[k * 4 + 1]).toBe(plan.lookalikeIndices[k]);
			expect(plan.pathData[k * 4 + 2]).toBe(138 + 28 * k);
			expect(plan.pathData[k * 4 + 3]).toBe(2);
		}

		// wave steps: kind 1 with bf = W(depth(dst))
		for (let s = K; s < count - 1; s++) {
			expect(plan.pathData[s * 4 + 3]).toBe(1);
			const dst = plan.pathData[s * 4 + 1];
			expect(plan.pathData[s * 4 + 2]).toBe(
				waveArrivalFrame(plan.hopDepths[dst], plan.hopSlot)
			);
			expect(plan.pathData[s * 4 + 2]).toBeLessThanOrEqual(514);
		}

		// arc last: cause → failure at 560, kind 1
		const last = count - 1;
		expect(plan.pathData[last * 4 + 0]).toBe(plan.causeIndex);
		expect(plan.pathData[last * 4 + 1]).toBe(plan.failureIndex);
		expect(plan.pathData[last * 4 + 2]).toBe(ARC_FRAME);
		expect(plan.pathData[last * 4 + 3]).toBe(1);
		void graph;
	});
});

// ---------------------------------------------------------------------------
// 12. setPathSteps contract
// ---------------------------------------------------------------------------

describe('pathMetas contract', () => {
	it('pathMetas is 1:1 with pathData steps (draw count = metas.length)', () => {
		const { plan } = planFor(mainFixture());
		expect(plan.pathMetas.length).toBe(plan.pathData.length / 4);
	});
});

// ---------------------------------------------------------------------------
// 13. Spine beats
// ---------------------------------------------------------------------------

describe('spine beats', () => {
	it('beatFrames strictly increasing and unique; real labels present', () => {
		const { plan } = planFor(mainFixture());
		const frames = plan.spineBeats.map((b) => b.beatFrame);
		for (let i = 1; i < frames.length; i++) {
			expect(frames[i]).toBeGreaterThan(frames[i - 1]);
		}
		expect(new Set(frames).size).toBe(frames.length);
		expect(frames[0]).toBe(DETONATE_FRAME);
		expect(frames[frames.length - 1]).toBe(VERDICT_START);

		const labels = plan.spineBeats.map((b) => b.label);
		expect(labels[0]).toContain('checkout 500s on submit');
		expect(labels.some((l) => l.includes('lookalike ✗'))).toBe(true);
		expect(labels.some((l) => l.includes('schema migration dropped index'))).toBe(true);
		expect(labels[labels.length - 1]).toBe('root cause found');
	});
});

// ---------------------------------------------------------------------------
// 14. Verdict receipt
// ---------------------------------------------------------------------------

describe('verdict', () => {
	it('real labels, real date, hop count and K', () => {
		const { plan } = planFor(mainFixture());
		expect(plan.verdict.causeLabel).toBe('schema migration dropped index');
		expect(plan.verdict.failureLabel).toBe('checkout 500s on submit');
		expect(plan.verdict.causeDate).toBe('2025-03-01');
		expect(plan.verdict.hops).toBe(3);
		expect(plan.verdict.k).toBe(4);
		expect(plan.verdict.receipt).toBe('3 hops back · 2025-03-01 · vector search: 0 for 4');
	});

	it('labels truncate at 64 chars with an ellipsis', () => {
		const long = 'x'.repeat(100);
		const r = mainFixture();
		const causeNode = r.nodes.find((n) => n.id === 'cause')!;
		causeNode.label = long;
		const { plan } = planFor(r);
		expect(plan.verdict.causeLabel.length).toBe(65);
		expect(plan.verdict.causeLabel.endsWith('…')).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// 15. Degenerates survive
// ---------------------------------------------------------------------------

describe('degenerate graphs', () => {
	it('0-node, 2-node and edgeless graphs → viable:false, no throw, min pathData', () => {
		const zero = gr([], [], '');
		const { plan: p0 } = planFor(zero);
		expect(p0.viable).toBe(false);
		expect(p0.pathData.length).toBe(4);

		const two = gr([gn('center', { isCenter: true }), gn('a')], []);
		const { plan: p2 } = planFor(two);
		expect(p2.viable).toBe(false);
		expect(p2.pathData.length).toBe(4);

		const edgeless = gr([gn('center', { isCenter: true }), gn('a'), gn('b'), gn('c')], []);
		const { plan: pe } = planFor(edgeless);
		expect(pe.viable).toBe(false);
		expect(pe.spineBeats).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
// 16. hopSlot clamping
// ---------------------------------------------------------------------------

describe('hopSlot', () => {
	it('D=3 → 84; D=18 → 14 (clamped); W(D) ≤ 514', () => {
		expect(hopSlotFor(3)).toBe(84);
		expect(hopSlotFor(18)).toBe(14);
		expect(waveArrivalFrame(3, hopSlotFor(3))).toBe(512);
		expect(waveArrivalFrame(18, hopSlotFor(18))).toBe(512);
		for (let d = 1; d <= 20; d++) {
			expect(waveArrivalFrame(d, hopSlotFor(d))).toBeLessThanOrEqual(514);
		}
		// the main fixture's plan agrees
		const { plan } = planFor(mainFixture());
		expect(plan.hopSlot).toBe(84);
		expect(plan.consts.hopSlot).toBe(84);
	});
});

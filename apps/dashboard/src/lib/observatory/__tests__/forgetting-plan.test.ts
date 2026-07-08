import { describe, it, expect } from 'vitest';

import {
	buildForgettingPlan,
	forgettingEnvelopes,
	horizonDrift,
	pickDrifting,
	pickRescued,
	rescueFrame,
	FORGETTING_K,
	FADING_BASE,
	FADING_INTERVAL,
	RESCUE_BASE,
	RESCUE_INTERVAL,
	SINK_BEAT_FRAME,
	RETRIEVABLE_BEAT_FRAME
} from '../forgetting-plan';
import { buildObservatoryGraph } from '../graph-upload';
import { PATH_KIND } from '../types';
import type { GraphNode, GraphEdge, GraphResponse } from '$types';

// ---------------------------------------------------------------------------
// Fixture helpers (cloned from rescue-plan.test.ts)
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

/** 10-node fixture: center + 9, varied retention → driftCount = max(3, round(2.25)) = 3. */
function tenNodeFixture(): GraphResponse {
	const nodes = [
		gn('center', { isCenter: true, retention: 0.9 }),
		gn('a', { retention: 0.05, label: 'oldest forgotten fact' }),
		gn('b', { retention: 0.12 }),
		gn('c', { retention: 0.2 }),
		gn('d', { retention: 0.55 }),
		gn('e', { retention: 0.6 }),
		gn('f', { retention: 0.65 }),
		gn('g', { retention: 0.7 }),
		gn('h', { retention: 0.75 }),
		gn('i', { retention: 0.8 })
	];
	const edges = [
		ge('center', 'a'),
		ge('center', 'b'),
		ge('b', 'c'),
		ge('center', 'd'),
		ge('d', 'e')
	];
	return gr(nodes, edges);
}

/** 40-node fixture: center + 39 → driftCount = round(0.25·39) = 10, K = 3 rescued. */
function fortyNodeFixture(): GraphResponse {
	const nodes: GraphNode[] = [gn('center', { isCenter: true, retention: 0.95 })];
	const edges: GraphEdge[] = [];
	for (let i = 0; i < 39; i++) {
		const id = `n${String(i).padStart(2, '0')}`;
		nodes.push(gn(id, { retention: 0.04 + i * 0.02, label: `memory ${id}` }));
		edges.push(ge('center', id));
	}
	// Extra connectivity inside the drifting band so rescue scores separate.
	edges.push(ge('n07', 'n08'));
	edges.push(ge('n07', 'n09'));
	edges.push(ge('n08', 'n09'));
	return gr(nodes, edges);
}

function planFor(response: GraphResponse) {
	const graph = buildObservatoryGraph(response);
	return { graph, plan: buildForgettingPlan(graph) };
}

// ---------------------------------------------------------------------------
// 1. Determinism
// ---------------------------------------------------------------------------

describe('determinism', () => {
	it('same graph → identical plan, byte-identical typed arrays', () => {
		const r = fortyNodeFixture();
		const { plan: p1 } = planFor(r);
		const { plan: p2 } = planFor(r);
		expect(p1).toEqual(p2);
		expect(Array.from(p1.horizonData)).toEqual(Array.from(p2.horizonData));
		expect(Array.from(p1.pathData)).toEqual(Array.from(p2.pathData));
	});
});

// ---------------------------------------------------------------------------
// 2. pickDrifting — 25% clamp formula, center excluded, tie → lower index
// ---------------------------------------------------------------------------

describe('pickDrifting', () => {
	it('10-node graph → driftCount 3 (floor of min(3, n−1) beats round(0.25·9)=2)', () => {
		const graph = buildObservatoryGraph(tenNodeFixture());
		const drifting = pickDrifting(graph);
		expect(drifting.length).toBe(3);
		expect(drifting).not.toContain(graph.centerIndex);
		// lowest retention first
		const ids = drifting.map((i) => graph.nodes[i].id);
		expect(ids).toEqual(['a', 'b', 'c']);
	});

	it('40-node graph → driftCount round(0.25·39) = 10', () => {
		const graph = buildObservatoryGraph(fortyNodeFixture());
		const drifting = pickDrifting(graph);
		expect(drifting.length).toBe(10);
		expect(drifting).not.toContain(graph.centerIndex);
		// retention non-decreasing along the rank order
		for (let i = 1; i < drifting.length; i++) {
			expect(graph.nodes[drifting[i]].retention).toBeGreaterThanOrEqual(
				graph.nodes[drifting[i - 1]].retention
			);
		}
	});

	it('retention tie → lower node index drifts first', () => {
		const r = gr(
			[
				gn('center', { isCenter: true, retention: 0.9 }),
				gn('a', { retention: 0.2 }),
				gn('b', { retention: 0.2 }),
				gn('c', { retention: 0.8 })
			],
			[]
		);
		const graph = buildObservatoryGraph(r);
		const drifting = pickDrifting(graph);
		const ia = graph.indexById.get('a')!;
		const ib = graph.indexById.get('b')!;
		expect(drifting.indexOf(ia)).toBeLessThan(drifting.indexOf(ib));
	});
});

// ---------------------------------------------------------------------------
// 3. pickRescued — subset of drifting, K = min(3, driftCount), score ordering
// ---------------------------------------------------------------------------

describe('pickRescued', () => {
	it('rescued ⊆ drifting, K = min(3, driftCount), highest 2·retention + degree/8 wins', () => {
		const graph = buildObservatoryGraph(fortyNodeFixture());
		const drifting = pickDrifting(graph);
		const rescued = pickRescued(graph, drifting);
		expect(rescued.length).toBe(Math.min(FORGETTING_K, drifting.length));
		for (const idx of rescued) expect(drifting).toContain(idx);

		// recompute scores and verify the rescued are the top-K
		const degree = new Uint32Array(graph.nodes.length);
		for (const e of graph.edges) {
			degree[e.sourceIndex]++;
			degree[e.targetIndex]++;
		}
		const score = (i: number) => 2 * graph.nodes[i].retention + Math.min(degree[i], 8) / 8;
		const expected = drifting.slice().sort((a, b) => score(b) - score(a) || a - b).slice(0, 3);
		expect(rescued).toEqual(expected);
	});

	it('driftCount 1 → K = 1, the sole drifter is rescued', () => {
		const r = gr([gn('center', { isCenter: true }), gn('a', { retention: 0.1 })], [
			ge('center', 'a')
		]);
		const graph = buildObservatoryGraph(r);
		const drifting = pickDrifting(graph);
		expect(drifting.length).toBe(1);
		const rescued = pickRescued(graph, drifting);
		expect(rescued).toEqual(drifting);
	});
});

// ---------------------------------------------------------------------------
// 4. Packing round-trip
// ---------------------------------------------------------------------------

describe('horizonData packing', () => {
	it('rank/isDrifting/isRescued/slot decode; non-drifting words are exactly 0', () => {
		const { graph, plan } = planFor(fortyNodeFixture());
		expect(plan.viable).toBe(true);
		const driftCount = plan.driftingIndices.length;

		plan.driftingIndices.forEach((idx, i) => {
			const w = plan.horizonData[idx];
			expect(w & 0x100).not.toBe(0);
			expect(w & 0xff).toBe(Math.round((255 * i) / Math.max(1, driftCount - 1)));
		});
		plan.rescuedIndices.forEach((idx, k) => {
			const w = plan.horizonData[idx];
			expect(w & 0x200).not.toBe(0);
			expect(w & 0x100).not.toBe(0); // rescued word also carries isDrifting
			expect((w >>> 10) & 0x3).toBe(k);
		});
		const roleSet = new Set(plan.driftingIndices);
		for (let i = 0; i < graph.nodes.length; i++) {
			if (!roleSet.has(i)) expect(plan.horizonData[i]).toBe(0);
		}
		expect(plan.horizonData[graph.centerIndex]).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// 5. SEAM PROOF — every word, frames 0 and 719 all-zero
// ---------------------------------------------------------------------------

describe('loop seam', () => {
	it('forgettingEnvelopes is exactly zero at frames 0 and 719 for EVERY packed word', () => {
		const { plan } = planFor(fortyNodeFixture());
		expect(plan.viable).toBe(true);
		for (const packed of plan.horizonData) {
			for (const frame of [0, 719]) {
				const e = forgettingEnvelopes(frame, packed);
				expect(Math.abs(e.x)).toBeLessThan(1e-6);
				expect(Math.abs(e.y)).toBeLessThan(1e-6);
				expect(Math.abs(e.z)).toBeLessThan(1e-6);
				expect(Math.abs(e.w)).toBeLessThan(1e-6);
			}
		}
	});
});

// ---------------------------------------------------------------------------
// 6. Envelopes fire on the beat map
// ---------------------------------------------------------------------------

describe('envelopes fire', () => {
	it('unrescued drift: z > 0.5 @420, > 0.95 @655, ≤ 1+1e-6 at all frames', () => {
		const { plan } = planFor(fortyNodeFixture());
		const rescuedSet = new Set(plan.rescuedIndices);
		const unrescued = plan.driftingIndices.filter((i) => !rescuedSet.has(i));
		expect(unrescued.length).toBeGreaterThan(0);
		for (const idx of unrescued) {
			const w = plan.horizonData[idx];
			expect(forgettingEnvelopes(420, w).z).toBeGreaterThan(0.5);
			expect(forgettingEnvelopes(655, w).z).toBeGreaterThan(0.95);
			for (let f = 0; f < 720; f++) {
				expect(forgettingEnvelopes(f, w).z).toBeLessThanOrEqual(1 + 1e-6);
			}
		}
	});

	it('rescued slot k: z < 1e-6 @rk+6, x > 0.95 @rk, x < 1e-6 @rk+140', () => {
		const { plan } = planFor(fortyNodeFixture());
		plan.rescuedIndices.forEach((idx, k) => {
			const w = plan.horizonData[idx];
			const rk = rescueFrame(k);
			expect(Math.abs(forgettingEnvelopes(rk + 6, w).z)).toBeLessThan(1e-6);
			expect(forgettingEnvelopes(rk, w).x).toBeGreaterThan(0.95);
			expect(Math.abs(forgettingEnvelopes(rk + 140, w).x)).toBeLessThan(1e-6);
		});
	});
});

// ---------------------------------------------------------------------------
// 7. Monotone unrescued sink over 0..655
// ---------------------------------------------------------------------------

describe('monotone sink', () => {
	it('unrescued z is non-decreasing over frames 0..655', () => {
		const { plan } = planFor(fortyNodeFixture());
		const rescuedSet = new Set(plan.rescuedIndices);
		const unrescued = plan.driftingIndices.filter((i) => !rescuedSet.has(i));
		for (const idx of unrescued) {
			const w = plan.horizonData[idx];
			let prev = forgettingEnvelopes(0, w).z;
			for (let f = 1; f <= 655; f++) {
				const z = forgettingEnvelopes(f, w).z;
				expect(z).toBeGreaterThanOrEqual(prev - 1e-9);
				prev = z;
			}
		}
	});
});

// ---------------------------------------------------------------------------
// 8. horizonDrift — CPU mirror of the demo-3 vertex displacement
// ---------------------------------------------------------------------------

describe('horizonDrift', () => {
	it('[0,0,0] at dz=0; ~40.5 units at dz=1, downward, away from axis; no NaN on-axis', () => {
		expect(horizonDrift(0, [30, 10, 40])).toEqual([0, 0, 0]);

		const p: [number, number, number] = [30, 10, 40];
		const d = horizonDrift(1, p);
		const mag = Math.hypot(d[0], d[1], d[2]);
		expect(mag).toBeGreaterThanOrEqual(38);
		expect(mag).toBeLessThanOrEqual(42);
		expect(d[1]).toBeLessThan(0);
		// away·p ≥ 0: the drift pushes outward, never inward
		expect(d[0] * p[0] + d[2] * p[2]).toBeGreaterThanOrEqual(0);

		// exactly on the y axis: r_xz clamps to 0.001, no NaN
		const axis = horizonDrift(1, [0, 55, 0]);
		expect(Number.isNaN(axis[0])).toBe(false);
		expect(Number.isNaN(axis[1])).toBe(false);
		expect(Number.isNaN(axis[2])).toBe(false);
		expect(axis).toEqual([0, -34, 0]);

		// dz clamps to 1
		const over = horizonDrift(2, p);
		expect(Math.hypot(over[0], over[1], over[2])).toBeCloseTo(mag, 9);
	});
});

// ---------------------------------------------------------------------------
// 9. Ribbon window + step shape
// ---------------------------------------------------------------------------

describe('path steps', () => {
	it('K kind-0 steps, src = center, dst = rescued_k, bf = 318+60k, window inside the loop', () => {
		const { graph, plan } = planFor(fortyNodeFixture());
		const count = plan.pathData.length / 4;
		expect(count).toBe(plan.rescuedIndices.length);
		expect(plan.pathMetas.length).toBe(count);

		plan.rescuedIndices.forEach((idx, k) => {
			expect(plan.pathData[k * 4 + 0]).toBe(graph.centerIndex);
			expect(plan.pathData[k * 4 + 1]).toBe(idx);
			expect(plan.pathData[k * 4 + 2]).toBe(RESCUE_BASE + RESCUE_INTERVAL * k);
			expect(plan.pathData[k * 4 + 3]).toBe(PATH_KIND.recall);
			const bf = plan.pathData[k * 4 + 2];
			expect(bf - 46).toBeGreaterThanOrEqual(0);
			expect(bf + 90).toBeLessThanOrEqual(719);
		});
	});
});

// ---------------------------------------------------------------------------
// 10. Spine beats
// ---------------------------------------------------------------------------

describe('spine beats', () => {
	it('strictly increasing unique frames; fading/recalled labels; sink + retrievable', () => {
		const { graph, plan } = planFor(fortyNodeFixture());
		const frames = plan.spineBeats.map((b) => b.beatFrame);
		for (let i = 1; i < frames.length; i++) {
			expect(frames[i]).toBeGreaterThan(frames[i - 1]);
		}
		expect(new Set(frames).size).toBe(frames.length);
		expect(frames[0]).toBe(FADING_BASE);
		expect(frames).toContain(SINK_BEAT_FRAME);
		expect(frames[frames.length - 1]).toBe(RETRIEVABLE_BEAT_FRAME);

		const labels = plan.spineBeats.map((b) => b.label);
		// fading beats carry real labels + retention percent
		const rescuedSet = new Set(plan.rescuedIndices);
		const fading = plan.driftingIndices.filter((i) => !rescuedSet.has(i)).slice(0, 3);
		fading.forEach((idx, j) => {
			const pct = Math.round(graph.nodes[idx].retention * 100);
			expect(labels[j]).toBe(
				`fading: ${graph.nodes[idx].label} · retention ${pct}%`
			);
			expect(plan.spineBeats[j].beatFrame).toBe(FADING_BASE + FADING_INTERVAL * j);
		});
		expect(labels.some((l) => l.startsWith('recalled: '))).toBe(true);
		expect(labels).toContain('the unrecalled sink · nothing is deleted');
		expect(labels).toContain('every memory still retrievable');
	});

	it('labels truncate at 64 chars with an ellipsis', () => {
		const r = fortyNodeFixture();
		// lowest-retention node (n00) is unrescued → becomes the first fading beat
		const n00 = r.nodes.find((n) => n.id === 'n00')!;
		n00.label = 'x'.repeat(100);
		const { plan } = planFor(r);
		const fadingLabel = plan.spineBeats[0].label;
		expect(fadingLabel.startsWith('fading: ')).toBe(true);
		const memLabel = fadingLabel.slice('fading: '.length).split(' · ')[0];
		expect(memLabel.length).toBe(65);
		expect(memLabel.endsWith('…')).toBe(true);
	});
});

// ---------------------------------------------------------------------------
// 11. Degenerates survive
// ---------------------------------------------------------------------------

describe('degenerate graphs', () => {
	it('0-node and 1-node → viable:false, min pathData, no throw; 2-node → viable', () => {
		const { plan: p0 } = planFor(gr([], [], ''));
		expect(p0.viable).toBe(false);
		expect(p0.pathData.length).toBe(4);
		expect(p0.pathMetas).toEqual([]);
		expect(p0.spineBeats).toEqual([]);

		const { plan: p1 } = planFor(gr([gn('center', { isCenter: true })], []));
		expect(p1.viable).toBe(false);
		expect(Array.from(p1.horizonData)).toEqual([0]);

		const { graph: g2, plan: p2 } = planFor(
			gr([gn('center', { isCenter: true }), gn('a', { retention: 0.1 })], [ge('center', 'a')])
		);
		expect(p2.viable).toBe(true);
		expect(p2.driftingIndices).toEqual([g2.indexById.get('a')!]);
		expect(p2.rescuedIndices).toEqual([g2.indexById.get('a')!]);
		expect(p2.pathData.length).toBe(4);
		expect(p2.pathData[2]).toBe(RESCUE_BASE);
	});
});

// ---------------------------------------------------------------------------
// 12. Lane hygiene — the rescue/firewall grammars can NEVER fire in demo 3
// ---------------------------------------------------------------------------

describe('lane hygiene', () => {
	it('y and w are exactly 0 for every word at frames 0..719 step 7', () => {
		const { plan } = planFor(fortyNodeFixture());
		for (const packed of plan.horizonData) {
			for (let f = 0; f < 720; f += 7) {
				const e = forgettingEnvelopes(f, packed);
				expect(e.y).toBe(0);
				expect(e.w).toBe(0);
			}
		}
	});
});

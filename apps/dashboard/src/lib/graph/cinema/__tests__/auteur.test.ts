import { describe, it, expect, beforeEach } from 'vitest';
import { planShotsDeterministic, resolveShots, SHOT_DEFAULTS, type DirectorPlan } from '../auteur';
import { planCinemaPath } from '../pathfinder';
import { computeSignals } from '../topology';
import { makeNode, makeEdge, resetNodeCounter } from '../../__tests__/helpers';

describe('auteur — carry-forward shot resolution', () => {
	beforeEach(() => resetNodeCounter());

	function smallPath() {
		const a = makeNode({ id: 'a' });
		const b = makeNode({ id: 'b' });
		const c = makeNode({ id: 'c' });
		const edges = [makeEdge('a', 'b', { weight: 0.8 }), makeEdge('b', 'c', { weight: 0.6 })];
		return { path: planCinemaPath([a, b, c], edges, 'a'), nodes: [a, b, c], edges };
	}

	it('fills EVERY axis from a one-field shot, defaulting to today constants', () => {
		const { path } = smallPath();
		const plan: DirectorPlan = {
			source: 'backend-llm',
			logline: 'x',
			arc: 'flat',
			shots: [{ nodeId: 'a', move: 'orbit', why: 'test' }],
		};
		const resolved = resolveShots(plan, path);
		expect(resolved).toHaveLength(path.beats.length);
		// The specified field is honored…
		expect(resolved[0].move).toBe('orbit');
		// …and every other axis is a real default, never undefined.
		expect(resolved[0].standoff).toBe(SHOT_DEFAULTS.standoff);
		expect(resolved[0].flightSeconds).toBe(SHOT_DEFAULTS.flightSeconds);
		expect(resolved[0].angle).toBe('eye');
		for (const s of resolved) {
			for (const k of Object.keys(SHOT_DEFAULTS) as (keyof typeof SHOT_DEFAULTS)[]) {
				expect(s[k]).toBeDefined();
			}
			expect(typeof s.why).toBe('string');
			expect(s.why.length).toBeGreaterThan(0);
		}
	});

	it('carries non-cut axes forward to subsequent beats', () => {
		const { path } = smallPath();
		const plan: DirectorPlan = {
			source: 'backend-llm',
			logline: 'x',
			arc: 'flat',
			// Only the FIRST beat sets standoff; later beats should inherit it.
			shots: [{ nodeId: path.beats[0].nodeId, standoff: 41, why: 'set' }],
		};
		const resolved = resolveShots(plan, path);
		expect(resolved[0].standoff).toBe(41);
		expect(resolved[resolved.length - 1].standoff).toBe(41); // carried forward
	});

	it('cut never carries forward — defaults to fly each beat', () => {
		const { path } = smallPath();
		const plan: DirectorPlan = {
			source: 'backend-llm',
			logline: 'x',
			arc: 'flat',
			shots: [{ nodeId: path.beats[0].nodeId, cut: 'hard_cut', why: 'cut' }],
		};
		const resolved = resolveShots(plan, path);
		expect(resolved[0].cut).toBe('hard_cut');
		if (resolved.length > 1) expect(resolved[1].cut).toBe('fly');
	});

	it('back-fills garbage / out-of-range LLM fields from defaults', () => {
		const { path } = smallPath();
		const plan = {
			source: 'backend-llm',
			logline: 'x',
			arc: 'flat',
			shots: [
				{
					nodeId: path.beats[0].nodeId,
					move: 'teleport', // invalid enum
					standoff: 9999, // out of range
					dwellSeconds: -5, // out of range
					why: '',
				},
			],
		} as unknown as DirectorPlan;
		const resolved = resolveShots(plan, path);
		expect(resolved[0].move).toBe(SHOT_DEFAULTS.move); // invalid → default
		expect(resolved[0].standoff).toBeLessThanOrEqual(90); // clamped
		expect(resolved[0].dwellSeconds).toBeGreaterThanOrEqual(0.6); // clamped
		expect(resolved[0].why.length).toBeGreaterThan(0); // empty why → fallback
	});

	it('a null plan still yields one default shot per beat', () => {
		const { path } = smallPath();
		const resolved = resolveShots(null, path);
		expect(resolved).toHaveLength(path.beats.length);
		expect(resolved[0].move).toBe(SHOT_DEFAULTS.move);
	});
});

describe('auteur — deterministic director', () => {
	beforeEach(() => resetNodeCounter());

	it('produces a valid plan: one grounded shot per beat, every why non-empty, every nodeId real', () => {
		const nodes = Array.from({ length: 8 }, (_, i) => makeNode({ id: `n${i}` }));
		const edges = [
			makeEdge('n0', 'n1', { weight: 0.9 }),
			makeEdge('n1', 'n2', { weight: 0.2, type: 'contradiction' }),
			makeEdge('n0', 'n3', { weight: 0.5 }),
			makeEdge('n3', 'n4', { weight: 0.7 }),
		];
		const path = planCinemaPath(nodes, edges, 'n0');
		const signals = computeSignals(nodes, edges);
		const plan = planShotsDeterministic(path, signals);
		const realIds = new Set(nodes.map((n) => n.id));
		expect(plan.shots).toHaveLength(path.beats.length);
		for (const s of plan.shots) {
			expect(realIds.has(s.nodeId)).toBe(true);
			expect(s.why && s.why.length).toBeGreaterThan(0);
		}
		expect(plan.source).toBe('deterministic');
		expect(plan.logline.length).toBeGreaterThan(0);
	});

	it('directs a contradiction beat as a Dutch hard-cut crimson collision', () => {
		const a = makeNode({ id: 'a' });
		const normal = makeNode({ id: 'normal' });
		const conflict = makeNode({ id: 'conflict' });
		const edges = [
			makeEdge('a', 'normal', { weight: 0.95 }),
			makeEdge('a', 'conflict', { weight: 0.2, type: 'contradiction' }),
		];
		const path = planCinemaPath([a, normal, conflict], edges, 'a');
		const signals = computeSignals([a, normal, conflict], edges);
		const plan = planShotsDeterministic(path, signals);
		const contradictionShot = plan.shots.find((_, i) => path.beats[i].kind === 'contradiction');
		expect(contradictionShot).toBeDefined();
		expect(contradictionShot!.stormMode).toBe('contradiction');
		expect(contradictionShot!.cut).toBe('hard_cut');
		expect(contradictionShot!.dutch).toBeGreaterThan(0);
		expect(contradictionShot!.scoreCue).toBe('minor_drop');
	});

	it('ends on a crane pull-back with a major resolve', () => {
		const nodes = Array.from({ length: 5 }, (_, i) => makeNode({ id: `m${i}` }));
		const edges = nodes.slice(1).map((n, i) => makeEdge(`m${i}`, n.id, { weight: 0.6 }));
		const path = planCinemaPath(nodes, edges, 'm0');
		const signals = computeSignals(nodes, edges);
		const plan = planShotsDeterministic(path, signals);
		const last = plan.shots[plan.shots.length - 1];
		expect(last.move).toBe('crane');
		expect(last.scoreCue).toBe('major_resolve');
	});

	it('is deterministic — same inputs yield the same plan', () => {
		const nodes = Array.from({ length: 6 }, (_, i) => makeNode({ id: `d${i}` }));
		const edges = [makeEdge('d0', 'd1', { weight: 0.8 }), makeEdge('d1', 'd2', { weight: 0.5 })];
		const path = planCinemaPath(nodes, edges, 'd0');
		const sig = computeSignals(nodes, edges);
		const p1 = planShotsDeterministic(path, sig);
		const p2 = planShotsDeterministic(path, sig);
		expect(p1.shots.map((s) => s.move)).toEqual(p2.shots.map((s) => s.move));
		expect(p1.logline).toBe(p2.logline);
	});
});

describe('topology — graph signals', () => {
	beforeEach(() => resetNodeCounter());

	it('computes betweenness, clusters, and peak keystone on a real shape', () => {
		// Two clusters bridged by 'hub' → hub has the highest betweenness.
		const hub = makeNode({ id: 'hub' });
		const l1 = makeNode({ id: 'l1' });
		const l2 = makeNode({ id: 'l2' });
		const r1 = makeNode({ id: 'r1' });
		const r2 = makeNode({ id: 'r2' });
		const edges = [
			makeEdge('l1', 'l2'),
			makeEdge('l2', 'hub'),
			makeEdge('hub', 'r1'),
			makeEdge('r1', 'r2'),
		];
		const sig = computeSignals([hub, l1, l2, r1, r2], edges);
		expect(sig.peakBetweennessId).toBe('hub');
		expect(sig.nodes.get('hub')!.betweenness).toBeGreaterThan(sig.nodes.get('l1')!.betweenness);
		expect(sig.clusterCount).toBe(1); // all connected through hub
		// All signals are finite and in range.
		for (const s of sig.nodes.values()) {
			expect(s.betweenness).toBeGreaterThanOrEqual(0);
			expect(s.betweenness).toBeLessThanOrEqual(1);
			expect(Number.isFinite(s.recencyRank)).toBe(true);
		}
	});

	it('flags contradiction edges and computes surprise in range', () => {
		const a = makeNode({ id: 'a' });
		const b = makeNode({ id: 'b' });
		const edges = [makeEdge('a', 'b', { weight: 0.1, type: 'contradiction' })];
		const sig = computeSignals([a, b], edges);
		expect(sig.edges[0].isContradiction).toBe(true);
		expect(sig.edges[0].surprise).toBeGreaterThanOrEqual(0);
		expect(sig.edges[0].surprise).toBeLessThanOrEqual(1);
	});
});

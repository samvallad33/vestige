/**
 * Unit tests for Spreading Activation helpers.
 *
 * Pure-logic coverage only — the SVG render layer is not exercised here
 * (no jsdom). The six concerns we test are the ones that actually decide
 * whether the burst looks right:
 *
 *   1. Per-tick decay math (Collins & Loftus 1975, 0.93/frame)
 *   2. Compound decay after N ticks
 *   3. Threshold filter (activation < 0.05 → invisible)
 *   4. Concentric-ring placement around a source (8-per-ring, even angles)
 *   5. Color mapping (source → synapse-glow, unknown type → fallback)
 *   6. Staggered edge delay (rank ordering, ring-2 bonus)
 *   7. Event-feed filter (only NEW ActivationSpread events since lastSeen)
 *
 * The test environment is Node (vitest `environment: 'node'`) — the same
 * harness the graph + dream helper tests use.
 */
import { describe, it, expect } from 'vitest';
import {
	DECAY,
	FALLBACK_COLOR,
	MIN_VISIBLE,
	RING_GAP,
	RING_1_CAPACITY,
	SOURCE_COLOR,
	STAGGER_PER_RANK,
	STAGGER_RING_2_BONUS,
	activationColor,
	applyDecay,
	compoundDecay,
	computeRing,
	edgeStagger,
	filterNewSpreadEvents,
	initialActivation,
	isVisible,
	layoutNeighbours,
	ringPositions,
	ticksUntilInvisible,
} from '../activation-helpers';
import { NODE_TYPE_COLORS, type VestigeEvent } from '$types';

// ---------------------------------------------------------------------------
// 1. Decay math — single tick
// ---------------------------------------------------------------------------

describe('applyDecay (Collins & Loftus 1975, 0.93/frame)', () => {
	it('multiplies activation by 0.93 per tick', () => {
		expect(applyDecay(1)).toBeCloseTo(0.93, 10);
	});

	it('matches the documented constant', () => {
		expect(DECAY).toBe(0.93);
	});

	it('returns 0 for zero / negative / non-finite input', () => {
		expect(applyDecay(0)).toBe(0);
		expect(applyDecay(-0.5)).toBe(0);
		expect(applyDecay(Number.NaN)).toBe(0);
		expect(applyDecay(Number.POSITIVE_INFINITY)).toBe(0);
	});

	it('preserves strict monotonic decrease', () => {
		let a = 1;
		let prev = a;
		for (let i = 0; i < 50; i++) {
			a = applyDecay(a);
			if (a === 0) break;
			expect(a).toBeLessThan(prev);
			prev = a;
		}
	});
});

// ---------------------------------------------------------------------------
// 2. Compound decay — N ticks
// ---------------------------------------------------------------------------

describe('compoundDecay', () => {
	it('0 ticks returns the input unchanged', () => {
		expect(compoundDecay(0.8, 0)).toBe(0.8);
	});

	it('N ticks equals applyDecay called N times', () => {
		let iterative = 1;
		for (let i = 0; i < 10; i++) iterative = applyDecay(iterative);
		expect(compoundDecay(1, 10)).toBeCloseTo(iterative, 10);
	});

	it('5 ticks from 1.0 lands in the 0.69..0.70 band', () => {
		// 0.93^5 ≈ 0.6957
		const result = compoundDecay(1, 5);
		expect(result).toBeGreaterThan(0.69);
		expect(result).toBeLessThan(0.7);
	});

	it('treats negative tick counts as no-op', () => {
		expect(compoundDecay(0.5, -3)).toBe(0.5);
	});
});

// ---------------------------------------------------------------------------
// 3. Threshold filter — fade/remove below MIN_VISIBLE
// ---------------------------------------------------------------------------

describe('isVisible / MIN_VISIBLE threshold', () => {
	it('MIN_VISIBLE is exactly 0.05', () => {
		expect(MIN_VISIBLE).toBe(0.05);
	});

	it('returns true at exactly the threshold (inclusive floor)', () => {
		expect(isVisible(0.05)).toBe(true);
	});

	it('returns false just below the threshold', () => {
		expect(isVisible(0.0499)).toBe(false);
	});

	it('returns false for zero / negative / NaN', () => {
		expect(isVisible(0)).toBe(false);
		expect(isVisible(-0.1)).toBe(false);
		expect(isVisible(Number.NaN)).toBe(false);
	});

	it('returns true for typical full-activation source', () => {
		expect(isVisible(1)).toBe(true);
	});
});

describe('ticksUntilInvisible', () => {
	it('returns 0 when input is already at/below MIN_VISIBLE', () => {
		expect(ticksUntilInvisible(MIN_VISIBLE)).toBe(0);
		expect(ticksUntilInvisible(0.03)).toBe(0);
		expect(ticksUntilInvisible(0)).toBe(0);
	});

	it('produces a count that actually crosses the threshold', () => {
		const n = ticksUntilInvisible(1);
		expect(n).toBeGreaterThan(0);
		// After n ticks we should be BELOW the threshold...
		expect(compoundDecay(1, n)).toBeLessThan(MIN_VISIBLE);
		// ...but one fewer tick should still be visible.
		expect(compoundDecay(1, n - 1)).toBeGreaterThanOrEqual(MIN_VISIBLE);
	});

	it('takes ~42 ticks for a full-strength burst to fade to threshold', () => {
		// log(0.05) / log(0.93) ≈ 41.27 → ceil → 42
		expect(ticksUntilInvisible(1)).toBe(42);
	});
});

// ---------------------------------------------------------------------------
// 4. Ring placement
// ---------------------------------------------------------------------------

describe('computeRing', () => {
	it('ranks 0..7 land on ring 1', () => {
		for (let r = 0; r < RING_1_CAPACITY; r++) {
			expect(computeRing(r)).toBe(1);
		}
	});

	it('rank 8 and beyond land on ring 2', () => {
		expect(computeRing(RING_1_CAPACITY)).toBe(2);
		expect(computeRing(15)).toBe(2);
		expect(computeRing(99)).toBe(2);
	});
});

describe('ringPositions (concentric circle layout)', () => {
	it('returns an empty array for count 0', () => {
		expect(ringPositions(0, 0, 0, 1)).toEqual([]);
	});

	it('places 4 nodes on ring 1 at radius RING_GAP, evenly spaced', () => {
		const pts = ringPositions(0, 0, 4, 1, 0);
		expect(pts).toHaveLength(4);
		// First point at angle 0 → (RING_GAP, 0)
		expect(pts[0].x).toBeCloseTo(RING_GAP, 6);
		expect(pts[0].y).toBeCloseTo(0, 6);
		// Every point sits on the circle of the correct radius.
		for (const p of pts) {
			const dist = Math.hypot(p.x, p.y);
			expect(dist).toBeCloseTo(RING_GAP, 6);
		}
	});

	it('places ring 2 at 2× RING_GAP from center', () => {
		const pts = ringPositions(0, 0, 3, 2, 0);
		for (const p of pts) {
			expect(Math.hypot(p.x, p.y)).toBeCloseTo(RING_GAP * 2, 6);
		}
	});

	it('honours the center (cx, cy)', () => {
		const pts = ringPositions(500, 280, 2, 1, 0);
		// With angleOffset=0 and 2 points, the two angles are 0 and π.
		expect(pts[0].x).toBeCloseTo(500 + RING_GAP, 6);
		expect(pts[0].y).toBeCloseTo(280, 6);
		expect(pts[1].x).toBeCloseTo(500 - RING_GAP, 6);
		expect(pts[1].y).toBeCloseTo(280, 6);
	});

	it('applies angleOffset to every point', () => {
		const unrot = ringPositions(0, 0, 3, 1, 0);
		const rot = ringPositions(0, 0, 3, 1, Math.PI / 2);
		for (let i = 0; i < 3; i++) {
			// Rotation preserves distance from center.
			expect(Math.hypot(rot[i].x, rot[i].y)).toBeCloseTo(
				Math.hypot(unrot[i].x, unrot[i].y),
				6,
			);
		}
		// And the first rotated point should now be near (0, RING_GAP) rather
		// than (RING_GAP, 0).
		expect(rot[0].x).toBeCloseTo(0, 6);
		expect(rot[0].y).toBeCloseTo(RING_GAP, 6);
	});
});

describe('layoutNeighbours (spills overflow to ring 2)', () => {
	it('returns one point per neighbour', () => {
		expect(layoutNeighbours(0, 0, 15, 0)).toHaveLength(15);
		expect(layoutNeighbours(0, 0, 3, 0)).toHaveLength(3);
		expect(layoutNeighbours(0, 0, 0, 0)).toHaveLength(0);
	});

	it('first 8 neighbours are on ring 1 (radius RING_GAP)', () => {
		const pts = layoutNeighbours(0, 0, 15, 0);
		for (let i = 0; i < RING_1_CAPACITY; i++) {
			expect(Math.hypot(pts[i].x, pts[i].y)).toBeCloseTo(RING_GAP, 6);
		}
	});

	it('neighbour 9..N are on ring 2 (radius 2*RING_GAP)', () => {
		const pts = layoutNeighbours(0, 0, 15, 0);
		for (let i = RING_1_CAPACITY; i < 15; i++) {
			expect(Math.hypot(pts[i].x, pts[i].y)).toBeCloseTo(RING_GAP * 2, 6);
		}
	});
});

describe('initialActivation', () => {
	it('rank 0 gets the highest activation', () => {
		const a0 = initialActivation(0, 10);
		const a1 = initialActivation(1, 10);
		expect(a0).toBeGreaterThan(a1);
	});

	it('ring-2 ranks get a 0.75 ring penalty', () => {
		// Rank 7 (last of ring 1) vs rank 8 (first of ring 2) — the jump in
		// activation between them should include the 0.75 ring factor.
		const ring1Last = initialActivation(7, 16);
		const ring2First = initialActivation(8, 16);
		expect(ring2First).toBeLessThan(ring1Last * 0.78);
	});

	it('returns values in (0, 1]', () => {
		for (let i = 0; i < 20; i++) {
			const a = initialActivation(i, 20);
			expect(a).toBeGreaterThan(0);
			expect(a).toBeLessThanOrEqual(1);
		}
	});

	it('returns 0 for invalid inputs', () => {
		expect(initialActivation(-1, 10)).toBe(0);
		expect(initialActivation(0, 0)).toBe(0);
		expect(initialActivation(Number.NaN, 10)).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// 5. Color mapping
// ---------------------------------------------------------------------------

describe('activationColor', () => {
	it('source nodes always use SOURCE_COLOR (synapse-glow)', () => {
		expect(activationColor('fact', true)).toBe(SOURCE_COLOR);
		expect(activationColor('concept', true)).toBe(SOURCE_COLOR);
		// Even if nodeType is garbage, source overrides.
		expect(activationColor('garbage-type', true)).toBe(SOURCE_COLOR);
	});

	it('fact → NODE_TYPE_COLORS.fact (#00A8FF)', () => {
		expect(activationColor('fact', false)).toBe(NODE_TYPE_COLORS.fact);
		expect(activationColor('fact', false)).toBe('#00A8FF');
	});

	it('every known node type resolves to its palette entry', () => {
		for (const type of Object.keys(NODE_TYPE_COLORS)) {
			expect(activationColor(type, false)).toBe(NODE_TYPE_COLORS[type]);
		}
	});

	it('unknown node type falls back to FALLBACK_COLOR (soft steel)', () => {
		expect(activationColor('not-a-real-type', false)).toBe(FALLBACK_COLOR);
		expect(FALLBACK_COLOR).toBe('#8B95A5');
	});

	it('null/undefined/empty nodeType also falls back', () => {
		expect(activationColor(null, false)).toBe(FALLBACK_COLOR);
		expect(activationColor(undefined, false)).toBe(FALLBACK_COLOR);
		expect(activationColor('', false)).toBe(FALLBACK_COLOR);
	});
});

// ---------------------------------------------------------------------------
// 6. Staggered edge delay
// ---------------------------------------------------------------------------

describe('edgeStagger', () => {
	it('rank 0 has zero delay (first edge lights up immediately)', () => {
		expect(edgeStagger(0)).toBe(0);
	});

	it('ring-1 edges are STAGGER_PER_RANK apart', () => {
		expect(edgeStagger(1)).toBe(STAGGER_PER_RANK);
		expect(edgeStagger(2)).toBe(STAGGER_PER_RANK * 2);
		expect(edgeStagger(7)).toBe(STAGGER_PER_RANK * 7);
	});

	it('ring-2 edges add STAGGER_RING_2_BONUS on top of rank×stagger', () => {
		expect(edgeStagger(8)).toBe(8 * STAGGER_PER_RANK + STAGGER_RING_2_BONUS);
		expect(edgeStagger(12)).toBe(12 * STAGGER_PER_RANK + STAGGER_RING_2_BONUS);
	});

	it('monotonically non-decreasing', () => {
		let prev = -1;
		for (let i = 0; i < 20; i++) {
			const s = edgeStagger(i);
			expect(s).toBeGreaterThanOrEqual(prev);
			prev = s;
		}
	});

	it('produces 15 distinct delays for a typical 15-neighbour burst', () => {
		const delays = Array.from({ length: 15 }, (_, i) => edgeStagger(i));
		expect(new Set(delays).size).toBe(15);
	});
});

// ---------------------------------------------------------------------------
// 7. Event-feed filter
// ---------------------------------------------------------------------------

function spreadEvent(
	source_id: string,
	target_ids: string[],
): VestigeEvent {
	return { type: 'ActivationSpread', data: { source_id, target_ids } };
}

describe('filterNewSpreadEvents', () => {
	it('returns [] on empty feed', () => {
		expect(filterNewSpreadEvents([], null)).toEqual([]);
	});

	it('returns all ActivationSpread payloads when lastSeen is null', () => {
		const feed = [
			spreadEvent('a', ['b', 'c']),
			spreadEvent('d', ['e']),
		];
		const out = filterNewSpreadEvents(feed, null);
		expect(out).toHaveLength(2);
	});

	it('returns in oldest-first order (feed itself is newest-first)', () => {
		const newest = spreadEvent('new', ['n1']);
		const older = spreadEvent('old', ['o1']);
		const out = filterNewSpreadEvents([newest, older], null);
		expect(out[0].source_id).toBe('old');
		expect(out[1].source_id).toBe('new');
	});

	it('stops at the lastSeen reference (object identity)', () => {
		const oldest = spreadEvent('o', ['x']);
		const middle = spreadEvent('m', ['y']);
		const newest = spreadEvent('n', ['z']);
		// Feed is prepended, so order is [newest, middle, oldest]
		const feed = [newest, middle, oldest];
		const out = filterNewSpreadEvents(feed, middle);
		// Only `newest` is fresh — middle and oldest were already processed.
		expect(out).toHaveLength(1);
		expect(out[0].source_id).toBe('n');
	});

	it('returns [] if lastSeen is already the newest event', () => {
		const e = spreadEvent('a', ['b']);
		const out = filterNewSpreadEvents([e], e);
		expect(out).toEqual([]);
	});

	it('ignores non-ActivationSpread events', () => {
		const feed: VestigeEvent[] = [
			{ type: 'MemoryCreated', data: { id: 'x' } },
			spreadEvent('a', ['b']),
			{ type: 'Heartbeat', data: {} },
		];
		const out = filterNewSpreadEvents(feed, null);
		expect(out).toHaveLength(1);
		expect(out[0].source_id).toBe('a');
	});

	it('skips malformed ActivationSpread events (missing / wrong-type fields)', () => {
		const feed: VestigeEvent[] = [
			{ type: 'ActivationSpread', data: {} }, // missing both
			{ type: 'ActivationSpread', data: { source_id: 'a' } }, // no targets
			{ type: 'ActivationSpread', data: { target_ids: ['b'] } }, // no source
			{
				type: 'ActivationSpread',
				data: { source_id: 'a', target_ids: 'not-an-array' },
			},
			{
				type: 'ActivationSpread',
				data: { source_id: 'a', target_ids: [123, null, 'x'] },
			},
		];
		const out = filterNewSpreadEvents(feed, null);
		// Only the last one survives, with numeric/null targets filtered out.
		expect(out).toHaveLength(1);
		expect(out[0].source_id).toBe('a');
		expect(out[0].target_ids).toEqual(['x']);
	});

	it('preserves target array contents faithfully', () => {
		const feed = [spreadEvent('src', ['t1', 't2', 't3'])];
		const out = filterNewSpreadEvents(feed, null);
		expect(out[0].target_ids).toEqual(['t1', 't2', 't3']);
	});

	it('does not mutate its inputs', () => {
		const feed = [spreadEvent('a', ['b', 'c'])];
		const snapshot = JSON.stringify(feed);
		filterNewSpreadEvents(feed, null);
		expect(JSON.stringify(feed)).toBe(snapshot);
	});
});

// ---------------------------------------------------------------------------
// Sanity: exported constants are the values the docstring promises
// ---------------------------------------------------------------------------

describe('exported constants (contract pinning)', () => {
	it('RING_1_CAPACITY is 8', () => {
		expect(RING_1_CAPACITY).toBe(8);
	});

	it('STAGGER_PER_RANK is 4 frames', () => {
		expect(STAGGER_PER_RANK).toBe(4);
	});

	it('STAGGER_RING_2_BONUS is 12 frames', () => {
		expect(STAGGER_RING_2_BONUS).toBe(12);
	});

	it('RING_GAP is 140px', () => {
		expect(RING_GAP).toBe(140);
	});

	it('SOURCE_COLOR is synapse-glow #818cf8', () => {
		expect(SOURCE_COLOR).toBe('#818cf8');
	});
});

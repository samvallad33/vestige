/**
 * MemoryAuditTrail — pure helper coverage.
 *
 * Runs in vitest's Node environment (no jsdom). Every assertion exercises
 * a function in `audit-trail-helpers.ts` with fully deterministic inputs.
 */
import { describe, it, expect } from 'vitest';

import {
	ALL_ACTIONS,
	META,
	VISIBLE_LIMIT,
	formatRetentionDelta,
	generateMockAuditTrail,
	hashSeed,
	makeRand,
	relativeTime,
	splitVisible,
	type AuditAction,
	type AuditEvent
} from '../audit-trail-helpers';

// Fixed reference point for all time-based tests. Millisecond precision so
// relative-time maths are exact, not drifting with wallclock time.
const NOW = Date.UTC(2026, 3, 20, 12, 0, 0); // 2026-04-20 12:00:00 UTC

// ---------------------------------------------------------------------------
// hashSeed + makeRand
// ---------------------------------------------------------------------------
describe('hashSeed', () => {
	it('is deterministic', () => {
		expect(hashSeed('abc')).toBe(hashSeed('abc'));
		expect(hashSeed('memory-42')).toBe(hashSeed('memory-42'));
	});

	it('different ids hash to different seeds', () => {
		expect(hashSeed('a')).not.toBe(hashSeed('b'));
		expect(hashSeed('memory-1')).not.toBe(hashSeed('memory-2'));
	});

	it('empty string hashes to 0', () => {
		expect(hashSeed('')).toBe(0);
	});

	it('returns an unsigned 32-bit integer', () => {
		// Stress: a long id should never produce a negative or non-integer seed.
		const seed = hashSeed('a'.repeat(256));
		expect(Number.isInteger(seed)).toBe(true);
		expect(seed).toBeGreaterThanOrEqual(0);
		expect(seed).toBeLessThan(2 ** 32);
	});
});

describe('makeRand', () => {
	it('is deterministic given the same seed', () => {
		const a = makeRand(42);
		const b = makeRand(42);
		for (let i = 0; i < 20; i++) expect(a()).toBe(b());
	});

	it('produces values strictly in [0, 1)', () => {
		// Seed with UINT32_MAX to force the edge case that exposed the original
		// `/ 0xffffffff` bug — the divisor must be 2^32, not 2^32 - 1.
		const rand = makeRand(0xffffffff);
		for (let i = 0; i < 5000; i++) {
			const v = rand();
			expect(v).toBeGreaterThanOrEqual(0);
			expect(v).toBeLessThan(1);
		}
	});

	it('different seeds produce different sequences', () => {
		const a = makeRand(1);
		const b = makeRand(2);
		expect(a()).not.toBe(b());
	});
});

// ---------------------------------------------------------------------------
// Deterministic generator
// ---------------------------------------------------------------------------
describe('generateMockAuditTrail — determinism', () => {
	it('same id + same now always yields the same sequence', () => {
		const a = generateMockAuditTrail('memory-xyz', NOW);
		const b = generateMockAuditTrail('memory-xyz', NOW);
		expect(a).toEqual(b);
	});

	it('different ids yield different sequences', () => {
		const a = generateMockAuditTrail('memory-a', NOW);
		const b = generateMockAuditTrail('memory-b', NOW);
		// Either different lengths or different event-by-event — anything but equal.
		expect(a).not.toEqual(b);
	});

	it('empty id yields no events — the panel should never fabricate history', () => {
		expect(generateMockAuditTrail('', NOW)).toEqual([]);
	});

	it('count fits the default 8-15 range', () => {
		// Sample a handful of ids — the distribution should stay in range.
		for (const id of ['a', 'abc', 'memory-1', 'memory-2', 'memory-3', 'x'.repeat(50)]) {
			const events = generateMockAuditTrail(id, NOW);
			expect(events.length).toBeGreaterThanOrEqual(8);
			expect(events.length).toBeLessThanOrEqual(15);
		}
	});

	it('first emitted event (newest-first order → last in array) is "created"', () => {
		const events = generateMockAuditTrail('deterministic-id', NOW);
		expect(events[events.length - 1].action).toBe('created');
		expect(events[events.length - 1].triggered_by).toBe('smart_ingest');
	});

	it('emits events in newest-first order', () => {
		const events = generateMockAuditTrail('order-check', NOW);
		for (let i = 1; i < events.length; i++) {
			const prev = new Date(events[i - 1].timestamp).getTime();
			const curr = new Date(events[i].timestamp).getTime();
			expect(prev).toBeGreaterThanOrEqual(curr);
		}
	});

	it('all timestamps are valid ISO strings in the past relative to NOW', () => {
		const events = generateMockAuditTrail('iso-check', NOW);
		for (const ev of events) {
			const t = new Date(ev.timestamp).getTime();
			expect(Number.isFinite(t)).toBe(true);
			expect(t).toBeLessThanOrEqual(NOW);
		}
	});

	it('respects countOverride — 16 events crosses the visibility threshold', () => {
		const events = generateMockAuditTrail('big', NOW, 16);
		expect(events).toHaveLength(16);
	});

	it('retention values never escape [0, 1]', () => {
		for (const id of ['x', 'y', 'z', 'memory-big']) {
			const events = generateMockAuditTrail(id, NOW, 30);
			for (const ev of events) {
				if (ev.old_value !== undefined) {
					expect(ev.old_value).toBeGreaterThanOrEqual(0);
					expect(ev.old_value).toBeLessThanOrEqual(1);
				}
				if (ev.new_value !== undefined) {
					expect(ev.new_value).toBeGreaterThanOrEqual(0);
					expect(ev.new_value).toBeLessThanOrEqual(1);
				}
			}
		}
	});
});

// ---------------------------------------------------------------------------
// Relative time
// ---------------------------------------------------------------------------
describe('relativeTime — boundary cases', () => {
	// Build an ISO timestamp `offsetMs` before NOW.
	const ago = (offsetMs: number) => new Date(NOW - offsetMs).toISOString();

	const cases: Array<[string, number, string]> = [
		['0s ago', 0, '0s ago'],
		['59s ago', 59 * 1000, '59s ago'],
		['60s flips to 1m', 60 * 1000, '1m ago'],
		['59m ago', 59 * 60 * 1000, '59m ago'],
		['60m flips to 1h', 60 * 60 * 1000, '1h ago'],
		['23h ago', 23 * 3600 * 1000, '23h ago'],
		['24h flips to 1d', 24 * 3600 * 1000, '1d ago'],
		['6d ago', 6 * 86400 * 1000, '6d ago'],
		['7d ago', 7 * 86400 * 1000, '7d ago'],
		['29d ago', 29 * 86400 * 1000, '29d ago'],
		['30d flips to 1mo', 30 * 86400 * 1000, '1mo ago'],
		['365d → 12mo flips to 1y', 365 * 86400 * 1000, '1y ago']
	];

	for (const [name, offset, expected] of cases) {
		it(name, () => {
			expect(relativeTime(ago(offset), NOW)).toBe(expected);
		});
	}

	it('future timestamps clamp to "0s ago"', () => {
		const future = new Date(NOW + 60_000).toISOString();
		expect(relativeTime(future, NOW)).toBe('0s ago');
	});
});

// ---------------------------------------------------------------------------
// Event type → marker mapping
// ---------------------------------------------------------------------------
describe('META — action to marker mapping', () => {
	it('covers all 8 audit actions exactly', () => {
		expect(Object.keys(META).sort()).toEqual([...ALL_ACTIONS].sort());
		expect(ALL_ACTIONS).toHaveLength(8);
	});

	it('every action has a distinct marker kind (8 kinds → 8 glyph shapes)', () => {
		const kinds = ALL_ACTIONS.map((a) => META[a].kind);
		expect(new Set(kinds).size).toBe(8);
	});

	it('every action has a non-empty label and hex color', () => {
		for (const action of ALL_ACTIONS) {
			const m = META[action];
			expect(m.label.length).toBeGreaterThan(0);
			expect(m.color).toMatch(/^#[0-9a-f]{6}$/i);
		}
	});
});

// ---------------------------------------------------------------------------
// Retention delta formatter
// ---------------------------------------------------------------------------
describe('formatRetentionDelta', () => {
	it('returns null when both values are missing', () => {
		expect(formatRetentionDelta(undefined, undefined)).toBeNull();
	});

	it('returns "set X.XX" when only new is defined', () => {
		expect(formatRetentionDelta(undefined, 0.5)).toBe('set 0.50');
		// Note: toFixed(2) uses float-to-string half-to-even; assert on values
		// that round unambiguously rather than on IEEE-754 tie edges.
		expect(formatRetentionDelta(undefined, 0.736)).toBe('set 0.74');
	});

	it('returns "was X.XX" when only old is defined', () => {
		expect(formatRetentionDelta(0.5, undefined)).toBe('was 0.50');
	});

	it('returns "old → new" when both are defined', () => {
		expect(formatRetentionDelta(0.5, 0.7)).toBe('0.50 → 0.70');
		expect(formatRetentionDelta(0.72, 0.85)).toBe('0.72 → 0.85');
	});

	it('handles descending deltas without changing the arrow', () => {
		// Suppression / demotion paths — old > new.
		expect(formatRetentionDelta(0.8, 0.6)).toBe('0.80 → 0.60');
	});

	it('rejects non-finite numbers', () => {
		expect(formatRetentionDelta(NaN, 0.5)).toBe('set 0.50');
		expect(formatRetentionDelta(0.5, NaN)).toBe('was 0.50');
		expect(formatRetentionDelta(NaN, NaN)).toBeNull();
	});
});

// ---------------------------------------------------------------------------
// splitVisible — 15-event cap
// ---------------------------------------------------------------------------
describe('splitVisible — collapse threshold', () => {
	const makeEvents = (n: number): AuditEvent[] =>
		Array.from({ length: n }, (_, i) => ({
			action: 'accessed' as AuditAction,
			timestamp: new Date(NOW - i * 60_000).toISOString()
		}));

	it('VISIBLE_LIMIT is 15', () => {
		expect(VISIBLE_LIMIT).toBe(15);
	});

	it('exactly 15 events → no toggle (hiddenCount 0)', () => {
		const { visible, hiddenCount } = splitVisible(makeEvents(15), false);
		expect(visible).toHaveLength(15);
		expect(hiddenCount).toBe(0);
	});

	it('14 events → no toggle', () => {
		const { visible, hiddenCount } = splitVisible(makeEvents(14), false);
		expect(visible).toHaveLength(14);
		expect(hiddenCount).toBe(0);
	});

	it('16 events collapsed → visible 15, hidden 1', () => {
		const { visible, hiddenCount } = splitVisible(makeEvents(16), false);
		expect(visible).toHaveLength(15);
		expect(hiddenCount).toBe(1);
	});

	it('16 events expanded → visible 16, hidden reports overflow count (1)', () => {
		const { visible, hiddenCount } = splitVisible(makeEvents(16), true);
		expect(visible).toHaveLength(16);
		expect(hiddenCount).toBe(1);
	});

	it('0 events → visible empty, hidden 0', () => {
		const { visible, hiddenCount } = splitVisible(makeEvents(0), false);
		expect(visible).toHaveLength(0);
		expect(hiddenCount).toBe(0);
	});

	it('preserves newest-first order when truncating', () => {
		const events = makeEvents(20);
		const { visible } = splitVisible(events, false);
		expect(visible[0]).toBe(events[0]);
		expect(visible[14]).toBe(events[14]);
	});
});

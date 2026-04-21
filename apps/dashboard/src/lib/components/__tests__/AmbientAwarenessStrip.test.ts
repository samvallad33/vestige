import { describe, it, expect } from 'vitest';
import {
	ACTIVITY_BUCKET_COUNT,
	ACTIVITY_BUCKET_MS,
	ACTIVITY_WINDOW_MS,
	bucketizeActivity,
	dreamInsightsCount,
	findRecentDream,
	formatAgo,
	hasRecentSuppression,
	isDreaming,
	parseEventTimestamp,
	type EventLike,
} from '../awareness-helpers';

// Fixed "now" — March 1 2026 12:00:00 UTC. All tests are clock-free.
const NOW = Date.parse('2026-03-01T12:00:00.000Z');

function mkEvent(
	type: string,
	data: Record<string, unknown> = {},
): EventLike {
	return { type, data };
}

// ─────────────────────────────────────────────────────────────────────────
// parseEventTimestamp
// ─────────────────────────────────────────────────────────────────────────
describe('parseEventTimestamp', () => {
	it('parses ISO-8601 string', () => {
		const e = mkEvent('Foo', { timestamp: '2026-03-01T12:00:00.000Z' });
		expect(parseEventTimestamp(e)).toBe(NOW);
	});

	it('parses numeric ms (> 1e12)', () => {
		const e = mkEvent('Foo', { timestamp: NOW });
		expect(parseEventTimestamp(e)).toBe(NOW);
	});

	it('parses numeric seconds (<= 1e12) by scaling x1000', () => {
		const secs = Math.floor(NOW / 1000);
		const e = mkEvent('Foo', { timestamp: secs });
		// Allow floating precision — must land in same second
		const result = parseEventTimestamp(e);
		expect(result).not.toBeNull();
		expect(Math.abs((result as number) - NOW)).toBeLessThan(1000);
	});

	it('falls back to `at` field', () => {
		const e = mkEvent('Foo', { at: '2026-03-01T12:00:00.000Z' });
		expect(parseEventTimestamp(e)).toBe(NOW);
	});

	it('falls back to `occurred_at` field', () => {
		const e = mkEvent('Foo', { occurred_at: '2026-03-01T12:00:00.000Z' });
		expect(parseEventTimestamp(e)).toBe(NOW);
	});

	it('prefers `timestamp` over `at` over `occurred_at`', () => {
		const e = mkEvent('Foo', {
			timestamp: '2026-03-01T12:00:00.000Z',
			at: '2020-01-01T00:00:00.000Z',
			occurred_at: '2019-01-01T00:00:00.000Z',
		});
		expect(parseEventTimestamp(e)).toBe(NOW);
	});

	it('returns null for missing data', () => {
		expect(parseEventTimestamp({ type: 'Foo' })).toBeNull();
	});

	it('returns null for empty data object', () => {
		expect(parseEventTimestamp(mkEvent('Foo', {}))).toBeNull();
	});

	it('returns null for bad ISO string', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: 'not-a-date' }))).toBeNull();
	});

	it('returns null for non-finite number (NaN)', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: Number.NaN }))).toBeNull();
	});

	it('returns null for non-finite number (Infinity)', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: Number.POSITIVE_INFINITY }))).toBeNull();
	});

	it('returns null for null timestamp', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: null as unknown as string }))).toBeNull();
	});

	it('returns null for non-string non-number timestamp (object)', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: {} as unknown as string }))).toBeNull();
	});

	it('returns null for a boolean timestamp', () => {
		expect(parseEventTimestamp(mkEvent('Foo', { timestamp: true as unknown as string }))).toBeNull();
	});
});

// ─────────────────────────────────────────────────────────────────────────
// bucketizeActivity
// ─────────────────────────────────────────────────────────────────────────
describe('bucketizeActivity', () => {
	it('returns 10 buckets of 30s each covering a 5-min window', () => {
		expect(ACTIVITY_BUCKET_COUNT).toBe(10);
		expect(ACTIVITY_BUCKET_MS).toBe(30_000);
		expect(ACTIVITY_WINDOW_MS).toBe(300_000);
		const result = bucketizeActivity([], NOW);
		expect(result).toHaveLength(10);
		expect(result.every((b) => b.count === 0 && b.ratio === 0)).toBe(true);
	});

	it('assigns newest event to the last bucket (index 9)', () => {
		const e = mkEvent('MemoryCreated', { timestamp: NOW - 100 });
		const result = bucketizeActivity([e], NOW);
		expect(result[9].count).toBe(1);
		expect(result[9].ratio).toBe(1);
		for (let i = 0; i < 9; i++) expect(result[i].count).toBe(0);
	});

	it('assigns oldest-edge event to bucket 0', () => {
		// Exactly 5 min ago → at start boundary → floor((0)/30s) = 0
		const e = mkEvent('MemoryCreated', { timestamp: NOW - ACTIVITY_WINDOW_MS + 1 });
		const result = bucketizeActivity([e], NOW);
		expect(result[0].count).toBe(1);
	});

	it('drops events older than 5 min (clock skew / pre-history)', () => {
		const e = mkEvent('MemoryCreated', { timestamp: NOW - ACTIVITY_WINDOW_MS - 1 });
		const result = bucketizeActivity([e], NOW);
		expect(result.every((b) => b.count === 0)).toBe(true);
	});

	it('drops future events (negative clock skew)', () => {
		const e = mkEvent('MemoryCreated', { timestamp: NOW + 5_000 });
		const result = bucketizeActivity([e], NOW);
		expect(result.every((b) => b.count === 0)).toBe(true);
	});

	it('drops Heartbeat events as noise', () => {
		const e = mkEvent('Heartbeat', { timestamp: NOW - 100 });
		const result = bucketizeActivity([e], NOW);
		expect(result.every((b) => b.count === 0)).toBe(true);
	});

	it('drops events with unparseable timestamps', () => {
		const e = mkEvent('MemoryCreated', { timestamp: 'garbage' });
		const result = bucketizeActivity([e], NOW);
		expect(result.every((b) => b.count === 0)).toBe(true);
	});

	it('distributes events across buckets and computes correct ratios', () => {
		const events = [
			// Bucket 9 (newest 30s): 3 events
			mkEvent('MemoryCreated', { timestamp: NOW - 5_000 }),
			mkEvent('MemoryCreated', { timestamp: NOW - 10_000 }),
			mkEvent('MemoryCreated', { timestamp: NOW - 15_000 }),
			// Bucket 8: 1 event (31s - 60s ago)
			mkEvent('MemoryCreated', { timestamp: NOW - 35_000 }),
			// Bucket 0 (oldest): 1 event (270s - 300s ago)
			mkEvent('MemoryCreated', { timestamp: NOW - 290_000 }),
		];
		const result = bucketizeActivity(events, NOW);
		expect(result[9].count).toBe(3);
		expect(result[8].count).toBe(1);
		expect(result[0].count).toBe(1);
		expect(result[9].ratio).toBe(1);
		expect(result[8].ratio).toBeCloseTo(1 / 3, 5);
		expect(result[0].ratio).toBeCloseTo(1 / 3, 5);
	});

	it('handles events with numeric ms timestamp', () => {
		const e = { type: 'MemoryCreated', data: { timestamp: NOW - 10_000 } };
		const result = bucketizeActivity([e], NOW);
		expect(result[9].count).toBe(1);
	});

	it('works with a mixed real-world feed (200 events, some stale)', () => {
		const events: EventLike[] = [];
		for (let i = 0; i < 200; i++) {
			const offset = i * 3_000; // one every 3s, oldest first
			events.unshift(mkEvent('MemoryCreated', { timestamp: NOW - offset }));
		}
		// add 10 Heartbeats mid-stream
		for (let i = 0; i < 10; i++) {
			events.push(mkEvent('Heartbeat', { timestamp: NOW - i * 1_000 }));
		}
		const result = bucketizeActivity(events, NOW);
		// 101 events fit in the [now-300s, now] window: offsets 0, 3s, 6s, …, 300s.
		// Heartbeats excluded. Sum should be exactly 101.
		const total = result.reduce((s, b) => s + b.count, 0);
		expect(total).toBe(101);
	});
});

// ─────────────────────────────────────────────────────────────────────────
// findRecentDream
// ─────────────────────────────────────────────────────────────────────────
describe('findRecentDream', () => {
	it('returns null on empty feed', () => {
		expect(findRecentDream([], NOW)).toBeNull();
	});

	it('returns null when no DreamCompleted in feed', () => {
		const feed = [
			mkEvent('MemoryCreated', { timestamp: NOW - 1000 }),
			mkEvent('DreamStarted', { timestamp: NOW - 500 }),
		];
		expect(findRecentDream(feed, NOW)).toBeNull();
	});

	it('returns the newest DreamCompleted within 24h', () => {
		const fresh = mkEvent('DreamCompleted', {
			timestamp: NOW - 60_000,
			insights_generated: 7,
		});
		const stale = mkEvent('DreamCompleted', {
			timestamp: NOW - 2 * 24 * 60 * 60 * 1000,
		});
		// Feed is newest-first
		const result = findRecentDream([fresh, stale], NOW);
		expect(result).toBe(fresh);
	});

	it('returns null when only DreamCompleted is older than 24h', () => {
		const stale = mkEvent('DreamCompleted', {
			timestamp: NOW - 25 * 60 * 60 * 1000,
		});
		expect(findRecentDream([stale], NOW)).toBeNull();
	});

	it('exactly 24h ago still counts (inclusive)', () => {
		const edge = mkEvent('DreamCompleted', {
			timestamp: NOW - 24 * 60 * 60 * 1000,
		});
		expect(findRecentDream([edge], NOW)).toBe(edge);
	});

	it('stops at first DreamCompleted in newest-first feed', () => {
		const newest = mkEvent('DreamCompleted', { timestamp: NOW - 1_000 });
		const older = mkEvent('DreamCompleted', { timestamp: NOW - 60_000 });
		expect(findRecentDream([newest, older], NOW)).toBe(newest);
	});

	it('falls back to nowMs for unparseable timestamps (treated as recent)', () => {
		const e = mkEvent('DreamCompleted', { timestamp: 'bad' });
		expect(findRecentDream([e], NOW)).toBe(e);
	});
});

// ─────────────────────────────────────────────────────────────────────────
// dreamInsightsCount
// ─────────────────────────────────────────────────────────────────────────
describe('dreamInsightsCount', () => {
	it('returns null for null input', () => {
		expect(dreamInsightsCount(null)).toBeNull();
	});

	it('returns null when missing', () => {
		expect(dreamInsightsCount(mkEvent('DreamCompleted', {}))).toBeNull();
	});

	it('reads insights_generated (snake_case)', () => {
		expect(
			dreamInsightsCount(mkEvent('DreamCompleted', { insights_generated: 5 })),
		).toBe(5);
	});

	it('reads insightsGenerated (camelCase)', () => {
		expect(
			dreamInsightsCount(mkEvent('DreamCompleted', { insightsGenerated: 3 })),
		).toBe(3);
	});

	it('prefers snake_case when both present', () => {
		expect(
			dreamInsightsCount(
				mkEvent('DreamCompleted', { insights_generated: 7, insightsGenerated: 99 }),
			),
		).toBe(7);
	});

	it('returns null for non-numeric value', () => {
		expect(
			dreamInsightsCount(mkEvent('DreamCompleted', { insights_generated: 'seven' as unknown as number })),
		).toBeNull();
	});
});

// ─────────────────────────────────────────────────────────────────────────
// isDreaming
// ─────────────────────────────────────────────────────────────────────────
describe('isDreaming', () => {
	it('returns false for empty feed', () => {
		expect(isDreaming([], NOW)).toBe(false);
	});

	it('returns false when no DreamStarted in feed', () => {
		expect(isDreaming([mkEvent('MemoryCreated', { timestamp: NOW })], NOW)).toBe(false);
	});

	it('returns true for DreamStarted in last 5 min with no DreamCompleted', () => {
		const feed = [mkEvent('DreamStarted', { timestamp: NOW - 60_000 })];
		expect(isDreaming(feed, NOW)).toBe(true);
	});

	it('returns false for DreamStarted older than 5 min with no DreamCompleted', () => {
		const feed = [mkEvent('DreamStarted', { timestamp: NOW - 6 * 60 * 1000 })];
		expect(isDreaming(feed, NOW)).toBe(false);
	});

	it('returns false when DreamCompleted newer than DreamStarted', () => {
		// Feed is newest-first: completed, then started
		const feed = [
			mkEvent('DreamCompleted', { timestamp: NOW - 30_000 }),
			mkEvent('DreamStarted', { timestamp: NOW - 60_000 }),
		];
		expect(isDreaming(feed, NOW)).toBe(false);
	});

	it('returns true when DreamCompleted is OLDER than DreamStarted (new cycle began)', () => {
		// Newest-first: started is newer, and there's an older completed from a prior cycle
		const feed = [
			mkEvent('DreamStarted', { timestamp: NOW - 30_000 }),
			mkEvent('DreamCompleted', { timestamp: NOW - 10 * 60 * 1000 }),
		];
		expect(isDreaming(feed, NOW)).toBe(true);
	});

	it('boundary: DreamStarted exactly 5 min ago → still dreaming (>= check)', () => {
		const feed = [mkEvent('DreamStarted', { timestamp: NOW - 5 * 60 * 1000 })];
		expect(isDreaming(feed, NOW)).toBe(true);
	});

	it('only considers FIRST DreamStarted / FIRST DreamCompleted (newest-first semantics)', () => {
		const feed = [
			mkEvent('DreamStarted', { timestamp: NOW - 10_000 }),
			mkEvent('DreamCompleted', { timestamp: NOW - 20_000 }), // older — prior cycle
			mkEvent('DreamStarted', { timestamp: NOW - 30_000 }), // ignored
		];
		expect(isDreaming(feed, NOW)).toBe(true);
	});

	it('unparseable DreamStarted timestamp falls back to nowMs (counts as dreaming)', () => {
		const feed = [mkEvent('DreamStarted', { timestamp: 'bad' })];
		expect(isDreaming(feed, NOW)).toBe(true);
	});
});

// ─────────────────────────────────────────────────────────────────────────
// hasRecentSuppression
// ─────────────────────────────────────────────────────────────────────────
describe('hasRecentSuppression', () => {
	it('returns false for empty feed', () => {
		expect(hasRecentSuppression([], NOW)).toBe(false);
	});

	it('returns false when no MemorySuppressed in feed', () => {
		const feed = [
			mkEvent('MemoryCreated', { timestamp: NOW }),
			mkEvent('DreamStarted', { timestamp: NOW }),
		];
		expect(hasRecentSuppression(feed, NOW)).toBe(false);
	});

	it('returns true for MemorySuppressed within 10s', () => {
		const feed = [mkEvent('MemorySuppressed', { timestamp: NOW - 5_000 })];
		expect(hasRecentSuppression(feed, NOW)).toBe(true);
	});

	it('returns false for MemorySuppressed older than 10s', () => {
		const feed = [mkEvent('MemorySuppressed', { timestamp: NOW - 11_000 })];
		expect(hasRecentSuppression(feed, NOW)).toBe(false);
	});

	it('respects custom threshold', () => {
		const feed = [mkEvent('MemorySuppressed', { timestamp: NOW - 8_000 })];
		expect(hasRecentSuppression(feed, NOW, 5_000)).toBe(false);
		expect(hasRecentSuppression(feed, NOW, 10_000)).toBe(true);
	});

	it('stops at first MemorySuppressed (newest-first short-circuit)', () => {
		const feed = [
			mkEvent('MemorySuppressed', { timestamp: NOW - 30_000 }), // first, outside window
			mkEvent('MemorySuppressed', { timestamp: NOW - 1_000 }), // inside, but never checked
		];
		expect(hasRecentSuppression(feed, NOW)).toBe(false);
	});

	it('boundary: exactly at threshold counts (>= check)', () => {
		const feed = [mkEvent('MemorySuppressed', { timestamp: NOW - 10_000 })];
		expect(hasRecentSuppression(feed, NOW, 10_000)).toBe(true);
	});

	it('unparseable timestamp falls back to nowMs (flash fires)', () => {
		const feed = [mkEvent('MemorySuppressed', { timestamp: 'bad' })];
		expect(hasRecentSuppression(feed, NOW)).toBe(true);
	});

	it('ignores non-MemorySuppressed events before finding one', () => {
		const feed = [
			mkEvent('MemoryCreated', { timestamp: NOW }),
			mkEvent('DreamStarted', { timestamp: NOW }),
			mkEvent('MemorySuppressed', { timestamp: NOW - 3_000 }),
		];
		expect(hasRecentSuppression(feed, NOW)).toBe(true);
	});
});

// ─────────────────────────────────────────────────────────────────────────
// formatAgo
// ─────────────────────────────────────────────────────────────────────────
describe('formatAgo', () => {
	it('formats seconds', () => {
		expect(formatAgo(5_000)).toBe('5s ago');
		expect(formatAgo(59_000)).toBe('59s ago');
		expect(formatAgo(0)).toBe('0s ago');
	});

	it('formats minutes', () => {
		expect(formatAgo(60_000)).toBe('1m ago');
		expect(formatAgo(59 * 60_000)).toBe('59m ago');
	});

	it('formats hours', () => {
		expect(formatAgo(60 * 60_000)).toBe('1h ago');
		expect(formatAgo(23 * 60 * 60_000)).toBe('23h ago');
	});

	it('formats days', () => {
		expect(formatAgo(24 * 60 * 60_000)).toBe('1d ago');
		expect(formatAgo(7 * 24 * 60 * 60_000)).toBe('7d ago');
	});

	it('clamps negative input to 0', () => {
		expect(formatAgo(-5_000)).toBe('0s ago');
	});
});

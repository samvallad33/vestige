/**
 * Tests for schedule / FSRS calendar helpers. These are the pure-logic core
 * of the `schedule` page + `FSRSCalendar.svelte` component — the Svelte
 * runtime is not exercised here (vitest runs `environment: node`, no jsdom).
 */
import { describe, it, expect } from 'vitest';
import type { Memory } from '$types';
import {
	MS_DAY,
	startOfDay,
	daysBetween,
	isoDate,
	classifyUrgency,
	daysUntilReview,
	weekBucketRange,
	avgRetention,
	gridCellPosition,
	gridStartForAnchor,
	computeScheduleStats,
} from '../schedule-helpers';

function makeMemory(overrides: Partial<Memory> = {}): Memory {
	return {
		id: 'm-' + Math.random().toString(36).slice(2, 8),
		content: 'test memory',
		nodeType: 'fact',
		tags: [],
		retentionStrength: 0.7,
		storageStrength: 0.5,
		retrievalStrength: 0.8,
		createdAt: '2026-01-01T00:00:00Z',
		updatedAt: '2026-01-01T00:00:00Z',
		...overrides,
	};
}

// Fixed anchor: 2026-04-20 12:00 local so offsets don't straddle midnight
// in the default test runner's tz. All relative timestamps are derived from
// this anchor to keep tests tz-independent.
function anchor(): Date {
	const d = new Date(2026, 3, 20, 12, 0, 0, 0); // Mon Apr 20 2026 12:00 local
	return d;
}

function offsetDays(base: Date, days: number, hour = 12): Date {
	const d = new Date(base);
	d.setDate(d.getDate() + days);
	d.setHours(hour, 0, 0, 0);
	return d;
}

describe('startOfDay', () => {
	it('zeros hours / minutes / seconds / ms', () => {
		const d = new Date(2026, 3, 20, 14, 35, 27, 999);
		const s = startOfDay(d);
		expect(s.getHours()).toBe(0);
		expect(s.getMinutes()).toBe(0);
		expect(s.getSeconds()).toBe(0);
		expect(s.getMilliseconds()).toBe(0);
		expect(s.getFullYear()).toBe(2026);
		expect(s.getMonth()).toBe(3);
		expect(s.getDate()).toBe(20);
	});

	it('does not mutate its input', () => {
		const input = new Date(2026, 3, 20, 14, 35);
		const before = input.getTime();
		startOfDay(input);
		expect(input.getTime()).toBe(before);
	});

	it('accepts an ISO string', () => {
		const s = startOfDay('2026-04-20T14:35:00');
		expect(s.getHours()).toBe(0);
	});
});

describe('daysBetween', () => {
	it('returns 0 for the same calendar day at different hours', () => {
		const a = new Date(2026, 3, 20, 0, 0);
		const b = new Date(2026, 3, 20, 23, 59);
		expect(daysBetween(a, b)).toBe(0);
		expect(daysBetween(b, a)).toBe(0);
	});

	it('returns positive for future, negative for past', () => {
		const today = anchor();
		expect(daysBetween(offsetDays(today, 3), today)).toBe(3);
		expect(daysBetween(offsetDays(today, -3), today)).toBe(-3);
	});

	it('is day-granular across the midnight boundary', () => {
		const midnight = new Date(2026, 3, 20, 0, 0, 0, 0);
		const justBefore = new Date(2026, 3, 19, 23, 59, 59, 999);
		expect(daysBetween(midnight, justBefore)).toBe(1);
	});
});

describe('isoDate', () => {
	it('formats as YYYY-MM-DD with zero-padding in LOCAL time', () => {
		expect(isoDate(new Date(2026, 0, 5))).toBe('2026-01-05'); // jan 5
		expect(isoDate(new Date(2026, 11, 31))).toBe('2026-12-31');
	});

	it('uses local day even for late-evening UTC-crossing timestamps', () => {
		// This is the whole reason isoDate uses get* not getUTC*: calendar cells
		// should match the user's perceived day.
		const d = new Date(2026, 3, 20, 23, 30); // apr 20 23:30 local
		expect(isoDate(d)).toBe('2026-04-20');
	});
});

describe('classifyUrgency', () => {
	const now = anchor();

	it('returns "none" for missing nextReviewAt', () => {
		expect(classifyUrgency(now, null)).toBe('none');
		expect(classifyUrgency(now, undefined)).toBe('none');
		expect(classifyUrgency(now, '')).toBe('none');
	});

	it('returns "none" for unparseable ISO strings', () => {
		expect(classifyUrgency(now, 'not-a-date')).toBe('none');
	});

	it('classifies overdue when due date is strictly before today', () => {
		expect(classifyUrgency(now, offsetDays(now, -1).toISOString())).toBe('overdue');
		expect(classifyUrgency(now, offsetDays(now, -5).toISOString())).toBe('overdue');
	});

	it('classifies today when due date is the same calendar day', () => {
		// Same day, earlier hour — still today, NOT overdue (day-granular).
		const earlier = new Date(now);
		earlier.setHours(3, 0);
		expect(classifyUrgency(now, earlier.toISOString())).toBe('today');
		const later = new Date(now);
		later.setHours(22, 0);
		expect(classifyUrgency(now, later.toISOString())).toBe('today');
	});

	it('classifies 1..=7 days out as "week"', () => {
		expect(classifyUrgency(now, offsetDays(now, 1).toISOString())).toBe('week');
		expect(classifyUrgency(now, offsetDays(now, 7).toISOString())).toBe('week');
	});

	it('classifies 8+ days out as "future"', () => {
		expect(classifyUrgency(now, offsetDays(now, 8).toISOString())).toBe('future');
		expect(classifyUrgency(now, offsetDays(now, 30).toISOString())).toBe('future');
	});

	it('boundary at midnight: 1 second after midnight tomorrow is "week" not "today"', () => {
		const tomorrowMidnight = startOfDay(offsetDays(now, 1, 0));
		tomorrowMidnight.setSeconds(1);
		expect(classifyUrgency(now, tomorrowMidnight.toISOString())).toBe('week');
	});
});

describe('daysUntilReview', () => {
	const now = anchor();

	it('returns null for missing / invalid input', () => {
		expect(daysUntilReview(now, null)).toBeNull();
		expect(daysUntilReview(now, undefined)).toBeNull();
		expect(daysUntilReview(now, 'garbage')).toBeNull();
	});

	it('returns 0 for today', () => {
		expect(daysUntilReview(now, now.toISOString())).toBe(0);
	});

	it('returns signed integer days', () => {
		expect(daysUntilReview(now, offsetDays(now, 5).toISOString())).toBe(5);
		expect(daysUntilReview(now, offsetDays(now, -3).toISOString())).toBe(-3);
	});
});

describe('weekBucketRange', () => {
	it('returns Sunday→Sunday exclusive for any weekday', () => {
		// Apr 20 2026 is a Monday. The week starts on Sunday Apr 19.
		const mon = new Date(2026, 3, 20, 14, 0);
		const { start, end } = weekBucketRange(mon);
		expect(start.getDay()).toBe(0); // Sunday
		expect(start.getDate()).toBe(19);
		expect(end.getDate()).toBe(26); // next Sunday
		expect(end.getTime() - start.getTime()).toBe(7 * MS_DAY);
	});

	it('for Sunday input, returns that same Sunday as start', () => {
		const sun = new Date(2026, 3, 19, 10, 0); // Sun Apr 19 2026
		const { start } = weekBucketRange(sun);
		expect(start.getDate()).toBe(19);
	});
});

describe('avgRetention', () => {
	it('returns 0 for empty array (no NaN)', () => {
		expect(avgRetention([])).toBe(0);
		expect(Number.isNaN(avgRetention([]))).toBe(false);
	});

	it('returns the single value for a length-1 list', () => {
		expect(avgRetention([makeMemory({ retentionStrength: 0.42 })])).toBeCloseTo(0.42);
	});

	it('returns the mean for a mixed list', () => {
		const ms = [
			makeMemory({ retentionStrength: 0.2 }),
			makeMemory({ retentionStrength: 0.8 }),
			makeMemory({ retentionStrength: 0.5 }),
		];
		expect(avgRetention(ms)).toBeCloseTo(0.5);
	});

	it('tolerates missing retentionStrength (treat as 0)', () => {
		const ms = [
			makeMemory({ retentionStrength: 1.0 }),
			makeMemory({ retentionStrength: undefined as unknown as number }),
		];
		expect(avgRetention(ms)).toBeCloseTo(0.5);
	});
});

describe('gridCellPosition', () => {
	it('maps row-major: index 0 → (0,0), index 7 → (1,0), index 41 → (5,6)', () => {
		expect(gridCellPosition(0)).toEqual({ row: 0, col: 0 });
		expect(gridCellPosition(6)).toEqual({ row: 0, col: 6 });
		expect(gridCellPosition(7)).toEqual({ row: 1, col: 0 });
		expect(gridCellPosition(15)).toEqual({ row: 2, col: 1 });
		expect(gridCellPosition(41)).toEqual({ row: 5, col: 6 });
	});

	it('returns null for out-of-range or non-integer indices', () => {
		expect(gridCellPosition(-1)).toBeNull();
		expect(gridCellPosition(42)).toBeNull();
		expect(gridCellPosition(100)).toBeNull();
		expect(gridCellPosition(3.5)).toBeNull();
	});
});

describe('gridStartForAnchor', () => {
	it('returns a Sunday at or before anchor-14 days', () => {
		// Apr 20 2026 (Mon) → anchor-14 = Apr 6 2026 (Mon) → back to Sun Apr 5.
		const start = gridStartForAnchor(anchor());
		expect(start.getDay()).toBe(0);
		expect(start.getFullYear()).toBe(2026);
		expect(start.getMonth()).toBe(3);
		expect(start.getDate()).toBe(5);
		expect(start.getHours()).toBe(0);
	});

	it('includes today in the 6-week window (row 2 or 3)', () => {
		const today = anchor();
		const start = gridStartForAnchor(today);
		const delta = daysBetween(today, start);
		expect(delta).toBeGreaterThanOrEqual(14);
		expect(delta).toBeLessThan(42);
	});
});

describe('computeScheduleStats', () => {
	const now = anchor();

	it('zeros everything for an empty corpus', () => {
		const s = computeScheduleStats(now, []);
		expect(s).toEqual({
			overdue: 0,
			dueToday: 0,
			dueThisWeek: 0,
			dueThisMonth: 0,
			avgDays: 0,
		});
	});

	it('counts each bucket independently (today ⊂ week ⊂ month)', () => {
		const ms = [
			makeMemory({ nextReviewAt: offsetDays(now, -2).toISOString() }), // overdue
			makeMemory({ nextReviewAt: new Date(now).toISOString() }), // today
			makeMemory({ nextReviewAt: offsetDays(now, 3).toISOString() }), // week
			makeMemory({ nextReviewAt: offsetDays(now, 15).toISOString() }), // month
			makeMemory({ nextReviewAt: offsetDays(now, 45).toISOString() }), // out of month
		];
		const s = computeScheduleStats(now, ms);
		expect(s.overdue).toBe(1);
		expect(s.dueToday).toBe(2); // overdue + today (delta <= 0)
		expect(s.dueThisWeek).toBe(3); // overdue + today + week
		expect(s.dueThisMonth).toBe(4); // overdue + today + week + month
	});

	it('skips memories without a nextReviewAt or with unparseable dates', () => {
		const ms = [
			makeMemory({ nextReviewAt: undefined }),
			makeMemory({ nextReviewAt: 'bogus' }),
			makeMemory({ nextReviewAt: offsetDays(now, 2).toISOString() }),
		];
		const s = computeScheduleStats(now, ms);
		expect(s.dueThisWeek).toBe(1);
	});

	it('computes average days across future-only memories', () => {
		const ms = [
			makeMemory({ nextReviewAt: offsetDays(now, -5).toISOString() }), // excluded (past)
			makeMemory({ nextReviewAt: offsetDays(now, 2).toISOString() }),
			makeMemory({ nextReviewAt: offsetDays(now, 4).toISOString() }),
		];
		const s = computeScheduleStats(now, ms);
		// avgDays is measured from today-at-midnight (not now-mid-day), so a
		// review tomorrow at noon is 1.5 days out. Two memories at +2d and +4d
		// (both hour=12) → (2.5 + 4.5) / 2 = 3.5.
		expect(s.avgDays).toBeCloseTo(3.5, 2);
	});
});

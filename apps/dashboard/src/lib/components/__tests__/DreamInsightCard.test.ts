/**
 * Tests for DreamInsightCard helpers.
 *
 * Pure logic only — the Svelte template is a thin wrapper around these.
 * Covers the boundaries of the gold-glow / muted novelty mapping, the
 * formatting helpers, and the source-memory link scheme.
 */
import { describe, it, expect } from 'vitest';

import {
	LOW_NOVELTY_THRESHOLD,
	HIGH_NOVELTY_THRESHOLD,
	clamp01,
	noveltyBand,
	formatDurationMs,
	formatConfidencePct,
	sourceMemoryHref,
	firstSourceIds,
	extraSourceCount,
	shortMemoryId,
} from '../dream-helpers';

// ---------------------------------------------------------------------------
// clamp01
// ---------------------------------------------------------------------------

describe('clamp01', () => {
	it.each<[number | null | undefined, number]>([
		[0, 0],
		[1, 1],
		[0.5, 0.5],
		[-0.1, 0],
		[-5, 0],
		[1.1, 1],
		[100, 1],
		[null, 0],
		[undefined, 0],
		[Number.NaN, 0],
		[Number.POSITIVE_INFINITY, 0],
		[Number.NEGATIVE_INFINITY, 0],
	])('clamp01(%s) → %s', (input, expected) => {
		expect(clamp01(input)).toBe(expected);
	});
});

// ---------------------------------------------------------------------------
// noveltyBand — the gold/muted visual classifier
// ---------------------------------------------------------------------------

describe('noveltyBand — gold-glow / muted classification', () => {
	it('has the documented thresholds', () => {
		// These constants are contractual — the component's class bindings
		// depend on them. If they change, the visual band shifts.
		expect(LOW_NOVELTY_THRESHOLD).toBe(0.3);
		expect(HIGH_NOVELTY_THRESHOLD).toBe(0.7);
	});

	it('classifies low-novelty (< 0.3) as muted', () => {
		expect(noveltyBand(0)).toBe('low');
		expect(noveltyBand(0.1)).toBe('low');
		expect(noveltyBand(0.29)).toBe('low');
		expect(noveltyBand(0.2999)).toBe('low');
	});

	it('classifies the boundary 0.3 exactly as neutral (NOT low)', () => {
		// The component uses `novelty < 0.3`, strictly exclusive.
		expect(noveltyBand(0.3)).toBe('neutral');
	});

	it('classifies mid-range as neutral', () => {
		expect(noveltyBand(0.3)).toBe('neutral');
		expect(noveltyBand(0.5)).toBe('neutral');
		expect(noveltyBand(0.7)).toBe('neutral');
	});

	it('classifies the boundary 0.7 exactly as neutral (NOT high)', () => {
		// The component uses `novelty > 0.7`, strictly exclusive.
		expect(noveltyBand(0.7)).toBe('neutral');
	});

	it('classifies high-novelty (> 0.7) as gold/high', () => {
		expect(noveltyBand(0.71)).toBe('high');
		expect(noveltyBand(0.7001)).toBe('high');
		expect(noveltyBand(0.9)).toBe('high');
		expect(noveltyBand(1.0)).toBe('high');
	});

	it('collapses null / undefined / NaN to the low band', () => {
		expect(noveltyBand(null)).toBe('low');
		expect(noveltyBand(undefined)).toBe('low');
		expect(noveltyBand(Number.NaN)).toBe('low');
	});

	it('clamps out-of-range values before classifying', () => {
		// 2.0 clamps to 1.0 → high; -1 clamps to 0 → low.
		expect(noveltyBand(2.0)).toBe('high');
		expect(noveltyBand(-1)).toBe('low');
	});
});

// ---------------------------------------------------------------------------
// formatDurationMs
// ---------------------------------------------------------------------------

describe('formatDurationMs', () => {
	it('renders sub-second values with "ms" suffix', () => {
		expect(formatDurationMs(0)).toBe('0ms');
		expect(formatDurationMs(1)).toBe('1ms');
		expect(formatDurationMs(500)).toBe('500ms');
		expect(formatDurationMs(999)).toBe('999ms');
	});

	it('renders second-and-above values with "s" suffix, 2 decimals', () => {
		expect(formatDurationMs(1000)).toBe('1.00s');
		expect(formatDurationMs(1500)).toBe('1.50s');
		expect(formatDurationMs(15000)).toBe('15.00s');
		expect(formatDurationMs(60000)).toBe('60.00s');
	});

	it('rounds fractional millisecond values in the "ms" band', () => {
		expect(formatDurationMs(0.4)).toBe('0ms');
		expect(formatDurationMs(12.7)).toBe('13ms');
	});

	it('returns "0ms" for null / undefined / NaN / negative', () => {
		expect(formatDurationMs(null)).toBe('0ms');
		expect(formatDurationMs(undefined)).toBe('0ms');
		expect(formatDurationMs(Number.NaN)).toBe('0ms');
		expect(formatDurationMs(-100)).toBe('0ms');
		expect(formatDurationMs(Number.POSITIVE_INFINITY)).toBe('0ms');
	});
});

// ---------------------------------------------------------------------------
// formatConfidencePct
// ---------------------------------------------------------------------------

describe('formatConfidencePct', () => {
	it('renders 0 / 0.5 / 1 as whole-percent strings', () => {
		expect(formatConfidencePct(0)).toBe('0%');
		expect(formatConfidencePct(0.5)).toBe('50%');
		expect(formatConfidencePct(1)).toBe('100%');
	});

	it('rounds intermediate values', () => {
		expect(formatConfidencePct(0.123)).toBe('12%');
		expect(formatConfidencePct(0.5049)).toBe('50%');
		expect(formatConfidencePct(0.505)).toBe('51%');
		expect(formatConfidencePct(0.999)).toBe('100%');
	});

	it('clamps out-of-range input first', () => {
		expect(formatConfidencePct(-0.5)).toBe('0%');
		expect(formatConfidencePct(2)).toBe('100%');
	});

	it('handles null / undefined / NaN', () => {
		expect(formatConfidencePct(null)).toBe('0%');
		expect(formatConfidencePct(undefined)).toBe('0%');
		expect(formatConfidencePct(Number.NaN)).toBe('0%');
	});
});

// ---------------------------------------------------------------------------
// sourceMemoryHref
// ---------------------------------------------------------------------------

describe('sourceMemoryHref — link format', () => {
	it('builds the canonical /memories/:id path with no base', () => {
		expect(sourceMemoryHref('abc123')).toBe('/memories/abc123');
	});

	it('prepends the SvelteKit base path when provided', () => {
		expect(sourceMemoryHref('abc123', '/dashboard')).toBe(
			'/dashboard/memories/abc123',
		);
	});

	it('handles an empty base (default behaviour)', () => {
		expect(sourceMemoryHref('abc', '')).toBe('/memories/abc');
	});

	it('passes through full UUIDs untouched', () => {
		const uuid = '550e8400-e29b-41d4-a716-446655440000';
		expect(sourceMemoryHref(uuid)).toBe(`/memories/${uuid}`);
	});
});

// ---------------------------------------------------------------------------
// firstSourceIds + extraSourceCount
// ---------------------------------------------------------------------------

describe('firstSourceIds', () => {
	it('returns [] for empty / null / undefined inputs', () => {
		expect(firstSourceIds([])).toEqual([]);
		expect(firstSourceIds(null)).toEqual([]);
		expect(firstSourceIds(undefined)).toEqual([]);
	});

	it('returns the single element when array has one entry', () => {
		expect(firstSourceIds(['a'])).toEqual(['a']);
	});

	it('returns the first 2 by default', () => {
		expect(firstSourceIds(['a', 'b', 'c', 'd'])).toEqual(['a', 'b']);
	});

	it('honours a custom N', () => {
		expect(firstSourceIds(['a', 'b', 'c', 'd'], 3)).toEqual(['a', 'b', 'c']);
		expect(firstSourceIds(['a', 'b', 'c'], 5)).toEqual(['a', 'b', 'c']);
	});

	it('returns [] for non-positive N', () => {
		expect(firstSourceIds(['a', 'b'], 0)).toEqual([]);
		expect(firstSourceIds(['a', 'b'], -1)).toEqual([]);
	});
});

describe('extraSourceCount', () => {
	it('returns 0 when there are no extras', () => {
		expect(extraSourceCount([])).toBe(0);
		expect(extraSourceCount(null)).toBe(0);
		expect(extraSourceCount(['a'])).toBe(0);
		expect(extraSourceCount(['a', 'b'])).toBe(0);
	});

	it('returns sources.length - shown when there are extras', () => {
		expect(extraSourceCount(['a', 'b', 'c'])).toBe(1);
		expect(extraSourceCount(['a', 'b', 'c', 'd', 'e'])).toBe(3);
	});

	it('honours a custom shown parameter', () => {
		expect(extraSourceCount(['a', 'b', 'c', 'd', 'e'], 3)).toBe(2);
		expect(extraSourceCount(['a', 'b'], 5)).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// shortMemoryId
// ---------------------------------------------------------------------------

describe('shortMemoryId', () => {
	it('returns the full string when 8 chars or fewer', () => {
		expect(shortMemoryId('abc')).toBe('abc');
		expect(shortMemoryId('12345678')).toBe('12345678');
	});

	it('slices to 8 chars when longer', () => {
		expect(shortMemoryId('123456789')).toBe('12345678');
		expect(shortMemoryId('550e8400-e29b-41d4-a716-446655440000')).toBe(
			'550e8400',
		);
	});

	it('handles empty string defensively', () => {
		expect(shortMemoryId('')).toBe('');
	});
});

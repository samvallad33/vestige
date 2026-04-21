/**
 * Contradiction Constellation — pure-helper coverage.
 *
 * Runs in the vitest `node` environment (no jsdom). We only test the pure
 * helpers extracted to `contradiction-helpers.ts`; the Svelte component is
 * covered indirectly because every classification, opacity, radius, and
 * color decision it renders routes through these functions.
 */
import { describe, it, expect } from 'vitest';

import {
	severityColor,
	severityLabel,
	nodeColor,
	nodeRadius,
	clampTrust,
	pairOpacity,
	truncate,
	uniqueMemoryCount,
	avgTrustDelta,
	NODE_COLORS,
	KNOWN_NODE_TYPES,
	NODE_COLOR_FALLBACK,
	NODE_RADIUS_MIN,
	NODE_RADIUS_RANGE,
	SEVERITY_STRONG_COLOR,
	SEVERITY_MODERATE_COLOR,
	SEVERITY_MILD_COLOR,
	UNFOCUSED_OPACITY,
	type ContradictionLike,
} from '../contradiction-helpers';

// ---------------------------------------------------------------------------
// severityColor — strict-greater-than thresholds at 0.5 and 0.7.
// ---------------------------------------------------------------------------

describe('severityColor', () => {
	it('returns mild yellow at or below 0.5', () => {
		expect(severityColor(0)).toBe(SEVERITY_MILD_COLOR);
		expect(severityColor(0.29)).toBe(SEVERITY_MILD_COLOR);
		expect(severityColor(0.3)).toBe(SEVERITY_MILD_COLOR);
		expect(severityColor(0.5)).toBe(SEVERITY_MILD_COLOR); // boundary → lower band
	});

	it('returns moderate amber strictly above 0.5 and up to 0.7', () => {
		expect(severityColor(0.51)).toBe(SEVERITY_MODERATE_COLOR);
		expect(severityColor(0.6)).toBe(SEVERITY_MODERATE_COLOR);
		expect(severityColor(0.7)).toBe(SEVERITY_MODERATE_COLOR); // boundary → lower band
	});

	it('returns strong red strictly above 0.7', () => {
		expect(severityColor(0.71)).toBe(SEVERITY_STRONG_COLOR);
		expect(severityColor(0.9)).toBe(SEVERITY_STRONG_COLOR);
		expect(severityColor(1.0)).toBe(SEVERITY_STRONG_COLOR);
	});

	it('handles out-of-range numbers without crashing', () => {
		expect(severityColor(-1)).toBe(SEVERITY_MILD_COLOR);
		expect(severityColor(1.5)).toBe(SEVERITY_STRONG_COLOR);
	});
});

// ---------------------------------------------------------------------------
// severityLabel — matches severityColor thresholds.
// ---------------------------------------------------------------------------

describe('severityLabel', () => {
	it('labels mild at 0, 0.29, 0.3, 0.5', () => {
		expect(severityLabel(0)).toBe('mild');
		expect(severityLabel(0.29)).toBe('mild');
		expect(severityLabel(0.3)).toBe('mild');
		expect(severityLabel(0.5)).toBe('mild');
	});

	it('labels moderate at 0.51, 0.7', () => {
		expect(severityLabel(0.51)).toBe('moderate');
		expect(severityLabel(0.7)).toBe('moderate');
	});

	it('labels strong at 0.71, 1.0', () => {
		expect(severityLabel(0.71)).toBe('strong');
		expect(severityLabel(1.0)).toBe('strong');
	});

	it('covers all 8 ordered boundary cases from the audit', () => {
		expect(severityLabel(0)).toBe('mild');
		expect(severityLabel(0.29)).toBe('mild');
		expect(severityLabel(0.3)).toBe('mild');
		expect(severityLabel(0.5)).toBe('mild');
		expect(severityLabel(0.51)).toBe('moderate');
		expect(severityLabel(0.7)).toBe('moderate');
		expect(severityLabel(0.71)).toBe('strong');
		expect(severityLabel(1.0)).toBe('strong');
	});
});

// ---------------------------------------------------------------------------
// nodeColor — 8 known types plus fallback.
// ---------------------------------------------------------------------------

describe('nodeColor', () => {
	it('returns distinct colors for each of the 8 known node types', () => {
		const colors = KNOWN_NODE_TYPES.map((t) => nodeColor(t));
		expect(colors.length).toBe(8);
		expect(new Set(colors).size).toBe(8); // all distinct
	});

	it('matches the canonical palette exactly', () => {
		expect(nodeColor('fact')).toBe(NODE_COLORS.fact);
		expect(nodeColor('concept')).toBe(NODE_COLORS.concept);
		expect(nodeColor('event')).toBe(NODE_COLORS.event);
		expect(nodeColor('person')).toBe(NODE_COLORS.person);
		expect(nodeColor('place')).toBe(NODE_COLORS.place);
		expect(nodeColor('note')).toBe(NODE_COLORS.note);
		expect(nodeColor('pattern')).toBe(NODE_COLORS.pattern);
		expect(nodeColor('decision')).toBe(NODE_COLORS.decision);
	});

	it('falls back to violet for unknown / missing types', () => {
		expect(nodeColor(undefined)).toBe(NODE_COLOR_FALLBACK);
		expect(nodeColor(null)).toBe(NODE_COLOR_FALLBACK);
		expect(nodeColor('')).toBe(NODE_COLOR_FALLBACK);
		expect(nodeColor('bogus')).toBe(NODE_COLOR_FALLBACK);
		expect(nodeColor('FACT')).toBe(NODE_COLOR_FALLBACK); // case-sensitive
	});

	it('violet fallback equals 0x8b5cf6', () => {
		expect(NODE_COLOR_FALLBACK).toBe('#8b5cf6');
	});
});

// ---------------------------------------------------------------------------
// nodeRadius + clampTrust — trust is defined on [0,1].
// ---------------------------------------------------------------------------

describe('nodeRadius', () => {
	it('returns the minimum radius at trust=0', () => {
		expect(nodeRadius(0)).toBe(NODE_RADIUS_MIN);
	});

	it('returns min + range at trust=1', () => {
		expect(nodeRadius(1)).toBe(NODE_RADIUS_MIN + NODE_RADIUS_RANGE);
	});

	it('scales linearly in between', () => {
		expect(nodeRadius(0.5)).toBeCloseTo(NODE_RADIUS_MIN + NODE_RADIUS_RANGE * 0.5);
	});

	it('clamps negative trust to 0 (minimum radius)', () => {
		expect(nodeRadius(-0.5)).toBe(NODE_RADIUS_MIN);
		expect(nodeRadius(-Infinity)).toBe(NODE_RADIUS_MIN);
	});

	it('clamps >1 trust to 1 (maximum radius)', () => {
		expect(nodeRadius(2)).toBe(NODE_RADIUS_MIN + NODE_RADIUS_RANGE);
		expect(nodeRadius(Infinity)).toBe(NODE_RADIUS_MIN);
		// ^ Infinity isn't finite — falls back to min, matching "suppress suspicious data"
	});

	it('treats NaN as minimum (suppress bad data)', () => {
		expect(nodeRadius(NaN)).toBe(NODE_RADIUS_MIN);
	});
});

describe('clampTrust', () => {
	it('returns values inside [0,1] unchanged', () => {
		expect(clampTrust(0)).toBe(0);
		expect(clampTrust(0.5)).toBe(0.5);
		expect(clampTrust(1)).toBe(1);
	});

	it('clamps negatives to 0 and >1 to 1', () => {
		expect(clampTrust(-0.3)).toBe(0);
		expect(clampTrust(1.3)).toBe(1);
	});

	it('collapses NaN / null / undefined / Infinity to 0', () => {
		expect(clampTrust(NaN)).toBe(0);
		expect(clampTrust(null)).toBe(0);
		expect(clampTrust(undefined)).toBe(0);
		expect(clampTrust(Infinity)).toBe(0);
		expect(clampTrust(-Infinity)).toBe(0);
	});
});

// ---------------------------------------------------------------------------
// pairOpacity — trinary: no focus = 1, focused = 1, unfocused = 0.12.
// ---------------------------------------------------------------------------

describe('pairOpacity', () => {
	it('returns 1 when no pair is focused (null)', () => {
		expect(pairOpacity(0, null)).toBe(1);
		expect(pairOpacity(5, null)).toBe(1);
	});

	it('returns 1 when no pair is focused (undefined)', () => {
		expect(pairOpacity(0, undefined)).toBe(1);
		expect(pairOpacity(5, undefined)).toBe(1);
	});

	it('returns 1 for the focused pair', () => {
		expect(pairOpacity(3, 3)).toBe(1);
		expect(pairOpacity(0, 0)).toBe(1);
	});

	it('returns 0.12 for a non-focused pair when something is focused', () => {
		expect(pairOpacity(0, 3)).toBe(UNFOCUSED_OPACITY);
		expect(pairOpacity(7, 3)).toBe(UNFOCUSED_OPACITY);
	});

	it('does not explode for a stale focus index that matches nothing', () => {
		// A focus index of 999 with only 5 pairs: every visible pair dims to 0.12.
		// The missing pair renders nothing (silent no-op is correct).
		for (let i = 0; i < 5; i++) {
			expect(pairOpacity(i, 999)).toBe(UNFOCUSED_OPACITY);
		}
	});
});

// ---------------------------------------------------------------------------
// truncate — length boundaries, empties, odd inputs.
// ---------------------------------------------------------------------------

describe('truncate', () => {
	it('returns strings shorter than max unchanged', () => {
		expect(truncate('hi', 10)).toBe('hi');
		expect(truncate('abc', 5)).toBe('abc');
	});

	it('returns empty strings unchanged', () => {
		expect(truncate('', 5)).toBe('');
		expect(truncate('', 0)).toBe('');
	});

	it('returns strings exactly at max unchanged', () => {
		expect(truncate('12345', 5)).toBe('12345');
		expect(truncate('abcdef', 6)).toBe('abcdef');
	});

	it('cuts strings longer than max, appending ellipsis within budget', () => {
		expect(truncate('1234567890', 5)).toBe('1234…');
		expect(truncate('hello world', 6)).toBe('hello…');
	});

	it('uses default max of 60', () => {
		const long = 'a'.repeat(100);
		const out = truncate(long);
		expect(out.length).toBe(60);
		expect(out.endsWith('…')).toBe(true);
	});

	it('null / undefined inputs return empty string', () => {
		expect(truncate(null)).toBe('');
		expect(truncate(undefined)).toBe('');
	});

	it('handles max=0 safely', () => {
		expect(truncate('any string', 0)).toBe('');
	});

	it('handles max=1 safely — one-char budget collapses to just the ellipsis', () => {
		expect(truncate('abc', 1)).toBe('…');
	});
});

// ---------------------------------------------------------------------------
// uniqueMemoryCount — union of memory_a_id + memory_b_id across pairs.
// ---------------------------------------------------------------------------

describe('uniqueMemoryCount', () => {
	const mkPair = (a: string, b: string): ContradictionLike => ({
		memory_a_id: a,
		memory_b_id: b,
	});

	it('returns 0 for empty input', () => {
		expect(uniqueMemoryCount([])).toBe(0);
	});

	it('counts both sides of every pair', () => {
		expect(uniqueMemoryCount([mkPair('a', 'b')])).toBe(2);
		expect(uniqueMemoryCount([mkPair('a', 'b'), mkPair('c', 'd')])).toBe(4);
	});

	it('deduplicates memories that appear in multiple pairs', () => {
		// 'a' appears on both sides of two separate pairs.
		expect(uniqueMemoryCount([mkPair('a', 'b'), mkPair('a', 'c')])).toBe(3);
		expect(uniqueMemoryCount([mkPair('a', 'b'), mkPair('b', 'a')])).toBe(2);
	});

	it('handles a memory conflicting with itself (same id both sides)', () => {
		expect(uniqueMemoryCount([mkPair('a', 'a')])).toBe(1);
	});

	it('ignores empty-string ids', () => {
		expect(uniqueMemoryCount([mkPair('', '')])).toBe(0);
		expect(uniqueMemoryCount([mkPair('a', '')])).toBe(1);
	});
});

// ---------------------------------------------------------------------------
// avgTrustDelta — safety against empty inputs.
// ---------------------------------------------------------------------------

describe('avgTrustDelta', () => {
	it('returns 0 on empty input (no NaN)', () => {
		expect(avgTrustDelta([])).toBe(0);
	});

	it('computes mean absolute delta', () => {
		const pairs = [
			{ trust_a: 0.9, trust_b: 0.1 }, // 0.8
			{ trust_a: 0.5, trust_b: 0.3 }, // 0.2
		];
		expect(avgTrustDelta(pairs)).toBeCloseTo(0.5);
	});

	it('takes absolute value (order does not matter)', () => {
		expect(avgTrustDelta([{ trust_a: 0.1, trust_b: 0.9 }])).toBeCloseTo(0.8);
		expect(avgTrustDelta([{ trust_a: 0.9, trust_b: 0.1 }])).toBeCloseTo(0.8);
	});

	it('returns 0 when both sides are equal', () => {
		expect(avgTrustDelta([{ trust_a: 0.5, trust_b: 0.5 }])).toBe(0);
	});
});

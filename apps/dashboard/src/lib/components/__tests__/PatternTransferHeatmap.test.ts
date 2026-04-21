/**
 * Unit tests for patterns-helpers — the pure logic backing
 * PatternTransferHeatmap.svelte + patterns/+page.svelte.
 *
 * Runs in the vitest `node` environment (no jsdom). We never touch Svelte
 * component internals here — only the exported helpers in patterns-helpers.ts.
 * Component-level integration (click, hover, DOM wiring) is covered by the
 * Playwright e2e suite; this file is pure-logic coverage of the contracts.
 */

import { describe, it, expect } from 'vitest';
import {
	cellIntensity,
	filterByCategory,
	buildTransferMatrix,
	matrixMaxCount,
	flattenNonZero,
	shortProjectName,
	PATTERN_CATEGORIES,
	type TransferPatternLike,
} from '../patterns-helpers';

// ---------------------------------------------------------------------------
// Test fixtures — mirror the mockFetchCrossProject shape in
// patterns/+page.svelte, but small enough to reason about by hand.
// ---------------------------------------------------------------------------

const PROJECTS = ['vestige', 'nullgaze', 'injeranet'] as const;

const PATTERNS: TransferPatternLike[] = [
	{
		name: 'Result<T, E>',
		category: 'ErrorHandling',
		origin_project: 'vestige',
		transferred_to: ['nullgaze', 'injeranet'],
		transfer_count: 2,
	},
	{
		name: 'Axum middleware',
		category: 'ErrorHandling',
		origin_project: 'nullgaze',
		transferred_to: ['vestige'],
		transfer_count: 1,
	},
	{
		name: 'proptest',
		category: 'Testing',
		origin_project: 'vestige',
		transferred_to: ['nullgaze'],
		transfer_count: 1,
	},
	{
		name: 'Self-reuse pattern',
		category: 'Architecture',
		origin_project: 'vestige',
		transferred_to: ['vestige'], // diagonal — self-reuse
		transfer_count: 1,
	},
];

// ===========================================================================
// cellIntensity — 0..1 opacity normaliser
// ===========================================================================

describe('cellIntensity', () => {
	it('returns 0 for a zero count', () => {
		expect(cellIntensity(0, 10)).toBe(0);
	});

	it('returns 1 at max', () => {
		expect(cellIntensity(10, 10)).toBe(1);
	});

	it('returns 1 when count exceeds max (defensive clamp)', () => {
		expect(cellIntensity(15, 10)).toBe(1);
	});

	it('scales linearly between 0 and max', () => {
		expect(cellIntensity(3, 10)).toBeCloseTo(0.3, 5);
		expect(cellIntensity(5, 10)).toBeCloseTo(0.5, 5);
		expect(cellIntensity(7, 10)).toBeCloseTo(0.7, 5);
	});

	it('returns 0 when max is 0 (div-by-zero guard)', () => {
		expect(cellIntensity(5, 0)).toBe(0);
	});

	it('returns 0 for negative counts', () => {
		expect(cellIntensity(-1, 10)).toBe(0);
	});

	it('returns 0 for NaN inputs', () => {
		expect(cellIntensity(NaN, 10)).toBe(0);
		expect(cellIntensity(5, NaN)).toBe(0);
	});

	it('returns 0 for Infinity inputs', () => {
		expect(cellIntensity(Infinity, 10)).toBe(0);
		expect(cellIntensity(5, Infinity)).toBe(0);
	});
});

// ===========================================================================
// filterByCategory — drives both heatmap + sidebar reflow
// ===========================================================================

describe('filterByCategory', () => {
	it("returns every pattern for 'All'", () => {
		const out = filterByCategory(PATTERNS, 'All');
		expect(out).toHaveLength(PATTERNS.length);
		// Should NOT return the same reference — helpers return a copy so
		// callers can mutate freely.
		expect(out).not.toBe(PATTERNS);
	});

	it('filters strictly by category equality', () => {
		const errorOnly = filterByCategory(PATTERNS, 'ErrorHandling');
		expect(errorOnly).toHaveLength(2);
		expect(errorOnly.every((p) => p.category === 'ErrorHandling')).toBe(true);
	});

	it('returns exactly one match for Testing', () => {
		const testing = filterByCategory(PATTERNS, 'Testing');
		expect(testing).toHaveLength(1);
		expect(testing[0].name).toBe('proptest');
	});

	it('returns an empty array for a category with no patterns', () => {
		const perf = filterByCategory(PATTERNS, 'Performance');
		expect(perf).toEqual([]);
	});

	it('returns an empty array for an unknown category string (no silent alias)', () => {
		// This is the "unknown category fallback" contract — we do NOT
		// quietly fall back to 'All'. An unknown category is a caller bug
		// and yields an empty list so the empty-state UI renders.
		expect(filterByCategory(PATTERNS, 'NotARealCategory')).toEqual([]);
		expect(filterByCategory(PATTERNS, '')).toEqual([]);
	});

	it('accepts an empty input array for any category', () => {
		expect(filterByCategory([], 'All')).toEqual([]);
		expect(filterByCategory([], 'ErrorHandling')).toEqual([]);
		expect(filterByCategory([], 'BogusCategory')).toEqual([]);
	});

	it('exposes all six supported categories', () => {
		expect([...PATTERN_CATEGORIES]).toEqual([
			'ErrorHandling',
			'AsyncConcurrency',
			'Testing',
			'Architecture',
			'Performance',
			'Security',
		]);
	});
});

// ===========================================================================
// buildTransferMatrix — directional N×N projects × projects grid
// ===========================================================================

describe('buildTransferMatrix', () => {
	it('constructs an N×N matrix over the projects axis', () => {
		const m = buildTransferMatrix(PROJECTS, []);
		for (const from of PROJECTS) {
			for (const to of PROJECTS) {
				expect(m[from][to]).toEqual({ count: 0, topNames: [] });
			}
		}
	});

	it('aggregates transfer counts directionally', () => {
		const m = buildTransferMatrix(PROJECTS, PATTERNS);
		// vestige → nullgaze: Result<T,E> + proptest = 2
		expect(m.vestige.nullgaze.count).toBe(2);
		// vestige → injeranet: Result<T,E> only = 1
		expect(m.vestige.injeranet.count).toBe(1);
		// nullgaze → vestige: Axum middleware = 1
		expect(m.nullgaze.vestige.count).toBe(1);
		// injeranet → anywhere: zero (no origin in injeranet in fixtures)
		expect(m.injeranet.vestige.count).toBe(0);
		expect(m.injeranet.nullgaze.count).toBe(0);
	});

	it('treats (A, B) and (B, A) as distinct directions (asymmetry confirmed)', () => {
		// The component's doc-comment says "Rows = origin project · Columns =
		// destination project" — the matrix MUST be directional. A copy-paste
		// bug that aggregates both directions into the same cell would pass
		// the "count" test above but fail this symmetry check.
		const m = buildTransferMatrix(PROJECTS, PATTERNS);
		expect(m.vestige.nullgaze.count).not.toBe(m.nullgaze.vestige.count);
	});

	it('records self-transfer on the diagonal', () => {
		const m = buildTransferMatrix(PROJECTS, PATTERNS);
		expect(m.vestige.vestige.count).toBe(1);
		expect(m.vestige.vestige.topNames).toEqual(['Self-reuse pattern']);
	});

	it('captures top pattern names per cell, capped at 3', () => {
		const manyPatterns: TransferPatternLike[] = Array.from({ length: 5 }, (_, i) => ({
			name: `pattern-${i}`,
			category: 'ErrorHandling',
			origin_project: 'vestige',
			transferred_to: ['nullgaze'],
			transfer_count: 1,
		}));
		const m = buildTransferMatrix(['vestige', 'nullgaze'], manyPatterns);
		expect(m.vestige.nullgaze.count).toBe(5);
		expect(m.vestige.nullgaze.topNames).toHaveLength(3);
		expect(m.vestige.nullgaze.topNames).toEqual(['pattern-0', 'pattern-1', 'pattern-2']);
	});

	it('silently drops patterns whose origin is not in the projects axis', () => {
		const orphan: TransferPatternLike = {
			name: 'Orphan',
			category: 'Security',
			origin_project: 'ghost-project',
			transferred_to: ['vestige'],
			transfer_count: 1,
		};
		const m = buildTransferMatrix(PROJECTS, [orphan]);
		// Nothing anywhere in the matrix should have ticked up.
		const total = matrixMaxCount(PROJECTS, m);
		expect(total).toBe(0);
		// Matrix structure intact — no ghost key added.
		expect((m as Record<string, unknown>)['ghost-project']).toBeUndefined();
	});

	it('silently drops transferred_to entries not in the projects axis', () => {
		const strayDest: TransferPatternLike = {
			name: 'StrayDest',
			category: 'Security',
			origin_project: 'vestige',
			transferred_to: ['ghost-project', 'nullgaze'],
			transfer_count: 2,
		};
		const m = buildTransferMatrix(PROJECTS, [strayDest]);
		// The known destination counts; the ghost doesn't.
		expect(m.vestige.nullgaze.count).toBe(1);
		expect((m.vestige as Record<string, unknown>)['ghost-project']).toBeUndefined();
	});

	it('respects a custom top-name cap', () => {
		const pats: TransferPatternLike[] = [
			{
				name: 'a',
				category: 'Testing',
				origin_project: 'vestige',
				transferred_to: ['nullgaze'],
				transfer_count: 1,
			},
			{
				name: 'b',
				category: 'Testing',
				origin_project: 'vestige',
				transferred_to: ['nullgaze'],
				transfer_count: 1,
			},
		];
		const m = buildTransferMatrix(['vestige', 'nullgaze'], pats, 1);
		expect(m.vestige.nullgaze.topNames).toEqual(['a']);
	});
});

// ===========================================================================
// matrixMaxCount
// ===========================================================================

describe('matrixMaxCount', () => {
	it('returns 0 for an empty matrix (div-by-zero guard prerequisite)', () => {
		const m = buildTransferMatrix(PROJECTS, []);
		expect(matrixMaxCount(PROJECTS, m)).toBe(0);
	});

	it('returns the hottest cell count across all pairs', () => {
		const m = buildTransferMatrix(PROJECTS, PATTERNS);
		// vestige→nullgaze has 2; everything else is ≤1
		expect(matrixMaxCount(PROJECTS, m)).toBe(2);
	});

	it('tolerates missing rows without crashing', () => {
		const partial: Record<string, Record<string, { count: number; topNames: string[] }>> = {
			vestige: { vestige: { count: 3, topNames: [] } },
		};
		expect(matrixMaxCount(['vestige', 'absent'], partial)).toBe(3);
	});
});

// ===========================================================================
// flattenNonZero — mobile fallback feed
// ===========================================================================

describe('flattenNonZero', () => {
	it('returns only non-zero pairs, sorted by count descending', () => {
		const m = buildTransferMatrix(PROJECTS, PATTERNS);
		const rows = flattenNonZero(PROJECTS, m);
		// Distinct non-zero cells in fixtures:
		//   vestige→nullgaze  = 2
		//   vestige→injeranet = 1
		//   vestige→vestige   = 1
		//   nullgaze→vestige  = 1
		expect(rows).toHaveLength(4);
		expect(rows[0]).toMatchObject({ from: 'vestige', to: 'nullgaze', count: 2 });
		// Later rows all tied at 1 — we only verify the leader.
		expect(rows.slice(1).every((r) => r.count === 1)).toBe(true);
	});

	it('returns an empty list when nothing is transferred', () => {
		const m = buildTransferMatrix(PROJECTS, []);
		expect(flattenNonZero(PROJECTS, m)).toEqual([]);
	});
});

// ===========================================================================
// shortProjectName
// ===========================================================================

describe('shortProjectName', () => {
	it('passes short names through unchanged', () => {
		expect(shortProjectName('vestige')).toBe('vestige');
		expect(shortProjectName('')).toBe('');
	});

	it('keeps names at the 12-char boundary', () => {
		expect(shortProjectName('123456789012')).toBe('123456789012');
	});

	it('truncates longer names to 11 chars + ellipsis', () => {
		expect(shortProjectName('1234567890123')).toBe('12345678901…');
		expect(shortProjectName('super-long-project-name')).toBe('super-long-…');
	});
});

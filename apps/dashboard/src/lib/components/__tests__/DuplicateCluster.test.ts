/**
 * Pure-logic tests for the Memory Hygiene / Duplicate Detection UI.
 *
 * The Svelte components themselves are render-level code (no jsdom in this
 * repo) — every ounce of behaviour worth testing is extracted into
 * `duplicates-helpers.ts` and exercised here. If this file is green, the
 * similarity bands, winner selection, suggested-action mapping, threshold
 * filtering, cluster-identity keying, and the "safe render" helpers are all
 * sound.
 */
import { describe, it, expect } from 'vitest';

import {
	similarityBand,
	similarityBandColor,
	similarityBandLabel,
	retentionColor,
	pickWinner,
	suggestedActionFor,
	filterByThreshold,
	clusterKey,
	previewContent,
	formatDate,
	safeTags,
} from '../duplicates-helpers';

// ---------------------------------------------------------------------------
// Similarity band — boundaries at 0.92 (red) and 0.80 (amber).
// The boundary value MUST land in the higher band (>= semantics).
// ---------------------------------------------------------------------------
describe('similarityBand', () => {
	it('0.92 exactly → near-identical (boundary)', () => {
		expect(similarityBand(0.92)).toBe('near-identical');
	});

	it('0.91 → strong (just below upper boundary)', () => {
		expect(similarityBand(0.91)).toBe('strong');
	});

	it('0.80 exactly → strong (boundary)', () => {
		expect(similarityBand(0.8)).toBe('strong');
	});

	it('0.79 → weak (just below strong boundary)', () => {
		expect(similarityBand(0.79)).toBe('weak');
	});

	it('0.50 → weak (well below)', () => {
		expect(similarityBand(0.5)).toBe('weak');
	});

	it('1.0 → near-identical', () => {
		expect(similarityBand(1.0)).toBe('near-identical');
	});

	it('0.0 → weak', () => {
		expect(similarityBand(0.0)).toBe('weak');
	});
});

describe('similarityBandColor', () => {
	it('near-identical → decay var (red)', () => {
		expect(similarityBandColor(0.95)).toBe('var(--color-decay)');
	});

	it('strong → warning var (amber)', () => {
		expect(similarityBandColor(0.85)).toBe('var(--color-warning)');
	});

	it('weak → yellow-300 literal', () => {
		expect(similarityBandColor(0.78)).toBe('#fde047');
	});

	it('is consistent at boundary 0.92', () => {
		expect(similarityBandColor(0.92)).toBe('var(--color-decay)');
	});

	it('is consistent at boundary 0.80', () => {
		expect(similarityBandColor(0.8)).toBe('var(--color-warning)');
	});
});

describe('similarityBandLabel', () => {
	it('labels near-identical', () => {
		expect(similarityBandLabel(0.97)).toBe('Near-identical');
	});

	it('labels strong', () => {
		expect(similarityBandLabel(0.85)).toBe('Strong match');
	});

	it('labels weak', () => {
		expect(similarityBandLabel(0.75)).toBe('Weak match');
	});
});

// ---------------------------------------------------------------------------
// Retention color — traffic-light: >0.7 green, >0.4 amber, else red.
// ---------------------------------------------------------------------------
describe('retentionColor', () => {
	it('0.85 → green', () => expect(retentionColor(0.85)).toBe('#10b981'));
	it('0.50 → amber', () => expect(retentionColor(0.5)).toBe('#f59e0b'));
	it('0.30 → red', () => expect(retentionColor(0.3)).toBe('#ef4444'));
	it('boundary 0.70 → amber (strict >)', () => expect(retentionColor(0.7)).toBe('#f59e0b'));
	it('boundary 0.40 → red (strict >)', () => expect(retentionColor(0.4)).toBe('#ef4444'));
	it('0.0 → red', () => expect(retentionColor(0)).toBe('#ef4444'));
});

// ---------------------------------------------------------------------------
// Winner selection — highest retention wins; ties → earliest index; empty
// list → null; NaN retentions never win.
// ---------------------------------------------------------------------------
describe('pickWinner', () => {
	it('picks highest retention', () => {
		const mem = [
			{ id: 'a', retention: 0.3 },
			{ id: 'b', retention: 0.9 },
			{ id: 'c', retention: 0.5 },
		];
		expect(pickWinner(mem)?.id).toBe('b');
	});

	it('tie-break: earliest wins (stable)', () => {
		const mem = [
			{ id: 'a', retention: 0.8 },
			{ id: 'b', retention: 0.8 },
			{ id: 'c', retention: 0.7 },
		];
		expect(pickWinner(mem)?.id).toBe('a');
	});

	it('three-way tie: earliest wins', () => {
		const mem = [
			{ id: 'x', retention: 0.5 },
			{ id: 'y', retention: 0.5 },
			{ id: 'z', retention: 0.5 },
		];
		expect(pickWinner(mem)?.id).toBe('x');
	});

	it('all retention = 0: earliest wins (not null)', () => {
		const mem = [
			{ id: 'a', retention: 0 },
			{ id: 'b', retention: 0 },
		];
		expect(pickWinner(mem)?.id).toBe('a');
	});

	it('single-member cluster: that member wins', () => {
		const mem = [{ id: 'solo', retention: 0.42 }];
		expect(pickWinner(mem)?.id).toBe('solo');
	});

	it('empty cluster: returns null', () => {
		expect(pickWinner([])).toBeNull();
	});

	it('NaN retention never wins over a real one', () => {
		const mem = [
			{ id: 'nan', retention: Number.NaN },
			{ id: 'real', retention: 0.1 },
		];
		expect(pickWinner(mem)?.id).toBe('real');
	});

	it('all NaN retentions: earliest wins (stable fallback)', () => {
		const mem = [
			{ id: 'a', retention: Number.NaN },
			{ id: 'b', retention: Number.NaN },
		];
		expect(pickWinner(mem)?.id).toBe('a');
	});
});

// ---------------------------------------------------------------------------
// Suggested action — >=0.92 merge, <0.85 review, 0.85..<0.92 null (caller
// honors upstream).
// ---------------------------------------------------------------------------
describe('suggestedActionFor', () => {
	it('0.95 → merge', () => expect(suggestedActionFor(0.95)).toBe('merge'));
	it('0.92 exactly → merge (boundary)', () => expect(suggestedActionFor(0.92)).toBe('merge'));
	it('0.91 → null (ambiguous corridor)', () => expect(suggestedActionFor(0.91)).toBeNull());
	it('0.85 exactly → null (corridor bottom boundary)', () =>
		expect(suggestedActionFor(0.85)).toBeNull());
	it('0.849 → review (just below corridor)', () =>
		expect(suggestedActionFor(0.849)).toBe('review'));
	it('0.70 → review', () => expect(suggestedActionFor(0.7)).toBe('review'));
	it('0.0 → review', () => expect(suggestedActionFor(0)).toBe('review'));
	it('1.0 → merge', () => expect(suggestedActionFor(1.0)).toBe('merge'));
});

// ---------------------------------------------------------------------------
// Threshold filter — strict >=.
// ---------------------------------------------------------------------------
describe('filterByThreshold', () => {
	const clusters = [
		{ similarity: 0.96, memories: [{ id: '1', retention: 1 }] },
		{ similarity: 0.88, memories: [{ id: '2', retention: 1 }] },
		{ similarity: 0.78, memories: [{ id: '3', retention: 1 }] },
	];

	it('0.80 keeps 0.96 and 0.88 (drops 0.78)', () => {
		const out = filterByThreshold(clusters, 0.8);
		expect(out.map((c) => c.similarity)).toEqual([0.96, 0.88]);
	});

	it('boundary: threshold = 0.88 keeps 0.88 (>=)', () => {
		const out = filterByThreshold(clusters, 0.88);
		expect(out.map((c) => c.similarity)).toEqual([0.96, 0.88]);
	});

	it('boundary: threshold = 0.881 drops 0.88', () => {
		const out = filterByThreshold(clusters, 0.881);
		expect(out.map((c) => c.similarity)).toEqual([0.96]);
	});

	it('0.95 (max) keeps only 0.96', () => {
		const out = filterByThreshold(clusters, 0.95);
		expect(out.map((c) => c.similarity)).toEqual([0.96]);
	});

	it('0.70 (min) keeps all three', () => {
		const out = filterByThreshold(clusters, 0.7);
		expect(out).toHaveLength(3);
	});

	it('empty input → empty output', () => {
		expect(filterByThreshold([], 0.8)).toEqual([]);
	});
});

// ---------------------------------------------------------------------------
// Cluster identity — stable across order shuffles and re-fetches.
// ---------------------------------------------------------------------------
describe('clusterKey', () => {
	it('identical member sets → identical keys (order-independent)', () => {
		const a = [
			{ id: 'a', retention: 0 },
			{ id: 'b', retention: 0 },
			{ id: 'c', retention: 0 },
		];
		const b = [
			{ id: 'c', retention: 0 },
			{ id: 'a', retention: 0 },
			{ id: 'b', retention: 0 },
		];
		expect(clusterKey(a)).toBe(clusterKey(b));
	});

	it('differing members → differing keys', () => {
		const a = [
			{ id: 'a', retention: 0 },
			{ id: 'b', retention: 0 },
		];
		const b = [
			{ id: 'a', retention: 0 },
			{ id: 'c', retention: 0 },
		];
		expect(clusterKey(a)).not.toBe(clusterKey(b));
	});

	it('does not mutate input order', () => {
		const mem = [
			{ id: 'z', retention: 0 },
			{ id: 'a', retention: 0 },
		];
		clusterKey(mem);
		expect(mem.map((m) => m.id)).toEqual(['z', 'a']);
	});

	it('empty cluster → empty string', () => {
		expect(clusterKey([])).toBe('');
	});
});

// ---------------------------------------------------------------------------
// previewContent — trim + collapse whitespace + truncate at 80.
// ---------------------------------------------------------------------------
describe('previewContent', () => {
	it('short content: unchanged', () => {
		expect(previewContent('hello world')).toBe('hello world');
	});

	it('collapses internal whitespace', () => {
		expect(previewContent('  hello    world  ')).toBe('hello world');
	});

	it('truncates with ellipsis', () => {
		const long = 'a'.repeat(120);
		const out = previewContent(long);
		expect(out.length).toBe(81); // 80 + ellipsis
		expect(out.endsWith('…')).toBe(true);
	});

	it('null-safe', () => {
		expect(previewContent(null)).toBe('');
		expect(previewContent(undefined)).toBe('');
	});

	it('honors custom max', () => {
		expect(previewContent('abcdefghij', 5)).toBe('abcde…');
	});
});

// ---------------------------------------------------------------------------
// formatDate — valid ISO → formatted; everything else → empty.
// ---------------------------------------------------------------------------
describe('formatDate', () => {
	it('valid ISO → non-empty formatted string', () => {
		const out = formatDate('2026-04-14T11:02:00Z');
		expect(out.length).toBeGreaterThan(0);
		expect(out).not.toBe('Invalid Date');
	});

	it('empty string → empty', () => {
		expect(formatDate('')).toBe('');
	});

	it('null → empty', () => {
		expect(formatDate(null)).toBe('');
	});

	it('undefined → empty', () => {
		expect(formatDate(undefined)).toBe('');
	});

	it('garbage string → empty (no "Invalid Date" leak)', () => {
		expect(formatDate('not-a-date')).toBe('');
	});

	it('non-string input → empty (defensive)', () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		expect(formatDate(12345 as any)).toBe('');
	});
});

// ---------------------------------------------------------------------------
// safeTags — tolerant of undefined / non-array / empty.
// ---------------------------------------------------------------------------
describe('safeTags', () => {
	it('normal array: slices to limit', () => {
		expect(safeTags(['a', 'b', 'c', 'd', 'e'], 3)).toEqual(['a', 'b', 'c']);
	});

	it('undefined → []', () => {
		expect(safeTags(undefined)).toEqual([]);
	});

	it('null → []', () => {
		expect(safeTags(null)).toEqual([]);
	});

	it('empty array → []', () => {
		expect(safeTags([])).toEqual([]);
	});

	it('non-array (defensive) → []', () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		expect(safeTags('bad' as any)).toEqual([]);
	});

	it('honors default limit = 4', () => {
		expect(safeTags(['a', 'b', 'c', 'd', 'e', 'f'])).toEqual(['a', 'b', 'c', 'd']);
	});
});

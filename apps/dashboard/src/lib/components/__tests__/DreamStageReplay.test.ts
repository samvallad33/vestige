/**
 * Tests for DreamStageReplay helpers.
 *
 * The Svelte component itself is rendered with CSS transforms + derived
 * state. We can't mount it in Node without jsdom, so we test the PURE
 * helpers it relies on — the same helpers also power the page's scrubber
 * and the insight card. If `clampStage` is green, the scrubber can't go
 * out of range; if `STAGE_NAMES` stays in sync with MemoryDreamer's 5
 * phases, the badge labels stay correct.
 */
import { describe, it, expect } from 'vitest';

import {
	STAGE_COUNT,
	STAGE_NAMES,
	clampStage,
	stageName,
} from '../dream-helpers';

describe('STAGE_NAMES — MemoryDreamer phase list', () => {
	it('has exactly 5 stages matching MemoryDreamer.run()', () => {
		expect(STAGE_COUNT).toBe(5);
		expect(STAGE_NAMES).toHaveLength(5);
	});

	it('lists the phases in the canonical order', () => {
		// Order is load-bearing: the stage replay animates in this sequence.
		// Replay → Cross-reference → Strengthen → Prune → Transfer.
		expect(STAGE_NAMES).toEqual([
			'Replay',
			'Cross-reference',
			'Strengthen',
			'Prune',
			'Transfer',
		]);
	});
});

describe('clampStage — valid-range enforcement', () => {
	it.each<[number, number]>([
		// Out-of-bounds low
		[0, 1],
		[-1, 1],
		[-100, 1],
		// In-range (exactly the valid stage indices)
		[1, 1],
		[2, 2],
		[3, 3],
		[4, 4],
		[5, 5],
		// Out-of-bounds high
		[6, 5],
		[7, 5],
		[100, 5],
	])('clampStage(%s) → %s', (input, expected) => {
		expect(clampStage(input)).toBe(expected);
	});

	it('floors fractional values before clamping', () => {
		expect(clampStage(1.9)).toBe(1);
		expect(clampStage(4.9)).toBe(4);
		expect(clampStage(5.1)).toBe(5);
	});

	it('collapses NaN / Infinity / -Infinity to stage 1', () => {
		expect(clampStage(Number.NaN)).toBe(1);
		expect(clampStage(Number.POSITIVE_INFINITY)).toBe(1);
		expect(clampStage(Number.NEGATIVE_INFINITY)).toBe(1);
	});

	it('returns a value usable as a 0-indexed STAGE_NAMES lookup', () => {
		// The page uses `STAGE_NAMES[stageIdx - 1]`. Every clamped value
		// must index a real name, not undefined.
		for (const raw of [-5, 0, 1, 3, 5, 10, Number.NaN]) {
			const idx = clampStage(raw);
			expect(STAGE_NAMES[idx - 1]).toBeDefined();
			expect(typeof STAGE_NAMES[idx - 1]).toBe('string');
		}
	});
});

describe('stageName — resolves to the visible label', () => {
	it('returns the matching name for every valid stage', () => {
		expect(stageName(1)).toBe('Replay');
		expect(stageName(2)).toBe('Cross-reference');
		expect(stageName(3)).toBe('Strengthen');
		expect(stageName(4)).toBe('Prune');
		expect(stageName(5)).toBe('Transfer');
	});

	it('falls back to the nearest valid name for out-of-range input', () => {
		expect(stageName(0)).toBe('Replay');
		expect(stageName(-1)).toBe('Replay');
		expect(stageName(6)).toBe('Transfer');
		expect(stageName(100)).toBe('Transfer');
	});

	it('never returns undefined, even for garbage input', () => {
		for (const raw of [Number.NaN, Number.POSITIVE_INFINITY, -Number.MAX_VALUE]) {
			expect(stageName(raw)).toBeDefined();
			expect(stageName(raw)).toMatch(/^[A-Z]/);
		}
	});
});

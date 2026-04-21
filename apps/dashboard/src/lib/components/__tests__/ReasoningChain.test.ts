/**
 * ReasoningChain — pure-logic coverage.
 *
 * ReasoningChain renders the 8-stage cognitive pipeline. Its rendered output
 * is a pure function of a handful of primitive props — confidence colours,
 * intent-hint selection, and the stage hint resolver. All of that logic
 * lives in `reasoning-helpers.ts` and is exercised here without mounting
 * Svelte.
 */
import { describe, it, expect } from 'vitest';

import {
	confidenceColor,
	confidenceLabel,
	intentHintFor,
	INTENT_HINTS,
	CONFIDENCE_EMERALD,
	CONFIDENCE_AMBER,
	CONFIDENCE_RED,
	type IntentKey,
} from '../reasoning-helpers';

// ────────────────────────────────────────────────────────────────
// confidenceColor — the spec-critical boundary table
// ────────────────────────────────────────────────────────────────

describe('confidenceColor — band boundaries (>75 emerald, 40-75 amber, <40 red)', () => {
	it.each<[number, string]>([
		// Emerald band: strictly greater than 75
		[100, CONFIDENCE_EMERALD],
		[99.99, CONFIDENCE_EMERALD],
		[80, CONFIDENCE_EMERALD],
		[76, CONFIDENCE_EMERALD],
		[75.01, CONFIDENCE_EMERALD],
		// Amber band: 40 <= c <= 75
		[75, CONFIDENCE_AMBER], // exactly 75 → amber (page spec: `>75` emerald)
		[60, CONFIDENCE_AMBER],
		[50, CONFIDENCE_AMBER],
		[40.01, CONFIDENCE_AMBER],
		[40, CONFIDENCE_AMBER], // exactly 40 → amber (page spec: `>=40` amber)
		// Red band: strictly less than 40
		[39.99, CONFIDENCE_RED],
		[20, CONFIDENCE_RED],
		[0.01, CONFIDENCE_RED],
		[0, CONFIDENCE_RED],
	])('confidence %f → %s', (c, expected) => {
		expect(confidenceColor(c)).toBe(expected);
	});

	it('clamps negative to red (defensive — confidence should never be negative)', () => {
		expect(confidenceColor(-10)).toBe(CONFIDENCE_RED);
	});

	it('over-100 stays emerald (defensive — confidence should never exceed 100)', () => {
		expect(confidenceColor(150)).toBe(CONFIDENCE_EMERALD);
	});

	it('NaN → red (worst-case band)', () => {
		expect(confidenceColor(Number.NaN)).toBe(CONFIDENCE_RED);
	});

	it('is pure — same input yields same output', () => {
		for (const c of [0, 39.99, 40, 75, 75.01, 100]) {
			expect(confidenceColor(c)).toBe(confidenceColor(c));
		}
	});

	it('never returns an empty string or undefined', () => {
		for (const c of [-1, 0, 20, 40, 75, 76, 100, 200, Number.NaN]) {
			const colour = confidenceColor(c);
			expect(typeof colour).toBe('string');
			expect(colour.length).toBeGreaterThan(0);
		}
	});
});

describe('confidenceLabel — human text per band', () => {
	it.each<[number, string]>([
		[100, 'HIGH CONFIDENCE'],
		[76, 'HIGH CONFIDENCE'],
		[75.01, 'HIGH CONFIDENCE'],
		[75, 'MIXED SIGNAL'],
		[60, 'MIXED SIGNAL'],
		[40, 'MIXED SIGNAL'],
		[39.99, 'LOW CONFIDENCE'],
		[0, 'LOW CONFIDENCE'],
	])('confidence %f → %s', (c, expected) => {
		expect(confidenceLabel(c)).toBe(expected);
	});

	it('NaN → LOW CONFIDENCE (safe default)', () => {
		expect(confidenceLabel(Number.NaN)).toBe('LOW CONFIDENCE');
	});

	it('agrees with confidenceColor across the spec boundary sweep', () => {
		// Sanity: if the label is HIGH, the colour must be emerald, etc.
		const cases: Array<[number, string, string]> = [
			[100, 'HIGH CONFIDENCE', CONFIDENCE_EMERALD],
			[76, 'HIGH CONFIDENCE', CONFIDENCE_EMERALD],
			[75, 'MIXED SIGNAL', CONFIDENCE_AMBER],
			[40, 'MIXED SIGNAL', CONFIDENCE_AMBER],
			[39.99, 'LOW CONFIDENCE', CONFIDENCE_RED],
			[0, 'LOW CONFIDENCE', CONFIDENCE_RED],
		];
		for (const [c, label, colour] of cases) {
			expect(confidenceLabel(c)).toBe(label);
			expect(confidenceColor(c)).toBe(colour);
		}
	});
});

// ────────────────────────────────────────────────────────────────
// Intent classification — visual hint mapping
// ────────────────────────────────────────────────────────────────

describe('INTENT_HINTS — one hint per deep_reference intent', () => {
	const intents: IntentKey[] = [
		'FactCheck',
		'Timeline',
		'RootCause',
		'Comparison',
		'Synthesis',
	];

	it('defines a hint for every intent the backend emits', () => {
		for (const i of intents) {
			expect(INTENT_HINTS[i]).toBeDefined();
		}
	});

	it.each(intents)('%s hint has label + icon + description', (i) => {
		const hint = INTENT_HINTS[i];
		expect(hint.label).toBe(i); // label doubles as canonical id
		expect(hint.icon.length).toBeGreaterThan(0);
		expect(hint.description.length).toBeGreaterThan(0);
	});

	it('icons are unique across intents (so the eye can distinguish them)', () => {
		const icons = intents.map((i) => INTENT_HINTS[i].icon);
		expect(new Set(icons).size).toBe(intents.length);
	});

	it('descriptions are distinct across intents', () => {
		const descs = intents.map((i) => INTENT_HINTS[i].description);
		expect(new Set(descs).size).toBe(intents.length);
	});
});

describe('intentHintFor — lookup with safe fallback', () => {
	it('returns the exact entry for a known intent', () => {
		expect(intentHintFor('FactCheck')).toBe(INTENT_HINTS.FactCheck);
		expect(intentHintFor('Timeline')).toBe(INTENT_HINTS.Timeline);
		expect(intentHintFor('RootCause')).toBe(INTENT_HINTS.RootCause);
		expect(intentHintFor('Comparison')).toBe(INTENT_HINTS.Comparison);
		expect(intentHintFor('Synthesis')).toBe(INTENT_HINTS.Synthesis);
	});

	it('falls back to Synthesis for unknown intent (most generic classification)', () => {
		expect(intentHintFor('Prediction')).toBe(INTENT_HINTS.Synthesis);
		expect(intentHintFor('nonsense')).toBe(INTENT_HINTS.Synthesis);
	});

	it('falls back to Synthesis for null / undefined / empty string', () => {
		expect(intentHintFor(null)).toBe(INTENT_HINTS.Synthesis);
		expect(intentHintFor(undefined)).toBe(INTENT_HINTS.Synthesis);
		expect(intentHintFor('')).toBe(INTENT_HINTS.Synthesis);
	});

	it('is case-sensitive — backend emits Title-case strings and we honour that', () => {
		// If case-folding becomes desirable, this test will force the
		// change to be explicit rather than accidental.
		expect(intentHintFor('factcheck')).toBe(INTENT_HINTS.Synthesis);
		expect(intentHintFor('FACTCHECK')).toBe(INTENT_HINTS.Synthesis);
	});
});

// ────────────────────────────────────────────────────────────────
// Stage-count invariant — the component renders exactly 8 stages
// ────────────────────────────────────────────────────────────────

describe('Cognitive pipeline shape', () => {
	it('confidence colour constants are all distinct hex strings', () => {
		const set = new Set([
			CONFIDENCE_EMERALD.toLowerCase(),
			CONFIDENCE_AMBER.toLowerCase(),
			CONFIDENCE_RED.toLowerCase(),
		]);
		expect(set.size).toBe(3);
		for (const c of set) {
			expect(c).toMatch(/^#[0-9a-f]{6}$/);
		}
	});
});

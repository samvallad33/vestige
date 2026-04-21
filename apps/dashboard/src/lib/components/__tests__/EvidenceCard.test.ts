/**
 * EvidenceCard — pure-logic coverage.
 *
 * The component itself mounts Svelte, which vitest cannot do in a node
 * environment. Every piece of logic that was reachable via props has been
 * extracted to `reasoning-helpers.ts`; this file exhaustively exercises
 * those helpers through the same import surface EvidenceCard uses. If
 * this file is green, the card's visual output is a 1:1 function of the
 * helper output.
 */
import { describe, it, expect } from 'vitest';

import {
	ROLE_META,
	roleMetaFor,
	trustColor,
	trustPercent,
	clampTrust,
	nodeTypeColor,
	formatDate,
	shortenId,
	CONFIDENCE_EMERALD,
	CONFIDENCE_AMBER,
	CONFIDENCE_RED,
	DEFAULT_NODE_TYPE_COLOR,
	type EvidenceRole,
} from '../reasoning-helpers';
import { NODE_TYPE_COLORS } from '$types';

// ────────────────────────────────────────────────────────────────
// clampTrust + trustPercent — numeric contract
// ────────────────────────────────────────────────────────────────

describe('clampTrust — 0-1 display range', () => {
	it.each<[number, number]>([
		[0, 0],
		[0.5, 0.5],
		[1, 1],
		[-0.1, 0],
		[-1, 0],
		[1.2, 1],
		[999, 1],
	])('clamps %f → %f', (input, expected) => {
		expect(clampTrust(input)).toBe(expected);
	});

	it('returns 0 for NaN (defensive — avoids NaN% in the UI)', () => {
		expect(clampTrust(Number.NaN)).toBe(0);
	});

	it('returns 0 for non-finite inputs (+/-Infinity) — safe default', () => {
		// Infinity indicates upstream garbage — degrade to empty bar rather
		// than saturate the UI to 100%.
		expect(clampTrust(-Infinity)).toBe(0);
		expect(clampTrust(Infinity)).toBe(0);
	});

	it('is idempotent (clamp of clamp is the same)', () => {
		for (const v of [-0.5, 0, 0.3, 0.75, 1, 2]) {
			expect(clampTrust(clampTrust(v))).toBe(clampTrust(v));
		}
	});
});

describe('trustPercent — 0-100 rendering', () => {
	it.each<[number, number]>([
		[0, 0],
		[0.5, 50],
		[1, 100],
		[-0.1, 0],
		[1.2, 100],
	])('converts trust %f → %f%%', (t, expected) => {
		expect(trustPercent(t)).toBe(expected);
	});

	it('handles NaN without producing NaN', () => {
		expect(trustPercent(Number.NaN)).toBe(0);
	});
});

// ────────────────────────────────────────────────────────────────
// trustColor — band boundaries for the card's trust bar
// ────────────────────────────────────────────────────────────────

describe('trustColor — boundary analysis', () => {
	it.each<[number, string]>([
		// Emerald band: strictly > 0.75 → > 75%
		[1.0, CONFIDENCE_EMERALD],
		[0.9, CONFIDENCE_EMERALD],
		[0.751, CONFIDENCE_EMERALD],
		// Amber band: 0.40 ≤ t ≤ 0.75
		[0.75, CONFIDENCE_AMBER], // boundary — amber at exactly 75%
		[0.5, CONFIDENCE_AMBER],
		[0.4, CONFIDENCE_AMBER], // boundary — amber at exactly 40%
		// Red band: < 0.40
		[0.399, CONFIDENCE_RED],
		[0.2, CONFIDENCE_RED],
		[0, CONFIDENCE_RED],
	])('trust %f → %s', (t, expected) => {
		expect(trustColor(t)).toBe(expected);
	});

	it('clamps negative to red and super-high to emerald (defensive)', () => {
		expect(trustColor(-0.5)).toBe(CONFIDENCE_RED);
		expect(trustColor(1.5)).toBe(CONFIDENCE_EMERALD);
	});

	it('returns red for NaN (lowest-confidence fallback)', () => {
		expect(trustColor(Number.NaN)).toBe(CONFIDENCE_RED);
	});
});

// ────────────────────────────────────────────────────────────────
// Role metadata — label + accent + icon
// ────────────────────────────────────────────────────────────────

describe('ROLE_META — completeness and shape', () => {
	const roles: EvidenceRole[] = ['primary', 'supporting', 'contradicting', 'superseded'];

	it('defines an entry for every role', () => {
		for (const r of roles) {
			expect(ROLE_META[r]).toBeDefined();
		}
	});

	it.each(roles)('%s has non-empty label + icon', (r) => {
		const meta = ROLE_META[r];
		expect(meta.label.length).toBeGreaterThan(0);
		expect(meta.icon.length).toBeGreaterThan(0);
	});

	it('maps to the expected accent tokens used by Tailwind (synapse/recall/decay/muted)', () => {
		expect(ROLE_META.primary.accent).toBe('synapse');
		expect(ROLE_META.supporting.accent).toBe('recall');
		expect(ROLE_META.contradicting.accent).toBe('decay');
		expect(ROLE_META.superseded.accent).toBe('muted');
	});

	it('accents are unique across roles (each role is visually distinct)', () => {
		const accents = roles.map((r) => ROLE_META[r].accent);
		expect(new Set(accents).size).toBe(4);
	});

	it('icons are unique across roles', () => {
		const icons = roles.map((r) => ROLE_META[r].icon);
		expect(new Set(icons).size).toBe(4);
	});

	it('labels are human-readable (first letter capital, no accents on the word)', () => {
		for (const r of roles) {
			const label = ROLE_META[r].label;
			expect(label[0]).toBe(label[0].toUpperCase());
		}
	});
});

describe('roleMetaFor — lookup with defensive fallback', () => {
	it('returns the exact entry for a known role', () => {
		expect(roleMetaFor('primary')).toBe(ROLE_META.primary);
		expect(roleMetaFor('contradicting')).toBe(ROLE_META.contradicting);
	});

	it('falls back to Supporting when handed an unknown role (deep_reference could add new ones)', () => {
		expect(roleMetaFor('unknown-role')).toBe(ROLE_META.supporting);
		expect(roleMetaFor('')).toBe(ROLE_META.supporting);
	});
});

// ────────────────────────────────────────────────────────────────
// nodeTypeColor — palette lookup with fallback
// ────────────────────────────────────────────────────────────────

describe('nodeTypeColor — palette lookup', () => {
	it('returns the fallback colour when nodeType is undefined/null/empty', () => {
		expect(nodeTypeColor(undefined)).toBe(DEFAULT_NODE_TYPE_COLOR);
		expect(nodeTypeColor(null)).toBe(DEFAULT_NODE_TYPE_COLOR);
		expect(nodeTypeColor('')).toBe(DEFAULT_NODE_TYPE_COLOR);
	});

	it('returns the palette entry for every known NODE_TYPE_COLORS key', () => {
		for (const [type, colour] of Object.entries(NODE_TYPE_COLORS)) {
			expect(nodeTypeColor(type)).toBe(colour);
		}
	});

	it('returns the fallback for an unknown nodeType', () => {
		expect(nodeTypeColor('quantum-state')).toBe(DEFAULT_NODE_TYPE_COLOR);
	});
});

// ────────────────────────────────────────────────────────────────
// formatDate — invalid-date handling (the real bug fixed here)
// ────────────────────────────────────────────────────────────────

describe('formatDate — ISO parsing with graceful degradation', () => {
	it('formats a valid ISO date into a locale string', () => {
		const out = formatDate('2026-04-20T12:00:00.000Z', 'en-US');
		// Example: "Apr 20, 2026"
		expect(out).toMatch(/2026/);
		expect(out).toMatch(/Apr/);
	});

	it('returns em-dash for empty / null / undefined', () => {
		expect(formatDate('')).toBe('—');
		expect(formatDate(null)).toBe('—');
		expect(formatDate(undefined)).toBe('—');
		expect(formatDate('   ')).toBe('—');
	});

	it('returns the original string when the input is unparseable (never "Invalid Date")', () => {
		// Regression: `new Date('not-a-date').toLocaleDateString()` returned
		// the literal text "Invalid Date" — EvidenceCard rendered that. Now
		// we surface the raw string so a reviewer can tell it was garbage.
		const garbage = 'not-a-date';
		expect(formatDate(garbage)).toBe(garbage);
		expect(formatDate(garbage)).not.toBe('Invalid Date');
	});

	it('handles ISO dates without time component', () => {
		const out = formatDate('2026-01-15', 'en-US');
		expect(out).toMatch(/2026/);
	});

	it('is pure — no global mutation between calls', () => {
		const a = formatDate('2026-04-20T00:00:00.000Z', 'en-US');
		const b = formatDate('2026-04-20T00:00:00.000Z', 'en-US');
		expect(a).toBe(b);
	});
});

// ────────────────────────────────────────────────────────────────
// shortenId — UUID → #abcdef01
// ────────────────────────────────────────────────────────────────

describe('shortenId — 8-char display prefix', () => {
	it('returns an 8-char prefix for a standard UUID', () => {
		expect(shortenId('a1b2c3d4-e5f6-0000-0000-000000000000')).toBe('a1b2c3d4');
	});

	it('returns the full string when already ≤ 8 chars', () => {
		expect(shortenId('abc')).toBe('abc');
		expect(shortenId('12345678')).toBe('12345678');
	});

	it('handles null/undefined/empty gracefully', () => {
		expect(shortenId(null)).toBe('');
		expect(shortenId(undefined)).toBe('');
		expect(shortenId('')).toBe('');
	});

	it('respects a custom length parameter', () => {
		expect(shortenId('abcdefghij', 4)).toBe('abcd');
		expect(shortenId('abcdefghij', 10)).toBe('abcdefghij');
	});
});

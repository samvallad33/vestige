import { describe, it, expect } from 'vitest';
import { pbrNeutralReference, VOID_CLEAR_HDR } from '../post/tone-reference';
import { BLOOM_STRENGTH } from '../post/post-chain';

const hexToRgb = (hex: string): [number, number, number] => [
	parseInt(hex.slice(1, 3), 16) / 255,
	parseInt(hex.slice(3, 5), 16) / 255,
	parseInt(hex.slice(5, 7), 16) / 255
];

const argmax = (v: readonly number[]) => v.indexOf(Math.max(...v));
const argmin = (v: readonly number[]) => v.indexOf(Math.min(...v));

describe('pbrNeutralReference', () => {
	it('void gate: tonemap(clear · (1 + BLOOM_STRENGTH)) is exactly #05060a', () => {
		// The normalized bloom chain has flat-field gain exactly 1, so a void
		// pixel enters the tonemap as (1 + BLOOM_STRENGTH) · VOID_CLEAR_HDR.
		const s = 1 + BLOOM_STRENGTH;
		const out = pbrNeutralReference([
			VOID_CLEAR_HDR.r * s,
			VOID_CLEAR_HDR.g * s,
			VOID_CLEAR_HDR.b * s
		]);
		expect(Math.abs(out[0] - 5 / 255)).toBeLessThan(1e-6);
		expect(Math.abs(out[1] - 6 / 255)).toBeLessThan(1e-6);
		expect(Math.abs(out[2] - 10 / 255)).toBeLessThan(1e-6);
	});

	it('void preimage stays below the compression knee (early-return branch)', () => {
		const s = 1 + BLOOM_STRENGTH;
		const peak = Math.max(VOID_CLEAR_HDR.r, VOID_CLEAR_HDR.g, VOID_CLEAR_HDR.b) * s;
		expect(peak).toBeLessThan(0.76);
	});

	it('below the knee is NOT identity: uniform −0.04 offset when min ≥ 0.08', () => {
		// (0.5, 0.3, 0.2): min = 0.2 ≥ 0.08 → offset = 0.04, peak 0.46 < 0.76.
		const out = pbrNeutralReference([0.5, 0.3, 0.2]);
		expect(out[0]).toBeCloseTo(0.46, 12);
		expect(out[1]).toBeCloseTo(0.26, 12);
		expect(out[2]).toBeCloseTo(0.16, 12);
	});

	it('black offset: min < 0.08 → out_min = 6.25·min²', () => {
		// (0.05, 0.3, 0.2): offset = 0.05 − 6.25·0.05² = 0.034375.
		const out = pbrNeutralReference([0.05, 0.3, 0.2]);
		expect(out[0]).toBeCloseTo(6.25 * 0.05 * 0.05, 12);
		expect(out[1]).toBeCloseTo(0.3 - 0.034375, 12);
		expect(out[2]).toBeCloseTo(0.2 - 0.034375, 12);
	});

	it('FSRS mint #10b981 stays below the knee: pure offset, order preserved', () => {
		const rgb = hexToRgb('#10b981');
		const x = rgb[0]; // min channel (16/255 < 0.08)
		const offset = x - 6.25 * x * x;
		// peak after offset ≈ 0.687 < 0.76 → early return, deltas = offset.
		expect(Math.max(...rgb) - offset).toBeLessThan(0.76);
		const out = pbrNeutralReference(rgb);
		expect(out[0]).toBeCloseTo(rgb[0] - offset, 12);
		expect(out[1]).toBeCloseTo(rgb[1] - offset, 12);
		expect(out[2]).toBeCloseTo(rgb[2] - offset, 12);
		expect(argmax(out)).toBe(argmax(rgb));
		expect(argmin(out)).toBe(argmin(rgb));
	});

	it('FSRS amber #f59e0b and violet #8b5cf6 hit compression: hue order still preserved', () => {
		// NOTE deviation from the design brief, which claimed all three FSRS
		// colors take the below-knee branch: after the black offset, amber's
		// peak ≈ 0.929 and violet's ≈ 0.925 — both ≥ 0.76, so the compression
		// branch runs. Hue preservation (channel ordering) is the real guard.
		for (const hex of ['#f59e0b', '#8b5cf6']) {
			const rgb = hexToRgb(hex);
			const x = Math.min(...rgb);
			const offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
			expect(Math.max(...rgb) - offset).toBeGreaterThanOrEqual(0.76);
			const out = pbrNeutralReference(rgb);
			expect(argmax(out)).toBe(argmax(rgb));
			expect(argmin(out)).toBe(argmin(rgb));
			// Compressed: peak shrinks, everything stays inside (0, 1).
			expect(Math.max(...out)).toBeLessThan(Math.max(...rgb));
			for (const ch of out) {
				expect(ch).toBeGreaterThan(0);
				expect(ch).toBeLessThan(1);
			}
		}
	});

	it('compression: HDR greys 2/4/8 map monotonically, all below 1', () => {
		const g2 = pbrNeutralReference([2, 2, 2])[0];
		const g4 = pbrNeutralReference([4, 4, 4])[0];
		const g8 = pbrNeutralReference([8, 8, 8])[0];
		expect(g2).toBeLessThan(g4);
		expect(g4).toBeLessThan(g8);
		expect(g8).toBeLessThan(1);
		// Greys stay grey (hue-preserving on the neutral axis).
		expect(pbrNeutralReference([4, 4, 4])).toEqual([g4, g4, g4]);
	});

	it('hue: argmax stable through compression for (1.5, 0.4, 0.2)', () => {
		const out = pbrNeutralReference([1.5, 0.4, 0.2]);
		expect(argmax(out)).toBe(0);
		expect(Math.max(...out)).toBeLessThan(1);
	});
});

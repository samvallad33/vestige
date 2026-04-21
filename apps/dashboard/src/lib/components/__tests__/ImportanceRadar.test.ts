/**
 * Unit tests for importance-helpers — the pure logic backing
 * ImportanceRadar.svelte + importance/+page.svelte.
 *
 * Runs in the vitest `node` environment (no jsdom). We exercise:
 *   - Composite channel weighting (matches backend ImportanceSignals)
 *   - 4-axis radar vertex geometry (Novelty top / Arousal right / Reward
 *     bottom / Attention left)
 *   - Value clamping at the helper boundary (defensive against a mis-
 *     scaled /api/importance response)
 *   - Size-preset mapping (sm 80 / md 180 / lg 320)
 *   - Trending-memory importance proxy (retention × log(reviews) / √age)
 *     including the age=0 division-by-zero edge case.
 */

import { describe, it, expect } from 'vitest';
import {
	clamp01,
	clampChannels,
	compositeScore,
	CHANNEL_WEIGHTS,
	sizePreset,
	radarRadius,
	radarVertices,
	verticesToPath,
	importanceProxy,
	rankByProxy,
	AXIS_ORDER,
	SIZE_PX,
	type ProxyMemoryLike,
} from '../importance-helpers';

// ===========================================================================
// clamp01
// ===========================================================================

describe('clamp01', () => {
	it('passes in-range values through', () => {
		expect(clamp01(0)).toBe(0);
		expect(clamp01(0.5)).toBe(0.5);
		expect(clamp01(1)).toBe(1);
	});

	it('clamps below zero to 0', () => {
		expect(clamp01(-0.3)).toBe(0);
		expect(clamp01(-100)).toBe(0);
	});

	it('clamps above one to 1', () => {
		expect(clamp01(1.0001)).toBe(1);
		expect(clamp01(42)).toBe(1);
	});

	it('folds null / undefined / NaN / Infinity to 0', () => {
		expect(clamp01(null)).toBe(0);
		expect(clamp01(undefined)).toBe(0);
		expect(clamp01(NaN)).toBe(0);
		expect(clamp01(Infinity)).toBe(0);
		expect(clamp01(-Infinity)).toBe(0);
	});
});

describe('clampChannels', () => {
	it('clamps every channel independently', () => {
		expect(clampChannels({ novelty: 2, arousal: -1, reward: 0.5, attention: NaN })).toEqual({
			novelty: 1,
			arousal: 0,
			reward: 0.5,
			attention: 0,
		});
	});

	it('fills missing channels with 0', () => {
		expect(clampChannels({ novelty: 0.8 })).toEqual({
			novelty: 0.8,
			arousal: 0,
			reward: 0,
			attention: 0,
		});
	});

	it('accepts null / undefined as "all zeros"', () => {
		expect(clampChannels(null)).toEqual({ novelty: 0, arousal: 0, reward: 0, attention: 0 });
		expect(clampChannels(undefined)).toEqual({
			novelty: 0,
			arousal: 0,
			reward: 0,
			attention: 0,
		});
	});
});

// ===========================================================================
// compositeScore — MUST match backend ImportanceSignals weights
// ===========================================================================

describe('compositeScore', () => {
	it('sums channel contributions with the documented weights', () => {
		const c = { novelty: 1, arousal: 1, reward: 1, attention: 1 };
		// 0.25 + 0.30 + 0.25 + 0.20 = 1.00
		expect(compositeScore(c)).toBeCloseTo(1.0, 5);
	});

	it('is zero for all-zero channels', () => {
		expect(compositeScore({ novelty: 0, arousal: 0, reward: 0, attention: 0 })).toBe(0);
	});

	it('weights match CHANNEL_WEIGHTS exactly (backend contract)', () => {
		expect(CHANNEL_WEIGHTS).toEqual({
			novelty: 0.25,
			arousal: 0.3,
			reward: 0.25,
			attention: 0.2,
		});
		// Weights sum to 1 — any drift here and the "composite ∈ [0,1]"
		// invariant falls over.
		const sum =
			CHANNEL_WEIGHTS.novelty +
			CHANNEL_WEIGHTS.arousal +
			CHANNEL_WEIGHTS.reward +
			CHANNEL_WEIGHTS.attention;
		expect(sum).toBeCloseTo(1.0, 10);
	});

	it('matches the exact weighted formula per channel', () => {
		// 0.4·0.25 + 0.6·0.30 + 0.2·0.25 + 0.8·0.20
		//  = 0.10 + 0.18 + 0.05 + 0.16 = 0.49
		expect(
			compositeScore({ novelty: 0.4, arousal: 0.6, reward: 0.2, attention: 0.8 }),
		).toBeCloseTo(0.49, 5);
	});

	it('clamps inputs before weighting (never escapes [0,1])', () => {
		// All over-max → should pin to 1, not to 2.
		expect(
			compositeScore({ novelty: 2, arousal: 2, reward: 2, attention: 2 }),
		).toBeCloseTo(1.0, 5);
		// Negative channels count as 0.
		expect(
			compositeScore({ novelty: -1, arousal: -1, reward: -1, attention: -1 }),
		).toBe(0);
	});
});

// ===========================================================================
// Size preset
// ===========================================================================

describe('sizePreset', () => {
	it('maps the three documented presets', () => {
		expect(sizePreset('sm')).toBe(80);
		expect(sizePreset('md')).toBe(180);
		expect(sizePreset('lg')).toBe(320);
	});

	it('exposes the SIZE_PX mapping for external consumers', () => {
		expect(SIZE_PX).toEqual({ sm: 80, md: 180, lg: 320 });
	});

	it('falls back to md (180) for unknown / missing keys', () => {
		expect(sizePreset(undefined)).toBe(180);
		expect(sizePreset('' as unknown as 'md')).toBe(180);
		expect(sizePreset('xl' as unknown as 'md')).toBe(180);
	});
});

// ===========================================================================
// radarRadius — component padding rules
// ===========================================================================

describe('radarRadius', () => {
	it('applies the correct padding per preset', () => {
		// sm: 80/2 - 4  = 36
		// md: 180/2 - 28 = 62
		// lg: 320/2 - 44 = 116
		expect(radarRadius('sm')).toBe(36);
		expect(radarRadius('md')).toBe(62);
		expect(radarRadius('lg')).toBe(116);
	});

	it('never returns a negative radius', () => {
		// Can't construct a sub-zero radius via normal presets, but the
		// helper floors at 0 defensively.
		expect(radarRadius('md')).toBeGreaterThanOrEqual(0);
	});
});

// ===========================================================================
// radarVertices — 4 SVG polygon points on the fixed axis order
// ===========================================================================

describe('radarVertices', () => {
	it('emits vertices in Novelty→Arousal→Reward→Attention order', () => {
		expect(AXIS_ORDER.map((a) => a.key)).toEqual([
			'novelty',
			'arousal',
			'reward',
			'attention',
		]);
	});

	it('places a 0-valued channel at the centre', () => {
		// Centre for md is (90, 90). novelty=0 means the top vertex sits AT
		// the centre — the polygon pinches inward.
		const v = radarVertices(
			{ novelty: 0, arousal: 0, reward: 0, attention: 0 },
			'md',
		);
		expect(v).toHaveLength(4);
		for (const p of v) {
			expect(p.x).toBeCloseTo(90, 5);
			expect(p.y).toBeCloseTo(90, 5);
		}
	});

	it('places a 1-valued channel on the correct axis edge', () => {
		// Size md: cx=cy=90, r=62.
		//   Novelty (angle -π/2, top)    → (90, 90 - 62) = (90, 28)
		//   Arousal (angle 0, right)     → (90 + 62, 90) = (152, 90)
		//   Reward  (angle π/2, bottom)  → (90, 90 + 62) = (90, 152)
		//   Attention (angle π, left)    → (90 - 62, 90) = (28, 90)
		const v = radarVertices(
			{ novelty: 1, arousal: 1, reward: 1, attention: 1 },
			'md',
		);
		expect(v[0].x).toBeCloseTo(90, 5);
		expect(v[0].y).toBeCloseTo(28, 5);

		expect(v[1].x).toBeCloseTo(152, 5);
		expect(v[1].y).toBeCloseTo(90, 5);

		expect(v[2].x).toBeCloseTo(90, 5);
		expect(v[2].y).toBeCloseTo(152, 5);

		expect(v[3].x).toBeCloseTo(28, 5);
		expect(v[3].y).toBeCloseTo(90, 5);
	});

	it('scales vertex radial distance linearly with the channel value', () => {
		// Arousal at 0.5 should land half-way from centre to the right edge.
		const v = radarVertices(
			{ novelty: 0, arousal: 0.5, reward: 0, attention: 0 },
			'md',
		);
		// radius=62, so right vertex x = 90 + 62*0.5 = 121.
		expect(v[1].x).toBeCloseTo(121, 5);
		expect(v[1].y).toBeCloseTo(90, 5);
	});

	it('clamps out-of-range inputs rather than exiting the SVG box', () => {
		// novelty=2 should pin to the edge (not overshoot to 90 - 124 = -34).
		const v = radarVertices(
			{ novelty: 2, arousal: -0.5, reward: NaN, attention: Infinity },
			'md',
		);
		// Novelty pinned to edge (y=28), arousal/reward/attention at 0 land at centre.
		expect(v[0].y).toBeCloseTo(28, 5);
		expect(v[1].x).toBeCloseTo(90, 5); // arousal=0 → centre
		expect(v[2].y).toBeCloseTo(90, 5); // reward=0 → centre
		expect(v[3].x).toBeCloseTo(90, 5); // attention=0 → centre
	});

	it('respects the active size preset', () => {
		// At sm (80px), radius=36. Novelty=1 → (40, 40-36) = (40, 4).
		const v = radarVertices({ novelty: 1, arousal: 0, reward: 0, attention: 0 }, 'sm');
		expect(v[0].x).toBeCloseTo(40, 5);
		expect(v[0].y).toBeCloseTo(4, 5);
	});
});

describe('verticesToPath', () => {
	it('serialises to an SVG path with M/L commands and Z close', () => {
		const path = verticesToPath([
			{ x: 10, y: 20 },
			{ x: 30, y: 40 },
			{ x: 50, y: 60 },
			{ x: 70, y: 80 },
		]);
		expect(path).toBe('M10.00,20.00 L30.00,40.00 L50.00,60.00 L70.00,80.00 Z');
	});

	it('returns an empty string for no points', () => {
		expect(verticesToPath([])).toBe('');
	});
});

// ===========================================================================
// importanceProxy — "Top Important Memories This Week" ranking formula
// ===========================================================================

describe('importanceProxy', () => {
	// Anchor everything to a fixed "now" so recency math is deterministic.
	const NOW = new Date('2026-04-20T12:00:00Z').getTime();

	function mem(over: Partial<ProxyMemoryLike>): ProxyMemoryLike {
		return {
			retentionStrength: 0.5,
			reviewCount: 0,
			createdAt: new Date(NOW - 2 * 86_400_000).toISOString(),
			...over,
		};
	}

	it('is zero for zero retention', () => {
		expect(importanceProxy(mem({ retentionStrength: 0 }), NOW)).toBe(0);
	});

	it('treats missing reviewCount as 0 (not a crash)', () => {
		const m = mem({ reviewCount: undefined, retentionStrength: 0.8 });
		const v = importanceProxy(m, NOW);
		expect(v).toBeGreaterThan(0);
		expect(Number.isFinite(v)).toBe(true);
	});

	it('matches the documented formula: retention × log1p(reviews+1) / √age', () => {
		// createdAt = 4 days before NOW → ageDays = 4, √4 = 2.
		// retention = 0.6, reviews = 3 → log1p(4) ≈ 1.6094
		// expected = 0.6 × 1.6094 / 2 ≈ 0.4828
		const m = mem({
			retentionStrength: 0.6,
			reviewCount: 3,
			createdAt: new Date(NOW - 4 * 86_400_000).toISOString(),
		});
		const v = importanceProxy(m, NOW);
		const expected = (0.6 * Math.log1p(4)) / 2;
		expect(v).toBeCloseTo(expected, 6);
	});

	it('clamps age to 1 day for a memory created RIGHT NOW (div-by-zero guard)', () => {
		// createdAt equals NOW → raw ageDays = 0. Without the clamp, the
		// recency boost would divide by zero. We assert the helper returns
		// a finite value equal to the "age=1" path.
		const zeroAge = importanceProxy(
			mem({
				retentionStrength: 0.5,
				reviewCount: 0,
				createdAt: new Date(NOW).toISOString(),
			}),
			NOW,
		);
		const oneDayAge = importanceProxy(
			mem({
				retentionStrength: 0.5,
				reviewCount: 0,
				createdAt: new Date(NOW - 1 * 86_400_000).toISOString(),
			}),
			NOW,
		);
		expect(Number.isFinite(zeroAge)).toBe(true);
		expect(zeroAge).toBeCloseTo(oneDayAge, 10);
	});

	it('also clamps future-dated memories to ageDays=1 rather than going negative', () => {
		const future = importanceProxy(
			mem({
				retentionStrength: 0.5,
				reviewCount: 0,
				createdAt: new Date(NOW + 7 * 86_400_000).toISOString(),
			}),
			NOW,
		);
		expect(Number.isFinite(future)).toBe(true);
		expect(future).toBeGreaterThan(0);
	});

	it('returns 0 for a malformed createdAt', () => {
		const m = {
			retentionStrength: 0.8,
			reviewCount: 3,
			createdAt: 'not-a-date',
		};
		expect(importanceProxy(m, NOW)).toBe(0);
	});

	it('returns 0 when retentionStrength is non-finite', () => {
		expect(importanceProxy(mem({ retentionStrength: NaN }), NOW)).toBe(0);
		expect(importanceProxy(mem({ retentionStrength: Infinity }), NOW)).toBe(0);
	});

	it('ranks recent + high-retention memories ahead of stale ones', () => {
		const fresh: ProxyMemoryLike = {
			retentionStrength: 0.9,
			reviewCount: 5,
			createdAt: new Date(NOW - 1 * 86_400_000).toISOString(),
		};
		const stale: ProxyMemoryLike = {
			retentionStrength: 0.9,
			reviewCount: 5,
			createdAt: new Date(NOW - 100 * 86_400_000).toISOString(),
		};
		expect(importanceProxy(fresh, NOW)).toBeGreaterThan(importanceProxy(stale, NOW));
	});
});

describe('rankByProxy', () => {
	const NOW = new Date('2026-04-20T12:00:00Z').getTime();

	it('sorts descending by the proxy score', () => {
		const items: (ProxyMemoryLike & { id: string })[] = [
			{ id: 'stale', retentionStrength: 0.9, reviewCount: 5, createdAt: new Date(NOW - 100 * 86_400_000).toISOString() },
			{ id: 'fresh', retentionStrength: 0.9, reviewCount: 5, createdAt: new Date(NOW - 1 * 86_400_000).toISOString() },
			{ id: 'dead', retentionStrength: 0.0, reviewCount: 0, createdAt: new Date(NOW - 2 * 86_400_000).toISOString() },
		];
		const ranked = rankByProxy(items, NOW);
		expect(ranked.map((r) => r.id)).toEqual(['fresh', 'stale', 'dead']);
	});

	it('does not mutate the input array', () => {
		const items: ProxyMemoryLike[] = [
			{ retentionStrength: 0.1, reviewCount: 0, createdAt: new Date(NOW - 10 * 86_400_000).toISOString() },
			{ retentionStrength: 0.9, reviewCount: 9, createdAt: new Date(NOW - 1 * 86_400_000).toISOString() },
		];
		const before = items.slice();
		rankByProxy(items, NOW);
		expect(items).toEqual(before);
	});
});

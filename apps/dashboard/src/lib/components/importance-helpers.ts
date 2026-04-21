/**
 * importance-helpers — Pure logic for the Importance Radar UI
 * (importance/+page.svelte + ImportanceRadar.svelte).
 *
 * Extracted so the radar geometry and importance-proxy maths can be unit-
 * tested in the vitest `node` environment without jsdom or Svelte harness.
 *
 * Contracts
 * ---------
 * - Backend channel weights (novelty 0.25, arousal 0.30, reward 0.25,
 *   attention 0.20) sum to 1.0 and mirror ImportanceSignals in vestige-core.
 * - `clamp01` folds NaN/Infinity/nullish → 0 and clips [0,1].
 * - `radarVertices` emits 4 SVG polygon points in the fixed axis order
 *   Novelty (top) → Arousal (right) → Reward (bottom) → Attention (left).
 *   A zero value places the vertex at centre; a one value places it at the
 *   unit-ring edge.
 * - `importanceProxy` is the SAME formula the page uses to rank the weekly
 *   list: retentionStrength × log1p(reviews + 1) / sqrt(max(1, ageDays)).
 *   Age is clamped to 1 so a freshly-created memory never divides by zero.
 * - `sizePreset` maps 'sm'|'md'|'lg' to 80|180|320 and defaults to 'md' for
 *   any unknown size key — matching the component's default prop.
 */

// -- Channel model ----------------------------------------------------------

export type ChannelKey = 'novelty' | 'arousal' | 'reward' | 'attention';

/** Weights applied server-side by ImportanceSignals. Must sum to 1.0. */
export const CHANNEL_WEIGHTS: Readonly<Record<ChannelKey, number>> = {
	novelty: 0.25,
	arousal: 0.3,
	reward: 0.25,
	attention: 0.2,
} as const;

export interface Channels {
	novelty: number;
	arousal: number;
	reward: number;
	attention: number;
}

/** Clamp a value to [0,1]. Null / undefined / NaN / Infinity → 0. */
export function clamp01(v: number | null | undefined): number {
	if (v === null || v === undefined) return 0;
	if (!Number.isFinite(v)) return 0;
	if (v < 0) return 0;
	if (v > 1) return 1;
	return v;
}

/** Clamp every channel to [0,1]. Safe for partial / malformed inputs. */
export function clampChannels(ch: Partial<Channels> | null | undefined): Channels {
	return {
		novelty: clamp01(ch?.novelty),
		arousal: clamp01(ch?.arousal),
		reward: clamp01(ch?.reward),
		attention: clamp01(ch?.attention),
	};
}

/**
 * Composite importance score — matches backend ImportanceSignals.
 *
 *   composite = 0.25·novelty + 0.30·arousal + 0.25·reward + 0.20·attention
 *
 * Every input is clamped first so out-of-range channels never puncture the
 * 0..1 composite range. The return value is guaranteed to be in [0,1].
 */
export function compositeScore(ch: Partial<Channels> | null | undefined): number {
	const c = clampChannels(ch);
	return (
		c.novelty * CHANNEL_WEIGHTS.novelty +
		c.arousal * CHANNEL_WEIGHTS.arousal +
		c.reward * CHANNEL_WEIGHTS.reward +
		c.attention * CHANNEL_WEIGHTS.attention
	);
}

// -- Size preset ------------------------------------------------------------

export type RadarSize = 'sm' | 'md' | 'lg';

export const SIZE_PX: Readonly<Record<RadarSize, number>> = {
	sm: 80,
	md: 180,
	lg: 320,
} as const;

/**
 * Resolve a size preset key to its px value. Unknown / missing keys fall
 * back to 'md' (180), matching the component's default prop. `sm` loses
 * axis labels in the renderer but that's rendering concern, not ours.
 */
export function sizePreset(size: RadarSize | string | undefined): number {
	if (size && (size === 'sm' || size === 'md' || size === 'lg')) {
		return SIZE_PX[size];
	}
	return SIZE_PX.md;
}

// -- Geometry ---------------------------------------------------------------

/**
 * Fixed axis order. Angles use SVG conventions (y grows downward):
 *   Novelty   → angle -π/2 (top)
 *   Arousal   → angle   0  (right)
 *   Reward    → angle  π/2 (bottom)
 *   Attention → angle   π  (left)
 */
export const AXIS_ORDER: ReadonlyArray<{ key: ChannelKey; angle: number }> = [
	{ key: 'novelty', angle: -Math.PI / 2 },
	{ key: 'arousal', angle: 0 },
	{ key: 'reward', angle: Math.PI / 2 },
	{ key: 'attention', angle: Math.PI },
] as const;

export interface RadarPoint {
	x: number;
	y: number;
}

/**
 * Compute the effective drawable radius inside the SVG box. This mirrors the
 * component's padding logic:
 *   sm → padding 4 (edge-to-edge, no labels)
 *   md → padding 28
 *   lg → padding 44
 * Radius = size/2 − padding, floored at 0 (a radius below zero would draw
 * an inverted polygon — defensive guard).
 */
export function radarRadius(size: RadarSize | string | undefined): number {
	const px = sizePreset(size);
	let padding: number;
	switch (size) {
		case 'lg':
			padding = 44;
			break;
		case 'sm':
			padding = 4;
			break;
		default:
			padding = 28;
	}
	return Math.max(0, px / 2 - padding);
}

/**
 * Compute the 4 SVG polygon vertices for a set of channel values at a given
 * radar size. Values are clamped to [0,1] first so out-of-range inputs can't
 * escape the radar bounds.
 *
 * Ordering is FIXED and matches AXIS_ORDER: [novelty, arousal, reward, attention].
 * A zero value places the vertex at the centre (cx, cy); a one value places
 * it at the unit-ring edge.
 */
export function radarVertices(
	ch: Partial<Channels> | null | undefined,
	size: RadarSize | string | undefined = 'md',
): RadarPoint[] {
	const px = sizePreset(size);
	const r = radarRadius(size);
	const cx = px / 2;
	const cy = px / 2;
	const values = clampChannels(ch);
	return AXIS_ORDER.map(({ key, angle }) => {
		const v = values[key];
		return {
			x: cx + Math.cos(angle) * v * r,
			y: cy + Math.sin(angle) * v * r,
		};
	});
}

/** Serialise vertices to an SVG "M…L…L…L… Z" path, 2-decimal precision. */
export function verticesToPath(points: RadarPoint[]): string {
	if (points.length === 0) return '';
	return (
		points
			.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
			.join(' ') + ' Z'
	);
}

// -- Trending-memory proxy --------------------------------------------------

export interface ProxyMemoryLike {
	retentionStrength: number;
	reviewCount?: number | null;
	createdAt: string;
}

/**
 * Proxy score for the "Top Important Memories This Week" ranking. Exact
 * formula from importance/+page.svelte:
 *
 *   ageDays      = max(1, (now - createdAt) / 86_400_000)
 *   reviews      = reviewCount ?? 0
 *   recencyBoost = 1 / sqrt(ageDays)
 *   proxy        = retentionStrength × log1p(reviews + 1) × recencyBoost
 *
 * Edge cases:
 *   - createdAt is the current instant → ageDays clamps to 1 (no div-by-0)
 *   - createdAt is in the future       → negative age also clamps to 1
 *   - reviewCount null/undefined       → treated as 0
 *   - non-finite retentionStrength     → returns 0 defensively
 *
 * `now` is injectable for deterministic tests. Defaults to `Date.now()`.
 */
export function importanceProxy(m: ProxyMemoryLike, now: number = Date.now()): number {
	if (!m || !Number.isFinite(m.retentionStrength)) return 0;
	const created = new Date(m.createdAt).getTime();
	if (!Number.isFinite(created)) return 0;
	const ageDays = Math.max(1, (now - created) / 86_400_000);
	const reviews = m.reviewCount ?? 0;
	const recencyBoost = 1 / Math.sqrt(ageDays);
	return m.retentionStrength * Math.log1p(reviews + 1) * recencyBoost;
}

/** Sort memories by the proxy, descending. Stable via `.sort` on a copy. */
export function rankByProxy<M extends ProxyMemoryLike>(
	memories: readonly M[],
	now: number = Date.now(),
): M[] {
	return memories.slice().sort((a, b) => importanceProxy(b, now) - importanceProxy(a, now));
}

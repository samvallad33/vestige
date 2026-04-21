/**
 * dream-helpers — Pure logic for Dream Cinema UI.
 *
 * Extracted so we can test it without jsdom / Svelte component harnessing.
 * The Vitest setup for this package runs in a Node environment; every helper
 * in this module is a pure function of its inputs, so it can be exercised
 * directly in `__tests__/*.test.ts` alongside the graph helpers.
 */

/** Stage 1..5 of the 5-phase consolidation cycle. */
export const STAGE_COUNT = 5 as const;

/** Display names for each stage index (1-indexed). */
export const STAGE_NAMES = [
	'Replay',
	'Cross-reference',
	'Strengthen',
	'Prune',
	'Transfer',
] as const;

export type StageIndex = 1 | 2 | 3 | 4 | 5;

/**
 * Clamp an arbitrary integer to the valid 1..5 stage range. Accepts any
 * number (NaN, Infinity, negatives, floats) and always returns an integer
 * in [1,5]. NaN and non-finite values fall back to 1 — this matches the
 * "start at stage 1" behaviour on a fresh dream.
 */
export function clampStage(n: number): StageIndex {
	if (!Number.isFinite(n)) return 1;
	const i = Math.floor(n);
	if (i < 1) return 1;
	if (i > STAGE_COUNT) return STAGE_COUNT;
	return i as StageIndex;
}

/**
 * Get the human-readable stage name for a (possibly invalid) stage number.
 * Uses `clampStage`, so out-of-range inputs return the nearest valid name.
 */
export function stageName(n: number): string {
	return STAGE_NAMES[clampStage(n) - 1];
}

// ---------------------------------------------------------------------------
// Novelty classification — drives the gold-glow / muted styling on insight
// cards. Thresholds are STRICTLY exclusive so `0.3` and `0.7` map to the
// neutral band on purpose. See DreamInsightCard.svelte.
// ---------------------------------------------------------------------------

export type NoveltyBand = 'high' | 'neutral' | 'low';

/** Upper bound for the muted "low novelty" band. Values BELOW this are low. */
export const LOW_NOVELTY_THRESHOLD = 0.3;
/** Lower bound for the gold "high novelty" band. Values ABOVE this are high. */
export const HIGH_NOVELTY_THRESHOLD = 0.7;

/**
 * Classify a novelty score into one of 3 visual bands.
 *
 * Thresholds are exclusive on both sides:
 *   novelty > 0.7  → 'high' (gold glow)
 *   novelty < 0.3  → 'low'  (muted / desaturated)
 *   otherwise      → 'neutral'
 *
 * `null` / `undefined` / `NaN` collapse to 0 → 'low'.
 */
export function noveltyBand(novelty: number | null | undefined): NoveltyBand {
	const n = clamp01(novelty);
	if (n > HIGH_NOVELTY_THRESHOLD) return 'high';
	if (n < LOW_NOVELTY_THRESHOLD) return 'low';
	return 'neutral';
}

/** Clamp a value into [0,1]. `null`/`undefined`/`NaN` → 0. */
export function clamp01(n: number | null | undefined): number {
	if (n === null || n === undefined || !Number.isFinite(n)) return 0;
	if (n < 0) return 0;
	if (n > 1) return 1;
	return n;
}

// ---------------------------------------------------------------------------
// Formatting helpers — mirror what the page + card render. Keeping these
// pure lets us test the exact output strings without rendering Svelte.
// ---------------------------------------------------------------------------

/**
 * Format a millisecond duration as a human-readable string.
 *   < 1000ms  → "{n}ms"          (e.g. "0ms", "500ms")
 *   ≥ 1000ms  → "{n.nn}s"        (e.g. "1.50s", "15.00s")
 * Negative / NaN values collapse to "0ms".
 */
export function formatDurationMs(ms: number | null | undefined): string {
	if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) {
		return '0ms';
	}
	if (ms < 1000) return `${Math.round(ms)}ms`;
	return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Format a 0..1 confidence as a whole-percent string ("0%", "50%", "100%").
 * Values outside [0,1] clamp first. Uses `Math.round` so 0.505 → "51%".
 */
export function formatConfidencePct(confidence: number | null | undefined): string {
	const c = clamp01(confidence);
	return `${Math.round(c * 100)}%`;
}

// ---------------------------------------------------------------------------
// Source memory link formatting.
// ---------------------------------------------------------------------------

/**
 * Build the href for a source memory link. We keep this behind a helper so
 * the route format is tested in one place. `base` corresponds to SvelteKit's
 * `$app/paths` base (may be ""). Invalid IDs still produce a URL — route
 * handling is the page's responsibility, not ours.
 */
export function sourceMemoryHref(id: string, base = ''): string {
	return `${base}/memories/${id}`;
}

/**
 * Return the first N source memory IDs from an insight's `sourceMemories`
 * array, safely handling null / undefined / empty. Default N = 2, matching
 * the card's "first 2 links" behaviour.
 */
export function firstSourceIds(
	sources: readonly string[] | null | undefined,
	n = 2,
): string[] {
	if (!sources || sources.length === 0) return [];
	return sources.slice(0, Math.max(0, n));
}

/** Count of sources beyond the first N. Used for the "(+N)" suffix. */
export function extraSourceCount(
	sources: readonly string[] | null | undefined,
	shown = 2,
): number {
	if (!sources) return 0;
	return Math.max(0, sources.length - shown);
}

/**
 * Truncate a memory UUID for display on the chip. Matches the previous
 * inline `shortId` logic: first 8 chars, or the whole string if shorter.
 */
export function shortMemoryId(id: string): string {
	if (!id) return '';
	return id.length > 8 ? id.slice(0, 8) : id;
}

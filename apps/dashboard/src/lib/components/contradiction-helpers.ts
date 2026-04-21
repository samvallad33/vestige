/**
 * contradiction-helpers — Pure logic for the Contradiction Constellation UI.
 *
 * Extracted from ContradictionArcs.svelte + contradictions/+page.svelte so
 * the math and classification live in one place and can be tested in the
 * vitest `node` environment without jsdom / Svelte harnessing.
 *
 * Contracts
 * ---------
 * - Severity thresholds are STRICTLY exclusive: similarity > 0.7 → strong,
 *   similarity > 0.5 → moderate, else → mild. The boundary values 0.5 and
 *   0.7 therefore fall into the LOWER band on purpose (so a similarity of
 *   exactly 0.7 is 'moderate', not 'strong').
 * - Node type palette has 8 known types; anything else — including
 *   `undefined`, `null`, empty string, or a typo — falls back to violet
 *   (#8b5cf6), matching the `concept` fallback tone used elsewhere.
 * - Pair opacity is a trinary rule: no focus → 1, focused match → 1,
 *   focused non-match → 0.12. `null` and `undefined` both mean "no focus".
 * - Trust is defined on [0,1]; `nodeRadius` clamps out-of-range values so
 *   a negative trust can't produce a sub-zero radius and a >1 trust can't
 *   balloon past the design maximum (14px).
 * - `uniqueMemoryCount` unions memory_a_id + memory_b_id across the whole
 *   pair list; duplicated pairs do not double-count.
 */

/** Shape used by the constellation. Mirrors ContradictionArcs.Contradiction. */
export interface ContradictionLike {
	memory_a_id: string;
	memory_b_id: string;
}

// ---------------------------------------------------------------------------
// Severity — similarity → colour + label.
// ---------------------------------------------------------------------------

export type SeverityLabel = 'strong' | 'moderate' | 'mild';

/** Strong threshold. Similarity STRICTLY above this is red. */
export const SEVERITY_STRONG_THRESHOLD = 0.7;
/** Moderate threshold. Similarity STRICTLY above this (and <= 0.7) is amber. */
export const SEVERITY_MODERATE_THRESHOLD = 0.5;

export const SEVERITY_STRONG_COLOR = '#ef4444';
export const SEVERITY_MODERATE_COLOR = '#f59e0b';
export const SEVERITY_MILD_COLOR = '#fde047';

/**
 * Severity colour by similarity. Boundaries at 0.5 and 0.7 fall into the
 * LOWER band (strictly-greater-than comparison).
 *
 *   sim > 0.7  → '#ef4444' (strong / red)
 *   sim > 0.5  → '#f59e0b' (moderate / amber)
 *   otherwise  → '#fde047' (mild / yellow)
 */
export function severityColor(sim: number): string {
	if (sim > SEVERITY_STRONG_THRESHOLD) return SEVERITY_STRONG_COLOR;
	if (sim > SEVERITY_MODERATE_THRESHOLD) return SEVERITY_MODERATE_COLOR;
	return SEVERITY_MILD_COLOR;
}

/** Severity label by similarity. Same thresholds as severityColor. */
export function severityLabel(sim: number): SeverityLabel {
	if (sim > SEVERITY_STRONG_THRESHOLD) return 'strong';
	if (sim > SEVERITY_MODERATE_THRESHOLD) return 'moderate';
	return 'mild';
}

// ---------------------------------------------------------------------------
// Node type palette.
// ---------------------------------------------------------------------------

/** Fallback colour used when a memory's node_type is missing or unknown. */
export const NODE_COLOR_FALLBACK = '#8b5cf6';

/** Canonical palette for the 8 known node types. */
export const NODE_COLORS: Record<string, string> = {
	fact: '#3b82f6',
	concept: '#8b5cf6',
	event: '#f59e0b',
	person: '#10b981',
	place: '#06b6d4',
	note: '#6b7280',
	pattern: '#ec4899',
	decision: '#ef4444',
};

/** Canonical list of known types (stable order — matches palette object). */
export const KNOWN_NODE_TYPES = Object.freeze([
	'fact',
	'concept',
	'event',
	'person',
	'place',
	'note',
	'pattern',
	'decision',
]) as readonly string[];

/**
 * Map a (possibly undefined) node_type to a colour. Unknown / missing /
 * empty / null strings fall back to violet (#8b5cf6).
 */
export function nodeColor(t?: string | null): string {
	if (!t) return NODE_COLOR_FALLBACK;
	return NODE_COLORS[t] ?? NODE_COLOR_FALLBACK;
}

// ---------------------------------------------------------------------------
// Trust → node radius.
// ---------------------------------------------------------------------------

/** Minimum circle radius at trust=0. */
export const NODE_RADIUS_MIN = 5;
/** Additional radius at trust=1. `r = 5 + trust * 9`, so r ∈ [5, 14]. */
export const NODE_RADIUS_RANGE = 9;

/**
 * Clamp `trust` to [0,1] before mapping to a radius so a bad FSRS value
 * can't produce a sub-zero or oversize node. Non-finite values collapse
 * to 0 (smallest radius — visually suppresses suspicious data).
 */
export function nodeRadius(trust: number): number {
	if (!Number.isFinite(trust)) return NODE_RADIUS_MIN;
	const t = trust < 0 ? 0 : trust > 1 ? 1 : trust;
	return NODE_RADIUS_MIN + t * NODE_RADIUS_RANGE;
}

/** Clamp trust to [0,1]. NaN/Infinity/undefined → 0. */
export function clampTrust(trust: number | null | undefined): number {
	if (trust === null || trust === undefined || !Number.isFinite(trust)) return 0;
	if (trust < 0) return 0;
	if (trust > 1) return 1;
	return trust;
}

// ---------------------------------------------------------------------------
// Focus / pair opacity.
// ---------------------------------------------------------------------------

/** Opacity applied to a non-focused pair when any pair is focused. */
export const UNFOCUSED_OPACITY = 0.12;

/**
 * Opacity for a pair given the current focus state.
 *
 *   focus = null/undefined  → 1  (nothing dimmed)
 *   focus === pairIndex     → 1  (the focused pair is fully lit)
 *   focus !== pairIndex     → 0.12 (dimmed)
 *
 * A focus index that doesn't match any rendered pair simply dims everything.
 * That's the intended "silent no-op" for a stale focusedPairIndex.
 */
export function pairOpacity(pairIndex: number, focusedPairIndex: number | null | undefined): number {
	if (focusedPairIndex === null || focusedPairIndex === undefined) return 1;
	return focusedPairIndex === pairIndex ? 1 : UNFOCUSED_OPACITY;
}

// ---------------------------------------------------------------------------
// Text truncation.
// ---------------------------------------------------------------------------

/**
 * Truncate a string to `max` characters with an ellipsis at the end.
 * Shorter-or-equal strings return unchanged. Empty strings return unchanged.
 * Non-string inputs collapse to '' rather than crashing.
 *
 * The ellipsis counts toward the length budget, so the cut-off content is
 * `max - 1` characters, matching the component's inline truncate() helper.
 */
export function truncate(s: string | null | undefined, max = 60): string {
	if (s === null || s === undefined) return '';
	if (typeof s !== 'string') return '';
	if (max <= 0) return '';
	if (s.length <= max) return s;
	return s.slice(0, max - 1) + '…';
}

// ---------------------------------------------------------------------------
// Stats.
// ---------------------------------------------------------------------------

/**
 * Count unique memory IDs across a list of contradiction pairs. Each pair
 * contributes memory_a_id and memory_b_id. Duplicates (e.g. one memory that
 * appears in multiple conflicts) are counted once.
 */
export function uniqueMemoryCount(pairs: readonly ContradictionLike[]): number {
	if (!pairs || pairs.length === 0) return 0;
	const set = new Set<string>();
	for (const p of pairs) {
		if (p.memory_a_id) set.add(p.memory_a_id);
		if (p.memory_b_id) set.add(p.memory_b_id);
	}
	return set.size;
}

/**
 * Average absolute trust delta across pairs. Returns 0 on empty input so
 * the UI can render `0.00` instead of `NaN`.
 */
export function avgTrustDelta(
	pairs: readonly { trust_a: number; trust_b: number }[],
): number {
	if (!pairs || pairs.length === 0) return 0;
	let sum = 0;
	for (const p of pairs) {
		sum += Math.abs((p.trust_a ?? 0) - (p.trust_b ?? 0));
	}
	return sum / pairs.length;
}

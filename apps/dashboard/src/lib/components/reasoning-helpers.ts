/**
 * reasoning-helpers — Pure logic for the Reasoning Theater UI.
 *
 * Extracted so we can test it without jsdom / Svelte component harnessing.
 * The Vitest setup for this package runs in a Node environment; every helper
 * in this module is a pure function of its inputs, so it can be exercised
 * directly in `__tests__/*.test.ts` alongside the graph helpers.
 */
import { NODE_TYPE_COLORS } from '$types';

// ────────────────────────────────────────────────────────────────
// Shared palette — keep in sync with Tailwind @theme values.
// ────────────────────────────────────────────────────────────────

export const CONFIDENCE_EMERALD = '#10b981';
export const CONFIDENCE_AMBER = '#f59e0b';
export const CONFIDENCE_RED = '#ef4444';

/** Fallback colour when a node-type has no mapping. */
export const DEFAULT_NODE_TYPE_COLOR = '#8B95A5';

// ────────────────────────────────────────────────────────────────
// Roles
// ────────────────────────────────────────────────────────────────

export type EvidenceRole = 'primary' | 'supporting' | 'contradicting' | 'superseded';

export interface RoleMeta {
	label: string;
	/** Tailwind / CSS colour token — see app.css. */
	accent: 'synapse' | 'recall' | 'decay' | 'muted';
	icon: string;
}

export const ROLE_META: Record<EvidenceRole, RoleMeta> = {
	primary: { label: 'Primary', accent: 'synapse', icon: '◈' },
	supporting: { label: 'Supporting', accent: 'recall', icon: '◇' },
	contradicting: { label: 'Contradicting', accent: 'decay', icon: '⚠' },
	superseded: { label: 'Superseded', accent: 'muted', icon: '⊘' },
};

/** Look up role metadata with a defensive fallback. */
export function roleMetaFor(role: EvidenceRole | string): RoleMeta {
	return (ROLE_META as Record<string, RoleMeta>)[role] ?? ROLE_META.supporting;
}

// ────────────────────────────────────────────────────────────────
// Intent classification (deep_reference `intent` field)
// ────────────────────────────────────────────────────────────────

export type IntentKey =
	| 'FactCheck'
	| 'Timeline'
	| 'RootCause'
	| 'Comparison'
	| 'Synthesis';

export interface IntentHint {
	label: string;
	icon: string;
	description: string;
}

export const INTENT_HINTS: Record<IntentKey, IntentHint> = {
	FactCheck: {
		label: 'FactCheck',
		icon: '◆',
		description: 'Direct verification of a single claim.',
	},
	Timeline: {
		label: 'Timeline',
		icon: '↗',
		description: 'Ordered evolution of a fact over time.',
	},
	RootCause: {
		label: 'RootCause',
		icon: '⚡',
		description: 'Why did this happen — causal chain.',
	},
	Comparison: {
		label: 'Comparison',
		icon: '⬡',
		description: 'Contrasting two or more options side-by-side.',
	},
	Synthesis: {
		label: 'Synthesis',
		icon: '❖',
		description: 'Cross-memory composition into a new insight.',
	},
};

/**
 * Map an arbitrary intent string to a hint. Unknown intents degrade to
 * Synthesis, which is the most generic classification.
 */
export function intentHintFor(intent: string | undefined | null): IntentHint {
	if (!intent) return INTENT_HINTS.Synthesis;
	const key = intent as IntentKey;
	return INTENT_HINTS[key] ?? INTENT_HINTS.Synthesis;
}

// ────────────────────────────────────────────────────────────────
// Confidence bands
// ────────────────────────────────────────────────────────────────

/**
 * Confidence colour band.
 *
 *   > 75   → emerald (HIGH)
 *   40-75  → amber (MIXED)
 *   < 40   → red (LOW)
 *
 * Boundaries: 75 is amber (strictly greater than 75 is emerald), 40 is amber
 * (>=40 is amber). Any non-finite input (NaN) is treated as lowest confidence
 * and returns red.
 */
export function confidenceColor(c: number): string {
	if (!Number.isFinite(c)) return CONFIDENCE_RED;
	if (c > 75) return CONFIDENCE_EMERALD;
	if (c >= 40) return CONFIDENCE_AMBER;
	return CONFIDENCE_RED;
}

/** Human-readable label for a confidence score (0-100). */
export function confidenceLabel(c: number): string {
	if (!Number.isFinite(c)) return 'LOW CONFIDENCE';
	if (c > 75) return 'HIGH CONFIDENCE';
	if (c >= 40) return 'MIXED SIGNAL';
	return 'LOW CONFIDENCE';
}

/**
 * Convert a 0-1 trust score to the same confidence band.
 *
 * Thresholds: >0.75 emerald, 0.40-0.75 amber, <0.40 red.
 * Matches `confidenceColor` semantics so the trust bar on an evidence card
 * and the confidence meter on the page agree at the boundaries.
 */
export function trustColor(t: number): string {
	if (!Number.isFinite(t)) return CONFIDENCE_RED;
	return confidenceColor(t * 100);
}

/** Clamp a trust score into the display range [0, 1]. */
export function clampTrust(t: number): number {
	if (!Number.isFinite(t)) return 0;
	if (t < 0) return 0;
	if (t > 1) return 1;
	return t;
}

/** Trust as a 0-100 percentage suitable for width / label rendering. */
export function trustPercent(t: number): number {
	return clampTrust(t) * 100;
}

// ────────────────────────────────────────────────────────────────
// Node-type colouring
// ────────────────────────────────────────────────────────────────

/** Resolve a node-type colour with a soft-steel fallback. */
export function nodeTypeColor(nodeType?: string | null): string {
	if (!nodeType) return DEFAULT_NODE_TYPE_COLOR;
	return NODE_TYPE_COLORS[nodeType] ?? DEFAULT_NODE_TYPE_COLOR;
}

// ────────────────────────────────────────────────────────────────
// Date formatting
// ────────────────────────────────────────────────────────────────

/**
 * Format an ISO date string for EvidenceCard display.
 *
 * Handles three failure modes that `new Date(str)` alone does not:
 *  1. Empty / null / undefined  → returns '—'
 *  2. Unparseable string (NaN)  → returns the original string unchanged
 *  3. Non-ISO but parseable      → best-effort locale format
 *
 * The previous try/catch-only approach silently rendered the literal text
 * "Invalid Date" because `Date` never throws on bad input — it produces a
 * valid object whose getTime() is NaN.
 */
export function formatDate(
	iso: string | null | undefined,
	locale?: string,
): string {
	if (iso == null) return '—';
	if (typeof iso !== 'string' || iso.trim() === '') return '—';
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return iso;
	try {
		return d.toLocaleDateString(locale, {
			month: 'short',
			day: 'numeric',
			year: 'numeric',
		});
	} catch {
		return iso;
	}
}

/** Compact month/day formatter for the evolution timeline. */
export function formatShortDate(
	iso: string | null | undefined,
	locale?: string,
): string {
	if (iso == null) return '—';
	if (typeof iso !== 'string' || iso.trim() === '') return '—';
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return iso;
	try {
		return d.toLocaleDateString(locale, { month: 'short', day: 'numeric' });
	} catch {
		return iso;
	}
}

// ────────────────────────────────────────────────────────────────
// Short-id for #abcdef01 style display
// ────────────────────────────────────────────────────────────────

/**
 * Return the first 8 characters of an id, or the full string if shorter.
 * Never throws on null/undefined — returns '' so the caller can render '#'.
 */
export function shortenId(id: string | null | undefined, length = 8): string {
	if (!id) return '';
	return id.length > length ? id.slice(0, length) : id;
}

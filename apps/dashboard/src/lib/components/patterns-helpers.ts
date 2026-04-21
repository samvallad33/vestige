/**
 * patterns-helpers — Pure logic for the Cross-Project Intelligence UI
 * (patterns/+page.svelte + PatternTransferHeatmap.svelte).
 *
 * Extracted so the behaviour can be unit-tested in the vitest `node`
 * environment without jsdom or Svelte component harnessing. Every helper
 * in this module is a pure function of its inputs.
 *
 * Contracts
 * ---------
 * - `cellIntensity`: returns opacity in [0,1] from count / max. count=0 → 0,
 *   count>=max → 1. `max<=0` collapses to 0 (avoids div-by-zero — the
 *   component uses `max || 1` for the same reason).
 * - `filterByCategory`: 'All' passes every pattern through. An unknown
 *   category string (not one of the 6 + 'All') returns an empty array —
 *   there is no hidden alias fallback.
 * - `buildTransferMatrix`: directional. `matrix[origin][dest]` counts how
 *   many patterns originated in `origin` and were transferred to `dest`.
 *   `origin === dest` captures self-transfer (a project reusing its own
 *   pattern — rare but real per the component's doc comment).
 */

export const PATTERN_CATEGORIES = [
	'ErrorHandling',
	'AsyncConcurrency',
	'Testing',
	'Architecture',
	'Performance',
	'Security',
] as const;

export type PatternCategory = (typeof PATTERN_CATEGORIES)[number];
export type CategoryFilter = 'All' | PatternCategory;

export interface TransferPatternLike {
	name: string;
	category: PatternCategory;
	origin_project: string;
	transferred_to: string[];
	transfer_count: number;
}

/**
 * Normalise a raw transfer count to a 0..1 opacity/intensity value against a
 * known max. Used by the heatmap cell colour ramp.
 *
 *   count <= 0          → 0      (dead cell)
 *   count >= max > 0    → 1      (hottest cell)
 *   otherwise           → count / max
 *
 * Non-finite / negative inputs collapse to 0. When `max <= 0` the result is
 * always 0 — the component's own guard (`maxCount || 1`) means this branch
 * is unreachable in practice, but defensive anyway.
 */
export function cellIntensity(count: number, max: number): number {
	if (!Number.isFinite(count) || count <= 0) return 0;
	if (!Number.isFinite(max) || max <= 0) return 0;
	if (count >= max) return 1;
	return count / max;
}

/**
 * Filter a pattern list by the active category tab.
 *   'All'                → full pass-through (same reference-equal array is
 *                          NOT guaranteed; callers must not rely on identity)
 *   one of the 6 enums   → strict equality on `category`
 *   unknown string       → empty array (no silent alias; caller bug)
 */
export function filterByCategory<P extends TransferPatternLike>(
	patterns: readonly P[],
	category: CategoryFilter | string,
): P[] {
	if (category === 'All') return patterns.slice();
	if (!(PATTERN_CATEGORIES as readonly string[]).includes(category)) {
		return [];
	}
	return patterns.filter((p) => p.category === category);
}

/** Cell in the directional N×N transfer matrix. */
export interface TransferCell {
	count: number;
	topNames: string[];
}

/** Dense row-major directional matrix: matrix[origin][destination]. */
export type TransferMatrix = Record<string, Record<string, TransferCell>>;

/**
 * Build the directional transfer matrix from patterns + the known projects
 * axis. Mirrors `PatternTransferHeatmap.svelte`'s `$derived` logic.
 *
 *   - Every (from, to) pair in `projects × projects` gets a zero cell.
 *   - Each pattern P contributes `+1` to `matrix[P.origin][dest]` for every
 *     `dest` in `P.transferred_to` that also appears in `projects`.
 *   - Patterns whose origin isn't in `projects` are silently skipped — that
 *     matches the component's `if (!m[from]) continue` guard.
 *   - `topNames` holds up to 3 pattern names per cell in insertion order.
 */
export function buildTransferMatrix(
	projects: readonly string[],
	patterns: readonly TransferPatternLike[],
	topNameCap = 3,
): TransferMatrix {
	const m: TransferMatrix = {};
	for (const from of projects) {
		m[from] = {};
		for (const to of projects) {
			m[from][to] = { count: 0, topNames: [] };
		}
	}
	for (const p of patterns) {
		const from = p.origin_project;
		if (!m[from]) continue;
		for (const to of p.transferred_to) {
			if (!m[from][to]) continue;
			m[from][to].count += 1;
			m[from][to].topNames.push(p.name);
		}
	}
	const cap = Math.max(0, topNameCap);
	for (const from of projects) {
		for (const to of projects) {
			m[from][to].topNames = m[from][to].topNames.slice(0, cap);
		}
	}
	return m;
}

/**
 * Maximum single-cell transfer count across the matrix. Floors at 0 for an
 * empty matrix, which callers should treat as "scale by 1" to avoid a div-
 * by-zero in the colour ramp.
 */
export function matrixMaxCount(
	projects: readonly string[],
	matrix: TransferMatrix,
): number {
	let max = 0;
	for (const from of projects) {
		const row = matrix[from];
		if (!row) continue;
		for (const to of projects) {
			const cell = row[to];
			if (cell && cell.count > max) max = cell.count;
		}
	}
	return max;
}

/**
 * Flatten a matrix into sorted-desc rows for the mobile fallback. Only
 * non-zero pairs are emitted, matching the component.
 */
export function flattenNonZero(
	projects: readonly string[],
	matrix: TransferMatrix,
): Array<{ from: string; to: string; count: number; topNames: string[] }> {
	const rows: Array<{ from: string; to: string; count: number; topNames: string[] }> = [];
	for (const from of projects) {
		for (const to of projects) {
			const cell = matrix[from]?.[to];
			if (cell && cell.count > 0) {
				rows.push({ from, to, count: cell.count, topNames: cell.topNames });
			}
		}
	}
	return rows.sort((a, b) => b.count - a.count);
}

/**
 * Truncate long project names for axis labels. Match the component's
 * `shortProject` behaviour: keep ≤12 chars, otherwise 11-char prefix + ellipsis.
 */
export function shortProjectName(name: string): string {
	if (!name) return '';
	return name.length > 12 ? name.slice(0, 11) + '…' : name;
}

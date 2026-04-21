/**
 * Pure helpers for the Memory Hygiene / Duplicate Detection UI.
 *
 * Extracted from DuplicateCluster.svelte + duplicates/+page.svelte so the
 * logic can be unit tested in the vitest `node` environment without jsdom.
 *
 * Contracts
 * ---------
 * - `similarityBand`: fixed thresholds at 0.92 (near-identical) and 0.80
 *   (strong). Boundary values MATCH the higher band (>= semantics).
 * - `pickWinner`: highest retention wins. Ties broken by earliest index
 *   (stable). Returns `null` on empty input — callers must guard.
 * - `suggestedActionFor`: >= 0.92 → 'merge', < 0.85 → 'review'. The 0.85..0.92
 *   corridor follows the upstream `suggestedAction` field from the MCP tool,
 *   so we only override the obvious cases. Default for the corridor is
 *   whatever the caller already had — this function returns null to signal
 *   "caller decides."
 * - `filterByThreshold`: strict `>=` against the provided similarity.
 * - `clusterKey`: stable identity across re-fetches — sorted member ids
 *   joined. Survives threshold changes that keep the same cluster members.
 */

export type SimilarityBand = 'near-identical' | 'strong' | 'weak';
export type SuggestedAction = 'merge' | 'review';

export interface ClusterMemoryLike {
	id: string;
	retention: number;
	tags?: string[];
	createdAt?: string;
}

export interface ClusterLike<M extends ClusterMemoryLike = ClusterMemoryLike> {
	similarity: number;
	memories: M[];
}

/** Color bands. Boundary at 0.92 → red. Boundary at 0.80 → amber. */
export function similarityBand(similarity: number): SimilarityBand {
	if (similarity >= 0.92) return 'near-identical';
	if (similarity >= 0.8) return 'strong';
	return 'weak';
}

export function similarityBandColor(similarity: number): string {
	const band = similarityBand(similarity);
	if (band === 'near-identical') return 'var(--color-decay)';
	if (band === 'strong') return 'var(--color-warning)';
	return '#fde047'; // yellow-300 — distinct from amber warning
}

export function similarityBandLabel(similarity: number): string {
	const band = similarityBand(similarity);
	if (band === 'near-identical') return 'Near-identical';
	if (band === 'strong') return 'Strong match';
	return 'Weak match';
}

/** Retention color dot. Matches the traffic-light scheme. */
export function retentionColor(retention: number): string {
	if (retention > 0.7) return '#10b981';
	if (retention > 0.4) return '#f59e0b';
	return '#ef4444';
}

/**
 * Pick the highest-retention memory. Stable tie-break: earliest wins.
 * Returns `null` if the cluster is empty. Treats non-finite retention as
 * -Infinity so a `retention=NaN` row never claims the throne.
 */
export function pickWinner<M extends ClusterMemoryLike>(memories: M[]): M | null {
	if (!memories || memories.length === 0) return null;
	let best = memories[0];
	let bestScore = Number.isFinite(best.retention) ? best.retention : -Infinity;
	for (let i = 1; i < memories.length; i++) {
		const m = memories[i];
		const s = Number.isFinite(m.retention) ? m.retention : -Infinity;
		if (s > bestScore) {
			best = m;
			bestScore = s;
		}
	}
	return best;
}

/**
 * Suggested action inference. Returns null in the ambiguous 0.85..0.92 band
 * so callers can honor an upstream suggestion from the backend.
 */
export function suggestedActionFor(similarity: number): SuggestedAction | null {
	if (similarity >= 0.92) return 'merge';
	if (similarity < 0.85) return 'review';
	return null;
}

/**
 * Filter clusters by the >= threshold contract. Separate pure function so the
 * mock fetch and any future real fetch both get the same semantics.
 */
export function filterByThreshold<C extends ClusterLike>(clusters: C[], threshold: number): C[] {
	return clusters.filter((c) => c.similarity >= threshold);
}

/**
 * Stable identity across re-fetches. Uses sorted member ids, so a cluster
 * that loses/gains a member gets a new key (intentional — the cluster has
 * changed). If you dismissed cluster [A,B,C] at 0.80 and refetch at 0.70
 * and it now contains [A,B,C,D], it reappears — correct behaviour: a new
 * member deserves fresh attention.
 */
export function clusterKey<M extends ClusterMemoryLike>(memories: M[]): string {
	return memories
		.map((m) => m.id)
		.slice()
		.sort()
		.join('|');
}

/**
 * Safe content preview — trims, collapses whitespace, truncates at 80 chars
 * with an ellipsis. Null-safe.
 */
export function previewContent(content: string | null | undefined, max: number = 80): string {
	if (!content) return '';
	const trimmed = content.trim().replace(/\s+/g, ' ');
	return trimmed.length <= max ? trimmed : trimmed.slice(0, max) + '…';
}

/**
 * Render an ISO date string safely — returns an empty string for missing,
 * non-string, or invalid input so the DOM shows nothing rather than
 * "Invalid Date".
 */
export function formatDate(iso: string | null | undefined): string {
	if (!iso || typeof iso !== 'string') return '';
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return '';
	return d.toLocaleDateString(undefined, {
		year: 'numeric',
		month: 'short',
		day: 'numeric',
	});
}

/** Safe tag slice — tolerates undefined or non-array inputs. */
export function safeTags(tags: string[] | null | undefined, limit: number = 4): string[] {
	if (!Array.isArray(tags)) return [];
	return tags.slice(0, limit);
}

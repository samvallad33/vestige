/**
 * FSRS-6 retrievability — an EXACT TypeScript port of the Rust closed form
 * (crates/vestige-core/src/fsrs/algorithm.rs:101 `retrievability_with_decay`).
 *
 * This is the moat, not decoration: the living decay field dims each memory on
 * its REAL per-memory forgetting curve. Every value here is verified against
 * source — nothing is approximated, so the field is honest by construction
 * (feed it noise instead of the real curve and the drift is legibly wrong).
 *
 *   factor = 0.9^(-1/w20) - 1
 *   R = (1 + factor * elapsed_days / stability)^(-w20)
 *
 * with the personalizable w20 defaulting to the engine's DEFAULT_DECAY.
 * stability is in DAYS; elapsed_days is days since last review.
 */

/**
 * FSRS-6 default forgetting-curve decay (w20). Mirrors the Rust
 * `DEFAULT_DECAY = 0.1542` constant. NOTE: not 0.01036 — that value appears
 * nowhere in the engine; verified against algorithm.rs on 2026-07-08.
 */
export const DEFAULT_DECAY = 0.1542;

/** FSRS-6 forgetting factor: `0.9^(-1/w20) - 1`. */
export function forgettingFactor(w20: number = DEFAULT_DECAY): number {
	return Math.pow(0.9, -1 / w20) - 1;
}

/**
 * Probability of recall (0..1) for a memory of the given `stability` (days)
 * after `elapsedDays` since its last review. Exact port of the Rust guard
 * clauses: stability<=0 → 0, elapsed<=0 → 1.
 */
export function retrievability(
	stability: number,
	elapsedDays: number,
	w20: number = DEFAULT_DECAY
): number {
	if (!(stability > 0)) return 0;
	if (!(elapsedDays > 0)) return 1;
	const factor = forgettingFactor(w20);
	const r = Math.pow(1 + (factor * elapsedDays) / stability, -w20);
	return r < 0 ? 0 : r > 1 ? 1 : r;
}

const MS_PER_DAY = 86_400_000;

/**
 * Days elapsed between an ISO timestamp (`lastAccessed`) and `nowMs`, plus an
 * optional forward projection. The projection is the Phase-1 honesty control:
 * with live w20 and multi-day stabilities, real drift over a viewing session
 * is imperceptibly slow, so the scrubber recomputes elapsed at `t + N days` on
 * the SAME true curve — fully honest, just legible.
 *
 * Returns 0 for a missing/unparseable timestamp so a node with no last-access
 * simply reads as freshly reviewed (R=1) rather than crashing the field.
 */
export function elapsedDays(
	lastAccessedIso: string | undefined,
	nowMs: number,
	projectionDays = 0
): number {
	if (!lastAccessedIso) return projectionDays > 0 ? projectionDays : 0;
	const t = Date.parse(lastAccessedIso);
	if (!Number.isFinite(t)) return projectionDays > 0 ? projectionDays : 0;
	const days = (nowMs - t) / MS_PER_DAY;
	return Math.max(0, days) + Math.max(0, projectionDays);
}

/**
 * Live retrievability for a memory from its real FSRS state, at `nowMs` with an
 * optional forward projection. This is the single value the decay field writes
 * into each node's `vel_retention.w` every frame.
 */
export function liveRetrievability(
	stability: number | undefined,
	lastAccessedIso: string | undefined,
	nowMs: number,
	projectionDays = 0,
	w20: number = DEFAULT_DECAY
): number {
	if (stability === undefined || !Number.isFinite(stability)) return 1;
	return retrievability(stability, elapsedDays(lastAccessedIso, nowMs, projectionDays), w20);
}

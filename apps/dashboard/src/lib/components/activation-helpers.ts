/**
 * activation-helpers — Pure logic for the Spreading Activation Live View.
 *
 * Extracted from ActivationNetwork.svelte + activation/+page.svelte so the
 * decay / geometry / event-filtering rules can be exercised in the Vitest
 * `node` environment without jsdom. Every helper in this module is a pure
 * function of its inputs; no DOM, no timers, no Svelte runes.
 *
 * The constants in this module are the single source of truth — the Svelte
 * component re-exports / re-uses them rather than hard-coding its own.
 *
 * References
 * ----------
 * - Collins & Loftus 1975 — spreading activation with exponential decay
 * - Anderson 1983 (ACT-R) — activation threshold for availability
 */
import { NODE_TYPE_COLORS } from '$types';
import type { VestigeEvent } from '$types';

/** Per-tick multiplicative decay factor (Collins & Loftus 1975). */
export const DECAY = 0.93;

/** Activation below this floor is invisible / garbage-collected. */
export const MIN_VISIBLE = 0.05;

/** Fallback node colour when NODE_TYPE_COLORS has no entry for the type. */
export const FALLBACK_COLOR = '#8B95A5';

/** Source node colour (synapse-glow). Distinct from any node-type colour. */
export const SOURCE_COLOR = '#818cf8';

/** Radial spacing between concentric rings (px). */
export const RING_GAP = 140;

/** Max neighbours that fit on ring 1 before spilling to ring 2. */
export const RING_1_CAPACITY = 8;

/** Edge draw stagger — frames of delay per rank inside a ring. */
export const STAGGER_PER_RANK = 4;

/** Extra stagger added to ring-2 edges so they light up after ring 1. */
export const STAGGER_RING_2_BONUS = 12;

// ---------------------------------------------------------------------------
// Decay math
// ---------------------------------------------------------------------------

/**
 * Apply a single tick of exponential decay. Clamps negative input to 0 so a
 * corrupt state never produces a creeping-positive value on the next tick.
 */
export function applyDecay(activation: number): number {
	if (!Number.isFinite(activation) || activation <= 0) return 0;
	return activation * DECAY;
}

/**
 * Compound decay over N ticks. N < 0 is treated as 0 (no change).
 * Equivalent to calling `applyDecay` N times.
 */
export function compoundDecay(activation: number, ticks: number): number {
	if (!Number.isFinite(activation) || activation <= 0) return 0;
	if (!Number.isFinite(ticks) || ticks <= 0) return activation;
	return activation * DECAY ** ticks;
}

/** True if the node's activation is at or above the visibility floor. */
export function isVisible(activation: number): boolean {
	if (!Number.isFinite(activation)) return false;
	return activation >= MIN_VISIBLE;
}

/**
 * How many ticks until `initial` decays below `MIN_VISIBLE`. Useful in tests
 * and for sizing animation budgets. Initial <= threshold returns 0.
 */
export function ticksUntilInvisible(initial: number): number {
	if (!Number.isFinite(initial) || initial <= MIN_VISIBLE) return 0;
	// initial * DECAY^n < MIN_VISIBLE  →  n > log(MIN_VISIBLE/initial) / log(DECAY)
	const n = Math.log(MIN_VISIBLE / initial) / Math.log(DECAY);
	return Math.ceil(n);
}

// ---------------------------------------------------------------------------
// Ring placement — concentric circles around a source
// ---------------------------------------------------------------------------

export interface Point {
	x: number;
	y: number;
}

/**
 * Classify a neighbour's 0-indexed rank into a ring number.
 * Ranks 0..RING_1_CAPACITY-1 → ring 1; rest → ring 2.
 */
export function computeRing(rank: number): 1 | 2 {
	if (!Number.isFinite(rank) || rank < RING_1_CAPACITY) return 1;
	return 2;
}

/**
 * Evenly distribute `count` positions on a circle of radius `ring * RING_GAP`
 * centred at (cx, cy). `angleOffset` rotates the whole ring so overlapping
 * bursts don't perfectly collide. Zero count returns `[]`.
 */
export function ringPositions(
	cx: number,
	cy: number,
	count: number,
	ring: number,
	angleOffset = 0,
): Point[] {
	if (!Number.isFinite(count) || count <= 0) return [];
	const radius = RING_GAP * ring;
	const positions: Point[] = [];
	for (let i = 0; i < count; i++) {
		const angle = angleOffset + (i / count) * Math.PI * 2;
		positions.push({
			x: cx + Math.cos(angle) * radius,
			y: cy + Math.sin(angle) * radius,
		});
	}
	return positions;
}

/**
 * Given the full neighbour list, produce a flat array of Points — ring 1
 * first, ring 2 after. The resulting length === neighbours.length.
 */
export function layoutNeighbours(
	cx: number,
	cy: number,
	neighbourCount: number,
	angleOffset = 0,
): Point[] {
	const ring1 = Math.min(neighbourCount, RING_1_CAPACITY);
	const ring2 = Math.max(0, neighbourCount - RING_1_CAPACITY);
	return [
		...ringPositions(cx, cy, ring1, 1, angleOffset),
		...ringPositions(cx, cy, ring2, 2, angleOffset),
	];
}

// ---------------------------------------------------------------------------
// Initial activation by rank
// ---------------------------------------------------------------------------

/**
 * Seed activation for a neighbour at 0-indexed `rank` given `total`.
 * Higher-ranked (earlier) neighbours get stronger initial activation.
 * Ring-2 neighbours get a 0.75× ring-factor penalty on top of the rank factor.
 * Returns a value in (0, 1].
 */
export function initialActivation(rank: number, total: number): number {
	if (!Number.isFinite(total) || total <= 0) return 0;
	if (!Number.isFinite(rank) || rank < 0) return 0;
	const rankFactor = 1 - (rank / total) * 0.35;
	const ringFactor = computeRing(rank) === 1 ? 1 : 0.75;
	return Math.min(1, rankFactor * ringFactor);
}

// ---------------------------------------------------------------------------
// Edge stagger
// ---------------------------------------------------------------------------

/**
 * Delay (in animation frames) before the edge at rank `i` starts drawing.
 * Ring 1 edges light up first, then ring 2 after a bonus delay.
 */
export function edgeStagger(rank: number): number {
	if (!Number.isFinite(rank) || rank < 0) return 0;
	const r = Math.floor(rank);
	const base = r * STAGGER_PER_RANK;
	return computeRing(r) === 1 ? base : base + STAGGER_RING_2_BONUS;
}

// ---------------------------------------------------------------------------
// Color mapping
// ---------------------------------------------------------------------------

/**
 * Colour for a node on the activation canvas.
 *   - source nodes always use SOURCE_COLOR (synapse-glow)
 *   - known node types use NODE_TYPE_COLORS
 *   - unknown node types fall back to FALLBACK_COLOR (soft steel)
 */
export function activationColor(
	nodeType: string | null | undefined,
	isSource: boolean,
): string {
	if (isSource) return SOURCE_COLOR;
	if (!nodeType) return FALLBACK_COLOR;
	return NODE_TYPE_COLORS[nodeType] ?? FALLBACK_COLOR;
}

// ---------------------------------------------------------------------------
// Event-feed filtering — "only fire on NEW ActivationSpread events"
// ---------------------------------------------------------------------------

export interface SpreadPayload {
	source_id: string;
	target_ids: string[];
}

/**
 * Extract ActivationSpread payloads from a websocket event feed. The feed
 * is prepended (newest at index 0, oldest at the end). Stop as soon as we
 * hit the reference of `lastSeen` — events at or past that point were
 * already processed by a prior tick.
 *
 * Returned payloads are in OLDEST-FIRST order so downstream callers can
 * fire them in the same narrative order they occurred.
 *
 * Payloads missing required fields are silently skipped.
 */
export function filterNewSpreadEvents(
	feed: readonly VestigeEvent[],
	lastSeen: VestigeEvent | null,
): SpreadPayload[] {
	if (!feed || feed.length === 0) return [];
	const fresh: SpreadPayload[] = [];
	for (const ev of feed) {
		if (ev === lastSeen) break;
		if (ev.type !== 'ActivationSpread') continue;
		const data = ev.data as { source_id?: unknown; target_ids?: unknown };
		if (typeof data.source_id !== 'string') continue;
		if (!Array.isArray(data.target_ids)) continue;
		const targets = data.target_ids.filter(
			(t): t is string => typeof t === 'string',
		);
		if (targets.length === 0) continue;
		fresh.push({ source_id: data.source_id, target_ids: targets });
	}
	// Reverse so oldest-first.
	return fresh.reverse();
}

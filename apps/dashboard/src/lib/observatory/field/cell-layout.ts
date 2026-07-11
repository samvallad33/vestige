/**
 * cell-layout — the ~20-line-per-organ contract. Turns a plain array of scored
 * data points into LivingCell[] laid out as a golden-angle galaxy so any organ
 * gets a full-bleed moving field for free. Each organ maps its REAL data →
 * FieldDatum (score/hue/scar), then calls layoutGalaxy(). No data invented: the
 * datum carries the real memory/pair/event id + its real metric.
 */

import type { LivingCell } from './living-field-pass';
import { retentionColor, rgb01, RETENTION, IMMUNE, CAUSAL } from '$lib/observatory/cognitive-palette';

export interface FieldDatum {
	id: string;
	/** 0..1 primary score — drives radius (sqrt) + orbit ring (bright = inner). */
	score: number;
	/** rgb 0..1 meaning color. Default: retention ramp of `score`. */
	hue?: [number, number, number];
	/** 0..1 glow. Default: score. */
	energy?: number;
	/** secondary metric 0..1 (retention-ish) for the membrane tint. */
	metric2?: number;
	selected?: boolean;
	scar?: boolean;
	kind?: string;
	payload?: unknown;
}

const GOLDEN_ANGLE = 2.399963229728653;

/**
 * Lay N data points on a golden-angle spiral filling the disc. High-score points
 * pulled toward the bright inner core; low-score drift to the cold rim. The whole
 * thing orbits (the LivingFieldPass spins it), so this is just the resting home.
 */
export function layoutGalaxy(
	data: FieldDatum[],
	opts: { maxRadius?: number; minCellR?: number; maxCellR?: number; jitter?: boolean } = {}
): LivingCell[] {
	const n = data.length;
	if (n === 0) return [];
	const maxR = opts.maxRadius ?? 0.92;
	const minCellR = opts.minCellR ?? 0.012;
	const maxCellR = opts.maxCellR ?? 0.05;
	// Sort a COPY by score desc so bright memories occupy the inner rings; the
	// original array order (and its real ids) is preserved in each datum.
	const order = data.map((d, i) => ({ d, i })).sort((a, b) => (b.d.score || 0) - (a.d.score || 0));
	return order.map(({ d }, rank) => {
		const t = n > 1 ? rank / (n - 1) : 0;
		// radial: bright (rank 0) near center, faint toward rim. sqrt keeps density
		// even across the disc instead of clumping at the middle.
		const rr = maxR * Math.sqrt(0.06 + 0.94 * t);
		const ang = rank * GOLDEN_ANGLE;
		const x = Math.cos(ang) * rr;
		const y = Math.sin(ang) * rr;
		const score = clamp01(d.score);
		const radius = minCellR + (maxCellR - minCellR) * Math.sqrt(score);
		const hue = d.hue ?? retentionColor(score);
		return {
			x,
			y,
			radius,
			hue: [hue[0], hue[1], hue[2]] as [number, number, number],
			energy: clamp01(d.energy ?? 0.35 + 0.65 * score),
			phase: rank / n,
			pickId: d.id,
			kind: d.kind,
			payload: d.payload ?? d,
			selected: d.selected,
			scar: d.scar,
			metric2: clamp01(d.metric2 ?? score),
			spin: 1
		} satisfies LivingCell;
	});
}

/**
 * Ring layout — points arranged on concentric rings by a bucket key (e.g. a
 * project, a status, a due-day). Good for organs where grouping is the story
 * (schedule buckets, patterns per project). Each ring gets its own spin phase.
 */
export function layoutRings(
	data: FieldDatum[],
	ringOf: (d: FieldDatum, i: number) => number,
	opts: { ringCount?: number; maxRadius?: number; minCellR?: number; maxCellR?: number } = {}
): LivingCell[] {
	const n = data.length;
	if (n === 0) return [];
	const maxR = opts.maxRadius ?? 0.9;
	const minCellR = opts.minCellR ?? 0.014;
	const maxCellR = opts.maxCellR ?? 0.05;
	const rings = Math.max(1, opts.ringCount ?? new Set(data.map(ringOf)).size);
	// group indices per ring to spread points evenly around each ring
	const perRing = new Map<number, number>();
	return data.map((d, i) => {
		const ring = ((ringOf(d, i) % rings) + rings) % rings;
		const seen = perRing.get(ring) ?? 0;
		perRing.set(ring, seen + 1);
		const ringR = maxR * (0.18 + 0.82 * (ring / Math.max(1, rings - 1)));
		const ang = seen * GOLDEN_ANGLE + ring * 0.7;
		const score = clamp01(d.score);
		const hue = d.hue ?? retentionColor(score);
		return {
			x: Math.cos(ang) * ringR,
			y: Math.sin(ang) * ringR,
			radius: minCellR + (maxCellR - minCellR) * Math.sqrt(score),
			hue: [hue[0], hue[1], hue[2]] as [number, number, number],
			energy: clamp01(d.energy ?? 0.35 + 0.65 * score),
			phase: ring / rings,
			pickId: d.id,
			kind: d.kind,
			payload: d.payload ?? d,
			selected: d.selected,
			scar: d.scar,
			metric2: clamp01(d.metric2 ?? score),
			spin: 1
		} satisfies LivingCell;
	});
}

/** Common meaning colors so organ mappers stay short + on-palette. */
export const FIELD_HUE = {
	oxygen: rgb01(RETENTION.luciferin),
	healthy: rgb01(RETENTION.healthy),
	recall: rgb01(RETENTION.recall),
	bridge: rgb01(RETENTION.bridge),
	debt: rgb01(RETENTION.debt),
	scarlet: rgb01(IMMUNE.veto),
	caution: rgb01(IMMUNE.caution),
	forward: rgb01(CAUSAL.forward),
	retrograde: rgb01(CAUSAL.retrograde)
} as const;

function clamp01(v: number): number {
	return Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));
}

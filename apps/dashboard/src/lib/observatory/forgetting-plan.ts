/**
 * Cognitive Observatory — forgetting-horizon demo plan (FSRS as a living system).
 *
 * Pure CPU: given the stable-indexed ObservatoryGraph, DETERMINISTICALLY pick
 * the drifting set (the lowest-retention ~25% of memories), the 3 rescued
 * memories (mid-retention, well-connected — the ones a recall would plausibly
 * save), the packed per-node role words, the recall PathSteps and the spine
 * beats. No Math.random(), no Date.now(), no layout needed — selection is a
 * pure function of the graph.
 *
 * The 720-frame beat map (fixed 60Hz, seamless loop):
 *   0-90    field at rest
 *   90-420  THE DRIFT — drifting memories dim and fall toward the horizon
 *           (demo.z rises to the 0.55 plateau, staggered by retention rank)
 *   300-480 THE RESCUES — 3 fading memories are recalled one by one (recall
 *           ribbons land at 318/378/438; demo.z snaps back, demo.x ignites)
 *   480-660 the unrescued sink to near-black (demo.z → exactly 1.0 by 640;
 *           the fragment floor keeps them at ~6% — never gone, always
 *           retrievable)
 *   660-720 master release — every lane decays to EXACTLY zero by frame 712
 *
 * `forgettingEnvelopes` and `horizonDrift` are the authoritative CPU mirrors
 * of shaders/forgetting.wgsl.ts and the demo-3 branch of render-nodes.wgsl.ts;
 * the seam-zero unit test machine-checks the loop guarantee against them.
 */

import { PATH_KIND, type ObservatoryGraph } from './types';
import type { PathStepMeta } from './path-builder';
import { truncateLabel } from './rescue-plan';

// ---------------------------------------------------------------------------
// Beat-map constants (shared with shaders/forgetting.wgsl.ts — keep in lockstep)
// ---------------------------------------------------------------------------

export const DRIFT_ONSET_BASE = 90;
export const DRIFT_ONSET_SPREAD = 42;
export const DRIFT_ENGULF = 210;
export const PHASE1_LEVEL = 0.55;
export const PHASE2_LEVEL = 0.45;
export const PHASE2_BASE = 480;
export const PHASE2_STAGGER = 24;
export const PHASE2_END = 640;
/** Rescue k ribbon lands at RESCUE_BASE + k * RESCUE_INTERVAL → 318/378/438. */
export const RESCUE_BASE = 318;
export const RESCUE_INTERVAL = 60;
export const MASTER_R0 = 660;
export const MASTER_R1 = 712;
export const LOOP_FRAMES = 720;
export const FORGETTING_K = 3;
/** Fading spine beat k lands at FADING_BASE + k * FADING_INTERVAL → 132/192/252. */
export const FADING_BASE = 132;
export const FADING_INTERVAL = 60;
export const SINK_BEAT_FRAME = 540;
export const RETRIEVABLE_BEAT_FRAME = 660;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ForgettingPlan {
	/** false ⇒ field renders at rest, story suppressed (no fake drift). */
	viable: boolean;
	/** Drifting node indices, retention asc (rank order). */
	driftingIndices: number[];
	/** Rescued node indices, slot order k = 0..K-1. Always ⊆ driftingIndices. */
	rescuedIndices: number[];
	/**
	 * 1 u32/node: bits 0-7 rank (0..255 across the drifting set),
	 * bit 8 isDrifting, bit 9 isRescued, bits 10-11 rescue slot k.
	 * Non-drifting nodes (incl. the center) are exactly 0.
	 */
	horizonData: Uint32Array;
	/** UINTS_PER_PATHSTEP u32 per step; min Uint32Array(4). */
	pathData: Uint32Array<ArrayBuffer>;
	/** MUST be 1:1 with pathData steps (setPathSteps contract). */
	pathMetas: PathStepMeta[];
	/** Curated spine beats (route state only, NEVER sent to GPU). */
	spineBeats: PathStepMeta[];
}

// ---------------------------------------------------------------------------
// Selection (exported for tests; all ties → ascending node index)
// ---------------------------------------------------------------------------

/**
 * The drifting set: exclude the center, sort (retention asc, index asc), take
 * driftCount = min(n−1, max(min(3, n−1), round(0.25·(n−1)))). Suppressed
 * memories are eligible — suppression is not forgetting.
 */
export function pickDrifting(graph: ObservatoryGraph): number[] {
	const cand: number[] = [];
	for (let i = 0; i < graph.nodes.length; i++) {
		if (i === graph.centerIndex) continue;
		cand.push(i);
	}
	cand.sort((a, b) => graph.nodes[a].retention - graph.nodes[b].retention || a - b);
	const eligible = cand.length;
	if (eligible === 0) return [];
	const driftCount = Math.min(
		eligible,
		Math.max(Math.min(FORGETTING_K, eligible), Math.round(0.25 * eligible))
	);
	return cand.slice(0, driftCount);
}

/**
 * The rescued: within the drifting set, the K = min(3, driftCount) most
 * plausible recall targets — score = 2·retention + min(degree, 8)/8
 * (mid-retention, well-connected), sort (score desc, index asc).
 */
export function pickRescued(graph: ObservatoryGraph, drifting: number[]): number[] {
	const degree = new Uint32Array(graph.nodes.length);
	for (const e of graph.edges) {
		degree[e.sourceIndex]++;
		degree[e.targetIndex]++;
	}
	const score = (i: number): number =>
		2 * graph.nodes[i].retention + Math.min(degree[i], 8) / 8;
	const sorted = drifting.slice().sort((a, b) => score(b) - score(a) || a - b);
	return sorted.slice(0, Math.min(FORGETTING_K, drifting.length));
}

export function rescueFrame(k: number): number {
	return RESCUE_BASE + RESCUE_INTERVAL * k;
}

// ---------------------------------------------------------------------------
// Envelope math — the authoritative CPU mirror of shaders/forgetting.wgsl.ts
// ---------------------------------------------------------------------------

function smooth(a: number, b: number, f: number): number {
	const t = Math.min(1, Math.max(0, (f - a) / (b - a)));
	return t * t * (3 - 2 * t);
}

function env(f: number, a0: number, a1: number, r0: number, r1: number): number {
	return smooth(a0, a1, f) * (1 - smooth(r0, r1, f));
}

/**
 * Pure function of (frame, packed horizon word) → the four demo lanes
 * (x rescue ignition, y ALWAYS 0, z horizon fade-and-fall, w ALWAYS 0).
 * Every term has attack a0 ≥ 90 (⇒ exactly 0 at frame 0) and is multiplied by
 * the master release 1−smoothstep(660, 712, f) (⇒ exactly 0 at frame 719) —
 * the machine-checked seam guarantee. No sines anywhere in this moment.
 * Keep in lockstep with forgetting_choreo in shaders/forgetting.wgsl.ts.
 */
export function forgettingEnvelopes(
	frame: number,
	packed: number
): { x: number; y: number; z: number; w: number } {
	const isDrifting = (packed & 0x100) !== 0;
	if (!isDrifting) return { x: 0, y: 0, z: 0, w: 0 };

	const rank01 = (packed & 0xff) / 255;
	const isRescued = (packed & 0x200) !== 0;
	const k = (packed >>> 10) & 0x3;

	const onset = DRIFT_ONSET_BASE + DRIFT_ONSET_SPREAD * rank01;
	const master = 1 - smooth(MASTER_R0, MASTER_R1, frame);
	const phase1 = PHASE1_LEVEL * smooth(onset, onset + DRIFT_ENGULF, frame);

	let x = 0;
	let z = 0;
	if (isRescued) {
		const rk = rescueFrame(k);
		// Snap-back starts 22 frames before the recall ribbon lands at rk —
		// the memory visibly rises to meet the call; exactly 0 from rk+6.
		z = master * phase1 * (1 - smooth(rk - 22, rk + 6, frame));
		// Ignition rides the EXISTING recall response (thin-film + white core
		// + 0.9·recall swell in render-nodes.wgsl) for free; released ≤ rk+130.
		x = master * env(frame, rk - 26, rk, rk + 60, rk + 130);
	} else {
		const phase2 =
			PHASE2_LEVEL * smooth(PHASE2_BASE + PHASE2_STAGGER * rank01, PHASE2_END, frame);
		// Plateau 0.55 by ≤342, then sink to exactly 1.0 over 640..660.
		z = master * (phase1 + phase2);
	}

	return { x, y: 0, z, w: 0 };
}

/**
 * CPU mirror of the demo-3 VERTEX displacement in render-nodes.wgsl.ts.
 * Visual-only, world-space: down (−34·dz) and away from the field axis
 * (+22·dz radially in the xz plane) ⇒ |drift| ≈ 40.5 units at dz = 1.
 * pos_radius is NEVER written — the force sim owns positions.
 */
export function horizonDrift(
	dz: number,
	p: [number, number, number]
): [number, number, number] {
	if (dz <= 0) return [0, 0, 0];
	const dzc = Math.min(1, Math.max(0, dz));
	const rXz = Math.max(Math.hypot(p[0], p[2]), 0.001);
	const ax = p[0] / rXz;
	const az = p[2] / rXz;
	return [dzc * ax * 22, dzc * -34, dzc * az * 22];
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

const UINTS_PER_STEP = 4;

function emptyPlan(nodeCount: number): ForgettingPlan {
	return {
		viable: false,
		driftingIndices: [],
		rescuedIndices: [],
		horizonData: new Uint32Array(nodeCount),
		pathData: new Uint32Array(4),
		pathMetas: [],
		spineBeats: []
	};
}

/**
 * Build the full deterministic forgetting-horizon plan. Same graph →
 * identical plan (byte-identical typed arrays). Empty/1-node graphs survive
 * with viable:false — the field breathes, nothing pretends to be forgotten.
 */
export function buildForgettingPlan(graph: ObservatoryGraph): ForgettingPlan {
	const n = graph.nodes.length;
	const drifting = pickDrifting(graph);
	if (n < 2 || drifting.length < 1) return emptyPlan(n);

	const rescued = pickRescued(graph, drifting);
	const driftCount = drifting.length;

	// --- horizonData packing (1 u32/node; non-drifting stays exactly 0) ---
	const horizonData = new Uint32Array(n);
	drifting.forEach((idx, i) => {
		const rank = Math.round((255 * i) / Math.max(1, driftCount - 1));
		horizonData[idx] = (rank & 0xff) | 0x100;
	});
	rescued.forEach((idx, k) => {
		horizonData[idx] |= 0x200 | (k << 10);
	});

	// --- PathStep emission: K recall ribbons, center → rescued_k ---
	// Window invariant: bf−46 ≥ 0 and bf+90 ≤ 719 for bf ∈ {318, 378, 438}.
	const pathData = new Uint32Array(Math.max(1, rescued.length) * UINTS_PER_STEP);
	const pathMetas: PathStepMeta[] = [];
	rescued.forEach((idx, k) => {
		const bf = rescueFrame(k);
		pathData[k * UINTS_PER_STEP + 0] = graph.centerIndex;
		pathData[k * UINTS_PER_STEP + 1] = idx;
		pathData[k * UINTS_PER_STEP + 2] = bf;
		pathData[k * UINTS_PER_STEP + 3] = PATH_KIND.recall;
		pathMetas.push({
			sourceIndex: graph.centerIndex,
			targetIndex: idx,
			beatFrame: bf,
			kind: PATH_KIND.recall,
			beatKind: 'recall',
			nodeId: graph.nodes[idx].id,
			label: truncateLabel(graph.nodes[idx].label)
		});
	});

	// --- Curated spine beats (unique, strictly increasing beatFrames) ---
	const spineBeats: PathStepMeta[] = [];
	const spine = (beatFrame: number, kind: number, label: string, nodeId: string) => {
		spineBeats.push({
			sourceIndex: graph.centerIndex,
			targetIndex: graph.centerIndex,
			beatFrame,
			kind,
			beatKind: 'horizon',
			nodeId,
			label
		});
	};

	// ≤3 lowest-retention NON-rescued drifting memories narrate the fade
	// (drifting is already retention-asc; emit only beats whose subject exists).
	const rescuedSet = new Set(rescued);
	const fading = drifting.filter((i) => !rescuedSet.has(i)).slice(0, 3);
	fading.forEach((idx, j) => {
		const pct = Math.round(graph.nodes[idx].retention * 100);
		spine(
			FADING_BASE + FADING_INTERVAL * j,
			1,
			`fading: ${truncateLabel(graph.nodes[idx].label)} · retention ${pct}%`,
			graph.nodes[idx].id
		);
	});
	rescued.forEach((idx, k) => {
		spine(rescueFrame(k), 0, `recalled: ${truncateLabel(graph.nodes[idx].label)}`, graph.nodes[idx].id);
	});
	if (fading.length > 0) {
		spine(SINK_BEAT_FRAME, 1, 'the unrecalled sink · nothing is deleted', 'horizon-sink');
	}
	spine(RETRIEVABLE_BEAT_FRAME, 0, 'every memory still retrievable', 'horizon-retrievable');

	return {
		viable: true,
		driftingIndices: drifting,
		rescuedIndices: rescued,
		horizonData,
		pathData,
		pathMetas,
		spineBeats
	};
}

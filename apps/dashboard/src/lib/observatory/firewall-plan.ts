/**
 * Cognitive Observatory — firewall demo plan (the immune response).
 *
 * Pure CPU: given the stable-indexed ObservatoryGraph + the demo seed,
 * DETERMINISTICALLY pick the intruder (prefer failure/guardrail/confusion
 * tags, else the lowest-retention leaf), precompute the per-node shockwave
 * delays from the REAL layout (layoutPositions — never reimplemented), the
 * severed-edge steps, the spine beats and the verdict copy. No Math.random(),
 * no Date.now().
 *
 * The 720-frame beat map (fixed 60Hz, seamless loop):
 *   0-90    field at rest
 *   90-150  INTRUSION — the suspicious memory flares sickly green-white
 *           (demo.y flare band (0..1], 36 integer sine cycles/loop)
 *   150-330 CRIMSON SHOCKWAVE — a radial front expands from the intruder;
 *           per-node arrival A = 150 + delay, delay = round(144·dist/maxDist)
 *           (demo.w rim, amplitude fading with distance; all rims dead by 320)
 *   330-480 QUARANTINE — probe beams to the intruder's neighbors flare then
 *           die one by one (kind 2, bf = 345 + 21k) while the MEMBRANE forms
 *           (demo.y membrane band [2.6..2.9] — the lane VALUE RANGE separates
 *           intrusion-flare (0..1] from membrane [2..3], one lane, two reads)
 *   480-620 VERDICT overlay — "threat quarantined" (RescueVerdict, tone
 *           quarantine, fadeWindow 480/495/605/620)
 *   620-720 every lane decays to EXACTLY zero (all releases r1 ≤ 680)
 *
 * `firewallEnvelopes` is the authoritative CPU mirror of
 * shaders/firewall.wgsl.ts — the seam-zero unit test machine-checks the loop
 * guarantee against it.
 */

import { FLOATS_PER_NODE, PATH_KIND, type ObservatoryGraph } from './types';
import type { PathStepMeta } from './path-builder';
import { layoutPositions, truncateLabel } from './rescue-plan';

// ---------------------------------------------------------------------------
// Beat-map constants (shared with shaders/firewall.wgsl.ts — keep in lockstep)
// ---------------------------------------------------------------------------

export const INTRUSION_FRAME = 90;
export const SHOCK_START = 150;
/** The front always crosses the field in exactly SHOCK_SPAN frames (adaptive speed). */
export const SHOCK_SPAN = 144;
export const MEMBRANE_START = 330;
/** Sever step k lands at SEVER_BASE + k * SEVER_INTERVAL → 345..450. */
export const SEVER_BASE = 345;
export const SEVER_INTERVAL = 21;
export const MAX_SEVERED = 6;
export const FIREWALL_VERDICT_START = 480;
export const FIREWALL_VERDICT_END = 620;
export const LOOP_FRAMES = 720;
/** Tags that mark a memory as suspicious (case-insensitive). */
export const INTRUDER_TAGS: readonly string[] = ['failure', 'guardrail', 'confusion'];

const TAU = Math.PI * 2;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FirewallVerdictCopy {
	headline: 'threat quarantined';
	intruderLabel: string;
	receipt: 'memory held in review · Memory PR opened';
}

export interface FirewallPlan {
	/** false ⇒ field renders at rest, story suppressed (no fake intruder). */
	viable: boolean;
	intruderIndex: number;
	/** Unique intruder neighbors, ascending index, ≤ MAX_SEVERED. */
	severedNeighborIndices: number[];
	/** Per-node shock delay, round(SHOCK_SPAN·dist/maxDist), intruder 0. */
	shockDelays: number[];
	/**
	 * 1 u32/node: bits 0-7 shockDelay (0..144), bit 8 isIntruder,
	 * bit 9 isSeverNeighbor, bits 10-13 sever slot k.
	 */
	fireData: Uint32Array;
	/** UINTS_PER_PATHSTEP u32 per step; min Uint32Array(4). */
	pathData: Uint32Array<ArrayBuffer>;
	/** MUST be 1:1 with pathData steps (setPathSteps contract). */
	pathMetas: PathStepMeta[];
	/** Curated spine beats (route state only, NEVER sent to GPU). */
	spineBeats: PathStepMeta[];
	verdict: FirewallVerdictCopy;
}

// ---------------------------------------------------------------------------
// Selection (exported for tests; all ties → ascending node index)
// ---------------------------------------------------------------------------

/**
 * Pick the intruder. Relaxation ladder, each tier ordered by
 * (retention asc, index asc):
 *   1. tagged (failure/guardrail/confusion) non-center unsuppressed
 *   2. leaf (degree ≤ 1) non-center unsuppressed
 *   3. any non-center unsuppressed
 *   4. any non-center
 * No candidate anywhere → -1 (plan viable:false).
 */
export function pickIntruderIndex(graph: ObservatoryGraph): number {
	const n = graph.nodes.length;
	if (n === 0) return -1;
	const degree = new Uint32Array(n);
	for (const e of graph.edges) {
		degree[e.sourceIndex]++;
		degree[e.targetIndex]++;
	}
	const isTagged = (i: number): boolean =>
		graph.nodes[i].tags.some((t) => INTRUDER_TAGS.includes(t.toLowerCase()));

	const tiers: Array<(i: number) => boolean> = [
		(i) => i !== graph.centerIndex && !graph.nodes[i].suppressed && isTagged(i),
		(i) => i !== graph.centerIndex && !graph.nodes[i].suppressed && degree[i] <= 1,
		(i) => i !== graph.centerIndex && !graph.nodes[i].suppressed,
		(i) => i !== graph.centerIndex
	];
	for (const accept of tiers) {
		let best = -1;
		for (let i = 0; i < n; i++) {
			if (!accept(i)) continue;
			// strict < keeps the lowest index on retention ties (ascending scan)
			if (best < 0 || graph.nodes[i].retention < graph.nodes[best].retention) best = i;
		}
		if (best >= 0) return best;
	}
	return -1;
}

/**
 * Per-node shockwave delay from LAYOUT distance to the intruder:
 * delay_i = round(SHOCK_SPAN · dist_i / maxDist), clamped [0, 255], intruder 0.
 * maxDist is floored 1e-6 → 1.0 (co-located degenerate: everything fires at
 * once — safe, deterministic). Adaptive speed: the front always crosses the
 * whole field in exactly SHOCK_SPAN frames.
 */
export function computeShockDelays(
	positions: Float32Array,
	nodeCount: number,
	intruderIndex: number
): number[] {
	const ix = positions[intruderIndex * FLOATS_PER_NODE + 0];
	const iy = positions[intruderIndex * FLOATS_PER_NODE + 1];
	const iz = positions[intruderIndex * FLOATS_PER_NODE + 2];
	const dists = new Array<number>(nodeCount);
	let maxDist = 0;
	for (let i = 0; i < nodeCount; i++) {
		const dx = positions[i * FLOATS_PER_NODE + 0] - ix;
		const dy = positions[i * FLOATS_PER_NODE + 1] - iy;
		const dz = positions[i * FLOATS_PER_NODE + 2] - iz;
		const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
		dists[i] = d;
		if (d > maxDist) maxDist = d;
	}
	if (maxDist < 1e-6) maxDist = 1.0;
	const delays = new Array<number>(nodeCount);
	for (let i = 0; i < nodeCount; i++) {
		delays[i] = Math.min(255, Math.max(0, Math.round((SHOCK_SPAN * dists[i]) / maxDist)));
	}
	delays[intruderIndex] = 0;
	return delays;
}

/**
 * The severed edges: unique intruder neighbors (self-loops excluded — also
 * already dropped by buildObservatoryGraph — and deduped), ascending index,
 * capped at MAX_SEVERED. Edgeless intruder → [] (still viable: the membrane
 * forms around a memory with nothing to sever).
 */
export function pickSeveredNeighbors(graph: ObservatoryGraph, intruderIndex: number): number[] {
	const nbrs = new Set<number>();
	for (const e of graph.edges) {
		if (e.sourceIndex === intruderIndex && e.targetIndex !== intruderIndex) {
			nbrs.add(e.targetIndex);
		}
		if (e.targetIndex === intruderIndex && e.sourceIndex !== intruderIndex) {
			nbrs.add(e.sourceIndex);
		}
	}
	return Array.from(nbrs)
		.sort((a, b) => a - b)
		.slice(0, MAX_SEVERED);
}

export function severFrame(k: number): number {
	return SEVER_BASE + SEVER_INTERVAL * k;
}

// ---------------------------------------------------------------------------
// Envelope math — the authoritative CPU mirror of shaders/firewall.wgsl.ts
// ---------------------------------------------------------------------------

function smooth(a: number, b: number, f: number): number {
	const t = Math.min(1, Math.max(0, (f - a) / (b - a)));
	return t * t * (3 - 2 * t);
}

function env(f: number, a0: number, a1: number, r0: number, r1: number): number {
	return smooth(a0, a1, f) * (1 - smooth(r0, r1, f));
}

/**
 * Pure function of (frame, packed fire word) → the four demo lanes
 * (x ALWAYS 0, y intruder flare/membrane, z ALWAYS 0, w shock rim/sever blink).
 * Every envelope has attack a0 ≥ 90 and release r1 ≤ 680 ⇒ exactly 0 at
 * frames 0 and 719 — the machine-checked seam guarantee. Sines are factors on
 * zero-at-seam envelopes and run INTEGER cycles per loop (36 flare, 12
 * membrane). demo.y value ranges: intrusion flare (0..1], membrane
 * [2.60..2.90] — the fragment shader separates the two reads by range.
 * Keep in lockstep with firewall_choreo in shaders/firewall.wgsl.ts.
 */
export function firewallEnvelopes(
	frame: number,
	packed: number
): { x: number; y: number; z: number; w: number } {
	const delay = packed & 0xff;
	const isIntruder = (packed & 0x100) !== 0;
	const isSever = (packed & 0x200) !== 0;
	const k = (packed >>> 10) & 0xf;
	const loopPhase = frame / LOOP_FRAMES;

	let y = 0;
	let w = 0;
	if (isIntruder) {
		// Intrusion flare: sickly strobe, band (0..1]. C¹ handoff into the
		// membrane over 330-332 — the rise sweeps the flare band exactly once
		// (the condensation read is intentional).
		y = env(frame, 90, 96, 310, 332) * (0.55 + 0.45 * Math.sin(TAU * 36 * loopPhase));
		// Membrane: sustained ring band [2.60..2.90], slow shimmer.
		y += env(frame, 330, 352, 620, 680) * (2.75 + 0.15 * Math.sin(TAU * 12 * loopPhase));
		// Source detonation as the front leaves.
		w = env(frame, 148, 153, 162, 196);
	} else {
		const a = SHOCK_START + delay;
		const amp = 0.9 - 0.45 * (delay / SHOCK_SPAN);
		// Crimson rim as the front passes; A ∈ [150, 294] ⇒ dead by 320 < 330.
		w = amp * env(frame, a - 2, a + 3, a + 8, a + 26);
		if (isSever) {
			// Node-side receipt of the severed edge; last release 450+24 = 474.
			const sk = severFrame(k);
			w += 0.6 * env(frame, sk - 4, sk, sk + 6, sk + 24);
		}
	}
	// x and z are hard 0.0: the recall/thin-film and horizon grammars can
	// never fire in demo 4 (enforced by the lane-hygiene test).
	return { x: 0, y, z: 0, w };
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

const UINTS_PER_STEP = 4;

/** An idle (viable:false) firewall plan — the field breathes, nothing fires. */
export function emptyFirewallPlan(nodeCount: number): FirewallPlan {
	return emptyPlan(nodeCount);
}

function emptyPlan(nodeCount: number): FirewallPlan {
	return {
		viable: false,
		intruderIndex: -1,
		severedNeighborIndices: [],
		shockDelays: [],
		fireData: new Uint32Array(nodeCount),
		pathData: new Uint32Array(4),
		pathMetas: [],
		spineBeats: [],
		verdict: {
			headline: 'threat quarantined',
			intruderLabel: '',
			receipt: 'memory held in review · Memory PR opened'
		}
	};
}

/**
 * Build the full deterministic firewall plan. Same graph + seed → identical
 * plan (byte-identical typed arrays). Empty/center-only graphs survive with
 * viable:false — the field breathes, nothing pretends to be a threat.
 */
export function buildFirewallPlan(graph: ObservatoryGraph, seed: string): FirewallPlan {
	return buildFirewallPlanFor(graph, seed, pickIntruderIndex(graph));
}

/**
 * v2.3 living field — build a firewall plan for a SPECIFIC intruder (a real
 * MemorySuppressed target, or a real contradiction-pair member) instead of the
 * deterministic demo pick. Everything downstream (shock delays from the real
 * layout, severed real neighbors, spine beats, verdict) is identical — only the
 * intruder is supplied by the live event. The neighbors are the intruder's REAL
 * graph edges (pickSeveredNeighbors), so the quarantine ring severs true
 * connections, never invented ones. Returns viable:false if the intruder isn't
 * in the field (nothing pretends).
 */
export function buildLiveFirewallPlan(
	graph: ObservatoryGraph,
	seed: string,
	intruderIndex: number
): FirewallPlan {
	if (intruderIndex < 0 || intruderIndex >= graph.nodes.length) {
		return emptyPlan(graph.nodes.length);
	}
	return buildFirewallPlanFor(graph, seed, intruderIndex);
}

function buildFirewallPlanFor(
	graph: ObservatoryGraph,
	seed: string,
	intruderIndex: number
): FirewallPlan {
	const n = graph.nodes.length;
	if (n === 0 || intruderIndex < 0) return emptyPlan(n);

	// Shock delays come from the REAL layout (rescue-plan.layoutPositions —
	// byte-identical to NodeRenderer.upload, never reimplemented).
	const positions = layoutPositions(graph, seed);
	const shockDelays = computeShockDelays(positions, n, intruderIndex);
	const severed = pickSeveredNeighbors(graph, intruderIndex);

	// --- fireData packing (1 u32/node; every node carries its shock delay) ---
	const fireData = new Uint32Array(n);
	for (let i = 0; i < n; i++) fireData[i] = shockDelays[i] & 0xff;
	fireData[intruderIndex] = 0x100; // intruder: delay forced 0
	severed.forEach((idx, k) => {
		fireData[idx] |= 0x200 | (k << 10);
	});

	// --- PathStep emission: probe beams (kind 2) intruder → neighbor, one per
	// severed edge — they flare then die (the visible severing).
	// Window invariant: bf−46 ≥ 0 and bf+90 ≤ 719 for bf ∈ [345, 450].
	const pathData = new Uint32Array(Math.max(1, severed.length) * UINTS_PER_STEP);
	const pathMetas: PathStepMeta[] = [];
	severed.forEach((idx, k) => {
		const bf = severFrame(k);
		pathData[k * UINTS_PER_STEP + 0] = intruderIndex;
		pathData[k * UINTS_PER_STEP + 1] = idx;
		pathData[k * UINTS_PER_STEP + 2] = bf;
		pathData[k * UINTS_PER_STEP + 3] = PATH_KIND.probe;
		pathMetas.push({
			sourceIndex: intruderIndex,
			targetIndex: idx,
			beatFrame: bf,
			kind: PATH_KIND.probe,
			beatKind: 'sever',
			nodeId: graph.nodes[idx].id,
			label: truncateLabel(graph.nodes[idx].label)
		});
	});

	// --- Curated spine beats (unique, strictly increasing beatFrames) ---
	const intruderLabel = truncateLabel(graph.nodes[intruderIndex].label);
	const spineBeats: PathStepMeta[] = [];
	const spine = (beatFrame: number, label: string, nodeId: string) => {
		spineBeats.push({
			sourceIndex: intruderIndex,
			targetIndex: intruderIndex,
			beatFrame,
			kind: 1,
			beatKind: 'firewall',
			nodeId,
			label
		});
	};
	spine(INTRUSION_FRAME, `intrusion · ${intruderLabel}`, graph.nodes[intruderIndex].id);
	spine(SHOCK_START, 'immune response · shockwave', 'firewall-shock');
	spine(MEMBRANE_START, 'membrane forming', 'firewall-membrane');
	severed.forEach((idx, k) => {
		spine(severFrame(k), `edge severed ✗ · ${truncateLabel(graph.nodes[idx].label)}`, graph.nodes[idx].id);
	});
	spine(FIREWALL_VERDICT_START, 'threat quarantined', 'firewall-verdict');

	const verdict: FirewallVerdictCopy = {
		headline: 'threat quarantined',
		intruderLabel,
		receipt: 'memory held in review · Memory PR opened'
	};

	return {
		viable: true,
		intruderIndex,
		severedNeighborIndices: severed,
		shockDelays,
		fireData,
		pathData,
		pathMetas,
		spineBeats,
		verdict
	};
}

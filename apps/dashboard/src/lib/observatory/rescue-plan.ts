/**
 * Cognitive Observatory — Retroactive Salience Backfill demo plan (salience-rescue).
 *
 * Pure CPU: given the API GraphResponse + the stable-indexed ObservatoryGraph +
 * the demo seed, DETERMINISTICALLY pick the failure memory, the K=4 lookalikes
 * (nearest in LAYOUT space — exactly what vector search would return), the true
 * cause (graph-distance ≥ 3, low retention, old), the BFS hop-depth map for the
 * backward wave, the PathSteps (probe beams, wave tree, causal arc), the spine
 * beats, and the verdict copy. No Math.random(), no Date.now() — Date.parse is
 * only applied to DATA timestamps.
 *
 * The 720-frame beat map (fixed 60Hz, seamless loop):
 *   0-90    field at rest
 *   90-120  DETONATION — failure flares crimson (demo.w)
 *   120-260 SEARCHLIGHT — K lookalikes flare cold-white sequentially (demo.y)
 *   260-520 BACKWARD WAVE — hop-by-hop interrogation away from the failure (demo.z)
 *   520-600 CAUSE IGNITES — thin-film recall envelope (demo.x) + causal arc
 *   600-660 VERDICT — DOM overlay
 *   660-720 decay to rest — every envelope is exactly 0 at frames 0 and 719
 *
 * `rescueEnvelopes` is the authoritative CPU mirror of shaders/rescue.wgsl.ts —
 * the seam-zero unit test machine-checks the loop guarantee against it.
 */

import type { GraphResponse } from '$types';
import { DemoClock } from './demo-clock';
import { buildNodeStateArray } from './graph-upload';
import { FLOATS_PER_NODE, PATH_KIND, type ObservatoryGraph } from './types';
import type { PathStepMeta } from './path-builder';

// ---------------------------------------------------------------------------
// Beat-map constants (shared with shaders/rescue.wgsl.ts via RescueShaderConsts)
// ---------------------------------------------------------------------------

export const RESCUE_K = 4;
export const DETONATE_FRAME = 90;
/** Lookalike k flares at LOOKALIKE_BASE + k * LOOKALIKE_INTERVAL → 138,166,194,222. */
export const LOOKALIKE_BASE = 138;
export const LOOKALIKE_INTERVAL = 28;
export const WAVE_START = 260;
export const WAVE_ARRIVAL_CAP = 514;
export const ARC_FRAME = 560;
export const VERDICT_START = 600;
export const VERDICT_END = 660;
export const UNREACHED = 0xffff;
export const MAX_WAVE_STEPS = 48;
export const LOOP_FRAMES = 720;

/** BFS parent preference: causal chains first — RSB semantics. Unknown → 5. */
export const EDGE_TYPE_RANK: Record<string, number> = {
	causal: 0,
	temporal: 1,
	shared_concepts: 2,
	complementary: 3,
	semantic: 4
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface RescueShaderConsts {
	/** Frames per BFS hop for the backward wave: clamp(⌊252/D⌋, 14, 84). */
	hopSlot: number;
	/** BFS depth of the cause (D). Wave lane fires only for 1 ≤ d ≤ D. */
	causeDepth: number;
}

export interface RescueVerdictCopy {
	/**
	 * Always a CANDIDATE. The heuristic (no-receipt) walk and the receipt-backed
	 * walk both surface a candidate cause; nothing in the field asserts a root
	 * cause automatically. Product rule: candidate causes, never automatic
	 * root cause.
	 */
	headline: 'candidate cause found';
	causeLabel: string;
	failureLabel: string;
	causeDate: string;
	hops: number;
	k: number;
	/** `${hops} hops back · ${causeDate} · vector search: 0 for ${k}` */
	receipt: string;
}

/** Exact evidence returned by the Backfill receipt. The scene must not infer
 * substitutes for any of these ids. Baseline probes are intentionally absent
 * until the backend supplies actual baseline candidates. */
export interface ReceiptBackfillEvidence {
	failureId: string;
	/** Ordered, backend-recorded candidate → failure route. This is required
	 * for a proof replay: any missing/in-field mismatch makes the plan
	 * non-viable instead of allowing topology inference. */
	pathIds?: string[];
	candidates: Array<{
		memoryId: string;
		sharedEntities: string[];
		ageDays: number;
		similarityRank: number | null;
		promoted: boolean;
	}>;
}

export interface RescuePlan {
	/** false ⇒ field renders, story suppressed (no fake cause on tiny graphs). */
	viable: boolean;
	failureIndex: number;
	causeIndex: number;
	lookalikeIndices: number[];
	/** Per-node BFS hop depth from the failure; UNREACHED; failure = 0. */
	hopDepths: Uint16Array;
	causeDepth: number;
	hopSlot: number;
	/** 1 u32/node: bits 0-15 hopDepth, 16 isFailure, 17 isCause, 18 isLookalike, 19-21 k. */
	waveData: Uint32Array;
	/** UINTS_PER_PATHSTEP u32 per step; min Uint32Array(4). */
	pathData: Uint32Array<ArrayBuffer>;
	/**
	 * MUST be 1:1 with pathData steps — setPathSteps sets params[4] and the
	 * ribbon draw count from THIS array's length.
	 */
	pathMetas: PathStepMeta[];
	/** Curated spine beats (route state only, NEVER sent to GPU). Unique beatFrames. */
	spineBeats: PathStepMeta[];
	verdict: RescueVerdictCopy;
	consts: RescueShaderConsts;
}

// ---------------------------------------------------------------------------
// Layout parity — the ONE correct way to know where nodes are on screen
// ---------------------------------------------------------------------------

/**
 * Byte-identical replica of NodeRenderer.upload's layout: same function
 * (buildNodeStateArray), same fresh DemoClock, same rng consumption (incl. the
 * center-node skip). Do NOT reimplement golden-angle placement here.
 */
export function layoutPositions(graph: ObservatoryGraph, seed: string): Float32Array {
	const layoutClock = new DemoClock({ seed });
	return buildNodeStateArray(graph, layoutClock.state.rng).data;
}

// ---------------------------------------------------------------------------
// Selection algorithms (all exported for tests; ties → ascending node index)
// ---------------------------------------------------------------------------

function nodeDegrees(graph: ObservatoryGraph): Uint32Array {
	const degree = new Uint32Array(graph.nodes.length);
	for (const e of graph.edges) {
		degree[e.sourceIndex]++;
		degree[e.targetIndex]++;
	}
	return degree;
}

/**
 * Pick the failure memory: non-center, unsuppressed, well-connected, prefer
 * failure/guardrail (then confusion/weak-spot) tags, prefer nodes away from
 * the field center in layout (≥ 0.45 · fieldRadius 120 = 54).
 * Relaxation ladder: degree ≥ 2 → any unsuppressed non-center → any
 * non-center → center (degenerate single-node field). Empty graph → -1.
 */
export function pickFailureIndex(graph: ObservatoryGraph, positions: Float32Array): number {
	const n = graph.nodes.length;
	if (n === 0) return -1;
	const degree = nodeDegrees(graph);

	const score = (i: number): number => {
		const node = graph.nodes[i];
		const tags = new Set(node.tags.map((t) => t.toLowerCase()));
		let s = 0;
		if (tags.has('failure') || tags.has('guardrail')) s += 3;
		if (tags.has('confusion') || tags.has('weak-spot')) s += 2;
		s += Math.min(degree[i], 8) / 8;
		const x = positions[i * FLOATS_PER_NODE + 0];
		const y = positions[i * FLOATS_PER_NODE + 1];
		const z = positions[i * FLOATS_PER_NODE + 2];
		if (Math.sqrt(x * x + y * y + z * z) >= 54) s += 0.5;
		return s;
	};

	const tiers: Array<(i: number) => boolean> = [
		(i) => i !== graph.centerIndex && !graph.nodes[i].suppressed && degree[i] >= 2,
		(i) => i !== graph.centerIndex && !graph.nodes[i].suppressed,
		(i) => i !== graph.centerIndex,
		() => true
	];
	for (const accept of tiers) {
		let best = -1;
		let bestScore = -Infinity;
		for (let i = 0; i < n; i++) {
			if (!accept(i)) continue;
			const s = score(i);
			// strict > keeps the lowest index on ties (ascending scan)
			if (s > bestScore) {
				bestScore = s;
				best = i;
			}
		}
		if (best >= 0) return best;
	}
	return -1;
}

/**
 * Undirected BFS from the failure. Hop depths are TRUE BFS distances
 * (layer order is edge-rank independent); parent pointers are chosen in a
 * second pass preferring causal/temporal edges (RSB semantics), then lowest
 * neighbor index — fully deterministic.
 */
export function bfsFromFailure(
	graph: ObservatoryGraph,
	failureIndex: number
): { depths: Uint16Array; parents: Int32Array } {
	const n = graph.nodes.length;
	const depths = new Uint16Array(n).fill(UNREACHED);
	const parents = new Int32Array(n).fill(-1);
	if (failureIndex < 0 || failureIndex >= n) return { depths, parents };

	const adj: Array<Array<{ nbr: number; rank: number }>> = Array.from({ length: n }, () => []);
	for (const e of graph.edges) {
		const rank = EDGE_TYPE_RANK[e.type] ?? 5;
		adj[e.sourceIndex].push({ nbr: e.targetIndex, rank });
		adj[e.targetIndex].push({ nbr: e.sourceIndex, rank });
	}
	for (const list of adj) list.sort((a, b) => a.rank - b.rank || a.nbr - b.nbr);

	depths[failureIndex] = 0;
	const queue = [failureIndex];
	for (let qi = 0; qi < queue.length; qi++) {
		const u = queue[qi];
		for (const { nbr } of adj[u]) {
			if (depths[nbr] === UNREACHED) {
				depths[nbr] = depths[u] + 1;
				queue.push(nbr);
			}
		}
	}

	// Parent pass: first depth-(d-1) neighbor in (rank, index) order wins.
	for (let v = 0; v < n; v++) {
		if (depths[v] === UNREACHED || depths[v] === 0) continue;
		for (const { nbr } of adj[v]) {
			if (depths[nbr] === depths[v] - 1) {
				parents[v] = nbr;
				break;
			}
		}
	}

	return { depths, parents };
}

/**
 * Pick the true cause: graph-distant (depth ≥ 3, relaxed 2 → 1), low
 * retention (hard ≤ 0.45, relaxed away if empty — visually slate/dim), old.
 * score = 2·(1−retention) + 0.5·min(depth,6)/6 + 0.5·ageRank, ageRank ∈ [0,1]
 * with the oldest createdAt → 1 (invalid/missing dates → 0).
 * Sort (score desc, depth desc, index asc). No candidate at any depth ⇒ -1.
 */
export function pickCauseIndex(
	response: GraphResponse,
	graph: ObservatoryGraph,
	depths: Uint16Array,
	failureIndex: number
): { index: number; depth: number } {
	const createdById = new Map<string, string>();
	for (const nd of response.nodes) createdById.set(nd.id, nd.createdAt);

	for (const minDepth of [3, 2, 1]) {
		const cand: number[] = [];
		for (let i = 0; i < graph.nodes.length; i++) {
			if (i === graph.centerIndex || i === failureIndex) continue;
			const d = depths[i];
			if (d === UNREACHED || d < minDepth) continue;
			cand.push(i);
		}
		if (cand.length === 0) continue;

		let pool = cand.filter((i) => graph.nodes[i].retention <= 0.45);
		if (pool.length === 0) pool = cand;

		// ageRank across the pool: oldest Date.parse(createdAt) → 1.
		const times = new Map<number, number>();
		let minT = Infinity;
		let maxT = -Infinity;
		for (const i of pool) {
			const raw = createdById.get(graph.nodes[i].id);
			const t = raw ? Date.parse(raw) : NaN;
			if (Number.isFinite(t)) {
				times.set(i, t);
				if (t < minT) minT = t;
				if (t > maxT) maxT = t;
			}
		}
		const ageRank = (i: number): number => {
			const t = times.get(i);
			if (t === undefined) return 0;
			if (maxT === minT) return 1;
			return (maxT - t) / (maxT - minT);
		};
		const score = (i: number): number =>
			2 * (1 - graph.nodes[i].retention) + (0.5 * Math.min(depths[i], 6)) / 6 + 0.5 * ageRank(i);

		pool.sort((a, b) => {
			const sa = score(a);
			const sb = score(b);
			if (sb !== sa) return sb - sa;
			if (depths[b] !== depths[a]) return depths[b] - depths[a];
			return a - b;
		});
		return { index: pool[0], depth: depths[pool[0]] };
	}
	return { index: -1, depth: 0 };
}

/**
 * K = min(4, eligible) nearest neighbors of the failure in LAYOUT space,
 * excluding failure/cause/center — "looks similar" is literal: these are the
 * nodes vector search would return. Flare order = nearest first.
 */
export function pickLookalikes(
	positions: Float32Array,
	nodeCount: number,
	failureIndex: number,
	causeIndex: number,
	centerIndex: number
): number[] {
	const fx = positions[failureIndex * FLOATS_PER_NODE + 0];
	const fy = positions[failureIndex * FLOATS_PER_NODE + 1];
	const fz = positions[failureIndex * FLOATS_PER_NODE + 2];
	const cand: Array<{ i: number; d2: number }> = [];
	for (let i = 0; i < nodeCount; i++) {
		if (i === failureIndex || i === causeIndex || i === centerIndex) continue;
		const dx = positions[i * FLOATS_PER_NODE + 0] - fx;
		const dy = positions[i * FLOATS_PER_NODE + 1] - fy;
		const dz = positions[i * FLOATS_PER_NODE + 2] - fz;
		cand.push({ i, d2: dx * dx + dy * dy + dz * dz });
	}
	cand.sort((a, b) => a.d2 - b.d2 || a.i - b.i);
	return cand.slice(0, RESCUE_K).map((c) => c.i);
}

// ---------------------------------------------------------------------------
// Wave timing
// ---------------------------------------------------------------------------

/** σ = clamp(⌊252/D⌋, 14, 84): the wave reaches the cause 6-10 frames before ignition. */
export function hopSlotFor(causeDepth: number): number {
	const d = Math.max(1, causeDepth);
	return Math.min(84, Math.max(14, Math.floor(252 / d)));
}

/** W(d) = min(WAVE_START + σ·d, WAVE_ARRIVAL_CAP). */
export function waveArrivalFrame(depth: number, hopSlot: number): number {
	return Math.min(WAVE_START + hopSlot * depth, WAVE_ARRIVAL_CAP);
}

export function lookalikeFrame(k: number): number {
	return LOOKALIKE_BASE + LOOKALIKE_INTERVAL * k;
}

export function truncateLabel(label: string): string {
	return label.length > 64 ? label.slice(0, 64) + '…' : label;
}

// ---------------------------------------------------------------------------
// Envelope math — the authoritative CPU mirror of shaders/rescue.wgsl.ts
// ---------------------------------------------------------------------------

function smooth(a: number, b: number, f: number): number {
	const t = Math.min(1, Math.max(0, (f - a) / (b - a)));
	return t * t * (3 - 2 * t);
}

function env(f: number, a0: number, a1: number, r0: number, r1: number): number {
	return smooth(a0, a1, f) * (1 - smooth(r0, r1, f));
}

/**
 * Pure function of (frame, packed wave word, shader consts) → the four demo
 * lanes (x recall/ignition, y searchlight, z wave, w shock). Every term is
 * A·S(a0,a1,f)·(1−S(r0,r1,f)) with all a0 ≥ 88 and all r1 ≤ 700, so the value
 * is EXACTLY zero at frames 0 and 719 — the machine-checked seam guarantee.
 * Keep in lockstep with rescue_choreo in shaders/rescue.wgsl.ts.
 */
export function rescueEnvelopes(
	frame: number,
	packed: number,
	c: RescueShaderConsts
): { x: number; y: number; z: number; w: number } {
	const depth = packed & 0xffff;
	const isFailure = (packed & 0x10000) !== 0;
	const isCause = (packed & 0x20000) !== 0;
	const isLook = (packed & 0x40000) !== 0;
	const lookK = (packed >>> 19) & 0x7;
	const loopPhase = frame / LOOP_FRAMES;

	let x = 0;
	let y = 0;
	let z = 0;
	let w = 0;

	if (isFailure) {
		// Detonation spike + wound simmer (burns through the investigation)
		// + recognition flare as the causal arc lands at 560.
		w += env(frame, 90, 96, 120, 168);
		w += 0.35 * env(frame, 100, 130, 600, 656);
		w += 0.35 * env(frame, 552, 562, 580, 640);
		// Symptom backlight while the cause burns.
		x += 0.4 * env(frame, 556, 566, 620, 668);
	}
	if (!isFailure && depth >= 1 && depth <= 12) {
		// Shockwave blink: crimson concussion at 3 frames/hop along REAL graph distance.
		w +=
			0.75 *
			Math.exp(-0.3 * depth) *
			env(frame, 92 + 3 * depth, 96 + 3 * depth, 96 + 3 * depth, 122 + 3 * depth);
	}
	if (isLook) {
		const fk = lookalikeFrame(lookK);
		// Searchlight flare: cold pop, sequential, on camera.
		y += env(frame, fk - 6, fk, fk + 10, fk + 26);
		// Ash residue: the struck-through lookalike stays in frame until the verdict.
		y += 0.15 * smooth(fk + 10, fk + 26, frame) * (1 - smooth(600, 656, frame));
	}
	if (!isFailure && depth >= 1 && depth <= c.causeDepth) {
		const wd = waveArrivalFrame(depth, c.hopSlot);
		// Interrogation flicker: 24 integer sine cycles per loop, per-depth phase.
		const flicker = 0.75 + 0.25 * Math.sin(2 * Math.PI * 24 * loopPhase + 1.7 * depth);
		z += env(frame, wd - 10, wd, wd + 28, wd + 64) * flicker;
		// Scanned ember.
		z += 0.08 * smooth(wd + 28, wd + 64, frame) * (1 - smooth(580, 640, frame));
	}
	if (isCause) {
		// Cause ignition rides the EXISTING recall response in render-nodes.wgsl:
		// spectral() thin-film band + white-hot core + sprite swell, for free.
		x += env(frame, 520, 546, 640, 700);
	}

	return { x, y, z, w };
}

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

const UINTS_PER_STEP = 4;

function emptyPlan(nodeCount: number): RescuePlan {
	const waveData = new Uint32Array(nodeCount);
	waveData.fill(UNREACHED);
	return {
		viable: false,
		failureIndex: -1,
		causeIndex: -1,
		lookalikeIndices: [],
		hopDepths: new Uint16Array(nodeCount).fill(UNREACHED),
		causeDepth: 0,
		hopSlot: hopSlotFor(3),
		waveData,
		pathData: new Uint32Array(4),
		pathMetas: [],
		spineBeats: [],
		verdict: {
			headline: 'candidate cause found',
			causeLabel: '',
			failureLabel: '',
			causeDate: '',
			hops: 0,
			k: 0,
			receipt: ''
		},
		consts: { hopSlot: hopSlotFor(3), causeDepth: 3 }
	};
}

/**
 * Build the full deterministic salience-rescue plan. Same graph + seed →
 * identical plan (byte-identical typed arrays). Empty/tiny/edgeless graphs
 * survive with viable:false — the field breathes, no fake cause.
 */
export function buildRescuePlan(
	response: GraphResponse,
	graph: ObservatoryGraph,
	seed: string,
	evidence?: ReceiptBackfillEvidence
): RescuePlan {
	if (evidence) return buildReceiptRescuePlan(response, graph, evidence);
	const n = graph.nodes.length;
	if (n === 0) return emptyPlan(0);

	const positions = layoutPositions(graph, seed);
	const failureIndex = pickFailureIndex(graph, positions);
	if (failureIndex < 0) return emptyPlan(n);

	const { depths, parents } = bfsFromFailure(graph, failureIndex);
	const cause = pickCauseIndex(response, graph, depths, failureIndex);
	if (cause.index < 0) {
		const plan = emptyPlan(n);
		plan.failureIndex = failureIndex;
		plan.hopDepths = depths;
		return plan;
	}

	const causeIndex = cause.index;
	const causeDepth = Math.max(1, cause.depth);
	const hopSlot = hopSlotFor(causeDepth);
	const W = (d: number) => waveArrivalFrame(d, hopSlot);

	const lookalikeIndices = pickLookalikes(positions, n, failureIndex, causeIndex, graph.centerIndex);
	const K = lookalikeIndices.length;

	// --- waveData packing (1 u32/node) ---
	const waveData = new Uint32Array(n);
	for (let i = 0; i < n; i++) {
		let word = depths[i] & 0xffff;
		if (i === failureIndex) word |= 1 << 16;
		if (i === causeIndex) word |= 1 << 17;
		waveData[i] = word;
	}
	lookalikeIndices.forEach((li, k) => {
		waveData[li] |= (1 << 18) | (k << 19);
	});

	// --- PathStep emission: probes, wave tree, arc ---
	interface Step {
		src: number;
		dst: number;
		bf: number;
		kind: number;
		beatKind: string;
	}
	const steps: Step[] = [];

	// Probe beams (vector search visibly probing and failing ON CAMERA).
	lookalikeIndices.forEach((li, k) => {
		steps.push({
			src: failureIndex,
			dst: li,
			bf: lookalikeFrame(k),
			kind: PATH_KIND.probe,
			beatKind: 'probe'
		});
	});

	// Wave tree, capped at MAX_WAVE_STEPS with the failure→cause parent chain
	// guaranteed, then remaining by (depth asc, index asc).
	const chainNodes: number[] = [];
	{
		let v = causeIndex;
		while (v !== failureIndex && v >= 0 && parents[v] >= 0) {
			chainNodes.push(v);
			v = parents[v];
		}
	}
	const chainSet = new Set(chainNodes);
	const rest: number[] = [];
	for (let v = 0; v < n; v++) {
		if (v === failureIndex || chainSet.has(v)) continue;
		const d = depths[v];
		if (d === UNREACHED || d < 1 || d > causeDepth) continue;
		if (parents[v] < 0) continue;
		rest.push(v);
	}
	rest.sort((a, b) => depths[a] - depths[b] || a - b);
	const waveNodes = [...chainNodes.slice().reverse(), ...rest].slice(0, MAX_WAVE_STEPS);
	waveNodes.sort((a, b) => depths[a] - depths[b] || a - b);
	for (const v of waveNodes) {
		steps.push({
			src: parents[v],
			dst: v,
			bf: W(depths[v]),
			kind: PATH_KIND.backwardCause,
			beatKind: 'wave'
		});
	}

	// The causal arc: cause → failure, lands at 560 (kind 1 = crimson-magenta tint).
	steps.push({
		src: causeIndex,
		dst: failureIndex,
		bf: ARC_FRAME,
		kind: PATH_KIND.backwardCause,
		beatKind: 'arc'
	});

	const pathData = new Uint32Array(Math.max(1, steps.length) * UINTS_PER_STEP);
	const pathMetas: PathStepMeta[] = [];
	steps.forEach((s, i) => {
		pathData[i * UINTS_PER_STEP + 0] = s.src;
		pathData[i * UINTS_PER_STEP + 1] = s.dst;
		pathData[i * UINTS_PER_STEP + 2] = s.bf;
		pathData[i * UINTS_PER_STEP + 3] = s.kind;
		pathMetas.push({
			sourceIndex: s.src,
			targetIndex: s.dst,
			beatFrame: s.bf,
			kind: s.kind,
			beatKind: s.beatKind,
			nodeId: graph.nodes[s.dst].id,
			label: truncateLabel(graph.nodes[s.dst].label)
		});
	});

	// --- Curated spine beats (unique, strictly increasing beatFrames) ---
	const failureLabel = truncateLabel(graph.nodes[failureIndex].label);
	const causeLabel = truncateLabel(graph.nodes[causeIndex].label);
	const spineBeats: PathStepMeta[] = [];
	const spine = (beatFrame: number, kind: number, label: string, nodeId: string) => {
		spineBeats.push({
			sourceIndex: failureIndex,
			targetIndex: failureIndex,
			beatFrame,
			kind,
			beatKind: 'rescue',
			nodeId,
			label
		});
	};
	spine(DETONATE_FRAME, 1, `failure: ${failureLabel}`, graph.nodes[failureIndex].id);
	lookalikeIndices.forEach((li, k) => {
		spine(lookalikeFrame(k), 0, `lookalike ✗ · ${truncateLabel(graph.nodes[li].label)}`, graph.nodes[li].id);
	});
	spine(W(1), 1, 'reaching backward through time', 'rescue-wave-start');
	if (causeDepth >= 2 && W(causeDepth) !== W(1)) {
		spine(W(causeDepth), 1, `scrubbing past · ${causeDepth} hops`, 'rescue-wave-deep');
	}
	spine(ARC_FRAME, 1, `causal arc · ${causeLabel}`, graph.nodes[causeIndex].id);
	spine(VERDICT_START, 1, 'candidate cause found', 'rescue-verdict');

	// --- Verdict copy (REAL memory labels + real date) ---
	const createdAt = response.nodes.find((nd) => nd.id === graph.nodes[causeIndex].id)?.createdAt ?? '';
	const causeDate = createdAt ? createdAt.slice(0, 10) : '';
	const verdict: RescueVerdictCopy = {
		headline: 'candidate cause found',
		causeLabel,
		failureLabel,
		causeDate,
		hops: causeDepth,
		k: K,
		// Honest provenance: this walk is a deterministic heuristic over the
		// field's own edges. It has no persisted receipt, and the copy says so.
		receipt: `${causeDepth} hops back · ${causeDate} · heuristic, no receipt · vector search: 0 for ${K}`
	};

	return {
		viable: true,
		failureIndex,
		causeIndex,
		lookalikeIndices,
		hopDepths: depths,
		causeDepth,
		hopSlot,
		waveData,
		pathData,
		pathMetas,
		spineBeats,
		verdict,
		consts: { hopSlot, causeDepth }
	};
}

/**
 * Receipt-first Backfill replay. Unlike the cinematic fallback above this
 * accepts no layout/age-selected stand-ins: if the exact failure or candidate
 * is absent from the field it returns non-viable and the host shows the receipt
 * rather than inventing a magenta route.
 */
export function buildReceiptRescuePlan(
	response: GraphResponse,
	graph: ObservatoryGraph,
	evidence: ReceiptBackfillEvidence
): RescuePlan {
	const n = graph.nodes.length;
	const failureIndex = graph.indexById.get(evidence.failureId) ?? -1;
	const pathIds = evidence.pathIds ?? [];
	// Receipt proof has a deliberately hard contract. A route is a recorded
	// fact, not an invitation to run a graph search. Require it to be ordered,
	// complete, and to land on the receipt's failure.
	if (
		failureIndex < 0 ||
		pathIds.length < 2 ||
		pathIds[pathIds.length - 1] !== evidence.failureId ||
		new Set(pathIds).size !== pathIds.length
	) return emptyPlan(n);
	const pathIndices = pathIds.map((id) => graph.indexById.get(id));
	if (pathIndices.some((index) => index === undefined)) return emptyPlan(n);
	const indexedPath = pathIndices as number[];
	const causeIndex = indexedPath[0];
	if (causeIndex === failureIndex) return emptyPlan(n);
	const cause = evidence.candidates.find((candidate) => candidate.memoryId === pathIds[0]);
	// If the route's origin isn't one of the receipt candidates, the receipt is
	// internally inconsistent. Keep the field honest and show the raw receipt.
	if (!cause) return emptyPlan(n);

	const hopDepths = new Uint16Array(n);
	hopDepths.fill(UNREACHED);
	hopDepths[failureIndex] = 0;
	indexedPath.forEach((index, i) => {
		hopDepths[index] = indexedPath.length - 1 - i;
	});
	const waveData = new Uint32Array(n);
	waveData[failureIndex] = 1 << 16;
	indexedPath.slice(0, -1).forEach((index) => {
		waveData[index] = hopDepths[index];
	});
	waveData[causeIndex] |= 1 << 17;
	const causeDepth = indexedPath.length - 1;
	const hopSlot = hopSlotFor(causeDepth);
	const causeLabel = truncateLabel(graph.nodes[causeIndex].label);
	const failureLabel = truncateLabel(graph.nodes[failureIndex].label);
	const causeDate = response.nodes.find((node) => node.id === cause.memoryId)?.createdAt?.slice(0, 10) ?? '';
	const pathData = new Uint32Array((indexedPath.length - 1) * UINTS_PER_STEP);
	const pathMetas: PathStepMeta[] = indexedPath.slice(0, -1).map((sourceIndex, i) => {
		const targetIndex = indexedPath[i + 1];
		const beatFrame = WAVE_START + i * hopSlot;
		pathData[i * UINTS_PER_STEP] = sourceIndex;
		pathData[i * UINTS_PER_STEP + 1] = targetIndex;
		pathData[i * UINTS_PER_STEP + 2] = beatFrame;
		pathData[i * UINTS_PER_STEP + 3] = PATH_KIND.backwardCause;
		return {
			sourceIndex,
			targetIndex,
			beatFrame,
			kind: PATH_KIND.backwardCause,
			beatKind: 'receipt-path',
			nodeId: pathIds[i + 1],
			label: `recorded path · ${truncateLabel(graph.nodes[targetIndex].label)}`
		};
	});
	const join = cause.sharedEntities.length ? cause.sharedEntities.join(', ') : 'recorded entity';
	const rank = cause.similarityRank === null ? 'rank unavailable' : `embedding rank #${cause.similarityRank}`;
	const spineBeats: PathStepMeta[] = [
		{ sourceIndex: failureIndex, targetIndex: failureIndex, beatFrame: DETONATE_FRAME, kind: 1, beatKind: 'receipt-failure', nodeId: evidence.failureId, label: `recorded failure · ${failureLabel}` },
		{ sourceIndex: failureIndex, targetIndex: failureIndex, beatFrame: WAVE_START, kind: 1, beatKind: 'receipt-join', nodeId: 'receipt-join', label: `shared entity · ${join}` },
		{ sourceIndex: causeIndex, targetIndex: failureIndex, beatFrame: WAVE_START + (causeDepth - 1) * hopSlot, kind: PATH_KIND.backwardCause, beatKind: 'receipt-candidate', nodeId: cause.memoryId, label: `candidate · ${causeLabel}` },
		{ sourceIndex: causeIndex, targetIndex: causeIndex, beatFrame: VERDICT_START, kind: 1, beatKind: 'receipt-verdict', nodeId: 'receipt-verdict', label: 'candidate cause found' }
	];
	return {
		viable: true,
		failureIndex,
		causeIndex,
		lookalikeIndices: [],
		hopDepths,
		causeDepth,
		hopSlot,
		waveData,
		pathData,
		pathMetas,
		spineBeats,
		verdict: {
			headline: 'candidate cause found',
			causeLabel,
			failureLabel,
			causeDate,
			hops: causeDepth,
			k: 0,
			receipt: `${cause.ageDays.toFixed(1)}d back · ${join} · ${rank}`
		},
		consts: { hopSlot, causeDepth }
	};
}

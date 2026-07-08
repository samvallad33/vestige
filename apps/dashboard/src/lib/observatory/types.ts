/**
 * Cognitive Observatory — shared CPU-side types + GPU buffer layout constants.
 *
 * The Observatory is a full-bleed WebGPU "living memory field". Node/particle
 * state lives in GPU storage buffers (boids ping-pong pattern, spec §1); this
 * module defines the exact byte layout so the TS uploader and the WGSL shaders
 * agree lane-for-lane.
 *
 * Visual DNA (spec §7): a node's BASE hue = its FSRS state color (meaning),
 * its ACTIVATION glow rides the thin-film spectral band (transcendence).
 */

import type { GraphNode, GraphEdge } from '$types';

/** Demo modes reachable as deterministic loops: ?demo=<mode>&loop=1 */
export type DemoMode =
	| 'recall-path'
	| 'engram-birth'
	| 'salience-rescue'
	| 'forgetting-horizon'
	| 'firewall';

export const DEMO_MODES: readonly DemoMode[] = [
	'recall-path',
	'engram-birth',
	'salience-rescue',
	'forgetting-horizon',
	'firewall'
] as const;

export function isDemoMode(v: string): v is DemoMode {
	return (DEMO_MODES as readonly string[]).includes(v);
}

// ---------------------------------------------------------------------------
// GPU buffer layout — NodeState
// ---------------------------------------------------------------------------
// Each node occupies 4 × vec4<f32> = 64 bytes, 16-byte aligned lanes so the
// WGSL struct maps 1:1 with no padding surprises.
//
//   lane 0  pos_radius     : xyz world position + w visual radius
//   lane 1  vel_retention  : xyz velocity        + w FSRS retention (0..1)
//   lane 2  color_flags    : rgb base color      + w packed flags (as f32)
//   lane 3  demo           : x recall intensity, y birth phase,
//                            z ripple phase,      w shock phase
//
// FLOATS_PER_NODE is what the uploader writes; keep it in lockstep with the
// WGSL `struct NodeState`.
export const FLOATS_PER_NODE = 16;
export const BYTES_PER_NODE = FLOATS_PER_NODE * 4; // 64

// Lane offsets (in floats) for the uploader.
export const NODE_LANE = {
	posRadius: 0, // +0..+3
	velRetention: 4, // +4..+7
	colorFlags: 8, // +8..+11
	demo: 12 // +12..+15
} as const;

// Packed visual flags (stored in color_flags.w as an f32 bit field via bitcast
// on the GPU; on the CPU side we assemble the integer then Math.fround it).
export const NODE_FLAG = {
	isCenter: 1 << 0,
	suppressed: 1 << 1,
	isAha: 1 << 2,
	isFailure: 1 << 3,
	isConfusion: 1 << 4
} as const;

// ---------------------------------------------------------------------------
// GPU buffer layout — EdgeIndex (static) and PathStep (demo path)
// ---------------------------------------------------------------------------
/** array<vec2<u32>> source/target node indices. 2 u32 per edge. */
export const UINTS_PER_EDGE = 2;

/**
 * array<vec4<u32>> per path beat:
 *   x source node index, y target node index, z beat frame, w kind
 * kind: 0 normal recall hop, 1 backward-cause hop (salience rescue),
 *       2 probe beam (salience rescue: vector search failing on camera).
 */
export const UINTS_PER_PATHSTEP = 4;

export const PATH_KIND = {
	recall: 0,
	backwardCause: 1,
	probe: 2
} as const;

// ---------------------------------------------------------------------------
// Per-frame uniforms — must match the WGSL `struct Params` exactly.
// ---------------------------------------------------------------------------
// Laid out as a flat Float32Array; sizes chosen so the whole block is a
// multiple of 16 bytes (WebGPU uniform alignment requirement).
//
//   [0]  frame            (loop frame, wraps at loopFrames)
//   [1]  loopPhase        (0..1)
//   [2]  nodeCount
//   [3]  edgeCount
//   [4]  pathCount
//   [5]  pulse            (0.5 + 0.5*sin — global breath, spec §7.2)
//   [6]  viewportW
//   [7]  viewportH
//   [8]  brightness       (from graphState)
//   [9]  demoId           (DemoMode index)
//   [10] time             (fixed sim seconds = frame / fps; NOT wall clock)
//   [11] _pad
export const PARAMS_FLOATS = 12;
export const PARAMS_BYTES = PARAMS_FLOATS * 4; // 48 (multiple of 16)

export function demoModeId(mode: DemoMode): number {
	const i = DEMO_MODES.indexOf(mode);
	return i < 0 ? 0 : i;
}

// ---------------------------------------------------------------------------
// CPU-side observatory graph (stable-indexed view of the API GraphResponse).
// ---------------------------------------------------------------------------
export interface ObservatoryNode {
	id: string;
	index: number; // stable position in the NodeState buffer
	label: string;
	type: string;
	retention: number;
	tags: string[];
	isCenter: boolean;
	suppressed: boolean;
}

export interface ObservatoryEdge {
	sourceIndex: number;
	targetIndex: number;
	weight: number;
	type: string;
}

export interface ObservatoryGraph {
	nodes: ObservatoryNode[];
	edges: ObservatoryEdge[];
	/** id -> stable buffer index */
	indexById: Map<string, number>;
	centerIndex: number;
}

/** Narrow the loose API GraphNode into what the Observatory needs. */
export function toObservatoryNode(n: GraphNode, index: number): ObservatoryNode {
	return {
		id: n.id,
		index,
		label: n.label,
		type: n.type,
		retention: typeof n.retention === 'number' ? n.retention : 0,
		tags: Array.isArray(n.tags) ? n.tags : [],
		isCenter: !!n.isCenter,
		suppressed: (n.suppression_count ?? 0) > 0
	};
}

export type { GraphNode, GraphEdge };

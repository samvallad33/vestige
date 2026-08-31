/**
 * Cognitive Observatory — graph → GPU upload preparation.
 *
 * Converts the API GraphResponse into stable-indexed typed arrays matching the
 * NodeState buffer layout (types.ts). Pure functions, fully unit-testable.
 *
 * Determinism contract (spec §6): node ordering is stable (sorted by id, center
 * first), positions come from deterministicSpherePosition seeded by the
 * DemoClock PRNG — same seed → identical layout, different seed → different.
 *
 * Visual DNA §7.1 — meaning layer: base color = FSRS state color from the real
 * dashboard palette (lib/graph/nodes.ts, used verbatim). Aha/failure/confusion
 * tags override, exactly like the Graph3D 'ahagraph' mode.
 */

import type { GraphResponse } from '$types';
import { getMemoryState, getAhaGraphColor, MEMORY_STATE_COLORS } from '$lib/memory-state';
import {
	FLOATS_PER_NODE,
	NODE_LANE,
	NODE_FLAG,
	UINTS_PER_EDGE,
	toObservatoryNode,
	type ObservatoryGraph,
	type ObservatoryNode,
	type ObservatoryEdge
} from './types';
import { deterministicSpherePosition } from './demo-clock';

/** Parse '#rrggbb' to 0..1 rgb. Falls back to slate on malformed input. */
export function hexToRgb01(hex: string): [number, number, number] {
	const m = /^#?([0-9a-fA-F]{6})$/.exec(hex.trim());
	if (!m) return [0x6b / 255, 0x72 / 255, 0x80 / 255]; // unavailable slate
	const v = parseInt(m[1], 16);
	return [((v >> 16) & 0xff) / 255, ((v >> 8) & 0xff) / 255, (v & 0xff) / 255];
}

/**
 * Meaning-layer base color for a node (visual DNA §7.1).
 * Tag kinds override via the REAL Graph3D mapping (getAhaGraphColor — case-
 * insensitive, aha gold → confusion red → failure/guardrail grey), else the
 * FSRS state palette. Reused verbatim so Observatory and Graph3D never drift.
 */
export function nodeBaseColor(node: ObservatoryNode): [number, number, number] {
	const tagColor = getAhaGraphColor({ tags: node.tags });
	if (tagColor) return hexToRgb01(tagColor);
	return hexToRgb01(MEMORY_STATE_COLORS[getMemoryState(node.retention)]);
}

/**
 * Build the stable-indexed observatory graph.
 * Ordering: center node first (index 0), then remaining nodes sorted by id —
 * deterministic regardless of API response order.
 */
export function buildObservatoryGraph(response: GraphResponse): ObservatoryGraph {
	const sorted = [...response.nodes].sort((a, b) => {
		if (a.isCenter !== b.isCenter) return a.isCenter ? -1 : 1;
		return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
	});

	const nodes: ObservatoryNode[] = sorted.map((n, i) => toObservatoryNode(n, i));
	const indexById = new Map<string, number>();
	for (const n of nodes) indexById.set(n.id, n.index);

	const edges: ObservatoryEdge[] = [];
	for (const e of response.edges) {
		const s = indexById.get(e.source);
		const t = indexById.get(e.target);
		if (s === undefined || t === undefined || s === t) continue;
		edges.push({ sourceIndex: s, targetIndex: t, weight: e.weight, type: e.type });
	}

	const centerIndex = nodes.findIndex((n) => n.isCenter);
	return { nodes, edges, indexById, centerIndex: centerIndex < 0 ? 0 : centerIndex };
}

export interface NodeStateBuild {
	/** FLOATS_PER_NODE floats per node, ready for the storage buffer. */
	data: Float32Array<ArrayBuffer>;
	nodeCount: number;
}

/**
 * Fill the NodeState array. `rng` must come from a fresh DemoClock seeded with
 * the demo seed so the layout is reproducible.
 */
export function buildNodeStateArray(
	graph: ObservatoryGraph,
	rng: () => number,
	fieldRadius = 120
): NodeStateBuild {
	const n = graph.nodes.length;
	const data = new Float32Array(n * FLOATS_PER_NODE);

	for (let i = 0; i < n; i++) {
		const node = graph.nodes[i];
		const base = i * FLOATS_PER_NODE;

		// lane 0: position + visual radius
		const [x, y, z] =
			node.isCenter && graph.centerIndex === i
				? [0, 0, 0] // the center memory anchors the field
				: deterministicSpherePosition(i, n, fieldRadius, rng);
		// radius: retention-weighted, center node reads largest (meaning layer)
		const radius = node.isCenter ? 4.2 : 1.4 + node.retention * 1.8;
		data[base + NODE_LANE.posRadius + 0] = x;
		data[base + NODE_LANE.posRadius + 1] = y;
		data[base + NODE_LANE.posRadius + 2] = z;
		data[base + NODE_LANE.posRadius + 3] = radius;

		// lane 1: velocity (rest) + retention
		data[base + NODE_LANE.velRetention + 3] = node.retention;

		// lane 2: base color + packed flags
		const [r, g, b] = nodeBaseColor(node);
		let flags = 0;
		if (node.isCenter) flags |= NODE_FLAG.isCenter;
		if (node.suppressed) flags |= NODE_FLAG.suppressed;
		// case-insensitive, mirroring getAhaGraphColor's tag semantics
		const tags = new Set(node.tags.map((t) => t.toLowerCase()));
		if (tags.has('aha')) flags |= NODE_FLAG.isAha;
		if (tags.has('failure') || tags.has('guardrail')) flags |= NODE_FLAG.isFailure;
		if (tags.has('confusion') || tags.has('weak-spot')) flags |= NODE_FLAG.isConfusion;
		data[base + NODE_LANE.colorFlags + 0] = r;
		data[base + NODE_LANE.colorFlags + 1] = g;
		data[base + NODE_LANE.colorFlags + 2] = b;
		data[base + NODE_LANE.colorFlags + 3] = flags;

		// lane 3: demo activations start at zero
	}

	return { data, nodeCount: n };
}

/** array<vec2<u32>> edge index buffer. */
export function buildEdgeIndexArray(graph: ObservatoryGraph): Uint32Array<ArrayBuffer> {
	const data = new Uint32Array(Math.max(1, graph.edges.length) * UINTS_PER_EDGE);
	graph.edges.forEach((e, i) => {
		data[i * UINTS_PER_EDGE] = e.sourceIndex;
		data[i * UINTS_PER_EDGE + 1] = e.targetIndex;
	});
	return data;
}

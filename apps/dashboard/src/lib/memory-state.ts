/**
 * Memory accessibility states + learning-tag colors - the LIVE module.
 *
 * Extracted from lib/graph/nodes.ts (the dead Three.js layer) so the
 * Observatory field, the state legend, and future state chips depend on
 * a small pure module instead of a 700-line renderer. The silent-state
 * violet here is DOM-legend legacy; the WebGPU field uses the cortex
 * palette. Phase 3 reconciles both to the doctrine ramp.
 */

import type { GraphNode } from '$types';
import { NODE_TYPE_COLORS } from '$types';

// ============================================================================
// v2.0.8: Memory state coloring (FSRS accessibility bucket)
// ============================================================================
//
// Every knowledge_node has an FSRS accessibility score computed from
// (retention × 0.5 + retrieval × 0.3 + storage × 0.2). That score gates which
// memories surface in search and drives the Active / Dormant / Silent /
// Unavailable lifecycle documented by Bjork & Bjork 1992 dual-strength model.
//
// The backend computes all three channels, but `GraphNode` only carries
// `retention` — which is already the dominant weight (0.5 of 1.0). Using
// retention alone as a proxy is a known approximation; the buckets line up
// with the same thresholds `execute_system_status` uses server-side, so the
// visual labelling matches what `/api/stats` reports in its
// `stateDistribution` block.

export type MemoryState = 'active' | 'dormant' | 'silent' | 'unavailable';

/// Map an FSRS retention score to its accessibility bucket.
///
/// Thresholds match `execute_system_status` at the backend so the 3D graph's
/// colours line up with the numbers reported by `/api/stats`.
export function getMemoryState(retention: number): MemoryState {
	if (retention >= 0.7) return 'active';
	if (retention >= 0.4) return 'dormant';
	if (retention >= 0.1) return 'silent';
	return 'unavailable';
}

/// FSRS state palette. Distinct from NODE_TYPE_COLORS so the two modes can
/// coexist in the UI without overloading a single colour channel.
export const MEMORY_STATE_COLORS: Record<MemoryState, string> = {
	active: '#10b981', // emerald — easily retrievable
	dormant: '#f59e0b', // amber — retrievable with effort
	silent: '#4A7D8C', // fossil cyan — difficult, needs cues (never purple)
	unavailable: '#6b7280', // slate — needs reinforcement
};

export const MEMORY_STATE_DESCRIPTIONS: Record<MemoryState, string> = {
	active: 'Easily retrievable (retention ≥ 70%)',
	dormant: 'Retrievable with effort (40–70%)',
	silent: 'Difficult, needs cues (10–40%)',
	unavailable: 'Needs reinforcement (< 10%)',
};

export type AhaGraphKind = 'aha' | 'confusion' | 'failure';

export const AHAGRAPH_COLORS: Record<AhaGraphKind, string> = {
	aha: '#FFD700',
	confusion: '#EF4444',
	failure: '#9CA3AF',
};

export const AHAGRAPH_DESCRIPTIONS: Record<AhaGraphKind, string> = {
	aha: 'Aha moments and breakthroughs',
	confusion: 'Confusions and weak spots',
	failure: 'Failures and guardrails',
};

/// Color mode controls whether node spheres are tinted by node type,
/// FSRS memory state, or AhaGraph learning tags.
/// Type mode is the long-standing default; state mode is the v2.0.8 addition.
export type ColorMode = 'type' | 'state' | 'ahagraph';

/// Pick a hex colour for a node given the active colour mode.
/// Falls back to the grey `unavailable` tone if the node's type is unknown.
export function getNodeColor(node: GraphNode, mode: ColorMode): string {
	if (mode === 'state') {
		return MEMORY_STATE_COLORS[getMemoryState(node.retention)];
	}
	if (mode === 'ahagraph') {
		return getAhaGraphColor(node) ?? NODE_TYPE_COLORS[node.type] ?? '#8B95A5';
	}
	return NODE_TYPE_COLORS[node.type] || '#8B95A5';
}

export function getAhaGraphColor(node: Pick<GraphNode, 'tags'>): string | null {
	const tags = new Set((node.tags ?? []).map((tag) => tag.toLowerCase()));
	if (tags.has('aha')) return AHAGRAPH_COLORS.aha;
	if (tags.has('confusion') || tags.has('weak-spot')) return AHAGRAPH_COLORS.confusion;
	if (tags.has('failure') || tags.has('guardrail')) return AHAGRAPH_COLORS.failure;
	return null;
}

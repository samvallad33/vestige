// ─────────────────────────────────────────────────────────────────────────────
// Reasoning trace — SHARED CPU layout.
//
// Both the MSDF text pass (labels/evidence/receipt) and the WGSL geometry pass
// (beam / ribbons / nucleus / interference fringe) compute their positions from
// THIS one deterministic layout so they can never disagree. Everything is in
// logical NDC (x,y ∈ [-1,1], +Y up); no Math.random / Date.now — capture-stable.
// ─────────────────────────────────────────────────────────────────────────────

import type { ReasoningScene, NormalizedEvidence } from './reasoning-scene';

export type EvidenceRole = NormalizedEvidence['role'];

// 8 gate lenses along a causal beam; spacing ACCELERATES toward the decision.
export const GATE_X = [-0.86, -0.526, -0.214, 0.074, 0.335, 0.562, 0.747, 0.86];
export const GATE_SHORT = [
	'INTENT',
	'RETRIEVE',
	'ACTIVATE',
	'EVIDENCE',
	'CHALLENGE',
	'SYNTH',
	'DECIDE',
	'SEAL'
];
export const BEAM_Y = 0.0;
export const EVIDENCE_GATE_INDEX = 3; // where evidence enters the trace
export const DECIDE_GATE_INDEX = 6; // recommendation nucleus
export const RECEIPT_GATE_INDEX = 7;

export const LANE_Y: Record<EvidenceRole, number> = {
	primary: 0.04,
	supporting: 0.24,
	contradicting: -0.22,
	superseded: -0.48
};
export const LANE_STEP = 0.052;
export const LANE_MAX = 6;

export interface EvidenceNode {
	id: string;
	role: EvidenceRole;
	trust: number;
	preview: string;
	x: number; // NDC — where this evidence sits (in its lane, right of evidence gate)
	y: number;
}

export interface GateNode {
	index: number;
	kind: string;
	label: string;
	short: string;
	x: number;
	lit: boolean;
	confidence: number;
	count: number;
}

export interface RibbonEdge {
	fromX: number; // evidence
	fromY: number;
	toX: number; // decide nucleus (or its gate)
	toY: number;
	trust: number;
	role: EvidenceRole;
	/** signed influence: +1 supports the decision, -1 opposes it. */
	sign: number;
}

export interface TraceLayout {
	query: string;
	gates: GateNode[];
	evidence: EvidenceNode[];
	ribbons: RibbonEdge[];
	nucleus: { x: number; y: number; confidence: number } | null;
	/** contradiction pairs → interference-fringe endpoints in NDC. */
	fringes: { ax: number; ay: number; bx: number; by: number; strength: number }[];
	/** supersession → scar (old) + amber filament to the replacement. */
	scars: { x: number; y: number; toX: number; toY: number }[];
	receiptX: number;
	sourceCount: number;
}

const EVIDENCE_ANCHOR_X = GATE_X[EVIDENCE_GATE_INDEX] - 0.02;
const EVIDENCE_FAN = 0.14; // how far evidence fans right of its gate by row

/** Deterministic left-to-right layout of one deep_reference response. */
export function computeTraceLayout(scene: ReasoningScene | null): TraceLayout | null {
	if (!scene || !scene.alive) return null;

	const query = String(scene.raw?.query ?? '');

	const gates: GateNode[] = (scene.stages ?? []).slice(0, GATE_X.length).map((g, i) => ({
		index: i,
		kind: g?.kind ?? String(i),
		label: g?.label ?? g?.kind ?? '',
		short: GATE_SHORT[i] ?? (g?.kind ?? '').toUpperCase(),
		x: GATE_X[i],
		lit: !!g?.lit,
		confidence: g?.confidence ?? 0,
		count: g?.count ?? 0
	}));

	// Evidence into role lanes, centered, capped.
	const byRole = new Map<EvidenceRole, NormalizedEvidence[]>();
	for (const e of scene.evidence ?? []) {
		const arr = byRole.get(e.role) ?? [];
		arr.push(e);
		byRole.set(e.role, arr);
	}
	const evidence: EvidenceNode[] = [];
	for (const [role, list] of byRole) {
		const laneY = LANE_Y[role];
		const shown = list.slice(0, LANE_MAX);
		shown.forEach((e, i) => {
			const y = laneY + (i - (shown.length - 1) / 2) * LANE_STEP;
			// fan slightly right with row so ribbons don't perfectly overlap
			const x = EVIDENCE_ANCHOR_X + (i / Math.max(1, shown.length - 1)) * EVIDENCE_FAN;
			evidence.push({ id: e.id, role, trust: e.trust, preview: e.preview, x, y });
		});
	}

	const nucleus =
		scene.recommended != null
			? {
					x: GATE_X[DECIDE_GATE_INDEX],
					y: BEAM_Y,
					confidence: Math.max(0, Math.min(1, scene.recommended.trust_score))
				}
			: null;

	// Ribbons: every evidence node flows toward the nucleus. supporting/primary
	// are +sign (feed the decision), contradicting is −sign (oppose), superseded
	// is a faded +sign that a scar will later cross.
	const ribbons: RibbonEdge[] = nucleus
		? evidence.map((e) => ({
				fromX: e.x,
				fromY: e.y,
				toX: nucleus.x,
				toY: nucleus.y,
				trust: e.trust,
				role: e.role,
				sign: e.role === 'contradicting' ? -1 : 1
			}))
		: [];

	// Contradiction fringes: place the two conflicting evidence nodes as the
	// interference sources (fall back to the contradicting lane if ids missing).
	const evById = new Map(evidence.map((e) => [e.id, e]));
	const fringes = (scene.contradictions ?? [])
		.map((c) => {
			const a = evById.get(c.stronger.id);
			const b = evById.get(c.weaker.id);
			if (!a || !b) return null;
			return { ax: a.x, ay: a.y, bx: b.x, by: b.y, strength: Math.max(0.35, c.topic_overlap || 0.5) };
		})
		.filter((f): f is NonNullable<typeof f> => f !== null);

	const scars = (scene.superseded ?? [])
		.map((sup) => {
			const old = evById.get(sup.id);
			if (!old) return null;
			return { x: old.x, y: old.y, toX: nucleus?.x ?? GATE_X[DECIDE_GATE_INDEX], toY: nucleus?.y ?? BEAM_Y };
		})
		.filter((s): s is NonNullable<typeof s> => s !== null);

	return {
		query,
		gates,
		evidence,
		ribbons,
		nucleus,
		fringes,
		scars,
		receiptX: GATE_X[RECEIPT_GATE_INDEX],
		sourceCount: scene.evidence?.length ?? 0
	};
}

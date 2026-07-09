import {
	assertProvenance,
	type Provenance,
	type RouteEvent,
	type RouteNode,
	type RouteReceipt,
	type RouteSceneModel
} from '$lib/observatory/route-scene';

export type ReasoningStageKind =
	| 'intent'
	| 'retrieve'
	| 'activate'
	| 'evidence'
	| 'contradiction'
	| 'synthesis'
	| 'recommendation'
	| 'receipt';

export interface ReasoningStageReceipt {
	index: number;
	kind: ReasoningStageKind;
	label: string;
	count: number;
	confidence: number;
	lit: boolean;
	provenance: Provenance;
	exposed: Record<string, unknown>;
	not_exposed_by_backend: string[];
	interrupt: 'none' | 'contradiction' | 'supersession';
}

export interface NormalizedEvidence {
	id: string;
	trust: number;
	date: string;
	role: 'primary' | 'supporting' | 'contradicting' | 'superseded';
	preview: string;
	nodeType?: string;
}

export interface NormalizedContradiction {
	stronger: NormalizedEvidence;
	weaker: NormalizedEvidence;
	topic_overlap: number;
	summary: string;
}

export interface NormalizedSupersession {
	id: string;
	preview: string;
	trust: number;
	date: string;
	superseded_by: string;
	reason: string;
}

export interface NormalizedRecommended {
	answer_preview: string;
	memory_id: string;
	trust_score: number;
	date: string;
}

export interface ReasoningScene extends RouteSceneModel {
	organ: 'reasoning';
	stages: ReasoningStageReceipt[];
	evidence: NormalizedEvidence[];
	contradictions: NormalizedContradiction[];
	superseded: NormalizedSupersession[];
	recommended: NormalizedRecommended | null;
	raw: Record<string, unknown>;
}

const STAGE_LABELS: Record<ReasoningStageKind, string> = {
	intent: 'Intent classification',
	retrieve: 'Memory retrieval',
	activate: 'Activation expansion',
	evidence: 'Evidence grounding',
	contradiction: 'Contradiction check',
	synthesis: 'Synthesis',
	recommendation: 'Recommendation',
	receipt: 'Composition receipt'
};

const STAGE_KINDS: ReasoningStageKind[] = [
	'intent',
	'retrieve',
	'activate',
	'evidence',
	'contradiction',
	'synthesis',
	'recommendation',
	'receipt'
];

function record(v: unknown): Record<string, unknown> {
	return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function text(v: unknown, fallback = ''): string {
	return typeof v === 'string' ? v : v == null ? fallback : String(v);
}

function num(v: unknown, fallback = 0): number {
	return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function clamp01(v: number): number {
	return Math.max(0, Math.min(1, v));
}

function confidence01(v: unknown): number {
	const n = num(v, 0);
	return clamp01(n > 1 ? n / 100 : n);
}

function role(v: unknown): NormalizedEvidence['role'] {
	return v === 'primary' || v === 'supporting' || v === 'contradicting' || v === 'superseded'
		? v
		: 'supporting';
}

function evidenceFromRecord(e: Record<string, unknown>, fallbackId = ''): NormalizedEvidence {
	const trust = confidence01(e.trust ?? e.trust_score ?? 0);
	return {
		id: text(e.id ?? e.memory_id ?? fallbackId),
		trust,
		date: text(e.date ?? e.created_at ?? ''),
		role: role(e.role),
		preview: text(e.preview ?? e.answer_preview ?? e.content ?? ''),
		nodeType: e.node_type ? text(e.node_type) : e.nodeType ? text(e.nodeType) : undefined
	};
}

function source(kind: Provenance['kind'], id: string, scalar?: Provenance['scalar']): Provenance {
	return scalar ? { kind, id, scalar } : { kind, id: id || `${kind}:unknown` };
}

function scalarSource(name: string, value: number): Provenance {
	return { kind: 'scalar', id: `deep_reference.${name}`, scalar: { name, value } };
}

export function normalizeDeepReferenceResponse(rawInput: Record<string, unknown>): ReasoningScene {
	const raw = record(rawInput);
	const evidenceRaw = Array.isArray(raw.evidence) ? raw.evidence.map(record) : [];
	const evidence = evidenceRaw
		.map((e) => evidenceFromRecord(e))
		.filter((e) => e.id.length > 0);

	const rec = record(raw.recommended);
	const recommended: NormalizedRecommended | null =
		Object.keys(rec).length > 0 || evidence.length > 0
			? {
					answer_preview: text(rec.answer_preview ?? evidence[0]?.preview ?? ''),
					memory_id: text(rec.memory_id ?? evidence[0]?.id ?? ''),
					trust_score: confidence01(rec.trust_score ?? evidence[0]?.trust ?? 0),
					date: text(rec.date ?? evidence[0]?.date ?? '')
				}
			: null;

	const contradictionsRaw = Array.isArray(raw.contradictions)
		? raw.contradictions.map(record)
		: Array.isArray(raw.claim_conflicts)
			? raw.claim_conflicts.map(record)
			: [];
	const contradictions: NormalizedContradiction[] = contradictionsRaw
		.map((c) => {
			// T2 current shape first: { stronger:{id,...}, weaker:{id,...}, topic_overlap }
			const strongerRaw = record(c.stronger);
			const weakerRaw = record(c.weaker);
			if (Object.keys(strongerRaw).length > 0 || Object.keys(weakerRaw).length > 0) {
				const stronger = evidenceFromRecord(strongerRaw);
				const weaker = evidenceFromRecord(weakerRaw);
				return {
					stronger,
					weaker,
					topic_overlap: clamp01(num(c.topic_overlap, 0)),
					summary: text(
						c.summary,
						`Trust-weighted conflict: ${stronger.id.slice(0, 8)} over ${weaker.id.slice(0, 8)}`
					)
				};
			}
			// Legacy fallback: { a_id, b_id, summary }
			const a = evidenceFromRecord({ id: c.a_id, role: 'contradicting' });
			const b = evidenceFromRecord({ id: c.b_id, role: 'contradicting' });
			return {
				stronger: a,
				weaker: b,
				topic_overlap: clamp01(num(c.topic_overlap, 0)),
				summary: text(c.summary ?? c.reason, 'Trust-weighted conflict between high-FSRS memories.')
			};
		})
		.filter((c) => c.stronger.id || c.weaker.id);

	const supersededRaw = Array.isArray(raw.superseded) ? raw.superseded.map(record) : [];
	const superseded: NormalizedSupersession[] = supersededRaw
		.map((s) => {
			// T2 current shape first: { id, preview, trust, date, superseded_by }
			if (s.id || s.superseded_by) {
				return {
					id: text(s.id),
					preview: text(s.preview ?? ''),
					trust: confidence01(s.trust ?? 0),
					date: text(s.date ?? ''),
					superseded_by: text(s.superseded_by ?? recommended?.memory_id ?? ''),
					reason: text(s.reason ?? 'Superseded by newer memory with higher trust.')
				};
			}
			// Legacy fallback: { old_id, new_id, reason }
			return {
				id: text(s.old_id),
				preview: text(s.preview ?? ''),
				trust: confidence01(s.trust ?? 0),
				date: text(s.date ?? ''),
				superseded_by: text(s.new_id ?? recommended?.memory_id ?? ''),
				reason: text(s.reason ?? 'Superseded by newer memory with higher trust.')
			};
		})
		.filter((s) => s.id || s.superseded_by);

	const confidence = confidence01(raw.confidence);
	const memoriesAnalyzed = num(raw.memoriesAnalyzed ?? raw.memories_analyzed, evidence.length);
	const activationExpanded = num(raw.activationExpanded ?? raw.activation_expanded, 0);
	const intent = text(raw.intent ?? '');
	const reasoning = text(raw.reasoning ?? raw.guidance ?? '');
	const guidance = text(raw.guidance ?? '');
	const compositionEventId = text(raw.composition_event_id ?? '');
	const compositionWriteStatus = text(raw.compositionWriteStatus ?? raw.composition_write_status ?? '');
	const receiptValue = compositionEventId || compositionWriteStatus;
	const receiptIsPersisted = compositionEventId.length > 0;

	const nodes: RouteNode[] = evidence.map((e, index) => ({
		source: source('memory', e.id),
		index,
		label: e.preview || e.id.slice(0, 8),
		retention: clamp01(e.trust),
		trust: clamp01(e.trust),
		lastAccessed: e.date || undefined,
		tags: [e.role, ...(e.nodeType ? [e.nodeType] : [])],
		type: e.nodeType ?? 'memory'
	}));

	const indexById = new Map(nodes.map((n) => [n.source.id, n.index]));
	const edges = contradictions.flatMap((c, i) => {
		const a = indexById.get(c.stronger.id);
		const b = indexById.get(c.weaker.id);
		if (a == null || b == null) return [];
		return [
			{
				source: source('pair', `contradiction:${c.stronger.id}:${c.weaker.id}`),
				sourceIndex: a,
				targetIndex: b,
				weight: Math.max(0.2, c.topic_overlap || 0.5),
				kind: 'contradiction'
			}
		];
	});

	const events: RouteEvent[] = [];
	if (contradictions.length > 0) {
		events.push({
			source: source('event', `deep_reference.contradictions.${contradictions.length}`),
			type: 'ReasoningContradictionInterrupt',
			targetIndex: -1,
			frame: 250,
			energy: contradictions.length
		});
	}
	if (superseded.length > 0) {
		events.push({
			source: source('event', `deep_reference.superseded.${superseded.length}`),
			type: 'ReasoningSupersessionInterrupt',
			targetIndex: -1,
			frame: 330,
			energy: superseded.length
		});
	}

	const receipts: RouteReceipt[] = [];
	if (receiptValue) {
		receipts.push({
			source: receiptIsPersisted
				? source('receipt', compositionEventId)
				: scalarSource('compositionWriteStatus', compositionWriteStatus === 'persisted' ? 1 : 0),
			label: receiptIsPersisted ? `receipt ${compositionEventId.slice(0, 8)}` : compositionWriteStatus,
			nodeIndices: nodes.map((n) => n.index)
		});
	}

	const mkStage = (
		index: number,
		kind: ReasoningStageKind,
		count: number,
		stageConfidence: number,
		exposed: Record<string, unknown>,
		provenance: Provenance,
		missing: string[],
		interrupt: ReasoningStageReceipt['interrupt'] = 'none'
	): ReasoningStageReceipt => ({
		index,
		kind,
		label: STAGE_LABELS[kind],
		count,
		confidence: clamp01(stageConfidence),
		lit: count > 0 || stageConfidence > 0,
		provenance,
		exposed,
		not_exposed_by_backend: missing,
		interrupt
	});

	const stages: ReasoningStageReceipt[] = STAGE_KINDS.map((kind, index) => {
		switch (kind) {
			case 'intent':
				return mkStage(index, kind, intent ? 1 : 0, intent ? confidence : 0, { intent }, scalarSource('intent_present', intent ? 1 : 0), ['raw classifier trace']);
			case 'retrieve':
				return mkStage(index, kind, memoriesAnalyzed || evidence.length, confidence, { memoriesAnalyzed, evidence_count: evidence.length }, scalarSource('memoriesAnalyzed', memoriesAnalyzed), ['candidate ids discarded before evidence']);
			case 'activate':
				return mkStage(index, kind, activationExpanded, activationExpanded > 0 ? confidence : 0, { activationExpanded }, scalarSource('activationExpanded', activationExpanded), ['activation path/map']);
			case 'evidence':
				return mkStage(index, kind, evidence.length, evidence.length > 0 ? confidence : 0, { evidence }, scalarSource('evidence_count', evidence.length), ['reranker discarded candidates']);
			case 'contradiction':
				return mkStage(index, kind, contradictions.length, contradictions.length > 0 ? confidence : 0, { contradictions, claim_conflicts: raw.claim_conflicts ?? [] }, scalarSource('contradiction_count', contradictions.length), ['full claim graph'], contradictions.length > 0 ? 'contradiction' : 'none');
			case 'synthesis':
				return mkStage(index, kind, reasoning || guidance ? 1 : 0, reasoning || guidance ? confidence : 0, { reasoning, guidance }, scalarSource('synthesis_present', reasoning || guidance ? 1 : 0), ['token-level chain internals']);
			case 'recommendation':
				return mkStage(index, kind, recommended?.memory_id ? 1 : 0, recommended?.trust_score ?? 0, { recommended }, recommended?.memory_id ? source('memory', recommended.memory_id) : scalarSource('recommended_present', 0), ['alternative recommendations']);
			case 'receipt':
				return mkStage(index, kind, receiptValue ? 1 : 0, receiptValue ? confidence : 0, { composition_event_id: compositionEventId, compositionWriteStatus }, receiptIsPersisted ? source('receipt', compositionEventId) : scalarSource('compositionWriteStatus', compositionWriteStatus === 'persisted' ? 1 : 0), ['separate receipt field'], superseded.length > 0 ? 'supersession' : 'none');
		}
	});

	const scene: ReasoningScene = {
		organ: 'reasoning',
		nodes,
		edges,
		events,
		receipts,
		scalars: {
			confidence,
			memoriesAnalyzed,
			activationExpanded,
			evidenceCount: evidence.length,
			contradictionCount: contradictions.length,
			supersededCount: superseded.length,
			compositionPersisted: receiptIsPersisted ? 1 : 0
		},
		alive: !!(
			intent ||
			reasoning ||
			guidance ||
			evidence.length ||
			contradictions.length ||
			superseded.length ||
			recommended?.memory_id ||
			receiptValue
		),
		stages,
		evidence,
		contradictions,
		superseded,
		recommended,
		raw
	};

	if (import.meta.env.DEV) assertProvenance(scene);
	return scene;
}

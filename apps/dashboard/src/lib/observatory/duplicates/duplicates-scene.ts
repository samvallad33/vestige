import { clusterKey, pickWinner } from '$components/duplicates-helpers';
import {
	assertProvenance,
	type Provenance,
	type RouteEdge,
	type RouteEvent,
	type RouteNode,
	type RouteSceneModel
} from '$lib/observatory/route-scene';
import type { DuplicateClusterGroup, DuplicateClusterMemory, DuplicatesResponse } from '$types';

export interface DuplicateFusionMemory extends DuplicateClusterMemory {
	index: number;
	preview: string;
	winner: boolean;
	mismatchTokens: string[];
}

export interface DuplicateFusionCluster {
	id: string;
	index: number;
	similarity: number;
	threshold: number;
	suggestedAction: 'merge' | 'review';
	winnerId: string;
	memories: DuplicateFusionMemory[];
	mismatchTokens: string[];
	source: Provenance;
}

export interface DuplicatesScene extends RouteSceneModel {
	organ: 'duplicates';
	threshold: number;
	total: number;
	clusters: DuplicateFusionCluster[];
	raw: DuplicatesResponse;
}

function clamp01(v: number): number {
	return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
}

function source(kind: Provenance['kind'], id: string, scalar?: Provenance['scalar']): Provenance {
	return scalar ? { kind, id, scalar } : { kind, id: id || `${kind}:unknown` };
}

function scalarSource(name: string, value: number): Provenance {
	return { kind: 'scalar', id: `duplicates.${name}`, scalar: { name, value } };
}

function preview(content: string, max = 84): string {
	const text = (content || '').trim().replace(/\s+/g, ' ');
	return text.length <= max ? text : `${text.slice(0, max)}…`;
}

function tokenize(content: string): string[] {
	return (content || '')
		.toLowerCase()
		.replace(/[^a-z0-9_\s-]/g, ' ')
		.split(/\s+/)
		.filter((token) => token.length >= 4)
		.slice(0, 80);
}

function mismatchTokens(memories: DuplicateClusterMemory[]): string[] {
	if (memories.length < 2) return [];
	const tokenSets = memories.map((m) => new Set(tokenize(m.content)));
	const counts = new Map<string, number>();
	for (const tokenSet of tokenSets) {
		for (const token of tokenSet) counts.set(token, (counts.get(token) ?? 0) + 1);
	}
	return Array.from(counts.entries())
		.filter(([, count]) => count > 0 && count < memories.length)
		.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
		.slice(0, 12)
		.map(([token]) => token);
}

function normalizeCluster(
	cluster: DuplicateClusterGroup,
	index: number,
	threshold: number
): DuplicateFusionCluster | null {
	const members = Array.isArray(cluster.memories) ? cluster.memories.filter((m) => m.id) : [];
	if (members.length < 2) return null;
	const id = clusterKey(members);
	const winner = pickWinner(members);
	const clusterMismatch = mismatchTokens(members);
	return {
		id,
		index,
		similarity: clamp01(cluster.similarity),
		threshold: clamp01(threshold),
		suggestedAction: cluster.suggestedAction === 'merge' ? 'merge' : 'review',
		winnerId: winner?.id ?? members[0].id,
		memories: members.map((memory, memoryIndex) => ({
			...memory,
			index: memoryIndex,
			preview: preview(memory.content),
			winner: memory.id === (winner?.id ?? members[0].id),
			mismatchTokens: clusterMismatch.filter((token) => tokenize(memory.content).includes(token)).slice(0, 8)
		})),
		mismatchTokens: clusterMismatch,
		source: source('pair', id)
	};
}

export function normalizeDuplicatesScene(rawInput: DuplicatesResponse): DuplicatesScene {
	const threshold = clamp01(rawInput.threshold ?? 0.8);
	const rawClusters = Array.isArray(rawInput.clusters) ? rawInput.clusters : [];
	const clusters = rawClusters
		.map((cluster, index) => normalizeCluster(cluster, index, threshold))
		.filter((cluster): cluster is DuplicateFusionCluster => cluster !== null);

	let nodeIndex = 0;
	const nodes: RouteNode[] = [];
	const nodeIndexByMemoryId = new Map<string, number>();
	for (const cluster of clusters) {
		for (const memory of cluster.memories) {
			if (nodeIndexByMemoryId.has(memory.id)) continue;
			const currentIndex = nodeIndex++;
			nodeIndexByMemoryId.set(memory.id, currentIndex);
			nodes.push({
				source: source('memory', memory.id),
				index: currentIndex,
				label: memory.preview || memory.id.slice(0, 8),
				retention: clamp01(memory.retention),
				trust: clamp01(cluster.similarity),
				lastAccessed: memory.createdAt,
				tags: [memory.nodeType, ...memory.tags, memory.winner ? 'winner' : 'candidate'].filter(Boolean),
				type: memory.nodeType || 'memory'
			});
		}
	}

	const edges: RouteEdge[] = [];
	for (const cluster of clusters) {
		const winnerIndex = nodeIndexByMemoryId.get(cluster.winnerId);
		if (winnerIndex == null) continue;
		for (const memory of cluster.memories) {
			const targetIndex = nodeIndexByMemoryId.get(memory.id);
			if (targetIndex == null || targetIndex === winnerIndex) continue;
			edges.push({
				source: source('pair', `${cluster.id}:${cluster.winnerId}:${memory.id}`),
				sourceIndex: winnerIndex,
				targetIndex,
				weight: Math.max(0.05, cluster.similarity),
				kind: cluster.suggestedAction === 'merge' ? 'fusion-candidate' : 'review-candidate'
			});
		}
	}

	const events: RouteEvent[] = clusters.map((cluster, index) => ({
		source: source('event', `duplicates.cluster.${cluster.id}`),
		type: cluster.suggestedAction === 'merge' ? 'DuplicateMergeCandidate' : 'DuplicateReviewCandidate',
		targetIndex: -1,
		frame: 20 + index * 14,
		energy: Math.max(0.1, cluster.similarity - threshold + 0.1)
	}));

	const total = Number.isFinite(rawInput.total) ? rawInput.total : clusters.length;
	const memoryCount = nodes.length;
	const maxSimilarity = clusters.reduce((max, cluster) => Math.max(max, cluster.similarity), 0);
	const mergeCandidates = clusters.filter((cluster) => cluster.suggestedAction === 'merge').length;
	const reviewCandidates = clusters.length - mergeCandidates;

	const scene: DuplicatesScene = {
		organ: 'duplicates',
		nodes,
		edges,
		events,
		receipts: [],
		scalars: {
			threshold: scalarSource('threshold', threshold).scalar?.value ?? threshold,
			clusterCount: clusters.length,
			memoryCount,
			maxSimilarity,
			mergeCandidates,
			reviewCandidates,
			total
		},
		alive: clusters.length > 0,
		threshold,
		total,
		clusters,
		raw: rawInput
	};

	if (import.meta.env.DEV) assertProvenance(scene);
	return scene;
}

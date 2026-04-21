<!--
  Memory Hygiene — Duplicate Detection
  Dashboard exposure of the `find_duplicates` MCP tool. Threshold slider
  (0.70-0.95) reruns cosine-similarity clustering. Each cluster renders as a
  DuplicateCluster with similarity bar, stacked memory cards, and merge /
  review / dismiss actions.
-->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import DuplicateCluster from '$components/DuplicateCluster.svelte';
	import { clusterKey, filterByThreshold } from '$components/duplicates-helpers';

	interface ClusterMemory {
		id: string;
		content: string;
		nodeType: string;
		tags: string[];
		retention: number;
		createdAt: string;
	}

	interface Cluster {
		similarity: number;
		memories: ClusterMemory[];
		suggestedAction: 'merge' | 'review';
	}

	interface DuplicatesResponse {
		clusters: Cluster[];
	}

	let threshold = $state(0.8);
	let clusters: Cluster[] = $state([]);
	// Dismissed clusters are tracked by stable identity (sorted member ids) so
	// dismissals survive a re-fetch. If the cluster membership changes, the key
	// changes and the cluster is treated as fresh.
	let dismissed = $state(new Set<string>());
	let loading = $state(true);
	let error: string | null = $state(null);
	let debounceTimer: ReturnType<typeof setTimeout> | undefined;

	// Mock realistic response. Swap for real fetch when backend ships.
	// TODO(backend-swap): replace `mockFetchDuplicates` with:
	//   const res = await fetch(`/api/duplicates?threshold=${t}`);
	//   return (await res.json()) as DuplicatesResponse;
	// The pure `filterByThreshold` helper in duplicates-helpers.ts mirrors the
	// server-side >= semantics so the UI behaves identically before and after.
	async function mockFetchDuplicates(t: number): Promise<DuplicatesResponse> {
		// Simulate latency so the skeleton is visible.
		await new Promise((r) => setTimeout(r, 450));

		const all: Cluster[] = [
			{
				similarity: 0.96,
				suggestedAction: 'merge',
				memories: [
					{
						id: 'm-001',
						content:
							'BUG FIX: Harmony parser dropped `final` channel tokens when tool call followed. Root cause: 5-layer fallback missed the final channel marker when channel switched mid-stream. Solution: added final-channel detector before tool-call pop. Files: src/parser/harmony.rs',
						nodeType: 'fact',
						tags: ['bug-fix', 'aimo3', 'parser'],
						retention: 0.91,
						createdAt: '2026-04-12T14:22:00Z',
					},
					{
						id: 'm-002',
						content:
							'Fixed Harmony parser final-channel bug — 5-layer fallback was missing the final channel marker when a tool call followed. Added detector before tool pop.',
						nodeType: 'fact',
						tags: ['bug-fix', 'aimo3'],
						retention: 0.64,
						createdAt: '2026-04-13T09:15:00Z',
					},
					{
						id: 'm-003',
						content:
							'Harmony parser: final channel dropped on tool-call. Patched the fallback stack.',
						nodeType: 'note',
						tags: ['parser'],
						retention: 0.38,
						createdAt: '2026-04-14T11:02:00Z',
					},
				],
			},
			{
				similarity: 0.88,
				suggestedAction: 'merge',
				memories: [
					{
						id: 'm-004',
						content:
							'DECISION: Use vLLM prefix caching at 0.35 gpu_memory_utilization for AIMO3 submissions. Alternatives considered: sglang (slower cold start), TensorRT-LLM (deployment friction).',
						nodeType: 'decision',
						tags: ['vllm', 'aimo3', 'inference'],
						retention: 0.84,
						createdAt: '2026-04-05T18:44:00Z',
					},
					{
						id: 'm-005',
						content:
							'Chose vLLM with prefix caching (0.35 mem util) over sglang and TensorRT-LLM for AIMO3 inference.',
						nodeType: 'decision',
						tags: ['vllm', 'aimo3'],
						retention: 0.72,
						createdAt: '2026-04-06T10:30:00Z',
					},
				],
			},
			{
				similarity: 0.83,
				suggestedAction: 'review',
				memories: [
					{
						id: 'm-006',
						content:
							'Sam prefers to ship one change per Kaggle submission — stacking changes destroyed signal at AIMO3 (30/50 regression from 12 stacked variables).',
						nodeType: 'pattern',
						tags: ['kaggle', 'methodology', 'aimo3'],
						retention: 0.88,
						createdAt: '2026-04-04T22:10:00Z',
					},
					{
						id: 'm-007',
						content:
							'One-variable-at-a-time rule: never stack multiple changes per submission. Paper 2603.27844 proves +/-2 points is noise.',
						nodeType: 'pattern',
						tags: ['kaggle', 'methodology'],
						retention: 0.67,
						createdAt: '2026-04-08T16:20:00Z',
					},
					{
						id: 'm-008',
						content: 'Lesson: stacking 12 changes at AIMO3 cost a submission. Always isolate variables.',
						nodeType: 'note',
						tags: ['methodology'],
						retention: 0.42,
						createdAt: '2026-04-15T08:55:00Z',
					},
				],
			},
			{
				similarity: 0.78,
				suggestedAction: 'review',
				memories: [
					{
						id: 'm-009',
						content:
							'Dimensional Illusion performance: 7-minute flow poi set, LED config Parthenos overcook preset, tempo 128 BPM.',
						nodeType: 'event',
						tags: ['dimensional-illusion', 'poi', 'performance'],
						retention: 0.76,
						createdAt: '2026-03-28T19:45:00Z',
					},
					{
						id: 'm-010',
						content: 'Dimensional Illusion set: 7 min, Parthenos LED overcook, 128 BPM.',
						nodeType: 'event',
						tags: ['dimensional-illusion', 'poi'],
						retention: 0.51,
						createdAt: '2026-04-02T12:12:00Z',
					},
				],
			},
			{
				similarity: 0.76,
				suggestedAction: 'review',
				memories: [
					{
						id: 'm-011',
						content:
							'Vestige v2.0.7 shipped active forgetting via Anderson 2025 top-down inhibition + Davis Rac1 cascade. Suppress compounds, reversible 24h.',
						nodeType: 'fact',
						tags: ['vestige', 'release', 'active-forgetting'],
						retention: 0.93,
						createdAt: '2026-04-17T03:22:00Z',
					},
					{
						id: 'm-012',
						content:
							'Active Forgetting feature: compounds on each suppress, 24h reversible labile window, violet implosion animation in graph view.',
						nodeType: 'concept',
						tags: ['vestige', 'active-forgetting'],
						retention: 0.81,
						createdAt: '2026-04-18T09:07:00Z',
					},
				],
			},
		];

		return { clusters: filterByThreshold(all, t) };
	}

	async function detect() {
		loading = true;
		error = null;
		try {
			// TODO: swap for real endpoint /api/duplicates when backend ships.
			// See comment on mockFetchDuplicates for the exact replacement.
			const res = await mockFetchDuplicates(threshold);
			clusters = res.clusters;
			// Prune dismissals whose clusters no longer exist — prevents
			// unbounded growth across sessions and keeps the set honest.
			const presentKeys = new Set(clusters.map((c) => clusterKey(c.memories)));
			const pruned = new Set<string>();
			for (const k of dismissed) if (presentKeys.has(k)) pruned.add(k);
			dismissed = pruned;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to detect duplicates';
			clusters = [];
		} finally {
			loading = false;
		}
	}

	function onThresholdChange() {
		clearTimeout(debounceTimer);
		debounceTimer = setTimeout(detect, 250);
	}

	function dismissCluster(key: string) {
		const next = new Set(dismissed);
		next.add(key);
		dismissed = next;
	}

	function mergeCluster(key: string, winnerId: string, loserIds: string[]) {
		// TODO: POST /api/duplicates/merge { winner, losers } when backend ships.
		// For now we optimistically dismiss the cluster so the UI reflects the
		// action and rerun counts stay consistent.
		console.log('Merge cluster', key, { winnerId, loserIds });
		dismissCluster(key);
	}

	const visibleClusters = $derived(
		clusters
			.map((c) => ({ c, key: clusterKey(c.memories) }))
			.filter(({ key }) => !dismissed.has(key))
	);

	const totalDuplicates = $derived(
		visibleClusters.reduce((sum, { c }) => sum + c.memories.length, 0)
	);

	// Cluster overflow: >50 would saturate the scroll. Show a warning and cap.
	const CLUSTER_RENDER_CAP = 50;
	const overflowed = $derived(visibleClusters.length > CLUSTER_RENDER_CAP);
	const renderedClusters = $derived(
		overflowed ? visibleClusters.slice(0, CLUSTER_RENDER_CAP) : visibleClusters
	);

	onMount(() => detect());
	onDestroy(() => clearTimeout(debounceTimer));
</script>

<div class="relative mx-auto max-w-5xl space-y-6 p-6">
	<!-- Header -->
	<header class="space-y-2">
		<h1 class="text-xl font-semibold text-bright">
			Memory Hygiene — Duplicate Detection
		</h1>
		<p class="text-sm text-dim">
			Cosine-similarity clustering over embeddings. Merges reinforce the winner's FSRS state;
			losers inherit into the merged node. Dismissed clusters are hidden for this session only.
		</p>
	</header>

	<!-- Controls panel -->
	<div class="glass-panel flex flex-wrap items-center gap-5 rounded-2xl p-4">
		<!-- Threshold slider -->
		<label class="flex flex-1 min-w-64 items-center gap-3 text-xs text-dim">
			<span class="whitespace-nowrap">Similarity threshold</span>
			<input
				type="range"
				min="0.70"
				max="0.95"
				step="0.01"
				bind:value={threshold}
				oninput={onThresholdChange}
				class="flex-1 accent-synapse"
				aria-label="Similarity threshold"
			/>
			<span class="w-14 text-right font-mono text-sm text-bright">
				{(threshold * 100).toFixed(0)}%
			</span>
		</label>

		<!-- Results pill -->
		<div
			class="flex items-center gap-2 rounded-full border border-synapse/20 bg-synapse/10 px-3 py-1.5 text-xs text-text"
			role="status"
			aria-live="polite"
		>
			{#if loading}
				<span class="h-2 w-2 animate-pulse rounded-full bg-synapse-glow"></span>
				<span>Detecting…</span>
			{:else if error}
				<span class="h-2 w-2 rounded-full bg-decay"></span>
				<span class="text-decay">Error</span>
			{:else}
				<span class="h-2 w-2 rounded-full bg-synapse-glow"></span>
				<span>
					{visibleClusters.length}
					{visibleClusters.length === 1 ? 'cluster' : 'clusters'},
					{totalDuplicates} potential duplicate{totalDuplicates === 1 ? '' : 's'}
				</span>
			{/if}
		</div>

		<button
			type="button"
			onclick={detect}
			disabled={loading}
			class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
		>
			Rerun
		</button>
	</div>

	<!-- Results -->
	{#if error}
		<div
			class="glass-panel flex flex-col items-center gap-3 rounded-2xl p-10 text-center"
		>
			<div class="text-sm text-decay">Couldn't detect duplicates</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={detect}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="space-y-3">
			{#each Array(3) as _}
				<div class="glass-subtle h-40 animate-pulse rounded-2xl"></div>
			{/each}
		</div>
	{:else if visibleClusters.length === 0}
		<div
			class="glass-panel flex flex-col items-center gap-2 rounded-2xl p-12 text-center"
		>
			<div class="text-3xl">·</div>
			<div class="text-sm font-medium text-bright">
				No duplicates found above threshold.
			</div>
			<div class="text-xs text-muted">Memory is clean.</div>
		</div>
	{:else}
		<div class="space-y-4">
			{#if overflowed}
				<div
					class="glass-subtle rounded-xl border border-warning/30 bg-warning/5 px-4 py-2 text-xs text-dim"
				>
					Showing first {CLUSTER_RENDER_CAP} of {visibleClusters.length} clusters. Raise the
					threshold to narrow results.
				</div>
			{/if}
			{#each renderedClusters as { c, key } (key)}
				<div class="animate-[fadeSlide_0.35s_ease-out_both]">
					<DuplicateCluster
						similarity={c.similarity}
						memories={c.memories}
						suggestedAction={c.suggestedAction}
						onDismiss={() => dismissCluster(key)}
						onMerge={(winnerId, loserIds) => mergeCluster(key, winnerId, loserIds)}
					/>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	@keyframes fadeSlide {
		from {
			opacity: 0;
			transform: translateY(8px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
</style>

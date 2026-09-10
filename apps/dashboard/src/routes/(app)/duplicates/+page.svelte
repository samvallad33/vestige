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
	import { clusterKey } from '$components/duplicates-helpers';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import RouteStage, { type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import { createDuplicatesPasses } from '$lib/observatory/duplicates/duplicates-pass';
	import {
		normalizeDuplicatesScene,
		type DuplicateFusionCluster,
		type DuplicatesScene
	} from '$lib/observatory/duplicates/duplicates-scene';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';
	import { api } from '$stores/api';
	import type { DuplicateClusterGroup } from '$types';

	let threshold = $state(0.8);
	let clusters: DuplicateClusterGroup[] = $state([]);
	// Backend's own cluster count (res.total) — previously thrown away, which let
	// the header mislabel the summed member count as "potential duplicates".
	let apiTotal = $state(0);
	// Above this member count a cluster is a similarity-CHAINED component (A~B~C~D
	// with A likely unrelated to D), not a mergeable duplicate set → quarantine.
	const OVERSIZED_MEMBERS = 12;
	// Dismissed clusters are tracked by stable identity (sorted member ids) so
	// dismissals survive a re-fetch. If the cluster membership changes, the key
	// changes and the cluster is treated as fresh.
	let dismissed = $state(new Set<string>());
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedCluster: DuplicateFusionCluster | null = $state(null);
	let debounceTimer: ReturnType<typeof setTimeout> | undefined;

	// PORTRAIT GATE — on a phone (aspect<0.85) the fixed slider-track + value row
	// can't fit small-Android width (the value clips past the right edge), so we
	// stack the slider label. Desktop (landscape) keeps the original inline row and
	// stays byte-identical. Threshold matches TextLayerPass.portraitAdapt.
	let isPortrait = $state(false);
	onMount(() => {
		const update = () => {
			isPortrait = window.innerWidth / Math.max(1, window.innerHeight) < 0.85;
		};
		update();
		window.addEventListener('resize', update);
		return () => window.removeEventListener('resize', update);
	});

	async function detect() {
		loading = true;
		error = null;
		// A rerun invalidates any field selection — the inspector must never show a
		// cluster that is no longer part of the current detection result.
		selectedCluster = null;
		try {
			const res = await api.duplicates(threshold);
			clusters = res.clusters;
			apiTotal = res.total ?? res.clusters.length;
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
		// If the dismissed cluster is the one in the field inspector, clear it —
		// the inspector must not outlive the visible result set.
		if (selectedCluster && clusterKey(selectedCluster.memories) === key) {
			selectedCluster = null;
		}
	}

	// Merge is plan-then-apply against the dedup tool through the API. After a
	// successful apply the cluster is dropped and detection reruns, so what the
	// page shows is what the store holds; nothing is dismissed optimistically.
	function onMerged(key: string) {
		dismissCluster(key);
		detect();
	}

	const visibleClusters = $derived(
		clusters
			.map((c) => ({ c, key: clusterKey(c.memories) }))
			.filter(({ key }) => !dismissed.has(key))
	);

	// Full implicated-memory count from the ENTIRE detection result — dismissal is
	// session-local concealment, not removal, so the headline must not shrink when
	// a cluster is hidden (GPT-5.6-sol cross-review).
	const totalImplicated = $derived(clusters.reduce((sum, c) => sum + c.memories.length, 0));

	// Cluster overflow: >50 would saturate the scroll. Show a warning and cap.
	const CLUSTER_RENDER_CAP = 50;
	const overflowed = $derived(visibleClusters.length > CLUSTER_RENDER_CAP);
	const renderedClusters = $derived(
		overflowed ? visibleClusters.slice(0, CLUSTER_RENDER_CAP) : visibleClusters
	);

	const duplicatesScene = $derived.by<DuplicatesScene>(() =>
		normalizeDuplicatesScene({
			threshold,
			total: visibleClusters.length,
			clusters: visibleClusters.map(({ c }) => c)
		})
	);

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'duplicate-neck' && pick.kind !== 'duplicate-memory') return;
		selectedCluster = pick.payload as DuplicateFusionCluster;
	}

	onMount(() => detect());
	onDestroy(() => clearTimeout(debounceTimer));
</script>

<RouteStage
	organ="duplicates"
	seed={`synaptic-fusion:${threshold}:${visibleClusters.length}:${totalImplicated}`}
	scene={duplicatesScene}
	passes={createDuplicatesPasses}
	loading={loading}
	error={error}
	emptyLabel={`NO DUPLICATES ABOVE ${(threshold * 100).toFixed(0)}% SIMILARITY`}
	onpick={handleRoutePick}
/>

<!-- Bounded, explicitly-scrollable results viewport (Wave 3): the overlay used to
     rely on inherited page scroll that the fixed RouteStage composition broke —
     with a 44,000px cluster card, every action below the fold was unreachable.
     pointer-events-none on the container + auto on children means wheel events
     over the cards scroll THIS viewport while the empty gaps still reach the
     WebGPU field beneath. -->
<div
	class="relative z-10 mx-auto max-h-dvh max-w-5xl space-y-6 overflow-y-auto overscroll-contain p-6 pb-28 pointer-events-none"
>
	<!-- Header -->
	<PageHeader
		icon="duplicates"
		title="Memory Hygiene: Duplicate Detection"
		subtitle="Cosine-similarity clustering over embeddings. Merge previews a reversible plan and applies it only on your say-so; dedup undo reverses it. Oversized similarity components are quarantined for review because they chain through pairwise similarity and are not safe to merge. Dismissed clusters are hidden for this session only."
		accent="synapse"
	>
		<!-- The badge reports the last fetch, not a hope: Live after a successful
		     detection, Refreshing while one runs, Offline when the API failed. -->
		{#if !error}
			<span
				class="ping-host flex h-2 w-2 items-center justify-center text-synapse-glow"
				aria-hidden="true"
			>
				<span class="breathe h-2 w-2 rounded-full bg-synapse-glow"></span>
			</span>
		{/if}
		<span class="text-xs text-dim">{error ? 'Offline' : loading ? 'Refreshing' : 'Live'}</span>
	</PageHeader>

	<!-- Controls panel -->
	<div class="glass-panel pointer-events-auto flex flex-wrap items-center gap-5 rounded-2xl p-4">
		<!-- Threshold slider. PORTRAIT: stack (label + value on top row, full-width
		     track below) so the value never clips a narrow phone. LANDSCAPE: original
		     inline row — desktop byte-identical. -->
		{#if isPortrait}
			<label class="flex w-full flex-col gap-2 text-xs text-dim">
				<span class="flex items-baseline justify-between gap-3">
					<span class="whitespace-nowrap">Similarity threshold</span>
					<span class="font-mono text-sm text-bright">{(threshold * 100).toFixed(0)}%</span>
				</span>
				<input
					type="range"
					min="0.70"
					max="0.95"
					step="0.01"
					bind:value={threshold}
					oninput={onThresholdChange}
					class="w-full accent-synapse"
					aria-label="Similarity threshold"
				/>
			</label>
		{:else}
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
		{/if}

		<!-- Results pill + Rerun. On a phone in the ERROR state these are suppressed so
		     the standalone error card below is the SINGLE error affordance (message +
		     Retry) instead of also showing an "Error" pill + "Rerun" here — redundant
		     messaging wastes scarce vertical space. Desktop (landscape) keeps both and
		     stays byte-identical. -->
		{#if !(error && isPortrait)}
			<div
				class="flex items-center gap-2 rounded-full border border-synapse/20 bg-synapse/10 px-3 py-1.5 text-xs text-text"
				role="status"
				aria-live="polite"
			>
				{#if loading}
					<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span>
					<span>Detecting…</span>
				{:else if error}
					<span class="h-2 w-2 rounded-full bg-decay"></span>
					<span class="text-decay">Error</span>
				{:else}
					<span class="breathe h-2 w-2 rounded-full bg-synapse-glow text-synapse-glow"></span>
					<span class="tabular-nums">
						{#if visibleClusters.length < apiTotal}
							<AnimatedNumber value={visibleClusters.length} /> visible of {apiTotal} clusters
						{:else}
							<AnimatedNumber value={visibleClusters.length} />
							{visibleClusters.length === 1 ? 'cluster' : 'clusters'}
						{/if}
						· <AnimatedNumber value={totalImplicated} /> memories implicated
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
		{/if}
	</div>

	<!-- Field pick inspector -->
	{#if selectedCluster}
		<div class="glass-panel pointer-events-auto rounded-2xl border border-synapse/25 bg-black/30 p-4">
			<div class="flex flex-wrap items-center justify-between gap-3">
				<div>
					<div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">
						Synaptic neck selected
					</div>
					<div class="mt-1 text-sm text-bright">
						{selectedCluster.memories.length} memories · {(selectedCluster.similarity * 100).toFixed(1)}% similar · winner {selectedCluster.winnerId.slice(0, 8)}
					</div>
					<div class="mt-1 max-w-2xl text-xs text-muted">
						Real pair key: {selectedCluster.id}. Mismatch filaments: {selectedCluster.mismatchTokens.length ? selectedCluster.mismatchTokens.join(', ') : 'none exposed'}.
					</div>
				</div>
				<button
					type="button"
					onclick={() => (selectedCluster = null)}
					class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
				>
					Clear field focus
				</button>
			</div>
		</div>
	{/if}

	<!-- Results -->
	{#if error}
		<div
			class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"
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
		<div class="pointer-events-auto space-y-3">
			{#each Array(3) as _}
				<div class="glass-subtle shimmer h-40 rounded-2xl"></div>
			{/each}
		</div>
	{:else if visibleClusters.length === 0}
		<div
			class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"
		>
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"
			>
				<Icon name="sparkle" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">
				No duplicates found — your memory is clean.
			</div>
			<div class="max-w-sm text-xs text-muted">
				Nothing clusters above {(threshold * 100).toFixed(0)}% similarity. Lower the threshold to
				surface looser matches.
			</div>
		</div>
	{:else}
		<div class="pointer-events-auto space-y-4">
			{#if overflowed}
				<div
					class="glass-subtle rounded-xl border border-warning/30 bg-warning/5 px-4 py-2 text-xs text-dim"
				>
					Showing first {CLUSTER_RENDER_CAP} of {visibleClusters.length} clusters. Raise the
					threshold to narrow results.
				</div>
			{/if}
			{#each renderedClusters as { c, key }, i (key)}
				<div
					class="spotlight-surface lift rounded-2xl"
					use:reveal={{ delay: Math.min(i * 40, 400), y: 14 }}
					use:spotlight
				>
					<div class="relative z-[1]">
						<DuplicateCluster
							similarity={c.similarity}
							memories={c.memories}
							suggestedAction={c.suggestedAction}
							oversized={c.memories.length > OVERSIZED_MEMBERS}
							onDismiss={() => dismissCluster(key)}
							onPlan={(ids) => api.duplicatesPlan(ids)}
							onApply={(planId) => api.duplicatesApply(planId)}
							onMerged={() => onMerged(key)}
						/>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
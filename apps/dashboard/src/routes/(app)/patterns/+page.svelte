<!--
  Cross-Project Intelligence — Pattern Transfer Heatmap
  Dashboard exposure of the CrossProjectLearner backend state. Visualizes
  which coding patterns were learned in one project and reused in another,
  across all six tracked categories (ErrorHandling, AsyncConcurrency, Testing,
  Architecture, Performance, Security).

  Category tabs filter the pattern set. Heatmap cell click filters the
  "Top Transferred Patterns" sidebar to a specific origin → destination pair.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import PatternTransferHeatmap from '$components/PatternTransferHeatmap.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import type {
		CrossProjectCategory as Category,
		CrossProjectPatternsResponse
	} from '$types';

	const CATEGORIES: readonly Category[] = [
		'ErrorHandling',
		'AsyncConcurrency',
		'Testing',
		'Architecture',
		'Performance',
		'Security'
	] as const;

	const CATEGORY_COLORS: Record<Category, string> = {
		ErrorHandling: 'var(--color-decay)',
		AsyncConcurrency: 'var(--color-synapse-glow)',
		Testing: 'var(--color-recall)',
		Architecture: 'var(--color-dream-glow)',
		Performance: 'var(--color-warning)',
		Security: 'var(--color-node-pattern)'
	};

	let activeCategory = $state<'All' | Category>('All');
	let data = $state<CrossProjectPatternsResponse>({ projects: [], patterns: [] });
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedCell = $state<{ from: string; to: string } | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			data = await api.crossProjectPatterns();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load pattern transfers';
			data = { projects: [], patterns: [] };
		} finally {
			loading = false;
		}
	}

	onMount(() => load());

	// Filter by active category first — this drives both the heatmap and sidebar.
	const categoryFiltered = $derived(
		activeCategory === 'All'
			? data.patterns
			: data.patterns.filter((p) => p.category === activeCategory)
	);

	// Sidebar list: if a cell is selected, show only A → B; else show all (top-N
	// by transfer_count). Still respects active category via categoryFiltered.
	const sidebarPatterns = $derived.by(() => {
		const list = selectedCell
			? categoryFiltered.filter(
					(p) =>
						p.origin_project === selectedCell!.from &&
						p.transferred_to.includes(selectedCell!.to)
				)
			: categoryFiltered;
		return [...list].sort((a, b) => b.transfer_count - a.transfer_count);
	});

	// Stats footer
	const totalTransfers = $derived(
		categoryFiltered.reduce((sum, p) => sum + p.transferred_to.length, 0)
	);
	const projectCount = $derived(data.projects.length);
	const patternCount = $derived(categoryFiltered.length);

	function selectCategory(c: 'All' | Category) {
		activeCategory = c;
		selectedCell = null; // clear cell filter when switching category
	}

	function onCellClick(from: string, to: string) {
		if (selectedCell && selectedCell.from === from && selectedCell.to === to) {
			selectedCell = null;
		} else {
			selectedCell = { from, to };
		}
	}

	function clearCellFilter() {
		selectedCell = null;
	}

	function relativeDate(iso: string): string {
		const then = new Date(iso).getTime();
		const now = Date.now();
		const days = Math.floor((now - then) / 86_400_000);
		if (days <= 0) return 'today';
		if (days === 1) return '1d ago';
		if (days < 30) return `${days}d ago`;
		const months = Math.floor(days / 30);
		return `${months}mo ago`;
	}
</script>

<div class="relative mx-auto max-w-7xl space-y-6 p-6">
	<PageHeader
		icon="patterns"
		title="Cross-Project Intelligence"
		subtitle="Patterns learned here, applied there — across every tracked project."
		accent="dream"
	>
		{#if !loading && !error}
			<span class="glass-subtle rounded-full px-3 py-1 text-xs text-dim tabular-nums">
				<span class="font-semibold text-bright"><AnimatedNumber value={patternCount} /></span>
				pattern{patternCount === 1 ? '' : 's'}
				·
				<span class="font-semibold text-bright"><AnimatedNumber value={totalTransfers} /></span>
				transfer{totalTransfers === 1 ? '' : 's'}
			</span>
		{/if}
	</PageHeader>

	<!-- Category tabs — switching category clears the selected heatmap cell. -->
	<div class="glass-panel flex flex-wrap items-center gap-1.5 rounded-2xl p-2 enter">
		<button
			type="button"
			onclick={() => selectCategory('All')}
			class="lift rounded-lg px-3 py-1.5 text-xs font-medium transition {activeCategory === 'All'
				? 'bg-synapse/25 text-synapse-glow shadow-[0_0_16px_-4px_var(--color-synapse-glow)]'
				: 'text-dim hover:bg-white/[0.04] hover:text-text'}"
		>
			<span class="inline-flex items-center gap-1.5">
				<Icon name="sparkle" size={13} />
				All
			</span>
		</button>
		{#each CATEGORIES as cat (cat)}
			<button
				type="button"
				onclick={() => selectCategory(cat)}
				class="lift flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition {activeCategory ===
				cat
					? 'bg-synapse/25 text-synapse-glow shadow-[0_0_16px_-4px_var(--color-synapse-glow)]'
					: 'text-dim hover:bg-white/[0.04] hover:text-text'}"
			>
				<span
					class="h-1.5 w-1.5 rounded-full transition-all duration-300 {activeCategory === cat
						? 'scale-150'
						: ''}"
					style="background: {CATEGORY_COLORS[cat]}; {activeCategory === cat
						? `box-shadow: 0 0 8px ${CATEGORY_COLORS[cat]};`
						: ''}"
				></span>
				{cat}
			</button>
		{/each}
	</div>

	{#if error}
		<div class="glass-panel enter flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-decay/80"><Icon name="contradictions" size={28} /></div>
			<div class="text-sm text-decay">Couldn't load pattern transfers</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={load}
				class="lift mt-2 inline-flex items-center gap-1.5 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30"
			>
				<Icon name="pulse" size={14} />
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
			<div class="glass-subtle shimmer h-[520px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[520px] rounded-2xl"></div>
		</div>
	{:else}
		<!-- Main grid: heatmap (70%) + sidebar -->
		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
			<!-- Heatmap column -->
			<div class="space-y-4 enter">
				<PatternTransferHeatmap
					projects={data.projects}
					patterns={categoryFiltered}
					{selectedCell}
					{onCellClick}
				/>

				{#if selectedCell}
					<div
						class="glass-subtle enter flex items-center justify-between rounded-xl px-4 py-2.5 text-xs"
					>
						<div class="flex items-center gap-2">
							<span class="text-synapse-glow"><Icon name="filter" size={13} /></span>
							<span class="text-muted">Filtered to</span>
							<span class="font-mono text-bright">{selectedCell.from}</span>
							<span class="text-synapse-glow">→</span>
							<span class="font-mono text-bright">{selectedCell.to}</span>
						</div>
						<button
							type="button"
							onclick={clearCellFilter}
							class="inline-flex items-center gap-1 rounded-md bg-white/[0.04] px-2 py-1 text-dim transition hover:bg-white/[0.08] hover:text-text"
						>
							<Icon name="close" size={12} />
							Clear
						</button>
					</div>
				{/if}
			</div>

			<!-- Sidebar: Top Transferred Patterns -->
			<aside class="glass-panel enter flex flex-col rounded-2xl p-4">
				<div class="mb-3 flex items-center justify-between">
					<h2 class="flex items-center gap-2 text-sm font-semibold text-bright">
						<span class="text-dream-glow"><Icon name="patterns" size={15} /></span>
						Top Transferred Patterns
					</h2>
					<span class="text-[11px] text-muted tabular-nums">
						<AnimatedNumber value={sidebarPatterns.length} />
						{sidebarPatterns.length === 1 ? 'pattern' : 'patterns'}
					</span>
				</div>

				{#if sidebarPatterns.length === 0}
					<div class="flex flex-1 flex-col items-center justify-center gap-3 py-12 text-center">
						<div class="breathe text-dim/70"><Icon name="explore" size={32} /></div>
						<div class="text-xs font-medium text-dim">No matching patterns</div>
						<div class="max-w-[220px] text-[11px] text-muted">
							{selectedCell
								? 'No patterns transferred from this origin to this destination yet — try another cell or clear the filter.'
								: 'Nothing in this category yet. Pick another category to explore what travels between projects.'}
						</div>
					</div>
				{:else}
					<ul class="flex-1 space-y-2 overflow-y-auto pr-1" style="max-height: 560px;">
						{#each sidebarPatterns as p, i (p.name)}
							<li
								use:reveal={{ y: 12, delay: Math.min(i * 45, 360) }}
								class="lift rounded-lg border border-synapse/5 bg-white/[0.02] p-3 transition hover:border-synapse/20 hover:bg-white/[0.04]"
							>
								<div class="flex items-start justify-between gap-2">
									<div class="min-w-0 flex-1 space-y-1.5">
										<div class="truncate text-xs font-medium text-bright" title={p.name}>
											{p.name}
										</div>
										<div class="flex flex-wrap items-center gap-1.5">
											<span
												class="rounded-full border px-1.5 py-0.5 text-[10px] font-medium"
												style="border-color: {CATEGORY_COLORS[
													p.category
												]}66; color: {CATEGORY_COLORS[p.category]}; background: {CATEGORY_COLORS[
													p.category
												]}1a;"
											>
												{p.category}
											</span>
											<span class="text-[10px] text-muted">{relativeDate(p.last_used)}</span>
										</div>
										<div class="flex items-center gap-1.5 text-[11px] text-dim">
											<span class="font-mono text-text">{p.origin_project}</span>
											<span class="text-synapse-glow">→</span>
											<span class="text-muted">
												{p.transferred_to.length}
												{p.transferred_to.length === 1 ? 'project' : 'projects'}
											</span>
										</div>
									</div>
									<div class="flex flex-shrink-0 flex-col items-end gap-1">
										<span
											class="rounded-full bg-synapse/15 px-2 py-0.5 text-xs font-semibold text-synapse-glow tabular-nums"
										>
											<AnimatedNumber value={p.transfer_count} />
										</span>
										<span class="text-[10px] text-muted tabular-nums">
											{(p.confidence * 100).toFixed(0)}%
										</span>
									</div>
								</div>
							</li>
						{/each}
					</ul>
				{/if}
			</aside>
		</div>

		<!-- Stats footer -->
		<footer
			class="glass-subtle enter flex flex-wrap items-center justify-between gap-3 rounded-xl px-4 py-3 text-xs text-dim"
		>
			<div class="tabular-nums">
				<span class="font-semibold text-bright"><AnimatedNumber value={patternCount} /></span>
				pattern{patternCount === 1 ? '' : 's'} across
				<span class="font-semibold text-bright"><AnimatedNumber value={projectCount} /></span>
				project{projectCount === 1 ? '' : 's'},
				<span class="font-semibold text-bright"><AnimatedNumber value={totalTransfers} /></span>
				total transfer{totalTransfers === 1 ? '' : 's'}
			</div>
			<div class="inline-flex items-center gap-1.5 text-muted">
				<span
					class="h-1.5 w-1.5 rounded-full"
					style="background: {activeCategory === 'All'
						? 'var(--color-synapse-glow)'
						: CATEGORY_COLORS[activeCategory]}"
				></span>
				{activeCategory === 'All' ? 'All categories' : activeCategory}
			</div>
		</footer>
	{/if}
</div>

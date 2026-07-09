<script lang="ts">
	import { onMount } from 'svelte';
	import ContradictionArcs, { type Contradiction } from '$components/ContradictionArcs.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import {
		severityColor,
		severityLabel,
		truncate,
		uniqueMemoryCount,
		avgTrustDelta as avgTrustDeltaFn,
	} from '$components/contradiction-helpers';

	// Live pairs from /api/contradictions — the contradiction-analysis
	// primitives behind deep_reference (only flagged when BOTH memories clear
	// the trust floor). Sorted by similarity desc by the backend.
	let contradictions = $state<Contradiction[]>([]);
	// System-wide count from the backend, vs. the derived stats below which
	// reflect only the pairs the page holds.
	let totalDetected = $state(0);
	let loading = $state(true);
	let error = $state<string | null>(null);

	async function load() {
		loading = true;
		error = null;
		try {
			const res = await api.contradictions();
			contradictions = res.contradictions;
			totalDetected = res.total;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load contradictions';
			contradictions = [];
			totalDetected = 0;
		} finally {
			loading = false;
		}
	}

	onMount(() => load());

	// --- Filters ---
	type Filter = 'all' | 'recent' | 'high-trust' | 'topic';
	let filter = $state<Filter>('all');
	let topicFilter = $state<string>('');

	const uniqueTopics = $derived(
		Array.from(new Set(contradictions.map((c) => c.topic))).sort()
	);

	// --- Clear, labelled dropdown options replace the bare filter buttons +
	// native <select>. These only drive the *control*, not the filter math:
	// `filterOptions` writes into the same `filter` state, `topicOptions` into
	// the same `topicFilter` state. ---
	const filterOptions: DropdownOption[] = [
		{ value: 'all', label: 'All contradictions', icon: 'contradictions' },
		{ value: 'recent', label: 'Recent (last 7 days)', icon: 'timeline' },
		{ value: 'high-trust', label: 'High trust (>60%)', icon: 'importance' },
		{ value: 'topic', label: 'By topic', icon: 'filter' },
	];
	const topicOptions = $derived<DropdownOption[]>([
		{ value: '', label: 'All topics' },
		...uniqueTopics.map((t) => ({
			value: t,
			label: t,
			badge: contradictions.filter((c) => c.topic === t).length,
		})),
	]);

	// The Dropdown emits string values; keep the filter-reset behaviour the
	// old buttons had (clearing focus when the lens changes) without altering
	// what each filter selects.
	function onFilterChange(v: string) {
		filter = v as Filter;
		focusedPairIndex = null;
	}

	const filtered = $derived.by<Contradiction[]>(() => {
		switch (filter) {
			case 'recent':
				// Within 7 days of now — keep pairs whose newest created date is
				// within the last week.
				{
					const now = Date.now();
					const week = 7 * 24 * 60 * 60 * 1000;
					return contradictions.filter((c) => {
						const aT = c.memory_a_created ? new Date(c.memory_a_created).getTime() : 0;
						const bT = c.memory_b_created ? new Date(c.memory_b_created).getTime() : 0;
						return now - Math.max(aT, bT) <= week;
					});
				}
			case 'high-trust':
				return contradictions.filter(
					(c) => Math.min(c.trust_a, c.trust_b) > 0.6
				);
			case 'topic':
				return topicFilter
					? contradictions.filter((c) => c.topic === topicFilter)
					: contradictions;
			case 'all':
			default:
				return contradictions;
		}
	});

	// --- Selection / focused pair ---
	let focusedPairIndex = $state<number | null>(null);

	function selectPair(i: number | null) {
		focusedPairIndex = i;
	}

	// --- Stats. `totalDetected` is the backend's system-wide count; everything
	// else is derived from the pairs the page actually holds so the numbers are
	// self-consistent with what the user sees. ---
	const totalMemoriesInvolved = $derived(uniqueMemoryCount(contradictions));
	const avgTrustDelta = $derived(avgTrustDeltaFn(contradictions));

	// Map filtered index -> original index in `contradictions` so the
	// constellation and sidebar stay in sync regardless of which filter is on.
	const visibleList = $derived.by<{ orig: number; c: Contradiction }[]>(() => {
		const byId = new Map(contradictions.map((c, i) => [c.memory_a_id + '|' + c.memory_b_id, i]));
		return filtered.map((c) => ({
			orig: byId.get(c.memory_a_id + '|' + c.memory_b_id) ?? 0,
			c
		}));
	});

	// The ContradictionArcs component receives the filtered list; its internal
	// indices run 0..filtered.length-1. We translate when the sidebar clicks.
	function sidebarClick(localIndex: number) {
		focusedPairIndex = focusedPairIndex === localIndex ? null : localIndex;
	}
</script>

<div class="min-h-full p-6 space-y-6">
	<!-- Header -->
	<PageHeader
		icon="contradictions"
		title="Contradiction Constellation"
		subtitle="Where your memory disagrees with itself"
		accent="warning"
	>
		<span class="text-dim text-sm tabular-nums inline-flex items-center gap-1.5">
			<AnimatedNumber value={filtered.length} /> in view
		</span>
	</PageHeader>

	{#if error}
		<div class="glass-panel flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">Couldn't load contradictions</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={load}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-20 rounded-xl"></div>
			{/each}
		</div>
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4">
			<div class="glass-subtle shimmer min-h-[520px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[520px] rounded-2xl"></div>
		</div>
	{:else if contradictions.length === 0}
		<div class="glass-panel enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center">
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"
			>
				<Icon name="sparkle" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">
				No contradictions found — your memory agrees with itself.
			</div>
			<div class="max-w-sm text-xs text-muted">
				Pairs appear here when two trusted memories about the same topic make opposing claims.
			</div>
		</div>
	{:else}
	<!-- Stats bar -->
	<div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
		<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl text-bright font-bold tabular-nums">
				<AnimatedNumber value={totalDetected} />
			</div>
			<div class="text-xs text-dim mt-1">
				contradictions across {totalMemoriesInvolved.toLocaleString()} memories
			</div>
		</div>
		<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl font-bold tabular-nums" style="color: #f59e0b">
				<AnimatedNumber value={avgTrustDelta} decimals={2} />
			</div>
			<div class="text-xs text-dim mt-1">average trust delta</div>
		</div>
		<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="text-2xl text-bright font-bold tabular-nums">
				<AnimatedNumber value={filtered.length} />
			</div>
			<div class="text-xs text-dim mt-1">visible in current filter</div>
		</div>
		<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
			<div class="flex items-center gap-2">
				<span class="ping-host inline-flex">
					<span class="w-2 h-2 rounded-full" style="background: #ef4444"></span>
				</span>
				<div class="text-2xl font-bold tabular-nums" style="color: #ef4444">
					<AnimatedNumber value={filtered.filter((c) => c.similarity > 0.7).length} />
				</div>
			</div>
			<div class="text-xs text-dim mt-1">strong conflicts</div>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="flex flex-wrap gap-3 items-end enter">
		<Dropdown
			options={filterOptions}
			value={filter}
			label="Lens"
			icon="filter"
			onChange={onFilterChange}
		/>
		{#if filter === 'topic'}
			<Dropdown
				options={topicOptions}
				bind:value={topicFilter}
				label="Topic"
				icon="contradictions"
				placeholder="All topics"
			/>
		{/if}
		{#if focusedPairIndex !== null}
			<button
				onclick={() => (focusedPairIndex = null)}
				class="ml-auto inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs border border-subtle/30 text-dim hover:text-text hover:border-synapse/30 hover:bg-white/[0.03] transition lift"
			>
				<Icon name="close" size={13} />
				Clear focus
			</button>
		{/if}
	</div>

	<!-- Main view: constellation + sidebar -->
	<div class="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4">
		<!-- Constellation. NOTE: no `use:reveal` on this wrapper or the
		     ContradictionArcs SVG container — a transform/opacity entrance here
		     would interfere with the constellation's own layout. -->
		<div class="glass-panel rounded-2xl p-3 min-h-[520px] relative">
			{#if filtered.length === 0}
				<div class="flex flex-col items-center justify-center h-full gap-3 text-center">
					<div class="text-dim opacity-50 breathe">
						<Icon name="contradictions" size={44} strokeWidth={1.2} />
					</div>
					<p class="text-dim text-sm">No contradictions match this filter.</p>
				</div>
			{:else}
				<ContradictionArcs
					contradictions={filtered}
					{focusedPairIndex}
					onSelectPair={selectPair}
					width={800}
					height={600}
				/>
			{/if}
		</div>

		<!-- Sidebar: pair list -->
		<aside use:reveal={{ delay: 120, y: 16 }} class="glass rounded-2xl p-3 space-y-2 max-h-[620px] overflow-y-auto">
			<div class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10">
				<span class="text-xs text-dim uppercase tracking-wider">Pairs</span>
				<span class="text-xs text-muted tabular-nums"><AnimatedNumber value={visibleList.length} /></span>
			</div>

			{#if visibleList.length === 0}
				<div class="text-xs text-muted p-3">No pairs visible.</div>
			{/if}

			{#each visibleList as entry, localIndex (entry.c.memory_a_id + '|' + entry.c.memory_b_id)}
				{@const c = entry.c}
				{@const isFocused = focusedPairIndex === localIndex}
				<button
					use:reveal={{ delay: Math.min(localIndex * 35, 350), y: 10 }}
					onclick={() => sidebarClick(localIndex)}
					class="w-full text-left p-3 rounded-xl border transition lift
						{isFocused
							? 'bg-synapse/10 border-synapse/40 shadow-[0_0_12px_rgba(99,102,241,0.18)]'
							: 'border-subtle/20 hover:border-synapse/30 hover:bg-white/[0.02]'}"
				>
					<div class="flex items-center gap-2 mb-2">
						<div
							class="w-2 h-2 rounded-full"
							style="background: {severityColor(c.similarity)}"
						></div>
						<span class="text-[10px] uppercase tracking-wider" style="color: {severityColor(c.similarity)}">
							{severityLabel(c.similarity)}
						</span>
						<span class="text-[10px] text-muted ml-auto">
							{(c.similarity * 100).toFixed(0)}% sim · {c.date_diff_days}d
						</span>
					</div>
					<div class="text-xs text-text font-medium mb-1 truncate">
						{c.topic}
					</div>
					<div class="space-y-1.5">
						<div class="flex items-start gap-2 text-[11px]">
							<span class="text-muted mt-0.5 shrink-0">A</span>
							<span class="text-dim">{truncate(c.memory_a_preview)}</span>
							<span class="ml-auto text-[10px] text-muted shrink-0">
								{(c.trust_a * 100).toFixed(0)}%
							</span>
						</div>
						<div class="flex items-start gap-2 text-[11px]">
							<span class="text-muted mt-0.5 shrink-0">B</span>
							<span class="text-dim">{truncate(c.memory_b_preview)}</span>
							<span class="ml-auto text-[10px] text-muted shrink-0">
								{(c.trust_b * 100).toFixed(0)}%
							</span>
						</div>
					</div>

					{#if isFocused}
						<div class="mt-3 pt-3 border-t border-subtle/20 space-y-2">
							<div class="text-[10px] text-muted uppercase tracking-wider">Full memory A</div>
							<div class="text-[11px] text-text">{c.memory_a_preview}</div>
							{#if c.memory_a_tags && c.memory_a_tags.length > 0}
								<div class="flex flex-wrap gap-1">
									{#each c.memory_a_tags as t}
										<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{t}</span>
									{/each}
								</div>
							{/if}
							<div class="text-[10px] text-muted uppercase tracking-wider pt-1">Full memory B</div>
							<div class="text-[11px] text-text">{c.memory_b_preview}</div>
							{#if c.memory_b_tags && c.memory_b_tags.length > 0}
								<div class="flex flex-wrap gap-1">
									{#each c.memory_b_tags as t}
										<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{t}</span>
									{/each}
								</div>
							{/if}
						</div>
					{/if}
				</button>
			{/each}
		</aside>
	</div>
	{/if}
</div>

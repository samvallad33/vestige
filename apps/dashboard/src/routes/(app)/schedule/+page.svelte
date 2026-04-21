<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import FSRSCalendar from '$components/FSRSCalendar.svelte';
	import {
		classifyUrgency,
		computeScheduleStats,
		daysUntilReview,
	} from '$components/schedule-helpers';

	type WindowFilter = 'today' | 'week' | 'month' | 'all';

	let memories: Memory[] = $state([]);
	let totalMemories = $state(0);
	let loading = $state(true);
	let errored = $state(false);
	let windowFilter: WindowFilter = $state<WindowFilter>('week');

	// The corpus cap. 2000 covers a very large personal corpus while keeping
	// the request fast; `truncated` surfaces when there's more to fetch.
	const FETCH_LIMIT = 2000;

	async function fetchMemories() {
		const res = await api.memories.list({ limit: String(FETCH_LIMIT) });
		memories = res.memories;
		totalMemories = res.total;
	}

	onMount(async () => {
		try {
			await fetchMemories();
		} catch {
			errored = true;
			memories = [];
		} finally {
			loading = false;
		}
	});

	// Only memories that actually have an FSRS next-review timestamp.
	let scheduled = $derived(memories.filter((m) => !!m.nextReviewAt));

	let now = $derived(new Date());
	let truncated = $derived(totalMemories > memories.length);

	// Memories that match the currently-selected window. The calendar itself
	// always renders the full 6-week window for spatial context — this filter
	// drives the sidebar counts and the right-hand list. Day-granular so the
	// buckets match the calendar cell colors (both go through classifyUrgency).
	let filtered = $derived(
		(() => {
			const wf: WindowFilter = windowFilter;
			if (wf === 'all') return scheduled;
			return scheduled.filter((m) => {
				const u = classifyUrgency(now, m.nextReviewAt);
				if (u === 'none') return false;
				if (wf === 'today') return u === 'overdue' || u === 'today';
				if (wf === 'week') return u !== 'future';
				// month: anything due within 30 whole days
				const d = daysUntilReview(now, m.nextReviewAt);
				return d !== null && d <= 30;
			});
		})()
	);

	// Stats — due today, this week, this month — and avg days-until-review.
	let stats = $derived(computeScheduleStats(now, scheduled));

	async function runConsolidation() {
		loading = true;
		try {
			await api.consolidate();
			await fetchMemories();
			errored = false;
		} catch {
			errored = true;
		} finally {
			loading = false;
		}
	}

	// The filter buttons.
	const FILTERS: { key: WindowFilter; label: string }[] = [
		{ key: 'today', label: 'Due today' },
		{ key: 'week', label: 'This week' },
		{ key: 'month', label: 'This month' },
		{ key: 'all', label: 'All upcoming' }
	];
</script>

<div class="p-6 max-w-7xl mx-auto space-y-6">
	<div class="flex items-center justify-between flex-wrap gap-3">
		<div>
			<h1 class="text-xl text-bright font-semibold">Review Schedule</h1>
			<p class="text-xs text-dim mt-1">FSRS-6 next-review dates across your memory corpus</p>
		</div>
		<div class="flex gap-1 p-1 glass-subtle rounded-xl">
			{#each FILTERS as f}
				<button
					type="button"
					onclick={() => (windowFilter = f.key)}
					class="px-3 py-1.5 text-xs rounded-lg transition-all
						{windowFilter === f.key
						? 'bg-synapse/20 text-synapse-glow border border-synapse/30'
						: 'text-dim hover:text-text hover:bg-white/[0.03] border border-transparent'}"
				>
					{f.label}
				</button>
			{/each}
		</div>
	</div>

	{#if !loading && !errored && truncated}
		<div class="px-3 py-2 glass-subtle rounded-lg text-[11px] text-dim">
			Showing the first {memories.length.toLocaleString()} of {totalMemories.toLocaleString()} memories.
			Schedule reflects this slice only.
		</div>
	{/if}

	{#if loading}
		<div class="grid lg:grid-cols-[1fr_280px] gap-6">
			<div class="space-y-3">
				<div class="h-14 glass-subtle rounded-xl animate-pulse"></div>
				<div class="grid grid-cols-7 gap-2">
					{#each Array(42) as _}
						<div class="aspect-square glass-subtle rounded-lg animate-pulse"></div>
					{/each}
				</div>
			</div>
			<div class="space-y-3">
				{#each Array(5) as _}
					<div class="h-20 glass-subtle rounded-xl animate-pulse"></div>
				{/each}
			</div>
		</div>
	{:else if errored}
		<div class="p-10 glass rounded-xl text-center space-y-3">
			<p class="text-sm text-decay">API unavailable.</p>
			<p class="text-xs text-dim">Could not fetch memories from /api/memories.</p>
		</div>
	{:else if scheduled.length === 0}
		<div class="p-10 glass rounded-xl text-center space-y-4">
			<div class="text-4xl text-dream/40">◷</div>
			<p class="text-sm text-bright font-medium">FSRS review schedule not yet populated.</p>
			<p class="text-xs text-dim max-w-md mx-auto">
				None of your {memories.length} memor{memories.length === 1 ? 'y has' : 'ies have'} a
				<code class="text-muted">nextReviewAt</code> timestamp yet. Run consolidation to compute
				next-review dates via FSRS-6.
			</p>
			<button
				type="button"
				onclick={runConsolidation}
				class="px-4 py-2 bg-warning/20 border border-warning/40 text-warning text-sm rounded-xl hover:bg-warning/30 transition"
			>
				Run Consolidation
			</button>
		</div>
	{:else}
		<div class="grid lg:grid-cols-[1fr_280px] gap-6">
			<!-- Calendar -->
			<div class="min-w-0">
				<FSRSCalendar memories={scheduled} />
			</div>

			<!-- Sidebar: stats -->
			<aside class="space-y-4">
				<div class="p-5 glass rounded-xl space-y-4">
					<h2 class="text-xs text-dim font-semibold uppercase tracking-wider">Queue</h2>
					<div class="space-y-3">
						{#if stats.overdue > 0}
							<div class="flex items-baseline justify-between">
								<span class="text-xs text-dim">Overdue</span>
								<span class="text-2xl font-bold text-decay">{stats.overdue}</span>
							</div>
						{/if}
						<div class="flex items-baseline justify-between">
							<span class="text-xs text-dim">Due today</span>
							<span class="text-2xl font-bold text-warning">{stats.dueToday}</span>
						</div>
						<div class="flex items-baseline justify-between">
							<span class="text-xs text-dim">This week</span>
							<span class="text-2xl font-bold text-synapse-glow">{stats.dueThisWeek}</span>
						</div>
						<div class="flex items-baseline justify-between">
							<span class="text-xs text-dim">This month</span>
							<span class="text-2xl font-bold text-dream-glow">{stats.dueThisMonth}</span>
						</div>
					</div>
					<div class="pt-3 border-t border-synapse/10">
						<div class="flex items-baseline justify-between">
							<span class="text-xs text-dim">Avg days until review</span>
							<span class="text-lg font-semibold text-text">{stats.avgDays.toFixed(1)}</span>
						</div>
						<p class="text-[10px] text-muted mt-1">
							Across {scheduled.length} scheduled memor{scheduled.length === 1 ? 'y' : 'ies'}
						</p>
					</div>
				</div>

				<!-- Filtered list preview -->
				<div class="p-5 glass-subtle rounded-xl space-y-3">
					<div class="flex items-center justify-between">
						<h2 class="text-xs text-dim font-semibold uppercase tracking-wider">
							{FILTERS.find((f) => f.key === windowFilter)?.label}
						</h2>
						<span class="text-xs text-muted">{filtered.length}</span>
					</div>
					{#if filtered.length === 0}
						<p class="text-xs text-muted italic">Nothing in this window.</p>
					{:else}
						<div class="space-y-2 max-h-96 overflow-y-auto pr-1">
							{#each filtered
								.slice()
								.sort((a, b) => (a.nextReviewAt ?? '').localeCompare(b.nextReviewAt ?? ''))
								.slice(0, 50) as m (m.id)}
								{@const urgency = classifyUrgency(now, m.nextReviewAt)}
								{@const delta = daysUntilReview(now, m.nextReviewAt) ?? 0}
								<div class="p-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition">
									<p class="text-xs text-text leading-snug line-clamp-2">{m.content}</p>
									<div class="flex items-center gap-2 mt-1 text-[10px]">
										<span
											class="{urgency === 'overdue'
												? 'text-decay'
												: urgency === 'today'
													? 'text-warning'
													: urgency === 'week'
														? 'text-synapse-glow'
														: 'text-dream-glow'}"
										>
											{urgency === 'overdue'
												? `${-delta}d overdue`
												: urgency === 'today'
													? 'today'
													: `in ${delta}d`}
										</span>
										<span class="text-muted">· {(m.retentionStrength * 100).toFixed(0)}%</span>
									</div>
								</div>
							{/each}
							{#if filtered.length > 50}
								<p class="text-[10px] text-muted text-center pt-1">
									+{filtered.length - 50} more
								</p>
							{/if}
						</div>
					{/if}
				</div>
			</aside>
		</div>
	{/if}
</div>

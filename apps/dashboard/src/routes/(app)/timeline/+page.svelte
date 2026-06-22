<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { TimelineDay } from '$types';
	import { NODE_TYPE_COLORS } from '$types';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import Dropdown, { type DropdownOption } from '$lib/components/Dropdown.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { reveal } from '$lib/actions/reveal';

	let timeline: TimelineDay[] = $state([]);
	let loading = $state(true);
	let days = $state(14);
	let expandedDay: string | null = $state(null);

	onMount(() => loadTimeline());

	async function loadTimeline() {
		loading = true;
		try {
			const res = await api.timeline(days, 500);
			timeline = res.timeline;
		} catch {
			timeline = [];
		} finally {
			loading = false;
		}
	}

	// Day-range filter as a clear, labelled dropdown (was a bare native <select>).
	const dayOptions: DropdownOption[] = [
		{ value: '7', label: 'Last 7 days' },
		{ value: '14', label: 'Last 14 days' },
		{ value: '30', label: 'Last 30 days' },
		{ value: '90', label: 'Last 90 days' },
		{ value: '365', label: 'Last year' },
	];
	// `days` is a number; the Dropdown binds to a string. Keep a string mirror
	// and reconcile through the change handler so reloads use the parsed value.
	let daysChoice = $state('14');
	function onDaysChange(v: string) {
		days = parseInt(v, 10);
		loadTimeline();
	}

	// Total memories across the loaded range — drives the live header count.
	let totalMemories = $derived(timeline.reduce((sum, d) => sum + d.count, 0));
</script>

<div class="p-6 max-w-4xl mx-auto space-y-6">
	<PageHeader
		icon="timeline"
		title="Timeline"
		subtitle="Watch your memories accumulate, day by day"
		accent="synapse"
	>
		<div class="flex items-center gap-4">
			<span class="text-dim text-sm tabular-nums">
				<AnimatedNumber value={totalMemories} /> memories
			</span>
			<Dropdown
				options={dayOptions}
				bind:value={daysChoice}
				label="Range"
				icon="schedule"
				onChange={onDaysChange}
			/>
		</div>
	</PageHeader>

	{#if loading}
		<div class="space-y-4">
			{#each Array(7) as _}
				<div class="h-16 glass-subtle rounded-xl shimmer"></div>
			{/each}
		</div>
	{:else if timeline.length === 0}
		<div class="enter flex flex-col items-center justify-center text-center py-20 gap-4">
			<div class="text-dim opacity-60 breathe"><Icon name="timeline" size={48} strokeWidth={1.2} /></div>
			<p class="text-dim text-sm max-w-sm">
				No memories in this window yet — widen the range or come back once Vestige has
				been remembering a while.
			</p>
		</div>
	{:else}
		<div class="relative">
			<!-- Timeline line -->
			<div class="absolute left-6 top-0 bottom-0 w-px bg-synapse/15"></div>

			<div class="space-y-4">
				{#each timeline as day, i (day.date)}
					<div use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }} class="relative pl-14">
						<!-- Dot -->
						<div class="absolute left-4 top-3 w-5 h-5 rounded-full border-2 border-synapse bg-void flex items-center justify-center">
							<div class="w-2 h-2 rounded-full bg-synapse breathe"></div>
						</div>

						<button onclick={() => expandedDay = expandedDay === day.date ? null : day.date}
							class="lift w-full text-left p-4 glass-subtle rounded-xl hover:bg-white/[0.04] transition-all duration-200
								{expandedDay === day.date ? '!border-synapse/40 glow-synapse' : ''}">
							<div class="flex items-center justify-between">
								<div class="flex items-baseline gap-2">
									<span class="text-sm text-bright font-medium">{day.date}</span>
									<span class="text-xs text-dim tabular-nums">
										<AnimatedNumber value={day.count} /> memories
									</span>
								</div>
								<!-- Dots for memory types -->
								<div class="flex items-center gap-1">
									{#each day.memories.slice(0, 10) as m}
										<div class="w-2 h-2 rounded-full" style="background: {NODE_TYPE_COLORS[m.nodeType] || '#8B95A5'}; opacity: {0.3 + m.retentionStrength * 0.7}; box-shadow: 0 0 5px {NODE_TYPE_COLORS[m.nodeType] || '#8B95A5'}66"></div>
									{/each}
									{#if day.memories.length > 10}
										<span class="text-xs text-muted tabular-nums">+{day.memories.length - 10}</span>
									{/if}
								</div>
							</div>

							{#if expandedDay === day.date}
								<div class="enter mt-3 pt-3 border-t border-synapse/10 space-y-2">
									{#each day.memories as m}
										<div class="flex items-start gap-2 text-sm">
											<div class="w-2 h-2 mt-1.5 rounded-full flex-shrink-0" style="background: {NODE_TYPE_COLORS[m.nodeType] || '#8B95A5'}"></div>
											<div class="flex-1 min-w-0">
												<span class="text-dim line-clamp-1">{m.content}</span>
											</div>
											<span class="text-xs text-muted flex-shrink-0 tabular-nums">{(m.retentionStrength * 100).toFixed(0)}%</span>
										</div>
									{/each}
								</div>
							{/if}
						</button>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

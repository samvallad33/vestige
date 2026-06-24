<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { IntentionItem } from '$types';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import Dropdown, { type DropdownOption } from '$lib/components/Dropdown.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';

	let intentions: IntentionItem[] = $state([]);
	let predictions: Record<string, unknown>[] = $state([]);
	let loading = $state(true);
	let statusFilter = $state('active');

	const STATUS_COLORS: Record<string, string> = {
		active: 'text-synapse-glow bg-synapse/10 border-synapse/30',
		fulfilled: 'text-recall bg-recall/10 border-recall/30',
		cancelled: 'text-dim bg-white/[0.03] border-subtle/20',
		snoozed: 'text-dream-glow bg-dream/10 border-dream/30',
	};

	const PRIORITY_LABELS: Record<number, string> = {
		4: 'critical',
		3: 'high',
		2: 'normal',
		1: 'low',
	};

	const PRIORITY_COLORS: Record<number, string> = {
		4: 'text-decay',
		3: 'text-amber-400',
		2: 'text-dim',
		1: 'text-muted',
	};

	// Each trigger kind gets a drawn-on Icon instead of a Unicode glyph, so the
	// list reads as part of the same premium icon system as the rest of the app.
	const TRIGGER_ICONS: Record<string, 'schedule' | 'graph' | 'pulse' | 'intentions'> = {
		time: 'schedule',
		context: 'graph',
		event: 'pulse',
		manual: 'intentions',
	};

	// Clear, labelled dropdown options replace the row of status tabs. Same values,
	// same filtering behavior — just clearer at a glance.
	const statusOptions: DropdownOption[] = [
		{ value: 'active', label: 'Active', color: '#6366f1' },
		{ value: 'fulfilled', label: 'Fulfilled', color: '#10b981' },
		{ value: 'snoozed', label: 'Snoozed', color: '#a78bfa' },
		{ value: 'cancelled', label: 'Cancelled', color: '#8B95A5' },
		{ value: 'all', label: 'All', color: '#00D4FF' },
	];

	function summarizeTrigger(intention: IntentionItem): string {
		// The API returns trigger_data as a JSON-encoded string. Parse it, pick the
		// most human-readable field, then truncate for display.
		let result: string;
		try {
			const data = JSON.parse(intention.trigger_data || '{}') as Record<string, unknown>;
			if (typeof data.condition === 'string' && data.condition) {
				result = data.condition;
			} else if (typeof data.topic === 'string' && data.topic) {
				result = data.topic;
			} else if (typeof data.at === 'string' && data.at) {
				try {
					result = new Date(data.at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
				} catch {
					result = data.at;
				}
			} else if (typeof data.in_minutes === 'number') {
				result = `in ${data.in_minutes} min`;
			} else if (typeof data.inMinutes === 'number') {
				result = `in ${data.inMinutes} min`;
			} else if (typeof data.codebase === 'string' && data.codebase) {
				const fp = typeof data.filePattern === 'string' && data.filePattern ? `/${data.filePattern}` : '';
				result = `${data.codebase}${fp}`;
			} else {
				result = intention.trigger_type;
			}
		} catch {
			result = intention.trigger_type;
		}
		return result.length > 40 ? result.slice(0, 37) + '...' : result;
	}

	onMount(async () => {
		await loadData();
	});

	async function loadData() {
		loading = true;
		try {
			const [intRes, predRes] = await Promise.all([
				api.intentions(statusFilter),
				api.predict()
			]);
			intentions = intRes.intentions || [];
			predictions = (predRes.predictions || []) as Record<string, unknown>[];
		} catch { /* ignore */ }
		finally { loading = false; }
	}

	async function changeFilter(status: string) {
		statusFilter = status;
		await loadData();
	}

	function formatDate(d: string | undefined): string {
		if (!d) return '';
		try {
			return new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
		} catch { return d; }
	}
</script>

<div class="p-6 max-w-5xl mx-auto space-y-8">
	<PageHeader
		icon="intentions"
		title="Intentions & Predictions"
		subtitle="Prospective memory and the needs Vestige sees coming"
		accent="memory"
	>
		<span class="text-dim text-sm tabular-nums">
			<AnimatedNumber value={intentions.length} /> intentions
		</span>
	</PageHeader>

	<!-- Intentions Section -->
	<div class="space-y-4 enter">
		<div class="flex flex-wrap items-center justify-between gap-3">
			<div class="flex items-center gap-2 min-w-0">
				<h2 class="text-sm text-bright font-semibold">Prospective Memory</h2>
				<span class="text-xs text-muted">"Remember to do X when Y happens"</span>
			</div>

			<!-- Status filter dropdown (same values & logic as the old tabs) -->
			<Dropdown
				options={statusOptions}
				value={statusFilter}
				label="Status"
				icon="filter"
				onChange={changeFilter}
			/>
		</div>

		{#if loading}
			<div class="space-y-2">
				{#each Array(4) as _}
					<div class="h-16 glass-subtle rounded-xl shimmer"></div>
				{/each}
			</div>
		{:else if intentions.length === 0}
			<div class="enter flex flex-col items-center justify-center text-center py-14 gap-4">
				<div class="text-dim opacity-60 breathe"><Icon name="intentions" size={44} strokeWidth={1.2} /></div>
				<p class="text-dim text-sm max-w-sm">
					No {statusFilter === 'all' ? '' : statusFilter + ' '}intentions yet — say "Remind me…" in conversation and Vestige will hold the thought for you.
				</p>
			</div>
		{:else}
			<div class="space-y-2">
				{#each intentions as intention, i (intention.id)}
					<div
						use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }}
						use:spotlight
						class="spotlight-surface p-4 glass-subtle rounded-xl lift transition-all duration-200"
					>
						<div class="relative z-[1] flex items-start gap-3">
							<!-- Trigger icon -->
							<div class="w-9 h-9 rounded-lg bg-white/[0.04] border border-synapse/10 flex items-center justify-center text-synapse-glow flex-shrink-0">
								<Icon name={TRIGGER_ICONS[intention.trigger_type] || 'intentions'} size={18} />
							</div>

							<div class="flex-1 min-w-0">
								<p class="text-sm text-text leading-relaxed">{intention.content}</p>
								<div class="flex flex-wrap gap-2 mt-2">
									<!-- Status badge -->
									<span class="px-2 py-0.5 text-[10px] rounded-lg border {STATUS_COLORS[intention.status] || 'text-dim bg-white/[0.03] border-subtle/20'}">
										{intention.status}
									</span>
									<!-- Priority -->
									<span class="text-[10px] {PRIORITY_COLORS[intention.priority] || 'text-muted'}">
										{PRIORITY_LABELS[intention.priority] || 'normal'} priority
									</span>
									<!-- Trigger -->
									<span class="text-[10px] text-muted">
										{intention.trigger_type}: {summarizeTrigger(intention)}
									</span>
									{#if intention.deadline}
										<span class="text-[10px] text-dream-glow">
											deadline: {formatDate(intention.deadline)}
										</span>
									{/if}
									{#if intention.snoozed_until}
										<span class="text-[10px] text-muted">
											snoozed until {formatDate(intention.snoozed_until)}
										</span>
									{/if}
								</div>
							</div>

							<span class="text-[10px] text-muted tabular-nums flex-shrink-0">{formatDate(intention.created_at)}</span>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Predictions Section -->
	<div class="pt-6 border-t border-synapse/10 space-y-4 enter">
		<div class="flex items-center gap-2">
			<span class="text-dream-glow"><Icon name="sparkle" size={16} /></span>
			<h2 class="text-sm text-bright font-semibold">Predicted Needs</h2>
			<span class="text-xs text-muted">What you might need next</span>
		</div>

		{#if loading}
			<div class="space-y-2">
				{#each Array(3) as _}
					<div class="h-14 glass-subtle rounded-xl shimmer"></div>
				{/each}
			</div>
		{:else if predictions.length === 0}
			<div class="enter flex flex-col items-center justify-center text-center py-10 gap-4">
				<div class="text-dim opacity-60 breathe"><Icon name="sparkle" size={40} strokeWidth={1.2} /></div>
				<p class="text-dim text-sm max-w-sm">
					No predictions yet — keep using Vestige and the predictive model will start surfacing what you'll reach for next.
				</p>
			</div>
		{:else}
			<div class="space-y-2">
				{#each predictions as pred, i}
					<div
						use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }}
						use:spotlight
						class="spotlight-surface p-3 glass-subtle rounded-xl lift transition-all duration-200"
					>
						<div class="relative z-[1] flex items-start gap-3">
							<div class="w-6 h-6 rounded-full bg-dream/20 text-dream-glow text-xs tabular-nums flex items-center justify-center flex-shrink-0 mt-0.5">
								{i + 1}
							</div>
							<div class="flex-1 min-w-0">
								<p class="text-sm text-text line-clamp-2">{pred.content}</p>
								<div class="flex gap-3 mt-1 text-xs text-muted">
									<span>{pred.nodeType}</span>
									{#if pred.retention}
										<span class="tabular-nums">{(Number(pred.retention) * 100).toFixed(0)}% retention</span>
									{/if}
									{#if pred.predictedNeed}
										<span class="text-dream-glow">{pred.predictedNeed} need</span>
									{/if}
								</div>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>

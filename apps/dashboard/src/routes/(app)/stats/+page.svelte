<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { SystemStats, HealthCheck, RetentionDistribution } from '$types';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';
	import { NODE_TYPE_COLORS } from '$types';

	let stats: SystemStats | null = $state(null);
	let health: HealthCheck | null = $state(null);
	let retention: RetentionDistribution | null = $state(null);
	let loading = $state(true);

	onMount(async () => {
		try {
			[stats, health, retention] = await Promise.all([
				api.stats(),
				api.health(),
				api.retentionDistribution()
			]);
		} catch {
			// API not available
		} finally {
			loading = false;
		}
	});

	function statusColor(status: string): string {
		return { healthy: '#10b981', degraded: '#f59e0b', critical: '#ef4444', empty: '#6b7280' }[status] || '#6b7280';
	}

	async function runConsolidation() {
		try {
			await api.consolidate();
			[stats, health, retention] = await Promise.all([api.stats(), api.health(), api.retentionDistribution()]);
		} catch {
			// API not available
		}
	}
</script>

<div class="p-6 max-w-5xl mx-auto space-y-6">
	<PageHeader
		icon="stats"
		title="System Stats"
		subtitle="Live health and retention across your memory store"
		accent="recall"
	>
		{#if health}
			<div class="flex items-center gap-2 px-3 py-1.5 rounded-full glass-subtle text-xs" style="color: {statusColor(health.status)}">
				<span class="ping-host w-2 h-2 rounded-full" style="color: {statusColor(health.status)}; background: {statusColor(health.status)}"></span>
				<span class="font-semibold tracking-wide">{health.status.toUpperCase()}</span>
				<span class="text-muted">v{health.version}</span>
			</div>
		{/if}
	</PageHeader>

	{#if loading}
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
			{#each Array(8) as _}
				<div class="h-24 glass-subtle rounded-xl shimmer"></div>
			{/each}
		</div>
	{:else if stats && health}
		<!-- Key metrics -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
			<div use:reveal={{ delay: 0 }} use:spotlight class="metric-card spotlight-surface lift p-4 glass rounded-xl">
				<div class="relative z-[1] text-3xl text-bright font-bold">
					<AnimatedNumber value={stats.totalMemories} />
				</div>
				<div class="relative z-[1] text-xs text-dim mt-1">Total Memories</div>
			</div>
			<div use:reveal={{ delay: 70 }} use:spotlight class="metric-card spotlight-surface lift p-4 glass rounded-xl">
				<div class="relative z-[1] text-3xl font-bold" style="color: {stats.averageRetention > 0.7 ? 'var(--color-recall)' : stats.averageRetention > 0.4 ? 'var(--color-warning)' : 'var(--color-decay)'}">
					<AnimatedNumber value={stats.averageRetention} scale={100} decimals={1} suffix="%" />
				</div>
				<div class="relative z-[1] text-xs text-dim mt-1">Avg Retention</div>
			</div>
			<div use:reveal={{ delay: 140 }} use:spotlight class="metric-card spotlight-surface lift p-4 glass rounded-xl">
				<div class="relative z-[1] text-3xl text-bright font-bold">
					<AnimatedNumber value={stats.dueForReview} />
				</div>
				<div class="relative z-[1] text-xs text-dim mt-1">Due for Review</div>
			</div>
			<div use:reveal={{ delay: 210 }} use:spotlight class="metric-card spotlight-surface lift p-4 glass rounded-xl">
				<div class="relative z-[1] text-3xl text-bright font-bold">
					<AnimatedNumber value={stats.embeddingCoverage} decimals={0} suffix="%" />
				</div>
				<div class="relative z-[1] text-xs text-dim mt-1">Embedding Coverage</div>
			</div>
		</div>

		<!-- Retention Distribution -->
		{#if retention}
			<div use:reveal class="p-6 glass rounded-xl">
				<h2 class="text-sm text-bright font-semibold mb-4">Retention Distribution</h2>
				<div class="flex items-end gap-1 h-40">
					{#each retention.distribution as bucket, i}
						{@const maxCount = Math.max(...retention.distribution.map(b => b.count), 1)}
						{@const height = (bucket.count / maxCount) * 100}
						{@const color = i < 3 ? '#ef4444' : i < 5 ? '#f59e0b' : i < 7 ? '#10b981' : '#6366f1'}
						<div class="flex-1 flex flex-col items-center gap-1">
							<span class="text-xs text-dim">{bucket.count}</span>
							<div class="w-full rounded-t transition-all duration-500" style="height: {height}%; background: {color}; opacity: 0.7; min-height: 2px"></div>
							<span class="text-xs text-muted">{bucket.range}</span>
						</div>
					{/each}
				</div>
			</div>

			<!-- Type breakdown -->
			<div use:reveal class="p-6 glass-subtle rounded-xl">
				<h2 class="text-sm text-bright font-semibold mb-4">Memory Types</h2>
				<div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
					{#each Object.entries(retention.byType) as [type, count]}
						<div class="flex items-center gap-2 text-sm rounded-lg px-2 py-1.5 hover:bg-white/[0.03] transition">
							<div class="w-3 h-3 rounded-full" style="background: {NODE_TYPE_COLORS[type] || '#8B95A5'}; box-shadow: 0 0 8px {NODE_TYPE_COLORS[type] || '#8B95A5'}80"></div>
							<span class="text-dim capitalize">{type}</span>
							<span class="text-muted ml-auto tabular-nums"><AnimatedNumber value={count} /></span>
						</div>
					{/each}
				</div>
			</div>

			<!-- Endangered memories -->
			{#if retention.endangered.length > 0}
				<div use:reveal class="p-6 glass rounded-xl !border-decay/20">
					<h2 class="text-sm text-decay font-semibold mb-3 flex items-center gap-2">
						<span class="breathe inline-block w-2 h-2 rounded-full bg-decay text-decay"></span>
						Endangered Memories ({retention.endangered.length})
					</h2>
					<div class="space-y-1 max-h-48 overflow-y-auto">
						{#each retention.endangered.slice(0, 20) as m}
							<div class="flex items-center gap-3 text-sm rounded-lg px-2 py-1 hover:bg-decay/[0.06] transition">
								<span class="text-xs text-decay tabular-nums w-9">{(m.retentionStrength * 100).toFixed(0)}%</span>
								<div class="w-16 h-1 rounded-full bg-deep overflow-hidden shrink-0">
									<div class="h-full rounded-full bg-decay" style="width: {m.retentionStrength * 100}%"></div>
								</div>
								<span class="text-dim truncate">{m.content}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		{/if}

		<!-- Actions -->
		<div use:reveal class="flex gap-3">
			<button onclick={runConsolidation}
				class="lift px-4 py-2 bg-warning/20 border border-warning/40 text-warning text-sm rounded-xl hover:bg-warning/30 transition">
				Run Consolidation
			</button>
		</div>
	{/if}
</div>

<style>
	.metric-card {
		cursor: default;
	}
</style>

<script lang="ts">
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import Icon, { type IconName } from '$lib/components/Icon.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';

	let searchQuery = $state('');
	let targetQuery = $state('');
	let sourceMemory: Memory | null = $state(null);
	let targetMemory: Memory | null = $state(null);
	let associations: Record<string, unknown>[] = $state([]);
	let mode = $state<'associations' | 'chains' | 'bridges'>('associations');
	let loading = $state(false);
	let importanceText = $state('');
	let importanceResult: Record<string, unknown> | null = $state(null);

	const MODE_INFO: Record<string, { icon: IconName; desc: string }> = {
		associations: { icon: 'activation', desc: 'Spreading activation — find related memories via graph traversal' },
		chains: { icon: 'reasoning', desc: 'Build reasoning path from source to target memory' },
		bridges: { icon: 'explore', desc: 'Find connecting memories between two concepts' },
	};

	async function findSource() {
		if (!searchQuery.trim()) return;
		loading = true;
		try {
			const res = await api.search(searchQuery, 1);
			if (res.results.length > 0) {
				sourceMemory = res.results[0];
				await explore();
			}
		} catch { /* ignore */ }
		finally { loading = false; }
	}

	async function findTarget() {
		if (!targetQuery.trim()) return;
		loading = true;
		try {
			const res = await api.search(targetQuery, 1);
			if (res.results.length > 0) {
				targetMemory = res.results[0];
				if (sourceMemory) await explore();
			}
		} catch { /* ignore */ }
		finally { loading = false; }
	}

	async function explore() {
		if (!sourceMemory) return;
		loading = true;
		try {
			const toId = (mode === 'chains' || mode === 'bridges') && targetMemory
				? targetMemory.id : undefined;
			const res = await api.explore(sourceMemory.id, mode, toId);
			associations = (res.results || res.nodes || res.chain || res.bridges || []) as Record<string, unknown>[];
		} catch { associations = []; }
		finally { loading = false; }
	}

	async function scoreImportance() {
		if (!importanceText.trim()) return;
		importanceResult = await api.importance(importanceText) as unknown as Record<string, unknown>;
	}

	function switchMode(m: typeof mode) {
		mode = m;
		if (sourceMemory) explore();
	}
</script>

<div class="p-6 max-w-5xl mx-auto space-y-8">
	<PageHeader
		icon="explore"
		title="Explore Connections"
		subtitle="Traverse the memory graph — spreading activation, reasoning chains, and conceptual bridges."
		accent="synapse"
	/>

	<!-- Mode selector -->
	<div class="grid grid-cols-3 gap-2">
		{#each (['associations', 'chains', 'bridges'] as const) as m}
			<button onclick={() => switchMode(m)}
				class="lift flex flex-col items-center gap-1 p-3 rounded-xl text-sm transition
					{mode === m
						? 'glass !border-synapse/30 text-synapse-glow'
						: 'glass-subtle text-dim hover:bg-white/[0.03]'}">
				<span class="{mode === m ? 'breathe' : ''}"><Icon name={MODE_INFO[m].icon} size={22} /></span>
				<span class="font-medium">{m.charAt(0).toUpperCase() + m.slice(1)}</span>
				<span class="text-[10px] text-muted text-center">{MODE_INFO[m].desc}</span>
			</button>
		{/each}
	</div>

	<!-- Search for source memory -->
	<div class="space-y-3">
		<span class="text-xs text-dim font-medium">Source Memory</span>
		<div class="flex gap-2">
			<input type="text" placeholder="Search for a memory to explore from..."
				bind:value={searchQuery}
				onkeydown={(e) => e.key === 'Enter' && findSource()}
				class="flex-1 px-4 py-2.5 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
					placeholder:text-muted focus:outline-none focus:border-synapse/40 transition backdrop-blur-sm" />
			<button onclick={findSource}
				class="px-4 py-2.5 bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm rounded-xl hover:bg-synapse/30 transition">
				Find
			</button>
		</div>
	</div>

	{#if sourceMemory}
		<div class="p-3 glass rounded-xl !border-synapse/20">
			<div class="text-[10px] text-synapse-glow mb-1 uppercase tracking-wider">Source</div>
			<p class="text-sm text-text">{sourceMemory.content.slice(0, 200)}</p>
			<div class="flex gap-2 mt-1.5 text-[10px] text-muted">
				<span>{sourceMemory.nodeType}</span>
				<span>{(sourceMemory.retentionStrength * 100).toFixed(0)}% retention</span>
			</div>
		</div>
	{/if}

	<!-- Target memory (for chains/bridges) -->
	{#if mode === 'chains' || mode === 'bridges'}
		<div class="space-y-3">
			<span class="text-xs text-dim font-medium">Target Memory <span class="text-muted">(for {mode})</span></span>
			<div class="flex gap-2">
				<input type="text" placeholder="Search for the target memory..."
					bind:value={targetQuery}
					onkeydown={(e) => e.key === 'Enter' && findTarget()}
					class="flex-1 px-4 py-2.5 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
						placeholder:text-muted focus:outline-none focus:border-dream/40 transition backdrop-blur-sm" />
				<button onclick={findTarget}
					class="px-4 py-2.5 bg-dream/20 border border-dream/40 text-dream-glow text-sm rounded-xl hover:bg-dream/30 transition">
					Find
				</button>
			</div>
		</div>

		{#if targetMemory}
			<div class="p-3 glass rounded-xl !border-dream/20">
				<div class="text-[10px] text-dream-glow mb-1 uppercase tracking-wider">Target</div>
				<p class="text-sm text-text">{targetMemory.content.slice(0, 200)}</p>
				<div class="flex gap-2 mt-1.5 text-[10px] text-muted">
					<span>{targetMemory.nodeType}</span>
					<span>{(targetMemory.retentionStrength * 100).toFixed(0)}% retention</span>
				</div>
			</div>
		{/if}
	{/if}

	<!-- Results -->
	{#if sourceMemory}
		{#if loading}
			<div class="space-y-3" aria-busy="true">
				<div class="flex items-center gap-2.5 text-dim">
					<Icon name="activation" size={18} class="breathe text-synapse-glow" />
					<span class="text-sm">Exploring {mode}…</span>
				</div>
				<div class="space-y-2">
					{#each Array(4) as _, i}
						<div class="shimmer p-3 glass-subtle rounded-xl flex items-start gap-3">
							<div class="shimmer w-6 h-6 rounded-full bg-white/[0.05] flex-shrink-0 mt-0.5"></div>
							<div class="flex-1 min-w-0 space-y-2">
								<div class="shimmer h-3.5 rounded bg-white/[0.05]" style="width: {88 - i * 9}%"></div>
								<div class="shimmer h-3 rounded bg-white/[0.04]" style="width: {52 - i * 6}%"></div>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{:else if associations.length > 0}
			<div class="space-y-4">
				<div class="flex items-center justify-between">
					<h2 class="text-sm text-bright font-semibold flex items-baseline gap-1.5">
						<AnimatedNumber value={associations.length} class="text-aurora font-bold" />
						<span>Connections Found</span>
					</h2>
				</div>
				<div class="space-y-2">
					{#each associations as assoc, i (i)}
						<div
							use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }}
							use:spotlight
							class="spotlight-surface lift p-3 glass-subtle rounded-xl hover:bg-white/[0.03] transition"
						>
							<div class="relative z-[1] flex items-start gap-3">
								<div class="w-6 h-6 rounded-full bg-synapse/15 text-synapse-glow text-xs flex items-center justify-center flex-shrink-0 mt-0.5 tabular-nums">
									{i + 1}
								</div>
								<div class="flex-1 min-w-0">
									<p class="text-sm text-text line-clamp-2">{assoc.content}</p>
									<div class="flex flex-wrap gap-3 mt-1.5 text-xs text-muted">
										{#if assoc.nodeType}<span class="px-1.5 py-0.5 bg-white/[0.04] rounded">{assoc.nodeType}</span>{/if}
										{#if assoc.score}<span class="tabular-nums">Score: {Number(assoc.score).toFixed(3)}</span>{/if}
										{#if assoc.similarity}<span class="tabular-nums">Similarity: {Number(assoc.similarity).toFixed(3)}</span>{/if}
										{#if assoc.retention}<span class="tabular-nums">{(Number(assoc.retention) * 100).toFixed(0)}% retention</span>{/if}
										{#if assoc.connectionType}<span class="text-synapse-glow">{assoc.connectionType}</span>{/if}
									</div>
								</div>
							</div>
						</div>
					{/each}
				</div>
			</div>
		{:else}
			<div class="enter text-center py-12 px-6 glass-subtle rounded-2xl">
				<Icon name="explore" size={40} class="breathe text-synapse-glow mx-auto mb-4 opacity-80" />
				<p class="text-sm text-bright font-medium">No connections surfaced yet</p>
				<p class="text-xs text-muted mt-1.5 max-w-sm mx-auto">
					{#if mode === 'associations'}
						This memory hasn't formed strong links here. Try a broader source query — the graph rewards more general seeds.
					{:else}
						No {mode} found between these two memories. Pick a different source or target and the path may light up.
					{/if}
				</p>
			</div>
		{/if}
	{/if}

	<!-- Importance Scorer -->
	<div class="pt-8 border-t border-synapse/10">
		<h2 class="text-lg text-bright font-semibold mb-4 flex items-center gap-2">
			<Icon name="importance" size={20} class="text-recall" />
			Importance Scorer
		</h2>
		<p class="text-xs text-muted mb-3">4-channel neuroscience scoring: novelty, arousal, reward, attention</p>
		<textarea
			bind:value={importanceText}
			placeholder="Paste any text to score its importance..."
			class="w-full h-24 px-4 py-3 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
				placeholder:text-muted resize-none focus:outline-none focus:border-synapse/40 transition backdrop-blur-sm"
		></textarea>
		<button onclick={scoreImportance}
			class="mt-2 px-4 py-2 bg-dream/20 border border-dream/40 text-dream-glow text-sm rounded-xl hover:bg-dream/30 transition">
			Score
		</button>

		{#if importanceResult}
			{@const channels = importanceResult.channels as Record<string, number> | undefined}
			{@const composite = Number(importanceResult.composite || importanceResult.compositeScore || 0)}
			<div class="enter mt-4 p-4 glass rounded-xl">
				<div class="flex items-center gap-3 mb-4">
					<AnimatedNumber value={composite} decimals={2} class="text-3xl text-aurora font-bold" />
					<span class="px-2 py-1 rounded-lg text-xs {composite > 0.6
						? 'bg-recall/20 text-recall border border-recall/30'
						: 'bg-white/[0.04] text-dim border border-subtle/20'}">
						{composite > 0.6 ? 'SAVE' : 'SKIP'}
					</span>
				</div>
				{#if channels}
					<div class="grid grid-cols-4 gap-3">
						{#each Object.entries(channels) as [channel, score]}
							<div>
								<div class="text-xs text-dim mb-1.5 capitalize">{channel}</div>
								<div class="h-2 bg-deep rounded-full overflow-hidden">
									<div class="h-full rounded-full transition-all duration-500
										{channel === 'novelty' ? 'bg-synapse' :
										 channel === 'arousal' ? 'bg-dream' :
										 channel === 'reward' ? 'bg-recall' : 'bg-amber-400'}"
										style="width: {score * 100}%"></div>
								</div>
								<div class="text-xs text-muted mt-1">{score.toFixed(2)}</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>

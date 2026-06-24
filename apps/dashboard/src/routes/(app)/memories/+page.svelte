<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import { NODE_TYPE_COLORS } from '$types';
	import MemoryAuditTrail from '$lib/components/MemoryAuditTrail.svelte';
	import PageHeader from '$lib/components/PageHeader.svelte';
	import AnimatedNumber from '$lib/components/AnimatedNumber.svelte';
	import Dropdown, { type DropdownOption } from '$lib/components/Dropdown.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { spotlight } from '$lib/actions/interactions';

	let memories: Memory[] = $state([]);
	let searchQuery = $state('');
	let selectedType = $state('');
	let selectedTag = $state('');
	let minRetention = $state(0);
	let loading = $state(true);
	let selectedMemory: Memory | null = $state(null);
	// Which inner tab of the expanded card is active. Keyed by memory id so
	// switching between cards remembers each one's last view independently.
	let expandedTab: Record<string, 'content' | 'audit'> = $state({});
	let debounceTimer: ReturnType<typeof setTimeout>;

	onMount(() => loadMemories());

	async function loadMemories() {
		loading = true;
		try {
			const params: Record<string, string> = {};
			if (searchQuery) params.q = searchQuery;
			if (selectedType) params.node_type = selectedType;
			if (selectedTag) params.tag = selectedTag;
			if (minRetention > 0) params.min_retention = String(minRetention);
			const res = await api.memories.list(params);
			memories = res.memories;
		} catch {
			memories = [];
		} finally {
			loading = false;
		}
	}

	function onSearch() {
		clearTimeout(debounceTimer);
		debounceTimer = setTimeout(loadMemories, 300);
	}

	function retentionColor(r: number): string {
		if (r > 0.7) return '#10b981';
		if (r > 0.4) return '#f59e0b';
		return '#ef4444';
	}

	// Clear, labelled dropdown options replace the dead native <select>.
	const typeOptions: DropdownOption[] = [
		{ value: '', label: 'All types' },
		{ value: 'fact', label: 'Fact', color: NODE_TYPE_COLORS.fact },
		{ value: 'concept', label: 'Concept', color: NODE_TYPE_COLORS.concept },
		{ value: 'event', label: 'Event', color: NODE_TYPE_COLORS.event },
		{ value: 'person', label: 'Person', color: NODE_TYPE_COLORS.person },
		{ value: 'place', label: 'Place', color: NODE_TYPE_COLORS.place },
		{ value: 'note', label: 'Note', color: NODE_TYPE_COLORS.note },
		{ value: 'pattern', label: 'Pattern', color: NODE_TYPE_COLORS.pattern },
		{ value: 'decision', label: 'Decision', color: NODE_TYPE_COLORS.decision },
	];

	// Retention filter as a clear dropdown of thresholds (was a bare range slider).
	const retentionOptions: DropdownOption[] = [
		{ value: '0', label: 'Any retention' },
		{ value: '0.3', label: '≥ 30% — fading & up' },
		{ value: '0.5', label: '≥ 50% — half-strength' },
		{ value: '0.7', label: '≥ 70% — well-retained' },
		{ value: '0.9', label: '≥ 90% — core memories' },
	];
	let retentionChoice = $state('0');
	function onRetentionChange(v: string) {
		minRetention = parseFloat(v);
		loadMemories();
	}
</script>

<div class="p-6 max-w-6xl mx-auto space-y-6">
	<PageHeader
		icon="memories"
		title="Memories"
		subtitle="Search, filter, and curate everything Vestige remembers"
		accent="memory"
	>
		<span class="text-dim text-sm tabular-nums">
			<AnimatedNumber value={memories.length} /> results
		</span>
	</PageHeader>

	<!-- Search & Filters -->
	<div class="flex gap-3 flex-wrap items-end enter">
		<div class="relative flex-1 min-w-64">
			<span class="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted pointer-events-none">
				<Icon name="search" size={16} />
			</span>
			<input
				type="text"
				placeholder="Search memories…  (press / to focus)"
				bind:value={searchQuery}
				oninput={onSearch}
				class="w-full pl-10 pr-4 py-2.5 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
					placeholder:text-muted focus:outline-none focus:border-synapse/40 focus:ring-2 focus:ring-synapse/15 transition backdrop-blur-sm"
			/>
		</div>
		<Dropdown
			options={typeOptions}
			bind:value={selectedType}
			label="Type"
			icon="filter"
			placeholder="All types"
			onChange={loadMemories}
		/>
		<Dropdown
			options={retentionOptions}
			bind:value={retentionChoice}
			label="Retention"
			icon="pulse"
			onChange={onRetentionChange}
		/>
	</div>

	<!-- Memory grid -->
	{#if loading}
		<div class="grid gap-3">
			{#each Array(8) as _}
				<div class="h-24 glass-subtle rounded-xl shimmer"></div>
			{/each}
		</div>
	{:else if memories.length === 0}
		<div class="enter flex flex-col items-center justify-center text-center py-20 gap-4">
			<div class="text-dim opacity-60 breathe"><Icon name="memories" size={48} strokeWidth={1.2} /></div>
			<p class="text-dim text-sm max-w-sm">
				{searchQuery || selectedType || minRetention > 0
					? 'No memories match these filters yet. Try widening your search.'
					: 'No memories yet — once Vestige starts remembering, they will surface here, alive and searchable.'}
			</p>
		</div>
	{:else}
		<div class="grid gap-3">
			{#each memories as memory, i (memory.id)}
				<button
					use:reveal={{ delay: Math.min(i * 35, 350), y: 12 }}
					use:spotlight
					onclick={() => selectedMemory = selectedMemory?.id === memory.id ? null : memory}
					class="spotlight-surface text-left p-4 glass-subtle rounded-xl hover:bg-white/[0.04]
						transition-all duration-200 group
						{selectedMemory?.id === memory.id ? '!border-synapse/40 glow-synapse live-border' : ''}"
				>
					<div class="relative z-[1] flex items-start justify-between gap-4">
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 mb-2">
								<span class="w-2 h-2 rounded-full" style="background: {NODE_TYPE_COLORS[memory.nodeType] || '#8B95A5'}; box-shadow: 0 0 6px {NODE_TYPE_COLORS[memory.nodeType] || '#8B95A5'}99"></span>
								<span class="text-xs text-dim capitalize">{memory.nodeType}</span>
								{#each memory.tags.slice(0, 3) as tag}
									<span class="text-xs px-1.5 py-0.5 bg-white/[0.04] rounded text-muted">{tag}</span>
								{/each}
							</div>
							<p class="text-sm text-text leading-relaxed line-clamp-2">{memory.content}</p>
						</div>
						<div class="flex flex-col items-end gap-1 flex-shrink-0">
							<div class="w-12 h-1.5 bg-deep rounded-full overflow-hidden">
								<div class="h-full rounded-full" style="width: {memory.retentionStrength * 100}%; background: {retentionColor(memory.retentionStrength)}"></div>
							</div>
							<span class="text-xs text-muted">{(memory.retentionStrength * 100).toFixed(0)}%</span>
						</div>
					</div>

					{#if selectedMemory?.id === memory.id}
						{@const activeTab = expandedTab[memory.id] ?? 'content'}
						<div class="relative z-[1] mt-4 pt-4 border-t border-synapse/10 space-y-3">
							<!-- Inner tab switcher: Content (default) vs Audit Trail. -->
							<div class="flex gap-1 text-[11px] uppercase tracking-wider">
								<span
									role="button"
									tabindex="0"
									onclick={(e) => { e.stopPropagation(); expandedTab[memory.id] = 'content'; }}
									onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); expandedTab[memory.id] = 'content'; } }}
									class="px-3 py-1.5 rounded-lg cursor-pointer select-none transition
										{activeTab === 'content' ? 'bg-synapse/20 text-synapse-glow border border-synapse/40' : 'bg-white/[0.03] text-dim hover:text-text border border-transparent'}"
								>Content</span>
								<span
									role="button"
									tabindex="0"
									onclick={(e) => { e.stopPropagation(); expandedTab[memory.id] = 'audit'; }}
									onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); expandedTab[memory.id] = 'audit'; } }}
									class="px-3 py-1.5 rounded-lg cursor-pointer select-none transition
										{activeTab === 'audit' ? 'bg-synapse/20 text-synapse-glow border border-synapse/40' : 'bg-white/[0.03] text-dim hover:text-text border border-transparent'}"
								>Audit Trail</span>
							</div>

							{#if activeTab === 'content'}
								<p class="text-sm text-text whitespace-pre-wrap">{memory.content}</p>
								<div class="grid grid-cols-3 gap-3 text-xs text-dim">
									<div>Storage: {(memory.storageStrength * 100).toFixed(1)}%</div>
									<div>Retrieval: {(memory.retrievalStrength * 100).toFixed(1)}%</div>
									<div>Created: {new Date(memory.createdAt).toLocaleDateString()}</div>
								</div>
							{:else}
								<div
									role="presentation"
									onclick={(e) => e.stopPropagation()}
									onkeydown={(e) => e.stopPropagation()}
								>
									<MemoryAuditTrail memoryId={memory.id} />
								</div>
							{/if}

							<div class="flex gap-2">
								<span role="button" tabindex="0" onclick={(e) => { e.stopPropagation(); api.memories.promote(memory.id); }}
									onkeydown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); api.memories.promote(memory.id); } }}
									class="px-3 py-1.5 bg-recall/20 text-recall text-xs rounded-lg hover:bg-recall/30 cursor-pointer select-none">Promote</span>
								<span role="button" tabindex="0" onclick={(e) => { e.stopPropagation(); api.memories.demote(memory.id); }}
									onkeydown={(e) => { if (e.key === 'Enter') { e.stopPropagation(); api.memories.demote(memory.id); } }}
									class="px-3 py-1.5 bg-decay/20 text-decay text-xs rounded-lg hover:bg-decay/30 cursor-pointer select-none">Demote</span>
								<!-- v2.0.7: suppress (active forgetting). Distinct from delete: the memory
									 persists but is inhibited from retrieval and actively decays. Each click
									 compounds. Graph plays the violet implosion via MemorySuppressed event. -->
								<span role="button" tabindex="0"
									onclick={async (e) => {
										e.stopPropagation();
										await api.memories.suppress(memory.id, 'dashboard trigger');
									}}
									onkeydown={async (e) => {
										if (e.key === 'Enter') {
											e.stopPropagation();
											await api.memories.suppress(memory.id, 'dashboard trigger');
										}
									}}
									title="Top-down inhibition (Anderson 2025). Compounds. Reversible for 24h."
									class="px-3 py-1.5 bg-purple-500/20 text-purple-400 text-xs rounded-lg hover:bg-purple-500/30 cursor-pointer select-none">Suppress</span>
								<span role="button" tabindex="0" onclick={async (e) => { e.stopPropagation(); await api.memories.delete(memory.id); loadMemories(); }}
									onkeydown={async (e) => { if (e.key === 'Enter') { e.stopPropagation(); await api.memories.delete(memory.id); loadMemories(); } }}
									class="px-3 py-1.5 bg-decay/10 text-decay/60 text-xs rounded-lg hover:bg-decay/20 ml-auto cursor-pointer select-none">Delete</span>
							</div>
						</div>
					{/if}
				</button>
			{/each}
		</div>
	{/if}
</div>

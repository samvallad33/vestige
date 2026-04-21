<script lang="ts">
	import ContradictionArcs, { type Contradiction } from '$components/ContradictionArcs.svelte';
	import {
		severityColor,
		severityLabel,
		truncate,
		uniqueMemoryCount,
		avgTrustDelta as avgTrustDeltaFn,
	} from '$components/contradiction-helpers';

	// TODO: swap for /api/contradictions when backend ships.
	// Expected shape matches the `Contradiction` interface in
	// $components/ContradictionArcs.svelte. Backend should derive pairs from the
	// contradiction-analysis step of deep_reference (only flag when BOTH memories
	// have >0.3 FSRS trust).
	const MOCK_CONTRADICTIONS: Contradiction[] = [
		{
			memory_a_id: 'a1',
			memory_b_id: 'b1',
			memory_a_preview: 'Dev server runs on port 3000 (default Vite config)',
			memory_b_preview: 'Dev server moved to port 3002 to avoid conflict',
			memory_a_type: 'fact',
			memory_b_type: 'decision',
			memory_a_created: '2026-01-14',
			memory_b_created: '2026-03-22',
			memory_a_tags: ['dev', 'vite'],
			memory_b_tags: ['dev', 'vite', 'decision'],
			trust_a: 0.42,
			trust_b: 0.91,
			similarity: 0.88,
			date_diff_days: 67,
			topic: 'dev server port'
		},
		{
			memory_a_id: 'a2',
			memory_b_id: 'b2',
			memory_a_preview: 'Prompt diversity helps at T>=0.6 per GPT-OSS paper',
			memory_b_preview: 'Prompt diversity monotonically HURTS at T>=0.6 (arxiv 2603.27844)',
			memory_a_type: 'concept',
			memory_b_type: 'fact',
			memory_a_created: '2026-03-30',
			memory_b_created: '2026-04-03',
			memory_a_tags: ['aimo3', 'prompting'],
			memory_b_tags: ['aimo3', 'prompting', 'evidence'],
			trust_a: 0.35,
			trust_b: 0.88,
			similarity: 0.92,
			date_diff_days: 4,
			topic: 'prompt diversity'
		},
		{
			memory_a_id: 'a3',
			memory_b_id: 'b3',
			memory_a_preview: 'Use min_p=0.05 for GPT-OSS-120B sampling',
			memory_b_preview: 'min_p scheduling fails at competition temperatures',
			memory_a_type: 'pattern',
			memory_b_type: 'fact',
			memory_a_created: '2026-04-01',
			memory_b_created: '2026-04-05',
			memory_a_tags: ['aimo3', 'sampling'],
			memory_b_tags: ['aimo3', 'sampling'],
			trust_a: 0.58,
			trust_b: 0.74,
			similarity: 0.81,
			date_diff_days: 4,
			topic: 'min_p sampling'
		},
		{
			memory_a_id: 'a4',
			memory_b_id: 'b4',
			memory_a_preview: 'LoRA rank 16 is enough for domain adaptation',
			memory_b_preview: 'LoRA rank 32 consistently outperforms rank 16 on math',
			memory_a_type: 'concept',
			memory_b_type: 'fact',
			memory_a_created: '2026-02-10',
			memory_b_created: '2026-04-12',
			memory_a_tags: ['lora', 'training'],
			memory_b_tags: ['lora', 'training', 'nemotron'],
			trust_a: 0.48,
			trust_b: 0.76,
			similarity: 0.74,
			date_diff_days: 61,
			topic: 'LoRA rank'
		},
		{
			memory_a_id: 'a5',
			memory_b_id: 'b5',
			memory_a_preview: 'Sam prefers Rust for all backend services',
			memory_b_preview: 'Sam chose Axum + Rust for Nullgaze backend',
			memory_a_type: 'note',
			memory_b_type: 'decision',
			memory_a_created: '2026-01-05',
			memory_b_created: '2026-02-18',
			memory_a_tags: ['preference', 'sam'],
			memory_b_tags: ['nullgaze', 'backend'],
			trust_a: 0.81,
			trust_b: 0.88,
			similarity: 0.42,
			date_diff_days: 44,
			topic: 'backend language'
		},
		{
			memory_a_id: 'a6',
			memory_b_id: 'b6',
			memory_a_preview: 'Warm-start from checkpoint saves 8h of training',
			memory_b_preview: 'Warm-start code never loaded the PEFT adapter correctly',
			memory_a_type: 'pattern',
			memory_b_type: 'fact',
			memory_a_created: '2026-03-11',
			memory_b_created: '2026-04-16',
			memory_a_tags: ['training', 'warm-start'],
			memory_b_tags: ['training', 'warm-start', 'bug-fix'],
			trust_a: 0.55,
			trust_b: 0.93,
			similarity: 0.79,
			date_diff_days: 36,
			topic: 'warm-start correctness'
		},
		{
			memory_a_id: 'a7',
			memory_b_id: 'b7',
			memory_a_preview: 'Three.js force-directed graph runs fine at 5k nodes',
			memory_b_preview: 'WebGL graph stutters above 2k nodes on M1 MacBook Air',
			memory_a_type: 'fact',
			memory_b_type: 'fact',
			memory_a_created: '2025-12-02',
			memory_b_created: '2026-03-29',
			memory_a_tags: ['vestige', 'graph', 'perf'],
			memory_b_tags: ['vestige', 'graph', 'perf'],
			trust_a: 0.39,
			trust_b: 0.72,
			similarity: 0.67,
			date_diff_days: 117,
			topic: 'graph performance'
		},
		{
			memory_a_id: 'a8',
			memory_b_id: 'b8',
			memory_a_preview: 'Submit GPT-OSS with 16384 token budget for AIMO',
			memory_b_preview: 'AIMO3 baseline at 32768 tokens scored 44/50',
			memory_a_type: 'pattern',
			memory_b_type: 'event',
			memory_a_created: '2026-04-04',
			memory_b_created: '2026-04-10',
			memory_a_tags: ['aimo3', 'tokens'],
			memory_b_tags: ['aimo3', 'baseline'],
			trust_a: 0.31,
			trust_b: 0.85,
			similarity: 0.73,
			date_diff_days: 6,
			topic: 'token budget'
		},
		{
			memory_a_id: 'a9',
			memory_b_id: 'b9',
			memory_a_preview: 'FSRS-6 parameters require ~1k reviews to train',
			memory_b_preview: 'FSRS-6 default parameters work fine out of the box',
			memory_a_type: 'concept',
			memory_b_type: 'concept',
			memory_a_created: '2026-01-22',
			memory_b_created: '2026-02-28',
			memory_a_tags: ['fsrs', 'training'],
			memory_b_tags: ['fsrs'],
			trust_a: 0.62,
			trust_b: 0.54,
			similarity: 0.57,
			date_diff_days: 37,
			topic: 'FSRS parameter tuning'
		},
		{
			memory_a_id: 'a10',
			memory_b_id: 'b10',
			memory_a_preview: 'Tailwind 4 requires explicit CSS import only',
			memory_b_preview: 'Tailwind 4 config still supports tailwind.config.js',
			memory_a_type: 'fact',
			memory_b_type: 'fact',
			memory_a_created: '2026-01-30',
			memory_b_created: '2026-02-14',
			memory_a_tags: ['tailwind', 'config'],
			memory_b_tags: ['tailwind', 'config'],
			trust_a: 0.47,
			trust_b: 0.33,
			similarity: 0.85,
			date_diff_days: 15,
			topic: 'Tailwind 4 config'
		},
		{
			memory_a_id: 'a11',
			memory_b_id: 'b11',
			memory_a_preview: 'Kaggle API silently ignores invalid modelDataSources slugs',
			memory_b_preview: 'Kaggle API throws an error when model slug is invalid',
			memory_a_type: 'fact',
			memory_b_type: 'concept',
			memory_a_created: '2026-04-07',
			memory_b_created: '2026-02-20',
			memory_a_tags: ['kaggle', 'bug-fix', 'api'],
			memory_b_tags: ['kaggle', 'api'],
			trust_a: 0.89,
			trust_b: 0.28,
			similarity: 0.91,
			date_diff_days: 46,
			topic: 'Kaggle API validation'
		},
		{
			memory_a_id: 'a12',
			memory_b_id: 'b12',
			memory_a_preview: 'USearch HNSW is 20x faster than FAISS for embeddings',
			memory_b_preview: 'FAISS IVF is the fastest vector index at scale',
			memory_a_type: 'fact',
			memory_b_type: 'concept',
			memory_a_created: '2026-02-01',
			memory_b_created: '2025-11-15',
			memory_a_tags: ['vectors', 'perf'],
			memory_b_tags: ['vectors', 'perf'],
			trust_a: 0.78,
			trust_b: 0.36,
			similarity: 0.69,
			date_diff_days: 78,
			topic: 'vector index perf'
		},
		{
			memory_a_id: 'a13',
			memory_b_id: 'b13',
			memory_a_preview: 'Orbit Wars leaderboard scores weight by top-10 consistency',
			memory_b_preview: 'Orbit Wars uses single-best-episode scoring',
			memory_a_type: 'fact',
			memory_b_type: 'fact',
			memory_a_created: '2026-04-18',
			memory_b_created: '2026-04-10',
			memory_a_tags: ['orbit-wars', 'scoring'],
			memory_b_tags: ['orbit-wars', 'scoring'],
			trust_a: 0.64,
			trust_b: 0.52,
			similarity: 0.82,
			date_diff_days: 8,
			topic: 'Orbit Wars scoring'
		},
		{
			memory_a_id: 'a14',
			memory_b_id: 'b14',
			memory_a_preview: 'Sam commits to morning posts 8am ET',
			memory_b_preview: 'Morning cadence moved to 9am ET after energy review',
			memory_a_type: 'decision',
			memory_b_type: 'decision',
			memory_a_created: '2026-03-01',
			memory_b_created: '2026-04-15',
			memory_a_tags: ['cadence', 'content'],
			memory_b_tags: ['cadence', 'content'],
			trust_a: 0.50,
			trust_b: 0.81,
			similarity: 0.58,
			date_diff_days: 45,
			topic: 'posting cadence'
		},
		{
			memory_a_id: 'a15',
			memory_b_id: 'b15',
			memory_a_preview: 'Dream cycle consolidates ~50 memories per run',
			memory_b_preview: 'Dream cycle replays closer to 120 memories in practice',
			memory_a_type: 'fact',
			memory_b_type: 'fact',
			memory_a_created: '2026-02-15',
			memory_b_created: '2026-04-08',
			memory_a_tags: ['vestige', 'dream'],
			memory_b_tags: ['vestige', 'dream'],
			trust_a: 0.44,
			trust_b: 0.79,
			similarity: 0.76,
			date_diff_days: 52,
			topic: 'dream cycle count'
		},
		{
			memory_a_id: 'a16',
			memory_b_id: 'b16',
			memory_a_preview: 'Never commit API keys to git; use .env files',
			memory_b_preview: 'Environment secrets should live in a 1Password vault',
			memory_a_type: 'pattern',
			memory_b_type: 'pattern',
			memory_a_created: '2025-10-11',
			memory_b_created: '2026-03-20',
			memory_a_tags: ['security', 'secrets'],
			memory_b_tags: ['security', 'secrets'],
			trust_a: 0.72,
			trust_b: 0.64,
			similarity: 0.48,
			date_diff_days: 160,
			topic: 'secret storage'
		}
	];

	// --- Filters ---
	type Filter = 'all' | 'recent' | 'high-trust' | 'topic';
	let filter = $state<Filter>('all');
	let topicFilter = $state<string>('');

	const uniqueTopics = $derived(
		Array.from(new Set(MOCK_CONTRADICTIONS.map((c) => c.topic))).sort()
	);

	const filtered = $derived.by<Contradiction[]>(() => {
		switch (filter) {
			case 'recent':
				// Within 7 days of "now" — use date_diff as a proxy by keeping pairs
				// where either memory was created within the last 7 days of our fixed
				// mock "today" (2026-04-20). Simple approach: keep pairs whose newest
				// created date is within 7 days of 2026-04-20.
				{
					const now = new Date('2026-04-20').getTime();
					const week = 7 * 24 * 60 * 60 * 1000;
					return MOCK_CONTRADICTIONS.filter((c) => {
						const aT = c.memory_a_created ? new Date(c.memory_a_created).getTime() : 0;
						const bT = c.memory_b_created ? new Date(c.memory_b_created).getTime() : 0;
						return now - Math.max(aT, bT) <= week;
					});
				}
			case 'high-trust':
				return MOCK_CONTRADICTIONS.filter(
					(c) => Math.min(c.trust_a, c.trust_b) > 0.6
				);
			case 'topic':
				return topicFilter
					? MOCK_CONTRADICTIONS.filter((c) => c.topic === topicFilter)
					: MOCK_CONTRADICTIONS;
			case 'all':
			default:
				return MOCK_CONTRADICTIONS;
		}
	});

	// --- Selection / focused pair ---
	let focusedPairIndex = $state<number | null>(null);

	function selectPair(i: number | null) {
		focusedPairIndex = i;
	}

	// --- Stats. `TOTAL_CONTRADICTIONS_DETECTED` stays illustrative so the tile
	// reads like a system-wide count once the backend ships; everything else
	// is derived from the pairs the page actually holds so the numbers are
	// self-consistent with what the user sees. ---
	const TOTAL_CONTRADICTIONS_DETECTED = 47;
	const totalMemoriesInvolved = $derived(uniqueMemoryCount(MOCK_CONTRADICTIONS));
	const avgTrustDelta = $derived(avgTrustDeltaFn(MOCK_CONTRADICTIONS));

	// Map filtered index -> original index in MOCK_CONTRADICTIONS so the
	// constellation and sidebar stay in sync regardless of which filter is on.
	const visibleList = $derived.by<{ orig: number; c: Contradiction }[]>(() => {
		const byId = new Map(MOCK_CONTRADICTIONS.map((c, i) => [c.memory_a_id + '|' + c.memory_b_id, i]));
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
	<header class="space-y-1">
		<h1 class="text-2xl text-bright font-semibold tracking-tight">
			Contradiction Constellation
		</h1>
		<p class="text-sm text-dim">Where your memory disagrees with itself</p>
	</header>

	<!-- Stats bar -->
	<div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
		<div class="p-4 glass rounded-xl">
			<div class="text-2xl text-bright font-bold">{TOTAL_CONTRADICTIONS_DETECTED}</div>
			<div class="text-xs text-dim mt-1">
				contradictions across {totalMemoriesInvolved.toLocaleString()} memories
			</div>
		</div>
		<div class="p-4 glass rounded-xl">
			<div class="text-2xl font-bold" style="color: #f59e0b">
				{avgTrustDelta.toFixed(2)}
			</div>
			<div class="text-xs text-dim mt-1">average trust delta</div>
		</div>
		<div class="p-4 glass rounded-xl">
			<div class="text-2xl text-bright font-bold">{filtered.length}</div>
			<div class="text-xs text-dim mt-1">visible in current filter</div>
		</div>
		<div class="p-4 glass rounded-xl">
			<div class="text-2xl font-bold" style="color: #ef4444">
				{filtered.filter((c) => c.similarity > 0.7).length}
			</div>
			<div class="text-xs text-dim mt-1">strong conflicts</div>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="flex flex-wrap gap-2 items-center">
		{#each [{ id: 'all', label: 'All' }, { id: 'recent', label: 'Recent (7d)' }, { id: 'high-trust', label: 'High trust (>60%)' }, { id: 'topic', label: 'By topic' }] as f (f.id)}
			<button
				onclick={() => {
					filter = f.id as Filter;
					focusedPairIndex = null;
				}}
				class="px-3 py-1.5 rounded-lg text-xs border transition
					{filter === f.id
						? 'bg-synapse/15 border-synapse/40 text-synapse-glow'
						: 'border-subtle/30 text-dim hover:text-text hover:bg-white/[0.03]'}"
			>
				{f.label}
			</button>
		{/each}
		{#if filter === 'topic'}
			<select
				bind:value={topicFilter}
				class="ml-2 px-3 py-1.5 rounded-lg text-xs glass-subtle border border-subtle/30 text-text"
			>
				<option value="">All topics</option>
				{#each uniqueTopics as t}
					<option value={t}>{t}</option>
				{/each}
			</select>
		{/if}
		{#if focusedPairIndex !== null}
			<button
				onclick={() => (focusedPairIndex = null)}
				class="ml-auto px-3 py-1.5 rounded-lg text-xs border border-subtle/30 text-dim hover:text-text"
			>
				Clear focus
			</button>
		{/if}
	</div>

	<!-- Main view: constellation + sidebar -->
	<div class="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4">
		<!-- Constellation -->
		<div class="glass-panel rounded-2xl p-3 min-h-[520px] relative">
			{#if filtered.length === 0}
				<div class="flex items-center justify-center h-full text-dim text-sm">
					No contradictions match this filter.
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
		<aside class="glass rounded-2xl p-3 space-y-2 max-h-[620px] overflow-y-auto">
			<div class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10">
				<span class="text-xs text-dim uppercase tracking-wider">Pairs</span>
				<span class="text-xs text-muted">{visibleList.length}</span>
			</div>

			{#if visibleList.length === 0}
				<div class="text-xs text-muted p-3">No pairs visible.</div>
			{/if}

			{#each visibleList as entry, localIndex (entry.c.memory_a_id + '|' + entry.c.memory_b_id)}
				{@const c = entry.c}
				{@const isFocused = focusedPairIndex === localIndex}
				<button
					onclick={() => sidebarClick(localIndex)}
					class="w-full text-left p-3 rounded-xl border transition
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
</div>

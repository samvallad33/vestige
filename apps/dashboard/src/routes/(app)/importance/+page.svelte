<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import { api } from '$stores/api';
	import type { ImportanceScore, Memory } from '$types';
	import { NODE_TYPE_COLORS } from '$types';
	import ImportanceRadar from '$components/ImportanceRadar.svelte';

	// ── Section 1: Test Importance ───────────────────────────────────────────
	let content = $state('');
	let score: ImportanceScore | null = $state(null);
	let scoring = $state(false);
	let scoreError: string | null = $state(null);

	// Keyed radar remount — we flip the key each time a new score lands so the
	// onMount grow-from-center animation re-fires instead of just mutating props.
	let radarKey = $state(0);

	async function scoreContent() {
		const trimmed = content.trim();
		if (!trimmed || scoring) return;
		scoring = true;
		scoreError = null;
		try {
			score = await api.importance(trimmed);
			radarKey++;
		} catch (e) {
			scoreError = e instanceof Error ? e.message : String(e);
			score = null;
		} finally {
			scoring = false;
		}
	}

	function onKeydown(e: KeyboardEvent) {
		// Cmd/Ctrl+Enter submits so the power-user flow isn't "click the button".
		if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
			e.preventDefault();
			scoreContent();
		}
	}

	// Which channel contributed the most to the composite? Drives the "why"
	// blurb under the recommendation. Uses the same weights ImportanceSignals
	// applies server-side (novelty 0.25 / arousal 0.30 / reward 0.25 / attention 0.20)
	// so the explanation lines up with the composite.
	const CHANNEL_WEIGHTS = { novelty: 0.25, arousal: 0.3, reward: 0.25, attention: 0.2 } as const;
	type ChannelKey = keyof typeof CHANNEL_WEIGHTS;

	const CHANNEL_BLURBS: Record<ChannelKey, { high: string; low: string }> = {
		novelty: {
			high: 'new information not already in your graph',
			low: 'overlaps heavily with what you already know'
		},
		arousal: {
			high: 'emotionally salient — decisions, bugs, or discoveries stick',
			low: 'neutral tone, no strong affect signal'
		},
		reward: {
			high: 'high reward value — preferences, wins, or solutions you will revisit',
			low: 'low reward value — transient or incidental detail'
		},
		attention: {
			high: 'strong attentional markers (imperatives, questions, urgency)',
			low: 'passive phrasing, no clear attentional hook'
		}
	};

	let topChannel = $derived.by<{ key: ChannelKey; contribution: number } | null>(() => {
		if (!score) return null;
		const ranked = (Object.keys(CHANNEL_WEIGHTS) as ChannelKey[])
			.map((k) => ({ key: k, contribution: score!.channels[k] * CHANNEL_WEIGHTS[k] }))
			.sort((a, b) => b.contribution - a.contribution);
		return ranked[0];
	});

	let weakestChannel = $derived.by<ChannelKey | null>(() => {
		if (!score) return null;
		return (Object.keys(CHANNEL_WEIGHTS) as ChannelKey[])
			.slice()
			.sort((a, b) => score!.channels[a] - score!.channels[b])[0];
	});

	// ── Section 2: Top Important Memories This Week ──────────────────────────
	// The Memory response does NOT include the per-memory importance channels,
	// so we approximate a "trending importance" proxy from the FSRS state we
	// DO have: retention strength × (1 + reviewCount) × recency-boost. Clients
	// who want the true composite would need the backend to include channels.
	// TODO: backend should include channels on Memory response directly
	let memories: Memory[] = $state([]);
	let loadingMemories = $state(true);
	// Per-memory radar channels, fetched lazily via api.importance(content).
	// Keyed by memory.id. Until populated, mini-radars render with zeroed props.
	let perMemoryScores: Record<string, ImportanceScore['channels']> = $state({});

	function importanceProxy(m: Memory): number {
		// retentionStrength × log(1 + reviewCount) / age_days.
		// Heavy short-term bias so the "this week" framing actually holds.
		const ageDays = Math.max(
			1,
			(Date.now() - new Date(m.createdAt).getTime()) / 86_400_000
		);
		const reviews = m.reviewCount ?? 0;
		const recencyBoost = 1 / Math.pow(ageDays, 0.5);
		return m.retentionStrength * Math.log1p(reviews + 1) * recencyBoost;
	}

	async function loadTrending() {
		loadingMemories = true;
		try {
			const res = await api.memories.list({ limit: '20' });
			// Sort client-side by our proxy, keep top 20.
			const ranked = res.memories
				.slice()
				.sort((a, b) => importanceProxy(b) - importanceProxy(a))
				.slice(0, 20);
			memories = ranked;
			// Lazily score each one so the mini-radars aren't all zeros. We fan
			// these out in parallel but don't await them before painting — the
			// list renders immediately and radars fill in as results arrive.
			memories.forEach(async (m) => {
				try {
					const s = await api.importance(m.content);
					perMemoryScores[m.id] = s.channels;
				} catch {
					// swallow — per-memory score is cosmetic, list still works
				}
			});
		} catch {
			memories = [];
		} finally {
			loadingMemories = false;
		}
	}

	onMount(loadTrending);

	function openMemory(id: string) {
		// The memories page doesn't support deep-linking to a specific memory
		// yet; navigate there and let the user scroll. base is '/dashboard'.
		goto(`${base}/memories`);
		void id;
	}
</script>

<div class="p-6 max-w-5xl mx-auto space-y-8">
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-xl text-bright font-semibold">Importance Radar</h1>
			<p class="text-sm text-dim mt-1">
				4-channel importance model: Novelty · Arousal · Reward · Attention
			</p>
		</div>
	</div>

	<!-- ── Section 1: Test Importance ─────────────────────────────────────── -->
	<section class="glass-panel rounded-2xl p-6 space-y-5">
		<div>
			<h2 class="text-sm font-semibold text-bright uppercase tracking-wider">Test Importance</h2>
			<p class="text-xs text-muted mt-1">
				Paste any content below. Vestige scores it across 4 channels and
				decides whether it is worth saving.
			</p>
		</div>

		<div class="grid md:grid-cols-[1fr_auto] gap-5 items-start">
			<div class="space-y-3">
				<textarea
					bind:value={content}
					onkeydown={onKeydown}
					placeholder="Type some content above to score its importance."
					class="w-full min-h-40 px-4 py-3 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
						placeholder:text-muted focus:outline-none focus:border-synapse/40 focus:ring-1 focus:ring-synapse/20
						transition backdrop-blur-sm resize-y font-mono"
				></textarea>
				<div class="flex items-center gap-3">
					<button
						onclick={scoreContent}
						disabled={scoring || !content.trim()}
						class="px-4 py-2 bg-synapse/20 text-synapse-glow text-sm rounded-xl border border-synapse/30
							hover:bg-synapse/30 hover:border-synapse/50 transition disabled:opacity-40 disabled:cursor-not-allowed"
					>
						{scoring ? 'Scoring…' : 'Score Importance'}
					</button>
					<span class="text-xs text-muted">⌘/Ctrl + Enter</span>
					{#if scoreError}
						<span class="text-xs text-decay">{scoreError}</span>
					{/if}
				</div>
			</div>

			<!-- Radar + composite readout -->
			<div class="flex flex-col items-center gap-4 md:min-w-[340px]">
				{#if score}
					<div class="text-center">
						<div class="text-[10px] uppercase tracking-widest text-muted">Composite</div>
						<div class="text-5xl font-semibold text-bright leading-none mt-1">
							{(score.composite * 100).toFixed(0)}<span class="text-xl text-dim">%</span>
						</div>
					</div>
					{#key radarKey}
						<ImportanceRadar
							novelty={score.channels.novelty}
							arousal={score.channels.arousal}
							reward={score.channels.reward}
							attention={score.channels.attention}
							size="lg"
						/>
					{/key}

					<!-- Recommendation -->
					{#if score.composite > 0.6}
						<div class="w-full text-center space-y-1">
							<div class="text-lg font-semibold text-recall">✓ Save</div>
							<p class="text-xs text-dim leading-relaxed">
								Composite {(score.composite * 100).toFixed(0)}% &gt; 60% threshold.
								{#if topChannel}
									Driven by <span class="text-bright">{topChannel.key}</span> — {CHANNEL_BLURBS[topChannel.key].high}.
								{/if}
							</p>
						</div>
					{:else}
						<div class="w-full text-center space-y-1">
							<div class="text-lg font-semibold text-decay">⨯ Skip</div>
							<p class="text-xs text-dim leading-relaxed">
								Composite {(score.composite * 100).toFixed(0)}% &lt; 60% threshold.
								{#if weakestChannel}
									Weakest channel: <span class="text-bright">{weakestChannel}</span> — {CHANNEL_BLURBS[weakestChannel].low}.
								{/if}
							</p>
						</div>
					{/if}
				{:else}
					<div class="flex flex-col items-center justify-center min-h-[320px] w-full text-center px-4">
						<div class="text-3xl text-muted mb-3">◫</div>
						<p class="text-sm text-dim">Type some content above to score its importance.</p>
						<p class="text-xs text-muted mt-2 max-w-xs">
							Composite = 0.25·novelty + 0.30·arousal + 0.25·reward + 0.20·attention.
							Threshold for save: 60%.
						</p>
					</div>
				{/if}
			</div>
		</div>
	</section>

	<!-- ── Section 2: Top Important Memories This Week ────────────────────── -->
	<section class="space-y-4">
		<div class="flex items-end justify-between">
			<div>
				<h2 class="text-sm font-semibold text-bright uppercase tracking-wider">
					Top Important Memories This Week
				</h2>
				<p class="text-xs text-muted mt-1">
					Ranked by retention × reviews ÷ age. Click any card to open it.
				</p>
			</div>
			<button
				onclick={loadTrending}
				class="text-xs text-muted hover:text-text transition"
			>
				Refresh
			</button>
		</div>

		{#if loadingMemories}
			<div class="grid gap-3 md:grid-cols-2">
				{#each Array(6) as _}
					<div class="h-28 glass-subtle rounded-xl animate-pulse"></div>
				{/each}
			</div>
		{:else if memories.length === 0}
			<div class="text-center py-12 text-dim">
				<p class="text-sm">No memories yet.</p>
			</div>
		{:else}
			<div class="grid gap-3 md:grid-cols-2">
				{#each memories as memory (memory.id)}
					{@const ch = perMemoryScores[memory.id]}
					<button
						type="button"
						onclick={() => openMemory(memory.id)}
						class="text-left p-4 glass-subtle rounded-xl hover:bg-white/[0.04] hover:border-synapse/30
							transition-all duration-200 flex items-start gap-4"
					>
						<div class="flex-1 min-w-0 space-y-2">
							<div class="flex items-center gap-2">
								<span
									class="w-2 h-2 rounded-full"
									style="background: {NODE_TYPE_COLORS[memory.nodeType] || '#8B95A5'}"
								></span>
								<span class="text-xs text-dim">{memory.nodeType}</span>
								<span class="text-xs text-muted">·</span>
								<span class="text-xs text-muted">
									{(memory.retentionStrength * 100).toFixed(0)}% retention
								</span>
								{#if memory.reviewCount}
									<span class="text-xs text-muted">·</span>
									<span class="text-xs text-muted">{memory.reviewCount} reviews</span>
								{/if}
							</div>
							<p class="text-sm text-text leading-relaxed line-clamp-3">
								{memory.content}
							</p>
							{#if memory.tags.length > 0}
								<div class="flex gap-1.5 flex-wrap">
									{#each memory.tags.slice(0, 4) as tag}
										<span class="text-[10px] px-1.5 py-0.5 bg-white/[0.04] rounded text-muted">
											{tag}
										</span>
									{/each}
								</div>
							{/if}
						</div>
						<div class="flex-shrink-0">
							<ImportanceRadar
								novelty={ch?.novelty ?? 0}
								arousal={ch?.arousal ?? 0}
								reward={ch?.reward ?? 0}
								attention={ch?.attention ?? 0}
								size="sm"
							/>
						</div>
					</button>
				{/each}
			</div>
		{/if}
	</section>
</div>

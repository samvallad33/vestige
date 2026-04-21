<!--
  Dream Cinema — scrubbable replay of Vestige's 5-stage dream consolidation.

  The /api/dream endpoint returns a DreamResult. We render the 5 phases of
  the MemoryDreamer pipeline (Replay → Cross-reference → Strengthen → Prune
  → Transfer) and a sorted insight list. Clicking "Dream Now" triggers a
  fresh dream; the scrubber then lets the user step through the stages.
-->
<script lang="ts">
	import { api } from '$stores/api';
	import type { DreamResult } from '$types';
	import DreamStageReplay from '$components/DreamStageReplay.svelte';
	import DreamInsightCard from '$components/DreamInsightCard.svelte';
	import {
		STAGE_NAMES,
		clampStage,
		formatDurationMs,
	} from '$components/dream-helpers';

	let dreamResult: DreamResult | null = $state(null);
	let stage = $state(1);
	let dreaming = $state(false);
	let error: string | null = $state(null);

	let hasDream = $derived(dreamResult !== null);

	let sortedInsights = $derived.by(() => {
		const r = dreamResult;
		if (!r) return [];
		return [...r.insights].sort((a, b) => (b.noveltyScore ?? 0) - (a.noveltyScore ?? 0));
	});

	async function runDream() {
		if (dreaming) return;
		dreaming = true;
		error = null;
		try {
			const result = await api.dream();
			dreamResult = result;
			stage = 1;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Dream failed';
		} finally {
			dreaming = false;
		}
	}

	function setStage(n: number) {
		stage = clampStage(n);
	}

	function onScrub(e: Event) {
		const v = Number((e.currentTarget as HTMLInputElement).value);
		setStage(v);
	}
</script>

<svelte:head>
	<title>Dream Cinema · Vestige</title>
</svelte:head>

<div class="p-6 max-w-7xl mx-auto space-y-6">
	<!-- Header -->
	<header class="flex items-start justify-between flex-wrap gap-4">
		<div>
			<h1 class="text-2xl text-bright font-semibold tracking-tight flex items-center gap-3">
				<span class="header-glyph">✦</span>
				Dream Cinema
			</h1>
			<p class="text-sm text-dim mt-1 max-w-xl leading-snug">
				Scrub through Vestige's 5-stage consolidation cycle. Replay, cross-reference,
				strengthen, prune, transfer. Watch episodic become semantic.
			</p>
		</div>

		<button
			type="button"
			onclick={runDream}
			disabled={dreaming}
			class="dream-button"
			class:is-dreaming={dreaming}
		>
			{#if dreaming}
				<span class="spinner" aria-hidden="true"></span>
				<span>Dreaming...</span>
			{:else}
				<span class="dream-icon" aria-hidden="true">✦</span>
				<span>Dream Now</span>
			{/if}
		</button>
	</header>

	{#if error}
		<div class="glass-subtle rounded-xl px-4 py-3 text-sm border !border-decay/40 text-decay">
			{error}
		</div>
	{/if}

	{#if !hasDream && !dreaming}
		<!-- Empty state -->
		<div class="empty-state glass-panel rounded-2xl p-12 text-center space-y-3">
			<div class="empty-glyph">✦</div>
			<p class="text-bright font-semibold">No dream yet.</p>
			<p class="text-dim text-sm">Click Dream Now to begin.</p>
		</div>
	{:else}
		<!-- Scrubber + stage markers -->
		<section class="glass-panel rounded-2xl p-5 space-y-4">
			<div class="flex items-center justify-between gap-2 flex-wrap">
				<div class="text-[11px] text-dream-glow uppercase tracking-[0.18em] font-semibold">
					Stage {stage} · {STAGE_NAMES[stage - 1]}
				</div>
				<div class="flex gap-1 text-[11px] text-dim">
					<button
						type="button"
						class="step-btn"
						onclick={() => setStage(stage - 1)}
						disabled={stage <= 1 || dreaming}
						aria-label="Previous stage">◀</button>
					<button
						type="button"
						class="step-btn"
						onclick={() => setStage(stage + 1)}
						disabled={stage >= 5 || dreaming}
						aria-label="Next stage">▶</button>
				</div>
			</div>

			<!-- Scrubber -->
			<div class="scrubber-wrap">
				<input
					type="range"
					min="1"
					max="5"
					step="1"
					value={stage}
					oninput={onScrub}
					disabled={dreaming}
					class="scrubber"
					aria-label="Dream stage scrubber"
				/>
				<div class="scrubber-ticks">
					{#each STAGE_NAMES as name, i (name)}
						<button
							type="button"
							class="tick"
							class:active={stage === i + 1}
							class:passed={stage > i + 1}
							onclick={() => setStage(i + 1)}
							disabled={dreaming}
						>
							<span class="tick-dot"></span>
							<span class="tick-label">{i + 1}. {name}</span>
						</button>
					{/each}
				</div>
			</div>
		</section>

		<!-- Main grid: stage replay + insights -->
		<section class="grid gap-6 lg:grid-cols-[1fr_360px]">
			<!-- Stage replay -->
			<DreamStageReplay {stage} {dreamResult} />

			<!-- Insights panel -->
			<aside class="glass-panel rounded-2xl p-4 space-y-3 min-h-[240px]">
				<div class="flex items-center justify-between">
					<h2 class="text-sm font-semibold text-bright">Insights</h2>
					<span class="text-[10px] text-dim uppercase tracking-wider">
						{sortedInsights.length} total · by novelty
					</span>
				</div>

				<div class="insights-scroll space-y-3">
					{#if sortedInsights.length === 0}
						<div class="text-center py-8 text-dim text-sm">
							{#if dreaming}
								Dreaming...
							{:else}
								No insights generated this cycle.
							{/if}
						</div>
					{:else}
						{#each sortedInsights as insight, i (i + '-' + (insight.insight?.slice(0, 32) ?? ''))}
							<DreamInsightCard {insight} index={i} />
						{/each}
					{/if}
				</div>
			</aside>
		</section>

		<!-- Stats footer -->
		{#if dreamResult}
			<footer class="glass-subtle rounded-2xl p-4 grid gap-3 grid-cols-2 md:grid-cols-5">
				<div class="stat-cell">
					<div class="stat-value">{dreamResult.memoriesReplayed ?? 0}</div>
					<div class="stat-label">Replayed</div>
				</div>
				<div class="stat-cell">
					<div class="stat-value">{dreamResult.stats?.newConnectionsFound ?? 0}</div>
					<div class="stat-label">Connections Found</div>
				</div>
				<div class="stat-cell">
					<div class="stat-value">{dreamResult.connectionsPersisted ?? 0}</div>
					<div class="stat-label">Connections Persisted</div>
				</div>
				<div class="stat-cell">
					<div class="stat-value">{dreamResult.stats?.insightsGenerated ?? 0}</div>
					<div class="stat-label">Insights</div>
				</div>
				<div class="stat-cell">
					<div class="stat-value">{formatDurationMs(dreamResult.stats?.durationMs)}</div>
					<div class="stat-label">Duration</div>
				</div>
			</footer>
		{/if}
	{/if}
</div>

<style>
	.header-glyph {
		display: inline-block;
		color: var(--color-dream-glow);
		text-shadow:
			0 0 12px var(--color-dream),
			0 0 24px color-mix(in srgb, var(--color-dream) 50%, transparent);
		animation: twinkle 4s ease-in-out infinite;
	}

	@keyframes twinkle {
		0%, 100% { opacity: 1; transform: rotate(0deg); }
		50% { opacity: 0.75; transform: rotate(10deg); }
	}

	.dream-button {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
		padding: 0.7rem 1.4rem;
		border-radius: 999px;
		font-size: 0.9rem;
		font-weight: 600;
		letter-spacing: 0.02em;
		color: white;
		background: linear-gradient(135deg, var(--color-dream), var(--color-synapse));
		border: 1px solid color-mix(in srgb, var(--color-dream-glow) 60%, transparent);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.18),
			0 8px 24px -6px rgba(168, 85, 247, 0.55),
			0 0 48px -10px rgba(168, 85, 247, 0.45);
		cursor: pointer;
		transition: transform 400ms cubic-bezier(0.34, 1.56, 0.64, 1),
			box-shadow 220ms ease, filter 220ms ease;
	}

	.dream-button:hover:not(:disabled) {
		transform: translateY(-2px) scale(1.03);
		box-shadow:
			inset 0 1px 0 0 rgba(255, 255, 255, 0.22),
			0 12px 32px -6px rgba(168, 85, 247, 0.7),
			0 0 64px -10px rgba(168, 85, 247, 0.55);
	}

	.dream-button:disabled {
		cursor: not-allowed;
		filter: saturate(0.85);
	}

	.dream-button.is-dreaming {
		background: linear-gradient(135deg, var(--color-synapse), var(--color-dream));
		animation: button-breathe 2s ease-in-out infinite;
	}

	@keyframes button-breathe {
		0%, 100% { box-shadow: 0 8px 24px -6px rgba(168, 85, 247, 0.5), 0 0 48px -10px rgba(168, 85, 247, 0.4); }
		50% { box-shadow: 0 12px 36px -6px rgba(168, 85, 247, 0.8), 0 0 80px -10px rgba(168, 85, 247, 0.6); }
	}

	.dream-icon {
		display: inline-block;
		animation: twinkle 3s ease-in-out infinite;
	}

	.spinner {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		border: 2px solid rgba(255, 255, 255, 0.25);
		border-top-color: white;
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.empty-state {
		border: 1px dashed rgba(168, 85, 247, 0.25);
	}

	.empty-glyph {
		font-size: 3rem;
		color: var(--color-dream-glow);
		opacity: 0.5;
		text-shadow: 0 0 20px var(--color-dream);
		animation: twinkle 4s ease-in-out infinite;
	}

	/* Scrubber */
	.scrubber-wrap {
		position: relative;
		padding: 4px 0 8px;
	}

	.scrubber {
		appearance: none;
		-webkit-appearance: none;
		width: 100%;
		height: 6px;
		border-radius: 999px;
		background: linear-gradient(
			90deg,
			var(--color-synapse-glow) 0%,
			var(--color-dream) 50%,
			var(--color-recall) 100%
		);
		opacity: 0.35;
		outline: none;
		cursor: pointer;
		transition: opacity 220ms ease;
	}

	.scrubber:hover:not(:disabled) {
		opacity: 0.55;
	}

	.scrubber::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 20px;
		height: 20px;
		border-radius: 50%;
		background: var(--color-dream-glow);
		border: 2px solid white;
		box-shadow:
			0 0 0 3px rgba(192, 132, 252, 0.25),
			0 0 20px var(--color-dream),
			0 4px 12px rgba(0, 0, 0, 0.4);
		cursor: grab;
		transition: transform 400ms cubic-bezier(0.34, 1.56, 0.64, 1);
	}

	.scrubber::-webkit-slider-thumb:hover {
		transform: scale(1.2);
	}

	.scrubber::-moz-range-thumb {
		width: 20px;
		height: 20px;
		border-radius: 50%;
		background: var(--color-dream-glow);
		border: 2px solid white;
		box-shadow:
			0 0 0 3px rgba(192, 132, 252, 0.25),
			0 0 20px var(--color-dream);
		cursor: grab;
	}

	.scrubber:disabled {
		cursor: not-allowed;
		opacity: 0.25;
	}

	.scrubber-ticks {
		display: flex;
		justify-content: space-between;
		margin-top: 10px;
		gap: 4px;
	}

	.tick {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
		background: transparent;
		border: none;
		cursor: pointer;
		padding: 2px 4px;
		color: var(--color-dim);
		font-size: 10px;
		letter-spacing: 0.04em;
		transition: color 220ms ease, transform 220ms cubic-bezier(0.34, 1.56, 0.64, 1);
	}

	.tick:disabled {
		cursor: not-allowed;
	}

	.tick:hover:not(:disabled) {
		color: var(--color-dream-glow);
		transform: translateY(-1px);
	}

	.tick-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid rgba(255, 255, 255, 0.15);
		transition: all 280ms ease;
	}

	.tick.passed .tick-dot {
		background: var(--color-synapse-glow);
		border-color: var(--color-synapse-glow);
		opacity: 0.7;
	}

	.tick.active .tick-dot {
		background: var(--color-dream-glow);
		border-color: white;
		box-shadow:
			0 0 0 3px rgba(192, 132, 252, 0.3),
			0 0 14px var(--color-dream);
		transform: scale(1.3);
	}

	.tick.active {
		color: var(--color-dream-glow);
		font-weight: 600;
	}

	.tick-label {
		white-space: nowrap;
	}

	.step-btn {
		width: 28px;
		height: 28px;
		border-radius: 6px;
		background: rgba(99, 102, 241, 0.1);
		border: 1px solid rgba(99, 102, 241, 0.2);
		color: var(--color-synapse-glow);
		cursor: pointer;
		transition: all 180ms ease;
		font-size: 11px;
	}

	.step-btn:hover:not(:disabled) {
		background: rgba(99, 102, 241, 0.2);
		transform: translateY(-1px);
	}

	.step-btn:disabled {
		opacity: 0.35;
		cursor: not-allowed;
	}

	/* Insights */
	.insights-scroll {
		max-height: 520px;
		overflow-y: auto;
		padding-right: 4px;
	}

	/* Stat cells */
	.stat-cell {
		padding: 0.5rem 0.75rem;
		border-left: 2px solid rgba(168, 85, 247, 0.3);
	}

	.stat-value {
		font-family: var(--font-mono);
		font-size: 1.25rem;
		font-weight: 700;
		color: var(--color-bright);
		font-variant-numeric: tabular-nums;
		line-height: 1.1;
	}

	.stat-label {
		font-size: 10px;
		color: var(--color-dim);
		text-transform: uppercase;
		letter-spacing: 0.1em;
		margin-top: 2px;
	}
</style>

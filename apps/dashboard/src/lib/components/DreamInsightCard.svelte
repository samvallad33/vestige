<!--
  DreamInsightCard — single insight from a dream cycle.
  High-novelty insights (>0.7) get a golden glow. Low-novelty (<0.3) muted.
  Source memory IDs are clickable → navigate to /memories/[id].
-->
<script lang="ts">
	import { base } from '$app/paths';
	import type { DreamInsight } from '$types';
	import {
		clamp01,
		noveltyBand,
		formatConfidencePct,
		firstSourceIds,
		extraSourceCount,
		sourceMemoryHref,
		shortMemoryId,
	} from './dream-helpers';

	interface Props {
		insight: DreamInsight;
		index?: number;
	}

	let { insight, index = 0 }: Props = $props();

	let novelty = $derived(clamp01(insight.noveltyScore));
	let confidence = $derived(clamp01(insight.confidence));
	let band = $derived(noveltyBand(insight.noveltyScore));
	let isHighNovelty = $derived(band === 'high');
	let isLowNovelty = $derived(band === 'low');
	let firstSources = $derived(firstSourceIds(insight.sourceMemories, 2));
	let extraCount = $derived(extraSourceCount(insight.sourceMemories, 2));

	const TYPE_COLORS: Record<string, string> = {
		connection: '#818cf8',
		pattern: '#ec4899',
		contradiction: '#ef4444',
		synthesis: '#c084fc',
		emergence: '#f59e0b',
		cluster: '#06b6d4'
	};

	let typeColor = $derived(TYPE_COLORS[insight.type?.toLowerCase() ?? ''] ?? '#a855f7');
</script>

<article
	class="insight-card glass-panel rounded-xl p-4 space-y-3"
	class:high-novelty={isHighNovelty}
	class:low-novelty={isLowNovelty}
	style="--insight-color: {typeColor}; --enter-delay: {index * 60}ms"
>
	<!-- Type badge + novelty halo -->
	<div class="flex items-center justify-between gap-2">
		<span
			class="text-[10px] uppercase tracking-[0.12em] font-semibold px-2 py-0.5 rounded-full"
			style="background: {typeColor}22; color: {typeColor}; border: 1px solid {typeColor}55"
		>
			{insight.type ?? 'insight'}
		</span>
		{#if isHighNovelty}
			<span class="text-[10px] text-warning font-semibold flex items-center gap-1">
				<span class="sparkle">✦</span> novel
			</span>
		{/if}
	</div>

	<!-- Insight text -->
	<p class="text-sm text-bright font-semibold leading-snug">
		{insight.insight}
	</p>

	<!-- Novelty bar -->
	<div class="space-y-1">
		<div class="flex items-center justify-between text-[10px] text-dim uppercase tracking-wider">
			<span>Novelty</span>
			<span class="tabular-nums text-text/80">{novelty.toFixed(2)}</span>
		</div>
		<div class="novelty-track">
			<div
				class="novelty-fill"
				style="width: {novelty * 100}%; background: linear-gradient(90deg, {typeColor}, var(--color-dream-glow))"
			></div>
		</div>
	</div>

	<!-- Confidence -->
	<div class="flex items-center justify-between text-[11px]">
		<span class="text-dim">Confidence</span>
		<span
			class="tabular-nums font-semibold"
			style="color: {confidence > 0.7 ? '#10b981' : confidence > 0.4 ? '#f59e0b' : '#ef4444'}"
		>
			{formatConfidencePct(confidence)}
		</span>
	</div>

	<!-- Source memories -->
	{#if firstSources.length > 0}
		<div class="pt-2 border-t border-white/5 space-y-1.5">
			<div class="text-[10px] text-dim uppercase tracking-wider">
				Sources
				{#if extraCount > 0}
					<span class="text-muted">(+{extraCount})</span>
				{/if}
			</div>
			<div class="flex flex-wrap gap-1.5">
				{#each firstSources as id (id)}
					<a
						href={sourceMemoryHref(id, base)}
						class="source-chip font-mono text-[10px] px-2 py-0.5 rounded"
						title="Open memory {id}"
					>
						{shortMemoryId(id)}
					</a>
				{/each}
			</div>
		</div>
	{/if}
</article>

<style>
	.insight-card {
		position: relative;
		border: 1px solid color-mix(in srgb, var(--insight-color) 20%, transparent);
		transition: transform 400ms cubic-bezier(0.34, 1.56, 0.64, 1),
			border-color 220ms ease, box-shadow 220ms ease;
		animation: card-in 420ms cubic-bezier(0.34, 1.56, 0.64, 1) both;
		animation-delay: var(--enter-delay, 0ms);
	}

	.insight-card:hover {
		transform: translateY(-2px) scale(1.01);
		border-color: color-mix(in srgb, var(--insight-color) 45%, transparent);
	}

	.insight-card.high-novelty {
		border-color: rgba(245, 158, 11, 0.4);
		box-shadow:
			0 0 0 1px rgba(245, 158, 11, 0.25),
			0 0 24px -4px rgba(245, 158, 11, 0.45),
			0 0 60px -12px rgba(245, 158, 11, 0.25),
			inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
		background:
			radial-gradient(at top right, rgba(245, 158, 11, 0.08), transparent 50%),
			rgba(10, 10, 26, 0.8);
	}

	.insight-card.low-novelty {
		opacity: 0.6;
		filter: saturate(0.7);
	}

	.insight-card.low-novelty:hover {
		opacity: 0.9;
		filter: saturate(1);
	}

	.novelty-track {
		height: 4px;
		background: rgba(255, 255, 255, 0.05);
		border-radius: 2px;
		overflow: hidden;
	}

	.novelty-fill {
		height: 100%;
		border-radius: 2px;
		transition: width 600ms cubic-bezier(0.34, 1.56, 0.64, 1);
		box-shadow: 0 0 8px color-mix(in srgb, var(--insight-color) 60%, transparent);
	}

	.source-chip {
		background: rgba(99, 102, 241, 0.12);
		border: 1px solid rgba(99, 102, 241, 0.25);
		color: var(--color-synapse-glow);
		text-decoration: none;
		transition: all 180ms ease;
	}

	.source-chip:hover {
		background: rgba(99, 102, 241, 0.25);
		border-color: rgba(129, 140, 248, 0.5);
		transform: translateY(-1px);
	}

	.sparkle {
		display: inline-block;
		animation: sparkle-spin 3s linear infinite;
	}

	@keyframes sparkle-spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}

	@keyframes card-in {
		from {
			opacity: 0;
			transform: translateY(8px) scale(0.97);
		}
		to {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.insight-card { animation: none; }
		.sparkle { animation: none; }
	}
</style>

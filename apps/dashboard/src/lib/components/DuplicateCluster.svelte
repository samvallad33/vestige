<!--
  DuplicateCluster — renders a single cosine-similarity cluster from the
  `find_duplicates` MCP tool. Shows similarity bar (color-coded by severity),
  stacked memory cards with type/retention/tags/date, and action controls
  (Merge all → highest-retention winner, Review → expand, Dismiss → hide).

  Pure helpers live in `./duplicates-helpers.ts` and are unit-tested there.
  Keep this file focused on rendering + glue.
-->
<script lang="ts">
	import { NODE_TYPE_COLORS } from '$types';
	import {
		similarityBandColor,
		similarityBandLabel,
		retentionColor,
		pickWinner,
		previewContent,
		formatDate,
		safeTags,
	} from './duplicates-helpers';

	interface ClusterMemory {
		id: string;
		content: string;
		nodeType: string;
		tags: string[];
		retention: number;
		createdAt: string;
	}

	interface Props {
		similarity: number;
		memories: ClusterMemory[];
		suggestedAction: 'merge' | 'review';
		onDismiss?: () => void;
		onMerge?: (winnerId: string, loserIds: string[]) => void;
	}

	let { similarity, memories, suggestedAction, onDismiss, onMerge }: Props = $props();

	let expanded = $state(false);

	// Winner = highest retention; others get merged into it. Stable tie-break
	// (first-wins). pickWinner returns null for empty input — render-guarded below.
	const winner = $derived(pickWinner(memories));
	const losers = $derived(
		winner ? memories.filter((m) => m.id !== winner.id).map((m) => m.id) : []
	);

	function handleMerge() {
		if (onMerge && winner) onMerge(winner.id, losers);
	}
</script>

{#if memories.length > 0 && winner}
	<div
		class="glass-panel rounded-2xl p-5 space-y-4 transition-all duration-300 hover:border-synapse/20"
	>
		<!-- Header row: similarity bar + suggested action badge -->
		<div class="flex items-start justify-between gap-4">
			<div class="flex-1 min-w-0 space-y-1.5">
				<div class="flex items-center gap-3">
					<span
						class="text-sm font-semibold"
						style="color: {similarityBandColor(similarity)}"
					>
						{(similarity * 100).toFixed(1)}%
					</span>
					<span class="text-xs text-dim">{similarityBandLabel(similarity)}</span>
					<span class="text-xs text-muted">· {memories.length} memories</span>
				</div>
				<div
					class="h-2 w-full overflow-hidden rounded-full bg-deep/60"
					role="progressbar"
					aria-label="Cosine similarity"
					aria-valuenow={Math.round(similarity * 100)}
					aria-valuemin="0"
					aria-valuemax="100"
				>
					<div
						class="h-full rounded-full transition-all duration-500"
						style="width: {(similarity * 100).toFixed(1)}%; background: {similarityBandColor(
							similarity
						)}; box-shadow: 0 0 12px {similarityBandColor(similarity)}66"
					></div>
				</div>
			</div>

			<!-- Suggested action badge — dream (review) or recall (merge) -->
			<span
				class="flex-shrink-0 rounded-full border px-3 py-1 text-xs font-medium {suggestedAction ===
				'merge'
					? 'border-recall/40 bg-recall/10 text-recall'
					: 'border-dream-glow/40 bg-dream/10 text-dream-glow'}"
			>
				Suggested: {suggestedAction === 'merge' ? 'Merge' : 'Review'}
			</span>
		</div>

		<!-- Stacked memory cards -->
		<div class="space-y-2">
			{#each memories as memory (memory.id)}
				<div
					class="group flex items-start gap-3 rounded-xl border border-synapse/5 bg-white/[0.02] p-3 transition-all duration-200 hover:border-synapse/20 hover:bg-white/[0.04] {memory.id ===
					winner.id
						? 'ring-1 ring-recall/30'
						: ''}"
				>
					<!-- Type dot -->
					<span
						class="mt-1.5 h-2 w-2 flex-shrink-0 rounded-full"
						style="background: {NODE_TYPE_COLORS[memory.nodeType] || '#8B95A5'}"
						title={memory.nodeType}
					></span>

					<div class="flex-1 min-w-0 space-y-1.5">
						<!-- Type + tags + winner flag -->
						<div class="flex flex-wrap items-center gap-1.5">
							<span class="text-xs text-dim">{memory.nodeType}</span>
							{#if memory.id === winner.id}
								<span class="rounded bg-recall/15 px-1.5 py-0.5 text-[10px] font-medium text-recall">
									WINNER
								</span>
							{/if}
							{#each safeTags(memory.tags, 4) as tag}
								<span class="rounded bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-muted"
									>{tag}</span
								>
							{/each}
						</div>

						<!-- Content preview (or full content if expanded) -->
						<p class="text-sm text-text leading-relaxed {expanded ? 'whitespace-pre-wrap' : ''}">
							{expanded ? memory.content : previewContent(memory.content)}
						</p>

						<!-- Date (empty string for invalid/missing — no "Invalid Date") -->
						{#if formatDate(memory.createdAt)}
							<div class="text-[11px] text-muted">
								{formatDate(memory.createdAt)}
							</div>
						{/if}
					</div>

					<!-- Retention bar + percent (right rail) -->
					<div class="flex flex-shrink-0 flex-col items-end gap-1">
						<div class="h-1.5 w-12 overflow-hidden rounded-full bg-deep">
							<div
								class="h-full rounded-full"
								style="width: {memory.retention * 100}%; background: {retentionColor(
									memory.retention
								)}"
							></div>
						</div>
						<span class="text-[11px] text-muted">
							{(memory.retention * 100).toFixed(0)}%
						</span>
					</div>
				</div>
			{/each}
		</div>

		<!-- Actions — native <button> elements, fully keyboard-accessible. -->
		<div class="flex flex-wrap items-center gap-2 pt-1">
			<button
				type="button"
				onclick={handleMerge}
				aria-label="Merge all memories into the highest-retention winner"
				class="rounded-lg bg-recall/20 px-3 py-1.5 text-xs font-medium text-recall transition hover:bg-recall/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60"
				title="Merge all into highest-retention memory ({(winner.retention * 100).toFixed(0)}%)"
			>
				Merge all → winner
			</button>
			<button
				type="button"
				onclick={() => (expanded = !expanded)}
				aria-expanded={expanded}
				class="rounded-lg bg-dream/20 px-3 py-1.5 text-xs font-medium text-dream-glow transition hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream-glow/60"
			>
				{expanded ? 'Collapse' : 'Review'}
			</button>
			<button
				type="button"
				onclick={onDismiss}
				aria-label="Dismiss cluster for this session"
				class="ml-auto rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Dismiss cluster
			</button>
		</div>
	</div>
{/if}

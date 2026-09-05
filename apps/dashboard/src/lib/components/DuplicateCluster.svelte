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
		mergePlanSummary,
	} from './duplicates-helpers';
	import type { DuplicateMergePlan, DuplicateMergeResult } from '$types';

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
		/**
		 * Oversized similarity component (similarity-chained, e.g. A~B~C~D where A is
		 * NOT similar to D). Quarantined: capped display, merge explicitly unsafe.
		 */
		oversized?: boolean;
		onDismiss?: () => void;
		/** Preview a reversible merge of these member ids. Nothing is written. */
		onPlan?: (memberIds: string[]) => Promise<DuplicateMergePlan>;
		/** Execute a previewed plan; returns the operation id `dedup undo` reverses. */
		onApply?: (planId: string) => Promise<DuplicateMergeResult>;
		/** Called after a successful apply so the page can drop the cluster and refetch. */
		onMerged?: (result: DuplicateMergeResult) => void;
	}
	let {
		similarity,
		memories,
		suggestedAction,
		oversized = false,
		onDismiss,
		onPlan,
		onApply,
		onMerged,
	}: Props = $props();

	// Merge is a two-step flow: plan (read-only preview) then apply (explicit).
	// Every state here is visible in the UI; there is no optimistic path.
	let plan: DuplicateMergePlan | null = $state(null);
	let planning = $state(false);
	let applying = $state(false);
	let mergeError: string | null = $state(null);
	let merged: DuplicateMergeResult | null = $state(null);
	const canMerge = $derived(!oversized && !!onPlan && !!onApply && !merged);

	async function previewMerge() {
		if (!onPlan || planning) return;
		planning = true;
		mergeError = null;
		try {
			plan = await onPlan(memories.map((m) => m.id));
		} catch (e) {
			mergeError = e instanceof Error ? e.message : 'Could not plan the merge';
		} finally {
			planning = false;
		}
	}

	async function applyMerge() {
		if (!onApply || !plan || applying) return;
		applying = true;
		mergeError = null;
		try {
			merged = await onApply(plan.planId);
			onMerged?.(merged);
		} catch (e) {
			mergeError = e instanceof Error ? e.message : 'Could not apply the merge';
		} finally {
			applying = false;
		}
	}

	let expanded = $state(false);

	// Cap the rendered member cards so an oversized component (428 members on the
	// real brain) can't produce a 44,000px card that strands everything below it.
	// Winner + metadata are ALWAYS computed from the FULL member array — display is
	// capped, cluster semantics are not.
	const DISPLAY_CAP = 12;

	// Winner = highest retention. pickWinner runs on the FULL array (never the
	// display subset) so capping cannot redefine who the winner is.
	const winner = $derived(pickWinner(memories));
	const displayMemories = $derived.by(() => {
		if (memories.length <= DISPLAY_CAP) return memories;
		// Winner-first so the capped view always shows the anchor of the cluster.
		const rest = memories.filter((m) => m.id !== winner?.id);
		return winner ? [winner, ...rest.slice(0, DISPLAY_CAP - 1)] : rest.slice(0, DISPLAY_CAP);
	});
	const hiddenCount = $derived(memories.length - displayMemories.length);
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

			<!-- Suggested action badge. Oversized components are quarantined: they are
			     similarity CHAINS (A~B~C~D), not mutually-similar sets — never mergeable. -->
			{#if oversized}
				<span
					class="flex-shrink-0 rounded-full border border-warning/50 bg-warning/10 px-3 py-1 text-xs font-medium text-warning"
				>
					REVIEW REQUIRED · NOT SAFE TO MERGE
				</span>
			{:else}
				<!-- Analysis classification, NOT an actionable suggestion — merge is
				     globally unavailable until the backend ships, so "Suggested: Merge"
				     beside a disabled merge button would contradict itself. -->
				<span
					class="flex-shrink-0 rounded-full border px-3 py-1 text-xs font-medium {suggestedAction ===
					'merge'
						? 'border-recall/40 bg-recall/10 text-recall'
						: 'border-dream-glow/40 bg-dream/10 text-dream-glow'}"
				>
					Classification: {suggestedAction === 'merge' ? 'merge candidate' : 'review'}
				</span>
			{/if}
		</div>

		<!-- Stacked memory cards (display-capped; winner always shown) -->
		<div class="space-y-2">
			{#each displayMemories as memory (memory.id)}
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
			{#if hiddenCount > 0}
				<div
					class="rounded-xl border border-warning/20 bg-warning/5 p-3 text-xs text-dim"
				>
					+{hiddenCount} linked candidates — oversized similarity component. Members
					chain through pairwise similarity; distant members may be unrelated. Raise
					the threshold to split it.
				</div>
			{/if}
		</div>

		<!-- Merge preview. Rendered only after the backend returned a plan, so
		     every number here came from the dedup tool, not from this component. -->
		{#if plan && !merged}
			<div class="rounded-xl border border-synapse/25 bg-synapse/5 p-3 text-xs">
				<div class="font-mono text-[11px] uppercase tracking-[0.18em] text-synapse-glow">
					Merge preview · nothing written yet
				</div>
				<div class="mt-1 text-text">{mergePlanSummary(plan)}</div>
				<div class="mt-1 text-muted">{plan.explanation}</div>
				<div class="mt-2 max-h-24 overflow-hidden text-muted">
					Result: {previewContent(plan.diff.resultContent, 240)}
				</div>
				<div class="mt-3 flex flex-wrap items-center gap-2">
					<button
						type="button"
						onclick={applyMerge}
						disabled={applying}
						class="rounded-lg bg-synapse/25 px-3 py-1.5 text-xs font-medium text-synapse-glow transition hover:bg-synapse/35 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
					>
						{applying ? 'Applying…' : 'Apply merge'}
					</button>
					<button
						type="button"
						onclick={() => (plan = null)}
						disabled={applying}
						class="rounded-lg bg-white/[0.04] px-3 py-1.5 text-xs text-dim transition hover:bg-white/[0.08] hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
					>
						Cancel
					</button>
				</div>
			</div>
		{/if}
		{#if merged}
			<div class="rounded-xl border border-consolidated/25 bg-consolidated/5 p-3 text-xs text-text">
				Merged into {merged.survivorId.slice(0, 8)}. Reversible: run dedup undo with
				operation id <span class="font-mono">{merged.operationId}</span>.
			</div>
		{/if}
		{#if mergeError}
			<div class="rounded-xl border border-decay/25 bg-decay/5 p-3 text-xs text-decay" role="alert">
				{mergeError}
			</div>
		{/if}

		<!-- Actions: native <button> elements, fully keyboard-accessible. Merge
		     is a plan-then-apply flow against POST /api/duplicates/plan and
		     /api/duplicates/apply; oversized components stay unmergeable because
		     they chain through pairwise similarity. -->
		<div class="flex flex-wrap items-center gap-2 pt-1">
			<button
				type="button"
				onclick={previewMerge}
				disabled={!canMerge || planning || !!plan}
				aria-disabled={!canMerge}
				aria-label={oversized ? 'Merge is not safe for an oversized component' : 'Preview a reversible merge'}
				class={canMerge
					? 'rounded-lg bg-synapse/20 px-3 py-1.5 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60'
					: 'cursor-not-allowed rounded-lg bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-muted/60'}
				title={oversized
					? 'Oversized similarity component: members chain through pairwise similarity, so a merge could fold unrelated memories together'
					: 'Preview first; nothing is written until you apply'}
			>
				{oversized ? 'Merge unsafe here' : planning ? 'Planning…' : 'Preview merge'}
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

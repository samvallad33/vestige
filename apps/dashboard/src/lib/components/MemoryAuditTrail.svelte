<script lang="ts">
	import { onMount } from 'svelte';
	import {
		META,
		generateMockAuditTrail,
		relativeTime,
		formatRetentionDelta,
		splitVisible,
		type AuditEvent
	} from './audit-trail-helpers';

	/**
	 * MemoryAuditTrail — per-memory "Sources" panel.
	 *
	 * Renders a vertical bioluminescent timeline of every event that has touched
	 * a memory: creation, accesses, promotions/demotions, edits, suppressions,
	 * dream-cycle activations, and reconsolidations (5-min labile window edits).
	 *
	 * Backend `/api/changelog?memory_id=X` does NOT yet exist. A typed mock
	 * fetcher lives in `audit-trail-helpers.ts` and will be swapped for
	 * `api.memoryChangelog(id)` once the backend route ships.
	 */

	interface Props {
		memoryId: string;
	}

	let { memoryId }: Props = $props();

	let events: AuditEvent[] = $state([]);
	let loading = $state(true);
	let errored = $state(false);
	let showAllOlder = $state(false);

	// TODO: swap for api.memoryChangelog(id) when backend ships
	async function fetchAuditTrail(id: string): Promise<AuditEvent[]> {
		return generateMockAuditTrail(id, Date.now());
	}

	onMount(async () => {
		if (!memoryId) {
			events = [];
			loading = false;
			return;
		}
		try {
			events = await fetchAuditTrail(memoryId);
		} catch {
			events = [];
			errored = true;
		} finally {
			loading = false;
		}
	});

	const split = $derived(splitVisible(events, showAllOlder));
	const visibleEvents = $derived(split.visible);
	const hiddenCount = $derived(split.hiddenCount);
</script>

<div class="audit-trail space-y-3" aria-label="Audit trail">
	{#if loading}
		<div class="space-y-2">
			{#each Array(5) as _}
				<div class="h-10 glass-subtle rounded-lg animate-pulse"></div>
			{/each}
		</div>
	{:else if errored}
		<p class="text-xs text-decay italic">Audit trail failed to load.</p>
	{:else if !memoryId}
		<p class="text-xs text-muted italic">No memory selected.</p>
	{:else if events.length === 0}
		<p class="text-xs text-muted italic">No audit events recorded yet.</p>
	{:else}
		<ol class="relative pl-6 border-l border-synapse/15 space-y-3">
			{#each visibleEvents as ev, i (ev.timestamp + i)}
				{@const m = META[ev.action]}
				{@const delta = formatRetentionDelta(ev.old_value, ev.new_value)}
				<li class="relative" style="animation-delay: {i * 40}ms;">
					<!-- Marker -->
					<span
						class="marker absolute -left-[29px] top-0.5 w-4 h-4 flex items-center justify-center rounded-full"
						style="background: rgba(10,10,26,0.9); box-shadow: 0 0 10px {m.color}88; border: 1px solid {m.color};"
						aria-hidden="true"
					>
						{#if m.kind === 'dot'}
							<span class="w-1.5 h-1.5 rounded-full" style="background: {m.color};"></span>
						{:else if m.kind === 'ring'}
							<span
								class="w-2 h-2 rounded-full border"
								style="border-color: {m.color}; background: transparent;"
							></span>
						{:else if m.kind === 'arrow-up'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill="none" stroke={m.color} stroke-width="2">
								<path d="M6 10V2M3 5l3-3 3 3" stroke-linecap="round" stroke-linejoin="round" />
							</svg>
						{:else if m.kind === 'arrow-down'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill="none" stroke={m.color} stroke-width="2">
								<path d="M6 2v8M3 7l3 3 3-3" stroke-linecap="round" stroke-linejoin="round" />
							</svg>
						{:else if m.kind === 'pencil'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill="none" stroke={m.color} stroke-width="1.5">
								<path d="M8.5 1.5l2 2L4 10l-3 1 1-3 6.5-6.5z" stroke-linejoin="round" />
							</svg>
						{:else if m.kind === 'x'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill="none" stroke={m.color} stroke-width="2">
								<path d="M2 2l8 8M10 2l-8 8" stroke-linecap="round" />
							</svg>
						{:else if m.kind === 'star'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill={m.color} stroke="none">
								<path d="M6 0.5l1.4 3.3 3.6.3-2.7 2.4.8 3.5L6 8.2l-3.1 1.8.8-3.5L1 4.1l3.6-.3L6 .5z" />
							</svg>
						{:else if m.kind === 'circle-arrow'}
							<svg viewBox="0 0 12 12" class="w-2.5 h-2.5" fill="none" stroke={m.color} stroke-width="1.5">
								<path d="M10 6a4 4 0 1 1-1.2-2.8" stroke-linecap="round" />
								<path d="M10 1.5V4H7.5" stroke-linecap="round" stroke-linejoin="round" />
							</svg>
						{/if}
					</span>

					<!-- Event content -->
					<div class="glass-subtle rounded-lg px-3 py-2 space-y-1">
						<div class="flex items-center justify-between gap-2 flex-wrap">
							<div class="flex items-center gap-2 text-xs">
								<span class="font-semibold" style="color: {m.color};">{m.label}</span>
								{#if ev.triggered_by}
									<span class="text-muted font-mono text-[10px]">{ev.triggered_by}</span>
								{/if}
							</div>
							<span class="text-[10px] text-muted font-mono" title={new Date(ev.timestamp).toLocaleString()}>
								{relativeTime(ev.timestamp)}
							</span>
						</div>
						{#if delta}
							<div class="text-[11px] text-dim font-mono">
								retention {delta}
							</div>
						{/if}
						{#if ev.reason}
							<div class="text-[11px] text-dim italic">{ev.reason}</div>
						{/if}
					</div>
				</li>
			{/each}
		</ol>

		{#if hiddenCount > 0}
			<button
				type="button"
				onclick={(e) => {
					e.stopPropagation();
					showAllOlder = !showAllOlder;
				}}
				class="text-xs text-synapse-glow hover:text-bright transition-colors underline-offset-4 hover:underline"
			>
				{showAllOlder ? 'Hide older events' : `Show ${hiddenCount} older event${hiddenCount === 1 ? '' : 's'}…`}
			</button>
		{/if}
	{/if}
</div>

<style>
	.audit-trail :global(ol > li) {
		animation: event-rise 400ms cubic-bezier(0.22, 0.8, 0.3, 1) backwards;
	}

	@keyframes event-rise {
		0% {
			opacity: 0;
			transform: translateX(6px);
		}
		100% {
			opacity: 1;
			transform: translateX(0);
		}
	}

	:global(.audit-trail .marker) {
		transition: transform 200ms ease;
	}

	:global(.audit-trail li:hover .marker) {
		transform: scale(1.15);
	}
</style>

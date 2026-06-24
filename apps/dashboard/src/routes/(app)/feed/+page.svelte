<script lang="ts">
	import { eventFeed, websocket, isConnected, isReconnecting } from '$stores/websocket';
	import { EVENT_TYPE_COLORS, type VestigeEvent } from '$types';
	import PipelineVisualizer from '$components/PipelineVisualizer.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon, { type IconName } from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';

	function formatTime(ts: string): string {
		return new Date(ts).toLocaleTimeString();
	}

	// Map each cognitive event type to a drawn route-icon so the feed speaks the
	// same visual language as the rest of the dashboard (no more Unicode glyphs).
	function eventIcon(type: string): IconName {
		const icons: Record<string, IconName> = {
			MemoryCreated: 'memories',
			MemoryUpdated: 'memories',
			MemoryDeleted: 'close',
			MemoryPromoted: 'importance',
			MemoryDemoted: 'importance',
			SearchPerformed: 'search',
			DreamStarted: 'dreams',
			DreamProgress: 'dreams',
			DreamCompleted: 'dreams',
			ConsolidationStarted: 'activation',
			ConsolidationCompleted: 'activation',
			RetentionDecayed: 'timeline',
			ConnectionDiscovered: 'graph',
			ActivationSpread: 'activation',
			ImportanceScored: 'importance',
			Heartbeat: 'pulse',
		};
		return icons[type] || 'sparkle';
	}

	function eventSummary(event: VestigeEvent): string {
		const d = event.data;
		switch (event.type) {
			case 'MemoryCreated': return `New ${d.node_type}: "${String(d.content_preview).slice(0, 60)}..."`;
			case 'SearchPerformed': return `Searched "${d.query}" → ${d.result_count} results (${d.duration_ms}ms)`;
			case 'DreamStarted': return `Dream started with ${d.memory_count} memories`;
			case 'DreamCompleted': return `Dream complete: ${d.connections_found} connections, ${d.insights_generated} insights (${d.duration_ms}ms)`;
			case 'ConsolidationStarted': return 'Consolidation cycle started';
			case 'ConsolidationCompleted': return `Consolidated ${d.nodes_processed} nodes, ${d.decay_applied} decayed (${d.duration_ms}ms)`;
			case 'ConnectionDiscovered': return `Connection: ${String(d.connection_type)} (weight: ${Number(d.weight).toFixed(2)})`;
			case 'ImportanceScored': return `Scored ${Number(d.composite_score).toFixed(2)}: "${String(d.content_preview).slice(0, 50)}..."`;
			case 'MemoryPromoted': return `Promoted → ${(Number(d.new_retention) * 100).toFixed(0)}% retention`;
			case 'MemoryDemoted': return `Demoted → ${(Number(d.new_retention) * 100).toFixed(0)}% retention`;
			default: return JSON.stringify(d).slice(0, 100);
		}
	}

	// Connection state drives the live pill — connected pings, reconnecting
	// breathes amber, offline goes quiet. Computed once, reused in the header.
	let pill = $derived(
		$isConnected
			? { color: 'var(--color-recall)', label: 'Live', live: true }
			: $isReconnecting
				? { color: 'var(--color-importance, #f59e0b)', label: 'Reconnecting', live: false }
				: { color: 'var(--color-decay, #8B95A5)', label: 'Offline', live: false }
	);
</script>

<div class="p-6 max-w-4xl mx-auto space-y-6 enter">
	<PageHeader
		icon="feed"
		title="Live Feed"
		subtitle="Every thought as it happens — memories born, searches fired, dreams consolidating, connections discovered. Vestige, thinking out loud."
		accent="synapse"
	>
		<span class="live-status" class:idle={!pill.live}>
			<span
				class="live-dot w-2 h-2 rounded-full"
				class:ping-host={pill.live}
				class:breathe={!pill.live}
				style="color:{pill.color};background:{pill.color}"
			></span>
			<span>{pill.label}</span>
		</span>

		<span class="count-chip">
			<AnimatedNumber value={$eventFeed.length} class="text-bright font-semibold" />
			<span class="text-dim">events</span>
		</span>

		<button
			onclick={() => websocket.clearEvents()}
			class="clear-btn"
			disabled={$eventFeed.length === 0}
		>
			<Icon name="close" size={13} />
			<span>Clear</span>
		</button>
	</PageHeader>

	{#if $eventFeed.length === 0}
		<div class="empty-state glass-panel rounded-2xl p-12 text-center space-y-4" use:reveal>
			<div class="empty-glyph" aria-hidden="true">
				<Icon name="feed" size={52} draw />
			</div>
			<p class="text-bright font-semibold text-lg">Quiet for now.</p>
			<p class="text-dim text-sm max-w-sm mx-auto leading-relaxed">
				New memory activity will stream in here the moment it happens — the feed
				wakes up the instant Vestige does.
			</p>
		</div>
	{:else}
		<div class="space-y-2">
			{#each $eventFeed as event, i (i)}
				{@const color = EVENT_TYPE_COLORS[event.type] || '#8B95A5'}
				<div
					use:reveal={{ delay: Math.min(i * 30, 300), y: 10 }}
					class="event-row lift flex items-start gap-3 p-3 glass-subtle rounded-xl"
					style="border-left: 3px solid {color}; --evt: {color};"
				>
					<div
						class="event-icon flex items-center justify-center flex-shrink-0"
						style="background: {color}1a; color: {color}"
					>
						<Icon name={eventIcon(event.type)} size={15} />
					</div>
					<div class="flex-1 min-w-0">
						<div class="flex items-center gap-2 mb-0.5">
							<span class="text-xs font-semibold tracking-wide" style="color: {color}">{event.type}</span>
							{#if event.data.timestamp}
								<span class="text-xs text-muted tabular-nums">{formatTime(String(event.data.timestamp))}</span>
							{/if}
						</div>
						<p class="text-sm text-dim leading-relaxed">{eventSummary(event)}</p>
						{#if event.type === 'SearchPerformed'}
							<div class="mt-2">
								<PipelineVisualizer
									resultCount={Number(event.data.result_count) || 0}
									durationMs={Number(event.data.duration_ms) || 0}
									active={true}
								/>
							</div>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	/* ── Live connection pill in the header right-slot ──────────────────────── */
	.live-status {
		display: inline-flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.3rem 0.7rem;
		border-radius: 999px;
		font-size: 0.75rem;
		font-weight: 600;
		color: var(--color-recall, #34d399);
		background: color-mix(in srgb, var(--color-recall, #34d399) 12%, transparent);
		border: 1px solid color-mix(in srgb, var(--color-recall, #34d399) 28%, transparent);
		white-space: nowrap;
	}
	.live-status.idle {
		color: var(--color-dim, #8b95a5);
		background: rgba(255, 255, 255, 0.04);
		border-color: rgba(255, 255, 255, 0.08);
	}
	.live-dot {
		display: inline-block;
		flex-shrink: 0;
	}

	/* ── Event-count chip ───────────────────────────────────────────────────── */
	.count-chip {
		display: inline-flex;
		align-items: baseline;
		gap: 0.35rem;
		padding: 0.3rem 0.7rem;
		border-radius: 999px;
		font-size: 0.78rem;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.07);
	}

	/* ── Clear button ───────────────────────────────────────────────────────── */
	.clear-btn {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.3rem 0.65rem;
		border-radius: 999px;
		font-size: 0.75rem;
		color: var(--color-muted, #6b7280);
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid rgba(255, 255, 255, 0.06);
		transition: color 0.2s ease, background 0.2s ease, border-color 0.2s ease;
	}
	.clear-btn:hover:not(:disabled) {
		color: var(--color-text, #e5e7eb);
		background: rgba(255, 255, 255, 0.05);
		border-color: rgba(255, 255, 255, 0.12);
	}
	.clear-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	/* ── Event rows ─────────────────────────────────────────────────────────── */
	.event-row {
		position: relative;
		transition:
			transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1),
			box-shadow 0.28s ease,
			background 0.2s ease;
	}
	.event-row:hover {
		background: color-mix(in srgb, var(--evt) 7%, transparent);
	}
	.event-icon {
		width: 1.6rem;
		height: 1.6rem;
		border-radius: 0.5rem;
	}

	/* ── Empty state (warm, not a frozen spinner) ───────────────────────────── */
	.empty-glyph {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		color: var(--color-synapse-glow, #818cf8);
		opacity: 0.85;
	}
	@media not (prefers-reduced-motion: reduce) {
		.empty-glyph {
			animation: breathe 3.2s ease-in-out infinite;
		}
	}
</style>

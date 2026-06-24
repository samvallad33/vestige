<script lang="ts">
	// ═══════════════════════════════════════════════════════════════════════
	//  AGENT BLACK BOX — the flight recorder for agent cognition.
	// ───────────────────────────────────────────────────────────────────────
	//  Watch the agent think. Watch memory change. Watch the receipt prove why.
	//
	//  Every MCP tool call carries a runId that threads, unbroken, through the
	//  tool output → SQLite trace rows → WebSocket → this page → the export →
	//  Cinema. This tab replays that exact run: a timeline scrubber, per-event
	//  detail, the suppressed memories, trust scores, contradiction decisions,
	//  and a one-click `.vestige-trace.json` export.
	//
	//  Live events are real — they arrive over the WebSocket backed by trace
	//  rows. No fake demo events.
	// ═══════════════════════════════════════════════════════════════════════
	import { onMount } from 'svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import ReceiptCard from '$components/ReceiptCard.svelte';
	import {
		api,
		type TraceRunSummary,
		type TraceEvent,
		type TraceDetail,
		type Receipt
	} from '$lib/stores/api';
	import { isConnected, liveRunId, lastTraceEvent, traceEvents } from '$lib/stores/websocket';
	import {
		eventColor,
		eventLabel,
		eventGlyph,
		eventSummary,
		eventMemoryIds,
		formatAt,
		relativeMs
	} from '$components/blackbox-helpers';

	// ---- state ----------------------------------------------------------
	let runs = $state<TraceRunSummary[]>([]);
	let selectedRunId = $state<string | null>(null);
	let detail = $state<TraceDetail | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);
	let scrubIndex = $state(0); // index into detail.events
	let proofMode = $state(false);
	let receipts = $state<Receipt[]>([]);

	// The events up to and including the scrubber position — what the agent had
	// "experienced" at that moment in the run.
	const visibleEvents = $derived(detail ? detail.events.slice(0, scrubIndex + 1) : []);
	const currentEvent = $derived<TraceEvent | null>(
		detail && detail.events.length ? detail.events[scrubIndex] : null
	);
	const startAt = $derived(detail?.events[0]?.at ?? 0);

	// Memory ids that have been touched up to the scrubber — the live pulse set.
	const pulsedIds = $derived(
		Array.from(new Set(visibleEvents.flatMap(eventMemoryIds)))
	);

	// Honest producer status for this run. Two event kinds depend on optional
	// upstream producers that are off by default — we say so explicitly instead
	// of rendering a confusing empty space.
	const hasVeto = $derived(detail?.events.some((e) => e.type === 'sanhedrin.veto') ?? false);
	const hasDream = $derived(detail?.events.some((e) => e.type === 'dream.patch') ?? false);
	const hasContradiction = $derived(
		detail?.events.some((e) => e.type === 'contradiction.detected') ?? false
	);

	async function loadRuns() {
		try {
			const res = await api.traces.list(100);
			runs = res.runs;
			if (!selectedRunId && runs.length) selectRun(runs[0].runId);
		} catch (e) {
			error = String(e);
		}
	}

	async function selectRun(runId: string) {
		selectedRunId = runId;
		loading = true;
		error = null;
		try {
			detail = await api.traces.get(runId);
			scrubIndex = Math.max(0, (detail.events.length || 1) - 1);
			// Receipts are the proof behind THIS run's retrievals — scoped to
			// the selected run (B5), not the global latest.
			receipts = (await api.receipts.listForRun(runId, 8)).receipts;
		} catch (e) {
			error = String(e);
			detail = null;
		} finally {
			loading = false;
		}
	}

	function exportTrace() {
		if (!selectedRunId) return;
		// Direct browser download of the .vestige-trace.json artifact.
		window.location.href = api.traces.exportUrl(selectedRunId);
	}

	// Live: when a trace event for the *currently open* run arrives, refresh it
	// so the timeline grows in real time. Also refresh the run list so new runs
	// appear at the top.
	$effect(() => {
		const last = $lastTraceEvent;
		if (!last) return;
		const evRunId = last.data?.run_id as string | undefined;
		if (evRunId && evRunId === selectedRunId) {
			// Re-fetch the open run (cheap; trace rows are local SQLite).
			api.traces.get(selectedRunId).then((d) => {
				detail = d;
				// Keep the scrubber pinned to the newest event in live mode.
				scrubIndex = Math.max(0, d.events.length - 1);
			});
		}
	});

	onMount(loadRuns);
</script>

<div class="mx-auto max-w-6xl px-5 py-6">
	<PageHeader
		icon="blackbox"
		title="Agent Black Box"
		subtitle="Watch the agent think. Watch memory change. Watch the receipt prove why."
		accent="synapse"
	>
		<button
			class="mode-toggle"
			class:on={proofMode}
			onclick={() => (proofMode = !proofMode)}
			title="Proof Mode: a clean launch-footage view"
		>
			<Icon name="sparkle" size={14} />
			Proof Mode
		</button>
		<button class="export-btn" onclick={exportTrace} disabled={!selectedRunId}>
			<Icon name="feed" size={14} />
			Export .vestige-trace.json
		</button>
	</PageHeader>

	<!-- ░░ LIVE SPINE HEADER — the proof line: one runId, end to end ░░ -->
	<div class="spine glass" use:reveal>
		<div class="spine-item">
			<span class="spine-label">WebSocket</span>
			<span class="spine-value" class:live={$isConnected}>
				<span class="dot" class:live={$isConnected}></span>
				{$isConnected ? 'Connected' : 'Offline'}
			</span>
		</div>
		<div class="spine-item">
			<span class="spine-label">Live runId</span>
			<code class="spine-run">{$liveRunId ?? '—'}</code>
		</div>
		<div class="spine-item">
			<span class="spine-label">Last event</span>
			<span class="spine-value">
				{#if $lastTraceEvent}
					<span class="ev-chip" style:--c={eventColor(($lastTraceEvent.data?.event as TraceEvent)?.type)}>
						{eventLabel(($lastTraceEvent.data?.event as TraceEvent)?.type)}
					</span>
				{:else}
					<span class="text-dim">awaiting…</span>
				{/if}
			</span>
		</div>
		<div class="spine-item">
			<span class="spine-label">Events seen</span>
			<span class="spine-value">
				<AnimatedNumber value={$traceEvents.length} />
			</span>
		</div>
	</div>

	{#if !proofMode}
		<div class="layout">
			<!-- ░░ RUN PICKER ░░ -->
			<aside class="runs glass" use:reveal>
				<h2 class="panel-title">Runs</h2>
				{#if runs.length === 0}
					<p class="empty">
						No agent runs recorded yet. Make an MCP tool call — every call is
						recorded here.
					</p>
				{:else}
					<ul>
						{#each runs as run (run.runId)}
							<li>
								<button
									class="run-row"
									class:active={run.runId === selectedRunId}
									onclick={() => selectRun(run.runId)}
								>
									<div class="run-top">
										<code class="run-id">{run.runId.replace('run_', '').slice(0, 10)}</code>
										<span class="run-tool">{run.firstTool ?? '—'}</span>
									</div>
									<div class="run-stats">
										<span title="events">{run.eventCount} ev</span>
										{#if run.retrievedCount}<span class="s-recall">↑{run.retrievedCount}</span>{/if}
										{#if run.suppressedCount}<span class="s-suppress">⊘{run.suppressedCount}</span>{/if}
										{#if run.writeCount}<span class="s-write">✎{run.writeCount}</span>{/if}
										{#if run.vetoCount}<span class="s-veto">⛔{run.vetoCount}</span>{/if}
									</div>
								</button>
							</li>
						{/each}
					</ul>
				{/if}
			</aside>

			<!-- ░░ REPLAY ░░ -->
			<section class="replay">
				{#if loading}
					<div class="glass center-msg">Loading trace…</div>
				{:else if error}
					<div class="glass center-msg err">{error}</div>
				{:else if !detail}
					<div class="glass center-msg">Select a run to replay.</div>
				{:else}
					<!-- Scrubber -->
					<div class="scrubber glass" use:reveal>
						<div class="scrub-head">
							<span class="scrub-title">
								Step <strong>{scrubIndex + 1}</strong> / {detail.events.length}
							</span>
							{#if currentEvent}
								<span class="scrub-time">+{relativeMs(currentEvent.at, startAt)}ms</span>
							{/if}
						</div>
						<input
							type="range"
							min="0"
							max={Math.max(0, detail.events.length - 1)}
							bind:value={scrubIndex}
							class="scrub-range"
						/>
						<!-- A tick row colored by event kind — the run at a glance. -->
						<div class="ticks">
							{#each detail.events as ev, i (i)}
								<button
									class="tick"
									class:past={i <= scrubIndex}
									style:--c={eventColor(ev.type)}
									onclick={() => (scrubIndex = i)}
									title={eventLabel(ev.type)}
									aria-label={`Step ${i + 1}: ${eventLabel(ev.type)}`}
								></button>
							{/each}
						</div>
					</div>

					<!-- Current event detail -->
					{#if currentEvent}
						<div class="event-detail glass" use:reveal style:--c={eventColor(currentEvent.type)}>
							<div class="ed-head">
								<span class="ed-glyph">{eventGlyph(currentEvent.type)}</span>
								<span class="ed-label">{eventLabel(currentEvent.type)}</span>
								<code class="ed-time">{formatAt(currentEvent.at)}</code>
							</div>
							<p class="ed-summary">{eventSummary(currentEvent)}</p>

							{#if currentEvent.type === 'memory.retrieve'}
								<div class="ids-grid">
									{#each currentEvent.ids as id (id)}
										<span class="id-chip" style:--a={currentEvent.activation[id] ?? 0}>
											<code>{id.slice(0, 8)}</code>
											{#if currentEvent.activation[id] != null}
												<small>{(currentEvent.activation[id] * 100).toFixed(0)}%</small>
											{/if}
										</span>
									{/each}
								</div>
							{:else if currentEvent.type === 'contradiction.detected'}
								<div class="contra">
									<span class="winner">kept {currentEvent.winnerId?.slice(0, 8)}</span>
									<span class="vs">vs</span>
									{#each currentEvent.ids.filter((i) => i !== currentEvent.winnerId) as id (id)}
										<span class="loser">{id.slice(0, 8)}</span>
									{/each}
								</div>
							{:else if currentEvent.type === 'sanhedrin.veto'}
								<div class="veto-evidence">
									{#each currentEvent.evidenceIds as id (id)}
										<code>{id.slice(0, 8)}</code>
									{/each}
								</div>
							{/if}
						</div>
					{/if}

					<!-- Pulse set: the memories touched so far -->
					<div class="pulse glass" use:reveal>
						<h3 class="panel-title">
							Memory pulse <span class="text-dim">— touched this run</span>
						</h3>
						{#if pulsedIds.length === 0}
							<p class="empty">No memories touched yet.</p>
						{:else}
							<div class="pulse-grid">
								{#each pulsedIds as id (id)}
									<code class="pulse-node">{id.slice(0, 8)}</code>
								{/each}
							</div>
						{/if}
					</div>

					<!-- Producer status — honest about what's live vs. off-by-default -->
					<div class="producers glass" use:reveal>
						<h3 class="panel-title">Event producers <span class="text-dim">— this run</span></h3>
						<ul class="producer-list">
							<li class="producer ok">
								<span class="p-dot"></span> mcp.call · memory.write · memory.retrieve · memory.suppress
								<span class="p-state">live</span>
							</li>
							<li class="producer" class:ok={hasContradiction}>
								<span class="p-dot"></span> contradiction.detected
								<span class="p-state">
									{hasContradiction ? 'fired this run' : 'no contradiction in this run'}
								</span>
							</li>
							<li class="producer caveat" class:ok={hasDream}>
								<span class="p-dot"></span> dream.patch
								<span class="p-state">
									{hasDream ? 'fired this run' : 'No dream run in this trace'}
								</span>
							</li>
							<li class="producer caveat" class:ok={hasVeto}>
								<span class="p-dot"></span> sanhedrin.veto
								<span class="p-state">
									{hasVeto ? 'fired this run' : 'No veto producer connected (optional Sanhedrin hook, off by default)'}
								</span>
							</li>
						</ul>
					</div>

					<!-- Receipts — the nutrition label behind this run's retrievals -->
					{#if receipts.length}
						<div class="receipts-panel glass" use:reveal>
							<h3 class="panel-title">
								Receipts <span class="text-dim">— proof behind retrievals</span>
							</h3>
							<div class="receipts-grid">
								{#each receipts.slice(0, 2) as r (r.receipt_id)}
									<ReceiptCard receipt={r} />
								{/each}
							</div>
						</div>
					{/if}

					<!-- Full event log -->
					<div class="log glass" use:reveal>
						<h3 class="panel-title">Event log</h3>
						<ol class="log-list">
							{#each detail.events as ev, i (i)}
								<li
									class="log-row"
									class:active={i === scrubIndex}
									class:dim={i > scrubIndex}
									style:--c={eventColor(ev.type)}
								>
									<button class="log-btn" onclick={() => (scrubIndex = i)}>
										<span class="log-glyph">{eventGlyph(ev.type)}</span>
										<span class="log-label">{eventLabel(ev.type)}</span>
										<span class="log-summary">{eventSummary(ev)}</span>
										<span class="log-t">+{relativeMs(ev.at, startAt)}ms</span>
									</button>
								</li>
							{/each}
						</ol>
					</div>
				{/if}
			</section>
		</div>
	{:else}
		<!-- ░░ PROOF MODE — clean launch-footage view ░░ -->
		<div class="proof-stage glass" use:reveal>
			<div class="proof-headline">
				<span class="dot big" class:live={$isConnected}></span>
				<code class="proof-run">{$liveRunId ?? 'awaiting run…'}</code>
			</div>
			{#if $lastTraceEvent}
				{@const ev = $lastTraceEvent.data?.event as TraceEvent}
				<div class="proof-event" style:--c={eventColor(ev?.type)}>
					<span class="proof-glyph">{eventGlyph(ev?.type)}</span>
					<div>
						<div class="proof-ev-label">{eventLabel(ev?.type)}</div>
						<div class="proof-ev-sum">{eventSummary(ev)}</div>
					</div>
				</div>
			{/if}
			<div class="proof-counter">
				<AnimatedNumber value={$traceEvents.length} />
				<span class="proof-counter-label">trace events</span>
			</div>
			<p class="proof-tagline">Watch the agent think. Watch memory change. Watch the receipt prove why.</p>
		</div>
	{/if}
</div>

<style>
	.mode-toggle,
	.export-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 7px 12px;
		font-size: 0.78rem;
		font-weight: 600;
		border-radius: 9px;
		border: 1px solid color-mix(in oklab, var(--color-synapse) 30%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 8%, transparent);
		color: var(--color-synapse-glow);
		cursor: pointer;
		transition: all 0.18s ease;
	}
	.mode-toggle:hover,
	.export-btn:hover:not(:disabled) {
		background: color-mix(in oklab, var(--color-synapse) 18%, transparent);
		transform: translateY(-1px);
	}
	.mode-toggle.on {
		background: var(--color-synapse);
		color: white;
	}
	.export-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	/* Spine header */
	.spine {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 1px;
		border-radius: 14px;
		padding: 14px 18px;
		margin-bottom: 18px;
		overflow: hidden;
	}
	.spine-item {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 2px 14px;
	}
	.spine-label {
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-text-dim, #8b8ba7);
	}
	.spine-value {
		font-size: 0.92rem;
		font-weight: 600;
		display: inline-flex;
		align-items: center;
		gap: 7px;
	}
	.spine-run {
		font-size: 0.82rem;
		color: var(--color-synapse-glow);
		word-break: break-all;
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #64748b;
		box-shadow: 0 0 0 0 transparent;
	}
	.dot.live {
		background: var(--color-recall, #10b981);
		animation: ping 2s ease-in-out infinite;
	}
	.dot.big {
		width: 12px;
		height: 12px;
	}
	@keyframes ping {
		0%,
		100% {
			box-shadow: 0 0 0 0 color-mix(in oklab, var(--color-recall) 60%, transparent);
		}
		50% {
			box-shadow: 0 0 0 6px transparent;
		}
	}
	.ev-chip {
		font-size: 0.74rem;
		font-weight: 700;
		padding: 2px 8px;
		border-radius: 6px;
		color: var(--c);
		background: color-mix(in oklab, var(--c) 14%, transparent);
		border: 1px solid color-mix(in oklab, var(--c) 35%, transparent);
	}

	/* Two-column layout */
	.layout {
		display: grid;
		grid-template-columns: 260px 1fr;
		gap: 16px;
		align-items: start;
	}
	@media (max-width: 860px) {
		.layout {
			grid-template-columns: 1fr;
		}
	}

	.glass {
		background: color-mix(in oklab, var(--color-void, #050510) 55%, transparent);
		border: 1px solid color-mix(in oklab, white 8%, transparent);
		backdrop-filter: blur(12px);
		border-radius: 14px;
	}

	.panel-title {
		font-size: 0.82rem;
		font-weight: 700;
		letter-spacing: 0.02em;
		margin: 0 0 12px;
	}
	.text-dim {
		color: var(--color-text-dim, #8b8ba7);
		font-weight: 400;
	}
	.empty {
		font-size: 0.82rem;
		color: var(--color-text-dim, #8b8ba7);
		line-height: 1.5;
	}

	.runs {
		padding: 16px;
		position: sticky;
		top: 16px;
		max-height: calc(100vh - 40px);
		overflow-y: auto;
	}
	.runs ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.run-row {
		width: 100%;
		text-align: left;
		padding: 9px 11px;
		border-radius: 10px;
		border: 1px solid transparent;
		background: color-mix(in oklab, white 3%, transparent);
		cursor: pointer;
		transition: all 0.16s ease;
	}
	.run-row:hover {
		background: color-mix(in oklab, var(--color-synapse) 12%, transparent);
	}
	.run-row.active {
		border-color: color-mix(in oklab, var(--color-synapse) 45%, transparent);
		background: color-mix(in oklab, var(--color-synapse) 16%, transparent);
	}
	.run-top {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 8px;
	}
	.run-id {
		font-size: 0.78rem;
		color: var(--color-synapse-glow);
	}
	.run-tool {
		font-size: 0.7rem;
		color: var(--color-text-dim, #8b8ba7);
		white-space: nowrap;
	}
	.run-stats {
		display: flex;
		gap: 8px;
		margin-top: 5px;
		font-size: 0.68rem;
		color: var(--color-text-dim, #8b8ba7);
	}
	.s-recall {
		color: var(--color-recall, #10b981);
	}
	.s-suppress {
		color: #a78bfa;
	}
	.s-write {
		color: #38bdf8;
	}
	.s-veto {
		color: #f43f5e;
	}

	.replay {
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-width: 0;
	}
	.center-msg {
		padding: 40px;
		text-align: center;
		color: var(--color-text-dim, #8b8ba7);
	}
	.center-msg.err {
		color: #f87171;
	}

	/* Scrubber */
	.scrubber {
		padding: 16px 18px;
	}
	.scrub-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 10px;
		font-size: 0.82rem;
	}
	.scrub-time {
		color: var(--color-synapse-glow);
		font-variant-numeric: tabular-nums;
	}
	.scrub-range {
		width: 100%;
		accent-color: var(--color-synapse);
	}
	.ticks {
		display: flex;
		gap: 2px;
		margin-top: 10px;
		height: 22px;
		align-items: stretch;
	}
	.tick {
		flex: 1;
		min-width: 2px;
		border: none;
		border-radius: 2px;
		background: color-mix(in oklab, var(--c) 22%, transparent);
		cursor: pointer;
		transition: all 0.16s ease;
		padding: 0;
	}
	.tick.past {
		background: var(--c);
		box-shadow: 0 0 6px -1px var(--c);
	}
	.tick:hover {
		transform: scaleY(1.25);
	}

	/* Event detail */
	.event-detail {
		padding: 16px 18px;
		border-left: 3px solid var(--c);
	}
	.ed-head {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.ed-glyph {
		font-size: 1.2rem;
		color: var(--c);
	}
	.ed-label {
		font-weight: 700;
		color: var(--c);
	}
	.ed-time {
		margin-left: auto;
		font-size: 0.74rem;
		color: var(--color-text-dim, #8b8ba7);
	}
	.ed-summary {
		margin: 10px 0 0;
		font-size: 0.9rem;
		line-height: 1.5;
	}
	.ids-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 12px;
	}
	.id-chip {
		display: inline-flex;
		align-items: center;
		gap: 5px;
		padding: 3px 8px;
		border-radius: 7px;
		background: color-mix(in oklab, var(--color-recall) calc(var(--a, 0) * 30%), transparent);
		border: 1px solid color-mix(in oklab, var(--color-recall) 30%, transparent);
		font-size: 0.74rem;
	}
	.id-chip small {
		color: var(--color-recall, #10b981);
		font-weight: 700;
	}
	.contra {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 12px;
		font-size: 0.8rem;
	}
	.winner {
		color: var(--color-recall, #10b981);
		font-weight: 700;
	}
	.vs {
		color: var(--color-text-dim, #8b8ba7);
	}
	.loser {
		color: #fb7185;
		text-decoration: line-through;
		opacity: 0.7;
	}
	.veto-evidence {
		display: flex;
		gap: 6px;
		margin-top: 12px;
		flex-wrap: wrap;
	}
	.veto-evidence code {
		padding: 2px 7px;
		border-radius: 6px;
		background: color-mix(in oklab, #f43f5e 12%, transparent);
		font-size: 0.74rem;
	}

	/* Pulse */
	.pulse {
		padding: 16px 18px;
	}
	.pulse-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.pulse-node {
		padding: 3px 8px;
		border-radius: 7px;
		background: color-mix(in oklab, var(--color-synapse) 12%, transparent);
		border: 1px solid color-mix(in oklab, var(--color-synapse) 28%, transparent);
		font-size: 0.74rem;
		color: var(--color-synapse-glow);
		animation: pulse-in 0.4s ease;
	}
	@keyframes pulse-in {
		from {
			transform: scale(0.85);
			opacity: 0;
		}
		to {
			transform: scale(1);
			opacity: 1;
		}
	}

	/* Receipts panel */
	.receipts-panel {
		padding: 16px 18px;
	}
	.receipts-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
		gap: 12px;
	}

	/* Producers — honest event-source status */
	.producers {
		padding: 16px 18px;
	}
	.producer-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 7px;
	}
	.producer {
		display: flex;
		align-items: center;
		gap: 9px;
		font-size: 0.78rem;
		color: var(--color-text-dim, #8b8ba7);
	}
	.producer .p-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #475569;
		flex-shrink: 0;
	}
	.producer.ok {
		color: var(--color-text, #e2e2f0);
	}
	.producer.ok .p-dot {
		background: var(--color-recall, #10b981);
		box-shadow: 0 0 6px -1px var(--color-recall, #10b981);
	}
	.producer.caveat:not(.ok) .p-dot {
		background: #f59e0b;
		opacity: 0.6;
	}
	.p-state {
		margin-left: auto;
		font-size: 0.7rem;
		font-style: italic;
		text-align: right;
		color: var(--color-text-dim, #8b8ba7);
	}
	.producer.caveat:not(.ok) .p-state {
		color: #f59e0b;
	}

	/* Log */
	.log {
		padding: 16px 18px;
	}
	.log-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.log-row {
		border-radius: 8px;
		border-left: 2px solid var(--c);
		transition: all 0.16s ease;
	}
	.log-row.active {
		background: color-mix(in oklab, var(--c) 14%, transparent);
	}
	.log-row.dim {
		opacity: 0.4;
	}
	.log-btn {
		width: 100%;
		display: grid;
		grid-template-columns: 22px 110px 1fr auto;
		align-items: center;
		gap: 8px;
		padding: 7px 10px;
		background: none;
		border: none;
		cursor: pointer;
		text-align: left;
		font-size: 0.8rem;
	}
	.log-glyph {
		color: var(--c);
		text-align: center;
	}
	.log-label {
		font-weight: 600;
		color: var(--c);
	}
	.log-summary {
		color: var(--color-text-dim, #8b8ba7);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.log-t {
		font-size: 0.7rem;
		color: var(--color-text-dim, #8b8ba7);
		font-variant-numeric: tabular-nums;
	}

	/* Proof mode */
	.proof-stage {
		padding: 60px 40px;
		text-align: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 26px;
		min-height: 50vh;
		justify-content: center;
	}
	.proof-headline {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	.proof-run {
		font-size: 1.4rem;
		color: var(--color-synapse-glow);
		font-weight: 700;
	}
	.proof-event {
		display: flex;
		align-items: center;
		gap: 16px;
		padding: 18px 26px;
		border-radius: 14px;
		border: 1px solid color-mix(in oklab, var(--c) 40%, transparent);
		background: color-mix(in oklab, var(--c) 10%, transparent);
	}
	.proof-glyph {
		font-size: 2rem;
		color: var(--c);
	}
	.proof-ev-label {
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--c);
	}
	.proof-ev-sum {
		font-size: 0.85rem;
		color: var(--color-text-dim, #8b8ba7);
	}
	.proof-counter {
		font-size: 3.4rem;
		font-weight: 800;
		line-height: 1;
		color: var(--color-synapse-glow);
	}
	.proof-counter-label {
		display: block;
		font-size: 0.8rem;
		font-weight: 500;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--color-text-dim, #8b8ba7);
		margin-top: 8px;
	}
	.proof-tagline {
		font-size: 1rem;
		color: var(--color-text-dim, #c0c0d8);
		max-width: 32ch;
		line-height: 1.5;
	}
</style>

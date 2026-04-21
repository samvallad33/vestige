<!--
  AmbientAwarenessStrip — persistent slim top-of-viewport band surfacing
  live cognitive engine vitals without demanding attention.

  Contents (left → right):
    1. Retention Vitals   — pulsing dot + "N memories · X% avg retention"
    2. At-Risk Count      — memories with retention < 0.3 (or "—" if unknown)
    3. Active Intentions  — count of active intentions, pings pink if >5
    4. Recent Dream       — last DreamCompleted within 24h summary
    5. Activity Pulse     — 10-bar sparkline of events/min over last 5 min
    6. Now Dreaming?      — violet pulsing dot while a Dream is in flight
    7. Sanhedrin Watch    — subtle red flash on MemorySuppressed in last 10s

  Design: full-width band, dark-glass backdrop, border-bottom synapse/15,
  height ≈36px, dim muted text with colored accents ONLY on pulsing/urgent
  items. Not clickable — ambient info only.

  Mobile: collapses to items 1, 2, 6 to save width.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import {
		memoryCount,
		avgRetention,
		eventFeed,
	} from '$stores/websocket';
	import { api } from '$stores/api';
	import {
		bucketizeActivity,
		dreamInsightsCount,
		findRecentDream,
		formatAgo,
		hasRecentSuppression,
		isDreaming as isDreamingFn,
		parseEventTimestamp,
	} from './awareness-helpers';

	// ─────────────────────────────────────────────────────────────────────────
	// 1. Retention vitals — derived straight from heartbeat stores
	// ─────────────────────────────────────────────────────────────────────────
	const retentionPct = $derived(Math.round(($avgRetention ?? 0) * 100));
	const retentionHealthy = $derived(($avgRetention ?? 0) >= 0.5);

	// ─────────────────────────────────────────────────────────────────────────
	// 2. At-risk count — fetched once from /retention-distribution.
	//    Sum buckets whose range label implies retention < 0.3 ("0-20%" and
	//    "20-40%"). Robust to absent/unknown backend: stays `null` → shows "—".
	// ─────────────────────────────────────────────────────────────────────────
	let atRiskCount = $state<number | null>(null);

	async function loadAtRisk(): Promise<void> {
		try {
			const dist = await api.retentionDistribution();
			// Prefer direct `endangered` list if backend populates it.
			if (Array.isArray(dist.endangered) && dist.endangered.length > 0) {
				atRiskCount = dist.endangered.length;
				return;
			}
			// Otherwise sum buckets whose lower bound < 30%.
			const buckets = dist.distribution ?? [];
			let total = 0;
			for (const b of buckets) {
				const m = /^(\d+)/.exec(b.range);
				if (!m) continue;
				const low = Number.parseInt(m[1], 10);
				if (Number.isFinite(low) && low < 30) total += b.count ?? 0;
			}
			atRiskCount = total;
		} catch {
			atRiskCount = null;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// 3. Active intentions — fetched once from /intentions?status=active
	// ─────────────────────────────────────────────────────────────────────────
	let intentionsCount = $state<number | null>(null);

	async function loadIntentions(): Promise<void> {
		try {
			const res = await api.intentions('active');
			intentionsCount = res.total ?? res.intentions?.length ?? 0;
		} catch {
			intentionsCount = null;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// 4 & 6. Dream awareness — pure helpers scan $eventFeed. Newest-first.
	// ─────────────────────────────────────────────────────────────────────────
	let nowTick = $state(Date.now());

	const dreamState = $derived.by(() => {
		const feed = $eventFeed;
		const recent = findRecentDream(feed, nowTick);
		const recentAt = recent ? parseEventTimestamp(recent) ?? nowTick : null;
		const recentMsAgo = recentAt !== null ? nowTick - recentAt : null;
		return {
			isDreaming: isDreamingFn(feed, nowTick),
			recent,
			recentMsAgo,
			insights: dreamInsightsCount(recent),
		};
	});

	// ─────────────────────────────────────────────────────────────────────────
	// 5. Activity pulse — bucket $eventFeed timestamps into 10 × 30s buckets
	//    over the last 5 minutes. Bucket 0 = oldest, 9 = newest.
	// ─────────────────────────────────────────────────────────────────────────
	const sparkline = $derived(bucketizeActivity($eventFeed, nowTick));

	// ─────────────────────────────────────────────────────────────────────────
	// 7. Sanhedrin watch — flash red on any MemorySuppressed in last 10s
	// ─────────────────────────────────────────────────────────────────────────
	const suppressionFlash = $derived(hasRecentSuppression($eventFeed, nowTick));

	// ─────────────────────────────────────────────────────────────────────────
	// Ticker — advance `nowTick` every second so time-based derived values
	// (dreaming window, activity window, suppression flash) refresh smoothly.
	// ─────────────────────────────────────────────────────────────────────────
	onMount(() => {
		void loadAtRisk();
		void loadIntentions();
		const tickHandle = setInterval(() => {
			nowTick = Date.now();
		}, 1000);
		// Refresh the slow API-backed counts every 60s so they don't go stale.
		const slowHandle = setInterval(() => {
			void loadAtRisk();
			void loadIntentions();
		}, 60_000);
		return () => {
			clearInterval(tickHandle);
			clearInterval(slowHandle);
		};
	});
</script>

<div
	class="ambient-strip relative flex h-9 w-full items-center gap-0 overflow-hidden border-b border-synapse/15 bg-black/40 px-3 text-[11px] text-dim backdrop-blur-md"
	class:ambient-flash={suppressionFlash}
	aria-label="Ambient cognitive vitals"
>
	<!-- 1. Retention vitals — always visible -->
	<div class="strip-item" title="Total memories and average retention strength">
		<span class="relative inline-flex h-2 w-2 items-center justify-center">
			<span
				class="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
				class:bg-recall={retentionHealthy}
				class:bg-warning={!retentionHealthy}
			></span>
			<span
				class="relative inline-flex h-2 w-2 rounded-full"
				class:bg-recall={retentionHealthy}
				class:bg-warning={!retentionHealthy}
			></span>
		</span>
		<span class="text-text/80 tabular-nums">{$memoryCount}</span>
		<span class="text-muted">memories</span>
		<span class="text-muted/60">·</span>
		<span class:text-recall={retentionHealthy} class:text-warning={!retentionHealthy}>
			{retentionPct}%
		</span>
		<span class="text-muted">avg retention</span>
	</div>

	<div class="strip-divider" aria-hidden="true"></div>

	<!-- 2. At-risk — always visible -->
	<div class="strip-item" title="Memories with retention below 30%">
		{#if atRiskCount !== null && atRiskCount > 0}
			<span class="font-semibold tabular-nums text-decay">{atRiskCount}</span>
			<span class="text-muted">at risk</span>
		{:else if atRiskCount === 0}
			<span class="text-muted tabular-nums">0</span>
			<span class="text-muted">at risk</span>
		{:else}
			<span class="text-muted/60">—</span>
			<span class="text-muted">at risk</span>
		{/if}
	</div>

	<!-- 3. Active intentions — hidden on mobile -->
	<div class="strip-divider hidden md:block" aria-hidden="true"></div>
	<div class="strip-item hidden md:inline-flex" title="Active intentions (prospective memory)">
		{#if intentionsCount !== null}
			<span
				class="inline-flex h-2 w-2 rounded-full"
				class:bg-node-pattern={intentionsCount > 5}
				class:animate-ping-slow={intentionsCount > 5}
				class:bg-muted={intentionsCount <= 5}
			></span>
			<span
				class="tabular-nums"
				class:text-node-pattern={intentionsCount > 5}
				class:text-text={intentionsCount > 0 && intentionsCount <= 5}
				class:text-muted={intentionsCount === 0}
			>
				{intentionsCount}
			</span>
			<span class="text-muted">intentions</span>
		{:else}
			<span class="text-muted/60">— intentions</span>
		{/if}
	</div>

	<!-- 4. Recent dream — hidden on mobile -->
	<div class="strip-divider hidden md:block" aria-hidden="true"></div>
	<div class="strip-item hidden md:inline-flex" title="Most recent Dream cycle completion">
		{#if dreamState.recent && dreamState.recentMsAgo !== null}
			<span class="text-dream/80">✦</span>
			<span class="text-muted">Last dream:</span>
			<span class="text-text/80">{formatAgo(dreamState.recentMsAgo)}</span>
			{#if dreamState.insights !== null}
				<span class="text-muted/60">·</span>
				<span class="text-text/80 tabular-nums">{dreamState.insights}</span>
				<span class="text-muted">insights</span>
			{/if}
		{:else}
			<span class="text-muted">No recent dream</span>
		{/if}
	</div>

	<!-- 5. Activity pulse sparkline — hidden on mobile -->
	<div class="strip-divider hidden md:block" aria-hidden="true"></div>
	<div
		class="strip-item hidden md:inline-flex"
		title="Event throughput over the last 5 minutes (events per 30s)"
	>
		<span class="text-muted">activity</span>
		<div class="flex h-4 items-end gap-[2px]" aria-hidden="true">
			{#each sparkline as bar}
				<div
					class="w-[3px] rounded-sm bg-synapse/70"
					style="height: {Math.max(10, bar.ratio * 100)}%; opacity: {bar.count === 0 ? 0.18 : 0.5 + bar.ratio * 0.5};"
				></div>
			{/each}
		</div>
	</div>

	<!-- 6. Now dreaming? — always visible when active -->
	{#if dreamState.isDreaming}
		<div class="strip-divider" aria-hidden="true"></div>
		<div class="strip-item" title="A Dream cycle is currently in progress">
			<span class="relative inline-flex h-2 w-2 items-center justify-center">
				<span
					class="absolute inline-flex h-full w-full animate-ping rounded-full bg-dream opacity-75"
				></span>
				<span class="relative inline-flex h-2 w-2 rounded-full bg-dream"></span>
			</span>
			<span class="font-semibold tracking-wider text-dream-glow">DREAMING...</span>
		</div>
	{/if}

	<!-- Spacer -->
	<div class="flex-1"></div>

	<!-- 7. Sanhedrin watch — subtle right-aligned flash, hidden on mobile -->
	{#if suppressionFlash}
		<div class="strip-item hidden md:inline-flex" title="A memory was just suppressed (Sanhedrin veto)">
			<span
				class="inline-flex h-2 w-2 animate-pulse rounded-full bg-decay shadow-[0_0_10px_rgba(239,68,68,0.7)]"
			></span>
			<span class="font-medium text-decay">Veto triggered</span>
		</div>
	{/if}
</div>

<style>
	.strip-item {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0 0.75rem;
		white-space: nowrap;
		flex-shrink: 0;
	}

	.strip-divider {
		width: 1px;
		height: 14px;
		background: rgba(99, 102, 241, 0.12);
		flex-shrink: 0;
	}

	/* Subtle red wash when a suppression just fired. */
	.ambient-strip.ambient-flash {
		background:
			linear-gradient(90deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0) 70%),
			rgba(0, 0, 0, 0.4);
		border-bottom-color: rgba(239, 68, 68, 0.35);
		transition: background 0.3s ease, border-color 0.3s ease;
	}

	/* Slower "ping" for the intentions pink dot — less aggressive than the
	   default Tailwind animate-ping. */
	@keyframes ping-slow {
		0% { transform: scale(1); opacity: 0.8; }
		80%, 100% { transform: scale(2); opacity: 0; }
	}
	:global(.animate-ping-slow) {
		animation: ping-slow 2.2s cubic-bezier(0, 0, 0.2, 1) infinite;
	}

	@media (prefers-reduced-motion: reduce) {
		.ambient-strip :global(.animate-ping),
		.ambient-strip :global(.animate-ping-slow),
		.ambient-strip :global(.animate-pulse) {
			animation: none !important;
		}
	}
</style>

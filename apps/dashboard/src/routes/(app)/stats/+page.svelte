<script lang="ts">
	import { onMount } from 'svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import Icon from '$components/Icon.svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteReceipt, RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import type { ConsolidationResult, HealthCheck, RetentionDistribution, SystemStats } from '$types';

	type VitalReceipt = RouteReceipt & {
		metric: string;
		rawValue: unknown;
		magnitude: number;
	};

	let stats = $state<SystemStats | null>(null);
	let retention = $state<RetentionDistribution | null>(null);
	let health = $state<HealthCheck | null>(null);
	let loading = $state(true);
	let error: string | null = $state(null);
	let consolidation = $state<ConsolidationResult | null>(null);
	let consolidating = $state(false);
	let actionError: string | null = $state(null);
	// Tracked when the user picks a vital. Selection only — no API call.
	// Consolidation runs only from the explicit labelled button in the DOM overlay.
	let selectedVitalId: string | null = $state(null);

	const totalMemories = $derived(health?.totalMemories ?? stats?.totalMemories ?? 0);
	const averageRetention = $derived(health?.averageRetention ?? stats?.averageRetention ?? 0);
	const embeddingCoverage = $derived(stats?.embeddingCoverage ?? 0);
	const dueForReview = $derived(stats?.dueForReview ?? 0);
	const distribution = $derived(retention?.distribution ?? []);
	const maxBucketCount = $derived(Math.max(1, ...distribution.map((bucket) => bucket.count)));
	const distributionTotal = $derived(distribution.reduce((sum, bucket) => sum + bucket.count, 0));
	const healthLabel = $derived(health?.status ?? 'unknown');

	const statsScene = $derived.by<RouteSceneModel>(() => {
		const receipts = stats ? buildReceipts(stats, consolidation) : [];
		const scalars = Object.fromEntries(receipts.map((r) => [r.metric, r.magnitude]));
		return {
			organ: 'stats',
			nodes: [],
			edges: [],
			events: [],
			receipts,
			scalars,
			alive: receipts.length > 0
		};
	});
	const selectedReceipt = $derived(
		statsScene.receipts.find((receipt) => `stats:${(receipt as VitalReceipt).metric}` === selectedVitalId) as
			| VitalReceipt
			| undefined
	);
	const consolidateDisabledReason = $derived(
		loading ? 'Waiting for live vitals' : !stats ? 'Vitals unavailable' : totalMemories === 0 ? 'No memories to consolidate' : null
	);

	onMount(() => {
		void loadStats();
	});

	async function loadStats() {
		loading = true;
		error = null;
		try {
			const [nextStats, nextRetention, nextHealth] = await Promise.all([
				api.stats(),
				api.retentionDistribution(),
				api.health()
			]);
			stats = nextStats;
			retention = nextRetention;
			health = nextHealth;
		} catch (err) {
			stats = null;
			retention = null;
			health = null;
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}

	// Dream ritual — absorbed from the retired /settings organ (which held no
	// settings, only duplicates of actions that belong with the vitals).
	let dreaming = $state(false);
	let dreamNotice = $state<string | null>(null);

	async function runDream() {
		if (dreaming || consolidating) return;
		dreaming = true;
		actionError = null;
		dreamNotice = null;
		try {
			const d = await api.dream();
			dreamNotice = `Dream complete: ${d.memoriesReplayed} memories replayed, ${d.stats?.newConnectionsFound ?? 0} new connections`;
			await loadStats();
		} catch (err) {
			actionError = err instanceof Error ? err.message : 'Dream cycle failed';
		} finally {
			dreaming = false;
		}
	}

	async function runConsolidate() {
		if (consolidating || consolidateDisabledReason) return;
		consolidating = true;
		actionError = null;
		try {
			consolidation = await api.consolidate();
			await loadStats();
		} catch (err) {
			actionError = err instanceof Error ? err.message : 'Consolidation failed';
		} finally {
			consolidating = false;
		}
	}

	function bucketColor(index: number): string {
		const progress = distribution.length <= 1 ? 1 : index / (distribution.length - 1);
		if (progress < 0.34) return '#ef4444';
		if (progress < 0.67) return '#f59e0b';
		return '#7dffb3';
	}

	function metricExplanation(metric: string): string {
		switch (metric) {
			case 'totalMemories':
				return 'The total durable memory population currently held by this Vestige brain.';
			case 'averageRetention':
				return 'The mean FSRS retrievability across memories; higher values indicate stronger expected recall.';
			case 'embeddingCoverage':
				return 'The share of memories with semantic embeddings available for similarity-aware recall.';
			case 'dueForReview':
				return 'Memories whose FSRS schedule says they are ready for maintenance in the next consolidation cycle.';
			default:
				return 'A live backend measurement represented by one of the breathing cells in the field.';
		}
	}

	function handleRoutePick(pick: RoutePick) {
		// Plain click on a vital must SELECT/INSPECT only — never mutate.
		// Consolidation is an expensive real FSRS mutation and lives behind the
		// EXPLICIT Consolidate memory action button in the overlay. Reading a
		// number here must not silently rewrite the whole memory system.
		if (pick.kind === 'stats-vital') {
			selectedVitalId = pick.id;
		}
	}

	function createStatsVitalsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		// The DOM overlay owns ALL vitals text (header + stat cards + retention chart),
		// so the canvas emits NO MSDF text — only the alive breathing field behind it.
		// Emitting in-canvas labels here would bleed a redundant "ghost" through the glass.
		const field = new StatsVitalsFieldPass(engine);
		field.uploadScene(scene);
		return [field];
	}

	/**
	 * Vitals as pulsing gauge-orbs: each real stat becomes a living cell whose
	 * radius + glow = its magnitude, hue by metric family. The whole set orbits
	 * as one galaxy so the vitals BREATHE instead of sitting as flat text.
	 */
	class StatsVitalsFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			// Text-heavy organ: the field is a DIM backdrop, not the star.
			this.field.setIntensity(0.22);
			// Vitals labels run down the left column (x=-0.82, y from +0.68 to -0.68).
			// Suppress the field there so every metric row stays legible.
			this.field.setReadingWell({ x: -0.5, y: 0, hw: 0.6, hh: 0.85, floor: 0.08, soft: 0.25 });
		}
		uploadScene(scene: RouteSceneModel): void {
			const receipts = scene.receipts as VitalReceipt[];
			const data: FieldDatum[] = receipts.map((r) => ({
				id: `stats:${r.metric}`,
				score: r.magnitude,
				hue: vitalHue(r.metric, r.magnitude),
				energy: 0.4 + 0.6 * r.magnitude,
				scar: (r.metric.includes('due') || r.metric.includes('Retention')) && r.magnitude < 0.4,
				kind: 'stats-vital',
				payload: r
			}));
			// Fewer, bigger orbs than a memory galaxy — vitals are gauges, not motes.
			this.field.setCells(layoutGalaxy(data, { maxRadius: 0.86, minCellR: 0.03, maxCellR: 0.11 }));
		}
		compute(encoder: GPUCommandEncoder): void {
			this.field.compute(encoder);
		}
		render(pass: GPURenderPassEncoder): void {
			this.field.render(pass);
		}
		pickAt(ndcX: number, ndcY: number): RoutePick | null {
			return this.field.pickAt(ndcX, ndcY);
		}
		dispose(): void {
			this.field.dispose();
		}
	}

	function vitalHue(metric: string, magnitude: number): [number, number, number] {
		if (metric.includes('due') || metric.includes('decay')) return magnitude > 0.4 ? FIELD_HUE.caution : FIELD_HUE.forward;
		if (metric.includes('Coverage') || metric.includes('Embeddings')) return magnitude > 0.7 ? FIELD_HUE.oxygen : FIELD_HUE.bridge;
		if (metric.includes('Retention') || metric.includes('Strength')) return magnitude < 0.4 ? FIELD_HUE.scarlet : FIELD_HUE.oxygen;
		return FIELD_HUE.recall;
	}

	function buildReceipts(currentStats: SystemStats, currentConsolidation: ConsolidationResult | null): VitalReceipt[] {
		const entries = Object.entries(currentStats);
		const numericValues = entries
			.map(([, value]) => (typeof value === 'number' && Number.isFinite(value) ? Math.abs(value) : null))
			.filter((value): value is number => value !== null);
		const maxNumeric = Math.max(1, ...numericValues);
		const receipts = entries.map(([metric, rawValue], index) => {
			const magnitude = metricMagnitude(metric, rawValue, maxNumeric);
			return makeReceipt(metric, rawValue, magnitude, index);
		});
		if (currentConsolidation) {
			for (const [metric, rawValue] of Object.entries(currentConsolidation)) {
				const prefixed = `consolidate.${metric}`;
				receipts.push(makeReceipt(prefixed, rawValue, metricMagnitude(prefixed, rawValue, maxNumeric), receipts.length));
			}
		}
		return receipts;
	}

	function makeReceipt(metric: string, rawValue: unknown, magnitude: number, index: number): VitalReceipt {
		return {
			source: {
				kind: 'scalar',
				id: metric,
				scalar: { name: metric, value: magnitude }
			},
			label: sanitizeAscii(`${metric} | ${formatValue(metric, rawValue)}`),
			nodeIndices: [],
			metric,
			rawValue,
			magnitude: clamp01(magnitude)
		};
	}

	function metricMagnitude(metric: string, value: unknown, maxNumeric: number): number {
		if (typeof value === 'number' && Number.isFinite(value)) {
			if (metric.includes('Coverage')) return clamp01(value > 1 ? value / 100 : value);
			if (metric.includes('average') || metric.includes('Retention') || metric.includes('Strength')) return clamp01(value);
			return clamp01(Math.log10(Math.abs(value) + 1) / Math.log10(maxNumeric + 1));
		}
		if (typeof value === 'string') {
			const parsed = Date.parse(value);
			if (Number.isFinite(parsed)) {
				const days = Math.max(0, (Date.now() - parsed) / 86_400_000);
				return clamp01(1 / (1 + days / 30));
			}
			return clamp01(value.length / 48);
		}
		return 0.5;
	}

	function formatValue(metric: string, value: unknown): string {
		if (typeof value === 'number' && Number.isFinite(value)) {
			if (metric.includes('Coverage')) return `${value.toFixed(value > 1 ? 0 : 2)}%`;
			if (metric.includes('average')) return `${(value * 100).toFixed(1)}%`;
			if (Number.isInteger(value)) return value.toLocaleString();
			return value.toFixed(3);
		}
		if (typeof value === 'string') {
			// Compact an ISO timestamp to "YYYY-MM-DD HH:MM" so the full value fits on
			// one line (the raw ISO string with ms + timezone overruns and truncates).
			const parsed = Date.parse(value);
			if (Number.isFinite(parsed)) return value.slice(0, 16).replace('T', ' ');
			return value;
		}
		return String(value);
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[\u2014\u2013]/g, '-')
			.replace(/[\u2018\u2019]/g, "'")
			.replace(/[\u201C\u201D]/g, '"')
			.replace(/\u2026/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}
</script>

<RouteStage
	organ="stats"
	seed={`stats-vitals:${stats?.totalMemories ?? 0}:${stats?.dueForReview ?? 0}:${stats?.averageRetention ?? 0}`}
	scene={statsScene}
	passes={createStatsVitalsPasses}
	emptyLabel=""
	loading={loading}
	error={error}
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full space-y-6 p-6 pointer-events-none">
	<div class="pointer-events-auto">
		<PageHeader
			icon="stats"
			title="System Vitals"
			subtitle="The live health of your memory: retention, coverage, and what is due."
			accent="recall"
		>
			<div class="flex items-center gap-2">
				<span class="inline-flex items-center gap-1.5 rounded-full border border-subtle/30 bg-deep/60 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wider text-dim">
					<span class="h-1.5 w-1.5 rounded-full {health?.status === 'healthy' ? 'bg-recall' : health?.status === 'empty' ? 'bg-muted' : 'bg-warning'}"></span>
					{healthLabel}
				</span>
				<div class="flex flex-col items-end gap-1">
					<button
						type="button"
						onclick={runConsolidate}
						disabled={consolidating || Boolean(consolidateDisabledReason)}
						class="inline-flex items-center gap-2 rounded-xl border border-synapse/35 bg-synapse/15 px-3.5 py-2 text-xs font-semibold text-synapse-glow transition hover:bg-synapse/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60 disabled:cursor-not-allowed disabled:opacity-45"
					>
						<Icon name="pulse" size={14} />
						{consolidating ? 'Consolidating…' : 'Consolidate memory'}
					</button>
					<button
						type="button"
						onclick={runDream}
						disabled={dreaming || consolidating}
						class="inline-flex items-center gap-2 rounded-xl border border-dream/35 bg-dream/15 px-3.5 py-2 text-xs font-semibold text-dream-glow transition hover:bg-dream/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream/60 disabled:cursor-not-allowed disabled:opacity-45"
					>
						<Icon name="dreams" size={14} />
						{dreaming ? 'Dreaming…' : 'Run dream cycle'}
					</button>
					{#if consolidateDisabledReason && !consolidating}
						<span class="text-[9px] text-muted">{consolidateDisabledReason}</span>
					{/if}
					{#if dreamNotice}
						<span class="text-[9px] text-recall">{dreamNotice}</span>
					{/if}
				</div>
			</div>
		</PageHeader>
	</div>

	{#if error}
		<section class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center" aria-live="polite">
			<div class="flex h-12 w-12 items-center justify-center rounded-2xl border border-decay/25 bg-decay/10 text-decay">
				<Icon name="pulse" size={24} />
			</div>
			<h2 class="text-sm font-semibold text-bright">System vitals are unavailable</h2>
			<p class="max-w-md text-xs text-muted">{error}</p>
			<button type="button" onclick={loadStats} class="mt-1 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60">
				Retry vitals
			</button>
		</section>
	{:else if loading}
		<div class="grid grid-cols-2 gap-3 lg:grid-cols-4 pointer-events-auto" aria-label="Loading system vitals">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-28 rounded-xl"></div>
			{/each}
		</div>
		<div class="glass-subtle shimmer h-[360px] rounded-2xl pointer-events-auto"></div>
	{:else}
		<section class="grid grid-cols-2 gap-3 lg:grid-cols-4 pointer-events-auto" aria-label="Live memory statistics">
			<button type="button" aria-pressed={selectedVitalId === 'stats:totalMemories'} onclick={() => (selectedVitalId = 'stats:totalMemories')} use:reveal={{ delay: 0, y: 12 }} class="glass rounded-xl p-5 text-left lift transition {selectedVitalId === 'stats:totalMemories' ? 'border-recall/50 bg-recall/10' : ''}">
				<div class="text-3xl font-bold tabular-nums text-bright"><AnimatedNumber value={totalMemories} /></div>
				<div class="mt-2 text-[11px] font-medium uppercase tracking-wider text-dim">Memories</div>
				<div class="mt-1 text-[10px] text-muted">Stored in the live brain</div>
			</button>
			<button type="button" aria-pressed={selectedVitalId === 'stats:averageRetention'} onclick={() => (selectedVitalId = 'stats:averageRetention')} use:reveal={{ delay: 60, y: 12 }} class="glass rounded-xl p-5 text-left lift transition {selectedVitalId === 'stats:averageRetention' ? 'border-recall/50 bg-recall/10' : ''}">
				<div class="text-3xl font-bold tabular-nums text-recall"><AnimatedNumber value={averageRetention * 100} decimals={1} /><span class="text-base">%</span></div>
				<div class="mt-2 text-[11px] font-medium uppercase tracking-wider text-dim">Avg retention</div>
				<div class="mt-1 text-[10px] text-muted">Current FSRS retrievability</div>
			</button>
			<button type="button" aria-pressed={selectedVitalId === 'stats:embeddingCoverage'} onclick={() => (selectedVitalId = 'stats:embeddingCoverage')} use:reveal={{ delay: 120, y: 12 }} class="glass rounded-xl p-5 text-left lift transition {selectedVitalId === 'stats:embeddingCoverage' ? 'border-synapse/50 bg-synapse/10' : ''}">
				<div class="text-3xl font-bold tabular-nums text-synapse-glow"><AnimatedNumber value={embeddingCoverage} decimals={1} /><span class="text-base">%</span></div>
				<div class="mt-2 text-[11px] font-medium uppercase tracking-wider text-dim">Embedding coverage</div>
				<div class="mt-1 text-[10px] text-muted">{stats?.withEmbeddings.toLocaleString() ?? 0} memories searchable by meaning</div>
			</button>
			<button type="button" aria-pressed={selectedVitalId === 'stats:dueForReview'} onclick={() => (selectedVitalId = 'stats:dueForReview')} use:reveal={{ delay: 180, y: 12 }} class="glass rounded-xl p-5 text-left lift transition {selectedVitalId === 'stats:dueForReview' ? 'border-warning/50 bg-warning/10' : ''}">
				<div class="text-3xl font-bold tabular-nums {dueForReview > 0 ? 'text-warning' : 'text-recall'}"><AnimatedNumber value={dueForReview} /></div>
				<div class="mt-2 text-[11px] font-medium uppercase tracking-wider text-dim">Due for review</div>
				<div class="mt-1 text-[10px] text-muted">Ready for the next consolidation cycle</div>
			</button>
		</section>

		<div class="glass pointer-events-auto flex min-h-16 items-center gap-3 rounded-xl border border-subtle/25 px-4 py-3">
			<div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-recall/10 text-recall">
				<Icon name={selectedReceipt ? 'pulse' : 'stats'} size={17} />
			</div>
			{#if selectedReceipt}
				<div class="min-w-0">
					<div class="text-[10px] font-medium uppercase tracking-wider text-muted">Selected vital · {selectedReceipt.metric}</div>
					<p class="mt-1 text-xs text-dim">{metricExplanation(selectedReceipt.metric)} Live value: <span class="font-medium text-bright">{formatValue(selectedReceipt.metric, selectedReceipt.rawValue)}</span>.</p>
				</div>
			{:else}
				<div>
					<div class="text-[10px] font-medium uppercase tracking-wider text-muted">What you're seeing</div>
					<p class="mt-1 text-xs text-dim">Select a stat card or a breathing field cell to inspect the live measurement. Selection never changes memory.</p>
				</div>
			{/if}
		</div>

		{#if actionError}
			<div class="glass pointer-events-auto flex items-center gap-2 rounded-xl border border-decay/25 px-4 py-3 text-xs text-decay" aria-live="polite">
				<Icon name="pulse" size={14} />
				Consolidation failed: {actionError}
			</div>
		{:else if consolidation}
			<div class="glass pointer-events-auto flex items-center gap-2 rounded-xl border border-recall/25 px-4 py-3 text-xs text-recall" aria-live="polite">
				<Icon name="sparkle" size={14} />
				Consolidation complete. Vitals and retention bands have been refreshed.
			</div>
		{/if}

		<section use:reveal={{ delay: 220, y: 16 }} class="glass-panel pointer-events-auto rounded-2xl p-5 lg:p-6">
			<div class="flex flex-wrap items-start justify-between gap-3">
				<div>
					<div class="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-bright">
						<Icon name="stats" size={15} />
						Retention distribution
					</div>
					<p class="mt-1.5 text-xs text-muted">Every memory grouped by its current probability of successful recall.</p>
				</div>
				<div class="text-right text-xs text-dim"><AnimatedNumber value={distributionTotal} /> memories measured</div>
			</div>

			{#if distribution.length === 0 || distributionTotal === 0}
				<div class="mt-6 flex min-h-56 flex-col items-center justify-center gap-3 rounded-xl border border-subtle/20 bg-deep/35 p-8 text-center">
					<div class="flex h-12 w-12 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall">
						<Icon name="stats" size={24} draw />
					</div>
					<h2 class="text-sm font-medium text-bright">No retention history yet</h2>
					<p class="max-w-sm text-xs text-muted">Add memories to build the first retention profile. The histogram will fill as Vestige learns their recall strength.</p>
					<button type="button" onclick={loadStats} class="rounded-lg bg-recall/15 px-4 py-2 text-xs font-medium text-recall transition hover:bg-recall/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60">
						Refresh vitals
					</button>
				</div>
			{:else}
				<div class="mt-7 grid min-h-64 grid-cols-[repeat(auto-fit,minmax(44px,1fr))] items-end gap-2" role="img" aria-label="Histogram of memory counts by retention range">
					{#each distribution as bucket, index (bucket.range)}
						<div class="group flex h-full min-w-0 flex-col items-center justify-end gap-2">
							<div class="text-xs font-semibold tabular-nums text-bright opacity-80 transition group-hover:opacity-100">{bucket.count.toLocaleString()}</div>
							<div class="relative flex h-44 w-full items-end overflow-hidden rounded-t-lg border border-subtle/20 bg-deep/45">
								<div
									class="w-full min-h-[3px] rounded-t-md transition-all duration-500 group-hover:brightness-125"
									style={`height: ${Math.max(2, (bucket.count / maxBucketCount) * 100)}%; background: ${bucketColor(index)}; box-shadow: 0 0 18px ${bucketColor(index)}55`}
								></div>
							</div>
							<div class="min-h-8 text-center text-[10px] leading-tight text-dim">{bucket.range}</div>
						</div>
					{/each}
				</div>
			{/if}

			<div class="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-subtle/20 pt-4 text-[10px] text-dim">
				<span class="font-medium uppercase tracking-wider text-muted">Legend</span>
				<span class="inline-flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-decay"></span>Fragile · review soon</span>
				<span class="inline-flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-warning"></span>Stabilizing</span>
				<span class="inline-flex items-center gap-1.5"><span class="h-2 w-2 rounded-full bg-recall"></span>Durable recall</span>
				<span class="ml-auto text-muted">Taller bars mean more memories in that retention band.</span>
			</div>
		</section>
	{/if}
</div>

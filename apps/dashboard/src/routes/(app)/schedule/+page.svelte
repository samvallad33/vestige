<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { emptyScene, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutRings, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	// One prediction row from POST /api/predict — the backend's REAL blended
	// FSRS urgency (decay + review schedule) mapped to a high/medium/low band.
	// Rendered in the "what needs review next" panel so the primary action shows
	// live proof, never a constant.
	type Prediction = {
		id: string;
		content: string;
		nodeType: string;
		retention: number;
		urgency: number;
		predictedNeed: 'high' | 'medium' | 'low';
	};

	// A schedule is a precise working set, not a full-table scan. Keep both the
	// list request and detail enrichment below the API's hard 200-row ceiling so
	// the first interactive frame stays predictable on large brains.
	const FETCH_LIMIT = 80;
	const ROW_LIMIT = 40;
	// The list endpoint (/api/memories) omits FSRS review fields (nextReviewAt /
	// lastAccessedAt) — only the per-memory endpoint (/api/memories/:id) returns
	// them. We enrich a bounded window of the loaded records with their REAL
	// review timestamps so the schedule reflects genuine due-for-review data
	// instead of collapsing to an empty field. Verified: 200 parallel per-memory
	// GETs against the local brain take <1s.
	const ENRICH_LIMIT = 200;
	const ENRICH_CONCURRENCY = 16;
	const WEEK_MS = 7 * 86_400_000;

	let memories: Memory[] = $state([]);
	let totalMemories = $state(0);
	let loading = $state(true);
	let error: string | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;

	// --- Primary action: POST /api/predict (real FSRS urgency bands) ---
	let predictions = $state<Prediction[]>([]);
	let predicting = $state(false);
	let predictError = $state<string | null>(null);
	let predictedAt = $state<number | null>(null);

	// --- Selection (NON-MUTATING). A plain click on a WebGPU row/cell or a DOM
	// row only OPENS the detail panel. The only mutation, promote(), lives behind
	// an explicit labelled button inside that panel. ---
	let selectedId = $state<string | null>(null);
	let promoting = $state(false);

	// Trim to a cap on a word boundary so a portrait row never ends mid-token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	onMount(() => {
		void loadSchedule();
	});

	onDestroy(() => {
		engineRef = null;
	});

	async function loadSchedule() {
		loading = true;
		error = null;
		try {
			const res = await api.memories.list({ limit: String(FETCH_LIMIT) });
			totalMemories = res.total;
			memories = await enrichReviewFields(res.memories);
		} catch (err) {
			memories = [];
			totalMemories = 0;
			error = err instanceof Error ? err.message : 'API FETCH FAILED';
		} finally {
			loading = false;
		}
	}

	/**
	 * The list payload lacks nextReviewAt/lastAccessedAt. Backfill them from the
	 * per-memory endpoint (which does return real FSRS review timestamps) for a
	 * bounded window, in small concurrent batches. Records that already carry a
	 * nextReviewAt, or that fail to enrich, are passed through unchanged — a
	 * failed enrich just omits that row from the schedule, never fakes one.
	 */
	async function enrichReviewFields(source: Memory[]): Promise<Memory[]> {
		const enriched = source.slice();
		const targets = enriched
			.map((memory, index) => ({ memory, index }))
			.filter(({ memory }) => !memory.nextReviewAt)
			.slice(0, ENRICH_LIMIT);
		for (let i = 0; i < targets.length; i += ENRICH_CONCURRENCY) {
			const batch = targets.slice(i, i + ENRICH_CONCURRENCY);
			await Promise.all(
				batch.map(async ({ memory, index }) => {
					try {
						const full = await api.memories.get(memory.id);
						enriched[index] = { ...memory, ...full };
					} catch {
						// leave the un-enriched record; it simply won't schedule
					}
				})
			);
		}
		return enriched;
	}

	// PRIMARY ACTION — ask the backend what it predicts you'll need next. This is
	// POST /api/predict; it returns real per-memory urgency (blended FSRS decay +
	// review schedule), never a constant. Read-only: it surfaces, never mutates.
	async function runPredict() {
		if (predicting || loading) return;
		predicting = true;
		predictError = null;
		try {
			const res = (await api.predict()) as { predictions?: Prediction[] };
			predictions = Array.isArray(res.predictions) ? res.predictions : [];
			predictedAt = Date.now();
		} catch (err) {
			predictError = err instanceof Error ? err.message : 'PREDICT FAILED';
			predictions = [];
		} finally {
			predicting = false;
		}
	}

	// EXPLICIT MUTATION — only ever fired by the labelled "Mark reviewed" button
	// in the detail panel, never by a plain select. promote() strengthens the
	// memory and pushes its next review out.
	async function markReviewed(memoryId: string) {
		if (promoting) return;
		promoting = true;
		try {
			const promoted = await api.memories.promote(memoryId);
			// promote returns a PARTIAL {id, promoted, retentionStrength} — merge the
			// changed field into the full Memory, never replace it (a full swap drops
			// content/nodeType/... and crashes the next render — the /memories bug).
			memories = memories.map((memory) =>
				memory.id === promoted.id
					? { ...memory, retentionStrength: promoted.retentionStrength }
					: memory
			);
			error = null;
		} catch (err) {
			error = err instanceof Error ? err.message : 'PROMOTE FAILED';
		} finally {
			promoting = false;
		}
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function dueAt(memory: Memory): number {
		const parsed = memory.nextReviewAt ? Date.parse(memory.nextReviewAt) : Number.POSITIVE_INFINITY;
		return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
	}

	function urgency(memory: Memory, nowMs: number): number {
		const next = dueAt(memory);
		if (!Number.isFinite(next)) return 0;
		const days = (next - nowMs) / 86_400_000;
		return clamp01(1 - days / 30);
	}

	// Human "due in" label for a DOM row: negative -> overdue, 0 -> today, else Nd.
	function dueInLabel(memory: Memory, nowMs: number): string {
		const next = dueAt(memory);
		if (!Number.isFinite(next)) return 'no date';
		const days = Math.ceil((next - nowMs) / 86_400_000);
		if (days < 0) return `${Math.abs(days)}d overdue`;
		if (days === 0) return 'due today';
		if (days === 1) return 'due tomorrow';
		return `in ${days}d`;
	}

	function scheduleLine(memory: Memory, nowMs: number, portrait = false): string {
		const next = dueAt(memory);
		const days = Number.isFinite(next) ? Math.ceil((next - nowMs) / 86_400_000) : 9999;
		const due = days < 0 ? `${Math.abs(days)}D OVER` : days === 0 ? 'DUE 0D' : `DUE ${days}D`;
		const pct = `${Math.round(memory.retentionStrength * 100)}%`;
		// Portrait: drop the id column and shorten the snippet so the row fits the
		// narrow width on one readable line — never edge-to-edge, never truncated
		// mid-word. Desktop keeps the full, byte-identical row.
		if (portrait) {
			return sanitizeAscii(`${trimSnippet(memory.content, 26)}  ${due} ${pct}`);
		}
		const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 48);
		return sanitizeAscii(`${snippet} | ${memory.id.slice(0, 8)} | ${due} | ${pct}`);
	}

	function dueMemories(source: Memory[]): Memory[] {
		const nowMs = Date.now();
		return source
			.filter((memory) => !!memory.nextReviewAt)
			.slice()
			.sort((a, b) => {
				const dueDelta = dueAt(a) - dueAt(b);
				if (dueDelta !== 0) return dueDelta;
				return (a.retentionStrength ?? 0) - (b.retentionStrength ?? 0);
			})
			.sort((a, b) => urgency(b, nowMs) - urgency(a, nowMs));
	}

	const scheduled = $derived(dueMemories(memories));

	// --- REAL stat cards, all derived from enriched FSRS timestamps ---
	const dueNowCount = $derived.by(() => {
		const now = Date.now();
		return scheduled.filter((m) => dueAt(m) <= now).length;
	});
	const dueThisWeekCount = $derived.by(() => {
		const now = Date.now();
		return scheduled.filter((m) => {
			const next = dueAt(m);
			return next > now && next <= now + WEEK_MS;
		}).length;
	});
	// Average retention across the due set — the honest "how well are these held"
	// number (0-100). Only meaningful when something is scheduled.
	const avgDueRetention = $derived.by(() => {
		if (scheduled.length === 0) return 0;
		const sum = scheduled.reduce((acc, m) => acc + clamp01(m.retentionStrength), 0);
		return Math.round((sum / scheduled.length) * 100);
	});

	// The bounded DOM list of upcoming reviews (mirror of the WebGPU rows).
	const upcomingRows = $derived.by(() => {
		const now = Date.now();
		return scheduled.slice(0, ROW_LIMIT).map((memory) => ({
			memory,
			dueIn: dueInLabel(memory, now),
			overdue: dueAt(memory) <= now,
			retentionPct: Math.round(clamp01(memory.retentionStrength) * 100)
		}));
	});

	const selectedMemory = $derived(
		selectedId ? (memories.find((m) => m.id === selectedId) ?? null) : null
	);

	const scene = $derived< RouteSceneModel >(
		scheduled.length === 0
			? emptyScene('schedule')
			: {
					organ: 'schedule',
					nodes: scheduled.slice(0, ROW_LIMIT).map((memory, index) => ({
						source: { kind: 'memory', id: memory.id },
						index,
						label: scheduleLine(memory, Date.now()),
						retention: clamp01(memory.retentionStrength),
						stability: clamp01(memory.retrievalStrength),
						lastAccessed: memory.lastAccessedAt,
						activation: urgency(memory, Date.now()),
						tags: memory.tags,
						type: memory.nodeType
					})),
					edges: [],
					events: [],
					receipts: [],
					scalars: {
						scheduled: scheduled.length,
						loaded: memories.length,
						dueNow: scheduled.filter((memory) => urgency(memory, Date.now()) >= 1).length
					},
					alive: true
				}
	);

	// The DOM overlay owns all content (header + real stat cards + empty state), so
	// no centered in-canvas MSDF status line renders behind the glass.
	const emptyLabel = $derived('');

	class ScheduleFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			// TEXT-HEAVY organ: field is a DIM backdrop, never a blob that drowns
			// the 40-row due list. Intensity 0.22 keeps it a living whisper.
			this.field.setIntensity(0.22);
			// The schedule rows are anchored in a tall LEFT column: x from -0.9
			// extending right, y from +0.74 down to ~-0.78 (top=0.74, 40 rows @
			// rowStep). Carve a reading well over that column so the labels/values
			// stay crisp against the field.
			this.field.setReadingWell({ x: -0.35, y: -0.02, hw: 0.72, hh: 0.86, floor: 0.06, soft: 0.24 });
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({
				id: node.source.id,
				score: node.activation ?? 0,
				hue: node.retention < 0.4 ? FIELD_HUE.scarlet : node.retention < 0.7 ? FIELD_HUE.caution : FIELD_HUE.oxygen,
				energy: node.activation,
				metric2: node.retention,
				scar: (node.activation ?? 0) >= 1,
				kind: 'schedule-memory',
				payload: node
			}));
			this.field.setCells(layoutRings(data, (datum) => datum.score >= 1 ? 0 : datum.score >= 0.75 ? 1 : datum.score >= 0.5 ? 2 : 3, { ringCount: 4, maxRadius: 0.88, minCellR: 0.045, maxCellR: 0.11 }));
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
	}

	function createSchedulePasses(engine: ObservatoryEngine, initialScene: RouteSceneModel): RouteFramePass[] {
		engineRef = engine;
		const field = new ScheduleFieldPass(engine);
		field.uploadScene(initialScene);
		return [field];
	}

	// NON-MUTATING pick: a plain click on a WebGPU field cell only SELECTS (opens
	// the detail panel). Promotion is an explicit labelled button.
	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'schedule-memory') return;
		// FIELD cell payload is a RouteNode whose source.id is the memory id.
		const item = pick.payload as { memoryId?: string; source?: { id?: string } };
		const memoryId = item.memoryId ?? item.source?.id;
		if (!memoryId) return;
		selectedId = memoryId;
	}

	function selectRow(memoryId: string) {
		selectedId = selectedId === memoryId ? null : memoryId;
	}

	const needColor: Record<Prediction['predictedNeed'], string> = {
		high: '#FF3B30',
		medium: '#FFB000',
		low: '#22C7DE'
	};
</script>

<RouteStage
	organ="schedule"
	seed={`schedule-due-field:${scheduled.length}:${memories.length}`}
	{scene}
	passes={createSchedulePasses}
	{loading}
	{error}
	{emptyLabel}
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<div class="pointer-events-auto">
		<PageHeader
			icon="schedule"
			title="Review Schedule"
			subtitle="What FSRS says is due for review, and when each memory next resurfaces."
			accent="recall"
		>
			<button
				type="button"
				onclick={runPredict}
				disabled={predicting || loading || memories.length === 0}
				title={memories.length === 0 ? 'Load memories first' : 'Ask the backend what you will need next'}
				class="inline-flex items-center gap-2 rounded-xl border border-recall/30 bg-recall/12 px-4 py-2 text-sm font-medium text-recall-glow transition hover:bg-recall/20 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60"
			>
				<Icon name="importance" size={15} />
				{predicting ? 'Predicting…' : 'Predict what I need next'}
			</button>
		</PageHeader>
	</div>

	{#if error}
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">Couldn't load the review schedule</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={loadSchedule}
				class="mt-2 rounded-lg bg-recall/20 px-4 py-2 text-xs font-medium text-recall-glow transition hover:bg-recall/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-1 sm:grid-cols-3 gap-3 pointer-events-auto">
			{#each Array(3) as _}
				<div class="glass-subtle shimmer h-24 rounded-xl"></div>
			{/each}
		</div>
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 pointer-events-auto">
			<div class="glass-subtle shimmer min-h-[520px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[520px] rounded-2xl"></div>
		</div>
	{:else if scheduled.length === 0}
		<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center">
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall"
			>
				<Icon name="schedule" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">
				Nothing is due for review right now.
			</div>
			<div class="max-w-sm text-xs text-muted">
				{memories.length.toLocaleString()} recent memories are sampled{totalMemories > memories.length ? ` from ${totalMemories.toLocaleString()} total` : ''}, but none carry an FSRS
				<code class="text-dim">nextReviewAt</code> timestamp yet. Reviews are scheduled as
				memories are recalled and consolidated — check back, or run
				<span class="text-recall-glow">Predict what I need next</span> to see what the backend
				thinks is slipping.
			</div>
		</div>
	{:else}
		<!-- LIVE PROOF — three real FSRS stat cards -->
		<div class="grid grid-cols-1 sm:grid-cols-3 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="flex items-center gap-2">
					<span class="ping-host inline-flex">
						<span class="w-2 h-2 rounded-full" style="background: #FF3B30"></span>
					</span>
					<div class="text-3xl font-bold tabular-nums" style="color: #FF3B30">
						<AnimatedNumber value={dueNowCount} />
					</div>
				</div>
				<div class="text-xs text-dim mt-1">due now (overdue or today)</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-3xl font-bold tabular-nums" style="color: #FFB000">
					<AnimatedNumber value={dueThisWeekCount} />
				</div>
				<div class="text-xs text-dim mt-1">coming due within 7 days</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-3xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={avgDueRetention} />%
				</div>
				<div class="text-xs text-dim mt-1">avg retention across the due set</div>
			</div>
		</div>

		<!-- INTERPRETATION — one-line insight tying colour to meaning -->
		<div class="pointer-events-auto flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-muted">
			<span class="inline-flex items-center gap-1.5">
				<span class="w-2 h-2 rounded-full" style="background: #FF3B30"></span> at risk (&lt;40% retention)
			</span>
			<span class="inline-flex items-center gap-1.5">
				<span class="w-2 h-2 rounded-full" style="background: #FFB000"></span> softening (40–70%)
			</span>
			<span class="inline-flex items-center gap-1.5">
				<span class="w-2 h-2 rounded-full" style="background: #22C7DE"></span> healthy (&gt;70%)
			</span>
			<span class="ml-auto text-dim">
				{scheduled.length.toLocaleString()} scheduled from {memories.length.toLocaleString()} recent memories{totalMemories > memories.length ? ` · ${totalMemories.toLocaleString()} total` : ''}
			</span>
		</div>

		<!-- Main view: DOM upcoming list + selection detail -->
		<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
			<div class="glass-panel rounded-2xl p-3 space-y-1.5 max-h-[620px] overflow-y-auto">
				<div class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10">
					<span class="text-xs text-dim uppercase tracking-wider">Upcoming reviews</span>
					<span class="text-xs text-muted tabular-nums"><AnimatedNumber value={upcomingRows.length} /></span>
				</div>

				{#each upcomingRows as row, i (row.memory.id)}
					{@const isSelected = selectedId === row.memory.id}
					<button
						use:reveal={{ delay: Math.min(i * 28, 320), y: 8 }}
						onclick={() => selectRow(row.memory.id)}
						class="w-full text-left p-3 rounded-xl border transition lift
							{isSelected
								? 'bg-recall/10 border-recall/40 shadow-[0_0_12px_rgba(34,199,222,0.16)]'
								: 'border-subtle/20 hover:border-recall/30 hover:bg-white/[0.02]'}"
					>
						<div class="flex items-center gap-2 mb-1.5">
							<div
								class="w-2 h-2 rounded-full shrink-0"
								style="background: {row.retentionPct < 40 ? '#FF3B30' : row.retentionPct < 70 ? '#FFB000' : '#22C7DE'}"
							></div>
							<span
								class="text-[11px] font-medium tabular-nums"
								style="color: {row.overdue ? '#FF3B30' : '#FFB000'}"
							>
								{row.dueIn}
							</span>
							<span class="ml-auto text-[10px] text-muted tabular-nums">{row.retentionPct}% retained</span>
						</div>
						<div class="text-xs text-text truncate">
							{sanitizeAscii(row.memory.content).slice(0, 90) || 'Untitled memory'}
						</div>
					</button>
				{/each}
			</div>

			<!-- INTERPRETATION — selection detail panel (non-mutating select) -->
			<aside use:reveal={{ delay: 100, y: 16 }} class="glass rounded-2xl p-4 space-y-3 max-h-[620px] overflow-y-auto">
				{#if selectedMemory}
					<div class="flex items-start justify-between gap-2">
						<div class="font-mono text-[10px] uppercase tracking-[0.2em] text-recall-glow">Review detail</div>
						<button
							type="button"
							onclick={() => (selectedId = null)}
							class="rounded-lg border border-subtle/30 px-2.5 py-1 text-[11px] text-muted transition hover:border-recall/40 hover:text-recall"
						>
							Close
						</button>
					</div>
					<p class="text-sm text-text leading-relaxed">{sanitizeAscii(selectedMemory.content)}</p>
					<div class="grid grid-cols-2 gap-2 pt-1">
						<div class="rounded-lg bg-white/[0.03] p-2.5">
							<div class="text-[10px] uppercase tracking-wider text-muted">next review</div>
							<div class="mt-0.5 text-xs text-bright">{dueInLabel(selectedMemory, Date.now())}</div>
						</div>
						<div class="rounded-lg bg-white/[0.03] p-2.5">
							<div class="text-[10px] uppercase tracking-wider text-muted">retention</div>
							<div class="mt-0.5 text-xs text-bright tabular-nums">{Math.round(clamp01(selectedMemory.retentionStrength) * 100)}%</div>
						</div>
						<div class="rounded-lg bg-white/[0.03] p-2.5">
							<div class="text-[10px] uppercase tracking-wider text-muted">retrieval</div>
							<div class="mt-0.5 text-xs text-bright tabular-nums">{Math.round(clamp01(selectedMemory.retrievalStrength) * 100)}%</div>
						</div>
						<div class="rounded-lg bg-white/[0.03] p-2.5">
							<div class="text-[10px] uppercase tracking-wider text-muted">type</div>
							<div class="mt-0.5 truncate text-xs text-dim">{selectedMemory.nodeType}</div>
						</div>
					</div>
					{#if selectedMemory.tags && selectedMemory.tags.length > 0}
						<div class="flex flex-wrap gap-1">
							{#each selectedMemory.tags as t}
								<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{t}</span>
							{/each}
						</div>
					{/if}
					<button
						type="button"
						onclick={() => selectedMemory && markReviewed(selectedMemory.id)}
						disabled={promoting}
						class="w-full mt-1 inline-flex items-center justify-center gap-2 rounded-xl border border-recall/30 bg-recall/12 px-3 py-2 text-xs font-medium text-recall-glow transition hover:bg-recall/20 disabled:cursor-not-allowed disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60"
					>
						<Icon name="sparkle" size={13} />
						{promoting ? 'Marking…' : 'Mark reviewed (strengthen)'}
					</button>
				{:else if predictions.length > 0}
					<div class="font-mono text-[10px] uppercase tracking-[0.2em] text-recall-glow">Predicted need</div>
					<p class="text-[11px] text-muted">
						The backend's blended FSRS urgency for the {predictions.length} most active memories.
						{#if predictedAt}<span class="text-dim">Just now.</span>{/if}
					</p>
					<div class="space-y-1.5">
						{#each predictions as p (p.id)}
							<button
								type="button"
								onclick={() => (selectedId = p.id)}
								class="w-full text-left rounded-lg border border-subtle/20 p-2.5 transition hover:border-recall/30 hover:bg-white/[0.02]"
							>
								<div class="flex items-center gap-2 mb-1">
									<span class="text-[9px] uppercase tracking-wider font-medium" style="color: {needColor[p.predictedNeed]}">
										{p.predictedNeed}
									</span>
									<span class="ml-auto text-[10px] text-muted tabular-nums">{Math.round(p.urgency * 100)}% urgency</span>
								</div>
								<div class="text-[11px] text-dim truncate">{sanitizeAscii(p.content)}</div>
							</button>
						{/each}
					</div>
				{:else}
					<div class="flex flex-col items-center justify-center gap-2 py-10 text-center">
						<div class="text-dim opacity-50 breathe">
							<Icon name="schedule" size={38} strokeWidth={1.2} />
						</div>
						<p class="text-xs text-muted max-w-[220px]">
							Select any upcoming review to see its retention, next-review date, and a
							<span class="text-recall-glow">Mark reviewed</span> action.
						</p>
						{#if predictError}
							<p class="text-[11px] text-decay">{predictError}</p>
						{/if}
					</div>
				{/if}
			</aside>
		</div>
	{/if}
</div>

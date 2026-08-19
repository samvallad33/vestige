<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import { api } from '$stores/api';
	import type { ImportanceScore, Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { emptyScene, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { reveal } from '$lib/actions/reveal';

	// ── Real data shapes ────────────────────────────────────────────────────
	// api.importance(content) -> { composite, channels{novelty,arousal,reward,attention}, recommendation }
	// api.predict()           -> { predictions: [{ id, retention, urgency, predictedNeed }], basedOn }
	// api.memories.list       -> { total, memories[] }
	type PredictedNeed = 'high' | 'medium' | 'low';
	interface Prediction {
		id: string;
		urgency: number;
		predictedNeed: PredictedNeed;
		retention: number;
	}
	interface ImportanceRecord {
		memory: Memory;
		score: ImportanceScore;
		/** Real FSRS urgency from api.predict() when the memory is in the recent window; else null. */
		urgency: number | null;
		predictedNeed: PredictedNeed | null;
	}

	const MEMORY_LIMIT = 36;
	const SCORE_CONCURRENCY = 6;

	let records = $state<ImportanceRecord[]>([]);
	let total = $state(0);
	let loading = $state(true);
	let refreshing = $state(false);
	let error = $state<string | null>(null);
	let promotingId = $state<string | null>(null);
	let selectedId = $state<string | null>(null);
	let engineRef: ObservatoryEngine | null = null;
	let fieldPass: LivingFieldPass | null = null;

	// ── Filters (control only — never mutates data) ─────────────────────────
	type Lens = 'all' | 'save' | 'skip' | 'urgent';
	let lens = $state<Lens>('all');
	const lensOptions: DropdownOption[] = [
		{ value: 'all', label: 'All memories', icon: 'importance' },
		{ value: 'save', label: 'Recommended: keep', icon: 'sparkle' },
		{ value: 'skip', label: 'Recommended: skip', icon: 'filter' },
		{ value: 'urgent', label: 'Urgent (FSRS due)', icon: 'schedule' }
	];

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
	}

	onMount(() => {
		void load();
	});

	onDestroy(() => {
		fieldPass?.dispose();
		fieldPass = null;
		engineRef = null;
	});

	async function load() {
		loading = records.length === 0;
		refreshing = records.length > 0;
		error = null;
		try {
			// Two real endpoints in parallel: the memory window we rank, and the
			// FSRS prediction pass that gives each recent memory its real urgency.
			const [list, prediction] = await Promise.all([
				api.memories.list({ limit: String(MEMORY_LIMIT) }),
				api.predict().catch(() => null)
			]);
			total = list.total;

			const predById = new Map<string, Prediction>();
			if (prediction && Array.isArray((prediction as { predictions?: unknown }).predictions)) {
				for (const raw of (prediction as { predictions: Prediction[] }).predictions) {
					if (raw && typeof raw.id === 'string') {
						predById.set(raw.id, {
							id: raw.id,
							urgency: clamp01(Number(raw.urgency)),
							predictedNeed: (raw.predictedNeed as PredictedNeed) ?? 'low',
							retention: clamp01(Number(raw.retention))
						});
					}
				}
			}

			// Real 4-channel salience score per memory (composite/novelty/arousal/…).
			const scored = await scoreWindow(list.memories, predById);
			records = scored
				.filter((r): r is PromiseFulfilledResult<ImportanceRecord> => r.status === 'fulfilled')
				.map((r) => r.value)
				.sort((a, b) => b.score.composite - a.score.composite);

			if (list.memories.length > 0 && records.length === 0) {
				error = 'The importance model returned no scores for this memory window.';
			}
			fieldPass?.setCells(buildFieldCells());
		} catch (err) {
			records = [];
			total = 0;
			error = err instanceof Error ? err.message : 'Failed to score importance.';
		} finally {
			loading = false;
			refreshing = false;
			engineRef?.demoClock.reset();
		}
	}

	/** Keep per-memory salience calls bounded instead of stampeding the API on refresh. */
	async function scoreWindow(memories: Memory[], predById: Map<string, Prediction>): Promise<PromiseSettledResult<ImportanceRecord>[]> {
		const settled: PromiseSettledResult<ImportanceRecord>[] = [];
		for (let i = 0; i < memories.length; i += SCORE_CONCURRENCY) {
			const batch = memories.slice(i, i + SCORE_CONCURRENCY);
			const batchResults = await Promise.allSettled(batch.map(async (memory) => {
				const score = await api.importance(memory.content);
				const pred = predById.get(memory.id) ?? null;
				return {
					memory,
					score,
					urgency: pred ? pred.urgency : null,
					predictedNeed: pred ? pred.predictedNeed : null
				} satisfies ImportanceRecord;
			}));
			settled.push(...batchResults);
		}
		return settled;
	}

	// ── Derived stats (all REAL — every value moves with the data) ───────────
	const scoredCount = $derived(records.length);
	const keepCount = $derived(records.filter((r) => r.score.recommendation === 'save').length);
	const avgComposite = $derived(
		records.length === 0
			? 0
			: records.reduce((sum, r) => sum + clamp01(r.score.composite), 0) / records.length
	);
	const urgentCount = $derived(
		records.filter((r) => r.predictedNeed === 'high' || (r.urgency ?? 0) >= 0.66).length
	);

	const filtered = $derived.by<ImportanceRecord[]>(() => {
		switch (lens) {
			case 'save':
				return records.filter((r) => r.score.recommendation === 'save');
			case 'skip':
				return records.filter((r) => r.score.recommendation === 'skip');
			case 'urgent':
				return records.filter((r) => r.predictedNeed === 'high' || (r.urgency ?? 0) >= 0.66);
			case 'all':
			default:
				return records;
		}
	});

	const selected = $derived(records.find((r) => r.memory.id === selectedId) ?? null);

	// ── Presentation helpers ────────────────────────────────────────────────
	function pct(v: number): string {
		return `${Math.round(clamp01(v) * 100)}%`;
	}
	function snippet(text: string, cap = 96): string {
		const s = (text ?? '').replace(/\s+/g, ' ').trim();
		return s.length > cap ? s.slice(0, cap - 1) + '…' : s;
	}
	function strongestChannel(score: ImportanceScore): string {
		return (Object.entries(score.channels) as [string, number][]).sort((a, b) => b[1] - a[1])[0][0];
	}
	function needColor(rec: ImportanceRecord): string {
		if (rec.predictedNeed === 'high' || (rec.urgency ?? 0) >= 0.66) return '#ef4444';
		if (rec.predictedNeed === 'medium' || (rec.urgency ?? 0) >= 0.33) return '#f59e0b';
		return '#10b981';
	}
	function needLabel(rec: ImportanceRecord): string {
		if (rec.urgency == null) return 'no FSRS signal';
		if (rec.predictedNeed === 'high' || rec.urgency >= 0.66) return 'urgent';
		if (rec.predictedNeed === 'medium' || rec.urgency >= 0.33) return 'soon';
		return 'stable';
	}

	// Selection is NON-MUTATING: a click only opens the detail panel.
	function selectRecord(id: string) {
		selectedId = selectedId === id ? null : id;
	}

	// The ONLY mutation — an explicit, labelled Promote button.
	async function promote(id: string, ev?: Event) {
		ev?.stopPropagation();
		if (promotingId) return;
		promotingId = id;
		try {
			const promoted = await api.memories.promote(id);
			// promote returns a PARTIAL payload ({ id, promoted, retentionStrength }).
			// Merge onto the existing memory so content/createdAt survive.
			records = records.map((r) =>
				r.memory.id === promoted.id
					? {
							...r,
							memory: {
								...r.memory,
								retentionStrength: promoted.retentionStrength ?? r.memory.retentionStrength
							}
						}
					: r
			);
			fieldPass?.setCells(buildFieldCells());
		} catch (err) {
			error = err instanceof Error ? err.message : 'Promote failed.';
		} finally {
			promotingId = null;
		}
	}

	// ── WebGPU field (the ALIVE backdrop — kept exactly, DOM reads on top) ────
	function buildFieldCells() {
		const data: FieldDatum[] = records.map((record) => ({
			id: record.memory.id,
			score: clamp01(record.score.composite),
			hue: record.score.recommendation === 'save' ? FIELD_HUE.oxygen : FIELD_HUE.caution,
			energy: Math.max(0.35, clamp01(record.score.composite)),
			metric2: clamp01(record.memory.retentionStrength),
			selected: record.memory.id === selectedId,
			kind: 'importance',
			payload: record
		}));
		return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.04, maxCellR: 0.1 });
	}

	const scene = $derived<RouteSceneModel>(
		records.length === 0
			? emptyScene('importance')
			: {
					organ: 'importance',
					nodes: records.map((record, index) => ({
						source: { kind: 'memory', id: record.memory.id },
						index,
						label: snippet(record.memory.content, 44),
						retention: clamp01(record.memory.retentionStrength),
						activation: clamp01(record.score.composite),
						tags: record.memory.tags,
						type: record.memory.nodeType
					})),
					edges: [],
					events: [],
					receipts: [],
					scalars: {
						scored: records.length,
						keep: keepCount,
						avgComposite,
						urgent: urgentCount
					},
					alive: true
				}
	);

	function createImportancePasses(engine: ObservatoryEngine): RouteFramePass[] {
		engineRef = engine;
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		// Text-heavy DOM organ: the field is a DIM living backdrop. The DOM glass
		// panels carry the reading, so carve a broad reading well and keep it low.
		field.setIntensity(0.24);
		field.setReadingWell({ x: 0, y: -0.02, hw: 0.7, hh: 0.86, floor: 0.08, soft: 0.24 });
		field.setCells(buildFieldCells());
		return [
			{
				compute: (encoder) => field.compute(encoder),
				render: (pass) => field.render(pass),
				pickAt: (x, y) => field.pickAt(x, y),
				dispose: () => {
					field.dispose();
					if (fieldPass === field) fieldPass = null;
				}
			}
		];
	}

	// A click on a glowing field orb selects that memory (non-mutating).
	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'importance') return;
		const rec = pick.payload as { memory?: { id?: string } };
		const id = rec.memory?.id;
		if (id) selectRecord(id);
	}
</script>

<svelte:head>
	<title>Importance & Salience · Vestige</title>
</svelte:head>

<RouteStage
	organ="importance"
	seed={`real-importance-field:${total}:${scoredCount}`}
	{scene}
	passes={createImportancePasses}
	{loading}
	{error}
	emptyLabel="EMPTY IMPORTANCE FIELD - INGEST MEMORIES TO RANK"
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<div class="pointer-events-auto">
		<PageHeader
			icon="importance"
			title="Importance & Salience"
			subtitle="Which memories matter most right now, ranked by real FSRS urgency and salience."
			accent="warning"
		>
			<button
				type="button"
				onclick={() => load()}
				disabled={loading || refreshing}
				title={loading || refreshing ? 'Scoring in progress…' : `Re-score the newest ${MEMORY_LIMIT} memories`}
				class="inline-flex items-center gap-1.5 rounded-lg bg-warning/15 px-3.5 py-2 text-xs font-medium text-warning transition hover:bg-warning/25 disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/60"
			>
				<Icon name="pulse" size={14} />
				{refreshing ? 'Re-scoring…' : 'Refresh ranking'}
			</button>
		</PageHeader>
	</div>

	{#if error}
		<div
			class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"
		>
			<div class="text-sm text-decay">Couldn't score importance</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={() => load()}
				class="mt-2 rounded-lg bg-warning/15 px-4 py-2 text-xs font-medium text-warning transition hover:bg-warning/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-24 rounded-xl"></div>
			{/each}
		</div>
		<div class="glass-subtle shimmer min-h-[520px] rounded-2xl pointer-events-auto"></div>
	{:else if records.length === 0}
		<div
			class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"
		>
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-warning/25 bg-warning/10 text-warning"
			>
				<Icon name="importance" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">No memories to rank yet.</div>
			<div class="max-w-sm text-xs text-muted">
				Salience ranking appears once your brain holds memories. Ingest a few via the MCP tools or
				the CLI, then hit <span class="text-warning">Refresh ranking</span>.
			</div>
		</div>
	{:else}
		<!-- LIVE PROOF: 4 real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={scoredCount} />
				</div>
				<div class="text-xs text-dim mt-1">newest {scoredCount.toLocaleString()} ranked of {total.toLocaleString()} total</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #10b981">
					<AnimatedNumber value={keepCount} />
				</div>
				<div class="text-xs text-dim mt-1">model recommends keeping</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #f59e0b">
					<AnimatedNumber value={Math.round(avgComposite * 100)} suffix="%" />
				</div>
				<div class="text-xs text-dim mt-1">average composite salience</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="flex items-center gap-2">
					<span class="ping-host inline-flex">
						<span class="w-2 h-2 rounded-full" style="background: #ef4444"></span>
					</span>
					<div class="text-2xl font-bold tabular-nums" style="color: #ef4444">
						<AnimatedNumber value={urgentCount} />
					</div>
				</div>
				<div class="text-xs text-dim mt-1">urgent by FSRS need</div>
			</div>
		</div>

		<!-- INTERPRETATION: legend explaining every metric -->
		<div
			use:reveal={{ delay: 200, y: 10 }}
			class="glass-subtle pointer-events-auto rounded-xl p-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-dim"
		>
			<span class="text-muted uppercase tracking-wider text-[10px]">Reading the columns</span>
			<span><span class="text-bright font-medium">Composite</span> — blended 4-channel salience</span>
			<span
				><span class="text-bright font-medium">Retention</span> — FSRS recall probability now</span
			>
			<span><span class="text-bright font-medium">Urgency</span> — how soon it's needed / due</span>
			<span
				class="inline-flex items-center gap-1.5"
				><span class="w-2 h-2 rounded-full" style="background:#10b981"></span>keep
				<span class="w-2 h-2 rounded-full ml-2" style="background:#f59e0b"></span>skip — model
				recommendation</span
			>
		</div>

		<!-- Filter lens -->
		<div class="flex flex-wrap items-end gap-3 enter pointer-events-auto">
			<Dropdown options={lensOptions} value={lens} label="Lens" icon="filter" onChange={(v) => (lens = v as Lens)} />
			<span class="text-dim text-xs tabular-nums ml-auto">
				<AnimatedNumber value={filtered.length} /> in view
			</span>
		</div>

		<!-- Ranked DOM list -->
		<div
			use:reveal={{ delay: 120, y: 16 }}
			class="glass-panel pointer-events-auto rounded-2xl overflow-hidden"
		>
			<!-- COLUMN HEADERS -->
			<div
				class="grid grid-cols-[2.2rem_1fr_5rem_5rem_6.5rem_5.5rem] gap-3 items-center px-4 py-2.5 border-b border-subtle/25 bg-white/[0.02] text-[10px] uppercase tracking-wider text-muted"
			>
				<span>#</span>
				<span>Memory</span>
				<span class="text-right">Composite</span>
				<span class="text-right">Retention</span>
				<span class="text-right">Urgency</span>
				<span class="text-right">Action</span>
			</div>

			{#if filtered.length === 0}
				<div class="flex flex-col items-center gap-2 p-10 text-center">
					<Icon name="filter" size={28} strokeWidth={1.3} />
					<p class="text-dim text-sm">No memories match this lens.</p>
				</div>
			{:else}
				<div class="max-h-[560px] overflow-y-auto divide-y divide-subtle/15">
					{#each filtered as rec, i (rec.memory.id)}
						{@const isSel = selectedId === rec.memory.id}
						<!-- Plain click = SELECT (non-mutating). Promote is the labelled button only. -->
						<div
							role="button"
							tabindex="0"
							onclick={() => selectRecord(rec.memory.id)}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									selectRecord(rec.memory.id);
								}
							}}
							class="grid grid-cols-[2.2rem_1fr_5rem_5rem_6.5rem_5.5rem] gap-3 items-center px-4 py-3 text-sm cursor-pointer transition {isSel
								? 'bg-warning/10'
								: 'hover:bg-white/[0.03]'}"
						>
							<span class="text-muted tabular-nums text-xs">{i + 1}</span>
							<div class="min-w-0">
								<div class="truncate text-text">{snippet(rec.memory.content, 88)}</div>
								<div class="flex items-center gap-2 mt-0.5 text-[10px] text-muted">
									<span class="font-mono">{rec.memory.id.slice(0, 8)}</span>
									<span>· {rec.memory.nodeType}</span>
									<span
										class="px-1.5 py-0.5 rounded"
										style="background: {rec.score.recommendation === 'save'
											? 'rgba(16,185,129,0.14)'
											: 'rgba(245,158,11,0.14)'}; color: {rec.score.recommendation === 'save'
											? '#10b981'
											: '#f59e0b'}"
									>
										{rec.score.recommendation === 'save' ? 'keep' : 'skip'}
									</span>
								</div>
							</div>
							<span class="text-right tabular-nums font-medium text-bright">{pct(rec.score.composite)}</span
							>
							<span class="text-right tabular-nums text-dim">{pct(rec.memory.retentionStrength)}</span>
							<span
								class="text-right tabular-nums font-medium inline-flex items-center justify-end gap-1.5"
								style="color: {needColor(rec)}"
							>
								<span class="w-1.5 h-1.5 rounded-full" style="background: {needColor(rec)}"></span>
								{rec.urgency == null ? '—' : pct(rec.urgency)}
							</span>
							<button
								type="button"
								onclick={(e) => promote(rec.memory.id, e)}
								disabled={promotingId === rec.memory.id}
								class="justify-self-end inline-flex items-center gap-1 rounded-lg border border-warning/30 px-2.5 py-1.5 text-[11px] font-medium text-warning transition hover:bg-warning/15 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-warning/60"
							>
								{promotingId === rec.memory.id ? '…' : 'Promote'}
							</button>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<!-- INTERPRETATION: selection detail panel (non-mutating) -->
		{#if selected}
			<section
				use:reveal={{ y: 12 }}
				class="glass-panel pointer-events-auto rounded-2xl p-5 border-warning/30 shadow-[0_0_40px_rgba(245,158,11,0.12)]"
			>
				<div class="flex flex-wrap items-start justify-between gap-3 border-b border-warning/20 pb-3">
					<div class="min-w-0">
						<div class="font-mono text-[10px] uppercase tracking-[0.22em] text-warning">
							Salience receipt
						</div>
						<div class="mt-1 font-mono text-xs text-muted">{selected.memory.id}</div>
					</div>
					<div class="flex items-center gap-2 shrink-0">
						<button
							type="button"
							onclick={(e) => promote(selected.memory.id, e)}
							disabled={promotingId === selected.memory.id}
							class="inline-flex items-center gap-1.5 rounded-lg bg-warning/15 px-3 py-1.5 text-xs font-medium text-warning transition hover:bg-warning/25 disabled:opacity-50"
						>
							<Icon name="importance" size={13} />
							{promotingId === selected.memory.id ? 'Promoting…' : 'Promote'}
						</button>
						<button
							type="button"
							onclick={() => (selectedId = null)}
							class="rounded-lg border border-subtle/30 px-3 py-1.5 text-xs text-muted transition hover:border-warning/40 hover:text-warning"
						>
							Close
						</button>
					</div>
				</div>

				<p class="mt-4 text-sm text-text leading-relaxed">{selected.memory.content}</p>

				{#if selected.memory.tags && selected.memory.tags.length > 0}
					<div class="mt-3 flex flex-wrap gap-1.5">
						{#each selected.memory.tags as tag}
							<span class="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{tag}</span>
						{/each}
					</div>
				{/if}

				<!-- 4-channel salience breakdown (all real) -->
				<div class="mt-4 grid gap-3 md:grid-cols-4">
					{#each Object.entries(selected.score.channels) as [name, value]}
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">{name}</div>
							<div class="mt-1 font-mono text-lg text-bright">{pct(value)}</div>
							<div class="mt-1.5 h-1 rounded-full bg-white/[0.06] overflow-hidden">
								<div
									class="h-full rounded-full"
									style="width: {pct(value)}; background: {name === strongestChannel(selected.score)
										? '#f59e0b'
										: 'rgba(245,158,11,0.4)'}"
								></div>
							</div>
						</div>
					{/each}
				</div>

				<div class="mt-4 grid gap-3 md:grid-cols-4">
					<div class="rounded-xl bg-white/[0.03] p-3">
						<div class="text-[10px] uppercase tracking-wider text-muted">composite</div>
						<div class="mt-1 font-mono text-lg text-warning">{pct(selected.score.composite)}</div>
					</div>
					<div class="rounded-xl bg-white/[0.03] p-3">
						<div class="text-[10px] uppercase tracking-wider text-muted">retention</div>
						<div class="mt-1 font-mono text-lg text-recall">{pct(selected.memory.retentionStrength)}</div>
					</div>
					<div class="rounded-xl bg-white/[0.03] p-3">
						<div class="text-[10px] uppercase tracking-wider text-muted">FSRS urgency</div>
						<div class="mt-1 font-mono text-lg" style="color: {needColor(selected)}">
							{selected.urgency == null ? '—' : pct(selected.urgency)}
						</div>
						<div class="mt-0.5 text-[10px]" style="color: {needColor(selected)}">
							{needLabel(selected)}
						</div>
					</div>
					<div class="rounded-xl bg-white/[0.03] p-3">
						<div class="text-[10px] uppercase tracking-wider text-muted">recommendation</div>
						<div
							class="mt-1 font-mono text-lg"
							style="color: {selected.score.recommendation === 'save' ? '#10b981' : '#f59e0b'}"
						>
							{selected.score.recommendation === 'save' ? 'keep' : 'skip'}
						</div>
						<div class="mt-0.5 text-[10px] text-muted">
							top channel: {strongestChannel(selected.score)}
						</div>
					</div>
				</div>
			</section>
		{/if}
	{/if}
</div>

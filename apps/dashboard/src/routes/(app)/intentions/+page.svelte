<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { IntentionItem } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';

	type RichIntention = IntentionItem & {
		retention?: number;
		retentionStrength?: number;
		confidence?: number;
		reminder_count?: number;
		tags?: string[];
		related_memories?: string[];
	};
	type PredictedIntent = {
		id: string;
		content: string;
		nodeType: string;
		predictedNeed: string;
		retention: number;
		urgency?: number;
	};

	const ACTIVE_FILTER = 'active';
	const ALL_FILTER = 'all';

	let intentions: RichIntention[] = $state([]);
	let predictions: PredictedIntent[] = $state([]);
	let total = $state(0);
	let filter = $state(ACTIVE_FILTER);
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedIntentionId: string | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;

	// Live viewport aspect (canvas px) from engine.params[6]/[7], with a window
	// fallback for the pre-frame-0 pass. Drives the field's portrait/desktop tuning.
	// NEVER a hardcoded phone width; desktop (aspect>=0.85) is untouched.
	function viewportAspect(): number {
		let vw = engineRef?.params[6] || 0;
		let vh = engineRef?.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return 1;
		return vw / vh;
	}

	onMount(() => {
		void loadIntentions(ACTIVE_FILTER);
	});

	onDestroy(() => {
		engineRef = null;
	});

	async function loadIntentions(nextFilter = filter) {
		filter = nextFilter;
		loading = true;
		error = null;
		try {
			const [res, predictionRes] = await Promise.all([api.intentions(nextFilter), api.predict()]);
			intentions = (res.intentions || []) as RichIntention[];
			predictions = (predictionRes.predictions ?? []) as PredictedIntent[];
			total = res.total ?? intentions.length;
		} catch (err) {
			intentions = [];
			predictions = [];
			total = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN INTENTION FETCH ERROR';
		} finally {
			loading = false;
		}
	}

	// Explicit, labelled control that swaps the filter. Kept as its own function so
	// the DOM dropdown and the in-field toggle share one non-mutating code path.
	function setFilter(next: string) {
		if (next === filter) return;
		void loadIntentions(next);
	}

	function createIntentionsPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		engineRef = engine;
		const field = new IntentionsFieldPass(engine);
		field.uploadScene(scene);
		// The DOM overlay owns all readable content; the field stays as the alive
		// backdrop only. No in-canvas MSDF text pass — it would ghost behind the DOM.
		return [field];
	}

	class IntentionsFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		private desktop: boolean;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			this.desktop = viewportAspect() >= 0.85;
			// Portrait keeps its verified dim backdrop exactly. Desktop can carry a much
			// richer field because the reading well protects the intention rows.
			this.field.setIntensity(this.desktop ? 1.6 : 0.24);
			// On desktop, leave more living field around the reading column while keeping
			// the complete row span inside a soft, low-luminance well.
			this.field.setReadingWell(
				this.desktop
					? { x: -0.2, y: 0.05, hw: 0.58, hh: 0.62, floor: 0.08, soft: 0.18 }
					: { x: -0.2, y: 0.05, hw: 0.85, hh: 0.92, floor: 0.08, soft: 0.25 }
			);
		}
		uploadScene(scene: RouteSceneModel): void {
			const data: FieldDatum[] = scene.nodes.map((node) => ({ id: node.source.id, score: node.activation ?? node.retention, hue: FIELD_HUE.forward, energy: node.activation, metric2: node.retention, selected: node.source.id === selectedIntentionId, kind: 'intention', payload: node }));
			// RouteStage now picks text chrome (front) before field cells (behind),
			// so the galaxy can fill without stealing the filter toggle's click.
			const sparse = data.length < 4;
			this.field.setCells(
				layoutGalaxy(data, {
					maxRadius: this.desktop ? 0.9 : 0.82,
					minCellR: sparse ? (this.desktop ? 0.56 : 0.22) : 0.025,
					maxCellR: sparse ? (this.desktop ? 0.7 : 0.3) : 0.075
				})
			);
		}
		compute(encoder: GPUCommandEncoder): void { this.field.compute(encoder); }
		render(pass: GPURenderPassEncoder): void { this.field.render(pass); }
		pickAt(x: number, y: number): RoutePick | null { return this.field.pickAt(x, y); }
		dispose(): void { this.field.dispose(); }
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

	function metricFrom(value: unknown, fallback: number): number {
		return typeof value === 'number' ? clamp01(value) : fallback;
	}

	function statusWeight(intention: RichIntention): number {
		const direct = metricFrom(intention.retention ?? intention.retentionStrength, Number.NaN);
		if (Number.isFinite(direct)) return direct;
		const reminderCount = typeof intention.reminder_count === 'number' ? intention.reminder_count : 0;
		const relatedCount = Array.isArray(intention.related_memories) ? intention.related_memories.length : 0;
		return clamp01((reminderCount + relatedCount + 1) / 6);
	}

	function intentionDepth(intention: RichIntention): number {
		return metricFrom(intention.confidence, clamp01((intention.priority || 2) / 4));
	}

	function summarizeTrigger(intention: RichIntention): string {
		try {
			const data = JSON.parse(intention.trigger_data || '{}') as Record<string, unknown>;
			const candidate = data.condition ?? data.topic ?? data.at ?? data.in_minutes ?? data.inMinutes ?? data.codebase;
			return sanitizeAscii(String(candidate ?? intention.trigger_type)).replace(/\s+/g, ' ').slice(0, 26);
		} catch {
			return sanitizeAscii(intention.trigger_type).slice(0, 26);
		}
	}

	const scene = $derived<RouteSceneModel>({
		organ: 'intentions',
		nodes: [
			...intentions.map((intention, index) => ({
			source: { kind: 'receipt' as const, id: intention.id },
			index,
			label: sanitizeAscii(intention.content).slice(0, 48),
			retention: statusWeight(intention),
			activation: intentionDepth(intention),
			trust: intentionDepth(intention),
			tags: intention.tags ?? [intention.status, intention.trigger_type].filter(Boolean),
			type: intention.trigger_type
			})),
			...predictions.map((prediction, offset) => ({
				source: { kind: 'memory' as const, id: prediction.id },
				index: intentions.length + offset,
				label: sanitizeAscii(prediction.content).slice(0, 48),
				retention: clamp01(prediction.retention),
				// Real continuous urgency (FSRS decay + review schedule) drives brightness.
				// Fall back to the high/medium/low band only if an older backend omits it.
				activation:
					typeof prediction.urgency === 'number'
						? clamp01(prediction.urgency)
						: prediction.predictedNeed === 'high'
							? 1
							: prediction.predictedNeed === 'medium'
								? 0.65
								: 0.35,
				trust: clamp01(prediction.retention),
				tags: [prediction.predictedNeed],
				type: prediction.nodeType
			}))
		],
		edges: [],
		events: [],
		receipts: intentions.map((intention, index) => ({
			source: { kind: 'receipt', id: intention.id },
			label: sanitizeAscii(intention.status),
			nodeIndices: [index]
		})),
		scalars: {
			total,
			visible: intentions.length,
			predicted: predictions.length,
			filter: filter === ALL_FILTER ? 1 : 0
		},
		alive: intentions.length > 0
	});

	function handlePick(pick: RoutePick) {
		if (pick.kind !== 'intention') return;
		// The pick comes from the FIELD pass (payload = RouteNode with .source.id),
		// so clicking a field cell selects that intention (highlight only).
		const payload = pick.payload as { source?: { id?: string } };
		selectedIntentionId = payload.source?.id ?? null;
	}

	// ── DOM overlay: real, legible reading surface on top of the WebGPU field ──
	// Every number below is derived from the real /api/intentions + /api/predict
	// payloads — none are constants. Selecting a row only sets selectedIntentionId
	// (highlight); it never mutates, snoozes, promotes, or deletes an intention.

	const PRIORITY_LABEL: Record<number, string> = {
		1: 'low',
		2: 'normal',
		3: 'high',
		4: 'critical'
	};
	function priorityLabel(p: number): string {
		return PRIORITY_LABEL[p] ?? 'normal';
	}
	// Urgency ~ priority band: critical/high read hot, normal/low cool. Pure
	// derivation of the real priority field, not a stored constant.
	function priorityColor(p: number): string {
		if (p >= 4) return '#ef4444';
		if (p >= 3) return '#f59e0b';
		if (p <= 1) return '#64748b';
		return '#22c7de';
	}

	const filterOptions: DropdownOption[] = [
		{ value: ACTIVE_FILTER, label: 'Active intentions', icon: 'intentions' },
		{ value: ALL_FILTER, label: 'All (incl. fired / snoozed)', icon: 'filter' }
	];

	// Live proof, all from the fetched payloads.
	const visibleCount = $derived(intentions.length);
	const highPriorityCount = $derived(intentions.filter((i) => (i.priority ?? 2) >= 3).length);
	const predictedCount = $derived(predictions.length);

	const selectedIntention = $derived(
		selectedIntentionId ? intentions.find((i) => i.id === selectedIntentionId) ?? null : null
	);

	function fmtDate(value?: string | null): string | null {
		if (!value) return null;
		const t = new Date(value).getTime();
		if (!Number.isFinite(t)) return null;
		try {
			return new Date(t).toLocaleString(undefined, {
				month: 'short',
				day: 'numeric',
				hour: '2-digit',
				minute: '2-digit'
			});
		} catch {
			return null;
		}
	}

	function selectRow(id: string) {
		// Plain select = highlight only. Non-mutating by contract.
		selectedIntentionId = selectedIntentionId === id ? null : id;
	}
</script>

<svelte:head>
	<title>Intentions · Vestige</title>
</svelte:head>

<RouteStage
	organ="intentions"
	seed={`real-intention-field:${filter}:${total}:${selectedIntentionId ?? 'none'}`}
	{scene}
	passes={createIntentionsPasses}
	loading={loading}
	error={error}
	emptyLabel=""
	onpick={handlePick}
/>

<!-- md:pl-24 clears the fixed os-dock rail (collapsed ~66px, left: 0.75rem) so the
     left edge of the reading panels never slides under it on desktop. Mobile keeps
     p-6 — the dock is hidden there (os-mobilebar replaces it). -->
<div class="relative z-10 min-h-full p-6 md:pl-24 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="intentions"
			title="Intentions"
			subtitle="Standing goals and intentions Vestige is tracking for you."
			accent="synapse"
		>
			<div class="flex items-center gap-2">
				<span class="text-dim text-sm tabular-nums inline-flex items-center gap-1.5">
					<AnimatedNumber value={visibleCount} /> {filter === ACTIVE_FILTER ? 'active' : 'total'}
				</span>
				<!-- (3) PRIMARY ACTION: explicit, labelled filter swap (real api reload). -->
				<Dropdown
					options={filterOptions}
					value={filter}
					label="Show"
					icon="filter"
					onChange={setFilter}
				/>
			</div>
		</PageHeader>
	</div>

	{#if error}
		<!-- (5) STATE GUIDANCE — error -->
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">Couldn't load intentions</div>
			<div class="max-w-md text-xs text-muted break-words">{error}</div>
			<button
				type="button"
				onclick={() => loadIntentions(filter)}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<!-- (5) STATE GUIDANCE — loading (shimmer skeletons) -->
		<div class="grid grid-cols-2 lg:grid-cols-3 gap-3 pointer-events-auto">
			{#each Array(3) as _}
				<div class="glass-subtle shimmer h-20 rounded-xl"></div>
			{/each}
		</div>
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 pointer-events-auto">
			<div class="glass-subtle shimmer min-h-[420px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[420px] rounded-2xl"></div>
		</div>
	{:else}
		<!-- (2) LIVE PROOF — big real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-3 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={visibleCount} />
				</div>
				<div class="text-xs text-dim mt-1">
					{filter === ACTIVE_FILTER ? 'active intentions' : 'intentions (all states)'}
				</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="flex items-center gap-2">
					<span class="w-2 h-2 rounded-full" style="background: #f59e0b"></span>
					<div class="text-2xl font-bold tabular-nums" style="color: #f59e0b">
						<AnimatedNumber value={highPriorityCount} />
					</div>
				</div>
				<div class="text-xs text-dim mt-1">high / critical priority</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #22c7de">
					<AnimatedNumber value={predictedCount} />
				</div>
				<div class="text-xs text-dim mt-1">predicted needs (FSRS)</div>
			</div>
		</div>

		{#if intentions.length === 0}
			<!-- (5) STATE GUIDANCE — designed empty state + (6) explanation -->
			<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center">
				<div
					class="flex h-14 w-14 items-center justify-center rounded-2xl border border-synapse/25 bg-synapse/10 text-synapse-glow"
				>
					<Icon name="intentions" size={26} draw />
				</div>
				<div class="text-sm font-medium text-bright">
					No {filter === ACTIVE_FILTER ? 'active ' : ''}intentions
				</div>
				<div class="max-w-md text-xs text-muted leading-relaxed">
					An intention is a standing goal Vestige holds for you — a
					<span class="text-dim">prospective memory</span> that stays dormant until its trigger
					fires (a time, a topic you return to, or a codebase you open). Nothing is being
					tracked yet.
				</div>
				{#if filter === ACTIVE_FILTER}
					<button
						type="button"
						onclick={() => setFilter(ALL_FILTER)}
						class="mt-1 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
					>
						Show all intentions (incl. fired &amp; snoozed)
					</button>
				{/if}
			</div>
		{:else}
			<!-- Populated: legible DOM list + selection detail -->
			<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
				<!-- Intention list -->
				<div class="glass-panel rounded-2xl p-3 space-y-2 max-h-[560px] overflow-y-auto">
					<div class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10">
						<span class="text-xs text-dim uppercase tracking-wider">Intentions</span>
						<span class="text-xs text-muted tabular-nums"><AnimatedNumber value={visibleCount} /></span>
					</div>

					{#each intentions as intention, i (intention.id)}
						{@const active = selectedIntentionId === intention.id}
						{@const trigger = summarizeTrigger(intention)}
						<button
							use:reveal={{ delay: Math.min(i * 30, 300), y: 10 }}
							onclick={() => selectRow(intention.id)}
							class="w-full text-left p-3 rounded-xl border transition lift
								{active
									? 'bg-synapse/10 border-synapse/40 shadow-[0_0_12px_rgba(99,102,241,0.18)]'
									: 'border-subtle/20 hover:border-synapse/30 hover:bg-white/[0.02]'}"
						>
							<div class="flex items-center gap-2 mb-1.5">
								<span
									class="w-2 h-2 rounded-full shrink-0"
									style="background: {priorityColor(intention.priority)}"
								></span>
								<span
									class="text-[10px] uppercase tracking-wider"
									style="color: {priorityColor(intention.priority)}"
								>
									{priorityLabel(intention.priority)}
								</span>
								<span class="text-[10px] text-muted ml-auto capitalize">{intention.status}</span>
							</div>
							<div class="text-sm text-text leading-snug">{intention.content}</div>
							<div class="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted">
								<span class="inline-flex items-center gap-1">
									<Icon name="pulse" size={11} />
									{intention.trigger_type}{trigger && trigger !== intention.trigger_type ? ` · ${trigger}` : ''}
								</span>
								{#if fmtDate(intention.deadline)}
									<span>due {fmtDate(intention.deadline)}</span>
								{/if}
							</div>
						</button>
					{/each}
				</div>

				<!-- (6) INTERPRETATION — selection detail panel -->
				<aside use:reveal={{ delay: 120, y: 16 }} class="glass rounded-2xl p-4 max-h-[560px] overflow-y-auto">
					{#if selectedIntention}
						<div class="flex items-start justify-between gap-3 border-b border-subtle/20 pb-3">
							<div class="min-w-0">
								<div class="font-mono text-[10px] uppercase tracking-[0.2em] text-synapse-glow">Intention</div>
								<div class="mt-1 flex items-center gap-2">
									<span
										class="text-[10px] uppercase tracking-wider"
										style="color: {priorityColor(selectedIntention.priority)}"
									>
										{priorityLabel(selectedIntention.priority)} priority
									</span>
									<span class="text-[10px] text-muted capitalize">· {selectedIntention.status}</span>
								</div>
							</div>
							<button
								type="button"
								onclick={() => selectRow(selectedIntention.id)}
								class="rounded-lg border border-subtle/30 px-2.5 py-1 text-xs text-muted transition hover:border-synapse/40 hover:text-text"
							>
								Close
							</button>
						</div>

						<p class="mt-3 text-sm text-text leading-relaxed">{selectedIntention.content}</p>

						<div class="mt-4 space-y-3 text-xs">
							<div>
								<div class="text-[10px] uppercase tracking-wider text-muted">Trigger</div>
								<div class="mt-0.5 text-text">
									{selectedIntention.trigger_type}
								</div>
								{#if summarizeTrigger(selectedIntention) && summarizeTrigger(selectedIntention) !== selectedIntention.trigger_type}
									<div class="text-dim">{summarizeTrigger(selectedIntention)}</div>
								{/if}
							</div>
							{#if fmtDate(selectedIntention.created_at)}
								<div>
									<div class="text-[10px] uppercase tracking-wider text-muted">Created</div>
									<div class="mt-0.5 text-dim">{fmtDate(selectedIntention.created_at)}</div>
								</div>
							{/if}
							{#if fmtDate(selectedIntention.deadline)}
								<div>
									<div class="text-[10px] uppercase tracking-wider text-muted">Deadline</div>
									<div class="mt-0.5 text-dim">{fmtDate(selectedIntention.deadline)}</div>
								</div>
							{/if}
							{#if fmtDate(selectedIntention.snoozed_until)}
								<div>
									<div class="text-[10px] uppercase tracking-wider text-muted">Snoozed until</div>
									<div class="mt-0.5 text-dim">{fmtDate(selectedIntention.snoozed_until)}</div>
								</div>
							{/if}
							<div class="pt-1 font-mono text-[10px] text-muted break-all">{selectedIntention.id}</div>
						</div>
					{:else}
						<div class="flex h-full min-h-[200px] flex-col items-center justify-center gap-2 text-center">
							<div class="text-dim opacity-60 breathe">
								<Icon name="intentions" size={34} strokeWidth={1.2} />
							</div>
							<p class="text-dim text-sm">Select an intention</p>
							<p class="max-w-[220px] text-xs text-muted leading-relaxed">
								Click a row (or a node in the field) to read its trigger, priority, and
								deadline. Selecting never changes it.
							</p>
						</div>
					{/if}
				</aside>
			</div>
		{/if}
	{/if}
</div>

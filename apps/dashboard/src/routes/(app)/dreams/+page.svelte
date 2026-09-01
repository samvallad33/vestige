<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { assertProvenance, type Provenance, type RouteNode, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { retentionColor } from '$lib/observatory/cognitive-palette';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import { osHref } from '$lib/os-nav';
	import type { ChangelogEvent } from '$stores/api';
	import type { DreamInsight, DreamResult, Memory } from '$types';

	type DreamRecord = {
		id: string;
		kind: 'dream-cycle' | 'dream-insight';
		text: string;
		depth: number;
		weight: number;
		source: Provenance;
		insightIndex?: number;
	};
	type DreamScene = RouteSceneModel & {
		organ: 'dreams';
		records: DreamRecord[];
		raw: DreamResult | null;
		selectedRecordId: string | null;
		busy: boolean;
	};
	type DreamResultEnvelope = DreamResult & { message?: string };

	let dreamResult: DreamResultEnvelope | null = $state(null);
	let loading = $state(false);
	let error: string | null = $state(null);
	let selectedRecordId: string | null = $state(null);
	let dormantPool: Memory[] = $state([]);
	// Real pool size (whole store, not just the sampled 80) — the dormant screen's
	// secondary stat: "N memories waiting to be replayed". Straight off /memories total.
	let dormantTotal = $state(0);

	// --- Real sleep history from the changelog (#changelog). Every DreamCompleted
	// event is a recorded past cycle. Gives the dormant screen real LIVE PROOF —
	// how many cycles have run, cumulative memories replayed / connections found /
	// insights consolidated, and when the mind last slept — before the user runs a
	// new cycle. All values come straight off /changelog; nothing is a constant. ---
	type DreamPast = {
		memoriesReplayed: number;
		connectionsFound: number;
		insightsGenerated: number;
		durationMs: number;
		timestamp: string;
	};
	let history = $state<DreamPast[]>([]);
	let historyLoading = $state(true);

	const totalCycles = $derived(history.length);
	const lastCycle = $derived<DreamPast | null>(history[0] ?? null);
	const cumulativeReplayed = $derived(history.reduce((sum, h) => sum + h.memoriesReplayed, 0));
	const cumulativeConnections = $derived(history.reduce((sum, h) => sum + h.connectionsFound, 0));
	const cumulativeInsights = $derived(history.reduce((sum, h) => sum + h.insightsGenerated, 0));

	function num(data: Record<string, unknown>, key: string): number {
		const v = data[key];
		const n = typeof v === 'number' ? v : Number(v);
		return Number.isFinite(n) ? n : 0;
	}

	function loadHistory() {
		historyLoading = true;
		void api
			.memoryChangelog(100)
			.then((res) => {
				const dreams = (res.events ?? [])
					.filter((e: ChangelogEvent) => e.type === 'DreamCompleted')
					.map((e: ChangelogEvent) => ({
						memoriesReplayed: num(e.data, 'memories_replayed'),
						connectionsFound: num(e.data, 'connections_found'),
						insightsGenerated: num(e.data, 'insights_generated'),
						durationMs: num(e.data, 'duration_ms'),
						timestamp: String(e.data.timestamp ?? e.timestamp ?? '')
					}))
					.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
				history = dreams;
			})
			.catch(() => {
				history = [];
			})
			.finally(() => {
				historyLoading = false;
			});
	}

	// Human-readable duration. FSRS-style consolidation cycles are milliseconds→seconds.
	function fmtDuration(ms: number): string {
		if (!Number.isFinite(ms) || ms <= 0) return '—';
		if (ms < 1000) return `${Math.round(ms)}ms`;
		return `${(ms / 1000).toFixed(1)}s`;
	}

	// Human-readable "time ago" for the last-slept line.
	function timeAgo(iso: string): string {
		const t = new Date(iso).getTime();
		if (!Number.isFinite(t) || t <= 0) return '—';
		const diff = Date.now() - t;
		if (diff < 0) return 'just now';
		const mins = Math.floor(diff / 60000);
		if (mins < 1) return 'just now';
		if (mins < 60) return `${mins}m ago`;
		const hrs = Math.floor(mins / 60);
		if (hrs < 24) return `${hrs}h ago`;
		const days = Math.floor(hrs / 24);
		return `${days}d ago`;
	}

	const dreamScene = $derived.by<DreamScene>(() => normalizeDreamScene(dreamResult, selectedRecordId, loading));

	// --- The last completed dream cycle's real result stats, human-labelled.
	// This is the "kill the number-salad" replacement: instead of a single MSDF
	// row of `status | 12 | 4 | 3 | 210` these become labelled cards. ---
	const insights = $derived.by<DreamInsight[]>(() => {
		const r = dreamResult;
		return r ? r.insights : [];
	});
	const selectedInsight = $derived.by<DreamInsight | null>(() => {
		if (!selectedRecordId) return null;
		const rec = dreamScene.records.find((r) => r.id === selectedRecordId);
		if (!rec || rec.kind !== 'dream-insight' || rec.insightIndex === undefined) return null;
		return insights[rec.insightIndex] ?? null;
	});

	onMount(() => {
		// Seed the dormant field so the organ is ALIVE before you hit RUN — a slow,
		// dim pool of real memories waiting to be replayed. On a dream run this same
		// field storms bright with the insight cells.
		void api.memories
			.list({ limit: '80' })
			.then((res) => {
				dormantPool = res.memories;
				// /memories total is the returned page size, not the whole store — fall
				// back to it, but prefer the real store count from /stats below.
				dormantTotal = Number(res.total ?? res.memories.length) || res.memories.length;
				fieldPass?.setCells(buildFieldCells(fieldEngine));
			})
			.catch(() => {});
		// Real store size for the dormant-pool line ("N memories waiting to be replayed").
		void api
			.stats()
			.then((s) => {
				if (Number.isFinite(s.totalMemories) && s.totalMemories > 0) dormantTotal = s.totalMemories;
			})
			.catch(() => {});
		loadHistory();
	});

	async function runDream() {
		if (loading) return;
		loading = true;
		error = null;
		try {
			dreamResult = await api.dream();
			selectedRecordId = null;
			fieldPass?.setCells(buildFieldCells(fieldEngine), { ambient: 0.5 });
			// Refresh the recorded sleep history — this cycle is now in the changelog.
			loadHistory();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	let fieldPass: LivingFieldPass | null = null;
	// Cached so setCells calls after a dream run can re-derive the live viewport aspect.
	let fieldEngine: ObservatoryEngine | null = null;

	/**
	 * Dormant → storm. At rest: the memory pool as slow dim cells (the mind
	 * asleep). After a dream: the insight nodes light up bright (consolidation
	 * storm). Both map to REAL data (memories list / dream insights).
	 */
	function buildFieldCells(engine: ObservatoryEngine | null = null) {
		const dreamInsights = dreamResult?.insights ?? [];
		let data: FieldDatum[];
		if (dreamInsights.length > 0) {
			data = dreamInsights.slice(0, 120).map((ins, i) => {
				const confidence = clamp01(Number(ins.confidence ?? 0));
				const novelty = clamp01(Number(ins.noveltyScore ?? confidence));
				return {
					id: ins.sourceMemories?.[0] ?? `dream-insight:${i}`,
					score: 0.5 + 0.5 * novelty,
					hue: novelty > 0.6 ? FIELD_HUE.retrograde : FIELD_HUE.bridge,
					energy: 0.6 + 0.4 * confidence,
					metric2: confidence,
					kind: 'dream-insight',
					payload: ins
				} satisfies FieldDatum;
			});
		} else {
			data = dormantPool.map((m) => {
				const retention = clamp01(m.retentionStrength);
				return {
					id: m.id,
					score: 0.3 + 0.3 * retention,
					hue: retentionColor(retention),
					energy: 0.12 + 0.28 * retention, // dim: the mind asleep
					metric2: retention,
					kind: 'dream-dormant',
					payload: m
				} satisfies FieldDatum;
			});
		}
		const cells = layoutGalaxy(data, { maxRadius: 0.92, minCellR: 0.014, maxCellR: 0.052 });

		// Portrait dormant screen only. The galaxy disc renders as a wide-but-short
		// band in a tall viewport, and the field is authored dim (0.26) for the
		// desktop read — so the phone shows a black void around one bit of centred
		// text. Portrait-gated (aspect < 0.85) tweak, derived from the LIVE viewport
		// aspect so desktop stays byte-identical: stretch the dormant substrate to
		// fill the whole tall screen.
		if (dreamResult || dormantPool.length === 0) return cells;
		const aspect = viewportAspect(engine);
		if (aspect >= 0.85) return cells;
		// The cell shader maps NDC y -> clip y 1:1 (no aspect divide), so a disc of
		// radius 0.92 already spans the height; nudge the vertical spread wider so the
		// substrate reaches the top and bottom thirds instead of hugging the middle.
		for (const c of cells) {
			c.y = Math.max(-0.98, Math.min(0.98, c.y * 1.35));
		}
		return cells;
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

	function scalarSource(name: string, value: number): Provenance {
		return { kind: 'scalar', id: `dreams.${name}`, scalar: { name, value } };
	}

	function insightId(insight: DreamInsight, index: number): string {
		const source = insight.sourceMemories?.[0] ?? `${insight.type}:${index}`;
		return sanitizeAscii(`${source}:${index}`).slice(0, 96);
	}

	function normalizeDreamScene(
		res: DreamResultEnvelope | null,
		selection: string | null,
		busy: boolean
	): DreamScene {
		if (!res) {
			return {
				organ: 'dreams',
				nodes: [],
				edges: [],
				events: [],
				receipts: [],
				scalars: {},
				alive: true,
				records: [],
				raw: null,
				selectedRecordId: selection,
				busy
			};
		}

		const resultInsights = Array.isArray(res.insights) ? res.insights : [];
		const memoriesReplayed = Number(res.memoriesReplayed ?? 0);
		const connectionsPersisted = Number(res.connectionsPersisted ?? 0);
		const generated = Number(res.stats?.insightsGenerated ?? resultInsights.length);
		const durationMs = Number(res.stats?.durationMs ?? 0);
		const newConnectionsFound = Number(res.stats?.newConnectionsFound ?? connectionsPersisted);
		const strengthened = Number(res.stats?.memoriesStrengthened ?? 0);
		const compressed = Number(res.stats?.memoriesCompressed ?? 0);
		const summaryParts = [
			res.status,
			String(memoriesReplayed),
			String(connectionsPersisted),
			String(generated),
			String(durationMs)
		].map((part) => sanitizeAscii(part ?? ''));
		const records: DreamRecord[] = [
			{
				id: 'dreams:cycle',
				kind: 'dream-cycle',
				text: summaryParts.join(' | '),
				depth: clamp01(memoriesReplayed / 50),
				weight: clamp01(newConnectionsFound / Math.max(1, memoriesReplayed)),
				source: scalarSource('memoriesReplayed', memoriesReplayed)
			}
		];

		if (res.message && resultInsights.length === 0) {
			records.push({
				id: 'dreams:message',
				kind: 'dream-cycle',
				text: sanitizeAscii(res.message),
				depth: clamp01(memoriesReplayed / 50),
				weight: clamp01(generated / Math.max(1, memoriesReplayed)),
				source: scalarSource('message', memoriesReplayed)
			});
		}

		resultInsights.forEach((insight, index) => {
			const confidence = clamp01(Number(insight.confidence ?? 0));
			const strength = clamp01(Number(insight.noveltyScore ?? confidence));
			const id = insightId(insight, index);
			const sourceMemory = insight.sourceMemories?.[0] ?? id;
			records.push({
				id: `dreams:insight:${id}`,
				kind: 'dream-insight',
				text: `C ${Math.round(confidence * 100)} · N ${Math.round(strength * 100)} | ${sanitizeAscii(insight.insight ?? '')}`,
				depth: confidence,
				weight: strength,
				source: { kind: 'memory', id: sourceMemory },
				insightIndex: index
			});
		});

		const nodes: RouteNode[] = records.map((record, index) => ({
			source: record.source,
			index,
			label: record.text,
			retention: record.weight,
			trust: record.depth,
			tags: [record.kind],
			type: record.kind
		}));
		const scene: DreamScene = {
			organ: 'dreams',
			nodes,
			edges: [],
			events: records.map((record, index) => ({
				source: record.source,
				type: record.kind,
				targetIndex: index,
				frame: 18 + index * 10,
				energy: record.weight
			})),
			receipts: [],
			scalars: {
				memoriesReplayed,
				connectionsPersisted,
				newConnectionsFound,
				insightsGenerated: generated,
				memoriesStrengthened: strengthened,
				memoriesCompressed: compressed,
				durationMs
			},
			alive: true,
			records,
			raw: res,
			selectedRecordId: selection,
			busy
		};
		if (import.meta.env.DEV) assertProvenance(scene);
		return scene;
	}

	/**
	 * Live viewport aspect, derived from the engine params (canvas px) with a
	 * window fallback — the SAME source TextLayerPass.portraitAdapt reads. Nothing
	 * is hardcoded to a phone width; the portrait branch keys off aspect < 0.85
	 * exactly like the shared adapter, so the desktop (aspect >= 0.85) render is
	 * byte-identical.
	 */
	function viewportAspect(engine: ObservatoryEngine | null): number {
		let vw = engine?.params[6] ?? 0;
		let vh = engine?.params[7] ?? 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return 1;
		return vw / vh;
	}

	function createDreamPasses(engine: ObservatoryEngine): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		fieldEngine = engine;
		// Portrait dormant screen ran near-black: intensity 0.26 + a wide reading well
		// left the whole phone void unlit. Gate off the LIVE viewport aspect (< 0.85)
		// so the phone gets a visibly-lit dream substrate, while desktop (aspect >= 0.85)
		// keeps a richer dormant/storm field. Nothing hardcoded to a pixel width.
		const portrait = viewportAspect(engine) < 0.85;
		if (portrait && !dreamResult) {
			field.setIntensity(0.42);
			field.setReadingWell({ x: 0, y: 0, hw: -1, hh: 0 });
		} else {
			field.setIntensity(portrait ? 0.26 : 0.72);
			field.setReadingWell({ x: 0.3, y: -0.05, hw: 0.72, hh: 0.9, floor: 0.06, soft: 0.22 });
		}
		field.setCells(buildFieldCells(engine));
		const fieldWrapper: RouteFramePass = {
			compute: (encoder) => field.compute(encoder),
			render: (pass) => field.render(pass),
			pickAt: (x, y) => field.pickAt(x, y),
			dispose: () => {
				field.dispose();
				if (fieldPass === field) fieldPass = null;
			}
		};
		return [fieldWrapper];
	}

	// A plain pick only SELECTS an insight (non-mutating). Running a cycle is an
	// explicit DOM button, never a silent side effect of a field click.
	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'dream-cycle' && pick.kind !== 'dream-insight') return;
		selectedRecordId = pick.id;
	}
</script>

<svelte:head>
	<title>Dream Consolidation · Vestige</title>
</svelte:head>

<RouteStage
	organ="dreams"
	seed={`dreams:${dreamResult?.status ?? 'pending'}:${dreamScene.records.length}:${dreamScene.scalars.durationMs ?? 0}`}
	scene={dreamScene}
	passes={createDreamPasses}
	loading={loading}
	error={error}
	emptyLabel=""
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="dreams"
			title="Dream Consolidation"
			subtitle="Run a sleep cycle: Vestige replays and strengthens memories, surfacing new connections."
			accent="dream"
		>
			<button
				type="button"
				onclick={runDream}
				disabled={loading || dormantTotal === 0}
				class="inline-flex items-center gap-2 rounded-xl bg-dream/20 px-4 py-2.5 text-sm font-semibold text-dream-glow transition
					hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream/60
					disabled:cursor-not-allowed disabled:opacity-50"
				title={dormantTotal === 0 ? 'No memories to replay yet — save some first' : 'Replay and strengthen memories'}
			>
				{#if loading}
					<span class="h-2 w-2 rounded-full bg-dream-glow animate-pulse"></span>
					Dreaming…
				{:else}
					<Icon name="dreams" size={16} />
					Run Dream Cycle
				{/if}
			</button>
		</PageHeader>
	</div>

	<!-- (3) disabled-reason, when the primary action can't run -->
	{#if dormantTotal === 0 && !loading}
		<div class="pointer-events-auto glass-subtle rounded-xl px-4 py-2.5 text-xs text-dim">
			The dream cycle needs memories to replay. Save some memories first, then run a cycle.
		</div>
	{/if}

	<!-- (5) STATE GUIDANCE: error -->
	{#if error}
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">The dream cycle failed</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={runDream}
				class="mt-2 rounded-lg bg-dream/20 px-4 py-2 text-xs font-medium text-dream-glow transition hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream/60"
			>
				Retry
			</button>
		</div>
	{/if}

	<!-- (2) LIVE PROOF — real sleep history from the changelog (dormant) OR this
	     cycle's real result stats (after a run). Never raw IDs or constants. -->
	{#if historyLoading && !dreamResult}
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-24 rounded-xl"></div>
			{/each}
		</div>
	{:else if !dreamResult}
		<!-- Dormant proof: what past sleep cycles have done, from /changelog. -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={dormantTotal} />
				</div>
				<div class="text-xs text-dim mt-1">memories in the pool, waiting to be replayed</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #a855f7">
					<AnimatedNumber value={totalCycles} />
				</div>
				<div class="text-xs text-dim mt-1">sleep cycles recorded</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #10b981">
					<AnimatedNumber value={cumulativeConnections} />
				</div>
				<div class="text-xs text-dim mt-1">connections found across all cycles</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					{#if lastCycle}
						{timeAgo(lastCycle.timestamp)}
					{:else}
						never
					{/if}
				</div>
				<div class="text-xs text-dim mt-1">last time the mind slept</div>
			</div>
		</div>
	{:else}
		<!-- After a run: the cycle's real result, human-labelled (kills the number salad). -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={Number(dreamScene.scalars.memoriesReplayed ?? 0)} />
				</div>
				<div class="text-xs text-dim mt-1">memories replayed</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #10b981">
					<AnimatedNumber value={Number(dreamScene.scalars.newConnectionsFound ?? 0)} />
				</div>
				<div class="text-xs text-dim mt-1">new connections found</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #a855f7">
					<AnimatedNumber value={Number(dreamScene.scalars.insightsGenerated ?? 0)} />
				</div>
				<div class="text-xs text-dim mt-1">insights surfaced</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					{fmtDuration(Number(dreamScene.scalars.durationMs ?? 0))}
				</div>
				<div class="text-xs text-dim mt-1">cycle duration</div>
			</div>
		</div>
	{/if}

	<!-- (5) STATE GUIDANCE: dormant/empty state — designed, never a blank void. -->
	{#if !dreamResult && !loading && !error}
		<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-4 rounded-2xl p-12 text-center">
			<div class="flex h-16 w-16 items-center justify-center rounded-2xl border border-dream/25 bg-dream/10 text-dream-glow breathe">
				<Icon name="dreams" size={30} draw />
			</div>
			<div class="text-base font-semibold text-bright">The mind is asleep</div>
			<p class="max-w-md text-sm text-muted">
				A dream cycle replays your memories the way sleep consolidates them: it strengthens the ones
				worth keeping and searches for new connections between them. Nothing runs until you start it.
			</p>
			<button
				type="button"
				onclick={runDream}
				disabled={loading || dormantTotal === 0}
				class="mt-1 inline-flex items-center gap-2 rounded-xl bg-dream/20 px-5 py-3 text-sm font-semibold text-dream-glow transition
					hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream/60
					disabled:cursor-not-allowed disabled:opacity-50"
			>
				<Icon name="dreams" size={16} />
				Run Dream Cycle
			</button>
			{#if !historyLoading && totalCycles > 0 && lastCycle}
				<p class="text-xs text-dim">
					Last cycle {timeAgo(lastCycle.timestamp)} replayed {lastCycle.memoriesReplayed.toLocaleString()} memories
					and found {lastCycle.connectionsFound.toLocaleString()} connections.
				</p>
			{/if}
		</div>
	{/if}

	<!-- (5) STATE GUIDANCE: loading a cycle -->
	{#if loading}
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-12 text-center enter">
			<div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-dream/25 bg-dream/10 text-dream-glow">
				<span class="h-3 w-3 rounded-full bg-dream-glow animate-ping"></span>
			</div>
			<div class="text-sm font-medium text-bright">Consolidating memory…</div>
			<p class="max-w-sm text-xs text-muted">
				Replaying the memory pool, strengthening the strong, and searching for new connections.
			</p>
		</div>
	{/if}

	<!-- (6) INTERPRETATION + populated result: the legible result panel + insight list. -->
	{#if dreamResult && !loading}
		<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px] gap-4 pointer-events-auto">
			<!-- Insight list — the real connections this cycle surfaced. -->
			<section use:reveal={{ delay: 60, y: 16 }} class="glass-panel rounded-2xl p-5">
				<div class="flex items-center justify-between border-b border-subtle/20 pb-3">
					<div>
						<div class="font-mono text-[10px] uppercase tracking-[0.2em] text-dream-glow">Dream receipt</div>
						<h2 class="mt-1 text-lg font-semibold text-bright">
							{insights.length > 0 ? 'New connections surfaced' : 'Cycle complete'}
						</h2>
					</div>
					<span class="text-xs text-dim tabular-nums">
						<AnimatedNumber value={insights.length} /> insight{insights.length === 1 ? '' : 's'}
					</span>
				</div>

				{#if insights.length === 0}
					<div class="flex flex-col items-center gap-2 py-8 text-center">
						<div class="text-dream-glow opacity-60"><Icon name="sparkle" size={30} draw /></div>
						<p class="text-sm text-muted">
							This cycle replayed {Number(dreamScene.scalars.memoriesReplayed ?? 0).toLocaleString()} memories
							but surfaced no new connections. Memories were still strengthened — run again after saving more.
						</p>
					</div>
				{:else}
					<ul class="mt-3 space-y-2">
						{#each dreamScene.records.filter((r) => r.kind === 'dream-insight') as record (record.id)}
							{@const ins = record.insightIndex !== undefined ? insights[record.insightIndex] : null}
							{@const isSelected = record.id === selectedRecordId}
							<li>
								<button
									type="button"
									onclick={() => (selectedRecordId = isSelected ? null : record.id)}
									class="w-full text-left p-3 rounded-xl border transition lift
										{isSelected
											? 'bg-dream/10 border-dream/40 shadow-[0_0_14px_rgba(168,85,247,0.18)]'
											: 'border-subtle/20 hover:border-dream/30 hover:bg-white/[0.02]'}"
								>
									<div class="flex items-center gap-2 mb-1.5">
										<span class="text-[10px] uppercase tracking-wider text-dream-glow">{ins?.type ?? 'connection'}</span>
										<span class="ml-auto text-[10px] text-muted tabular-nums">
											confidence {Math.round(clamp01(ins?.confidence ?? 0) * 100)}% · novelty {Math.round(clamp01(ins?.noveltyScore ?? 0) * 100)}%
										</span>
									</div>
									<p class="text-sm text-text">{ins?.insight ?? record.text}</p>
								</button>
							</li>
						{/each}
					</ul>
				{/if}
			</section>

			<!-- (6) INTERPRETATION: selection detail panel — "what you're seeing". -->
			<aside use:reveal={{ delay: 120, y: 16 }} class="glass rounded-2xl p-5 space-y-3 h-max">
				<div class="font-mono text-[10px] uppercase tracking-[0.2em] text-dim">What you're seeing</div>
				{#if selectedInsight}
					<h3 class="text-base font-semibold text-bright">{selectedInsight.type} connection</h3>
					<p class="text-sm text-text">{selectedInsight.insight}</p>
					<div class="grid grid-cols-2 gap-2 pt-2">
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">confidence</div>
							<div class="mt-1 font-mono text-lg text-dream-glow">{Math.round(clamp01(selectedInsight.confidence) * 100)}%</div>
						</div>
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">novelty</div>
							<div class="mt-1 font-mono text-lg" style="color: #22d3ee">{Math.round(clamp01(selectedInsight.noveltyScore) * 100)}%</div>
						</div>
					</div>
					<div class="rounded-xl bg-white/[0.03] p-3">
						<div class="text-[10px] uppercase tracking-wider text-muted">source memories</div>
						<div class="mt-1 font-mono text-lg text-bright tabular-nums">{selectedInsight.sourceMemories?.length ?? 0}</div>
						{#if selectedInsight.sourceMemories?.length}
							<div class="mt-2 flex flex-col gap-1">
								{#each selectedInsight.sourceMemories.slice(0, 6) as id (id)}
									<a class="font-mono text-[11px] text-synapse-glow hover:underline break-all" href={osHref('/memories', { memory: id })}>{id}</a>
								{/each}
							</div>
						{/if}
					</div>
				{:else}
					<p class="text-sm text-muted">
						This cycle strengthened {Number(dreamScene.scalars.memoriesStrengthened ?? 0).toLocaleString()} memories
						and searched for links between them. Select an insight to see the connection Vestige found and how
						confident it is.
					</p>
					<div class="grid grid-cols-2 gap-2 pt-1">
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">strengthened</div>
							<div class="mt-1 font-mono text-lg text-recall">{Number(dreamScene.scalars.memoriesStrengthened ?? 0).toLocaleString()}</div>
						</div>
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">compressed</div>
							<div class="mt-1 font-mono text-lg text-dim">{Number(dreamScene.scalars.memoriesCompressed ?? 0).toLocaleString()}</div>
						</div>
					</div>
				{/if}
				<button
					type="button"
					onclick={runDream}
					disabled={loading}
					class="mt-2 w-full inline-flex items-center justify-center gap-2 rounded-xl bg-dream/20 px-4 py-2.5 text-sm font-semibold text-dream-glow transition
						hover:bg-dream/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-dream/60 disabled:opacity-50"
				>
					<Icon name="dreams" size={15} />
					Dream Again
				</button>
			</aside>
		</div>
	{/if}
</div>

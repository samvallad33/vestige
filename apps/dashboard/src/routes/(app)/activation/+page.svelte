<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from 'svelte/store';
	import { page } from '$app/stores';
	import { api } from '$stores/api';
	import type { Memory, GraphResponse, GraphNode, GraphEdge } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass, type LivingCell } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { retentionColor, rgb01 } from '$lib/observatory/cognitive-palette';
	import PageHeader from '$components/PageHeader.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import { reveal } from '$lib/actions/reveal';

	// ── Config ──────────────────────────────────────────────────────────────
	// How many candidate seeds to offer, and how far the activation spreads.
	const SEED_LIMIT = 60;
	const GRAPH_DEPTH = 3;
	const MAX_NODES = 80;

	// Activated node with a REAL activation strength: 1.0 at the seed, decaying
	// by graph hop distance (BFS) weighted by edge strength. This is derived from
	// the actual returned subgraph — never a constant.
	interface Activated {
		node: GraphNode;
		/** 0..1 activation strength: 1 at seed, decays by hop distance. */
		strength: number;
		/** graph hop distance from the seed (0 = seed itself). */
		hops: number;
	}

	// ── State ───────────────────────────────────────────────────────────────
	let seedPool = $state<Memory[]>([]);
	let seedId = $state<string>('');
	let graph = $state<GraphResponse | null>(null);
	let activated = $state<Activated[]>([]);
	let selectedId = $state<string | null>(null);

	let seedsLoading = $state(true);
	let loading = $state(true);
	let error = $state<string | null>(null);

	let fieldPass: LivingFieldPass | null = null;

	// ── Colors ──────────────────────────────────────────────────────────────
	const CYAN = rgb01('#22C7DE');

	// ── Derived stats (all real, from the returned subgraph) ────────────────
	const litCount = $derived(activated.length);
	const edgeCount = $derived(graph?.edgeCount ?? 0);
	const reach = $derived(activated.reduce((m, a) => Math.max(m, a.hops), 0));
	const peakStrength = $derived(
		activated.length > 1 ? Math.max(...activated.filter((a) => a.hops > 0).map((a) => a.strength), 0) : 0
	);
	const seedMemory = $derived(seedPool.find((m) => m.id === seedId) ?? null);
	const selected = $derived(activated.find((a) => a.node.id === selectedId) ?? null);

	const seedOptions = $derived<DropdownOption[]>(
		seedPool.map((m) => ({
			value: m.id,
			label: preview(m.content, 52),
			badge: `${Math.round(clamp01(m.retentionStrength) * 100)}%`
		}))
	);

	// ── Load ────────────────────────────────────────────────────────────────
	onMount(() => {
		void loadSeeds();
	});

	async function loadSeeds() {
		seedsLoading = true;
		try {
			const res = await api.memories.list({ limit: String(SEED_LIMIT) });
			seedPool = res.memories;
			// Honor an incoming ?seed= / ?q= else start from the newest memory.
			const url = get(page).url;
			const wanted = url.searchParams.get('seed') || url.searchParams.get('center_id') || '';
			const initial = seedPool.find((m) => m.id === wanted)?.id ?? seedPool[0]?.id ?? '';
			seedId = initial;
			if (initial) {
				await spread(initial);
			} else {
				loading = false;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load seed memories';
			loading = false;
		} finally {
			seedsLoading = false;
		}
	}

	// Fire spreading activation from a seed memory: pull its subgraph, then
	// BFS-decay activation strength out from the seed across real edges.
	async function spread(fromId: string) {
		loading = true;
		error = null;
		selectedId = null;
		try {
			const res = await api.graph({ center_id: fromId, depth: GRAPH_DEPTH, max_nodes: MAX_NODES });
			graph = res;
			activated = computeActivation(res, fromId);
			fieldPass?.setCells(buildFieldCells());
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to spread activation';
			graph = null;
			activated = [];
		} finally {
			loading = false;
		}
	}

	function onSeedChange(id: string) {
		seedId = id;
		void spread(id);
	}

	// Explicit, labeled primary action — re-fire the spread from the current seed.
	function reignite() {
		if (!seedId) return;
		void spread(seedId);
	}

	// ── Activation math (real: BFS over the returned edges) ─────────────────
	function computeActivation(g: GraphResponse, fromId: string): Activated[] {
		const adj = new Map<string, { to: string; weight: number }[]>();
		const link = (from: string, to: string, weight: number) => {
			const list = adj.get(from) ?? [];
			list.push({ to, weight });
			adj.set(from, list);
		};
		for (const e of g.edges as GraphEdge[]) {
			const w = clamp01(e.weight);
			link(e.source, e.target, w);
			link(e.target, e.source, w);
		}
		// hop distance + best edge-weighted strength from the seed outward.
		const hop = new Map<string, number>();
		const strength = new Map<string, number>();
		hop.set(fromId, 0);
		strength.set(fromId, 1);
		const queue: string[] = [fromId];
		while (queue.length) {
			const cur = queue.shift()!;
			const curHop = hop.get(cur)!;
			const curStr = strength.get(cur)!;
			for (const { to, weight } of adj.get(cur) ?? []) {
				// each hop attenuates; a strong edge passes more activation through.
				const next = curStr * (0.45 + 0.5 * weight);
				if (next > (strength.get(to) ?? 0)) {
					strength.set(to, next);
					hop.set(to, curHop + 1);
					queue.push(to);
				}
			}
		}
		return (g.nodes as GraphNode[])
			.map((node) => ({
				node,
				strength: node.id === fromId ? 1 : clamp01(strength.get(node.id) ?? 0),
				hops: hop.get(node.id) ?? (node.id === fromId ? 0 : 99)
			}))
			.filter((a) => a.strength > 0 || a.node.id === fromId)
			.sort((a, b) => b.strength - a.strength);
	}

	// ── Field: activated cells lit by their real activation strength ────────
	function buildFieldCells(): LivingCell[] {
		const data: FieldDatum[] = activated.map((a) => {
			const seed = a.hops === 0;
			return {
				id: a.node.id,
				score: a.strength,
				hue: seed ? CYAN : retentionColor(clamp01(a.node.retention)),
				energy: 0.25 + 0.75 * a.strength,
				metric2: clamp01(a.node.retention),
				selected: a.node.id === selectedId,
				kind: 'activation-node',
				payload: { id: a.node.id }
			} satisfies FieldDatum;
		});
		return layoutGalaxy(data, { maxRadius: 0.94, minCellR: 0.014, maxCellR: 0.055 });
	}

	function createActivationPasses(engine: ObservatoryEngine): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		// Field is the hero spread; on portrait dim it + open a reading well so the
		// DOM overlay stays legible. Derived from the live engine aspect only.
		let vw = engine.params[6];
		let vh = engine.params[7];
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		const aspect = vw / Math.max(1, vh);
		if (aspect < 0.85) {
			field.setIntensity(0.28);
			field.setReadingWell({ x: 0, y: 0, hw: 0.9, hh: 0.85, floor: 0.08, soft: 0.26 });
		} else {
			field.setIntensity(1.15);
			field.setReadingWell({ x: 0, y: 0, hw: 0.52, hh: 0.56, floor: 0.1, soft: 0.18 });
		}
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

	// Field scene seed — real scalars so the backdrop is honest, never invented.
	const activationScene = $derived<RouteSceneModel>({
		organ: 'activation',
		nodes: [],
		edges: [],
		events: [],
		receipts: [],
		scalars: { lit: litCount, edges: edgeCount, reach },
		alive: activated.length > 0
	});

	// Plain click = SELECT (never mutate). Highlights the node in the DOM readout.
	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'activation-node') return;
		const payload = pick.payload as { id?: string };
		const id = payload?.id ?? pick.id;
		if (!id) return;
		selectedId = selectedId === id ? null : id;
		fieldPass?.setCells(buildFieldCells());
	}

	function selectNode(id: string) {
		selectedId = selectedId === id ? null : id;
		fieldPass?.setCells(buildFieldCells());
	}

	// ── Helpers ─────────────────────────────────────────────────────────────
	function clamp01(v: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));
	}
	function preview(text: string, n: number): string {
		const t = (text ?? '').replace(/\s+/g, ' ').trim();
		return t.length > n ? t.slice(0, n - 1) + '…' : t;
	}
	function strengthColor(s: number): string {
		// cyan-to-dim ramp so the readout mirrors the field's activation brightness.
		const a = Math.round(40 + 60 * clamp01(s));
		return `rgba(34, 199, 222, ${a / 100})`;
	}
</script>

<svelte:head>
	<title>Spreading Activation · Vestige</title>
</svelte:head>

<RouteStage
	organ="activation"
	seed={`activation-spread:${seedId}:${litCount}:${edgeCount}`}
	scene={activationScene}
	passes={createActivationPasses}
	{loading}
	{error}
	emptyLabel="PICK A SEED THOUGHT TO SPREAD ACTIVATION"
	onpick={handleRoutePick}
/>

<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="activation"
			title="Spreading Activation"
			subtitle="Watch activation spread through the memory graph from a seed thought."
			accent="synapse"
		>
			{#if activated.length > 0}
				<span class="text-dim text-sm tabular-nums inline-flex items-center gap-1.5">
					<AnimatedNumber value={litCount} /> lit
				</span>
			{/if}
		</PageHeader>
	</div>

	{#if error}
		<!-- (5) ERROR STATE -->
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="flex h-12 w-12 items-center justify-center rounded-xl border border-decay/30 bg-decay/10 text-decay">
				<Icon name="close" size={22} />
			</div>
			<div class="text-sm text-decay">Couldn't spread activation</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={reignite}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Retry
			</button>
		</div>
	{:else if loading && activated.length === 0}
		<!-- (5) LOADING STATE — shimmer skeletons -->
		<div class="grid grid-cols-2 lg:grid-cols-3 gap-3 pointer-events-auto">
			{#each Array(3) as _}
				<div class="glass-subtle shimmer h-24 rounded-xl"></div>
			{/each}
		</div>
		<div class="glass-subtle shimmer h-12 w-72 rounded-xl pointer-events-auto"></div>
		<div class="glass-subtle shimmer min-h-[420px] rounded-2xl pointer-events-auto"></div>
	{:else}
		<!-- (2) LIVE PROOF — real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-3 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={litCount} />
				</div>
				<div class="text-xs text-dim mt-1">memories activated</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #22C7DE">
					<AnimatedNumber value={edgeCount} />
				</div>
				<div class="text-xs text-dim mt-1">edges the signal traveled</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={reach} /><span class="text-sm text-dim ml-1">hops</span>
				</div>
				<div class="text-xs text-dim mt-1">deepest reach from seed</div>
			</div>
		</div>

		<!-- (3) PRIMARY ACTION + seed picker -->
		<div class="flex flex-wrap items-end gap-3 pointer-events-auto enter">
			<Dropdown
				options={seedOptions}
				value={seedId}
				label="Seed thought"
				icon="activation"
				placeholder={seedsLoading ? 'Loading memories…' : 'Pick a memory'}
				onChange={onSeedChange}
			/>
			<button
				type="button"
				onclick={reignite}
				disabled={!seedId || loading}
				title={!seedId ? 'Pick a seed memory first' : loading ? 'Spreading…' : 'Re-fire the spread'}
				class="inline-flex items-center gap-2 rounded-xl bg-synapse/20 px-4 py-2.5 text-sm font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60 disabled:cursor-not-allowed disabled:opacity-40"
			>
				<Icon name="sparkle" size={15} draw />
				{loading ? 'Spreading…' : 'Spread activation'}
			</button>
			{#if !seedId && !seedsLoading}
				<span class="text-xs text-muted">No memories yet — add one to seed a spread.</span>
			{/if}
		</div>

		{#if activated.length === 0}
			<!-- (5) EMPTY STATE -->
			<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center">
				<div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-synapse/25 bg-synapse/10 text-synapse-glow">
					<Icon name="activation" size={26} draw />
				</div>
				<div class="text-sm font-medium text-bright">
					{seedMemory ? 'This seed has no connected memories yet.' : 'Pick a seed thought to spread activation.'}
				</div>
				<div class="max-w-sm text-xs text-muted">
					Activation fans out across the memory graph. Choose a seed above, then hit
					<span class="text-synapse-glow">Spread activation</span> to see which memories light up.
				</div>
			</div>
		{:else}
			<!-- POPULATED — activated readout + selection detail -->
			<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
				<!-- Activated set readout -->
				<div class="glass-panel rounded-2xl p-3 space-y-1.5 max-h-[560px] overflow-y-auto">
					<div class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10">
						<span class="text-xs text-dim uppercase tracking-wider">Activated set</span>
						<span class="text-xs text-muted tabular-nums"><AnimatedNumber value={litCount} /></span>
					</div>
					{#each activated as a, i (a.node.id)}
						{@const isSel = selectedId === a.node.id}
						{@const isSeed = a.hops === 0}
						<button
							use:reveal={{ delay: Math.min(i * 28, 320), y: 8 }}
							onclick={() => selectNode(a.node.id)}
							class="w-full text-left p-3 rounded-xl border transition lift
								{isSel
									? 'bg-synapse/10 border-synapse/40 shadow-[0_0_12px_rgba(99,102,241,0.18)]'
									: 'border-subtle/20 hover:border-synapse/30 hover:bg-white/[0.02]'}"
						>
							<div class="flex items-center gap-2 mb-1.5">
								<span
									class="inline-block w-2.5 h-2.5 rounded-full shrink-0"
									style="background: {isSeed ? '#22C7DE' : strengthColor(a.strength)}"
								></span>
								{#if isSeed}
									<span class="text-[10px] uppercase tracking-wider text-[#22C7DE]">Seed</span>
								{:else}
									<span class="text-[10px] uppercase tracking-wider text-muted">{a.hops} hop{a.hops === 1 ? '' : 's'}</span>
								{/if}
								<span class="ml-auto text-[11px] tabular-nums text-dim">
									{(a.strength * 100).toFixed(0)}%
								</span>
							</div>
							<div class="text-xs text-text truncate">{preview(a.node.label, 72)}</div>
							<!-- activation strength meter (real strength, not a constant) -->
							<div class="mt-2 h-1 rounded-full bg-white/[0.06] overflow-hidden">
								<div
									class="h-full rounded-full"
									style="width: {Math.max(4, a.strength * 100)}%; background: {isSeed ? '#22C7DE' : strengthColor(Math.max(0.35, a.strength))}"
								></div>
							</div>
						</button>
					{/each}
				</div>

				<!-- (6) INTERPRETATION — selection detail / insight -->
				<aside use:reveal={{ delay: 120, y: 16 }} class="glass rounded-2xl p-4 space-y-3 self-start">
					{#if selected}
						<div class="flex items-center justify-between gap-2 border-b border-subtle/20 pb-2">
							<span class="text-[10px] uppercase tracking-[0.18em] text-synapse-glow">Activated memory</span>
							<button
								type="button"
								onclick={() => (selectedId = null)}
								class="rounded-lg border border-subtle/30 px-2.5 py-1 text-[11px] text-muted transition hover:border-synapse/40 hover:text-synapse-glow"
							>
								Clear
							</button>
						</div>
						<p class="text-sm text-text leading-relaxed">{selected.node.label}</p>
						<div class="grid grid-cols-2 gap-2 pt-1">
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">activation</div>
								<div class="mt-0.5 font-mono text-lg" style="color: #22C7DE">{(selected.strength * 100).toFixed(0)}%</div>
							</div>
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">distance</div>
								<div class="mt-0.5 font-mono text-lg text-bright">{selected.hops} hop{selected.hops === 1 ? '' : 's'}</div>
							</div>
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">retention</div>
								<div class="mt-0.5 font-mono text-lg text-bright">{(clamp01(selected.node.retention) * 100).toFixed(0)}%</div>
							</div>
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">type</div>
								<div class="mt-0.5 font-mono text-sm text-dim truncate">{selected.node.type || '—'}</div>
							</div>
						</div>
						{#if selected.node.tags?.length}
							<div class="flex flex-wrap gap-1 pt-1">
								{#each selected.node.tags.slice(0, 8) as t}
									<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{t}</span>
								{/each}
							</div>
						{/if}
					{:else}
						<div class="flex items-center gap-2 border-b border-subtle/20 pb-2">
							<Icon name="sparkle" size={15} />
							<span class="text-[10px] uppercase tracking-[0.18em] text-dim">What you're seeing</span>
						</div>
						<p class="text-xs text-muted leading-relaxed">
							Each lit dot is a memory the seed reached. Brightness and strength fall off with every
							hop across the graph — the seed sits at
							<span class="text-[#22C7DE]">100%</span>, its neighbors dimmer, distant memories faintest.
							Click any dot or row to inspect it.
						</p>
						{#if seedMemory}
							<div class="rounded-lg bg-white/[0.03] p-3">
								<div class="text-[10px] uppercase tracking-wider text-muted mb-1">Current seed</div>
								<p class="text-xs text-text leading-relaxed">{preview(seedMemory.content, 160)}</p>
							</div>
						{/if}
						<div class="rounded-lg bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">peak neighbor strength</div>
							<div class="mt-0.5 font-mono text-lg" style="color: #22C7DE">{(peakStrength * 100).toFixed(0)}%</div>
						</div>
					{/if}
				</aside>
			</div>
		{/if}
	{/if}
</div>

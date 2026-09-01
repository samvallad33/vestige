<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import Icon from '$components/Icon.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import Dropdown, { type DropdownOption } from '$components/Dropdown.svelte';
	import { reveal } from '$lib/actions/reveal';
	import { api } from '$stores/api';
	import { assertProvenance, type RouteNode, type RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutRings, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import type { CrossProjectCategory, CrossProjectPattern, CrossProjectPatternsResponse } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';

	type PatternScene = RouteSceneModel & {
		organ: 'patterns';
		patterns: CrossProjectPattern[];
		projects: string[];
		maxTransferCount: number;
	};

	const ROW_LIMIT = 42;

	let data = $state<CrossProjectPatternsResponse>({ projects: [], patterns: [] });
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedCategory: CrossProjectCategory | null = $state(null);
	// The specific pattern the user clicked (DOM row or field cell) — drives the
	// interpretation panel. NON-MUTATING: selecting only opens a read-only receipt.
	let selectedPattern = $state<CrossProjectPattern | null>(null);
	// The corpus cross-project patterns are mined FROM — a dim living substrate so
	// the organ breathes even when there are zero standing transfers today (the
	// pattern set is recomputed and legitimately empty at times). Real memories,
	// so the field still passes the discipline test.
	let poolCells: FieldDatum[] = [];
	let patternField: PatternFieldPass | null = null;

	onMount(() => {
		void loadPatterns();
		void api.memories
			.list({ limit: '90' })
			.then((res) => {
				poolCells = res.memories.map((m) => {
					const retention = clamp01(m.retentionStrength);
					return {
						id: m.id,
						score: 0.28 + 0.4 * retention,
						hue: FIELD_HUE.bridge,
						energy: 0.14 + 0.28 * retention,
						metric2: retention,
						kind: 'pattern-pool',
						payload: m
					} satisfies FieldDatum;
				});
				patternField?.refresh();
			})
			.catch(() => {});
	});

	async function loadPatterns() {
		loading = true;
		error = null;
		try {
			data = await api.crossProjectPatterns();
		} catch (err) {
			data = { projects: [], patterns: [] };
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	}

	const visiblePatterns = $derived.by(() => {
		const patterns = selectedCategory
			? data.patterns.filter((pattern) => pattern.category === selectedCategory)
			: data.patterns;
		return [...patterns].sort((a, b) => b.transfer_count - a.transfer_count || b.confidence - a.confidence);
	});

	// --- Real stat proof, all derived from the wire response (never constants). ---
	const patternCount = $derived(data.patterns.length);
	const projectCount = $derived(data.projects.length);
	const totalTransfers = $derived(
		data.patterns.reduce((sum, p) => sum + finite(p.transfer_count), 0)
	);
	// Strongest theme = the category carrying the most cross-project transfers.
	const strongestTheme = $derived.by<{ category: CrossProjectCategory; transfers: number } | null>(() => {
		if (data.patterns.length === 0) return null;
		const byCategory = new Map<CrossProjectCategory, number>();
		for (const p of data.patterns) {
			byCategory.set(p.category, (byCategory.get(p.category) ?? 0) + finite(p.transfer_count));
		}
		let best: { category: CrossProjectCategory; transfers: number } | null = null;
		for (const [category, transfers] of byCategory) {
			if (!best || transfers > best.transfers) best = { category, transfers };
		}
		return best;
	});

	// --- Category lens dropdown. Only tracked categories that actually appear in
	// the data are offered, so the control never lies about what's there. ---
	const presentCategories = $derived.by<CrossProjectCategory[]>(() => {
		const set = new Set<CrossProjectCategory>();
		for (const p of data.patterns) set.add(p.category);
		return Array.from(set).sort();
	});
	const categoryOptions = $derived<DropdownOption[]>([
		{ value: '', label: 'All themes', icon: 'patterns' },
		...presentCategories.map((c) => ({
			value: c,
			label: prettyCategory(c),
			badge: data.patterns.filter((p) => p.category === c).length
		}))
	]);
	function onCategoryChange(v: string) {
		selectedCategory = v ? (v as CrossProjectCategory) : null;
		selectedPattern = null;
	}

	// NON-MUTATING selection: opens the read-only detail panel, toggles off on
	// re-click. No API mutation is ever triggered by a click.
	function selectPattern(pattern: CrossProjectPattern) {
		selectedPattern = isSamePattern(selectedPattern, pattern) ? null : pattern;
	}

	function isSamePattern(a: CrossProjectPattern | null, b: CrossProjectPattern | null): boolean {
		if (!a || !b) return false;
		return patternKey(a) === patternKey(b);
	}

	const patternScene = $derived.by<PatternScene>(() => normalizePatternScene(data.projects, visiblePatterns));

	function normalizePatternScene(projects: string[], patterns: CrossProjectPattern[]): PatternScene {
		const maxTransferCount = Math.max(1, ...patterns.map((pattern) => finite(pattern.transfer_count)));
		const nodes: RouteNode[] = patterns.slice(0, ROW_LIMIT).map((pattern, index) => {
			const strength = clamp01(finite(pattern.transfer_count) / maxTransferCount);
			const confidence = clamp01(pattern.confidence);
			return {
				source: { kind: 'pattern', id: patternKey(pattern) },
				index,
				label: patternLine(pattern),
				retention: confidence,
				activation: strength,
				trust: confidence,
				lastAccessed: pattern.last_used,
				tags: [pattern.category, pattern.origin_project, ...pattern.transferred_to],
				type: pattern.category
			};
		});

		const scene: PatternScene = {
			organ: 'patterns',
			nodes,
			edges: [],
			events: patterns.slice(0, ROW_LIMIT).map((pattern, index) => ({
				source: { kind: 'event', id: `patterns.${patternKey(pattern)}.${pattern.last_used}` },
				type: pattern.category,
				targetIndex: index,
				frame: 12 + index * 3,
				energy: clamp01(pattern.confidence)
			})),
			receipts: [],
			scalars: {
				projectCount: projects.length,
				patternCount: patterns.length,
				maxTransferCount,
				totalTransfers: patterns.reduce((sum, pattern) => sum + finite(pattern.transfer_count), 0)
			},
			alive: patterns.length > 0,
			patterns,
			projects,
			maxTransferCount
		};
		if (import.meta.env.DEV) assertProvenance(scene);
		return scene;
	}

	// Stable ring index per tracked category so each project-family (category) is
	// its own concentric ring. Order is fixed (not data-dependent) so the same
	// category always lands on the same ring across reloads/filters.
	const CATEGORY_RING: Record<CrossProjectCategory, number> = {
		ErrorHandling: 0,
		AsyncConcurrency: 1,
		Testing: 2,
		Architecture: 3,
		Performance: 4,
		Security: 5
	};

	function createPatternPasses(engine: ObservatoryEngine): RouteFramePass[] {
		// Only the living ring field renders in-canvas. The DOM overlay owns every
		// readable row now, so no MSDF text pass is added here (removing it kills the
		// redundant "ghost" text that used to bleed through behind the glass).
		const field = new PatternFieldPass(engine);
		patternField = field;
		return [field];
	}

	/**
	 * Cross-project patterns as a living field of concentric rings — one ring per
	 * category (project-family). Each cell is a REAL scene.node: radius/glow scale
	 * with transfer strength (activation), the oxygen membrane tints by retention
	 * (confidence), and high-transfer patterns burn CAUSAL.forward while the rest
	 * hold the bridge hue. The field breathes behind the readable MSDF rows.
	 */
	class PatternFieldPass implements RouteFramePass {
		private field: LivingFieldPass;
		private lastNodes: RouteNode[] = [];
		private portrait: boolean;
		constructor(engine: ObservatoryEngine) {
			this.field = new LivingFieldPass(engine);
			this.portrait = viewportAspect(engine) < 0.85;
			// Keep the verified portrait treatment byte-for-byte dim, but let landscape
			// screens carry a materially richer field outside the reading well. Use the
			// live engine viewport (with only a pre-frame window fallback), never a
			// device-width constant.
			this.field.setIntensity(this.portrait ? 0.24 : 0.62);
			// Rows anchor at x=-0.88 and run wide (maxWidthEm 58) from y~+0.74 down to
			// y~-0.74. Quiet the field across that left-weighted reading band so glyphs
			// stay legible while the rings still breathe in the margins.
			this.field.setReadingWell({ x: -0.15, y: 0, hw: 0.85, hh: 0.85, floor: 0.08, soft: 0.25 });
		}
		uploadScene(scene: RouteSceneModel): void {
			this.lastNodes = (scene as PatternScene).nodes;
			this.apply();
		}
		/** Re-apply when the fallback memory pool arrives after mount. */
		refresh(): void {
			this.apply();
		}
		private apply(): void {
			const nodes = this.lastNodes;
			// No standing cross-project patterns today → breathe the real memory pool
			// they're mined from (dim tissue), so the organ is alive, not black.
			if (nodes.length === 0) {
				this.field.setCells(
					layoutRings(poolCells, (_d, i) => i % 6, {
						ringCount: 6,
						maxRadius: 0.92,
						minCellR: this.portrait ? 0.02 : 0.06,
						maxCellR: this.portrait ? 0.05 : 0.14
					})
				);
				return;
			}
			const data: FieldDatum[] = nodes.map((node) => {
				const strength = clamp01(finite(node.activation ?? 0));
				return {
					id: node.source.id,
					score: strength,
					hue: strength >= 0.5 ? FIELD_HUE.forward : FIELD_HUE.bridge,
					// Lift the glow floor so faint-transfer patterns still emit membrane
					// plasma and the whole field fills instead of pinpricking. Retention
					// (confidence) still drives the top of the range.
					energy: clamp01(0.5 + 0.5 * finite(node.retention ?? 0)),
					metric2: clamp01(finite(node.trust ?? 0)),
					kind: 'pattern',
					payload: node
				};
			});
			this.field.setCells(
				layoutRings(data, (_d, i) => ringOfNode(nodes[i]), {
					ringCount: 6,
					maxRadius: 0.92,
					minCellR: 0.055,
					maxCellR: 0.15
				})
			);
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

	function ringOfNode(node: RouteNode): number {
		const category = node.type as CrossProjectCategory;
		return CATEGORY_RING[category] ?? 0;
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind !== 'pattern') return;
		// Field-cell pick opens the matching pattern's read-only receipt — never a
		// mutation. Each cell carries its scene node, whose source id is the pattern
		// key (see normalizePatternScene), so we resolve straight back to the pattern.
		const node = pick.payload as RouteNode;
		const key = node?.source?.id;
		const match = key ? (data.patterns.find((p) => patternKey(p) === key) ?? null) : null;
		if (match) selectPattern(match);
	}

	function patternLine(pattern: CrossProjectPattern): string {
		return sanitizeAscii(
			[
				pattern.name,
				pattern.origin_project,
				pattern.transferred_to.join(','),
				pattern.category,
				String(pattern.transfer_count),
				String(Math.round(clamp01(pattern.confidence) * 100)),
				pattern.last_used
			].join(' | ')
		).slice(0, 118);
	}

	function patternKey(pattern: CrossProjectPattern): string {
		return sanitizeAscii(
			[pattern.name, pattern.origin_project, pattern.category, pattern.last_used].join(':')
		).slice(0, 180);
	}

	function prettyCategory(category: CrossProjectCategory): string {
		// Split the PascalCase enum into words for human-readable labels.
		return category.replace(/([a-z])([A-Z])/g, '$1 $2');
	}

	function categoryAccent(category: CrossProjectCategory): string {
		return CATEGORY_RING[category] >= 4 ? '#FF3B30' : CATEGORY_RING[category] >= 2 ? '#FFB020' : '#22C7DE';
	}

	function formatDate(iso: string): string {
		const t = Date.parse(iso);
		if (Number.isNaN(t)) return iso;
		return new Date(t).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function viewportAspect(engine: ObservatoryEngine): number {
		let vw = engine.params[6] || 0;
		let vh = engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		return vw > 0 && vh > 0 ? vw / vh : 1.6;
	}

	function finite(value: number): number {
		return Number.isFinite(value) ? value : 0;
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
	}
</script>

<svelte:head>
	<title>Patterns · Vestige</title>
</svelte:head>

<RouteStage
	organ="patterns"
	seed={`cross-project-patterns:${data.projects.length}:${visiblePatterns.length}:${selectedCategory ?? 'all'}`}
	scene={patternScene}
	passes={createPatternPasses}
	loading={loading}
	error={error}
	emptyLabel=""
	onpick={handleRoutePick}
/>

<!--
	DOM-hybrid overlay (mirrors /contradictions). The WebGPU ring field stays alive
	behind; this layer carries the identity, live proof, primary action, state
	guidance, and interpretation. pointer-events pass through to the field except on
	interactive children (pointer-events-auto).
-->
<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<!-- (1) IDENTITY -->
	<div class="pointer-events-auto">
		<PageHeader
			icon="patterns"
			title="Cross-Project Patterns"
			subtitle="Recurring structures Vestige detects across your projects and topics."
			accent="recall"
		>
			<span class="text-dim text-sm tabular-nums inline-flex items-center gap-1.5">
				<AnimatedNumber value={visiblePatterns.length} /> in view
			</span>
			<!-- (3) PRIMARY ACTION -->
			<button
				type="button"
				onclick={loadPatterns}
				disabled={loading}
				class="inline-flex items-center gap-1.5 rounded-xl border border-recall/30 bg-recall/10 px-3 py-2 text-xs font-medium text-recall transition hover:bg-recall/20 disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60 lift"
				title={loading ? 'Re-scanning your projects…' : 'Re-scan projects for transferred patterns'}
			>
				<Icon name="patterns" size={13} />
				{loading ? 'Scanning…' : 'Re-scan patterns'}
			</button>
		</PageHeader>
	</div>

	{#if error}
		<!-- (5) STATE GUIDANCE — error -->
		<div class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">Couldn't load cross-project patterns</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={loadPatterns}
				class="mt-2 rounded-lg bg-recall/20 px-4 py-2 text-xs font-medium text-recall transition hover:bg-recall/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-recall/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<!-- (5) STATE GUIDANCE — loading -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-20 rounded-xl"></div>
			{/each}
		</div>
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4 pointer-events-auto">
			<div class="glass-subtle shimmer min-h-[420px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[420px] rounded-2xl"></div>
		</div>
	{:else if patternCount === 0}
		<!-- (5) STATE GUIDANCE — empty (designed, not a void) -->
		<div class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center">
			<div class="flex h-14 w-14 items-center justify-center rounded-2xl border border-recall/25 bg-recall/10 text-recall">
				<Icon name="patterns" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">No cross-project patterns standing today.</div>
			<div class="max-w-md text-xs text-muted">
				A pattern appears here once a solved approach in one project — an error-handling
				shape, a testing strategy, an architecture — shows up again in another. Keep working
				across projects, then hit <span class="text-recall">Re-scan patterns</span> to mine them.
			</div>
			{#if projectCount > 0}
				<div class="mt-1 text-[11px] text-dim tabular-nums">
					Watching <AnimatedNumber value={projectCount} /> {projectCount === 1 ? 'project' : 'projects'} · none have shared a structure yet
				</div>
			{/if}
		</div>
	{:else}
		<!-- (2) LIVE PROOF — real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={patternCount} />
				</div>
				<div class="text-xs text-dim mt-1">patterns detected</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={projectCount} />
				</div>
				<div class="text-xs text-dim mt-1">projects linked</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #22C7DE">
					<AnimatedNumber value={totalTransfers} />
				</div>
				<div class="text-xs text-dim mt-1">total transfers</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				{#if strongestTheme}
					<div class="text-lg font-bold leading-tight" style="color: {categoryAccent(strongestTheme.category)}">
						{prettyCategory(strongestTheme.category)}
					</div>
					<div class="text-xs text-dim mt-1 tabular-nums">
						strongest theme · {strongestTheme.transfers} transfers
					</div>
				{:else}
					<div class="text-lg font-bold text-muted">—</div>
					<div class="text-xs text-dim mt-1">strongest theme</div>
				{/if}
			</div>
		</div>

		<!-- Filter bar (drives lens only; non-mutating) -->
		<div class="flex flex-wrap gap-3 items-end enter pointer-events-auto">
			<Dropdown
				options={categoryOptions}
				value={selectedCategory ?? ''}
				label="Theme"
				icon="filter"
				onChange={onCategoryChange}
			/>
			{#if selectedCategory}
				<button
					onclick={() => {
						selectedCategory = null;
						selectedPattern = null;
					}}
					class="ml-auto inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs border border-subtle/30 text-dim hover:text-text hover:border-recall/30 hover:bg-white/[0.03] transition lift"
				>
					<Icon name="close" size={13} />
					Clear theme
				</button>
			{/if}
		</div>

		<!-- Main: pattern grid + interpretation panel -->
		<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
			<!-- Pattern list/grid -->
			<div class="space-y-2">
				{#if visiblePatterns.length === 0}
					<div class="glass-panel flex flex-col items-center gap-2 rounded-2xl p-10 text-center">
						<div class="text-dim opacity-60 breathe"><Icon name="patterns" size={40} strokeWidth={1.2} /></div>
						<p class="text-dim text-sm">No patterns in the <span class="text-recall">{selectedCategory ? prettyCategory(selectedCategory) : ''}</span> theme.</p>
					</div>
				{:else}
					<div class="grid grid-cols-1 xl:grid-cols-2 gap-3">
						{#each visiblePatterns as pattern, i (patternKey(pattern))}
							{@const focused = isSamePattern(selectedPattern, pattern)}
							<button
								use:reveal={{ delay: Math.min(i * 30, 320), y: 10 }}
								onclick={() => selectPattern(pattern)}
								class="w-full text-left p-4 rounded-xl border transition lift glass
									{focused
										? 'border-recall/45 shadow-[0_0_14px_rgba(34,199,222,0.18)]'
										: 'border-subtle/20 hover:border-recall/30 hover:bg-white/[0.02]'}"
							>
								<div class="flex items-center gap-2 mb-2">
									<span class="w-2 h-2 rounded-full shrink-0" style="background: {categoryAccent(pattern.category)}"></span>
									<span class="text-[10px] uppercase tracking-wider" style="color: {categoryAccent(pattern.category)}">
										{prettyCategory(pattern.category)}
									</span>
									<span class="ml-auto text-[10px] text-muted tabular-nums">
										{pattern.transfer_count}× · {Math.round(clamp01(pattern.confidence) * 100)}%
									</span>
								</div>
								<div class="text-sm text-bright font-medium mb-1.5 truncate">{pattern.name}</div>
								<div class="flex flex-wrap items-center gap-1.5 text-[11px] text-dim">
									<span class="px-1.5 py-0.5 rounded bg-white/[0.05] text-muted">{pattern.origin_project}</span>
									<Icon name="explore" size={11} />
									<span class="truncate">{pattern.transferred_to.join(', ') || 'unshared'}</span>
								</div>
							</button>
						{/each}
					</div>
				{/if}
			</div>

			<!-- (6) INTERPRETATION — selection detail / hint -->
			<aside use:reveal={{ delay: 120, y: 16 }} class="glass-panel rounded-2xl p-4 h-fit sticky top-6">
				{#if selectedPattern}
					{@const p = selectedPattern}
					<div class="flex items-start justify-between gap-2 border-b border-subtle/20 pb-3">
						<div class="min-w-0">
							<div class="font-mono text-[10px] uppercase tracking-[0.22em]" style="color: {categoryAccent(p.category)}">
								{prettyCategory(p.category)}
							</div>
							<h2 class="mt-1 text-base font-semibold text-bright leading-tight break-words">{p.name}</h2>
						</div>
						<button
							type="button"
							onclick={() => (selectedPattern = null)}
							class="shrink-0 rounded-lg border border-subtle/30 px-2.5 py-1 text-xs text-muted transition hover:border-recall/40 hover:text-recall"
						>
							Close
						</button>
					</div>

					<p class="mt-3 text-xs text-dim leading-relaxed">
						This pattern was first solved in
						<span class="text-text font-medium">{p.origin_project}</span>
						and Vestige later matched it in
						<span class="text-text font-medium">{p.transferred_to.join(', ') || '— no other project yet'}</span>.
					</p>

					<div class="mt-4 grid grid-cols-2 gap-2">
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">transfers</div>
							<div class="mt-1 font-mono text-lg text-bright tabular-nums">{p.transfer_count}</div>
						</div>
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">confidence</div>
							<div class="mt-1 font-mono text-lg tabular-nums" style="color: {categoryAccent(p.category)}">
								{Math.round(clamp01(p.confidence) * 100)}%
							</div>
						</div>
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">reached</div>
							<div class="mt-1 font-mono text-lg text-bright tabular-nums">{p.transferred_to.length}</div>
						</div>
						<div class="rounded-xl bg-white/[0.03] p-3">
							<div class="text-[10px] uppercase tracking-wider text-muted">last used</div>
							<div class="mt-1 font-mono text-xs text-dim">{formatDate(p.last_used)}</div>
						</div>
					</div>

					<div class="mt-3 flex flex-wrap gap-1">
						<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{p.origin_project}</span>
						{#each p.transferred_to as proj}
							<span class="text-[9px] px-1.5 py-0.5 rounded bg-recall/10 text-recall">{proj}</span>
						{/each}
					</div>
				{:else}
					<div class="flex flex-col items-center gap-2 py-6 text-center">
						<div class="text-dim opacity-60 breathe"><Icon name="sparkle" size={30} draw /></div>
						<div class="text-xs font-medium text-bright">Pick a pattern</div>
						<p class="max-w-[15rem] text-[11px] text-muted leading-relaxed">
							Click any card — or a glowing ring in the field — to see where the structure
							started and which projects it spread to. Each ring is one theme.
						</p>
					</div>
				{/if}
			</aside>
		</div>
	{/if}
</div>

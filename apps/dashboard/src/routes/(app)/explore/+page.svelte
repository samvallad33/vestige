<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { replaceState } from '$app/navigation';
	import { api } from '$stores/api';
	import type { Memory, SearchResult } from '$types';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import PageHeader from '$components/PageHeader.svelte';
	import AnimatedNumber from '$components/AnimatedNumber.svelte';
	import Icon from '$components/Icon.svelte';
	import { reveal } from '$lib/actions/reveal';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type SearchMemory = Memory & {
		similarity?: number;
		score?: number;
		relevance?: number;
		retention?: number;
	};
	type ExploreTextItem = TextLayerItem & { memoryId?: string };

	const SEARCH_LIMIT = 40;

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	let results: SearchMemory[] = $state([]);
	let searchResult = $state<SearchResult | null>(null);
	let searchQuery = $state('');
	// The visible input is decoupled from the live query: the user types into
	// `inputValue`, then Enter / Walk commits it into `searchQuery`. That keeps a
	// half-typed query from firing a search on every keystroke while still letting
	// the field + list track the committed thought.
	let inputValue = $state('');
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeRun: string | null = null;
	// Semantic walk (Wave 3): clicking a neighbor re-centers the whole neighborhood
	// on that thought, in place. walkingFrom carries the 8-char id of the thought
	// being walked from (shown in the loading status); seededFrom marks the newest-
	// memory auto-seed used when the page is opened with no ?q=.
	let walkingFrom: string | null = $state(null);
	let seededFrom: string | null = $state(null);
	let centerId: string | null = $state(null);
	// The content of the memory the current neighborhood is centered on — powers
	// the "you are here" interpretation cue in the DOM overlay.
	let centerContent: string | null = $state(null);
	// Generation token: two rapid walks race (B finishes, then stale A overwrites
	// B's neighborhood while the URL keeps B's query). Only the newest generation
	// may apply results / clear the walking label. (GPT-5.6-sol cross-review.)
	let loadGen = 0;

	// A memory id passed from another organ (e.g. Graph's "Explore Connections")
	// seeds the walk from THAT thought instead of re-seeding from the newest one,
	// so the selected memory survives the Graph → Explore handoff.
	let seedMemoryId: string | null = $state(null);

	// The neighbor the user has focused in the DOM list — drives the detail panel.
	// Selection is NON-MUTATING: focusing a neighbor only reads it; walking (an
	// explicit labelled action) is what re-centers the neighborhood.
	let selectedId: string | null = $state(null);
	const selected = $derived(results.find((m) => m.id === selectedId) ?? null);

	onMount(() => {
		const params = new URLSearchParams(window.location.search);
		searchQuery = params.get('q') ?? '';
		inputValue = searchQuery;
		// `memory` is the canonical cross-organ selection contract (center = legacy).
		seedMemoryId = params.get('memory') ?? params.get('center');
		void loadNeighbors();
	});

	onDestroy(() => {
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	// Live viewport aspect (canvas px) — same source portraitAdapt reads, never a
	// hardcoded phone width. Falls back to the window before frame 0, then 1.
	function viewportAspect(): number {
		const vw = engineRef?.params[6] || 0;
		const vh = engineRef?.params[7] || 0;
		if (vw > 0 && vh > 0) return vw / vh;
		if (typeof window !== 'undefined' && window.innerHeight > 0) {
			return window.innerWidth / window.innerHeight;
		}
		return 1;
	}

	// Field brightness for the CURRENT viewport. On desktop the neighborhood is the
	// hero (0.75). On a phone the same intensity blooms into a blinding blob behind
	// the text and destroys contrast — so portrait drops it to the documented dim-
	// backdrop range (~0.26). Gated on the LIVE aspect; desktop is unchanged.
	function fieldIntensity(): number {
		return viewportAspect() < 0.85 ? 0.24 : 0.75;
	}

	// The DOM overlay column sits over the field's bright selected-cell bloom.
	// Carve a reading well so the field dims under the text and the glass panels
	// always read. The well auto-reflows for portrait inside the pass.
	function applyReadingWell(field: LivingFieldPass) {
		if (viewportAspect() < 0.85) {
			field.setReadingWell({ x: 0, y: 0, hw: 1.05, hh: 0.95, floor: 0.08, soft: 0.3 });
		} else {
			field.setReadingWell({ x: -0.32, y: 0, hw: 0.72, hh: 0.95, floor: 0.12, soft: 0.24 });
		}
	}

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		field.setIntensity(fieldIntensity());
		applyReadingWell(field);
		field.setCells(buildFieldCells());
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	/**
	 * Squash a memory's content into a compact semantic seed (URL + API safe).
	 * Prefers a word boundary near the cap so the seed never ends mid-word (or,
	 * worse, mid-surrogate-pair — raw slice cuts UTF-16 code units).
	 */
	function walkQuery(content: string): string {
		const squashed = content.replace(/\s+/g, ' ').trim();
		if (squashed.length <= 280) return squashed;
		const hard = squashed.slice(0, 280);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > 200 ? hard.slice(0, lastSpace) : hard;
	}

	/** Sync ?q= without a history entry; a failed sync is warned, never silent. */
	function syncQueryToUrl() {
		try {
			replaceState(`?q=${encodeURIComponent(searchQuery)}`, {});
		} catch (err) {
			console.warn('[explore] URL sync failed — walk state is not shareable:', err);
		}
	}

	async function loadNeighbors() {
		const gen = ++loadGen;
		error = null;
		// The search backend rejects an empty query (500). With no ?q= we AUTO-SEED
		// the expedition from the newest memory (verified: /api/memories returns
		// newest-first) — landing here is immediately alive, and every click walks
		// onward from a real thought.
		if (!searchQuery.trim()) {
			try {
				// If another organ handed us a memory (?memory=<id>), seed the walk
				// from THAT thought so the selection survives the handoff. Otherwise
				// auto-seed from the newest memory so landing here is always alive.
				const seedMemory = seedMemoryId
					? await api.memories.get(seedMemoryId).catch(() => null)
					: null;
				const start = seedMemory ?? (await api.memories.list({ limit: '1' })).memories?.[0] ?? null;
				if (gen !== loadGen) return; // a newer walk superseded this load
				if (start?.content) {
					searchQuery = walkQuery(start.content);
					inputValue = searchQuery;
					centerId = start.id;
					centerContent = start.content;
					seededFrom = start.id.slice(0, 8);
					// Consume the one-shot seed so a later re-walk doesn't re-pin it.
					seedMemoryId = null;
					// The seeded expedition must be shareable/reproducible too — a
					// reload mid-ingest would otherwise seed from a different memory.
					syncQueryToUrl();
				}
			} catch {
				/* fall through to the calm empty state */
			}
		}
		if (!searchQuery.trim()) {
			loading = false;
			searchResult = null;
			results = [];
			textPass?.setText(buildTextItems());
			engineRef?.demoClock.reset();
			return;
		}
		loading = true;
		textPass?.setText(buildTextItems());
		try {
			// If we know the exact center memory (auto-seed or a click-walk) we ask
			// the graph for its true associations; otherwise the raw query is a
			// semantic search over the corpus.
			const res = centerId
				? await api.explore(centerId, 'associations', undefined, SEARCH_LIMIT)
				: await api.search(searchQuery, SEARCH_LIMIT);
			if (gen !== loadGen) return; // stale response — the newer walk owns state
			searchResult = 'query' in res ? (res as SearchResult) : null;
			results = (res.results as SearchMemory[]) ?? [];
		} catch (err) {
			if (gen !== loadGen) return;
			searchResult = null;
			results = [];
			error = err instanceof Error ? err.message : 'UNKNOWN EXPLORE FETCH ERROR';
		} finally {
			if (gen === loadGen) {
				loading = false;
				walkingFrom = null; // only the owning generation clears the label
				// A brand-new neighborhood invalidates any prior selection.
				if (!results.some((m) => m.id === selectedId)) selectedId = null;
				textPass?.setText(buildTextItems());
				// Re-assert intensity for the current aspect (a rotate between load and
				// now would otherwise keep the stale desktop bloom on a phone).
				fieldPass?.setIntensity(fieldIntensity());
				if (fieldPass) applyReadingWell(fieldPass);
				fieldPass?.setCells(buildFieldCells());
				engineRef?.demoClock.reset();
			}
		}
	}

	function buildFieldCells() {
		const data: FieldDatum[] = results.map((memory, index) => ({
			id: memory.id,
			score: similarity(memory),
			hue: index === 0 ? FIELD_HUE.oxygen : FIELD_HUE.bridge,
			energy: similarity(memory),
			metric2: retention(memory),
			selected: memory.id === selectedId || (selectedId === null && index === 0),
			kind: 'explore-neighbor',
			payload: memory
		}));
		return layoutGalaxy(data, { maxRadius: 0.9, minCellR: 0.035, maxCellR: 0.1 });
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

	function scalar(value: unknown, fallback = 0.5): number {
		return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
	}

	function similarity(memory: SearchMemory): number {
		return clamp01(
			scalar(
				memory.similarity,
				scalar(memory.score, scalar(memory.relevance, scalar(memory.combinedScore, memory.retrievalStrength)))
			)
		);
	}

	function retention(memory: SearchMemory): number {
		return clamp01(scalar(memory.retention, memory.retentionStrength));
	}

	// --- Real derived stats for the LIVE PROOF cards (never constants) ---
	const neighborCount = $derived(results.length);
	const avgSimilarity = $derived(
		results.length ? results.reduce((s, m) => s + similarity(m), 0) / results.length : 0
	);
	const topSimilarity = $derived(
		results.length ? Math.max(...results.map((m) => similarity(m))) : 0
	);
	const searchMs = $derived<number>(searchResult ? searchResult.durationMs : 0);

	// Trim to a cap on a word boundary near the cap so a portrait row never ends
	// mid-token; hard-slice fallback for a single unbroken token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	// The DOM overlay is the ONE reading surface: it carries the full neighbor list
	// (content preview + similarity %) as legible, click-to-walk rows. The WebGPU
	// field stays alive purely as the constellation of glowing cells behind it —
	// we deliberately emit NO MSDF text so the field never doubles as a raw id/data
	// dump behind the glass (the exact failure mode the redesign is fixing).
	function buildTextItems(): ExploreTextItem[] {
		return [];
	}

	function pointerToNdc(e: PointerEvent | MouseEvent): { x: number; y: number } | null {
		if (!hostEl) return null;
		const rect = hostEl.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return null;
		return {
			x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
			y: -(((e.clientY - rect.top) / rect.height) * 2 - 1)
		};
	}

	function writeCursorLens(ndc: { x: number; y: number }) {
		if (!hostEl || !engineRef) return;
		const rect = hostEl.getBoundingClientRect();
		const aspect = Math.max(0.0001, rect.width / Math.max(1, rect.height));
		const raw = { x: ndc.x * Math.max(aspect, 1), y: ndc.y / Math.min(aspect, 1) };
		const prev = cursorSmoothed ?? raw;
		const next = { x: prev.x + (raw.x - prev.x) * 0.35, y: prev.y + (raw.y - prev.y) * 0.35 };
		cursorSmoothed = next;
		engineRef.setCursorPreNdc(next.x, next.y, next.x - prev.x, next.y - prev.y);
	}

	function handlePointerMove(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		writeCursorLens(ndc);
		const hit = textPass?.pickAt(ndc.x, ndc.y) ?? null;
		const nextRun = hit?.kind === 'explore-neighbor' ? hit.id : null;
		if (nextRun !== activeRun) {
			activeRun = nextRun;
			textPass?.setRunDepth(nextRun, 1);
		}
		if (hostEl) hostEl.style.cursor = nextRun ? 'crosshair' : 'default';
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		activeRun = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		textPass?.setRunDepth(null);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	// Clicking a constellation label SELECTS that neighbor (non-mutating: it only
	// opens the detail panel). Walking is the explicit labelled button in the panel.
	function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc || !textPass) return;
		const hit = textPass.pickAt(ndc.x, ndc.y);
		if (hit?.kind !== 'explore-neighbor') return;
		const item = hit.payload as ExploreTextItem;
		if (!item.memoryId) return;
		selectNeighbor(item.memoryId);
	}

	// --- DOM interactions ---

	// SELECT (non-mutating): open the detail panel for a neighbor, refresh the
	// field selection highlight. No API call, no re-center.
	function selectNeighbor(id: string) {
		selectedId = selectedId === id ? null : id;
		fieldPass?.setCells(buildFieldCells());
		textPass?.setText(buildTextItems());
	}

	// WALK (explicit action): re-center the neighborhood on a memory. This is the
	// only path that mutates the view; it is always driven by a labelled control.
	async function walkTo(memory: SearchMemory) {
		if (!memory.content) return;
		walkingFrom = memory.id.slice(0, 8);
		centerId = memory.id;
		centerContent = memory.content;
		seededFrom = null;
		selectedId = null;
		searchQuery = walkQuery(memory.content);
		inputValue = searchQuery;
		syncQueryToUrl();
		await loadNeighbors();
	}

	// Commit the search box: a free-text query is a semantic search (no fixed
	// center), so clear centerId and let the /search path run.
	async function runSearch() {
		const q = inputValue.trim();
		if (!q) return;
		centerId = null;
		centerContent = null;
		seededFrom = null;
		selectedId = null;
		searchQuery = q;
		syncQueryToUrl();
		await loadNeighbors();
	}

	function onSearchKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			void runSearch();
		}
	}

	// The primary action is disabled with a reason when there is nothing to commit.
	const primaryDisabled = $derived(loading || inputValue.trim().length === 0);
	const primaryDisabledReason = $derived(
		loading ? 'Walking…' : inputValue.trim().length === 0 ? 'Type a thought to search' : ''
	);
</script>

<svelte:head>
	<title>Semantic Explorer · Vestige</title>
</svelte:head>

<!-- Living field BEHIND (unchanged WebGPU aesthetic) -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={hostEl}
	class="fixed inset-0 bg-[#020307]"
	onpointerdown={handlePointerDown}
	onpointermove={handlePointerMove}
	onpointerleave={handlePointerLeave}
>
	<ObservatoryCanvas
		demo="recall-path"
		seed={`real-explore-neighbors:${searchQuery}:${searchResult?.total ?? 0}`}
		onready={handleReady}
	/>
</div>

<!-- DOM overlay ON TOP (contradictions hybrid pattern) -->
<div class="relative z-10 min-h-full p-6 space-y-6 pointer-events-none">
	<div class="pointer-events-auto">
		<PageHeader
			icon="explore"
			title="Semantic Explorer"
			subtitle="Walk your memory by meaning: pick a thought, see its nearest neighbors, walk deeper."
			accent="synapse"
		>
			{#if seededFrom}
				<span class="text-dim text-xs tabular-nums inline-flex items-center gap-1.5">
					seeded from {seededFrom}
				</span>
			{/if}
		</PageHeader>
	</div>

	<!-- PRIMARY ACTION: visible search / seed input + Walk button -->
	<div class="pointer-events-auto glass rounded-xl p-3 flex flex-col sm:flex-row gap-2 sm:items-center">
		<div class="relative flex-1">
			<span class="absolute left-3 top-1/2 -translate-y-1/2 text-dim pointer-events-none">
				<Icon name="search" size={16} />
			</span>
			<input
				type="text"
				bind:value={inputValue}
				onkeydown={onSearchKeydown}
				placeholder="Search a thought by meaning…"
				aria-label="Search memory by meaning"
				class="w-full bg-white/[0.03] border border-subtle/25 rounded-lg pl-9 pr-3 py-2.5 text-sm text-text placeholder:text-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/50 transition"
			/>
		</div>
		<button
			type="button"
			onclick={runSearch}
			disabled={primaryDisabled}
			title={primaryDisabledReason}
			class="inline-flex items-center justify-center gap-2 rounded-lg bg-synapse/20 px-4 py-2.5 text-sm font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60 disabled:opacity-40 disabled:cursor-not-allowed"
		>
			<Icon name="explore" size={15} />
			{loading ? 'Walking…' : 'Explore'}
		</button>
	</div>
	{#if primaryDisabled && primaryDisabledReason}
		<div class="pointer-events-auto -mt-4 text-[11px] text-muted pl-1">{primaryDisabledReason}</div>
	{/if}

	{#if error}
		<div
			class="glass-panel pointer-events-auto flex flex-col items-center gap-3 rounded-2xl p-10 text-center"
		>
			<div class="text-sm text-decay">Couldn't load neighbors</div>
			<div class="max-w-md text-xs text-muted break-words">{error}</div>
			<button
				type="button"
				onclick={loadNeighbors}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			{#each Array(4) as _}
				<div class="glass-subtle shimmer h-20 rounded-xl"></div>
			{/each}
		</div>
		<div class="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 pointer-events-auto">
			<div class="glass-subtle shimmer min-h-[520px] rounded-2xl"></div>
			<div class="glass-subtle shimmer h-[520px] rounded-2xl"></div>
		</div>
	{:else if results.length === 0}
		<div
			class="glass-panel pointer-events-auto enter flex flex-col items-center gap-3 rounded-2xl p-12 text-center"
		>
			<div
				class="flex h-14 w-14 items-center justify-center rounded-2xl border border-synapse/25 bg-synapse/10 text-synapse"
			>
				<Icon name="explore" size={26} draw />
			</div>
			<div class="text-sm font-medium text-bright">
				{searchQuery.trim()
					? `No neighbors for "${searchQuery.slice(0, 60)}"`
					: 'Nothing to explore yet'}
			</div>
			<div class="max-w-sm text-xs text-muted">
				{searchQuery.trim()
					? 'Try a broader phrase, or ingest more memories on this topic.'
					: 'Ingest a memory, then search a thought above to walk its nearest neighbors by meaning.'}
			</div>
		</div>
	{:else}
		<!-- LIVE PROOF: real stat cards -->
		<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 pointer-events-auto">
			<div use:reveal={{ delay: 0, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={neighborCount} />
				</div>
				<div class="text-xs text-dim mt-1">nearest neighbors</div>
			</div>
			<div use:reveal={{ delay: 60, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #22C7DE">
					<AnimatedNumber value={avgSimilarity} scale={100} decimals={0} suffix="%" />
				</div>
				<div class="text-xs text-dim mt-1">avg similarity</div>
			</div>
			<div use:reveal={{ delay: 120, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl font-bold tabular-nums" style="color: #29F2A9">
					<AnimatedNumber value={topSimilarity} scale={100} decimals={0} suffix="%" />
				</div>
				<div class="text-xs text-dim mt-1">closest match</div>
			</div>
			<div use:reveal={{ delay: 180, y: 12 }} class="p-4 glass rounded-xl lift">
				<div class="text-2xl text-bright font-bold tabular-nums">
					<AnimatedNumber value={searchMs} suffix="ms" />
				</div>
				<div class="text-xs text-dim mt-1">retrieval time</div>
			</div>
		</div>

		<!-- INTERPRETATION: where you are -->
		{#if centerContent}
			<div class="pointer-events-auto glass-subtle rounded-xl px-4 py-3 text-xs text-dim flex items-start gap-2">
				<span class="text-synapse-glow mt-0.5 shrink-0"><Icon name="search" size={14} /></span>
				<span>
					<span class="text-muted uppercase tracking-wider text-[10px]">Centered on</span>
					<span class="text-text ml-1">{trimSnippet(centerContent, 120)}</span>
				</span>
			</div>
		{/if}

		<!-- Neighborhood list + detail panel -->
		<div class="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-4 pointer-events-auto">
			<div class="glass-panel rounded-2xl p-3 space-y-2 max-h-[640px] overflow-y-auto">
				<div
					class="flex items-center justify-between px-1 pb-2 sticky top-0 bg-deep/60 backdrop-blur-sm z-10"
				>
					<span class="text-xs text-dim uppercase tracking-wider">Neighborhood · click to select, walk to re-center</span>
					<span class="text-xs text-muted tabular-nums"><AnimatedNumber value={neighborCount} /></span>
				</div>

				{#each results as memory, i (memory.id)}
					{@const sim = similarity(memory)}
					{@const isSel = selectedId === memory.id}
					<button
						use:reveal={{ delay: Math.min(i * 30, 320), y: 10 }}
						onclick={() => selectNeighbor(memory.id)}
						class="w-full text-left p-3 rounded-xl border transition lift
							{isSel
								? 'bg-synapse/10 border-synapse/40 shadow-[0_0_12px_rgba(99,102,241,0.18)]'
								: 'border-subtle/20 hover:border-synapse/30 hover:bg-white/[0.02]'}"
					>
						<div class="flex items-center gap-2 mb-1.5">
							<div
								class="w-2 h-2 rounded-full shrink-0"
								style="background: #22C7DE; opacity: {0.35 + sim * 0.65}"
							></div>
							<span class="text-[10px] uppercase tracking-wider text-muted">{memory.nodeType}</span>
							<span class="ml-auto text-[11px] tabular-nums font-medium" style="color: #22C7DE">
								{Math.round(sim * 100)}% match
							</span>
						</div>
						<div class="text-xs text-text leading-relaxed">
							{trimSnippet(memory.content, isSel ? 300 : 140)}
						</div>
						<div class="mt-1.5 flex items-center gap-3 text-[10px] text-muted tabular-nums">
							<span>{memory.id.slice(0, 8)}</span>
							<span>retention {Math.round(retention(memory) * 100)}%</span>
						</div>
					</button>
				{/each}
			</div>

			<!-- Detail / walk panel -->
			<aside class="space-y-3">
				{#if selected}
					<div use:reveal={{ y: 12 }} class="glass-panel rounded-2xl p-5 space-y-4">
						<div class="flex items-start justify-between gap-3 border-b border-subtle/20 pb-3">
							<div>
								<div class="font-mono text-[10px] uppercase tracking-[0.22em] text-synapse-glow">
									Selected thought
								</div>
								<div class="mt-1 text-[11px] text-muted font-mono break-all">{selected.id}</div>
							</div>
							<button
								type="button"
								onclick={() => selectNeighbor(selected!.id)}
								class="rounded-lg border border-subtle/30 px-2.5 py-1 text-xs text-muted transition hover:border-synapse/40 hover:text-synapse-glow"
							>
								<Icon name="close" size={13} />
							</button>
						</div>
						<p class="text-sm text-text leading-relaxed">{selected.content}</p>
						<div class="grid grid-cols-2 gap-2">
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">similarity</div>
								<div class="mt-0.5 font-mono text-lg" style="color: #22C7DE">
									{Math.round(similarity(selected) * 100)}%
								</div>
							</div>
							<div class="rounded-lg bg-white/[0.03] p-2.5">
								<div class="text-[10px] uppercase tracking-wider text-muted">retention</div>
								<div class="mt-0.5 font-mono text-lg" style="color: #29F2A9">
									{Math.round(retention(selected) * 100)}%
								</div>
							</div>
						</div>
						{#if selected.tags && selected.tags.length > 0}
							<div class="flex flex-wrap gap-1">
								{#each selected.tags as t}
									<span class="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{t}</span>
								{/each}
							</div>
						{/if}
						<!-- WALK: the explicit, labelled re-center action (never a silent mutation) -->
						<button
							type="button"
							onclick={() => walkTo(selected!)}
							disabled={loading}
							class="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-synapse/20 px-4 py-2.5 text-sm font-medium text-synapse-glow transition hover:bg-synapse/30 focus:outline-none focus-visible:ring-2 focus-visible:ring-synapse/60 disabled:opacity-40 disabled:cursor-not-allowed"
						>
							<Icon name="explore" size={15} />
							Walk from this thought
						</button>
					</div>
				{:else}
					<div class="glass-subtle rounded-2xl p-6 text-center space-y-2">
						<div class="text-dim opacity-60 flex justify-center breathe">
							<Icon name="graph" size={34} strokeWidth={1.2} />
						</div>
						<p class="text-xs text-dim">
							Click a neighbor to inspect it. Then <span class="text-synapse-glow">Walk</span> to
							re-center the map on that thought and explore deeper.
						</p>
					</div>
				{/if}
			</aside>
		</div>
	{/if}
</div>

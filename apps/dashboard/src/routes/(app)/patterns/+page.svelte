<!--
  Cross-Project Intelligence — Pattern Transfer Heatmap
  Dashboard exposure of the CrossProjectLearner backend state. Visualizes
  which coding patterns were learned in one project and reused in another,
  across all six tracked categories (ErrorHandling, AsyncConcurrency, Testing,
  Architecture, Performance, Security).

  Category tabs filter the pattern set. Heatmap cell click filters the
  "Top Transferred Patterns" sidebar to a specific origin → destination pair.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import PatternTransferHeatmap from '$components/PatternTransferHeatmap.svelte';

	type Category =
		| 'ErrorHandling'
		| 'AsyncConcurrency'
		| 'Testing'
		| 'Architecture'
		| 'Performance'
		| 'Security';

	interface Pattern {
		name: string;
		category: Category;
		origin_project: string;
		transferred_to: string[];
		transfer_count: number;
		last_used: string;
		confidence: number;
	}

	interface CrossProjectResponse {
		projects: string[];
		patterns: Pattern[];
	}

	const CATEGORIES: readonly Category[] = [
		'ErrorHandling',
		'AsyncConcurrency',
		'Testing',
		'Architecture',
		'Performance',
		'Security'
	] as const;

	const CATEGORY_COLORS: Record<Category, string> = {
		ErrorHandling: 'var(--color-decay)',
		AsyncConcurrency: 'var(--color-synapse-glow)',
		Testing: 'var(--color-recall)',
		Architecture: 'var(--color-dream-glow)',
		Performance: 'var(--color-warning)',
		Security: 'var(--color-node-pattern)'
	};

	let activeCategory = $state<'All' | Category>('All');
	let data = $state<CrossProjectResponse>({ projects: [], patterns: [] });
	let loading = $state(true);
	let error: string | null = $state(null);
	let selectedCell = $state<{ from: string; to: string } | null>(null);

	// TODO: swap for real fetch to /api/patterns/cross-project when backend ships.
	// The CrossProjectLearner already tracks these categories in Rust — exposing
	// it over HTTP is a straightforward map-to-DTO. Matching shape below so the
	// swap is a one-liner.
	async function mockFetchCrossProject(): Promise<CrossProjectResponse> {
		await new Promise((r) => setTimeout(r, 420));

		const projects = [
			'vestige',
			'nullgaze',
			'injeranet',
			'nemotron',
			'orbit-wars',
			'nightvision',
			'aimo3'
		];

		const patterns: Pattern[] = [
			// ErrorHandling — widely transferred
			{
				name: 'Result<T, E> with thiserror context',
				category: 'ErrorHandling',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet', 'nemotron', 'nightvision'],
				transfer_count: 4,
				last_used: '2026-04-18T14:22:00Z',
				confidence: 0.94
			},
			{
				name: 'Axum error middleware with tower-http',
				category: 'ErrorHandling',
				origin_project: 'nullgaze',
				transferred_to: ['vestige', 'nightvision'],
				transfer_count: 2,
				last_used: '2026-04-17T09:10:00Z',
				confidence: 0.88
			},
			{
				name: 'Graceful shutdown on SIGINT/SIGTERM',
				category: 'ErrorHandling',
				origin_project: 'vestige',
				transferred_to: ['vestige', 'injeranet', 'nightvision'],
				transfer_count: 3,
				last_used: '2026-04-15T22:01:00Z',
				confidence: 0.82
			},
			{
				name: 'Python try/except with contextual re-raise',
				category: 'ErrorHandling',
				origin_project: 'aimo3',
				transferred_to: ['nemotron'],
				transfer_count: 1,
				last_used: '2026-04-10T11:30:00Z',
				confidence: 0.7
			},

			// AsyncConcurrency
			{
				name: 'Arc<Mutex<Connection>> reader/writer split',
				category: 'AsyncConcurrency',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet'],
				transfer_count: 2,
				last_used: '2026-04-14T16:42:00Z',
				confidence: 0.91
			},
			{
				name: 'tokio::select! for cancellation propagation',
				category: 'AsyncConcurrency',
				origin_project: 'injeranet',
				transferred_to: ['vestige', 'nightvision'],
				transfer_count: 2,
				last_used: '2026-04-19T08:05:00Z',
				confidence: 0.86
			},
			{
				name: 'Bounded mpsc channel with backpressure',
				category: 'AsyncConcurrency',
				origin_project: 'injeranet',
				transferred_to: ['vestige', 'nullgaze'],
				transfer_count: 2,
				last_used: '2026-04-12T13:18:00Z',
				confidence: 0.83
			},
			{
				name: 'asyncio.gather with return_exceptions',
				category: 'AsyncConcurrency',
				origin_project: 'nemotron',
				transferred_to: ['aimo3'],
				transfer_count: 1,
				last_used: '2026-04-08T20:45:00Z',
				confidence: 0.72
			},

			// Testing
			{
				name: 'Property-based tests with proptest',
				category: 'Testing',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet'],
				transfer_count: 2,
				last_used: '2026-04-11T10:22:00Z',
				confidence: 0.89
			},
			{
				name: 'Snapshot testing with insta',
				category: 'Testing',
				origin_project: 'nullgaze',
				transferred_to: ['vestige'],
				transfer_count: 1,
				last_used: '2026-04-16T14:00:00Z',
				confidence: 0.81
			},
			{
				name: 'Vitest + Playwright dashboard harness',
				category: 'Testing',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet'],
				transfer_count: 2,
				last_used: '2026-04-19T18:30:00Z',
				confidence: 0.87
			},
			{
				name: 'One-variable-at-a-time Kaggle submission',
				category: 'Testing',
				origin_project: 'aimo3',
				transferred_to: ['nemotron', 'orbit-wars'],
				transfer_count: 2,
				last_used: '2026-04-20T07:15:00Z',
				confidence: 0.95
			},
			{
				name: 'Kaggle pre-flight Input-panel screenshot',
				category: 'Testing',
				origin_project: 'aimo3',
				transferred_to: ['nemotron', 'orbit-wars'],
				transfer_count: 2,
				last_used: '2026-04-20T06:50:00Z',
				confidence: 0.98
			},

			// Architecture
			{
				name: 'SvelteKit 2 + Svelte 5 runes dashboard',
				category: 'Architecture',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'nightvision'],
				transfer_count: 2,
				last_used: '2026-04-19T12:10:00Z',
				confidence: 0.92
			},
			{
				name: 'glass-panel + cosmic-dark design system',
				category: 'Architecture',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'nightvision', 'injeranet'],
				transfer_count: 3,
				last_used: '2026-04-20T09:00:00Z',
				confidence: 0.9
			},
			{
				name: 'Tauri 2 + Rust/Axum sidecar',
				category: 'Architecture',
				origin_project: 'injeranet',
				transferred_to: ['nightvision'],
				transfer_count: 1,
				last_used: '2026-04-13T19:44:00Z',
				confidence: 0.78
			},
			{
				name: 'MCP server with 23 stateful tools',
				category: 'Architecture',
				origin_project: 'vestige',
				transferred_to: ['injeranet'],
				transfer_count: 1,
				last_used: '2026-04-17T11:05:00Z',
				confidence: 0.85
			},

			// Performance
			{
				name: 'USearch HNSW index for vector search',
				category: 'Performance',
				origin_project: 'vestige',
				transferred_to: ['nullgaze'],
				transfer_count: 1,
				last_used: '2026-04-09T15:20:00Z',
				confidence: 0.88
			},
			{
				name: 'SQLite WAL mode for concurrent reads',
				category: 'Performance',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet', 'nightvision'],
				transfer_count: 3,
				last_used: '2026-04-18T21:33:00Z',
				confidence: 0.93
			},
			{
				name: 'vLLM prefix caching at 0.35 mem util',
				category: 'Performance',
				origin_project: 'aimo3',
				transferred_to: ['nemotron'],
				transfer_count: 1,
				last_used: '2026-04-11T08:00:00Z',
				confidence: 0.84
			},
			{
				name: 'Cross-encoder rerank at k=30',
				category: 'Performance',
				origin_project: 'vestige',
				transferred_to: ['nullgaze'],
				transfer_count: 1,
				last_used: '2026-04-14T17:55:00Z',
				confidence: 0.79
			},

			// Security
			{
				name: 'Rotated auth token in env var',
				category: 'Security',
				origin_project: 'vestige',
				transferred_to: ['nullgaze', 'injeranet', 'nightvision'],
				transfer_count: 3,
				last_used: '2026-04-16T20:12:00Z',
				confidence: 0.96
			},
			{
				name: 'Parameterized SQL via rusqlite params!',
				category: 'Security',
				origin_project: 'vestige',
				transferred_to: ['nullgaze'],
				transfer_count: 1,
				last_used: '2026-04-10T13:40:00Z',
				confidence: 0.89
			},
			{
				name: '664-pattern secret scanner',
				category: 'Security',
				origin_project: 'nullgaze',
				transferred_to: ['vestige', 'nightvision', 'injeranet'],
				transfer_count: 3,
				last_used: '2026-04-20T05:30:00Z',
				confidence: 0.97
			},
			{
				name: 'CSP header with nonce-based script allow',
				category: 'Security',
				origin_project: 'nullgaze',
				transferred_to: ['nightvision'],
				transfer_count: 1,
				last_used: '2026-04-05T16:08:00Z',
				confidence: 0.8
			}
		];

		return { projects, patterns };
	}

	async function load() {
		loading = true;
		error = null;
		try {
			// TODO: const res = await fetch('/api/patterns/cross-project');
			//       data = await res.json();
			data = await mockFetchCrossProject();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load pattern transfers';
			data = { projects: [], patterns: [] };
		} finally {
			loading = false;
		}
	}

	onMount(() => load());

	// Filter by active category first — this drives both the heatmap and sidebar.
	const categoryFiltered = $derived(
		activeCategory === 'All'
			? data.patterns
			: data.patterns.filter((p) => p.category === activeCategory)
	);

	// Sidebar list: if a cell is selected, show only A → B; else show all (top-N
	// by transfer_count). Still respects active category via categoryFiltered.
	const sidebarPatterns = $derived.by(() => {
		const list = selectedCell
			? categoryFiltered.filter(
					(p) =>
						p.origin_project === selectedCell!.from &&
						p.transferred_to.includes(selectedCell!.to)
				)
			: categoryFiltered;
		return [...list].sort((a, b) => b.transfer_count - a.transfer_count);
	});

	// Stats footer
	const totalTransfers = $derived(
		categoryFiltered.reduce((sum, p) => sum + p.transferred_to.length, 0)
	);
	const projectCount = $derived(data.projects.length);
	const patternCount = $derived(categoryFiltered.length);

	function selectCategory(c: 'All' | Category) {
		activeCategory = c;
		selectedCell = null; // clear cell filter when switching category
	}

	function onCellClick(from: string, to: string) {
		if (selectedCell && selectedCell.from === from && selectedCell.to === to) {
			selectedCell = null;
		} else {
			selectedCell = { from, to };
		}
	}

	function clearCellFilter() {
		selectedCell = null;
	}

	function relativeDate(iso: string): string {
		const then = new Date(iso).getTime();
		const now = Date.now();
		const days = Math.floor((now - then) / 86_400_000);
		if (days <= 0) return 'today';
		if (days === 1) return '1d ago';
		if (days < 30) return `${days}d ago`;
		const months = Math.floor(days / 30);
		return `${months}mo ago`;
	}
</script>

<div class="relative mx-auto max-w-7xl space-y-6 p-6">
	<!-- Header -->
	<header class="space-y-2">
		<h1 class="text-xl font-semibold text-bright">Cross-Project Intelligence</h1>
		<p class="text-sm text-dim">Patterns learned here, applied there.</p>
	</header>

	<!-- Category tabs -->
	<div class="glass-panel flex flex-wrap items-center gap-1.5 rounded-2xl p-2">
		<button
			type="button"
			onclick={() => selectCategory('All')}
			class="rounded-lg px-3 py-1.5 text-xs font-medium transition {activeCategory === 'All'
				? 'bg-synapse/25 text-synapse-glow'
				: 'text-dim hover:bg-white/[0.04] hover:text-text'}"
		>
			All
		</button>
		{#each CATEGORIES as cat (cat)}
			<button
				type="button"
				onclick={() => selectCategory(cat)}
				class="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition {activeCategory ===
				cat
					? 'bg-synapse/25 text-synapse-glow'
					: 'text-dim hover:bg-white/[0.04] hover:text-text'}"
			>
				<span
					class="h-1.5 w-1.5 rounded-full"
					style="background: {CATEGORY_COLORS[cat]}"
				></span>
				{cat}
			</button>
		{/each}
	</div>

	{#if error}
		<div class="glass-panel flex flex-col items-center gap-3 rounded-2xl p-10 text-center">
			<div class="text-sm text-decay">Couldn't load pattern transfers</div>
			<div class="max-w-md text-xs text-muted">{error}</div>
			<button
				type="button"
				onclick={load}
				class="mt-2 rounded-lg bg-synapse/20 px-4 py-2 text-xs font-medium text-synapse-glow transition hover:bg-synapse/30"
			>
				Retry
			</button>
		</div>
	{:else if loading}
		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
			<div class="glass-subtle h-[520px] animate-pulse rounded-2xl"></div>
			<div class="glass-subtle h-[520px] animate-pulse rounded-2xl"></div>
		</div>
	{:else}
		<!-- Main grid: heatmap (70%) + sidebar -->
		<div class="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
			<!-- Heatmap column -->
			<div class="space-y-4">
				<PatternTransferHeatmap
					projects={data.projects}
					patterns={categoryFiltered}
					{selectedCell}
					{onCellClick}
				/>

				{#if selectedCell}
					<div
						class="glass-subtle flex items-center justify-between rounded-xl px-4 py-2.5 text-xs"
					>
						<div class="flex items-center gap-2">
							<span class="text-muted">Filtered to</span>
							<span class="font-mono text-bright">{selectedCell.from}</span>
							<span class="text-synapse-glow">→</span>
							<span class="font-mono text-bright">{selectedCell.to}</span>
						</div>
						<button
							type="button"
							onclick={clearCellFilter}
							class="rounded-md bg-white/[0.04] px-2 py-1 text-dim transition hover:bg-white/[0.08] hover:text-text"
						>
							Clear
						</button>
					</div>
				{/if}
			</div>

			<!-- Sidebar: Top Transferred Patterns -->
			<aside class="glass-panel flex flex-col rounded-2xl p-4">
				<div class="mb-3 flex items-center justify-between">
					<h2 class="text-sm font-semibold text-bright">Top Transferred Patterns</h2>
					<span class="text-[11px] text-muted">
						{sidebarPatterns.length}
						{sidebarPatterns.length === 1 ? 'pattern' : 'patterns'}
					</span>
				</div>

				{#if sidebarPatterns.length === 0}
					<div class="flex flex-1 flex-col items-center justify-center gap-2 py-10 text-center">
						<div class="text-xs font-medium text-dim">No matching patterns</div>
						<div class="max-w-[220px] text-[11px] text-muted">
							{selectedCell
								? 'No patterns transferred from this origin to this destination.'
								: 'No patterns in this category.'}
						</div>
					</div>
				{:else}
					<ul class="flex-1 space-y-2 overflow-y-auto pr-1" style="max-height: 560px;">
						{#each sidebarPatterns as p (p.name)}
							<li
								class="rounded-lg border border-synapse/5 bg-white/[0.02] p-3 transition hover:border-synapse/20 hover:bg-white/[0.04]"
							>
								<div class="flex items-start justify-between gap-2">
									<div class="min-w-0 flex-1 space-y-1.5">
										<div class="truncate text-xs font-medium text-bright" title={p.name}>
											{p.name}
										</div>
										<div class="flex flex-wrap items-center gap-1.5">
											<span
												class="rounded-full border px-1.5 py-0.5 text-[10px] font-medium"
												style="border-color: {CATEGORY_COLORS[
													p.category
												]}66; color: {CATEGORY_COLORS[p.category]}; background: {CATEGORY_COLORS[
													p.category
												]}1a;"
											>
												{p.category}
											</span>
											<span class="text-[10px] text-muted">{relativeDate(p.last_used)}</span>
										</div>
										<div class="flex items-center gap-1.5 text-[11px] text-dim">
											<span class="font-mono text-text">{p.origin_project}</span>
											<span class="text-synapse-glow">→</span>
											<span class="text-muted">
												{p.transferred_to.length}
												{p.transferred_to.length === 1 ? 'project' : 'projects'}
											</span>
										</div>
									</div>
									<div class="flex flex-shrink-0 flex-col items-end gap-1">
										<span
											class="rounded-full bg-synapse/15 px-2 py-0.5 text-xs font-semibold text-synapse-glow"
										>
											{p.transfer_count}
										</span>
										<span class="text-[10px] text-muted">
											{(p.confidence * 100).toFixed(0)}%
										</span>
									</div>
								</div>
							</li>
						{/each}
					</ul>
				{/if}
			</aside>
		</div>

		<!-- Stats footer -->
		<footer
			class="glass-subtle flex flex-wrap items-center justify-between gap-3 rounded-xl px-4 py-3 text-xs text-dim"
		>
			<div>
				<span class="font-semibold text-bright">{patternCount}</span>
				pattern{patternCount === 1 ? '' : 's'} across
				<span class="font-semibold text-bright">{projectCount}</span>
				project{projectCount === 1 ? '' : 's'},
				<span class="font-semibold text-bright">{totalTransfers}</span>
				total transfer{totalTransfers === 1 ? '' : 's'}
			</div>
			<div class="text-muted">
				{activeCategory === 'All' ? 'All categories' : activeCategory}
			</div>
		</footer>
	{/if}
</div>

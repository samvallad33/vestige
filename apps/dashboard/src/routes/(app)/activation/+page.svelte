<script lang="ts">
	/**
	 * Spreading Activation Live View.
	 *
	 * Two sources of bursts feed the ActivationNetwork canvas:
	 *   1. User search — type a query, we pick the top-1 match and fetch its
	 *      associations (up to 15), then pass `{source, neighbours}` as props.
	 *   2. Live mode — subscribe to `$eventFeed` and, on every NEW
	 *      `ActivationSpread` event, trigger an overlay burst at a randomised
	 *      offset. Old events (those present before mount, or already
	 *      processed) never re-fire; we track `lastSeen` by object identity
	 *      so overlapping batches inside the same Svelte update tick are
	 *      still handled.
	 *
	 * All heavy lifting (decay, geometry, color, event filter) lives in
	 * `$components/activation-helpers` so it's unit-tested in Node without
	 * a browser.
	 */
	import { onMount, onDestroy } from 'svelte';
	import { api } from '$stores/api';
	import { eventFeed } from '$stores/websocket';
	import ActivationNetwork, {
		type ActivationNode,
	} from '$components/ActivationNetwork.svelte';
	import { filterNewSpreadEvents } from '$components/activation-helpers';
	import type { Memory, VestigeEvent } from '$types';

	let searchQuery = $state('');
	let loading = $state(false);
	let searched = $state(false); // true after the first submitted search
	let errorMessage = $state<string | null>(null);

	let focusedSource = $state<ActivationNode | null>(null);
	let focusedNeighbours = $state<ActivationNode[]>([]);

	let liveEnabled = $state(true);
	let liveBurstKey = $state(0);
	let liveBurst = $state<{
		source: ActivationNode;
		neighbours: ActivationNode[];
	} | null>(null);
	let liveBurstsFired = $state(0);

	// Track every memory we've seen so live-mode events (which carry only
	// IDs) can be rendered with real labels + node types. If a spread event
	// references an unknown ID we fall back to a short hash so the burst
	// still renders — this mirrors how the 3D graph degrades gracefully.
	const memoryCache = new Map<string, Memory>();

	function rememberMemory(m: Memory) {
		memoryCache.set(m.id, m);
	}

	function memoryToNode(m: Memory): ActivationNode {
		return {
			id: m.id,
			label: labelFor(m.content, m.id),
			nodeType: m.nodeType,
		};
	}

	function labelFor(content: string | undefined, id: string): string {
		if (content && content.trim().length > 0) {
			const trimmed = content.trim();
			return trimmed.length > 60 ? trimmed.slice(0, 60) + '…' : trimmed;
		}
		return id.slice(0, 8);
	}

	function fallbackNode(id: string): ActivationNode {
		const cached = memoryCache.get(id);
		if (cached) return memoryToNode(cached);
		return { id, label: id.slice(0, 8), nodeType: 'note' };
	}

	// ────────────────────────────────────────────────────────────────
	// User-driven search → focused burst
	// ────────────────────────────────────────────────────────────────

	async function runSearch() {
		const q = searchQuery.trim();
		if (!q) {
			// Empty query is a no-op — don't clobber the current burst.
			errorMessage = null;
			return;
		}

		loading = true;
		searched = true;
		errorMessage = null;
		focusedSource = null;
		focusedNeighbours = [];

		try {
			const searchRes = await api.search(q, 1);
			if (!searchRes.results || searchRes.results.length === 0) {
				// Leave `searched=true` + `focusedSource=null` → UI shows
				// the "no matches" empty state rather than crashing on
				// `searchRes.results[0]`.
				return;
			}
			const top = searchRes.results[0];
			rememberMemory(top);
			focusedSource = memoryToNode(top);

			const assocRes = (await api.explore(top.id, 'associations', undefined, 15)) as
				| {
						results?: Memory[];
						nodes?: Memory[];
						// The backend has shipped at least two shapes; accept both.
						associations?: Memory[];
				  }
				| null
				| undefined;

			const rawList =
				assocRes?.results ?? assocRes?.nodes ?? assocRes?.associations ?? [];

			const neighbours: ActivationNode[] = [];
			for (const n of rawList) {
				if (!n || typeof n !== 'object' || !('id' in n)) continue;
				const mem = n as Memory;
				rememberMemory(mem);
				neighbours.push(memoryToNode(mem));
			}
			focusedNeighbours = neighbours;
		} catch (e) {
			errorMessage = e instanceof Error ? e.message : String(e);
			focusedSource = null;
			focusedNeighbours = [];
		} finally {
			loading = false;
		}
	}

	// ────────────────────────────────────────────────────────────────
	// Live mode — $eventFeed → overlay bursts
	// ────────────────────────────────────────────────────────────────

	let feedUnsub: (() => void) | null = null;
	// Object identity of the most recently processed event. We walk the
	// feed head until we hit this reference, so mid-burst batches in one
	// Svelte tick are all processed. Mirrors toast.ts.
	let lastSeenEvent: VestigeEvent | null = null;
	let primedLiveBaseline = false;

	onMount(() => {
		feedUnsub = eventFeed.subscribe((events) => {
			if (!events || events.length === 0) return;
			// Prime lastSeen to the current head BEFORE we're live — we don't
			// want to flood the canvas with every ActivationSpread in the
			// 200-event ring buffer on first mount. Post-prime, only new
			// events fire bursts.
			if (!primedLiveBaseline) {
				lastSeenEvent = events[0];
				primedLiveBaseline = true;
				return;
			}
			if (!liveEnabled) {
				// Still advance the baseline so toggling live back on doesn't
				// dump a backlog.
				lastSeenEvent = events[0];
				return;
			}
			const spreads = filterNewSpreadEvents(events, lastSeenEvent);
			lastSeenEvent = events[0];
			if (spreads.length === 0) return;
			for (const s of spreads) {
				const srcNode = fallbackNode(s.source_id);
				const nbrs = s.target_ids.map((tid) => fallbackNode(tid));
				liveBurstKey += 1;
				liveBurst = { source: srcNode, neighbours: nbrs };
				liveBurstsFired += 1;
			}
		});
	});

	onDestroy(() => {
		if (feedUnsub) feedUnsub();
	});
</script>

<div class="p-6 max-w-6xl mx-auto space-y-6">
	<header class="space-y-2">
		<h1 class="text-xl text-bright font-semibold">Spreading Activation</h1>
		<p class="text-xs text-muted">
			Collins &amp; Loftus 1975 — activation spreads from a seed memory to
			neighbours along semantic edges, decaying by 0.93 per animation frame
			until it drops below 0.05. Search seeds a focused burst; live mode
			overlays every spread event fired by the cognitive engine in real time.
		</p>
	</header>

	<!-- Search -->
	<div class="space-y-3">
		<span class="text-xs text-dim font-medium">Seed Memory</span>
		<div class="flex gap-2">
			<input
				type="text"
				placeholder="Search for a memory to activate..."
				bind:value={searchQuery}
				onkeydown={(e) => e.key === 'Enter' && runSearch()}
				class="flex-1 px-4 py-2.5 bg-white/[0.03] border border-synapse/10 rounded-xl text-text text-sm
					placeholder:text-muted focus:outline-none focus:border-synapse/40 transition backdrop-blur-sm"
			/>
			<button
				onclick={runSearch}
				disabled={loading}
				class="px-4 py-2.5 bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm rounded-xl hover:bg-synapse/30 transition disabled:opacity-50"
			>
				{loading ? 'Activating…' : 'Activate'}
			</button>
		</div>
	</div>

	<!-- Live toggle + stats -->
	<div class="flex items-center justify-between text-xs">
		<label class="flex items-center gap-2 text-dim cursor-pointer select-none">
			<input
				type="checkbox"
				bind:checked={liveEnabled}
				class="accent-synapse-glow"
			/>
			<span>Live mode — overlay bursts from cognitive engine events</span>
		</label>
		<span class="text-muted">
			Live bursts fired: <span class="text-synapse-glow">{liveBurstsFired}</span>
		</span>
	</div>

	<!-- Canvas + empty/error states -->
	<div
		class="glass rounded-2xl overflow-hidden !border-synapse/15 bg-deep/40"
		style="min-height: 560px;"
	>
		{#if loading}
			<div class="flex items-center justify-center h-[560px] text-dim">
				<div class="text-center">
					<div class="text-2xl animate-pulse mb-2">◎</div>
					<p class="text-sm">Computing activation...</p>
				</div>
			</div>
		{:else if errorMessage}
			<div class="flex items-center justify-center h-[560px] text-dim">
				<div class="text-center max-w-md px-6">
					<div class="text-3xl opacity-30 mb-3">⚠</div>
					<p class="text-sm text-bright mb-1">Activation failed</p>
					<p class="text-xs text-muted">{errorMessage}</p>
				</div>
			</div>
		{:else if !focusedSource && searched}
			<div class="flex items-center justify-center h-[560px] text-dim">
				<div class="text-center max-w-md px-6">
					<div class="text-3xl opacity-20 mb-3">◬</div>
					<p class="text-sm text-bright mb-1">No matching memory</p>
					<p class="text-xs text-muted">
						Nothing in the graph matches
						<span class="text-text">"{searchQuery}"</span>. Try a broader
						query or switch on live mode to watch the engine fire its own
						bursts.
					</p>
				</div>
			</div>
		{:else if !focusedSource}
			<div class="flex items-center justify-center h-[560px] text-dim">
				<div class="text-center max-w-md px-6">
					<div class="text-3xl opacity-20 mb-3">◎</div>
					<p class="text-sm text-bright mb-1">Waiting for activation</p>
					<p class="text-xs text-muted">
						Seed a burst with the search bar above, or enable live mode to
						overlay bursts from the cognitive engine as they happen.
					</p>
				</div>
			</div>
		{:else}
			<ActivationNetwork
				width={1040}
				height={560}
				source={focusedSource}
				neighbours={focusedNeighbours}
				{liveBurstKey}
				{liveBurst}
			/>
		{/if}
	</div>

	<!-- Focused burst metadata -->
	{#if focusedSource}
		<div class="p-3 glass rounded-xl !border-synapse/20">
			<div class="text-[10px] text-synapse-glow mb-1 uppercase tracking-wider">
				Seed
			</div>
			<p class="text-sm text-text">{focusedSource.label}</p>
			<div class="flex gap-2 mt-1.5 text-[10px] text-muted">
				<span>{focusedSource.nodeType}</span>
				<span>{focusedNeighbours.length} neighbours</span>
			</div>
		</div>
	{/if}
</div>

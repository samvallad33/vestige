<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import Graph3D from '$components/Graph3D.svelte';
	import RetentionCurve from '$components/RetentionCurve.svelte';
	import TimeSlider from '$components/TimeSlider.svelte';
	import MemoryStateLegend from '$components/MemoryStateLegend.svelte';
	import { api } from '$stores/api';
	import { eventFeed } from '$stores/websocket';
	import { graphState } from '$stores/graph-state.svelte';
	import type { GraphResponse, GraphNode, GraphEdge, Memory } from '$types';
	import type { GraphMutation } from '$lib/graph/events';
	import type { ColorMode } from '$lib/graph/nodes';
	import { filterByDate } from '$lib/graph/temporal';

	let graphData: GraphResponse | null = $state(null);
	let selectedMemory: Memory | null = $state(null);
	let loading = $state(true);
	let error = $state('');
	let isDreaming = $state(false);
	let searchQuery = $state('');
	let maxNodes = $state(150);
	let temporalEnabled = $state(false);
	let temporalDate = $state(new Date());
	// v2.0.8: colour spheres by node type (default) or by FSRS memory state
	// (Active / Dormant / Silent / Unavailable). Legend overlay renders when
	// state mode is active.
	let colorMode: ColorMode = $state('type');

	// Live counts that update on mutations
	let liveNodeCount = $state(0);
	let liveEdgeCount = $state(0);

	// Filtered graph data based on temporal mode
	let displayNodes = $derived.by((): GraphNode[] => {
		if (!graphData) return [];
		if (!temporalEnabled) return graphData.nodes;
		return filterByDate(graphData.nodes, graphData.edges, temporalDate).visibleNodes;
	});

	let displayEdges = $derived.by((): GraphEdge[] => {
		if (!graphData) return [];
		if (!temporalEnabled) return graphData.edges;
		return filterByDate(graphData.nodes, graphData.edges, temporalDate).visibleEdges;
	});

	function handleGraphMutation(mutation: GraphMutation) {
		if (!graphData) return;

		switch (mutation.type) {
			case 'nodeAdded':
				graphData.nodes = [...graphData.nodes, mutation.node];
				graphData.nodeCount = graphData.nodes.length;
				liveNodeCount = graphData.nodeCount;
				break;
			case 'nodeRemoved':
				graphData.nodes = graphData.nodes.filter((n) => n.id !== mutation.nodeId);
				graphData.nodeCount = graphData.nodes.length;
				liveNodeCount = graphData.nodeCount;
				break;
			case 'edgeAdded':
				graphData.edges = [...graphData.edges, mutation.edge];
				graphData.edgeCount = graphData.edges.length;
				liveEdgeCount = graphData.edgeCount;
				break;
			case 'edgesRemoved':
				graphData.edges = graphData.edges.filter(
					(e) => e.source !== mutation.nodeId && e.target !== mutation.nodeId
				);
				graphData.edgeCount = graphData.edges.length;
				liveEdgeCount = graphData.edgeCount;
				break;
			case 'nodeUpdated': {
				const node = graphData.nodes.find((n) => n.id === mutation.nodeId);
				if (node) node.retention = mutation.retention;
				break;
			}
		}
	}

	onMount(() => loadGraph());

	async function loadGraph(query?: string, centerId?: string) {
		loading = true;
		error = '';
		try {
			const isDefault = !query && !centerId;
			graphData = await api.graph({
				max_nodes: maxNodes,
				depth: 3,
				query: query || undefined,
				center_id: centerId || undefined,
				// Center on the newest memory by default. Prevents the old
				// "most-connected" behaviour from clustering on historical
				// hotspots and hiding today's memories behind the 150-node
				// cap. Future UI toggle can flip this to 'connected'.
				sort: isDefault ? 'recent' : undefined
			});

			// Fallback: if the newest memory is isolated (1 node, 0 edges),
			// fall back to the connected hotspot so the user sees context
			// instead of a lonely orb. Only applies to the default load —
			// explicit queries/centerId honor the user's choice even if the
			// subgraph is sparse.
			if (
				isDefault &&
				graphData &&
				graphData.nodeCount <= 1 &&
				graphData.edgeCount === 0
			) {
				const connected = await api.graph({
					max_nodes: maxNodes,
					depth: 3,
					sort: 'connected'
				});
				if (connected && connected.nodeCount > graphData.nodeCount) {
					graphData = connected;
				}
			}

			if (graphData) {
				liveNodeCount = graphData.nodeCount;
				liveEdgeCount = graphData.edgeCount;
			}
		} catch (e) {
			// Distinguish three failure modes so the error message is actually
			// helpful. Before: all failures (backend offline, empty DB, real
			// 500) rendered identical cryptic text. That made the dashboard
			// look broken on first-run or on backend-down, when the root
			// cause is ALWAYS "the MCP server isn't running."
			//   (1) Backend offline — vite dev proxy returns 500 with no body
			//       (upstream EHOSTUNREACH / ECONNREFUSED). Surface clearly:
			//       tell the user to start vestige-mcp.
			//   (2) Empty database — fresh install, no memories yet. Happy
			//       first-run state, not an error.
			//   (3) Real backend error — a genuine 500 with a response body,
			//       or a 4xx with content. Show the sanitized upstream msg.
			const rawMsg = e instanceof Error ? e.message : String(e);
			const safeMsg = rawMsg
				.replace(/\/[\w./-]+\.(sqlite|rs|db|toml|lock)\b/g, '[path]')
				.slice(0, 200);

			// Network-level failure: fetch itself rejects (TypeError) OR vite
			// proxy passes back a body-less 500 when upstream :3927 is
			// unreachable. Both mean "backend offline."
			const isBackendOffline =
				e instanceof TypeError ||
				/failed to fetch|NetworkError|load failed/i.test(rawMsg) ||
				/^API 500:?\s*(Internal Server Error)?\s*$/i.test(rawMsg.trim());

			const isEmpty =
				(graphData?.nodeCount ?? 0) === 0 &&
				/not found|404|empty|no memor/i.test(rawMsg);

			if (isBackendOffline) {
				error = 'OFFLINE';
			} else if (isEmpty) {
				error = 'EMPTY';
			} else {
				error = `Failed to load graph: ${safeMsg}`;
			}
		} finally {
			loading = false;
		}
	}

	async function triggerDream() {
		isDreaming = true;
		try {
			await api.dream();
			await loadGraph();
		} catch { /* dream failed */ }
		finally { isDreaming = false; }
	}

	async function onNodeSelect(nodeId: string) {
		try {
			selectedMemory = await api.memories.get(nodeId);
		} catch {
			selectedMemory = null;
		}
	}

	function searchGraph() {
		if (searchQuery.trim()) loadGraph(searchQuery);
	}
</script>

<div class="h-full relative">
	{#if loading}
		<div class="h-full flex items-center justify-center">
			<div class="text-center space-y-4">
				<div class="w-16 h-16 mx-auto rounded-full border-2 border-synapse/30 border-t-synapse animate-spin"></div>
				<p class="text-dim text-sm">Loading memory graph...</p>
			</div>
		</div>
	{:else if error === 'OFFLINE'}
		<div class="h-full flex items-center justify-center">
			<div class="text-center space-y-5 max-w-lg px-8">
				<div class="text-5xl opacity-40">⚡</div>
				<h2 class="text-xl text-bright">MCP Backend Offline</h2>
				<p class="text-dim text-sm leading-relaxed">
					The Vestige MCP server isn't reachable on <code class="font-mono text-muted">:3927</code>.
					The dashboard is running but has nothing to query.
				</p>
				<div class="glass-subtle rounded-xl p-4 text-left text-xs font-mono text-dim space-y-2">
					<div class="text-muted text-[10px] uppercase tracking-wider">Start the backend:</div>
					<code class="block whitespace-pre-wrap break-all text-text">nohup bash -c 'tail -f /dev/null | VESTIGE_DASHBOARD_ENABLED=true ~/.local/bin/vestige-mcp' &gt; /tmp/vestige.log 2&gt;&amp;1 &amp;
disown</code>
				</div>
				<div class="flex gap-2 justify-center">
					<button onclick={() => loadGraph()}
						class="px-4 py-2 bg-synapse/20 border border-synapse/40 text-synapse-glow text-xs rounded-xl hover:bg-synapse/30 transition">
						Retry
					</button>
					<a href="{base}/settings"
						class="px-4 py-2 bg-dream/20 border border-dream/40 text-dream-glow text-xs rounded-xl hover:bg-dream/30 transition">
						Try demos (no backend needed)
					</a>
				</div>
			</div>
		</div>
	{:else if error === 'EMPTY'}
		<div class="h-full flex items-center justify-center">
			<div class="text-center space-y-4 max-w-md px-8">
				<div class="text-5xl opacity-30">◎</div>
				<h2 class="text-xl text-bright">Your Mind Awaits</h2>
				<p class="text-dim text-sm">No memories yet. Start using Vestige to populate your graph.</p>
			</div>
		</div>
	{:else if error}
		<div class="h-full flex items-center justify-center">
			<div class="text-center space-y-4 max-w-md px-8">
				<div class="text-5xl opacity-30">◎</div>
				<h2 class="text-xl text-bright">Your Mind Awaits</h2>
				<p class="text-dim text-sm">{error}</p>
			</div>
		</div>
	{:else if graphData}
		<Graph3D
			nodes={displayNodes}
			edges={displayEdges}
			centerId={graphData.center_id}
			events={$eventFeed}
			{isDreaming}
			{colorMode}
			onSelect={onNodeSelect}
			onGraphMutation={handleGraphMutation}
		/>
	{/if}

	<!-- Top controls bar -->
	<div class="absolute top-4 left-4 right-4 z-10 flex items-center gap-3">
		<!-- Search -->
		<div class="flex gap-2 flex-1 max-w-md">
			<input
				type="text"
				placeholder="Center graph on..."
				bind:value={searchQuery}
				onkeydown={(e) => e.key === 'Enter' && searchGraph()}
				class="flex-1 px-3 py-2 glass rounded-xl text-text text-sm
					placeholder:text-muted focus:outline-none focus:!border-synapse/40 transition"
			/>
			<button onclick={searchGraph}
				class="px-3 py-2 bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm rounded-xl hover:bg-synapse/30 transition backdrop-blur-sm">
				Focus
			</button>
		</div>

		<div class="flex gap-2 ml-auto">
			<!-- v2.0.8: colour mode toggle. Switches sphere tint between node type
				 (fact / concept / event / …) and FSRS memory state (active / dormant /
				 silent / unavailable). Legend auto-renders in state mode. -->
			<div class="flex glass rounded-xl p-0.5 text-xs" role="radiogroup" aria-label="Colour mode">
				<button
					type="button"
					role="radio"
					aria-checked={colorMode === 'type'}
					onclick={() => (colorMode = 'type')}
					class="px-3 py-1.5 rounded-lg transition {colorMode === 'type' ? 'bg-synapse/25 text-synapse-glow' : 'text-dim hover:text-text'}"
					title="Colour by node type (fact, concept, event, …)"
				>
					Type
				</button>
				<button
					type="button"
					role="radio"
					aria-checked={colorMode === 'state'}
					onclick={() => (colorMode = 'state')}
					class="px-3 py-1.5 rounded-lg transition {colorMode === 'state' ? 'bg-synapse/25 text-synapse-glow' : 'text-dim hover:text-text'}"
					title="Colour by FSRS memory state (active / dormant / silent / unavailable)"
				>
					State
				</button>
			</div>

			<!-- Node count -->
			<select bind:value={maxNodes} onchange={() => loadGraph()}
				class="px-2 py-2 glass rounded-xl text-dim text-xs">
				<option value={50}>50 nodes</option>
				<option value={100}>100 nodes</option>
				<option value={150}>150 nodes</option>
				<option value={200}>200 nodes</option>
			</select>

			<!-- Brightness slider (persists in localStorage). Scales node emissive,
				 glow, and distance-compensated fog falloff. Default 1.0, range 0.5-2.5. -->
			<label
				class="flex items-center gap-2 px-3 py-2 glass rounded-xl text-dim text-xs select-none"
				title="Adjust graph brightness ({graphState.brightness.toFixed(1)}x). Combines with auto distance compensation."
			>
				<span class="text-synapse-glow">☀</span>
				<input
					type="range"
					min={graphState.brightnessMin}
					max={graphState.brightnessMax}
					step="0.1"
					bind:value={graphState.brightness}
					class="w-20 accent-synapse cursor-pointer"
					aria-label="Graph brightness"
				/>
				<span class="font-mono text-[10px] text-muted w-8 text-right">
					{graphState.brightness.toFixed(1)}x
				</span>
			</label>

			<!-- Dream button -->
			<button
				onclick={triggerDream}
				disabled={isDreaming}
				class="px-4 py-2 rounded-xl bg-dream/20 border border-dream/40 text-dream-glow text-sm
					hover:bg-dream/30 transition-all backdrop-blur-sm disabled:opacity-50
					{isDreaming ? 'glow-dream animate-pulse-glow' : ''}"
			>
				{isDreaming ? '◈ Dreaming...' : '◈ Dream'}
			</button>

			<!-- Reload -->
			<button onclick={() => loadGraph()}
				class="px-3 py-2 glass rounded-xl text-dim text-sm hover:text-text transition">
				↻
			</button>
		</div>
	</div>

	<!-- Bottom stats -->
	<div class="absolute bottom-4 left-4 z-10 text-xs text-dim glass rounded-xl px-3 py-2">
		{#if graphData}
			<span>{liveNodeCount} nodes</span>
			<span class="mx-2 text-subtle">·</span>
			<span>{liveEdgeCount} edges</span>
			<span class="mx-2 text-subtle">·</span>
			<span>depth {graphData.depth}</span>
		{/if}
	</div>

	<!-- v2.0.8: FSRS memory-state legend. Only rendered in state mode so the
		 legend doesn't compete with the node-type palette in type mode. -->
	{#if colorMode === 'state'}
		<div class="absolute bottom-4 right-4 z-10">
			<MemoryStateLegend />
		</div>
	{/if}

	<!-- Temporal playback slider -->
	{#if graphData}
		<TimeSlider
			nodes={graphData.nodes}
			onDateChange={(date) => { temporalDate = date; }}
			onToggle={(enabled) => { temporalEnabled = enabled; }}
		/>
	{/if}

	<!-- Selected memory panel -->
	{#if selectedMemory}
		<div class="absolute right-0 top-0 h-full w-96 glass-panel p-6 overflow-y-auto z-20
			transition-transform duration-300">
			<div class="flex justify-between items-start mb-4">
				<h3 class="text-bright text-sm font-semibold">Memory Detail</h3>
				<button onclick={() => selectedMemory = null} class="text-dim hover:text-text text-lg leading-none">×</button>
			</div>

			<div class="space-y-4">
				<div class="flex gap-2 flex-wrap">
					<span class="px-2 py-0.5 rounded-lg text-xs bg-synapse/20 text-synapse-glow">{selectedMemory.nodeType}</span>
					{#each selectedMemory.tags as tag}
						<span class="px-2 py-0.5 rounded-lg text-xs bg-white/[0.04] text-dim">{tag}</span>
					{/each}
				</div>

				<div class="text-sm text-text leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">{selectedMemory.content}</div>

				<!-- FSRS bars -->
				<div class="space-y-2">
					{#each [
						{ label: 'Retention', value: selectedMemory.retentionStrength },
						{ label: 'Storage', value: selectedMemory.storageStrength },
						{ label: 'Retrieval', value: selectedMemory.retrievalStrength }
					] as bar}
						<div>
							<div class="flex justify-between text-xs text-dim mb-0.5">
								<span>{bar.label}</span>
								<span>{(bar.value * 100).toFixed(1)}%</span>
							</div>
							<div class="h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
								<div
									class="h-full rounded-full transition-all duration-500"
									style="width: {bar.value * 100}%; background: {
										bar.value > 0.7 ? '#10b981' :
										bar.value > 0.4 ? '#f59e0b' : '#ef4444'
									}"
								></div>
							</div>
						</div>
					{/each}
				</div>

				<!-- FSRS Decay Curve -->
				<div>
					<div class="text-xs text-dim mb-1 font-medium">Retention Forecast</div>
					<RetentionCurve
						retention={selectedMemory.retentionStrength}
						stability={selectedMemory.storageStrength * 30}
					/>
				</div>

				<div class="text-xs text-muted space-y-1">
					<div>Created: {new Date(selectedMemory.createdAt).toLocaleString()}</div>
					<div>Updated: {new Date(selectedMemory.updatedAt).toLocaleString()}</div>
					{#if selectedMemory.lastAccessedAt}
						<div>Accessed: {new Date(selectedMemory.lastAccessedAt).toLocaleString()}</div>
					{/if}
					<div>Reviews: {selectedMemory.reviewCount ?? 0}</div>
				</div>

				<div class="flex gap-2 pt-2">
					<button
						onclick={() => { if (selectedMemory) { api.memories.promote(selectedMemory.id); } }}
						class="flex-1 px-3 py-2 rounded-xl bg-recall/20 text-recall text-xs hover:bg-recall/30 transition"
					>
						↑ Promote
					</button>
					<button
						onclick={() => { if (selectedMemory) { api.memories.demote(selectedMemory.id); } }}
						class="flex-1 px-3 py-2 rounded-xl bg-decay/20 text-decay text-xs hover:bg-decay/30 transition"
					>
						↓ Demote
					</button>
				</div>

				<!-- Explore from this node -->
				<a
					href="{base}/explore"
					class="block text-center px-3 py-2 rounded-xl bg-dream/10 text-dream-glow text-xs hover:bg-dream/20 transition border border-dream/20"
				>
					◬ Explore Connections
				</a>
			</div>
		</div>
	{/if}
</div>

<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import { websocket, isConnected, memoryCount, avgRetention } from '$stores/websocket';
	import { fireDemoSequence } from '$stores/toast';

	// v2.3 Birth Ritual demo — injects a synthetic MemoryCreated event so
	// Graph3D spawns a birth orb without needing a real ingest. Node types
	// cycle so back-to-back clicks show different colors. Pure dev/demo
	// affordance; production users see orbs fire on real ingests.
	const DEMO_NODE_TYPES = ['fact', 'concept', 'pattern', 'decision', 'person', 'place'];
	let birthCount = $state(0);
	function fireBirthRitualDemo() {
		const type = DEMO_NODE_TYPES[birthCount % DEMO_NODE_TYPES.length];
		birthCount++;
		websocket.injectEvent({
			type: 'MemoryCreated',
			data: {
				id: `demo-birth-${Date.now()}`,
				content: `Demo memory #${birthCount} — ${type}`,
				node_type: type,
				tags: ['demo', 'v2.3-birth-ritual'],
				retention: 0.9,
			},
		});
	}

	// Operation states
	let consolidating = $state(false);
	let dreaming = $state(false);
	let consolidationResult = $state<Record<string, unknown> | null>(null);
	let dreamResult = $state<Record<string, unknown> | null>(null);

	// Stats
	let stats = $state<Record<string, unknown> | null>(null);
	let retentionDist = $state<Record<string, unknown> | null>(null);
	let loadingStats = $state(true);

	// Health
	let health = $state<Record<string, unknown> | null>(null);

	onMount(() => {
		loadAllData();
	});

	async function loadAllData() {
		loadingStats = true;
		try {
			const [s, h, r] = await Promise.all([
				api.stats().catch(() => null),
				api.health().catch(() => null),
				api.retentionDistribution().catch(() => null),
			]);
			stats = s as Record<string, unknown> | null;
			health = h as Record<string, unknown> | null;
			retentionDist = r as Record<string, unknown> | null;
		} finally {
			loadingStats = false;
		}
	}

	async function runConsolidation() {
		consolidating = true;
		consolidationResult = null;
		try {
			consolidationResult = await api.consolidate() as unknown as Record<string, unknown>;
			await loadAllData();
		} catch { /* ignore */ }
		finally { consolidating = false; }
	}

	async function runDream() {
		dreaming = true;
		dreamResult = null;
		try {
			dreamResult = await api.dream() as unknown as Record<string, unknown>;
			await loadAllData();
		} catch { /* ignore */ }
		finally { dreaming = false; }
	}
</script>

<div class="p-6 max-w-4xl mx-auto space-y-8">
	<div class="flex items-center justify-between">
		<h1 class="text-xl text-bright font-semibold">Settings & System</h1>
		<button onclick={loadAllData} class="text-xs text-dim hover:text-text transition">Refresh</button>
	</div>

	<!-- System Health Overview -->
	<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
		<div class="p-4 glass rounded-xl text-center">
			<div class="text-2xl text-bright font-bold">{$memoryCount}</div>
			<div class="text-xs text-dim mt-1">Memories</div>
		</div>
		<div class="p-4 glass rounded-xl text-center">
			<div class="text-2xl font-bold" style="color: {$avgRetention > 0.7 ? '#10b981' : $avgRetention > 0.4 ? '#f59e0b' : '#ef4444'}">{($avgRetention * 100).toFixed(1)}%</div>
			<div class="text-xs text-dim mt-1">Avg Retention</div>
		</div>
		<div class="p-4 glass rounded-xl text-center">
			<div class="text-2xl text-bright font-bold flex items-center justify-center gap-2">
				<div class="w-2.5 h-2.5 rounded-full {$isConnected ? 'bg-recall animate-pulse-glow' : 'bg-decay'}"></div>
				<span class="text-sm">{$isConnected ? 'Online' : 'Offline'}</span>
			</div>
			<div class="text-xs text-dim mt-1">WebSocket</div>
		</div>
		<div class="p-4 glass rounded-xl text-center">
			<div class="text-2xl text-synapse-glow font-bold">v2.1</div>
			<div class="text-xs text-dim mt-1">Vestige</div>
		</div>
	</div>

	<!-- Cognitive Operations -->
	<section class="space-y-4">
		<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
			<span class="text-dream">◈</span> Cognitive Operations
		</h2>

		<!-- v2.2 Pulse — demo the InsightToast stream -->
		<div class="p-4 glass rounded-xl space-y-3">
			<div class="flex items-center justify-between">
				<div>
					<div class="text-sm text-text font-medium">Pulse Toast Preview</div>
					<div class="text-xs text-dim">Fire a synthetic event sequence — useful for UI demos</div>
				</div>
				<button onclick={fireDemoSequence}
					class="px-4 py-2 bg-synapse/20 border border-synapse/40 text-synapse-glow text-sm rounded-xl hover:bg-synapse/30 transition flex items-center gap-2">
					<span>✦</span> Preview Pulse
				</button>
			</div>
		</div>

		<!-- v2.3 Terrarium — demo the Memory Birth Ritual on the Graph page -->
		<div class="p-4 glass rounded-xl space-y-3">
			<div class="flex items-center justify-between">
				<div>
					<div class="text-sm text-text font-medium">Birth Ritual Preview</div>
					<div class="text-xs text-dim">Inject a synthetic memory — switch to Graph to watch the orb fly in</div>
				</div>
				<button onclick={fireBirthRitualDemo}
					class="px-4 py-2 bg-dream/20 border border-dream/40 text-dream-glow text-sm rounded-xl hover:bg-dream/30 transition flex items-center gap-2">
					<span>✺</span> Trigger Birth
				</button>
			</div>
		</div>

		<!-- Consolidation -->
		<div class="p-4 glass rounded-xl space-y-3">
			<div class="flex items-center justify-between">
				<div>
					<div class="text-sm text-text font-medium">FSRS-6 Consolidation</div>
					<div class="text-xs text-dim">Apply spaced-repetition decay, regenerate embeddings, run maintenance</div>
				</div>
				<button onclick={runConsolidation} disabled={consolidating}
					class="px-4 py-2 bg-warning/20 border border-warning/40 text-warning text-sm rounded-xl hover:bg-warning/30 transition disabled:opacity-50 flex items-center gap-2">
					{#if consolidating}
						<span class="w-3 h-3 border border-warning/50 border-t-warning rounded-full animate-spin"></span>
						Running...
					{:else}
						Consolidate
					{/if}
				</button>
			</div>
			{#if consolidationResult}
				<div class="bg-white/[0.02] p-3 rounded-lg border border-synapse/10">
					<div class="grid grid-cols-3 gap-3 text-center">
						{#if consolidationResult.nodesProcessed !== undefined}
							<div>
								<div class="text-lg text-text font-semibold">{consolidationResult.nodesProcessed}</div>
								<div class="text-[10px] text-muted">Processed</div>
							</div>
						{/if}
						{#if consolidationResult.decayApplied !== undefined}
							<div>
								<div class="text-lg text-decay font-semibold">{consolidationResult.decayApplied}</div>
								<div class="text-[10px] text-muted">Decayed</div>
							</div>
						{/if}
						{#if consolidationResult.embeddingsGenerated !== undefined}
							<div>
								<div class="text-lg text-synapse-glow font-semibold">{consolidationResult.embeddingsGenerated}</div>
								<div class="text-[10px] text-muted">Embedded</div>
							</div>
						{/if}
					</div>
				</div>
			{/if}
		</div>

		<!-- Dream -->
		<div class="p-4 glass rounded-xl space-y-3">
			<div class="flex items-center justify-between">
				<div>
					<div class="text-sm text-text font-medium">Memory Dream Cycle</div>
					<div class="text-xs text-dim">Replay memories, discover hidden connections, synthesize insights</div>
				</div>
				<button onclick={runDream} disabled={dreaming}
					class="px-4 py-2 bg-dream/20 border border-dream/40 text-dream-glow text-sm rounded-xl hover:bg-dream/30 transition disabled:opacity-50 flex items-center gap-2
						{dreaming ? 'glow-dream animate-pulse-glow' : ''}">
					{#if dreaming}
						<span class="w-3 h-3 border border-dream/50 border-t-dream rounded-full animate-spin"></span>
						Dreaming...
					{:else}
						Dream
					{/if}
				</button>
			</div>
			{#if dreamResult}
				<div class="bg-white/[0.02] p-3 rounded-lg border border-synapse/10 space-y-2">
					{#if dreamResult.insights && Array.isArray(dreamResult.insights)}
						<div class="text-xs text-bright font-medium">Insights Discovered:</div>
						{#each dreamResult.insights as insight}
							<div class="text-xs text-dim bg-dream/5 border border-dream/10 rounded-lg p-2">
								{typeof insight === 'string' ? insight : JSON.stringify(insight)}
							</div>
						{/each}
					{/if}
					{#if dreamResult.connections_found !== undefined}
						<div class="text-xs text-dim">Connections found: <span class="text-dream-glow">{dreamResult.connections_found}</span></div>
					{/if}
					{#if dreamResult.memories_replayed !== undefined}
						<div class="text-xs text-dim">Memories replayed: <span class="text-text">{dreamResult.memories_replayed}</span></div>
					{/if}
				</div>
			{/if}
		</div>
	</section>

	<!-- Retention Distribution -->
	{#if retentionDist}
		<section class="space-y-4">
			<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
				<span class="text-recall">◫</span> Retention Distribution
			</h2>
			<div class="p-4 glass rounded-xl">
				{#if retentionDist.distribution && Array.isArray(retentionDist.distribution)}
					<div class="flex items-end gap-1 h-32">
						{#each retentionDist.distribution as bucket, i}
							{@const maxCount = Math.max(...(retentionDist.distribution as {count: number}[]).map((b: {count: number}) => b.count), 1)}
							{@const height = ((bucket as {count: number}).count / maxCount) * 100}
							{@const color = i < 2 ? '#ef4444' : i < 4 ? '#f59e0b' : i < 7 ? '#6366f1' : '#10b981'}
							<div class="flex-1 flex flex-col items-center gap-1">
								<div class="text-[9px] text-muted">{(bucket as {count: number}).count}</div>
								<div
									class="w-full rounded-t transition-all duration-500"
									style="height: {Math.max(height, 2)}%; background: {color}; opacity: 0.7"
								></div>
								<div class="text-[9px] text-muted">{i * 10}%</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</section>
	{/if}

	<!-- Keyboard Shortcuts -->
	<section class="space-y-4">
		<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
			<span class="text-synapse">⌨</span> Keyboard Shortcuts
		</h2>
		<div class="p-4 glass-subtle rounded-xl">
			<div class="grid grid-cols-2 gap-2 text-xs">
				{#each [
					{ key: '⌘ K', desc: 'Command palette' },
					{ key: '/', desc: 'Focus search' },
					{ key: 'G', desc: 'Go to Graph' },
					{ key: 'M', desc: 'Go to Memories' },
					{ key: 'T', desc: 'Go to Timeline' },
					{ key: 'F', desc: 'Go to Feed' },
					{ key: 'E', desc: 'Go to Explore' },
					{ key: 'S', desc: 'Go to Stats' },
				] as shortcut}
					<div class="flex items-center gap-2 py-1">
						<kbd class="px-1.5 py-0.5 bg-white/[0.04] rounded text-[10px] font-mono text-muted min-w-[2rem] text-center">{shortcut.key}</kbd>
						<span class="text-dim">{shortcut.desc}</span>
					</div>
				{/each}
			</div>
		</div>
	</section>

	<!-- About -->
	<section class="space-y-4">
		<h2 class="text-sm text-bright font-semibold flex items-center gap-2">
			<span class="text-memory">◎</span> About
		</h2>
		<div class="p-4 glass rounded-xl space-y-3">
			<div class="flex items-center gap-4">
				<div class="w-12 h-12 rounded-xl bg-gradient-to-br from-dream to-synapse flex items-center justify-center text-bright text-xl font-bold shadow-lg shadow-synapse/20">
					V
				</div>
				<div>
					<div class="text-sm text-bright font-semibold">Vestige v2.1 "Nuclear Dashboard"</div>
					<div class="text-xs text-dim">Your AI's long-term memory system</div>
				</div>
			</div>
			<div class="grid grid-cols-2 gap-2 text-xs text-dim pt-2 border-t border-synapse/10">
				<div>29 cognitive modules</div>
				<div>FSRS-6 spaced repetition</div>
				<div>Nomic Embed v1.5 (256d)</div>
				<div>Jina Reranker v1 Turbo</div>
				<div>USearch HNSW (20x FAISS)</div>
				<div>Local-first, zero cloud</div>
			</div>
			<div class="text-[10px] text-muted pt-1">
				Built with Rust + Axum + SvelteKit 2 + Svelte 5 + Three.js + Tailwind CSS 4
			</div>
		</div>
	</section>
</div>

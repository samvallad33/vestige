<script lang="ts">
	import { onMount } from 'svelte';
	import RouteStage, { type RouteFramePass } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, FIELD_HUE, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { api } from '$stores/api';
	import { isConnected } from '$stores/websocket';
	import type { ConsolidationResult, DreamResult, HealthCheck, RetentionDistribution, SystemStats } from '$types';

	let stats = $state<SystemStats | null>(null);
	let health = $state<HealthCheck | null>(null);
	let retention = $state<RetentionDistribution | null>(null);
	let consolidation = $state<ConsolidationResult | null>(null);
	let dream = $state<DreamResult | null>(null);
	let busy = $state<null | 'consolidate' | 'dream' | 'refresh'>(null);
	let statusLine = $state('Ready to maintain the local memory system.');
	let loading = $state(true);
	let error = $state<string | null>(null);
	let systemField: LivingFieldPass | null = null;

	const settingsScene = $derived.by<RouteSceneModel>(() => ({
		organ: 'settings', nodes: [], edges: [], events: [], receipts: [], alive: true,
		scalars: { memories: stats?.totalMemories ?? 0, retention: stats?.averageRetention ?? 0, coverage: stats?.embeddingCoverage ?? 0 }
	}));

	$effect(() => {
		void [retention, stats, busy];
		systemField?.setCells(buildFieldCells());
	});

	onMount(() => void loadData());

	function buildFieldCells() {
		const data: FieldDatum[] = [];
		for (const [index, bucket] of (retention?.distribution ?? []).entries()) {
			const score = (index + 0.5) / Math.max(1, retention?.distribution.length ?? 1);
			for (let cell = 0; cell < Math.min(60, Math.ceil(bucket.count / 8)); cell += 1) {
				data.push({
					id: `settings:${index}:${cell}`, score, energy: 0.35 + score * 0.6, metric2: score,
					hue: score > .66 ? FIELD_HUE.oxygen : score > .33 ? FIELD_HUE.healthy : FIELD_HUE.debt,
					scar: score < .2, kind: 'retention-tissue', payload: { range: bucket.range, count: bucket.count }
				});
			}
		}
		return layoutGalaxy(data, { maxRadius: .96, minCellR: .01, maxCellR: .04 });
	}

	function createSettingsPasses(engine: ObservatoryEngine, _scene: RouteSceneModel): RouteFramePass[] {
		const field = new LivingFieldPass(engine);
		systemField = field;
		field.setIntensity(.7);
		field.setReadingWell({ x: -.1, y: 0, hw: .68, hh: .78, floor: .1, soft: .18 });
		field.setCells(buildFieldCells());
		return [{
			compute: (encoder) => field.compute(encoder),
			render: (renderPass) => field.render(renderPass),
			dispose: () => { field.dispose(); if (systemField === field) systemField = null; }
		}];
	}

	async function loadData() {
		loading = true;
		error = null;
		try {
			const [nextStats, nextRetention, nextHealth] = await Promise.all([api.stats(), api.retentionDistribution(), api.health()]);
			stats = nextStats;
			retention = nextRetention;
			health = nextHealth;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to load system state';
			statusLine = 'System state could not be loaded.';
		} finally {
			loading = false;
		}
	}

	async function runRefresh() {
		busy = 'refresh';
		statusLine = 'Refreshing live system vitals…';
		try { await loadData(); if (!error) statusLine = 'Live system vitals refreshed.'; } finally { busy = null; }
	}

	async function runConsolidate() {
		busy = 'consolidate'; consolidation = null; dream = null;
		statusLine = 'Consolidating memory: recalculating retention and maintenance state…';
		try {
			consolidation = await api.consolidate();
			await loadData();
			statusLine = 'Consolidation complete. The receipt below is from this run.';
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Consolidation failed';
			statusLine = 'Consolidation did not complete.';
		} finally { busy = null; }
	}

	async function runDream() {
		busy = 'dream'; dream = null; consolidation = null;
		statusLine = 'Running a dream cycle: replaying memory and finding connections…';
		try {
			dream = await api.dream();
			await loadData();
			statusLine = 'Dream cycle complete. The insight below is from this run.';
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Dream cycle failed';
			statusLine = 'Dream cycle did not complete.';
		} finally { busy = null; }
	}

	function bandWidth(count: number) { return `${Math.max(4, Math.min(100, (count / Math.max(1, ...(retention?.distribution ?? []).map((bucket) => bucket.count))) * 100))}%`; }
</script>

<svelte:head><title>System Care · Vestige</title></svelte:head>

<RouteStage organ="settings" seed={`settings-field:${stats?.totalMemories ?? 0}:${retention?.total ?? 0}`} scene={settingsScene} passes={createSettingsPasses} loading={false} {error} />

<main class="system-shell">
	<header class="system-head">
		<div><p class="eyebrow">SYSTEM CARE</p><h1>Keep the memory system alive.</h1><p>These controls run real local maintenance. Their results are recorded below, never implied by animation.</p></div>
		<div class:online={$isConnected} class="connection"><span></span>{$isConnected ? 'Live local connection' : 'Connecting locally'}</div>
	</header>

	{#if loading}
		<div class="glass-panel system-state">Loading live system state…</div>
	{:else if error && !stats}
		<div class="glass-panel system-state error">{error}<button type="button" onclick={runRefresh}>Try again</button></div>
	{:else}
		<dl class="system-vitals" aria-label="Current system metrics">
			<div><dt>Local memories</dt><dd>{stats?.totalMemories ?? 0}</dd></div>
			<div><dt>Average retention</dt><dd>{Math.round((stats?.averageRetention ?? 0) * 100)}%</dd></div>
			<div><dt>Embedding coverage</dt><dd>{Math.round(stats?.embeddingCoverage ?? 0)}%</dd></div>
			<div><dt>Running version</dt><dd>v{health?.version ?? 'unknown'}</dd></div>
		</dl>

		<section class="system-grid">
			<figure class="glass-panel retention"><figcaption><span>RETENTION DISTRIBUTION</span><small>{retention?.total ?? 0} memories</small></figcaption>
				{#each retention?.distribution ?? [] as bucket}
					<div class="retention-row"><span>{bucket.range}</span><div><i style={`width:${bandWidth(bucket.count)}`}></i></div><strong>{bucket.count}</strong></div>
				{/each}
			</figure>

			<section class="glass-panel rituals" aria-label="Memory maintenance actions">
				<p class="eyebrow">MAINTENANCE RITUALS</p><h2>Run with intent.</h2>
				<button type="button" disabled={busy !== null} onclick={runConsolidate}><strong>{busy === 'consolidate' ? 'Consolidating…' : 'Consolidate memory'}</strong><span>Recalculate retention, decay, embeddings and duplicates.</span></button>
				<button type="button" disabled={busy !== null} onclick={runDream}><strong>{busy === 'dream' ? 'Dreaming…' : 'Run dream cycle'}</strong><span>Replay local memories and discover durable connections.</span></button>
				<button type="button" class="refresh" disabled={busy !== null} onclick={runRefresh}>{busy === 'refresh' ? 'Refreshing…' : 'Refresh live vitals'}</button>
			</section>
		</section>
	{/if}

	<section class="glass-panel operation-receipt" aria-live="polite"><p class="eyebrow">OPERATION STATUS</p><output>{statusLine}</output>
		{#if consolidation}<dl><div><dt>Processed</dt><dd>{consolidation.nodesProcessed}</dd></div><div><dt>Decayed</dt><dd>{consolidation.decayApplied}</dd></div><div><dt>Embeddings</dt><dd>{consolidation.embeddingsGenerated}</dd></div><div><dt>Merged</dt><dd>{consolidation.duplicatesMerged}</dd></div><div><dt>Duration</dt><dd>{consolidation.durationMs} ms</dd></div></dl>{/if}
		{#if dream}<dl><div><dt>Replayed</dt><dd>{dream.memoriesReplayed}</dd></div><div><dt>Connections</dt><dd>{dream.connectionsPersisted}</dd></div><div><dt>Insights</dt><dd>{dream.insights.length}</dd></div></dl>{#if dream.insights[0]}<blockquote>{dream.insights[0].insight}</blockquote>{/if}{/if}
	</section>
</main>

<style>
	.system-shell{position:relative;z-index:2;max-width:1180px;min-height:100%;margin:0 auto;padding:2rem clamp(1rem,3vw,2.5rem) 5rem;color:#eaf9f6;pointer-events:none}.system-head,.system-grid,.system-vitals,.operation-receipt dl{display:flex}.system-head{justify-content:space-between;align-items:flex-end;gap:2rem}.eyebrow{margin:0;color:#66e6d3;font:700 .68rem/1.2 ui-monospace,monospace;letter-spacing:.14em}.system-head h1{max-width:20ch;margin:.55rem 0;font-size:clamp(1.7rem,3.3vw,2.75rem);line-height:1.05;letter-spacing:-.045em}.system-head p:not(.eyebrow){max-width:62ch;color:#a9c4c0;line-height:1.5}.connection{flex:0 0 auto;color:#9ab9b3;font:.7rem ui-monospace,monospace}.connection span{display:inline-block;width:.5rem;height:.5rem;margin-right:.4rem;border-radius:50%;background:#e3b554}.connection.online span{background:#62e6d1;box-shadow:0 0 12px #62e6d1}.glass-panel{border:1px solid rgba(124,198,187,.2);border-radius:1rem;background:linear-gradient(135deg,rgba(7,24,27,.9),rgba(4,12,15,.84));backdrop-filter:blur(14px);box-shadow:0 18px 70px rgba(0,0,0,.24)}.system-vitals{gap:.65rem;justify-content:space-between;margin:1.25rem 0}.system-vitals div{min-width:0;flex:1;border-left:1px solid rgba(102,230,211,.3);padding-left:.8rem}.system-vitals dt{color:#8da9a5;font-size:.7rem}.system-vitals dd{margin:.25rem 0 0;color:#72e7d5;font-size:1.35rem}.system-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(300px,.85fr);gap:1rem;pointer-events:auto}.retention,.rituals,.operation-receipt,.system-state{padding:1.1rem}.retention figcaption{display:flex;justify-content:space-between;margin-bottom:1rem;color:#b4d4ce;font:700 .68rem ui-monospace,monospace;letter-spacing:.08em}.retention figcaption small{color:#66e6d3;font-size:.64rem}.retention-row{display:grid;grid-template-columns:4.5rem 1fr 2rem;align-items:center;gap:.55rem;margin:.55rem 0;color:#9bb9b4;font:.69rem ui-monospace,monospace}.retention-row>div{height:.45rem;border-radius:99px;background:rgba(123,202,190,.12);overflow:hidden}.retention-row i{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#38bea9,#9af06a);box-shadow:0 0 13px rgba(86,231,207,.4)}.retention-row strong{color:#d8f5ef;text-align:right}.rituals h2{margin:.65rem 0 1rem;font-size:1.25rem}.rituals button{display:block;width:100%;margin:.55rem 0;border:1px solid rgba(95,227,205,.28);border-radius:.7rem;background:rgba(4,23,25,.82);padding:.8rem;color:#e8fdf8;text-align:left;cursor:pointer}.rituals button:hover:not(:disabled){border-color:#65e5d1;background:rgba(0,216,188,.12)}.rituals button:disabled{cursor:wait;opacity:.6}.rituals button strong,.rituals button span{display:block}.rituals button strong{font-size:.82rem}.rituals button span{margin-top:.22rem;color:#91b3ad;font-size:.71rem;line-height:1.35}.rituals .refresh{color:#6ce4d1;font-weight:700;text-align:center}.operation-receipt{position:relative;margin-top:1rem;pointer-events:auto}.operation-receipt output{display:block;margin:.6rem 0;color:#d9f4ee;line-height:1.45}.operation-receipt dl{flex-wrap:wrap;gap:.75rem;margin:.85rem 0 0}.operation-receipt dl div{border-left:1px solid rgba(102,230,211,.25);padding-left:.6rem}.operation-receipt dt{color:#8ba9a4;font-size:.67rem}.operation-receipt dd{margin:.2rem 0 0;color:#70e6d4;font-size:.9rem}.operation-receipt blockquote{margin:.8rem 0 0;border-left:2px solid #69e2d0;padding-left:.75rem;color:#b6d5cf;font-size:.82rem;line-height:1.5}.system-state{pointer-events:auto;color:#a9c7c2}.system-state button{margin-left:.75rem;border:0;border-radius:.45rem;background:#3cc8b4;padding:.45rem .6rem;color:#05100f;cursor:pointer}.error{color:#ffa198}@media(max-width:760px){.system-head,.system-grid{display:grid;grid-template-columns:1fr}.system-vitals{display:grid;grid-template-columns:1fr 1fr}.system-head{gap:.5rem}.connection{margin-top:.25rem}.operation-receipt dl{display:grid;grid-template-columns:1fr 1fr}}
</style>

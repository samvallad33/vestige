<script lang="ts">
	import { onMount } from 'svelte';
	import { osGoto } from '$lib/os-nav';
	import { page } from '$app/stores';
	import Icon from '$components/Icon.svelte';
	// MEMORY CINEMA IS PROTECTED. The Witness route keeps its finished
	// flythrough intact as a separate, opt-in emotional surface.
	import MemoryCinema from '$components/MemoryCinema.svelte';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { WitnessVolumePass } from '$lib/observatory/witness/witness-volume-pass';
	import { buildWitnessScene, type WitnessShard } from '$lib/observatory/witness/witness-scene';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { api, type Receipt, type TraceDetail, type TraceRunSummary } from '$lib/stores/api';
	import type { GraphResponse, Memory } from '$types';

	let runs = $state<TraceRunSummary[]>([]);
	let selectedRunId = $state<string | null>(null);
	let detail = $state<TraceDetail | null>(null);
	let receipts = $state<Receipt[]>([]);
	let selectedReceiptId = $state<string | null>(null);
	let memoryById = $state<Map<string, Memory>>(new Map());
	let selectedShardId = $state<string | null>(null);
	let cinemaGraph = $state<GraphResponse | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let witnessVolume: WitnessVolumePass | null = null;
	let playhead = $state(1);

	const selectedReceipt = $derived(
		receipts.find((receipt) => receipt.receipt_id === selectedReceiptId) ?? receipts[0] ?? null
	);
	const isTraceEvidence = $derived(Boolean(selectedReceipt?.receipt_id.startsWith('trace:')));
	const witnessScene = $derived(buildWitnessScene(detail, selectedReceipt, memoryById));
	const selectedShard = $derived(
		witnessScene.shards.find((shard) => shard.id === selectedShardId) ?? witnessScene.shards[0] ?? null
	);

	function createWitnessPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		const volume = new WitnessVolumePass(engine, scene);
		witnessVolume = volume;
		return [volume];
	}

	function handleWitnessPick(pick: RoutePick) {
		if (pick.kind !== 'witness-shard') return;
		const shard = pick.payload as WitnessShard;
		selectedShardId = shard.id;
		witnessVolume?.setSelected(shard.id);
	}

	function evidenceIds(receipt: Receipt | null): string[] {
		if (!receipt) return [];
		return [
			...receipt.activation_path,
			...receipt.retrieved,
			...receipt.mutations.map((mutation) => mutation.id),
			...receipt.suppressed.map((suppression) => suppression.id)
		].filter((id, index, all) => Boolean(id) && all.indexOf(id) === index);
	}

	/**
	 * Some local tool runs have retrieval trace rows before a durable receipt is
	 * written. Show that exact evidence, but visibly label it UNSEALED rather
	 * than borrowing the authority of a receipt that does not exist.
	 */
	function traceEvidence(detail: TraceDetail): Receipt | null {
		const retrieved = detail.events.flatMap((event) => event.type === 'memory.retrieve' ? event.ids : []);
		const activationPath = detail.events.flatMap((event) => {
			if (event.type !== 'memory.retrieve') return [];
			return Object.entries(event.activation)
				.sort(([, left], [, right]) => right - left)
				.map(([id]) => id);
		});
		if (!retrieved.length && !activationPath.length) return null;
		const suppressed = detail.events.flatMap((event) =>
			event.type === 'memory.suppress' ? [{ id: event.id, reason: event.reason }] : []
		);
		const mutations = detail.events.flatMap((event) =>
			event.type === 'memory.write' ? [{ id: event.id, kind: 'write' }] : []
		);
		return {
			receipt_id: `trace:${detail.runId}`,
			retrieved: retrieved.filter((id, index, all) => all.indexOf(id) === index),
			suppressed,
			activation_path: activationPath.filter((id, index, all) => all.indexOf(id) === index),
			trust_floor: 0,
			decay_risk: 'medium',
			mutations
		};
	}

	async function hydrateEvidence(receipt: Receipt | null) {
		const missing = evidenceIds(receipt)
			.slice(0, 64)
			.filter((id) => !memoryById.has(id));
		if (!missing.length) return;
		const loaded = await Promise.all(
			missing.map(async (id) => {
				try {
					return await api.memories.get(id);
				} catch {
					// Receipts can outlive a purged memory. The receipt still remains
					// valid evidence, so show its ID rather than erase the shard.
					return null;
				}
			})
		);
		const next = new Map(memoryById);
		for (const memory of loaded) if (memory) next.set(memory.id, memory);
		memoryById = next;
	}

	async function selectRun(runId: string) {
		selectedRunId = runId;
		loading = true;
		error = null;
		try {
			const [nextDetail, receiptResult] = await Promise.all([
				api.traces.get(runId),
				api.receipts.listForRun(runId, 24)
			]);
			detail = nextDetail;
			const fallback = traceEvidence(nextDetail);
			receipts = receiptResult.receipts.length ? receiptResult.receipts : fallback ? [fallback] : [];
			const requestedReceipt = $page.url.searchParams.get('receipt');
			selectedReceiptId =
				receipts.find((receipt) => receipt.receipt_id === requestedReceipt)?.receipt_id ??
				receipts[0]?.receipt_id ??
				null;
			selectedShardId = null;
			playhead = 1;
			await hydrateEvidence(
				receipts.find((receipt) => receipt.receipt_id === selectedReceiptId) ?? null
			);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
			detail = null;
			receipts = [];
		} finally {
			loading = false;
		}
	}

	async function chooseReceipt(receipt: Receipt) {
		selectedReceiptId = receipt.receipt_id;
		selectedShardId = null;
		playhead = 1;
		await hydrateEvidence(receipt);
	}

	async function loadWitness() {
		loading = true;
		try {
			const response = await api.traces.list(48);
			runs = response.runs;
			const requestedRun = $page.url.searchParams.get('run');
			const run = response.runs.find((candidate) => candidate.runId === requestedRun) ?? response.runs[0];
			if (run) await selectRun(run.runId);
			else loading = false;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
			loading = false;
		}
	}

	async function loadCinema() {
		try {
			// This data is for the protected Cinema only. The Witness Loom never
			// renders these corpus edges, so it cannot recreate the old hairball.
			cinemaGraph = await api.graph({ max_nodes: 48, depth: 2, sort: 'connected' });
		} catch {
			cinemaGraph = null;
		}
	}

	function openBlackBox() {
		if (!selectedRunId) return;
		void osGoto('/blackbox', { run: selectedRunId });
	}

	function openMemory(id: string) {
		void osGoto('/memories', { memory: id });
	}

	$effect(() => {
		void selectedShardId;
		witnessVolume?.setSelected(selectedShardId);
	});

	onMount(() => {
		void loadWitness();
		void loadCinema();
	});
</script>

<RouteStage
	organ="witness"
	seed={`witness:${selectedRunId ?? 'armed'}:${selectedReceiptId ?? 'none'}`}
	scene={witnessScene}
	passes={createWitnessPasses}
	maxDpr={1.25}
	{loading}
	{error}
	emptyLabel="NO RECEIPT SELECTED - WITNESS CHAMBER ARMED"
	onpick={handleWitnessPick}
/>

<main class="witness-shell relative z-10 mx-auto min-h-full max-w-[1520px] px-4 py-5 sm:px-7 sm:py-7">
	<header class="witness-header">
		<div>
			<p class="eyebrow"><span></span> WITNESS / RECEIPT-BOUND MEMORY ACCOUNTABILITY</p>
			<h1>What shaped the decision?</h1>
			<p class="lede">A forensic loom of exactly what this agent run retrieved, changed, suppressed, and proved.</p>
		</div>
		<div class="header-actions">
			<button class="instrument-button" onclick={openBlackBox} disabled={!selectedRunId}>
				<Icon name="blackbox" size={15} /> Open Black Box
			</button>
			<!-- MEMORY CINEMA — PROTECTED. Kept intact and fed a bounded graph only
			     when a user deliberately launches the finished flythrough. -->
			{#if cinemaGraph}
				<MemoryCinema nodes={cinemaGraph.nodes} edges={cinemaGraph.edges} centerId={cinemaGraph.center_id} />
			{/if}
		</div>
	</header>

	<section class="run-strip" aria-label="Agent runs">
		<div class="strip-label">AGENT RUNS <span>{runs.length}</span></div>
		<div class="run-list">
			{#each runs as run (run.runId)}
				<button
					class:active={run.runId === selectedRunId}
					class="run-chip"
					onclick={() => selectRun(run.runId)}
					title={`Open run ${run.runId}`}
				>
					<span class="run-pulse"></span>
					<span>{run.firstTool ?? 'agent run'}</span>
					<small>{run.retrievedCount} recalled</small>
				</button>
			{/each}
			{#if !loading && runs.length === 0}
				<p class="empty-run">No Black Box run has been recorded locally yet.</p>
			{/if}
		</div>
	</section>

	{#if selectedReceipt}
		<section class="receipt-rail" aria-label="Receipts for selected run">
			<div class="strip-label">RECEIPTS <span>{receipts.length}</span></div>
			<div class="receipt-list">
				{#each receipts as receipt, index (receipt.receipt_id)}
					<button
						class:active={receipt.receipt_id === selectedReceiptId}
						class="receipt-chip"
						onclick={() => chooseReceipt(receipt)}
					>
						<span>#{String(index + 1).padStart(2, '0')}</span>
						<small>{receipt.retrieved.length} evidence</small>
						<i class:risk-high={receipt.decay_risk === 'high'} class:risk-medium={receipt.decay_risk === 'medium'}>{receipt.decay_risk}</i>
					</button>
				{/each}
			</div>
		</section>
	{/if}

	<section class="volume-stage" aria-label="Three dimensional witness volume">
		<div class="volume-caption">
			<p class="eyebrow"><span></span> WITNESS STRATA / RECEIPT-SCOPED PROOF VOLUME</p>
			<h2>One run. One proof boundary.</h2>
			<p>Trace time becomes depth. Activation pins its stratum. Each smoked-glass specimen is an exact memory that crossed this run&rsquo;s evidence boundary.</p>
			<div class="volume-key" aria-label="Witness volume legend">
				<span><i class="path"></i> verified path</span>
				<span><i class="retrieved"></i> retrieved</span>
				<span><i class="mutation"></i> mutation</span>
				<span><i class="suppressed"></i> suppressed</span>
			</div>
		</div>

		<div class="volume-controls" aria-label="Witness playback controls">
			<div class="control-label"><span>TRACE SLICER</span><b>{Math.round(playhead * Math.max(0, witnessScene.eventCount))} / {witnessScene.eventCount}</b></div>
			<input
				aria-label="Trace event position"
				type="range"
				min="0"
				max="1"
				step="0.01"
				bind:value={playhead}
				oninput={() => witnessVolume?.setPlayhead(playhead)}
			/>
			<button class="replay-button" onclick={() => witnessVolume?.replay()} disabled={!witnessScene.shards.length}>
				Replay evidence <span aria-hidden="true">↗</span>
			</button>
		</div>

		<aside class="inspector" aria-live="polite">
			<div class="panel-heading">
				<div>
					<p class="eyebrow"><span></span> SELECTED SPECIMEN</p>
					<h2>{selectedShard ? selectedShard.role : 'Select a shard'}</h2>
				</div>
				{#if selectedReceipt}
					<span class="trust">{Math.round(selectedReceipt.trust_floor * 100)}% trust</span>
				{/if}
			</div>

			{#if selectedShard}
				<div class="specimen" class:scarred={selectedShard.suppressed}>
					<div class="specimen-cap">MEMORY / {selectedShard.id.slice(0, 12)}</div>
					<p>{selectedShard.content || selectedShard.label}</p>
				</div>
				<dl class="readout">
					<div><dt>Activation</dt><dd>{Math.round(selectedShard.activation * 100)}%</dd></div>
					<div><dt>Retention</dt><dd>{Math.round(selectedShard.retention * 100)}%</dd></div>
					<div><dt>State</dt><dd>{selectedShard.suppressed ? 'suppressed' : selectedShard.mutated ? 'mutated' : 'witnessed'}</dd></div>
				</dl>
				<button class="memory-button" onclick={() => openMemory(selectedShard.id)}>
					Inspect memory <Icon name="chevron" size={14} />
				</button>
			{:else}
				<p class="inspector-empty">Choose any visible evidence wafer. The chamber never substitutes an unrelated corpus edge for a proof path.</p>
			{/if}

			{#if selectedReceipt}
				<div class="receipt-seal" class:unsealed={isTraceEvidence}>
					<span>{isTraceEvidence ? 'UNSEALED TRACE EVIDENCE' : 'RECEIPT SEALED'}</span>
					<code>{selectedReceipt.receipt_id}</code>
					<div>
						<b>{selectedReceipt.retrieved.length}</b> retrieved
						<b>{selectedReceipt.suppressed.length}</b> suppressed
						<b>{selectedReceipt.mutations.length}</b> mutations
					</div>
				</div>
			{/if}
		</aside>

		{#if !loading && !witnessScene.shards.length}
			<div class="volume-empty">
				<Icon name="blackbox" size={26} />
				<p>Choose a recorded run with retrieval evidence to unseal its 3D witness volume.</p>
			</div>
		{/if}
	</section>
</main>

<style>
	:global(body) { background: #070907; }
	/* The WebGPU field is the primary surface. The DOM shell must not create an
	   invisible full-page click shield over the evidence wafers; only its actual
	   controls opt back into pointer input below. */
	.witness-shell { color: #e5e2d8; pointer-events:none; }
	.witness-header,.run-strip,.receipt-rail,.volume-controls,.inspector { pointer-events:auto; }
	.witness-header { display:flex; align-items:flex-start; justify-content:space-between; gap:1.5rem; padding:1rem 0 1.35rem; border-bottom:1px solid rgba(95,175,138,.18); }
	.eyebrow { display:flex; align-items:center; gap:.5rem; margin:0; color:#5faf8a; font:700 .65rem var(--font-mono,ui-monospace,monospace); letter-spacing:.15em; }
	.eyebrow span { width:.42rem; height:.42rem; border-radius:50%; background:#5faf8a; }
	h1,h2 { margin:.45rem 0 0; color:#e5e2d8; font-family:var(--font-display,ui-sans-serif,sans-serif); font-weight:520; letter-spacing:-.035em; }
	h1 { font-size:clamp(2rem,4vw,3.45rem); } h2 { font-size:1.12rem; }
	.lede { max-width:44rem; margin:.55rem 0 0; color:#8d9792; font-size:.94rem; }
	.header-actions { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:.55rem; }
	.instrument-button,.memory-button { display:inline-flex; align-items:center; gap:.48rem; border:1px solid rgba(95,175,138,.46); border-radius:.38rem; padding:.66rem .78rem; background:rgba(11,15,13,.94); color:#c8d8cb; font:650 .72rem var(--font-mono,ui-monospace,monospace); letter-spacing:.04em; transition:border-color .16s,background .16s,transform .16s; }
	.instrument-button:hover:not(:disabled),.memory-button:hover { border-color:#c58a4a; background:rgba(27,28,20,.98); transform:translateY(-1px); }
	.instrument-button:disabled { cursor:not-allowed; opacity:.42; }
	.run-strip,.receipt-rail { display:grid; grid-template-columns:8.5rem minmax(0,1fr); gap:1rem; padding:.82rem 0; border-bottom:1px solid rgba(95,175,138,.12); }
	.strip-label { padding:.55rem 0; color:#6e7772; font:700 .61rem var(--font-mono,ui-monospace,monospace); letter-spacing:.14em; }.strip-label span { margin-left:.35rem; color:#e5e2d8; }
	.run-list,.receipt-list { display:flex; gap:.42rem; overflow-x:auto; padding-bottom:.15rem; }
	.run-chip,.receipt-chip { display:inline-flex; flex:0 0 auto; align-items:center; gap:.48rem; border:1px solid rgba(141,151,146,.24); border-radius:.3rem; padding:.48rem .6rem; background:rgba(10,13,12,.9); color:#aeb7b0; font:600 .67rem var(--font-mono,ui-monospace,monospace); transition:.16s; }
	.run-chip:hover,.receipt-chip:hover,.run-chip.active,.receipt-chip.active { border-color:rgba(95,175,138,.75); background:rgba(16,27,21,.98); color:#f0eee5; }
	.run-pulse { width:.36rem; height:.36rem; border-radius:50%; background:#5faf8a; }.run-chip small,.receipt-chip small { color:#6e7772; font-size:.58rem; }
	.receipt-chip i { color:#c58a4a; font-size:.56rem; font-style:normal; text-transform:uppercase; }.receipt-chip i.risk-medium { color:#c58a4a; }.receipt-chip i.risk-high { color:#ab5a51; }.empty-run { margin:.5rem 0; color:#7d8781; font-size:.8rem; }
	.volume-stage { position:relative; isolation:isolate; min-height:max(42rem,calc(100svh - 13.3rem)); padding:1.7rem 0 2.2rem; }
	/* The chamber earns depth from the WebGPU specimens, not a page-sized fog. */
	.volume-stage::before { position:absolute; z-index:-1; inset:-1rem -8vw 0; pointer-events:none; content:''; background:linear-gradient(90deg,rgba(7,9,7,.2),transparent 24%,transparent 76%,rgba(7,9,7,.24)); }
	.volume-caption { width:min(25rem,39vw); margin-left:clamp(4.4rem,6vw,6.8rem); padding:.95rem 0; pointer-events:none; }.volume-caption h2 { font-size:clamp(1.35rem,2.2vw,2rem); letter-spacing:-.045em; }
	.volume-caption > p:not(.eyebrow) { max-width:23rem; margin:.62rem 0 0; color:#8d9792; font-size:.8rem; line-height:1.58; }
	.volume-key { display:flex; flex-wrap:wrap; gap:.52rem .85rem; margin-top:1rem; color:#7c8680; font:600 .54rem var(--font-mono,ui-monospace,monospace); letter-spacing:.09em; text-transform:uppercase; }.volume-key span { display:inline-flex; align-items:center; gap:.28rem; }.volume-key i { display:inline-block; width:.34rem; height:.34rem; background:#e5e2d8; }.volume-key i.path { background:#5faf8a; }.volume-key i.retrieved { background:#e5e2d8; }.volume-key i.mutation { background:#c58a4a; }.volume-key i.suppressed { background:#ab5a51; }
	.volume-controls { position:absolute; z-index:2; bottom:2.4rem; left:clamp(4.4rem,6vw,6.8rem); width:min(28rem,46vw); padding:.76rem 0 .68rem; border-top:1px solid rgba(95,175,138,.24); border-bottom:1px solid rgba(95,175,138,.1); background:linear-gradient(90deg,rgba(9,12,10,.97),rgba(9,12,10,.58)); }
	.control-label { display:flex; justify-content:space-between; margin-bottom:.48rem; color:#738078; font:700 .56rem var(--font-mono,ui-monospace,monospace); letter-spacing:.13em; }.control-label b { color:#e5e2d8; }
	.volume-controls input { display:block; width:100%; height:3px; appearance:none; border-radius:999px; outline:none; background:linear-gradient(90deg,#5faf8a,rgba(197,138,74,.6),rgba(94,105,97,.42)); accent-color:#c58a4a; cursor:pointer; }.volume-controls input::-webkit-slider-thumb { width:12px; height:12px; appearance:none; border:1px solid #e5e2d8; border-radius:50%; background:#101411; }
	.replay-button { display:inline-flex; align-items:center; gap:.42rem; margin-top:.72rem; border:1px solid rgba(95,175,138,.5); border-radius:.3rem; padding:.47rem .62rem; background:rgba(12,20,15,.94); color:#c8d8cb; font:700 .61rem var(--font-mono,ui-monospace,monospace); letter-spacing:.06em; text-transform:uppercase; transition:background .16s,border-color .16s,transform .16s; }.replay-button:hover:not(:disabled) { border-color:#c58a4a; background:rgba(32,29,19,.98); transform:translateY(-1px); }.replay-button:disabled { cursor:not-allowed; opacity:.35; }
	.inspector { position:absolute; z-index:2; top:1.7rem; right:0; display:flex; flex-direction:column; width:min(19rem,28vw); min-height:0; padding:1rem; border:1px solid rgba(141,151,146,.28); background:linear-gradient(145deg,rgba(12,16,14,.98),rgba(7,9,8,.96)); }.inspector::before { position:absolute; top:.55rem; right:.7rem; content:'3D / RECEIPT BOUND'; color:rgba(95,175,138,.56); font:600 .47rem var(--font-mono,ui-monospace,monospace); letter-spacing:.13em; }
	.panel-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }.trust { border:1px solid rgba(197,138,74,.34); border-radius:999px; padding:.3rem .45rem; color:#c58a4a; font:700 .6rem var(--font-mono,ui-monospace,monospace); white-space:nowrap; }
	.specimen { margin-top:1.05rem; padding:.8rem; border:1px solid rgba(95,175,138,.28); background:linear-gradient(145deg,rgba(18,26,21,.96),rgba(9,11,10,.98)); }.specimen.scarred { border-color:rgba(171,90,81,.7); }.specimen-cap { color:#5faf8a; font:.57rem var(--font-mono,ui-monospace,monospace); letter-spacing:.1em; }.specimen p { display:-webkit-box; overflow:hidden; margin:.55rem 0 0; color:#dedbd1; font-size:.78rem; line-height:1.52; line-clamp:8; -webkit-box-orient:vertical; -webkit-line-clamp:8; }
	.readout { display:grid; grid-template-columns:repeat(3,1fr); gap:.45rem; margin:1rem 0; }.readout div { padding-top:.5rem; border-top:1px solid rgba(141,151,146,.18); }.readout dt { color:#6e7772; font:.55rem var(--font-mono,ui-monospace,monospace); text-transform:uppercase; }.readout dd { margin:.22rem 0 0; color:#c58a4a; font:700 .7rem var(--font-mono,ui-monospace,monospace); }.memory-button { width:100%; justify-content:center; }
	.inspector-empty { margin:1.5rem 0; color:#818a84; font-size:.83rem; line-height:1.6; }.receipt-seal { margin-top:1.3rem; padding-top:1rem; border-top:1px solid rgba(141,151,146,.18); color:#7f8983; font:.58rem var(--font-mono,ui-monospace,monospace); letter-spacing:.08em; }.receipt-seal > span { color:#5faf8a; }.receipt-seal code { display:block; overflow:hidden; margin:.45rem 0; color:#c5cbc4; text-overflow:ellipsis; white-space:nowrap; }.receipt-seal div { display:flex; flex-wrap:wrap; gap:.7rem; letter-spacing:0; }.receipt-seal b { color:#c58a4a; }.receipt-seal.unsealed { border-color:rgba(197,138,74,.38); }.receipt-seal.unsealed > span { color:#c58a4a; }
	.volume-empty { position:absolute; top:50%; left:50%; display:grid; width:min(20rem,80vw); place-items:center; gap:.8rem; transform:translate(-50%,-40%); color:#8d9792; text-align:center; font-size:.82rem; line-height:1.55; }.volume-empty p { margin:0; }
	@media (max-width:900px) { .witness-header { flex-direction:column; }.header-actions { justify-content:flex-start; }.run-strip,.receipt-rail { grid-template-columns:1fr; gap:.1rem; }.strip-label { padding-bottom:.1rem; }.volume-stage { min-height:46rem; }.volume-caption { width:min(27rem,72vw); margin-left:0; }.inspector { top:auto; right:0; bottom:2rem; width:min(23rem,46vw); min-height:24rem; }.volume-controls { bottom:2rem; left:0; width:min(25rem,45vw); } }
	@media (max-width:620px) { .witness-shell { padding:.75rem; }.witness-header { padding-top:.2rem; } h1 { font-size:2.15rem; }.volume-stage { min-height:49rem; padding-top:1rem; }.volume-caption { width:100%; }.inspector { right:0; bottom:1rem; width:100%; min-height:19.5rem; }.volume-controls { bottom:22rem; left:0; width:100%; }.volume-key { gap:.45rem .65rem; } }
</style>

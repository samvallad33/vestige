<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { Memory, MemoryAuditEvent, TimelineDay } from '$types';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { createTimelinePasses } from '$lib/observatory/timeline/timeline-pass';
	import {
	normalizeTimelineScene,
	type TimelineCell,
	type TimelineRing,
	type TimelineScene
	} from '$lib/observatory/timeline/timeline-scene';

	const RANGE_CYCLE = [7, 14, 30, 90, 365] as const;

	let timeline = $state<TimelineDay[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let days = $state<(typeof RANGE_CYCLE)[number]>(14);
	let selectedDate = $state<string | null>(null);
	let selectedMemoryId = $state<string | null>(null);
	let auditLoading = $state(false);
	let audits = $state<Record<string, MemoryAuditEvent[]>>({});

	onMount(() => void loadTimeline());

	async function loadTimeline() {
		loading = true;
		error = null;
		try {
			const response = await api.timeline(days, 500);
			timeline = response.timeline;
			if (selectedDate && !response.timeline.some((day) => day.date === selectedDate)) {
				selectedDate = null;
				selectedMemoryId = null;
			}
		} catch (cause) {
			timeline = [];
			error = cause instanceof Error ? cause.message : 'Failed to load timeline';
		} finally {
			loading = false;
		}
	}

	async function selectRange(next: (typeof RANGE_CYCLE)[number]) {
		if (next === days) return;
		days = next;
		selectedDate = null;
		selectedMemoryId = null;
		await loadTimeline();
	}

	async function fetchAudit(memoryId: string) {
		if (audits[memoryId]) return;
		auditLoading = true;
		try {
			const response = await api.memoryAudit(memoryId, 100);
			audits = { ...audits, [memoryId]: response.events };
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to load memory audit';
		} finally {
			auditLoading = false;
		}
	}

	function selectDay(date: string) {
		selectedDate = date;
		selectedMemoryId = null;
	}

	function selectMemory(memory: Memory, date: string) {
		selectedDate = date;
		selectedMemoryId = memory.id;
		void fetchAudit(memory.id);
	}

	const allMemories = $derived(timeline.flatMap((day) => day.memories));
	const totalMemories = $derived(timeline.reduce((sum, day) => sum + day.count, 0));
	const rewriteCount = $derived(allMemories.filter((memory) => memory.updatedAt !== memory.createdAt).length);
	const avgRetention = $derived(
		allMemories.length
			? allMemories.reduce((sum, memory) => sum + (memory.retentionStrength ?? 0), 0) / allMemories.length
			: 0
	);
	const selectedDay = $derived(timeline.find((day) => day.date === selectedDate) ?? null);
	const selectedMemory = $derived(allMemories.find((memory) => memory.id === selectedMemoryId) ?? null);
	const selectedAudit = $derived(selectedMemoryId ? (audits[selectedMemoryId] ?? []) : []);
	const timelineScene: TimelineScene = $derived(normalizeTimelineScene({ days: timeline, totalMemories, audits }));

	function createTimelineOrganPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		// Rings are the live data field. All text, selection and receipts are DOM so
		// the proof remains readable, keyboard reachable and screen-reader visible.
		return createTimelinePasses(engine, scene);
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind === 'timeline-cell') {
			const cell = pick.payload as TimelineCell;
			const memory = allMemories.find((item) => item.id === cell.memoryId);
			if (memory) selectMemory(memory, cell.day);
		} else if (pick.kind === 'timeline-ring') {
			const ring = pick.payload as TimelineRing;
			selectDay(ring.date);
		}
	}

	function formatTime(value: string | undefined) {
		return value ? new Date(value).toLocaleString() : 'Not recorded';
	}
</script>

<svelte:head><title>Memory Timeline · Vestige</title></svelte:head>

<RouteStage
	organ="timeline"
	seed={`timeline-growth-rings:${days}:${totalMemories}`}
	scene={timelineScene}
	passes={createTimelineOrganPasses}
	loading={false}
	{error}
	emptyLabel="NO MEMORY GROWTH RINGS IN THIS WINDOW"
	onpick={handleRoutePick}
/>

<main class="timeline-shell">
	<header class="timeline-head">
		<div>
			<p class="eyebrow">BITEMPORAL MEMORY HISTORY</p>
			<h1>Watch memory grow. Inspect every change.</h1>
			<p class="lede">The rings are real valid-time history. Choose a date or a memory to open its transaction-time receipt.</p>
		</div>
		<div class="range-control" aria-label="Timeline range">
			<span>TIME WINDOW</span>
			{#each RANGE_CYCLE as range}
				<button type="button" class:active={days === range} aria-pressed={days === range} onclick={() => selectRange(range)}>{range}D</button>
			{/each}
		</div>
	</header>

	<dl class="vitals" aria-label="Timeline metrics">
		<div><dt>Memories</dt><dd>{totalMemories}</dd></div>
		<div><dt>Rewritten</dt><dd>{rewriteCount}</dd></div>
		<div><dt>Calendar slices</dt><dd>{timeline.length}</dd></div>
		<div><dt>Average retention</dt><dd>{Math.round(avgRetention * 100)}%</dd></div>
	</dl>

	<section class="timeline-grid">
		<div class="glass-panel day-list">
			<div class="panel-label"><span>VALID-TIME SLICES</span><strong>{days} DAYS</strong></div>
			{#if loading}
				<p class="state-line">Weaving the live memory history…</p>
			{:else if error}
				<p class="state-line error">{error}</p>
			{:else if timeline.length === 0}
				<p class="state-line">No memory growth in this window.</p>
			{:else}
				<div class="day-rows">
					{#each timeline as day (day.date)}
						<button type="button" class:active={selectedDate === day.date} onclick={() => selectDay(day.date)}>
							<span>{day.date}</span><strong>{day.count}</strong><small>{Math.round((day.memories.reduce((sum, memory) => sum + memory.retentionStrength, 0) / Math.max(1, day.memories.length)) * 100)}% retained</small>
						</button>
					{/each}
				</div>
			{/if}
		</div>

		<aside class="glass-panel receipt" aria-live="polite">
			{#if selectedMemory}
				<p class="eyebrow">TIME-SLICE RECEIPT</p>
				<h2>{selectedMemory.content}</h2>
				<dl class="receipt-metrics">
					<div><dt>Memory ID</dt><dd><code>{selectedMemory.id}</code></dd></div>
					<div><dt>Valid time</dt><dd>{formatTime(selectedMemory.validFrom ?? selectedMemory.createdAt)}</dd></div>
					<div><dt>Transaction time</dt><dd>{formatTime(selectedMemory.updatedAt)}</dd></div>
					<div><dt>Retention</dt><dd>{Math.round(selectedMemory.retentionStrength * 100)}%</dd></div>
				</dl>
				<h3>Audit events</h3>
				{#if auditLoading}<p class="state-line">Loading this memory’s audit…</p>
				{:else if selectedAudit.length === 0}<p class="state-line">No audit events returned for this record.</p>
				{:else}<ol>{#each selectedAudit.slice(0, 8) as event}<li><strong>{event.action}</strong><span>{formatTime(event.timestamp)}</span></li>{/each}</ol>{/if}
			{:else if selectedDay}
				<p class="eyebrow">DATE SLICE</p><h2>{selectedDay.date}</h2><p class="slice-summary">{selectedDay.count} memories entered this valid-time slice. Select one below to inspect its receipt.</p>
			{:else}
				<p class="eyebrow">FIELD IS LIVE</p><h2>Choose a ring, date, or memory.</h2><p class="slice-summary">The field shows growth. This panel makes the evidence legible.</p>
			{/if}
		</aside>
	</section>

	{#if selectedDay}
		<section class="memory-strip glass-panel">
			<div class="panel-label"><span>MEMORIES IN {selectedDay.date}</span><strong>{selectedDay.memories.length} RECORDS</strong></div>
			<div class="memory-buttons">
				{#each selectedDay.memories.slice(0, 20) as memory (memory.id)}
					<button type="button" class:active={selectedMemoryId === memory.id} onclick={() => selectMemory(memory, selectedDay.date)}><strong>{memory.content}</strong><small>{memory.id.slice(0, 8)} · {Math.round(memory.retentionStrength * 100)}% retention</small></button>
				{/each}
			</div>
		</section>
	{/if}
</main>

<style>
	.timeline-shell{position:relative;z-index:2;max-width:1180px;min-height:100%;margin:0 auto;padding:2rem clamp(1rem,3vw,2.5rem) 5rem;color:#eaf9f6;pointer-events:none}.timeline-head,.timeline-grid,.vitals,.memory-buttons{display:flex}.timeline-head{justify-content:space-between;align-items:flex-end;gap:2rem}.eyebrow,.panel-label{margin:0;color:#66e6d3;font:700 .68rem/1.2 ui-monospace,monospace;letter-spacing:.14em}.timeline-head h1{max-width:22ch;margin:.55rem 0;font-size:clamp(1.7rem,3.3vw,2.75rem);line-height:1.05;letter-spacing:-.045em}.lede,.slice-summary{max-width:62ch;color:#a9c4c0;line-height:1.5}.range-control{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:.4rem;max-width:19rem;pointer-events:auto}.range-control span{width:100%;color:#9ab8b3;font:700 .65rem ui-monospace,monospace;text-align:right}.range-control button,.day-rows button,.memory-buttons button{border:1px solid rgba(104,202,187,.2);border-radius:.55rem;background:rgba(4,17,19,.82);color:#a9c4c0;cursor:pointer}.range-control button{padding:.5rem .6rem;font:700 .72rem ui-monospace,monospace}.range-control button:hover,.range-control button.active{border-color:#60e2cf;background:rgba(0,222,193,.16);color:#eafffb}.vitals{gap:.65rem;justify-content:space-between;margin:1.25rem 0}.vitals div{min-width:0;flex:1;border-left:1px solid rgba(102,230,211,.3);padding-left:.8rem}.vitals dt{color:#8da9a5;font-size:.7rem}.vitals dd{margin:.25rem 0 0;color:#72e7d5;font-size:1.45rem}.timeline-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,.8fr);gap:1rem;pointer-events:auto}.glass-panel{border:1px solid rgba(124,198,187,.2);border-radius:1rem;background:linear-gradient(135deg,rgba(7,24,27,.9),rgba(4,12,15,.84));backdrop-filter:blur(14px);box-shadow:0 18px 70px rgba(0,0,0,.24)}.day-list,.receipt,.memory-strip{padding:1rem}.panel-label{display:flex;justify-content:space-between;color:#a7c8c2}.panel-label strong{color:#66e6d3;font-size:.63rem}.day-rows{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:.55rem;margin-top:1rem}.day-rows button{padding:.72rem;text-align:left}.day-rows button:hover,.day-rows button.active,.memory-buttons button:hover,.memory-buttons button.active{border-color:rgba(91,231,207,.58);background:rgba(0,220,189,.1)}.day-rows span,.day-rows strong,.day-rows small,.memory-buttons strong,.memory-buttons small{display:block}.day-rows strong{margin:.3rem 0;color:#75e5d4;font-size:1.25rem}.day-rows small,.memory-buttons small{color:#87a9a3;font:.63rem ui-monospace,monospace}.receipt{min-height:22rem}.receipt h2{margin:.7rem 0 1rem;color:#f1fffc;font-size:1.1rem;line-height:1.48}.receipt h3{margin:1.2rem 0 .5rem;color:#b8d8d2;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}.receipt-metrics{margin:0}.receipt-metrics div{border-top:1px solid rgba(137,190,183,.14);padding:.55rem 0}.receipt-metrics dt{color:#87a6a1;font-size:.68rem}.receipt-metrics dd{margin:.2rem 0 0;color:#d9f4ee;font-size:.77rem;overflow-wrap:anywhere}.receipt code{color:#7be6d6;font-size:.65rem}.receipt ol{margin:0;padding:0;list-style:none}.receipt li{display:flex;justify-content:space-between;gap:.75rem;border-top:1px solid rgba(137,190,183,.12);padding:.48rem 0;color:#a9c6c1;font-size:.74rem}.receipt li strong{color:#6ce4d1}.receipt li span{color:#7f9f9a;text-align:right}.state-line{color:#96b9b3;font-size:.82rem}.error{color:#ff9d93}.memory-strip{position:relative;margin-top:1rem;pointer-events:auto}.memory-buttons{flex-wrap:wrap;gap:.5rem;margin-top:.8rem}.memory-buttons button{max-width:22rem;padding:.65rem;text-align:left}.memory-buttons strong{overflow:hidden;color:#dff6f0;font-size:.78rem;line-height:1.35;text-overflow:ellipsis;white-space:nowrap}@media(max-width:760px){.timeline-head,.timeline-grid{display:grid;grid-template-columns:1fr}.range-control{justify-content:flex-start;max-width:none}.range-control span{text-align:left}.vitals{display:grid;grid-template-columns:1fr 1fr}.timeline-head{gap:.5rem}.day-rows{grid-template-columns:1fr 1fr}.receipt li{display:block}.receipt li span{display:block;margin-top:.2rem;text-align:left}}
</style>

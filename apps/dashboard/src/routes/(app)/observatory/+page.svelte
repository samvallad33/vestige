<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { GraphResponse } from '$types';
	import TelemetryStrip from '$lib/observatory/overlays/TelemetryStrip.svelte';
	import TimelineSpine from '$lib/observatory/overlays/TimelineSpine.svelte';
	import RescueVerdict from '$lib/observatory/overlays/RescueVerdict.svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import { isDemoMode, type DemoMode } from '$lib/observatory/types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { NodeRenderer } from '$lib/observatory/node-renderer';
	import { BirthRenderer } from '$lib/observatory/birth-renderer';
	import { RescueRenderer } from '$lib/observatory/rescue-renderer';
	import { buildRescuePlan, type RescuePlan } from '$lib/observatory/rescue-plan';
	import { ForgettingRenderer } from '$lib/observatory/forgetting-renderer';
	import { buildForgettingPlan } from '$lib/observatory/forgetting-plan';
	import { FirewallRenderer } from '$lib/observatory/firewall-renderer';
	import { buildFirewallPlan, type FirewallPlan } from '$lib/observatory/firewall-plan';
	import type { PathStepMeta } from '$lib/observatory/path-builder';

	// URL contract: ?demo=recall-path&seed=vestige-observatory-v1
	const params = new URLSearchParams(window.location.search);
	const demoParam = params.get('demo') ?? 'recall-path';
	const demoMode: DemoMode = isDemoMode(demoParam) ? demoParam : 'recall-path';
	const seedValue = params.get('seed') ?? 'vestige-observatory-v1';
	// Capture mode: ?frame=N freezes the sim at one loop frame (identical pixels).
	const frameParam = params.get('frame');
	const freezeFrame = frameParam !== null && frameParam !== '' ? Number(frameParam) : null;
	// Recording mode: ?capture=1 hides EVERY DOM instrument — pure canvas for
	// clips/stills. &hud=1 keeps the instruments; H toggles at runtime.
	let showHud = $state(!(params.get('capture') === '1' && params.get('hud') !== '1'));
	const isCapture = params.get('capture') === '1';

	function onKeydown(e: KeyboardEvent) {
		if (e.key === 'h' || e.key === 'H') showHud = !showHud;
	}

	let graphData: GraphResponse | null = $state(null);
	let loading = $state(true);
	let error = $state('');

	// Telemetry state — frames come from the engine's deterministic DemoClock.
	let frameCount = $state(0);
	let fpsEstimate = $state(0);
	let nodeCount = $state(0);
	let edgeCount = $state(0);
	let centerId = $state('');

	// Engine + renderer handles (upload happens once both are ready).
	let engine = $state<ObservatoryEngine | null>(null);
	let renderer: NodeRenderer | null = null;
	let birthRenderer: BirthRenderer | null = null;
	let rescueRenderer: RescueRenderer | null = null;
	let rescuePlan = $state<RescuePlan | null>(null);
	let forgettingRenderer: ForgettingRenderer | null = null;
	let firewallRenderer: FirewallRenderer | null = null;
	let firewallPlan = $state<FirewallPlan | null>(null);
	let uploaded = false;
	let pathSteps = $state<PathStepMeta[]>([]);

	async function loadGraph() {
		loading = true;
		error = '';
		try {
			const data = await api.graph({
				max_nodes: 300,
				depth: 3,
				sort: 'recent'
			});
			graphData = data;
			nodeCount = data.nodeCount;
			edgeCount = data.edgeCount;
			centerId = data.center_id;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load graph data';
		} finally {
			loading = false;
		}
	}

	function handleFrame(frame: number, fps: number) {
		frameCount = frame;
		fpsEstimate = fps;
	}

	function handleReady(e: ObservatoryEngine) {
		// A fresh engine (first boot or HMR re-create) always needs an upload.
		uploaded = false;
		engine = e;
		renderer = new NodeRenderer(e);
	}

	// Upload the memory field once the engine AND the graph are both ready.
	$effect(() => {
		if (engine && renderer && graphData && !uploaded) {
			uploaded = true;
			const isBirth = demoMode === 'engram-birth';
			const isRescue = demoMode === 'salience-rescue';
			const isHorizon = demoMode === 'forgetting-horizon';
			const isFirewall = demoMode === 'firewall';
			renderer.upload(graphData, seedValue, {
				recallPath: !isBirth && !isRescue && !isHorizon && !isFirewall
			});

			if (isBirth) {
				// Moment B: violet dust converges into a newborn memory.
				birthRenderer = new BirthRenderer({ engine, nodeRenderer: renderer, seed: seedValue });
				birthRenderer.upload(seedValue);

				// B6: the engrave steps ride the proven recall wavefront system —
				// pulses travel outward from the newborn, neighbors bloom on landing.
				const engrave = birthRenderer.engraveSteps;
				const engraveMetas = [];
				for (let i = 0; i < engrave.length / 4; i++) {
					engraveMetas.push({
						sourceIndex: engrave[i * 4],
						targetIndex: engrave[i * 4 + 1],
						beatFrame: engrave[i * 4 + 2],
						kind: engrave[i * 4 + 3],
						beatKind: 'engrave',
						nodeId: `engrave-${i}`,
						label: 'edge engraved'
					});
				}
				renderer.setPathSteps(engrave, engraveMetas);

				// Spine shows the birth beats (converge → flash → engrave).
				pathSteps = birthRenderer.timeline.map((b, i) => ({
					sourceIndex: 0,
					targetIndex: 0,
					beatFrame: b.startFrame,
					kind: 0,
					beatKind: 'birth',
					nodeId: `birth-${i}`,
					label: b.label
				}));
			} else if (isRescue) {
				// Moment C: Retroactive Salience Backfill — vector search fails on
				// camera, then Vestige reaches backward through time and ignites
				// the true cause. All choreography is a pure CPU plan; the
				// RescueRenderer overwrites the demo lanes AFTER recall_sim
				// (construction order = pass order — load-bearing for the seam).
				const plan = buildRescuePlan(graphData, renderer.graph!, seedValue);
				rescuePlan = plan;
				if (plan.viable) {
					rescueRenderer = new RescueRenderer({ engine, nodeRenderer: renderer, plan });
					rescueRenderer.upload();
					// Probe beams + backward wave tree + causal arc ride the proven
					// path-ribbon machinery. Metas are 1:1 with the GPU steps
					// (setPathSteps sets params[4] + draw count from META length).
					renderer.setPathSteps(plan.pathData, plan.pathMetas);
				}
				// Spine shows the curated story beats (unique beatFrames), never
				// the raw GPU steps.
				pathSteps = plan.spineBeats;
			} else if (isHorizon) {
				// Moment D: the forgetting horizon — FSRS as a visible living
				// system. The lowest-retention memories dim and fall; three are
				// recalled back on camera; the rest sink to a 6% floor (never
				// deleted). Pure CPU plan; the ForgettingRenderer overwrites the
				// demo lanes AFTER recall_sim (construction order = pass order).
				const plan = buildForgettingPlan(renderer.graph!);
				if (plan.viable) {
					forgettingRenderer = new ForgettingRenderer({ engine, nodeRenderer: renderer, plan });
					forgettingRenderer.upload();
					// The 3 rescue ribbons ride the proven path-ribbon machinery.
					renderer.setPathSteps(plan.pathData, plan.pathMetas);
				}
				// No verdict overlay for this moment — the spine narrates.
				pathSteps = plan.spineBeats;
			} else if (isFirewall) {
				// Moment E: the immune response — a suspicious memory flares,
				// a crimson shockwave crosses the field, its edges sever one by
				// one, a membrane rings it, the verdict card lands. Pure CPU
				// plan; the FirewallRenderer overwrites the demo lanes AFTER
				// recall_sim (construction order = pass order).
				const plan = buildFirewallPlan(renderer.graph!, seedValue);
				firewallPlan = plan;
				if (plan.viable) {
					firewallRenderer = new FirewallRenderer({ engine, nodeRenderer: renderer, plan });
					firewallRenderer.upload();
					// The sever probe-beams ride the proven path-ribbon machinery.
					renderer.setPathSteps(plan.pathData, plan.pathMetas);
				}
				pathSteps = plan.spineBeats;
			} else {
				pathSteps = renderer.pathSteps;
			}

			// Start the story at frame 0 now that the field exists — where the
			// loop begins must never depend on how long the API call took.
			engine.demoClock.reset();
		}
	});

	onMount(() => {
		loadGraph();
	});
</script>

<svelte:window onkeydown={onKeydown} />

<!-- Full-bleed void stage: #05060a -->
<div class="fixed inset-0 overflow-hidden bg-[#05060a]" class:cursor-none={isCapture}>
	<!-- Canvas layer (z-index 0) — the living memory field -->
	<div class="absolute inset-0 z-0">
		<ObservatoryCanvas
			demo={demoMode}
			seed={seedValue}
			{freezeFrame}
			onframe={handleFrame}
			onready={handleReady}
		/>
	</div>

	<!-- DOM overlay layer (pointer-events:none) — instruments only.
	     ?capture=1 hides it all (pure canvas for recording); H toggles. -->
	{#if showHud}
	<div class="absolute inset-0 z-10 pointer-events-none">
		<!-- Top telemetry strip -->
		<TelemetryStrip
			demoMode={demoMode}
			seed={seedValue}
			nodeCount={nodeCount}
			edgeCount={edgeCount}
			centerId={centerId}
			frameCount={frameCount}
			fpsEstimate={fpsEstimate}
			{freezeFrame}
			loading={loading}
			error={error}
		/>

		<!-- Loading state -->
		{#if loading}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-auto">
				<div class="text-[#5dcaa5] font-mono text-sm tracking-widest animate-pulse">
					LOADING MEMORY FIELD...
				</div>
			</div>
		{/if}

		<!-- Error state -->
		{#if error && !loading}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-auto">
				<div class="text-red-400 font-mono text-sm border border-red-900/50 bg-red-950/30 px-4 py-2 rounded">
					{error}
				</div>
			</div>
		{/if}

		<!-- Timeline spine: beat ticks + playhead riding the loop -->
		<TimelineSpine steps={pathSteps} frame={frameCount} />

		<!-- Salience-rescue verdict card (frames 600-660, frame-driven opacity) -->
		{#if demoMode === 'salience-rescue' && rescuePlan?.viable}
			<RescueVerdict frame={frameCount} verdict={rescuePlan.verdict} />
		{/if}

		<!-- Firewall quarantine verdict card (frames 480-620, crimson tone) -->
		{#if demoMode === 'firewall' && firewallPlan?.viable}
			<RescueVerdict
				frame={frameCount}
				tone="quarantine"
				fadeWindow={[480, 495, 605, 620]}
				verdict={{
					headline: firewallPlan.verdict.headline,
					causeLabel: firewallPlan.verdict.intruderLabel,
					receipt: firewallPlan.verdict.receipt
				}}
			/>
		{/if}

		<!-- Empty graph state -->
		{#if !loading && graphData && graphData.nodeCount === 0}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-auto">
				<div class="text-[#5dcaa5] font-mono text-sm tracking-widest">
					NO MEMORIES IN FIELD
				</div>
			</div>
		{/if}
	</div>
	{/if}
</div>

<!-- VISUAL DNA §7: void #05060a is the base (bg utility above) — no exceptions -->

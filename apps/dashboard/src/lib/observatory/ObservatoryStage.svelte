<script lang="ts">
	/**
	 * ObservatoryStage — the full Cognitive Observatory experience as a
	 * reusable, props-driven component so it can live in TWO places:
	 *
	 *   1. The MAIN dashboard graph page (primary integration): launched from
	 *      the graph control bar next to Dream / Memory Cinema, full-bleed
	 *      overlay, with the on-screen demo switcher + exit control.
	 *   2. The /observatory route (thin wrapper): deep-linking + ?frame=N
	 *      capture workflow for recordings stays byte-compatible.
	 *
	 * All engine/plan/overlay wiring moved verbatim from the route page —
	 * behavior is identical; only the input source changed (props, not URL).
	 */
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { GraphResponse } from '$types';
	import TelemetryStrip from '$lib/observatory/overlays/TelemetryStrip.svelte';
	import TimelineSpine from '$lib/observatory/overlays/TimelineSpine.svelte';
	import RescueVerdict from '$lib/observatory/overlays/RescueVerdict.svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import { DEMO_MODES, type DemoMode } from '$lib/observatory/types';
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

	interface Props {
		demo: DemoMode;
		seed?: string;
		/** ?frame=N capture freeze — null runs the live loop. */
		freezeFrame?: number | null;
		/** Recording mode: hides EVERY DOM instrument (pure canvas). */
		capture?: boolean;
		/** Show the on-screen demo switcher (the 5 moments). */
		showSwitcher?: boolean;
		/** Called when the user picks another demo moment. */
		ondemochange?: (demo: DemoMode) => void;
		/** When provided, renders the exit control (× back to the dashboard). */
		onexit?: () => void;
		/**
		 * Embedded mode — the stage fills its PARENT (absolute) instead of the
		 * viewport (fixed). Used by the main graph page, where the field is the
		 * default renderer living under the page's own chrome.
		 */
		embedded?: boolean;
		/**
		 * 'full'  — telemetry strip, spine, verdicts, switcher, exit (the route
		 *           and the takeover overlay).
		 * 'none'  — pure living canvas + loading/error text only; the host page
		 *           provides all chrome (the main graph's field renderer).
		 */
		chrome?: 'full' | 'none';
	}

	let {
		demo,
		seed = 'vestige-observatory-v1',
		freezeFrame = null,
		capture = false,
		showSwitcher = true,
		ondemochange,
		onexit,
		embedded = false,
		chrome = 'full'
	}: Props = $props();

	// Human labels for the switcher chips — short, mono, uppercase (visual DNA §7.3).
	const DEMO_LABELS: Record<DemoMode, string> = {
		'recall-path': 'RECALL',
		'engram-birth': 'BIRTH',
		'salience-rescue': 'RESCUE',
		'forgetting-horizon': 'HORIZON',
		firewall: 'FIREWALL'
	};

	// Intentional initial-value capture: `capture` is fixed for the lifetime of
	// a mount ({#key} remounts on change) and the H key mutates showHud freely
	// afterward — a $derived would fight the toggle.
	// svelte-ignore state_referenced_locally
	let showHud = $state(!capture);

	function onKeydown(e: KeyboardEvent) {
		if (e.key === 'h' || e.key === 'H') showHud = !showHud;
		if (e.key === 'Escape' && onexit) onexit();
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
			// Pull the DENSE real subgraph (the well-connected hotspot), not the
			// newest-memory neighborhood — 'recent' centers on a lonely fresh node
			// (~12 nodes), while 'connected' surfaces the populous, edge-rich field
			// (~150 real memories, thousands of edges) so the Observatory reads as
			// a living brain doing real work, not a sparse placeholder.
			const data = await api.graph({
				max_nodes: 200,
				depth: 3,
				sort: 'connected'
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
			const isBirth = demo === 'engram-birth';
			const isRescue = demo === 'salience-rescue';
			const isHorizon = demo === 'forgetting-horizon';
			const isFirewall = demo === 'firewall';
			renderer.upload(graphData, seed, {
				recallPath: !isBirth && !isRescue && !isHorizon && !isFirewall
			});

			if (isBirth) {
				// Moment B: violet dust converges into a newborn memory.
				birthRenderer = new BirthRenderer({ engine, nodeRenderer: renderer, seed });
				birthRenderer.upload(seed);

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
				const plan = buildRescuePlan(graphData, renderer.graph!, seed);
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
				const plan = buildFirewallPlan(renderer.graph!, seed);
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

<!-- Void stage: #05060a — full-bleed (route/takeover) or parent-filling (embedded) -->
<div
	class="{embedded ? 'absolute' : 'fixed'} inset-0 overflow-hidden bg-[#05060a]"
	class:cursor-none={capture}
>
	<!-- Canvas layer (z-index 0) — the living memory field -->
	<div class="absolute inset-0 z-0">
		<ObservatoryCanvas
			{demo}
			{seed}
			{freezeFrame}
			onframe={handleFrame}
			onready={handleReady}
		/>
	</div>

	<!-- DOM overlay layer (pointer-events:none) — instruments only.
	     capture hides it all (pure canvas for recording); H toggles;
	     chrome='none' leaves only loading/error (host page owns the chrome). -->
	{#if showHud}
	<div class="absolute inset-0 z-10 pointer-events-none">
		{#if chrome === 'full'}
		<!-- Top telemetry strip -->
		<TelemetryStrip
			demoMode={demo}
			{seed}
			{nodeCount}
			{edgeCount}
			{centerId}
			{frameCount}
			{fpsEstimate}
			{freezeFrame}
			{loading}
			{error}
		/>
		{/if}

		<!-- Exit control — back to the dashboard graph. Esc works too. -->
		{#if chrome === 'full' && onexit}
			<button
				onclick={onexit}
				class="absolute top-10 right-4 pointer-events-auto font-mono text-xs tracking-widest
					text-[#5dcaa5]/70 hover:text-[#5dcaa5] border border-[#5dcaa5]/25 hover:border-[#5dcaa5]/60
					bg-[#05060a]/70 rounded px-3 py-1.5 transition-colors"
				title="Exit Observatory (Esc)"
			>
				× EXIT
			</button>
		{/if}

		<!-- Demo moment switcher — the 5 cognitive moments, one click each. -->
		{#if chrome === 'full' && showSwitcher}
			<div class="absolute top-10 left-4 pointer-events-auto flex flex-col gap-1.5">
				{#each DEMO_MODES as mode (mode)}
					<button
						onclick={() => ondemochange?.(mode)}
						class="font-mono text-[11px] tracking-widest text-left rounded px-3 py-1.5 border transition-colors
							{mode === demo
								? 'text-[#05060a] bg-[#5dcaa5] border-[#5dcaa5]'
								: 'text-[#5dcaa5]/60 hover:text-[#5dcaa5] bg-[#05060a]/70 border-[#5dcaa5]/20 hover:border-[#5dcaa5]/50'}"
						title="Play the {DEMO_LABELS[mode]} moment"
					>
						{DEMO_LABELS[mode]}
					</button>
				{/each}
			</div>
		{/if}

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
		{#if chrome === 'full'}
			<TimelineSpine steps={pathSteps} frame={frameCount} />
		{/if}

		<!-- Salience-rescue verdict card (frame-driven opacity) -->
		{#if chrome === 'full' && demo === 'salience-rescue' && rescuePlan?.viable}
			<RescueVerdict frame={frameCount} verdict={rescuePlan.verdict} />
		{/if}

		<!-- Firewall quarantine verdict card (frames 480-620, crimson tone) -->
		{#if chrome === 'full' && demo === 'firewall' && firewallPlan?.viable}
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

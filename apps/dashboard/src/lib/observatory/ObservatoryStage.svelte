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
	import {
		ReceiptReplay,
		receiptsToReplayItems,
		fieldNodesToReplayItems,
		mostRecalledMemories
	} from '$lib/observatory/receipt-replay';
	import { eventFeed } from '$stores/websocket';
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
	import {
		buildRescuePlan,
		type ReceiptBackfillEvidence,
		type RescuePlan
	} from '$lib/observatory/rescue-plan';
	import { ForgettingRenderer } from '$lib/observatory/forgetting-renderer';
	import { buildForgettingPlan } from '$lib/observatory/forgetting-plan';
	import { FirewallRenderer } from '$lib/observatory/firewall-renderer';
	import { buildFirewallPlan, type FirewallPlan } from '$lib/observatory/firewall-plan';
	import type { PathStepMeta } from '$lib/observatory/path-builder';
	import { LiveBridge } from '$lib/observatory/live-bridge';
	import { ChronoShuttlePass } from '$lib/observatory/chrono/shuttle-pass';
	import { FossilLightTransportPass } from '$lib/observatory/chrono/radiance-cascade-pass';
	import { CameraRigController } from '$lib/observatory/camera-rig';
	import PickReceipt, { type PickProvenance } from '$lib/observatory/overlays/PickReceipt.svelte';

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
		/**
		 * GPU picking: when provided, clicking a memory node in the field calls
		 * back with its memory id (the host opens its inspector panel).
		 */
		onpick?: (memoryId: string) => void;
		/** Route-local all-WebGPU instrument overlays can attach their own passes. */
		onready?: (engine: ObservatoryEngine) => void;
		/**
		 * Devicepixel clamp forwarded to the engine. Export mounts pin this to 1
		 * so a 1920×1080 host renders a bitmap of exactly 1920×1080 — the fixed
		 * resolution the byte-identical clip contract requires.
		 */
		maxDpr?: number;
		/**
		 * When a Memory Receipt opens the Observatory, constrain the field to the
		 * receipt's real retrieved/suppressed ids. No topology-derived stand-ins.
		 */
		focusIds?: string[];
		/** Exact Backfill receipt data. When present, salience-rescue renders only
		 * those evidence ids; it never falls back to graph/layout inference. */
		backfillEvidence?: ReceiptBackfillEvidence;
		/**
		 * v2.3 living field — subscribe the field to the REAL backend event
		 * stream ($eventFeed) so it renders live FSRS decay, the contradiction
		 * firewall, the dream storm, and the causal recall wavefront driven by
		 * real cognitive events instead of the deterministic DemoClock. Only the
		 * main graph's field renderer turns this on; the scripted ?demo= moments
		 * stay deterministic (live=false).
		 */
		live?: boolean;
		/**
		 * Offline structure field (`?brain=`). When set, skip the live graph
		 * fetch and upload this topology instead. Labels are type·index tokens
		 * only — zero memory text.
		 */
		graphOverride?: import('$types').GraphResponse | null;
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
		chrome = 'full',
		onpick,
		onready,
		maxDpr = 2,
		focusIds = [],
		backfillEvidence,
		live = false,
		graphOverride = null
	}: Props = $props();

	// FOSSIL LIGHT — the memory time axis. ONE signed control in days relative
	// to NOW: negative rewinds the whole field into the brain's real past
	// (retention re-evaluated at that instant on the true closed form,
	// memories unborn before their createdAt pop out of existence), positive
	// is the original forward forgetting-horizon projection. 0 = live now.
	let timeAxisDays = $state(0);
	let chronoScrubbing = $state(false);
	// Lower bound = the oldest memory's birthday (set after the graph loads).
	let chronoMinDays = $state(0);
	// ?t=<ISO8601> deep link — applied once the graph loads so it clamps to
	// the real oldest-memory bound. Composes with ?frame=N capture.
	let pendingChronoIso: string | null = null;
	const projectionDays = $derived(Math.max(0, timeAxisDays));
	const chronoOffsetDays = $derived(Math.min(0, timeAxisDays));
	const chronoLabel = $derived(
		timeAxisDays === 0
			? 'now'
			: timeAxisDays > 0
				? `+${Math.round(timeAxisDays)}d`
				: new Date(Date.now() + timeAxisDays * 86_400_000).toLocaleDateString(undefined, {
						month: 'short',
						day: 'numeric'
					})
	);
	let liveBridge: LiveBridge | null = null;
	let shuttlePass: ChronoShuttlePass | null = null;
	let radiancePass: FossilLightTransportPass | null = null;
	let shuttleReady = $state(false);
	let liveDecayReady = $state(false);

	// FOSSIL LIGHT — grab the GPU rail itself. The phosphor rail band (bottom
	// of the canvas, mirroring shuttle-pass RAIL_* clip constants) is a direct
	// pointer target: dragging maps clientX across the rail span onto the SAME
	// [oldest, +365d] domain the shuttle renders; release keeps momentum with
	// friction; a glide that comes home within a day of NOW snaps to it.
	// Time has mass, and NOW is magnetic.
	const RAIL_SPAN = 0.835; // shuttle-pass RAIL_LEFT/RIGHT (NDC)
	const RAIL_Y_FRAC = (1 + 0.685) / 2; // shuttle-pass RAIL_Y → screen-y fraction
	let railDragging = false;
	let railGlideRaf = 0;
	let railVel = 0; // days per 60Hz frame
	let railLastT = 0;
	let suppressNextClick = false;

	function railXToDays(clientX: number): number {
		const rect = canvasLayerEl?.getBoundingClientRect();
		if (!rect || rect.width === 0) return timeAxisDays;
		const ndc = ((clientX - rect.left) / rect.width) * 2 - 1;
		const t = Math.max(0, Math.min(1, (ndc / RAIL_SPAN) * 0.5 + 0.5));
		return chronoMinDays + t * (365 - chronoMinDays);
	}
	function inRailBand(e: PointerEvent): boolean {
		const rect = canvasLayerEl?.getBoundingClientRect();
		if (!rect || rect.height === 0) return false;
		const fy = (e.clientY - rect.top) / rect.height;
		return fy > RAIL_Y_FRAC - 0.075 && fy < RAIL_Y_FRAC + 0.075;
	}
	function stopGlide() {
		if (railGlideRaf) cancelAnimationFrame(railGlideRaf);
		railGlideRaf = 0;
	}
	function railPointerDown(e: PointerEvent) {
		if (!shuttleReady || capture || !inRailBand(e)) return;
		stopGlide();
		railDragging = true;
		chronoScrubbing = true;
		railVel = 0;
		railLastT = performance.now();
		timeAxisDays = railXToDays(e.clientX);
		(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
		e.preventDefault();
	}
	function railPointerMove(e: PointerEvent) {
		if (!railDragging) return;
		const now = performance.now();
		const days = railXToDays(e.clientX);
		const dt = Math.max(1, now - railLastT);
		// days-per-16ms fling velocity, low-pass filtered so one jittery
		// pointer sample cannot launch the timeline into orbit.
		railVel = railVel * 0.6 + ((days - timeAxisDays) / dt) * 16 * 0.4;
		railLastT = now;
		timeAxisDays = days;
	}
	function railPointerUp(e: PointerEvent) {
		if (!railDragging) return;
		railDragging = false;
		suppressNextClick = true;
		(e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
		const glide = () => {
			railGlideRaf = 0;
			railVel *= 0.94;
			let next = timeAxisDays + railVel;
			if (next <= chronoMinDays) {
				next = chronoMinDays;
				railVel = 0;
			}
			if (next >= 365) {
				next = 365;
				railVel = 0;
			}
			// NOW-snap: a glide heading home within a day of 0 locks to it.
			const towardNow = (timeAxisDays < 0 && railVel > 0) || (timeAxisDays > 0 && railVel < 0);
			if (Math.abs(next) < 1 && towardNow) {
				next = 0;
				railVel = 0;
			}
			timeAxisDays = next;
			if (Math.abs(railVel) > 0.02) railGlideRaf = requestAnimationFrame(glide);
			else chronoScrubbing = false;
		};
		if (Math.abs(railVel) > 0.05) railGlideRaf = requestAnimationFrame(glide);
		else chronoScrubbing = false;
	}
	function railPointerCancel() {
		railDragging = false;
		chronoScrubbing = false;
		stopGlide();
	}

	// Motion control (WCAG): the field's ambient orbit/sim drift runs >5s, so it
	// needs a persistent pause control AND must honor prefers-reduced-motion.
	// When paused the clock freezes (drift stops) but live event pulses still
	// land — they are information, not decoration. Auto-pauses under
	// reduced-motion; the user can still override with the button.
	let paused = $state(false);
	let userSetPause = $state(false);
	function initReducedMotion() {
		if (typeof window === 'undefined') return;
		const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
		if (mq.matches && !userSetPause) paused = true;
		const onChange = (e: MediaQueryListEvent) => {
			if (!userSetPause) paused = e.matches;
		};
		mq.addEventListener('change', onChange);
		return () => mq.removeEventListener('change', onChange);
	}
	function togglePause() {
		userSetPause = true;
		paused = !paused;
	}
	// Push the pause state to the engine whenever either changes.
	$effect(() => {
		engine?.setPaused(paused);
	});
	// Live contradiction-firewall verdict — set when a real MemorySuppressed /
	// contradiction event quarantines a memory on camera. Held ~7s then fades.
	let liveFirewallLabel = $state('');
	let liveFirewallAt = $state(0);
	let liveFirewallVisible = $derived(liveFirewallLabel !== '' && liveFirewallAt > 0);

	// GPU picking — screen px → NDC → NodeRenderer.pickAt (one readback/click).
	let canvasLayerEl: HTMLDivElement | null = $state(null);
	const cameraRig = new CameraRigController();
	let lastPick = $state<PickProvenance | null>(null);
	let hoverAt = 0;
	let hoverLabel = $state('');

	async function handleFieldClick(e: MouseEvent) {
		// A rail drag OR camera orbit ends in a click on this same layer —
		// never turn the tail of a scrub/orbit into an accidental GPU pick.
		if (suppressNextClick) {
			suppressNextClick = false;
			return;
		}
		if (!renderer || !canvasLayerEl) return;
		const rect = canvasLayerEl.getBoundingClientRect();
		if (rect.width === 0 || rect.height === 0) return;
		const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
		const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
		const hit = await renderer.pickAt(ndcX, ndcY);
		if (hit) {
			lastPick = { kind: 'memory', id: hit.id, label: 'Field cell' };
			onpick?.(hit.id);
		}
	}

	function fieldPointerDown(e: PointerEvent) {
		railPointerDown(e);
		if (railDragging || capture) return;
		cameraRig.enabled = !capture;
		cameraRig.onPointerDown(e);
	}
	function fieldPointerMove(e: PointerEvent) {
		railPointerMove(e);
		if (railDragging || capture) return;
		if (cameraRig.onPointerMove(e)) {
			suppressNextClick = true;
			renderer?.setCameraRig(cameraRig.state);
		}
		const now = performance.now();
		if (now - hoverAt < 120 || !renderer || !canvasLayerEl) return;
		hoverAt = now;
		const rect = canvasLayerEl.getBoundingClientRect();
		if (rect.width === 0) return;
		const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
		const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
		void renderer.pickAt(ndcX, ndcY).then((hit) => {
			renderer?.setHovered(hit?.index ?? -1);
			hoverLabel = hit?.id?.slice(0, 8) ?? '';
			if (canvasLayerEl) canvasLayerEl.style.cursor = hit ? 'crosshair' : 'grab';
		});
	}
	function fieldPointerUp(e: PointerEvent) {
		railPointerUp(e);
		cameraRig.onPointerUp(e);
	}
	function fieldPointerCancel() {
		railPointerCancel();
		cameraRig.onPointerUp({ pointerId: -1 } as PointerEvent);
	}
	function fieldWheel(e: WheelEvent) {
		if (capture || inRailBand(e as unknown as PointerEvent)) return;
		if (cameraRig.onWheel(e)) renderer?.setCameraRig(cameraRig.state);
	}

	// Human labels for the switcher chips — short, mono, uppercase (visual DNA §7.3).
	const DEMO_LABELS: Record<DemoMode, string> = {
		'recall-path': 'RECALL',
		'engram-birth': 'BIRTH',
		'salience-rescue': 'RESCUE',
		'forgetting-horizon': 'HORIZON',
		firewall: 'FIREWALL'
	};

	/**
	 * Bounded memory-light source selection. Every index names a real graph
	 * memory; the GPU subsequently owns its position, camera projection, live
	 * FSRS retention, birth mask, and suppression flag. Ranking is a one-time
	 * cost guard, never a substitute for the full 3D graph's data.
	 */
	function fossilLightSourceIndices(): Uint32Array {
		if (!renderer?.graph) return new Uint32Array();
		return new Uint32Array(
			renderer.graph.nodes
				.map((node) => ({ index: node.index, id: node.id, retention: node.retention }))
				.sort((a, b) => b.retention - a.retention || a.id.localeCompare(b.id))
				.slice(0, 64)
				.map((node) => node.index)
				.sort((a, b) => a - b)
		);
	}

	// Intentional initial-value capture: `capture` is fixed for the lifetime of
	// a mount ({#key} remounts on change) and the H key mutates showHud freely
	// afterward — a $derived would fight the toggle.
	// svelte-ignore state_referenced_locally
	let showHud = $state(!capture);

	function onKeydown(e: KeyboardEvent) {
		// Never steal shortcuts from a real DOM input, text area, or editable
		// inspector. The field is keyboard-operable, but text entry always wins.
		const target = e.target as HTMLElement | null;
		if (
			target?.isContentEditable ||
			target?.tagName === 'INPUT' ||
			target?.tagName === 'TEXTAREA' ||
			target?.tagName === 'SELECT'
		)
			return;
		if (e.key === 'h' || e.key === 'H') showHud = !showHud;
		if (e.key === 'Escape' && onexit) onexit();
		if ((e.key === ' ' || e.key.toLowerCase() === 'p') && !capture) {
			e.preventDefault();
			togglePause();
		}
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
			if (graphOverride) {
				graphData = graphOverride;
				nodeCount = graphOverride.nodeCount;
				edgeCount = graphOverride.edgeCount;
				centerId = graphOverride.center_id;
				return;
			}
			const focus = new Set(focusIds.filter(Boolean));
			const scoped = focus.size
				? await (async () => {
						// Receipt scope is an exact-ID contract. Resolve each requested
						// memory as the graph center before filtering, rather than filtering
						// an arbitrary connected cluster that may not contain it.
						const graphs = await Promise.all(
							[...focus].map((center_id) => api.graph({ center_id, max_nodes: 200, depth: 3 }))
						);
						const nodes = [
							...new Map(graphs.flatMap((graph) => graph.nodes).map((node) => [node.id, node])).values()
						].filter((node) => focus.has(node.id));
						const ids = new Set(nodes.map((node) => node.id));
						const edges = [
							...new Map(
								graphs.flatMap((graph) => graph.edges).map((edge) => [`${edge.source}:${edge.target}`, edge])
							).values()
						].filter((edge) => ids.has(edge.source) && ids.has(edge.target));
						return {
							...graphs[0],
							nodes,
							edges,
							center_id: nodes[0]?.id ?? graphs[0]?.center_id ?? '',
							nodeCount: nodes.length,
							edgeCount: edges.length
						} satisfies GraphResponse;
					})()
				: await api.graph({ max_nodes: 200, depth: 3, sort: 'connected' });
			graphData = scoped;
			nodeCount = scoped.nodeCount;
			edgeCount = scoped.edgeCount;
			centerId = scoped.center_id;
		} catch (e) {
			// Empty brain: get_graph 404s when the user has zero memories (a real
			// launch-day first-run cohort). Render the friendly "NO MEMORIES IN
			// FIELD" empty state (gated on graphData.nodeCount === 0), not a raw
			// "API 404: Not Found" error box.
			const msg = e instanceof Error ? e.message : 'Failed to load graph data';
			if (/\b404\b/.test(msg)) {
				graphData = {
					nodes: [],
					edges: [],
					nodeCount: 0,
					edgeCount: 0,
					center_id: ''
				} as unknown as GraphResponse;
				nodeCount = 0;
				edgeCount = 0;
				centerId = '';
			} else {
				error = msg;
			}
		} finally {
			loading = false;
		}
	}

	// COLD-OPEN AHA — the self-driving replay of the user's REAL recall history.
	// Built after the live bridge exists (needs field membership); ticked every
	// frame so past recalls fire through the same GCaMP wavefront as live ones.
	let receiptReplay: ReceiptReplay | null = null;
	// "This is uniquely YOUR data" proof surface: the user's most-recalled
	// memories (by real receipt frequency) — the nodes that saturate hottest.
	let topRecalled = $state<{ id: string; recalls: number; label: string }[]>([]);
	// Which real signal the proof panel is showing — receipt recall counts, or
	// (when receipts don't intersect the field) FSRS retention. Both real; the
	// label must say which so the number is never misread.
	let topRecalledSource = $state<'recalls' | 'retention'>('recalls');

	function handleFrame(frame: number, fps: number) {
		frameCount = frame;
		fpsEstimate = fps;
		// Drive the ambient receipt replay. Paused while the user actively
		// scrubs time (chrono) — the past shouldn't self-recall under the hand.
		if (receiptReplay && !chronoScrubbing) receiptReplay.tick(frame);
	}

	// Load the user's real recall receipts and wire the cold-open replay + the
	// most-recalled proof readout. Both are pure functions of THIS user's data.
	async function initReceiptReplay() {
		if (!liveBridge || !renderer?.graph) return;
		const graph = renderer.graph;
		const inField = (id: string) => graph.indexById.has(id);
		const labelFor = (id: string) =>
			graph.nodes[graph.indexById.get(id) ?? -1]?.label ?? id.slice(0, 8);

		let receipts: Awaited<ReturnType<typeof api.receipts.list>>['receipts'] = [];
		try {
			receipts = (await api.receipts.list(60))?.receipts ?? [];
		} catch {
			// Offline backend → field stays live-driven; fall through to field-node
			// replay so the cold-open is still alive.
		}

		// Prefer the user's REAL past recalls; fall back to recalling the field's
		// own real memories when receipts don't intersect the loaded field.
		let items = receiptsToReplayItems(receipts, inField);
		if (items.length === 0) {
			items = fieldNodesToReplayItems(graph.nodes, 12);
		}
		if (items.length > 0) {
			receiptReplay = new ReceiptReplay(liveBridge, { intervalFrames: 240 });
			receiptReplay.setItems(items);
		}

		// "Most recalled" proof: from real receipts when they overlap the field,
		// else the field's own most-retained memories (real per-user signal).
		const fromReceipts = mostRecalledMemories(receipts, inField, 3);
		if (fromReceipts.length > 0) {
			topRecalledSource = 'recalls';
			topRecalled = fromReceipts.map((m) => ({ ...m, label: labelFor(m.id) }));
		} else {
			topRecalledSource = 'retention';
			topRecalled = [...graph.nodes]
				// Skip empty-content memories so the marquee "your data" panel never
				// shows a blank name; fall back to the id stub if a label is empty.
				.filter((n) => (n.label ?? '').trim().length > 0)
				.sort((a, b) => b.retention - a.retention)
				.slice(0, 3)
				.map((n) => ({
					id: n.id,
					recalls: Math.round(n.retention * 100),
					label: n.label || n.id.slice(0, 8)
				}));
		}
	}

	function handleReady(e: ObservatoryEngine) {
		// A fresh engine (first boot or HMR re-create) always needs an upload.
		uploaded = false;
		engine = e;
		renderer = new NodeRenderer(e);
		cameraRig.enabled = !capture;
		if (capture) cameraRig.reset();
		renderer.setCameraRig(cameraRig.state);
		onready?.(e);
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
				// Moment B: luciferin dust converges into a newborn memory.
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
				const plan = buildRescuePlan(graphData, renderer.graph!, seed, backfillEvidence);
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

			// v2.3 living field — wire the field to the REAL backend event stream.
			// Only the main graph's field renderer (live=true); the scripted
			// ?demo= moments stay deterministic. The bridge drives the live lanes
			// + per-node FSRS decay via the engine's preFrameHook; the eventFeed
			// subscription below feeds it. Created here (once) because it needs
			// both the engine AND renderer.graph, which only exist post-upload.
			if (live && renderer.graph && graphData) {
				liveBridge = new LiveBridge({
					engine,
					renderer,
					graph: renderer.graph,
					response: graphData,
					seed,
					projectionDays: () => projectionDays,
					chronoOffsetDays: () => chronoOffsetDays,
					onFirewall: (info) => {
						liveFirewallLabel = info.intruderLabel;
						liveFirewallAt = Date.now();
					}
				});
				liveDecayReady = liveBridge.liveDecayAvailable;
				engine.setPreFrameHook((simFrame) => liveBridge?.drain(simFrame));

				// COLD-OPEN AHA — replay the user's REAL past recalls so the field
				// is alive the instant the dashboard loads. Never during capture
				// (?frame=N must freeze to identical pixels — a self-firing recall
				// would break reproducibility).
				if (!capture) void initReceiptReplay();

				// FOSSIL LIGHT — time-axis lower bound = the oldest memory's real
				// birthday (padded a day so its unbirth is reachable), then apply
				// any ?t= deep link now that the bound is known.
				let oldest = Number.POSITIVE_INFINITY;
				for (const n of renderer.graph.nodes) {
					if (n.createdAt) {
						const t = Date.parse(n.createdAt);
						if (Number.isFinite(t) && t < oldest) oldest = t;
					}
				}
				if (Number.isFinite(oldest)) {
					chronoMinDays = Math.floor((oldest - Date.now()) / 86_400_000) - 1;
				}
				if (pendingChronoIso) {
					const t = Date.parse(pendingChronoIso);
					if (Number.isFinite(t)) {
						timeAxisDays = Math.min(365, Math.max(chronoMinDays, (t - Date.now()) / 86_400_000));
					}
					pendingChronoIso = null;
				}
				// FOSSIL LIGHT W1 — bounded memory light transport. Sources are a
				// small deterministic subset of real graph memories, but projection,
				// Chrono/FSRS state and suppression are read from NodeRenderer's live
				// GPU buffers after simulation, so the light stays locked to the
				// moving 3D field without a CPU approximation or readback.
				// Capture mode (?frame=N / ?capture) must stay pixel-reproducible:
				// the shuttle's scrub position and the light field's idle cadence
				// both derive from wall-clock NOW, so neither instrument mounts
				// during a capture — the field itself stays byte-identical.
				if (!capture) {
					radiancePass = new FossilLightTransportPass(engine, renderer, fossilLightSourceIndices());
					engine.addPass(radiancePass);
					// The GPU shuttle is an instrument, not a second time model. It reads
					// the same graph timestamps the closed-form decay bridge uses, then
					// receives only the small signed control uniform while the user scrubs.
					shuttlePass = new ChronoShuttlePass(engine, renderer.graph.nodes);
					engine.addPass(shuttlePass);
					shuttleReady = true;
				}
				// Dev/verification hook — read live state from the console. Guarded
				// so it never runs in SSR and is harmless in production.
				if (typeof window !== 'undefined') {
					(window as unknown as { __vestigeLiveBridge?: unknown }).__vestigeLiveBridge =
						liveBridge;
				}
			}

			// Start the story at frame 0 now that the field exists — where the
			// loop begins must never depend on how long the API call took.
			engine.demoClock.reset();
		}
	});

	// Feed the live bridge every time the event store changes (newest-first).
	// The bridge dedupes internally (only events newer than the last seen are
	// applied) and is a no-op when live=false / the bridge isn't built yet.
	$effect(() => {
		const events = $eventFeed;
		if (liveBridge) liveBridge.ingest(events);
	});

	// The time axis recomputes decay immediately on change so dragging the
	// control is responsive (the per-frame drain throttles ambient decay).
	$effect(() => {
		void timeAxisDays;
		liveBridge?.refreshDecay();
		shuttlePass?.setTimeline(timeAxisDays, chronoScrubbing);
		radiancePass?.setScrubbing(chronoScrubbing);
	});

	// Auto-clear the live firewall verdict ~7s after it fires (matches the
	// quarantine choreography length so the card leaves with the crimson ring).
	$effect(() => {
		if (!liveFirewallAt) return;
		const t = setTimeout(() => {
			liveFirewallLabel = '';
			liveFirewallAt = 0;
		}, 7000);
		return () => clearTimeout(t);
	});

	onMount(() => {
		// FOSSIL LIGHT deep link — ?t=<ISO8601> opens the field already
		// scrubbed to that instant (applied post-load, once bounds are known).
		pendingChronoIso = new URLSearchParams(window.location.search).get('t');
		loadGraph();
		const cleanupMotion = initReducedMotion();
		return () => {
			stopGlide();
			cleanupMotion?.();
			// Clear the dev/verification global so a demo/receipt remount doesn't
			// retain a disposed LiveBridge → engine → renderer → 200-node graph
			// forever (one orphaned brain per switch otherwise).
			if (typeof window !== 'undefined') {
				const w = window as unknown as { __vestigeLiveBridge?: unknown };
				if (w.__vestigeLiveBridge === liveBridge) delete w.__vestigeLiveBridge;
			}
		};
	});
</script>

<svelte:window onkeydown={onKeydown} />

<!-- Void stage: #05060a — full-bleed (route/takeover) or parent-filling (embedded) -->
<div
	class="{embedded ? 'absolute' : 'fixed'} inset-0 overflow-hidden bg-[#05060a]"
	class:cursor-none={capture}
>
	<!-- Canvas layer (z-index 0) — the living memory field. When the host
	     passes onpick, clicks GPU-pick the memory under the cursor. -->
	<!-- The GPU picker is a custom pointer surface. Pause/resume is separately
	     exposed as a native button below for keyboard and assistive technology. -->
	<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
	<div
		bind:this={canvasLayerEl}
		class="absolute inset-0 z-0 touch-none"
		class:cursor-crosshair={!!onpick && !capture}
		role="application"
		aria-label="Interactive 3D memory field"
		onclick={handleFieldClick}
		onpointerdown={fieldPointerDown}
		onpointermove={fieldPointerMove}
		onpointerup={fieldPointerUp}
		onpointercancel={fieldPointerCancel}
		onwheel={fieldWheel}
	>
		<ObservatoryCanvas
			{demo}
			{seed}
			{freezeFrame}
			{maxDpr}
			onframe={handleFrame}
			onready={handleReady}
		/>
	</div>

	<!-- DOM overlay layer (pointer-events:none) — instruments only.
	     capture hides it all (pure canvas for recording); H toggles;
	     chrome='none' leaves only loading/error (host page owns the chrome). -->
	{#if showHud}
	<div class="absolute inset-0 z-10 pointer-events-none">
		<!-- "This is uniquely YOUR data" proof — the user's most-recalled
		     memories, ranked by REAL receipt frequency. These are the nodes the
		     agent leans on hardest, and (via GCaMP nonlinear summation) the ones
		     that saturate hottest in the field. Two users see different names
		     here because their minds are different — the discipline test as the
		     sales pitch. Only shown when real receipts exist. -->
		{#if live && topRecalled.length > 0}
			<div
				class="absolute top-20 right-4 sm:right-6 max-w-[15rem] flex flex-col gap-1.5
					px-3.5 py-3 rounded-xl border border-[#A8FF5E]/15 bg-[#05060a]/55 backdrop-blur-[2px]"
			>
				<div class="font-mono text-[10px] tracking-[0.16em] text-[#A8FF5E]/70 uppercase">
					{topRecalledSource === 'recalls' ? 'Most recalled · your mind' : 'Strongest memories · your mind'}
				</div>
				{#each topRecalled as m, i (m.id)}
					<div class="flex items-baseline gap-2 font-mono text-[11px]">
						<span class="text-[#E9FFB7]/90 tabular-nums w-4">{i + 1}</span>
						<span class="text-[#d8ded0]/90 truncate flex-1" title={m.label}>{m.label}</span>
						<span class="text-[#A8FF5E]/80 tabular-nums whitespace-nowrap"
							>{m.recalls}{topRecalledSource === 'recalls' ? '×' : '%'}</span
						>
					</div>
				{/each}
			</div>
		{/if}

		<!-- v2.3 living field — LIVE contradiction firewall verdict. Fires only
		     when a real MemorySuppressed / contradiction event quarantines a
		     memory on camera; the crimson quarantine ring plays on the field
		     itself (firewall.wgsl live path). Crimson tone = the immune response. -->
		{#if live && liveFirewallVisible}
			<div
				class="absolute top-20 left-1/2 -translate-x-1/2 pointer-events-none
					flex flex-col items-center gap-1 px-5 py-3 rounded-xl border border-[#ff2d55]/40
					bg-[#1a0508]/85 backdrop-blur-sm text-center enter"
			>
				<div class="font-mono text-[11px] tracking-[0.2em] text-[#ff5c78] uppercase">
					⬤ threat quarantined
				</div>
				<div class="font-mono text-[13px] text-[#ffd0d8] max-w-sm truncate">
					{liveFirewallLabel}
				</div>
				<div class="font-mono text-[10px] tracking-wide text-[#ff5c78]/70">
					memory held in review · Memory PR opened
				</div>
			</div>
		{/if}

		<!-- Motion pause control (WCAG: persistent, for >5s ambient motion). Live
		     field only. Pausing freezes ambient drift; live event pulses persist.
		     Auto-on under prefers-reduced-motion. -->
		{#if !capture}
			<button
				onclick={togglePause}
				class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
					rounded-xl border border-[#22C7DE]/25 bg-[#05060a]/80 backdrop-blur-sm
					font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
					hover:border-[#22C7DE]/50 transition-colors"
				title={paused ? 'Resume field motion' : 'Pause field motion'}
				aria-pressed={paused}
				aria-label={paused ? 'Resume 3D memory field motion' : 'Pause 3D memory field motion'}
			>
				{paused ? '▶ RESUME' : '❚❚ PAUSE'}
			</button>
		{/if}

		<!-- v2.3 living field — forward-projection scrubber. Real per-memory FSRS
		     decay drifts too slowly to watch in one session, so this projects the
		     field N days forward on the SAME true forgetting curve (honest, not
		     faked). Only shown when the backend serves real FSRS state. -->
		{#if live && liveDecayReady}
			<div
				class="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-auto
					flex items-center gap-3 px-3 py-1.5 rounded-full border border-[#91ad8a]/20
					bg-[#05060a]/45 backdrop-blur-[2px] font-mono text-[10px] tracking-[0.14em]"
				class:opacity-100={shuttleReady}
				class:opacity-75={!shuttleReady}
			>
				<span class="text-[#91ad8a]/80 uppercase whitespace-nowrap">Chrono</span>
				<input
					type="range"
					min={chronoMinDays}
					max="365"
					step="0.25"
					bind:value={timeAxisDays}
					oninput={() => (chronoScrubbing = true)}
					onchange={() => (chronoScrubbing = false)}
					onpointerup={() => (chronoScrubbing = false)}
					onpointercancel={() => (chronoScrubbing = false)}
					onblur={() => (chronoScrubbing = false)}
					class="w-36 sm:w-52 accent-[#91ad8a] cursor-ew-resize opacity-75 hover:opacity-100 transition-opacity"
					aria-label="Scrub the memory field through time — back to the oldest memory, forward on the forgetting curve"
					title="Rewind the whole brain to any instant, or project it forward — every memory relit on its real FSRS curve"
				/>
				<span
					class="w-16 text-right tabular-nums"
					class:text-[#b9d9a9]={timeAxisDays >= 0}
					class:text-[#dfc68e]={timeAxisDays < 0}
				>
					{chronoLabel}
				</span>
				{#if timeAxisDays !== 0}
					<button
						onclick={() => (timeAxisDays = 0)}
						class="text-[#d8ded0]/55 hover:text-[#d8ded0] transition-colors"
						title="Return to now"
					>
						now
					</button>
				{/if}
			</div>
		{/if}

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

<PickReceipt pick={lastPick} onclose={() => (lastPick = null)} />
{#if hoverLabel && !capture}
	<div class="pointer-events-none fixed left-4 bottom-24 z-30 font-mono text-[10px] tracking-widest text-[#7ff3e6]/80">
		HOVER {hoverLabel}
	</div>
{/if}

<!-- VISUAL DNA §7: void #05060a is the base (bg utility above) — no exceptions -->

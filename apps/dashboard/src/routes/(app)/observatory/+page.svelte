<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import ObservatoryStage from '$lib/observatory/ObservatoryStage.svelte';
	import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { DEMO_MODES, isDemoMode, type DemoMode } from '$lib/observatory/types';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';
	import { retentionColor } from '$lib/observatory/cognitive-palette';
	import { api } from '$stores/api';
	import type { GraphNode, GraphResponse } from '$types';

	type ObservatoryTextItem = TextLayerItem & { action?: 'demo' | 'exit'; demo?: DemoMode };

	const params = browser ? new URLSearchParams(window.location.search) : new URLSearchParams();
	const demoParam = params.get('demo') ?? 'recall-path';
	let demo = $state<DemoMode>(isDemoMode(demoParam) ? demoParam : 'recall-path');
	const seedValue = params.get('seed') ?? 'vestige-observatory-v1';
	const frameParam = params.get('frame');
	const freezeFrame = frameParam !== null && frameParam !== '' ? Number(frameParam) : null;

	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const GREEN = [...rgb01(RETENTION.recall), 0.76] satisfies [number, number, number, number];
	const OXYGEN = [...rgb01(RETENTION.luciferin), 0.9] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.92] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.82] satisfies [number, number, number, number];
	const GRAPH_LIMIT = 18;

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let graphData: GraphResponse | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);
	let focusedRun: string | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;

	onMount(() => {
		void loadGraph();
	});

	onDestroy(() => {
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		// Field FIRST (renders behind) — the real 200-memory cortex galaxy on top
		// of the recall-path scene. Then the MSDF text HUD on top of that.
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		// The "living nervous system" home: the field stays ALIVE (0.60) but a reading
		// well dims it under the instrument overlay so the nav labels + HUD read. The
		// well covers the left nav column (RECALL/ENGRAM/... at x=-0.91) and the
		// right telemetry (NODES/EDGES at x~0.4), the two text regions.
		//
		// Portrait phones: the HUD collapses to ONE centred vertical stack
		// (buildPortraitItems), so the desktop left-side well would miss it and the
		// bright bloom core would sit right behind the text. On portrait, dim the
		// whole field harder (backdrop, not a blob) and centre a tall well over the
		// stack. Everything derives from the live aspect — desktop is untouched.
		const aspect = portraitAspect();
		if (aspect !== null) {
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.46));
			field.setIntensity(0.32 - 0.1 * portraitness);
			// The centred stack's labels run left-anchored to the right, so bias the
			// well slightly right and widen it to cover the full label extent — the
			// recall wavefront sweeping the right side must stay UNDER the text.
			field.setReadingWell({ x: 0.05, y: 0.05, hw: 0.82, hh: 0.95, floor: 0.04, soft: 0.3 });
		} else {
			field.setIntensity(0.6);
			field.setReadingWell({ x: -0.55, y: 0.1, hw: 0.5, hh: 0.75, floor: 0.08, soft: 0.25 });
		}
		field.setCells(buildFieldCells());
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	/**
	 * The densest cortex: every real memory in the graph (up to 200) becomes a
	 * bioluminescent cell, retention = oxygen hue + inner-ring pull, center memory
	 * selected, suppressed = scar. Swap retention for Math.random and the galaxy's
	 * bright core would legibly scatter — the discipline test.
	 */
	function buildFieldCells() {
		const nodes = graphData?.nodes ?? [];
		const data: FieldDatum[] = nodes.map((n) => ({
			id: n.id,
			score: clamp01(n.retention),
			hue: retentionColor(clamp01(n.retention)),
			energy: 0.4 + 0.6 * clamp01(n.retention),
			selected: !!n.isCenter,
			scar: (n.suppression_count ?? 0) > 0,
			metric2: clamp01(n.retention),
			kind: 'observatory-cell',
			payload: n
		}));
		return layoutGalaxy(data, { maxRadius: 0.95, minCellR: 0.01, maxCellR: 0.038 });
	}

	async function loadGraph() {
		loading = true;
		error = null;
		textPass?.setText(buildTextItems());
		try {
			graphData = await api.graph({ max_nodes: 200, depth: 3, sort: 'connected' });
		} catch (err) {
			graphData = null;
			error = err instanceof Error ? err.message : 'UNKNOWN OBSERVATORY GRAPH ERROR';
		} finally {
			loading = false;
			textPass?.setText(buildTextItems());
			fieldPass?.setCells(buildFieldCells());
			engineRef?.demoClock.reset();
		}
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[\u2014\u2013]/g, '-')
			.replace(/[\u2018\u2019]/g, "'")
			.replace(/[\u201C\u201D]/g, '"')
			.replace(/\u2026/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function demoLabel(mode: DemoMode): string {
		const [head] = mode.split('-');
		return sanitizeAscii(head.toUpperCase());
	}

	function graphMetric(value: number, max: number): number {
		return clamp01(max <= 0 ? 0 : value / max);
	}

	function nodeLine(node: GraphNode): string {
		const label = sanitizeAscii(node.label).replace(/\s+/g, ' ').trim().slice(0, 44);
		const tag = node.tags[0] ? sanitizeAscii(node.tags[0]).slice(0, 14) : sanitizeAscii(node.type);
		return sanitizeAscii(`${label} | ${node.id.slice(0, 8)} | ${tag} | ${Math.round(clamp01(node.retention) * 100)}%`);
	}

	function statusItem(text: string, color = OXYGEN): ObservatoryTextItem {
		return {
			id: 'observatory:status',
			kind: 'observatory-status',
			text: sanitizeAscii(text),
			x: -0.48,
			y: 0.02,
			size: 0.044,
			color,
			depth: 0.76,
			weight: 0.66,
			revealSpan: 32,
			maxWidthEm: 52
		};
	}

	/**
	 * Live viewport aspect from the engine params (canvas px) with a window
	 * fallback — the SAME source TextLayerPass.portraitAdapt reads. Returns the
	 * aspect only when genuinely portrait/narrow (aspect < 0.85); null otherwise.
	 * Nothing is hardcoded per phone width; the whole portrait layout scales off
	 * this one live number, so desktop (>=0.85) is byte-identical.
	 */
	function portraitAspect(): number | null {
		let vw = engineRef?.params[6] || 0;
		let vh = engineRef?.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return null;
		const aspect = vw / vh;
		return aspect < 0.85 ? aspect : null;
	}

	/**
	 * Portrait HUD: the desktop layout is FOUR competing columns (nav at x=-0.91,
	 * receipts at x=-0.67, telemetry at x=0.39, exit at x=0.75). On a phone the
	 * shared portraitAdapt pulls them all toward centre, so they overprint (the
	 * "DENTER"/FIREWALL collisions). Instead, author ONE readable vertical stack:
	 * a centred demo-mode list, a compact telemetry block below it, and EXIT
	 * pinned top — no side-by-side columns, no dense receipt lines running off the
	 * right edge. Everything is authored centred (x~0) so portraitAdapt's x-pull
	 * barely moves it; the y bands are spaced so nothing shares a row.
	 */
	function buildPortraitItems(aspect: number): ObservatoryTextItem[] {
		const items: ObservatoryTextItem[] = [];
		const graph = graphData;
		const nodeDepth = graph ? graphMetric(graph.nodeCount, Math.max(1, graph.nodes.length, 200)) : 0.5;
		const edgeWeight = graph ? graphMetric(graph.edgeCount, Math.max(1, graph.nodeCount * 8)) : 0.5;

		// MSDF labels are LEFT-anchored at their x, so an x of 0 sits the column
		// right-of-centre. Left-shift the whole stack by a portraitness-scaled
		// amount (bigger shift on the narrowest phones) so the longest label
		// ("CENTER c5a42e31-c5f") reads visually centred. portraitAdapt scales x by
		// (1-xPull); pre-divide so the on-screen anchor lands where we want.
		const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.46));
		const anchorX = -0.34 - 0.06 * portraitness;

		// EXIT — top of the stack.
		items.push({
			id: 'observatory:exit',
			kind: 'observatory-exit',
			action: 'exit',
			text: sanitizeAscii('EXIT'),
			x: anchorX,
			y: 0.86,
			size: 0.03,
			color: AMBER,
			depth: 1,
			weight: edgeWeight,
			revealSpan: 10,
			maxWidthEm: 12,
			hitPadX: 0.08,
			hitPadY: 0.05
		});

		// Demo-mode list — the primary interactive column, centred, generously
		// spaced so touch targets don't crowd. This is the ONE focal point.
		const navTop = 0.66;
		const navStep = 0.15;
		DEMO_MODES.forEach((mode, i) => {
			items.push({
				id: `observatory:demo:${mode}`,
				kind: 'observatory-demo',
				action: 'demo',
				demo: mode,
				text: demoLabel(mode),
				x: anchorX,
				y: navTop - i * navStep,
				size: 0.032,
				color: mode === demo ? OXYGEN : GREEN,
				depth: mode === demo ? 1 : nodeDepth,
				weight: mode === demo ? 0.9 : edgeWeight,
				startFrame: i * 2,
				revealSpan: 14,
				maxWidthEm: 18,
				hitPadX: 0.14,
				hitPadY: 0.05
			});
		});

		if (loading) return [...items, statusItem('LOADING MEMORY FIELD...', CYAN)];
		if (error) return [...items, statusItem(`ERROR - ${error}`.slice(0, 60), SCARLET)];
		if (!graph || graph.nodeCount === 0) return [...items, statusItem('NO MEMORIES IN FIELD', GREEN)];

		// Telemetry — a compact block BELOW the nav list (not a right-hand column),
		// so it never shares a row with anything. Short labels only; the long
		// receipt lines that overflowed the right edge on desktop are dropped on
		// portrait (desktop density a phone can't read).
		const telemetry = [
			`NODES ${graph.nodeCount}`,
			`EDGES ${graph.edgeCount}`,
			`DEPTH ${graph.depth}`,
			`CENTER ${graph.center_id.slice(0, 12)}`
		];
		const telTop = navTop - DEMO_MODES.length * navStep - 0.06;
		const telStep = 0.075;
		telemetry.forEach((text, i) => {
			items.push({
				id: `observatory:telemetry:${i}`,
				kind: 'observatory-telemetry',
				text: sanitizeAscii(text),
				x: anchorX,
				y: telTop - i * telStep,
				size: 0.024,
				color: CYAN,
				depth: 0.9,
				weight: edgeWeight,
				startFrame: 8 + i * 2,
				revealSpan: 10,
				maxWidthEm: 24
			});
		});
		return items;
	}

	function buildTextItems(): ObservatoryTextItem[] {
		const aspect = portraitAspect();
		if (aspect !== null) return buildPortraitItems(aspect);

		const items: ObservatoryTextItem[] = [];
		const graph = graphData;
		const nodeDepth = graph ? graphMetric(graph.nodeCount, Math.max(1, graph.nodes.length, 200)) : 0.5;
		const edgeWeight = graph ? graphMetric(graph.edgeCount, Math.max(1, graph.nodeCount * 8)) : 0.5;

		DEMO_MODES.forEach((mode, i) => {
			items.push({
				id: `observatory:demo:${mode}`,
				kind: 'observatory-demo',
				action: 'demo',
				demo: mode,
				text: demoLabel(mode),
				x: -0.91,
				y: 0.76 - i * 0.075,
				size: 0.03,
				color: mode === demo ? OXYGEN : GREEN,
				depth: mode === demo ? 1 : nodeDepth,
				weight: mode === demo ? 0.9 : edgeWeight,
				startFrame: i * 2,
				revealSpan: 14,
				maxWidthEm: 18,
				hitPadX: 0.03,
				hitPadY: 0.026
			});
		});

		items.push({
			id: 'observatory:exit',
			kind: 'observatory-exit',
			action: 'exit',
			text: sanitizeAscii('EXIT'),
			x: 0.75,
			y: 0.82,
			size: 0.03,
			color: AMBER,
			depth: nodeDepth,
			weight: edgeWeight,
			revealSpan: 10,
			maxWidthEm: 12,
			hitPadX: 0.03,
			hitPadY: 0.05
		});

		if (loading) return [...items, statusItem('LOADING MEMORY FIELD...', CYAN)];
		if (error) return [...items, statusItem(`ERROR - ${error}`.slice(0, 76), SCARLET)];
		if (!graph || graph.nodeCount === 0) return [...items, statusItem('NO MEMORIES IN FIELD', GREEN)];

		const telemetry = [
			[`NODES ${graph.nodeCount}`, graphMetric(graph.nodeCount, Math.max(1, graph.nodes.length, 200))],
			[`EDGES ${graph.edgeCount}`, graphMetric(graph.edgeCount, Math.max(1, graph.nodeCount * 8))],
			[`DEPTH ${graph.depth}`, clamp01(graph.depth / 4)],
			[`CENTER ${graph.center_id.slice(0, 12)}`, 1]
		] as const;
		telemetry.forEach(([text, value], i) => {
			items.push({
				id: `observatory:telemetry:${i}`,
				kind: 'observatory-telemetry',
				text: sanitizeAscii(text),
				x: 0.39,
				y: 0.72 - i * 0.045,
				size: 0.022,
				color: CYAN,
				depth: value,
				weight: edgeWeight,
				startFrame: 8 + i * 2,
				revealSpan: 10,
				maxWidthEm: 24
			});
		});

		const sortedNodes = [...graph.nodes]
			.sort((a, b) => Number(b.isCenter) - Number(a.isCenter) || clamp01(b.retention) - clamp01(a.retention))
			.slice(0, GRAPH_LIMIT);
		const top = 0.47;
		const rowStep = 1.06 / Math.max(1, GRAPH_LIMIT - 1);
		sortedNodes.forEach((node, i) => {
			const retention = clamp01(node.retention);
			items.push({
				id: `observatory:node:${node.id}`,
				kind: 'observatory-node',
				text: nodeLine(node),
				x: -0.67,
				y: top - i * rowStep,
				size: 0.024,
				color: node.isCenter ? OXYGEN : CYAN,
				depth: node.isCenter ? 1 : retention,
				weight: retention,
				startFrame: 18 + i * 2,
				revealSpan: 18,
				maxWidthEm: 54
			});
		});
		return items;
	}

	function switchDemo(next: DemoMode) {
		if (next === demo) return;
		demo = next;
		const url = new URL(window.location.href);
		url.searchParams.set('demo', next);
		history.replaceState(history.state, '', url);
		textPass?.setText(buildTextItems());
		engineRef?.demoClock.reset();
	}

	function pointerToNdc(e: PointerEvent | MouseEvent): { x: number; y: number } | null {
		if (!hostEl) return null;
		const rect = hostEl.getBoundingClientRect();
		if (rect.width <= 0 || rect.height <= 0) return null;
		return {
			x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
			y: -(((e.clientY - rect.top) / rect.height) * 2 - 1)
		};
	}

	function writeCursorLens(ndc: { x: number; y: number }) {
		if (!hostEl || !engineRef) return;
		const rect = hostEl.getBoundingClientRect();
		const aspect = Math.max(0.0001, rect.width / Math.max(1, rect.height));
		const raw = { x: ndc.x * Math.max(aspect, 1), y: ndc.y / Math.min(aspect, 1) };
		const prev = cursorSmoothed ?? raw;
		const next = { x: prev.x + (raw.x - prev.x) * 0.35, y: prev.y + (raw.y - prev.y) * 0.35 };
		cursorSmoothed = next;
		engineRef.setCursorPreNdc(next.x, next.y, next.x - prev.x, next.y - prev.y);
	}

	function handlePointerMove(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		writeCursorLens(ndc);
		const hit = textPass?.pickAt(ndc.x, ndc.y) ?? null;
		const nextRun = hit?.kind === 'observatory-demo' || hit?.kind === 'observatory-exit' ? hit.id : null;
		if (nextRun !== focusedRun) {
			focusedRun = nextRun;
			textPass?.setRunDepth(nextRun, 1);
		}
		if (hostEl) hostEl.style.cursor = nextRun ? 'crosshair' : 'default';
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		focusedRun = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		textPass?.setRunDepth(null);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	async function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc || !textPass) return;
		const hit = textPass.pickAt(ndc.x, ndc.y);
		const item = hit?.payload as ObservatoryTextItem | undefined;
		if (item?.action === 'demo' && item.demo) {
			switchDemo(item.demo);
		} else if (item?.action === 'exit') {
			await goto(`${base}/graph`);
		}
	}
</script>

<svelte:head>
	<title>Observatory · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	{#key demo}
		<ObservatoryStage
			{demo}
			seed={seedValue}
			{freezeFrame}
			capture={true}
			showSwitcher={false}
			chrome="none"
			onready={handleReady}
			onexit={() => goto(`${base}/graph`)}
		/>
	{/key}
</div>

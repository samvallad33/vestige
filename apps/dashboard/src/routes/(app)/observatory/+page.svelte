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
	import { saliencePalette, salienceEnergy } from '$lib/observatory/cognitive-palette';
	import { api, type Receipt } from '$stores/api';
	import type { GraphNode, GraphResponse } from '$types';

	type ObservatoryTextItem = TextLayerItem & { action?: 'demo' | 'exit'; demo?: DemoMode };

	const params = browser ? new URLSearchParams(window.location.search) : new URLSearchParams();
	const receiptParam = params.get('receipt');
	const demoParam = receiptParam ? 'recall-path' : (params.get('demo') ?? 'recall-path');
	let demo = $state<DemoMode>(isDemoMode(demoParam) ? demoParam : 'recall-path');
	const seedValue = params.get('seed') ?? 'vestige-observatory-v1';
	const frameParam = params.get('frame');
	const freezeFrame = frameParam !== null && frameParam !== '' ? Number(frameParam) : null;
	// Capture/footage mode HIDES the cursor + chrome for clean hero recordings.
	// It must ONLY be on when actually recording (?capture=1 or ?frame=N) — NOT for
	// normal users, or they get an invisible cursor and cannot navigate. Was
	// hardcoded `true`, which made every real visit un-navigable.
	const captureMode = params.has('capture') || freezeFrame !== null;

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
	let receipt = $state<Receipt | null>(null);
	let receiptError: string | null = $state(null);
	let loading = $state(true);
	let error: string | null = $state(null);
	let focusedRun: string | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;

	// The salience "vote": the ids currently ignited gold-white. Ramps in on entry
	// (the deterministic rescue spine) so the cortex resolves from a grey resting
	// field into "these are the memories the system is spending color on right now."
	let rescuedIds = new Set<string>();
	let voteTimers: ReturnType<typeof setTimeout>[] = [];
	const VOTE_K = 7; // how many memories win the vote (the salient few, not the many)
	const reducedMotion =
		typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	onMount(() => {
		void loadReceiptThenGraph();
	});

	onDestroy(() => {
		voteTimers.forEach(clearTimeout);
		voteTimers = [];
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
		// Keep the field QUIET: only the sharp glowing memory cells on near-black,
		// NOT the cloudy grey-green membrane base-coat (which read as ugly smog with
		// no meaning to a user). A low intensity + a full-frame reading well pushes
		// the membrane cloud down to almost nothing while the additive HDR cells stay
		// bright. Same treatment portrait + desktop — clean dots, no fog.
		const aspect = portraitAspect();
		field.setIntensity(aspect !== null ? 0.16 : 0.2);
		field.setReadingWell({ x: 0, y: 0, hw: 1.15, hh: 1.15, floor: 0.06, soft: 0.4 });
		field.setCells(buildFieldCells(rescuedIds));
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	/**
	 * Real per-memory SALIENCE — the decision-weight the cortex spends color on.
	 * A blend of REAL FSRS state: retention (how retrievable now), stability (how
	 * deeply consolidated, log-compressed since it spans days→years), and recency
	 * of access. The center memory gets a floor. Swap any term for Math.random and
	 * the "vote" would ignite the wrong memories — the discipline test. 0..1.
	 */
	function memorySalience(n: GraphNode): number {
		const retention = clamp01(n.retention);
		// stability is in days and highly skewed (minutes → 500k+); log-compress it
		// into 0..1 so a deeply-consolidated memory reads as salient without a single
		// outlier pinning the whole scale.
		const stabilityDays = Math.max(0, Number.isFinite(n.stability) ? (n.stability as number) : 0);
		const stability = clamp01(Math.log10(1 + stabilityDays) / 5); // 1e5 days → ~1
		// recency: accessed today = 1, decaying over ~30 days.
		let recency = 0.3;
		if (n.lastAccessed) {
			const ageMs = Date.now() - new Date(n.lastAccessed).getTime();
			const ageDays = ageMs / 86_400_000;
			recency = clamp01(1 - ageDays / 30);
		}
		const base = 0.5 * retention + 0.32 * stability + 0.18 * recency;
		return clamp01(n.isCenter ? Math.max(base, 0.9) : base);
	}

	/**
	 * The resting cortex: every real memory becomes a bioluminescent cell whose
	 * COLOR IS EARNED BY SALIENCE (saliencePalette) — the crowd is cold graphite,
	 * only the memories the system currently cares about carry color, and the top
	 * ones blaze. `rescuedIds` is the salience "vote" winner set (the ~K most
	 * salient), forced to gold-white ignition. Layout still pulls the salient
	 * memories to the bright inner core (rank = salience).
	 *
	 * Salience is RANK-NORMALIZED (percentile), not absolute: real corpora cluster
	 * high on retention, so an absolute threshold would light the whole crowd. The
	 * percentile guarantees the grey-mind→earned-color contrast regardless of how
	 * the corpus's retention happens to be distributed — the bottom ~65% stay
	 * graphite, the top decile blazes, on ANY brain.
	 */
	function buildFieldCells(rescuedIds: Set<string> = new Set()) {
		const nodes = graphData?.nodes ?? [];
		// Rank every memory by real salience → percentile in [0,1].
		const ranked = [...nodes]
			.map((n) => ({ id: n.id, s: memorySalience(n) }))
			.sort((a, b) => a.s - b.s);
		const pct = new Map<string, number>();
		const denom = Math.max(1, ranked.length - 1);
		ranked.forEach((r, i) => pct.set(r.id, i / denom));
		const data: FieldDatum[] = nodes.map((n) => {
			const percentile = pct.get(n.id) ?? 0;
			const rescued = rescuedIds.has(n.id);
			// The center memory always reads as salient; everything else earns it.
			const s = n.isCenter ? Math.max(percentile, 0.92) : percentile;
			return {
				id: n.id,
				score: s,
				hue: saliencePalette(s, rescued),
				energy: salienceEnergy(s, rescued),
				selected: !!n.isCenter || rescued,
				scar: (n.suppression_count ?? 0) > 0,
				metric2: clamp01(n.retention),
				kind: 'observatory-cell',
				payload: n
			};
		});
		return layoutGalaxy(data, { maxRadius: 0.95, minCellR: 0.01, maxCellR: 0.04 });
	}

	/** The salience "vote": the K most-salient memories, ignited on entry. Real
	 *  ranking over real salience — the deterministic rescue spine, honest data. */
	function topSalientIds(k: number): Set<string> {
		const nodes = graphData?.nodes ?? [];
		const ranked = [...nodes]
			.map((n) => ({ id: n.id, s: memorySalience(n) }))
			.sort((a, b) => b.s - a.s)
			.slice(0, k)
			.map((r) => r.id);
		return new Set(ranked);
	}

	const backfillEvidence = $derived.by(() => {
		const proof = receipt?.backfill;
		if (!proof) return undefined;
		return {
			failureId: proof.failure_id,
			pathIds: proof.path_ids,
			candidates: proof.candidates.map((candidate) => ({
				memoryId: candidate.memory_id,
				sharedEntities: candidate.shared_entities,
				ageDays: candidate.age_days_before_failure,
				similarityRank: candidate.similarity_rank,
				promoted: candidate.promoted
			}))
		};
	});

	const receiptIds = $derived(
		receipt
			? [
					...new Set([
						...receipt.retrieved,
						...receipt.suppressed.map((entry) => entry.id),
						...(receipt.backfill
							? [
									receipt.backfill.failure_id,
									...(receipt.backfill.path_ids ?? []),
									...receipt.backfill.candidates.map((candidate) => candidate.memory_id)
								]
							: [])
					])
				]
			: []
	);

	async function loadReceiptThenGraph() {
		if (receiptParam) {
			try {
				receipt = await api.receipts.get(receiptParam);
				if (receipt.backfill) demo = 'salience-rescue';
			} catch (err) {
				receiptError = err instanceof Error ? err.message : 'Receipt unavailable';
			}
		}
		await loadGraph();
	}

	async function loadGraph() {
		loading = true;
		error = null;
		textPass?.setText(buildTextItems());
		try {
			const ids = new Set(receiptIds);
			if (ids.size) {
				// The normal graph endpoint chooses an unrelated default center. For a
				// receipt we must resolve each named memory first, then reduce the result
				// back to the exact retrieved/suppressed IDs — never a nearby substitute.
				const scopedGraphs = await Promise.all(
					[...ids].map((center_id) => api.graph({ center_id, max_nodes: 200, depth: 3 }))
				);
				const allNodes = scopedGraphs.flatMap((data) => data.nodes);
				const allEdges = scopedGraphs.flatMap((data) => data.edges);
				const nodes = [...new Map(allNodes.map((node) => [node.id, node])).values()].filter((node) =>
					ids.has(node.id)
				);
				const visibleIds = new Set(nodes.map((node) => node.id));
				const edges = [...new Map(allEdges.map((edge) => [`${edge.source}:${edge.target}`, edge])).values()].filter(
					(edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)
				);
				graphData = {
					...scopedGraphs[0],
					nodes,
					edges,
					center_id: nodes[0]?.id ?? scopedGraphs[0]?.center_id ?? '',
					nodeCount: nodes.length,
					edgeCount: edges.length
				};
			} else {
				graphData = await api.graph({ max_nodes: 200, depth: 3, sort: 'connected' });
			}
		} catch (err) {
			graphData = null;
			error = err instanceof Error ? err.message : 'UNKNOWN OBSERVATORY GRAPH ERROR';
		} finally {
			loading = false;
			textPass?.setText(buildTextItems());
			igniteVote();
			engineRef?.demoClock.reset();
		}
	}

	/**
	 * The salience vote plays out on entry: the grey resting cortex resolves, then
	 * the K most-salient memories ignite gold-white one after another in a ripple —
	 * the deterministic rescue spine, driven by REAL salience ranking. Under
	 * reduced-motion or ?frame= capture the whole vote is applied at once (no
	 * animation, identical final pixels), preserving determinism.
	 */
	function igniteVote() {
		voteTimers.forEach(clearTimeout);
		voteTimers = [];
		rescuedIds = new Set();
		const winners = [...topSalientIds(VOTE_K)];
		if (!winners.length) {
			fieldPass?.setCells(buildFieldCells(rescuedIds));
			return;
		}
		// Deterministic capture / reduced motion: no ripple, final state immediately.
		if (freezeFrame !== null || reducedMotion) {
			rescuedIds = new Set(winners);
			fieldPass?.setCells(buildFieldCells(rescuedIds));
			return;
		}
		// Show the grey resting cortex first (the crowd, color earned but no vote yet),
		// then ripple the winners in so the eye watches the decision happen.
		fieldPass?.setCells(buildFieldCells(rescuedIds));
		const rippleStart = 420; // let the resting field read for a beat
		const rippleStep = 130; // one winner ignites every ~130ms
		winners.forEach((id, i) => {
			voteTimers.push(
				setTimeout(
					() => {
						rescuedIds.add(id);
						fieldPass?.setCells(buildFieldCells(rescuedIds));
					},
					rippleStart + i * rippleStep
				)
			);
		});
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
		// The DOM control layer (.obs-ui) now owns ALL text + controls — title,
		// stats, the 5 clickable demo cards, EXIT. Rendering the old in-canvas MSDF
		// labels (EXIT / RECALL / telemetry / node list) on top of the field just
		// leaves illegible ghosts bleeding through behind the DOM. Return nothing so
		// the canvas is a pure field; interaction lives in the DOM buttons.
		return [];
	}

	function buildTextItemsLegacy(): ObservatoryTextItem[] {
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

	// ── DOM control layer ────────────────────────────────────────────────
	// The floating MSDF words are beautiful but a first-time user cannot tell
	// they are clickable, cannot see a cursor over them, and has no idea what the
	// page is. A real DOM overlay (title + one-line promise + 5 clickable demo
	// cards) sits ON TOP of the living WebGPU field so the page is instantly
	// legible and operable with a normal cursor — the field stays the backdrop.
	const DEMO_CARDS: { mode: DemoMode; label: string; blurb: string }[] = [
		{ mode: 'recall-path', label: 'Recall', blurb: 'Watch a memory get retrieved — the path lights up.' },
		{ mode: 'engram-birth', label: 'Engram', blurb: 'A new memory forms and wires into the field.' },
		{ mode: 'salience-rescue', label: 'Salience', blurb: 'The few memories that matter ignite gold.' },
		{ mode: 'forgetting-horizon', label: 'Forgetting', blurb: 'FSRS decay pulls weak memories toward the dark.' },
		{ mode: 'firewall', label: 'Firewall', blurb: 'A contradiction is caught and quarantined.' }
	];
	const demoCards = DEMO_CARDS.filter((c) => DEMO_MODES.includes(c.mode));
	function centerShort(): string {
		return graphData?.center_id?.slice(0, 8) ?? '—';
	}

	function pickDemo(mode: DemoMode) {
		if (receipt) return;
		switchDemo(mode);
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
		if (!ndc) return;
		const hit = textPass?.pickAt(ndc.x, ndc.y);
		const item = hit?.payload as ObservatoryTextItem | undefined;
		if (item?.action === 'demo' && item.demo) {
			switchDemo(item.demo);
			return;
		}
		if (item?.action === 'exit') {
			await goto(`${base}/graph`);
			return;
		}
		// Fall through to the cortex itself: every salience cell is a real
		// memory — clicking one cuts to its record in the library
		// (click-as-incision: the home canvas is an instrument, not a poster).
		const cell = fieldPass?.pickAt(ndc.x, ndc.y);
		if (cell && typeof cell.id === 'string') {
			await goto(`${base}/memories?memory=${encodeURIComponent(cell.id)}`);
		}
	}
</script>

<svelte:head>
	<title>Observatory · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	{#key `${demo}:${receipt?.receipt_id ?? 'all'}`}
		<ObservatoryStage
			{demo}
			seed={seedValue}
			{freezeFrame}
			capture={captureMode}
			showSwitcher={false}
			chrome="none"
			live
			focusIds={receiptIds}
			{backfillEvidence}
			onready={handleReady}
			onexit={() => goto(`${base}/graph`)}
		/>
	{/key}
</div>

<!--
	DOM control layer over the living WebGPU field. Legible, cursor-visible,
	obviously clickable. The field (behind) stays the alive backdrop. Every
	number is real /api/graph data; each card drives the same switchDemo() the
	canvas used, so the interaction is honest and now discoverable.
-->
<div class="obs-ui">
	<header class="obs-head">
		<h1 class="obs-title">{receipt?.backfill ? 'Backfill Replay' : receipt ? 'Memory Replay' : 'Cognitive Observatory'}</h1>
		<p class="obs-sub">
			{#if receipt?.backfill}
				This field replays only the recorded Backfill candidate evidence in receipt <code>{receipt.receipt_id}</code>.
			{:else if receipt}
				This field contains only memories named by receipt <code>{receipt.receipt_id}</code>.
			{:else}
				Your agent's live memory field. Play a cognitive moment and watch the mind react.
			{/if}
		</p>
		<div class="obs-stats">
			{#if receipt}
				<span class="obs-stat obs-proof"><b>Proven:</b> retrieved in this run</span>
				<span class="obs-stat obs-attributed"><b>Attributed:</b> likely influence</span>
			{:else if receiptError}
				<span class="obs-stat obs-err"><b>!</b> {receiptError}</span>
			{/if}
			{#if loading}
				<span class="obs-stat"><b>…</b> loading field</span>
			{:else if error}
				<span class="obs-stat obs-err"><b>!</b> {error}</span>
			{:else if graphData}
				<span class="obs-stat"><b>{graphData.nodeCount.toLocaleString()}</b> memories</span>
				<span class="obs-stat"><b>{graphData.edgeCount.toLocaleString()}</b> connections</span>
				<span class="obs-stat"><b>{centerShort()}</b> center</span>
			{/if}
		</div>
	</header>

	<nav class="obs-demos" aria-label={receipt ? 'Receipt evidence' : 'Cognitive moments'}>
		{#if receipt}
			<span class="obs-demos-label">Receipt evidence only</span>
			<span class="obs-card is-active receipt-scope">
				<span class="obs-card-label">{receipt.retrieved.length} retrieved · {receipt.suppressed.length} suppressed</span>
				<span class="obs-card-blurb">This proves memory retrieval. It does not claim an answer changed.</span>
			</span>
		{:else}
			<span class="obs-demos-label">Play a moment</span>
			{#each demoCards as card (card.mode)}
				<button
					type="button"
					class="obs-card {card.mode === demo ? 'is-active' : ''}"
					aria-pressed={card.mode === demo}
					onclick={() => pickDemo(card.mode)}
				>
					<span class="obs-card-label">{card.label}</span>
					<span class="obs-card-blurb">{card.blurb}</span>
				</button>
			{/each}
		{/if}
		<a class="obs-exit" href="{base}/graph">Open full graph →</a>
	</nav>
</div>

<style>
	/* The DOM control layer. Sits above the WebGPU canvas (z 0) but its container
	   is pointer-events:none so the field still receives hover/clicks in the gaps;
	   only the actual controls capture the pointer. Normal cursor everywhere. */
	.obs-ui {
		position: fixed;
		inset: 0;
		z-index: 10;
		pointer-events: none;
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		padding: 1.5rem clamp(1rem, 3vw, 2.5rem);
		cursor: auto;
	}
	.obs-ui > * {
		pointer-events: auto;
	}

	.obs-head {
		max-width: 34rem;
	}
	.obs-title {
		margin: 0;
		font-size: clamp(1.6rem, 3.4vw, 2.4rem);
		font-weight: 700;
		letter-spacing: 0.01em;
		color: #eafffb;
		text-shadow: 0 0 26px rgba(34, 199, 222, 0.35);
	}
	.obs-sub {
		margin: 0.35rem 0 0;
		font-size: clamp(0.85rem, 1.4vw, 1rem);
		line-height: 1.5;
		color: rgba(190, 224, 226, 0.82);
	}
	.obs-stats {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1.1rem;
		margin-top: 0.85rem;
	}
	.obs-stat {
		font-size: 0.78rem;
		letter-spacing: 0.02em;
		color: rgba(150, 190, 195, 0.7);
	}
	.obs-stat b {
		color: #7ff3e6;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
	}
	.obs-err b {
		color: #ff6b5e;
	}

	.obs-demos {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: min(24rem, 84vw);
	}
	.obs-demos-label {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.18em;
		color: rgba(140, 175, 180, 0.6);
		margin-bottom: 0.15rem;
	}
	.obs-card {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		text-align: left;
		padding: 0.7rem 0.9rem;
		border-radius: 0.7rem;
		border: 1px solid rgba(120, 160, 165, 0.22);
		background: rgba(6, 16, 20, 0.55);
		-webkit-backdrop-filter: blur(8px);
		backdrop-filter: blur(8px);
		color: #dff6f4;
		cursor: pointer;
		transition:
			border-color 0.16s ease,
			background 0.16s ease,
			transform 0.16s ease;
	}
	.obs-card:hover {
		border-color: rgba(34, 199, 222, 0.55);
		background: rgba(10, 26, 30, 0.72);
		transform: translateX(3px);
	}
	.obs-card:focus-visible {
		outline: 2px solid rgba(34, 199, 222, 0.7);
		outline-offset: 2px;
	}
	.obs-card.is-active {
		border-color: rgba(34, 199, 222, 0.85);
		background: rgba(16, 40, 46, 0.8);
		box-shadow: 0 0 22px -6px rgba(34, 199, 222, 0.6);
	}
	.obs-card-label {
		font-size: 0.98rem;
		font-weight: 700;
		letter-spacing: 0.03em;
		color: #9ff8ec;
	}
	.obs-card.is-active .obs-card-label {
		color: #eafffb;
	}
	.obs-card-blurb {
		font-size: 0.76rem;
		line-height: 1.35;
		color: rgba(180, 210, 212, 0.72);
	}
	.obs-exit {
		margin-top: 0.35rem;
		align-self: flex-start;
		font-size: 0.8rem;
		letter-spacing: 0.02em;
		color: #7ff3e6;
		text-decoration: none;
		cursor: pointer;
	}
	.obs-exit:hover {
		text-decoration: underline;
	}

	@media (max-aspect-ratio: 85/100) {
		.obs-ui {
			padding: 1rem 1rem 5.5rem;
		}
		.obs-demos {
			width: 100%;
		}
	}
</style>

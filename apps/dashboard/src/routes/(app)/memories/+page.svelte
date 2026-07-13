<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import { api } from '$stores/api';
	import type { Memory } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01, retentionColor } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { LivingFieldPass } from '$lib/observatory/field/living-field-pass';
	import { layoutGalaxy, type FieldDatum } from '$lib/observatory/field/cell-layout';

	type MemoryTextItem = TextLayerItem & { memoryId?: string };

	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.62] satisfies [number, number, number, number];
	const MEMORY_LIMIT = 40;
	const ROW_LIMIT = 36;
	const MIN_VISIBLE_DEPTH = 0.62;
	// The MSDF reveal is frame-driven per glyph: the shared text layer packs
	// ageFrame = startFrame + globalGlyphIndex * 2. Across many long rows that
	// global index reaches the thousands, so late glyphs would never reveal
	// before the demo clock wraps. Anchor the reveal far in the past so the
	// complete memory field is legible immediately.
	const REVEAL_ANCHOR = -100000;

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let textPass: TextLayerPass | null = null;
	let fieldPass: LivingFieldPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	// Local suppressed-id set — Memory has no suppression field, so we track the
	// immune state we just applied so the field cell can scar immediately.
	let suppressedIds = $state<Set<string>>(new Set());
	let memories: Memory[] = $state([]);
	let total = $state(0);
	let loading = $state(true);
	let error: string | null = $state(null);
	let activeRun: string | null = null;
	// Tracked when the user picks a memory from the canvas. Selection only —
	// no API call. Mutations are gated to explicit modifier keys (shift/alt)
	// whose affordances are mirrored by the on-canvas hint line.
	let selectedMemoryId: string | null = $state(null);
	let stopReducedMotion: (() => void) | null = null;

	onMount(() => {
		void loadMemories();
	});

	onDestroy(() => {
		stopReducedMotion?.();
		textPass?.dispose();
		fieldPass?.dispose();
		textPass = null;
		fieldPass = null;
		engineRef = null;
	});

	// This organ mounts ObservatoryCanvas directly (not RouteStage), so it must
	// wire prefers-reduced-motion itself: a reduce-motion user gets a frozen field
	// (the engine still renders one static frame) instead of continuous orbit.
	function initReducedMotion(engine: ObservatoryEngine): () => void {
		if (typeof window === 'undefined') return () => {};
		const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
		engine.setPaused(mq.matches);
		const onChange = (e: MediaQueryListEvent) => engine.setPaused(e.matches);
		mq.addEventListener('change', onChange);
		return () => mq.removeEventListener('change', onChange);
	}

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		stopReducedMotion = initReducedMotion(engine);
		// Engram galaxy FIRST (behind the parallax MSDF text): every real memory a
		// glowing cell, retention = oxygen. The cursor-parallax text field rides on top.
		const field = new LivingFieldPass(engine);
		fieldPass = field;
		field.setIntensity(0.8);
		field.setCells(buildFieldCells());
		engine.addPass(field);
		const pass = new TextLayerPass(engine);
		textPass = pass;
		await pass.init();
		pass.setText(buildTextItems());
		engine.addPass(pass);
		engine.demoClock.reset();
	}

	function buildFieldCells() {
		const data: FieldDatum[] = memories.map((m) => {
			const retention = clamp01(m.retentionStrength);
			return {
				id: m.id,
				score: clamp01(m.retrievalStrength ?? retention),
				hue: retentionColor(retention),
				energy: 0.4 + 0.6 * clamp01(m.retrievalStrength ?? retention),
				metric2: retention,
				scar: suppressedIds.has(m.id),
				kind: 'memory',
				payload: m
			} satisfies FieldDatum;
		});
		return layoutGalaxy(data, { maxRadius: 0.95, minCellR: 0.014, maxCellR: 0.05 });
	}

	async function loadMemories() {
		loading = true;
		error = null;
		textPass?.setText(buildTextItems());
		try {
			const res = await api.memories.list({ limit: String(MEMORY_LIMIT) });
			memories = res.memories;
			total = res.total;
		} catch (err) {
			memories = [];
			total = 0;
			error = err instanceof Error ? err.message : 'UNKNOWN MEMORY FETCH ERROR';
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

	// Live viewport aspect (canvas px) — same source portraitAdapt reads, never a
	// hardcoded phone width. Falls back to the window before frame 0, then 1.
	function viewportAspect(): number {
		const vw = engineRef?.params[6] || 0;
		const vh = engineRef?.params[7] || 0;
		if (vw > 0 && vh > 0) return vw / vh;
		if (typeof window !== 'undefined' && window.innerHeight > 0) {
			return window.innerWidth / window.innerHeight;
		}
		return 1;
	}

	// Portrait rows must READ, not dump. A phone can only fit ~10-12 rows with the
	// generous spacing the taste bar demands, and long lines would shrink to
	// illegible to fit width. So on portrait we shorten each line (drop the id
	// column, tighten the snippet) so it never runs edge-to-edge, and the caller
	// caps the row count. Everything is gated on the LIVE aspect — desktop keeps
	// the full id column and long snippet, byte-identical.
	// Trim a snippet to a cap, breaking on the last word boundary near the cap so a
	// portrait row never ends mid-token ("Jul 202"); falls back to a hard slice for
	// a single unbroken token.
	function trimSnippet(text: string, cap: number): string {
		const s = sanitizeAscii(text).replace(/\s+/g, ' ').trim();
		if (s.length <= cap) return s;
		const hard = s.slice(0, cap);
		const lastSpace = hard.lastIndexOf(' ');
		return lastSpace > cap * 0.6 ? hard.slice(0, lastSpace) : hard;
	}

	function memoryLine(memory: Memory, portrait: boolean): string {
		const pct = `${Math.round(memory.retentionStrength * 100)}%`;
		if (portrait) return sanitizeAscii(`${trimSnippet(memory.content, 28)}  ${pct}`);
		return sanitizeAscii(`${trimSnippet(memory.content, 52)} | ${memory.id.slice(0, 8)} | ${pct}`);
	}

	function clamp01(value: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
	}

	function statusItem(text: string, color = MUTED): MemoryTextItem {
		return {
			id: 'memories:status',
			kind: 'memory-status',
			text: sanitizeAscii(text),
			x: -0.58,
			y: 0.02,
			size: 0.044,
			color,
			depth: 0.78,
			weight: 0.62,
			revealSpan: 32,
			maxWidthEm: 50
		};
	}

	function buildTextItems(): MemoryTextItem[] {
		if (loading) return [statusItem('LOADING MEMORY FIELD...', CYAN)];
		if (error) return [statusItem(`ERROR - ${error}`.slice(0, 72), SCARLET)];
		if (memories.length === 0) return [statusItem('EMPTY MEMORY FIELD', MUTED)];

		// Persistent on-canvas affordance: a plain click selects; shift and alt are
		// the only modifiers that mutate. Anchored low so it never collides with the
		// row band above, and shaped as a "memory-hint" kind so it is NEVER picked as
		// a memory row by the text layer.
		const hint: MemoryTextItem = {
			id: 'memories:hint',
			kind: 'memory-hint',
			text: 'click: select  ·  shift: suppress  ·  alt: unsuppress',
			x: -0.82,
			y: -0.9,
			size: 0.022,
			color: MUTED,
			depth: 0.7,
			weight: 0.5,
			startFrame: REVEAL_ANCHOR,
			revealSpan: 24,
			maxWidthEm: 50
		};

		const aspect = viewportAspect();
		const portrait = aspect < 0.85;

		if (portrait) {
			// Phone plan: ONE focal header + a short, well-spaced column that fits the
			// screen band with real negative space. Row count and spacing derive from
			// the live aspect (taller/narrower → fewer rows), never a fixed phone
			// number. portraitAdapt maps authored-y straight to screen-y (its inv
			// reclaim and the shader's aspect crush cancel), so we author in screen NDC.
			const portraitness = clamp01((0.85 - aspect) / (0.85 - 0.42));
			// 12 rows at the wide-portrait edge, down to 9 on the tallest phones —
			// enough to be a real list, few enough to breathe and never collide.
			const rowCount = Math.max(9, Math.round(12 - 3 * portraitness));
			const rows = memories.slice(0, rowCount);
			const headerY = 0.8;
			const top = 0.62;
			const bottom = -0.72;
			const rowStep = rows.length > 1 ? (top - bottom) / (rows.length - 1) : 0;
			const header: MemoryTextItem = {
				id: 'memories:header',
				kind: 'memory-header',
				text: `MEMORY FIELD  ${total} TRACES`,
				x: -0.82,
				y: headerY,
				size: 0.03,
				color: MUTED,
				depth: 0.85,
				weight: 0.6,
				startFrame: REVEAL_ANCHOR,
				revealSpan: 24,
				maxWidthEm: 30
			};
			return [
				hint,
				header,
				...rows.map((memory, i) => {
					const retrieval = clamp01(memory.retrievalStrength);
					const retention = clamp01(memory.retentionStrength);
					return {
						id: `mem:${memory.id}`,
						kind: 'memory',
						memoryId: memory.id,
						text: memoryLine(memory, true),
						x: -0.82,
						y: top - i * rowStep,
						size: 0.03,
						color: CYAN,
						depth: Math.max(MIN_VISIBLE_DEPTH, retrieval),
						weight: retention,
						startFrame: REVEAL_ANCHOR + i * 2,
						revealSpan: 20,
						maxWidthEm: 34,
						hitPadX: 0.04,
						hitPadY: 0.03
					} satisfies MemoryTextItem;
				})
			];
		}

		const rows = memories.slice(0, ROW_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		return [
			hint,
			...rows.map((memory, i) => {
				const retrieval = clamp01(memory.retrievalStrength);
				const retention = clamp01(memory.retentionStrength);
				return {
					id: `mem:${memory.id}`,
					kind: 'memory',
					memoryId: memory.id,
					text: memoryLine(memory, false),
					x: -0.88,
					y: top - i * rowStep,
					size: 0.026,
					color: CYAN,
					depth: Math.max(MIN_VISIBLE_DEPTH, retrieval),
					weight: retention,
					startFrame: REVEAL_ANCHOR + i * 2,
					revealSpan: 20,
					maxWidthEm: 46,
					hitPadX: 0.03,
					hitPadY: 0.015
				};
			})
		];
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
		const nextRun = hit?.kind === 'memory' ? hit.id : null;
		if (nextRun !== activeRun) {
			activeRun = nextRun;
			textPass?.setRunDepth(nextRun, 1);
		}
		if (hostEl) hostEl.style.cursor = nextRun ? 'crosshair' : 'default';
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		activeRun = null;
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		textPass?.setRunDepth(null);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	async function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc || !textPass) return;
		const hit = textPass.pickAt(ndc.x, ndc.y);
		if (hit?.kind !== 'memory') return;
		const item = hit.payload as MemoryTextItem;
		if (!item.memoryId) return;
		// Read-intent click must SELECT only — no API call. Mutations are
		// gated to explicit modifier keys whose affordances are mirrored by
		// the on-canvas hint line ("shift: suppress  ·  alt: unsuppress").
		//   plain click  -> select (no mutation)
		//   shift-click  -> suppress (macrophage engulfs; cell scars)
		//   alt-click    -> unsuppress (labile release)
		selectedMemoryId = item.memoryId;
		if (e.shiftKey || e.altKey) {
			try {
				if (e.shiftKey) {
					await api.memories.suppress(item.memoryId, 'suppressed from memories field');
					suppressedIds = new Set(suppressedIds).add(item.memoryId);
				} else if (e.altKey) {
					// unsuppress COMPOUNDS down by one — a memory suppressed twice is still
					// suppressed after one unsuppress. Only clear the scar when the backend
					// says it's fully released (stillSuppressed=false), else the cell would
					// lie (render healthy while retrieval is still penalized).
					const res = await api.memories.unsuppress(item.memoryId);
					if (!res.stillSuppressed) {
						const next = new Set(suppressedIds);
						next.delete(item.memoryId);
						suppressedIds = next;
					}
				}
				error = null;
				textPass.setText(buildTextItems());
				fieldPass?.setCells(buildFieldCells());
			} catch (err) {
				error = err instanceof Error ? err.message : 'UNKNOWN MEMORY ACTION ERROR';
				textPass.setText(buildTextItems());
			}
		}
	}
</script>

<svelte:head>
	<title>Memories · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div bind:this={hostEl} class="fixed inset-0 bg-[#020307]" onpointerdown={handlePointerDown} onpointermove={handlePointerMove} onpointerleave={handlePointerLeave}>
	<ObservatoryCanvas demo="recall-path" seed={`real-memory-field:${total}`} onready={handleReady} />
</div>

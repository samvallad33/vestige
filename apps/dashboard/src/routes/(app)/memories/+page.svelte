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

	onMount(() => {
		void loadMemories();
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
		// Engram galaxy FIRST (behind the parallax MSDF text): every real memory a
		// glowing cell, retention = oxygen. The cursor-parallax text field rides on top.
		const field = new LivingFieldPass(engine);
		fieldPass = field;
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

	function memoryLine(memory: Memory): string {
		const snippet = sanitizeAscii(memory.content).replace(/\s+/g, ' ').trim().slice(0, 52);
		return sanitizeAscii(
			`${snippet} | ${memory.id.slice(0, 8)} | ${Math.round(memory.retentionStrength * 100)}%`
		);
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

		const rows = memories.slice(0, ROW_LIMIT);
		const top = 0.72;
		const rowStep = 1.5 / Math.max(1, ROW_LIMIT - 1);
		return rows.map((memory, i) => {
			const retrieval = clamp01(memory.retrievalStrength);
			const retention = clamp01(memory.retentionStrength);
			return {
				id: `mem:${memory.id}`,
				kind: 'memory',
				memoryId: memory.id,
				text: memoryLine(memory),
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
		});
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
		// Immune actions surfaced right on the field (backend demoability):
		//   shift-click  -> suppress (macrophage engulfs; cell scars)
		//   alt-click    -> unsuppress (labile release)
		//   plain click  -> promote (retention boost)
		try {
			if (e.shiftKey) {
				await api.memories.suppress(item.memoryId, 'suppressed from memories field');
				suppressedIds = new Set(suppressedIds).add(item.memoryId);
			} else if (e.altKey) {
				await api.memories.unsuppress(item.memoryId);
				const next = new Set(suppressedIds);
				next.delete(item.memoryId);
				suppressedIds = next;
			} else {
				const promoted = await api.memories.promote(item.memoryId);
				memories = memories.map((memory) =>
					memory.id === promoted.id
						? { ...memory, retentionStrength: promoted.retentionStrength }
						: memory
				);
			}
			error = null;
			textPass.setText(buildTextItems());
			fieldPass?.setCells(buildFieldCells());
		} catch (err) {
			error = err instanceof Error ? err.message : 'UNKNOWN MEMORY ACTION ERROR';
			textPass.setText(buildTextItems());
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

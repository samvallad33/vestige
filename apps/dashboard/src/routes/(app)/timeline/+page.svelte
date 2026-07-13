<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$stores/api';
	import type { MemoryAuditEvent, TimelineDay } from '$types';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import RouteStage, { type RouteFramePass, type RoutePick } from '$lib/observatory/RouteStage.svelte';
	import type { RouteSceneModel } from '$lib/observatory/route-scene';
	import { createTimelinePasses } from '$lib/observatory/timeline/timeline-pass';
	import {
		normalizeTimelineScene,
		type TimelineCell,
		type TimelineRing,
		type TimelineScene
	} from '$lib/observatory/timeline/timeline-scene';

	// ── palette (Causal Bioluminescent Cortex — cyan accent, indigo = bitemporal) ──
	const CYAN = [...rgb01('#22C7DE'), 1] satisfies [number, number, number, number];
	const INDIGO = [...rgb01('#7C6CFF'), 0.95] satisfies [number, number, number, number];
	const OXYGEN = [...rgb01('#A8FF5E'), 0.92] satisfies [number, number, number, number];
	const SCARLET = [...rgb01('#FF3B30'), 0.92] satisfies [number, number, number, number];
	const MUTED = [...rgb01('#29F2A9'), 0.6] satisfies [number, number, number, number];
	// Portrait-only: the small captions sit over the (dimmed) ring band, so a 0.6-alpha
	// MUTED green washes out. A full-alpha brighter mint lifts them clear of the field.
	const LABEL_HI = [...rgb01('#8AF7D0'), 1] satisfies [number, number, number, number];

	const RANGE_CYCLE = [7, 14, 30, 90, 365] as const;

	// ── real backend state ──
	let timeline: TimelineDay[] = $state([]);
	let loading = $state(true);
	let error: string | null = $state(null);
	let days = $state(14);
	let selectedCell: TimelineCell | null = $state(null);
	let selectedRing: TimelineRing | null = $state(null);
	let auditLoading = $state(false);
	let audits = $state<Record<string, MemoryAuditEvent[]>>({});

	// text pass handle (owned by the route pass factory below)
	let textPass: TextLayerPass | null = null;
	let focusedRun: string | null = null;
	// engine handle captured in the pass factory so buildTextItems can read the live
	// viewport aspect (params[6]/[7]) for portrait-only vital-sign stacking.
	let engineHandle: ObservatoryEngine | null = null;

	onMount(() => loadTimeline());

	async function loadTimeline() {
		loading = true;
		error = null;
		try {
			const res = await api.timeline(days, 500);
			timeline = res.timeline;
		} catch (err) {
			timeline = [];
			error = err instanceof Error ? err.message : 'Failed to load timeline';
		} finally {
			loading = false;
		}
	}

	function cycleRange() {
		const i = RANGE_CYCLE.indexOf(days as (typeof RANGE_CYCLE)[number]);
		days = RANGE_CYCLE[(i + 1) % RANGE_CYCLE.length];
		selectedCell = null;
		selectedRing = null;
		loadTimeline();
	}

	// ── derived real metrics ──
	let totalMemories = $derived(timeline.reduce((sum, d) => sum + d.count, 0));
	let rewriteCount = $derived(
		timeline.reduce(
			(sum, day) =>
				sum + day.memories.filter((m) => m.updatedAt && m.createdAt && m.updatedAt !== m.createdAt).length,
			0
		)
	);
	let avgRetention = $derived.by(() => {
		const memories = timeline.flatMap((day) => day.memories);
		return memories.length
			? memories.reduce((sum, m) => sum + (m.retentionStrength ?? 0), 0) / memories.length
			: 0;
	});
	let timelineScene: TimelineScene = $derived(
		normalizeTimelineScene({ days: timeline, totalMemories, audits })
	);

	async function fetchAudit(memoryId: string) {
		if (audits[memoryId]) return;
		auditLoading = true;
		try {
			const res = await api.memoryAudit(memoryId, 100);
			audits = { ...audits, [memoryId]: res.events };
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to load memory audit';
		} finally {
			auditLoading = false;
		}
	}

	function memoryByCell(cell: TimelineCell | null) {
		if (!cell) return null;
		return timeline.flatMap((day) => day.memories).find((m) => m.id === cell.memoryId) ?? null;
	}

	let selectedMemory = $derived(memoryByCell(selectedCell));
	let selectedAudit = $derived.by(() => {
		const cell = selectedCell;
		return cell ? (audits[cell.memoryId] ?? []) : [];
	});

	// Re-render the MSDF HUD whenever any real signal changes (Svelte 5 rune effect).
	$effect(() => {
		// touch reactive deps so the effect re-runs on change
		void [timeline, loading, error, days, selectedCell, selectedRing, selectedMemory, selectedAudit, auditLoading];
		textPass?.setText(buildTextItems());
	});

	// ── ASCII hygiene (atlas is 0x20–0x7E only) ──
	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function clamp01(v: number): number {
		return Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0.5));
	}

	// Live viewport aspect, same source (and window fallback) TextLayerPass.portraitAdapt
	// uses. Portrait (aspect < 0.85) needs its own vital-sign stacking so the big
	// numbers don't collide with their captions and the receipt doesn't overrun the
	// HUD column. NOTHING here is hardcoded per phone width — it scales with aspect.
	function viewportAspect(): number {
		const vw = engineHandle?.params[6] || 0;
		const vh = engineHandle?.params[7] || 0;
		if (vw > 0 && vh > 0) return vw / vh;
		if (typeof window !== 'undefined' && window.innerHeight > 0) return window.innerWidth / window.innerHeight;
		return 1.6;
	}

	function line(
		id: string,
		kind: string,
		text: string,
		x: number,
		y: number,
		size: number,
		color: [number, number, number, number],
		extra: Partial<TextLayerItem> = {}
	): TextLayerItem {
		return {
			id,
			kind,
			text: sanitizeAscii(text),
			x,
			y,
			size,
			color,
			depth: 0.7,
			weight: 0.6,
			revealSpan: 18,
			maxWidthEm: 60,
			...extra
		};
	}

	// The entire readable surface is MSDF — HUD (top-left), receipt (right), status.
	function buildTextItems(): TextLayerItem[] {
		const items: TextLayerItem[] = [];

		// ── HUD: title + real vital signs (top-left) ──
		items.push(line('tl:title', 'tl-title', 'BITEMPORAL GROWTH RINGS', -0.94, 0.9, 0.04, CYAN, { depth: 1, weight: 0.9 }));
		items.push(
			line(
				'tl:range',
				'tl-range',
				`RANGE ${days}D  [click to cycle]`,
				-0.94,
				0.82,
				0.024,
				CYAN,
				{ depth: 0.85, hitPadX: 0.03, hitPadY: 0.05 }
			)
		);

		if (loading) {
			items.push(line('tl:status', 'tl-status', 'WEAVING VALID-TIME RINGS...', -0.3, 0.02, 0.04, CYAN, { revealSpan: 40 }));
			return items;
		}
		if (error) {
			items.push(line('tl:status', 'tl-status', `ERROR - ${error}`.slice(0, 70), -0.5, 0.02, 0.032, SCARLET, { revealSpan: 12 }));
			return items;
		}
		if (timeline.length === 0) {
			items.push(line('tl:status', 'tl-status', 'NO MEMORY GROWTH RINGS IN THIS WINDOW', -0.5, 0.02, 0.03, MUTED, { revealSpan: 24 }));
			return items;
		}

		// vital signs stacked under the title — each a real derived metric
		const vitals: Array<[string, string, [number, number, number, number]]> = [
			[`${totalMemories}`, 'MEMORIES IN VALID-TIME RINGS', CYAN],
			[`${rewriteCount}`, 'TRANSACTION-TIME SHADOWS', INDIGO],
			[`${timeline.length}`, 'CALENDAR SLICES', CYAN],
			[`${Math.round(avgRetention * 100)}%`, 'AVERAGE RETENTION OXYGEN', OXYGEN]
		];

		// Portrait: the big number (size 0.05, boosted ~2.7x by portraitAdapt) grows
		// TALLER than the authored 0.05 num->label gap, so it lands on top of its own
		// caption; and the right-hand receipt column (x=0.3) gets center-pulled into
		// the HUD column and collides. So in portrait we author a single left column:
		// smaller numbers with a wider caption gap, then the receipt STACKED below the
		// vitals instead of beside them. Landscape/desktop keeps the exact old layout.
		const portrait = viewportAspect() < 0.85;

		if (portrait) {
			// Portrait bind: each big number and its caption must read as ONE unit, and
			// every unit must be clearly separated from the next. The old layout had the
			// caption gap (0.09) nearly equal to the leftover pitch (0.155-0.09=0.065), so
			// a number floated ambiguously between its own label and the next number — the
			// last vital ('AVERAGE RETENTION OXYGEN') looked label-only because its number
			// had drifted up beside the previous caption. Fix: a SMALL number->label gap
			// (tight caption) and a LARGE row pitch (clear break) so binding is unambiguous.
			// portraitAdapt multiplies every y by inv (>1 in portrait), so these authored
			// gaps scale with the live aspect — nothing here is a hardcoded phone width.
			const vTop = 0.66;
			const vGap = 0.052; // number -> its own caption (tight)
			const vPitch = 0.17; // row -> next row (loose, > 2x the caption gap)
			vitals.forEach(([value, label, color], i) => {
				const y = vTop - i * vPitch;
				items.push(line(`tl:vital-num:${i}`, 'tl-vital', value, -0.92, y, 0.03, color, { depth: 0.9, weight: 0.85 }));
				items.push(line(`tl:vital-lbl:${i}`, 'tl-vital-lbl', label, -0.92, y - vGap, 0.016, LABEL_HI, { depth: 0.7, weight: 0.7, maxWidthEm: 30 }));
			});
			// Receipt/hint below the last caption. Bigger header + hint text so the
			// 'CLICK A GROWTH RING...' subtext is readable on a phone (bumped in
			// buildReceiptItems via the portrait flag, not a magic size here).
			buildReceiptItems(items, -0.92, vTop - vitals.length * vPitch - 0.02, 0.028, LABEL_HI, true);
			return items;
		}

		vitals.forEach(([value, label, color], i) => {
			const y = 0.72 - i * 0.11;
			items.push(line(`tl:vital-num:${i}`, 'tl-vital', value, -0.92, y, 0.05, color, { depth: 0.9, weight: 0.85 }));
			items.push(line(`tl:vital-lbl:${i}`, 'tl-vital-lbl', label, -0.92, y - 0.05, 0.017, MUTED, { depth: 0.5 }));
		});

		buildReceiptItems(items, 0.3, 0.9, 0.022);

		return items;
	}

	// Receipt / audit block. `rx` is the left anchor, `ry0` the header baseline, and
	// `hdrSize` the header size — desktop pins it to the right column (0.3, 0.9), but
	// portrait stacks it below the vitals in the single left column. Everything below
	// the header is authored relative to ry0 so both layouts share one code path.
	function buildReceiptItems(items: TextLayerItem[], rx: number, ry0: number, hdrSize: number, muted = MUTED, portrait = false): void {
		items.push(line('tl:receipt-hdr', 'tl-receipt-hdr', 'TIME-SLICE RECEIPT', rx, ry0, hdrSize, INDIGO, { depth: 0.85 }));
		const rowTop = ry0 - 0.08;

		if (selectedCell && selectedMemory) {
			const c = selectedCell;
			const snippet = selectedMemory.content.replace(/\s+/g, ' ').trim().slice(0, 46);
			const state = c.suppressed ? 'suppressed' : c.rewritten ? 'rewritten' : 'created';
			const rows: Array<[string, [number, number, number, number]]> = [
				[`#${c.memoryId.slice(0, 12)}`, CYAN],
				[snippet, MUTED],
				[`valid  ${String(c.validFrom).slice(0, 19)}`, OXYGEN],
				[`tx     ${String(c.transactionAt).slice(0, 19)}`, INDIGO],
				[`retain ${Math.round(c.retention * 100)}%   state ${state}`, state === 'suppressed' ? SCARLET : OXYGEN]
			];
			rows.forEach(([t, color], i) => {
				items.push(line(`tl:receipt:${i}`, 'tl-receipt', t, rx, rowTop - i * 0.06, 0.018, color, { revealSpan: 10, startFrame: i * 2 }));
			});

			// real audit events (memoryAudit)
			const auditY0 = rowTop - rows.length * 0.06 - 0.04;
			items.push(line('tl:audit-hdr', 'tl-audit-hdr', auditLoading ? 'MEMORY-AUDIT (loading...)' : 'MEMORY-AUDIT', rx, auditY0, 0.016, MUTED, { depth: 0.5 }));
			selectedAudit.slice(0, 10).forEach((ev, i) => {
				const t = `${ev.action}  ${String(ev.timestamp).slice(0, 19)}`;
				items.push(line(`tl:audit:${i}`, 'tl-audit', t, rx, auditY0 - 0.03 - i * 0.045, 0.015, CYAN, { revealSpan: 8, startFrame: i * 2, depth: 0.6 }));
			});
			if (!auditLoading && selectedAudit.length === 0) {
				items.push(line('tl:audit-empty', 'tl-audit', 'no audit events returned', rx, auditY0 - 0.03, 0.015, MUTED));
			}
		} else if (selectedRing) {
			const r = selectedRing;
			const rows: Array<[string, [number, number, number, number]]> = [
				[r.date, CYAN],
				[`memories     ${r.count}`, OXYGEN],
				[`avg retain   ${Math.round(r.retention * 100)}%`, OXYGEN],
				[`rewrites     ${r.updatedCount}`, INDIGO],
				[`suppressed   ${r.suppressedCount}`, r.suppressedCount > 0 ? SCARLET : MUTED],
				['click a cell for its audit receipt', MUTED]
			];
			rows.forEach(([t, color], i) => {
				items.push(line(`tl:ring:${i}`, 'tl-receipt', t, rx, rowTop - i * 0.06, 0.017, color, { revealSpan: 10, startFrame: i * 2 }));
			});
		} else {
			// The idle hint is the only call-to-action when nothing is selected, so on a
			// phone it must be readable, not a dim whisper. portraitAdapt caps a line's
			// size to fit its full char count on ONE line (it can't see the wrap), so a
			// 90-char hint is forced tiny no matter the authored size. In portrait we use
			// a SHORTER hint (fewer chars -> the size boost isn't capped away) at a larger
			// authored size and full-alpha LABEL_HI (passed in as `muted`) so it lifts
			// clear of the dimmed ring field. Desktop keeps the exact old long hint/size.
			const hintText = portrait
				? 'TAP A RING FOR ITS DATE SLICE'
				: 'CLICK A GROWTH RING FOR ITS DATE SLICE, OR A CELL FOR ITS VALID-TIME VS TRANSACTION-TIME RECEIPT';
			const hintSize = portrait ? 0.026 : 0.017;
			items.push(
				line(
					'tl:receipt-hint',
					'tl-receipt',
					hintText,
					rx,
					rowTop - 0.02,
					hintSize,
					muted,
					{ maxWidthEm: portrait ? 22 : 34, revealSpan: 20 }
				)
			);
		}
	}

	// ── the route pass factory: growth-rings organ + MSDF HUD/receipt, zero DOM ──
	function createTimelineOrganPasses(engine: ObservatoryEngine, scene: RouteSceneModel): RouteFramePass[] {
		engineHandle = engine;
		const ringPasses = createTimelinePasses(engine, scene);
		const text = new TextLayerPass(engine);
		textPass = text;
		void text.init().then(() => text.setText(buildTextItems()));

		const textFramePass: RouteFramePass = {
			render(renderPass) {
				text.render(renderPass);
			},
			pickAt(ndcX, ndcY) {
				const hit = text.pickAt(ndcX, ndcY);
				const nextRun = hit && (hit.kind === 'tl-vital' || hit.kind === 'tl-range') ? hit.id : null;
				if (nextRun !== focusedRun) {
					focusedRun = nextRun;
					text.setRunDepth(nextRun, 1);
				}
				if (hit?.kind === 'tl-range') {
					cycleRange();
					return null; // consume the click; no external pick
				}
				return null; // ring/cell picks are handled by the ring passes below
			},
			dispose() {
				text.dispose();
				if (textPass === text) textPass = null;
			}
		};

		// ring passes first (they own ring/cell picking), text HUD on top
		return [...ringPasses, textFramePass];
	}

	function handleRoutePick(pick: RoutePick) {
		if (pick.kind === 'timeline-cell') {
			const cell = pick.payload as TimelineCell;
			selectedCell = cell;
			selectedRing = null;
			void fetchAudit(cell.memoryId);
		} else if (pick.kind === 'timeline-ring') {
			const ring = pick.payload as TimelineRing;
			selectedRing = ring;
			selectedCell = null;
		}
	}
</script>

<svelte:head>
	<title>Bitemporal Growth Rings · Vestige</title>
</svelte:head>

<RouteStage
	organ="timeline"
	seed={`timeline-growth-rings:${days}:${totalMemories}`}
	scene={timelineScene}
	passes={createTimelineOrganPasses}
	loading={loading || auditLoading}
	{error}
	emptyLabel="NO MEMORY GROWTH RINGS IN THIS WINDOW"
	onpick={handleRoutePick}
/>

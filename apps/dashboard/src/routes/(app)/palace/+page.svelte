<script lang="ts">
	// Spatial Palace — the map route. The 19 organ routes rendered as ONE 3D
	// glowing constellation you orbit and click to enter. Uses the bespoke
	// PalaceNodePass (hero-scale orbs, own close-orbit camera, family colors,
	// CPU-reprojection picking) + on-node MSDF labels that ride each orb as the
	// camera turns. Clicking a region enters that organ (hard-cut goto for now;
	// the seamless dolly+crossfade dive lands in UNIT 4).
	import { onDestroy } from 'svelte';
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { PalaceNodePass } from '$lib/observatory/palace-node-pass';
	import { buildOrganLabels, type OrganScreenPos } from '$lib/observatory/palace-labels';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import { rgb01 } from '$lib/observatory/cognitive-palette';
	import { ORGAN_REGIONS } from '$lib/observatory/palace-map';

	// HUD text must punch THROUGH the bright additive nebula behind it — so use
	// near-white hot colors at full alpha (the MSDF renders into the same HDR
	// scene as the bloom; a dim tint loses, a bright one wins).
	const TITLE = [...rgb01('#EAFBFF'), 1] satisfies [number, number, number, number]; // hot cyan-white
	const SUB = [...rgb01('#CFFFE9'), 1] satisfies [number, number, number, number]; // bright mint-white

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let pass: PalaceNodePass | null = null;
	let textPass: TextLayerPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	let hoveredHref: string | null = $state(null);

	onDestroy(() => {
		textPass?.dispose();
		textPass = null;
		pass = null;
		engineRef = null;
	});

	async function handleReady(engine: ObservatoryEngine) {
		engineRef = engine;
		// The bespoke 3D organ constellation (does NOT auto-register — addPass it).
		pass = new PalaceNodePass(engine);
		engine.addPass(pass);
		pass.uploadRegions(ORGAN_REGIONS);

		// MSDF text: title/subtitle HUD + the per-node floating organ labels.
		const t = new TextLayerPass(engine);
		textPass = t;
		await t.init();
		engine.addPass(t);
		engine.demoClock.reset();
		t.setText(buildAllText());
	}

	// ── depth conversion: getScreenPositions() reports clip-w (larger = FARTHER);
	// buildOrganLabels wants 0..1 NEARNESS (1 = closest). Normalize per-frame over
	// the visible nodes so labels size/brighten by real front-to-back depth. ──
	function toOrganPositions(): OrganScreenPos[] {
		if (!pass) return [];
		const raw = pass.getScreenPositions();
		const visible = raw.filter((p) => p.visible);
		let min = Infinity;
		let max = -Infinity;
		for (const p of visible) {
			if (p.depth < min) min = p.depth;
			if (p.depth > max) max = p.depth;
		}
		const degenerate = max === min; // one visible node, or all at one depth
		const span = max - min || 1;
		return raw.map((p) => ({
			href: p.href,
			ndcX: p.ndcX,
			ndcY: p.ndcY,
			// invert: nearest (smallest clip-w) → 1, farthest → 0. When the span
			// is degenerate (a single visible orb), treat it as NEAREST (1), not
			// far (0) — otherwise the lone on-screen orb renders tiny/dim.
			depth: p.visible ? (degenerate ? 1 : Math.min(1, Math.max(0, (max - p.depth) / span))) : 0,
			visible: p.visible
		}));
	}

	function sanitizeAscii(v: string): string {
		return v
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function hudLines(): TextLayerItem[] {
		return [
			{
				id: 'palace:title',
				kind: 'palace-hud',
				text: sanitizeAscii('THE MEMORY PALACE'),
				x: -0.94,
				y: 0.88,
				size: 0.062,
				color: TITLE,
				depth: 1,
				weight: 1,
				revealSpan: 18
			},
			{
				id: 'palace:sub',
				kind: 'palace-hud',
				text: sanitizeAscii(`${ORGAN_REGIONS.length} ORGANS - CLICK A REGION TO ENTER`),
				x: -0.94,
				y: 0.79,
				size: 0.032,
				color: SUB,
				depth: 1,
				weight: 0.85,
				revealSpan: 18,
				maxWidthEm: 70
			}
		];
	}

	function buildAllText(): TextLayerItem[] {
		const labels = buildOrganLabels(toOrganPositions(), { hoveredHref, dimUnhovered: !!hoveredHref });
		return [...hudLines(), ...labels];
	}

	// Per-frame: re-anchor the floating labels to their orbs as the camera orbits.
	function handleFrame() {
		textPass?.setText(buildAllText());
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
		if (!pass) return;
		const hit = pass.pickAt(ndc.x, ndc.y);
		const nextHref = hit?.href ?? null;
		if (nextHref !== hoveredHref) {
			hoveredHref = nextHref;
			// Focus+context: tell the field which organ is focused so it glows and
			// parts the others (hover-to-inspect navigation).
			pass.setHovered(hit?.index ?? -1);
			if (hostEl) hostEl.style.cursor = nextHref ? 'pointer' : 'default';
		}
	}

	function handlePointerLeave() {
		cursorSmoothed = null;
		hoveredHref = null;
		pass?.setHovered(-1);
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		if (hostEl) hostEl.style.cursor = 'default';
	}

	function handlePointerDown(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc || !pass || pass.isDiving) return;
		const hit = pass.pickAt(ndc.x, ndc.y);
		if (!hit) return;
		// Portal dive: the camera rushes THROUGH the orb, then we enter the organ.
		const started = pass.startDive(hit.href, (href) => {
			void goto(`${base}${href}`);
		});
		// If the dive couldn't start (already diving / unknown), enter directly.
		if (!started) void goto(`${base}${hit.href}`);
	}
</script>

<svelte:head>
	<title>The Memory Palace · Vestige</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={hostEl}
	class="fixed inset-0 bg-[#020307]"
	onpointerdown={handlePointerDown}
	onpointermove={handlePointerMove}
	onpointerleave={handlePointerLeave}
>
	<ObservatoryCanvas demo="recall-path" seed="vestige-spatial-palace-v1" onframe={handleFrame} onready={handleReady} />
</div>

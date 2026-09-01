<script lang="ts">
	import { onDestroy } from 'svelte';
	import { page } from '$app/stores';
	import { base } from '$app/paths';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine } from '$lib/observatory/engine';
	import { PalaceBrainPass } from '$lib/observatory/palace-brain-pass';
	import { buildOrganLabels } from '$lib/observatory/palace-labels';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import {
		BITEMPORAL,
		IMMUNE,
		RETENTION,
		rgb01
	} from '$lib/observatory/cognitive-palette';
	import { ORGAN_REGIONS, regionByHref, type OrganRegion } from '$lib/observatory/palace-map';
	import { findRoute } from '$lib/os-routes';
	import { burstNavigate } from '$lib/stores/route-burst';

	const TITLE = [...rgb01('#F5FFF2'), 1] satisfies [number, number, number, number];
	const SUB = [...rgb01('#9DFFEB'), 1] satisfies [number, number, number, number];
	const DIM = [...rgb01('#7DAFA9'), 0.82] satisfies [number, number, number, number];

	const FAMILY_COLOR: Record<OrganRegion['family'], string> = {
		reasoning: RETENTION.bridge,
		memory: RETENTION.recall,
		immune: IMMUNE.veto,
		signal: BITEMPORAL.supersession,
		temporal: BITEMPORAL.txShadow,
		system: RETENTION.luciferin
	};

	let hostEl: HTMLDivElement | null = $state(null);
	let engineRef: ObservatoryEngine | null = null;
	let pass: PalaceBrainPass | null = null;
	let textPass: TextLayerPass | null = null;
	let cursorSmoothed: { x: number; y: number } | null = null;
	let hoveredHref: string | null = $state(null);
	let pressed: { x: number; y: number; href: string | null } | null = null;
	let clickPoint = { x: 0, y: 0 };
	let navigationStarted = false;
	let reducedMotion = false;
	let lastLabelFrame = -1;

	let freezeFrame = $derived.by(() => {
		const raw = $page.url.searchParams.get('frame');
		if (raw === null) return null;
		const n = Number(raw);
		return Number.isFinite(n) ? Math.floor(n) : null;
	});

	onDestroy(() => {
		if (engineRef && pass) engineRef.removePass(pass);
		else pass?.dispose();
		if (engineRef && textPass) engineRef.removePass(textPass);
		else textPass?.dispose();
		pass = null;
		textPass = null;
		engineRef = null;
	});

	async function handleReady(engine: ObservatoryEngine) {
		try {
			engineRef = engine;
			reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
			const swarm = new PalaceBrainPass(engine);
			pass = swarm;
			swarm.setReducedMotion(reducedMotion);
			engine.addPass(swarm);
			swarm.uploadRegions(ORGAN_REGIONS);
			const text = new TextLayerPass(engine);
			textPass = text;
			await text.init();
			engine.addPass(text);
			engine.demoClock.reset();
			text.setText(buildText());
		} catch (error) {
			console.error('[palace] Failed to initialize swarm:', error);
		}
	}

	function sanitizeAscii(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function buildText(): TextLayerItem[] {
		const items: TextLayerItem[] = [
			{
				id: 'palace:title',
				kind: 'palace-hud',
				text: 'VESTIGE // MEMORY PALACE',
				x: -0.92,
				y: 0.88,
				size: 0.052,
				color: TITLE,
				depth: 1,
				weight: 1,
				revealSpan: 24
			},
			{
				id: 'palace:sub',
				kind: 'palace-hud',
				text: sanitizeAscii(`${ORGAN_REGIONS.length} LIVING ORGANS - HOVER TO REVEAL - CLICK TO ENTER`),
				x: -0.92,
				y: 0.8,
				size: 0.025,
				color: SUB,
				depth: 1,
				weight: 0.86,
				revealSpan: 28,
				maxWidthEm: 66
			},
			{
				id: 'palace:hint',
				kind: 'palace-hud',
				text: navigationStarted ? 'PORTAL LOCKED // COLLAPSING COGNITIVE FIELD' : 'MOVE THROUGH THE FIELD',
				x: -0.92,
				y: -0.87,
				size: 0.02,
				color: DIM,
				depth: 0.9,
				weight: 0.72,
				revealSpan: 18
			}
		];

		if (hoveredHref) {
			const region = regionByHref(hoveredHref);
			const route = findRoute(hoveredHref);
			if (region) {
				const accent = [...rgb01(FAMILY_COLOR[region.family]), 1] satisfies [number, number, number, number];
				items.push(
					{
						id: 'palace:focus-label',
						kind: 'palace-focus',
						text: navigationStarted ? `ENTERING ${region.label}` : region.label,
						x: 0.34,
						y: 0.88,
						size: 0.046,
						color: accent,
						depth: 1,
						weight: 1,
						revealSpan: 18
					},
					{
						id: 'palace:focus-purpose',
						kind: 'palace-focus',
						text: sanitizeAscii(route?.purpose ?? 'ENTER THIS COGNITIVE ORGAN'),
						x: 0.34,
						y: 0.8,
						size: 0.021,
						color: TITLE,
						depth: 1,
						weight: 0.8,
						revealSpan: 30,
						maxWidthEm: 42
					}
				);
			}
		}
		return items;
	}

	function refreshText() {
		const hud = buildText();
		const labels = pass
			? buildOrganLabels(pass.getScreenPositions(), {
					hoveredHref,
					dimUnhovered: Boolean(hoveredHref),
					aspect:
						(engineRef?.params[6] || 0) / Math.max(1, engineRef?.params[7] || 1)
				})
			: [];
		textPass?.setText([...hud, ...labels]);
	}

	function handleFrame(frame: number) {
		if (frame === lastLabelFrame) return;
		if (frame % 2 !== 0) return;
		lastLabelFrame = frame;
		refreshText();
	}

	function pointerToNdc(e: PointerEvent): { x: number; y: number } | null {
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
		if (!pass || pass.isBursting) return;
		const hit = pass.pickAt(ndc.x, ndc.y);
		const nextHref = hit?.href ?? null;
		if (nextHref !== hoveredHref) {
			hoveredHref = nextHref;
			pass.setHovered(hit?.index ?? -1);
			refreshText();
			if (hostEl) hostEl.style.cursor = nextHref ? 'pointer' : 'crosshair';
		}
	}

	function handlePointerLeave() {
		if (pass?.isBursting) return;
		pressed = null;
		cursorSmoothed = null;
		hoveredHref = null;
		pass?.setHovered(-1);
		engineRef?.setCursorPreNdc(999, 999, 0, 0);
		refreshText();
		if (hostEl) hostEl.style.cursor = 'crosshair';
	}

	function handlePointerDown(e: PointerEvent) {
		if (e.button !== 0 || !pass || pass.isBursting) return;
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		pressed = { x: e.clientX, y: e.clientY, href: pass.pickAt(ndc.x, ndc.y)?.href ?? null };
	}

	function handlePointerCancel() {
		pressed = null;
	}

	function handlePointerUp(e: PointerEvent) {
		const down = pressed;
		pressed = null;
		if (!down || !pass || pass.isBursting) return;
		if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 9) return;
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		const hit = pass.pickAt(ndc.x, ndc.y);
		if (!hit || hit.href !== down.href) return;
		clickPoint = { x: e.clientX, y: e.clientY };
		hoveredHref = hit.href;
		pass.setHovered(hit.index);
		navigationStarted = true;
		refreshText();
		if (hostEl) hostEl.style.cursor = 'wait';
		const started = pass.startBurst(hit.href, navigateAtPeak);
		if (!started) void navigateAtPeak(hit.href);
	}

	async function navigateAtPeak(href: string) {
		const region = regionByHref(href);
		await burstNavigate(`${base}${href}`, {
			clientX: clickPoint.x,
			clientY: clickPoint.y,
			color: region ? FAMILY_COLOR[region.family] : RETENTION.luciferin,
			reduced: reducedMotion
		});
	}
</script>

<svelte:head>
	<title>Memory Palace · VestigeOS</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={hostEl}
	class="palace-host fixed inset-0 bg-[#020307]"
	role="application"
	aria-label="VestigeOS Memory Palace. Nine living cognitive organs. Use the Command palette for keyboard navigation."
	onpointerdown={handlePointerDown}
	onpointerup={handlePointerUp}
	onpointercancel={handlePointerCancel}
	onpointermove={handlePointerMove}
	onpointerleave={handlePointerLeave}
>
	<ObservatoryCanvas
		demo="recall-path"
		seed="vestige-palace-swarm-v2"
		{freezeFrame}
		onframe={handleFrame}
		onready={handleReady}
	/>
</div>

<style>
	.palace-host {
		cursor: crosshair;
		touch-action: manipulation;
		isolation: isolate;
	}

	/* Atmosphere visible during adapter boot; the GPU canvas takes over without a
	   color discontinuity because both layers share the blackwater medium. */
	.palace-host::before {
		content: '';
		position: absolute;
		inset: 0;
		z-index: -1;
		background:
			radial-gradient(circle at 50% 44%, rgba(41, 242, 169, 0.09), transparent 26%),
			radial-gradient(circle at 24% 18%, rgba(29, 214, 255, 0.06), transparent 34%),
			#020307;
	}
</style>

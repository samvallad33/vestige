<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { goto } from '$app/navigation';
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
	import { CAUSAL, IMMUNE, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
	import { type DemoMode } from '$lib/observatory/types';
	import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';
	import {
		createNavLayerPass,
		type NavLayerPass,
		type NavPick
	} from '$lib/observatory/nav/nav-layer';
	import {
		assertProvenance,
		emptyScene,
		type RouteOrgan,
		type RouteSceneModel
	} from '$lib/observatory/route-scene';

	export interface RoutePick {
		id: string;
		kind: string;
		index?: number;
		payload?: unknown;
	}

	export interface RouteFramePass extends FramePass {
		uploadScene?: (scene: RouteSceneModel) => void;
		pickAt?: (ndcX: number, ndcY: number) => Promise<RoutePick | null> | RoutePick | null;
		dispose?: () => void;
	}

	export type RoutePassFactory = (
		engine: ObservatoryEngine,
		scene: RouteSceneModel
	) => RouteFramePass[];

	interface Props {
		organ: RouteOrgan;
		seed?: string;
		scene?: RouteSceneModel | null;
		passes?: RoutePassFactory | RouteFramePass[];
		embedded?: boolean;
		onpick?: (pick: RoutePick) => void;
		/** Existing ObservatoryEngine needs an existing DemoMode; route id stays pass-local. */
		demo?: DemoMode;
		loading?: boolean;
		error?: string | null;
		emptyLabel?: string;
	}

	let {
		organ,
		seed = 'vestige-route-organ-v1',
		scene = null,
		passes,
		embedded = false,
		onpick,
		demo = 'recall-path',
		loading = false,
		error = null,
		emptyLabel = 'NO ROUTE DATA IN FIELD'
	}: Props = $props();

	let currentScene = $derived(scene ?? emptyScene(organ));
	let canvasLayerEl: HTMLDivElement | null = $state(null);
	let engine = $state<ObservatoryEngine | null>(null);
	let chromeText: TextLayerPass | null = null;
	let navPass: NavLayerPass | null = null;
	let routePasses: RouteFramePass[] = [];
	let frameCount = $state(0);
	let fpsEstimate = $state(0);
	let paused = $state(false);
	let userSetPause = $state(false);
	let ready = $state(false);
	let cursorSmoothed: { x: number; y: number } | null = null;
	let focusedChromeRun: string | null = null;

	const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
	const DIM_GREEN = [...rgb01(RETENTION.recall), 0.58] satisfies [number, number, number, number];
	const OXYGEN = [...rgb01(RETENTION.luciferin), 0.86] satisfies [number, number, number, number];
	const SCARLET = [...rgb01(IMMUNE.veto), 0.9] satisfies [number, number, number, number];
	const AMBER = [...rgb01(IMMUNE.caution), 0.78] satisfies [number, number, number, number];

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
		updateChromeText();
	}

	$effect(() => {
		engine?.setPaused(paused);
		updateChromeText();
	});

	function upload(sceneToUpload: RouteSceneModel) {
		if (!engine) return;
		if (import.meta.env.DEV) assertProvenance(sceneToUpload);
		for (const pass of routePasses) pass.uploadScene?.(sceneToUpload);
		engine.demoClock.reset();
		updateChromeText();
	}

	function handleFrame(frame: number, fps: number) {
		frameCount = frame;
		fpsEstimate = fps;
		navPass?.setActivePath(currentDashboardPath());
		updateChromeText(frame, fps);
	}

	async function handleReady(e: ObservatoryEngine) {
		ready = false;
		engine = e;
		for (const pass of routePasses) pass.dispose?.();
		chromeText?.dispose();
		navPass?.dispose();
		routePasses = typeof passes === 'function' ? passes(e, currentScene) : (passes ?? []);
		for (const pass of routePasses) e.addPass(pass);

		navPass = createNavLayerPass(e, { activePath: currentDashboardPath() });
		chromeText = new TextLayerPass(e);
		e.addPass(navPass);
		e.addPass(chromeText);
		await Promise.all([navPass.init(), chromeText.init()]);
		if (engine !== e) return;
		ready = true;
		upload(currentScene);
	}

	$effect(() => {
		// Track ONLY the scene: re-upload when the data changes, never every frame.
		// upload() calls updateChromeText(), which reads the per-frame frameCount /
		// fpsEstimate $state; without untrack the effect would take a reactive dep on
		// those and re-run every frame, calling demoClock.reset() each time — pinning
		// params.frame at ~0 and permanently zeroing the MSDF reveal gate (text goes
		// black on every text-primary RouteStage organ). untrack severs that leak.
		const nextScene = currentScene;
		if (!engine || !ready) return;
		untrack(() => upload(nextScene));
	});

	function currentDashboardPath(): string {
		const path = $page.url.pathname;
		return path.startsWith(base) ? path.slice(base.length) || '/' : path;
	}

	// MSDF atlas is ASCII-only: any non-ASCII (em-dashes in organ emptyLabels,
	// unicode in backend error messages) renders as a literal '?'. Sanitize the two
	// chrome strings that come from OUTSIDE this file (emptyLabel prop, error prop)
	// at the single point they enter the text pass, so no organ can ship a '?'.
	function asciiSafe(value: string): string {
		return value
			.replace(/[—–]/g, '-')
			.replace(/[‘’]/g, "'")
			.replace(/[“”]/g, '"')
			.replace(/…/g, '...')
			.replace(/[^\x20-\x7E]/g, '?');
	}

	function makeChromeItems(frame = frameCount, fps = fpsEstimate): TextLayerItem[] {
		const items: TextLayerItem[] = [
			{
				id: 'route-chrome:pause',
				kind: 'route-chrome',
				text: paused ? '> RESUME' : '|| PAUSE',
				x: 0.66,
				y: -0.86,
				size: 0.034,
				color: paused ? AMBER : CYAN,
				revealSpan: 1
			},
			{
				id: 'route-chrome:telemetry',
				kind: 'route-telemetry',
				text: `${organ.toUpperCase()} - ${frame}F - ${fps}FPS`,
				x: 0.44,
				y: 0.88,
				size: 0.022,
				color: DIM_GREEN,
				revealSpan: 1
			}
		];

		if (loading) {
			items.push({
				id: 'route-chrome:loading',
				kind: 'route-status',
				text: 'REPLAYING COGNITIVE RECEIPT...',
				x: -0.23,
				y: 0.02,
				size: 0.046,
				color: OXYGEN,
				startFrame: Math.max(0, frame - 90),
				revealSpan: 72
			});
		} else if (error) {
			items.push(
				{
					id: 'route-chrome:error-pulse',
					kind: 'route-status-pulse',
					text: '!!!!!!!!!!!!!!!!!!!!!!!!',
					x: -0.36,
					y: -0.035,
					size: 0.025,
					color: SCARLET,
					revealSpan: 1
				},
				{
					id: 'route-chrome:error',
					kind: 'route-status',
					text: asciiSafe(`ERROR - ${error}`).slice(0, 72),
					x: -0.54,
					y: 0.025,
					size: 0.032,
					color: SCARLET,
					revealSpan: 14,
					maxWidthEm: 48
				}
			);
		} else if (!currentScene.alive) {
			items.push({
				id: 'route-chrome:empty',
				kind: 'route-status',
				text: asciiSafe(emptyLabel),
				x: -0.36,
				y: 0.02,
				size: 0.034,
				color: DIM_GREEN,
				revealSpan: 24,
				maxWidthEm: 48
			});
		}
		return items;
	}

	function updateChromeText(frame = frameCount, fps = fpsEstimate) {
		chromeText?.setText(makeChromeItems(frame, fps));
	}

	function pointerToNdc(e: PointerEvent | MouseEvent): { x: number; y: number } | null {
		if (!canvasLayerEl) return null;
		const rect = canvasLayerEl.getBoundingClientRect();
		if (rect.width === 0 || rect.height === 0) return null;
		return {
			x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
			y: -(((e.clientY - rect.top) / rect.height) * 2 - 1)
		};
	}

	function writeCursorLens(ndc: { x: number; y: number }) {
		if (!canvasLayerEl || !engine) return;
		const rect = canvasLayerEl.getBoundingClientRect();
		if (rect.width === 0 || rect.height === 0) return;
		const aspect = Math.max(0.0001, rect.width / Math.max(1, rect.height));
		const raw = {
			x: ndc.x * Math.max(aspect, 1),
			y: ndc.y / Math.min(aspect, 1)
		};
		const prev = cursorSmoothed ?? raw;
		const next = {
			x: prev.x + (raw.x - prev.x) * 0.35,
			y: prev.y + (raw.y - prev.y) * 0.35
		};
		cursorSmoothed = next;
		engine.setCursorPreNdc(next.x, next.y, next.x - prev.x, next.y - prev.y);
	}

	function handlePointerMove(e: PointerEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		writeCursorLens(ndc);
		const navHit = navPass?.setHoverFromNdc(ndc.x, ndc.y);
		const chromeHit = chromeText?.pickAt(ndc.x, ndc.y);
		const nextFocus = chromeHit?.id ?? null;
		if (nextFocus !== focusedChromeRun) {
			focusedChromeRun = nextFocus;
			chromeText?.setRunDepth(nextFocus, 1.0);
		}
		if (canvasLayerEl) canvasLayerEl.style.cursor = navHit || chromeHit || onpick ? 'crosshair' : 'default';
	}

	function handlePointerLeave() {
		navPass?.clearHover();
		focusedChromeRun = null;
		chromeText?.setRunDepth(null);
		cursorSmoothed = null;
		engine?.setCursorPreNdc(999, 999, 0, 0);
		if (canvasLayerEl) canvasLayerEl.style.cursor = 'default';
	}

	async function handleFieldClick(e: MouseEvent) {
		const ndc = pointerToNdc(e);
		if (!ndc) return;
		const navHit: NavPick | null = navPass?.pickAt(ndc.x, ndc.y) ?? null;
		if (navHit) {
			await goto(`${base}${navHit.payload.href}`);
			return;
		}
		const chromeHit = chromeText?.pickAt(ndc.x, ndc.y);
		if (chromeHit?.id === 'route-chrome:pause') {
			togglePause();
			return;
		}
		// Pick TOP-DOWN: routePasses render back-to-front (organs return
		// [field, ...text/chrome], so text draws ON TOP of the field). Picking must
		// mirror what the user sees on top — so iterate in REVERSE (front first).
		// Otherwise a background field cell that happens to sit under a foreground
		// text control would steal the click from the control the user actually sees.
		for (let i = routePasses.length - 1; i >= 0; i--) {
			const hit = await routePasses[i].pickAt?.(ndc.x, ndc.y);
			if (hit) {
				onpick?.(hit);
				return;
			}
		}
	}

	onMount(() => {
		return () => {
			for (const pass of routePasses) pass.dispose?.();
			chromeText?.dispose();
			navPass?.dispose();
			routePasses = [];
			chromeText = null;
			navPass = null;
		};
	});

	onMount(() => initReducedMotion());
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div
	bind:this={canvasLayerEl}
	class="{embedded ? 'absolute' : 'fixed'} inset-0 overflow-hidden"
	onclick={handleFieldClick}
	onpointermove={handlePointerMove}
	onpointerleave={handlePointerLeave}
>
	<ObservatoryCanvas {demo} {seed} onframe={handleFrame} onready={handleReady} />
</div>

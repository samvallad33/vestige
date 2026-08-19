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
		/** Per-organ WebGPU pixel-density budget. */
		maxDpr?: number;
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
		maxDpr = 2,
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
	let lastChromeSignature: string | null = null;

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
		// The Witness chamber has no live telemetry. Keeping these as reactive state
		// on every rAF made Svelte invalidate the whole route 60 times per second
		// while the shader was otherwise settled. Development telemetry on other
		// organs stays live, but is sampled below instead of forcing a full text
		// buffer rebuild every frame.
		if (import.meta.env.DEV && organ !== 'witness') {
			frameCount = frame;
			fpsEstimate = fps;
		}
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
		lastChromeSignature = null;
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

	// Portrait/phone check — same live-aspect signal the text + field layers use
	// (engine.params[6]/[7], window fallback). On a phone the in-canvas HUD chrome
	// (dev telemetry + the floating PAUSE) is SUPPRESSED: it has fixed landscape
	// NDC anchors that overprint the reflowed content, the telemetry is debug-only
	// noise a real user shouldn't see, and the DOM MobileNav already owns the
	// bottom-thumb zone. Desktop keeps the full chrome unchanged.
	function isPortrait(): boolean {
		let vw = engine?.params[6] || 0;
		let vh = engine?.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return false;
		return vw / vh < 0.85;
	}

	function makeChromeItems(frame = frameCount, fps = fpsEstimate): TextLayerItem[] {
		const portrait = isPortrait();
		// Desktop: floating PAUSE (always) + FPS telemetry (dev builds only — it's
		// debug noise a launch user shouldn't see). Phone: neither (see isPortrait).
		const items: TextLayerItem[] = portrait
			? []
			: [
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
					...(import.meta.env.DEV && organ !== 'witness'
						? [
								{
									id: 'route-chrome:telemetry',
									kind: 'route-telemetry',
									text: `${organ.toUpperCase()} - ${frame}F - ${fps}FPS`,
									x: 0.44,
									y: 0.88,
									size: 0.022,
									color: DIM_GREEN,
									revealSpan: 1
								} satisfies TextLayerItem
							]
						: [])
				];

		if (loading) {
			items.push({
				id: 'route-chrome:loading',
				kind: 'route-status',
				text: 'REPLAYING COGNITIVE RECEIPT...',
				x: portrait ? -0.82 : -0.23,
				y: 0.02,
				size: portrait ? 0.03 : 0.046,
				color: OXYGEN,
				startFrame: Math.max(0, frame - 90),
				revealSpan: 72,
				maxWidthEm: portrait ? 26 : undefined
			});
		} else if (error) {
			items.push(
				{
					id: 'route-chrome:error-pulse',
					kind: 'route-status-pulse',
					// The '!!!' pulse bar is a landscape flourish; drop it on a phone
					// (it overprints the error text once the field reflows narrow).
					text: portrait ? '' : '!!!!!!!!!!!!!!!!!!!!!!!!',
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
					x: portrait ? -0.82 : -0.54,
					y: 0.025,
					size: portrait ? 0.028 : 0.032,
					color: SCARLET,
					revealSpan: 14,
					maxWidthEm: portrait ? 24 : 48
				}
			);
		} else if (!currentScene.alive) {
			items.push({
				id: 'route-chrome:empty',
				kind: 'route-status',
				text: asciiSafe(emptyLabel),
				x: portrait ? -0.82 : -0.36,
				y: 0.02,
				size: portrait ? 0.03 : 0.034,
				color: DIM_GREEN,
				revealSpan: 24,
				maxWidthEm: 48
			});
		}
		return items;
	}

	function updateChromeText(frame = frameCount, fps = fpsEstimate) {
		if (!chromeText) return;
		const showsTelemetry = import.meta.env.DEV && organ !== 'witness';
		// Telemetry is diagnostic chrome, not simulation input. Sample it at 10 Hz
		// so the MSDF glyph layout/storage upload is never on the critical frame
		// path. Witness and production routes now upload their static chrome only
		// when an actual visible state changes (pause, load, error, empty, aspect).
		const sampledFrame = showsTelemetry ? Math.floor(frame / 6) * 6 : 0;
		const sampledFps = showsTelemetry ? Math.round(fps / 5) * 5 : 0;
		const signature = [
			organ,
			paused ? 'paused' : 'running',
			loading ? 'loading' : 'ready',
			error ?? '',
			currentScene.alive ? 'alive' : 'empty',
			isPortrait() ? 'portrait' : 'landscape',
			showsTelemetry ? `${sampledFrame}:${sampledFps}` : ''
		].join('|');
		if (signature === lastChromeSignature) return;
		lastChromeSignature = signature;
		chromeText.setText(makeChromeItems(sampledFrame, sampledFps));
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
			lastChromeSignature = null;
		};
	});

	onMount(() => initReducedMotion());
</script>

<!-- The GPU picker is a custom pointer surface. Pause/resume is separately
     exposed as a native button below for keyboard and assistive technology. -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<div
	bind:this={canvasLayerEl}
	class="{embedded ? 'absolute' : 'fixed'} inset-0 overflow-hidden"
	role="application"
	aria-label={`Interactive 3D ${organ} field`}
	onclick={handleFieldClick}
	onpointermove={handlePointerMove}
	onpointerleave={handlePointerLeave}
>
	<ObservatoryCanvas {demo} {seed} {maxDpr} onframe={handleFrame} onready={handleReady} />
	<button
		type="button"
		class="route-motion-control"
		onclick={togglePause}
		aria-pressed={paused}
		aria-label={paused ? 'Resume 3D field motion' : 'Pause 3D field motion'}
	>
		{paused ? 'RESUME MOTION' : 'PAUSE MOTION'}
	</button>
</div>

<style>
	.route-motion-control {
		position: fixed;
		right: 1rem;
		bottom: 1rem;
		z-index: 2;
		border: 1px solid rgba(34, 199, 222, 0.32);
		border-radius: 0.7rem;
		background: rgba(5, 6, 10, 0.84);
		color: rgba(143, 232, 242, 0.95);
		padding: 0.45rem 0.65rem;
		font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
		font-size: 0.65rem;
		letter-spacing: 0.1em;
		cursor: pointer;
	}

	.route-motion-control:focus-visible {
		outline: 2px solid #8fe8f2;
		outline-offset: 3px;
	}

	@media (max-width: 640px) {
		.route-motion-control {
			right: 0.75rem;
			bottom: 4.6rem;
		}
	}
</style>

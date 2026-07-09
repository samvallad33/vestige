<script lang="ts">
	import { onMount } from 'svelte';
	import ObservatoryCanvas from '$lib/components/ObservatoryCanvas.svelte';
	import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
	import { type DemoMode } from '$lib/observatory/types';
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
	let routePasses: RouteFramePass[] = [];
	let frameCount = $state(0);
	let fpsEstimate = $state(0);
	let paused = $state(false);
	let userSetPause = $state(false);
	let ready = $state(false);

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

	$effect(() => {
		engine?.setPaused(paused);
	});

	function upload(sceneToUpload: RouteSceneModel) {
		if (!engine) return;
		if (import.meta.env.DEV) assertProvenance(sceneToUpload);
		for (const pass of routePasses) pass.uploadScene?.(sceneToUpload);
		engine.demoClock.reset();
	}

	function handleFrame(frame: number, fps: number) {
		frameCount = frame;
		fpsEstimate = fps;
	}

	function handleReady(e: ObservatoryEngine) {
		ready = true;
		engine = e;
		for (const pass of routePasses) pass.dispose?.();
		routePasses = typeof passes === 'function' ? passes(e, currentScene) : (passes ?? []);
		for (const pass of routePasses) e.addPass(pass);
		upload(currentScene);
	}

	$effect(() => {
		if (!engine || !ready) return;
		upload(currentScene);
	});

	async function handleFieldClick(e: MouseEvent) {
		if (!onpick || !canvasLayerEl) return;
		const rect = canvasLayerEl.getBoundingClientRect();
		if (rect.width === 0 || rect.height === 0) return;
		const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
		const ndcY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
		for (const pass of routePasses) {
			const hit = await pass.pickAt?.(ndcX, ndcY);
			if (hit) {
				onpick(hit);
				return;
			}
		}
	}

	onMount(() => {
		return () => {
			for (const pass of routePasses) pass.dispose?.();
			routePasses = [];
		};
	});

	onMount(() => initReducedMotion());
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="{embedded ? 'absolute' : 'fixed'} inset-0 overflow-hidden route-stage">
	<div
		bind:this={canvasLayerEl}
		class="absolute inset-0 z-0"
		class:cursor-crosshair={!!onpick}
		onclick={handleFieldClick}
	>
		<ObservatoryCanvas {demo} {seed} onframe={handleFrame} onready={handleReady} />
	</div>

	<div class="absolute inset-0 z-10 pointer-events-none">
		<slot name="chrome" {frameCount} {fpsEstimate} {paused} {ready} />

		<button
			onclick={togglePause}
			class="absolute bottom-4 right-4 pointer-events-auto flex items-center gap-2 px-3 py-1.5
				rounded-xl border border-[#22C7DE]/25 bg-[#020307]/80 backdrop-blur-sm
				font-mono text-[11px] tracking-wide text-[#22C7DE]/80 hover:text-[#22C7DE]
				hover:border-[#22C7DE]/50 transition-colors"
			title={paused ? 'Resume route organ motion' : 'Pause route organ motion'}
			aria-pressed={paused}
		>
			{paused ? '▶ RESUME' : '❚❚ PAUSE'}
		</button>

		<div class="absolute top-4 right-4 font-mono text-[10px] tracking-[0.18em] text-[#7fe6c0]/60 uppercase">
			{organ} · {frameCount}f · {fpsEstimate}fps
		</div>

		{#if loading}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-auto">
				<div class="text-[#A8FF5E] font-mono text-sm tracking-widest animate-pulse">
					REPLAYING COGNITIVE RECEIPT...
				</div>
			</div>
		{/if}

		{#if error && !loading}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-auto">
				<div class="text-[#ff8a8a] font-mono text-sm border border-[#FF3B30]/50 bg-[#1a0508]/70 px-4 py-2 rounded">
					{error}
				</div>
			</div>
		{/if}

		{#if !loading && !error && !currentScene.alive}
			<div class="absolute inset-0 flex items-center justify-center pointer-events-none">
				<div class="text-[#5dcaa5]/70 font-mono text-xs tracking-widest border border-[#5dcaa5]/15 bg-[#020307]/45 px-4 py-2 rounded-xl backdrop-blur-sm">
					{emptyLabel}
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.route-stage {
		background: #020307;
	}
</style>

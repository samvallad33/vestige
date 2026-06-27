<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import type { Component } from 'svelte';

	interface Props {
		seed?: number;
		reducedMotion?: boolean;
		syncTarget?: HTMLElement;
		suppress?: boolean;
	}

	type EngineProps = {
		seed?: number;
		reducedMotion?: boolean;
		class?: string;
		syncTarget?: HTMLElement;
		suppress?: boolean;
	};

	let { seed, reducedMotion = false, syncTarget, suppress = false }: Props = $props();
	let Engine = $state<Component<EngineProps> | null>(null);
	let loading = false;
	let bridgeCanvas: HTMLCanvasElement | undefined;
	let bridgeFrame = 0;

	function liveEngineMode() {
		if (typeof document === 'undefined') return null;
		return document.querySelector('.raw-vestige-engine.live-engine')?.getAttribute('data-mode');
	}

	function scheduleEngineLoad() {
		if (loading || Engine) return;
		loading = true;
		import('$lib/launch/RawVestigeEngine.svelte')
			.then((module) => {
				Engine = module.default as Component<EngineProps>;
			})
			.catch((error) => {
				loading = false;
				console.warn('[launch] visual engine failed to load:', error);
			});
	}

	function drawBridge(now = performance.now()) {
		const mode = liveEngineMode();
		if (!bridgeCanvas || reducedMotion || mode === 'webgpu' || mode === 'fallback') return;
		const rect = bridgeCanvas.getBoundingClientRect();
		const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
		const width = Math.max(1, Math.floor((rect.width || window.innerWidth || 1) * dpr));
		const height = Math.max(1, Math.floor((rect.height || window.innerHeight || 1) * dpr));
		if (bridgeCanvas.width !== width || bridgeCanvas.height !== height) {
			bridgeCanvas.width = width;
			bridgeCanvas.height = height;
		}

		const ctx = bridgeCanvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		const w = width / dpr;
		const h = height / dpr;
		const t = now * 0.001;
		const bridgeSeed = seed ?? 20260625;
		const cx = w * 0.5;
		const cy = h * (w < 760 ? 0.42 : 0.46);
		const span = Math.min(w, h) * (w < 760 ? 0.38 : 0.34);

		ctx.clearRect(0, 0, w, h);
		ctx.globalCompositeOperation = 'lighter';
		for (let i = 0; i < 140; i += 1) {
			const phase = i * 2.399 + bridgeSeed * 0.0001;
			const ring = (i % 7) / 7;
			const z = Math.sin(t * 0.8 + phase) * 0.5 + 0.5;
			const radius = span * (0.18 + ring * 0.86) * (0.74 + z * 0.36);
			const angle = phase + t * (0.18 + ring * 0.08);
			const squash = 0.42 + z * 0.24;
			const x = cx + Math.cos(angle) * radius;
			const y = cy + Math.sin(angle * 1.23) * radius * squash;
			const size = 1.1 + z * 2.3;
			const hue = i % 3 === 0 ? '93,255,166' : i % 3 === 1 ? '54,240,255' : '185,140,255';
			ctx.fillStyle = `rgba(${hue},${0.32 + z * 0.42})`;
			ctx.shadowColor = `rgba(${hue},0.85)`;
			ctx.shadowBlur = 10 + z * 16;
			ctx.fillRect(x - size * 0.5, y - size * 0.5, size, size);
		}
		ctx.globalCompositeOperation = 'source-over';
		ctx.shadowBlur = 0;
		bridgeFrame = requestAnimationFrame(drawBridge);
	}

	onMount(() => {
		drawBridge();
		requestAnimationFrame(() => setTimeout(scheduleEngineLoad, 0));
	});

	onDestroy(() => {
		if (typeof window !== 'undefined') cancelAnimationFrame(bridgeFrame);
	});
</script>

<div class="instant-engine" aria-hidden="true">
	<div class="instant-grid"></div>
	<div class="instant-core"></div>
	<canvas bind:this={bridgeCanvas} class="instant-canvas"></canvas>
	<div class="instant-nodes"></div>
</div>

{#if Engine}
	<Engine
		{seed}
		{reducedMotion}
		class="live-engine"
		{syncTarget}
		{suppress}
	/>
{/if}

<style>
	.instant-engine {
		position: fixed;
		inset: 0;
		z-index: 0;
		overflow: hidden;
		pointer-events: none;
		background:
			linear-gradient(145deg, rgba(13, 255, 178, 0.12), transparent 34%),
			linear-gradient(215deg, rgba(115, 120, 255, 0.14), transparent 40%),
			#02030a;
		opacity: 1;
		transition: opacity 700ms ease;
	}

	.instant-grid,
	.instant-core,
	.instant-canvas,
	.instant-nodes {
		position: absolute;
		inset: 0;
	}

	.instant-canvas {
		width: 100%;
		height: 100%;
		display: block;
	}

	.instant-grid {
		background-image:
			linear-gradient(rgba(93, 255, 166, 0.08) 1px, transparent 1px),
			linear-gradient(90deg, rgba(82, 230, 255, 0.07) 1px, transparent 1px);
		background-size: 44px 44px;
		mask-image: linear-gradient(180deg, transparent, #000 22%, #000 68%, transparent);
		opacity: 0.42;
		transform: perspective(700px) rotateX(64deg) translateY(22%);
		transform-origin: 50% 72%;
	}

	.instant-core {
		background:
			conic-gradient(
				from 220deg at 50% 39%,
				transparent 0deg,
				rgba(93, 255, 166, 0.24) 54deg,
				rgba(54, 240, 255, 0.2) 116deg,
				rgba(185, 140, 255, 0.18) 182deg,
				transparent 255deg,
				rgba(93, 255, 166, 0.14) 320deg,
				transparent 360deg
			);
		filter: blur(18px) saturate(1.25);
		opacity: 0.8;
		animation: instant-turn 8s linear infinite;
	}

	.instant-nodes {
		background-image:
			radial-gradient(circle at 22% 34%, rgba(93, 255, 166, 0.88) 0 1px, transparent 2px),
			radial-gradient(circle at 34% 46%, rgba(82, 230, 255, 0.9) 0 1px, transparent 2px),
			radial-gradient(circle at 50% 29%, rgba(185, 140, 255, 0.88) 0 1px, transparent 2px),
			radial-gradient(circle at 62% 44%, rgba(93, 255, 166, 0.86) 0 1px, transparent 2px),
			radial-gradient(circle at 74% 33%, rgba(82, 230, 255, 0.86) 0 1px, transparent 2px),
			radial-gradient(circle at 44% 57%, rgba(185, 140, 255, 0.76) 0 1px, transparent 2px),
			radial-gradient(circle at 58% 62%, rgba(93, 255, 166, 0.72) 0 1px, transparent 2px);
		opacity: 0.9;
		filter: drop-shadow(0 0 8px rgba(82, 230, 255, 0.75));
	}

	:global(.raw-vestige-engine.live-engine) {
		z-index: 1;
	}

	@keyframes instant-turn {
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 760px) {
		.instant-grid {
			background-size: 34px 34px;
			transform: perspective(520px) rotateX(66deg) translateY(26%);
		}

		.instant-core {
			filter: blur(14px) saturate(1.2);
			opacity: 0.72;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.instant-core {
			animation: none;
		}
	}
</style>

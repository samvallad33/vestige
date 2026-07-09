<script lang="ts">
	/**
	 * Cognitive Observatory — WebGPU canvas host.
	 *
	 * Owns the ObservatoryEngine lifecycle: mount → boot → resize → dispose.
	 * If WebGPU is unavailable, renders a readable fallback instead of crashing
	 * (Increment 3 gate, spec §4).
	 */
	import { onMount, onDestroy } from 'svelte';
	import { ObservatoryEngine, type EngineStatus } from '$lib/observatory/engine';
	import type { DemoMode } from '$lib/observatory/types';

	interface Props {
		demo: DemoMode;
		seed: string;
		/** Capture mode: freeze the sim at this loop frame (?frame=N). */
		freezeFrame?: number | null;
		/** Telemetry callback: loop frame + fps estimate. */
		onframe?: (frame: number, fps: number) => void;
		/** Fired when the engine is running (the route uploads the graph here). */
		onready?: (engine: ObservatoryEngine) => void;
	}

	let { demo, seed, freezeFrame = null, onframe, onready }: Props = $props();

	let canvasEl: HTMLCanvasElement;
	let engine: ObservatoryEngine | null = null;
	let status = $state<EngineStatus>({ state: 'booting' });
	let unsubStatus: (() => void) | null = null;
	let resizeObserver: ResizeObserver | null = null;

	onMount(() => {
		engine = new ObservatoryEngine({
			canvas: canvasEl,
			demo,
			seed,
			freezeFrame,
			maxDpr: 2,
			onFrame: (frame, fps) => onframe?.(frame, fps)
		});
		unsubStatus = engine.onStatus((s) => (status = s));

		// Keep the drawing buffer in lockstep with layout size (DPR-clamped).
		resizeObserver = new ResizeObserver(() => engine?.resize());
		resizeObserver.observe(canvasEl);

		engine.start().then((ok) => {
			if (ok && engine) {
				engine.resize();
				onready?.(engine);
			}
		});
	});

	onDestroy(() => {
		unsubStatus?.();
		resizeObserver?.disconnect();
		engine?.dispose();
		engine = null;
	});
</script>

<!-- Full-bleed canvas: the living memory field (void #05060a is cleared on-GPU). -->
<canvas bind:this={canvasEl} class="observatory-canvas" aria-label="Vestige memory field"
></canvas>

{#if status.state === 'unsupported' || status.state === 'error'}
	<!-- Readable fallback — never a crash (spec §4 Increment 3 gate). -->
	<div class="fallback" role="alert">
		<div class="fallback-title">MEMORY FIELD OFFLINE</div>
		<div class="fallback-reason">{status.reason}</div>
		<div class="fallback-hint">
			WebGPU is required for the Observatory. Chrome 113+, Edge 113+, or Safari 18+ —
			the classic <a href="./graph">Graph view</a> works everywhere.
		</div>
	</div>
{/if}

<style>
	.observatory-canvas {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		display: block;
		background: #05060a;
	}

	/* Fallback panel follows the instrument-overlay grammar (§7.3):
	   SF-Mono, faint glow, floating on the void — not a grey error card. */
	.fallback {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.75rem;
		text-align: center;
		padding: 0 10%;
		font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace;
		pointer-events: auto;
	}

	.fallback-title {
		color: #5dcaa5;
		font-size: 0.95rem;
		letter-spacing: 0.28em;
		text-shadow: 0 0 20px rgba(93, 202, 165, 0.4);
	}

	.fallback-reason {
		color: #9fd0e4;
		font-size: 0.8rem;
		letter-spacing: 0.04em;
		opacity: 0.85;
	}

	.fallback-hint {
		color: #7c8a97;
		font-size: 0.75rem;
		max-width: 34rem;
		line-height: 1.6;
	}

	.fallback-hint a {
		color: #cfe9ff;
		text-decoration: underline;
		text-underline-offset: 3px;
	}
</style>

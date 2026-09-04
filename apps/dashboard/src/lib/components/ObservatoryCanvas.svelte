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
		/**
		 * Upper pixel-density budget for this organ. A bounded 3D receipt view
		 * remains crisp at 1.25 DPR while avoiding a four-times-larger HDR/bloom
		 * target on high-density displays.
		 */
		maxDpr?: number;
		/** Telemetry callback: loop frame + fps estimate. */
		onframe?: (frame: number, fps: number) => void;
		/** Fired when the engine is running (the route uploads the graph here). */
		onready?: (engine: ObservatoryEngine) => void;
	}

	let { demo, seed, freezeFrame = null, maxDpr = 2, onframe, onready }: Props = $props();

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
			maxDpr,
			onFrame: (frame, fps) => onframe?.(frame, fps)
		});
		let recovering = false;
		unsubStatus = engine.onStatus((s) => {
			status = s;
			// A device-loss recovery re-runs boot inside the same engine. The
			// owner must re-register its passes exactly as on first boot, so
			// re-fire onready when `running` follows `recovering`.
			if (s.state === 'recovering') recovering = true;
			if (s.state === 'running' && recovering && engine) {
				recovering = false;
				engine.resize();
				onready?.(engine);
			}
		});

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
<canvas
	bind:this={canvasEl}
	class="observatory-canvas"
	aria-label="Vestige 3D memory field"
	aria-describedby="webgpu-field-status"
></canvas>

{#if status.state === 'recovering'}
	<!-- The GPU device was lost (reset, driver update, backgrounding). The
	     engine re-acquires it with backoff; the field returns on its own. -->
	<div id="webgpu-field-status" class="fallback" role="status" aria-live="polite">
		<div class="fallback-title">GPU DEVICE LOST · RECOVERING</div>
		<div class="fallback-reason">
			Re-acquiring the graphics device (attempt {status.attempt} of 5). {status.reason}
		</div>
	</div>
{:else if status.state === 'unsupported' || status.state === 'error'}
	<!-- This is deliberately a truthful failure state, not a deceptive substitute
	     visual. Graph used to be advertised here as an SVG fallback, but it is
	     itself GPU-rendered in the current product. Navigation remains available
	     through the persistent shell; this surface never claims that a 3D field
	     has rendered when WebGPU could not be created. -->
	<div id="webgpu-field-status" class="fallback" role="alert">
		<div class="fallback-title">3D MEMORY FIELD UNAVAILABLE</div>
		<div class="fallback-reason">
			This browser or device could not create a WebGPU graphics context, so this
			visual field has not rendered.
		</div>
		<div class="fallback-hint">
			Your local memories have not been changed. Use the persistent navigation to
			continue in another tool, or open this view in a WebGPU-capable browser.
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
</style>

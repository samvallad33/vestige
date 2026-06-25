<script lang="ts">
	// Fullscreen living hero. Uses the REAL Memory Cinema WebGPU storm engine
	// (CinemaSandbox) where WebGPU is available, and falls back to the WebGL
	// NeuralFlow field otherwise. Never modifies Memory Cinema itself.
	import { onMount, onDestroy } from 'svelte';
	import * as THREE from 'three';
	import type { CinemaSandbox } from '$lib/graph/cinema/sandbox';
	import type { SemanticRole } from '$lib/graph/cinema/storm';
	import { NeuralFlow } from '$lib/landing/neuralFlow';

	interface Props {
		/** Deterministic seed so each visitor's hero is one of one. */
		seed?: number;
		reducedMotion?: boolean;
	}
	let { seed = 1234, reducedMotion = false }: Props = $props();

	let host = $state<HTMLDivElement | undefined>(undefined);
	let sandbox: CinemaSandbox | null = null;
	let fallback: NeuralFlow | null = null;
	let raf = 0;
	let usingWebGPU = $state(false);
	let last = 0;

	// Cinema's three storm "worlds". We cycle them slowly so the hero keeps
	// transforming (the explode -> reform look) without the full Cinema director.
	const ROLES: SemanticRole[] = ['anchor', 'connection', 'contradiction'];
	let roleIndex = 0;
	const ORIGIN = new THREE.Vector3(0, 0, 0);

	// pointer for the fallback's liquid cursor
	function onPointerMove(e: PointerEvent) {
		if (!fallback || !host) return;
		const r = host.getBoundingClientRect();
		const nx = ((e.clientX - r.left) / r.width) * 2 - 1;
		const ny = -(((e.clientY - r.top) / r.height) * 2 - 1);
		fallback.setCursor(nx, ny);
	}

	async function bootSandbox(): Promise<boolean> {
		if (!host) return false;
		try {
			const { CinemaSandbox, isWebGPUSupported } = await import('$lib/graph/cinema/sandbox');
			if (!isWebGPUSupported()) return false;
			sandbox = new CinemaSandbox(host);
			await sandbox.boot();
			// Kick the first world: the storm forms from its initial cloud.
			sandbox.transitionTo('anchor', ORIGIN, 'I', 0);
			if (!reducedMotion) sandbox.setFlythrough(0.45);
			return true;
		} catch (e) {
			console.warn('[hero] WebGPU sandbox unavailable, using WebGL fallback:', e);
			sandbox?.dispose();
			sandbox = null;
			return false;
		}
	}

	function bootFallback() {
		if (!host) return;
		fallback = new NeuralFlow(host, { seed, reducedMotion });
		window.addEventListener('pointermove', onPointerMove, { passive: true });
	}

	// camera orbit for the sandbox: slow auto-rotate so the storm breathes; the
	// sandbox re-clamps distance + looks at origin every frame.
	let camAngle = 0;
	function driveSandboxCamera(dt: number) {
		if (!sandbox) return;
		const cam = sandbox.cameraRef;
		camAngle += dt * 0.06 * (reducedMotion ? 0 : 1);
		const radius = 58;
		cam.position.set(Math.sin(camAngle) * radius, 14 + Math.sin(camAngle * 0.5) * 6, Math.cos(camAngle) * radius);
	}

	let roleTimer = 0;
	async function loop(now: number) {
		raf = requestAnimationFrame(loop);
		const dt = Math.min((now - last) / 1000, 0.05);
		last = now;

		if (sandbox) {
			driveSandboxCamera(dt);
			// cycle storm worlds every ~7s for the perpetual transform
			roleTimer += dt;
			if (roleTimer > 7 && !reducedMotion) {
				roleTimer = 0;
				roleIndex = (roleIndex + 1) % ROLES.length;
				sandbox.transitionTo(ROLES[roleIndex], ORIGIN, 'I', roleIndex);
			}
			try {
				await sandbox.render(dt);
			} catch {
				/* one bad frame must not kill the loop */
			}
		}
		// the WebGL fallback runs its own internal rAF, so nothing to do here
	}

	onMount(() => {
		(async () => {
			usingWebGPU = await bootSandbox();
			if (usingWebGPU) {
				last = performance.now();
				raf = requestAnimationFrame(loop);
			} else {
				bootFallback();
			}
		})();
	});

	onDestroy(() => {
		cancelAnimationFrame(raf);
		window.removeEventListener('pointermove', onPointerMove);
		sandbox?.dispose();
		fallback?.dispose();
		sandbox = null;
		fallback = null;
	});
</script>

<div class="hero-stage" bind:this={host} aria-hidden="true"></div>

<style>
	.hero-stage {
		position: fixed;
		inset: 0;
		z-index: 0;
		width: 100vw;
		height: 100vh;
		height: 100svh;
	}
	.hero-stage :global(canvas) {
		width: 100% !important;
		height: 100% !important;
		display: block;
	}
</style>

<script lang="ts">
	// Mounts the full-viewport NodeEngine (edge -> converge -> EXPLODE -> reform
	// -> loop) as the fullscreen landing hero. Reseeds per visitor.
	import { onMount, onDestroy } from 'svelte';
	import { NodeEngine } from '$lib/hero/nodeEngine';

	interface Props {
		seed?: number;
		reducedMotion?: boolean;
		onTextBeat?: (isText: boolean) => void;
	}
	let { seed = 1234, reducedMotion = false, onTextBeat }: Props = $props();

	let host = $state<HTMLDivElement | undefined>(undefined);
	let engine: NodeEngine | null = null;

	onMount(() => {
		if (host) {
			try {
				const qs = new URLSearchParams(window.location.search);
				const fp = qs.get('phase');
				const fs = qs.get('shape');
				const forcePhase = fp !== null ? parseFloat(fp) : undefined;
				const forceShape = fs !== null ? parseInt(fs, 10) : undefined;
				engine = new NodeEngine(host, { seed, reducedMotion, forcePhase, forceShape, onTextBeat });
			} catch (e) {
				console.warn('[hero] NodeEngine failed to boot:', e);
			}
		}
	});

	onDestroy(() => {
		engine?.dispose();
		engine = null;
	});
</script>

<div class="node-stage" bind:this={host} aria-hidden="true"></div>

<style>
	.node-stage {
		position: fixed;
		inset: 0;
		z-index: 0;
		width: 100vw;
		height: 100vh;
		height: 100svh;
	}
	.node-stage :global(canvas) {
		width: 100% !important;
		height: 100% !important;
		display: block;
	}
</style>

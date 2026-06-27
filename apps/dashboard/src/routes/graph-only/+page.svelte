<script lang="ts">
	import { onMount } from 'svelte';
	import LaunchEngineHost from '$lib/launch/LaunchEngineHost.svelte';

	// A stripped copy of the launch hero with ALL text/overlay removed — just the
	// raw WebGPU particle brain, full viewport. For demo GIFs / screenshots where
	// the graph is the only thing on screen.
	const heroSeed = 20260625;

	let mounted = $state(false);
	let prefersReducedMotion = $state(false);
	let shell = $state<HTMLElement | undefined>(undefined);

	onMount(() => {
		mounted = true;
		prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
	});
</script>

<svelte:head>
	<title>Vestige · memory that watches itself think</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<main class="graph-shell" bind:this={shell} aria-label="Vestige memory graph">
	<LaunchEngineHost
		seed={heroSeed}
		reducedMotion={prefersReducedMotion}
		syncTarget={shell}
		suppress={false}
	/>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #02030a;
	}
	:global(*) {
		box-sizing: border-box;
	}

	/* The engine writes --burst / --flash onto this each frame; static fallbacks so
	   the first frame is never NaN. Same shell contract as the launch hero, minus
	   every overlay child. */
	.graph-shell {
		--burst: 0;
		--flash: 0;
		position: relative;
		width: 100vw;
		min-height: 100vh;
		min-height: 100svh;
		overflow: hidden;
		background: #02030a;
		isolation: isolate;
	}
	@property --burst {
		syntax: '<number>';
		inherits: true;
		initial-value: 0;
	}
	@property --flash {
		syntax: '<number>';
		inherits: true;
		initial-value: 0;
	}
</style>

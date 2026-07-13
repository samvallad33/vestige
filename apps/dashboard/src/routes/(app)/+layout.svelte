<script lang="ts">
	import MobileNav from '$lib/components/MobileNav.svelte';
	let { children } = $props();
</script>

<!--
	Full-bleed positioned host for every organ route. The (app) routes render
	full-screen WebGPU canvases with `absolute inset-0` / `h-full`, which resolve
	against the nearest positioned ancestor with a real height box. Without this
	wrapper the parent is `display:contents` (zero-height), so `h-full` collapses
	to 0px and the canvas renders black (the /graph regression). `100dvh` +
	`relative` gives every route a stable full-viewport containing block.
-->
<div class="relative h-[100dvh] w-full overflow-hidden">
	{@render children()}
	<MobileNav />
</div>

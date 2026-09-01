<script lang="ts">
	// Mobile navigation is now the ONE registry-driven os-mobilebar in the root
	// +layout.svelte (which also handles ?frame=/?capture= hiding). The old
	// MobileNav is removed to avoid a duplicate mobile bar + a capture-mode leak.
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
</div>

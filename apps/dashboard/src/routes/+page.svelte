<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import LaunchPage from './launch/+page.svelte';

	// This standalone promo deploy serves the waitlist/launch page DIRECTLY at the
	// site root ("/"), prerendered, so visitors see the signup instantly with no
	// dashboard flash or router delay. Gated by VITE_ROOT_REDIRECT so the normal
	// dashboard/embedded build is unaffected: when it is set to '/launch' the root
	// IS the launch page; otherwise the root behaves like the dashboard home
	// (redirect to /graph) exactly as before.
	const promoRoot = (import.meta.env.VITE_ROOT_REDIRECT as string | undefined)?.trim() === '/launch';

	onMount(() => {
		if (!promoRoot) goto('/graph', { replaceState: true });
	});
</script>

{#if promoRoot}
	<LaunchPage />
{/if}

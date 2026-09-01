<script lang="ts">
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { HOME_ROUTE } from '$lib/os-routes';

	// The VestigeOS home is the Palace (the spatial launcher), NOT /graph. Use the
	// base-safe path so /dashboard/ lands on /dashboard/palace — a bare
	// goto('/graph') escapes the configured base and 404s on the deployed site.
	onMount(() => goto(`${base}${HOME_ROUTE}`, { replaceState: true }));
</script>

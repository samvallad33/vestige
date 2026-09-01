<script lang="ts">
	import { page } from '$app/stores';
	import { base } from '$app/paths';

	const missing = $derived($page.status === 404);
	const message = $derived($page.error?.message ?? '');
</script>

<svelte:head>
	<title>{missing ? 'Organ not found' : 'Dashboard error'} · Vestige</title>
</svelte:head>

<main class="void">
	<p class="kicker">{missing ? '404 · EMPTY FIELD' : `ERROR ${$page.status}`}</p>
	<h1>{missing ? 'This organ is not in the cortex.' : 'The instrument failed to boot.'}</h1>
	<p>
		{message ||
			(missing
				? 'The route you opened is not a Vestige organ. The field is still live — pick a real one.'
				: 'A dashboard surface threw. Your memories were not touched.')}
	</p>
	<nav>
		<a href="{base}/observatory">Open Observatory</a>
		<a href="{base}/memories">Open memory library</a>
	</nav>
</main>

<style>
	.void {
		min-height: 100dvh;
		display: grid;
		place-content: center;
		gap: 0.85rem;
		padding: 2rem;
		background: #020307;
		color: #eafffb;
		text-align: center;
	}
	.kicker {
		margin: 0;
		letter-spacing: 0.22em;
		font-size: 0.68rem;
		color: #7ff3e6;
	}
	h1 {
		margin: 0;
		font-size: clamp(1.6rem, 4vw, 2.6rem);
		letter-spacing: -0.04em;
	}
	p {
		max-width: 46ch;
		margin: 0 auto;
		color: #8aa9a5;
		line-height: 1.5;
	}
	nav {
		display: flex;
		justify-content: center;
		gap: 1.2rem;
		margin-top: 0.6rem;
	}
	a {
		color: #22c7de;
		text-decoration: none;
		font-size: 0.85rem;
	}
	a:hover {
		text-decoration: underline;
	}
</style>

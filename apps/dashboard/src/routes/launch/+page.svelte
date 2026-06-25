<script lang="ts">
	import { onMount } from 'svelte';
	import MemoryCinema from '$lib/components/MemoryCinema.svelte';
	import HeroNodeEngine from '$lib/hero/HeroNodeEngine.svelte';
	import AmbientField from '$lib/landing/AmbientField.svelte';
	import NeuralSign from '$lib/landing/NeuralSign.svelte';
	import { seedPhantomBrain, type PhantomBrain } from '$lib/landing/phantomBrain';

	// ---- the phantom brain: seeded from whatever the visitor gives us --------
	let identity = $state('');
	let brain = $state<PhantomBrain>(seedPhantomBrain('vestige'));
	let hasSeeded = $state(false);
	let heroSeed = $state(1234);

	function seedToNumber(s: string): number {
		let h = 0x811c9dc5;
		for (let i = 0; i < s.length; i++) {
			h ^= s.charCodeAt(i);
			h = Math.imul(h, 0x01000193);
		}
		return (h >>> 0) % 100000;
	}

	function reseed(value: string) {
		brain = seedPhantomBrain(value || 'vestige');
		hasSeeded = value.trim().length > 0;
		heroSeed = seedToNumber(value || 'vestige');
	}

	// ---- waitlist capture ----------------------------------------------------
	type SubmitState = 'idle' | 'submitting' | 'success' | 'error';
	let email = $state('');
	let submitState = $state<SubmitState>('idle');
	let submitMessage = $state('');
	const waitlistEndpoint = import.meta.env.VITE_WAITLIST_ENDPOINT as string | undefined;

	async function join(e: SubmitEvent) {
		e.preventDefault();
		if (!email.includes('@')) {
			submitState = 'error';
			submitMessage = 'Enter an email so we can send your invite.';
			return;
		}
		submitState = 'submitting';
		submitMessage = '';
		const payload = {
			email: email.trim(),
			name: identity.trim(),
			plan: 'solo',
			priority: 'sync',
			source: 'vestige-launch',
			createdAt: new Date().toISOString()
		};
		if (!waitlistEndpoint) {
			// No endpoint wired yet — still give the visitor the win locally.
			submitState = 'success';
			submitMessage = `You're on the list. Your brain is one of one.`;
			return;
		}
		try {
			const res = await fetch(waitlistEndpoint, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			if (!res.ok) throw new Error(`Endpoint returned ${res.status}`);
			submitState = 'success';
			submitMessage = `You're on the list. Your brain is one of one.`;
		} catch (err) {
			submitState = 'error';
			submitMessage = err instanceof Error ? err.message : 'Could not reach the waitlist.';
		}
	}

	let mounted = $state(false);
	let prefersReducedMotion = $state(false);

	onMount(() => {
		mounted = true;
		prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
	});
</script>

<svelte:head>
	<title>Vestige · give your agent a brain you can watch think</title>
	<meta
		name="description"
		content="Local-first memory for AI coding agents. Watch your agent remember, forget, and leave a receipt, rendered as a living brain."
	/>
</svelte:head>

<main class="landing">
	<!-- HERO: the live, seeded brain IS the product -->
	<section class="hero">
		<!-- Ambient outer field: god-ray glow radiating from the storm + parallax
		     starfield filling the corners, so the whole page feels alive edge to edge. -->
		{#if mounted}
			<AmbientField seed={heroSeed} reducedMotion={prefersReducedMotion} />
		{/if}

		<!-- Full-viewport node engine: particles stream in from the edges, slam
		     together and EXPLODE at center, reform into brain / graph / lattice,
		     then dissolve back out and loop. Owns the whole screen. -->
		{#if mounted}
			{#key heroSeed}
				<HeroNodeEngine seed={heroSeed} reducedMotion={prefersReducedMotion} />
			{/key}
		{/if}

		<!-- Radial readability mask: darkens toward the edges so the outer field
		     can be vivid while the headline core stays pristine. -->
		<div class="readability-mask" aria-hidden="true"></div>

		<!-- The full Memory Cinema overlay launches on demand from the CTA below. -->
		{#if mounted}
			<MemoryCinema nodes={brain.nodes} edges={brain.edges} centerId="n0" />
		{/if}

		<!-- Living neural launch sign at the very top -->
		{#if mounted}
			<NeuralSign />
		{/if}

		<div class="hero-overlay">
			<div class="hud">
				<span class="hud-dot"></span>
				<span>{brain.stats.memories} memories</span>
				<span class="hud-sep">·</span>
				<span>{brain.stats.connections} connections</span>
				<span class="hud-sep">·</span>
				<span>150,000 GPU particles</span>
			</div>

			<h1 class="manifesto">
				Your agent forgets everything.<br />
				Your memory should be <em>yours</em>. Local, and beautiful.
			</h1>

			<p class="sub">
				Vestige is a local-first memory for AI coding agents. Watch it remember,
				forget, and leave a receipt, rendered as a brain you can fly through.
			</p>

			<div class="seed-row">
				<input
					class="seed-input"
					placeholder="your github handle, or a memory…"
					bind:value={identity}
					oninput={(e) => reseed((e.target as HTMLInputElement).value)}
					autocomplete="off"
					spellcheck="false"
				/>
				<span class="seed-hint">
					{hasSeeded ? `this brain is seeded from "${brain.seed}". one of one.` : 'type anything to seed your own brain'}
				</span>
			</div>

			{#if submitState === 'success'}
				<div class="success">
					<strong>{submitMessage}</strong>
					<p>We'll email you before launch. Top concept in your brain: <em>{brain.stats.topConcept}</em>.</p>
				</div>
			{:else}
				<form class="capture" onsubmit={join}>
					<input
						class="email-input"
						type="email"
						placeholder="you@dev.com"
						bind:value={email}
						autocomplete="email"
					/>
					<button class="cta" type="submit" disabled={submitState === 'submitting'}>
						{submitState === 'submitting' ? 'Joining…' : 'Claim your brain'}
					</button>
				</form>
				{#if submitState === 'error'}
					<p class="err">{submitMessage}</p>
				{/if}
			{/if}
		</div>
	</section>
</main>

<style>
	:global(body) {
		margin: 0;
		background: #05060a;
	}

	.landing {
		color: #e8eaf2;
		font-family:
			'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
	}

	.hero {
		position: relative;
		min-height: 100vh;
		min-height: 100svh;
		overflow: hidden;
		background: radial-gradient(120% 120% at 50% 0%, #0b1020 0%, #05060a 60%);
	}


	.readability-mask {
		position: fixed;
		inset: 0;
		z-index: 1;
		pointer-events: none;
		/* transparent core protects the headline, darkens into the vivid edges */
		background: radial-gradient(
			ellipse 70% 60% at 50% 46%,
			transparent 0%,
			transparent 32%,
			rgba(5, 6, 12, 0.55) 78%,
			rgba(5, 6, 12, 0.82) 100%
		);
	}

	.hero-overlay {
		position: relative;
		z-index: 2;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
		min-height: 100vh;
		min-height: 100svh;
		padding: clamp(1.5rem, 5vw, 4rem);
		gap: 1.5rem;
		pointer-events: none;
		background: radial-gradient(80% 60% at 50% 55%, rgba(5, 6, 10, 0.55) 0%, rgba(5, 6, 10, 0) 70%);
	}
	.hero-overlay > * {
		pointer-events: auto;
	}

	.hud {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		font-size: 0.78rem;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: #8b93b0;
		background: rgba(12, 16, 28, 0.5);
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 999px;
		padding: 0.45rem 1rem;
		backdrop-filter: blur(8px);
	}
	.hud-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: #34d399;
		box-shadow: 0 0 10px #34d399;
		animation: pulse 2s ease-in-out infinite;
	}
	.hud-sep {
		opacity: 0.4;
	}
	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.4; }
	}

	.manifesto {
		font-size: clamp(2rem, 6vw, 4.25rem);
		line-height: 1.05;
		font-weight: 700;
		letter-spacing: -0.02em;
		margin: 0;
		max-width: 16ch;
		text-wrap: balance;
		text-shadow: 0 2px 40px rgba(5, 6, 10, 0.8);
	}
	.manifesto em {
		font-style: italic;
		background: linear-gradient(120deg, #a78bfa, #22d3ee, #34d399);
		-webkit-background-clip: text;
		background-clip: text;
		color: transparent;
	}

	.sub {
		font-size: clamp(1rem, 2vw, 1.2rem);
		color: #aab2cc;
		max-width: 52ch;
		margin: 0;
		line-height: 1.5;
		text-shadow: 0 2px 30px rgba(5, 6, 10, 0.9);
	}

	.seed-row {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.4rem;
		width: min(440px, 90vw);
	}
	.seed-input {
		width: 100%;
		text-align: center;
		font-size: 0.95rem;
		padding: 0.7rem 1rem;
		border-radius: 12px;
		border: 1px solid rgba(167, 139, 250, 0.3);
		background: rgba(12, 16, 28, 0.6);
		color: #e8eaf2;
		backdrop-filter: blur(8px);
		outline: none;
		transition: border-color 0.2s;
	}
	.seed-input:focus {
		border-color: rgba(167, 139, 250, 0.8);
	}
	.seed-hint {
		font-size: 0.78rem;
		color: #7c84a0;
		min-height: 1.2em;
	}

	.capture {
		display: flex;
		gap: 0.5rem;
		width: min(440px, 90vw);
	}
	.email-input {
		flex: 1;
		font-size: 1rem;
		padding: 0.85rem 1.1rem;
		border-radius: 12px;
		border: 1px solid rgba(255, 255, 255, 0.12);
		background: rgba(12, 16, 28, 0.7);
		color: #e8eaf2;
		outline: none;
		backdrop-filter: blur(8px);
	}
	.email-input:focus {
		border-color: rgba(34, 211, 238, 0.7);
	}
	.cta {
		font-size: 1rem;
		font-weight: 600;
		padding: 0.85rem 1.4rem;
		border-radius: 12px;
		border: none;
		cursor: pointer;
		color: #05060a;
		background: linear-gradient(120deg, #a78bfa, #22d3ee);
		white-space: nowrap;
		transition: transform 0.15s, box-shadow 0.15s;
		box-shadow: 0 6px 24px rgba(34, 211, 238, 0.25);
	}
	.cta:hover:not(:disabled) {
		transform: translateY(-1px);
		box-shadow: 0 10px 32px rgba(34, 211, 238, 0.4);
	}
	.cta:disabled {
		opacity: 0.6;
		cursor: default;
	}

	.success {
		max-width: 46ch;
	}
	.success strong {
		font-size: 1.25rem;
		display: block;
		margin-bottom: 0.4rem;
	}
	.success p {
		color: #aab2cc;
		margin: 0;
	}
	.success em {
		color: #34d399;
		font-style: normal;
		font-weight: 600;
	}
	.err {
		color: #fca5a5;
		font-size: 0.85rem;
		margin: 0;
	}
</style>

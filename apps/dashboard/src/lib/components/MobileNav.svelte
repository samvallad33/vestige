<script lang="ts">
	// ─────────────────────────────────────────────────────────────────────────
	// MobileNav — the ONE reliable navigation surface for phones.
	//
	// The organs are zero-DOM WebGPU. Their in-canvas nav rail is a desktop
	// hover-to-expand affordance that CANNOT open on a touchscreen (no hover), and
	// it only exists on RouteStage organs — ObservatoryStage / direct-canvas organs
	// (observatory, graph, memories, explore, palace) have no in-canvas nav at all.
	// A phone user would be stranded. This DOM bar is rendered by the (app) shell so
	// EVERY organ — every stage type, and even devices with NO WebGPU — gets the
	// same tap-to-navigate bar. It shows ONLY on narrow/touch viewports, so the
	// desktop experience stays pure WebGPU with zero DOM chrome.
	//
	// Routes mirror the curated COGNITIVE_OS_ROUTES set (single source of truth
	// shared with the in-canvas rail + palace hero set).
	// ─────────────────────────────────────────────────────────────────────────
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import { COGNITIVE_OS_ROUTES } from '$lib/observatory/nav/nav-layer';

	// Show only on coarse-pointer (touch) OR narrow viewports. Reactive to resize
	// and orientation so a desktop window shrunk narrow also gets the bar. Driven
	// entirely by matchMedia — nothing hardcoded to a device.
	let show = $state(false);
	let open = $state(false);

	function evaluate() {
		if (typeof window === 'undefined') return;
		const narrow = window.matchMedia('(max-width: 820px)').matches;
		const coarse = window.matchMedia('(pointer: coarse)').matches;
		const portrait = window.innerHeight > window.innerWidth;
		show = coarse || (narrow && portrait);
	}

	$effect(() => {
		evaluate();
		const onChange = () => evaluate();
		window.addEventListener('resize', onChange);
		window.addEventListener('orientationchange', onChange);
		return () => {
			window.removeEventListener('resize', onChange);
			window.removeEventListener('orientationchange', onChange);
		};
	});

	// Active-route detection against the curated set, honouring the base path.
	const activeHref = $derived.by(() => {
		const path = $page.url.pathname.replace(base, '') || '/';
		const match = COGNITIVE_OS_ROUTES.find((r) => path === r.href || path.endsWith(r.href));
		if (match) return match.href;
		if (path === '/' || path === '') return '/observatory';
		return null;
	});
</script>

{#if show}
	<!-- Floating pill that expands to the full organ list. Bottom-centre so it sits
	     in the thumb zone and never overlaps the top-of-page instrument text. -->
	<nav class="mobile-nav" class:open aria-label="Organs">
		{#if open}
			<div class="sheet" role="menu">
				{#each COGNITIVE_OS_ROUTES as route (route.href)}
					<a
						class="row"
						class:active={route.href === activeHref}
						href={`${base}${route.href}`}
						role="menuitem"
						onclick={() => (open = false)}
					>
						<span class="key">{route.shortcut ?? route.label.charAt(0)}</span>
						<span class="label">{route.label}</span>
					</a>
				{/each}
			</div>
		{/if}
		<button
			class="fab"
			aria-expanded={open}
			aria-label={open ? 'Close navigation' : 'Open navigation'}
			onclick={() => (open = !open)}
		>
			{#if open}
				<span class="fab-x">esc</span>
			{:else}
				<span class="fab-key">{COGNITIVE_OS_ROUTES.find((r) => r.href === activeHref)?.shortcut ?? '≡'}</span>
				<span class="fab-word">{COGNITIVE_OS_ROUTES.find((r) => r.href === activeHref)?.label ?? 'Menu'}</span>
			{/if}
		</button>
	</nav>
{/if}

<style>
	.mobile-nav {
		position: fixed;
		left: 0;
		right: 0;
		bottom: max(1rem, env(safe-area-inset-bottom));
		z-index: 60;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.6rem;
		pointer-events: none;
	}
	.mobile-nav > * {
		pointer-events: auto;
	}

	.fab {
		display: inline-flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.7rem 1.15rem;
		border-radius: 999px;
		border: 1px solid rgba(0, 245, 212, 0.4);
		background: rgba(4, 6, 12, 0.82);
		backdrop-filter: blur(14px);
		-webkit-backdrop-filter: blur(14px);
		color: #cfeee9;
		font: 600 0.95rem/1 ui-monospace, 'SF Mono', Menlo, monospace;
		letter-spacing: 0.06em;
		box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55), 0 0 22px rgba(0, 245, 212, 0.14);
		cursor: pointer;
	}
	.fab-key {
		display: inline-grid;
		place-items: center;
		width: 1.4rem;
		height: 1.4rem;
		border-radius: 0.4rem;
		background: rgba(0, 245, 212, 0.16);
		color: #00f5d4;
		font-size: 0.85rem;
	}
	.fab-word {
		text-transform: uppercase;
	}
	.fab-x {
		text-transform: uppercase;
		color: #00f5d4;
	}

	.sheet {
		width: min(88vw, 24rem);
		max-height: 60vh;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		padding: 0.4rem;
		border-radius: 1rem;
		border: 1px solid rgba(0, 245, 212, 0.22);
		background: rgba(4, 6, 12, 0.9);
		backdrop-filter: blur(18px);
		-webkit-backdrop-filter: blur(18px);
		box-shadow: 0 18px 50px rgba(0, 0, 0, 0.65);
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.8rem 0.9rem;
		border-radius: 0.7rem;
		color: #a9c7c2;
		text-decoration: none;
		font: 500 1rem/1 ui-monospace, 'SF Mono', Menlo, monospace;
		letter-spacing: 0.04em;
	}
	.row:active {
		background: rgba(0, 245, 212, 0.1);
	}
	.row.active {
		color: #00f5d4;
		background: rgba(0, 245, 212, 0.12);
	}
	.row .key {
		display: inline-grid;
		place-items: center;
		width: 1.55rem;
		height: 1.55rem;
		border-radius: 0.45rem;
		background: rgba(255, 255, 255, 0.06);
		color: inherit;
		font-size: 0.85rem;
	}
	.row.active .key {
		background: rgba(0, 245, 212, 0.18);
	}
	.row .label {
		text-transform: uppercase;
	}

	@media (prefers-reduced-motion: no-preference) {
		.sheet {
			animation: rise 0.16s ease-out;
		}
		@keyframes rise {
			from {
				opacity: 0;
				transform: translateY(8px);
			}
			to {
				opacity: 1;
				transform: translateY(0);
			}
		}
	}
</style>

<script lang="ts">
	import Icon, { type IconName } from './Icon.svelte';

	// ═══════════════════════════════════════════════════════════════════
	//  PAGE HEADER — the shared "alive" page title used on every route.
	// ───────────────────────────────────────────────────────────────────
	//  A drawn-on unique route icon in a glowing tile, an aurora-gradient
	//  title, an optional subtitle, and an optional right-aligned slot for
	//  live counts / actions. Replaces the flat `<h1>` each page had, giving
	//  the whole app one premium, consistent, animated masthead.
	// ═══════════════════════════════════════════════════════════════════
	interface Props {
		icon: IconName;
		title: string;
		subtitle?: string;
		/** Tailwind color token name for the icon tile accent (e.g. 'synapse'). */
		accent?: string;
		children?: import('svelte').Snippet;
	}
	let { icon, title, subtitle, accent = 'synapse', children }: Props = $props();
</script>

<header class="flex items-start justify-between gap-4 mb-6 enter">
	<div class="flex items-center gap-3.5 min-w-0">
		<div
			class="header-tile relative flex items-center justify-center w-11 h-11 rounded-xl shrink-0
				bg-{accent}/12 border border-{accent}/25 text-{accent}-glow"
		>
			<Icon name={icon} size={22} draw />
		</div>
		<div class="min-w-0">
			<h1 class="text-2xl font-bold text-aurora leading-tight text-balance">{title}</h1>
			{#if subtitle}
				<p class="text-sm text-dim mt-0.5 text-pretty">{subtitle}</p>
			{/if}
		</div>
	</div>
	{#if children}
		<div class="flex items-center gap-2 shrink-0 flex-wrap justify-end">
			{@render children()}
		</div>
	{/if}
</header>

<style>
	/* Soft outward glow that gently pulses, so the masthead icon reads as
	   "live" the moment the page lands. */
	.header-tile::after {
		content: '';
		position: absolute;
		inset: -1px;
		border-radius: inherit;
		box-shadow: 0 0 18px -2px currentColor;
		opacity: 0.35;
		pointer-events: none;
	}
	@media not (prefers-reduced-motion: reduce) {
		.header-tile::after {
			animation: tile-glow 4s ease-in-out infinite;
		}
		@keyframes tile-glow {
			0%, 100% { opacity: 0.22; }
			50% { opacity: 0.5; }
		}
	}

	.text-balance { text-wrap: balance; }
	.text-pretty { text-wrap: pretty; }
</style>

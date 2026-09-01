<script lang="ts">
	import Icon, { type IconName } from './Icon.svelte';

	// Static accent map — Tailwind cannot see `bg-{accent}/12` at compile time,
	// so every masthead was silently unstyled. These are doctrine hues, never purple.
	const ACCENT_TILE: Record<string, string> = {
		synapse: 'background: rgba(34, 199, 222, 0.12); border-color: rgba(34, 199, 222, 0.28); color: #7ff3e6;',
		recall: 'background: rgba(41, 242, 169, 0.12); border-color: rgba(41, 242, 169, 0.28); color: #29F2A9;',
		warning: 'background: rgba(255, 209, 102, 0.12); border-color: rgba(255, 209, 102, 0.28); color: #FFD166;',
		dream: 'background: rgba(168, 255, 94, 0.12); border-color: rgba(168, 255, 94, 0.28); color: #A8FF5E;',
		decay: 'background: rgba(255, 59, 48, 0.12); border-color: rgba(255, 59, 48, 0.28); color: #FF3B30;',
		memory: 'background: rgba(27, 214, 255, 0.12); border-color: rgba(27, 214, 255, 0.28); color: #1BD6FF;'
	};

	interface Props {
		icon: IconName;
		title: string;
		subtitle?: string;
		/** Token name for the icon tile accent (e.g. 'synapse'). */
		accent?: string;
		children?: import('svelte').Snippet;
	}
	let { icon, title, subtitle, accent = 'synapse', children }: Props = $props();
	const tileStyle = $derived(ACCENT_TILE[accent] ?? ACCENT_TILE.synapse);
</script>

<header class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 sm:gap-4 mb-6 enter">
	<div class="flex items-center gap-3.5 min-w-0">
		<div
			class="header-tile relative flex items-center justify-center w-11 h-11 rounded-xl shrink-0 border"
			style={tileStyle}
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
		<div class="flex items-center gap-2 shrink-0 flex-wrap sm:justify-end">
			{@render children()}
		</div>
	{/if}
</header>

<style>
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

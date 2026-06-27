<script lang="ts">
	import Icon, { type IconName } from './Icon.svelte';
	import { reveal } from '$lib/actions/reveal';

	let {
		variant = 'empty',          // 'empty' | 'offline' | 'error' | 'dormant'
		title,
		body,
		detail = '',                // raw error string, demoted into <details>
		actionLabel = '',
		onAction,
		compact = false
	}: {
		variant?: 'empty' | 'offline' | 'error' | 'dormant';
		title: string;
		body: string;
		detail?: string;
		actionLabel?: string;
		onAction?: () => void;
		compact?: boolean;
	} = $props();

	const glyph: Record<string, IconName> = {
		empty: 'graph',
		offline: 'activation',
		error: 'reasoning',
		dormant: 'dreams'
	};
	const tone: Record<string, string> = {
		empty: 'var(--color-synapse-glow)',
		offline: 'var(--color-warning)',
		error: 'var(--color-warning)',
		dormant: 'var(--color-dream-glow)'
	};
</script>

<div
	class="life-state glass edge-live"
	class:is-alarm={variant === 'error' || variant === 'offline'}
	class:is-live={variant === 'empty'}
	class:is-dream={variant === 'dormant'}
	class:compact
	data-variant={variant}
	style:--tone={tone[variant]}
	use:reveal
>
	{#if variant === 'error'}<span class="heal" aria-hidden="true"></span>{/if}
	<div class="halo" class:dormant-aura={variant === 'dormant'} aria-hidden="true"></div>

	<div class="glyph" class:breathe={variant !== 'dormant'} class:seek-host={variant === 'offline'}>
		<Icon name={glyph[variant]} size={compact ? 30 : 48} strokeWidth={1.2} />
	</div>

	<h2 class="title text-aurora veil">{title}</h2>
	<p class="body veil">{body}</p>

	{#if detail}
		<details class="detail veil-strong"><summary>Technical detail</summary><code>{detail}</code></details>
	{/if}
	{#if actionLabel && onAction}
		<button class="act" onclick={onAction}>{actionLabel}</button>
	{/if}
</div>

<style>
	.life-state {
		position: relative;
		isolation: isolate;
		overflow: hidden;
		display: flex;
		flex-direction: column;
		align-items: center;
		text-align: center;
		gap: 14px;
		padding: 48px 32px;
		border-radius: 16px;
		max-width: 32rem;
		margin: 8vh auto 0;
	}
	.life-state.compact {
		padding: 22px 18px;
		gap: 9px;
		margin: 0;
		max-width: none;
	}
	.halo {
		position: absolute;
		z-index: -1;
		width: 240px;
		height: 240px;
		top: 8px;
		border-radius: 50%;
		opacity: 0.5;
		background: radial-gradient(circle, color-mix(in oklab, var(--tone) 38%, transparent), transparent 70%);
	}
	.glyph {
		color: var(--tone);
		display: grid;
		place-items: center;
	}
	.title {
		font-size: 1.25rem;
		font-weight: 700;
		margin: 0;
	}
	.compact .title {
		font-size: 0.95rem;
	}
	.body {
		font-size: 0.86rem;
		line-height: 1.55;
		/* --color-text, not --color-dim: AA-legible over a bright field cluster */
		color: var(--color-text);
		max-width: 34ch;
		margin: 0;
	}
	.detail {
		font-size: 0.72rem;
		/* --color-dim, not --color-muted: muted failed AA even with the veil */
		color: var(--color-dim);
	}
	.detail summary {
		cursor: pointer;
	}
	.detail code {
		display: block;
		margin-top: 6px;
		padding: 8px 10px;
		border-radius: 8px;
		background: color-mix(in oklab, var(--color-decay) 8%, transparent);
		color: #fca5a5;
		word-break: break-all;
		text-align: left;
	}
	.act {
		margin-top: 6px;
		padding: 9px 18px;
		border-radius: 10px;
		font-size: 0.8rem;
		font-weight: 600;
		cursor: pointer;
		color: var(--color-synapse-glow);
		background: color-mix(in oklab, var(--color-synapse) 14%, transparent);
		border: 1px solid color-mix(in oklab, var(--color-synapse) 32%, transparent);
		transition: background 0.18s ease, transform 0.18s ease;
	}
	.act:hover {
		background: color-mix(in oklab, var(--color-synapse) 24%, transparent);
		transform: translateY(-1px);
	}
	.heal {
		position: absolute;
		inset-inline: 0;
		top: 0;
		height: 38%;
		z-index: -1;
		pointer-events: none;
		background: linear-gradient(180deg, transparent,
			color-mix(in oklab, var(--color-warning) 22%, transparent), transparent);
	}
	@media not (prefers-reduced-motion: reduce) {
		.heal { animation: heal-sweep 3.4s ease-in-out infinite; }
	}
	@media (prefers-reduced-motion: reduce) {
		.act:hover { transform: none; }
	}
</style>

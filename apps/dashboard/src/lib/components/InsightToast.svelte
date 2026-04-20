<!--
  InsightToast — v2.2 Pulse
  Renders the toast queue as a floating overlay. Desktop: bottom-right
  stack. Mobile: top-center stack. Each toast has a colored left border
  matching its event type, a progress bar showing dwell, and click-to-
  dismiss. This is the "brain coming alive" surface — when Vestige dreams,
  consolidates, forgets, or discovers a new bridge, you see it happen.
-->
<script lang="ts">
	import { toasts, type Toast } from '$stores/toast';
	import type { VestigeEventType } from '$types';

	const ICONS: Partial<Record<VestigeEventType, string>> = {
		DreamCompleted: '✦',
		ConsolidationCompleted: '◉',
		ConnectionDiscovered: '⟷',
		MemoryPromoted: '↑',
		MemoryDemoted: '↓',
		MemorySuppressed: '◬',
		MemoryUnsuppressed: '◉',
		Rac1CascadeSwept: '✺',
		MemoryDeleted: '✕',
	};

	function iconFor(type: VestigeEventType): string {
		return ICONS[type] ?? '◆';
	}

	function handleClick(t: Toast) {
		toasts.dismiss(t.id);
	}

	function handleKey(e: KeyboardEvent, t: Toast) {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			toasts.dismiss(t.id);
		}
	}
</script>

<div
	class="toast-layer"
	aria-live="polite"
	aria-atomic="false"
>
	{#each $toasts as t (t.id)}
		<button
			type="button"
			class="toast-item"
			aria-label="{t.title}: {t.body}. Click to dismiss."
			onclick={() => handleClick(t)}
			onkeydown={(e) => handleKey(e, t)}
			onmouseenter={() => toasts.pauseDwell(t.id, t.dwellMs)}
			onmouseleave={() => toasts.resumeDwell(t.id)}
			onfocus={() => toasts.pauseDwell(t.id, t.dwellMs)}
			onblur={() => toasts.resumeDwell(t.id)}
			style="--toast-color: {t.color}; --toast-dwell: {t.dwellMs}ms;"
		>
			<div class="toast-accent" aria-hidden="true"></div>
			<div class="toast-body">
				<div class="toast-head">
					<span class="toast-icon" aria-hidden="true">{iconFor(t.type)}</span>
					<span class="toast-title">{t.title}</span>
				</div>
				<div class="toast-sub">{t.body}</div>
			</div>
			<div class="toast-progress" aria-hidden="true">
				<div class="toast-progress-fill"></div>
			</div>
		</button>
	{/each}
</div>

<style>
	.toast-layer {
		position: fixed;
		z-index: 60;
		pointer-events: none;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		right: 1.25rem;
		bottom: 1.25rem;
		max-width: 22rem;
		width: calc(100vw - 2.5rem);
	}

	@media (max-width: 768px) {
		.toast-layer {
			right: 0.75rem;
			left: 0.75rem;
			bottom: auto;
			top: 0.75rem;
			max-width: none;
			width: auto;
			align-items: stretch;
		}
	}

	.toast-item {
		pointer-events: auto;
		position: relative;
		display: flex;
		gap: 0.75rem;
		align-items: stretch;
		text-align: left;
		font: inherit;
		color: inherit;
		background: rgba(12, 14, 22, 0.72);
		backdrop-filter: blur(14px) saturate(160%);
		-webkit-backdrop-filter: blur(14px) saturate(160%);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 0.75rem;
		padding: 0.75rem 0.9rem 0.75rem 0.5rem;
		overflow: hidden;
		box-shadow:
			0 10px 40px -12px rgba(0, 0, 0, 0.8),
			0 0 22px -6px var(--toast-color);
		cursor: pointer;
		animation: toast-in 0.32s cubic-bezier(0.16, 1, 0.3, 1);
		transform-origin: right center;
		transition: transform 0.15s ease, box-shadow 0.15s ease;
	}

	.toast-item:hover {
		transform: translateY(-1px) scale(1.015);
		box-shadow:
			0 14px 48px -12px rgba(0, 0, 0, 0.85),
			0 0 32px -4px var(--toast-color);
	}

	.toast-item:focus-visible {
		outline: 1px solid var(--toast-color);
		outline-offset: 2px;
	}

	.toast-accent {
		width: 3px;
		border-radius: 2px;
		background: var(--toast-color);
		box-shadow: 0 0 10px var(--toast-color);
		flex-shrink: 0;
	}

	.toast-body {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		flex: 1;
		min-width: 0;
	}

	.toast-head {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.toast-icon {
		color: var(--toast-color);
		font-size: 0.95rem;
		text-shadow: 0 0 8px var(--toast-color);
		line-height: 1;
		width: 1rem;
		display: inline-flex;
		justify-content: center;
	}

	.toast-title {
		color: #F5F5FA;
		font-size: 0.82rem;
		font-weight: 600;
		letter-spacing: 0.01em;
	}

	.toast-sub {
		color: #B0B6C4;
		font-size: 0.74rem;
		line-height: 1.35;
		padding-left: 1.5rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.toast-progress {
		position: absolute;
		left: 0;
		bottom: 0;
		height: 2px;
		width: 100%;
		background: rgba(255, 255, 255, 0.04);
	}

	.toast-progress-fill {
		height: 100%;
		background: var(--toast-color);
		opacity: 0.55;
		transform-origin: left center;
		animation: toast-progress var(--toast-dwell) linear forwards;
	}

	/* Hover Panic — freeze auto-dismiss while the user is engaged.
	 * Pairs with toasts.pauseDwell/resumeDwell on the JS side. */
	.toast-item:hover .toast-progress-fill,
	.toast-item:focus-visible .toast-progress-fill {
		animation-play-state: paused;
	}

	@keyframes toast-in {
		from {
			opacity: 0;
			transform: translateX(24px) scale(0.98);
		}
		to {
			opacity: 1;
			transform: translateX(0) scale(1);
		}
	}

	@media (max-width: 768px) {
		.toast-item {
			transform-origin: top center;
			animation: toast-in-mobile 0.3s cubic-bezier(0.16, 1, 0.3, 1);
		}
	}

	@keyframes toast-in-mobile {
		from {
			opacity: 0;
			transform: translateY(-12px) scale(0.98);
		}
		to {
			opacity: 1;
			transform: translateY(0) scale(1);
		}
	}

	@keyframes toast-progress {
		from { transform: scaleX(1); }
		to { transform: scaleX(0); }
	}

	@media (prefers-reduced-motion: reduce) {
		.toast-item {
			animation: none;
		}
		.toast-progress-fill {
			animation: none;
			transform: scaleX(0.5);
		}
	}
</style>

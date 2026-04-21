<!--
  ThemeToggle — closes GitHub issue #11.
  Small 30px icon button. Click cycles dark → light → auto → dark.
  Shows the current mode via icon + aria-label + tooltip. Smooth 200ms
  fade/scale crossfade between icons so the state change feels tactile.

  Theme overrides live in $stores/theme (injected stylesheet approach —
  app.css is never mutated). The button itself uses existing glass
  tokens so it drops cleanly into the sidebar/header.
-->
<script lang="ts">
	import { theme, cycleTheme, type Theme } from '$stores/theme';

	// Cycle order determines the label shown in the tooltip/aria.
	const LABELS: Record<Theme, string> = {
		dark: 'Dark',
		light: 'Light',
		auto: 'Auto (system)'
	};

	const NEXT: Record<Theme, Theme> = {
		dark: 'light',
		light: 'auto',
		auto: 'dark'
	};

	let current = $derived($theme);
	let nextMode = $derived(NEXT[current]);
	let ariaLabel = $derived(`Toggle theme: ${LABELS[current]} (click for ${LABELS[nextMode]})`);
</script>

<button
	type="button"
	class="theme-toggle"
	aria-label={ariaLabel}
	title={ariaLabel}
	onclick={cycleTheme}
	data-mode={current}
>
	<!-- Three SVG icons stacked, crossfade by opacity. Only the active
	     one is visible; aria-hidden on all since the button label
	     carries the semantics. -->
	<span class="icon-wrap">
		<!-- MOON (dark mode) -->
		<svg
			class="icon"
			class:active={current === 'dark'}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="1.75"
			stroke-linecap="round"
			stroke-linejoin="round"
			aria-hidden="true"
		>
			<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
		</svg>

		<!-- SUN (light mode) -->
		<svg
			class="icon"
			class:active={current === 'light'}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="1.75"
			stroke-linecap="round"
			stroke-linejoin="round"
			aria-hidden="true"
		>
			<circle cx="12" cy="12" r="4" />
			<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
		</svg>

		<!-- AUTO (half-moon with gear teeth) -->
		<svg
			class="icon"
			class:active={current === 'auto'}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="1.75"
			stroke-linecap="round"
			stroke-linejoin="round"
			aria-hidden="true"
		>
			<!-- Left half filled (dark side), right half outlined (light side) -->
			<circle cx="12" cy="12" r="8" />
			<path d="M12 4 A8 8 0 0 0 12 20 Z" fill="currentColor" stroke="none" />
			<!-- Tiny gear notches to signal 'system / automatic' -->
			<path d="M12 2v1.5M12 20.5V22M3.5 12H2M22 12h-1.5" />
		</svg>
	</span>
</button>

<style>
	.theme-toggle {
		width: 30px;
		height: 30px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		border-radius: 8px;
		background: rgba(99, 102, 241, 0.06);
		border: 1px solid rgba(99, 102, 241, 0.14);
		color: var(--color-text);
		cursor: pointer;
		transition:
			background 200ms ease,
			border-color 200ms ease,
			color 200ms ease,
			transform 120ms ease;
		-webkit-tap-highlight-color: transparent;
	}

	.theme-toggle:hover {
		background: rgba(99, 102, 241, 0.14);
		border-color: rgba(99, 102, 241, 0.3);
		color: var(--color-bright);
	}

	.theme-toggle:active {
		transform: scale(0.94);
	}

	.theme-toggle:focus-visible {
		outline: 1px solid var(--color-synapse);
		outline-offset: 2px;
	}

	.icon-wrap {
		position: relative;
		width: 18px;
		height: 18px;
		display: inline-block;
	}

	.icon {
		position: absolute;
		inset: 0;
		width: 18px;
		height: 18px;
		opacity: 0;
		transform: scale(0.7) rotate(-30deg);
		transition:
			opacity 200ms ease,
			transform 200ms cubic-bezier(0.16, 1, 0.3, 1);
		pointer-events: none;
	}

	.icon.active {
		opacity: 1;
		transform: scale(1) rotate(0deg);
	}

	/* Subtle mode-specific accent tint so the button itself reflects
	 * the active mode at a glance. */
	.theme-toggle[data-mode='dark'] {
		color: var(--color-synapse-glow, #818cf8);
	}
	.theme-toggle[data-mode='light'] {
		color: var(--color-warning, #f59e0b);
	}
	.theme-toggle[data-mode='auto'] {
		color: var(--color-dream-glow, #c084fc);
	}

	@media (prefers-reduced-motion: reduce) {
		.theme-toggle,
		.icon {
			transition: none;
		}
	}
</style>

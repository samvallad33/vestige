<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { goto, onNavigate } from '$app/navigation';
	import { base } from '$app/paths';
	import {
		websocket,
		isConnected,
		memoryCount,
		avgRetention,
		suppressedCount,
		uptimeSeconds,
		formatUptime,
	} from '$stores/websocket';
	import ForgettingIndicator from '$lib/components/ForgettingIndicator.svelte';
	import InsightToast from '$lib/components/InsightToast.svelte';
	import AmbientAwarenessStrip from '$lib/components/AmbientAwarenessStrip.svelte';
	import VerdictBar from '$lib/components/VerdictBar.svelte';
	import ThemeToggle from '$lib/components/ThemeToggle.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import { initTheme } from '$stores/theme';
	import { OS_ROUTES, DOCK_ROUTES, routesByGroup, HOME_ROUTE } from '$lib/os-routes';
	import { shell } from '$lib/stores/shell.svelte';

	let { children } = $props();
	let showCommandPalette = $state(false);
	let cmdQuery = $state('');
	let cmdInput = $state<HTMLInputElement>(undefined as unknown as HTMLInputElement);
	let dashboardPath = $derived(
		$page.url.pathname.startsWith(base) ? $page.url.pathname.slice(base.length) || '/' : $page.url.pathname
	);
	let isMarketingRoute = $derived(dashboardPath === '/waitlist' || dashboardPath.startsWith('/waitlist/'));
	// The organs are full-bleed WebGPU canvases. The OS shell (persistent dock +
	// ⌘K palette) is NOT the old flex-sidebar — it FLOATS over the canvas as an
	// overlay so it never fights the `fixed inset-0` organ layout. It shows on
	// every dashboard route so no canvas page is a navigation island (the launch
	// audit: 6 organs had no desktop nav, Palace had none at all).
	//
	// Recording cleanliness: ?capture=1 (or ?capture) AND ?frame=N (deterministic
	// still capture) both hide ALL chrome so hero footage / ?frame captures stay
	// pure canvas. The old MobileNav had no such gate and leaked into captures.
	let isCaptureMode = $derived(
		($page.url.searchParams.has('capture') && $page.url.searchParams.get('capture') !== '0') ||
			$page.url.searchParams.has('frame')
	);
	// Show the floating OS shell on real dashboard routes (not marketing, not capture).
	let showShell = $derived(!isMarketingRoute && !isCaptureMode);

	onMount(() => {
		// Live nervous system: every immersive organ consumes the WebSocket
		// ($isConnected, live event feed, birth/salience/firewall pulses). Only
		// true marketing/waitlist pages skip it. NOTE: this is deliberately NOT
		// gated on isImmersiveRoute — immersive is the whole point of connecting.
		// (The prior guard `!isMarketingRoute && !isImmersiveRoute` was always
		// false since isImmersiveRoute === !isMarketingRoute, so the socket never
		// connected on any route and the live system was dead everywhere.)
		if (!isMarketingRoute) {
			websocket.connect();
		}
		const teardownTheme = initTheme();

		function onKeyDown(e: KeyboardEvent) {
			// While a full-screen takeover (Memory Cinema, a demo) owns the keyboard,
			// the OS shell must NOT react to ⌘K/Escape — otherwise ⌘K opens the
			// palette behind the takeover and steals Escape so it can't be closed.
			// Cinema is PROTECTED, so we detect its active overlay read-only via the
			// DOM (.cinema-overlay mounts only while open) rather than editing it.
			if (isMarketingRoute || isCaptureMode || takeoverActive()) return;
			if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
				e.preventDefault();
				showCommandPalette = !showCommandPalette;
				cmdQuery = '';
				if (showCommandPalette) {
					requestAnimationFrame(() => cmdInput?.focus());
				}
				return;
			}
			if (e.key === 'Escape' && showCommandPalette) {
				showCommandPalette = false;
				return;
			}
			if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
			// Single-key navigation shortcuts — derived from the canonical registry
			// so there is ONE source of truth (no drift between dock/palette/keys).
			const target = SHORTCUT_MAP[e.key.toLowerCase()];
			if (target && !e.metaKey && !e.ctrlKey && !e.altKey) {
				e.preventDefault();
				goto(`${base}${target}`);
			}
		}

		window.addEventListener('keydown', onKeyDown);
		return () => {
			websocket.disconnect();
			window.removeEventListener('keydown', onKeyDown);
			teardownTheme();
		};
	});

	// Native View Transitions for client-side route navigation. Crossfades route
	// changes when supported; respects prefers-reduced-motion. This replaces the
	// old hand-rolled .animate-page-in keyframe on the route content wrapper.
	onNavigate((navigation) => {
		if (!document.startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
		return new Promise((resolve) => {
			document.startViewTransition(async () => {
				resolve();
				await navigation.complete;
			});
		});
	});

	// ── All nav surfaces derive from the ONE canonical registry (os-routes.ts) ──
	// Single-key shortcut map, built from the registry (no hand-maintained drift).
	const SHORTCUT_MAP: Record<string, string> = Object.fromEntries(
		OS_ROUTES.filter((r) => r.shortcut).map((r) => [r.shortcut!.toLowerCase(), r.href])
	);

	function isActive(href: string, currentPath: string): boolean {
		const path = currentPath.startsWith(base) ? currentPath.slice(base.length) || '/' : currentPath;
		if (href === HOME_ROUTE) return path === '/' || path === href;
		return path === href || path.startsWith(href + '/');
	}

	// The command palette searches label + purpose across ALL 20 organs, grouped.
	let paletteGroups = $derived(
		routesByGroup()
			.map((g) => ({
				group: g.group,
				routes: cmdQuery
					? g.routes.filter(
							(r) =>
								r.label.toLowerCase().includes(cmdQuery.toLowerCase()) ||
								r.purpose.toLowerCase().includes(cmdQuery.toLowerCase())
						)
					: g.routes
			}))
			.filter((g) => g.routes.length > 0)
	);
	let paletteFlat = $derived(paletteGroups.flatMap((g) => g.routes));

	function cmdNavigate(href: string) {
		showCommandPalette = false;
		cmdQuery = '';
		goto(`${base}${href}`);
	}

	// A full-screen takeover is active if either the shell store was raised OR a
	// protected Cinema overlay is mounted (read-only DOM probe — we never touch
	// the protected MemoryCinema component).
	function takeoverActive(): boolean {
		if (shell.overlayActive) return true;
		return typeof document !== 'undefined' && document.querySelector('.cinema-overlay') !== null;
	}

	// Dialog a11y: trap focus inside the palette while open, mark the rest of the
	// page inert, and restore focus to the element that opened it on close. A
	// Svelte action so it wires/tears-down with the palette's mount lifecycle.
	function paletteDialog(node: HTMLElement) {
		const opener = document.activeElement as HTMLElement | null;
		const appRoot = document.querySelector('[data-app-root]') as HTMLElement | null;
		appRoot?.setAttribute('inert', '');
		function focusables(): HTMLElement[] {
			return Array.from(
				node.querySelectorAll<HTMLElement>(
					'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
				)
			).filter((el) => el.offsetParent !== null);
		}
		function onKey(e: KeyboardEvent) {
			if (e.key !== 'Tab') return;
			const items = focusables();
			if (items.length === 0) return;
			const first = items[0];
			const last = items[items.length - 1];
			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
		node.addEventListener('keydown', onKey);
		return {
			destroy() {
				node.removeEventListener('keydown', onKey);
				appRoot?.removeAttribute('inert');
				opener?.focus?.();
			}
		};
	}
</script>

<!-- Organs always render FULL-BLEED (they own the viewport as fixed-inset
     WebGPU canvases). The OS shell floats OVER them, so it never fights the
     canvas layout the way the old flex-sidebar did. -->
<div data-app-root class="contents">
	{@render children()}
</div>

{#if showShell}
	<!-- ── Persistent desktop dock (floating, left edge) ──────────────────────
	     Every route can reach every organ + ⌘K. No canvas page is an island. -->
	<nav
		class="os-dock hidden md:flex"
		aria-label="VestigeOS navigation"
	>
		<a
			href="{base}{HOME_ROUTE}"
			class="os-dock-logo"
			title="Palace — home"
			aria-label="Palace, VestigeOS home"
		>
			<Icon name="logo" size={18} strokeWidth={1.8} />
		</a>

		<div class="os-dock-items">
			{#each DOCK_ROUTES as item}
				{@const active = isActive(item.href, $page.url.pathname)}
				<a
					href="{base}{item.href}"
					class="os-dock-link {active ? 'os-dock-active' : ''}"
					title="{item.label} — {item.purpose}"
					aria-current={active ? 'page' : undefined}
				>
					<span class="os-dock-icon"><Icon name={item.icon} size={20} /></span>
					<span class="os-dock-label">{item.label}</span>
					{#if item.shortcut}<span class="os-dock-key">{item.shortcut}</span>{/if}
				</a>
			{/each}

			<!-- Command / More — the doorway to all 20 organs. -->
			<button
				class="os-dock-link os-dock-command"
				onclick={() => { showCommandPalette = true; cmdQuery = ''; requestAnimationFrame(() => cmdInput?.focus()); }}
				title="Command palette (⌘K) — jump to any organ"
			>
				<span class="os-dock-icon"><Icon name="command" size={20} /></span>
				<span class="os-dock-label">Command</span>
				<span class="os-dock-key">⌘K</span>
			</button>
		</div>

		<div class="os-dock-footer">
			<div class="os-dock-status" title={$isConnected ? 'Live' : 'Offline'}>
				<span class="os-dot {$isConnected ? 'os-dot-live' : 'os-dot-off'}"></span>
				<span class="os-dock-label os-dock-status-text">{$isConnected ? 'Live' : 'Offline'}</span>
			</div>
			<div class="os-dock-theme"><ThemeToggle /></div>
		</div>
	</nav>

	<!-- ── Mobile bottom bar (primary organs + More→palette) ──────────────── -->
	<nav class="os-mobilebar md:hidden safe-bottom" aria-label="VestigeOS navigation">
		{#each DOCK_ROUTES.slice(0, 5) as item}
			{@const active = isActive(item.href, $page.url.pathname)}
			<a
				href="{base}{item.href}"
				class="os-mobile-link {active ? 'os-mobile-active' : ''}"
				aria-current={active ? 'page' : undefined}
			>
				<Icon name={item.icon} size={20} />
				<span class="os-mobile-label">{item.label}</span>
			</a>
		{/each}
		<button
			class="os-mobile-link"
			onclick={() => { showCommandPalette = true; cmdQuery = ''; requestAnimationFrame(() => cmdInput?.focus()); }}
			aria-label="More organs"
		>
			<Icon name="command" size={20} />
			<span class="os-mobile-label">More</span>
		</button>
	</nav>

	<!-- v2.2 Pulse — InsightToast overlay (floating, fixed) -->
	<InsightToast />
{/if}

<!-- ── Command palette — ALL 20 organs, grouped, searchable, everywhere ──── -->
{#if showCommandPalette && showShell && !shell.overlayActive}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-[100] flex items-start justify-center pt-[10vh] md:pt-[14vh] px-4 bg-void/70 backdrop-blur-md"
		onkeydown={(e) => { if (e.key === 'Escape') showCommandPalette = false; }}
		onclick={(e) => { if (e.target === e.currentTarget) showCommandPalette = false; }}
	>
		<div
			class="w-full max-w-xl glass-panel rounded-xl shadow-2xl shadow-synapse/10 overflow-hidden"
			role="dialog"
			aria-modal="true"
			aria-label="Command palette — jump to any organ"
			use:paletteDialog
		>
			<div class="flex items-center gap-3 px-4 py-3 border-b border-synapse/10">
				<span class="text-synapse"><Icon name="search" size={16} /></span>
				<input
					bind:this={cmdInput}
					bind:value={cmdQuery}
					type="text"
					placeholder="Jump to any organ…"
					class="flex-1 bg-transparent text-text text-sm placeholder:text-muted focus:outline-none"
					onkeydown={(e) => {
						if (e.key === 'Enter' && paletteFlat.length > 0) cmdNavigate(paletteFlat[0].href);
					}}
				/>
				<span class="text-[10px] text-muted font-mono bg-white/[0.04] px-1.5 py-0.5 rounded">esc</span>
			</div>
			<div class="max-h-[60vh] overflow-y-auto py-1">
				{#each paletteGroups as grp}
					<div class="px-4 pt-3 pb-1 text-[10px] uppercase tracking-[0.16em] text-muted/70 font-mono">{grp.group}</div>
					{#each grp.routes as item}
						<button
							onclick={() => cmdNavigate(item.href)}
							class="w-full flex items-center gap-3 px-4 py-2 text-left text-sm text-dim hover:text-text hover:bg-white/[0.05] transition"
						>
							<span class="w-5 flex justify-center text-synapse/80"><Icon name={item.icon} size={17} /></span>
							<span class="flex-1 min-w-0">
								<span class="block">{item.label}</span>
								<span class="block text-[11px] text-muted/60 truncate">{item.purpose}</span>
							</span>
							{#if item.shortcut}<span class="ml-auto text-[10px] text-muted/50 font-mono hidden md:block">{item.shortcut}</span>{/if}
						</button>
					{/each}
				{/each}
				{#if paletteFlat.length === 0}
					<div class="px-4 py-6 text-center text-sm text-muted">No matches</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.safe-bottom {
		padding-bottom: env(safe-area-inset-bottom, 0px);
	}

	/* ── Floating desktop dock ─────────────────────────────────────────────
	   Overlays the full-bleed canvas at the left edge. Collapsed to icons by
	   default; expands to labels on hover so it never steals canvas real
	   estate but stays one glance away. */
	.os-dock {
		position: fixed;
		top: 50%;
		left: 0.75rem;
		transform: translateY(-50%);
		z-index: 60;
		flex-direction: column;
		align-items: stretch;
		gap: 0.35rem;
		max-height: calc(100dvh - 1.5rem);
		padding: 0.5rem 0.4rem;
		border-radius: 1rem;
		border: 1px solid rgba(129, 140, 248, 0.16);
		background: rgba(6, 8, 14, 0.72);
		backdrop-filter: blur(16px);
		-webkit-backdrop-filter: blur(16px);
		box-shadow: 0 10px 40px rgba(0, 0, 0, 0.55);
		width: 3.4rem;
		transition: width 0.18s ease;
		overflow: hidden;
	}
	.os-dock:hover,
	.os-dock:focus-within {
		width: 13.5rem;
	}
	.os-dock-logo {
		display: grid;
		place-items: center;
		width: 2.6rem;
		height: 2.6rem;
		margin: 0 auto 0.35rem;
		border-radius: 0.7rem;
		background: linear-gradient(135deg, var(--dream, #818cf8), var(--synapse, #6366f1));
		color: #fff;
		flex-shrink: 0;
		box-shadow: 0 0 18px rgba(99, 102, 241, 0.35);
		transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
	}
	.os-dock-logo:hover {
		transform: rotate(-6deg) scale(1.06);
	}
	.os-dock-items {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		overflow-y: auto;
		min-height: 0;
		flex: 1;
	}
	.os-dock-link {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		width: 100%;
		padding: 0.55rem 0.6rem;
		border-radius: 0.6rem;
		color: #9aa7c2;
		font-size: 0.85rem;
		white-space: nowrap;
		border: 1px solid transparent;
		background: none;
		cursor: pointer;
		text-align: left;
	}
	.os-dock-link:hover {
		color: #e8ecf1;
		background: rgba(255, 255, 255, 0.04);
	}
	.os-dock-active {
		color: #a5b4fc;
		background: rgba(99, 102, 241, 0.15);
		border-color: rgba(99, 102, 241, 0.3);
	}
	.os-dock-icon {
		width: 1.4rem;
		display: grid;
		place-items: center;
		flex-shrink: 0;
	}
	.os-dock-active .os-dock-icon :global(svg) {
		filter: drop-shadow(0 0 6px rgba(129, 140, 248, 0.6));
	}
	.os-dock-label {
		flex: 1;
		opacity: 0;
		transition: opacity 0.12s ease;
	}
	.os-dock:hover .os-dock-label,
	.os-dock:focus-within .os-dock-label {
		opacity: 1;
	}
	.os-dock-key {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.65rem;
		color: rgba(154, 167, 194, 0.5);
		opacity: 0;
	}
	.os-dock:hover .os-dock-key,
	.os-dock:focus-within .os-dock-key {
		opacity: 1;
	}
	.os-dock-command {
		margin-top: 0.25rem;
		border-top: 1px solid rgba(129, 140, 248, 0.1);
		border-radius: 0.6rem;
		color: #7c8aa8;
	}
	.os-dock-footer {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.4rem 0.6rem 0.1rem;
		margin-top: 0.3rem;
		border-top: 1px solid rgba(129, 140, 248, 0.1);
	}
	.os-dock-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.72rem;
		color: #7c8aa8;
	}
	.os-dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 999px;
		flex-shrink: 0;
	}
	.os-dot-live {
		background: #29f2a9;
		box-shadow: 0 0 8px #29f2a9;
	}
	.os-dot-off {
		background: #ff6b6b;
	}
	.os-dock-status-text {
		opacity: 0;
	}
	.os-dock:hover .os-dock-status-text,
	.os-dock:focus-within .os-dock-status-text {
		opacity: 1;
	}
	.os-dock-theme {
		margin-left: auto;
		opacity: 0;
	}
	.os-dock:hover .os-dock-theme,
	.os-dock:focus-within .os-dock-theme {
		opacity: 1;
	}

	/* ── Mobile bottom bar ───────────────────────────────────────────────── */
	.os-mobilebar {
		position: fixed;
		bottom: 0;
		left: 0;
		right: 0;
		z-index: 60;
		display: flex;
		align-items: center;
		justify-content: space-around;
		padding: 0.3rem 0.4rem;
		border-top: 1px solid rgba(129, 140, 248, 0.14);
		background: rgba(6, 8, 14, 0.9);
		backdrop-filter: blur(14px);
		-webkit-backdrop-filter: blur(14px);
	}
	.os-mobile-link {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		min-width: 3.2rem;
		min-height: 2.75rem;
		padding: 0.35rem 0.5rem;
		border-radius: 0.6rem;
		color: #7c8aa8;
		background: none;
		border: none;
		cursor: pointer;
	}
	.os-mobile-active {
		color: #a5b4fc;
	}
	.os-mobile-label {
		font-size: 0.6rem;
	}

	@media (prefers-reduced-motion: reduce) {
		.os-dock,
		.os-dock-label,
		.os-dock-key,
		.os-dock-logo,
		.os-dock-theme,
		.os-dock-status-text {
			transition: none;
		}
	}
</style>

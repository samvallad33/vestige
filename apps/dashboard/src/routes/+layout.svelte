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
	import Icon, { type IconName } from '$lib/components/Icon.svelte';
	import { initTheme } from '$stores/theme';

	let { children } = $props();
	let showCommandPalette = $state(false);
	let cmdQuery = $state('');
	let cmdInput = $state<HTMLInputElement>(undefined as unknown as HTMLInputElement);
	let dashboardPath = $derived(
		$page.url.pathname.startsWith(base) ? $page.url.pathname.slice(base.length) || '/' : $page.url.pathname
	);
	let isMarketingRoute = $derived(dashboardPath === '/waitlist' || dashboardPath.startsWith('/waitlist/'));

	onMount(() => {
		if (!isMarketingRoute) {
			websocket.connect();
		}
		const teardownTheme = initTheme();

		function onKeyDown(e: KeyboardEvent) {
			if (isMarketingRoute) return;
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
			if (e.key === '/') {
				e.preventDefault();
				const searchInput = document.querySelector<HTMLInputElement>('input[type="text"]');
				searchInput?.focus();
				return;
			}
			// Single-key navigation shortcuts
			const shortcutMap: Record<string, string> = {
				g: '/graph', m: '/memories', t: '/timeline', f: '/feed',
				e: '/explore', i: '/intentions', s: '/stats',
				r: '/reasoning', a: '/activation', d: '/dreams',
				c: '/schedule', p: '/importance', u: '/duplicates',
				x: '/contradictions', n: '/patterns',
			};
			const target = shortcutMap[e.key.toLowerCase()];
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

	// Each nav item carries a UNIQUE semantic icon (see Icon.svelte). The old
	// set reused the same Unicode glyph across multiple items; every entry here
	// now has a distinct silhouette that reads instantly.
	const nav: { href: string; label: string; icon: IconName; shortcut: string }[] = [
		{ href: '/blackbox', label: 'Black Box', icon: 'blackbox', shortcut: 'B' },
		{ href: '/memory-prs', label: 'Memory PRs', icon: 'memorypr', shortcut: 'Q' },
		{ href: '/graph', label: 'Graph', icon: 'graph', shortcut: 'G' },
		{ href: '/reasoning', label: 'Reasoning', icon: 'reasoning', shortcut: 'R' },
		{ href: '/memories', label: 'Memories', icon: 'memories', shortcut: 'M' },
		{ href: '/timeline', label: 'Timeline', icon: 'timeline', shortcut: 'T' },
		{ href: '/feed', label: 'Feed', icon: 'feed', shortcut: 'F' },
		{ href: '/explore', label: 'Explore', icon: 'explore', shortcut: 'E' },
		{ href: '/activation', label: 'Activation', icon: 'activation', shortcut: 'A' },
		{ href: '/dreams', label: 'Dreams', icon: 'dreams', shortcut: 'D' },
		{ href: '/schedule', label: 'Schedule', icon: 'schedule', shortcut: 'C' },
		{ href: '/importance', label: 'Importance', icon: 'importance', shortcut: 'P' },
		{ href: '/duplicates', label: 'Duplicates', icon: 'duplicates', shortcut: 'U' },
		{ href: '/contradictions', label: 'Contradictions', icon: 'contradictions', shortcut: 'X' },
		{ href: '/patterns', label: 'Patterns', icon: 'patterns', shortcut: 'N' },
		{ href: '/intentions', label: 'Intentions', icon: 'intentions', shortcut: 'I' },
		{ href: '/stats', label: 'Stats', icon: 'stats', shortcut: 'S' },
		{ href: '/settings', label: 'Settings', icon: 'settings', shortcut: ',' },
	];

	// Mobile nav shows top 5 items
	const mobileNav = nav.slice(0, 5);

	function isActive(href: string, currentPath: string): boolean {
		const path = currentPath.startsWith(base) ? currentPath.slice(base.length) || '/' : currentPath;
		if (href === '/graph') return path === '/' || path === '/graph';
		return path.startsWith(href);
	}

	let filteredNav = $derived(
		cmdQuery
			? nav.filter(n => n.label.toLowerCase().includes(cmdQuery.toLowerCase()))
			: nav
	);

	function cmdNavigate(href: string) {
		showCommandPalette = false;
		cmdQuery = '';
		goto(`${base}${href}`);
	}
</script>

{#if isMarketingRoute}
	{@render children()}
{:else}
	<!-- Ambient background orbs -->
	<div class="ambient-orb ambient-orb-1" aria-hidden="true"></div>
	<div class="ambient-orb ambient-orb-2" aria-hidden="true"></div>
	<div class="ambient-orb ambient-orb-3" aria-hidden="true"></div>

	<!-- Desktop: sidebar + content -->
	<!-- Mobile: content + bottom nav -->
	<div class="flex flex-col md:flex-row h-screen overflow-hidden bg-void relative z-[1]">
		<!-- Desktop Sidebar (hidden on mobile) -->
		<nav class="hidden md:flex w-16 lg:w-56 flex-shrink-0 glass-sidebar flex-col">
			<!-- Logo -->
			<a href="{base}/graph" class="logo-link flex items-center gap-3 px-4 py-5 border-b border-synapse/10">
				<div class="logo-mark w-8 h-8 rounded-lg bg-gradient-to-br from-dream to-synapse flex items-center justify-center text-bright shadow-lg shadow-synapse/20">
					<Icon name="logo" size={18} strokeWidth={1.8} />
				</div>
				<span class="hidden lg:block text-sm font-semibold text-bright tracking-[0.18em]">VESTIGE</span>
			</a>

			<!-- Nav items -->
			<div class="flex-1 min-h-0 overflow-y-auto py-3 flex flex-col gap-1 px-2">
				{#each nav as item}
					{@const active = isActive(item.href, $page.url.pathname)}
					<a
						href="{base}{item.href}"
						class="nav-link group flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 text-sm
							{active
								? 'bg-synapse/15 text-synapse-glow border border-synapse/30 shadow-[0_0_12px_rgba(99,102,241,0.15)] nav-active-border'
								: 'text-dim hover:text-text hover:bg-white/[0.03] border border-transparent'}"
					>
						<span class="nav-icon w-5 flex justify-center transition-transform duration-200 group-hover:scale-110">
							<Icon name={item.icon} size={18} />
						</span>
						<span class="hidden lg:block">{item.label}</span>
						<span class="hidden lg:block ml-auto text-[10px] text-muted/50 font-mono">{item.shortcut}</span>
					</a>
				{/each}
			</div>

			<!-- Quick action -->
			<div class="px-2 pb-2">
				<button
					onclick={() => { showCommandPalette = true; cmdQuery = ''; requestAnimationFrame(() => cmdInput?.focus()); }}
					class="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-muted hover:text-dim hover:bg-white/[0.03] transition border border-subtle/15"
				>
					<Icon name="command" size={14} />
					<span class="hidden lg:block">Command</span>
					<span class="hidden lg:block ml-auto text-[10px] font-mono bg-white/[0.04] px-1.5 py-0.5 rounded">⌘K</span>
				</button>
			</div>

			<!-- Status footer -->
			<div class="px-3 py-4 border-t border-synapse/10 space-y-2">
				<div class="flex items-center gap-2 text-xs">
					<div class="w-2 h-2 rounded-full {$isConnected ? 'bg-recall animate-pulse-glow' : 'bg-decay'}"></div>
					<span class="hidden lg:block text-dim">{$isConnected ? 'Connected' : 'Offline'}</span>
					<div class="ml-auto">
						<ThemeToggle />
					</div>
				</div>
				<div class="hidden lg:block text-xs text-muted space-y-0.5">
					<div>{$memoryCount} memories</div>
					<div>{($avgRetention * 100).toFixed(0)}% retention</div>
					<!-- v2.0.7: surface uptime_secs from the Heartbeat event. Fires
						 every 30s so this self-refreshes. "up 3d 4h" format. -->
					{#if $uptimeSeconds > 0}
						<div title="MCP server uptime">up {formatUptime($uptimeSeconds)}</div>
					{/if}
				</div>
				{#if $suppressedCount > 0}
					<div class="hidden lg:block pt-1">
						<ForgettingIndicator />
					</div>
				{/if}
			</div>
		</nav>

		<!-- Main content -->
		<main class="flex-1 flex flex-col min-h-0 pb-16 md:pb-0">
			<AmbientAwarenessStrip />
			<VerdictBar />
			<div class="flex-1 min-h-0 overflow-y-auto">
				{@render children()}
			</div>
		</main>

		<!-- Mobile Bottom Nav (hidden on desktop) -->
		<nav class="md:hidden fixed bottom-0 inset-x-0 glass border-t border-synapse/10 z-40 safe-bottom">
			<div class="flex items-center justify-around px-2 py-1">
				{#each mobileNav as item}
					{@const active = isActive(item.href, $page.url.pathname)}
					<a
						href="{base}{item.href}"
						class="flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg transition-all min-w-[3.5rem]
							{active ? 'text-synapse-glow' : 'text-muted'}"
					>
						<Icon name={item.icon} size={20} />
						<span class="text-[9px]">{item.label}</span>
					</a>
				{/each}
				<!-- More button opens command palette on mobile -->
				<button
					onclick={() => { showCommandPalette = true; cmdQuery = ''; requestAnimationFrame(() => cmdInput?.focus()); }}
					class="flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg text-muted min-w-[3.5rem]"
				>
					<span class="text-lg">⋯</span>
					<span class="text-[9px]">More</span>
				</button>
			</div>
		</nav>
	</div>

	<!-- v2.2 Pulse — InsightToast overlay (floating, fixed) -->
	<InsightToast />
{/if}

<!-- Command Palette overlay -->
{#if showCommandPalette && !isMarketingRoute}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] md:pt-[15vh] px-4 bg-void/60 backdrop-blur-sm"
		onkeydown={(e) => { if (e.key === 'Escape') showCommandPalette = false; }}
		onclick={(e) => { if (e.target === e.currentTarget) showCommandPalette = false; }}
	>
		<div class="w-full max-w-lg glass-panel rounded-xl shadow-2xl shadow-synapse/10 overflow-hidden">
			<div class="flex items-center gap-3 px-4 py-3 border-b border-synapse/10">
				<span class="text-synapse"><Icon name="search" size={16} /></span>
				<input
					bind:this={cmdInput}
					bind:value={cmdQuery}
					type="text"
					placeholder="Navigate to..."
					class="flex-1 bg-transparent text-text text-sm placeholder:text-muted focus:outline-none"
					onkeydown={(e) => {
						if (e.key === 'Enter' && filteredNav.length > 0) {
							cmdNavigate(filteredNav[0].href);
						}
					}}
				/>
				<span class="text-[10px] text-muted font-mono bg-white/[0.04] px-1.5 py-0.5 rounded">esc</span>
			</div>
			<div class="max-h-72 overflow-y-auto py-1">
				{#each filteredNav as item}
					<button
						onclick={() => cmdNavigate(item.href)}
						class="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-dim hover:text-text hover:bg-white/[0.04] transition"
					>
						<span class="w-5 flex justify-center"><Icon name={item.icon} size={17} /></span>
						<span>{item.label}</span>
						<span class="ml-auto text-[10px] text-muted/50 font-mono hidden md:block">{item.shortcut}</span>
					</button>
				{/each}
				{#if filteredNav.length === 0}
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

	/* Logo breathes a faint synapse glow on hover — the mark feels live. */
	.logo-mark {
		transition:
			transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1),
			box-shadow 0.3s ease;
	}
	.logo-link:hover .logo-mark {
		transform: rotate(-6deg) scale(1.08);
		box-shadow:
			0 0 0 1px rgba(129, 140, 248, 0.4),
			0 0 22px rgba(99, 102, 241, 0.5);
	}

	/* The active nav item's icon picks up a soft drop-shadow glow so the
	   current location reads at a glance even in the collapsed (icon-only)
	   sidebar. */
	.nav-link.text-synapse-glow .nav-icon :global(svg),
	.nav-active-border .nav-icon :global(svg) {
		filter: drop-shadow(0 0 6px rgba(129, 140, 248, 0.55));
	}
</style>

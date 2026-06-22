<script lang="ts" module>
	// ═══════════════════════════════════════════════════════════════════
	//  VESTIGE ICON SYSTEM
	// ───────────────────────────────────────────────────────────────────
	//  Hand-tuned inline SVG icons — ZERO runtime dependency. Every nav
	//  item gets a UNIQUE, semantically-meaningful silhouette that reads
	//  instantly even at 18px. All strokes use `currentColor` so a single
	//  parent `color` (or our glow utilities) themes the whole set.
	//
	//  Design language: 24×24 viewBox, 1.6 stroke, round caps/joins, a
	//  cosmic/neural motif. Distinct silhouettes are the contract — no two
	//  icons may be confused at a glance (the old Unicode set reused
	//  ◎ ◈ ◉ ◷ across multiple items; that bug is dead here).
	// ═══════════════════════════════════════════════════════════════════
	export type IconName =
		| 'blackbox'
		| 'memorypr'
		| 'graph'
		| 'reasoning'
		| 'memories'
		| 'timeline'
		| 'feed'
		| 'explore'
		| 'activation'
		| 'dreams'
		| 'schedule'
		| 'importance'
		| 'duplicates'
		| 'contradictions'
		| 'patterns'
		| 'intentions'
		| 'stats'
		| 'settings'
		| 'command'
		| 'search'
		| 'filter'
		| 'sparkle'
		| 'chevron'
		| 'close'
		| 'pulse'
		| 'logo';

	// Each entry is the inner markup of a 24×24 SVG. Strokes inherit
	// currentColor; fills are explicit where a solid accent reads better.
	export const ICON_PATHS: Record<IconName, string> = {
		// Flight recorder — a radar-pulse sweep inside a recorder box.
		blackbox: `<rect x="3.5" y="6" width="17" height="12" rx="2.2"/><circle cx="12" cy="12" r="3.4"/><path d="M12 12 14.6 9.4" /><circle cx="12" cy="12" r="0.9" fill="currentColor" stroke="none"/><path d="M7 19.5h10" opacity=".5"/>`,
		// Git-branch with a review check — approve changes to the brain.
		memorypr: `<circle cx="6.5" cy="6" r="2"/><circle cx="6.5" cy="18" r="2"/><circle cx="17" cy="9" r="2"/><path d="M6.5 8v8M6.5 12h6.2A2.3 2.3 0 0 0 15 9.7V11"/><path d="m14.8 17 1.5 1.6 3-3.4" fill="none"/>`,
		// Connected nodes — a literal knowledge graph.
		graph: `<circle cx="6" cy="7" r="2.1"/><circle cx="18" cy="6" r="2.1"/><circle cx="12" cy="17.5" r="2.3"/><path d="M7.7 8.4 10.6 15.5M16.4 7.6 13.2 15.6M8 7l8-1"/>`,
		// Branching logic tree with a spark — deduction.
		reasoning: `<path d="M12 3v4M12 7 7 11M12 7l5 4"/><circle cx="12" cy="3" r="1.6"/><circle cx="7" cy="12" r="1.8"/><circle cx="17" cy="12" r="1.8"/><path d="m7 16 .9 1.9 1.9.6-1.4 1.5.2 2-1.6-1-1.6 1 .2-2L4.2 18.5l1.9-.6z" fill="currentColor" stroke="none" opacity=".55"/>`,
		// Stacked memory cards / layered recall.
		memories: `<rect x="5" y="8" width="14" height="11" rx="2.2"/><path d="M7.5 8V6.4A1.4 1.4 0 0 1 8.9 5h8.2A1.9 1.9 0 0 1 19 7v8.5"/><path d="M9 12.5h6M9 15.5h4"/>`,
		// Time axis with marked moments.
		timeline: `<path d="M3 12h18"/><circle cx="7" cy="12" r="2" fill="currentColor" stroke="none"/><circle cx="13" cy="12" r="2.2"/><circle cx="19" cy="12" r="1.6" fill="currentColor" stroke="none"/><path d="M7 12V7M13 12v6M19 12V9"/>`,
		// Streaming feed lines.
		feed: `<path d="M4 7h16M4 12h16M4 17h10"/><circle cx="18.5" cy="17" r="1.4" fill="currentColor" stroke="none"/>`,
		// Compass — exploration.
		explore: `<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z" fill="currentColor" stroke="none" opacity=".8"/><circle cx="12" cy="12" r="1" fill="var(--color-void,#050510)" stroke="none"/>`,
		// Radiating activation — spreading energy from a core.
		activation: `<circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>`,
		// Crescent moon + stars — dream consolidation.
		dreams: `<path d="M20 13.5A7.5 7.5 0 1 1 11 4.2a6 6 0 0 0 9 9.3Z"/><path d="m6.5 6 .5 1.4L8.4 8 7 8.6 6.5 10 6 8.6 4.6 8 6 7.4z" fill="currentColor" stroke="none" opacity=".7"/>`,
		// Calendar — scheduling.
		schedule: `<rect x="4" y="5.5" width="16" height="14" rx="2"/><path d="M4 9.5h16M8 3.5v4M16 3.5v4"/><circle cx="9" cy="14" r="1" fill="currentColor" stroke="none"/><circle cx="13" cy="14" r="1" fill="currentColor" stroke="none"/><circle cx="9" cy="17" r="1" fill="currentColor" stroke="none"/>`,
		// Filled award star — importance.
		importance: `<path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 17l-5.2 2.6 1-5.8L3.5 9.7l5.9-.9z" fill="currentColor" stroke="none"/>`,
		// Two overlapping cards — duplicates.
		duplicates: `<rect x="4" y="7" width="11" height="11" rx="2"/><path d="M9 7V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-1"/>`,
		// Opposing arrows colliding — contradiction.
		contradictions: `<path d="M3 9h8l-2.5-2.5M3 9l2.5 2.5M21 15h-8l2.5 2.5M21 15l-2.5-2.5"/><path d="M12 4v16" opacity=".35" stroke-dasharray="2 2.5"/>`,
		// Tessellated grid — recurring patterns.
		patterns: `<path d="M4 8.5 8.5 4 13 8.5 8.5 13zM13 8.5 17.5 4 22 8.5 17.5 13zM8.5 17 13 12.5 17.5 17 13 21.5z" opacity=".9"/>`,
		// Crosshair / target — intentions / aim.
		intentions: `<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/><path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/>`,
		// Bar chart — stats.
		stats: `<path d="M4 20h16"/><rect x="5.5" y="11" width="3.2" height="6.5" rx="1" fill="currentColor" stroke="none"/><rect x="10.4" y="7" width="3.2" height="10.5" rx="1" fill="currentColor" stroke="none" opacity=".85"/><rect x="15.3" y="13.5" width="3.2" height="4" rx="1" fill="currentColor" stroke="none" opacity=".7"/>`,
		// Gear — settings.
		settings: `<circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v2.6M12 18.9v2.6M21.5 12h-2.6M5.1 12H2.5M18.7 5.3l-1.9 1.9M7.2 16.8l-1.9 1.9M18.7 18.7l-1.9-1.9M7.2 7.2 5.3 5.3"/>`,

		// ── UI utility glyphs (not nav, but part of the system) ──
		command: `<path d="M9 6.5A2.5 2.5 0 1 0 6.5 9H9V6.5ZM9 9v6M9 9h6M15 9V6.5A2.5 2.5 0 1 1 17.5 9H15ZM15 15h2.5A2.5 2.5 0 1 1 15 17.5V15ZM15 15H9M9 15v2.5A2.5 2.5 0 1 1 6.5 15H9Z"/>`,
		search: `<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.3-4.3"/>`,
		filter: `<path d="M4 5.5h16l-6 7v5l-4 2v-7z"/>`,
		sparkle: `<path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6z" fill="currentColor" stroke="none"/><path d="m18.5 14 .7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" fill="currentColor" stroke="none" opacity=".6"/>`,
		chevron: `<path d="m8 10 4 4 4-4"/>`,
		close: `<path d="m6 6 12 12M18 6 6 18"/>`,
		pulse: `<path d="M3 12h4l2-6 4 12 2-6h6"/>`,
		logo: `<circle cx="6" cy="8" r="2"/><circle cx="17" cy="6.5" r="2"/><circle cx="12" cy="17" r="2.4" fill="currentColor" stroke="none"/><circle cx="18" cy="16" r="1.6"/><path d="M7.7 9.2 10.5 15M15.3 7.4 12.7 15M8 8l7-1M14 16.5l2.4-.3"/>`,
	};
</script>

<script lang="ts">
	interface Props {
		name: IconName;
		size?: number;
		strokeWidth?: number;
		class?: string;
		/** When true, the icon plays a one-shot stroke-draw on mount. */
		draw?: boolean;
	}
	let { name, size = 20, strokeWidth = 1.6, class: cls = '', draw = false }: Props = $props();
</script>

<svg
	xmlns="http://www.w3.org/2000/svg"
	viewBox="0 0 24 24"
	width={size}
	height={size}
	fill="none"
	stroke="currentColor"
	stroke-width={strokeWidth}
	stroke-linecap="round"
	stroke-linejoin="round"
	class="vestige-icon {draw ? 'vestige-icon-draw' : ''} {cls}"
	aria-hidden="true"
	role="presentation"
>
	{@html ICON_PATHS[name]}
</svg>

<style>
	.vestige-icon {
		display: inline-block;
		flex-shrink: 0;
		vertical-align: middle;
		overflow: visible;
	}

	/* One-shot stroke-draw — used on page-title icons so each route entry
	   feels like the icon is being "written" for you. Reduced-motion users
	   get the fully-drawn icon with no animation. */
	@media not (prefers-reduced-motion: reduce) {
		.vestige-icon-draw :global(path),
		.vestige-icon-draw :global(circle),
		.vestige-icon-draw :global(rect) {
			stroke-dasharray: 64;
			stroke-dashoffset: 64;
			animation: icon-draw 0.7s cubic-bezier(0.65, 0, 0.35, 1) forwards;
		}
		@keyframes icon-draw {
			to {
				stroke-dashoffset: 0;
			}
		}
	}
</style>

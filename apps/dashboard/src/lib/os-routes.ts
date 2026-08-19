// ─────────────────────────────────────────────────────────────────────────────
// os-routes.ts — THE canonical VestigeOS route registry.
//
// ONE source of truth for every navigation surface. Before this file there were
// four disagreeing inventories (nav-layer's 9-organ COGNITIVE_OS_ROUTES,
// palace-map's 19 ORGAN_REGIONS, MobileNav, and the e2e ALL_ROUTES) — so
// desktop canvas pages became navigation islands (Palace itself was in NONE of
// the visible nav surfaces). Every consumer now derives from this list:
//   - the persistent shell dock          (primary organs)
//   - the grouped command palette (⌘K)   (ALL organs, by group)
//   - the mobile menu                     (ALL groups + organs)
//   - the in-canvas WebGPU nav rail       (enhancement only)
//   - the Palace constellation            (visible organs)
//   - the e2e route smoke list            (all reachable)
//
// A route is only shippable customer-facing nav if it appears here with
// visible !== 'internal'. `_msdftest` is deliberately excluded (internal only).
// ─────────────────────────────────────────────────────────────────────────────

import type { IconName } from '$lib/components/Icon.svelte';

/** Which OS group an organ belongs to (drives palette + mobile grouping). */
export type OsGroup = 'Primary' | 'Understand' | 'Maintain' | 'Reflect' | 'System';

/** How the route mounts its renderer (informational; used by fallback logic). */
export type StageType = 'observatory' | 'route-stage' | 'canvas' | 'dom';

/** Launch visibility: dock = in the persistent dock; nav = in palette/mobile/palace;
 *  hidden = reachable by URL + palette only; internal = never in customer nav. */
export type Visibility = 'dock' | 'nav' | 'hidden' | 'internal';

export interface OsRoute {
	/** Path WITHOUT the base prefix, e.g. '/graph'. Use osHref() to render. */
	href: string;
	/** Human label shown in every nav surface. */
	label: string;
	/** One-line purpose, shown in the command palette + organ overlay. */
	purpose: string;
	group: OsGroup;
	/** Single-key shortcut (⌘K palette + in-canvas rail). Unique across the set. */
	shortcut?: string;
	icon: IconName;
	stage: StageType;
	visibility: Visibility;
	/** true once the organ is launch-ready; false surfaces a "beta" affordance. */
	ready: boolean;
}

// ── The registry — all 20 customer organs (order = canonical display order) ──
export const OS_ROUTES: OsRoute[] = [
	// PRIMARY — the spine of the product, in the persistent dock.
	{ href: '/palace', label: 'Palace', purpose: 'The spatial launcher — every organ as a constellation you fly into.', group: 'Primary', shortcut: 'P', icon: 'logo', stage: 'canvas', visibility: 'dock', ready: true },
	{ href: '/observatory', label: 'Observatory', purpose: 'The living memory field + the deterministic cognitive moments (Salience Rescue).', group: 'Primary', shortcut: 'O', icon: 'sparkle', stage: 'observatory', visibility: 'dock', ready: true },
	{ href: '/graph', label: 'Witness', purpose: 'Receipt-bounded evidence loom — prove which memories shaped an agent run.', group: 'Primary', shortcut: 'G', icon: 'graph', stage: 'canvas', visibility: 'dock', ready: true },
	{ href: '/memories', label: 'Memories', purpose: 'Browse, search, and inspect individual memories with their FSRS state.', group: 'Primary', shortcut: 'M', icon: 'memories', stage: 'observatory', visibility: 'dock', ready: true },
	{ href: '/timeline', label: 'Timeline', purpose: 'Bitemporal history — when memories were valid vs recorded, with audit windows.', group: 'Primary', shortcut: 'T', icon: 'timeline', stage: 'route-stage', visibility: 'dock', ready: true },
	{ href: '/blackbox', label: 'Black Box', purpose: 'The receipt — what Vestige concluded, the evidence, and why, exportable.', group: 'Primary', shortcut: 'B', icon: 'blackbox', stage: 'route-stage', visibility: 'dock', ready: true },

	// UNDERSTAND — the reasoning + exploration organs.
	{ href: '/reasoning', label: 'Reasoning', purpose: 'Watch a live deep_reference decision trace form from evidence.', group: 'Understand', shortcut: 'R', icon: 'reasoning', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/explore', label: 'Explore', purpose: 'A shareable semantic walk through memory neighborhoods.', group: 'Understand', shortcut: 'E', icon: 'explore', stage: 'observatory', visibility: 'nav', ready: true },
	{ href: '/contradictions', label: 'Contradictions', purpose: 'Trust-weighted conflict pairs — where your memory disagrees with itself.', group: 'Understand', shortcut: 'C', icon: 'contradictions', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/patterns', label: 'Patterns', purpose: 'Cross-project patterns mined from the corpus.', group: 'Understand', icon: 'patterns', stage: 'route-stage', visibility: 'nav', ready: true },

	// MAINTAIN — the memory-hygiene organs.
	{ href: '/duplicates', label: 'Duplicates', purpose: 'Cosine-similarity clusters quarantined for review before merge.', group: 'Maintain', shortcut: 'D', icon: 'duplicates', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/memory-prs', label: 'Memory PRs', purpose: 'Proposed memory changes held for review before they touch the graph.', group: 'Maintain', icon: 'memorypr', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/importance', label: 'Importance', purpose: 'Which memories rank highest by the 4-channel importance model, and why.', group: 'Maintain', icon: 'importance', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/activation', label: 'Activation', purpose: 'The activation field — which memories light up for a query.', group: 'Maintain', shortcut: 'A', icon: 'activation', stage: 'observatory', visibility: 'nav', ready: true },

	// REFLECT — the temporal / ambient organs.
	{ href: '/feed', label: 'Activity', purpose: 'The live event stream — recalls, dreams, suppressions, as they happen.', group: 'Reflect', shortcut: 'F', icon: 'feed', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/dreams', label: 'Dreams', purpose: 'Replay consolidation cycles and the connections they discover.', group: 'Reflect', icon: 'dreams', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/schedule', label: 'Schedule', purpose: 'Review urgency — what is due now, overdue, and coming next.', group: 'Reflect', shortcut: 'H', icon: 'schedule', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/intentions', label: 'Intentions', purpose: 'Active and predicted intentions grouped by state.', group: 'Reflect', shortcut: 'I', icon: 'intentions', stage: 'route-stage', visibility: 'nav', ready: true },
	{ href: '/stats', label: 'Stats', purpose: 'System vitals — retention distribution, coverage, throughput.', group: 'Reflect', shortcut: 'S', icon: 'stats', stage: 'route-stage', visibility: 'nav', ready: true },

	// SYSTEM
	{ href: '/embeddings', label: 'Embeddings', purpose: 'Own local embedding profiles: install, evaluate, migrate, activate, and roll back with receipts.', group: 'System', icon: 'embeddings', stage: 'dom', visibility: 'nav', ready: true },
	{ href: '/settings', label: 'Settings', purpose: 'Tune the cognitive engine and run the maintenance rituals.', group: 'System', shortcut: ',', icon: 'settings', stage: 'route-stage', visibility: 'nav', ready: true }
];

// ── Derived views (every consumer uses these, never re-lists routes) ──────────

/** The persistent desktop dock: primary organs + a Command entry (added by shell). */
export const DOCK_ROUTES: OsRoute[] = OS_ROUTES.filter((r) => r.visibility === 'dock');

/** Everything a customer can navigate to (dock + nav + hidden), for the palette. */
export const NAV_ROUTES: OsRoute[] = OS_ROUTES.filter((r) => r.visibility !== 'internal');

/** Group order for the command palette + mobile menu. */
export const OS_GROUP_ORDER: OsGroup[] = ['Primary', 'Understand', 'Maintain', 'Reflect', 'System'];

/** Routes bucketed by group, in canonical order — drives the palette + mobile menu. */
export function routesByGroup(): { group: OsGroup; routes: OsRoute[] }[] {
	return OS_GROUP_ORDER.map((group) => ({
		group,
		routes: NAV_ROUTES.filter((r) => r.group === group)
	})).filter((g) => g.routes.length > 0);
}

/** The home route — Palace is the actual VestigeOS home. */
export const HOME_ROUTE = '/palace';

/** Look up a route by its href (with or without a leading base). */
export function findRoute(pathNoBase: string): OsRoute | undefined {
	const p = pathNoBase || '/';
	return OS_ROUTES.find((r) => p === r.href || p.startsWith(r.href + '/')) ??
		(p === '/' ? OS_ROUTES.find((r) => r.href === HOME_ROUTE) : undefined);
}

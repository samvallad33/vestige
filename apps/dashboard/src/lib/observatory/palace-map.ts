// Spatial Palace — the organ registry.
//
// The palace is the set of dashboard organs rendered as a navigable 3D
// constellation. Nine curated hero routes are particle organs; clicking one
// collapses the field into that destination. Rendering is owned by
// PalaceSwarmPass; this module declares WHAT the organs are and resolves a
// picked href back to its region. Formation anchors are deterministic, so the
// constellation remains capture-stable.

/** One organ region in the palace. `href` is the dashboard route it dives into. */
export interface OrganRegion {
	href: string;
	label: string;
	/** Cognitive-palette family — drives the region's base color via retention proxy. */
	family: 'reasoning' | 'immune' | 'memory' | 'temporal' | 'signal' | 'system';
	/** Whether this is the gravitational center of the palace (largest, at origin). */
	center?: boolean;
}

/**
 * The organ constellation surfaced by the palace.
 *
 * LAUNCH CURATION (Sam, Jul 10-13 2026): trimmed from 19 to a TIGHT DEMO of nine
 * hero organs. 20 organs cognitively overloaded first-time users — the
 * memory-hygiene / maintainer-tool pages (duplicates, contradictions, patterns,
 * memory-prs, ...) read as confusing plumbing, not "my AI's memory is alive".
 * The hidden organs are NOT deleted: their routes stay in the codebase, remain
 * reachable by direct URL, and stay tested — they just aren't palace-linked. To
 * restore any, move it from HIDDEN_ORGANS back into this array. Fully reversible.
 *
 * The Observatory is the center (the "cortex" the other organs orbit). Graph +
 * Memory Cinema live inside Observatory/Graph and are not separate regions here.
 */
export const ORGAN_REGIONS: OrganRegion[] = [
	{ href: '/observatory', label: 'OBSERVATORY', family: 'system', center: true },
	{ href: '/graph', label: 'GRAPH', family: 'memory' },
	{ href: '/memories', label: 'MEMORIES', family: 'memory' },
	{ href: '/timeline', label: 'TIMELINE', family: 'temporal' },
	{ href: '/feed', label: 'FEED', family: 'signal' },
	{ href: '/explore', label: 'EXPLORE', family: 'reasoning' },
	{ href: '/reasoning', label: 'REASONING', family: 'reasoning' },
	{ href: '/blackbox', label: 'BLACK BOX', family: 'signal' },
	{ href: '/contradictions', label: 'CONTRADICTIONS', family: 'immune' }
];

/**
 * Organs that exist and work (routes live, URL-reachable, tested) but are HIDDEN
 * from the launch palace to keep the first impression clean. These are the
 * power-user / memory-hygiene / neuro-frontier tools. Restore one by moving its
 * entry into ORGAN_REGIONS above. Kept here (not deleted) so the full set stays
 * documented and one edit away.
 */
export const HIDDEN_ORGANS: OrganRegion[] = [
	{ href: '/duplicates', label: 'DUPLICATES', family: 'memory' },
	{ href: '/memory-prs', label: 'MEMORY PRS', family: 'immune' },
	{ href: '/activation', label: 'ACTIVATION', family: 'signal' },
	{ href: '/dreams', label: 'DREAMS', family: 'temporal' },
	{ href: '/schedule', label: 'SCHEDULE', family: 'temporal' },
	{ href: '/importance', label: 'IMPORTANCE', family: 'reasoning' },
	{ href: '/patterns', label: 'PATTERNS', family: 'reasoning' },
	{ href: '/intentions', label: 'INTENTIONS', family: 'temporal' },
	{ href: '/stats', label: 'STATS', family: 'system' },
	{ href: '/settings', label: 'SETTINGS', family: 'system' }
];

/** Resolve a picked palace node id (an href) back to its organ region. */
export function regionByHref(href: string): OrganRegion | undefined {
	return ORGAN_REGIONS.find((o) => o.href === href);
}

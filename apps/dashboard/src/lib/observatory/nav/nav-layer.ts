import type { ObservatoryEngine, FramePass } from '$lib/observatory/engine';
import { CAUSAL, MEDIUM, RETENTION, rgb01 } from '$lib/observatory/cognitive-palette';
import { TextLayerPass, type TextLayerItem } from '$lib/observatory/text/text-layer';

export type NavRoute = {
	label: string;
	href: string;
	shortcut?: string;
};

export type NavPick = {
	id: string;
	kind: 'route-nav';
	payload: NavRoute;
};

export interface NavLayerOptions {
	activePath?: string;
	routes?: NavRoute[];
}

export interface NavLayerPass extends FramePass {
	init(): Promise<void>;
	setActivePath(path: string): void;
	setHoverFromNdc(ndcX: number, ndcY: number): NavPick | null;
	clearHover(): void;
	pickAt(ndcX: number, ndcY: number): NavPick | null;
	dispose(): void;
}

// LAUNCH CURATION (Sam, Jul 10 2026): the in-canvas nav rail mirrors the palace's
// curated hero set (ORGAN_REGIONS in palace-map.ts) so there is ONE surfaced set
// across the palace and every organ's nav. The 10 hidden power-user/hygiene
// organs (contradictions, blackbox, duplicates, memory-prs, activation, dreams,
// schedule, importance, patterns, intentions) still work by direct URL — they are
// just not listed here or in the palace. Restore one by adding it back to BOTH
// this list and ORGAN_REGIONS. Shortcuts kept from the original mapping.
export const COGNITIVE_OS_ROUTES: NavRoute[] = [
	{ href: '/observatory', label: 'Observatory', shortcut: 'O' },
	{ href: '/graph', label: 'Graph', shortcut: 'G' },
	{ href: '/memories', label: 'Memories', shortcut: 'M' },
	{ href: '/timeline', label: 'Timeline', shortcut: 'T' },
	{ href: '/feed', label: 'Feed', shortcut: 'F' },
	{ href: '/explore', label: 'Explore', shortcut: 'E' },
	{ href: '/reasoning', label: 'Reasoning', shortcut: 'R' },
	{ href: '/stats', label: 'Stats', shortcut: 'S' },
	{ href: '/settings', label: 'Settings', shortcut: ',' }
];

const NAV_X = -0.93;
const NAV_Y = 0.83;
const NAV_SIZE = 0.024;
const NAV_STEP = 0.058;
// Collapsed dock (Jul 11 2026, Claude + GPT-5.6-sol): at rest the rail is a thin
// column of just shortcut letters at NAV_MARKER_X, clear of all organ content
// (which starts at x>=-0.94). It expands to the full labelled rail only while the
// cursor is in the left activation/retention band — so the nav never overprints a
// page's content, but stays one edge-hover away. Zone-triggered (NOT glyph-hover,
// since collapsed markers are too thin to reliably pick). Hysteresis: expand at
// a tight edge, retain across the whole expanded footprint, collapse outside it.
const NAV_MARKER_X = -0.965; // single left-anchored letters grow rightward → safe
const NAV_EXPAND_X = -0.9; // collapsed → expand when cursor x <= this…
const NAV_EXPAND_Y_LO = -0.4; // …and inside this vertical band
const NAV_EXPAND_Y_HI = 0.92;
const NAV_RETAIN_X = -0.73; // stay expanded while cursor x <= this (wider)…
const NAV_RETAIN_Y_LO = -0.44; // …within this slightly taller band
const NAV_RETAIN_Y_HI = 0.96;
const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
const HOVER = [...rgb01(RETENTION.luciferin), 1] satisfies [number, number, number, number];
const DIM = [...rgb01(RETENTION.bridge), 0.36] satisfies [number, number, number, number];
// Resting collapsed markers: dimmer than DIM so the strip reads as latent chrome,
// not content — but perceptible enough that a first-time viewer notices the edge
// affordance. 0.36 (Claude + GPT-5.6-sol @ max effort): 0.28 read as
// undiscoverable, 0.40 crept back toward a persistent rail. Active route = CYAN.
const MARKER_DIM = [...rgb01(RETENTION.bridge), 0.36] satisfies [number, number, number, number];
const BLACKWATER = [...rgb01(MEDIUM.blackwater), 0.15] satisfies [number, number, number, number];

export function createNavLayerPass(engine: ObservatoryEngine, opts: NavLayerOptions = {}): NavLayerPass {
	return new TextNavLayer(engine, opts);
}

class TextNavLayer implements NavLayerPass {
	private readonly text: TextLayerPass;
	private readonly routes: NavRoute[];
	private activePath: string;
	private hoverHref: string | null = null;
	private expanded = false;
	private ready = false;
	private readonly engine: ObservatoryEngine;

	constructor(engine: ObservatoryEngine, opts: NavLayerOptions) {
		this.engine = engine;
		this.text = new TextLayerPass(engine);
		this.routes = opts.routes ?? COGNITIVE_OS_ROUTES;
		this.activePath = opts.activePath ?? '';
	}

	/**
	 * Touch/portrait phones have NO hover, so the desktop hover-to-expand rail can
	 * never open and the organ becomes un-navigable. On a narrow viewport we switch
	 * to a directly-TAPPABLE dock: always-visible shortcut letters at the true left
	 * edge, each its own pick target so one tap navigates. Derived from the live
	 * viewport aspect — nothing hardcoded, and desktop is untouched.
	 */
	private isMobile(): boolean {
		let vw = this.engine.params[6] || 0;
		let vh = this.engine.params[7] || 0;
		if ((vw <= 0 || vh <= 0) && typeof window !== 'undefined') {
			vw = window.innerWidth;
			vh = window.innerHeight;
		}
		if (vw <= 0 || vh <= 0) return false;
		return vw / vh < 0.85;
	}

	async init(): Promise<void> {
		await this.text.init();
		this.ready = true;
		this.rebuild();
	}

	setActivePath(path: string): void {
		if (this.activePath === path) return;
		this.activePath = path;
		this.rebuild();
	}

	setHoverFromNdc(ndcX: number, ndcY: number): NavPick | null {
		// Mobile dock is always-visible + directly tappable — there is no hover-to-
		// expand. Just report whether a marker is under the point (for cursor state)
		// without any rebuild.
		if (this.isMobile()) return this.pickAt(ndcX, ndcY);
		// 1. Decide expand/collapse from the cursor ZONE (with hysteresis) BEFORE
		//    picking, so that on the first move into the edge the labels are built
		//    and immediately hoverable in this same call (no dead pointermove).
		const inExpandZone =
			ndcX <= NAV_EXPAND_X && ndcY >= NAV_EXPAND_Y_LO && ndcY <= NAV_EXPAND_Y_HI;
		const inRetainZone =
			ndcX <= NAV_RETAIN_X && ndcY >= NAV_RETAIN_Y_LO && ndcY <= NAV_RETAIN_Y_HI;
		const nextExpanded = this.expanded ? inRetainZone : inExpandZone;
		if (nextExpanded !== this.expanded) {
			this.expanded = nextExpanded;
			if (!nextExpanded) this.hoverHref = null; // clear stale hover on collapse
			this.rebuild(); // 2. structural rebuild so pickAt sees current items
		}
		// 3. Pick against the (now current) item set. Collapsed → markers aren't
		//    'route-nav', so pickAt returns null and nothing hovers, by design.
		const hit = this.pickAt(ndcX, ndcY);
		const next = hit?.payload.href ?? null;
		if (next !== this.hoverHref) {
			this.hoverHref = next; // 4. update hover, rebuild once more if changed
			this.rebuild();
		}
		return hit;
	}

	clearHover(): void {
		// Pointer left the canvas → collapse the dock deterministically (capture-
		// stable) AND clear hover. Guard covers the expanded-but-unhovered case
		// (GPT-5.6-sol caught this: a plain hoverHref===null early-return would
		// leave an expanded rail stranded when the cursor exits without hovering).
		if (!this.expanded && this.hoverHref === null) return;
		this.expanded = false;
		this.hoverHref = null;
		this.rebuild();
	}

	pickAt(ndcX: number, ndcY: number): NavPick | null {
		const hit = this.text.pickAt(ndcX, ndcY);
		if (!hit || hit.kind !== 'route-nav') return null;
		const route = hit.payload as TextLayerItem & { route?: NavRoute };
		if (!route.route) return null;
		return { id: hit.id, kind: 'route-nav', payload: route.route };
	}

	render(pass: GPURenderPassEncoder): void {
		this.text.render(pass);
	}

	dispose(): void {
		this.text.dispose();
	}

	private rebuild(): void {
		if (!this.ready) return;
		// On mobile the in-canvas rail is SUPPRESSED entirely: a global DOM MobileNav
		// (rendered by the (app) shell) is the single, reliable tap-to-navigate
		// surface for phones — it works on every stage type AND on devices with no
		// WebGPU, which this in-canvas rail cannot. Rendering nothing here avoids a
		// confusing double nav. Desktop keeps the full hover-to-expand rail.
		if (this.isMobile()) {
			this.text.setText([]);
			return;
		}
		this.text.setText(this.expanded ? this.buildExpanded() : this.buildCollapsed());
	}

	/**
	 * Collapsed dock — a thin left-edge column of just the shortcut letters at
	 * NAV_MARKER_X, sharing the EXACT Y anchors of the expanded labels so the rail
	 * appears to unfold from stable points rather than swap in. Resting markers are
	 * very dim (latent chrome); the active route stays CYAN so "where am I" is
	 * always answered. No header, no hover ring. Markers are NOT 'route-nav', so
	 * pickAt() returns null while collapsed (no clicking a thin target — you hover
	 * to expand, then click the full label).
	 */
	private buildCollapsed(): TextLayerItem[] {
		return this.routes.map((route, i) => {
			const active = this.isActive(route.href);
			const glyph = route.shortcut ?? route.label.charAt(0).toUpperCase();
			return {
				id: `route-nav-marker:${route.href}`,
				kind: 'route-nav-marker',
				text: glyph,
				x: NAV_MARKER_X,
				y: NAV_Y - i * NAV_STEP,
				size: NAV_SIZE, // same size as expanded → no scale-pop on unfold
				color: active ? CYAN : MARKER_DIM,
				startFrame: 0,
				revealSpan: 1,
				maxWidthEm: 4
			} satisfies TextLayerItem;
		});
	}

	/** Expanded rail — the full labelled nav (header + 9 routes + hover ring). */
	private buildExpanded(): TextLayerItem[] {
		return [
			{
				id: 'route-nav:rail',
				kind: 'route-nav-rail',
				text: 'COGNITIVE OS',
				x: NAV_X,
				y: NAV_Y + 0.075,
				size: 0.021,
				color: BLACKWATER,
				revealSpan: 1
			},
			...this.routes.map((route, i) => {
				const active = this.isActive(route.href);
				const hovered = this.hoverHref === route.href;
				const color = active ? CYAN : hovered ? HOVER : DIM;
				const marker = active ? '>' : hovered ? '+' : '-';
				const tail = route.shortcut ? ` ${route.shortcut}` : '';
				return {
					id: `route-nav:${route.href}`,
					kind: 'route-nav',
					text: `${marker} ${route.label.toUpperCase()}${tail}`,
					x: active || hovered ? NAV_X + 0.012 : NAV_X,
					y: NAV_Y - i * NAV_STEP,
					size: active || hovered ? NAV_SIZE * 1.06 : NAV_SIZE,
					color,
					startFrame: 0,
					revealSpan: 1,
					maxWidthEm: 16,
					// Generous hit padding so the expanded labels are easy to click
					// (bare glyph boxes are ~14px thin — the systemic hitPad fix).
					hitPadX: 0.03,
					hitPadY: 0.02,
					route
				} satisfies TextLayerItem & { route: NavRoute };
			}),
			...(this.hoverHref
				? [
						{
							id: 'route-nav:hover-ring',
							kind: 'route-nav-focus',
							text: '>>>>>>>>>>>>>>>>',
							x: NAV_X - 0.01,
							y: NAV_Y - this.routes.findIndex((r) => r.href === this.hoverHref) * NAV_STEP - 0.029,
							size: 0.012,
							color: CYAN,
							revealSpan: 1
						} satisfies TextLayerItem
					]
				: [])
		];
	}

	private isActive(href: string): boolean {
		const path = this.activePath || '';
		return path === href || path.endsWith(href) || (href === '/observatory' && (path === '/' || path === ''));
	}
}

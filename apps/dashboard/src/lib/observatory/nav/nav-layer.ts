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

export const COGNITIVE_OS_ROUTES: NavRoute[] = [
	{ href: '/observatory', label: 'Observatory', shortcut: 'O' },
	{ href: '/reasoning', label: 'Reasoning', shortcut: 'R' },
	{ href: '/contradictions', label: 'Contradictions', shortcut: 'X' },
	{ href: '/blackbox', label: 'Black Box', shortcut: 'B' },
	{ href: '/timeline', label: 'Timeline', shortcut: 'T' },
	{ href: '/duplicates', label: 'Duplicates', shortcut: 'U' },
	{ href: '/graph', label: 'Graph', shortcut: 'G' },
	{ href: '/memory-prs', label: 'Memory PRs', shortcut: 'Q' },
	{ href: '/memories', label: 'Memories', shortcut: 'M' },
	{ href: '/feed', label: 'Feed', shortcut: 'F' },
	{ href: '/explore', label: 'Explore', shortcut: 'E' },
	{ href: '/activation', label: 'Activation', shortcut: 'A' },
	{ href: '/dreams', label: 'Dreams', shortcut: 'D' },
	{ href: '/schedule', label: 'Schedule', shortcut: 'C' },
	{ href: '/importance', label: 'Importance', shortcut: 'P' },
	{ href: '/patterns', label: 'Patterns', shortcut: 'N' },
	{ href: '/intentions', label: 'Intentions', shortcut: 'I' },
	{ href: '/stats', label: 'Stats', shortcut: 'S' },
	{ href: '/settings', label: 'Settings', shortcut: ',' }
];

const NAV_X = -0.93;
const NAV_Y = 0.83;
const NAV_SIZE = 0.024;
const NAV_STEP = 0.058;
const CYAN = [...rgb01(CAUSAL.forward), 1] satisfies [number, number, number, number];
const HOVER = [...rgb01(RETENTION.luciferin), 1] satisfies [number, number, number, number];
const DIM = [...rgb01(RETENTION.bridge), 0.36] satisfies [number, number, number, number];
const BLACKWATER = [...rgb01(MEDIUM.blackwater), 0.15] satisfies [number, number, number, number];

export function createNavLayerPass(engine: ObservatoryEngine, opts: NavLayerOptions = {}): NavLayerPass {
	return new TextNavLayer(engine, opts);
}

class TextNavLayer implements NavLayerPass {
	private readonly text: TextLayerPass;
	private readonly routes: NavRoute[];
	private activePath: string;
	private hoverHref: string | null = null;
	private ready = false;

	constructor(engine: ObservatoryEngine, opts: NavLayerOptions) {
		this.text = new TextLayerPass(engine);
		this.routes = opts.routes ?? COGNITIVE_OS_ROUTES;
		this.activePath = opts.activePath ?? '';
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
		const hit = this.pickAt(ndcX, ndcY);
		const next = hit?.payload.href ?? null;
		if (next !== this.hoverHref) {
			this.hoverHref = next;
			this.rebuild();
		}
		return hit;
	}

	clearHover(): void {
		if (this.hoverHref === null) return;
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
		const items: TextLayerItem[] = [
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
		this.text.setText(items);
	}

	private isActive(href: string): boolean {
		const path = this.activePath || '';
		return path === href || path.endsWith(href) || (href === '/observatory' && (path === '/' || path === ''));
	}
}

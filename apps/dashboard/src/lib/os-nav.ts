// ─────────────────────────────────────────────────────────────────────────────
// os-nav.ts — base-aware routing + the URL cognitive-context contract.
//
// Fixes the audit's launch-blocker: components navigated with unbased paths like
// goto(`/graph?center=...`) which ESCAPE the configured /dashboard base and 404
// on the deployed site. Every cross-organ link must go through osHref()/osGoto()
// so it stays inside the base.
//
// The "cognitive context" is a small, shared set of URL params that carry
// selection/state ACROSS organs so a journey (Rescue → Graph → Reason → Black
// Box → Cinema) keeps its selections intact. `memory` is the canonical
// cross-organ contract; `center` is kept as a temporary Graph alias.
// ─────────────────────────────────────────────────────────────────────────────

import { base } from '$app/paths';
import { goto } from '$app/navigation';

/** The URL-backed cognitive context shared across every organ. */
export interface CognitiveContext {
	/** The focused memory id — the cross-organ selection contract. */
	memory?: string;
	/** A Black Box run id. */
	run?: string;
	/** A receipt id. */
	receipt?: string;
	/** A free-text query (Explore/Activation/Reasoning). */
	q?: string;
	/** Comma-separated ids to emphasize (Graph focus set). */
	focus?: string[];
	/** Launch Memory Cinema on arrival. */
	cinema?: boolean;
	/** Time window in days (Timeline/Schedule). */
	days?: number;
	/** A generic per-organ filter value. */
	filter?: string;
	/** Temporary Graph alias for `memory` (legacy ?center=). */
	center?: string;
}

/** Serialize a cognitive context to a query string (stable key order, base-free). */
export function contextToQuery(ctx: CognitiveContext): string {
	const p = new URLSearchParams();
	if (ctx.memory) p.set('memory', ctx.memory);
	if (ctx.run) p.set('run', ctx.run);
	if (ctx.receipt) p.set('receipt', ctx.receipt);
	if (ctx.q) p.set('q', ctx.q);
	if (ctx.focus && ctx.focus.length) p.set('focus', ctx.focus.join(','));
	if (ctx.cinema) p.set('cinema', '1');
	if (typeof ctx.days === 'number') p.set('days', String(ctx.days));
	if (ctx.filter) p.set('filter', ctx.filter);
	if (ctx.center) p.set('center', ctx.center);
	const s = p.toString();
	return s ? `?${s}` : '';
}

/** Read the cognitive context out of a URL (SvelteKit page.url or a URLSearchParams). */
export function contextFromUrl(url: URL): CognitiveContext {
	const p = url.searchParams;
	const focus = p.get('focus');
	const days = p.get('days');
	return {
		// `memory` is canonical; fall back to the legacy `center` alias.
		memory: p.get('memory') ?? p.get('center') ?? undefined,
		run: p.get('run') ?? undefined,
		receipt: p.get('receipt') ?? undefined,
		q: p.get('q') ?? undefined,
		focus: focus ? focus.split(',').filter(Boolean) : undefined,
		cinema: p.get('cinema') === '1',
		days: days != null && days !== '' && Number.isFinite(Number(days)) ? Number(days) : undefined,
		filter: p.get('filter') ?? undefined,
		center: p.get('center') ?? undefined
	};
}

/**
 * Build a base-safe href for an organ route (path is WITHOUT base, e.g. '/graph').
 * ALWAYS use this for cross-organ links — never a bare `/graph` string.
 */
export function osHref(pathNoBase: string, ctx: CognitiveContext = {}): string {
	const clean = pathNoBase.startsWith('/') ? pathNoBase : `/${pathNoBase}`;
	return `${base}${clean}${contextToQuery(ctx)}`;
}

/** Navigate to an organ route carrying cognitive context, base-safe. */
export function osGoto(pathNoBase: string, ctx: CognitiveContext = {}): Promise<void> {
	return goto(osHref(pathNoBase, ctx));
}

/** The current dashboard path with the base stripped (for active-route matching). */
export function stripBase(pathname: string): string {
	return pathname.startsWith(base) ? pathname.slice(base.length) || '/' : pathname;
}

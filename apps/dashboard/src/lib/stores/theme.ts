// Theme store — closes GitHub issue #11 (Dark/Light theme toggle).
//
// Design:
//  - Three modes: 'dark' (current default, bioluminescent), 'light'
//    (slate-on-white, muted glow), 'auto' (follows system preference).
//  - A single writable `theme` store holds the user preference.
//    A derived `resolvedTheme` collapses 'auto' into the concrete 'dark' |
//    'light' that matchMedia is reporting right now.
//  - On any change we flip `document.documentElement.dataset.theme`. All
//    CSS variable overrides key off `[data-theme='light']` in the
//    injected stylesheet below (app.css is deliberately left untouched so
//    the dark defaults still cascade when no attribute is set).
//  - Preference persists to `localStorage['vestige.theme']`.
//  - `initTheme()` is called once from +layout.svelte onMount. It (a)
//    reads localStorage, (b) injects the light-mode stylesheet into
//    <head>, (c) sets dataset.theme, (d) attaches a matchMedia listener
//    so 'auto' tracks the OS in real time.
//
// Light-mode override strategy:
//  We inject a single <style id="vestige-theme-light"> block at init time
//  rather than editing app.css. This keeps the dark-first design pristine
//  and lets us ship the toggle as a purely additive change. Overrides
//  target the real token names used in app.css (`--color-void`,
//  `--color-text`, `--color-bright`, `--color-dim`, `--color-muted`,
//  `--color-surface`, etc.) plus halve the glow shadows so neon accents
//  don't wash out on a slate-50 canvas.

import { writable, derived, get, type Readable } from 'svelte/store';
import { browser } from '$app/environment';

export type Theme = 'dark' | 'light' | 'auto';
export type ResolvedTheme = 'dark' | 'light';

const STORAGE_KEY = 'vestige.theme';
const STYLE_ELEMENT_ID = 'vestige-theme-light';

/** User preference — 'dark' | 'light' | 'auto'. Persists to localStorage. */
export const theme = writable<Theme>('dark');

/**
 * System preference at this moment — tracked via matchMedia and kept in
 * sync by the listener wired up in `initTheme`. Defaults to 'dark' so
 * SSR/first paint matches the dark-first design.
 */
const systemPrefersDark = writable<boolean>(true);

/**
 * The concrete theme after resolving 'auto' → matchMedia. This is what
 * actually gets written to `document.documentElement.dataset.theme`.
 */
export const resolvedTheme: Readable<ResolvedTheme> = derived(
	[theme, systemPrefersDark],
	([$theme, $prefersDark]) => {
		if ($theme === 'auto') return $prefersDark ? 'dark' : 'light';
		return $theme;
	}
);

/** Runtime guard — TypeScript callers are already narrowed, but the store is
 *  also exposed via the dashboard window for devtools / demo sequences.
 *  We silently ignore unknown values rather than throwing so a fat-finger
 *  console poke can't wedge the UI. */
function isValidTheme(v: unknown): v is Theme {
	return v === 'dark' || v === 'light' || v === 'auto';
}

/** Public setter. Also persists to localStorage. Invalid inputs are ignored. */
export function setTheme(next: Theme): void {
	if (!isValidTheme(next)) return;
	theme.set(next);
	if (browser) {
		try {
			localStorage.setItem(STORAGE_KEY, next);
		} catch {
			// Private mode / disabled storage — silent no-op.
		}
	}
}

/** Cycle dark → light → auto → dark. Used by the ThemeToggle button. */
export function cycleTheme(): void {
	const current = get(theme);
	const next: Theme = current === 'dark' ? 'light' : current === 'light' ? 'auto' : 'dark';
	setTheme(next);
}

/**
 * Injects the light-mode variable overrides into <head>. Idempotent —
 * safe to call multiple times. We target the real tokens from app.css
 * and halve the glow intensity so bioluminescent accents remain readable
 * but don't bloom on a pale canvas.
 */
function ensureLightStylesheet(): void {
	if (!browser) return;
	if (document.getElementById(STYLE_ELEMENT_ID)) return;

	const style = document.createElement('style');
	style.id = STYLE_ELEMENT_ID;
	style.textContent = `
/* Vestige light-mode overrides — injected by theme.ts.
 * Activated by [data-theme='light'] on <html>.
 * Tokens mirror the real names used in app.css so the cascade stays clean. */
[data-theme='light'] {
	/* Core surface palette (slate scale) */
	--color-void: #f8fafc;        /* slate-50 — page background */
	--color-abyss: #f1f5f9;       /* slate-100 */
	--color-deep: #e2e8f0;        /* slate-200 */
	--color-surface: #f1f5f9;     /* slate-100 */
	--color-elevated: #e2e8f0;    /* slate-200 */
	--color-subtle: #cbd5e1;      /* slate-300 */
	--color-muted: #94a3b8;       /* slate-400 */
	--color-dim: #475569;         /* slate-600 */
	--color-text: #0f172a;        /* slate-900 */
	--color-bright: #020617;      /* slate-950 */
}

/* Baseline body/html wiring — app.css sets these against the dark
 * tokens; we just let the variables do the work. Reassert for clarity. */
[data-theme='light'] html,
html[data-theme='light'] {
	background: var(--color-void);
	color: var(--color-text);
}

/* Glass surfaces — recompose on a light canvas. The original alphas
 * are tuned for dark; invert-and-tint for light so panels still read
 * as elevated instead of vanishing. */
[data-theme='light'] .glass {
	background: rgba(255, 255, 255, 0.65);
	border: 1px solid rgba(99, 102, 241, 0.12);
	box-shadow:
		inset 0 1px 0 0 rgba(255, 255, 255, 0.6),
		0 4px 24px rgba(15, 23, 42, 0.08);
}
[data-theme='light'] .glass-subtle {
	background: rgba(255, 255, 255, 0.55);
	border: 1px solid rgba(99, 102, 241, 0.1);
	box-shadow:
		inset 0 1px 0 0 rgba(255, 255, 255, 0.5),
		0 2px 12px rgba(15, 23, 42, 0.06);
}
[data-theme='light'] .glass-sidebar {
	background: rgba(248, 250, 252, 0.82);
	border-right: 1px solid rgba(99, 102, 241, 0.14);
	box-shadow:
		inset -1px 0 0 0 rgba(255, 255, 255, 0.4),
		4px 0 24px rgba(15, 23, 42, 0.08);
}
[data-theme='light'] .glass-panel {
	background: rgba(255, 255, 255, 0.75);
	border: 1px solid rgba(99, 102, 241, 0.14);
	box-shadow:
		inset 0 1px 0 0 rgba(255, 255, 255, 0.5),
		0 8px 32px rgba(15, 23, 42, 0.1);
}

/* Halve glow intensity — neon accents stay recognizable without
 * washing out on slate-50. */
[data-theme='light'] .glow-synapse {
	box-shadow: 0 0 10px rgba(99, 102, 241, 0.15), 0 0 30px rgba(99, 102, 241, 0.05);
}
[data-theme='light'] .glow-dream {
	box-shadow: 0 0 10px rgba(168, 85, 247, 0.15), 0 0 30px rgba(168, 85, 247, 0.05);
}
[data-theme='light'] .glow-memory {
	box-shadow: 0 0 10px rgba(59, 130, 246, 0.15), 0 0 30px rgba(59, 130, 246, 0.05);
}

/* Ambient orbs are gorgeous on black and blinding on white. Tame them. */
[data-theme='light'] .ambient-orb {
	opacity: 0.18;
	filter: blur(100px);
}

/* Scrollbar recolor for the lighter surface. */
[data-theme='light'] ::-webkit-scrollbar-thumb {
	background: #cbd5e1;
}
[data-theme='light'] ::-webkit-scrollbar-thumb:hover {
	background: #94a3b8;
}
`;
	document.head.appendChild(style);
}

/** Apply the resolved theme to <html> so CSS selectors activate. */
function applyDocumentAttribute(resolved: ResolvedTheme): void {
	if (!browser) return;
	document.documentElement.dataset.theme = resolved;
}

let mediaQuery: MediaQueryList | null = null;
let mediaListener: ((e: MediaQueryListEvent) => void) | null = null;
let themeUnsub: (() => void) | null = null;
let resolvedUnsub: (() => void) | null = null;

/**
 * Boot the theme system. Call once from +layout.svelte onMount.
 * Idempotent — safe to call repeatedly; subsequent calls are no-ops.
 * Returns a teardown fn for tests / HMR.
 */
export function initTheme(): () => void {
	if (!browser) return () => {};

	// Tear down any prior init so repeated calls don't leak listeners or
	// subscriptions. This is the hot-reload / double-mount safety net.
	if (mediaQuery && mediaListener) {
		mediaQuery.removeEventListener('change', mediaListener);
	}
	resolvedUnsub?.();
	themeUnsub?.();
	mediaQuery = null;
	mediaListener = null;
	resolvedUnsub = null;
	themeUnsub = null;

	ensureLightStylesheet();

	// 1. Read persisted preference.
	let saved: Theme = 'dark';
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw === 'dark' || raw === 'light' || raw === 'auto') saved = raw;
	} catch {
		// ignore
	}
	theme.set(saved);

	// 2. Prime system preference + attach matchMedia listener.
	mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
	systemPrefersDark.set(mediaQuery.matches);
	mediaListener = (e: MediaQueryListEvent) => systemPrefersDark.set(e.matches);
	mediaQuery.addEventListener('change', mediaListener);

	// 3. Apply the currently-resolved theme and subscribe for future changes.
	applyDocumentAttribute(get(resolvedTheme));
	resolvedUnsub = resolvedTheme.subscribe(applyDocumentAttribute);

	// Silence the unused-import lint on `theme` — already used above,
	// but also keep a subscription handle for teardown symmetry.
	themeUnsub = theme.subscribe(() => {});

	return () => {
		if (mediaQuery && mediaListener) {
			mediaQuery.removeEventListener('change', mediaListener);
		}
		mediaQuery = null;
		mediaListener = null;
		resolvedUnsub?.();
		themeUnsub?.();
		resolvedUnsub = null;
		themeUnsub = null;
	};
}

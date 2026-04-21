/**
 * Unit tests for the theme store.
 *
 * Scope: pure-store behavior — setter validation, cycle order, derived
 * resolution, localStorage persistence + fallback, matchMedia listener
 * wiring, idempotent style injection, SSR safety.
 *
 * Environment notes:
 *  - Vitest runs in Node (no jsdom). We fabricate the window / document /
 *    localStorage / matchMedia globals the store touches, then mock
 *    `$app/environment` so `browser` flips between true and false per
 *    test group. SSR tests leave `browser` false and verify no globals
 *    are touched.
 *  - The store caches module-level state (mediaQuery, listener,
 *    resolvedUnsub). We `vi.resetModules()` before every test so each
 *    loadTheme() returns a pristine instance.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { get } from 'svelte/store';

// --- Controllable `browser` flag ------------------------------------------
// vi.mock is hoisted — we reference a module-level `browserFlag` the tests
// mutate between blocks. Casting via globalThis keeps the hoist happy.
const browserState = { value: true };
vi.mock('$app/environment', () => ({
	get browser() {
		return browserState.value;
	},
}));

// --- Fabricated DOM / storage / matchMedia --------------------------------
// Each test's setup wires these onto globalThis so the store's `browser`
// branch can read them. They are intentionally minimal — only the methods
// theme.ts actually calls are implemented.

type FakeMediaListener = (e: { matches: boolean }) => void;

interface FakeMediaQueryList {
	matches: boolean;
	addEventListener: (type: 'change', listener: FakeMediaListener) => void;
	removeEventListener: (type: 'change', listener: FakeMediaListener) => void;
	// Test-only helpers
	_emit: (matches: boolean) => void;
	_listenerCount: () => number;
}

function createFakeMediaQuery(initialMatches: boolean): FakeMediaQueryList {
	const listeners = new Set<FakeMediaListener>();
	return {
		matches: initialMatches,
		addEventListener: (_type, listener) => {
			listeners.add(listener);
		},
		removeEventListener: (_type, listener) => {
			listeners.delete(listener);
		},
		_emit(matches: boolean) {
			this.matches = matches;
			for (const l of listeners) l({ matches });
		},
		_listenerCount() {
			return listeners.size;
		},
	};
}

interface FakeStorageBehavior {
	throwOnGet?: boolean;
	throwOnSet?: boolean;
	corruptRaw?: string | null;
}

function installFakeLocalStorage(behavior: FakeStorageBehavior = {}) {
	const store = new Map<string, string>();
	if (behavior.corruptRaw !== undefined && behavior.corruptRaw !== null) {
		store.set('vestige.theme', behavior.corruptRaw);
	}
	const fake = {
		getItem: (key: string) => {
			if (behavior.throwOnGet) throw new Error('SecurityError: storage disabled');
			return store.has(key) ? store.get(key)! : null;
		},
		setItem: (key: string, value: string) => {
			if (behavior.throwOnSet) throw new Error('QuotaExceededError');
			store.set(key, value);
		},
		removeItem: (key: string) => {
			store.delete(key);
		},
		clear: () => store.clear(),
		key: () => null,
		length: 0,
		_store: store, // test-only peek
	};
	vi.stubGlobal('localStorage', fake);
	return fake;
}

/**
 * Install a fake `document` with only the APIs theme.ts calls:
 *  - getElementById (style-dedup check)
 *  - createElement('style')
 *  - head.appendChild
 *  - documentElement.dataset
 * Returns handles so tests can inspect the head children and data-theme.
 */
function installFakeDocument() {
	const headChildren: Array<{ id: string; textContent: string; tagName: string }> = [];
	const docEl = {
		dataset: {} as Record<string, string>,
	};
	const fakeDocument = {
		getElementById: (id: string) =>
			headChildren.find((el) => el.id === id) ?? null,
		createElement: (tag: string) => ({
			id: '',
			textContent: '',
			tagName: tag.toUpperCase(),
		}),
		head: {
			appendChild: (el: { id: string; textContent: string; tagName: string }) => {
				headChildren.push(el);
				return el;
			},
		},
		documentElement: docEl,
	};
	vi.stubGlobal('document', fakeDocument);
	return { fakeDocument, headChildren, docEl };
}

/**
 * Install a fake `window` with just `matchMedia`. We keep the returned
 * MQL handle so tests can emit change events.
 */
function installFakeWindow(initialPrefersDark: boolean) {
	const mql = createFakeMediaQuery(initialPrefersDark);
	const fakeWindow = {
		matchMedia: vi.fn(() => mql),
	};
	vi.stubGlobal('window', fakeWindow);
	return { fakeWindow, mql };
}

/**
 * Fresh module import. The theme store caches matchMedia/listener handles
 * at module level, so every test that exercises initTheme wants a clean
 * copy. Returns the full export surface.
 */
async function loadTheme() {
	vi.resetModules();
	return await import('../theme');
}

// Baseline: every test starts with browser=true, fake window/doc/storage
// installed, and fresh module state. SSR-specific tests override these.
beforeEach(() => {
	browserState.value = true;
	installFakeDocument();
	installFakeWindow(true); // system prefers dark by default
	installFakeLocalStorage();
});

afterEach(() => {
	vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Export surface
// ---------------------------------------------------------------------------
describe('theme store — exports', () => {
	it('exports theme writable, resolvedTheme derived, setTheme, cycleTheme, initTheme', async () => {
		const mod = await loadTheme();
		expect(mod.theme).toBeDefined();
		expect(typeof mod.theme.subscribe).toBe('function');
		expect(typeof mod.theme.set).toBe('function');
		expect(mod.resolvedTheme).toBeDefined();
		expect(typeof mod.resolvedTheme.subscribe).toBe('function');
		// Derived stores do NOT expose .set — this guards against accidental
		// conversion to a writable during refactors.
		expect((mod.resolvedTheme as unknown as { set?: unknown }).set).toBeUndefined();
		expect(typeof mod.setTheme).toBe('function');
		expect(typeof mod.cycleTheme).toBe('function');
		expect(typeof mod.initTheme).toBe('function');
	});

	it('theme defaults to dark before initTheme is called', async () => {
		const mod = await loadTheme();
		expect(get(mod.theme)).toBe('dark');
	});
});

// ---------------------------------------------------------------------------
// setTheme — input validation + persistence
// ---------------------------------------------------------------------------
describe('setTheme', () => {
	it('accepts dark/light/auto and updates the store', async () => {
		const { theme, setTheme } = await loadTheme();
		setTheme('light');
		expect(get(theme)).toBe('light');
		setTheme('auto');
		expect(get(theme)).toBe('auto');
		setTheme('dark');
		expect(get(theme)).toBe('dark');
	});

	it('rejects invalid values — store is unchanged, localStorage untouched', async () => {
		const { theme, setTheme } = await loadTheme();
		setTheme('light'); // seed a known value
		const ls = installFakeLocalStorage();
		// Reset any prior writes so we only see what happens during the bad call.
		ls._store.clear();

		// Cast a bad value through the public API.
		setTheme('midnight' as unknown as 'dark');
		expect(get(theme)).toBe('light'); // unchanged
		expect(ls._store.has('vestige.theme')).toBe(false);

		setTheme('' as unknown as 'dark');
		setTheme(undefined as unknown as 'dark');
		setTheme(null as unknown as 'dark');
		expect(get(theme)).toBe('light');
	});

	it('persists the valid value to localStorage under the vestige.theme key', async () => {
		const ls = installFakeLocalStorage();
		const { setTheme } = await loadTheme();
		setTheme('auto');
		expect(ls._store.get('vestige.theme')).toBe('auto');
	});

	it('swallows localStorage write errors (private mode / disabled storage)', async () => {
		installFakeLocalStorage({ throwOnSet: true });
		const { theme, setTheme } = await loadTheme();
		// Must not throw.
		expect(() => setTheme('light')).not.toThrow();
		// Store still updated even though persistence failed — UI should
		// reflect the click; the next session will just start fresh.
		expect(get(theme)).toBe('light');
	});

	it('no-ops localStorage write when browser=false (SSR safety)', async () => {
		browserState.value = false;
		const ls = installFakeLocalStorage();
		const { theme, setTheme } = await loadTheme();
		setTheme('light');
		// Store update is still safe (pure JS object), but persistence is skipped.
		expect(get(theme)).toBe('light');
		expect(ls._store.has('vestige.theme')).toBe(false);
	});
});

// ---------------------------------------------------------------------------
// cycleTheme — dark → light → auto → dark
// ---------------------------------------------------------------------------
describe('cycleTheme', () => {
	it('cycles dark → light', async () => {
		const { theme, cycleTheme } = await loadTheme();
		// Default is 'dark'.
		expect(get(theme)).toBe('dark');
		cycleTheme();
		expect(get(theme)).toBe('light');
	});

	it('cycles light → auto', async () => {
		const { theme, setTheme, cycleTheme } = await loadTheme();
		setTheme('light');
		cycleTheme();
		expect(get(theme)).toBe('auto');
	});

	it('cycles auto → dark (closes the loop)', async () => {
		const { theme, setTheme, cycleTheme } = await loadTheme();
		setTheme('auto');
		cycleTheme();
		expect(get(theme)).toBe('dark');
	});

	it('full triple-click returns to the starting value', async () => {
		const { theme, cycleTheme } = await loadTheme();
		const start = get(theme);
		cycleTheme();
		cycleTheme();
		cycleTheme();
		expect(get(theme)).toBe(start);
	});

	it('persists each step to localStorage', async () => {
		const ls = installFakeLocalStorage();
		const { cycleTheme } = await loadTheme();
		cycleTheme();
		expect(ls._store.get('vestige.theme')).toBe('light');
		cycleTheme();
		expect(ls._store.get('vestige.theme')).toBe('auto');
		cycleTheme();
		expect(ls._store.get('vestige.theme')).toBe('dark');
	});
});

// ---------------------------------------------------------------------------
// resolvedTheme — derived from theme + systemPrefersDark
// ---------------------------------------------------------------------------
describe('resolvedTheme', () => {
	it('dark → dark (independent of system preference)', async () => {
		const { resolvedTheme, setTheme } = await loadTheme();
		setTheme('dark');
		expect(get(resolvedTheme)).toBe('dark');
	});

	it('light → light (independent of system preference)', async () => {
		const { resolvedTheme, setTheme } = await loadTheme();
		setTheme('light');
		expect(get(resolvedTheme)).toBe('light');
	});

	it('auto + system prefers dark → dark', async () => {
		const { mql } = installFakeWindow(true);
		const { resolvedTheme, setTheme, initTheme } = await loadTheme();
		initTheme(); // primes systemPrefersDark from matchMedia
		setTheme('auto');
		expect(mql.matches).toBe(true);
		expect(get(resolvedTheme)).toBe('dark');
	});

	it('auto + system prefers light → light', async () => {
		installFakeWindow(false);
		const { resolvedTheme, setTheme, initTheme } = await loadTheme();
		initTheme(); // primes systemPrefersDark=false
		setTheme('auto');
		expect(get(resolvedTheme)).toBe('light');
	});

	it('auto flips live when the matchMedia listener fires (OS changes scheme)', async () => {
		const { mql } = installFakeWindow(true);
		const { resolvedTheme, setTheme, initTheme } = await loadTheme();
		initTheme();
		setTheme('auto');
		expect(get(resolvedTheme)).toBe('dark');
		// OS user toggles to light mode — matchMedia fires 'change' with matches=false.
		mql._emit(false);
		expect(get(resolvedTheme)).toBe('light');
		// And back to dark.
		mql._emit(true);
		expect(get(resolvedTheme)).toBe('dark');
	});
});

// ---------------------------------------------------------------------------
// initTheme — idempotence, teardown, localStorage hydration
// ---------------------------------------------------------------------------
describe('initTheme', () => {
	it('returns a teardown function', async () => {
		const { initTheme } = await loadTheme();
		const teardown = initTheme();
		expect(typeof teardown).toBe('function');
		teardown();
	});

	it('injects exactly one <style id="vestige-theme-light"> into <head>', async () => {
		const { headChildren } = installFakeDocument();
		const { initTheme } = await loadTheme();
		initTheme();
		const styleEls = headChildren.filter((el) => el.id === 'vestige-theme-light');
		expect(styleEls.length).toBe(1);
		expect(styleEls[0].tagName).toBe('STYLE');
		// Sanity — CSS uses the REAL token names from app.css.
		expect(styleEls[0].textContent).toContain('--color-void');
		expect(styleEls[0].textContent).toContain('--color-bright');
		expect(styleEls[0].textContent).toContain('--color-text');
		expect(styleEls[0].textContent).toContain("[data-theme='light']");
	});

	it('is idempotent — double init does NOT duplicate the style element', async () => {
		const { headChildren } = installFakeDocument();
		const { initTheme } = await loadTheme();
		initTheme();
		initTheme();
		initTheme();
		const styleEls = headChildren.filter((el) => el.id === 'vestige-theme-light');
		expect(styleEls.length).toBe(1);
	});

	it('double init does not leak matchMedia listeners (tears down the prior one)', async () => {
		const { mql } = installFakeWindow(true);
		const { initTheme } = await loadTheme();
		initTheme();
		expect(mql._listenerCount()).toBe(1);
		initTheme();
		// Still exactly one — the second init removed the first before adding a new one.
		expect(mql._listenerCount()).toBe(1);
		initTheme();
		expect(mql._listenerCount()).toBe(1);
	});

	it('teardown removes the matchMedia listener', async () => {
		const { mql } = installFakeWindow(true);
		const { initTheme } = await loadTheme();
		const teardown = initTheme();
		expect(mql._listenerCount()).toBe(1);
		teardown();
		expect(mql._listenerCount()).toBe(0);
	});

	it('hydrates theme from localStorage when a valid value is stored', async () => {
		installFakeLocalStorage({ corruptRaw: 'light' });
		const { theme, initTheme } = await loadTheme();
		initTheme();
		expect(get(theme)).toBe('light');
	});

	it('falls back to dark when localStorage contains a corrupt/unknown value', async () => {
		installFakeLocalStorage({ corruptRaw: 'hyperdark' });
		const { theme, initTheme } = await loadTheme();
		initTheme();
		expect(get(theme)).toBe('dark');
	});

	it('falls back to dark when localStorage.getItem throws (private mode)', async () => {
		installFakeLocalStorage({ throwOnGet: true });
		const { theme, initTheme } = await loadTheme();
		// Must not throw — error swallowed, default preserved.
		expect(() => initTheme()).not.toThrow();
		expect(get(theme)).toBe('dark');
	});

	it('writes documentElement.dataset.theme to the resolved value', async () => {
		const { docEl } = installFakeDocument();
		installFakeWindow(true);
		const { setTheme, initTheme } = await loadTheme();
		initTheme();
		setTheme('light');
		expect(docEl.dataset.theme).toBe('light');
		setTheme('dark');
		expect(docEl.dataset.theme).toBe('dark');
		// auto + system=dark → 'dark'
		setTheme('auto');
		expect(docEl.dataset.theme).toBe('dark');
	});

	it('uses the correct matchMedia query: (prefers-color-scheme: dark)', async () => {
		const { fakeWindow } = installFakeWindow(true);
		const { initTheme } = await loadTheme();
		initTheme();
		expect(fakeWindow.matchMedia).toHaveBeenCalledWith('(prefers-color-scheme: dark)');
	});
});

// ---------------------------------------------------------------------------
// SSR safety — browser=false means every function is a safe no-op
// ---------------------------------------------------------------------------
describe('SSR safety (browser=false)', () => {
	beforeEach(() => {
		browserState.value = false;
		// Deliberately DO NOT install fake window/document/localStorage.
		// If the store touches them while browser=false, ReferenceError fires.
		vi.unstubAllGlobals();
		// But `setup.ts` (shared graph test setup) installs a minimal global
		// `document` stub. That's fine — the point is the store must not
		// call window.matchMedia or localStorage while browser=false.
	});

	it('initTheme returns a no-op teardown and does not throw', async () => {
		const { initTheme } = await loadTheme();
		let teardown: () => void = () => {};
		expect(() => {
			teardown = initTheme();
		}).not.toThrow();
		expect(typeof teardown).toBe('function');
		expect(() => teardown()).not.toThrow();
	});

	it('setTheme updates the store but skips localStorage', async () => {
		const { theme, setTheme } = await loadTheme();
		expect(() => setTheme('light')).not.toThrow();
		expect(get(theme)).toBe('light');
	});

	it('cycleTheme cycles without touching browser globals', async () => {
		const { theme, cycleTheme } = await loadTheme();
		expect(() => cycleTheme()).not.toThrow();
		expect(get(theme)).toBe('light');
	});

	it('resolvedTheme returns the concrete value for dark/light, defaults to dark for auto', async () => {
		const { resolvedTheme, setTheme } = await loadTheme();
		setTheme('dark');
		expect(get(resolvedTheme)).toBe('dark');
		setTheme('light');
		expect(get(resolvedTheme)).toBe('light');
		// In SSR we never primed matchMedia, so systemPrefersDark is its
		// default (true) → auto resolves to dark. This keeps server-rendered
		// HTML matching the dark-first design.
		setTheme('auto');
		expect(get(resolvedTheme)).toBe('dark');
	});
});

import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const appVersion = process.env.VESTIGE_DASHBOARD_VERSION ?? process.env.npm_package_version ?? 'dev';

// Base path the app is served from. Defaults to '/dashboard' for local dev and
// the embedded release binary. CI overrides it (e.g. '/vestige') so assets
// resolve correctly when published to a GitHub Pages project subpath.
const basePath = process.env.VESTIGE_BASE_PATH ?? '/dashboard';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: true,
			strict: false
		}),
		paths: {
			base: basePath
		},
		version: {
			name: appVersion
		},
		alias: {
			$lib: 'src/lib',
			$components: 'src/lib/components',
			$stores: 'src/lib/stores',
			$types: 'src/lib/types'
		}
	}
};

export default config;

/// <reference types="vitest/config" />
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

// The dashboard HTTP server is the REST/WebSocket surface on 3931. 3928 is
// the MCP transport, so proxying the browser's /api and /ws requests there
// makes every data-backed organ fail with a 404.
const dashboardApiTarget = process.env.VESTIGE_API_TARGET ?? 'http://127.0.0.1:3931';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		port: 5173,
		proxy: {
			// SvelteKit prefixes absolute browser fetches with kit.paths.base, so the
			// development request is /dashboard/api/... rather than /api/.... Strip
			// that UI-only prefix before it reaches the dashboard server.
			'/dashboard/api': {
				target: dashboardApiTarget,
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/dashboard/, '')
			},
			'/api': {
				target: dashboardApiTarget,
				changeOrigin: true
			},
			'/ws': {
				target: dashboardApiTarget.replace('http', 'ws'),
				ws: true
			}
		}
	},
	test: {
		include: ['src/**/*.test.ts'],
		environment: 'node',
		setupFiles: ['src/lib/graph/__tests__/setup.ts'],
		alias: {
			$lib: new URL('./src/lib', import.meta.url).pathname,
			$components: new URL('./src/lib/components', import.meta.url).pathname,
			$stores: new URL('./src/lib/stores', import.meta.url).pathname,
			$types: new URL('./src/lib/types', import.meta.url).pathname,
		},
	},
});

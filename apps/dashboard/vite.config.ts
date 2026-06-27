/// <reference types="vitest/config" />
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		port: process.env.PORT ? Number(process.env.PORT) : 5173,
		proxy: {
			'/api': {
				target: 'http://127.0.0.1:3927',
				changeOrigin: true
			},
			'/ws': {
				target: 'ws://127.0.0.1:3927',
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

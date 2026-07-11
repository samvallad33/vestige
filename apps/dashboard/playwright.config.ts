import { defineConfig } from '@playwright/test';

const PORT = 5199;

export default defineConfig({
	testDir: './e2e',
	timeout: 60_000,
	expect: { timeout: 10_000 },
	fullyParallel: false,
	retries: 0,
	reporter: [['list'], ['html', { open: 'never' }]],
	use: {
		baseURL: `http://localhost:${PORT}`,
		screenshot: 'on',
		video: 'retain-on-failure',
		trace: 'retain-on-failure',
		// Pin motion ON so the living-field motion assertions (isAnimating) are
		// stable: a CI/host that reports prefers-reduced-motion would otherwise
		// freeze every field and false-fail the "field is moving" gate.
		reducedMotion: 'no-preference',
		launchOptions: {
			args: ['--use-gl=angle', '--ignore-gpu-blocklist'],
		},
	},
	webServer: {
		command: `npx vite dev --port ${PORT}`,
		port: PORT,
		reuseExistingServer: true,
		timeout: 30_000,
	},
});

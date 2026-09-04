import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Privacy regression guard for the Observatory stage.
 *
 * `chrome='full'` renders overlays that carry REAL memory labels (spine
 * beats, verdict cards). The export/capture paths mount `ObservatoryStage`,
 * so the prop must default to `'none'` and no route may opt into `'full'`
 * without a deliberate change to this test.
 */
const OBSERVATORY = new URL('..', import.meta.url).pathname;
const ROUTES = new URL('../../../routes', import.meta.url).pathname;

function svelteFiles(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir)) {
		const full = join(dir, entry);
		if (statSync(full).isDirectory()) out.push(...svelteFiles(full));
		else if (full.endsWith('.svelte')) out.push(full);
	}
	return out;
}

describe('ObservatoryStage chrome contract', () => {
	it('defaults chrome to none so a forgetful mount never exports labels', () => {
		const src = readFileSync(join(OBSERVATORY, 'ObservatoryStage.svelte'), 'utf8');
		expect(src).toMatch(/\n\t\tchrome = 'none',/);
		expect(src).not.toMatch(/\n\t\tchrome = 'full',/);
	});

	it('no route mounts the stage with chrome="full"', () => {
		const offenders: string[] = [];
		for (const file of svelteFiles(ROUTES)) {
			const src = readFileSync(file, 'utf8');
			if (!src.includes('<ObservatoryStage')) continue;
			if (/chrome=["']full["']/.test(src)) offenders.push(file.slice(ROUTES.length));
		}
		expect(offenders).toEqual([]);
	});
});

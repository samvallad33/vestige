import { describe, expect, it } from 'vitest';
import type { Receipt } from '$lib/stores/api';
import { buildWitnessScene, witnessEvidenceIds } from './witness-scene';

const first = '11111111-1111-4111-8111-111111111111';
const second = '22222222-2222-4222-8222-222222222222';
const third = '33333333-3333-4333-8333-333333333333';
function receipt(path: string[], retrieved = [first, second]): Receipt {
	return { receipt_id: 'fixture-receipt', activation_path: path, retrieved,
		mutations: [], suppressed: [], trust_floor: 0.6, decay_risk: 'low' };
}

describe('Witness receipt evidence identities', () => {
	it('never hydrates or renders reasoning prose as a memory', () => {
		const value = receipt(['SYNTHESIS: 85% confidence / memory?query=private', `${first} -> ${second}`]);
		expect(witnessEvidenceIds(value)).toEqual([first, second]);
		const scene = buildWitnessScene(null, value, new Map());
		expect(scene.shards.map(shard => shard.id)).toEqual([first, second]);
		expect(scene.edges).toEqual([]);
	});
	it('preserves real ordered path identities and deduplicates attribution', () => {
		const value = receipt([third, first, second, first]);
		expect(witnessEvidenceIds(value)).toEqual([third, first, second]);
		expect(buildWitnessScene(null, value, new Map()).edges).toHaveLength(3);
	});
	it('does not bridge a prose gap into a claimed memory edge', () => {
		const value = receipt([first, 'reasoning gap 90%', second]);
		expect(buildWitnessScene(null, value, new Map()).edges).toEqual([]);
	});
	it('keeps explicit legacy identities and bounds the rendered scene', () => {
		const value = receipt(['legacy-memory'], ['legacy-memory', ...Array.from({ length: 100 }, (_, i) => `memory-${i}`)]);
		expect(witnessEvidenceIds(value)[0]).toBe('legacy-memory');
		expect(buildWitnessScene(null, value, new Map()).shards).toHaveLength(64);
		expect(witnessEvidenceIds(null)).toEqual([]);
	});
});

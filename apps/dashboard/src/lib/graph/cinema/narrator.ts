// Memory Cinema — narration tiers 1 & 2.
//
// Tier 1 (premium): a backend LLM endpoint (/api/narrative) authors rich prose
//   from the planned path. Used only when the backend advertises it.
// Tier 2 (smart local default): deterministic, structured captions generated
//   purely from the real node/edge data — no network, no LLM, instant. This is
//   what the static HN demo and any backend-without-LLM setup uses.
//
// Tier 3 (the BFS camera engine in director.ts) always runs underneath; the
// narrator only decides what TEXT accompanies each beat. If everything here
// fails, captions fall back to Tier 2, which cannot fail.

import type { CinemaBeat, CinemaPath } from './pathfinder';

export interface BeatNarration {
	nodeId: string;
	/** The caption shown + optionally spoken for this beat. */
	text: string;
	/** Short label for the beat kind, shown as a chip. */
	chip: string;
}

export type NarrationSource = 'backend-llm' | 'local-captions';

export interface CinemaNarration {
	source: NarrationSource;
	beats: BeatNarration[];
}

// `satisfies` makes the compiler error if a new CinemaBeat['kind'] is added
// without a chip here — closes the silent "undefined chip → blank UI" gap.
const KIND_CHIP = {
	origin: 'Origin',
	connection: 'Connection',
	contradiction: 'Tension',
	recent: 'Now',
	bridge: 'Jump',
	surprise: 'Surprise',
} satisfies Record<CinemaBeat['kind'], string>;

function snippet(content: string, max = 90): string {
	const s = (content ?? '').replace(/\s+/g, ' ').trim();
	if (s.length <= max) return s;
	return s.slice(0, max - 1).trimEnd() + '…';
}

function typeLabel(nodeType: string): string {
	const t = (nodeType ?? 'memory').toLowerCase();
	return t.charAt(0).toUpperCase() + t.slice(1);
}

/**
 * Tier 2 — deterministic structured captions from real data only.
 * Never throws; always returns a caption per beat.
 */
export function localCaptions(path: CinemaPath): CinemaNarration {
	const beats: BeatNarration[] = path.beats.map((beat, i) => {
		const n = beat.node;
		const what = snippet(n.label || `(${typeLabel(n.type)} memory)`);
		let text: string;
		switch (beat.kind) {
			case 'origin':
				text = `We begin at a ${typeLabel(n.type).toLowerCase()} the graph is centered on — "${what}".`;
				break;
			case 'contradiction': {
				const via = beat.viaEdge?.type ? beat.viaEdge.type.replace(/_/g, ' ') : 'a conflict';
				text = `This is held in tension with the last memory through ${via}: "${what}".`;
				break;
			}
			case 'recent':
				text = `And where the mind is now — a recent memory: "${what}".`;
				break;
			case 'bridge':
				text = `Crossing to a separate cluster — "${what}".`;
				break;
			default: {
				const w = beat.viaEdge?.weight ?? 0;
				const strength = w > 0.66 ? 'strongly' : w > 0.33 ? 'closely' : 'loosely';
				text = `${strength} connected from there: a ${typeLabel(n.type).toLowerCase()} — "${what}".`;
			}
		}
		// Tags add texture when present.
		if (n.tags && n.tags.length > 0 && i > 0) {
			text += ` [${n.tags.slice(0, 3).join(', ')}]`;
		}
		return { nodeId: beat.nodeId, text, chip: KIND_CHIP[beat.kind] };
	});
	return { source: 'local-captions', beats };
}

/**
 * Resolve the best available narration for a path.
 *
 * @param fetchBackend optional async fn that returns backend-LLM narration
 *   beats (Tier 1). If it's absent, rejects, times out, or returns a mismatched
 *   shape, we silently fall back to Tier 2 local captions. The caller passes
 *   this only when the backend has advertised /api/narrative support.
 */
export async function resolveNarration(
	path: CinemaPath,
	fetchBackend?: () => Promise<BeatNarration[] | null>
): Promise<CinemaNarration> {
	const fallback = localCaptions(path);
	if (!fetchBackend) return fallback;

	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		const backend = await Promise.race([
			fetchBackend(),
			new Promise<null>((resolve) => {
				timer = setTimeout(() => resolve(null), 6000);
			}),
		]);

		// Keep only well-formed backend beats (guards against null/empty/garbage
		// entries that would otherwise produce blank captions mid-tour).
		const valid = Array.isArray(backend)
			? backend.filter(
					(b): b is BeatNarration =>
						!!b && typeof b.nodeId === 'string' && typeof b.text === 'string' && b.text.trim().length > 0
				)
			: [];
		if (valid.length === 0) return fallback;

		// Align backend beats to the real path by nodeId; fill any gap from the
		// bounds-safe local caption so every beat always has text (never blank).
		const byNode = new Map(valid.map((b) => [b.nodeId, b]));
		const beats: BeatNarration[] = path.beats.map((beat, i) => {
			const hit = byNode.get(beat.nodeId);
			if (hit) {
				const chip = typeof hit.chip === 'string' && hit.chip.trim() ? hit.chip : KIND_CHIP[beat.kind];
				return { nodeId: beat.nodeId, text: hit.text, chip };
			}
			return (
				fallback.beats[i] ?? {
					nodeId: beat.nodeId,
					text: beat.node.label || '(unlabeled memory)',
					chip: KIND_CHIP[beat.kind],
				}
			);
		});
		return { source: 'backend-llm', beats };
	} catch {
		return fallback;
	} finally {
		if (timer) clearTimeout(timer);
	}
}

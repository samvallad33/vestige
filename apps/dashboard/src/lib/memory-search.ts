/**
 * The Memories search box. Vestige 4.0 finds a memory by exact handle, the
 * way `recall` does: a memory id, a unique id prefix (8 characters or more),
 * or an exact tag, across the whole store. It never matches free text, so a
 * query that is not a handle says so instead of showing an empty list that
 * reads like "nothing in your memory mentions this".
 */
import type { Memory, MemoryListResponse } from '$types';

export interface HandleSearchView {
	memories: Memory[];
	/** One line under the box: what matched, or why nothing did. */
	note: string;
}

export const HANDLE_SEARCH_HINT =
	'Vestige 4.0 finds memories by exact handle: a memory id, the start of one (8+ characters), or an exact tag. Free-text search is not available.';

export function handleSearchView(query: string, response: MemoryListResponse): HandleSearchView {
	const handle = query.trim();
	const memories = response.memories ?? [];
	const resolution = response.resolution;
	const total = response.total ?? memories.length;
	if (!resolution || memories.length === 0) {
		return {
			memories: [],
			note: `No memory has the id, id prefix or tag "${handle}". ${HANDLE_SEARCH_HINT}`
		};
	}
	if (resolution.ambiguous) {
		return {
			memories,
			note: `"${handle}" starts more than one memory id. Type more of it to pick one.`
		};
	}
	if (resolution.kind === 'tag') {
		const shown = memories.length < total ? `, showing ${memories.length}` : '';
		return {
			memories,
			note: `${total} ${total === 1 ? 'memory' : 'memories'} tagged "${handle}"${shown}.`
		};
	}
	return {
		memories,
		note: resolution.exact ? `Memory ${memories[0].id}.` : `Memory ${memories[0].id}, by its id prefix.`
	};
}

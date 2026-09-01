/**
 * Memory PR diff — structure the fetched `pr.diff` blob so the queue can
 * render proposed content, a target link, and before/after instead of
 * approving 38 PRs blind.
 */

export interface PrDiffView {
	targetId: string | null;
	proposed: string | null;
	before: string | null;
	after: string | null;
	kind: string | null;
	rest: { key: string; value: string }[];
}

const TARGET_KEYS = ['target_id', 'targetId', 'subject_id', 'subjectId', 'memory_id', 'memoryId', 'id'];
const PROPOSED_KEYS = ['proposed_content', 'proposedContent', 'proposed', 'new_content', 'newContent', 'content'];
const BEFORE_KEYS = ['before_content', 'beforeContent', 'before', 'old_content', 'oldContent', 'previous'];
const AFTER_KEYS = ['after_content', 'afterContent', 'after'];
const KIND_KEYS = ['kind', 'action', 'op'];

function asString(value: unknown): string | null {
	if (typeof value === 'string' && value.trim()) return value;
	if (typeof value === 'number' && Number.isFinite(value)) return String(value);
	return null;
}

function pick(record: Record<string, unknown>, keys: string[]): string | null {
	for (const key of keys) {
		const hit = asString(record[key]);
		if (hit) return hit;
	}
	for (const key of keys) {
		for (const [path, value] of Object.entries(record)) {
			if (path === key || path.endsWith(`.${key}`)) {
				const hit = asString(value);
				if (hit) return hit;
			}
		}
	}
	return null;
}

function flatten(value: unknown, prefix = ''): Record<string, unknown> {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return prefix ? { [prefix]: value as unknown } : {};
	}
	const out: Record<string, unknown> = {};
	for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
		const path = prefix ? `${prefix}.${k}` : k;
		if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(out, flatten(v, path));
		else out[path] = v;
	}
	return out;
}

export function viewMemoryPrDiff(diff: Record<string, unknown> | null | undefined): PrDiffView {
	const flat = flatten(diff ?? {});
	const used = new Set<string>();
	const take = (keys: string[]) => {
		const hit = pick(flat, keys);
		if (hit) {
			for (const key of keys) {
				if (flat[key] !== undefined) used.add(key);
				for (const path of Object.keys(flat)) {
					if (path === key || path.endsWith(`.${key}`)) used.add(path);
				}
			}
		}
		return hit;
	};
	const view: PrDiffView = {
		targetId: take(TARGET_KEYS),
		proposed: take(PROPOSED_KEYS),
		before: take(BEFORE_KEYS),
		after: take(AFTER_KEYS),
		kind: take(KIND_KEYS),
		rest: []
	};
	view.rest = Object.entries(flat)
		.filter(([key]) => !used.has(key))
		.map(([key, value]) => ({
			key,
			value: typeof value === 'string' ? value : JSON.stringify(value)
		}))
		.filter((row) => row.value && row.value !== '{}' && row.value !== 'null')
		.slice(0, 12);
	return view;
}

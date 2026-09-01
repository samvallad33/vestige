/**
 * Feed switchboard — every event that names a memory / run / receipt / PR
 * becomes a jump chip. Raw JSON is the fallback, never the primary UI.
 */

export interface FeedJump {
	kind: 'memory' | 'run' | 'receipt' | 'pr';
	id: string;
	href: string;
	label: string;
}

const ID_KEYS: { kind: FeedJump['kind']; keys: string[] }[] = [
	{ kind: 'memory', keys: ['memory_id', 'memoryId', 'id'] },
	{ kind: 'run', keys: ['run_id', 'runId'] },
	{ kind: 'receipt', keys: ['receipt_id', 'receiptId'] },
	{ kind: 'pr', keys: ['pr_id', 'prId'] }
];

function hrefFor(kind: FeedJump['kind'], id: string, basePath: string): string {
	if (kind === 'memory') return `${basePath}/memories?memory=${encodeURIComponent(id)}`;
	if (kind === 'run') return `${basePath}/blackbox?run=${encodeURIComponent(id)}`;
	if (kind === 'receipt') return `${basePath}/observatory?receipt=${encodeURIComponent(id)}`;
	return `${basePath}/memory-prs`;
}

const LABEL: Record<FeedJump['kind'], string> = {
	memory: 'Open memory',
	run: 'Open run',
	receipt: 'Open receipt',
	pr: 'Open PR queue'
};

function isUuidish(value: string): boolean {
	return value.length >= 4 && /[a-z0-9_-]/i.test(value);
}

export function feedJumps(
	data: Record<string, unknown>,
	eventType?: string,
	basePath = ''
): FeedJump[] {
	const out: FeedJump[] = [];
	const seen = new Set<string>();
	for (const { kind, keys } of ID_KEYS) {
		if (kind === 'memory' && eventType && /Heartbeat|TraceEvent/i.test(eventType)) continue;
		for (const key of keys) {
			const raw = data[key];
			if (typeof raw !== 'string' || !isUuidish(raw)) continue;
			if (kind === 'memory' && eventType === 'MemoryPrOpened') continue;
			const token = `${kind}:${raw}`;
			if (seen.has(token)) continue;
			seen.add(token);
			out.push({ kind, id: raw, href: hrefFor(kind, raw, basePath), label: LABEL[kind] });
			break;
		}
	}
	return out;
}

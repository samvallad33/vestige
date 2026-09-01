/**
 * Timeline audit diffs — the payload is already fetched; this shapes
 * old→new + reason so the receipt is not action+time only.
 */

export interface AuditDiffLike {
	action: string;
	timestamp: string;
	old_value?: number;
	new_value?: number;
	reason?: string;
	triggered_by?: string;
}

export function formatRetentionArrow(oldValue?: number, newValue?: number): string | null {
	if (typeof oldValue !== 'number' || typeof newValue !== 'number') return null;
	if (!Number.isFinite(oldValue) || !Number.isFinite(newValue)) return null;
	const lo = Math.round(oldValue * 100);
	const hi = Math.round(newValue * 100);
	const delta = hi - lo;
	const sign = delta > 0 ? '+' : '';
	return `${lo}% → ${hi}% (${sign}${delta})`;
}

export function auditDiffLines(event: AuditDiffLike): string[] {
	const lines: string[] = [];
	const arrow = formatRetentionArrow(event.old_value, event.new_value);
	if (arrow) lines.push(arrow);
	if (event.reason?.trim()) lines.push(event.reason.trim());
	if (event.triggered_by?.trim()) lines.push(`by ${event.triggered_by.trim()}`);
	return lines;
}

export function isRewrittenMemory(memory: { createdAt?: string; updatedAt?: string }): boolean {
	return Boolean(memory.createdAt && memory.updatedAt && memory.createdAt !== memory.updatedAt);
}

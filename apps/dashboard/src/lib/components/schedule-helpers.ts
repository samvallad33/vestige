/**
 * Pure helpers for the FSRS review schedule page + calendar.
 *
 * Extracted from `FSRSCalendar.svelte` and `routes/(app)/schedule/+page.svelte`
 * so that bucket / grid / urgency / retention math can be tested in isolation
 * (vitest `environment: node`, no jsdom required).
 */
import type { Memory } from '$types';

export const MS_DAY = 24 * 60 * 60 * 1000;

/**
 * Zero-out the time component of a date, returning a NEW Date at local
 * midnight. Used for day-granular bucketing so comparisons are stable across
 * any hour-of-day the user loads the page.
 */
export function startOfDay(d: Date | string): Date {
	const x = typeof d === 'string' ? new Date(d) : new Date(d);
	x.setHours(0, 0, 0, 0);
	return x;
}

/**
 * Signed integer count of whole local days between two timestamps, normalized
 * to midnight. Positive means `a` is in the future relative to `b`, negative
 * means `a` is in the past. Zero means same calendar day.
 */
export function daysBetween(a: Date, b: Date): number {
	return Math.floor((startOfDay(a).getTime() - startOfDay(b).getTime()) / MS_DAY);
}

/** YYYY-MM-DD in LOCAL time (not UTC) so calendar cells align with user's day. */
export function isoDate(d: Date): string {
	const y = d.getFullYear();
	const m = String(d.getMonth() + 1).padStart(2, '0');
	const day = String(d.getDate()).padStart(2, '0');
	return `${y}-${m}-${day}`;
}

/**
 * Urgency bucket for a review date relative to "now". Used by the right-hand
 * list and the calendar cell color. Day-granular (not hour-granular) so a
 * memory due at 23:59 today does not suddenly become "in 1d" at 00:01
 * tomorrow UX-wise — it becomes "overdue" cleanly at midnight.
 *
 * - `none` — no valid `nextReviewAt`
 * - `overdue` — due date's calendar day is strictly before today
 * - `today` — due date's calendar day is today
 * - `week` — due in 1..=7 whole days
 * - `future` — due in 8+ whole days
 */
export type Urgency = 'none' | 'overdue' | 'today' | 'week' | 'future';

export function classifyUrgency(now: Date, nextReviewAt: string | null | undefined): Urgency {
	if (!nextReviewAt) return 'none';
	const d = new Date(nextReviewAt);
	if (Number.isNaN(d.getTime())) return 'none';
	const delta = daysBetween(d, now);
	if (delta < 0) return 'overdue';
	if (delta === 0) return 'today';
	if (delta <= 7) return 'week';
	return 'future';
}

/**
 * Signed whole-day count from today → due date. Negative means overdue by
 * |n| days; zero means today; positive means n days out. Returns `null`
 * if the ISO string is invalid or missing.
 */
export function daysUntilReview(now: Date, nextReviewAt: string | null | undefined): number | null {
	if (!nextReviewAt) return null;
	const d = new Date(nextReviewAt);
	if (Number.isNaN(d.getTime())) return null;
	return daysBetween(d, now);
}

/**
 * The [start, end) window for the week containing `d`, starting Sunday at
 * local midnight. End is the following Sunday at local midnight — exclusive.
 */
export function weekBucketRange(d: Date): { start: Date; end: Date } {
	const start = startOfDay(d);
	start.setDate(start.getDate() - start.getDay()); // back to Sunday
	const end = new Date(start);
	end.setDate(end.getDate() + 7);
	return { start, end };
}

/**
 * Mean retention strength across a list of memories. Returns 0 for an empty
 * list (never NaN) so the sidebar can safely render "0%".
 */
export function avgRetention(memories: Memory[]): number {
	if (memories.length === 0) return 0;
	let sum = 0;
	for (const m of memories) sum += m.retentionStrength ?? 0;
	return sum / memories.length;
}

/**
 * Given a day-index `i` into a 42-cell calendar grid (6 rows × 7 cols), return
 * its row / column. The grid is laid out row-major: cell 0 = row 0 col 0,
 * cell 7 = row 1 col 0, cell 41 = row 5 col 6. Returns `null` for indices
 * outside `[0, 42)`.
 */
export function gridCellPosition(i: number): { row: number; col: number } | null {
	if (!Number.isInteger(i) || i < 0 || i >= 42) return null;
	return { row: Math.floor(i / 7), col: i % 7 };
}

/**
 * The inverse: given a calendar anchor date (today), compute the Sunday
 * at-or-before `anchor - 14 days` that seeds row 0 of the 6×7 grid. Pure,
 * deterministic, local-time.
 */
export function gridStartForAnchor(anchor: Date): Date {
	const base = startOfDay(anchor);
	base.setDate(base.getDate() - 14);
	base.setDate(base.getDate() - base.getDay()); // back to Sunday
	return base;
}

/**
 * Bucket counts used by the sidebar stats block. Day-granular, consistent
 * with `classifyUrgency`.
 */
export interface ScheduleStats {
	overdue: number;
	dueToday: number;
	dueThisWeek: number;
	dueThisMonth: number;
	avgDays: number;
}

export function computeScheduleStats(now: Date, scheduled: Memory[]): ScheduleStats {
	let overdue = 0;
	let dueToday = 0;
	let dueThisWeek = 0;
	let dueThisMonth = 0;
	let sumDays = 0;
	let futureCount = 0;
	const today = startOfDay(now);
	for (const m of scheduled) {
		if (!m.nextReviewAt) continue;
		const d = new Date(m.nextReviewAt);
		if (Number.isNaN(d.getTime())) continue;
		const delta = daysBetween(d, now);
		if (delta < 0) overdue++;
		if (delta <= 0) dueToday++;
		if (delta <= 7) dueThisWeek++;
		if (delta <= 30) dueThisMonth++;
		if (delta >= 0) {
			// Use hour-resolution days-until for the average so "due in 2.3 days"
			// is informative even when bucketing is day-granular elsewhere.
			sumDays += (d.getTime() - today.getTime()) / MS_DAY;
			futureCount++;
		}
	}
	const avgDays = futureCount > 0 ? sumDays / futureCount : 0;
	return { overdue, dueToday, dueThisWeek, dueThisMonth, avgDays };
}

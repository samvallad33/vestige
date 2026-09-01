/**
 * CameraRig — user orbit/zoom/pan offsets on top of the deterministic loop
 * camera. Capture/export ignore the rig so ?frame=N and loop clips stay
 * byte-identical. pickAt MUST use the same rig as compute() so the click
 * mirror stays exact by construction.
 */

import { orbitCamera, type OrbitCamera } from './camera';

export interface CameraRigState {
	/** Extra yaw (radians) added to the loop-phase orbit. */
	yaw: number;
	/** Added to the default 0.35 elevation. */
	pitch: number;
	/** Distance divisor. 1 = default; >1 dollies in. */
	zoom: number;
}

export const IDENTITY_RIG: CameraRigState = { yaw: 0, pitch: 0, zoom: 1 };

export const RIG_ZOOM_MIN = 0.38;
export const RIG_ZOOM_MAX = 2.6;
export const RIG_PITCH_MIN = -0.18;
export const RIG_PITCH_MAX = 0.82;

export function clamp(n: number, lo: number, hi: number): number {
	return Math.min(hi, Math.max(lo, n));
}

export function normalizeRig(rig: CameraRigState): CameraRigState {
	return {
		yaw: Number.isFinite(rig.yaw) ? rig.yaw : 0,
		pitch: clamp(Number.isFinite(rig.pitch) ? rig.pitch : 0, RIG_PITCH_MIN, RIG_PITCH_MAX),
		zoom: clamp(Number.isFinite(rig.zoom) ? rig.zoom : 1, RIG_ZOOM_MIN, RIG_ZOOM_MAX)
	};
}

/** Same orbit as NodeRenderer, with the user's rig applied. */
export function orbitWithRig(
	phase: number,
	aspect: number,
	distance: number,
	rig: CameraRigState = IDENTITY_RIG
): OrbitCamera {
	const n = normalizeRig(rig);
	return orbitCamera(phase, aspect, distance / n.zoom, 0.35 + n.pitch, n.yaw);
}

export class CameraRigController {
	state: CameraRigState = { ...IDENTITY_RIG };
	private dragging = false;
	private pointerId: number | null = null;
	private lastX = 0;
	private lastY = 0;
	private pinch0 = 0;
	private pointers = new Map<number, { x: number; y: number }>();
	enabled = true;

	reset(): void {
		this.state = { ...IDENTITY_RIG };
		this.dragging = false;
		this.pointerId = null;
		this.pointers.clear();
	}

	onPointerDown(e: PointerEvent): void {
		if (!this.enabled || e.button !== 0) return;
		this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
		if (this.pointers.size === 1) {
			this.dragging = true;
			this.pointerId = e.pointerId;
			this.lastX = e.clientX;
			this.lastY = e.clientY;
			(e.currentTarget as HTMLElement | null)?.setPointerCapture?.(e.pointerId);
		} else if (this.pointers.size === 2) {
			this.pinch0 = pinchDistance(this.pointers);
		}
	}

	onPointerMove(e: PointerEvent): boolean {
		if (!this.enabled) return false;
		if (this.pointers.has(e.pointerId)) {
			this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
		}
		if (this.pointers.size === 2 && this.pinch0 > 0) {
			const d = pinchDistance(this.pointers);
			const ratio = d / this.pinch0;
			this.state = normalizeRig({
				...this.state,
				zoom: this.state.zoom * clamp(ratio, 0.94, 1.06)
			});
			this.pinch0 = d;
			return true;
		}
		if (!this.dragging || e.pointerId !== this.pointerId) return false;
		const dx = e.clientX - this.lastX;
		const dy = e.clientY - this.lastY;
		this.lastX = e.clientX;
		this.lastY = e.clientY;
		this.state = normalizeRig({
			yaw: this.state.yaw - dx * 0.005,
			pitch: this.state.pitch + dy * 0.003,
			zoom: this.state.zoom
		});
		return true;
	}

	onPointerUp(e: PointerEvent): void {
		this.pointers.delete(e.pointerId);
		if (e.pointerId === this.pointerId) {
			this.dragging = false;
			this.pointerId = null;
		}
		if (this.pointers.size < 2) this.pinch0 = 0;
	}

	onWheel(e: WheelEvent): boolean {
		if (!this.enabled) return false;
		e.preventDefault();
		const delta = e.deltaY > 0 ? 0.92 : 1.08;
		this.state = normalizeRig({
			...this.state,
			zoom: this.state.zoom * delta
		});
		return true;
	}
}

function pinchDistance(pointers: Map<number, { x: number; y: number }>): number {
	const pts = [...pointers.values()];
	if (pts.length < 2) return 0;
	return Math.hypot(pts[0]!.x - pts[1]!.x, pts[0]!.y - pts[1]!.y);
}

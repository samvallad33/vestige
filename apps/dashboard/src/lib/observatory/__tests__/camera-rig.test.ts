import { describe, it, expect } from 'vitest';
import { orbitCamera } from '../camera';
import {
	CameraRigController,
	IDENTITY_RIG,
	clamp,
	normalizeRig,
	orbitWithRig,
	RIG_ZOOM_MAX,
	RIG_ZOOM_MIN
} from '../camera-rig';

describe('camera rig', () => {
	it('identity rig matches the deterministic orbit', () => {
		const a = orbitCamera(0.25, 16 / 9, 300);
		const b = orbitWithRig(0.25, 16 / 9, 300, IDENTITY_RIG);
		expect([...b.viewProj]).toEqual([...a.viewProj]);
		expect(b.eye).toEqual(a.eye);
	});

	it('same phase + same rig → identical camera (determinism)', () => {
		const rig = normalizeRig({ yaw: 0.4, pitch: 0.1, zoom: 1.3 });
		const a = orbitWithRig(0.61, 1, 300, rig);
		const b = orbitWithRig(0.61, 1, 300, rig);
		expect([...a.viewProj]).toEqual([...b.viewProj]);
	});

	it('zoom dollies in without breaking aspect', () => {
		const far = orbitWithRig(0, 1, 300, IDENTITY_RIG);
		const near = orbitWithRig(0, 1, 300, { yaw: 0, pitch: 0, zoom: 2 });
		const farR = Math.hypot(...far.eye);
		const nearR = Math.hypot(...near.eye);
		expect(nearR).toBeLessThan(farR);
	});

	it('clamps zoom and pitch', () => {
		const n = normalizeRig({ yaw: 12, pitch: 9, zoom: 99 });
		expect(n.zoom).toBe(RIG_ZOOM_MAX);
		expect(n.pitch).toBeLessThanOrEqual(0.82);
		expect(normalizeRig({ yaw: 0, pitch: 0, zoom: 0 }).zoom).toBe(RIG_ZOOM_MIN);
	});

	it('wheel zooms; drag yaws', () => {
		const c = new CameraRigController();
		const el = { preventDefault() {} } as unknown as WheelEvent;
		Object.assign(el, { deltaY: -100 });
		c.onWheel(el);
		expect(c.state.zoom).toBeGreaterThan(1);
		c.onPointerDown({
			button: 0,
			pointerId: 1,
			clientX: 10,
			clientY: 10,
			currentTarget: { setPointerCapture() {} }
		} as unknown as PointerEvent);
		c.onPointerMove({
			pointerId: 1,
			clientX: 40,
			clientY: 10
		} as unknown as PointerEvent);
		expect(c.state.yaw).not.toBe(0);
	});

	it('clamp is inclusive', () => {
		expect(clamp(0, 0, 1)).toBe(0);
		expect(clamp(1, 0, 1)).toBe(1);
		expect(clamp(2, 0, 1)).toBe(1);
	});
});

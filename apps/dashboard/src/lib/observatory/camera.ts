/**
 * Cognitive Observatory — minimal column-major mat4 camera math.
 *
 * Spec §6: no new dependencies — small local helpers instead of three.js.
 * All outputs are Float32Array(16) in WebGPU/WGSL column-major order.
 */

export type Mat4 = Float32Array;

/** Perspective projection (right-handed, depth 0..1 as WebGPU expects). */
export function perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
	const f = 1 / Math.tan(fovYRad / 2);
	const nf = 1 / (near - far);
	// column-major
	const m = new Float32Array(16);
	m[0] = f / aspect;
	m[5] = f;
	m[10] = far * nf;
	m[11] = -1;
	m[14] = far * near * nf;
	return m;
}

/** Right-handed lookAt view matrix. */
export function lookAt(
	eye: [number, number, number],
	target: [number, number, number],
	up: [number, number, number]
): Mat4 {
	const [ex, ey, ez] = eye;
	let zx = ex - target[0];
	let zy = ey - target[1];
	let zz = ez - target[2];
	let len = Math.hypot(zx, zy, zz) || 1;
	zx /= len;
	zy /= len;
	zz /= len;

	// x = up × z
	let xx = up[1] * zz - up[2] * zy;
	let xy = up[2] * zx - up[0] * zz;
	let xz = up[0] * zy - up[1] * zx;
	len = Math.hypot(xx, xy, xz) || 1;
	xx /= len;
	xy /= len;
	xz /= len;

	// y = z × x
	const yx = zy * xz - zz * xy;
	const yy = zz * xx - zx * xz;
	const yz = zx * xy - zy * xx;

	const m = new Float32Array(16);
	m[0] = xx;
	m[1] = yx;
	m[2] = zx;
	m[4] = xy;
	m[5] = yy;
	m[6] = zy;
	m[8] = xz;
	m[9] = yz;
	m[10] = zz;
	m[12] = -(xx * ex + xy * ey + xz * ez);
	m[13] = -(yx * ex + yy * ey + yz * ez);
	m[14] = -(zx * ex + zy * ey + zz * ez);
	m[15] = 1;
	return m;
}

/** out = a × b (column-major). */
export function multiply(a: Mat4, b: Mat4): Mat4 {
	const out = new Float32Array(16);
	for (let c = 0; c < 4; c++) {
		for (let r = 0; r < 4; r++) {
			out[c * 4 + r] =
				a[r] * b[c * 4] +
				a[4 + r] * b[c * 4 + 1] +
				a[8 + r] * b[c * 4 + 2] +
				a[12 + r] * b[c * 4 + 3];
		}
	}
	return out;
}

export interface OrbitCamera {
	viewProj: Mat4;
	/** world-space camera right vector (billboarding) */
	right: [number, number, number];
	/** world-space camera up vector (billboarding) */
	up: [number, number, number];
	eye: [number, number, number];
}

/**
 * Deterministic slow orbit camera: angle driven by the loop phase (frames),
 * never wall clock — the same frame always yields the same view. Returns the
 * view-projection plus the camera basis for GPU billboards.
 */
export function orbitCamera(
	phase: number,
	aspect: number,
	distance: number,
	elevation = 0.35
): OrbitCamera {
	const angle = phase * Math.PI * 2;
	const eye: [number, number, number] = [
		Math.sin(angle) * distance,
		distance * elevation,
		Math.cos(angle) * distance
	];
	const proj = perspective((50 * Math.PI) / 180, aspect, 0.1, 4000);
	const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);

	// camera basis: forward = normalize(target - eye), right = f × up, up = r × f
	let fx = -eye[0],
		fy = -eye[1],
		fz = -eye[2];
	let len = Math.hypot(fx, fy, fz) || 1;
	fx /= len;
	fy /= len;
	fz /= len;
	let rx = fy * 0 - fz * 1;
	let ry = fz * 0 - fx * 0;
	let rz = fx * 1 - fy * 0;
	len = Math.hypot(rx, ry, rz) || 1;
	rx /= len;
	ry /= len;
	rz /= len;
	const ux = ry * fz - rz * fy;
	const uy = rz * fx - rx * fz;
	const uz = rx * fy - ry * fx;

	return {
		viewProj: multiply(proj, view),
		right: [rx, ry, rz],
		up: [ux, uy, uz],
		eye
	};
}

/**
 * Spatial Palace orbit — a richer, palace-only motion that the SHARED graph
 * camera constants (ORBIT_DISTANCE in node-renderer.ts) must never be forced to
 * carry. The 19 organs are laid out on a wider field, so the palace flies at a
 * closer, fitted distance and adds living motion the flat graph orbit lacks:
 *
 *  - continuous azimuth orbit (loop_phase drives a full 360° per loop → the
 *    constellation slowly turns; seamless at the wrap because it is a pure
 *    function of phase),
 *  - a gentle vertical BOB (elevation eases up and down twice per loop) so the
 *    camera never sits on a dead rail,
 *  - a slow dolly BREATH (distance swells and contracts a few percent) so the
 *    field subtly inhales — flying through a living brain-galaxy, not a turntable.
 *
 * All three cycle an INTEGER number of times per loop, so the seam is invisible
 * and ?frame=N capture stays exact. Returns the same OrbitCamera contract as
 * orbitCamera(), so a palace pass writes the identical 24-float camera uniform.
 *
 * @param phase     loop phase 0..1 (engine params[1])
 * @param aspect    viewport w/h
 * @param distance  base fly distance (fitted to the palace field radius)
 */
export function palaceCamera(phase: number, aspect: number, distance: number): OrbitCamera {
	const tau = Math.PI * 2;
	// 1 full azimuth revolution per loop.
	const angle = phase * tau;
	// Vertical bob: elevation eases between ~0.20 and ~0.62, twice per loop.
	const elevation = 0.41 + 0.21 * Math.sin(phase * tau * 2);
	// Dolly breath: ±6% distance, three times per loop (integer → seamless).
	const dist = distance * (1 + 0.06 * Math.sin(phase * tau * 3));

	const eye: [number, number, number] = [
		Math.sin(angle) * dist,
		dist * elevation,
		Math.cos(angle) * dist
	];
	const proj = perspective((52 * Math.PI) / 180, aspect, 0.1, 6000);
	const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);

	// Camera basis (forward = normalize(target - eye)), same derivation as
	// orbitCamera — billboards + edge ribbons extrude against right/up.
	let fx = -eye[0],
		fy = -eye[1],
		fz = -eye[2];
	let len = Math.hypot(fx, fy, fz) || 1;
	fx /= len;
	fy /= len;
	fz /= len;
	let rx = fy * 0 - fz * 1;
	let ry = fz * 0 - fx * 0;
	let rz = fx * 1 - fy * 0;
	len = Math.hypot(rx, ry, rz) || 1;
	rx /= len;
	ry /= len;
	rz /= len;
	const ux = ry * fz - rz * fy;
	const uy = rz * fx - rx * fz;
	const uz = rx * fy - ry * fx;

	return {
		viewProj: multiply(proj, view),
		right: [rx, ry, rz],
		up: [ux, uy, uz],
		eye
	};
}

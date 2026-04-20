/**
 * Lightweight Three.js mock for unit/integration tests.
 * Implements the subset of Three.js APIs used by the graph modules.
 */
import { vi } from 'vitest';

export class Vector3 {
	x: number;
	y: number;
	z: number;

	constructor(x = 0, y = 0, z = 0) {
		this.x = x;
		this.y = y;
		this.z = z;
	}

	clone() {
		return new Vector3(this.x, this.y, this.z);
	}

	copy(v: Vector3) {
		this.x = v.x;
		this.y = v.y;
		this.z = v.z;
		return this;
	}

	set(x: number, y: number, z: number) {
		this.x = x;
		this.y = y;
		this.z = z;
		return this;
	}

	add(v: Vector3) {
		this.x += v.x;
		this.y += v.y;
		this.z += v.z;
		return this;
	}

	sub(v: Vector3) {
		this.x -= v.x;
		this.y -= v.y;
		this.z -= v.z;
		return this;
	}

	subVectors(a: Vector3, b: Vector3) {
		this.x = a.x - b.x;
		this.y = a.y - b.y;
		this.z = a.z - b.z;
		return this;
	}

	multiplyScalar(s: number) {
		this.x *= s;
		this.y *= s;
		this.z *= s;
		return this;
	}

	normalize() {
		const len = this.length() || 1;
		this.x /= len;
		this.y /= len;
		this.z /= len;
		return this;
	}

	length() {
		return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
	}

	distanceTo(v: Vector3) {
		const dx = this.x - v.x;
		const dy = this.y - v.y;
		const dz = this.z - v.z;
		return Math.sqrt(dx * dx + dy * dy + dz * dz);
	}

	lerp(v: Vector3, alpha: number) {
		this.x += (v.x - this.x) * alpha;
		this.y += (v.y - this.y) * alpha;
		this.z += (v.z - this.z) * alpha;
		return this;
	}

	setScalar(s: number) {
		this.x = s;
		this.y = s;
		this.z = s;
		return this;
	}

	addVectors(a: Vector3, b: Vector3) {
		this.x = a.x + b.x;
		this.y = a.y + b.y;
		this.z = a.z + b.z;
		return this;
	}

	applyQuaternion(_q: Quaternion) {
		// Mock: identity transform. Tests don't care about actual
		// camera-relative positioning; production uses real THREE math.
		return this;
	}
}

export class Quaternion {
	x = 0;
	y = 0;
	z = 0;
	w = 1;
}

export class QuadraticBezierCurve3 {
	v0: Vector3;
	v1: Vector3;
	v2: Vector3;
	constructor(v0: Vector3, v1: Vector3, v2: Vector3) {
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;
	}
	getPoint(t: number): Vector3 {
		// Standard quadratic Bezier evaluation, faithful enough for tests
		// that only care that points land on the curve.
		const one = 1 - t;
		return new Vector3(
			one * one * this.v0.x + 2 * one * t * this.v1.x + t * t * this.v2.x,
			one * one * this.v0.y + 2 * one * t * this.v1.y + t * t * this.v2.y,
			one * one * this.v0.z + 2 * one * t * this.v1.z + t * t * this.v2.z
		);
	}
}

export class Texture {
	needsUpdate = false;
	dispose() {}
}

export class Vector2 {
	x: number;
	y: number;
	constructor(x = 0, y = 0) {
		this.x = x;
		this.y = y;
	}
}

export class Color {
	r: number;
	g: number;
	b: number;

	constructor(colorOrR?: number | string, g?: number, b?: number) {
		if (typeof colorOrR === 'string') {
			this.r = 1;
			this.g = 1;
			this.b = 1;
		} else if (typeof colorOrR === 'number' && g === undefined) {
			this.r = ((colorOrR >> 16) & 255) / 255;
			this.g = ((colorOrR >> 8) & 255) / 255;
			this.b = (colorOrR & 255) / 255;
		} else {
			this.r = colorOrR ?? 1;
			this.g = g ?? 1;
			this.b = b ?? 1;
		}
	}

	clone() {
		const c = new Color();
		c.r = this.r;
		c.g = this.g;
		c.b = this.b;
		return c;
	}

	copy(c: Color) {
		this.r = c.r;
		this.g = c.g;
		this.b = c.b;
		return this;
	}

	lerp(c: Color, alpha: number) {
		this.r += (c.r - this.r) * alpha;
		this.g += (c.g - this.g) * alpha;
		this.b += (c.b - this.b) * alpha;
		return this;
	}

	setHSL(h: number, s: number, l: number) {
		this.r = h;
		this.g = s;
		this.b = l;
		return this;
	}

	offsetHSL(_h: number, _s: number, _l: number) {
		return this;
	}

	multiplyScalar(s: number) {
		this.r *= s;
		this.g *= s;
		this.b *= s;
		return this;
	}

	setRGB(r: number, g: number, b: number) {
		this.r = r;
		this.g = g;
		this.b = b;
		return this;
	}
}

export class BufferAttribute {
	array: Float32Array;
	itemSize: number;
	count: number;
	needsUpdate = false;

	constructor(array: Float32Array, itemSize: number) {
		this.array = array;
		this.itemSize = itemSize;
		this.count = array.length / itemSize;
	}

	getX(index: number) {
		return this.array[index * this.itemSize];
	}
	getY(index: number) {
		return this.array[index * this.itemSize + 1];
	}
	getZ(index: number) {
		return this.array[index * this.itemSize + 2];
	}

	setX(index: number, x: number) {
		this.array[index * this.itemSize] = x;
	}
	setY(index: number, y: number) {
		this.array[index * this.itemSize + 1] = y;
	}
	setZ(index: number, z: number) {
		this.array[index * this.itemSize + 2] = z;
	}

	setXYZ(index: number, x: number, y: number, z: number) {
		const i = index * this.itemSize;
		this.array[i] = x;
		this.array[i + 1] = y;
		this.array[i + 2] = z;
	}
}

export class BufferGeometry {
	attributes: Record<string, BufferAttribute> = {};

	setAttribute(name: string, attr: BufferAttribute) {
		this.attributes[name] = attr;
		return this;
	}

	getAttribute(name: string) {
		return this.attributes[name];
	}

	setFromPoints(points: Vector3[]) {
		const arr = new Float32Array(points.length * 3);
		points.forEach((p, i) => {
			arr[i * 3] = p.x;
			arr[i * 3 + 1] = p.y;
			arr[i * 3 + 2] = p.z;
		});
		this.setAttribute('position', new BufferAttribute(arr, 3));
		return this;
	}

	dispose() {}
}

export class SphereGeometry extends BufferGeometry {
	constructor(_radius?: number, _w?: number, _h?: number) {
		super();
	}
}

export class RingGeometry extends BufferGeometry {
	constructor(_inner?: number, _outer?: number, _segments?: number) {
		super();
	}
}

class BaseMaterial {
	color = new Color();
	transparent = false;
	opacity = 1;
	blending = 0;
	side = 0;
	map: { dispose: () => void } | null = null;
	emissive = new Color();
	emissiveIntensity = 0;
	roughness = 0;
	metalness = 0;
	depthTest = true;
	sizeAttenuation = true;
	size = 1;
	needsUpdate = false;

	dispose() {}
}

export class MeshStandardMaterial extends BaseMaterial {
	constructor(params?: Record<string, unknown>) {
		super();
		if (params) {
			if (params.color instanceof Color) this.color = params.color;
			if (params.emissive instanceof Color) this.emissive = params.emissive;
			if (typeof params.emissiveIntensity === 'number') this.emissiveIntensity = params.emissiveIntensity;
			if (typeof params.opacity === 'number') this.opacity = params.opacity;
			if (typeof params.transparent === 'boolean') this.transparent = params.transparent;
			if (typeof params.roughness === 'number') this.roughness = params.roughness;
			if (typeof params.metalness === 'number') this.metalness = params.metalness;
		}
	}
}

export class MeshBasicMaterial extends BaseMaterial {
	constructor(params?: Record<string, unknown>) {
		super();
		if (params) {
			if (typeof params.opacity === 'number') this.opacity = params.opacity;
			if (typeof params.transparent === 'boolean') this.transparent = params.transparent;
		}
	}
}

export class LineBasicMaterial extends BaseMaterial {
	depthWrite = true;
	constructor(params?: Record<string, unknown>) {
		super();
		if (params) {
			if (typeof params.opacity === 'number') this.opacity = params.opacity;
			if (typeof params.transparent === 'boolean') this.transparent = params.transparent;
			if (params.color instanceof Color) this.color = params.color;
			else if (typeof params.color === 'number') this.color = new Color(params.color);
			if (typeof params.blending === 'number') this.blending = params.blending;
			if (typeof params.depthWrite === 'boolean') this.depthWrite = params.depthWrite;
		}
	}
}

export class PointsMaterial extends BaseMaterial {
	constructor(params?: Record<string, unknown>) {
		super();
		if (params) {
			if (params.color instanceof Color) this.color = params.color;
			else if (typeof params.color === 'number') this.color = new Color(params.color);
			if (typeof params.size === 'number') this.size = params.size;
			if (typeof params.opacity === 'number') this.opacity = params.opacity;
		}
	}
}

export class SpriteMaterial extends BaseMaterial {
	depthWrite = true;
	constructor(params?: Record<string, unknown>) {
		super();
		if (params) {
			if (typeof params.opacity === 'number') this.opacity = params.opacity;
			if (typeof params.transparent === 'boolean') this.transparent = params.transparent;
			if (params.color instanceof Color) this.color = params.color;
			else if (typeof params.color === 'number') this.color = new Color(params.color);
			if (typeof params.blending === 'number') this.blending = params.blending;
			if (typeof params.depthWrite === 'boolean') this.depthWrite = params.depthWrite;
			if (params.map && typeof params.map === 'object') {
				this.map = params.map as { dispose: () => void };
			}
		}
	}
}

export class Object3D {
	position = new Vector3();
	scale = new Vector3(1, 1, 1);
	quaternion = new Quaternion();
	renderOrder = 0;
	userData: Record<string, unknown> = {};
	children: Object3D[] = [];
	parent: Object3D | null = null;

	add(child: Object3D) {
		this.children.push(child);
		child.parent = this;
		return this;
	}

	remove(child: Object3D) {
		const idx = this.children.indexOf(child);
		if (idx !== -1) {
			this.children.splice(idx, 1);
			child.parent = null;
		}
		return this;
	}

	traverse(callback: (obj: Object3D) => void) {
		callback(this);
		for (const child of this.children) {
			child.traverse(callback);
		}
	}

	lookAt(_target: Vector3) {}
}

export class Group extends Object3D {}
export class Scene extends Object3D {}

export class Mesh extends Object3D {
	geometry: BufferGeometry;
	material: BaseMaterial;

	constructor(geometry?: BufferGeometry, material?: BaseMaterial) {
		super();
		this.geometry = geometry ?? new BufferGeometry();
		this.material = material ?? new BaseMaterial();
	}
}

export class Line extends Object3D {
	geometry: BufferGeometry;
	material: BaseMaterial;

	constructor(geometry?: BufferGeometry, material?: BaseMaterial) {
		super();
		this.geometry = geometry ?? new BufferGeometry();
		this.material = material ?? new BaseMaterial();
	}
}

export class Points extends Object3D {
	geometry: BufferGeometry;
	material: BaseMaterial;

	constructor(geometry?: BufferGeometry, material?: BaseMaterial) {
		super();
		this.geometry = geometry ?? new BufferGeometry();
		this.material = material ?? new BaseMaterial();
	}
}

export class Sprite extends Object3D {
	material: BaseMaterial;

	constructor(material?: BaseMaterial) {
		super();
		this.material = material ?? new BaseMaterial();
	}
}

export class PerspectiveCamera extends Object3D {
	fov = 60;
	aspect = 1;
	near = 0.1;
	far = 2000;
}

export class Camera extends Object3D {}

export class CanvasTexture {
	needsUpdate = false;
	constructor(_canvas: unknown) {}
	dispose() {}
}

// Blending constants
export const AdditiveBlending = 2;
export const DoubleSide = 2;

// Install mock globally for 'three' imports
export function installThreeMock() {
	vi.mock('three', () => ({
		Vector3,
		Vector2,
		Color,
		Quaternion,
		QuadraticBezierCurve3,
		Texture,
		BufferAttribute,
		BufferGeometry,
		SphereGeometry,
		RingGeometry,
		MeshStandardMaterial,
		MeshBasicMaterial,
		LineBasicMaterial,
		PointsMaterial,
		SpriteMaterial,
		Object3D,
		Group,
		Scene,
		Mesh,
		Line,
		Points,
		Sprite,
		PerspectiveCamera,
		Camera,
		CanvasTexture,
		AdditiveBlending,
		DoubleSide,
	}));
}

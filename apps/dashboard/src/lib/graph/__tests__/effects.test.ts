import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('three', async () => {
	const mock = await import('./three-mock');
	return { ...mock };
});

import { EffectManager } from '../effects';
import { Vector3, Color, Scene } from './three-mock';

describe('EffectManager', () => {
	let scene: InstanceType<typeof Scene>;
	let effects: EffectManager;
	let nodeMeshMap: Map<string, any>;
	let camera: any;

	beforeEach(() => {
		scene = new Scene();
		effects = new EffectManager(scene as any);
		camera = { position: new Vector3(0, 30, 80) };
		nodeMeshMap = new Map();
	});

	function createMockMesh(id: string, pos: InstanceType<typeof Vector3>) {
		const mesh = {
			scale: new Vector3(1, 1, 1),
			position: pos.clone(),
			material: {
				emissive: new Color(0x000000),
				emissiveIntensity: 0.5,
			},
			userData: { nodeId: id },
		};
		nodeMeshMap.set(id, mesh);
		return mesh;
	}

	describe('pulse effects', () => {
		it('adds pulse and decays it over time', () => {
			createMockMesh('a', new Vector3(0, 0, 0));
			effects.addPulse('a', 1.0, new Color(0xff0000) as any, 0.1);

			expect(effects.pulseEffects.length).toBe(1);

			// Update a few times
			for (let i = 0; i < 5; i++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(effects.pulseEffects[0].intensity).toBeLessThan(1.0);
		});

		it('removes pulse when intensity reaches zero', () => {
			createMockMesh('a', new Vector3(0, 0, 0));
			effects.addPulse('a', 0.5, new Color(0xff0000) as any, 0.1);

			for (let i = 0; i < 20; i++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(effects.pulseEffects.length).toBe(0);
		});

		it('modulates mesh emissive color and intensity', () => {
			const mesh = createMockMesh('a', new Vector3(0, 0, 0));
			const pulseColor = new Color(0xff0000);
			effects.addPulse('a', 1.0, pulseColor as any, 0.05);

			effects.update(nodeMeshMap, camera);

			// Emissive intensity should be elevated
			expect(mesh.material.emissiveIntensity).toBeGreaterThan(0.5);
		});
	});

	describe('createSpawnBurst', () => {
		it('adds particles to the scene', () => {
			const childCount = scene.children.length;
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			expect(scene.children.length).toBe(childCount + 1);
		});

		it('creates 60 particles', () => {
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[scene.children.length - 1] as any;
			expect(pts.geometry.attributes.position.count).toBe(60);
		});

		it('particles move outward and fade over 120 frames', () => {
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			for (let i = 0; i < 121; i++) {
				effects.update(nodeMeshMap, camera);
			}

			// Burst should be cleaned up
			expect(scene.children.length).toBe(0);
		});

		it('particle opacity decreases over time', () => {
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[0] as any;

			effects.update(nodeMeshMap, camera);
			const earlyOpacity = pts.material.opacity;

			for (let i = 0; i < 60; i++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(pts.material.opacity).toBeLessThan(earlyOpacity);
		});
	});

	describe('createRainbowBurst', () => {
		it('creates 120 particles (2x normal burst)', () => {
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[scene.children.length - 1] as any;
			expect(pts.geometry.attributes.position.count).toBe(120);
		});

		it('has a 180-frame (3-second) lifespan', () => {
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			// Run for 179 frames — should still exist
			for (let i = 0; i < 179; i++) {
				effects.update(nodeMeshMap, camera);
			}
			expect(scene.children.length).toBe(1);

			// Frame 180 — should be cleaned up
			effects.update(nodeMeshMap, camera);
			expect(scene.children.length).toBe(1); // age increments to 180

			effects.update(nodeMeshMap, camera);
			expect(scene.children.length).toBe(0);
		});

		it('color cycles through rainbow HSL', () => {
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[0] as any;
			const initialColor = { r: pts.material.color.r, g: pts.material.color.g, b: pts.material.color.b };

			// Advance several frames
			for (let i = 0; i < 30; i++) {
				effects.update(nodeMeshMap, camera);
			}

			// Color should have changed due to HSL cycling
			const currentColor = pts.material.color;
			const colorChanged =
				Math.abs(currentColor.r - initialColor.r) > 0.01 ||
				Math.abs(currentColor.g - initialColor.g) > 0.01 ||
				Math.abs(currentColor.b - initialColor.b) > 0.01;
			expect(colorChanged).toBe(true);
		});

		it('particle size pulses (not monotonically decreasing)', () => {
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[0] as any;
			const sizes: number[] = [];

			for (let i = 0; i < 30; i++) {
				effects.update(nodeMeshMap, camera);
				sizes.push(pts.material.size);
			}

			// Check that size varies (pulses) — not monotonically decreasing
			let monotonic = true;
			for (let i = 1; i < sizes.length; i++) {
				if (sizes[i] > sizes[i - 1]) {
					monotonic = false;
					break;
				}
			}
			expect(monotonic).toBe(false);
		});

		it('has hueOffset attribute for per-particle variation', () => {
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);

			const pts = scene.children[0] as any;
			expect(pts.geometry.attributes.hueOffset).toBeDefined();
			expect(pts.geometry.attributes.hueOffset.count).toBe(120);
		});
	});

	describe('createRippleWave', () => {
		it('creates a ripple wave state', () => {
			const nodePositions = new Map<string, any>([
				['n1', new Vector3(5, 0, 0)],
				['n2', new Vector3(15, 0, 0)],
			]);
			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			// Ripple wave is internal state — verify it runs without error
			for (let i = 0; i < 100; i++) {
				effects.update(nodeMeshMap, camera, nodePositions);
			}
		});

		it('pulses nearby nodes as wavefront reaches them', () => {
			const nodePositions = new Map<string, any>([
				['close', new Vector3(5, 0, 0)],
				['far', new Vector3(50, 0, 0)],
			]);

			const closeMesh = createMockMesh('close', new Vector3(5, 0, 0));
			const farMesh = createMockMesh('far', new Vector3(50, 0, 0));

			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			// Run until wavefront reaches "close" (dist=5, speed=1.2, ~4 frames)
			for (let i = 0; i < 10; i++) {
				effects.update(nodeMeshMap, camera, nodePositions);
			}

			// Should have pulsed the close node — check for pulse effect
			const closeHasPulse = effects.pulseEffects.some((p) => p.nodeId === 'close');
			expect(closeHasPulse).toBe(true);
		});

		it('pulses each node only once', () => {
			const nodePositions = new Map<string, any>([
				['n1', new Vector3(3, 0, 0)],
			]);
			createMockMesh('n1', new Vector3(3, 0, 0));

			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			// Run many frames
			for (let i = 0; i < 30; i++) {
				effects.update(nodeMeshMap, camera, nodePositions);
			}

			// Count pulses for n1 — should be exactly 1
			const n1Pulses = effects.pulseEffects.filter((p) => p.nodeId === 'n1');
			// Could be 0 if already decayed, but should have been created once
			expect(n1Pulses.length).toBeLessThanOrEqual(1);
		});

		it('adds pulse to contacted nodes instead of direct scale mutation', () => {
			const nodePositions = new Map<string, any>([
				['bump', new Vector3(3, 0, 0)],
			]);
			createMockMesh('bump', new Vector3(3, 0, 0));

			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			// Run until wavefront reaches the node
			for (let i = 0; i < 10; i++) {
				effects.update(nodeMeshMap, camera, nodePositions);
			}

			// Ripple wave should add a pulse effect (not a direct scale mutation)
			const bumpPulses = effects.pulseEffects.filter(p => p.nodeId === 'bump');
			expect(bumpPulses.length).toBeGreaterThan(0);
		});

		it('completes and cleans up after 90 frames', () => {
			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			for (let i = 0; i < 95; i++) {
				effects.update(nodeMeshMap, camera, new Map());
			}

			// Internal rippleWaves array should be empty (no way to check directly,
			// but running more frames should not cause any errors)
			effects.update(nodeMeshMap, camera, new Map());
		});
	});

	describe('createImplosion', () => {
		it('creates 40 particles', () => {
			effects.createImplosion(new Vector3(5, 5, 5) as any, new Color(0xff4757) as any);

			const pts = scene.children[scene.children.length - 1] as any;
			expect(pts.geometry.attributes.position.count).toBe(40);
		});

		it('particles start spread out around the target', () => {
			const center = new Vector3(5, 5, 5);
			effects.createImplosion(center as any, new Color(0xff4757) as any);

			const pts = scene.children[0] as any;
			const positions = pts.geometry.attributes.position;

			// At least some particles should be far from center
			let maxDist = 0;
			for (let i = 0; i < positions.count; i++) {
				const px = positions.getX(i);
				const py = positions.getY(i);
				const pz = positions.getZ(i);
				const dist = Math.sqrt(
					(px - center.x) ** 2 + (py - center.y) ** 2 + (pz - center.z) ** 2
				);
				if (dist > maxDist) maxDist = dist;
			}
			expect(maxDist).toBeGreaterThan(2);
		});

		it('particles move INWARD toward center', () => {
			const center = new Vector3(0, 0, 0);
			effects.createImplosion(center as any, new Color(0xff4757) as any);

			const pts = scene.children[0] as any;
			const positions = pts.geometry.attributes.position;

			// Record initial average distance
			let initialAvgDist = 0;
			for (let i = 0; i < positions.count; i++) {
				const px = positions.getX(i);
				const py = positions.getY(i);
				const pz = positions.getZ(i);
				initialAvgDist += Math.sqrt(px * px + py * py + pz * pz);
			}
			initialAvgDist /= positions.count;

			// Advance 30 frames
			for (let f = 0; f < 30; f++) {
				effects.update(nodeMeshMap, camera);
			}

			// Record new average distance
			let newAvgDist = 0;
			for (let i = 0; i < positions.count; i++) {
				const px = positions.getX(i);
				const py = positions.getY(i);
				const pz = positions.getZ(i);
				newAvgDist += Math.sqrt(px * px + py * py + pz * pz);
			}
			newAvgDist /= positions.count;

			expect(newAvgDist).toBeLessThan(initialAvgDist);
		});

		it('creates a flash at convergence (frame 60)', () => {
			effects.createImplosion(new Vector3(0, 0, 0) as any, new Color(0xff4757) as any);

			// Run to convergence
			for (let f = 0; f < 60; f++) {
				effects.update(nodeMeshMap, camera);
			}

			// Should have particles + flash mesh
			expect(scene.children.length).toBe(2);
		});

		it('flash fades out and everything cleans up by frame 80', () => {
			effects.createImplosion(new Vector3(0, 0, 0) as any, new Color(0xff4757) as any);

			for (let f = 0; f < 85; f++) {
				effects.update(nodeMeshMap, camera);
			}

			// Everything should be cleaned up
			expect(scene.children.length).toBe(0);
		});

		it('flash sphere expands during fade-out', () => {
			effects.createImplosion(new Vector3(0, 0, 0) as any, new Color(0xff4757) as any);

			for (let f = 0; f < 65; f++) {
				effects.update(nodeMeshMap, camera);
			}

			// Find the flash mesh (should be the second child)
			const flash = scene.children.find(
				(c) => c instanceof Object && 'geometry' in c && !(('attributes' in (c as any).geometry))
			);

			// Flash should have expanded beyond scale 1
			if (flash) {
				expect(flash.scale.x).toBeGreaterThan(1);
			}
		});
	});

	describe('createShockwave', () => {
		it('adds a ring mesh to the scene', () => {
			effects.createShockwave(
				new Vector3(0, 0, 0) as any,
				new Color(0x00ffd1) as any,
				camera
			);

			expect(scene.children.length).toBe(1);
		});

		it('ring expands over time', () => {
			effects.createShockwave(
				new Vector3(0, 0, 0) as any,
				new Color(0x00ffd1) as any,
				camera
			);

			const ring = scene.children[0] as any;
			const initialScale = ring.scale.x;

			for (let f = 0; f < 30; f++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(ring.scale.x).toBeGreaterThan(initialScale);
		});

		it('ring fades out and cleans up after 60 frames', () => {
			effects.createShockwave(
				new Vector3(0, 0, 0) as any,
				new Color(0x00ffd1) as any,
				camera
			);

			for (let f = 0; f < 65; f++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(scene.children.length).toBe(0);
		});
	});

	describe('createConnectionFlash', () => {
		it('creates a line between two points', () => {
			effects.createConnectionFlash(
				new Vector3(0, 0, 0) as any,
				new Vector3(10, 10, 10) as any,
				new Color(0x00d4ff) as any
			);

			expect(scene.children.length).toBe(1);
		});

		it('fades out and cleans up', () => {
			effects.createConnectionFlash(
				new Vector3(0, 0, 0) as any,
				new Vector3(10, 10, 10) as any,
				new Color(0x00d4ff) as any
			);

			for (let f = 0; f < 100; f++) {
				effects.update(nodeMeshMap, camera);
			}

			expect(scene.children.length).toBe(0);
		});
	});

	describe('multiple simultaneous effects', () => {
		it('handles all effect types simultaneously', () => {
			const nodePositions = new Map<string, any>([
				['n1', new Vector3(5, 0, 0)],
			]);
			createMockMesh('n1', new Vector3(5, 0, 0));

			effects.addPulse('n1', 1.0, new Color(0xff0000) as any, 0.05);
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);
			effects.createRainbowBurst(new Vector3(5, 5, 5) as any, new Color(0xff00ff) as any);
			effects.createShockwave(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any, camera);
			effects.createConnectionFlash(
				new Vector3(0, 0, 0) as any,
				new Vector3(10, 0, 0) as any,
				new Color(0x00d4ff) as any
			);
			effects.createImplosion(new Vector3(-5, -5, -5) as any, new Color(0xff4757) as any);
			effects.createRippleWave(new Vector3(0, 0, 0) as any);

			// Should not throw for 200 frames
			for (let f = 0; f < 200; f++) {
				effects.update(nodeMeshMap, camera, nodePositions);
			}

			// All effects should have cleaned up
			expect(scene.children.length).toBe(0);
		});
	});

	describe('dispose', () => {
		it('cleans up all active effects', () => {
			effects.createSpawnBurst(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any);
			effects.createRainbowBurst(new Vector3(0, 0, 0) as any, new Color(0xff00ff) as any);
			effects.createImplosion(new Vector3(0, 0, 0) as any, new Color(0xff4757) as any);
			effects.createShockwave(new Vector3(0, 0, 0) as any, new Color(0x00ffd1) as any, camera);
			effects.createConnectionFlash(
				new Vector3(0, 0, 0) as any,
				new Vector3(10, 0, 0) as any,
				new Color(0x00d4ff) as any
			);

			effects.dispose();

			expect(effects.pulseEffects.length).toBe(0);
		});
	});

	describe('createBirthOrb (v2.3 Memory Birth Ritual)', () => {
		// Build a camera with a Quaternion for createBirthOrb's view-space
		// projection. The three-mock's applyQuaternion is identity, so the
		// start position collapses to `camera.position + (0, 0, -distance)`.
		function makeCamera() {
			return {
				position: new Vector3(0, 30, 80),
				quaternion: new (class {
					x = 0; y = 0; z = 0; w = 1;
				})(),
			} as any;
		}

		it('adds exactly 2 sprites to the scene on spawn', () => {
			const cam = makeCamera();
			const baseline = scene.children.length;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			expect(scene.children.length).toBe(baseline + 2);
		});

		it('both sprite and core use additive blending', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0xff8800) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;
			const core = scene.children[1] as any;
			// AdditiveBlending constant from three-mock is 2
			expect(halo.material.blending).toBe(2);
			expect(core.material.blending).toBe(2);
			// depthTest:false is passed to the SpriteMaterial constructor in
			// effects.ts so the orb stays visible through other nodes. The
			// three-mock's SpriteMaterial constructor does not persist this
			// param, so we can't assert it at the instance level here; the
			// production behavior is covered by ui-fixes.test.ts source grep.
			expect(halo.material.transparent).toBe(true);
			expect(core.material.transparent).toBe(true);
		});

		it('positions the orb at camera-relative cosmic center on spawn', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {},
				{ distanceFromCamera: 40 }
			);
			const halo = scene.children[0] as any;
			const core = scene.children[1] as any;
			// mock applyQuaternion is identity, so startPos = camera.pos + (0,0,-40)
			expect(halo.position.x).toBeCloseTo(0);
			expect(halo.position.y).toBeCloseTo(30);
			expect(halo.position.z).toBeCloseTo(40); // 80 + (-40)
			expect(core.position.x).toBeCloseTo(halo.position.x);
			expect(core.position.y).toBeCloseTo(halo.position.y);
			expect(core.position.z).toBeCloseTo(halo.position.z);
		});

		it('gestation phase: position stays at startPos for all 48 frames', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(100, 100, 100) as any, // far-away target
				() => {}
			);
			const halo = scene.children[0] as any;
			const startX = halo.position.x;
			const startY = halo.position.y;
			const startZ = halo.position.z;

			for (let f = 0; f < 48; f++) {
				effects.update(nodeMeshMap, cam);
				expect(halo.position.x).toBeCloseTo(startX);
				expect(halo.position.y).toBeCloseTo(startY);
				expect(halo.position.z).toBeCloseTo(startZ);
			}
		});

		it('gestation phase: opacity rises from 0 toward 0.95', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;
			const core = scene.children[1] as any;

			// Spawn opacity
			expect(halo.material.opacity).toBe(0);
			expect(core.material.opacity).toBe(0);

			effects.update(nodeMeshMap, cam); // age 1
			const earlyHaloOp = halo.material.opacity;
			expect(earlyHaloOp).toBeGreaterThan(0);
			expect(earlyHaloOp).toBeLessThan(0.2);

			// Run to end of gestation
			for (let f = 0; f < 47; f++) effects.update(nodeMeshMap, cam);
			expect(halo.material.opacity).toBeCloseTo(0.95, 1);
			expect(core.material.opacity).toBeCloseTo(1.0, 1);
			// Monotonic-ish growth: late gestation > early gestation
			expect(halo.material.opacity).toBeGreaterThan(earlyHaloOp);
		});

		it('gestation phase: sprite scale grows substantially', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			effects.update(nodeMeshMap, cam); // age 1
			const earlyScale = halo.scale.x;

			for (let f = 0; f < 47; f++) effects.update(nodeMeshMap, cam); // age 48
			const lateScale = halo.scale.x;

			// Halo grows from ~0.5 toward ~5 during gestation (with pulse variation).
			expect(lateScale).toBeGreaterThan(earlyScale);
			expect(lateScale).toBeGreaterThan(2);
		});

		it('gestation phase: halo color tints toward event color', () => {
			const cam = makeCamera();
			const eventColor = new Color(0xff0000); // pure red
			effects.createBirthOrb(
				cam,
				eventColor as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			effects.update(nodeMeshMap, cam); // age 1 — factor ≈ 0.72
			const earlyR = halo.material.color.r;

			for (let f = 0; f < 47; f++) effects.update(nodeMeshMap, cam); // age 48 — factor = 1.0
			const lateR = halo.material.color.r;

			// Red channel should approach the event color's red (1.0) from a dimmer value
			expect(lateR).toBeGreaterThan(earlyR);
			expect(lateR).toBeCloseTo(1.0, 1);
			// Green/blue stay at 0 (event color is pure red)
			expect(halo.material.color.g).toBeCloseTo(0);
			expect(halo.material.color.b).toBeCloseTo(0);
		});

		it('flight phase: Bezier arc passes ABOVE the linear midpoint at t=0.5', () => {
			const cam = makeCamera();
			// startPos = (0, 30, 40), target = (0, 0, 0)
			// linear midpoint y = 15; control point y = 15 + 30 + dist*0.15 = 52.5
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			// Drive past gestation (48) + half of flight (45) = 93 frames → t=0.5
			for (let f = 0; f < 93; f++) effects.update(nodeMeshMap, cam);

			// Linear midpoint y is 15; Bezier midpoint should be notably higher.
			expect(halo.position.y).toBeGreaterThan(15);
			// And not as high as the control point itself (52.5) — Bezier
			// passes through midpoint-ish at t=0.5, biased upward by the arc.
			expect(halo.position.y).toBeLessThan(52.5);
		});

		it('flight phase: orb moves from startPos toward target', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			// End of gestation
			for (let f = 0; f < 48; f++) effects.update(nodeMeshMap, cam);
			const gestZ = halo.position.z;

			// One tick into flight
			effects.update(nodeMeshMap, cam);
			const earlyFlightZ = halo.position.z;

			// Near end of flight
			for (let f = 0; f < 88; f++) effects.update(nodeMeshMap, cam);
			const lateFlightZ = halo.position.z;

			// Z moves from 40 toward 0
			expect(earlyFlightZ).toBeLessThan(gestZ);
			expect(lateFlightZ).toBeLessThan(earlyFlightZ);
			expect(lateFlightZ).toBeLessThan(5); // close to target z=0
		});

		it('dynamic target tracking: changing getTargetPos mid-flight redirects the orb', () => {
			const cam = makeCamera();
			let target = new Vector3(0, 0, 0);
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => target as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			// Drive to mid-flight (gestation 48 + 30 flight frames = 78)
			for (let f = 0; f < 78; f++) effects.update(nodeMeshMap, cam);
			const xBeforeRedirect = halo.position.x;

			// Redirect target far to the +X side
			target = new Vector3(200, 0, 0);

			// A few more flight frames — orb should track the new target
			for (let f = 0; f < 10; f++) effects.update(nodeMeshMap, cam);
			const xAfterRedirect = halo.position.x;

			// With the original target at (0,0,0), x stays near 0 throughout.
			// After redirect, x should swing toward the new target's +200.
			expect(xAfterRedirect).toBeGreaterThan(xBeforeRedirect + 5);
		});

		it('onArrive fires exactly once at frame 139 (totalFrames + 1)', () => {
			const cam = makeCamera();
			let arriveCount = 0;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {
					arriveCount++;
				}
			);

			// Drive through gestation (48) + flight (90) = 138 frames. Should NOT have fired.
			for (let f = 0; f < 138; f++) effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(0);

			// Frame 139 — fires onArrive
			effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(1);

			// Drive many more frames — must stay at 1
			for (let f = 0; f < 50; f++) effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(1);
		});

		it('post-arrival fade: orb disposes from scene after ~8 fade frames', () => {
			const cam = makeCamera();
			const baseline = scene.children.length;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			expect(scene.children.length).toBe(baseline + 2);

			// Gestation + flight + arrive + fade = 138 + 1 + 8 = 147 frames
			for (let f = 0; f < 150; f++) effects.update(nodeMeshMap, cam);

			// Both orb sprites should be gone
			expect(scene.children.length).toBe(baseline);
		});

		it('onArrive callback wrapped in try/catch so a throw does not crash the loop', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {
					throw new Error('caller blew up');
				}
			);

			// Should not throw — the production code swallows arrival-callback errors.
			expect(() => {
				for (let f = 0; f < 160; f++) effects.update(nodeMeshMap, cam);
			}).not.toThrow();
		});

		it('Sanhedrin Shatter: onArrive NEVER fires when target vanishes mid-flight', () => {
			const cam = makeCamera();
			let arriveCount = 0;
			let target: Vector3 | undefined = new Vector3(0, 0, 0);

			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => target as any,
				() => {
					arriveCount++;
				}
			);

			// Finish gestation (48 frames) with target present
			for (let f = 0; f < 48; f++) effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(0);

			// Stop hook yanks the target mid-flight
			target = undefined;

			// Run enough frames to cover the entire orb lifecycle
			for (let f = 0; f < 200; f++) effects.update(nodeMeshMap, cam);

			// onArrive must NEVER fire on aborted orbs
			expect(arriveCount).toBe(0);
		});

		it('Sanhedrin Shatter: implosion is spawned when target vanishes mid-flight', () => {
			const cam = makeCamera();
			let target: Vector3 | undefined = new Vector3(0, 0, 0);

			const baseline = scene.children.length;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => target as any,
				() => {}
			);
			// baseline + 2 sprites
			expect(scene.children.length).toBe(baseline + 2);

			// Finish gestation
			for (let f = 0; f < 48; f++) effects.update(nodeMeshMap, cam);

			// Yank target → abort triggers on next tick
			target = undefined;
			const beforeAbort = scene.children.length;
			effects.update(nodeMeshMap, cam);
			// Scene should have grown by at least 1 (the implosion particles)
			expect(scene.children.length).toBeGreaterThan(beforeAbort);
		});

		it('Sanhedrin Shatter: halo turns blood-red on abort', () => {
			const cam = makeCamera();
			let target: Vector3 | undefined = new Vector3(0, 0, 0);

			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any, // cyan — NOT red
				() => target as any,
				() => {}
			);
			const halo = scene.children[0] as any;

			// Finish gestation
			for (let f = 0; f < 48; f++) effects.update(nodeMeshMap, cam);

			// Sanity: halo is NOT red yet (event color cyan has r≈0)
			expect(halo.material.color.r).toBeLessThan(0.5);

			// Yank target; abort triggers next tick
			target = undefined;
			effects.update(nodeMeshMap, cam);

			// Halo should now be blood red (1.0, 0.15, 0.2)
			expect(halo.material.color.r).toBeGreaterThan(0.9);
			expect(halo.material.color.g).toBeLessThan(0.3);
			expect(halo.material.color.b).toBeLessThan(0.3);
		});

		it('Sanhedrin Shatter: orb eventually disposes from scene', () => {
			const cam = makeCamera();
			let target: Vector3 | undefined = new Vector3(0, 0, 0);

			const baseline = scene.children.length;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => target as any,
				() => {}
			);

			// Finish gestation
			for (let f = 0; f < 48; f++) effects.update(nodeMeshMap, cam);
			// Yank target
			target = undefined;

			// Drive a long time — orb + implosion should both dispose
			// (orb fade ~8 frames, implosion lifetime ~80 frames)
			for (let f = 0; f < 200; f++) effects.update(nodeMeshMap, cam);

			expect(scene.children.length).toBe(baseline);
		});

		it('dispose() removes active birth orbs from the scene', () => {
			const cam = makeCamera();
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => {}
			);
			effects.createBirthOrb(
				cam,
				new Color(0xff00ff) as any,
				() => new Vector3(10, 10, 10) as any,
				() => {}
			);
			// 4 sprites in scene (2 per orb)
			expect(scene.children.length).toBeGreaterThanOrEqual(4);

			effects.dispose();

			// All orb sprites should be gone
			expect(scene.children.length).toBe(0);
		});

		it('multiple orbs in flight: all 3 onArrive callbacks fire exactly once each', () => {
			const cam = makeCamera();
			let c1 = 0, c2 = 0, c3 = 0;

			effects.createBirthOrb(
				cam,
				new Color(0xff0000) as any,
				() => new Vector3(10, 0, 0) as any,
				() => { c1++; }
			);
			effects.createBirthOrb(
				cam,
				new Color(0x00ff00) as any,
				() => new Vector3(-10, 0, 0) as any,
				() => { c2++; }
			);
			effects.createBirthOrb(
				cam,
				new Color(0x0000ff) as any,
				() => new Vector3(0, 0, -10) as any,
				() => { c3++; }
			);

			// Drive past arrival (139) with margin
			for (let f = 0; f < 160; f++) effects.update(nodeMeshMap, cam);

			expect(c1).toBe(1);
			expect(c2).toBe(1);
			expect(c3).toBe(1);
		});

		it('custom gestation/flight frame counts are honored', () => {
			const cam = makeCamera();
			let arriveCount = 0;
			effects.createBirthOrb(
				cam,
				new Color(0x00ffd1) as any,
				() => new Vector3(0, 0, 0) as any,
				() => { arriveCount++; },
				{ gestationFrames: 10, flightFrames: 20 }
			);

			// Before frame 31 — no arrival
			for (let f = 0; f < 30; f++) effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(0);

			// Frame 31 — fires
			effects.update(nodeMeshMap, cam);
			expect(arriveCount).toBe(1);
		});

		it('zero-alloc invariant (advisory): flight phase runs without throwing across many orbs', () => {
			// Advisory test — vitest has no allocator introspection, but the
			// inline algebraic Bezier eval in effects.ts is intentionally zero-
			// allocation per frame (no `new Vector3`, no `new QuadraticBezierCurve3`).
			// Here we just smoke-test that running many orbs across the full
			// flight phase does not throw and completes cleanly.
			const cam = makeCamera();
			for (let k = 0; k < 6; k++) {
				effects.createBirthOrb(
					cam,
					new Color(0x00ffd1) as any,
					() => new Vector3(k * 5, 0, 0) as any,
					() => {}
				);
			}
			expect(() => {
				for (let f = 0; f < 150; f++) effects.update(nodeMeshMap, cam);
			}).not.toThrow();
			// All orbs should have cleaned up
			expect(scene.children.length).toBe(0);
		});
	});
});

import * as THREE from 'three';
import { getGlowTexture } from './nodes';

export interface PulseEffect {
	nodeId: string;
	intensity: number;
	color: THREE.Color;
	decay: number;
}

interface SpawnBurst {
	position: THREE.Vector3;
	age: number;
	particles: THREE.Points;
}

interface RainbowBurst {
	position: THREE.Vector3;
	age: number;
	maxAge: number;
	particles: THREE.Points;
	baseColor: THREE.Color;
}

interface RippleWave {
	origin: THREE.Vector3;
	radius: number;
	speed: number;
	age: number;
	maxAge: number;
	pulsedNodes: Set<string>;
}

interface ImplosionEffect {
	position: THREE.Vector3;
	age: number;
	maxAge: number;
	particles: THREE.Points;
	flash: THREE.Mesh | null;
}

interface Shockwave {
	mesh: THREE.Mesh;
	age: number;
	maxAge: number;
}

interface ConnectionFlash {
	line: THREE.Line;
	intensity: number;
}

// v2.3 Memory Birth Ritual. The orb gestates at a camera-relative "cosmic
// center" point for `gestationFrames`, then flies along a dynamic quadratic
// Bezier curve to the live position of its target node for `flightFrames`,
// then calls `onArrive` and disposes itself. The target position is
// resolved via `getTargetPos` on every frame so the force simulation can
// move the node during the flight and the orb stays glued to it.
interface BirthOrb {
	sprite: THREE.Sprite;
	core: THREE.Sprite;
	startPos: THREE.Vector3;
	getTargetPos: () => THREE.Vector3 | undefined;
	color: THREE.Color;
	age: number;
	gestationFrames: number;
	flightFrames: number;
	arriveFired: boolean;
	onArrive: () => void;
	/** Last known target position. If the target disappears mid-flight we keep
	 *  using this snapshot so the orb still lands somewhere sensible. */
	lastTargetPos: THREE.Vector3;
	/** v2.3: Sanhedrin-Shatter state. Set true when getTargetPos returns
	 *  undefined after gestation — the Stop hook deleted the target node
	 *  mid-ritual, so we short-circuit the arrival cascade and implode
	 *  the orb in place as the "cognitive immune system" visual. */
	aborted: boolean;
}

export class EffectManager {
	pulseEffects: PulseEffect[] = [];
	private spawnBursts: SpawnBurst[] = [];
	private rainbowBursts: RainbowBurst[] = [];
	private rippleWaves: RippleWave[] = [];
	private implosions: ImplosionEffect[] = [];
	private shockwaves: Shockwave[] = [];
	private connectionFlashes: ConnectionFlash[] = [];
	private birthOrbs: BirthOrb[] = [];
	private scene: THREE.Scene;

	constructor(scene: THREE.Scene) {
		this.scene = scene;
	}

	addPulse(nodeId: string, intensity: number, color: THREE.Color, decay: number) {
		this.pulseEffects.push({ nodeId, intensity, color, decay });
	}

	createSpawnBurst(position: THREE.Vector3, color: THREE.Color) {
		const count = 60;
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(count * 3);
		const velocities = new Float32Array(count * 3);

		for (let i = 0; i < count; i++) {
			positions[i * 3] = position.x;
			positions[i * 3 + 1] = position.y;
			positions[i * 3 + 2] = position.z;
			const theta = Math.random() * Math.PI * 2;
			const phi = Math.acos(2 * Math.random() - 1);
			const speed = 0.3 + Math.random() * 0.5;
			velocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed;
			velocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed;
			velocities[i * 3 + 2] = Math.cos(phi) * speed;
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

		const mat = new THREE.PointsMaterial({
			color,
			size: 0.6,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
			sizeAttenuation: true,
		});

		const pts = new THREE.Points(geo, mat);
		this.scene.add(pts);
		this.spawnBursts.push({ position: position.clone(), age: 0, particles: pts });
	}

	createShockwave(position: THREE.Vector3, color: THREE.Color, camera: THREE.Camera) {
		const geo = new THREE.RingGeometry(0.1, 0.5, 64);
		const mat = new THREE.MeshBasicMaterial({
			color,
			transparent: true,
			opacity: 0.8,
			side: THREE.DoubleSide,
			blending: THREE.AdditiveBlending,
		});
		const ring = new THREE.Mesh(geo, mat);
		ring.position.copy(position);
		ring.lookAt(camera.position);
		this.scene.add(ring);
		this.shockwaves.push({ mesh: ring, age: 0, maxAge: 60 });
	}

	createRainbowBurst(position: THREE.Vector3, baseColor: THREE.Color) {
		const count = 120;
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(count * 3);
		const velocities = new Float32Array(count * 3);
		const hueOffsets = new Float32Array(count);

		for (let i = 0; i < count; i++) {
			positions[i * 3] = position.x;
			positions[i * 3 + 1] = position.y;
			positions[i * 3 + 2] = position.z;
			const theta = Math.random() * Math.PI * 2;
			const phi = Math.acos(2 * Math.random() - 1);
			const speed = 0.2 + Math.random() * 0.6;
			velocities[i * 3] = Math.sin(phi) * Math.cos(theta) * speed;
			velocities[i * 3 + 1] = Math.sin(phi) * Math.sin(theta) * speed;
			velocities[i * 3 + 2] = Math.cos(phi) * speed;
			hueOffsets[i] = Math.random();
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
		geo.setAttribute('hueOffset', new THREE.BufferAttribute(hueOffsets, 1));

		const mat = new THREE.PointsMaterial({
			color: baseColor,
			size: 0.8,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
			sizeAttenuation: true,
		});

		const pts = new THREE.Points(geo, mat);
		this.scene.add(pts);
		this.rainbowBursts.push({
			position: position.clone(),
			age: 0,
			maxAge: 180, // 3 seconds at 60fps
			particles: pts,
			baseColor: baseColor.clone(),
		});
	}

	createRippleWave(origin: THREE.Vector3) {
		this.rippleWaves.push({
			origin: origin.clone(),
			radius: 0,
			speed: 1.2,
			age: 0,
			maxAge: 90,
			pulsedNodes: new Set(),
		});
	}

	createImplosion(position: THREE.Vector3, color: THREE.Color) {
		const count = 40;
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(count * 3);
		const velocities = new Float32Array(count * 3);

		// Particles start at random positions in a sphere around the target
		const startRadius = 8;
		for (let i = 0; i < count; i++) {
			const theta = Math.random() * Math.PI * 2;
			const phi = Math.acos(2 * Math.random() - 1);
			const r = startRadius * (0.5 + Math.random() * 0.5);
			positions[i * 3] = position.x + Math.sin(phi) * Math.cos(theta) * r;
			positions[i * 3 + 1] = position.y + Math.sin(phi) * Math.sin(theta) * r;
			positions[i * 3 + 2] = position.z + Math.cos(phi) * r;
			// Velocity points INWARD toward the center
			velocities[i * 3] = (position.x - positions[i * 3]) * 0.04;
			velocities[i * 3 + 1] = (position.y - positions[i * 3 + 1]) * 0.04;
			velocities[i * 3 + 2] = (position.z - positions[i * 3 + 2]) * 0.04;
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));

		const mat = new THREE.PointsMaterial({
			color,
			size: 0.5,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
			sizeAttenuation: true,
		});

		const pts = new THREE.Points(geo, mat);
		this.scene.add(pts);
		this.implosions.push({
			position: position.clone(),
			age: 0,
			maxAge: 60,
			particles: pts,
			flash: null,
		});
	}

	createConnectionFlash(from: THREE.Vector3, to: THREE.Vector3, color: THREE.Color) {
		const points = [from.clone(), to.clone()];
		const geo = new THREE.BufferGeometry().setFromPoints(points);
		const mat = new THREE.LineBasicMaterial({
			color,
			transparent: true,
			opacity: 1.0,
			blending: THREE.AdditiveBlending,
		});
		const line = new THREE.Line(geo, mat);
		this.scene.add(line);
		this.connectionFlashes.push({ line, intensity: 1.0 });
	}

	/**
	 * v2.3 Memory Birth Ritual. Spawn a glowing orb at a point in front of the
	 * camera ("cosmic center"), gestate for ~800ms, then arc along a quadratic
	 * Bezier curve to the live position of the target node, which is resolved
	 * on every frame via `getTargetPos`. On arrival, `onArrive` fires — caller
	 * is responsible for adding the real node to the graph and triggering
	 * arrival-time bursts.
	 *
	 * The target getter can return undefined if the node has been removed
	 * mid-flight; the orb then flies to the last known target position so the
	 * burst still fires somewhere coherent rather than snapping to origin.
	 */
	createBirthOrb(
		camera: THREE.Camera,
		color: THREE.Color,
		getTargetPos: () => THREE.Vector3 | undefined,
		onArrive: () => void,
		opts: { gestationFrames?: number; flightFrames?: number; distanceFromCamera?: number } = {}
	) {
		const gestationFrames = opts.gestationFrames ?? 48; // ~800ms
		const flightFrames = opts.flightFrames ?? 90; // ~1500ms
		const distanceFromCamera = opts.distanceFromCamera ?? 40;

		// Place the orb slightly in front of the camera, in view-space,
		// projected back into world coordinates. This way the orb always
		// appears "right in front of the user's face" regardless of where
		// the camera has been orbited to.
		const startPos = new THREE.Vector3(0, 0, -distanceFromCamera)
			.applyQuaternion(camera.quaternion)
			.add(camera.position);

		// Outer glow halo
		const haloMat = new THREE.SpriteMaterial({
			map: getGlowTexture(),
			color: color.clone(),
			transparent: true,
			opacity: 0.0,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			depthTest: false, // always visible, even through other nodes
		});
		const sprite = new THREE.Sprite(haloMat);
		sprite.position.copy(startPos);
		sprite.scale.set(0.5, 0.5, 1);
		sprite.renderOrder = 999;

		// Inner bright core — stays hot white during gestation, tints at launch
		const coreMat = new THREE.SpriteMaterial({
			map: getGlowTexture(),
			color: new THREE.Color(0xffffff),
			transparent: true,
			opacity: 0.0,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
			depthTest: false,
		});
		const core = new THREE.Sprite(coreMat);
		core.position.copy(startPos);
		core.scale.set(0.2, 0.2, 1);
		core.renderOrder = 1000;

		this.scene.add(sprite);
		this.scene.add(core);

		// Snapshot the current target so we have a fallback.
		const initialTarget = getTargetPos()?.clone() ?? startPos.clone();

		this.birthOrbs.push({
			sprite,
			core,
			startPos,
			getTargetPos,
			color: color.clone(),
			age: 0,
			gestationFrames,
			flightFrames,
			arriveFired: false,
			onArrive,
			lastTargetPos: initialTarget,
			aborted: false,
		});
	}

	update(
		nodeMeshMap: Map<string, THREE.Mesh>,
		camera: THREE.Camera,
		nodePositions?: Map<string, THREE.Vector3>
	) {
		// Pulse effects
		for (let i = this.pulseEffects.length - 1; i >= 0; i--) {
			const pulse = this.pulseEffects[i];
			pulse.intensity -= pulse.decay;
			if (pulse.intensity <= 0) {
				this.pulseEffects.splice(i, 1);
				continue;
			}
			const mesh = nodeMeshMap.get(pulse.nodeId);
			if (mesh) {
				const mat = mesh.material as THREE.MeshStandardMaterial;
				mat.emissive.lerp(pulse.color, pulse.intensity * 0.3);
				mat.emissiveIntensity = Math.max(mat.emissiveIntensity, pulse.intensity);
			}
		}

		// Spawn bursts (original)
		for (let i = this.spawnBursts.length - 1; i >= 0; i--) {
			const burst = this.spawnBursts[i];
			burst.age++;
			if (burst.age > 120) {
				this.scene.remove(burst.particles);
				burst.particles.geometry.dispose();
				(burst.particles.material as THREE.Material).dispose();
				this.spawnBursts.splice(i, 1);
				continue;
			}
			const positions = burst.particles.geometry.attributes.position as THREE.BufferAttribute;
			const vels = burst.particles.geometry.attributes.velocity as THREE.BufferAttribute;
			for (let j = 0; j < positions.count; j++) {
				positions.setX(j, positions.getX(j) + vels.getX(j));
				positions.setY(j, positions.getY(j) + vels.getY(j));
				positions.setZ(j, positions.getZ(j) + vels.getZ(j));
				vels.setX(j, vels.getX(j) * 0.97);
				vels.setY(j, vels.getY(j) * 0.97);
				vels.setZ(j, vels.getZ(j) * 0.97);
			}
			positions.needsUpdate = true;
			const mat = burst.particles.material as THREE.PointsMaterial;
			mat.opacity = Math.max(0, 1 - burst.age / 120);
			mat.size = 0.6 * (1 - burst.age / 200);
		}

		// Rainbow bursts — HSL cycling, pulsing size, 3-second lifespan
		for (let i = this.rainbowBursts.length - 1; i >= 0; i--) {
			const rb = this.rainbowBursts[i];
			rb.age++;
			if (rb.age > rb.maxAge) {
				this.scene.remove(rb.particles);
				rb.particles.geometry.dispose();
				(rb.particles.material as THREE.Material).dispose();
				this.rainbowBursts.splice(i, 1);
				continue;
			}
			const positions = rb.particles.geometry.attributes.position as THREE.BufferAttribute;
			const vels = rb.particles.geometry.attributes.velocity as THREE.BufferAttribute;
			for (let j = 0; j < positions.count; j++) {
				positions.setX(j, positions.getX(j) + vels.getX(j));
				positions.setY(j, positions.getY(j) + vels.getY(j));
				positions.setZ(j, positions.getZ(j) + vels.getZ(j));
				vels.setX(j, vels.getX(j) * 0.98);
				vels.setY(j, vels.getY(j) * 0.98);
				vels.setZ(j, vels.getZ(j) * 0.98);
			}
			positions.needsUpdate = true;

			const progress = rb.age / rb.maxAge;
			const mat = rb.particles.material as THREE.PointsMaterial;
			// Rainbow HSL cycling blended with base color
			const hue = (rb.age * 0.02) % 1;
			const rainbowColor = new THREE.Color().setHSL(hue, 1.0, 0.6);
			mat.color.copy(rb.baseColor).lerp(rainbowColor, 0.6);
			mat.opacity = Math.max(0, 1 - progress * progress);
			// Pulsing size
			mat.size = 0.8 * (1 - progress * 0.5) * (1 + Math.sin(rb.age * 0.3) * 0.2);
		}

		// Ripple waves — expanding wavefront, pulse nearby nodes on contact
		if (nodePositions) {
			for (let i = this.rippleWaves.length - 1; i >= 0; i--) {
				const rw = this.rippleWaves[i];
				rw.age++;
				rw.radius += rw.speed;

				if (rw.age > rw.maxAge) {
					this.rippleWaves.splice(i, 1);
					continue;
				}

				// Check nodes in range of the expanding wavefront
				const waveFront = rw.radius;
				const waveWidth = 3.0;
				nodePositions.forEach((pos, id) => {
					if (rw.pulsedNodes.has(id)) return;
					const dist = pos.distanceTo(rw.origin);
					if (dist >= waveFront - waveWidth && dist <= waveFront + waveWidth) {
						rw.pulsedNodes.add(id);
						// Mini-pulse on contact
						this.addPulse(id, 0.8, new THREE.Color(0x00ffd1), 0.03);
						// Pulse handles the visual bump — no direct scale mutation
						// (multiplyScalar was cumulative and fought with animation system)
					}
				});
			}
		}

		// Implosion effects — particles rush inward, converge, then flash
		for (let i = this.implosions.length - 1; i >= 0; i--) {
			const imp = this.implosions[i];
			imp.age++;

			if (imp.age > imp.maxAge + 20) {
				this.scene.remove(imp.particles);
				imp.particles.geometry.dispose();
				(imp.particles.material as THREE.Material).dispose();
				if (imp.flash) {
					this.scene.remove(imp.flash);
					imp.flash.geometry.dispose();
					(imp.flash.material as THREE.Material).dispose();
				}
				this.implosions.splice(i, 1);
				continue;
			}

			if (imp.age <= imp.maxAge) {
				const positions = imp.particles.geometry.attributes.position as THREE.BufferAttribute;
				const vels = imp.particles.geometry.attributes.velocity as THREE.BufferAttribute;
				// Accelerate inward
				const accelFactor = 1 + imp.age * 0.02;
				for (let j = 0; j < positions.count; j++) {
					positions.setX(j, positions.getX(j) + vels.getX(j) * accelFactor);
					positions.setY(j, positions.getY(j) + vels.getY(j) * accelFactor);
					positions.setZ(j, positions.getZ(j) + vels.getZ(j) * accelFactor);
				}
				positions.needsUpdate = true;

				const mat = imp.particles.material as THREE.PointsMaterial;
				mat.opacity = Math.min(1.0, imp.age / 15);
				mat.size = 0.5 + (imp.age / imp.maxAge) * 0.3;
			}

			// Flash at convergence point
			if (imp.age === imp.maxAge && !imp.flash) {
				const flashGeo = new THREE.SphereGeometry(2, 16, 16);
				const flashMat = new THREE.MeshBasicMaterial({
					color: 0xffffff,
					transparent: true,
					opacity: 1.0,
					blending: THREE.AdditiveBlending,
				});
				imp.flash = new THREE.Mesh(flashGeo, flashMat);
				imp.flash.position.copy(imp.position);
				this.scene.add(imp.flash);
				// Hide particles
				(imp.particles.material as THREE.PointsMaterial).opacity = 0;
			}

			// Flash fade out
			if (imp.flash && imp.age > imp.maxAge) {
				const flashProgress = (imp.age - imp.maxAge) / 20;
				(imp.flash.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 1 - flashProgress);
				imp.flash.scale.setScalar(1 + flashProgress * 3);
			}
		}

		// Shockwaves
		for (let i = this.shockwaves.length - 1; i >= 0; i--) {
			const sw = this.shockwaves[i];
			sw.age++;
			if (sw.age > sw.maxAge) {
				this.scene.remove(sw.mesh);
				sw.mesh.geometry.dispose();
				(sw.mesh.material as THREE.Material).dispose();
				this.shockwaves.splice(i, 1);
				continue;
			}
			const progress = sw.age / sw.maxAge;
			sw.mesh.scale.setScalar(1 + progress * 20);
			(sw.mesh.material as THREE.MeshBasicMaterial).opacity = 0.8 * (1 - progress);
			sw.mesh.lookAt(camera.position);
		}

		// Connection flashes
		for (let i = this.connectionFlashes.length - 1; i >= 0; i--) {
			const flash = this.connectionFlashes[i];
			flash.intensity -= 0.015;
			if (flash.intensity <= 0) {
				this.scene.remove(flash.line);
				flash.line.geometry.dispose();
				(flash.line.material as THREE.Material).dispose();
				this.connectionFlashes.splice(i, 1);
				continue;
			}
			(flash.line.material as THREE.LineBasicMaterial).opacity = flash.intensity;
		}

		// v2.3 Birth orbs — gestate at cosmic center, then arc to live node
		// position along a quadratic Bezier curve. Target position is
		// re-resolved every frame so the force simulation can move the
		// destination during flight without the orb losing its mark.
		for (let i = this.birthOrbs.length - 1; i >= 0; i--) {
			const orb = this.birthOrbs[i];
			orb.age++;
			const totalFrames = orb.gestationFrames + orb.flightFrames;

			const haloMat = orb.sprite.material as THREE.SpriteMaterial;
			const coreMat = orb.core.material as THREE.SpriteMaterial;

			// Refresh the live target snapshot. If the target getter returns
			// undefined DURING flight (not just at spawn), the node was
			// aborted mid-ritual — typically a Sanhedrin veto deleting a
			// hallucination node while the orb was still in transit. Trigger
			// the anti-birth: turn red, implode in place, stop tracking.
			const live = orb.getTargetPos();
			if (live) {
				orb.lastTargetPos.copy(live);
			} else if (orb.age > orb.gestationFrames && !orb.aborted) {
				orb.aborted = true;
				// Fire an implosion where the orb currently is, then splice
				// out on the next tick by jumping age to the end of life.
				const pos = orb.sprite.position;
				haloMat.color.setRGB(1.0, 0.15, 0.2); // blood red
				coreMat.color.setRGB(1.0, 0.6, 0.6);
				this.createImplosion(pos, new THREE.Color(0xff2533));
				orb.arriveFired = true;
				orb.age = totalFrames + 1;
			}

			if (orb.age <= orb.gestationFrames) {
				// Gestation phase — pulse brighter + grow from a tiny spark
				// into a full orb. Sits still at the cosmic center.
				const t = orb.age / orb.gestationFrames;
				const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
				const pulse = 0.85 + Math.sin(orb.age * 0.35) * 0.15;
				const haloScale = 0.5 + ease * 4.5 * pulse;
				const coreScale = 0.2 + ease * 1.8 * pulse;
				orb.sprite.scale.set(haloScale, haloScale, 1);
				orb.core.scale.set(coreScale, coreScale, 1);
				haloMat.opacity = ease * 0.95;
				coreMat.opacity = ease;
				// Subtle warm-up — core white, halo tints toward the event
				// color as gestation completes.
				haloMat.color.copy(orb.color).multiplyScalar(0.7 + ease * 0.3);
				orb.sprite.position.copy(orb.startPos);
				orb.core.position.copy(orb.startPos);
			} else if (orb.age <= totalFrames) {
				// Flight phase — inline quadratic Bezier eval. Zero-alloc:
				// no new Vector3 or QuadraticBezierCurve3 per frame, which
				// would flood the GC when several orbs are in flight.
				const t = (orb.age - orb.gestationFrames) / orb.flightFrames;
				const ease = t < 0.5
					? 2 * t * t
					: 1 - Math.pow(-2 * t + 2, 2) / 2; // easeInOutQuad

				const s = orb.startPos;
				const tgt = orb.lastTargetPos;
				const dx = tgt.x - s.x;
				const dy = tgt.y - s.y;
				const dz = tgt.z - s.z;
				const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
				const cx = (s.x + tgt.x) * 0.5;
				const cy = (s.y + tgt.y) * 0.5 + 30 + dist * 0.15;
				const cz = (s.z + tgt.z) * 0.5;

				const oneMinusE = 1 - ease;
				const w0 = oneMinusE * oneMinusE;
				const w1 = 2 * oneMinusE * ease;
				const w2 = ease * ease;
				const px = w0 * s.x + w1 * cx + w2 * tgt.x;
				const py = w0 * s.y + w1 * cy + w2 * tgt.y;
				const pz = w0 * s.z + w1 * cz + w2 * tgt.z;

				orb.sprite.position.set(px, py, pz);
				orb.core.position.set(px, py, pz);

				// Trail effect — shrink + brighten as it approaches target
				const shrink = 1 - ease * 0.35;
				orb.sprite.scale.setScalar(5 * shrink);
				orb.core.scale.setScalar(2 * shrink);
				haloMat.opacity = 0.95;
				coreMat.opacity = 1.0;
				// Shift halo fully to the event color during flight
				haloMat.color.copy(orb.color);
			} else if (!orb.arriveFired) {
				// Docking — fire the arrival callback once. Let the caller
				// trigger burst/ripple effects at the exact target point.
				orb.arriveFired = true;
				try {
					orb.onArrive();
				} catch (e) {
					// Effects must never take down the render loop.
					// eslint-disable-next-line no-console
					console.warn('[birth-orb] onArrive threw', e);
				}
				// Fade the orb out over a few more frames instead of popping.
			} else {
				// Post-arrival fade (8 frames ≈ 130ms)
				const fadeAge = orb.age - totalFrames;
				const fade = Math.max(0, 1 - fadeAge / 8);
				haloMat.opacity = 0.95 * fade;
				coreMat.opacity = 1.0 * fade;
				orb.sprite.scale.setScalar(5 * (1 + (1 - fade) * 2));
				if (fade <= 0) {
					this.scene.remove(orb.sprite);
					this.scene.remove(orb.core);
					haloMat.dispose();
					coreMat.dispose();
					this.birthOrbs.splice(i, 1);
				}
			}
		}
	}

	dispose() {
		for (const burst of this.spawnBursts) {
			this.scene.remove(burst.particles);
			burst.particles.geometry.dispose();
			(burst.particles.material as THREE.Material).dispose();
		}
		for (const rb of this.rainbowBursts) {
			this.scene.remove(rb.particles);
			rb.particles.geometry.dispose();
			(rb.particles.material as THREE.Material).dispose();
		}
		for (const imp of this.implosions) {
			this.scene.remove(imp.particles);
			imp.particles.geometry.dispose();
			(imp.particles.material as THREE.Material).dispose();
			if (imp.flash) {
				this.scene.remove(imp.flash);
				imp.flash.geometry.dispose();
				(imp.flash.material as THREE.Material).dispose();
			}
		}
		for (const sw of this.shockwaves) {
			this.scene.remove(sw.mesh);
			sw.mesh.geometry.dispose();
			(sw.mesh.material as THREE.Material).dispose();
		}
		for (const flash of this.connectionFlashes) {
			this.scene.remove(flash.line);
			flash.line.geometry.dispose();
			(flash.line.material as THREE.Material).dispose();
		}
		for (const orb of this.birthOrbs) {
			this.scene.remove(orb.sprite);
			this.scene.remove(orb.core);
			(orb.sprite.material as THREE.Material).dispose();
			(orb.core.material as THREE.Material).dispose();
		}
		this.pulseEffects = [];
		this.spawnBursts = [];
		this.rainbowBursts = [];
		this.rippleWaves = [];
		this.implosions = [];
		this.shockwaves = [];
		this.connectionFlashes = [];
		this.birthOrbs = [];
	}
}

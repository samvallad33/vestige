import * as THREE from 'three';
import type { GraphNode } from '$types';
import { NODE_TYPE_COLORS } from '$types';

// ============================================================================
// v2.0.8: Memory state coloring (FSRS accessibility bucket)
// ============================================================================
//
// Every knowledge_node has an FSRS accessibility score computed from
// (retention × 0.5 + retrieval × 0.3 + storage × 0.2). That score gates which
// memories surface in search and drives the Active / Dormant / Silent /
// Unavailable lifecycle documented by Bjork & Bjork 1992 dual-strength model.
//
// The backend computes all three channels, but `GraphNode` only carries
// `retention` — which is already the dominant weight (0.5 of 1.0). Using
// retention alone as a proxy is a known approximation; the buckets line up
// with the same thresholds `execute_system_status` uses server-side, so the
// visual labelling matches what `/api/stats` reports in its
// `stateDistribution` block.

export type MemoryState = 'active' | 'dormant' | 'silent' | 'unavailable';

/// Map an FSRS retention score to its accessibility bucket.
///
/// Thresholds match `execute_system_status` at the backend so the 3D graph's
/// colours line up with the numbers reported by `/api/stats`.
export function getMemoryState(retention: number): MemoryState {
	if (retention >= 0.7) return 'active';
	if (retention >= 0.4) return 'dormant';
	if (retention >= 0.1) return 'silent';
	return 'unavailable';
}

/// FSRS state palette. Distinct from NODE_TYPE_COLORS so the two modes can
/// coexist in the UI without overloading a single colour channel.
export const MEMORY_STATE_COLORS: Record<MemoryState, string> = {
	active: '#10b981', // emerald — easily retrievable
	dormant: '#f59e0b', // amber — retrievable with effort
	silent: '#8b5cf6', // violet — difficult, needs cues
	unavailable: '#6b7280', // slate — needs reinforcement
};

export const MEMORY_STATE_DESCRIPTIONS: Record<MemoryState, string> = {
	active: 'Easily retrievable (retention ≥ 70%)',
	dormant: 'Retrievable with effort (40–70%)',
	silent: 'Difficult, needs cues (10–40%)',
	unavailable: 'Needs reinforcement (< 10%)',
};

/// Color mode controls whether node spheres are tinted by node type
/// (fact / concept / event / …) or by FSRS memory state.
/// Type mode is the long-standing default; state mode is the v2.0.8 addition.
export type ColorMode = 'type' | 'state';

/// Pick a hex colour for a node given the active colour mode.
/// Falls back to the grey `unavailable` tone if the node's type is unknown.
export function getNodeColor(node: GraphNode, mode: ColorMode): string {
	if (mode === 'state') {
		return MEMORY_STATE_COLORS[getMemoryState(node.retention)];
	}
	return NODE_TYPE_COLORS[node.type] || '#8B95A5';
}

// Shared radial-gradient texture used for every node's glow Sprite.
// Without a map, THREE.Sprite renders as a flat coloured plane — additive-
// blending + UnrealBloomPass then amplifies its square edges into the
// hard-edged "glowing cubes" artefact reported in issue #31. Using a
// soft radial gradient gives a real round halo and lets bloom do its job.
let sharedGlowTexture: THREE.Texture | null = null;
export function getGlowTexture(): THREE.Texture {
	if (sharedGlowTexture) return sharedGlowTexture;
	const size = 128;
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext('2d');
	if (!ctx) {
		// Fallback: empty 1x1 texture; halos will be invisible but nothing crashes.
		sharedGlowTexture = new THREE.Texture();
		return sharedGlowTexture;
	}
	const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
	gradient.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
	gradient.addColorStop(0.25, 'rgba(255, 255, 255, 0.7)');
	gradient.addColorStop(0.55, 'rgba(255, 255, 255, 0.2)');
	gradient.addColorStop(1.0, 'rgba(255, 255, 255, 0.0)');
	ctx.fillStyle = gradient;
	ctx.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	sharedGlowTexture = tex;
	return tex;
}

function easeOutElastic(t: number): number {
	if (t === 0 || t === 1) return t;
	const p = 0.3;
	return Math.pow(2, -10 * t) * Math.sin(((t - p / 4) * (2 * Math.PI)) / p) + 1;
}

function easeInBack(t: number): number {
	const s = 1.70158;
	return t * t * ((s + 1) * t - s);
}

interface MaterializingNode {
	id: string;
	frame: number;
	totalFrames: number;
	mesh: THREE.Mesh;
	glow: THREE.Sprite;
	label: THREE.Sprite;
	targetScale: number;
}

interface DissolvingNode {
	id: string;
	frame: number;
	totalFrames: number;
	mesh: THREE.Mesh;
	glow: THREE.Sprite;
	label: THREE.Sprite;
	originalScale: number;
}

interface GrowingNode {
	id: string;
	frame: number;
	totalFrames: number;
	startScale: number;
	targetScale: number;
}

export class NodeManager {
	group: THREE.Group;
	meshMap = new Map<string, THREE.Mesh>();
	glowMap = new Map<string, THREE.Sprite>();
	positions = new Map<string, THREE.Vector3>();
	labelSprites = new Map<string, THREE.Sprite>();
	hoveredNode: string | null = null;
	selectedNode: string | null = null;
	/// v2.0.8: colour nodes by FSRS memory state (active/dormant/silent/unavailable)
	/// instead of node type. Switched at runtime via `setColorMode`.
	colorMode: ColorMode = 'type';

	private materializingNodes: MaterializingNode[] = [];
	private dissolvingNodes: DissolvingNode[] = [];
	private growingNodes: GrowingNode[] = [];

	constructor() {
		this.group = new THREE.Group();
	}

	/// Switch the active colour mode and re-tint every live node in place.
	/// Safe to call mid-animation — the mesh + glow materials are mutable.
	/// Suppressed nodes keep their 20% opacity / zero-emissive treatment
	/// since that is a separate visual channel (v2.0.5 SIF).
	setColorMode(mode: ColorMode) {
		if (this.colorMode === mode) return;
		this.colorMode = mode;
		for (const [id, mesh] of this.meshMap) {
			const retention = (mesh.userData.retention as number | undefined) ?? 0;
			const type = (mesh.userData.type as string | undefined) ?? 'fact';
			const stubNode = {
				id,
				label: '',
				type,
				retention,
				tags: [],
				createdAt: '',
				updatedAt: '',
				isCenter: false,
			} as GraphNode;
			const hex = getNodeColor(stubNode, mode);
			const newColor = new THREE.Color(hex);
			const mat = mesh.material as THREE.MeshStandardMaterial;
			mat.color.copy(newColor);
			mat.emissive.copy(newColor);
			const glow = this.glowMap.get(id);
			if (glow) {
				(glow.material as THREE.SpriteMaterial).color.copy(newColor);
			}
		}
	}

	createNodes(nodes: GraphNode[]): Map<string, THREE.Vector3> {
		const phi = (1 + Math.sqrt(5)) / 2;
		const count = nodes.length;

		for (let i = 0; i < count; i++) {
			const node = nodes[i];

			// Fibonacci sphere distribution for initial positions
			const y = 1 - (2 * i) / (count - 1 || 1);
			const radius = Math.sqrt(1 - y * y);
			const theta = (2 * Math.PI * i) / phi;
			const spread = 30 + count * 0.5;

			const pos = new THREE.Vector3(
				radius * Math.cos(theta) * spread,
				y * spread,
				radius * Math.sin(theta) * spread
			);

			if (node.isCenter) pos.set(0, 0, 0);

			this.positions.set(node.id, pos);
			this.createNodeMeshes(node, pos, 1.0);
		}

		return this.positions;
	}

	private createNodeMeshes(node: GraphNode, pos: THREE.Vector3, initialScale: number) {
		const size = 0.5 + node.retention * 2;
		// v2.0.8: respect the active colour mode. Newly-added nodes during the
		// same session follow the mode toggled at the UI layer.
		const color = getNodeColor(node, this.colorMode);

		// v2.0.5 Active Forgetting: suppressed memories dim to 20% opacity
		// and lose their emissive glow, mimicking inhibitory-control silencing.
		const isSuppressed = (node.suppression_count ?? 0) > 0;

		// Node mesh
		const geometry = new THREE.SphereGeometry(size, 16, 16);
		const material = new THREE.MeshStandardMaterial({
			color: new THREE.Color(color),
			emissive: new THREE.Color(color),
			emissiveIntensity: isSuppressed ? 0.0 : 0.3 + node.retention * 0.5,
			roughness: 0.3,
			metalness: 0.1,
			transparent: true,
			opacity: isSuppressed ? 0.2 : 0.3 + node.retention * 0.7,
		});

		const mesh = new THREE.Mesh(geometry, material);
		mesh.position.copy(pos);
		mesh.scale.setScalar(initialScale);
		mesh.userData = { nodeId: node.id, type: node.type, retention: node.retention };
		this.meshMap.set(node.id, mesh);
		this.group.add(mesh);

		// Glow sprite — radial-gradient texture kills the square-halo artefact
		// from issue #31. depthWrite:false prevents z-fighting with the sphere.
		const spriteMat = new THREE.SpriteMaterial({
			map: getGlowTexture(),
			color: new THREE.Color(color),
			transparent: true,
			opacity: initialScale > 0 ? (isSuppressed ? 0.1 : 0.3 + node.retention * 0.35) : 0,
			blending: THREE.AdditiveBlending,
			depthWrite: false,
		});
		const sprite = new THREE.Sprite(spriteMat);
		// Slightly larger halo — the gradient falls off quickly so we need
		// more screen real estate for a visible soft bloom footprint.
		sprite.scale.set(size * 6 * initialScale, size * 6 * initialScale, 1);
		sprite.position.copy(pos);
		sprite.userData = { isGlow: true, nodeId: node.id };
		this.glowMap.set(node.id, sprite);
		this.group.add(sprite);

		// Text label sprite
		const labelText = node.label || node.type;
		const labelSprite = this.createTextSprite(labelText, '#94a3b8');
		labelSprite.position.copy(pos);
		labelSprite.position.y += size * 2 + 1.5;
		labelSprite.userData = { isLabel: true, nodeId: node.id, offset: size * 2 + 1.5 };
		this.group.add(labelSprite);
		this.labelSprites.set(node.id, labelSprite);

		return { mesh, glow: sprite, label: labelSprite, size };
	}

	addNode(
		node: GraphNode,
		initialPosition?: THREE.Vector3,
		options: { isBirthRitual?: boolean } = {}
	): THREE.Vector3 {
		const pos =
			initialPosition?.clone() ??
			new THREE.Vector3(
				(Math.random() - 0.5) * 40,
				(Math.random() - 0.5) * 40,
				(Math.random() - 0.5) * 40
			);

		this.positions.set(node.id, pos);

		// Create meshes at scale 0
		const { mesh, glow, label } = this.createNodeMeshes(node, pos, 0);
		mesh.scale.setScalar(0.001); // Avoid zero-scale issues
		glow.scale.set(0.001, 0.001, 1);
		(glow.material as THREE.SpriteMaterial).opacity = 0;
		(label.material as THREE.SpriteMaterial).opacity = 0;

		if (options.isBirthRitual) {
			// v2.3 Birth Ritual: reserve the physics slot but don't show
			// anything until the orb docks. Hiding via .visible keeps the
			// force simulation + positions map fully active, so getTargetPos()
			// can still resolve the live destination for the orb. `igniteNode`
			// below flips visibility and kicks off the materialization anim.
			mesh.visible = false;
			glow.visible = false;
			label.visible = false;
			mesh.userData.birthRitualPending = {
				totalFrames: 30,
				targetScale: 0.5 + node.retention * 2,
			};
		} else {
			this.materializingNodes.push({
				id: node.id,
				frame: 0,
				totalFrames: 30,
				mesh,
				glow,
				label,
				targetScale: 0.5 + node.retention * 2,
			});
		}

		return pos;
	}

	/**
	 * v2.3 Birth Ritual docking. Flip visibility and hand the node over to
	 * the materialization queue so it springs up via easeOutElastic at the
	 * exact moment the orb hits. No-op if the node wasn't created with
	 * `isBirthRitual:true` or was already ignited.
	 */
	igniteNode(id: string) {
		const mesh = this.meshMap.get(id);
		const glow = this.glowMap.get(id);
		const label = this.labelSprites.get(id);
		if (!mesh || !glow || !label) return;
		const pending = mesh.userData.birthRitualPending as
			| { totalFrames: number; targetScale: number }
			| undefined;
		if (!pending) return;
		mesh.visible = true;
		glow.visible = true;
		label.visible = true;
		delete mesh.userData.birthRitualPending;
		this.materializingNodes.push({
			id,
			frame: 0,
			totalFrames: pending.totalFrames,
			mesh,
			glow,
			label,
			targetScale: pending.targetScale,
		});
	}

	removeNode(id: string) {
		const mesh = this.meshMap.get(id);
		const glow = this.glowMap.get(id);
		const label = this.labelSprites.get(id);
		if (!mesh || !glow || !label) return;

		// Cancel any active materialization
		this.materializingNodes = this.materializingNodes.filter((m) => m.id !== id);

		this.dissolvingNodes.push({
			id,
			frame: 0,
			totalFrames: 60,
			mesh,
			glow,
			label,
			originalScale: mesh.scale.x,
		});
	}

	growNode(id: string, newRetention: number) {
		const mesh = this.meshMap.get(id);
		if (!mesh) return;

		const currentScale = mesh.scale.x;
		const targetScale = 0.5 + newRetention * 2;
		mesh.userData.retention = newRetention;

		this.growingNodes.push({
			id,
			frame: 0,
			totalFrames: 30,
			startScale: currentScale,
			targetScale,
		});
	}

	/// Render a label as a dark rounded "pill" with dim slate text.
	///
	/// The scene runs an UnrealBloomPass with threshold 0.2, so any bright
	/// canvas pixels get smeared into a halo. Previously the labels were
	/// near-white (#e2e8f0) text on a transparent background, which bloomed
	/// into unreadable white blobs (issue filed by Sam 2026-04-19). The fix:
	///
	///   1. A ~85%-opaque dark pill under the text so the background is
	///      well below the bloom threshold, stopping the halo before it
	///      spreads past the label bounds.
	///   2. Mid-luminance slate text (#94a3b8 by default) — still legible
	///      but dim enough that bloom only adds a soft glow, not a blast.
	///   3. Smaller font (22px) and tighter sprite scale (9×1.2) so labels
	///      don't visually compete with the node spheres they annotate.
	private createTextSprite(text: string, color: string): THREE.Sprite {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		if (!ctx) {
			const tex = new THREE.Texture();
			return new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0 }));
		}
		canvas.width = 512;
		canvas.height = 64;

		const label = text.length > 40 ? text.slice(0, 37) + '...' : text;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		// Measure the label so the backing pill hugs the text instead of
		// spanning the full canvas width (which would leave a giant empty
		// dark bar on short labels like "fact" or "note").
		ctx.font = '600 22px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
		const metrics = ctx.measureText(label);
		const textWidth = metrics.width;
		const padX = 14;
		const padY = 9;
		const pillW = Math.min(textWidth + padX * 2, canvas.width - 4);
		const pillH = 40;
		const pillX = (canvas.width - pillW) / 2;
		const pillY = (canvas.height - pillH) / 2;
		const radius = pillH / 2;

		// Dark glass pill — low enough luminance that UnrealBloomPass at
		// threshold 0.2 does not amplify its pixels.
		ctx.fillStyle = 'rgba(10, 16, 28, 0.82)';
		ctx.beginPath();
		ctx.moveTo(pillX + radius, pillY);
		ctx.lineTo(pillX + pillW - radius, pillY);
		ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + radius);
		ctx.lineTo(pillX + pillW, pillY + pillH - radius);
		ctx.quadraticCurveTo(
			pillX + pillW,
			pillY + pillH,
			pillX + pillW - radius,
			pillY + pillH
		);
		ctx.lineTo(pillX + radius, pillY + pillH);
		ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - radius);
		ctx.lineTo(pillX, pillY + radius);
		ctx.quadraticCurveTo(pillX, pillY, pillX + radius, pillY);
		ctx.closePath();
		ctx.fill();

		// Hairline stroke for definition at small camera distances.
		ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
		ctx.lineWidth = 1;
		ctx.stroke();

		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillStyle = color;
		ctx.fillText(label, canvas.width / 2, canvas.height / 2 + 1);

		const texture = new THREE.CanvasTexture(canvas);
		texture.needsUpdate = true;

		const mat = new THREE.SpriteMaterial({
			map: texture,
			transparent: true,
			opacity: 0,
			depthTest: false,
			sizeAttenuation: true,
		});

		const sprite = new THREE.Sprite(mat);
		sprite.scale.set(9, 1.2, 1);
		return sprite;
	}

	updatePositions() {
		this.group.children.forEach((child) => {
			if (child.userData.nodeId) {
				const pos = this.positions.get(child.userData.nodeId);
				if (!pos) return;

				if (child.userData.isGlow) {
					child.position.copy(pos);
				} else if (child.userData.isLabel) {
					child.position.copy(pos);
					child.position.y += child.userData.offset;
				} else if (child instanceof THREE.Mesh) {
					child.position.copy(pos);
				}
			}
		});
	}

	animate(
		time: number,
		nodes: GraphNode[],
		camera: THREE.PerspectiveCamera,
		brightness: number = 1.0
	) {
		// Materialization animations — elastic scale-up from 0
		for (let i = this.materializingNodes.length - 1; i >= 0; i--) {
			const mn = this.materializingNodes[i];
			mn.frame++;
			const t = Math.min(mn.frame / mn.totalFrames, 1);
			const scale = easeOutElastic(t);

			// Mesh scales up with elastic spring
			mn.mesh.scale.setScalar(Math.max(0.001, scale));

			// Glow fades in between frames 5-10
			if (mn.frame >= 5) {
				const glowT = Math.min((mn.frame - 5) / 5, 1);
				const glowMat = mn.glow.material as THREE.SpriteMaterial;
				glowMat.opacity = glowT * 0.4;
				const glowSize = mn.targetScale * 6 * scale;
				mn.glow.scale.set(glowSize, glowSize, 1);
			}

			// Label fades in after frame 40 (10 frames after mesh finishes)
			if (mn.frame >= 40) {
				const labelT = Math.min((mn.frame - 40) / 20, 1);
				(mn.label.material as THREE.SpriteMaterial).opacity = labelT * 0.9;
			}

			if (mn.frame >= 60) {
				this.materializingNodes.splice(i, 1);
			}
		}

		// Dissolution animations — easeInBack shrink
		for (let i = this.dissolvingNodes.length - 1; i >= 0; i--) {
			const dn = this.dissolvingNodes[i];
			dn.frame++;
			const t = Math.min(dn.frame / dn.totalFrames, 1);
			const shrink = 1 - easeInBack(t);
			const scale = Math.max(0.001, dn.originalScale * shrink);

			dn.mesh.scale.setScalar(scale);
			const glowScale = scale * 6;
			dn.glow.scale.set(glowScale, glowScale, 1);

			// Fade opacity
			const mat = dn.mesh.material as THREE.MeshStandardMaterial;
			mat.opacity *= 0.97;
			(dn.glow.material as THREE.SpriteMaterial).opacity *= 0.95;
			(dn.label.material as THREE.SpriteMaterial).opacity *= 0.93;

			if (dn.frame >= dn.totalFrames) {
				// Clean up
				this.group.remove(dn.mesh);
				this.group.remove(dn.glow);
				this.group.remove(dn.label);
				dn.mesh.geometry.dispose();
				(dn.mesh.material as THREE.Material).dispose();
				(dn.glow.material as THREE.SpriteMaterial).map?.dispose();
				(dn.glow.material as THREE.Material).dispose();
				(dn.label.material as THREE.SpriteMaterial).map?.dispose();
				(dn.label.material as THREE.Material).dispose();

				this.meshMap.delete(dn.id);
				this.glowMap.delete(dn.id);
				this.labelSprites.delete(dn.id);
				this.positions.delete(dn.id);

				this.dissolvingNodes.splice(i, 1);
			}
		}

		// Growth animations — smooth scale transition for promoted nodes
		for (let i = this.growingNodes.length - 1; i >= 0; i--) {
			const gn = this.growingNodes[i];
			gn.frame++;
			const t = Math.min(gn.frame / gn.totalFrames, 1);
			const scale = gn.startScale + (gn.targetScale - gn.startScale) * easeOutElastic(t);

			const mesh = this.meshMap.get(gn.id);
			if (mesh) mesh.scale.setScalar(scale);

			const glow = this.glowMap.get(gn.id);
			if (glow) {
				const glowSize = scale * 6;
				glow.scale.set(glowSize, glowSize, 1);
			}

			if (gn.frame >= gn.totalFrames) {
				this.growingNodes.splice(i, 1);
			}
		}

		// Node breathing (skip nodes being animated)
		const animatingIds = new Set([
			...this.materializingNodes.map((m) => m.id),
			...this.dissolvingNodes.map((d) => d.id),
			...this.growingNodes.map((g) => g.id),
		]);

		this.meshMap.forEach((mesh, id) => {
			if (animatingIds.has(id)) return;
			const node = nodes.find((n) => n.id === id);
			if (!node) return;
			const breathe =
				1 + Math.sin(time * 1.5 + nodes.indexOf(node) * 0.5) * 0.15 * node.retention;
			mesh.scale.setScalar(breathe);

			// Distance compensation: FogExp2 attenuates exponentially with camera
			// distance, so nodes past ~80 units go nearly black unless we push
			// emissive harder. Boost runs 1.0x at <60 units → ~2.4x at 200 units.
			// Combined with the user brightness multiplier this gives a visible
			// floor at every zoom level without blowing out close-up highlights.
			const pos = this.positions.get(id);
			const dist = pos ? camera.position.distanceTo(pos) : 0;
			const distanceBoost = 1 + Math.min(1.4, Math.max(0, (dist - 60) / 100));

			const mat = mesh.material as THREE.MeshStandardMaterial;
			if (id === this.hoveredNode) {
				mat.emissiveIntensity = 1.0 * brightness;
			} else if (id === this.selectedNode) {
				mat.emissiveIntensity = 0.8 * brightness;
			} else {
				const baseIntensity = 0.3 + node.retention * 0.5;
				const breatheIntensity =
					baseIntensity + Math.sin(time * (0.8 + node.retention * 0.7)) * 0.1 * node.retention;
				mat.emissiveIntensity = breatheIntensity * brightness * distanceBoost;
			}

			// Opacity also gets the distance boost (capped at 1.0) so the node
			// body stays visible against the dark void at far zoom.
			const baseOpacity = 0.3 + node.retention * 0.7;
			mat.opacity = Math.min(1.0, baseOpacity * brightness * distanceBoost);

			// Mirror the boost onto the glow sprite so the halo tracks the core.
			const glow = this.glowMap.get(id);
			if (glow) {
				const glowMat = glow.material as THREE.SpriteMaterial;
				const baseGlow = 0.3 + node.retention * 0.35;
				glowMat.opacity = Math.min(0.95, baseGlow * brightness * distanceBoost);
			}
		});

		// Distance-based label visibility
		this.labelSprites.forEach((sprite, id) => {
			if (animatingIds.has(id)) return;
			const pos = this.positions.get(id);
			if (!pos) return;
			const dist = camera.position.distanceTo(pos);
			const mat = sprite.material as THREE.SpriteMaterial;
			const targetOpacity =
				id === this.hoveredNode || id === this.selectedNode
					? 1.0
					: dist < 40
						? 0.9
						: dist < 80
							? 0.9 * (1 - (dist - 40) / 40)
							: 0;
			mat.opacity += (targetOpacity - mat.opacity) * 0.1;
		});
	}

	getMeshes(): THREE.Mesh[] {
		return Array.from(this.meshMap.values());
	}

	dispose() {
		this.group.traverse((obj) => {
			if (obj instanceof THREE.Mesh) {
				obj.geometry?.dispose();
				(obj.material as THREE.Material)?.dispose();
			} else if (obj instanceof THREE.Sprite) {
				(obj.material as THREE.SpriteMaterial)?.map?.dispose();
				(obj.material as THREE.Material)?.dispose();
			}
		});
		this.materializingNodes = [];
		this.dissolvingNodes = [];
		this.growingNodes = [];
	}
}

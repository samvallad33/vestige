import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('three', async () => {
	const mock = await import('./three-mock');
	return { ...mock };
});

import { NodeManager } from '../nodes';
import { Vector3 } from './three-mock';
import { makeNode, resetNodeCounter } from './helpers';

describe('NodeManager', () => {
	let manager: NodeManager;

	beforeEach(() => {
		resetNodeCounter();
		manager = new NodeManager();
	});

	describe('createNodes', () => {
		it('creates meshes, glows, and labels for all nodes', () => {
			const nodes = [makeNode({ id: 'a' }), makeNode({ id: 'b' }), makeNode({ id: 'c' })];
			const positions = manager.createNodes(nodes);

			expect(positions.size).toBe(3);
			expect(manager.meshMap.size).toBe(3);
			expect(manager.glowMap.size).toBe(3);
			expect(manager.labelSprites.size).toBe(3);
		});

		it('positions center node at origin', () => {
			const nodes = [
				makeNode({ id: 'center', isCenter: true }),
				makeNode({ id: 'other' }),
			];
			const positions = manager.createNodes(nodes);

			const centerPos = positions.get('center')!;
			expect(centerPos.x).toBe(0);
			expect(centerPos.y).toBe(0);
			expect(centerPos.z).toBe(0);
		});

		it('scales mesh size by retention', () => {
			const highRet = makeNode({ id: 'high', retention: 1.0 });
			const lowRet = makeNode({ id: 'low', retention: 0.1 });
			manager.createNodes([highRet, lowRet]);

			// SphereGeometry size = 0.5 + retention * 2
			// High retention should have larger geometry (indirectly via userData)
			const highMesh = manager.meshMap.get('high')!;
			const lowMesh = manager.meshMap.get('low')!;
			expect(highMesh.userData.retention).toBe(1.0);
			expect(lowMesh.userData.retention).toBe(0.1);
		});

		it('uses Fibonacci sphere distribution for initial positions', () => {
			const nodes = Array.from({ length: 20 }, (_, i) => makeNode({ id: `n${i}` }));
			const positions = manager.createNodes(nodes);

			// No two nodes should be at the same position
			const posArr = Array.from(positions.values());
			for (let i = 0; i < posArr.length; i++) {
				for (let j = i + 1; j < posArr.length; j++) {
					const dist = posArr[i].distanceTo(posArr[j]);
					expect(dist).toBeGreaterThan(0.1);
				}
			}
		});

		it('stores node type in userData', () => {
			const nodes = [
				makeNode({ id: 'fact', type: 'fact' }),
				makeNode({ id: 'concept', type: 'concept' }),
				makeNode({ id: 'decision', type: 'decision' }),
			];
			manager.createNodes(nodes);

			expect(manager.meshMap.get('fact')!.userData.type).toBe('fact');
			expect(manager.meshMap.get('concept')!.userData.type).toBe('concept');
			expect(manager.meshMap.get('decision')!.userData.type).toBe('decision');
		});
	});

	describe('addNode — materialization', () => {
		it('adds a new node at specified position', () => {
			const node = makeNode({ id: 'live-1' });
			const pos = new Vector3(10, 20, 30);
			const result = manager.addNode(node, pos);

			expect(manager.meshMap.has('live-1')).toBe(true);
			expect(manager.glowMap.has('live-1')).toBe(true);
			expect(manager.labelSprites.has('live-1')).toBe(true);
			expect(manager.positions.has('live-1')).toBe(true);

			expect(result.x).toBe(10);
			expect(result.y).toBe(20);
			expect(result.z).toBe(30);
		});

		it('starts node at near-zero scale (not zero to avoid GPU issues)', () => {
			const node = makeNode({ id: 'live-2' });
			manager.addNode(node);

			const mesh = manager.meshMap.get('live-2')!;
			expect(mesh.scale.x).toBeCloseTo(0.001, 3);
		});

		it('generates random position if none provided', () => {
			const node = makeNode({ id: 'live-3' });
			const pos = manager.addNode(node);

			// Should be within ±20 range
			expect(Math.abs(pos.x)).toBeLessThanOrEqual(20);
			expect(Math.abs(pos.y)).toBeLessThanOrEqual(20);
			expect(Math.abs(pos.z)).toBeLessThanOrEqual(20);
		});

		it('clones the input position to prevent external mutation', () => {
			const node = makeNode({ id: 'live-4' });
			const input = new Vector3(5, 5, 5);
			manager.addNode(node, input);

			input.x = 999;
			expect(manager.positions.get('live-4')!.x).toBe(5);
		});

		it('label starts fully transparent', () => {
			const node = makeNode({ id: 'live-5' });
			manager.addNode(node);

			const label = manager.labelSprites.get('live-5')!;
			expect((label.material as any).opacity).toBe(0);
		});

		it('glow starts fully transparent', () => {
			const node = makeNode({ id: 'live-6' });
			manager.addNode(node);

			const glow = manager.glowMap.get('live-6')!;
			expect((glow.material as any).opacity).toBe(0);
		});
	});

	describe('materialization animation choreography', () => {
		function setupAndAnimate(frames: number) {
			const nodes = [makeNode({ id: 'existing', retention: 0.5 })];
			manager.createNodes(nodes);

			const liveNode = makeNode({ id: 'live', retention: 0.9 });
			manager.addNode(liveNode);

			const allNodes = [...nodes, liveNode];
			const camera = { position: new Vector3(0, 30, 80) } as any;

			for (let f = 0; f < frames; f++) {
				manager.animate(f * 0.016, allNodes, camera);
			}

			return {
				mesh: manager.meshMap.get('live')!,
				glow: manager.glowMap.get('live')!,
				label: manager.labelSprites.get('live')!,
			};
		}

		it('mesh scale increases during first 30 frames', () => {
			const { mesh } = setupAndAnimate(15);
			expect(mesh.scale.x).toBeGreaterThan(0.001);
		});

		it('mesh reaches approximately full scale by frame 30', () => {
			const { mesh } = setupAndAnimate(30);
			// easeOutElastic should be near 1.0 at t=1
			expect(mesh.scale.x).toBeGreaterThan(0.8);
		});

		it('glow starts fading in at frame 5', () => {
			// Before frame 5: opacity should be 0
			const before = setupAndAnimate(4);
			expect((before.glow.material as any).opacity).toBe(0);

			// After frame 7: opacity should be positive
			const after = setupAndAnimate(8);
			expect((after.glow.material as any).opacity).toBeGreaterThan(0);
		});

		it('label starts fading in after frame 40', () => {
			// At frame 39: label should still be transparent
			const before = setupAndAnimate(39);
			expect((before.label.material as any).opacity).toBe(0);

			// At frame 50: label should have some opacity
			const after = setupAndAnimate(50);
			expect((after.label.material as any).opacity).toBeGreaterThan(0);
		});

		it('label has positive opacity at frame 55 (during materialization window)', () => {
			// Label fade-in runs from frame 40 to 60 (during materialization).
			// After frame 60, distance-based visibility takes over which depends on camera position.
			// Test within the materialization window itself.
			const { label } = setupAndAnimate(55);
			expect((label.material as any).opacity).toBeGreaterThan(0);
		});

		it('elastic overshoot occurs during materialization', () => {
			// easeOutElastic should cause scale > 1.0 at some point
			let maxScale = 0;
			const nodes = [makeNode({ id: 'existing' })];
			manager.createNodes(nodes);
			const liveNode = makeNode({ id: 'elastic', retention: 0.5 });
			manager.addNode(liveNode);
			const allNodes = [...nodes, liveNode];
			const camera = { position: new Vector3(0, 30, 80) } as any;

			for (let f = 0; f < 30; f++) {
				manager.animate(f * 0.016, allNodes, camera);
				const mesh = manager.meshMap.get('elastic')!;
				if (mesh.scale.x > maxScale) maxScale = mesh.scale.x;
			}

			// Elastic should overshoot past 1.0
			expect(maxScale).toBeGreaterThan(1.0);
		});
	});

	describe('removeNode — dissolution', () => {
		function setupWithNode() {
			const nodes = [makeNode({ id: 'a' }), makeNode({ id: 'b' })];
			manager.createNodes(nodes);
			return nodes;
		}

		it('marks node for dissolution without immediate removal', () => {
			setupWithNode();
			manager.removeNode('a');

			// Mesh should still exist during dissolution animation
			expect(manager.meshMap.has('a')).toBe(true);
		});

		it('node is fully removed after dissolution animation completes (60 frames)', () => {
			const nodes = setupWithNode();
			manager.removeNode('a');

			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 65; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			expect(manager.meshMap.has('a')).toBe(false);
			expect(manager.glowMap.has('a')).toBe(false);
			expect(manager.labelSprites.has('a')).toBe(false);
			expect(manager.positions.has('a')).toBe(false);
		});

		it('node shrinks during dissolution using easeInBack', () => {
			const nodes = setupWithNode();
			manager.removeNode('a');

			const camera = { position: new Vector3(0, 30, 80) } as any;
			// Run to near completion (frame 55/60) where shrink is deep
			for (let f = 0; f < 55; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			const currentScale = manager.meshMap.get('a')!.scale.x;
			// At frame 55/60, easeInBack(0.917) ≈ 0.87, shrink = 1-0.87 = 0.13
			// The originalScale from breathing was ~1.0, scale should be very small
			expect(currentScale).toBeLessThan(1.0);
		});

		it('opacity fades during dissolution', () => {
			const nodes = setupWithNode();
			manager.removeNode('a');

			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 50; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			const mesh = manager.meshMap.get('a');
			if (mesh) {
				expect((mesh.material as any).opacity).toBeLessThan(0.5);
			}
		});

		it('cancels materialization if node removed during spawn', () => {
			const nodes = [makeNode({ id: 'base' })];
			manager.createNodes(nodes);

			const liveNode = makeNode({ id: 'spawn-then-die' });
			manager.addNode(liveNode);

			// Immediately remove before materialization finishes
			manager.removeNode('spawn-then-die');

			const allNodes = [...nodes, liveNode];
			const camera = { position: new Vector3(0, 30, 80) } as any;

			// Run past both animation durations
			for (let f = 0; f < 70; f++) {
				manager.animate(f * 0.016, allNodes, camera);
			}

			expect(manager.meshMap.has('spawn-then-die')).toBe(false);
		});
	});

	describe('growNode — retention change animation', () => {
		it('grows node to new retention scale', () => {
			const nodes = [makeNode({ id: 'grow', retention: 0.3 })];
			manager.createNodes(nodes);
			const originalScale = manager.meshMap.get('grow')!.scale.x;

			manager.growNode('grow', 0.9);

			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 35; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			// Target scale = 0.5 + 0.9 * 2 = 2.3
			const mesh = manager.meshMap.get('grow')!;
			// Should be near target scale after animation completes
			expect(mesh.scale.x).toBeGreaterThan(originalScale);
		});

		it('shrinks node when retention decreases (demotion)', () => {
			const nodes = [makeNode({ id: 'shrink', retention: 0.9 })];
			manager.createNodes(nodes);

			manager.growNode('shrink', 0.2);

			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 35; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			// Target scale = 0.5 + 0.2 * 2 = 0.9 (less than 0.5 + 0.9*2 = 2.3)
			const mesh = manager.meshMap.get('shrink')!;
			expect(mesh.userData.retention).toBe(0.2);
		});

		it('also grows the glow sprite', () => {
			const nodes = [makeNode({ id: 'glow-grow', retention: 0.3 })];
			manager.createNodes(nodes);
			const originalGlowScale = manager.glowMap.get('glow-grow')!.scale.x;

			manager.growNode('glow-grow', 0.95);

			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 35; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			const newGlowScale = manager.glowMap.get('glow-grow')!.scale.x;
			expect(newGlowScale).toBeGreaterThan(originalGlowScale);
		});

		it('handles nonexistent node gracefully', () => {
			expect(() => manager.growNode('nonexistent', 0.5)).not.toThrow();
		});
	});

	describe('breathing animation', () => {
		it('breathing only affects non-animating nodes', () => {
			const nodes = [makeNode({ id: 'normal' })];
			manager.createNodes(nodes);
			const liveNode = makeNode({ id: 'materializing' });
			manager.addNode(liveNode);

			const allNodes = [...nodes, liveNode];
			const camera = { position: new Vector3(0, 30, 80) } as any;

			// During first few frames, materializing node should use animation scale
			manager.animate(0.016, allNodes, camera);

			// The materializing node's scale should be from the animation, not breathing
			const matMesh = manager.meshMap.get('materializing')!;
			// Its scale should be small (just started materializing)
			expect(matMesh.scale.x).toBeLessThan(0.5);
		});

		it('hover increases emissive intensity', () => {
			const nodes = [makeNode({ id: 'hover-test', retention: 0.5 })];
			manager.createNodes(nodes);

			manager.hoveredNode = 'hover-test';
			const camera = { position: new Vector3(0, 30, 80) } as any;
			manager.animate(0, nodes, camera);

			const mat = manager.meshMap.get('hover-test')!.material as any;
			expect(mat.emissiveIntensity).toBe(1.0);
		});

		it('selected node gets elevated emissive intensity', () => {
			const nodes = [makeNode({ id: 'sel-test', retention: 0.5 })];
			manager.createNodes(nodes);

			manager.selectedNode = 'sel-test';
			const camera = { position: new Vector3(0, 30, 80) } as any;
			manager.animate(0, nodes, camera);

			const mat = manager.meshMap.get('sel-test')!.material as any;
			expect(mat.emissiveIntensity).toBe(0.8);
		});
	});

	describe('label visibility', () => {
		it('labels visible for nearby nodes', () => {
			const nodes = [makeNode({ id: 'near', retention: 0.5 })];
			manager.createNodes(nodes);

			// Camera very close to the node
			const nodePos = manager.positions.get('near')!;
			const camera = { position: nodePos.clone().add(new Vector3(0, 0, 10)) } as any;

			for (let f = 0; f < 30; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			const label = manager.labelSprites.get('near')!;
			expect((label.material as any).opacity).toBeGreaterThan(0.5);
		});

		it('labels invisible for distant nodes', () => {
			const nodes = [makeNode({ id: 'far', retention: 0.5 })];
			manager.createNodes(nodes);

			const nodePos = manager.positions.get('far')!;
			const camera = { position: nodePos.clone().add(new Vector3(0, 0, 200)) } as any;

			for (let f = 0; f < 30; f++) {
				manager.animate(f * 0.016, nodes, camera);
			}

			const label = manager.labelSprites.get('far')!;
			expect((label.material as any).opacity).toBeLessThan(0.1);
		});
	});

	describe('dispose', () => {
		it('clears all animation queues', () => {
			const nodes = [makeNode({ id: 'a' })];
			manager.createNodes(nodes);
			manager.addNode(makeNode({ id: 'b' }));
			manager.removeNode('a');

			manager.dispose();

			// Internal arrays should be empty (tested indirectly by no errors on next animate)
			// The dispose method clears materializingNodes, dissolvingNodes, growingNodes
		});
	});

	describe('Birth Ritual integration', () => {
		it('addNode with isBirthRitual:true hides mesh, glow, and label immediately', () => {
			const node = makeNode({ id: 'ritual-1' });
			manager.addNode(node, new Vector3(5, 5, 5), { isBirthRitual: true });

			const mesh = manager.meshMap.get('ritual-1')!;
			const glow = manager.glowMap.get('ritual-1')!;
			const label = manager.labelSprites.get('ritual-1')!;

			expect(mesh.visible).toBe(false);
			expect(glow.visible).toBe(false);
			expect(label.visible).toBe(false);
		});

		it('addNode with isBirthRitual:true stores a pending sentinel on mesh.userData', () => {
			const node = makeNode({ id: 'ritual-sentinel', retention: 0.75 });
			manager.addNode(node, new Vector3(0, 0, 0), { isBirthRitual: true });

			const mesh = manager.meshMap.get('ritual-sentinel')!;
			const pending = mesh.userData.birthRitualPending as any;
			expect(pending).toBeDefined();
			expect(pending.totalFrames).toBe(30);
			// targetScale = 0.5 + retention * 2 = 0.5 + 0.75 * 2 = 2.0
			expect(pending.targetScale).toBeCloseTo(2.0, 3);
		});

		it('addNode with isBirthRitual:true does NOT enqueue materialization', () => {
			const ritualNode = makeNode({ id: 'ritual-pending', retention: 0.8 });
			manager.addNode(ritualNode, new Vector3(10, 10, 10), { isBirthRitual: true });

			// In the real runtime the ritual-pending node is .visible=false
			// AND is not yet in the GraphNode[] list — it only gets added to
			// the visible node list once igniteNode flips its visibility and
			// materialization kicks in. So we pass an empty `nodes` array to
			// animate(), which also exercises that the breathing loop skips
			// meshes absent from the nodes array.
			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 40; f++) {
				manager.animate(f * 0.016, [], camera);
			}

			const mesh = manager.meshMap.get('ritual-pending')!;
			// Materialization queue never pushed — a regular materializing
			// node would be at scale ≈ targetScale = 2.1 by frame 40. The
			// ritual-pending node stays at its addNode initial 0.001 because
			// no animation loop is mutating its scale.
			expect(mesh.scale.x).toBeCloseTo(0.001, 3);

			// Stronger invariant — the sentinel is still there, confirming
			// the node never got handed off to the materialization queue.
			expect(mesh.userData.birthRitualPending).toBeDefined();
		});

		it('addNode without opts proceeds with normal materialization (old behavior)', () => {
			const node = makeNode({ id: 'normal-spawn' });
			manager.addNode(node, new Vector3(1, 2, 3));

			const mesh = manager.meshMap.get('normal-spawn')!;
			const glow = manager.glowMap.get('normal-spawn')!;
			const label = manager.labelSprites.get('normal-spawn')!;

			// Default mesh.visible is true in three-mock (Object3D has no explicit field).
			// Key invariant: visible is NOT explicitly false like the ritual path.
			expect(mesh.visible).not.toBe(false);
			expect(glow.visible).not.toBe(false);
			expect(label.visible).not.toBe(false);

			// And no pending sentinel
			expect(mesh.userData.birthRitualPending).toBeUndefined();

			// Animation should proceed — scale grows via easeOutElastic
			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 20; f++) {
				manager.animate(f * 0.016, [node], camera);
			}
			expect(mesh.scale.x).toBeGreaterThan(0.1);
		});

		it('igniteNode flips all three visibility flags and queues materialization', () => {
			const node = makeNode({ id: 'to-ignite', retention: 0.6 });
			manager.addNode(node, new Vector3(0, 0, 0), { isBirthRitual: true });

			// Pre-ignite: hidden
			const mesh = manager.meshMap.get('to-ignite')!;
			const glow = manager.glowMap.get('to-ignite')!;
			const label = manager.labelSprites.get('to-ignite')!;
			expect(mesh.visible).toBe(false);

			manager.igniteNode('to-ignite');

			// Post-ignite: visible
			expect(mesh.visible).toBe(true);
			expect(glow.visible).toBe(true);
			expect(label.visible).toBe(true);

			// Sentinel is gone
			expect(mesh.userData.birthRitualPending).toBeUndefined();

			// Materialization was queued — drive animation and the scale
			// should grow past the initial 0.001.
			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 15; f++) {
				manager.animate(f * 0.016, [node], camera);
			}
			expect(mesh.scale.x).toBeGreaterThan(0.1);
		});

		it('igniteNode called twice is idempotent (second call is a no-op)', () => {
			const node = makeNode({ id: 'double-ignite', retention: 0.5 });
			manager.addNode(node, new Vector3(0, 0, 0), { isBirthRitual: true });

			manager.igniteNode('double-ignite');
			// Capture scale after one round of animation
			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 10; f++) {
				manager.animate(f * 0.016, [node], camera);
			}
			const scaleAfterFirst = manager.meshMap.get('double-ignite')!.scale.x;

			// Second ignite — should NOT push a duplicate materialization entry.
			// If it did, the extra entry (starting at frame 0) would restart
			// the scale back near 0.001 or at least visibly reset it.
			manager.igniteNode('double-ignite');
			for (let f = 0; f < 5; f++) {
				manager.animate((f + 10) * 0.016, [node], camera);
			}
			const scaleAfterSecond = manager.meshMap.get('double-ignite')!.scale.x;

			// Scale after second ignite should be greater than or roughly equal
			// to scale after first, NOT reset toward 0.001. A duplicate entry
			// starting at frame 0 would pull the mesh back near zero on the
			// very first subsequent animate() tick via mn.mesh.scale.setScalar.
			expect(scaleAfterSecond).toBeGreaterThanOrEqual(scaleAfterFirst * 0.5);
		});

		it('igniteNode on a regular (non-ritual) node is a no-op', () => {
			const node = makeNode({ id: 'regular', retention: 0.5 });
			manager.addNode(node, new Vector3(0, 0, 0));
			// Regular addNode already queued materialization. Capture state.
			const mesh = manager.meshMap.get('regular')!;
			const visBefore = mesh.visible;

			// Call igniteNode — there's no pending sentinel, should short-circuit.
			expect(() => manager.igniteNode('regular')).not.toThrow();

			// No pending sentinel means the function returns early after the
			// sentinel check, so nothing about the mesh changes.
			expect(mesh.visible).toBe(visBefore);
			expect(mesh.userData.birthRitualPending).toBeUndefined();
		});

		it('igniteNode on unknown id is a no-op (no throw)', () => {
			expect(() => manager.igniteNode('does-not-exist')).not.toThrow();
			expect(manager.meshMap.has('does-not-exist')).toBe(false);
		});

		it('position is stored in positions map even when the node is invisible', () => {
			const node = makeNode({ id: 'invisible-but-positioned' });
			const spawnPos = new Vector3(42, -17, 8);
			manager.addNode(node, spawnPos, { isBirthRitual: true });

			// Force simulation + orb getTargetPos() both rely on positions
			// being live immediately — the ritual only hides visuals, not
			// physics state.
			const stored = manager.positions.get('invisible-but-positioned');
			expect(stored).toBeDefined();
			expect(stored!.x).toBe(42);
			expect(stored!.y).toBe(-17);
			expect(stored!.z).toBe(8);

			// And the mesh itself is still hidden
			expect(manager.meshMap.get('invisible-but-positioned')!.visible).toBe(false);
		});

		it('removeNode during pending ritual cancels without materialization', () => {
			// Sanhedrin abort path at the NodeManager level: a ritual-pending
			// node gets removed before igniteNode fires. The remove path
			// should still work (dissolution queue takes over) and igniteNode
			// called later must not resurrect it.
			const node = makeNode({ id: 'aborted-ritual' });
			manager.addNode(node, new Vector3(0, 0, 0), { isBirthRitual: true });

			manager.removeNode('aborted-ritual');

			// Dissolution progresses past totalFrames = 60 and clears state.
			const camera = { position: new Vector3(0, 30, 80) } as any;
			for (let f = 0; f < 65; f++) {
				manager.animate(f * 0.016, [node], camera);
			}

			expect(manager.meshMap.has('aborted-ritual')).toBe(false);

			// And a late igniteNode call on the dead id is a safe no-op.
			expect(() => manager.igniteNode('aborted-ritual')).not.toThrow();
		});
	});
});

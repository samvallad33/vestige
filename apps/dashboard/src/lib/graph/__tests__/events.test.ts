import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('three', async () => {
	const mock = await import('./three-mock');
	return { ...mock };
});

import { mapEventToEffects, resetLiveSpawnTracking, type GraphMutationContext, type GraphMutation } from '../events';
import { NodeManager } from '../nodes';
import { EdgeManager } from '../edges';
import { EffectManager } from '../effects';
import { ForceSimulation } from '../force-sim';
import { Vector3, Scene, RingGeometry, Mesh, Points, Sprite } from './three-mock';
import { makeNode, makeEdge, makeEvent, resetNodeCounter } from './helpers';
import type { GraphNode, VestigeEvent } from '$types';

describe('Event-to-Mutation Pipeline', () => {
	let nodeManager: NodeManager;
	let edgeManager: EdgeManager;
	let effects: EffectManager;
	let forceSim: ForceSimulation;
	let scene: InstanceType<typeof Scene>;
	let camera: any;
	let mutations: GraphMutation[];
	let allNodes: GraphNode[];
	let ctx: GraphMutationContext;

	beforeEach(() => {
		resetNodeCounter();
		resetLiveSpawnTracking();
		scene = new Scene();
		camera = { position: new Vector3(0, 30, 80) };
		nodeManager = new NodeManager();
		edgeManager = new EdgeManager();
		effects = new EffectManager(scene as any);
		mutations = [];

		// Create initial graph with 5 nodes
		const initialNodes = [
			makeNode({ id: 'n1', type: 'fact', tags: ['rust', 'bug-fix'] }),
			makeNode({ id: 'n2', type: 'concept', tags: ['architecture'] }),
			makeNode({ id: 'n3', type: 'decision', tags: ['rust'] }),
			makeNode({ id: 'n4', type: 'fact', tags: ['testing'] }),
			makeNode({ id: 'n5', type: 'event', tags: ['session'] }),
		];

		const positions = nodeManager.createNodes(initialNodes);
		edgeManager.createEdges(
			[makeEdge('n1', 'n2'), makeEdge('n2', 'n3'), makeEdge('n3', 'n4')],
			positions
		);
		forceSim = new ForceSimulation(positions);

		allNodes = [...initialNodes];

		ctx = {
			effects,
			nodeManager,
			edgeManager,
			forceSim,
			camera,
			onMutation: (m: GraphMutation) => mutations.push(m),
		};
	});

	describe('MemoryCreated', () => {
		it('creates a new node in all managers', () => {
			const event = makeEvent('MemoryCreated', {
				id: 'new-1',
				content: 'I love Rust',
				node_type: 'fact',
				tags: ['rust', 'preference'],
				retention: 0.9,
			});

			mapEventToEffects(event, ctx, allNodes);

			expect(nodeManager.meshMap.has('new-1')).toBe(true);
			expect(nodeManager.positions.has('new-1')).toBe(true);
			expect(forceSim.positions.has('new-1')).toBe(true);
		});

		it('emits nodeAdded mutation', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'new-2',
					content: 'test',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const nodeAdded = mutations.find((m) => m.type === 'nodeAdded');
			expect(nodeAdded).toBeDefined();
			expect((nodeAdded as any).node.id).toBe('new-2');
		});

		it('builds GraphNode from event data', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'new-3',
					content: 'Complex memory about architecture decisions in Rust systems',
					node_type: 'decision',
					tags: ['architecture', 'rust'],
					retention: 0.75,
				}),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeAdded') as any;
			expect(mutation.node.type).toBe('decision');
			expect(mutation.node.tags).toEqual(['architecture', 'rust']);
			expect(mutation.node.retention).toBe(0.75);
			expect(mutation.node.label).toBe('Complex memory about architecture decisions in Rust systems');
			expect(mutation.node.isCenter).toBe(false);
		});

		it('truncates label to 60 characters', () => {
			const longContent = 'A'.repeat(100);
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'long',
					content: longContent,
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeAdded') as any;
			expect(mutation.node.label.length).toBe(60);
		});

		it('spawns node near related nodes (tag overlap scoring)', () => {
			// Create a memory with rust tag — should spawn near n1 (which has rust tag)
			const n1Pos = nodeManager.positions.get('n1')!;

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'rust-memory',
					content: 'Rust borrow checker tip',
					node_type: 'fact',
					tags: ['rust'],
				}),
				ctx,
				allNodes
			);

			const newPos = nodeManager.positions.get('rust-memory')!;
			const distToN1 = newPos.distanceTo(n1Pos);

			// Should be relatively close to n1 (within jitter range ~10 units)
			expect(distToN1).toBeLessThan(20);
		});

		it('spawns a v2.3 birth orb in the scene', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'new-burst',
					content: 'test',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// Birth orb adds a halo sprite + bright core sprite to the scene
			// immediately. The arrival-cascade effects (rainbow burst, shockwaves,
			// ripple wave) are deferred to the orb's onArrive callback — covered
			// by the "fires arrival cascade after ritual" test below.
			expect(scene.children.length).toBeGreaterThanOrEqual(childrenBefore + 2);
		});

		it('fires the arrival cascade after the birth ritual completes', () => {
			vi.useFakeTimers();

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'cascade-check',
					content: 'test',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const afterSpawn = scene.children.length;

			// Drive the effects update loop past the full ritual duration
			// (gestation 48 + flight 90 = 138 frames). Each tick is one frame;
			// we run 150 to give onArrive room to fire.
			for (let i = 0; i < 150; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// Advance the setTimeout that schedules the delayed second shockwave.
			vi.advanceTimersByTime(250);

			// The arrival cascade should have added a rainbow burst, shockwave,
			// ripple wave, and delayed second shockwave to the scene. Even after
			// the orb fades out and is removed, the burst particles persist long
			// enough that children.length should exceed the post-spawn count.
			expect(scene.children.length).toBeGreaterThan(afterSpawn);

			vi.useRealTimers();
		});

		it('uses default values when event data is incomplete', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', { id: 'minimal' }),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeAdded') as any;
			expect(mutation.node.type).toBe('fact');
			expect(mutation.node.retention).toBe(0.9);
			expect(mutation.node.tags).toEqual([]);
		});

		it('ignores event without id', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', { content: 'no id' }),
				ctx,
				allNodes
			);

			expect(mutations.length).toBe(0);
		});
	});

	describe('FIFO eviction', () => {
		it('evicts oldest live node when exceeding 50 cap', () => {
			// Create 51 live nodes
			for (let i = 0; i < 51; i++) {
				mapEventToEffects(
					makeEvent('MemoryCreated', {
						id: `live-${i}`,
						content: `Memory ${i}`,
						node_type: 'fact',
					}),
					ctx,
					allNodes
				);
			}

			// First live node should have been evicted
			const removedMutations = mutations.filter((m) => m.type === 'nodeRemoved');
			expect(removedMutations.length).toBeGreaterThan(0);
			expect((removedMutations[0] as any).nodeId).toBe('live-0');
		});

		it('evicted node is removed from all managers', () => {
			for (let i = 0; i < 51; i++) {
				mapEventToEffects(
					makeEvent('MemoryCreated', {
						id: `evict-${i}`,
						content: `Memory ${i}`,
						node_type: 'fact',
					}),
					ctx,
					allNodes
				);
			}

			// First node should be gone from node manager and force sim
			expect(forceSim.positions.has('evict-0')).toBe(false);
		});

		it('initial nodes are NOT subject to FIFO eviction', () => {
			// Even after adding 50 live nodes, initial nodes should still exist
			for (let i = 0; i < 50; i++) {
				mapEventToEffects(
					makeEvent('MemoryCreated', {
						id: `extra-${i}`,
						content: `Memory ${i}`,
						node_type: 'fact',
					}),
					ctx,
					allNodes
				);
			}

			expect(nodeManager.meshMap.has('n1')).toBe(true);
			expect(nodeManager.meshMap.has('n2')).toBe(true);
			expect(nodeManager.meshMap.has('n3')).toBe(true);
		});
	});

	describe('ConnectionDiscovered', () => {
		it('adds edge with growth animation', () => {
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n4',
					weight: 0.8,
					connection_type: 'causal',
				}),
				ctx,
				allNodes
			);

			// Edge should have been added
			expect(edgeManager.group.children.length).toBeGreaterThan(3); // 3 initial + 1 new
		});

		it('emits edgeAdded mutation with correct data', () => {
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n5',
					weight: 0.7,
					connection_type: 'semantic',
				}),
				ctx,
				allNodes
			);

			const edgeMutation = mutations.find((m) => m.type === 'edgeAdded') as any;
			expect(edgeMutation).toBeDefined();
			expect(edgeMutation.edge.source).toBe('n1');
			expect(edgeMutation.edge.target).toBe('n5');
			expect(edgeMutation.edge.weight).toBe(0.7);
			expect(edgeMutation.edge.type).toBe('semantic');
		});

		it('creates connection flash between endpoints', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n2',
				}),
				ctx,
				allNodes
			);

			expect(scene.children.length).toBeGreaterThan(childrenBefore);
		});

		it('pulses both endpoint nodes', () => {
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n2',
				}),
				ctx,
				allNodes
			);

			const n1Pulse = effects.pulseEffects.find((p) => p.nodeId === 'n1');
			const n2Pulse = effects.pulseEffects.find((p) => p.nodeId === 'n2');
			expect(n1Pulse).toBeDefined();
			expect(n2Pulse).toBeDefined();
		});

		it('uses default weight and type when not provided', () => {
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n5',
				}),
				ctx,
				allNodes
			);

			const edgeMutation = mutations.find((m) => m.type === 'edgeAdded') as any;
			expect(edgeMutation.edge.weight).toBe(0.5);
			expect(edgeMutation.edge.type).toBe('semantic');
		});
	});

	describe('MemoryDeleted', () => {
		it('removes node from all managers', () => {
			mapEventToEffects(
				makeEvent('MemoryDeleted', { id: 'n1' }),
				ctx,
				allNodes
			);

			// Force sim should have removed the node
			expect(forceSim.positions.has('n1')).toBe(false);
		});

		it('removes connected edges', () => {
			mapEventToEffects(
				makeEvent('MemoryDeleted', { id: 'n2' }),
				ctx,
				allNodes
			);

			// Should emit both edgesRemoved and nodeRemoved mutations
			const edgesRemoved = mutations.find((m) => m.type === 'edgesRemoved');
			const nodeRemoved = mutations.find((m) => m.type === 'nodeRemoved');
			expect(edgesRemoved).toBeDefined();
			expect(nodeRemoved).toBeDefined();
		});

		it('creates implosion effect at node position', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('MemoryDeleted', { id: 'n3' }),
				ctx,
				allNodes
			);

			// Should have added implosion particles
			expect(scene.children.length).toBeGreaterThan(childrenBefore);
		});

		it('removes from live tracking if was live-spawned', () => {
			// First, create a live node
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'temp-live',
					content: 'temporary',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			expect(nodeManager.meshMap.has('temp-live')).toBe(true);

			// Now delete it
			mutations = [];
			mapEventToEffects(
				makeEvent('MemoryDeleted', { id: 'temp-live' }),
				ctx,
				allNodes
			);

			const nodeRemoved = mutations.find((m) => m.type === 'nodeRemoved');
			expect(nodeRemoved).toBeDefined();
		});

		it('ignores event without id', () => {
			mapEventToEffects(
				makeEvent('MemoryDeleted', {}),
				ctx,
				allNodes
			);

			expect(mutations.length).toBe(0);
		});
	});

	describe('MemoryPromoted', () => {
		it('grows the node to new retention', () => {
			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'n1', new_retention: 0.95 }),
				ctx,
				allNodes
			);

			// Should have updated userData
			expect(nodeManager.meshMap.get('n1')!.userData.retention).toBe(0.95);
		});

		it('emits nodeUpdated mutation', () => {
			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'n2', new_retention: 0.98 }),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeUpdated') as any;
			expect(mutation).toBeDefined();
			expect(mutation.nodeId).toBe('n2');
			expect(mutation.retention).toBe(0.98);
		});

		it('creates green pulse + shockwave + spawn burst', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'n1' }),
				ctx,
				allNodes
			);

			// Should have green pulse
			const greenPulse = effects.pulseEffects.find((p) => p.nodeId === 'n1');
			expect(greenPulse).toBeDefined();

			// Should have added visual effects to scene
			expect(scene.children.length).toBeGreaterThan(childrenBefore);
		});

		it('uses default retention when not provided', () => {
			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'n1' }),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeUpdated') as any;
			expect(mutation.retention).toBe(0.95); // default
		});

		it('ignores nonexistent node', () => {
			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'nonexistent' }),
				ctx,
				allNodes
			);

			expect(mutations.length).toBe(0);
		});
	});

	describe('MemoryDemoted', () => {
		it('shrinks the node', () => {
			mapEventToEffects(
				makeEvent('MemoryDemoted', { id: 'n1', new_retention: 0.3 }),
				ctx,
				allNodes
			);

			expect(nodeManager.meshMap.get('n1')!.userData.retention).toBe(0.3);
		});

		it('emits nodeUpdated mutation', () => {
			mapEventToEffects(
				makeEvent('MemoryDemoted', { id: 'n2', new_retention: 0.2 }),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeUpdated') as any;
			expect(mutation).toBeDefined();
			expect(mutation.retention).toBe(0.2);
		});

		it('creates red pulse (subtler than promotion)', () => {
			mapEventToEffects(
				makeEvent('MemoryDemoted', { id: 'n1', new_retention: 0.3 }),
				ctx,
				allNodes
			);

			const pulse = effects.pulseEffects.find((p) => p.nodeId === 'n1');
			expect(pulse).toBeDefined();
			expect(pulse!.decay).toBe(0.03); // faster decay = subtler
		});
	});

	describe('MemoryUpdated', () => {
		it('creates blue pulse on existing node', () => {
			mapEventToEffects(
				makeEvent('MemoryUpdated', { id: 'n1' }),
				ctx,
				allNodes
			);

			const pulse = effects.pulseEffects.find((p) => p.nodeId === 'n1');
			expect(pulse).toBeDefined();
		});

		it('updates retention if provided', () => {
			mapEventToEffects(
				makeEvent('MemoryUpdated', { id: 'n1', retention: 0.85 }),
				ctx,
				allNodes
			);

			const mutation = mutations.find((m) => m.type === 'nodeUpdated') as any;
			expect(mutation).toBeDefined();
			expect(mutation.retention).toBe(0.85);
		});

		it('does not emit mutation without retention data', () => {
			mapEventToEffects(
				makeEvent('MemoryUpdated', { id: 'n1' }),
				ctx,
				allNodes
			);

			expect(mutations.length).toBe(0);
		});

		it('ignores nonexistent node', () => {
			mapEventToEffects(
				makeEvent('MemoryUpdated', { id: 'nonexistent' }),
				ctx,
				allNodes
			);

			expect(mutations.length).toBe(0);
			expect(effects.pulseEffects.length).toBe(0);
		});
	});

	describe('SearchPerformed', () => {
		it('pulses all nodes', () => {
			mapEventToEffects(
				makeEvent('SearchPerformed', {}),
				ctx,
				allNodes
			);

			expect(effects.pulseEffects.length).toBe(5); // 5 initial nodes
		});
	});

	describe('DreamStarted', () => {
		it('pulses all nodes with purple', () => {
			mapEventToEffects(
				makeEvent('DreamStarted', {}),
				ctx,
				allNodes
			);

			expect(effects.pulseEffects.length).toBe(5);
			// Purple pulse with slow decay
			effects.pulseEffects.forEach((p) => {
				expect(p.decay).toBe(0.005);
			});
		});
	});

	describe('DreamProgress', () => {
		it('pulses specific memory with high intensity', () => {
			mapEventToEffects(
				makeEvent('DreamProgress', { memory_id: 'n3' }),
				ctx,
				allNodes
			);

			const pulse = effects.pulseEffects.find((p) => p.nodeId === 'n3');
			expect(pulse).toBeDefined();
			expect(pulse!.intensity).toBe(1.5);
		});

		it('ignores nonexistent memory', () => {
			mapEventToEffects(
				makeEvent('DreamProgress', { memory_id: 'nonexistent' }),
				ctx,
				allNodes
			);

			expect(effects.pulseEffects.length).toBe(0);
		});
	});

	describe('DreamCompleted', () => {
		it('creates center burst + shockwave', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('DreamCompleted', {}),
				ctx,
				allNodes
			);

			expect(scene.children.length).toBeGreaterThan(childrenBefore);
		});
	});

	describe('RetentionDecayed', () => {
		it('adds red pulse to decayed node', () => {
			mapEventToEffects(
				makeEvent('RetentionDecayed', { id: 'n2' }),
				ctx,
				allNodes
			);

			const pulse = effects.pulseEffects.find((p) => p.nodeId === 'n2');
			expect(pulse).toBeDefined();
		});
	});

	describe('ConsolidationCompleted', () => {
		it('pulses all nodes with orange', () => {
			mapEventToEffects(
				makeEvent('ConsolidationCompleted', {}),
				ctx,
				allNodes
			);

			expect(effects.pulseEffects.length).toBe(5);
		});
	});

	describe('ActivationSpread', () => {
		it('creates flashes from source to all targets', () => {
			const childrenBefore = scene.children.length;

			mapEventToEffects(
				makeEvent('ActivationSpread', {
					source_id: 'n1',
					target_ids: ['n2', 'n3', 'n4'],
				}),
				ctx,
				allNodes
			);

			expect(scene.children.length).toBe(childrenBefore + 3);
		});
	});

	describe('spawn position scoring', () => {
		it('type match scores higher than tag match', () => {
			// n1 is type: 'fact', tags: ['rust', 'bug-fix']
			// n2 is type: 'concept', tags: ['architecture']
			// Creating a 'fact' with 'architecture' tag — should favor n1 (type match = 2 points)
			// vs n2 (tag match = 1 point)
			const n1Pos = nodeManager.positions.get('n1')!;
			const n2Pos = nodeManager.positions.get('n2')!;

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'type-vs-tag',
					content: 'test',
					node_type: 'fact',
					tags: ['architecture'],
				}),
				ctx,
				allNodes
			);

			const newPos = nodeManager.positions.get('type-vs-tag')!;
			const distToN1 = newPos.distanceTo(n1Pos);
			const distToN2 = newPos.distanceTo(n2Pos);

			// Should be closer to n1 (type match wins)
			expect(distToN1).toBeLessThan(distToN2);
		});

		it('falls back to random position when no matches', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'no-match',
					content: 'test',
					node_type: 'place', // no existing 'place' nodes
					tags: ['geography'], // no matching tags
				}),
				ctx,
				allNodes
			);

			const pos = nodeManager.positions.get('no-match')!;
			// Should be somewhere in the graph space
			expect(Math.abs(pos.x)).toBeLessThan(100);
			expect(Math.abs(pos.y)).toBeLessThan(100);
		});
	});

	describe('full lifecycle integration', () => {
		it('create → promote → delete lifecycle', () => {
			// 1. Create
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'lifecycle',
					content: 'lifecycle test',
					node_type: 'fact',
					retention: 0.7,
				}),
				ctx,
				allNodes
			);

			expect(nodeManager.meshMap.has('lifecycle')).toBe(true);

			// 2. Promote
			mutations = [];
			mapEventToEffects(
				makeEvent('MemoryPromoted', { id: 'lifecycle', new_retention: 0.95 }),
				ctx,
				allNodes
			);

			expect(nodeManager.meshMap.get('lifecycle')!.userData.retention).toBe(0.95);

			// 3. Delete
			mutations = [];
			mapEventToEffects(
				makeEvent('MemoryDeleted', { id: 'lifecycle' }),
				ctx,
				allNodes
			);

			expect(forceSim.positions.has('lifecycle')).toBe(false);
		});

		it('rapid-fire 10 creates without errors', () => {
			for (let i = 0; i < 10; i++) {
				mapEventToEffects(
					makeEvent('MemoryCreated', {
						id: `rapid-${i}`,
						content: `Rapid memory ${i}`,
						node_type: i % 2 === 0 ? 'fact' : 'concept',
						tags: ['rapid'],
						retention: 0.5 + Math.random() * 0.5,
					}),
					ctx,
					allNodes
				);
			}

			expect(nodeManager.meshMap.size).toBe(15); // 5 initial + 10 new
			expect(forceSim.positions.size).toBe(15);

			// All mutations should have been emitted
			const nodeAdded = mutations.filter((m) => m.type === 'nodeAdded');
			expect(nodeAdded.length).toBe(10);
		});

		it('create + connection discovered pipeline', () => {
			// Create two new memories
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'connect-a',
					content: 'Connection source',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'connect-b',
					content: 'Connection target',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// Then discover a connection between them
			mutations = [];
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'connect-a',
					target_id: 'connect-b',
					weight: 0.9,
				}),
				ctx,
				allNodes
			);

			const edgeMutation = mutations.find((m) => m.type === 'edgeAdded');
			expect(edgeMutation).toBeDefined();
		});

		it('dream sequence: start → progress → complete → connections', () => {
			mapEventToEffects(makeEvent('DreamStarted', {}), ctx, allNodes);
			expect(effects.pulseEffects.length).toBe(5);

			mapEventToEffects(makeEvent('DreamProgress', { memory_id: 'n1' }), ctx, allNodes);
			mapEventToEffects(makeEvent('DreamProgress', { memory_id: 'n3' }), ctx, allNodes);

			mapEventToEffects(makeEvent('DreamCompleted', {}), ctx, allNodes);

			// Connections discovered during dream
			mapEventToEffects(
				makeEvent('ConnectionDiscovered', {
					source_id: 'n1',
					target_id: 'n5',
					weight: 0.6,
				}),
				ctx,
				allNodes
			);

			// Should have emitted edgeAdded
			expect(mutations.some((m) => m.type === 'edgeAdded')).toBe(true);
		});
	});

	describe('v2.3 Birth Ritual wiring', () => {
		/** Count shockwave rings currently in the scene by their RingGeometry. */
		function countRings(s: InstanceType<typeof Scene>): number {
			let n = 0;
			for (const child of s.children) {
				if (child instanceof Mesh && child.geometry instanceof RingGeometry) n++;
			}
			return n;
		}

		/** Count Points children — rainbow bursts, spawn bursts, implosions. */
		function countPoints(s: InstanceType<typeof Scene>): number {
			let n = 0;
			for (const child of s.children) if (child instanceof Points) n++;
			return n;
		}

		/** Count Sprite children — birth orb adds a halo + core sprite. */
		function countSprites(s: InstanceType<typeof Scene>): number {
			let n = 0;
			for (const child of s.children) if (child instanceof Sprite) n++;
			return n;
		}

		it('node mesh is hidden immediately after MemoryCreated dispatch', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'ritual-create',
					content: 'fresh memory',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// Ritual path: mesh/glow/label are all .visible = false until
			// igniteNode fires on orb arrival.
			const mesh = nodeManager.meshMap.get('ritual-create')!;
			const glow = nodeManager.glowMap.get('ritual-create')!;
			const label = nodeManager.labelSprites.get('ritual-create')!;
			expect(mesh.visible).toBe(false);
			expect(glow.visible).toBe(false);
			expect(label.visible).toBe(false);

			// Pending sentinel is stamped on userData.
			expect(mesh.userData.birthRitualPending).toBeDefined();
		});

		it('does NOT fire burst/ripple/shockwave at spawn (only the birth orb)', () => {
			const ringsBefore = countRings(scene);
			const pointsBefore = countPoints(scene);
			const spritesBefore = countSprites(scene);

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'spawn-quiet',
					content: 'test',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// Birth orb adds exactly 2 sprites (halo + core). NodeManager's
			// addNode also adds a glow Sprite + label Sprite to the NodeManager
			// GROUP, not to the scene — so spritesBefore -> after delta is +2.
			expect(countSprites(scene) - spritesBefore).toBe(2);

			// No arrival-cascade effects yet: no shockwave rings, no rainbow
			// burst/spawn burst/ripple particles.
			expect(countRings(scene)).toBe(ringsBefore);
			expect(countPoints(scene)).toBe(pointsBefore);
		});

		it('drives through the full ritual: onArrive fires, node becomes visible, scale grows', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'full-ritual',
					content: 'visible after arrival',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const mesh = nodeManager.meshMap.get('full-ritual')!;
			expect(mesh.visible).toBe(false);

			// Drive the effects update loop past the full ritual duration
			// (gestation 48 + flight 90 = 138 frames). After frame 138 the
			// orb fires onArrive which ignites the node and queues materialization.
			for (let i = 0; i < 140; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// Node is now visible and sentinel is cleared.
			expect(mesh.visible).toBe(true);
			expect(mesh.userData.birthRitualPending).toBeUndefined();

			// Run node animation a few frames to let materialization scale grow.
			// Note: onArrive bumped scale by 1.8x (from 0.001 -> 0.0018), then
			// materialization easeOutElastic pulls it toward targetScale.
			for (let f = 0; f < 10; f++) {
				nodeManager.animate(f * 0.016, allNodes, camera);
			}
			expect(mesh.scale.x).toBeGreaterThan(0.001);
		});

		it("Newton's Cradle — target mesh scale is multiplied by 1.8x on arrival", () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'newton-cradle',
					content: 'recoil test',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const mesh = nodeManager.meshMap.get('newton-cradle')!;
			// Pre-arrival: scale is the addNode initial 0.001.
			expect(mesh.scale.x).toBeCloseTo(0.001, 6);

			// Drive just to the moment onArrive fires. Gestation (48) +
			// flight (90) = 138 frames. Arrival bumps scale by 1.8x BEFORE
			// materialization has run any ticks, so the scale should be
			// exactly 0.001 * 1.8 = 0.0018 at that instant. We check right
			// after onArrive (frame 139) — but effects.update progresses the
			// orb's age counter by one each call, and on the tick where
			// orb.age > totalFrames, onArrive fires. We then must NOT tick
			// nodeManager.animate (or materialization would diverge the scale).
			for (let i = 0; i < 140; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// onArrive fired. Scale was 0.001, got multiplied by 1.8 -> 0.0018.
			// Materialization is queued but hasn't run yet (no animate() calls).
			expect(mesh.scale.x).toBeCloseTo(0.0018, 6);
		});

		it('dual shockwave — arrival cascade adds TWO RingGeometry meshes, not one', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'dual-shock',
					content: 'layered crash',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			const ringsBefore = countRings(scene);

			// Drive past full ritual so onArrive fires.
			for (let i = 0; i < 140; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// Both shockwaves fire synchronously in the onArrive callback
			// (the previous setTimeout-delayed second shockwave was dropped
			// because it could outlive the scene on route change).
			const ringsAfter = countRings(scene);
			expect(ringsAfter - ringsBefore).toBe(2);
		});

		it('re-reads position on arrival — fires cascade at force-sim-moved position', () => {
			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'moving-target',
					content: 'follow the node',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// Grab the spawn position, then mutate it to simulate the force
			// simulation pushing the node during the ritual.
			const movedPos = new Vector3(123, 456, -789);
			nodeManager.positions.set('moving-target', movedPos);

			// Drive past full ritual.
			for (let i = 0; i < 140; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// The onArrive callback re-reads nodeManager.positions and fires
			// the cascade at the LIVE position. The two shockwave Ring meshes
			// should have been created at movedPos. Find them and check.
			const rings = scene.children.filter(
				(c) => c instanceof Mesh && c.geometry instanceof RingGeometry
			);
			expect(rings.length).toBeGreaterThanOrEqual(2);
			// Rings for this node: their .position copies from arrivePos at
			// spawn time inside createShockwave.
			const atMovedPos = rings.filter(
				(r) => r.position.x === 123 && r.position.y === 456 && r.position.z === -789
			);
			expect(atMovedPos.length).toBe(2);
		});

		it('Sanhedrin abort path — removeNode before arrival prevents the regular cascade', () => {
			// Spy on the three arrival-cascade emitters so we can assert
			// they were NEVER called when the target is vetoed mid-ritual.
			const burstSpy = vi.spyOn(effects, 'createRainbowBurst');
			const shockwaveSpy = vi.spyOn(effects, 'createShockwave');
			const rippleSpy = vi.spyOn(effects, 'createRippleWave');

			mapEventToEffects(
				makeEvent('MemoryCreated', {
					id: 'vetoed',
					content: 'about to be shattered',
					node_type: 'fact',
				}),
				ctx,
				allNodes
			);

			// The orb's getTargetPos() closure reads
			// nodeManager.positions.get('vetoed'). Dropping the position
			// directly simulates the "target gone" state that the Sanhedrin
			// veto produces after dissolution completes — without needing to
			// drive the full 60-frame dissolution animation.
			nodeManager.positions.delete('vetoed');
			expect(nodeManager.positions.has('vetoed')).toBe(false);

			// Snapshot the orb reference before the update loop disposes it.
			// The abort branch flips `aborted` and tints the halo red; we
			// assert on those fields after the ritual unwinds.
			const orbs = (effects as any).birthOrbs as Array<{
				sprite: { material: { color: any } };
				core: { material: { color: any } };
				aborted: boolean;
			}>;
			expect(orbs.length).toBe(1);
			const orbRef = orbs[0];

			// Drive effects past the full ritual. During flight the orb will
			// see getTargetPos() === undefined, enter the Sanhedrin branch,
			// call createImplosion (anti-birth visual) and SKIP onArrive —
			// so the regular rainbow-burst + dual-shockwave + ripple cascade
			// never fires.
			for (let i = 0; i < 200; i++) {
				effects.update(nodeManager.meshMap, camera, nodeManager.positions);
			}

			// Core assertion: the three regular-cascade emitters were never
			// invoked for the vetoed node.
			expect(burstSpy).not.toHaveBeenCalled();
			expect(shockwaveSpy).not.toHaveBeenCalled();
			expect(rippleSpy).not.toHaveBeenCalled();

			// Also confirm the orb actually took the abort branch, not the
			// gestation-only no-op path (otherwise this test would pass for
			// the wrong reason). The aborted flag is set exactly once inside
			// the Sanhedrin branch.
			expect(orbRef.aborted).toBe(true);
			expect(orbRef.sprite.material.color.r).toBeCloseTo(1.0, 3);
			expect(orbRef.sprite.material.color.g).toBeCloseTo(0.15, 3);

			burstSpy.mockRestore();
			shockwaveSpy.mockRestore();
			rippleSpy.mockRestore();
		});
	});
});

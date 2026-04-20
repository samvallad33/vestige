import * as THREE from 'three';
import type { VestigeEvent, GraphNode, GraphEdge } from '$types';
import { NODE_TYPE_COLORS } from '$types';
import type { EffectManager } from './effects';
import type { NodeManager } from './nodes';
import type { EdgeManager } from './edges';
import type { ForceSimulation } from './force-sim';

/** Maximum number of live-spawned nodes before FIFO eviction */
const MAX_LIVE_NODES = 50;

export interface GraphMutationContext {
	effects: EffectManager;
	nodeManager: NodeManager;
	edgeManager: EdgeManager;
	forceSim: ForceSimulation;
	camera: THREE.Camera;
	onMutation: (mutation: GraphMutation) => void;
}

export type GraphMutation =
	| { type: 'nodeAdded'; node: GraphNode }
	| { type: 'nodeRemoved'; nodeId: string }
	| { type: 'edgeAdded'; edge: GraphEdge }
	| { type: 'edgesRemoved'; nodeId: string }
	| { type: 'nodeUpdated'; nodeId: string; retention: number };

/** Track live-spawned node IDs in insertion order for FIFO eviction */
const liveSpawnedNodes: string[] = [];

/** Reset live spawn tracking (for tests) */
export function resetLiveSpawnTracking() {
	liveSpawnedNodes.length = 0;
}

function findSpawnPosition(
	newNode: { tags?: string[]; type?: string },
	existingNodes: GraphNode[],
	positions: Map<string, THREE.Vector3>
): THREE.Vector3 {
	const tags = newNode.tags ?? [];
	const type = newNode.type ?? '';

	// Score existing nodes by tag overlap + type match
	let bestId: string | null = null;
	let bestScore = 0;

	for (const existing of existingNodes) {
		let score = 0;
		if (existing.type === type) score += 2;
		for (const tag of existing.tags) {
			if (tags.includes(tag)) score += 1;
		}
		if (score > bestScore) {
			bestScore = score;
			bestId = existing.id;
		}
	}

	if (bestId && bestScore > 0) {
		const nearPos = positions.get(bestId);
		if (nearPos) {
			// Spawn nearby with some jitter
			return new THREE.Vector3(
				nearPos.x + (Math.random() - 0.5) * 10,
				nearPos.y + (Math.random() - 0.5) * 10,
				nearPos.z + (Math.random() - 0.5) * 10
			);
		}
	}

	// Fallback: random position in graph space
	return new THREE.Vector3(
		(Math.random() - 0.5) * 40,
		(Math.random() - 0.5) * 40,
		(Math.random() - 0.5) * 40
	);
}

function evictOldestLiveNode(ctx: GraphMutationContext, allNodes: GraphNode[]) {
	if (liveSpawnedNodes.length <= MAX_LIVE_NODES) return;
	const evictId = liveSpawnedNodes.shift()!;
	ctx.edgeManager.removeEdgesForNode(evictId);
	ctx.nodeManager.removeNode(evictId);
	ctx.forceSim.removeNode(evictId);
	ctx.onMutation({ type: 'edgesRemoved', nodeId: evictId });
	ctx.onMutation({ type: 'nodeRemoved', nodeId: evictId });
	// Remove from allNodes tracking
	const idx = allNodes.findIndex((n) => n.id === evictId);
	if (idx !== -1) allNodes.splice(idx, 1);
}

export function mapEventToEffects(
	event: VestigeEvent,
	ctx: GraphMutationContext,
	allNodes: GraphNode[]
) {
	const { effects, nodeManager, edgeManager, forceSim, camera, onMutation } = ctx;
	const nodePositions = nodeManager.positions;
	const nodeMeshMap = nodeManager.meshMap;

	switch (event.type) {
		case 'MemoryCreated': {
			const data = event.data as {
				id?: string;
				content?: string;
				node_type?: string;
				tags?: string[];
				retention?: number;
			};
			if (!data.id) break;

			// Build a GraphNode from event data
			const newNode: GraphNode = {
				id: data.id,
				label: (data.content ?? '').slice(0, 60),
				type: data.node_type ?? 'fact',
				retention: Math.max(0, Math.min(1, data.retention ?? 0.9)),
				tags: data.tags ?? [],
				createdAt: new Date().toISOString(),
				updatedAt: new Date().toISOString(),
				isCenter: false,
			};

			// Find spawn position near related nodes
			const spawnPos = findSpawnPosition(newNode, allNodes, nodePositions);

			// Reserve the physics slot but hide the node until the orb docks.
			// `isBirthRitual:true` skips the 30-frame materialization push, so
			// the mesh/glow/label stay invisible; `igniteNode` below flips
			// visibility and kicks off the elastic scale-up AT the exact
			// millisecond the orb lands — not 100 frames before.
			const pos = nodeManager.addNode(newNode, spawnPos, { isBirthRitual: true });
			forceSim.addNode(data.id, pos);

			// FIFO eviction
			liveSpawnedNodes.push(data.id);
			evictOldestLiveNode(ctx, allNodes);

			// v2.3 Memory Birth Ritual — cosmic-center orb, Bezier flight,
			// arrival burst cascade. The burst/ripple/shockwave cascade
			// fires on arrival at the docking target, not at spawn, so the
			// eye tracks the orb in and the visuals peak on contact.
			const color = new THREE.Color(NODE_TYPE_COLORS[newNode.type] || '#00ffd1');
			const hueShifted = color.clone();
			hueShifted.offsetHSL(0.15, 0, 0);

			effects.createBirthOrb(
				camera,
				color,
				// Re-resolve the live target position every frame — the node
				// is being pushed around by the force sim during flight.
				// Returning undefined here signals "node was aborted" and
				// triggers the Sanhedrin-Shatter anti-birth in effects.ts.
				() => nodeManager.positions.get(newNode.id),
				() => {
					// Dock. Ignite the node (flips visibility + starts
					// materialization) and fire the arrival cascade at the
					// node's CURRENT position — the force sim may have moved
					// the target during the ritual, so we re-read positions.
					nodeManager.igniteNode(newNode.id);
					const arrivePos = nodeManager.positions.get(newNode.id) ?? spawnPos;

					// Newton's Cradle — kinetic transfer into the graph.
					// Bump the mesh scale on impact so the easeOutElastic
					// materialization + force-sim springs physically recoil
					// instead of the orb docking silently.
					const mesh = nodeManager.meshMap.get(newNode.id);
					if (mesh) mesh.scale.multiplyScalar(1.8);

					effects.createRainbowBurst(arrivePos, color);
					effects.createShockwave(arrivePos, color, camera);
					// Fire BOTH shockwaves immediately (different scales /
					// colors for layered crash feel). The previous 166ms
					// setTimeout could outlive the scene on route change
					// and throw an unhandled rejection.
					effects.createShockwave(arrivePos, hueShifted, camera);
					effects.createRippleWave(arrivePos);
				}
			);

			onMutation({ type: 'nodeAdded', node: newNode });
			break;
		}

		case 'ConnectionDiscovered': {
			const data = event.data as {
				source_id?: string;
				target_id?: string;
				weight?: number;
				connection_type?: string;
			};
			if (!data.source_id || !data.target_id) break;

			const srcPos = nodePositions.get(data.source_id);
			const tgtPos = nodePositions.get(data.target_id);

			const newEdge: GraphEdge = {
				source: data.source_id,
				target: data.target_id,
				weight: data.weight ?? 0.5,
				type: data.connection_type ?? 'semantic',
			};

			// Add edge with growth animation
			edgeManager.addEdge(newEdge, nodePositions);

			// Cyan flash + pulse both endpoints
			if (srcPos && tgtPos) {
				effects.createConnectionFlash(srcPos, tgtPos, new THREE.Color(0x00d4ff));
			}
			if (data.source_id && nodeMeshMap.has(data.source_id)) {
				effects.addPulse(data.source_id, 1.0, new THREE.Color(0x00d4ff), 0.02);
			}
			if (data.target_id && nodeMeshMap.has(data.target_id)) {
				effects.addPulse(data.target_id, 1.0, new THREE.Color(0x00d4ff), 0.02);
			}

			onMutation({ type: 'edgeAdded', edge: newEdge });
			break;
		}

		case 'MemoryDeleted': {
			const data = event.data as { id?: string };
			if (!data.id) break;

			const pos = nodePositions.get(data.id);
			if (pos) {
				// Implosion effect first
				const color = new THREE.Color(0xff4757);
				effects.createImplosion(pos, color);
			}

			// Dissolve edges then node
			edgeManager.removeEdgesForNode(data.id);
			nodeManager.removeNode(data.id);
			forceSim.removeNode(data.id);

			// Remove from live tracking
			const liveIdx = liveSpawnedNodes.indexOf(data.id);
			if (liveIdx !== -1) liveSpawnedNodes.splice(liveIdx, 1);

			onMutation({ type: 'edgesRemoved', nodeId: data.id });
			onMutation({ type: 'nodeRemoved', nodeId: data.id });
			break;
		}

		case 'MemoryPromoted': {
			const data = event.data as { id?: string; new_retention?: number };
			const promoId = data?.id;
			if (!promoId) break;

			const newRetention = data.new_retention ?? 0.95;

			if (nodeMeshMap.has(promoId)) {
				// Grow the node
				nodeManager.growNode(promoId, newRetention);

				// Green pulse + shockwave + mini burst
				effects.addPulse(promoId, 1.2, new THREE.Color(0x00ff88), 0.01);
				const promoPos = nodePositions.get(promoId);
				if (promoPos) {
					effects.createShockwave(promoPos, new THREE.Color(0x00ff88), camera);
					effects.createSpawnBurst(promoPos, new THREE.Color(0x00ff88));
				}

				onMutation({ type: 'nodeUpdated', nodeId: promoId, retention: newRetention });
			}
			break;
		}

		case 'MemoryDemoted': {
			const data = event.data as { id?: string; new_retention?: number };
			const demoteId = data?.id;
			if (!demoteId) break;

			const newRetention = data.new_retention ?? 0.3;

			if (nodeMeshMap.has(demoteId)) {
				// Shrink the node
				nodeManager.growNode(demoteId, newRetention);

				// Red pulse — subtle
				effects.addPulse(demoteId, 0.8, new THREE.Color(0xff4757), 0.03);

				onMutation({ type: 'nodeUpdated', nodeId: demoteId, retention: newRetention });
			}
			break;
		}

		case 'MemoryUpdated': {
			const data = event.data as { id?: string; retention?: number };
			const updateId = data?.id;
			if (!updateId || !nodeMeshMap.has(updateId)) break;

			// Subtle blue pulse on update
			effects.addPulse(updateId, 0.6, new THREE.Color(0x818cf8), 0.02);

			if (data.retention !== undefined) {
				nodeManager.growNode(updateId, data.retention);
				onMutation({ type: 'nodeUpdated', nodeId: updateId, retention: data.retention });
			}
			break;
		}

		case 'SearchPerformed': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 0.6 + Math.random() * 0.4, new THREE.Color(0x818cf8), 0.02);
			});
			break;
		}

		case 'DreamStarted': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 1.0, new THREE.Color(0xa855f7), 0.005);
			});
			break;
		}

		case 'DreamProgress': {
			const memoryId = (event.data as { memory_id?: string })?.memory_id;
			if (memoryId && nodeMeshMap.has(memoryId)) {
				effects.addPulse(memoryId, 1.5, new THREE.Color(0xc084fc), 0.01);
			}
			break;
		}

		case 'DreamCompleted': {
			effects.createSpawnBurst(new THREE.Vector3(0, 0, 0), new THREE.Color(0xa855f7));
			effects.createShockwave(new THREE.Vector3(0, 0, 0), new THREE.Color(0xa855f7), camera);
			break;
		}

		case 'RetentionDecayed': {
			const decayId = (event.data as { id?: string })?.id;
			if (decayId && nodeMeshMap.has(decayId)) {
				effects.addPulse(decayId, 0.8, new THREE.Color(0xff4757), 0.03);
			}
			break;
		}

		case 'ConsolidationCompleted': {
			nodeMeshMap.forEach((_, id) => {
				effects.addPulse(id, 0.4 + Math.random() * 0.3, new THREE.Color(0xffb800), 0.015);
			});
			break;
		}

		case 'ActivationSpread': {
			const spreadData = event.data as { source_id?: string; target_ids?: string[] };
			if (spreadData.source_id && spreadData.target_ids) {
				const srcPos = nodePositions.get(spreadData.source_id);
				if (srcPos) {
					for (const targetId of spreadData.target_ids) {
						const tgtPos = nodePositions.get(targetId);
						if (tgtPos) {
							effects.createConnectionFlash(srcPos, tgtPos, new THREE.Color(0x14e8c6));
						}
					}
				}
			}
			break;
		}

		// v2.0.5 Active Forgetting — Anderson 2025 SIF + Davis Rac1
		// These events fire when the `suppress` MCP tool is called and when
		// the background Rac1 cascade worker sweeps recently-suppressed seeds.
		// Before these handlers landed, suppression fired silently in the graph.

		case 'MemorySuppressed': {
			const data = event.data as {
				id?: string;
				suppression_count?: number;
			};
			if (!data.id) break;
			const pos = nodePositions.get(data.id);
			if (pos) {
				// Violet implosion: top-down inhibitory control collapsing into the node
				effects.createImplosion(pos, new THREE.Color(0xa855f7));
				// Plus a slow violet pulse so the suppressed node is visually marked
				// even after the implosion frames finish. Strength scales with
				// compounding suppression_count (more hits = stronger pulse).
				const count = Math.max(1, data.suppression_count ?? 1);
				const strength = Math.min(0.4 + count * 0.15, 1.0);
				effects.addPulse(data.id, strength, new THREE.Color(0xa855f7), 0.04);
			}
			break;
		}

		case 'MemoryUnsuppressed': {
			const data = event.data as { id?: string; remaining_count?: number };
			if (!data.id) break;
			const pos = nodePositions.get(data.id);
			if (pos && nodeMeshMap.has(data.id)) {
				// Reversal within the 24h labile window — bring the memory back.
				// Rainbow spawn burst celebrates the reversal, then a green pulse
				// to mark the node as "active again" (paralleling MemoryPromoted).
				effects.createRainbowBurst(pos, new THREE.Color(0x00ff88));
				effects.addPulse(data.id, 1.0, new THREE.Color(0x00ff88), 0.02);
			}
			break;
		}

		case 'Rac1CascadeSwept': {
			// Rac1 cascade runs as a background sweep. The event carries counts,
			// not specific node IDs, so we visualize it as a subtle violet wave
			// rippling through random sampled neighbors to indicate "decay is
			// spreading through co-activated memories." Future v2.1 events may
			// carry the actual affected IDs; this handler can be tightened then.
			const data = event.data as {
				seeds?: number;
				neighbors_affected?: number;
			};
			const affected = data.neighbors_affected ?? 0;
			if (affected === 0) break;
			const allIds = Array.from(nodeMeshMap.keys());
			const sampleSize = Math.min(affected, allIds.length, 12);
			for (let i = 0; i < sampleSize; i++) {
				const idx = Math.floor(Math.random() * allIds.length);
				const targetId = allIds[idx];
				effects.addPulse(targetId, 0.5, new THREE.Color(0xa855f7), 0.035);
			}
			break;
		}

		// v2.0.6: wire three previously-silent core events. Before this, the
		// live feed showed Connected / ConsolidationStarted / ImportanceScored
		// firing but the 3D graph stayed motionless — users perceived the
		// dashboard as unresponsive during real cognitive work.

		case 'Connected': {
			// WebSocket handshake just completed. A gentle ripple from the
			// first node signals "link is live" without dominating the scene.
			const firstId = nodeMeshMap.keys().next().value;
			if (!firstId) break;
			const pos = nodePositions.get(firstId);
			if (pos) {
				effects.createRippleWave(pos);
			}
			break;
		}

		case 'ConsolidationStarted': {
			// FSRS-6 consolidation cycle starting. Amber pulses across a
			// random sample signal "retention scores are recomputing across
			// the graph." Intentionally subtle — consolidation runs for
			// seconds, so the visual shouldn't demand attention the whole
			// time. Colour matches the ConsolidationStarted feed entry.
			const allIds = Array.from(nodeMeshMap.keys());
			const sampleSize = Math.min(allIds.length, 20);
			for (let i = 0; i < sampleSize; i++) {
				const idx = Math.floor(Math.random() * allIds.length);
				const targetId = allIds[idx];
				effects.addPulse(targetId, 0.45, new THREE.Color(0xffb800), 0.025);
			}
			break;
		}

		case 'ImportanceScored': {
			// A memory just had its 4-channel importance score recomputed
			// (novelty + arousal + reward + attention). Magenta pulse on the
			// scored node with strength proportional to composite score so
			// users can visually rank importance across the graph.
			const data = event.data as {
				id?: string;
				composite_score?: number;
			};
			if (!data.id) break;
			const pos = nodePositions.get(data.id);
			if (pos && nodeMeshMap.has(data.id)) {
				const score = Math.max(0, Math.min(1, data.composite_score ?? 0.5));
				const strength = 0.3 + score * 0.7;
				effects.addPulse(data.id, strength, new THREE.Color(0xff3cac), 0.03);
			}
			break;
		}
	}
}

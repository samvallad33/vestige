<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { GraphNode, GraphEdge, VestigeEvent } from '$types';
	import { createScene, resizeScene, disposeScene, type SceneContext } from '$lib/graph/scene';
	import { ForceSimulation } from '$lib/graph/force-sim';
	import { NodeManager, type ColorMode } from '$lib/graph/nodes';
	import { EdgeManager } from '$lib/graph/edges';
	import { ParticleSystem } from '$lib/graph/particles';
	import { EffectManager } from '$lib/graph/effects';
	import { DreamMode } from '$lib/graph/dream-mode';
	import { mapEventToEffects, type GraphMutationContext, type GraphMutation } from '$lib/graph/events';
	import { createNebulaBackground, updateNebula } from '$lib/graph/shaders/nebula.frag';
	import { createPostProcessing, updatePostProcessing, type PostProcessingStack } from '$lib/graph/shaders/post-processing';
	import { graphState } from '$lib/stores/graph-state.svelte';
	import type * as THREE from 'three';

	interface Props {
		nodes: GraphNode[];
		edges: GraphEdge[];
		centerId: string;
		events?: VestigeEvent[];
		isDreaming?: boolean;
		/// v2.0.8: colour mode for node spheres. "type" tints by node type
		/// (fact/concept/event/…); "state" tints by FSRS accessibility bucket
		/// (active/dormant/silent/unavailable). Toggled live from the graph page.
		colorMode?: ColorMode;
		onSelect?: (nodeId: string) => void;
		onGraphMutation?: (mutation: GraphMutation) => void;
	}

	let {
		nodes,
		edges,
		centerId,
		events = [],
		isDreaming = false,
		colorMode = 'type',
		onSelect,
		onGraphMutation,
	}: Props = $props();

	// Re-tint every live node whenever the color mode flips. The NodeManager's
	// setColorMode is idempotent and mutates materials in place, so this
	// effect runs once per toggle and doesn't rebuild the scene.
	$effect(() => {
		nodeManager?.setColorMode(colorMode);
	});

	let container: HTMLDivElement;
	let ctx: SceneContext;
	let animationId: number;

	// Modules
	let nodeManager: NodeManager;
	let edgeManager: EdgeManager;
	let particles: ParticleSystem;
	let effects: EffectManager;
	let forceSim: ForceSimulation;
	let dreamMode: DreamMode;
	let nebulaMaterial: THREE.ShaderMaterial;
	let postStack: PostProcessingStack;

	// Event tracking — we track the last-processed event by reference identity
	// rather than by count, because the WebSocket store PREPENDS new events
	// at index 0 and CAPS the array at MAX_EVENTS, so a numeric high-water
	// mark would drift out of alignment (and did for ~3 versions — v2.3
	// demo uncovered this while trying to fire multiple MemoryCreated events
	// in sequence).
	let lastProcessedEvent: VestigeEvent | null = null;

	// Internal tracking: initial nodes + live-added nodes
	let allNodes: GraphNode[] = [];

	onMount(() => {
		ctx = createScene(container);

		// Nebula background
		const nebula = createNebulaBackground(ctx.scene);
		nebulaMaterial = nebula.material;

		// Post-processing (added after bloom)
		postStack = createPostProcessing(ctx.composer);

		// Modules
		particles = new ParticleSystem(ctx.scene);
		nodeManager = new NodeManager();
		// Apply the initial colour mode before node creation so the first paint
		// already reflects the user's prop choice. Prevents a visible flash from
		// type-colour to state-colour on mount when the page defaults to state.
		nodeManager.colorMode = colorMode;
		edgeManager = new EdgeManager();
		effects = new EffectManager(ctx.scene);
		dreamMode = new DreamMode();

		// Build graph
		const positions = nodeManager.createNodes(nodes);
		edgeManager.createEdges(edges, positions);
		forceSim = new ForceSimulation(positions);

		// Track all nodes (initial set)
		allNodes = [...nodes];

		ctx.scene.add(edgeManager.group);
		ctx.scene.add(nodeManager.group);

		animate();

		window.addEventListener('resize', onResize);
		container.addEventListener('pointermove', onPointerMove);
		container.addEventListener('click', onClick);
	});

	onDestroy(() => {
		cancelAnimationFrame(animationId);
		window.removeEventListener('resize', onResize);
		container?.removeEventListener('pointermove', onPointerMove);
		container?.removeEventListener('click', onClick);
		effects?.dispose();
		particles?.dispose();
		nodeManager?.dispose();
		edgeManager?.dispose();
		if (ctx) disposeScene(ctx);
	});

	// 120Hz Governor. All physics and effect counters are frame-based
	// (orb.age++, forceSim.tick, materialization frames). On a ProMotion
	// display the browser drives rAF at 120 FPS, which would double-speed
	// every ritual. Clamping to ~60 FPS keeps the visual timing identical
	// across displays without rewriting every counter to use delta time.
	// The `- (dt % 16)` carry avoids long-term drift.
	let govLastTime = 0;

	function animate() {
		animationId = requestAnimationFrame(animate);
		const now = performance.now();
		if (govLastTime === 0) govLastTime = now;
		const dt = now - govLastTime;
		if (dt < 16) return;
		govLastTime = now - (dt % 16);

		const time = now * 0.001;

		// Force simulation
		forceSim.tick(edges);

		// Update positions
		nodeManager.updatePositions();
		edgeManager.updatePositions(nodeManager.positions);

		// Animate edge growth/dissolution
		edgeManager.animateEdges(nodeManager.positions);

		// Animate
		particles.animate(time);
		nodeManager.animate(time, allNodes, ctx.camera, graphState.brightness);

		// Dream mode
		dreamMode.setActive(isDreaming);
		dreamMode.update(ctx.scene, ctx.bloomPass, ctx.controls, ctx.lights, time);

		// Nebula + post-processing
		updateNebula(
			nebulaMaterial,
			time,
			dreamMode.current.nebulaIntensity,
			container.clientWidth,
			container.clientHeight
		);
		updatePostProcessing(postStack, time, dreamMode.current.nebulaIntensity);

		// Events + effects
		processEvents();
		effects.update(nodeManager.meshMap, ctx.camera, nodeManager.positions);

		ctx.controls.update();
		ctx.composer.render();
	}

	function processEvents() {
		if (!events || events.length === 0) return;

		// Walk the feed from newest (index 0) backward until we hit the last
		// event we already processed. Everything between is fresh. This is
		// robust against both (a) prepend ordering and (b) the MAX_EVENTS cap
		// dropping old entries off the tail.
		const fresh: VestigeEvent[] = [];
		for (const e of events) {
			if (e === lastProcessedEvent) break;
			fresh.push(e);
		}
		if (fresh.length === 0) return;

		// Event Horizon Guard. If the last-processed reference fell off the
		// end of the capped array (burst of >MAX_EVENTS events in one tick),
		// the walk above consumed the ENTIRE buffer — we'd try to animate
		// 200 simultaneous births and melt the GPU. Detect the overflow and
		// drop this batch on the floor; state is already current via
		// lastProcessedEvent pointing forward.
		if (fresh.length === events.length && events.length >= 200) {
			// eslint-disable-next-line no-console
			console.warn('[vestige] Event horizon overflow: dropping visuals for', fresh.length, 'events');
			lastProcessedEvent = events[0];
			return;
		}

		lastProcessedEvent = events[0];

		const mutationCtx: GraphMutationContext = {
			effects,
			nodeManager,
			edgeManager,
			forceSim,
			camera: ctx.camera,
			onMutation: (mutation: GraphMutation) => {
				// Update internal allNodes tracking
				if (mutation.type === 'nodeAdded') {
					allNodes = [...allNodes, mutation.node];
				} else if (mutation.type === 'nodeRemoved') {
					allNodes = allNodes.filter((n) => n.id !== mutation.nodeId);
				}
				// Notify parent
				onGraphMutation?.(mutation);
			},
		};

		// Process oldest-first so cause precedes effect (e.g. MemoryCreated
		// fires before a ConnectionDiscovered that references the new node).
		// `fresh` is newest-first from the walk above, so iterate reversed.
		for (let i = fresh.length - 1; i >= 0; i--) {
			mapEventToEffects(fresh[i], mutationCtx, allNodes);
		}
	}

	function onResize() {
		if (!container || !ctx) return;
		resizeScene(ctx, container);
	}

	function onPointerMove(event: PointerEvent) {
		const rect = container.getBoundingClientRect();
		ctx.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
		ctx.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

		ctx.raycaster.setFromCamera(ctx.mouse, ctx.camera);
		const intersects = ctx.raycaster.intersectObjects(nodeManager.getMeshes());

		if (intersects.length > 0) {
			nodeManager.hoveredNode = intersects[0].object.userData.nodeId;
			container.style.cursor = 'pointer';
		} else {
			nodeManager.hoveredNode = null;
			container.style.cursor = 'grab';
		}
	}

	function onClick() {
		if (nodeManager.hoveredNode) {
			nodeManager.selectedNode = nodeManager.hoveredNode;
			onSelect?.(nodeManager.hoveredNode);

			const pos = nodeManager.positions.get(nodeManager.hoveredNode);
			if (pos) {
				ctx.controls.target.lerp(pos.clone(), 0.5);
			}
		}
	}
</script>

<div bind:this={container} class="w-full h-full"></div>

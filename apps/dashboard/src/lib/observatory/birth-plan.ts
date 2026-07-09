/**
 * Cognitive Observatory â deterministic birth-plan CPU helpers (Moment B, Task B1).
 *
 * Pure CPU: pick a birth target, precompute deterministic `BirthParticle` initial
 * arrays, build birth beat metadata. No GPU code, no Math.random().
 *
 * Particle layout (16 floats / 64 bytes per particle):
 *   start_life  : xyz start position, w seed/life scalar (phase offset)
 *   target_size : xyz target position, w base size (1.0 + rng * 1.8)
 *   color_phase : rgb target/base node color, w phase offset
 *   state       : xyz current position (shader computes), w alpha
 *
 * All start positions form a deterministic hollow shell around the target:
 *   70% spherical shell (radius 110-180)
 *   20% tendrils along incident edge directions
 *   10% near-camera dust plane
 */

import { DemoClock, deterministicSpherePosition } from './demo-clock';
import type { ObservatoryGraph } from './types';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface TimelineBeat {
  label: string;
  startFrame: number;
  endFrame: number;
}

export interface BirthPlan {
  targetIndex: number;
  targetNodeId: string;
  /** 16 floats per particle (64 bytes). */
  particles: Float32Array;
  /** 4 u32 per edge step (source, target, beatFrame, kind). */
  edgeSteps: Uint32Array;
  timeline: TimelineBeat[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FLOATS_PER_BIRTH_PARTICLE = 16;
const UINTS_PER_BIRTH_EDGE_STEP = 4;

// Shell radii
const SHELL_MIN_RADIUS = 110;
const SHELL_MAX_RADIUS = 180;

// Particle distribution fractions
const FRACTION_SHELL = 0.70;
const FRACTION_TENDRIL = 0.20;
const FRACTION_DUST = 0.10;

// Edge engraving: each incident edge gets a pulse starting at frame 360
const ENGRAVE_START_FRAME = 360;
const ENGRAVE_INTERVAL = 18; // frames between successive edge pulses

// ---------------------------------------------------------------------------
// Target selection
// ---------------------------------------------------------------------------

/**
 * Pick the birth target deterministically from the graph.
 *
 * Priority:
 * 1. Center node's highest-retention neighbor (if graph has edges).
 * 2. First non-center node after stable graph ordering.
 * 3. Center node.
 */
export function pickTargetIndex(graph: ObservatoryGraph): number {
  // 1. Prefer center's highest-retention neighbor
  if (graph.edges.length > 0) {
    const centerIdx = graph.centerIndex;
    const incidentEdges = graph.edges.filter(
      (e) => e.sourceIndex === centerIdx || e.targetIndex === centerIdx
    );
    if (incidentEdges.length > 0) {
      let bestNeighborIdx = -1;
      let bestRetention = -1;
      for (const edge of incidentEdges) {
        const neighborIdx =
          edge.sourceIndex === centerIdx ? edge.targetIndex : edge.sourceIndex;
        const neighbor = graph.nodes[neighborIdx];
        if (neighbor && neighbor.retention > bestRetention) {
          bestRetention = neighbor.retention;
          bestNeighborIdx = neighborIdx;
        }
      }
      if (bestNeighborIdx >= 0) return bestNeighborIdx;
    }
  }

  // 2. First non-center node after stable ordering
  for (let i = 0; i < graph.nodes.length; i++) {
    if (i !== graph.centerIndex) return i;
  }

  // 3. Center node (fallback)
  return graph.centerIndex;
}

// ---------------------------------------------------------------------------
// Particle precomputation
// ---------------------------------------------------------------------------

/**
 * Build the deterministic particle array for a birth event.
 *
 * Uses a fresh DemoClock seeded with `seed + ':birth:' + targetNodeId` so
 * the same graph + seed always produces the same layout.
 */
export function buildBirthPlan(
  graph: ObservatoryGraph,
  seed: string,
  particleCount = 8192
): BirthPlan {
  const targetIndex = pickTargetIndex(graph);
  const targetNode = graph.nodes[targetIndex];
  const targetNodeId = targetNode.id;

  // Get target node position from the graph (will be in the node state buffer)
  const targetPos = getNodePosition(graph, targetIndex);

  // Build a fresh DemoClock for deterministic particle placement
  const clock = new DemoClock({ seed: seed + ':birth:' + targetNodeId });
  const rng = clock.state.rng;

  // Build particles
  const particles = new Float32Array(particleCount * FLOATS_PER_BIRTH_PARTICLE);

  const shellCount = Math.floor(particleCount * FRACTION_SHELL);
  const tendrilCount = Math.floor(particleCount * FRACTION_TENDRIL);
  const dustCount = particleCount - shellCount - tendrilCount;

  // --- 70%: spherical shell around target ---
  for (let i = 0; i < shellCount; i++) {
    const base = i * FLOATS_PER_BIRTH_PARTICLE;

    // Deterministic position on a sphere around the target
    const [sx, sy, sz] = deterministicSpherePosition(
      i,
      shellCount,
      SHELL_MIN_RADIUS + rng() * (SHELL_MAX_RADIUS - SHELL_MIN_RADIUS),
      rng
    );

    // World-space start position = target + shell offset
    particles[base + 0] = targetPos[0] + sx;
    particles[base + 1] = targetPos[1] + sy;
    particles[base + 2] = targetPos[2] + sz;
    // w: phase offset (stagger)
    particles[base + 3] = rng();

    // Target position (same for all particles â convergence target)
    particles[base + 4] = targetPos[0];
    particles[base + 5] = targetPos[1];
    particles[base + 6] = targetPos[2];
    // w: base size
    particles[base + 7] = 1.0 + rng() * 1.8;

    // Color: violet dust base (0.55, 0.32, 1.00)
    particles[base + 8] = 0.55;
    particles[base + 9] = 0.32;
    particles[base + 10] = 1.00;
    // w: phase offset for spectral rim
    particles[base + 11] = rng();

    // state: zeroed (shader computes current position)
    particles[base + 12] = 0;
    particles[base + 13] = 0;
    particles[base + 14] = 0;
    particles[base + 15] = 0;
  }

  // --- 20%: tendrils along incident edge directions ---
  const incidentEdges = graph.edges.filter(
    (e) => e.sourceIndex === targetIndex || e.targetIndex === targetIndex
  );

  for (let i = 0; i < tendrilCount; i++) {
    const base = (shellCount + i) * FLOATS_PER_BIRTH_PARTICLE;

    // Skip tendrils when no incident edges (center-only graph)
    if (incidentEdges.length === 0) {
      // Just place as shell particles (already done above)
      continue;
    }

    // Pick an incident edge direction (cycle through edges)
    const edgeIdx = i % incidentEdges.length;
    const edge = incidentEdges[edgeIdx];

    // Direction from target to neighbor
    const neighborIdx =
      edge.sourceIndex === targetIndex ? edge.targetIndex : edge.sourceIndex;
    const neighborPos = getNodePosition(graph, neighborIdx);
    const dx = neighborPos[0] - targetPos[0];
    const dy = neighborPos[1] - targetPos[1];
    const dz = neighborPos[2] - targetPos[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

    // Place along the edge direction, spread out
    const t = (i / Math.max(1, tendrilCount)) * 2.0 + 0.5; // 0.5 to 2.5
    const spread = rng() * 30; // perpendicular spread

    // Perpendicular offset (simple cross with a fixed axis)
    const px = -dy * spread / (len || 1);
    const py = dx * spread / (len || 1);
    const pz = 0;

    particles[base + 0] = targetPos[0] + (dx / len) * t * 80 + px;
    particles[base + 1] = targetPos[1] + (dy / len) * t * 80 + py;
    particles[base + 2] = targetPos[2] + (dz / len) * t * 80 + pz;
    particles[base + 3] = rng();

    particles[base + 4] = targetPos[0];
    particles[base + 5] = targetPos[1];
    particles[base + 6] = targetPos[2];
    particles[base + 7] = 1.0 + rng() * 1.8;

    particles[base + 8] = 0.55;
    particles[base + 9] = 0.32;
    particles[base + 10] = 1.00;
    particles[base + 11] = rng();

    particles[base + 12] = 0;
    particles[base + 13] = 0;
    particles[base + 14] = 0;
    particles[base + 15] = 0;
  }

  // --- 10%: near-camera dust plane for depth sparkle ---
  // Camera orbits around the field; a "near camera" plane is roughly at
  // orbit distance. We place these in front of the target along the
  // camera's approximate view direction (z-axis in our coordinate system).
  const ORBIT_DISTANCE = 300;
  for (let i = 0; i < dustCount; i++) {
    const base = (shellCount + tendrilCount + i) * FLOATS_PER_BIRTH_PARTICLE;

    // Spread in a plane near the camera orbit distance
    const angle = rng() * Math.PI * 2;
    const spread = rng() * 120;

    particles[base + 0] = targetPos[0] + Math.cos(angle) * spread;
    particles[base + 1] = targetPos[1] + Math.sin(angle) * spread;
    particles[base + 2] = targetPos[2] + ORBIT_DISTANCE * 0.6 + rng() * 40;
    particles[base + 3] = rng();

    particles[base + 4] = targetPos[0];
    particles[base + 5] = targetPos[1];
    particles[base + 6] = targetPos[2];
    particles[base + 7] = 1.0 + rng() * 1.8;

    particles[base + 8] = 0.55;
    particles[base + 9] = 0.32;
    particles[base + 10] = 1.00;
    particles[base + 11] = rng();

    particles[base + 12] = 0;
    particles[base + 13] = 0;
    particles[base + 14] = 0;
    particles[base + 15] = 0;
  }

  // ---------------------------------------------------------------------------
  // Edge steps for engraving
  // ---------------------------------------------------------------------------

  const edgeSteps = buildEdgeSteps(graph, targetIndex);

  // ---------------------------------------------------------------------------
  // Timeline beats
  // ---------------------------------------------------------------------------

  const timeline = buildTimeline();

  return {
    targetIndex,
    targetNodeId,
    particles,
    edgeSteps,
    timeline,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Get the world-space position of a node from the graph.
 * Uses deterministic sphere placement (same as graph-upload).
 */
function getNodePosition(
  graph: ObservatoryGraph,
  nodeIndex: number
): [number, number, number] {
  const node = graph.nodes[nodeIndex];
  const n = graph.nodes.length;

  if (node.isCenter && graph.centerIndex === nodeIndex) {
    return [0, 0, 0];
  }

  // Use a fixed seed for position (not the birth seed) so positions match
  // what the NodeRenderer will produce. We use a deterministic "position seed"
  // derived from the node's index and the graph's center.
  // The actual NodeRenderer uses the demo seed for perturbation, but for
  // the birth plan we need to know where the target node will be.
  // We use a simple golden-angle placement with a fixed perturbation seed.
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (nodeIndex / (n - 1 || 1)) * 2;
  const radiusAtY = Math.sqrt(1 - y * y);
  const theta = goldenAngle * nodeIndex;
  const fieldRadius = 120;

  // Fixed perturbation (no random â deterministic by index)
  const px = ((nodeIndex * 7 + 3) % 100) / 100 * 0.1 * fieldRadius - 0.05 * fieldRadius;
  const py = ((nodeIndex * 13 + 7) % 100) / 100 * 0.1 * fieldRadius - 0.05 * fieldRadius;
  const pz = ((nodeIndex * 17 + 11) % 100) / 100 * 0.1 * fieldRadius - 0.05 * fieldRadius;

  return [
    Math.cos(theta) * radiusAtY * fieldRadius + px,
    y * fieldRadius + py,
    Math.sin(theta) * radiusAtY * fieldRadius + pz,
  ];
}

/**
 * Build edge steps for the engraving phase (frames 360+).
 * Each incident edge from the target gets a pulse.
 */
function buildEdgeSteps(
  graph: ObservatoryGraph,
  targetIndex: number
): Uint32Array {
  const incidentEdges = graph.edges.filter(
    (e) => e.sourceIndex === targetIndex || e.targetIndex === targetIndex
  );

  const stepCount = incidentEdges.length;
  // No incident edges → zero steps (plan L464): BirthRenderer skips the
  // engrave buffer entirely rather than drawing a phantom 0→0 pulse.
  if (stepCount === 0) {
    return new Uint32Array(0);
  }

  const data = new Uint32Array(stepCount * UINTS_PER_BIRTH_EDGE_STEP);

  for (let k = 0; k < stepCount; k++) {
    const edge = incidentEdges[k];
    const neighborIdx =
      edge.sourceIndex === targetIndex ? edge.targetIndex : edge.sourceIndex;
    const beatFrame = ENGRAVE_START_FRAME + k * ENGRAVE_INTERVAL;

    data[k * UINTS_PER_BIRTH_EDGE_STEP + 0] = targetIndex; // source = target
    data[k * UINTS_PER_BIRTH_EDGE_STEP + 1] = neighborIdx; // target = neighbor
    data[k * UINTS_PER_BIRTH_EDGE_STEP + 2] = beatFrame;
    data[k * UINTS_PER_BIRTH_EDGE_STEP + 3] = 0; // kind: normal
  }

  return data;
}

/**
 * Build the timeline beats for the 720-frame birth loop.
 * Matches the choreography schedule from the plan.
 */
function buildTimeline(): TimelineBeat[] {
  return [
    { label: 'latent trace condensing', startFrame: 60, endFrame: 239 },
    { label: 'engram coalescence', startFrame: 240, endFrame: 329 },
    { label: 'memory ignition', startFrame: 330, endFrame: 359 },
    { label: 'associations engrave', startFrame: 360, endFrame: 509 },
    { label: 'stabilization', startFrame: 510, endFrame: 659 },
  ];
}

/**
 * Memory Cinema 3D Causal DAG Time-Travel Scrubber & Particle Impulse Engine (#833).
 *
 * Pure, deterministic 3D geometry & state projection layer driving WebGL/Three.js
 * causal DAG time-travel scrubbing (`seq` / `timestampMs`), supersession dimming,
 * and Catmull-Rom spline particle impulse trajectories along recorded Strata edges.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface CinemaDagNode {
  id: string;
  seq: number;
  timestampMs: number;
  position: Vec3;
  supersededBySeq?: number;
  quarantined?: boolean;
}

export interface CinemaDagEdge {
  id: string;
  from: string;
  to: string;
  seq: number;
  kind: "causes" | "fixes" | "supersedes" | "cites" | "touched";
}

export interface ScrubbedNodeState {
  id: string;
  visible: boolean;
  opacity: number;
  birthPulse: number;
}

/**
 * Compute deterministic node visibility, supersession dimming, and birth shockwave
 * pulse intensity at an arbitrary Strata `scrubSeq` cursor (#833).
 */
export function evaluateCinemaAtSeq(
  nodes: readonly CinemaDagNode[],
  scrubSeq: number,
  pulseWindowSeqs = 3
): ScrubbedNodeState[] {
  return nodes.map((node) => {
    if (node.seq > scrubSeq) {
      return { id: node.id, visible: false, opacity: 0, birthPulse: 0 };
    }
    const ageSeq = scrubSeq - node.seq;
    const birthPulse =
      ageSeq <= pulseWindowSeqs ? Math.max(0, 1 - ageSeq / (pulseWindowSeqs + 1)) : 0;
    const isSuperseded =
      node.supersededBySeq !== undefined && node.supersededBySeq <= scrubSeq;
    const baseOpacity = node.quarantined ? 0.35 : isSuperseded ? 0.25 : 1.0;
    return {
      id: node.id,
      visible: true,
      opacity: baseOpacity,
      birthPulse,
    };
  });
}

/**
 * Sample a smooth 3D arched particle trajectory along a directed causal edge
 * for `t` in `[0, 1]` with deterministic arc lift proportional to chord length.
 */
export function sampleCausalParticleArc(start: Vec3, end: Vec3, t: number): Vec3 {
  const clamped = Math.max(0, Math.min(1, t));
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const dz = end.z - start.z;
  const chord = Math.hypot(dx, dy, dz);
  const arcLift = Math.sin(clamped * Math.PI) * (chord * 0.18);
  return {
    x: start.x + dx * clamped,
    y: start.y + dy * clamped + arcLift,
    z: start.z + dz * clamped,
  };
}

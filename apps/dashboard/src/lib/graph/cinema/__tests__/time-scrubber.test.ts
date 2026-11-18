import { describe, expect, it } from "vitest";
import { evaluateCinemaAtSeq, sampleCausalParticleArc } from "../time-scrubber";

describe("Memory Cinema 3D Time-Travel Scrubber & Particle Arc (#833)", () => {
  it("hides future nodes, dims superseded nodes, and pulses newly born nodes", () => {
    const nodes = [
      { id: "m1", seq: 1, timestampMs: 1000, position: { x: 0, y: 0, z: 0 }, supersededBySeq: 5 },
      { id: "m2", seq: 5, timestampMs: 5000, position: { x: 10, y: 0, z: 0 } },
      { id: "m3", seq: 9, timestampMs: 9000, position: { x: 20, y: 0, z: 0 } },
    ];
    const at5 = evaluateCinemaAtSeq(nodes, 5);
    expect(at5[0].visible).toBe(true);
    expect(at5[0].opacity).toBe(0.25); // superseded at seq 5
    expect(at5[1].visible).toBe(true);
    expect(at5[1].birthPulse).toBe(1); // born right at seq 5
    expect(at5[2].visible).toBe(false); // future node hidden
  });

  it("samples smooth 3D arched particle trajectories along causal edges", () => {
    const start = { x: 0, y: 0, z: 0 };
    const end = { x: 10, y: 0, z: 0 };
    const mid = sampleCausalParticleArc(start, end, 0.5);
    expect(mid.x).toBeCloseTo(5.0, 5);
    expect(mid.y).toBeGreaterThan(1.5);
    expect(sampleCausalParticleArc(start, end, 1.0)).toEqual(end);
  });
});

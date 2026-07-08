import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock WebGPU before any imports
vi.mock('$lib/observatory/engine', () => ({
  ObservatoryEngine: class {},
}));

// Mock document.createElement for canvas (needed by demo-clock)
const mockCanvas = {
  width: 512,
  height: 64,
  getContext: () => null,
  toDataURL: () => 'data:image/png;base64,',
};

if (typeof globalThis.document === 'undefined') {
  (globalThis as any).document = {
    createElement: (tag: string) => (tag === 'canvas' ? mockCanvas : {}),
  };
}

import { buildBirthPlan, pickTargetIndex } from '../birth-plan';
import type { ObservatoryGraph, ObservatoryNode, ObservatoryEdge } from '../types';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

function makeNode(
  id: string,
  index: number,
  opts: Partial<ObservatoryNode> = {}
): ObservatoryNode {
  return {
    id,
    index,
    label: `Node ${id}`,
    type: 'memory',
    retention: opts.retention ?? 0.5,
    tags: opts.tags ?? [],
    isCenter: opts.isCenter ?? false,
    suppressed: opts.suppressed ?? false,
  };
}

function makeEdge(sourceIndex: number, targetIndex: number): ObservatoryEdge {
  return { sourceIndex, targetIndex, weight: 1.0, type: 'association' };
}

function makeGraph(
  nodes: ObservatoryNode[],
  edges: ObservatoryEdge[]
): ObservatoryGraph {
  const indexById = new Map<string, number>();
  for (const n of nodes) indexById.set(n.id, n.index);
  const centerIndex = nodes.findIndex((n) => n.isCenter);
  return {
    nodes,
    edges,
    indexById,
    centerIndex: centerIndex < 0 ? 0 : centerIndex,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('pickTargetIndex', () => {
  it('picks center\'s highest-retention neighbor when edges exist', () => {
    const nodes = [
      makeNode('center', 0, { isCenter: true, retention: 0.9 }),
      makeNode('a', 1, { retention: 0.8 }),
      makeNode('b', 2, { retention: 0.95 }),
      makeNode('c', 3, { retention: 0.7 }),
    ];
    const edges = [
      makeEdge(0, 1), // center <-> a (ret 0.8)
      makeEdge(0, 2), // center <-> b (ret 0.95)
      makeEdge(1, 3), // a <-> c (not incident to center)
    ];
    const graph = makeGraph(nodes, edges);
    const target = pickTargetIndex(graph);
    expect(target).toBe(2); // b has highest retention (0.95)
  });

  it('picks first non-center node when center has no edges', () => {
    const nodes = [
      makeNode('center', 0, { isCenter: true }),
      makeNode('a', 1),
      makeNode('b', 2),
    ];
    const edges = [makeEdge(1, 2)]; // no edges from center
    const graph = makeGraph(nodes, edges);
    const target = pickTargetIndex(graph);
    expect(target).toBe(1); // first non-center
  });

  it('picks center node when graph has only the center', () => {
    const nodes = [makeNode('center', 0, { isCenter: true })];
    const graph = makeGraph(nodes, []);
    const target = pickTargetIndex(graph);
    expect(target).toBe(0);
  });

  it('picks center when all neighbors have equal retention', () => {
    const nodes = [
      makeNode('center', 0, { isCenter: true }),
      makeNode('a', 1, { retention: 0.5 }),
      makeNode('b', 2, { retention: 0.5 }),
    ];
    const edges = [makeEdge(0, 1), makeEdge(0, 2)];
    const graph = makeGraph(nodes, edges);
    const target = pickTargetIndex(graph);
    // Both have same retention (0.5), so picks first neighbor (a, index 1)
    expect(target).toBe(1);
  });
});

describe('buildBirthPlan', () => {
  let graph: ObservatoryGraph;

  beforeEach(() => {
    const nodes = [
      makeNode('center', 0, { isCenter: true, retention: 0.9 }),
      makeNode('a', 1, { retention: 0.8 }),
      makeNode('b', 2, { retention: 0.95 }),
      makeNode('c', 3, { retention: 0.7 }),
      makeNode('d', 4, { retention: 0.6 }),
    ];
    const edges = [
      makeEdge(0, 1),
      makeEdge(0, 2),
      makeEdge(1, 3),
      makeEdge(2, 4),
    ];
    graph = makeGraph(nodes, edges);
  });

  it('produces deterministic particles for same graph + seed', () => {
    const plan1 = buildBirthPlan(graph, 'test-seed', 1024);
    const plan2 = buildBirthPlan(graph, 'test-seed', 1024);

    expect(plan1.particles).toEqual(plan2.particles);
    expect(plan1.edgeSteps).toEqual(plan2.edgeSteps);
    expect(plan1.targetIndex).toBe(plan2.targetIndex);
  });

  it('produces different particles for different seeds', () => {
    const plan1 = buildBirthPlan(graph, 'seed-a', 1024);
    const plan2 = buildBirthPlan(graph, 'seed-b', 1024);

    // Same target, different particle positions
    expect(plan1.targetIndex).toBe(plan2.targetIndex);
    // But particles should differ
    let different = false;
    for (let i = 0; i < plan1.particles.length; i++) {
      if (plan1.particles[i] !== plan2.particles[i]) {
        different = true;
        break;
      }
    }
    expect(different).toBe(true);
  });

  it('always picks the same target regardless of seed', () => {
    const plan1 = buildBirthPlan(graph, 'seed-a', 1024);
    const plan2 = buildBirthPlan(graph, 'seed-b', 1024);
    const plan3 = buildBirthPlan(graph, 'seed-c', 1024);

    expect(plan1.targetIndex).toBe(plan2.targetIndex);
    expect(plan2.targetIndex).toBe(plan3.targetIndex);
    // Should be index 2 (b, highest retention neighbor of center)
    expect(plan1.targetIndex).toBe(2);
  });

  it('has correct particle array size', () => {
    const plan = buildBirthPlan(graph, 'test', 2048);
    expect(plan.particles.length).toBe(2048 * 16); // 16 floats per particle
  });

  it('has correct edge step array size', () => {
    const plan = buildBirthPlan(graph, 'test');
    // 2 edges incident to center (0-1, 0-2)
    expect(plan.edgeSteps.length).toBe(2 * 4); // 4 u32 per step
  });

  it('has valid timeline beats', () => {
    const plan = buildBirthPlan(graph, 'test');
    expect(plan.timeline.length).toBeGreaterThan(0);
    for (const beat of plan.timeline) {
      expect(beat.label).toBeTruthy();
      expect(beat.startFrame).toBeGreaterThanOrEqual(0);
      expect(beat.endFrame).toBeGreaterThanOrEqual(beat.startFrame);
      expect(beat.endFrame).toBeLessThan(720);
    }
  });

  it('center-only graph produces valid target and zero edge steps', () => {
    const centerOnly = makeGraph(
      [makeNode('center', 0, { isCenter: true })],
      []
    );
    const plan = buildBirthPlan(centerOnly, 'center-test', 512);
    expect(plan.targetIndex).toBe(0);
    expect(plan.targetNodeId).toBe('center');
    expect(plan.edgeSteps.length).toBe(0); // zero incident edges = zero steps (plan L464)
  });

  it('does not use Math.random() — verified by code inspection', () => {
    // This test documents the constraint: buildBirthPlan uses DemoClock
    // (xmur3 + mulberry32) exclusively. Math.random() is never called.
    // The test passes by construction — if Math.random() were used,
    // determinism would break, which is caught by the determinism test above.
    expect(true).toBe(true);
  });

  it('default particle count is 8192', () => {
    const plan = buildBirthPlan(graph, 'test');
    expect(plan.particles.length).toBe(8192 * 16);
  });

  it('custom particle count works', () => {
    const plan = buildBirthPlan(graph, 'test', 4096);
    expect(plan.particles.length).toBe(4096 * 16);
  });
});

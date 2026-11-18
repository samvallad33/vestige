import assert from "node:assert/strict";
import test from "node:test";
import { VestigeClient } from "../index.js";
import { createVestigeTools } from "../adapters/vercel-ai.js";
import { VestigeLettaAdapter } from "../adapters/letta.js";

function createFakeTransport() {
  const store = [];
  const suppressed = new Set();
  return async (tool, args) => {
    if (tool === "remember") {
      const id = `mem-${String(store.length + 1).padStart(16, "0")}`;
      const rec = {
        id,
        nodeId: id,
        receiptId: `eff-${String(store.length + 1).padStart(16, "0")}`,
        canonicalHash: "cd".repeat(32),
        content: args.content,
        tags: args.tags || [],
      };
      store.push(rec);
      return rec;
    }
    if (tool === "recall") {
      const q = (args.query || "").toLowerCase();
      const results = store.filter((r) => {
        if (suppressed.has(r.id)) return false;
        if (args.handle) return r.id === args.handle || r.tags.includes(args.handle);
        return q && r.content.toLowerCase().includes(q);
      });
      return { results: results.slice(0, args.limit || 10) };
    }
    if (tool === "curate" && args.action === "suppress") {
      suppressed.add(args.handle);
      return { suppressed: args.handle };
    }
    return { ok: true };
  };
}

test("@vestige/client core + Vercel AI SDK + Letta adapters", async () => {
  const client = new VestigeClient({ transport: createFakeTransport() });

  // Vercel AI SDK tools
  const tools = createVestigeTools(client);
  const saved = await tools.vestigeRemember.execute({
    content: "Use Ed25519 receipts on every gate decision",
    nodeType: "decision",
    tags: ["crypto"],
  });
  assert.ok(saved.nodeId.startsWith("mem-"));
  const recalled = await tools.vestigeRecall.execute({ handle: "crypto" });
  assert.equal(recalled.results.length, 1);

  // Letta / MemGPT adapter
  const letta = new VestigeLettaAdapter(client, { agentId: "agent-01" });
  const r2 = await letta.archivalMemoryInsert("Strata never erases bytes");
  const hits = await letta.archivalMemorySearch("erases");
  assert.equal(hits.length, 1);
  await letta.archivalMemoryDelete(r2.nodeId);
  const afterSuppress = await letta.archivalMemorySearch("erases");
  assert.equal(afterSuppress.length, 0);
});

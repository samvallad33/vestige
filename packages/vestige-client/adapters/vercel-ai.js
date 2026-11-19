import { VestigeClient } from "../index.js";

/**
 * Drop-in Vercel AI SDK (`ai` package) tools & middleware backed by Vestige Strata.
 */
export function createVestigeTools(client = new VestigeClient(), { scope = "user" } = {}) {
  return {
    vestigeRemember: {
      description: "Save a verified memory or architectural decision to the local Strata causal log.",
      parameters: {
        type: "object",
        properties: {
          content: { type: "string" },
          nodeType: { type: "string", enum: ["fact", "pattern", "decision", "note", "preference"] },
          tags: { type: "array", items: { type: "string" } },
        },
        required: ["content"],
      },
      execute: async ({ content, nodeType = "note", tags = [] }) => {
        return client.remember(content, { nodeType, tags, scope });
      },
    },
    vestigeRecall: {
      description: "Recall memories by exact handle or exact-token BM25 with proof receipts.",
      parameters: {
        type: "object",
        properties: {
          handle: { type: "string" },
          query: { type: "string" },
          limit: { type: "number" },
        },
      },
      execute: async ({ handle, query, limit = 5 }) => {
        return client.recall({ handle, query, limit, scope });
      },
    },
  };
}

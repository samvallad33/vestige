import { VestigeClient } from "../index.js";

/**
 * Drop-in Letta / MemGPT archival & recall memory adapter backed by Vestige Strata.
 */
export class VestigeLettaAdapter {
  constructor(client = new VestigeClient(), { agentId = "default-agent", scope = "user" } = {}) {
    this.client = client;
    this.agentId = agentId;
    this.scope = scope;
  }

  async archivalMemoryInsert(content, tags = []) {
    return this.client.remember(content, {
      nodeType: "fact",
      tags: [`letta:${this.agentId}`, ...tags],
      scope: this.scope,
    });
  }

  async archivalMemorySearch(query, { count = 5 } = {}) {
    const res = await this.client.recall({ query, limit: count, scope: this.scope });
    return res.results || res.memories || [];
  }

  async archivalMemoryDelete(memoryHandle) {
    // Strata append-only contract: suppress hides from reads without erasing bytes.
    return this.client.curate("suppress", memoryHandle, { scope: this.scope });
  }
}

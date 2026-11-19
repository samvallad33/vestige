import { ProofReceipt, VestigeClient } from "../index.js";

export class VestigeLettaAdapter {
  client: VestigeClient;
  agentId: string;
  scope: string;
  constructor(client?: VestigeClient, options?: { agentId?: string; scope?: string });
  archivalMemoryInsert(content: string, tags?: string[]): Promise<ProofReceipt>;
  archivalMemorySearch(query: string, options?: { count?: number }): Promise<unknown[]>;
  archivalMemoryDelete(memoryHandle: string): Promise<Record<string, unknown>>;
}

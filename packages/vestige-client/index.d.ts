export interface ProofReceipt {
  nodeId: string | null;
  receiptId: string | null;
  canonicalHash: string | null;
  raw: Record<string, unknown>;
}

export interface MemoryPacket {
  id: string;
  content: string;
  nodeType?: "fact" | "pattern" | "decision" | "note" | "preference";
  tags?: string[];
  receiptId?: string;
}

export interface CausalEdge {
  from: string;
  to: string;
  kind?: string;
}

export interface RememberOptions {
  nodeType?: "fact" | "pattern" | "decision" | "note" | "preference";
  tags?: string[];
  files?: string[];
  links?: string[];
  scope?: string;
}

export interface RecallOptions {
  handle?: string;
  query?: string;
  limit?: number;
  scope?: string;
}

export type TransportFn = (
  tool: string,
  args: Record<string, unknown>
) => Promise<Record<string, unknown>>;

export interface VestigeClientOptions {
  mcpBin?: string;
  socketPath?: string;
  transport?: TransportFn;
}

export class VestigeError extends Error {
  code: string | number | null;
  constructor(message: string, code?: string | number | null);
}

export function defaultReflexSocket(): string;

export class VestigeClient {
  constructor(options?: VestigeClientOptions);
  callTool(tool: string, args?: Record<string, unknown>): Promise<Record<string, unknown>>;
  session(action?: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  remember(content: string, options?: RememberOptions): Promise<ProofReceipt>;
  recall(options?: RecallOptions): Promise<Record<string, unknown>>;
  causalWalk(fromHandle: string, options?: { direction?: "backward" | "forward"; depth?: number }): Promise<Record<string, unknown>>;
  curate(action: string, handle: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  precedent(action?: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>;
  close(): Promise<void>;
}

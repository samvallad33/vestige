import { VestigeClient } from "../index.js";

export function createVestigeTools(
  client?: VestigeClient,
  options?: { scope?: string }
): Record<string, {
  description: string;
  parameters: Record<string, unknown>;
  execute: (args: any) => Promise<unknown>;
}>;

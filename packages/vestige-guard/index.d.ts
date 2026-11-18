export interface GuardRuleEntry {
  id: string;
  phrase: string;
}

export interface GuardVerdict {
  verdict: "ALLOW" | "CONTRADICTED" | "UNPROVEN";
  citedMemories: string[];
  receiptDigest: string;
}

export function verifyClaim(
  claim: string,
  options?: {
    facts?: GuardRuleEntry[];
    forbids?: GuardRuleEntry[];
    signingKey?: string | Buffer | null;
  }
): GuardVerdict;

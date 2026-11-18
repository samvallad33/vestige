import crypto from "node:crypto";

/**
 * Check an agent claim against active Strata memories and forbidden assertions (#420).
 * Returns a deterministic signed GuardVerdict (`ALLOW` | `CONTRADICTED` | `UNPROVEN`).
 */
export function verifyClaim(claim, { facts = [], forbids = [], signingKey = null } = {}) {
  const low = String(claim || "").toLowerCase();
  const contradictedBy = forbids
    .filter((f) => low.includes(String(f.phrase || "").toLowerCase()))
    .map((f) => f.id);

  let verdict = "UNPROVEN";
  let citedMemories = [];

  if (contradictedBy.length > 0) {
    verdict = "CONTRADICTED";
    citedMemories = contradictedBy;
  } else {
    const supportedBy = facts
      .filter((f) => low.includes(String(f.phrase || "").toLowerCase()))
      .map((f) => f.id);
    if (supportedBy.length > 0) {
      verdict = "ALLOW";
      citedMemories = supportedBy;
    }
  }

  const payload = JSON.stringify({ claim, verdict, citedMemories });
  const digest = signingKey
    ? crypto.createHmac("sha256", signingKey).update(payload).digest("hex")
    : crypto.createHash("sha256").update(payload).digest("hex");

  return {
    verdict,
    citedMemories,
    receiptDigest: digest,
  };
}

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { verifyClaim } from "../index.js";

test("vestige-guard (#420) & MCP registry manifest (#570)", () => {
  const contradicted = verifyClaim("Deleted segment 0001 from disk", {
    facts: [{ id: "mem-1", phrase: "append-only" }],
    forbids: [{ id: "mem-forbid-1", phrase: "deleted segment" }],
    signingKey: "owner-key",
  });
  assert.equal(contradicted.verdict, "CONTRADICTED");
  assert.deepEqual(contradicted.citedMemories, ["mem-forbid-1"]);

  const allowed = verifyClaim("Strata log is append-only", {
    facts: [{ id: "mem-1", phrase: "append-only" }],
  });
  assert.equal(allowed.verdict, "ALLOW");
  assert.deepEqual(allowed.citedMemories, ["mem-1"]);

  const serverJson = JSON.parse(
    readFileSync(path.resolve(import.meta.dirname, "../../../server.json"), "utf8")
  );
  assert.equal(serverJson.name, "io.github.samvallad33/vestige");
  assert.equal(serverJson.packages[0].name, "@vestige/mcp");
});

#!/usr/bin/env bash
# Builds one take of the 60-second proof-carrying PR demo: an empty Vestige
# store, a tiny Node project with two tests, and Claude Code settings that wire
# Reflex to that store. Every run makes a new take folder and points
# $DEMO_DIR/current at it; nothing is ever deleted, so a retake is just
# another run.
#
#   DEMO_DIR             where the takes live (default ~/vestige-proof-demo)
#   VESTIGE_MCP          a vestige-mcp with proof-carrying PRs
#   VESTIGE_VERIFY_PROOF a vestige-verify-proof
set -euo pipefail

DEMO_DIR="${DEMO_DIR:-$HOME/vestige-proof-demo}"
VESTIGE_MCP="${VESTIGE_MCP:-$(command -v vestige-mcp || true)}"
VERIFY="${VESTIGE_VERIFY_PROOF:-$(command -v vestige-verify-proof || true)}"

[ -x "$VESTIGE_MCP" ] || { echo "Set VESTIGE_MCP to a vestige-mcp built with proof-carrying PRs." >&2; exit 2; }
[ -x "$VERIFY" ] || { echo "Set VESTIGE_VERIFY_PROOF to a vestige-verify-proof binary." >&2; exit 2; }
command -v node >/dev/null || { echo "Node is needed for the demo project." >&2; exit 2; }

take="$DEMO_DIR/take-$(date +%Y%m%d-%H%M%S)"
[ ! -e "$take" ] || { echo "$take already exists; run again in a second." >&2; exit 2; }
mkdir -p "$take/store" "$take/gatehome/receipts" "$take/repo/reports"
ln -sfn "$take" "$DEMO_DIR/current"

cat >"$take/store/vestige.toml" <<'EOF'
[witness]
commands = ["npm test"]

[witness.junit]
"npm test" = "reports/junit.xml"
EOF

repo="$take/repo"
cat >"$repo/package.json" <<'EOF'
{
  "name": "calc",
  "private": true,
  "scripts": {
    "test": "node --test --test-reporter=spec --test-reporter-destination=stdout --test-reporter=junit --test-reporter-destination=reports/junit.xml"
  }
}
EOF
cat >"$repo/calc.js" <<'EOF'
exports.add = (a, b) => a + b;
exports.mul = (a, b) => a * b;
EOF
cat >"$repo/calc.test.js" <<'EOF'
const test = require('node:test');
const assert = require('node:assert');
const { add, mul } = require('./calc');

test('adds', () => assert.equal(add(2, 3), 5));
test('multiplies', () => assert.equal(mul(2, 3), 6));
EOF
printf 'reports/\nnode_modules/\n' >"$repo/.gitignore"
git -C "$repo" init -q -b main
git -C "$repo" add -A
git -C "$repo" -c user.name=Demo -c user.email=demo@example.com -c commit.gpgsign=false \
  commit -q -m "calc with two tests"

# Starting the server once mints the store's actor key, the key bundles are
# signed with. Its public half, a did:key, is the key the check trusts.
"$VESTIGE_MCP" --data-dir "$take/store" </dev/null >/dev/null 2>&1 || true
node -e '
const fs = require("fs"), c = require("crypto");
const seed = fs.readFileSync(process.argv[1]);
const der = Buffer.concat([Buffer.from("302e020100300506032b657004220420", "hex"), seed]);
const pub = c.createPublicKey(c.createPrivateKey({ key: der, format: "der", type: "pkcs8" }))
  .export({ format: "der", type: "spki" }).subarray(-32);
const A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
let n = BigInt("0x" + Buffer.concat([Buffer.from([0xed, 1]), pub]).toString("hex")), s = "";
while (n > 0n) { s = A[Number(n % 58n)] + s; n /= 58n; }
fs.writeFileSync(process.argv[2], "did:key:z" + s + "\n");
' "$take/store/actor.key" "$take/signer.txt"

node -e '
const fs = require("fs"), [dir, mcp] = process.argv.slice(1);
const input = { event: "${hook_event_name}", session: "${session_id}", tool: "${tool_name}",
  call: "${tool_use_id}", agent: "${agent_id}", command: "${tool_input.command}", source: "${source}",
  error: "${error}", interrupt: "${is_interrupt}", file: "${tool_input.file_path}", cwd: "${cwd}",
  stdout: "${tool_response.stdout}", stderr: "${tool_response.stderr}",
  interrupted: "${tool_response.interrupted}", background: "${tool_input.run_in_background}", host: "claude" };
const hook = () => [{ matcher: "*", hooks: [{ type: "mcp_tool", server: "vestige-demo", tool: "reflex", input }] }];
fs.writeFileSync(dir + "/settings.json", JSON.stringify({ hooks: {
  PreToolUse: hook(), PostToolUse: hook(), PostToolUseFailure: hook() } }, null, 1));
fs.writeFileSync(dir + "/mcp.json", JSON.stringify({ mcpServers: { "vestige-demo": {
  command: mcp, args: ["--data-dir", dir + "/store"],
  env: { VESTIGE_HTTP_ENABLED: "0", VESTIGE_DASHBOARD_ENABLED: "false",
         VESTIGE_AUTOPILOT_ENABLED: "0", OPERATOR_HOME: dir + "/gatehome" } } } }, null, 1));
' "$take" "$VESTIGE_MCP"

printf 'Run npm test. Then prove it: call the receipt tool with action prove, repoPath %s and commit HEAD. Tell me the Reflex line and the prove summary.\n' "$repo" >"$take/prompt.txt"

cat >"$take/start.sh" <<EOF
#!/usr/bin/env bash
cd "$repo" && exec claude --settings "$take/settings.json" --strict-mcp-config \\
  --mcp-config "$take/mcp.json" --allowedTools "Bash(npm test)" "mcp__vestige-demo"
EOF
cat >"$take/check.sh" <<EOF
#!/usr/bin/env bash
exec "$VERIFY" --repo "$repo" --trust "\$(cat "$take/signer.txt")"
EOF
cat >"$take/flip.sh" <<EOF
#!/usr/bin/env bash
set -e
f="\$(ls "$repo"/.vestige/proof/*.intoto.jsonl | head -n 1)"
cp "\$f" "$take/bundle.orig"
node -e 'const fs=require("fs");const b=fs.readFileSync(process.argv[1]);b[b.length>>1]^=1;fs.writeFileSync(process.argv[1],b)' "\$f"
echo "Changed one byte of \$(basename "\$f")"
EOF
cat >"$take/unflip.sh" <<EOF
#!/usr/bin/env bash
set -e
cp "$take/bundle.orig" "\$(ls "$repo"/.vestige/proof/*.intoto.jsonl | head -n 1)"
echo "Bundle restored"
EOF
chmod +x "$take"/start.sh "$take"/check.sh "$take"/flip.sh "$take"/unflip.sh

echo "Take ready: $DEMO_DIR/current -> $take"
echo "Signer: $(cat "$take/signer.txt")"
echo "Prompt: $(cat "$take/prompt.txt")"

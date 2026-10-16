/**
 * Vestige Operator Lite — OpenClaw plugin
 *
 * Subscribes to before_tool_call and routes shell-like tool calls through the
 * bundled Operator Lite gate (stdlib-only Python, analyzer-only).
 *   gate exit 0 -> allow (return undefined, normal flow continues)
 *   gate exit 2 -> block, with the gate's reason on stderr
 *   spawn error -> fail-open unless the raw text plainly looks destructive
 *
 * Gate resolution order:
 *   1. $OPERATOR_GATE (explicit override)
 *   2. the gate bundled in this package (gate/operator-gate.py)
 *   3. ~/.operator/gate/operator-gate.py (a previously installed copy)
 *
 * Shadow mode by default: the gate classifies and receipts but always exits 0.
 * Flip to blocking: echo enforce > ~/.operator/mode
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const BUNDLED_GATE = fileURLToPath(new URL("./gate/operator-gate.py", import.meta.url));
const INSTALLED_GATE = join(homedir(), ".operator", "gate", "operator-gate.py");

function resolveGate() {
  if (process.env.OPERATOR_GATE) return process.env.OPERATOR_GATE;
  if (existsSync(BUNDLED_GATE)) return BUNDLED_GATE;
  if (existsSync(INSTALLED_GATE)) return INSTALLED_GATE;
  return BUNDLED_GATE; // surface the missing-file error from the gate spawn
}

const GATE_TIMEOUT_MS = 8000;

const DESTRUCTIVE_LIKE =
  /\brm\s+-[a-zA-Z]*[rR]|\bpush\s+.*--force|\bDROP\s+TABLE|vestige\.db|fly\s+deploy|mkfs|\bdd\s+if=/i;

function extractCommand(event) {
  const p = event?.params ?? event?.input ?? event?.arguments ?? event?.toolInput ?? {};
  if (typeof p === "string") return p;
  return p.command ?? p.cmd ?? p.script ?? null;
}

function runGate(gatePath, payload) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (r) => {
      if (!settled) {
        settled = true;
        resolve(r);
      }
    };
    // python3 exits 2 when it cannot open the gate file, and exit 2 reads as STOP: a missing gate would
    // block every call. Treat it as unreachable instead (the fail-open rule below).
    if (!existsSync(gatePath)) return done({ ok: false, err: `gate file missing: ${gatePath}` });
    let child;
    try {
      child = spawn(process.env.OPERATOR_PYTHON || "python3", [gatePath, "hook", "--source", "openclaw"], {
        env: { ...process.env, OPERATOR_AGENT_SESSION: "1" },
        stdio: ["pipe", "ignore", "pipe"],
      });
    } catch (err) {
      return done({ ok: false, err: String(err) });
    }
    let stderr = "";
    child.stderr?.on("data", (d) => {
      stderr += d.toString();
    });
    const timer = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {}
      done({ ok: false, err: "gate timeout" });
    }, GATE_TIMEOUT_MS);
    child.on("error", (err) => {
      clearTimeout(timer);
      done({ ok: false, err: String(err) });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      done({ ok: true, code: code ?? 1, reason: stderr.trim() });
    });
    try {
      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();
    } catch (err) {
      clearTimeout(timer);
      done({ ok: false, err: String(err) });
    }
  });
}

const plugin = {
  id: "vestige-operator-lite",
  name: "Vestige Operator Lite",
  description:
    "Deterministic pre-tool gate: 27 rules, cd/vars/heredoc-aware shell analysis, GuardFall-proof, hash-chained receipts. Gate bundled — shadow mode by default.",
  kind: "security",

  register(api) {
    const gatePath = resolveGate();
    api.logger.info(`operator-lite: registered (gate=${gatePath})`);

    api.on("before_tool_call", async (event, ctx) => {
      const tool = String(event?.tool ?? event?.toolName ?? event?.name ?? "").toLowerCase();
      const command = extractCommand(event);
      if (command == null) return undefined; // nothing shell-like to judge

      const payload = {
        tool_name: tool || "exec",
        tool_input: { command },
        cwd: event?.cwd ?? process.cwd(),
        session_id: ctx?.sessionKey ?? ctx?.runId ?? undefined,
      };
      const res = await runGate(gatePath, payload);
      if (!res.ok) {
        // gate unreachable: fail-open only for plainly non-destructive text
        if (DESTRUCTIVE_LIKE.test(String(command))) {
          api.logger.error(`operator-lite: UNREACHABLE and command looks destructive — blocking. ${res.err}`);
          return {
            block: true,
            blockReason: "Operator gate unreachable and the command looks destructive; ask the owner.",
          };
        }
        api.logger.warn(`operator-lite: unreachable, failing open (${res.err})`);
        return undefined;
      }
      if (res.code === 2) {
        api.logger.warn(`operator-lite: STOP -> ${String(command).slice(0, 160)}`);
        // OpenClaw contract (dist/loader runBeforeToolCall): { block, blockReason }
        return {
          block: true,
          blockReason: res.reason || "OPERATOR LITE: STOPPED",
        };
      }
      return undefined; // allowed (shadow receipts written by the gate itself)
    });
  },
};

export default plugin;

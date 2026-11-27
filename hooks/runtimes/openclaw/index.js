/**
 * Vestige Universal Hook Plugin for OpenClaw.
 * Routes OpenClaw lifecycle & tool calls through vestige-bridge.cjs.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

function resolveBridge() {
  if (process.env.VESTIGE_BRIDGE_PATH && existsSync(process.env.VESTIGE_BRIDGE_PATH)) {
    return process.env.VESTIGE_BRIDGE_PATH;
  }
  const installed = join(homedir(), ".vestige", "memory", "hooks", "vestige-bridge.cjs");
  if (existsSync(installed)) return installed;
  const repoLocal = fileURLToPath(new URL("../../../packages/vestige-init/hooks/vestige-bridge.cjs", import.meta.url));
  if (existsSync(repoLocal)) return repoLocal;
  return installed;
}

function callBridge(event, payload) {
  const bridge = resolveBridge();
  if (!existsSync(bridge)) return {};
  const res = spawnSync(
    process.execPath,
    [bridge, "--runtime", "openclaw", "--event", event],
    {
      input: JSON.stringify({ ...payload, hook_event_name: event }),
      encoding: "utf8",
      timeout: 12000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    }
  );
  if (res.status !== 0) {
    return { deny: true, reason: String(res.stderr || "").trim() || `Vestige blocked (${res.status})` };
  }
  try {
    return JSON.parse(String(res.stdout || "{}").trim().split(/\r?\n/).pop() || "{}");
  } catch {
    return {};
  }
}

export default {
  id: "vestige-hooks",
  name: "Vestige Full Cognitive & Operator Hook Suite",
  register(api) {
    api.on("before_prompt_build", async (event) => {
      const sid = String(event?.sessionId || event?.sessionKey || "openclaw-default");
      const ans = callBridge("UserPromptSubmit", {
        session_id: sid,
        prompt: String(event?.prompt || ""),
        cwd: event?.cwd || process.cwd(),
      });
      if (ans && typeof ans.additionalContext === "string" && ans.additionalContext) {
        return { prependContext: ans.additionalContext };
      }
      return undefined;
    });

    api.on("before_tool_call", async (event) => {
      const sid = String(event?.sessionId || "openclaw-default");
      const ans = callBridge("PreToolUse", {
        session_id: sid,
        tool_name: event?.toolName || event?.name || "",
        tool_input: event?.params ?? event?.input ?? event?.arguments ?? {},
        cwd: event?.cwd || process.cwd(),
      });
      if (ans && (ans.deny || ans.permission === "deny")) {
        return { block: true, blockReason: ans.reason || "Vestige blocked this tool call." };
      }
      if (ans && ans.updatedInput && typeof ans.updatedInput === "object") {
        return { params: ans.updatedInput };
      }
      return undefined;
    });

    api.on("after_tool_call", async (event) => {
      const sid = String(event?.sessionId || "openclaw-default");
      callBridge(event?.error ? "PostToolUseFailure" : "PostToolUse", {
        session_id: sid,
        tool_name: event?.toolName || event?.name || "",
        tool_input: event?.params ?? event?.input ?? {},
        tool_response: {
          exit_code: event?.error ? 1 : 0,
          stderr: event?.error ? String(event.error) : "",
          stdout: typeof event?.result === "string" ? event.result : JSON.stringify(event?.result ?? ""),
        },
        cwd: event?.cwd || process.cwd(),
      });
    });

    api.on("agent_end", async (event) => {
      const sid = String(event?.sessionId || "openclaw-default");
      callBridge("Stop", {
        session_id: sid,
        last_assistant_message: String(event?.finalMessage || ""),
        cwd: event?.cwd || process.cwd(),
      });
    });
  },
};

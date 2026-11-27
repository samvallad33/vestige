/**
 * Vestige Universal Hook Plugin for OpenCode (V1 + V2 compatible).
 *
 * Drop this folder into `.opencode/plugins/vestige/` (project) or
 * `~/.config/opencode/plugins/vestige/` (global).
 *
 * Hooks wired:
 *   - tool.execute.before                -> Operator Guard + Anti-Spiral + Call Coach (throws Error to block; mutates output.args to rewrite)
 *   - tool.execute.after                 -> Failure pairing, git post-commit receipt, curate alerts
 *   - experimental.chat.system.transform -> Zero-Prompt Recall + Capsule + Active Intentions + Pending Alerts
 *   - experimental.session.compacting    -> PreCompact memory flush + Reflex compact
 *
 * Note: OpenCode's loader calls every named export as a plugin function, so this module exports one plugin function.
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

function callBridge(event, payload, directory) {
  const bridge = resolveBridge();
  if (!existsSync(bridge)) return {};
  const res = spawnSync(
    process.execPath,
    [bridge, "--runtime", "opencode", "--event", event],
    {
      cwd: directory || process.cwd(),
      input: JSON.stringify({ ...payload, hook_event_name: event, cwd: payload.cwd || directory || process.cwd() }),
      encoding: "utf8",
      timeout: 12000,
      maxBuffer: 1024 * 1024,
      windowsHide: true,
    }
  );
  if (res.status !== 0) {
    const errText = String(res.stderr || "").trim();
    return { deny: true, reason: errText || `Vestige bridge blocked execution (exit ${res.status}).` };
  }
  const out = String(res.stdout || "").trim();
  if (!out) return {};
  try {
    return JSON.parse(out.split(/\r?\n/).pop() || "{}");
  } catch {
    return {};
  }
}

export const VestigePlugin = async ({ directory, worktree }) => {
  const cwd = worktree || directory || process.cwd();
  const seenSessions = new Set();

  return {
    "experimental.chat.system.transform": async (input, output) => {
      const sid = String(input?.sessionID || input?.sessionId || "opencode-default");
      const ev = seenSessions.has(sid) ? "UserPromptSubmit" : "SessionStart";
      seenSessions.add(sid);
      const ans = callBridge(ev, { session_id: sid, cwd }, cwd);
      if (ans && typeof ans.additionalContext === "string" && ans.additionalContext.trim()) {
        if (Array.isArray(output?.system)) {
          output.system.push(ans.additionalContext);
        }
      }
    },

    "tool.execute.before": async (input, output) => {
      const sid = String(input?.sessionID || input?.sessionId || "opencode-default");
      const ans = callBridge(
        "PreToolUse",
        {
          session_id: sid,
          tool_use_id: input?.callID || input?.callId || "",
          tool_name: input?.tool || "",
          tool_input: (output && output.args) || {},
          cwd,
        },
        cwd
      );
      if (ans && (ans.deny || ans.permission === "deny" || ans.permissionDecision === "deny")) {
        throw new Error(ans.reason || "Vestige blocked this tool execution.");
      }
      if (ans && ans.updatedInput && typeof ans.updatedInput === "object" && output) {
        output.args = ans.updatedInput;
      }
    },

    "tool.execute.after": async (input, output) => {
      const sid = String(input?.sessionID || input?.sessionId || "opencode-default");
      const hasErr = Boolean(output?.metadata?.error || (output?.output && /error|failed/i.test(String(output.title || ""))));
      const ans = callBridge(
        hasErr ? "PostToolUseFailure" : "PostToolUse",
        {
          session_id: sid,
          tool_use_id: input?.callID || input?.callId || "",
          tool_name: input?.tool || "",
          tool_input: input?.args || {},
          tool_response: {
            exit_code: hasErr ? 1 : 0,
            stdout: typeof output?.output === "string" ? output.output : "",
            stderr: hasErr ? String(output?.output || output?.metadata?.error || "") : "",
          },
          cwd,
        },
        cwd
      );
      if (ans && typeof ans.additionalContext === "string" && ans.additionalContext && output && typeof output.output === "string") {
        output.output = `${output.output}\n\n[Vestige Reflex]\n${ans.additionalContext}`;
      }
    },

    "experimental.session.compacting": async (input, output) => {
      const sid = String(input?.sessionID || input?.sessionId || "opencode-default");
      const ans = callBridge("PreCompact", { session_id: sid, cwd, trigger: "auto" }, cwd);
      if (ans && typeof ans.additionalContext === "string" && ans.additionalContext && Array.isArray(output?.context)) {
        output.context.push(ans.additionalContext);
      }
    },
  };
};

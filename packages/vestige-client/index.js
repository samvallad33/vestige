import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";

export class VestigeError extends Error {
  constructor(message, code = null) {
    super(message);
    this.name = "VestigeError";
    this.code = code;
  }
}

export function defaultReflexSocket() {
  if (process.platform === "win32") {
    return "\\\\.\\pipe\\vestige-reflex";
  }
  return path.join(os.homedir(), ".vestige", "reflex.sock");
}

export class VestigeClient {
  constructor(options = {}) {
    this.mcpBin = options.mcpBin || "vestige-mcp";
    this.socketPath = options.socketPath || defaultReflexSocket();
    this.transport = options.transport || null;
    this._proc = null;
    this._rl = null;
    this._pending = new Map();
    this._reqId = 0;
    this._initialized = false;
  }

  _nextId() {
    this._reqId += 1;
    return this._reqId;
  }

  async _ensureStdio() {
    if (!this._proc || this._proc.exitCode !== null) {
      this._proc = spawn(this.mcpBin, [], {
        stdio: ["pipe", "pipe", "ignore"],
      });
      this._rl = readline.createInterface({ input: this._proc.stdout });
      this._rl.on("line", (line) => {
        try {
          const msg = JSON.parse(line);
          if (msg.id !== undefined && this._pending.has(msg.id)) {
            const { resolve, reject } = this._pending.get(msg.id);
            this._pending.delete(msg.id);
            if (msg.error) {
              reject(new VestigeError(msg.error.message || "MCP error", msg.error.code));
            } else {
              resolve(msg.result || {});
            }
          }
        } catch {
          // ignore non-JSON diagnostic lines
        }
      });
      this._initialized = false;
    }
    if (!this._initialized) {
      await this._rpcStdio("initialize", {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "@vestige/client", version: "7.1.0" },
      });
      this._initialized = true;
    }
  }

  _rpcStdio(method, params) {
    return new Promise((resolve, reject) => {
      const id = this._nextId();
      this._pending.set(id, { resolve, reject });
      this._proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  _trySocket(tool, args) {
    return new Promise((resolve, reject) => {
      if (process.platform !== "win32" && !existsSync(this.socketPath)) {
        reject(new Error("no-socket"));
        return;
      }
      const sock = net.createConnection(this.socketPath);
      let buf = "";
      sock.on("connect", () => {
        sock.write(JSON.stringify({ tool, arguments: args }) + "\n");
      });
      sock.on("data", (chunk) => {
        buf += chunk.toString("utf8");
        if (buf.includes("\n")) {
          sock.end();
          try {
            resolve(JSON.parse(buf.trim()));
          } catch (err) {
            reject(err);
          }
        }
      });
      sock.on("error", reject);
    });
  }

  async callTool(tool, args = {}) {
    if (this.transport) {
      return await this.transport(tool, args);
    }
    try {
      return await this._trySocket(tool, args);
    } catch {
      // Fallback to auto-spawned vestige-mcp stdio
    }
    await this._ensureStdio();
    const res = await this._rpcStdio("tools/call", { name: tool, arguments: args });
    if (Array.isArray(res.content) && res.content[0]?.text) {
      try {
        return JSON.parse(res.content[0].text);
      } catch {
        return { text: res.content[0].text };
      }
    }
    return res;
  }

  async session(action = "start", params = {}) {
    return this.callTool("session", { action, ...params });
  }

  async remember(content, options = {}) {
    const args = {
      action: "save",
      content,
      node_type: options.nodeType || "note",
      scope: options.scope || "user",
    };
    if (options.tags) args.tags = options.tags;
    if (options.files) args.files = options.files;
    if (options.links) args.links = options.links;
    const raw = await this.callTool("remember", args);
    return {
      nodeId: raw.nodeId || raw.id || null,
      receiptId: raw.receiptId || null,
      canonicalHash: raw.canonicalHash || null,
      raw,
    };
  }

  async recall({ handle, query, limit = 10, scope = "user" } = {}) {
    const args = { limit, scope };
    if (handle !== undefined) args.handle = handle;
    if (query !== undefined) args.query = query;
    return this.callTool("recall", args);
  }

  async causalWalk(fromHandle, { direction = "backward", depth = 5 } = {}) {
    return this.callTool("causal_walk", { from_handle: fromHandle, direction, depth });
  }

  async curate(action, handle, params = {}) {
    return this.callTool("curate", { action, handle, ...params });
  }

  async precedent(action = "list", params = {}) {
    return this.callTool("precedent", { action, ...params });
  }

  async close() {
    if (this._rl) {
      this._rl.close();
      this._rl = null;
    }
    if (this._proc && this._proc.exitCode === null) {
      this._proc.kill();
      this._proc = null;
    }
  }
}

#!/usr/bin/env python3
"""operator-stop-memory: every Operator stop becomes a Vestige memory. A Claude Code hook that
`npx @vestige/init` registers once a gate is installed.

The Operator gate (Lite or paid) writes a hash-chained receipt for every verdict to
~/.operator/receipts/<UTC date>.jsonl. This hook reads the STOP receipts it has not recorded yet and
saves each one into Vestige through the MCP server's remember, so any agent on the same store can
recall it later by exact handle:

  stop memory   node_type event; tags <rule>, op-receipt-<first 16 hex of the receipt digest>,
                operator, operator-stop; its text quotes that digest; derived_from its rule memory
  rule memory   node_type pattern; tags <rule>, op-rule-<rule>, operator, operator-rule; written the
                first time that rule stops a call

These are the handles `recall` and the PRECEDENT journey (`npx @vestige/init`) look up. A stop already
in memory under its op-receipt tag, from this hook or any other writer, is not written again.

Write path: the hook starts `vestige-mcp` (the --mcp <path> argument, else VESTIGE_MCP_BIN, else the one on
PATH; `npx @vestige/init` registers it with --mcp so no PATH is needed). It attaches to the
server that already owns the store, or opens the store when none does, and the hook speaks MCP over its
stdin: initialize, notifications/initialized, then tools/call. A write from here passes the same gate and
gets the same receipt as an agent's own remember.

The hook never blocks a tool call. It works within a time budget, logs any failure to
~/.operator/state/stop-memory.log, tries that stop again on a later call, and always exits 0. On
PostToolUse and UserPromptSubmit it tells the agent, as additional context, which stops of its own
session it just recorded and the handle that recalls each one. On SessionStart it hands the new session
the most recent stops it recorded on this machine (the last RECENT_DAYS days, at most RECENT_SHOWN), each
with its handle, so the next session knows what was stopped and why before it tries the same thing.

Wire it after the gate's own PreToolUse hook: hooks/settings.stop-memory.fragment.json, or let
`npx @vestige/init` do it, which copies this file to ~/.vestige/hooks/ once Operator Lite is installed.
Env: OPERATOR_HOME (default ~/.operator), VESTIGE_MCP_BIN, OPERATOR_STOP_MEMORY_BUDGET (seconds, default 20).

The server can record these stops itself through `session(action="reflex")` on the v6.2 surface. While
that server is active it touches <data dir>/reflex/heartbeat, and this hook steps aside so a stop is never
handled twice. With an older server, or none, this hook works exactly as before.
Env for that check: VESTIGE_DATA_DIR, else the platform's default Vestige data directory.
Stdlib only, Python 3.9 and later.
"""
import calendar
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

try:
    import fcntl
except ImportError:                                   # Windows: one writer is not enforced there
    fcntl = None

VERSION = "0.2.0"
HOME = os.path.expanduser("~")
OP_HOME = os.environ.get("OPERATOR_HOME") or os.path.join(HOME, ".operator")
RECEIPTS = os.path.join(OP_HOME, "receipts")
STATE_DIR = os.path.join(OP_HOME, "state")
STATE = os.path.join(STATE_DIR, "stop-memory.json")
LOG = os.path.join(STATE_DIR, "stop-memory.log")
LOCK = os.path.join(STATE_DIR, "stop-memory.lock")
MAX_PER_RUN = 8                                       # stops written per hook call; the rest wait for the next
MAX_PENDING = 500
MAX_DONE = 5000
FIRST_RUN_FILES = 30                                  # day files read the first time, newest first
RECENT_KEEP = 20                                      # recorded stops remembered for SessionStart
RECENT_SHOWN = 5                                      # stops a new session is told about
RECENT_DAYS = 7
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
RULE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
NODE_RE = re.compile(r"^(mem-[0-9a-f]{16}|[0-9a-f-]{36})$")
REFLEX_FRESH_SECS = 900                               # a reflex heartbeat younger than this means reflex owns stops


def log(msg):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        if os.path.exists(LOG) and os.path.getsize(LOG) > 262144:
            os.replace(LOG, LOG + ".1")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg))
    except Exception:
        pass


def data_dirs():
    """Where a Vestige server keeps its data directory on this machine, most specific first."""
    dirs = []
    custom = (os.environ.get("VESTIGE_DATA_DIR") or "").strip()
    if custom:
        dirs.append(os.path.expanduser(custom))
    if sys.platform == "darwin":
        dirs.append(os.path.join(HOME, "Library", "Application Support", "com.vestige.core"))
    elif sys.platform.startswith("win"):
        dirs.append(os.path.join(os.environ.get("APPDATA") or os.path.join(HOME, "AppData", "Roaming"), "vestige", "core", "data"))
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(HOME, ".local", "share")
        dirs += [os.path.join(base, "vestige"), os.path.join(base, "core")]
    return dirs


def reflex_active(now=None):
    """True while the native Reflex service records gate stops itself."""
    now = time.time() if now is None else now
    for d in data_dirs():
        try:
            if now - os.path.getmtime(os.path.join(d, "reflex", "heartbeat")) <= REFLEX_FRESH_SECS:
                return True
        except OSError:
            pass
    return False


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            st = json.load(f)
        if isinstance(st, dict):
            st.setdefault("offsets", {})
            st.setdefault("pending", [])
            st.setdefault("done", {})
            st.setdefault("rules", {})
            st.setdefault("recent", [])
            return st
    except Exception:
        pass
    return {"offsets": {}, "pending": [], "done": {}, "rules": {}, "recent": []}


def save_state(st):
    st["pending"] = st["pending"][-MAX_PENDING:]
    st["recent"] = st.get("recent", [])[-RECENT_KEEP:]
    if len(st["done"]) > MAX_DONE:
        st["done"] = dict(list(st["done"].items())[-MAX_DONE:])
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, STATE)


def stop_row(rec):
    """The fields of a STOP receipt this hook keeps, or None when the line is not a usable stop."""
    if not isinstance(rec, dict) or rec.get("decision") != "STOP":
        return None
    digest, rule = str(rec.get("digest") or ""), str(rec.get("commitment") or "")
    if not DIGEST_RE.match(digest) or not RULE_RE.match(rule):
        return None
    clip = lambda v, n: " ".join(str(v or "").split())[:n]
    return {"digest": digest, "prev": clip(rec.get("prev"), 64), "rule": rule, "rule_name": clip(rec.get("rule"), 80),
            "ts": clip(rec.get("ts"), 32), "source": clip(rec.get("source"), 32), "session": clip(rec.get("session"), 64),
            "cwd": clip(rec.get("cwd"), 300), "tool": clip(rec.get("tool"), 64), "preview": clip(rec.get("action_preview"), 300),
            "detail": clip(rec.get("detail"), 300), "action": clip(rec.get("action_digest"), 24),
            "gate_version": clip(rec.get("gate_version"), 16)}


def scan(st):
    """New STOP rows from the receipt files since the last call; the read position is kept per file."""
    try:
        names = sorted(n for n in os.listdir(RECEIPTS) if re.match(r"^\d{4}-\d{2}-\d{2}\.jsonl$", n))
    except Exception:
        return []
    if not st["offsets"]:
        for n in names[:-FIRST_RUN_FILES]:            # the first run reads the newest files only
            try:
                st["offsets"][n] = os.path.getsize(os.path.join(RECEIPTS, n))
            except Exception:
                pass
    rows = []
    for n in names:
        path = os.path.join(RECEIPTS, n)
        try:
            size = os.path.getsize(path)
            start = st["offsets"].get(n, 0)
            if start > size:                          # the file was replaced: read it again
                start = 0
            if start == size:
                continue
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(size - start)
            end = data.rfind(b"\n") + 1               # a line still being written waits for the next call
            for line in data[:end].decode("utf-8", "replace").splitlines():
                try:
                    row = stop_row(json.loads(line))
                except ValueError:
                    continue
                if row:
                    rows.append(row)
            st["offsets"][n] = start + end
        except Exception as exc:
            log("scan %s: %r" % (n, exc))
    keep = set(names)
    st["offsets"] = {k: v for k, v in st["offsets"].items() if k in keep}
    return rows


class Mcp(object):
    """A short MCP client over a spawned vestige-mcp's stdio."""

    def __init__(self, binary, deadline):
        self.deadline = deadline
        self.next_id = 1
        self.lines = queue.Queue()
        self.proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        threading.Thread(target=self._pump, daemon=True).start()
        try:
            self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "operator-stop-memory", "version": VERSION}})
            self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.close()  # construction failure must not leave an orphan MCP process
            raise

    def _pump(self):
        for raw in self.proc.stdout:
            self.lines.put(raw)
        self.lines.put(None)

    def send(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def request(self, method, params):
        rid, self.next_id = self.next_id, self.next_id + 1
        self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            left = self.deadline - time.time()
            if left <= 0:
                raise RuntimeError("%s did not answer within the budget" % method)
            try:
                raw = self.lines.get(timeout=left)
            except queue.Empty:
                raise RuntimeError("%s did not answer within the budget" % method)
            if raw is None:
                raise RuntimeError("vestige-mcp closed before answering %s" % method)
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if isinstance(msg, dict) and "method" not in msg and msg.get("id") == rid:
                if msg.get("error"):
                    raise RuntimeError("%s: %s" % (method, json.dumps(msg["error"])[:300]))
                return msg.get("result") or {}

    def tool(self, name, arguments):
        """The structured result of one tool call (structuredContent, else the JSON text of content[0])."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = ""
        content = result.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            text = str(content[0].get("text") or "")
        if result.get("isError"):
            raise RuntimeError("%s failed: %s" % (name, (text or json.dumps(result))[:300]))
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        try:
            value = json.loads(text)
        except ValueError:
            raise RuntimeError("%s answered with no JSON" % name)
        return value if isinstance(value, dict) else {}

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=max(0.001, min(0.02, self.deadline - time.time())))
        except Exception:
            self.proc.kill()


def tagged(mcp, tag):
    """The id of the memory under this exact tag, or None when no memory carries it."""
    found = mcp.tool("recall", {"handle": tag, "detail_level": "brief"})
    if found.get("error") == "handle_required":
        return None
    for node in found.get("nodes") or []:
        if isinstance(node, dict) and tag in (node.get("tags") or []) and NODE_RE.match(str(node.get("id") or "")):
            return node["id"]
    return None


def ingest(mcp, item):
    out = mcp.tool("remember", {"items": [item]})
    res = (out.get("results") or [{}])[0]
    node = str(res.get("nodeId") or "")
    if not NODE_RE.match(node):
        raise RuntimeError("remember saved nothing: %s" % json.dumps(res)[:300])
    return node


def source_of(row):
    return "operator-gate receipt %s, operator-stop-memory %s" % (row["digest"], VERSION)


def rule_item(row):
    rule = row["rule"]
    return {"node_type": "pattern", "tags": [rule, "op-rule-" + rule, "operator", "operator-rule"],
            "source": source_of(row),
            "content": ("OPERATOR RULE %s%s. Recorded the first time this rule stopped a call: %s UTC, host %s, "
                        "session %s. The gate's reason that time: %s. Every stop under this rule is a memory tagged "
                        "operator-stop and %s that is derived_from this memory, so a causal walk from any of those "
                        "stops reaches the rule." % (rule, " " + row["rule_name"] if row["rule_name"] else "", row["ts"],
                                                    row["source"] or "unknown", row["session"][:8] or "unknown",
                                                    row["detail"] or "not given", rule))}


def stop_item(row, rule_node):
    d16 = row["digest"][:16]
    return {"node_type": "event", "tags": [row["rule"], "op-receipt-" + d16, "operator", "operator-stop"],
            "source": source_of(row), "links": [{"kind": "derived_from", "to": rule_node}],
            "content": ("OPERATOR STOP, recorded from the gate's own receipt. When: %s UTC. Host: %s, session %s. "
                        "Working directory: %s. Tool: %s. Rule: %s%s. Command, as the gate's redacted preview: %s. "
                        "Why: %s. Receipt digest %s, chain prev %s, gate version %s, action id %s. Nothing was run. "
                        "Derived from the rule memory %s. The way forward is the compliant path the stop named, or a "
                        "one-time permit for that action id, which only the owner grants from their own terminal."
                        % (row["ts"], row["source"] or "unknown", row["session"][:8] or "unknown", row["cwd"] or "unknown",
                           row["tool"] or "unknown", row["rule"], " " + row["rule_name"] if row["rule_name"] else "",
                           row["preview"] or "none", row["detail"] or "not given", d16, row["prev"][:16] or "start",
                           row["gate_version"] or "unknown", row["action"] or "none", rule_node))}


def mcp_binary(argv):
    """The vestige-mcp to start: --mcp <path>, else VESTIGE_MCP_BIN, else the one on PATH."""
    for i, arg in enumerate(argv):
        if arg == "--mcp" and i + 1 < len(argv) and argv[i + 1]:
            return argv[i + 1]
        if arg.startswith("--mcp=") and len(arg) > 6:
            return arg[6:]
    return os.environ.get("VESTIGE_MCP_BIN") or shutil.which("vestige-mcp")


def record(st, binary=None):
    """Write the pending stops, newest first, within the budget. Returns [(row, node_id)] written now."""
    try:
        budget = float(os.environ.get("OPERATOR_STOP_MEMORY_BUDGET") or 20)
    except ValueError:
        budget = 20.0
    todo = [r for r in reversed(st["pending"]) if r["digest"][:16] not in st["done"]][:MAX_PER_RUN]
    if not todo:
        return []
    binary = binary or mcp_binary([])
    if not binary:
        log("no vestige-mcp: no --mcp, VESTIGE_MCP_BIN is unset and none is on PATH; %d stops wait" % len(todo))
        return []
    written, mcp = [], None
    try:
        mcp = Mcp(binary, time.time() + budget)
        for row in todo:
            d16 = row["digest"][:16]
            node = tagged(mcp, "op-receipt-" + d16)
            if node:                                  # already in memory, from this hook or another writer
                st["done"][d16] = node
                st.setdefault("recent", []).append({"d16": d16, "node": node, "rule": row["rule"], "ts": row["ts"],
                                                    "preview": row["preview"][:120]})
                continue
            rule_node = st["rules"].get(row["rule"]) or tagged(mcp, "op-rule-" + row["rule"])
            if not rule_node:
                rule_node = ingest(mcp, rule_item(row))
            st["rules"][row["rule"]] = rule_node
            node = ingest(mcp, stop_item(row, rule_node))
            st["done"][d16] = node
            st.setdefault("recent", []).append({"d16": d16, "node": node, "rule": row["rule"], "ts": row["ts"],
                                                "preview": row["preview"][:120]})
            written.append((row, node))
    except Exception as exc:
        log("record: %r" % (exc,))
    finally:
        if mcp:
            mcp.close()
    st["pending"] = [r for r in st["pending"] if r["digest"][:16] not in st["done"]]
    return written


def recent_lines(recent, now):
    """What a new session is told: the newest recorded stops of the last RECENT_DAYS days, with handles."""
    shown = []
    for item in reversed(recent):
        try:
            when = calendar.timegm(time.strptime(item["ts"], "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            continue
        if now - when <= RECENT_DAYS * 86400:
            shown.append(item)
        if len(shown) == RECENT_SHOWN:
            break
    if not shown:
        return []
    lines = ["Operator stopped %d action%s on this machine in the last %d days. Each stop is saved in Vestige memory; "
             "recall its handle to see why, and take the safe path it names instead of trying the same thing again:"
             % (len(shown), "" if len(shown) == 1 else "s", RECENT_DAYS)]
    for item in shown:
        lines.append("- %s %s: %s -> recall(handle=\"op-receipt-%s\")"
                     % (item["rule"], item["ts"], item.get("preview") or "(no preview)", item["d16"]))
    return lines


def main():
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    event = str(payload.get("hook_event_name") or "")
    session = str(payload.get("session_id") or "")
    if reflex_active():
        return 0                                      # the server's reflex records and tells these stops
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        lock = open(LOCK, "a+")
    except Exception:
        return 0
    try:
        if fcntl:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return 0                              # another call is recording; it picks these up
        st = load_state()
        known = set(st["done"]) | set(r["digest"][:16] for r in st["pending"])
        for row in scan(st):
            if row["digest"][:16] not in known:
                st["pending"].append(row)
                known.add(row["digest"][:16])
        written = record(st, mcp_binary(sys.argv[1:]))
        save_state(st)
    except Exception as exc:
        log("main: %r" % (exc,))
        return 0
    finally:
        lock.close()
    if event == "SessionStart":
        lines = recent_lines(st.get("recent", []), time.time())
        if lines:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n".join(lines)}}))
        return 0
    mine = [(r, n) for r, n in written if session and r["session"] == session]
    if mine and event in ("PostToolUse", "UserPromptSubmit"):
        lines = ["Vestige recorded %d Operator stop%s from this session as memory:"
                 % (len(mine), "" if len(mine) == 1 else "s")]
        for r, n in mine:
            lines.append("- %s under %s (receipt %s): recall(handle=\"op-receipt-%s\")"
                         % (n, r["rule"], r["digest"][:16], r["digest"][:16]))
        print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n".join(lines)}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

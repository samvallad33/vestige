#!/usr/bin/env python3
"""operator-gate -- Vestige Operator pre-tool gate for Claude Code, ZCode and Codex.

One script, three hosts. Every host runs it as a PreToolUse command hook:

    python3 ~/.operator/gate/operator-gate.py hook --source claude|zcode|codex

Protocol (identical in all three hosts):
    ALLOW -> exit 0, no output.
    STOP  -> exit 2, reason on stderr (the agent sees it and must change course).

Boundary, stated once and honestly: the gate blocks actions that are routed
through a hooked tool call and that its closed rule set recognises. It cannot
block an action that bypasses the hooked tools, and shell obfuscation it does
not parse is a residual gap. Receipts are hash-chained digests, NOT signatures
(integrity = "reference_digest_not_signature"); do not describe them as signed
or tamper-evident.

Stdlib only, Python 3.9 compatible (macOS /usr/bin/python3).
"""
import calendar
import glob
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import stat as _stat
import subprocess
import sys
import tempfile
import time
import urllib.parse

try:
    import fcntl
except ImportError:                                  # Windows: receipts lock through msvcrt instead
    fcntl = None

VERSION = "6.0.1"
EDITION = "lite"                 # the paid gate is "operator"; install never replaces a gate that outranks this one
INTEGRITY = "reference_digest_not_signature"
IS_WINDOWS = os.name == "nt"


def canon_windows(path):
    """A Windows path in one spelling: forward slashes, an upper-case drive letter, and the Git Bash
    forms /c/x and /cygdrive/c/x read as C:/x. Pure string work, so it can be tested anywhere."""
    p = path.replace("\\", "/")
    m = re.match(r"^/(?:cygdrive/)?([A-Za-z])(/.*)?$", p)
    if m:
        return m.group(1).upper() + ":" + (m.group(2) or "/")
    if re.match(r"^[A-Za-z]:", p):
        return p[0].upper() + p[1:]
    return p


def canon(path):
    """One spelling for a path on every platform. On POSIX this changes nothing."""
    return canon_windows(path) if IS_WINDOWS and path else path


def pj(*parts):
    return canon(os.path.join(*parts))


def is_abs(path):
    if IS_WINDOWS:
        return path.startswith("/") or bool(re.match(r"^[A-Za-z]:/", path))
    return os.path.isabs(path)


HOME = canon(os.path.expanduser("~"))
OP_HOME = canon(os.environ.get("OPERATOR_HOME", pj(HOME, ".operator")))
PERMIT_TTL_S = 15 * 60
MAX_DEPTH = 4

SHELL_TOOLS = {"bash", "shell", "local_shell", "exec_command", "container.exec",
               "run_shell_command", "terminal", "run_terminal_cmd"}
WRITE_TOOLS = {"write", "edit", "multiedit", "notebookedit", "apply_patch", "str_replace_editor",
               "create_file", "edit_file", "write_file"}
FETCH_TOOLS = {"webfetch", "web_fetch", "fetch", "fetch_url"}


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def norm(path, cwd):
    if not path:
        return ""
    p = path.strip().strip("'\"")
    p = p.replace("${HOME}", HOME).replace("$HOME", HOME)
    if p == "~" or p.startswith("~/"):
        p = HOME + p[1:]
    p = canon(p)
    if not is_abs(p):
        p = posixpath.join(canon(cwd or os.getcwd()), p)
    return posixpath.normpath(p)


def real(path):
    """Symlinks resolved (including symlinked ancestor directories when the leaf does not exist yet);
    on Windows this also turns an 8.3 short name into the long one."""
    try:
        r = canon(os.path.realpath(path))
        if r != path or os.path.exists(path):
            return r
        cur = path.rstrip("/")
        tail = []
        while cur and cur != "/":
            parent, leaf = posixpath.split(cur)
            if not leaf or parent == cur:
                break
            tail.append(leaf)
            cur = parent
            if os.path.exists(cur):
                rp = canon(os.path.realpath(cur)).rstrip("/")
                return rp + "/" + "/".join(reversed(tail))
        return r
    except Exception:
        return path


def variants(path):
    """Lexical + symlink-resolved form, so ~/vestige and ~/Developer/vestige are one target."""
    out = {path, real(path)}
    out = {v.rstrip("/") or "/" for v in out}
    return {v.casefold() for v in out} if IS_WINDOWS else out      # Windows paths ignore case


def is_same_or_ancestor(target, protected):
    """True when deleting/moving `target` removes `protected` (equal or an ancestor of it)."""
    for t in variants(target):
        for p in variants(protected):
            if t == p or p.startswith(t.rstrip("/") + "/") or t == "/":
                return True
    return False


def same_path(a, b):
    """The same place, whatever the spelling: symlinks resolved, and on Windows slashes and case."""
    return bool(a and b and (variants(a) & variants(b)))


def is_inside(target, root):
    for t in variants(target):
        for r in variants(root):
            if t == r or t.startswith(r.rstrip("/") + "/"):
                return True
    return False


# --------------------------------------------------------------------------- #
# commitments (the owner's rules) -- defaults in code, extended by commitments.json
# --------------------------------------------------------------------------- #
def default_protected_roots():
    d = pj(HOME, "Developer")
    return [
        pj(d, "vestige"), pj(HOME, "vestige"),
        pj(d, "vestige-launch-private"), pj(d, "vestige-operator"),
        pj(d, "vestige-cloud"), pj(d, "vestige-evidence"),
        pj(d, "vestige-LIMEN"), pj(d, "vestige-nc"),
        pj(d, "vestige-ollama"), pj(d, "vestige-extra"),
        d,
        pj(HOME, ".vestige"), pj(HOME, ".zcode"),
        pj(HOME, ".claude"), pj(HOME, ".codex"),
        pj(HOME, ".operator"), HOME,
    ]


def gate_homes():
    return sorted({os.path.abspath(OP_HOME), pj(HOME, ".operator")})


def touches_gate_home(path):
    return any(is_inside(path, g) for g in gate_homes())


def self_protected_files():
    return [
        pj(HOME, ".claude", "settings.json"),
        pj(HOME, ".claude", "settings.local.json"),
        pj(HOME, ".zcode", "cli", "config.json"),
        pj(HOME, ".codex", "hooks.json"),
        pj(HOME, ".codex", "config.toml"),
    ]


RULES = {
    "OP-000": ("protect-the-gate", "STOP",
               "Editing the gate, its rules, permits, mode or the hook registrations is owner-only."),
    "OP-001": ("protect-workspaces", "STOP",
               "Deleting or moving a registered workspace root (or a parent of one) is owner-only."),
    "OP-002": ("protect-memory", "STOP",
               "Wiping, deleting or rewriting the Vestige memory store, or an agent's own record (its "
               "transcripts and shell history), is owner-only."),
    "OP-003": ("no-blind-recursive-delete", "STOP",
               "Recursive delete outside scratch/build directories needs the owner's approval."),
    "OP-004": ("no-history-destruction", "STOP",
               "Force-pushing over or deleting a shared branch, expiring reflogs, or deleting .git destroys history."),
    "OP-005": ("no-unreviewed-publish", "STOP",
               "Publishing a release/package or deleting a public repo needs the owner's review."),
    "OP-006": ("no-paid-deploy", "STOP",
               "Production deploys, infrastructure/bucket/project teardown, live database changes and live billing actions need the owner's approval."),
    "OP-007": ("no-destructive-sql", "STOP",
               "DROP/TRUNCATE, unscoped or tautological DELETE/UPDATE, database/schema resets, or Docker volume wipes need the owner's approval."),
    "OP-008": ("no-shell-init-write", "STOP",
               "Writing shell rc/init files plants commands that fire later; that is code execution by install."),
    "OP-009": ("no-reverse-shell", "STOP",
               "/dev/tcp, /dev/udp, nc -e and DNS-tunneling tools are raw outbound command channels, not tooling."),
    "OP-010": ("no-cloud-metadata", "STOP",
               "Cloud metadata endpoints hand out instance credentials; an agent has no legitimate reason to query them."),
    "OP-011": ("no-unrestorable-loss", "STOP",
               "Discarding uncommitted git work, deleting/overwriting/shredding/emptying files outside the project, or wiping the crontab cannot be undone."),
    "OP-014": ("no-machine-wreck", "STOP",
               "Writing a raw disk, formatting or erasing a volume, halting/rebooting, kill -1, a fork bomb, or changing permissions or owners recursively on / or home wrecks the machine."),
    "OP-018": ("no-blind-fetch-run", "STOP",
               "Piping a download straight into a shell or interpreter, or running it through <(...) or $(...), runs code nobody has read."),
    "OP-012": ("the-floor", "STOP",
               "A recursive delete that reaches a drive or filesystem root, a home folder or a parent of the "
               "working directory is refused in every mode, and no permit exists for it."),
    "OP-013": ("no-root-reach", "STOP",
               "A root job started in the background, or a kill aimed at every process or at a process "
               "group as root, needs the owner's approval."),
    "OP-015": ("no-repo-code-run", "STOP",
               "A git setting that runs a program (core.fsmonitor, core.sshCommand, an external diff or "
               "filter, a ! alias) turns an ordinary git command into code execution. Setting one, or "
               "running git where one is set, needs the owner's approval."),
    "OP-016": ("no-vulnerable-write", "STOP",
               "The code about to be written carries a known critical flaw (a server key in client code, "
               "certificate or signature checking switched off, an unsafe loader, a secret baked into an "
               "image). Writing it needs the owner's approval."),
    "OP-017": ("no-agent-arming", "STOP",
               "Some project files are run by an agent or an editor on its own the next time it starts "
               "(MCP server lists, hook settings, notify commands, folder-open tasks, .envrc). Writing a "
               "command into one needs the owner's approval."),
    "OP-019": ("no-harmful-chain", "STOP",
               "Steps that are each ordinary but together run a file this session fetched, carry a secret "
               "out, write through a link the session made, run a build file it wired to launch a process, "
               "or push past a check it switched off, need the owner's approval."),
    "OP-020": ("no-second-spelling", "STOP",
               "The gate already stopped an action on this target in this session. Reaching the same target "
               "another way (a script, inline code, a loop, a different command) is the same action and "
               "needs the owner's approval."),
    "OP-021": ("no-found-key", "STOP",
               "A credential this session found in a file is not permission to change a remote system: "
               "a write, delete or deploy carrying that key needs the owner's approval."),
    "OP-024": ("runaway-brake", "STOP",
               "A session the gate stopped again and again in a few minutes is a runaway. Until it goes quiet or the "
               "owner releases it, its calls that change anything are held; reads still pass."),
    "OP-023": ("no-remote-delete", "STOP",
               "An HTTP DELETE, or a GraphQL mutation that deletes, destroys, drops, purges, wipes or terminates, "
               "sent to a remote API removes something the owner may not get back. It needs the owner's approval."),
    "OP-022": ("protected-file", "STOP",
               "The owner gave this file a contract: append-only files may only grow at the end, read-only "
               "files may not change. Anything else needs the owner."),
    "OP-025": ("watch-turnover", "STOP",
               "A session that was compacted takes the watch before it changes anything: it reads what it was "
               "stopped from, what the owner allowed and the permits still open, from the gate's own receipts, "
               "not from a summary. Reads go ahead meanwhile."),
    "OP-CANARY": ("canary", "STOP",
                  "The verification canary string, which proves a host is wired, or a value the owner planted as a "
                  "canary with `canary plant`: nothing legitimate ever uses one, so a call carrying it is stopped."),
    # shadow-only candidates: recorded, never blocking, promote after reviewing the log
    "OP-S01": ("work-loss", "SHADOW", "git reset --hard / clean / branch -D can discard uncommitted work."),
    "OP-S02": ("outbound-comms", "SHADOW", "Sending a message/email/forward on the owner's behalf."),
    "OP-S03": ("public-mutation", "SHADOW", "Creating or editing a pull request, issue or discussion in the owner's name."),
    "OP-S04": ("secret-shaped-write", "SHADOW", "Writing a credential-shaped string to disk."),
    "OP-S05": ("unresolved-delete", "SHADOW", "Delete whose targets cannot be resolved statically (xargs, inline code)."),
    "OP-S06": ("pipe-to-interpreter", "SHADOW",
               "Piping fetched/encoded content straight into sh/python executes unreviewed code."),
    "OP-S07": ("persistence-write", "SHADOW",
               "cron/launchd/systemd/at/tmux -d writes schedule code to run outside every audit window."),
    "OP-S08": ("env-hijack", "SHADOW",
               "LD_PRELOAD/DYLD_*/NODE_OPTIONS/BASH_ENV/PATH= turn allow-listed binaries into arbitrary code."),
    "OP-S11": ("exfil-shape", "SHADOW",
               "Uploading/POSTing file contents (esp. credential-shaped paths) to the network."),
    "OP-S12": ("invisible-characters", "SHADOW",
               "Zero-width/Bidi control characters in a command are visual deception (Trojan Source)."),
    "OP-S13": ("mcp-argument-exfil", "SHADOW",
               "Secrets or sensitive files traveling through MCP tool arguments."),
    "OP-S14": ("agent-config-write", "SHADOW",
               "Writing agent instruction/config files (CLAUDE.md, .cursorrules, .mcp.json) plants future instructions (Nx, CurXecute)."),
    "OP-S15": ("paste-tunnel-egress", "SHADOW",
               "Paste sites, webhook bins and tunnel services are the standard exfil destinations (EchoLeak, S1ngularity)."),
    "OP-S16": ("new-git-remote", "SHADOW",
               "Adding a git remote or creating repos mid-session is the S1ngularity/Shai-Hulud egress path."),
    "OP-S17": ("sandbox-escape-primitive", "SHADOW",
               "Container escape verbs, git trust-anchor reconfiguration and trace tampering are sandbox-break stages."),
}

# v0.2.4 — agent instruction/config paths (Nx poisoned CLAUDE.md; CurXecute rewrote mcp.json)
AGENT_CONFIG_NAMES = {"claude.md", "agents.md", ".cursorrules", ".windsurfrules", ".mcp.json",
                      "mcp_settings.json", ".cursor", "copilot-instructions.md"}

SCRATCH_OK = ("/tmp", "/private/tmp", "/var/folders", "/private/var/folders")
BUILD_DIRS = {"node_modules", "target", "dist", "build", ".venv", "venv", "__pycache__",
              ".next", ".turbo", ".cache", "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              ".parcel-cache", ".gradle", "DerivedData", "out", ".svelte-kit", ".nuxt", ".dart_tool"}

# v0.2.2 — the incident-wave rule constants
SHELL_INIT_FILES = (".zshrc", ".zshenv", ".zprofile", ".bashrc", ".bash_profile", ".profile",
                    ".ssh/rc", ".zlogin", ".config/fish/config.fish")
HIJACK_VARS = {"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
               "NODE_OPTIONS", "NODE_PATH", "PYTHONSTARTUP", "PYTHONPATH", "BASH_ENV", "ENV",
               "PERL5OPT", "RUBYOPT", "GLIBC_TUNABLES", "SHELL"}
PERSISTENCE_PROGS = {"crontab", "at", "batch", "launchctl", "systemctl", "tmux", "screen",
                     "anacron", "schtasks"}
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")
EXFIL_SECRET_PATH_RE = re.compile(r"\.(env|pem|key|p12|pfx|kdbx)$|id_rsa|credentials|\.aws[/\\]|\.netrc", re.I)
MCP_SECRET_ARG_RE = re.compile(
    r"AKIA[0-9A-Z]{16}|sk-ant-\S{10,}|sk-proj-\S{10,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{16,}|"
    r"gho_[A-Za-z0-9]{16,}|github_pat_\S{20,}|xox[bp]-\S{10,}|BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY|"
    r"api[_-]?key[\"'=:\s]{1,3}[A-Za-z0-9_\-]{16,}", re.I)
CLOUD_METADATA_RE = re.compile(r"169\.254\.169\.254|metadata\.google\.internal|100\.100\.100\.200|/latest/api/token|/metadata/identity/oauth2/token")
PASTE_TUNNEL_RE = re.compile(r"webhook\.site|requestbin|pipedream|paste\.rs|pastebin|hastebin|termbin|transfer\.sh|file\.io|0x0\.st|tmpfiles\.org|ngrok|localtunnel|serveo|bore\.pub|trycloudflare|pasty\.", re.I)
DNS_TUNNEL_RE = re.compile(r"\b(dnscat2?|iodine|dns2tcp|dnsteal|h2nc)\b", re.I)
CONTAINER_ESCAPE_RE = re.compile(
    r"docker\s+run\s+[^|;&\n]*(--privileged|--pid=host|--network=host|-v\s+/:/|-v\s+/var/run/docker\.sock)"
    r"|\bnsenter\b|\bunshare\b|chroot\s+[^\n]*/bin/sh|/proc/sys/kernel/core_pattern|/sys/fs/cgroup"
    r"|/var/run/docker\.sock", re.I)
GIT_ANCHOR_RE = re.compile(r"core\.fsmonitor|core\.hooksPath|attr\.tree|\bfilter\.", re.I)
TRACE_TAMPER_RE = re.compile(r"projects[/\\][^\n]*\.jsonl|reflog|\.git/logs|journalctl[^\n]*vacuum|log\s+erase", re.I)
MCP_SENSITIVE_PATH_RE = re.compile(
    r"\.ssh[/\\]|id_rsa|\.env\b|\.mcp\.json|credentials|keychain|\.netrc|\.aws[/\\]|Login Data|Cookies", re.I)


def load_config():
    cfg = {"protected_roots": [], "protected_files": [], "disabled_rules": [], "mode_overrides": {},
           "scratch_ok": [], "file_contracts": []}     # file_contracts: OP-022, set by the owner's `protect`
    try:
        with open(pj(OP_HOME, "commitments.json"), encoding="utf-8") as f:
            user = json.load(f)
        for k in cfg:
            if k in user:
                cfg[k] = user[k]
    except Exception:
        pass
    return cfg


# --------------------------------------------------------------------------- #
#
# Monotonic invariant (Progent, arXiv 2504.11703): a commitment can only ADD a
# constraint. There is no syntax for allowing anything -- the 27 built-in OP
# rules are the floor and the owner's law is an overlay that may only narrow.
# The hook never parses the live commitments file: `operator-gate commitments
# check` validates it and atomically writes a cache stamped with law_digest.
# A broken edit can neither brick the gate nor take effect.
# --------------------------------------------------------------------------- #




















def global_mode():
    m = os.environ.get("OPERATOR_GATE_MODE")
    if not m:
        try:
            with open(pj(OP_HOME, "mode"), encoding="utf-8") as f:
                m = f.read().strip()
        except Exception:
            m = "enforce"
    return m if m in ("enforce", "shadow", "off") else "enforce"


# Every Lite install before 0.3.9 (the gate's own and each port's) wrote `shadow` into the mode file, and
# shadow blocks nothing. The mode is settled once per gate home, by the first install, `mode` command or
# hook call of 0.3.9 or later; this file records that it was.
MODE_SETTLED = "mode-settled.json"


def owner_chose_shadow():
    """True when the owner's last word on the mode, a MODE_SET receipt from the `mode` command in their
    terminal (0.3.8 and later), chose shadow."""
    rdir = pj(OP_HOME, "receipts")
    try:
        days = sorted((n for n in os.listdir(rdir) if n.endswith(".jsonl")), reverse=True)
    except Exception:
        return False
    for name in days:
        last = None
        try:
            with open(pj(rdir, name), encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "MODE_SET" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("decision") == "MODE_SET" and rec.get("source") == "owner-tty":
                        last = rec.get("mode")
        except Exception:
            continue
        if last:
            return last == "shadow"
    return False


def installed_gates():
    return sorted({pj(OP_HOME, "gate", "operator-gate.py"), pj(HOME, ".operator", "gate", "operator-gate.py")})


def record_settled(by, mode, switched):
    try:
        ensure_dirs()
        with open(pj(OP_HOME, "state", MODE_SETTLED), "w", encoding="utf-8") as f:
            json.dump({"by": by, "gate_version": VERSION, "mode": mode, "switched": switched,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    except Exception:
        pass


def settle_mode(by):
    """Settle the mode file once, and return (mode, switched). A `shadow` that an install before 0.3.9
    left becomes enforce, with a MODE_SET receipt whose source is "upgrade". A shadow the owner chose with
    the `mode` command stays, and every choice made after the settling stays. Where the paid gate is
    installed it owns the mode, and nothing is changed.
    by="hook" settles only when an installed gate shows that an installer ran here: with none, a shadow
    file is the owner's own `echo shadow`, and it stays. Never raises."""
    try:
        with open(pj(OP_HOME, "mode"), encoding="utf-8") as f:
            current = f.read().strip()
    except Exception:
        current = None
    try:
        if os.path.exists(pj(OP_HOME, "state", MODE_SETTLED)):
            return current, False
        gates = [g for g in installed_gates() if os.path.isfile(g)]
        if any(gate_identity(g)[0] == "operator" for g in gates):
            return current, False
        switched = current == "shadow" and (by != "hook" or bool(gates)) and not owner_chose_shadow()
        if switched:
            with open(pj(OP_HOME, "mode"), "w", encoding="utf-8") as f:
                f.write("enforce\n")
            write_receipt({"decision": "MODE_SET", "mode": "enforce", "was": "shadow", "source": "upgrade", "by": by})
            current = "enforce"
        record_settled(by, current or "enforce", switched)
        return current, switched
    except Exception:
        return current, False


SETTLED_NOTE = ("Operator Lite before 0.3.9 installed in shadow mode, which blocks nothing; "
                "the gate now blocks what its rules stop")


# --------------------------------------------------------------------------- #
# shell parsing (closed, conservative)
# --------------------------------------------------------------------------- #
def split_segments(cmd):
    """Split on ; && || | & and newlines, respecting quotes. Returns list of strings."""
    segs, cur, q, i, n = [], [], None, 0, len(cmd)
    while i < n:
        c = cmd[i]
        if q:
            cur.append(c)
            if c == "\\" and q == '"' and i + 1 < n:
                cur.append(cmd[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"":
            q = c; cur.append(c)
        elif c == "\\" and i + 1 < n:
            cur.append(c); cur.append(cmd[i + 1]); i += 1
        elif c in ";\n":
            segs.append("".join(cur)); cur = []
        elif c in "&|":
            if i + 1 < n and cmd[i + 1] == c:
                i += 1
            segs.append("".join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    segs.append("".join(cur))
    return [s.strip() for s in segs if s.strip()]


def tokenize(seg):
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


WRAPPERS = {"sudo", "doas", "env", "command", "builtin", "nohup", "time", "nice", "exec", "stdbuf",
            "caffeinate", "timeout", "xargs", "unbuffer", "setsid", "busybox", "toybox",
            "ionice", "chrt", "taskset", "arch", "setarch", "pkexec", "eatmydata", "xcrun"}
WRAPPER_ARG_FLAGS = {
    "sudo": {"-u", "-g", "-C", "-D", "-R", "-T", "-U", "-h", "-p",
             "--user", "--group", "--chdir", "--chroot", "--other-user", "--prompt", "--role", "--type"},
    "doas": {"-u", "-C"},
    "env": {"-u", "--unset", "-C", "--chdir"},
    "xargs": {"-I", "-i", "-L", "-l", "-n", "-P", "-s", "-d", "-a", "-E", "-e",
              "--delimiter", "--max-args", "--max-procs", "--max-lines", "--replace", "--arg-file", "--eof"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "ionice": {"-c", "-n", "-p", "-P", "-u"},
    "chrt": {"-p"},
    "taskset": {"-c", "-p"},
    "arch": {"-arch"},
    "xcrun": {"-sdk", "--sdk", "-toolchain", "--toolchain"},
}


def xargs_repl_token(toks):
    """Custom replacement token from `xargs -I <tok>` or `--replace=<tok>`, default `{}`."""
    for i, t in enumerate(toks):
        if os.path.basename(t) == "xargs":
            j = i + 1
            while j < len(toks) and toks[j].startswith("-"):
                f = toks[j]
                if f in ("-I", "-i", "--replace") and j + 1 < len(toks):
                    return toks[j + 1]
                if f.startswith("-I") and len(f) > 2:
                    return f[2:]
                if f.startswith("--replace=") and len(f) > 10:
                    return f.split("=", 1)[1]
                j += 2 if f in WRAPPER_ARG_FLAGS["xargs"] and "=" not in f else 1
    return "{}"


def strip_wrappers(toks):
    """Drop leading VAR=x, wrappers and their flags. Returns (tokens, via_xargs)."""
    toks = list(toks)
    via_xargs = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", t):
            i += 1; continue
        base = os.path.basename(t)
        if base in WRAPPERS:
            if base == "xargs":
                via_xargs = True
            i += 1
            arg_flags = WRAPPER_ARG_FLAGS.get(base, ())
            while i < len(toks) and toks[i].startswith("-"):
                f = toks[i]
                if f == "--":
                    i += 1
                    break
                if base == "env" and (f in ("-S", "--split-string") and i + 1 < len(toks)
                                      or f.startswith("-S") and len(f) > 2
                                      or f.startswith("--split-string=")):
                    if f in ("-S", "--split-string"):
                        raw_s = toks[i + 1]
                        toks = toks[:i] + tokenize(raw_s) + toks[i + 2:]
                    else:
                        raw_s = f[2:] if f.startswith("-S") else f.split("=", 1)[1]
                        toks = toks[:i] + tokenize(raw_s) + toks[i + 1:]
                    continue
                i += 1
                if f in arg_flags and i < len(toks):
                    i += 1
                elif base in ("timeout", "nice", "sudo") and i < len(toks) and re.match(r"^[0-9]+$", toks[i]):
                    i += 1
            if base == "timeout" and i < len(toks) and re.match(r"^[0-9.]+[smhd]?$", toks[i]):
                i += 1
            continue
        break
    return toks[i:], via_xargs


REDIRECT_TOKEN_RE = re.compile(r"^(?:[0-9]*|&)(?:>>|>&|>\||<<<|<>|>|<)(.*)$")


def strip_leading_redirects(toks):
    """A redirection may come before the command: `>log rm -rf x` and `2>/dev/null rm -rf x` run rm.
    Drop leading redirections, whether fused with their target (`>log`) or followed by it (`> log`)."""
    i = 0
    while i < len(toks):
        m = REDIRECT_TOKEN_RE.match(toks[i])
        if not m:
            break
        i += 1 if m.group(1) else 2
    return toks[i:]


def flags_and_args(toks):
    flags, args, dashdash = [], [], False
    for t in toks:
        if dashdash:
            args.append(t)
        elif t == "--":
            dashdash = True
        elif t.startswith("-") and len(t) > 1:
            flags.append(t)
        else:
            args.append(t)
    return flags, args


def has_flag(flags, short, long_=None):
    for f in flags:
        if long_ and f == long_:
            return True
        if f.startswith("--"):
            continue
        if short in f[1:]:
            return True
    return False


UNKNOWN_CWD = "/__unknown_cwd__"


def vestige_data_dirs(env, home, platform):
    """Where Vestige keeps its Strata store: VESTIGE_DATA_DIR first, then the platform default
    (docs/STORAGE.md). On Linux, 5.0 stores live in $XDG_DATA_HOME/vestige, and a 4.x store in
    $XDG_DATA_HOME/core keeps being used until the new folder holds one, so both are guarded. Pure, so
    every platform's answer can be tested anywhere."""
    dirs = []
    custom = (env.get("VESTIGE_DATA_DIR") or "").strip()
    if custom:
        dirs.append(canon(home + custom[1:] if custom.startswith("~") else custom).rstrip("/"))
    if platform == "darwin":
        dirs.append(home + "/Library/Application Support/com.vestige.core")
    elif platform.startswith("win"):
        dirs.append(canon_windows(env.get("APPDATA") or home + "/AppData/Roaming").rstrip("/") + "/vestige/core/data")
    else:
        xdg = env.get("XDG_DATA_HOME") or ""
        base = xdg.rstrip("/") if xdg.startswith("/") else home + "/.local/share"
        dirs += [base + "/vestige", base + "/core"]
    return dirs


STRATA_DIRS = [canon(d) for d in vestige_data_dirs(os.environ, HOME, sys.platform)]
MEMORY_DIRS = [pj(HOME, ".vestige")] + STRATA_DIRS
MEMORY_FILE_RE = re.compile(r"vestige\.db|\.wal$|\.shm$|\.sqlite3?$", re.I)
# An agent's own record: the transcripts of what it was told and did, and the shell's history of what
# it ran. Reading them is fine; erasing, emptying or rewriting them is the owner's call (#529).
RECORD_DIRS = [pj(HOME, ".claude", "projects"), pj(HOME, ".codex", "sessions"), pj(HOME, ".codex", "archived_sessions"),
               pj(HOME, ".gemini", "tmp")]
if os.environ.get("CLAUDE_CONFIG_DIR", "").startswith("/"):
    RECORD_DIRS.append(pj(canon(os.environ["CLAUDE_CONFIG_DIR"]), "projects"))
RECORD_FILES = [pj(HOME, n) for n in (".zsh_history", ".zhistory", ".bash_history", ".sh_history", ".history")] + [
    pj(HOME, ".local", "share", "fish", "fish_history"), pj(HOME, ".codex", "history.jsonl")]
if os.environ.get("HISTFILE", "").startswith("/"):
    RECORD_FILES.append(canon(os.environ["HISTFILE"]))
RECORD_HINT = "The record outlives the task: the owner decides what is erased and when, with a one-time permit."
# Inside a Strata data dir: the hash-chained log (segments, head.state, strata.key), the keys that sign
# its receipts and identify its actor, the store marker, and the backups. Exports and logs stay writable.
STRATA_CRITICAL_RE = re.compile(r"^(?:log|backups)(?:/.*)?$|^(?:actor\.key|receipt-signing\.key|store\.meta)$")
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "{", "(", "!", "}", ")", "done", "fi", "esac"}
LOOP_HEADS = {"for", "case", "select"}
FUNC_HEAD_RE = re.compile(r"^[^\s(){}|&;<>=$`]+\(\)\{?$")


def strip_function_head(toks):
    """`name() { body` and `function name { body`: the body's first command is the program."""
    if toks and toks[0] == "function":
        toks = toks[2:]
        if toks and toks[0] in ("()", "(){"):
            toks = toks[1:]
    elif toks and FUNC_HEAD_RE.match(toks[0]):
        toks = toks[1:]
    elif len(toks) > 1 and toks[1] in ("()", "(){") and re.match(r"^[^\s(){}|&;<>=$`]+$", toks[0]):
        toks = toks[2:]
    return toks
TRUNCATE_RE = re.compile(r"^\s*(?:(?::|true|cat\s+/dev/null|echo\s+-n(?:\s+(?:''|\"\"))?)?\s*>\|?\s*(?P<a>[^\s;&|<>]+)\s*$"
                         r"|^\s*truncate\s+(?:-s\s*0|--size[= ]0)\s+(?P<b>[^\s;&|<>]+)\s*$"
                         r"|^\s*cp\s+/dev/null\s+(?P<c>[^\s;&|<>]+)\s*$)")
HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
SUBST_RE = re.compile(r"\$\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)|`[^`]*`")
INTERPRETERS = ("python", "python3", "node", "perl", "ruby", "deno", "bun", "php", "osascript", "lua")


def extract_heredocs(cmd):
    """Remove heredoc bodies (data, not commands). Returns (command_text, [bodies])."""
    lines, out, bodies, i = cmd.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in list(HEREDOC_RE.finditer(line)):
            word, body = m.group(2), []
            while i < len(lines) and lines[i].strip() != word:
                body.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            bodies.append("\n".join(body))
    return "\n".join(out), bodies


def mask_quotes(text, mask_double=True):
    """Same length as text, with every character inside single quotes, shell `#` comments, and
    (when mask_double is True) double quotes replaced by a space, so structural scans skip quoted
    data and comments."""
    out, q, i, n = [], None, 0, len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                out.append(" " if mask_double else c)
                out.append(" " if mask_double else text[i + 1])
                i += 2
                continue
            if c == q:
                out.append(c); q = None
            elif c == "\n":
                out.append("\n")
            elif q == "'" or mask_double:
                out.append(" ")
            else:
                out.append(c)
        elif c == "#" and (i == 0 or text[i - 1] in " \t\n;&|("):
            while i < n and text[i] != "\n":
                out.append(" "); i += 1
            continue
        elif c in "'\"":
            q = c; out.append(c)
        elif c == "\\" and i + 1 < n:
            out.append(c); out.append(text[i + 1]); i += 2
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def expand_vars(text, vars_):
    # Strip empty positional/special parameters glued inside tokens (`r$@m`, `r$*m`, `r$9m`, `r${@}m`)
    if "@" not in vars_ and "*" not in vars_:
        text = re.sub(r"(?<=[A-Za-z0-9_./~-])\$(?:\{[@*1-9]\}|[@*1-9])|\$(?:\{[@*1-9]\}|[@*1-9])(?=[A-Za-z0-9_./~-])", "", text)

    def rep_brace(m):
        name, op, arg = m.group(1), m.group(2) or "", m.group(3) or ""
        if name == "IFS":
            return " "
        v = vars_.get(name)
        if op in (":-", "-"):
            if v is not None and (v != "" or op == "-"):
                return v
            return expand_vars(arg, vars_)
        if op in (":=", "="):
            if v is not None and (v != "" or op == "="):
                return v
            val = expand_vars(arg, vars_)
            vars_[name] = tilde(val)
            return val
        if op in (":+", "+"):
            if v is not None and (v != "" or op == "+"):
                return expand_vars(arg, vars_)
            return ""
        if op == ",,":
            return v.lower() if v is not None else m.group(0)
        if op == "^^":
            return v.upper() if v is not None else m.group(0)
        if op == ",":
            return (v[:1].lower() + v[1:]) if v is not None else m.group(0)
        if op == "^":
            return (v[:1].upper() + v[1:]) if v is not None else m.group(0)
        if op in ("/", "//") and v is not None:
            parts = arg.split("/", 1)
            pat = parts[0]
            repl = parts[1] if len(parts) > 1 else ""
            return v.replace(pat, repl) if op == "//" else v.replace(pat, repl, 1)
        if op == ":" and v is not None:
            parts = arg.split(":", 1)
            try:
                off = int(parts[0].strip())
                ln = int(parts[1].strip()) if len(parts) > 1 else None
                return v[off:off + ln] if ln is not None else v[off:]
            except ValueError:
                return v
        return v if v is not None else m.group(0)

    for _ in range(3):
        prev = text
        text = re.sub(
            r"\$\{!([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda m: vars_.get(vars_.get(m.group(1), ""), m.group(0)), text)
        text = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-|:=|:\+|[-=+]|,,|\^\^|,|\^|//|/|:)([^{}]*))?\}",
            rep_brace, text)
        if text == prev:
            break

    def rep_simple(m):
        name = m.group(1) or m.group(2)
        if name == "IFS":                      # word separator: `rm$IFS-rf` must reassemble
            return " "
        v = vars_.get(name)
        return v if v is not None else m.group(0)
    text = text.replace("${0}", vars_.get("0", "sh"))
    return re.sub(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|(0)(?![0-9]))", rep_simple, text)


def tilde(v):
    return HOME + v[1:] if v == "~" or v.startswith("~/") else v


BRACE_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")
GLOB_CHARS_RE = re.compile(r"[*?\[]")
UNSET_ROOT_RE = re.compile(r"^(?:\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*(?::-?)?\})+/\*?$")
MAX_EXPANSIONS = 64


def expand_braces(word):
    """Bash brace expansion over one token: /{a,b,c} -> ["/a","/b","/c"]. Capped."""
    out = [word]
    for _ in range(6):                      # nesting depth cap
        nxt = []
        grew = False
        for w in out:
            m = BRACE_RE.search(w)
            if not m:
                nxt.append(w)
                continue
            grew = True
            for alt in m.group(1).split(","):
                nxt.append(w[:m.start()] + alt + w[m.end():])
                if len(nxt) > MAX_EXPANSIONS:
                    return out + [word + "__BRACES_UNBOUNDED__"]
        out = nxt
        if not grew:
            break
    return out


def glob_targets(pattern, cwd):
    """Read-only glob against the live cwd -- the same matches bash would get. Capped."""
    base = norm(pattern, cwd)
    try:
        import glob as _glob
        hits = sorted(_glob.glob(base))[:256]
        if len(hits) >= 256:
            hits.append(base + "__GLOB_UNBOUNDED__")
        return hits or [base]
    except Exception:
        return [base]


def resolve_targets(args, cwd):
    """braces -> globs (live cwd, read-only) -> absolute paths. One target may become many."""
    out = []
    for a in args or []:
        for variant in expand_braces(a):
            if UNSET_ROOT_RE.match(variant):
                out.append("/")
            elif GLOB_CHARS_RE.search(variant) and cwd:
                out += [norm(h, cwd) for h in glob_targets(variant, cwd)]
            else:
                out.append(resolve(variant, cwd))
    return out


def resolve(path, cwd):
    """norm() that keeps unknown-cwd relatives recognisable."""
    if path and not is_abs(canon(tilde(path.replace("${HOME}", HOME).replace("$HOME", HOME)))) and cwd is None:
        return UNKNOWN_CWD + "/" + path
    return norm(path, cwd)


def redirect_targets(seg):
    """Unquoted > / >> targets in a segment (skips 2>&1 and >(...))."""
    out, q, i, n = [], None, 0, len(seg)
    while i < n:
        c = seg[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                i += 2
                continue
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == "\\":
            i += 2
            continue
        elif c == ">":
            j = i + 1
            if j < n and seg[j] == ">":
                j += 1
            if j < n and seg[j] in "&(":
                i = j
                continue
            while j < n and seg[j] == " ":
                j += 1
            m = re.match(r"""('[^']*'|"[^"]*"|[^\s;&|<>]+)""", seg[j:])
            if m:
                out.append(m.group(1).strip("'\""))
            i = j
            continue
        i += 1
    return out


WRITE_ALL_ARGS = {"tee", "touch", "truncate", "chmod", "chown", "chflags", "xattr", "mkfifo"}
WRITE_DEST_LAST = {"cp", "install", "rsync", "ln", "mv", "scp"}


def write_targets(prog, rest, flags, args, seg, cwd):
    """Paths this segment may create/overwrite (destination-aware: `cp a b` writes b, not a)."""
    out = list(redirect_targets(seg))
    if prog in WRITE_ALL_ARGS:
        out += args
    elif prog in ("sed", "gsed", "perl", "awk", "gawk", "mawk", "nawk", "ruby", "ed", "ex"):
        inplace = any(f == "--in-place" or f.startswith("--in-place=") or
                      (not f.startswith("--") and "i" in f[1:] and prog in ("sed", "gsed", "perl", "ruby")) or f == "-i"
                      for f in flags) or (prog in ("awk", "gawk", "mawk", "nawk") and "inplace" in " ".join(rest)) or prog in ("ed", "ex")
        if inplace:
            out += args
        if prog in ("sed", "gsed"):
            for r in rest:
                for m in re.finditer(r"(?:^|[;\n\s]|[0-9$,!]+)s(?:/(?:\\.|[^/])*/(?:\\.|[^/])*/[gpimI]*w\s+([^\s;'\"]+))"
                                     r"|(?:^|[;\n\s]|[0-9$,!]+)w\s+([^\s;'\"]+)", r):
                    out.append(m.group(1) or m.group(2))
        if prog in ("awk", "gawk", "mawk", "nawk"):
            for r in rest:
                for m in re.finditer(r'>>?\s*"([^"]+)"', r):
                    out.append(m.group(1))
    elif prog in WRITE_DEST_LAST:
        tv = None
        for k, t in enumerate(rest):
            if t in ("-t", "--target-directory") and k + 1 < len(rest):
                tv = rest[k + 1]
            elif t.startswith("--target-directory="):
                tv = t.split("=", 1)[1]
        dest = tv or (args[-1] if len(args) > 1 or (prog in ("ln",) and args) else None)
        if dest:
            out.append(dest)
    elif prog == "dd":
        out += [t[3:] for t in rest if t.startswith("of=")]
    elif prog == "find":
        for k, t in enumerate(rest[:-1]):
            if t in ("-fprint", "-fprint0", "-fprintf", "-fls"):
                out.append(rest[k + 1])
    elif prog in ("curl", "wget"):
        for k, t in enumerate(rest):
            if t in ("-o", "-O", "--output", "--output-document") and k + 1 < len(rest):
                out.append(rest[k + 1])
    elif prog == "sort":
        for k, t in enumerate(rest):
            if t in ("-o", "--output") and k + 1 < len(rest):
                out.append(rest[k + 1])
            elif t.startswith("--output="):
                out.append(t.split("=", 1)[1])
            elif t.startswith("-o") and len(t) > 2 and not t.startswith("--"):
                out.append(t[2:])
    elif prog == "history":
        for k, t in enumerate(rest):
            if t in ("-a", "-w") and k + 1 < len(rest):
                out.append(rest[k + 1])
    return [resolve(w, cwd) for w in out if w]


# A fork bomb is a function that pipes into itself in the background, then is called:
# `:(){ :|:& };:`, `bomb() { bomb | bomb & }; bomb`, `function f { f|f& }; f`.
FORK_NAME = r"[^\s(){}|&;<>]+"
FORK_BOMB_RE = re.compile(
    r"(?:^|[\s;&|(])(?:function\s+)?(" + FORK_NAME + r")\s*\(\s*\)\s*\{\s*\1\s*\|&?\s*\1\s*&\s*;?\s*\}\s*;?\s*\1(?=$|[\s;&|)])"
    r"|(?:^|[\s;&|(])function\s+(" + FORK_NAME + r")\s*\{\s*\2\s*\|&?\s*\2\s*&\s*;?\s*\}\s*;?\s*\2(?=$|[\s;&|)])")
PIPE_SINK_RE = re.compile(
    r"\|\s*(?:(?:sudo|env|command|nohup|timeout)\s+)*"
    r"(bash|sh|zsh|dash|ksh|fish|python3?|node|perl|ruby|osascript)\b")
B64_BLOB_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])")
HEX_BLOB_RE = re.compile(r"(?<![0-9a-fA-F])((?:[0-9a-fA-F]{2}){6,})(?![0-9a-fA-F])")


def decode_b64_text(cmd):
    """Best-effort decode of base64/hex/reversed/escaped pipelines so encoded commands are analyzed."""
    out = []
    has_b64_cmd = bool(re.search(r"\b(?:base64|b64decode|atob)\b", cmd))
    for m in B64_BLOB_RE.finditer(cmd):
        blob = m.group(1)
        if len(blob) < 24 and not has_b64_cmd:
            continue
        if len(blob) < 12 and "=" not in blob:
            continue
        try:
            import base64
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8", "replace")
        except Exception:
            continue
        if "\ufffd" in dec:
            continue
        ascii_cmd = all(32 <= ord(c) < 127 or c in "\n\r\t" for c in dec) and any(c in dec for c in (" ", "/", ">", ";", "|"))
        if any(c.isalpha() for c in dec) and (
                (has_b64_cmd and ascii_cmd) or
                any(k in dec for k in ("rm ", "mv ", "sh ", "bash", "curl", "wget", "eval", "python",
                                       "chmod", "dd ", "mkfs", "git ", "find ", "sed ", "truncate"))):
            out.append(dec)
    if re.search(r"\bxxd\s+-[a-zA-Z]*r|\bfromhex\b|\bunhexlify\b", cmd):
        for m in HEX_BLOB_RE.finditer(cmd):
            try:
                dec = bytes.fromhex(m.group(1)).decode("utf-8", "replace")
                if "\ufffd" not in dec and all(32 <= ord(c) < 127 or c in "\n\r\t" for c in dec) and any(c.isalpha() for c in dec):
                    out.append(dec)
            except Exception:
                pass
    masked = mask_quotes(cmd)
    if re.search(r"\|\s*rev\b", masked):
        for m in re.finditer(r"(?:echo|printf)\s+(['\"])(.*?)\1\s*\|\s*rev\b", cmd):
            out.append(m.group(2)[::-1])
    if PIPE_SINK_RE.search(masked):
        for m in re.finditer(r"(?:printf|echo\s+-e)\s+(['\"])((?:[^'\"\\]|\\.)*)\1\s*\|", cmd):
            if "\\x" in m.group(2) or re.search(r"\\[0-7]{2,3}", m.group(2)):
                out.append(decode_ansi_c(m.group(2)))
        for m in re.finditer(r"(?:echo|printf)\s+(['\"]?)([^\n|'\"]+)\1\s*\|\s*tr\s+-d\s+(['\"]?)([^\s|'\"]+)\3\s*\|", cmd):
            body_s, drop_s = m.group(2), m.group(4)
            out.append("".join(c for c in body_s if c not in drop_s))
    return out


ANSI_C_RE = re.compile(r"\$'((?:[^'\\]|\\.)*)'")


def decode_ansi_c(body):
    """Decode $'...' ANSI-C escapes: \\xHH \\uHHHH \\nnn \\e \\cX and standard escapes."""
    out, i, n = [], 0, len(body)
    simple = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v",
              "\\": "\\", "'": "'", '"': '"'}
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        e = body[i + 1]
        i += 2
        if e in simple:
            out.append(simple[e])
        elif e == "x":
            m = re.match(r"[0-9a-fA-F]{1,2}", body[i:])
            if m:
                out.append(chr(int(m.group(0), 16)))
                i += len(m.group(0))
        elif e == "u" or e == "U":
            w = 4 if e == "u" else 8
            m = re.match(r"[0-9a-fA-F]{1,%d}" % w, body[i:])
            if m:
                try:
                    out.append(chr(int(m.group(0), 16)))
                    i += len(m.group(0))
                except ValueError:
                    pass
        elif e == "c":
            if i < n:
                out.append(chr(ord(body[i].upper()) & 0x1f))
                i += 1
        elif e == "e":
            out.append("\x1b")
        elif e.isdigit():
            m = re.match(r"[0-7]{1,3}", body[i - 1:])
            if m:
                out.append(chr(int(m.group(0), 8) & 0xff))
                i += len(m.group(0)) - 1
        else:
            out.append(e)
    return "".join(out)


def decode_ansi_c_quotes(cmd):
    """Replace $'...' literals with their decoded, safely-quoted values (top gap from the taxonomy)."""
    def rep(m):
        try:
            return "'" + decode_ansi_c(m.group(1)).replace("'", "'\\''") + "'"
        except Exception:
            return "'__ANSI_C_UNDECODABLE__'"
    return ANSI_C_RE.sub(rep, cmd)


MAX_SCRIPT_BYTES = 262144
DB_CLIENT_PROGS = ("psql", "sqlite3", "mysql", "mariadb", "duckdb", "mongosh", "redis-cli", "supabase")
SQL_FILE_EXT = (".sql", ".psql", ".ddl")
# `< FILE` and `0< FILE`; never `<<`, `<<<`, `<(...)` or `<&`
DB_INPUT_RE = re.compile(r"(?:(?<=\s)|^)0?<(?![<(&>])\s*([^\s<>|&;()]+)")


def sql_file_text(path):
    """A SQL file's text when it is a local, regular, non-empty text file under MAX_SCRIPT_BYTES, else None."""
    try:
        st = os.stat(path)
        if not _stat.S_ISREG(st.st_mode) or st.st_size == 0 or st.st_size > MAX_SCRIPT_BYTES:
            return None
        with open(path, "rb") as f:
            raw = f.read(MAX_SCRIPT_BYTES)
    except Exception:
        return None
    return None if b"\0" in raw[:4096] else raw.decode("utf-8", "replace")


def db_input_sql(fed, rest, seg, ecwd):
    """The SQL a database client is handed besides its own arguments, so OP-007 judges it the same way:
    the words an echo or printf pipes in, the files a cat, tac, head or tail pipes in, `< FILE`,
    `-f FILE`, a positional .sql file, and `.read FILE` / `\\i FILE`. A file the gate cannot read
    (missing, binary, too large, or the working directory is unknown) is not judged."""
    texts, names = [], []
    if fed:
        words = []
        for w in fed[1]:
            if REDIRECT_TOKEN_RE.match(w):
                break
            words.append(w)
        if fed[0] in ("echo", "printf"):
            text = " ".join(w for w in words if not (w.startswith("-") and len(w) <= 3)).replace("\\n", "\n")
            if text.strip() and "__SUBST__" not in text:
                texts.append(text)
        elif fed[0] in ("cat", "tac", "head", "tail"):
            names += [w for w in words if not w.startswith("-")]
    for i, t in enumerate(rest):
        if REDIRECT_TOKEN_RE.match(t):
            break                                       # what follows `>` is an output file, not input
        if t in ("-f", "--file", "-init", "--init") and i + 1 < len(rest):
            names.append(rest[i + 1])
        elif t.startswith("--file="):
            names.append(t[7:])
        elif t.startswith("--init="):
            names.append(t[7:])
        elif t == "--init-command" and i + 1 < len(rest):
            texts.append(rest[i + 1])
        elif t.startswith("--init-command="):
            texts.append(t.split("=", 1)[1])
        elif not t.startswith("-") and t.lower().endswith(SQL_FILE_EXT):
            names.append(t)
    names += re.findall(r"(?:\.read|\\ir?)\s+([^\s'\";]+)", seg)
    m = DB_INPUT_RE.search(mask_quotes(seg))
    if m:
        names.append(seg[m.start(1):m.end(1)].strip("'\""))
    if ecwd is not None:
        seen = set()
        for n in names:
            p = resolve(n, ecwd)
            if p not in seen:
                seen.add(p)
                body = sql_file_text(p)
                if body:
                    texts.append(body)
    return "\n".join(texts)


SCRIPT_ARG_INTERPRETERS = ("bash", "sh", "zsh", "dash", "ksh", "fish",
                           "python", "python3", "python2", "perl", "ruby",
                           "node", "deno", "bun", "php", "lua", "source", ".")


def resolve_script_arg(rest, ecwd):
    """First plausible script-path token from interpreter args (post-expansion)."""
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in ("-c", "-e", "--eval", "-m") or (t.startswith("-") and not t.startswith("--") and t.endswith(("c", "e"))):
            return None
        if t in ("--import", "--loader", "--require", "-r") and i + 1 < len(rest):
            mod_t = rest[i + 1]
            if not mod_t.startswith("data:") and ("/" in mod_t or mod_t.endswith((".js", ".mjs", ".cjs", ".ts"))):
                return resolve(mod_t, ecwd) if ecwd else tilde(mod_t)
            i += 2
            continue
        elif re.match(r"^--(?:import|loader|require)=(.+)$", t):
            mod_t = t.split("=", 1)[1]
            if not mod_t.startswith("data:") and ("/" in mod_t or mod_t.endswith((".js", ".mjs", ".cjs", ".ts"))):
                return resolve(mod_t, ecwd) if ecwd else tilde(mod_t)
            i += 1
            continue
        if t.startswith("-"):
            i += 1
            continue
        if t in (".", ".."):
            i += 1
            continue
        if "/" in t or t.endswith((".sh", ".bash", ".py", ".pl", ".rb", ".js",
                                   ".mjs", ".cjs", ".ts", ".php", ".lua", ".zsh",
                                   ".ksh", ".fish")) or t.startswith("~"):
            return resolve(t, ecwd) if ecwd else tilde(t)
        # bare name: only treat as a script if the file exists beside cwd
        cand = resolve(t, ecwd) if ecwd else None
        if cand and os.path.isfile(cand):
            return cand
        return None
    return None


def is_gate_source(path):
    """True when the file is a copy of this gate (installed, repo checkout or delivery tree)."""
    try:
        with open(path, "r", errors="replace", encoding="utf-8") as f:
            head = f.read(6000)
    except Exception:
        return False
    return 'os.environ.get("OPERATOR_HOME"' in head and "VERSION = " in head


NON_SHELL_SCRIPT_EXT = (".py", ".pl", ".rb", ".js", ".mjs", ".cjs", ".ts", ".php", ".lua")
NON_SHELL_SHEBANG_RE = re.compile(r"^#!.*\b(python[0-9.]*|node|deno|bun|ruby|perl|php|lua)\b")


def script_body_effects(path, ecwd, depth, vars_, mode="auto", prog=None, seg=""):
    """Analyze an existing script file's body. Never executes it.

    mode="shell": the file is run by a shell (`bash x`, `source x`), walk it as shell lines.
    mode="auto": a shell file (by extension or shebang) is walked as shell; a Python, JS, Ruby,
    Perl, PHP or Lua file becomes ONE inline-code effect in that language, so its variable
    names and string literals can never be mistaken for shell programs (2026-10-03 phantom
    `rg` from `rg, _ = call(...)` inside a .py file), while its file-writing calls are still
    judged by the inline-code analysis."""
    try:
        st = os.stat(path)
        if not _stat.S_ISREG(st.st_mode) or st.st_size == 0 or st.st_size > MAX_SCRIPT_BYTES:
            return []
        with open(path, "r", errors="replace", encoding="utf-8") as f:
            body = f.read(MAX_SCRIPT_BYTES)
    except Exception:
        return []
    if not body.strip() or depth + 1 > MAX_DEPTH:
        return []
    base = os.path.basename(path)
    first = body.split("\n", 1)[0]
    prog_base = os.path.basename(prog or "").rstrip("0123456789.")
    non_shell = mode != "shell" and (
        path.endswith(NON_SHELL_SCRIPT_EXT)
        or bool(NON_SHELL_SHEBANG_RE.match(first))
        or prog_base in ("python", "perl", "ruby", "node", "deno", "bun", "php", "lua")
    )
    if non_shell:
        return [{"prog": prog or "script", "seg": seg or base, "targets": [], "recursive": False, "kind": "inline",
                 "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": ecwd or UNKNOWN_CWD,
                 "writes": [], "text": body, "script_body": base}]
    effs = analyze(body, ecwd, depth + 1, vars_)
    for e in effs:
        e["script_body"] = base
    return effs


def npm_script_effects(prog, rest, ecwd, depth, vars_):
    """Analyze package.json scripts invoked by `npm run <name>`, `npm test`, `pnpm/yarn/bun run <name>`."""
    if not ecwd or depth + 1 > MAX_DEPTH:
        return []
    pkg_cwd, pargs, i = ecwd, [], 0
    while i < len(rest):
        t = rest[i]
        if t in ("--prefix", "-C", "--cwd") and i + 1 < len(rest):
            pkg_cwd = resolve(rest[i + 1], pkg_cwd)
            i += 2; continue
        if t.startswith(("--prefix=", "--cwd=")):
            pkg_cwd = resolve(t.split("=", 1)[1], pkg_cwd)
            i += 1; continue
        if t == "--":
            break
        if not t.startswith("-"):
            pargs.append(t)
        i += 1
    sname = None
    if pargs[:1] in (["run"], ["run-script"]) and len(pargs) >= 2:
        sname = pargs[1]
    elif pargs[:1] in (["test"], ["t"], ["tst"], ["start"], ["stop"], ["restart"]):
        sname = "test" if pargs[0] in ("t", "tst") else pargs[0]
    elif prog in ("pnpm", "yarn") and pargs and pargs[0] not in (
            "add", "install", "i", "publish", "dlx", "exec", "x", "init", "create", "remove",
            "rm", "unlink", "link", "pack", "login", "logout", "version", "info", "why", "list",
            "ls", "outdated", "audit", "config", "cache"):
        sname = pargs[0]
    if not sname:
        return []
    raw = sql_file_text(pj(pkg_cwd, "package.json"))
    if not raw:
        return []
    try:
        scripts = (json.loads(raw) or {}).get("scripts") or {}
    except Exception:
        return []
    if not isinstance(scripts, dict):
        return []
    out = []
    for k in ("pre" + sname, sname, "post" + sname):
        body = scripts.get(k)
        if isinstance(body, str) and body.strip():
            for e in analyze(body, pkg_cwd, depth + 1, vars_):
                e.setdefault("script_body", "package.json:%s" % k)
                out.append(e)
    return out


def make_target_effects(rest, ecwd, depth, vars_):
    """Analyze Makefile recipes for the target(s) `make` / `gmake` would run."""
    if not ecwd or depth + 1 > MAX_DEPTH:
        return []
    make_cwd, mfile, targets, i = ecwd, None, [], 0
    while i < len(rest):
        t = rest[i]
        if t in ("-n", "--dry-run", "--just-print", "--recon"):
            return []
        if t in ("-C", "--directory") and i + 1 < len(rest):
            make_cwd = resolve(rest[i + 1], make_cwd); i += 2; continue
        if t.startswith("--directory="):
            make_cwd = resolve(t.split("=", 1)[1], make_cwd); i += 1; continue
        if t in ("-f", "--file", "--makefile") and i + 1 < len(rest):
            mfile = resolve(rest[i + 1], make_cwd); i += 2; continue
        if t.startswith(("--file=", "--makefile=")):
            mfile = resolve(t.split("=", 1)[1], make_cwd); i += 1; continue
        if t in ("-j", "-l", "-I", "-o", "-W") and i + 1 < len(rest):
            i += 2; continue
        if not t.startswith("-") and "=" not in t:
            targets.append(t)
        i += 1
    if not mfile:
        for cand in ("Makefile", "makefile", "GNUmakefile"):
            p = pj(make_cwd, cand)
            if os.path.isfile(p):
                mfile = p; break
    if not mfile:
        return []
    raw = sql_file_text(mfile)
    if not raw:
        return []
    mvars, recipes, deps, cur_targets, first_target = {}, {}, {}, [], None
    lines, buf = [], ""
    for line in raw.splitlines():
        if buf:
            buf = buf[:-1] + " " + line.lstrip()
        else:
            buf = line
        if not buf.endswith("\\"):
            lines.append(buf); buf = ""
    if buf:
        lines.append(buf)
    for line in lines:
        if line.startswith("\t") and cur_targets:
            rline = line.lstrip("\t").lstrip("@-+").strip()
            if rline and not rline.startswith("#"):
                for ct in cur_targets:
                    recipes.setdefault(ct, []).append(rline)
            continue
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped or stripped[0].isspace():
            continue
        vm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|\?=|\+=|=)\s*(.*)$", stripped)
        if vm:
            cur_targets = []
            mvars[vm.group(1)] = vm.group(2).strip()
            continue
        tm = re.match(r"^([A-Za-z0-9_./%+-][A-Za-z0-9_./%+ -]*?)\s*::?(?![=])\s*(.*)$", stripped)
        if tm:
            cur_targets = [x for x in tm.group(1).split() if x]
            if first_target is None:
                for ct in cur_targets:
                    if not ct.startswith("."):
                        first_target = ct; break
            rhs = tm.group(2)
            dep_part, semi, inline_r = rhs.partition(";")
            for ct in cur_targets:
                deps.setdefault(ct, []).extend(x for x in dep_part.split() if x)
                if semi and inline_r.strip():
                    recipes.setdefault(ct, []).append(inline_r.strip().lstrip("@-+").strip())
    want = targets or ([first_target] if first_target else [])
    seen, queue, out = set(), list(want), []
    base = os.path.basename(mfile)
    while queue:
        t = queue.pop(0)
        if not t or t in seen:
            continue
        seen.add(t)
        for d in deps.get(t, []):
            if d not in seen:
                queue.append(d)
        for rline in recipes.get(t, []):
            expanded = re.sub(r"\$(?:\(([A-Za-z_][A-Za-z0-9_]*)\)|\{([A-Za-z_][A-Za-z0-9_]*)\})",
                              lambda m: mvars.get(m.group(1) or m.group(2), m.group(0)), rline)
            for e in analyze(expanded, make_cwd, depth + 1, vars_):
                e.setdefault("script_body", "%s:%s" % (base, t))
                out.append(e)
    return out


FOR_LOOP_RE = re.compile(
    r"(?:^|[;&|\n]\s*)for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([^;\n]+?)\s*(?:;|\n)\s*do\b([\s\S]*?)(?:;|\n|\s)\bdone\b"
)
AWK_SYSTEM_RE = re.compile(r"\bsystem\s*\(\s*(['\"])(.*?)\1\s*\)")


# A pipe sink given one of these flags takes its program from the argument, so stdin is data:
# `| python3 -m json.tool`, `| node -e '...'`, `| perl -ne '...'`. Shells: any short flag with c.
CODE_ARG_FLAGS = {"python": r"-[A-Za-z]*[cm]", "python3": r"-[A-Za-z]*[cm]",
                  "node": r"-[A-Za-z]*[epc]\b|--(?:eval|print|check)\b", "perl": r"-[A-Za-z]*[eE]",
                  "ruby": r"-[A-Za-z]*[ec]", "osascript": r"-[A-Za-z]*e"}
FETCH_PROG = r"(?:sudo\s+(?:-\S+\s+)*)?(?:curl|wget|aria2c|fetch|https?|iwr|irm|Invoke-WebRequest|Invoke-RestMethod|dig|nslookup|host|drill)\b"
RUN_HEAD = r"(?:^|[\s;&|(])(?:sudo\s+(?:-\S+\s+)*)?(?:[\w./~-]*/)?"
# `bash <(curl X)`, `source <(curl X)`: read on the quote-masked command, where a quoted `<(` is text
PROC_SUBST_RUN_RE = re.compile(RUN_HEAD + r"(bash|sh|zsh|dash|ksh|fish|python[23]?|node|perl|ruby|source|\.)"
                               r"\s+(?:-\S+\s+)*<\(\s*" + FETCH_PROG)
# `sh -c "$(curl X)"`, `eval "$(wget -qO- X)"`, `python3 -c "$(curl X)"`: the head is read on the masked
# command, then the substitution that follows it on the real one (it expands inside double quotes)
CODE_SUBST_HEAD_RE = re.compile(RUN_HEAD + r"(?:(bash|sh|zsh|dash|ksh|fish)\s+(?:-\S+\s+)*?-\w*c\w*"
                                r"|(python[23]?|perl|ruby|node|osascript)\s+(?:-\S+\s+)*?-[A-Za-z]*[ce]|(eval))\s+\"?")
FETCH_SUBST_RE = re.compile(r"(?:\$\(|`)\s*" + FETCH_PROG)

EXEC_VARS = {"PAGER", "GIT_PAGER", "MANPAGER", "EDITOR", "VISUAL", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR",
             "GIT_SSH_COMMAND", "GIT_EXTERNAL_DIFF", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_PROXY_COMMAND",
             "LESSOPEN", "LESSCLOSE", "PROMPT_COMMAND", "KUBECTL_EXTERNAL_DIFF", "RUSTC_WRAPPER"}
GIT_EXEC_KEYS = ("core.pager", "core.editor", "core.sshcommand", "core.fsmonitor", "sequence.editor",
                 "diff.external", "credential.helper", "gpg.program", "uploadpack.packobjectshook")
GIT_HOOK_CONFIG_KEYS = ("core.fsmonitor", "core.hookspath", "uploadpack.packobjectshook")
SAFE_COMPRESSORS = {"gzip", "pigz", "bzip2", "pbzip2", "xz", "pxz", "zstd", "pzstd", "lz4", "lzop", "compress", "lzip"}
ARITH_SUBSCRIPT_CMDS = {"let", "declare", "typeset", "local", "readonly", "unset", "test", "[", "[["}


def find_command_substs(text, masked=None):
    """Yield (start, end, inner, top_level) for every $(...) and `...` in text across arbitrary nesting depth."""
    if masked is None:
        masked = mask_quotes(text, mask_double=False)
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i:i + 2] == "$(" and masked[i] != " ":
            if text[i:i + 3] == "$((":
                i += 3
                continue
            depth, j, q = 1, i + 2, None
            while j < n and depth > 0:
                c = text[j]
                if q:
                    if c == "\\" and q == '"' and j + 1 < n:
                        j += 2; continue
                    if c == q:
                        q = None
                elif c in "'\"":
                    q = c
                elif c == "\\" and j + 1 < n:
                    j += 2; continue
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = text[i + 2:j - 1]
                out.append((i, j, inner, True))
                for sub_s, sub_e, sub_in, _ in find_command_substs(inner):
                    out.append((i + 2 + sub_s, i + 2 + sub_e, sub_in, False))
                i = j
                continue
        elif text[i] == "`" and masked[i] != " ":
            j = i + 1
            while j < n and text[j] != "`":
                if text[j] == "\\" and j + 1 < n:
                    j += 2; continue
                j += 1
            if j < n:
                out.append((i, j + 1, text[i + 1:j], True))
                i = j + 1
                continue
        i += 1
    return out


def eval_trivial_subst(inner, vars_):
    """Evaluate trivial `echo ...` or `printf ...` command substitutions to their literal output, else None."""
    s = (inner or "").strip()
    if not s or any(c in s for c in ";|&><\n`") or "$(" in s:
        return None
    toks = tokenize(expand_vars(s, vars_))
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    rest = toks[1:]
    if prog == "echo":
        esc = False
        while rest and rest[0] in ("-n", "-e", "-E", "-ne", "-en"):
            if "e" in rest[0]:
                esc = True
            rest = rest[1:]
        val = " ".join(rest)
        if esc:
            val = decode_ansi_c(val)
        return val if re.match(r"^[A-Za-z0-9_./~ -]+$", val) else None
    if prog == "printf" and rest:
        if len(rest) == 1:
            val = decode_ansi_c(rest[0])
            return val if re.match(r"^[A-Za-z0-9_./~ -]+$", val) else None
        if len(rest) == 2 and rest[0] in ("%s", "%b"):
            val = decode_ansi_c(rest[1]) if rest[0] == "%b" else rest[1]
            return val if re.match(r"^[A-Za-z0-9_./~ -]+$", val) else None
    return None


def replace_command_substs(cmd, substs, vars_):
    """Replace top-level $(...) and `...` spans with trivial literal output when static, else __SUBST__."""
    tops = sorted([ item for item in substs if item[3] ], key=lambda x: x[0], reverse=True)
    out = cmd
    for s, e, inner, _ in tops:
        lit = eval_trivial_subst(inner, vars_)
        out = out[:s] + (lit if lit is not None else "__SUBST__") + out[e:]
    return SUBST_RE.sub("__SUBST__", out)


GATE_TAMPER_VARS = {"OPERATOR_HOME", "OPERATOR_GATE_MODE", "VESTIGE_DATA_DIR"}
# Prefixing these is a sandbox harness (`OPERATOR_HOME=$H python3 operator-gate.py`), not tamper.
_SANDBOX_PROGS = ("python", "python3", "node", "sh", "bash", "zsh", "vestige", "vestige-mcp",
                  "cargo", "npm", "npx", "echo", "mkdir", "cd", "for", "ls", "cat", "cp", "mv")
_GATE_DISABLE_MODES = ("off", "shadow", "disabled", "0", "false")
_DESTRUCTIVE_PROGS = ("rm", "rmdir", "mv", "shred", "dd", "truncate")


def _remember_custom_strata(val, ecwd, vars_):
    """Point later deletes at a command-local Strata dir without treating the assignment itself as tamper."""
    if not val:
        return
    r_vdir = resolve(val, ecwd)
    if r_vdir and not r_vdir.startswith(UNKNOWN_CWD) and "$" not in r_vdir and "__SUBST__" not in r_vdir:
        vars_["__CUSTOM_STRATA_DIR__"] = r_vdir


def _remaining_command(toks):
    """Tokens after leading env/export wrappers and NAME=value prefixes."""
    i = 0
    while i < len(toks):
        if re.match(r"^([A-Za-z_][A-Za-z0-9_%-]*)=(.*)$", toks[i], re.S):
            i += 1
            continue
        if os.path.basename(toks[i]) in ("env", "export"):
            i += 1
            continue
        break
    return toks[i:]


def _sandbox_invocation(rem_toks):
    """A test/interpreter/toolchain harness, or no command at all (a bare assignment)."""
    if not rem_toks:
        return True
    prog = os.path.basename(rem_toks[0])
    if prog in _SANDBOX_PROGS or prog.startswith("vestige-"):
        return True
    return any("gate" in a or "operator" in a or "test_" in a for a in rem_toks)


def _gate_mode_tamper(val):
    """OPERATOR_GATE_MODE is tamper only when set to a disabling value."""
    return (val or "").strip().lower() in _GATE_DISABLE_MODES


def _git_force_push(eff):
    if os.path.basename(str(eff.get("prog") or "")) != "git" or eff.get("sub") != "push":
        return False
    ga = eff.get("gargs") or []
    return any(a in ("--force", "-f", "--force-with-lease") or str(a).startswith("--force")
               or (str(a).startswith("+") and len(str(a)) > 1) for a in ga)


def _env_tamper_effect(name, val, ecwd, seg=""):
    return {"prog": "export", "seg": seg, "kind": "env_tamper", "var": name, "val": val, "target": val,
            "targets": [], "recursive": False, "filtered": False, "unresolved": False,
            "flags": [], "args": [], "cwd": ecwd or UNKNOWN_CWD, "writes": [], "text": seg or val}


def carried_by_vars(toks, ecwd, depth, vars_):
    """Analyze commands carried in execution environment variables (`GIT_SSH_COMMAND=... git fetch`),
    Shellshock function exports (`env x='() { :;}; cmd' bash -c :`, CVE-2014-6271), and gate/store
    environment variable tampering (`OPERATOR_HOME=...`, `OPERATOR_GATE_MODE=off`, `VESTIGE_DATA_DIR=...`).

    `VESTIGE_DATA_DIR` only records `__CUSTOM_STRATA_DIR__` so a later delete of that store is OP-002.
    `OPERATOR_GATE_MODE` is tamper only for a disabling value, and not when the remaining command is a
    test or sandbox runner. `OPERATOR_HOME` is tamper only for a non-runner command in the same segment.
    """
    out = []
    rem_toks = _remaining_command(toks)
    for t in toks:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_%-]*)=(.*)$", t, re.S)
        if not m:
            if os.path.basename(t) in ("env", "export"):
                continue
            break
        name, val = m.group(1), m.group(2).strip()
        if name == "VESTIGE_DATA_DIR" and val:
            _remember_custom_strata(val, ecwd, vars_)
        elif name == "OPERATOR_GATE_MODE" and _gate_mode_tamper(val) and not _sandbox_invocation(rem_toks):
            out.append(_env_tamper_effect(name, val, ecwd))
        elif name == "OPERATOR_HOME" and val and not _sandbox_invocation(rem_toks):
            out.append(_env_tamper_effect(name, val, ecwd))
        sm = re.match(r"^\(\s*\)\s*\{([^}]*)\}\s*;?\s*(.*)$", val, re.S)
        if sm:
            if sm.group(1).strip() and sm.group(1).strip() != ":":
                out += analyze(sm.group(1).strip(), ecwd, depth + 1, vars_)
            if sm.group(2).strip():
                out += analyze(sm.group(2).strip(), ecwd, depth + 1, vars_)
            continue
        if name in EXEC_VARS and val and val not in ("cat", "less", "more", "true", "false", ":"):
            out += analyze(val, ecwd, depth + 1, vars_)
    return out


def carried_by_git(rest, sub, gargs):
    """Return (commands_to_analyze, [(hook_key, hook_val)]) from a git invocation."""
    cmds, hooks = [], []
    i = 0
    while i < len(rest):
        t = rest[i]
        kv = None
        if t == "-c" and i + 1 < len(rest) and "=" in rest[i + 1]:
            kv = rest[i + 1]; i += 2
        elif t.startswith("-c") and len(t) > 2 and "=" in t[2:]:
            kv = t[2:]; i += 1
        elif re.match(r"^--(?:upload-pa\w*|receive-pa\w*|exec)=(.+)$", t):
            cmds.append(t.split("=", 1)[1]); i += 1; continue
        elif re.match(r"^--(?:upload-pa\w*|receive-pa\w*|exec)$", t) and i + 1 < len(rest):
            cmds.append(rest[i + 1]); i += 2; continue
        elif t.startswith("ext::") and len(t) > 5:
            raw_ext = re.sub(r"%[SsGV]\S*", "", t[5:]).strip()
            ext_toks = [w.replace("\x00", " ").strip() for w in raw_ext.replace("% ", "\x00").split() if w.strip()]
            if ext_toks:
                cmds.append(" ".join(shlex.quote(w) for w in ext_toks))
            i += 1; continue
        elif re.match(r"(?i)^ssh://[^/\s]*?-o(?:ProxyCommand|LocalCommand)=(.+)$", t):
            pm = re.match(r"(?i)^ssh://[^/\s]*?-o(?:ProxyCommand|LocalCommand)=(.+)$", t)
            cmds.append(pm.group(1).rstrip("/")); i += 1; continue
        else:
            i += 1
            continue
        key, val = kv.split("=", 1)
        kl = key.strip().lower()
        v = val.strip()
        if v.lower() in ("", "false", "0", "off", "none", "true", "cat", "less"):
            continue
        if kl in GIT_HOOK_CONFIG_KEYS or (kl.startswith("filter.") and kl.endswith((".clean", ".smudge", ".process"))):
            hooks.append((kl, v))
            cmds.append(v.lstrip("!"))
        elif kl in GIT_EXEC_KEYS or kl.startswith("pager.") or (kl.startswith("alias.") and v.startswith("!")):
            cmds.append(v.lstrip("!"))
    if sub == "config":
        if not any(f in ("--get", "--get-all", "--get-regexp", "--list", "-l", "--unset", "--unset-all") for f in gargs if f.startswith("-")):
            words = [a for a in gargs if not a.startswith("-")]
            if len(words) >= 2:
                kl, v = words[0].lower(), words[1].strip()
                if v.lower() not in ("", "false", "0", "off", "none", "true", "cat", "less"):
                    if (kl in GIT_HOOK_CONFIG_KEYS or kl in GIT_EXEC_KEYS or kl.startswith("pager.") or
                            (kl.startswith("alias.") and v.startswith("!")) or
                            (kl.startswith("filter.") and kl.endswith((".clean", ".smudge", ".process")))):
                        hooks.append((kl, v))
                        cmds.append(v.lstrip("!"))
    for j, a in enumerate(gargs):
        nxt = gargs[j + 1] if j + 1 < len(gargs) else None
        if sub in ("difftool", "mergetool") and a in ("-x", "--extcmd") and nxt:
            cmds.append(nxt)
        elif sub in ("difftool", "mergetool") and a.startswith("--extcmd="):
            cmds.append(a.split("=", 1)[1])
        elif sub == "rebase" and a in ("-x", "--exec") and nxt:
            cmds.append(nxt)
        elif sub == "rebase" and a.startswith("--exec="):
            cmds.append(a.split("=", 1)[1])
        elif sub == "filter-branch" and re.match(r"^--[a-z\-]+-filter$", a) and nxt:
            cmds.append(nxt)
    if sub == "bisect" and gargs[:1] == ["run"] and len(gargs) > 1:
        cmds.append(" ".join(shlex.quote(a) for a in gargs[1:]))
    if sub == "submodule" and "foreach" in gargs:
        tail = [a for a in gargs[gargs.index("foreach") + 1:] if a != "--recursive"]
        if tail:
            cmds.append(tail[0] if len(tail) == 1 else " ".join(shlex.quote(a) for a in tail))
    return cmds, hooks


def carried_by_archive_or_transfer(prog, rest):
    """Extract shell commands carried by tar/rsync/zip/scp/sftp/ssh/man/sort/vim execution flags."""
    cmds = []
    if prog in ("tar", "gtar", "bsdtar"):
        for i, t in enumerate(rest):
            m = re.match(r"^--checkpoint-action=exec=(.+)$", t)
            if m:
                cmds.append(m.group(1))
            elif t.startswith("--to-command="):
                cmds.append(t.split("=", 1)[1])
            elif t == "--to-command" and i + 1 < len(rest):
                cmds.append(rest[i + 1])
            elif t.startswith("--use-compress-program="):
                c = t.split("=", 1)[1]
                if c not in SAFE_COMPRESSORS:
                    cmds.append(c)
            elif t in ("-I", "--use-compress-program", "-F", "--info-script") and i + 1 < len(rest):
                c = rest[i + 1]
                if c not in SAFE_COMPRESSORS:
                    cmds.append(c)
            elif t.startswith("--info-script="):
                cmds.append(t.split("=", 1)[1])
    elif prog == "rsync":
        for i, t in enumerate(rest):
            if t in ("-e", "--rsh") and i + 1 < len(rest):
                cmds.append(rest[i + 1])
            elif t.startswith("--rsh="):
                cmds.append(t.split("=", 1)[1])
            elif t.startswith("-e") and len(t) > 2 and not t.startswith("--"):
                cmds.append(t[2:])
    elif prog == "zip":
        for i, t in enumerate(rest):
            if t in ("-TT", "--unzip-command") and i + 1 < len(rest):
                cmds.append(rest[i + 1])
            elif t.startswith("--unzip-command="):
                cmds.append(t.split("=", 1)[1])
    elif prog in ("scp", "sftp", "ssh"):
        for i, t in enumerate(rest):
            if prog in ("scp", "sftp") and t == "-S" and i + 1 < len(rest):
                cmds.append(rest[i + 1])
            elif t == "-o" and i + 1 < len(rest):
                m = re.match(r"(?i)^(?:ProxyCommand|LocalCommand)=(.+)$", rest[i + 1])
                if m:
                    cmds.append(m.group(1))
            elif t.startswith("-o") and len(t) > 2:
                m = re.match(r"(?i)^(?:ProxyCommand|LocalCommand)=(.+)$", t[2:])
                if m:
                    cmds.append(m.group(1))
    elif prog == "man":
        for i, t in enumerate(rest):
            if t in ("-P", "--pager", "-H", "--html") and i + 1 < len(rest):
                if rest[i + 1] not in ("cat", "less", "more"):
                    cmds.append(rest[i + 1])
            elif re.match(r"^--(?:pager|html)=(.+)$", t):
                c = t.split("=", 1)[1]
                if c not in ("cat", "less", "more"):
                    cmds.append(c)
    elif prog == "sort":
        for i, t in enumerate(rest):
            if t == "--compress-program" and i + 1 < len(rest):
                if rest[i + 1] not in SAFE_COMPRESSORS:
                    cmds.append(rest[i + 1])
            elif t.startswith("--compress-program="):
                c = t.split("=", 1)[1]
                if c not in SAFE_COMPRESSORS:
                    cmds.append(c)
    elif prog in ("vi", "vim", "nvim", "rvim", "ex", "view"):
        for i, t in enumerate(rest):
            ex_s = None
            if t == "-c" and i + 1 < len(rest):
                ex_s = rest[i + 1]
            elif t.startswith("+") and len(t) > 1:
                ex_s = t[1:]
            if ex_s:
                em = re.match(r"^:?!(?:silent\s+)?(.+)$", ex_s.strip())
                if em:
                    cmds.append(em.group(1))
                pm = re.match(r"^:?py(?:thon)?3?\s+(.+)$", ex_s.strip())
                if pm:
                    cmds.append("python3 -c " + shlex.quote(pm.group(1)))
    elif prog == "go":
        for i, t in enumerate(rest):
            if t == "-exec" and i + 1 < len(rest):
                cmds.append(rest[i + 1])
            elif t.startswith("-exec="):
                cmds.append(t.split("=", 1)[1])
    elif prog == "cargo":
        for i, t in enumerate(rest):
            cfg_v = None
            if t == "--config" and i + 1 < len(rest):
                cfg_v = rest[i + 1]
            elif t.startswith("--config="):
                cfg_v = t.split("=", 1)[1]
            if cfg_v:
                m = re.search(r'(?:rustc-wrapper|runner)\s*=\s*(?:\[([^\]]+)\]|["\']([^"\']+)["\']|(\S+))', cfg_v)
                if m:
                    if m.group(1):
                        parts = [w.strip().strip("'\"") for w in m.group(1).split(",") if w.strip()]
                        cmds.append(" ".join(parts))
                    else:
                        cmds.append((m.group(2) or m.group(3)).strip("'\""))
    elif prog in ("node", "bun", "deno"):
        for i, t in enumerate(rest):
            mod_v = None
            if t in ("--import", "--loader", "--require", "-r") and i + 1 < len(rest):
                mod_v = rest[i + 1]
            elif re.match(r"^--(?:import|loader|require)=(.+)$", t):
                mod_v = t.split("=", 1)[1]
            if mod_v and re.match(r"(?i)^data:(?:text|application)/javascript", mod_v):
                if "," in mod_v:
                    hdr, body_d = mod_v.split(",", 1)
                    try:
                        import base64 as _b64
                        js_code = _b64.b64decode(body_d).decode("utf-8", "replace") if ";base64" in hdr.lower() else urllib.parse.unquote(body_d)
                        if js_code.strip():
                            cmds.append("node -e " + shlex.quote(js_code))
                    except Exception:
                        pass
    elif prog == "tmux":
        # `tmux new-session -d 'cmd'`, `tmux split-window 'cmd'`, `tmux send-keys -t 0 'cmd' Enter`
        sub_i = next((idx for idx, tok in enumerate(rest) if not tok.startswith("-")), None)
        if sub_i is not None:
            tsub = rest[sub_i]
            tail = rest[sub_i + 1:]
            if tsub in ("new-session", "new", "split-window", "splitw", "respawn-pane", "respawnp", " respawn-window", " pipe-pane", "pipep"):
                j = 0
                while j < len(tail):
                    if tail[j] in ("-s", "-n", "-t", "-c", "-F", "-e", "-x", "-y") and j + 1 < len(tail):
                        j += 2
                    elif tail[j].startswith("-"):
                        j += 1
                    else:
                        break
                if j < len(tail):
                    cmds.append(tail[j] if len(tail[j:]) == 1 else " ".join(shlex.quote(x) for x in tail[j:]))
            elif tsub in ("send-keys", "send"):
                j = 0
                keys = []
                while j < len(tail):
                    if tail[j] in ("-t", "-N") and j + 1 < len(tail):
                        j += 2
                    elif tail[j].startswith("-"):
                        j += 1
                    else:
                        if tail[j] not in ("Enter", "C-m", "KPEnter"):
                            keys.append(tail[j])
                        j += 1
                if keys:
                    cmds.append(" ".join(keys))
    elif prog == "screen":
        # `screen -dmS name cmd args...` or `screen -d -m cmd args...`
        if any(f.startswith("-") and ("d" in f[1:] or "m" in f[1:]) for f in rest if f.startswith("-")):
            j = 0
            while j < len(rest):
                t = rest[j]
                if t in ("-S", "-t", "-c", "-L", "-Logfile", "-p", "-h", "-e") and j + 1 < len(rest):
                    j += 2
                elif t.startswith("-") and "S" in t[1:] and t.endswith("S") and j + 1 < len(rest):
                    j += 2
                elif t.startswith("-"):
                    j += 1
                else:
                    break
            if j < len(rest):
                cmds.append(rest[j] if len(rest[j:]) == 1 else " ".join(shlex.quote(x) for x in rest[j:]))
    return cmds


def db_shell_and_writes(prog, text, ecwd):
    """Extract shell-escape commands and file-write targets from sqlite3/duckdb/psql/mysql scripts."""
    cmds, writes = [], []
    if prog in ("sqlite3", "duckdb"):
        for m in re.finditer(r"(?:^|[\n;\"'])\s*\.(?:shell|system)\s+([^\n\"']+)", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"(?:^|[\n;\"'])\s*\.(?:output|once|import)\s+\|\s*([^\n\"']+)", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"(?:^|[\n;\"'])\s*\.(?:output|once|backup|save|clone)\s+(?!\|)([^\s;\"']+)", text):
            writes.append(resolve(m.group(1), ecwd))
    elif prog == "psql":
        for m in re.finditer(r"\\!\s+([^\n\"']+)", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"\\o\s+\|\s*([^\n\"']+)", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"(?i)\bCOPY\b[^;\n]*?\b(?:TO|FROM)\s+PROGRAM\s+'([^']+)'", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"(?i)\\copy\b[^\n]*?\b(?:to|from)\s+program\s+'([^']+)'", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"\\o\s+(?!\|)([^\s;\"']+)", text):
            writes.append(resolve(m.group(1), ecwd))
    elif prog in ("mysql", "mariadb"):
        for m in re.finditer(r"(?:\\!|(?:^|[\n;])\s*system)\s+([^\n;\"']+)", text):
            cmds.append(m.group(1).strip())
        for m in re.finditer(r"(?i)\bINTO\s+(?:OUTFILE|DUMPFILE)\s+['\"]([^'\"]+)['\"]", text):
            writes.append(resolve(m.group(1), ecwd))
    return cmds, writes


INLINE_SHELL_CALL_RE = re.compile(
    r"(?:\b(?:os\.(?:system|popen|execl|execlp|execv|execvp|spawnl|spawnlp)|"
    r"subprocess\.(?:run|call|check_call|check_output|Popen|getoutput|getstatusoutput)|"
    r"pty\.spawn|commands\.getoutput|child_process\.(?:execSync|exec|spawnSync|spawn|execFileSync|execFile)|"
    r"execSync|spawnSync|execFileSync|execFile|"
    r"Bun\.(?:spawnSync|spawn)|Deno\.Command|"
    r"Kernel\.(?:system|exec)|IO\.popen|os\.execute|io\.popen|"
    r"shell_exec|passthru|proc_open|popen|system|exec)|"
    r"\.\s*(?:execSync|exec|spawnSync|spawn|system|popen))\s*\("
)


def inline_shell_commands(text, prog):
    """Extract shell commands spawned inside an inline `-c`/`-e` interpreter snippet."""
    code = _code_of(text)
    if not code or not code.strip():
        return []
    out = []
    pbase = os.path.basename(prog or "").rstrip("0123456789.")
    if pbase == "python":
        try:
            import ast, warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(code)
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call) or not n.args:
                    continue
                fn = ""
                if isinstance(n.func, ast.Name):
                    fn = n.func.id
                elif isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name):
                        fn = n.func.value.id + "." + n.func.attr
                    elif isinstance(n.func.value, ast.Call) and isinstance(n.func.value.func, ast.Name) \
                            and n.func.value.func.id == "__import__" and n.func.value.args \
                            and isinstance(n.func.value.args[0], ast.Constant):
                        fn = str(n.func.value.args[0].value) + "." + n.func.attr
                if fn in ("os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.check_call",
                          "subprocess.check_output", "subprocess.Popen", "subprocess.getoutput",
                          "subprocess.getstatusoutput", "pty.spawn", "system", "popen", "run", "call",
                          "check_call", "check_output", "Popen"):
                    a0 = n.args[0]
                    if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                        out.append(a0.value)
                    elif isinstance(a0, (ast.List, ast.Tuple)):
                        words = []
                        for elt in a0.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                words.append(elt.value)
                            elif isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute) \
                                    and elt.func.attr == "expanduser" and elt.args \
                                    and isinstance(elt.args[0], ast.Constant) and isinstance(elt.args[0].value, str):
                                words.append(tilde(elt.args[0].value))
                        if words:
                            out.append(" ".join(shlex.quote(w) for w in words))
        except Exception:
            pass
    masked = _mask_code_strings(code)
    for m in INLINE_SHELL_CALL_RE.finditer(masked):
        open_idx = code.find("(", m.start())
        if open_idx < 0:
            continue
        args = _balanced_args(code, open_idx)
        if not args:
            continue
        a0 = args[0].strip()
        sm = re.match(r"^(['\"`])(.*)\1$", a0, re.S)
        if sm:
            cmd_s = sm.group(2)
            if len(args) >= 2 and args[1].strip().startswith("["):
                arr_items = re.findall(r"(['\"])(.*?)\1", args[1])
                if arr_items:
                    cmd_s = " ".join([cmd_s] + [shlex.quote(x[1]) for x in arr_items])
            if cmd_s not in out:
                out.append(cmd_s)
        elif a0.startswith("["):
            arr_items = re.findall(r"(['\"])(.*?)\1", a0)
            if arr_items:
                cmd_s = " ".join(shlex.quote(x[1]) for x in arr_items)
                if cmd_s not in out:
                    out.append(cmd_s)
    if pbase == "osascript":
        for m in re.finditer(r'do\s+shell\s+script\s+"((?:[^"\\]|\\.)*)"', code, re.I):
            out.append(re.sub(r'\\(.)', r'\1', m.group(1)))
    elif pbase in ("ruby", "perl", "php"):
        for m in re.finditer(r"(?<![\w$@])`([^`\n]{2,400})`", code):
            out.append(m.group(1))
        for m in re.finditer(r"\b(?:qx|%x)\s*([({\[/])(.*?)[)}\]/]", code):
            out.append(m.group(2))
        for m in re.finditer(r"(?m)^[ \t]*(?:system|exec)[ \t]+(['\"])(.*?)\1", code):
            out.append(m.group(2))
        for m in re.finditer(r"\bopen\s*\(\s*[^,)]+\s*,\s*(['\"])\s*(?:\|\s*([^'\"]+?)|([^'\"]+?)\s*\|)\s*\1\s*\)", code):
            out.append((m.group(2) or m.group(3)).strip())
    return out


KNOWN_EXEC_BINS = {"rm", "rmdir", "shred", "unlink", "dd", "mkfs", "sh", "bash", "zsh", "dash", "ksh", "fish",
                   "python", "python3", "perl", "ruby", "node", "curl", "wget", "git", "find", "sed", "awk",
                   "sqlite3", "psql", "mysql", "mv", "cp", "chmod", "chown"}

# ---- xargs fed a here-string: the command it runs is the command that runs ---- #
_XARGS_OPTS = r"((?:\s+-(?!I\b)[A-Za-z0-9-]+(?:[= ](?!-)\S+)?)*)"
XARGS_HERESTRING_RE = re.compile(r"\bxargs\b" + _XARGS_OPTS +
                                 r"\s+-I\s*(\S+)\s+([^|;&]+?)\s*<<<\s*('[^']*'|\"[^\"]*\"|[^\s;&|]+)")


def xargs_herestring_runs(cmd):
    """The commands `xargs -I R cmd <<< WORD` runs, with the here-string in place of R. The quoting of
    WORD is kept, so `"$HOME/x"` still expands and `'~/x'` stays literal. Heredoc bodies were extracted
    before this runs, and an xargs inside quotes is text, not the one that runs."""
    masked = mask_quotes(cmd)
    out = []
    for m in XARGS_HERESTRING_RE.finditer(cmd):
        if masked[m.start():m.start() + 5] != "xargs":
            continue
        ph, body, word = m.group(2), m.group(3), m.group(4)
        if word[0] in "'\"":
            words = [word]                                  # -I takes the whole line as one argument
        else:
            try:
                words = shlex.split(word)
            except ValueError:
                words = [word]
            words = [w if re.fullmatch(r"[\w@%+=:,./~${}-]+", w) else shlex.quote(w) for w in words[:8]]
        for w in words:
            out.append(body.replace(ph, w).strip())
    return list(dict.fromkeys(out))


def shell_noexec(rest):
    """`sh -n x.sh`, `bash -nv x.sh`: the shell reads the script's syntax and runs none of it."""
    for t in rest:
        if not t.startswith("-") or t == "--":
            return False
        if not t.startswith("--") and "n" in t[1:]:
            return True
    return False


SCRIPT_QUIET_PROGS = {"cat", "less", "more", "head", "tail", "bat", "grep", "egrep", "rg", "awk", "sed", "base64",
                      "xxd", "strings", "cut", "tee", "nl", "wc", "ls", "stat", "file", "shasum", "sha256sum", "md5",
                      "diff", "cmp", "cp", "git", "chmod", "shellcheck", "echo", "printf"}


def script_may_run(cmd, name, bodies=()):
    """False only when the command checks the script's syntax with `sh -n` and every other mention of it
    is a read: nothing in the command can run it. Any mention this cannot place counts as a run."""
    if any(name in b for b in bodies):
        return True
    checked = False
    for seg in split_segments(cmd or ""):
        if name not in seg:
            continue
        try:
            toks = tokenize(seg)
        except Exception:
            return True
        at = [i for i, t in enumerate(toks) if posixpath.basename(t) == name]
        if not at:
            return True                                     # named inside a string or a substitution
        prog = os.path.basename(toks[0])
        if prog in ("bash", "sh", "zsh", "dash", "ksh", "fish"):
            if shell_noexec(toks[1:]):
                checked = True
                continue
            return True
        if prog in SCRIPT_QUIET_PROGS and 0 not in at:
            continue
        return True
    return not checked


def analyze(cmd, cwd, depth=0, vars_=None):
    """Walk a shell command tracking cd and simple VAR=value, yielding effects (never executes anything)."""
    effects = []
    if depth > MAX_DEPTH or not cmd or not cmd.strip():
        return effects
    if vars_ is None:
        vars_ = {"HOME": HOME, "USERPROFILE": HOME, "PATH": "/usr/bin:/bin", "SHELL": "/bin/sh", "0": "sh"}
        if os.environ.get("TMPDIR"):
            vars_["TMPDIR"] = canon(os.environ["TMPDIR"].rstrip("/"))
    else:
        vars_ = dict(vars_)
    raw_vars = {}
    alias_bins = {}
    symlink_map = {}
    state = {"cwd": cwd}
    if INVISIBLE_RE.search(cmd):
        effects.append({"prog": "\u200b", "seg": cmd[:120], "targets": [], "recursive": False,
                        "kind": "invisible-chars", "filtered": False, "unresolved": False, "flags": [],
                        "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [], "text": cmd[:120]})
        cmd = INVISIBLE_RE.sub("", cmd)
    cmd = decode_ansi_c_quotes(cmd)
    cmd, bodies = extract_heredocs(cmd)
    # subshell groups: `(sleep 300; rm -rf ~) &` -- analyze the inside as its own command.
    # Groups inside quotes are data (a regex alternation, a message), never a subshell.
    for m in re.finditer(r"\(([^()]*)\)", mask_quotes(cmd)):
        inner = cmd[m.start(1):m.end(1)].strip()
        if m.start() and cmd[m.start() - 1] == "=":
            continue                                        # `name=(...)`, `name+=(...)`: an array literal
        if inner:
            effects += analyze(inner, state["cwd"], depth + 1, vars_)
    substs = find_command_substs(cmd, mask_quotes(cmd, mask_double=False))
    for _, _, inner, _ in substs:
        effects += analyze(inner, state["cwd"], depth + 1, vars_)
    # Process substitution `<(echo ...)` / `<(printf ...)` handed to a shell (`bash <(echo "rm -rf ~/Documents")`)
    masked = mask_quotes(cmd)
    for pm in re.finditer(RUN_HEAD + r"(bash|sh|zsh|dash|ksh|fish|source|\.)\s+(?:-\S+\s+)*<\(([^()]+)\)", masked):
        p_inner = cmd[pm.start(2):pm.end(2)].strip()
        lit = eval_trivial_subst(p_inner, vars_)
        if lit:
            effects += analyze(lit, state["cwd"], depth + 1, vars_)
    # pipe-to-interpreter: only when the interpreter reads its PROGRAM from the pipe
    # (`curl X | sh`). `curl X | python3 -c 'code'` pipes data, not code -- benign. A pipe inside
    # quotes is text (a commit message, a README line), so the scan reads the quote-masked command.
    for m in PIPE_SINK_RE.finditer(masked):
        tail = cmd[m.end():cmd.find("\n", m.end()) if "\n" in cmd[m.end():] else len(cmd)]
        tail = re.split(r"[|;&]", tail)[0]
        code_flag = CODE_ARG_FLAGS.get(m.group(1), r"-\w*c\w*\b")
        code_from_stdin = not re.search(r"\s(?:%s)|/dev/fd|<[)\w]" % code_flag, tail) and \
            not re.search(r"^\s+[\w./~$]", tail)
        if code_from_stdin:
            line_start = cmd.rfind("\n", 0, m.start()) + 1
            effects.append({"prog": m.group(1), "seg": m.group(0), "targets": [], "recursive": False,
                            "kind": "pipe-exec", "filtered": False, "unresolved": False, "flags": [],
                            "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [],
                            "text": cmd[line_start:m.end()]})
    # a download run as the program with no pipe: `bash <(curl X)`, `sh -c "$(curl X)"`, `eval "$(wget -qO- X)"`
    runs = [(m.group(1), m.start(), m.end()) for m in PROC_SUBST_RUN_RE.finditer(masked)]
    runs += [(m.group(1) or m.group(2) or m.group(3), m.start(), m.end()) for m in CODE_SUBST_HEAD_RE.finditer(masked)
             if FETCH_SUBST_RE.match(cmd, m.end())]
    for prog, s, end in runs:
        line_end = cmd.find("\n", end)
        effects.append({"prog": prog, "seg": cmd[s:end].strip(), "targets": [], "recursive": False,
                        "kind": "pipe-exec", "how": "run", "filtered": False, "unresolved": False, "flags": [],
                        "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [],
                        "text": cmd[cmd.rfind("\n", 0, s) + 1:line_end if line_end >= 0 else len(cmd)]})
    for decoded in decode_b64_text(cmd):
        effects += analyze(decoded, state["cwd"], depth + 1, vars_)
    for fm in FOR_LOOP_RE.finditer(masked):
        lvar = cmd[fm.start(1):fm.end(1)].strip()
        raw_items = tokenize(expand_vars(cmd[fm.start(2):fm.end(2)], vars_))
        lbody = cmd[fm.start(3):fm.end(3)].strip()
        items = []
        for it in raw_items:
            items += expand_braces(it)
        for val in items[:16]:
            if val and "$" not in val and "__SUBST__" not in val and lbody:
                effects += analyze(lbody, state["cwd"], depth + 1, dict(vars_, **{lvar: tilde(val)}))
    if FORK_BOMB_RE.search(masked):                        # quoted, it is text; `sh -c '...'` is walked as its own command
        effects.append({"prog": ":()", "seg": cmd[:200], "targets": [], "recursive": False, "kind": "fork-bomb",
                        "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": UNKNOWN_CWD,
                        "writes": [], "text": cmd[:200]})
    for run in xargs_herestring_runs(cmd):                 # `xargs -I {} rm -rf {} <<< ~/Documents`
        effects += analyze(run, state["cwd"], depth + 1, vars_)
    raw_segs = split_segments(cmd)
    cmd = replace_command_substs(cmd, substs, vars_)
    for seg in split_segments(cmd):                     # `: > f`, `> f`, `truncate -s 0 f`, `cp /dev/null f`
        m = TRUNCATE_RE.match(seg)
        if m:
            t = resolve(expand_vars(m.group("a") or m.group("b") or m.group("c"), vars_), state["cwd"])
            effects.append({"prog": "truncate", "seg": seg, "targets": [t], "recursive": False, "kind": "truncate",
                            "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": state["cwd"] or UNKNOWN_CWD,
                            "writes": [], "text": seg})
    hd, cursor, feeder = 0, 0, None
    for seg_idx, seg in enumerate(split_segments(cmd)):
        raw_seg = raw_segs[seg_idx] if seg_idx < len(raw_segs) else seg
        # Prompt expansion `${VAR@P}` (CVE-2026-29783): evaluates $VAR as a prompt with $(...) active
        for pm in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)@P\}", raw_seg):
            raw_v = raw_vars.get(pm.group(1)) or vars_.get(pm.group(1))
            if raw_v:
                effects += analyze(raw_v, state["cwd"], depth + 1, vars_)
        nh = len(HEREDOC_RE.findall(seg))
        my_bodies, hd = bodies[hd:hd + nh], hd + nh
        # Extract here-string (`<<< "cmd"` or `<<<cmd`) before tokenizing
        hs_bodies = []
        hs_m = re.search(r"<<<\s*(['\"]?)(.+?)\1(?:\s*$|\s+[;&|])", raw_seg)
        if hs_m:
            hs_bodies.append(expand_vars(hs_m.group(2), vars_))
            seg = seg[:hs_m.start()].strip()
        toks = tokenize(expand_vars(seg, vars_))
        # Re-check `${VAR@P}` after `expand_vars` in case `${VAR:=cmd}` assigned VAR in the same segment (CVE-2026-29783)
        for pm in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)@P\}", raw_seg):
            if pm.group(1) not in raw_vars and vars_.get(pm.group(1)):
                effects += analyze(vars_[pm.group(1)], state["cwd"], depth + 1, vars_)
        for _ in range(3):                                  # `if f() { rm ...`: keywords and function heads
            while toks and toks[0] in KEYWORDS:
                toks = toks[1:]
            toks = strip_function_head(toks)
        # Brace command expansion (`{rm,-rf,~/Documents}` or `{/bin/rm,-rf,~/Documents}`)
        if toks and re.match(r"^\{[^{}\s]+,[^{}\s]+\}$", toks[0]):
            parts_b = [tilde(p) for p in toks[0][1:-1].split(",") if p]
            if parts_b:
                toks = parts_b + toks[1:]
        # Arithmetic subscript evaluation in single/double quotes: `let 'a[$(cmd)]'`, `unset 'a[$(cmd)]'`, `test -v 'a[$(cmd)]'`
        if toks and (os.path.basename(toks[0]) in ARITH_SUBSCRIPT_CMDS or
                     os.path.basename(toks[0]) == "printf" and any(re.search(r"%[diuxXo]", t) for t in toks[1:])):
            for am in re.finditer(r"\[[^\]]*?(\$\([^()]+\)|`[^`]+`)[^\]]*?\]", raw_seg):
                sub_s = am.group(1)
                effects += analyze(sub_s[2:-1] if sub_s.startswith("$(") else sub_s[1:-1], state["cwd"], depth + 1, vars_)
        if toks and toks[0] == "for" and len(toks) >= 4 and toks[2] == "in":
            if "$" not in toks[3] and "__SUBST__" not in toks[3]:
                vars_[toks[1]] = tilde(toks[3])
            continue
        if not toks or toks[0] in LOOP_HEADS:
            continue
        j = 0
        while j < len(toks) and toks[j] in ("export", "local", "declare", "readonly", "typeset"):
            j += 1
        k = j
        while k < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", toks[k]):
            k += 1
        if k > j and k == len(toks):                       # pure assignment segment
            raw_toks_assign = tokenize(raw_seg)
            for idx_t, t in enumerate(toks[j:k]):
                name, val = t.split("=", 1)
                if j + idx_t < len(raw_toks_assign) and "=" in raw_toks_assign[j + idx_t]:
                    raw_vars[name.rstrip("+")] = raw_toks_assign[j + idx_t].split("=", 1)[1]
                if name.endswith("+"):                      # `name+=value` appends: the value is no longer known
                    vars_.pop(name[:-1], None)
                elif "__SUBST__" in val or "$" in val:
                    if name == "HOME":
                        vars_["HOME"] = "$UNRESOLVED_HOME"
                    else:
                        vars_.pop(name, None)
                else:
                    vars_[name] = tilde(val)
                    if name == "VESTIGE_DATA_DIR" and val.strip():
                        _remember_custom_strata(val.strip(), state["cwd"], vars_)
                    elif name == "OPERATOR_GATE_MODE" and _gate_mode_tamper(val):
                        effects.append(_env_tamper_effect(name, val.strip(), state["cwd"], seg))
            continue
        pre_toks = strip_leading_redirects(toks)
        effects += carried_by_vars(pre_toks, state["cwd"], depth, vars_)
        xrepl = xargs_repl_token(pre_toks)
        toks, via_xargs = strip_wrappers(pre_toks)
        toks = strip_leading_redirects(toks)
        if not toks:
            continue
        if re.match(r"^\{[^{}\s]+,[^{}\s]+\}$", toks[0]):
            parts_b = [tilde(p) for p in toks[0][1:-1].split(",") if p]
            if parts_b:
                toks = parts_b + toks[1:]
        prog_tok = toks[0]
        if "$" in prog_tok and "__SUBST__" not in prog_tok:
            stripped_p = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*", "", prog_tok)
            if os.path.basename(stripped_p) in KNOWN_EXEC_BINS:
                prog_tok = stripped_p
        if GLOB_CHARS_RE.search(prog_tok):
            prog_tok = re.sub(r"\[([^\[\]\\])\]", r"\1", prog_tok)
            if GLOB_CHARS_RE.search(prog_tok) and prog_tok.startswith("/"):
                try:
                    import glob as _glob
                    for gh in sorted(_glob.glob(prog_tok))[:16]:
                        if os.path.basename(gh) in KNOWN_EXEC_BINS:
                            prog_tok = gh
                            break
                except Exception:
                    pass
        prog = os.path.basename(prog_tok)
        if resolve(prog_tok, state["cwd"]) in alias_bins:
            prog = alias_bins[resolve(prog_tok, state["cwd"])]
        elif prog in alias_bins:
            prog = alias_bins[prog]
        rest = toks[1:]
        at = cmd.find(seg, cursor)                          # is this segment fed by a pipe from the one before?
        piped = at > 0 and re.search(r"(?<!\|)\|&?\s*$", cmd[:at]) is not None
        cursor = at + len(seg) if at >= 0 else cursor
        fed, feeder = (feeder if piped else None), (prog, rest, my_bodies)
        if piped and fed and len(fed) >= 3 and fed[0] in ("cat", "tee") and fed[2]:
            my_bodies = list(fed[2]) + list(my_bodies)
        my_bodies = list(my_bodies) + hs_bodies
        handed = cmd_exe_line(toks)
        if handed is not None:                              # `cmd /c "rmdir /s /q X"` from a shell
            effects += cmd_exe_effects(handed, state["cwd"])
            continue
        if "__SUBST__" in prog or "$" in prog:
            # dynamic command name: `$(echo rm) -rf x` / `$CMD ~/vestige` -- treat as a delete-shaped unknown
            dflags, dargs = flags_and_args(rest)
            effects.append({"prog": prog, "seg": seg, "targets": [resolve(a, state["cwd"]) for a in dargs],
                            "recursive": has_flag(dflags, "r") or has_flag(dflags, "R"), "kind": "delete",
                            "filtered": False, "unresolved": True, "flags": dflags, "args": dargs,
                            "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [], "text": seg})
            continue
        if prog in ("cd", "pushd"):
            tgt = rest[0] if rest and not rest[0].startswith("-") else (HOME if not rest else None)
            if tgt is None or "$" in tgt or "__SUBST__" in tgt:
                state["cwd"] = None
            else:
                state["cwd"] = resolve(tgt, state["cwd"])
                if state["cwd"].startswith(UNKNOWN_CWD):
                    state["cwd"] = None
            continue
        flags, args = flags_and_args(rest)
        if via_xargs and fed and fed[0] in ("echo", "printf"):
            fed_words = [w for w in fed[1] if not REDIRECT_TOKEN_RE.match(w) and not (w.startswith("-") and len(w) <= 3)]
            if fed[0] == "printf" and fed_words and "%" in fed_words[0]:
                fed_words = fed_words[1:]
            if fed_words and all("$" not in w and "__SUBST__" not in w for w in fed_words):
                if any(xrepl in a for a in args):
                    args = [fw if a == xrepl else a.replace(xrepl, fw) for a in args for fw in fed_words]
                else:
                    args = list(args) + fed_words
                via_xargs = False
        ecwd = state["cwd"]
        eff = {"prog": prog, "seg": seg, "targets": [], "recursive": False, "kind": None, "filtered": False,
               "unresolved": via_xargs, "flags": flags, "args": args, "rest": rest, "cwd": ecwd or UNKNOWN_CWD,
               "writes": write_targets(prog, rest, flags, args, expand_vars(seg, vars_), ecwd), "text": seg,
               "custom_strata_dir": vars_.get("__CUSTOM_STRATA_DIR__")}
        for sub_cmd in carried_by_archive_or_transfer(prog, rest):
            effects += analyze(sub_cmd, ecwd, depth + 1, vars_)

        if prog in ("su", "sg"):
            for jj, t in enumerate(rest):
                if t in ("-c", "--command") and jj + 1 < len(rest):
                    effects += analyze(rest[jj + 1], ecwd, depth + 1, vars_)
                    break
                if t.startswith("--command="):
                    effects += analyze(t.split("=", 1)[1], ecwd, depth + 1, vars_)
                    break
            continue
        if prog == "parallel" and ":::" in rest:
            idx_p = rest.index(":::")
            p_cmd = [w for w in rest[:idx_p] if not w.startswith("-")]
            p_args = [w for w in rest[idx_p + 1:] if w != ":::"]
            if p_cmd and p_args:
                effects += analyze(" ".join(p_cmd + p_args), ecwd, depth + 1, vars_)
            continue

        if prog in ("bash", "sh", "zsh", "dash", "ksh", "fish"):
            if eff["writes"]:
                effects.append(eff)
            inner_cmd = None
            for jj, t in enumerate(rest):
                if t in ("-c", "-lc", "-ic", "-cl") or (t.startswith("-") and t.endswith("c") and not t.startswith("--")):
                    if jj + 1 < len(rest):
                        inner_cmd = rest[jj + 1]
                    break
            if inner_cmd is not None:
                effects += analyze(inner_cmd, ecwd, depth + 1, vars_)
            for body in my_bodies:                          # `bash <<EOF` executes its body
                effects += analyze(body, ecwd, depth + 1, vars_)
            if piped and fed and fed[0] in ("echo", "printf"):
                fed_sh = decode_ansi_c(" ".join(w for w in fed[1] if not (w.startswith("-") and len(w) <= 3)))
                if fed_sh.strip():
                    effects += analyze(fed_sh, ecwd, depth + 1, vars_)
            # opaque-script resolution: `bash totally_harmless.sh` executes the FILE body as shell;
            # `bash -n x.sh` only checks its syntax, and the script is judged when something runs it
            sp = resolve_script_arg(rest, ecwd)
            if sp and (not shell_noexec(rest) or script_may_run(cmd, os.path.basename(sp), bodies)):
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="shell")
            continue
        if prog == "eval":
            if eff["writes"]:
                effects.append(eff)
            effects += analyze(" ".join(rest), ecwd, depth + 1, vars_)
            continue
        if prog == "trap":                                   # `trap 'body' EXIT` runs the body when the shell exits
            if eff["writes"]:
                effects.append(eff)
            ops = [a for a in rest if a != "--"]
            if ops and ops[0] not in ("-", "-p", "-l") and ops[0].strip():
                effects += analyze(ops[0], ecwd, depth + 1, vars_)
            continue

        if prog in ("source", "."):                          # `source x.sh` executes the file body
            if eff["writes"]:
                effects.append(eff)
            sp = resolve_script_arg(rest, ecwd)
            if sp:
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="shell")
            continue

        if prog in ("rm", "rmdir", "unlink", "shred", "trash", "rip", "srm") or \
                (prog == "gio" and rest[:1] == ["trash"]):
            eff["kind"] = "delete"
            eff["recursive"] = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags
            eff["targets"] = [t for t in resolve_targets([a for a in args if not (prog == "gio" and a == "trash")], ecwd)]
            if prog in ("trash", "rip", "gio") and any(os.path.isdir(t) for t in eff["targets"]):
                eff["recursive"] = True                     # sending a folder to the trash takes everything in it
            if prog in ("shred", "srm") or prog == "rm" and "-P" in flags:
                eff["unrecoverable"] = True
        elif prog in ("mv", "rename"):
            tv = [i for i, t in enumerate(rest) if t in ("-t", "--target-directory")]
            srcs = [a for a in args if a != rest[tv[0] + 1]] if tv else list(args)[:-1]
            dest = rest[tv[0] + 1] if tv else (args[-1] if args else None)
            if dest in ("/dev/null",):           # `mv x /dev/null` is a delete wearing a move's clothes
                eff["kind"] = "delete"
                eff["recursive"] = True
                eff["targets"] = resolve_targets(srcs, ecwd)
            else:
                eff["kind"] = "move"
                eff["targets"] = resolve_targets(srcs, ecwd)
        elif prog == "find":
            roots = []
            for a in rest:
                if a.startswith("-") or a in ("(", ")", "!"):
                    break
                roots.append(a)
            filtered_find = any(a in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-mtime",
                                      "-mmin", "-newer", "-size", "-user", "-perm", "-type", "-empty")
                                for a in rest)
            is_del_find = "-delete" in rest
            idx_f = 0
            while idx_f < len(rest):
                if rest[idx_f] in ("-exec", "-execdir", "-ok", "-okdir"):
                    end_f = idx_f + 1
                    while end_f < len(rest) and rest[end_f] not in (";", "\\;", "+"):
                        end_f += 1
                    ex_toks, _ = strip_wrappers(rest[idx_f + 1:end_f])
                    if ex_toks:
                        ex_prog = os.path.basename(ex_toks[0])
                        if ex_prog in ("rm", "rmdir", "shred", "unlink", "trash", "srm"):
                            is_del_find = True
                        else:
                            for r_root in (roots or ["."]):
                                sub_toks = [r_root if w == "{}" else w.replace("{}", r_root) for w in ex_toks]
                                effects += analyze(" ".join(shlex.quote(w) for w in sub_toks), ecwd, depth + 1, vars_)
                    idx_f = end_f + 1
                else:
                    idx_f += 1
            if is_del_find:
                eff["kind"] = "delete"
                eff["recursive"] = True
                eff["filtered"] = filtered_find
                eff["targets"] = resolve_targets(roots or ["."], ecwd)
        elif prog == "rsync" and any("--remove-source-files" in f or f == "--delete" for f in flags):
            eff["kind"] = "delete"
            eff["recursive"] = True
            eff["targets"] = [resolve(a, ecwd) for a in args[:-1]]
        elif prog == "git":
            eff["kind"] = "git"
            g = list(rest)
            gdir = None
            while g and g[0].startswith("-"):
                if g[0] == "-C" and len(g) > 1:              # git -C <dir>: the repository the command acts on
                    gdir = resolve(g[1], gdir or ecwd)
                g = g[2:] if g[0] in ("-C", "-c", "--git-dir", "--work-tree") else g[1:]
            eff["sub"] = g[0] if g else ""
            eff["gargs"] = g[1:]
            eff["gdir"] = gdir
            git_cmds, git_hooks = carried_by_git(rest, eff["sub"], eff["gargs"])
            eff["git_hooks"] = git_hooks
            for gcmd in git_cmds:
                effects += analyze(gcmd, gdir or ecwd, depth + 1, vars_)
        elif prog in INTERPRETERS:
            eff["kind"] = "inline"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
            sp = None if ("-" in rest and my_bodies) else resolve_script_arg(rest, ecwd)
            if sp and is_gate_source(sp):
                # `python3 .../operator-gate.py <subcommand>` is the gate itself. Judge it as the
                # operator-gate program (approve, install, onboard stay owner-only); never walk its source.
                eff["prog"] = "operator-gate"
                eff["kind"] = "cli"
                eff["args"] = [a for a in args if a != "-" and not a.endswith("operator-gate.py")]
                eff["text"] = seg
            elif sp:
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="auto", prog=prog, seg=seg)
            else:
                for sub_cmd in inline_shell_commands(eff["text"], prog):
                    effects += analyze(sub_cmd, ecwd, depth + 1, vars_)
        elif prog in ("grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack"):
            # `... | grep -v x` filters text already on the pipe; `grep -rn x src` and `rg x` read files.
            operands = args[1:]
            recursive = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags or prog in ("rg", "ripgrep", "ag", "ack")
            eff["kind"] = "filter" if not operands and not recursive else "read"
            if prog in ("rg", "ripgrep"):
                for jj, t in enumerate(rest):
                    if t == "--pre" and jj + 1 < len(rest):
                        eff["pre_cmd"] = rest[jj + 1]
                        effects += analyze(rest[jj + 1], ecwd, depth + 1, vars_)
                    elif t.startswith("--pre="):
                        eff["pre_cmd"] = t.split("=", 1)[1]
                        effects += analyze(eff["pre_cmd"], ecwd, depth + 1, vars_)
        elif prog in DB_CLIENT_PROGS:
            eff["kind"] = "db"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
            try:                                            # `echo SQL | psql`, `cat x.sql | sqlite3`, `< x.sql`, `-f x.sql`
                fed_sql = db_input_sql(fed, rest, seg, ecwd)
            except Exception:
                fed_sql = ""
            if fed_sql:
                eff["text"] += "\n" + fed_sql
            db_cmds, db_writes = db_shell_and_writes(prog, eff["text"], ecwd)
            eff["writes"] += db_writes
            for dcmd in db_cmds:
                effects += analyze(dcmd, ecwd, depth + 1, vars_)
        elif prog in ("cp", "ln") and len(args) >= 2:
            src_b = os.path.basename(args[-2])
            if src_b in KNOWN_EXEC_BINS:
                alias_bins[resolve(args[-1], ecwd)] = src_b
                alias_bins[os.path.basename(args[-1])] = src_b
            if prog == "ln" or has_flag(flags, "s") or "--symbolic-link" in flags:
                src_p = resolve(args[0], ecwd)
                dst_p = resolve(args[-1], ecwd)
                if not src_p.startswith(UNKNOWN_CWD) and not dst_p.startswith(UNKNOWN_CWD):
                    symlink_map[dst_p.rstrip("/")] = src_p.rstrip("/")
                    eff["symlink_src"] = src_p
                    eff["symlink_dst"] = dst_p
        elif prog in ("curl", "wget"):
            for jj, t in enumerate(rest):
                cfg_f = None
                if prog == "curl" and t in ("-K", "--config") and jj + 1 < len(rest):
                    cfg_f = rest[jj + 1]
                elif t.startswith("--config="):
                    cfg_f = t.split("=", 1)[1]
                elif prog == "curl" and t.startswith("-K") and len(t) > 2 and not t.startswith("--"):
                    cfg_f = t[2:]
                if cfg_f and ecwd:
                    cfg_body = sql_file_text(resolve(cfg_f, ecwd))
                    if cfg_body:
                        eff["text"] += "\n" + cfg_body
                        for cm in re.finditer(r'(?:^|\s)(?:-o|--output(?:-document)?|\boutput\s*=?)\s*["\']?([^\s"\']+)', cfg_body, re.M):
                            eff["writes"].append(resolve(cm.group(1), ecwd))
        elif prog == "sort":
            if piped and fed and fed[0] in ("echo", "printf") and any("compress-program" in t for t in rest):
                fed_txt = decode_ansi_c(" ".join(w for w in fed[1] if not w.startswith("-")))
                for fline in fed_txt.splitlines():
                    if fline.strip():
                        effects += analyze(fline.strip(), ecwd, depth + 1, vars_)
        elif prog in ("ed", "ex"):
            ed_lines = list(my_bodies)
            if piped and fed and fed[0] in ("echo", "printf"):
                ed_lines.append(decode_ansi_c(" ".join(w for w in fed[1] if not w.startswith("-"))))
            for el in "\n".join(ed_lines).splitlines():
                if el.strip().startswith("!"):
                    effects += analyze(el.strip()[1:], ecwd, depth + 1, vars_)
        elif prog in ("sed", "gsed"):
            for a in rest:
                for em in re.finditer(r"(?:^|[;\n]|[0-9$,!]+)e\s+([^\n;]+)", a):
                    effects += analyze(em.group(1), ecwd, depth + 1, vars_)
                for sm in re.finditer(r"s/(?:\\.|[^/])*/((?:\\.|[^/])*)/[gpimI]*e\b", a):
                    s_cmd = sm.group(1).replace("\\/", "/").strip()
                    if s_cmd:
                        effects += analyze(s_cmd, ecwd, depth + 1, vars_)
        elif prog in ("awk", "gawk", "mawk", "nawk"):
            eff["kind"] = "inline"
            for a in rest:
                for sm in AWK_SYSTEM_RE.finditer(a):
                    effects += analyze(sm.group(2), ecwd, depth + 1, vars_)
                for pm in re.finditer(r'"((?:[^"\\]|\\.)+)"\s*\|\s*getline\b|\|\s*"((?:[^"\\]|\\.)+)"', a):
                    effects += analyze(pm.group(1) or pm.group(2), ecwd, depth + 1, vars_)
        elif prog in ("make", "gmake"):
            eff["kind"] = "cli"
            effects += make_target_effects(rest, ecwd, depth, vars_)
        elif prog in ("fly", "flyctl", "stripe", "vercel", "wrangler", "netlify", "gh", "npm", "pnpm", "yarn",
                      "cargo", "twine", "docker", "vestige", "vestige-mcp", "operator-gate"):
            eff["kind"] = "cli"
            if prog in ("npm", "pnpm", "yarn"):
                effects += npm_script_effects(prog, rest, ecwd, depth, vars_)
                if "config" in args and "set" in args:
                    set_i = args.index("set")
                    if set_i + 1 < len(args):
                        ck = args[set_i + 1]
                        cv = args[set_i + 2] if set_i + 2 < len(args) else ""
                        if "=" in ck and not cv:
                            ck, cv = ck.split("=", 1)
                        if ck.lower() in ("script-shell", "scriptshell", "onload-script", "cafile") and cv:
                            eff["git_hooks"] = [(prog + "." + ck, cv)]
                            effects += analyze(cv, ecwd, depth + 1, vars_)
            if prog == "cargo":
                for t in rest:
                    if ("rustc-wrapper" in t or "runner" in t) and ("--config" in rest or t.startswith("--config=")):
                        eff["git_hooks"] = [("cargo.config", t)]
        elif prog == "openssl" and ("-engine" in rest or "engine" in rest):
            eff["git_hooks"] = [("openssl.engine", " ".join(rest))]
        # Substitute any in-command symlinks created earlier in the same command line (`ln -s ~/Documents ./d && rm -rf ./d`)
        if symlink_map and not eff.get("symlink_src"):
            for key in ("targets", "writes"):
                mapped = []
                for x in eff[key]:
                    xr = x.rstrip("/")
                    sub_x = x
                    for link_p, target_p in symlink_map.items():
                        if xr == link_p:
                            sub_x = target_p
                            eff["via_symlink"] = True
                            break
                        if xr.startswith(link_p + "/"):
                            sub_x = target_p + xr[len(link_p):]
                            eff["via_symlink"] = True
                            break
                    mapped.append(sub_x)
                eff[key] = mapped
        # anything that resolved into an unknown cwd cannot be judged statically
        for key in ("targets", "writes"):
            kept = [x for x in eff[key] if not x.startswith(UNKNOWN_CWD) and "$" not in x and "__SUBST__" not in x]
            if len(kept) != len(eff[key]):
                eff["unresolved"] = True
            eff[key] = kept
        effects.append(eff)
    return effects


# --------------------------------------------------------------------------- #
# PowerShell (the Windows shell tool): the delete cmdlets, read into the same effect model
# --------------------------------------------------------------------------- #
POWERSHELL_TOOLS = {"powershell", "pwsh"}
PS_DELETE = {"remove-item", "rm", "ri", "del", "erase", "rd", "rmdir"}
PS_ENV_RE = re.compile(r"\$\{?env:([A-Za-z_][A-Za-z0-9_]*)\}?", re.I)


def ps_statements(cmd):
    """PowerShell text split into statements on a newline, ; | and && outside quotes."""
    out, cur, q, i, n = [], [], None, 0, len(cmd)
    while i < n:
        c = cmd[i]
        if q:
            cur.append(c)
            if c == "`" and q == '"' and i + 1 < n:
                cur.append(cmd[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"":
            q = c; cur.append(c)
        elif c == "`" and i + 1 < n:
            cur.append(c); cur.append(cmd[i + 1]); i += 1
        elif c in "\n;|" or (c == "&" and i + 1 < n and cmd[i + 1] == "&"):
            out.append("".join(cur)); cur = []
            if c == "&":
                i += 1
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def ps_tokens(stmt):
    """PowerShell words: quotes group, a backtick escapes, and a backslash is an ordinary character."""
    toks, cur, q, i, n, started = [], [], None, 0, len(stmt), False
    while i < n:
        c = stmt[i]
        if q:
            if c == "`" and q == '"' and i + 1 < n:
                cur.append(stmt[i + 1]); i += 1
            elif c == q:
                q = None
            else:
                cur.append(c)
        elif c in "'\"":
            q = c; started = True
        elif c == "`" and i + 1 < n:
            cur.append(stmt[i + 1]); i += 1; started = True
        elif c.isspace():
            if cur or started:
                toks.append("".join(cur)); cur = []; started = False
        else:
            cur.append(c); started = True
        i += 1
    if cur or started:
        toks.append("".join(cur))
    return toks


def ps_expand(text):
    """$env:USERPROFILE, $env:HOME and the temp variables, as PowerShell would read them."""
    def env(m):
        name = m.group(1).upper()
        if name in ("USERPROFILE", "HOME"):
            return HOME
        if name in ("TEMP", "TMP"):
            return canon(tempfile.gettempdir())
        return m.group(0)
    return PS_ENV_RE.sub(env, text)


def ps_param(word, full):
    """PowerShell accepts any unambiguous prefix of a parameter name: -r, -rec and -Recurse are one."""
    low = word.lower()
    return len(low) >= 2 and full.startswith(low)


CMD_DELETE = {"rmdir", "rd", "del", "erase"}


def cmd_exe_effects(line, cwd):
    """Delete effects in a cmd.exe command line (what follows `cmd /c`). `&`, `&&` and `|` separate
    commands, double quotes group, and a one-letter `/x` word is a switch: `/s` makes it recursive."""
    effects = []
    for part in re.split(r"&&|&|\|", line or ""):
        toks = [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', part)]
        if not toks or toks[0].lower() not in CMD_DELETE:
            continue
        switches = [t.lower() for t in toks[1:] if re.match(r"^/[A-Za-z](:.*)?$", t)]
        paths = [t for t in toks[1:] if not re.match(r"^/[A-Za-z](:.*)?$", t)]
        resolved = resolve_targets(paths, cwd)
        kept = [t for t in resolved if not t.startswith(UNKNOWN_CWD) and "$" not in t and "%" not in t]
        effects.append({"prog": toks[0].lower(), "seg": part.strip(), "targets": kept, "recursive": "/s" in switches,
                        "kind": "delete", "filtered": False, "unresolved": len(kept) != len(resolved) or not paths,
                        "flags": switches, "args": paths, "cwd": cwd or UNKNOWN_CWD, "writes": [], "text": part.strip()})
    return effects


def cmd_exe_line(toks):
    """The command line handed to cmd.exe by `cmd /c ...` or `cmd.exe /k ...`, or None."""
    if not toks or os.path.basename(toks[0]).lower() not in ("cmd", "cmd.exe"):
        return None
    for i, t in enumerate(toks[1:], 1):
        if t.lower() in ("/c", "/k"):
            return " ".join(toks[i + 1:])
    return None


def analyze_powershell(cmd, cwd):
    """Effects of a PowerShell command. Remove-Item and its aliases become delete effects; every other
    statement goes through the shell walker, which reads git, npm and the like the same way."""
    effects = []
    for stmt in ps_statements(cmd or ""):
        toks = ps_tokens(ps_expand(stmt))
        while toks and toks[0] in ("&", "."):                # call operators
            toks = toks[1:]
        if not toks:
            continue
        handed = cmd_exe_line(toks)
        if handed is not None:                              # `cmd /c rmdir /s /q C:\\x` from PowerShell
            effects += cmd_exe_effects(handed, cwd)
            continue
        if toks[0].lower() not in PS_DELETE:
            effects += analyze(stmt, cwd)
            continue
        flags, paths, filtered, i = [], [], False, 1
        while i < len(toks):
            t = toks[i]
            if t.startswith("-") and len(t) > 1:
                flags.append(t)
                if ps_param(t, "-path") or ps_param(t, "-literalpath"):
                    if i + 1 < len(toks):
                        paths += [p for p in toks[i + 1].split(",") if p]
                        i += 1
                elif ps_param(t, "-filter") or ps_param(t, "-include") or ps_param(t, "-exclude"):
                    filtered = True
                    i += 1                                   # the value belongs to the parameter
            else:
                paths += [p for p in t.split(",") if p]
            i += 1
        if any(ps_param(f, "-whatif") for f in flags):       # a dry run deletes nothing
            continue
        resolved = resolve_targets(paths, cwd)
        kept = [t for t in resolved if not t.startswith(UNKNOWN_CWD) and "$" not in t]
        effects.append({"prog": "Remove-Item", "seg": stmt, "targets": kept,
                        "recursive": any(ps_param(f, "-recurse") for f in flags), "kind": "delete",
                        "filtered": filtered, "unresolved": len(kept) != len(resolved) or not paths,
                        "flags": flags, "args": paths, "cwd": cwd or UNKNOWN_CWD, "writes": [], "text": stmt})
    return effects


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def digest_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def scratch_roots():
    """Scratch prefixes beyond the fixed list: TMPDIR, and on Windows the temp directory in both its
    short and long spelling, compared without case."""
    roots = set()
    tmp = os.environ.get("TMPDIR", "").rstrip("/")
    if tmp:
        roots.add(tmp)
    if IS_WINDOWS:
        roots |= {v for v in variants(canon(tempfile.gettempdir()))}
        roots |= {b.casefold() for b in roots}
    return roots


def is_scratch(path):
    build = {b.casefold() for b in BUILD_DIRS} if IS_WINDOWS else BUILD_DIRS
    for t in variants(path):
        if any(t == s or t.startswith(s + "/") for s in SCRATCH_OK):
            return True
        if any(t == s or t.startswith(s + "/") for s in scratch_roots()):
            return True
        if posixpath.basename(t) in build:
            return True
    return False


def target_hits(kind, t, recursive, roots, gate_files, extra_strata=()):
    """Checks for one delete/move target."""
    hits = []
    for r in roots:
        if is_same_or_ancestor(t, r):
            if kind == "delete" and not recursive and os.path.isfile(t):
                continue
            rid = "OP-000" if touches_gate_home(t) else "OP-001"
            hits.append((rid, "%s of %s" % (kind, t),
                         "If this is a reorganisation, leave a symlink at the old path and ask the owner; "
                         "if it is cleanup, delete children explicitly, never the workspace root."))
            break
    hits += write_hits(t, gate_files, kind, extra_strata=extra_strata)
    if (os.path.basename(t) == ".git" or "/.git/" in t) and kind in ("delete", "move") \
            and not is_scratch(t):
        hits.append(("OP-004", "%s of %s" % (kind, t), "Do not remove or rewrite .git internals; ask the owner."))
    return hits


def rel_inside(target, root):
    """The part of `target` below `root`, or None when it is not below it."""
    for t in variants(target):
        for r in variants(root):
            r = r.rstrip("/")
            if t.startswith(r + "/"):
                return t[len(r) + 1:]
    return None


def memory_store_hit(w, verb, extra_strata=()):
    """Why touching `w` is OP-002, or None. A Strata store loses its history when its log, keys, marker or
    backups go, or when the data dir (or a folder holding it, short of HOME) is deleted or moved."""
    for d in list(STRATA_DIRS) + [x for x in extra_strata if x]:
        if verb in ("delete", "move") and is_same_or_ancestor(w, d) and not is_same_or_ancestor(w, HOME):
            return "%s of the Vestige store %s" % (verb, d)
        rel = rel_inside(w, d)
        if rel and STRATA_CRITICAL_RE.match(rel.rstrip("/")):
            return "%s Vestige store path %s" % (verb, w)
    if any(is_inside(w, d) for d in MEMORY_DIRS) and MEMORY_FILE_RE.search(w):
        return "%s memory store path %s" % (verb, w)
    for d in RECORD_DIRS:
        if is_inside(w, d) or (verb in ("delete", "move") and is_same_or_ancestor(w, d)
                               and not is_same_or_ancestor(w, HOME)):
            return "%s an agent's own record %s" % (verb, w)
    if any(same_path(w, f) for f in RECORD_FILES):
        return "%s shell history %s" % (verb, w)
    return None


def write_hits(w, gate_files, verb="write to", extra_strata=()):
    hits = []
    if touches_gate_home(w) or any(is_inside(w, f) for f in gate_files):
        hits.append(("OP-000", "%s gate/hook path %s" % (verb, w),
                     "Ask the owner to make this change, or to grant a one-time permit."))
    store = memory_store_hit(w, verb, extra_strata=extra_strata)
    if store:
        hits.append(("OP-002", store, RECORD_HINT if ("own record" in store or "shell history" in store) else
                     "Use `vestige backup` first; erasure is not something an agent decides."))
    base = w.rstrip("/").rsplit("/", 1)[-1]
    parent = w.rstrip("/").rsplit("/", 1)[-2] if w.rstrip("/").count("/") >= 1 else ""
    is_init = (base in SHELL_INIT_FILES or (base == "config.fish" and parent.endswith("fish"))
               or w.endswith("/.ssh/rc")
               or base in (".envrc", ".psqlrc", ".gdbinit", ".lldbinit", ".Rprofile")
               or ("/.git/hooks/" in w and not w.endswith(".sample"))
               or ("/.husky/" in w and not w.endswith(("_", ".gitignore")))
               or w.endswith("/.cargo/config.toml") or w.endswith("/.cargo/config")
               or any(w.startswith(d + "/") for d in ("/Library/LaunchAgents", "/Library/LaunchDaemons",
                                                      HOME + "/Library/LaunchAgents", "/etc/cron",
                                                      "/var/spool/cron", "/etc/systemd/system",
                                                      HOME + "/.config/systemd/user")))
    if is_init:
        hits.append(("OP-008", "%s shell init or persistent hook file %s" % (verb, w),
                     "Init and hook files execute automatically on future commands or shells; "
                     "propose the change to the owner instead of writing it."))
    if base.lower() in AGENT_CONFIG_NAMES or parent.endswith("/.cursor/rules") or "/.cursor/rules/" in w:
        hits.append(("OP-S14", "%s agent config %s" % (verb, w),
                     "Agent instruction files are pre-stage prompt injection (Nx/CurXecute); "
                     "propose the change to the owner instead."))
    return hits


NET_FETCH_RE = re.compile(r"\b(?:curl|wget|fetch|aria2c|http|https|iwr|irm|Invoke-WebRequest|Invoke-RestMethod|dig|nslookup|host|drill)\b|https?://")
REVSHELL_CODE_RE = re.compile(r"\bsocket\b|\bnet\.(?:Socket|connect|createConnection)\b|IO::Socket|TCPSocket")
DB_RESET_PROGS = {"npx", "pnpm", "pnpx", "yarn", "bunx", "bun", "prisma", "rails", "rake", "bin/rails", "python",
                  "python3", "php", "sequelize", "knex", "drizzle-kit", "dropdb", "mongosh", "mongo", "bundle",
                  "flyway", "alembic", "typeorm", "liquibase"}
DB_RESET_RE = re.compile(
    r"\bprisma\s+(?:migrate\s+reset|db\s+push\b[^\n;|&]*--force-reset)|\b(?:rails|rake)\s+db:(?:drop|reset|purge|schema:load)\b"
    r"|\bmanage\.py\s+(?:flush|reset_db)\b|\bartisan\s+(?:migrate:fresh|migrate:reset|db:wipe)\b"
    r"|\bsequelize(?:-cli)?\s+db:drop\b|\bknex\s+migrate:rollback\b[^\n;|&]*--all|\bdrizzle-kit\s+drop\b"
    r"|\bflyway\b[^\n;|&]*\bclean\b|\balembic\b[^\n;|&]*\bdowngrade\s+base\b"
    r"|\btypeorm\s+schema:drop\b|\bliquibase\b[^\n;|&]*\bdrop(?:-?all|All)\b"
    r"|^dropdb\b|\bdropDatabase\s*\(|\bdb\.[\w.]+\.drop\s*\(")
SYSTEM_DIRS = ("/", "/usr", "/etc", "/bin", "/sbin", "/System", "/Library", "/var", "/opt", "/private", "/Applications",
               "/Users", "/home", "/root", "/boot", "/lib", "/lib64")


def teardown_or_deploy(prog, args, flags):
    """Production deploys and infrastructure teardown, one command at a time."""
    a0, a = (args[:1] or [""])[0], args[:4]
    return (prog == "railway" and a0 in ("up", "deploy", "down", "delete")
            or prog == "netlify" and a0 == "deploy" and "--prod" in flags
            or prog in ("terraform", "tofu", "terragrunt") and a0 in ("apply", "destroy")
            or prog == "pulumi" and a0 in ("up", "update", "destroy")
            or prog in ("cdk", "serverless", "sls") and a0 in ("deploy", "destroy", "remove")
            or prog == "kubectl" and a0 in ("delete", "drain", "replace")
            or prog == "helm" and a0 in ("uninstall", "delete")
            or prog == "eksctl" and a0 == "delete"
            or prog == "heroku" and bool(re.match(r"^(?:apps|pg):(?:destroy|reset)$", a0))
            or prog == "aws" and (args[:2] == ["s3", "rb"] or args[:2] == ["s3", "rm"] and "--recursive" in flags
                                  or any(re.match(r"^(?:delete|terminate|remove)-", x) for x in args[1:2]))
            or prog == "gsutil" and (a0 == "rb" or a0 == "rm")
            or prog == "rclone" and a0 in ("purge", "delete") and "-n" not in flags
            or prog == "vercel" and a0 in ("rm", "remove")
            or prog == "firebase" and (bool(re.search(r":delete$", a0)) or a0 == "deploy")
            or prog in ("gcloud", "az", "doctl", "supabase") and "delete" in a)


def machine_wreck(prog, args, flags, ecwd):
    """Raw disk writes, formats, volume erases and recursive permission or owner changes on system or home folders."""
    if prog == "dd" and any(x.startswith("of=/dev/") for x in args):
        return "dd writes the raw device %s" % [x for x in args if x.startswith("of=/dev/")][0][3:]
    if re.match(r"^(?:mkfs(?:\..+)?|newfs(?:_\w+)?|wipefs|sfdisk|fdisk|gdisk|sgdisk|parted|mkswap)$", prog) \
            and any(x.startswith("/dev/") for x in args):
        return "%s rewrites the disk %s" % (prog, [x for x in args if x.startswith("/dev/")][0])
    if prog == "diskutil" and args and (re.match(r"^(?:erase\w*|zeroDisk|randomDisk|secureErase|partitionDisk|reformat)$", args[0])
                                        or args[0] == "apfs" and args[1:2] and args[1].startswith("delete")):
        return "diskutil %s erases a volume" % args[0]
    if prog in ("chmod", "chown", "chgrp") and (has_flag(flags, "R") or "--recursive" in flags):
        for x in args[1:]:
            t = resolve(x, ecwd)
            if t in SYSTEM_DIRS or same_path(t, HOME):
                return "%s -R on %s" % (prog, t)
    if prog in ("shutdown", "reboot", "halt", "poweroff") and "-c" not in flags \
            or prog in ("init", "telinit") and args[:1] in (["0"], ["6"]) \
            or prog == "systemctl" and args[:1] in (["poweroff"], ["reboot"], ["halt"]):
        return "%s halts or reboots the machine" % prog
    if prog == "kill" and "-1" in flags and (len(flags) > 1 or not args):
        return "kill -1 signals every process owned by the user"
    return None


def dirty_git_path(target, where):
    """First tracked file with uncommitted changes under target in where's git repo, or None."""
    try:
        r = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", where,
                            "status", "--porcelain", "--untracked-files=no", "--", target],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.splitlines()[0][3:].strip()
    except Exception:
        pass
    return None


def git_work_loss(sub, ga, where):
    """What a git reset/clean/checkout/restore/stash would destroy that git cannot bring back, or None."""
    def git(*a):
        # read-only queries; fsmonitor off, so the repository's config cannot make the gate run a program
        r = subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", where or "."] + list(a),
                           capture_output=True, text=True, timeout=2)
        return r.returncode, r.stdout
    try:
        if sub == "reset" and "--hard" in ga:
            rc, out = git("status", "--porcelain", "--untracked-files=no")
            if rc != 0:
                return None
            if out.strip():
                return "git reset --hard discards uncommitted changes to %s" % out.splitlines()[0][3:].strip()
            target = [x for x in ga if not x.startswith("-")]
            if target:
                rc, n = git("rev-list", "--count", "%s..HEAD" % target[0])
                if rc == 0 and n.strip().isdigit() and int(n) > 0:
                    return "git reset --hard %s drops %s commit(s) from this branch" % (target[0], n.strip())
        if sub == "clean" and any(re.match(r"^-[a-zA-Z]*f", x) or x == "--force" for x in ga):
            dry = ["-n"] + [re.sub("f", "", x) for x in ga if x.startswith("-") and x != "--force" and re.sub("f", "", x) != "-"]
            rc, out = git("clean", *(dry + [x for x in ga if not x.startswith("-")]))
            gone = [l[len("Would remove "):] for l in out.splitlines() if l.startswith("Would remove ")]
            kept = [g for g in gone if g.rstrip("/").split("/")[-1] not in BUILD_DIRS]
            if rc == 0 and kept:
                return "git clean deletes untracked files such as %s" % kept[0]
        if sub in ("checkout", "switch") and any(x in ("-f", "--force", "--discard-changes") for x in ga):
            rc, out = git("status", "--porcelain", "--untracked-files=no")
            if rc == 0 and out.strip():
                return "git %s -f discards uncommitted changes to %s" % (sub, out.splitlines()[0][3:].strip())
        if sub in ("checkout", "restore") and ("--" in ga or ga[-1:] == ["."]) or sub == "restore" and ga \
                and not ("--staged" in ga and "--worktree" not in ga):
            paths = ga[ga.index("--") + 1:] if "--" in ga else [x for x in ga if not x.startswith("-")]
            if paths:
                rc, out = git("status", "--porcelain", "--untracked-files=no", "--", *paths)
                if rc == 0 and out.strip():
                    return "git %s discards uncommitted changes to %s" % (sub, out.splitlines()[0][3:].strip())
        if sub == "stash" and ga[:1] in (["drop"], ["clear"]):
            rc, out = git("stash", "list")
            if rc == 0 and out.strip():
                return "git stash %s deletes saved work" % ga[0]
    except Exception:
        return None
    return None


CREDENTIAL_DIRS = (
    pj(HOME, ".ssh"),
    pj(HOME, ".gnupg"),
    pj(HOME, ".aws"),
    pj(HOME, ".kube"),
    pj(HOME, ".config", "gcloud"),
    pj(HOME, "Library", "Keychains"),
)
PRECIOUS_EXTS = {
    ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".key", ".pages", ".numbers",
    ".pdf", ".rtf", ".odt", ".ods", ".odp", ".jpg", ".jpeg", ".png", ".heic", ".gif",
    ".tiff", ".raw", ".mov", ".mp4", ".m4a", ".mp3", ".wav", ".psd", ".sketch", ".fig",
}


def is_precious_outside_file(path):
    """True when an existing non-empty file outside the project is under a credential folder
    or is a personal document or media file by extension."""
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return False
    if any(is_inside(path, d) for d in CREDENTIAL_DIRS):
        return True
    if is_inside(path, pj(HOME, "Library", "Application Support", "Claude")) \
            or re.search(r"/(?:fixtures?|__fixtures__|test_fixtures)(?:/|$)", path):
        return False
    _, ext = os.path.splitext(path)
    return ext.lower() in PRECIOUS_EXTS


def classify_effect(e, cfg, cwd):
    """Yield (rule_id, reason_detail, rewrite_hint) for one effect."""
    hits = []
    kind, text = e.get("kind"), e.get("text", "")
    roots = default_protected_roots() + [norm(p, cwd) for p in cfg["protected_roots"]]
    gate_files = self_protected_files() + [norm(p, cwd) for p in cfg["protected_files"]]
    ecwd = e.get("cwd") if e.get("cwd") and not e["cwd"].startswith(UNKNOWN_CWD) else None
    followed_inline = kind == "inline" and bool(e.get("script_body"))
    extra_strata = (e.get("custom_strata_dir"),) if e.get("custom_strata_dir") else ()

    for w in e.get("writes", []):
        hits += write_hits(w, gate_files, extra_strata=extra_strata)
        if is_precious_outside_file(w) and not is_scratch(w) and not (ecwd and is_inside(w, ecwd)) \
                and not any(h[0] in ("OP-000", "OP-001", "OP-002", "OP-008", "OP-011") for h in hits):
            hits.append(("OP-011", "overwrites %s, which is outside the project" % w,
                         "Copy it to a dated backup first, or ask the owner."))

    if kind == "fork-bomb":
        hits.append(("OP-014", "a fork bomb: it copies itself until the machine hangs",
                     "Nothing an agent builds needs this; ask the owner."))
    if kind == "pipe-exec":
        if NET_FETCH_RE.search(text):
            hits.append(("OP-018", "a download %s %s" % ("run straight by" if e.get("how") == "run" else
                                                        "piped straight into", e.get("prog")),
                         "Download to a file, read it, then run it deliberately; the owner reviews the script."))
        else:
            hits.append(("OP-S06", "code piped straight into %s" % e.get("prog"),
                         "Download to a file, read it, then run it deliberately; the owner reviews the script."))
    if kind == "truncate":
        for t in e["targets"]:
            store = memory_store_hit(t, "empty")          # `> ~/.zsh_history` names no program, only the file
            if store:
                hits.append(("OP-002", store, RECORD_HINT if ("own record" in store or "shell history" in store)
                             else "Use `vestige backup` first; erasure is not something an agent decides."))
            if is_precious_outside_file(t) and not is_scratch(t) and not (ecwd and is_inside(t, ecwd)):
                hits.append(("OP-011", "empties %s, which is outside the project" % t,
                             "Copy it to a dated backup first, or ask the owner."))

    if kind == "invisible-chars":
        hits.append(("OP-S12", "zero-width/Bidi control characters in the command",
                     "Legitimate commands never contain invisible characters; retype it plainly."))

    if e.get("pre_cmd"):
        hits.append(("OP-018", "rg --pre runs %s on matched files" % e["pre_cmd"],
                     "Run rg without --pre, or ask the owner."))
    if e.get("git_hooks"):
        hk, hv = e["git_hooks"][0]
        hits.append(("OP-008", "execution hook (%s=%s)" % (hk, hv),
                     "Hooks and compiler wrappers execute automatically on commands; ask the owner."))
    if e.get("kind") == "env_tamper":
        tvar, tval = e.get("var", ""), e.get("val", "")
        if tvar in ("OPERATOR_HOME", "OPERATOR_GATE_MODE"):
            hits.append(("OP-000", "%s=%s overrides the gate's own home or mode" % (tvar, tval),
                         "Never override OPERATOR_HOME or OPERATOR_GATE_MODE from an agent."))
        elif tvar == "VESTIGE_DATA_DIR":
            hits.append(("OP-002", "VESTIGE_DATA_DIR=%s redirects or touches the Vestige store" % tval,
                         "Do not override VESTIGE_DATA_DIR from an agent; use the configured store."))
    if e.get("symlink_src") and e.get("symlink_dst"):
        ssrc, sdst = e["symlink_src"], e["symlink_dst"]
        if touches_gate_home(ssrc) or any(is_inside(ssrc, f) for f in gate_files):
            hits.append(("OP-000", "symlink aliases gate path %s -> %s" % (sdst, ssrc),
                         "Do not create symlinks pointing into the gate directory."))
        elif any(is_same_or_ancestor(ssrc, d) or bool(rel_inside(ssrc, d) and STRATA_CRITICAL_RE.match(rel_inside(ssrc, d).rstrip("/")))
                 for d in STRATA_DIRS) and not same_path(ssrc, HOME):
            hits.append(("OP-002", "symlink aliases Vestige store path %s -> %s" % (sdst, ssrc),
                         "Do not create symlinks pointing into the Vestige store."))
        elif any(is_inside(ssrc, d) for d in CREDENTIAL_DIRS):
            hits.append(("OP-011", "symlink aliases credential path %s -> %s" % (sdst, ssrc),
                         "Do not create symlinks pointing into credential directories."))

    prog0, text0 = e.get("prog", ""), e.get("text", "")
    if not followed_inline and CONTAINER_ESCAPE_RE.search(text0):
        hits.append(("OP-S17", "container/VM escape primitive in command",
                     "Docker socket mounts, privileged containers and namespace tools are sandbox-break stages; ask the owner."))
    if not followed_inline and (
            "/dev/tcp/" in text0 or "/dev/udp/" in text0 or
            prog0 in ("nc", "ncat", "netcat") and any(f in ("-e", "-c") for f in e.get("flags", [])) or
            DNS_TUNNEL_RE.search(text0)):
        hits.append(("OP-009", "raw outbound shell/DNS channel",
                     "No agent task needs a reverse shell or DNS tunnel; ask the owner if connectivity is genuinely required."))
    if not followed_inline and REVSHELL_CODE_RE.search(text0) and \
            re.search(r"\.connect\s*\(|net\.(?:Socket|connect|createConnection)", text0) \
            and re.search(r"dup2|pty\.spawn|/bin/(?:ba|z)?sh\b|cmd\.exe|child_process|subprocess", text0):
        hits.append(("OP-009", "code opens a socket and hands it a shell",
                     "No agent task needs a reverse shell; ask the owner if connectivity is genuinely required."))
    if not followed_inline and CLOUD_METADATA_RE.search(text0):
        hits.append(("OP-010", "queries a cloud metadata credential endpoint",
                     "Metadata endpoints mint instance credentials; fetch nothing from 169.254.169.254-class addresses."))
    # one command, whatever kind of effect the walker gave it: teardown, resets, machine wreck, crontab -r
    flags0, args0 = e.get("flags", []), e.get("args", [])
    if "--dry-run" not in flags0 and teardown_or_deploy(prog0, args0, flags0):
        hits.append(("OP-006", "%s %s" % (prog0, " ".join(args0[:2])),
                     "Run the plan or a dry run first, then ask the owner for a one-time permit."))
    if not followed_inline and os.path.basename(prog0) in DB_RESET_PROGS and DB_RESET_RE.search(text0):
        hits.append(("OP-007", "database reset (%s)" % DB_RESET_RE.search(text0).group(0).strip(),
                     "Reset a throwaway database, or take a backup and ask the owner for a permit."))
    wreck = machine_wreck(prog0, args0, flags0, ecwd)
    if wreck:
        hits.append(("OP-014", wreck, "Nothing an agent builds needs this; ask the owner."))
    if prog0 == "crontab" and "-r" in flags0:
        hits.append(("OP-011", "crontab -r deletes every scheduled job",
                     "Edit the one entry with `crontab -e`, or ask the owner."))
    for w in e.get("writes", []):
        if "/.git/config" in w and GIT_ANCHOR_RE.search(text0):
            hits.append(("OP-S17", "git trust-anchor reconfiguration (%s)" % w,
                         "fsmonitor/hooksPath/filters execute outside every sandbox (GitSpawn); ask the owner."))
        if TRACE_TAMPER_RE.search(w):
            hits.append(("OP-S17", "tampering with execution traces (%s)" % w,
                         "Evidence deletion is a learned default under goal conflict; the log outlives the task."))
    if PASTE_TUNNEL_RE.search(text0) and prog0 in ("curl", "wget", "nc", "ncat", "git", "gh", "scp", "rsync", "ftp"):
        hits.append(("OP-S15", "sends data toward a paste/tunnel service",
                     "Exfil destinations are attacker-chosen; propose the share target to the owner first."))
    if prog0 == "git" and e.get("sub") in ("remote",) and any(a in ("add", "set-url") for a in e.get("gargs", [])):
        hits.append(("OP-S16", "git remote add/set-url", "New remotes are the documented exfil path; ask the owner."))
    if prog0 == "git" and e.get("sub") == "config" and GIT_ANCHOR_RE.search(" ".join(e.get("gargs", []))):
        hits.append(("OP-S17", "git trust-anchor reconfiguration (%s)" % " ".join(e.get("gargs", [])[:2]),
                     "fsmonitor/hooksPath/filters execute outside every sandbox (GitSpawn); ask the owner."))


    # persistence: crontab/at/launchctl/systemd/schtasks/tmux-detached
    if prog0 in PERSISTENCE_PROGS:
        sched_mutate = (
            (prog0 == "crontab" and not any(f in ("-l", "-r") for f in flags0)) or
            prog0 in ("at", "batch", "anacron") or
            (prog0 == "schtasks" and any(a.lower() in ("/create", "-create") for a in flags0 + args0)) or
            (prog0 == "launchctl" and any(a in ("load", "bootstrap", "submit", "enable") for a in args0)) or
            (prog0 == "systemctl" and any(a in ("enable", "start", "link", "reenable") for a in args0))
        )
        if sched_mutate:
            hits.append(("OP-008", "%s schedules or installs persistent background execution" % prog0,
                         "Scheduled or daemonized execution escapes the session audit window; ask the owner."))
        elif any(a in ("install", "new-session", "write", "-", "kickstart", "run") for a in args0):
            hits.append(("OP-S07", "%s schedules or installs code for later execution" % prog0,
                         "Scheduled execution escapes every audit window; propose it and let the owner run it."))

    # env-hijack: assignments or env-prefixes with loader/interpreter variable names
    toks_l = [t for t in (e.get("seg") or "").replace(";", " ").replace("&&", " ").split() if "=" in t]
    for t in toks_l:
        name = t.split("=", 1)[0]
        if name in HIJACK_VARS or name == "PATH":
            hits.append(("OP-S08", "%s= assignment mutates the execution environment" % name,
                         "Never mutate loader/interpreter/PATH variables; run binaries by absolute path instead."))
            break

    # exfil shapes: curl/wget posting/uploading file contents, esp. credential-shaped paths
    if prog0 in ("curl", "wget"):
        rest0 = e.get("rest") or (e.get("flags", []) + e.get("args", []))
        upload_paths = []
        for ii, a in enumerate(rest0):
            if a in ("-T", "--upload-file", "--post-file") and ii + 1 < len(rest0):
                upload_paths.append(rest0[ii + 1])
            elif a.startswith(("--upload-file=", "--post-file=")):
                upload_paths.append(a.split("=", 1)[1])
            elif a.startswith("-T") and len(a) > 2 and not a.startswith("--"):
                upload_paths.append(a[2:])
            elif a in ("-F", "--form", "--form-string", "-d", "--data", "--data-raw", "--data-binary", "--data-urlencode") and ii + 1 < len(rest0):
                v_up = rest0[ii + 1]
                if "@" in v_up:
                    upload_paths.append(v_up.split("@", 1)[1].split(";", 1)[0])
            elif a.startswith(("-F", "--form=", "-d@", "--data=", "--data-binary=", "--data-raw=", "--data-urlencode=")) and "@" in a:
                upload_paths.append(a.split("@", 1)[1].split(";", 1)[0])
            elif a.startswith("@") and len(a) > 1:
                upload_paths.append(a[1:])
        for path in upload_paths:
            if not path:
                continue
            rpath = resolve(path, ecwd)
            if any(is_inside(rpath, d) for d in CREDENTIAL_DIRS) or \
                    re.search(r"(?:^|/)(?:\.ssh/|\.aws/|\.gnupg/|id_(?:rsa|ed25519|ecdsa|dsa)|\.netrc$|\.npmrc$|\.pypirc$)", rpath, re.I) or \
                    rpath == pj(HOME, ".env"):
                hits.append(("OP-010", "uploads credential file %s to the network" % path,
                             "Never send credential or key files to remote endpoints."))
            if EXFIL_SECRET_PATH_RE.search(path) or EXFIL_SECRET_PATH_RE.search(rpath):
                hits.append(("OP-S11", "uploads %s to the network" % path,
                             "Never send file contents to remote endpoints; redact and ask the owner."))
                break
        if re.search(r"(?:-X|--request)\s*=?\s*DELETE\b", text0, re.I):
            if re.search(r"https?://(?:api\.(?:vercel|supabase|railway|fly|render|cloudflare|netlify|heroku|digitalocean|stripe)\.com|backboard\.railway\.com)", text0, re.I):
                hits.append(("OP-006", "%s DELETE against production cloud API" % prog0,
                             "Run the plan or a dry run first, then ask the owner for a one-time permit."))
            elif re.search(r"https?://api\.github\.com/repos/[^/\s'\"]+/[^/\s'\"]+(?:/releases)?(?:['\"\s]|$)", text0, re.I):
                hits.append(("OP-005", "%s DELETE against GitHub repository API" % prog0,
                             "Repository and release deletions are owner-only."))
        if re.search(r"\b(?:volumeDelete|projectDelete|serviceDelete|environmentDelete|databaseDelete|clusterDelete)\s*\(", text0):
            hits.append(("OP-006", "%s GraphQL infrastructure deletion mutation" % prog0,
                         "Infrastructure teardown mutations require a one-time permit from the owner."))

    if kind == "delete":
        hits += guarded(floor_hits, [], e, ecwd or cwd)

    if kind in ("delete", "move"):
        for t in e["targets"]:
            if e.get("filtered"):      # `find X -name ... -delete` never removes X itself
                hits += write_hits(t, gate_files, kind, extra_strata=extra_strata)
                if any(is_inside(t, d) for d in MEMORY_DIRS + list(extra_strata)):
                    hits.append(("OP-002", "filtered delete inside the memory store %s" % t,
                                 "Use `vestige backup` first; erasure is not something an agent decides."))
                elif not is_scratch(t) and (t == "/" or same_path(t, HOME) or t.count("/") <= 2 or ecwd is None
                                            or not is_inside(t, ecwd)):
                    hits.append(("OP-003", "deletes matching files under %s, outside the project" % t,
                                 "Scope the find to a named folder inside the project, or ask the owner."))
                elif any(is_same_or_ancestor(t, r) for r in roots):
                    hits.append(("OP-S01", "filtered mass delete inside workspace %s" % t,
                                 "Narrow the filter or delete named files; git protects committed work only."))
            else:
                hits += target_hits(kind, t, e["recursive"], roots, gate_files, extra_strata=extra_strata)
        if kind == "delete" and e["recursive"] and any(t in ("/",) for t in e["targets"]):
            hits.append(("OP-003", "recursive sweep rooted at /",
                         "Scope the find to a named project directory; never sweep from /."))
        if kind == "delete" and e["recursive"] and not e.get("filtered"):
            unsafe = [t for t in e["targets"] if not is_scratch(t)]
            broad = ecwd in (None, "/") or same_path(ecwd, HOME) or any(
                same_path(ecwd, pj(HOME, d)) for d in ("Developer", "Documents", "Downloads", "Desktop"))
            wide = [t for t in unsafe if t == "/" or same_path(t, HOME) or t.count("/") <= 2]
            outside = [t for t in unsafe if broad or not is_inside(t, ecwd)]
            working = [t for t in unsafe if ecwd and is_same_or_ancestor(t, ecwd)]
            repos = [t for t in unsafe if os.path.isdir(pj(t, ".git"))]
            known = e["targets"] and not (e["unresolved"] and e.get("prog") not in ("__SUBST__",) and "$" not in e.get("prog", "")
                                          and not e.get("prog", "").startswith("__SUBST__"))
            if (wide or outside or working) and known and \
                    not any(h[0] in ("OP-001", "OP-000", "OP-002") for h in hits):
                hits.append(("OP-003", "recursive delete of %s" % (wide or outside or working)[0],
                             "Move it to a dated folder under /tmp, or delete only build outputs "
                             "(target/, node_modules/, dist/)."))
            elif repos and not any(h[0] in ("OP-001", "OP-000", "OP-002", "OP-004") for h in hits):
                hits.append(("OP-004", "delete of the git repository %s" % repos[0],
                             "Delete named files inside it, or ask the owner."))
        if kind == "delete" and e.get("unrecoverable"):
            gone = [t for t in e["targets"] if not is_scratch(t)]
            if gone:
                hits.append(("OP-011", "%s destroys %s beyond recovery" % (e.get("prog"), gone[0]),
                             "Use rm on a copy you no longer need, or ask the owner."))
        if kind == "delete" and not e["recursive"] and not e.get("filtered"):
            for t in e["targets"]:
                if is_precious_outside_file(t) \
                        and not is_scratch(t) and not (ecwd and is_inside(t, ecwd)) \
                        and not any(h[0] in ("OP-000", "OP-001", "OP-002", "OP-003", "OP-004", "OP-011") for h in hits):
                    hits.append(("OP-011", "deletes %s, which is outside the project" % t,
                                 "Copy it to a dated backup first, or ask the owner."))
        if kind == "delete" and ecwd and not e.get("filtered"):
            for t in e["targets"]:
                if not is_scratch(t) and is_inside(t, ecwd) \
                        and not any(h[0] in ("OP-000", "OP-001", "OP-002", "OP-003", "OP-004", "OP-011") for h in hits):
                    dirty = dirty_git_path(t, ecwd)
                    if dirty:
                        hits.append(("OP-011", "deletes %s with uncommitted changes to %s" % (t, dirty),
                                     "Commit or stash the work first (`git stash`), or ask the owner."))
        if e["unresolved"] and kind in ("delete", "move") and not e["targets"]:
            hits.append(("OP-S05", "delete/move with targets from stdin", ""))

    elif kind == "git":
        sub, ga = e.get("sub", ""), e.get("gargs", [])
        force = any(a in ("--force", "-f", "--force-with-lease") or a.startswith("--force-with-lease=") or
                    (a.startswith("+") and len(a) > 1) for a in ga)
        if sub == "push" and force:
            protected = re.compile(r"(^|[:/+ ])(main|master|trunk|release[/\-].*|prod.*)$")
            refs = [a for a in ga if not a.startswith("-")]
            # The branch each refspec writes on the remote. HEAD, @ or no refspec at all push the branch
            # checked out where git runs (its -C directory, else the working directory).
            dsts, current = [], len(refs) < 2
            for spec in refs[1:]:
                dst = spec.lstrip("+").split(":", 1)[-1]
                if dst in ("HEAD", "@", ""):
                    current = True
                else:
                    dsts.append(dst)
            bad = any(protected.search(d) for d in dsts)
            if current and not bad:
                try:
                    r = subprocess.run(["git", "-C", e.get("gdir") or ecwd or cwd or ".", "symbolic-ref", "--short",
                                        "HEAD"], capture_output=True, text=True, timeout=1)
                    cur = r.stdout.strip() if r.returncode == 0 else ""
                    bad = not cur or bool(protected.search(cur))   # unreadable branch: fail closed
                except Exception:
                    bad = True
            if bad:
                hits.append(("OP-004", "force push to a shared branch", "Push a new branch and open a PR; "
                             "never rewrite main/master."))
        loss = git_work_loss(sub, ga, e.get("gdir") or ecwd or cwd)
        if loss:
            hits.append(("OP-011", loss, "Commit or stash the work first (`git stash`), or ask the owner."))
        elif sub == "reset" and "--hard" in ga or sub == "clean" and any(re.match(r"^-[a-z]*f", a) for a in ga) \
                or sub == "branch" and "-D" in ga or sub in ("stash",) and ga[:1] in (["drop"], ["clear"]) \
                or sub in ("checkout", "restore") and ga[-1:] in (["."], ["--", "."]):
            hits.append(("OP-S01", "git %s %s" % (sub, " ".join(ga[:3])), ""))
        if sub in ("push",) and (
                ("--delete" in ga or "-d" in ga) and any(re.search(r"(^|/)(main|master|trunk|release[/\-].*|prod.*)$", a) for a in ga)
                or any(re.match(r"^:(?:refs/\w+/)?(main|master|trunk|release[/\-].*|prod.*)$", a) for a in ga)):
            hits.append(("OP-004", "delete of remote main", "Ask the owner."))
        if sub == "reflog" and any(a in ("expire", "delete") for a in ga):
            hits.append(("OP-004", "git reflog %s erases recovery history" % ga[0], "Ask the owner."))

    elif kind in ("cli", "db", "inline"):
        prog, flags, args = e["prog"], e["flags"], e["args"]
        lo = text.lower()
        dry = "--dry-run" in flags or (prog in ("npm", "pnpm", "yarn") and "-n" in flags)
        draft = "--draft" in flags and args[1:2] == ["create"]
        if prog == "gh":
            if args[:1] == ["release"] and args[1:2] in (["create"], ["delete"], ["edit"], ["upload"]) and not draft or \
               args[:1] == ["repo"] and args[1:2] in (["delete"], ["archive"], ["rename"], ["edit"]) or \
               args[:1] == ["api"] and re.search(r"(?:-X|--method)\s*=?\s*(?:DELETE|PATCH)\b", text, re.I) and \
               ("/releases" in text or re.search(r"(?:^|[\s'\"])/?repos/[^/\s'\"]+/[^/\s'\"]+(?:['\"\s]|$)", text)):
                hits.append(("OP-005", "gh %s" % " ".join(args[:2]),
                             "Prepare the artifact and notes as a draft and hand them to the owner."))
            elif args[:1] in (["pr"], ["issue"], ["discussion"], ["gist"]) and \
                    args[1:2] in (["create"], ["edit"], ["comment"], ["close"], ["merge"], ["reopen"]):
                hits.append(("OP-S03", "gh %s" % " ".join(args[:2]), ""))
        if not dry and (prog in ("npm", "pnpm", "yarn") and args[:1] == ["publish"] or
                        prog == "cargo" and args[:1] == ["publish"] or prog == "twine" and args[:1] == ["upload"] or
                        prog == "docker" and args[:1] == ["push"]):
            hits.append(("OP-005", "%s %s" % (prog, args[0]), "Dry-run only (`--dry-run`) and hand off to the owner."))
        if not dry and prog in ("docker", "podman") and (
                args[:2] in (["volume", "rm"], ["volume", "remove"], ["volume", "prune"])
                or args[:2] == ["system", "prune"] and "--volumes" in flags
                or args[:2] == ["compose", "down"] and (has_flag(flags, "v") or "--volumes" in flags)):
            hits.append(("OP-007", "%s %s" % (prog, " ".join(args[:2])),
                         "Removing volumes destroys container data stores; back up first or ask the owner."))
        fly_groups = {"apps", "app", "secrets", "machine", "machines", "postgres", "pg", "volumes", "volume", "certs",
                      "ips", "regions", "autoscale", "mpg", "redis", "extensions", "tokens", "orgs"}
        fly_mut = {"destroy", "create", "set", "unset", "import", "update", "run", "stop", "start", "restart",
                   "suspend", "resume", "attach", "detach", "failover", "extend", "release", "allocate", "move",
                   "rename", "delete", "remove", "add", "kill", "clone", "fork", "deploy", "scale"}
        if prog in ("fly", "flyctl") and args and (args[0] in ("deploy", "launch", "scale") or
                                                   (args[0] in fly_groups and any(a in fly_mut for a in args[1:]))):
            hits.append(("OP-006", "fly %s" % " ".join(args[:2]),
                         "Ask the owner for a one-time permit for this deploy."))
        if prog == "supabase" and args[:2] in (["db", "push"], ["db", "reset"], ["migration", "up"],
                                                ["functions", "deploy"], ["db", "remote"]):
            hits.append(("OP-006", "supabase %s" % " ".join(args[:2]), "Apply it to staging, or ask the owner for a one-time permit."))
        if prog == "stripe" and re.search(r"--live|sk_live|rk_live", text):
            hits.append(("OP-006", "stripe live action", "Use test mode, or ask the owner for a one-time permit."))
        if prog in ("vercel",) and "--prod" in flags or prog == "wrangler" and args[:1] in (["deploy"], ["publish"]):
            hits.append(("OP-006", "%s production deploy" % prog, "Ask for a one-time permit."))
        if prog == "vestige" and args[:1] in (["gc"], ["purge"], ["wipe"], ["reset"], ["erase"]):
            hits.append(("OP-002", "vestige %s" % args[0], "Run `vestige backup` and ask the owner before erasing memory."))
        if prog == "operator-gate" and args[:1] in (["approve"], ["mode"], ["install"], ["uninstall"]):
            hits.append(("OP-000", "agent invoked `operator-gate %s`" % args[0],
                         "Approvals and installs are run by the owner in their own terminal."))
        if prog in ("psql", "sqlite3", "mysql", "mariadb", "duckdb", "supabase", "mongosh", "redis-cli"):
            sql = re.sub(r"pragma\s+wal_checkpoint\s*(\(\s*\w+\s*\))?", "", lo)
            sql = re.sub(r"insert\s+into\s+(\w+)\s*\(\s*\1\s*\)\s*values\s*\(\s*'[a-z\-]+'\s*\)", "", sql)  # fts5 control
            unscoped_update = any(not re.search(r"\bwhere\b", m.group(1))
                                  for m in re.finditer(r"\bupdate\s+[\w.\"`]+\s+set\b([^;]*)", sql))
            taut_where = bool(re.search(
                r"\b(?:delete\s+from|update)\s+[\w.\"`]+\b[^;]*?\bwhere\s+(?:true\b|1\s*=\s*1\b|'(\w*)'\s*=\s*'\1'|\"(\w*)\"\s*=\s*\"\2\")",
                sql))
            if re.search(r"\bdrop\s+(table|database|schema|index)\b|(?<![\w(])truncate\s+(table\s+)?[\w\"`]|"
                         r"\bflush(all|db)\b|\bdropdatabase\b|\bdb\.[\w.]+\.drop\s*\(", sql) \
                    or re.search(r"\bdelete\s+from\s+\w+\s*(;|\"|'|$)", sql) \
                    or unscoped_update or taut_where:
                hits.append(("OP-007", "destructive SQL", "Run it as a SELECT first, scope with WHERE, "
                             "and take a backup; ask the owner for a permit."))
            if prog == "sqlite3":
                dbs = [a for a in args if re.search(r"\.(db|sqlite3?|db3)$", a)] or args[:1]
                live = [d for d in dbs if any(is_inside(resolve(d, ecwd), m) for m in MEMORY_DIRS)]
                if live and re.search(r"\b(delete|drop|update|insert|alter|vacuum|replace|truncate)\b", sql):
                    hits.append(("OP-002", "write against the live Vestige store via sqlite3",
                                 "Work on a copy (`vestige backup`, then edit the copy); read-only queries are fine."))
        if e["kind"] == "inline":
            found, unresolved, mentions_gate = inline_file_calls(text, ecwd, roots, gate_files)
            if found:
                hits += [(rid, d if d.startswith("inline code") else "inline code: " + d, h) for rid, d, h in found]
            elif unresolved and mentions_gate and not e.get("script_body"):
                # a write or delete whose target the gate cannot resolve, in code that names a gate path:
                # fail closed, as before 0.3.5
                hits.append(("OP-000", "inline code has a %s whose target the gate cannot resolve, and names a gate path" % unresolved,
                             "Name the target as a plain literal so the gate can judge it, or ask the owner."))
            elif unresolved == "delete" and not e.get("script_body"):
                # an unresolved write in code that names no gate path is ordinary scripting, not a finding
                hits.append(("OP-S05", "inline code has a delete whose target the gate cannot resolve", ""))
    if hits and e.get("script_body"):
        hits = [(r, "%s -- inside script %s" % (d, e["script_body"]), a) for (r, d, a) in hits]
    return hits


INLINE_WRITE_CALL_RE = re.compile(
    r"\bopen\s*\(|\.write_text\s*\(|\.write_bytes\s*\(|\bwriteFile(?:Sync)?\s*\(|\bappendFile(?:Sync)?\s*\(|"
    r"\bcopyfile\s*\(|shutil\.copy2?\s*\(|fs\.(?:write|append|copy|truncate)\w*\s*\(|File\.write\s*\(|\btofile\s*\(|"
    r"\bos\.truncate\s*\(")
# __import__('os').remove(...) is os.remove(...) and require('fs').rmSync(...) is fs.rmSync(...): read the
# module call as its name before judging the code
INLINE_IMPORT_CALL_RE = re.compile(r"(?:__import__|\brequire)\(\s*(['\"])(?:node:)?([A-Za-z_][\w.]*)\1\s*\)")
INLINE_DELETE_CALL_RE = re.compile(
    r"\brmtree\s*\(|shutil\.move\s*\(|os\.(?:remove|unlink|rmdir|removedirs|rename|replace)\s*\(|\brmSync\s*\(|"
    r"\brimraf\s*\(|\bunlinkSync\s*\(|fs\.(?:rm|unlink|rename)\w*\s*\(|File\.delete\s*\(|FileUtils\.(?:rm|mv)\w*\s*\(|"
    r"\bremove_tree\s*\(|\.(?:unlink|rmdir)\s*\(|\bunlink\s*\(")
INLINE_RECURSIVE_RE = re.compile(r"rmtree|rmSync|rimraf|remove_tree|rm_r|rm_rf|rmdirSync|^fs\.rm\b|fs\.promises\.rm|FileUtils\.rm")
INLINE_MODE_RE = re.compile(r"^(?:mode\s*=\s*)?['\"](?:[wax][+bt]*|r[bt]*\+[bt]*)['\"]$")   # r+ writes too
INLINE_LITERAL_RE = re.compile(
    r"^(?:os\.path\.expanduser\(|Path\(|pathlib\.Path\()?\s*(['\"])(.*?)\1\s*\)?(?:\.(?:expanduser|resolve|absolute)\(\s*\))*$")
INLINE_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*((?:os\.path\.expanduser\(|Path\(|pathlib\.Path\()?\s*(['\"]).*?\3\s*\)?(?:\.(?:expanduser|resolve|absolute)\(\s*\))*)\s*$", re.M)


def _balanced_args(text, open_idx):
    """Split the argument list of the call whose '(' is at open_idx into top-level args."""
    depth, i, n, args, cur, q = 0, open_idx, len(text), [], [], None
    while i < n:
        c = text[i]
        if q:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(text[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"":
            q = c; cur.append(c)
        elif c in "([{":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip()); return [a for a in args if a != ""]
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    return None                                              # unbalanced: unknown


HOME_EXPR_RES = (
    re.compile(r"^(?:require\(\s*['\"]os['\"]\s*\)|os)\.homedir\(\)\s*\+\s*(['\"])(/[^'\"]*)\1$"),
    re.compile(r"^(?:process\.env\.HOME|os\.environ\[\s*['\"]HOME['\"]\s*\]|os\.getenv\(\s*['\"]HOME['\"]\s*\))\s*\+\s*(['\"])(/[^'\"]*)\1$"),
    re.compile(r"^\$ENV\{\s*['\"]?HOME['\"]?\s*\}\s*\.\s*(['\"])(/[^'\"]*)\1$"),
    re.compile(r'^"\$ENV\{\s*HOME\s*\}(/[^"]*)"$'),
    re.compile(r"^(?:ENV\[\s*['\"]HOME['\"]\s*\]|Dir\.home(?:\(\))?)\s*\+\s*(['\"])(/[^'\"]*)\1$"),
)
HOME_JOIN_RES = (
    re.compile(r"^(?:path\.)?(?:join|resolve)\(\s*(?:require\(\s*['\"]os['\"]\s*\)|os)\.homedir\(\)\s*,\s*(['\"])([^'\"]*)\1\s*\)$"),
    re.compile(r"^os\.path\.join\(\s*os\.path\.expanduser\(\s*(['\"])~\1\s*\)\s*,\s*(['\"])([^'\"]*)\2\s*\)$"),
    re.compile(r"^(?:pathlib\.)?Path\.home\(\)\s*/\s*(['\"])([^'\"]*)\1$"),
)


def _literal_path(expr, literals):
    expr = re.sub(r"(?:\.(?:expanduser|resolve|absolute)\(\s*\))+$", "", (expr or "").strip()).strip()
    for r in HOME_EXPR_RES:
        m = r.match(expr)
        if m:
            return "~" + m.group(2)
    for r in HOME_JOIN_RES:
        m = r.match(expr)
        if m:
            return "~/" + m.group(m.lastindex).lstrip("/")
    m = INLINE_LITERAL_RE.match(expr)
    if m:
        return m.group(2)
    if re.match(r"^[A-Za-z_]\w*$", expr) and expr in literals:
        return literals[expr]
    return None


TEMPFILE_CALL_RE = re.compile(r"^tempfile\.(?:mkdtemp|mkstemp|TemporaryDirectory|NamedTemporaryFile|gettempdir)\s*\(")
ANY_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+?)\s*$", re.M)


def _temp_expr(expr, temp):
    """True when expr is a fresh temp folder from tempfile, a name built only from one, or
    os.path.join / dirname / abspath of one with plain relative literal parts."""
    e = (expr or "").strip()
    if TEMPFILE_CALL_RE.match(e) or e in temp:
        return True
    m = re.match(r"^os\.path\.(join|dirname|abspath|realpath)\(", e)
    if not m:
        return False
    args = _balanced_args(e, m.end() - 1)
    if not args or not _temp_expr(args[0], temp):
        return False
    for a in args[1:]:
        lit = re.match(r"^(['\"])([^'\"]*)\1$", a)
        if m.group(1) != "join" or not lit or lit.group(2).startswith(("/", "~", "\\")) or ".." in lit.group(2):
            return False
    return True


def _code_of(text):
    """The program text: the -c/-e argument or heredoc body of an interpreter call, or the text itself (a script body)."""
    first, sep, rest_lines = text.partition("\n")
    if sep and "__HEREDOC_" in first:
        return rest_lines
    try:
        toks = shlex.split(text, posix=True)
    except ValueError:
        return text
    if toks and os.path.basename(toks[0]).rstrip("0123456789.") in INTERPRETERS:
        for i, t in enumerate(toks[1:-1], 1):
            if t in ("-c", "-e", "--eval") or (t.startswith("-") and not t.startswith("--") and t.endswith(("c", "e"))):
                return toks[i + 1]
    return text


def _mask_code_strings(code):
    """Same length as code, with the interior of string literals (including triple-quoted strings)
    and # / // / /* */ comments replaced by spaces, so string constants (test cases, docs, examples)
    are never read as file-writing or file-deleting calls."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if c == "#" or code[i:i + 2] == "//":
            while i < n and code[i] != "\n":
                out.append(" "); i += 1
            continue
        if code[i:i + 2] == "/*":
            out.append(" "); out.append(" "); i += 2
            while i < n and code[i:i + 2] != "*/":
                out.append("\n" if code[i] == "\n" else " "); i += 1
            if i < n:
                out.append(" "); out.append(" "); i += 2
            continue
        if code[i:i + 3] in ('"""', "'''"):
            q3 = code[i:i + 3]
            out.append(q3); i += 3
            while i < n and code[i:i + 3] != q3:
                if code[i] == "\\" and i + 1 < n:
                    out.append(" "); out.append("\n" if code[i + 1] == "\n" else " "); i += 2
                    continue
                out.append("\n" if code[i] == "\n" else " "); i += 1
            if i < n:
                out.append(q3); i += 3
            continue
        if c in ('"', "'", "`"):
            q = c
            out.append(q); i += 1
            while i < n and code[i] != q:
                if code[i] == "\\" and i + 1 < n:
                    out.append(" "); out.append("\n" if code[i + 1] == "\n" else " "); i += 2
                    continue
                if code[i] == "\n" and q != "`":
                    break
                out.append("\n" if code[i] == "\n" else " "); i += 1
            if i < n and code[i] == q:
                out.append(q); i += 1
            continue
        out.append(c); i += 1
    return "".join(out)


def _statements(text):
    """Statements of the program, line by line: a `#` comment is dropped and `;` outside quotes
    splits a line. Quote state never carries over a line, so one stray quote cannot hide the rest."""
    out = []
    for line in _code_of(text).split("\n"):
        cur, q, i, n = [], None, 0, len(line)
        while i < n:
            c = line[i]
            if q:
                cur.append(c)
                if c == "\\" and i + 1 < n:
                    cur.append(line[i + 1]); i += 1
                elif c == q:
                    q = None
            elif c in "'\"":
                q = c; cur.append(c)
            elif c == "#":
                break
            elif c == ";":
                out.append("".join(cur)); cur = []
            else:
                cur.append(c)
            i += 1
        out.append("".join(cur))
    return [s for s in out if s.strip()]


REBIND_RES = (re.compile(r"\b([A-Za-z_]\w*)\s*(?:[-+*/%|&^@]|//|\*\*|<<|>>)=(?!=)"),
              re.compile(r"\b([A-Za-z_]\w*)\s*:="),
              re.compile(r"\bas\s+([A-Za-z_]\w*)"))
REBIND_LIST_RES = (re.compile(r"\bfor\s+([^\n:]*?)\s+in\b"), re.compile(r"\bdef\s+\w+\s*\(([^)]*)\)"),
                   re.compile(r"\blambda\s+([^:]*):"), re.compile(r"\bas\s*\(([^)]*)\)"),
                   re.compile(r"^[ \t]*([A-Za-z_][\w \t,*()\[\]]*,[\w \t,*()\[\]]*)=(?!=)", re.M))


def _temp_names(text):
    """Names every binding of which is a fresh temp folder (see _temp_expr). A name rebound any
    other way (`+=`, `:=`, `for x in`, `as x`, a parameter) is never treated as a temp folder."""
    rhs, stmts = {}, _statements(text)
    for s in stmts:
        m = ANY_ASSIGN_RE.match(s)
        if m:
            rhs.setdefault(m.group(1), []).append(m.group(2))
    code, rebound = _mask_code_strings("\n".join(stmts)), set()
    for r in REBIND_RES:
        rebound |= set(r.findall(code))
    for r in REBIND_LIST_RES:
        for group in r.findall(code):
            rebound |= set(re.findall(r"[A-Za-z_]\w*", group))
    for name in rebound:
        rhs.pop(name, None)
    temp, changed = set(), True
    while changed:
        changed = False
        for name, values in rhs.items():
            if name not in temp and all(_temp_expr(v, temp) for v in values):
                temp.add(name); changed = True
    return temp


def inline_file_calls(text, ecwd, roots, gate_files):
    """Judge each file-writing or file-deleting call in inline code by its own path operand.
    Returns (hits, unresolved_kind_or_None, mentions_gate_path)."""
    code = INLINE_IMPORT_CALL_RE.sub(lambda m: m.group(2), _code_of(text))
    masked = _mask_code_strings(code)
    temp = _temp_names(code)
    literals = {}
    for m in INLINE_ASSIGN_RE.finditer(code):
        if masked[m.start(1)] == " ":
            continue
        lit = _literal_path(m.group(2), {})
        if lit is not None and m.group(1) not in literals:
            literals[m.group(1)] = lit
    found, unresolved = [], None
    for kind, call_re in (("write", INLINE_WRITE_CALL_RE), ("delete", INLINE_DELETE_CALL_RE)):
        for m in call_re.finditer(masked):
            open_idx = code.index("(", m.start())
            args = _balanced_args(code, open_idx)
            name = code[m.start():open_idx].strip()
            if args is None:
                unresolved = "delete" if kind == "delete" else (unresolved or kind)
                continue
            operands = []
            if name.startswith((".write_", ".unlink", ".rmdir")):
                recv = re.search(
                    r"([A-Za-z_]\w*|(?:pathlib\.)?Path\(\s*(['\"]).*?\2\s*\)(?:\.(?:expanduser|resolve|absolute)\(\s*\))*)\s*$",
                    code[:m.start()])
                operands = [recv.group(1)] if recv else []
            elif name == "open":
                if not any(INLINE_MODE_RE.match(a) for a in args[1:]):
                    continue                                 # a read; not a write
                operands = args[:1]
            elif kind == "write" and (name.endswith("copyfile") or name.endswith("copy") or name.endswith("copy2")
                                       or ".copy" in name):
                operands = args[1:2]                          # destination is the write
            elif kind == "delete" and ("rename" in name or "replace" in name or "move" in name or ".mv" in name):
                operands = args[:2]
            else:
                operands = args[:1]
            if not operands:
                unresolved = "delete" if kind == "delete" else (unresolved or kind)
                continue
            for op in operands:
                if _temp_expr(op, temp):                     # a folder the code made with tempfile: scratch
                    continue
                lit = _literal_path(op, literals)
                if lit is None:
                    unresolved = "delete" if kind == "delete" else (unresolved or kind)
                    continue
                target = resolve(lit, ecwd)
                if kind == "write":
                    found += write_hits(target, gate_files, "inline code writes to")
                    if is_precious_outside_file(target) and not is_scratch(target) and not (ecwd and is_inside(target, ecwd)) \
                            and not any(h[0] in ("OP-000", "OP-001", "OP-002", "OP-008", "OP-011") for h in found):
                        found.append(("OP-011", "inline code overwrites %s, which is outside the project" % target,
                                      "Copy it to a dated backup first, or ask the owner."))
                else:
                    own = target_hits("delete", target, True, roots, gate_files)
                    found += own
                    if not own and INLINE_RECURSIVE_RE.search(name) and not is_scratch(target) and (
                            target == "/" or same_path(target, HOME) or target.count("/") <= 2 or not ecwd
                            or not is_inside(target, ecwd) or is_same_or_ancestor(target, ecwd)):
                        found.append(("OP-003", "inline code: recursive delete of %s" % target,
                                      "Move it to a dated folder under /tmp, or ask the owner."))
                    elif not own and not is_scratch(target):
                        if is_precious_outside_file(target) and not (ecwd and is_inside(target, ecwd)):
                            found.append(("OP-011", "inline code deletes %s, which is outside the project" % target,
                                          "Copy it to a dated backup first, or ask the owner."))
                        elif ecwd and is_inside(target, ecwd):
                            dirty = dirty_git_path(target, ecwd)
                            if dirty:
                                found.append(("OP-011", "inline code deletes %s with uncommitted changes to %s" % (target, dirty),
                                              "Commit or stash the work first (`git stash`), or ask the owner."))
    mentions_gate = any(touches_gate_home(resolve(q, ecwd)) or any(is_inside(resolve(q, ecwd), f) for f in gate_files)
                        for q in re.findall(r"(?:~|\$HOME|\$\{HOME\}|/)[^\s'\"),;\]]*", code))
    return found, unresolved, mentions_gate


# ---- deterministic rewrites (proposed, never applied; one per original) ---- #










# ---- the taste layer ------------------------------------------------------- #


def write_paths(tool, ti, cwd):
    paths = []
    for k in ("file_path", "path", "notebook_path", "filename", "target_file", "file"):
        v = ti.get(k)
        if isinstance(v, str):
            paths.append(norm(v, cwd))
    if tool == "apply_patch":
        patch = ti.get("patch") or ti.get("input") or ti.get("command") or ""
        if isinstance(patch, list):
            patch = " ".join(map(str, patch))
        for m in re.finditer(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", str(patch), re.M):
            paths.append(norm(m.group(1), cwd))
    return paths


LITE_OWNER_SUBS = ("", "install", "uninstall", "mode", "approve", "upgrade")
LITE_PACKAGE_RE = re.compile(r"(?:@[\w.-]+/)?operator-lite(?:@[^\s/]*)?")
CLAUDE_SETTINGS_RE = re.compile(r"/\.claude/settings(?:\.local)?\.json$")


def lite_owner_sub(prog, args):
    """The owner-only subcommand ("" for the bare install) when a command runs the operator-lite package
    through npx, bunx, pnpx, pnpm/yarn dlx, npm/pnpm exec or its own bin; None for anything else."""
    if prog in ("npx", "bunx", "pnpx"):
        rest = list(args)
    elif prog in ("yarn", "pnpm") and args[:1] == ["dlx"] or prog in ("npm", "pnpm") and args[:1] in (["exec"], ["x"]):
        rest = list(args[1:])
    elif prog == "operator-lite":
        rest = ["operator-lite"] + list(args)
    else:
        return None
    rest = [a for a in rest if a != "--"]
    for i, a in enumerate(rest):
        if LITE_PACKAGE_RE.fullmatch(a):
            while i + 1 < len(rest) and LITE_PACKAGE_RE.fullmatch(rest[i + 1]):   # npx -p operator-lite operator-lite
                i += 1
            sub = rest[i + 1] if i + 1 < len(rest) else ""
            return sub if sub in LITE_OWNER_SUBS else None
    return None


def write_body_text(tool, ti):
    """Extract the added/written text from a file-writing tool call."""
    if tool == "multiedit" and isinstance(ti.get("edits"), list):
        return "\n".join(str(e.get("new_string") or e.get("content") or "") for e in ti["edits"] if isinstance(e, dict))
    if tool == "apply_patch":
        patch = str(ti.get("patch") or ti.get("input") or ti.get("command") or "")
        return "\n".join(ln[1:] for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return str(ti.get("content") or ti.get("new_string") or ti.get("file_text") or ti.get("new_str") or "")


def write_manifest_hits(path, body, cfg, cwd):
    """Cross-turn split-horizon defense: inspect commands staged into package.json scripts, Makefiles,
    git/husky hooks, .envrc/.psqlrc, and conftest.py/setup.py/build.rs via Write/Edit/apply_patch."""
    if not body or not body.strip():
        return []
    hits = []
    base = path.rstrip("/").rsplit("/", 1)[-1].lower()
    pdir = os.path.dirname(path) or cwd
    staged_cmds = []
    if base == "package.json":
        try:
            pkg = json.loads(body)
            if isinstance(pkg, dict) and isinstance(pkg.get("scripts"), dict):
                staged_cmds += [str(v) for v in pkg["scripts"].values() if isinstance(v, str)]
        except Exception:
            for sm in re.finditer(r'"scripts"\s*:\s*\{([^}]*)\}', body, re.S):
                staged_cmds += [m.group(1) for m in re.finditer(r':\s*"((?:[^"\\]|\\.)*)"', sm.group(1))]
            if not staged_cmds:
                for sm in re.finditer(r'"(?:preinstall|install|postinstall|prepare|prepublish|pretest|test|posttest|start|build)"\s*:\s*"((?:[^"\\]|\\.)*)"', body):
                    staged_cmds.append(sm.group(1).replace('\\"', '"'))
    elif base in ("makefile", "gnumakefile"):
        for ln in body.splitlines():
            if ln.startswith("\t") and ln.strip() and not ln.strip().startswith("#"):
                staged_cmds.append(ln.strip().lstrip("@-+").strip())
    elif base in (".envrc", ".psqlrc", ".gdbinit", ".lldbinit") or "/.git/hooks/" in path or "/.husky/" in path:
        staged_cmds.append(body)
    elif base in ("conftest.py", "setup.py", "build.rs"):
        lang = "python3" if base.endswith(".py") else "node"
        for sc in inline_shell_commands(body, lang):
            staged_cmds.append(sc)
        roots = [norm(r, cwd) for r in cfg["protected_roots"]] + [cwd]
        gate_files = self_protected_files() + [norm(p, cwd) for p in cfg["protected_files"]]
        found, _, _ = inline_file_calls(body, pdir, roots, gate_files)
        for rid, d, h in found:
            hits.append((rid, "%s -- staged in %s" % (d, os.path.basename(path)), h))
    for sc in staged_cmds[:24]:
        for eff in analyze(sc, pdir):
            for rid, d, h in classify_effect(eff, cfg, pdir):
                if not rid.startswith("OP-S"):
                    hits.append((rid, "%s -- staged in %s" % (d, os.path.basename(path)), h))
    return hits


def hook_settings_hit(path, text):
    """Claude Code reads hooks from every .claude/settings*.json, and a project's settings outrank the
    user's, so a project file that turns hooks off or rewires them switches the gate off. OP-000 when the
    write carries hooks or disableAllHooks; a recorded OP-S14 when its content cannot be seen."""
    if not CLAUDE_SETTINGS_RE.search(path):
        return None
    if re.search(r"disableAllHooks|\"hooks\"|\bhooks\s*[:=]", text or ""):
        return ("OP-000", "agent writes hook settings in %s" % path,
                "Hook settings are the owner's; propose the change instead of writing it.")
    return ("OP-S14", "write to Claude Code settings %s" % path, "")



# ---------------------------------------------------------------------------
# Session and per-call rules from Vestige Operator, open in Lite since 6.0.0:
# OP-012 the floor, OP-013 root reach, OP-015 repo code run, OP-016 vulnerable
# write, OP-017 agent arming, OP-019 harmful chain, OP-020 second spelling,
# OP-021 found key, OP-022 protected file, OP-023 remote delete, OP-024 runaway
# brake and OP-025 watch turnover. Owner laws, history across sessions and
# agents, and team use stay in the paid Operator.
# ---------------------------------------------------------------------------


def active_laws():
    """Lite carries no owner laws; the paid Operator's laws are not part of it."""
    return {}, []


# --------------------------------------------------------------------------- #
# measured loss: what a stopped action would have removed (0.3.11)
# --------------------------------------------------------------------------- #
LOSS_WALK_CAP = 20000


LOSS_WALK_S = 0.6


def measure_loss(effects, cwd):
    """Counted from the disk as it is now: the files and bytes under the delete targets, and how many
    files or uncommitted changes nothing could bring back. The count stops at a size and time cap;
    `capped` says the real figure is at least this."""
    targets, discards = [], 0
    for e in effects or []:
        if e.get("kind") == "delete" and not e.get("filtered"):
            targets += [t for t in e.get("targets") or [] if os.path.lexists(t) and t not in targets]
        elif e.get("kind") == "git" and e.get("sub") in ("reset", "clean", "checkout", "restore", "stash"):
            root = git_top(e.get("git_dir") or e.get("cwd") or cwd)
            entries = _status(root, ["--untracked-files=all"], None) if root else None
            for state, _rel in entries or []:
                untracked = state == "??"
                if e["sub"] == "stash" or untracked == (e["sub"] == "clean"):
                    discards += 1
    if not targets and not discards:
        return None
    files = size = 0
    capped, start = False, time.time()
    for t in targets:
        if os.path.isdir(t) and not os.path.islink(t):
            for base, _dirs, names in os.walk(t):
                for n in names:
                    try:
                        size += os.lstat(os.path.join(base, n)).st_size
                    except OSError:
                        pass
                    files += 1
                if files >= LOSS_WALK_CAP or time.time() - start > LOSS_WALK_S:
                    capped = True
                    break
        else:
            try:
                size += os.lstat(t).st_size
            except OSError:
                pass
            files += 1
        if capped:
            break
    lost = guarded(_lost_list, [], targets) if targets else []
    return {"files": files, "bytes": size, "unrecoverable": len(lost) + discards, "capped": capped}


def human_bytes(n):
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%d %s" % (n, unit) if unit == "bytes" else "%.1f %s" % (n, unit)
        n /= 1024.0


def loss_line(loss):
    """One sentence for the stop message, the Board and the Letter, or ''."""
    if not loss:
        return ""
    least = "at least " if loss.get("capped") else ""
    files, gone = int(loss.get("files") or 0), int(loss.get("unrecoverable") or 0)
    if not files:
        return "%d uncommitted change%s nothing could bring back" % (gone, "" if gone == 1 else "s") if gone else ""
    text = "%s%s file%s, %s" % (least, format(files, ","), "" if files == 1 else "s", human_bytes(int(loss.get("bytes") or 0)))
    if gone:
        text += "; %s%s of them exist%s nowhere else" % (least if gone >= 4000 else "", format(gone, ","), "s" if gone == 1 else "")
    return text


FLOOR_RULES = {"OP-012"}                  # the floor holds in every mode but off, and no permit exists for it
BRAKE_TRIPPED = {"sid": ""}               # set by write_receipt when a stop trips the brake


def call_of(payload):
    """The host's id for this one tool call, recorded on the receipt so a stop names the exact call."""
    p = payload or {}
    return str(p.get("tool_use_id") or p.get("toolUseId") or p.get("call_id") or p.get("tool_call_id") or "")[:80]

DEFECTS = []                                         # readers that failed during this judgement (see guarded)


def command_text(cmd):
    """The command with heredoc bodies and quoted strings blanked, followed by the strings a shell is
    handed with -c. Text that is only written or printed is not a command; `sh -c '...'` is."""
    body = extract_heredocs(cmd)[0]
    inner = [m.group(2) for m in re.finditer(r"\b(?:sh|bash|zsh|dash|ksh)\s+(?:-[a-zA-Z]*c)\s+(['\"])(.*?)\1", cmd, re.S)]
    return mask_quotes(body) + "\n" + "\n".join(inner)


def short_home(p):
    """A path under the home folder written from ~, the way the owner reads it."""
    p = str(p or "")
    return "~" + p[len(HOME):] if p == HOME or p.startswith(HOME + "/") else p


# Every shell an agent can hand a command string or a script to. One list, used wherever the gate
# asks "is this a shell": POSIX shells, their restricted and static builds, and the newer ones.
ALL_SHELLS = ("sh", "bash", "zsh", "dash", "ksh", "ksh93", "mksh", "pdksh", "oksh", "loksh", "ash", "hush", "fish",
              "csh", "tcsh", "nu", "nushell", "xonsh", "elvish", "osh", "ysh", "oil", "yash", "rc", "es", "ion",
              "murex", "rbash", "rksh", "rzsh", "bash-static", "zsh-static", "posh", "sash", "bosh")


CODE_PROG = {"python": "py", "pypy": "py", "ipython": "py", "node": "js", "nodejs": "js", "deno": "js", "bun": "js",
             "tsx": "js", "ts-node": "js", "ts-node-esm": "js", "zx": "js", "ruby": "rb", "jruby": "rb",
             "perl": "pl", "php": "php", "lua": "lua", "luajit": "lua", "osascript": "applescript", "rscript": "r",
             "awk": "awk", "gawk": "awk", "mawk": "awk", "nawk": "awk"}


def guarded(fn, default, *args, **kw):
    """Run one of the readers added in 0.3.8. On a defect it reports nothing instead of failing the
    whole judgement, so the older checks still decide. OPERATOR_DEBUG=1 lets the defect through (tests)."""
    try:
        return fn(*args, **kw)
    except Exception:
        if os.environ.get("OPERATOR_DEBUG"):
            raise
        DEFECTS.append(getattr(fn, "__name__", "reader"))    # strict mode stops a call a reader failed on
        return default


def code_prog(prog):
    """The language an interpreter runs (`python3.12` is py, `ts-node` is js), or None."""
    p = re.sub(r"\.exe$", "", (prog or "").lower())
    return CODE_PROG.get(p) or CODE_PROG.get(re.sub(r"[\d.]+$", "", p))


# --------------------------------------------------------------------------- #
# work nothing can bring back (OP-011): read from the state of the files and the repository
# --------------------------------------------------------------------------- #
FRESH_S = 3600                           # a file made in the last hour is the session's own scratch


LOSS_HINT = "Commit it, stash it (`git stash -u`) or move it first, or ask the owner for a one-time permit."


# ignored by git and still not replaceable: secrets, keys, local databases, infrastructure state
PRECIOUS_RE = re.compile(r"(?:^|/)(?:\.env(?:\.[\w.\-]+)?|[^/]*\.(?:sqlite3?|db|pem|key|p12|pfx|keystore|jks|kdbx|"
                         r"tfstate|tfstate\.backup)|id_(?:rsa|ed25519|ecdsa)|credentials(?:\.json)?|secrets?\.\w+)$", re.I)


def git_top(path):
    """The repository a path sits in: the nearest folder at or above it that holds .git. None outside one."""
    p = path if os.path.isdir(path) else posixpath.dirname(path.rstrip("/"))
    for _ in range(40):
        if p and os.path.exists(pj(p, ".git")):
            return p
        up = posixpath.dirname(p.rstrip("/")) if p else ""
        if not up or up == p:
            return None
        p = up
    return None


GIT_ASKED = {}                           # answers for the command being judged; cleared with SEEN_FILES


def git_out(root, args, timeout=3):
    """The output of a read-only git query, or None. The file monitor, prompts and optional locks are
    off, so asking changes nothing and runs nothing the repository configures."""
    key = (root, tuple(args))
    if key in GIT_ASKED:
        return GIT_ASKED[key]
    if len(GIT_ASKED) >= 24:                                 # a command that needs more is not judged this way
        return None
    GIT_ASKED[key] = _git_out(root, args, timeout)
    return GIT_ASKED[key]


def _git_out(root, args, timeout):
    try:
        r = subprocess.run(["git", "-C", root, "-c", "core.fsmonitor=false", "-c", "core.quotepath=false"] + args,
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace",
                           env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0"))
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _old(path, now):
    """True when the file was there before this session's last hour. Where the system records when a
    file was made (macOS, Windows) that is used; elsewhere, when it was last changed."""
    try:
        st = os.lstat(path)
        made = getattr(st, "st_birthtime", None) or (st.st_ctime if IS_WINDOWS else st.st_mtime)
        return now - min(made, st.st_mtime) > FRESH_S
    except OSError:
        return False


def _files_under(path, limit=4000):
    """Up to `limit` files under a path, build and cache folders left out."""
    if not os.path.isdir(path) or os.path.islink(path):
        return [path] if os.path.lexists(path) else []
    out = []
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in BUILD_DIRS and d != ".git"]
        out += [pj(canon(base), f) for f in files]
        if len(out) >= limit:
            break
    return out[:limit]


def _status(root, args, paths):
    """(two-letter state, path) for each entry of `git status` under these paths, or None."""
    out = git_out(root, ["status", "--porcelain", "-z"] + args + ["--"] + (paths or ["."]))
    if out is None:
        return None
    entries, parts, i = [], out.split("\0"), 0
    while i < len(parts):
        ent = parts[i]
        i += 1
        if len(ent) < 4:
            continue
        if ent[0] in "RC":
            i += 1                                           # a rename carries its old name as the next field
        entries.append((ent[:2], ent[3:]))
    return entries


def _named(lost):
    return "%d file%s nothing can bring back, such as %s" % (len(lost), "" if len(lost) == 1 else "s", lost[0])


def _lost_list(targets):
    """The files that deleting these paths loses for good. Inside a repository that
    is the files with uncommitted changes, the untracked files, and the ignored files that hold
    secrets or data (.env, a key, a local database, infrastructure state). Outside any repository
    it is every file. Files made in the last hour are the session's own and are not counted."""
    now, lost, by_root = time.time(), [], {}
    for t in targets:
        if not is_scratch(t) and os.path.lexists(t):
            by_root.setdefault(git_top(t), []).append(t)
    for root, paths in by_root.items():
        entries = _status(root, ["--untracked-files=all", "--ignored=matching"], paths) if root else None
        if entries is None:                                  # no repository, or git could not answer
            for t in paths:
                lost += [f.replace(HOME, "~") for f in _files_under(t) if _old(f, now)]
            continue
        for state, rel in entries:
            full = pj(root, rel.rstrip("/"))
            if any(part in BUILD_DIRS for part in rel.split("/")):
                continue
            if state == "!!":                                # ignored: only what cannot be made again
                if rel.endswith("/"):
                    lost += [f.replace(root + "/", "") for f in _files_under(full, 2000)
                             if PRECIOUS_RE.search(f) and _old(f, now)]
                elif PRECIOUS_RE.search(rel) and _old(full, now):
                    lost.append(rel)
            elif state != "??" or _old(full, now):            # an uncommitted change is work whenever it was saved
                if os.path.lexists(full):
                    lost.append(rel)
    return lost


def lost_work(targets):
    """Why deleting these paths loses work for good, as a phrase, or None."""
    lost = _lost_list(targets)
    return _named(lost) if lost else None


# --------------------------------------------------------------------------- #
# root reach: root work nobody is watching, and signals that leave their target (0.3.12)
# --------------------------------------------------------------------------- #
ROOT_PROGS = {"sudo", "doas", "pkexec", "run0"}


SUDO_VALUE_FLAGS = {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U", "-R", "--user", "--group",
                    "--close-from", "--chdir", "--host", "--prompt", "--role", "--type",
                    "--command-timeout", "--other-user", "--chroot"}


DETACH_WRAPPERS = {"nohup", "setsid"}


SHELL_PROGS = set(ALL_SHELLS)


BACKGROUND_RE = re.compile(r"(?<![&|><])&(?![&>])")


KILL_ALL_CODE_RE = re.compile(r"\bos\.kill\(\s*-\s*1\s*,")


KILL_GROUP_CODE_RE = re.compile(r"\bos\.killpg\(")


ROOT_HINT = ("Start long root work as a transient service: `sudo systemd-run --unit=<name> --collect "
             "<command>`. The service manager owns it, outside this session's process group. Stop it "
             "with `sudo systemctl stop <name>`, read it with `journalctl -u <name>`. Ask the owner.")


KILL_HINT = ("Signal the one process id you mean. A negative id is a whole process group, -1 is every "
             "process, and procps-ng 4 reads only the first digit of a negative id. Ask the owner.")


def kill_targets(args):
    """The process ids a `kill` command line names, as written. The first dash word is the signal
    unless -s or -n already gave one; after `--` every word is an id."""
    ids, signal_seen, i = [], False, 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return ids + args[i + 1:]
        if a in ("-s", "-n", "--signal"):
            signal_seen = True; i += 2; continue
        if a in ("-l", "-L", "--list", "--table"):
            return []
        if a.startswith("-") and not signal_seen:
            signal_seen = True; i += 1; continue
        ids.append(a); i += 1
    return ids


def root_form(toks):
    """How one command segment is launched: (runs as root, detached by its own words, fed by
    xargs, the bare command, the command string handed to a shell or to su). Reads through
    `( ... )` and `{ ...; }` grouping, VAR=x prefixes, wrappers, sudo's flags, and `su -c`."""
    toks = list(toks)
    while toks and toks[0] in ("(", "{", "!"):
        toks = toks[1:]
    if toks and toks[0].startswith("(") and len(toks[0]) > 1:
        toks[0] = toks[0].lstrip("(")
    while toks and toks[-1] in (")", "}"):
        toks = toks[:-1]
    if toks and toks[-1].endswith(")") and "(" not in toks[-1]:
        toks[-1] = toks[-1].rstrip(")")
    root = detached = via_xargs = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", t):
            i += 1; continue
        base = os.path.basename(t)
        if base in ROOT_PROGS:
            root = True; i += 1
            while i < len(toks) and toks[i].startswith("-"):
                f = toks[i]; i += 1
                if f == "--":
                    break
                if f == "--background" or (base == "sudo" and re.match(r"^-[A-Za-z]*b[A-Za-z]*$", f)):
                    detached = True
                if f in SUDO_VALUE_FLAGS and i < len(toks):
                    i += 1
            continue
        if base in WRAPPERS:
            detached = detached or base in DETACH_WRAPPERS
            via_xargs = via_xargs or base == "xargs"
            i += 1
            while i < len(toks) and toks[i].startswith("-"):
                i += 1
                if base in ("timeout", "nice") and i < len(toks) and re.match(r"^[0-9]+$", toks[i]):
                    i += 1
            if base == "timeout" and i < len(toks) and re.match(r"^[0-9.]+[smhd]?$", toks[i]):
                i += 1
            continue
        break
    bare, inner = toks[i:], None
    if bare:
        prog, args = os.path.basename(bare[0]), bare[1:]
        if prog == "su":
            user = None
            for j, a in enumerate(args):
                if a in ("-c", "--command") and j + 1 < len(args):
                    inner = args[j + 1]
                elif a.startswith("--command="):
                    inner = a.split("=", 1)[1]
                elif not a.startswith("-") and user is None and (j == 0 or args[j - 1] not in ("-c", "--command")):
                    user = a
            if inner is not None and user in (None, "root"):
                root = True
            elif inner is not None:
                inner = None
        elif prog in SHELL_PROGS:
            for j, a in enumerate(args):
                if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a) and j + 1 < len(args):
                    inner = args[j + 1]
                    break
    return root, detached, via_xargs, bare, inner


def peek_script(path):
    """A script's text for the root-reach read, comments dropped. None when it is not a small text
    file. Separate from read_script, which hands a file out once per command."""
    try:
        st = os.stat(path)
        if not _stat.S_ISREG(st.st_mode) or st.st_size == 0 or st.st_size > MAX_SCRIPT_BYTES:
            return None
        with open(path, "rb") as f:
            raw = f.read(MAX_SCRIPT_BYTES)
    except Exception:
        return None
    if b"\0" in raw[:4096]:
        return None
    return "\n".join(l for l in raw.decode("utf-8", "replace").split("\n") if not l.lstrip().startswith("#"))


def root_reach(cmd, background, cwd=None, depth=0, as_root=False, label=""):
    """OP-013 for one shell command: a root job that outlives the call, and kills that reach past
    one process. Reads the command's words, the string handed to `sh -c` or `su -c`, and one level
    into a script the command runs. Runs nothing."""
    found = []
    detached = background or bool(BACKGROUND_RE.search(mask_quotes(cmd)))
    tail = (" -- inside script %s" % label) if label else ""
    for seg in split_segments(cmd):
        root, own_detach, via_xargs, bare, inner = root_form(tokenize(seg))
        root = root or as_root
        if not bare:
            continue
        prog, args = os.path.basename(bare[0]), bare[1:]
        gone = detached or own_detach
        if inner is not None and depth < 2:
            found += root_reach(inner, gone, cwd, depth + 1, root, label)
            continue
        if prog == "killall5":
            found.append(("OP-013", "killall5 signals every process" + tail, KILL_HINT))
        elif prog == "kill":
            ids = kill_targets(args)
            if root and via_xargs and not any(re.match(r"^-?\d+$", t) for t in ids):
                found.append(("OP-013", "root kill of targets fed in by xargs, which the gate cannot see" + tail, KILL_HINT))
            for t in ids:
                if t == "-1":
                    found.append(("OP-013", "kill aimed at every process (-1)" + tail, KILL_HINT))
                elif root and re.match(r"^-\d+$", t):
                    found.append(("OP-013", "root kill of process group %s%s" % (t[1:], tail), KILL_HINT))
                elif root and re.match(r"^-[$`]", t):
                    found.append(("OP-013", "root kill of a process group held in a variable" + tail, KILL_HINT))
        elif prog in ("pkill", "killall") and root and any(       # killall's -s is a signal, pkill's a session
                re.match(r"^(-[A-Za-z]*[%s]\d*|--(pgroup|session|euid|uid|user|process-group)(=.*)?)$"
                         % ("gsuU" if prog == "pkill" else "gu"), a) for a in args):
            found.append(("OP-013", "root %s by process group, session or user%s" % (prog, tail), KILL_HINT))
        elif (code_prog(prog) == "py" or prog.startswith("python")) and KILL_ALL_CODE_RE.search(seg):
            found.append(("OP-013", "os.kill(-1, ...) signals every process" + tail, KILL_HINT))
        elif (code_prog(prog) == "py" or prog.startswith("python")) and root and KILL_GROUP_CODE_RE.search(seg):
            found.append(("OP-013", "root os.killpg of a process group" + tail, KILL_HINT))
        elif root and prog == "systemd-run" and "--scope" not in args:
            pass                                             # a transient service: the service manager owns it
        elif root and gone:
            found.append(("OP-013", "root job started in the background: %s%s" % (prog, tail), ROOT_HINT))
        elif root and (prog == "tmux" and any(re.match(r"^-[A-Za-z]*d", a) for a in args)
                       or prog == "screen" and any(re.match(r"^-[A-Za-z]*d[A-Za-z]*m|^-[A-Za-z]*m[A-Za-z]*d", a) for a in args)):
            found.append(("OP-013", "root job started detached in %s%s" % (prog, tail), ROOT_HINT))
        elif depth == 0 and cwd:
            path = None
            if prog in SHELL_PROGS:
                path = guarded(resolve_script_arg, None, args, cwd)
            elif "/" in bare[0] or bare[0].endswith((".sh", ".bash", ".zsh")):
                path = guarded(resolve, None, bare[0], cwd)
            body = guarded(peek_script, None, path) if path else None
            if body:
                found += root_reach(body, gone, cwd, 1, root, os.path.basename(path))
    return list(dict.fromkeys(found))[:2]


WRECK_HINT = ("Work on an image file or a loop device, never on the disk itself; change permissions "
              "on the one folder you mean, inside the project. The owner does the rest by hand.")


# --------------------------------------------------------------------------- #
# repo code run: git settings that execute a program (GitSpawn, October 2026) (0.3.12)
# --------------------------------------------------------------------------- #
GIT_BOOL = {"true", "false", "yes", "no", "on", "off", "1", "0", ""}


GIT_NET_SUBS = {"fetch", "pull", "push", "clone", "ls-remote", "remote", "submodule"}


GIT_DIFF_SUBS = {"diff", "show", "log", "status", "add", "checkout", "restore", "stash", "commit"}


SPAWN_HINT = ("Read what the setting runs and show the owner; use `git -c core.fsmonitor=false` to look at "
              "a repository you do not trust. The owner sets program-running git config by hand.")


GIT_SECTION_RE = re.compile(r'^\s*\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]')


def git_exec_setting(key, value):
    """Why a git setting runs a program, as a short phrase, or None. `key` is lower case."""
    v = (value or "").strip()
    if key == "core.fsmonitor" and v.lower() not in GIT_BOOL:
        return "core.fsmonitor runs `%s` on git status, diff, add and other index reads" % v[:60]
    if key in ("core.sshcommand", "core.gitproxy") and v:
        return "%s runs `%s` on every fetch, pull and push" % (key, v[:60])
    if key in ("diff.external", "uploadpack.packobjectshook") and v:
        return "%s runs `%s`" % (key, v[:60])
    if re.match(r"^diff\..+\.(command|textconv)$", key) and v:
        return "%s runs `%s` when a diff is shown" % (key, v[:60])
    if re.match(r"^filter\..+\.(clean|smudge|process)$", key) and v and not v.startswith("git-lfs "):
        return "%s runs `%s` on checkout, add and status" % (key, v[:60])
    if key.startswith("alias.") and v.startswith("!"):
        return "%s is a shell alias: `%s`" % (key, v[:60])
    if key == "credential.helper" and v.startswith("!") and not re.match(r"^!\s*gh\s+auth\s+git-credential\s*$", v):
        return "credential.helper runs `%s`" % v[:60]      # `!gh auth git-credential` is gh's documented helper
    return None


def repo_git_config(cwd):
    """(path, {key: value}) of the git config that governs `cwd`, read as text; git is not run.
    Covers a normal checkout and a folder that is itself a bare repository."""
    d, path = cwd, None
    for _ in range(40):
        if not d:
            break
        dot = pj(d, ".git")
        if os.path.isdir(dot):
            path = pj(dot, "config"); break
        if os.path.isfile(pj(d, "HEAD")) and os.path.isfile(pj(d, "config")) and os.path.isdir(pj(d, "objects")) \
                and os.path.isdir(pj(d, "refs")):
            path = pj(d, "config"); break
        up = posixpath.dirname(d.rstrip("/"))
        if up == d or not up:
            break
        d = up
    if not path:
        return None, {}
    try:
        if os.path.getsize(path) > 262144:
            return path, {}
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return path, {}
    out, section = {}, ""
    for line in text.split("\n"):
        m = GIT_SECTION_RE.match(line)
        if m:
            section = m.group(1).lower() + ("." + m.group(2) if m.group(2) is not None else "")
            continue
        if "=" in line and section and not line.lstrip().startswith(("#", ";")):
            k, v = line.split("=", 1)
            out["%s.%s" % (section, k.strip().lower())] = v.strip().strip('"')
    return path, out


def repo_code_run(cmd, cwd):
    """OP-015 for one shell command: planting a program-running git setting, and running git where
    the repository's own config already carries one. Reads words and one config file; runs nothing."""
    found = []
    for seg in split_segments(cmd):
        toks = tokenize(seg)
        for t in toks:                                       # GIT_EXTERNAL_DIFF=x git diff, GIT_SSH_COMMAND=x git pull
            m = re.match(r"^(GIT_EXTERNAL_DIFF|GIT_SSH_COMMAND|GIT_SSH|GIT_PROXY_COMMAND)=(.*)$", t)
            if not m:
                break
            name, val = m.group(1), m.group(2).strip()
            if val and not (name == "GIT_SSH_COMMAND" and re.match(r"^ssh(\s|$)", val)):
                found.append(("OP-015", "%s makes git run `%s`" % (name, val[:60]), SPAWN_HINT))
        _, _, _, bare, inner = root_form(toks)
        if inner is not None:
            found += repo_code_run(inner, cwd)
            continue
        if not bare or os.path.basename(bare[0]) != "git":
            continue
        args, here, i = bare[1:], cwd, 0
        while i < len(args) and args[i].startswith("-"):     # git's own options, before the subcommand
            if args[i] == "-c" and i + 1 < len(args):
                k, _, v = args[i + 1].partition("=")
                why = git_exec_setting(k.lower(), v)
                if why:
                    found.append(("OP-015", "git -c: " + why, SPAWN_HINT))
                i += 2; continue
            if args[i] == "-C" and i + 1 < len(args):
                here = norm(args[i + 1], here); i += 2; continue
            i += 1
        sub, rest = (args[i] if i < len(args) else ""), args[i + 1:]
        if sub == "config":
            plain = [a for a in rest if not a.startswith("-")]
            reading = any(a in ("--get", "--get-all", "--get-regexp", "--list", "-l", "--unset", "--unset-all",
                                "--remove-section", "--show-origin") for a in rest)
            if len(plain) >= 2 and not reading:
                why = git_exec_setting(plain[0].lower(), plain[1])
                if why:
                    found.append(("OP-015", "git config sets a program: " + why, SPAWN_HINT))
            continue
        if not sub or any(a.lower() == "core.fsmonitor=false" for a in args[:i]):
            continue
        path, conf = repo_git_config(here)
        for k, v in conf.items():
            why = git_exec_setting(k, v)
            if not why:
                continue
            fires = k == "core.fsmonitor" or (k in ("core.sshcommand", "core.gitproxy", "credential.helper") and sub in GIT_NET_SUBS) \
                or (k.startswith("diff.") and sub in ("diff", "show", "log", "stash", "format-patch")) \
                or (k.startswith("filter.") and sub in GIT_DIFF_SUBS) or (k.startswith("alias.") and k[6:] == sub.lower())
            if fires:
                found.append(("OP-015", "this repository's config (%s): %s" % (path, why), SPAWN_HINT))
                break
    return list(dict.fromkeys(found))[:2]


# --------------------------------------------------------------------------- #
# written code: critical flaws in what an agent is about to write (0.3.12)
# --------------------------------------------------------------------------- #
# The first eight detectors, ported from the Nullgaze scanner (ai_patterns.rs, patterns 1, 12, 14,
# 104, 109, 117, 129, 140): each is rated Critical at 0.90 confidence or above there. They read the
# text of a Write or an Edit line by line. Tests, fixtures, examples and prose are not judged.
CLIENT_EXTS = (".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".html")


SERVER_DIR_RE = re.compile(r"(^|/)(server|api|backend|functions|lambda|middleware|supabase/functions)(/|$)")


NOT_SHIPPED_RE = re.compile(r"(^|/)(tests?|__tests__|spec|specs|fixtures?|examples?|samples?|docs?|node_modules|vendor)(/|$)"
                            r"|(\.|_|-)(test|spec)\.[a-z]+$|^test_[^/]*$|\.(md|mdx|rst|txt|lock)$", re.I)


EXAMPLE_LINE_RE = re.compile(r"<code|</code|<pre|</pre|<samp|&lt;|&gt;|&amp;|example|snippet|demo code|sample code", re.I)


COMMENT_LINE_RE = re.compile(r"^\s*(//|#|\*|/\*|<!--|--)")


WRITE_HINT = "Write it the safe way named above, or hand the owner the reason this file needs the exception."


W_SERVICE_ROLE_RE = re.compile(r"service[_\-]?role", re.I)


W_FIREBASE_ADMIN_RE = re.compile(r"""from\s+['"]firebase-admin|require\s*\(\s*['"]firebase-admin|initializeApp\s*\(\s*\{[^}]*credential""")


W_SERVICE_ACCOUNT_RE = re.compile(r'"type"\s*:\s*"service_account"')


W_JWT_NONE_RE = re.compile(r"""algorithms?\s*[=:]\s*\[?\s*["']none["']|\balg\s*[=:]\s*["']none["']""", re.I)


W_JWT_NOVERIFY_RE = re.compile(r""""?verify_signature"?\s*[:=]\s*(false|False)""")


W_TLS_OFF_RE = re.compile(r"""rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true|ssl_verify\s*=\s*False"""
                          r"""|NODE_TLS_REJECT_UNAUTHORIZED['"\]]*\s*=\s*['"]?0|CURLOPT_SSL_VERIFY(PEER|HOST)\s*,\s*(0|false)"""
                          r"""|\bverify_ssl\s*=\s*False""")


W_VERIFY_FALSE_RE = re.compile(r"\bverify\s*=\s*False")


W_HTTP_CONTEXT_RE = re.compile(r"requests\.|httpx\.|aiohttp|urllib|session|Session|\.(get|post|put|patch|delete|request)\s*\(|ssl|tls|cert|https?:", re.I)


W_DOCKER_SOCK_RE = re.compile(r"/var/run/docker\.sock\s*:|[-\s]v\s+/var/run/docker\.sock|path\s*:\s*/var/run/docker\.sock")


W_YAML_LOAD_RE = re.compile(r"\byaml\.load\s*\(|\bYAML\.load\s*\(")


W_YAML_SAFE_RE = re.compile(r"Loader\s*=\s*(yaml\.)?(Safe|Base|CSafe)Loader|\bsafe_load\b")


W_DOCKER_SECRET_COPY_RE = re.compile(r"^(COPY|ADD)\s+(--\S+\s+)*(\.env(\.(?!example|sample|template|dist)[\w-]+)?(?=\s|$)|\S*(credential|secret|\.pem|\.p12|\.pfx|id_rsa|id_ecdsa|id_ed25519|\.kubeconfig)\S*)", re.I)


def written_code(path, body):
    """OP-016 for one file about to be written: (rule, what and where, the safe way). Reads text only."""
    p = canon(path or "")
    if not body or NOT_SHIPPED_RE.search(p) or len(body) > 2_000_000:
        return []
    base = posixpath.basename(p)
    client = p.endswith(CLIENT_EXTS) and not SERVER_DIR_RE.search(p)
    dockerfile = base == "Dockerfile" or base.startswith("Dockerfile.") or base.endswith(".dockerfile")
    found = []

    def hit(n, what, safe):
        found.append(("OP-016", "%s (%s line %d)" % (what, base, n), safe + " " + WRITE_HINT))

    for n, line in enumerate(body.split("\n"), 1):
        if len(line) > 2000 or COMMENT_LINE_RE.match(line):
            continue
        if client and not EXAMPLE_LINE_RE.search(line):
            if W_SERVICE_ROLE_RE.search(line):
                hit(n, "a Supabase service_role key in client code bypasses every row-level policy",
                    "Keep the service_role key on the server and use the anon key in the client.")
            if W_FIREBASE_ADMIN_RE.search(line):
                hit(n, "the Firebase Admin SDK in client code ships full project access to every visitor",
                    "Use the Firebase client SDK in the browser and the Admin SDK only on the server.")
        if W_SERVICE_ACCOUNT_RE.search(line):
            hit(n, "a cloud service-account key is being written into the project",
                "Load the key from the environment or a secret manager and keep it out of the tree.")
        if W_JWT_NONE_RE.search(line) or W_JWT_NOVERIFY_RE.search(line):
            hit(n, "JWT signature checking is switched off, so any token is accepted",
                "Name the real algorithm and verify the signature.")
        if W_TLS_OFF_RE.search(line) or (W_VERIFY_FALSE_RE.search(line) and W_HTTP_CONTEXT_RE.search(line)):
            hit(n, "TLS certificate checking is switched off, so any server can pose as the real one",
                "Leave verification on and add the CA you need to the trust store.")
        if W_DOCKER_SOCK_RE.search(line):
            hit(n, "the Docker socket is mounted into a container, which hands it the host",
                "Use a socket proxy with a narrow allow-list or a rootless builder.")
        if p.endswith((".py", ".rb")) and W_YAML_LOAD_RE.search(line) and not W_YAML_SAFE_RE.search(line):
            hit(n, "yaml.load without a safe loader runs code from the file it reads",
                "Use yaml.safe_load, or pass Loader=yaml.SafeLoader.")
        if dockerfile and W_DOCKER_SECRET_COPY_RE.search(line.strip()):
            hit(n, "a secret file is copied into the image, where every layer keeps it",
                "Pass it at run time or use a build secret mount.")
    return list(dict.fromkeys(found))[:3]


# --------------------------------------------------------------------------- #
# agent arming: project files a harness or editor runs by itself on start (0.3.12)
# --------------------------------------------------------------------------- #
# The class behind CurXecute and MCPoison (Cursor, CVE-2025-54135 and 54136), the Codex project
# config advisories (CVE-2025-61260, the notify field), CVE-2026-25725 (a SessionStart hook written
# into .claude/settings.json) and the Gemini .gemini/.env advisories (CVE-2026-12537, 13745): the
# write looks like an edit to a config file, and the command in it runs with no further question.
# (path pattern, what reads it, what in the content makes it a command; None = any write counts)
ARMING_FILES = (
    (re.compile(r"(^|/)\.codex/config\.toml$"), "Codex runs its notify command and MCP servers from this file",
     re.compile(r"^\s*notify\s*=|^\s*\[mcp_servers|^\s*command\s*=", re.M)),
    (re.compile(r"(^|/)\.gemini/\.env$"), "Gemini CLI loads this file into its own environment on start", None),
    (re.compile(r"(^|/)\.gemini/settings\.json$"), "Gemini CLI starts the MCP servers named in this file",
     re.compile(r'"command"\s*:|"mcpServers"\s*:')),
    (re.compile(r"(^|/)(\.cursor/mcp\.json|\.mcp\.json|\.vscode/mcp\.json|\.roo/mcp\.json|\.windsurf/mcp_config\.json"
                r"|mcp_settings\.json|cline_mcp_settings\.json|claude_desktop_config\.json)$"),
     "the agent starts every MCP server command listed in this file", re.compile(r'"command"\s*:')),
    (re.compile(r"(^|/)\.cursor/hooks\.json$"), "Cursor runs the hook commands in this file", re.compile(r'"command"\s*:')),
    (re.compile(r"(^|/)\.claude/settings(\.local)?\.json$"), "Claude Code runs the hooks and helper commands in this file",
     re.compile(r'"(hooks|command|apiKeyHelper|statusLine|awsAuthRefresh)"\s*:')),
    (re.compile(r"(^|/)\.vscode/tasks\.json$"), "the editor runs a folder-open task from this file when the project is opened",
     re.compile(r'"runOn"\s*:\s*"folderOpen"')),
    (re.compile(r"(^|/)\.envrc$"), "direnv runs this file on every cd into the project",
     re.compile(r"^(?!\s*(#|$|export\s+[A-Za-z_][A-Za-z0-9_]*=[^$`(;&|]*$|dotenv\b|dotenv_if_exists\b|layout\s|use\s|PATH_add\s|source_up\b|watch_file\s)).+", re.M)),
)


ARMING_HINT = ("Show the owner the exact command and let them add it by hand, or ask for a one-time permit.")


def agent_arming(path, body):
    """OP-017 for one file about to be written. `body` is the new text, or None when a shell command
    writes it and the text cannot be seen, in which case the write itself is the finding."""
    p = canon(path or "")
    for rx, reads, signal in ARMING_FILES:
        if not rx.search(p):
            continue
        if body is None:
            return [("OP-017", "shell write to %s: %s" % (p, reads), ARMING_HINT)]
        if signal is None or signal.search(body[:400000]):
            return [("OP-017", "command written into %s: %s" % (p, reads), ARMING_HINT)]
    return []


# --------------------------------------------------------------------------- #
# loss reach: the same losses by other routes (0.3.12)
# --------------------------------------------------------------------------- #
# A delete sent through a carrier, a delete that wears another tool's name, work thrown away by a
# git maintenance verb, a file of secrets overwritten, a remote store emptied, and fetched code run
# unread. Each uses the rule that already names the loss; only the route is new.
CARRIED_DELETE_RE = re.compile(r"\brm\s+(?:-[A-Za-z]+\s+)*-[A-Za-z]*[rR]|\brm\s+--recursive\b|\bRemove-Item\b.*-Recurse"
                                 r"|\bfind\b.*\s-delete\b|\bshred\s|\bmkfs|\bdd\s+.*\bof=/dev/")


SSH_VALUE_FLAGS = set("-i -p -o -l -J -F -L -R -D -b -c -e -m -S -W -w -E -B -I -Q".split())


FETCH_PIPE_RE = re.compile(r"\b(?:curl|wget)\b[^|;&\n]*?(https?://[^\s'\"|;&)]+)[^|;&\n]*\|\s*(?:sudo\s+(?:-\S+\s+)*)?(?:ba|z|da|k)?sh\b")
# `bash <(curl URL)`; Lite's FETCH_SUBST_RE is the `$(curl ...)` form, without the URL as a group
FETCH_PROC_SUBST_RE = re.compile(r"\b(?:(?:ba|z)?sh|source|\.)\s+<\(\s*(?:curl|wget)\b[^)]*?(https?://[^\s'\")]+)")


KNOWN_INSTALLERS_RE = re.compile(
    r"^https://(sh\.rustup\.rs|get\.docker\.com|bun\.sh/install|astral\.sh/uv/install\.sh|deno\.land/install\.sh|get\.pnpm\.io/"
    r"|fly\.io/install\.sh|ollama\.com/install\.sh|claude\.ai/install\.sh|sh\.brew\.sh"
    r"|raw\.githubusercontent\.com/(Homebrew/install|nvm-sh/nvm|pyenv/pyenv-installer|ohmyzsh/ohmyzsh|rbenv/rbenv-installer)/)")


JUNK_NAME_RE = re.compile(r"^(\*\.(pyc|pyo|o|obj|class|log|tmp|temp|swp|swo|orig|rej|bak|map)|\*~|__pycache__|\.DS_Store|Thumbs\.db"
                          r"|node_modules|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.cache|dist|build|target|\*\.egg-info|\.tox|coverage|\.coverage)$")


SHARED_BRANCH_RE = re.compile(r"^(main|master|trunk|develop|release[/\-].*|prod.*)$")


CARRIER_HINT = "Name the one path, show the owner the command, and let them run it where it lands."


FETCH_HINT = "Download it to a file, read it, then run the file; or ask the owner for a one-time permit."


REMOTE_HINT = "Run it with the tool's dry-run flag first and show the owner the list it would remove."


FIND_SCAN_CAP = 30000                                        # directory entries looked at, matches or not


def _find_matches(roots, patterns, cwd, limit=4000):
    """The files a `find` with these -name patterns would select under these roots. Read-only. Capped
    twice: by matches found, and by entries looked at, so a find over a home folder or a whole disk
    costs the same small walk as one over a project. The cap is a count, not a clock, so the same
    tree gives the same answer on any machine."""
    import fnmatch
    out, looked = [], 0
    for r in roots:
        base = norm(r, cwd)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in BUILD_DIRS and d != ".git"]
            looked += 1 + len(filenames)
            for name in filenames:
                if not patterns or any(fnmatch.fnmatch(name, pat) for pat in patterns):
                    out.append(pj(dirpath, name))
                    if len(out) >= limit:
                        return out
            if looked >= FIND_SCAN_CAP:
                return out
    return out


def loss_reach(cmd, cwd):
    """Extra STOP hits for one shell command. Reads words and file names; runs nothing."""
    found = []
    for rx in (FETCH_PIPE_RE, FETCH_PROC_SUBST_RE):
        m = rx.search(command_text(cmd))                      # a URL inside a quoted argument is data (receipt cdbaf1cdeea5d8a6)
        if m and not KNOWN_INSTALLERS_RE.match(m.group(1)):
            found.append(("OP-018", "network code handed straight to a shell: %s" % m.group(1)[:80], FETCH_HINT))
    if re.search(r"\|\s*(?:at|batch)\b", mask_quotes(cmd)) and CARRIED_DELETE_RE.search(cmd):
        found.append(("OP-003", "recursive delete scheduled through at, to run when nobody is watching", CARRIER_HINT))
    for seg in split_segments(cmd):
        _, _, via_xargs, bare, inner = root_form(tokenize(seg))
        if inner is not None:
            found += [h for h in loss_reach(inner, cwd) if h not in found]
            continue
        if not bare:
            continue
        prog, args = os.path.basename(bare[0]), bare[1:]
        flags = [a for a in args if a.startswith("-")]
        plain = [a for a in args if not a.startswith("-")]
        dry = any(f in ("--dry-run", "--dryrun", "-n") for f in flags)
        sent = None
        if prog in ("ssh", "mosh"):
            i = 0
            while i < len(args) and args[i].startswith("-"):
                i += 2 if args[i] in SSH_VALUE_FLAGS else 1
            sent = " ".join(args[i + 1:])
        elif prog in ("docker", "podman", "nerdctl", "kubectl", "oc") and any(a in ("exec", "run") for a in args[:3]):
            sent = " ".join(args[min(i for i, a in enumerate(args) if a in ("exec", "run")) + 1:])
        elif prog == "tmux" and plain[:1] in (["send-keys"], ["send"]):
            sent = " ".join(plain[1:])
        elif prog in ("busybox", "toybox"):
            sent = " ".join(args)
        elif prog == "parallel":
            sent = " ".join(args[:args.index(":::")] if ":::" in args else args)
        if sent and CARRIED_DELETE_RE.search(sent):
            found.append(("OP-003", "recursive delete sent through %s: %s" % (prog, sent[:70]), CARRIER_HINT))
        elif via_xargs and prog == "rm" and any(re.match(r"^-[A-Za-z]*[rR]", f) for f in flags) \
                and re.search(r"(?<![<\d])<(?![<(])", mask_quotes(seg)):
            found.append(("OP-003", "recursive delete fed by xargs from a file: the targets cannot be seen", CARRIER_HINT))
        elif prog == "tar" and "--remove-files" in flags and re.search(r"(-[A-Za-z]*f\s+|--file[= ])(/dev/null|-)(\s|$)", seg):
            found.append(("OP-003", "tar --remove-files with the archive thrown away", CARRIER_HINT))
        elif prog == "osascript" and re.search(r"Finder", seg) and re.search(r"\bdelete\b|\bempty\b.*\btrash\b", seg, re.I):
            found.append(("OP-003", "Finder told to delete or empty the trash by script", CARRIER_HINT))
        elif prog == "truncate" and re.search(r"(-s\s*|--size[= ])0\b", seg):
            targets = resolve_targets([a for a in plain if a != "0"], cwd)
            why = guarded(lost_work, None, targets)
            if why:
                found.append(("OP-011", "truncate empties %s" % why, LOSS_HINT))
        elif prog == "shred":
            why = guarded(lost_work, None, resolve_targets(plain, cwd))
            if why:
                found.append(("OP-011", "shred destroys %s" % why, LOSS_HINT))
        elif prog == "find" and ("-delete" in args or re.search(r"-exec(?:dir)?\s+(?:\S*/)?rm\b", seg)):
            roots = []
            for a in args:
                if a.startswith(("-", "(", "!")):
                    break
                roots.append(a)
            names = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-name", "-iname")]
            if not (names and all(JUNK_NAME_RE.match(n) for n in names)):
                why = guarded(lost_work, None, guarded(_find_matches, [], roots or ["."], names, cwd)[:600])
                if why:
                    found.append(("OP-011", "find removes %s" % why, LOSS_HINT))
        elif prog == "git" and plain[:1] == ["branch"] and (set(flags) & {"-D"} or {"--delete", "--force"} <= set(flags)) \
                and any(SHARED_BRANCH_RE.match(b) for b in plain[1:]):
            found.append(("OP-004", "forced delete of local branch %s" % next(b for b in plain[1:] if SHARED_BRANCH_RE.match(b)),
                          "Leave the shared branch; delete your own feature branch by name."))
        elif prog == "git" and plain[:2] == ["reflog", "expire"] and any(re.match(r"--expire(-unreachable)?=(now|all|0)", f) for f in flags):
            found.append(("OP-011", "git reflog expire removes the record that makes lost commits recoverable", LOSS_HINT))
        elif prog == "crontab" and "-r" in flags and "-l" not in flags:
            found.append(("OP-011", "crontab -r removes every scheduled job and keeps no copy",
                          "Save it first with `crontab -l > crontab.backup`, then edit with `crontab -e`."))
        elif prog == "rclone" and plain[:1] in (["purge"], ["delete"]) and not dry:
            found.append(("OP-006", "rclone %s empties %s" % (plain[0], " ".join(plain[1:2])[:60]), REMOTE_HINT))
        elif prog == "aws" and plain[:2] == ["s3", "sync"] and "--delete" in flags and not dry:
            found.append(("OP-006", "aws s3 sync --delete removes remote objects that are not in the source", REMOTE_HINT))
        elif prog == "gsutil" and "rsync" in plain and any(re.match(r"^-[A-Za-z]*d", f) for f in flags) and not dry:
            found.append(("OP-006", "gsutil rsync -d removes remote objects that are not in the source", REMOTE_HINT))
        elif prog in ("npm", "pnpm", "yarn") and plain[:1] == ["unpublish"] and not dry:
            found.append(("OP-005", "npm unpublish removes a published package version for everyone",
                          "Deprecate it instead, or hand the owner the exact version to remove."))
        # a file of secrets or data overwritten: `> .env`, `cp x .env`, `mv x .env`
        dst = None
        m = re.search(r"(?<![>&\d])>(?![>&])\s*([^\s;&|]+)", mask_quotes(seg))
        if m:
            dst = seg[m.start(1):m.end(1)].strip("'\"")
        elif prog in ("cp", "mv") and len(plain) >= 2:
            dst = plain[-1]
        if dst and PRECIOUS_RE.search(norm(dst, cwd)) and not norm(dst, cwd).endswith(ENV_TEMPLATE_ENDS):
            full = norm(dst, cwd)
            try:
                size = os.path.getsize(full) if os.path.isfile(full) else 0
            except OSError:
                size = 0
            why = guarded(lost_work, None, [full]) if size else None
            if why:
                found.append(("OP-011", "overwriting %s replaces %s" % (posixpath.basename(full), why), LOSS_HINT))
    return list(dict.fromkeys(found))[:3]


# --------------------------------------------------------------------------- #
# windows wreck: PowerShell and cmd.exe forms of the same losses (0.3.12)
# --------------------------------------------------------------------------- #
# These name their effect in the verb, so they are read on any host and from any shell tool: an
# agent on Linux can still type `powershell -c` or `cmd /c`. (pattern, rule, what it does, safe way)
WINDOWS_WRECK = (
    (re.compile(r"\b(Format-Volume|Clear-Disk|Remove-Partition|Reset-PhysicalDisk)\b", re.I), "OP-014",
     "a disk or volume is wiped", WRECK_HINT),
    (re.compile(r"\bdiskpart(\.exe)?\s+/s\b|(?:^|[;&|]\s*)format(\.com)?\s+[A-Za-z]:", re.I), "OP-014",
     "a disk or volume is formatted", WRECK_HINT),
    (re.compile(r"\bcipher(\.exe)?\s+/w\b", re.I), "OP-014", "free space is overwritten, which ends file recovery", WRECK_HINT),
    (re.compile(r"\bvssadmin(\.exe)?\s+delete\s+shadows\b|\bwmic\b.*\bshadowcopy\b.*\bdelete\b|\bwbadmin(\.exe)?\s+delete\b", re.I), "OP-011",
     "the restore points are deleted, the copies Windows keeps for getting files back", LOSS_HINT),
    (re.compile(r"\bwevtutil(\.exe)?\s+(cl|clear-log)\b|\bClear-EventLog\b", re.I), "OP-011",
     "an event log is cleared and nothing keeps a copy", LOSS_HINT),
    (re.compile(r"\brobocopy(\.exe)?\b.*\s/(MIR|PURGE)\b", re.I), "OP-003",
     "robocopy /MIR deletes everything in the destination that is not in the source", CARRIER_HINT),
    (re.compile(r"\[(?:System\.)?IO\.Directory\]::Delete\s*\([^)]*,\s*\$true", re.I), "OP-003",
     "a .NET recursive directory delete", CARRIER_HINT),
    (re.compile(r"\b(?:iex|Invoke-Expression)\b[^;|\n]*\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod|DownloadString)\b"
                r"|\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^;\n]*\|\s*(?:iex|Invoke-Expression)\b"
                r"|DownloadString\s*\([^)]*\)[^;\n]*\|\s*(?:iex|Invoke-Expression)\b", re.I), "OP-018",
     "network code handed straight to PowerShell", FETCH_HINT),
    (re.compile(r"\bStop-Process\b[^;|\n]*-Name\s+['\"]?\*|\btaskkill(\.exe)?\b[^;|\n]*/IM\s+['\"]?\*", re.I), "OP-013",
     "every process is told to stop", KILL_HINT),
    (re.compile(r"\breg(\.exe)?\s+delete\s+['\"]?(HKLM|HKEY_LOCAL_MACHINE|HKCR|HKEY_CLASSES_ROOT)(\\(SYSTEM|SOFTWARE|SAM|SECURITY))?['\"]?(\s|$)", re.I),
     "OP-014", "a whole registry hive or its top key is deleted", WRECK_HINT),
)


def windows_wreck(cmd):
    """Extra STOP hits for the PowerShell and cmd.exe verbs above. Quoted text that is only printed
    or committed is not a command, so the scan runs on the command with quoted strings masked,
    except for the fetch-and-run and .NET forms, whose arguments are quoted by nature."""
    found, masked = [], mask_quotes(cmd)
    for m in re.finditer(r"\b(?:powershell|pwsh|cmd)(?:\.exe)?\b[^\n]*?\s(?:-c(?:ommand)?|/c)\s+(['\"])(.*?)\1", cmd, re.I | re.S):
        found += windows_wreck(m.group(2))                   # the string handed to powershell -c or cmd /c
    for rx, rule, what, hint in WINDOWS_WRECK:
        text = cmd if rule in ("OP-018",) or "IO" in rx.pattern else masked
        if rule == "OP-018" and re.match(r"^\s*(echo|Write-Host|Write-Output|git\s+commit)\b", cmd, re.I):
            continue
        if rx.search(text):
            found.append((rule, what, hint))
    return list(dict.fromkeys(found))[:2]


# --------------------------------------------------------------------------- #
# chains: steps that are each ordinary and only harmful together (0.3.12)
# --------------------------------------------------------------------------- #
# The gate keeps a small ledger per session: one line for each step that could be the first half of
# a chain, written only when that step was let through. A later call is judged against it. The
# verdict is a function of the ledger and the call: the same history and the same call give the
# same answer, and a stop names each earlier step with its time and its receipt.
CHAIN_WINDOW_S = 12 * 3600


CHAIN_PENDING = {"sid": "", "marks": []}


LOCAL_HOST_RE = re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|\[?::1\]?|host\.docker\.internal|[\w.-]+\.(local|localhost|test|internal))$", re.I)


SECRET_FILE_RE = re.compile(r"(^|/)(\.env(\.(?!example|sample|template|dist)[\w.-]+)?|id_(rsa|ed25519|ecdsa|dsa)|\.netrc|\.git-credentials|\.pypirc"
                            r"|[^/]*service[_-]?account[^/]*\.json|[^/]*\.(pem|key|p12|pfx))$"
                            r"|(^|/)\.aws/credentials$|(^|/)\.config/gh/hosts\.yml$|(^|/)\.kube/config$|(^|/)\.docker/config\.json$")


READ_PROGS = {"cat", "less", "more", "head", "tail", "bat", "grep", "egrep", "rg", "awk", "sed", "base64", "xxd", "strings", "cut", "tee", "nl"}


BUILD_HOOK_FILES = {"conftest.py": ("pytest", "tox", "nox"), "noxfile.py": ("nox",), "setup.py": ("pip", "pip3", "uv", "python", "python3"),
                    "build.rs": ("cargo",)}


PROCESS_CALL_RE = re.compile(r"\bos\.system\s*\(|\bsubprocess\.|\bos\.popen\s*\(|\bpty\.spawn\s*\(|std::process::Command|Command::new\s*\(")


CHAIN_HINT = "Show the owner the earlier steps and this one together, and let them allow it once."


def session_of(payload):
    """The host's session for this call. Cursor names it conversation_id on tool events."""
    p = payload or {}
    return str(p.get("session_id") or p.get("sessionId") or p.get("conversation_id") or p.get("conversationId") or "")[:64]


def _chain_path(sid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)[:80]
    return pj(OP_HOME, "state", "chain", safe + ".jsonl") if safe else None


def chain_ledger(sid):
    """The marks this session has left, oldest first, inside the look-back window."""
    path, out, now = _chain_path(sid), [], time.time()
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f.readlines()[-600:]:
                try:
                    m = json.loads(line)
                except Exception:
                    continue
                if isinstance(m, dict) and now - float(m.get("t", 0)) <= CHAIN_WINDOW_S:
                    out.append(m)
    except Exception:
        pass
    return out


def chain_commit(receipt_digest):
    """Write the pending marks of a call that was let through. Called from write_receipt."""
    sid, marks = CHAIN_PENDING.get("sid"), CHAIN_PENDING.get("marks") or []
    CHAIN_PENDING["marks"] = []
    path = _chain_path(sid or "")
    if not path or not marks:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for m in marks:
                f.write(json.dumps(dict(m, t=time.time(), receipt=(receipt_digest or "")[:16]), sort_keys=True) + "\n")
    except Exception:
        pass


def _hosts(seg):
    return [h for h in re.findall(r"https?://(\[[^\]]+\]|[^/\s:'\"@]+)", seg)]


def _sends_out(prog, args, seg):
    """Where this command sends data, or None. A local address is not out."""
    hosts = [h for h in _hosts(seg) if not LOCAL_HOST_RE.match(h)]
    if prog in ("curl", "wget", "http", "https", "xh"):
        posts = any(re.match(r"^(-d|--data.*|-F|--form.*|-T|--upload-file|--post-data.*|--post-file.*|--json)$", a) for a in args) \
            or re.search(r"(-X|--request)[= ]?\s*(POST|PUT|PATCH)\b", seg, re.I) or (prog in ("http", "https", "xh") and args[:1] in (["POST"], ["PUT"]))
        return hosts[0] if posts and hosts else None
    if prog in ("nc", "ncat", "netcat", "socat", "telnet"):
        dest = next((a for a in args if not a.startswith("-") and not LOCAL_HOST_RE.match(a) and re.search(r"[A-Za-z]", a)), None)
        return dest
    if prog in ("scp", "rsync", "sftp"):
        dest = next((a for a in args if re.match(r"^[\w.-]+@?[\w.-]*:[^/\\]|^[\w.-]+@[\w.-]+:", a)), None)
        return dest.split(":")[0] if dest and not LOCAL_HOST_RE.match(dest.split("@")[-1].split(":")[0]) else None
    if prog == "gh" and args[:2] == ["gist", "create"]:
        return "a GitHub gist"
    return None


def chain_marks(payload, cwd, effects, known=None):
    """What this call leaves behind for a later call to be judged against. `known` is what the
    session already holds; left out, it is read from the session's ledger."""
    tool = str(payload.get("tool_name") or payload.get("toolName") or "").lower()
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    if not isinstance(ti, dict):
        ti = {"command": ti}
    marks, cmd = [], str(ti.get("command") or ti.get("cmd") or "")
    if known is None:
        known = chain_ledger(session_of(payload))
    note = lambda kind, path, **extra: marks.append(dict(extra, kind=kind, path=path, preview=redact(cmd or ("%s %s" % (tool, path)))[:140]))
    if tool in ("read", "view", "read_file"):
        fp = norm(str(ti.get("file_path") or ti.get("path") or ""), cwd)
        if fp and SECRET_FILE_RE.search(fp):
            note("read_secret", fp)
        prints = file_key_prints(fp) if fp else set()
        if prints:                                            # fingerprints only; the key itself is never kept
            note("found_key", fp, prints=sorted(prints)[:20])
    if tool in WRITE_TOOLS:
        body = str(ti.get("content") or ti.get("new_string") or "")
        for wp in write_paths(tool, ti, cwd):
            m = PROCESS_CALL_RE.search(body) if posixpath.basename(wp) in BUILD_HOOK_FILES else None
            if m:
                note("build_hook", wp, line=body[:m.start()].count("\n") + 1)
    for seg in split_segments(cmd) if cmd else []:
        _, _, _, bare, inner = root_form(tokenize(seg))
        if inner is not None:
            sub = chain_marks({"tool_name": "bash", "tool_input": {"command": inner}}, cwd, [], list(known) + marks)
            marks += [dict(m, preview=redact(cmd)[:140]) for m in sub]
            continue
        if not bare:
            continue
        prog, args = os.path.basename(bare[0]), bare[1:]
        plain = [a for a in args if not a.startswith("-")]
        if prog in ("curl", "wget"):
            url = next((a for a in args if re.match(r"^https?://", a)), "")
            out = None
            for i, a in enumerate(args[:-1]):
                if a in ("-o", "--output", "-O", "--output-document") and not args[i + 1].startswith("-"):
                    out = args[i + 1]
            m = re.search(r"(?<![>&\d])>(?![>&])\s*([^\s;&|]+)", mask_quotes(seg))
            if m and not out:
                out = seg[m.start(1):m.end(1)].strip("'\"")
            if not out and url and (prog == "wget" or any(a in ("-O", "--remote-name") or re.match(r"^-[A-Za-z]*O", a) for a in args)):
                out = posixpath.basename(url.split("?")[0]) or None
            if out and url and out not in ("-", "/dev/null") and not KNOWN_INSTALLERS_RE.match(url):
                note("fetched", norm(out, cwd), url=url[:120])
        elif prog == "ln" and any(re.match(r"^-[A-Za-z]*s", a) for a in args) and len(plain) >= 2:
            target, link = norm(plain[-2], cwd), norm(plain[-1], cwd)
            if os.path.isdir(link):
                link = pj(link, posixpath.basename(target))
            if not is_inside(target, cwd) or SECRET_FILE_RE.search(target):
                note("symlinked", link, target=target)
        elif prog in READ_PROGS or prog == "cp":
            for a in (plain[:-1] if prog == "cp" else plain):
                fp = norm(a, cwd)
                if SECRET_FILE_RE.search(fp):
                    note("read_secret", fp)
                    break
            for a in (plain[:-1] if prog == "cp" else plain)[:12]:
                fp = norm(a, cwd)
                prints = file_key_prints(fp)
                if prints:                                    # fingerprints only; the key itself is never kept
                    note("found_key", fp, prints=sorted(prints)[:20])
        elif prog == "git" and (plain[:1] == ["commit"] and (set(args) & {"--no-verify", "-n"})):
            note("check_off", cwd, what="a commit made with --no-verify")
        elif prog == "git" and plain[:2] == ["config", "core.hooksPath"] and len(plain) >= 3 and plain[2] in ("/dev/null", "", "/var/empty"):
            note("check_off", cwd, what="git hooks pointed at nothing")
        elif prog in ("rm", "mv", "chmod") and any(re.search(r"(^|/)(\.git/hooks/|\.husky/)", norm(a, cwd)) for a in plain):
            if prog != "chmod" or any(re.match(r"^(-x|a-x|[0-6][0-6][0-6])$", a) for a in args):
                note("check_off", cwd, what="a git hook removed or switched off")
        if prog in ("cp", "mv") and len(plain) >= 2:          # a fetched file keeps its origin under a new name
            src, dst = norm(plain[-2], cwd), norm(plain[-1], cwd)
            origin = next((k for k in reversed(list(known) + marks) if k.get("kind") == "fetched" and k.get("path") == src), None)
            if origin:
                note("fetched", pj(dst, posixpath.basename(src)) if os.path.isdir(dst) else dst, url=origin.get("url"))
    return marks


CHAIN_RUNNABLE_RE = re.compile(r"\.(sh|bash|zsh|ksh|command|py|js|mjs|cjs|ts|rb|pl|php)$", re.I)


CHAIN_LAUNCH_RE = re.compile(PROCESS_CALL_RE.pattern + r"|\bimport\s+subprocess\b|\bfrom\s+subprocess\b|\bos\.exec[lv]p?e?\s*\(|\bos\.spawn[lv]p?e?\s*\("
                             r"|\bchild_process\b|\bexec(File)?(Sync)?\s*\(|\bspawn(Sync)?\s*\(|\bsystem\s*\(|\bpopen\s*\(|\bshell_exec\s*\("
                             r"|\bpassthru\s*\(|\brunpy\.|%x[({\[]|`")


CHAIN_ARGV_RE = re.compile(r"\bsys\.argv\b|\bprocess\.argv\b|\bARGV\b|\bargparse\b|\boptparse\b|\bgetopt\b")


CODE_FLAGS = ("-c", "-e", "--eval")


def chain_launched(args, cwd, fetched):
    """Fetched files an interpreter would start as a process from its own code: inline code, or the
    script it is handed, that makes a process call and either names the file in a string or hands on
    its arguments while the file is one of them. A fetched file that is only data is not counted,
    and neither is Operator's own sealed-copy runner, which keeps the run inside a sealed copy."""
    inline, script, rest, i = None, None, [], 0
    while i < len(args):
        a = args[i]
        if a in CODE_FLAGS and inline is None and script is None and i + 1 < len(args):
            inline = args[i + 1]
            i += 2
            continue
        if script is not None or inline is not None:
            rest.append(a)
        elif not a.startswith("-"):
            script = a
        i += 1
    body, path = "", norm(script, cwd) if script else ""
    if path and os.path.isfile(path) and os.path.getsize(path) <= 262144:
        if os.path.realpath(path) == os.path.realpath(pj(os.path.dirname(os.path.abspath(__file__)), "sealed.py")):
            return []
        with open(path, encoding="utf-8", errors="replace") as f:
            body = f.read()
    code = inline if inline is not None else body
    if not code or not CHAIN_LAUNCH_RE.search(code):
        return []
    handed = [norm(a, cwd) for a in rest if not a.startswith("-")] if CHAIN_ARGV_RE.search(code) else []
    hits = []
    for fp in fetched:
        name = posixpath.basename(fp)
        if not name or not (CHAIN_RUNNABLE_RE.search(name) or os.access(fp, os.X_OK)):
            continue
        named = (name in code) if inline is not None else bool(re.search(r"['\"][^'\"\n]*" + re.escape(name), code))
        if named or fp in handed:
            hits.append(fp)
    return hits


def chain_runs(bare, seg, cwd, piped=None, fetched=()):
    """The files this step would execute: the program itself when it is named by path, the script
    handed to a shell or an interpreter, a file fed to one on stdin, the file `cat` pipes into it,
    or a fetched file the interpreter's own code starts as a process."""
    while len(bare) > 1 and bare[0] in ("exec", "command", "builtin", "time"):
        bare = bare[1:]
    prog, args = os.path.basename(bare[0]), bare[1:]
    plain = [a for a in args if not a.startswith("-")]
    if prog in SHELL_PROGS or code_prog(prog) or prog in ("source", "."):
        out = [norm(a, cwd) for a in plain[:1]]
        masked = mask_quotes(seg)
        m = re.search(r"(?<![<\d&])<(?![<(&])\s*([^\s;&|<>]+)", masked)
        if m:
            out.append(norm(seg[m.start(1):m.end(1)].strip("'\""), cwd))
        if not plain and piped:
            out.append(piped)
        if code_prog(prog) and fetched:
            out += guarded(chain_launched, [], args, cwd, list(fetched))
        return out
    if "/" in bare[0] or bare[0].endswith((".sh", ".py", ".js", ".rb", ".pl")):
        return [norm(bare[0], cwd)]
    return []


def chain_reach(payload, cwd, effects, seen=None):
    """OP-019: does this call complete a chain? The earlier step can be an earlier call of this
    session, read from its ledger, or an earlier step of this same command (a && b, a; b, a | b)."""
    sid = session_of(payload)
    if seen is None:
        seen = chain_ledger(sid) if sid else []
    tool = str(payload.get("tool_name") or payload.get("toolName") or "").lower()
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    if not isinstance(ti, dict):
        ti = {"command": ti}
    cmd, found = str(ti.get("command") or ti.get("cmd") or ""), []
    when = lambda m: time.strftime("%H:%M:%S", time.localtime(float(m.get("t", 0))))
    step = lambda m: ("earlier in this same command, `%s`" % str(m.get("preview", ""))[:90]) if m.get("same") else \
        "at %s `%s` (receipt %s)" % (when(m), str(m.get("preview", ""))[:90], m.get("receipt") or "n/a")
    last = lambda kind, test=lambda m: True: next((m for m in reversed(seen) if m.get("kind") == kind and test(m)), None)
    link_at = lambda path: last("symlinked", lambda m: m.get("path") == path)
    through = lambda wp, m: ("OP-019", "write through %s, a link this session made to %s %s" % (wp, m.get("target"), step(m)), CHAIN_HINT)
    # a write through a link an earlier call made
    written = list(write_paths(tool, ti, cwd)) if tool in WRITE_TOOLS else []
    written += [w for e in effects or [] for w in (e.get("writes") or [])]
    for wp in written:
        m = link_at(norm(wp, cwd))
        if m:
            found.append(through(wp, m))
            break
    piped = None                                              # the file the step before this one printed with cat
    for seg in split_segments(cmd) if cmd else []:
        _, _, _, bare, inner = root_form(tokenize(seg))
        if inner is None and bare and os.path.basename(bare[0]) in SHELL_PROGS and "-c" in bare[1:-1]:
            inner = bare[bare.index("-c", 1) + 1]             # sh -c '...' with no sudo in front
        if inner is not None:
            found += chain_reach(dict(payload, tool_input={"command": inner}), cwd, [], seen)
            piped = None
            continue
        if bare:
            prog, args = os.path.basename(bare[0]), bare[1:]
            plain = [a for a in args if not a.startswith("-")]
            # running a file this session fetched
            for ran in chain_runs(bare, seg, cwd, piped, [x.get("path") for x in seen if x.get("kind") == "fetched"]):
                m = last("fetched", lambda x, ran=ran: x.get("path") == ran)
                if m:
                    found.append(("OP-019", "running %s, which this session fetched from %s %s" % (ran, m.get("url"), step(m)),
                                  ("Fetch it on its own first, read the file" if m.get("same") else "Read the file")
                                  + ", show the owner what it does, and let them allow it once."))
                    break
            # sending data out after reading a secret
            dest = _sends_out(prog, args, seg)
            m = last("read_secret") if dest else None
            if m:
                found.append(("OP-019", "sending data to %s after this session read %s %s" % (dest, m.get("path"), step(m)), CHAIN_HINT))
            # running the tool that executes a build file the session wired to launch a process
            for name, tools in BUILD_HOOK_FILES.items():
                runs = prog in tools and not (prog in ("python", "python3") and not any(a in ("setup.py", "pytest", "-m") for a in args)) \
                    and not (prog in ("pip", "pip3", "uv") and not any(a in (".", "-e", "install") for a in args))
                m = last("build_hook", lambda x, name=name: posixpath.basename(x.get("path", "")) == name and is_inside(x.get("path", ""), cwd)) if runs else None
                if m:
                    found.append(("OP-019", "%s will run %s, which this session wrote with a process call on line %s %s"
                                  % (prog, m.get("path"), m.get("line"), step(m)), CHAIN_HINT))
                    break
            # pushing after a check was switched off
            if prog == "git" and plain[:1] == ["push"]:
                m = last("check_off", lambda x: same_path(x.get("path", ""), cwd) or is_inside(cwd, x.get("path", "")))
                if m:
                    found.append(("OP-019", "push after %s %s" % (m.get("what"), step(m)),
                                  "Run the checks that were skipped, or show the owner why they were skipped."))
            # a write through a link made earlier in this same command
            if any(x.get("same") and x.get("kind") == "symlinked" for x in seen):
                for wp in [w for e in guarded(analyze, [], seg, cwd, 1) for w in (e.get("writes") or [])]:
                    m = link_at(norm(wp, cwd))
                    if m and m.get("same"):
                        found.append(through(wp, m))
                        break
        # what this step leaves behind for the steps after it in the same command
        seen += [dict(m, same=True) for m in guarded(chain_marks, [], {"tool_name": "bash", "tool_input": {"command": seg}}, cwd, [], seen)]
        cat = [a for a in bare[1:] if not a.startswith("-")] if bare and os.path.basename(bare[0]) == "cat" else []
        piped = norm(cat[0], cwd) if len(cat) == 1 else None
    return list(dict.fromkeys(found))[:2]


FLOOR_HINT = ("Name the one folder you mean, inside the project, as a literal path; write \"${VAR:?}\" "
              "when a variable is part of it. The owner deletes a root or a home folder by hand.")


SHORT_NAME_RE = re.compile(r"~\d+$")


WIN_DRIVE_RE = re.compile(r"^(?:([A-Za-z]):[\\/]*|/(cygdrive/|mnt/)?([A-Za-z])/?)$")


WIN_PATH_RE = re.compile(r"^(?:([A-Za-z]):[\\/]+|/(cygdrive/|mnt/)?([A-Za-z])/)(.+)$")


WIN_SYSTEM_DIRS = {"windows", "program files", "program files (x86)", "programdata"}


PURE_SUBST_RE = re.compile(r"\$\(\s*([a-z]+)((?:\s+-[A-Za-z-]+)*)\s+('[^']*'|\"[^\"$`]*\"|[^\s()$`'\"]+)\s*\)")


PWD_SUBST_RE = re.compile(r"\$\(\s*pwd(?:\s+-[LP])?\s*\)|`\s*pwd\s*`")


UNKNOWN_RE = re.compile(r"\$\((?:[^()]|\([^()]*\))*\)|`[^`]*`|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")


def floor_eval(word, cwd):
    """One delete argument with the substitutions the gate can make the way the shell will: the home
    folder, the working directory, cygpath, dirname, realpath. Returns the text, and the pieces only
    the running shell can fill in."""
    def pure(m):
        prog, arg = m.group(1), m.group(3).strip("'\"")
        if prog == "cygpath":
            return canon_windows(arg)
        if prog == "dirname":
            return posixpath.dirname(norm(arg, cwd)) or "/"
        if prog in ("realpath", "readlink"):
            return norm(arg, cwd)
        return m.group(0)

    w = word
    for _ in range(4):
        before = w
        w = w.replace("${HOME}", HOME).replace("$HOME", HOME)
        if cwd:
            w = w.replace("${PWD}", cwd).replace("$PWD", cwd)
            w = PWD_SUBST_RE.sub(lambda m: cwd, w)
        w = PURE_SUBST_RE.sub(pure, w)
        if w == before:
            break
    return tilde(w), UNKNOWN_RE.findall(w)


def floor_reach(text, cwd):
    """Why this path is below the floor, as a phrase, or None: a filesystem or drive root, a home
    folder or a parent of one, a Windows short-name alias for a top-level folder, a parent of the
    working directory, a top-level system folder."""
    t = (text or "").strip()
    if not t:
        return None
    if not t.strip("/\\"):
        return "the filesystem root"
    m = WIN_DRIVE_RE.match(t)
    if m and (m.group(1) or m.group(2) or IS_WINDOWS):
        return "the root of drive %s:" % (m.group(1) or m.group(3)).upper()
    m = WIN_PATH_RE.match(t)
    if m and (m.group(1) or m.group(2) or IS_WINDOWS):
        parts = [p for p in re.split(r"[\\/]+", m.group(4)) if p]
        if parts and parts[0].lower() in ("users", "documents and settings", "home") and len(parts) <= 2:
            return "a Windows home folder (%s)" % t if len(parts) == 2 else "the folder that holds every home folder (%s)" % t
        if parts and len(parts) <= 2 and SHORT_NAME_RE.search(parts[-1]):
            return "a short-name alias for a top-level folder (%s)" % t
        if len(parts) == 1 and parts[0].lower() in WIN_SYSTEM_DIRS:
            return "a Windows system folder (%s)" % t
        if not IS_WINDOWS:
            return None
    p = real(norm(t, cwd))
    if p == "/":
        return "the filesystem root"
    if same_path(p, HOME):
        return "your home folder"
    if is_same_or_ancestor(p, HOME):
        return "a parent of your home folder (%s)" % p
    c = real(canon(cwd)) if cwd and not cwd.startswith(UNKNOWN_CWD) else ""
    if c and not same_path(p, c) and is_same_or_ancestor(p, c) and not (is_scratch(p) and p.count("/") > 2):
        return "a parent of the working directory (%s)" % p
    if p.count("/") == 1 and os.path.isdir(p) and not is_scratch(p):
        return "a top-level system folder (%s)" % p
    return None


# `cmd /c`, `rmdir /s /q`, `del /q`: a slash and one letter is a switch, not the Git Bash form of a drive
CMD_SWITCH_PROGS = {"cmd", "rmdir", "rd", "del", "erase"}
CMD_SWITCH_RE = re.compile(r"^/[A-Za-z?](?::\S*)?$")


def floor_hits(e, cwd):
    """OP-012 for one recursive delete. A resolved target is judged as it stands. A target only the
    running shell can fill in is judged at its worst case, every unknown piece empty, because that
    is how `rm -rf "$DIR/"` becomes a delete of the root."""
    if e.get("kind") != "delete" or not e.get("recursive") or e.get("filtered"):
        return []
    try:
        words = shlex.split(e.get("seg") or "", posix=True)
    except ValueError:
        words = []
    prog = re.sub(r"\.exe$", "", posixpath.basename(words[0].replace("\\", "/")).lower()) if words else ""
    words = [w for w in words[1:] if w and not w.startswith("-")]
    if prog in CMD_SWITCH_PROGS:
        words = [w for w in words if not CMD_SWITCH_RE.match(w)]
    found = []
    for t in e.get("targets") or []:
        if prog in CMD_SWITCH_PROGS and CMD_SWITCH_RE.match(t):
            continue
        why = floor_reach(t, cwd)
        if why:
            found.append("recursive delete of %s" % why)
    for w in words:
        text, unknown = floor_eval(w, cwd)
        if not unknown:
            if e.get("unresolved") or not (set(w) & set("$`")):
                why = floor_reach(text, cwd)
                if why:
                    found.append("recursive delete of %s" % why)
            continue
        if not e.get("unresolved") or all(":?" in u for u in unknown):
            continue                                         # resolved by the analyzer, or the shell aborts when empty
        worst = UNKNOWN_RE.sub("", text).rstrip("*")
        why = floor_reach(worst, cwd) if worst.strip() else None
        if why:
            found.append("recursive delete of %s if %s %s empty when this runs"
                         % (why, " and ".join(unknown[:2]), "is" if len(unknown) == 1 else "are"))
    return [("OP-012", d, FLOOR_HINT) for d in list(dict.fromkeys(found))[:1]]


# ---- protected files: the owner's contracts on how a named file may change (OP-022) --------------- #
# append-only: bytes may only be added at the end, so what is there now is never rewritten.
# read-only:   no change at all without the owner.
# Contracts live in commitments.json under "file_contracts", which only the owner writes (OP-000).
FILE_CONTRACTS = ("append-only", "read-only")


APPEND_ONLY_PROGS = {"echo", "printf", "cat", "date", "jq"}        # programs that write nothing but stdout


FULL_WRITE_TOOLS = {"write", "create_file", "write_file"}


INLINE_WRITE_RE = re.compile(
    r"""open\s*\([^)]*,\s*(?:mode\s*=\s*)?['"][^'"]*[wax+]|\.write(?:_text|_bytes)?\s*\(|write(?:File|FileSync)|"""
    r"""append(?:File|FileSync)|unlink|remove\s*\(|rename|replace\s*\(|truncate|rmSync|rmtree|copyfile|copy2?\s*\(|move\s*\(""")


REDIRECT_RE = re.compile(r"(\d*|&)(>>|>\||>)\s*([^\s<>|&;()]+)")


PROTECTED_EVIDENCE = []                  # append-only files a passing call adds to, with the bytes before


def file_contracts(cfg, cwd):
    """[(path, contract)] from the owner's commitments."""
    out = []
    for c in (cfg or {}).get("file_contracts") or []:
        if isinstance(c, dict) and c.get("contract") in FILE_CONTRACTS and isinstance(c.get("path"), str) and c["path"].strip():
            out.append((norm(c["path"], cwd), c["contract"]))
    return out


def _file_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def prefix_evidence(path):
    """{path, size_before, sha256_before} for a file as it is now (size 0 when it does not exist yet)."""
    data = _file_bytes(path) or b""
    return {"path": path, "size_before": len(data), "sha256_before": hashlib.sha256(data).hexdigest()}


def _pure_append(e, path, cwd):
    """True when this segment adds to path only at the end: an append redirection from a program that
    writes nothing else, or tee -a."""
    prog = posixpath.basename(str(e.get("prog") or ""))
    seg = str(e.get("seg") or "")
    reds = [m.group(2) for m in REDIRECT_RE.finditer(seg)
            if m.group(3) and not m.group(3).startswith("&") and same_path(norm(m.group(3).strip("'\""), e.get("cwd") or cwd), path)]
    if prog == "tee":
        return ("-a" in (e.get("flags") or []) or "--append" in seg) and not [r for r in reds if r != ">>"]
    return prog in APPEND_ONLY_PROGS and bool(reds) and all(r == ">>" for r in reds)


def _inline_writes(e, path):
    """True when inline code (python -c, node -e) names this file and writes, moves or removes something."""
    text = str(e.get("text") or e.get("seg") or "")
    names = {path, posixpath.basename(path), short_home(path)}
    return any(n and n in text for n in names) and bool(INLINE_WRITE_RE.search(text))


def _git_touches(e, path, cwd):
    """True when this git command would delete, move or throw away the current bytes of path. A file git
    holds unchanged loses nothing to a discard; git rm and git mv change it whatever its state."""
    root = git_top(e.get("git_dir") or e.get("cwd") or cwd)
    if not root or not is_inside(path, root):
        return False
    sub, ga = str(e.get("sub") or ""), [str(a) for a in e.get("gargs") or []]
    rel = os.path.relpath(path, root)
    here = e.get("cwd") or cwd
    named = any(a in (".", rel) or same_path(norm(a, here), path) or is_same_or_ancestor(norm(a, here), path)
                for a in ga if not a.startswith("-"))
    if sub in ("rm", "mv"):
        return named
    discards = {"clean": True, "restore": True, "stash": True,
                "reset": any(a in ("--hard", "--merge", "--keep") for a in ga),
                "checkout": "--" in ga or "-f" in ga or "--force" in ga or named,
                "switch": any(a in ("-f", "--force", "--discard-changes") for a in ga)}
    if not discards.get(sub):
        return False
    st = git_out(root, ["status", "--porcelain", "--untracked-files=all", "--", rel])
    if st is None:
        return True                          # git could not say: the contract holds
    if not st.strip():
        return False                         # git holds these bytes unchanged
    untracked = st.startswith("??")
    if sub == "clean":
        return untracked
    if sub == "stash":
        return not untracked or any(a in ("-u", "--include-untracked", "-a", "--all") for a in ga)
    return not untracked


def protected_change(path, contract, tool, ti, effects, cwd):
    """Why this call breaks the contract on path, or '' when it keeps it."""
    if tool in WRITE_TOOLS:
        if not any(same_path(w, path) for e in effects for w in e.get("writes") or []):
            return ""
        if contract == "append-only" and tool in FULL_WRITE_TOOLS:
            body, old = ti.get("content"), _file_bytes(path) or b""
            if isinstance(body, str) and body.encode("utf-8").startswith(old):
                PROTECTED_EVIDENCE.append(prefix_evidence(path))
                return ""                          # the whole old content, then more: a pure append
        return "the %s tool would %s it" % (tool, "replace" if tool in FULL_WRITE_TOOLS else "edit")
    appended = False
    for e in effects:
        seg = str(e.get("seg") or "")[:70]
        for t in e.get("targets") or []:
            if isinstance(t, str) and is_same_or_ancestor(t, path):
                return "`%s` would %s it" % (seg, "move" if e.get("kind") == "move" else "delete")
        if any(isinstance(w, str) and same_path(w, path) for w in e.get("writes") or []):
            if contract == "append-only" and _pure_append(e, path, cwd):
                appended = True
                continue
            return "`%s` would rewrite it" % seg
        if e.get("kind") == "inline" and _inline_writes(e, path):
            return "inline code in `%s` would write to it" % seg
        if e.get("kind") == "git" and guarded(_git_touches, True, e, path, cwd):
            return "`%s` would throw its current bytes away" % seg
    if appended:
        PROTECTED_EVIDENCE.append(prefix_evidence(path))
    return ""


def protected_hits(payload, cwd, effects, cfg):
    """OP-022: a call that would change a protected file in a way its contract does not allow."""
    contracts = file_contracts(cfg, cwd)
    if not contracts:
        return []
    tool = str(payload.get("tool_name") or payload.get("toolName") or "").lower()
    ti = payload.get("tool_input") or {}
    ti = ti if isinstance(ti, dict) else {}
    hits = []
    for path, contract in contracts:
        why = protected_change(path, contract, tool, ti, effects, cwd)
        if why:
            hint = ("Add to the end with `>>` or `%s append %s`; anything else needs the owner." % (self_cmd(), short_home(path))
                    if contract == "append-only" else "Leave it as it is, or ask the owner.")
            hits.append(("OP-022", "%s is %s, and %s" % (short_home(path), contract, why), hint))
    return hits


def receipt_rows():
    """Every receipt row, oldest first."""
    rows, rdir = [], pj(OP_HOME, "receipts")
    try:
        names = sorted(n for n in os.listdir(rdir) if n.endswith(".jsonl"))
    except OSError:
        return rows
    for n in names:
        try:
            with open(pj(rdir, n), encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            continue
    return rows


def protected_check(path, rows):
    """(checked, first_broken_row) for one file: each recorded append says how many bytes were there
    before and their digest; those bytes must still be there, byte for byte."""
    data = _file_bytes(path) or b""
    checked = 0
    for r in rows:
        for ev in ([r] if r.get("decision") == "APPEND" else (r.get("protected") or [])):
            if not isinstance(ev, dict) or not same_path(str(ev.get("path") or ""), path):
                continue
            n = int(ev.get("size_before") or 0)
            checked += 1
            if len(data) < n or hashlib.sha256(data[:n]).hexdigest() != ev.get("sha256_before"):
                return checked, r
    return checked, None


def _when(ts):
    try:
        return time.strftime("%m-%d %H:%M UTC", time.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return str(ts or "?")


def cmd_append(argv):
    """`append <path>`: add stdin to the end of a file under an append-only contract. The receipt, with
    the size and digest of what was there before, is written first: no receipt, no append."""
    if not argv:
        print("usage: %s append <path>   (reads what to add from stdin)" % self_cmd())
        return 2
    cwd = os.getcwd()
    path = norm(argv[0], cwd)
    if not any(same_path(p, path) and c == "append-only" for p, c in file_contracts(load_config(), cwd)):
        print("Refusing: %s has no append-only contract. The owner adds one with `%s protect %s --append-only`."
              % (short_home(path), self_cmd(), short_home(path)))
        return 2
    data = sys.stdin.buffer.read()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        ev = prefix_evidence(path)
        rd = write_receipt(dict(ev, decision="APPEND", source="owner-tty" if sys.stdin.isatty() else "append",
                                contract="append-only", bytes=len(data)))
        if rd is None:
            print("Not appended: the receipt could not be written, and an append without one would be unprovable.")
            return 1
        left = memoryview(data)
        while left:
            n = os.write(fd, left)
            left = left[n:]
        os.fsync(fd)
        after = os.fstat(fd).st_size
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
    if after < ev["size_before"] + len(data):
        write_receipt({"decision": "APPEND_PARTIAL", "path": path, "size_after": after, "of": ev["size_before"] + len(data)})
        print("Appended only part: %d of %d bytes (receipt %s)." % (after - ev["size_before"], len(data), rd[:16]))
        return 1
    print("Appended %d bytes to %s; the %d bytes before them are recorded (receipt %s)."
          % (len(data), short_home(path), ev["size_before"], rd[:16]))
    return 0


CANARY_MIN = 8          # a planted value shorter than this would match ordinary text


def planted_canaries():
    """The values the owner planted with `canary plant`, from <gate home>/canaries.json."""
    try:
        with open(pj(OP_HOME, "canaries.json"), encoding="utf-8") as f:
            d = json.load(f)
        rows = d.get("canaries") if isinstance(d, dict) else None
        return [c for c in rows or [] if isinstance(c, dict) and len(str(c.get("value") or "")) >= CANARY_MIN]
    except (OSError, ValueError):
        return []


def _strings(x, out):
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _strings(v, out)
    return out


def planted_hits(ti):
    """OP-CANARY for a planted value anywhere in the call: a command, a file's content, a URL, MCP arguments.
    A decoy no legitimate lookup reaches was read and is now being used."""
    rows = planted_canaries()
    if not rows:
        return []
    texts = _strings(ti, [])
    for c in rows:
        v = str(c["value"])
        if any(v in t for t in texts):
            return [("OP-CANARY", "a value the owner planted as a canary (%s) is in this call"
                     % (c.get("label") or "planted %s" % str(c.get("planted") or "")[:10]),
                     "Nothing legitimate uses a planted canary: stop, and tell the owner where the value came from.")]
    return []


def cmd_canary(argv):
    """`canary plant <value> [--label <text>]`, `canary remove <value>`, `canary list`: decoy values the owner
    plants where only a leak or an injected instruction would find them. The owner runs plant and remove."""
    sub = argv[0] if argv else "list"
    path = pj(OP_HOME, "canaries.json")
    rows = planted_canaries()
    if sub == "list":
        if not rows:
            print("No planted canaries. The owner plants one with `%s canary plant <value> --label <where it sits>`." % self_cmd())
            return 0
        for c in rows:
            v = str(c["value"])
            print("%s...%s  %-30s planted %s" % (v[:3], v[-2:], c.get("label") or "", str(c.get("planted") or "")[:19]))
        return 0
    if sub not in ("plant", "remove") or len(argv) < 2:
        print("usage: %s canary plant <value> [--label <text>] | canary remove <value> | canary list" % self_cmd())
        return 2
    if any(os.environ.get(k) for k in AGENT_ENV):
        print("Refusing: canaries are the owner's, so the owner runs %s from their own terminal." % sub)
        return 3
    value = argv[1]
    if sub == "plant":
        if len(value) < CANARY_MIN:
            print("A canary needs at least %d characters, so ordinary text never matches it." % CANARY_MIN)
            return 2
        label = argv[argv.index("--label") + 1] if "--label" in argv[:-1] else ""
        rows = [c for c in rows if str(c["value"]) != value] + [
            {"value": value, "label": label[:80], "planted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    else:
        if not any(str(c["value"]) == value for c in rows):
            print("No such canary.")
            return 2
        rows = [c for c in rows if str(c["value"]) != value]
    os.makedirs(OP_HOME, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"canaries": rows}, f, indent=1)
    os.replace(tmp, path)
    write_receipt({"decision": "CANARY_PLANTED" if sub == "plant" else "CANARY_REMOVED",
                   "canary": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], "source": "owner-tty"})
    print("Canary %s. A tool call that carries it is stopped under OP-CANARY." % ("planted" if sub == "plant" else "removed"))
    return 0


def cmd_protected(argv):
    """`protected` lists the owner's file contracts and checks every one against its receipts."""
    contracts = file_contracts(load_config(), os.getcwd())
    if not contracts:
        print("No protected files. The owner adds one with `%s protect <path> --append-only` (or --read-only)." % self_cmd())
        return 0
    rows, bad = receipt_rows(), 0
    for path, contract in contracts:
        checked, broken = protected_check(path, rows)
        if broken:
            bad += 1
            print("BROKEN  %s (%s): bytes recorded before an append at %s have changed (receipt %s)"
                  % (short_home(path), contract, broken.get("ts"), str(broken.get("digest") or "")[:16]))
        elif checked:
            print("OK      %s (%s): its earlier bytes match all %d recorded append%s"
                  % (short_home(path), contract, checked, "" if checked == 1 else "s"))
        else:
            print("OK      %s (%s): no appends recorded yet" % (short_home(path), contract))
    return 1 if bad else 0


def cmd_protect(argv):
    """`protect <path> --append-only|--read-only|--remove`: the owner names a file and its contract."""
    if any(os.environ.get(k) for k in AGENT_ENV):
        print("Refusing: protect changes the owner's rules, so the owner runs it from their own terminal.")
        return 3
    flags = [a for a in argv if a.startswith("--")]
    paths = [a for a in argv if not a.startswith("--")]
    want = {"--append-only": "append-only", "--read-only": "read-only", "--remove": None}
    if len(paths) != 1 or len(flags) != 1 or flags[0] not in want:
        print("usage: %s protect <path> --append-only|--read-only|--remove" % self_cmd())
        return 2
    path = norm(paths[0], os.getcwd())
    cpath = pj(OP_HOME, "commitments.json")
    try:
        with open(cpath, encoding="utf-8") as f:
            user = json.load(f)
    except FileNotFoundError:
        user = {}
    except ValueError:
        print("Refusing: %s does not parse; fix it first, it was left as it is." % cpath)
        return 1
    keep = [c for c in user.get("file_contracts") or [] if not (isinstance(c, dict) and same_path(norm(str(c.get("path") or ""), "/"), path))]
    if want[flags[0]]:
        keep.append({"path": path, "contract": want[flags[0]]})
    user["file_contracts"] = keep
    os.makedirs(OP_HOME, exist_ok=True)
    tmp = cpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(user, f, indent=2)
    os.replace(tmp, cpath)
    write_receipt({"decision": "CONTRACT", "path": path, "contract": want[flags[0]] or "removed", "source": "owner-tty"})
    print("%s is now %s." % (short_home(path), want[flags[0]] or "unprotected"))
    return 0


def xargs_substituted(cmd):
    """The commands `xargs -I R cmd` runs when its input is a here-string or a literal echo/printf.
    Heredoc bodies and quoted text are data: an xargs written there is not the one that runs."""
    found = []
    cmd = extract_heredocs(cmd)[0]
    masked = mask_quotes(cmd)
    live = lambda at: masked[at:at + 5] == "xargs"
    for m in re.finditer(r"\bxargs\b" + _XARGS_OPTS + r"\s+-I\s*(\S+)\s+([^|;&]+?)\s*<<<\s*('[^']*'|\"[^\"]*\"|[^\s;&|]+)", cmd):
        if live(m.start()):
            found.append((m.group(2), m.group(3), m.group(4)))
    for m in re.finditer(r"(?:^|[;&|]\s*)(?:echo|printf(?:\s+'[^']*'|\s+\"[^\"]*\")?)\s+([^|;&]+?)\s*\|\s*(xargs)\b"
                         + _XARGS_OPTS + r"\s+-I\s*(\S+)\s+([^|;&]+)", cmd):
        if live(m.start(2)):
            found.append((m.group(4), m.group(5), m.group(1)))
    out = []
    for ph, body, word in found:
        try:
            words = shlex.split(word)
        except ValueError:
            words = [word]
        for w in words[:8]:
            w = w if re.fullmatch(r"[\w@%+=:,./~-]+", w) else shlex.quote(w)
            out.append(body.replace(ph, w).strip())
    return list(dict.fromkeys(out))


# ---- memory at the gate: what this session was already stopped for, and keys it found ------------ #
LOSS_RULES = {"OP-001", "OP-002", "OP-003", "OP-011", "OP-012", "OP-014"}


# a recursive delete stopped under these is the same delete when it goes piece by piece; a workspace
# root (OP-001) is different: deleting its children one by one is that rule's own compliant path
PIECEWISE_RULES = {"OP-002", "OP-003", "OP-012"}


KEY_VALUE_RES = (
    re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}|sk_live_[A-Za-z0-9]{16,}|rk_live_[A-Za-z0-9]{16,}"
               r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|glpat-[A-Za-z0-9_\-]{20,}"
               r"|xox[abprs]-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}"
               r"|eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"),
    re.compile(r"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:TOKEN|API_?KEY|SECRET|PASSWORD|PASSWD|AUTH)[A-Z0-9_]*\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{16,})"),
)


CANDIDATE_RE = re.compile(r"[A-Za-z0-9_\-./+=]{16,}")


CLOUD_CLI = {"railway", "fly", "flyctl", "vercel", "heroku", "netlify", "aws", "gcloud", "az", "doctl", "kubectl",
             "terraform", "pulumi", "supabase", "firebase", "stripe", "render", "wrangler", "neonctl", "turso", "planetscale", "pscale"}


REMOTE_CHANGE_RE = re.compile(r"(?i)(?:-X|--request)[= ]?\s*['\"]?(?:DELETE|PUT|PATCH|POST)\b|\bmutation\b"
                              r"|\b(?:delete|destroy|drop|purge|wipe|remove|terminate|teardown|down|deploy|apply|scale)\b")


def key_prints(text):
    """Short fingerprints of the credential-shaped values in a text. The values are never stored."""
    prints = set()
    for rx in KEY_VALUE_RES:
        for m in rx.finditer(text or ""):
            v = m.group(1)
            if v and len(v) >= 16:
                prints.add(hashlib.sha256(v.encode("utf-8", "replace")).hexdigest()[:16])
    return prints


def file_key_prints(path):
    try:
        if os.path.isfile(path) and os.path.getsize(path) <= 262144:
            with open(path, encoding="utf-8", errors="replace") as f:
                return key_prints(f.read())
    except Exception:
        pass
    return set()


def chain_note_stop(sid, effects, meta, rule, receipt):
    """Remember what a STOP was aimed at, so a different spelling of the same action meets it."""
    path = _chain_path(sid or "")
    targets, writes = [], []
    for e in effects or []:
        if e.get("kind") == "delete":
            targets += e.get("targets") or []
        writes += e.get("writes") or []
    targets += (meta or {}).get("lite_deletes") or []
    writes += (meta or {}).get("lite_writes") or []
    # a write is a target only when it replaces a file that is there: a file the stopped call would have
    # made fresh (a script it meant to run) loses nothing when it is written again
    targets += [w for w in writes if isinstance(w, str) and os.path.isfile(w)]
    targets = [t for t in dict.fromkeys(targets) if isinstance(t, str) and t.startswith("/")][:20]
    if not path or not targets:
        return
    mark = {"kind": "stopped", "targets": targets, "rule": rule, "receipt": (receipt or "")[:16],
            "preview": redact(" ".join(str((meta or {}).get("normalized") or "").split()))[:140], "t": time.time(),
            "spelling": str((meta or {}).get("spelling") or ""),
            "rules": list((meta or {}).get("stop_rules") or [rule])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(mark, sort_keys=True) + "\n")


MACHINE_LEDGER_MAX = 4000                # lines kept in the shared ledger


def _machine_path():
    return pj(OP_HOME, "state", "chain", "_machine.jsonl")


def machine_note(mark):
    """Add a stop to the ledger every session on this machine reads, trimmed to its newest lines."""
    path = _machine_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(mark, sort_keys=True) + "\n")
    if os.path.getsize(path) > 2 * 1024 * 1024:
        with open(path, encoding="utf-8") as f:
            keep = f.readlines()[-MACHINE_LEDGER_MAX // 2:]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)


def machine_stops(sid):
    """Targets the gate stopped in other sessions on this machine, inside the look-back window."""
    path, out, now = _machine_path(), [], time.time()
    if not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-MACHINE_LEDGER_MAX:]
    except OSError:
        return out
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if isinstance(m, dict) and m.get("kind") == "stopped" and m.get("targets") and m.get("sid") != sid \
                and now - float(m.get("t", 0)) <= CHAIN_WINDOW_S:
            out.append(m)
    return out


def _home_forms(path):
    """The ways a command or script names a path under the home folder."""
    if HOME and path.startswith(HOME + "/"):
        rest = path[len(HOME):]
        return {path, "~" + rest, "$HOME" + rest, "${HOME}" + rest}
    return {path}


def _empties(folder, deletes):
    """True when these deletes take everything visible in folder: a wildcard over it, or every one of its
    entries named. Clearing a few chosen entries (build output) is not emptying it."""
    if any(isinstance(d, str) and d.endswith("/*") and same_path(d[:-2], folder) for d in deletes):
        return True
    try:
        entries = [n for n in os.listdir(folder) if not n.startswith(".")]
    except OSError:
        return False
    return bool(entries) and all(any(same_path(d, pj(folder, n)) for d in deletes) for n in entries)


def spelling_of(payload, cwd):
    """The exact call: its tool, working folder and input. Sent again unchanged it is the same action, not
    a second spelling of it, so it keeps its action id and the owner's permit for that id still applies."""
    p = payload or {}
    ti = p.get("tool_input") or p.get("toolInput") or {}
    tool = str(p.get("tool_name") or p.get("toolName") or "")
    return hashlib.sha256(json.dumps([tool, cwd, ti], sort_keys=True, ensure_ascii=False, default=str)
                          .encode("utf-8")).hexdigest()[:16]


def second_spelling(payload, cwd, effects, meta, stopped, cross=False):
    """OP-020: this call reaches a target the gate stopped earlier by another route. `stopped` holds this
    session's marks, or with cross=True the marks of other sessions on the machine, where only a route
    that deletes, empties or overwrites the target counts, never merely naming it. In this session, the
    stopped call itself sent again is not another route: its own rules judge it, as they did the first time."""
    tool = str(payload.get("tool_name") or payload.get("toolName") or "").lower()
    if tool in WRITE_TOOLS:
        return []                                             # editing files is the compliant path, never a respelling
    same = "" if cross else guarded(spelling_of, "", payload, cwd)
    prior = [m for m in stopped if same and m.get("spelling") == same]
    if prior:
        # this exact call was stopped before: it gets that verdict again, so it keeps its action id
        if "OP-020" not in (prior[-1].get("rules") or [prior[-1].get("rule")]):
            return []                                         # its own rule stopped it, and judges it again
        stopped = [m for m in stopped if m.get("spelling") != same]
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    cmd = str((ti.get("command") or ti.get("cmd") or "") if isinstance(ti, dict) else ti)
    deletes, overwrites, opaque, texts = set(), set(), False, []
    for e in effects or []:
        if e.get("kind") == "delete":
            deletes.update(e.get("targets") or [])
        overwrites.update(e.get("writes") or [])
        if e.get("unresolved") or e.get("kind") == "inline":
            opaque = True
    deletes.update((meta or {}).get("lite_deletes") or [])
    overwrites.update((meta or {}).get("lite_writes") or [])
    deleting = bool(deletes) or any(e.get("kind") == "delete" for e in effects or []) \
        or bool(re.search(r"(^|[\s;&|(`])(rm|rmdir|unlink|shred|find|rsync|trash)\b", cmd))
    for seg in split_segments(cmd) if cmd else []:
        _, _, _, bare, _ = root_form(tokenize(seg))
        if not bare:
            continue
        if code_prog(os.path.basename(bare[0])) and any(a in CODE_FLAGS for a in bare[1:]):
            opaque = True
        for ran in guarded(chain_runs, [], bare, seg, cwd):
            if os.path.isfile(ran) and os.path.getsize(ran) <= 262144:
                with open(ran, encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
                opaque = True
    if opaque and cmd:
        texts.append(cmd)
    wild = []
    if cross:
        # another session's stop counts only for a route the ordinary rules cannot read (a script, inline code,
        # an unresolved command): a direct command the rules judged is left to them. Only the paid reader's
        # deletes and writes count, and Lite's reading only to see a folder emptied; a syntax check runs nothing.
        if not opaque:
            return []
        deletes = {d for e in effects or [] if e.get("kind") == "delete" for d in e.get("targets") or []}
        overwrites = {w for e in effects or [] for w in e.get("writes") or []}
        if not re.search(r"(^|[\s;&|(])(?:ba|z|k|da)?sh\s+-n\b", cmd):
            wild = [d for d in (meta or {}).get("lite_deletes") or [] if isinstance(d, str)]
    for m in reversed(stopped):
        for t in m.get("targets") or []:
            # the project or a folder holding it is named by ordinary work all the time; only a delete
            # that covers it, or that empties it with a wildcard, is the stopped action again
            holds_cwd = same_path(t, cwd) or is_inside(cwd, t)
            forms = {f for f in _home_forms(t) if len(f) > 3}
            how = None
            if any(same_path(d, t) or is_inside(t, d) for d in deletes):
                how = "deletes"
            elif cross and _empties(t, wild):
                how = "empties"
            elif not cross and not holds_cwd and m.get("rule") in PIECEWISE_RULES and any(is_inside(d, t) for d in deletes):
                how = "deletes what is inside"                # piece by piece is the same delete
            elif deleting and any(re.search(re.escape(f) + r"/(\*|\.\*|\{)", cmd) for f in forms):
                how = "empties"
            elif m.get("rule") in LOSS_RULES and any(same_path(w, t) for w in overwrites) \
                    and not re.search(r">>\s*['\"]?[^\s;&|]*" + re.escape(posixpath.basename(t)) + r"\b", cmd):
                how = "overwrites"                            # an append keeps what is there; a truncation does not
            elif not cross and not holds_cwd and texts and any(f in x for x in texts for f in forms):
                how = "names"
            if how:
                when = time.strftime("%H:%M:%S", time.localtime(float(m.get("t", 0))))
                where = (" in another session (%s, %s)" % (str(m.get("sid") or "?")[:8], m.get("source") or "unknown host")) if cross else ""
                return [("OP-020", "%s %s, which the gate stopped at %s%s under %s (receipt %s, `%s`)"
                         % (how, t, when, where, m.get("rule"), m.get("receipt") or "n/a", str(m.get("preview") or "")[:80]),
                         "Do what the first stop named, or ask the owner to allow that first action once.")]
    return []


def found_key_use(payload, cwd, keys):
    """OP-021: a remote write, delete or deploy carrying a key this session found in a file."""
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    cmd = str((ti.get("command") or ti.get("cmd") or "") if isinstance(ti, dict) else ti)
    if not cmd:
        return []
    prints = {hashlib.sha256(c.encode("utf-8", "replace")).hexdigest()[:16] for c in CANDIDATE_RE.findall(cmd)}
    hit = next((m for m in reversed(keys) if prints & set(m.get("prints") or [])), None)
    if not hit:
        return []
    remote = False
    for seg in split_segments(cmd):
        _, _, _, bare, _ = root_form(tokenize(seg))
        if not bare:
            continue
        prog, args = os.path.basename(bare[0]), bare[1:]
        if _sends_out(prog, args, seg) or (prog in CLOUD_CLI and REMOTE_CHANGE_RE.search(seg)) \
                or (prog in ("curl", "wget", "http", "https", "xh", "gh") and REMOTE_CHANGE_RE.search(seg)):
            remote = True
            break
    if not remote:
        return []
    when = time.strftime("%H:%M:%S", time.localtime(float(hit.get("t", 0))))
    return [("OP-021", "this call carries a key the session found in %s at %s (receipt %s)"
             % (hit.get("path"), when, hit.get("receipt") or "n/a"),
             "Ask the owner which credential this task should use; a key that turned up in a file is not one.")]


def memory_reach(payload, cwd, effects, meta):
    """The rules that need this session's memory: OP-020 and OP-021."""
    sid = session_of(payload)
    seen = chain_ledger(sid) if sid else []
    found = []
    stopped = [m for m in seen if m.get("kind") == "stopped" and m.get("targets")]
    if stopped:
        found += guarded(second_spelling, [], payload, cwd, effects, meta, stopped)
    # In Lite a stop holds for the rest of its own session. Carrying the owner's no across sessions,
    # hosts and other agent CLIs is the paid Operator's.
    keys = [m for m in seen if m.get("kind") == "found_key" and m.get("prints")]
    if keys:
        found += guarded(found_key_use, [], payload, cwd, keys)
    return found


# ---- the brake: a session stopped again and again is a runaway, and its changes are held (OP-024) ---- #
BRAKE_AT = 8                             # stops of one session inside the window that trip the brake


BRAKE_WINDOW_S = 300


BRAKE_QUIET_S = 15 * 60                  # the brake lets go after this long without a stop


GIT_READ_SUBS = {"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "grep", "describe",
                 "shortlog", "reflog", "cat-file", "ls-tree", "remote", "branch", "tag", "config",
                 "diff-tree", "diff-index", "diff-files", "ls-remote", "merge-base", "rev-list", "for-each-ref",
                 "show-ref", "name-rev", "check-ignore", "check-attr", "count-objects", "cherry", "range-diff",
                 "whatchanged", "var"}


GIT_READ_VERBS = {"worktree": {"list"}, "stash": {"list", "show"}}   # read only with these verbs: a bare `git stash` pushes


QUIET_SINKS = {"/dev/null", "/dev/stdout", "/dev/stderr"}           # writing here changes nothing on disk


AWK_PROGS = {"awk", "gawk", "mawk", "nawk"}


AWK_WRITE_RE = re.compile(r"\bprintf?\b[^;}\n]*?(?:>|\|)|\bsystem\s*\(|\|&|\|\s*getline")


# Vestige MCP tools that only read, and the action or mode that makes a mixed tool read
VESTIGE_READ_TOOLS = {"recall", "forgotten_lesson", "precedent", "memory_status", "session_start", "selftest", "receipt"}


VESTIGE_READ_CHOICES = {"memory": ("action", {"get", "get_batch", "state"}),
                        "codebase": ("action", {"get_context", "verify"}),
                        "intention": ("action", {"list", "check"}),
                        "dedup": ("action", {"scan", "policy"}),
                        "project": ("action", {"preview"}),
                        "backfill": ("action", {"receipts"}),
                        "ghostlink": ("mode", {"propose", "bounty", "map", "inspect", "explore", "predict"})}


MCP_READ_VERB_RE = re.compile(r"(?:get|list|read|search|describe|view|show|find|lookup)(?:_|$)")


def brake_path():
    return pj(OP_HOME, "state", "brake.json")


def _brake_state():
    try:
        with open(brake_path(), encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _brake_save(st):
    path = brake_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, path)


def brake_at():
    """Stops that trip the brake: OPERATOR_BRAKE_AT in the hook's own environment (the owner's, never an
    agent shell's), default BRAKE_AT; 0 turns the brake off."""
    try:
        return max(0, int(os.environ.get("OPERATOR_BRAKE_AT", BRAKE_AT)))
    except ValueError:
        return BRAKE_AT


def brake_note_stop(sid):
    """Count a stop for this session. True when this stop is the one that trips the brake."""
    if not brake_at():
        return False
    now, st = time.time(), _brake_state()
    e = st.get(sid) or {}
    e["stops"] = [t for t in e.get("stops") or [] if now - float(t) <= BRAKE_WINDOW_S] + [now]
    e["last_stop"] = now
    tripped = not e.get("engaged") and len(e["stops"]) >= brake_at()
    if tripped:
        e["engaged"] = now
    st[sid] = e
    st = {k: v for k, v in st.items() if now - float(v.get("last_stop") or 0) <= max(BRAKE_QUIET_S, BRAKE_WINDOW_S)}
    _brake_save(st)
    return tripped


def brake_engaged(sid):
    """The brake record of this session when its brake is on, else None."""
    e = _brake_state().get(sid) if sid else None
    if not e or not e.get("engaged") or time.time() - float(e.get("last_stop") or 0) > BRAKE_QUIET_S:
        return None
    return e


def mcp_reads_only(tool, tool_input):
    """An MCP call reads when its tool is a known Vestige read (or a mixed tool called with a reading action or
    mode), or another server's tool whose name starts with a reading verb. Anything else counts as a change."""
    parts = str(tool or "").lower().split("__")
    server, name = (parts[1] if len(parts) > 2 else ""), parts[-1]
    ti = tool_input if isinstance(tool_input, dict) else {}
    if "vestige" in server:
        if name in VESTIGE_READ_TOOLS:
            return True
        if name == "causal_walk":
            return ti.get("promote") is not True          # promote records trail edges on the legacy engine
        if name == "backfill" and ti.get("dry_run") is True:
            return True
        key, reads = VESTIGE_READ_CHOICES.get(name, (None, ()))
        return bool(key) and str(ti.get(key) or "") in reads
    return bool(MCP_READ_VERB_RE.match(name))


def awk_reads_only(e):
    """An awk program that only prints to standard output reads: no `print > file`, `print | cmd`, system(),
    `cmd | getline`, program file or in-place edit."""
    if posixpath.basename(str(e.get("prog") or "")) not in AWK_PROGS:
        return False
    if any(str(f).startswith(("-f", "-i", "-l", "-E", "--file", "--include", "--load", "--exec", "--source"))
           for f in e.get("flags") or []):
        return False
    return not any(AWK_WRITE_RE.search(str(a)) or "inplace" in str(a) for a in e.get("args") or [])


def git_reads_only(e):
    """A git call reads when its subcommand only reads, or takes a reading verb (`worktree list`, `stash show`)."""
    sub = str(e.get("sub") or "")
    if sub in GIT_READ_SUBS:
        return True
    args = [str(a) for a in e.get("args") or []]
    verbs = [a for a in args[args.index(sub) + 1:] if not a.startswith("-")] if sub in args else []
    return bool(verbs) and verbs[0] in GIT_READ_VERBS.get(sub, ())


def changes_anything(tool, effects, tool_input=None):
    """False only for a call that reads: no write, delete, move, code that could write, outbound send or
    changing git. An MCP call is judged by its tool and action (mcp_reads_only)."""
    t = str(tool or "").lower()
    if t.startswith("mcp__"):
        return not mcp_reads_only(t, tool_input)
    if t in WRITE_TOOLS:
        return True
    for e in effects or []:
        if any(str(w) not in QUIET_SINKS for w in e.get("writes") or []):
            return True
        if e.get("targets") or e.get("kind") in ("delete", "move") or e.get("unresolved"):
            return True
        if e.get("kind") == "inline" and not awk_reads_only(e):
            return True
        if e.get("kind") == "git" and not git_reads_only(e):
            return True
        prog = posixpath.basename(str(e.get("prog") or ""))
        if prog in HTTP_PROGS and _sends_out(prog, [str(a) for a in e.get("args") or []] + [str(f) for f in e.get("flags") or []], str(e.get("seg") or "")):
            return True
    return False


def brake_hit(payload, tool, effects):
    """OP-024: this session's brake is on, and the call would change something."""
    sid = session_of(payload)
    e = brake_engaged(sid)
    if not e or not changes_anything(tool, effects, payload.get("tool_input")):
        return []
    left = max(1, int((BRAKE_QUIET_S - (time.time() - float(e.get("last_stop") or 0))) // 60))
    return [("OP-024", "this session was stopped %d times inside %d minutes, so its changes are held until it goes %d minutes "
             "without a stop (about %d left) or the owner releases it" % (len(e.get("stops") or []), BRAKE_WINDOW_S // 60,
                                                                         BRAKE_QUIET_S // 60, left),
             "Stop and tell the owner what you were trying to do. They release it with `%s brake --release %s`." % (self_cmd(), sid))]


def cmd_brake(argv):
    """`brake` lists sessions whose brake is on; `brake --release <session>` lets one go (the owner)."""
    if argv[:1] == ["--release"]:
        if any(os.environ.get(k) for k in AGENT_ENV):
            print("Refusing: releasing the brake is the owner's call, so the owner runs it from their own terminal.")
            return 3
        if len(argv) < 2:
            print("usage: %s brake --release <session>" % self_cmd())
            return 2
        st = _brake_state()
        if st.pop(argv[1], None) is None:
            print("No brake on %s." % argv[1])
            return 2
        _brake_save(st)
        write_receipt({"decision": "BRAKE_RELEASED", "released": argv[1], "source": "owner-tty"})
        print("Released the brake on %s." % argv[1])
        return 0
    on = [(k, v) for k, v in _brake_state().items() if brake_engaged(k)]
    if not on:
        print("No session's brake is on. It trips when one session is stopped %d times inside %d minutes." % (BRAKE_AT, BRAKE_WINDOW_S // 60))
        return 0
    for sid, e in on:
        print("ON  %s  %d stops, last %s; release: %s brake --release %s"
              % (sid, len(e.get("stops") or []), time.strftime("%H:%M:%S", time.localtime(float(e.get("last_stop") or 0))), self_cmd(), sid))
    return 0


# ---- watchstander turnover: a compacted session takes the watch before it changes anything ------------ #
# At a compaction the conversation keeps only a summary. The PreCompact hook seals what this session hands
# over, read from the gate's own receipts; SessionStart (compact) gives it to the session that continues;
# until that session takes the watch by its id, its changing calls are held.
TURNOVER_DAYS = 3                        # day files of receipts a turnover reads for the session's record


TURNOVER_KEEP_S = 7 * 24 * 3600          # a turnover never taken stops holding the session after a week


TURNOVER_LIST = 12                       # stopped actions, allowed actions and kept copies a turnover names


TURNOVER_ACK_RE = re.compile(r"\s*(?:python3?\s+)?\S*operator-gate(?:\.py)?\s+turnover\s+--(?:ack|show)\s+[0-9a-f]{16}\s*")


HOLD_RULES = {"OP-024", "OP-025"}        # holds on a session, not verdicts on a target: no ledger mark, no drafted law


def turnover_on():
    """OPERATOR_TURNOVER=0 in the hook's own environment turns turnover off."""
    return os.environ.get("OPERATOR_TURNOVER", "1").strip() != "0"


def turnover_path(sid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid or "")[:80]
    return pj(OP_HOME, "state", "turnover", safe + ".json") if safe else None


def _turnover_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) and isinstance(st.get("chit"), dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _turnover_save(path, st):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def session_rows(sid):
    """This session's receipt rows in the last TURNOVER_DAYS day files, oldest first."""
    rdir, rows = pj(OP_HOME, "receipts"), []
    try:
        names = sorted(n for n in os.listdir(rdir) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", n))[-TURNOVER_DAYS:]
    except OSError:
        return rows
    needle = json.dumps(sid, ensure_ascii=False)
    for n in names:
        try:
            with open(pj(rdir, n), encoding="utf-8") as f:
                for line in f:
                    if needle not in line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get("session") == sid:
                        rows.append(r)
        except OSError:
            continue
    return rows


def open_permits(sid):
    """Permits the owner granted to this session that are still unused and unexpired."""
    out, pdir, now = [], pj(OP_HOME, "permits"), time.time()
    try:
        names = sorted(n for n in os.listdir(pdir) if re.fullmatch(r"[0-9a-f]{24}\.json", n))
    except OSError:
        return out
    for n in names:
        try:
            with open(pj(pdir, n), encoding="utf-8") as f:
                pm = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(pm, dict) and pm.get("session") == sid and float(pm.get("expires") or 0) > now:
            out.append({"action_id": str(pm.get("action_digest") or n[:24]), "expires": int(float(pm["expires"]))})
    return out


def turnover_chit(sid, trigger, cwd):
    """What this session hands over, from the gate's own record: the actions it was stopped from, what the
    owner allowed, permits still open, copies Rewind kept, its brake, the laws and protected files in force,
    and how many stop marks other sessions on this machine left."""
    stops, allowed, kept = {}, [], []
    for r in session_rows(sid):
        d = r.get("decision")
        if d == "STOP" and r.get("commitment") not in HOLD_RULES:
            key = r.get("action_digest") or r.get("digest")
            s = stops.setdefault(key, {"rule": str(r.get("commitment") or ""),
                                       "name": str(r.get("rule") or ("your law" if r.get("layer") == "taste" else "")),
                                       "action": str(r.get("action_preview") or "")[:160],
                                       "action_id": str(r.get("action_digest") or ""), "times": 0})
            s["times"] += 1
            s["receipt"], s["ts"] = str(r.get("digest") or "")[:16], str(r.get("ts") or "")
        elif d == "ALLOW" and r.get("via") == "permit":
            allowed.append({"rule": ",".join(r.get("commitments") or []) or str(r.get("commitment") or ""),
                            "action": str(r.get("action_preview") or "")[:160],
                            "receipt": str(r.get("digest") or "")[:16]})
        if isinstance(r.get("rewind"), dict) and r["rewind"].get("id"):
            kept.append(str(r["rewind"]["id"]))
    law, laws = active_laws()
    cfg = guarded(load_config, {}) or {}
    return {"session": sid, "trigger": str(trigger or ""), "gate_version": VERSION,
            "sealed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stops": sorted(stops.values(), key=lambda s: s.get("ts") or "")[-TURNOVER_LIST:],
            "allowed": allowed[-TURNOVER_LIST:], "permits": open_permits(sid), "rewind": kept[-TURNOVER_LIST:],
            "brake": bool(brake_engaged(sid)),
            "laws": [str(c.get("commitment_id")) for c in laws][:24], "law_digest": str(law.get("law_digest") or "")[:12],
            "protected": ["%s (%s)" % (p, c) for p, c in guarded(file_contracts, [], cfg, cwd)][:TURNOVER_LIST]}


def chit_weight(chit):
    """What a chit hands over that belongs to this session alone. Laws and protected files come back at
    every session start anyway, so on their own they hold nothing."""
    return (len(chit.get("stops") or []) + len(chit.get("allowed") or []) + len(chit.get("permits") or [])
            + len(chit.get("rewind") or []) + (1 if chit.get("brake") else 0))


def turnover_seal(payload, source):
    """PreCompact: seal what this session hands over as a receipt, and keep it for the session that continues."""
    sid = session_of(payload)
    path = turnover_path(sid)
    if not turnover_on() or not path:
        return None
    chit = turnover_chit(sid, payload.get("trigger"), canon(payload.get("cwd") or os.getcwd()))
    cid = hashlib.sha256(json.dumps(chit, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    holds = chit_weight(chit) > 0
    rd = write_receipt({"decision": "TURNOVER_SEALED", "session": sid, "chit": cid, "source": source,
                        "trigger": chit["trigger"], "stops": len(chit["stops"]), "allowed": len(chit["allowed"]),
                        "permits": len(chit["permits"]), "holds": holds})
    st = {"id": cid, "chit": chit, "sealed_t": time.time(), "receipt": (rd or "")[:16], "holds": holds, "acked": None}
    _turnover_save(path, st)
    return st


def turnover_pending(sid):
    """The turnover this session has not taken yet, when it hands anything over; else None."""
    st = _turnover_load(turnover_path(sid)) if sid and turnover_on() else None
    if not st or st.get("acked") or not st.get("holds") or time.time() - float(st.get("sealed_t") or 0) > TURNOVER_KEEP_S:
        return None
    return st


def turnover_text(st):
    """The chit as the session that continues reads it."""
    c = st["chit"]
    n = lambda k: len(c.get(k) or [])
    lines = ["Watch turnover %s for session %s, sealed %s at a %s compaction (Vestige Operator %s). This is the "
             "gate's own record, not the summary:" % (st["id"], str(c.get("session"))[:8], _when(c.get("sealed")),
                                                      c.get("trigger") or "", c.get("gate_version"))]
    if n("stops"):
        lines.append("Stopped in this session (%d action%s). Do not try these again, or another way round:"
                     % (n("stops"), "" if n("stops") == 1 else "s"))
        lines += ["  - %s %s: `%s`%s, receipt %s" % (s.get("rule"), s.get("name"), s.get("action"),
                                                    " (%d times)" % s["times"] if s.get("times", 0) > 1 else "",
                                                    s.get("receipt") or "?") for s in c["stops"]]
    if n("allowed"):
        lines.append("Allowed once by the owner, and already used:")
        lines += ["  - %s: `%s`, receipt %s" % (a.get("rule"), a.get("action"), a.get("receipt")) for a in c["allowed"]]
    if n("permits"):
        lines.append("Permits still open for this session (each lets one exact action through once):")
        lines += ["  - action %s, expires %s" % (p.get("action_id"), time.strftime("%H:%M", time.localtime(p.get("expires") or 0)))
                  for p in c["permits"]]
    if n("rewind"):
        lines.append("Rewind kept %d cop%s of work before calls destroyed it; `%s rewind` lists them."
                     % (n("rewind"), "y" if n("rewind") == 1 else "ies", self_cmd()))
    if c.get("brake"):
        lines.append("The brake is on for this session: its changes stay held until it goes quiet or the owner releases it.")
    if n("laws"):
        lines.append("Owner laws in force (law digest %s): %s." % (c.get("law_digest"), ", ".join(c["laws"])))
    if n("protected"):
        lines.append("Protected files: %s." % "; ".join(c["protected"]))
    if c.get("other_sessions"):
        lines.append("Other sessions on this machine left %d stop mark%s in the last %d hours; reaching one of those "
                     "targets from here is stopped too." % (c["other_sessions"], "" if c["other_sessions"] == 1 else "s",
                                                           CHAIN_WINDOW_S // 3600))
    return lines


def turnover_hit(payload, tool, effects):
    """OP-025: this session was compacted and has not taken the watch, and the call would change something.
    Held are the calls the gate judges by their effects, shell commands and file writes; MCP calls keep to
    their own rules, so the session can still read its memory while it takes the watch."""
    st = turnover_pending(session_of(payload))
    if not st or str(tool or "").lower().startswith("mcp__") or not changes_anything(tool, effects):
        return []
    ti = payload.get("tool_input") or {}
    cmd = str(ti.get("command") or "") if isinstance(ti, dict) else ""
    if cmd and len(split_segments(cmd)) == 1 and TURNOVER_ACK_RE.fullmatch(cmd):
        return []                                    # taking the watch is the way out of the hold
    c = st["chit"]
    k_s, k_a, k_p = len(c.get("stops") or []), len(c.get("allowed") or []), len(c.get("permits") or [])
    return [("OP-025", "this session was compacted at %s and has not taken the watch yet: turnover %s hands over %d "
             "stopped action%s, %d the owner allowed and %d open permit%s"
             % (_when(c.get("sealed")), st["id"], k_s, "" if k_s == 1 else "s", k_a, k_p, "" if k_p == 1 else "s"),
             "Take the watch first: run `%s turnover --ack %s`. It prints the record; then make this call again."
             % (self_cmd(), st["id"]))]


def turnover_session_start(argv):
    """SessionStart after a compaction: the session that continues is handed the watch from the gate's own
    record, in the shape its host reads. It reads only, prints nothing when there is nothing to hand over,
    and the session starts whatever happens here."""
    host = argv[argv.index("--host") + 1] if "--host" in argv[:-1] else "claude"
    try:
        raw = read_stdin()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    st = guarded(turnover_pending, None, session_of(payload)) if payload.get("source") == "compact" else None
    if not st:
        return 0
    text = "\n".join(guarded(turnover_text, [], st) + [
        "Before changing anything, take the watch: run `%s turnover --ack %s`. The gate holds this session's "
        "changes until then; reads go ahead." % (self_cmd(), st["id"])])
    if host == "cursor":
        print(json.dumps({"additional_context": text}))
    else:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))
    return 0


def cmd_turnover(argv):
    """`turnover --pre-compact` (the PreCompact hook) seals what this session hands over; `turnover
    --session-start` (the SessionStart hook) hands it to the session that continues; `turnover --ack <id>`
    takes the watch and prints it; `turnover --show <id>` prints one; `turnover` lists them."""
    if "--session-start" in argv:
        return turnover_session_start(argv)
    if argv[:1] == ["--pre-compact"]:
        source = argv[argv.index("--source") + 1] if "--source" in argv[:-1] else "claude"
        try:
            raw = read_stdin()
            payload = json.loads(raw) if raw.strip() else {}
            if isinstance(payload, dict) and not os.path.exists(pj(OP_HOME, "DISABLED")) and global_mode() != "off":
                turnover_seal(payload, source)
        except Exception:
            pass                                     # a compaction is never blocked or delayed by the gate
        return 0
    tdir, found = pj(OP_HOME, "state", "turnover"), []
    try:
        names = sorted(n for n in os.listdir(tdir) if n.endswith(".json"))
    except OSError:
        names = []
    for name in names:
        st = _turnover_load(pj(tdir, name))
        if st:
            found.append((pj(tdir, name), st))
    if argv[:1] in (["--ack"], ["--show"]):
        if len(argv) < 2 or not re.fullmatch(r"[0-9a-f]{16}", argv[1]):
            print("usage: %s turnover --ack <16-hex turnover id>" % self_cmd())
            return 2
        hit = next(((p, st) for p, st in found if st.get("id") == argv[1]), None)
        if not hit:
            print("No turnover %s. `%s turnover` lists them." % (argv[1], self_cmd()))
            return 2
        path, st = hit
        print("\n".join(turnover_text(st)))
        if argv[0] == "--ack":
            if not st.get("acked"):
                st["acked"] = time.time()
                _turnover_save(path, st)
                write_receipt({"decision": "TURNOVER_TAKEN", "session": st["chit"].get("session"), "chit": st["id"],
                               "source": "agent" if any(os.environ.get(k) for k in AGENT_ENV) else "owner-tty"})
            print("Watch taken. This session's changes go ahead again, each judged as usual.")
        return 0
    if not found:
        print("No turnovers yet. The PreCompact hook seals one for a session at each compaction.")
        return 0
    for _, st in sorted(found, key=lambda x: -float(x[1].get("sealed_t") or 0)):
        c = st["chit"]
        state = "taken" if st.get("acked") else ("waiting" if st.get("holds") else "nothing to hand over")
        print("%s  session %s  sealed %s  %d stopped, %d allowed, %d permits  %s"
              % (st["id"], str(c.get("session"))[:8], _when(c.get("sealed")), len(c.get("stops") or []),
                 len(c.get("allowed") or []), len(c.get("permits") or []), state))
    return 0


HTTP_PROGS = {"curl", "wget", "http", "https", "xh"}


GRAPHQL_DESTROY_RE = re.compile(r"(?i)\b\w*(?:delete|destroy|drop|purge|wipe|terminate)\w*\s*\(")


def remote_delete(cmd, cwd):
    """OP-023: an HTTP DELETE, or a GraphQL mutation that deletes, destroys, drops, purges, wipes or
    terminates, sent to a remote API (the PocketOS case). This machine and development hosts are not remote."""
    hint = "Show the owner what this call would remove and let them allow it once; a local or staging endpoint needs no approval."
    for seg in split_segments(cmd):
        _, _, _, bare, _ = root_form(tokenize(seg))
        if not bare or os.path.basename(bare[0]) not in HTTP_PROGS:
            continue
        prog, args = os.path.basename(bare[0]), bare[1:]
        hosts = [h for h in _hosts(seg) if not LOCAL_HOST_RE.match(h)]
        if not hosts:
            continue
        method = None
        for i, a in enumerate(args):
            if a in ("-X", "--request", "--method") and i + 1 < len(args):
                method = args[i + 1]
            elif re.match(r"^-X[A-Za-z]+$", a):
                method = a[2:]
            elif a.startswith(("--request=", "--method=")):
                method = a.split("=", 1)[1]
        if prog in ("http", "https", "xh") and args and args[0].upper() in ("DELETE", "PUT", "POST", "PATCH", "GET"):
            method = args[0]
        if method and method.strip("'\"").upper() == "DELETE":
            return [("OP-023", "an HTTP DELETE to %s" % hosts[0], hint)]
        body = " ".join(args)
        at = re.search(r"\bmutation\b", body)
        m = GRAPHQL_DESTROY_RE.search(body[at.end():]) if at else None
        if m:
            return [("OP-023", "a GraphQL mutation `%s` sent to %s" % (m.group(0).rstrip("( "), hosts[0]), hint)]
    return []


def classify(payload, cfg):
    tool_raw = str(payload.get("tool_name") or payload.get("toolName") or "")
    tool = tool_raw.lower()
    ti = payload.get("tool_input") or payload.get("toolInput") or payload.get("input") or {}
    if not isinstance(ti, dict):
        ti = {"command": ti}
    cwd = canon(payload.get("cwd") or os.getcwd())
    hits, previews = [], []
    meta = {"normalized": "", "transforms": []}
    effects_out = []
    blob = json.dumps(ti, ensure_ascii=False)

    # The canary is written as two adjacent literals, which Python joins, so this file never holds it
    # whole: a host that sends a whole file through the gate (Cursor's StrReplace) can still edit it.
    if "OPERATOR_CANARY_" "STOP_7f3a" in blob:
        hits.append(("OP-CANARY", "canary string present", "Nothing to fix: this proves the gate ran."))
    hits += guarded(planted_hits, [], ti)

    if tool in SHELL_TOOLS or "command" in ti and tool not in WRITE_TOOLS and not tool.startswith("mcp__"):
        cmd = ti.get("command") or ti.get("cmd") or ""
        if isinstance(cmd, list):
            cmd = " ".join(shlex.quote(str(c)) for c in cmd)
        cmd = str(cmd)
        previews.append(cmd)
        work = cmd
        try:
            _, hd_bodies = extract_heredocs(work)
            if hd_bodies:
                meta["transforms"].append("heredoc-extracted")
            if SUBST_RE.search(work):
                meta["transforms"].append("command-substitution-analyzed")
            expanded = expand_vars(work, {"HOME": HOME})
            has_assign = bool(re.search(r"(?:^|[;&|\n]\s*)[A-Za-z_][A-Za-z0-9_]*=[^\s]+", work))
            if expanded != work or (has_assign and "$" in work):
                meta["transforms"].append("variables-expanded")
            if PIPE_SINK_RE.search(work):
                meta["transforms"].append("pipe-sink-analyzed")
            if decode_b64_text(work):
                meta["transforms"].append("base64-decoded")
            meta["normalized"] = expanded[:400]
        except Exception:
            pass
        for eff in (analyze_powershell(cmd, cwd) if tool in POWERSHELL_TOOLS else analyze(cmd, cwd)):
            effects_out.append(eff)
            hits += classify_effect(eff, cfg, cwd)
            if eff.get("via_symlink") and "symlink-escaped" not in meta["transforms"]:
                meta["transforms"].append("symlink-escaped")
            if eff.get("git_hooks") and "execution-hook-analyzed" not in meta["transforms"]:
                meta["transforms"].append("execution-hook-analyzed")
            sub = lite_owner_sub(eff.get("prog", ""), eff.get("args", []))
            if sub is not None:
                hits.append(("OP-000", "agent invoked `operator-lite %s`" % (sub or "install"),
                             "Approvals and installs are run by the owner in their own terminal."))
            for w in eff.get("writes", []):
                hit = hook_settings_hit(w, cmd)
                if hit:
                    hits.append(hit)
        # The floor reads a delete with its command substitutions put back: the analyzer judges each
        # $(...) on its own and leaves a placeholder where its output would land.
        subs = iter(SUBST_RE.findall(cmd))
        for eff in effects_out:
            if eff.get("kind") == "delete" and "__SUBST__" in (eff.get("seg") or ""):
                seg = re.sub("__SUBST__", lambda m: next(subs, "__SUBST__"), eff["seg"])
                hits += guarded(floor_hits, [], dict(eff, seg=seg, unresolved=True), eff.get("cwd") or cwd)
        if tool not in POWERSHELL_TOOLS:
            hits += guarded(root_reach, [], cmd, ti.get("run_in_background") is True, cwd)
            hits += guarded(repo_code_run, [], cmd, cwd)
            hits += guarded(loss_reach, [], cmd, cwd)
            hits += guarded(remote_delete, [], cmd, cwd)
        hits += guarded(windows_wreck, [], cmd)               # read on every shell tool, PowerShell included
        for eff in effects_out:                               # a redirect, cp, tee or sed -i onto an arming file
            for w in eff.get("writes") or []:
                hits += guarded(agent_arming, [], w, None)
        if re.search(r"operator-gate(\.py)?['\"]?\s+(approve|mode|install|uninstall|upgrade\s+--install|protect|"
                     r"brake\s[^;&|\n]*--release|canary\s+(?:plant|remove))\b", cmd):
            hits.append(("OP-000", "agent invoked the gate's owner commands",
                         "Approvals and installs are run by the owner in their own terminal."))
        for m in re.finditer(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", cmd, re.M):
            pp = norm(m.group(1), cwd)
            if touches_gate_home(pp) or any(is_inside(pp, f) for f in self_protected_files()):
                hits.append(("OP-000", "patch touches gate/hook path %s" % pp, "Ask the owner."))
        if re.search(r"(sk_live|rk_live|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})", cmd) and \
                re.search(r">|tee|git\s+(add|commit)", cmd):
            hits.append(("OP-S04", "credential-shaped string written to disk", ""))
    elif tool in WRITE_TOOLS:
        gate_files = self_protected_files() + [norm(p, cwd) for p in cfg["protected_files"]]
        body = write_body_text(tool, ti)
        for p in write_paths(tool, ti, cwd):
            previews.append("%s %s" % (tool_raw, p))
            hits += write_hits(p, gate_files)   # full coverage: OP-000, OP-002, OP-008, OP-S14
            mhits = write_manifest_hits(p, body, cfg, cwd)
            if mhits:
                hits += mhits
                if "write-manifest-analyzed" not in meta["transforms"]:
                    meta["transforms"].append("write-manifest-analyzed")
            effects_out.append({"prog": tool, "seg": "%s %s" % (tool_raw, p), "targets": [],
                                "recursive": False, "kind": None, "filtered": False,
                                "unresolved": False, "flags": [], "args": [],
                                "cwd": cwd, "writes": [p], "text": "%s %s" % (tool_raw, p)})
            hit = hook_settings_hit(p, body)
            if hit:
                hits.append(hit)
        for p in write_paths(tool, ti, cwd):
            hits += guarded(written_code, [], p, body)
            hits += guarded(agent_arming, [], p, body)
        if re.search(r"(sk_live|rk_live|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})", body):
            hits.append(("OP-S04", "credential-shaped string in write", ""))
    elif tool in FETCH_TOOLS:                                # a fetch tool reaches the network without a shell
        target = " ".join(str(ti.get(k) or "") for k in ("url", "prompt"))
        previews.append("%s %s" % (tool_raw, str(ti.get("url") or "")[:200]))
        if CLOUD_METADATA_RE.search(target):
            hits.append(("OP-010", "fetches a cloud metadata credential endpoint",
                         "Metadata endpoints mint instance credentials; fetch nothing from 169.254.169.254-class addresses."))
    elif tool.startswith("mcp__"):
        previews.append(tool_raw)
        mcp_hits = []
        # 1. MCP shell execution delegation (`mcp__shell__run_command`, `mcp__terminal__execute`, etc.)
        if re.search(r"__(?:run_command|execute_command|run_shell|bash|shell|exec|terminal_command|exec_command)(?:_|$)", tool):
            mcmd = str(ti.get("command") or ti.get("cmd") or ti.get("script") or "")
            if mcmd:
                for eff in analyze(mcmd, cwd):
                    effects_out.append(eff)
                    mcp_hits += classify_effect(eff, cfg, cwd)
        # 2. MCP destructive SQL (`mcp__postgres__query`, `mcp__supabase__execute_sql`, `mcp__sqlite__query`, etc.)
        for sk in ("sql", "query", "statement", "statements"):
            sv = ti.get(sk)
            if isinstance(sv, str) and sv.strip():
                lo_s = re.sub(r"pragma\s+wal_checkpoint\s*(\(\s*\w+\s*\))?", "", sv.lower())
                unscoped_u = any(not re.search(r"\bwhere\b", m.group(1))
                                 for m in re.finditer(r"\bupdate\s+[\w.\"`]+\s+set\b([^;]*)", lo_s))
                taut_w = bool(re.search(
                    r"\b(?:delete\s+from|update)\s+[\w.\"`]+\b[^;]*?\bwhere\s+(?:true\b|1\s*=\s*1\b|'(\w*)'\s*=\s*'\1'|\"(\w*)\"\s*=\s*\"\2\")",
                    lo_s))
                if (re.search(r"\bdrop\s+(table|database|schema|index)\b|(?<![\w(])truncate\s+(table\s+)?[\w\"`]|"
                              r"\bflush(all|db)\b|\bdropdatabase\b|\bdb\.[\w.]+\.drop\s*\(", lo_s) or
                        re.search(r"\bdelete\s+from\s+[\w.\"`]+\s*(;|\"|'|$)", lo_s) or unscoped_u or taut_w):
                    mcp_hits.append(("OP-007", "%s executes destructive SQL" % tool_raw,
                                     "Run it as a SELECT first, scope with WHERE, and ask the owner for a permit."))
        # 3. MCP filesystem delete/write/move parity (`mcp__filesystem__delete_file`, `mcp__fs__write_file`, etc.)
        is_mcp_del = bool(re.search(r"__(?:delete|remove|unlink|rmdir|rm|purge|erase|trash)(?:_|$)", tool) and
                          not re.search(r"__(?:delete_memory|delete_entity|delete_observations|delete_relations)$", tool))
        is_mcp_write = bool(re.search(r"__(?:write|edit|create_file|update_file|save|overwrite|put_file|patch|move|rename|copy)(?:_|$)", tool))
        if is_mcp_del or is_mcp_write:
            roots = [norm(r, cwd) for r in cfg["protected_roots"]] + [cwd]
            gate_files = self_protected_files() + [norm(p, cwd) for p in cfg["protected_files"]]
            mpaths = []
            for pk in ("path", "file_path", "filepath", "source", "src", "destination", "dest", "target", "directory", "dir"):
                pv = ti.get(pk)
                if isinstance(pv, str) and pv.strip():
                    mpaths.append(norm(pv, cwd))
                elif isinstance(pv, list):
                    mpaths += [norm(str(x), cwd) for x in pv if isinstance(x, str)]
            for mp in mpaths:
                if is_mcp_del:
                    eff_m = {"prog": tool_raw, "seg": "%s %s" % (tool_raw, mp), "targets": [mp],
                             "recursive": True, "kind": "delete", "filtered": False, "unresolved": False,
                             "flags": [], "args": [mp], "cwd": cwd, "writes": [], "text": "%s %s" % (tool_raw, mp)}
                    effects_out.append(eff_m)
                    mcp_hits += classify_effect(eff_m, cfg, cwd)
                else:
                    mcp_hits += write_hits(mp, gate_files, "%s writes to" % tool_raw)
                    mbody = str(ti.get("content") or ti.get("new_string") or ti.get("text") or "")
                    h_hit = hook_settings_hit(mp, mbody)
                    if h_hit:
                        mcp_hits.append(h_hit)
                    if is_precious_outside_file(mp) and not is_scratch(mp) and not is_inside(mp, cwd):
                        mcp_hits.append(("OP-011", "%s overwrites %s outside the project" % (tool_raw, mp),
                                         "Copy it to a dated backup first, or ask the owner."))
        # 4. MCP cloud/repo teardown parity (`mcp__github__delete_repository`, `mcp__supabase__delete_project`, etc.)
        if re.search(r"__(?:delete_repo|delete_repository|archive_repo|delete_release|delete_branch)(?:_|$)", tool):
            mcp_hits.append(("OP-005", "%s deletes or mutates a repository/release/branch" % tool_raw,
                             "Repository and release deletions are owner-only."))
        if re.search(r"__(?:delete_project|destroy_project|delete_database|drop_database|delete_bucket|delete_cluster|"
                     r"delete_service|delete_volume|delete_environment|delete_namespace)(?:_|$)", tool):
            mcp_hits.append(("OP-006", "%s tears down cloud infrastructure" % tool_raw,
                             "Run a plan or dry-run first, then ask the owner for a one-time permit."))
        if mcp_hits:
            hits += mcp_hits
            if "mcp-cross-channel-analyzed" not in meta["transforms"]:
                meta["transforms"].append("mcp-cross-channel-analyzed")
        if re.search(r"__(send_imessage|send_message|forward|reply)$", tool):
            hits.append(("OP-S02", tool_raw, ""))
        blob_l = blob
        if MCP_SECRET_ARG_RE.search(blob_l):
            hits.append(("OP-S13", "%s carries a credential-shaped argument" % tool_raw,
                         "Secrets never belong in tool arguments; pass a file path the owner approved or nothing."))
        for m in re.finditer(r"(?:^|[,\"{\s])((?:~|\$HOME|/Users/|/home/)[^\"',}\s]*)", blob_l):
            if MCP_SENSITIVE_PATH_RE.search(m.group(1)):
                hits.append(("OP-S13", "%s reads sensitive path %s" % (tool_raw, m.group(1)),
                             "Ask the owner before tool calls that touch credentials or identity files."))
                break
    if not payload.get("_inner"):
        shell_call = tool in SHELL_TOOLS or ("command" in ti and tool not in WRITE_TOOLS and not tool.startswith("mcp__"))
        if shell_call:
            for sub in guarded(xargs_substituted, [], str(ti.get("command") or ti.get("cmd") or "")):
                _, _, sub_hits, _, _, sub_fx = classify(dict(payload, tool_input={"command": sub}, _inner=True), cfg)
                if sub_hits:
                    hits += [(h[0], "xargs -I runs `%s`: %s" % (sub[:80], h[1]), h[2]) for h in sub_hits]
                    effects_out += sub_fx
                    if "xargs-substituted" not in meta["transforms"]:
                        meta["transforms"].append("xargs-substituted")
    hits = list(dict.fromkeys(hits))                         # the same finding reached by two routes is one hit
    return tool_raw, cwd, hits, " | ".join(previews)[:400], meta, effects_out


# --------------------------------------------------------------------------- #
# redaction, receipts, permits
# --------------------------------------------------------------------------- #
SECRET_RE = re.compile(r"(sk_live_\S+|rk_live_\S+|sk-[A-Za-z0-9_\-]{16,}|ghp_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|"
                       r"(?i:(api[_-]?key|token|secret|password)\s*[=:]\s*)\S+|Bearer\s+\S+)")


def redact(s):
    return SECRET_RE.sub("[REDACTED]", s or "")[:200]


def ensure_dirs():
    for d in ("receipts", "permits", "state"):
        os.makedirs(pj(OP_HOME, d), mode=0o700, exist_ok=True)


def lock_file(f):
    if fcntl:
        fcntl.flock(f, fcntl.LOCK_EX)
    else:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)


def unlock_file(f):
    if fcntl:
        fcntl.flock(f, fcntl.LOCK_UN)
    else:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def receipt_tail_digest(rdir):
    """Last complete append, checked under the writer lock. HEAD is only a cache.

    A malformed/incomplete tail is preserved and prevents another append; never skip evidence.
    Historical chain links are left untouched, including previously recorded forks.
    """
    files = sorted(f for f in os.listdir(rdir) if f.endswith(".jsonl"))
    for name in reversed(files):
        with open(pj(rdir, name), "rb") as f:
            end = f.seek(0, os.SEEK_END)
            if not end:
                continue
            f.seek(end - 1)
            if f.read(1) != b"\n":
                raise ValueError("incomplete receipt tail")
            pos, chunks = end - 1, []
            while pos:
                start = max(0, pos - 4096)
                f.seek(start)
                chunk = f.read(pos - start)
                split = chunk.rfind(b"\n")
                chunks.append(chunk[split + 1:] if split >= 0 else chunk)
                if split >= 0:
                    break
                pos = start
            rec = json.loads(b"".join(reversed(chunks)).decode("utf-8"))
        receipt_digest = rec.pop("digest")
        if (not re.fullmatch(r"[0-9a-f]{64}", str(receipt_digest))
                or digest_of(rec) != receipt_digest):
            raise ValueError("invalid receipt tail digest")
        return receipt_digest
    return "0" * 64


def write_receipt(rec):
    """Append a hash-chained receipt. Never raises into the caller."""
    try:
        ensure_dirs()
        rdir = pj(OP_HOME, "receipts")
        head_path = pj(rdir, "HEAD")
        lock = open(pj(rdir, ".lock"), "a+", encoding="utf-8")
        lock_file(lock)
        try:
            prev = receipt_tail_digest(rdir)
            rec = dict(rec)
            rec.update({"v": 1, "integrity": INTEGRITY, "gate_version": VERSION, "prev": prev,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"})
            rec["digest"] = digest_of(rec)
            day = time.strftime("%Y-%m-%d", time.gmtime())
            with open(pj(rdir, day + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            try:
                with open(head_path, "w", encoding="utf-8") as f:
                    f.write(rec["digest"])
            except OSError:
                pass                              # the append is authoritative; a cache failure cannot fork it
            if rec.get("decision") != "STOP":                # a stopped call did not happen and leaves no mark
                guarded(chain_commit, None, rec["digest"])
            else:
                CHAIN_PENDING["marks"] = []
                if rec.get("session"):
                    BRAKE_TRIPPED["sid"] = rec["session"] if guarded(brake_note_stop, False, rec["session"]) else ""
            return rec["digest"]
        finally:
            unlock_file(lock)
            lock.close()
    except Exception:
        return None


def consume_permit(action_digest):
    p = pj(OP_HOME, "permits", action_digest + ".json")
    try:
        with open(p, encoding="utf-8") as f:
            pm = json.load(f)
        if time.time() > float(pm.get("expires", 0)):
            os.rename(p, p + ".expired")
            return None
        os.rename(p, p + ".used")           # single use; rename is atomic
        return pm
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# hook entrypoint
# --------------------------------------------------------------------------- #
def install_mode_line(mode):
    """What install says about the mode it leaves the gate in."""
    me = self_cmd()
    if mode == "enforce":
        return ("mode: enforce. The gate stops what its rules block, starting with your agent's next tool call. "
                "To record without blocking: %s mode shadow" % me)
    if mode == "shadow":
        return ("mode: shadow. Nothing is blocked: the gate only records. To have it block: %s mode enforce" % me)
    return "mode: %s. The gate checks nothing. To have it block: %s mode enforce" % (mode, me)


def stop_lines(top, transforms, adigest, receipt):
    """The three lines every agent reads right after a stop: what was stopped
    (rule and receipt), what to do next, and who can allow it once."""
    def sentence(text):
        text = (text or "").strip()
        return text if text[-1:] in ".!?" else text + "."
    seen = " (seen through: %s)" % ", ".join(transforms) if transforms else ""
    return ["OPERATOR: STOPPED (%s %s): %s%s. Receipt: %s"
            % (top[0], top[1], redact(top[3] or "").strip().rstrip("."), seen, (receipt or "unavailable")[:16]),
            "Next: do not retry it or try a workaround.%s Call precedent(rule=\"%s\") to see why and whether a permit exists."
            % ((" " + sentence(top[4])) if top[4] else "", top[0]),
            "If it is really needed, ask the owner for ALLOW ONCE. Only they can grant it, from their own terminal: %s approve %s"
            % (self_cmd(), adigest)]


def stop(reason_lines):
    sys.stderr.write("\n".join(reason_lines) + "\n")
    sys.stderr.flush()
    sys.exit(2)


def end_turn(reason):
    """Ask the host to end the agent's turn as well as block this call (Claude Code's `continue: false`)."""
    sys.stdout.write(json.dumps({"continue": False, "stopReason": reason}) + "\n")
    sys.stdout.flush()


def degraded_hit(raw):
    """Gate crashed: fail closed only for text that plainly looks destructive, else fail open."""
    return bool(re.search(r"\brm\s+-[a-zA-Z]*[rR]|\bmv\s+\S*(vestige|Developer|\.zcode|\.claude)|"
                          r"push\s+.*--force|\bDROP\s+TABLE|\.vestige/vestige\.db|fly\s+deploy|OPERATOR_CANARY_" r"STOP_7f3a",
                          raw or "", re.I))




def read_stdin():
    """The hook payload is UTF-8. Decode it from bytes here: the platform default on Windows is the
    ANSI code page, which cannot represent some UTF-8 input and would make the hook fail open."""
    try:
        return sys.stdin.buffer.read().decode("utf-8", errors="replace")
    except AttributeError:                           # stdin replaced by a text object
        return sys.stdin.read()


def utf8_streams():
    """Write UTF-8 whatever the console code page is, and never raise on a character it lacks: a
    stop message that cannot be printed must not turn into an allow."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def hook_main(argv):
    source = "unknown"
    if "--source" in argv:
        try:
            source = argv[argv.index("--source") + 1]
        except IndexError:
            pass
    raw = ""
    try:
        raw = read_stdin()
        if os.path.exists(pj(OP_HOME, "DISABLED")):
            return 0
        settle_mode("hook")                         # once: a shadow an older installer left becomes enforce
        if global_mode() == "off":
            return 0
        payload = json.loads(raw) if raw.strip() else {}
        cfg = load_config()
        del PROTECTED_EVIDENCE[:]
        tool, cwd, hits, preview, meta, fx_effects = classify(payload, cfg)
        # Session rules: a chain of ordinary steps, a second spelling of a stopped action, a key found in a
        # file, a file the owner gave a contract, a runaway session and a compacted session's watch.
        meta["source"] = source
        meta["spelling"] = guarded(spelling_of, "", payload, cwd)
        hits += guarded(chain_reach, [], payload, cwd, fx_effects)
        hits += guarded(memory_reach, [], payload, cwd, fx_effects, meta)
        hits += guarded(protected_hits, [], payload, cwd, fx_effects, cfg)
        hits += guarded(brake_hit, [], payload, tool, fx_effects)
        hits += guarded(turnover_hit, [], payload, tool, fx_effects)
        BRAKE_TRIPPED["sid"] = ""
        CHAIN_PENDING["sid"] = session_of(payload)
        CHAIN_PENDING["marks"] = guarded(chain_marks, [], payload, cwd, fx_effects)
        hits = [h for h in hits if h[0] not in cfg["disabled_rules"]]

        gmode = global_mode()
        enforce_hits, shadow_hits = [], []
        for rid, detail, hint in hits:
            name, base, why = RULES[rid]
            mode = cfg["mode_overrides"].get(rid, "shadow" if base == "SHADOW" else "enforce")
            if gmode == "shadow":
                mode = "shadow"
            if rid in FLOOR_RULES:                           # the floor holds in every mode but off
                mode = "enforce"
            (enforce_hits if mode == "enforce" else shadow_hits).append((rid, name, why, detail, hint))

        action = {"tool": tool, "source": source, "hits": sorted(set(h[0] for h in hits)),
                  "preview": redact(preview)}
        adigest = digest_of({"tool": tool, "cwd": cwd, "hits": action["hits"], "preview": preview})[:24]
        base_rec = {"source": source, "session": str(payload.get("session_id") or payload.get("sessionId") or "")[:64],
                    "call": call_of(payload),
                    "cwd": cwd, "tool": tool, "action_digest": adigest, "action_preview": action["preview"],
                    "normalized": redact(meta.get("normalized", "")),
                    "transforms": meta.get("transforms", [])}
        try:
            own = sorted(own_law_classes(fx_effects))
        except Exception:
            own = []
        if own:                     # what no built-in rule decides; `status` counts these for the owner
            base_rec["own"] = own

        # specificity ordering: rule class first (canary > self > memory > workspaces), then the
        # most specific target (deepest path) -- the hit reported to the agent is the one to act on.
        priority = ["OP-CANARY", "OP-000", "OP-022", "OP-002", "OP-001", "OP-012", "OP-013", "OP-014", "OP-015",
                    "OP-016", "OP-017", "OP-018", "OP-019", "OP-010", "OP-009", "OP-008", "OP-004",
                    "OP-003", "OP-011", "OP-005", "OP-006", "OP-023", "OP-007", "OP-021", "OP-020", "OP-024",
                    "OP-025", "OP-S06", "OP-S17", "OP-S12", "OP-S13", "OP-S07", "OP-S08", "OP-S11", "OP-S01",
                    "OP-S02", "OP-S03", "OP-S04", "OP-S05", "OP-S14", "OP-S15", "OP-S16"]

        def hit_depth(detail):
            best = 0
            for m in re.finditer(r"/[^\s\"']+", detail or ""):
                best = max(best, m.group(0).count("/"))
            return best

        enforce_hits.sort(key=lambda h: (priority.index(h[0]) if h[0] in priority else len(priority),
                                         -hit_depth(h[3]), h[0]))
        soft = {h[0] for h in enforce_hits} <= {"OP-003", "OP-005", "OP-006", "OP-007"}
        if enforce_hits:
            floor = any(h[0] in FLOOR_RULES for h in enforce_hits)
            permit = None if floor else consume_permit(adigest)
            if permit:
                write_receipt(dict(base_rec, decision="ALLOW", via="permit", commitments=[h[0] for h in enforce_hits]))
                return 0
            top = enforce_hits[0]
            rd = write_receipt(dict(base_rec, decision="STOP", commitment=top[0], rule=top[1],
                                    commitments=[h[0] for h in enforce_hits], detail=redact(top[3])))
            meta["stop_rules"] = [h[0] for h in enforce_hits]
            if top[0] not in HOLD_RULES:                  # a hold on the session says nothing about the target
                guarded(chain_note_stop, None, base_rec["session"], fx_effects, meta, top[0], rd)
            if top[0] == "OP-020" and source == "claude":
                # a second try at an action already stopped this session ends the turn (#530): the block
                # still holds through exit 2, and Claude Code reads this JSON with it
                end_turn("Operator Lite ended this turn. The agent tried again, another way, an action the gate "
                         "already stopped in this session: %s. Only the owner can allow it once, from their own "
                         "terminal: operator-gate approve %s" % (redact(top[3]), adigest))
            stop(stop_lines(top, meta.get("transforms"), adigest, rd))
        if shadow_hits:
            write_receipt(dict(base_rec, decision="SHADOW_STOP", commitments=[h[0] for h in shadow_hits],
                               detail=redact("; ".join(h[3] for h in shadow_hits))))
        else:
            write_receipt(dict(base_rec, decision="PASS", commitments=[],    # no rule matched
                               **({"protected": list(PROTECTED_EVIDENCE)} if PROTECTED_EVIDENCE else {})))

        return 0
    except SystemExit:
        raise
    except Exception as exc:  # gate defect: degrade, never brick
        write_receipt({"source": source, "decision": "GATE_ERROR", "error": redact(repr(exc)),
                       "action_preview": redact(raw[:300])})
        if degraded_hit(raw):
            stop(["OPERATOR: STOPPED (degraded) -- the gate hit an internal error and this call looks destructive.",
                  "Ask the owner to review; the error is recorded in ~/.operator/receipts."])
        return 0


# --------------------------------------------------------------------------- #
# owner CLI
# --------------------------------------------------------------------------- #
def cmd_approve(argv):
    if len(argv) < 1 or not re.match(r"^[0-9a-f]{24}$", argv[0]):
        print("usage: %s approve <24-hex action digest>" % self_cmd())
        return 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: approvals need an interactive terminal run by the owner.")
        return 3
    ensure_dirs()
    r = {}                          # the STOP this permit answers, from today's or yesterday's receipts (UTC days)
    for day in (time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400)), time.strftime("%Y-%m-%d", time.gmtime())):
        try:
            with open(pj(OP_HOME, "receipts", day + ".jsonl"), encoding="utf-8") as f:
                for line in f:
                    if argv[0] in line:
                        rec = json.loads(line)
                        if rec.get("decision") == "STOP":
                            r = rec
        except Exception:
            pass
    if r:
        print("Blocked action: %s\n  rule: %s\n  preview: %s" % (r.get("tool"), r.get("rule"), r.get("action_preview")))
    else:
        print("No matching STOP receipt since yesterday for %s." % argv[0])
    if input("Type ALLOW ONCE to grant a single-use permit (%d min): " % (PERMIT_TTL_S // 60)).strip() != "ALLOW ONCE":
        print("Cancelled.")
        return 1
    p = pj(OP_HOME, "permits", argv[0] + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"action_digest": argv[0], "granted": time.time(), "expires": time.time() + PERMIT_TTL_S,
                   "single_use": True, "by": "owner-tty"}, f)
    os.chmod(p, 0o600)
    grant = {"decision": "PERMIT_GRANTED", "action_digest": argv[0], "source": "owner-tty"}
    if r.get("commitment"):
        grant["commitment"] = r["commitment"]
    write_receipt(grant)
    print("Permit granted for one use.")
    permit_offer(r.get("commitment"))
    return 0


def cmd_verify(argv):
    rdir = pj(OP_HOME, "receipts")
    prev, n, bad = "0" * 64, 0, []
    files = sorted(f for f in os.listdir(rdir) if f.endswith(".jsonl")) if os.path.isdir(rdir) else []
    for fn in files:
        with open(pj(rdir, fn), encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                rec = json.loads(line)
                d = rec.pop("digest")
                if rec.get("prev") != prev or digest_of(rec) != d:
                    bad.append("%s:%d" % (fn, i))
                prev = d
                n += 1
    print("receipts=%d chain=%s integrity=%s" % (n, "OK" if not bad else "BROKEN at " + ",".join(bad[:5]), INTEGRITY))
    print("note: a digest chain detects accidental edits; it is not a signature and proves no authorship.")
    upgrade_hint()
    return 0 if not bad else 1


def self_cmd():
    """How the owner runs this gate from a terminal: the launcher when install could place one on
    PATH, otherwise the full command, which always works."""
    if shutil.which("operator-gate"):
        return "operator-gate"
    path = pj(HOME, ".operator", "gate", "operator-gate.py")
    if not os.path.exists(path):
        path = canon(os.path.abspath(__file__))
    if IS_WINDOWS:
        return 'python "%s"' % path
    return "python3 %s" % (("~" + path[len(HOME):]) if path.startswith(HOME + "/") else path)


def place_launcher(dst):
    """Put an `operator-gate` command on PATH when a user bin directory is already on it. Shell init
    files are never edited: with no such directory, hints print the full command instead."""
    if IS_WINDOWS:
        return None
    on_path = [p.rstrip("/") for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for d in (pj(HOME, ".local", "bin"), pj(HOME, "bin")):
        if d not in on_path or not os.path.isdir(d) or not os.access(d, os.W_OK):
            continue
        target = pj(d, "operator-gate")
        if os.path.exists(target):
            try:
                with open(target, encoding="utf-8") as f:
                    if ".operator/gate/operator-gate.py" not in f.read():
                        continue                             # someone else's file: leave it alone
            except Exception:
                continue
        with open(target, "w", encoding="utf-8") as f:
            f.write('#!/bin/sh\nexec python3 "%s" "$@"\n' % dst)
        os.chmod(target, 0o755)
        return target
    return None


def hook_wired():
    """Is this gate registered as a Claude Code PreToolUse hook?"""
    try:
        with open(pj(HOME, ".claude", "settings.json"), encoding="utf-8") as f:
            settings = json.load(f)
        return any("operator-gate" in hh.get("command", "")
                   for h in settings.get("hooks", {}).get("PreToolUse", []) for hh in h.get("hooks", []))
    except Exception:
        return False


def receipt_time(rec):
    try:
        return calendar.timegm(time.strptime(str(rec.get("ts"))[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0


def recent_receipts(days):
    """Receipts from the last `days` days, oldest first. Lines that do not parse are skipped."""
    rdir, out, cutoff = pj(OP_HOME, "receipts"), [], time.time() - days * 86400
    try:
        files = sorted(f for f in os.listdir(rdir) if f.endswith(".jsonl"))
    except OSError:
        return out
    for fn in files[-(days + 2):]:
        try:
            with open(pj(rdir, fn), encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict) and receipt_time(rec) >= cutoff:
                        out.append(rec)
        except OSError:
            pass
    return out


def ago(then):
    s_ = max(0, int(time.time() - then))
    if s_ < 90:
        return "just now"
    if s_ < 5400:
        return "%d minutes ago" % (s_ // 60)
    if s_ < 129600:
        return "%d hours ago" % (s_ // 3600)
    return "%d days ago" % (s_ // 86400)


def cmd_status(argv):
    """What the gate did on this machine. `--rules` prints the rule table instead."""
    mode = "off" if os.path.exists(pj(OP_HOME, "DISABLED")) else global_mode()
    if "--rules" in argv:
        print("operator-gate %s  home=%s  mode=%s" % (VERSION, OP_HOME, mode))
        for rid, (name, base, why) in sorted(RULES.items()):
            print("  %-9s %-7s %-28s %s" % (rid, base, name, why))
        return 0
    me, wired = self_cmd(), hook_wired()
    says = {"enforce": "enforce (blocking)", "shadow": "shadow (recording, not blocking)",
            "off": "off (nothing is checked)"}[mode]
    print("Operator Lite %s   mode: %s   Claude Code hook: %s" % (VERSION, says, "registered" if wired else "NOT registered"))
    try:
        with open(pj(OP_HOME, "state", MODE_SETTLED), encoding="utf-8") as f:
            settled = json.load(f)
    except Exception:
        settled = {}
    if isinstance(settled, dict) and settled.get("switched") and mode == "enforce":
        print("Switched from shadow to enforce on %s: Operator Lite before 0.3.9 installed in shadow mode, which "
              "blocks nothing. To record without blocking: %s mode shadow" % (str(settled.get("ts"))[:10], me))
    if not wired:
        print("Register it: %s install" % me)
    recs = recent_receipts(7)
    calls = [r for r in recs if r.get("decision") in ("PASS", "ALLOW", "STOP", "SHADOW_STOP", "GATE_ERROR")]
    if not calls:
        print("\nNo tool calls checked in the last 7 days. Start a Claude Code session: every tool call passes\n"
              "through the gate, and this view fills in.")
        print("\nYour last 30 days, replayed: %s replay\nAll rules: %s status --rules" % (me, me))
        upgrade_hint()
        return 0

    def hard(rec):
        return any(RULES.get(c, ("", "", ""))[1] == "STOP" for c in rec.get("commitments") or [])

    stopped = [r for r in calls if r.get("decision") == "STOP"]
    ran = [r for r in calls if r.get("decision") == "SHADOW_STOP" and hard(r)]
    flagged = [r for r in calls if r.get("decision") == "SHADOW_STOP" and (r.get("commitments") or []) and not hard(r)]
    permits = [r for r in calls if r.get("via") == "permit"]
    errors = [r for r in calls if r.get("decision") == "GATE_ERROR"]
    print("Last call checked %s." % ago(receipt_time(calls[-1])))
    print("\nLast 7 days on this machine:")
    print("  %s  tool calls checked" % paint("%7s" % format(len(calls), ","), "1"))
    if stopped or mode == "enforce":
        print("  %s  stopped" % paint("%7s" % format(len(stopped), ","), "1;32"))
    if ran:
        print("  %s  would have been stopped, and ran: the gate was in shadow mode" % paint("%7s" % format(len(ran), ","), "1;31"))
    print("  %s  flagged: recorded, not stopped" % paint("%7s" % format(len(flagged), ","), "1"))
    if permits:
        print("  %s  allowed once on your permit" % paint("%7s" % format(len(permits), ","), "1"))
    if errors:
        print("  %s  gate errors, recorded" % paint("%7s" % format(len(errors), ","), "1"))

    def show(title, rows):
        if not rows:
            return
        print("\n" + paint(title, "1"))
        for r in sorted(rows, key=receipt_time, reverse=True)[:5]:
            rid = r.get("commitment") or next((c for c in r.get("commitments") or []
                                               if RULES.get(c, ("", "", ""))[1] == "STOP"), (r.get("commitments") or ["?"])[0])
            when = time.strftime("%b %d %H:%M", time.localtime(receipt_time(r)))
            project = os.path.basename(str(r.get("cwd") or "").rstrip("/"))[:16]
            why = " ".join(str(r.get("detail") or "").split(";")[0].replace(HOME, "~").split())[:64]
            ran_ = " ".join(str(r.get("action_preview") or "").replace(HOME, "~").split())[:72]
            print("  %s  %-16s %s %s" % (when, project, paint(rid, "31"), why))
            print("  %s  %-16s %s" % (" " * len(when), "", paint(ran_, "2")))
        if len(rows) > 5:
            print("  and %d more in ~/.operator/receipts" % (len(rows) - 5))

    show("Stopped:", stopped)
    show("Ran, because the gate was in shadow mode:", ran)
    if mode == "shadow":
        print("\nShadow mode records and never blocks. To have the gate block from now on: %s mode enforce" % me)
    print("\nYour last 30 days, replayed: %s replay\nAll rules: %s status --rules" % (me, me))
    undecided = {}
    for r in calls:                 # calls no built-in rule decided: they ran with nothing to say about them
        if r.get("decision") == "PASS" or (r.get("decision") == "SHADOW_STOP" and not hard(r)):
            for c in r.get("own") or []:
                undecided[c] = undecided.get(c, 0) + 1
    if not weekly_offer(undecided):
        upgrade_hint()
    return 0


def cmd_mode(argv):
    """Owner command: switch between recording and blocking. Works the same on every shell."""
    if not argv:
        print("mode is %s. To change it: %s mode enforce|shadow|off" % (global_mode(), self_cmd()))
        return 0
    if argv[0] == "--settle":
        # The port installers' call: settle the mode once (settle_mode), then print it. It can only
        # turn an older installer's shadow into enforce, so it needs no terminal.
        mode, switched = settle_mode("installer")
        print("mode=%s%s" % (mode or "enforce", " (%s)" % SETTLED_NOTE if switched else ""))
        return 0
    if argv[0] not in ("enforce", "shadow", "off"):
        print("usage: %s mode enforce|shadow|off" % self_cmd())
        return 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: the mode is changed by the owner in an interactive terminal.")
        return 3
    ensure_dirs()
    with open(pj(OP_HOME, "mode"), "w", encoding="utf-8") as f:
        f.write(argv[0] + "\n")
    write_receipt({"decision": "MODE_SET", "mode": argv[0], "source": "owner-tty"})
    record_settled("owner", argv[0], False)
    print({"enforce": "The gate now blocks what its rules stop.",
           "shadow": "The gate now records and blocks nothing.",
           "off": "The gate is off: nothing is checked or recorded."}[argv[0]])
    return 0


# The offer, in one place: where the paid gate is sold and what it costs.
OPERATOR_URL = "https://payhip.com/b/d4xvu"
OPERATOR_PRICE = "$149 once, yours to keep, 30-day refund"


def at_terminal():
    """A person is reading: stdout is a terminal and this is not an agent's session."""
    return sys.stdout.isatty() and not os.environ.get("OPERATOR_AGENT_SESSION")


def last_replay():
    """The summary the last `replay` left in the gate home, or {}."""
    try:
        with open(pj(OP_HOME, "state", "replay.json"), encoding="utf-8") as f:
            last = json.load(f)
        return last if isinstance(last, dict) else {}
    except Exception:
        return {}


def upgrade_hint():
    """One line for the person at the terminal. Never printed to an agent, a pipe or a script,
    and never part of a verdict: stop messages go to the model, and a pitch does not belong there."""
    if not at_terminal():
        return
    last = last_replay()
    undecided, laws = int(last.get("own_calls") or 0), len(last.get("laws") or [])
    if undecided and laws:
        print("\nYour last replay drafted %s from %s no built-in rule decides. "
              "Operator enforces them: %s upgrade" % (plural(laws, "law"), plural(undecided, "action"), self_cmd()))
    else:
        print("\nSee what your agents already ran: %s replay\n"
              "Your own laws, a Board and a Letter: %s upgrade" % (self_cmd(), self_cmd()))


# How the week's undecided actions are counted out loud, one noun per OWN_LAW class.
OWN_NOUNS = {"push": ("push", "pushes"), "no-verify": ("commit that skipped git hooks", "commits that skipped git hooks"),
             "packages": ("package install", "package installs"), "deploy": ("deploy", "deploys"),
             "database": ("database run", "database runs"), "ci-config": ("CI config write", "CI config writes"),
             "env-file": ("env file write", "env file writes")}
# A law an owner might write in Operator after allowing the same rule again and again.
PERMIT_LAWS = {"OP-003": "Deleting build folders inside my projects is fine.",
               "OP-004": "Force pushes to my own feature branches are fine; never to main.",
               "OP-005": "Publishing a release from main is fine after the tests pass.",
               "OP-006": "Preview deploys are fine; production is mine.",
               "OP-007": "Dropping tables in my local test databases is fine."}


def weekly_offer(undecided):
    """At a terminal: the last 7 days' actions no built-in rule decided, and a law the owner could write
    for the commonest. True when printed. Never shown to an agent, a pipe or a script."""
    if not undecided or not at_terminal():
        return False
    order = sorted(undecided, key=lambda c: (-undecided[c], [k for k, _, _ in OWN_LAW].index(c)))
    said = ", ".join("%d %s" % (undecided[c], OWN_NOUNS[c][0 if undecided[c] == 1 else 1]) for c in order)
    law = next(w for k, _, w in OWN_LAW if k == order[0])
    print("\nLast 7 days, no built-in rule decided: %s." % said)
    print("Operator makes these your laws, in your words, such as \"%s\" (%s): %s upgrade"
          % (law, OPERATOR_PRICE, self_cmd()))
    return True


def ordinal(n):
    return {3: "third", 6: "sixth", 9: "ninth", 12: "twelfth"}.get(n, "%dth" % n)


def permit_offer(rid):
    """At a terminal, on every third permit for the same rule in 30 days: the owner keeps saying yes to
    the same thing, which is a law waiting to be written. Never shown to an agent or a pipe."""
    if not rid or not at_terminal():
        return
    n = sum(1 for r in recent_receipts(30) if r.get("decision") == "PERMIT_GRANTED" and r.get("commitment") == rid)
    if n < 3 or n % 3:
        return
    name = RULES.get(rid, ("", "", ""))[0]
    law = PERMIT_LAWS.get(rid, "Say once when %s is fine and when it is not." % (name or rid))
    print("\nThat is the %s time in 30 days you allowed %s %s. In Operator you write that rule once, in your "
          "words, such as \"%s\", instead of approving it each time (%s): %s upgrade"
          % (ordinal(n), rid, name, law, OPERATOR_PRICE, self_cmd()))


def upgrade_install(argv):
    """After buying: unpack the downloaded Operator archive into ~/vestige-operator and start its
    wizard. One command on every platform. An owner command: it needs an interactive terminal."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: the paid gate is installed by the owner in an interactive terminal.")
        return 3
    try:
        path = os.path.expanduser(argv[argv.index("--install") + 1])
    except IndexError:
        print("usage: %s upgrade --install <the vestige-operator archive you downloaded>" % self_cmd())
        return 2
    if not os.path.isfile(path):
        print("No such file: %s" % path)
        return 2
    import tarfile
    with open(path, "rb") as f:
        print("Archive: %s\nSHA-256: %s" % (path, hashlib.sha256(f.read()).hexdigest()))
    try:
        with tarfile.open(path) as tar:
            for m in tar.getmembers():
                name = m.name.replace("\\", "/")
                inside = name == "vestige-operator" or name.startswith("vestige-operator/")
                if not inside or ".." in name.split("/") or m.issym() or m.islnk():
                    print("Refusing: this archive has an entry that does not belong in vestige-operator/ (%s)." % m.name)
                    return 2
            try:
                tar.extractall(HOME, filter="data")
            except TypeError:                            # Python before the extraction filter existed
                tar.extractall(HOME)
    except (tarfile.TarError, OSError) as exc:
        print("That file could not be unpacked (%s)." % exc)
        return 2
    gate = pj(HOME, "vestige-operator", "gate", "operator-gate.py")
    if not os.path.isfile(gate):
        print("That archive holds no gate.")
        return 2
    print("Unpacked to %s. Starting the wizard.\n" % pj(HOME, "vestige-operator"))
    return subprocess.call([sys.executable, gate, "onboard"])


def cmd_upgrade(argv):
    """What the paid gate adds and where to get it. `--open` opens the page in a browser.
    `--install <archive>` unpacks the Operator archive you bought and starts its wizard."""
    if "--install" in argv:
        return upgrade_install(argv)
    last = last_replay()
    if last.get("laws"):
        print("From your last replay (%s, %s), the laws your own history drafted:" % (
            str(last.get("ts") or "")[:10], "%d days" % last["days"] if last.get("days") else "all history"))
        for law in last["laws"]:
            print("  %-54s %s" % ('"%s"' % law.get("law"), plural(int(law.get("times") or 0), "time")))
        print()
    print("""Vestige Operator: the owner's version of this gate, %s.

  Your own laws   Sentences you write become rules the gate enforces on every host, with a
                  compliant rewrite or a stop, and a one-time permit only you can grant.
  The Board       Today's stops and law violations as cards, built from your receipts.
  The Letter      One weekly digest of what your agents tried and what stopped them.
  Onboarding      A five-minute wizard that writes your first laws and proves one stop.

Operator Lite stays free. Operator blocks what is routed through it, and its receipts are
hash-chained digests, not signatures.

Buy:  %s
Pay once. Every later version is yours at no charge. Download the archive, and one command
installs it and starts the wizard:
  %s upgrade --install <the archive you downloaded>""" % (OPERATOR_PRICE, OPERATOR_URL, self_cmd()))
    if "--open" in argv and sys.stdout.isatty() and not os.environ.get("OPERATOR_AGENT_SESSION"):
        try:
            import webbrowser
            webbrowser.open(OPERATOR_URL)
        except Exception:
            pass
    return 0


# --------------------------------------------------------------------------- #
# replay: this machine's agent history through the gate (classification only)
# --------------------------------------------------------------------------- #
REPLAY_WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
PKG_ADD = {"npm": ("install", "i", "add"), "pnpm": ("add", "install", "i"), "yarn": ("add",),
           "bun": ("add", "install", "i"), "pip": ("install",), "pip3": ("install",), "uv": ("add",),
           "cargo": ("add", "install"), "go": ("get", "install"), "gem": ("install",),
           "brew": ("install",)}
DEPLOY_SUBS = {"fly": ("deploy",), "flyctl": ("deploy",), "vercel": ("deploy",), "netlify": ("deploy",),
               "wrangler": ("deploy", "publish"), "railway": ("up",), "firebase": ("deploy",),
               "serverless": ("deploy",), "sls": ("deploy",), "cdk": ("deploy",), "pulumi": ("up",),
               "terraform": ("apply", "destroy"), "tofu": ("apply", "destroy"),
               "kubectl": ("apply", "delete", "rollout", "scale"),
               "helm": ("install", "upgrade", "uninstall"), "docker": ("push",)}
MIGRATE_SUBS = {"prisma": ("migrate", "db"), "alembic": ("upgrade", "downgrade"),
                "drizzle-kit": ("push", "migrate"), "diesel": ("migration",), "sqlx": ("migrate",),
                "dbmate": ("up", "down", "migrate", "rollback")}
CI_FILE_NAMES = {"dockerfile", ".gitlab-ci.yml", "jenkinsfile", "fly.toml", "vercel.json",
                 "netlify.toml", "wrangler.toml", "docker-compose.yml", "docker-compose.yaml",
                 "compose.yml", "compose.yaml", "procfile"}
ENV_TEMPLATE_ENDS = (".example", ".sample", ".template", ".dist")
# Actions no built-in rule decides, because whether they are fine depends on the owner. Each
# carries the sentence an owner would write for it; Operator, the paid gate, enforces such
# sentences. (class, what happened, the law in the owner's words)
OWN_LAW = (("push", "pushed to a remote", "No push without my permit."),
           ("no-verify", "skipped git hooks", "Never skip git hooks."),
           ("packages", "installed packages by name", "No new package without my review."),
           ("deploy", "deployed or changed infrastructure", "Deploys and infrastructure changes are mine."),
           ("database", "ran a database client or migration", "No database client or migration without my permit."),
           ("ci-config", "wrote CI or deploy config", "CI and deploy config are mine to change."),
           ("env-file", "wrote an env file", "Env files are mine to change."))
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
LITE_URL = "https://github.com/samvallad33/vestige/tree/main/operator-lite"


def own_law_classes(effects):
    """Which OWN_LAW classes a call falls in, each with the command segment that put it there.
    Read from the parsed effects (program, subcommand, flags, write targets), never from the raw text."""
    out = {}
    for e in effects:
        prog, args, flags = e.get("prog") or "", list(e.get("args") or []), e.get("flags") or []
        if prog in ("npx", "bunx", "pnpx") and args:
            prog, args = os.path.basename(args[0]), args[1:]
        sub = args[0] if args else ""
        dry = "--dry-run" in flags
        if e.get("kind") == "git":
            sub = e.get("sub") or ""
            if sub == "push" and not dry and "-n" not in flags:
                out.setdefault("push", e.get("seg") or "")
            if "--no-verify" in flags or (sub == "commit" and has_flag(flags, "n")):
                out.setdefault("no-verify", e.get("seg") or "")
        elif prog in PKG_ADD and sub in PKG_ADD[prog]:
            named = [a for a in args[1:] if a != "."]
            if named and not dry and not any(f in ("-r", "--requirement", "-e", "--editable") for f in flags):
                out.setdefault("packages", e.get("seg") or "")
        elif prog in DEPLOY_SUBS and not dry and \
                (sub in DEPLOY_SUBS[prog] or (prog == "vercel" and "--prod" in flags)):
            out.setdefault("deploy", e.get("seg") or "")
        elif e.get("kind") == "db" or (prog in MIGRATE_SUBS and sub in MIGRATE_SUBS[prog]):
            out.setdefault("database", e.get("seg") or "")
        for p in e.get("writes") or []:
            base = os.path.basename(p).lower()
            if "/.github/workflows/" in p or "/.circleci/" in p or base in CI_FILE_NAMES \
                    or base.startswith("dockerfile."):
                out.setdefault("ci-config", e.get("seg") or "")
            if base == ".env" or (base.startswith(".env.") and not base.endswith(ENV_TEMPLATE_ENDS)):
                out.setdefault("env-file", e.get("seg") or "")
    return out


def replay_history(days, budget, here):
    """Classify every unique tool call in the local Claude Code transcripts, newest session first.
    Reads history and the filesystem; executes nothing and writes no receipt."""
    cfg = load_config()
    base = pj(HOME, ".claude", "projects")
    files = glob.glob(pj(base, "*", "*.jsonl")) + \
        glob.glob(pj(base, "*", "*", "subagents", "*.jsonl"))
    cutoff = time.time() - days * 86400 if days else 0
    cutoff_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(cutoff)) if days else ""
    dated = []
    for path in files:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime >= cutoff:
            dated.append((mtime, path))
    dated.sort(reverse=True)
    out = {"calls": 0, "sessions": 0, "stopped": 0, "flagged": 0, "own_calls": 0, "errors": 0,
           "by_stop": {}, "by_shadow": {}, "by_own": {}, "samples": {}, "incidents": [],
           "projects": 0, "truncated": False, "days": days or 0, "files": len(dated)}

    def bump(table, key, sample):
        out[table][key] = out[table].get(key, 0) + 1
        out["samples"].setdefault(key, sample)

    seen, projects, t0 = set(), set(), time.time()
    base_parts = len(base.rstrip("/").split("/"))
    for _, path in dated:
        if time.time() - t0 > budget:
            out["truncated"] = True
            break
        used = False
        try:
            fh = open(path, errors="replace", encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"tool_use"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                ts = str(rec.get("timestamp") or "")
                if cutoff_iso and ts and ts[:19] < cutoff_iso:
                    continue
                cwd = rec.get("cwd") or ""
                if here and not (cwd and is_inside(cwd, here)):
                    continue
                msg = rec.get("message")
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, list):
                    continue
                for blk in content:
                    if not (isinstance(blk, dict) and blk.get("type") == "tool_use"):
                        continue
                    name, ti = str(blk.get("name") or ""), blk.get("input") or {}
                    if not (name in ("Bash", "PowerShell") or name in REPLAY_WRITE_TOOLS or name.startswith("mcp__")):
                        continue
                    # one call, one count: a resumed session copies earlier records with the same
                    # call id, while the same command run again is a new call with a new id
                    key = str(blk.get("id") or "")
                    if not key:
                        try:
                            key = digest_of([name, cwd, ti])
                        except Exception:
                            continue
                    if key in seen:
                        continue
                    seen.add(key)
                    out["calls"] += 1
                    used = True
                    project = os.path.basename(cwd.rstrip("/")) if cwd else ""
                    projects.add(canon(path).split("/")[base_parts])   # Claude Code keeps one folder per project
                    try:
                        _, _, hits, preview, _, effects = classify(
                            {"tool_name": name, "tool_input": ti, "cwd": cwd or "/tmp"}, cfg)
                    except Exception:
                        out["errors"] += 1
                        continue
                    rids = sorted(set(h[0] for h in hits
                                      if h[0] not in cfg["disabled_rules"] and h[0] != "OP-CANARY"))

                    def show(text, width=60):
                        text = " ".join(redact(text or preview).split())
                        if cwd:
                            text = text.replace(cwd.rstrip("/") + "/", "")
                        return text.replace(HOME, "~")[:width]

                    why = {}
                    for rid, detail, _ in hits:
                        why.setdefault(rid, detail)
                    stops = [r for r in rids if cfg["mode_overrides"].get(
                        r, "shadow" if RULES[r][1] == "SHADOW" else "enforce") == "enforce"]
                    if stops:
                        out["stopped"] += 1
                        for r in stops:
                            bump("by_stop", r, show(why.get(r)))
                        ran = ""                    # the part of the command the rule fired on
                        for e in effects:
                            try:
                                if any(h[0] == stops[0] for h in classify_effect(e, cfg, cwd or "/tmp")):
                                    ran = e.get("seg") or ""
                                    break
                            except Exception:
                                break
                        out["incidents"].append({"ts": ts[:19], "project": project, "rule": stops[0],
                                                 "why": show(why.get(stops[0]), 72), "cmd": show(ran or preview, 72)})
                    elif rids:
                        out["flagged"] += 1
                        for r in rids:
                            bump("by_shadow", r, show(why.get(r)))
                    else:
                        classes = own_law_classes(effects)
                        if classes:
                            out["own_calls"] += 1
                        for c, seg in classes.items():
                            bump("by_own", c, show(seg))
        if used:
            out["sessions"] += 1
    out["projects"] = len(projects)
    out["incidents"] = sorted(out["incidents"], key=lambda i: i["ts"], reverse=True)[:20]
    out["laws"] = [{"class": c, "law": law, "times": out["by_own"][c]}
                   for c, _, law in sorted(OWN_LAW, key=lambda row: -out["by_own"].get(row[0], 0))
                   if out["by_own"].get(c)]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def paint(text, code):
    """Colour for a person at a terminal; plain text for everything else. NO_COLOR is honoured."""
    if not at_terminal() or os.environ.get("NO_COLOR"):
        return text
    if IS_WINDOWS and not (os.environ.get("WT_SESSION") or os.environ.get("TERM") or os.environ.get("ANSICON")):
        return text                                  # the old Windows console prints the codes as text
    return "\033[%sm%s\033[0m" % (code, text)


def plural(n, word):
    return "%s %s%s" % (format(n, ","), word, "" if n == 1 else "s")


def cmd_replay(argv):
    """Run this machine's agent history through classify(). Classification only; nothing executes."""
    days, budget = 30, 30.0
    try:
        if "--days" in argv:
            days = max(1, int(argv[argv.index("--days") + 1]))
        if "--budget" in argv:
            budget = max(1.0, float(argv[argv.index("--budget") + 1]))
    except (IndexError, ValueError):
        print("usage: operator-gate replay [--days N | --all] [--here] [--budget SECONDS] [--json | --share]")
        return 2
    if "--all" in argv:
        days = 0
    here = os.getcwd() if "--here" in argv else None
    r = replay_history(days, budget, here)
    if "--json" in argv:
        print(json.dumps(r, sort_keys=True))
        return 0
    span = "the last %d days" % days if days else "all history"
    if "--share" in argv:                           # counts only: nothing from the history itself
        print("In %s my coding agents made %s." % (span, plural(r["calls"], "tool call")))
        print("Operator Lite would have stopped %d, flagged %d, and found %d that only I can rule on."
              % (r["stopped"], r["flagged"], r["own_calls"]))
        print("Free, one file, runs on your own history: %s" % LITE_URL)
        return 0
    if not r["files"]:
        print("operator-gate replay: no Claude Code history in ~/.claude/projects for that period.")
        return 0
    print("operator-gate replay: %s on this machine%s. Nothing was executed."
          % (span, ", under %s" % here if here else ""))
    if r["truncated"]:
        print("(stopped at the %d second budget, newest sessions first; --budget N reads more)" % budget)
    print()
    print("  %7s  tool calls your agents made (%s, %s)" % (
        paint("%7s" % format(r["calls"], ","), "1"), plural(r["sessions"], "Claude Code session"),
        plural(r["projects"], "project")))
    print("  %7s  a built-in rule would have stopped" % paint("%7s" % format(r["stopped"], ","), "1;31"))
    print("  %7s  flagged in shadow: recorded, not stopped" % paint("%7s" % format(r["flagged"], ","), "1"))
    print("  %7s  no built-in rule decides: only you can" % paint("%7s" % format(r["own_calls"], ","), "1;33"))

    def table(counts, label):
        for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print("  %5d  %-34s %s" % (n, label(key), paint(r["samples"].get(key, ""), "2")))

    if r["stopped"]:
        shown = r["incidents"][:5]
        print("\n" + paint("Would have been stopped", "1") + " (%s):" % (
            "the %d most recent of %d" % (len(shown), r["stopped"]) if r["stopped"] > len(shown) else "all of them"))
        for inc in shown:
            when = "      "
            if len(inc["ts"]) >= 10 and inc["ts"][5:7].isdigit() and 1 <= int(inc["ts"][5:7]) <= 12:
                when = "%s %s" % (MONTHS[int(inc["ts"][5:7]) - 1], inc["ts"][8:10])
            print("  %s  %-18s %s %s" % (when, inc["project"][:18], paint(inc["rule"], "31"), inc["why"]))
            print("  %s  %-18s %s" % ("      ", "", paint(inc["cmd"], "2")))
        print("By rule:")
        table(r["by_stop"], lambda k: "%s %s" % (k, RULES[k][0]))
    if r["flagged"]:
        print("\n" + paint("Flagged in shadow", "1") + ", recorded and not stopped:")
        table(r["by_shadow"], lambda k: "%s %s" % (k, RULES[k][0]))
    if r["own_calls"]:
        names = dict((c, what) for c, what, _ in OWN_LAW)
        print("\n" + paint("No built-in rule decides these. They ran:", "1"))
        table(r["by_own"], lambda k: names[k])
        print("\n" + paint("Your first laws, drafted from this history:", "1"))
        for law in r["laws"]:
            print("  %-54s %s" % ('"%s"' % law["law"], plural(law["times"], "time")))
    if not here:
        try:                                       # remembered for the status hint and for `upgrade`
            if os.path.isdir(OP_HOME):
                ensure_dirs()
                keep = {k: r[k] for k in ("calls", "sessions", "projects", "stopped", "flagged", "own_calls",
                                          "days", "by_own", "laws")}
                keep["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                path = pj(OP_HOME, "state", "replay.json")
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(keep, f, sort_keys=True)
                os.replace(path + ".tmp", path)
        except Exception:
            pass
    if at_terminal():
        if r["own_calls"]:
            print("\nOperator enforces those laws from your agent's next command, on every host you hook: it stops\n"
                  "the action and waits for a permit only you can grant. %s.\n"
                  "  %s\n  details: %s upgrade" % (OPERATOR_PRICE, OPERATOR_URL, self_cmd()))
        if "--from-install" not in argv and \
                not os.path.exists(pj(HOME, ".operator", "gate", "operator-gate.py")):
            print("\nTo have the built-in rules watch from now on: %s install" % self_cmd())
    return 0


def cmd_corpus(argv):
    """Run an eval corpus through classify(). Pure classification; nothing executes."""
    name = argv[0] if argv else "guardfall"
    path = pj(os.path.dirname(os.path.abspath(__file__)), "corpora", name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print("corpus %s is not beside this file: the corpus ships in the repository, not with the single\n"
              "gate file. From a clone of github.com/samvallad33/vestige: python3 operator-lite/operator-gate.py corpus %s"
              % (name, name))
        return 2
    except Exception as exc:
        print("cannot load corpus %s: %s" % (name, exc))
        return 2
    os.environ["OPERATOR_HOME"] = os.environ.get("OPERATOR_HOME", tempfile.mkdtemp(prefix="opgate-corpus-"))
    cfg = load_config()
    passed, failed = 0, []
    for case in corpus["cases"]:
        cid, cmd, expect = case["id"], case["cmd"], case["expect"]
        cwd = "/tmp"
        tmpd = None
        if case.get("fixtures"):                       # materialize script fixtures (never executed)
            tmpd = tempfile.mkdtemp(prefix="opgate-fixture-")
            for fx in case["fixtures"]:
                fp = pj(tmpd, fx["name"])
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(fx["body"])
            cwd = tmpd
        _, _, hits, _, _, _ = classify({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": cwd}, cfg)
        if tmpd:
            try:
                import shutil as _sh
                _sh.rmtree(tmpd, True)
            except Exception:
                pass
        rids = sorted(set(h[0] for h in hits))
        if expect == "ALLOW":
            ok = not [r for r in rids if RULES.get(r, ("", "enforce", ""))[1] == "STOP"]
        elif expect == "SHADOW":
            ok = any(r.startswith("OP-S") for r in rids) or bool(rids)
        else:
            ok = expect in rids
        if ok:
            passed += 1
        else:
            failed.append("%s expected %s got %s" % (cid, expect, ",".join(rids) or "ALLOW"))
    print("corpus=%s cases=%d passed=%d failed=%d" % (name, len(corpus["cases"]), passed, len(failed)))
    for line in failed:
        print("  FAIL " + line)
    return 0 if not failed else 1


def hook_command(dst, source):
    """The command a host runs for every tool call. Windows has no python3 on PATH by default, so
    there the command names this interpreter; forward slashes work in cmd and in Git Bash alike."""
    if IS_WINDOWS:
        return '"%s" "%s" hook --source %s' % (canon(sys.executable), dst, source)
    return "python3 %s hook --source %s" % (dst, source)


def claude_hook_command(dst):
    """The command registered in Claude Code, which runs it through a shell. python3 exits 2 when it
    cannot open the gate file, and Claude Code reads exit 2 as a block, so a gate that went missing would
    block every tool call. The test exits 1 instead (a non-blocking error) and the path is quoted. Windows
    hosts run the command in cmd, PowerShell or Git Bash, so there the check is made by Python itself."""
    if IS_WINDOWS:
        return guarded_python_command(canon(sys.executable), dst)
    q = shlex.quote(dst)
    return "[ -f %s ] && python3 %s hook --source claude" % (q, q)


def guarded_python_command(python, dst):
    """One quoted interpreter and one -c program, the same in every Windows shell: run the gate when the
    file exists (its exit code, 2 on STOP, passes through), exit 0 when it is gone. -I keeps the working
    directory off sys.path, so a project file cannot stand in for a stdlib module."""
    code = ("import os,runpy,sys;g=%r;sys.argv=[g,'hook','--source','claude'];"
            "os.path.isfile(g) and runpy.run_path(g,run_name='__main__')" % dst)
    return '"%s" -I -c "%s"' % (python, code)


def claude_side_command(dst, args):
    """A Claude Code hook that runs one more gate command, `turnover --pre-compact` or `turnover
    --session-start`, guarded the same way as the PreToolUse hook."""
    if IS_WINDOWS:
        return guarded_python_command(canon(sys.executable), dst).replace(
            "'hook','--source','claude'", ",".join(repr(a) for a in args))
    q = shlex.quote(dst)
    return "[ -f %s ] && python3 %s %s" % (q, q, " ".join(args))


TURNOVER_HOOKS = (("PreCompact", "manual|auto", ["turnover", "--pre-compact"]),
                  ("SessionStart", "compact", ["turnover", "--session-start"]))


def is_claude_gate_hook(command):
    """A registration of this gate in Claude Code, in this version's form or an older one."""
    return "operator-gate.py" in command and ("hook --source claude" in command or "'--source','claude'" in command)


# --------------------------------------------------------------------------- #
# install guard: never replace a gate that outranks this one
# --------------------------------------------------------------------------- #
# Every paid gate since 0.3.1 defines load_law and cmd_commitments, and no Lite ever has; its $149-tier
# modules sit beside it. A gate that states EDITION is read by that first.
PAID_MARK_RE = re.compile(r"^def (?:load_law|cmd_commitments|_feature)\(", re.M)
PAID_SIBLINGS = ("board.py", "letter.py", "sealed.py", "almost_forgot.py")
# Claude Code sets CLAUDECODE=1 and CLAUDE_CODE_ENTRYPOINT in every shell its Bash tool starts, and does
# not set OPERATOR_AGENT_SESSION (checked 2026-10-15). OPERATOR_AGENT_SESSION stays the marker any
# other host can set.
AGENT_ENV = ("OPERATOR_AGENT_SESSION", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


def agent_session():
    """The variable that marks this shell as an AI agent's, or None."""
    for name in AGENT_ENV:
        if os.environ.get(name):
            return name
    return None


def version_tuple(text):
    try:
        return tuple(int(x) for x in str(text).split("."))
    except ValueError:
        return None


def gate_identity(path):
    """(edition, version) of a gate file, read as text and never run. edition is "operator" (the paid
    gate), "lite", or "unknown" (a gate from before either marker); version is "" when unreadable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return "unknown", ""
    m = re.search(r"^VERSION\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    version = m.group(1) if m and version_tuple(m.group(1)) else ""
    m = re.search(r"^EDITION\s*=\s*[\"'](\w+)[\"']", text, re.M)
    if m:
        return m.group(1).lower(), version
    here = os.path.dirname(os.path.abspath(path))
    if PAID_MARK_RE.search(text) or any(os.path.isfile(pj(here, n)) for n in PAID_SIBLINGS):
        return "operator", version
    return ("lite" if "Operator Lite" in text else "unknown"), version


def gate_label(edition, version):
    name = {"operator": "Operator", "lite": "Operator Lite"}.get(edition, "a gate of unknown edition")
    return "%s %s" % (name, version or "(version unreadable)")


def install_refusal(dst, src):
    """Why copying this gate over the one at dst would be a downgrade, or None when it would not."""
    if not os.path.lexists(dst) or os.path.realpath(dst) == os.path.realpath(src):
        return None
    edition, version = gate_identity(dst)
    if edition == "operator":
        return "it is the paid Operator gate"
    if edition not in ("lite", "unknown"):
        return "it is the %s edition, not Operator Lite" % edition
    theirs = version_tuple(version) if version else None
    if theirs is None:
        return "its version cannot be read"
    if theirs > version_tuple(VERSION):
        return "it is newer than this one"
    return None


def cmd_install(argv):
    """Owner-side install: copies the gate to ~/.operator/gate and wires Claude Code. A fresh install starts
    in enforce mode; an existing mode file is kept, except the shadow an install before 0.3.9 wrote,
    which becomes enforce once (settle_mode).
    It refuses to replace the paid gate or a newer gate unless the owner passes --force."""
    agent = agent_session()
    if agent:
        print("Refusing: installs are run by the owner in their own terminal (%s is set: this shell "
              "belongs to an agent)." % agent)
        return 3
    dst_dir = pj(HOME, ".operator", "gate")
    src = os.path.abspath(__file__)
    dst = pj(dst_dir, "operator-gate.py")
    why = install_refusal(dst, src)
    replaced = gate_identity(dst) if os.path.lexists(dst) and os.path.realpath(src) != os.path.realpath(dst) else None
    if why:
        theirs = gate_label(*replaced)
        print("\n".join([
            "%s: %s holds %s, and %s." % ("--force given" if "--force" in argv else "Refusing to install",
                                         canon(dst), theirs, why),
            "  installed:  %-28s %s" % (theirs, canon(dst)),
            "  this one:   %-28s %s" % (gate_label(EDITION, VERSION), canon(src))]))
        if "--force" not in argv:
            print("Nothing was changed; the installed gate keeps running.\n"
                  "To replace it with Operator Lite %s anyway, run the install again with --force "
                  "(npx operator-lite --force)." % VERSION)
            return 4
        bdir = pj(HOME, ".operator", "install-backups")
        backup = pj(bdir, "operator-gate-replaced-%s.py" % time.strftime("%Y%m%d-%H%M%S"))
        try:
            os.makedirs(bdir, exist_ok=True)
            shutil.copyfile(dst, backup)
        except Exception as exc:
            print("Refusing: the installed gate could not be kept before replacing it (%s). Nothing was changed." % exc)
            return 4
        print("WARNING: --force replaces %s with Operator Lite %s. What %s does beyond Lite stops now.\n"
              "  the replaced gate is kept at %s\n"
              "  to put it back: cp '%s' '%s'" % (theirs, VERSION, theirs, canon(backup), canon(backup), canon(dst)))
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.realpath(src) != os.path.realpath(dst):
        shutil.copyfile(src, dst)                  # bytes: the installed gate is the downloaded file exactly
        print("gate: %s -> %s%s" % (gate_label(EDITION, VERSION), canon(dst),
                                     " (replaced %s)" % gate_label(*replaced) if replaced else ""))
    os.chmod(dst, 0o755)
    ensure_dirs()
    mode_path = pj(OP_HOME, "mode")
    if not os.path.exists(mode_path):             # a fresh install blocks from the first call
        with open(mode_path, "w", encoding="utf-8") as f:
            f.write("enforce\n")
    _, switched = settle_mode("install")          # an upgrade: the shadow an older install wrote becomes enforce
    # Claude Code PreToolUse hook: merge, never clobber. A settings.json that does not parse is the
    # owner's file in the middle of an edit; it is left exactly as it is.
    settings_path = pj(HOME, ".claude", "settings.json")
    hook_cmd = claude_hook_command(dst)
    registered, raw, settings = False, None, {}
    try:
        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as f:
                raw = f.read()
            settings = json.loads(raw) if raw.strip() else {}
            if not isinstance(settings, dict):
                raise ValueError("its top level is not a JSON object")
    except Exception as exc:
        settings = None
        print("claude: %s could not be read as JSON (%s). It was left as it is." % (settings_path, exc))
    if settings is not None:
        try:
            groups = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
            ours = [hh for g in groups for hh in g.get("hooks", []) if is_claude_gate_hook(hh.get("command", ""))]
            if ours:                                  # an older registration of this gate: update it in place
                ours[0]["command"] = hook_cmd
                for g in groups:
                    g["hooks"] = [hh for hh in g.get("hooks", []) if hh is ours[0] or hh not in ours[1:]]
                groups[:] = [g for g in groups if g.get("hooks")]
            else:
                groups.append({"matcher": "*", "hooks": [{"type": "command", "command": hook_cmd}]})
            # OP-025: at a compaction the session seals what the gate recorded for it, and the session that
            # continues is handed it before it changes anything
            for event, matcher, args in TURNOVER_HOOKS:
                want = claude_side_command(dst, args)
                lst = settings["hooks"].setdefault(event, [])
                mine = [hh for g in lst for hh in g.get("hooks", [])
                        if "operator-gate.py" in hh.get("command", "") and args[1] in hh.get("command", "")]
                if mine:
                    mine[0]["command"] = want
                else:
                    lst.append({"matcher": matcher, "hooks": [{"type": "command", "command": want, "timeout": 10}]})
            text = json.dumps(settings, indent=2)
            if raw is None or json.loads(raw or "{}") != settings:
                os.makedirs(os.path.dirname(settings_path), exist_ok=True)   # Claude Code not started yet here
                if raw is not None:
                    backup = "%s.bak-operator-%s" % (settings_path, time.strftime("%Y%m%d-%H%M%S"))
                    with open(backup, "w", encoding="utf-8") as f:
                        f.write(raw)
                    print("claude: the previous settings are kept at %s" % backup)
                with open(settings_path, "w", encoding="utf-8") as f:
                    f.write(text)
            registered = True
            print("claude: PreToolUse hook registered in %s" % settings_path)
            print("claude: PreCompact hook seals what each compacted session was stopped from, and the "
                  "SessionStart hook hands it over before the session changes anything")
        except Exception as exc:
            print("claude: could not register hook (%s)." % exc)
    if not registered:
        print("  Nothing was written. Add this group to \"hooks\" in %s by hand, then run install again:" % settings_path)
        print('  "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": %s}]}]'
              % json.dumps(hook_cmd))
    print("other hosts: register this as their pre-tool hook -> %s" % hook_command(dst, "<host>"))
    try:
        launcher = place_launcher(dst)
    except Exception:
        launcher = None
    if launcher:
        print("command: operator-gate (%s)" % launcher)
    if switched:
        print("%s." % SETTLED_NOTE)
    print(install_mode_line(global_mode()))
    if "--no-replay" not in argv:                 # what this gate would have said about last month
        try:
            print()
            cmd_replay(["--budget", "12", "--from-install"])
        except Exception:
            pass
    if not registered:
        print("\nThe gate is installed, but Claude Code is not wired to it yet (see above).")
        return 5
    print("\nFrom now on every tool call your agents make passes through the gate.\n"
          "See what it caught: %s status" % self_cmd())
    return 0


def cmd_uninstall(argv):
    agent = agent_session()
    if agent:
        print("Refusing: uninstalls are run by the owner in their own terminal (%s is set: this shell "
              "belongs to an agent)." % agent)
        return 3
    settings_path = pj(HOME, ".claude", "settings.json")
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        for event in ("PreToolUse",) + tuple(e for e, _, _ in TURNOVER_HOOKS):
            if event not in settings.get("hooks", {}):
                continue
            for pre in settings["hooks"][event]:
                pre["hooks"] = [h for h in pre.get("hooks", []) if "operator-gate" not in h.get("command", "")]
            settings["hooks"][event] = [p for p in settings["hooks"][event] if p.get("hooks")]
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        print("claude hook removed")
    except Exception as exc:
        print("claude settings untouched (%s)" % exc)
    for d in (pj(HOME, ".local", "bin"), pj(HOME, "bin")):       # the launcher, when it is ours
        target = pj(d, "operator-gate")
        try:
            with open(target, encoding="utf-8") as f:
                ours = ".operator/gate/operator-gate.py" in f.read()
            if ours:
                os.remove(target)
                print("launcher removed: %s" % target)
        except Exception:
            pass
    return 0


def main():
    utf8_streams()
    argv = sys.argv[1:]
    if not argv or argv[0] == "hook":
        sys.exit(hook_main(argv[1:]))
    cmd = {"approve": cmd_approve, "verify": cmd_verify, "status": cmd_status, "corpus": cmd_corpus,
           "test": cmd_corpus, "install": cmd_install, "uninstall": cmd_uninstall,
           "upgrade": cmd_upgrade, "replay": cmd_replay, "mode": cmd_mode,
           "append": cmd_append, "protect": cmd_protect, "protected": cmd_protected,
           "brake": cmd_brake, "turnover": cmd_turnover, "canary": cmd_canary}.get(argv[0])
    if not cmd:
        print("usage: operator-gate hook|status|replay|mode|approve|verify|corpus|install|uninstall|upgrade"
              "|append|protect|protected|brake|turnover|canary")
        sys.exit(2)
    sys.exit(cmd(argv[1:]))


if __name__ == "__main__":
    main()

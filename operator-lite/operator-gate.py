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
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat as _stat
import subprocess
import sys
import tempfile
import time

VERSION = "0.3.4"
INTEGRITY = "reference_digest_not_signature"
HOME = os.path.expanduser("~")
OP_HOME = os.environ.get("OPERATOR_HOME", os.path.join(HOME, ".operator"))
PERMIT_TTL_S = 15 * 60
MAX_DEPTH = 4

SHELL_TOOLS = {"bash", "shell", "local_shell", "exec_command", "container.exec",
               "run_shell_command", "terminal", "run_terminal_cmd"}
WRITE_TOOLS = {"write", "edit", "multiedit", "notebookedit", "apply_patch", "str_replace_editor",
               "create_file", "edit_file", "write_file"}


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
    if not os.path.isabs(p):
        p = os.path.join(cwd or os.getcwd(), p)
    return os.path.normpath(p)


def real(path):
    try:
        return os.path.realpath(path)
    except Exception:
        return path


def variants(path):
    """Lexical + symlink-resolved form, so ~/vestige and ~/Developer/vestige are one target."""
    out = {path, real(path)}
    return {v.rstrip("/") or "/" for v in out}


def is_same_or_ancestor(target, protected):
    """True when deleting/moving `target` removes `protected` (equal or an ancestor of it)."""
    for t in variants(target):
        for p in variants(protected):
            if t == p or p.startswith(t.rstrip("/") + "/") or t == "/":
                return True
    return False


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
    d = os.path.join(HOME, "Developer")
    return [
        os.path.join(d, "vestige"), os.path.join(HOME, "vestige"),
        os.path.join(d, "vestige-launch-private"), os.path.join(d, "vestige-operator"),
        os.path.join(d, "vestige-cloud"), os.path.join(d, "vestige-evidence"),
        os.path.join(d, "vestige-LIMEN"), os.path.join(d, "vestige-nc"),
        os.path.join(d, "vestige-ollama"), os.path.join(d, "vestige-extra"),
        d,
        os.path.join(HOME, ".vestige"), os.path.join(HOME, ".zcode"),
        os.path.join(HOME, ".claude"), os.path.join(HOME, ".codex"),
        os.path.join(HOME, ".operator"), HOME,
    ]


def gate_homes():
    return sorted({os.path.abspath(OP_HOME), os.path.join(HOME, ".operator")})


def touches_gate_home(path):
    return any(is_inside(path, g) for g in gate_homes())


def self_protected_files():
    return [
        os.path.join(HOME, ".claude", "settings.json"),
        os.path.join(HOME, ".claude", "settings.local.json"),
        os.path.join(HOME, ".zcode", "cli", "config.json"),
        os.path.join(HOME, ".codex", "hooks.json"),
        os.path.join(HOME, ".codex", "config.toml"),
    ]


RULES = {
    "OP-000": ("protect-the-gate", "STOP",
               "Editing the gate, its rules, permits, mode or the hook registrations is owner-only."),
    "OP-001": ("protect-workspaces", "STOP",
               "Deleting or moving a registered workspace root (or a parent of one) is owner-only."),
    "OP-002": ("protect-memory", "STOP",
               "Wiping, deleting or rewriting the Vestige memory store is owner-only."),
    "OP-003": ("no-blind-recursive-delete", "STOP",
               "Recursive delete outside scratch/build directories needs the owner's approval."),
    "OP-004": ("no-history-destruction", "STOP",
               "Force-pushing over a shared branch or deleting .git destroys history."),
    "OP-005": ("no-unreviewed-publish", "STOP",
               "Publishing a release/package or deleting a public repo needs the owner's review."),
    "OP-006": ("no-paid-deploy", "STOP",
               "Deploying the billing/control plane or acting on live billing is not cleared (launch gates not passed)."),
    "OP-007": ("no-destructive-sql", "STOP",
               "DROP/TRUNCATE/unscoped DELETE against a database needs the owner's approval."),
    "OP-008": ("no-shell-init-write", "STOP",
               "Writing shell rc/init files plants commands that fire later; that is code execution by install."),
    "OP-009": ("no-reverse-shell", "STOP",
               "/dev/tcp, /dev/udp, nc -e and DNS-tunneling tools are raw outbound command channels, not tooling."),
    "OP-010": ("no-cloud-metadata", "STOP",
               "Cloud metadata endpoints hand out instance credentials; an agent has no legitimate reason to query them."),
    "OP-CANARY": ("canary", "STOP",
                  "Verification canary: this string exists only to prove the gate is wired."),
    # shadow-only candidates: recorded, never blocking, promote after reviewing the log
    "OP-S01": ("work-loss", "SHADOW", "git reset --hard / clean / branch -D can discard uncommitted work."),
    "OP-S02": ("outbound-comms", "SHADOW", "Sending a message/email/forward on the owner's behalf."),
    "OP-S03": ("public-mutation", "SHADOW", "gh pr/issue/discussion mutation (claim-gate territory)."),
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
               "Secrets or sensitive files traveling through MCP tool arguments (the Invariant/postmark/Supabase pattern)."),
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
           "scratch_ok": []}
    try:
        with open(os.path.join(OP_HOME, "commitments.json")) as f:
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
            with open(os.path.join(OP_HOME, "mode")) as f:
                m = f.read().strip()
        except Exception:
            m = "enforce"
    return m if m in ("enforce", "shadow", "off") else "enforce"


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
            "caffeinate", "timeout", "xargs", "unbuffer", "setsid"}


def strip_wrappers(toks):
    """Drop leading VAR=x, wrappers and their flags. Returns (tokens, via_xargs)."""
    via_xargs = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            i += 1; continue
        base = os.path.basename(t)
        if base in WRAPPERS:
            if base == "xargs":
                via_xargs = True
            i += 1
            while i < len(toks) and toks[i].startswith("-"):
                i += 1
                if base in ("timeout", "nice", "sudo") and i < len(toks) and re.match(r"^[0-9]+$", toks[i]):
                    i += 1
            if base == "timeout" and i < len(toks) and re.match(r"^[0-9.]+[smhd]?$", toks[i]):
                i += 1
            continue
        break
    return toks[i:], via_xargs


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
MEMORY_DIRS = [os.path.join(HOME, ".vestige"),
               os.path.join(HOME, "Library", "Application Support", "com.vestige.core")]
MEMORY_FILE_RE = re.compile(r"vestige\.db|strata|\.wal$|\.shm$|\.sqlite3?$", re.I)
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "{", "(", "!", "}", ")", "done", "fi", "esac"}
LOOP_HEADS = {"for", "case", "select", "function"}
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


def expand_vars(text, vars_):
    def rep(m):
        name = m.group(1) or m.group(2)
        if name == "IFS":                      # word separator: `rm$IFS-rf` must reassemble
            return " "
        v = vars_.get(name)
        return v if v is not None else m.group(0)
    return re.sub(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", rep, text)


def tilde(v):
    return HOME + v[1:] if v == "~" or v.startswith("~/") else v


BRACE_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")
GLOB_CHARS_RE = re.compile(r"[*?\[]")
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
            if GLOB_CHARS_RE.search(variant) and cwd:
                out += [norm(h, cwd) for h in glob_targets(variant, cwd)]
            else:
                out.append(resolve(variant, cwd))
    return out


def resolve(path, cwd):
    """norm() that keeps unknown-cwd relatives recognisable."""
    if path and not os.path.isabs(tilde(path.replace("${HOME}", HOME).replace("$HOME", HOME))) and cwd is None:
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
    elif prog in ("sed", "perl", "awk", "gawk", "ruby", "ed", "ex"):
        inplace = any(f == "--in-place" or f.startswith("--in-place=") or
                      (not f.startswith("--") and "i" in f[1:] and prog in ("sed", "perl", "ruby")) or f == "-i"
                      for f in flags) or (prog in ("awk", "gawk") and "inplace" in " ".join(rest)) or prog in ("ed", "ex")
        if inplace:
            out += args
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
    elif prog in ("curl", "wget"):
        for k, t in enumerate(rest):
            if t in ("-o", "-O", "--output", "--output-document") and k + 1 < len(rest):
                out.append(rest[k + 1])
    return [resolve(w, cwd) for w in out if w]


FORK_BOMB_RE = re.compile(r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&?\s*\}\s*;?\s*:"
                          r"|\(\s*\)\s*\{\s*.*\|\s*&?\s*.*\}\s*;\s*.*&")
PIPE_SINK_RE = re.compile(
    r"\|\s*(?:(?:sudo|env|command|nohup|timeout)\s+)*"
    r"(bash|sh|zsh|dash|ksh|fish|python3?|node|perl|ruby|osascript)\b")
B64_BLOB_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/=]{24,})(?![A-Za-z0-9+/=])")


def decode_b64_text(cmd):
    """Best-effort decode of base64-looking blobs, so `echo <b64> | base64 -d | sh` is analyzed."""
    out = []
    for m in B64_BLOB_RE.finditer(cmd):
        blob = m.group(1)
        try:
            import base64
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8", "replace")
        except Exception:
            continue
        if any(c.isalpha() for c in dec) and any(k in dec for k in ("rm ", "sh ", "bash", "curl", "wget",
                                                                   "eval", "python", "chmod", "dd ", "mkfs")):
            out.append(dec)
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

SCRIPT_ARG_INTERPRETERS = ("bash", "sh", "zsh", "dash", "ksh", "fish",
                           "python", "python3", "python2", "perl", "ruby",
                           "node", "deno", "bun", "php", "lua", "source", ".")


def resolve_script_arg(rest, ecwd):
    """First plausible script-path token from interpreter args (post-expansion)."""
    for t in rest:
        if t.startswith("-"):
            continue
        if t in (".", ".."):
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


def script_body_effects(path, ecwd, depth, vars_):
    """Analyze an existing script file's body through the same walker. Never executes it."""
    try:
        st = os.stat(path)
        if not _stat.S_ISREG(st.st_mode) or st.st_size == 0 or st.st_size > MAX_SCRIPT_BYTES:
            return []
        with open(path, "r", errors="replace") as f:
            body = f.read(MAX_SCRIPT_BYTES)
    except Exception:
        return []
    if not body.strip() or depth + 1 > MAX_DEPTH:
        return []
    effs = analyze(body, ecwd, depth + 1, vars_)
    base = os.path.basename(path)
    for e in effs:
        e["script_body"] = base
    return effs


def analyze(cmd, cwd, depth=0, vars_=None):
    """Walk a shell command tracking cd and simple VAR=value, yielding effects (never executes anything)."""
    effects = []
    if depth > MAX_DEPTH or not cmd or not cmd.strip():
        return effects
    vars_ = dict(vars_) if vars_ is not None else {"HOME": HOME}
    state = {"cwd": cwd}
    if INVISIBLE_RE.search(cmd):
        effects.append({"prog": "\u200b", "seg": cmd[:120], "targets": [], "recursive": False,
                        "kind": "invisible-chars", "filtered": False, "unresolved": False, "flags": [],
                        "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [], "text": cmd[:120]})
        cmd = INVISIBLE_RE.sub("", cmd)
    cmd = decode_ansi_c_quotes(cmd)
    cmd, bodies = extract_heredocs(cmd)
    # subshell groups: `(sleep 300; rm -rf ~) &` -- analyze the inside as its own command
    for m in re.finditer(r"\(([^()]*)\)", cmd):
        inner = m.group(1).strip()
        if inner:
            effects += analyze(inner, state["cwd"], depth + 1, vars_)
    for inner in SUBST_RE.findall(cmd):
        effects += analyze(inner[2:-1] if inner.startswith("$(") else inner[1:-1], state["cwd"], depth + 1, vars_)
    # pipe-to-interpreter: only when the interpreter reads its PROGRAM from the pipe
    # (`curl X | sh`). `curl X | python3 -c 'code'` pipes data, not code -- benign.
    for m in PIPE_SINK_RE.finditer(cmd):
        tail = cmd[m.end():cmd.find("\n", m.end()) if "\n" in cmd[m.end():] else len(cmd)]
        tail = re.split(r"[|;&]", tail)[0]
        code_from_stdin = not re.search(r"\s-\w*c\w*\b|/dev/fd|<[)\w]", tail) and \
            not re.search(r"^\s+[\w./~$]", tail)
        if code_from_stdin:
            effects.append({"prog": m.group(1), "seg": m.group(0), "targets": [], "recursive": False,
                            "kind": "pipe-exec", "filtered": False, "unresolved": False, "flags": [],
                            "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [], "text": m.group(0)})
    for decoded in decode_b64_text(cmd):
        effects += analyze(decoded, state["cwd"], depth + 1, vars_)
    if FORK_BOMB_RE.search(cmd):
        effects.append({"prog": ":()", "seg": cmd[:200], "targets": ["/"], "recursive": True, "kind": "delete",
                        "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": UNKNOWN_CWD,
                        "writes": [], "text": cmd[:200]})
    cmd = SUBST_RE.sub("__SUBST__", cmd)
    hd = 0
    for seg in split_segments(cmd):
        nh = len(HEREDOC_RE.findall(seg))
        my_bodies, hd = bodies[hd:hd + nh], hd + nh
        toks = tokenize(expand_vars(seg, vars_))
        while toks and toks[0] in KEYWORDS:
            toks = toks[1:]
        if not toks or toks[0] in LOOP_HEADS:
            continue
        j = 0
        while j < len(toks) and toks[j] in ("export", "local", "declare", "readonly", "typeset"):
            j += 1
        k = j
        while k < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[k]):
            k += 1
        if k > j and k == len(toks):                       # pure assignment segment
            for t in toks[j:k]:
                name, val = t.split("=", 1)
                if "__SUBST__" in val or "$" in val:
                    vars_.pop(name, None)
                else:
                    vars_[name] = tilde(val)
            continue
        toks, via_xargs = strip_wrappers(toks)
        if not toks:
            continue
        prog = os.path.basename(toks[0])
        rest = toks[1:]
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
        ecwd = state["cwd"]
        eff = {"prog": prog, "seg": seg, "targets": [], "recursive": False, "kind": None, "filtered": False,
               "unresolved": via_xargs, "flags": flags, "args": args, "cwd": ecwd or UNKNOWN_CWD,
               "writes": write_targets(prog, rest, flags, args, expand_vars(seg, vars_), ecwd), "text": seg}

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
            # opaque-script resolution: `bash totally_harmless.sh` executes the FILE body
            sp = resolve_script_arg(rest, ecwd)
            if sp:
                effects += script_body_effects(sp, ecwd, depth, vars_)
            continue
        if prog == "eval":
            if eff["writes"]:
                effects.append(eff)
            effects += analyze(" ".join(rest), ecwd, depth + 1, vars_)
            continue

        if prog in ("source", "."):                          # `source x.sh` executes the file body
            if eff["writes"]:
                effects.append(eff)
            sp = resolve_script_arg(rest, ecwd)
            if sp:
                effects += script_body_effects(sp, ecwd, depth, vars_)
            continue

        if prog in ("rm", "rmdir", "unlink", "shred", "trash", "rip", "srm") or \
                (prog == "gio" and rest[:1] == ["trash"]):
            eff["kind"] = "delete"
            eff["recursive"] = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags
            eff["targets"] = [t for t in resolve_targets([a for a in args if not (prog == "gio" and a == "trash")], ecwd)]
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
            joined = " ".join(rest)
            if "-delete" in rest or re.search(r"-exec(dir)?\s+(rm|shred|unlink)\b", joined):
                eff["kind"] = "delete"
                eff["recursive"] = True
                eff["filtered"] = any(a in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-mtime",
                                            "-mmin", "-newer", "-size", "-user", "-perm", "-type", "-empty")
                                      for a in rest)
                eff["targets"] = resolve_targets(roots or ["."], ecwd)
        elif prog == "rsync" and any("--remove-source-files" in f or f == "--delete" for f in flags):
            eff["kind"] = "delete"
            eff["recursive"] = True
            eff["targets"] = [resolve(a, ecwd) for a in args[:-1]]
        elif prog == "git":
            eff["kind"] = "git"
            g = list(rest)
            while g and g[0].startswith("-"):
                g = g[2:] if g[0] in ("-C", "-c", "--git-dir", "--work-tree") else g[1:]
            eff["sub"] = g[0] if g else ""
            eff["gargs"] = g[1:]
        elif prog in INTERPRETERS:
            eff["kind"] = "inline"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
            sp = resolve_script_arg(rest, ecwd)
            if sp:
                effects += script_body_effects(sp, ecwd, depth, vars_)
        elif prog in ("psql", "sqlite3", "mysql", "mariadb", "duckdb", "mongosh", "redis-cli", "supabase"):
            eff["kind"] = "db"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
        elif prog in ("fly", "flyctl", "stripe", "vercel", "wrangler", "netlify", "gh", "npm", "pnpm", "yarn",
                      "cargo", "twine", "docker", "vestige", "vestige-mcp", "operator-gate"):
            eff["kind"] = "cli"
        # anything that resolved into an unknown cwd cannot be judged statically
        for key in ("targets", "writes"):
            kept = [x for x in eff[key] if not x.startswith(UNKNOWN_CWD) and "$" not in x and "__SUBST__" not in x]
            if len(kept) != len(eff[key]):
                eff["unresolved"] = True
            eff[key] = kept
        effects.append(eff)
    return effects


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def digest_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def is_scratch(path):
    for t in variants(path):
        if any(t == s or t.startswith(s + "/") for s in SCRATCH_OK):
            return True
        tmp = os.environ.get("TMPDIR", "").rstrip("/")
        if tmp and (t == tmp or t.startswith(tmp + "/")):
            return True
        if os.path.basename(t) in BUILD_DIRS:
            return True
    return False


def target_hits(kind, t, recursive, roots, gate_files):
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
    hits += write_hits(t, gate_files, kind)
    if (os.path.basename(t) == ".git" or "/.git/" in t) and kind in ("delete", "move") \
            and not is_scratch(t):
        hits.append(("OP-004", "%s of %s" % (kind, t), "Do not remove or rewrite .git internals; ask the owner."))
    return hits


def write_hits(w, gate_files, verb="write to"):
    hits = []
    if touches_gate_home(w) or any(is_inside(w, f) for f in gate_files):
        hits.append(("OP-000", "%s gate/hook path %s" % (verb, w),
                     "Ask the owner to make this change, or to grant a one-time permit."))
    if any(is_inside(w, d) for d in MEMORY_DIRS) and MEMORY_FILE_RE.search(w):
        hits.append(("OP-002", "%s memory store path %s" % (verb, w),
                     "Use `vestige backup` first; erasure is not something an agent decides."))
    base = w.rstrip("/").rsplit("/", 1)[-1]
    parent = w.rstrip("/").rsplit("/", 1)[-2] if w.rstrip("/").count("/") >= 1 else ""
    is_init = base in SHELL_INIT_FILES or (base == "config.fish" and parent.endswith("fish")) \
        or (w.endswith("/.ssh/rc"))
    if is_init:
        hits.append(("OP-008", "%s shell init file %s" % (verb, w),
                     "Init files execute on every future shell; propose the line to the owner instead "
                     "of writing it."))
    if base.lower() in AGENT_CONFIG_NAMES or parent.endswith("/.cursor/rules") or "/.cursor/rules/" in w:
        hits.append(("OP-S14", "%s agent config %s" % (verb, w),
                     "Agent instruction files are pre-stage prompt injection (Nx/CurXecute); "
                     "propose the change to the owner instead."))
    return hits


def classify_effect(e, cfg, cwd):
    """Yield (rule_id, reason_detail, rewrite_hint) for one effect."""
    hits = []
    kind, text = e.get("kind"), e.get("text", "")
    roots = default_protected_roots() + [norm(p, cwd) for p in cfg["protected_roots"]]
    gate_files = self_protected_files() + [norm(p, cwd) for p in cfg["protected_files"]]
    ecwd = e.get("cwd") if e.get("cwd") and not e["cwd"].startswith(UNKNOWN_CWD) else None

    for w in e.get("writes", []):
        hits += write_hits(w, gate_files)

    if kind == "pipe-exec":
        hits.append(("OP-S06", "code piped straight into %s" % e.get("prog"),
                     "Download to a file, read it, then run it deliberately; the owner reviews the script."))

    if kind == "invisible-chars":
        hits.append(("OP-S12", "zero-width/Bidi control characters in the command",
                     "Legitimate commands never contain invisible characters; retype it plainly."))

    prog0, text0 = e.get("prog", ""), e.get("text", "")
    if CONTAINER_ESCAPE_RE.search(text0):
        hits.append(("OP-S17", "container/VM escape primitive in command",
                     "Docker socket mounts, privileged containers and namespace tools are sandbox-break stages; ask the owner."))
    if "/dev/tcp/" in text0 or "/dev/udp/" in text0 or \
            prog0 in ("nc", "ncat", "netcat") and any(f in ("-e", "-c") for f in e.get("flags", [])) or \
            DNS_TUNNEL_RE.search(text0):
        hits.append(("OP-009", "raw outbound shell/DNS channel",
                     "No agent task needs a reverse shell or DNS tunnel; ask the owner if connectivity is genuinely required."))
    if CLOUD_METADATA_RE.search(text0):
        hits.append(("OP-010", "queries a cloud metadata credential endpoint",
                     "Metadata endpoints mint instance credentials; fetch nothing from 169.254.169.254-class addresses."))
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


    # persistence: crontab/at/launchctl/systemd/tmux-detached
    if prog0 in PERSISTENCE_PROGS:
        mutating = any(a in ("install", "load", "submit", "enable", "new-session", "write", "-") for a in e.get("args", []))
        if prog0 in ("crontab", "at", "batch") or mutating or \
                any(t.startswith("/Library/LaunchAgents") or t.startswith(os.path.join(HOME, "Library", "LaunchAgents"))
                    for t in e.get("writes", [])):
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
        for a in e.get("args", []):
            if a.startswith("@"):
                path = a[1:]
                if path and EXFIL_SECRET_PATH_RE.search(path) or path and os.path.isfile(os.path.expanduser(path)) \
                        and EXFIL_SECRET_PATH_RE.search(path):
                    hits.append(("OP-S11", "uploads %s to the network" % a,
                                 "Never send file contents to remote endpoints; redact and ask the owner."))
                break
            if re.search(r"\.(env|pem|key)$|id_rsa", a, re.I) and a.startswith(("-T", "--upload")):
                hits.append(("OP-S11", "uploads %s" % a, "Redact first; ask the owner."))

    if kind in ("delete", "move"):
        for t in e["targets"]:
            if e.get("filtered"):      # `find X -name ... -delete` never removes X itself
                hits += write_hits(t, gate_files, kind)
                if any(is_inside(t, d) for d in MEMORY_DIRS):
                    hits.append(("OP-002", "filtered delete inside the memory store %s" % t,
                                 "Use `vestige backup` first; erasure is not something an agent decides."))
                elif any(is_same_or_ancestor(t, r) for r in roots):
                    hits.append(("OP-S01", "filtered mass delete inside workspace %s" % t,
                                 "Narrow the filter or delete named files; git protects committed work only."))
            else:
                hits += target_hits(kind, t, e["recursive"], roots, gate_files)
        if kind == "delete" and e["recursive"] and any(t in ("/",) for t in e["targets"]):
            hits.append(("OP-003", "recursive sweep rooted at /",
                         "Scope the find to a named project directory; never sweep from /."))
        if kind == "delete" and e["recursive"] and not e.get("filtered"):
            unsafe = [t for t in e["targets"] if not is_scratch(t)]
            broad = ecwd in (None, "/", HOME) or ecwd in tuple(os.path.join(HOME, d) for d in
                                                              ("Developer", "Documents", "Downloads", "Desktop"))
            wide = [t for t in unsafe if t in ("/", HOME) or t.count("/") <= 2]
            outside = [t for t in unsafe if broad or not is_inside(t, ecwd)]
            if (wide or outside) and not e["unresolved"] and \
                    not any(h[0] in ("OP-001", "OP-000", "OP-002") for h in hits):
                hits.append(("OP-003", "recursive delete of %s" % (wide or outside)[0],
                             "Move it to a dated folder under /tmp, or delete only build outputs "
                             "(target/, node_modules/, dist/)."))
        if e["unresolved"] and kind in ("delete", "move") and not e["targets"]:
            hits.append(("OP-S05", "delete/move with targets from stdin", ""))

    elif kind == "git":
        sub, ga = e.get("sub", ""), e.get("gargs", [])
        force = any(a in ("--force", "-f", "--force-with-lease") or a.startswith("--force-with-lease=") or
                    (a.startswith("+") and len(a) > 1) for a in ga)
        if sub == "push" and force:
            protected = re.compile(r"(^|[:/+ ])(main|master|trunk|release[/\-].*|prod.*)$")
            refs = [a for a in ga if not a.startswith("-")]
            branches = refs[1:] if len(refs) > 1 else []
            bad = any(protected.search(b) for b in branches)
            if not branches:
                try:
                    cur = subprocess.run(["git", "-C", ecwd or cwd or ".", "symbolic-ref", "--short", "HEAD"],
                                         capture_output=True, text=True, timeout=1).stdout.strip()
                    bad = bool(protected.search(cur))
                except Exception:
                    bad = True
            if bad:
                hits.append(("OP-004", "force push to a shared branch", "Push a new branch and open a PR; "
                             "never rewrite main/master."))
        if sub == "reset" and "--hard" in ga or sub == "clean" and any(re.match(r"^-[a-z]*f", a) for a in ga) \
                or sub == "branch" and "-D" in ga or sub in ("stash",) and ga[:1] in (["drop"], ["clear"]) \
                or sub in ("checkout", "restore") and ga[-1:] in (["."], ["--", "."]):
            hits.append(("OP-S01", "git %s %s" % (sub, " ".join(ga[:3])), ""))
        if sub in ("push",) and re.search(r"--delete|:refs/", " ".join(ga)) and "main" in " ".join(ga):
            hits.append(("OP-004", "delete of remote main", "Ask the owner."))

    elif kind in ("cli", "db", "inline"):
        prog, flags, args = e["prog"], e["flags"], e["args"]
        lo = text.lower()
        dry = "--dry-run" in flags or (prog in ("npm", "pnpm", "yarn") and "-n" in flags)
        draft = "--draft" in flags and args[1:2] == ["create"]
        if prog == "gh":
            if args[:1] == ["release"] and args[1:2] in (["create"], ["delete"], ["edit"], ["upload"]) and not draft or \
               args[:1] == ["repo"] and args[1:2] in (["delete"], ["archive"], ["rename"], ["edit"]) or \
               args[:1] == ["api"] and re.search(r"-X\s*(DELETE|PATCH)|--method\s*(DELETE)", text) and "/releases" in text:
                hits.append(("OP-005", "gh %s" % " ".join(args[:2]),
                             "Prepare the artifact and notes as a draft and hand them to the owner."))
            elif args[:1] in (["pr"], ["issue"], ["discussion"], ["gist"]) and \
                    args[1:2] in (["create"], ["edit"], ["comment"], ["close"], ["merge"], ["reopen"]):
                hits.append(("OP-S03", "gh %s" % " ".join(args[:2]), ""))
        if not dry and (prog in ("npm", "pnpm", "yarn") and args[:1] == ["publish"] or
                        prog == "cargo" and args[:1] == ["publish"] or prog == "twine" and args[:1] == ["upload"] or
                        prog == "docker" and args[:1] == ["push"]):
            hits.append(("OP-005", "%s %s" % (prog, args[0]), "Dry-run only (`--dry-run`) and hand off to the owner."))
        fly_groups = {"apps", "app", "secrets", "machine", "machines", "postgres", "pg", "volumes", "volume", "certs",
                      "ips", "regions", "autoscale", "mpg", "redis", "extensions", "tokens", "orgs"}
        fly_mut = {"destroy", "create", "set", "unset", "import", "update", "run", "stop", "start", "restart",
                   "suspend", "resume", "attach", "detach", "failover", "extend", "release", "allocate", "move",
                   "rename", "delete", "remove", "add", "kill", "clone", "fork", "deploy", "scale"}
        if prog in ("fly", "flyctl") and args and (args[0] in ("deploy", "launch", "scale") or
                                                   (args[0] in fly_groups and any(a in fly_mut for a in args[1:]))):
            hits.append(("OP-006", "fly %s" % " ".join(args[:2]), "The Pro bridge is not cleared to deploy; "
                         "ask the owner for a one-time permit for a staging deploy."))
        if prog == "supabase" and args[:2] in (["db", "push"], ["db", "reset"], ["migration", "up"],
                                                ["functions", "deploy"], ["db", "remote"]):
            hits.append(("OP-006", "supabase %s" % " ".join(args[:2]), "Apply migrations to staging only, with a permit."))
        if prog == "stripe" and re.search(r"--live|sk_live|rk_live", text):
            hits.append(("OP-006", "stripe live action", "Use Test mode; live billing is not cleared."))
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
            if re.search(r"\bdrop\s+(table|database|schema|index)\b|(?<![\w(])truncate\s+(table\s+)?[\w\"`]|"
                         r"\bflush(all|db)\b|\bdropdatabase\b", sql) \
                    or re.search(r"\bdelete\s+from\s+\w+\s*(;|\"|'|$)", sql):
                hits.append(("OP-007", "destructive SQL", "Run it as a SELECT first, scope with WHERE, "
                             "and take a backup; ask the owner for a permit."))
            if prog == "sqlite3":
                dbs = [a for a in args if re.search(r"\.(db|sqlite3?|db3)$", a)] or args[:1]
                live = [d for d in dbs if any(is_inside(resolve(d, ecwd), m) for m in MEMORY_DIRS)]
                if live and re.search(r"\b(delete|drop|update|insert|alter|vacuum|replace|truncate)\b", sql):
                    hits.append(("OP-002", "write against the live Vestige store via sqlite3",
                                 "Work on a copy (`vestige backup`, then edit the copy); read-only queries are fine."))
        if e["kind"] == "inline":
            destructive = re.search(r"rmtree|shutil\.move|os\.(remove|unlink|rmdir|removedirs|rename|replace)|rmSync|"
                                    r"rimraf|unlinkSync|fs\.(rm|unlink|rename)|File\.delete|FileUtils\.(rm|mv)|"
                                    r"remove_tree|\bunlink\b", text)
            writing = re.search(r"\bopen\s*\([^)]*['\"][wax][+b]*['\"]|mode\s*=\s*['\"][wax]|write_text|write_bytes|"
                                r"writeFile|appendFile|copyfile|shutil\.copy|fs\.(write|append|copy)|File\.write|tofile",
                                text)
            if destructive or writing:
                paths = [resolve(q, ecwd) for q in re.findall(r"(?:~|\$HOME|\$\{HOME\}|/)[^\s'\"),;\]]*", text)]
                found = []
                for q in paths:
                    if destructive:
                        found += target_hits("delete", q, True, roots, gate_files)
                    if writing:
                        found += write_hits(q, gate_files, "inline code writes to")
                if found:
                    hits += [(rid, "inline code: " + d, h) for rid, d, h in found]
                elif destructive:
                    hits.append(("OP-S05", "inline code deletes/moves files", ""))
    if hits and e.get("script_body"):
        hits = [(r, "%s -- inside script %s" % (d, e["script_body"]), a) for (r, d, a) in hits]
    return hits


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


def classify(payload, cfg):
    tool_raw = str(payload.get("tool_name") or payload.get("toolName") or "")
    tool = tool_raw.lower()
    ti = payload.get("tool_input") or payload.get("toolInput") or payload.get("input") or {}
    if not isinstance(ti, dict):
        ti = {"command": ti}
    cwd = payload.get("cwd") or os.getcwd()
    hits, previews = [], []
    meta = {"normalized": "", "transforms": []}
    effects_out = []
    blob = json.dumps(ti, ensure_ascii=False)

    if "OPERATOR_CANARY_STOP_7f3a" in blob:
        hits.append(("OP-CANARY", "canary string present", "Nothing to fix: this proves the gate ran."))

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
        for eff in analyze(cmd, cwd):
            effects_out.append(eff)
            hits += classify_effect(eff, cfg, cwd)
        if re.search(r"operator-gate(\.py)?['\"]?\s+(approve|mode|install|uninstall)\b", cmd):
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
        for p in write_paths(tool, ti, cwd):
            previews.append("%s %s" % (tool_raw, p))
            hits += write_hits(p, gate_files)   # full coverage: OP-000, OP-002, OP-008, OP-S14
            effects_out.append({"prog": tool, "seg": "%s %s" % (tool_raw, p), "targets": [],
                                "recursive": False, "kind": None, "filtered": False,
                                "unresolved": False, "flags": [], "args": [],
                                "cwd": cwd, "writes": [p], "text": "%s %s" % (tool_raw, p)})
        body = str(ti.get("content") or ti.get("new_string") or "")
        if re.search(r"(sk_live|rk_live|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})", body):
            hits.append(("OP-S04", "credential-shaped string in write", ""))
    elif tool.startswith("mcp__"):
        previews.append(tool_raw)
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
        os.makedirs(os.path.join(OP_HOME, d), mode=0o700, exist_ok=True)


def write_receipt(rec):
    """Append a hash-chained receipt. Never raises into the caller."""
    try:
        ensure_dirs()
        rdir = os.path.join(OP_HOME, "receipts")
        head_path = os.path.join(rdir, "HEAD")
        lock = open(os.path.join(rdir, ".lock"), "a+")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                with open(head_path) as f:
                    prev = f.read().strip() or ("0" * 64)
            except Exception:
                prev = "0" * 64
            rec = dict(rec)
            rec.update({"v": 1, "integrity": INTEGRITY, "gate_version": VERSION, "prev": prev,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"})
            rec["digest"] = digest_of(rec)
            day = time.strftime("%Y-%m-%d", time.gmtime())
            with open(os.path.join(rdir, day + ".jsonl"), "a") as f:
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            with open(head_path, "w") as f:
                f.write(rec["digest"])
            return rec["digest"]
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    except Exception:
        return None


def consume_permit(action_digest):
    p = os.path.join(OP_HOME, "permits", action_digest + ".json")
    try:
        with open(p) as f:
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
def stop(reason_lines):
    sys.stderr.write("\n".join(reason_lines) + "\n")
    sys.stderr.flush()
    sys.exit(2)


def degraded_hit(raw):
    """Gate crashed: fail closed only for text that plainly looks destructive, else fail open."""
    return bool(re.search(r"\brm\s+-[a-zA-Z]*[rR]|\bmv\s+\S*(vestige|Developer|\.zcode|\.claude)|"
                          r"push\s+.*--force|\bDROP\s+TABLE|\.vestige/vestige\.db|fly\s+deploy|OPERATOR_CANARY_STOP_7f3a",
                          raw or "", re.I))




def hook_main(argv):
    source = "unknown"
    if "--source" in argv:
        try:
            source = argv[argv.index("--source") + 1]
        except IndexError:
            pass
    raw = ""
    try:
        raw = sys.stdin.read()
        if os.path.exists(os.path.join(OP_HOME, "DISABLED")) or global_mode() == "off":
            return 0
        payload = json.loads(raw) if raw.strip() else {}
        cfg = load_config()
        tool, cwd, hits, preview, meta, fx_effects = classify(payload, cfg)
        hits = [h for h in hits if h[0] not in cfg["disabled_rules"]]

        gmode = global_mode()
        enforce_hits, shadow_hits = [], []
        for rid, detail, hint in hits:
            name, base, why = RULES[rid]
            mode = cfg["mode_overrides"].get(rid, "shadow" if base == "SHADOW" else "enforce")
            if gmode == "shadow":
                mode = "shadow"
            (enforce_hits if mode == "enforce" else shadow_hits).append((rid, name, why, detail, hint))

        action = {"tool": tool, "source": source, "hits": sorted(set(h[0] for h in hits)),
                  "preview": redact(preview)}
        adigest = digest_of({"tool": tool, "cwd": cwd, "hits": action["hits"], "preview": preview})[:24]
        base_rec = {"source": source, "session": str(payload.get("session_id") or payload.get("sessionId") or "")[:64],
                    "cwd": cwd, "tool": tool, "action_digest": adigest, "action_preview": action["preview"],
                    "normalized": redact(meta.get("normalized", "")),
                    "transforms": meta.get("transforms", [])}

        # specificity ordering: rule class first (canary > self > memory > workspaces), then the
        # most specific target (deepest path) -- the hit reported to the agent is the one to act on.
        priority = ["OP-CANARY", "OP-000", "OP-002", "OP-001", "OP-010", "OP-009", "OP-008", "OP-004",
                    "OP-003", "OP-005", "OP-006", "OP-007",
                    "OP-S06", "OP-S17", "OP-S12", "OP-S13", "OP-S07", "OP-S08", "OP-S11", "OP-S01",
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
            permit = consume_permit(adigest)
            if permit:
                write_receipt(dict(base_rec, decision="ALLOW", via="permit", commitments=[h[0] for h in enforce_hits]))
                return 0
            top = enforce_hits[0]
            rd = write_receipt(dict(base_rec, decision="STOP", commitment=top[0], rule=top[1],
                                    commitments=[h[0] for h in enforce_hits], detail=redact(top[3])))
            lines = ["OPERATOR: STOPPED (%s %s)" % (top[0], top[1]),
                     "Why: %s -- %s." % (top[2], top[3])]
            if top[4]:
                lines.append("Compliant path: %s" % top[4])
            if meta.get("transforms"):
                lines.append("Gate saw through: %s." % ", ".join(meta["transforms"]))
            if meta.get("normalized") and meta["normalized"] != preview:
                lines.append("Normalized form: %s" % redact(meta["normalized"]))
            lines += ["No action was taken. Do not retry this action or a variation of it.",
                      "Only the owner can allow it once, from their own terminal: operator-gate approve %s" % adigest,
                      "Receipt: %s" % ((rd or "unavailable")[:16])]
            stop(lines)
        write_receipt(dict(base_rec, decision="SHADOW_STOP", commitments=[h[0] for h in shadow_hits],
                           detail=redact("; ".join(h[3] for h in shadow_hits))))

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
        print("usage: operator-gate approve <24-hex action digest>")
        return 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: approvals need an interactive terminal run by the owner.")
        return 3
    ensure_dirs()
    recent = ""
    try:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with open(os.path.join(OP_HOME, "receipts", day + ".jsonl")) as f:
            for line in f:
                if argv[0] in line:
                    recent = line
    except Exception:
        pass
    if recent:
        r = json.loads(recent)
        print("Blocked action: %s\n  rule: %s\n  preview: %s" % (r.get("tool"), r.get("rule"), r.get("action_preview")))
    else:
        print("No matching STOP receipt today for %s." % argv[0])
    if input("Type ALLOW ONCE to grant a single-use permit (%d min): " % (PERMIT_TTL_S // 60)).strip() != "ALLOW ONCE":
        print("Cancelled.")
        return 1
    p = os.path.join(OP_HOME, "permits", argv[0] + ".json")
    with open(p, "w") as f:
        json.dump({"action_digest": argv[0], "granted": time.time(), "expires": time.time() + PERMIT_TTL_S,
                   "single_use": True, "by": "owner-tty"}, f)
    os.chmod(p, 0o600)
    write_receipt({"decision": "PERMIT_GRANTED", "action_digest": argv[0], "source": "owner-tty"})
    print("Permit granted for one use.")
    return 0


def cmd_verify(argv):
    rdir = os.path.join(OP_HOME, "receipts")
    prev, n, bad = "0" * 64, 0, []
    files = sorted(f for f in os.listdir(rdir) if f.endswith(".jsonl")) if os.path.isdir(rdir) else []
    for fn in files:
        with open(os.path.join(rdir, fn)) as f:
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


def cmd_status(argv):
    print("operator-gate %s  home=%s  mode=%s  disabled_file=%s" %
          (VERSION, OP_HOME, global_mode(), os.path.exists(os.path.join(OP_HOME, "DISABLED"))))
    for rid, (name, base, why) in sorted(RULES.items()):
        print("  %-9s %-7s %-28s %s" % (rid, base, name, why))
    upgrade_hint()
    return 0


OPERATOR_URL = "https://vestige-pro-production.fly.dev/account"


def upgrade_hint():
    """One line for the person at the terminal. Never printed to an agent, a pipe or a script,
    and never part of a verdict: stop messages go to the model, and a pitch does not belong there."""
    if sys.stdout.isatty() and not os.environ.get("OPERATOR_AGENT_SESSION"):
        print("\nOperator adds your own laws, a daily Board and a weekly Letter: operator-gate upgrade")


def cmd_upgrade(argv):
    """What the paid gate adds and where to get it. `--open` opens the page in a browser."""
    print("""Vestige Operator: the owner's version of this gate, $149 a month.

  Your own laws   Sentences you write become rules the gate enforces on every host, with a
                  compliant rewrite or a stop, and a one-time permit only you can grant.
  The Board       Today's stops and law violations as cards, built from your receipts.
  The Letter      One weekly digest of what your agents tried and what stopped them.
  Onboarding      A five-minute wizard that writes your first laws and proves one stop.

Operator Lite stays free. Operator blocks what is routed through it, and its receipts are
hash-chained digests, not signatures.

Buy:  %s
After checkout the gate arrives by email as a small archive with its checksum.""" % OPERATOR_URL)
    if "--open" in argv and sys.stdout.isatty() and not os.environ.get("OPERATOR_AGENT_SESSION"):
        try:
            import webbrowser
            webbrowser.open(OPERATOR_URL)
        except Exception:
            pass
    return 0


def cmd_corpus(argv):
    """Run an eval corpus through classify(). Pure classification; nothing executes."""
    name = argv[0] if argv else "guardfall"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora", name + ".json")
    try:
        with open(path) as f:
            corpus = json.load(f)
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
                fp = os.path.join(tmpd, fx["name"])
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as f:
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


def cmd_install(argv):
    """Owner-side install: copies the gate to ~/.operator/gate and wires Claude Code. Mode starts SHADOW."""
    if os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: installs are run by the owner in their own terminal.")
        return 3
    dst_dir = os.path.join(HOME, ".operator", "gate")
    src = os.path.abspath(__file__)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "operator-gate.py")
    with open(src) as f:
        body = f.read()
    with open(dst, "w") as f:
        f.write(body)
    os.chmod(dst, 0o755)
    ensure_dirs()
    mode_path = os.path.join(OP_HOME, "mode")
    if not os.path.exists(mode_path):
        with open(mode_path, "w") as f:
            f.write("shadow\n")
    # Claude Code PreToolUse hook (merge, never clobber)
    settings_path = os.path.join(HOME, ".claude", "settings.json")
    hook_cmd = "python3 %s hook --source claude" % dst
    try:
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except Exception:
            settings = {}
        hooks = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
        entry = {"matcher": "*", "hooks": [{"type": "command", "command": hook_cmd}]}
        if not any(h.get("hooks") and any(hh.get("command") == hook_cmd for hh in h["hooks"]) for h in hooks):
            hooks.append(entry)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print("claude: PreToolUse hook registered in %s" % settings_path)
    except Exception as exc:
        print("claude: could not register hook (%s) -- add manually:" % exc)
        print('  "hooks": {"PreToolUse": [{"matcher": "*", "hooks": '
              '[{"type": "command", "command": "%s"}]}]}' % hook_cmd)
    print("zcode:  add to your ZCode hooks config -> %s hook --source zcode" % dst)
    print("codex:  add to ~/.codex/hooks.json        -> %s hook --source codex" % dst)
    print("mode=shadow (log-only). Flip with: echo enforce > ~/.operator/mode")
    return 0


def cmd_uninstall(argv):
    if os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: uninstalls are run by the owner in their own terminal.")
        return 3
    settings_path = os.path.join(HOME, ".claude", "settings.json")
    try:
        with open(settings_path) as f:
            settings = json.load(f)
        for pre in settings.get("hooks", {}).get("PreToolUse", []):
            pre["hooks"] = [h for h in pre.get("hooks", []) if "operator-gate" not in h.get("command", "")]
        settings["hooks"]["PreToolUse"] = [p for p in settings["hooks"]["PreToolUse"] if p.get("hooks")]
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print("claude hook removed")
    except Exception as exc:
        print("claude settings untouched (%s)" % exc)
    return 0


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] == "hook":
        sys.exit(hook_main(argv[1:]))
    cmd = {"approve": cmd_approve, "verify": cmd_verify, "status": cmd_status, "corpus": cmd_corpus,
           "test": cmd_corpus, "install": cmd_install, "uninstall": cmd_uninstall,
           "upgrade": cmd_upgrade}.get(argv[0])
    if not cmd:
        print("usage: operator-gate hook|approve|verify|status|corpus|install|uninstall|upgrade")
        sys.exit(2)
    sys.exit(cmd(argv[1:]))


if __name__ == "__main__":
    main()

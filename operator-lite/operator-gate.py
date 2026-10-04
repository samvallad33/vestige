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

try:
    import fcntl
except ImportError:                                  # Windows: receipts lock through msvcrt instead
    fcntl = None

VERSION = "0.3.8"
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
MAX_DEPTH = 6

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
    p = canon(p)
    if not is_abs(p):
        p = posixpath.join(canon(cwd or os.getcwd()), p)
    return posixpath.normpath(p)


def real(path):
    """Symlinks resolved; on Windows this also turns an 8.3 short name into the long one."""
    try:
        return canon(os.path.realpath(path))
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
               "Wiping, deleting or rewriting the Vestige memory store is owner-only."),
    "OP-003": ("no-blind-recursive-delete", "STOP",
               "Recursive delete outside scratch/build directories needs the owner's approval."),
    "OP-004": ("no-history-destruction", "STOP",
               "Force-pushing over a shared branch or deleting .git destroys history."),
    "OP-005": ("no-unreviewed-publish", "STOP",
               "Publishing a release/package or deleting a public repo needs the owner's review."),
    "OP-006": ("no-paid-deploy", "STOP",
               "Production deploys, live database changes and live billing actions need the owner's approval."),
    "OP-007": ("no-destructive-sql", "STOP",
               "DROP/TRUNCATE/unscoped DELETE, or a command that resets or drops a database, needs the owner's approval."),
    "OP-008": ("no-shell-init-write", "STOP",
               "Writing shell rc/init files plants commands that fire later; that is code execution by install."),
    "OP-009": ("no-reverse-shell", "STOP",
               "/dev/tcp, /dev/udp, nc -e and DNS-tunneling tools are raw outbound command channels, not tooling."),
    "OP-010": ("no-cloud-metadata", "STOP",
               "Cloud metadata endpoints hand out instance credentials; an agent has no legitimate reason to query them."),
    "OP-011": ("no-unrestorable-loss", "STOP",
               "Deleting or discarding work that nothing can bring back needs the owner's approval."),
    "OP-CANARY": ("canary", "STOP",
                  "Verification canary: this string exists only to prove the gate is wired."),
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
           "scratch_ok": []}
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
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", t):
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
MEMORY_DIRS = [pj(HOME, ".vestige"),
               pj(HOME, "Library", "Application Support", "com.vestige.core")]
MEMORY_FILE_RE = re.compile(r"vestige\.db|strata|\.wal$|\.shm$|\.sqlite3?$", re.I)
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "{", "(", "!", "}", ")", "done", "fi", "esac"}
LOOP_HEADS = {"for", "case", "select", "function"}
HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
ARITH_RE = re.compile(r"\$?\(\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)\)")
SUBST_RE = re.compile(r"\$\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)|`[^`]*`")
INTERPRETERS = ("python", "python3", "node", "perl", "ruby", "deno", "bun", "php", "osascript", "lua")
DB_CLIENTS = ("psql", "sqlite3", "mysql", "mariadb", "duckdb", "mongosh", "mongo", "redis-cli", "supabase", "pgcli",
              "mycli", "litecli", "usql", "sqlcmd", "mysqlsh", "cockroach", "clickhouse", "clickhouse-client", "bq",
              "turso")


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


def mask_quotes(text):
    """Same length as text, with every character inside single or double quotes replaced by a
    space, so structural scans (parentheses, pipes) skip quoted data."""
    out, q, i, n = [], None, 0, len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                out.append(" "); out.append(" "); i += 2
                continue
            out.append(c if c == q else " ")
            if c == q:
                q = None
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
    def rep(m):
        name = m.group(1) or m.group(2)
        if name == "IFS":                      # word separator: `rm$IFS-rf` must reassemble
            return " "
        v = vars_.get(name)
        return v if v is not None else m.group(0)
    return re.sub(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*|[0-9]+|[@*])\}|([A-Za-z_][A-Za-z0-9_]*|[0-9]|[@*]))", rep, text)


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


B64_SHORT_RE = re.compile(r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])")
HEX_BLOB_RE = re.compile(r"(?<![0-9A-Za-z])((?:[0-9a-fA-F]{2}){6,})(?![0-9A-Za-z])")
DECODE_CMD_RE = re.compile(r"\bbase64\s+(?:-\w*[dD]\w*|--decode)\b|\bopenssl\s+(?:enc\s+)?(?:-\w+\s+)*-?base64\s+-d|"
                           r"\bb64decode\b|\batob\b")
UNHEX_CMD_RE = re.compile(r"\bxxd\s+(?:-\w+\s+)*-r\b|\bfromhex\b|\bunhexlify\b")
RUNNABLE_WORDS = ("rm ", "sh ", "bash", "curl", "wget", "eval", "python", "chmod", "dd ", "mkfs", "git ", "php ",
                  "node ", "npx ", "rails ", "artisan", "dropdb", "psql", "mysql", "mongosh", "redis-cli",
                  "terraform", "kubectl", "docker", "del ", "rmdir", "Remove-Item", "find ", "mv ", "shred",
                  "truncate", "> /", "drop ", "DROP ")


def decode_b64_text(cmd):
    """Best-effort decode of base64-looking blobs, so `echo <b64> | base64 -d | sh` is analyzed. A blob
    of any length counts when the command itself decodes (`base64 -d`), a long one always; hex is
    read the same way when the command undoes hex (`xxd -r`)."""
    out, seen = [], set()
    blobs = [m.group(1) for m in B64_BLOB_RE.finditer(cmd)]
    if DECODE_CMD_RE.search(cmd):
        blobs += [m.group(1) for m in B64_SHORT_RE.finditer(cmd)]
    for blob in blobs:
        if blob in seen:
            continue
        seen.add(blob)
        try:
            import base64
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8", "replace")
        except Exception:
            continue
        if any(c.isalpha() for c in dec) and "\ufffd" not in dec[:40] and any(k in dec for k in RUNNABLE_WORDS):
            out.append(dec)
    if UNHEX_CMD_RE.search(cmd):
        for m in HEX_BLOB_RE.finditer(cmd):
            try:
                dec = bytes.fromhex(m.group(1)).decode("utf-8")
            except Exception:
                continue
            if any(k in dec for k in RUNNABLE_WORDS):
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
        if cand and is_file(cand):
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
NON_SHELL_SHEBANG_RE = re.compile(r"^#!.*\b(python[0-9.]*|pypy[0-9.]*|node|deno|bun|ruby|perl|php|lua)\b")

# --------------------------------------------------------------------------- #
# files an agent runs are judged by what is inside them, never by their names
# --------------------------------------------------------------------------- #
SEEN_FILES = set()                       # files already opened for the command being judged
WRITTEN = {}                             # files the command being judged writes itself: path -> text, or None
MAX_FILES = 60
UNKNOWN_WORD = "$__CODE"                  # a word the code only knows at run time
CODE_EXT = {".py": "py", ".pyw": "py", ".js": "js", ".mjs": "js", ".cjs": "js", ".ts": "js", ".mts": "js",
            ".cts": "js", ".tsx": "js", ".jsx": "js", ".rb": "rb", ".pl": "pl", ".pm": "pl", ".php": "php",
            ".lua": "lua", ".go": "go", ".r": "r", ".applescript": "applescript"}
CODE_PROG = {"python": "py", "pypy": "py", "ipython": "py", "node": "js", "nodejs": "js", "deno": "js", "bun": "js",
             "tsx": "js", "ts-node": "js", "ts-node-esm": "js", "zx": "js", "ruby": "rb", "jruby": "rb",
             "perl": "pl", "php": "php", "lua": "lua", "luajit": "lua", "osascript": "applescript", "rscript": "r",
             "awk": "awk", "gawk": "awk", "mawk": "awk", "nawk": "awk"}
# the flag after which an interpreter takes its program from the command line
CODE_FLAG = {"py": ("-c",), "js": ("-e", "--eval", "-p", "--print"), "rb": ("-e",), "pl": ("-e", "-E"),
             "php": ("-r",), "lua": ("-e",), "applescript": ("-e",), "r": ("-e",)}
# `python -m <module> file`: these read the file and do not run it
PY_NO_RUN = {"py_compile", "compileall", "black", "ruff", "flake8", "pylint", "mypy", "isort", "autopep8", "yapf",
             "pyflakes", "pycodestyle", "bandit", "pyright", "pydoc", "tokenize", "ast", "dis", "json.tool",
             "tabnanny", "pyclbr", "symtable", "vulture", "radon", "pyupgrade", "pip", "venv", "ensurepip"}
# `deno fmt x.ts`, `bun install`: subcommands that run no file of yours
JS_NO_RUN = {"fmt", "lint", "check", "doc", "info", "cache", "compile", "bundle", "vendor", "install", "i", "add",
             "remove", "rm", "upgrade", "update", "types", "completions", "repl", "init", "build", "pm", "link",
             "unlink", "create", "publish", "outdated", "audit"}
INPUT_REDIRECT_RE = re.compile(r"(?<![<&0-9])<(?![<(&])\s*([^\s<>|&;()]+)")


def guarded(fn, default, *args, **kw):
    """Run one of the readers added in 0.3.8. On a defect it reports nothing instead of failing the
    whole judgement, so the older checks still decide. OPERATOR_DEBUG=1 lets the defect through (tests)."""
    try:
        return fn(*args, **kw)
    except Exception:
        if os.environ.get("OPERATOR_DEBUG"):
            raise
        return default


def code_prog(prog):
    """The language an interpreter runs (`python3.12` is py, `ts-node` is js), or None."""
    p = re.sub(r"\.exe$", "", (prog or "").lower())
    return CODE_PROG.get(p) or CODE_PROG.get(re.sub(r"[\d.]+$", "", p))


def is_file(path):
    """True for a file on disk, and for one an earlier step of the same command writes."""
    return path in WRITTEN or os.path.isfile(path)


def args_after(rest, path, ecwd):
    """The words a script is handed: what follows its own name on the command line."""
    for i, t in enumerate(rest):
        if not t.startswith("-") and t not in (".", "..") and (path is None or resolve(t, ecwd) == path):
            return [tilde(a) for a in rest[i + 1:] if not REDIRECT_TOKEN_RE.match(a)]
    return []


def plain_words(rest):
    """The words of a command up to its first redirection."""
    out = []
    for t in rest:
        if REDIRECT_TOKEN_RE.match(t):
            break
        out.append(t)
    return out


def note_written(prog, rest, bodies, writes):
    """Remember what a command writes into a file when the text is on the command line, so a later
    step that runs the file is judged by it: `echo ... > x.sh && bash x.sh`, `cat > x.py <<EOF`.
    A download is remembered as a file whose text the gate cannot read."""
    text = None
    if prog in ("echo", "printf"):
        text = " ".join(w for w in plain_words(rest) if not (w.startswith("-") and len(w) <= 3)).replace("\\n", "\n")
    elif prog in ("cat", "tee") and bodies:
        text = bodies[0]
    for w in writes:
        if w.startswith(UNKNOWN_CWD) or "$" in w or "__SUBST__" in w:
            continue
        if text is not None and "__SUBST__" not in text:
            WRITTEN[w] = ((WRITTEN[w] + "\n") if WRITTEN.get(w) else "") + text
        elif prog in ("curl", "wget"):
            WRITTEN[w] = None


def stdin_text(fed, rest, ecwd):
    """What a command is handed on standard input when the command line spells it out: a here-string,
    the words an echo or printf pipes in, or the file a cat pipes in. Returns (text, file)."""
    for i, t in enumerate(rest):
        if t == "<<<" and i + 1 < len(rest):
            return rest[i + 1], None
        if t.startswith("<<<") and len(t) > 3:
            return t[3:], None
    if fed:
        fprog, words = fed[0], plain_words(fed[1])
        if fprog in ("echo", "printf"):
            text = " ".join(w for w in words if not (w.startswith("-") and len(w) <= 3)).replace("\\n", "\n")
            return (text, None) if text.strip() and "__SUBST__" not in text else (None, None)
        if fprog in ("cat", "tac", "head", "tail") and ecwd:
            files = [f for f in (resolve(w, ecwd) for w in words if not w.startswith("-")) if is_file(f)]
            if files:
                return None, files[0]
    return None, None


def read_script(path):
    """A script's text. None for a file that is missing, empty, too large, a compiled program, or
    already read for this command."""
    if path in WRITTEN:
        return WRITTEN[path]
    try:
        st = os.stat(path)
        if not _stat.S_ISREG(st.st_mode) or st.st_size == 0 or st.st_size > MAX_SCRIPT_BYTES:
            return None
        key = real(path)
        if key in SEEN_FILES or len(SEEN_FILES) >= MAX_FILES:
            return None
        with open(path, "rb") as f:
            raw = f.read(MAX_SCRIPT_BYTES)
    except Exception:
        return None
    if b"\0" in raw[:4096]:
        return None
    SEEN_FILES.add(key)
    return raw.decode("utf-8", "replace")


def input_file(seg, ecwd):
    """The file a command reads on standard input (`psql app < wipe.sql`, `bash < x.sh`), or None."""
    if ecwd is None:
        return None
    m = INPUT_REDIRECT_RE.search(mask_quotes(seg))
    if not m:
        return None
    p = resolve(seg[m.start(1):m.end(1)].strip("'\""), ecwd)
    return p if is_file(p) else None


def local_module(mod, here, ecwd):
    """The file behind a Python module name when it lives beside the script or in the working directory."""
    rel = (mod or "").replace(".", "/")
    if not rel or rel.startswith("/"):
        return None
    for base in (here, ecwd, pj(ecwd, "src") if ecwd else None):
        if not base:
            continue
        for cand in (pj(base, rel + ".py"), pj(base, rel, "__main__.py"), pj(base, rel, "__init__.py")):
            if os.path.isfile(cand):
                return cand
    return None


def script_file_arg(lang, rest, ecwd):
    """The file an interpreter is about to run, read from its arguments, or None. Its own flags and
    subcommand words (`deno run`, `bun run`) are passed over; a program given on the command line
    (`-c`, `-e`) means there is no file."""
    if ecwd is None:
        return None
    if lang == "awk":                                        # awk runs a file only when told to with -f
        named = [rest[i + 1] for i, t in enumerate(rest[:-1]) if t in ("-f", "--file")]
        return resolve(named[0], ecwd) if named and os.path.isfile(resolve(named[0], ecwd)) else None
    code_flags, i, first = CODE_FLAG.get(lang, ()), 0, True
    while i < len(rest):
        t = rest[i]
        i += 1
        if t in code_flags or t == "-":
            return None
        if lang == "py" and t == "-m":
            mod = rest[i] if i < len(rest) else ""
            if mod in PY_NO_RUN or mod.split(".")[0] in PY_NO_RUN:
                return None
            found = local_module(mod, ecwd, ecwd)
            if found:
                return found
            i, first = i + 1, False
            continue
        if t.startswith("-") or t in (".", ".."):
            continue
        if first and lang == "js" and t in JS_NO_RUN:
            return None
        cand = resolve(t, ecwd)
        if is_file(cand) and (first or os.path.splitext(cand)[1].lower() in CODE_EXT):
            return cand
        first = False
    return None


def inline_codes(lang, rest):
    """The programs handed to an interpreter on its command line: `python -c CODE`, `node -e CODE`."""
    flags, out = CODE_FLAG.get(lang, ()), []
    if lang == "awk":                                        # the program is the first word that is not an option
        i = 0
        while i < len(rest):
            if rest[i] in ("-f", "--file"):
                return []
            if rest[i] in ("-F", "-v", "--assign", "--field-separator"):
                i += 2
            elif rest[i].startswith("-"):
                i += 1
            else:
                return [rest[i]]
        return []
    for i, t in enumerate(rest[:-1]):
        if t in flags or (lang == "js" and t == "eval" and i == 0):
            out.append(rest[i + 1])
    return out


# ---- reading code as text: every language but Python, and Python that does not parse ---------- #
STR_LIT_RE = re.compile(r"""^[rbufRBUF]{0,2}(?:'((?:\\.|[^'\\])*)'|"((?:\\.|[^"\\])*)")$""", re.S)
HOME_EXPR_RE = re.compile(
    r"""^(?:(?:os\.path\.)?expanduser\(\s*['"]~['"]\s*\)|(?:pathlib\.)?Path\.home\(\)|"""
    r"""os\.environ(?:\.get)?[\[(]\s*['"](?:HOME|USERPROFILE)['"]\s*[\])]|os\.getenv\(\s*['"](?:HOME|USERPROFILE)['"]\s*\)|"""
    r"""process\.env\.(?:HOME|USERPROFILE)|process\.env\[\s*['"](?:HOME|USERPROFILE)['"]\s*\]|(?:os\.)?homedir\(\)|"""
    r"""require\(\s*['"](?:node:)?os['"]\s*\)\.homedir\(\)|Dir\.home|ENV\[\s*['"]HOME['"]\s*\]|"""
    r"""ENV\.fetch\(\s*['"]HOME['"]\s*\)|os\.Getenv\(\s*"HOME"\s*\)|getenv\(\s*['"]HOME['"]\s*\)|"""
    r"""\$_SERVER\[\s*['"]HOME['"]\s*\]|\$ENV\{\s*['"]?HOME['"]?\s*\}|\$HOME|Deno\.env\.get\(\s*['"]HOME['"]\s*\)|"""
    r"""Bun\.env\.HOME|os\.UserHomeDir\(\))$""")
CWD_EXPR_RE = re.compile(r"^(?:os\.getcwd\(\)|process\.cwd\(\)|Dir\.pwd|getcwd\(\)|(?:pathlib\.)?Path\.cwd\(\))$")
HERE_NAMES = ("__dirname", "__DIR__", "__dir__", "import.meta.dirname", "File.dirname(__FILE__)",
              "dirname(__FILE__)", "os.path.dirname(__file__)", "os.path.dirname(os.path.abspath(__file__))")
ASSIGN_RE = re.compile(r"^[ \t]*(?:export[ \t]+)?(?:const|let|var|my|our|local)?[ \t]*([$@]?[A-Za-z_]\w*)[ \t]*"
                       r"(?::[ \t]*[\w<>\[\], .|?]+?)?[ \t]*:?=(?![=>])[ \t]*(.+?)[ \t]*;?[ \t]*$", re.M)
SHELL_OUT_RE = re.compile(
    r"(?:\b(?:os\.system|os\.popen|os\.execute|io\.popen|subprocess\.(?:run|call|check_call|check_output|Popen|"
    r"getoutput|getstatusoutput)|pexpect\.(?:run|spawn)|shell_exec|passthru|proc_open|system2?|IO\.popen|"
    r"Kernel\.system|Open3\.(?:capture2e?|capture3|popen2e?|popen3)|exec\.CommandContext|exec\.Command|"
    r"Deno\.Command|execaCommandSync|execaCommand|execaSync|execa)"
    r"|(?:\b|\.)(?:execSync|execFileSync|spawnSync|execFile|spawn|exec|popen))\s*\(")
# languages where a call's further arguments are the words of the command, not options
ARGV_LANGS = ("rb", "pl", "go", "r")
RECURSIVE_DELETE_RE = re.compile(r"rmtree|rimraf|rm_rf|rm_r$|remove_tree|removedirs|RemoveAll|remove_dir|remove_entry|"
                                 r"emptyDir|removeSync|fse?\.remove$|deleteDirectory")
CODE_DELETE_RE = re.compile(
    r"\brmtree\s*\(|\brmSync\s*\(|\brimraf(?:\.sync|Sync)?\s*\(|fs\.(?:rm|rmdir)\w*\s*\(|FileUtils\.(?:rm_rf|rm_r|"
    r"remove_dir|remove_entry_secure|remove_entry)\s*\(|\bremove_tree\s*\(|os\.RemoveAll\s*\(|"
    r"\bemptyDir(?:Sync)?\s*\(|\bremoveSync\s*\(|\bfse?\.remove\s*\(|\bunlink\s*\(|(?:File|Storage)::deleteDirectory\s*\(")
# The same act as a database reset, written as code. Read on lower-cased text.
DB_API = (r"\bdrop_all\s*\(|\bdrop_?database\s*\(|\bdrop_collection\s*\(|\.deletemany\s*\(\s*(?:\{\s*\})?\s*\)|"
          r"\bdelete_many\s*\(\s*\{\s*\}\s*\)|\bsync\s*\(\s*\{\s*force\s*:\s*true|\bflush(?:all|db)\s*\(|"
          r"\.objects\.all\(\)\.delete\(\)|(?:::|->)truncate\(\)|schema::dropall(?:tables|views)\s*\(|"
          r"call_command\(\s*['\"](?:flush|reset_db)['\"]|"
          r"artisan::call\(\s*['\"](?:migrate:(?:fresh|refresh|reset)|db:wipe)['\"]|"
          r"rake::task\[\s*['\"]db:(?:drop|reset|purge|schema:load|truncate_all)")
SQL_DESTROY = (r"(?:drop\s+(?:table|database|schema)\b|truncate\s+(?:table\s+)?\w|"
               r"delete\s+from\s+[\w.\"`\[\]]+\s*(?:['\"`;]|$))")
# in a file, SQL counts where it is handed to the database or kept under a name that says SQL;
# the same words inside any other string are text (a test's sample, a comment, a message)
DB_BODY_RE = re.compile(
    DB_API + r"|(?:execute\w*|exec|query|raw|run|statement|unprepared|prepare|text|sql|\$executeraw\w*|\$queryraw\w*)"
    r"\s*[(`]\s*[frb]?['\"`]{0,3}\s*" + SQL_DESTROY +
    r"|\b\w*(?:sql|query|stmt|statement|ddl)\w*\s*=\s*[frb]?['\"`]{1,3}\s*" + SQL_DESTROY, re.M)
DB_CODE_CASE_RE = re.compile(r"\b[A-Z]\w*(?:::[A-Z]\w*)*\.(?:delete_all|destroy_all)\b(?!\s*\()")
SQL_DESTROY_RE = re.compile(r"\bdrop\s+(?:table|database|schema|index)\b|(?<![\w(])truncate\s+(?:table\s+)?[\w\"`]|"
                            r"\bdelete\s+from\s+[\w.\"`\[\]]+\s*(?:;|$)", re.I)
FETCH_RUN_HINT = "Download it to a file, read it, then run it deliberately; the owner reviews the script."
FETCH_RUN_RE = re.compile(
    r"\b(?:exec|eval)\s*\(\s*(?:await\s+)?(?:\(\s*await\s+)?(?:urllib\.request\.urlopen|urlopen|requests\.(?:get|post)|"
    r"httpx\.get|fetch|file_get_contents\s*\(\s*['\"]https?:|open\s*\(\s*['\"]https?:|Net::HTTP\.get|"
    r"URI\.open|curl_exec)", re.I)
REVERSE_SHELL_CODE_RE = re.compile(
    r"os\.dup2\s*\(\s*\w+\.fileno\(\)|pty\.spawn\s*\(\s*['\"]/bin/(?:ba|z)?sh|"
    r"fsockopen\s*\([^\n]{0,200}(?:/bin/(?:ba)?sh|cmd\.exe)|TCPSocket\.(?:new|open)\s*\([^\n]{0,200}/bin/(?:ba)?sh")


HASH_COMMENT_LANGS = ("py", "rb", "pl", "php", "r", "awk")
SLASH_COMMENT_LANGS = ("js", "go", "php")
TMP_EXPR_RE = re.compile(r"""^(?:tempfile\.gettempdir\(\)|(?:os\.)?tmpdir\(\)|Dir\.tmpdir|sys_get_temp_dir\(\)|"""
                         r"""os\.TempDir\(\)|require\(\s*['"](?:node:)?os['"]\s*\)\.tmpdir\(\))$""")
# calls that reach the network: what they are handed is read for the addresses it names
NET_CALL_RE = re.compile(
    r"\b(?:requests\.\w+|urllib\.request\.\w+|urllib2\.\w+|urlopen|urlretrieve|httpx\.\w+|aiohttp\.\w+|"
    r"http\.client\.\w+|fetch|axios(?:\.\w+)?|got(?:\.\w+)?|https?\.(?:get|request)|net\.(?:connect|createConnection)|"
    r"\w+\.connect|create_connection|curl_init|curl_setopt|file_get_contents|fsockopen|Net::HTTP\.\w+|URI\.open|"
    r"HTTParty\.\w+|Faraday\.\w+|http\.(?:Get|Post)|net\.Dial)\s*\(")
# a payload is only a payload when the code both decodes something and runs something
DECODES_RE = re.compile(r"\b(?:atob|b64decode|base64_decode|decode64|decodebytes|fromhex|unhexlify|hex2bin|a2b_base64)\s*\(|"
                        r"Buffer\.from\s*\(|\bunpack\s*\(|\bpack\s*\(")
RUNS_RE = re.compile(r"\b(?:eval|exec|system|Function|shell_exec|passthru|popen|execSync|spawn\w*|os\.system|"
                     r"subprocess\.\w+|proc_open)\s*\(|`")


def code_mask(text, lang):
    """The text with the insides of its string literals and its comments blanked, at the same length.
    A pattern that starts where the mask is blank is text the code carries, not code it runs."""
    out, i, n = list(text), 0, len(text)
    hashes, slashes = lang in HASH_COMMENT_LANGS, lang in SLASH_COMMENT_LANGS
    while i < n:
        c = text[i]
        if c in "'\"" or (c == "`" and lang == "js"):
            end = text[i:i + 3] if (lang == "py" and text[i:i + 3] in ("'''", '"""')) else c
            j = i + len(end)
            while j < n and not text.startswith(end, j):
                if text[j] == "\\" and j + 1 < n:
                    j += 1
                elif text[j] == "\n" and len(end) == 1 and c != "`":
                    break                                    # a quote left open ends with its line
                j += 1
            for k in range(i + len(end), min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + len(end)
        elif (hashes and c == "#") or (slashes and text.startswith("//", i)) or (lang == "lua" and text.startswith("--", i)):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif slashes and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def in_code(mask, text, pos):
    """True when the character at pos is code: outside every string literal and comment."""
    return mask is None or (pos < len(mask) and mask[pos] == text[pos] and not text[pos].isspace())


def net_call_text(body, mask):
    """What the code hands to its network calls, as text: the only place an address in a file is used."""
    out = []
    for m in NET_CALL_RE.finditer(body):
        if in_code(mask, body, m.start()):
            args = _balanced_args(body, m.end() - 1)
            if args:
                out.append(" ".join(args)[:400])
    return "\n".join(out)


def _split_top(text, sep):
    """Split on a separator that stands outside quotes and brackets."""
    parts, cur, depth, q, i, n = [], [], 0, None, 0, len(text)
    while i < n:
        c = text[i]
        if q:
            cur.append(c)
            if c == "\\" and i + 1 < n:
                cur.append(text[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"`":
            q = c; cur.append(c)
        elif c in "([{":
            depth += 1; cur.append(c)
        elif c in ")]}":
            depth -= 1; cur.append(c)
        elif depth == 0 and text.startswith(sep, i):
            parts.append("".join(cur)); cur = []; i += len(sep); continue
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _wrapped(text):
    """True when the first bracket of the text closes at its last character."""
    depth, q, i, n = 0, None, 0, len(text)
    while i < n:
        c = text[i]
        if q:
            if c == "\\" and i + 1 < n:
                i += 1
            elif c == q:
                q = None
        elif c in "'\"`":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i == n - 1
        i += 1
    return False


def _join_path(vals):
    out = ""
    for v in vals:
        out = v if (not out or v.startswith(("/", "~"))) else out.rstrip("/") + "/" + v
    return out


def code_str(expr, lits, d=0):
    """The string a simple expression stands for: a literal, the home folder, a joined path, a known
    variable, or those put together. None when the code only knows it at run time."""
    e = (expr or "").strip().rstrip(";").strip()
    if not e or d > 8 or len(e) > 600:
        return None
    while e.startswith("(") and _wrapped(e):
        e = e[1:-1].strip()
    m = STR_LIT_RE.match(e)
    if m:
        double = m.group(1) is None
        s = re.sub(r"\\(.)", r"\1", m.group(2) if double else m.group(1))
        prefix = e[:e.index('"' if double else "'")].lower()
        if double and "r" not in prefix:                     # "#{Dir.home}/x" in Ruby, "$home/x" in PHP and Perl
            s = re.sub(r"#\{([^{}]*)\}", lambda k: code_str(k.group(1), lits, d + 1) or UNKNOWN_WORD, s)
            s = re.sub(r"\{?\$\{?([A-Za-z_]\w*)\}?\}?",
                       lambda k: lits.get(k.group(1)) or lits.get("$" + k.group(1)) or k.group(0), s)
        if "f" in prefix:                                    # f"{home}/x"
            s = re.sub(r"\{([A-Za-z_]\w*)\}", lambda k: lits.get(k.group(1)) or UNKNOWN_WORD, s)
        return s
    if len(e) > 1 and e[0] == "`" and e[-1] == "`" and "`" not in e[1:-1]:       # a template literal
        return re.sub(r"\$\{([^{}]*)\}", lambda k: code_str(k.group(1), lits, d + 1) or UNKNOWN_WORD, e[1:-1])
    if HOME_EXPR_RE.match(e):
        return HOME
    if CWD_EXPR_RE.match(e):
        return "."
    if TMP_EXPR_RE.match(e):
        return canon(tempfile.gettempdir())
    if e in HERE_NAMES:
        return lits.get("__here__")
    for sep, joined in ((" . ", False), ("+", False), ("/", True)):
        parts = _split_top(e, sep)
        if len(parts) > 1:
            vals = [code_str(p, lits, d + 1) for p in parts]
            if any(v is None for v in vals):
                return None
            return _join_path(vals) if joined else "".join(vals)
    m = re.match(r"^([\w.:$\\]+)(\(.*\))$", e, re.S)
    if m and _wrapped(m.group(2)):
        short = re.split(r"[.:\\]+", m.group(1))[-1]
        args = [a for a in _split_top(m.group(2)[1:-1], ",") if a]
        vals = [code_str(a, lits, d + 1) for a in args]
        if short in ("expanduser", "expand_path") and vals[:1] and vals[0] is not None:
            return HOME + vals[0][1:] if vals[0].startswith("~") else vals[0]
        if short in ("join", "Join", "resolve", "Path", "PurePath", "PosixPath") and vals and None not in vals:
            return _join_path(vals)
        if short in ("abspath", "realpath", "normpath", "normalize", "str", "String", "fspath", "Clean") and vals[:1]:
            return vals[0]
        if short == "dirname" and vals[:1] and vals[0]:
            return posixpath.dirname(vals[0].rstrip("/")) or "."
        if short in ("mkdtemp", "mkdtempSync", "makeTempDirSync"):          # a fresh folder in the temp directory
            if short == "mkdtemp" and "dir" not in m.group(2):
                return pj(canon(tempfile.gettempdir()), "made-at-run-time")
            if short != "mkdtemp" and vals[:1] and vals[0]:
                return vals[0] + "made-at-run-time"
        return None
    m = re.match(r"^(.+)\.(?:as_posix|resolve|absolute|expanduser|toString|to_s|to_path|decode|strip)\(\)$", e, re.S)
    if m:
        v = code_str(m.group(1), lits, d + 1)
        return HOME + v[1:] if v and v.startswith("~") else v
    if re.match(r"^[$@]?[A-Za-z_]\w*$", e):
        return lits.get(e) if e in lits else lits.get(e.lstrip("$@"))
    return None


def code_literals(body, here=None, mask=None):
    """Names a file gives to strings it can spell out: `home = os.path.expanduser("~")`."""
    lits = {"__here__": here} if here else {}
    for _ in range(2):
        for m in ASSIGN_RE.finditer(body):
            name = m.group(1)
            if name not in lits and in_code(mask, body, m.start(1)):
                v = code_str(m.group(2), lits)
                if v is not None:
                    lits[name] = v
                    lits.setdefault(name.lstrip("$@"), v)
    for m in re.finditer(r"^[ \t]*(\w+)\s*,\s*\w+\s*:?=\s*os\.UserHomeDir\(\)", body, re.M):
        lits[m.group(1)] = HOME
    return lits


def _words(args, lits):
    return [code_str(a, lits) or UNKNOWN_WORD for a in args]


def _list_items(expr):
    """The items of a list written out in code: [a, b], (a, b), c(a, b), array(a, b), or None."""
    e = expr.strip()
    m = re.match(r"^(?:\[|\(|c\(|array\()", e)
    if not m or e[-1] not in "])" or not _wrapped(e[m.end() - 1:]):
        return None
    items = [a for a in _split_top(e[m.end():-1], ",") if a]
    return items if (m.group(0) != "(" or len(items) > 1) else None


def code_command(name, args, lits, lang):
    """The shell command a shell-out call runs, read from its arguments. None when the program
    it runs is only known at run time."""
    if name.endswith("CommandContext"):
        args = args[1:]
    if not args:
        return None
    items = _list_items(args[0])
    if items is not None:                                    # ["rm", "-rf", path]
        words = _words(items, lits)
    else:
        first = code_str(args[0], lits)
        if first is None:
            return None
        rest = [a.strip() for a in args[1:]]
        more = _list_items(rest[0]) if rest else None
        if more is not None:                                 # spawn("rm", ["-rf", path])
            words = [first] + _words(more, lits)
        elif lang in ARGV_LANGS and rest:                    # system("rm", "-rf", path)
            words = [first]
            for a in rest:
                if re.match(r"^:?[A-Za-z_]\w*\s*(?:=>|:(?!:)|=(?!=))", a) or a[:1] == "{":
                    break                                    # an option, not a word of the command
                words.append(code_str(a, lits) or UNKNOWN_WORD)
        else:
            return first                                     # a whole command line in one string
    if not words or words[0] == UNKNOWN_WORD:
        return None
    return " ".join(shlex.quote(w) for w in words)


def _b64_texts(body, mask=None):
    """Text hidden in the file as base64 or hex, decoded: what the code would run after decoding it.
    Only for code that both decodes and runs something; a blob that nothing decodes is data."""
    if mask is not None and not (any(in_code(mask, body, m.start()) for m in DECODES_RE.finditer(body)) and
                                 any(in_code(mask, body, m.start()) for m in RUNS_RE.finditer(body))):
        return []
    out = []
    for m in B64_BLOB_RE.finditer(body):
        blob = m.group(1)
        try:
            import base64
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
        except Exception:
            continue
        out.append(dec)
    for m in re.finditer(r"['\"]((?:[0-9a-fA-F]{2}){8,})['\"]", body):
        try:
            out.append(bytes.fromhex(m.group(1)).decode("utf-8"))
        except Exception:
            continue
    return [t for t in out if len(t) > 5 and sum(c.isprintable() or c in "\n\t" for c in t) >= 0.95 * len(t)
            and re.search(r"[A-Za-z]{2}", t) and re.search(r"[ (]", t)][:8]


def local_files(body, lang, here, mask=None):
    """The local files a piece of code pulls in: ./x required from JavaScript, require_relative in Ruby,
    include in PHP, and so on. Only files that exist beside the script are returned."""
    out = []

    def finditer(pattern, flags=0):
        return [m for m in re.finditer(pattern, body, flags) if in_code(mask, body, m.start())]

    def add(spec, exts):
        if not here or not spec:
            return
        base = resolve(spec, here)
        cands = [base] + [base + x for x in exts] + [pj(base, "index" + x) for x in exts]
        if base.endswith(".js"):
            cands.append(base[:-3] + ".ts")
        for cand in cands:
            if os.path.isfile(cand):
                if cand not in out:
                    out.append(cand)
                return

    if lang == "js":
        for m in finditer(r"""(?:\brequire\s*\(\s*|\bimport\s*\(\s*|\bfrom\s+|\bimport\s+|\bfork\s*\(\s*)"""
                          r"""['"](\.{1,2}/[^'"]+)['"]"""):
            add(m.group(1), (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".mts", ".cts"))
    elif lang == "rb":
        for m in finditer(r"""\b(?:require_relative|require|load)\s*\(?\s*['"]([^'"]+)['"]"""):
            add(m.group(1), (".rb",))
    elif lang == "php":
        lits = {"__DIR__": here}
        for m in finditer(r"\b(?:require|include)(?:_once)?\s*\(?\s*([^;\n]+?)\s*\)?\s*;"):
            v = code_str(m.group(1).replace("__DIR__", "'%s'" % here), lits)
            add(v, ())
    elif lang == "pl":
        for m in finditer(r"""\b(?:require|do)\s+['"]([^'"]+\.p[lm])['"]"""):
            add(m.group(1), ())
    elif lang == "lua":
        for m in finditer(r"""\b(?:require|dofile)\s*\(?\s*['"]([\w./\-]+)['"]"""):
            add(m.group(1).replace(".", "/") if not m.group(1).endswith(".lua") else m.group(1), (".lua",))
    elif lang == "py":
        for m in finditer(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import|import[ \t]+([\w.]+))", re.M):
            f = local_module(m.group(1) or m.group(2), here, None)
            if f and f not in out:
                out.append(f)
    return out[:12]


def text_findings(body, lang, here, ecwd, mask=None):
    """What a piece of code does, read as text: the commands it hands to a shell, the folders it
    deletes outright, the database-destroying calls it makes, the payloads it decodes at run time,
    and the local files it pulls in. Every pattern is matched where the code is; the same words
    inside a string or a comment are text the file carries, and are not read as something it does."""
    out = {"commands": [], "deletes": [], "findings": [], "payloads": [], "files": []}
    mask = mask if mask is not None else code_mask(body, lang)
    lits = code_literals(body, here, mask)

    def code_matches(pattern, text=None):
        return [m for m in pattern.finditer(text if text is not None else body) if in_code(mask, body, m.start())]

    names = SHELL_OUT_RE
    extra = set()
    for m in re.finditer(r"from\s+subprocess\s+import\s+([\w, \t]+)", body):
        extra |= {w for w in re.split(r"[,\s]+", m.group(1)) if w in ("run", "call", "check_call", "check_output", "Popen")}
    for m in re.finditer(r"""\{([^{}]*)\}\s*=\s*require\(\s*['"](?:node:)?child_process['"]\s*\)|"""
                         r"""import\s*\{([^{}]*)\}\s*from\s*['"](?:node:)?child_process['"]""", body):
        for item in (m.group(1) or m.group(2) or "").split(","):       # { execSync: run } / { execSync as run }
            pair = re.split(r"\s*:\s*|\s+as\s+", item.strip())
            if len(pair) == 2 and re.match(r"^\w+$", pair[1]):
                extra.add(pair[1])
    for m in re.finditer(r"""\b(?:const|let|var)\s+(\w+)\s*=\s*require\(\s*['"](?:node:)?child_process['"]\s*\)\.\w+""", body):
        extra.add(m.group(1))
    if extra:
        names = re.compile("(?:" + SHELL_OUT_RE.pattern[:-len(r"\s*\(")] + r"|\b(?:%s))\s*\(" % "|".join(sorted(extra)))
    for m in code_matches(names):
        open_idx = m.end() - 1
        args = _balanced_args(body, open_idx)
        if args:
            cmd = code_command(body[m.start():open_idx].strip().lstrip("."), args, lits, lang)
            if cmd:
                out["commands"].append(cmd)
    if lang in ("rb", "pl", "php"):
        for m in code_matches(re.compile(r"(?<![\w$@])`([^`\n]{2,400})`")):               # `rm -rf x`
            out["commands"].append(code_str('"%s"' % m.group(1).replace('"', '\\"'), lits) or m.group(1))
        for m in code_matches(re.compile(r"(?:%x|qx)\s*([\[({])(.{2,400}?)[\])}]")):
            out["commands"].append(m.group(2))
        for m in code_matches(re.compile(r"""(?m)^[ \t]*(?:system|exec)[ \t]+((?:['"]).+)$""")):   # system "rm -rf x"
            cmd = code_command("system", [a for a in _split_top(m.group(1), ",") if a], lits, lang)
            if cmd:
                out["commands"].append(cmd)
        for m in code_matches(re.compile(r"FileUtils\.(rm_rf|rm_r|remove_dir|remove_entry_secure|remove_entry)"
                                         r"[ \t]+([^\n#;(][^\n#;]*)")):
            t = code_str(_split_top(m.group(2), ",")[0], lits)
            if t:
                out["deletes"].append((t, "FileUtils." + m.group(1)))
    if lang == "js":
        for m in code_matches(re.compile(r"\$\s*`([^`]{2,400})`")):                        # zx and Bun: $`rm -rf x`
            out["commands"].append(code_str("`%s`" % m.group(1), lits) or m.group(1))
    if lang == "awk":                                        # "cmd" | getline, print | "cmd"
        for m in code_matches(re.compile(r'"((?:[^"\\]|\\.)+)"\s*\|\s*getline|\|\s*"((?:[^"\\]|\\.)+)"')):
            out["commands"].append(re.sub(r"\\(.)", r"\1", m.group(1) or m.group(2)))
    for m in code_matches(re.compile(r'do shell script\s+"((?:[^"\\]|\\.)*)"')):           # AppleScript
        out["commands"].append(re.sub(r"\\(.)", r"\1", m.group(1)))
    for m in code_matches(CODE_DELETE_RE):
        open_idx = m.end() - 1
        name = body[m.start():open_idx].strip()
        args = _balanced_args(body, open_idx)
        if not args:
            continue
        if RECURSIVE_DELETE_RE.search(name) or "recursive" in " ".join(args[1:]).lower():
            t = code_str(args[0], lits)
            if t:
                out["deletes"].append((t, name))
    hit = (code_matches(DB_BODY_RE, body.lower()) or code_matches(DB_CODE_CASE_RE) or [None])[0]
    if hit:
        out["findings"].append(("OP-007", "code that empties or drops a database: %s" %
                                " ".join(hit.group(0).split())[:60], DB_RESET_HINT))
    if code_matches(FETCH_RUN_RE):
        out["findings"].append(("OP-S06", "code downloaded at run time and executed", FETCH_RUN_HINT))
    out["payloads"] = _b64_texts(body, mask)
    out["files"] = local_files(body, lang, here, mask)
    return out


# ---- reading Python by its syntax tree: a string that mentions a command is not a call --------- #
PY_SHELL = {"os.system", "os.popen", "subprocess.run", "subprocess.call", "subprocess.check_call",
            "subprocess.check_output", "subprocess.Popen", "subprocess.getoutput", "subprocess.getstatusoutput",
            "pexpect.run", "pexpect.spawn", "commands.getoutput", "asyncio.create_subprocess_shell"}
PY_ARGV = {"asyncio.create_subprocess_exec", "os.execl", "os.execlp", "os.spawnl", "os.spawnlp"}
PY_RMTREE = {"shutil.rmtree", "distutils.dir_util.remove_tree"}
PY_B64 = {"base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode", "base64.decodebytes",
          "binascii.a2b_base64"}
PY_HEX = {"bytes.fromhex", "bytearray.fromhex", "binascii.unhexlify", "binascii.a2b_hex"}
PY_UNPACK = {"zlib.decompress": "zlib", "gzip.decompress": "gzip", "bz2.decompress": "bz2", "lzma.decompress": "lzma"}
PY_FETCH = {"urllib.request.urlopen", "urllib.request.urlretrieve", "urllib2.urlopen", "requests.get",
            "requests.post", "httpx.get", "urlopen"}
PY_SQL_CALLS = {"execute", "executemany", "executescript", "exec_driver_sql", "raw", "query", "run", "exec", "text"}


def py_findings(body, here, ecwd, argv_given=None):
    """What Python source does, read from its syntax tree, so a command that only appears inside a
    string is never taken for a call. None when the source does not parse; the caller then reads it
    as text."""
    try:
        import ast
        import warnings
        with warnings.catch_warnings():                      # the file's own warnings are not the hook's to print
            warnings.simplefilter("ignore")
            tree = ast.parse(body)
    except Exception:
        return None
    alias, env, lists = {}, {"__file__": pj(here or ".", "__main__.py")}, {}
    out = {"commands": [], "deletes": [], "findings": [], "payloads": [], "files": []}

    def add_module(mod, level=0):
        base = here
        for _ in range(max(level - 1, 0)):
            base = posixpath.dirname(base) if base else base
        f = local_module(mod, base, None if level else ecwd) if mod else None
        if f and f not in out["files"]:
            out["files"].append(f)

    def const(n):
        return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None

    def dotted(n):
        if isinstance(n, ast.Name):
            return alias.get(n.id, n.id)
        if isinstance(n, ast.Attribute):
            b = dotted(n.value)
            return (b + "." + n.attr) if b else None
        if isinstance(n, ast.Call):
            f = dotted(n.func)
            if f in ("__import__", "importlib.import_module") and n.args and const(n.args[0]):
                return const(n.args[0])
            if f == "getattr" and len(n.args) >= 2 and const(n.args[1]):
                b = dotted(n.args[0])
                return (b + "." + const(n.args[1])) if b else None
        return None

    def raw(n, d=0):
        """The bytes an expression yields when it decodes a literal: b64decode("..."), bytes.fromhex("...")."""
        if d > 10:
            return None
        if isinstance(n, ast.Constant):
            v = n.value
            return v if isinstance(v, bytes) else (v.encode("utf-8", "replace") if isinstance(v, str) else None)
        if isinstance(n, ast.Name):
            v = env.get(n.id)
            return v.encode("utf-8", "replace") if v is not None else None
        if not isinstance(n, ast.Call):
            return None
        name = dotted(n.func) or ""
        if isinstance(n.func, ast.Attribute) and n.func.attr in ("encode", "decode", "strip") and \
                name not in PY_B64 and name not in PY_HEX and name not in PY_UNPACK and name != "codecs.decode":
            return raw(n.func.value, d + 1)
        inner = raw(n.args[0], d + 1) if n.args else None
        if inner is None:
            return None
        try:
            if name in PY_B64:
                import base64
                return base64.b64decode(inner + b"=" * (-len(inner) % 4))
            if name in PY_HEX:
                return bytes.fromhex(inner.decode("ascii"))
            if name in PY_UNPACK:
                return __import__(PY_UNPACK[name]).decompress(inner)
            if name == "codecs.decode" and len(n.args) > 1 and const(n.args[1]):
                import codecs
                kind = const(n.args[1]).lower().replace("-", "_")
                if kind in ("rot13", "rot_13"):
                    return codecs.decode(inner.decode("utf-8"), "rot_13").encode("utf-8")
                return codecs.decode(inner, kind)
        except Exception:
            return None
        return None

    def val(n, d=0):
        if n is None or d > 12:
            return None
        if isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                return n.value
            return n.value.decode("utf-8", "replace") if isinstance(n.value, bytes) else None
        if isinstance(n, ast.Name):
            return env.get(n.id)
        if isinstance(n, ast.JoinedStr):
            parts = [val(v.value, d + 1) if isinstance(v, ast.FormattedValue) else val(v, d + 1) for v in n.values]
            return None if None in parts else "".join(parts)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Div)):
            left, right = val(n.left, d + 1), val(n.right, d + 1)
            if left is None or right is None:
                return None
            return left + right if isinstance(n.op, ast.Add) else _join_path([left, right])
        if isinstance(n, ast.Subscript):
            if dotted(n.value) == "os.environ" and const(n.slice) in ("HOME", "USERPROFILE"):
                return HOME
            k = n.slice.value if isinstance(n.slice, ast.Constant) else None
            if dotted(n.value) == "sys.argv" and isinstance(k, int) and argv_given and 1 <= k <= len(argv_given):
                return argv_given[k - 1]                     # the word the script was run with
            return None
        if isinstance(n, ast.Attribute):
            if n.attr == "parent":
                b = val(n.value, d + 1)
                return (posixpath.dirname(b.rstrip("/")) or "/") if b else None
            return None
        if not isinstance(n, ast.Call):
            return None
        name = dotted(n.func) or ""
        vals = [val(a, d + 1) for a in n.args]
        a0 = vals[0] if vals else None
        if name in PY_B64 or name in PY_HEX or name in PY_UNPACK or name == "codecs.decode":
            b = raw(n)
            return b.decode("utf-8", "replace") if b is not None else None
        if name.endswith("Path.home") or name.endswith("Path.home()"):
            return HOME
        if name in ("os.path.expanduser", "posixpath.expanduser"):
            return (HOME + a0[1:] if a0.startswith("~") else a0) if a0 is not None else None
        if name in ("os.getenv", "os.environ.get"):
            return HOME if n.args and const(n.args[0]) in ("HOME", "USERPROFILE") else None
        if name in ("os.getcwd", "pathlib.Path.cwd"):
            return ecwd or "."
        if name == "tempfile.gettempdir":
            return canon(tempfile.gettempdir())
        if name == "tempfile.mkdtemp" and not any(k.arg == "dir" for k in n.keywords) and len(n.args) < 3:
            return pj(canon(tempfile.gettempdir()), "made-at-run-time")
        if name in ("os.path.join", "posixpath.join") or re.match(r"^pathlib\.(?:Pure)?(?:Posix|Windows)?Path$", name):
            return None if (not vals or None in vals) else _join_path(vals)
        if name in ("os.path.abspath", "os.path.realpath", "os.path.normpath", "os.fspath", "str", "os.path.expandvars"):
            return a0
        if name == "os.path.dirname":
            return (posixpath.dirname(a0.rstrip("/")) or "/") if a0 else None
        if isinstance(n.func, ast.Attribute):
            attr = n.func.attr
            if attr in ("decode", "strip", "rstrip", "lstrip", "as_posix", "resolve", "absolute", "expanduser", "encode"):
                v = raw(n) if attr == "decode" else None
                v = v.decode("utf-8", "replace") if v is not None else val(n.func.value, d + 1)
                return (HOME + v[1:]) if (v and attr == "expanduser" and v.startswith("~")) else v
            if attr == "join" and len(n.args) == 1 and isinstance(n.args[0], (ast.List, ast.Tuple)):
                sep, items = val(n.func.value, d + 1), [val(e, d + 1) for e in n.args[0].elts]
                return None if (sep is None or None in items) else sep.join(items)
            if attr == "joinpath":
                b = val(n.func.value, d + 1)
                return None if (b is None or None in vals) else _join_path([b] + vals)
        return None

    def argv(n):
        if isinstance(n, (ast.List, ast.Tuple)):
            return [val(e) if val(e) is not None else UNKNOWN_WORD for e in n.elts]
        if isinstance(n, ast.Name) and n.id in lists:
            return lists[n.id]
        if isinstance(n, ast.Call):
            name = dotted(n.func) or ""
            if isinstance(n.func, ast.Attribute) and n.func.attr == "split" and not n.args:
                s = val(n.func.value)
                return s.split() if s else None
            if name == "shlex.split" and n.args:
                s = val(n.args[0])
                try:
                    return shlex.split(s) if s else None
                except ValueError:
                    return s.split()
        return None

    nodes = list(ast.walk(tree))
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    alias[a.asname] = a.name
                else:
                    alias[a.name.split(".")[0]] = a.name.split(".")[0]
                add_module(a.name)
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            add_module(mod, n.level)
            for a in n.names:
                alias[a.asname or a.name] = (mod + "." + a.name) if mod else a.name
                add_module((mod + "." + a.name) if mod else a.name, n.level)
    for _ in range(2):
        for n in nodes:
            target = None
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                target = n.targets[0].id
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                target = n.target.id
            if target and target not in env and target not in lists:
                v = val(n.value)
                if v is not None:
                    env[target] = v
                else:
                    words = argv(n.value)
                    if words is not None:
                        lists[target] = words

    def finding(what):
        out["findings"].append(("OP-007", "code that empties or drops a database: %s" % what, DB_RESET_HINT))

    for n in nodes:
        if not isinstance(n, ast.Call):
            continue
        name = dotted(n.func) or ""
        attr = n.func.attr if isinstance(n.func, ast.Attribute) else ""
        args = n.args
        if name in PY_SHELL and args:
            words = argv(args[0])
            if words is not None:
                if words and words[0] != UNKNOWN_WORD:
                    out["commands"].append(" ".join(shlex.quote(w) for w in words))
            else:
                s = val(args[0])
                if s:
                    out["commands"].append(s)
        elif name in PY_ARGV and args:
            words = [val(a) or UNKNOWN_WORD for a in args]
            if words[0] != UNKNOWN_WORD:
                out["commands"].append(" ".join(shlex.quote(w) for w in words))
        elif name in PY_RMTREE and args:
            t = val(args[0])
            if t:
                out["deletes"].append((t, name))
        elif name in ("exec", "eval") and args:
            code = val(args[0])
            if code:
                out["payloads"].append(code)
            elif any(isinstance(x, ast.Call) and (dotted(x.func) or "") in PY_FETCH for x in ast.walk(args[0])):
                out["findings"].append(("OP-S06", "code downloaded at run time and executed", FETCH_RUN_HINT))
        elif name.endswith("call_command") and args:
            words = [const(a) for a in args if const(a)]
            if words[:1] and (words[0] in ("flush", "reset_db") or (words[0] == "migrate" and "zero" in words[1:])):
                finding("call_command(%r)" % words[0])
        elif name in ("runpy.run_path",) and args and val(args[0]) and here:
            f = resolve(val(args[0]), here)
            if os.path.isfile(f) and f not in out["files"]:
                out["files"].append(f)
        elif name in ("runpy.run_module", "importlib.import_module", "__import__") and args and const(args[0]):
            add_module(const(args[0]))
        elif attr in ("drop_all", "drop_database", "dropDatabase", "drop_collection", "flushall", "flushdb"):
            finding(".%s()" % attr)
        elif attr in ("delete_many", "remove") and args and isinstance(args[0], ast.Dict) and not args[0].keys:
            finding(".%s({})" % attr)
        elif attr == "drop" and not args and not n.keywords and isinstance(n.func.value, (ast.Attribute, ast.Subscript)):
            finding(".drop() on a collection")
        elif attr == "delete" and not args and isinstance(n.func.value, ast.Call) \
                and isinstance(n.func.value.func, ast.Attribute):
            inner = n.func.value.func
            if inner.attr == "all" and isinstance(inner.value, ast.Attribute) and inner.value.attr == "objects":
                finding("Model.objects.all().delete()")
            elif inner.attr == "query":
                finding("query(Model).delete() with no filter")
        elif (attr in PY_SQL_CALLS or name in ("sqlalchemy.text", "text")) and args:
            sql = val(args[0])
            if sql and SQL_DESTROY_RE.search(sql):
                finding("SQL: %s" % " ".join(sql.split())[:60])
    # a loop that lists a folder and deletes what it finds, file by file, empties the folder
    listed, loose = [], False
    for n in nodes:
        if not isinstance(n, ast.Call):
            continue
        name = dotted(n.func) or ""
        attr = n.func.attr if isinstance(n.func, ast.Attribute) else ""
        if name in ("os.walk", "os.listdir", "os.scandir") and n.args and val(n.args[0]):
            listed.append(val(n.args[0]))
        elif attr in ("rglob", "iterdir") and val(n.func.value):
            listed.append(val(n.func.value))
        elif name in ("os.remove", "os.unlink", "os.rmdir", "shutil.rmtree") and n.args and val(n.args[0]) is None:
            loose = True
        elif attr in ("unlink", "rmdir") and not n.args and val(n.func.value) is None:
            loose = True
    for root in listed if loose else ():
        t = resolve(root, ecwd)
        if not t.startswith(UNKNOWN_CWD) and not is_scratch(t) and (ecwd is None or not is_inside(t, ecwd)):
            out["findings"].append(("OP-003", "code that deletes the files it lists under %s, outside the working directory"
                                    % t.replace(HOME, "~"), "Name the files, or run it inside the project; ask the owner for anything wider."))
    return out


def code_effects(body, lang, here, ecwd, depth, vars_, label, argv=None):
    """What a piece of code sets in motion beyond its own lines. Each command it hands to a shell is
    walked like a typed command, each folder it deletes outright becomes a delete, a payload it
    decodes at run time is decoded and read the same way, and the local files it imports are opened."""
    out = []
    if depth > MAX_DEPTH or not body or not body.strip():
        return out
    mask = code_mask(body, lang)
    found = (py_findings(body, here, ecwd, argv) if lang == "py" else None) or text_findings(body, lang, here, ecwd, mask)
    if any(in_code(mask, body, m.start()) for m in REVERSE_SHELL_CODE_RE.finditer(body)):
        found["findings"].append(("OP-009", "code that hands a shell to a network socket",
                                  "No agent task needs a reverse shell; ask the owner if connectivity is genuinely required."))
    base = {"targets": [], "recursive": False, "filtered": False, "unresolved": False, "flags": [], "args": [],
            "cwd": ecwd or UNKNOWN_CWD, "writes": []}

    def tag(effs):
        for e in effs:
            if label:
                e["script_body"] = label
            e["quiet"] = True                                # an unresolved word inside a file is ordinary scripting
        return effs

    for cmd in found["commands"]:
        out += tag(analyze(cmd, ecwd, depth + 1, vars_))
    for target, name in found["deletes"]:
        t = resolve(target, ecwd)
        if not t.startswith(UNKNOWN_CWD) and UNKNOWN_WORD not in t:
            what = "%s(%s)" % (name, target)
            out += tag([dict(base, prog=name, seg=what, kind="delete", recursive=True, targets=[t], text=what)])
    for rule, detail, hint in found["findings"]:
        out += tag([dict(base, prog="code", seg=detail, kind="finding", finding=(rule, detail, hint), text=detail)])
    for code in found["payloads"]:
        out += tag(analyze(code, ecwd, depth + 1, vars_))
        out += tag([dict(base, prog="code", seg="payload decoded at run time", kind="inline", text=code,
                         codes=[(code, lang)], script_body=label or "decoded payload")])
        out += code_effects(code, lang, here, ecwd, depth + 1, vars_, label)
    for path in found["files"]:
        pulled = script_body_effects(path, ecwd, depth, vars_, mode="auto")
        for e in pulled:
            e["quiet"] = True
        out += pulled
    return out


def notebook_effects(body, here, ecwd, depth, vars_, label):
    """A notebook's code cells: `!` lines and %%bash cells are shell, the rest is Python."""
    try:
        cells = json.loads(body).get("cells") or []
    except Exception:
        return []
    code, effs = [], []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        lines = ("".join(src) if isinstance(src, list) else str(src)).split("\n")
        if lines and re.match(r"^\s*%%(?:bash|sh|zsh|script\s+(?:ba|z)?sh)\b", lines[0]):
            effs += analyze("\n".join(lines[1:]), ecwd, depth + 1, vars_)
            continue
        for ln in lines:
            s = ln.strip()
            if s.startswith("!"):
                effs += analyze(s[1:], ecwd, depth + 1, vars_)
            elif not s.startswith("%"):
                code.append(ln)
    py = "\n".join(code)
    effs.append({"prog": "notebook", "seg": label, "targets": [], "recursive": False, "kind": "inline",
                 "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": ecwd or UNKNOWN_CWD,
                 "writes": [], "text": net_call_text(py, code_mask(py, "py")), "codes": [(py, "py")]})
    return effs + code_effects(py, "py", here, ecwd, depth + 1, vars_, label)


def batch_effects(body, ecwd, depth, vars_):
    """A .bat or .cmd file, line by line: its delete forms, and every other line as a command."""
    effs = []
    for line in body.splitlines():
        s = line.strip().lstrip("@")
        if not s or re.match(r"(?i)^(?:rem\b|::|echo\b|setlocal|endlocal|goto\b|:)", s):
            continue
        dels = cmd_exe_effects(s, ecwd)
        effs += dels if dels else analyze(s, ecwd, depth + 1, vars_)
    return effs


def script_body_effects(path, ecwd, depth, vars_, mode="auto", prog=None, seg="", lang=None, argv=None):
    """Analyze an existing script file's body. Never executes it.

    mode="shell": the file is run by a shell (`bash x`, `source x`), walk it as shell lines.
    mode="auto": a shell file (by extension or shebang) is walked as shell; a Python, JS, Ruby,
    Perl, PHP or Lua file becomes ONE inline-code effect in that language, so its variable
    names and string literals can never be mistaken for shell programs (2026-10-03 phantom
    `rg` from `rg, _ = call(...)` inside a .py file), while its file-writing calls are still
    judged by the inline-code analysis. Since 0.3.8 that code is also read for what it sets in
    motion (the commands it hands to a shell, the folders it deletes, the database calls it makes,
    the payloads it decodes, the local files it imports): a file is judged by its contents."""
    if depth + 1 > MAX_DEPTH:
        return []
    base = os.path.basename(path)
    if path in WRITTEN and WRITTEN[path] is None:            # `curl -o x.sh URL && bash x.sh`
        detail = "runs %s, which this command downloads: the gate cannot read it" % base
        return [{"prog": prog or "script", "seg": seg or base, "targets": [], "recursive": False, "kind": "finding",
                 "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": ecwd or UNKNOWN_CWD,
                 "writes": [], "text": detail, "finding": ("OP-S06", detail, FETCH_RUN_HINT)}]
    body = read_script(path)
    if body is None or not body.strip():
        return []
    ext = os.path.splitext(path)[1].lower()
    first = body.split("\n", 1)[0]
    if mode != "shell":
        m = NON_SHELL_SHEBANG_RE.match(first)
        lang = CODE_EXT.get(ext) or (code_prog(m.group(1)) if m else None) or \
            (lang if not first.startswith("#!") else None)
    if mode != "shell" and ext == ".ipynb":
        effs = notebook_effects(body, os.path.dirname(path), ecwd, depth, vars_, base)
    elif mode != "shell" and ext == ".ps1":
        effs = analyze_powershell(body, ecwd, depth + 1)
    elif mode != "shell" and ext in (".bat", ".cmd"):
        effs = batch_effects(body, ecwd, depth, vars_)
    elif mode != "shell" and lang:
        effs = [{"prog": prog or "script", "seg": seg or base, "targets": [], "recursive": False, "kind": "inline",
                 "filtered": False, "unresolved": False, "flags": [], "args": [], "cwd": ecwd or UNKNOWN_CWD,
                 "writes": [], "text": net_call_text(body, code_mask(body, lang)), "codes": [(body, lang)],
                 "script_body": base}]
        return effs + guarded(code_effects, [], body, lang, os.path.dirname(path), ecwd, depth + 1, vars_, base, argv)
    else:
        if argv is not None:                                 # "$1", "$@": what the script was handed
            vars_ = dict(vars_ if vars_ is not None else {"HOME": HOME, "USERPROFILE": HOME})
            for i, a in enumerate(argv[:9], 1):
                vars_[str(i)] = a
            vars_["@"] = vars_["*"] = " ".join(argv)
        effs = analyze(body, ecwd, depth + 1, vars_)
    for e in effs:
        e["script_body"] = base
    return effs


# ---- commands carried by something else ------------------------------------------------------- #
# variables whose value some program runs as a command: `PAGER='rm -rf x' git log`
EXEC_VARS = {"PAGER", "GIT_PAGER", "MANPAGER", "EDITOR", "VISUAL", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR",
             "GIT_SSH_COMMAND", "GIT_EXTERNAL_DIFF", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_PROXY_COMMAND", "LESSOPEN",
             "LESSCLOSE", "BROWSER", "PROMPT_COMMAND", "KUBECTL_EXTERNAL_DIFF", "RUSTC_WRAPPER"}
# variables naming a file a shell reads before it starts
EXEC_FILE_VARS = {"BASH_ENV", "ENV"}
# git settings whose value git runs
GIT_EXEC_KEYS = ("core.pager", "core.editor", "core.sshcommand", "core.fsmonitor", "sequence.editor",
                 "diff.external", "credential.helper", "gpg.program")


def carried_by_vars(toks, ecwd, depth, vars_):
    """Effects of the commands a command line hands over in variables that get run."""
    out = []
    for t in toks:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", t)
        if not m:
            if os.path.basename(t) in ("env", "export"):
                continue
            break
        name, value = m.group(1), m.group(2).strip()
        if name in EXEC_VARS and value:
            out += analyze(value, ecwd, depth + 1, vars_)
        elif name in EXEC_FILE_VARS and value and ecwd:
            out += script_body_effects(resolve(value, ecwd), ecwd, depth, vars_, mode="shell")
    return out


def carried_by_git(rest, sub, gargs):
    """The command lines a git invocation is told to run: -c core.pager=CMD, an alias beginning with !,
    difftool -x CMD, rebase --exec CMD, bisect run CMD, submodule foreach CMD, filter-branch filters."""
    out = []
    for i, t in enumerate(rest[:-1]):
        if t == "-c" and "=" in rest[i + 1]:
            key, value = rest[i + 1].split("=", 1)
            key = key.lower()
            if key in GIT_EXEC_KEYS or key.startswith("pager.") or (key.startswith("alias.") and value.startswith("!")):
                out.append(value.lstrip("!"))
    if sub == "config":
        words = [a for a in gargs if not a.startswith("-")]
        if len(words) >= 2 and (words[0].lower() in GIT_EXEC_KEYS or words[0].lower().startswith("pager.") or
                                (words[0].lower().startswith("alias.") and words[1].startswith("!"))):
            out.append(words[1].lstrip("!"))
    for i, a in enumerate(gargs):
        nxt = gargs[i + 1] if i + 1 < len(gargs) else None
        if sub in ("difftool", "mergetool") and a in ("-x", "--extcmd") and nxt:
            out.append(nxt)
        elif sub in ("difftool", "mergetool") and a.startswith("--extcmd="):
            out.append(a.split("=", 1)[1])
        elif sub == "rebase" and a in ("-x", "--exec") and nxt:
            out.append(nxt)
        elif sub == "rebase" and a.startswith("--exec="):
            out.append(a.split("=", 1)[1])
        elif sub == "filter-branch" and re.match(r"^--[a-z\-]+-filter$", a) and nxt:
            out.append(nxt)
    if sub == "bisect" and gargs[:1] == ["run"] and len(gargs) > 1:
        out.append(" ".join(shlex.quote(a) for a in gargs[1:]))
    if sub == "submodule" and "foreach" in gargs:
        tail = [a for a in gargs[gargs.index("foreach") + 1:] if a != "--recursive"]
        if tail:
            out.append(tail[0] if len(tail) == 1 else " ".join(shlex.quote(a) for a in tail))
    return out


# ---- the commands that set other files running ------------------------------------------------ #
RUNNERS = {"uv": ("run",), "poetry": ("run",), "pipenv": ("run",), "pdm": ("run",), "hatch": ("run",),
           "rye": ("run",), "pixi": ("run",), "conda": ("run",), "mamba": ("run",), "micromamba": ("run",),
           "pipx": ("run",), "bundle": ("exec",), "npm": ("exec", "x"), "pnpm": ("exec", "dlx"),
           "yarn": ("exec", "dlx"), "bun": ("x",), "doppler": ("run",), "op": ("run",), "infisical": ("run",),
           "direnv": ("exec",), "npx": (), "pnpx": (), "bunx": (), "uvx": (), "dotenv": (), "watch": (),
           "xvfb-run": ()}
RUNNER_VALUE_FLAGS = {"-p", "--package", "--with", "--python", "--project", "--env-file", "--directory", "-n",
                      "--name", "--prefix", "-e", "-f", "--cwd", "--config"}
PKG_BUILTINS = {"add", "remove", "install", "i", "ci", "upgrade", "update", "up", "why", "list", "ls", "exec", "dlx",
                "x", "create", "init", "link", "unlink", "publish", "pack", "audit", "outdated", "config", "cache",
                "info", "login", "logout", "workspace", "workspaces", "version", "run", "global", "import",
                "dedupe", "rebuild", "store", "patch", "env", "set", "get", "help", "bin", "prune", "fetch",
                "pm", "repl", "upgrade-interactive"}
GIT_HOOKS = {"commit": ("pre-commit", "prepare-commit-msg", "commit-msg", "post-commit"), "push": ("pre-push",),
             "merge": ("pre-merge-commit", "post-merge"), "pull": ("post-merge",), "checkout": ("post-checkout",),
             "switch": ("post-checkout",), "rebase": ("pre-rebase", "post-rewrite"),
             "am": ("applypatch-msg", "pre-applypatch", "post-applypatch")}
SQL_FILE_EXT = (".sql", ".psql", ".ddl", ".cql", ".js", ".mongodb")


def wrapped_command(prog, rest):
    """The command a runner prefix wraps (`uv run`, `npx`, `bundle exec`, `dotenv --`), as words."""
    subs = RUNNERS.get(prog)
    if subs is None:
        return None
    i = 0

    def skip_flags(i):
        while i < len(rest) and rest[i].startswith("-") and rest[i] != "--":
            i += 2 if (rest[i] in RUNNER_VALUE_FLAGS and "=" not in rest[i]) else 1
        return i

    if subs:
        i = skip_flags(i)
        if i >= len(rest) or rest[i] not in subs:
            return None
        i += 1
    if prog in ("npx", "pnpx") and "-c" in rest[i:]:
        j = rest.index("-c", i)
        if j + 1 < len(rest):
            return tokenize(rest[j + 1])
    i = skip_flags(i)
    if "--" in rest[i:]:
        i = rest.index("--", i) + 1
    if prog == "direnv":
        i += 1
    inner = rest[i:]
    if not inner:
        return None
    if prog in ("uv", "uvx", "pdm", "hatch", "rye", "pixi") and inner[0].lower().endswith(".py"):
        inner = ["python"] + inner
    return inner


def package_scripts(prog, rest, ecwd):
    """The package.json scripts a package-manager command runs, as (directory, name, script)."""
    here, words, i = ecwd, [], 0
    while i < len(rest):
        t = rest[i]
        m = re.match(r"^--(?:prefix|dir|cwd)=(.+)$", t)
        if t in ("--prefix", "-C", "--dir", "--cwd") and i + 1 < len(rest):
            here = resolve(rest[i + 1], ecwd)
            i += 1
        elif m:
            here = resolve(m.group(1), ecwd)
        elif t == "--":
            break
        elif not t.startswith("-"):
            words.append(t)
        i += 1
    try:
        with open(pj(here, "package.json"), encoding="utf-8") as f:
            scripts = json.load(f).get("scripts") or {}
    except Exception:
        return []
    if not isinstance(scripts, dict):
        return []
    w0 = words[0] if words else ""
    names = []
    if w0 in ("run", "run-script", "rum", "urn") and len(words) > 1:
        names = ["pre" + words[1], words[1], "post" + words[1]]
    elif w0 in ("install", "i", "ci") or (prog == "yarn" and not words):
        names = ["preinstall", "install", "postinstall", "prepare"]
    elif w0 in ("test", "t", "tst", "start", "stop", "restart") and prog != "bun":
        w0 = "test" if w0 in ("t", "tst") else w0
        names = ["pre" + w0, w0, "post" + w0]
    elif prog != "npm" and w0 in scripts and w0 not in PKG_BUILTINS:
        names = ["pre" + w0, w0, "post" + w0]
    return [(here, n, scripts[n]) for n in names if isinstance(scripts.get(n), str)]


def make_recipes(path):
    """The targets of a makefile, each with its prerequisites and recipe lines, in file order."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read(MAX_SCRIPT_BYTES).replace("\\\n", " ")
    except Exception:
        return {}
    targets, cur = {}, []
    for line in text.split("\n"):
        if line.startswith("\t"):
            for name in cur:
                targets[name][1].append(line.strip())
            continue
        m = re.match(r"^([A-Za-z0-9_./\-]+(?:[ \t]+[A-Za-z0-9_./\-]+)*)[ \t]*:(?![=:])[ \t]*([^#=]*)(?:#.*)?$", line)
        if m:
            cur = m.group(1).split()
            for name in cur:
                targets.setdefault(name, ([], []))[0].extend(m.group(2).split())
        elif line.strip() and not line.startswith("#"):
            cur = []
    return targets


def make_lines(rest, ecwd):
    """(directory, target, recipe line) for the targets a `make` command builds, prerequisites first."""
    here, mf, goals, i = ecwd, None, [], 0
    while i < len(rest):
        t = rest[i]
        if t in ("-C", "--directory", "-f", "--file", "--makefile", "-j", "-o", "-I", "-W", "-l") and i + 1 < len(rest):
            if t in ("-C", "--directory"):
                here = resolve(rest[i + 1], ecwd)
            elif t in ("-f", "--file", "--makefile"):
                mf = rest[i + 1]
            i += 1
        elif t in ("-n", "--just-print", "--dry-run", "--recon", "-q", "--question", "-p", "--print-data-base"):
            return []
        elif not t.startswith("-") and "=" not in t:
            goals.append(t)
        i += 1
    path = resolve(mf, here) if mf else next((pj(here, n) for n in ("GNUmakefile", "makefile", "Makefile")
                                              if os.path.isfile(pj(here, n))), None)
    targets = make_recipes(path) if path else {}
    if not targets:
        return []
    if not goals:
        goals = [next((n for n in targets if not n.startswith(".")), "")]
    out, seen = [], set()

    def walk(name, d):
        if name in seen or name not in targets or d > 4:
            return
        seen.add(name)
        deps, lines = targets[name]
        for dep in deps:
            walk(dep, d + 1)
        for ln in lines:
            ln = re.sub(r"^[@+\-\s]+", "", ln).replace("$$", "$")
            ln = re.sub(r"\$[({]MAKE[)}]", "make", ln)
            if ln:
                out.append((here, name, ln))

    for g in goals:
        walk(g, 0)
    return out


def composer_scripts(rest, ecwd):
    """The composer.json scripts a `composer` command runs, as (name, command line)."""
    words = [t for t in rest if not t.startswith("-")]
    try:
        with open(pj(ecwd, "composer.json"), encoding="utf-8") as f:
            scripts = json.load(f).get("scripts") or {}
    except Exception:
        return []
    if not isinstance(scripts, dict) or not words:
        return []
    w0 = words[0]
    if w0 in ("run", "run-script") and len(words) > 1:
        names = [words[1]]
    elif w0 in ("install", "update", "dump-autoload", "dumpautoload"):
        names = ["pre-%s-cmd" % w0, "post-%s-cmd" % w0, "post-autoload-dump"]
    else:
        names = [w0] if w0 in scripts else []
    out = []

    def expand(name, d):
        entry = scripts.get(name)
        for line in ([entry] if isinstance(entry, str) else (entry if isinstance(entry, list) else [])):
            if not isinstance(line, str):
                continue
            if line.startswith("@php "):
                out.append((name, "php " + line[5:]))
            elif line.startswith("@composer "):
                continue
            elif line.startswith("@") and d < 3:
                expand(line[1:].split()[0], d + 1)
            elif not re.match(r"^[\w\\]+::\w+$", line) and not line.startswith("@"):
                out.append((name, line))

    for n in names:
        expand(n, 0)
    return out


def git_hook_files(eff, rest, ecwd):
    """The hook files a git command will run: .git/hooks, core.hooksPath and .husky."""
    sub, gargs = eff.get("sub") or "", eff.get("gargs") or []
    names = list(GIT_HOOKS.get(sub, ()))
    if not names or "--dry-run" in gargs:
        return [], None
    if "--no-verify" in gargs or (sub == "commit" and any(re.match(r"^-[a-zA-Z]*n", a) for a in gargs)):
        names = [n for n in names if n not in ("pre-commit", "commit-msg", "pre-push", "pre-merge-commit")]
    root = resolve(rest[rest.index("-C") + 1], ecwd) if "-C" in rest[:-1] else ecwd
    for _ in range(8):
        if os.path.isdir(pj(root, ".git")):
            break
        parent = posixpath.dirname(root.rstrip("/"))
        if not parent or parent == root:
            return [], None
        root = parent
    else:
        return [], None
    dirs = [pj(root, ".git", "hooks"), pj(root, ".husky")]
    try:
        with open(pj(root, ".git", "config"), encoding="utf-8", errors="replace") as f:
            m = re.search(r"(?mi)^\s*hookspath\s*=\s*(.+?)\s*$", f.read(65536))
        if m:
            dirs.insert(0, resolve(m.group(1), root))
    except Exception:
        pass
    return [pj(d, n) for d in dirs for n in names if os.path.isfile(pj(d, n))], root


def db_files_text(prog, rest, seg, ecwd):
    """The text of the files a database client is told to run: -f FILE, < FILE, .read FILE, \\i FILE."""
    names = []
    for i, t in enumerate(rest):
        if t in ("-f", "--file", "-i", "--queries-file", "--input-file") and i + 1 < len(rest):
            names.append(rest[i + 1])
        m = re.match(r"^--(?:file|queries-file|input-file)=(.+)$", t)
        if m:
            names.append(m.group(1))
        if not t.startswith("-") and t.lower().endswith(SQL_FILE_EXT):
            names.append(t)
    names += re.findall(r"(?:\.read|\\ir?|\\\.|\bsource)\s+([^\s'\";]+)", seg)
    files = [resolve(n, ecwd) for n in names]
    stdin = input_file(seg, ecwd)
    if stdin:
        files.append(stdin)
    out = []
    for p in files:
        body = read_script(p)
        if body:
            out.append(body)
    return "\n".join(out)


def run_effects(eff, toks, rest, seg, ecwd, depth, vars_):
    """Effects of the files and scripts a command sets running besides itself: the file it names as
    its program, the command a runner prefix wraps, a package.json script, a Makefile target, a
    composer script, a local setup.py, the git hooks it fires, a notebook it executes, a PowerShell
    script it is handed. Each is read and judged like a command the agent typed."""
    out = []
    prog, flags, args = eff["prog"], eff["flags"], eff["args"]
    low = re.sub(r"\.exe$", "", prog.lower())

    def named(effs, label):
        for e in effs:
            e["script_body"] = label
            e["quiet"] = True
        return effs

    def opened(path, **kw):
        effs = script_body_effects(path, ecwd, depth, vars_, mode="auto", **kw)
        for e in effs:
            e["quiet"] = True
        return effs

    head = toks[0]
    if ("/" in head or head.startswith("~")) and not code_prog(prog):
        p = resolve(head, ecwd)                              # ./deploy.sh, scripts/lint, /abs/tool
        if is_file(p) and not is_gate_source(p):
            out += opened(p, argv=[tilde(a) for a in plain_words(rest)])
    inner = wrapped_command(low, rest)
    if inner:
        out += analyze(" ".join(shlex.quote(t) for t in inner), ecwd, depth + 1, vars_)
    if low in ("npm", "yarn", "pnpm", "bun"):
        for here, name, script in package_scripts(low, rest, ecwd):
            out += named(analyze(script, here, depth + 1, vars_), "package.json: %s" % name)
    elif low in ("make", "gmake"):
        for here, name, line in make_lines(rest, ecwd):
            out += named(analyze(line, here, depth + 1, vars_), "Makefile: %s" % name)
    elif low == "composer":
        for name, line in composer_scripts(rest, ecwd):
            out += named(analyze(line, ecwd, depth + 1, vars_), "composer.json: %s" % name)
    elif low in ("pwsh", "powershell"):
        for i, t in enumerate(rest):
            tl = t.lower()
            if tl in ("-file", "-f") and i + 1 < len(rest):
                out += opened(resolve(rest[i + 1].replace("\\", "/"), ecwd))
            elif tl in ("-command", "-c") and i + 1 < len(rest):
                out += analyze_powershell(rest[i + 1], ecwd, depth + 1)
            elif not t.startswith("-") and tl.endswith(".ps1") and (i == 0 or rest[i - 1].lower() not in ("-file", "-f")):
                out += opened(resolve(t.replace("\\", "/"), ecwd))
    elif low == "go" and args[:1] == ["run"]:
        for a in args[1:]:
            if a.endswith(".go"):
                out += opened(resolve(a, ecwd), lang="go")
            elif a in (".", "./"):
                for name in sorted(os.listdir(ecwd))[:200]:
                    if name.endswith(".go") and not name.endswith("_test.go"):
                        out += opened(pj(ecwd, name), lang="go")
    elif low in ("jupyter", "papermill", "jupyter-nbconvert", "jupyter-execute", "jupyter-run"):
        runs = low != "jupyter" or args[:1] in (["execute"], ["run"]) or "--execute" in flags
        if runs and (low != "jupyter-nbconvert" or "--execute" in flags):
            nb = next((a for a in args if a.lower().endswith(".ipynb")), None)
            if nb:
                out += opened(resolve(nb, ecwd))
    if "install" in toks and any(_tool(t) in ("pip", "pip3") for t in toks[:toks.index("install")]):
        for a in toks[toks.index("install") + 1:]:
            if not a.startswith("-") and os.path.isfile(pj(resolve(a, ecwd), "setup.py")):
                out += opened(pj(resolve(a, ecwd), "setup.py"), lang="py")
    if eff.get("kind") == "git":
        hooks, root = git_hook_files(eff, rest, ecwd)
        for p in hooks:
            effs = script_body_effects(p, root, depth, vars_, mode="auto")
            out += named(effs, "git hook %s" % os.path.basename(p))
    return out


def analyze(cmd, cwd, depth=0, vars_=None):
    """Walk a shell command tracking cd and simple VAR=value, yielding effects (never executes anything)."""
    effects = []
    if depth > MAX_DEPTH or not cmd or not cmd.strip():
        return effects
    if depth == 0:
        SEEN_FILES.clear()
        GIT_ASKED.clear()
        WRITTEN.clear()
    vars_ = dict(vars_) if vars_ is not None else {"HOME": HOME, "USERPROFILE": HOME}
    state = {"cwd": cwd}
    if INVISIBLE_RE.search(cmd):
        effects.append({"prog": "\u200b", "seg": cmd[:120], "targets": [], "recursive": False,
                        "kind": "invisible-chars", "filtered": False, "unresolved": False, "flags": [],
                        "args": [], "cwd": state["cwd"] or UNKNOWN_CWD, "writes": [], "text": cmd[:120]})
        cmd = INVISIBLE_RE.sub("", cmd)
    cmd = decode_ansi_c_quotes(cmd)
    cmd, bodies = extract_heredocs(cmd)
    # $(( a / b )) and (( i++ )) hold numbers, not a command; a $(...) inside one still runs
    # (found by rulereceipt on claude-code#2544: `echo $(( $(stat -f%z f) / 1048576 ))` read as a delete of /)
    for m in list(ARITH_RE.finditer(cmd)):
        for inner in SUBST_RE.findall(m.group(0)[m.group(0).index("((") + 2:-2]):
            effects += analyze(inner[2:-1] if inner.startswith("$(") else inner[1:-1], state["cwd"], depth + 1, vars_)
    cmd = ARITH_RE.sub(" 0 ", cmd)
    # subshell groups: `(sleep 300; rm -rf ~) &` -- analyze the inside as its own command.
    # Groups inside quotes are data (a regex alternation, a message), never a subshell.
    for m in re.finditer(r"\(([^()]*)\)", mask_quotes(cmd)):
        inner = cmd[m.start(1):m.end(1)].strip()
        if m.start() and cmd[m.start() - 1] == "=":
            continue                                        # `name=(...)`, `name+=(...)`: an array literal
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
    hd, cursor, feeder, saved = 0, 0, None, [False, False]
    for seg in split_segments(cmd):
        nh = len(HEREDOC_RE.findall(seg))
        my_bodies, hd = bodies[hd:hd + nh], hd + nh
        toks = tokenize(expand_vars(seg, vars_))
        while toks and toks[0] in KEYWORDS:
            toks = toks[1:]
        # `f() { ...; }` and `function f { ...; }`: the body is read where it is written
        if len(toks) > 1 and toks[0] == "function":
            toks = toks[2:]
        elif toks and re.match(r"^[A-Za-z_][\w\-]*\(\)$", toks[0]):
            toks = toks[1:]
        elif len(toks) > 1 and toks[1] == "()" and re.match(r"^[A-Za-z_][\w\-]*$", toks[0]):
            toks = toks[1:]
        if toks and toks[0] == "()":
            toks = toks[1:]
        while toks and toks[0] in KEYWORDS:
            toks = toks[1:]
        if not toks or toks[0] in LOOP_HEADS:
            continue
        j = 0
        while j < len(toks) and toks[j] in ("export", "local", "declare", "readonly", "typeset"):
            j += 1
        k = j
        while k < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", toks[k]):
            k += 1
        if k > j and k == len(toks):                       # pure assignment segment
            effects += guarded(carried_by_vars, [], toks[j:k], state["cwd"], depth, vars_)
            for t in toks[j:k]:
                name, val = t.split("=", 1)
                if name.endswith("+"):                      # `name+=value` appends: the value is no longer known
                    vars_.pop(name[:-1], None)
                elif "__SUBST__" in val or "$" in val:
                    vars_.pop(name, None)
                else:
                    vars_[name] = tilde(val)
            continue
        effects += guarded(carried_by_vars, [], strip_leading_redirects(toks), state["cwd"], depth, vars_)
        toks, via_xargs = strip_wrappers(strip_leading_redirects(toks))
        toks = strip_leading_redirects(toks)
        if not toks:
            continue
        prog = os.path.basename(toks[0])
        rest = toks[1:]
        at = cmd.find(seg, cursor)                          # is this segment fed by a pipe from the one before?
        piped = at > 0 and re.search(r"(?<!\|)\|&?\s*$", cmd[:at]) is not None
        cursor = at + len(seg) if at >= 0 else cursor
        fed, feeder = (feeder if piped else None), (prog, rest)
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
        ecwd = state["cwd"]
        eff = {"prog": prog, "seg": seg, "targets": [], "recursive": False, "kind": None, "filtered": False,
               "unresolved": via_xargs, "flags": flags, "args": args, "cwd": ecwd or UNKNOWN_CWD,
               "writes": write_targets(prog, rest, flags, args, expand_vars(seg, vars_), ecwd), "text": seg,
               "toks": toks}
        guarded(note_written, None, prog, rest, my_bodies, eff["writes"])
        handed_text, handed_file = guarded(stdin_text, (None, None), fed, rest, ecwd)

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
            # opaque-script resolution: `bash totally_harmless.sh` executes the FILE body as shell
            sp = resolve_script_arg(rest, ecwd)
            if sp:
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="shell", argv=args_after(rest, None, ecwd))
            fp = input_file(seg, ecwd)                       # `bash < x.sh` runs the file too
            if fp:
                effects += script_body_effects(fp, ecwd, depth, vars_, mode="shell")
            if inner_cmd is None and not sp and not fp and not my_bodies:
                if handed_text:                              # `echo '...' | bash`, `bash <<< '...'`
                    effects += analyze(handed_text, ecwd, depth + 1, vars_)
                elif handed_file:                            # `cat x.sh | bash`
                    effects += script_body_effects(handed_file, ecwd, depth, vars_, mode="shell")
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
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="shell", argv=args_after(rest, None, ecwd))
            continue

        if prog in ("rm", "rmdir", "unlink", "shred", "trash", "rip", "srm") or \
                (prog == "gio" and rest[:1] == ["trash"]):
            eff["kind"] = "delete"
            eff["recursive"] = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags
            eff["targets"] = [t for t in resolve_targets([a for a in args if not (prog == "gio" and a == "trash")], ecwd)]
            if via_xargs and fed:                           # `echo ~/x | xargs rm -rf`, `find ~/x | xargs rm -rf`
                fprog, frest = fed
                fflags, fargs = flags_and_args(frest)
                given = None
                if fprog in ("echo", "printf"):
                    given = [a for a in fargs if "__SUBST__" not in a and "$" not in a]
                elif fprog == "find":
                    given = []
                    for a in frest:
                        if a.startswith("-") or a in ("(", ")", "!"):
                            break
                        given.append(a)
                    eff["filtered"] = any(a in ("-name", "-iname", "-path", "-regex", "-mtime", "-mmin", "-newer",
                                                "-size", "-empty") for a in frest)
                if given:
                    eff["targets"] = resolve_targets(given, ecwd) + [t for t in eff["targets"] if not t.endswith("/{}")]
                    eff["unresolved"] = False
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
                                            "-mmin", "-newer", "-size", "-user", "-perm", "-empty")
                                      for a in rest)                 # `-type f` alone still takes every file
                eff["targets"] = resolve_targets(roots or ["."], ecwd)
        elif prog == "rsync" and any(f == "--del" or f.startswith("--delete") for f in flags) and len(args) > 1:
            # --delete removes from the destination whatever the source does not hold
            if not re.match(r"^[\w.\-@]+:", args[-1]):
                eff["kind"] = "delete"
                eff["recursive"] = True
                eff["targets"] = [resolve(args[-1], ecwd)]
        elif prog == "rsync" and "--remove-source-files" in flags:
            eff["kind"] = "move"
            eff["targets"] = [resolve(a, ecwd) for a in args[:-1]]
        elif prog == "git":
            eff["kind"] = "git"
            g = list(rest)
            while g and g[0].startswith("-"):
                g = g[2:] if g[0] in ("-C", "-c", "--git-dir", "--work-tree") else g[1:]
            eff["sub"] = g[0] if g else ""
            eff["gargs"] = g[1:]
            for line in guarded(carried_by_git, [], rest, eff["sub"], eff["gargs"]):
                effects += analyze(line, ecwd, depth + 1, vars_)
            if "-C" in rest[:-1] and ecwd:
                eff["git_dir"] = resolve(rest[rest.index("-C") + 1], ecwd)
            eff["saved"] = tuple(saved)                     # what earlier steps of this command put away
            ga = eff["gargs"]
            if eff["sub"] == "stash" and (not ga or ga[0] in ("push", "save") or ga[0].startswith("-")):
                saved[0] = True
                saved[1] = saved[1] or any(a in ("-u", "--include-untracked", "-a", "--all") for a in ga)
            elif eff["sub"] == "commit" and any(a == "--all" or re.match(r"^-[a-zA-Z]*a", a) for a in ga):
                saved[0] = True
        elif code_prog(prog):
            lang = code_prog(prog)
            eff["kind"] = "inline"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
            sp = script_file_arg(lang, rest, ecwd) or input_file(seg, ecwd)
            given = inline_codes(lang, rest) + list(my_bodies)              # `python -c CODE`, a heredoc
            if not sp and not given and lang != "awk":
                if handed_text:                              # `echo CODE | python3`, `python3 <<< CODE`
                    given = [handed_text]
                elif handed_file:                            # `cat x.py | python3`
                    sp = handed_file
            if given:
                eff["codes"] = [(code, lang) for code in given]
            for code in given:
                effects += guarded(code_effects, [], code, lang, ecwd, ecwd, depth + 1, vars_, None)
            if sp and is_gate_source(sp):
                # `python3 .../operator-gate.py <subcommand>` is the gate itself. Judge it as the
                # operator-gate program (approve, install, onboard stay owner-only); never walk its source.
                eff["prog"] = "operator-gate"
                eff["kind"] = "cli"
                eff["args"] = [a for a in args if a != "-" and not a.endswith("operator-gate.py")]
                eff["text"] = seg
            elif sp:
                effects += script_body_effects(sp, ecwd, depth, vars_, mode="auto", prog=prog, seg=seg, lang=lang,
                                               argv=args_after(rest, sp, ecwd))
        elif prog in ("grep", "egrep", "fgrep", "rg", "ag", "ack"):
            # `... | grep -v x` filters text already on the pipe; `grep -rn x src` and `rg x` read files.
            operands = args[1:]
            recursive = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags or prog in ("rg", "ag", "ack")
            eff["kind"] = "filter" if not operands and not recursive else "read"
        elif prog in DB_CLIENTS:
            eff["kind"] = "db"
            if my_bodies:
                eff["text"] = seg + "\n" + "\n".join(my_bodies)
            handed = guarded(db_files_text, "", prog, rest, seg, ecwd) if ecwd else ""   # `psql -f wipe.sql`, `< wipe.sql`
            if handed:
                eff["text"] += "\n" + handed
        elif prog in ("fly", "flyctl", "stripe", "vercel", "wrangler", "netlify", "gh", "npm", "pnpm", "yarn",
                      "cargo", "twine", "docker", "vestige", "vestige-mcp", "operator-gate"):
            eff["kind"] = "cli"
        if ecwd is not None and depth < MAX_DEPTH:           # the files and scripts this command sets running
            effects += guarded(run_effects, [], eff, toks, rest, seg, ecwd, depth, vars_)
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


def ps_as_shell(stmt):
    """A PowerShell statement as the shell walker reads it: the call operator dropped and a program given
    by its Windows path named plainly, so `& C:\\msys64\\usr\\bin\\bash.exe -lc "..."` is read as `bash -lc "..."`
    and what it hands to bash is walked like any nested shell."""
    s = re.sub(r"^\s*(?:&\s*|\.\s+)", "", stmt)
    m = re.match(r"\s*(\"[^\"]*\"|'[^']*'|\S+)", s)
    if not m:
        return s
    prog = m.group(1).strip("\"'")
    if "\\" not in prog and not prog.lower().endswith(".exe"):
        return s
    return re.sub(r"\.exe$", "", prog.replace("\\", "/").rsplit("/", 1)[-1], flags=re.I) + s[m.end():]


def analyze_powershell(cmd, cwd, depth=0):
    """Effects of a PowerShell command. Remove-Item and its aliases become delete effects; every other
    statement goes through the shell walker, which reads git, npm and the like the same way."""
    effects = []
    if depth > MAX_DEPTH:
        return effects
    if depth == 0:
        SEEN_FILES.clear()
        WRITTEN.clear()
    for stmt in ps_statements(cmd or ""):
        toks = ps_tokens(ps_expand(stmt))
        while toks and toks[0] in ("&", "."):                # call operators
            toks = toks[1:]
        if not toks:
            continue
        head = toks[0].strip("\"'")
        if cwd and re.search(r"\.(?:ps1|bat|cmd)$", head, re.I):     # .\cleanup.ps1 runs the file
            effects += script_body_effects(resolve(head.replace("\\", "/"), cwd), cwd, depth, None, mode="auto")
            continue
        handed = cmd_exe_line(toks)
        if handed is not None:                              # `cmd /c rmdir /s /q C:\\x` from PowerShell
            effects += cmd_exe_effects(handed, cwd)
            continue
        if toks[0].lower() not in PS_DELETE:
            nested = analyze(ps_as_shell(stmt), cwd, depth)
            if re.search(r'"[^"]*\$', ps_expand(stmt)):
                # Inside double quotes PowerShell fills in $X itself before the nested shell starts, and a
                # backslash does not protect it: `bash -lc "rm -rf \$X"` hands bash `rm -rf \`.
                for e in nested:
                    if e.get("kind") == "delete" and e.get("recursive") and e.get("unresolved") and not e.get("targets"):
                        e["ps_filled"] = True
            effects += nested
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


def lost_work(targets):
    """Why deleting these paths loses work for good, as a phrase, or None. Inside a repository that
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
    return _named(lost) if lost else None


def git_discards(sub, gargs, root, saved=(False, False)):
    """Why a git command throws away work nothing can bring back, or None: reset --hard, checkout or
    restore over changed files, a forced switch, clean -f. Read from what the repository holds now."""
    if not root or not os.path.isdir(root) or not git_top(root):
        return None
    flags = [a for a in gargs if a.startswith("-") and a != "--"]
    words = [a for a in gargs if not a.startswith("-")]
    now = time.time()

    def changed(paths, worktree_only):
        entries = _status(root, ["--untracked-files=no"], paths) or []
        return [rel for state, rel in entries if (state[1] != " " if worktree_only else state.strip())]

    if sub == "reset" and ("--hard" in flags or "--merge" in flags) and not saved[0]:
        names = changed([], False)
        if names:
            return "git reset %s discards uncommitted changes: %s" % (flags[0], _named(names))
    forced = any(f in ("-f", "--force", "--discard-changes") for f in flags)
    paths = None
    if sub == "restore" and not ("--staged" in flags and "--worktree" not in flags and "-W" not in flags
                                 and not any(re.match(r"^-[A-Za-z]*W", f) for f in flags)):
        paths = words
    elif sub == "checkout":
        if "--" in gargs:
            paths = gargs[gargs.index("--") + 1:]
        elif forced:
            paths = ["."]
        else:
            paths = [w for w in words if w == "." or os.path.lexists(pj(root, w))]
    elif sub == "switch" and forced:
        paths = ["."]
    if paths and not saved[0]:
        names = changed(paths, True)
        if names:
            return "git %s discards uncommitted changes: %s" % (sub, _named(names))
    if sub == "stash" and gargs[:1] in (["drop"], ["clear"]):
        held = [ln for ln in (git_out(root, ["stash", "list"]) or "").splitlines() if ln.strip()]
        if held:
            return "git stash %s throws away stashed work nothing else holds (%d stash%s kept now)" % (
                gargs[0], len(held), "" if len(held) == 1 else "es")
        if saved[0] and changed([], False):
            return "git stash %s throws away the changes this command stashes first" % gargs[0]
    if sub == "clean" and not saved[1] and any(f == "--force" or re.match(r"^-[a-zA-Z]*f", f) for f in flags) \
            and not any(f == "--dry-run" or re.match(r"^-[a-zA-Z]*n", f) for f in flags):
        keep = []
        for f in flags:
            if f == "--force":
                continue
            f = ("-" + f[1:].replace("f", "")) if not f.startswith("--") else f
            if f != "-":
                keep.append(f)
        lost = []
        for extra, only_precious in (([], False), (["-x"], True)) if any("x" in f.lower() for f in keep) else (([], False),):
            base = [f for f in keep if not re.match(r"^-[a-zA-Z]*[xX]", f)] + ["-d"] * (not only_precious and any("d" in f for f in keep))
            out = git_out(root, ["clean", "-n"] + (keep if only_precious else base) + ["--"] + words)
            for line in (out or "").splitlines():
                rel = line[len("Would remove "):].strip() if line.startswith("Would remove ") else ""
                if not rel or any(part in BUILD_DIRS for part in rel.rstrip("/").split("/")):
                    continue
                for f in _files_under(pj(root, rel.rstrip("/")), 2000):
                    if _old(f, now) and (PRECIOUS_RE.search(f) if only_precious else True) and f not in lost:
                        lost.append(f)
        if lost:
            return "git clean deletes untracked work: %s" % _named([f.replace(root + "/", "") for f in lost])
    return None


# --------------------------------------------------------------------------- #
# infrastructure torn down in one command (OP-006)
# --------------------------------------------------------------------------- #
INFRA_HINT = "It tears down running infrastructure or stored data. Ask the owner for a one-time permit."
K8S_HEAVY = {"namespace", "namespaces", "ns", "pv", "pvc", "persistentvolume", "persistentvolumes",
             "persistentvolumeclaim", "persistentvolumeclaims", "crd", "crds", "customresourcedefinition", "node", "nodes"}
INFRA_FORMS = (
    ({"terraform", "tofu", "terragrunt"}, lambda t, f: "destroy" in t[:2] or ("apply" in t[:2] and "-destroy" in f), "terraform"),
    ({"pulumi"}, lambda t, f: t[:1] == ["destroy"] or _has_seq(t, "stack", "rm"), "pulumi"),
    ({"kubectl", "oc"}, lambda t, f: t[:1] == ["delete"] and (bool(K8S_HEAVY & set(x.split("/")[0] for x in t[1:]))
                                                               or "--all" in f or "--all-namespaces" in f), "kubectl"),
    ({"helm"}, lambda t, f: t[:1] in (["uninstall"], ["delete"], ["del"]), "helm"),
    ({"aws"}, lambda t, f: _has_seq(t, "s3", "rb") or (_has_seq(t, "s3", "rm") and "--recursive" in f) or
     any(x in t for x in ("delete-bucket", "terminate-instances", "delete-stack", "delete-cluster", "delete-function",
                          "delete-file-system", "delete-volume")), "aws"),
    ({"gsutil"}, lambda t, f: t[:1] in (["rm"], ["rb"]) and (bool({"-r", "-R", "-a"} & f) or t[:1] == ["rb"]), "gsutil"),
    ({"gcloud"}, lambda t, f: "delete" in t and any(x in t for x in ("projects", "instances", "clusters", "buckets", "disks"))
     and "sql" not in t or (_has_seq(t, "storage", "rm") and bool({"-r", "--recursive"} & f)), "gcloud"),
    ({"az"}, lambda t, f: "delete" in t and any(x in t for x in ("group", "vm", "aks", "account", "webapp", "disk")), "az"),
    ({"cdk"}, lambda t, f: t[:1] == ["destroy"], "cdk"),
    ({"sst"}, lambda t, f: t[:1] == ["remove"], "sst"),
    ({"serverless", "sls"}, lambda t, f: t[:1] == ["remove"], "serverless"),
    ({"sam"}, lambda t, f: t[:1] == ["delete"], "sam"),
    ({"heroku"}, lambda t, f: any(x in t for x in ("apps:destroy", "destroy", "addons:destroy")), "heroku"),
    ({"vercel"}, lambda t, f: t[:1] in (["remove"], ["rm"]) or _has_seq(t, "project", "rm"), "vercel"),
    ({"netlify"}, lambda t, f: "sites:delete" in t, "netlify"),
    ({"railway"}, lambda t, f: t[:1] in (["down"], ["delete"]), "railway"),
    ({"doctl"}, lambda t, f: "delete" in t and bool({"-f", "--force"} & f), "doctl"),
)


# --------------------------------------------------------------------------- #
# database resets: one framework or admin command that empties or drops a database (OP-007)
# --------------------------------------------------------------------------- #
DB_RESET_HINT = ("It empties or drops the database it is pointed at. Check which environment it reaches and "
                 "take a backup, then ask the owner for a one-time permit.")
# A task name handed to a program that only prints, searches or records text is data, not a run.
DB_TEXT_ONLY = {"echo", "printf", "cat", "grep", "egrep", "fgrep", "rg", "ag", "ack", "sed", "awk", "head", "tail",
                "less", "more", "wc", "ls", "find", "fd", "git", "gh", "man", "tldr", "which", "type", "whereis", "code"}
# Programs that run their arguments as a command somewhere else: a quoted command inside is read word by word.
DB_CARRIERS = {"ssh", "su", "docker", "docker-compose", "podman", "nerdctl", "kubectl", "oc", "heroku", "fly",
               "flyctl", "railway", "kamal", "dokku", "vagrant", "wsl", "ddev", "lando", "sail"}
# Asking for help or a rehearsal runs nothing.
DB_DRY_FLAGS = {"--help", "--dry-run", "--dryrun", "--pretend", "--dump", "--dump-sql"}
# Task names that reset a database whatever launches them: Laravel, Rails and rake, Sequelize, Doctrine,
# TypeORM, MikroORM, Ecto, Heroku. Rails names a database after the task (`db:drop:primary`).
DB_TASK_RE = re.compile(
    r"^(?:migrate:(?:fresh|refresh|reset)|db:wipe|db:(?:drop|reset|purge|truncate_all)(?::[\w.\-]+)?|"
    r"db:migrate:reset(?::[\w.\-]+)?|db:(?:schema|structure):load(?::[\w.\-]+)?|db:seed:replant|"
    r"db:migrate:undo:all|doctrine:(?:database|schema):drop|d:[ds]:d|schema:(?:drop|fresh)|migration:fresh|"
    r"ecto\.(?:drop|reset)|pg:reset)$", re.I)
# A command that names the test environment resets the test database, which is what a test run is for.
DB_TEST_ENV_RE = re.compile(
    r"(?:^|[\s\"';&|(])(?:RAILS_ENV|RACK_ENV|APP_ENV|NODE_ENV|MIX_ENV|DJANGO_ENV)=(?:test|testing)\b|"
    r"--env(?:ironment)?[= ](?:test|testing)\b|(?:^|\s)-e\s+test(?:ing)?(?:\s|$)")


def _tool(word):
    """A command word as a tool name: no directory, no Windows suffix, no @version, lower case."""
    t = word.replace("\\", "/").rsplit("/", 1)[-1].lower().lstrip(".")
    t = re.sub(r"\.(?:exe|cmd|bat|ps1|phar)$", "", t)
    at = t.find("@", 1)
    return t[:at] if at > 0 else t


def _has_seq(words, *seq):
    return any(tuple(words[i:i + len(seq)]) == seq for i in range(len(words) - len(seq) + 1))


# (tools, modules run as `-m <module>`, test on the plain words after the tool and the flags, name)
DB_RESET_FORMS = (
    ({"manage.py", "django-admin", "django-admin.py"}, ("django",),
     lambda t, f: t[:1] in (["flush"], ["reset_db"]) or t[:1] == ["migrate"] and "zero" in t[1:], "django"),
    ({"prisma"}, (),
     lambda t, f: _has_seq(t, "migrate", "reset") or
     _has_seq(t, "db", "push") and bool({"--force-reset", "--accept-data-loss"} & f), "prisma"),
    ({"drizzle-kit"}, (), lambda t, f: t[:1] == ["push"] and "--force" in f, "drizzle-kit"),
    ({"alembic"}, ("alembic",), lambda t, f: _has_seq(t, "downgrade", "base"), "alembic"),
    ({"flask"}, ("flask",), lambda t, f: _has_seq(t, "db", "downgrade", "base"), "flask"),
    ({"dotnet", "dotnet-ef"}, (),
     lambda t, f: _has_seq(t, "database", "drop") or _has_seq(t, "database", "update", "0"), "dotnet ef"),
    ({"dropdb"}, (), lambda t, f: bool(t), "dropdb"),
    ({"mysqladmin", "mariadb-admin"}, (), lambda t, f: "drop" in t, "mysqladmin"),
    ({"turso"}, (), lambda t, f: _has_seq(t, "db", "destroy"), "turso"),
    ({"pscale"}, (), lambda t, f: _has_seq(t, "database", "delete") or _has_seq(t, "branch", "delete"), "pscale"),
    ({"neon", "neonctl"}, (),
     lambda t, f: any(_has_seq(t, a, b) for a, b in (("projects", "delete"), ("databases", "delete"),
                                                     ("branches", "delete"), ("branches", "reset"))), "neon"),
    ({"gcloud"}, (),
     lambda t, f: "sql" in t and "delete" in t and ("instances" in t or "databases" in t), "gcloud"),
    ({"aws"}, (),
     lambda t, f: any(x in t for x in ("delete-db-instance", "delete-db-cluster", "delete-table")), "aws"),
    ({"firebase"}, (), lambda t, f: "firestore:delete" in t or "database:remove" in t, "firebase"),
    ({"bq"}, (), lambda t, f: t[:1] == ["rm"], "bq"),
    # the volumes a containerised database lives on
    ({"docker", "docker-compose", "podman", "podman-compose", "nerdctl"}, (),
     lambda t, f: ("down" in t and bool({"-v", "--volumes"} & f)) or _has_seq(t, "volume", "rm") or
     _has_seq(t, "volume", "prune") or (_has_seq(t, "system", "prune") and "--volumes" in f), "container volumes:"),
)


def _run_words(toks, seg=""):
    """The words of a command as its program will see them, with the tool each word names and the
    flags, or None when the command only prints, searches or rehearses. A quoted command handed to
    ssh, docker, kubectl and the like is read word by word."""
    if not toks:
        return None
    head = _tool(toks[0])
    if head in DB_TEXT_ONLY:
        return None
    words, several = list(toks), False
    if head in DB_CARRIERS:                                  # `ssh host "cd app && bin/rails db:drop"`
        several = any(re.search(r"[;|&]", t) for t in toks)
        words = [w for t in toks for w in re.split(r"[\s;|&]+", t) if w]
    flags = {w.split("=", 1)[0].lower() for w in words if w.startswith("-")}
    if flags & DB_DRY_FLAGS:
        return None
    return words, [_tool(w) for w in words], flags, several


def _form_hit(forms, words, names, flags):
    for entry in forms:
        tools, modules, test, name = entry if len(entry) == 4 else (entry[0], (), entry[1], entry[2])
        for i, n in enumerate(names):
            if n in tools or (n in modules and i and words[i - 1] == "-m"):
                after = [w.lower() for w in words[i + 1:] if not w.startswith("-")]
                if test(after, flags):
                    return "%s %s" % (name, " ".join(after[:3]))
                break
    return None


def infra_destroy(toks, seg=""):
    """What a command tears down when it destroys infrastructure in one go, as a phrase, or None."""
    read = _run_words(toks, seg)
    if not read:
        return None
    words, names, flags, _ = read
    hit = _form_hit(INFRA_FORMS, words, names, flags)
    return ("infrastructure teardown: %s" % hit) if hit else None


def db_reset(toks, seg=""):
    """What a command does when it empties or drops a database, as a short phrase, or None.
    Reads the command's words and runs nothing. Covers the reset tasks of the common frameworks and
    migration tools, the admin commands that drop a database, the hosted-database delete commands,
    and the container commands that delete the volumes a database lives on."""
    read = _run_words(toks, seg)
    if not read:
        return None
    words, names, flags, several = read
    if not several and DB_TEST_ENV_RE.search(seg or " ".join(toks)):
        return None
    plain = [w.lower() for w in words[1:] if not w.startswith("-")]
    for i, w in enumerate(words[1:], 1):
        if w.startswith("-") or not DB_TASK_RE.match(w) or words[i - 1].lower() == "help":
            continue
        if w.lower().startswith(("doctrine:", "d:")) and not ({"--force", "-f"} & flags):
            continue                                         # without --force Doctrine prints and stops
        return "database reset: %s" % w
    if "migrate:rollback" in plain and "--all" in flags:
        return "database reset: migrate:rollback --all"
    if "ecto.rollback" in plain and "--all" in flags:
        return "database reset: ecto.rollback --all"
    if ("doctrine:fixtures:load" in plain or "d:f:l" in plain) and "--append" not in flags:
        return "database reset: doctrine:fixtures:load purges before it loads"
    if "db:migrate" in plain and re.search(r"(?:^|\s)VERSION=0(?:\s|$)", seg or " ".join(toks)):
        return "database reset: db:migrate VERSION=0"
    hit = _form_hit(DB_RESET_FORMS, words, names, flags)
    return ("database reset: %s" % hit) if hit else None


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

    if kind == "finding":                                    # read out of a file's code; see code_effects
        hits.append(e["finding"])

    prog0, text0 = e.get("prog", ""), e.get("text", "")
    reset = guarded(db_reset, None, e.get("toks") or [], e.get("seg") or "")
    if reset:
        hits.append(("OP-007", reset, DB_RESET_HINT))
    torn = guarded(infra_destroy, None, e.get("toks") or [], e.get("seg") or "")
    if torn:
        hits.append(("OP-006", torn, INFRA_HINT))
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
                any(t.startswith("/Library/LaunchAgents") or t.startswith(pj(HOME, "Library", "LaunchAgents"))
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
            broad = ecwd in (None, "/") or same_path(ecwd, HOME) or any(
                same_path(ecwd, pj(HOME, d)) for d in ("Developer", "Documents", "Downloads", "Desktop"))
            wide = [t for t in unsafe if t == "/" or same_path(t, HOME) or t.count("/") <= 2]
            outside = [t for t in unsafe if broad or not is_inside(t, ecwd)]
            if (wide or outside) and not e["unresolved"] and \
                    not any(h[0] in ("OP-001", "OP-000", "OP-002") for h in hits):
                hits.append(("OP-003", "recursive delete of %s" % (wide or outside)[0],
                             "Move it to a dated folder under /tmp, or delete only build outputs "
                             "(target/, node_modules/, dist/)."))
        if kind == "delete" and e.get("filtered") and not e["unresolved"] and \
                not any(h[0] in ("OP-001", "OP-000", "OP-002", "OP-003") for h in hits):
            broad = ecwd in (None, "/") or same_path(ecwd, HOME) or any(
                same_path(ecwd, pj(HOME, d)) for d in ("Developer", "Documents", "Downloads", "Desktop"))
            away = [t for t in e["targets"] if not is_scratch(t) and (broad or not is_inside(t, ecwd))]
            if away:                                        # `find ~/Documents -name '*.pdf' -delete`
                hits.append(("OP-003", "filtered delete under %s, outside the working directory" % away[0],
                             "Name the files, or run it inside the project; ask the owner for anything wider."))
        if kind == "delete" and not e["unresolved"] and not e.get("filtered") and \
                e.get("prog") not in ("trash", "gio", "rip") and \
                not any(h[0] in ("OP-000", "OP-001", "OP-002", "OP-003", "OP-004") for h in hits):
            # without -r a folder is only removed when it is empty, so nothing in it can be lost
            held = [t for t in e["targets"] if not is_scratch(t) and (e["recursive"] or not os.path.isdir(t))]
            repo = next((t for t in held if e["recursive"] and os.path.isdir(t) and os.path.exists(pj(t, ".git"))
                         and _old(pj(t, ".git"), time.time())), None)    # a clone made this hour is the session's own
            if repo:                                        # `rm -rf .` in a project, `rm -rf ./vendored-repo`
                hits.append(("OP-004", "delete of the repository %s: its history goes with it" % repo,
                             "Do not remove a repository; ask the owner."))
            else:
                why = guarded(lost_work, None, held)
                if why:
                    hits.append(("OP-011", "delete of %s loses %s" % (held[0] if len(held) == 1 else "%d paths" % len(held), why),
                                 LOSS_HINT))
        if e["unresolved"] and kind in ("delete", "move") and not e["targets"]:
            if e.get("ps_filled"):
                hits.append(("OP-003", "recursive delete whose target PowerShell fills in before the shell runs",
                             "Inside double quotes PowerShell replaces $X itself, and a backslash does not stop it. "
                             "Use single quotes, or name the path as a literal."))
            else:
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
        if sub == "push":
            shared = re.compile(r"^(?:refs/heads/)?(?:main|master|trunk|release[/\-].*|prod.*)$")
            named = [a for a in ga if not a.startswith("-")][1:]
            gone = [r[1:] for r in named if r.startswith(":")]              # `git push origin :main`
            if any(a in ("--delete", "-d") for a in ga):
                gone += named
            lost_ref = next((r for r in gone if shared.match(r)), None)
            if lost_ref:
                hits.append(("OP-004", "delete of remote %s" % lost_ref.replace("refs/heads/", ""), "Ask the owner."))
            if "--mirror" in ga:
                hits.append(("OP-004", "push --mirror overwrites and deletes remote branches to match this clone",
                             "Push the branches you mean by name; ask the owner."))
        why = guarded(git_discards, None, sub, ga, e.get("git_dir") or ecwd or cwd, e.get("saved") or (False, False))
        if why:
            hits.append(("OP-011", why, LOSS_HINT))

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
        if prog in DB_CLIENTS or prog == "wrangler" and "d1" in args:
            sql = re.sub(r"pragma\s+wal_checkpoint\s*(\(\s*\w+\s*\))?", "", lo)
            sql = re.sub(r"insert\s+into\s+(\w+)\s*\(\s*\1\s*\)\s*values\s*\(\s*'[a-z\-]+'\s*\)", "", sql)  # fts5 control
            if re.search(r"\bdrop\s+(table|database|schema|index)\b|(?<![\w(])truncate\s+(table\s+)?[\w\"`]|"
                         r"\bflush(all|db)\b|\bdropdatabase\b|\.drop\s*\(\s*\)|"
                         r"\.(?:deletemany|remove)\s*\(\s*\{\s*\}\s*\)", sql) \
                    or re.search(r"\bdelete\s+from\s+\w+\s*(;|\"|'|$)", sql):
                hits.append(("OP-007", "destructive SQL", "Run it as a SELECT first, scope with WHERE, "
                             "and take a backup; ask the owner for a permit."))
            if prog == "sqlite3":
                dbs = [a for a in args if re.search(r"\.(db|sqlite3?|db3)$", a)] or args[:1]
                live = [d for d in dbs if any(is_inside(resolve(d, ecwd), m) for m in MEMORY_DIRS)]
                if live and re.search(r"\b(delete|drop|update|insert|alter|vacuum|replace|truncate)\b", sql):
                    hits.append(("OP-002", "write against the live Vestige store via sqlite3",
                                 "Work on a copy (`vestige backup`, then edit the copy); read-only queries are fine."))
        for code, clang in (e.get("codes") or [(text, None)]) if e["kind"] == "inline" else ():
            # code the gate could lift out is read with its strings and comments set apart; text it
            # could not take apart (a here-string, a pipe) is read whole, as before
            found, unresolved, mentions_gate = inline_file_calls(
                code, ecwd, roots, gate_files, code_mask(code, clang) if clang else None)
            if found:
                hits += [(rid, d if d.startswith("inline code") else "inline code: " + d, h) for rid, d, h in found]
            elif unresolved and mentions_gate:
                # a write or delete whose target the gate cannot resolve, in code that names a gate path:
                # fail closed, as before 0.3.5
                hits.append(("OP-000", "inline code has a %s whose target the gate cannot resolve, and names a gate path" % unresolved,
                             "Name the target as a plain literal so the gate can judge it, or ask the owner."))
            elif unresolved == "delete":
                # an unresolved write in code that names no gate path is ordinary scripting, not a finding
                hits.append(("OP-S05", "inline code has a delete whose target the gate cannot resolve", ""))
    if e.get("quiet"):
        hits = [h for h in hits if h[0] != "OP-S05"]
    if hits and e.get("script_body"):
        hits = [(r, "%s -- inside script %s" % (d, e["script_body"]), a) for (r, d, a) in hits]
    return hits


INLINE_WRITE_CALL_RE = re.compile(
    r"\bopen\s*\(|\.write_text\s*\(|\.write_bytes\s*\(|\bwriteFile(?:Sync)?\s*\(|\bappendFile(?:Sync)?\s*\(|"
    r"\bcopyfile\s*\(|shutil\.copy2?\s*\(|fs\.(?:write|append|copy)\w*\s*\(|File\.write\s*\(|\btofile\s*\(")
INLINE_DELETE_CALL_RE = re.compile(
    r"\brmtree\s*\(|shutil\.move\s*\(|os\.(?:remove|unlink|rmdir|removedirs|rename|replace)\s*\(|\brmSync\s*\(|"
    r"\brimraf\s*\(|\bunlinkSync\s*\(|fs\.(?:rm|unlink|rename)\w*\s*\(|File\.delete\s*\(|FileUtils\.(?:rm|mv)\w*\s*\(|"
    r"\bremove_tree\s*\(|\bunlink\s*\(")
INLINE_MODE_RE = re.compile(r"^(?:mode\s*=\s*)?['\"][wax][+bt]*['\"]$")
INLINE_LITERAL_RE = re.compile(r"^(?:os\.path\.expanduser\(|Path\(|pathlib\.Path\()?\s*(['\"])(.*?)\1\s*\)?$")
INLINE_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*((?:os\.path\.expanduser\(|Path\(|pathlib\.Path\()?\s*(['\"]).*?\3\s*\)?)\s*$", re.M)


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


def _literal_path(expr, literals):
    expr = (expr or "").strip()
    m = INLINE_LITERAL_RE.match(expr)
    if m:
        return m.group(2)
    if re.match(r"^[A-Za-z_]\w*$", expr) and expr in literals:
        return literals[expr]
    v = code_str(expr, literals)                             # os.path.join(home, "x"), Path.home() / "x"
    return v if (v and UNKNOWN_WORD not in v) else None


def inline_file_calls(text, ecwd, roots, gate_files, mask=None):
    """Judge each file-writing or file-deleting call in inline code by its own path operand.
    Returns (hits, unresolved_kind_or_None, mentions_gate_path). With a mask, a call is only a call
    where the code is: the same words inside a string or a comment are not read."""
    literals = code_literals(text, None, mask) if mask is not None else {}
    for m in INLINE_ASSIGN_RE.finditer(text):
        lit = _literal_path(m.group(2), {})
        if lit is not None and m.group(1) not in literals and in_code(mask, text, m.start(1)):
            literals[m.group(1)] = lit
    found, unresolved = [], None
    for kind, call_re in (("write", INLINE_WRITE_CALL_RE), ("delete", INLINE_DELETE_CALL_RE)):
        for m in call_re.finditer(text):
            if not in_code(mask, text, m.start()):
                continue
            open_idx = text.index("(", m.start())
            args = _balanced_args(text, open_idx)
            name = text[m.start():open_idx].strip()
            if args is None:
                unresolved = "delete" if kind == "delete" else (unresolved or kind)
                continue
            operands = []
            if name.startswith(".write_"):
                recv = re.search(r"([A-Za-z_]\w*|(?:pathlib\.)?Path\(\s*(['\"]).*?\2\s*\))\s*$", text[:m.start()])
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
                lit = _literal_path(op, literals)
                if lit is None:
                    unresolved = "delete" if kind == "delete" else (unresolved or kind)
                    continue
                target = resolve(lit, ecwd)
                if kind == "write":
                    found += write_hits(target, gate_files, "inline code writes to")
                else:
                    found += target_hits("delete", target, True, roots, gate_files)
    mentions_gate = any(touches_gate_home(resolve(q, ecwd)) or any(is_inside(resolve(q, ecwd), f) for f in gate_files)
                        for q in re.findall(r"(?:~|\$HOME|\$\{HOME\}|/)[^\s'\"),;\]]*", text))
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
        for eff in (analyze_powershell(cmd, cwd) if tool in POWERSHELL_TOOLS else analyze(cmd, cwd)):
            effects_out.append(eff)
            hits += classify_effect(eff, cfg, cwd)
        if re.search(r"operator-gate(\.py)?['\"]?\s+(approve|mode|install|uninstall|upgrade\s+--install)\b", cmd):
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


def write_receipt(rec):
    """Append a hash-chained receipt. Never raises into the caller."""
    try:
        ensure_dirs()
        rdir = pj(OP_HOME, "receipts")
        head_path = pj(rdir, "HEAD")
        lock = open(pj(rdir, ".lock"), "a+", encoding="utf-8")
        lock_file(lock)
        try:
            try:
                with open(head_path, encoding="utf-8") as f:
                    prev = f.read().strip() or ("0" * 64)
            except Exception:
                prev = "0" * 64
            rec = dict(rec)
            rec.update({"v": 1, "integrity": INTEGRITY, "gate_version": VERSION, "prev": prev,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"})
            rec["digest"] = digest_of(rec)
            day = time.strftime("%Y-%m-%d", time.gmtime())
            with open(pj(rdir, day + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            with open(head_path, "w", encoding="utf-8") as f:
                f.write(rec["digest"])
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
def stop(reason_lines):
    sys.stderr.write("\n".join(reason_lines) + "\n")
    sys.stderr.flush()
    sys.exit(2)


def degraded_hit(raw):
    """Gate crashed: fail closed only for text that plainly looks destructive, else fail open."""
    return bool(re.search(r"\brm\s+-[a-zA-Z]*[rR]|\bmv\s+\S*(vestige|Developer|\.zcode|\.claude)|"
                          r"push\s+.*--force|\bDROP\s+TABLE|\.vestige/vestige\.db|fly\s+deploy|OPERATOR_CANARY_STOP_7f3a",
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
        if os.path.exists(pj(OP_HOME, "DISABLED")) or global_mode() == "off":
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
                    "OP-003", "OP-011", "OP-005", "OP-006", "OP-007",
                    "OP-S06", "OP-S17", "OP-S12", "OP-S13", "OP-S07", "OP-S08", "OP-S11", "OP-S01",
                    "OP-S02", "OP-S03", "OP-S04", "OP-S05", "OP-S14", "OP-S15", "OP-S16"]

        def hit_depth(detail):
            best = 0
            for m in re.finditer(r"/[^\s\"']+", detail or ""):
                best = max(best, m.group(0).count("/"))
            return best

        enforce_hits.sort(key=lambda h: (priority.index(h[0]) if h[0] in priority else len(priority),
                                         -hit_depth(h[3]), h[0]))
        soft = {h[0] for h in enforce_hits} <= {"OP-003", "OP-005", "OP-006", "OP-007", "OP-011"}
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
                      "Only the owner can allow it once, from their own terminal: %s approve %s" % (self_cmd(), adigest),
                      "Receipt: %s" % ((rd or "unavailable")[:16])]
            stop(lines)
        if shadow_hits:
            write_receipt(dict(base_rec, decision="SHADOW_STOP", commitments=[h[0] for h in shadow_hits],
                               detail=redact("; ".join(h[3] for h in shadow_hits))))
        else:
            write_receipt(dict(base_rec, decision="PASS", commitments=[]))    # no rule matched

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
    recent = ""
    try:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with open(pj(OP_HOME, "receipts", day + ".jsonl"), encoding="utf-8") as f:
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
    p = pj(OP_HOME, "permits", argv[0] + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"action_digest": argv[0], "granted": time.time(), "expires": time.time() + PERMIT_TTL_S,
                   "single_use": True, "by": "owner-tty"}, f)
    os.chmod(p, 0o600)
    write_receipt({"decision": "PERMIT_GRANTED", "action_digest": argv[0], "source": "owner-tty"})
    print("Permit granted for one use.")
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
    upgrade_hint()
    return 0


def cmd_mode(argv):
    """Owner command: switch between recording and blocking. Works the same on every shell."""
    if not argv:
        print("mode is %s. To change it: %s mode enforce|shadow|off" % (global_mode(), self_cmd()))
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
    print({"enforce": "The gate now blocks what its rules stop.",
           "shadow": "The gate now records and blocks nothing.",
           "off": "The gate is off: nothing is checked or recorded."}[argv[0]])
    return 0


# The offer, in one place: where the paid gate is sold and what it costs.
OPERATOR_URL = "https://payhip.com/b/d4xvu"
OPERATOR_PRICE = "$149 once, yours to keep"


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


def replay_history(days, budget, here, every=False):
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
    out["incidents"] = sorted(out["incidents"], key=lambda i: i["ts"], reverse=True)[:None if every else 20]
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
        print("usage: operator-gate replay [--days N | --all] [--here] [--stops] [--budget SECONDS] [--json | --share]")
        return 2
    if "--all" in argv:
        days = 0
    here = os.getcwd() if "--here" in argv else None
    r = replay_history(days, budget, here, "--stops" in argv)
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
        shown = r["incidents"] if "--stops" in argv else r["incidents"][:5]
        print("\n" + paint("Would have been stopped", "1") + " (%s):" % (
            "the %d most recent of %d" % (len(shown), r["stopped"]) if r["stopped"] > len(shown) else "all of them"))
        for inc in shown:
            when = "      "
            if len(inc["ts"]) >= 10 and inc["ts"][5:7].isdigit() and 1 <= int(inc["ts"][5:7]) <= 12:
                when = "%s %s" % (MONTHS[int(inc["ts"][5:7]) - 1], inc["ts"][8:10])
            print("  %s  %-18s %s %s" % (when, inc["project"][:18], paint(inc["rule"], "31"), inc["why"]))
            print("  %s  %-18s %s" % ("      ", "", paint(inc["cmd"], "2")))
        if r["stopped"] > len(shown):
            print("  every one of them, to check each yourself: %s replay --stops" % self_cmd())
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


def cmd_check(argv):
    """Say what would be decided about one command. Nothing is installed, recorded or run: the command
    is read, and so are the files it would set running."""
    shell, cwd, words, as_json, i = "Bash", os.getcwd(), [], False, 0
    while i < len(argv):
        if argv[i] == "--shell" and i + 1 < len(argv):
            shell = "PowerShell" if argv[i + 1].lower() in ("powershell", "pwsh", "ps") else "Bash"
            i += 1
        elif argv[i] == "--cwd" and i + 1 < len(argv):
            cwd = os.path.abspath(os.path.expanduser(argv[i + 1]))
            i += 1
        elif argv[i] == "--json":
            as_json = True
        else:
            words.append(argv[i])
        i += 1
    cmd = " ".join(words) if words else ("" if sys.stdin.isatty() else sys.stdin.read())
    if not cmd.strip():
        print("usage: operator-gate check [--shell bash|powershell] [--cwd DIR] [--json] '<command>'")
        return 2
    cfg = load_config()
    _, cwd, hits, _, meta, _ = classify({"tool_name": shell, "tool_input": {"command": cmd}, "cwd": cwd}, cfg)
    stops, records = [], []
    for rid, detail, hint in hits:
        if rid in cfg["disabled_rules"] or rid == "OP-CANARY":
            continue
        mode = cfg["mode_overrides"].get(rid, "shadow" if RULES[rid][1] == "SHADOW" else "enforce")
        (stops if mode == "enforce" else records).append((rid, detail, hint))
    if as_json:
        print(json.dumps({"verdict": "stop" if stops else ("record" if records else "pass"), "cwd": cwd,
                          "stops": [{"rule": r, "name": RULES[r][0], "detail": redact(d), "do_instead": h} for r, d, h in stops],
                          "records": [{"rule": r, "name": RULES[r][0], "detail": redact(d)} for r, d, _ in records],
                          "read_through": meta.get("transforms", [])}, sort_keys=True))
        return 2 if stops else 0
    for rid, detail, hint in stops:
        print("%s  %s %s" % (paint("STOP  ", "1;31"), rid, RULES[rid][0]))
        print("        %s" % redact(detail).replace(HOME, "~"))
        print("        rule: %s" % RULES[rid][2])
        if hint:
            print("        instead: %s" % hint)
    for rid, detail, _ in records:
        print("%s  %s %s" % (paint("record", "1;33"), rid, RULES[rid][0]))
        print("        %s" % redact(detail).replace(HOME, "~"))
    if not stops and not records:
        print("%s  no rule matched this command" % paint("pass  ", "1;32"))
    elif not stops:
        print("        recorded and let through: these rules watch, they do not stop")
    if meta.get("transforms"):
        print("read through: %s" % ", ".join(meta["transforms"]))
    print("Judged in %s. Nothing was run and nothing was recorded." % cwd.replace(HOME, "~"))
    return 2 if stops else 0


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


def cmd_install(argv):
    """Owner-side install: copies the gate to ~/.operator/gate and wires Claude Code. Mode starts SHADOW."""
    if os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: installs are run by the owner in their own terminal.")
        return 3
    dst_dir = pj(HOME, ".operator", "gate")
    src = os.path.abspath(__file__)
    os.makedirs(dst_dir, exist_ok=True)
    dst = pj(dst_dir, "operator-gate.py")
    if os.path.realpath(src) != os.path.realpath(dst):
        shutil.copyfile(src, dst)                  # bytes: the installed gate is the downloaded file exactly
    os.chmod(dst, 0o755)
    ensure_dirs()
    mode_path = pj(OP_HOME, "mode")
    if not os.path.exists(mode_path):
        with open(mode_path, "w", encoding="utf-8") as f:
            f.write("shadow\n")
    # Claude Code PreToolUse hook (merge, never clobber)
    settings_path = pj(HOME, ".claude", "settings.json")
    hook_cmd = hook_command(dst, "claude")
    try:
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            settings = {}
        hooks = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
        entry = {"matcher": "*", "hooks": [{"type": "command", "command": hook_cmd}]}
        if not any(h.get("hooks") and any(hh.get("command") == hook_cmd for hh in h["hooks"]) for h in hooks):
            hooks.append(entry)
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)   # Claude Code not started yet on this machine
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        print("claude: PreToolUse hook registered in %s" % settings_path)
    except Exception as exc:
        print("claude: could not register hook (%s) -- add manually:" % exc)
        print('  "hooks": {"PreToolUse": [{"matcher": "*", "hooks": '
              '[{"type": "command", "command": "%s"}]}]}' % hook_cmd)
    print("other hosts: register this as their pre-tool hook -> %s" % hook_command(dst, "<host>"))
    try:
        launcher = place_launcher(dst)
    except Exception:
        launcher = None
    if launcher:
        print("command: operator-gate (%s)" % launcher)
    print("mode: %s. It records every verdict and blocks nothing until you run: %s mode enforce" % (global_mode(), self_cmd()))
    if "--no-replay" not in argv:                 # what this gate would have said about last month
        try:
            print()
            cmd_replay(["--budget", "12", "--from-install"])
        except Exception:
            pass
    print("\nFrom now on every tool call your agents make passes through the gate.\n"
          "See what it caught: %s status" % self_cmd())
    return 0


def cmd_uninstall(argv):
    if os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: uninstalls are run by the owner in their own terminal.")
        return 3
    settings_path = pj(HOME, ".claude", "settings.json")
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        for pre in settings.get("hooks", {}).get("PreToolUse", []):
            pre["hooks"] = [h for h in pre.get("hooks", []) if "operator-gate" not in h.get("command", "")]
        settings["hooks"]["PreToolUse"] = [p for p in settings["hooks"]["PreToolUse"] if p.get("hooks")]
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
           "upgrade": cmd_upgrade, "replay": cmd_replay, "mode": cmd_mode, "check": cmd_check}.get(argv[0])
    if not cmd:
        print("usage: operator-gate hook|check|status|replay|mode|approve|verify|corpus|install|uninstall|upgrade")
        sys.exit(2)
    sys.exit(cmd(argv[1:]))


if __name__ == "__main__":
    main()

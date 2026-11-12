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
import collections
import csv
import glob
import fnmatch
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
from urllib.parse import urlsplit

try:
    import fcntl
except ImportError:                                  # Windows: receipts lock through msvcrt instead
    fcntl = None

VERSION = "6.2.2"
INTEGRITY = "reference_digest_not_signature"
# Operator is a strict superset of Operator Lite: the Lite gate ships beside this file, byte for byte,
# and every call is also judged by it. A copy whose bytes differ from the pin is not loaded.
# The pin is operator-lite/operator-gate.py in this repository (Operator Lite 6.2.2).
LITE_CORE_FILE = "lite_core.py"
LITE_CORE_VERSION = "6.2.2"
LITE_CORE_SHA256 = "9de59e80f5ffa1d8ce380351d950dd612660d998020f791e483b17493f95a63d"
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
    if not path:
        return set()
    out = {path, real(path)}
    out = {v.rstrip("/") or "/" for v in out if v}
    return {v.casefold() for v in out} if IS_WINDOWS else out      # Windows paths ignore case


def is_same_or_ancestor(target, protected):
    """True when deleting/moving `target` removes `protected` (equal or an ancestor of it)."""
    if not target or not protected:
        return False
    for t in variants(target):
        for p in variants(protected):
            if t == p or p.startswith(t.rstrip("/") + "/") or t == "/":
                return True
    return False


def same_path(a, b):
    """The same place, whatever the spelling: symlinks resolved, and on Windows slashes and case."""
    return bool(a and b and (variants(a) & variants(b)))


def is_inside(target, root):
    if not target or not root:
        return False
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
               "Live billing actions and billing or control-plane deploys need the owner's explicit approval."),
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
    "OP-012": ("the-floor", "STOP",
               "A recursive delete that reaches a drive or filesystem root, a home folder or a parent of the "
               "working directory is refused in every mode, and no permit exists for it."),
    "OP-013": ("no-root-reach", "STOP",
               "A root job started in the background, or a kill aimed at every process or at a process "
               "group as root, needs the owner's approval."),
    "OP-014": ("no-machine-wreck", "STOP",
               "Writing over a disk, making a filesystem on one, or sweeping permissions or ownership "
               "across a home folder, a root or a parent of the working directory needs the owner's approval."),
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
    "OP-018": ("no-blind-fetch-run", "STOP",
               "Code fetched from the network and handed straight to a shell runs before anyone has read "
               "it. Outside the well-known installers it needs the owner's approval."),
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
    "OP-026": ("test-integrity", "STOP",
               "The checks that say the work is good belong to the owner: skipping git hooks, removing, skipping or "
               "focusing tests, weakening a CI job or the coverage floor, or saying the tests pass when no test run "
               "is on record needs the owner's approval."),
    "OP-027": ("no-container-escape", "STOP",
               "Mounting docker.sock, --privileged containers, nsenter, or host-root volume mounts breaks container isolation and hands the agent host root."),
    "OP-028": ("no-keychain-dump", "STOP",
               "Dumping OS keychains or credential store passwords (security dump-keychain, secret-tool) is prohibited."),
    "OP-029": ("protect-ssh-keys", "STOP",
               "Writing or appending to ~/.ssh/authorized_keys plants persistent remote backdoors."),
    "OP-030": ("no-untrusted-registry", "STOP",
               "Installing packages from untrusted external package indexes (--index-url, --registry) risks dependency confusion RCE."),
    "OP-031": ("no-credential-store-read", "STOP",
               "Reading or copying home credential stores or browser/app secret files is owner-only."),
    "OP-032": ("no-secret-cli-dump", "STOP",
               "Commands that dump passwords or credentials from password managers or cloud tokens are owner-only."),
    "OP-033": ("no-secret-file-egress", "STOP",
               "Sending credential files, private keys, or secret files off-machine is owner-only."),
    "OP-034": ("no-paste-exfil", "STOP",
               "Sending local data to pastebins, request bins, or OAST hosts is owner-only."),
    "OP-035": ("no-dns-exfil", "STOP",
               "DNS lookups carrying command substitutions or encoded secrets in query labels are owner-only."),
    "OP-036": ("no-mcp-secret-arg", "STOP",
               "Passing credential values or sensitive keys in MCP tool arguments is owner-only."),
    "OP-037": ("no-registry-config-write", "STOP",
               "Configuring an untrusted package index in persistent config files (.npmrc, pip.conf) is owner-only."),
    "OP-038": ("no-workflow-poison", "STOP",
               "Editing CI workflows to run untrusted PR code with elevated tokens or inject into runner env is owner-only."),
    "OP-039": ("no-loader-inject", "STOP",
               "Setting dynamic loader variables (LD_PRELOAD, DYLD_INSERT_LIBRARIES) or shell startup hooks is owner-only."),
    "OP-040": ("no-interpreter-hook", "STOP",
               "Interpreter options that load foreign code (NODE_OPTIONS=--require, -javaagent:) are owner-only."),
    "OP-041": ("no-search-path-escape", "STOP",
               "Search path variables adding foreign directories or dropping system paths are owner-only."),
    "OP-042": ("no-foreign-debug", "STOP",
               "Attaching a debugger to an existing process or reading process memory is owner-only."),
    "OP-043": ("no-persistence-control", "STOP",
               "Installing scheduled tasks, launchd daemons, or systemd services is owner-only."),
    "OP-044": ("no-persistence-file", "STOP",
               "Writing persistence files into LaunchAgents, cron directories, or systemd units is owner-only."),
    "OP-045": ("no-os-trust-tamper", "STOP",
               "Disabling platform security controls (SIP, Gatekeeper, packet filters, sudoers) is owner-only."),
    "OP-046": ("no-trust-store-inject", "STOP",
               "Installing certificate authorities into system trust stores is owner-only."),
    "OP-047": ("no-quarantine-strip", "STOP",
               "Stripping quarantine marks from downloaded apps or binaries is owner-only."),
    "OP-048": ("no-inbound-exec", "STOP",
               "Starting an inbound network listener that executes a shell or command is owner-only."),
    "OP-049": ("no-remote-tunnel", "STOP",
               "Reverse tunnels or funnel proxies exposing local ports to the internet are owner-only."),
    "OP-050": ("no-resolver-tamper", "STOP",
               "Modifying name resolution (/etc/hosts, DNS servers) or network routing is owner-only."),
    "OP-051": ("no-private-key-material", "STOP",
               "Reading, copying, or outputting private key files or crypto wallets is owner-only."),
    "OP-052": ("no-secret-export", "STOP",
               "Commands whose job is to export or unwrap private keys or keystore seeds are owner-only."),
    "OP-053": ("no-ssh-exec-config", "STOP",
               "SSH options or config keys that execute local programs (ProxyCommand, LocalCommand) are owner-only."),
    "OP-054": ("no-agent-bypass-spawn", "STOP",
               "Spawning a nested coding agent with its safety gates or approval prompts stripped is owner-only."),
    "OP-055": ("no-local-model-mutate", "STOP",
               "Deleting, overwriting, or replacing local models or opening their admin APIs is owner-only."),
    "OP-056": ("no-hook-strip", "STOP",
               "Modifying agent configuration to remove or disable the safety gate hooks is owner-only."),
    "OP-057": ("no-checker-weaken", "STOP",
               "Weakening typecheckers, linters, or compiler strictness in configuration files is owner-only."),
    "OP-058": ("no-suppression-spray", "STOP",
               "Spraying blanket linter/type suppression directives (@ts-nocheck, noqa) is owner-only."),
    "OP-059": ("no-error-swallow", "STOP",
               "Writing empty exception handlers that swallow errors in production code is owner-only."),
    "OP-060": ("no-canned-oracle", "STOP",
               "Adding test-detecting branches that return canned constants for fixtures is owner-only."),
    "OP-061": ("no-fake-test-script", "STOP",
               "Replacing test runner scripts in package manifests with no-op commands is owner-only."),
    "OP-062": ("no-migration-rewrite", "STOP",
               "Modifying, truncating, or deleting tracked database migrations already in git HEAD is owner-only."),
    "OP-063": ("no-lockfile-destroy", "STOP",
               "Deleting, truncating, or replacing tracked package lockfiles without permission is owner-only."),
    "OP-064": ("no-history-rewrite", "STOP",
               "Rewriting existing git history (git filter-repo, git filter-branch, git replace) is owner-only."),
    "OP-065": ("no-ref-clobber", "STOP",
               "Deleting git references, tags, or unmerged branches (update-ref -d, tag -d, branch -D) is owner-only."),
    "OP-066": ("no-cluster-drain", "STOP",
               "Draining nodes or deleting Kubernetes core resources without dry-run is owner-only."),
    "OP-067": ("no-auto-approve-apply", "STOP",
               "Executing infrastructure-as-code apply with auto-approval or prompt bypass is owner-only."),
    "OP-068": ("no-platform-teardown", "STOP",
               "Teardown commands on cloud platform CLIs (azd down, wrangler delete, hcloud delete) are owner-only."),
    "OP-069": ("no-schema-clean", "STOP",
               "Database schema reset commands (flyway clean, liquibase dropAll, atlas schema clean) are owner-only."),
    "OP-070": ("no-github-vandalism", "STOP",
               "Deleting GitHub issues, locking discussions, or deleting repository keys/collaborators is owner-only."),
    "OP-071": ("no-outbound-blast", "STOP",
               "Broadcasting emails, SMS messages, or chat posts through production APIs is owner-only."),
    "OP-072": ("no-gpu-burn", "STOP",
               "Provisioning expensive GPU instances or submitting paid model fine-tuning jobs is owner-only."),
    "OP-073": ("no-opaque-protected-write", "STOP",
               "Writes or redirects whose destination cannot be resolved but touch protected files are owner-only."),
    "OP-074": ("no-atomic-replace", "STOP",
               "Atomic file replacement or rename targeting protected files or migrations is owner-only."),
    "OP-075": ("no-loop-redirect", "STOP",
               "Shell loops writing through variables that iterate over protected files are owner-only."),
    "OP-076": ("no-checker-delete", "STOP",
               "Deleting typechecker or linter configuration files (tsconfig.json, ruff.toml) is owner-only."),
    "OP-077": ("no-unmonitored-agent-home", "STOP",
               "Redirecting agent home configuration directories to unmonitored locations is owner-only."),
    "OP-078": ("no-instruction-inject", "STOP",
               "Injecting prompt overrides or instruction bypasses into project documentation or rules is owner-only."),
    "OP-079": ("no-global-skill-write", "STOP",
               "Mutating user-global agent skills, commands, or custom rules outside git repos is owner-only."),
    "OP-080": ("no-git-identity-spoof", "STOP",
               "Spoofing git author identities, emails, or disabling commit signing is owner-only."),
    "OP-081": ("no-governance-weaken", "STOP",
               "Deleting or weakening repository governance files (CODEOWNERS, LICENSE, SECURITY.md) is owner-only."),
    "OP-082": ("no-public-bucket-acl", "STOP",
               "Making cloud storage buckets or objects publicly accessible is owner-only."),
    "OP-083": ("no-cloud-iam-mutation", "STOP",
               "Creating cloud access keys, modifying IAM policies, or tampering with KMS keys is owner-only."),
    "OP-084": ("no-dns-tls-prod-change", "STOP",
               "Deleting production DNS zones, modifying DNS records, or revoking TLS certificates is owner-only."),
    "OP-085": ("no-cloud-secret-destroy", "STOP",
               "Deleting or destroying cloud secrets, parameters, or vault entries is owner-only."),
    "OP-098": ("no-tls-disable", "STOP",
               "Disabling TLS certificate verification in HTTP clients, env vars, or tool configs is owner-only."),
    "OP-099": ("no-untrusted-package-source", "STOP",
               "Configuring cleartext HTTP package dependencies or injecting shell commands into lifecycle scripts is owner-only."),
    "OP-101": ("protect-local-state", "STOP",
               "Deleting, truncating, or replacing gitignored local developer state (.env, databases, uploads) is owner-only."),
    "OP-102": ("no-archive-clobber", "STOP",
               "Extracting archives over existing work or escaping working directories is owner-only."),
    "OP-103": ("no-stream-wipe", "STOP",
               "Emptying existing non-empty files via shell redirection, truncate, or stream wipes is owner-only."),
    "OP-104": ("no-sync-delete", "STOP",
               "Executing mirror sync deletions without dry-run confirmation is owner-only."),
    "OP-105": ("no-backup-destroy", "STOP",
               "Deleting OS backups, APFS snapshots, ZFS pools, or Volume Shadow Copies is owner-only."),
    "OP-106": ("no-git-object-destroy", "STOP",
               "Deleting or force-moving git branches holding unique unmerged commits or force-pruning objects is owner-only."),
    "OP-107": ("no-live-browser-debug", "STOP",
               "Launching user browsers with DevTools remote debugging without isolated profiles is owner-only."),
    "OP-108": ("no-gui-inject", "STOP",
               "Synthesizing keyboard or mouse input to control desktop GUI applications or bypass prompts is owner-only."),
    "OP-109": ("no-sensor-capture", "STOP",
               "Recording the screen, webcam, or microphone devices is owner-only."),
    "OP-110": ("no-clipboard-abuse", "STOP",
               "Exfiltrating clipboard data off-machine or writing secrets into the system clipboard is owner-only."),
    "OP-111": ("no-extension-sideload", "STOP",
               "Sideloading browser or editor extensions or modifying global editor shell hooks is owner-only."),
    "OP-112": ("no-mcp-outbound", "STOP",
               "Sending outbound communications or deleting messages via MCP tools is owner-only."),
    "OP-113": ("no-mcp-workspace-destroy", "STOP",
               "Deleting repositories, issues, or merging pull requests via MCP tools is owner-only."),
    "OP-114": ("no-mcp-billing", "STOP",
               "Initiating financial charges, payouts, or deleting cloud infrastructure via MCP tools is owner-only."),
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
                    ".ssh/rc", ".zlogin", ".config/fish/config.fish", ".bash_login", ".bash_logout", ".zlogout")
HIJACK_VARS = {"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
               "NODE_OPTIONS", "NODE_PATH", "PYTHONSTARTUP", "PYTHONPATH", "BASH_ENV", "ENV",
               "PERL5OPT", "RUBYOPT", "GLIBC_TUNABLES", "SHELL", "ZDOTDIR"}
# 0.3.14: Claude Code sources one of these files before every Bash call (`zsh -c 'source SNAPSHOT && eval CMD'`),
# so the owner's aliases and functions are live in each call and a line added to a snapshot reaches every later
# call of the session. The gate reads them to judge a name the way the shell will run it, and guards them as init files.
SHELL_SNAPSHOT_DIRS = sorted({pj(os.environ.get("CLAUDE_CONFIG_DIR") or pj(HOME, ".claude"), "shell-snapshots"),
                              pj(HOME, ".claude", "shell-snapshots")})
SNAPSHOT_MAX_BYTES = 8 * 1024 * 1024
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


# Native Pro hard stops. These consume the analyzer's quote-stripped, expanded
# argv, not raw command text (which may just contain an example in a message).
CONTAINER_RUN_VALUES = {
    "--add-host", "--annotation", "--attach", "--blkio-weight", "--cap-add", "--cap-drop",
    "--cgroup-parent", "--cgroupns", "--cidfile", "--cpus", "--cpuset-cpus", "--device",
    "--device-cgroup-rule", "--dns", "--dns-option", "--dns-search", "--entrypoint", "--env",
    "--env-file", "--expose", "--gpus", "--group-add", "--health-cmd", "--health-interval",
    "--hostname", "--ipc", "--label", "--label-file", "--link", "--log-driver", "--log-opt",
    "--memory", "--memory-swap", "--mount", "--name", "--network", "--pid", "--platform",
    "--publish", "--pull", "--restart", "--runtime", "--security-opt", "--shm-size", "--stop-signal",
    "--storage-opt", "--sysctl", "--tmpfs", "--ulimit", "--user", "--userns", "--uts", "--volume",
    "--volume-driver", "--volumes-from", "--workdir", "--pod", "--secret", "--network-alias",
}
CONTAINER_GLOBAL_VALUES = {"--config", "--context", "--host", "--log-level", "--namespace",
                           "--address", "--url", "--connection", "--root", "--runroot"}
CONTAINER_GLOBAL_BOOLEANS = {"--debug", "--help", "--version", "--tls", "--tlsverify",
                             "--rootless", "--experimental", "--remote", "--syslog", "--transient-store",
                             "--debug-full", "--insecure-registry", "--kube-hide-dupe", "--selinux-enabled"}
UNSHARE_BOOLEANS = {"--mount", "--mount-proc", "--uts", "--ipc", "--net", "--pid", "--user",
                   "--cgroup", "--time", "--fork", "--kill-child", "--map-root-user",
                   "--map-current-user", "--map-auto", "--map-subids", "--keep-caps", "--help", "--version"}
CONTAINER_RUN_BOOLEANS = {
    "--detach", "--disable-content-trust", "--help", "--init", "--interactive", "--no-healthcheck",
    "--oom-kill-disable", "--privileged", "--publish-all", "--read-only", "--rm", "--sig-proxy",
    "--tty", "--use-api-socket", "--no-hosts", "--http-proxy", "--replace",
    "--rmi", "--rootfs", "--tls-verify", "--quiet", "--read-only-tmpfs", "--passwd", "--env-host",
}
PYTHON_REGISTRY_HOSTS = {"pypi.org", "pythonhosted.org", "files.pythonhosted.org"}
NODE_REGISTRY_HOSTS = {"registry.npmjs.org", "registry.yarnpkg.com"}
PYTHON_PACKAGE_VALUES = {
    "--config-settings", "--config-setting", "--constraint", "--requirement", "--build-constraint",
    "--target", "--prefix", "--root", "--src", "--platform", "--python-version", "--implementation",
    "--abi", "--no-binary", "--only-binary", "--log", "--proxy", "--retries", "--timeout", "--exists-action",
    "--cert", "--client-cert", "--cache-dir", "--keyring-provider", "--use-feature", "--use-deprecated",
    "--group", "--python", "--directory", "--project", "--config-file", "--link-mode", "--resolution",
    "--index-strategy", "--python-preference", "--python-downloads", "--trusted-host", "--find-links",
}
NODE_PACKAGE_VALUES = {"--script-shell", "--userconfig", "--globalconfig", "--cache", "--prefix",
                       "--workspace", "--otp", "--tag", "--access", "--loglevel", "--location",
                       "--cafile", "--cert", "--key", "--proxy", "--https-proxy"}
SSH_AUTHORIZED_RE = re.compile(r"(?:^|/)\.ssh/authorized_keys2?$")
# Apple SecurityTool dispatches to the FIRST command matching the supplied
# prefix. Preserve its order (verified with security help and Apple's source),
# so "dump-k"/"find-generic-pass" cannot bypass and "d" still means default-keychain.
SECURITY_COMMANDS = (
    "help", "list-keychains", "list-smartcards", "default-keychain", "login-keychain", "create-keychain",
    "delete-keychain", "lock-keychain", "unlock-keychain", "set-keychain-settings", "set-keychain-password",
    "show-keychain-info", "dump-keychain", "create-keypair", "add-generic-password", "add-internet-password",
    "add-certificates", "find-generic-password", "delete-generic-password", "set-generic-password-partition-list",
    "find-internet-password", "delete-internet-password", "set-internet-password-partition-list", "find-key",
    "set-key-partition-list", "find-certificate", "find-identity", "delete-certificate", "delete-identity",
    "set-identity-preference", "get-identity-preference", "create-db", "export", "import", "export-smartcard",
    "cms", "install-mds", "add-trusted-cert", "remove-trusted-cert", "dump-trust-settings",
    "user-trust-settings-enable", "trust-settings-export", "trust-settings-import", "verify-cert", "authorize",
    "authorizationdb", "execute-with-privileges", "leaks", "error", "create-filevaultmaster-keychain", "smartcards",
    "translocate-policy-check", "translocate-status-check", "translocate-original-path", "requirement-evaluate",
    "filevault", "platformsso",
)


def ssh_authorized_target(path):
    if SSH_AUTHORIZED_RE.search(path):
        return True
    if not SSH_AUTHORIZED_RE.search(path.lower()):
        return False
    # Case aliases matter only when the filesystem actually resolves them.
    # Probe metadata for the nearest existing path, never read key contents or
    # create a test file in an owner's directory.
    parent = path
    while parent and not os.path.exists(parent):
        following = os.path.dirname(parent)
        if following == parent:
            return False
        parent = following
    try:
        alias = os.path.join(os.path.dirname(parent), os.path.basename(parent).swapcase())
        return alias != parent and os.path.samefile(parent, alias)
    except OSError:
        return False


def leading_options(words, long_values=(), short_values="", inferred_values=()):
    """Read CLI options up to the first operand; retain attached/equals values.

    Unlike flags_and_args, this keeps option values out of the operand list and
    never reads a container's image command as options for the container engine.
    """
    options, i = [], 0
    while i < len(words):
        word = words[i]
        if word == "--":
            return options, words[i + 1:]
        if not word.startswith("-") or word == "-":
            break
        if word.startswith("--"):
            name, sep, value = word.partition("=")
            if not sep and name in long_values:
                if name in inferred_values and i + 1 < len(words) and words[i + 1].startswith("-"):
                    value = None                    # an unknown boolean cannot eat the next security flag
                else:
                    i += 1
                    value = words[i] if i < len(words) else ""
            options.append((name, value if sep or name in long_values else None))
        else:
            for k, flag in enumerate(word[1:], 1):
                value = None
                if flag in short_values:
                    value = word[k + 1:]
                    if not value:
                        i += 1
                        value = words[i] if i < len(words) else ""
                options.append(("-" + flag, value))
                if flag in short_values:
                    break
        i += 1
    return options, words[i:]


def container_escape(toks, cwd):
    if not toks:
        return None
    prog, rest = os.path.basename(toks[0]), toks[1:]
    if prog == "nsenter":
        if rest[:1] in (["--help"], ["-h"], ["--version"], ["-V"]):
            return None
        return "invokes the host namespace entry utility"
    if prog == "unshare":
        values = {w.split("=", 1)[0] for w in rest
                  if w.startswith("--") and w.split("=", 1)[0] not in UNSHARE_BOOLEANS}
        options, _ = leading_options(rest, values, "RwSG", values)
        if any(name in ("--help", "-h", "--version", "-V") for name, _ in options):
            return None
        if any(name in ("--mount", "--mount-proc", "-m") for name, _ in options):
            return "creates a mount namespace with unshare"
        return None
    if prog not in ("docker", "podman", "nerdctl", "docker-compose"):
        return None
    global_values = CONTAINER_GLOBAL_VALUES | {w.split("=", 1)[0] for w in rest
                                               if w.startswith("--") and
                                               w.split("=", 1)[0] not in CONTAINER_GLOBAL_BOOLEANS}
    _, rest = leading_options(rest, global_values, "Hcnal", global_values - CONTAINER_GLOBAL_VALUES)
    if rest[:1] == ["container"]:
        rest = rest[1:]
    if rest[:1] not in (["run"], ["create"]):
        return None
    values = CONTAINER_RUN_VALUES | {w.split("=", 1)[0] for w in rest[1:]
                                     if w.startswith("--") and w.split("=", 1)[0] not in CONTAINER_RUN_BOOLEANS}
    options, _ = leading_options(rest[1:], values, "acehlmpuvw", values - CONTAINER_RUN_VALUES)
    if any(name == "--help" and value not in ("false", "0") for name, value in options):
        return None
    for name, value in options:
        if name == "--privileged" and (value is None or value.lower() not in ("false", "0")):
            return "runs a privileged container"
        # Host namespaces
        if name in ("--pid", "--network", "--net", "--ipc", "--uts", "--userns") and value == "host":
            return "shares host %s namespace with container" % name.lstrip("-")
        # Sensitive capabilities
        if name == "--cap-add" and value and value.upper() in ("SYS_ADMIN", "SYS_PTRACE", "SYS_MODULE", "NET_ADMIN", "DAC_OVERRIDE", "ALL"):
            return "grants elevated container capability %s" % value.upper()
        # Security opts
        if name == "--security-opt" and value in ("apparmor=unconfined", "apparmor:unconfined", "seccomp=unconfined", "seccomp:unconfined", "label=disable"):
            return "disables container security confinement (%s)" % value
        # Raw devices
        if name == "--device" and value and any(value.startswith(d) for d in ("/dev/mem", "/dev/kmem", "/dev/sda", "/dev/nvme", "/dev/disk")):
            return "exposes raw host device %s to container" % value
        if name not in ("-v", "--volume", "--mount") or value is None:
            continue
        if name == "--mount":
            try:
                parts = next(csv.reader([value], strict=True))
            except (csv.Error, StopIteration):
                return "uses an unresolvable container mount"
            fields = dict(part.split("=", 1) for part in parts if "=" in part)
            pairs = [part.split("=", 1) for part in parts if "=" in part]
            mount_type = fields.get("type", "bind")
            if mount_type == "volume" and fields.get("volume-driver", "local") == "local":
                volume_opts = [v.split("=", 1) for k, v in pairs if k == "volume-opt" and "=" in v]
                bind = any(k == "o" and any(flag in ("bind", "rbind") for flag in v.split(","))
                           for k, v in volume_opts)
                sources = [v for k, v in volume_opts if k == "device"] if bind else []
            elif mount_type == "bind":
                sources = [v for k, v in pairs if k in ("source", "src")]
            else:
                continue
        else:
            sources = [value.split(":", 1)[0]]
        paths = {p for source in sources if source for p in variants(norm(source, cwd))}
        if any(os.path.basename(p) == "docker.sock" for p in paths):
            return "mounts the host Docker control socket"
        if any(p in ("/", "/var", "/private/var", "/etc", HOME, "/home", "/proc", "/sys", "/boot") for p in paths):
            return "bind-mounts a host root or sensitive host directory"
    return None


def credential_dump(toks):
    if not toks:
        return False
    prog, rest = os.path.basename(toks[0]), toks[1:]
    if prog == "security":
        options, rest = leading_options(rest, (), "p")
        if any(name in ("-h", "--help") for name, _ in options):
            return False
        if any(name == "-i" for name, _ in options):
            return True                              # stdin can dispatch dump/export subcommands
        if not rest and any(name == "-p" for name, _ in options):
            return True                              # a custom prompt implies interactive dispatch
        command = next((name for name in SECURITY_COMMANDS if rest and name.startswith(rest[0])), "")
        return command in {"dump-keychain", "find-generic-password", "find-internet-password", "export"}
    return prog == "secret-tool" and bool(rest and rest[0] in ("lookup", "search"))


def registry_host(value):
    """Accept an unambiguous HTTPS URL; never return its credentials or path."""
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) <= 32 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        _ = parsed.port                            # validates malformed/out-of-range ports
        if parsed.scheme.lower() != "https" or not host or not parsed.netloc or parsed.fragment:
            return None
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host):
            return None
        return host
    except ValueError:
        return None


def trusted_registry_hosts(cfg, ecosystem):
    defaults = PYTHON_REGISTRY_HOSTS if ecosystem == "python" else NODE_REGISTRY_HOSTS
    extra = cfg.get("trusted_%s_registries" % ecosystem, [])
    # Mirrors are exact DNS names in the owner's protected commitments file;
    # neither command-line nor environment input can enlarge this allowlist.
    extra = extra if isinstance(extra, list) else []
    return defaults | {h.lower().rstrip(".") for h in extra if isinstance(h, str)
                       and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", h)}


def package_command(toks):
    """Return (ecosystem, args), including Python's -m pip entry point."""
    if not toks:
        return None, []
    prog, rest = os.path.basename(toks[0]), toks[1:]
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", prog):
        i = 0
        while i < len(rest):
            if rest[i] == "-m":
                if rest[i + 1:i + 2] in (["pip"], ["pip3"]):
                    return "python", rest[i + 2:]
                return None, []
            if re.fullmatch(r"-[bBdEIOPqRsSuvVx]*m", rest[i]):
                if rest[i + 1:i + 2] in (["pip"], ["pip3"]):
                    return "python", rest[i + 2:]
                return None, []
            if rest[i] in ("-c", "--") or not rest[i].startswith("-"):
                break
            i += 2 if rest[i] in ("-W", "-X") else 1
        return None, []
    if prog in ("pip", "pip3", "uv") or re.fullmatch(r"pip\d+\.\d+", prog):
        return "python", rest
    if prog in ("npm", "yarn", "pnpm", "bun"):
        return "node", rest
    return None, []


def untrusted_registry(toks, cfg):
    ecosystem, words = package_command(toks)
    if ecosystem is None:
        return None
    names = {"--index-url", "--extra-index-url", "--index", "--default-index"} if ecosystem == "python" else {"--registry"}
    trusted, i = trusted_registry_hosts(cfg, ecosystem), 0
    value_options = PYTHON_PACKAGE_VALUES if ecosystem == "python" else NODE_PACKAGE_VALUES
    while i < len(words):
        word = words[i]
        if word == "--":
            break
        name, sep, value = word.partition("=")
        prog = os.path.basename(toks[0])
        if ecosystem == "python" and prog != "uv" and name.startswith("--") and len(name) > 2:
            # pip uses optparse's unique long-option prefixes.
            matches = [full for full in ("--index-url", "--extra-index-url") if full.startswith(name)]
            if len(matches) == 1:
                name = matches[0]
        elif prog == "npm" and name.startswith("--reg") and "--registry".startswith(name):
            name = "--registry"                      # npm's documented nopt shorthand/prefix expansion
        if ecosystem == "python" and name.startswith("--") and name not in names and len(name) > 2:
            possible = [full for full in value_options if full.startswith(name)]
            if len(possible) == 1:
                name = possible[0]
        if name in value_options:
            i += 1 if sep else 2                      # backend/config option values are data, not argv
            continue
        if ecosystem == "python" and re.match(r"^-[Crcft]", word) and not word.startswith("--"):
            i += 2 if len(word) == 2 else 1
            continue
        if ecosystem == "python" and word.startswith("-i") and not word.startswith("--"):
            name, sep, value = "--index-url", "=", word[2:]
            if not value:
                sep = ""
        if name in names:
            if not sep:
                i += 1
                value = words[i] if i < len(words) else ""
            if os.path.basename(toks[0]) == "uv" and name in ("--index", "--default-index"):
                label, equal, named_url = value.partition("=")
                if equal and re.fullmatch(r"[A-Za-z0-9_.-]+", label):
                    value = named_url                     # uv accepts name=https://registry
            host = registry_host(value)
            if host not in trusted:
                return "%s overrides its package registry with %s" % (os.path.basename(toks[0]),
                                                                      host or "an unresolved or invalid HTTPS index")
        i += 1
    return None


# --------------------------------------------------------------------------- #
# Unified Super-Kernel Detectors (OP-031 through OP-114)
# --------------------------------------------------------------------------- #
import os
import re
import sys
import json
import csv
import subprocess

CRED_STORE_FILE_RE = re.compile(
    r'(?:^|/)(?:\.aws/(?:credentials|config)|\.config/gcloud/(?:credentials\.db|application_default_credentials\.json)|'
    r'\.azure/(?:accessTokens\.json|azureProfile\.json)|\.kube/config|\.docker/config\.json|'
    r'\.config/gh/hosts\.yml|\.netrc|\.git-credentials|\.vault-token|'
    r'\.terraform\.d/credentials\.tfrc\.json|\.config/op|'
    r'(?:Library/Application Support|AppData/Local|\.config)/.+/'
    r'(?:Login Data|Cookies|Web Data|Local State|Network/Cookies)(?:-journal)?|'
    r'(?:Library/Application Support/Firefox/Profiles|\.mozilla/firefox)/[^/]+/(?:cookies\.sqlite|logins\.json|key4\.db|cert9\.db)|'
    r'Library/Cookies/Cookies\.binarycookies|'
    r'var/run/secrets/kubernetes\.io/serviceaccount/(?:token|\.\.data))\b', re.I)

READ_OR_COPY_PROGS = {"cat", "head", "tail", "bat", "less", "more", "strings", "xxd", "hexdump", "base64",
                      "cp", "mv", "rsync", "scp", "sqlite3", "od", "nl", "ditto", "tar", "zip"}

def check_cred_store_read(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    if prog not in READ_OR_COPY_PROGS:
        return None
    for a in toks[1:]:
        if a.startswith("-"):
            continue
        p = resolve(a, cwd)
        root = git_top(p) or cwd
        if is_inside(p, cwd) or (root and is_inside(p, root)):
            if not any(k in p for k in ("/.aws/", "/.kube/", "/.azure/", "/.config/gcloud/", "/.docker/config.json", "/.config/gh/")):
                continue
        if CRED_STORE_FILE_RE.search(p) or CRED_STORE_FILE_RE.search(a):
            return "reads or copies home credential store (%s)" % os.path.basename(p)
    return None

def check_secret_cli_dump(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = [a for a in toks[1:] if not a.startswith("-") or a in ("--raw", "-r", "--export")]
    raw_args = toks[1:]
    if any(a in ("--help", "-h", "--version") for a in raw_args) or raw_args[:1] == ["help"]:
        return None
    if prog == "op":
        if any(a in ("read", "inject") for a in raw_args) or \
           (len(args) >= 2 and args[0] in ("item", "document") and args[1] in ("get", "list")):
            return "op dumps secrets from 1Password vault"
        if len(args) >= 1 and args[0] == "run" and any(a in ("env", "printenv") or a == "--no-masking" for a in raw_args):
            return "op run dumps injected environment secrets"
    if prog == "bw" and (any(a in ("get", "export") for a in raw_args) or (len(args) >= 2 and args[0] == "list" and args[1] == "items")):
        if not (len(args) >= 2 and args[0] == "get" and args[1] == "template"):
            return "bw dumps Bitwarden vault credentials"
    if prog in ("pass", "gopass") and any(a in ("show", "grep", "cat") for a in raw_args):
        return "%s outputs stored passwords" % prog
    if prog == "lpass" and any(a in ("show", "export") for a in raw_args):
        return "lpass outputs LastPass vault credentials"
    if prog == "keepassxc-cli" and any(a in ("show", "export", "extract", "clip", "attachment-export") for a in raw_args):
        return "keepassxc-cli extracts database credentials"
    if prog == "gh" and len(args) >= 2 and args[0] == "auth" and args[1] == "token":
        return "gh auth token prints GitHub authentication token"
    if prog == "gcloud" and len(args) >= 2 and args[0] == "auth" and any(args[1].endswith(k) for k in ("print-access-token", "print-identity-token")):
        return "gcloud auth prints cloud access token"
    if prog == "aws" and len(args) >= 2 and args[0] == "configure" and args[1] == "export-credentials":
        return "aws configure export-credentials dumps AWS credentials"
    if prog == "az" and len(args) >= 2 and args[0] == "account" and args[1] == "get-access-token":
        return "az account get-access-token prints Azure access token"
    if prog in ("kubectl", "oc") and len(args) >= 2 and args[0] == "config" and args[1] == "view" and any(a.startswith("--raw") for a in raw_args):
        return "kubectl config view --raw dumps unredacted certificates and tokens"
    if prog in ("cmdkey", "vaultcmd") and any(a.lower() in ("/list", "/listcreds") for a in raw_args):
        return "%s lists Windows stored credentials" % prog
    if prog == "doppler" and (len(args) >= 1 and args[0] == "secrets" and any(a in ("download", "get") for a in raw_args) or args == ["secrets"]):
        return "doppler dumps managed secrets"
    if prog == "infisical" and any(a in ("export",) for a in raw_args):
        return "infisical export dumps project secrets"
    if prog == "vercel" and len(args) >= 2 and args[0] == "env" and args[1] == "pull":
        return "vercel env pull downloads production environment secrets"
    if prog == "railway" and len(args) >= 1 and args[0] in ("variables", "vars") and not any(a in ("set", "delete") for a in raw_args):
        return "railway variables dumps environment secrets"
    if prog == "heroku" and len(args) >= 1 and args[0] in ("config", "config:get"):
        return "heroku config dumps environment secrets"
    return None

EGRESS_HTTP_CLIENTS = {"curl", "wget", "http", "https", "xh"}
def check_secret_file_egress(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in EGRESS_HTTP_CLIENTS:
        for idx, a in enumerate(args):
            val = None
            if a in ("-d", "--data", "--data-binary", "--data-raw", "-T", "--upload-file", "-F", "--form"):
                if idx + 1 < len(args):
                    val = args[idx + 1]
            elif a.startswith(("--data-binary=", "--data=", "-F", "--form=", "-T=")):
                val = a.split("=", 1)[1]
            if val:
                target_str = None
                if "@" in val:
                    target_str = val.split("@", 1)[1]
                if target_str:
                    target = resolve(target_str, cwd)
                    if EXFIL_SECRET_PATH_RE.search(target) or CRED_STORE_FILE_RE.search(target):
                        return "%s uploads secret file %s" % (prog, os.path.basename(target))
    elif prog in ("scp", "sftp", "rsync"):
        for a in args:
            if not a.startswith("-") and ":" not in a:
                p = resolve(a, cwd)
                if (EXFIL_SECRET_PATH_RE.search(p) or CRED_STORE_FILE_RE.search(p)) and not is_scratch(p):
                    return "%s sends secret file %s off-machine" % (prog, os.path.basename(p))
    elif prog == "gh" and len(args) >= 2 and args[0] == "gist" and args[1] == "create":
        for a in args[2:]:
            if not a.startswith("-"):
                p = resolve(a, cwd)
                if EXFIL_SECRET_PATH_RE.search(p) or CRED_STORE_FILE_RE.search(p):
                    return "gh gist create publishes secret file %s" % os.path.basename(p)
    return None

PASTE_OAST_RE = re.compile(
    r"\b(?:webhook\.site|requestbin|pipedream|paste\.rs|pastebin|hastebin|termbin|"
    r"transfer\.sh|file\.io|0x0\.st|tmpfiles\.org|oastify\.com|interact\.sh|oast\.fun|"
    r"burpcollaborator\.net|canarytokens\.com|hookbin|pasty\.)\b", re.I)

def check_paste_tunnel_exfil(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    if prog in EGRESS_HTTP_CLIENTS or prog in ("nc", "ncat", "socat"):
        if PASTE_OAST_RE.search(text):
            has_body = any(a in ("-d", "--data", "--data-binary", "--data-raw", "-F", "--form", "-T") or
                           a.startswith(("-d", "--data", "-F", "-T")) for a in toks[1:])
            has_subst = "$(" in text or "`" in text or "<" in text
            if has_body or has_subst:
                m = PASTE_OAST_RE.search(text)
                return "%s sends data to paste/OAST service (%s)" % (prog, m.group(0))
    return None

DNS_CLIENTS = {"dig", "nslookup", "host", "drill"}
def check_dns_exfil(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    if prog in DNS_CLIENTS:
        if "__SUBST__" in text or "$(" in text or "`" in text:
            return "%s executes command substitution in DNS query label" % prog
        for a in toks[1:]:
            if not a.startswith("-") and re.search(r"^[A-Za-z0-9+/]{30,}={0,2}\.", a):
                return "%s executes base64-encoded DNS exfiltration" % prog
    return None

def check_registry_config_write(toks, text, cfg):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    trusted = {"registry.npmjs.org", "registry.yarnpkg.com", "pypi.org", "pythonhosted.org"}
    if prog in ("npm", "pnpm", "bun") and len(args) >= 3 and args[0] == "config" and args[1] == "set" and args[2] == "registry":
        if len(args) >= 4 and not any(h in args[3] for h in trusted):
            return "%s config set overrides registry with untrusted %s" % (prog, args[3])
    if prog == "yarn" and len(args) >= 3 and args[0] == "config" and args[1] == "set" and "Registry" in args[2]:
        if len(args) >= 4 and not any(h in args[3] for h in trusted):
            return "yarn config set overrides registry with untrusted %s" % args[3]
    if prog in ("pip", "pip3", "uv") and len(args) >= 3 and args[0] == "config" and args[1] == "set" and "index-url" in args[2]:
        if len(args) >= 4 and not any(h in args[3] for h in trusted):
            return "%s config set overrides index with untrusted %s" % (prog, args[3])
    return None

LOADER_VARS = {"LD_PRELOAD", "LD_AUDIT", "LD_PROFILE", "DYLD_INSERT_LIBRARIES", "PYTHONSTARTUP", "BASH_ENV", "ENV", "DOTNET_STARTUP_HOOKS"}
SANITIZERS = {"libasan", "libubsan", "libtsan", "liblsan", "libeatmydata", "libfaketime", "libjemalloc", "libgmalloc"}
def check_loader_inject(name, val, cwd):
    if name in LOADER_VARS and val:
        if name == "ENV":
            # Allow single tokens without / or . (ENV=prod, ENV=dev)
            if "/" not in val and not val.startswith((".", "~")) and " " not in val:
                return None
            return "sets shell startup hook ENV=%s" % val[:40]
        if name == "BASH_ENV":
            return "sets shell startup script BASH_ENV=%s" % val[:40]
        if name == "PYTHONSTARTUP":
            p = resolve(val, cwd)
            root = git_top(p) or cwd
            if is_inside(p, cwd) or (root and is_inside(p, root)):
                return None
            return "PYTHONSTARTUP points outside workspace (%s)" % val[:40]
        if name == "DOTNET_STARTUP_HOOKS":
            p = resolve(val, cwd)
            if is_inside(p, cwd):
                return None
            return "DOTNET_STARTUP_HOOKS points outside workspace (%s)" % val[:40]
        base = os.path.basename(val).lower()
        if any(s in base for s in SANITIZERS):
            p = resolve(val, cwd)
            if not is_scratch(p) and (is_inside(p, cwd) or any(p.startswith(pfx) for pfx in ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/Library/Developer", "/opt/homebrew", "/usr/local/lib"))):
                return None
        return "sets dynamic loader injection variable %s=%s" % (name, val[:40])
    return None

def check_interpreter_hook(name, val, cwd):
    if name == "NODE_OPTIONS" and val:
        opts = val.split()
        for idx, o in enumerate(opts):
            if o in ("-r", "--require", "--import", "--loader", "--experimental-loader", "-e", "--eval") or any(o.startswith(pfx) for pfx in ("--require=", "--import=", "--loader=")):
                return "NODE_OPTIONS uses code-loading option %s" % o
            if o in ("--inspect", "--inspect-brk") and idx + 1 < len(opts):
                target = opts[idx + 1]
                if not any(h in target for h in ("127.0.0.1", "localhost", "[::1]")):
                    return "NODE_OPTIONS binds debugger to non-loopback address"
            if o.startswith(("--inspect=", "--inspect-brk=")):
                target = o.split("=", 1)[1]
                if not any(h in target for h in ("127.0.0.1", "localhost", "[::1]")) and ":" in target:
                    return "NODE_OPTIONS binds debugger to non-loopback address"
    if name == "PERL5OPT" and val and any(re.search(r"^-M", f) or re.search(r"^-m", f) for f in val.split()):
        return "PERL5OPT loads module via -M"
    if name == "RUBYOPT" and val and any(f in ("-r", "--require") or f.startswith("-r") for f in val.split()):
        return "RUBYOPT loads code via -r"
    if name in ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS") and val:
        if any(f in val for f in ("-javaagent:", "-agentlib:", "-agentpath:", "-Xbootclasspath")):
            return "%s loads Java agent code" % name
    if name == "GOFLAGS" and val and "-toolexec" in val:
        return "GOFLAGS uses -toolexec wrapper"
    if name in ("RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER") and val:
        p = resolve(val, cwd)
        base = os.path.basename(p).lower()
        if not (base.startswith("sccache") or base.startswith("cachepot")):
            return "%s points to unapproved wrapper %s" % (name, base)
    if name == "GIT_EXEC_PATH" and val:
        p = resolve(val, cwd)
        if not is_inside(p, cwd) and not any(p.startswith(pfx) for pfx in ("/usr/lib/git-core", "/usr/libexec/git-core", "/opt/homebrew", "/Library/Developer")):
            return "GIT_EXEC_PATH points to foreign directory %s" % val
    if name == "OPENSSL_CONF" and val:
        p = resolve(val, cwd)
        if not any(p.startswith(pfx) for pfx in ("/etc/ssl", "/etc/pki", "/private/etc/ssl", "/usr/lib/ssl", "/opt/homebrew/etc/openssl", "/usr/local/etc/openssl")):
            return "OPENSSL_CONF points to non-system config %s" % val
    if name == "PYTHONINSPECT" and val and val.lower() not in ("0", "false", "no", "off"):
        return "PYTHONINSPECT forces interactive shell entry"
    return None

SEARCH_PATH_VARS = {"LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "PYTHONPATH", "NODE_PATH", "PERL5LIB", "RUBYLIB"}
TRUSTED_SEARCH_PREFIXES = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt/homebrew", "/opt/local", "/usr/local", "/nix/store", "/Library/Developer", "/Library/Frameworks", "/System")

def check_search_path_escape(name, val, cwd):
    if name in SEARCH_PATH_VARS and val:
        for part in val.split(os.pathsep):
            if not part or part in (".", "./src", "src"):
                continue
            if part.startswith(("$", "${")):
                continue
            p = resolve(part, cwd)
            if is_scratch(p) or part.startswith(("/tmp", "/var/tmp", "/dev/shm", "/var/folders")):
                return "%s adds untrusted scratch directory %s" % (name, part)
            root = git_top(p) or cwd
            if is_inside(p, cwd) or (root and is_inside(p, root)):
                continue
            if not any(p.startswith(tp) for tp in TRUSTED_SEARCH_PREFIXES) and not any(part.startswith(pfx) for pfx in ("~/.local/bin", "~/.cargo/bin", "~/.rustup", "~/.nvm", "~/.pyenv", "~/.bun", "~/go/bin")):
                return "%s points outside workspace and trusted paths (%s)" % (name, part)
    if name == "PATH" and val:
        parts = val.split(os.pathsep)
        if "$PATH" not in val and "${PATH}" not in val and not any(b in parts for b in ("/usr/bin", "/bin")):
            return "PATH drops standard system binaries (/usr/bin)"
        for part in parts:
            if part in (".", ""):
                return "PATH adds current directory (.) to executable search path"
            p = resolve(part, cwd)
            if is_scratch(p) or part.startswith(("/tmp", "/var/tmp", "/dev/shm", "/var/folders")):
                return "PATH adds untrusted scratch directory to search path"
    if name == "ZDOTDIR" and val:
        p = resolve(val, cwd)
        if p != HOME and not is_inside(p, cwd):
            return "ZDOTDIR overrides zsh configuration directory (%s)" % val
    return None

DEBUG_PROGS = {"gdb", "lldb", "strace", "dtruss", "dtrace", "frida", "procdump"}
def check_foreign_debug(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if any(a in ("--help", "-h", "--version", "-v") for a in args):
        return None
    if prog in ("gdb", "lldb"):
        if any(a in ("-p", "--pid", "--attach-pid", "--attach", "--waitfor") or a.startswith(("--pid=", "--attach-pid=")) for a in args):
            return "%s attaches to foreign running process" % prog
        if any(a in ("-ex", "-x", "--eval-command", "-o") for a in args) and any("attach" in a for a in args):
            return "%s attaches to process via command script" % prog
    if prog in ("strace", "dtruss", "dtrace") and any(a in ("-p", "--attach") or a.startswith("-p") for a in args):
        return "%s attaches to foreign running process" % prog
    if prog == "frida" and any(a in ("-p", "--pid", "-n", "--attach") for a in args):
        return "frida attaches to foreign process"
    if prog in ("gcore", "procdump") and any(a.isdigit() for a in args):
        return "%s dumps memory of running process" % prog
    if prog in ("sample", "vmmap", "heap", "leaks") and len(args) >= 1 and args[0].isdigit():
        return "%s reads memory of foreign process %s" % (prog, args[0])
    return None

PERSIST_CTRL_CMDS = {
    "launchctl": {"load", "bootstrap", "bootout", "enable", "disable", "submit", "kickstart", "setenv"},
    "systemctl": {"enable", "disable", "link", "mask", "preset", "edit"},
    "at": set(),
    "batch": set(),
    "schtasks": {"/create", "/change", "/run"}
}
def check_persistence_control(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = [a.lower() for a in toks[1:] if not a.startswith("-")]
    raw_args = toks[1:]
    if any(a in ("--help", "-h", "--version", "-v", "status", "list", "show", "cat") for a in raw_args):
        return None
    if prog == "crontab":
        if raw_args == ["-l"]:
            return None
        return "crontab mutates user cron table"
    if prog == "at":
        if any(a in ("-l", "-c", "atq") for a in raw_args):
            return None
        return "at schedules delayed background task"
    if prog == "batch":
        return "batch schedules background queue execution"
    if prog in PERSIST_CTRL_CMDS:
        acts = PERSIST_CTRL_CMDS[prog]
        if not acts or any(a in acts for a in args) or any(a in acts for a in raw_args):
            return "%s installs persistent daemon or job" % prog
    if prog == "osascript" and any(k in " ".join(raw_args).lower() for k in ("login item", "make new login item")):
        return "osascript creates persistent login item"
    if prog == "reg" and len(args) >= 1 and args[0] == "add":
        if any(re.search(r"\\Software\\Microsoft\\Windows\\CurrentVersion\\(?:Run|RunOnce)", a, re.I) for a in raw_args):
            return "reg add installs Windows startup persistence key"
    return None

PERSIST_PATHS = (
    "/Library/LaunchAgents", "/Library/LaunchDaemons", "/etc/systemd/system", "/usr/lib/systemd/system",
    "/etc/crontab", "/etc/anacrontab", "/etc/cron.", "/var/spool/cron", "/etc/environment"
)
def check_persistence_file(path, cwd):
    if not path:
        return None
    p = resolve(path, cwd)
    if is_inside(p, cwd):
        return None
    if any(k in p for k in PERSIST_PATHS) or any(p.startswith(os.path.expanduser(k)) for k in ("~/Library/LaunchAgents", "~/.config/systemd/user", "~/.pam_environment")):
        return "writes persistent startup configuration (%s)" % os.path.basename(p)
    return None

OS_TAMPER_CMDS = {
    "csrutil": {"disable", "authenticated-root"},
    "spctl": {"--master-disable", "--global-disable"},
    "setenforce": {"0", "permissive"},
    "aa-disable": set(),
    "pfctl": {"-d", "-f"},
    "ufw": {"disable"}
}
def check_os_trust_tamper(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in OS_TAMPER_CMDS:
        targets = OS_TAMPER_CMDS[prog]
        if not targets or any(a in targets for a in args):
            return "%s disables operating system security control" % prog
    if prog in ("iptables", "ip6tables") and (any(a in ("-F", "--flush", "-X", "--delete-chain") for a in args) or ("-P" in args and "ACCEPT" in args)):
        return "%s flushes or permits all firewall traffic" % prog
    if prog == "nft" and any(a == "flush" for a in args):
        return "nft flushes packet filter ruleset"
    if prog in ("insmod", "kextload"):
        return "%s loads kernel extension or module" % prog
    if prog == "kmutil" and any(a == "load" for a in args):
        return "kmutil loads kernel extension"
    if prog == "modprobe" and not any(a in ("-n", "--dry-run", "--show-depends", "-c", "-r") for a in args):
        return "modprobe loads kernel module"
    if prog == "visudo" and "-c" not in args:
        return "visudo edits system sudoers configuration"
    return None

def check_trust_store_inject(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "security" and any(a in ("add-trusted-cert", "trust-settings-import") for a in args):
        return "security installs certificate authority into trust store"
    if prog in ("update-ca-certificates", "update-ca-trust") and not any(a in ("--help", "-h") for a in args):
        return "%s rebuilds system root certificates" % prog
    if prog == "certutil" and any(a in ("-addstore",) for a in args) and any(a.lower() in ("root", "authroot", "trust") for a in args):
        return "certutil adds certificate to root certificate store"
    return None

def check_quarantine_strip(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "xattr":
        if any(a == "com.apple.quarantine" for a in args) and any(a in ("-d",) for a in args):
            return "xattr strips com.apple.quarantine security attribute"
        if any(a in ("-c", "-cr") for a in args):
            for a in args:
                if not a.startswith("-"):
                    p = resolve(a, cwd)
                    if not is_inside(p, cwd) or any(p.endswith(ext) for ext in (".app", ".dmg", ".pkg", ".command", ".tool")):
                        return "xattr clears all quarantine attributes from %s" % os.path.basename(p)
    return None

def check_inbound_exec(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "socat" and any("EXEC:" in a or "SYSTEM:" in a for a in args):
        return "socat exposes execution shell on network listener"
    if prog in ("ncat", "nc", "netcat") and (any(a in ("--exec", "--sh-exec", "-e", "-c") for a in args) or any(a.startswith(("--exec=", "--sh-exec=")) for a in args)):
        return "%s spawns interactive shell for incoming connections" % prog
    return None

TUNNEL_PROGS = {"ngrok", "localtunnel", "lt", "bore", "serveo", "inlets", "zrok", "chisel", "frp", "frpc", "sshuttle"}
def check_remote_tunnel(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("ssh", "autossh"):
        if any(a in ("-R", "-w") or a.startswith(("-R", "-w")) for a in args):
            return "ssh opens remote port forwarding tunnel"
        if any(a in ("-D",) or a.startswith("-D") for a in args):
            # check if not loopback
            for idx, a in enumerate(args):
                val = a[2:] if a.startswith("-D") and len(a) > 2 else (args[idx+1] if a == "-D" and idx+1 < len(args) else "")
                if val and not any(h in val for h in ("127.0.0.1", "localhost", "[::1]")) and ":" in val:
                    return "ssh opens non-loopback dynamic SOCKS proxy"
        if any(a.startswith("-oGatewayPorts=yes") or a.startswith("-oGatewayPorts=clientspecified") for a in args):
            return "ssh opens gateway ports to network"
    if prog == "tailscale" and any(a in ("funnel", "serve") for a in args):
        return "tailscale opens public funnel or port serve"
    if prog == "cloudflared" and any(a in ("tunnel",) for a in args) and not any(a in ("login", "version", "--help") for a in args):
        return "cloudflared opens public network tunnel"
    if prog in TUNNEL_PROGS:
        return "%s starts reverse network tunnel" % prog
    return None

RESOLVER_FILES = {"/etc/hosts", "/private/etc/hosts", "/etc/resolv.conf", "/private/etc/resolv.conf"}
def check_resolver_tamper(toks, path):
    if path and any(path == r or path.endswith(r) for r in RESOLVER_FILES):
        return "modifies system name resolution file (%s)" % path
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "networksetup" and any(a in ("-setdnsservers", "-setsearchdomains") for a in args):
        return "networksetup reconfigures system DNS servers"
    if prog in ("route", "ip") and any(a in ("add", "del", "delete", "replace", "change") for a in args):
        return "%s modifies network routing tables" % prog
    return None

PRIVATE_KEY_NAMES = re.compile(r"(?:^|/)(?:id_rsa|id_ecdsa|id_ed25519|id_dsa|ssh_host_[^/]+_key|\.gnupg/private-keys-v1\.d/.*|secring\.gpg)$")
def check_private_key_material(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    if prog in READ_OR_COPY_PROGS:
        for a in toks[1:]:
            if not a.startswith("-") and not a.endswith(".pub"):
                p = resolve(a, cwd)
                if (PRIVATE_KEY_NAMES.search(p) or "/.gnupg/private-keys" in p or "/.config/age" in p) and not is_scratch(p):
                    if not any(t in p for t in ("/tests/", "/fixtures/", "/testdata/")):
                        return "%s accesses private key material (%s)" % (prog, os.path.basename(p))
    if prog == "ssh-keygen" and any(a in ("-p", "--change-passphrase") for a in toks[1:]):
        return "ssh-keygen modifies private key passphrase"
    return None

def check_secret_export(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "gpg" and any(a in ("--export-secret-keys", "--export-secret-subkeys") for a in args):
        return "gpg exports private secret keys"
    if prog == "cast" and len(args) >= 2 and args[0] == "wallet" and args[1] in ("private-key", "pk", "decrypt-keystore", "dk", "derive"):
        return "cast wallet exports Ethereum private key"
    if prog == "bitcoin-cli" and any(a in ("dumpprivkey", "dumpwallet") for a in args):
        return "bitcoin-cli dumps crypto wallet private key"
    if prog == "solana-keygen" and any(a in ("recover",) for a in args):
        return "solana-keygen exports wallet secret seed"
    if prog == "geth" and len(args) >= 2 and args[0] == "account" and args[1] == "export":
        return "geth account export dumps private key"
    return None

def check_ssh_exec_config(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("ssh", "scp", "sftp"):
        for idx, a in enumerate(args):
            val = None
            if a.startswith("-o") and len(a) > 2:
                val = a[2:]
            elif a == "-o" and idx + 1 < len(args):
                val = args[idx + 1]
            if val:
                for k in ("ProxyCommand", "LocalCommand", "RemoteCommand"):
                    if val.startswith(k + "=") and val != "ProxyCommand=none":
                        return "%s executes local command via SSH %s" % (prog, k)
            if a in ("-A",) or (val and val == "ForwardAgent=yes"):
                return "%s enables SSH agent forwarding" % prog
    return None

AGENT_PROGS = {"claude", "codex", "aider", "cursor-agent", "openclaw", "opencode", "gemini", "goose", "amp"}
def check_agent_bypass_spawn(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in AGENT_PROGS or (prog in ("python", "python3") and any(a in AGENT_PROGS for a in args)):
        if any(a in ("--dangerously-skip-permissions", "--dangerously-bypass-approvals-and-sandbox", "--yolo", "--full-auto", "--yes-always") for a in args) or \
           any(a.startswith("--permission-mode=bypass") or a.startswith("--permission-mode=dontAsk") or a in ("bypassPermissions", "dontAsk") for a in args) or \
           ("--ask-for-approval" in args and "never" in args and "--sandbox" in args and "danger-full-access" in args):
            return "spawns %s agent with safety permissions and approval bypassed" % prog
        if prog == "aider" and "--yes" in args:
            return "spawns aider with --yes automatic execution"
    return None

def check_local_model_mutate(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "ollama" and any(a in ("rm", "delete", "push", "create", "cp") for a in args):
        return "ollama mutates or deletes local AI model"
    if prog in EGRESS_HTTP_CLIENTS:
        text = " ".join(toks)
        if any(p in text for p in (":11434", ":1234")) and any(ep in text for ep in ("/api/delete", "/api/push", "/api/create", "/api/blobs", "/v1/models")):
            if any(m in text for m in ("DELETE", "POST", "-X DELETE", "-X POST", "-XDELETE")):
                return "%s mutates model repository on local model API" % prog
    return None

MIGRATION_PATH_RE = re.compile(r"(?:^|/)(?:migrations/|db/migrate/|alembic/versions/|prisma/migrations/|drizzle/|supabase/migrations/).*\.(?:sql|py)$")
def check_migration_rewrite(path, cwd):
    if not path or not MIGRATION_PATH_RE.search(path):
        return None
    if path.endswith("__init__.py"):
        return None
    p = resolve(path, cwd)
    root = git_top(p)
    if root:
        rel = os.path.relpath(p, root)
        out = git_out(root, ["cat-file", "-e", "HEAD:" + rel])
        if out is not None:
            return "modifies applied database migration %s tracked in HEAD" % os.path.basename(p)
    return None

LOCKFILE_NAMES = {"package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb", "Cargo.lock", "uv.lock", "poetry.lock", "Pipfile.lock", "composer.lock", "Gemfile.lock", "go.sum"}
def check_lockfile_destroy(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("rm", "unlink") or (prog == "git" and "rm" in args):
        for a in args:
            base = os.path.basename(a)
            if base in LOCKFILE_NAMES:
                p = resolve(a, cwd)
                root = git_top(p)
                if root:
                    rel = os.path.relpath(p, root)
                    out = git_out(root, ["cat-file", "-e", "HEAD:" + rel])
                    if out is not None:
                        return "deletes tracked package lockfile %s" % base
    return None

def check_history_rewrite(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("git-filter-repo", "bfg"):
        return "%s rewrites git history" % prog
    if prog == "git":
        if any(a in ("filter-repo", "filter-branch", "replace") for a in args):
            return "git %s rewrites git commit history" % args[0]
    return None

def check_ref_clobber(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "git":
        if any(a == "update-ref" for a in args) and any(a in ("-d", "--delete") for a in args):
            return "git update-ref deletes git reference"
        if any(a == "tag" for a in args) and any(a in ("-d", "--delete") for a in args):
            return "git tag -d deletes git tag"
    return None

def check_cluster_drain(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if any(a in ("--help", "-h", "--dry-run", "--dry-run=server", "--dry-run=client") for a in args):
        return None
    if prog in ("kubectl", "oc"):
        if any(a in ("drain", "cordon") for a in args):
            return "kubectl %s evicts workloads from node" % args[0]
        if any(a == "delete" for a in args) and any(a in ("--force",) for a in args) and any(a.startswith("--grace-period=0") or a == "0" for a in args):
            return "kubectl delete --force --grace-period=0 forcibly terminates pods"
    return None

def check_auto_approve_apply(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("terraform", "tofu", "terragrunt") and "apply" in args:
        if any(a in ("-auto-approve", "--auto-approve") for a in args):
            return "%s apply --auto-approve skips infrastructure confirmation" % prog
    if prog == "pulumi" and "up" in args:
        if any(a in ("--yes", "-y", "--skip-preview") for a in args):
            return "pulumi up --yes skips deployment review"
    if prog == "cdk" and "deploy" in args:
        if any(a.startswith("--require-approval=never") or a == "--require-approval never" for a in args):
            return "cdk deploy --require-approval never bypasses security prompts"
    return None

def check_platform_teardown(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "azd" and any(a in ("down",) for a in args):
        return "azd down tears down Azure application architecture"
    if prog == "wrangler" and any(a in ("delete",) for a in args) and not any(a in ("key",) for a in args):
        return "wrangler delete tears down Cloudflare worker or storage"
    if prog == "hcloud" and any(a in ("delete",) for a in args):
        return "hcloud delete destroys cloud resources"
    if prog == "linode-cli" and any(a in ("delete",) for a in args):
        return "linode-cli delete destroys cloud infrastructure"
    return None

def check_schema_clean(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "flyway" and any(a == "clean" for a in args):
        return "flyway clean drops all schema objects"
    if prog == "liquibase" and any(a in ("dropall", "drop-all") for a in [x.lower() for x in args]):
        return "liquibase dropAll drops all database tables"
    if prog == "atlas" and any(a == "clean" for a in args) and any(a == "schema" for a in args):
        return "atlas schema clean wipes database schema"
    return None

def check_github_vandalism(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "gh":
        # filter out flags like --repo owner/name
        clean_args = []
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if a in ("-R", "--repo"):
                skip_next = True
                continue
            if a.startswith(("-R=", "--repo=")):
                continue
            clean_args.append(a)
        if len(clean_args) >= 2 and clean_args[0] in ("issue", "pr", "discussion") and clean_args[1] in ("delete", "lock", "transfer"):
            return "gh %s %s vandalizes public repository tracker" % (clean_args[0], clean_args[1])
        if any(a in ("-X", "--method") for a in args) or any(a.startswith(("-X", "--method=")) for a in args):
            if any(a in ("DELETE", "-XDELETE", "--method=DELETE") for a in args):
                if any(k in " ".join(args) for k in ("/collaborators/", "/hooks/", "/keys/", "/invitations/")):
                    return "gh api DELETE modifies repository access permissions"
    return None

OUTBOUND_BLAST_HOSTS = {"api.sendgrid.com", "api.resend.com", "api.postmarkapp.com", "api.mailgun.net", "api.twilio.com"}
def check_outbound_blast(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("twilio", "sendgrid", "resend", "postmark", "mailgun"):
        if any(a in ("send", "messages:create", "emails:send", "sms") for a in args):
            return "%s sends outbound broadcast communication" % prog
    if prog == "aws" and len(args) >= 2 and args[0] == "ses" and args[1].startswith("send-"):
        return "aws ses %s broadcasts outbound email" % args[1]
    if prog in ("git-send-email", "sendmail", "mailx"):
        return "%s sends outbound email transmission" % prog
    if prog in EGRESS_HTTP_CLIENTS and any(h in text for h in OUTBOUND_BLAST_HOSTS):
        return "%s sends outbound email/SMS through production API" % prog
    return None

def check_gpu_burn(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if any(a in ("--help", "-h", "--dry-run") for a in args):
        return None
    if prog == "aws" and "run-instances" in args:
        if any(re.search(r"^(?:p|g|inf|trn)\d", a) or any(k in a.lower() for k in ("a100", "h100", "a10g")) for a in args):
            return "aws ec2 run-instances launches expensive GPU compute cluster"
    if prog == "modal" and any(a in ("deploy", "run", "serve") for a in args) and any(a == "--gpu" or a.startswith("--gpu=") for a in args):
        return "modal %s provisions GPU resources" % args[0]
    if prog in ("runpodctl", "vastai") and any(a in ("create", "start") for a in args):
        return "%s provisions paid GPU instances" % prog
    return None

CHECKER_CONFIG_NAMES = {
    "tsconfig.json", "tsconfig.node.json", "tsconfig.app.json",
    "ruff.toml", ".ruff.toml", "biome.json", "biome.jsonc",
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml",
    "mypy.ini", ".mypy.ini", "pyproject.toml"
}
def check_checker_delete(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("rm", "unlink") or (prog == "git" and "rm" in args):
        for a in args:
            base = os.path.basename(a)
            if base in CHECKER_CONFIG_NAMES:
                p = resolve(a, cwd)
                root = git_top(p)
                if root:
                    rel = os.path.relpath(p, root)
                    out = git_out(root, ["cat-file", "-e", "HEAD:" + rel])
                    if out is not None:
                        return "deletes tracked typechecker/linter config %s" % base
                else:
                    return "deletes typechecker/linter config %s" % base
    return None

def check_public_bucket_acl(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "aws":
        if "s3api" in args:
            if any(a in ("put-bucket-acl", "put-object-acl") for a in args):
                if any(k in args for k in ("public-read", "public-read-write", "--acl=public-read", "--acl=public-read-write")) or \
                   any("AllUsers" in a for a in args):
                    return "aws s3api sets world-readable public bucket/object ACL"
            if "delete-public-access-block" in args:
                return "aws s3api removes public access block protection"
            if "put-public-access-block" in args and any(k in " ".join(args) for k in ("BlockPublicAcls=false", "RestrictPublicBuckets=false")):
                return "aws s3api disables public access block controls"
        if "s3" in args and any(a in ("cp", "sync") for a in args):
            if any(k in args for k in ("--acl public-read", "--acl=public-read", "public-read")):
                return "aws s3 copies files with public-read ACL"
    if prog == "gsutil" and "iam" in args and "ch" in args:
        if any(a.startswith(("allUsers:", "allAuthenticatedUsers:")) for a in args) and "-d" not in args:
            return "gsutil iam ch grants public access to bucket"
    if prog == "gcloud" and "add-iam-policy-binding" in args:
        if any(a.startswith(("--member=allUsers", "--member=allAuthenticatedUsers")) or a in ("allUsers", "allAuthenticatedUsers") for a in args):
            return "gcloud adds public allUsers IAM policy binding"
    if prog == "az" and "storage" in args and any(a.startswith("--public-access=") or a == "--public-access" for a in args):
        if any(k in " ".join(args) for k in ("blob", "container")):
            return "az storage enables public container/blob access"
    return None

def check_cloud_iam_mutation(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "aws":
        if "iam" in args:
            if any(a in ("create-access-key", "create-user", "create-login-profile", "attach-user-policy", "attach-role-policy", "put-user-policy", "delete-user", "delete-role") for a in args):
                return "aws iam executes high-privilege IAM identity mutation"
        if "kms" in args and any(a in ("schedule-key-deletion", "disable-key") for a in args):
            return "aws kms schedules deletion or disables encryption key"
        if "organizations" in args and any(a in ("leave-organization", "close-account") for a in args):
            return "aws organizations leaves organization or closes account"
    if prog == "gcloud":
        if "service-accounts" in args and "keys" in args and any(a in ("create", "delete") for a in args):
            return "gcloud service-accounts keys creates or deletes service account key"
        if any(a in ("add-iam-policy-binding", "remove-iam-policy-binding") for a in args):
            return "gcloud mutates project or organization IAM policy binding"
    if prog == "az":
        if "ad" in args and "app" in args and "credential" in args and "reset" in args:
            return "az ad app credential reset mutates app secret"
        if "role" in args and "assignment" in args and any(a in ("create", "delete") for a in args):
            return "az role assignment mutates cloud role binding"
    if prog == "vault":
        if "policy" in args and any(a in ("write", "delete") for a in args):
            return "vault policy mutates authorization policy"
        if "operator" in args and any(a in ("init", "rekey") for a in args):
            return "vault operator mutates root vault keys"
    return None

def check_dns_tls_prod_change(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "aws":
        if "route53" in args and any(a in ("delete-hosted-zone", "change-resource-record-sets") for a in args):
            return "aws route53 mutates or deletes production DNS zone records"
        if "acm" in args and "delete-certificate" in args:
            return "aws acm deletes TLS certificate"
        if "cloudfront" in args and "delete-distribution" in args:
            return "aws cloudfront deletes CDN distribution"
    if prog == "gcloud" and "dns" in args:
        if any(a in ("delete",) for a in args) or ("transaction" in args and "execute" in args):
            return "gcloud dns deletes or publishes production DNS changes"
    if prog == "az" and "network" in args and "dns" in args and "delete" in args:
        return "az network dns deletes DNS zone or record-set"
    if prog == "certbot" and any(a in ("revoke", "delete") for a in args):
        return "certbot revokes or deletes TLS certificate"
    if prog == "acme.sh" and any(a in ("--revoke", "--remove") for a in args):
        return "acme.sh revokes or removes TLS certificate"
    return None

def check_cloud_secret_destroy(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "aws":
        if "secretsmanager" in args and any(a in ("delete-secret", "put-secret-value") for a in args):
            return "aws secretsmanager deletes or force-overwrites secret"
        if "ssm" in args and (any(a in ("delete-parameter", "delete-parameters") for a in args) or ("put-parameter" in args and "--overwrite" in args)):
            return "aws ssm deletes or force-overwrites parameter store secret"
    if prog == "gcloud" and "secrets" in args and any(a in ("delete", "destroy") for a in args):
        return "gcloud secrets destroys secret or version"
    if prog == "az" and "keyvault" in args:
        if any(a in ("delete", "purge") for a in args):
            return "az keyvault deletes or purges vault secret or key"
    if prog == "vercel" and len(args) >= 2 and args[0] == "env" and args[1] in ("rm", "remove"):
        return "vercel env rm deletes environment secret"
    if prog == "railway" and len(args) >= 2 and args[0] in ("variable", "variables") and args[1] == "delete":
        return "railway variable delete deletes production secret"
    if prog == "heroku" and len(args) >= 1 and args[0] == "config:unset":
        return "heroku config:unset deletes production config var"
    if prog == "gh" and len(args) >= 2 and args[0] in ("secret", "variable") and args[1] == "delete":
        return "gh secret delete removes repository secret"
    if prog == "doppler" and "secrets" in args and "delete" in args:
        return "doppler secrets delete removes managed secret"
    if prog == "vault" and "kv" in args and any(a in ("destroy", "delete") for a in args):
        return "vault kv destroys vault secret key"
    return None

def check_tls_disable(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("curl", "wget"):
        if any(a in ("-k", "--insecure", "--no-check-certificate") for a in args):
            # check URL target
            for a in args:
                if not a.startswith("-") and (a.startswith("http://") or a.startswith("https://")):
                    if not any(h in a for h in ("localhost", "127.0.0.1", "[::1]")):
                        return "%s connects to remote host with TLS verification disabled" % prog
    if prog in ("pip", "pip3", "uv") and any(a.startswith("--trusted-host") for a in args):
        return "%s installs packages with TLS verification bypassed (--trusted-host)" % prog
    if prog in ("npm", "yarn") and "config" in args and "set" in args:
        if any(a in ("strict-ssl=false", "enableStrictSsl=false") or (idx+1 < len(args) and args[idx+1].lower() in ("false", "0")) for idx, a in enumerate(args)):
            return "%s config set disables strict SSL validation" % prog
    if prog == "git" and "config" in args and any("http.sslverify=false" in a.lower() or (idx+1 < len(args) and args[idx+1].lower() in ("false", "0")) for idx, a in enumerate(args)):
        return "git config disables SSL verification"
    return None

def check_protect_local_state(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    LOCAL_STATE_NAMES = {".env", ".env.local", ".env.production", ".env.development", "secrets.json", "secret.json", "credentials.json"}
    LOCAL_STATE_EXTS = (".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".sqlite-wal", ".sqlite-shm", ".duckdb", ".duckdb.wal", ".pem", ".key", ".kdbx")
    if prog in ("rm", "unlink", "rmdir", "shred") or (prog == "git" and "rm" in args):
        for a in args:
            base = os.path.basename(a)
            if base in LOCAL_STATE_NAMES or any(base.endswith(ext) for ext in LOCAL_STATE_EXTS) or "uploads" in a:
                p = resolve(a, cwd)
                root = git_top(p)
                if root:
                    # check if git ignored
                    rel = os.path.relpath(p, root)
                    out = git_out(root, ["check-ignore", "-q", rel])
                    if out is not None or os.path.exists(p):
                        return "deletes gitignored local developer state (%s)" % base
    return None

def check_archive_clobber(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("tar", "gtar", "bsdtar"):
        if any(a in ("-x", "--extract", "-xf", "-xvf") or (a.startswith("-") and "x" in a and not any(k in a for k in ("t", "c", "O"))) for a in args):
            if any(a in ("-P", "--absolute-names") or (a.startswith("-") and "P" in a) for a in args):
                return "tar extracts archive with absolute paths (-P)"
            if "-k" not in args and "--keep-old-files" not in args and "--skip-old-files" not in args:
                for idx, a in enumerate(args):
                    if a in ("-C", "--directory") and idx + 1 < len(args):
                        dest = resolve(args[idx + 1], cwd)
                        if dest in ("/", HOME) or not is_inside(dest, cwd):
                            return "tar extracts archive into root or parent directory"
    if prog == "unzip" and "-o" in args and "-n" not in args:
        return "unzip overwrites existing files (-o) without keep flag"
    return None

def check_stream_wipe(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("truncate", "gtruncate") and any(a in ("-s", "--size") for a in args):
        for idx, a in enumerate(args):
            if a in ("-s", "--size") and idx + 1 < len(args) and args[idx+1] in ("0", "0B"):
                return "truncate empties existing file"
            if a.startswith(("-s0", "--size=0")):
                return "truncate empties existing file"
    if prog in ("cp", "install") and len(args) >= 2 and args[0] == "/dev/null":
        return "copies /dev/null over existing file"
    if prog == "dd" and any(a.startswith("of=") for a in args) and (any(a in ("if=/dev/null", "count=0") for a in args)):
        return "dd empties target file with count=0 or /dev/null"
    return None

def check_sync_delete(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if any(a in ("--dry-run", "--dryrun", "-n") for a in args):
        return None
    if prog == "rsync" and any(a.startswith(("--del", "--delete", "--remove-source-files")) for a in args):
        return "rsync executes mirror deletion without dry-run"
    if prog == "rclone" and any(a in ("sync", "bisync", "purge") for a in args):
        return "rclone executes mirror deletion without dry-run"
    if prog == "robocopy" and any(a in ("/MIR", "/PURGE") for a in args) and "/L" not in args:
        return "robocopy /MIR executes mirror purge deletion"
    return None

def check_backup_destroy(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "tmutil" and any(a in ("delete", "disable", "disablelocal", "thinlocalsnapshots", "deletelocalsnapshots") for a in args):
        return "tmutil destroys Time Machine backups or local snapshots"
    if prog == "diskutil" and len(args) >= 2 and args[0] == "apfs" and any(args[1].startswith(k) for k in ("deleteSnapshot", "deleteVolume", "deleteContainer")):
        return "diskutil apfs destroys APFS snapshots or container"
    if prog == "zfs" and any(a in ("destroy", "rollback") for a in args) and "-n" not in args:
        return "zfs destroys ZFS dataset or snapshot"
    if prog == "vssadmin" and "delete" in args and any(a in ("shadows", "shadowstorage") for a in args):
        return "vssadmin destroys Volume Shadow Copies"
    return None

def check_live_browser_debug(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    BROWSER_NAMES = {"google-chrome", "chrome", "chromium", "brave", "microsoft-edge", "msedge", "arc"}
    if prog in BROWSER_NAMES or (prog == "open" and any(b in " ".join(args) for b in ("Google Chrome", "Brave", "Edge", "Chromium"))):
        if any(a.startswith(("--remote-debugging-port", "--remote-debugging-pipe")) for a in args):
            # check if --user-data-dir is missing or points to live root
            has_isolated_dir = False
            for a in args:
                if a.startswith("--user-data-dir="):
                    val = a.split("=", 1)[1]
                    if val and not any(k in val for k in ("Application Support", ".config", "User Data")):
                        has_isolated_dir = True
            if not has_isolated_dir:
                return "%s attaches DevTools remote debugging port to live browser profile" % prog
    return None

def check_gui_inject(toks, text):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "cliclick" and not any(a in ("-h", "--help", "-V", "--version") for a in args):
        if not (len(args) == 1 and args[0].startswith("p")):
            return "cliclick synthesizes mouse clicks or keyboard events"
    if prog in ("xdotool", "ydotool", "wtype", "dotool", "xte"):
        if any(a in ("key", "type", "click", "mousedown", "mouseup", "mousemove") for a in args):
            return "%s injects synthetic keyboard or mouse events" % prog
    if prog == "tmux" and len(args) >= 2 and args[0] in ("send-keys", "send"):
        return "tmux send-keys injects keystrokes into terminal pane"
    if prog in ("autohotkey", "ahk") and not any(a in ("/?", "-h", "--help", "/version") for a in args):
        return "autohotkey runs automated GUI scripting"
    if prog == "osascript" and any(k in text.lower() for k in ("keystroke", "key code", "click button", "perform action")):
        return "osascript injects synthetic GUI keystrokes or clicks"
    return None

def check_sensor_capture(toks):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    CAPTURE_TOOLS = {"screencapture", "imagesnap", "grim", "scrot", "peek", "wf-recorder", "fswebcam",
                     "gnome-screenshot", "flameshot", "spectacle", "maim", "parecord", "parec", "pw-record", "rec"}
    if prog in CAPTURE_TOOLS and not any(a in ("--help", "-h", "--version", "-v", "-l") for a in args):
        return "%s captures screen, webcam, or microphone stream" % prog
    if prog == "ffmpeg":
        CAPTURE_FORMATS = {"avfoundation", "x11grab", "gdigrab", "dshow", "v4l2", "pulse", "alsa", "oss"}
        for idx, a in enumerate(args):
            if a == "-f" and idx + 1 < len(args) and args[idx + 1] in CAPTURE_FORMATS:
                return "ffmpeg captures hardware audio/video device (%s)" % args[idx + 1]
            if a in ("/dev/video0", "/dev/video1", "/dev/dsp") or a.startswith("/dev/video"):
                return "ffmpeg captures hardware camera device (%s)" % a
    return None

def check_clipboard_abuse(toks, fed_by):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    CLIPBOARD_READERS = {"pbpaste", "wl-paste", "get-clipboard"}
    if fed_by:
        up_prog, _ = fed_by
        if up_prog in CLIPBOARD_READERS:
            if prog in EGRESS_HTTP_CLIENTS or prog in ("nc", "ncat", "socat", "scp", "rsync"):
                return "pipes clipboard contents to remote network client (%s)" % prog
            if prog in ("sh", "bash", "zsh", "python", "node"):
                return "pipes clipboard contents into script interpreter execution"
    if prog == "pbcopy":
        text = " ".join(args)
        if any(k in text for k in ("id_rsa", "id_ed25519", "credentials", ".env")):
            return "pbcopy copies private key or credential material into system clipboard"
    return None

def check_extension_sideload(toks, path, cwd):
    if path:
        p = resolve(path, cwd)
        if any(p.startswith(os.path.expanduser(k)) for k in ("~/.vscode/extensions", "~/.cursor/extensions", "~/.config/nvim", "~/.vimrc")):
            if not is_inside(p, cwd):
                return "writes configuration or plugin to editor extension directory (%s)" % os.path.basename(p)
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("code", "code-insiders", "cursor", "cursor-insiders", "windsurf", "codium"):
        if any(a.startswith(("--install-extension", "--install-extension=")) for a in args):
            return "%s sideloads editor extension" % prog
    if prog in ("google-chrome", "chrome", "chromium", "brave", "microsoft-edge"):
        if any(a.startswith(("--load-extension", "--disable-extensions-except")) for a in args):
            return "%s sideloads browser extension" % prog
    return None



# --------------------------------------------------------------------------- #
# Unified Super-Kernel Detectors Part 2 (OP-038, OP-077..OP-081, OP-099, MCP)
# --------------------------------------------------------------------------- #
import os
import re
import sys
import json
import base64

def check_workflow_poison(path, content, cwd):
    if not path or not re.search(r"\.github/workflows/.*\.ya?ml$", path, re.I):
        return None
    if not content:
        return None
    # 1. pull_request_target + checkout head ref
    if "pull_request_target" in content:
        if re.search(r"ref:\s*['\"]?\${{\s*github\.event\.pull_request\.head\.(?:sha|ref)", content) or \
           re.search(r"ref:\s*['\"]?\${{\s*github\.head_ref", content):
            return "workflow combines pull_request_target with untrusted PR checkout"
    # 2. untrusted event field piped to GITHUB_ENV or GITHUB_PATH
    if any(k in content for k in ("$GITHUB_ENV", "$GITHUB_PATH", "${GITHUB_ENV}", "${GITHUB_PATH}")):
        if re.search(r"github\.event\.(?:issue|pull_request|comment|review|discussion)\.(?:title|body)", content) or \
           re.search(r"github\.event\.head_commit\.message", content):
            return "workflow pipes untrusted event field into runner environment"
    # 3. workflow_run artifact execution
    if "workflow_run" in content:
        if "download-artifact" in content and any(cmd in content for cmd in ("bash", "sh", "source", "chmod +x", "python", "node")):
            return "workflow executes unverified artifacts from workflow_run"
    return None

def check_unmonitored_agent_home(toks, env_vars, cwd):
    AGENT_HOMES = {
        "CLAUDE_CONFIG_DIR": ("~/.claude", "settings.json"),
        "CODEX_HOME": ("~/.codex", "hooks.json"),
        "CURSOR_CONFIG_DIR": ("~/.cursor", "hooks.json")
    }
    for var, (def_home, hook_file) in AGENT_HOMES.items():
        val = env_vars.get(var)
        if val:
            p = resolve(val, cwd)
            default_p = resolve(def_home, cwd)
            if os.path.realpath(p) != os.path.realpath(default_p):
                hf = os.path.join(p, hook_file)
                if not os.path.exists(hf):
                    return "%s points to unmonitored home lacking %s" % (var, hook_file)
                try:
                    with open(hf, "r", encoding="utf-8", errors="ignore") as f:
                        if "operator-gate" not in f.read():
                            return "%s points to agent home lacking operator-gate hook" % var
                except Exception:
                    return "%s points to unreadable agent configuration" % var
    return None

INSTRUCTION_GATE_RE = re.compile(r"(?:^|/)(?:CLAUDE\.md|AGENTS\.md|SKILL\.md|README\.md|\.cursor/rules/.*)$", re.I)
INSTRUCTION_OVERRIDE_PHRASES = (
    "IGNORE ALL PREVIOUS INSTRUCTIONS",
    "IGNORE PREVIOUS INSTRUCTIONS",
    "SYSTEM PROMPT OVERRIDE"
)

def check_instruction_inject(path, content):
    if not path or not INSTRUCTION_GATE_RE.search(path):
        return None
    if not content:
        return None
    upper = content.upper()
    for phrase in INSTRUCTION_OVERRIDE_PHRASES:
        if phrase in upper:
            return "injects instruction override phrase '%s' into %s" % (phrase, os.path.basename(path))
    # Check hidden HTML comments
    for m in re.finditer(r"<!--(.*?)-->", content, re.DOTALL):
        comment_text = m.group(1).lower()
        if any(k in comment_text for k in ("ignore previous", "system prompt", "do not tell the user", "bypass", "operator-gate", "claude_config_dir")):
            return "injects prompt override inside HTML comment into %s" % os.path.basename(path)
    # Check base64 blobs of at least 24 characters
    for m in re.finditer(r"[A-Za-z0-9+/]{24,}={0,2}", content):
        blob = m.group(0)
        try:
            decoded = base64.b64decode(blob).decode("utf-8", errors="ignore")
            if any(h in decoded for h in ("hooks", "command", "PreToolUse", "operator-gate")):
                if any(c in decoded for c in ("curl", "rm -rf", "| sh", "CLAUDE_CONFIG_DIR", "dangerously-skip")):
                    return "injects base64 payload into instruction file %s" % os.path.basename(path)
        except Exception:
            pass
    return None

GLOBAL_SKILL_DIRS = (
    "~/.claude/skills", "~/.claude/commands", "~/.claude/agents", "~/.claude/CLAUDE.md",
    "~/.cursor/skills", "~/.cursor/skills-cursor", "~/.cursor/agents", "~/.cursor/rules",
    "~/.codex/skills", "~/.codex/AGENTS.md", "~/.agents/skills"
)

def check_global_skill_write(path, cwd):
    if not path:
        return None
    p = resolve(path, cwd)
    root = git_top(p)
    # If it is inside the working tree repository, it is local repo configuration (allowed)
    if is_inside(p, cwd) or (root and is_inside(p, root)):
        return None
    for g in GLOBAL_SKILL_DIRS:
        target = resolve(g, cwd)
        if p == target or is_inside(p, target):
            return "mutates user-global agent configuration or custom skills (%s)" % os.path.basename(p)
    return None

def check_git_identity_spoof(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "git":
        # Check git -c overrides
        for idx, a in enumerate(args):
            val = None
            if a.startswith("-c") and len(a) > 2:
                val = a[2:]
            elif a == "-c" and idx + 1 < len(args):
                val = args[idx + 1]
            if val and any(val.startswith(k) for k in ("user.name=", "user.email=", "user.signingkey=")):
                k, v = val.split("=", 1)
                root = git_top(cwd) or cwd
                stored = git_out(root, ["config", "--get", k])
                if stored and stored.strip() != v.strip():
                    return "git -c overrides %s with conflicting identity %s" % (k, v)
            if val and val in ("commit.gpgsign=false", "tag.gpgSign=false"):
                root = git_top(cwd) or cwd
                stored = git_out(root, ["config", "--get", val.split("=")[0]])
                if stored and stored.strip().lower() in ("true", "1", "yes"):
                    return "git -c disables configured GPG commit signing"
        # Check git config set
        if "config" in args and not any(a in ("--get", "--get-all", "--list", "-l") for a in args):
            for k in ("user.name", "user.email", "user.signingkey"):
                if k in args:
                    root = git_top(cwd) or cwd
                    stored = git_out(root, ["config", "--get", k])
                    idx_k = args.index(k)
                    new_val = args[idx_k + 1] if idx_k + 1 < len(args) else ""
                    if stored and new_val and stored.strip() != new_val.strip():
                        return "git config overrides stored identity %s" % k
        # Check commit --author
        if "commit" in args:
            for idx, a in enumerate(args):
                val = None
                if a.startswith("--author="):
                    val = a.split("=", 1)[1]
                elif a == "--author" and idx + 1 < len(args):
                    val = args[idx + 1]
                if val:
                    if "$(" in val or "__SUBST__" in val:
                        continue
                    root = git_top(cwd) or cwd
                    stored_email = git_out(root, ["config", "--get", "user.email"])
                    if stored_email and stored_email.strip() and stored_email.strip() not in val:
                        return "git commit --author overrides configured author email"
            if any(a in ("--no-gpg-sign",) for a in args):
                root = git_top(cwd) or cwd
                stored = git_out(root, ["config", "--get", "commit.gpgsign"])
                if stored and stored.strip().lower() in ("true", "1", "yes"):
                    return "git commit --no-gpg-sign disables configured commit signing"
    return None

GOVERNANCE_FILES = {"CODEOWNERS", "LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md", "SECURITY.md"}
def check_governance_weaken(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog in ("rm", "unlink") or (prog == "git" and "rm" in args):
        for a in args:
            base = os.path.basename(a)
            if base in GOVERNANCE_FILES or a.endswith("/CODEOWNERS"):
                p = resolve(a, cwd)
                root = git_top(p)
                if root:
                    rel = os.path.relpath(p, root)
                    out = git_out(root, ["cat-file", "-e", "HEAD:" + rel])
                    if out is not None:
                        return "deletes tracked governance file %s" % base
                else:
                    return "deletes governance file %s" % base
    return None

def check_untrusted_package_source(toks, cwd):
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    args = toks[1:]
    if prog == "git" and len(args) >= 2 and args[0] == "submodule" and args[1] == "add":
        for a in args[2:]:
            if a.startswith("http://") and not any(h in a for h in ("localhost", "127.0.0.1")):
                return "git submodule add uses cleartext HTTP URL"
            if re.search(r"^https?://\d+\.\d+\.\d+\.\d+", a):
                return "git submodule add uses raw IP address"
    if prog in ("pip", "pip3", "npm", "yarn", "pnpm") and any(a in ("install", "add") for a in args):
        for a in args:
            if a.startswith("http://") and not any(h in a for h in ("localhost", "127.0.0.1")):
                return "%s installs package from cleartext HTTP URL" % prog
    return None

def check_mcp_control_plane(tool_name, tool_input):
    if not tool_name:
        return None
    # normalize namespace and verb
    name = tool_name
    args = tool_input or {}
    if name in ("CallDynamicTool", "CallMcpTool"):
        name = args.get("toolName") or args.get("name") or ""
        args = args.get("arguments") or args
    verb = name.split("__")[-1] if "__" in name else (name.split(":")[-1] if ":" in name else name)
    verb = verb.lower()

    # OP-112 outbound communications
    OUTBOUND_VERBS = {
        "send_message", "send_email", "send_mail", "send_imessage", "send_draft",
        "reply", "forward", "post_message", "chat_postmessage", "trash_thread",
        "trash_message", "delete_draft", "delete_message", "mark_thread_spam"
    }
    if verb in OUTBOUND_VERBS:
        return ("OP-112", "MCP tool %s sends outbound communication or destroys messages" % verb,
                "Outbound message transmission requires explicit owner review.")

    # OP-113 workspace destroy
    WORKSPACE_DESTROY_VERBS = {
        "delete_issue", "delete_project", "archive_project", "archive_repository",
        "merge_pull_request", "merge_pr", "dismiss_pull_request_review", "delete_branch",
        "delete_ref", "transfer_repository", "transfer_repo", "delete_repository", "delete_pull_request"
    }
    if verb in WORKSPACE_DESTROY_VERBS:
        return ("OP-113", "MCP tool %s executes destructive repository mutation" % verb,
                "Deleting repositories, issues, or merging PRs via MCP requires owner approval.")

    # OP-114 billing & cloud teardown
    BILLING_VERBS = {
        "create_charge", "create_payment_intent", "confirm_payment_intent", "capture_payment_intent",
        "create_payout", "create_refund", "cancel_subscription", "delete_customer",
        "delete_database", "delete_service", "delete_cluster", "delete_bucket", "delete_instance"
    }
    if verb in BILLING_VERBS:
        return ("OP-114", "MCP tool %s initiates financial charge or tears down infrastructure" % verb,
                "Financial charges and cloud resource teardown via MCP require owner approval.")

    return None



def hard_stop_effect_hits(e, cfg, cwd):
    hits, toks = [], e.get("toks") or []
    ecwd = e.get("cwd") or cwd
    text = e.get("text") or " ".join(toks)
    path = e.get("path") or (toks[1] if len(toks) > 1 and not toks[1].startswith("-") else None)
    fed_by = e.get("fed_by")

    escape = container_escape(toks, ecwd)
    if escape:
        hits.append(("OP-027", escape, "Keep host control sockets, namespaces and host roots outside agent containers."))
    if credential_dump(toks):
        hits.append(("OP-028", "requests credentials from an OS credential manager",
                     "The owner accesses the credential store directly; reference credentials by name only."))
    registry = untrusted_registry(toks, cfg)
    if registry:
        hits.append(("OP-030", registry, "Use an official HTTPS registry or an exact enterprise mirror approved by the owner."))

    # OP-031 .. OP-114 Hard Stop Evaluations
    r31 = check_cred_store_read(toks, ecwd)
    if r31:
        hits.append(("OP-031", r31, "The owner manages credential stores directly; do not read host credential files."))
    r32 = check_secret_cli_dump(toks)
    if r32:
        hits.append(("OP-032", r32, "Ask the owner before dumping credentials or access tokens from CLI managers."))
    r33 = check_secret_file_egress(toks, ecwd)
    if r33:
        hits.append(("OP-033", r33, "Uploading or transmitting secret credential files off-host is owner-only."))
    r34 = check_paste_tunnel_exfil(toks, text)
    if r34:
        hits.append(("OP-034", r34, "Sending data to pastebins, request bins, or OAST services is owner-only."))
    r35 = check_dns_exfil(toks, text)
    if r35:
        hits.append(("OP-035", r35, "Executing secret-extracting command substitutions in DNS queries is owner-only."))
    r37 = check_registry_config_write(toks, text, cfg)
    if r37:
        hits.append(("OP-037", r37, "Persisting untrusted package registries in config files is owner-only."))
    for w in e.get("writes", []):
        r44 = check_persistence_file(w, ecwd)
        if r44:
            hits.append(("OP-044", r44, "Installing persistent launchd daemons, scheduled cron jobs, or services is owner-only."))
    for t in toks:
        if "=" in t and not t.startswith("-"):
            vn, vv = t.split("=", 1)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", vn):
                r39 = check_loader_inject(vn, vv, ecwd)
                if r39:
                    hits.append(("OP-039", r39, "Setting dynamic loader injection variables is owner-only."))
                r40 = check_interpreter_hook(vn, vv, ecwd)
                if r40:
                    hits.append(("OP-040", r40, "Setting interpreter hooks to load external code is owner-only."))
                r41 = check_search_path_escape(vn, vv, ecwd)
                if r41:
                    hits.append(("OP-041", r41, "Escaping library or binary search paths to untrusted directories is owner-only."))
    r42 = check_foreign_debug(toks)
    if r42:
        hits.append(("OP-042", r42, "Attaching debuggers to external processes or reading process memory is owner-only."))
    r43 = check_persistence_control(toks)
    if r43:
        hits.append(("OP-043", r43, "Installing persistent launchd daemons, scheduled cron jobs, or services is owner-only."))
    r45 = check_os_trust_tamper(toks)
    if r45:
        hits.append(("OP-045", r45, "Disabling operating system security controls (SIP, Gatekeeper, firewall) is owner-only."))
    r46 = check_trust_store_inject(toks)
    if r46:
        hits.append(("OP-046", r46, "Installing root certificate authorities into system trust stores is owner-only."))
    r47 = check_quarantine_strip(toks, ecwd)
    if r47:
        hits.append(("OP-047", r47, "Stripping quarantine marks from downloaded applications is owner-only."))
    r48 = check_inbound_exec(toks, text)
    if r48:
        hits.append(("OP-048", r48, "Spawning shell-executing network listener sockets is owner-only."))
    r49 = check_remote_tunnel(toks)
    if r49:
        hits.append(("OP-049", r49, "Opening reverse network tunnels or funnel proxies is owner-only."))
    r50 = check_resolver_tamper(toks, path)
    if r50:
        hits.append(("OP-050", r50, "Tampering with system name resolution or network routing is owner-only."))
    r51 = check_private_key_material(toks, ecwd)
    if r51:
        hits.append(("OP-051", r51, "Accessing, reading, or copying private key files is owner-only."))
    r52 = check_secret_export(toks)
    if r52:
        hits.append(("OP-052", r52, "Exporting or unwrapping private keys or wallet seeds is owner-only."))
    r53 = check_ssh_exec_config(toks, text)
    if r53:
        hits.append(("OP-053", r53, "Configuring local command execution via SSH options is owner-only."))
    r54 = check_agent_bypass_spawn(toks)
    if r54:
        hits.append(("OP-054", r54, "Launching child coding agents with safety permissions stripped is owner-only."))
    r55 = check_local_model_mutate(toks)
    if r55:
        hits.append(("OP-055", r55, "Deleting or wiping local AI models is owner-only."))
    r63 = check_lockfile_destroy(toks, ecwd)
    if r63:
        hits.append(("OP-063", r63, "Deleting tracked package lockfiles risks supply-chain drift; owner review required."))
    r64 = check_history_rewrite(toks)
    if r64:
        hits.append(("OP-064", r64, "Rewriting existing git history destroys commit lineage; owner review required."))
    r65 = check_ref_clobber(toks)
    if r65:
        hits.append(("OP-065", r65, "Deleting git references or unmerged branches is owner-only."))
    r66 = check_cluster_drain(toks)
    if r66:
        hits.append(("OP-066", r66, "Draining Kubernetes nodes or force-deleting pods is owner-only."))
    r67 = check_auto_approve_apply(toks)
    if r67:
        hits.append(("OP-067", r67, "Infrastructure-as-code auto-approve bypasses owner confirmation."))
    r68 = check_platform_teardown(toks)
    if r68:
        hits.append(("OP-068", r68, "Tearing down cloud platform architecture is owner-only."))
    r69 = check_schema_clean(toks)
    if r69:
        hits.append(("OP-069", r69, "Wiping database schemas via migration clean commands is owner-only."))
    r70 = check_github_vandalism(toks)
    if r70:
        hits.append(("OP-070", r70, "Deleting GitHub issues or repository keys is owner-only."))
    r71 = check_outbound_blast(toks, text)
    if r71:
        hits.append(("OP-071", r71, "Sending outbound email/SMS broadcast communications is owner-only."))
    r72 = check_gpu_burn(toks, text)
    if r72:
        hits.append(("OP-072", r72, "Provisioning expensive GPU instances or fine-tuning runs is owner-only."))
    r76 = check_checker_delete(toks, ecwd)
    if r76:
        hits.append(("OP-076", r76, "Deleting tracked typechecker or linter configuration is owner-only."))

    # Extended rules OP-080 .. OP-114
    r80 = check_git_identity_spoof(toks, ecwd)
    if r80:
        hits.append(("OP-080", r80, "Spoofing git author identities or disabling commit signing is owner-only."))
    r81 = check_governance_weaken(toks, ecwd)
    if r81:
        hits.append(("OP-081", r81, "Deleting or weakening repository governance files is owner-only."))
    r82 = check_public_bucket_acl(toks)
    if r82:
        hits.append(("OP-082", r82, "Making cloud storage buckets or objects publicly accessible is owner-only."))
    r83 = check_cloud_iam_mutation(toks)
    if r83:
        hits.append(("OP-083", r83, "Mutating cloud IAM credentials, roles, or KMS keys is owner-only."))
    r84 = check_dns_tls_prod_change(toks)
    if r84:
        hits.append(("OP-084", r84, "Mutating or deleting production DNS zones or TLS certificates is owner-only."))
    r85 = check_cloud_secret_destroy(toks)
    if r85:
        hits.append(("OP-085", r85, "Deleting or force-overwriting cloud secrets or vault entries is owner-only."))
    r98 = check_tls_disable(toks, text)
    if r98:
        hits.append(("OP-098", r98, "Disabling TLS certificate verification is owner-only."))
    r99 = check_untrusted_package_source(toks, ecwd)
    if r99:
        hits.append(("OP-099", r99, "Configuring cleartext HTTP package dependencies is owner-only."))
    r101 = check_protect_local_state(toks, ecwd)
    if r101:
        hits.append(("OP-101", r101, "Deleting gitignored local developer state is owner-only."))
    r102 = check_archive_clobber(toks, ecwd)
    if r102:
        hits.append(("OP-102", r102, "Extracting archives over existing work without keep flags is owner-only."))
    r103 = check_stream_wipe(toks, ecwd)
    if r103:
        hits.append(("OP-103", r103, "Emptying existing non-empty files via truncate or /dev/null is owner-only."))
    r104 = check_sync_delete(toks)
    if r104:
        hits.append(("OP-104", r104, "Executing mirror sync deletions without dry-run is owner-only."))
    r105 = check_backup_destroy(toks)
    if r105:
        hits.append(("OP-105", r105, "Deleting OS backups, APFS snapshots, or Volume Shadow Copies is owner-only."))
    r107 = check_live_browser_debug(toks)
    if r107:
        hits.append(("OP-107", r107, "Attaching remote debugging port to live browser profile is owner-only."))
    r108 = check_gui_inject(toks, text)
    if r108:
        hits.append(("OP-108", r108, "Synthesizing GUI input or keystroke injection is owner-only."))
    r109 = check_sensor_capture(toks)
    if r109:
        hits.append(("OP-109", r109, "Recording screen, webcam, or microphone devices is owner-only."))
    r110 = check_clipboard_abuse(toks, fed_by)
    if r110:
        hits.append(("OP-110", r110, "Piping clipboard contents to network or copying secrets is owner-only."))
    r111 = check_extension_sideload(toks, path, ecwd)
    if r111:
        hits.append(("OP-111", r111, "Sideloading browser or editor extensions is owner-only."))
    return hits


# ---- constitution gaps: rules declared above whose detectors were not reached ------------------ #
# OP-056..OP-062, OP-073..OP-075, OP-077..OP-079 and OP-106 are in RULES. Their checks either
# existed and were never called, or the rule had no reader. Every one of them is deterministic:
# the command text, the write body, and git's read-only queries. Nothing here opens a network
# connection or calls a model.
HOOK_SETTINGS_RE = re.compile(
    r"(?:^|/)(?:\.claude/settings(?:\.local)?\.json|\.codex/(?:hooks\.json|config\.toml)|"
    r"\.cursor/hooks\.json|\.zcode/cli/config\.json)$")
PROTECTED_NAME_RE = re.compile(
    r"(?:^|/)(?:\.claude/settings(?:\.local)?\.json|\.codex/(?:hooks\.json|config\.toml)|"
    r"\.cursor/hooks\.json|\.operator/(?:gate/operator-gate\.py|commitments\.json)|commitments\.json)$")
CHECKER_FILE_RE = re.compile(
    r"(?:^|/)(?:tsconfig(?:\.[^/]+)?\.json|pyproject\.toml|ruff\.toml|\.ruff\.toml|mypy\.ini|"
    r"setup\.cfg|\.eslintrc(?:\.[^/]+)?|eslint\.config\.[cm]?js|biome\.jsonc?)$")
CHECKER_WEAKEN_RE = re.compile(
    r'"(?:strict|noImplicitAny|strictNullChecks|alwaysStrict)"\s*:\s*false'
    r"|ignore_errors\s*=\s*[Tt]rue|strict\s*=\s*false"
    r'''|"(?:no-explicit-any|ban-ts-comment)"\s*:\s*"off"''')
BLANKET_SUPPRESS_RE = re.compile(
    r"(?m)^\s*(?://\s*@ts-nocheck|#\s*noqa(?:\s|$)|#\s*type:\s*ignore(?:\s|$)|"
    r"/\*\s*eslint-disable\s*\*/|#\[allow\()")
SWALLOW_RE = re.compile(
    r"except\s+(?:Exception|BaseException)\s*:\s*(?:#.*\n)?\s*pass\b"
    r"|catch\s*\([^)]*\)\s*\{\s*\}")
CANNED_RE = re.compile(
    r"(?:PYTEST_CURRENT_TEST|sys\.modules\.get\(\s*['\"]pytest['\"]|"
    r"process\.env\.(?:NODE_ENV|VITEST|JEST_WORKER_ID))"
    r"[\s\S]{0,240}?return\s+(?:\{|\d|true|false|null|None|\"|')")
NOOP_TEST_RE = re.compile(r"^(?:true|false|:|echo(?:\s+\S+)?|exit\s+0)$")
SOURCE_EXT_RE = re.compile(r"\.(?:py|js|jsx|ts|tsx|mjs|cjs|go|java|rb|rs|kt)$", re.I)
ENV_PREFIX_RE = re.compile(r"(?:^|[\n;&|])\s*((?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+)")


def _env_prefixes(text):
    env = {}
    for m in ENV_PREFIX_RE.finditer(text or ""):
        for tok in m.group(1).split():
            if "=" in tok:
                key, val = tok.split("=", 1)
                env[key] = val.strip("'\"")
    return env


def check_hook_strip(text):
    """OP-056: a command that deletes the gate's hook or turns every hook off."""
    raw = text or ""
    if re.search(r"\b(?:sed|perl)\b", raw) and "operator-gate" in raw and re.search(r"/d\b|disableAllHooks", raw):
        return "strips the operator-gate hook from agent configuration"
    if re.search(r"\bjq\b[^\n]*del\(\s*\.hooks", raw):
        return "deletes hook registrations from agent configuration"
    if re.search(r'"disableAllHooks"\s*:\s*true', raw):
        return "disables every hook in agent settings"
    return None


def check_hook_strip_write(path, content):
    if not path or not HOOK_SETTINGS_RE.search(path):
        return None
    body = content or ""
    if re.search(r'"disableAllHooks"\s*:\s*true', body):
        return "sets disableAllHooks on %s" % os.path.basename(path)
    if "operator-gate" in body:
        return None
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                old = f.read()
            if "operator-gate" in old:
                return "replaces %s and drops operator-gate" % os.path.basename(path)
    except OSError:
        pass
    return None


def check_checker_weaken(path, content):
    """OP-057: a checker config flipped from strict to off."""
    if not path or not content or not CHECKER_FILE_RE.search(path):
        return None
    if CHECKER_WEAKEN_RE.search(content):
        return "weakens checker strictness in %s" % os.path.basename(path)
    return None


def check_suppression_spray(path, content):
    """OP-058: a file-level linter or type suppression, not one annotated line."""
    if not path or not content or not SOURCE_EXT_RE.search(path):
        return None
    if BLANKET_SUPPRESS_RE.search(content):
        return "adds a file-level suppression directive to %s" % os.path.basename(path)
    return None


def check_error_swallow(path, content):
    """OP-059: an empty handler in production code. Tests are allowed to stub."""
    if not path or not content or not SOURCE_EXT_RE.search(path):
        return None
    base = os.path.basename(path)
    if "/test" in path or base.startswith("test_") or base.endswith("_test.py"):
        return None
    if SWALLOW_RE.search(content):
        return "swallows errors with an empty handler in %s" % base
    return None


def check_canned_oracle(path, content):
    """OP-060: a branch that notices a test runner and returns a fixed value."""
    if not path or not content or not SOURCE_EXT_RE.search(path):
        return None
    if CANNED_RE.search(content):
        return "returns a canned value when it detects a test runner in %s" % os.path.basename(path)
    return None


def _pkg_test_script(text):
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    scripts = data.get("scripts") if isinstance(data, dict) else None
    if not isinstance(scripts, dict):
        return None
    val = scripts.get("test")
    return val if isinstance(val, str) else None


def check_fake_test_script(path, content):
    """OP-061: package.json's test script replaced with a command that cannot fail."""
    if not path or os.path.basename(path) != "package.json" or not content:
        return None
    new = _pkg_test_script(content)
    if new is None or not NOOP_TEST_RE.fullmatch(new.strip()):
        return None
    try:
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            old = _pkg_test_script(f.read())
    except OSError:
        return None
    if old and not NOOP_TEST_RE.fullmatch(old.strip()) and old.strip() != new.strip():
        return "replaces the package test script with a no-op (%s)" % new.strip()
    return None


def check_opaque_protected(effects, text, cfg, cwd):
    """OP-073: a redirect whose destination is not a literal, aimed at a protected file."""
    raw = text or ""
    if ">" not in raw or ("$" not in raw and "__SUBST__" not in raw and "$(" not in raw):
        return None
    names = [p for p in self_protected_files() if p and (p in raw or os.path.basename(p) in raw)]
    for c in (cfg or {}).get("file_contracts") or []:
        if isinstance(c, dict) and isinstance(c.get("path"), str) and c["path"] in raw:
            names.append(c["path"])
    if not names and not PROTECTED_NAME_RE.search(raw):
        return None
    opaque = False
    for e in effects or []:
        seg = str(e.get("seg") or "")
        if e.get("unresolved") or ("$" in seg and ">" in seg) or "__SUBST__" in seg:
            opaque = True
            break
    if opaque or "$" in raw:
        return "a redirect whose destination cannot be resolved names a protected file"
    return None


def check_atomic_replace(toks, cwd):
    """OP-074: mv, rename or install onto a protected file or a tracked migration."""
    if not toks:
        return None
    prog = os.path.basename(toks[0])
    if prog not in ("mv", "rename", "install"):
        return None
    args = [a for a in toks[1:] if not a.startswith("-")]
    if len(args) < 2:
        return None
    dest = resolve(args[-1], cwd)
    if touches_gate_home(dest) or any(same_path(dest, f) or is_inside(dest, f) for f in self_protected_files()):
        return "atomically replaces protected file %s" % os.path.basename(dest)
    why = check_migration_rewrite(dest, cwd)
    if why:
        return "atomically replaces %s" % why
    return None


def check_loop_redirect(text):
    """OP-075: for f in <protected files>; do ... > "$f"; done."""
    m = re.search(r"\bfor\s+\w+\s+in\s+(.+?)\s*;\s*do\b(.+?)\bdone\b", text or "", re.S)
    if not m or ">" not in m.group(2):
        return None
    header = m.group(1)
    if PROTECTED_NAME_RE.search(header) or any(
            name in header for name in (".claude/settings.json", ".codex/hooks.json", ".cursor/hooks.json",
                                         "operator-gate.py", "commitments.json")):
        return "a shell loop redirects output onto a protected file"
    return None


def check_git_object_destroy(toks, cwd):
    """OP-106: force-prune unreachable objects, or delete a branch that is not in HEAD."""
    if not toks or os.path.basename(toks[0]) != "git":
        return None
    args = toks[1:]
    if "gc" in args and any(a == "--prune=now" or a.startswith("--prune=now") for a in args):
        return "git gc --prune=now force-prunes unreachable objects"
    if args[:1] == ["prune"] or ("prune" in args and any(a in ("-f", "--expire=now") for a in args)):
        return "git prune deletes unreachable objects"
    if "branch" in args and "-D" in args:
        names = [a for a in args if not a.startswith("-") and a != "branch"]
        root = git_top(cwd)
        for name in names:
            if not root:
                break
            exists = git_out(root, ["rev-parse", "--verify", "--quiet", name])
            merged = git_out(root, ["merge-base", "--is-ancestor", name, "HEAD"])
            if exists is not None and merged is None:
                return "git branch -D deletes %s which is not merged into HEAD" % name
    return None


def _effect_toks(effects, text):
    seen = []
    for e in effects or []:
        toks = e.get("toks") or []
        if toks and toks not in seen:
            seen.append(toks)
    if seen or not text:
        return seen
    try:
        return [shlex.split(text)]
    except ValueError:
        return [text.split()]


def gap_rule_hits(payload, cwd, tool, ti, effects, cfg):
    """Hits for the constitution rules the earlier passes do not call."""
    hits = []
    ti = ti if isinstance(ti, dict) else {}
    text = str(ti.get("command") or ti.get("cmd") or "")
    body = str(ti.get("content") or ti.get("new_string") or "")
    if not body and isinstance(ti.get("edits"), list):
        body = "\n".join(str(e.get("new_string") or "") for e in ti["edits"] if isinstance(e, dict))
    paths = []
    if tool in WRITE_TOOLS:
        for p in write_paths(tool, ti, cwd):
            if p not in paths:
                paths.append(p)
    for e in effects or []:
        for w in e.get("writes") or []:
            if isinstance(w, str) and w not in paths:
                paths.append(w)
    blob = text + "\n" + body
    r56 = check_hook_strip(blob)
    if r56:
        hits.append(("OP-056", r56, "Leave the gate's hooks in place, or ask the owner."))
    r75 = check_loop_redirect(text)
    if r75:
        hits.append(("OP-075", r75, "Name the protected file in one command the owner can read."))
    r73 = check_opaque_protected(effects, text, cfg, cwd)
    if r73:
        hits.append(("OP-073", r73, "Name the destination as a plain path so the gate can judge it."))
    env = _env_prefixes(text)
    if env:
        r77 = check_unmonitored_agent_home([], env, cwd)
        if r77:
            hits.append(("OP-077", r77,
                         "Point the agent home at a directory that still runs operator-gate, or leave it unset."))
    for toks in _effect_toks(effects, text):
        r74 = check_atomic_replace(toks, cwd)
        if r74:
            hits.append(("OP-074", r74, "Ask the owner before replacing a protected file or a tracked migration."))
        r106 = check_git_object_destroy(toks, cwd)
        if r106:
            hits.append(("OP-106", r106, "Leave unreachable objects and unmerged branches until the owner prunes them."))
    for p in paths:
        for rid, why, hint, fn in (
            ("OP-056", None, "Leave the gate's hooks in place, or ask the owner.", lambda: check_hook_strip_write(p, body)),
            ("OP-057", None, "Keep the checker as strict as it is, or ask the owner.", lambda: check_checker_weaken(p, body)),
            ("OP-058", None, "Suppress one line with a reason, or ask the owner.", lambda: check_suppression_spray(p, body)),
            ("OP-059", None, "Handle the error or let it surface.", lambda: check_error_swallow(p, body)),
            ("OP-060", None, "Return the real result under test too.", lambda: check_canned_oracle(p, body)),
            ("OP-061", None, "Keep the test runner the package already uses.", lambda: check_fake_test_script(p, body)),
            ("OP-062", None, "Add a new migration instead of editing one that has shipped.",
             lambda: check_migration_rewrite(p, cwd) if (body or tool in WRITE_TOOLS) else None),
            ("OP-078", None, "Write instructions the owner can read; do not bury an override.",
             lambda: check_instruction_inject(p, body)),
            ("OP-079", None, "Change skills inside the repository, or ask the owner.",
             lambda: check_global_skill_write(p, cwd)),
        ):
            detail = fn()
            if detail:
                hits.append((rid, detail, hint))
    return hits



def load_config():
    cfg = {"protected_roots": [], "protected_files": [], "disabled_rules": [], "mode_overrides": {},
           "scratch_ok": [], "file_contracts": [],
           "trusted_python_registries": [], "trusted_node_registries": []}
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
# taste lock v0 — owner commitments as laws (TasteLock-0)
#
# Monotonic invariant (Progent, arXiv 2504.11703): a commitment can only ADD a
# constraint. There is no syntax for allowing anything -- the 27 built-in OP
# rules are the floor and the owner's law is an overlay that may only narrow.
# The hook never parses the live commitments file: `operator-gate commitments
# check` validates it and atomically writes a cache stamped with law_digest.
# A broken edit can neither brick the gate nor take effect.
# --------------------------------------------------------------------------- #
TASTE_MATCH_KEYS = {"prog_any_of", "sub_any_of", "flag_any_of", "shape_any_of",
                    "path_prefix_any_of", "path_glob_any_of", "effect_kind_any_of",
                    "write_only", "recursive_only",
                    # vetoes that narrow a law to what it is about (Sam, 2026-10-20: judge what the call
                    # is doing, not the word in it): a step inside a script the agent runs, a search of one
                    # named file, a build or a read under a deploy law
                    "own_command_only", "hunts_only", "sub_none_of", "kind_none_of"}
# A law may name tools after which ONE stopped call is lifted (recall-before-grep: recall first, and if
# the agent still needs the search, it runs once). The lift is earned by a recorded call, never granted
# by the agent, and it never reaches below the built-in rules.
LAW_LIFT_TTL_S = 30 * 60
TASTE_COMMITMENT_KEYS = {"commitment_id", "version", "supersedes_commitment_id", "attribution",
                         "authority", "state", "source_class", "effective_at", "valid_until",
                         "scope", "directive", "match", "on_violation"}
TASTE_SCOPE_KEYS = {"project_id", "environment", "action_class"}
TASTE_EFFECT_KINDS = {"delete", "move", "git", "cli", "db", "inline", "pipe-exec", "read", "filter", "list"}
TASTE_REWRITES = {"append_dry_run", "drop_prod_markers", "move_to_trash"}
LAW_CACHE = pj(OP_HOME, "state", "commitments-validated.json")


def validate_commitments(doc):
    """Parse + validate a commitments document. Returns (laws, errors). Monotonic: any
    syntax that would widen the gate is invalid by construction (none exists)."""
    errors, laws = [], []
    if not isinstance(doc, dict):
        return [], ["document must be a JSON object"]
    for key in doc:
        if key not in ("schema_version", "taste_mode", "projects", "commitments",
                       "protected_roots", "protected_files", "disabled_rules",
                       "mode_overrides", "scratch_ok", "file_contracts",
                       "trusted_python_registries", "trusted_node_registries"):
            errors.append("unknown top-level key: %s" % key)
    for key in ("trusted_python_registries", "trusted_node_registries"):
        hosts = doc.get(key, [])
        if not isinstance(hosts, list) or any(
                not isinstance(h, str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", h)
                for h in hosts):
            errors.append("%s must be a list of exact DNS hostnames (no URLs or wildcards)" % key)
    seen = {}
    for i, c in enumerate(doc.get("commitments") or []):
        label = "commitment[%d]" % i
        if not isinstance(c, dict):
            errors.append("%s: must be an object" % label)
            continue
        for k in c:
            if k not in TASTE_COMMITMENT_KEYS:
                errors.append("%s: unknown key '%s'" % (label, k))
        cid = c.get("commitment_id") or ""
        if not re.match(r"^[a-z0-9][a-z0-9-]{2,60}$", cid):
            errors.append("%s: commitment_id must be kebab-case (got %r)" % (label, cid))
            continue
        label = "commitment '%s'" % cid
        if not isinstance(c.get("directive"), str) or len(c["directive"]) < 8:
            errors.append("%s: directive must be a human sentence (>=8 chars)" % label)
        if c.get("attribution") != "owner" or c.get("authority") != "owner":
            errors.append("%s: only owner-authored, owner-authority commitments are valid "
                          "(memory is evidence, not instruction)" % label)
        if c.get("state") not in ("active", "superseded"):
            errors.append("%s: state must be active|superseded" % label)
        if not c.get("effective_at"):
            errors.append("%s: effective_at is required" % label)
        if c.get("source_class") not in (None, "owner_record"):
            errors.append("%s: source_class must be owner_record" % label)
        scope = c.get("scope") or {}
        if not isinstance(scope, dict) or set(scope) - TASTE_SCOPE_KEYS:
            errors.append("%s: scope keys limited to %s" % (label, sorted(TASTE_SCOPE_KEYS)))
        elif scope.get("environment") not in (None, "all"):
            errors.append("%s: environment scoping lands post-v0.3; use 'all' for now" % label)
        m = c.get("match") or {}
        if not isinstance(m, dict) or not m:
            errors.append("%s: match must be a non-empty object over %s" % (label, sorted(TASTE_MATCH_KEYS)))
            continue
        for k in m:
            if k not in TASTE_MATCH_KEYS:
                errors.append("%s: unknown match key '%s' (closed vocabulary)" % (label, k))
            elif k == "effect_kind_any_of" and isinstance(m.get(k), list) and \
                    any(x not in TASTE_EFFECT_KINDS for x in m[k]):
                errors.append("%s: effect kinds limited to %s" % (label, sorted(TASTE_EFFECT_KINDS)))
        for list_key in ("prog_any_of", "sub_any_of", "flag_any_of", "shape_any_of",
                         "path_prefix_any_of", "path_glob_any_of", "effect_kind_any_of",
                         "sub_none_of", "kind_none_of"):
            if list_key in m and not (isinstance(m[list_key], list) and m[list_key]):
                errors.append("%s: match.%s must be a non-empty list" % (label, list_key))
        if isinstance(m.get("kind_none_of"), list) and any(x not in TASTE_EFFECT_KINDS for x in m["kind_none_of"]):
            errors.append("%s: effect kinds limited to %s" % (label, sorted(TASTE_EFFECT_KINDS)))
        for bool_key in ("write_only", "recursive_only", "own_command_only", "hunts_only"):
            if bool_key in m and not isinstance(m[bool_key], bool):
                errors.append("%s: match.%s must be true or false" % (label, bool_key))
        on_v = c.get("on_violation") or {"decision": "stop"}
        if not isinstance(on_v, dict) or on_v.get("decision") not in ("stop", "rewrite"):
            errors.append("%s: on_violation.decision must be stop|rewrite" % label)
        elif on_v["decision"] == "rewrite" and on_v.get("rewrite") not in TASTE_REWRITES:
            errors.append("%s: rewrite must be one of %s" % (label, sorted(TASTE_REWRITES)))
        elif "lift_once_after_tool" in on_v and (
                on_v["decision"] != "stop" or not isinstance(on_v["lift_once_after_tool"], list) or
                not on_v["lift_once_after_tool"] or
                not all(isinstance(t, str) and t for t in on_v["lift_once_after_tool"])):
            errors.append("%s: on_violation.lift_once_after_tool must be a non-empty list of tool names "
                          "on a stop law" % label)
        # supersession chain: version must increase over the named predecessor
        prev = c.get("supersedes_commitment_id")
        if prev:
            if prev not in seen:
                errors.append("%s: supersedes unknown commitment '%s' (order matters: define "
                              "predecessors first)" % (label, prev))
            elif c.get("version", 0) <= seen[prev]:
                errors.append("%s: version must exceed superseded commitment's version" % label)
            else:
                errors.append("__mark_superseded__%s" % prev)
        seen[cid] = c.get("version", 0)
        for other in laws:
            if other.get("state") == "active" and c.get("state") == "active" and \
                    other.get("match") == c.get("match") and \
                    other.get("scope") == c.get("scope") and \
                    c.get("supersedes_commitment_id") != other.get("commitment_id"):
                errors.append("%s: duplicate match+scope vs commitment '%s' -- ambiguous law"
                              % (label, other.get("commitment_id")))
        laws.append(c)
    # second pass: mark superseded predecessors
    final = []
    for c in laws:
        cc = dict(c)
        for e in errors[:]:
            if e.startswith("__mark_superseded__") and e[len("__mark_superseded__"):] == cc.get("commitment_id"):
                cc["state"] = "superseded"
                errors.remove(e)
        final.append(cc)
    return final, errors


def write_law_cache(doc):
    """Validate + atomically write the law cache. Returns (law_digest, error or None)."""
    laws, errors = validate_commitments(doc)
    if errors:
        return None, "; ".join(errors[:5])
    digest = digest_of({"laws": laws, "schema": doc.get("schema_version", "1.0.0"),
                        "taste_mode": doc.get("taste_mode", "shadow")})
    cache = {"law_digest": digest, "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "taste_mode": doc.get("taste_mode", "shadow"), "laws": laws}
    os.makedirs(os.path.dirname(LAW_CACHE), exist_ok=True)
    tmp = LAW_CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, LAW_CACHE)
    return digest, None


def load_law():
    """Read the validated law cache. Returns None when no valid law has ever landed."""
    try:
        with open(LAW_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cmd_commitments(argv):
    """`operator-gate commitments check` — validate the owner's law file and land the cache."""
    if argv[:1] != ["check"]:
        print("usage: operator-gate commitments check")
        return 2
    path = pj(OP_HOME, "commitments.json")
    if not os.path.exists(path):
        print("no law file at %s -- copy a template there and edit it by hand." % path)
        return 1
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as exc:
        print("INVALID law file (not JSON): %s" % exc)
        return 1
    digest, err = write_law_cache(doc)
    if err:
        print("INVALID law file: %s" % err)
        return 1
    active = [l for l in doc.get("commitments", []) if l.get("state") == "active"]
    print("law ok: %d active, %d superseded, law_digest %s"
          % (len(active), len(doc.get("commitments", [])) - len(active), digest[:12]))
    for c in active:
        print("  %s v%s [%s] %s" % (c["commitment_id"], c.get("version", 1),
                                    c.get("scope", {}).get("action_class", "all"), c.get("directive", "")))
    print("taste_mode: %s (shadow logs TASTE_SHADOW receipts and allows; set 'enforce' and re-run check)"
          % doc.get("taste_mode", "shadow"))
    return 0


# ---- taste evaluation ------------------------------------------------------ #
ACTION_CLASS_BY_PROG = {
    "stripe": "billing", "fly": "deployment", "flyctl": "deployment", "vercel": "deployment",
    "wrangler": "deployment", "supabase": "deployment",
    "npm": "publication", "pnpm": "publication", "yarn": "publication",
    "cargo": "publication", "twine": "publication",
}
PROD_MARKERS = ("--prod", "--live", "sk_live", "rk_live", "RAILS_ENV=production", "-a .*-prod")


def project_candidate(effects):
    """Deterministic candidate from analyzer effects: what the gate OBSERVED. Closed shape."""
    classes, prod_shaped, write, recursive, resolved = set(), False, False, False, True
    for e in effects:
        prog = e.get("prog", "")
        if e.get("unresolved") or (e.get("cwd") or "").startswith(UNKNOWN_CWD):
            resolved = False
        cl = ACTION_CLASS_BY_PROG.get(prog)
        if cl:
            classes.add(cl)
        if e.get("kind") == "git" and e.get("sub") == "push":
            classes.add("source_provenance")
        # the text of inline code is data until a command is lifted out of it (that becomes a cli effect of
        # its own); a marker in a string literal is not a deploy (receipts f675af5f5dba27cf, b40dac003b2f57e2)
        text = "" if e.get("kind") == "inline" else (e.get("text") or "")
        if any(re.search(p, text) for p in PROD_MARKERS):
            prod_shaped = True
            classes.add("deployment")
        if e.get("kind") in ("delete", "move") or e.get("writes"):
            write = True
            classes.add("launch_scope" if any("/vestige-launch" in t for t in e.get("targets", []) + e.get("writes", []))
                        else "source_provenance")
        if prog in ("npm", "pnpm", "yarn", "cargo") and e.get("kind") == "cli" and \
                ("--dry-run" in e.get("flags", []) or "-n" in e.get("flags", [])):
            classes.discard("publication")          # a dry run performs no external mutation
        if e.get("recursive"):
            recursive = True
        if not e.get("targets") and e.get("kind") in ("delete", "move"):
            resolved = False
    return {"classes": sorted(classes), "prod_shaped": prod_shaped, "write": write,
            "recursive": recursive, "resolved": resolved,
            "effects": effects}


def commitment_eligible(c, project_id, now_iso):
    """Port of the evaluator's _commitment_resolution: the part that makes a commitment a LAW."""
    if c.get("state") != "active" or c.get("attribution") != "owner" or c.get("authority") != "owner":
        return False
    scope = c.get("scope") or {}
    pid = scope.get("project_id", "all")
    if pid not in ("all", project_id):
        return False
    eff = c.get("effective_at") or ""
    if eff and eff > now_iso:
        return False
    until = c.get("valid_until")
    if until and now_iso > until:
        return False
    return True


def match_commitment(c, cand):
    """Closed-vocabulary predicates over observed effects. Returns matched evidence or None."""
    m = c.get("match") or {}
    scope = c.get("scope") or {}
    ac = scope.get("action_class")
    evidence = []
    if ac and ac != "all":
        # action-class scoping needs the class to be observed, or a prod-shape claim on deployment
        if ac in cand["classes"]:
            evidence.append("action_class=%s" % ac)
        elif ac == "deployment" and cand["prod_shaped"]:
            evidence.append("prod-shaped command")
        else:
            return None
    # Every predicate the owner wrote must hold on the same effect (AND across keys);
    # the alternatives inside one key stay OR. "stripe" + "--live" means a live stripe
    # action, not every stripe command. write_only / recursive_only remain vetoes.
    required = [k for k in PREDICATE_KEYS if k in m]
    hit = False
    for e in cand["effects"]:
        prog = e.get("prog", "")
        text = "" if e.get("kind") == "inline" else (e.get("text") or "")
        matched = {}
        if "prog_any_of" in m and prog in m["prog_any_of"]:
            matched["prog_any_of"] = "prog=%s" % prog
        if "sub_any_of" in m and any(s in (e.get("args") or []) or s == e.get("sub") for s in m["sub_any_of"]):
            matched["sub_any_of"] = "sub matched"
        if "flag_any_of" in m and any(f in e.get("flags", []) for f in m["flag_any_of"]):
            matched["flag_any_of"] = "flag matched"
        if "shape_any_of" in m and any(re.search(p, text) for p in m["shape_any_of"]):
            matched["shape_any_of"] = "shape matched"
        if "effect_kind_any_of" in m and e.get("kind") in m["effect_kind_any_of"]:
            matched["effect_kind_any_of"] = "effect=%s" % e.get("kind")
        paths = e.get("targets", []) + e.get("writes", [])
        if "path_prefix_any_of" in m and any(
                any(t.startswith(norm(p, e.get("cwd") if e.get("cwd") and not str(e.get("cwd")).startswith(UNKNOWN_CWD) else None)) or
                    is_inside(t, norm(p, None)) for t in paths)
                for p in m["path_prefix_any_of"]):
            matched["path_prefix_any_of"] = "path under protected prefix"
        if "path_glob_any_of" in m and any(
                any(fnmatch.fnmatch(t, norm(p, None)) for t in paths) for p in m["path_glob_any_of"]):
            matched["path_glob_any_of"] = "path matched glob"
        local = [matched[k] for k in required] if required and all(k in matched for k in required) else []
        if prog in ("npm", "pnpm", "yarn", "cargo") and \
                ("--dry-run" in e.get("flags", []) or "-n" in e.get("flags", [])):
            local = []          # a dry run provably performs no external mutation
        # vetoes are terminal for this effect -- a later path match cannot undo them
        if "write_only" in m and m["write_only"] and not (e.get("writes") or e.get("kind") in ("delete", "move")):
            local = []
        if "recursive_only" in m and m["recursive_only"] and not e.get("recursive"):
            local = []
        if m.get("own_command_only") and e.get("script_body"):
            local = []          # a step inside a script the agent runs belongs to the script, not to this call
        if m.get("hunts_only") and not e.get("hunt"):
            local = []          # one named file is raw content to read, not a hunt for knowledge
        if "kind_none_of" in m and e.get("kind") in m["kind_none_of"]:
            local = []          # text that only mentions a marker; a real command inside it is its own effect
        if "sub_none_of" in m and set(lead_words(e)) & set(m["sub_none_of"]):
            local = []          # `ng build --prod`, `heroku logs -a shop-prod`: not the act the law is about
        if local and m.get("hunts_only"):
            local.append("searches a folder tree or a memory store Vestige holds")
        if local:
            hit = True
            evidence.append("%s: %s" % (prog, ", ".join(local)))
    if required and not hit:
        return None             # the owner wrote predicates; the class alone is not the law
    if not hit and not (ac and evidence):
        return None
    return evidence or ["action_class=%s" % ac]


PREDICATE_KEYS = ("prog_any_of", "sub_any_of", "flag_any_of", "shape_any_of",
                  "effect_kind_any_of", "path_prefix_any_of", "path_glob_any_of")


def lead_words(e):
    """The subcommand and the first two plain words after the program: what the command DOES
    (`stripe customers list`, `heroku logs`, `ng build`), never a flag's value further along."""
    words = [e.get("sub")] if e.get("sub") else []
    words += [a for a in (e.get("args") or []) if isinstance(a, str) and not a.startswith("-")][:2]
    return words


# ---- law lifts: one stopped call runs after the tool the law names (recall, then search once) ---- #
def law_lift_tools(c):
    return [t.lower() for t in ((c.get("on_violation") or {}).get("lift_once_after_tool") or [])]


def _law_lift_update(session, change):
    """Read-modify-write this session's lift record under a lock, so two parallel calls after one recall
    cannot both spend the one search. Returns what `change` returns."""
    d = pj(OP_HOME, "state", "law-lifts")
    os.makedirs(d, exist_ok=True)
    path = pj(d, hashlib.sha256(session.encode("utf-8")).hexdigest()[:24] + ".json")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        raw = b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            raw += chunk
        try:
            rec = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except ValueError:
            rec = {}
        now = time.time()
        rec = {k: v for k, v in rec.items() if isinstance(v, dict) and now - v.get("ts", 0) < LAW_LIFT_TTL_S}
        out = change(rec, now)
        data = json.dumps(rec).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, data)
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
    return out


def law_lift_seen(law, tool, session):
    """The agent called a tool a law names (Vestige recall): its next call stopped by that law runs once."""
    if not session or not law or not tool:
        return
    ids = [c.get("commitment_id") for c in law.get("laws", [])
           if c.get("state") == "active" and tool.lower() in law_lift_tools(c)]
    if not ids:
        return

    def change(rec, now):
        for cid in ids:
            st = rec.get(cid) or {}
            rec[cid] = {"stopped": st.get("stopped", 0), "lifted": now, "used": False, "ts": now}
    _law_lift_update(session, change)


def law_lift_take(c, session):
    """True exactly once after a lifting tool ran in this session; otherwise records this stop, so the
    agent's next recall opens one call. Fails closed: any error means the law stops."""
    if not session or not law_lift_tools(c):
        return False
    cid = c.get("commitment_id")

    def change(rec, now):
        st = rec.get(cid) or {}
        if st.get("lifted", 0) > st.get("stopped", 0) and not st.get("used"):
            rec[cid] = dict(st, used=True, ts=now)
            return True
        rec[cid] = {"stopped": now, "lifted": 0, "used": False, "ts": now}
        return False
    return bool(_law_lift_update(session, change))


def evaluate_taste(law, effects, project_id, now_iso):
    """Returns (hits, eligibles, candidate). Ambiguity is rejected at check time."""
    cand = project_candidate(effects)
    eligible = [c for c in law.get("laws", []) if commitment_eligible(c, project_id, now_iso)]
    hits = []
    for c in eligible:
        ev = match_commitment(c, cand)
        if ev:
            hits.append((c, ev))
    hits.sort(key=lambda h: (-(len(h[1])), h[0]["commitment_id"]))
    return hits, eligible, cand


def global_mode():
    m = os.environ.get("OPERATOR_GATE_MODE")
    if not m:
        try:
            with open(pj(OP_HOME, "mode"), encoding="utf-8") as f:
                m = f.read().strip()
        except Exception:
            m = "enforce"
    return m if m in ("enforce", "shadow", "off") else "enforce"


DEFECTS = []                                         # readers that failed during this judgement (see guarded)


def strict_on():
    """Strict is the owner's fail-closed switch: a call the gate cannot read is stopped instead of
    passed. On when the file STRICT sits in the gate home or OPERATOR_STRICT=1 is in the hook's
    environment. Off by default: a reader that stops what it cannot read also stops unusual lines
    that would have run fine."""
    if os.environ.get("OPERATOR_STRICT", "").strip().lower() in ("1", "on", "true", "yes"):
        return True
    return os.path.exists(pj(OP_HOME, "STRICT"))


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
        elif c == "#" and (i == 0 or cmd[i - 1].isspace() or cmd[i - 1] in ";|&()"):
            while i < n and cmd[i] != "\n":
                i += 1
            segs.append("".join(cur)); cur = []
            continue
        elif c in ";\n":
            segs.append("".join(cur)); cur = []
        elif c in "&|" and ((i > 0 and cmd[i - 1] == ">") or
                             (c == "&" and i + 1 < n and cmd[i + 1] == ">")):
            cur.append(c)                              # >|, >& and &> are redirections, not pipelines
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
    via_xargs, words = False, list(toks)
    values = {
        "sudo": ({"--user", "--group", "--host", "--prompt", "--close-from", "--chdir", "--chroot",
                  "--command-timeout", "--role", "--type", "--other-user"}, "ughpCDRTUrt"),
        "doas": (set(), "uC"),
        "env": ({"--unset", "--chdir", "--split-string", "--argv0"}, "uCSa"),
        "timeout": ({"--signal", "--kill-after"}, "sk"),
        "nice": ({"--adjustment"}, "n"),
        "stdbuf": ({"--input", "--output", "--error"}, "ioe"),
        "caffeinate": (set(), "tw"),
        "xargs": ({"--arg-file", "--delimiter", "--eof", "--replace", "--max-lines",
                   "--max-args", "--max-procs"}, "adEILnP"),
    }
    while words:
        t = words[0]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\+?=", t):
            words = words[1:]
            continue
        base = os.path.basename(t)
        if base not in WRAPPERS:
            break
        via_xargs = via_xargs or base == "xargs"
        wrapped = list(words)
        long_values, short_values = values.get(base, ((), ""))
        options, words = leading_options(words[1:], long_values, short_values)
        if any(name == "--help" for name, _ in options) or \
                (base == "command" and any(name in ("-v", "-V") for name, _ in options)) or \
                (base == "sudo" and any(name in ("-l", "--list") for name, _ in options)):
            return wrapped, via_xargs                 # inspect the command, never execute its child
        if base == "env":
            split = [value for name, value in options if name in ("-S", "--split-string")]
            if split:
                words = tokenize(split[-1] or "") + words
        if base == "timeout" and words and re.fullmatch(r"[0-9.]+[smhd]?", words[0]):
            words = words[1:]
    return words, via_xargs


REDIRECT_TOKEN_RE = re.compile(r"^(?:[0-9]*|&)(?:>>[|!]?|>&|>[|!]|<<<|<>|>|<)(.*)$")


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


OWNER_READ_RE = re.compile(r"(?:^|[;&|({\n]\s*)(?:sudo\s+)?(?:\S*python3?\s+)?\S*operator-gate(?:\.py)?\s+"
                           r"(?:(?:approve|mode)(?:\s+(?:--help|-h))?|(?:strict|install|uninstall|onboard|protect)"
                           r"\s+(?:--help|-h))\s*(?:$|[;&|)\n]|\d*>)", re.M)
# `operator-gate mode` or `approve` with nothing after it only prints, and `--help` prints usage: reads
# (receipts 909c12c05d44618d, 29ca345d61d9b7e7, 41a76af4a714b152). Every other form still stops.
OWNER_CMD_RE = re.compile(r"(?:^|[;&|({\n]\s*)(?:sudo\s+)?(?:\S*python3?\s+)?\S*operator-gate(?:\.py)?\s+"
                          r"(?:(?:approve|mode)\b(?!\s*(?:--help\b|-h\b|$|[;&|)\n]|\d*>))"
                          r"|(?:strict|install|uninstall|onboard|protect|laws\s+adopt|brake\s[^;&|\n]*--release"
                          r"|rewind\s[^;&|\n]*--restore)\b(?!\s+(?:--help|-h)\b))", re.M)


def command_text(cmd):
    """The command with heredoc bodies and quoted strings blanked, followed by the strings a shell is
    handed with -c. Text that is only written or printed is not a command; `sh -c '...'` is."""
    body = extract_heredocs(cmd)[0]
    inner = [m.group(2) for m in re.finditer(r"\b(?:sh|bash|zsh|dash|ksh)\s+(?:-[a-zA-Z]*c)\s+(['\"])(.*?)\1", cmd, re.S)]
    return mask_quotes(body) + "\n" + "\n".join(inner)


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


def short_home(p):
    """A path under the home folder written from ~, the way the owner reads it."""
    p = str(p or "")
    return "~" + p[len(HOME):] if p == HOME or p.startswith(HOME + "/") else p


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
    """Unquoted output targets, including >| and concatenated shell quotes.

    Scan original syntax so a quoted literal '>' remains data, then tokenize
    just the destination word to resolve quote fragments and backslash escapes.
    """
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
            if j < n and seg[j] in "|!":
                j += 1
            if j < n and seg[j] == "(":
                i = j
                continue
            if j < n and seg[j] == "&":
                j += 1
                if j < n and (seg[j].isdigit() or seg[j] == "-"):
                    i = j
                    continue
            while j < n and seg[j].isspace():
                j += 1
            start, quote = j, None
            while j < n:
                char = seg[j]
                if quote:
                    if char == "\\" and quote == '"' and j + 1 < n:
                        j += 2
                        continue
                    if char == quote:
                        quote = None
                elif char in "'\"":
                    quote = char
                elif char == "\\" and j + 1 < n:
                    j += 2
                    continue
                elif char.isspace() or char in ";&|<>":
                    break
                j += 1
            words = tokenize(seg[start:j])
            if words:
                out.append(words[0])
            i = max(j, i + 1)
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
            option, equal, attached = t.partition("=")
            target_option = option.startswith("--") and len(option) > 2 and "--target-directory".startswith(option)
            if (t == "-t" or target_option and not equal) and k + 1 < len(rest):
                tv = rest[k + 1]
            elif target_option and equal:
                tv = attached
            elif t.startswith("-t") and not t.startswith("--") and len(t) > 2:
                tv = t[2:]
        dest = tv or (args[-1] if len(args) > 1 or (prog in ("ln",) and args) else None)
        if dest:
            out.append(dest)
            # A copy/install to a directory writes source basenames inside it.
            # Include .ssh even when it has not been created on this host yet.
            target = resolve(dest, cwd)
            if tv or os.path.basename(target.rstrip("/")) == ".ssh" or os.path.isdir(target):
                out += [os.path.join(target, os.path.basename(a.rstrip("/")))
                        for a in args if a != dest and not a.endswith("/")]
    elif prog == "dd":
        out += [t[3:] for t in rest if t.startswith("of=")]
    elif prog in ("curl", "wget"):
        for k, t in enumerate(rest):
            if t in ("-o", "-O", "--output", "--output-document") and k + 1 < len(rest):
                out.append(rest[k + 1])
    return [resolve(w, cwd) for w in out if w]


# A fork bomb is a function that pipes itself into itself in the background and is then called:
# `:(){ :|:& };:`, `bomb(){ bomb|bomb& };bomb`. A function that merely contains a pipe and is followed by
# `&&` or `2>&1` is ordinary shell (receipts f79bc9c48327ade3, ab087dad929b8689, 2b0207774bd2cbef).
FORK_BOMB_RE = re.compile(r"(?<![\w-])([A-Za-z_:][\w:-]*)\s*\(\s*\)\s*\{[^}]*?(?<![\w-])\1\s*\|\s*&?\s*\1\s*&"
                          r"[^}]*\}\s*;?\s*\1(?![\w-])")
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

# Every shell an agent can hand a command string or a script to. One list, used wherever the gate
# asks "is this a shell": POSIX shells, their restricted and static builds, and the newer ones.
ALL_SHELLS = ("sh", "bash", "zsh", "dash", "ksh", "ksh93", "mksh", "pdksh", "oksh", "loksh", "ash", "hush", "fish",
              "csh", "tcsh", "nu", "nushell", "xonsh", "elvish", "osh", "ysh", "oil", "yash", "rc", "es", "ion",
              "murex", "rbash", "rksh", "rzsh", "bash-static", "zsh-static", "posh", "sash", "bosh")
SCRIPT_ARG_INTERPRETERS = ALL_SHELLS + (
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
        DEFECTS.append(getattr(fn, "__name__", "reader"))    # strict mode stops a call a reader failed on
        return default


DATA_EXT = {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".csv", ".tsv", ".log", ".html", ".htm", ".xml",
            ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
            ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".dmg", ".pkg"}


# Stores whose knowledge Vestige answers by exact handle (recall, precedent): searching them is a lookup to
# make there first (receipts ac969c75d76a0ffb, 736b581c7596c7f5).
KNOWLEDGE_ROOTS = (pj(HOME, ".operator", "receipts"), pj(HOME, ".operator", "permits"), pj(HOME, ".codex", "memories"),
                   pj(HOME, "Library", "Application Support", "com.vestige.core"))


def is_knowledge_file(path):
    p = str(path or "")
    return posixpath.basename(p) == "MEMORY.md" or any(p == r or p.startswith(r + "/") for r in KNOWLEDGE_ROOTS) or \
        re.search(r"/\.claude/projects/[^/]+/memory/", p) is not None


GREP_VALUE_FLAGS = {"-A", "-B", "-C", "-m", "-e", "-f", "-d", "-D", "-g", "-t", "-T", "-M", "-j",
                    "--context", "--after-context", "--before-context", "--max-count", "--regexp", "--file",
                    "--glob", "--type", "--type-not", "--max-columns", "--threads"}


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
    out = {"commands": [], "deletes": [], "writes": [], "findings": [], "payloads": [], "files": []}

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
        elif attr in ("write_text", "write_bytes", "touch"):
            target = val(n.func.value)
            if target:
                out["writes"].append((target, attr))
        elif name in ("open", "io.open", "builtins.open") or attr == "open":
            path_call = attr == "open" and name not in ("io.open", "builtins.open")
            target = val(n.func.value) if path_call else (val(args[0]) if args else None)
            if not path_call:
                target = next((val(k.value) for k in n.keywords if k.arg == "file"), target)
            mode = val(args[0]) if path_call and args else (val(args[1]) if len(args) > 1 else "r")
            mode = next((val(k.value) for k in n.keywords if k.arg == "mode"), mode)
            if target and mode and any(c in mode for c in "wax+"):
                out["writes"].append((target, "open"))
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
    for target, name in found.get("writes", []):
        t = resolve(target, ecwd)
        if not t.startswith(UNKNOWN_CWD) and UNKNOWN_WORD not in t:
            what = "%s(%s)" % (name, target)
            out += tag([dict(base, prog=name, seg=what, kind="write", writes=[t], text=what)])
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
    if mode != "shell" and ext in DATA_EXT and not first.startswith("#!"):
        return []           # a data file handed to a program is read by it, not run (2026-10-04 receipt 20346397274df626)
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
    words = list(toks)
    values = {
        "sudo": ({"--user", "--group", "--host", "--prompt", "--close-from", "--chdir", "--chroot",
                  "--command-timeout", "--role", "--type", "--other-user"}, "ughpCDRTUrt"),
        "doas": (set(), "uC"),
        "env": ({"--unset", "--chdir", "--split-string", "--argv0"}, "uCSa"),
        "timeout": ({"--signal", "--kill-after"}, "sk"),
        "nice": ({"--adjustment"}, "n"),
        "stdbuf": ({"--input", "--output", "--error"}, "ioe"),
        "caffeinate": (set(), "tw"),
        "xargs": ({"--arg-file", "--delimiter", "--eof", "--replace", "--max-lines",
                   "--max-args", "--max-procs"}, "adEILnP"),
    }
    assignments = []
    while words:
        t = words[0]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\+?=(.*)$", t)
        if m:
            assignments.append((m.group(1), m.group(2).strip().strip("'\"")))
            words = words[1:]
            continue
        base = os.path.basename(t)
        if base in ("export", "local", "declare", "readonly", "typeset"):
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
            continue
        if base == "setenv" and len(words) >= 3:
            assignments.append((words[1], words[2].strip().strip("'\"")))
            words = words[3:]
            continue
        if base == "launchctl" and len(words) >= 3 and words[1] == "setenv":
            assignments.append((words[2], words[3].strip().strip("'\"") if len(words) > 3 else ""))
            words = words[4:]
            continue
        if base == "systemctl" and any(w.startswith("set-environment") for w in words):
            for w in words[1:]:
                if "=" in w:
                    sn, sv = w.split("=", 1)
                    assignments.append((sn, sv.strip().strip("'\"")))
            break
        if base not in WRAPPERS:
            break
        long_values, short_values = values.get(base, ((), ""))
        options, words = leading_options(words[1:], long_values, short_values)
        if base == "env":
            split = [val for opt_name, val in options if opt_name in ("-S", "--split-string")]
            if split:
                words = tokenize(split[-1] or "") + words
        if base == "timeout" and words and re.fullmatch(r"[0-9.]+[smhd]?", words[0]):
            words = words[1:]

    for name, value in assignments:
        r39 = check_loader_inject(name, value, ecwd)
        if r39:
            out.append({"prog": "env", "seg": "%s=%s" % (name, value), "targets": [], "recursive": False,
                        "kind": "finding", "filtered": False, "unresolved": False, "flags": [], "args": [],
                        "cwd": ecwd or UNKNOWN_CWD, "writes": [], "text": r39,
                        "finding": ("OP-039", r39, "Setting dynamic loader injection variables is owner-only.")})
        r40 = check_interpreter_hook(name, value, ecwd)
        if r40:
            out.append({"prog": "env", "seg": "%s=%s" % (name, value), "targets": [], "recursive": False,
                        "kind": "finding", "filtered": False, "unresolved": False, "flags": [], "args": [],
                        "cwd": ecwd or UNKNOWN_CWD, "writes": [], "text": r40,
                        "finding": ("OP-040", r40, "Setting interpreter hooks to load external code is owner-only.")})
        r41 = check_search_path_escape(name, value, ecwd)
        if r41:
            out.append({"prog": "env", "seg": "%s=%s" % (name, value), "targets": [], "recursive": False,
                        "kind": "finding", "filtered": False, "unresolved": False, "flags": [], "args": [],
                        "cwd": ecwd or UNKNOWN_CWD, "writes": [], "text": r41,
                        "finding": ("OP-041", r41, "Escaping library or binary search paths to untrusted directories is owner-only.")})
        if name in EXEC_VARS and value:
            out += analyze(value, ecwd, depth + 1, vars_)
        elif name in EXEC_FILE_VARS and value and ecwd:
            out += script_body_effects(resolve(value, ecwd), ecwd, depth, vars_, mode="shell")
        elif name == "ZDOTDIR" and value and ecwd:
            for rc in (".zshenv", ".zprofile", ".zshrc", ".zlogin"):
                out += script_body_effects(pj(resolve(value, ecwd), rc), ecwd, depth, vars_, mode="shell")
    return out


# ---- the owner's aliases and functions -------------------------------------------------------- #
FUNC_OPEN_RE = re.compile(r"^(?:function\s+)?([^\s()=;|&<>'\"]+)\s*\(\)\s*\{?\s*$|^function\s+([^\s()=;|&<>'\"]+)\s*\{?\s*$")
PROFILE_DEFS = None                      # {name: [(kind, text)]}, read once per process
DEFS_SEEN = set()                        # definitions already judged for the command being judged


def profile_defs():
    """The aliases and functions of the owner's shell, as Claude Code's shell snapshots record them:
    {name: [(kind, text)]} with kind "alias", "global" (zsh `alias -g`, expanded anywhere in a line),
    "suffix" (zsh `alias -s`, not judged) or "function". Every snapshot is read: the gate is not told which
    one the calling session sources, and they come from the same profile."""
    global PROFILE_DEFS
    if PROFILE_DEFS is not None:
        return PROFILE_DEFS
    defs = {}
    for d in SHELL_SNAPSHOT_DIRS:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            p = pj(d, fn)
            try:
                if not fn.endswith(".sh") or not os.path.isfile(p) or os.path.getsize(p) > SNAPSHOT_MAX_BYTES:
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().split("\n")
            except OSError:
                continue
            i = 0
            while i < len(lines):
                line = lines[i]
                m = FUNC_OPEN_RE.match(line)
                if m:                                        # `name () {` ... `}` in column 0 (zsh and bash print it so)
                    j = i + 1
                    while j < len(lines) and lines[j] != "}":
                        j += 1
                    fname, body = m.group(1) or m.group(2), ("function", "\n".join(lines[i + 1:j]))
                    if body not in defs.setdefault(fname, []):
                        defs[fname].append(body)
                    i = j + 1
                    continue
                if line.startswith("alias "):                # `alias -- name=value`, `alias -g -- name=value`
                    toks = tokenize(line)
                    kind = "global" if "-g" in toks[1:] else ("suffix" if "-s" in toks[1:] else "alias")
                    for t in toks[1:]:
                        if "=" in t and not t.startswith("-"):
                            n, v = t.split("=", 1)
                            if (kind, v) not in defs.setdefault(n, []):
                                defs[n].append((kind, v))
                i += 1
    PROFILE_DEFS = defs
    return defs


def profile_def_effects(toks, ecwd, depth, vars_):
    """What a command runs when its name is one of the owner's aliases or functions, or a word in it is a
    zsh global alias: the definition is judged as if it had been typed. An alias keeps the words after it;
    a function gets them as "$1"... "$@". Each definition is read once per command. Never executes anything."""
    defs = profile_defs()
    if not defs or not toks or depth + 1 > MAX_DEPTH:
        return []
    name, rest = toks[0], toks[1:]
    uses = [(name, d) for d in defs.get(name, []) if d[0] in ("alias", "function")]
    uses += [(t, d) for t in toks for d in defs.get(t, []) if d[0] == "global"]
    out = []
    for word, (kind, text) in uses:
        key = (word, kind, text, tuple(rest) if kind != "function" else ())
        if key in DEFS_SEEN:
            continue
        DEFS_SEEN.add(key)
        if kind == "function":
            fv = dict(vars_ if vars_ is not None else {"HOME": HOME, "USERPROFILE": HOME})
            for i, a in enumerate(rest[:9], 1):
                fv[str(i)] = a
            fv["@"] = fv["*"] = " ".join(rest)
            # "$@" is every argument as its own word, quotes and spaces intact; ${1+"$@"} is the same thing
            # (Claude Code's grep wrapper ends in `command grep ${1+"$@"}`). Pasting the words in unquoted split
            # a pattern like '(src|href)="[^"]+"' into file names that were never given.
            body = re.sub(r'\$\{1\+"\$(?:@|\{@\})"\}|"\$\{@\}"', '"$@"', text)
            body = body.replace('"$@"', " ".join(shlex.quote(a) for a in rest))
            effs = analyze(body, ecwd, depth + 1, fv)
            label = "the function %s from the owner's shell profile" % word
        else:
            if kind == "alias":                              # the value takes the name's place; the words stay
                line = " ".join([text] + [shlex.quote(t) for t in rest])
            else:                                            # a global alias takes its word's place anywhere in the line
                line = " ".join(text if t == word else shlex.quote(t) for t in toks)
            effs = analyze(line, ecwd, depth + 1, vars_)
            label = "the alias %s='%s' from the owner's shell profile" % (word, text[:80])
        for e in effs:
            e.setdefault("shell_def", label)
        out += effs
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
        DEFS_SEEN.clear()
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
        if "/" not in toks[0]:                              # a name the owner's shell defines runs its definition
            effects += guarded(profile_def_effects, [], toks, state["cwd"], depth, vars_)
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
               "toks": toks, "fed_by": fed}
        guarded(note_written, None, prog, rest, my_bodies, eff["writes"])
        handed_text, handed_file = guarded(stdin_text, (None, None), fed, rest, ecwd)

        # PowerShell, cmd.exe and the terminal recorders carry a command the same way a shell does
        low = prog.lower()
        if low in ("pwsh", "powershell", "pwsh.exe", "powershell.exe"):
            for jj, t in enumerate(rest):
                if t.lower() in ("-c", "-command", "-commandwithargs") and jj + 1 < len(rest):
                    effects += guarded(analyze_powershell, [], rest[jj + 1], ecwd, depth + 1)
                    effects += analyze(rest[jj + 1], ecwd, depth + 1, vars_)
                    break
        elif low in ("cmd", "cmd.exe"):
            for jj, t in enumerate(rest):
                if t.lower() in ("/c", "/k") and jj + 1 < len(rest):
                    effects += guarded(cmd_exe_effects, [], " ".join(rest[jj + 1:]), ecwd)
                    effects += analyze(" ".join(rest[jj + 1:]), ecwd, depth + 1, vars_)
                    break
        elif low == "script" and depth < 6:                  # script [-q] [file] command...
            jj = 0
            while jj < len(rest) and rest[jj].startswith("-"):   # script's own flags come first, then the file
                jj += 1
            if len(rest) > jj + 1:
                effects += analyze(" ".join(rest[jj + 1:]), ecwd, depth + 1, vars_)
        elif low == "expect" and depth < 6:                  # expect -c 'spawn command...'
            for jj, t in enumerate(rest):
                if t == "-c" and jj + 1 < len(rest):
                    for m in re.finditer(r"\bspawn\s+([^;\n]+)", rest[jj + 1]):
                        effects += analyze(m.group(1), ecwd, depth + 1, vars_)

        if prog in ALL_SHELLS:
            if eff["writes"]:
                effects.append(eff)
            lead = []
            for t in rest:
                if not t.startswith("-") or t == "--":
                    break
                lead.append(t)
            if any(t == "--noexec" or not t.startswith("--") and "n" in t[1:] for t in lead):
                continue                                     # `bash -n x.sh` (noexec) reads the syntax and runs nothing
            for jj, t in enumerate(rest[:-1]):              # `bash --rcfile F -i`: F runs before the command
                if t in ("--rcfile", "--init-file"):
                    effects += script_body_effects(resolve(rest[jj + 1], ecwd), ecwd, depth, vars_, mode="shell")
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
        if prog == "alias":                                  # `alias gp='git push -f'`: the value is read where it is
            if eff["writes"]:                                # written, as a function body is (a later line of a bash
                effects.append(eff)                          # script with expand_aliases on runs it)
            for t in rest:
                if "=" in t and not t.startswith("-"):
                    for e in analyze(t.split("=", 1)[1], ecwd, depth + 1, vars_):
                        e.setdefault("shell_def", "the alias %s defined in this command" % t.split("=", 1)[0])
                        effects.append(e)
            continue
        if prog == "hash":                                   # `hash -p FILE name` (bash), `hash name=FILE` (zsh):
            if eff["writes"]:                                # the name runs FILE from now on
                effects.append(eff)
            for jj, t in enumerate(rest):
                f = rest[jj + 1] if t == "-p" and jj + 1 < len(rest) else \
                    (t.split("=", 1)[1] if "=" in t and not t.startswith("-") else None)
                if f:
                    effects += script_body_effects(resolve(f, ecwd), ecwd, depth, vars_)
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
            for jj, t in enumerate(rest[:-1]):              # `find . -exec grep x {} +` searches every file find walks
                if t in ("-exec", "-execdir", "-ok", "-okdir") and \
                        os.path.basename(rest[jj + 1]) in ("grep", "egrep", "fgrep", "rg", "ag", "ack"):
                    effects.append(dict(eff, prog=os.path.basename(rest[jj + 1]), kind="read", hunt=True,
                                        recursive=True, targets=[], writes=[]))
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
            # The value after -A/-B/-C/-m/-e/-f/-g/-t is an option value, not a file (receipt 658b551ec2664c49).
            words, skip, pattern_given = [], False, False
            for t in rest:
                if skip:
                    skip = False
                    continue
                if t in GREP_VALUE_FLAGS:
                    skip, pattern_given = True, pattern_given or t in ("-e", "--regexp", "-f", "--file")
                elif not t.startswith("-") and not re.match(r"^\d*[<>]", t):
                    words.append(t)
            operands = words if pattern_given else words[1:]
            recursive = has_flag(flags, "r") or has_flag(flags, "R") or "--recursive" in flags or prog in ("rg", "ag", "ack")
            if prog == "rg" and "--files" in flags:
                eff["kind"] = "list"                         # `rg --files` lists names like `ls`; it reads no content
            elif not operands and (not recursive or piped) and not via_xargs:
                eff["kind"] = "filter"                       # `cargo test | rg x` filters the pipe, as grep does;
                                                             # under xargs the pipe carries file names to search
            else:
                eff["kind"] = "read"
            # A hunt for knowledge: a search with no file named walks the folder tree (`grep -rn x`, `rg x`), and
            # so does one aimed at a folder, a pattern or a path it cannot see; a search of a memory store Vestige
            # already answers (receipts, MEMORY.md) is a lookup to make there. `grep -c ERROR app.log` is neither.
            eff["hunt"] = eff["kind"] == "read" and (
                not operands or not ecwd or
                any(not is_file(resolve(o, ecwd)) or is_knowledge_file(resolve(o, ecwd)) for o in operands))
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
            found = cmd_exe_effects(handed, cwd)
            if '\\"' in stmt:
                # PowerShell does not read a backslash as an escape, so `\"` ends the string there and
                # cmd.exe is handed a path with a stray backslash at its start or end. A lone backslash is
                # the root of the current drive, so a recursive delete is judged at that worst case.
                for e in found:
                    if e.get("recursive"):
                        e["targets"] = sorted(set(e["targets"]) | {(os.path.splitdrive(cwd or "")[0] or "") + "/"})
                        e["quote_slip"] = True
            effects += found
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


# Caches every tool rebuilds on demand (receipts 00815a8250a270bd, 4197d3ecdf764754, e018aa9ad8313988).
CACHE_ROOTS = (pj(HOME, "Library", "Caches"), pj(HOME, ".cache"), pj(HOME, ".npm", "_cacache"),
               pj(HOME, ".cargo", "registry"), pj(HOME, ".cargo", "git"))     # never ~/.cargo/bin: installed tools
OWNER_SCRATCH = []


def owner_scratch():
    """The owner's own scratch folders, `scratch_ok` in commitments.json, which only the owner writes (OP-000
    stops an agent's write to it). On a machine whose owner forbids /tmp, these are where agents work. A root
    that is the home folder, holds it, or sits inside the gate's own home is never scratch."""
    if not OWNER_SCRATCH:
        home, gate_home = os.path.realpath(HOME), os.path.realpath(OP_HOME)
        roots = []
        for p in load_config().get("scratch_ok") or []:
            r = norm(os.path.expanduser(str(p)), None).rstrip("/")
            real = os.path.realpath(r)
            if not r or home == real or home.startswith(real + "/") or real == gate_home or \
                    real.startswith(gate_home + "/"):
                continue
            roots.append(r)
        OWNER_SCRATCH.append(roots)
    return OWNER_SCRATCH[0]


def inside_for_real(t, root, root_too=False):
    """t is inside root as written AND as the file system resolves it (`..`, symlinks): a link in the scratch
    folder that points at the home folder is not scratch, and neither is scratch/../clients."""
    if not (t.startswith(root + "/") or (root_too and t == root)):
        return False
    rt, rr = os.path.realpath(t), os.path.realpath(root)
    return rt.startswith(rr + "/") or (root_too and rt == rr)


def is_scratch(path):
    build = {b.casefold() for b in BUILD_DIRS} if IS_WINDOWS else BUILD_DIRS
    for t in variants(path):
        if any(t == s or t.startswith(s + "/") for s in SCRATCH_OK):
            return True
        if any(t == s or t.startswith(s + "/") for s in scratch_roots()):
            return True
        if any(inside_for_real(t, s, root_too=True) for s in CACHE_ROOTS):
            return True
        if any(inside_for_real(t, s) for s in owner_scratch()):   # inside the owner's scratch, never the root itself
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
    r44 = check_persistence_file(w, None)
    if r44:
        hits.append(("OP-044", "%s: %s" % (verb, r44),
                     "Installing persistent launchd daemons, scheduled cron jobs, or services is owner-only."))
    if any(ssh_authorized_target(p) for p in variants(w)):
        hits.append(("OP-029", "%s SSH authorized keys %s" % (verb, w),
                     "The owner manages SSH access directly; do not install or append authorized keys."))
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
    if any(is_inside(w, d) for d in SHELL_SNAPSHOT_DIRS):
        hits.append(("OP-008", "%s agent shell snapshot %s" % (verb, w),
                     "Claude Code sources this file before every later Bash call of the session, so a line in it "
                     "changes what each later command runs; propose the line to the owner instead of writing it."))
    elif is_init:
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


# --------------------------------------------------------------------------- #
# machine wreck: disks, filesystems, and permission sweeps (0.3.12)
# --------------------------------------------------------------------------- #
# A real disk. Loop devices and device-mapper nodes are left out on purpose: image builders make
# filesystems on them all day.
DISK_RE = re.compile(r"^/dev/(r?disk\d|sd[a-z]|nvme\d|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d|md\d)")
DISK_REDIRECT_RE = re.compile(r">\s*/dev/(r?disk\d|sd[a-z]|nvme\d|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d|md\d)")
DISKUTIL_ERASE = {"erasedisk", "erasevolume", "zerodisk", "secureerase", "reformat", "partitiondisk",
                  "randomdisk", "deletecontainer", "deletevolume", "erasecd"}
WRECK_HINT = ("Work on an image file or a loop device, never on the disk itself; change permissions "
              "on the one folder you mean, inside the project. The owner does the rest by hand.")
SELF_OWNER_RE = re.compile(r"^(\$\{?USER\}?|\$\(whoami\)|\$\(id -un?\)|`whoami`|%s)(:.*)?$" % re.escape(
    os.environ.get("USER") or os.environ.get("USERNAME") or "\0"))
NO_ACCESS_MODE_RE = re.compile(r"^(0?000|[ugoa]*-[rwx]*r[rwx]*|[ugoa]*=)$")


def _wide(path, cwd):
    """A root, a top-level folder, a home folder or one above it, or a parent of the working directory."""
    return path == "/" or bool(re.match(r"^/[^/]+$", path)) or is_same_or_ancestor(path, HOME) \
        or (is_same_or_ancestor(path, cwd) and not same_path(path, cwd))


def machine_wreck(cmd, cwd):
    """OP-014 for one shell command. Reads the command's words and runs nothing."""
    found = []
    if DISK_REDIRECT_RE.search(mask_quotes(cmd)):
        found.append(("OP-014", "output redirected onto a disk device", WRECK_HINT))
    for seg in split_segments(cmd):
        _, _, _, bare, inner = root_form(tokenize(seg))
        if inner is not None:
            found += machine_wreck(inner, cwd)
            continue
        if not bare:
            continue
        prog, args = os.path.basename(bare[0]).lower(), bare[1:]
        disks = [a for a in args if DISK_RE.match(a)]
        if prog == "dd":
            hit = next((a[3:] for a in args if a.startswith("of=") and DISK_RE.match(a[3:])), None)
            if hit:
                found.append(("OP-014", "dd writes over %s" % hit, WRECK_HINT))
        elif (prog.startswith(("mkfs", "newfs")) or prog in ("mke2fs", "mkswap", "blkdiscard", "shred")) and disks:
            found.append(("OP-014", "%s on %s" % (prog, disks[0]), WRECK_HINT))
        elif prog == "wipefs" and disks:
            found.append(("OP-014", "wipefs erases the signatures on %s" % disks[0], WRECK_HINT))
        elif prog == "sgdisk" and disks and any(a in ("-Z", "--zap-all", "-o", "--clear", "-z", "--zap") for a in args):
            found.append(("OP-014", "sgdisk wipes the partition table of %s" % disks[0], WRECK_HINT))
        elif prog == "parted" and disks and "mklabel" in args:
            found.append(("OP-014", "parted writes a new partition table on %s" % disks[0], WRECK_HINT))
        elif prog == "diskutil":
            words = [a.lower() for a in args if not a.startswith("-")]
            verb = next((w for w in words[:2] if w in DISKUTIL_ERASE), None)
            if verb:
                found.append(("OP-014", "diskutil %s" % verb, WRECK_HINT))
        elif prog in ("chmod", "chown", "chgrp"):
            flags = [a for a in args if a.startswith("-") and not re.match(r"^-[rwxXst]+$", a)]
            if not any(a == "--recursive" or re.match(r"^-[A-Za-z]*R", a) for a in flags):
                continue
            plain = [a for a in args if a not in flags]
            if len(plain) < 2:
                continue
            what, targets = plain[0], plain[1:]
            for t in targets:
                if re.search(r"[$`*?]", t.replace("$HOME", "").replace("${HOME}", "")):
                    continue
                path = norm(t, cwd)
                if _wide(path, cwd):
                    found.append(("OP-014", "recursive %s across %s" % (prog, path), WRECK_HINT))
                elif is_inside(path, cwd) or is_scratch(path):
                    continue
                elif prog != "chmod" and not SELF_OWNER_RE.match(what):
                    found.append(("OP-014", "recursive %s of %s to %s, outside the working directory" % (prog, path, what), WRECK_HINT))
                elif prog == "chmod" and NO_ACCESS_MODE_RE.match(what):
                    found.append(("OP-014", "recursive chmod %s removes access to %s" % (what, path), WRECK_HINT))
    return list(dict.fromkeys(found))[:2]


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
FETCH_SUBST_RE = re.compile(r"\b(?:(?:ba|z)?sh|source|\.)\s+<\(\s*(?:curl|wget)\b[^)]*?(https?://[^\s'\")]+)")
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
    for rx in (FETCH_PIPE_RE, FETCH_SUBST_RE):
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


CURSOR_FORWARDED = {"Shell", "Write", "Edit", "MultiEdit", "Delete", "NotebookEdit"}
_CURSOR_PORT = {}


def cursor_forwards(payload):
    """True when Cursor's own hooks already send this call to the gate through cursor-hook.py (shell
    commands, MCP calls and file edits), so the copy Cursor runs from ~/.claude/settings.json stands down.
    Read only when cursor-hook.py is registered in ~/.cursor/hooks.json; otherwise this copy judges."""
    if "on" not in _CURSOR_PORT:
        try:
            with open(pj(HOME, ".cursor", "hooks.json"), encoding="utf-8") as f:
                _CURSOR_PORT["on"] = "cursor-hook.py" in f.read()
        except OSError:
            _CURSOR_PORT["on"] = False
    tool = str((payload or {}).get("tool_name") or "")
    return _CURSOR_PORT["on"] and (tool in CURSOR_FORWARDED or tool.startswith("MCP:"))


def session_of(payload):
    """The host's session for this call. Cursor names it conversation_id on tool events."""
    p = payload or {}
    return str(p.get("session_id") or p.get("sessionId") or p.get("conversation_id") or p.get("conversationId") or "")[:64]


def call_of(payload):
    """The host's id for this one tool call, so `coverage` can find the calls that never reached the gate."""
    p = payload or {}
    return str(p.get("tool_use_id") or p.get("toolUseId") or p.get("call_id") or p.get("tool_call_id") or "")[:80]


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


# --------------------------------------------------------------------------- #
# the floor: what a recursive delete really reaches (0.3.11)
# --------------------------------------------------------------------------- #
FLOOR_RULES = {"OP-012"}
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


def floor_hits(e, cwd):
    """OP-012 for one recursive delete. A resolved target is judged as it stands. A target only the
    running shell can fill in is judged at its worst case, every unknown piece empty, because that
    is how `rm -rf "$DIR/"` becomes a delete of the root."""
    if e.get("kind") != "delete" or not e.get("recursive") or e.get("filtered"):
        return []
    try:
        words = shlex.split(e.get("seg") or "", posix=True)[1:]
    except ValueError:
        words = []
    words = [w for w in words if w and not w.startswith("-")]
    found = []
    for t in e.get("targets") or []:
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


# ---- rewind: work an allowed call was about to destroy, kept so the owner can put it back -------- #
REWIND_TIME_S = 1.5                      # the most keeping may add to one call
REWIND_FILE_CAP = 20000
REWIND_COPY_CAP = 256 * 1024 * 1024      # a plain copy (no copy-on-write clone) stops here
REWIND_KEEP_S = 7 * 86400
REWIND_MAX = 500
# Partial edits are left to the host's own checkpoints. A Write tool's whole-file replacement is not:
# it is kept like a shell overwrite (the case in anthropics/claude-code#67917).
REWIND_HOST_TOOLS = {"edit", "multiedit", "notebookedit", "apply_patch", "str_replace_editor", "edit_file"}


def rewind_dir():
    return pj(OP_HOME, "rewind")


def rewind_plan(effects, cwd):
    """The existing work this call would destroy, as (paths, git rows). Paths are delete targets that
    hold work nothing could bring back (the Kept-safe reading), existing files it overwrites, and the
    changed or untracked files a git discard throws away; a git row records a repository's stashes
    before a stash drop or clear. Scratch and build output are never kept."""
    deletes, paths, git = [], [], []
    for e in effects or []:
        kind = e.get("kind")
        if kind == "delete" and not e.get("filtered"):
            deletes += [t for t in e.get("targets") or [] if isinstance(t, str) and os.path.lexists(t) and not is_scratch(t)]
        paths += [w for w in e.get("writes") or [] if isinstance(w, str) and os.path.isfile(w) and not is_scratch(w)]
        if kind == "git":
            root = git_top(e.get("git_dir") or e.get("cwd") or cwd)
            sub, ga = e.get("sub", ""), e.get("gargs", [])
            if not root or not guarded(git_discards, None, sub, ga, root, e.get("saved") or (False, False)):
                continue
            if sub == "stash":
                shas = (git_out(root, ["stash", "list", "--format=%H"]) or "").split()
                if shas:
                    git.append({"root": root, "sub": sub, "stashes": shas[:50]})
                continue
            wanted = ("??",) if sub == "clean" else None
            ignored = sub == "clean" and any(re.match(r"^-[a-zA-Z]*x", a) for a in ga)
            for state, rel in _status(root, ["--untracked-files=all"] + (["--ignored=matching"] if ignored else []), None) or []:
                full = pj(root, rel.rstrip("/"))
                if any(part in BUILD_DIRS for part in rel.split("/")) or not os.path.isfile(full):
                    continue
                if (state in wanted or (ignored and state == "!!")) if wanted else state not in ("??", "!!"):
                    paths.append(full)
    if deletes and guarded(_lost_list, [], deletes):
        paths = deletes + paths
    return list(dict.fromkeys(paths))[:REWIND_FILE_CAP], git


def _same_device(a, b):
    try:
        return os.lstat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def _keep_path(src, dst, budget):
    """Keep one path in a snapshot: a copy-on-write clone where the disk has one (instant, no extra
    space), else a plain copy within the time and size budget. Returns (method, complete)."""
    os.makedirs(os.path.dirname(dst), mode=0o700, exist_ok=True)
    left = budget["deadline"] - time.time()
    if left <= 0:
        return "none", False
    if not IS_WINDOWS and _same_device(src, os.path.dirname(dst)):
        argv = ["/bin/cp", "-c", "-R", "-p", src, dst] if sys.platform == "darwin" else ["cp", "-a", "--reflink=always", src, dst]
        try:
            if subprocess.run(argv, capture_output=True, timeout=max(0.2, left)).returncode == 0:
                return "clone", True
        except subprocess.TimeoutExpired:
            return "clone", False
        except OSError:
            pass
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst, ignore_errors=True)
        elif os.path.lexists(dst):
            os.remove(dst)
    complete = True

    def one(s, d):
        if time.time() > budget["deadline"] or budget["bytes"] > REWIND_COPY_CAP or budget["files"] >= REWIND_FILE_CAP:
            return False
        try:
            if os.path.islink(s):
                os.symlink(os.readlink(s), d)
            else:
                shutil.copy2(s, d)
                budget["bytes"] += os.lstat(s).st_size
            budget["files"] += 1
        except OSError:
            pass
        return True

    if os.path.isdir(src) and not os.path.islink(src):
        for base, _dirs, names in os.walk(src):
            out = pj(dst, os.path.relpath(base, src))
            os.makedirs(out, mode=0o700, exist_ok=True)
            for n in names:
                if not one(pj(base, n), pj(out, n)):
                    return "copy", False
    else:
        complete = one(src, dst)
    return "copy", complete


def _git_holds(path):
    """True when git already holds this file's current bytes: tracked, and unchanged in its repository."""
    root = git_top(path)
    if not root:
        return False
    rel = os.path.relpath(path, root)
    tracked = git_out(root, ["ls-files", "--", rel])
    return bool(tracked and tracked.strip()) and git_out(root, ["status", "--porcelain", "--", rel]) == ""


def rewind_keep(effects, cwd, preview, tool, label="before"):
    """Before a call the gate lets through destroys work, keep that work. Returns None when the call
    destroys nothing worth keeping, else a record for the receipt: {id, paths, git, complete, needed,
    missed}, plus `error` when keeping failed. `needed` counts what should have been kept and `missed`
    what was not, so a copy that should have been kept and was not never reads as nothing to keep.
    Edit-style tools are left to the host's own checkpoints; a Write tool's whole-file replacement of a
    file git does not already hold is kept, like a shell overwrite."""
    name = str(tool or "").lower()
    if name in REWIND_HOST_TOOLS:
        return None
    paths, git = rewind_plan(effects, cwd)
    if name in WRITE_TOOLS:
        paths = [p for p in paths if not _git_holds(p)]
    if not paths and not git:
        return None
    needed = len(paths) + len(git)
    try:
        return rewind_take(paths, git, preview, cwd, label)
    except Exception as exc:
        if os.environ.get("OPERATOR_DEBUG"):
            raise
        return {"id": None, "paths": 0, "git": 0, "complete": False, "needed": needed, "missed": needed,
                "error": type(exc).__name__}


def rewind_now(effects, cwd, preview, tool):
    """rewind_keep for the hook: a defect in reading what the call destroys is reported as a miss of
    unknown size, never as nothing to keep."""
    try:
        return rewind_keep(effects, cwd, preview, tool)
    except Exception as exc:
        if os.environ.get("OPERATOR_DEBUG"):
            raise
        DEFECTS.append("rewind_keep")
        return {"id": None, "paths": 0, "git": 0, "complete": False, "needed": None, "missed": None,
                "error": type(exc).__name__}


def rewind_must_hold(kept):
    """True when work that should have been kept was not, all or part of it, and its size is known."""
    return bool(kept) and bool(kept.get("needed")) and bool(kept.get("missed") or kept.get("error"))


def rewind_miss_line(kept):
    """One sentence on what Rewind could not keep, or '' when nothing was missed."""
    if not kept or not (kept.get("missed") or kept.get("error")):
        return ""
    if kept.get("needed") is None:
        return "the gate could not read what this call would destroy (%s), so nothing was kept" % kept.get("error")
    n = int(kept["needed"])
    if kept.get("error"):
        return "keeping a copy failed (%s), so %d path%s would be gone for good" % (kept["error"], n, "" if n == 1 else "s")
    m = int(kept["missed"])
    return "%d of %d path%s could not be kept in time, so %s would be gone for good" % (
        m, n, "" if n == 1 else "s", "it" if m == 1 else "they")


def rewind_hold(source, payload, cwd, tool, preview, meta, kept):
    """An otherwise allowed call would destroy work Rewind could not keep. Under OP-011 it waits for the
    owner like any stop: a permit lets this one call through (returns True); otherwise it stops here."""
    rid = "OP-011"
    name, _base, why = RULES[rid]
    session = session_of(payload)
    adigest = digest_of({"tool": tool, "cwd": cwd, "hits": [rid], "preview": preview})[:24]
    base_rec = {"source": source, "session": session, "call": call_of(payload), "cwd": cwd, "tool": tool,
                "action_digest": adigest, "action_preview": redact(preview),
                "normalized": redact(meta.get("normalized", "")), "transforms": meta.get("transforms", []),
                "rewind": kept}
    if consume_permit(adigest, session):
        write_receipt(dict(base_rec, decision="ALLOW", via="permit", commitments=[rid]))
        return True
    detail = "Rewind: " + rewind_miss_line(kept)
    rd = write_receipt(dict(base_rec, decision="STOP", commitment=rid, rule=name, commitments=[rid], detail=redact(detail)))
    stop(["OPERATOR: STOPPED (%s %s)" % (rid, name),
          "Why: %s -- Rewind keeps a copy before the gate lets a call destroy work, and this time %s." % (why, rewind_miss_line(kept)),
          "Compliant path: commit or stash the work first (`git stash -u`), or copy it somewhere safe.",
          "No action was taken. Do not retry this action or a variation of it.",
          "Only the owner can allow it once, from their own terminal: %s approve %s" % (self_cmd(), adigest),
          "Receipt: %s" % ((rd or "unavailable")[:16]),
          "Operator receipt: %s" % (rd or "unavailable")])


def rewind_note(kept):
    """A call that goes ahead although Rewind missed work says so on stderr, beside its receipt."""
    line = rewind_miss_line(kept)
    if line:
        sys.stderr.write("Operator Rewind: %s. The call goes ahead; the receipt records what was not kept.\n" % line)


def rewind_take(paths, git, preview, cwd, label="before"):
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha256((str(preview) + repr(time.time())).encode()).hexdigest()[:6]
    base = pj(rewind_dir(), rid)
    os.makedirs(base, mode=0o700)
    budget = {"deadline": time.time() + REWIND_TIME_S, "files": 0, "bytes": 0}
    items, complete = [], True
    for i, p in enumerate(paths):
        method, ok = _keep_path(p, pj(base, "files", "%04d" % i), budget)
        if method == "none":
            complete = False
            break
        items.append({"path": p, "stored": "files/%04d" % i, "dir": os.path.isdir(p) and not os.path.islink(p),
                      "method": method, "complete": ok})
        complete = complete and ok
    missed = len(paths) - sum(1 for it in items if it["complete"])
    manifest = {"id": rid, "t": time.time(), "label": label, "preview": redact(str(preview))[:240], "cwd": cwd,
                "items": items, "git": git, "complete": complete, "needed": len(paths) + len(git), "missed": missed,
                "gate_version": VERSION}
    with open(pj(base, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    guarded(rewind_prune, None)
    return {"id": rid, "paths": len(items), "git": len(git), "complete": complete,
            "needed": len(paths) + len(git), "missed": missed}


def rewind_prune():
    """Snapshots older than a week go, and never more than REWIND_MAX are kept."""
    try:
        names = sorted(n for n in os.listdir(rewind_dir()) if os.path.isfile(pj(rewind_dir(), n, "manifest.json")))
    except OSError:
        return
    cutoff = time.time() - REWIND_KEEP_S
    for i, n in enumerate(names):
        p = pj(rewind_dir(), n)
        try:
            old = os.path.getmtime(pj(p, "manifest.json")) < cutoff
        except OSError:
            old = False
        if old or i < len(names) - REWIND_MAX:
            shutil.rmtree(p, ignore_errors=True)


def rewind_manifests():
    out = []
    try:
        names = sorted(os.listdir(rewind_dir()), reverse=True)
    except OSError:
        return out
    for n in names:
        try:
            with open(pj(rewind_dir(), n, "manifest.json"), encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    return out


def rewind_restore(m):
    """Put the kept work back. Nothing is deleted: every kept file is copied over its path, and the
    files it replaces are kept first in a new snapshot, so the restore can itself be rewound."""
    base = pj(rewind_dir(), m["id"])
    plan, back = [], []
    for it in m.get("items") or []:
        src = pj(base, it["stored"])
        if it.get("dir") and os.path.isdir(src):
            for b, _d, names in os.walk(src):
                for n in names:
                    rel = os.path.relpath(pj(b, n), src)
                    plan.append((pj(b, n), pj(it["path"], rel)))
        elif os.path.lexists(src):
            plan.append((src, it["path"]))
    replaced = [dst for _s, dst in plan if os.path.isfile(dst)]
    if replaced:
        back.append(rewind_take(replaced, [], "before restoring %s" % m["id"], m.get("cwd") or "", "before-restore"))
    done = 0
    for s, d in plan:
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
            else:
                shutil.copy2(s, d)
            done += 1
        except OSError:
            pass
    stashes = 0
    for g in m.get("git") or []:
        have = set((git_out(g["root"], ["stash", "list", "--format=%H"]) or "").split())
        for sha in reversed(g.get("stashes") or []):
            if sha not in have:
                r = subprocess.run(["git", "-C", g["root"], "-c", "core.fsmonitor=false", "stash", "store",
                                    "-m", "restored by operator rewind %s" % m["id"], sha], capture_output=True, text=True)
                stashes += r.returncode == 0
    return {"files": done, "of": len(plan), "stashes": stashes, "before": back[0]["id"] if back else None}


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


# ---- told once: a stop says what the record already knows about the same action -------------------- #
TOLD_DAYS = 14                           # day files of receipts read when a call is stopped
TOLD_DRAFT_AT = 3                        # the third stop of one action, never allowed, drafts a law


def law_drafts_path():
    return pj(OP_HOME, "state", "law-drafts.json")


def told_before(adigest, current):
    """What the last TOLD_DAYS days of receipts say about this exact action (the same action digest):
    earlier stops, the sessions they came from, and the owner's permits. None when there is no history."""
    rdir = pj(OP_HOME, "receipts")
    try:
        names = sorted(n for n in os.listdir(rdir) if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", n))[-TOLD_DAYS:]
    except OSError:
        return None
    stops, permits = [], []
    for n in names:
        try:
            with open(pj(rdir, n), encoding="utf-8") as f:
                for line in f:
                    if adigest not in line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("action_digest") != adigest or r.get("digest") == current:
                        continue
                    if r.get("decision") == "STOP":
                        stops.append(r)
                    elif r.get("decision") == "PERMIT_GRANTED":
                        permits.append(r)
        except OSError:
            continue
    if not stops and not permits:
        return None
    return {"stops": len(stops), "sessions": len({s.get("session") for s in stops}),
            "first": stops[0].get("ts") if stops else None, "last": stops[-1].get("ts") if stops else None,
            "permits": len(permits), "last_permit": permits[-1].get("ts") if permits else None}


def _when(ts):
    try:
        return time.strftime("%m-%d %H:%M UTC", time.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return str(ts or "?")


def told_line(t):
    """One sentence for the stop message, from told_before."""
    parts = []
    if t["stops"]:
        parts.append("this exact action was stopped %d time%s before, in %d session%s (first %s, last %s)"
                     % (t["stops"], "" if t["stops"] == 1 else "s", t["sessions"], "" if t["sessions"] == 1 else "s",
                        _when(t["first"]), _when(t["last"])))
    if t["permits"]:
        parts.append("the owner allowed it once %d time%s, last %s" % (t["permits"], "" if t["permits"] == 1 else "s", _when(t["last_permit"])))
    else:
        parts.append("the owner has not allowed it")
    return "Told before: %s." % "; ".join(parts)


def draft_law(adigest, rid, rule_name, effects, stops):
    """The third stop of one action drafts a law the owner may adopt, so every session starts knowing.
    Returns the draft's id, or None when no program can be named for the law to match."""
    eff = next((e for e in effects or [] if e.get("prog")), None)
    if not eff:
        return None
    prog = posixpath.basename(str(eff["prog"]))
    text = str(eff.get("text") or eff.get("seg") or "")[:160]
    if not prog or not text:
        return None
    cid = "told-%s-%s" % (rid.lower(), adigest[:8])
    law = {"commitment_id": cid, "version": 1, "attribution": "owner", "authority": "owner", "state": "active",
           "source_class": "owner_record", "scope": {"project_id": "all"},
           "directive": ("Agents tried `%s` %d times and the gate stopped it each time under %s %s. Do not try it "
                         "again; take the safe path the stop names, or ask the owner." % (redact(text)[:120], stops, rid, rule_name)),
           "match": {"prog_any_of": [prog], "shape_any_of": [re.escape(text)]},
           "on_violation": {"decision": "stop"}}
    path = law_drafts_path()
    try:
        with open(path, encoding="utf-8") as f:
            drafts = json.load(f)
    except (OSError, ValueError):
        drafts = {}
    drafts[cid] = {"law": law, "drafted": time.time(), "action_digest": adigest, "stops": stops}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=1)
    os.replace(tmp, path)
    return cid


def told_lines(adigest, rid, rule_name, effects, current):
    """The lines a stop adds from the record: what happened before, and a drafted law on the third stop."""
    t = told_before(adigest, current)
    if not t:
        return []
    lines = [told_line(t)]
    if t["stops"] + 1 >= TOLD_DRAFT_AT and not t["permits"]:
        cid = guarded(draft_law, None, adigest, rid, rule_name, effects, t["stops"] + 1)
        if cid:
            lines.append("Drafted for the owner, so every session starts knowing this: `%s laws adopt %s`." % (self_cmd(), cid))
    return lines


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


def classify_effect(e, cfg, cwd):
    """Yield (rule_id, reason_detail, rewrite_hint) for one effect."""
    hits = hard_stop_effect_hits(e, cfg, cwd)
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

    if kind == "delete":
        hits += guarded(floor_hits, [], e, ecwd or cwd)

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
                             "Move it to a dated folder under your home folder, or delete only build outputs "
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
                    cur = subprocess.run(["git", "-C", ecwd or cwd or ".", "-c", "core.fsmonitor=false", "symbolic-ref", "--short", "HEAD"],
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
            hits.append(("OP-006", "fly %s" % " ".join(args[:2]), "Deploys to Fly are owner-only; "
                         "ask the owner for a one-time permit, or deploy to a staging app."))
        if prog == "supabase" and args[:2] in (["db", "push"], ["db", "reset"], ["migration", "up"],
                                                ["functions", "deploy"], ["db", "remote"]):
            hits.append(("OP-006", "supabase %s" % " ".join(args[:2]), "Apply migrations to staging only, with a permit."))
        if prog == "stripe" and re.search(r"--live|sk_live|rk_live", text) and \
                args[1:2] not in (["list"], ["retrieve"], ["search"]) and args[:1] not in (["status"], ["version"], ["logs"]):
            hits.append(("OP-006", "stripe live action", "Use Test mode; live billing needs the owner's one-time permit."))
        if prog in ("vercel",) and "--prod" in flags or prog == "wrangler" and args[:1] in (["deploy"], ["publish"]):
            hits.append(("OP-006", "%s production deploy" % prog, "Ask for a one-time permit."))
        if prog == "vestige" and args[:1] in (["gc"], ["purge"], ["wipe"], ["reset"], ["erase"]):
            hits.append(("OP-002", "vestige %s" % args[0], "Run `vestige backup` and ask the owner before erasing memory."))
        # `mode` alone prints the mode and `--help` prints usage (receipts 909c12c05d44618d, 29ca345d61d9b7e7)
        more = [a for a in args[1:] if not re.match(r"^\d*[<>]", a)]
        reads_only = bool({"--help", "-h"} & set(flags + args[1:2])) or \
            (args[:1] in (["approve"], ["mode"]) and not more)
        if prog == "operator-gate" and args[:1] in (["approve"], ["mode"], ["install"],
                                                    ["uninstall"], ["onboard"]) and not reads_only:
            hits.append(("OP-000", "agent invoked `operator-gate %s`" % args[0],
                         "Approvals, installs and onboarding are run by the owner in their own terminal."))
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
    if hits and e.get("shell_def"):
        hits = [(r, "%s -- run by %s" % (d, e["shell_def"]), a) for (r, d, a) in hits]
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
# an assignment starts a line or follows a `;`: `import json; p='~/.operator/commitments.json'` names the gate
# path as surely as a line of its own does (2026-10-20, the scratch_ok lockdown)
INLINE_ASSIGN_RE = re.compile(r"(?:^|;)\s*([A-Za-z_]\w*)\s*=\s*((?:os\.path\.expanduser\(|Path\(|pathlib\.Path\()?\s*(['\"]).*?\3\s*\)?)\s*(?=;|$)", re.M)


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


def _pathlike(s):
    """A string literal that could be a path. Prose that mentions a path is not one (2026-10-02 receipt
    04bac93555155044: a memory note naming the gate's file was taken for a write target)."""
    return bool(s) and len(s) <= 400 and "\n" not in s and "`" not in s and not re.search(r"[.,;:!?]\s", s)


def _literal_path(expr, literals):
    expr = (expr or "").strip()
    m = INLINE_LITERAL_RE.match(expr)
    if m:
        return m.group(2) if _pathlike(m.group(2)) else None
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
    found, unresolved, unresolved_ops, blind = [], None, [], False
    for kind, call_re in (("write", INLINE_WRITE_CALL_RE), ("delete", INLINE_DELETE_CALL_RE)):
        for m in call_re.finditer(text):
            if not in_code(mask, text, m.start()):
                continue
            open_idx = text.index("(", m.start())
            args = _balanced_args(text, open_idx)
            name = text[m.start():open_idx].strip()
            if args is None:
                unresolved = "delete" if kind == "delete" else (unresolved or kind)
                blind = True
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
                blind = True
                continue
            for op in operands:
                lit = _literal_path(op, literals)
                if lit is None:
                    unresolved = "delete" if kind == "delete" else (unresolved or kind)
                    unresolved_ops.append(op)
                    continue
                target = resolve(lit, ecwd)
                if kind == "write":
                    found += write_hits(target, gate_files, "inline code writes to")
                else:
                    found += target_hits("delete", target, True, roots, gate_files)
    gate_strings = [q for q in re.findall(r"(?:~|\$HOME|\$\{HOME\}|/)[^\s'\"),;\]]*", text)
                    if touches_gate_home(resolve(q, ecwd)) or any(is_inside(resolve(q, ecwd), f) for f in gate_files)]
    # the gate path has to reach the unresolved call: through its operand, or through a name assigned
    # from a string that holds it. A gate path mentioned elsewhere in the code is data the code carries
    # (2026-10-10 receipts ed2189cf0c12fddb, 2c381096c323d522 and 0e9a1997b645271c).
    gate_names = {m.group(1) for m in INLINE_ASSIGN_RE.finditer(text) if any(g in m.group(2) for g in gate_strings)}
    mentions_gate = bool(gate_strings) and (blind or any(
        any(g in op for g in gate_strings) or any(n in gate_names for n in re.findall(r"[A-Za-z_]\w*", op))
        for op in unresolved_ops))
    return found, unresolved, mentions_gate


# ---- deterministic rewrites (proposed, never applied; one per original) ---- #
REWRITE_COUNTERS = pj(OP_HOME, "state", "rewrite-counters.json")
REWRITE_TTL_S = 24 * 3600


def rewrite_count_get(key):
    try:
        with open(REWRITE_COUNTERS, encoding="utf-8") as f:
            d = json.load(f)
        rec = d.get(key)
        if rec and time.time() - rec.get("ts", 0) < REWRITE_TTL_S:
            return rec.get("count", 0)
    except Exception:
        pass
    return 0


def rewrite_count_bump(key):
    try:
        os.makedirs(os.path.dirname(REWRITE_COUNTERS), exist_ok=True)
        with open(REWRITE_COUNTERS, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d = {k: v for k, v in d.items() if time.time() - v.get("ts", 0) < REWRITE_TTL_S}
    d[key] = {"count": d.get(key, {}).get("count", 0) + 1, "ts": time.time()}
    with open(REWRITE_COUNTERS, "w", encoding="utf-8") as f:
        json.dump(d, f)


def propose_rewrite(rewrite_id, eff):
    """Deterministic transform over the observed segment. Returns the proposed command, or None
    when no provably-weaker form exists (then the commitment is a plain STOP)."""
    text = eff.get("text") or ""
    prog = eff.get("prog", "")
    if rewrite_id == "append_dry_run":
        if prog in ("npm", "pnpm", "yarn") and " --dry-run" not in text:
            return re.sub(r"^(\s*(?:sudo\s+)?%s\s+publish\b)" % prog, r"\1 --dry-run", text, count=1)
        if prog == "cargo" and "--dry-run" not in text:
            return text.replace("cargo publish", "cargo publish --dry-run", 1)
        return None
    if rewrite_id == "drop_prod_markers":
        stripped = re.sub(r"\s*(--prod|--live|sk_live\S*|rk_live\S*|RAILS_ENV=production)", "", text)
        return stripped if stripped != text and stripped.strip() else None
    if rewrite_id == "move_to_trash":
        targets = e_targets_of(eff)
        if targets:
            return "mkdir -p ~/.operator/trash/$(date +%%s) && mv %s ~/.operator/trash/$(date +%%s)/" % \
                " ".join("'" + t.replace("'", "'\\''") + "'" for t in targets)
    return None


def e_targets_of(eff):
    return eff.get("targets") or eff.get("writes") or []


# ---- the taste layer ------------------------------------------------------- #
def run_taste(law, effects, cwd, action_digest):
    """Evaluate owner commitments against observed effects.
    Returns (stop_info or None, shadow_taste_hits, conflicts, candidate)."""
    project_id = os.path.basename((cwd or "").rstrip("/")) or "unknown"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hits, eligible, cand = evaluate_taste(law, effects, project_id, now_iso)
    conflicts = {}
    for c, _ in hits:
        ac = (c.get("scope") or {}).get("action_class")
        if ac:
            conflicts.setdefault(ac, set()).add(c["commitment_id"])
    conflicts = {k: sorted(v) for k, v in conflicts.items() if len(v) > 1}
    if not hits:
        return None, [], conflicts, cand
    top_c, evidence = hits[0]
    if rewrite_count_get(action_digest) >= 1 and (top_c.get("on_violation") or {}).get("decision") == "rewrite":
        return {"commitment": top_c, "evidence": evidence, "conflicts": conflicts,
                "hard": True, "rewrite": None, "cand": cand}, [], conflicts, cand
    proposal = None
    if (top_c.get("on_violation") or {}).get("decision") == "rewrite":
        rid = top_c["on_violation"].get("rewrite")
        for e in effects:
            proposal = propose_rewrite(rid, e)
            if proposal:
                break
        if not proposal:
            return {"commitment": top_c, "evidence": evidence, "conflicts": conflicts,
                    "hard": True, "rewrite": None, "cand": cand}, [], conflicts, cand
    return {"commitment": top_c, "evidence": evidence, "conflicts": conflicts,
            "hard": False, "rewrite": proposal, "cand": cand}, \
        [(c, ev) for c, ev in hits[1:]], conflicts, cand


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
        for m in re.finditer(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", str(patch), re.M):
            paths.append(norm(m.group(1), cwd))
    return paths


# --------------------------------------------------------------------------- #
# test integrity (OP-026): the checks that say the work is good are the owner's
# --------------------------------------------------------------------------- #
# An agent told to make the tests pass can get there by switching off what checks it: the hooks (--no-verify,
# HUSKY=0), the tests (deleted, skipped, focused, assertions taken out), the CI job (continue-on-error, a step
# removed) or the coverage floor; or it can say the tests pass when nothing ran (claude-code #40117, #46940).
# Each is visible in the call before it runs. Only files that exist are judged: a new test is never weakening.
TEST_CODE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rb", ".rs", ".java", ".kt", ".cs",
                 ".swift", ".php", ".ex", ".exs", ".dart", ".scala")
TEST_PATH_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|specs?)/|(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.(?:py|go|rb|exs?|dart)|"
                          r"[^/]+\.(?:test|spec)\.[cm]?[jt]sx?|[^/]+_spec\.rb|[^/]+Tests?\.(?:java|kt|cs|swift|php)|conftest\.py)$")
CI_PATH_RE = re.compile(r"(?:^|/)(?:\.github/workflows/[^/]+\.ya?ml|\.gitlab-ci\.ya?ml|\.circleci/config\.ya?ml|"
                        r"azure-pipelines[^/]*\.ya?ml|Jenkinsfile|\.buildkite/[^/]+\.ya?ml|bitbucket-pipelines\.ya?ml|"
                        r"\.travis\.ya?ml|\.drone\.ya?ml|\.pre-commit-config\.ya?ml|lefthook\.ya?ml|\.husky/[^/_][^/]*)$")
COV_PATH_RE = re.compile(r"(?:^|/)(?:pyproject\.toml|setup\.cfg|\.coveragerc|tox\.ini|pytest\.ini|package\.json|"
                         r"\.nycrc(?:\.json|\.ya?ml)?|(?:jest|vitest|vite|karma)\.config\.[cm]?[jt]s|\.?codecov\.ya?ml|"
                         r"phpunit\.xml(?:\.dist)?|Makefile|\.github/workflows/[^/]+\.ya?ml)$")
TEST_DECL_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+test\w*\s*\(|^\s*func\s+(?:\([^)]*\)\s*)?Test\w*\s*\(|#\[(?:\w+::)?test\b|@Test\b|"
    r"\[(?:Test|Fact|Theory|TestMethod|TestCase)\b|^\s*(?:public\s+)?function\s+test\w*\s*\(|^\s*func\s+test\w*\s*\(|"
    r"(?<![\w.$])(?:it|test|specify|scenario)(?:\.(?:skip|only|todo|each\s*\([^)]*\)|concurrent|failing))?\s*\(|"
    r"^\s*(?:it|specify|scenario)\s+['\"]|(?<![\w.$])(?:xit|xtest|fit)\s*\(", re.M)
TEST_SKIP_RE = re.compile(
    r"@pytest\.mark\.(?:skip|xfail)\b|\bpytest\.(?:skip|xfail)\s*\(|@unittest\.(?:skip|expectedFailure)\b|"
    r"\bself\.skipTest\s*\(|\bskip(?:if|If)?\s*\(\s*(?:True|1|true)\s*[,)]|"
    r"(?<![\w$])(?:it|test|describe|context|suite)\.(?:skip|only|todo|fixme)\b|"
    r"(?<![\w.$])(?:xit|xtest|xdescribe|xcontext|fit|fdescribe)\s*\(|\bt\.Skip(?:f|Now)?\s*\(|#\[ignore\b|"
    r"@(?:Disabled|Ignore)\b|\[Ignore\b|\bSkip\s*=\s*\"|\bmarkTest(?:Skipped|Incomplete)\s*\(|\bXCTSkip\b|"
    r"^\s*(?:skip|pending)\b(?:\s|$|\()", re.M)
TEST_ASSERT_RE = re.compile(                         # Rust's `.expect("..")` unwraps a value: it is not an assertion
    r"\bassert\w*\b|(?<![\w.])expect\s*\(|\bAssert\.\w+\s*\(|\bt\.(?:Error|Errorf|Fatal|Fatalf|Fail|FailNow)\s*\(|"
    r"\b(?:require|assert)\.\w+\s*\(|\.should\b|\bpytest\.raises\s*\(|(?<![\w.])verify\s*\(")
TRIVIAL_ASSERT_RE = re.compile(r"\bassert\s+(?:True|1|not\s+False)\s*(?:$|[,;#)])|\bassertTrue\(\s*True\s*\)|"
                               r"expect\(\s*true\s*\)\.toBe\(\s*true\s*\)|\bassert!\(\s*true\s*\)|"
                               r"\bassert\.ok\(\s*true\s*\)|\bXCTAssertTrue\(\s*true\s*\)")
CI_MASK_RE = re.compile(r"continue-on-error\s*:\s*true|allow_failure\s*:\s*true|\|\|\s*(?:true|:|exit\s+0)\b|"
                        r"\bif\s*:\s*(?:false|\$\{\{\s*false\s*\}\})|\bwhen\s*:\s*never\b|--passWithNoTests\b|"
                        r"\bset\s+\+e\b|^\s*exit\s+0\s*$", re.M)
CI_STEP_RE = re.compile(r"^\s*-\s+(?:name|run|uses|id|script|stage)\s*:|^\s*stage\s*\(", re.M)
COV_KEY_RE = re.compile(r"\b(fail[_-]under|cov-fail-under|min(?:imum)?[_-]?coverage|coverage[_-]?threshold|branches|functions|"
                        r"lines|statements|target|threshold)\b[\"']?\s*[=:]?\s*[\"']?(\d+(?:\.\d+)?)", re.I)
COV_CONTEXT_RE = re.compile(r"coverage|fail[_-]under|cov-fail-under|threshold|codecov|\bnyc\b", re.I)
HOOKED_SUBS = {"commit": ("pre-commit", "commit-msg", "prepare-commit-msg"), "push": ("pre-push",),
               "merge": ("pre-merge-commit", "commit-msg"), "rebase": ("pre-rebase",),
               "am": ("pre-applypatch", "applypatch-msg")}
HOOK_SKIP_VARS = {"HUSKY": "off", "HUSKY_SKIP_HOOKS": "on", "SKIP": "on", "LEFTHOOK": "off", "LEFTHOOK_EXCLUDE": "on",
                  "SKIP_SIMPLE_GIT_HOOKS": "on", "OVERCOMMIT_DISABLE": "on"}
COMMIT_ARG_OPTS = {"-m", "-F", "-C", "-c", "-t", "--message", "--file", "--author", "--date", "--template", "--fixup",
                   "--squash", "--trailer", "--reuse-message", "--reedit-message", "--cleanup"}
TEST_RUN_RE = re.compile(
    r"\b(?:pytest|py\.test|unittest|nose2|tox|nox|jest|vitest|mocha|ava|jasmine|karma|rspec|minitest|phpunit|pest|ctest|"
    r"playwright\s+test|cypress\s+run|go\s+test|cargo\s+(?:test|nextest)|mvnw?\b[^|;&]*\b(?:test|verify)|"
    r"gradlew?\b[^|;&]*\b(?:test|check)|dotnet\s+test|swift\s+test|mix\s+test|deno\s+test|bun\s+test|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:-\w+\s+)*(?:run\s+)?(?:test|t)\b|(?:npm|pnpm|yarn)\s+run\s+[\w:.-]*test[\w:.-]*|"
    r"make\s+[\w.-]*(?:test|check)[\w.-]*|(?:just|task)\s+[\w.-]*test[\w.-]*|"
    r"(?:bazel|buck2?|sbt|lein|stack|cabal|flutter|dart|turbo(?:\s+run)?|nx(?:\s+run)?|rake|rails)\s+(?:test|spec)\b|"
    r"\./[\w/.-]*test[\w.-]*(?:\.sh)?\b|python3?\s+(?:-\w+\s+)*-m\s+(?:pytest|unittest)|python3?\s+[\w/.-]*test[\w.-]*\.py)")
CLAIM_RE = re.compile(
    r"\b(?:all\s+)?(?:unit\s+|integration\s+|e2e\s+)?(?:tests?|specs?|checks?)\s+(?:are\s+|now\s+|all\s+)*"
    r"(?:pass(?:es|ed|ing)?|green|succeed\w*)\b|\b(?:passing|green)\s+(?:tests?|specs?|checks?|ci|build)\b|"
    r"\bci\s+(?:is\s+)?(?:green|pass\w*)\b|\b\d+\s*/\s*\d+\s+(?:tests?\s+)?pass\w*", re.I)
HOOK_HINT = "Run it with its hooks; if a hook fails, fix what it reports, or ask the owner for a one-time permit."
TEST_HINT = ("Make the code pass the tests as they are written; if a test is wrong, say why and ask the owner "
             "for a one-time permit.")
CI_HINT = "Leave the CI and hook checks as they are and fix what they report, or ask the owner."
COV_HINT = "Add tests to reach the floor, or ask the owner to change it."
CLAIM_HINT = ("Run the tests in this session first (the run is recorded), then commit; or take the claim out of "
              "the message.")


def test_integrity_on():
    """OPERATOR_TEST_INTEGRITY=0 in the hook's own environment turns OP-026 off."""
    return os.environ.get("OPERATOR_TEST_INTEGRITY", "1").strip() != "0"


def _code_only(text):
    """The lines that are code: a commented-out assertion or step no longer counts."""
    keep = []
    for line in (text or "").split("\n"):
        t = line.strip()
        if not t or (t.startswith(("#", "//", "/*", "*", "--", "<!--", "%")) and not t.startswith(("#[", "#!"))):
            continue
        keep.append(line)
    return "\n".join(keep)


def file_kinds(path, cwd):
    """Which checks a file carries: "test" (test code), "ci" (a CI job or hook config), "cov" (a coverage floor).
    Paths are read from the repository root, so a project folder called test/ does not make every file a test."""
    root = git_top(path) or (cwd if cwd and is_inside(path, cwd) else None)
    rel = posixpath.relpath(path, root) if root else posixpath.basename(path)
    kinds = set()
    if TEST_PATH_RE.search(rel) and rel.lower().endswith(TEST_CODE_EXT):
        kinds.add("test")
    if CI_PATH_RE.search(rel):
        kinds.add("ci")
    if COV_PATH_RE.search(rel):
        kinds.add("cov")
    return kinds


def _file_text(path, limit=1048576):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > limit:
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _cov_floors(text):
    """(key, value) for each coverage floor in a config, in order."""
    if not COV_CONTEXT_RE.search(text or ""):
        return []
    out = []
    for m in COV_KEY_RE.finditer(text):
        try:
            out.append((m.group(1).lower().replace("-", "_"), float(m.group(2))))
        except ValueError:
            continue
    return out


def _assert_lines(code):
    """The lines of test code that assert something real, whitespace folded: `assert True` checks nothing."""
    return collections.Counter(re.sub(r"\s+", " ", line.strip()) for line in code.split("\n")
                               if TEST_ASSERT_RE.search(line) and not TRIVIAL_ASSERT_RE.search(line))


def integrity_findings(path, kinds, before, after):
    """What a change to an existing test, CI or coverage file takes away. Never runs anything."""
    hits, base = [], posixpath.basename(path)
    if "test" in kinds:
        b, a = _code_only(before), _code_only(after)
        tb, ta = len(TEST_DECL_RE.findall(b)), len(TEST_DECL_RE.findall(a))
        sb, sa = len(TEST_SKIP_RE.findall(b)), len(TEST_SKIP_RE.findall(a))
        lb, la = _assert_lines(b), _assert_lines(a)
        gone, came = sum((lb - la).values()), sum((la - lb).values())
        if ta < tb:
            hits.append(("OP-026", "removes %d test%s from %s" % (tb - ta, "" if tb - ta == 1 else "s", path), TEST_HINT))
        if sa > sb:
            new = [m for m in TEST_SKIP_RE.findall(a) if m not in TEST_SKIP_RE.findall(b)] or TEST_SKIP_RE.findall(a)
            hits.append(("OP-026", "adds a skip or focus marker (%s) to %s" % (new[0].strip()[:40], path), TEST_HINT))
        # taking assertions out: more go than come back, and fewer than half come back. A rewrite that swaps
        # three checks for two stricter ones (claude session c70794df, test_gate.py) is not a weakening.
        if gone > came and gone >= 2 * came:
            k = gone - came
            hits.append(("OP-026", "takes %d assertion%s out of %s" % (k, "" if k == 1 else "s", path), TEST_HINT))
    if "ci" in kinds:
        b, a = _code_only(before), _code_only(after)
        mb, ma = CI_MASK_RE.findall(b), CI_MASK_RE.findall(a)
        if len(ma) > len(mb):
            new = [m for m in ma if m not in mb] or ma
            hits.append(("OP-026", "makes a check in %s unable to fail (%s)" % (base, new[0].strip()[:40]), CI_HINT))
        husky = "/.husky/" in path                           # a husky hook is a shell script: each line is a step
        sb = len(b.split("\n")) if husky and b else len(CI_STEP_RE.findall(b))
        sa = len(a.split("\n")) if husky and a else len(CI_STEP_RE.findall(a))
        if sa < sb:
            hits.append(("OP-026", "takes %d step%s or hook%s out of %s" % (sb - sa, "" if sb - sa == 1 else "s",
                                                                          "" if sb - sa == 1 else "s", base), CI_HINT))
    if "cov" in kinds:
        fb, fa = _cov_floors(before), _cov_floors(after)
        if fb:
            seen = {}
            for k, v in fa:
                seen.setdefault(k, []).append(v)
            for k, v in fb:
                vals = seen.get(k) or []
                if not vals:
                    hits.append(("OP-026", "removes the coverage floor %s (%g) from %s" % (k, v, base), COV_HINT))
                    break
                w = vals.pop(0)
                if w < v:
                    hits.append(("OP-026", "lowers the coverage floor %s from %g to %g in %s" % (k, v, w, base), COV_HINT))
                    break
    return hits


def _edited_text(tool, ti, before):
    """The whole file after a Write, Edit or MultiEdit call, or None when the call cannot be applied."""
    if isinstance(ti.get("content"), str) and tool in ("write", "create", "create_file"):
        return ti["content"]
    edits = ti.get("edits") if isinstance(ti.get("edits"), list) else [ti]
    after = before
    for e in edits:
        if not isinstance(e, dict):
            return None
        old, new = e.get("old_string"), e.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str) or not old or old not in after:
            return None
        after = after.replace(old, new) if e.get("replace_all") else after.replace(old, new, 1)
    return after


def test_integrity_write(tool, ti, path, cwd):
    """OP-026 for a file tool: an existing test, CI or coverage file changed so it checks less."""
    if not test_integrity_on():
        return []
    kinds = file_kinds(path, cwd)
    before = _file_text(path) if kinds else None
    if before is None:
        return []
    after = _edited_text(tool, ti, before)
    return integrity_findings(path, kinds, before, after) if after is not None else []


def active_hooks(root):
    """(hook names git would run, hooks folder) for a repository: core.hooksPath or the repository's own folder,
    executable files only, the .sample files git ships left out."""
    hp = git_out(root, ["config", "--get", "core.hooksPath"])
    if hp is None:                                            # git not asked (the per-command budget is spent): read the file
        m = re.search(r"(?im)^\s*hookspath\s*=\s*(.+?)\s*$", _file_text(pj(root, ".git", "config")) or "")
        hp = m.group(1).strip("\"'") if m else ""
    hp = hp.strip()
    if hp:
        d = norm(hp, root)
    else:
        gp = (git_out(root, ["rev-parse", "--git-path", "hooks"]) or "").strip()
        d = norm(gp, root) if gp else pj(root, ".git", "hooks")
    names = set()
    try:
        for n in os.listdir(d):
            p = pj(d, n)
            if not n.endswith(".sample") and os.path.isfile(p) and os.access(p, os.X_OK):
                names.add(n)
    except OSError:
        pass
    return names, d


def _git_line(toks):
    """(subcommand, its arguments, -c settings, -C folder) for a git command line, or None."""
    i, conf, cdir = 1, [], None
    while i < len(toks):
        t = toks[i]
        if t == "-c" and i + 1 < len(toks):
            conf.append(toks[i + 1])
            i += 2
            continue
        if t.startswith("-c") and len(t) > 2 and "=" in t:
            conf.append(t[2:])
            i += 1
            continue
        if t == "-C" and i + 1 < len(toks):
            cdir = toks[i + 1]
            i += 2
            continue
        if t in ("--git-dir", "--work-tree", "--namespace", "--exec-path") and i + 1 < len(toks):
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        return t, toks[i + 1:], conf, cdir
    return None


def _skips_verify(sub, args):
    """True when the git command is told not to run its hooks."""
    i = 0
    while i < len(args):
        t = args[i]
        if t == "--":
            break
        if t == "--no-verify":
            return True
        if sub == "commit":
            if t in COMMIT_ARG_OPTS:
                i += 2
                continue
            if t.startswith("--"):
                i += 1
                continue
            if t.startswith("-") and len(t) > 1:
                for ch in t[1:]:
                    if ch == "n":
                        return True
                    if "-" + ch in COMMIT_ARG_OPTS:          # the rest of the word is that option's value
                        break
        i += 1
    return False


def _skip_var_set(name, value):
    how = HOOK_SKIP_VARS.get(name)
    v = (value or "").strip().strip("'\"").lower()
    if how == "off":
        return v in ("0", "false", "no", "off")
    return bool(how) and v not in ("", "0", "false", "no", "off")


def test_integrity_shell(cmd, cwd, effects):
    """OP-026 for a shell command: hooks told not to run or taken away, a stash ahead of a commit, test and CI
    files deleted, emptied or rewritten so they check less. Reads git and the files; runs nothing."""
    if not test_integrity_on():
        return []
    hits = []
    text, _ = extract_heredocs(cmd)
    here, skip_env, stashed = cwd, {}, False
    for seg in split_segments(text):
        toks = tokenize(seg)
        lead = []
        for t in toks:                                        # assignments ahead of the program, and `export X=`
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", t)
            if m:
                lead.append((m.group(1), m.group(2)))
            elif os.path.basename(t) not in WRAPPERS and t != "export":
                break
        for name, value in lead:
            if name in HOOK_SKIP_VARS and _skip_var_set(name, value):
                skip_env[name] = value
        body, _ = strip_wrappers(strip_leading_redirects(toks))
        if not body:
            continue
        prog = os.path.basename(body[0])
        if prog in ("cd", "pushd") and len(body) > 1 and "$" not in body[1]:
            here = resolve(body[1], here)
            continue
        if prog == "chmod" and len(body) > 2:
            mode = body[1]
            if re.match(r"^[ugoa]*-[rwxXst]*x", mode) or (re.fullmatch(r"0?[0-7]{3}", mode) and
                                                          all(int(c) % 2 == 0 for c in mode[-3:])):
                for a in body[2:]:
                    p = resolve(a, here)
                    root = git_top(p)
                    if root and is_inside(p, active_hooks(root)[1]) and os.path.basename(p) in active_hooks(root)[0]:
                        hits.append(("OP-026", "makes the git hook %s unable to run" % os.path.basename(p), HOOK_HINT))
            continue
        if prog == "sed" and any(t == "-i" or t.startswith("-i") or t == "--in-place" for t in body[1:]):
            words, script = [t for t in body[1:] if t and not t.startswith("-")], None   # '' after -i is a backup suffix
            for k, t in enumerate(body[1:-1], 1):
                if t in ("-e", "--expression"):
                    script = body[k + 1]
            if script is None:
                script = words.pop(0) if words else ""
            else:
                words = [w for w in words if w != script]
            if re.search(r"(?:^|[;/}\s])\$?d(?:$|[;\s}])|/d\b", script):
                for a in words:
                    p = resolve(a, here)
                    if "test" in file_kinds(p, here) and os.path.isfile(p):
                        hits.append(("OP-026", "deletes lines from the test file %s with sed" % p, TEST_HINT))
            continue
        if prog != "git":
            continue
        line = _git_line(body)
        if not line:
            continue
        sub, args, conf, cdir = line
        where = resolve(cdir, here) if cdir else here
        root = git_top(where) if where and not str(where).startswith(UNKNOWN_CWD) else None
        if not root:
            continue
        names, hdir = active_hooks(root)
        if sub == "stash" and (not args or args[0] in ("push", "save") or args[0].startswith("-")):
            stashed = True
            continue
        if sub == "stash" and args and args[0] in ("pop", "apply"):
            stashed = False
            continue
        if sub == "config":
            low = [a.lower() for a in args]
            if "core.hookspath" in low and not any(a in ("--get", "--get-all", "--list", "-l", "--show-origin") for a in low):
                if "--unset" in low or "--unset-all" in low:
                    if names:
                        hits.append(("OP-026", "turns off the repository's git hooks (core.hooksPath unset)", HOOK_HINT))
                else:
                    k = low.index("core.hookspath")
                    val = args[k + 1] if k + 1 < len(args) else ""
                    if val and norm(val, root) != hdir:
                        hits.append(("OP-026", "points git at another hooks folder (core.hooksPath %s)" % val, HOOK_HINT))
            continue
        if sub == "rm":
            for a in [x for x in args if not x.startswith("-")]:
                p = resolve(a, where)
                kinds = file_kinds(p, where)
                if os.path.isdir(p) and re.search(r"(?:^|/)(?:tests?|__tests__|specs?)$", p.rstrip("/")):
                    hits.append(("OP-026", "removes the test folder %s from the repository" % p, TEST_HINT))
                elif ("test" in kinds and TEST_DECL_RE.search(_code_only(_file_text(p) or ""))) or "ci" in kinds:
                    hits.append(("OP-026", "removes %s, which the checks depend on" % p, TEST_HINT if "test" in kinds else CI_HINT))
            continue
        relevant = names & set(HOOKED_SUBS.get(sub, ()))
        if not relevant:
            continue
        if _skips_verify(sub, args):
            hits.append(("OP-026", "git %s --no-verify skips the %s hook%s" % (sub, ", ".join(sorted(relevant)),
                                                                            "" if len(relevant) == 1 else "s"), HOOK_HINT))
        for c in conf:
            if c.lower().startswith("core.hookspath=") and norm(c.split("=", 1)[1] or "/dev/null", root) != hdir:
                hits.append(("OP-026", "git -c %s runs %s without the repository's hooks" % (c, sub), HOOK_HINT))
        if skip_env:
            k = sorted(skip_env)[0]
            hits.append(("OP-026", "%s=%s turns the hooks off for git %s" % (k, skip_env[k], sub), HOOK_HINT))
        if sub == "commit" and stashed:
            hits.append(("OP-026", "stashes work, then commits: the hooks judge a different tree than the one worked on", HOOK_HINT))
    for e in effects or []:
        kind = e.get("kind")
        if e.get("unresolved") or "__SUBST__" in str(e.get("prog")) or "$" in str(e.get("prog")):
            continue                                          # `$(PYTHON) -m unittest discover -s tests` names, not deletes
        for t in (e.get("targets") or []) if kind in ("delete", "move") else []:
            if not isinstance(t, str) or t.startswith(UNKNOWN_CWD):
                continue
            root = git_top(t)
            if root:
                names, hdir = active_hooks(root)
                if (is_inside(t, hdir) and (os.path.basename(t) in names or t == hdir)) or \
                        re.search(r"(?:^|/)\.husky(?:/|$)", t):
                    hits.append(("OP-026", "%s the git hook %s" % ("deletes" if kind == "delete" else "moves", t), HOOK_HINT))
                    continue
            kinds = file_kinds(t, None)
            if "ci" in kinds and os.path.isfile(t):
                hits.append(("OP-026", "%s the CI file %s" % ("deletes" if kind == "delete" else "moves", t), CI_HINT))
            elif os.path.isdir(t) and re.search(r"(?:^|/)(?:tests?|__tests__|specs?)$", t.rstrip("/")):
                hits.append(("OP-026", "deletes the test folder %s" % t, TEST_HINT))
            elif "test" in kinds and TEST_DECL_RE.search(_code_only(_file_text(t) or "")):
                hits.append(("OP-026", "%s the test file %s" % ("deletes" if kind == "delete" else "moves", t), TEST_HINT))
        seg = e.get("seg") or ""
        if e.get("prog") in ("cat", "tee", "echo", "printf") and ">>" not in seg and not re.search(r"\btee\s+(?:-\w*a|--append)", seg):
            for w in e.get("writes") or []:
                after = WRITTEN.get(w)
                kinds = file_kinds(w, None) if isinstance(after, str) else set()
                before = _file_text(w) if kinds else None
                if before is not None:
                    hits += integrity_findings(w, kinds, before, after)
    return list(dict.fromkeys(hits))


def _message_values(args, short, long_):
    """The values a command line hands its message options: `-m x`, `-mx`, `-am x`, `--message=x`, `--message x`."""
    out = []
    for i, t in enumerate(args):
        nxt = args[i + 1] if i + 1 < len(args) else None
        for o in long_:
            if t == o and nxt is not None:
                out.append((o, nxt))
            elif t.startswith(o + "="):
                out.append((o, t[len(o) + 1:]))
        if t.startswith("-") and not t.startswith("--") and len(t) > 1:
            for k, ch in enumerate(t[1:], 1):
                if ch in short:
                    v = t[k + 1:] or nxt
                    if v is not None:
                        out.append(("-" + ch, v))
                    break
    return out


def commit_claim(cmd, cwd):
    """(the words that say the tests pass, the repository, whether the same command ran tests first) for a git
    commit or a gh pull request whose own message says so, or None. Only the message is read, never the rest
    of the command: a log check that prints "ALL CHECKS PASSED" is not a claim."""
    text, bodies = extract_heredocs(cmd)
    here, hd, ran = cwd, 0, False
    for seg in split_segments(text):
        nh = len(HEREDOC_RE.findall(seg))
        mine, hd = bodies[hd:hd + nh], hd + nh
        body, _ = strip_wrappers(strip_leading_redirects(tokenize(seg)))
        if not body:
            continue
        prog = os.path.basename(body[0])
        if prog in ("cd", "pushd") and len(body) > 1 and "$" not in body[1]:
            here = resolve(body[1], here)
            continue
        if prog == "git":
            line = _git_line(body)
            if not line or line[0] != "commit":
                ran = ran or bool(TEST_RUN_RE.search(seg))
                continue
            where = resolve(line[3], here) if line[3] else here
            vals = _message_values(line[1], "mF", ("--message", "--file"))
        elif prog == "gh" and len(body) > 2 and body[1] == "pr" and body[2] in ("create", "edit"):
            where = here
            vals = _message_values(body[3:], "tbF", ("--title", "--body", "--body-file"))
        else:
            ran = ran or bool(TEST_RUN_RE.search(seg))
            continue
        msg = list(mine)
        for o, v in vals:
            if o in ("-F", "--file", "--body-file"):
                v = (_file_text(resolve(v, where), 65536) or "") if v != "-" else ""
            msg.append(v)
        said = CLAIM_RE.search("\n".join(msg))
        root = git_top(where) if said and where and not str(where).startswith(UNKNOWN_CWD) else None
        if said and root:
            return said.group(0), root, ran
    return None


def runs_tests(tool, payload):
    """True when a shell call starts a test runner, in its own words or in a script it runs (the files the gate
    read for this command): `~/scratch/verify.sh` that calls `cargo test` is a test run."""
    if tool.lower() not in SHELL_TOOLS | POWERSHELL_TOOLS:
        return False
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    cmd = str(ti.get("command") or ti.get("cmd") or "") if isinstance(ti, dict) else ""
    if TEST_RUN_RE.search(cmd):
        return True
    for f in list(SEEN_FILES)[:MAX_FILES]:
        body = _file_text(f, MAX_SCRIPT_BYTES)
        if body and TEST_RUN_RE.search(body):
            return True
    return False


def claim_unbacked(rows, root):
    """The last change inside `root` that no later test run covers, from receipt rows oldest first; None when
    the rows show no change there or a test run after the last one."""
    last = None
    for i, r in enumerate(rows):
        if str(r.get("tool") or "").lower() in WRITE_TOOLS and r.get("decision") in ("PASS", "ALLOW"):
            path = str(r.get("action_preview") or "").split(" ", 1)[-1].split(" | ", 1)[0]
            if path and is_inside(path, root):
                last = i
    if last is None:
        return None
    shells = SHELL_TOOLS | POWERSHELL_TOOLS
    for r in rows[last + 1:]:
        if r.get("decision") in ("PASS", "ALLOW") and str(r.get("tool") or "").lower() in shells and \
                (r.get("tests") or TEST_RUN_RE.search(str(r.get("action_preview") or ""))):
            return None
    return str(rows[last].get("action_preview") or "")


def test_claim_hit(payload, tool, cwd):
    """OP-026 for a claim: a commit or pull request whose message says the tests pass, from a session that
    changed files in that repository and has no test run on record since. Reads this session's receipts."""
    if not test_integrity_on() or tool.lower() not in SHELL_TOOLS | POWERSHELL_TOOLS:
        return []
    ti = payload.get("tool_input") or payload.get("toolInput") or {}
    cmd = str(ti.get("command") or ti.get("cmd") or "") if isinstance(ti, dict) else ""
    claim = commit_claim(cmd, cwd) if re.search(r"\bcommit\b|\bgh\s+pr\b", cmd) else None
    sid = session_of(payload)
    if not claim or claim[2] or not sid:                      # claim[2]: the same command runs the tests first
        return []
    last = claim_unbacked(session_rows(sid), claim[1])
    if last is None:
        return []
    return [("OP-026", "the message says \"%s\", but this session has no test run on record since its last change "
             "in %s (%s)" % (claim[0][:40], posixpath.basename(claim[1]), last[:100]), CLAIM_HINT)]


# ---- Operator Lite inside Operator -------------------------------------------------------------- #
_LITE_CORE = {}


def lite_core():
    """The Operator Lite gate that ships beside this file, loaded once. None when it is missing, or when
    its bytes differ from the pin: a changed copy is never run, and `status` says why."""
    if "mod" in _LITE_CORE:
        return _LITE_CORE["mod"]
    _LITE_CORE.update(mod=None, cfg=None, status="missing")
    path = pj(os.path.dirname(os.path.abspath(__file__)), LITE_CORE_FILE)
    try:
        with open(path, "rb") as f:
            body = f.read()
        # a Windows checkout with core.autocrlf turns \n into \r\n; that is the same file, any other change is not
        if hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest() != LITE_CORE_SHA256:
            _LITE_CORE["status"] = "refused: %s differs from the pinned Operator Lite %s" % (LITE_CORE_FILE, LITE_CORE_VERSION)
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location("operator_lite_core", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LITE_CORE.update(mod=mod, cfg=mod.load_config(), status="loaded")
        return mod
    except Exception as exc:
        _LITE_CORE["status"] = "unavailable (%s)" % type(exc).__name__
        return None


SAFE_DEVICE_RE = re.compile(r"/dev/(null|loop\d+\w*|mapper/loop\w*)\b")
LIST_ONLY_RE = re.compile(r"\b(s?fdisk|gdisk|parted)\b[^|;&]*\s(-l|--list)\b")


def lite_hit_kept(hit, tool, cmd):
    """Where Operator judges a family more precisely than Lite, a Lite stop that Operator's own reading
    shows is safe is not added: /dev/null and loop devices are files, `fdisk -l` only lists, and a
    PowerShell call has no /dev. Everything else Lite stops, Operator stops."""
    rid, detail = hit[0], str(hit[1] or "")
    drop = lambda why, target=None: lite_dropped(rid, detail, target, why)
    if rid == "OP-011" and detail.startswith("git "):
        return drop("git work loss is Operator's own, finer reading (stash, commit, dry run)")
    if rid == "OP-004" and detail.startswith("delete of the git repository"):
        return drop("a repository delete is measured by Operator: a fresh clone loses nothing, unpushed work is OP-011")
    if rid in ("OP-003", "OP-011") and re.match(r"\s*(sudo\s+)?rsync\b", cmd or ""):
        return drop("rsync --delete removes extras in the destination; Operator reads its direction")
    if rid == "OP-000" and detail.startswith("agent invoked the gate's owner commands"):
        live = command_text(cmd or "")
        if not OWNER_CMD_RE.search(live) and "operator-lite" not in live:
            return drop("the owner command is only named in a heredoc or a string: data, not a command")
    if rid == "OP-000" and detail.startswith("agent invoked `operator-gate "):
        live = command_text(cmd or "")
        if OWNER_READ_RE.search(live) and not OWNER_CMD_RE.search(live):
            return drop("`mode` or `approve` alone, or `--help`: it only prints")
    if rid == "OP-003" and detail.startswith("recursive delete of "):
        target = detail[len("recursive delete of "):].split(" -- ", 1)[0].strip()
        if target.startswith("/") and is_scratch(target):
            return drop("the real path is inside the owner's scratch_ok, or a cache every tool rebuilds", target)
    if rid == "OP-000" and detail.startswith("inline code has a write whose target the gate cannot resolve"):
        return drop("Operator ties a gate path to the write's own operand (inline_file_calls)")
    if rid == "OP-006" and detail == "stripe live action":
        return drop("Operator stops every live stripe call but its reads (list, retrieve, search, status)")
    if rid == "OP-014" and "/dev/" in detail:
        if SAFE_DEVICE_RE.search(detail) or tool == "powershell":
            return drop("a /dev/null or loop device is a file, and a PowerShell call has no /dev")
        if "rewrites the disk" in detail and LIST_ONLY_RE.search(cmd or ""):
            return drop("the disk tool only lists")
    if " -- inside script " in detail:
        # `bash -n script` only checks the syntax; the script is judged when something can run it
        name = posixpath.basename(detail.rsplit(" -- inside script ", 1)[1].strip())
        if name and not script_may_run(cmd, name):
            return drop("the script is only syntax-checked, never run, by this call", name)
    return True


LITE_DROPS = []                                       # Lite stops this call dropped, each written as its own receipt


def lite_dropped(rule, detail, target, why):
    """Record that Operator's finer reading set aside one of Lite's stops; the hook writes a LITE_DROPPED
    receipt for each, naming the Lite rule, the target and why (Sam, 2026-10-20)."""
    LITE_DROPS.append({"lite_rule": rule, "lite_detail": redact(detail)[:300],
                       "target": redact(target or detail)[:300], "why": why})
    return False


QUIET_PROGS = READ_PROGS | {"chmod", "ls", "stat", "file", "wc", "shasum", "sha256sum", "md5", "diff", "cmp", "cp", "git"}


def script_may_run(cmd, name):
    """False only when the command checks the script's syntax with `sh -n` and every other mention of it
    is a read: nothing in the command can run it. Any mention this cannot place counts as a run."""
    text, bodies = extract_heredocs(cmd or "")
    if any(name in b for b in bodies):
        return True
    checked = False
    for seg in split_segments(text):
        if name not in seg:
            continue
        try:
            _, _, _, bare, _ = root_form(tokenize(seg))
        except Exception:
            return True
        at = [i for i, t in enumerate(bare) if posixpath.basename(t) == name]
        if not at:
            return True                                       # named inside a string or a substitution
        prog = os.path.basename(bare[0])
        if prog in ALL_SHELLS:
            lead = []
            for t in bare[1:]:
                if not t.startswith("-") or t == "--":
                    break
                lead.append(t)
            if any(not t.startswith("--") and "n" in t[1:] for t in lead):
                checked = True
                continue
            return True
        if prog in QUIET_PROGS and 0 not in at:
            continue
        return True
    return not checked


def lite_judgment(payload):
    """Operator Lite's own hits, transforms, and the paths its effects delete and write."""
    mod = lite_core()
    if not mod:
        return [], [], [], []
    _, _, hits, _, lmeta, leffects = mod.classify(payload, _LITE_CORE["cfg"])
    absolute = lambda xs: [t for t in xs or [] if isinstance(t, str) and t.startswith("/")]
    deletes, writes = [], []
    for e in leffects or []:
        if e.get("kind") == "delete":
            deletes += absolute(e.get("targets"))
        writes += absolute(e.get("writes"))
    return [h for h in hits if h and h[0] in RULES], list((lmeta or {}).get("transforms") or []), deletes, writes


# ---- xargs fed a literal: the command it runs is the command that runs -------------------------- #
_XARGS_OPTS = r"((?:\s+-(?!I\b)[A-Za-z-]+(?:[= ](?!-)\S+)?)*)"


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
    # one ledger across every agent: the same mark, with the session and host it came from
    guarded(machine_note, None, dict(mark, sid=sid, source=str((meta or {}).get("source") or "")))


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
    if not found:                                            # one ledger: stops other sessions met here
        others = guarded(machine_stops, [], sid)
        if others:
            found += guarded(second_spelling, [], payload, cwd, effects, meta, others, True)
    keys = [m for m in seen if m.get("kind") == "found_key" and m.get("prints")]
    if keys:
        found += guarded(found_key_use, [], payload, cwd, keys)
    return found


# ---- the brake: a session stopped again and again is a runaway, and its changes are held (OP-024) ---- #
BRAKE_AT = 8                             # stops of one session inside the window that trip the brake
BRAKE_WINDOW_S = 300
BRAKE_QUIET_S = 15 * 60                  # the brake lets go after this long without a stop
BRAKE_TRIPPED = {"sid": ""}              # set by write_receipt when a stop trips the brake
GIT_READ_SUBS = {"status", "log", "diff", "show", "blame", "rev-parse", "ls-files", "grep", "describe",
                 "shortlog", "reflog", "cat-file", "ls-tree", "remote", "branch", "tag", "config"}


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


def changes_anything(tool, effects):
    """False only for a call that reads: no write, delete, move, inline code, outbound send or changing git."""
    t = str(tool or "").lower()
    if t in WRITE_TOOLS or t.startswith("mcp__"):
        return True
    for e in effects or []:
        if e.get("writes") or e.get("targets") or e.get("kind") in ("delete", "move", "inline") or e.get("unresolved"):
            return True
        if e.get("kind") == "git" and str(e.get("sub") or "") not in GIT_READ_SUBS:
            return True
        prog = posixpath.basename(str(e.get("prog") or ""))
        if prog in HTTP_PROGS and _sends_out(prog, [str(a) for a in e.get("args") or []] + [str(f) for f in e.get("flags") or []], str(e.get("seg") or "")):
            return True
    return False


def brake_hit(payload, tool, effects):
    """OP-024: this session's brake is on, and the call would change something."""
    sid = session_of(payload)
    e = brake_engaged(sid)
    if not e or not changes_anything(tool, effects):
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
            "protected": ["%s (%s)" % (p, c) for p, c in guarded(file_contracts, [], cfg, cwd)][:TURNOVER_LIST],
            "other_sessions": len(guarded(machine_stops, [], sid))}


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


def cmd_turnover(argv):
    """`turnover --pre-compact` (the PreCompact hook) seals what this session hands over; `turnover --ack <id>`
    takes the watch and prints it; `turnover --show <id>` prints one; `turnover` lists them."""
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
        # the floor reads a delete with its command substitutions put back: the analyzer judges each
        # $(...) on its own and leaves a placeholder where its output would land
        subs = iter(SUBST_RE.findall(cmd))
        for eff in effects_out:
            if eff.get("kind") == "delete" and "__SUBST__" in (eff.get("seg") or ""):
                seg = re.sub("__SUBST__", lambda m: next(subs, "__SUBST__"), eff["seg"])
                hits += guarded(floor_hits, [], dict(eff, seg=seg, unresolved=True), eff.get("cwd") or cwd)
        if tool not in POWERSHELL_TOOLS:
            hits += guarded(root_reach, [], cmd, ti.get("run_in_background") is True, cwd)
            hits += guarded(machine_wreck, [], cmd, cwd)
            hits += guarded(repo_code_run, [], cmd, cwd)
            hits += guarded(loss_reach, [], cmd, cwd)
            hits += guarded(remote_delete, [], cmd, cwd)
        hits += guarded(test_integrity_shell, [], cmd, cwd, effects_out)
        hits += guarded(windows_wreck, [], cmd)               # read on every shell tool, PowerShell included
        for eff in effects_out:                               # a redirect, cp, tee or sed -i onto an arming file
            for w in eff.get("writes") or []:
                hits += guarded(agent_arming, [], w, None)
        if OWNER_CMD_RE.search(command_text(cmd)):           # not a mention inside a heredoc or a string (receipt 9b55a6634ac46d56)
            hits.append(("OP-000", "agent invoked the gate's owner commands",
                         "Approvals, installs and onboarding are run by the owner in their own terminal."))
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
        if not body and isinstance(ti.get("edits"), list):   # MultiEdit carries a list of replacements
            body = "\n".join(str(e.get("new_string") or "") for e in ti["edits"] if isinstance(e, dict))
        for p in write_paths(tool, ti, cwd):
            r38 = check_workflow_poison(p, body, cwd)
            if r38:
                hits.append(("OP-038", r38, "Workflow poison (pull_request_target checkout, unverified runner env) is owner-only."))
            r44 = check_persistence_file(p, cwd)
            if r44:
                hits.append(("OP-044", r44, "Installing persistent launchd daemons, scheduled cron jobs, or services is owner-only."))
            hits += guarded(written_code, [], p, body)
            hits += guarded(agent_arming, [], p, body)
            hits += guarded(test_integrity_write, [], tool, ti, p, cwd)
        if re.search(r"(sk_live|rk_live|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})", body):
            hits.append(("OP-S04", "credential-shaped string in write", ""))
    elif tool.startswith("mcp__") or tool in ("calldynamictool", "callmcptool"):
        previews.append(tool_raw)
        mcp_hit = check_mcp_control_plane(tool_raw, ti)
        if mcp_hit:
            hits.append(mcp_hit)
        if re.search(r"__(send_imessage|send_message|forward|reply)$", tool):
            hits.append(("OP-S02", tool_raw, ""))
        blob_l = blob
        if MCP_SECRET_ARG_RE.search(blob_l) or re.search(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY", blob_l):
            hits.append(("OP-036", "%s carries a credential-shaped argument" % tool_raw,
                         "Secrets never belong in tool arguments; pass a file path the owner approved or nothing."))
        for m in re.finditer(r'(?:^|[,"{\s])((?:~|\$HOME|/Users/|/home/)[^"\',}\s]*)', blob_l):
            if MCP_SENSITIVE_PATH_RE.search(m.group(1)):
                hits.append(("OP-036", "%s reads sensitive path %s" % (tool_raw, m.group(1)),
                             "Ask the owner before tool calls that touch credentials or identity files."))
                break
    elif tool in ("read", "read_file", "view_file", "grep", "search", "grep_search", "file_search") or "file_path" in ti:
        target_path = ti.get("path") or ti.get("file_path") or ti.get("target_file") or ti.get("filepath")
        if target_path and isinstance(target_path, str):
            p = resolve(target_path, cwd)
            if CRED_STORE_FILE_RE.search(p) or CRED_STORE_FILE_RE.search(target_path):
                hits.append(("OP-031", "reads home credential store (%s)" % os.path.basename(p),
                             "The owner manages credential stores directly; do not read host credential files."))
            if PRIVATE_KEY_NAMES.search(p) or "/.gnupg/private-keys" in p or "/.config/age" in p:
                hits.append(("OP-051", "accesses private key material (%s)" % os.path.basename(p),
                             "Accessing, reading, or copying private key files is owner-only."))
    if not payload.get("_inner"):
        hits += guarded(gap_rule_hits, [], payload, cwd, tool, ti, effects_out, cfg)
        # xargs fed a literal runs a command the analyzer can judge directly
        shell_call = tool in SHELL_TOOLS or ("command" in ti and tool not in WRITE_TOOLS and not tool.startswith("mcp__"))
        if shell_call:
            for sub in guarded(xargs_substituted, [], str(ti.get("command") or ti.get("cmd") or "")):
                _, _, sub_hits, _, _, sub_fx = classify(dict(payload, tool_input={"command": sub}, _inner=True), cfg)
                if sub_hits:
                    hits += [(h[0], "xargs -I runs `%s`: %s" % (sub[:80], h[1]), h[2]) for h in sub_hits]
                    effects_out += sub_fx
                    if "xargs-substituted" not in meta["transforms"]:
                        meta["transforms"].append("xargs-substituted")
        # every protection Operator Lite has, Operator has: Lite judges the same call, and what it stops is added
        l_hits, l_transforms, l_deletes, l_writes = guarded(lite_judgment, ([], [], [], []), payload)
        have = {h[0] for h in hits}
        call_text = str(ti.get("command") or ti.get("cmd") or "")
        for h in l_hits:
            if h[0] not in have and lite_hit_kept(h, tool, call_text):
                hits.append(h)
                have.add(h[0])
        for t in l_transforms:
            if t not in meta["transforms"]:
                meta["transforms"].append(t)
        meta["lite_deletes"], meta["lite_writes"] = l_deletes, l_writes
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
                chain_commit(rec["digest"])
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


def consume_permit(action_digest, session):
    p = pj(OP_HOME, "permits", action_digest + ".json")
    try:
        with open(p, encoding="utf-8") as f:
            pm = json.load(f)
        # A different session or an old unbound permit leaves the permit untouched.
        if (not session or pm.get("session") != session or pm.get("action_digest") != action_digest
                or not re.fullmatch(r"[0-9a-f]{64}", str(pm.get("stop_receipt") or ""))):
            return None
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


OPEN_NAMES = {"'": "a single quote", "$'": "a single quote", '"': "a double quote", "`": "a backtick",
              "$(": "a $( substitution", "$((": "a $(( expression", "${": "a ${ expansion", "(": "a parenthesis",
              "@'": "a here-string", '@"': "a here-string"}


def unclosed(cmd, powershell=False):
    """What a shell line still has open when it ends: a quote, a backtick, $( , $(( , ${ or a parenthesis.
    None when the line closes everything it opens. Heredoc bodies and comments are skipped. Strict mode
    only: this is a reader of its own and stops some unusual lines a shell would accept."""
    if powershell:
        q, i, n = None, 0, len(cmd)
        while i < n:
            c = cmd[i]
            if q in ("@'", '@"'):                            # here-string: closed by its quote at a line start
                end = cmd.find("\n" + q[1] + "@", i)
                if end < 0:
                    return OPEN_NAMES[q]
                i, q = end + 3, None
                continue
            if q == "'":
                if c == "'":
                    if cmd.startswith("''", i):
                        i += 1
                    else:
                        q = None
            elif q == '"':
                if c == "`":
                    i += 1
                elif c == '"':
                    if cmd.startswith('""', i):
                        i += 1
                    else:
                        q = None
            elif c == "`":
                i += 1
            elif c == "@" and cmd[i + 1:i + 2] in ("'", '"'):
                q = "@" + cmd[i + 1]
                i += 1
            elif c in "'\"":
                q = c
            elif c == "#" and (i == 0 or cmd[i - 1] in " \t\n;"):
                j = cmd.find("\n", i)
                i = n if j < 0 else j
            i += 1
        return OPEN_NAMES[q] if q else None
    text, _ = extract_heredocs(cmd)
    stack, i, n = [], 0, len(text)
    while i < n:
        c, t = text[i], (stack[-1] if stack else None)
        if t == "'":
            if c == "'":
                stack.pop()
        elif t == "$'":
            if c == "\\":
                i += 1
            elif c == "'":
                stack.pop()
        elif c == "\\":
            i += 1
        elif t == '"':
            if c == '"':
                stack.pop()
            elif c == "`":
                stack.append("`")
            elif text.startswith("$((", i):
                stack.append("$((")
                i += 2
            elif text.startswith("$(", i):
                stack.append("$(")
                i += 1
            elif text.startswith("${", i):
                stack.append("${")
                i += 1
        elif c == "`":
            if t == "`":
                stack.pop()
            else:
                stack.append("`")
        elif c == "'":
            stack.append("$'" if i and text[i - 1] == "$" else "'")
        elif c == '"':
            stack.append('"')
        elif text.startswith("$((", i):
            stack.append("$((")
            i += 2
        elif text.startswith("$(", i):
            stack.append("$(")
            i += 1
        elif text.startswith("${", i):
            stack.append("${")
            i += 1
        elif c == "(":
            stack.append("(")
        elif c == ")":
            if t == "$((" and text.startswith("))", i):
                stack.pop()
                i += 1
            elif t in ("$(", "("):
                stack.pop()                                  # a ) with nothing open is a case pattern
        elif c == "}" and t == "${":
            stack.pop()
        elif c == "#" and t != "${" and (i == 0 or text[i - 1] in " \t\n;"):
            j = text.find("\n", i)
            i = n if j < 0 else j
        i += 1
    return OPEN_NAMES[stack[-1]] if stack else None


def unreadable_payload(raw):
    """Why this hook payload cannot be judged, or None. Strict mode only."""
    if not raw.strip():
        return "the host sent an empty payload"
    try:
        p = json.loads(raw)
    except ValueError:
        return "the payload is not JSON"
    if not isinstance(p, dict):
        return "the payload is not a JSON object"
    tool = str(p.get("tool_name") or p.get("toolName") or "")
    if not tool:
        return "the payload names no tool"
    tl = tool.lower()
    ti = p.get("tool_input") or p.get("toolInput") or p.get("input") or {}
    if not isinstance(ti, dict):
        ti = {"command": ti}
    # the same test classify uses to decide a call is a shell command
    if tl in SHELL_TOOLS or tl in POWERSHELL_TOOLS or ("command" in ti and tl not in WRITE_TOOLS and not tl.startswith("mcp__")):
        cmd = ti.get("command") or ti.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        if not isinstance(cmd, str) or not cmd.strip():
            return "the %s call carries no command" % tool
        left = unclosed(cmd, tl in POWERSHELL_TOOLS)
        if left:
            return "the command leaves %s open, so it cannot be read the way the shell will run it" % left
    return None


def strict_stop(source, raw, why):
    """Strict mode: stop a call the gate could not read, record it, and say how to go on."""
    p = {}
    try:
        p = json.loads(raw)
    except ValueError:
        pass
    if not isinstance(p, dict):
        p = {}
    ti = p.get("tool_input") or p.get("toolInput") or p.get("input") or {}
    cmd = (ti.get("command") or ti.get("cmd")) if isinstance(ti, dict) else None
    rd = write_receipt({"source": source, "decision": "STOP", "layer": "strict", "commitment": "OP-STRICT",
                        "rule": "cannot-read-call", "commitments": ["OP-STRICT"], "detail": redact(why),
                        "session": session_of(p), "call": call_of(p),
                        "tool": str(p.get("tool_name") or p.get("toolName") or ""), "cwd": p.get("cwd"),
                        "action_preview": redact(cmd if isinstance(cmd, str) else raw[:300])})
    stop(["OPERATOR: STOPPED (OP-STRICT cannot-read-call)",
          "Why: strict mode is on, and %s." % why,
          "Compliant path: write the command in a plain form that closes every quote and bracket it opens, "
          "or ask the owner to run it.",
          "No action was taken.",
          "Receipt: %s" % ((rd or "unavailable")[:16]),
          "Operator receipt: %s" % (rd or "unavailable")])


def degraded_hit(raw):
    """Gate crashed: fail closed only for text that plainly looks destructive, else fail open."""
    return bool(re.search(r"\brm\s+-[a-zA-Z]*[rR]|\bmv\s+\S*(vestige|Developer|\.zcode|\.claude)|"
                          r"push\s+.*--force|\bDROP\s+TABLE|\.vestige/vestige\.db|fly\s+deploy|OPERATOR_CANARY_STOP_7f3a",
                          raw or "", re.I))


def adigest_of(payload):
    """Pre-classification action digest used by the taste layer's rewrite counters."""
    ti = payload.get("tool_input") or payload.get("toolInput") or payload.get("input") or {}
    return digest_of({"tool": str(payload.get("tool_name") or payload.get("toolName") or ""),
                      "cwd": payload.get("cwd"), "input": ti})[:24]


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
        strict = strict_on() and global_mode() == "enforce"
        if strict:
            cannot = unreadable_payload(raw)
            if cannot:
                strict_stop(source, raw, cannot)
        payload = json.loads(raw) if raw.strip() else {}
        if source == "claude" and isinstance(payload, dict) and payload.get("cursor_version"):
            # Cursor also runs ~/.claude/settings.json: this is a Cursor call, recorded as one
            source = "cursor"
            ti = payload.get("tool_input")
            if isinstance(ti, dict) and "path" in ti and "file_path" not in ti:
                ti["file_path"] = ti["path"]
            if cursor_forwards(payload):
                return 0                     # cursor-hook.py judges this call once, so a permit is spent once
        cfg = load_config()
        del DEFECTS[:]
        del PROTECTED_EVIDENCE[:]
        del LITE_DROPS[:]
        tool, cwd, hits, preview, meta, fx_effects = classify(payload, cfg)
        for drop in LITE_DROPS:                             # every Lite stop set aside leaves its own record, whatever
            write_receipt(dict({"source": source, "session": session_of(payload), "call": call_of(payload),  # follows
                                "cwd": cwd, "tool": tool, "decision": "LITE_DROPPED",
                                "action_preview": redact(preview)}, **drop))
        ran_tests = guarded(runs_tests, False, tool, payload)     # read now, while the files this call runs are known
        meta["source"] = source
        meta["spelling"] = guarded(spelling_of, "", payload, cwd)   # a stop's mark names the exact call it stopped
        hits += guarded(chain_reach, [], payload, cwd, fx_effects)
        hits += guarded(memory_reach, [], payload, cwd, fx_effects, meta)
        hits += guarded(protected_hits, [], payload, cwd, fx_effects, cfg)
        hits += guarded(brake_hit, [], payload, tool, fx_effects)
        hits += guarded(turnover_hit, [], payload, tool, fx_effects)
        hits += guarded(test_claim_hit, [], payload, tool, cwd)
        BRAKE_TRIPPED["sid"] = ""
        CHAIN_PENDING["sid"] = session_of(payload)
        CHAIN_PENDING["marks"] = guarded(chain_marks, [], payload, cwd, fx_effects)
        hits = [h for h in hits if h[0] not in cfg["disabled_rules"]]
        failed_readers = sorted(set(DEFECTS))                # a reader that failed judged nothing

        law = load_law()
        guarded(law_lift_seen, None, law, str(payload.get("tool_name") or payload.get("toolName") or ""),
                session_of(payload))
        tmode = (law or {}).get("taste_mode", "shadow")
        t_stop, t_shadow, t_conflicts, t_cand = None, [], {}, {}
        taste_effects = fx_effects
        if law and law.get("laws") and not taste_effects:
            # owner laws govern reads too (recall-before-grep, 2026-10-02):
            # synthesize a read effect so commitments can match read-only commands
            _cmd = str((payload.get("tool_input") or {}).get("command") or "").strip()
            if _cmd:
                _argv = _cmd.split()
                taste_effects = [{"prog": _argv[0] if _argv else "", "seg": _cmd[:120],
                                  "targets": [], "recursive": False, "kind": "read",
                                  "flags": [], "args": _argv[1:4], "writes": [],
                                  "text": _cmd[:200]}]
        if law and taste_effects and law.get("laws"):
            t_key = adigest_of(payload)
            t_stop, t_shadow, t_conflicts, t_cand = run_taste(law, taste_effects, cwd, t_key)
        if strict and failed_readers and not hits:
            strict_stop(source, raw, "a reader inside the gate failed on this call (%s)" % ", ".join(failed_readers))
        if not hits and not t_stop and not t_shadow:
            # compliant-rewrite linkage: this call matches a rewrite the gate proposed
            linked = False
            try:
                pf = pj(OP_HOME, "state", "proposed-rewrites.json")
                cmd_text = str((payload.get("tool_input") or {}).get("command") or "").strip()
                pending = json.load(open(pf, encoding="utf-8")) if os.path.exists(pf) else {}
                hit = None
                for proposed, p_meta in list(pending.items()):
                    if time.time() - p_meta.get("ts", 0) > REWRITE_TTL_S:
                        pending.pop(proposed, None)
                        continue
                    if cmd_text and cmd_text == proposed.strip():
                        hit = (proposed, p_meta)
                        break
                if hit:
                    proposed, p_meta = hit
                    pending.pop(proposed, None)
                    json.dump(pending, open(pf, "w", encoding="utf-8"))
                    write_receipt({"decision": "ALLOW", "rewrite_of": p_meta.get("orig"), "attempt": 1,
                                   "source": source, "session": session_of(payload), "call": call_of(payload),
                                   "action_preview": redact(cmd_text)})
                    linked = True
            except Exception:
                pass
            if not linked:                                   # no rule and no law matched
                kept = rewind_now(fx_effects, cwd, preview, tool)
                if rewind_must_hold(kept) and global_mode() == "enforce":
                    rewind_hold(source, payload, cwd, tool, preview, meta, kept)   # stops unless the owner allowed it
                    return 0
                write_receipt(dict({"source": source, "cwd": cwd, "tool": tool, "decision": "PASS", "commitments": [],
                                    "session": session_of(payload), "call": call_of(payload),
                                    "action_preview": redact(preview)}, **({"rewind": kept} if kept else {}),
                                   **({"tests": True} if ran_tests else {}),
                                   **({"protected": list(PROTECTED_EVIDENCE)} if PROTECTED_EVIDENCE else {})))
                rewind_note(kept)
            return 0

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
        base_rec = {"source": source, "session": session_of(payload), "call": call_of(payload),
                    "cwd": cwd, "tool": tool, "action_digest": adigest, "action_preview": action["preview"],
                    "normalized": redact(meta.get("normalized", "")),
                    "transforms": meta.get("transforms", [])}
        if ran_tests:
            base_rec["tests"] = True                           # an allowed call that runs tests backs a later claim

        # specificity ordering: rule class first (canary > self > memory > workspaces), then the
        # most specific target (deepest path) -- the hit reported to the agent is the one to act on.
        priority = ["OP-CANARY", "OP-000", "OP-022", "OP-002", "OP-001", "OP-012", "OP-013", "OP-014", "OP-015", "OP-016", "OP-017", "OP-018", "OP-019", "OP-010", "OP-009", "OP-008", "OP-004",
                    "OP-003", "OP-011", "OP-005", "OP-006", "OP-023", "OP-007", "OP-026", "OP-021", "OP-020", "OP-024",
                    "OP-025", "OP-027", "OP-028", "OP-029", "OP-030",
                    "OP-031", "OP-032", "OP-033", "OP-034", "OP-035", "OP-036", "OP-037", "OP-038", "OP-039", "OP-040", "OP-041", "OP-042", "OP-043", "OP-044", "OP-045", "OP-046", "OP-047", "OP-048", "OP-049", "OP-050", "OP-051", "OP-052", "OP-053", "OP-054", "OP-055", "OP-056", "OP-057", "OP-058", "OP-059", "OP-060", "OP-061", "OP-062", "OP-063", "OP-064", "OP-065", "OP-066", "OP-067", "OP-068", "OP-069", "OP-070", "OP-071", "OP-072", "OP-073", "OP-074", "OP-075", "OP-076", "OP-077", "OP-078", "OP-079", "OP-080", "OP-081", "OP-082", "OP-083", "OP-084", "OP-085", "OP-098", "OP-099", "OP-101", "OP-102", "OP-103", "OP-104", "OP-105", "OP-106", "OP-107", "OP-108", "OP-109", "OP-110", "OP-111", "OP-112", "OP-113", "OP-114",
                    "OP-S06", "OP-S17", "OP-S12", "OP-S13", "OP-S07", "OP-S08", "OP-S11", "OP-S01",
                    "OP-S02", "OP-S03", "OP-S04", "OP-S05", "OP-S14", "OP-S15", "OP-S16"]

        def hit_depth(detail):
            best = 0
            for m in re.finditer(r"/[^\s\"']+", detail or ""):
                best = max(best, m.group(0).count("/"))
            return best

        enforce_hits.sort(key=lambda h: (priority.index(h[0]) if h[0] in priority else len(priority),
                                         -hit_depth(h[3]), h[0]))
        taste_enforcing = tmode == "enforce" and gmode == "enforce"
        soft = {h[0] for h in enforce_hits} <= {"OP-003", "OP-005", "OP-006", "OP-007", "OP-011", "OP-026"}
        if enforce_hits and t_stop and taste_enforcing and soft:
            enforce_hits = []          # owner law explicitly governs this class: taste verdict presides
        loss = guarded(measure_loss, None, fx_effects, cwd) if (enforce_hits or shadow_hits or t_stop) else None
        if loss:
            base_rec["loss"] = loss
        if enforce_hits:
            floor = any(h[0] in FLOOR_RULES for h in enforce_hits)
            permit = None if floor else consume_permit(adigest, base_rec["session"])
            if permit:
                kept = rewind_now(fx_effects, cwd, preview, tool)     # the owner allowed it: it goes ahead either way
                write_receipt(dict(base_rec, decision="ALLOW", via="permit", commitments=[h[0] for h in enforce_hits],
                                   **({"rewind": kept} if kept else {})))
                rewind_note(kept)
                return 0
            top = enforce_hits[0]
            rd = write_receipt(dict(base_rec, decision="STOP", commitment=top[0], rule=top[1],
                                    commitments=[h[0] for h in enforce_hits], detail=redact(top[3])))
            hold = top[0] in HOLD_RULES                   # a hold on the session says nothing about the target
            meta["stop_rules"] = [h[0] for h in enforce_hits]
            if not hold:
                guarded(chain_note_stop, None, base_rec["session"], fx_effects, meta, top[0], rd)
            lines = ["OPERATOR: STOPPED (%s %s)" % (top[0], top[1]),
                     "Why: %s -- %s." % (top[2], top[3])]
            if not hold:
                lines += guarded(told_lines, [], adigest, top[0], top[1], fx_effects, rd)
            if loss_line(loss):
                lines.append("Kept safe: %s." % loss_line(loss))
            if top[4]:
                lines.append("Compliant path: %s" % top[4])
            if meta.get("transforms"):
                lines.append("Gate saw through: %s." % ", ".join(meta["transforms"]))
            if meta.get("normalized") and meta["normalized"] != preview:
                lines.append("Normalized form: %s" % redact(meta["normalized"]))
            if top[0] == "OP-025":
                lines.append("No action was taken. Take the watch, then make this call again; it is judged as usual.")
            else:
                lines.append("No action was taken. Do not retry this action or a variation of it.")
                lines.append("This rule holds in every mode and has no permit." if floor else
                             "Only the owner can allow it once, from their own terminal: %s approve %s" % (self_cmd(), adigest))
            if BRAKE_TRIPPED.get("sid"):
                lines.append("The brake is now on for this session: it was stopped %d times inside %d minutes, so its "
                             "changes are held until it goes %d minutes without a stop or the owner releases it."
                             % (brake_at(), BRAKE_WINDOW_S // 60, BRAKE_QUIET_S // 60))
            lines.append("Receipt: %s" % ((rd or "unavailable")[:16]))
            lines.append("Operator receipt: %s" % (rd or "unavailable"))
            stop(lines)
        # ---- taste layer: owner commitments as laws (reached only when the constitution allows) ---- #
        if law and taste_effects and law.get("laws") and t_stop and taste_enforcing:
            c = t_stop["commitment"]
            permit = consume_permit(adigest, base_rec["session"])
            if permit:
                # the owner wrote the law and the owner, at a terminal, allowed this one action
                kept = rewind_now(fx_effects, cwd, preview, tool)
                write_receipt(dict(base_rec, decision="ALLOW", via="permit", layer="taste", **({"rewind": kept} if kept else {}),
                                   commitment=c.get("commitment_id"), law_digest=law.get("law_digest", "")))
                rewind_note(kept)
                return 0
            if guarded(law_lift_take, False, c, base_rec["session"]):
                # the law names a tool to ask first (recall); it was asked, so this one call runs
                write_receipt(dict(base_rec, decision="ALLOW", via="law_lift", layer="taste",
                                   commitment=c.get("commitment_id"), law_digest=law.get("law_digest", ""),
                                   detail=redact("; ".join(t_stop["evidence"]))))
                return 0
            rd = write_receipt(dict(base_rec, decision="STOP", layer="taste",
                                    reason_code="TASTE_VIOLATION", commitment=c.get("commitment_id"),
                                    governing_commitments=[{"id": c.get("commitment_id"), "version": c.get("version", 1)}],
                                    candidate={"classes": t_cand.get("classes", []), "prod_shaped": t_cand.get("prod_shaped", False),
                                               "resolved": t_cand.get("resolved", True)},
                                    law_digest=law.get("law_digest", ""), detail=redact("; ".join(t_stop["evidence"]))))
            guarded(chain_note_stop, None, base_rec["session"], fx_effects, meta, c.get("commitment_id"), rd)
            if rd is None:
                stop(["OPERATOR: STOPPED (RECEIPT_WRITE_FAILED) -- the taste layer could not record this verdict.",
                      "Fail-closed: fix ~/.operator and retry; nothing was executed."])
            if t_stop.get("rewrite"):
                rewrite_count_bump(t_key)
                try:
                    pf = pj(OP_HOME, "state", "proposed-rewrites.json")
                    os.makedirs(os.path.dirname(pf), exist_ok=True)
                    try:
                        pending = json.load(open(pf, encoding="utf-8"))
                    except Exception:
                        pending = {}
                    pending = {k: v for k, v in pending.items()
                               if time.time() - v.get("ts", 0) < REWRITE_TTL_S}
                    pending[t_stop["rewrite"].strip()] = {"orig": t_key, "ts": time.time()}
                    json.dump(pending, open(pf, "w", encoding="utf-8"))
                except Exception:
                    pass
            lines = ["OPERATOR: STOPPED BY YOUR LAW (%s v%s)" % (c.get("commitment_id"), c.get("version", 1)),
                     'Your law: "%s"' % c.get("directive", ""),
                     "Observed: %s" % ("; ".join(t_stop["evidence"]) or redact(preview))]
            told = guarded(told_before, None, adigest, rd)
            if told:
                lines.append(told_line(told))
            if loss_line(loss):
                lines.append("Kept safe: %s." % loss_line(loss))
            if t_stop.get("hard"):
                lines.append("A compliant rewrite was already attempted for this action. Stop and ask the owner.")
            elif t_stop.get("rewrite"):
                lines.append("Compliant path: %s" % t_stop["rewrite"])
            if t_conflicts:
                lines.append("Ambiguous commitments (%s); ask the owner to resolve." % t_conflicts)
            lift_tools = (c.get("on_violation") or {}).get("lift_once_after_tool")
            if lift_tools:
                lines.append("No action was taken. Ask %s first (an exact tag or id). If that does not answer "
                             "it, your next search like this runs once." % " or ".join(lift_tools))
            else:
                lines.append("No action was taken. Do not retry this action or a variation of it.")
            lines += ["Only the owner can allow it once, from their own terminal: %s approve %s" % (self_cmd(), adigest),
                      "Receipt: %s   Law: %s" % ((rd or "unavailable")[:16], law.get("law_digest", "")[:12]),
                      "Operator receipt: %s" % rd]
            stop(lines)
        # the call goes ahead: what it would destroy is kept first, so the owner can put it back
        kept = rewind_now(fx_effects, cwd, preview, tool)
        if rewind_must_hold(kept) and gmode == "enforce":
            rewind_hold(source, payload, cwd, tool, preview, meta, kept)       # stops unless the owner allowed it
        elif kept:
            rewind_note(kept)
        if kept:
            base_rec["rewind"] = kept
        if PROTECTED_EVIDENCE:
            base_rec["protected"] = list(PROTECTED_EVIDENCE)
        if shadow_hits:
            write_receipt(dict(base_rec, decision="SHADOW_STOP", commitments=[h[0] for h in shadow_hits],
                               detail=redact("; ".join(h[3] for h in shadow_hits))))

        # taste receipts land last: the owner's law is the final word on the record
        if t_stop:
            write_receipt(dict(base_rec, decision="SHADOW_TASTE", commitment=t_stop["commitment"].get("commitment_id"),
                               law_digest=law.get("law_digest", ""), detail=redact("; ".join(t_stop["evidence"]))))
        elif t_shadow:
            write_receipt(dict(base_rec, decision="SHADOW_TASTE_MULTI",
                               detail=redact("; ".join(ev for _, ev in t_shadow))))
        return 0
    except SystemExit:
        raise
    except Exception as exc:  # gate defect: degrade, never brick
        try:
            seen = json.loads(raw) if raw.strip() else {}
            seen = seen if isinstance(seen, dict) else {}
        except Exception:
            seen = {}
        write_receipt({"source": source, "decision": "GATE_ERROR", "error": redact(repr(exc)),
                       "session": session_of(seen), "call": call_of(seen),
                       "action_preview": redact(raw[:300])})
        if degraded_hit(raw):
            stop(["OPERATOR: STOPPED (degraded) -- the gate hit an internal error and this call looks destructive.",
                  "Ask the owner to review; the error is recorded in ~/.operator/receipts."])
        try:
            strict = strict_on() and global_mode() == "enforce"
        except Exception:
            strict = False
        if strict:
            stop(["OPERATOR: STOPPED (OP-STRICT cannot-read-call)",
                  "Why: strict mode is on, and the gate hit an internal error while judging this call.",
                  "No action was taken. Ask the owner to review; the error is recorded in ~/.operator/receipts."])
        return 0


# --------------------------------------------------------------------------- #
# owner CLI
# --------------------------------------------------------------------------- #
def rule_hint(rec):
    """The rule's own sentence for a stop receipt, with the loss hint on OP-011: the compliant path."""
    cid = str(rec.get("commitment") or "")
    msg = (RULES.get(cid) or ("", "", ""))[2]
    return (msg + " " + LOSS_HINT).strip() if cid == "OP-011" else msg


def cmd_approve(argv):
    if len(argv) < 1 or not re.match(r"^[0-9a-f]{24}$", argv[0]):
        print("usage: %s approve <24-hex action digest> [--prove [--no-walk]]" % self_cmd())
        return 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: approvals need an interactive terminal run by the owner.")
        return 3
    ensure_dirs()
    matched = None
    try:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with open(pj(OP_HOME, "receipts", day + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Approval starts from an actual STOP, never a substring or later ALLOW/PASS.
                    if rec.get("decision") != "STOP" or rec.get("action_digest") != argv[0]:
                        continue
                    receipt_digest = rec.get("digest") or ""
                    content = dict(rec)
                    content.pop("digest", None)
                    if (re.fullmatch(r"[0-9a-f]{64}", receipt_digest)
                            and receipt_digest == digest_of(content) and rec.get("session")):
                        matched = rec
                except Exception:
                    continue
    except Exception:
        pass
    if matched:
        r = matched
        print("Blocked action: %s\n  rule: %s\n  preview: %s" % (r.get("tool"), r.get("rule") or r.get("commitment"), r.get("action_preview")))
    else:
        print("No matching STOP receipt today for %s." % argv[0])
        return 4
    evidence = None
    if "--prove" in argv[1:]:
        # replay the stopped action in a sealed copy on the owner's side, so the owner decides on what it did
        try:
            sealed = _feature("sealed")
            rec = matched
            # the folder replay, and beside it a frozen copy of memory walked inside the same sandbox
            # (--no-walk keeps the replay alone); the live store receives nothing from either
            evidence = sealed.prove_receipt(rec, walk="--no-walk" not in argv[1:], hint=rule_hint(rec))
            print("\n".join(sealed.card(evidence)))
            if (not isinstance(evidence, dict) or not evidence.get("id")
                    or evidence.get("error") or evidence.get("exit_code") != 0
                    or ("--no-walk" not in argv[1:] and
                        (not isinstance(evidence.get("walk"), dict) or evidence["walk"].get("error")
                         or not evidence["walk"].get("id")))):
                print("Refusing: requested proof is incomplete; no permit was granted.")
                return 5
        except Exception as exc:
            print("Sealed copy not run: %s" % redact(str(exc)))
            return 5
    if input("Type ALLOW ONCE to grant a single-use permit (%d min): " % (PERMIT_TTL_S // 60)).strip() != "ALLOW ONCE":
        print("Cancelled.")
        return 1
    proof = {"stop_receipt": matched["digest"], "session": matched["session"],
             "origin_source": matched.get("source") or "", "single_use": True}
    if evidence:
        proof["evidence"] = evidence["id"]
    if evidence and isinstance(evidence.get("walk"), dict) and not evidence["walk"].get("error"):
        proof["walk"] = sealed.walk_summary(evidence)          # the frozen-store verdict, on the permit and its receipt
    p = pj(OP_HOME, "permits", argv[0] + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict({"action_digest": argv[0], "granted": time.time(), "expires": time.time() + PERMIT_TTL_S,
                        "single_use": True, "by": "owner-tty"}, **proof), f)
    os.chmod(p, 0o600)
    write_receipt(dict({"decision": "PERMIT_GRANTED", "action_digest": argv[0], "source": "owner-tty"}, **proof))
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


ON_FAILURE_MIN = (2, 1, 295)             # the first Claude Code whose command hooks take onFailure
GATE_HOOK_TIMEOUT_S = 60                 # with onFailure block, a gate that hangs this long stops the call


def claude_code_version():
    """The installed Claude Code's version as a tuple, or None when it cannot be read."""
    exe = shutil.which("claude")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out or "")
    return tuple(int(x) for x in m.groups()) if m else None


def hook_fails_closed():
    """Does Claude Code stop the call when the gate itself fails (crashes, hangs, cannot start)?"""
    try:
        with open(pj(HOME, ".claude", "settings.json"), encoding="utf-8") as f:
            settings = json.load(f)
        return any("operator-gate" in hh.get("command", "") and hh.get("onFailure") == "block"
                   for h in settings.get("hooks", {}).get("PreToolUse", []) for hh in h.get("hooks", []))
    except Exception:
        return False


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


def active_laws():
    law = guarded(load_law, None) or {}
    laws = [c for c in (law.get("laws") or []) if isinstance(c, dict) and c.get("state", "active") == "active"]
    return law, laws


def _law_drafts():
    try:
        with open(law_drafts_path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def laws_drafts():
    """`laws drafts`: the laws repeated stops drafted, waiting for the owner."""
    drafts = _law_drafts()
    if not drafts:
        print("No drafted laws. The third stop of the same action, never allowed, drafts one here.")
        return 0
    print("Laws drafted from repeated stops (the owner adopts one with `%s laws adopt <id>`):" % self_cmd())
    for cid, d in sorted(drafts.items(), key=lambda kv: -kv[1].get("drafted", 0)):
        print("  %s  (%d stops)  %s" % (cid, d.get("stops", 0), (d.get("law") or {}).get("directive", "")[:160]))
    return 0


def laws_adopt(argv):
    """`laws adopt <id>`: the owner makes a drafted law one of their own. It is validated with the rest of
    the owner's laws before anything is written."""
    if any(os.environ.get(k) for k in AGENT_ENV):
        print("Refusing: adopting a law changes the owner's rules, so the owner runs it from their own terminal.")
        return 3
    drafts = _law_drafts()
    if not argv or argv[0] not in drafts:
        print("usage: %s laws adopt <id>   (`%s laws drafts` lists them)" % (self_cmd(), self_cmd()))
        return 2
    law = dict(drafts[argv[0]]["law"], effective_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    cpath = pj(OP_HOME, "commitments.json")
    try:
        with open(cpath, encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        doc = {"schema_version": "1.0.0", "taste_mode": "enforce"}
    except ValueError:
        print("Refusing: %s does not parse; fix it first, it was left as it is." % cpath)
        return 1
    doc["commitments"] = [c for c in doc.get("commitments") or [] if c.get("commitment_id") != law["commitment_id"]] + [law]
    digest, err = write_law_cache(doc)
    if err:
        print("Not adopted: the law would not validate (%s)." % err)
        return 1
    tmp = cpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, cpath)
    drafts.pop(argv[0], None)
    with open(law_drafts_path(), "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=1)
    write_receipt({"decision": "LAW_ADOPTED", "commitment": law["commitment_id"], "law_digest": digest, "source": "owner-tty"})
    print("Adopted %s (law digest %s, taste mode %s). Every session now starts with it." % (law["commitment_id"], digest[:12], doc.get("taste_mode", "shadow")))
    return 0


def cmd_laws(argv):
    """The owner's laws in force. `--session-start` prints them as Claude Code SessionStart context, so
    they come back at every session start and after every compaction; the gate enforces them whatever
    the conversation says. It reads only, and prints nothing when there is nothing to say."""
    if argv[:1] == ["drafts"]:
        return laws_drafts()
    if argv[:1] == ["adopt"]:
        return laws_adopt(argv[1:])
    if "--session-start" in argv:
        return session_briefing(argv)
    law, laws = active_laws()
    if not laws:
        print("No owner laws in force. Write them in %s and run: %s commitments check" % (pj(OP_HOME, "commitments.json"), self_cmd()))
        return 0
    print("Owner laws in force (law digest %s):" % str(law.get("law_digest") or "")[:12])
    for c in laws:
        print("  %s v%s: %s" % (c.get("commitment_id"), c.get("version", 1), c.get("directive", "")))
    return 0


def cmd_upgrade(argv):
    """Operator Lite's `upgrade` command, answered by the gate that is the upgrade."""
    mod = lite_core()
    print("This is Vestige Operator %s, the paid gate. Nothing to upgrade to." % VERSION)
    print("It includes every protection in Operator Lite %s (%s), plus: OP-012 the-floor, OP-013 no-root-reach, "
          "OP-015 no-repo-code-run, OP-016 no-vulnerable-write, OP-017 no-agent-arming, OP-019 no-harmful-chain, "
          "OP-020 no-second-spelling, OP-021 no-found-key, your own laws, coverage (the calls it never saw), the Board, the Letter and onboarding."
          % (LITE_CORE_VERSION, "judging every call" if mod else _LITE_CORE.get("status", "missing")))
    print("All rules: %s status --rules" % self_cmd())
    return 0


def cmd_rewind(argv):
    """`rewind` lists what the gate kept; `rewind <id>` shows one; `rewind <id> --restore` puts it back
    (the owner, from their own terminal)."""
    ms = rewind_manifests()
    if not argv or argv[0] in ("--list", "list"):
        if not ms:
            print("Rewind has kept nothing yet. Before a call the gate lets through destroys work that exists "
                  "nowhere else, the gate keeps that work here for a week.")
            return 0
        print("Kept before calls the gate let through (newest first; restore: %s rewind <id> --restore):" % self_cmd())
        for m in ms[:40]:
            files = len(m.get("items") or [])
            print("  %s  %s  %d path%s%s%s  %s" % (m["id"], time.strftime("%m-%d %H:%M", time.localtime(m.get("t", 0))), files,
                  "" if files == 1 else "s", ", %d stash list" % len(m["git"]) if m.get("git") else "",
                  "" if m.get("complete") else (" (partial: %d not kept)" % m["missed"] if m.get("missed") else " (partial)"),
                  m.get("preview", "")[:80]))
        return 0
    m = next((x for x in ms if x.get("id") == argv[0]), None)
    if not m:
        print("No snapshot %s. `%s rewind` lists them." % (argv[0], self_cmd()))
        return 2
    if "--restore" not in argv:
        print("%s, kept %s before: %s" % (m["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m.get("t", 0))), m.get("preview", "")))
        for it in m.get("items") or []:
            print("  %s%s  (%s%s)" % (short_home(it["path"]), "/" if it.get("dir") else "", it.get("method"), "" if it.get("complete") else ", partial"))
        for g in m.get("git") or []:
            print("  %s: %d stash%s" % (short_home(g["root"]), len(g.get("stashes") or []), "" if len(g.get("stashes") or []) == 1 else "es"))
        print("Put it back: %s rewind %s --restore" % (self_cmd(), m["id"]))
        return 0
    if any(os.environ.get(k) for k in AGENT_ENV):
        print("Refusing: rewind --restore writes files back, so the owner runs it from their own terminal.")
        return 3
    res = rewind_restore(m)
    write_receipt({"decision": "REWIND_RESTORED", "rewind": m["id"], "files": res["files"], "stashes": res["stashes"],
                   "before": res["before"], "source": "owner-tty"})
    print("Restored %d of %d kept files%s from %s." % (res["files"], res["of"],
          (" and %d stash%s" % (res["stashes"], "" if res["stashes"] == 1 else "es")) if res["stashes"] else "", m["id"]))
    if res["before"]:
        print("The files it replaced are kept as %s, so this restore can be rewound too." % res["before"])
    return 0


# ---- coverage: the tool calls that never reached the gate ------------------------------------- #
CODEX_CALL_TYPES = {"function_call", "custom_tool_call", "local_shell_call"}
JUDGED_DECISIONS = {"PASS", "STOP", "ALLOW", "SHADOW_STOP", "SHADOW_TASTE", "SHADOW_TASTE_MULTI", "GATE_ERROR"}


def _iso_time(s):
    try:
        return calendar.timegm(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except Exception:
        return 0


def _call_preview(name, inp):
    if isinstance(inp, dict):
        for k in ("command", "cmd", "file_path", "path", "notebook_path", "url", "pattern", "query", "skill", "description"):
            if inp.get(k):
                return "%s %s" % (name, " ".join(str(inp[k]).split()))
    return str(name or "")


def _preview_key(tool, preview):
    """The first words of a call's command or path, comparable between a receipt and a transcript."""
    text = " ".join(str(preview or "").split())
    if tool and text.startswith(str(tool) + " "):
        text = text[len(str(tool)) + 1:]
    text = re.sub(r"[^A-Za-z0-9/._~-]+", " ", text).strip()
    return text[:24]


def claude_calls(since):
    """Every tool call in Claude Code's transcripts since `since`, in main sessions and in subagents."""
    base = pj(HOME, ".claude", "projects")
    out = []
    for path in glob.glob(pj(base, "*", "*.jsonl")) + glob.glob(pj(base, "*", "*", "subagents", "*.jsonl")):
        try:
            if os.path.getmtime(path) < since:
                continue
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"tool_use"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = _iso_time(d.get("timestamp"))
                    if t < since or not isinstance(d.get("message"), dict):
                        continue
                    for c in d["message"].get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("id"):
                            out.append({"host": "claude", "session": str(d.get("sessionId") or "")[:64],
                                        "sub": bool(d.get("isSidechain")), "agent": str(d.get("agentId") or ""),
                                        "call": str(c["id"])[:80], "tool": str(c.get("name") or ""), "t": t,
                                        "preview": _call_preview(c.get("name"), c.get("input"))})
        except OSError:
            continue
    return out


def codex_threads(since):
    """Tool calls per Codex thread since `since`, read from ~/.codex/sessions rollouts."""
    out = {}
    for dp, _, names in os.walk(pj(HOME, ".codex", "sessions")):
        for name in names:
            path = pj(dp, name)
            try:
                if not (name.startswith("rollout-") and name.endswith(".jsonl")) or os.path.getmtime(path) < since:
                    continue
                meta, n, last = {}, 0, 0
                with open(path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        p = d.get("payload") if isinstance(d.get("payload"), dict) else {}
                        if d.get("type") == "session_meta":
                            meta = p
                        elif d.get("type") == "response_item" and p.get("type") in CODEX_CALL_TYPES:
                            t = _iso_time(d.get("timestamp"))
                            if t >= since:
                                n, last = n + 1, max(last, t)
                if n:
                    tid = str(meta.get("id") or name)[:64]
                    out[tid] = {"session": str(meta.get("session_id") or "")[:64], "calls": n, "last": last,
                                "cwd": str(meta.get("cwd") or "")}
            except OSError:
                continue
    return out


CURSOR_FAMILY = {"Shell": "shell", "run_terminal_cmd": "shell", "RunTerminalCommand": "shell",
                 "Write": "edit", "StrReplace": "edit", "EditFile": "edit", "edit_file": "edit", "MultiEdit": "edit",
                 "Delete": "edit", "DeleteFile": "edit", "delete_file": "edit", "EditNotebook": "edit", "ApplyPatch": "edit",
                 "CallDynamicTool": "mcp", "CallMcpTool": "mcp",
                 "Read": "read", "ReadFile": "read", "read_file": "read", "Grep": "read", "Glob": "read",
                 "grep_search": "read", "file_search": "read", "codebase_search": "read", "list_dir": "read", "LS": "read"}


def call_family(tool):
    """shell, edit, mcp, read or other: how a Cursor transcript's tool name and a receipt's tool line up."""
    t = str(tool or "")
    if t.startswith("mcp__") or t.startswith("MCP:"):
        return "mcp"
    if t in ("Edit", "MultiEdit", "NotebookEdit"):
        return "edit"
    return CURSOR_FAMILY.get(t, "other")


def cursor_conversations(since):
    """Tool calls per Cursor agent conversation touched since `since`, by family. Cursor's transcripts
    carry no times, so a conversation is counted whole."""
    out = {}
    for path in glob.glob(pj(HOME, ".cursor", "projects", "*", "agent-transcripts", "*", "*.jsonl")):
        try:
            if os.path.getmtime(path) < since:
                continue
            fam = {}
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"tool_use"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    msg = d.get("message") if isinstance(d.get("message"), dict) else {}
                    for c in msg.get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            f = call_family(c.get("name"))
                            fam[f] = fam.get(f, 0) + 1
            if fam:
                out[os.path.basename(os.path.dirname(path))[:64]] = {"calls": sum(fam.values()), "families": fam,
                                                                      "last": os.path.getmtime(path)}
        except OSError:
            continue
    return out


def coverage_report(days):
    """What reached the gate, measured against what the agents did: Claude Code call by call (by the
    call id, or for receipts written before 0.3.14 by session, tool and time), Codex and Cursor by count."""
    since = time.time() - days * 86400
    recs = [r for r in recent_receipts(int(days) + 1) if r.get("decision") in JUDGED_DECISIONS and receipt_time(r) >= since - 60]
    by_call = {r["call"] for r in recs if r.get("call")}
    loose = {}
    for r in recs:
        if not r.get("call") and r.get("source") == "claude":
            loose.setdefault((r.get("session") or "", r.get("tool") or ""), []).append(
                (receipt_time(r), _preview_key(r.get("tool"), r.get("action_preview"))))
    for v in loose.values():
        v.sort()
    # a resumed or forked session copies the earlier rows into its own file under its own id, so one
    # call can sit in several transcripts: it is one call, judged under whichever session made it
    one = {}
    for c in sorted(claude_calls(since), key=lambda c: c["t"]):
        if c["call"] in one:
            one[c["call"]]["sessions"].add(c["session"])
        else:
            one[c["call"]] = dict(c, sessions={c["session"]})
    calls = sorted(one.values(), key=lambda c: c["t"])
    unseen, by_id, by_time = [], 0, 0
    for c in calls:
        if c["call"] in by_call:
            by_id += 1
            continue
        # receipts before 0.3.14 carry no call id: take the nearest receipt of the same tool in a session
        # holding the call, preferring one whose command agrees. A hook can fire seconds before the row is
        # written, or minutes after when the call waited behind others.
        key, best = _preview_key(c["tool"], c["preview"]), None
        for s in sorted(c["sessions"]):
            rows = loose.get((s, c["tool"])) or []
            for i, (t, pk) in enumerate(rows):
                if not (c["t"] - 15 <= t <= c["t"] + 300):
                    continue
                score = (0 if (not pk or not key or pk == key) else 1, abs(t - c["t"]))
                if best is None or score < best[0]:
                    best = (score, rows, i)
        if best and best[0][0] == 0:
            best[1].pop(best[2])
            by_time += 1
            continue
        unseen.append(c)
    for c in calls:
        c["sessions"] = sorted(c["sessions"])
    per_source = {}
    for r in recs:
        per_source.setdefault(r.get("source") or "unknown", {}).setdefault(r.get("session") or "", 0)
        per_source[r.get("source") or "unknown"][r.get("session") or ""] += 1
    codex = codex_threads(since)
    cx_recs = per_source.get("codex", {})
    for tid, v in codex.items():
        v["receipts"] = cx_recs.get(tid, 0) + (cx_recs.get(v["session"], 0) if v["session"] and v["session"] != tid else 0)
    cursor = cursor_conversations(since)
    cu_recs, cu_fam = {}, {}
    if cursor:
        # a conversation is counted whole, so its receipts are counted over the week before the window too
        for r in recent_receipts(int(days) + 8):
            sid = r.get("session") or ""
            # gates before 0.3.14 recorded Cursor's import of ~/.claude/settings.json as source claude
            if sid in cursor and r.get("decision") in JUDGED_DECISIONS:
                cu_recs[sid] = cu_recs.get(sid, 0) + 1
                f = call_family(r.get("tool"))
                cu_fam[f] = cu_fam.get(f, 0) + 1
    for cid, v in cursor.items():
        v["receipts"] = cu_recs.get(cid, 0)
    last = {}
    for r in recs:
        s = r.get("source") or "unknown"
        last[s] = max(last.get(s, 0), receipt_time(r))
    try:
        receipt_tail_digest(pj(OP_HOME, "receipts"))         # a tail that does not verify stops every append
        chain = "ok"
    except Exception as exc:
        chain = str(exc) or type(exc).__name__
    return {"days": days, "since": since, "chain": chain, "claude": {"calls": len(calls), "by_id": by_id, "by_time": by_time,
            "unseen": unseen, "sessions": len({c["session"] for c in calls}),
            "subagent_calls": sum(1 for c in calls if c["sub"]),
            "subagent_unseen": sum(1 for c in unseen if c["sub"])},
            "codex": codex, "cursor": cursor, "cursor_receipt_families": cu_fam, "last_receipt": last,
            "receipts_codex": sum(cx_recs.values()), "receipts_cursor": sum(cu_recs.values())}


def cmd_coverage(argv):
    """The tool calls that never reached the gate. `coverage [--days N] [--all] [--json] [--fail-on-gap]`."""
    days = 1.0
    if "--days" in argv:
        try:
            days = max(0.05, min(float(argv[argv.index("--days") + 1]), 30.0))
        except (IndexError, ValueError):
            print("usage: operator-gate coverage [--days N] [--all] [--json] [--fail-on-gap]")
            return 2
    rep = coverage_report(days)
    cl = rep["claude"]
    if "--json" in argv:
        print(json.dumps(rep, sort_keys=True, default=str))
        return 3 if ("--fail-on-gap" in argv and cl["unseen"]) else 0
    pct = lambda a, b: ("%.1f%%" % (100.0 * a / b)) if b else "n/a"
    judged = cl["by_id"] + cl["by_time"]
    print("Operator coverage, last %s (since %s)" % (("%g days" % days) if days != 1 else "24 hours",
          time.strftime("%Y-%m-%d %H:%M", time.localtime(rep["since"]))))
    if rep["chain"] != "ok":
        print("RECEIPTS ARE NOT BEING WRITTEN: the last receipt in %s does not verify (%s), and the gate adds no "
              "receipt after a row it cannot verify. Run `%s verify` from your own terminal."
              % (short_home(pj(OP_HOME, "receipts")), rep["chain"], self_cmd()))
    print("Claude Code: judged %s of %s tool calls (%s) in %d sessions; %d never reached the gate."
          % (format(judged, ","), format(cl["calls"], ","), pct(judged, cl["calls"]), cl["sessions"], len(cl["unseen"])))
    if cl["by_time"]:
        print("  matched by call id %s, by session, tool and time %s (receipts from a gate before 0.3.14 carry no call id)"
              % (format(cl["by_id"], ","), format(cl["by_time"], ",")))
    if cl["subagent_calls"]:
        print("  subagents: %s of %s calls judged" % (format(cl["subagent_calls"] - cl["subagent_unseen"], ","),
                                                     format(cl["subagent_calls"], ",")))
    for host, label, word in (("codex", "Codex", "threads"), ("cursor", "Cursor", "conversations")):
        rows = rep[host]
        n = sum(v["calls"] for v in rows.values())
        got = rep["receipts_" + host]
        if not rows:
            print("%s: no agent tool calls found in this window." % label)
        elif not got:
            print("%s: %s tool calls in %d %s, and no %s call reached the gate: the gate is not on %s's path."
                  % (label, format(n, ","), len(rows), word, label, label))
        else:
            short = sorted(((k, v) for k, v in rows.items() if v["receipts"] < v["calls"]), key=lambda kv: -kv[1]["last"])
            print("%s: %s tool calls in %d %s, %s gate receipts from %s; %d %s have fewer receipts than calls%s."
                  % (label, format(n, ","), len(rows), word, format(got, ","), label, len(short), word,
                     " (Cursor transcripts carry no times, so each conversation is counted whole)" if host == "cursor" else ""))
            if host == "cursor":
                fam = {}
                for v in rows.values():
                    for f, k in v["families"].items():
                        fam[f] = fam.get(f, 0) + k
                got_f = rep["cursor_receipt_families"]
                for f, label_f in (("shell", "shell commands"), ("edit", "file edits"), ("mcp", "MCP tools"), ("read", "reads and searches")):
                    if fam.get(f):
                        note = "  <- these never reach the gate" if not got_f.get(f) else ""
                        print("  %-19s %6s calls, %6s receipts%s" % (label_f, format(fam[f], ","), format(got_f.get(f, 0), ","), note))
    for s, t in sorted(rep["last_receipt"].items()):
        print("  last call judged from %s: %s" % (s, ago(t)))
    if cl["unseen"]:
        by_tool = {}
        for c in cl["unseen"]:
            by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
        print("Never reached the gate, by tool: %s" % ", ".join("%s %d" % kv for kv in sorted(by_tool.items(), key=lambda kv: -kv[1])))
        show = cl["unseen"][::-1] if "--all" in argv else cl["unseen"][::-1][:25]
        for c in show:
            print("  %s  %s%s  %-14s %s" % (time.strftime("%m-%d %H:%M:%S", time.localtime(c["t"])), c["session"][:8],
                                           (" sub " + c["agent"][:6]) if c["sub"] else "",
                                           c["tool"][:14], redact(c["preview"])[:90]))
        if len(show) < len(cl["unseen"]):
            print("  ... %d more: %s coverage --all" % (len(cl["unseen"]) - len(show), self_cmd()))
    return 3 if ("--fail-on-gap" in argv and cl["unseen"]) else 0


def cmd_status(argv):
    """What the gate did on this machine. `--rules` prints the rule table instead."""
    mode = "off" if os.path.exists(pj(OP_HOME, "DISABLED")) else global_mode()
    lite = lite_core()
    lite_line = ("Operator Lite %s inside: judging every call (sha256 %s)" % (LITE_CORE_VERSION, LITE_CORE_SHA256[:16])
                 if lite else "Operator Lite %s inside: %s" % (LITE_CORE_VERSION, _LITE_CORE.get("status", "missing")))
    if "--rules" in argv:
        print("operator-gate %s  home=%s  mode=%s" % (VERSION, OP_HOME, mode))
        print(lite_line)
        for rid, (name, base, why) in sorted(RULES.items()):
            print("  %-9s %-7s %-28s %s" % (rid, base, name, why))
        return 0
    me, wired = self_cmd(), hook_wired()
    says = {"enforce": "enforce (blocking)", "shadow": "shadow (recording, not blocking)",
            "off": "off (nothing is checked)"}[mode]
    closed = wired and hook_fails_closed()
    print("Operator %s   mode: %s   Claude Code hook: %s" % (VERSION, says, ("registered, fails closed" if closed else "registered")
                                                            if wired else "NOT registered"))
    print(lite_line)
    if not wired:
        print("Register it: %s install" % me)
    elif not closed:
        print("The hook fails open: if the gate crashed or hung, Claude Code would let the call through. On Claude Code "
              "%s or later, `%s install` makes it fail closed." % (".".join(map(str, ON_FAILURE_MIN)), me))
    recs = recent_receipts(7)
    calls = [r for r in recs if r.get("decision") in ("PASS", "ALLOW", "STOP", "SHADOW_STOP", "SHADOW_TASTE", "GATE_ERROR")]
    if not calls:
        print("\nNo tool calls checked in the last 7 days. Start a Claude Code session: every tool call passes\n"
              "through the gate, and this view fills in.")
        print("\nYour last 30 days, replayed: %s replay\nAll rules: %s status --rules" % (me, me))
        return 0

    def hard(rec):
        return any(RULES.get(c, ("", "", ""))[1] == "STOP" for c in rec.get("commitments") or [])

    stopped = [r for r in calls if r.get("decision") == "STOP"]
    ran = [r for r in calls if (r.get("decision") == "SHADOW_STOP" and hard(r)) or r.get("decision") == "SHADOW_TASTE"]
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


def cmd_strict(argv):
    """Owner command: turn strict on or off. Strict stops a call the gate cannot read (a payload it
    cannot parse, a command that leaves a quote or bracket open, a reader that failed)."""
    flag = pj(OP_HOME, "STRICT")
    if not argv:
        print("strict is %s. To change it: %s strict on|off" % ("on" if strict_on() else "off", self_cmd()))
        return 0
    if argv[0] not in ("on", "off"):
        print("usage: %s strict on|off" % self_cmd())
        return 2
    if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get("OPERATOR_AGENT_SESSION"):
        print("Refusing: strict is changed by the owner in an interactive terminal.")
        return 3
    ensure_dirs()
    if argv[0] == "on":
        with open(flag, "w", encoding="utf-8") as f:
            f.write("on\n")
    elif os.path.exists(flag):
        os.remove(flag)
    write_receipt({"decision": "STRICT_SET", "strict": argv[0], "source": "owner-tty"})
    print({"on": "Strict is on: a call the gate cannot read is stopped.",
           "off": "Strict is off: a call the gate cannot read is judged on what it can recognise."}[argv[0]])
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


def at_terminal():
    """A person is reading: stdout is a terminal and this is not an agent's session."""
    return sys.stdout.isatty() and not os.environ.get("OPERATOR_AGENT_SESSION")


def replay_law_hits(law, effects, ti, cwd, now_iso):
    """The owner's laws that would have fired on this call: [(commitment_id, directive)]."""
    if not (law and law.get("laws")):
        return []
    fx = effects
    if not fx:                                      # laws govern reads too, as in the hook
        cmd = str(ti.get("command") or "").strip() if isinstance(ti, dict) else ""
        if not cmd:
            return []
        argv = cmd.split()
        fx = [{"prog": argv[0] if argv else "", "seg": cmd[:120], "targets": [], "recursive": False,
               "kind": "read", "flags": [], "args": argv[1:4], "writes": [], "text": cmd[:200]}]
    project_id = os.path.basename((cwd or "").rstrip("/")) or "unknown"
    try:
        hits, _, _ = evaluate_taste(law, fx, project_id, now_iso)
    except Exception:
        return []
    return [(c.get("commitment_id") or "", c.get("directive") or "") for c, _ in hits]


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
    law = load_law() or {}
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
           "projects": 0, "by_law": {}, "law_text": {}, "law_calls": 0,
           "laws_active": len(law.get("laws") or []), "truncated": False, "days": days or 0, "files": len(dated)}

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
                    else:
                        fired = replay_law_hits(law, effects, ti, cwd, now_iso)
                        if fired:                   # the owner's law decides this call
                            out["law_calls"] += 1
                            for cid, directive in fired:
                                bump("by_law", cid, show(preview))
                                out["law_text"][cid] = directive
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
        print("Operator would have stopped %d, flagged %d, and found %d that only I can rule on."
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
    if r["laws_active"]:
        print("  %7s  your own laws would have stopped" % paint("%7s" % format(r["law_calls"], ","), "1;32"))
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
    if r["by_law"]:
        print("\n" + paint("Your laws, replayed over this history:", "1"))
        for cid, n in sorted(r["by_law"].items(), key=lambda kv: (-kv[1], kv[0])):
            print("  %5d  %-54s %s" % (n, '"%s"' % r["law_text"].get(cid, cid), paint(r["samples"].get(cid, ""), "2")))
    if r["flagged"]:
        print("\n" + paint("Flagged in shadow", "1") + ", recorded and not stopped:")
        table(r["by_shadow"], lambda k: "%s %s" % (k, RULES[k][0]))
    if r["own_calls"]:
        names = dict((c, what) for c, what, _ in OWN_LAW)
        print("\n" + paint("No built-in rule decides these. They ran:", "1"))
        table(r["by_own"], lambda k: names[k])
        print("\n" + paint("Laws this history drafts that you have not written yet:", "1"))
        for law in r["laws"]:
            print("  %-54s %s" % ('"%s"' % law["law"], plural(law["times"], "time")))
    if not here:
        try:                                       # remembered for the status hint and for `upgrade`
            if os.path.isdir(OP_HOME):
                ensure_dirs()
                keep = {k: r[k] for k in ("calls", "sessions", "projects", "stopped", "flagged", "own_calls",
                                          "days", "by_own", "laws", "by_law", "law_calls")}
                keep["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                path = pj(OP_HOME, "state", "replay.json")
                with open(path + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(keep, f, sort_keys=True)
                os.replace(path + ".tmp", path)
        except Exception:
            pass
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


AGENT_ENV = ("OPERATOR_AGENT_SESSION", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")


def agent_session():
    """The variable that marks this shell as an AI agent's, or None.

    The installer calls this before it copies the gate. An agent shell is refused.
    """
    for name in AGENT_ENV:
        if os.environ.get(name):
            return name
    return None
OPERATOR_MODE_SETTLED = "operator-mode-settled.json"


def owner_chose_shadow():
    """True when the owner's last word on the mode, a MODE_SET receipt from the `mode` command in their
    terminal, chose shadow."""
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


def settle_operator_mode():
    """Operator blocks from install, like Operator Lite. A fresh install writes enforce. The shadow that
    an Operator install before 0.3.14 wrote becomes enforce once, with a MODE_SET receipt; a shadow the
    owner chose with `mode shadow` stays, and so does every choice after the settling."""
    mode_path, marker = pj(OP_HOME, "mode"), pj(OP_HOME, "state", OPERATOR_MODE_SETTLED)
    if not os.path.exists(mode_path):
        with open(mode_path, "w", encoding="utf-8") as f:
            f.write("enforce\n")
        switched = False
    elif os.path.exists(marker):
        return False
    else:
        with open(mode_path, encoding="utf-8") as f:
            current = f.read().strip()
        switched = current == "shadow" and not owner_chose_shadow()
        if switched:
            with open(mode_path, "w", encoding="utf-8") as f:
                f.write("enforce\n")
            write_receipt({"decision": "MODE_SET", "mode": "enforce", "was": "shadow", "source": "install"})
    try:
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"gate_version": VERSION, "switched": switched,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    except Exception:
        pass
    return switched


def cmd_install(argv):
    """Owner-side install: copies the gate to ~/.operator/gate and wires Claude Code. A fresh install
    blocks from the first call (enforce), the same as Operator Lite."""
    agent = next((name for name in AGENT_ENV if os.environ.get(name)), None)
    if agent:
        print("Refusing: installs are run by the owner in their own terminal (%s is set: this shell "
              "belongs to an agent)." % agent)
        return 3
    dst_dir = pj(HOME, ".operator", "gate")
    src = os.path.abspath(__file__)
    os.makedirs(dst_dir, exist_ok=True)
    dst = pj(dst_dir, "operator-gate.py")
    with open(src, encoding="utf-8") as f:
        body = f.read()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if os.path.exists(dst) and os.path.abspath(dst) != src:
        # the gate being replaced is kept, so an install can always be undone
        try:
            with open(dst, encoding="utf-8") as f:
                prev = f.read()
            if prev != body:
                os.makedirs(pj(HOME, ".operator", "install-backups"), exist_ok=True)
                with open(pj(HOME, ".operator", "install-backups", "operator-gate-replaced-%s.py" % stamp), "w",
                          encoding="utf-8") as f:
                    f.write(prev)
        except Exception:
            pass
    with open(dst, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(dst, 0o755)
    # the $149-tier modules load from the gate's own directory (see _feature), so they travel with it
    for name in ("onboard.py", "board.py", "letter.py", "almost_forgot.py", "replay.py", "sealed.py", "briefing.py", "commitments.template.json",
                 LITE_CORE_FILE):
        companion = pj(os.path.dirname(src), name)
        if os.path.exists(companion):
            with open(companion, "rb") as f:                   # byte for byte: lite_core.py is checked against its pin
                body = f.read()
            with open(pj(dst_dir, name), "wb") as f:
                f.write(body)
    src_corpora = pj(os.path.dirname(src), "corpora")
    if os.path.isdir(src_corpora):
        os.makedirs(pj(dst_dir, "corpora"), exist_ok=True)
        for name in os.listdir(src_corpora):
            if name.endswith(".json"):
                with open(pj(src_corpora, name), encoding="utf-8") as f:
                    body = f.read()
                with open(pj(dst_dir, "corpora", name), "w", encoding="utf-8") as f:
                    f.write(body)
    ensure_dirs()
    mode_path = pj(OP_HOME, "mode")
    if guarded(settle_operator_mode, False):
        print("mode: the shadow an earlier Operator install wrote is now enforce, like Operator Lite. "
              "To record without blocking: %s mode shadow" % self_cmd())
    # Claude Code PreToolUse hook (merge, never clobber)
    settings_path = pj(HOME, ".claude", "settings.json")
    hook_cmd = hook_command(dst, "claude")
    try:
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        raw_settings = None
        if os.path.exists(settings_path):
            with open(settings_path, encoding="utf-8") as f:
                raw_settings = f.read()
        if raw_settings is not None and raw_settings.strip():
            # a settings file that does not parse is the owner's to fix; it is never replaced with an empty one
            settings = json.loads(raw_settings)
            if not isinstance(settings, dict):
                raise ValueError("the top level of settings.json is not an object")
        else:
            settings = {}
        hooks = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
        # fail closed: a gate that crashes, hangs or cannot start stops the call instead of letting it through
        cc = claude_code_version()
        closed = bool(cc) and cc >= ON_FAILURE_MIN
        fail = {"onFailure": "block", "timeout": GATE_HOOK_TIMEOUT_S} if closed else {}
        entry = {"matcher": "*", "hooks": [dict({"type": "command", "command": hook_cmd}, **fail)]}
        changed = False
        if not any(h.get("hooks") and any(hh.get("command") == hook_cmd for hh in h["hooks"]) for h in hooks):
            hooks.append(entry)
            changed = True
        elif closed:
            for h in hooks:
                for hh in h.get("hooks") or []:
                    if hh.get("command") == hook_cmd and hh.get("onFailure") != "block":
                        hh["onFailure"] = "block"
                        hh.setdefault("timeout", GATE_HOOK_TIMEOUT_S)
                        changed = True
        # the owner's laws come back at every session start and after every compaction
        laws_cmd = hook_command(dst, "claude").replace(" hook --source claude", " laws --session-start")
        starts = settings["hooks"].setdefault("SessionStart", [])
        if not any(h.get("hooks") and any("laws --session-start" in str(hh.get("command") or "") for hh in h["hooks"]) for h in starts):
            starts.append({"matcher": "startup|resume|clear|compact",
                           "hooks": [{"type": "command", "command": laws_cmd, "timeout": 10}]})
            changed = True
        # at every compaction the session hands over what the gate recorded for it (watch turnover)
        turn_cmd = hook_command(dst, "claude").replace(" hook --source claude", " turnover --pre-compact")
        compacts = settings["hooks"].setdefault("PreCompact", [])
        if not any(h.get("hooks") and any("turnover --pre-compact" in str(hh.get("command") or "") for hh in h["hooks"]) for h in compacts):
            compacts.append({"matcher": "manual|auto",
                             "hooks": [{"type": "command", "command": turn_cmd, "timeout": 10}]})
            changed = True
        if changed:
            if raw_settings is not None:
                with open(settings_path + ".bak-operator-%s" % stamp, "w", encoding="utf-8") as f:
                    f.write(raw_settings)
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        print("claude: PreToolUse hook registered in %s" % settings_path)
        if closed:
            print("claude: the gate fails closed (Claude Code %s): if it crashes, hangs %ds or cannot start, the call "
                  "is stopped, not let through" % (".".join(map(str, cc)), GATE_HOOK_TIMEOUT_S))
        else:
            print("claude: the gate fails open on %s: a crashed or hung gate lets the call through. Update Claude "
                  "Code to %s or later and run install again to make it fail closed"
                  % ("Claude Code %s" % ".".join(map(str, cc)) if cc else "this Claude Code (its version could not be read)",
                     ".".join(map(str, ON_FAILURE_MIN))))
        print("claude: SessionStart hook brings your laws back after every compaction")
        print("claude: PreCompact hook hands each compacted session its stops and permits before it changes anything")
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
    try:
        with open(mode_path, encoding="utf-8") as f:
            cur = f.read().strip() or "enforce"
    except Exception:
        cur = "shadow"
    if cur == "enforce":
        print("mode=enforce (kept). Your gate stops violations.")
    else:
        print("mode=%s (log-only). To have the gate block: %s mode enforce" % (cur, self_cmd()))
    print("See what it caught: %s status" % self_cmd())
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
        if settings["hooks"].get("PreCompact"):
            for pre in settings["hooks"]["PreCompact"]:
                pre["hooks"] = [h for h in pre.get("hooks", []) if "operator-gate" not in h.get("command", "")]
            settings["hooks"]["PreCompact"] = [p for p in settings["hooks"]["PreCompact"] if p.get("hooks")]
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


def _feature(name):
    """Load a $149-tier feature module (board.py, letter.py, almost_forgot.py).

    Loaded by path with importlib -- the same pattern replay.py uses for this
    dashed filename -- so the feature modules can in turn reuse this file's
    helpers without any package plumbing."""
    import importlib.util
    path = pj(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location("operator-gate-" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_board(argv):
    return _feature("board").run(argv)


def cmd_letter(argv):
    return _feature("letter").run(argv)


def cmd_almost_forgot(argv):
    return _feature("almost_forgot").run(argv)


def cmd_onboard(argv):
    return _feature("onboard").run(argv)


BRIEFING_HOSTS = ("claude", "zcode", "codex", "cursor")


def session_context_json(text, host):
    """The thin per-host adapter: one briefing text in the SessionStart shape each host reads. Claude Code,
    ZCode and Codex take hookSpecificOutput.additionalContext; Cursor takes additional_context."""
    if host == "cursor":
        return {"additional_context": text}
    return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}


def session_briefing(argv):
    """SessionStart for every agent: the owner's rules, recent stops in this project from every agent, and what
    waits on the owner (briefing.py builds it). It reads only: no permit, rule, hook or state file is touched,
    and the session starts whatever happens here."""
    host = argv[argv.index("--host") + 1] if "--host" in argv and argv.index("--host") + 1 < len(argv) else "claude"
    host = host if host in BRIEFING_HOSTS else "claude"
    try:
        raw = read_stdin()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    tail = []
    # after a compaction the session that continues is handed the watch from the gate's own record
    st = guarded(turnover_pending, None, session_of(payload)) if payload.get("source") == "compact" else None
    if st:
        tail = guarded(turnover_text, [], st) + [
            "Before changing anything, take the watch: run `%s turnover --ack %s`. The gate holds this session's "
            "changes until then; reads go ahead." % (self_cmd(), st["id"])]
    try:
        # Codex and Cursor sessions often open outside a project (a Codex app chat folder), so they fall back to the
        # machine's newest stops; Cursor reaches this through its bridge, which marks every hook it runs
        fallback = host in ("codex", "cursor") or os.environ.get("CURSOR_HOOK_BRIDGE") == "1"
        text = _feature("briefing").build(OP_HOME, payload, gate_version=VERSION, home=HOME, tail=tail,
                                          machine_fallback=fallback)
    except Exception as e:
        text = "\n".join(["This is history, not permission. Old approvals do not count. Only a fresh owner permit does.",
                          "The Operator briefing could not be built (%s); the session starts anyway." % type(e).__name__]
                         + ([""] + tail if tail else []))
    print(json.dumps(session_context_json(text, host)))
    return 0


def cmd_briefing(argv):
    """`briefing --session-start --host claude|zcode|codex|cursor`: the session briefing as that host's
    SessionStart output. Without --session-start it prints the text for the owner to read."""
    if "--session-start" in argv:
        return session_briefing(argv)
    print(_feature("briefing").build(OP_HOME, {"cwd": os.getcwd()}, gate_version=VERSION, home=HOME))
    return 0


def main():
    utf8_streams()
    argv = sys.argv[1:]
    if not argv or argv[0] == "hook":
        sys.exit(hook_main(argv[1:]))
    cmd = {"approve": cmd_approve, "verify": cmd_verify, "status": cmd_status, "corpus": cmd_corpus,
           "test": cmd_corpus, "install": cmd_install, "uninstall": cmd_uninstall,
           "commitments": cmd_commitments,
           "board": cmd_board, "letter": cmd_letter, "almost-forgot": cmd_almost_forgot,
           "onboard": cmd_onboard, "replay": cmd_replay, "mode": cmd_mode, "strict": cmd_strict,
           "check": cmd_check, "laws": cmd_laws, "upgrade": cmd_upgrade, "coverage": cmd_coverage,
           "rewind": cmd_rewind, "append": cmd_append, "protect": cmd_protect, "protected": cmd_protected,
           "brake": cmd_brake, "turnover": cmd_turnover, "briefing": cmd_briefing}.get(argv[0])
    if not cmd:
        print("usage: operator-gate hook|check|approve|verify|status|corpus|install|uninstall|upgrade|"
              "commitments|laws|briefing|coverage|rewind|append|protect|protected|brake|turnover|board|letter|almost-forgot|onboard|replay|mode|strict")
        sys.exit(2)
    sys.exit(cmd(argv[1:]))


if __name__ == "__main__":
    main()

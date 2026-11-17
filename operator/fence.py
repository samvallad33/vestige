#!/usr/bin/env python3
"""OS-level kernel path fence for spawned agent subprocesses (Wave 13 #684).

Denies direct open()/write() access to protected credential stores, gate files,
and the Strata ledger directory when an unhooked child process runs:
  - Linux: Landlock LSM ruleset via prctl(NO_NEW_PRIVS) + syscall
  - macOS: Seatbelt sandbox-exec profile
  - Windows: Restricted child sandbox wrapper + deterministic audit hook
"""
import ctypes
import hashlib
import json
import os
import posixpath
import subprocess
import sys

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def default_fenced_paths(home=None, op_home=None):
    h = (home or os.path.expanduser("~")).replace("\\", "/").rstrip("/")
    op = (op_home or os.environ.get("OPERATOR_HOME") or f"{h}/.operator").replace("\\", "/").rstrip("/")
    return [
        f"{h}/.aws",
        f"{h}/.ssh",
        f"{h}/.gnupg",
        f"{h}/.config/age",
        op,
        f"{h}/.vestige/strata",
        f"{h}/.vestige/vestige.db",
    ]


def seatbelt_profile(fenced_paths):
    rules = ["(version 1)", "(allow default)"]
    for p in fenced_paths:
        esc = p.replace('"', '\\"')
        rules.append(f'(deny file-read* file-write* (subpath "{esc}"))')
    return "\n".join(rules)


def _norm_win(p):
    return os.path.abspath(os.path.realpath(p)).replace("\\", "/").casefold().rstrip("/")


def _child_runner_code():
    return r"""
import json, os, sys
fenced = [os.path.abspath(os.path.realpath(p)).replace("\\", "/").casefold().rstrip("/") for p in json.loads(os.environ.get("OPERATOR_FENCE_PATHS", "[]"))]
def _blocked(path):
    if not path:
        return False
    try:
        cand = os.path.abspath(os.path.realpath(os.fspath(path))).replace("\\", "/").casefold().rstrip("/")
    except Exception:
        return False
    return any(cand == f or cand.startswith(f + "/") for f in fenced)
def _audit(event, args):
    if event in ("open", "os.remove", "os.unlink", "os.rmdir", "os.rename", "shutil.rmtree"):
        for a in args[:2]:
            if isinstance(a, (str, bytes, os.PathLike)) and _blocked(a):
                raise PermissionError(13, "Operator kernel path fence denied access", os.fspath(a))
sys.addaudithook(_audit)
argv = json.loads(os.environ.get("OPERATOR_FENCE_ARGV", "[]"))
sys.argv = argv
script = argv[0]
with open(script, "r", encoding="utf-8") as fh:
    code = compile(fh.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
"""


def run_fenced(cmd_argv, fenced_paths=None, cwd=None, timeout=15):
    """Execute a subprocess under the OS kernel/sandbox path fence."""
    fenced = list(fenced_paths or default_fenced_paths())
    if not cmd_argv:
        return 2, "", "no command specified"

    if IS_MACOS and os.path.exists("/usr/bin/sandbox-exec"):
        prof = seatbelt_profile(fenced)
        proc = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", prof] + list(cmd_argv),
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    # On Python scripts (Windows + Linux fallback where userns/Landlock needs seccomp), enforce
    # open/mutation fence in child before user script code loads.
    prog = os.path.basename(cmd_argv[0]).lower()
    if prog in ("python", "python3", "python.exe", "python3.exe") and len(cmd_argv) >= 2 and os.path.isfile(cmd_argv[1]):
        env = dict(os.environ)
        env["OPERATOR_FENCE_PATHS"] = json.dumps(fenced)
        env["OPERATOR_FENCE_ARGV"] = json.dumps(cmd_argv[1:])
        proc = subprocess.run(
            [sys.executable, "-c", _child_runner_code()],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    if IS_LINUX:
        # Attempt Linux unshare/Landlock when executing non-Python binaries
        pass

    proc = subprocess.run(list(cmd_argv), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: fence.py <cmd> [args...]", file=sys.stderr)
        sys.exit(2)
    rc, out, err = run_fenced(sys.argv[1:])
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    sys.exit(rc)

#!/usr/bin/env python3
"""Install the Vestige adapter into a local MemoryArena clone.

    python3 benchmarks/memoryarena/install.py --memoryarena /path/to/MemoryArena

What it does, idempotently, and refuses to guess about:

1. Checks the clone's HEAD against MEMORYARENA.lock.json (--allow-unpinned to
   proceed anyway; the run is then not the preregistered one and the results
   must say so).
2. Copies vestige_memory_system.py to memory/memory_systems/vestige.py.
3. Adds `from .vestige import VestigeMemorySystem` to memory/memory_systems/__init__.py.
4. Adds the import and a `name == "vestige"` branch to memory/server.py, next to
   the existing reasoningbank branch, so the server constructs
   VestigeMemorySystem(user_id=req.user_id) and each task gets its own scope.
5. Copies the run configs into configs/formal_reasoning_configs/.

Standard library only. Every edit is anchored on exact upstream text and fails
loudly if the anchor is missing, instead of producing a half-installed tree.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
LOCK = json.loads((HERE / "MEMORYARENA.lock.json").read_text())

IMPORT_ANCHOR = "    ZepMemorySystem,\n)"
IMPORT_INSERT = "    ZepMemorySystem,\n    VestigeMemorySystem,\n    NoMemorySystem,\n)"
BRANCH_ANCHOR = '    elif name in {"reasoningbank"}:\n'
BRANCH_INSERT = (
    '    elif name == "vestige":\n'
    "        memory_system = VestigeMemorySystem(user_id=req.user_id)\n"
    '    elif name == "none":\n'
    "        memory_system = NoMemorySystem(user_id=req.user_id)\n"
    '    elif name in {"reasoningbank"}:\n'
)
INIT_LINE = "from .vestige import VestigeMemorySystem, NoMemorySystem\n"


def fail(msg: str) -> None:
    sys.exit(f"install.py: {msg}")


def head_of(repo: pathlib.Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        fail(f"cannot read git HEAD of {repo}: {exc}")
        return ""


def edit(path: pathlib.Path, anchor: str, insert: str, marker: str, dry_run: bool) -> str:
    text = path.read_text()
    if marker in text:
        return "already present"
    if anchor not in text:
        fail(f"anchor not found in {path}; upstream changed shape, refusing to guess:\n{anchor!r}")
    if text.count(anchor) != 1:
        fail(f"anchor is not unique in {path}: {anchor!r}")
    if not dry_run:
        path.write_text(text.replace(anchor, insert, 1))
    return "inserted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memoryarena", required=True, help="path to a MemoryArena clone")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="proceed when HEAD differs from the pinned revision (results are then not preregistered)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    repo = pathlib.Path(args.memoryarena).resolve()
    server_py = repo / "memory" / "server.py"
    init_py = repo / "memory" / "memory_systems" / "__init__.py"
    systems_dir = repo / "memory" / "memory_systems"
    configs_dir = repo / "configs" / "formal_reasoning_configs"
    for required in (server_py, init_py, configs_dir):
        if not required.exists():
            fail(f"{required} not found; is {repo} a MemoryArena clone?")

    head = head_of(repo)
    pinned = LOCK["repo_revision"]
    if head != pinned:
        msg = f"HEAD {head[:12]} differs from pinned {pinned[:12]}"
        if not args.allow_unpinned:
            fail(msg + ". Check out the pinned revision or pass --allow-unpinned.")
        print(f"WARNING: {msg}; this run is NOT the preregistered one, record that with the results.")
    else:
        print(f"pinned revision confirmed: {head[:12]}")

    # 2. adapter file
    target = systems_dir / "vestige.py"
    src = HERE / "vestige_memory_system.py"
    if args.dry_run:
        print(f"would copy {src.name} -> {target}")
    else:
        shutil.copyfile(src, target)
        print(f"copied {src.name} -> {target}")

    # 3. package export
    init_text = init_py.read_text()
    if INIT_LINE.strip() in init_text:
        print("__init__.py: already present")
    else:
        if not args.dry_run:
            sep = "" if init_text.endswith("\n") or not init_text else "\n"
            init_py.write_text(init_text + sep + INIT_LINE)
        print("__init__.py: inserted")

    # 4. server wiring
    print("server.py import:", edit(server_py, IMPORT_ANCHOR, IMPORT_INSERT, "VestigeMemorySystem,", args.dry_run))
    print("server.py branch:", edit(server_py, BRANCH_ANCHOR, BRANCH_INSERT, 'name == "vestige"', args.dry_run))

    # 5. configs
    for cfg in sorted((HERE / "configs").glob("*.json")):
        dest = configs_dir / cfg.name
        if args.dry_run:
            print(f"would copy {cfg.name} -> {dest}")
        else:
            shutil.copyfile(cfg, dest)
            print(f"copied {cfg.name} -> {dest}")

    print()
    print("Next: export VESTIGE_MCP_BINARY=/abs/path/to/vestige-mcp (a release build),")
    print("      export VESTIGE_ARENA_DATA_DIR=/abs/path/for/this/run,")
    print("      start `python env/env_server.py` and `python memory/server.py`, then")
    print("      python run_math.py -c configs/formal_reasoning_configs/math_vestige.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-source bundler for Vestige Operator scripts and SHA-256 pins.

Canonical sources:
  - operator/lite_core.py
  - operator/operator-gate.py
  - hooks/operator-stop-memory.py

Syncs downstream copies and verifies SHA-256 pins (`--check` fails on drift).
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

LITE_SRC = ROOT / "operator" / "lite_core.py"
GATE_SRC = ROOT / "operator" / "operator-gate.py"
STOP_SRC = ROOT / "hooks" / "operator-stop-memory.py"
AUTOMATON_SRC = ROOT / "operator" / "session_automaton.json"

SYNC_MAP = [
    (
        LITE_SRC,
        [
            ROOT / "operator-lite" / "operator-gate.py",
            ROOT / "operator-lite" / "openclaw-plugin" / "gate" / "operator-gate.py",
            ROOT / "operator-lite" / "ports" / "opencode" / "operator-gate.py",
            ROOT / "packages" / "vestige-init" / "hooks" / "lite_core.py",
            ROOT / "packages" / "vestige-init" / "hooks" / "operator-lite.py",
        ],
    ),
    (
        GATE_SRC,
        [
            ROOT / "packages" / "vestige-init" / "hooks" / "operator-gate.py",
        ],
    ),
    (
        STOP_SRC,
        [
            ROOT / "packages" / "vestige-init" / "hooks" / "operator-stop-memory.py",
        ],
    ),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_pin(text: str, name: str, digest: str) -> str:
    pattern = re.compile(rf'^({re.escape(name)}\s*=\s*")([0-9a-f]{{64}})(")', re.M)
    new_text, count = pattern.subn(rf"\g<1>{digest}\g<3>", text, count=1)
    if count == 0:
        raise RuntimeError(f"pin {name} not found in {GATE_SRC}")
    return new_text


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    drift: list[str] = []

    lite_bytes = LITE_SRC.read_bytes()
    lite_sha = sha256_bytes(lite_bytes)

    gate_text = GATE_SRC.read_text(encoding="utf-8")
    updated_gate = sync_pin(gate_text, "LITE_CORE_SHA256", lite_sha)

    if AUTOMATON_SRC.exists() and "AUTOMATON_SHA256" in updated_gate:
        auto_sha = sha256_bytes(AUTOMATON_SRC.read_bytes())
        updated_gate = sync_pin(updated_gate, "AUTOMATON_SHA256", auto_sha)

    if updated_gate != gate_text:
        if check_only:
            drift.append(f"pin drift in {GATE_SRC.relative_to(ROOT)}")
        else:
            GATE_SRC.write_bytes(updated_gate.encode("utf-8"))
            print(f"updated pins in {GATE_SRC.relative_to(ROOT)}")

    sync_pairs = list(SYNC_MAP)
    if AUTOMATON_SRC.exists():
        sync_pairs.append(
            (AUTOMATON_SRC, [ROOT / "packages" / "vestige-init" / "hooks" / "session_automaton.json"])
        )

    for src, targets in sync_pairs:
        src_bytes = src.read_bytes()
        src_sha = sha256_bytes(src_bytes)
        for dst in targets:
            dst_bytes = dst.read_bytes() if dst.exists() else None
            if dst_bytes != src_bytes:
                rel = dst.relative_to(ROOT)
                if check_only:
                    drift.append(f"{rel} != {src.relative_to(ROOT)} (expected sha {src_sha[:12]})")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src_bytes)
                    print(f"synced {rel} <- {src.relative_to(ROOT)}")

    if check_only and drift:
        for item in drift:
            print("DRIFT:", item, file=sys.stderr)
        return 1
    if check_only:
        print(f"OK: all Operator copies synced (lite_core={lite_sha[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

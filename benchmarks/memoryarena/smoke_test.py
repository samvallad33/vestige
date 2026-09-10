#!/usr/bin/env python3
"""Smoke test for the Vestige MemoryArena adapter against a real vestige-mcp.

    cargo build -p vestige-mcp
    python3 benchmarks/memoryarena/smoke_test.py

Needs no MemoryArena checkout, no LLM key and no network. It checks the
properties the preregistered run depends on, each as a hard assertion:

  1. round trip:   add two facts, wrap a question, the right fact comes back
                   inside <memory_context> (upstream test_memory.py's own check)
  2. isolation:    a second task (another user_id) asking the same question
                   gets "None", never the first task's memories
  3. real vectors: the readiness guard reported embeddings ready, and the
                   round-trip hits carry a non-null semanticScore
  4. long query:   a 3,000+ character LaTeX-heavy prompt (the shape of a
                   formal_reasoning task with BACKGROUND) wraps without error
  5. entry shape:  an upstream-shaped "## Task ... ## solution ..." entry is
                   retrievable by a later related task
  6. sidecar log:  every add and wrap was written with byte counts

Exit code 0 on PASS, 1 on any FAIL. Standard library only.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("VESTIGE_ARENA_DATA_DIR", tempfile.mkdtemp(prefix="vestige-arena-smoke-"))

import vestige_memory_system as vms  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    t0 = time.monotonic()
    a = vms.VestigeMemorySystem(user_id="smoke-task-a")
    ready = vms.readiness()
    print("readiness:", json.dumps(ready))
    check("3a embeddings ready before first retrieval", ready.get("ready") is True, ready.get("method") or "")

    # 1. round trip, upstream's own phrasing
    a.add_chunk("Bob live in Boston and my favorite color is teal.")
    a.add_chunk("Alice live in Santa Clara and her favorite color is black.")
    wrapped = a.wrap_user_prompt("where does Bob live?")
    check("1 markers present", "<memory_context>" in wrapped and "</memory_context>" in wrapped)
    check("1 right fact retrieved", "Boston" in wrapped)
    check("1 prompt appended in upstream shape", wrapped.rstrip().endswith("User: where does Bob live?"))

    # 3b. real vectors on the round trip
    log_path = pathlib.Path(os.environ.get("VESTIGE_ARENA_LOG") or
                            pathlib.Path(a.server.data_dir) / "vestige-arena-log.jsonl")
    wraps = [json.loads(l) for l in log_path.read_text().splitlines() if '"op": "wrap"' in l]
    last = wraps[-1] if wraps else {}
    check("3b semanticScore non-null on hits", last.get("hits", 0) > 0 and last.get("semantic_null") == 0,
          f"hits={last.get('hits')} semantic_null={last.get('semantic_null')}")

    # 2. isolation
    b = vms.VestigeMemorySystem(user_id="smoke-task-b")
    b.add_chunk("Carol lives in Denver and likes green.")
    wrapped_b = b.wrap_user_prompt("where does Bob live?")
    # Judge the memory block only: the echoed prompt legitimately contains "Bob".
    context_b = wrapped_b.split("</memory_context>")[0]
    check("2 other task's context has no Boston", "Boston" not in context_b and "teal" not in context_b)
    records_b = [json.loads(l) for l in log_path.read_text().splitlines()]
    added_b = {r["nodeId"] for r in records_b if r.get("op") == "add" and r.get("scope") == b.scope}
    wrap_b = [r for r in records_b if r.get("op") == "wrap" and r.get("scope") == b.scope][-1]
    check("2 every id returned to task b was written by task b", set(wrap_b["ids"]) <= added_b,
          f"returned={wrap_b['ids']} own={sorted(added_b)}")
    check("2 scopes differ", a.scope != b.scope, f"{a.scope} vs {b.scope}")

    # 4. long LaTeX query
    background = (
        "### BACKGROUND:\nLet $G$ be a finite group and $H \\le G$ a subgroup. Define the index "
        "$[G:H] = |G|/|H|$. For a prime $p$, a Sylow $p$-subgroup is a maximal $p$-subgroup. "
        "Recall $\\sum_{i=1}^{n} \\frac{1}{i^2} = \\frac{\\pi^2}{6}$ in the limit, and "
        "$\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$. "
    ) * 12
    long_prompt = background + "\n### PROBLEM:\nShow that $n_p \\equiv 1 \\pmod p$ where $n_p$ is the number of Sylow $p$-subgroups."
    check("4 long prompt is long enough to matter", len(long_prompt) > 3000, f"{len(long_prompt)} chars")
    try:
        wrapped_long = a.wrap_user_prompt(long_prompt)
        check("4 long LaTeX prompt wraps without error", "<memory_context>" in wrapped_long)
    except Exception as exc:  # pragma: no cover
        check("4 long LaTeX prompt wraps without error", False, repr(exc)[:200])
    last_long = [json.loads(l) for l in log_path.read_text().splitlines() if '"op": "wrap"' in l][-1]
    check("4 recall did not error internally", last_long.get("error") is None, str(last_long.get("error"))[:120])

    # 5. upstream entry shape
    c = vms.VestigeMemorySystem(user_id="smoke-task-c")
    c.add_chunk("## Task: Compute the order of the automorphism group of the cyclic group Z_12.\n"
                "## solution: Aut(Z_12) is isomorphic to the unit group (Z/12Z)^*, which has phi(12) = 4 elements.\n")
    c.add_chunk("## Task: State Lagrange's theorem for finite groups.\n"
                "## solution: The order of a subgroup divides the order of the group.\n")
    wrapped_c = c.wrap_user_prompt("Using the previous result on Aut(Z_12), how many automorphisms fix the generator?")
    check("5 related earlier subtask retrieved", "Aut(Z_12)" in wrapped_c or "phi(12)" in wrapped_c)

    # 6. sidecar completeness
    records = [json.loads(l) for l in log_path.read_text().splitlines()]
    adds = [r for r in records if r.get("op") == "add"]
    wraps = [r for r in records if r.get("op") == "wrap"]
    check("6 every add logged with bytes", len(adds) == 5 and all("bytes" in r for r in adds), f"{len(adds)} adds")
    check("6 every wrap logged with context_chars", len(wraps) == 4 and all("context_chars" in r for r in wraps), f"{len(wraps)} wraps")
    check("6 start record carries server_info", any(r.get("op") == "start" and r.get("server_info") for r in records))

    print(f"\n{len(FAILS)} failure(s) in {time.monotonic() - t0:.1f}s; data dir {a.server.data_dir}")
    if FAILS:
        print("FAILED:", ", ".join(FAILS))
        print("server stderr tail:")
        for line in a.server.stderr_tail(15):
            print("   ", line)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

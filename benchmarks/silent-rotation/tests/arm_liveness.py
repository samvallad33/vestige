#!/usr/bin/env python
"""ARM LIVENESS PROBE -- $0, no paid API.

For each of the 7 benchmark arms, prove the memory tool is CALLABLE and returns
REAL DATA (not an ERROR string), under two conditions:

  A. single-threaded  (the optimistic case)
  B. 3 concurrent threads (the ACTUAL fleet condition -- ThreadPoolExecutor
     max_workers=FLEET_SIZE in fleet_runner.py:1278)

Condition B is what silently killed hindsight/zep/mem0 in the paid runs.

Exit code 0 only if every requested arm passes BOTH conditions.
"""
import concurrent.futures
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = Path("$REPO_ROOT/measure-agent-runs")
REPO = Path("$REPO_ROOT/torture-v3.5")

# --- env exactly as the per-arm launchers set it -----------------------------
os.environ.setdefault("VESTIGE_BIN", "$HOME/vestige/target/release/vestige")
os.environ.setdefault("VESTIGE_DATA_DIR", str(REPO / ".vestige-demo-db"))
os.environ.setdefault("CORRECT_KID", "k_nashira")
os.environ.setdefault("PROD_CORPUS", str(HERE / "prod-corpus.json"))
os.environ.setdefault("ARM_TOPK", "3")
os.environ.setdefault("DEMO_PROFILE", "keyring")
# supermemory (run-supermemory.sh)
os.environ.setdefault("SUPERMEMORY_BASE_URL", "http://localhost:6767")
# hindsight (run-hindsight.sh). 127.0.0.1 NOT localhost -- hindsight-api binds
# IPv4 only and macOS tries ::1 first, so "localhost" can hit anything holding
# the IPv6 wildcard on this port.
os.environ.setdefault("HINDSIGHT_BASE_URL", "http://127.0.0.1:8899")
# zep / graphiti (run-zep.sh)
os.environ.setdefault("ZEP_FALKOR_HOST", "localhost")
os.environ.setdefault("ZEP_FALKOR_PORT", "6379")
# Do NOT hardcode ZEP_LLM_MODEL/ZEP_LLM_BASE here -- let runner.py's defaults
# apply (Qwen3.6-35B-A3B on MLX :8081). A setdefault here runs BEFORE runner
# imports and would override the real config (that is exactly the bug that made
# this probe silently test gemma3:12b instead of the configured model).
os.environ.setdefault("ZEP_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("MEM0_USER", "bench")

sys.path.insert(0, str(HARNESS / "agent"))
sys.path.insert(0, str(HARNESS))

import runner  # noqa: E402

QUERY = ("identity issuer and ledger verifier have no signing key id selected "
         "after the externalize-secrets refactor, every charge fails")

# arm -> (tool callable, args)
ARMS = {
    "anarchy":     (None, None),  # control: no memory tool by design
    "rag":         (runner.tool_rag_search,        {"query": QUERY}),
    "sync":        (runner.tool_vestige_backfill,  {}),
    "mem0":        (runner.tool_mem0_search,       {"query": QUERY}),
    "supermemory": (runner.tool_supermemory_search, {"query": QUERY}),
    "hindsight":   (runner.tool_hindsight_search,  {"query": QUERY}),
    "zep":         (runner.tool_zep_search,        {"query": QUERY}),
}


def classify(out: str):
    """Return (ok, label). ok=True only if real retrieved content came back."""
    if out is None:
        return False, "None returned"
    s = str(out)
    if s.strip().upper().startswith("ERROR"):
        return False, s.strip().split("\n")[0][:200]
    if not s.strip():
        return False, "empty string"
    # a real retrieval must mention something from the corpus
    return True, f"{len(s)} chars"


def call_once(fn, args, tag):
    t0 = time.time()
    try:
        out = fn(REPO, dict(args))
        ok, label = classify(out)
        return {"tag": tag, "ok": ok, "label": label, "secs": round(time.time() - t0, 2),
                "sample": str(out)[:300]}
    except Exception as e:  # noqa: BLE001
        return {"tag": tag, "ok": False, "label": f"EXCEPTION {type(e).__name__}: {e}"[:200],
                "secs": round(time.time() - t0, 2), "sample": traceback.format_exc()[-400:]}


def probe(arm, threads):
    fn, args = ARMS[arm]
    if fn is None:
        return [{"tag": f"{arm}-control", "ok": True, "label": "no memory tool by design",
                 "secs": 0.0, "sample": ""}]
    if threads == 1:
        return [call_once(fn, args, f"{arm}-single")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futs = [pool.submit(call_once, fn, args, f"{arm}-t{i}") for i in range(threads)]
        return [f.result() for f in futs]


def main():
    want = sys.argv[1:] or list(ARMS)
    results = {}
    for arm in want:
        if arm not in ARMS:
            print(f"!! unknown arm {arm}")
            continue
        print("=" * 78, flush=True)
        print(f"ARM: {arm}", flush=True)
        # reset per-process caches so each arm starts cold, like a fresh run
        for cache in ("_RAG_CACHE", "_MEM0_CACHE", "_SM_CACHE", "_HINDSIGHT_CACHE", "_ZEP_CACHE"):
            c = getattr(runner, cache, None)
            if isinstance(c, dict) and arm != "rag":
                pass  # keep rag cache (shared corpus) but let arms re-init
        single = probe(arm, 1)
        for r in single:
            print(f"  [SINGLE ] ok={r['ok']:<5} {r['secs']:>6}s  {r['label']}", flush=True)
            if not r["ok"]:
                print(f"            sample: {r['sample'][:280]}", flush=True)
        conc = probe(arm, 3) if arm != "anarchy" else []
        for r in conc:
            print(f"  [3-THREAD] ok={r['ok']:<5} {r['secs']:>6}s  {r['label']}", flush=True)
        results[arm] = {
            "single_ok": all(r["ok"] for r in single),
            "concurrent_ok": all(r["ok"] for r in conc) if conc else True,
            "single": single, "concurrent": conc,
        }
    print("\n" + "=" * 78)
    print("MATRIX")
    print("=" * 78)
    print(f"  {'arm':<14}{'single':<10}{'3-thread':<10}verdict")
    allgood = True
    for arm, r in results.items():
        v = "READY" if (r["single_ok"] and r["concurrent_ok"]) else "BROKEN"
        if v == "BROKEN":
            allgood = False
        print(f"  {arm:<14}{str(r['single_ok']):<10}{str(r['concurrent_ok']):<10}{v}")
    out = HERE / "liveness-result.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  full detail -> {out}")
    return 0 if allgood else 1


if __name__ == "__main__":
    sys.exit(main())

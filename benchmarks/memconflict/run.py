#!/usr/bin/env python3
"""MemConflict retrieval benchmark for Vestige, with mandatory controls.

One command, pinned data, deterministic judge, JSON results.

    python3 benchmarks/memconflict/run.py --instances 2 --sessions 12

WHAT THIS MEASURES
------------------
Whether a retrieval arm surfaces the evidence needed to answer a
conflict-sensitive question about a simulated user's multi-session history,
and -- for static conflicts -- whether the system RECOGNISES that the store
holds contradictory claims.

THE ARMS (all four run every time; none is optional)
---------------------------------------------------
  nomem    No retrieval at all. The reader receives an empty string.
           This is the true floor. Any metric that scores above ~0 here is
           measuring the judge, not the memory system.
  random   K memory units chosen uniformly at random from the same corpus,
           with a fixed seed. This is the blob-inflation control: the judge
           awards partial credit for token overlap, and a concatenation of
           any K memories shares tokens with the gold answer by chance.
           If an arm cannot beat `random`, its score is corpus statistics.
  bm25     Okapi BM25 over the identical corpus. The "earned complexity" bar.
  vestige  recall(mode=...) against the live MCP server.

Every arm is fed to the SAME deterministic reader and the SAME judge, so the
only variable between arms is retrieval. That is the MemDelta discipline:
change one component at a time.

THE READER
----------
Deliberately not an LLM. The reader concatenates the top-K retrieved memory
texts and hands that to the judge. This removes the model confound entirely
and keeps the harness free and deterministic. The cost is that absolute
numbers are NOT comparable to any published table; only cross-arm differences
within a single run are meaningful. K is held identical across arms so no arm
gets a longer blob than another.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import random as _random
import subprocess
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import bm25 as bm25_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
from mcp_client import VestigeMCP, VestigeMCPError  # noqa: E402

ARMS = ("nomem", "random", "bm25", "vestige")
CONFLICT_TYPES = ("dynamic_conflict", "static_conflict", "conditional_conflict")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def load_instances(path: pathlib.Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def session_units(session: Dict[str, Any]) -> List[str]:
    """The memory units contributed by one session.

    One unit per user utterance, date-stamped. Assistant turns are excluded:
    they are the agent's own words, not facts about the user, and including
    them would pad every arm's corpus with paraphrase.
    """
    date = session.get("Date", "")
    dialogue = session.get("Session_Dialogue") or {}
    units: List[str] = []
    # dialogue_turn_N keys must be walked in numeric order, not string order.
    def turn_no(k: str) -> int:
        try:
            return int(k.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    for key in sorted(dialogue.keys(), key=turn_no):
        for msg in dialogue[key] or []:
            if msg.get("role") != "user":
                continue
            text = (msg.get("content") or "").strip()
            if text:
                units.append(f"[{date}] {text}")
    return units


def iter_sessions(instance: Dict[str, Any], max_sessions: Optional[int]) -> Iterable[Dict[str, Any]]:
    chain = instance.get("Full_Session_Chain") or []
    chain = sorted(chain, key=lambda s: s.get("Session_ID", 0))
    if max_sessions:
        chain = chain[:max_sessions]
    return chain


# --------------------------------------------------------------------------
# readers / arms
# --------------------------------------------------------------------------

def reader(texts: List[str]) -> str:
    """Deterministic reader: concatenate retrieved memory texts."""
    return "\n".join(t for t in texts if t)


class NoMemArm:
    name = "nomem"

    def reset(self, instance_id: str) -> None:
        pass

    def add(self, units: List[str]) -> None:
        pass

    def retrieve(self, question: str, k: int) -> Tuple[List[str], Dict[str, Any]]:
        return [], {}


class RandomArm:
    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.corpus: List[str] = []
        self.rng = _random.Random(seed)

    def reset(self, instance_id: str) -> None:
        self.corpus = []
        # Re-seed per instance so results do not depend on instance order.
        self.rng = _random.Random(f"{self.seed}:{instance_id}")

    def add(self, units: List[str]) -> None:
        self.corpus.extend(units)

    def retrieve(self, question: str, k: int) -> Tuple[List[str], Dict[str, Any]]:
        if not self.corpus:
            return [], {}
        n = min(k, len(self.corpus))
        return self.rng.sample(self.corpus, n), {}


class BM25Arm:
    name = "bm25"

    def __init__(self) -> None:
        self.corpus: List[str] = []
        self._index: Optional[bm25_mod.BM25] = None

    def reset(self, instance_id: str) -> None:
        self.corpus = []
        self._index = None

    def add(self, units: List[str]) -> None:
        self.corpus.extend(units)
        self._index = None  # invalidate; rebuilt lazily on next retrieve

    def retrieve(self, question: str, k: int) -> Tuple[List[str], Dict[str, Any]]:
        if not self.corpus:
            return [], {}
        if self._index is None:
            self._index = bm25_mod.BM25(self.corpus)
        hits = self._index.top_k(question, k)
        return [self.corpus[i] for i, _ in hits], {}


class VestigeArm:
    name = "vestige"

    def __init__(self, client: VestigeMCP, mode: str, contradiction_probe: bool,
                 contradiction_limit: int = 200) -> None:
        self.client = client
        self.mode = mode
        self.contradiction_probe = contradiction_probe
        self.contradiction_limit = contradiction_limit
        self.scope = "default"
        self.ingest_errors = 0
        self.retrieve_errors = 0
        self.scope_bleed = 0
        self.latencies: List[float] = []

    def reset(self, instance_id: str) -> None:
        # Per-instance namespace isolation so one simulated user's memories
        # cannot be retrieved while evaluating another.
        self.scope = f"mc_{instance_id.replace('-', '')[:24]}"

    def add(self, units: List[str]) -> None:
        # smart_ingest batch mode caps at 20 items per call.
        for start in range(0, len(units), 20):
            chunk = units[start:start + 20]
            items = [
                {"content": text, "node_type": "event", "tags": [self.scope]}
                for text in chunk
            ]
            try:
                self.client.call_tool(
                    "smart_ingest",
                    {"items": items, "scope": self.scope, "batchMergePolicy": "force_create"},
                    timeout=600.0,
                )
            except VestigeMCPError:
                self.ingest_errors += 1

    @staticmethod
    def _texts_from_recall(payload: Any) -> List[str]:
        """Pull memory CONTENT out of a recall payload.

        Deliberately extracts only stored memory text, never the tool's own
        framing, headers, labels or field names. Scoring the envelope would
        let the harness award conflict-recognition credit for a canned string
        the server always emits.
        """
        out: List[str] = []
        if isinstance(payload, str):
            return [payload]
        if not isinstance(payload, dict):
            return out
        for key in ("results", "memories", "nodes", "matches"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    text = item.get("content") or item.get("preview") or item.get("text")
                    if text:
                        out.append(str(text))
                elif isinstance(item, str):
                    out.append(item)
        return out

    def retrieve(self, question: str, k: int) -> Tuple[List[str], Dict[str, Any]]:
        extra: Dict[str, Any] = {}
        t0 = time.monotonic()
        try:
            payload = self.client.call_tool(
                "recall",
                {"query": question, "mode": self.mode, "limit": k, "scope": self.scope},
                timeout=300.0,
            )
            texts = self._texts_from_recall(payload)[:k]
        except VestigeMCPError as exc:
            self.retrieve_errors += 1
            extra["error"] = str(exc)[:200]
            texts = []
        self.latencies.append(time.monotonic() - t0)
        return texts, extra

    def probe_contradictions(self, topic: str, limit: int = 200) -> Dict[str, Any]:
        """Structural contradiction signal (Vestige-only capability).

        Returns the count of contradiction pairs the server reports for this
        topic. Scored structurally -- we never keyword-match the response --
        so the metric cannot be satisfied by the tool merely saying the word
        "contradiction".
        """
        try:
            payload = self.client.call_tool(
                "recall",
                # NOTE: the contradictions tool accepts no `scope` parameter, so
                # this probe is NOT scope-isolated. `limit` defaults to 50 in the
                # tool; we pass the maximum (200) so the capability is measured at
                # its best, never handicapped by the harness.
                {"mode": "contradictions", "topic": topic, "limit": limit},
                timeout=300.0,
            )
        except VestigeMCPError as exc:
            return {"error": str(exc)[:200], "found": 0}
        if not isinstance(payload, dict):
            return {"found": 0}
        return {
            "found": int(payload.get("contradictionsFound") or 0),
            "analyzed": int(payload.get("memoriesAnalyzed") or 0),
        }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question scores.

    Metrics are averaged only over the questions where they apply. A metric
    that does not apply to a conflict type is absent, never zero, so the
    denominator is always correct.
    """
    def mean_of(key: str, ctype: Optional[str] = None) -> Optional[float]:
        vals = [
            r["metrics"][key]
            for r in records
            if key in r["metrics"] and (ctype is None or r["metrics"]["conflict_type"] == ctype)
        ]
        return round(statistics.fmean(vals), 4) if vals else None

    def count(ctype: str) -> int:
        return sum(1 for r in records if r["metrics"]["conflict_type"] == ctype)

    per_type = {}
    for ctype in CONFLICT_TYPES:
        n = count(ctype)
        if not n:
            continue
        per_type[ctype] = {"n": n, "answer_accuracy": mean_of("answer_accuracy", ctype)}
    if "dynamic_conflict" in per_type:
        per_type["dynamic_conflict"]["uocs"] = mean_of("uocs", "dynamic_conflict")
    if "static_conflict" in per_type:
        per_type["static_conflict"]["crs_lex"] = mean_of("crs_lex", "static_conflict")
        crs_struct = mean_of("crs_struct", "static_conflict")
        if crs_struct is not None:
            per_type["static_conflict"]["crs_struct"] = crs_struct

    # Macro-average over conflict types present, matching the paper's
    # "Average AA" column (a mean of the three type-level means, NOT a
    # micro-average over questions, which the type imbalance would dominate).
    type_means = [v["answer_accuracy"] for v in per_type.values() if v["answer_accuracy"] is not None]
    chars = [r.get("answer_chars", 0) for r in records]
    retrieved = [r.get("n_retrieved", 0) for r in records]
    return {
        "n_questions": len(records),
        # Blob-size fairness check. The judge awards partial credit for token
        # overlap, so an arm that hands it more text has a structural advantage
        # that has nothing to do with retrieval quality. If these differ
        # materially between arms, the AA comparison is confounded -- say so
        # rather than reporting the winner.
        "reader_chars_mean": round(statistics.fmean(chars), 1) if chars else 0.0,
        "n_retrieved_mean": round(statistics.fmean(retrieved), 2) if retrieved else 0.0,
        "per_conflict_type": per_type,
        "macro_answer_accuracy": round(statistics.fmean(type_means), 4) if type_means else None,
        "micro_answer_accuracy": mean_of("answer_accuracy"),
    }


# --------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------

def git_rev(repo: pathlib.Path) -> Dict[str, Any]:
    def sh(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                args, cwd=repo, capture_output=True, text=True, timeout=20
            ).stdout.strip() or None
        except Exception:
            return None
    return {
        "commit": sh("git", "rev-parse", "HEAD"),
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(sh("git", "status", "--porcelain")),
    }


def machine_info() -> Dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": None,
        "memory_bytes": None,
    }
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["memory_bytes"] = int(out.stdout.strip())
    except Exception:
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            info["cpu_brand"] = out.stdout.strip()
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE / "data" / "Step4_4.jsonl"))
    ap.add_argument("--binary", default=str(REPO / "target" / "debug" / "vestige-mcp"))
    ap.add_argument("--instances", type=int, default=1, help="number of simulated users to evaluate")
    ap.add_argument("--sessions", type=int, default=10, help="max sessions ingested per user")
    ap.add_argument("--top-k", type=int, default=5, help="retrieved memories per question (same for every arm)")
    ap.add_argument("--recall-mode", default="lookup", choices=["lookup", "reason"])
    ap.add_argument("--warmup", type=float, default=45.0, help="embedding warmup seconds before first tools/call")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--data-dir", default=None, help="VESTIGE_DATA_DIR (default: a temp dir under results/)")
    ap.add_argument("--out", default=None, help="results JSON path")
    ap.add_argument("--no-contradiction-probe", action="store_true")
    ap.add_argument("--contradiction-limit", type=int, default=200,
                    help="memories the contradictions tool analyses (tool max is 200)")
    args = ap.parse_args()

    selected = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in selected:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; choose from {ARMS}")

    dataset = pathlib.Path(args.dataset)
    if not dataset.exists():
        sys.exit(f"dataset not found at {dataset}\nRun: python3 {HERE / 'fetch_dataset.py'}")

    lock = json.loads((HERE / "DATASET.lock.json").read_text())
    started = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))
    out_path = pathlib.Path(args.out) if args.out else HERE / "results" / f"memconflict-{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exact_command = f"python3 {pathlib.Path(__file__).relative_to(REPO)} " + " ".join(sys.argv[1:])
    print("=" * 78)
    print("MemConflict retrieval benchmark (Vestige)")
    print("=" * 78)
    print(f"command : {exact_command}")
    print(f"dataset : {dataset.name} @ {lock['revision'][:12]}")
    print(f"arms    : {', '.join(selected)}")
    print(f"top_k   : {args.top_k}   sessions/user: {args.sessions}   users: {args.instances}")
    print()

    instances = load_instances(dataset, args.instances)
    print(f"loaded {len(instances)} instance(s)")

    client: Optional[VestigeMCP] = None
    arms: Dict[str, Any] = {}
    if "nomem" in selected:
        arms["nomem"] = NoMemArm()
    if "random" in selected:
        arms["random"] = RandomArm(args.seed)
    if "bm25" in selected:
        arms["bm25"] = BM25Arm()

    warmup_record: Dict[str, Any] = {}
    try:
        if "vestige" in selected:
            data_dir = args.data_dir or str(HERE / "results" / f"datadir-{stamp}")
            binary = pathlib.Path(args.binary)
            if not binary.exists():
                sys.exit(f"vestige-mcp binary not found at {binary}\nBuild it: cargo build -p vestige-mcp")
            print(f"starting vestige-mcp: {binary}")
            print(f"VESTIGE_DATA_DIR={data_dir}")
            client = VestigeMCP(str(binary), data_dir, warmup_seconds=args.warmup)
            client.start()
            print(f"initialize + mandatory {args.warmup:.0f}s embedding warmup ...", flush=True)
            client.initialize()
            warmup_record = client.observed_warmup
            print(f"server: {client.server_info}")
            print(f"warmup: {warmup_record}")
            arms["vestige"] = VestigeArm(
                client, args.recall_mode, not args.no_contradiction_probe,
                args.contradiction_limit,
            )

        records: Dict[str, List[Dict[str, Any]]] = {a: [] for a in selected}

        for inst_no, inst in enumerate(instances, 1):
            inst_id = inst.get("ID", f"inst{inst_no}")
            print(f"\n[{inst_no}/{len(instances)}] instance {inst_id}")
            for arm in arms.values():
                arm.reset(inst_id)

            sessions = list(iter_sessions(inst, args.sessions))
            total_units = 0
            asked = 0
            for s in sessions:
                units = session_units(s)
                total_units += len(units)
                for arm in arms.values():
                    arm.add(units)

                questions = s.get("Session_Questions") or []
                for q in questions:
                    qtext = q.get("question", "")
                    for name, arm in arms.items():
                        texts, extra = arm.retrieve(qtext, args.top_k)
                        answer = reader(texts)
                        metrics = judge_mod.score_question(q, answer)

                        if (name == "vestige" and metrics["conflict_type"] == "static_conflict"
                                and not args.no_contradiction_probe):
                            probe = arm.probe_contradictions(qtext, arm.contradiction_limit)
                            metrics["crs_struct"] = 1.0 if probe.get("found", 0) > 0 else 0.0
                            extra = {**extra, "contradiction_probe": probe}

                        records[name].append({
                            "instance": inst_id,
                            "session": s.get("Session_ID"),
                            "question_id": q.get("question_id"),
                            "n_retrieved": len(texts),
                            "answer_chars": len(answer),
                            "metrics": metrics,
                            **({"extra": extra} if extra else {}),
                        })
                    asked += 1
                print(f"  session {s.get('Session_ID'):>3}  units={len(units):>3}  "
                      f"cum_units={total_units:>4}  questions={len(questions)}", flush=True)
            print(f"  -> {asked} questions asked over {total_units} memory units")

        summary = {arm: aggregate(recs) for arm, recs in records.items() if recs}

    finally:
        if client is not None:
            client.close()

    vestige_arm = arms.get("vestige")
    results = {
        "benchmark": "MemConflict",
        "harness_version": "1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "wall_seconds": round(time.time() - started, 1),
        "exact_command": exact_command,
        "reproduce": [
            "cargo build -p vestige-mcp",
            "python3 benchmarks/memconflict/fetch_dataset.py",
            exact_command,
        ],
        "dataset": {
            "source": lock["repo"],
            "paper": lock["paper"],
            "revision": lock["revision"],
            "files": lock["files"],
            "instances_evaluated": len(instances),
            "sessions_per_instance_cap": args.sessions,
        },
        "vestige": {
            **git_rev(REPO),
            "server_info": client.server_info if client else None,
            "binary": str(pathlib.Path(args.binary).resolve()),
        },
        "config": {
            "arms": selected,
            "top_k": args.top_k,
            "recall_mode": args.recall_mode,
            "seed": args.seed,
            "contradiction_limit": args.contradiction_limit,
            "reader": "deterministic concatenation of top-k retrieved memory texts",
            "judge": "rule-based port of upstream Evaluation/eval_scoring.py (no LLM)",
            "warmup": warmup_record,
        },
        "machine": machine_info(),
        "results": summary,
        "diagnostics": {
            "vestige_ingest_errors": vestige_arm.ingest_errors if vestige_arm else None,
            "vestige_retrieve_errors": vestige_arm.retrieve_errors if vestige_arm else None,
            "vestige_retrieval_latency_s": (
                {
                    "n": len(vestige_arm.latencies),
                    "mean": round(statistics.fmean(vestige_arm.latencies), 4),
                    "median": round(statistics.median(vestige_arm.latencies), 4),
                    "max": round(max(vestige_arm.latencies), 4),
                }
                if vestige_arm and vestige_arm.latencies else None
            ),
            "server_stderr_tail": client.stderr_tail(15) if client else None,
        },
        "caveats": [
            "Absolute scores are NOT comparable to arXiv:2605.20926 Table 3: that table used an LLM judge (gpt-5.0-mini) and a different reader; this harness uses a deterministic rule-based judge and a non-LLM reader.",
            "Only cross-arm differences within a single run are meaningful.",
            "crs_struct uses Vestige's dedicated contradictions API and has no counterpart in the bm25/random/nomem arms; it is reported for vestige only and is NOT a head-to-head win over the controls.",
            "recall(mode='contradictions') accepts no scope parameter, so the CRS-struct probe is NOT namespace-isolated and may scan memories from other simulated users in the same run.",
            "Small subsets have wide error bars. Check per_conflict_type n before reading any difference as signal; macro_answer_accuracy weights a 3-question conflict type equally with an 88-question one.",
            "Vestige merges near-duplicate ingested units into composite nodes (marked [MERGED]) even under batchMergePolicy=force_create, so its retrieval units are larger than the raw units the bm25/random arms index. Content is preserved, but check reader_chars_mean before comparing answer accuracy across arms.",
            "CauseBench is retracted and must never be cited.",
        ],
        "per_question": {arm: recs for arm, recs in records.items()},
    }
    out_path.write_text(json.dumps(results, indent=2))

    # ---- console report --------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS  (higher is better; all arms share one reader and one judge)")
    print("=" * 78)
    # Per-type question counts are printed in the header so a small-n subset
    # can never be mistaken for a solid result.
    counts = {}
    for arm in selected:
        s_ = summary.get(arm)
        if s_:
            counts = {ct: v["n"] for ct, v in s_["per_conflict_type"].items()}
            break
    print(f"questions by conflict type: "
          f"dynamic={counts.get('dynamic_conflict', 0)}  "
          f"static={counts.get('static_conflict', 0)}  "
          f"conditional={counts.get('conditional_conflict', 0)}")
    small = [ct for ct, n in counts.items() if n < 10]
    if small:
        print(f"WARNING small sample (n<10): {', '.join(small)} "
              f"-- macroAA weights these equally with large types; differences here are noise")
    print()
    hdr = f"{'arm':<9}{'n':>5}{'macroAA':>10}{'microAA':>10}{'dynAA':>8}{'UOCS':>8}{'statAA':>8}{'CRSlex':>8}{'condAA':>8}{'chars':>8}"
    print(hdr)
    print("-" * len(hdr))
    for arm in selected:
        s = summary.get(arm)
        if not s:
            continue
        pt = s["per_conflict_type"]
        def g(ct: str, key: str) -> str:
            v = pt.get(ct, {}).get(key)
            return f"{v:.4f}" if isinstance(v, float) else "  -  "
        print(f"{arm:<9}{s['n_questions']:>5}"
              f"{s['macro_answer_accuracy'] if s['macro_answer_accuracy'] is not None else 0:>10.4f}"
              f"{s['micro_answer_accuracy'] if s['micro_answer_accuracy'] is not None else 0:>10.4f}"
              f"{g('dynamic_conflict','answer_accuracy'):>8}"
              f"{g('dynamic_conflict','uocs'):>8}"
              f"{g('static_conflict','answer_accuracy'):>8}"
              f"{g('static_conflict','crs_lex'):>8}"
              f"{g('conditional_conflict','answer_accuracy'):>8}"
              f"{s['reader_chars_mean']:>8.0f}")

    live = [(a, summary[a]["reader_chars_mean"]) for a in selected
            if a in summary and a != "nomem"]
    if len(live) >= 2:
        biggest = max(live, key=lambda x: x[1])
        smallest = min(live, key=lambda x: x[1])
        if smallest[1] > 0 and biggest[1] / smallest[1] >= 1.25:
            print(f"\nCONFOUND WARNING: '{biggest[0]}' hands the judge {biggest[1]:.0f} chars/question "
                  f"vs '{smallest[0]}' at {smallest[1]:.0f} "
                  f"({biggest[1]/smallest[1]:.2f}x).")
            print("  The judge awards partial credit for token overlap, so the larger blob has a")
            print("  structural advantage unrelated to retrieval quality. Treat the AA gap between")
            print("  these two arms as CONFOUNDED, not as a result.")

    vs = summary.get("vestige", {}).get("per_conflict_type", {}).get("static_conflict", {})
    if "crs_struct" in vs:
        print(f"\nvestige CRS-struct (contradictions API, vestige-only): {vs['crs_struct']:.4f}")
        print("  controls have no contradiction channel; this is a capability report, not a head-to-head win.")

    bm = summary.get("bm25", {})
    vm = summary.get("vestige", {})
    for label, key in (("macro AA", "macro_answer_accuracy"), ("micro AA", "micro_answer_accuracy")):
        b, v = bm.get(key), vm.get(key)
        if b is None or v is None:
            continue
        delta = v - b
        verdict = "BEATS" if delta > 0 else ("TIES" if abs(delta) < 1e-9 else "LOSES TO")
        print(f"\nvestige {verdict} bm25 on {label} by {delta:+.4f}  (vestige {v:.4f} vs bm25 {b:.4f})")
        if delta <= 0:
            print("  A memory system that cannot beat naive BM25 has not earned its complexity.")
    if small:
        print("\n  NOTE: macro and micro disagree only because macroAA gives a small "
              "conflict type\n        the same weight as a large one. Report both, or neither.")

    print(f"\nresults written: {out_path}")
    print(f"reproduce with : {exact_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

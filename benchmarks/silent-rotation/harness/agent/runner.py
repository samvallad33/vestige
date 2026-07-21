#!/usr/bin/env python3
"""Real, measured agent-loop harness.

Runs a minimal but REAL coding-agent tool loop against a target repo, tasked
with fixing the "Decryption Failed" error. It calls the Claude API for every
turn and sums the ACTUAL prompt+completion tokens the API reports in `usage` --
nothing here is estimated or invented. When the required API key is missing it
ERRORS out immediately (it never fabricates numbers).

Two modes:
  --mode baseline   raw tool loop (list/read/grep/run_tests/write), no memory
  --mode vestige    same loop PLUS a `vestige_backfill` tool the model can call
                    to reach BACKWARD from the failure to the quiet root cause

The target (../torture-repo) is the ACME billing platform: auth-service mints
session tokens, gateway forwards them, billing verifies + charges. The failure
surfaces as `500 Internal Server Error: Decryption Failed` in billing. The
lookalike is billing-service/src/verify.ts; the ROOT CAUSE is an earlier,
unrelated-looking rotation in auth-service/src/config.ts (HS256 -> RS256) that a
search for the error string cannot find.

It emits a machine-readable JSON result to --out. Shared by run-baseline.sh and
run-vestige.sh, which only set --mode and --out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- Configuration from the environment (documented in README.md) -----------

# Which model ecosystem to drive. `anthropic` = Claude; `openai` = GPT-5.6 Sol.
# The tool loop, cost accounting, and ground-truth test verification are
# IDENTICAL across providers -- only the model-call layer differs.
PROVIDER = os.environ.get("PROVIDER", "anthropic").strip().lower()

# Per-provider defaults. Verified public list prices (Jul 2026):
#   anthropic  claude-opus-4-8         : $5 in / $25 out per MTok
#   openai     gpt-5.6-sol             : $5 in / $30 out per MTok (GA 2026-07-09)
#   openrouter moonshotai/kimi-k2.7-code : $0.72 in / $3.50 out per MTok
#              (verified openrouter.ai/moonshotai/kimi-k2.7-code, Jul 17 2026)
#   moonshot   kimi-k3                   : $3 in / $15 out per MTok
#              (Moonshot DIRECT api.moonshot.ai/v1, OpenAI-compatible; used when
#               OpenRouter's relay to K3 is upstream-rate-limited on launch week.
#               Verified platform.kimi.ai/docs + kimik3.pro, Jul 19 2026.)
_PROVIDER_DEFAULTS = {
    "anthropic": {"model": "claude-opus-4-8", "in": 5.0, "out": 25.0},
    "openai": {"model": "gpt-5.6-sol", "in": 5.0, "out": 30.0},
    "openrouter": {"model": "moonshotai/kimi-k2.7-code", "in": 0.72, "out": 3.50},
    "moonshot": {"model": "kimi-k3", "in": 3.0, "out": 15.0},
    # DeepSeek DIRECT (api.deepseek.com) -- Sam's explicit rule: run DeepSeek V4
    # on Sam's OWN DeepSeek key, NOT via OpenRouter. OpenAI-compatible
    # chat.completions, so it reuses OpenRouterProvider. base_url is
    # https://api.deepseek.com (NO /v1). CoT comes back in reasoning_content,
    # which turn() already reads. Model deepseek-v4-pro (verified live Jul 19
    # 2026 vs api-docs.deepseek.com; legacy deepseek-chat/reasoner deprecate
    # Jul 24 2026). Rates: $0.28/$0.42 per Mtok (deepseek-v4-pro list price).
    "deepseek": {"model": "deepseek-v4-pro", "in": 0.28, "out": 0.42},
    # MOCK -- a $0, no-network scripted provider for the arm-wiring sanity check.
    # It calls the arm's memory tool once (proving ingest+retrieve+write) then
    # finishes. Never used for a real measured run. model/rates are placeholders.
    "mock": {"model": "mock-scripted", "in": 0.0, "out": 0.0},
}

if PROVIDER not in _PROVIDER_DEFAULTS:
    print(
        f"ERROR: unknown PROVIDER '{PROVIDER}'. Use 'anthropic', 'openai', "
        "'openrouter', 'moonshot', 'deepseek', or 'mock'.",
        file=sys.stderr,
    )
    sys.exit(2)

_DEF = _PROVIDER_DEFAULTS[PROVIDER]

MODEL = os.environ.get("MODEL", _DEF["model"])

# Per-MILLION-token USD rates. Default to the selected provider's verified list
# price; override with COST_PER_MTOK_INPUT/OUTPUT to match your exact model.
COST_PER_MTOK_INPUT = float(os.environ.get("COST_PER_MTOK_INPUT", str(_DEF["in"])))
COST_PER_MTOK_OUTPUT = float(os.environ.get("COST_PER_MTOK_OUTPUT", str(_DEF["out"])))

# Hard iteration cap so a looping agent terminates and is recorded as "looped".
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "16"))

# Max tokens per model response. On the OpenAI Responses API this ceiling bounds
# REASONING + visible output COMBINED, so at high/xhigh/max reasoning effort a low
# cap silently truncates the reasoning chain (status="incomplete") and degrades
# the run -- which would flatten the very token delta this demo measures. Default
# is generous; the run loop tolerates it because we only pay for tokens used.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))

# The command that reproduces + verifies (exit 0 == fixed).
TEST_CMD = os.environ.get("TEST_CMD", "npm test")

# The vestige binary/CLI (only used in --mode vestige).
VESTIGE_BIN = os.environ.get("VESTIGE_BIN", "vestige")
# Optional isolated data dir for the demo memory (keeps it out of your real DB).
VESTIGE_DATA_DIR = os.environ.get("VESTIGE_DATA_DIR", "")

TASK = "make the failing end-to-end charge test pass"


def fail(msg: str) -> None:
    """Print an error to stderr and exit non-zero. NEVER fabricate a result."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


# --- Tool implementations (real, run locally) -------------------------------

# Directories the agent's file tools NEVER see. Includes demo INFRASTRUCTURE:
# .repo-snapshot (the pristine copy), .fixtures (build/verify helpers), and the
# vestige demo DB. Exposing any of these can leak the answer or the harness
# internals into the agent's context.
_SKIP_DIRS = {
    "node_modules", "dist", ".git",
    ".vestige-demo-data", ".vestige-demo-db",
    ".repo-snapshot", ".fixtures",
}

# Individual files the agent's tools NEVER see, even by exact path. The seed
# script contains the SEEDED MEMORY (i.e. the answer) and must never be readable
# from the checkout -- the answer is only supposed to reach the agent through the
# Vestige backfill tool, not by cat-ing a file.
_SKIP_FILES = {".vestige-seed.sh"}


def _is_skipped(rel: Path) -> bool:
    if any(part in _SKIP_DIRS for part in rel.parts):
        return True
    if rel.name in _SKIP_FILES:
        return True
    return False


def _iter_repo_files(repo: Path):
    for p in sorted(repo.rglob("*")):
        rel = p.relative_to(repo)
        if _is_skipped(rel):
            continue
        if p.is_file():
            yield p


def tool_list_files(repo: Path, _args: dict) -> str:
    out = [str(p.relative_to(repo)) for p in _iter_repo_files(repo)]
    return "\n".join(out) if out else "(no files)"


def tool_read_file(repo: Path, args: dict) -> str:
    rel = args.get("path", "")
    target = (repo / rel).resolve()
    if not str(target).startswith(str(repo.resolve())):
        return "ERROR: path escapes repo"
    # Refuse demo-infrastructure / seed files even by exact path: the answer only
    # reaches the agent through the vestige tools, never by reading a file.
    try:
        rel_resolved = target.relative_to(repo.resolve())
        if _is_skipped(rel_resolved):
            return f"ERROR: no such file: {rel}"
    except ValueError:
        pass
    if not target.is_file():
        return f"ERROR: no such file: {rel}"
    text = target.read_text(errors="replace")
    return text[:10000]


def tool_grep(repo: Path, args: dict) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return "ERROR: empty pattern"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: bad regex: {exc}"
    hits = []
    for p in _iter_repo_files(repo):
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(repo)}:{i}: {line.strip()}")
                    if len(hits) >= 80:
                        return "\n".join(hits)
        except (OSError, UnicodeError):
            continue
    return "\n".join(hits) if hits else "(no matches)"


#: Environment variables the harness uses to carry the trial's ground truth.
#: These must NEVER reach a process an agent can influence.
_GROUND_TRUTH_ENV = ("CORRECT_KID", "PROD_CORPUS", "MASTER_SEED")


def _scrubbed_env() -> dict:
    """A copy of os.environ with the trial's ground truth REMOVED.

    run-experiment.sh exports CORRECT_KID (the answer) and PROD_CORPUS (the
    production replay corpus) so the harness can score the trial. _run_tests
    previously spawned `npm test` with no env= argument, so the agent's test
    command inherited both. Since agents have unrestricted write_file into a
    checkout whose test command globs test/*.test.ts, any agent in ANY arm
    could write a test that prints the answer. Verified live on Jul 20 2026:
    a planted test returned "LEAK CORRECT_KID= k_nashira" plus the corpus path.

    No observed agent exploited this, but its mere existence falsifies the
    experiment's headline claim that the key is "provably absent" -- the
    keyring leak audit scans FILES and never the environment. Scrubbing here
    makes the claim literally true rather than true-in-practice.
    """
    env = dict(os.environ)
    for key in _GROUND_TRUTH_ENV:
        env.pop(key, None)
    return env


def _run_tests(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        TEST_CMD,
        cwd=str(repo),
        shell=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=_scrubbed_env(),  # never hand the agent's test command the answer
    )


def tool_run_tests(repo: Path, _args: dict) -> str:
    """Run the test suite and return combined output + exit code."""
    try:
        proc = _run_tests(repo)
    except subprocess.TimeoutExpired:
        return "TIMEOUT after 180s"
    combined = (proc.stdout + "\n" + proc.stderr)[-6000:]
    return f"exit_code={proc.returncode}\n{combined}"


def tool_write_file(repo: Path, args: dict) -> str:
    rel = args.get("path", "")
    content = args.get("content", "")
    target = (repo / rel).resolve()
    if not str(target).startswith(str(repo.resolve())):
        return "ERROR: path escapes repo"
    if not target.parent.exists():
        return f"ERROR: parent dir does not exist for {rel}"
    target.write_text(content)
    return f"wrote {len(content)} bytes to {rel}"


def tool_vestige_backfill(_repo: Path, _args: dict) -> str:
    """Real call to `vestige backfill` -- reach BACKWARD from the failure to the
    quiet earlier memory that caused it. Only registered in --mode vestige."""
    cmd = [VESTIGE_BIN]
    if VESTIGE_DATA_DIR:
        cmd += ["--data-dir", VESTIGE_DATA_DIR]
    cmd += ["backfill", "--manual", "--json", "--lookback-days", "3650"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return (
            f"ERROR: vestige binary not found at '{VESTIGE_BIN}'. "
            "Set VESTIGE_BIN to the vestige CLI path."
        )
    except subprocess.TimeoutExpired:
        return "ERROR: vestige backfill timed out"
    # BOTH of these MUST be phrased so _classify_retrieval() catches them as
    # failures. They previously read "vestige backfill exit=N ..." and
    # "(vestige backfill returned no output)" -- neither starts with ERROR, nor
    # contains "index empty"/"no matching", so the classifier fell through to
    # "ok" and counted a DEAD Vestige arm as a successful retrieval
    # (memory_layer_alive=True). Every competitor's failure strings were already
    # classified correctly, so the ONLY arm that could fake liveness was our own
    # product -- the exact direction of error that would destroy the benchmark's
    # credibility. Caught by the pre-spend audit, Jul 21 2026.
    if proc.returncode != 0:
        return f"ERROR: vestige backfill exit={proc.returncode}\n{proc.stderr[:2000]}"
    return proc.stdout[:6000] or "(vestige index empty -- no memories to retrieve.)"


# --- RAG arm: a REAL, strong dense retriever over the SAME seeded memories ----
#
# This is the arm that answers "isn't Vestige just RAG?". It is NOT a strawman:
#   - embeds with nomic-embed-text -- the SAME model family Vestige itself uses
#   - retrieves over the SAME facts in the SAME Vestige DB (exported at call time)
#   - pure cosine top-K, no entity join -- i.e. exactly what "standard RAG" means
#   - same K and same 6000-char budget as vestige_backfill
# It fails on the demo scenario because the CAUSE (the k_quasar rotation runbook)
# is genuinely semantically distant from the failure symptom (it ranks ~#6-7 of
# 8 by cosine -- proven by tests/gate_rag_burial.py), so a fair top-3 returns the
# lookalike memories and NEVER hands the model the deciding fact. Same memory,
# similarity retrieval, still blind.
RAG_TOPK = int(os.environ.get("ARM_TOPK", "3"))
# `vestige export` feeds the SHARED corpus to every competitor arm. 60s was too
# tight on a loaded machine (3 fleet threads + ollama + local vector DBs all
# competing), and a miss there silently emptied every competitor's corpus while
# leaving Vestige's own backfill untouched. Generous timeout + retry + a lock.
RAG_EXPORT_TIMEOUT_S = int(os.environ.get("RAG_EXPORT_TIMEOUT_S", "300"))
# Per-embedding HTTP timeout against ollama. See _rag_embed for why 60s was too
# tight: OLLAMA_NUM_PARALLEL=1 serializes every request behind in-flight
# generation, so embeddings legitimately queue for minutes during an arm ingest.
RAG_EMBED_TIMEOUT_S = int(os.environ.get("RAG_EMBED_TIMEOUT_S", "300"))
# Import-time lock (never lazy -- a lazily built lock needs a lock).
_RAG_LOAD_LOCK = threading.Lock()
RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text:latest")
RAG_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/embeddings")

# Cache the exported facts + their embeddings per process (the DB is seeded once
# per trial, so this is stable within a run).
_RAG_CACHE: dict = {}


def _rag_embed(text: str) -> list:
    """Embed one string via ollama.

    TIMEOUT RATIONALE: ollama is deliberately run with OLLAMA_NUM_PARALLEL=1
    (required -- without it the fleet's concurrent requests balloon a small
    model's KV cache and wedge the machine). That serialization means an
    embedding request QUEUES behind any in-flight generation. While an arm is
    ingesting through llama3.1:8b, a nomic-embed-text call can wait minutes.
    The old 60s cap turned that ordinary queueing into
    "shared corpus unavailable (timed out)", which emptied the corpus for every
    competitor arm at once. Generous timeout + retry instead.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    last: Exception | None = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                RAG_OLLAMA_URL,
                data=json.dumps({"model": RAG_EMBED_MODEL, "prompt": text}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=RAG_EMBED_TIMEOUT_S) as r:
                return json.load(r)["embedding"]
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"ollama embedding failed after 3 attempts via {RAG_OLLAMA_URL} "
        f"(model {RAG_EMBED_MODEL}): {type(last).__name__}: {last}"
    )


def _rag_load_facts() -> list:
    """Export the SAME memories Vestige holds and embed them once. Source of truth
    is the seeded Vestige DB -- so RAG retrieves over the exact same facts, never a
    re-authored copy."""
    if _RAG_CACHE.get("facts"):
        return _RAG_CACHE["facts"]
    # LOCKED: this is the SHARED corpus loader for every competitor arm (rag,
    # mem0, supermemory, hindsight, zep -- see the call sites). The fleet runs
    # FLEET_SIZE agents as threads, so without a lock all three would shell out
    # to `vestige export` simultaneously and each re-embed the whole corpus.
    with _RAG_LOAD_LOCK:
        if _RAG_CACHE.get("facts"):
            return _RAG_CACHE["facts"]
        facts: list = []
        last_err: str = ""
        if VESTIGE_DATA_DIR:
            import tempfile  # noqa: PLC0415

            cmd = [VESTIGE_BIN, "--data-dir", VESTIGE_DATA_DIR, "export",
                   "--format", "json"]
            # Retry: under a loaded machine (three arms + ollama + a local
            # vector DB) a single export can exceed its timeout. A transient
            # blip here used to empty the corpus for EVERY competitor arm while
            # never touching Vestige's own backfill, which reads the DB
            # directly -- a failure that is silent, asymmetric, and always in
            # Vestige's favour. Retry, then fail LOUDLY.
            for attempt in range(3):
                out = Path(tempfile.mktemp(suffix=".json"))
                try:
                    proc = subprocess.run(
                        [*cmd, str(out)], capture_output=True, text=True,
                        timeout=RAG_EXPORT_TIMEOUT_S,
                    )
                    if proc.returncode != 0:
                        last_err = f"vestige export exit {proc.returncode}: {proc.stderr[:200]}"
                    elif not out.exists():
                        last_err = "vestige export wrote no file"
                    else:
                        raw = json.loads(out.read_text())
                        rows = raw if isinstance(raw, list) else raw.get(
                            "memories", raw.get("nodes", [])
                        )
                        for m in rows:
                            body = m.get("content") or m.get("body") or ""
                            if body:
                                facts.append(body)
                        if facts:
                            break
                        last_err = "vestige export returned 0 memories"
                except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
                    last_err = f"{type(exc).__name__}: {exc}"[:200]
                finally:
                    try:
                        out.unlink()
                    except OSError:
                        pass
        # Exclude the failure memory itself from the retrievable corpus (a RAG
        # indexes prior history, not the live error it is being asked about).
        facts = [f for f in facts if "PRODUCTION OUTAGE" not in f]
        if not facts:
            # NEVER cache an empty corpus. The old code cached whatever it got,
            # so ONE transient export failure poisoned the cache for the rest of
            # the process and every later retrieval silently returned nothing.
            # Raising here makes the arm report ERROR (and the new liveness
            # telemetry mark it memory_unavailable) instead of quietly
            # pretending the competitor's memory was empty.
            raise RuntimeError(
                "shared corpus export produced no memories after 3 attempts "
                f"({last_err or 'no detail'}). Refusing to hand any arm an empty "
                "corpus -- that would score as a retrieval loss."
            )
        embs = [_rag_embed(f) for f in facts]
        _RAG_CACHE["facts"] = facts
        _RAG_CACHE["embs"] = embs
    return facts


def tool_rag_search(_repo: Path, args: dict) -> str:
    """Standard dense-RAG retrieval: embed the query, return the top-K most
    cosine-similar memories. Same embedder, same facts, same budget as Vestige --
    only the causal join is removed. Registered only in --mode rag."""
    query = (args or {}).get("query") or ""
    if not query.strip():
        return "ERROR: rag_search requires a non-empty 'query'."
    try:
        facts = _rag_load_facts()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: RAG index unavailable ({exc}). Is ollama running?"
    if not facts:
        return "(RAG index empty -- no memories to retrieve.)"
    embs = _RAG_CACHE["embs"]

    def _cos(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    qv = _rag_embed(query)
    ranked = sorted(
        ((_cos(qv, e), f) for e, f in zip(embs, facts)),
        key=lambda t: t[0], reverse=True,
    )[:RAG_TOPK]
    lines = [
        f"[{i}] (similarity {score:.3f}) {body}"
        for i, (score, body) in enumerate(ranked, 1)
    ]
    header = (
        f"Top-{RAG_TOPK} memories by semantic similarity to your query "
        f"(cosine, {RAG_EMBED_MODEL}):\n"
    )
    return (header + "\n".join(lines))[:6000]


BASE_TOOLS = {
    "list_files": (tool_list_files, "List every source file in the repo (skips node_modules/dist).", {"type": "object", "properties": {}}),
    "read_file": (
        tool_read_file,
        "Read a file's contents (first 10000 chars).",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    "grep": (
        tool_grep,
        "Search the repo source for a regular expression.",
        {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    ),
    "run_tests": (
        tool_run_tests,
        "Run the test suite (npm test) and return output + exit code. Use to "
        "reproduce the failure and to verify a fix. exit_code=0 means fixed.",
        {"type": "object", "properties": {}},
    ),
    "write_file": (
        tool_write_file,
        "Overwrite a file with new content (use to apply a fix).",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    ),
    "finish": (
        None,
        "Call when done. Set fixed=true only if run_tests now exits 0.",
        {"type": "object", "properties": {"fixed": {"type": "boolean"}, "summary": {"type": "string"}}, "required": ["fixed"]},
    ),
}

VESTIGE_TOOL = {
    "vestige_backfill": (
        tool_vestige_backfill,
        "Retroactive Salience Backfill: reach BACKWARD from the current failure "
        "to surface the quiet EARLIER change that caused it -- the root cause a "
        "plain code/vector search for the error string cannot find (it may live "
        "in a different service). Call this when a failure's cause is not visible "
        "at the failure site.",
        {"type": "object", "properties": {}},
    ),
}


RAG_TOOL = {
    "rag_search": (
        tool_rag_search,
        "Semantic memory search: retrieve the most relevant memories about this "
        "failure from the team's history. Embeds your query and returns the "
        "top-K most similar past notes/incidents/decisions. Use it to find prior "
        "context that explains the failure -- the cause may live in an earlier "
        "change in a different service.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}


# ============================================================================
# mem0 arm: mem0ai OSS v2.0.12, FULLY LOCAL (Ollama LLM + Ollama embeddings +
# Chroma on-disk). The fair competitor to "isn't Vestige just a memory layer?".
# It uses mem0's OWN RECOMMENDED path -- default infer=True, so .add() runs an
# LLM fact-extraction pass (llama3.1:8b) over each input, and .search() embeds
# the query (nomic-embed-text) and returns top-K with scores. It retrieves over
# the SAME shared corpus every arm gets (_rag_load_facts -- the seeded Vestige
# DB, PRODUCTION OUTAGE excluded), with the SAME top-K budget (ARM_TOPK), SAME
# 6000-char cap and SAME output shape as tool_rag_search. Only mem0's own
# extraction+store+retrieval differ.
#
# TWO fatal bugs were found + fixed vs the real installed v2.0.12 source
# (audit b02721ac, re-verified live Jul 19 2026):
#   BUG 1 search sig: v2.0.12 is search(query, *, top_k=20, filters=None,
#     threshold=0.1, ...). user_id/limit as top-level kwargs CRASH via
#     _reject_top_level_entity_params. FIX: filters={"user_id":...},
#     top_k=MEM0_TOPK, threshold=0.0 (0.0 for fairness -- the default 0.1 drops
#     the low-ranked buried causal fact; the RAG arm has no threshold).
#   BUG 2 chroma cfg: ChromaDbConfig only accepts
#     collection_name/client/path/host/port/api_key/tenant -- "embedding_model_dims"
#     is REJECTED at construction. FIX: omit it; the embedder's embedding_dims=768
#     is the correct+sufficient place for nomic-embed-text dims.

MEM0_TOPK = int(os.environ.get("ARM_TOPK", "3"))  # SAME budget as the RAG arm.
MEM0_USER = os.environ.get("MEM0_USER", "bench")
MEM0_LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", "llama3.1:8b")
MEM0_EMBED_MODEL = os.environ.get("MEM0_EMBED_MODEL", "nomic-embed-text:latest")
MEM0_OLLAMA_URL = os.environ.get("MEM0_OLLAMA_URL", "http://localhost:11434")
# On-disk Qdrant path -- embedded, no server process, no port. Isolated per run
# so a stale collection can't leak across trials. Directory name is backend-
# specific on purpose: the store moved from chroma to qdrant (mem0's own default,
# and the only one of the two that supports mem0's hybrid BM25 retrieval), and
# the two backends' on-disk formats are not interchangeable. A fresh path avoids
# silently loading a chroma directory as if it were a qdrant one.
MEM0_STORE_DIR = os.environ.get(
    "MEM0_STORE_DIR",
    str(Path(VESTIGE_DATA_DIR or ".").parent / "arm_store" / "mem0_qdrant"),
)

# Cache the Memory instance + ingest state per process, like _RAG_CACHE:
# from_config() is expensive and the corpus is seeded once per trial.
_MEM0_CACHE: dict = {}
# Import-time lock -- see _zep_ingest_lock() for why this must not be lazy.
_MEM0_INGEST_LOCK = threading.Lock()


def _mem0_client():
    """Build (once) a FULLY LOCAL mem0 Memory: Ollama for BOTH the LLM (fact
    extraction on .add()) AND embeddings, Chroma embedded on-disk. Raises if
    mem0ai is not installed -- the caller turns that into a clean ERROR string,
    mirroring how tool_rag_search reports a missing ollama."""
    if _MEM0_CACHE.get("client") is not None:
        return _MEM0_CACHE["client"]
    from mem0 import Memory  # noqa: PLC0415  (lazy: missing dep -> ERROR string)

    Path(MEM0_STORE_DIR).mkdir(parents=True, exist_ok=True)
    config = {
        # QDRANT, because it is mem0's OWN DEFAULT vector store (verified:
        # mem0.vector_stores.configs.VectorStoreConfig().provider == "qdrant"),
        # and because chroma silently DISABLES half of mem0's retrieval. Running
        # on chroma, mem0 itself warns at runtime:
        #
        #   "The 'chroma' vector store does not support keyword search. Hybrid
        #    (BM25) scoring will be disabled and search will use semantic
        #    similarity only. To enable hybrid search, switch to a store with
        #    keyword_search support (e.g. qdrant, elasticsearch, pgvector)."
        #
        # Benchmarking a competitor on a store its own library says cripples it,
        # while the product arm runs its full pipeline, is not a fair test. The
        # standard applied across every arm here is: RUN EACH PRODUCT ON ITS OWN
        # VENDOR DEFAULT. (Same reason supermemory was restored to its default
        # bge-base-en-v1.5 768d embedder after having been pinned to a weaker
        # 384d model.) qdrant runs embedded on-disk via `path` exactly like
        # chroma did -- no server, no new infrastructure.
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "vestige_bench",
                "path": MEM0_STORE_DIR,          # embedded, on-disk; NO server
                # qdrant's own default is 1536 (OpenAI-sized); nomic-embed-text
                # is 768, and the collection dim must match the embedder below
                # or every insert is rejected.
                "embedding_model_dims": 768,
            },
        },
        # BOTH blocks MUST be ollama. A missing llm OR embedder block silently
        # falls back to OpenAI and requires OPENAI_API_KEY.
        "llm": {
            "provider": "ollama",
            "config": {
                "model": MEM0_LLM_MODEL,
                "temperature": 0,
                "max_tokens": 2000,
                # Python field is ollama_base_url (the TS SDK uses `url`; in
                # Python `url` is silently ignored -> OpenAI fallback).
                "ollama_base_url": MEM0_OLLAMA_URL,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": MEM0_EMBED_MODEL,
                "embedding_dims": 768,           # nomic-embed-text = 768
                "ollama_base_url": MEM0_OLLAMA_URL,
            },
        },
    }
    client = Memory.from_config(config)
    _MEM0_CACHE["client"] = client
    return client


def _mem0_ingest_once():
    """Ingest the SHARED corpus (_rag_load_facts) ONCE per process into mem0's
    own store via its RECOMMENDED path: default infer=True, so each .add() runs
    an LLM fact-extraction pass. This is what a real mem0 user runs, so per the
    harness fairness rule it is the FAIR ingest for mem0. One fact per .add() so
    llama3.1's num_ctx never truncates a batch. add() does NOT call the reject
    guard (only search/get_all do), so user_id as a top-level kwarg is fine."""
    if _MEM0_CACHE.get("ingested"):
        return
    with _MEM0_INGEST_LOCK:
        # Re-check inside the lock. The fleet calls this from FLEET_SIZE threads;
        # the previous unlocked check-then-set let all three ingest the SAME
        # corpus concurrently, which stored each fact up to 3x and collapsed
        # mem0's effective top-K from 3 distinct facts to ~1.
        if _MEM0_CACHE.get("ingested"):
            return
        client = _mem0_client()
        facts = _rag_load_facts()  # SAME corpus as every arm -- never re-authored.
        for fact in facts:
            client.add(
                [{"role": "user", "content": fact}],
                user_id=MEM0_USER,
                # infer=True is the DEFAULT + recommended path -- omitted on purpose.
            )
        _MEM0_CACHE["ingested"] = True


def tool_mem0_search(_repo: Path, args: dict) -> str:
    """mem0's RECOMMENDED retrieval: .search() embeds the query (nomic-embed-text)
    and returns the top-K stored memories with similarity scores. Same corpus,
    same K, same 6000-char budget and same "[i] (similarity X) <body>" shape as
    tool_rag_search -- only mem0's own extraction+store+retrieval differ."""
    query = (args or {}).get("query") or ""
    if not query.strip():
        return "ERROR: mem0_search requires a non-empty 'query'."
    try:
        _mem0_ingest_once()
        client = _mem0_client()
    except ImportError:
        return (
            "ERROR: mem0 index unavailable -- the 'mem0ai' package is not "
            "installed. Run: pip install mem0ai chromadb"
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: mem0 index unavailable ({exc}). Is ollama running at "
            f"{MEM0_OLLAMA_URL} with '{MEM0_LLM_MODEL}' and "
            f"'{MEM0_EMBED_MODEL}' pulled?"
        )

    try:
        # v2.0.12 signature: search(query, *, top_k, filters, threshold, ...).
        # filters MUST carry >=1 entity id (user_id/agent_id/run_id) or it
        # raises. threshold=0.0 for fairness (default 0.1 would drop the buried
        # low-ranked causal fact; the RAG arm applies no threshold).
        resp = client.search(
            query,
            filters={"user_id": MEM0_USER},
            top_k=MEM0_TOPK,
            threshold=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: mem0 search failed ({exc})."
    # Current OSS returns {"results": [...]}; older snippets return a bare list.
    results = resp["results"] if isinstance(resp, dict) else (resp or [])
    if not results:
        return "(mem0 index empty -- no memories to retrieve.)"

    lines = []
    for i, r in enumerate(results, 1):
        body = (r.get("memory") or r.get("text") or "") if isinstance(r, dict) else str(r)
        sim = (r.get("score") or 0.0) if isinstance(r, dict) else 0.0
        try:
            sim = float(sim)
        except (TypeError, ValueError):
            sim = 0.0
        lines.append(f"[{i}] (similarity {sim:.3f}) {body}")
    header = (
        f"Top-{MEM0_TOPK} memories from mem0 (LLM-extracted on ingest, retrieved "
        f"by semantic similarity, {MEM0_EMBED_MODEL}):\n"
    )
    return (header + "\n".join(lines))[:6000]


MEM0_TOOL = {
    "mem0_search": (
        tool_mem0_search,
        "Semantic memory search over the team's history via mem0: retrieve the "
        "most relevant past notes/incidents/decisions about this failure. mem0 "
        "extracted and stored these facts on ingest; your query is embedded and "
        "the top-K most similar memories are returned. Use it to find prior "
        "context that explains the failure -- the cause may live in an earlier "
        "change in a different service.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}


# ============================================================================
# supermemory arm: self-hosted supermemory-server (supermemoryai/supermemory),
# FULLY LOCAL. The "isn't a turnkey memory PRODUCT better than Vestige?" arm.
# Not a strawman: ingests the SAME seeded corpus (_rag_load_facts, exported from
# the SAME Vestige DB, never re-authored), lets supermemory run its OWN
# recommended pipeline (local ONNX embeddings + ollama LLM extraction on ingest),
# and retrieves via its OWN recommended path POST /v3/search. Same K (ARM_TOPK),
# same 6000-char cap as the RAG/Vestige arms -- only store + retriever change.
# Server at http://localhost:6767 (bundled ONNX embeddings, own local key,
# encrypted local storage). Verified 0 bugs vs live /v3 API (Jul 19 2026).
SM_TOPK = int(os.environ.get("ARM_TOPK", "3"))
SM_BASE_URL = os.environ.get("SUPERMEMORY_BASE_URL", "http://localhost:6767")
# First boot prints an API key; the local server auto-applies it for
# unauthenticated localhost requests, so a blank key still works locally.
SM_API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")
# PER-TRIAL ISOLATION (fairness): supermemory-server has ONE persistent store, so
# a FIXED container tag leaks every trial's runbook into every other trial -- by
# trial 5 the arm searches 5 trials' runbooks, each naming a different "live"
# key, an unfair extra-decoy handicap the other arms (which wipe per trial) do
# NOT face. Namespace the container to THIS trial's correct key (CORRECT_KID,
# unique per trial, set by run-experiment.sh) so each trial only ever sees its
# OWN seeded corpus -- matching the per-trial reset of vestige/rag/mem0/zep.
_SM_TRIAL = os.environ.get("CORRECT_KID", "") or os.environ.get("MASTER_SEED", "x")
SM_CONTAINER = os.environ.get("SUPERMEMORY_CONTAINER", f"benchmark-{_SM_TRIAL}")
_SM_CACHE: dict = {}
# Import-time lock -- see _zep_ingest_lock() for why this must not be lazy.
_SM_INGEST_LOCK = threading.Lock()


def _sm_req(method: str, path: str, body: dict | None = None) -> dict:
    import urllib.request  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if SM_API_KEY:
        headers["Authorization"] = f"Bearer {SM_API_KEY}"
    # Retry transient errors (timeouts / 5xx / connection resets) with backoff.
    # Under heavy local load the server can briefly stall; a momentary hiccup
    # must NOT turn the whole arm into an ERROR (that would look like a strawman).
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                f"{SM_BASE_URL}{path}", data=data, method=method, headers=headers
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # 4xx are real; only retry 5xx
            if e.code < 500 or attempt == 3:
                raise
            last_exc = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 3:
                raise
            last_exc = e
        time.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("supermemory request failed after retries")


def _sm_ingest_once() -> None:
    """Ingest the SAME shared corpus into supermemory ONCE per process, then BLOCK
    until every doc is indexed (POST /v3/documents is ASYNC -> status='queued';
    poll GET /v3/documents/{id} until 'done' or the query fires against an
    unindexed store and recall is silently 0)."""
    if _SM_CACHE.get("ready"):
        return
    with _SM_INGEST_LOCK:
        # Re-check inside the lock -- FLEET_SIZE threads reach this at once and
        # the previous unlocked check-then-set let all of them POST the same
        # corpus, duplicating every document in the container.
        if _SM_CACHE.get("ready"):
            return
        facts = _rag_load_facts()  # SAME corpus as every arm -- do NOT re-author.
        doc_ids = []
        for i, fact in enumerate(facts):
            resp = _sm_req("POST", "/v3/documents", {
                "content": fact,
                "containerTag": SM_CONTAINER,
                "customId": f"fact-{i}",
                "dreaming": "instant",
            })
            doc_ids.append(resp["id"])
        deadline = time.time() + 300
        for did in doc_ids:
            while time.time() < deadline:
                st = _sm_req("GET", f"/v3/documents/{did}").get("status")
                if st == "done":
                    break
                if st == "failed":
                    raise RuntimeError(f"supermemory doc {did} failed to index")
                time.sleep(1.5)
        _SM_CACHE["ready"] = True


def tool_supermemory_search(_repo: Path, args: dict) -> str:
    """RECOMMENDED retrieval: POST /v3/search (document search) -> ranked docs,
    each with a `score` and `chunks[]`. Same facts, same K, same 6000-char budget
    as the RAG arm -- only the store/retriever is supermemory's own."""
    query = (args or {}).get("query") or ""
    if not query.strip():
        return "ERROR: supermemory_search requires a non-empty 'query'."
    try:
        _sm_ingest_once()
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: supermemory unavailable ({exc}). "
            f"Is supermemory-server running on {SM_BASE_URL}?"
        )
    try:
        resp = _sm_req("POST", "/v3/search", {
            "q": query,                     # field is `q`, NOT `query`
            "containerTags": [SM_CONTAINER],
            "limit": SM_TOPK,
            "chunkThreshold": 0,            # 0 = keep all ranked chunks (don't cripple recall)
        })
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: supermemory /v3/search failed ({exc})."
    results = (resp.get("results") or [])[:SM_TOPK]
    if not results:
        return "(supermemory returned no matching memories.)"
    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        chunks = [c.get("content", "") for c in (r.get("chunks") or [])
                  if c.get("isRelevant", True)]
        body = " ".join(chunks).strip() or r.get("title", "")
        lines.append(f"[{i}] (similarity {score:.3f}) {body}")
    header = f"Top-{SM_TOPK} memories by supermemory /v3/search (relevance score):\n"
    return (header + "\n".join(lines))[:6000]


SUPERMEMORY_TOOL = {
    "supermemory_search": (
        tool_supermemory_search,
        "Semantic memory search over the team's history via supermemory. Retrieve "
        "the top-K most relevant past notes/incidents/decisions for your query. "
        "Use it to find prior context that explains the failure -- the cause may "
        "live in an earlier change in a different service.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}


# ============================================================================
# hindsight arm: vectorize-io/hindsight (MIT), FULLY LOCAL. Agent-memory server
# with embedded Postgres + local embedder (BAAI/bge-small-en-v1.5) + local
# cross-encoder reranker (ms-marco-MiniLM-L-6-v2). Fair competitor to RAG:
# ingests the SAME seeded corpus (_rag_load_facts, never re-authored), SAME
# ARM_TOPK, SAME 6000-char cap, but through Hindsight's OWN recommended path:
# retain() (LLM fact-extraction on ingest, forced to ollama not OpenAI) +
# recall() (semantic + BM25 + local cross-encoder rerank). Bank pinned to
# verbatim extraction so stored text == corpus body. Server = separate process
# (hindsight-api), started once, fully local. Verified 0 bugs vs live 0.8.4.
HINDSIGHT_TOPK = int(os.environ.get("ARM_TOPK", "3"))
# Pinned to 127.0.0.1, NOT "localhost". hindsight-api (uvicorn) binds IPv4
# 127.0.0.1 only, while macOS resolves "localhost" to ::1 FIRST. Any process
# holding the IPv6 wildcard on this port therefore intercepts every call. That
# is not theoretical: on Jul 20 2026 a stray `python -m http.server 8899` bound
# *:8899 and answered every hindsight request with HTTP 501 "Unsupported method
# ('POST')" from SimpleHTTP -- the arm looked dead while the real service was
# healthy on IPv4 the whole time. Pinning the literal IPv4 address removes the
# entire class of failure.
HINDSIGHT_BASE_URL = os.environ.get("HINDSIGHT_BASE_URL", "http://127.0.0.1:8899")
# PER-TRIAL ISOLATION (fairness): hindsight-api's embedded Postgres persists, so
# a FIXED bank_id accumulates every trial's runbook (same cross-trial-leak
# handicap as supermemory had). Namespace the bank to THIS trial's correct key
# so each trial only sees its own seeded corpus, matching every other arm's
# per-trial reset.
_HS_TRIAL = os.environ.get("CORRECT_KID", "") or os.environ.get("MASTER_SEED", "x")
HINDSIGHT_BANK = os.environ.get("HINDSIGHT_BANK", f"benchmark-{_HS_TRIAL}")
_HINDSIGHT_CACHE: dict = {}
# See _zep_ingest_lock() for why this is created at import time, not lazily.
_HINDSIGHT_INGEST_LOCK = threading.Lock()
# Hindsight's retain() runs LLM extraction per fact through ollama, which is
# minutes for the whole corpus. Matches the server's own API_TIMEOUT_MS=900000.
# This budget is for the PRE-WARM phase, not the agent's turn -- see the
# pre-warm step in fleet_runner, which ingests before the paid loop starts.
HINDSIGHT_INGEST_TIMEOUT_S = int(os.environ.get("HINDSIGHT_INGEST_TIMEOUT_S", "900"))


def _hs_loop():
    """Dedicated event loop thread for Hindsight (same rationale as _zep_loop).

    hindsight-client's SYNC methods route through its own _run_async(), which
    calls asyncio.get_event_loop() + run_until_complete. Called from the fleet's
    worker THREADS that raises "Timeout context manager should be used inside a
    task" (aiohttp's timeout requires a running task on the loop that owns the
    session), which is precisely the error recorded on 100% of hindsight calls
    in the paid runs. The client's own docstring says to prefer the async
    variants (arecall/aretain/acreate_bank) outside a plain script context, so
    we drive those on one owned loop and submit from any thread.
    """
    loop = _HINDSIGHT_CACHE.get("loop")
    if loop is None or loop.is_closed():
        import asyncio  # noqa: PLC0415

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="hindsight-event-loop", daemon=True
        )
        thread.start()
        _HINDSIGHT_CACHE["loop"] = loop
        _HINDSIGHT_CACHE["loop_thread"] = thread
    return loop


def _hs_run(coro, timeout: int = 300):
    """Submit a Hindsight coroutine to the dedicated loop from ANY thread."""
    import asyncio  # noqa: PLC0415

    future = asyncio.run_coroutine_threadsafe(coro, _hs_loop())
    try:
        return future.result(timeout=timeout)
    except BaseException:
        future.cancel()
        raise


def _hindsight_client():
    """Construct (once) the Hindsight client, or raise with a clear message if
    the dep/service is missing. The server (hindsight-api) must already be
    running -- the pure client does not spawn one."""
    if _HINDSIGHT_CACHE.get("client") is not None:
        return _HINDSIGHT_CACHE["client"]
    try:
        from hindsight_client import Hindsight  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "hindsight-client not installed. "
            "pip install 'hindsight-all>=0.8.4' hindsight-client"
        ) from exc
    client = Hindsight(base_url=HINDSIGHT_BASE_URL)  # local server, no api_key
    _HINDSIGHT_CACHE["client"] = client
    return client


def _hindsight_ensure_ingested(facts: list) -> None:
    """Ingest the SHARED corpus into Hindsight's OWN local store exactly once per
    process. Bank in verbatim mode so stored text == input fact (fair, no LLM
    paraphrase corrupting the '[i] body' match). retain() is synchronous, so
    every fact is committed before any recall()."""
    if _HINDSIGHT_CACHE.get("ingested"):
        return
    with _HINDSIGHT_INGEST_LOCK:
        # Re-check inside the lock: the fleet calls this from FLEET_SIZE threads
        # at once, and the previous unlocked check-then-set let all of them
        # ingest the same corpus concurrently.
        if _HINDSIGHT_CACHE.get("ingested"):
            return
        client = _hindsight_client()
        try:
            _hs_run(
                client.acreate_bank(
                    bank_id=HINDSIGHT_BANK,
                    retain_extraction_mode="verbatim",
                    enable_observations=False,
                )
            )
        except Exception:  # noqa: BLE001  -- bank likely already exists; reuse it
            pass
        for i, body in enumerate(facts):
            _hs_run(
                client.aretain(
                    bank_id=HINDSIGHT_BANK, content=body, document_id=f"fact-{i}"
                ),
                timeout=HINDSIGHT_INGEST_TIMEOUT_S,
            )
        _HINDSIGHT_CACHE["ingested"] = True


def tool_hindsight_search(_repo: Path, args: dict) -> str:
    """Hindsight's RECOMMENDED retrieval: recall() = semantic + BM25 + local
    cross-encoder rerank over the SAME seeded corpus. Same K, same 6000-char
    budget as rag_search -- only the retrieval engine differs."""
    query = (args or {}).get("query") or ""
    if not query.strip():
        return "ERROR: hindsight_search requires a non-empty 'query'."
    try:
        facts = _rag_load_facts()           # SHARED corpus -- identical to RAG
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: shared corpus unavailable ({exc}). Is the vestige CLI exportable?"
    if not facts:
        return "(Hindsight index empty -- no memories to retrieve.)"
    try:
        _hindsight_ensure_ingested(facts)
        client = _hindsight_client()
        # arecall (async variant) on the dedicated loop -- the sync recall()
        # wrapper uses get_event_loop()+run_until_complete and breaks when
        # called from the fleet's worker threads.
        resp = _hs_run(
            client.arecall(
                bank_id=HINDSIGHT_BANK,
                query=query,
                budget="low",
                types=["world", "experience"],
            )
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: Hindsight retrieval unavailable ({type(exc).__name__}: {exc}). "
            f"Is the hindsight-api server running at {HINDSIGHT_BASE_URL} and "
            "Ollama up for retain()?"
        )
    results = getattr(resp, "results", None) or []
    if not results:
        return "(Hindsight returned no matching memories.)"
    lines = [f"[{i}] {getattr(r, 'text', '')}" for i, r in enumerate(results[:HINDSIGHT_TOPK])]
    header = (
        f"Top-{HINDSIGHT_TOPK} memories by Hindsight recall "
        f"(semantic + BM25 + cross-encoder rerank):\n"
    )
    return (header + "\n".join(lines))[:6000]


HINDSIGHT_TOOL = {
    "hindsight_search": (
        tool_hindsight_search,
        "Semantic memory search: retrieve the most relevant memories about this "
        "failure from the team's history. Uses agent-memory recall (hybrid "
        "semantic + keyword + rerank) and returns the top-K most relevant past "
        "notes/incidents/decisions. Use it to find prior context that explains "
        "the failure -- the cause may live in an earlier change in a different "
        "service.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}


# ============================================================================
# zep arm: Zep's open-source engine Graphiti (graphiti-core[falkordb]), FULLY
# LOCAL -- NOT zep-cloud, NO ZEP_API_KEY. A temporal knowledge-graph competitor:
# extract entity/edge nodes on ingest (LLM, forced to ollama), retrieve via
# hybrid semantic + BM25 + graph RRF. Ingests the SAME seeded corpus
# (_rag_load_facts, never re-authored), SAME ARM_TOPK, SAME 6000-char cap. Local
# FalkorDB backing store (:6379), ollama for LLM + embeddings + reranker.
# graphiti-core is fully ASYNC, so this arm owns one persistent event loop + one
# client per process, ingesting once (cached like _RAG_CACHE).
#
# THREE audit-found fairness bugs FIXED (all rigged the arm to LOSE, disqualifying):
#   FIX 1 (prompt): the missing _agent_system_prompt "zep" branch is added in
#     fleet_runner.py so zep agents are told to CALL zep_search (else they got the
#     anarchy "you have NO memory" prompt). [handled in fleet_runner]
#   FIX 2 (cross-trial contamination): a fixed group_id on a PERSISTENT FalkorDB
#     accumulated every prior trial's facts. We now CLEAR this group's graph
#     before ingest each process, so each trial reads only its own facts.
#   FIX 3 (chronology): reference_time=datetime.now() for every episode flattened
#     all facts to "now", disabling temporal supersession (the arm's whole point).
#     We now assign a monotonically increasing reference_time in corpus order so
#     Graphiti has a real chronology to reason over.
# CRITICAL: disable Graphiti's PostHog telemetry phone-home. It is ON by default
# (GRAPHITI_TELEMETRY_ENABLED defaults to 'true') and fires an UN-TIMED HTTPS call
# to us.i.posthog.com (Cloudflare 104.18.x). On Jul 19 2026 this HUNG the zep arm
# for 23+ min mid-run (python 0% CPU blocked on the socket; ollama idle; FalkorDB
# fine). Set it BEFORE graphiti_core is imported so the flag is read at init.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
# Hard wall-clock cap for the whole zep tool call. Even with telemetry off, any
# single Graphiti/Ollama/FalkorDB stall must surface as a clean arm ERROR, never
# an indefinite hang that blocks the entire experiment loop.
ZEP_TIMEOUT_S = int(os.environ.get("ZEP_TIMEOUT_S", "600"))
# Graphiti's INGEST is a different cost class from its SEARCH and needs its own
# budget. Ingest runs one LLM entity/edge extraction pass per fact through
# gemma3:12b; measured Jul 20 2026, that exceeds 600s on this corpus and the arm
# aborted mid-build with a partial 21-node graph. A partial graph is worse than
# none -- it retrieves badly and looks like a product failure.
#
# The 600s cap made sense when ingest ran INSIDE the agent's bounded tool call.
# It no longer does: fleet_runner pre-warms every arm's store before the paid
# loop starts, so a long ingest costs wall-clock only -- zero tokens, zero agent
# turns. Give it room to finish. SEARCH keeps the tighter ZEP_TIMEOUT_S, because
# a slow search during a live turn genuinely should abort.
#
# gemma3:12b is REQUIRED here and must not be swapped for a smaller model to go
# faster: llama3.1:8b was tested and fails Graphiti's JSON extraction, emitting
# edges whose entities were never created as nodes -> empty retrieval. Making
# the competitor faster by making its graph malformed is not a fairness fix.
ZEP_INGEST_TIMEOUT_S = int(os.environ.get("ZEP_INGEST_TIMEOUT_S", "2400"))

ZEP_TOPK = int(os.environ.get("ARM_TOPK", "3"))
# Capable extraction model -- SMALL models fail Graphiti's JSON extraction and
# produce a malformed graph (verified Jul 19: llama3.1:8b emitted edges whose
# entities were never created as nodes -> empty retrieval; gemma3:27b builds a
# clean graph -> retrieves the buried causal fact at rank 1). Default to the
# stronger local model so Graphiti gets its fair best shot.
# qwen2.5:14b, NOT gemma3:12b. Verified from Zep's own docs (Jul 21 2026 web
# research): Graphiti's extraction REQUIRES a model with real structured-output
# support, and its docs explicitly warn that smaller/weaker models "may not
# accurately extract data or output the correct JSON structures required by
# Graphiti." gemma3 has NO native function/tool-calling (prompt-engineered only),
# so it emitted garbage predicates ("CARRRIES"), orphan edges, and a ~13-node
# graph that retrieved nothing -- the exact failure in getzep/graphiti #868/#1171.
# That measured gemma3, not Zep.
# DEFAULT = qwen2.5:14b on OLLAMA. Chosen because Graphiti REQUIRES the LLM
# server to ENFORCE structured output (response_format), and only ollama does
# among what's local:
#   - ollama ENFORCES response_format -> returns valid JSON (verified Jul 21:
#     qwen2.5:14b returned parseable JSON). Graphiti's extraction works.
#   - mlx_lm.server (:8081) IGNORES response_format and returns free-form PROSE
#     even for json_schema AND json_object (verified Jul 21). Graphiti then gets
#     non-JSON / empty content -> JSONDecodeError. So the strong Qwen3.6-35B-A3B
#     Sam has on MLX CANNOT drive Graphiti through mlx_lm.server, despite the
#     model being fully capable -- it's a SERVER limitation, not a model one.
# qwen2.5:14b is pulled, ~9GB/16GB VRAM (safe on 64GB), native structured output,
# and a verified-working Graphiti extraction model (Flo976 reference repo).
# QWEN3.6 UPGRADE PATH (Sam's preferred stronger model): `ollama pull
# qwen3.6:35b-a3b` (it IS in the ollama registry) then set ZEP_LLM_MODEL=
# qwen3.6:35b-a3b -- ollama enforces structured output so Graphiti works, and it
# gives zep an even stronger extractor. Requires a ~20GB pull. NOT the MLX one.
# qwen2.5-graphiti = qwen2.5:14b with num_ctx=8192 (Modelfile). THE fix for the
# garbage-predicate/orphan-edge symptom: Ollama defaults num_ctx to 2048 and
# SILENTLY truncates the FRONT of longer prompts -- and Graphiti's extraction
# schema+rules sit at the top, so the model never sees them and emits
# schema-shaped-but-wrong output. Verified community root cause (Jul 21 2026
# research: jangwook.net num_ctx truncation, Ollama #7741). qwen2.5:14b is the
# de-facto working local Graphiti model (Flo976/graphiti-mcp-ollama reference).
ZEP_LLM_MODEL = os.environ.get("ZEP_LLM_MODEL", "qwen2.5-graphiti")
# Graphiti ingest concurrency. helpers.py defaults SEMAPHORE_LIMIT to 20 (tuned
# for cloud APIs); against a single local Ollama (OLLAMA_NUM_PARALLEL=1) that
# many concurrent extraction calls is what wedged Ollama into a stuck-unload
# state twice. Zep's docs + the known-good local reference repo use 1-5. Set it
# BEFORE graphiti_core is imported (helpers.py reads it at import time). Env wins
# if the operator already set one.
os.environ.setdefault("SEMAPHORE_LIMIT", "3")
ZEP_EMBED_MODEL = os.environ.get("ZEP_EMBED_MODEL", "nomic-embed-text")
ZEP_EMBED_DIM = int(os.environ.get("ZEP_EMBED_DIM", "768"))  # nomic-embed-text = 768
ZEP_OLLAMA_BASE = os.environ.get("ZEP_OLLAMA_BASE", "http://localhost:11434/v1")
# The LLM (extraction) and the EMBEDDER run on DIFFERENT servers on purpose:
#   - LLM extraction  -> MLX (mlx_lm.server :8081) serving Qwen3.6-35B-A3B, a
#     frontier-class model with the structured output Graphiti requires. Running
#     the heavy extraction on MLX keeps it OFF ollama, so it can't wedge the
#     ollama instance the other arms share (that wedge killed two prior runs).
#     REQUIRES the MLX server started with --chat-template-args
#     '{"enable_thinking": false}' or every response is empty (qwen3.6 is a
#     thinking model that otherwise returns message.reasoning with content=None).
#   - Embedder -> ollama nomic-embed-text :11434 (768d), unchanged.
# Override ZEP_LLM_BASE/ZEP_LLM_MODEL to fall back to the all-ollama qwen2.5:14b
# path if MLX is unavailable.
ZEP_LLM_BASE = os.environ.get("ZEP_LLM_BASE", "http://localhost:11434/v1")
ZEP_EMBED_BASE = os.environ.get("ZEP_EMBED_BASE", ZEP_OLLAMA_BASE)
# 127.0.0.1, NOT "localhost" -- same IPv4-only trap that silently killed the
# hindsight arm. FalkorDB runs in a container whose port is forwarded to IPv4
# only; verified: 127.0.0.1:6379 answers PING with +PONG while [::1]:6379 is
# ConnectionRefused. "localhost" only works by falling back after the IPv6
# attempt fails, and anything that grabs the IPv6 wildcard intercepts it first.
ZEP_FALKOR_HOST = os.environ.get("ZEP_FALKOR_HOST", "127.0.0.1")
ZEP_FALKOR_PORT = int(os.environ.get("ZEP_FALKOR_PORT", "6379"))
ZEP_GROUP_ID = os.environ.get("ZEP_GROUP_ID", "benchmark")
_ZEP_CACHE: dict = {}
# Created at import time -- see _zep_ingest_lock(). The fleet runs FLEET_SIZE
# agents as threads sharing this module, so every ingest guard below needs a
# real lock rather than an unlocked check-then-set on a plain dict flag.
_ZEP_INGEST_LOCK = threading.Lock()


def _zep_loop():
    """One persistent event loop OWNED BY ITS OWN DEDICATED THREAD.

    graphiti-core is fully async and its aiohttp / FalkorDB clients bind to the
    loop that created them, so every coroutine must run on that SAME loop.
    But the fleet runs FLEET_SIZE agents as THREADS (fleet_runner.py's
    ThreadPoolExecutor), and calling loop.run_until_complete() from more than one
    thread raises "RuntimeError: This event loop is already running".

    That is not hypothetical: it is exactly what killed this arm. In every prior
    paid run the zep memory layer returned an ERROR on 100% of calls, and the
    harness scored those crashes as substantive retrieval losses. Verified
    reproduction (python 3.12.11): 3 threads sharing one loop -> 1 ok, 2 raise
    "This event loop is already running".

    Owning the loop on a dedicated thread and submitting work via
    run_coroutine_threadsafe (see _zep_run) is the thread-safe way to share one
    loop across N caller threads, and keeps the Graphiti client on its own loop.
    """
    loop = _ZEP_CACHE.get("loop")
    if loop is None or loop.is_closed():
        import asyncio  # noqa: PLC0415
        import threading  # noqa: PLC0415
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="zep-event-loop", daemon=True
        )
        thread.start()
        _ZEP_CACHE["loop"] = loop
        _ZEP_CACHE["loop_thread"] = thread
    return loop


def _zep_run(coro, timeout: int):
    """Submit a coroutine to the dedicated zep loop from ANY thread and wait.

    Raises concurrent.futures.TimeoutError ONLY on a genuine wall-clock timeout.
    This matters: Python 3.12 aliases asyncio.TimeoutError, socket.timeout and
    builtins.TimeoutError to the SAME class, and TimeoutError is an OSError
    subclass -- so a bare `except TimeoutError` also swallows connection refused
    / connection reset and mislabels them as a timeout. Callers must therefore
    distinguish concurrent.futures.TimeoutError (real stall) from OSError
    (service down), instead of reporting every failure as an N-second stall.
    """
    import asyncio  # noqa: PLC0415

    future = asyncio.run_coroutine_threadsafe(coro, _zep_loop())
    try:
        return future.result(timeout=timeout)
    except BaseException:
        future.cancel()
        raise


def _zep_ingest_lock():
    """Process-wide lock serializing zep ingest across the fleet's threads.

    The lock object itself is created at MODULE IMPORT time (_ZEP_INGEST_LOCK),
    not lazily -- a lazily-created lock would need a lock to create it safely,
    which is the exact check-then-set race this is here to eliminate.
    """
    return _ZEP_INGEST_LOCK


async def _zep_build_client():
    """Build the FULLY LOCAL Graphiti client (Ollama LLM + embeddings + reranker +
    local FalkorDB), create indices, and CLEAR this group's prior graph (FIX 2:
    no cross-trial contamination). Three cloud phone-home traps are closed:
    OpenAIGenericClient (not OpenAIClient), explicit local reranker, explicit
    local embedder."""
    from graphiti_core import Graphiti  # noqa: PLC0415
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: PLC0415
    from graphiti_core.llm_client.config import LLMConfig  # noqa: PLC0415
    from graphiti_core.llm_client.openai_generic_client import (  # noqa: PLC0415
        OpenAIGenericClient,
    )
    from graphiti_core.embedder.openai import (  # noqa: PLC0415
        OpenAIEmbedder,
        OpenAIEmbedderConfig,
    )
    from graphiti_core.cross_encoder.openai_reranker_client import (  # noqa: PLC0415
        OpenAIRerankerClient,
    )

    llm_config = LLMConfig(
        api_key="ollama",          # placeholder; Ollama ignores it (must be non-empty)
        model=ZEP_LLM_MODEL,
        small_model=ZEP_LLM_MODEL,
        base_url=ZEP_LLM_BASE,
    )
    # DEFAULT json_schema mode: Ollama constrains generation to the schema via
    # a GBNF grammar (llama.cpp), which FORCES the model to emit a valid instance
    # -- the model cannot echo the schema definition back. Tried json_object
    # (schema injected into the prompt) and qwen2.5:14b fumbled it: it returned
    # the schema's $defs instead of data -> "ExtractedEntities edges Field
    # required" validation error. json_schema + Ollama's grammar enforcement is
    # the correct combination for a local model.
    llm_client = OpenAIGenericClient(config=llm_config)
    falkor_driver = FalkorDriver(
        host=ZEP_FALKOR_HOST, port=ZEP_FALKOR_PORT, username=None, password=None
    )
    graphiti = Graphiti(
        graph_driver=falkor_driver,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",
                embedding_model=ZEP_EMBED_MODEL,
                embedding_dim=ZEP_EMBED_DIM,
                base_url=ZEP_EMBED_BASE,
            )
        ),
        # MUST set cross_encoder EXPLICITLY. If omitted, Graphiti EAGERLY builds
        # OpenAIRerankerClient() at init (no config) which demands OPENAI_API_KEY
        # and crashes construction -- this is eager, not latent. Reuse the SAME
        # local ollama client so the reranker stays fully local (no phone-home).
        cross_encoder=OpenAIRerankerClient(config=llm_config, client=llm_client.client),
    )
    await graphiti.build_indices_and_constraints()
    # FIX 2: clear any prior data in THIS group before ingest so a persistent
    # FalkorDB volume can't leak earlier trials' facts into this run. clear_data
    # (verified in graphiti-core 0.29.2) drops exactly this group's nodes/edges.
    try:
        from graphiti_core.utils.maintenance.graph_data_operations import (  # noqa: PLC0415
            clear_data,
        )
        await clear_data(graphiti.driver, group_ids=[ZEP_GROUP_ID])
    except Exception:  # noqa: BLE001  -- a fresh group_id per run still isolates
        pass
    return graphiti


async def _zep_ingest(graphiti, facts: list) -> None:
    """Ingest the SHARED corpus. Each fact is one text episode; Graphiti runs LLM
    extraction to build entity/edge nodes. FIX 3: assign a monotonically
    increasing reference_time in corpus order (oldest first) so Graphiti has a
    real chronology for temporal reasoning, instead of flattening all to now()."""
    from datetime import datetime, timezone, timedelta  # noqa: PLC0415
    from graphiti_core.nodes import EpisodeType  # noqa: PLC0415

    # Anchor the chronology so the LAST fact is "most recent" but still in the
    # past; space episodes an hour apart in corpus order (a stable, fair ordering
    # every arm's corpus shares -- _rag_load_facts returns a deterministic order).
    base = datetime.now(timezone.utc) - timedelta(hours=len(facts) + 1)
    for i, fact in enumerate(facts):
        await graphiti.add_episode(
            name=f"fact_{i}",
            episode_body=fact,
            source=EpisodeType.text,
            source_description="benchmark corpus",
            reference_time=base + timedelta(hours=i),
            group_id=ZEP_GROUP_ID,
        )


async def _zep_search(graphiti, query: str, top_k: int) -> list:
    """RECOMMENDED retrieval: hybrid semantic + BM25 + graph RRF. Returns
    EntityEdge objects; the human-readable fact is r.fact."""
    results = await graphiti.search(query, group_ids=[ZEP_GROUP_ID], num_results=top_k)
    return [r.fact for r in results if getattr(r, "fact", None)]


def _zep_ensure_ingested():
    """Build the client + ingest the shared corpus ONCE per process (idempotent).
    Returns the live Graphiti client, or raises on a missing dep / dead service.

    LOCKED: the fleet calls this from FLEET_SIZE threads at once. The previous
    unlocked check-then-set let all three threads ingest the same corpus
    concurrently -- triple LLM extraction work against a serialized ollama
    (OLLAMA_NUM_PARALLEL=1), which is the most likely source of the recorded
    ingest stalls. One thread ingests; the others wait and reuse the result.
    """
    if _ZEP_CACHE.get("ingested"):
        return _ZEP_CACHE["client"]
    with _zep_ingest_lock():
        # Re-check inside the lock: another thread may have finished while we waited.
        if _ZEP_CACHE.get("ingested"):
            return _ZEP_CACHE["client"]
        graphiti = _ZEP_CACHE.get("client")
        if graphiti is None:
            graphiti = _zep_run(_zep_build_client(), timeout=ZEP_INGEST_TIMEOUT_S)
            _ZEP_CACHE["client"] = graphiti
        facts = _rag_load_facts()  # SAME shared corpus -- never re-author.
        if not facts:
            raise RuntimeError("shared corpus empty -- no memories to ingest")
        # The long call (one LLM extraction per fact). Bounded so a stall can't
        # hang, but on the INGEST budget -- this runs in pre-warm, outside the
        # paid loop, so it must be allowed to finish rather than leave a partial
        # graph that retrieves badly and looks like a product failure.
        _zep_run(_zep_ingest(graphiti, facts), timeout=ZEP_INGEST_TIMEOUT_S)
        _ZEP_CACHE["ingested"] = True
        return graphiti


def tool_zep_search(_repo: Path, args: dict) -> str:
    """Graphiti temporal-knowledge-graph retrieval over the SAME seeded corpus:
    extract entities/edges on ingest, then hybrid (semantic + BM25 + graph RRF)
    search. Same facts, same top-K budget (ARM_TOPK), same 6000-char cap as the
    RAG arm -- only the retrieval mechanism differs (a real knowledge graph)."""
    query = (args or {}).get("query") or ""
    if not query.strip():
        return "ERROR: zep_search requires a non-empty 'query'."
    try:
        import graphiti_core  # noqa: F401,PLC0415
    except ImportError:
        return (
            "ERROR: graphiti-core is not installed. Run: "
            'pip install "graphiti-core[falkordb]==0.29.2"'
        )
    # NOTE on exception ordering below: concurrent.futures.TimeoutError is the
    # ONLY signal of a genuine wall-clock stall here. builtins.TimeoutError must
    # NOT be used for that -- in python 3.12 asyncio.TimeoutError, socket.timeout
    # and builtins.TimeoutError are the SAME class, and it subclasses OSError, so
    # `except TimeoutError` also catches connection-refused/reset. The old code
    # did exactly that and reported sub-second socket failures as
    # "Graphiti ingest exceeded 600s" -- a false performance claim about a
    # third-party product, in runs whose total wall clock was ~100s.
    import concurrent.futures  # noqa: PLC0415

    try:
        graphiti = _zep_ensure_ingested()
    except concurrent.futures.TimeoutError:
        return (
            f"ERROR: Graphiti ingest exceeded {ZEP_INGEST_TIMEOUT_S}s and was "
            "aborted (stalled extraction). Arm ERRORED rather than hang the run."
        )
    except OSError as exc:
        return (
            f"ERROR: Graphiti backend unreachable ({type(exc).__name__}: {exc}). "
            f"Is Ollama running (models {ZEP_LLM_MODEL} + {ZEP_EMBED_MODEL} "
            f"pulled) and FalkorDB listening on "
            f"{ZEP_FALKOR_HOST}:{ZEP_FALKOR_PORT}?"
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"ERROR: Graphiti index unavailable ({type(exc).__name__}: {exc}). "
            f"Is Ollama running (models {ZEP_LLM_MODEL} + {ZEP_EMBED_MODEL} "
            f"pulled) and FalkorDB listening on "
            f"{ZEP_FALKOR_HOST}:{ZEP_FALKOR_PORT}?"
        )
    try:
        facts = _zep_run(_zep_search(graphiti, query, ZEP_TOPK), timeout=ZEP_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return (
            f"ERROR: Graphiti search exceeded {ZEP_TIMEOUT_S}s and was aborted. "
            "Arm ERRORED rather than hang the run."
        )
    except OSError as exc:
        return f"ERROR: Graphiti search backend unreachable ({type(exc).__name__}: {exc})."
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Graphiti search failed ({type(exc).__name__}: {exc})."
    if not facts:
        return "(Graphiti knowledge graph returned no matching facts.)"
    lines = [f"[{i}] {body}" for i, body in enumerate(facts, 1)]
    header = (
        f"Top-{ZEP_TOPK} facts from the temporal knowledge graph (hybrid "
        f"semantic + BM25 + graph RRF, Graphiti):\n"
    )
    return (header + "\n".join(lines))[:6000]


ZEP_TOOL = {
    "zep_search": (
        tool_zep_search,
        "Knowledge-graph memory search: retrieve the most relevant facts about "
        "this failure from the team's history, stored as a temporal entity/edge "
        "graph. Returns the top-K facts by hybrid semantic + keyword + graph "
        "relevance. Use it to find prior context that explains the failure -- the "
        "cause may live in an earlier change in a different service.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}


def build_tool_specs(mode: str):
    tools = dict(BASE_TOOLS)
    if mode == "vestige":
        tools.update(VESTIGE_TOOL)
    elif mode == "rag":
        tools.update(RAG_TOOL)
    elif mode == "mem0":
        tools.update(MEM0_TOOL)
    elif mode == "supermemory":
        tools.update(SUPERMEMORY_TOOL)
    elif mode == "hindsight":
        tools.update(HINDSIGHT_TOOL)
    elif mode == "zep":
        tools.update(ZEP_TOOL)
    specs = [
        {"name": name, "description": desc, "input_schema": schema}
        for name, (_fn, desc, schema) in tools.items()
    ]
    return specs, tools


def system_prompt(mode: str) -> str:
    base = (
        "You are a coding agent working inside a small TypeScript monorepo (the "
        "Meridian platform: identity-service mints session tokens, gateway-service "
        "forwards them, ledger-service verifies them and charges). Your task: "
        f"{TASK}. The end-to-end charge test currently fails. Reproduce it first "
        "with run_tests, investigate across the services, apply a fix with "
        "write_file, then re-run run_tests to verify. Call finish(fixed=true) "
        "ONLY when run_tests exits 0. Be efficient: do not re-read files you have "
        "already read."
    )
    if mode == "vestige":
        base += (
            " You also have vestige_backfill, which reaches backward from the "
            "current failure to the quiet earlier change that caused it -- possibly "
            "a past decision in a different service, one that a search of the "
            "current code cannot reveal. If the cause is not obvious at the failure "
            "site, use it before guessing."
        )
    elif mode == "rag":
        base += (
            " You also have rag_search, which searches the team's memory of past "
            "notes, incidents, and decisions and returns the most relevant ones -- "
            "possibly a past decision in a different service, one that a search of "
            "the current code cannot reveal. If the cause is not obvious at the "
            "failure site, use it before guessing."
        )
    elif mode == "mem0":
        base += (
            " You also have mem0_search, which searches the team's mem0 memory of "
            "past notes, incidents, and decisions and returns the most relevant "
            "ones -- possibly a past decision in a different service, one that a "
            "search of the current code cannot reveal. If the cause is not obvious "
            "at the failure site, use it before guessing."
        )
    elif mode == "supermemory":
        base += (
            " You also have supermemory_search, which searches the team's "
            "supermemory store of past notes, incidents, and decisions and returns "
            "the most relevant ones -- possibly a past decision in a different "
            "service, one that a search of the current code cannot reveal. If the "
            "cause is not obvious at the failure site, use it before guessing."
        )
    elif mode == "hindsight":
        base += (
            " You also have hindsight_search, which searches the team's Hindsight "
            "agent-memory of past notes, incidents, and decisions and returns the "
            "most relevant ones -- possibly a past decision in a different service, "
            "one that a search of the current code cannot reveal. If the cause is "
            "not obvious at the failure site, use it before guessing."
        )
    elif mode == "zep":
        base += (
            " You also have zep_search, which searches the team's Zep/Graphiti "
            "temporal knowledge graph of past notes, incidents, and decisions and "
            "returns the most relevant facts -- possibly a past decision in a "
            "different service, one that a search of the current code cannot "
            "reveal. If the cause is not obvious at the failure site, use it "
            "before guessing."
        )
    return base


# --- Provider adapters -------------------------------------------------------
#
# Each provider exposes the SAME contract to the loop below:
#   turn(system, specs, transcript) -> (calls, in_tokens, out_tokens)
# where `calls` is a normalized list of dicts: {id, name, input}. Each adapter
# owns its own SDK, message shape, tool-spec format, and usage field names, and
# mutates `transcript` (its own native message list) in place. Neither
# fabricates anything: a missing key surfaces as a clean ERROR via fail().


class AnthropicProvider:
    """Claude via the Anthropic Messages API."""

    def __init__(self) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            fail("the 'anthropic' package is not installed. Run: pip install anthropic")
        self._sdk = anthropic
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile). Missing
        # credentials raise AuthenticationError on the first call -> clean ERROR.
        self.client = anthropic.Anthropic()

    def new_transcript(self, first_user: str) -> list:
        return [{"role": "user", "content": first_user}]

    def tool_specs(self, specs: list) -> list:
        return [
            {"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]}
            for s in specs
        ]

    def turn(self, system: str, specs: list, transcript: list):
        try:
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=self.tool_specs(specs),
                messages=transcript,
            )
        except self._sdk.AuthenticationError:
            fail(
                "Claude API authentication failed. Set ANTHROPIC_API_KEY (or run "
                "`ant auth login`). No numbers were fabricated."
            )
        except self._sdk.APIError as exc:
            fail(f"Claude API error: {exc}. No numbers were fabricated.")

        in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        transcript.append({"role": "assistant", "content": resp.content})
        calls = [
            {"id": b.id, "name": b.name, "input": b.input or {}}
            for b in resp.content
            if b.type == "tool_use"
        ]
        return calls, in_tok, out_tok

    def add_no_tool_nudge(self, transcript: list) -> None:
        transcript.append({"role": "user", "content": "Use a tool or call finish."})

    def add_tool_results(self, transcript: list, results: list) -> None:
        # results: [{id, content}]
        transcript.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                    for r in results
                ],
            }
        )


class OpenAIProvider:
    """GPT-5.6 Sol via the OpenAI RESPONSES API (/v1/responses).

    Sol is a reasoning model: function tools are NOT supported on the legacy
    /v1/chat/completions endpoint unless reasoning is disabled -- so we use the
    Responses API, which is the recommended path for reasoning models AND keeps
    Sol's reasoning ON (the whole point of the demo). Field shapes verified
    against the installed openai SDK types (Responses API):
      - tool spec:        {type:"function", name, parameters, strict}
      - a tool call item: {type:"function_call", call_id, name, arguments(str)}
      - a tool result:    {type:"function_call_output", call_id, output(str)}
      - usage:            resp.usage.input_tokens / output_tokens
      - system prompt:    the `instructions=` argument
      - conversation:     the `input=` list (we accumulate items across turns)
    """

    # Set True if any turn's response was truncated (status=incomplete). Surfaced
    # in the result JSON so a degraded run is never mistaken for a clean one.
    saw_truncation = False

    def __init__(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            fail("the 'openai' package is not installed. Run: pip install openai")
        self._sdk = openai
        if not os.environ.get("OPENAI_API_KEY"):
            fail(
                "OPENAI_API_KEY is not set. Set it before running the OpenAI path. "
                "No numbers were fabricated."
            )
        self.client = openai.OpenAI()
        # Reasoning effort for Sol. Default `xhigh` — this is a HARD multi-service
        # debugging task, which is exactly the regime where you run Sol at
        # xhigh/max, and where the token gap between reasoning-without-the-causal-
        # context and getting the flashback from Vestige actually opens up.
        # Accepts: none|low|medium|high|xhigh|max. Override with OPENAI_REASONING_EFFORT.
        self.effort = os.environ.get("OPENAI_REASONING_EFFORT", "xhigh").strip().lower()
        # Reasoning SUMMARY: request the model's human-readable summary of its
        # chain of thought (the sanctioned API surface — NOT raw CoT, which
        # OpenAI forbids extracting). This is what lets the demo SHOW *why* an
        # agent chose a key: the anarchy agent rationalizing a guess vs the sync
        # agent citing the retrieved production fact. Values: detailed|auto|
        # concise|none. Default detailed so every run self-documents its reasoning.
        # We degrade gracefully (detailed -> auto -> none) if the org isn't
        # entitled to a given summary level, so a run never dies over this.
        self.summary = os.environ.get("OPENAI_REASONING_SUMMARY", "detailed").strip().lower()

    def new_transcript(self, first_user: str) -> list:
        # The Responses `input` list. We start with the first user turn and
        # append the model's own output items + our function_call_output items.
        return [{"role": "user", "content": first_user}]

    def tool_specs(self, specs: list) -> list:
        # Responses API function tools are FLAT (name/parameters at top level),
        # unlike chat.completions which nests them under a `function` key.
        return [
            {
                "type": "function",
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
                "strict": False,
            }
            for s in specs
        ]

    def _reasoning_param(self):
        """Build the reasoning param, including the summary level unless disabled."""
        r = {"effort": self.effort}
        if self.summary and self.summary != "none":
            r["summary"] = self.summary
        return r

    def turn(self, system: str, specs: list, transcript: list):
        def _create(reasoning):
            return self.client.responses.create(
                model=MODEL,
                instructions=system,
                input=transcript,
                tools=self.tool_specs(specs),
                tool_choice="auto",
                reasoning=reasoning,
                max_output_tokens=MAX_TOKENS,
            )

        try:
            try:
                resp = _create(self._reasoning_param())
            except self._sdk.BadRequestError:
                # Some orgs/models aren't entitled to a given summary level. Fall
                # back detailed -> auto -> no-summary rather than kill the run.
                if self.summary in ("detailed", "concise"):
                    self.summary = "auto"
                    resp = _create(self._reasoning_param())
                elif self.summary == "auto":
                    self.summary = "none"
                    resp = _create({"effort": self.effort})
                else:
                    raise
        except self._sdk.AuthenticationError:
            fail(
                "OpenAI API authentication failed. Check OPENAI_API_KEY. "
                "No numbers were fabricated."
            )
        except self._sdk.APIError as exc:
            fail(f"OpenAI API error: {exc}. No numbers were fabricated.")

        usage = resp.usage
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0

        # Surface truncation. If the response was cut off at max_output_tokens the
        # reasoning chain is incomplete and the measured numbers are degraded --
        # make that LOUD rather than let a truncated run masquerade as a clean one.
        if getattr(resp, "status", None) == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", "unknown")
            print(
                f"  ! WARNING: model response was truncated (status=incomplete, "
                f"reason={reason}). Raise MAX_TOKENS -- these numbers are degraded.",
                file=sys.stderr,
            )
            OpenAIProvider.saw_truncation = True

        # Persist EVERY output item (reasoning, message, function_call) back into
        # the input list so the next turn has full context and the
        # function_call_output items reference valid call_ids. The API returns
        # output-only fields (e.g. `status`) that it REJECTS when echoed back as
        # input ("Unknown parameter: input[N].status"), and a null `id` is also
        # rejected -- so strip those before re-sending.
        calls = []
        for item in resp.output:
            dumped = item.model_dump(exclude_none=True)
            dumped.pop("status", None)
            transcript.append(dumped)
            if getattr(item, "type", None) == "function_call":
                try:
                    parsed = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                calls.append({"id": item.call_id, "name": item.name, "input": parsed})
        return calls, in_tok, out_tok

    def add_no_tool_nudge(self, transcript: list) -> None:
        transcript.append({"role": "user", "content": "Use a tool or call finish."})

    def add_tool_results(self, transcript: list, results: list) -> None:
        # Responses API: each tool result is a function_call_output input item.
        for r in results:
            transcript.append(
                {
                    "type": "function_call_output",
                    "call_id": r["id"],
                    "output": r["content"],
                }
            )


class OpenRouterProvider:
    """Kimi K2.7 Code (or any model) via OpenRouter's OpenAI-compatible
    /v1/chat/completions API.

    This is the CROSS-MODEL arm: it proves the retrieval fracture is not a quirk
    of one lab's model. It uses the standard chat.completions loop (NOT the
    OpenAI Responses API), and crucially it captures the model's CHAIN-OF-THOUGHT
    from `message.reasoning` on every turn -- the narrative that shows *why* an
    agent chose a key (the memoryless agent rationalizing a guess vs the Vestige
    agent citing the retrieved production fact). OpenAI gates that narration
    behind org verification; OpenRouter/Kimi return it ungated.

    Field shapes verified Jul 17 2026:
      - OpenRouter reasoning request param: top-level `reasoning={"effort": ...}`
        (openrouter.ai/docs/use-cases/reasoning-tokens). effort in
        max|xhigh|high|medium|low|minimal|none.
      - reasoning is returned in `message.reasoning` (NOT `reasoning_content`;
        that is official-Moonshot's field name -- OpenRouter normalizes to
        `message.reasoning`).
      - Kimi K2.7 Code forces thinking ON regardless, so reasoning always appears.
      - tool spec (chat.completions): {type:"function", function:{name,
        description, parameters}}.
      - a tool call:   choice.message.tool_calls[i] with .id, .function.name,
        .function.arguments (JSON string).
      - a tool result: {role:"tool", tool_call_id, content}.
      - usage:         resp.usage.prompt_tokens / completion_tokens.
    """

    # Every turn's reasoning text, in order -- so the run JSON can SHOW the CoT.
    def __init__(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            fail("the 'openai' package is not installed. Run: pip install openai")
        self._sdk = openai
        # Two transports share this class (both OpenAI-compatible chat.completions):
        #   PROVIDER=openrouter -> openrouter.ai relay (OPENROUTER_API_KEY)
        #   PROVIDER=moonshot   -> Moonshot DIRECT api.moonshot.ai/v1 (MOONSHOT_API_KEY),
        #                          used when OpenRouter's K3 relay is upstream-rate-limited.
        # Moonshot returns chain-of-thought in `reasoning_content`; OpenRouter
        # normalizes to `message.reasoning`. The turn() extractor already reads
        # both (message.reasoning, then model_extra reasoning/reasoning_content),
        # so no other change is needed for the CoT capture to work on either.
        if PROVIDER == "moonshot":
            key = os.environ.get("MOONSHOT_API_KEY")
            if not key:
                fail(
                    "MOONSHOT_API_KEY is not set. Get one at platform.moonshot.ai "
                    "and export it. No numbers were fabricated."
                )
            base_url = os.environ.get(
                "MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"
            )
        elif PROVIDER == "deepseek":
            # DeepSeek DIRECT on Sam's own key (never OpenRouter for DeepSeek).
            # base_url is https://api.deepseek.com WITHOUT /v1 (verified live vs
            # api-docs.deepseek.com Jul 19 2026). CoT returns in reasoning_content,
            # which turn() already extracts via model_extra.
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                fail(
                    "DEEPSEEK_API_KEY is not set. Get one at platform.deepseek.com "
                    "and export it. No numbers were fabricated."
                )
            base_url = os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            )
        else:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                fail(
                    "OPENROUTER_API_KEY is not set. Get one at openrouter.ai/keys "
                    "and export it. No numbers were fabricated."
                )
            base_url = os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
        self.client = openai.OpenAI(api_key=key, base_url=base_url)
        # Reasoning effort. Kimi forces thinking on; effort tunes depth.
        # Accepts: none|minimal|low|medium|high|xhigh|max. Default `high`.
        self.effort = os.environ.get(
            "OPENROUTER_REASONING_EFFORT", "high"
        ).strip().lower()
        # Collected chain-of-thought, one entry per turn that produced reasoning.
        self.reasoning_log: list = []

    def new_transcript(self, first_user: str) -> list:
        return [{"role": "user", "content": first_user}]

    def tool_specs(self, specs: list) -> list:
        # chat.completions nests the tool under a `function` key.
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in specs
        ]

    def turn(self, system: str, specs: list, transcript: list):
        # System prompt rides as the first message on each call (cheap; chat API
        # has no separate system arg on this path). Only inject once.
        if not transcript or transcript[0].get("role") != "system":
            transcript.insert(0, {"role": "system", "content": system})

        reasoning_param = {}
        if self.effort and self.effort != "none":
            reasoning_param = {"reasoning": {"effort": self.effort}}
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=transcript,
                tools=self.tool_specs(specs),
                tool_choice="auto",
                max_tokens=MAX_TOKENS,
                extra_body=reasoning_param,
            )
        except self._sdk.AuthenticationError:
            _keyname = {
                "moonshot": "MOONSHOT_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
            }.get(PROVIDER, "OPENROUTER_API_KEY")
            fail(
                f"{PROVIDER} authentication failed. Check {_keyname}. "
                "No numbers were fabricated."
            )
        except self._sdk.APIError as exc:
            fail(f"{PROVIDER} API error: {exc}. No numbers were fabricated.")

        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        msg = resp.choices[0].message
        # Capture the chain-of-thought. Kimi/GLM/Sol return it in message.reasoning;
        # Anthropic Claude (Fable 5, Mythos) uses ADAPTIVE thinking and never emits
        # raw CoT, so OpenRouter surfaces a SUMMARIZED trace in message.reasoning_details
        # (an array of {type: reasoning.summary|reasoning.text, text|summary|data}).
        # Read reasoning first, then fold in reasoning_details so every provider's
        # trace is captured (verified against Anthropic extended-thinking docs +
        # OpenRouter reasoning-tokens docs, Jul 19 2026).
        reasoning = getattr(msg, "reasoning", None)
        if not reasoning:
            rd = getattr(msg, "reasoning_details", None)
            if rd is None and isinstance(getattr(msg, "model_extra", None), dict):
                rd = msg.model_extra.get("reasoning_details")
            if isinstance(rd, list) and rd:
                parts = []
                for d in rd:
                    if isinstance(d, dict):
                        parts.append(d.get("text") or d.get("summary") or d.get("data") or "")
                    elif isinstance(d, str):
                        parts.append(d)
                reasoning = " ".join(p for p in parts if p).strip() or None
        if reasoning is None and isinstance(getattr(msg, "model_extra", None), dict):
            # Some SDK versions stash unknown fields under model_extra.
            reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get(
                "reasoning_content"
            )
        if reasoning:
            self.reasoning_log.append(reasoning)

        # Echo the assistant message back into the transcript, preserving
        # tool_calls (and reasoning, so multi-turn thinking is retained).
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        if reasoning:
            # Round-trip reasoning so the model keeps its own thread (Kimi
            # preserve-thinking). Harmless if the API ignores it on input.
            assistant_msg["reasoning"] = reasoning
        transcript.append(assistant_msg)

        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                parsed = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                parsed = {}
            calls.append({"id": tc.id, "name": tc.function.name, "input": parsed})
        return calls, in_tok, out_tok

    def add_no_tool_nudge(self, transcript: list) -> None:
        transcript.append({"role": "user", "content": "Use a tool or call finish."})

    def add_tool_results(self, transcript: list, results: list) -> None:
        # chat.completions: each tool result is a {role:"tool"} message.
        for r in results:
            transcript.append(
                {"role": "tool", "tool_call_id": r["id"], "content": r["content"]}
            )


class MockProvider:
    """A $0, no-network SCRIPTED provider used ONLY to verify arm wiring end to
    end (ingest -> retrieve -> write results) without spending a cent on a real
    model. It is NEVER used for a measured run.

    Contract-identical to the real providers (new_transcript / tool_specs / turn
    / add_no_tool_nudge / add_tool_results), but turn() is scripted by a per-
    instance step counter instead of an API call:
      step 1: if the arm exposes a memory/retrieval tool, CALL it once with a
              query about the failure (this drives _rag_load_facts ingest +
              retrieval for rag/mem0, or the backfill/hub read for vestige/sync).
      step 2: call finish(fixed=false) so the loop exits cleanly and the run
              JSON is written. We deliberately do NOT fix the repo -- the mock
              proves plumbing, not problem-solving, so tests_pass_after_run is
              expected False and that is fine for a wiring check.
    The retrieved tool output is printed to stderr so the mock run visibly shows
    each arm returning real memories (or a clean ERROR if a service is down)."""

    # The memory/retrieval tool each arm registers, in priority order. The mock
    # calls the first one present in the arm's specs.
    _MEMORY_TOOLS = ("vestige_backfill", "rag_search", "mem0_search",
                     "zep_search", "hindsight_search", "supermemory_search")

    def __init__(self) -> None:
        self._step = 0

    def new_transcript(self, first_user: str) -> list:
        return [{"role": "user", "content": first_user}]

    def tool_specs(self, specs: list) -> list:
        return specs

    def turn(self, system: str, specs: list, transcript: list):
        self._step += 1
        names = {s["name"] for s in specs}
        # Step 1: exercise the arm's memory tool if it has one.
        if self._step == 1:
            for tname in self._MEMORY_TOOLS:
                if tname in names:
                    schema = next(s["input_schema"] for s in specs if s["name"] == tname)
                    args = {}
                    if "query" in (schema.get("properties") or {}):
                        args["query"] = (
                            "Production token verification is failing after a key "
                            "rotation. Which signing key is production-safe and why?"
                        )
                    print(f"  [mock] calling {tname}({args})", file=sys.stderr)
                    return ([{"id": f"mock_{self._step}", "name": tname, "input": args}], 0, 0)
            # Anarchy (no memory tool): nothing to exercise -> fall through to finish.
        # Step 2+ (or anarchy step 1): finish so the loop exits and JSON is written.
        print("  [mock] calling finish(fixed=false) to close the run", file=sys.stderr)
        return (
            [{
                "id": f"mock_{self._step}",
                "name": "finish",
                "input": {"fixed": False, "summary": "mock wiring check -- not a real fix"},
            }],
            0,
            0,
        )

    def add_no_tool_nudge(self, transcript: list) -> None:
        pass

    def add_tool_results(self, transcript: list, results: list) -> None:
        # Surface each retrieved memory to stderr so the mock run VISIBLY proves
        # the arm retrieved something real (or shows a clean ERROR if down).
        for r in results:
            body = str(r.get("content", ""))
            preview = body if len(body) <= 400 else body[:400] + " ...[truncated]"
            print(f"  [mock] tool result:\n    {preview}", file=sys.stderr)


def _make_provider():
    if PROVIDER == "mock":
        return MockProvider()
    if PROVIDER == "openai":
        return OpenAIProvider()
    if PROVIDER in ("openrouter", "moonshot", "deepseek"):
        # Same OpenAI-compatible chat.completions class; the constructor picks
        # the key/base_url from PROVIDER (OpenRouter relay vs Moonshot direct vs
        # DeepSeek direct).
        return OpenRouterProvider()
    return AnthropicProvider()


def run(mode: str, repo: Path, out_path: Path) -> None:
    provider = _make_provider()

    specs, tools = build_tool_specs(mode)
    # specs came back in Anthropic's {name, description, input_schema} shape from
    # build_tool_specs; each adapter re-maps it to its own format.
    transcript = provider.new_transcript(f"The repo is at {repo}. Task: {TASK}.")

    total_in = 0
    total_out = 0
    iterations = 0
    status = "looped"  # default if we hit the cap without finishing
    finish_summary = ""
    t0 = time.time()

    for iterations in range(1, MAX_ITERATIONS + 1):
        calls, in_tok, out_tok = provider.turn(system_prompt(mode), specs, transcript)
        total_in += in_tok
        total_out += out_tok

        if not calls:
            provider.add_no_tool_nudge(transcript)
            continue

        results = []
        did_finish = False
        for c in calls:
            if c["name"] == "finish":
                did_finish = True
                fixed = bool(c["input"].get("fixed"))
                finish_summary = str(c["input"].get("summary", ""))
                status = "fixed_claimed" if fixed else "gave_up"
                results.append({"id": c["id"], "content": "acknowledged"})
                continue
            fn = tools[c["name"]][0]
            try:
                output = fn(repo, c["input"] or {})
            except Exception as exc:  # noqa: BLE001
                output = f"ERROR running {c['name']}: {exc}"
            results.append({"id": c["id"], "content": output})

        provider.add_tool_results(transcript, results)
        if did_finish:
            break

    wall = time.time() - t0

    # Ground-truth verification: run the tests ourselves; don't trust the claim.
    try:
        verify = _run_tests(repo)
        really_fixed = verify.returncode == 0
    except subprocess.TimeoutExpired:
        really_fixed = False

    if status == "fixed_claimed":
        status = "fixed" if really_fixed else "hallucinated"
    elif really_fixed:
        status = "fixed"

    cost = (total_in / 1_000_000) * COST_PER_MTOK_INPUT + (
        total_out / 1_000_000
    ) * COST_PER_MTOK_OUTPUT

    result = {
        "mode": mode,
        "provider": PROVIDER,
        "model": MODEL,
        "task": TASK,
        "repo": str(repo),
        "measured": True,
        "tokens": {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        },
        "cost_usd": round(cost, 6),
        "cost_rate": {
            "input_per_mtok": COST_PER_MTOK_INPUT,
            "output_per_mtok": COST_PER_MTOK_OUTPUT,
        },
        "wall_clock_seconds": round(wall, 2),
        "iterations": iterations,
        "max_iterations": MAX_ITERATIONS,
        "status": status,
        "tests_pass_after_run": really_fixed,
        "reasoning_effort": (
            os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")
            if PROVIDER == "openai"
            else os.environ.get("OPENROUTER_REASONING_EFFORT", "high")
            if PROVIDER in ("openrouter", "moonshot", "deepseek")
            else None
        ),
        "truncated": bool(getattr(OpenAIProvider, "saw_truncation", False)),
        "finish_summary": finish_summary,
        # The captured chain-of-thought (openrouter/kimi only) -- the narrative
        # that SHOWS why the agent chose a key. Empty for providers whose reasoning
        # is not returned by the API (e.g. OpenAI without a verified org).
        "reasoning_log": getattr(provider, "reasoning_log", []),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["baseline", "rag", "vestige"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        fail(f"repo not found: {repo}")
    run(args.mode, repo, Path(args.out).resolve())


if __name__ == "__main__":
    main()

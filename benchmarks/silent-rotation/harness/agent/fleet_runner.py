#!/usr/bin/env python3
"""Real, measured MULTI-AGENT fleet harness (anarchy vs synaptic-hub-sync).

This is the multi-agent counterpart to runner.py. It runs a FLEET of N real
coding-agent loops against the SAME torture-v2 bug and measures what happens
when they work WITHOUT a shared memory (anarchy) versus WITH Vestige as a
shared coordination bus (sync).

Nothing here is scripted to fail or scripted to win. Each agent is a real tool
loop calling a real model (Claude or GPT-5.6 Sol) and we sum the ACTUAL tokens
the API reports. The anarchy failure and the sync success are both EMERGENT:
they fall out of the torture-v2 bug being *direction-ambiguous*, not out of any
hardcoded outcome. When a required key is missing we ERROR (never fabricate).

------------------------------------------------------------------------------
WHY THE FLEET SPLITS (the honest mechanism)
------------------------------------------------------------------------------
torture-v2's bug is a CANONICAL-DIGEST FIELD-ORDER DRIFT:

  identity-service/src/canonical.ts  signs   over  [sub, plan, exp, iat]
  ledger-service/src/canonical.ts    verifies over  [sub, plan, iat, exp]

Both files are internally valid and self-consistent. NEITHER states which order
is authoritative. So an agent that only reads the code sees TWO equally-plausible
fixes:

  (A) reorder LEDGER to match identity   -> correct: identity is the issuer and
                                            every already-minted production token
                                            is signed its way; the verifier adapts.
  (B) reorder IDENTITY to match ledger   -> WRONG: it makes THIS test pass but
                                            invalidates every token already in
                                            the wild (the direction only memory
                                            can disambiguate).

* ANARCHY (no shared memory): each of the N agents works in its OWN isolated
  checkout and independently guesses a direction. Some pick (A), some pick (B).
  Each isolated checkout may even go green on its own. Then we INTEGRATE their
  edits into one shared tree with a real 3-way merge. Divergent directions do
  NOT compose: two agents editing the same canonical.ts differently, or editing
  OPPOSITE files, is a genuine textual conflict. The integrated repo fails.
  That is the emergent anarchy: N smart agents, no shared source of truth, work
  that collides.

* SYNC (Vestige shared bus): the N agents share ONE Vestige DB seeded with the
  project history. Before editing, each agent calls vestige_backfill / recall,
  reaches back to the 4-day-old identity-service canonical-order decision via the
  canonical_digest join, and learns the DIRECTION: identity is source of truth,
  ledger must adapt. They converge on the SAME fix (A). The merge is clean. The
  integrated repo goes green. Vestige did not write the code; it removed the
  ambiguity that made the fleet diverge.

------------------------------------------------------------------------------
This module reuses runner.py's real tool implementations and provider adapters
verbatim (same measured token accounting, same ground-truth test verification),
so the per-agent loop is identical to the audited single-agent harness -- only
the fan-out, the shared bus, and the integration merge are new.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse the audited single-agent building blocks verbatim. runner.py lives next
# to this file; import its real tools, provider adapters, and helpers so the
# per-agent loop is byte-identical to the single-agent harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402  (local module)


# --- Fleet configuration (env-driven, documented in README) -----------------

# The canonical set of valid fleet modes -- referenced by BOTH the run_fleet
# guard and the argparse choices so a new arm is added in exactly ONE place.
# anarchy/rag/sync are the originals; mem0 is the mem0ai competitor arm. Add
# zep/hindsight/supermemory here as each arm's service is stood up + verified.
FLEET_MODES = ("anarchy", "rag", "sync", "mem0", "supermemory", "hindsight", "zep",
               # Pre-registered ablation arms (PREREGISTRATION.md). Each isolates
               # ONE axis of the sync arm's three-way bundle:
               "rag-bus",       # typed query, no edge, bus ON  (pure coordination)
               "sync-noedge",   # event anchor, EDGE STRIPPED, bus ON
               "anchor-dense")  # event anchor, no edge, no bus (competitor stack)

# Every arm's memory-retrieval tool, in one place. Used for arm-agnostic
# retrieval telemetry so a dead competitor backend can never again be scored as
# a substantive retrieval loss. anarchy has no entry by design (it is the
# no-memory control).
MEMORY_TOOL_NAMES = (
    "vestige_backfill",   # sync, sync-noedge (Vestige)
    "rag_search",         # rag, rag-bus
    "mem0_search",        # mem0
    "supermemory_search",  # supermemory
    "hindsight_search",   # hindsight
    "zep_search",         # zep
    "anchor_dense_search",  # anchor-dense (ablation: event anchor on cosine stack)
)

def _classify_retrieval(output: str) -> tuple[str, str]:
    """Classify one memory-tool result as error / empty_store / no_match / ok.

    Three states, not two. An earlier version of this only checked whether the
    output started with "ERROR", which silently counted SIX different
    failure/empty strings as successful retrievals -- reintroducing the exact
    bug this telemetry exists to prevent (a dead arm looking alive). The strings
    are, in runner.py:
        "(RAG index empty -- no memories to retrieve.)"
        "(mem0 index empty -- no memories to retrieve.)"
        "(Hindsight index empty -- no memories to retrieve.)"
        "(supermemory returned no matching memories.)"
        "(Hindsight returned no matching memories.)"
        "(Graphiti knowledge graph returned no matching facts.)"

    They are NOT equivalent, and conflating them would be its own dishonesty:

    - "index empty"  -> the arm's STORE was never populated. Setup failure. The
                        memory layer cannot be said to have answered. Counted as
                        an error so the arm reports memory_unavailable.
    - "no matching"  -> the backend answered from a populated store and honestly
                        found nothing relevant for that query. The layer IS
                        alive; this is a legitimate retrieval outcome and must
                        NOT be scored as a crash. Counted OK, and separately
                        tallied as retrieval_empty_count so it stays visible.

    Returns (kind, detail) where detail is a short line for the error samples.
    """
    text = (output or "").strip()
    if not text:
        return "error", "empty tool output"
    if text.upper().startswith("ERROR"):
        return "error", text.split("\n")[0][:200]
    low = text.lower()
    if "index empty" in low or "no memories to retrieve" in low:
        return "empty_store", text.split("\n")[0][:200]
    if "no matching" in low:
        return "no_match", ""
    return "ok", ""


# Which memory tool each arm is expected to call. Drives the pre-warm step and
# the liveness gate.
ARM_MEMORY_TOOL = {
    "sync": "vestige_backfill",
    "rag": "rag_search",
    "mem0": "mem0_search",
    "supermemory": "supermemory_search",
    "hindsight": "hindsight_search",
    "zep": "zep_search",
    # Pre-registered ablation arms.
    "rag-bus": "rag_search",
    "sync-noedge": "vestige_backfill",
    "anchor-dense": "anchor_dense_search",
}

# How many agents in the fleet. 3 is the demo default (enough to diverge).
FLEET_SIZE = int(os.environ.get("FLEET_SIZE", "3"))

# Per-agent iteration cap. Kept modest so a thrashing agent terminates and is
# recorded, not left spinning. Overrides runner.MAX_ITERATIONS for the fleet.
FLEET_MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))

# --- Demo profile ------------------------------------------------------------
# The fleet supports two torture scenarios, selected by DEMO_PROFILE:
#   "canonical" (default) -> torture-v2: a canonical field-ORDER drift. The two
#                            canonical.ts files are contested; "direction" is the
#                            field order each agent lands on.
#   "keyring"             -> torture-v3.5 "Silent Rotation": a signing-KEY-id
#                            selection. The two config files are contested;
#                            "direction" is the key id each agent chooses. Adds an
#                            out-of-band PRODUCTION-CONFORMANCE replay after the
#                            integrated test: the fix is only truly correct if the
#                            integrated ledger still verifies already-issued
#                            production tokens (held in the harness, in no checkout).
DEMO_PROFILE = os.environ.get("DEMO_PROFILE", "canonical").strip().lower()

if DEMO_PROFILE == "keyring":
    CONTESTED_FILES = [
        "identity-service/src/config.ts",
        "ledger-service/src/config.ts",
    ]
else:
    CONTESTED_FILES = [
        "identity-service/src/canonical.ts",
        "ledger-service/src/canonical.ts",
    ]

# Back-compat alias: existing code refers to CANONICAL_FILES.
CANONICAL_FILES = CONTESTED_FILES

# Files that are legitimately editable to fix the bug. Any edit OUTSIDE this set
# is recorded as an out-of-bounds change (part of the anarchy signal) but not
# used to decide the merge.
FIXABLE_FILES = set(CONTESTED_FILES)

# Path to the production-conformance corpus (keyring profile only): already-issued
# production tokens, held in the harness, present in NO agent checkout.
PROD_CORPUS_PATH = os.environ.get(
    "PROD_CORPUS",
    str(Path(__file__).resolve().parent.parent / "prod-corpus" / "production-tokens.json"),
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --- Transcript capture helpers ---------------------------------------------


def _sanitize_for_json(obj):
    """Best-effort convert arbitrary provider output items (SDK model objects,
    Anthropic content blocks, dicts) into plain JSON-serializable structures so
    the transcript can be written verbatim. Never raises: anything it cannot
    convert becomes its repr string, so the transcript is always writable."""
    # Fast path for JSON primitives.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # Pydantic / SDK models expose model_dump(); Anthropic blocks expose to_dict.
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _sanitize_for_json(fn())
            except Exception:  # noqa: BLE001
                pass
    # Generic object with a __dict__.
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return {str(k): _sanitize_for_json(v) for k, v in d.items()}
    return repr(obj)


def _extract_reasoning(raw_items) -> str:
    """Pull any human-readable reasoning text out of the sanitized raw model
    output items. OpenAI Responses reasoning items carry a `summary` (a list of
    {type, text}); Anthropic 'thinking' blocks carry `thinking`. Returns a joined
    string, or '' when the model exposed no visible reasoning text."""
    chunks: list[str] = []
    items = raw_items if isinstance(raw_items, list) else [raw_items]
    for it in items:
        if not isinstance(it, dict):
            continue
        # OpenRouter / Kimi (chat.completions): the assistant message dict carries
        # the chain-of-thought as a top-level `reasoning` string (see
        # OpenRouterProvider.turn). No `type` field on these items.
        if it.get("role") == "assistant" and isinstance(it.get("reasoning"), str):
            if it["reasoning"].strip():
                chunks.append(it["reasoning"].strip())
            continue
        typ = it.get("type")
        if typ == "reasoning":
            summary = it.get("summary") or []
            if isinstance(summary, list):
                for s in summary:
                    if isinstance(s, dict) and s.get("text"):
                        chunks.append(str(s["text"]))
                    elif isinstance(s, str):
                        chunks.append(s)
        elif typ in ("thinking", "redacted_thinking"):
            if it.get("thinking"):
                chunks.append(str(it["thinking"]))
        # OpenAI message items may carry output_text content too.
        elif typ == "message":
            for part in it.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text") and part.get("text"):
                    chunks.append(str(part["text"]))
    return "\n\n".join(chunks).strip()


# --- The shared coordination bus (sync mode only) ---------------------------
#
# In sync mode the fleet shares ONE Vestige data-dir. The bus is a thin wrapper
# that (a) exposes vestige_backfill to every agent (already in runner.VESTIGE_TOOL)
# and (b) adds a vestige_log tool so an agent can WRITE a finding other agents
# can later read -- i.e. real agent-to-agent coordination through the hub, not
# just read-only recall. Writes are serialized under a lock because SQLite
# single-writer semantics mean concurrent ingests would otherwise contend.


class SharedBus:
    """Vestige-backed shared memory bus for the sync fleet."""

    def __init__(self, data_dir: str, vestige_bin: str) -> None:
        self.data_dir = data_dir
        self.vestige_bin = vestige_bin
        # In-process record of every finding published this run. Needed by the
        # rag-bus ablation arm: _rag_load_facts caches the corpus once per
        # process, so a finding ingested into the DB would NEVER surface in a
        # later rag_search -- the bus would be write-only, silently rigging the
        # reviewer's arm to lose. rag-bus reads coordinate through this list
        # (every retrieval appends the findings verbatim), which makes its
        # coordination at least as strong as sync's, never weaker.
        self.findings: list[str] = []
        import threading

        self._lock = threading.Lock()

    def _run(self, args: list, timeout: int = 60) -> subprocess.CompletedProcess:
        cmd = [self.vestige_bin, "--data-dir", self.data_dir, *args]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def log_finding(self, agent_id: str, content: str) -> str:
        """An agent writes a coordination note other agents can recall. Tagged
        canonical_digest so it joins the same causal thread as the seeded
        history and the failure."""
        body = f"[fleet {agent_id}] {content}"
        with self._lock:
            try:
                proc = self._run(
                    [
                        "ingest",
                        body,
                        "--node-type",
                        "note",
                        "--tags",
                        "canonical_digest,fleet_coordination",
                        "--source",
                        f"fleet-agent-{agent_id}",
                    ]
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return f"ERROR logging to shared bus: {exc}"
        if proc.returncode != 0:
            return f"vestige ingest exit={proc.returncode}\n{proc.stderr[:1000]}"
        self.findings.append(body)
        return "logged to shared memory (other agents can now recall this)."


def _bus_backfill_tool(bus: SharedBus):
    """Wrap runner's backfill against the shared bus data-dir."""

    def tool(_repo: Path, _args: dict) -> str:
        # runner.tool_vestige_backfill reads VESTIGE_BIN / VESTIGE_DATA_DIR from
        # the environment; the fleet sets those per shared run, so we just defer.
        return runner.tool_vestige_backfill(_repo, _args)

    return tool


def _bus_log_tool(bus: SharedBus, agent_id: str):
    def tool(_repo: Path, args: dict) -> str:
        content = str(args.get("finding", "")).strip()
        if not content:
            return "ERROR: empty finding"
        return bus.log_finding(agent_id, content)

    return tool


# --- Per-agent run (one member of the fleet) --------------------------------


def _build_agent_tools(mode: str, agent_id: str, bus: SharedBus | None):
    """Return (specs, tools) for one agent.

    anarchy: exactly runner.BASE_TOOLS (no memory, no coordination).
    rag    : BASE_TOOLS + rag_search (independent dense-cosine retrieval over the
             SAME seeded Vestige DB). No shared bus, no coordination -- like
             anarchy PLUS a retrieval tool. Each agent retrieves on its own.
    sync   : BASE_TOOLS + vestige_backfill (read the hub) + vestige_log (write
             to the hub so peers can coordinate).
    """
    tools = dict(runner.BASE_TOOLS)
    if mode == "rag":
        # Independent, uncoordinated retrieval: register the single-agent RAG
        # tool exactly as-is. Its fn reads VESTIGE_DATA_DIR (the seeded DB) at
        # call time, so every rag agent queries the same seeded memory by pure
        # cosine. No bus, no vestige_log -- rag agents do NOT coordinate.
        r_fn, r_desc, r_schema = runner.RAG_TOOL["rag_search"]
        tools["rag_search"] = (r_fn, r_desc, r_schema)
    elif mode == "mem0":
        # Independent, uncoordinated retrieval like rag: BASE_TOOLS + a single
        # mem0 retrieval tool. Its fn ingests the SAME shared corpus once per
        # process (_rag_load_facts) and retrieves via mem0's recommended path.
        # No bus, no vestige_log -- mem0 agents do NOT coordinate.
        m_fn, m_desc, m_schema = runner.MEM0_TOOL["mem0_search"]
        tools["mem0_search"] = (m_fn, m_desc, m_schema)
    elif mode == "supermemory":
        # Independent, uncoordinated retrieval through a self-hosted supermemory
        # server. Its fn ingests the SAME seeded corpus (_rag_load_facts) into
        # supermemory's own local store once per process, then retrieves via
        # /v3/search. No bus, no vestige_log -- supermemory agents do NOT coordinate.
        s_fn, s_desc, s_schema = runner.SUPERMEMORY_TOOL["supermemory_search"]
        tools["supermemory_search"] = (s_fn, s_desc, s_schema)
    elif mode == "hindsight":
        # Independent, uncoordinated retrieval through Hindsight's own agent-memory
        # recall path (semantic + BM25 + local cross-encoder rerank) over the SAME
        # seeded corpus (_rag_load_facts). No bus, no vestige_log -- hindsight
        # agents do NOT coordinate (anarchy PLUS retrieval).
        h_fn, h_desc, h_schema = runner.HINDSIGHT_TOOL["hindsight_search"]
        tools["hindsight_search"] = (h_fn, h_desc, h_schema)
    elif mode == "zep":
        # Independent, uncoordinated knowledge-graph retrieval over the SAME
        # seeded corpus (_rag_load_facts) via Zep's OSS Graphiti engine on a local
        # FalkorDB. No bus, no vestige_log -- zep agents do NOT coordinate
        # (anarchy PLUS a temporal-knowledge-graph memory tool).
        z_fn, z_desc, z_schema = runner.ZEP_TOOL["zep_search"]
        tools["zep_search"] = (z_fn, z_desc, z_schema)
    elif mode in ("sync", "sync-noedge"):
        # Read side: the same backfill tool the single-agent vestige mode uses.
        # sync-noedge is byte-identical at the TOOL level -- its ablation lives
        # entirely in the seed (the cause's shared entity tag is stripped, see
        # prepare_trial.py), so any difference in outcome is attributable to the
        # hand-authored causal edge and nothing else.
        b_fn, b_desc, b_schema = runner.VESTIGE_TOOL["vestige_backfill"]
        tools["vestige_backfill"] = (_bus_backfill_tool(bus), b_desc, b_schema)
        # Write side: coordination. Lets an agent publish a finding for peers.
        tools["vestige_log"] = (
            _bus_log_tool(bus, agent_id),
            "Publish a short finding to the SHARED team memory so the other "
            "agents in the fleet can recall it and avoid duplicating or "
            "colliding with your work. Use it to record the fix direction you "
            "concluded and why (e.g. which service is the source of truth).",
            {
                "type": "object",
                "properties": {"finding": {"type": "string"}},
                "required": ["finding"],
            },
        )
    elif mode == "rag-bus":
        # ABLATION (PREREGISTRATION.md arm 1): pure coordination. Same typed-
        # query cosine retrieval as rag, PLUS the same write bus as sync. The
        # read path appends every peer finding verbatim to each retrieval
        # result, because the cached corpus cannot surface post-seed ingests --
        # see SharedBus.findings. Coordination here is therefore at least as
        # visible to agents as sync's, by construction.
        r_fn, r_desc, r_schema = runner.RAG_TOOL["rag_search"]

        def _rag_bus_search(repo: Path, args: dict, _fn=r_fn, _bus=bus) -> str:
            out = _fn(repo, args)
            if _bus is not None and _bus.findings:
                notes = "\n".join(f"- {f}" for f in _bus.findings)
                out = (
                    out
                    + "\n\nTEAM FINDINGS (published by your fleet peers so far):\n"
                    + notes
                )
            return out[:8000]

        tools["rag_search"] = (_rag_bus_search, r_desc, r_schema)
        tools["vestige_log"] = (
            _bus_log_tool(bus, agent_id),
            "Publish a short finding to the SHARED team memory so the other "
            "agents in the fleet can see it in their retrieval results and "
            "avoid duplicating or colliding with your work. Use it to record "
            "the fix direction you concluded and why.",
            {
                "type": "object",
                "properties": {"finding": {"type": "string"}},
                "required": ["finding"],
            },
        )
    elif mode == "anchor-dense":
        # ABLATION (PREREGISTRATION.md arm 3): event-anchoring alone, on the
        # competitor's retrieval stack. No Vestige, no edge, no bus. The tool
        # takes no query; the failure event is the query.
        a_fn, a_desc, a_schema = runner.ANCHOR_DENSE_TOOL["anchor_dense_search"]
        tools["anchor_dense_search"] = (a_fn, a_desc, a_schema)
    specs = [
        {"name": name, "description": desc, "input_schema": schema}
        for name, (_fn, desc, schema) in tools.items()
    ]
    return specs, tools


def _agent_system_prompt(mode: str, agent_id: str) -> str:
    base = (
        "You are ONE agent in a fleet of autonomous coding agents all assigned "
        "the SAME task on a shared TypeScript monorepo (the Meridian platform: "
        "identity-service mints session tokens, gateway-service forwards them, "
        "ledger-service verifies them and charges). Your task: "
        f"{runner.TASK}. The end-to-end charge test currently fails. Reproduce it "
        "with run_tests, investigate across the services, apply a fix with "
        "write_file, then re-run run_tests to verify. Call finish(fixed=true) "
        "ONLY when run_tests exits 0. Be efficient; do not re-read files you have "
        "already read. IMPORTANT: your edits will later be INTEGRATED with the "
        "edits of the other agents into one shared codebase, so the DIRECTION of "
        "your fix matters, not just whether your local copy passes."
    )
    if mode in ("sync", "sync-noedge"):
        base += (
            " You are NOT working blind: you share a team memory with the rest of "
            "the fleet. Call vestige_backfill to reach backward from the failure "
            "to the quiet earlier change that caused it (it may be a past decision "
            "in a different service that the current code does not reveal), and "
            "call vestige_log to publish the fix DIRECTION you conclude so your "
            "peers converge on the same change instead of colliding. Consult the "
            "shared memory BEFORE deciding which file is authoritative."
        )
    elif mode == "rag-bus":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions. Call rag_search to retrieve the most "
            "relevant memories about this failure (the cause may live in a past "
            "decision in a different service that the current code does not "
            "reveal); its results also include any findings your fleet peers "
            "have published. Call vestige_log to publish the fix DIRECTION you "
            "conclude so your peers converge on the same change instead of "
            "colliding. Consult that memory BEFORE deciding which file is "
            "authoritative."
        )
    elif mode == "anchor-dense":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions. Call anchor_dense_search (it takes no "
            "arguments; the current failure itself is the query) to retrieve the "
            "memories most similar to the failure event (the cause may live in a "
            "past decision in a different service that the current code does not "
            "reveal). Consult that memory BEFORE deciding which file is "
            "authoritative."
        )
    elif mode == "rag":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions. Call rag_search to retrieve the most "
            "relevant memories about this failure (the cause may live in a past "
            "decision in a different service that the current code does not "
            "reveal). Consult that memory BEFORE deciding which file is "
            "authoritative."
        )
    elif mode == "mem0":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions stored in mem0. Call mem0_search to "
            "retrieve the most relevant memories about this failure (the cause "
            "may live in a past decision in a different service that the current "
            "code does not reveal). Consult that memory BEFORE deciding which "
            "file is authoritative."
        )
    elif mode == "supermemory":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions stored in supermemory. Call "
            "supermemory_search to retrieve the most relevant memories about this "
            "failure (the cause may live in a past decision in a different service "
            "that the current code does not reveal). Consult that memory BEFORE "
            "deciding which file is authoritative."
        )
    elif mode == "hindsight":
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions stored in Hindsight. Call hindsight_search "
            "to retrieve the most relevant memories about this failure (the cause "
            "may live in a past decision in a different service that the current "
            "code does not reveal). Consult that memory BEFORE deciding which file "
            "is authoritative."
        )
    elif mode == "zep":
        # FIX 1: without this branch, zep falls through to the anarchy else-branch
        # ("you have NO shared memory") -- a fairness bug that rigs the arm to lose.
        base += (
            " You are NOT working blind: you have a team memory of past notes, "
            "incidents, and decisions stored as a temporal knowledge graph "
            "(Zep/Graphiti). Call zep_search to retrieve the most relevant "
            "memories about this failure (the cause may live in a past decision in "
            "a different service that the current code does not reveal). Consult "
            "that memory BEFORE deciding which file is authoritative."
        )
    else:
        base += (
            " You have NO shared memory and no way to coordinate with the other "
            "agents. Decide and act on your own."
        )
    return base


def _run_one_agent(
    mode: str,
    agent_id: str,
    checkout: Path,
    bus: SharedBus | None,
    transcript_dir: Path | None = None,
) -> dict:
    """Run a single agent loop in its OWN checkout. Returns a per-agent result
    dict plus the concrete edits it made to the canonical files (for the merge).

    This mirrors runner.run() exactly (same provider, same measured usage, same
    ground-truth verification) but is scoped to one fleet member + captures the
    post-run contents of the contested files.
    """
    provider = runner._make_provider()
    specs, tools = _build_agent_tools(mode, agent_id, bus)
    transcript = provider.new_transcript(
        f"The repo is at {checkout}. Task: {runner.TASK}."
    )

    total_in = 0
    total_out = 0
    iterations = 0
    status = "looped"
    finish_summary = ""
    writes: list[str] = []  # repo-relative paths this agent wrote
    turn_error = ""  # non-empty if a model turn failed mid-run
    t0 = time.time()
    system = _agent_system_prompt(mode, agent_id)

    # Full turn-by-turn transcript capture -- this is the ON-CAMERA record of what
    # each agent actually saw and did: its reasoning + tool calls per turn, the
    # verbatim tool results (crucially the raw vestige_backfill output the sync
    # agents receive), and the token delta per turn. Written to results/ after the
    # loop. Reasoning text (OpenAI Responses reasoning items) is captured when the
    # SDK exposes it; if a reasoning summary is unavailable the reasoning items are
    # still recorded structurally so the tool-call sequence is fully auditable.
    turn_log: list[dict] = []

    for iterations in range(1, FLEET_MAX_ITERATIONS + 1):
        # A model turn can fail (rate-limit, transient 5xx, auth). The reused
        # runner.OpenAIProvider.turn calls runner.fail() -> sys.exit(2) on such
        # errors, which raises SystemExit (a BaseException, NOT Exception). In a
        # fleet that MUST NOT kill the whole process and torch the paid run of
        # the sibling agents -- so we catch BOTH here, record this agent as
        # errored, and let the rest of the fleet drain normally. The tokens this
        # agent already spent are preserved in total_in/total_out.
        transcript_len_before = len(transcript)
        try:
            calls, in_tok, out_tok = provider.turn(system, specs, transcript)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            turn_error = f"{type(exc).__name__}: {exc}"
            status = "errored"
            turn_log.append({
                "turn": iterations,
                "error": turn_error,
                "note": "model turn raised; agent recorded as errored, fleet continues",
            })
            _log(f"   agent {agent_id}: turn error ({turn_error}); recording partial + continuing fleet")
            break
        total_in += in_tok
        total_out += out_tok

        # The provider appended its raw output item(s) to `transcript` in-place
        # (for OpenAI: reasoning + message + function_call items; for Anthropic:
        # the assistant content blocks). Snapshot exactly what it added this turn
        # so the transcript file shows the model's real, unedited output.
        raw_model_output = _sanitize_for_json(transcript[transcript_len_before:])
        reasoning_text = _extract_reasoning(raw_model_output)

        turn_record = {
            "turn": iterations,
            "tokens": {"input": in_tok, "output": out_tok},
            "reasoning": reasoning_text,       # model's visible reasoning, if any
            "raw_model_output": raw_model_output,  # verbatim provider items
            "tool_calls": [
                {"name": c["name"], "input": c["input"]} for c in calls
            ],
            "tool_results": [],  # filled below with verbatim outputs
        }

        if not calls:
            provider.add_no_tool_nudge(transcript)
            turn_record["note"] = "no tool call; nudged"
            turn_log.append(turn_record)
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
                turn_record["tool_results"].append({
                    "name": "finish", "input": c["input"], "output": "acknowledged",
                })
                continue
            # Guard against a hallucinated / misspelled tool name: dispatch must
            # not KeyError out of the loop and crash the agent. Return a normal
            # tool error the model can recover from.
            entry = tools.get(c["name"])
            if entry is None:
                msg = f"ERROR: unknown tool '{c['name']}'. Available: {sorted(tools)}."
                results.append({"id": c["id"], "content": msg})
                turn_record["tool_results"].append({
                    "name": c["name"], "input": c["input"], "output": msg,
                })
                continue
            fn = entry[0]
            ok = True
            try:
                output = fn(checkout, c["input"] or {})
            except Exception as exc:  # noqa: BLE001
                output = f"ERROR running {c['name']}: {exc}"
                ok = False
            # Only record a write we actually attempted successfully (a rejected
            # path-escape / missing-parent write returns an ERROR string and must
            # not pollute files_written / out_of_bounds_writes).
            if c["name"] == "write_file" and ok and not str(output).startswith("ERROR"):
                p = str((c["input"] or {}).get("path", ""))
                if p:
                    writes.append(p)
            results.append({"id": c["id"], "content": output})
            # Record the VERBATIM tool output -- for vestige_backfill this is the
            # exact JSON flashback the agent saw (the money shot for the demo).
            turn_record["tool_results"].append({
                "name": c["name"], "input": c["input"], "output": output,
            })

        provider.add_tool_results(transcript, results)
        turn_log.append(turn_record)
        if did_finish:
            break

    wall = time.time() - t0

    # Ground-truth: did THIS agent's isolated checkout end green? (Independent of
    # whether the fleet's integrated tree will.) Catch ANY error here (not just
    # timeout): the agent's paid API turns are already done, so a flaky test
    # subprocess (OSError, etc.) must NOT escape and cost us the already-spent
    # token count -- fall through to the normal return that preserves it.
    try:
        verify = runner._run_tests(checkout)
        local_pass = verify.returncode == 0
    except Exception as exc:  # noqa: BLE001
        local_pass = False
        if not turn_error:
            turn_error = f"verify failed: {type(exc).__name__}: {exc}"

    if status == "fixed_claimed":
        status = "fixed" if local_pass else "hallucinated"
    elif local_pass:
        status = "fixed"

    # Capture the post-run contents of the contested files so the integrator can
    # do a real merge and classify the fix direction each agent chose.
    edited_canon = {}
    for rel in CANONICAL_FILES:
        fp = checkout / rel
        if fp.is_file():
            edited_canon[rel] = fp.read_text(errors="replace")

    out_of_bounds = sorted(
        {w for w in writes if w not in FIXABLE_FILES and w.strip()}
    )

    # Coordination evidence, computed from the captured transcript: did this agent
    # actually consult the shared Vestige hub? These are the on-camera proof that
    # the sync win came from memory, not luck.
    tool_names_used = [
        tr["name"] for t in turn_log for tr in t.get("tool_results", [])
    ]
    used_backfill = "vestige_backfill" in tool_names_used
    used_vestige_log = "vestige_log" in tool_names_used
    used_rag_search = "rag_search" in tool_names_used

    # --- ARM-AGNOSTIC RETRIEVAL TELEMETRY ---------------------------------
    # Previously this block tracked ONLY Vestige's tools plus rag_search, so a
    # result JSON could report agents_errored=0 for an arm whose memory backend
    # answered nothing. That is exactly how three dead competitor arms
    # (hindsight 19/19 errors, zep 22/22, mem0 14/15) reached a published
    # scoreboard looking like substantive retrieval losses: the output format
    # was structurally incapable of expressing "the backend never answered".
    # Every arm is now measured identically: which memory tool it called, and
    # how many of those calls returned REAL DATA vs an ERROR string.
    memory_tool_used = None
    retrieval_ok_count = 0
    retrieval_err_count = 0
    retrieval_empty_count = 0
    retrieval_errors: list[str] = []
    for _t in turn_log:
        for _tr in _t.get("tool_results", []):
            if _tr.get("name") not in MEMORY_TOOL_NAMES:
                continue
            memory_tool_used = _tr.get("name")
            kind, detail = _classify_retrieval(str(_tr.get("output", "")))
            if kind == "error":
                retrieval_err_count += 1
                if detail and detail not in retrieval_errors:
                    retrieval_errors.append(detail)
            elif kind == "empty_store":
                # The backend answered, but its STORE was never populated. That is
                # a setup failure, not a retrieval result -- counting it as a
                # successful retrieval is how a dead arm hides.
                retrieval_err_count += 1
                if detail and detail not in retrieval_errors:
                    retrieval_errors.append(detail)
            elif kind == "no_match":
                # The backend answered from a populated store and genuinely found
                # nothing relevant. The memory layer IS alive; this is a real (if
                # unhelpful) retrieval result and must NOT be scored as a crash.
                retrieval_ok_count += 1
                retrieval_empty_count += 1
            else:
                retrieval_ok_count += 1
    # True only when this agent's memory layer answered at least once. An arm
    # where this is False for every agent did not lose on retrieval quality --
    # it did not run, and must be reported as memory_unavailable, not a loss.
    memory_layer_alive = retrieval_ok_count > 0

    # Persist the FULL turn-by-turn transcript to results/ so we can prove on
    # camera exactly what this agent saw (esp. the verbatim vestige_backfill
    # flashback) and did. One file per agent per mode.
    if transcript_dir is not None:
        try:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            tpath = transcript_dir / f"transcript-{mode}-{agent_id}.json"
            tpath.write_text(json.dumps({
                "agent_id": agent_id,
                "mode": mode,
                "provider": runner.PROVIDER,
                "model": runner.MODEL,
                "task": runner.TASK,
                "status": status,
                "iterations": iterations,
                "tokens": {"input": total_in, "output": total_out, "total": total_in + total_out},
                "used_vestige_backfill": used_backfill,
                "used_vestige_log": used_vestige_log,
                "used_rag_search": used_rag_search,
                "memory_tool_used": memory_tool_used,
                "retrieval_ok_count": retrieval_ok_count,
                "retrieval_err_count": retrieval_err_count,
                "retrieval_empty_count": retrieval_empty_count,
                "retrieval_errors": retrieval_errors,
                "memory_layer_alive": memory_layer_alive,
                "final_contested_signatures": {
                    rel: _signature(rel, body) for rel, body in edited_canon.items()
                },
                "turns": turn_log,
            }, indent=2))
        except Exception as exc:  # noqa: BLE001  -- transcript write must never fail the run
            _log(f"   agent {agent_id}: WARNING could not write transcript: {exc}")

    return {
        "agent_id": agent_id,
        "mode": mode,
        "tokens": {"input": total_in, "output": total_out, "total": total_in + total_out},
        "iterations": iterations,
        "wall_clock_seconds": round(wall, 2),
        "status": status,
        "local_tests_pass": local_pass,
        "files_written": sorted(set(writes)),
        "out_of_bounds_writes": out_of_bounds,
        "used_vestige_backfill": used_backfill,
        "used_vestige_log": used_vestige_log,
        "used_rag_search": used_rag_search,
        "memory_tool_used": memory_tool_used,
        "retrieval_ok_count": retrieval_ok_count,
        "retrieval_err_count": retrieval_err_count,
        "retrieval_empty_count": retrieval_empty_count,
        "retrieval_errors": retrieval_errors,
        "memory_layer_alive": memory_layer_alive,
        "finish_summary": finish_summary,
        "turn_error": turn_error,  # non-empty if a model turn failed mid-run
        "_canonical_after": edited_canon,  # consumed by the integrator, stripped from JSON
    }


# --- Fix-direction classification + integration merge -----------------------


def _extract_kid(text: str, field: str) -> str | None:
    """[keyring profile] Extract the key id a config file selects for `field`
    (activeKid / trustedKid). Returns the chosen kid string, '' if explicitly
    blank (the RED base state), or None if the field can't be found.

    Models commonly 'externalize' the key as an env lookup with a literal
    fallback, e.g. `activeKid: process.env.IDENTITY_ACTIVE_KID ?? "k_yildun"`.
    When no env var is set (the test condition), the FALLBACK literal is the
    effective key. So we must read the fallback, not just a bare `field: "kid"`.
    Missing this classified a correct fix as `kid_unset` and produced a FALSE
    merge conflict."""
    import re

    # Isolate the field's value expression up to the end of the statement
    # (comma, newline, or closing brace) so we don't bleed into other fields.
    stmt = re.search(rf'{field}\s*[:=]\s*([^\n,}}]+)', text)
    if stmt is None:
        return None
    expr = stmt.group(1)

    # Collect every string literal in the value expression. For an env-fallback
    # (`process.env.X ?? "k_yildun"`) the LAST literal is the effective default;
    # for a bare `"k_yildun"` there is exactly one. An env-var NAME is not quoted,
    # so quoted literals are only real key ids / the blank "".
    lits = re.findall(r"""['"]([^'"]*)['"]""", expr)
    if not lits:
        # value present but no string literal (e.g. only an unquoted env var and
        # no fallback) -> no effective key id selected.
        return ""
    return lits[-1]


def _keyring_leak_audit(base_repo: Path, correct_kid: str) -> list[str]:
    """[keyring profile] Scan every checkout-visible file for anything that would
    let an agent DERIVE the correct key id without memory. Returns a list of leak
    descriptions (file:line: why); empty list == clean.

    Leaks we catch:
      * the correct kid appearing OUTSIDE the neutral keyring map (i.e. anywhere
        that could label it active/prod/correct);
      * the correct kid's raw key MATERIAL appearing outside keyring.ts (a
        fixture / corpus / pre-signed token that pins it);
      * a committed production-token corpus or truth-table / answer file.
    """
    import re

    keyring_rel = "shared/src/keyring.ts"
    # Recover the correct kid's hex material from the keyring so we can detect it
    # leaking elsewhere.
    correct_hex = None
    kr = base_repo / keyring_rel
    if kr.is_file():
        m = re.search(rf'{re.escape(correct_kid)}:\s*"([0-9a-fA-F]{{16,}})"', kr.read_text())
        if m:
            correct_hex = m.group(1)

    answer_file_hint = re.compile(r"(truth[-_ ]?table|answer|solution|prod[-_ ]?corpus|production[-_ ]tokens)", re.I)
    leaks: list[str] = []

    for p in base_repo.rglob("*"):
        rel = p.relative_to(base_repo)
        # Only audit files an AGENT can actually read (mirror runner's skip
        # logic). The seed script + demo-infra are unreadable from the checkout,
        # so a leak there is not agent-reachable.
        if runner._is_skipped(rel):
            continue
        if not p.is_file():
            continue
        relstr = str(rel)
        # An obviously answer-shaped file name is a leak on its own.
        if answer_file_hint.search(p.name):
            leaks.append(f"{relstr}: file name looks like an answer/corpus file")
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            # correct kid outside the neutral keyring map
            if correct_kid in line and relstr != keyring_rel:
                leaks.append(f"{relstr}:{i}: names the correct key id '{correct_kid}'")
            # correct key material outside keyring.ts
            if correct_hex and correct_hex in line and relstr != keyring_rel:
                leaks.append(f"{relstr}:{i}: contains the correct key's raw material")
    # De-dup while preserving order.
    seen = set()
    out = []
    for l in leaks:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def _signature(rel: str, text: str):
    """Profile-aware semantic value of a contested file, used for merge grouping
    and direction classification. Two edits with the SAME signature agree; a
    different signature is a divergent fix.

    - canonical profile: the claim field ORDER the file emits (a tuple).
    - keyring profile: the key id the config selects (activeKid for identity,
      trustedKid for ledger) as ('kid', <value>).
    """
    if DEMO_PROFILE == "keyring":
        field = "activeKid" if "identity" in rel else "trustedKid"
        kid = _extract_kid(text, field)
        return None if kid is None else ("kid", kid)
    order = _canonical_order(text)
    return None if order is None else ("order", *order)


def _canonical_order(text: str) -> list[str] | None:
    """Extract the claim field order the canonical.ts body emits, robust to the
    common ways a model may rewrite the file. Returns e.g. ['sub','plan','exp',
    'iat'] or None if no ordering can be recovered.

    The canonical string is `key=value\\n`-joined, so the ORDER that matters for
    the digest is the order the claim keys first appear as they are folded in.
    We look for the keys attached to a `claims.` / `claims[...]` access (the
    tuple form `["sub", claims.sub]`, a value-array `claims.sub`, template
    literals ``${claims.sub}``, single or double quotes) and, failing that, fall
    back to the order of the literal `sub=`/`plan=`/`iat=`/`exp=` label tokens.
    De-duplicates while preserving first-seen order so a key referenced twice
    doesn't distort the sequence."""
    import re

    keys = ("sub", "plan", "iat", "exp")
    # Primary: keys bound to a claims.* / claims[...] access, in source order.
    pat = re.compile(
        r'claims\s*(?:\.\s*(sub|plan|iat|exp)\b'
        r'|\[\s*[\'"](sub|plan|iat|exp)[\'"]\s*\])'
    )
    order: list[str] = []
    for m in pat.finditer(text):
        k = m.group(1) or m.group(2)
        if k and k not in order:
            order.append(k)
    if len(order) == len(keys):
        return order
    # Fallback: the `key=` label tokens the digest emits (e.g. `${k}=` templates
    # or explicit `sub=` literals).
    order2: list[str] = []
    for m in re.finditer(r'\b(sub|plan|iat|exp)\s*=', text):
        k = m.group(1)
        if k not in order2:
            order2.append(k)
    if len(order2) == len(keys):
        return order2
    # Return whatever partial primary order we found (better than nothing for
    # classification), or None if we found nothing at all.
    return order or None


def _classify_direction(pristine_id: list, pristine_ld: list, res: dict) -> str:
    """Given an agent's post-run contested files, classify which way it fixed.

    canonical profile returns: 'adapt_ledger' (correct), 'change_identity'
    (wrong), 'both', 'neither', 'fixed_unparseable'.

    keyring profile returns: 'kid:<value>' where <value> is the key id the agent
    converged BOTH services onto (e.g. 'kid:k_borealis' = correct). If the two
    services disagree on the kid, returns 'kid_split' (a divergent fix). If a
    service is still blank/unset, 'kid_unset'.
    """
    canon = res.get("_canonical_after", {})
    id_body = canon.get(CANONICAL_FILES[0], "")
    ld_body = canon.get(CANONICAL_FILES[1], "")

    if DEMO_PROFILE == "keyring":
        active = _extract_kid(id_body, "activeKid")
        trusted = _extract_kid(ld_body, "trustedKid")
        if not active or not trusted:
            return "kid_unset"
        if active != trusted:
            return "kid_split"
        return f"kid:{active}"

    id_after = _canonical_order(id_body)
    ld_after = _canonical_order(ld_body)

    # Byte-level "did this file change at all" as a backstop when the order can't
    # be parsed -- so a real but unparseable edit isn't mislabeled as a no-op.
    changed_id = (id_after is not None and id_after != pristine_id) or (
        id_after is None and bool(id_body.strip())
    )
    changed_ld = (ld_after is not None and ld_after != pristine_ld) or (
        ld_after is None and bool(ld_body.strip())
    )

    # If a file changed but we couldn't parse its new order, we can't name the
    # direction with confidence -- flag it rather than guess.
    if (changed_id and id_after is None) or (changed_ld and ld_after is None):
        return "fixed_unparseable"

    if changed_id and changed_ld:
        return "both"
    if changed_ld and not changed_id:
        return "adapt_ledger"
    if changed_id and not changed_ld:
        return "change_identity"
    return "neither"


_PROD_REPLAY_JS = r"""
// Production-conformance replay (ES module): import the INTEGRATED ledger's real
// verifyToken and check it still accepts every already-issued production token.
// Prints a JSON verdict. Any token that fails = production outage (voided live
// tokens). Runs under tsx so the .ts import resolves.
import { readFileSync } from "node:fs";
async function main() {
  const corpus = JSON.parse(readFileSync(process.argv[2], "utf8"));
  let mod;
  try {
    mod = await import(process.argv[3]); // integrated ledger verify.ts
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: "import failed: " + (e && e.message), verified: 0, total: corpus.tokens.length }));
    return;
  }
  const verifyToken = mod.verifyToken;
  let verified = 0;
  const failures = [];
  for (const t of corpus.tokens) {
    try { verifyToken(t.token); verified++; }
    catch (e) { failures.push({ sub: t.sub, error: String((e && e.message) || e) }); }
  }
  console.log(JSON.stringify({
    ok: failures.length === 0, verified, total: corpus.tokens.length, failures,
    signed_under: corpus.signed_under,
  }));
}
main();
"""


def _production_replay(integration_repo: Path, corpus_path: Path) -> dict:
    """[keyring profile] Replay the already-issued production token corpus against
    the INTEGRATED ledger's real verify.ts. The corpus lives in the harness and is
    in NO agent checkout, so no agent could have seen or fit to it. Returns the
    verdict dict. This is what separates a locally-green fix from a
    production-SAFE one."""
    if not corpus_path.is_file():
        return {"ran": False, "reason": f"no corpus at {corpus_path}"}
    verify_ts = integration_repo / "ledger-service" / "src" / "verify.ts"
    if not verify_ts.is_file():
        return {"ran": False, "reason": "integrated verify.ts missing"}
    # Write the replay driver into the integration tree and run it with tsx so the
    # TS import resolves against the integrated modules + workspace node_modules.
    driver = integration_repo / "_prod_replay.mjs"
    driver.write_text(_PROD_REPLAY_JS)
    try:
        proc = subprocess.run(
            ["npx", "tsx", str(driver), str(corpus_path), str(verify_ts)],
            cwd=str(integration_repo),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"ran": False, "reason": f"replay exec failed: {exc}"}
    out = proc.stdout.strip().splitlines()
    parsed = None
    for line in reversed(out):  # last JSON line is the verdict
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        return {"ran": False, "reason": "no JSON verdict", "stderr": proc.stderr[-500:]}
    parsed["ran"] = True
    return parsed


def _integrate(
    agent_results: list,
    integration_repo: Path,
    pristine_snapshot: Path,
    node_modules_source: Path | None = None,
) -> dict:
    """Perform a REAL integration of the fleet's edits into one shared tree and
    run the test suite on the result.

    Mechanism (honest, deterministic, no scripting of the outcome):
      * Start the integration tree from the pristine (broken) snapshot.
      * For each contested canonical file, collect the DISTINCT post-run contents
        the agents produced.
      * If all agents who touched a file agree on its content -> apply it (a
        clean merge for that file).
      * If they DISAGREE (>1 distinct version of the same file) OR the fleet
        edited BOTH files in opposite directions -> that is a real merge
        conflict; we record it and apply a git-style conflict marker so the file
        no longer compiles, exactly as a real 3-way merge would leave it.
      * Run the tests on the integrated tree. Its pass/fail is the fleet's real
        outcome.

    node_modules_source: a repo whose node_modules is linked into the integration
    tree so `npm test` can resolve the @meridian/* workspaces (the pristine
    snapshot deliberately excludes node_modules).
    """
    # Reset the integration tree to pristine broken state. Strip demo-infra here
    # too (same ignore as agent checkouts) so no seed/db/corpus rides along.
    if integration_repo.exists():
        shutil.rmtree(integration_repo)
    shutil.copytree(
        pristine_snapshot, integration_repo, symlinks=True,
        ignore=shutil.ignore_patterns(
            ".vestige-seed.sh", ".vestige-demo-db", ".vestige-demo-data",
            ".repo-snapshot", ".fixtures", "prod-corpus", "*.db",
        ),
    )

    # The snapshot has no node_modules; link the base repo's so tests can run.
    if node_modules_source is not None:
        nm = node_modules_source / "node_modules"
        dst = integration_repo / "node_modules"
        if nm.is_dir() and not dst.exists():
            try:
                dst.symlink_to(nm.resolve())
            except OSError:
                shutil.copytree(nm, dst, symlinks=True)

    # Pristine signatures of the two contested files (used by canonical-profile
    # direction classification; keyring profile classifies by chosen kid instead).
    pristine_id = _canonical_order((integration_repo / CANONICAL_FILES[0]).read_text())
    pristine_ld = _canonical_order((integration_repo / CANONICAL_FILES[1]).read_text())

    # Gather, per contested file, the edits agents made -- grouped by SEMANTICS
    # (the canonical field ORDER the body emits), not raw bytes. This is the
    # right merge key for this demo: two agents that both put the ledger into
    # [sub,plan,exp,iat] AGREE, even if their file formatting (comments,
    # whitespace, trailing newline) differs -- a byte-level merge would wrongly
    # flag that as a conflict. Only a genuinely DIFFERENT field order (or an
    # unparseable rewrite) counts as a divergent version.
    #
    # semantic_key(body): a tuple like ('order', 'sub','plan','exp','iat') when
    # parseable, else ('raw', <body>) so an unparseable edit stays its own bucket
    # (it can neither silently collapse into a parseable one nor hide divergence).
    def semantic_key(body: str, rel: str):
        sig = _signature(rel, body)
        if sig is not None:
            return sig
        return ("raw", body)

    per_file_groups: dict[str, dict[tuple, dict]] = {f: {} for f in CANONICAL_FILES}
    directions: dict[str, str] = {}
    for res in agent_results:
        directions[res["agent_id"]] = _classify_direction(pristine_id, pristine_ld, res)
        canon = res.get("_canonical_after", {})
        for rel in CANONICAL_FILES:
            body = canon.get(rel)
            if body is None:
                continue
            pristine_body = (pristine_snapshot / rel).read_text(errors="replace")
            if semantic_key(body, rel) == semantic_key(pristine_body, rel):
                continue  # no SEMANTIC change (cosmetic-only edit) -> untouched
            grp = per_file_groups[rel].setdefault(
                semantic_key(body, rel), {"agents": [], "body": body}
            )
            grp["agents"].append(res["agent_id"])

    conflicts: list[dict] = []

    # Detect cross-file divergence: some agents adapted ledger, others changed
    # identity. Those are logically incompatible fixes even if each file alone
    # has agreement -- integrating both yields a still-broken (or double-broken)
    # tree.
    dir_values = set(directions.values())
    if DEMO_PROFILE == "keyring":
        # Divergence = agents converged on DIFFERENT key ids (or split within an
        # agent). More than one distinct 'kid:*' choice across the fleet, or any
        # 'kid_split', means the fleet did not agree on the key.
        kid_choices = {d for d in dir_values if d.startswith("kid:")}
        cross_file_divergence = len(kid_choices) > 1 or "kid_split" in dir_values
    else:
        cross_file_divergence = (
            "adapt_ledger" in dir_values and "change_identity" in dir_values
        )

    for rel in CANONICAL_FILES:
        groups = per_file_groups[rel]
        if not groups:
            continue  # nobody semantically changed this file; pristine stays
        if len(groups) == 1:
            # All agents who touched this file agree on the field ORDER -> clean
            # merge. Apply one representative body (they are semantically equal).
            (grp,) = groups.values()
            (integration_repo / rel).write_text(grp["body"])
        else:
            # Genuinely different field orders for the SAME file -> real conflict.
            versions = {g["body"]: g["agents"] for g in groups.values()}
            conflicts.append(
                {
                    "file": rel,
                    "kind": "same_file_divergent_edits",
                    "distinct_orders": [
                        list(k[1:]) if k[0] == "order" else "unparseable"
                        for k in groups
                    ],
                    "agents": {i: g["agents"] for i, g in enumerate(groups.values())},
                }
            )
            (integration_repo / rel).write_text(_conflict_markers(rel, versions))

    if cross_file_divergence and not conflicts:
        # Divergent fixes that don't compose: the fleet disagreed on the value the
        # two files must share (the field order, or the key id). Applying both
        # leaves the integrated tree mismatched. Record it and mark the ledger file.
        detail = (
            "agents converged on different key ids / a service was left split; "
            "the choices are incompatible and do not compose."
            if DEMO_PROFILE == "keyring"
            else "some agents adapted the ledger, others changed identity; "
            "the two fixes are logically incompatible and do not compose."
        )
        conflicts.append(
            {
                "file": " + ".join(CONTESTED_FILES),
                "kind": "cross_file_divergent_choices",
                "detail": detail,
            }
        )
        # Force the integrated tree RED by conflict-marking the ledger config so a
        # divergent fleet cannot masquerade as green.
        ld_rel = CONTESTED_FILES[1]
        ld_versions = {
            g["body"]: g["agents"]
            for g in per_file_groups[ld_rel].values()
        }
        if len(ld_versions) >= 1:
            (integration_repo / ld_rel).write_text(_conflict_markers(ld_rel, ld_versions))

    # Run the tests on the integrated tree (ground truth for the fleet outcome).
    try:
        proc = runner._run_tests(integration_repo)
        integrated_pass = proc.returncode == 0
        test_tail = (proc.stdout + "\n" + proc.stderr)[-1200:]
    except subprocess.TimeoutExpired:
        integrated_pass = False
        test_tail = "TIMEOUT"

    result = {
        "integrated_tests_pass": integrated_pass,
        "merge_conflicts": conflicts,
        "conflict_count": len(conflicts),
        "cross_file_divergence": cross_file_divergence,
        "fix_directions": directions,
        "integrated_test_tail": test_tail,
    }

    # [keyring profile] Out-of-band PRODUCTION-CONFORMANCE replay: even a
    # locally-green integration is only production-SAFE if the integrated ledger
    # still verifies the already-issued production tokens. This corpus lives in
    # the harness, in no checkout, so no agent could have fit to it.
    if DEMO_PROFILE == "keyring":
        if integrated_pass:
            # Read PROD_CORPUS from the env at CALL time, not the import-time module
            # constant -- in an N-trial run the corpus path changes every trial, so
            # a stale import-time value would replay against the wrong trial's tokens.
            corpus_path = os.environ.get("PROD_CORPUS", PROD_CORPUS_PATH)
            replay = _production_replay(integration_repo, Path(corpus_path))
            result["production_replay"] = replay
            result["prod_replay_pass"] = bool(replay.get("ok")) if replay.get("ran") else None
        else:
            # Tree is already red; no point replaying, but record why.
            result["production_replay"] = {"ran": False, "reason": "integrated tree is red"}
            result["prod_replay_pass"] = None

    return result


def _conflict_markers(rel: str, versions: dict) -> str:
    """Produce a git-style conflict-marked file body from divergent versions, so
    the integrated tree genuinely fails to compile (a real, not simulated,
    conflict)."""
    bodies = list(versions.keys())
    head, *rest = bodies
    chunks = [f"<<<<<<< fleet-integration ({rel})\n", head]
    for i, b in enumerate(rest, 1):
        chunks.append(f"\n======= divergent-agent-edit-{i}\n")
        chunks.append(b)
    chunks.append("\n>>>>>>> end-of-conflict\n")
    return "".join(chunks)


# --- Memory-layer pre-warm + liveness ----------------------------------------


# A neutral warm-up query. Deliberately describes the FAILURE in the agent's own
# words rather than anything key-specific, so pre-warming cannot leak the answer
# or advantage any arm: it is the same probe text for every competitor, and its
# RESULT is discarded. Only "did the backend answer" is kept.
_PREWARM_QUERY = (
    "identity issuer and ledger verifier have no signing key id selected after "
    "the externalize-secrets refactor, so every charge fails verification"
)


def _prewarm_memory_layer(mode: str) -> dict:
    """Build/ingest this arm's memory store ONCE, single-threaded, before the
    fleet spawns and before any paid token is spent.

    Returns {attempted, ok, error, seconds}. Never raises: a failure here must
    be RECORDED (so the arm can be reported memory_unavailable) rather than
    crash the experiment or, worse, silently become a competitor's "loss".
    """
    tool_name = ARM_MEMORY_TOOL.get(mode)
    if tool_name is None:  # anarchy -- no memory layer by design
        return {"attempted": False, "ok": True, "error": "", "seconds": 0.0}

    registry = {
        "vestige_backfill": getattr(runner, "VESTIGE_TOOL", {}),
        "rag_search": getattr(runner, "RAG_TOOL", {}),
        "mem0_search": getattr(runner, "MEM0_TOOL", {}),
        "supermemory_search": getattr(runner, "SUPERMEMORY_TOOL", {}),
        "hindsight_search": getattr(runner, "HINDSIGHT_TOOL", {}),
        "zep_search": getattr(runner, "ZEP_TOOL", {}),
        "anchor_dense_search": getattr(runner, "ANCHOR_DENSE_TOOL", {}),
    }.get(tool_name, {})
    entry = registry.get(tool_name)
    if not entry:
        return {"attempted": True, "ok": False,
                "error": f"no tool registered for {tool_name}", "seconds": 0.0}

    fn = entry[0]
    # vestige_backfill and anchor_dense_search take no query (empty schema);
    # every other competitor tool requires one.
    call_args: dict = (
        {}
        if tool_name in ("vestige_backfill", "anchor_dense_search")
        else {"query": _PREWARM_QUERY}
    )

    _log(f"   pre-warming {mode} memory layer via {tool_name} ...")
    started = time.time()
    try:
        out = str(fn(Path("."), call_args))
    except Exception as exc:  # noqa: BLE001
        secs = round(time.time() - started, 1)
        return {"attempted": True, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": secs}
    secs = round(time.time() - started, 1)
    kind, detail = _classify_retrieval(out)
    if kind in ("error", "empty_store"):
        # "index empty" during PRE-WARM means the store did not get populated --
        # exactly the condition pre-warm exists to catch, so it must fail here
        # rather than surface later as a competitor "loss".
        return {"attempted": True, "ok": False,
                "error": (detail or out.strip().split("\n")[0])[:300], "seconds": secs}
    _log(f"   pre-warm OK ({mode}/{tool_name}, {secs}s, {len(out)} chars returned)")
    return {"attempted": True, "ok": True, "error": "", "seconds": secs}


# --- Fleet driver ------------------------------------------------------------


def run_fleet(mode: str, base_repo: Path, out_path: Path) -> None:
    if mode not in FLEET_MODES:
        runner.fail(f"unknown fleet mode '{mode}' (use {'|'.join(FLEET_MODES)})")

    pristine = base_repo / ".repo-snapshot"
    if not pristine.is_dir():
        runner.fail(
            f"no pristine snapshot at {pristine}. Run reset-repo.sh once to "
            "capture the broken state before a fleet run."
        )

    # Sanity: the base repo must ship RED, and RED for the RIGHT reason -- the
    # intended torture bug, not an unrelated failure (missing deps, a compile
    # error). Any of these off => refuse, so we never spend paid tokens on a repo
    # whose RED is not the torture bug.
    if DEMO_PROFILE == "keyring":
        # keyring: both services must have a BLANK key id (the unset-by-refactor
        # base state). If either is already set, the bug isn't present.
        active = _extract_kid((base_repo / CONTESTED_FILES[0]).read_text(), "activeKid")
        trusted = _extract_kid((base_repo / CONTESTED_FILES[1]).read_text(), "trustedKid")
        if active is None or trusted is None:
            runner.fail(
                "could not parse activeKid/trustedKid from the base repo config "
                "files -- the keyring torture repo is not in the expected shape."
            )
        if active or trusted:
            runner.fail(
                f"base repo key ids are already set (activeKid={active!r}, "
                f"trustedKid={trusted!r}) -- the unset-key torture bug is not "
                "present. Restore the blank base state before a fleet run."
            )
        # LEAK AUDIT (the safeguard that would have caught the .fixtures leak):
        # the correct key id MUST NOT be derivable from anything an agent can
        # read. Scan every checkout-visible file for the correct kid or its key
        # material appearing anywhere OTHER than the neutral keyring map, and for
        # the production corpus. Any hit => refuse the paid run.
        correct_kid = os.environ.get("CORRECT_KID", "k_borealis")
        leaks = _keyring_leak_audit(base_repo, correct_kid)
        if leaks:
            runner.fail(
                "LEAK AUDIT FAILED -- the correct answer is derivable from the "
                "repo, so an agent can 'solve' it without memory and there is no "
                "contrast. Remove these leaks before any paid run:\n  - "
                + "\n  - ".join(leaks)
            )
    else:
        base_id_order = _canonical_order((base_repo / CONTESTED_FILES[0]).read_text())
        base_ld_order = _canonical_order((base_repo / CONTESTED_FILES[1]).read_text())
        if base_id_order is None or base_ld_order is None:
            runner.fail(
                "could not parse the canonical field order from the base repo's "
                f"{CONTESTED_FILES[0]} / {CONTESTED_FILES[1]} -- the torture repo "
                "is not in the expected shape."
            )
        if base_id_order == base_ld_order:
            runner.fail(
                "base repo canonical orders MATCH "
                f"(identity={base_id_order}, ledger={base_ld_order}) -- the "
                "canonical-digest drift is not present, so this is not the torture "
                "bug. Restore the drift (reset-repo.sh) before a fleet run."
            )
    try:
        pre = runner._run_tests(base_repo)
        if pre.returncode == 0:
            runner.fail(
                "base repo tests PASS before the run -- the torture bug is not "
                "present (repo is green). A fleet run against a green repo is "
                "meaningless. Restore the torture bug first."
            )
        blob = (pre.stdout + "\n" + pre.stderr).lower()
        drift_signals = ("500", "signature", "err_assertion", "no active", "no trusted", "key")
        looks_like_drift = any(s in blob for s in drift_signals)
        looks_like_tooling = (
            "command not found" in blob
            or "cannot find module" in blob
            or "code 127" in blob
            or "tsx: not found" in blob
        )
        if looks_like_tooling or not looks_like_drift:
            runner.fail(
                "base repo is RED, but the failure does not look like the intended "
                "torture bug (no expected signal, or a tooling/deps error was "
                "detected). Refusing to spend paid tokens on the wrong RED. Test "
                "output tail:\n" + (pre.stdout + "\n" + pre.stderr)[-600:]
            )
    except subprocess.TimeoutExpired:
        runner.fail("base repo test run timed out during the pre-flight check.")

    bus: SharedBus | None = None
    if mode in ("sync", "sync-noedge", "rag-bus"):
        data_dir = os.environ.get("VESTIGE_DATA_DIR", "")
        vbin = os.environ.get("VESTIGE_BIN", "vestige")
        if not data_dir:
            runner.fail(f"{mode} mode needs VESTIGE_DATA_DIR (the shared bus DB).")
        bus = SharedBus(data_dir, vbin)

    work_root = Path(tempfile.mkdtemp(prefix=f"fleet-{mode}-"))
    checkouts: list[Path] = []
    # DEFENSE IN DEPTH: physically strip demo-infrastructure from every agent
    # checkout so the answer never sits on disk in the agent's directory, even
    # if a future tool gap bypassed runner's read guards. The seed script names
    # the correct key; the snapshot/demo-db are harness internals. The agent's
    # answer must come ONLY from the vestige_backfill tool (sync) or nowhere
    # (anarchy) -- never from a file it could read/grep/cat.
    _CHECKOUT_IGNORE = shutil.ignore_patterns(
        ".vestige-seed.sh", ".vestige-seed-noedge.sh", ".vestige-demo-db",
        ".vestige-demo-data", ".repo-snapshot", ".fixtures", "prod-corpus", "*.db",
    )
    for i in range(FLEET_SIZE):
        co = work_root / f"agent-{i}"
        shutil.copytree(pristine, co, symlinks=True, ignore=_CHECKOUT_IGNORE)
        # The checkout must be able to run tests: link node_modules from the base
        # repo so we don't npm install N times.
        base_nm = base_repo / "node_modules"
        if base_nm.is_dir():
            try:
                (co / "node_modules").symlink_to(base_nm.resolve())
            except OSError:
                # Fallback: some workspaces need real dirs; copy if symlink fails.
                shutil.copytree(base_nm, co / "node_modules", symlinks=True)
        checkouts.append(co)

    # Transcripts land next to the result JSON (results/), one file per agent, so
    # the on-camera proof of what each agent saw + did lives alongside the numbers.
    transcript_dir = out_path.parent

    _log(f"== FLEET {mode.upper()} ==  size={FLEET_SIZE}  provider={runner.PROVIDER}  model={runner.MODEL}")
    _log(f"   work_root={work_root}")
    _log(f"   transcripts -> {transcript_dir}/transcript-{mode}-a*.json")

    # --- PRE-WARM THE MEMORY LAYER (before any paid turn) -----------------
    # Vestige's arm has always been seeded BEFORE the loop (run-sync.sh runs
    # .vestige-seed.sh in the shell and exits loudly on failure), while every
    # competitor built its store INSIDE the agent's first tool call. That single
    # asymmetry produced three separate defects:
    #   1. TIMEOUTS -- mem0's ingest measured 563.6s ($0 liveness probe,
    #      Jul 20 2026); running that inside a bounded tool call returns an
    #      ERROR string that the scoreboard then counts as a retrieval loss.
    #      Pre-warmed, the same mem0 query answers in 0.16s.
    #   2. AN INGEST RACE -- FLEET_SIZE threads each hit an unlocked
    #      check-then-set guard and ingested the same corpus concurrently.
    #   3. A BUDGET ASYMMETRY -- competitors burned turns on setup that Vestige
    #      never paid for.
    # Warming single-threaded here fixes all three, and makes the arm's health
    # observable BEFORE a single token is spent.
    prewarm = _prewarm_memory_layer(mode)
    if prewarm.get("attempted") and not prewarm.get("ok"):
        _log(f"   !! PRE-WARM FAILED for {mode}: {prewarm.get('error')}")
        _log("      This arm's memory layer is NOT answering. Continuing so the")
        _log("      run records it, but the result will be marked")
        _log("      memory_unavailable rather than scored as a retrieval loss.")

    t0 = time.time()
    agent_results: list[dict] = []
    # Agents run concurrently -- that is the point of a fleet. In sync mode the
    # shared-bus WRITES serialize under the bus lock; READS (backfill) are fine
    # concurrently.
    with ThreadPoolExecutor(max_workers=FLEET_SIZE) as pool:
        futs = {
            pool.submit(_run_one_agent, mode, f"a{i}", checkouts[i], bus, transcript_dir): i
            for i in range(FLEET_SIZE)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                res = fut.result()
            except (Exception, SystemExit) as exc:  # noqa: BLE001
                # _run_one_agent already catches turn/tool errors internally and
                # returns a partial result WITH the tokens it spent, so this path
                # is only for a truly unexpected crash. We cannot recover the
                # spent-token count here (it died before returning) -- flag that
                # honestly rather than silently reporting 0 as if nothing was
                # billed.
                res = {
                    "agent_id": f"a{i}",
                    "mode": mode,
                    "status": "errored",
                    "error": f"{type(exc).__name__}: {exc}",
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "tokens_uncounted": True,  # real spend may be > 0; unrecoverable
                    "iterations": 0,
                    "wall_clock_seconds": 0.0,
                    "_canonical_after": {},
                }
            agent_results.append(res)
            _log(
                f"   agent {res['agent_id']}: status={res.get('status')} "
                f"tokens={res.get('tokens', {}).get('total')} "
                f"iters={res.get('iterations')} local_pass={res.get('local_tests_pass')}"
            )

    fleet_wall = time.time() - t0
    agent_results.sort(key=lambda r: r["agent_id"])

    # Integrate the fleet's edits into one shared tree and get the real outcome.
    integration_repo = work_root / "_integration"
    integ = _integrate(agent_results, integration_repo, pristine, node_modules_source=base_repo)

    # Strip the heavy internal field before serializing.
    for r in agent_results:
        r.pop("_canonical_after", None)

    total_in = sum(r["tokens"]["input"] for r in agent_results)
    total_out = sum(r["tokens"]["output"] for r in agent_results)
    total_iters = sum(r.get("iterations", 0) for r in agent_results)
    cost = (total_in / 1_000_000) * runner.COST_PER_MTOK_INPUT + (
        total_out / 1_000_000
    ) * runner.COST_PER_MTOK_OUTPUT

    n_fixed_local = sum(1 for r in agent_results if r.get("local_tests_pass"))
    directions = integ["fix_directions"]
    n_errored = sum(1 for r in agent_results if r.get("status") == "errored")
    any_uncounted = any(r.get("tokens_uncounted") for r in agent_results)
    n_used_backfill = sum(1 for r in agent_results if r.get("used_vestige_backfill"))
    n_used_log = sum(1 for r in agent_results if r.get("used_vestige_log"))

    # Arm-agnostic memory-layer health, aggregated across the fleet. The arm is
    # only considered ALIVE if at least one agent's memory tool returned real
    # data; an arm where every call errored is memory_unavailable, not a loss.
    n_mem_alive = sum(1 for r in agent_results if r.get("memory_layer_alive"))
    retrieval_ok_total = sum(int(r.get("retrieval_ok_count") or 0) for r in agent_results)
    retrieval_err_total = sum(int(r.get("retrieval_err_count") or 0) for r in agent_results)
    retrieval_empty_total = sum(int(r.get("retrieval_empty_count") or 0) for r in agent_results)
    retrieval_error_samples: list[str] = []
    for r in agent_results:
        for msg in (r.get("retrieval_errors") or []):
            if msg not in retrieval_error_samples:
                retrieval_error_samples.append(msg)
    memory_layer_alive_fleet = n_mem_alive > 0
    if ARM_MEMORY_TOOL.get(mode) is not None and not memory_layer_alive_fleet:
        _log(f"   !! MEMORY LAYER DEAD for arm '{mode}': "
             f"{retrieval_err_total} retrieval calls, ALL errored, 0 succeeded.")
        for msg in retrieval_error_samples[:3]:
            _log(f"      {msg}")
        _log("      -> this arm's result is memory_unavailable, NOT a retrieval loss.")
    ip = integ["integrated_tests_pass"]

    if DEMO_PROFILE == "keyring":
        # Correct = converged on the true production key id; wrong = any other
        # committed kid; the rest (split/unset) are neither.
        correct_kid = os.environ.get("CORRECT_KID", "k_borealis")
        n_correct_dir = sum(1 for d in directions.values() if d == f"kid:{correct_kid}")
        n_wrong_dir = sum(
            1 for d in directions.values()
            if d.startswith("kid:") and d != f"kid:{correct_kid}"
        )
        prod_ok = integ.get("prod_replay_pass")
        # Verdict combines integration-green AND production-conformance. A
        # locally-green build that voids production tokens is the money-shot
        # failure the whole demo is about.
        if ip and integ["conflict_count"] == 0 and prod_ok is True and n_wrong_dir == 0:
            fleet_verdict = "fixed_correctly"
        elif ip and prod_ok is False:
            fleet_verdict = "green_but_voids_prod"  # passing build, dead production
        elif ip and n_wrong_dir > 0:
            fleet_verdict = "green_but_voids_prod"
        elif not ip and integ["conflict_count"] > 0:
            fleet_verdict = "failed_merge_conflict"
        elif not ip:
            fleet_verdict = "failed_still_red"
        else:
            fleet_verdict = "fixed_correctly"
    else:
        # canonical profile (torture-v2)
        n_correct_dir = sum(1 for d in directions.values() if d == "adapt_ledger")
        n_wrong_dir = sum(1 for d in directions.values() if d == "change_identity")
        # The honest fleet verdict COMBINES the integration outcome with the fix
        # DIRECTION -- npm test alone cannot see that a unanimously wrong-direction
        # fix is wrong (green while invalidating already-minted production tokens).
        if ip and n_wrong_dir == 0 and integ["conflict_count"] == 0:
            fleet_verdict = "fixed_correctly"
        elif ip and n_wrong_dir > 0:
            fleet_verdict = "green_but_wrong_direction"
        elif not ip and integ["conflict_count"] > 0:
            fleet_verdict = "failed_merge_conflict"
        elif not ip:
            fleet_verdict = "failed_still_red"
        else:
            fleet_verdict = "fixed_correctly"

    result = {
        "mode": mode,
        "fleet_size": FLEET_SIZE,
        "provider": runner.PROVIDER,
        "model": runner.MODEL,
        "task": runner.TASK,
        "repo": str(base_repo),
        "measured": True,
        "fleet_tokens": {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        },
        "fleet_cost_usd": round(cost, 6),
        "cost_rate": {
            "input_per_mtok": runner.COST_PER_MTOK_INPUT,
            "output_per_mtok": runner.COST_PER_MTOK_OUTPUT,
        },
        "fleet_wall_clock_seconds": round(fleet_wall, 2),
        "fleet_iterations_total": total_iters,
        "agents_local_pass": n_fixed_local,
        "agents_errored": n_errored,
        "tokens_uncounted": any_uncounted,  # true if any errored agent's spend is unrecoverable
        # Coordination proof: how many agents consulted the shared Vestige hub.
        # In sync mode this should be the whole fleet; in anarchy it is always 0
        # (the tools aren't even offered), which is the point.
        "agents_used_vestige_backfill": n_used_backfill,
        "agents_used_vestige_log": n_used_log,
        # --- ARM-AGNOSTIC MEMORY-LAYER HEALTH (fairness-critical) ----------
        # These fields exist so this arm's number can never again be published
        # without disclosing whether its memory backend actually answered.
        # memory_layer_alive=False means the arm did NOT lose on retrieval
        # quality -- it did not run. Report it as memory_unavailable / N/A,
        # never aggregate it as a substantive defeat.
        "memory_tool": ARM_MEMORY_TOOL.get(mode),
        "memory_prewarm": prewarm,
        "agents_memory_alive": n_mem_alive,
        "retrieval_ok_total": retrieval_ok_total,
        "retrieval_err_total": retrieval_err_total,
        # Backend answered but found nothing relevant. Distinct from an error --
        # this is a real retrieval outcome, not a dead arm.
        "retrieval_empty_total": retrieval_empty_total,
        "retrieval_error_samples": retrieval_error_samples[:5],
        "memory_layer_alive": memory_layer_alive_fleet,
        "memory_unavailable": (
            ARM_MEMORY_TOOL.get(mode) is not None and not memory_layer_alive_fleet
        ),
        "transcripts": [
            f"transcript-{mode}-{r['agent_id']}.json" for r in agent_results
        ],
        "demo_profile": DEMO_PROFILE,
        "fix_direction_counts": {
            "correct": n_correct_dir,
            "wrong": n_wrong_dir,
            "other": FLEET_SIZE - n_correct_dir - n_wrong_dir,
        },
        # THE honest headline: integration outcome AND correctness (direction for
        # canonical; production-conformance for keyring).
        "fleet_verdict": fleet_verdict,
        # did the fleet's INTEGRATED work make npm test pass? (necessary, not
        # sufficient -- read alongside fleet_verdict + fix_direction_counts.)
        "integrated_tests_pass": integ["integrated_tests_pass"],
        # [keyring] out-of-band production-conformance replay result.
        "prod_replay_pass": integ.get("prod_replay_pass"),
        "production_replay": integ.get("production_replay"),
        "merge_conflict_count": integ["conflict_count"],
        "merge_conflicts": integ["merge_conflicts"],
        "cross_file_divergence": integ["cross_file_divergence"],
        "fix_directions": directions,
        "reasoning_effort": (
            os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")
            if runner.PROVIDER == "openai"
            else None
        ),
        "truncated": bool(getattr(runner.OpenAIProvider, "saw_truncation", False)),
        "agents": agent_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # Leave the work_root for inspection but note it; the caller may clean it.
    _log(f"   (fleet work tree left at {work_root} for inspection)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(FLEET_MODES))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        runner.fail(f"repo not found: {repo}")
    run_fleet(args.mode, repo, Path(args.out).resolve())


if __name__ == "__main__":
    main()
